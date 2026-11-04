"""M5C.3 tests: synthetic pose observations; no CUDA, video, or Ultralytics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.action.tracking import TrackingError, run_tracking


KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def person(cx: float, *, wrist_dx: float = 0.05, hidden: bool = False):
    points = {}
    for name in KEYPOINTS:
        x = cx + (wrist_dx if name.endswith("wrist") else 0)
        points[name] = {
            "x": None if hidden and name.endswith("wrist") else x,
            "y": None if hidden and name.endswith("wrist") else 0.5,
            "score": 0.1 if hidden and name.endswith("wrist") else 0.95,
            "reported": not (hidden and name.endswith("wrist")),
        }
    return {"personDetectionScore": .97, "bboxXyxyNormalized": [cx - .10, .1, cx + .10, .9], "keypoints": points}


def frame(i, t, people):
    return {"frameIndex": i, "clipTimeSeconds": t, "personCount": len(people), "people": people}


def put_doc(path: Path, clips):
    doc = {
        "schemaVersion": 1,
        "stage": "UNVERIFIED_POSE_OBSERVATIONS",
        "combatEventsDetected": False,
        "modelId": "yolo11n-pose.pt",
        "jobId": "test-job-123",
        "videoId": "test-video-456",
        "clipCount": len(clips),
        "annotations": [{"candidateIndex": i + 1, "sampledFrameCount": len(frames), "frames": frames}
                        for i, frames in enumerate(clips)],
    }
    (path / "pose_observations.json").write_text(json.dumps(doc), encoding="utf-8")
    (path / "action_predictions.json").write_text("M5B-untouched", encoding="utf-8")
    (path / "action_clips.json").write_text("M5A-untouched", encoding="utf-8")
    return doc


def load_output(path):
    return json.loads((path / "pose_tracks.json").read_text(encoding="utf-8"))


def test_two_people_matched_and_hand_motion_measured(tmp_path):
    put_doc(tmp_path, [[frame(0, 0.0, [person(.25), person(.75)]),
                        frame(5, .2, [person(.26, wrist_dx=.08), person(.74, wrist_dx=.06)])]])
    result = run_tracking(tmp_path)
    doc = load_output(tmp_path)
    assert result.clips == 1
    assert result.tracks == 2
    assert result.matches == 2
    assert result.hand_motion_samples == 4
    assert doc["combatEventsDetected"] is False
    assert doc["stableFighterIdentities"] is False
    rows = doc["annotations"][0]["frames"]
    assert rows[0]["associations"][0]["trackId"] == rows[1]["associations"][0]["trackId"]
    assert rows[0]["associations"][1]["trackId"] == rows[1]["associations"][1]["trackId"]
    assert rows[1]["associations"][0]["associationStatus"] == "MATCHED_ADJACENT"
    assert rows[1]["associations"][0]["handMotion"]["left"]["shoulderRelativeRate"] > 0
    assert (tmp_path / "action_predictions.json").read_text() == "M5B-untouched"
    assert (tmp_path / "action_clips.json").read_text() == "M5A-untouched"
    assert not (tmp_path / "pose_tracks.json.tmp").exists()


def test_track_ids_reset_between_clips(tmp_path):
    put_doc(tmp_path, [[frame(0, 0., [person(.4)])], [frame(0, 0., [person(.6)])]])
    doc = load_output(Path(run_tracking(tmp_path).path).parent)
    assert doc["clipCount"] == 2
    assert doc["totalClipLocalTracks"] == 2
    assert doc["annotations"][0]["frames"][0]["associations"][0]["trackId"] == "c01_t01"
    assert doc["annotations"][1]["frames"][0]["associations"][0]["trackId"] == "c02_t01"


def test_empty_people_does_not_invent_tracks(tmp_path):
    put_doc(tmp_path, [[frame(0, 0, []), frame(2, .2, [])]])
    result = run_tracking(tmp_path)
    assert result.tracks == result.matches == result.hand_motion_samples == 0
    assert load_output(tmp_path)["annotations"][0]["frames"][0]["associations"] == []


def test_ambiguous_overlapping_people_do_not_receive_false_identity(tmp_path):
    put_doc(tmp_path, [[frame(0, 0., [person(.4), person(.45)]),
                        frame(2, .2, [person(.425)])]])
    result = run_tracking(tmp_path)
    assert result.ambiguous_restarts == 1
    row = load_output(tmp_path)["annotations"][0]["frames"][1]["associations"][0]
    assert row["associationStatus"] == "AMBIGUOUS_NEW_TRACK"
    assert row["trackId"] == "c01_t03"
    assert row["handMotion"]["left"]["measurable"] is False


def test_occlusion_causes_new_track_not_reidentification(tmp_path):
    put_doc(tmp_path, [[frame(0, 0., [person(.45)]), frame(2, .2, []), frame(4, .4, [person(.46)])]])
    doc = load_output(Path(run_tracking(tmp_path).path).parent)
    a, _, c = doc["annotations"][0]["frames"]
    assert a["associations"][0]["trackId"] != c["associations"][0]["trackId"]
    assert doc["matchedAdjacentDetections"] == 0


def test_long_time_gap_causes_new_track(tmp_path):
    put_doc(tmp_path, [[frame(0, 0., [person(.45)]), frame(100, 2., [person(.46)])]])
    assert run_tracking(tmp_path).matches == 0


def test_low_confidence_wrist_is_not_measured(tmp_path):
    put_doc(tmp_path, [[frame(0, 0., [person(.45, hidden=True)]), frame(2, .2, [person(.46)])]])
    result = run_tracking(tmp_path)
    assert result.matches == 1
    assert result.hand_motion_samples == 0
    assert load_output(tmp_path)["annotations"][0]["frames"][1]["associations"][0]["handMotion"]["left"]["shoulderRelativeRate"] is None


def test_bad_stage_refused_without_artifact(tmp_path):
    doc = put_doc(tmp_path, [[frame(0, 0., [person(.4)])]])
    doc["stage"] = "VERIFIED_PUNCHES"
    (tmp_path / "pose_observations.json").write_text(json.dumps(doc))
    with pytest.raises(TrackingError, match="schema"):
        run_tracking(tmp_path)
    assert not (tmp_path / "pose_tracks.json").exists()


def test_too_many_people_rejected(tmp_path):
    put_doc(tmp_path, [[frame(0, 0., [person(.45)] * 5)]])
    with pytest.raises(TrackingError, match="limit"):
        run_tracking(tmp_path)


def test_invalid_bbox_rejected(tmp_path):
    doc = put_doc(tmp_path, [[frame(0, 0., [person(.45)])]])
    doc["annotations"][0]["frames"][0]["people"][0]["bboxXyxyNormalized"] = [0.5, 0.5, 0.4, 0.8]
    (tmp_path / "pose_observations.json").write_text(json.dumps(doc))
    with pytest.raises(TrackingError, match="bounding box"):
        run_tracking(tmp_path)


def test_output_symlink_not_overwritten(tmp_path):
    target = tmp_path.parent / "sensitive-output.txt"
    target.write_text("secret")
    put_doc(tmp_path, [[frame(0, 0., [person(.45)])]])
    link = tmp_path / "pose_tracks.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with pytest.raises(TrackingError, match="Unsafe"):
        run_tracking(tmp_path)
    assert target.read_text() == "secret"


def test_repeated_execution_is_deterministic(tmp_path):
    put_doc(tmp_path, [[frame(0, 0., [person(.45)]), frame(2, .2, [person(.47)])]])
    run_tracking(tmp_path)
    first = (tmp_path / "pose_tracks.json").read_bytes()
    run_tracking(tmp_path)
    assert (tmp_path / "pose_tracks.json").read_bytes() == first


def test_symlinked_job_directory_is_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    put_doc(actual, [[frame(0, 0., [person(.45)])]])
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with pytest.raises(TrackingError, match="unsafe"):
        run_tracking(alias)
    assert not (actual / "pose_tracks.json").exists()
