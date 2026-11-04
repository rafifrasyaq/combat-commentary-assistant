"""M5C.2: deterministic tests without model download, ultralytics or CUDA."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.action.pose import (
    COCO_KEYPOINT_NAMES,
    FrameSample,
    PoseError,
    _format_keypoints,
    _sample_indices,
    run_pose_observations,
)


def make_job(tmp_path: Path, count: int = 2):
    clips_dir = tmp_path / "action_clips"
    clips_dir.mkdir()
    entries = []
    for idx in range(1, count + 1):
        file = clips_dir / f"candidate_{idx:02d}.mp4"
        file.write_bytes(b"synthetic clip")
        entries.append({
            "candidateIndex": idx,
            "path": str(file),
            "startSeconds": idx * 3,
            "endSeconds": idx * 3 + 2,
            "peakSeconds": idx * 3 + 1,
        })
    info = {
        "jobId": str(uuid4()), "videoId": str(uuid4()),
        "clipCount": count, "clips": entries,
    }
    (tmp_path / "action_clips.json").write_text(json.dumps(info), encoding="utf-8")
    (tmp_path / "action_predictions.json").write_text("original M5B placeholder", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("original manifest placeholder", encoding="utf-8")
    return info


def person():
    return {
        "personDetectionScore": 0.95,
        "bboxXyxyNormalized": [0.1, 0.1, 0.9, 0.9],
        "keypoints": _format_keypoints([[0.5, 0.5]] * 17, [0.9] * 17),
    }


class FakePose:
    def __init__(self, people=2, preview=True):
        self.calls = []
        self.people = people
        self.preview = preview

    def predict(self, rgb_frame, *, preview=False):
        self.calls.append((rgb_frame, preview))
        return {
            "people": [person() for _ in range(self.people)],
            "previewJpeg": b"\xff\xd8SYNTHETIC\xff\xd9" if preview and self.preview else None,
        }


def fake_reader(path):
    return [FrameSample(0, 0.0, "frame0"), FrameSample(4, 0.3333, "frame4")]


def test_sample_indices_select_bounded_distinct_frames():
    assert _sample_indices(27, 12)[0] == 0
    assert _sample_indices(27, 12)[-1] == 26
    assert len(_sample_indices(27, 12)) == 12
    assert len(set(_sample_indices(27, 12))) == 12
    assert _sample_indices(2, 12) == [0, 1]
    assert _sample_indices(1, 12) == [0]


@pytest.mark.parametrize("count,target", [(0, 12), (5, 0), (5, 13)])
def test_invalid_sample_args(count, target):
    with pytest.raises(PoseError, match="sampling"):
        _sample_indices(count, target)


def test_keypoint_schema_and_visibility():
    coords = [[0.1, 0.2]] * 17
    scores = [0.8] * 17
    scores[10] = 0.12
    points = _format_keypoints(coords, scores)
    assert len(points) == 17
    assert tuple(points) == COCO_KEYPOINT_NAMES
    assert points["left_wrist"]["reported"] is True
    assert points["right_wrist"]["reported"] is False
    assert points["right_wrist"]["x"] is None
    assert points["right_wrist"]["y"] is None


@pytest.mark.parametrize("coords,scores", [
    ([[0.1, 0.2]] * 16, [0.9] * 16),
    ([[0.1, 0.2]] * 17, [0.9] * 16),
])
def test_invalid_keypoint_layout(coords, scores):
    with pytest.raises(PoseError, match="layout"):
        _format_keypoints(coords, scores)


def test_pose_json_and_previews_preserve_old_artifacts(tmp_path):
    original = make_job(tmp_path, count=2)
    fake = FakePose()
    result = run_pose_observations(tmp_path, estimator=fake, frame_reader=fake_reader)
    assert result.clip_count == 2
    assert result.analyzed_frames == 4
    assert result.previews == 2
    assert len(fake.calls) == 4
    assert [x[1] for x in fake.calls] == [False, True, False, True]
    doc = json.loads((tmp_path / "pose_observations.json").read_text(encoding="utf-8"))
    assert doc["jobId"] == original["jobId"]
    assert doc["videoId"] == original["videoId"]
    assert doc["combatEventsDetected"] is False
    assert doc["stage"] == "UNVERIFIED_POSE_OBSERVATIONS"
    assert doc["analyzedFrameCount"] == 4
    assert doc["annotations"][0]["frames"][1]["sourceTimeSeconds"] == 3.3333
    assert doc["annotations"][0]["framesWithTwoOrMorePeople"] == 2
    assert doc["annotations"][0]["frames"][0]["people"][0]["keypoints"]["left_wrist"]["reported"]
    assert (tmp_path / "pose_previews" / "candidate_01.jpg").read_bytes().startswith(b"\xff\xd8")
    assert (tmp_path / "pose_previews" / "candidate_02.jpg").exists()
    assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == "original manifest placeholder"
    assert (tmp_path / "action_predictions.json").read_text(encoding="utf-8") == "original M5B placeholder"
    assert not (tmp_path / "pose_observations.json.tmp").exists()


def test_no_person_and_no_preview_is_not_a_punch(tmp_path):
    make_job(tmp_path, 1)
    result = run_pose_observations(tmp_path, estimator=FakePose(people=0, preview=False), frame_reader=fake_reader)
    doc = json.loads(Path(result.path).read_text(encoding="utf-8"))
    assert result.previews == 0
    assert doc["annotations"][0]["framesWithPerson"] == 0
    assert doc["combatEventsDetected"] is False


def test_empty_clips_skip_model_and_gpu(tmp_path):
    make_job(tmp_path, 0)
    result = run_pose_observations(tmp_path, estimator=None, frame_reader=fake_reader)
    assert result.clip_count == 0
    assert result.analyzed_frames == 0


def test_invalid_clip_path_fails_before_inference(tmp_path):
    info = make_job(tmp_path, 1)
    info["clips"][0]["path"] = str(tmp_path.parent / "outside.mp4")
    (tmp_path / "action_clips.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(PoseError, match="outside expected directory"):
        run_pose_observations(tmp_path, estimator=FakePose(), frame_reader=fake_reader)
    assert not (tmp_path / "pose_observations.json").exists()


def test_invalid_prediction_fails_without_publishing_json(tmp_path):
    make_job(tmp_path, 1)

    class BrokenEstimator:
        def predict(self, rgb_frame, *, preview=False):
            return {"people": [{"keypoints": {}}], "previewJpeg": None}

    with pytest.raises(PoseError, match="keypoints"):
        run_pose_observations(tmp_path, estimator=BrokenEstimator(), frame_reader=fake_reader)
    assert not (tmp_path / "pose_observations.json").exists()


def test_excessive_sample_count_rejected(tmp_path):
    make_job(tmp_path, 1)
    too_many = lambda _: [FrameSample(i, 0.0, None) for i in range(13)]
    with pytest.raises(PoseError, match="frame count"):
        run_pose_observations(tmp_path, estimator=FakePose(), frame_reader=too_many)


def test_no_publishing_if_model_throws(tmp_path):
    make_job(tmp_path, 1)

    class BrokenEstimator:
        def predict(self, rgb_frame, *, preview=False):
            raise PoseError("deliberate failure")

    with pytest.raises(PoseError, match="deliberate failure"):
        run_pose_observations(tmp_path, estimator=BrokenEstimator(), frame_reader=fake_reader)
    assert not (tmp_path / "pose_observations.json").exists()
