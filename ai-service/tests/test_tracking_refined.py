"""CPU-only synthetic tests for experimental M5C.4 tracking refinement."""
import copy
import json
import os
from pathlib import Path

import pytest

from app.action import tracking as baseline
from app.action import tracking_refined as subject


def person(cx, *, cy=0.58, w=0.18, h=0.44, score=0.9):
    box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    assert 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
    shoulders = {"left_shoulder": (cx - w*0.22, cy - h*0.18),
                 "right_shoulder": (cx + w*0.22, cy - h*0.18),
                 "left_elbow": (cx - w*0.30, cy - h*0.06),
                 "right_elbow": (cx + w*0.30, cy - h*0.06),
                 "left_wrist": (cx - w*0.42, cy + h*0.02),
                 "right_wrist": (cx + w*0.42, cy + h*0.02)}
    return {"personDetectionScore": score, "bboxXyxyNormalized": box,
            "keypoints": {k: {"reported": True, "score": score, "x": x, "y": y}
                          for k, (x,y) in shoulders.items()}}


def make_doc(frame_people_by_clip):
    annotations = []
    for cid, frames in enumerate(frame_people_by_clip, 1):
        annotations.append({"candidateIndex": cid, "sampledFrameCount": len(frames),
                            "frames": [{"frameIndex": i, "clipTimeSeconds": round(i*0.2, 6),
                                        "personCount": len(people), "people": people}
                                       for i, people in enumerate(frames)]})
    return {"stage": "UNVERIFIED_POSE_OBSERVATIONS", "schemaVersion": 1,
            "jobId": "synthetic-test", "videoId": "synthetic-video",
            "modelId": "fixture", "clipCount": len(annotations),
            "combatEventsDetected": False, "annotations": annotations}


def run(tmp_path, frame_people_by_clip, with_baseline=True):
    doc = make_doc(frame_people_by_clip)
    (tmp_path / "pose_observations.json").write_text(json.dumps(doc), encoding="utf-8")
    if with_baseline:
        baseline.run_tracking(tmp_path)
    output = subject.run_refined_tracking(tmp_path)
    new = json.loads((tmp_path / subject.OUTPUT_NAME).read_text(encoding="utf-8"))
    comparison = json.loads((tmp_path / subject.COMPARISON_NAME).read_text(encoding="utf-8"))
    return output, new, comparison


def test_two_people_continuous_and_unique(tmp_path):
    frames = [[person(0.25 + d), person(0.75 - d)] for d in (0, .02, .03, .04)]
    result, doc, comparison = run(tmp_path, [frames])
    assert result.tracks == 2 and result.matched_adjacent == 6
    assert result.gap_reconnections == 0
    assert result.one_frame_tracks == 0
    assert comparison['old']['totalClipLocalTracks'] == 2
    each_frame = doc['annotations'][0]['frames']
    assert all(len({a['trackId'] for a in f['associations']}) == 2 for f in each_frame)
    assert each_frame[0]['associations'][0]['trackId'] == each_frame[-1]['associations'][0]['trackId']
    assert doc['stableFighterIdentities'] is False and doc['punchesDetected'] is False


def test_single_missed_frame_reconnect_without_interpolated_motion(tmp_path):
    frames = [[person(.25),person(.75)], [person(.75)], [person(.27),person(.74)]]
    result, doc, _ = run(tmp_path, [frames])
    assert result.gap_reconnections == 1
    first = doc['annotations'][0]['frames'][0]['associations'][0]['trackId']
    last = doc['annotations'][0]['frames'][2]['associations'][0]
    assert last['trackId'] == first
    assert last['associationStatus'] == 'RECONNECTED_AFTER_GAP'
    assert last['missingSampledFrames'] == 1
    assert all(not x['measurable'] for x in last['handMotion'].values())


def test_ambiguous_observations_abstain_instead_of_claiming_identity(tmp_path):
    frames = [[person(.30),person(.70)], [person(.50),person(.50)]]
    result, doc, _ = run(tmp_path, [frames])
    assert result.ambiguous_restarts == 2
    old_ids = {a['trackId'] for a in doc['annotations'][0]['frames'][0]['associations']}
    second = doc['annotations'][0]['frames'][1]['associations']
    assert all(a['trackId'] not in old_ids for a in second)
    assert all(a['associationStatus'] == 'AMBIGUOUS_NEW_TRACK' for a in second)


def test_rejects_large_motion_that_might_be_camera_cut(tmp_path):
    frames = [[person(.15)], [person(.85)]]
    result, doc, _ = run(tmp_path, [frames])
    assert result.tracks == 2 and result.matched_adjacent == 0
    assert doc['annotations'][0]['frames'][1]['associations'][0]['associationStatus'] == 'NEW_TRACK'


def test_rejects_extreme_box_scale_change(tmp_path):
    frames = [[person(.5, w=.1,h=.2)], [person(.5,w=.5,h=.75)]]
    result, _, _ = run(tmp_path, [frames])
    assert result.tracks == 2


def test_allows_well_aligned_moderate_scale_change(tmp_path):
    frames = [[person(.50,w=.21,h=.32)], [person(.52,w=.40,h=.57)]]
    result, _, _ = run(tmp_path, [frames])
    assert result.tracks == 1


def test_no_reconnect_after_more_than_one_missing_frame(tmp_path):
    frames = [[person(.35)], [], [], [person(.36)]]
    result, doc, _ = run(tmp_path, [frames])
    assert result.tracks == 2 and result.gap_reconnections == 0


def test_new_and_old_outputs_are_distinct_and_input_not_changed(tmp_path):
    frames = [[person(.25),person(.75)], [person(.26),person(.74)]]
    data = make_doc([frames])
    file = tmp_path / baseline.INPUT_NAME
    file.write_text(json.dumps(data), encoding='utf-8')
    baseline.run_tracking(tmp_path)
    original = (tmp_path / baseline.OUTPUT_NAME).read_bytes()
    input_bytes = file.read_bytes()
    result = subject.run_refined_tracking(tmp_path)
    assert (tmp_path / baseline.OUTPUT_NAME).read_bytes() == original
    assert file.read_bytes() == input_bytes
    assert (tmp_path / subject.COMPARISON_MARKDOWN).exists()
    assert result.path != str(tmp_path / baseline.OUTPUT_NAME)


def test_baseline_mismatch_is_rejected(tmp_path):
    frames = [[person(.35)], [person(.38)]]
    run(tmp_path, [frames])
    altered = json.loads((tmp_path / baseline.OUTPUT_NAME).read_text(encoding='utf-8'))
    altered['jobId'] = 'DIFFERENT'
    (tmp_path / baseline.OUTPUT_NAME).write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(baseline.TrackingError, match='not comparable'):
        subject.run_refined_tracking(tmp_path)


def test_invalid_input_schema_fails(tmp_path):
    doc = make_doc([[[person(.4)]]])
    doc['combatEventsDetected'] = True
    (tmp_path / baseline.INPUT_NAME).write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(baseline.TrackingError, match='schema'):
        subject.run_refined_tracking(tmp_path)


def test_no_baseline_reports_null_and_per_clip_scope(tmp_path):
    frames = [[person(.35)], [person(.37)]]
    result, doc, comparison = run(tmp_path, [frames, frames], with_baseline=False)
    assert comparison['old'] is None
    assert result.clips == 2 and result.tracks == 2
    assert doc['annotations'][0]['frames'][0]['associations'][0]['trackId'] == 'c01_r01'
    assert doc['annotations'][1]['frames'][0]['associations'][0]['trackId'] == 'c02_r01'


@pytest.mark.skipif(os.name == 'nt', reason='symlink permissions vary on Windows')
def test_refuses_symlink_output(tmp_path):
    data = make_doc([[[person(.35)]]])
    (tmp_path / baseline.INPUT_NAME).write_text(json.dumps(data), encoding='utf-8')
    secret = tmp_path / 'safe.txt'
    secret.write_text('untouched', encoding='utf-8')
    (tmp_path / subject.OUTPUT_NAME).symlink_to(secret)
    with pytest.raises(baseline.TrackingError, match='symlink'):
        subject.run_refined_tracking(tmp_path)
    assert secret.read_text(encoding='utf-8') == 'untouched'
