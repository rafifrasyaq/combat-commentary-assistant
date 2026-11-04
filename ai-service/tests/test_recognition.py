"""M5B tests use a fake classifier: no network, GPU, or model weights needed."""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.action.recognition import (
    RecognitionError, publish_recognition_manifest, run_action_recognition,
    uniform_indices,
)


class FakeClassifier:
    def __init__(self, unknown=False):
        self.calls = []
        self.unknown = unknown

    def predict(self, path):
        self.calls.append(path)
        return {
            "activityLabel": "UNKNOWN" if self.unknown else "punching person (boxing)",
            "status": "UNKNOWN_LOW_SCORE" if self.unknown else "UNVERIFIED_ACTIVITY_LABEL",
            "topPredictions": [{"label": "punching person (boxing)", "softmaxScore": 0.21 if self.unknown else 0.55}],
            "sampledFrameCount": 16,
            "inferenceDevice": "cuda",
        }


def make_job(tmp_path, count=2):
    clips_dir = tmp_path / "action_clips"
    clips_dir.mkdir()
    entries = []
    for i in range(1, count + 1):
        clip = clips_dir / f"candidate_{i:02d}.mp4"
        clip.write_bytes(b"dummy video placeholder")
        entries.append({
            "candidateIndex": i,
            "path": str(clip),
            "startSeconds": 2 * i,
            "endSeconds": 2 * i + 2,
            "peakSeconds": 2 * i + 1,
        })
    doc = {"jobId": str(uuid4()), "videoId": str(uuid4()), "clipCount": count, "clips": entries}
    (tmp_path / "action_clips.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schemaVersion": 3, "stage": "ACTION_CLIPS_PREPARED_ONLY", "combatEventsDetected": False,
        "actionPreparation": {"clipCount": count, "modelInferenceStatus": "NOT_CONFIGURED"},
    }), encoding="utf-8")
    return entries


def test_uniform_sampling_repeats_if_short():
    assert uniform_indices(3, 5) == [0, 0, 1, 1, 2]
    assert uniform_indices(27, 16)[0] == 0
    assert uniform_indices(27, 16)[-1] == 26
    assert len(uniform_indices(27, 16)) == 16


@pytest.mark.parametrize("frame_count,samples", [(0, 16), (2, 0), (2, 65)])
def test_invalid_frame_sample_size(frame_count, samples):
    with pytest.raises(RecognitionError):
        uniform_indices(frame_count, samples)


def test_runs_all_clips_and_publishes_unverified_predictions(tmp_path):
    make_job(tmp_path)
    fake = FakeClassifier()
    result = run_action_recognition(tmp_path, classifier=fake)
    assert result.clip_count == 2
    assert result.unknown_count == 0
    assert len(fake.calls) == 2
    artifact = json.loads(Path(result.path).read_text(encoding="utf-8"))
    assert artifact["combatEventsDetected"] is False
    assert artifact["stage"] == "UNVERIFIED_ACTIVITY_PREDICTIONS"
    assert artifact["predictions"][0]["activityLabel"] == "punching person (boxing)"
    assert artifact["predictions"][1]["candidateIndex"] == 2
    assert artifact["model"]["scoreType"] == "uncalibrated_softmax"
    assert not Path(result.path + ".tmp").exists()
    publish_recognition_manifest(tmp_path / "manifest.json", result)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schemaVersion"] == 4
    assert manifest["combatEventsDetected"] is False
    assert manifest["actionRecognition"]["resultPath"] == str(Path(result.path))
    assert manifest["actionPreparation"]["modelInferenceStatus"] == "COMPLETED"


def test_unknown_does_not_look_like_verified_event(tmp_path):
    make_job(tmp_path, 1)
    result = run_action_recognition(tmp_path, classifier=FakeClassifier(unknown=True))
    doc = json.loads(Path(result.path).read_text(encoding="utf-8"))
    assert result.unknown_count == 1
    assert doc["predictions"][0]["activityLabel"] == "UNKNOWN"
    assert doc["predictions"][0]["status"] == "UNKNOWN_LOW_SCORE"


def test_zero_clips_needs_no_classifier_or_gpu(tmp_path):
    make_job(tmp_path, 0)
    fake = FakeClassifier()
    result = run_action_recognition(tmp_path, classifier=fake)
    assert result.clip_count == 0
    assert fake.calls == []


def test_rejects_path_outside_job_dir(tmp_path):
    entries = make_job(tmp_path, 1)
    outside = tmp_path.parent / "outside-video.mp4"
    outside.write_bytes(b"not real video")
    doc_file = tmp_path / "action_clips.json"
    doc = json.loads(doc_file.read_text())
    doc["clips"][0]["path"] = str(outside)
    doc_file.write_text(json.dumps(doc))
    with pytest.raises(RecognitionError, match="outside expected directory"):
        run_action_recognition(tmp_path, classifier=FakeClassifier())


def test_rejects_invalid_m5a_timestamps(tmp_path):
    make_job(tmp_path, 1)
    path = tmp_path / "action_clips.json"
    doc = json.loads(path.read_text())
    doc["clips"][0]["endSeconds"] = 999
    path.write_text(json.dumps(doc))
    with pytest.raises(RecognitionError, match="timestamps"):
        run_action_recognition(tmp_path, classifier=FakeClassifier())


def test_rejects_more_than_eight_clips(tmp_path):
    make_job(tmp_path, 9)
    with pytest.raises(RecognitionError, match="clip count"):
        run_action_recognition(tmp_path, classifier=FakeClassifier())


def test_classifier_failure_does_not_publish_partial_results(tmp_path):
    make_job(tmp_path, 1)

    class Broken:
        def predict(self, path):
            raise RuntimeError("inference failed")

    with pytest.raises(RecognitionError, match="classifier failed"):
        run_action_recognition(tmp_path, classifier=Broken())
    assert not (tmp_path / "action_predictions.json").exists()


def test_feature_flag_defaults_off(monkeypatch):
    from app.action.recognition import recognition_enabled
    monkeypatch.delenv("ACTION_RECOGNITION_ENABLED", raising=False)
    assert recognition_enabled() is False
    monkeypatch.setenv("ACTION_RECOGNITION_ENABLED", "1")
    assert recognition_enabled() is True


def test_inference_enabled_pipeline_keeps_java_contract_and_motion_artifact(monkeypatch, tmp_path):
    """Run real FFmpeg pipeline but fake ML; no GPU or network needed."""
    import shutil
    import subprocess
    from fastapi.testclient import TestClient
    from app.main import app

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg missing")
    video = tmp_path / "tiny.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=12",
                    "-t", "2", "-c:v", "mpeg4", str(video)],
                   check=True, capture_output=True, timeout=25)
    monkeypatch.setenv("VIDEO_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDEO_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ACTION_RECOGNITION_ENABLED", "true")
    fake = FakeClassifier()
    monkeypatch.setattr("app.action.recognition._get_classifier", lambda: fake)
    resp = TestClient(app).post("/analyze", json={
        "videoId": str(uuid4()), "jobId": str(uuid4()), "storagePath": str(video)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["engine"] == "ffmpeg-motion-clips-videomae-v1"
    manifest = json.loads(Path(resp.json()["manifestPath"]).read_text())
    assert manifest["stage"] == "UNVERIFIED_ACTIVITY_PREDICTIONS"
    assert manifest["combatEventsDetected"] is False
    assert Path(manifest["motionAnalysis"]["resultPath"]).is_file()
    assert Path(manifest["actionRecognition"]["resultPath"]).is_file()
    assert manifest["actionRecognition"]["clipCount"] == len(fake.calls)
