import json
import shutil
import subprocess
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.media.preprocessing import _sample_times

client = TestClient(app)


def _payload(path):
    return {"jobId": str(uuid4()), "videoId": str(uuid4()), "storagePath": str(path)}


def test_sample_times_bounded():
    assert len(_sample_times(60, 6)) == 6
    assert len(_sample_times(1, 6)) <= 2
    assert _sample_times(0, 6) == []


def test_rejects_outside_storage(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not a movie")
    monkeypatch.setenv("VIDEO_STORAGE_ROOT", str(allowed))
    response = client.post("/analyze", json=_payload(outside))
    assert response.status_code == 422


def test_rejects_corrupted_media(monkeypatch, tmp_path):
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg tools unavailable")
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a valid video")
    monkeypatch.setenv("VIDEO_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDEO_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    response = client.post("/analyze", json=_payload(fake))
    assert response.status_code == 422


def test_real_media_generates_manifest_and_frames(monkeypatch, tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    video = tmp_path / "tiny.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
        "-t", "2", "-c:v", "mpeg4", str(video),
    ], check=True, capture_output=True, timeout=25)
    monkeypatch.setenv("VIDEO_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDEO_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    response = client.post("/analyze", json=_payload(video))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine"] == "ffmpeg-motion-clips-v1"
    manifest = json.loads(__import__("pathlib").Path(body["manifestPath"]).read_text(encoding="utf-8"))
    assert manifest["metadata"]["width"] == 320
    assert manifest["metadata"]["height"] == 240
    assert manifest["combatEventsDetected"] is False
    assert manifest["schemaVersion"] == 3
    assert manifest["stage"] == "ACTION_CLIPS_PREPARED_ONLY"
    assert manifest["combatEventsDetected"] is False
    assert manifest["actionPreparation"]["modelInferenceStatus"] == "NOT_CONFIGURED"
    clip_doc = json.loads(__import__("pathlib").Path(manifest["actionPreparation"]["resultPath"]).read_text(encoding="utf-8"))
    assert clip_doc["stage"] == "ACTION_CLIPS_PREPARED_ONLY"
    assert clip_doc["modelInference"]["predictedEvents"] == []
    assert manifest["motionAnalysis"]["candidateCount"] >= 0
    motion = json.loads(__import__("pathlib").Path(manifest["motionAnalysis"]["resultPath"]).read_text(encoding="utf-8"))
    assert motion["analysisType"] == "HEURISTIC_MOTION_CANDIDATES_ONLY"
    assert isinstance(motion["motionCandidates"], list)
    assert manifest["metadata"]["hasAudio"] is False
    assert len(clip_doc["clips"]) == manifest["actionPreparation"]["clipCount"]
    for clip in clip_doc["clips"]:
        assert __import__("pathlib").Path(clip["path"]).is_file()
        assert clip["modelPredictionStatus"] == "NOT_RUN"
    assert len(manifest["frames"]) >= 1
    for frame in manifest["frames"]:
        assert __import__("pathlib").Path(frame["path"]).is_file()
