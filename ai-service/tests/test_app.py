from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_payload(file_path):
    return {"jobId": str(uuid4()), "videoId": str(uuid4()), "storagePath": str(file_path)}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_valid_video_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_STORAGE_ROOT", str(tmp_path))
    file_path = tmp_path / "sample.mp4"
    file_path.write_bytes(b"fake-mp4-fixture")
    response = client.post("/analyze", json=make_payload(file_path))
    assert response.status_code == 200
    assert response.json()["fileSizeBytes"] == len(b"fake-mp4-fixture")
    assert response.json()["engine"] == "filesystem-stub-v1"


def test_path_outside_root_rejected(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("VIDEO_STORAGE_ROOT", str(allowed))
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"private")
    response = client.post("/analyze", json=make_payload(outside))
    assert response.status_code == 422


def test_missing_file_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_STORAGE_ROOT", str(tmp_path))
    response = client.post("/analyze", json=make_payload(tmp_path / "not-there.mp4"))
    assert response.status_code == 404
