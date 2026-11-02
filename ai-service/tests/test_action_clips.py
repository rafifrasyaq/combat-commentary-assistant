import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from app.action.clips import CLIP_METHOD, MAX_CLIP_SECONDS, _clip_window, prepare_action_clips
from app.media.preprocessing import MediaDecodeError


def test_clip_window_adds_context_and_stays_in_bounds():
    start, end = _clip_window({"startSeconds": 15.5, "peakSeconds": 15.75,
                               "endSeconds": 16.25}, 55.496)
    assert (start, end) == (14.75, 17.0)


def test_clip_window_caps_long_intervals_around_peak():
    start, end = _clip_window({"startSeconds": 8, "peakSeconds": 12,
                               "endSeconds": 18}, 30)
    assert end - start == MAX_CLIP_SECONDS
    assert start <= 12 <= end


@pytest.mark.parametrize("candidate", [
    {"startSeconds": -1, "peakSeconds": 0, "endSeconds": 1},
    {"startSeconds": 2, "peakSeconds": 1, "endSeconds": 3},
    {"startSeconds": 2, "peakSeconds": 3, "endSeconds": 99},
    {"startSeconds": float("nan"), "peakSeconds": 1, "endSeconds": 2},
    {"startSeconds": "invalid", "peakSeconds": 1, "endSeconds": 2},
])
def test_clip_window_rejects_bad_motion_data(candidate):
    with pytest.raises(MediaDecodeError):
        _clip_window(candidate, 20)


def test_too_many_candidates_rejected(tmp_path):
    with pytest.raises(MediaDecodeError, match="number"):
        prepare_action_clips(video_path=tmp_path / "fake.mp4", video_id=uuid4(),
                             job_id=uuid4(), job_directory=tmp_path,
                             motion_candidates=[{}] * 9, duration_seconds=10)


def test_empty_candidates_generate_honest_metadata(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg not installed")
    out = prepare_action_clips(video_path=tmp_path / "unused.mp4", video_id=uuid4(),
                               job_id=uuid4(), job_directory=tmp_path,
                               motion_candidates=[], duration_seconds=2)
    metadata = json.loads(Path(out.path).read_text(encoding="utf-8"))
    assert metadata["clipCount"] == 0
    assert metadata["modelInference"] == {"status": "NOT_CONFIGURED", "predictedEvents": []}


def test_real_ffmpeg_clip_extraction(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg tools unavailable")
    video = tmp_path / "moving.mp4"
    subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24", "-t", "3",
                    "-c:v", "mpeg4", str(video)], check=True, capture_output=True, timeout=25)
    result = prepare_action_clips(
        video_path=video, video_id=uuid4(), job_id=uuid4(),
        job_directory=tmp_path,
        motion_candidates=[{"startSeconds": 0.5, "peakSeconds": 1.2, "endSeconds": 1.5}],
        duration_seconds=3,
    )
    doc = json.loads(Path(result.path).read_text(encoding="utf-8"))
    assert doc["method"] == CLIP_METHOD
    assert doc["clipCount"] == 1
    assert doc["modelInference"]["status"] == "NOT_CONFIGURED"
    clip = Path(doc["clips"][0]["path"])
    assert clip.exists() and clip.stat().st_size > 100
    probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream=width,height,r_frame_rate",
                            "-of", "json", str(clip)], check=True, text=True,
                            capture_output=True, timeout=15)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["width"] <= 640
    assert stream["r_frame_rate"] == "12/1"
