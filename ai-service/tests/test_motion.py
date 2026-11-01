import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from app.media.preprocessing import MediaDecodeError
from app.motion.detection import (
    _extract_candidates, _score_pairs, detect_motion, FPS, METHOD,
)


def test_no_motion_produces_no_candidates():
    assert _extract_candidates([0.0] * 100)[0] == []
    assert _score_pairs(bytes([0] * 12), frame_bytes=4) == [0.0, 0.0]


def test_pulse_detected_but_never_claimed_as_a_strike():
    signals = [0.004] * 20 + [0.09, 0.14, 0.08] + [0.004] * 20
    candidates, discarded = _extract_candidates(signals)
    assert len(candidates) == 1
    assert discarded == 0
    assert candidates[0]["candidateType"] == "MOTION_ACTIVITY_CANDIDATE"
    assert candidates[0]["verifiedCombatEvent"] is False
    assert candidates[0]["motionScore"] == pytest.approx(0.14)


def test_large_cut_is_excluded():
    candidates, excluded = _extract_candidates([0.002] * 10 + [0.7] + [0.002] * 10)
    assert excluded == 1
    assert candidates == []


def test_raw_proxy_rejects_incomplete_frame():
    with pytest.raises(MediaDecodeError):
        _score_pairs(b"hello", frame_bytes=4)


def test_duration_limit_is_enforced(tmp_path):
    with pytest.raises(ValueError):
        detect_motion(video_path=tmp_path / "x.mp4", video_id=uuid4(), job_id=uuid4(),
                      job_directory=tmp_path, duration_seconds=10, max_seconds=121)


def test_real_ffmpeg_motion_artifact(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not installed")
    # Moving test bars are enough to verify decoded temporal samples.
    movie = tmp_path / "moving.mp4"
    subprocess.run([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=12", "-t", "3",
        "-c:v", "mpeg4", str(movie),
    ], check=True, capture_output=True, timeout=30)
    job_id, video_id = uuid4(), uuid4()
    job = tmp_path / "job"
    job.mkdir()
    result = detect_motion(video_path=movie, video_id=video_id, job_id=job_id,
                           job_directory=job, duration_seconds=3)
    doc = json.loads(Path(result.path).read_text(encoding="utf-8"))
    assert doc["method"] == METHOD
    assert doc["sampledFrameCount"] >= 2
    assert doc["scoredFramePairs"] >= 1
    assert isinstance(doc["motionCandidates"], list)
    assert doc["analysisType"] == "HEURISTIC_MOTION_CANDIDATES_ONLY"
    assert not doc["truncatedByDurationLimit"]

# Milestone 4 refinement: synthetic, deterministic signal-level coverage.
from app.motion.detection import _pair_metrics, _select_candidates


def test_scene_cut_below_old_absolute_cut_threshold_is_suppressed():
    scores = [0.01] * 12 + [0.22] + [0.01] * 12
    histogram = [0.01] * 12 + [0.18] + [0.01] * 12
    result = _select_candidates(scores, histogram)
    assert result.scene_cut_count == 1
    assert result.suppressed_neighbor_count == 2
    assert result.candidates == []


def test_sustained_movement_is_preserved_despite_histogram_changes():
    scores = [0.012] * 12 + [0.12, 0.19, 0.21, 0.2, 0.16] + [0.012] * 12
    histogram = [0.01] * 12 + [0.12] * 5 + [0.01] * 12
    result = _select_candidates(scores, histogram)
    assert result.scene_cut_count == 0
    assert len(result.candidates) == 1
    assert result.candidates[0]["supportingSamples"] >= 2
    assert result.candidates[0]["verifiedCombatEvent"] is False


def test_scene_transition_and_sustained_motion_in_same_video():
    scores = [0.01] * 10 + [0.23] + [0.01] * 10 + [0.08, 0.13, 0.16, 0.11] + [0.01] * 10
    hist = [0.01] * 10 + [0.2] + [0.01] * 10 + [0.04] * 4 + [0.01] * 10
    selection = _select_candidates(scores, hist)
    assert selection.scene_cut_count == 1
    assert len(selection.candidates) == 1
    assert selection.candidates[0]["peakSeconds"] == pytest.approx(6.0)


def test_single_frame_motion_spike_is_not_an_event():
    signals = [0.001] * 10 + [0.2] + [0.001] * 10
    selection = _select_candidates(signals)
    assert selection.candidates == []
    assert selection.discarded_short_bursts == 1


def test_inconsistent_histogram_size_fails_fast():
    with pytest.raises(ValueError, match="equal lengths"):
        _select_candidates([0.1, 0.2], [0.1])


def test_pair_metrics_uses_a_valid_normalized_histogram():
    score, distances = _pair_metrics(bytes([0, 0, 0, 0, 255, 255, 255, 255]), frame_bytes=4)
    assert score == [1.0]
    assert distances == [1.0]


def test_capped_candidates_even_with_many_bursts():
    signals = ([0.01] * 5 + [0.12, 0.16, 0.13] + [0.01] * 6) * 20
    result = _select_candidates(signals)
    assert 0 < len(result.candidates) <= 8
