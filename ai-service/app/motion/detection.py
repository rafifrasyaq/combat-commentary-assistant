"""Bounded temporal-motion candidates, with conservative scene-cut suppression.

This is a deterministic, non-ML heuristic. Its scores measure grayscale pixel
changes, not a probability of a punch or a verified combat event. Editing,
camera motion and foreground overlays may still affect the result.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import subprocess
from uuid import UUID

from app.media.preprocessing import MediaDecodeError, MediaTimeout, MediaToolMissing, _tools

PROXY_SIZE = 96
FPS = 4
MAX_SECONDS = 120
MAX_CANDIDATES = 8
BYTES_PER_FRAME = PROXY_SIZE * PROXY_SIZE
METHOD = "grayscale-motion-with-scene-suppression-v2"

# Distances are empirical heuristic thresholds, NOT model probabilities.
HARD_CUT_SCORE = 0.45
CUT_HISTOGRAM_DISTANCE = 0.10
CUT_SPIKE_RATIO = 1.30
CUT_MIN_SCORE = 0.14
MIN_ACTIVE_SAMPLES = 2
MIN_SEPARATION_SECONDS = 1.0


@dataclass(frozen=True)
class MotionResult:
    path: str
    candidates: list[dict]
    analyzed_seconds: float
    frames_scored: int
    excluded_scene_changes: int
    truncated: bool


@dataclass(frozen=True)
class CandidateSelection:
    candidates: list[dict]
    scene_cut_count: int
    suppressed_neighbor_count: int
    discarded_short_bursts: int


def _score_pairs(raw: bytes, frame_bytes: int = BYTES_PER_FRAME) -> list[float]:
    """Compatibility helper for existing M4 tests and signal diagnostics."""
    if not raw or len(raw) % frame_bytes:
        raise MediaDecodeError("Motion proxy contains incomplete video frames")
    prev = memoryview(raw)[:frame_bytes]
    scores: list[float] = []
    for offset in range(frame_bytes, len(raw), frame_bytes):
        curr = memoryview(raw)[offset:offset + frame_bytes]
        change = sum(abs(a - b) for a, b in zip(prev, curr)) / (frame_bytes * 255)
        scores.append(change)
        prev = curr
    return scores


def _histogram(frame: memoryview) -> list[int]:
    counts = [0] * 16
    for pixel in frame:
        counts[pixel >> 4] += 1
    return counts


def _pair_metrics(raw: bytes, frame_bytes: int = BYTES_PER_FRAME) -> tuple[list[float], list[float]]:
    """Return mean pixel changes and 16-bin histogram distances per frame pair."""
    if not raw or len(raw) % frame_bytes:
        raise MediaDecodeError("Motion proxy contains incomplete video frames")
    prev = memoryview(raw)[:frame_bytes]
    prev_hist = _histogram(prev)
    scores: list[float] = []
    histogram_distances: list[float] = []
    for offset in range(frame_bytes, len(raw), frame_bytes):
        curr = memoryview(raw)[offset:offset + frame_bytes]
        change = sum(abs(a - b) for a, b in zip(prev, curr)) / (frame_bytes * 255)
        curr_hist = _histogram(curr)
        hist_distance = sum(abs(a - b) for a, b in zip(prev_hist, curr_hist)) / (frame_bytes * 2)
        scores.append(change)
        histogram_distances.append(hist_distance)
        prev, prev_hist = curr, curr_hist
    return scores, histogram_distances


def _select_candidates(scores: list[float], histogram_distances: list[float] | None = None,
                       *, fps: int = FPS, max_candidates: int = MAX_CANDIDATES) -> CandidateSelection:
    if fps <= 0 or max_candidates < 0:
        raise ValueError("fps must be positive and max_candidates cannot be negative")
    if histogram_distances is not None and len(histogram_distances) != len(scores):
        raise ValueError("Histogram and motion signals must have equal lengths")
    if not scores:
        return CandidateSelection([], 0, 0, 0)

    valid = [value for value in scores if value < HARD_CUT_SCORE]
    if not valid:
        return CandidateSelection([], len(scores), 0, 0)
    baseline = median(valid)
    mad = median([abs(value - baseline) for value in valid])
    threshold = max(0.028, baseline + 2.0 * mad)

    # A scene edit frequently creates a one-frame jump in BOTH mean pixel
    # distance and global histogram distance. Requiring both, plus temporal
    # isolation, reduces the risk of rejecting sustained camera/fighter motion.
    cuts: set[int] = set()
    for index, value in enumerate(scores):
        if value >= HARD_CUT_SCORE:
            cuts.add(index)
            continue
        if histogram_distances is None or not (0 < index < len(scores) - 1):
            continue
        adjacent = max(scores[index - 1], scores[index + 1])
        if (value >= max(CUT_MIN_SCORE, threshold)
                and histogram_distances[index] >= CUT_HISTOGRAM_DISTANCE
                and value >= CUT_SPIKE_RATIO * max(adjacent, 0.00001)):
            cuts.add(index)

    # Mask one neighbor on either side of each cut. Never label a transition
    # artifact or its immediate bleed-through as a motion event.
    suppressed = set(cuts)
    for index in cuts:
        if index > 0:
            suppressed.add(index - 1)
        if index + 1 < len(scores):
            suppressed.add(index + 1)
    neighbor_count = len(suppressed - cuts)

    # Continuous movement needs supporting samples. One below-threshold sample
    # may bridge an otherwise continuous burst; two consecutive inactive samples
    # end it. A solitary spike is intentionally ignored at 4 FPS.
    active_indices = [i for i, value in enumerate(scores)
                      if value >= threshold and i not in suppressed]
    bursts: list[list[int]] = []
    current: list[int] = []
    for index in active_indices:
        if current and index - current[-1] > 2:
            bursts.append(current)
            current = []
        current.append(index)
    if current:
        bursts.append(current)
    stable = [burst for burst in bursts if len(burst) >= MIN_ACTIVE_SAMPLES]
    discarded = len(bursts) - len(stable)

    # Rank by robust burst prominence rather than its single highest spike.
    ranked: list[tuple[float, int, list[int]]] = []
    for burst in stable:
        peak_idx = max(burst, key=lambda i: scores[i])
        prominence = median([scores[i] for i in burst])
        ranked.append((prominence, peak_idx, burst))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected: list[tuple[float, int, list[int]]] = []
    for entry in ranked:
        _, peak_idx, _ = entry
        if all(abs(peak_idx - prior[1]) >= fps * MIN_SEPARATION_SECONDS for prior in selected):
            selected.append(entry)
        if len(selected) >= max_candidates:
            break
    selected.sort(key=lambda item: item[1])

    candidates: list[dict] = []
    for _, peak_idx, burst in selected:
        # scores[i] represents a change ending at approximately (i+1)/fps.
        center = round((peak_idx + 1) / fps, 3)
        start = round(max(0.0, (burst[0] + 1) / fps - 0.25), 3)
        end = round(min((len(scores) + 1) / fps, (burst[-1] + 1) / fps + 0.25), 3)
        candidates.append({
            "candidateType": "MOTION_ACTIVITY_CANDIDATE",
            "startSeconds": start,
            "peakSeconds": center,
            "endSeconds": end,
            "motionScore": round(scores[peak_idx], 5),
            "scoreMeaning": "mean_absolute_grayscale_frame_difference_normalized_0_to_1",
            "supportingSamples": len(burst),
            "verifiedCombatEvent": False,
        })
    return CandidateSelection(candidates, len(cuts), neighbor_count, discarded)


def _extract_candidates(scores: list[float], *, fps: int = FPS,
                        max_candidates: int = MAX_CANDIDATES) -> tuple[list[dict], int]:
    """Backwards-compatible scoring helper (without histogram diagnostics)."""
    result = _select_candidates(scores, fps=fps, max_candidates=max_candidates)
    return result.candidates, result.scene_cut_count


def detect_motion(*, video_path: Path, video_id: UUID, job_id: UUID,
                  job_directory: Path, duration_seconds: float,
                  max_seconds: int = MAX_SECONDS) -> MotionResult:
    """Read a bounded grayscale proxy via FFmpeg and write JSON atomically."""
    if not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError("Motion analysis duration must be between 1 and 120 seconds")
    _, ffmpeg = _tools()
    window = min(duration_seconds, max_seconds)
    if window <= 0:
        raise MediaDecodeError("Motion analysis requires a positive video duration")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-t", f"{window:.3f}",
        "-vf", f"fps={FPS},scale={PROXY_SIZE}:{PROXY_SIZE}:flags=fast_bilinear,format=gray",
        "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
    ]
    try:
        proc = subprocess.run(command, capture_output=True, check=True, timeout=90)
    except FileNotFoundError as exc:
        raise MediaToolMissing("FFmpeg executable is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaTimeout("Bounded motion sampling exceeded the time limit") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaDecodeError("Unable to decode motion sampling frames") from exc
    if len(proc.stdout) > (MAX_SECONDS * FPS + 8) * BYTES_PER_FRAME:
        raise MediaDecodeError("Video generated too many motion proxy frames")
    if not proc.stdout:
        raise MediaDecodeError("Video did not yield motion proxy frames")

    scores, histogram_distances = _pair_metrics(proc.stdout)
    selection = _select_candidates(scores, histogram_distances)
    observed_frames = len(proc.stdout) // BYTES_PER_FRAME
    analysis = {
        "schemaVersion": 2,
        "jobId": str(job_id),
        "videoId": str(video_id),
        "method": METHOD,
        "analysisType": "HEURISTIC_MOTION_CANDIDATES_ONLY",
        "analyzedDurationSeconds": round(min(window, observed_frames / FPS), 3),
        "samplingFps": FPS,
        "sampledFrameCount": observed_frames,
        "scoredFramePairs": len(scores),
        "excludedLikelySceneChanges": selection.scene_cut_count,
        "suppressedSceneCutNeighborPairs": selection.suppressed_neighbor_count,
        "discardedShortMotionBursts": selection.discarded_short_bursts,
        "truncatedByDurationLimit": duration_seconds > max_seconds,
        "motionCandidates": selection.candidates,
        "limitations": (
            "Heuristic motion is NOT evidence of punches, knockdowns, or fighter identity. "
            "Scene suppression is approximate: camera movement, edits and overlays can "
            "cause false positives and false negatives. Scores are NOT probabilities. "
            "At 4 FPS, brief strikes may be missed; timestamps are approximate."
        ),
    }
    final = job_directory / "motion_candidates.json"
    tmp = job_directory / "motion_candidates.json.tmp"
    tmp.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, final)
    return MotionResult(
        path=str(final.resolve()), candidates=selection.candidates,
        analyzed_seconds=analysis["analyzedDurationSeconds"],
        frames_scored=len(scores), excluded_scene_changes=selection.scene_cut_count,
        truncated=analysis["truncatedByDurationLimit"],
    )
