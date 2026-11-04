"""M5C.4: read-only, evidence-limited diagnostic of M5C.3 track fragmentation.

Reads *existing* M5C.2/M5C.3 artifacts and creates new reports. Does not
attempt person re-identification or infer verified fighter identities/punches.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

MAX_BYTES = 25_000_000
ROOT_FILES = ('pose_observations.json', 'pose_tracks.json')
AUDIT_JSON = 'tracking_quality_report.json'
AUDIT_MD = 'tracking_quality_report.md'
MIN_DELTA_SECONDS = 0.04
MAX_GAP_SECONDS = 1.0
MAX_CENTER_SHIFT = 0.28
MAX_AREA_RATIO = 3.0


class AuditError(RuntimeError):
    pass


def _load(job_dir: Path, name: str) -> dict:
    path = job_dir / name
    if not path.is_file() or path.is_symlink():
        raise AuditError(f'Missing or unsafe source: {name}')
    if path.stat().st_size > MAX_BYTES:
        raise AuditError(f'Source too large: {name}')
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise AuditError(f'Invalid JSON: {name}') from exc
    if not isinstance(doc, dict):
        raise AuditError(f'Unexpected JSON top-level type: {name}')
    return doc


def _bbox(p: dict) -> tuple[float, float, float, float]:
    try:
        raw = p['bboxXyxyNormalized']
        x1, y1, x2, y2 = tuple(float(i) for i in raw)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1) or not all(math.isfinite(a) for a in (x1, y1, x2, y2)):
            raise ValueError()
        return x1, y1, x2, y2
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError('Invalid observed person box') from exc


def _center_shift(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)


def _scale_ratio(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    aa = (a[2]-a[0])*(a[3]-a[1])
    bb = (b[2]-b[0])*(b[3]-b[1])
    return max(aa, bb)/min(aa, bb)


def _classify_new(prev: dict | None, curr: dict, person_idx: int, association_status: str) -> tuple[str, dict]:
    """Classify which *known tracking rule* can explain a fresh track.

    A reason is not proof the ID should have been reused.
    """
    if prev is None:
        return 'FIRST_SAMPLED_FRAME', {}
    old_people = prev['people']
    if not old_people:
        return 'NO_PERSON_IN_PREVIOUS_FRAME', {}
    delta = float(curr['clipTimeSeconds'])-float(prev['clipTimeSeconds'])
    if not MIN_DELTA_SECONDS <= delta <= MAX_GAP_SECONDS:
        return 'TIME_GAP_GATE', {'frameGapSeconds': round(delta, 4)}
    b = _bbox(curr['people'][person_idx])
    pairs = [(_center_shift(_bbox(p), b), _scale_ratio(_bbox(p), b)) for p in old_people]
    near = [(d, s) for d,s in pairs if d <= MAX_CENTER_SHIFT]
    if not near:
        return 'CENTER_SHIFT_GATE', {'closestCenterShift': round(min(d for d,_ in pairs), 4)}
    valid = [(d,s) for d,s in near if s <= MAX_AREA_RATIO]
    if not valid:
        return 'BOX_SCALE_GATE', {'minimumScaleRatioAmongNear': round(min(s for _,s in near), 4)}
    if association_status == 'AMBIGUOUS_NEW_TRACK':
        return 'AMBIGUITY_ABSTENTION', {'feasiblePreviousDetections': len(valid)}
    return 'UNPAIRED_WITH_FEASIBLE_CANDIDATE', {'feasiblePreviousDetections': len(valid)}


def _safe_positive_int(n: Any, label: str) -> int:
    if type(n) is not int or n < 0:
        raise AuditError(f'Invalid integer in {label}')
    return n


def inspect(observations: dict, tracks: dict) -> dict:
    if observations.get('stage') != 'UNVERIFIED_POSE_OBSERVATIONS':
        raise AuditError('Unexpected pose observations stage')
    if tracks.get('stage') != 'UNVERIFIED_CLIP_LOCAL_POSE_ASSOCIATIONS':
        raise AuditError('Unexpected tracking output stage')
    if observations.get('combatEventsDetected') is not False or tracks.get('combatEventsDetected') is not False or tracks.get('punchesDetected') is not False or tracks.get('stableFighterIdentities') is not False:
        raise AuditError('Unexpected verified-event/identity flags')
    if observations.get('jobId') != tracks.get('jobId'):
        raise AuditError('Mismatched job IDs')
    pcs, tcs = observations.get('annotations'), tracks.get('annotations')
    if not isinstance(pcs, list) or not isinstance(tcs, list) or len(pcs) != len(tcs) or len(pcs) > 50:
        raise AuditError('Mismatched or invalid clip arrays')
    if observations.get('clipCount') != len(pcs) or tracks.get('clipCount') != len(tcs):
        raise AuditError('Mismatched clip counts')
    summary = Counter()
    per_clip = []
    examples = []
    intervals_all = []
    motions = []
    for pose, track in zip(pcs, tcs):
        cid = pose.get('candidateIndex')
        if type(cid) is not int or cid != track.get('candidateIndex'):
            raise AuditError('Clip indices do not agree')
        pframes, tframes = pose.get('frames'), track.get('frames')
        if not isinstance(pframes,list) or not isinstance(tframes,list) or len(pframes) != len(tframes) or len(pframes)>12:
            raise AuditError('Clip frames do not agree')
        reasons = Counter()
        statuses = Counter()
        intervals = []
        lengths = track.get('trackLengthsInSampledFrames')
        if not isinstance(lengths, dict) or any(type(k) is not str or type(v) is not int or v < 1 for k,v in lengths.items()):
            raise AuditError('Invalid track lengths')
        for i, (pf, tf) in enumerate(zip(pframes, tframes)):
            if pf.get('frameIndex') != tf.get('frameIndex') or float(pf.get('clipTimeSeconds')) != float(tf.get('clipTimeSeconds')):
                raise AuditError('Frame timestamps or indices do not agree')
            people = pf.get('people')
            assocs = tf.get('associations')
            if not isinstance(people, list) or not isinstance(assocs, list) or len(people) != len(assocs) or len(people)>4:
                raise AuditError('Person observations and track associations disagree')
            if i:
                delta = float(pf['clipTimeSeconds'])-float(pframes[i-1]['clipTimeSeconds'])
                if not math.isfinite(delta) or delta < 0:
                    raise AuditError('Invalid time gap')
                intervals.append(delta)
                intervals_all.append(delta)
            summary['sampledFrames'] += 1
            summary['personObservations'] += len(people)
            summary['framesWithTwoOrMorePeople'] += (len(people) >= 2)
            for j, a in enumerate(assocs):
                if not isinstance(a, dict) or a.get('detectionIndex') != j or not isinstance(a.get('trackId'), str):
                    raise AuditError('Invalid association index or ID')
                stat = a.get('associationStatus')
                if stat not in ('MATCHED_ADJACENT','NEW_TRACK','AMBIGUOUS_NEW_TRACK'):
                    raise AuditError('Unknown association status')
                statuses[stat] += 1
                summary[stat] += 1
                if stat != 'MATCHED_ADJACENT':
                    reason, detail = _classify_new(pframes[i-1] if i else None, pf, j, stat)
                    reasons[reason] += 1
                    summary['cause_'+reason] += 1
                    if len(examples)<25 and reason != 'FIRST_SAMPLED_FRAME':
                        examples.append({'candidateIndex':cid,'frameIndex':pf['frameIndex'],
                                         'clipTimeSeconds':pf['clipTimeSeconds'],'detectionIndex':j,
                                         'reason':reason,**detail})
                else:
                    hands = a.get('handMotion',{})
                    for side in ('left','right'):
                        sample = hands.get(side,{})
                        if sample.get('measurable') is True:
                            rate = sample.get('shoulderRelativeRate')
                            if isinstance(rate, (int,float)) and not isinstance(rate,bool) and math.isfinite(rate) and rate>=0:
                                motions.append(float(rate))
        if sum(lengths.values()) != sum(statuses.values()) or len(lengths) != track.get('trackCount'):
            raise AuditError('Track counts or lengths do not agree')
        observed = sum(statuses.values())
        per_clip.append({
            'candidateIndex':cid, 'sampledFrames':len(pframes),
            'personObservations':observed,
            'framesWithTwoOrMorePeople':sum(len(x['people'])>=2 for x in pframes),
            'tracksCreated':len(lengths), 'singleFrameTracks':sum(v==1 for v in lengths.values()),
            'medianTrackLengthFrames':median(lengths.values()) if lengths else 0,
            'matchedAdjacent':statuses['MATCHED_ADJACENT'], 'newTracks':statuses['NEW_TRACK'],
            'ambiguousNewTracks':statuses['AMBIGUOUS_NEW_TRACK'],
            'trackCreationReasons':dict(sorted(reasons.items())),
            'medianSampleIntervalSeconds':round(median(intervals),3) if intervals else None,
            'maxSampleIntervalSeconds':round(max(intervals),3) if intervals else None,
        })
        summary['tracksCreated'] += len(lengths)
        summary['singleFrameTracks'] += sum(v==1 for v in lengths.values())
    if summary['tracksCreated'] != tracks.get('totalClipLocalTracks') or summary['MATCHED_ADJACENT'] != tracks.get('matchedAdjacentDetections') or summary['AMBIGUOUS_NEW_TRACK'] != tracks.get('ambiguousRestarts'):
        raise AuditError('Reported M5C.3 totals do not match detailed associations')
    return {
        'schemaVersion':1, 'stage':'M5C4_TRACKING_QUALITY_AUDIT',
        'jobId':observations.get('jobId'), 'clipCount':len(per_clip),
        'overview':{
            'sampledFrames':summary['sampledFrames'],
            'personObservations':summary['personObservations'],
            'framesWithTwoOrMorePeople':summary['framesWithTwoOrMorePeople'],
            'tracksCreated':summary['tracksCreated'],
            'singleFrameTracks':summary['singleFrameTracks'],
            'matchedAdjacent':summary['MATCHED_ADJACENT'],
            'newTracks':summary['NEW_TRACK'],
            'ambiguousNewTracks':summary['AMBIGUOUS_NEW_TRACK'],
            'medianSampleIntervalSeconds':round(median(intervals_all),3) if intervals_all else None,
            'maxSampleIntervalSeconds':round(max(intervals_all),3) if intervals_all else None,
            'medianMeasurableHandRelativeRate':round(median(motions),4) if motions else None,
        },
        'trackCreationReasons':dict(sorted((k[6:],v) for k,v in summary.items() if k.startswith('cause_'))),
        'perClip':per_clip,
        'examples':examples,
        'warnings':[
            'This diagnostic classifies tracker decisions, not ground-truth identity switches.',
            'Fewer tracks is NOT proof of better accuracy; forcing ambiguous matches may swap fighters.',
            'All associations are within sampled frames of one clip; no fighter A/B identity or verified punches.',
            'Rates are uncalibrated 2D image-relative measurements, not physical punch speed.',
        ],
    }


def render_markdown(report: dict) -> str:
    overview = report['overview']
    lines = [
        '# M5C.4 — Tracking Quality Audit', '',
        f"Job: `{report['jobId']}`", '',
        '## Summary', '',
        f"- Clips / sampled frames: **{report['clipCount']} / {overview['sampledFrames']}**",
        f"- Person observations: **{overview['personObservations']}**",
        f"- Tentative tracks: **{overview['tracksCreated']}**; one-frame tracks: **{overview['singleFrameTracks']}**",
        f"- Adjacent matches: **{overview['matchedAdjacent']}**; ambiguous abstentions: **{overview['ambiguousNewTracks']}**",
        f"- Median / maximum time between sampled frames: **{overview['medianSampleIntervalSeconds']} / {overview['maxSampleIntervalSeconds']} s**",
        '', '## Why fresh tracks were created (rule-based diagnosis)', ''
    ]
    for reason, value in sorted(report['trackCreationReasons'].items(), key=lambda kv:-kv[1]):
        lines.append(f'- `{reason}`: **{value}**')
    lines.extend(['', '## Per clip', '', '| Clip | People obs. | Tracks | One-frame | Matches | Ambiguous | Median gap (s) |',
                  '|---:|---:|---:|---:|---:|---:|---:|'])
    for c in report['perClip']:
        lines.append(f"| {c['candidateIndex']} | {c['personObservations']} | {c['tracksCreated']} | {c['singleFrameTracks']} | {c['matchedAdjacent']} | {c['ambiguousNewTracks']} | {c['medianSampleIntervalSeconds']} |")
    lines.extend(['', '## Interpretation', '',
                  '- `FIRST_SAMPLED_FRAME` and `NO_PERSON_IN_PREVIOUS_FRAME` are expected restarts.',
                  '- `TIME_GAP_GATE` means the sampled-frame interval failed the 0.04–1.0 s gate.',
                  '- `CENTER_SHIFT_GATE` and `BOX_SCALE_GATE` indicate a strict bounding-box matching gate.',
                  '- `AMBIGUITY_ABSTENTION` means the tracker declined to guess an identity.',
                  '- `UNPAIRED_WITH_FEASIBLE_CANDIDATE` may be caused by competition for the same prior person.',
                  '', '## Important limits', ''])
    lines.extend('- '+w for w in report['warnings'])
    lines.extend(['', 'Next: review these counts and representative frames before changing thresholds or reidentification logic.', ''])
    return '\n'.join(lines)


def audit(job_dir: Path) -> tuple[Path, Path, dict]:
    job_dir = Path(job_dir)
    if not job_dir.is_dir() or job_dir.is_symlink():
        raise AuditError('Job directory missing or unsafe')
    job_dir = job_dir.resolve()
    p = _load(job_dir, ROOT_FILES[0]); t = _load(job_dir, ROOT_FILES[1])
    report = inspect(p, t)
    for name in (AUDIT_JSON, AUDIT_MD, AUDIT_JSON+'.tmp', AUDIT_MD+'.tmp'):
        if (job_dir/name).is_symlink():
            raise AuditError('Unsafe report output file')
    jpath, mpath = job_dir/AUDIT_JSON, job_dir/AUDIT_MD
    try:
        (job_dir/(AUDIT_JSON+'.tmp')).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        (job_dir/(AUDIT_MD+'.tmp')).write_text(render_markdown(report), encoding='utf-8')
        os.replace(job_dir/(AUDIT_JSON+'.tmp'),jpath)
        os.replace(job_dir/(AUDIT_MD+'.tmp'),mpath)
    finally:
        for name in (AUDIT_JSON+'.tmp', AUDIT_MD+'.tmp'):
            if (job_dir/name).exists():
                (job_dir/name).unlink()
    return jpath, mpath, report


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit M5C.3 fragmentation without model inference')
    parser.add_argument('--job-dir',required=True,type=Path)
    args=parser.parse_args()
    jp, mp, r = audit(args.job_dir)
    o=r['overview']
    print(f"M5C.4 AUDIT SUCCESS: {r['clipCount']} clips; {o['sampledFrames']} frames")
    print(f"Tracks: {o['tracksCreated']} (one-frame: {o['singleFrameTracks']}); adjacent matches: {o['matchedAdjacent']}; ambiguous restarts: {o['ambiguousNewTracks']}")
    print('Reasons for new tracks:')
    for k,v in sorted(r['trackCreationReasons'].items(),key=lambda kv:-kv[1]):
        print(f'  {k}: {v}')
    print('JSON:',jp)
    print('Readable report:',mp)
    print('Audit only — no tracker/model changes or verified fighter IDs.')


if __name__=='__main__':
    main()
