"""M5C.5: guardrails and synthetic vision QA (no neural inference)."""
from __future__ import annotations

import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.action.identity_review import IdentityReviewError, _palette, _load_and_validate, run_review


@dataclass
class Sample:
    frame_index: int
    clip_time_seconds: float
    rgb: object


def fixtures(tmp_path):
    job = tmp_path / 'job'; job.mkdir()
    clip = job / 'clip.mp4'; clip.write_bytes(b'fake')
    person = {'bboxXyxyNormalized':[0.1,0.1,0.6,0.8], 'keypoints':{}}
    pose = {'stage':'UNVERIFIED_POSE_OBSERVATIONS','clipCount':1,'jobId':'job-1','videoId':'video-1',
            'annotations':[{'candidateIndex':1,'frames':[{'frameIndex':0,'clipTimeSeconds':0.0,'sourceTimeSeconds':1.0,'people':[person]}]}]}
    track = {'stage':'UNVERIFIED_EXPERIMENTAL_CLIP_LOCAL_POSE_ASSOCIATIONS','clipCount':1,'jobId':'job-1','videoId':'video-1',
             'annotations':[{'candidateIndex':1,'frames':[{'frameIndex':0,'clipTimeSeconds':0.0,'associations':[
                {'detectionIndex':0,'trackId':'c01_r01','associationStatus':'AMBIGUOUS_NEW_TRACK'}]}]}]}
    for filename,doc in [('pose_observations.json',pose),('pose_tracks_refined.json',track)]:
        (job/filename).write_text(json.dumps(doc), encoding='utf-8')
    clips = [({'candidateIndex':1,'startSeconds':1.0,'jobId':'job-1'},clip)]
    return job, clip, pose, track, clips


def test_real_output_has_correct_source_alignment():
    folder = Path('/mnt/data/m5c4_work')
    if not (folder / 'pose_tracks_refined.json').exists():
        pytest.skip('Original M5C.4 review data not bundled')
    # Avoid any assumption about a developer's machine when running in user's repo.
    p = json.loads((folder/'pose_observations.json').read_text())
    t = json.loads((folder/'pose_tracks_refined.json').read_text())
    assert len(p['annotations']) == len(t['annotations']) == 7
    assert sum(len(x['frames']) for x in p['annotations']) == 84
    for pa,ta in zip(p['annotations'],t['annotations']):
        assert [f['frameIndex'] for f in pa['frames']] == [f['frameIndex'] for f in ta['frames']]


def test_build_review_images_csv(tmp_path):
    job,clip,pose,track,clips = fixtures(tmp_path)
    def frames(_path): return [Sample(0,0.0,np.zeros((150,240,3),dtype=np.uint8))]
    result = run_review(job,frame_reader=frames,clip_validator=lambda _: ({},clips))
    assert (job/'identity_review/candidate_01_tracks.jpg').stat().st_size > 5000
    with (job/'identity_review/priority_cases.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['visualVerdict']=='UNREVIEWED'
    assert result.priority_cases==1 and result.rendered_clips==1
    assert json.loads((job/'identity_review/review_manifest.json').read_text())['stage']=='HUMAN_IDENTITY_REVIEW_PENDING'


def test_no_false_positive_in_unreviewed_state(tmp_path):
    job,clip,pose,track,clips=fixtures(tmp_path)
    assert _load_and_validate(job,clips)[0][1]['frames'][0]['associations'][0]['associationStatus']=='AMBIGUOUS_NEW_TRACK'


@pytest.mark.parametrize('change,expected',[('wrong_video','different jobs/videos'),('bad_id','Invalid track ID'),('bad_box','Invalid normalized'),('wrong_index','association order/index'),('wrong_time','alignment mismatch'),('duplicate_track','Detection and association counts differ')])
def test_invalid_inputs_rejected(tmp_path,change,expected):
    job,clip,pose,track,clips=fixtures(tmp_path)
    if change=='wrong_video':track['videoId']='other'
    if change=='bad_id':track['annotations'][0]['frames'][0]['associations'][0]['trackId']='bad'
    if change=='bad_box':pose['annotations'][0]['frames'][0]['people'][0]['bboxXyxyNormalized']=[.8,.1,.2,.7]
    if change=='wrong_index':track['annotations'][0]['frames'][0]['associations'][0]['detectionIndex']=2
    if change=='wrong_time':track['annotations'][0]['frames'][0]['clipTimeSeconds']=1.0
    if change=='duplicate_track':track['annotations'][0]['frames'][0]['associations']=[]
    (job/'pose_observations.json').write_text(json.dumps(pose))
    (job/'pose_tracks_refined.json').write_text(json.dumps(track))
    with pytest.raises(IdentityReviewError,match=expected):_load_and_validate(job,clips)


def test_palette_stable():
    assert _palette('c01_r01')==_palette('c01_r01')
    assert _palette('c01_r01')!=_palette('c01_r02')


def test_reject_symlink_input(tmp_path):
    job,clip,pose,track,clips=fixtures(tmp_path)
    path=job/'pose_tracks_refined.json';path.unlink()
    target=tmp_path/'target.json';target.write_text(json.dumps(track))
    try:path.symlink_to(target)
    except OSError:pytest.skip('symlinks unavailable')
    with pytest.raises(IdentityReviewError,match='unsafe'):_load_and_validate(job,clips)


def test_preserve_manual_review_on_second_run(tmp_path):
    job, clip, pose, track, clips = fixtures(tmp_path)
    def frames(_path): return [Sample(0,0.0,np.zeros((150,240,3),dtype=np.uint8))]
    run_review(job, frame_reader=frames, clip_validator=lambda _: ({},clips))
    review = job/'identity_review/manual_clip_review.csv'
    review.write_text('human judgment: pass', encoding='utf-8')
    with pytest.raises(IdentityReviewError, match='refusing to overwrite human judgments'):
        run_review(job, frame_reader=frames, clip_validator=lambda _: ({},clips))
    assert review.read_text(encoding='utf-8') == 'human judgment: pass'
