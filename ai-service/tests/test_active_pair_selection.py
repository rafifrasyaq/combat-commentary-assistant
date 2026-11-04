"""M5C.6 tests: explicitly evaluate abstention and preserve review judgments."""
import csv
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.action.active_pair_selection import (
    PairSelectionError,
    _write_output,
    generate,
    propose_frame,
    run,
)


def human(x, area=.25, conf=.9, left_wrist=None):
    # Bounding box fits a frame; shoulder/wrist are intentionally simple.
    width = area**.5
    x1 = max(0.0, min(1.0-width, x))
    y1 = .2
    y2 = min(1.0, y1 + width)
    p = {"personDetectionScore": conf, "bboxXyxyNormalized": [x1,y1,x1+width,y2],
         "keypoints": {"left_shoulder":{"reported":True,"x":x1+.35*width,"y":y1+.2*width}}}
    if left_wrist is not None:
        p["keypoints"]["left_wrist"] = {"reported":True,"x":left_wrist,"y":y1+.3*width}
    return p


def frame_and_assoc(persons, *, frame_idx=0, clip_id=1):
    f = {"frameIndex":frame_idx,"clipTimeSeconds":float(frame_idx)/12,"sourceTimeSeconds":15+float(frame_idx)/12,"people":persons}
    tr = {"frameIndex":frame_idx,"clipTimeSeconds":float(frame_idx)/12,"associations":[{"detectionIndex":i,"trackId":f"c{clip_id:02d}_r{i+1:02d}","associationStatus":"NEW_TRACK"} for i in range(len(persons))]}
    return f, tr


def fixture_docs(persons=None):
    f, tr = frame_and_assoc(persons if persons is not None else [human(.08),human(.56)])
    p = {"schemaVersion":1,"stage":"UNVERIFIED_POSE_OBSERVATIONS","jobId":"sample-job","videoId":"sample-video","clipCount":1,"annotations":[{"candidateIndex":1,"frames":[f]}]}
    t = {"schemaVersion":1,"stage":"UNVERIFIED_EXPERIMENTAL_CLIP_LOCAL_POSE_ASSOCIATIONS","jobId":"sample-job","videoId":"sample-video","clipCount":1,"annotations":[{"candidateIndex":1,"frames":[tr]}]}
    return p,t


def test_no_people_abstains():
    f,tr=frame_and_assoc([])
    a=propose_frame(f,tr)
    assert a['status']=='UNKNOWN_INSUFFICIENT_PEOPLE'
    assert a['selectedTrackIds']==[]


def test_one_person_abstains():
    f,tr=frame_and_assoc([human(.15)])
    assert propose_frame(f,tr)['status']=='UNKNOWN_INSUFFICIENT_PEOPLE'


def test_small_person_not_assumed_fighter():
    f,tr=frame_and_assoc([human(.12),human(.55,area=.01)])
    a=propose_frame(f,tr)
    assert a['status']=='UNKNOWN_INSUFFICIENT_PEOPLE'
    assert all(p['assignment']=='UNASSIGNED' for p in a['personAssignments'])


def test_close_pair_can_be_proposed_but_is_unverified():
    people = [human(.17, left_wrist=.54), human(.51, left_wrist=.40)]
    f,tr=frame_and_assoc(people)
    a=propose_frame(f,tr)
    assert a['status']=='PAIR_CANDIDATE_UNVERIFIED'
    assert a['selectedDetectionIndexes']==[0,1]
    assert a['provisionalScore'] is not None


def test_ambiguous_three_person_frame_abstains():
    # Three identical candidates: even if there is a high pair score, the
    # runner-up cannot be differentiated and the output MUST abstain.
    people=[human(.20,left_wrist=.40),human(.20,left_wrist=.40),human(.20,left_wrist=.40)]
    f,tr=frame_and_assoc(people)
    a=propose_frame(f,tr)
    assert a['status']=='AMBIGUOUS_COMPETING_PAIRS'
    assert not a['selectedDetectionIndexes']


def test_weak_evidence_abstains():
    people=[human(.0,area=.08),human(.70,area=.08)]
    f,tr=frame_and_assoc(people)
    a=propose_frame(f,tr)
    assert a['status']=='UNKNOWN_WEAK_PAIR_EVIDENCE'


def test_all_unassigned_when_competing_pairs():
    people=[human(.2,left_wrist=.4)]*3
    f,tr=frame_and_assoc(people)
    a=propose_frame(f,tr)
    assert all(p['assignment']=='UNASSIGNED' for p in a['personAssignments'])


def test_source_job_mismatch_rejected():
    p,t=fixture_docs()
    t['videoId']='wrong'
    with pytest.raises(PairSelectionError):generate(p,t)


def test_pose_tracking_frame_mismatch_rejected():
    p,t=fixture_docs()
    t['annotations'][0]['frames'][0]['frameIndex']=2
    with pytest.raises(PairSelectionError):generate(p,t)


def test_duplicate_misaligned_detection_index_rejected():
    p,t=fixture_docs()
    t['annotations'][0]['frames'][0]['associations'][1]['detectionIndex']=0
    with pytest.raises(PairSelectionError):generate(p,t)


def test_nan_invalid_bbox_rejected():
    p,t=fixture_docs()
    p['annotations'][0]['frames'][0]['people'][0]['bboxXyxyNormalized'][0]=float('nan')
    with pytest.raises(PairSelectionError):generate(p,t)


def test_does_not_claim_truth():
    p,t=fixture_docs([human(.17,left_wrist=.54),human(.51,left_wrist=.40)])
    result=generate(p,t)
    assert result['verifiedFighterIdentity'] is False
    assert result['combatEventsDetected'] is False
    assert result['verifiedPairSelectionAccuracy'] is False
    assert result['provisionalScoreIsCalibratedProbability'] is False


def test_input_dicts_are_not_modified():
    p,t=fixture_docs()
    a,b=deepcopy(p),deepcopy(t)
    generate(p,t)
    assert p==a and t==b


def test_review_csv_not_overwritten(tmp_path):
    p,t=fixture_docs()
    output=generate(p,t)
    out=_write_output(tmp_path,output)
    csvpath=out/'active_pair_review.csv'
    csvpath.write_text('my precious human edits',encoding='utf-8')
    with pytest.raises(PairSelectionError,match='refusing to overwrite'):
        _write_output(tmp_path,output)
    assert csvpath.read_text(encoding='utf-8')=='my precious human edits'


def test_run_without_previews_preserves_sources(tmp_path):
    p,t=fixture_docs()
    (tmp_path/'pose_observations.json').write_text(json.dumps(p),encoding='utf8')
    (tmp_path/'pose_tracks_refined.json').write_text(json.dumps(t),encoding='utf8')
    output=run(tmp_path,previews=False)
    assert output.frame_count==1 and output.preview_count==0
    assert (output.output_dir/'active_pair_report.md').is_file()
    with (output.output_dir/'active_pair_review.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['humanPairVerdict']=='UNREVIEWED'
    assert json.loads((tmp_path/'pose_tracks_refined.json').read_text())==t


def test_preview_renders_with_injected_frames(tmp_path):
    np = pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    p,t=fixture_docs([human(.18,left_wrist=.55),human(.52,left_wrist=.40)])
    (tmp_path/'pose_observations.json').write_text(json.dumps(p),encoding='utf8')
    (tmp_path/'pose_tracks_refined.json').write_text(json.dumps(t),encoding='utf8')
    def fake_reader(_):
        return [SimpleNamespace(frame_index=0,clip_time_seconds=0.0,rgb=np.zeros((120,160,3),dtype='uint8'))]
    res=run(tmp_path,previews=True,frame_reader=fake_reader,clips=[({'candidateIndex':1},tmp_path/'fake.mp4')])
    assert res.preview_count==1
    assert (res.output_dir/'candidate_01_pair_proposals.jpg').is_file()


def test_run_prevents_rerun_after_csv_review(tmp_path):
    p,t=fixture_docs()
    (tmp_path/'pose_observations.json').write_text(json.dumps(p),encoding='utf8')
    (tmp_path/'pose_tracks_refined.json').write_text(json.dumps(t),encoding='utf8')
    run(tmp_path, previews=False)
    with pytest.raises(PairSelectionError,match='Existing human review CSV'):
        run(tmp_path, previews=False)
