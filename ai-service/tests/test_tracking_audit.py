"""M5C.4 audit tests: synthetic reports, no CUDA/Ultralytics needed."""
import json

import pytest

from app.action.tracking_audit import AuditError, audit, inspect, render_markdown


def p(cx, halfwidth=.1, height=.8):
    return {'bboxXyxyNormalized':[cx-halfwidth,.1,cx+halfwidth,.1+height]}


def raw_frame(idx,t,people):
    return {'frameIndex':idx,'clipTimeSeconds':t,'people':people,'personCount':len(people)}


def tracked_frame(idx,t,track_statuses):
    rows=[]
    for j,(tid,status) in enumerate(track_statuses):
        rows.append({'detectionIndex':j,'trackId':tid,'associationStatus':status,
                     'handMotion':{'left':{'measurable':False},'right':{'measurable':False}}})
    return {'frameIndex':idx,'clipTimeSeconds':t,'associations':rows}


def docs(frames, tracks, lengths):
    statuses=[a['associationStatus'] for f in tracks for a in f['associations']]
    obs={'stage':'UNVERIFIED_POSE_OBSERVATIONS','combatEventsDetected':False,'jobId':'job-x',
         'clipCount':1,'annotations':[{'candidateIndex':1,'frames':frames}]}
    out={'stage':'UNVERIFIED_CLIP_LOCAL_POSE_ASSOCIATIONS','combatEventsDetected':False,
         'punchesDetected':False,'stableFighterIdentities':False,'jobId':'job-x','clipCount':1,
         'totalClipLocalTracks':len(lengths),
         'matchedAdjacentDetections':statuses.count('MATCHED_ADJACENT'),
         'ambiguousRestarts':statuses.count('AMBIGUOUS_NEW_TRACK'),
         'annotations':[{'candidateIndex':1,'frames':tracks,'trackLengthsInSampledFrames':lengths,'trackCount':len(lengths)}]}
    return obs,out


def test_identifies_two_initial_tracks_plus_matches():
    a,b=docs([raw_frame(0,0,[p(.3),p(.7)]),raw_frame(1,.2,[p(.31),p(.69)])],
             [tracked_frame(0,0,[('c01_t01','NEW_TRACK'),('c01_t02','NEW_TRACK')]),
              tracked_frame(1,.2,[('c01_t01','MATCHED_ADJACENT'),('c01_t02','MATCHED_ADJACENT')])],
             {'c01_t01':2,'c01_t02':2})
    r=inspect(a,b)
    assert r['overview']['tracksCreated']==2
    assert r['overview']['matchedAdjacent']==2
    assert r['trackCreationReasons']=={'FIRST_SAMPLED_FRAME':2}
    assert r['overview']['singleFrameTracks']==0
    assert 'Tracking Quality Audit' in render_markdown(r)


def test_center_shift_causes_restart():
    a,b=docs([raw_frame(0,0,[p(.2)]),raw_frame(1,.2,[p(.8)])],
             [tracked_frame(0,0,[('c01_t01','NEW_TRACK')]),tracked_frame(1,.2,[('c01_t02','NEW_TRACK')])],
             {'c01_t01':1,'c01_t02':1})
    r=inspect(a,b)
    assert r['trackCreationReasons']['CENTER_SHIFT_GATE']==1
    assert r['overview']['singleFrameTracks']==2


def test_time_gap_causes_restart():
    a,b=docs([raw_frame(0,0,[p(.2)]),raw_frame(1,1.3,[p(.21)])],
             [tracked_frame(0,0,[('c01_t01','NEW_TRACK')]),tracked_frame(1,1.3,[('c01_t02','NEW_TRACK')])],
             {'c01_t01':1,'c01_t02':1})
    assert inspect(a,b)['trackCreationReasons']['TIME_GAP_GATE']==1


def test_ambiguous_and_previous_empty_restarts():
    a,b=docs([raw_frame(0,0,[p(.2),p(.5)]),raw_frame(1,.2,[p(.35)]),raw_frame(2,.4,[]),raw_frame(3,.6,[p(.34)])],
             [tracked_frame(0,0,[('c01_t01','NEW_TRACK'),('c01_t02','NEW_TRACK')]),
              tracked_frame(1,.2,[('c01_t03','AMBIGUOUS_NEW_TRACK')]),tracked_frame(2,.4,[]),
              tracked_frame(3,.6,[('c01_t04','NEW_TRACK')])],
             {'c01_t01':1,'c01_t02':1,'c01_t03':1,'c01_t04':1})
    r=inspect(a,b)
    assert r['trackCreationReasons']['AMBIGUITY_ABSTENTION']==1
    assert r['trackCreationReasons']['NO_PERSON_IN_PREVIOUS_FRAME']==1


def test_tampering_detected():
    a,b=docs([raw_frame(0,0,[p(.2)])],[tracked_frame(0,0,[('c01_t01','NEW_TRACK')])],{'c01_t01':1})
    b['jobId']='wrong'
    with pytest.raises(AuditError,match='job'):
        inspect(a,b)


def test_run_generates_only_new_reports(tmp_path):
    a,b=docs([raw_frame(0,0,[p(.2)])],[tracked_frame(0,0,[('c01_t01','NEW_TRACK')])],{'c01_t01':1})
    (tmp_path/'pose_observations.json').write_text(json.dumps(a))
    (tmp_path/'pose_tracks.json').write_text(json.dumps(b))
    other=tmp_path/'action_predictions.json'
    other.write_text('untouched')
    before=(tmp_path/'pose_tracks.json').read_bytes()
    outjson,outmd,r=audit(tmp_path)
    assert outjson.exists() and outmd.exists()
    assert r['clipCount']==1
    assert (tmp_path/'pose_tracks.json').read_bytes()==before
    assert other.read_text()=='untouched'
    assert not (tmp_path/'tracking_quality_report.json.tmp').exists()


def test_mismatched_track_count_fails():
    a,b=docs([raw_frame(0,0,[p(.2)])],[tracked_frame(0,0,[('c01_t01','NEW_TRACK')])],{'c01_t01':1})
    b['totalClipLocalTracks']=40
    with pytest.raises(AuditError,match='totals'):
        inspect(a,b)
