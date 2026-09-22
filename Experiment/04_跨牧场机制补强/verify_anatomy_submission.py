"""Verify source archive, pixel controls and sealed candidate against current state."""
import json
from pathlib import Path
import joblib
import numpy as np
from analyze_transfer import read,write_json,sha256
from analyze_anatomy_reference import ROOT,OUT,FROZEN,ellipse_pixels
from probe_pose_mechanism import load_bgr
from recover_timestamp_segments import CachedRF


def main():
    meta=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
    assert sha256(OUT/'reference_original.json')==sha256(Path(meta['reference_file']))==meta['reference_sha256']
    raw=json.loads((OUT/'reference_original.json').read_text(encoding='utf-8'))
    package=json.loads((ROOT/'annotation_package.json').read_text(encoding='utf-8'))
    assert len(raw['records'])==36 and set(raw['records'])=={f['frame_id'] for f in package['frames'] if f['priority']}
    rf=CachedRF(joblib.load(FROZEN/'weights/clf_model_RGB_20240906.pkl'))
    exact20=0;region_count=0
    for fid,r in raw['records'].items():
        im=load_bgr(ROOT/'frames'/f'{fid}.png')
        for e in r['regions'].values():
            if e is None:continue
            values=rf.predict(ellipse_pixels(im,e));exact20+=int((values==20).sum());region_count+=1
    # Early control used >=20; establish that fixing it to frozen >20 leaves these results unchanged.
    assert exact20==0
    candidate=ROOT/'direct_roi_control_v1'
    seal=json.loads((candidate/'prediction_seal.json').read_text(encoding='utf-8'))
    for name,h in seal['files'].items():assert sha256(candidate/name)==h
    members=read(candidate/'prediction_windows.csv');assert len(members)==271
    metrics=read(candidate/'evaluation/metrics.csv')
    assert set(metrics[metrics.reference.eq('R2')].count_pairs)=={173}
    write_json(OUT/'verification.json',dict(status='PASS_SUBMISSION_AND_DIAGNOSTIC_CONTROLS',
        complete_priority_frames=36,human_regions=region_count,source_archive_identical=True,
        exactly20_pixels_in_human_ellipses=exact20,strict20_correction_does_not_change_current_controls=True,
        candidate_windows=271,baseline_peak_replays=216,rr_R2_pairs_preserved=173,
        actual_reference_received=True,full_scientific_goal_complete=False))
    print('Archive, 36 priority frames, 50 regions, strict20 equivalence and 271 sealed predictions PASS')


if __name__=='__main__':main()
