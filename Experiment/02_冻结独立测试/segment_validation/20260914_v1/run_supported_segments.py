"""Local segment inference on internal artifacts, no human count reference loaded."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, sha256, write_csv, write_json
from supported_segments import POLICY, segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    frozen = HOLDOUT / "method_snapshots/20260909_v1"
    for item in json.loads((frozen / "freeze_manifest.json").read_text(encoding="utf-8"))["files"]:
        if sha256(frozen/item["snapshot_relative_path"]) != item["sha256"]:
            raise ValueError("Frozen method changed")
    sys.path.insert(0,str(frozen/"scripts"))
    import paper_repro_rr as rr
    config = rr.ReproConfig(**json.loads((frozen/"method_config.json").read_text(encoding="utf-8"))["signal_config"])
    if config.truth_csv is not None or config.optimize_peaks or config.adaptive_peak_retuning:
        raise ValueError("Truth-dependent configuration forbidden")
    upstream = HOLDOUT / "signal_guard_validation/20260914_v1"
    seal = json.loads((upstream/"prediction_seal.json").read_text(encoding="utf-8"))
    for key,file in [("internal_predictions_sha256","internal_predictions_before_scoring.csv"),("raw_predictions_sha256","raw_predictions_unscored.csv")]:
        if sha256(upstream/file) != seal[key]:
            raise ValueError("Upstream predictions changed")
    for p in [Path(__file__),Path(__file__).with_name("supported_segments.py")]:
        (args.out/p.name).write_bytes(p.read_bytes())
    inputs = []
    for cohort,filename in [("internal49","internal_predictions_before_scoring.csv"),("raw49_unscored","raw_predictions_unscored.csv")]:
        rows = pd.read_csv(upstream/filename,dtype={"video_id":str},keep_default_na=False)
        if len(rows)!=49 or rows.video_id.nunique()!=49:
            raise ValueError("Internal cohort mismatch")
        for row in rows.to_dict("records"):
            item = {"cohort":cohort,**row}
            item["upstream_rejected"] = str(row["reason"]).startswith("upstream_")
            if not item["upstream_rejected"]:
                path = (upstream/"internal49/input_temperatures"/f"{row['video_id']}.csv") if cohort=="internal49" else HOLDOUT/"raw_pipeline_validation/20260912_v1/adaptive_roi_20260914_v2/temperatures"/f"{row['video_id']}.csv"
                for kind,p in [("temperature",path),("support",upstream/cohort/"support"/f"{row['video_id']}.csv")]:
                    dest = args.out/"inputs"/cohort/kind/p.name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(p.read_bytes())
                    item[kind+"_path"] = str(dest)
                    item[kind+"_sha256"] = sha256(dest)
            inputs.append(item)
    write_csv(args.out/"input_manifest.csv",inputs)
    write_json(args.out/"protocol_before_results.json",{"policy":POLICY,"input_manifest_sha256":sha256(args.out/"input_manifest.csv"),
               "method_manifest_sha256":sha256(frozen/"freeze_manifest.json"),"reference_counts_read":False,
               "adoption":"diagnostic_candidate_only_no_accuracy_claim_no_default_change","upstream_time_scope":"inherited_N048_N049_not_new_decoder_validation"})
    windows, all_spans, events = [], [], []
    for item in inputs:
        cohort,video_id = item["cohort"],item["video_id"]
        duration = float(item["duration_seconds"])
        result = {"cohort":cohort,"video_id":video_id,"window_seconds":duration,"full_window_count":None,
                  "partial_count":None,"eligible_seconds":0.,"partial_rr_bpm":None,"status":"no_eligible_segment"}
        if item["upstream_rejected"]:
            result["status"] = item["reason"]
            windows.append(result)
            continue
        table = pd.read_csv(item["temperature_path"],float_precision="round_trip")
        support = pd.read_csv(item["support_path"],float_precision="round_trip")
        if len(table)!=len(support) or len(table)!=int(item["frames"]) or not np.array_equal(support.frame_index,np.arange(len(table))):
            raise ValueError("Frame/support alignment mismatch")
        mask = support.support_after_short_gap_policy.to_numpy(bool)
        dt = duration/len(table)
        spans = segments(mask,duration)
        count = 0
        seconds = 0.
        accepted = 0
        for ordinal,span in enumerate(spans):
            record = {"cohort":cohort,"video_id":video_id,"segment_id":ordinal,**span,"segment_count":None,"segment_rr_bpm":None,"status":"short_fragment"}
            if span["eligible_duration"]:
                start,stop = span["start_frame"],span["stop_frame_exclusive"]
                full = start==0 and stop==len(table)
                local_config = config if full else replace(config,fusion_mode=item["selected_mode"])
                local = table.iloc[start:stop].reset_index(drop=True)
                curve,summary = rr.fuse_temperature_curve(local,local_config,truth_row=None)
                if full and int(summary["peaks"])!=int(float(item["control_count"])):
                    raise ValueError("Full-window baseline replay drift")
                indices = np.flatnonzero(curve.is_peak.to_numpy(bool))+start
                indices = indices[(indices>=span["core_start_frame"]) & (indices<span["core_stop_frame_exclusive"])]
                record["segment_count"] = len(indices)
                record["status"] = "eligible" if len(indices)>=POLICY["minimum_peaks_for_segment_RR"] else "insufficient_peaks_for_RR"
                curve["original_frame_index"] = np.arange(len(curve))+start
                curve["window_time_seconds"] = curve.original_frame_index*dt
                write_csv(args.out/"segment_curves"/cohort/video_id/f"segment_{ordinal:03d}.csv",curve)
                if record["status"]=="eligible":
                    count += len(indices)
                    seconds += span["core_seconds"]
                    accepted += 1
                    record["segment_rr_bpm"] = 60*len(indices)/span["core_seconds"]
                    for index in indices:
                        events.append({"cohort":cohort,"video_id":video_id,"segment_id":ordinal,"original_frame_index":int(index),"time_seconds":index*dt,"human_reference":False})
            all_spans.append(record)
        result.update(accepted_segments=accepted,supported_seconds=float(mask.sum()*dt),direct_selected_seconds=float(support.selected_inputs_direct.sum()*dt),eligible_seconds=seconds)
        if accepted:
            result.update(partial_count=count,partial_rr_bpm=60*count/seconds,status="partial_observation_only")
            if len(spans)==1 and bool(mask.all()) and abs(seconds-duration)<1e-8:
                result.update(full_window_count=count,status="full_window_supported_proxy")
        windows.append(result)
    write_csv(args.out/"window_results.csv",windows)
    write_csv(args.out/"segments.csv",all_spans)
    write_csv(args.out/"algorithm_events.csv",events)
    summary = {}
    for cohort in ["internal49","raw49_unscored"]:
        data = pd.DataFrame(windows).query("cohort == @cohort")
        summary[cohort] = {"windows":49,"any_segment_output":int(data.partial_count.notna().sum()),"full_window_outputs":int(data.full_window_count.notna().sum()),
                           "eligible_seconds":float(data.eligible_seconds.sum()),"requested_seconds":float(data.window_seconds.sum())}
    write_json(args.out/"summary.json",{"cohorts":summary,"accuracy_scored":False,"default_changed":False,"external_predictions":0})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
