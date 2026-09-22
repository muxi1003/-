"""Additional hard gate for actual loaded method, runtime, timing and prediction replay."""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, read_csv, sha256, write_json
from run_frozen_external import verify_lock, verify_checkpoint
from timestamp_adapter import sample_map


def verify(run):
    lock, windows = verify_lock(run)
    release = json.loads((Path(lock["release_dir"]) / "test_release.json").read_text(encoding="utf-8"))
    frozen = Path(lock["method_snapshot"])
    if frozen.resolve() != Path(release["method_snapshot"]).resolve():
        raise ValueError("Actual method path differs from verified release")
    runtime = lock["runtime"]
    if runtime["python"] != sys.version or Path(runtime["executable"]).resolve() != Path(sys.executable).resolve() or runtime["platform"] != platform.platform():
        raise ValueError("Python/runtime changed")
    for name, expected in runtime["packages"].items():
        if version(name) != expected:
            raise ValueError(f"Package changed: {name}")
    seal = json.loads((run / "prediction_seal.json").read_text(encoding="utf-8"))
    if sha256(run / "protocol_lock.json") != seal["protocol_sha256"] or sha256(run / "prediction_windows.csv") != seal["predictions_sha256"]:
        raise ValueError("Final seal mismatch")
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    if Path(rr.__file__).resolve() != (frozen / "scripts/paper_repro_rr.py").resolve():
        raise ValueError("Unexpected signal module import")
    config = rr.ReproConfig(**json.loads((frozen / "method_config.json").read_text(encoding="utf-8"))["signal_config"])
    preflight = pd.read_csv(HOLDOUT / "preflight/20260911_v1/decoded_frame_timestamps.csv", float_precision="round_trip")
    checked, replayed = [], 0
    for row in windows.itertuples():
        directory = run / "windows" / row.window_id
        if sha256(directory / "checkpoint.json") != seal["checkpoints"][row.window_id]:
            raise ValueError("Changed checkpoint seal")
        result = verify_checkpoint(directory)
        if sha256(Path(row.source_path)) != row.source_sha256:
            raise ValueError("Raw source changed since release")
        time = pd.read_csv(directory / "timestamps.csv", float_precision="round_trip")
        earlier = preflight.loc[preflight.window_id.eq(row.window_id), "opencv_ffmpeg_timestamp_seconds"].to_numpy(float)
        if len(earlier) != len(time) or not np.allclose(earlier, time.absolute_seconds.to_numpy(), atol=1e-6, rtol=0):
            raise ValueError("Decode differs from pre-outcome preflight")
        if len(time) < 2 or time.absolute_seconds.iloc[-1] < 30 or abs(time.absolute_seconds.iloc[0]) > 1e-6:
            raise ValueError("No verified decoded boundary frame; EOF is not a successful full30 decode")
        mapping, status = sample_map(time.relative_seconds.to_numpy(), 30.0)
        if result["timestamp_policy_status"] != status["prediction_status"]:
            raise ValueError("Timestamp classification mismatch")
        if mapping is None:
            if result["prediction_status"] != "abstain" or result["predicted_count"] is not None or result["reason"] != status["reason"]:
                raise ValueError("Rejected timing emitted count")
        else:
            stored = pd.read_csv(directory / "map.csv", float_precision="round_trip")
            pd.testing.assert_frame_equal(stored, mapping, check_dtype=False, atol=1e-10, rtol=0)
            temperatures = pd.read_csv(directory / "temperatures.csv", float_precision="round_trip")
            if len(temperatures) != len(mapping):
                raise ValueError("Temporal length mismatch")
            if result["prediction_status"] == "ok":
                curve, summary = rr.fuse_temperature_curve(temperatures, config, truth_row=None)
                original = pd.read_csv(directory / "curve.csv", float_precision="round_trip")
                if int(summary["peaks"]) != result["predicted_count"] or not np.array_equal(curve.is_peak, original.is_peak):
                    raise ValueError("Peak replay mismatch")
                replayed += 1
            elif np.isfinite(temperatures[["left_temp", "right_temp"]].to_numpy(float)).any():
                raise ValueError("Unexpected abstention despite finite signal")
        checked.append(row.window_id)
    return {"status": "PASS_ENGINEERING_GATE_NOT_ACCURACY", "n_windows": len(checked), "window_ids": checked,
            "replayed_numeric_predictions": replayed, "protocol_sha256": sha256(run / "protocol_lock.json"),
            "prediction_seal_sha256": sha256(run / "prediction_seal.json"),
            "code_sha256": sha256(Path(__file__)), "review_independence": "deterministic_executor_gate; same_family_review_separately_disclosed",
            "limits": ["conservative_original_boundary_gap_rejections_retained_not_retuned_after_external_run", "runtime_current_and_lock_match_not_historical_independent_process_attestation", "same_OpenCV_backend_not_independent_decoder", "no_absolute_temperature_or_anatomical_visibility_validation"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    result = verify(args.run)
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "window_ids"}, ensure_ascii=False))
