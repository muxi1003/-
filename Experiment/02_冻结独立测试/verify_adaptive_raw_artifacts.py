"""Replay adaptive policy selection and signal counts; never use human references."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    frozen = HOLDOUT / "method_snapshots/20260909_v1"
    sys.path.insert(0, str(frozen / "scripts"))
    import paper_repro_rr as rr
    import evaluate_lindian_adaptive_roi as roi
    config_data = json.loads((frozen / "method_config.json").read_text(encoding="utf-8"))
    config, policy = rr.ReproConfig(**config_data["signal_config"]), config_data["roi_policy"]
    predictions = pd.read_csv(args.root / "predictions_unscored.csv", dtype={"video_id": str}, keep_default_na=False)
    assert len(predictions) == predictions.video_id.nunique() == 49
    compared, frame_count = 0, 0
    for row in predictions.itertuples():
        if row.prediction_status == "abstain":
            assert row.predicted_breath_count == "" and row.predicted_rr_bpm == ""
            continue
        radius_table = pd.read_csv(args.root / "radius_tables" / f"{row.video_id}.csv", float_precision="round_trip")
        cv = roi.nostril_spacing_cv(radius_table)
        exponent = policy["damped_exponent"] if math.isfinite(cv) and cv > policy["spacing_cv_threshold"] else policy["linear_exponent"]
        replay, radii = roi.radius_policy_table(radius_table, base_radius=policy["base_radius"], exponent=exponent, clip_low=policy["min_radius"], clip_high=policy["max_radius"])
        saved = pd.read_csv(args.root / "temperatures" / f"{row.video_id}.csv", float_precision="round_trip")
        for field in ["left_temp", "right_temp", "adaptive_roi_radius", "nostril_spacing_px", "reference_nostril_spacing_px"]:
            np.testing.assert_allclose(replay[field], saved[field], atol=1e-9, rtol=1e-10, equal_nan=True)
        assert radii.min() >= policy["min_radius"] and radii.max() <= policy["max_radius"]
        curve, summary = rr.fuse_temperature_curve(replay, config, truth_row=None)
        stored_curve = pd.read_csv(args.root / "curves" / f"{row.video_id}.csv")
        np.testing.assert_array_equal(curve.is_peak, stored_curve.is_peak)
        count = float(row.predicted_breath_count)
        assert count.is_integer() and int(summary["peaks"]) == int(count)
        assert float(row.predicted_rr_bpm) == 2 * int(count)
        compared += 1
        frame_count += len(replay)
    write_json(args.root / "artifact_verification.json", {"status":"POLICY_AND_COUNT_REPLAY_PASS",
               "windows":49,"replayed_predicted_windows":compared,"replayed_grid_frames":frame_count,
               "accuracy_validated":False,"predictions_sha256":sha256(args.root / "predictions_unscored.csv")})
    print(f"Policy/count replay: {compared} predicted windows; no reference accuracy claimed")


if __name__ == "__main__":
    main()
