"""Check raw pipeline output consistency without animal reference labels."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, sha256, write_json
from timestamp_adapter import sample_map


def verify(root):
    frozen = HOLDOUT / "method_snapshots/20260909_v1"
    manifest = json.loads((frozen / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        assert sha256(frozen / item["snapshot_relative_path"]) == item["sha256"]
    table = pd.read_csv(root / "predictions_unscored.csv", keep_default_na=False)
    assert len(table) == 49 and table.video_id.nunique() == 49
    successful, abstained, frames, events = 0, 0, 0, 0
    for row in table.itertuples():
        times = pd.read_csv(root / "timestamps" / f"{row.video_id}.csv", float_precision="round_trip")
        mapping, status = sample_map(times.relative_seconds.to_numpy(), 30.0)
        if row.prediction_status == "abstain":
            assert row.predicted_breath_count == "" and row.predicted_rr_bpm == ""
            assert mapping is None or row.reason == "no_valid_temperature_signal"
            if mapping is None:
                assert row.reason == status["reason"]
            abstained += 1
            continue
        assert row.prediction_status == "predicted_internal_only" and mapping is not None
        actual = pd.read_csv(root / "maps" / f"{row.video_id}.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(mapping, actual, check_dtype=False, atol=1e-10, rtol=0)
        hashes = pd.read_csv(root / "frame_hashes" / f"{row.video_id}.csv")
        assert len(hashes) == len(actual) and np.array_equal(hashes.target_index, actual.target_index)
        assert np.array_equal(hashes.source_frame_index, actual.source_frame_index)
        for item in hashes.itertuples():
            path = root / "sampled_frames" / str(row.video_id) / f"frame_{item.target_index:06d}.jpg"
            assert sha256(path) == item.jpeg_sha256
        temperatures = pd.read_csv(root / "temperatures" / f"{row.video_id}.csv")
        curve = pd.read_csv(root / "curves" / f"{row.video_id}.csv")
        event = pd.read_csv(root / "algorithm_events" / f"{row.video_id}.csv")
        assert len(temperatures) == len(curve) == len(mapping)
        indices = np.flatnonzero(curve.is_peak.to_numpy(bool))
        assert np.array_equal(indices, event.grid_index.to_numpy(int))
        count = float(row.predicted_breath_count)
        assert count.is_integer() and len(event) == int(count)
        assert float(row.predicted_rr_bpm) == 2 * len(event)
        assert not event.reference.any()
        np.testing.assert_allclose(event.time_seconds, mapping.iloc[indices].target_time_seconds, atol=1e-10)
        successful += 1
        frames += len(actual)
        events += len(event)
    return {"status": "ARTIFACT_CONSISTENCY_PASS", "windows": 49,
            "predicted_windows": successful, "abstained_windows": abstained,
            "verified_prediction_frames": frames, "algorithm_events_not_ground_truth": events,
            "frozen_files_unchanged": len(manifest["files"]),
            "executed_runner_sha256": sha256(root / "validate_raw_pipeline_internal49.py"),
            "executed_adapter_sha256": sha256(root / "timestamp_adapter.py"),
            "accuracy_validated": False, "prediction_csv_sha256": sha256(root / "predictions_unscored.csv")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    write_json(args.root / "artifact_verification.json", result)
    print(json.dumps(result, indent=2))
