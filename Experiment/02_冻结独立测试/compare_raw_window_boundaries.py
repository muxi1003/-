"""Describe raw-frame content changes, without transferring or reading count labels."""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import HOLDOUT, read_csv, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    audit_root = HOLDOUT / "raw_input_validation/20260912_v1"
    old = read_csv(audit_root / "raw_window_audit.csv").set_index("video_id")
    results = []
    for video_id, row in old.iterrows():
        original = pd.read_csv(audit_root / "raw_window_timestamps" / f"{video_id}.csv")
        current = pd.read_csv(args.root / "timestamps" / f"{video_id}.csv")
        inside = current[current.relative_seconds.lt(30)]
        n_old, n_new = int(row.frames_decoded), len(inside)
        common = min(n_old, n_new)
        same_origin = len(current) > 0 and abs(float(current.iloc[0].absolute_seconds)-float(original.iloc[0].raw_timestamp_seconds)) < 1e-6
        shared_times_match = common > 0 and (abs(current.iloc[:common].absolute_seconds.to_numpy()-original.iloc[:common].raw_timestamp_seconds.to_numpy()) < 1e-6).all()
        results.append({"video_id": video_id, "old_raw_frame_count": n_old,
                        "new_raw_frames_before_30_seconds": n_new,
                        "new_minus_old_frame_count": n_new-n_old,
                        "same_origin_timestamp": bool(same_origin), "shared_frame_timestamps_match": bool(shared_times_match),
                        "old_last_raw_relative_seconds": float(original.iloc[-1].raw_relative_to_first_frame_seconds),
                        "frame_content_scope": "same_index_range" if n_new == n_old and same_origin and shared_times_match else "boundary_or_origin_changed",
                        "reference_transfer": "not_performed; no_human_counts_read; review_before_any_relabeling"})
    write_csv(args.root / "reference_window_boundary_comparison.csv", results)
    print(pd.DataFrame(results).frame_content_scope.value_counts().to_string())


if __name__ == "__main__":
    main()
