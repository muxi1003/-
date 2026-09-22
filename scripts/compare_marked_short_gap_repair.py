from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


MARKER_PATTERN = re.compile(r"漏检|[比必]实际[多少].*呼吸")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    annotation_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Compare paper_repro_rr outputs only for videos carrying miss/count-error txt markers."
    )
    parser.add_argument("--annotation-root", type=Path, default=annotation_root)
    parser.add_argument(
        "--before-summary",
        type=Path,
        default=annotation_root / "paper_repro_summary.csv",
    )
    parser.add_argument("--after-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def marked_annotations(root: Path) -> dict[str, list[str]]:
    annotations: dict[str, list[str]] = {}
    for marker in sorted(root.rglob("*.txt"), key=lambda path: str(path).lower()):
        if MARKER_PATTERN.search(marker.stem):
            annotations.setdefault(marker.parent.name, []).append(marker.stem)
    return annotations


def annotation_types(values: list[str]) -> str:
    types = []
    if any("漏检" in value for value in values):
        types.append("漏检")
    if any(re.search(r"[比必]实际多.*呼吸", value) for value in values):
        types.append("比实际多呼吸")
    if any(re.search(r"比实际少.*呼吸", value) for value in values):
        types.append("比实际少呼吸")
    return ";".join(types)


def numeric(row: pd.Series, name: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(name, math.nan)]), errors="coerce").iloc[0]
    return float(value) if not pd.isna(value) else math.nan


def compare_outcome(before_count_error: float, after_count_error: float, before_rr_error: float, after_rr_error: float) -> str:
    before_count_abs = abs(before_count_error)
    after_count_abs = abs(after_count_error)
    if after_count_abs < before_count_abs:
        return "improved"
    if after_count_abs > before_count_abs:
        return "worsened"
    before_rr_abs = abs(before_rr_error)
    after_rr_abs = abs(after_rr_error)
    if math.isfinite(before_rr_abs) and math.isfinite(after_rr_abs):
        if after_rr_abs < before_rr_abs - 1e-9:
            return "improved"
        if after_rr_abs > before_rr_abs + 1e-9:
            return "worsened"
    return "unchanged"


def build_comparison(
    annotations: dict[str, list[str]],
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    before_by_id = before.set_index("video_id", drop=False)
    after_by_id = after.set_index("video_id", drop=False)
    missing_before = sorted(set(annotations) - set(before_by_id.index.astype(str)))
    missing_after = sorted(set(annotations) - set(after_by_id.index.astype(str)))
    if missing_before or missing_after:
        raise ValueError(
            f"Marked videos missing from summaries: before={missing_before}, after={missing_after}"
        )

    rows: list[dict[str, object]] = []
    for video_id, marker_values in sorted(annotations.items()):
        before_row = before_by_id.loc[video_id]
        after_row = after_by_id.loc[video_id]
        truth_count = numeric(after_row, "truth_count")
        if not math.isfinite(truth_count):
            truth_count = numeric(before_row, "truth_count")
        truth_rr = numeric(after_row, "truth_rr")
        if not math.isfinite(truth_rr):
            truth_rr = numeric(before_row, "truth_rr")
        before_peaks = numeric(before_row, "peaks")
        after_peaks = numeric(after_row, "peaks")
        before_rr = numeric(before_row, "rr_bpm")
        after_rr = numeric(after_row, "rr_bpm")
        before_count_error = before_peaks - truth_count
        after_count_error = after_peaks - truth_count
        before_rr_error = before_rr - truth_rr
        after_rr_error = after_rr - truth_rr
        outcome = compare_outcome(
            before_count_error,
            after_count_error,
            before_rr_error,
            after_rr_error,
        )
        rows.append(
            {
                "video_id": video_id,
                "annotations": ";".join(marker_values),
                "annotation_types": annotation_types(marker_values),
                "truth_peak_count": int(truth_count),
                "before_peak_count": int(before_peaks),
                "after_peak_count": int(after_peaks),
                "peak_count_delta": int(after_peaks - before_peaks),
                "before_abs_peak_error": int(abs(before_count_error)),
                "after_abs_peak_error": int(abs(after_count_error)),
                "truth_rr_bpm": truth_rr,
                "before_rr_bpm": before_rr,
                "after_rr_bpm": after_rr,
                "rr_delta_bpm": after_rr - before_rr,
                "before_abs_rr_error_bpm": abs(before_rr_error),
                "after_abs_rr_error_bpm": abs(after_rr_error),
                "improved": "yes" if outcome == "improved" else "no",
                "outcome": outcome,
                "是否改善": "是" if outcome == "improved" else "否",
                "改善结论": {"improved": "改善", "unchanged": "未变化", "worsened": "变差"}[outcome],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    annotations = marked_annotations(args.annotation_root.resolve())
    before = pd.read_csv(args.before_summary.resolve())
    after = pd.read_csv(args.after_summary.resolve())
    comparison = build_comparison(annotations, before, after)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(comparison)} marked-video comparisons: {output}")
    print(comparison["outcome"].value_counts().to_dict())


if __name__ == "__main__":
    main()
