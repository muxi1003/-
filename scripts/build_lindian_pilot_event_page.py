"""Build an isolated event-annotation page for source-time Lindian clips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROUND = "L190588-P1"
COHORT = "lindian190588"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "Experiment/03_可靠事件参考/event_workspace_template.html",
    )
    return parser.parse_args()


def replace_exact(text: str, old: str, new: str, expected: int = 1) -> str:
    if text.count(old) != expected:
        raise ValueError(f"Template changed: expected {expected} occurrences of {old!r}")
    return text.replace(old, new)


def build_windows(segments: Path) -> list[dict[str, str]]:
    with segments.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    windows = []
    for row in rows:
        if row["status"] != "CUT":
            continue
        clip = Path(row["clip_path"]).resolve(strict=True)
        window_id = f"190588_{row['segment_id']}_source30s"
        windows.append(
            {
                "window_id": window_id,
                "video_id": f"190588_{row['segment_id']}",
                "video_path": str(clip),
                "source_path": row["source_path"],
                "source_start_seconds": row["source_start_seconds"],
                "duration_seconds": "30",
                "cohort": COHORT,
                "annotation_round": ROUND,
                "annotator": "",
                "annotation_status": "pending",
                "manual_breath_count": "",
                "event_definition": "expiration_peak",
                "predictions_hidden": "",
                "reference_notes": "",
                "prior_algorithm_exposure": "yes; earlier 190588 development window viewed",
                "browser_video_path": str(clip),
                "view_start_seconds": "0",
            }
        )
    if not windows or len({row["window_id"] for row in windows}) != len(windows):
        raise ValueError("No unique cut windows found")
    return windows


def build_page(template: str, windows: list[dict[str, str]]) -> str:
    page = replace_exact(template, "const initial=__WINDOW_DATA__;", "const initial=__WINDOW_DATA__;")
    page = replace_exact(page, "R2", ROUND, 8)
    page = replace_exact(page, "20260915.2", "20260924")
    page = replace_exact(page, '<option value="lindian49">林甸49窗</option><option value="jiufu271">久福271窗</option>',
                         f'<option value="{COHORT}">林甸190588试切</option>')
    page = replace_exact(page, "x.cohort==='lindian49'?'林甸：存在历史算法接触，不能宣称原始盲法。':'久福：本轮接触情况请如实登记。'",
                         "'同一原视频的多个窗口不独立；鼻部曝光和不可观察区间请逐窗记录。'")
    page = replace_exact(page, "cow-rr-event-L190588-P1-20260914-v1", "cow-rr-event-L190588-P1-20260924-v1")
    page = replace_exact(page, "'annotation_windows.csv'", "'lindian190588_annotation_windows.csv'")
    page = replace_exact(page, "'reference_events.csv'", "'lindian190588_reference_events.csv'")
    page = replace_exact(page, "'unobservable_intervals.csv'", "'lindian190588_unobservable_intervals.csv'")
    payload = json.dumps(windows, ensure_ascii=False).replace("<", "\\u003c")
    return replace_exact(page, "__WINDOW_DATA__", payload)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Will not overwrite annotation page: {args.output}")
    windows = build_windows(args.segments)
    template = args.template.read_text(encoding="utf-8")
    page = build_page(template, windows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"Created {args.output.resolve()} with {len(windows)} windows")


if __name__ == "__main__":
    main()
