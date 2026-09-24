"""Read-only audit: do exact-count windows still contain unmatched respiratory events?

Question
--------
Route B (an evaluation/benchmark paper) is only viable if count agreement does
not imply event correctness. This script measures that directly on the frozen
``full_v2`` same-input verification run.

Definitions
-----------
* Exact window: ``predicted_count == truth_count`` (the "complete count correct"
  criterion used throughout the project).
* For an exact window with reference count ``N`` and ``tp`` matched events, the
  count identity gives ``fp == fn == N - tp``. So ``N - tp`` is the number of
  events that are mismatched in *both* directions, i.e. pure timing error that
  the count metric cannot see.
* A window is "event-clean" only when ``tp == N`` (equivalently ``fp == fn == 0``).

Reads only. Writes one CSV plus one JSON summary next to itself.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

RUN_DIR = Path(r"E:\real\use_code\yoloV8\Experiment\05_方法核证\runs\full_v2")
OUT_DIR = Path(r"E:\real\use_code\yoloV8\Experiment\05_方法核证\runs\event_audit_v1")

COHORT = "anchored49"
ARMS = ["C0P0", "C1P0", "C0P1", "C1P1"]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def quantiles(values: list[float]) -> dict:
    """Five-number summary; uses inclusive median like the project reports."""
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = p * (n - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {
        "n": n,
        "min": ordered[0],
        "p25": pct(0.25),
        "median": statistics.median(ordered),
        "p75": pct(0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def main() -> int:
    predictions = read_csv(RUN_DIR / "predictions.csv")
    cases = read_csv(RUN_DIR / "cases.csv")

    # ---- restrict to the main-analysis cohort and reference disposition ----
    pred_keys = {}
    for row in predictions:
        if row["cohort"] != COHORT:
            continue
        key = (row["video_id"], row["arm"])
        pred_keys[key] = row

    case_by_key = {(row["video_id"], row["arm"]): row for row in cases}

    pairs = sorted(set(pred_keys) & set(case_by_key))
    print(f"joined windows on {COHORT}: {len(pairs)}")
    missing = sorted(set(pred_keys) ^ set(case_by_key))
    if missing:
        print(f"WARNING unjoined keys: {missing[:10]} (n={len(missing)})")

    # ---- reference disposition per video: must be identical across arms ----
    disposition: dict[str, set[str]] = defaultdict(set)
    duration: dict[str, float] = {}
    for (video_id, _arm), row in pred_keys.items():
        disposition[video_id].add(row["reference_status"].strip())
        value = as_float(row.get("duration_seconds"))
        if value is not None:
            duration[video_id] = value
    inconsistent = {v: sorted(s) for v, s in disposition.items() if len(s) != 1}
    if inconsistent:
        print(f"WARNING inconsistent reference_status: {inconsistent}")

    complete_videos = {v for v, s in disposition.items() if s == {"complete"}}
    print(f"videos with complete reference: {len(complete_videos)}")

    # ---- per-arm window table ----
    rows_out: list[dict] = []
    for (video_id, arm) in pairs:
        if video_id not in complete_videos:
            continue
        case = case_by_key[(video_id, arm)]
        pred = pred_keys[(video_id, arm)]
        truth = as_float(case["truth_count"])
        predicted = as_float(case["predicted_count"])
        tp = as_float(case["tp"])
        fp = as_float(case["fp"])
        fn = as_float(case["fn"])
        if None in (truth, predicted, tp, fp, fn):
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision == precision and recall == recall and (precision + recall) > 0
            else float("nan")
        )

        count_error = int(round(predicted - truth))
        # identity check: for any window, predicted - truth == fp - fn
        identity_gap = (predicted - truth) - (fp - fn)

        rows_out.append(
            {
                "cohort": COHORT,
                "video_id": video_id,
                "arm": arm,
                "cluster_id": pred.get("cluster_id", ""),
                "duration_seconds": duration.get(video_id, ""),
                "truth_count": int(round(truth)),
                "predicted_count": int(round(predicted)),
                "count_error": count_error,
                "exact_count": count_error == 0,
                "tp": int(round(tp)),
                "fp": int(round(fp)),
                "fn": int(round(fn)),
                "precision": precision,
                "recall": recall,
                "event_f1": f1,
                "unmatched_each_way": int(round(truth - tp)),
                "event_clean": int(round(tp)) == int(round(truth)),
                "output_status": pred.get("output_status", ""),
                "selected_mode": pred.get("selected_mode", ""),
                "identity_gap": identity_gap,
            }
        )

    # ---- invariant: predicted - truth == fp - fn ---------------------------------
    worst_identity = max((abs(r["identity_gap"]) for r in rows_out), default=0.0)
    print(f"max |(pred-truth)-(fp-fn)| = {worst_identity:.6f} (expect ~0)")

    # ---- summary per arm ---------------------------------------------------------
    summary: dict = {
        "run_dir": str(RUN_DIR),
        "cohort": COHORT,
        "note": (
            "Read-only audit of frozen full_v2 outputs. Exact window means "
            "predicted_count == truth_count. event_clean means tp == truth_count."
        ),
        "identity_check_max_abs_gap": worst_identity,
        "arms": {},
    }

    for arm in ARMS:
        arm_rows = [r for r in rows_out if r["arm"] == arm]
        if not arm_rows:
            continue
        exact = [r for r in arm_rows if r["exact_count"]]
        inexact = [r for r in arm_rows if not r["exact_count"]]

        exact_f1 = [r["event_f1"] for r in exact if r["event_f1"] == r["event_f1"]]
        exact_mismatch = [r["unmatched_each_way"] for r in exact]
        clean = [r for r in exact if r["event_clean"]]

        # how many exact windows have each level of double-sided timing error
        buckets = {
            "0": sum(1 for m in exact_mismatch if m == 0),
            "1": sum(1 for m in exact_mismatch if m == 1),
            "2": sum(1 for m in exact_mismatch if m == 2),
            "3+": sum(1 for m in exact_mismatch if m >= 3),
        }

        summary["arms"][arm] = {
            "windows": len(arm_rows),
            "exact_windows": len(exact),
            "exact_rate": len(exact) / len(arm_rows),
            "inexact_windows": len(inexact),
            "exact_and_event_clean": len(clean),
            "exact_but_not_event_clean": len(exact) - len(clean),
            "exact_but_not_event_clean_rate": (
                (len(exact) - len(clean)) / len(exact) if exact else None
            ),
            "exact_event_f1_summary": quantiles(exact_f1),
            "exact_unmatched_each_way_total": sum(exact_mismatch),
            "exact_unmatched_each_way_summary": quantiles(
                [float(m) for m in exact_mismatch]
            ),
            "exact_unmatched_buckets": buckets,
            "exact_f1_min": min(exact_f1) if exact_f1 else None,
            "exact_f1_max": max(exact_f1) if exact_f1 else None,
            "arm_event_f1_all_windows": quantiles(
                [r["event_f1"] for r in arm_rows if r["event_f1"] == r["event_f1"]]
            ),
            "arm_unmatched_each_way_total": sum(
                r["unmatched_each_way"] for r in arm_rows
            ),
        }

        # share of total timing error that hides inside count-exact windows
        total_mismatch = sum(r["unmatched_each_way"] for r in arm_rows)
        if total_mismatch:
            summary["arms"][arm]["exact_share_of_all_timing_error"] = (
                sum(exact_mismatch) / total_mismatch
            )

    # ---- worst exact windows, ranked by event error ------------------------------
    worst = sorted(
        (r for r in rows_out if r["exact_count"]),
        key=lambda r: (-r["unmatched_each_way"], r["event_f1"]),
    )
    summary["worst_exact_windows"] = [
        {
            "video_id": r["video_id"],
            "arm": r["arm"],
            "truth_count": r["truth_count"],
            "predicted_count": r["predicted_count"],
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "event_f1": r["event_f1"],
            "unmatched_each_way": r["unmatched_each_way"],
        }
        for r in worst[:15]
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with (OUT_DIR / "exact_count_event_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nwrote {OUT_DIR / 'exact_count_event_audit.csv'} ({len(rows_out)} rows)")
    print(f"wrote {OUT_DIR / 'summary.json'}")

    # ---- console summary --------------------------------------------------------
    print("\n=== per-arm ===")
    for arm in ARMS:
        s = summary["arms"].get(arm)
        if not s:
            continue
        f1 = s["exact_event_f1_summary"]
        print(
            f"{arm}: windows={s['windows']} exact={s['exact_windows']} "
            f"({s['exact_rate']:.1%}) | exact-but-dirty={s['exact_but_not_event_clean']} "
            f"({(s['exact_but_not_event_clean_rate'] or 0):.1%}) | "
            f"exact F1 med={f1.get('median', float('nan')):.3f} "
            f"min={s['exact_f1_min']:.3f} | "
            f"unmatched-each-way in exact windows={s['exact_unmatched_each_way_total']} "
            f"({s.get('exact_share_of_all_timing_error', 0):.1%} of all)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
