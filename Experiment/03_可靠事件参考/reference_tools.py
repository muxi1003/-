"""Validate human event references and score one-to-one respiratory events."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common import *

TOLERANCE_SECONDS = 0.30


def match_events(predicted, reference, tolerance=TOLERANCE_SECONDS):
    """Maximum-cardinality, then minimum-time-error bipartite assignment."""
    p, r = np.asarray(predicted, float), np.asarray(reference, float)
    if tolerance <= 0 or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("Invalid event times or tolerance")
    n, m = len(p), len(r)
    if not n or not m:
        return [], list(range(n)), list(range(m))
    distances = abs(p[:, None] - r[None, :])
    cost = np.zeros((n + m, n + m))
    cost[:n, :m] = np.where(distances <= tolerance + 1e-9,
                            distances / tolerance / (n + m + 1), 1e6)
    cost[:n, m:] = 1.
    cost[n:, :m] = 1.
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(i), int(j), float(distances[i, j])) for i, j in zip(rows, cols)
               if i < n and j < m and distances[i, j] <= tolerance + 1e-9]
    used_p, used_r = {i for i, _, _ in matches}, {j for _, j, _ in matches}
    return matches, [i for i in range(n) if i not in used_p], [j for j in range(m) if j not in used_r]


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def validate_tables(windows, events, intervals):
    errors = []
    keys = ["window_id", "annotation_round"]
    if windows.duplicated(keys).any():
        errors.append("duplicate_window_round")
    records = {(r.window_id, r.annotation_round): r for r in windows.itertuples()}
    valid_status = {"pending", "complete", "partial", "unobservable"}
    for row in windows.itertuples():
        key = (row.window_id, row.annotation_round)
        duration = number(row.duration_seconds)
        if not np.isfinite(duration) or duration <= 0:
            errors.append(f"{key}:invalid_duration")
        if row.annotation_status not in valid_status:
            errors.append(f"{key}:invalid_annotation_status")
        if row.annotation_status in {"complete", "partial"}:
            count = number(row.manual_breath_count)
            if not np.isfinite(count) or count < 0 or not count.is_integer():
                errors.append(f"{key}:explicit_integer_manual_count_required")
            if not row.annotator.strip() or not is_true(row.predictions_hidden):
                errors.append(f"{key}:annotator_and_prediction_hidden_confirmation_required")
            if row.event_definition != "expiration_peak":
                errors.append(f"{key}:event_phase_must_match_protocol")
            accepted = events[(events.window_id == row.window_id) & (events.annotation_round == row.annotation_round) &
                              (events.confidence == "confirmed")]
            uncertain = events[(events.window_id == row.window_id) & (events.annotation_round == row.annotation_round) &
                               (events.confidence != "confirmed")]
            if np.isfinite(count) and len(accepted) != count:
                errors.append(f"{key}:count_event_mismatch")
            if row.annotation_status == "complete" and len(uncertain):
                errors.append(f"{key}:uncertain_events_not_complete_reference")
    if events.duplicated(keys + ["event_id"]).any():
        errors.append("duplicate_event_id")
    if events.duplicated(keys + ["event_time_seconds"]).any():
        errors.append("duplicate_event_time")
    for row in events.itertuples():
        key = (row.window_id, row.annotation_round)
        window = records.get(key)
        t = number(row.event_time_seconds)
        if window is None:
            errors.append(f"{key}:unknown_window")
            continue
        if not np.isfinite(t) or not 0 <= t < number(window.duration_seconds):
            errors.append(f"{key}/{row.event_id}:time_outside_half_open_window")
        if not row.event_id.strip() or not row.annotator.strip():
            errors.append(f"{key}:missing_event_id_or_annotator")
        if row.confidence not in {"confirmed", "uncertain"} or row.event_type != "expiration_peak":
            errors.append(f"{key}/{row.event_id}:invalid_confidence_or_phase")
        if row.event_start_seconds or row.event_end_seconds:
            a, b = number(row.event_start_seconds), number(row.event_end_seconds)
            if not np.isfinite([a, b]).all() or not 0 <= a <= t <= b <= number(window.duration_seconds):
                errors.append(f"{key}/{row.event_id}:invalid_event_interval")
    for key, group in intervals.groupby(keys):
        window = records.get(key)
        if window is None:
            errors.append(f"{key}:unknown_interval_window")
            continue
        spans = []
        for row in group.itertuples():
            a, b = number(row.start_seconds), number(row.end_seconds)
            if not np.isfinite([a, b]).all() or not 0 <= a < b <= number(window.duration_seconds):
                errors.append(f"{key}:invalid_unobservable_interval")
            if not row.reason.strip() or not row.annotator.strip():
                errors.append(f"{key}:interval_reason_and_annotator_required")
            spans.append((a, b))
        spans.sort()
        if any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
            errors.append(f"{key}:overlapping_unobservable_intervals")
        if window.annotation_status == "complete":
            errors.append(f"{key}:complete_reference_cannot_contain_unobservable_intervals")
        confirmed = events[(events.window_id == key[0]) & (events.annotation_round == key[1]) &
                           (events.confidence == "confirmed")]
        if any(a <= number(t) < b for a, b in spans for t in confirmed.event_time_seconds):
            errors.append(f"{key}:confirmed_event_inside_unobservable_interval")
    return sorted(set(errors))


def load_reference(directory):
    return (read_csv(directory / "annotation_windows.csv"), read_csv(directory / "reference_events.csv"),
            read_csv(directory / "unobservable_intervals.csv"))


def score(directory: Path, predicted_path: Path, prediction_windows_path: Path, out: Path, annotation_round: str):
    windows, events, intervals = load_reference(directory)
    errors = validate_tables(windows, events, intervals)
    if errors:
        raise ValueError("Invalid human reference: " + "; ".join(errors[:10]))
    windows = windows[windows.annotation_round == annotation_round]
    complete = windows[windows.annotation_status == "complete"]
    if complete.empty:
        raise ValueError("No complete human event reference; blank forms are not zero-breath truth")
    prediction_windows = read_csv(prediction_windows_path)
    predicted = read_csv(predicted_path)
    if prediction_windows.window_id.duplicated().any() or predicted.duplicated(["window_id", "event_id"]).any():
        raise ValueError("Duplicate prediction window/event")
    if set(prediction_windows.window_id) != set(windows.window_id):
        raise ValueError("Prediction coverage must list every reference window, including abstentions")
    if not set(predicted.window_id).issubset(set(windows.window_id)):
        raise ValueError("Unknown prediction event window")
    rows, details = [], []
    predicted_events_total = 0
    for window in complete.itertuples():
        metadata = prediction_windows.set_index("window_id").loc[window.window_id]
        if not is_true(metadata.timebase_verified) or metadata.timebase != "annotation_video_seconds":
            raise ValueError("Actual annotation-video time alignment required; nominal frame/8.7 is insufficient")
        if metadata.prediction_status not in {"ok", "abstain"}:
            raise ValueError("Invalid prediction status")
        predicted_duration = number(metadata.duration_seconds)
        if not np.isfinite(predicted_duration) or abs(predicted_duration - float(window.duration_seconds)) > 1e-6:
            raise ValueError("Prediction/reference window durations differ")
        pp = predicted[predicted.window_id == window.window_id]
        if len(pp) and metadata.prediction_status == "abstain":
            raise ValueError("Abstention cannot contain predicted events")
        if "predicted_count" in metadata.index:
            count = number(metadata.predicted_count)
            if metadata.prediction_status == "abstain":
                if str(metadata.predicted_count).strip():
                    raise ValueError("Abstention cannot declare a count")
            elif not np.isfinite(count) or not count.is_integer() or count != len(pp):
                raise ValueError("Predicted count must equal event rows")
        r = events[(events.window_id == window.window_id) & (events.annotation_round == annotation_round) &
                   (events.confidence == "confirmed")]
        pt, rt = pp.event_time_seconds.to_numpy(float), r.event_time_seconds.to_numpy(float)
        if not np.isfinite(pt).all() or np.any((pt < 0) | (pt >= float(window.duration_seconds))):
            raise ValueError("Predicted event outside its window")
        matched, false_p, false_n = match_events(pt, rt)
        predicted_events_total += len(pt)
        rows.append({"window_id": window.window_id, "prediction_status": metadata.prediction_status,
                     "tp": len(matched), "fp": len(false_p), "fn": len(false_n),
                     "predicted_count": len(pt) if metadata.prediction_status == "ok" else "",
                     "truth_count": int(window.manual_breath_count),
                     "mean_matched_time_error_seconds": float(np.mean([d for _, _, d in matched])) if matched else ""})
        for i, j, d in matched:
            details.append({"window_id": window.window_id, "type": "TP", "predicted_event_id": pp.iloc[i].event_id,
                            "reference_event_id": r.iloc[j].event_id, "time_error_seconds": d})
        for indices, label, table in ((false_p, "FP", pp), (false_n, "FN", r)):
            for index in indices:
                details.append({"window_id": window.window_id, "type": label,
                                "predicted_event_id": table.iloc[index].event_id if label == "FP" else "",
                                "reference_event_id": table.iloc[index].event_id if label == "FN" else "",
                                "time_error_seconds": ""})
    result = pd.DataFrame(rows)
    tp, fp, fn = [int(result[k].sum()) for k in ("tp", "fp", "fn")]
    write_csv(out / "event_metrics_by_window.csv", result)
    write_csv(out / "event_matches.csv", details, columns=["window_id", "type", "predicted_event_id", "reference_event_id", "time_error_seconds"])
    write_json(out / "event_metrics.json", {
        "status": "SCORED_HUMAN_EVENT_REFERENCE", "reference_round": annotation_round,
        "matching_rule": "one_to_one_max_cardinality_then_min_time_error", "tolerance_seconds": TOLERANCE_SECONDS,
        "all_annotation_windows": len(windows), "complete_reference_windows": len(complete),
        "reference_window_coverage": len(complete) / len(windows),
        "algorithm_output_coverage_on_complete": float(result.prediction_status.eq("ok").mean()),
        "tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "abstention_policy": "reference events in abstained complete windows count as FN; coverage reported",
        "partial_reference_policy": "excluded from complete-window primary score; excluded windows explicitly listed",
        "input_hashes": {str(p): sha256(p) for p in [directory / "annotation_windows.csv", directory / "reference_events.csv",
                                                    directory / "unobservable_intervals.csv", predicted_path, prediction_windows_path]}})
    write_csv(out / "excluded_reference_windows.csv", windows[windows.annotation_status != "complete"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "score"])
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction-events", type=Path)
    parser.add_argument("--prediction-windows", type=Path)
    parser.add_argument("--round", default="R1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.command == "score":
        if not args.prediction_events or not args.prediction_windows:
            parser.error("score requires prediction events and window status files")
        score(args.annotations, args.prediction_events, args.prediction_windows, args.out, args.round)
        return
    windows, events, intervals = load_reference(args.annotations)
    errors = validate_tables(windows, events, intervals)
    completed = int(windows.annotation_status.eq("complete").sum())
    write_json(args.out / "reference_validation.json", {
        "status": "INVALID" if errors else "COMPLETE_REFERENCE_AVAILABLE" if completed else "AWAIT_HUMAN_EVENTS",
        "windows": len(windows), "complete_windows": completed, "event_rows": len(events), "errors": errors,
        "event_accuracy_metrics": None, "blank_manual_counts_are_not_zero": True})
    print(f"Reference windows={len(windows)}, complete={completed}, events={len(events)}, errors={len(errors)}")
    if errors:
        raise SystemExit("Invalid reference tables")


if __name__ == "__main__":
    main()
