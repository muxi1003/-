"""Create source-linked tables, case plots and independent numeric checks."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import run_verification as run

HERE = Path(__file__).resolve().parent
METRIC_LABELS = {"rr_mae_bpm": "RR MAE (bpm)", "rr_rmse_bpm": "RR RMSE (bpm)", "rr_r2": "RR R2",
                 "mean_count_accuracy": "Mean relative accuracy", "exact_count_rate": "Exact count rate",
                 "exact_count": "Exact count windows", "event_f1": "Event F1", "tp": "Event TP", "fp": "Event FP", "fn": "Event FN"}
SCOPE_LABELS = {"anchored49": "Anchored complete (primary)", "P1": "P1", "P2": "P2", "P3": "P3", "all82": "TOTAL complete ~30s"}
COLORS = {"anchored49": "#287F68", "P1": "#346DB4", "P2": "#BD7934", "P3": "#B84457"}


def md_table(frame):
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if value is None or pd.isna(value):
                cells.append("NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def create_sheet(book, title, frame):
    sheet = book.create_sheet(title)
    sheet.append(list(frame.columns))
    for record in frame.itertuples(index=False, name=None):
        sheet.append([None if value is None or pd.isna(value) else value.item() if isinstance(value, np.generic) else value for value in record])
    sheet.freeze_panes = "D2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    for cell in sheet[1]:
        cell.font = Font(name="Microsoft YaHei", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="285B50")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[1].height = 54
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Microsoft YaHei", size=10)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if isinstance(cell.value, float):
                cell.number_format = "0.000000"
            if cell.row % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="F1F5F3")
    for index, name in enumerate(frame.columns, 1):
        sheet.column_dimensions[get_column_letter(index)].width = 24 if len(str(name)) < 25 else 36
    return sheet


def attribution_decomposition(predictions):
    errors = predictions.pivot(index="segment_id", columns="arm", values="absolute_rr_error_bpm")
    result = pd.DataFrame(index=errors.index)
    result["original_package_delta"] = errors.SELECT_LINEAR - errors.MAX_HALF
    result["selection_given_linear_delta"] = errors.SELECT_LINEAR - errors.MAX_LINEAR
    result["filling_given_max_delta"] = errors.MAX_LINEAR - errors.MAX_HALF
    result["selection_given_half_delta"] = errors.SELECT_HALF - errors.MAX_HALF
    result["filling_given_selection_delta"] = errors.SELECT_LINEAR - errors.SELECT_HALF
    result["interaction"] = result.selection_given_linear_delta - result.selection_given_half_delta
    residual = result.original_package_delta - result.selection_given_linear_delta - result.filling_given_max_delta
    assert np.allclose(residual, 0, atol=1e-12)
    return result.reset_index()


def validate_outputs(predictions, metrics, intervals):
    assert len(predictions) == 1312 and predictions.segment_id.nunique() == 82
    assert predictions.groupby("segment_id").arm.agg(set).map(lambda values: values == set(run.ARMS)).all()
    assert not predictions.duplicated(["segment_id", "arm"]).any()
    assert np.allclose(predictions.predicted_rr_bpm, 60 * predictions.predicted_count / predictions.duration_seconds)
    assert np.allclose(predictions.truth_rr_bpm, 60 * predictions.truth_count / predictions.duration_seconds)
    assert np.allclose(predictions.absolute_rr_error_bpm, abs(predictions.predicted_rr_bpm - predictions.truth_rr_bpm))
    checked_curves = 0
    for segment, group in predictions.groupby("segment_id"):
        curves = pd.read_csv(HERE / "curves" / f"{segment}.csv")
        for row in group.itertuples(index=False):
            indices = curves.loc[curves[f"{row.arm}_peak"].eq(1), "frame_index"].to_numpy(int)
            assert len(indices) == row.predicted_count
            values = curves[f"{row.arm}_fused"].to_numpy(float)
            assert np.isfinite(values).all()
            independent_smooth = np.convolve(values, np.ones(3) / 3, mode="valid")
            independent_peaks, _ = run.find_peaks(independent_smooth, distance=6, prominence=.05)
            assert np.array_equal(independent_peaks + 1, indices)
            checked_curves += 1
    matches = pd.read_csv(HERE / "event_matches.csv")
    tp = matches[matches.type.eq("TP")]
    assert tp.error_seconds.abs().le(.30 + 1e-9).all()
    assert not tp.duplicated(["segment_id", "arm", "predicted_index"]).any()
    assert not tp.duplicated(["segment_id", "arm", "reference_index"]).any()
    counts = matches.groupby(["segment_id", "arm", "type"]).size()
    for row in predictions.itertuples(index=False):
        for key in ("tp", "fp", "fn"):
            assert getattr(row, key) == counts.get((row.segment_id, row.arm, key.upper()), 0)
    for row in metrics.itertuples(index=False):
        group = run.select_scope(predictions, row.scope, row.split)
        group = group[group.arm.eq(row.arm)]
        y, p = group.truth_rr_bpm.to_numpy(), group.predicted_rr_bpm.to_numpy()
        errors = p - y
        assert np.isclose(np.abs(errors).mean(), row.rr_mae_bpm, atol=1e-12)
        assert np.isclose(np.sqrt(np.mean(errors ** 2)), row.rr_rmse_bpm, atol=1e-12)
        sst = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(errors ** 2) / sst if sst > 0 else np.nan
        assert np.isclose(r2, row.rr_r2, equal_nan=True, atol=1e-12)
        assert int((group.predicted_count == group.truth_count).sum()) == row.exact_count
        assert len(group) == row.scored_windows
    for row in intervals.itertuples(index=False):
        pool = run.select_scope(predictions, row.scope, row.split)
        rebuilt = run.paired_ci(pool[pool.arm.eq(row.candidate)], pool[pool.arm.eq(row.control)])
        for key in ("delta_mae_bpm", "ci95_low", "ci95_high"):
            assert np.isclose(rebuilt[key], getattr(row, key), rtol=0, atol=1e-12)
    run.verify_frozen()
    return {"status": "passed", "predictions": len(predictions), "curves_independently_recounted": checked_curves,
            "metrics_recomputed": len(metrics), "paired_intervals_rebuilt": len(intervals),
            "event_match_rows_checked": len(matches), "tp_one_to_one_and_within_tolerance": True,
            "original_source_hashes_unchanged": True}


def plot_attribution(frame):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, x, title in zip(axes, ["valid_ratio_difference", "bilateral_missing_ratio"],
                              ["Left/right valid-frame imbalance", "Bilateral missing-frame fraction"]):
        for cohort, group in frame.groupby("cohort", sort=False):
            axis.scatter(group[x], group.delta_mae_bpm, s=33, alpha=.8, color=COLORS[cohort], label=f"{cohort} (n={len(group)})")
        axis.axhline(0, color="#555555", linewidth=.8)
        axis.set(xlabel=title, ylabel="Selection minus linear-Max absolute RR error (bpm)", xlim=(-.02, 1.02))
        axis.grid(alpha=.17)
    axes[1].legend(fontsize=8)
    figure.suptitle("82 development windows: continuous covariates, no outcome-selected cutoffs", fontsize=11)
    figure.savefig(HERE / "figures/continuous_attribution.png", dpi=180)
    plt.close(figure)


def plot_arms(metrics):
    columns = []
    for scope, split, label in (("primary47", "all", "Primary47"), ("all82", "all", "All82"), ("all82", "val", "YOLO-val13")):
        series = metrics[metrics.scope.eq(scope) & metrics.split.eq(split)].set_index("arm").rr_mae_bpm.reindex(run.ARMS)
        columns.append(series.to_numpy())
    values = np.column_stack(columns)
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    image = axis.imshow(values, cmap="YlGnBu", aspect="auto")
    axis.set_yticks(range(16), run.ARMS, fontsize=8)
    axis.set_xticks(range(3), ["Primary47", "All82", "YOLO-val13"])
    for i in range(16):
        for j in range(3):
            axis.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=9,
                      color="white" if values[i, j] > .65 * values.max() else "#17211E")
    axis.set_title("RR MAE (bpm): lower is better; all are development analyses", fontsize=11)
    figure.colorbar(image, ax=axis, label="RR MAE (bpm)", shrink=.65)
    figure.savefig(HERE / "figures/arm_mae_matrix.png", dpi=180)
    plt.close(figure)


def plot_ablation(intervals):
    figure, axis = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    for offset, scope, split, color, label in ((-.22, "primary47", "all", "#287F68", "Primary47"),
                                              (0, "all82", "all", "#346DB4", "All82"),
                                              (.22, "all82", "val", "#BD7934", "YOLO-val13")):
        selected = intervals[intervals.scope.eq(scope) & intervals.split.eq(split)].set_index("contrast")
        values = selected.reindex([f"remove_{term}" for term in run.TERMS])
        y = values.delta_mae_bpm.to_numpy()
        x = np.arange(4) + offset
        axis.errorbar(x, y, yerr=[y-values.ci95_low.to_numpy(), values.ci95_high.to_numpy()-y], fmt="o", capsize=3, color=color, label=label)
    axis.axhline(0, color="#555555", linewidth=.8)
    axis.set_xticks(range(4), ["Remove prominence", "Remove amplitude", "Remove interval CV", "Remove missing penalty"], fontsize=9)
    axis.set(ylabel="Ablation minus full-selector MAE (bpm)", title="One-term removals: exploratory paired cluster intervals")
    axis.grid(axis="y", alpha=.17)
    axis.legend(fontsize=8)
    figure.savefig(HERE / "figures/score_term_ablation.png", dpi=180)
    plt.close(figure)


def plot_cases(cases, predictions):
    refs = run.read("reference_events.csv")
    sections = ["# 全部82窗中的六个极值病例", "在全部82窗中按SELECT_LINEAR减MAX_LINEAR逐窗绝对RR误差排序，改善与退化各3例；不是主要47窗各3例，也不是独立生理复核，人工事件沿用冻结版本。"]
    for row in cases.itertuples(index=False):
        curves = pd.read_csv(HERE / "curves" / f"{row.segment_id}.csv")
        group = predictions[predictions.segment_id.eq(row.segment_id)].set_index("arm")
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.3), sharex=True, constrained_layout=True)
        axes[0].plot(curves.time_seconds, curves.left_norm, label="Left normalized", color="#346DB4", linewidth=1.1)
        axes[0].plot(curves.time_seconds, curves.right_norm, label="Right normalized", color="#BD7934", linewidth=1.1)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set(ylabel="Normalized observed temperature", title=f"All82 extremes: {row.segment_id} | {row.case_type}")
        axes[0].title.set_fontsize(11)
        for arm, label, color, marker in (("MAX_LINEAR", "Fixed Max + linear", "#646B69", "x"),
                                          ("SELECT_LINEAR", f"Quality selected: {row.selected_mode}", "#287F68", "o")):
            axes[1].plot(curves.time_seconds, curves[f"{arm}_smooth"], color=color, label=f"{label}; N={int(group.loc[arm,'predicted_count'])}", linewidth=1.1)
            peaks = curves[f"{arm}_peak"].eq(1)
            axes[1].scatter(curves.loc[peaks, "time_seconds"], curves.loc[peaks, f"{arm}_smooth"], marker=marker, s=22, color=color)
        for time in refs[refs.segment_id.eq(row.segment_id)].time_seconds:
            axes[1].axvline(time, color="#B84457", alpha=.22, linewidth=.7)
        reference_n = int(group.iloc[0].truth_count)
        duration = group.iloc[0].duration_seconds
        axes[1].set(xlabel="Frozen annotation/playback time (s)", ylabel="Smoothed completed signal",
                    title=f"Reference N={reference_n}; T={duration:.4f}s; delta absolute RR error={row.delta_mae_bpm:+.4f} bpm")
        axes[1].legend(loc="upper right", fontsize=8)
        for axis in axes:
            axis.grid(alpha=.13)
        path = HERE / "figures/cases" / f"{row.segment_id}.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        sections += [f"\n## {row.segment_id}", f"- 类型：{row.case_type}；ΔMAE={row.delta_mae_bpm:+.6f}次/分；选择{row.selected_mode}。",
                     f"- 左/右有效比例{row.left_valid_ratio:.3f}/{row.right_valid_ratio:.3f}，双侧缺失{row.bilateral_missing_ratio:.3f}。",
                     f"![case]({path.as_posix()})"]
    (HERE / "CASES.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def main():
    if (HERE / "numeric_validation.json").exists():
        raise FileExistsError("Preserve prior report snapshot")
    predictions, metrics, intervals = (pd.read_csv(HERE / name) for name in ("predictions.csv", "metrics.csv", "paired_intervals.csv"))
    validation = validate_outputs(predictions, metrics, intervals)
    decomposition = attribution_decomposition(predictions)
    run.save_csv(HERE / "conditional_error_decomposition.csv", decomposition)
    book = Workbook()
    book.remove(book.active)
    notes = pd.DataFrame([
        ["Design", "82 complete ~30s development windows; primary47; train69 / YOLO-val13; no new labels or YOLO run."],
        ["Primary contrast", "SELECT_LINEAR minus MAX_LINEAR on primary47; negative MAE delta favors selection."],
        ["Accuracy", "Mean max(0,1-relative count error), not per-event detection accuracy; exact count rate separately shown."],
        ["Missing data", "LINEAR is attribution replay, not true respiratory observation; all sources/reference/durations unchanged."],
        ["Intervals", "Cow-group bootstrap 10000, seed20260929; exploratory, unadjusted; no new confirmation claim."],
        ["Workbook", "Read-only imported statistical snapshot, no live formulas; raw CSVs and source hashes included."],
        ["Ablations", "No retuning; term removals are diagnostics, not adopted replacements."],
    ], columns=["Item", "Definition"])
    create_sheet(book, "说明", notes).column_dimensions["B"].width = 115
    for split in run.SPLITS:
        rows = []
        for key, label in METRIC_LABELS.items():
            for scope in ("anchored49", "P1", "P2", "P3", "all82"):
                pool = metrics[metrics.scope.eq(scope) & metrics.split.eq(split)].set_index("arm")
                records = {"Metric": label, "Scope": SCOPE_LABELS[scope], "n": int(pool.iloc[0].scored_windows) if len(pool) else 0}
                records.update({arm: pool.loc[arm, key] if len(pool) else None for arm in run.ARMS})
                rows.append(records)
        frame = pd.DataFrame(rows)
        run.save_csv(HERE / f"by_metric_{split}.csv", frame)
        sheet = create_sheet(book, f"指标_{split}", frame)
        for number, row in enumerate(rows, 2):
            if row["Metric"] in ("Mean relative accuracy", "Exact count rate"):
                for cell in sheet[number][3:]:
                    cell.number_format = "0.00%"
            if row["Scope"].startswith("TOTAL"):
                for cell in sheet[number]:
                    cell.fill = PatternFill("solid", fgColor="DCEBE6")
    for title, frame in (("原始汇总", metrics), ("全部配对区间", intervals), ("逐窗16组预测", predictions),
                         ("候选质量", pd.read_csv(HERE / "candidate_quality.csv")),
                         ("条件差值分解", decomposition), ("病例", pd.read_csv(HERE / "failure_cases.csv"))):
        create_sheet(book, title, frame)
    frozen = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    create_sheet(book, "来源哈希", pd.DataFrame(frozen["sources"]))
    book.save(HERE / "metric_comparison.xlsx")
    (HERE / "figures").mkdir(exist_ok=True)
    plot_attribution(pd.read_csv(HERE / "attribution_by_window.csv"))
    plot_arms(metrics)
    plot_ablation(intervals)
    plot_cases(pd.read_csv(HERE / "failure_cases.csv"), predictions)
    checked = load_workbook(HERE / "metric_comparison.xlsx", data_only=False)
    for sheet in checked:
        for row in sheet:
            for cell in row:
                assert cell.data_type not in ("f", "e")
    validation.update(workbook_sheets=checked.sheetnames, figures=len(list((HERE / "figures").rglob("*.png"))),
                      conditional_decomposition_identity_passed=True, workbook_error_cells=0)
    run.write_json(HERE / "numeric_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
