"""Save computed review evidence and deterministic source-existence checks."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

import run_verification as run
from build_reports import md_table

HERE = Path(__file__).resolve().parent


def main():
    if (HERE / "REVIEW_EVIDENCE.json").exists():
        raise FileExistsError("Preserve review evidence")
    metrics = pd.read_csv(HERE / "metrics.csv")
    paired = pd.read_csv(HERE / "paired_intervals.csv")
    global_mode = json.loads((HERE / "global_fixed_mode.json").read_text(encoding="utf-8"))
    def metric(scope, split, arm):
        return metrics[metrics.scope.eq(scope) & metrics.split.eq(split) & metrics.arm.eq(arm)].iloc[0].to_dict()
    def contrast(scope, split, name):
        return paired[paired.scope.eq(scope) & paired.split.eq(split) & paired.contrast.eq(name)].iloc[0].to_dict()
    evidence = {"classification": "development_attribution_only", "client_date": "2026-10-19", "primary_windows": 47, "all_windows": 82,
                "primary_control": metric("primary47", "all", "MAX_LINEAR"),
                "primary_selection": metric("primary47", "all", "SELECT_LINEAR"),
                "primary_contrast": contrast("primary47", "all", "selection_given_linear_primary"),
                "all_control": metric("all82", "all", "MAX_LINEAR"), "all_selection": metric("all82", "all", "SELECT_LINEAR"),
                "all_contrast": contrast("all82", "all", "selection_given_linear_primary"),
                "primary_filling": contrast("primary47", "all", "filling_given_max"),
                "all_filling": contrast("all82", "all", "filling_given_max"),
                "original_package": contrast("all82", "all", "original_package"),
                "fixed_comparator": global_mode,
                "all_amplitude_ablation": metric("all82", "all", "SELECT_LINEAR_NO_AMPLITUDE"),
                "do_not_adopt_ablation_as_new_method": True}
    attribution = pd.read_csv(HERE / "attribution_by_window.csv")
    main = attribution[attribution.cohort.eq("anchored49")]
    best = main.sort_values(["delta_mae_bpm", "segment_id"]).iloc[0]
    evidence["case_concentration"] = {"largest_primary_improvement_segment": best.segment_id,
        "largest_case_delta_bpm": float(best.delta_mae_bpm), "primary_total_delta_sum_bpm": float(main.delta_mae_bpm.sum()),
        "other_46_delta_sum_bpm": float(main.delta_mae_bpm.sum() - best.delta_mae_bpm),
        "interpretation": "Error-sum diagnostic only; no window excluded or primary endpoint changed"}
    evidence["numeric_gates"] = {
        "primary_ci_upper_below_zero": bool(evidence["primary_contrast"]["ci95_high"] < 0),
        "all82_direction_not_reversed": bool(evidence["all_contrast"]["delta_mae_bpm"] <= 0),
        "val13_not_worse_than_train_selected_fixed": bool(global_mode["validation_selector"]["rr_mae_bpm"] <= global_mode["validation_fixed"]["rr_mae_bpm"]),
        "stop_both_primary_and_all_no_improvement": bool(evidence["primary_contrast"]["delta_mae_bpm"] >= 0 and evidence["all_contrast"]["delta_mae_bpm"] >= 0),
    }
    run.write_json(HERE / "REVIEW_EVIDENCE.json", evidence)
    claims = [{"id": "primary_selection_mae", "value": str(evidence["primary_selection"]["rr_mae_bpm"]), "source": "REVIEW_EVIDENCE.json"},
              {"id": "primary_ci_upper", "value": str(evidence["primary_contrast"]["ci95_high"]), "source": "REVIEW_EVIDENCE.json"},
              {"id": "all_pure_selection_delta", "value": str(evidence["all_contrast"]["delta_mae_bpm"]), "source": "REVIEW_EVIDENCE.json"},
              {"id": "fixed_mode", "value": global_mode["chosen_arm"], "source": "REVIEW_EVIDENCE.json"},
              {"id": "filling_not_entire_gain", "value": str(evidence["primary_filling"]["delta_mae_bpm"]), "source": "REVIEW_EVIDENCE.json"}]
    run.write_json(HERE / "claim_evidence_inputs.json", claims)
    checker = run.REPO / ".aris/upstream/aris_repo/tools/evidence_check.py"
    spec = importlib.util.spec_from_file_location("quality_attribution_evidence_checker", checker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = module.check_batch(claims, root=str(HERE))
    run.write_json(HERE / "evidence_precheck.json", checks)
    literature = pd.DataFrame(json.loads((HERE / "literature_comparison.json").read_text(encoding="utf-8")))
    run.save_csv(HERE / "literature_comparison.csv", literature)
    lines = ["# 六篇原始研究的针对性对照", "这是有限范围的方法差异核查，不是穷尽性系统综述或独立创新证明。五篇全文已取得，一篇仅公开摘要/方法索引，访问限制已保留。"]
    for record in literature.to_dict("records"):
        lines += [f"\n## {record['title']} ({record['year']})", f"来源：[{record['venue']}]({record['url']})；读取等级：{record['read_level']}。",
                  f"- 候选：{record['candidates']}。", f"- 评分：{record['quality_basis']}。", f"- 决策与验证：{record['decision_unit']}；{record['validation']}。",
                  f"- 本项目差异：{record['project_delta']}。", f"- 不能推出：{record['novelty_limit']}。", f"- 定位：{record['evidence_location']}。"]
    lines += ["\n## 综合定位", "不能以质量选曲概念或更换YOLO独占新颖性。可检验的窄差异是五条双鼻孔候选的时域/缺失联合决策及补齐控制归因；这不是已获确认的有效性或领域首次结论。最近半年预印本外围命中另见literature/SEARCH_LOG.md。"]
    (HERE / "LITERATURE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_precheck": checks.get("summary"), "gates": evidence["numeric_gates"], "case_concentration": evidence["case_concentration"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
