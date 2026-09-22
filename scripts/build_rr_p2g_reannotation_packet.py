from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_rr_external_breath_annotation_packet import build_worklist, html_document


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    assets = (
        repo_root
        / "Dataset_new"
        / "72video"
        / "al_images"
        / "paper_repro_quality_residual_paper_assets"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build blinded P0/P1 reannotation packets for high-impact calibration-free "
            "thermal-color external clips."
        )
    )
    parser.add_argument("--assets-dir", type=Path, default=assets)
    parser.add_argument(
        "--scope",
        choices=["priority_p0_p1", "primary_all", "extension_all"],
        default="priority_p0_p1",
        help=(
            "priority_p0_p1 creates the 35-clip high-impact label-reliability queue; "
            "primary_all creates the complete 94-clip P2g confirmatory annotation scope; "
            "extension_all creates the pre-specified duration-eligible external extension pool."
        ),
    )
    parser.add_argument(
        "--priorities",
        default="P0_dual_blinded_recount_and_adjudication,P1_second_independent_recount",
    )
    parser.add_argument("--annotator-a-id", default="P2G_A")
    parser.add_argument("--annotator-b-id", default="P2G_B")
    parser.add_argument("--seed-a", type=int, default=20260713)
    parser.add_argument("--seed-b", type=int, default=20260714)
    parser.add_argument("--extension-min-duration-seconds", type=float, default=20.0)
    return parser.parse_args()


def write_packet(
    output_dir: Path,
    fieldwork: pd.DataFrame,
    inventory: pd.DataFrame | None,
    priority: pd.DataFrame,
    annotator_id: str,
    seed: int,
    packet_tag: str,
    packet_title: str,
) -> dict[str, object]:
    worklist = build_worklist(
        fieldwork,
        inventory,
        blind=True,
        blind_seed=seed,
        prefill_existing_labels=False,
    )
    suffix = annotator_id.lower()
    worklist_path = output_dir / f"paper_p2g_{packet_tag}_worklist_{suffix}.csv"
    html_path = output_dir / f"paper_p2g_{packet_tag}_packet_{suffix}.html"
    mapping_path = output_dir / f"paper_p2g_{packet_tag}_blind_mapping_{suffix}.csv"
    report_path = output_dir / f"paper_p2g_{packet_tag}_packet_{suffix}.md"
    export_name = f"paper_p2g_{packet_tag}_{suffix}_export.csv"
    worklist.to_csv(worklist_path, index=False)
    mapping = worklist[["external_video_id", "blinded_id", "display_order"]].merge(
        priority,
        left_on="external_video_id",
        right_on="video_id",
        how="left",
        validate="one_to_one",
    )
    mapping.to_csv(mapping_path, index=False)
    html_path.write_text(
        html_document(
            worklist,
            f"p2g_{packet_tag}_{suffix}_blind_{seed}",
            export_name,
            annotator_id,
            True,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(
            [
                f"# {packet_title}",
                "",
                f"Annotator: `{annotator_id}`",
                "",
                f"Blind seed: `{seed}`",
                "",
                f"Clips: `{len(worklist)}`",
                "",
                "The HTML packet intentionally hides source identifiers, existing manual "
                "counts, model predictions, and priority labels. Export the completed CSV "
                "without manually editing identifiers. The blind-mapping CSV is for the "
                "coordinator only and must not be provided to the annotator.",
                "",
                f"HTML packet: `{html_path}`",
                f"Annotator worklist: `{worklist_path}`",
                f"Coordinator blind mapping: `{mapping_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "annotator_id": annotator_id,
        "seed": seed,
        "clips": len(worklist),
        "html_packet": str(html_path),
        "worklist": str(worklist_path),
        "blind_mapping": str(mapping_path),
        "report": str(report_path),
    }


def main() -> None:
    args = parse_args()
    assets = args.assets_dir.resolve()
    priority_path = assets / "paper_calibration_free_thermal_index_reannotation_priority.csv"
    fieldwork_path = assets / "paper_external_validation_split_all_use_fieldwork_template.csv"
    inventory_path = assets / "paper_external_validation_split_all_use_inventory.csv"
    reference_path = assets / "paper_external_validation_single_reference_rows.csv"
    priority = pd.read_csv(priority_path)
    reference = pd.read_csv(reference_path, dtype=str, keep_default_na=False)
    fieldwork = pd.read_csv(fieldwork_path, dtype=str, keep_default_na=False)
    inventory = (
        pd.read_csv(inventory_path, dtype=str, keep_default_na=False)
        if inventory_path.exists()
        else None
    )
    if args.scope == "priority_p0_p1":
        priorities = {item.strip() for item in args.priorities.split(",") if item.strip()}
        priority = priority[priority["review_priority"].astype(str).isin(priorities)].copy()
        packet_tag = "reannotation"
        packet_title = "P2G Priority Reannotation Packet"
        scope_text = "P0/P1 clips. This is a label-reliability subset, not the complete 94-clip confirmatory external validation scope."
        score_scope = "priority_p0_p1"
    elif args.scope == "primary_all":
        primary_ids = set(
            reference.loc[
                reference["primary_analysis_include"].astype(str).str.lower().eq("true"),
                "external_video_id",
            ].astype(str)
        )
        priority = priority[priority["video_id"].astype(str).isin(primary_ids)].copy()
        packet_tag = "primary_all"
        packet_title = "P2G Complete Primary Blinded Annotation Packet"
        scope_text = "all 94 primary P2g clips. This is the complete external-consensus annotation scope."
        score_scope = "primary_all"
    else:
        if inventory is None:
            raise FileNotFoundError(f"Missing extension inventory: {inventory_path}")
        primary_ids = set(
            reference.loc[
                reference["primary_analysis_include"].astype(str).str.lower().eq("true"),
                "external_video_id",
            ].astype(str)
        )
        duration = pd.to_numeric(inventory["clip_duration_seconds"], errors="coerce")
        scoreable = inventory["clip_duration_is_scoreable"].astype(str).str.lower().eq("true")
        extension_ids = inventory.loc[
            scoreable
            & duration.ge(float(args.extension_min_duration_seconds))
            & ~inventory["external_video_id"].astype(str).isin(primary_ids),
            "external_video_id",
        ].astype(str)
        priority = pd.DataFrame(
            {
                "video_id": extension_ids,
                "review_priority": "E0_pre_registered_duration_eligible_extension",
                "session_abs_rr_error": 0.0,
                "abs_rr_error_bpm": 0.0,
            }
        )
        packet_tag = "extension_all"
        packet_title = "P2G Pre-Registered External Extension Annotation Packet"
        scope_text = (
            "all duration-eligible non-primary external clips with available raw video. "
            "This pool is fixed by source availability and duration only; it must not be "
            "filtered by model prediction, error, or newly entered breath count."
        )
        score_scope = "extension_all"
    priority = priority.drop_duplicates("video_id").sort_values(
        ["review_priority", "session_abs_rr_error", "abs_rr_error_bpm"],
        ascending=[True, False, False],
    )
    fieldwork = fieldwork[fieldwork["external_video_id"].astype(str).isin(priority["video_id"].astype(str))]
    if len(fieldwork) != len(priority):
        missing = sorted(set(priority["video_id"].astype(str)) - set(fieldwork["external_video_id"].astype(str)))
        raise ValueError(f"Priority clips missing from fieldwork template: {missing}")
    fieldwork_subset = fieldwork.copy()
    fieldwork_subset["include_in_external_validation"] = "yes"
    if args.scope == "extension_all":
        duration_map = inventory.set_index("external_video_id")["clip_duration_seconds"]
        fieldwork_subset["manual_breath_count"] = ""
        fieldwork_subset["manual_rr_bpm"] = ""
        fieldwork_subset["reference_rr_annotator"] = ""
        fieldwork_subset["reference_count_uncertainty_breaths"] = ""
        fieldwork_subset["reference_rr_lower_bpm"] = ""
        fieldwork_subset["reference_rr_upper_bpm"] = ""
        fieldwork_subset["manual_duration_seconds"] = fieldwork_subset["external_video_id"].map(
            duration_map
        )
        fieldwork_subset["reference_protocol_notes"] = (
            "Pre-registered P2g extension candidate. Recount independently; "
            "historical count is intentionally cleared."
        )
    fieldwork_subset_path = assets / f"paper_p2g_{packet_tag}_fieldwork_subset.csv"
    fieldwork_subset.to_csv(fieldwork_subset_path, index=False)
    packets = [
        write_packet(
            assets,
            fieldwork_subset,
            inventory,
            priority,
            args.annotator_a_id,
            args.seed_a,
            packet_tag,
            packet_title,
        ),
        write_packet(
            assets,
            fieldwork_subset,
            inventory,
            priority,
            args.annotator_b_id,
            args.seed_b,
            packet_tag,
            packet_title,
        ),
    ]
    summary = pd.DataFrame(packets)
    summary_path = assets / f"paper_p2g_{packet_tag}_packet_summary.csv"
    summary.to_csv(summary_path, index=False)
    consensus_dir = assets / f"paper_p2g_{packet_tag}_consensus"
    plan_path = assets / f"paper_p2g_{packet_tag}_scoring_plan.md"
    plan_path.write_text(
        "\n".join(
            [
                f"# {packet_title} Scoring Plan",
                "",
                f"Scope: `{len(fieldwork_subset)}` {scope_text}",
                "",
                "1. Keep the two annotator exports separate and do not give them either blind mapping.",
                "2. For every clip, record visibility as clear, uncertain, or unreadable and record "
                "the count uncertainty when known. Unreadable clips remain in the flow table but do not enter RR scoring.",
                "3. Run `audit_rr_external_breath_annotation_agreement.py` with the fieldwork subset, "
                f"A/B exports, and an output directory of `{consensus_dir.name}`.",
                (
                    "4. Resolve every row in the generated adjudication CSV, then rerun the audit until "
                    f"`consensus_ready_rows` equals the {len(fieldwork_subset)} included rows. A primary-scope "
                    "unreadable clip is a failed confirmatory gate and cannot be silently omitted."
                    if score_scope != "extension_all"
                    else "4. Resolve every disagreement in the generated adjudication CSV. Keep every unreadable "
                    "clip in the flow table; at least 10 extension clips must remain consensus-ready for the "
                    "combined 104-clip target."
                ),
                (
                    f"5. Run `score_rr_p2g_blinded_consensus.py --scope {score_scope}` with the consensus CSV. "
                    "The scorer only reuses frozen P2g predictions and records hashes of every input."
                    if score_scope != "extension_all"
                    else (
                        "5. The extension P2g predictions are frozen in "
                        "`paper_p2g_primary_plus_extension_frozen_predictions.csv`. After primary and "
                        "extension consensus are merged, run `score_rr_p2g_blinded_consensus.py "
                        "--scope primary_plus_extension --p2g-predictions-csv "
                        "paper_p2g_primary_plus_extension_frozen_predictions.csv`. Keep an explicit "
                        "flow table for every pre-registered clip that remains unreadable."
                    )
                ),
                "",
                f"Fieldwork subset: `{fieldwork_subset_path}`",
                f"Consensus output directory: `{consensus_dir}`",
                f"P2g A worklist: `{packets[0]['worklist']}`",
                f"P2g B worklist: `{packets[1]['worklist']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved P2G {args.scope} packet summary: {summary_path}")
    print(f"Saved P2G {args.scope} fieldwork subset: {fieldwork_subset_path}")
    print(f"Saved P2G {args.scope} scoring plan: {plan_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
