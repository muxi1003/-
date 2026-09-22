from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

import pandas as pd


WORKLIST_COLUMNS = [
    "external_video_id",
    "blinded_id",
    "display_order",
    "source_session_id",
    "clip_index",
    "cow_id",
    "collection_date",
    "collection_time",
    "manual_duration_seconds",
    "raw_video_path",
    "video_file_uri",
    "manual_breath_count",
    "manual_rr_bpm",
    "reference_rr_annotator",
    "camera_id",
    "annotation_visibility",
    "count_uncertainty_breaths",
    "annotation_notes",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description=(
            "Build a local HTML packet for manual breath-count annotation of "
            "included external RR clips."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--fieldwork-csv", type=Path, default=None)
    parser.add_argument("--inventory-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--packet-id", default="split_all_use_external_rr_v1")
    parser.add_argument("--annotator-id", default="")
    parser.add_argument("--export-name", default="")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Randomize display order and hide video/cow/date/session identifiers in the HTML interface.",
    )
    parser.add_argument(
        "--blind-seed",
        type=int,
        default=20260710,
        help="Deterministic seed used when --blind randomizes the packet order.",
    )
    parser.add_argument(
        "--prefill-existing-labels",
        action="store_true",
        help="Prefill manual breath-count fields from the source worksheet. Disabled by default for blind annotation.",
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir or args.input_root / f"{args.corrected_prefix}_paper_assets"


def default_fieldwork_csv(args: argparse.Namespace, output_dir: Path) -> Path:
    return output_dir / "paper_external_validation_split_all_use_fieldwork_template.csv"


def default_inventory_csv(args: argparse.Namespace, output_dir: Path) -> Path:
    return output_dir / "paper_external_validation_split_all_use_inventory.csv"


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def yes_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        ["yes", "y", "true", "1", "include", "included"]
    )


def video_uri(raw_path: object) -> str:
    text = normalize_text(raw_path)
    if not text:
        return ""
    path = Path(text)
    try:
        return path.resolve().as_uri()
    except ValueError:
        return ""


def build_worklist(
    fieldwork: pd.DataFrame,
    inventory: pd.DataFrame | None,
    blind: bool = False,
    blind_seed: int = 20260710,
    prefill_existing_labels: bool = False,
) -> pd.DataFrame:
    included = fieldwork[
        yes_mask(fieldwork.get("include_in_external_validation", pd.Series("", index=fieldwork.index)))
    ].copy()
    if inventory is not None and not inventory.empty and "external_video_id" in inventory.columns:
        keep = [
            column
            for column in [
                "external_video_id",
                "source_session_id",
                "clip_index",
                "clip_duration_seconds",
                "clip_duration_screen_reason",
            ]
            if column in inventory.columns
        ]
        included = included.merge(
            inventory[keep],
            on="external_video_id",
            how="left",
            suffixes=("", "_inventory"),
        )
    else:
        included["source_session_id"] = ""
        included["clip_index"] = ""

    rows: list[dict[str, object]] = []
    for _, row in included.iterrows():
        duration = normalize_text(row.get("manual_duration_seconds", ""))
        item = {
            "external_video_id": normalize_text(row.get("external_video_id", "")),
            "blinded_id": "",
            "display_order": "",
            "source_session_id": normalize_text(row.get("source_session_id", "")),
            "clip_index": normalize_text(row.get("clip_index", "")),
            "cow_id": normalize_text(row.get("cow_id", "")),
            "collection_date": normalize_text(row.get("collection_date", "")),
            "collection_time": normalize_text(row.get("collection_time", "")),
            "manual_duration_seconds": duration,
            "raw_video_path": normalize_text(row.get("raw_video_path", "")),
            "video_file_uri": video_uri(row.get("raw_video_path", "")),
            "manual_breath_count": normalize_text(row.get("manual_breath_count", "")) if prefill_existing_labels else "",
            "manual_rr_bpm": normalize_text(row.get("manual_rr_bpm", "")) if prefill_existing_labels else "",
            "reference_rr_annotator": normalize_text(row.get("reference_rr_annotator", "")) if prefill_existing_labels else "",
            "camera_id": normalize_text(row.get("camera_id", "")),
            "annotation_visibility": "",
            "count_uncertainty_breaths": "",
            "annotation_notes": "",
        }
        rows.append(item)
    worklist = pd.DataFrame(rows, columns=WORKLIST_COLUMNS)
    sort_columns = [
        column
        for column in ["collection_date", "collection_time", "source_session_id", "clip_index"]
        if column in worklist.columns
    ]
    if sort_columns:
        worklist = worklist.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if blind and not worklist.empty:
        order = list(range(len(worklist)))
        random.Random(blind_seed).shuffle(order)
        worklist = worklist.iloc[order].reset_index(drop=True)
    worklist["display_order"] = [str(index + 1) for index in range(len(worklist))]
    worklist["blinded_id"] = [
        f"BLIND_{index + 1:04d}" if blind else normalize_text(video_id)
        for index, video_id in enumerate(worklist["external_video_id"].tolist())
    ]
    worklist = worklist[WORKLIST_COLUMNS]
    return worklist


def html_document(
    worklist: pd.DataFrame,
    packet_id: str,
    export_name: str,
    annotator_id: str = "",
    blind_mode: bool = False,
) -> str:
    rows = worklist.fillna("").astype(str).to_dict(orient="records")
    rows_json = json.dumps(rows, ensure_ascii=False)
    packet_id_json = json.dumps(packet_id)
    export_name_json = json.dumps(export_name)
    annotator_id_json = json.dumps(annotator_id)
    blind_mode_json = json.dumps(bool(blind_mode))
    filter_placeholder = "blind_id or display order" if blind_mode else "cow_id, date, session, video_id"
    packet_heading = "Blinded External RR Breath Annotation Packet" if blind_mode else "External RR Breath Annotation Packet"
    warning = (
        "Count complete breaths visible in each included clip. This packet is randomized and shows only blinded clip IDs; "
        "do not use filenames, source folders, model output, or error reports while annotating. The RR value is computed as "
        "breath_count * 60 / duration. Draft entries are stored only in this browser's local storage until exported."
        if blind_mode
        else "Count complete breaths visible in each included clip. The RR value is computed as breath_count * 60 / duration. "
        "Draft entries are stored only in this browser's local storage until exported."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>External RR Breath Annotation Packet</title>
<style>
:root {{
  color-scheme: light;
  font-family: Arial, Helvetica, sans-serif;
  --border: #ccd3dd;
  --muted: #5a6675;
  --bg: #f6f8fb;
  --panel: #ffffff;
  --accent: #1d4ed8;
}}
body {{
  margin: 0;
  background: var(--bg);
  color: #111827;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 5;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 14px 18px;
}}
h1 {{
  margin: 0 0 10px;
  font-size: 20px;
  line-height: 1.25;
}}
.toolbar {{
  display: grid;
  grid-template-columns: repeat(5, minmax(130px, 1fr));
  gap: 10px;
  align-items: end;
}}
label {{
  display: grid;
  gap: 4px;
  color: var(--muted);
  font-size: 12px;
}}
input, textarea, button {{
  font: inherit;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 7px 8px;
  background: #fff;
}}
button {{
  cursor: pointer;
  border-color: #244ea8;
  background: var(--accent);
  color: #fff;
  min-height: 36px;
}}
main {{
  padding: 18px;
}}
.session {{
  margin: 0 0 22px;
}}
.session h2 {{
  margin: 0 0 8px;
  font-size: 16px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 12px;
}}
.clip {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}}
video {{
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: 6px;
}}
.fields {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}}
.wide {{
  grid-column: 1 / -1;
}}
.rr {{
  color: #0f766e;
  font-weight: 700;
  align-self: center;
}}
.warning {{
  color: #9a3412;
  font-size: 12px;
}}
@media (max-width: 760px) {{
  .toolbar, .fields, .grid {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<header>
  <h1>{packet_heading}</h1>
  <div class="toolbar">
    <label>Global annotator<input id="globalAnnotator" autocomplete="off"></label>
    <label>Global camera_id<input id="globalCamera" autocomplete="off"></label>
    <label>Show<input id="filter" placeholder="{filter_placeholder}"></label>
    <button type="button" id="applyGlobals">Apply to blank fields</button>
    <button type="button" id="exportCsv">Export annotation CSV</button>
  </div>
</header>
<main>
  <p class="warning">{warning}</p>
  <div id="root"></div>
</main>
<script>
const rows = {rows_json};
const packetId = {packet_id_json};
const exportName = {export_name_json};
const defaultAnnotator = {annotator_id_json};
const blindMode = {blind_mode_json};
const storageKey = "rr-breath-annotation:" + packetId;
const root = document.getElementById("root");
const draft = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
document.getElementById("globalAnnotator").value = defaultAnnotator || "";

function escapeHtml(value) {{
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }}[char]));
}}

function valueFor(id, field, fallback) {{
  return (draft[id] && draft[id][field] !== undefined) ? draft[id][field] : (fallback || "");
}}

function keyFor(row) {{
  return blindMode ? (row.blinded_id || row.external_video_id) : row.external_video_id;
}}

function saveValue(id, field, value) {{
  draft[id] = draft[id] || {{}};
  draft[id][field] = value;
  localStorage.setItem(storageKey, JSON.stringify(draft));
  updateRr(id);
}}

function computedRr(id) {{
  const row = rows.find((item) => keyFor(item) === id);
  const count = Number(valueFor(id, "manual_breath_count", row.manual_breath_count));
  const duration = Number(row.manual_duration_seconds);
  if (!Number.isFinite(count) || !Number.isFinite(duration) || duration <= 0) return "";
  return (count * 60 / duration).toFixed(3);
}}

function updateRr(id) {{
  const target = document.querySelector(`[data-rr="${{CSS.escape(id)}}"]`);
  if (target) target.textContent = computedRr(id) ? computedRr(id) + " bpm" : "--";
}}

function rowMatches(row, filterText) {{
  if (!filterText) return true;
  const haystack = blindMode
    ? [row.blinded_id, row.display_order].join(" ").toLowerCase()
    : [
        row.external_video_id, row.source_session_id, row.cow_id,
        row.collection_date, row.collection_time
      ].join(" ").toLowerCase();
  return haystack.includes(filterText.toLowerCase());
}}

function render() {{
  const filterText = document.getElementById("filter").value.trim();
  const grouped = new Map();
  rows.filter((row) => rowMatches(row, filterText)).forEach((row) => {{
    const key = blindMode ? "Blinded randomized queue" : (row.source_session_id || "unknown_session");
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }});
  root.innerHTML = "";
  grouped.forEach((items, sessionId) => {{
    const section = document.createElement("section");
    section.className = "session";
    section.innerHTML = `<h2>${{escapeHtml(sessionId)}} <span class="warning">(${{items.length}} clips)</span></h2><div class="grid"></div>`;
    const grid = section.querySelector(".grid");
    items.forEach((row) => {{
      const id = keyFor(row);
      const displayId = blindMode ? (row.blinded_id || id) : id;
      const metaHtml = blindMode
        ? `
          <span>${{escapeHtml(displayId)}}</span>
          <span>order=${{escapeHtml(row.display_order)}}</span>
          <span>duration=${{escapeHtml(row.manual_duration_seconds)}}s</span>`
        : `
          <span>${{escapeHtml(id)}}</span>
          <span>cow=${{escapeHtml(row.cow_id)}}</span>
          <span>${{escapeHtml(row.collection_date)}} ${{escapeHtml(row.collection_time)}}</span>
          <span>duration=${{escapeHtml(row.manual_duration_seconds)}}s</span>`;
      const clip = document.createElement("article");
      clip.className = "clip";
      clip.innerHTML = `
        <div class="meta">${{metaHtml}}</div>
        <video controls preload="metadata" src="${{escapeHtml(row.video_file_uri)}}"></video>
        <div class="fields">
          <label>breath_count<input type="number" min="0" step="1" data-id="${{escapeHtml(id)}}" data-field="manual_breath_count" value="${{escapeHtml(valueFor(id, "manual_breath_count", row.manual_breath_count))}}"></label>
          <div class="rr" data-rr="${{escapeHtml(id)}}">${{computedRr(id) ? computedRr(id) + " bpm" : "--"}}</div>
          <label>visibility<select data-id="${{escapeHtml(id)}}" data-field="annotation_visibility">
            <option value="">--</option>
            <option value="clear" ${{valueFor(id, "annotation_visibility", row.annotation_visibility) === "clear" ? "selected" : ""}}>clear</option>
            <option value="uncertain" ${{valueFor(id, "annotation_visibility", row.annotation_visibility) === "uncertain" ? "selected" : ""}}>uncertain</option>
            <option value="unreadable" ${{valueFor(id, "annotation_visibility", row.annotation_visibility) === "unreadable" ? "selected" : ""}}>unreadable</option>
          </select></label>
          <label>count_uncertainty<input type="number" min="0" step="1" data-id="${{escapeHtml(id)}}" data-field="count_uncertainty_breaths" value="${{escapeHtml(valueFor(id, "count_uncertainty_breaths", row.count_uncertainty_breaths))}}"></label>
          <label>annotator<input data-id="${{escapeHtml(id)}}" data-field="reference_rr_annotator" value="${{escapeHtml(valueFor(id, "reference_rr_annotator", row.reference_rr_annotator))}}"></label>
          <label>camera_id<input data-id="${{escapeHtml(id)}}" data-field="camera_id" value="${{escapeHtml(valueFor(id, "camera_id", row.camera_id))}}"></label>
          <label class="wide">notes<textarea rows="2" data-id="${{escapeHtml(id)}}" data-field="annotation_notes">${{escapeHtml(valueFor(id, "annotation_notes", row.annotation_notes))}}</textarea></label>
        </div>`;
      grid.appendChild(clip);
    }});
    root.appendChild(section);
  }});
  root.querySelectorAll("input[data-id], textarea[data-id], select[data-id]").forEach((node) => {{
    node.addEventListener("input", (event) => {{
      const target = event.target;
      saveValue(target.dataset.id, target.dataset.field, target.value);
    }});
  }});
}}

function csvCell(value) {{
  const text = String(value ?? "");
  return /[",\\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}}

function exportCsv() {{
  const columns = ["external_video_id", "blinded_id", "display_order", "manual_breath_count", "manual_rr_bpm", "reference_rr_annotator", "camera_id", "annotation_visibility", "count_uncertainty_breaths", "annotation_notes"];
  const lines = [columns.join(",")];
  rows.forEach((row) => {{
    const id = keyFor(row);
    const out = {{
      external_video_id: row.external_video_id,
      blinded_id: row.blinded_id,
      display_order: row.display_order,
      manual_breath_count: valueFor(id, "manual_breath_count", row.manual_breath_count),
      manual_rr_bpm: computedRr(id),
      reference_rr_annotator: valueFor(id, "reference_rr_annotator", row.reference_rr_annotator),
      camera_id: valueFor(id, "camera_id", row.camera_id),
      annotation_visibility: valueFor(id, "annotation_visibility", row.annotation_visibility),
      count_uncertainty_breaths: valueFor(id, "count_uncertainty_breaths", row.count_uncertainty_breaths),
      annotation_notes: valueFor(id, "annotation_notes", row.annotation_notes)
    }};
    lines.push(columns.map((column) => csvCell(out[column])).join(","));
  }});
  const blob = new Blob([lines.join("\\n") + "\\n"], {{type: "text/csv;charset=utf-8"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = exportName;
  link.click();
  URL.revokeObjectURL(link.href);
}}

document.getElementById("filter").addEventListener("input", render);
document.getElementById("applyGlobals").addEventListener("click", () => {{
  const annotator = document.getElementById("globalAnnotator").value.trim();
  const camera = document.getElementById("globalCamera").value.trim();
  rows.forEach((row) => {{
    const id = row.external_video_id;
    if (annotator && !valueFor(id, "reference_rr_annotator", row.reference_rr_annotator)) {{
      saveValue(id, "reference_rr_annotator", annotator);
    }}
    if (camera && !valueFor(id, "camera_id", row.camera_id)) {{
      saveValue(id, "camera_id", camera);
    }}
  }});
  render();
}});
document.getElementById("exportCsv").addEventListener("click", exportCsv);
render();
</script>
</body>
</html>
"""


def markdown_report(
    path: Path,
    worklist: pd.DataFrame,
    html_path: Path,
    worklist_path: Path,
    fieldwork_csv: Path,
    annotator_id: str,
    blind_mode: bool,
    blind_seed: int,
    prefill_existing_labels: bool,
) -> None:
    sessions = int(worklist["source_session_id"].nunique()) if not worklist.empty else 0
    cows = int(worklist["cow_id"].nunique()) if not worklist.empty else 0
    dates = int(worklist["collection_date"].nunique()) if not worklist.empty else 0
    text = f"""# External RR Breath Annotation Packet

HTML packet: `{html_path}`

Worklist CSV: `{worklist_path}`

Source fieldwork CSV: `{fieldwork_csv}`

Annotator ID: `{annotator_id or 'not preset'}`

Blind mode: `{'enabled' if blind_mode else 'disabled'}`

Blind randomization seed: `{blind_seed if blind_mode else 'not used'}`

Existing manual labels prefilled: `{'yes' if prefill_existing_labels else 'no'}`

## Scope

- Included clips: `{len(worklist)}`
- Source sessions: `{sessions}`
- cow_id labels: `{cows}`
- Collection dates: `{dates}`

## Use

Open the HTML packet in a browser, watch each clip, fill `breath_count`,
`reference_rr_annotator`, and `camera_id`, then export the annotation CSV.
Merge the exported CSV back into the fieldwork worksheet with
`scripts/merge_rr_external_breath_annotations.py`.

`manual_rr_bpm` is computed from `breath_count * 60 / duration_seconds`; it is
not a separate manual label requirement.

When blind mode is enabled, the on-screen queue hides `external_video_id`,
`cow_id`, date/time, and source-session metadata and displays `BLIND_0001` style
IDs in a seeded random order. The exported CSV keeps `external_video_id` so the
agreement and merge scripts can recover the correct row after annotation.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldwork_csv = (
        args.fieldwork_csv.resolve()
        if args.fieldwork_csv is not None
        else default_fieldwork_csv(args, output_dir).resolve()
    )
    inventory_csv = (
        args.inventory_csv.resolve()
        if args.inventory_csv is not None
        else default_inventory_csv(args, output_dir).resolve()
    )
    if not fieldwork_csv.exists():
        raise FileNotFoundError(f"Missing fieldwork CSV: {fieldwork_csv}")
    fieldwork = pd.read_csv(fieldwork_csv, dtype=str, keep_default_na=False)
    inventory = (
        pd.read_csv(inventory_csv, dtype=str, keep_default_na=False)
        if inventory_csv.exists()
        else None
    )
    worklist = build_worklist(
        fieldwork,
        inventory,
        args.blind,
        args.blind_seed,
        args.prefill_existing_labels,
    )

    annotator_suffix = (
        "_" + "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in args.annotator_id.strip())
        if args.annotator_id.strip()
        else ""
    )
    shared_worklist_path = output_dir / "paper_external_validation_breath_annotation_worklist.csv"
    worklist_path = (
        output_dir / f"paper_external_validation_breath_annotation_worklist{annotator_suffix}.csv"
        if annotator_suffix
        else shared_worklist_path
    )
    html_path = output_dir / f"paper_external_validation_breath_annotation_packet{annotator_suffix}.html"
    report_path = output_dir / f"paper_external_validation_breath_annotation_packet{annotator_suffix}.md"
    export_name = (
        args.export_name.strip()
        if args.export_name.strip()
        else f"paper_external_validation_breath_annotation{annotator_suffix}_export.csv"
    )
    packet_id = f"{args.packet_id}{annotator_suffix}"
    if args.blind:
        packet_id = f"{packet_id}_blind_{args.blind_seed}"

    worklist.to_csv(worklist_path, index=False)
    if worklist_path != shared_worklist_path:
        worklist.to_csv(shared_worklist_path, index=False)
    html_path.write_text(
        html_document(worklist, packet_id, export_name, args.annotator_id.strip(), args.blind),
        encoding="utf-8",
    )
    markdown_report(
        report_path,
        worklist,
        html_path,
        worklist_path,
        fieldwork_csv,
        args.annotator_id.strip(),
        args.blind,
        args.blind_seed,
        args.prefill_existing_labels,
    )

    print(f"Saved annotation worklist: {worklist_path}")
    if worklist_path != shared_worklist_path:
        print(f"Saved shared annotation worklist: {shared_worklist_path}")
    print(f"Saved annotation HTML packet: {html_path}")
    print(f"Saved annotation report: {report_path}")
    print(
        f"Included clips={len(worklist)}, sessions={worklist['source_session_id'].nunique() if not worklist.empty else 0}"
    )


if __name__ == "__main__":
    main()
