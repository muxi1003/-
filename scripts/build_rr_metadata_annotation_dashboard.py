from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METADATA_COLUMNS = [
    "video_id",
    "cow_id",
    "collection_date",
    "collection_start_date",
    "collection_end_date",
    "collection_time",
    "collection_location_country",
    "collection_location_province",
    "collection_location_county",
    "collection_site",
    "camera_id",
    "scene_id",
    "ambient_temperature_c",
    "relative_humidity_percent",
    "thi",
    "athi",
    "posture",
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
    "operator_or_annotator",
    "external_test_split",
    "notes",
]

REQUIRED_COLUMNS = [
    "cow_id",
    "collection_date",
    "camera_id",
    "scene_id",
    "ambient_temperature_c",
    "relative_humidity_percent",
    "head_motion_score_0_3",
    "occlusion_score_0_3",
    "nostril_visibility_score_0_3",
    "external_test_split",
]

CONTEXT_COLUMNS = [
    "annotation_priority",
    "q2_annotation_score",
    "q2_annotation_batch",
    "q2_missing_required_fields",
    "annotation_reason",
    "truth_rr",
    "rr_bpm",
    "corrected_rr_bpm",
    "signal_consensus_rr_bpm",
    "truth_count",
    "peaks",
    "corrected_peaks",
    "signal_consensus_peaks",
    "count_error",
    "corrected_count_error",
    "selective_abs_count_error",
    "abs_count_error",
    "applied_adjust",
    "signal_consensus_adjust",
    "signal_consensus_adjust_reason",
    "signal_consensus_adjust_signal_votes",
    "residual_confidence",
    "residual_margin",
    "signal_count_vote_agreement",
    "selective_review_score",
    "strict_auto_accept",
    "score_auto_accept",
    "selective_action",
    "selective_triage_reason",
    "selective_abs_rr_error",
    "error_type",
    "error_reason",
    "case_label",
    "duration_seconds",
    "selected_fusion_mode",
    "selection_missing_rate",
    "spectral_count_estimate",
    "fft_count_estimate",
    "autocorr_count_estimate",
    "sample_frame_first",
    "sample_frame_middle",
    "sample_frame_last",
    "curve_png",
    "review_png",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "Dataset_new" / "72video" / "al_images"
    parser = argparse.ArgumentParser(
        description="Build an offline HTML dashboard for Q2 metadata annotation."
    )
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", default="paper_repro")
    parser.add_argument("--corrected-prefix", default="paper_repro_quality_residual")
    parser.add_argument("--annotation-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def clean(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def file_uri(value: object) -> str:
    text = clean(value)
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = path.resolve()
    try:
        return path.as_uri()
    except ValueError:
        return text


def load_rows(annotation_csv: Path) -> list[dict[str, str]]:
    if not annotation_csv.exists():
        raise FileNotFoundError(f"Missing annotation CSV: {annotation_csv}")
    data = pd.read_csv(annotation_csv, dtype=str, keep_default_na=False)
    for column in [*METADATA_COLUMNS, *CONTEXT_COLUMNS]:
        if column not in data.columns:
            data[column] = ""
    data["q2_annotation_score_numeric"] = pd.to_numeric(
        data["q2_annotation_score"], errors="coerce"
    ).fillna(0)
    data = data.sort_values(
        ["q2_annotation_score_numeric", "q2_annotation_batch", "video_id"],
        ascending=[False, True, True],
    )
    rows: list[dict[str, str]] = []
    for _, row in data.iterrows():
        item = {column: clean(row.get(column, "")) for column in [*METADATA_COLUMNS, *CONTEXT_COLUMNS]}
        item["sample_frame_first_uri"] = file_uri(item["sample_frame_first"])
        item["sample_frame_middle_uri"] = file_uri(item["sample_frame_middle"])
        item["sample_frame_last_uri"] = file_uri(item["sample_frame_last"])
        item["curve_png_uri"] = file_uri(item["curve_png"])
        item["review_png_uri"] = file_uri(item["review_png"])
        rows.append(item)
    return rows


def html_document(rows: list[dict[str, str]], source_csv: Path) -> str:
    payload = {
        "rows": rows,
        "metadataColumns": METADATA_COLUMNS,
        "requiredColumns": REQUIRED_COLUMNS,
        "sourceCsv": str(source_csv),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thermal RR Metadata Annotation Dashboard</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #172026;
      --muted: #5d6975;
      --line: #d9e0e7;
      --accent: #0b6bcb;
      --accent-2: #007f5f;
      --warn: #a65300;
      --bad: #b42318;
      --good: #176b3a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 18px;
      background: #12212f;
      color: #fff;
      border-bottom: 1px solid #07131f;
    }}
    header h1 {{ font-size: 18px; margin: 0; font-weight: 700; }}
    header .meta {{ font-size: 12px; color: #cbd5df; }}
    button, select, input, textarea {{
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
    }}
    button {{
      padding: 7px 10px;
      cursor: pointer;
      background: #eef5ff;
      border-color: #b8d4f7;
      color: #0b4e91;
      font-weight: 700;
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 190px 150px 190px 160px 120px;
      gap: 10px;
      padding: 12px 18px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    .toolbar input, .toolbar select {{ width: 100%; padding: 7px 8px; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 39vw) 1fr;
      min-height: calc(100vh - 107px);
    }}
    .list {{
      overflow: auto;
      border-right: 1px solid var(--line);
      background: #fff;
      max-height: calc(100vh - 107px);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #eef2f6; z-index: 1; color: #344454; }}
    tr {{ cursor: pointer; }}
    tr.active {{ background: #eaf3ff; }}
    tr.done td:first-child {{ border-left: 4px solid var(--good); }}
    .priority-critical {{ color: var(--bad); font-weight: 700; }}
    .priority-high {{ color: var(--warn); font-weight: 700; }}
    .priority-medium {{ color: #7058a5; font-weight: 700; }}
    .detail {{
      overflow: auto;
      padding: 14px 18px 22px;
      max-height: calc(100vh - 107px);
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(6, minmax(100px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .metric {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 58px;
    }}
    .metric span {{ display: block; font-size: 11px; color: var(--muted); }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 15px; }}
    .video-head {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .video-head h2 {{ margin: 0; font-size: 22px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 7px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      font-size: 12px;
      color: #334155;
    }}
    .badge.bad {{ border-color: #f1b4ad; background: #fff1ef; color: var(--bad); }}
    .badge.good {{ border-color: #a8dcb8; background: #eefaf1; color: var(--good); }}
    .badge.warn {{ border-color: #f4cd9a; background: #fff7e8; color: var(--warn); }}
    .media {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      overflow: hidden;
      min-height: 160px;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: 210px;
      object-fit: contain;
      background: #101820;
    }}
    .frames figure img {{ height: 150px; }}
    figcaption {{
      padding: 6px 8px;
      font-size: 12px;
      color: var(--muted);
      border-top: 1px solid var(--line);
      background: #fff;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 10px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    label {{ display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted); }}
    label input, label select, label textarea {{ padding: 7px 8px; min-height: 34px; }}
    label textarea {{ min-height: 78px; resize: vertical; }}
    label.wide {{ grid-column: span 2; }}
    .context {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .context-box {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .context-box h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .actions {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .empty {{ padding: 28px; color: var(--muted); }}
    @media (max-width: 1080px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      .layout {{ grid-template-columns: 1fr; }}
      .list {{ max-height: 42vh; border-right: none; border-bottom: 1px solid var(--line); }}
      .detail {{ max-height: none; }}
      .summary {{ grid-template-columns: repeat(2, 1fr); }}
      .media, .context {{ grid-template-columns: 1fr; }}
      .form-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Thermal RR Metadata Annotation Dashboard</h1>
      <div class="meta" id="sourceLabel"></div>
    </div>
    <div class="actions">
      <button id="saveLocalBtn">Save Local</button>
      <button id="clearLocalBtn">Clear Local</button>
      <button class="primary" id="exportBtn">Export CSV</button>
    </div>
  </header>
  <section class="toolbar">
    <input id="searchBox" placeholder="Search video, reason, batch">
    <select id="batchFilter"></select>
    <select id="priorityFilter"></select>
    <select id="actionFilter"></select>
    <select id="completionFilter">
      <option value="all">All rows</option>
      <option value="incomplete">Incomplete required</option>
      <option value="complete">Complete required</option>
    </select>
    <button id="nextBtn">Next Open</button>
  </section>
  <main class="layout">
    <section class="list">
      <table>
        <thead>
          <tr>
            <th>Video</th>
            <th>Score</th>
            <th>Priority</th>
            <th>Batch</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody id="rowList"></tbody>
      </table>
    </section>
    <section class="detail" id="detail"></section>
  </main>
  <script id="dashboard-data" type="application/json">{payload_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('dashboard-data').textContent);
    const rows = data.rows;
    const metadataColumns = data.metadataColumns;
    const requiredColumns = data.requiredColumns;
    const storageKey = 'thermal_rr_metadata_dashboard_v1';
    let edits = {{}};
    let activeIndex = 0;

    function loadLocal() {{
      try {{ edits = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }}
      catch {{ edits = {{}}; }}
    }}
    function saveLocal() {{ localStorage.setItem(storageKey, JSON.stringify(edits)); }}
    function rowValue(row, column) {{
      return (edits[row.video_id] && edits[row.video_id][column] !== undefined)
        ? edits[row.video_id][column]
        : (row[column] || '');
    }}
    function setRowValue(row, column, value) {{
      if (!edits[row.video_id]) edits[row.video_id] = {{}};
      edits[row.video_id][column] = value;
      saveLocal();
      renderList();
      renderDetail(row);
    }}
    function isComplete(row) {{
      return requiredColumns.every(col => String(rowValue(row, col)).trim() !== '');
    }}
    function missingRequired(row) {{
      return requiredColumns.filter(col => String(rowValue(row, col)).trim() === '');
    }}
    function uniqueOptions(column) {{
      return [...new Set(rows.map(row => row[column] || '').filter(Boolean))].sort();
    }}
    function fillFilter(id, label, values) {{
      const select = document.getElementById(id);
      select.innerHTML = `<option value="all">${{label}}</option>` + values.map(v => `<option value="${{escapeHtml(v)}}">${{escapeHtml(v)}}</option>`).join('');
    }}
    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}
    function filteredRows() {{
      const q = document.getElementById('searchBox').value.trim().toLowerCase();
      const batch = document.getElementById('batchFilter').value;
      const priority = document.getElementById('priorityFilter').value;
      const action = document.getElementById('actionFilter').value;
      const completion = document.getElementById('completionFilter').value;
      return rows.filter(row => {{
        const haystack = [row.video_id, row.annotation_reason, row.q2_annotation_batch, row.error_type, row.case_label].join(' ').toLowerCase();
        if (q && !haystack.includes(q)) return false;
        if (batch !== 'all' && row.q2_annotation_batch !== batch) return false;
        if (priority !== 'all' && row.annotation_priority !== priority) return false;
        if (action !== 'all' && row.selective_action !== action) return false;
        if (completion === 'complete' && !isComplete(row)) return false;
        if (completion === 'incomplete' && isComplete(row)) return false;
        return true;
      }});
    }}
    function renderList() {{
      const tbody = document.getElementById('rowList');
      const visible = filteredRows();
      tbody.innerHTML = visible.map(row => {{
        const idx = rows.indexOf(row);
        const done = isComplete(row) ? 'done' : '';
        const active = idx === activeIndex ? 'active' : '';
        const priorityClass = `priority-${{row.annotation_priority || 'normal'}}`;
        return `<tr class="${{done}} ${{active}}" data-index="${{idx}}">
          <td><strong>${{escapeHtml(row.video_id)}}</strong><br><span>${{escapeHtml(row.case_label || row.error_type || '')}}</span></td>
          <td>${{escapeHtml(row.q2_annotation_score || '')}}</td>
          <td class="${{priorityClass}}">${{escapeHtml(row.annotation_priority || '')}}</td>
          <td>${{escapeHtml((row.q2_annotation_batch || '').replace('batch_', 'b'))}}</td>
          <td>${{escapeHtml(row.selective_action || '')}}</td>
        </tr>`;
      }}).join('');
      tbody.querySelectorAll('tr').forEach(tr => {{
        tr.addEventListener('click', () => {{
          activeIndex = Number(tr.dataset.index);
          renderList();
          renderDetail(rows[activeIndex]);
        }});
      }});
      if (!visible.includes(rows[activeIndex]) && visible.length) {{
        activeIndex = rows.indexOf(visible[0]);
        renderDetail(rows[activeIndex]);
      }}
      updateSourceLabel();
    }}
    function metric(label, value) {{
      return `<div class="metric"><span>${{escapeHtml(label)}}</span><strong>${{escapeHtml(value || '')}}</strong></div>`;
    }}
    function badge(text, cls='') {{
      if (!text) return '';
      return `<span class="badge ${{cls}}">${{escapeHtml(text)}}</span>`;
    }}
    function imgFigure(label, src, fallbackText='missing image') {{
      if (!src) return `<figure><figcaption>${{escapeHtml(label)}}: ${{fallbackText}}</figcaption></figure>`;
      return `<figure><img src="${{escapeHtml(src)}}" alt="${{escapeHtml(label)}}"><figcaption>${{escapeHtml(label)}}</figcaption></figure>`;
    }}
    function inputField(row, column, label, type='text', options=null, wide=false) {{
      const value = rowValue(row, column);
      const cls = wide ? ' class="wide"' : '';
      if (options) {{
        return `<label${{cls}}>${{escapeHtml(label)}}<select data-column="${{column}}">
          ${{options.map(opt => `<option value="${{escapeHtml(opt)}}" ${{String(value)===String(opt)?'selected':''}}>${{escapeHtml(opt || '')}}</option>`).join('')}}
        </select></label>`;
      }}
      if (type === 'textarea') {{
        return `<label${{cls}}>${{escapeHtml(label)}}<textarea data-column="${{column}}">${{escapeHtml(value)}}</textarea></label>`;
      }}
      return `<label${{cls}}>${{escapeHtml(label)}}<input data-column="${{column}}" type="${{type}}" value="${{escapeHtml(value)}}"></label>`;
    }}
    function renderDetail(row) {{
      const detail = document.getElementById('detail');
      if (!row) {{ detail.innerHTML = '<div class="empty">No rows</div>'; return; }}
      const missing = missingRequired(row);
      detail.innerHTML = `
        <div class="video-head">
          <div>
            <h2>${{escapeHtml(row.video_id)}}</h2>
            <div class="badges">
              ${{badge(row.annotation_priority, row.annotation_priority === 'critical' ? 'bad' : row.annotation_priority === 'high' ? 'warn' : '')}}
              ${{badge(row.q2_annotation_batch)}}
              ${{badge(row.selective_action, row.selective_action === 'manual_review_required' ? 'warn' : 'good')}}
              ${{badge(isComplete(row) ? 'complete' : `${{missing.length}} required missing`, isComplete(row) ? 'good' : 'bad')}}
            </div>
          </div>
          <div class="actions">
            <button id="prevRowBtn">Prev</button>
            <button id="nextRowBtn">Next</button>
          </div>
        </div>
        <div class="summary">
          ${{metric('Q2 score', row.q2_annotation_score)}}
          ${{metric('Truth / predicted RR', `${{row.truth_rr || ''}} / ${{row.signal_consensus_rr_bpm || row.corrected_rr_bpm || row.rr_bpm || ''}}`)}}
          ${{metric('Counts', `${{row.truth_count || ''}} / ${{row.signal_consensus_peaks || row.corrected_peaks || row.peaks || ''}}`)}}
          ${{metric('Review score', row.selective_review_score)}}
          ${{metric('Signal votes', row.signal_count_vote_agreement)}}
          ${{metric('Fusion', row.selected_fusion_mode)}}
        </div>
        <div class="media">
          <div class="frames">
            ${{imgFigure('first frame', row.sample_frame_first_uri)}}
            ${{imgFigure('middle frame', row.sample_frame_middle_uri)}}
            ${{imgFigure('last frame', row.sample_frame_last_uri)}}
          </div>
          <div class="frames">
            ${{imgFigure('curve', row.curve_png_uri)}}
            ${{imgFigure('peak review', row.review_png_uri)}}
            ${{imgFigure('middle frame zoom', row.sample_frame_middle_uri)}}
          </div>
        </div>
        <div class="form-grid">
          ${{inputField(row, 'cow_id', 'cow_id')}}
          ${{inputField(row, 'collection_date', 'collection_date', 'date')}}
          ${{inputField(row, 'collection_time', 'collection_time', 'time')}}
          ${{inputField(row, 'camera_id', 'camera_id')}}
          ${{inputField(row, 'scene_id', 'scene_id')}}
          ${{inputField(row, 'ambient_temperature_c', 'ambient_temperature_c', 'number')}}
          ${{inputField(row, 'relative_humidity_percent', 'relative_humidity_percent', 'number')}}
          ${{inputField(row, 'thi', 'thi', 'number')}}
          ${{inputField(row, 'athi', 'athi', 'number')}}
          ${{inputField(row, 'posture', 'posture', 'text', ['', 'standing', 'lying', 'walking', 'unknown'])}}
          ${{inputField(row, 'head_motion_score_0_3', 'head_motion_score_0_3', 'text', ['', '0', '1', '2', '3'])}}
          ${{inputField(row, 'occlusion_score_0_3', 'occlusion_score_0_3', 'text', ['', '0', '1', '2', '3'])}}
          ${{inputField(row, 'nostril_visibility_score_0_3', 'nostril_visibility_score_0_3', 'text', ['', '0', '1', '2', '3'])}}
          ${{inputField(row, 'operator_or_annotator', 'operator_or_annotator')}}
          ${{inputField(row, 'external_test_split', 'external_test_split', 'text', ['', 'internal', 'external', 'holdout', 'train', 'validation', 'test'])}}
          ${{inputField(row, 'notes', 'notes', 'textarea', null, true)}}
        </div>
        <div class="context">
          <div class="context-box"><h3>Annotation Reason</h3>${{escapeHtml(row.annotation_reason || '')}}</div>
          <div class="context-box"><h3>Signal Context</h3>${{escapeHtml([
            `error_type=${{row.error_type || ''}}`,
            `case_label=${{row.case_label || ''}}`,
            `selective_triage_reason=${{row.selective_triage_reason || ''}}`,
            `signal_consensus_adjust=${{row.signal_consensus_adjust || ''}}`,
            `residual_margin=${{row.residual_margin || ''}}`,
            `missing_required=${{missing.join(';')}}`
          ].join('\\n'))}}</div>
        </div>
      `;
      detail.querySelectorAll('[data-column]').forEach(el => {{
        el.addEventListener('input', event => setRowValue(row, event.target.dataset.column, event.target.value));
        el.addEventListener('change', event => setRowValue(row, event.target.dataset.column, event.target.value));
      }});
      document.getElementById('prevRowBtn').addEventListener('click', () => moveActive(-1));
      document.getElementById('nextRowBtn').addEventListener('click', () => moveActive(1));
    }}
    function moveActive(delta) {{
      const visible = filteredRows();
      if (!visible.length) return;
      const current = visible.indexOf(rows[activeIndex]);
      const next = current < 0 ? 0 : (current + delta + visible.length) % visible.length;
      activeIndex = rows.indexOf(visible[next]);
      renderList();
      renderDetail(rows[activeIndex]);
    }}
    function nextOpen() {{
      const visible = filteredRows();
      const found = visible.find(row => !isComplete(row));
      if (found) {{
        activeIndex = rows.indexOf(found);
        renderList();
        renderDetail(found);
      }}
    }}
    function csvEscape(value) {{
      const text = String(value ?? '');
      return /[",\\n\\r]/.test(text) ? `"${{text.replace(/"/g, '""')}}"` : text;
    }}
    function exportCsv() {{
      const lines = [metadataColumns.join(',')];
      rows.forEach(row => {{
        lines.push(metadataColumns.map(column => csvEscape(rowValue(row, column))).join(','));
      }});
      const blob = new Blob([lines.join('\\n') + '\\n'], {{type: 'text/csv;charset=utf-8'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'paper_repro_metadata_annotation_filled.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}
    function updateSourceLabel() {{
      const complete = rows.filter(isComplete).length;
      const visible = filteredRows().length;
      document.getElementById('sourceLabel').textContent = `${{rows.length}} videos | ${{visible}} shown | ${{complete}} complete | source: ${{data.sourceCsv}}`;
    }}
    function init() {{
      loadLocal();
      fillFilter('batchFilter', 'All batches', uniqueOptions('q2_annotation_batch'));
      fillFilter('priorityFilter', 'All priorities', uniqueOptions('annotation_priority'));
      fillFilter('actionFilter', 'All actions', uniqueOptions('selective_action'));
      ['searchBox','batchFilter','priorityFilter','actionFilter','completionFilter'].forEach(id => {{
        document.getElementById(id).addEventListener('input', renderList);
        document.getElementById(id).addEventListener('change', renderList);
      }});
      document.getElementById('saveLocalBtn').addEventListener('click', saveLocal);
      document.getElementById('clearLocalBtn').addEventListener('click', () => {{
        if (confirm('Clear local edits for this browser?')) {{
          edits = {{}};
          localStorage.removeItem(storageKey);
          renderList();
          renderDetail(rows[activeIndex]);
        }}
      }});
      document.getElementById('exportBtn').addEventListener('click', exportCsv);
      document.getElementById('nextBtn').addEventListener('click', nextOpen);
      renderList();
      renderDetail(rows[activeIndex]);
    }}
    init();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / f"{args.corrected_prefix}_paper_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_csv = args.annotation_csv or input_root / f"{args.output_prefix}_metadata_annotation_sheet.csv"
    rows = load_rows(annotation_csv)
    html = html_document(rows, annotation_csv)

    root_output = input_root / f"{args.output_prefix}_metadata_annotation_dashboard.html"
    assets_output = output_dir / "paper_metadata_annotation_dashboard.html"
    root_output.write_text(html, encoding="utf-8")
    assets_output.write_text(html, encoding="utf-8")
    print(f"Saved metadata annotation dashboard: {root_output}")
    print(f"Saved paper-assets metadata annotation dashboard: {assets_output}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
