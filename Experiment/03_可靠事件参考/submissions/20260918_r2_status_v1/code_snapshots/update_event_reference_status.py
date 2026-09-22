"""Version a status-only human R2 correction; reuse verified frozen predictions."""
import json
import shutil
from pathlib import Path

import pandas as pd

from experiment_common import *
from evaluate_event_submission import compare_export, evaluate, ref, require
from report_event_submission import verify_scoring


def main():
    version = '20260918_r2_status_v1'
    old_sub = REFERENCE / 'submissions/20260917_r2_v2'
    old_run = HOLDOUT / 'event_evaluation/20260917_r2_v2'
    old_delivery = ROOT / 'delivery_20260917_r2_v2'
    source = Path('C:/Users/muxi/Desktop/标注')
    sub = REFERENCE / 'submissions' / version
    out = HOLDOUT / 'event_evaluation' / version
    delivery = ROOT / ('delivery_' + version)
    require(not any(p.exists() for p in [sub, out, delivery]), 'Output already exists')
    old_w, old_e, old_i = ref.load_reference(old_sub / 'raw')
    w, e, intervals = ref.load_reference(source)
    backup = json.loads((source / 'event_reference_R2_backup.json').read_text(encoding='utf-8-sig'))
    for table, key in [(w, 'windows'), (e, 'events'), (intervals, 'intervals')]:
        compare_export(table, backup[key])
    require(e.equals(old_e) and intervals.equals(old_i), 'Event or interval changes require broader review')
    require(w.window_id.tolist() == old_w.window_id.tolist(), 'Window inventory changed')
    ids = ['jiufu_c2f523dbca4113_first30', 'jiufu_f6508f398bcf0f_first30']
    difference = w.compare(old_w)
    require(set(w.loc[difference.index, 'window_id']) == set(ids), 'Unexpected changed windows')
    require(set(difference.columns.get_level_values(0)) == {'annotation_status', 'manual_breath_count'}, 'Unexpected changed fields')
    before, after = old_w.set_index('window_id'), w.set_index('window_id')
    for wid, count in zip(ids, ['16', '10']):
        require(before.loc[wid, 'annotation_status'] == 'pending' and before.loc[wid, 'manual_breath_count'] == '', 'Unexpected previous state')
        require(after.loc[wid, 'annotation_status'] == 'complete' and after.loc[wid, 'manual_breath_count'] == count, 'Unexpected corrected state')
    errors = ref.validate_tables(w, e, intervals)
    require(not errors, str(errors))
    metrics = json.loads((old_run / 'event_metrics.json').read_text(encoding='utf-8'))
    for path, digest in metrics['input_hashes'].items():
        require(sha256(Path(path)) == digest, 'Previously evaluated input changed')
    prior_check = json.loads((old_delivery / 'completion_check.json').read_text(encoding='utf-8'))
    require(prior_check['status'] == 'PASS_DETERMINISTIC_DELIVERY', 'Prior delivery gate absent')
    provenance = json.loads((old_run / 'prediction_provenance.json').read_text(encoding='utf-8'))
    seal_path = HOLDOUT / 'external_transfer/20260914_v2/prediction_seal.json'
    require(sha256(seal_path) == provenance['prediction_seal_sha256'], 'Prediction seal changed')
    sealed = json.loads(seal_path.read_text(encoding='utf-8'))
    for filename, key in [('prediction_windows.csv', 'predictions_sha256'), ('protocol_lock.json', 'protocol_sha256')]:
        require(sha256(seal_path.parent / filename) == sealed[key], 'Sealed prediction/protocol changed')
    raw = sub / 'raw'
    raw.mkdir(parents=True)
    manifests = []
    for name in ['annotation_windows.csv', 'reference_events.csv', 'unobservable_intervals.csv', 'event_reference_R2_backup.json']:
        shutil.copyfile(source / name, raw / name)
        manifests.append(dict(source=str(source / name), snapshot=str(raw / name), sha256=sha256(raw / name)))
    write_json(sub / 'source_manifest.json', manifests)
    write_json(sub / 'validation.json', dict(status='PASS', events_unchanged=True, intervals_unchanged=True,
               only_two_pending_to_complete=True, errors=errors, previous_submission=str(old_sub)))
    write_csv(sub / 'changes.csv', [dict(window_id=wid, old_status='pending', new_status='complete',
              old_manual_count='', new_manual_count=after.loc[wid, 'manual_breath_count']) for wid in ids])
    write_csv(sub / 'coverage.csv', w.groupby(['cohort', 'annotation_status']).size().reset_index(name='n'))
    for cohort in ['lindian49', 'jiufu271']:
        subset = w[w.cohort.eq(cohort)]
        for name, table in [('annotation_windows.csv', subset), ('reference_events.csv', e[e.window_id.isin(subset.window_id)]),
                            ('unobservable_intervals.csv', intervals[intervals.window_id.isin(subset.window_id)])]:
            write_csv(sub / cohort / name, table)
    code = []
    for path in [Path(__file__), ROOT/'evaluate_event_submission.py', ROOT/'report_event_submission.py', REFERENCE/'reference_tools.py',
                 ROOT/'experiment_common.py', HOLDOUT/'score_external_counts.py']:
        target = sub / 'code_snapshots' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        code.append(dict(path=str(path.resolve()), snapshot=str(target), sha256=sha256(path)))
    write_json(sub / 'code_manifest.json', code)
    prediction_windows = read_csv(old_run / 'prediction_windows.csv')
    prediction_events = read_csv(old_run / 'prediction_events.csv')
    released = read_csv(seal_path.parent / 'test_windows.csv')
    result = evaluate(sub/'jiufu271', prediction_windows, prediction_events, out,
                      dict(zip(released.window_id, released.verified_cow_id)))
    summary = dict(cohort='jiufu271', variant='frozen_transfer_20260914_v2', **result)
    write_csv(out / 'metrics_summary.csv', [summary])
    checked = verify_scoring(out, sub/'jiufu271', summary)
    old_rows = read_csv(old_run/'event_metrics_by_window.csv').set_index('window_id')
    new_rows = read_csv(out/'event_metrics_by_window.csv').set_index('window_id')
    require(new_rows.loc[old_rows.index].equals(old_rows), 'Old complete-window matches changed')
    write_csv(out/'two_corrected_windows.csv', new_rows.loc[ids].reset_index())
    require(new_rows.loc[ids[0], 'predicted_count'] == '16', 'First corrected window count differs')
    require(new_rows.loc[ids[1], 'prediction_status'] == 'abstain' and int(new_rows.loc[ids[1], 'fn']) == 10, 'Abstained events not counted as FN')
    write_json(out/'verification.json', dict(status='PASS', independent_DP_and_formulas=checked,
               previous_214_window_results_unchanged=True, prior_prediction_seal_sha256=sha256(seal_path),
               previous_prediction_adapter_hashes={name:sha256(old_run/name) for name in ['prediction_windows.csv','prediction_events.csv']},
               validation_scope='status_only_revision; reused_prior_verified_time_mapping; no_raw_video_or_inference_rerun'))
    delivery.mkdir()
    previous_summary = read_csv(old_delivery/'metrics_summary.csv')
    combined = pd.concat([previous_summary[previous_summary.cohort.eq('lindian49')], pd.DataFrame([summary])], ignore_index=True)
    write_csv(delivery/'metrics_summary.csv', combined)
    text = f'''# 两个pending窗口修订后结果

2026-09-18正式导出，保留原R2事件定义和0.30秒容差。只修改了两窗状态与确认总数，4345个事件时间及区间表不变。未重跑算法、调参或覆盖旧版。

| 项目 | 修订后 |
|---|---:|
| 久福完整参考窗 | {result['complete_reference_windows']} / 271 |
| RR/计数配对窗 | {result['n']} |
| RR R² | {result['rr_r2']:.6f} |
| RR MAE（次/分） | {result['rr_mae_bpm']:.6f} |
| 次数平均准确度 | {result['mean_count_accuracy_percent']:.6f}% |
| 事件Precision | {result['precision']:.6f} |
| 事件Recall | {result['recall']:.6f} |
| 事件F1 | {result['f1']:.6f} |
| TP / FP / FN | {result['tp']} / {result['fp']} / {result['fn']} |

20240801T105204-919.MP4：16个确定事件，算法也是16次；总数相同不代表逐事件全部匹配，详见two_corrected_windows.csv。
20240806T095740-373.MP4：10个确定事件，算法拒判，10个事件计FN，不进入RR配对评分。

久福剩余8 partial、47 unobservable，pending=0；216个完整参考窗中43个算法拒判。全271窗算法仍216输出/55拒判。参考完整数与算法输出数恰好均为216，不是同一集合。

林甸47个完整窗及8组消融结果完全不变，沿用01_同口径消融/reference_updates/20260917_r2_v2。两版指标差异仅来自分析集修订，不是模型改进。

## 三部分文件入口

- 01同口径消融：{ABLATION / 'reference_updates/20260917_r2_v2/RESULTS.md'}，不需要重跑。
- 02冻结测试：{out}，更新指标、逐事件匹配、计数明细、聚类区间与核验。
- 03可靠事件参考：{sub}，原始导出、两窗差异、覆盖表及代码快照。

原版20260917_r2_v2历史结果保留。旧版“2 pending”仅适用于旧提交，不再是当前待办。不可观察区间仍0行，不做部分窗口评分；历史暴露确认、独立生理相位、双观察者一致性与绝对温标限制继续保留。不能据此声称高精度跨场泛化，也不能用本次外测结果调参再称独立测试。
'''
    write_text(delivery/'交付说明.md', text)
    write_text(out/'RESULTS.md', text)
    write_text(sub/'README.md', '# R2状态修订归档\n\n来源为桌面“标注”2026-09-18导出。raw为原字节备份，changes.csv为两窗变化，coverage.csv为全部窗口状态，按牧场拆分三表用于复核，validation.json为一致性检查，code_snapshots为本次工具版本。\n\n事件及区间未变化，只把两个pending改为complete并确认16/10次。pending已清零，不把47个unobservable或8个partial当作0次。结果入口：'+str(out/'RESULTS.md')+'\n')
    for item in manifests:
        require(sha256(Path(item['source'])) == item['sha256'], 'User export changed during evaluation')
    write_json(delivery/'completion_check.json', dict(status='PASS_STATUS_ONLY_REFERENCE_UPDATE',
               original_exports_unchanged=True, no_predictions_rerun=True, pending=0, complete_jiufu=216,
               paired_jiufu=173, internal_ablation_unchanged=True, previous_214_matches_unchanged=True))
    write_csv(delivery/'artifact_manifest.csv', [dict(path=str(p), sha256=sha256(p), bytes=p.stat().st_size)
              for folder in [sub,out,delivery] for p in sorted(folder.rglob('*')) if p.is_file()])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
