"""Independent numerical checks and bounded reports for archived human R2 events."""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_common import *


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered_matching(a, b, tol=.30):
    """Independent sorted-time DP: maximize matches, then minimize total time error."""
    a, b = sorted(a), sorted(b)
    dp = [[(0, 0.) for _ in range(len(b)+1)] for _ in range(len(a)+1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            choices = [dp[i-1][j], dp[i][j-1]]
            if abs(x-y) <= tol + 1e-9:
                count, cost = dp[i-1][j-1]
                choices.append((count+1, cost+abs(x-y)))
            dp[i][j] = min(choices, key=lambda pair: (-pair[0], pair[1]))
    return dp[-1][-1]


def verify_scoring(folder, annotations, summary):
    w = read_csv(annotations / 'annotation_windows.csv')
    r = read_csv(annotations / 'reference_events.csv')
    p = read_csv(folder / 'prediction_events.csv')
    pw = read_csv(folder / 'prediction_windows.csv').set_index('window_id')
    results = read_csv(folder / 'event_metrics_by_window.csv').set_index('window_id')
    details = read_csv(folder / 'event_matches.csv')
    complete = w[w.annotation_status.eq('complete')]
    require(set(results.index) == set(complete.window_id), 'Complete set mismatch')
    m = json.loads((folder / 'event_metrics.json').read_text(encoding='utf-8'))
    for path, digest in m['input_hashes'].items():
        require(sha256(Path(path)) == digest, 'Scoring input changed')
    for row in complete.itertuples():
        pt = p[p.window_id.eq(row.window_id)]
        rt = r[r.window_id.eq(row.window_id) & r.confidence.eq('confirmed')]
        count, cost = ordered_matching(pt.event_time_seconds.astype(float), rt.event_time_seconds.astype(float))
        result = results.loc[row.window_id]
        require((count, len(pt)-count, len(rt)-count) == tuple(int(result[k]) for k in ['tp', 'fp', 'fn']), 'Independent DP mismatch')
        match = details[details.window_id.eq(row.window_id)]
        tp = match[match.type.eq('TP')]
        require(abs(tp.time_error_seconds.astype(float).sum()-cost) < 1e-7, 'Minimum-error matching differs')
        require(not match.loc[match.predicted_event_id.ne(''), 'predicted_event_id'].duplicated().any(), 'Duplicate predicted match')
        require(not match.loc[match.reference_event_id.ne(''), 'reference_event_id'].duplicated().any(), 'Duplicate reference match')
        require(set(match.loc[match.predicted_event_id.ne(''), 'predicted_event_id']) == set(pt.event_id), 'Missing prediction detail')
        require(set(match.loc[match.reference_event_id.ne(''), 'reference_event_id']) == set(rt.event_id), 'Missing reference detail')
        for match_row in tp.itertuples():
            actual = abs(float(pt.set_index('event_id').loc[match_row.predicted_event_id, 'event_time_seconds']) - float(rt.set_index('event_id').loc[match_row.reference_event_id, 'event_time_seconds']))
            require(abs(actual-float(match_row.time_error_seconds)) < 1e-9 and actual <= .30+1e-9, 'Invalid matched detail')
    tp, fp, fn = [results[k].astype(int).sum() for k in ['tp', 'fp', 'fn']]
    require((tp, fp, fn) == (m['tp'], m['fp'], m['fn']), 'Aggregate mismatch')
    require(abs(m['f1'] - 2*tp/(2*tp+fp+fn)) < 1e-12, 'F1 mismatch')
    derived = dict(complete_reference_windows=len(complete), tp=tp, fp=fp, fn=fn,
                   precision=tp/(tp+fp), recall=tp/(tp+fn), f1=2*tp/(2*tp+fp+fn),
                   algorithm_output_coverage_on_complete=float(pw.loc[complete.window_id].prediction_status.eq('ok').mean()))
    for key, value in derived.items():
        require(abs(float(m[key])-value) < 1e-10 and abs(float(summary[key])-value) < 1e-10, f'Event summary differs: {key}')
    require(m['all_annotation_windows'] == len(w) and abs(m['reference_window_coverage']-len(complete)/len(w)) < 1e-10, 'Coverage mismatch')
    pair = read_csv(folder / 'paired_count_results.csv')
    expected_ids = set(complete.window_id) & set(pw[pw.prediction_status.eq('ok')].index)
    require(set(pair.window_id) == expected_ids, 'Count analysis set mismatch')
    truth = complete.set_index('window_id').loc[pair.window_id, 'manual_breath_count'].astype(float).to_numpy()
    counts = pw.loc[pair.window_id, 'predicted_count'].astype(float).to_numpy()
    duration = complete.set_index('window_id').loc[pair.window_id, 'duration_seconds'].astype(float).to_numpy()
    y, pred = 60*truth/duration, 60*counts/duration
    cm = json.loads((folder / 'count_metrics.json').read_text(encoding='utf-8'))['metrics']
    require(np.allclose(pair.truth_count.astype(float), truth) and np.allclose(pair.predicted_count.astype(float), counts), 'Count row mismatch')
    require(abs(cm['rr_r2'] - (1-np.square(pred-y).sum()/np.square(y-y.mean()).sum())) < 1e-10, 'R2 mismatch')
    require(abs(cm['rr_mae_bpm'] - np.abs(pred-y).mean()) < 1e-10, 'MAE mismatch')
    require(abs(cm['mean_count_accuracy_percent'] - 100*np.maximum(0, 1-np.abs(counts[truth>0]-truth[truth>0])/truth[truth>0]).mean()) < 1e-10, 'Count accuracy mismatch')
    direct = dict(n=len(pair), rr_r2=1-np.square(pred-y).sum()/np.square(y-y.mean()).sum(), rr_mae_bpm=np.abs(pred-y).mean(),
                  rr_rmse_bpm=np.sqrt(np.square(pred-y).mean()), count_mae=np.abs(counts-truth).mean(),
                  exact_count=int((counts==truth).sum()), within_one_count=int((np.abs(counts-truth)<=1).sum()),
                  mean_count_accuracy_percent=100*np.maximum(0,1-np.abs(counts[truth>0]-truth[truth>0])/truth[truth>0]).mean(),
                  bias_bpm=(pred-y).mean(), loa_low_bpm=(pred-y).mean()-1.96*(pred-y).std(ddof=1),
                  loa_high_bpm=(pred-y).mean()+1.96*(pred-y).std(ddof=1))
    for key, value in direct.items():
        require(abs(float(cm[key])-value) < 1e-9 and abs(float(summary[key])-value) < 1e-9, f'Count summary differs: {key}')
    return dict(folder=str(folder), complete=len(complete), paired=len(pair), tp=int(tp), fp=int(fp), fn=int(fn), status='PASS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='20260917_r2_v2')
    args = parser.parse_args()
    sub = REFERENCE / 'submissions' / args.id
    internal = ABLATION / 'reference_updates' / args.id
    external = HOLDOUT / 'event_evaluation' / args.id
    out = ROOT / ('delivery_' + args.id)
    manifest = read_csv(out / 'artifact_manifest.csv')
    for row in manifest.itertuples():
        require(sha256(Path(row.path)) == row.sha256, 'Artifact changed: ' + row.path)
    previous = read_csv(ROOT / 'delivery_20260914_v1/文件用途与哈希.csv')
    prediction = ABLATION / 'runs/20260909_v1/paired_predictions.csv'
    old = previous[previous.path.map(Path).eq(prediction)]
    require(len(old) == 1 and sha256(prediction) == old.iloc[0].sha256, 'Historical ablation predictions changed')
    policy = json.loads((sub / 'analysis_policy_before_scoring.json').read_text(encoding='utf-8'))
    for item in policy['source_exports']:
        require(sha256(Path(item['snapshot'])) == item['sha256'], 'Archived user bytes changed')
    summaries = read_csv(out / 'metrics_summary.csv')
    expected = {('lindian49', f'F{f}G{g}P{p}') for f in [0, 1] for g in [0, 1] for p in [0, 1]}
    expected.add(('jiufu271', 'frozen_transfer_20260914_v2'))
    require(len(summaries) == 9 and not summaries.duplicated(['cohort', 'variant']).any() and set(zip(summaries.cohort, summaries.variant)) == expected, 'Expected nine unique methods')
    checks = []
    for row in summaries.itertuples():
        folder = external if row.cohort == 'jiufu271' else internal / row.variant
        checks.append(verify_scoring(folder, sub / row.cohort, row._asdict()))
    write_json(out / 'numerical_verification.json', {'status': 'PASS', 'checks': checks,
        'method': 'independent_sorted_DP_cardinality_and_total_time_error; unique_event_detail_checks; direct_count_formulas; output_and_source_hashes',
        'human_phase_correctness_verified': False, 'review_independence': 'same_executor_deterministic_not_independent_scientific_certification'})

    # Recover old notes only as explicitly versioned background, never rewrite R2 truth/status.
    w = read_csv(sub / 'raw/annotation_windows.csv')
    prior = read_csv(REFERENCE / 'submissions/20260911_v2/jiufu271/count_reference_only.csv').set_index('window_id')
    excluded = w[w.annotation_status.ne('complete')].copy()
    excluded['R1_note_background_only'] = excluded.window_id.map(prior['备注']).fillna('')
    excluded['R1_status_background_only'] = excluded.window_id.map(prior.annotation_status).fillna('')
    excluded['submission_source'] = '2026-09-17 Desktop/标注; prior Downloads export superseded'
    write_csv(sub / 'exclusion_reason_reconciliation.csv', excluded)
    pending = excluded[excluded.annotation_status.eq('pending')]
    notes_n = int(pending.R1_note_background_only.ne('').sum())

    # R1/R2 compare counts on the common complete set, not event repeatability.
    comparisons = []
    for cohort in ['lindian49', 'jiufu271']:
        r1 = read_csv(REFERENCE / f'submissions/20260911_v2/{cohort}/count_reference_only.csv')
        r2 = w[w.cohort.eq(cohort)]
        common = r2.merge(r1[['window_id', 'annotation_status', 'manual_breath_count']], on='window_id', suffixes=('_R2', '_R1'), validate='one_to_one')
        eligible = common.annotation_status_R2.eq('complete') & common.annotation_status_R1.isin(['complete', 'completed'])
        common['count_delta_R2_minus_R1'] = ''
        common.loc[eligible, 'count_delta_R2_minus_R1'] = common.loc[eligible, 'manual_breath_count_R2'].astype(float) - common.loc[eligible, 'manual_breath_count_R1'].astype(float)
        common['common_complete_count_comparison'] = eligible
        write_csv(sub / cohort / 'R1_R2_count_comparison.csv', common)
        differences = common.loc[eligible, 'count_delta_R2_minus_R1'].astype(float)
        comparisons.append(dict(cohort=cohort, n=len(differences), exact=int(differences.eq(0).sum()), mean_absolute_count_difference=float(differences.abs().mean())))
    write_csv(sub / 'R1_R2_count_agreement.csv', comparisons)

    header = '| 范围 / 方法 | 完整事件窗 | 计数配对窗 | R² | RR MAE | 次数平均准确度 | Precision | Recall | F1 |'
    lines = [header, '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries.itertuples():
        lines.append(f'| {r.cohort} / {r.variant} | {r.complete_reference_windows} | {r.n} | {float(r.rr_r2):.6f} | {float(r.rr_mae_bpm):.4f} | {float(r.mean_count_accuracy_percent):.2f}% | {float(r.precision):.4f} | {float(r.recall):.4f} | {float(r.f1):.4f} |')
    table = '\n'.join(lines)
    boundaries = '''
## 口径和限制

- R2来自用户逐事件导出，不由算法峰生成。原R1及本次原始CSV/JSON均保留，不把旧总次数覆盖为新次数。
- 固定0.30秒容差，最大数量一对一匹配，再最小化时间误差。没有按结果移动相位、放宽容差或修改峰。
- 事件主分析只用complete。算法拒判窗中的真实事件全部记FN；计数R²/MAE仅用complete与算法ok交集，两种指标的分母不同。
- partial、不确定事件、unobservable及pending不充当完整真值。不可观察区间表为0行，不据此认定所有窗口都可见，不进行部分窗口事件评分。
- 林甸属于内部开发集且有历史算法接触，本轮隐藏页面不等于独立盲测；旧视频CFR时间轴不证明原始VFR时间轴已修复。
- 久福复用9月14日封存预测；这是同一外测队列的新参考评估，不是第二批独立测试。单人页面声明本轮隐藏预测，不是双盲或独立传感器金标准。
- 旧外测锁定文件的终点是count-only/no-event-F1；本轮事件指标是新增参考后的补充终点，不冒称原预注册事件终点。原协议与预测封存不改写。
- 林甸按原视频、久福按已核验牛号聚类bootstrap；不能把林甸分组写成独立牛。bootstrap不消除人工选择与缺失偏差。
- RR总次数相近仍可能有时间错位、伪峰和漏峰抵消，不能用R²替代事件F1。人工呼气相与热曲线峰的生理对应尚无独立传感器验证。
- 绝对温标未知，RF映射只按伪彩色代理信号迁移解释。YOLO同训练划分的网络消融未运行，不属于已完成结果。
- 未更换默认算法或按久福结果调参。73短视频原R1消融结果保持不变，不把47窗R2指标冒称73视频或全49窗性能。
'''
    write_text(internal / 'RESULTS.md', '# 林甸R2同口径事件消融\n\n固定已有8组预测，仅更新独立保存的R2参考评价；不是重训或新推理。\n\n' + '\n'.join(lines[:2]+[line for line in lines[2:] if 'lindian49' in line]) + '\n' + boundaries)
    write_text(external / 'RESULTS.md', '# 久福封存预测的R2事件评估\n\n全部271窗保留；214窗有完整事件参考。计数配对集合与事件评价集合见下表，拒判不填0。\n\n' + '\n'.join(lines[:2]+[line for line in lines[2:] if 'jiufu271' in line]) + '\n' + boundaries)
    write_text(sub / 'README.md', f'''# 本次人工事件参考提交

来源：用户指定2026-09-17桌面“标注”目录三表及JSON。CSV与备份语义一致；12和12.0只按数值等价处理，原字节不修改。下载目录9月16日旧版停止评估、保留为superseded，不用于最终报告。

320窗、4345个事件（4340 confirmed、5 uncertain）、不可观察区间0行。
林甸47 complete、2 partial；久福214 complete、8 partial、47 unobservable、2 pending。

此前152窗问题属于错误选择的9月16日旧导出，不适用于新版。新版仅剩2窗pending，无事件与本轮备注，旧R1其中{notes_n}条有备注，背景列单列而不回写R2。不将排除窗标成0或强制改状态。

- raw/：原字节副本及JSON；不得回写。
- lindian49/、jiufu271/：按牧场拆分三表及R1/R2次数比较。
- reference_validation.json：结构、声明、事件/次数一致性、元数据绑定检查。
- exclusion_reason_reconciliation.csv：所有排除窗口、原状态、新旧备注和用户声明。
- lindian_frame_time_bindings.csv：实际解码的观看视频逐帧秒数。
- analysis_policy_before_scoring.json：评分规则、原始文件哈希及用户说明。
- R1_R2_count_agreement.csv：仅总次数的一致性，不是两轮逐事件重复性。

原R1没有事件时刻，不能计算观察者内逐事件重测一致性。软件检查通过不证明人工相位本身准确。
''')

    numeric = summaries.copy()
    for column in ['rr_r2', 'rr_mae_bpm', 'f1', 'precision', 'recall']:
        numeric[column] = numeric[column].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    ld = numeric[numeric.cohort.eq('lindian49')]
    for ax, key, title in zip(axes, ['rr_r2', 'rr_mae_bpm', 'f1'], ['RR R-squared', 'RR MAE (breaths/min)', 'Event F1 (0.30 s)']):
        ax.bar(ld.variant, ld[key], color=['#167d8d' if v != 'F1G1P1' else '#c86c30' for v in ld.variant])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=50)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Lindian R2 / 47 complete windows / fixed predictions')
    fig.savefig(internal / 'ablation_R2.png', dpi=180)
    plt.close(fig)
    pair = read_csv(external / 'paired_count_results.csv')
    y, p = pair.truth_rr_bpm.astype(float), pair.predicted_rr_bpm.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].scatter(y, p, color='#167d8d', alpha=.7)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    axes[0].plot([lo, hi], [lo, hi], '--', color='#888888')
    axes[0].set(xlabel='Human R2 RR (breaths/min)', ylabel='Frozen prediction RR (breaths/min)', title=f'Jiufu paired count set: n={len(pair)}')
    error = p-y
    axes[1].scatter((p+y)/2, error, color='#c86c30', alpha=.7)
    for value in [error.mean(), error.mean()-1.96*error.std(ddof=1), error.mean()+1.96*error.std(ddof=1)]:
        axes[1].axhline(value, color='#777777', linestyle='--')
    axes[1].set(xlabel='Mean RR (breaths/min)', ylabel='Prediction - human RR', title='Bland-Altman: paired outputs only')
    fig.savefig(external / 'count_scatter_BA_R2.png', dpi=180)
    plt.close(fig)
    write_text(out / '交付说明.md', '# 三部分证据补齐：R2事件参考评估\n\n' + table + '\n' + boundaries + f'''
## 文件入口

1. 同口径消融：`{internal}`，RESULTS.md、metrics_summary.csv、8组事件匹配明细、R2图、聚类区间。
2. 冻结测试：`{external}`，RESULTS.md、逐窗TP/FP/FN及配对计数、时间匹配、聚类区间和封存核验。
3. 可靠事件参考：`{sub}`，原始标注副本、来源验证、排除说明核对、时间轴及R1/R2次数差异。

本轮可自动完成的评分与核验已完成。新版仅2个pending窗保留待确认；不需要把看不清的视频强行标出事件。当前提交不支持部分窗口事件分析或双人一致性。久福prior_algorithm_exposure仍是模板待确认值，本轮predictions_hidden=true只说明页面声明，不夸大为独立盲法证据。

独立数值核验见numerical_verification.json。额外同族审查另列，不能替代独立科学认证。
''')
    write_csv(out / 'pending_confirmation.csv', w[w.annotation_status.eq('pending')])
    shutil.copyfile(Path(__file__), out / 'reporter_snapshot.py')
    write_json(out / 'reporter_provenance.json', {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__)),
                                               'snapshot_sha256': sha256(out / 'reporter_snapshot.py')})
    write_csv(out / 'report_artifact_manifest.csv', [{'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size}
              for folder in [sub, internal, external, out] for p in sorted(folder.rglob('*')) if p.is_file()])
    print('Numerical verification PASS; reports and figures saved', flush=True)


if __name__ == '__main__':
    main()
