"""Evidence tables and full-frame overlays for the fixed training-budget control."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
from analyze_transfer import read, write_csv
from train_nostril_grouped_probe import ROOT, OUT as BASE, REF
from train_nostril_longer_probe import OUT
from probe_pose_mechanism import load_bgr


def main():
    destination = OUT / 'figures'
    destination.mkdir(exist_ok=False)
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    loss_rows = []
    for fold, ax in enumerate(axes):
        for name, folder, color in [('30 epochs', BASE, '#a34b31'), ('200 epochs', OUT, '#18789b')]:
            d = read(folder / 'runs' / f'fold{fold}/results.csv')
            d.columns = d.columns.str.strip()
            ax.plot(d.epoch, d['train/cls_loss'], color=color, label=name)
            loss_rows.append(dict(fold=fold, schedule=name, final_epoch=int(d.epoch.iloc[-1]),
                final_cls_loss=float(d['train/cls_loss'].iloc[-1]), final_box_loss=float(d['train/box_loss'].iloc[-1])))
        ax.set(title=f'Fold {fold}', xlabel='Epoch', ylabel='Training classification loss')
        ax.legend(); ax.grid(alpha=.2)
    figure.savefig(destination / 'training_loss.png', dpi=160)
    plt.close(figure)
    write_csv(OUT / 'training_loss_comparison.csv', loss_rows)
    detail = read(OUT / 'evaluation/per_frame.csv')
    base = detail[detail.method.eq('original_pose')].set_index('frame_id')
    candidate = detail[detail.method.eq('direct_nostril_oof')].set_index('frame_id')
    delta = candidate.correct - base.correct
    selected = list(dict.fromkeys([delta.idxmax(), delta.idxmin()]))
    raw = json.loads(REF.read_text(encoding='utf-8'))
    poses = read(ROOT / 'pose_keypoints.csv')
    poses = poses[poses.variant.eq('baseline') & poses.confidence.ge(.5) & poses.inside_native]
    predictions = read(OUT / 'out_of_fold_predictions.csv')
    for fid in selected:
        image = load_bgr(ROOT / 'frames' / f'{fid}.png')[:, :, ::-1]
        fig, axs = plt.subplots(1, 2, figsize=(8, 6), constrained_layout=True)
        for method, ax in zip(['Original pose', '200-epoch held-out detector'], axs):
            ax.imshow(image)
            for e in raw['records'][fid]['regions'].values():
                if e is not None:
                    ax.add_patch(Ellipse((e['cx'], e['cy']), 2*e['rx'], 2*e['ry'],
                                        fill=False, color='#19f04b', linewidth=1.4))
            if method == 'Original pose':
                p = poses[poses.frame_id.eq(fid)]
                ax.scatter(p.x, p.y, c='#00f3ff', marker='+', s=65)
            else:
                p = predictions[predictions.frame_id.eq(fid)]
                for r in p.itertuples():
                    ax.add_patch(Rectangle((r.x0, r.y0), r.x1-r.x0, r.y1-r.y0,
                                           fill=False, color='#00f3ff', linewidth=1.3))
                    ax.plot(r.x, r.y, '+', color='#00f3ff')
            ax.set_title(method, fontsize=10); ax.axis('off')
        fig.suptitle(f'{fid} | green: human ellipse; cyan: prediction\n'
                     f'Correct points: {int(base.loc[fid,"correct"])} -> {int(candidate.loc[fid,"correct"])}; '
                     'extreme-change diagnostic, not a typical case', fontsize=9)
        fig.savefig(destination / f'{fid}_full_frame.png', dpi=170)
        plt.close(fig)
    metrics = read(OUT / 'evaluation/summary.csv')
    old_fit = read(BASE / 'training_fit_diagnostic/summary.csv')
    fit = read(OUT / 'training_fit_diagnostic/summary.csv')
    lines = ['# 固定训练长度对照：鼻孔漏检的进一步定位', '',
        '2026-09-22；N065实验附件。原始人工标注、原Pose模型、默认RR均未修改。', '',
        '## 1. 为何继续训练诊断', '',
        '30轮直接鼻孔检测器不仅折外零输出，在训练帧上固定conf=0.25也仅匹配5/100个区域出现次数。'
        '三折训练集有重复帧，这100不是100个独立人工鼻孔。108份标签坐标回算和原图尺寸核验通过。'
        '低阈值0.001仅用于诊断，训练匹配54/100、折外17/50且误检很多，不能降阈值后宣称问题解决。', '',
        '## 2. 固定对照', '',
        '保留三折牛号划分、原始初始化、数据增强、640输入、freeze10和主要推理conf=0.25。'
        '预算从30轮延长到200轮，关闭提前停止，采用200轮last.pt。学习率随总轮数的时间表也随之延长，'
        '所以这是训练预算/调度对照，不是隔离出单一损失项的因果实验。没有按折外结果选轮次或阈值。', '',
        '## 3. 训练拟合与折外表现', '',
        '| 预算 | 集合 | 帧-模型对 | 区域出现次数 | 输出 | 正确 | 召回率 | 点精确率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for schedule, d in [('30轮', old_fit), ('200轮', fit)]:
        for r in d[d.threshold.eq(.25)].itertuples():
            lines.append(f'| {schedule} | {r.subset} | {r.frame_model_pairs} | {r.expected} | '
                         f'{r.predicted} | {r.correct} | {r.recall:.4f} | {r.precision:.4f} |')
    lines += ['', 'training为三折合计72个帧-模型对，36张独立图片各被用于另外两折训练；'
              'heldout为36张图片各一次，50个人工区域。所有帧均保留，包括6张无可见鼻孔帧。', '',
              '![训练损失](figures/training_loss.png)', '', '## 4. 与原Pose的同帧比较', '',
              '| 方法 | 输出点 | 正确点 | 召回率 | 点精确率 |', '|---|---:|---:|---:|---:|']
    for r in metrics.itertuples():
        lines.append(f'| {r.method} | {r.predicted_points} | {r.correct_points} | {r.recall:.4f} | {r.precision:.4f} |')
    for fid in selected:
        lines += ['', f'![全帧定位对照 {fid}](figures/{fid}_full_frame.png)']
    lines += ['', '配图按正确点变化最大/最小各选一帧，完整图片不裁去误检区域；'
              '所有36帧数据见evaluation/per_frame.csv，不能只看改善案例。', '',
              '## 5. 解释及采用边界', '',
              '实际结果：200轮训练帧正确95/100、精确率98.96%，折外正确18/50、精确率72%。'
              '训练拟合已明显恢复，但跨牛差距仍在。相较原Pose，直接检测7帧改善、6帧退化，正确点总数不变；'
              '无点兜底增加2个正确点，也增加2个错误点，未通过既定采用门槛，DO_NOT_ADOPT。'
              '这解释了新检测器30轮零输出的部分原因，但不能倒推出原Pose全部漏检均由训练不足或过拟合造成。', '',
              '本轮判断必须同时考虑训练是否拟合、折外是否改善，以及误检是否增加。'
              '即使点精度改善，也尚未证明整段可观测温度、左右身份连续性与逐呼气相位可靠。'
              '完整RR未重算，不能给出新的RR R²或用点匹配F1代替事件F1。', '',
              '数据来自已复核久福的12个牛号，三折隔离只防止本次训练/评价同牛重叠，'
              '不能恢复这批数据最初的独立测试身份。所有方法开发均按回顾性披露。', '',
              '证据：protocol_before_training.json、prediction_seal.json、frame_predictions.csv、'
              'fold_checkpoints.csv、training_fit_diagnostic/以及evaluation/。']
    (OUT / '训练长度诊断报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Report and full-frame diagnostic figures written.')


if __name__ == '__main__':
    main()
