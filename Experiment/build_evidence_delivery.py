"""Package actual evidence only; never fills human events or estimates absent results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_common import ROOT, ABLATION, HOLDOUT, REFERENCE, sha256, write_csv, write_json, write_text


def main():
    out=ROOT/'delivery_20260914_v1'
    out.mkdir(exist_ok=False)
    external=HOLDOUT/'external_transfer/20260914_v2'
    results=external/'count_evaluation'
    metrics=json.loads((results/'metrics.json').read_text(encoding='utf-8'))
    ui=REFERENCE/'event_workspace/20260914_v3'
    ui_check=json.loads((ui/'ui_validation.json').read_text(encoding='utf-8'))
    if ui_check['status']!='PASS':
        raise ValueError('Annotation interface has not passed QA')
    events=pd.read_csv(ui/'reference_events.csv')
    spans=pd.read_csv(ui/'unobservable_intervals.csv')
    if len(events) or len(spans):
        raise ValueError('New human edits present; review before packaging')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    pair=pd.read_csv(results/'paired_count_results.csv')
    if len(pair):
        y,p=pair.truth_rr_bpm.to_numpy(float),pair.predicted_rr_bpm.to_numpy(float)
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        low=min(y.min(),p.min())-3;high=max(y.max(),p.max())+3
        axes[0].scatter(y,p,c='#137b8c',alpha=.65,s=34,edgecolors='white',linewidths=.4)
        axes[0].plot([low,high],[low,high],color='#c14545',linestyle='--',label='Identity')
        axes[0].set(xlim=(low,high),ylim=(low,high),xlabel='Manual RR (breaths/min)',ylabel='Predicted RR (breaths/min)',title=f"Frozen Jiufu: paired n={len(pair)}; R2={metrics['metrics']['rr_r2']:.3f}")
        axes[0].legend()
        mean=(y+p)/2;delta=p-y
        axes[1].scatter(mean,delta,c='#5178a5',alpha=.65,s=34)
        for value,label,color in [(metrics['metrics']['bias_bpm'],'Bias','#137b8c'),(metrics['metrics']['loa_low_bpm'],'Lower LoA','#c14545'),(metrics['metrics']['loa_high_bpm'],'Upper LoA','#c14545')]:
            axes[1].axhline(value,color=color,linestyle='--',label=f'{label}: {value:.2f}')
        axes[1].set(xlabel='Mean RR (breaths/min)',ylabel='Prediction - manual (breaths/min)',title='Bland-Altman (descriptive LoA)')
        axes[1].legend(fontsize=9)
        for ax in axes:ax.grid(alpha=.15)
        fig.savefig(results/'scatter_bland_altman.png',dpi=180);fig.savefig(results/'scatter_bland_altman.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    labels=['All released','Manual count available','Algorithm output','Paired primary']
    values=[metrics['n_released'],metrics['n_complete_reference'],metrics['n_algorithm_ok'],metrics['n_primary']]
    ax.barh(labels[::-1],values[::-1],color=['#278762','#167b8b','#b97836','#6c7f89'])
    for i,n in enumerate(values[::-1]):ax.text(n+2,i,str(n),va='center')
    ax.set(xlim=(0,300),xlabel='Windows (not independent cow count)',title='Jiufu first30: report coverage alongside accuracy')
    fig.savefig(results/'coverage.png',dpi=180);plt.close(fig)
    ablation=pd.read_csv(ABLATION/'runs/20260909_v1/metrics.csv')
    selected=ablation.loc[ablation.variant.eq('F1G0P1'),['cohort','analysis_set','n','rr_r2','rr_mae_bpm','rr_rmse_bpm']].copy()
    selected['role']='internal_development_fixed_r20_not_external_baseline'
    if metrics['metrics']:
        selected=pd.concat([selected,pd.DataFrame([dict(cohort='jiufu271',analysis_set='complete_reference_AND_algorithm_ok',n=metrics['n_primary'],rr_r2=metrics['metrics']['rr_r2'],rr_mae_bpm=metrics['metrics']['rr_mae_bpm'],rr_rmse_bpm=metrics['metrics']['rr_rmse_bpm'],role='frozen_transfer_adaptive_ROI_count_reference_only')])],ignore_index=True)
    write_csv(out/'evidence_metrics_different_scopes.csv',selected)
    status={
      'same_policy_signal_ablation':{'status':'COMPLETED_INTERNAL_DEVELOPMENT','cohorts':[73,49],'arms':8,'prediction_rows':976,'network_same_training_split_ablation':'NOT_RUN_NOT_CLAIMED'},
      'frozen_external_count_test':{'status':metrics['status'],'released':metrics['n_released'],'paired':metrics['n_primary'],'temperature_scope':'RF_pseudocolor_proxy_not_validated_absolute_Celsius','event_F1':None},
      'reliable_event_reference':{'tool_status':'COMPLETED_QA_PASSED','human_event_status':'AWAIT_HUMAN_EVENT_TIMES','events_submitted':0,'intervals_submitted':0,'double_blind':'NOT_IMPLEMENTED_USER_PAUSED','original_total_counts_preserved':True},
      'all_three_scientific_experiments_complete':False,
      'automatable_current_scope':'complete_if_final_manifest_and_reference_integrity_checks_pass',
      'never_claim':['all271_accuracy_from_paired_subset','event_F1_from_total_counts','zero_target_farm_calibration','new_network_ablation','camera_scale_confirmed']}
    write_json(out/'DELIVERY_STATUS.json',status)
    files=[
      (ABLATION/'runs/20260909_v1/RESULTS.md','8组同口径信号消融结果及边界'),
      (ABLATION/'runs/20260909_v1/metrics.csv','73/49/39分层指标'),
      (ABLATION/'runs/20260909_v1/paired_predictions.csv','976条消融预测明细'),
      (ABLATION/'runs/20260909_v1/factorial_contrasts.csv','配对模块效应及置信区间'),
      (external/'protocol_lock.json','外测前代码/参数/输入/运行环境锁'),
      (external/'prediction_seal.json','271窗预测封存，先于参考合并'),
      (external/'engineering_verification.json','全部时间映射/来源/峰重放验收'),
      (external/'prediction_windows.csv','全部预测与拒绝输出，不隐藏失败'),
      (external/'REVIEW_AND_DISPOSITION.md','同族审查发现及处置/剩余限制'),
      (results/'RESULTS.md','冻结代理信号迁移的总次数外测结果'),
      (results/'metrics.json','R2/MAE/RMSE/平均计数准确度/按牛聚类区间'),
      (results/'all_271_windows.csv','全部窗口参考状态、预测状态、排除原因'),
      (results/'paired_count_results.csv','主分析可计数且有预测的交集'),
      (results/'coverage.png','全部/人工可观察/算法可输出/主分析覆盖图'),
      (ui/'index.html','320窗人工逐事件标注页面，隐藏历史总数和算法'),
      (ui/'使用说明.md','怎样填写事件、区间及导出三表'),
      (ui/'ui_validation.json','真实本地视频播放和桌面/手机页面测试'),
      (REFERENCE/'event_workspace/browser_media_v1/verification.json','49个观看副本逐帧像素与时间等价核查'),
      (REFERENCE/'reference_tools.py','人工事件验证与一对一匹配评分，尚无事件输入'),
      (REFERENCE/'submissions/20260911_v2/delivery_artifact_manifest.csv','R1人工提交归档及原字节哈希')]
    if (results/'scatter_bland_altman.png').exists():files.append((results/'scatter_bland_altman.png','外测散点和Bland-Altman图，不把温度R2当RR R2'))
    write_csv(out/'文件用途与哈希.csv',[dict(path=str(p.resolve()),purpose=purpose,sha256=sha256(p),bytes=p.stat().st_size) for p,purpose in files])
    text='''# 三部分证据交付说明

## 01 同口径消融
已完成信号处理8组、73和49分别分析，共976条预测。文件在01_同口径消融/runs/20260909_v1。不得称作YOLO网络同训练划分消融；该网络实验未运行。73与49时长和来源不同，不合并样本。

## 02 冻结后的独立测试
久福271窗全部保留。完整结果在02_冻结独立测试/external_transfer/20260914_v2/count_evaluation；先锁定、推理、封存、验收，再合并R1人工总次数。主分析是人工可计数与算法有输出的交集，不是全部271窗的准确率。坏结果照实保存，不据此继续调参后仍称独立测试。温标未知，结论限定为RF映射代理信号，不证明真实摄氏温度准确；目标场独立标定图已使用，不是零目标场标定。

## 03 可靠事件参考
工具、320窗新R2三表、观看入口和评估程序已经完成；真实事件时刻和不可观察区间仍需人工，不能由算法填造。打开03_可靠事件参考/event_workspace/20260914_v3/index.html。原R1总次数不变，林甸历史非盲与久福R1对算法不可见分别披露。不能报告事件F1或双观察者一致性。

## 你仍需完成的事
在新R2工作区记录可确认呼吸的时刻，并标出看不清的区间；保存三个导出CSV以及JSON备份。若无法补齐，只能把当前研究写成总次数/RR验证加局限性，不能将“可靠事件参考”说成已经完成。未知的往届拍摄温标可以继续记未知，不要求你凭记忆补造。

当前没有要求你重填已有总次数，也没有启动网络重训、修改默认算法或补造人工标签。逐文件用途和SHA256见文件用途与哈希.csv；指标口径汇总见evidence_metrics_different_scopes.csv。它只是交付索引，持续项目总记录仍是docs/cow_rr_project_log.md。
'''
    write_text(out/'交付说明.md',text)
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
