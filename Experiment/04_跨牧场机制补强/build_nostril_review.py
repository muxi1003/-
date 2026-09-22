"""Build a prediction-blind native-coordinate annotation package."""
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_transfer import HERE, HOLDOUT, read, write_csv, write_json
from probe_pose_mechanism import ROOT, load_bgr


def main():
    frames=read(ROOT/'frame_manifest.csv')
    package_id=hashlib.sha256((ROOT/'frame_manifest.csv').read_bytes()).hexdigest()
    payload=dict(package_id=package_id,frames=[])
    for row in frames.itertuples():
        payload['frames'].append(dict(frame_id=row.frame_id,window_id=row.window_id,
            source_seconds=row.source_seconds,width=row.width,height=row.height,
            image=f'frames/{row.frame_id}.png',image_sha256=row.image_sha256,
            priority=row.target_seconds in [0,10,20]))
    template=(HERE/'nostril_review_template.html').read_text(encoding='utf-8')
    (ROOT/'index.html').write_text(template.replace('__PACKAGE_JSON__',json.dumps(payload,ensure_ascii=True)),encoding='utf-8')
    write_json(ROOT/'annotation_package.json',payload)
    write_json(ROOT/'annotation_blank.json',dict(schema_version=1,package_id=package_id,
        prediction_overlay_shown=False,records={}))
    (ROOT/'README.md').write_text('''# 鼻孔位置与可见性复核

直接打开同目录index.html；不用安装软件，也不用重新数呼吸。页面不加载算法坐标或预测次数。

1. 默认优先36帧：每个既有久福复核窗口的0、10、20秒附近原帧。全部集合为72帧，含5、15、25秒。先完成36帧可得到初步定位证据；72帧完整后再报告全部抽样结果。
2. 在原画面上拖动一个矩形范围，页面以其内接椭圆表示鼻孔开口。只包住能辨认的鼻孔区域，不画整个鼻子。A/B只是两个区域编号，不要求左右解剖命名。
3. 选择两侧可见、仅一侧可见、不可辨认或无法确定。看不清不能猜坐标；不可辨认/不确定时写原因。不要把看到牛头等同于鼻孔可辨认。
4. 填写复核人，点保存本帧并下一帧。数字1/2切换A/B；A/D切换前后帧；滚动和缩放用于放大原图。输入框内不响应快捷键。
5. 完成后导出JSON到C:/Users/muxi/Desktop/实验/鼻孔复核/，将实际文件路径发给Codex。浏览器本地保存不是正式提交；换浏览器或清理缓存前请导出。导入JSON可恢复当前批次。

所有标记保留原始图像像素坐标。人工椭圆仅是解剖开口的近似区域，不是像素级分割金标准。该复核是单观察者、隐藏本页预测的补充参考，不能追溯宣称从未看过算法结果或双盲。

## 文件用途

- frames：72张固定时间采样的原图，未叠加预测；源文件与时间见frame_manifest.csv。
- index.html：可离线打开的人工复核页面；不发网络请求。
- annotation_package.json / annotation_blank.json：成员及空白导出格式，不包含伪造标记。
- pose_probes.csv / pose_keypoints.csv：504次诊断预测，不能当人工参考；复核前不建议查看。
- phase_audit：已有R3事件与真实有限温度极值的诊断结果，不是新人工真值。
- figures：给研究讨论用的诊断图，人工复核前不建议查看。

收到实际JSON后运行上级evaluate_nostril_reference.py --input <实际JSON> --out <新结果目录>，计算可见帧漏检、关键点落入人工区域比例、归一化定位误差和圆ROI落在人工椭圆外的像素比例。它不会训练模型或覆盖原R2/R3。
''',encoding='utf-8')
    points=read(ROOT/'pose_keypoints.csv')
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    for fid in ['R3-01_t05','R3-02_t10','R3-10_t00']:
        im=load_bgr(ROOT/'frames'/f'{fid}.png')[:,:,::-1]
        fig,axes=plt.subplots(1,3,figsize=(10,7),layout='constrained')
        for ax,variant in zip(axes,['baseline','rotate_minus30','clahe_l']):
            ax.imshow(im)
            pp=points[points.frame_id.eq(fid)&points.variant.eq(variant)&points.confidence.ge(.5)&points.inside_native]
            for p in pp.itertuples():
                ax.add_patch(plt.Circle((p.x,p.y),20,fill=False,color='#00ffff',lw=1.3))
                ax.plot(p.x,p.y,'+',color='#00ffff',ms=7)
                ax.text(p.x+12,p.y-8,f'{p.keypoint}: {p.confidence:.2f}',color='white',fontsize=8,
                    bbox=dict(facecolor='black',alpha=.6,edgecolor='none'))
            ax.set(title=f'{variant}\n{len(pp)} high-confidence points');ax.axis('off')
        fig.suptitle(f'{fid}: all predictions mapped back to unchanged native image\n20px diagnostic circles, NOT human anatomy labels',fontsize=11)
        fig.savefig(figdir/f'{fid}_pose_comparison.png',dpi=160);plt.close(fig)
    frozen=[]
    for f in frames.itertuples():
        folder=HOLDOUT/'external_transfer/20260914_v2/windows'/f.window_id
        path=folder/'temperatures.csv'
        if not path.exists():continue
        temp=read(path);mapping=read(folder/'map.csv')
        ix=int(np.argmin(abs(mapping.target_time_seconds-f.target_seconds)))
        t=temp.iloc[ix]
        for k,side in enumerate(['left','right']):
            if not np.isfinite([t[side+'_x'],t[side+'_y'],t[side+'_temp']]).all():continue
            frozen.append(dict(frame_id=f.frame_id,variant='frozen_roi',keypoint=k,
                x=t[side+'_x'],y=t[side+'_y'],radius=t.adaptive_roi_radius,
                confidence=t[side+'_conf'],inside_native=True,source=t[side+'_source']))
    write_csv(ROOT/'frozen_roi_keypoints.csv',frozen)
    print('72 frames; priority36; prediction-blind package',package_id)


if __name__=='__main__':main()
