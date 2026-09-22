"""Report the submitted human reference and the rejected full-cohort control."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
from analyze_transfer import read
from analyze_anatomy_reference import ROOT,OUT
from probe_pose_mechanism import load_bgr


def main():
    raw=json.loads((OUT/'reference_original.json').read_text(encoding='utf-8'))
    points=read(ROOT/'frozen_roi_keypoints.csv')
    for fid in ['R3-01_t00','R3-02_t10','R3-09_t00','R3-11_t20','R3-14_t10']:
        im=load_bgr(ROOT/'frames'/f'{fid}.png')[:,:,::-1];r=raw['records'][fid]
        pp=points[points.frame_id.eq(fid)]
        els=[(name,e) for name,e in r['regions'].items() if e is not None]
        bounds=[]
        for name,e in els:bounds.extend([(e['cx']-e['rx'],e['cy']-e['ry']),(e['cx']+e['rx'],e['cy']+e['ry'])])
        for p in pp.itertuples():bounds.extend([(p.x-p.radius,p.y-p.radius),(p.x+p.radius,p.y+p.radius)])
        fig,axes=plt.subplots(1,2,figsize=(11,7),layout='constrained')
        for ax in axes:
            ax.imshow(im)
            for name,e in els:
                ax.add_patch(Ellipse((e['cx'],e['cy']),2*e['rx'],2*e['ry'],fill=False,color='#15f238',lw=1.8))
                ax.text(e['cx'],e['cy']-e['ry']-9,name,color='#15f238',fontsize=10,ha='center',
                    bbox=dict(facecolor='black',alpha=.6,edgecolor='none',pad=1))
            for p in pp.itertuples():
                ax.add_patch(Circle((p.x,p.y),p.radius,fill=False,color='#00d5ff',lw=1.5))
                ax.plot(p.x,p.y,'+',color='#00d5ff',ms=6)
            ax.axis('off')
        axes[0].set_title('Full native frame',fontsize=11)
        if bounds:
            b=np.asarray(bounds);pad=80
            axes[1].set_xlim(max(0,b[:,0].min()-pad),min(im.shape[1],b[:,0].max()+pad))
            axes[1].set_ylim(min(im.shape[0],b[:,1].max()+pad),max(0,b[:,1].min()-pad))
        axes[1].set_title(f'Native-pixel detail | {r["visibility"]}',fontsize=11)
        fig.suptitle(f'{fid}: human opening (green) vs saved pipeline ROI (cyan)',fontsize=12)
        fig.savefig(OUT/'figures'/f'{fid}_native_detail.png',dpi=160);plt.close(fig)
    text='''# 36帧鼻孔人工复核：定位、ROI与全量对照结果

## 1. 本次参考已收到并完成评分

用户提交：C:/Users/muxi/Desktop/实验/nostril_reference_2026-09-22T06-26-48-117Z.json。
SHA256：2f7a8af055d1fab3b6454764818a2ecd87e8967645233414887f0fe8e981ba36。

36帧全部complete，恰好对应12个久福复核窗口的优先0/10/20秒抽样：20帧双侧、10帧单侧可辨认、6帧不可辨认，共50个人工鼻孔椭圆。没有修改坐标、备注和状态，没有把剩余36帧未提交参考填0。导出声明本页未展示预测；单观察者近似区域不是像素分割金标准，也不是新的独立牧场测试。R3-05_t00备注“可能会不清晰”保留，不擅自改变其双侧可见状态。

原件字节归档为reference_original.json，评分及像素对照均针对这一版，不再以“缺少全部位置参考”为阻点。

## 2. 可见时为什么还会漏检

在30个鼻孔可辨认的帧中，12帧没有检测框，共涉及19个人工鼻孔。也就是说，这12帧不是有框后关键点置信度0.5过滤，而是先在鼻子检测层中断。R3-01三个已标帧均属于可见但无框。

剩余18帧共31个人工鼻孔，其中18个获得落入人工区域的高置信点。合计50个人工鼻孔只有18个匹配，召回36%；27个预测点中18个匹配，精确率66.67%，其余9个未获人工区域支持。这里“正确”的固定判据是无序一对一匹配后预测中心位于人工椭圆内，不能把该数称为检测AP。

人体工标注开口的短轴直径缩放到640输入后，中位数约15.42像素，属于需要关注的细小结构，但尚未通过训练或尺寸因果对照证明“小目标”是唯一原因。常见旋转、裁剪、对比度变换的504次固定探针未解决检出覆盖；仅调关键点阈值无法挽救完全无框的这些帧。

![可见鼻孔但整帧无检测](figures/R3-01_t00_native_detail.png)

## 3. ROI的确与人工区域存在偏移，不能只看中心是否落入

实际保存的pipeline ROI有24个能与人工区域建立几何配对。其圆面积位于人工鼻孔椭圆之外的比例，中位数40.86%；保持原半径、只将圆心移到人工中心后降为8.27%。这是人工辅助的几何对照，不是自动算法已达到的效果，也不等于40.86%的像素必然没有任何呼吸信息。

将人工椭圆整体缩至80%或放至120%，对应的原圆外比例中位数约60.07%/21.96%，重新居中后约32.82%/0.04%。数值受人工区域大小影响，但“居中后减少区域外像素”的方向在这三种几何尺度下保持。

在同样24个配对上，原圆温度与人工椭圆温度的绝对差中位数约0.1965℃，均由同一冻结BGR RF与原图像素计算。它证明区域选择会改变当前代理温度值，不是绝对测温误差，也不能用三个稀疏时点推断整段呼吸曲线已经改善。

![中心接近仍有圆形范围差异](figures/R3-02_t10_native_detail.png)

![已有定位偏移与推断侧ROI](figures/R3-09_t00_native_detail.png)

## 4. 原流程的坐标补全没有等比例增加可靠信号

不能把之前整窗时间戳拒判与定位失败混在一起。只取允许原流程执行的同27帧，其中22帧可见、36个人工鼻孔：

| 结果来源 | 输出点 | 匹配人工区域 | 未匹配/区域外点 |
|---|---:|---:|---:|
| 原始YOLO高置信点 | 21 | 15 | 6 |
| 原流程最终保存ROI | 29 | 16 | 13 |

补全过程增加8个ROI点，但只增加1个匹配点、7个未获人工区域支持的点。R3-11_t20被标为不可辨认，但原流程仍保存1个detected与1个low_conf_inferred坐标。因此，不仅补全可能失败，直接检测也会在不可靠画面产生坐标。

![不可辨认帧仍存在输出ROI](figures/R3-11_t20_native_detail.png)

## 5. 已进行全量修复对照，但结果不支持采用

针对上述证据，单独试验“只保留source=detected温度，其余来源置缺失”，其余融合、插值与峰规则全保留。预测器不读取人工坐标或事件；规则设计受到本次人工复核启发，因此是回顾性对照，不是未接触外测。271个登记窗全部处理，原55个拒判保留，216个原输出的基线峰逐窗重放一致，173个计数配对不变；没有通过减少样本提高R²。所有有输出窗均存在直接温度，未触发无直接温度回退。

| 同口径评价 | 原冻结 | 移除非直接ROI温度 |
|---|---:|---:|
| R2：173配对RR R² | 0.419629 | 0.320727 |
| R2：173配对RR MAE / 次·min⁻¹ | 5.156069 | 5.768786 |
| R2：216完整参考事件F1 | 0.492010 | 0.487731 |
| R3：7配对RR R² | 0.346154 | 0.269231 |
| R3：11完整参考事件F1 | 0.362832 | 0.401786 |

R2 MAE增加0.612717次/min，按牛号聚类重采样2000次95%区间[0.149383,1.094146]。R2少了83个FP，却也少了42个TP，完整计数反而更差。R3局部事件F1升高不能掩盖全R2退化。决定DO_NOT_ADOPT，默认算法不变。

“原图无高置信点才尝试下方75%裁剪”的已有探针组合也未采用：36帧正确点18→18，仅多一个未匹配点。这些负结果说明不能靠统一删补全点或加一个裁剪兜底解决问题。

## 6. 哪些极值是真呼气：本轮能够和不能够回答的部分

N062已按R3原事件时间输出逐事件极大/极小值对应，严格9窗117事件中34个附近无算法有效温度、28个有温度但无0.3秒内极值、36个仅极大值接近、5个仅极小值接近、14个二者均接近。无温度不等于人不可见，本轮已确认这种区分很重要。

新的36帧解剖参考是0/10/20秒稀疏采样，不是覆盖每个呼气峰的逐帧位置参考，不能把它线性扩展到整个视频、将人造轨迹当成观察到的温度，或宣称已验证每次生理呼气。已有R3对应结果保留，原R2不替换，不拟合统一时间平移。

## 7. 下一步与完成边界

本轮真正缩小了问题范围：一部分信息在鼻子检测框阶段丢失；已有坐标还存在偏移，圆形区域覆盖也会混入人工开口之外的像素；盲目坐标补全并未可靠解决，但一刀切删掉它又损害完整RR。

接下来应针对“可见性受约束的局部重定位/ROI修正”建立自动候选，并首先用这36帧验证坐标和区域指标，再检查完整曲线与全量R2/R3。检测器完全无框的问题仍需单独的跨场定位改进，不能用修圆半径冒充修好。当前50个近似椭圆可用于机制核验，但不是已完成的新检测器训练集或独立测试。

目标已从缺参考状态恢复推进，尚未完整解决。用户本次优先36帧已经全部处理，无需重填；剩余36帧未提交保持独立待复核状态，不因此作废本次结果。

## 8. 验证与文件

- 原文件与归档SHA256一致，36个成员正好为优先集合；状态与区域数量校验通过。
- summary.csv、frame_results.csv、matched_region_results.csv：按人工区域的固定评分。
- annotation_audit.csv、anatomical_stage_summary.csv、timestamp_executed_same_frames.csv：定位断点和同执行口径。
- ROI_geometry_sensitivity.csv、human_ROI_temperature_control.csv：位置/范围及同图RF对照，人工辅助、不作为自动预测。
- ../direct_roi_control_v1：271窗单候选、216基线重放、预测封存与独立评分，不采用。
- 3项新增针对性单元测试通过；参考归档、50区域像素阈值与271预测封存核验通过。人体区域RF筛选已统一为原代码严格>20℃；实查所有人工区域中无恰等于20的预测像素，所以该实现修正不改变本次控制结果。

旧报告与原标注保留，新结论以本报告及明确样本口径为准。
'''
    text=text.replace('人体工标注','人工标注').replace('人体区域RF','人工区域RF')
    (OUT/'人工鼻孔复核结果.md').write_text(text,encoding='utf-8')
    print('Anatomical submission report and native detail figures written')


if __name__=='__main__':main()
