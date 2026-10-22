# 帧随机8:2训练对照

用户2026-10-19明确要求将所有已纳入帧随机打散。本实验与07牛号分组训练和08质量选择核验独立，不覆盖旧结果。

- 输入：同一批40542帧/199片段；旧标签7488帧、新复核33054帧，R3久福仍暂缓。
- 随机：frame_key排序后default_rng(20260929)一次打散，train32433/val8109，不按牛号或视频分组。
- 配置：标准YOLO11n-Pose，从原预训练权重重新训练；200轮上限、patience50、batch8、640图像、原优化参数，不接着旧best训练。
- 后续：训练best权重完成后自动推理199片段，沿用原四组温度/RR流程，输出曲线和事件，再做同窗旧/新训练策略比较。
- 解释：同牛和相邻帧跨train/val，val非独立test；完整RR窗口包含训练帧，不能作为未见数据确认。质量选择与补齐的归因限制仍沿用08，不因新模型结果改写。

## 文件

- `PROTOCOL.md`：实验界限及固定配置。
- `frame_split_manifest.csv`：每帧源路径、旧分组、新分组、图像/标签SHA256；准备完成后出现。
- `split_audit.json`、`segment_exposure.csv`：跨集合身份/视频及重复内容审计。
- `dataset/`：独立复制的图像标签及dataset.yaml。
- `freeze.json`：来源和新输入冻结清单；`resources.json`：本机资源，不安装或更新环境。
- `logs/status.json`：监督PID、训练子PID、阶段及每30秒心跳；阶段文字本身不是进程存活证据。
- `training_runs/yolo11n_pose_frame_random_20261019_v1/`：新训练日志、验证指标和权重。
- `rr_replay/`：冻结参考、时间及后续四组温度/曲线/预测输出。
- `comparison_metrics.csv`与`RESULTS.md`：只有训练及全部RR完成后生成。
- `COMPLETED.json`：上述所有阶段完成后才生成；没有此文件不能说训练与RR实验全部完成。

## 使用

在项目根目录执行只读核验（准备后）：

```powershell
& 'E:/real/anaconda/envs/plant_gpu/python.exe' 'Experiment/09_random_frame_pose_retrain/20261019_v1/run.py' verify
```

首次持久启动使用`launch.ps1`，其调用`launch_detached.py`，脱离工具终端并关闭标准输入继承；已有status或启动日志时拒绝重复启动，不自动恢复/重跑。子进程stage=train也可能仍处于全量文件preflight，须看到train.log中PREFLIGHT通过和实际训练输出才算进入训练轮次。查看存活状态：

```powershell
& 'E:/real/anaconda/envs/plant_gpu/python.exe' 'Experiment/07_all_video_pose_retrain/inspect_pipeline.py' 'Experiment/09_random_frame_pose_retrain/20261019_v1/logs/status.json'
```

若失败保留日志，先定位再另行决定恢复。不从本实验自动启动第二轮、修改默认权重或宣称稳定方法创新。
