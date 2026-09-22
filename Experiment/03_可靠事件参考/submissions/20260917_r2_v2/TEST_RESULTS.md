# 本轮软件回归检查

2026-09-17，Python：E:/real/anaconda/envs/plant_gpu/python.exe。

以下命令实际执行且退出码为0：

```powershell
& E:/real/anaconda/envs/plant_gpu/python.exe -m unittest discover -s Experiment -p test_event_submission.py
& E:/real/anaconda/envs/plant_gpu/python.exe -m unittest discover -s Experiment -p test_evidence_tools.py
```

新增7项与既有18项均通过，共25项。新增范围包括CSV/JSON数值等价、时点/备注变化拒绝、完整参考中拒判计FN、partial排除、非有限预测时长拒绝、声明次数不等于事件行数拒绝、拒判不允许填0计数。

所有测试为合成软件样例，不能作为奶牛真实事件或准确率证据。实际提交结构校验与数值复算分别保存，不将软件测试通过写成人工标注正确性认证。
