# 标注页面快捷键更新

用途：给原20260914_v3/index.html增加空格播放/暂停、Q确定峰、W不确定峰、E起终点交替、A/D前后视频。不改变事件定义、人工CSV、窗口来源/时长或浏览器缓存键；已存在鼠标控件继续可用。

index_before.html保存编辑前页面；原20260914交付清单中的页面SHA对应这个历史快照，不再对应更新后的live index.html。旧交付审核不静默重写。

keyboard_validation.json记录编辑前后SHA、三份标注CSV哈希及隔离Edge浏览器中的功能测试。测试使用临时软件事件，关闭前清除隔离缓存，没有向人工CSV或用户当前浏览器写入测试标注。模板event_workspace_template.html和当前页面逻辑一致。

运行验证：node Experiment/03_可靠事件参考/test_event_keyboard.cjs。此脚本只操作隔离测试浏览器，不复算呼吸率或重新运行视频推理。
