# GO-AI V7 + Active Learning 工作说明

开始任何该目录相关任务前，按以下顺序阅读：

1. `CURRENT_STATE.md`
2. `V7_BASELINE.md`
3. `ACTIVE_LEARNING_PLAN.md`
4. `DATA_CONTRACT.md`
5. `DECISIONS.md`
6. `NEXT_STEPS.md`
7. 根目录 `manifests/artifacts.yaml`

## 绝对规则

* 不得把 V7 三 seed 的结果与 Direct-MLP AL v2.2 说成一个已经集成的系统。
* 不得使用隐藏 test proteome、错误对齐的 CSV 行、或把 `NaN` 当作 0。
* 每次实验只改变一个主要因素；数据边界、fold、seed、预算与评价协议必须可比。
* 对于任何大文件，先找到 artifact ID，再下载/校验；不要依赖旧服务器绝对路径。
* BCR 是已完成比较，默认不重启或扩展；若要重审，先阅读其限制与 `DECISIONS.md`。
* 新结果必须记录假设、输入版本、唯一变化、主/辅指标、失败模式和下一步。
