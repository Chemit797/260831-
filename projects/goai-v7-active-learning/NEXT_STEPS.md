# 下一步清单

## 恢复后第一天

- [ ] 下载并校验 `goai-v7-active-data-20260831` 与三 seed baseline 包。
- [ ] 在新路径下运行数据合同检查，确认 `sample_ID`、mask、descriptor ID 映射。
- [ ] 下载/阅读 AL v2.2 formal 与 geometry 结果，列出目前 acquisition 的确切输入/输出。
- [ ] 将 V7 硬编码旧路径替换为配置或 artifact-root 环境变量。

## 开始新实验之前

- [ ] 写一个 V7-to-AL interface note：候选单元、可见标签、V7 输出、acquisition、预算、指标。
- [ ] 写最小 smoke experiment 的 preregistration：一个 seed、一个 batch 预算、random 对照。
- [ ] 定义 leakage tests 与 zero/shuffle/plate-instrument guardrail。
- [ ] 为每个新 run 建立目录、配置快照和 ledger 行。

## 不做的事

- [ ] 不直接把 BCR 大型 OOF 重新搬回 active worktree。
- [ ] 不把旧 stage-1 的模型编号当作新实验默认起点。
- [ ] 不在没有基线对照和固定预算的情况下开大规模搜索。
