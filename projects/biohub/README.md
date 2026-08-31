# Biohub Kaggle 档案

Biohub 是独立竞赛作品，不是 GO-AI V7 主线。它的目标是保留可理解、可复核的作品与过程，而非保留 95 GB 原始数据和每一次候选试验。

## 两份不能混合的源码状态

| 快照 | 基线 | 特点 | 恢复规则 |
|---|---|---|---|
| main worktree | Git HEAD `c602111` | 10 个 tracked 修改、约 1,179 个未跟踪文件 | 以 Git bundle + main snapshot 恢复 |
| Codex worktree | 同一基线 `c602111` | 21 个 tracked 修改、239 个未跟踪文件；P915/R001/harmonic 线 | 以独立 patch/untracked archive 恢复，**不可覆盖 main** |

主仓库没有 Git remote，所以 Drive 中的 Git bundle 是重要的可恢复基线。`configs/storage.env`、`.env*`、凭据/密钥文件必须从所有源代码快照中排除。

## 精选竞赛制品

保留 E002 b1 15ep 模型、R001 两模型与其回执/metrics、四个历史提交 CSV 与提交/门控记录、E003 校准资料。四个 CSV 不等于四个被接受的提交：至少一项正式接受、一项门控阻止、一项随后接受、一项超时跳过；恢复时必须阅读同包的 receipt/日志，而非从文件名猜状态。

## 明确不迁移

* 82 GB 原始 Kaggle 数据；
* 约 11 GB `durable/` 的重复候选工作目录；
* 7.6 GB `.venv`；
* 全量 GEFF/OOF 与可重建 cache。

这些对象只保留竞赛来源、可追溯路径、指标和选择理由。
