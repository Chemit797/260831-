# Biohub 源码恢复索引

Biohub 源码有两个不同快照；两者都在私有 Drive，且都不进入 GitHub。

1. `biohub-main-git-history-20260831.bundle`：先用 `git clone <bundle> biohub-main` 获得基线 `c602111` 和完整 refs。
2. `biohub-main-source-snapshot-20260831.tar.zst`：再解压到独立目录，查看 main dirty worktree 的 tracked/untracked 状态。
3. `biohub-codex-worktree-snapshot-20260831.tar.zst`：在另一个目录解压；它是 P915/R001/harmonic 的独立 dirty tree，不能覆盖第 2 步。
4. `biohub-promoted-artifacts-20260831.tar.zst`：只在查看模型、提交或门控证据时解压。

所有 `.venv`、原始竞赛 data、`configs/storage.env`、`.env*`、key/token 和 cache 都已明确排除。恢复后应重新配置自己的环境和竞赛凭据。
