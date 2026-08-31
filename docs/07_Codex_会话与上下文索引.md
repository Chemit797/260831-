# Codex 会话与上下文索引

本文件是私有 Codex 档案的入口，不复制会话正文。新智能体不必、也不应先读原始 JSONL；它先读根仓库的结构化研究文档。

## 当前档案地图

| 类别 | 约计 | 归档目的 |
|---|---:|---|
| `sessions/` | 542 个文件，3.245 GiB | 当前原始会话历史 |
| `logs_2.sqlite` | 1.034 GiB | 应用日志/会话相关记录 |
| provider backups | 1.957 GiB | 旧 provider 迁移历史，可能与当前内容重叠 |
| `attachments/` | 102 项，0.485 GiB | 图片、PDF、CSV、ZIP 等会话附件 |
| `thread_history_1.sqlite` | 0.283 GiB | 本地线程历史索引 |
| `worktrees/` | 20.9 MiB | 重点为 Biohub 独立工作树 |
| `visualizations/` | 18.8 MiB | 生成图件 |

## 已上传归档包与阅读顺序

```text
01_recovery-kit/       rules、skills、worktree snapshot、visualizations、索引、小型一致性 DB 备份
02_live-context/       sessions、attachments、live SQLite 一致性备份
03_provider-history/   provider backup 中的会话/archived sessions，不含其 config
99_checksums-and-restore-notes/  文件目录、SHA-256、恢复说明
```

1. 先读 `MIGRATION_RECEIPT_20260831.md`、`00_请先读我_服务器迁移总蓝图.md`、`docs/01_研究总览.md`、对应项目 `CURRENT_STATE.md`。
2. 只有在要追溯某决定时，先用 `session_index`、`history` 及文件时间/大小定位候选 session；完整路径、SHA-256 与归档包以 `manifests/artifacts.yaml` 为准。
3. 下载、校验并解压到独立、只读目录；按具体 session/附件定位。
4. 把经过确认的结论回写到 `DECISIONS.md` 或 ledger，避免未来再次依赖原始历史。

## 完整历史不等于“恢复记忆”

可让 Codex 恢复工作方式的核心（rules、skills、worktree、visualizations、索引、小型状态）约 56.7 MiB；完整历史的去凭据输入约 7.07 GiB。前者用于可用性，后者用于证据回溯。本次压缩后的三包实际为 2.545 GiB，已按用户紧急授权进入其私有 Drive，不进 GitHub。

## 一致性要求

迁移时 `logs_2.sqlite`、`thread_history_1.sqlite`、`state_5.sqlite` 等存在 WAL 写入。本次预快照已对 SQLite 使用 `.backup` 生成一致副本，而不是只 tar 正在变化的主文件。若旧服务器仍可用，在 Codex 静默/退出后做最终增量会减少最后时段的缺口；它是增强，不是当前归档可用性的前提。

## 永久排除

不归档/不上传：`auth.json`、`secrets/**`、所有 `config.toml*`、`official.config.toml`、旧 provider backup 中的 config、`~/.config/rclone/rclone.conf`、`~/.ssh/**`，以及 tmp/cache/IPC/daemon/安装 ID 等可重建状态。
