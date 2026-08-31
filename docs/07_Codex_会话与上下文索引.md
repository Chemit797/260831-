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

## 归档包与阅读顺序

```text
01_recovery-kit/       rules、skills、worktree snapshot、visualizations、索引、小型一致性 DB 备份
02_live-context/       sessions、attachments、live SQLite 一致性备份
03_provider-history/   provider backup 中的会话/archived sessions，不含其 config
99_checksums/          文件目录、SHA-256、恢复说明
```

1. 先读 `00_请先读我_服务器迁移总蓝图.md`、`docs/01_研究总览.md`、对应项目 `CURRENT_STATE.md`。
2. 只有在要追溯某决定时，查询即将生成的 `session-file-catalog.tsv`：原路径、大小、mtime、SHA-256、归档包。
3. 解密/解压到独立、只读目录；按具体 session/附件定位。
4. 把经过确认的结论回写到 `DECISIONS.md` 或 ledger，避免未来再次依赖原始历史。

## 完整历史不等于“恢复记忆”

可让 Codex 恢复工作方式的核心（rules、skills、worktree、visualizations、索引、小型状态）约 56.7 MiB；完整历史约 7.07 GiB。前者用于可用性，后者用于证据回溯。会话本身可能包含研究路径、提示词或意外粘贴的敏感信息，因此只进入用户掌握密钥的私有加密 Drive，不进 GitHub。

## 一致性要求

迁移时 `logs_2.sqlite`、`thread_history_1.sqlite`、`state_5.sqlite` 等存在 WAL 写入。最终快照必须在 Codex 静默/退出后，使用 SQLite `.backup` 生成一致副本；不能只 tar 正在变化的 `.sqlite` 主文件。初始快照可用于尽早验证上传，最终增量才是可恢复快照。

## 永久排除

不归档/不上传：`auth.json`、`secrets/**`、所有 `config.toml*`、`official.config.toml`、旧 provider backup 中的 config、`~/.config/rclone/rclone.conf`、`~/.ssh/**`，以及 tmp/cache/IPC/daemon/安装 ID 等可重建状态。
