# 紧急迁移收据：关键资料已保全

> 完成时间：2026-08-31 UTC
> 状态：**紧急迁移范围已完成并逐包校验。**旧服务器上的原件没有被删除或移动。

这是清理服务器前最重要的一页：Google Drive 已有可以恢复当前研究和回看 Codex 历史的精选副本。未来的人或智能体应先读本仓库，不要直接把原始聊天档案当作研究说明书。

## 已验证的迁移量

| 范围 | 包数 | 字节 | GiB | 状态 |
|---|---:|---:|---:|---|
| 当前 V7、主动学习与历史研究精选制品 | 14 | 1,641,096,976 | 1.528 | 已上传并校验 |
| Codex 去凭据恢复与历史预快照 | 3 | 2,732,699,354 | 2.545 | 已上传并校验 |
| **总计** | **17** | **4,373,796,330** | **4.073** | **已完成** |

每个包都有本地 SHA-256，并在上传后执行了 `rclone check --one-way --checksum`；结果均为 `0 differences found`。完整逻辑 ID、Drive 相对路径、大小、哈希和用途见 [`manifests/artifacts.yaml`](manifests/artifacts.yaml) 与 [`manifests/checksums-20260831.sha256`](manifests/checksums-20260831.sha256)。

## Codex 的三份实际档案

这些 `.tar.zst` 包按用户在紧急迁移时的明确授权，直接存入用户管理的私有 Drive。它们不含登录凭据、OAuth token、`auth.json`、`secrets/`、任何 `config.toml` 变体、SSH 密钥或 rclone 配置。

| Drive 相对路径 | 大小 | SHA-256 前缀 | 读它是为了什么 |
|---|---:|---|---|
| `90_CODEX_PRIVATE_COLD_BACKUP/01_recovery-kit/codex-recovery-kit-preliminary-20260831.tar.zst` | 27,111,697 B | `67c2a86…` | rules、skills、worktree、visualizations、索引和小型状态数据库；恢复 Codex 工作方式。 |
| `90_CODEX_PRIVATE_COLD_BACKUP/02_live-context/codex-live-context-preliminary-20260831.tar.zst` | 1,865,558,550 B | `f780115…` | 当前 sessions、attachments、archived sessions 以及日志/线程历史的一致性 SQLite 备份。 |
| `90_CODEX_PRIVATE_COLD_BACKUP/03_provider-history/codex-provider-history-preliminary-20260831.tar.zst` | 840,029,107 B | `90fcfe3…` | 三份 provider migration backup 的会话历史；只在需要追溯旧上下文时打开。 |

这是在 Codex 仍在运行时的**预快照**：SQLite 由 `.backup` 制成一致副本，目录先复制到 staging 再压缩。它足以作为今晚清理前的恢复层；若旧服务器还有时间，关闭/静默 Codex 后再制作一个小型最终增量会更完美，但不是此收据的完成前提。

## 新人或新智能体如何开始

1. 克隆本仓库，按 [`AGENTS.md`](AGENTS.md) 的顺序阅读。
2. 先理解当前 GO-AI V7 主线：`docs/01_研究总览.md`、`projects/goai-v7-active-learning/CURRENT_STATE.md`、`NEXT_STEPS.md`。
3. 需要二进制文件时，按 `manifests/artifacts.yaml` 的逻辑 ID 从 Drive 下载，先运行 `sha256sum -c` 再解包。
4. 只有需要确认旧聊天、附件或某个历史决定时，阅读 `docs/06_Codex_完整归档与恢复指南.md` 与 `docs/07_Codex_会话与上下文索引.md`，并将档案解压到独立目录；**不要覆盖新的 `~/.codex`**。

示例（已配置对应 Drive remote 的机器）：

```bash
rclone copy "my_drive:90_CODEX_PRIVATE_COLD_BACKUP/01_recovery-kit/codex-recovery-kit-preliminary-20260831.tar.zst" ./restore/
cd ./restore
sha256sum -c <(printf '%s  %s\n' '67c2a86f729a2dbc2b09bdc39db15bf506abd721771284aa9ff6cd526b042242' 'codex-recovery-kit-preliminary-20260831.tar.zst')
mkdir inspect-codex-recovery
tar --use-compress-program=unzstd -xf codex-recovery-kit-preliminary-20260831.tar.zst -C inspect-codex-recovery
```

更完整的人类阅读路径、每类旧服务器文件的由来、取舍与恢复顺序，在 [`00_请先读我_服务器迁移总蓝图.md`](00_请先读我_服务器迁移总蓝图.md) 和 `docs/` 中。
