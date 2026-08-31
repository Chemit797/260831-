# 全量研究备份：第二波迁移（进行中）

> 启动时间：2026-08-31 UTC
> Drive 根目录：`10_全量研究备份_20260831/`
> 本页性质：**执行中的范围与恢复说明，不是“已验证收据”。**已验证的 17 个精选包仍以 `MIGRATION_RECEIPT_20260831.md` 和 `manifests/` 为准。

这第二波的目的，是在原有 4.073 GiB 精选恢复集之外，把用户自己的大体量研究历史（runs、OOF、模型、网络盘资料）尽可能保住。它不是把服务器所有内容不加选择地复制走；尤其不包含别人的项目，也不包含用户决定可日后重新下载的 Biohub 竞赛原始数据。

## 容量口径与当前总览

本轮锁定的逻辑范围约为 **307.5 GiB**。这是当前执行计划的管理口径，表示要保留的用户个人研究树的源端逻辑量；它不是已上传量、不是已校验量，也不等于 Drive 实际计费量。硬链接、稀疏文件、压缩归档和上传过程中的变化都会使最终字节数不同。

**307.5 GiB 不包含下文列出的个人回收站候选。**这些候选将作为独立优先清单处理，不能在没有记录的情况下混入“第二波已完成”的结论。

| Drive 目录 / 文件 | 源范围 | 估计逻辑量 | 当前状态 | 恢复角色 |
|---|---|---:|---|---|
| `01_Project_研究工作区/` | `/home/chenyuming/Project` 的研究树（按排除规则） | 60.62 GiB | **正在直接上传** | 本地 Project 的广泛历史快照；不取代当前最小 V7 包 |
| `02_Biohub_原始数据与历史/durable_archive/bulk-biohub-durable-full-20260831.tar.zst` | `biohub_root_fallback/durable` | 6.02 GiB 压缩档案 | **已验证**：`rclone check --checksum` 报 `0 differences`（1 matching file） | Biohub 的完整 durable 历史证据；保留硬链接，冷备读取 |
| `03_Omics_GPU_历史资料/01_chenyuming_go-ai-nightly/` | `/mnt/Omics_GPU/chenyuming/go-ai` | 30.79 GiB | **正在直接上传** | GO-AI 夜间运行、模型、结果的历史回溯层 |
| `03_Omics_GPU_历史资料/02_DiscoBAX-pilot_il2_3seed_archive/bulk-discobax-pilot_il2_3seed-full-20260831.tar.zst` | GeneDisco 外部软链接目标 `DiscoBAX/pilot_il2_3seed` | 114 MiB 源，压缩后约 35.7 MiB | **已验证**：archive 上传后 `rclone check` 报 `0 differences`（1 matching file） | IL-2 pilot 的完整日志/结果快照；不是公开 GeneDisco 数据缓存 |
| `03_Omics_GPU_历史资料/03_goai-rna-transfer-external/` | RNA transfer 外部输入 | 836 MiB | **已验证**：`rclone check --checksum` 报 `0 differences`（5 matching files） | M9.6/RNA 历史输入与 OOF 回溯 |
| `04_VirtualCODEX_完整研究档案/` | 用户个人 VirtualCODEX 根（排除 `mousebrain/`） | 约 206.19 GiB | **已写入首批约 6.3 GiB；当前可续传暂停** | 用户个人的完整网络盘历史；恢复时按相对路径按需取用 |
| `04_VirtualCODEX_完整研究档案/CYM_DD/FAXP2.0Pro_two_stage_20260831/` | 原先活跃的 FAX 训练目录；已包含在上一行 206.19 GiB 内 | 1.158 GiB / 103 files | **已验证**：训练进程结束后，`rclone check --checksum` 报 `0 differences`（103 matching files） | 可直接恢复的 FAX 终态快照 |

表中的 FAX 行是 VirtualCODEX 的子集，**不能再与 206.19 GiB 相加**。剩余全树曾恢复为低并发上传并已写入首批约 6.3 GiB；实测它会明显挤占 Project 与 Omics GO-AI 两条高优先级链路，因此当前以可续传方式暂停，待任一主链路完成后立即恢复。它始终使用同一套排除规则，绝不另起无过滤的全盘 copy。

早期一次聚焦 FAX copy 曾把同一来源的一份**重复、非规范副本**散落在 Drive 的 `04_VirtualCODEX_完整研究档案/CYM_DD/` 父层（而不是其 `FAXP2.0Pro_two_stage_20260831/` 子目录）。该重复副本先保留以避免在紧急期做远端删除；未来恢复和验收只认上表的嵌套 canonical 路径，父层散件不可当作独立项目。

## 硬性边界：未来人和智能体不得改写

以下边界来自用户的明确决定，不是“有空再评估”的建议：

1. **绝不上传 `mousebrain/`。**它是其他人借用用户磁盘完成的项目，不属于用户的个人迁移范围。VirtualCODEX 上传规则在源根层级排除此目录；远端检查时也不得把“缺少 mousebrain”当作漏传。
2. **绝不上传 Biohub 竞赛原始数据。**`/home/chenyuming/biohub_root_fallback/data/` 与所有 `data.partial*` 均不进入 Drive、GitHub、临时 archive 或回收站迁移。用户会在未来需要时按竞赛来源自行下载。此前已启动的不完整 `data/` Drive 副本（329.7 MiB）已移除，不能把它当作可恢复数据集；如未来必须盘点该源根，使用 `runbooks/bulk-biohub-hard-excludes.txt`，但该文件本身不授权广泛 copy。
3. **Biohub 仅额外保留 `durable/` 的完整压缩档案。**它与第 2 条不矛盾：`durable/` 是用户自己的历史实验目录，不是 Kaggle 原始比赛数据。完整 archive 已通过远端 checksum；先前的 `durable/` 逐文件直接 copy 是未完成尝试，应标记为 `superseded`，不得用于恢复。
4. **绝不上传认证与私钥材料。**包括 `auth.json`、`secrets/`、`.env*`、`.kaggle/`、`kaggle.json`、credential/token/secret JSON、API key 文件、`.pem`、`.key`、`id_rsa*`、rclone 配置与 OAuth token。新机器重新登录。
5. Project 中的 `.venv/`、`venv/`、`.conda-env/`、bytecode/test cache、Git object store、既有迁移 staging、控制仓库自身，以及 `/mnt/Omics_GPU/chenyuming/.pnpm-store` 都是可重建或已另有收据的对象，不进入本轮直接树 copy。

这五条优先于“尽可能传得多”的任何临时想法。新智能体若只看到 Drive 目录而没有本仓库，必须先读取本页和根 `AGENTS.md`，不能基于目录名自行补传排除项。

## 传输方法与状态含义

### 直接目录 copy

`Project`、Omics GO-AI 与 RNA transfer 使用可续传的 `rclone copy`。这样不需要制造数百 GiB 临时 tar，单文件失败可重试，也便于从 Drive 按相对路径取回。Project 内的外部软链接不自动跟随；有价值的 GeneDisco 目标另行归档，一个 GO-AI rebuild 外部链接目标为空。

### 压缩 archive

Biohub `durable/` 有约 34 万级小文件并跨目录硬链接。逐文件 copy 预计耗时过长且不能自然保留硬链接，故使用 `tar.zst` 完整档案。GeneDisco 的 `pilot_il2_3seed` 同样改用一个小型 `tar.zst`，以尽快保存结果目录而非被众多小文件拖慢。

Biohub archive 的本地文件已生成：

```text
bulk-biohub-durable-full-20260831.tar.zst
size:    6,469,548,564 B
sha256:  32b25295a29113dc7e88bcb7867149cc949ebc66a0c6afa56e6f7990e13d49a3
```

这份 Biohub archive 的 SHA-256 是**本地 archive 的身份信息**；它现已完成远端 `rclone check --checksum`（`0 differences`，1 个 matching file），可写为 `verified`。DiscoBAX archive 也已完成同项校验。恢复任何 archive 时均先校验 SHA-256 和 `zstd -t`，再解压到独立目录，保留原有相对路径，不要直接覆盖新机器工作区。

### 状态术语

| 术语 | 含义 | 不能据此得出的结论 |
|---|---|---|
| 正在上传 | 传输已启动、可继续或重试 | 不表示 Drive 中已有完整可恢复副本 |
| 已传完，待验 | rclone copy 已返回/末尾数据已完成，但尚未做源端与远端一致性检查 | 不表示已验证 |
| 已验证 | 完成大小/哈希或 `rclone check --one-way --checksum`，并记录目标和结果 | 才可作为删除源文件的必要证据之一；仍不授权删除 |
| 暂缓 | 为带宽优先级停止或不启动某一部分 | 不表示排除或完成；重启时仍要套用同样边界 |

验收命令的逻辑形式如下；实际 remote 名称在新机器自行配置，绝不复制旧 rclone 配置：

```bash
rclone check <source> <drive-target> --one-way --checksum
```

对 archive，先核对 archive 本地 SHA-256，再以对应 Drive 文件执行 checksum 一致性检查。每次通过后才把真实字节数、SHA-256、Drive 相对路径和验证时间写入 `manifests/` 与迁移收据。

## 活跃 FAX 实验的两阶段保护

`CYM_DD/FAXP2.0Pro_two_stage_20260831` 在迁移开始时仍可能写入，因此一次 upload 不可能自动成为最终一致快照。执行顺序固定如下：

1. 现在优先重试该目录的第一份快照，先让当前已经生成的文件落到 Drive；
2. 在训练停止、目录静止后，对**同一源和同一目标**再运行一次 `rclone copy` 补齐变更；
3. 对静止版本执行 checksum 检查，记录完成时刻和源端最后修改时间；
4. 不使用 `--ignore-existing`，否则可能错过被训练过程更新的同名文件；也不使用 `sync` 或任何会删除目标文件的命令。

本次实际完成了第 2–3 步：检查时训练进程已结束，canonical 嵌套路径包含 103 个文件、1.158 GiB，源—Drive checksum 为 0 差异。因此它现在可标作**已验证终态快照**，而不是仅“活跃快照”。

## 个人回收站研究恢复候选（不计入 307.5 GiB）

`~/.local/share/Trash` 不能整树上传：它混有其他人的内容、系统临时项及本轮生成的无效临时 archive。下面只列出已辨认为用户本人研究工作的**定向候选**；上传前后均不清空回收站、不删除源项。

| 候选源项 | 量级 | 为什么优先 | Drive 目标 / 当前状态 |
|---|---:|---|---|
| `producers*`（一组被删除的 GO-AI run 输出） | 9.439 GiB | 夜间/score-sprint 的 OOF、prediction 等结果，可能不在当前 Project 直传树内 | canonical 为 `00_ARCHIVES/trash-go-ai-producers-full-20260831.parts/` 的 11 个 900 MiB 分卷；**已验证**：`rclone check --checksum` 报 `0 differences`（11 matching files）。完整 archive SHA-256：`198391ae…ed8acc0` |
| `Go-AI-Optimal` | 218.6 MB | 被删除的项目 Git 工作树及未提交的经典模型报告 | `00_ARCHIVES/trash-go-ai-optimal-full-20260831.tar.zst`；**已验证**（SHA-256 `9f58cd9f…05525ee`，远端 0 differences） |
| `phasea_smoke_ntHGJ9` | 238.7 MB | V7/calibration smoke 结果，与当前 runs 的哈希不完全相同 | `00_ARCHIVES/trash-phasea-smoke-full-20260831.tar.zst`；**已验证**（SHA-256 `85be56f3…2216e8`，远端 0 differences） |
| M12 两个候选目录 | 约 390 MB | 可能含保留包之外的原始 CSV/运行证据 | `00_ARCHIVES/trash-goai-m12-raw-candidates-full-20260831.tar.zst`；**已验证**（SHA-256 `3bc613bc…c1d626`，远端 0 differences） |

**明确排除：**本轮曾生成但已取消的 `bulk-biohub-data-full-20260831.tar.zst` 是为了误启动的 Biohub 原始 `data/` 备份而产生的临时文件；它不是研究资产，不属于上述候选，也绝不能被重新上传。对回收站执行操作时只能指定上表的精确目录，禁止使用“复制整个 Trash”或“清空 Trash”。

早期的逐文件目标 `01_go-ai_producers_recovered/`、`02_Go-AI-Optimal/`、`03_phasea_smoke_ntHGJ9/` 是因代理小文件阻塞而留下的不完整尝试，**不是 canonical 恢复输入**；在紧急期先保留但不使用。分卷恢复时，在独立目录按名称排序重组：`cat part-*.part > trash-go-ai-producers-full-20260831.tar.zst`，随后校验上表完整 archive SHA-256、执行 `zstd -t`，再解压。

## 新机器如何读取第二波内容

日常研究仍应先用 `MIGRATION_RECEIPT_20260831.md` 中的 17 个精选包恢复 V7/AL。第二波是“完整回溯层”：只有在需要某个历史 run、原始 OOF、未晋级 checkpoint、网络盘日志或 Biohub durable 证据时，才按本页的相对路径从 Drive 下载。

1. 先查 `MIGRATION_STATUS.md` 与本页，确认目标是否已到 `已验证`；正在上传或待验的对象不可作为唯一副本。
2. 对直接目录，下载所需的最小子目录到新机器的单独恢复位置，再根据对应 README、ledger 或日志判断能否复用。
3. 对 `*.tar.zst`，先取得 manifest/收据中的 SHA-256，执行 checksum 与 `zstd -t`，列出 archive 内容，再解压到独立目录。
4. 不要把历史 OOF、旧 checkpoint 或 `durable/` 的候选实验自动当作当前 V7 输入；当前主线仍是成熟 V7 上的扰动主动学习。
5. 若发现一个 Drive 路径在文档中不存在，先把它当作未归档散件；创建清单和验证记录后才可作为研究依据。

## 未来执行者的检查清单

- [ ] 读取本页、根 `AGENTS.md`、`MIGRATION_STATUS.md`，而不是猜测上传边界。
- [ ] 确认 `mousebrain/`、Biohub `data/`、`data.partial*` 与取消的 Biohub data tar 不在 source list、filter、archive 或 Drive 目标中。
- [ ] FAX 训练静止后完成第二次增量 copy；其余 VirtualCODEX 全树恢复时使用同样的 mousebrain 排除。
- [ ] 每个顶层对象完成 `rclone check --one-way --checksum`；archive 另做 `zstd -t`/`tar -t`。
- [ ] 将验证后的字节数、SHA-256、Drive 路径、时间、命令结果追加到 `manifests/` 和正式收据。
- [ ] 未经用户再次明确授权，绝不删除源树、Trash 候选、缓存或服务器文件。
