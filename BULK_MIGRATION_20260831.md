# 全量研究备份：第二波迁移（进行中）

> 启动时间：2026-08-31 UTC
> Drive 根目录：`10_全量研究备份_20260831/`
> 目标：在原有 4.074 GiB 精选恢复包之外，保留尽可能完整的个人研究原始树、runs、模型、OOF、数据与历史证据。

## 当前范围

| Drive 目录 | 源范围 | 估计逻辑量 | 状态 |
|---|---|---:|---|
| `01_Project_研究工作区/` | `/home/chenyuming/Project` 的研究树 | 60.62 GiB | 正在上传 |
| `02_Biohub_原始数据与历史/data/` | `biohub_root_fallback/data` | 81.59 GiB | 正在上传 |
| `02_Biohub_原始数据与历史/durable/` | `biohub_root_fallback/durable` | 8.95 GiB | 正在上传 |
| `03_Omics_GPU_历史资料/01_chenyuming_go-ai-nightly/` | `/mnt/Omics_GPU/chenyuming/go-ai` | 30.79 GiB | 正在上传 |
| `03_Omics_GPU_历史资料/02_DiscoBAX-pilot_il2_3seed/` | GeneDisco 外部软链接目标 | 114 MiB | 正在上传 |
| `03_Omics_GPU_历史资料/03_goai-rna-transfer-external/` | RNA transfer 外部输入 | 836 MiB | 正在上传 |
| `04_VirtualCODEX_完整研究档案/` | 用户个人 VirtualCODEX 档案（不含 mousebrain） | 约 206.19 GiB | 正在上传 |

合计约 **389 GiB** 的第二波范围。实际 Drive 占用以逐文件上传后大小为准；硬链接和稀疏文件可能使其与逻辑量不同。

## 有意排除

1. `mousebrain/`：用户已明确说明它是其他人使用其磁盘完成的项目，**绝不上传**。上传规则已在源根目录层面硬性排除它，且远端检查未发现该目录。
2. 所有认证材料：`auth.json`、`secrets/`、`.env*`、`.kaggle/`、`kaggle.json`、credential/token/secret JSON、API key 文件、`.pem`、`.key`、`id_rsa*`。
3. Project 中可重建的环境和重复临时树：`.venv/`、`venv/`、`.conda-env/`、bytecode/test cache、Git object store、此前已验证上传的迁移 staging、控制仓库自身。
4. `/mnt/Omics_GPU/chenyuming/.pnpm-store`：可重装的依赖缓存。

Project 内的外部软链接不随 `rclone copy` 自动跟随；其有价值目标（GeneDisco pilot logs/results）已作为 `03_Omics_GPU_历史资料/02_DiscoBAX-pilot_il2_3seed/` 单独迁移。一个 GO-AI rebuild 外部链接目标当前为空。

## 活跃实验的处理

`CYM_DD/FAXP2.0Pro_two_stage_20260831` 正在写入。当前全树 copy 是第一份可中断恢复的快照；训练停止后，应对同一源与目标再运行一次 `rclone copy`，补齐变更文件。不要使用 `--ignore-existing`。

## 传输与验收

采用直接目录 copy，而不是制作数百 GiB 临时 tar：它不额外占用服务器稀缺的本地磁盘，单个文件失败可重试，传输中断可续传。每个顶层目录完成后将执行：

```bash
rclone check <source> <drive-target> --one-way --checksum
```

对于硬链接密集的 Biohub `durable/`，Drive 按普通文件保存会额外占用空间；这在当前 4.95 TiB 可用容量下可接受，换来可断点续传和逐文件恢复。

## 新机器如何使用

日常研究仍优先使用 `MIGRATION_RECEIPT_20260831.md` 中的 17 个精选包。全量树是“完整回溯层”：只有需要某个历史 run、原始 OOF、未晋级 checkpoint 或完整 Kaggle 数据时，才从这里按相对路径下载。不要把它和当前 V7 的最小恢复集混为一谈。
