# V7 基线说明

## 制品

V7 的可恢复核心是三个完整运行目录：

| Seed | 原目录后缀 | 包含 |
|---:|---|---|
| 42 | `biostate-seed42-20260828-134253` | `checkpoint.pt`、validation predictions、逐蛋白指标、训练史、报告 |
| 43 | `biostate-seed43-20260828-134646` | 同上 |
| 44 | `biostate-seed44-20260828-135009` | 同上 |

每个 checkpoint 为约 296 MB；完整三种子结果包约 1.03 GB。三者都应保留，不能只留一个“最佳 seed”，否则无法检查稳定性。

## 输入合同摘要

* 官方训练元数据：`WAYB_WAYC_metadata_train_val.csv`；
* 官方训练蛋白表：`WAYB_WAYC_proteome_raw_train_val.csv`；
* 化合物 512D descriptor：`chemical_embeddings.csv`；
* 菌株 RAW4096 descriptor：`strain_embeddings.csv`。

详细文件哈希、敏感级别和解压位置见 `DATA_CONTRACT.md` 与根 `manifests/artifacts.yaml`。

## 解释边界

V7 的成功是完整配方复现信号：训练/数据/observer 的组合在当前验证中有效。它不自动证明任何单个信息源是原因。特别是 instrument/plate observer、模型容量和 descriptor 表征都可能影响比较，后续应使用受控消融来分辨贡献。

## 与 BCR 的关系

BCR 是一条已结束的、并非严格等条件架构归因的比较线。它使用 historical RAW4096，存在 DHY210 映射和 CalV2 fold-local 可识别性限制。可说“当前 released 场景中 V7 更好”；不可说“BCR 理论上被完全否定”。
