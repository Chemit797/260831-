# GOAI 条件级主动学习 v2.1 正式试验技术报告

## 执行结论

**保留 Random 作为下一阶段默认主动学习基线；不提升单 seed 的 CoreSet 或
MC-dropout uncertainty，也不提升 rank-64 low-rank backbone。** 在主
`interpolation` 任务的预算 1,024 处，Random 的本地 `delta_skill` 为
0.153704，高于 CoreSet 的 0.150921 和 Uncertainty 的 0.140233；实际预算轴
AULC 也以 Random 最高。三者的 80% achievable-improvement 目标均未达到。
同时，使用完全相同 nested-random IDs 的 direct predictor 在 128、512、2,670
三个预算均优于 rank-64。因此，本次结果没有提供 CoreSet 或 MC uncertainty
提高样本效率的证据，也不支持把 rank-64 设为下一阶段默认表征。

这是单 seed、回顾性、matched-control natural log2-delta 空间中的**本地 GOAI-AL
proxy**。`delta_skill` 在本文中仅是产物字段 `delta_skill_zero` 的简称，不是官方
GOAI 指标、submission 或 leaderboard score；本文不作生物学 discovery 声明。
正式证据来自不可变目录
[`results/pilot_v2_formal-20260824-v21`](results/pilot_v2_formal-20260824-v21/manifest.json)，
其 manifest SHA-256 为
`1c41562578da8dc4cf00b41a7265c28e0c71872159849c44ee253440056062a5`。

## 范围与冻结协议

冻结合同见 [`FRAMEWORK_SPEC.md`](FRAMEWORK_SPEC.md)，实际执行合同见
[`manifest.json`](results/pilot_v2_formal-20260824-v21/manifest.json)。一个 query 是

`(strain, chemical, medium, temperature, pert_time, pert_time_unit)`

定义的 biological condition；一次 query 揭晓该 condition 的完整 4,422 蛋白
matched-control response，而非单蛋白。公开 metadata 无浓度字段，故本 benchmark
不区分剂量。全局共有 4,920 个 condition；官方 train 3,337 个，经 metadata-only、
seed 42 的 level-preserving 20% 留出后，唯一可 query 的 pool 为 2,670，固定
`interpolation` 为 667。清洗后的辅助 validation 分别为
`val_chem_only=503`、`val_strain_only=874`、`val_both=126`、`val_time=80`；固定
assignment 可见
[`split_assignments.csv`](results/pilot_v2_formal-20260824-v21/split_assignments.csv)。

正式协议为单 seed 42；三策略 Random、CoreSet、Uncertainty 共用相同的 128 个
deterministic initial queries；固定 batch 128，在 128、256、384、512、640、768、
896、1,024 均从头 fit，仅在预注册的 128、256、512、1,024 checkpoint 评价。每次
fit 80 epochs，Uncertainty 用 8 次 MC-dropout；模型 seed 为
`(42 * 1000003 + current_budget) mod (2^31 - 1)`。2,670 full-pool reference 使用
同一 rank-64 backbone；direct/rank-64 只在共同 nested-random 128、512、2,670
比较，不做 post-hoc rank sweep。完整 fit 台账见
[`model_fit_receipts.csv`](results/pilot_v2_formal-20260824-v21/model_fit_receipts.csv)。

## 泄漏与 control 审计

condition ID 不含 split。原发布数据中检测到 46 个 `val_time` condition 与 train
重叠，均从 validation 移除；其 52 条 validation treatment rows 也未进入 train
condition 的 response 聚合。因此，train oracle 恰由 **5,078 条 train treatment
rows** 构造，而不是把同 condition 的 validation labels 合并进去。全数据共 7,884
条 released treatment rows；剔除这 52 条后，oracle 聚合使用 7,832 条 measurement
rows。上述计数、46 个 ID 清单和 provenance 见
[`data_audit.json`](results/pilot_v2_formal-20260824-v21/data_audit.json)。

Water/DMSO control 按 source、instrument、plate、strain、medium、temperature、time、
time unit exact match，在 log2 abundance 中取均值并从 treatment 扣除。正式审计记录
16,776 个 control links、956 个实际使用的独立 control measurements；每个 treatment
匹配 1--3 个 control，中位数 2。共有 1,478 个 treatment measurements 只能匹配到
跨 split control，其中包括 12 个 train measurements。这是冻结合同明确允许的 assay
overhead：control abundance 不进入 predictor 或 acquisition，control 也不是 query
candidate。不过，当前数据没有 treatment 到 Water/DMSO vehicle 的显式映射，故现实现
按 exact context 合并可用 Water/DMSO；这是一项必须保留的 control 语义限制。

信息边界审计结果如下：

- pool 与五个 evaluation sets 的 condition IDs 互斥；evaluation IDs 不可由 oracle
  `reveal`。
- Acquisition 只收到 public ID、target-free descriptor、已标注 ID 和可选的 predictor
  uncertainty；[`acquisitions.csv`](results/pilot_v2_formal-20260824-v21/acquisitions.csv)
  仅有 `strategy, seed, round, selection_type, rank_in_batch, budget_before,
  budget_after, condition_id` 八列，无 response、target 或 oracle 列。
- 每轮的 response mean/scale、missing-value imputation、SVD basis 和模型参数只由当轮
  revealed labels 拟合；每策略使用隔离 oracle。低秩 spectrum 与 tensor audit 是
  `acquisition_input=false` 的描述性审计，未用于选点或 rank 选择。
- 固定蛋白 schema 为 4,422；4,920 x 4,422 condition-protein positions 的缺失率为
  **14.1606%**。truth 有限处要求 prediction 也有限。

## 架构、指标与解释边界

Target-free encoder 对 strain、chemical、medium、temperature 做 one-hot，并将 time 与
unit 换算为一个标准化连续分钟列；在 2,670 pool metadata 上 fit 后共 46 维。每轮
rank-64 模型只在已揭晓 response 上拟合 per-protein 标准化与 response SVD basis，再由
dropout MLP 从 condition descriptor 预测 latent response，以 observed-value mask 的
natural-delta reconstruction loss 训练，并从头初始化。CoreSet 在固定 descriptor 上做
farthest-first；Uncertainty 使用同一 predictor 的 MC-dropout natural-delta variance。
协议与 feature/information contracts 均记录在
[`manifest.json`](results/pilot_v2_formal-20260824-v21/manifest.json)。

全部指标在 matched-control natural log2-delta 空间计算，正式明细见
[`active_metrics.csv`](results/pilot_v2_formal-20260824-v21/active_metrics.csv)：

- `delta_rmse` 是所有可观测 condition-protein position 的均方根误差，越低越好。
- 本文 `delta_skill` 指本地字段 `delta_skill_zero = 1 - SSE(model) / SSE(delta=0)`；
  1 为完美、0 等同 zero-response、负值更差。它不是官方指标。
- `condition_pcc_median` 是逐 condition 跨 proteins 的 PCC，再取有限值中位数。
- `protein_r2_median` 是逐 protein 跨 conditions 的 R²，再取有限值中位数；它不同于
  pooled PCC，也不应解释为单蛋白生物学发现。

## 正式 QA 不变量

正式产物通过以下验收：

- `active_metrics` 为 **60 x 24**，`acquisitions` 为 **3,072 x 8**，
  `representation_metrics` 为 **30 x 22**，`model_fit_receipts` 为 **31 x 18**；另有
  **24 个 standalone round receipts**。相应证据见
  [`active_metrics.csv`](results/pilot_v2_formal-20260824-v21/active_metrics.csv)、
  [`representation_metrics.csv`](results/pilot_v2_formal-20260824-v21/representation_metrics.csv)
  与 [`manifest.json`](results/pilot_v2_formal-20260824-v21/manifest.json)。
- 三策略 initial set 完全相同；每策略恰有 **1,024 个唯一 query**，无重复、无越界；
  每轮预算守恒。global/model/acquisition seed 调度一致，评价 seed 公平。
- acquisition 表无禁用列；未 query response 未出现在 policy artifact；每一固定 batch
  都有 label-free receipt，四个 checkpoint 标记正确。
- manifest inventory 共 38 项：37 个外部可哈希产物的现场 byte count 与 SHA-256
  全部匹配、零失配；第 38 项 manifest 是 self-describing，其自身哈希已在本报告开头
  独立记录。
- 在 manifest 所记正式环境与 `PYTHONPATH=src` 下，自动化验证为 **20 tests passed**
  （另有 2 个 pandas deprecation warnings，不影响通过状态）。

## 主结果：interpolation learning curve

下表为预注册主 split 上的本地 `delta_skill`；128 处三策略相同是共享 initial set 的
预期结果。原始 60 行指标见
[`active_metrics.csv`](results/pilot_v2_formal-20260824-v21/active_metrics.csv)，汇总定义与
实际预算轴积分见
[`analysis_summary.json`](results/pilot_v2_formal-20260824-v21/analysis_summary.json)。

| Budget | Random | CoreSet | Uncertainty |
|---:|---:|---:|---:|
| 128 | 0.043628 | 0.043628 | 0.043628 |
| 256 | 0.066909 | 0.063461 | 0.064085 |
| 512 | 0.088395 | 0.088200 | 0.084477 |
| 1,024 | 0.153704 | 0.150921 | 0.140233 |

normalized AULC 是在真实 budget 横轴上做 trapezoidal integration 后除以预算跨度。
B80 的目标为从各策略共同 initial 值到同-backbone full reference 可实现改善的 80%，
使用 monotone envelope 与线性插值且不外推。

| 策略 | `delta_skill` AULC | 1,024 已实现 full gain | B80 |
|---|---:|---:|:---|
| Random | 0.099253 | 67.35% | `not_reached` |
| CoreSet | 0.097635 | 65.65% | `not_reached` |
| Uncertainty | 0.093120 | 59.11% | `not_reached` |

## 预算 1,024 与 full reference

full reference 是预算 2,670 的同-backbone low-rank fit，仅用于 achievable-improvement
参照，不是 acquisition policy。完整五 split reference 见
[`full_reference_metrics.csv`](results/pilot_v2_formal-20260824-v21/full_reference_metrics.csv)。

| 模型/策略 | Budget | 本地 `delta_skill` | Condition PCC median | Protein R² median | RMSE |
|:---|---:|---:|---:|---:|---:|
| Random | 1,024 | 0.153704 | 0.227165 | 0.066652 | 0.369723 |
| CoreSet | 1,024 | 0.150921 | 0.233053 | 0.065406 | 0.370331 |
| Uncertainty | 1,024 | 0.140233 | 0.169700 | 0.052797 | 0.372654 |
| Full low-rank reference | 2,670 | 0.207063 | 0.295127 | 0.116116 | 0.357878 |

CoreSet 的 final condition PCC 略高于 Random，但本地 skill、protein R² median 和 RMSE
均未胜出，且 AULC 更低；单 seed 下不能据一个辅助维度宣布 policy 优越。

## 清洗后 evaluation splits 的 final 本地 delta_skill

下表只使用 budget 1,024 的三策略和预算 2,670 的 full low-rank reference。这里的
“官方 validation”只描述数据 split 的来源，**不把本地 `delta_skill` 变成官方指标**。

| Split | CoreSet | Random | Uncertainty | Full low-rank reference |
|:---|---:|---:|---:|---:|
| interpolation | 0.150921 | 0.153704 | 0.140233 | 0.207063 |
| val_chem_only | -0.078710 | -0.116808 | -0.243763 | -0.119887 |
| val_strain_only | 0.043474 | 0.039906 | -0.016375 | 0.054299 |
| val_both | -0.016808 | -0.035568 | -0.156354 | -0.018573 |
| val_time | 0.161854 | 0.168879 | 0.162368 | 0.224421 |

这些 OOD 数值不得用于选 policy。encoder 对当前 labelled set 不支持的 categorical
columns 置零，因此未知实体本身没有身份信息：`val_chem_only` 的 503 个 condition 的
chemical 全不受 pool 支持，折叠为 96 个 descriptor signatures，产生 407 次碰撞；
`val_both` 的 126 个 condition 在 strain 和 chemical 两轴均不受支持，折叠为 24 个
signatures，产生 102 次碰撞；`val_strain_only` 的 874 个 condition 的 strain 全不受
支持，但其他字段组合使 874 个 signatures 仍保持唯一。故当前 OOD 是
**identity-blind lower bound**，不是有语义实体表示的真实外推比较。
这些计数按正式 [`manifest.json`](results/pilot_v2_formal-20260824-v21/manifest.json)
冻结的 input hashes、46 维 `feature_contract` 与清洗后 split contract 独立重算。

此外，清洗后的 `val_time=80` 已不再与 train condition 重叠，但原始 126 个中有 46 个
被移除，不能把旧 split 名称解释为未经限定的严格 condition-cold test。

## Direct 与 rank-64 表征对照

下表只比较共同 nested-random IDs 上的 `interpolation`；512 的 low-rank 数值来自独立
representation fit，不应与 AL curve 的 Random 512 行混用。原始 30 行见
[`representation_metrics.csv`](results/pilot_v2_formal-20260824-v21/representation_metrics.csv)。

| Budget | 表征 | 本地 `delta_skill` | Condition PCC median | Protein R² median | RMSE |
|---:|:---|---:|---:|---:|---:|
| 128 | Direct | 0.063980 | 0.172803 | -0.000797 | 0.388829 |
| 128 | Rank-64 | 0.043628 | 0.153086 | -0.001824 | 0.393033 |
| 512 | Direct | 0.155130 | 0.225465 | 0.074857 | 0.369412 |
| 512 | Rank-64 | 0.084643 | 0.188690 | 0.022754 | 0.384513 |
| 2,670 | Direct | 0.267194 | 0.373398 | 0.179176 | 0.344041 |
| 2,670 | Rank-64 | 0.207050 | 0.289985 | 0.115655 | 0.357881 |

Direct 在三个预算和四项指标上均胜出；因此不能以压缩或训练便利为由提升当前 rank-64
backbone。

## Tensor occupancy 与低秩 spectrum

官方 train 五维 tensor 的 observed/possible cells 为 **3,337/3,552**，occupancy
**93.9471%**。但 chemical fiber 仅 **4/96** 完整，complete fraction 4.1667%；高总体
occupancy 不等于每个 chemical fiber 完整。完整 fiber 表见
[`tensor_coverage.csv`](results/pilot_v2_formal-20260824-v21/tensor_coverage.csv)。

官方 train response 的 post-hoc randomized-SVD 累计能量如下；它是描述性 oracle
audit，未进入 acquisition，也不是 formal rank 选择。原值见
[`low_rank_spectrum.csv`](results/pilot_v2_formal-20260824-v21/low_rank_spectrum.csv)。

| Rank | Raw centered | Per-protein standardized |
|---:|---:|---:|
| 8 | 29.7553% | 31.1691% |
| 16 | 36.8774% | 38.6386% |
| 32 | 44.6422% | 46.2315% |
| 64 | 53.1685% | 54.2292% |
| 128 | 61.8744% | 62.2061% |

高 tensor occupancy 支持下一阶段探索 masked tensor completion；但 chemical-fiber
完整度很低，rank 64 仅解释约 53--54% 能量，且 direct 实测持续更好。因此这应是谨慎的
follow-up，而不是“tensor/low-rank 占优”的结论。

## 决策与下一步实验

1. 冻结 Random 为默认 acquisition baseline；当前单 seed 没有 CoreSet 或 MC
   uncertainty 提高样本效率的证据。
2. 不提升 rank-64 backbone。先冻结 direct predictor，或预先确定一个更高 rank 的
   factorized predictor；避免借本次正式结果做 post-hoc rank grid selection。
3. 为 chemical 与 strain 增加 target-free semantic descriptors，使未知实体在不读取
   response 的前提下仍可区分；在此之前不从 identity-blind OOD 结果选择 policy。
4. 显式建立 treatment 到 solvent/vehicle-specific control 的映射，再审计跨 split
   control 依赖，替代当前 Water/DMSO context 聚合假设。
5. 协议与表征稳定后才运行 multi-seed AL 比较，并预注册 seed 数、聚合统计与不确定区间。
6. 可并行设计 masked tensor completion follow-up，但须按 chemical fibers 报告完整度，
   并与冻结 direct baseline 在相同 IDs 上比较。

## 局限

- 只有 seed 42，策略间的小差值没有方差估计，不能推断稳健 superiority。
- 这是 historical measurement 的 retrospective oracle，不等同于前瞻实验成本、失败率或
  batch effects；metadata 也不能可靠区分 biological 与 technical replicate。
- 无浓度字段；control 缺 treatment-specific vehicle 映射；1,478 个 treatment 使用
  cross-split-only controls。control 虽被合同限定为 assay overhead，仍是解释边界。
- 14.1606% response 缺失；蛋白 schema 由官方 train 缺失率冻结，但不存在生物学机制
  模型。不得把 protein-level 指标解释成 discovery。
- 当前 OOD one-hot representation 对未知 chemical/strain identity-blind；负 skill
  主要是下界诊断，不能据此选 acquisition policy。
- low-rank spectrum 仅是 post-hoc 描述，rank 64 能量中等；tensor 高 occupancy 与
  chemical fibers 低完整度并存，均不支持路线支配性声明。
- Impact、Hit Ratio/hit recall 已废弃且正式产物中不存在；所有报告分数均是本地 proxy，
  不是官方 GOAI 分数。

## 复现与 provenance

正式执行命令与 [`manifest.json`](results/pilot_v2_formal-20260824-v21/manifest.json)
逐字符一致：

```text
'/home/chenyuming/Project/Biohub - Cell Tracking During Development/.venv/bin/python3.12' -m goai_al.experiment --config configs/pilot.yaml --output-suffix formal-20260824-v21
```

运行于 `2026-08-24T16:59:04.742492+00:00` 至
`2026-08-24T17:00:35.529272+00:00`；`PYTHONPATH=src`，
`CUDA_VISIBLE_DEVICES=0`，Python 3.12.13、NumPy 2.5.2、pandas 3.0.5、
PyTorch 2.9.1+cu126，单张 NVIDIA A100-PCIE-40GB。输入与配置哈希为：

| 对象 | SHA-256 |
|:---|:---|
| Metadata | `9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef` |
| Proteome | `a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb` |
| Config | `e8bd0ff11ba1b90b31d4118cac23d21e539bc788abe62150139765be9e4c3bf6` |

八个 source hashes、完整环境、38 项 artifact inventory 和 corrected smoke 的独立
non-scientific 记录见 [`MODEL_LEDGER.md`](MODEL_LEDGER.md)。正式结果的主要可审计入口为
[`analysis_summary.json`](results/pilot_v2_formal-20260824-v21/analysis_summary.json)、
[`data_audit.json`](results/pilot_v2_formal-20260824-v21/data_audit.json)、
[`active_metrics.csv`](results/pilot_v2_formal-20260824-v21/active_metrics.csv)、
[`full_reference_metrics.csv`](results/pilot_v2_formal-20260824-v21/full_reference_metrics.csv)、
[`representation_metrics.csv`](results/pilot_v2_formal-20260824-v21/representation_metrics.csv)、
[`tensor_coverage.csv`](results/pilot_v2_formal-20260824-v21/tensor_coverage.csv) 和
[`low_rank_spectrum.csv`](results/pilot_v2_formal-20260824-v21/low_rank_spectrum.csv)。
