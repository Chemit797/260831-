# GOAI Active Learning Framework v2：冻结实验合同

本文件是第一阶段 benchmark 的预注册执行合同。它优先于旧版
`pilot_v1` 的配置和结果；旧版中的 `Impact`、`Hit Ratio`/`hit recall`
不属于本合同，也不得用于生物学 discovery 声明。

## 1. 问题与 query unit

一个 query 是一个 biological condition：

\[
c=(\text{strain},\text{chemical},\text{medium},\text{temperature},
\text{time in minutes}).
\]

真实字段为 `Strains`、`perturbation_no_concentration`、`Medium`、
`Temperature`、`pert_time` 和 `pert_time_unit`。公开 metadata 没有浓度字段，
因此本 benchmark 不声称区分剂量。`data_source`、`instrument`、
`Yeast_cell_plate` 和 `protein_well` 是测量上下文，不是 acquisition 轴。

一次 query 揭晓该 condition 的完整 4,422 维 matched-control response，而不是
单个蛋白。对 historical measurement row \(r\) 和蛋白 \(p\)：

\[
\Delta y_{rp}=\log_2 y^{treat}_{rp}-
\operatorname{mean}_{k\in C(r)}\log_2 y^{control}_{kp},
\]

其中 \(C(r)\) 是在 source、instrument、plate、strain、medium、temperature、
time 和 time unit 上 exact match 的 Water/DMSO controls。query label 是同一
biological condition 下可用 historical measurement responses 的逐蛋白均值。
metadata 没有足够字段可靠区分 biological 与 technical replicate，因此统一称为
“available matched measurement replicates”，不作未经证实的 replicate 类型声明。

Matched control 被预注册为每次实验的 assay overhead，只用于构造 retrospective
oracle label；所有已发布 split 中 exact-context controls 都保留为该 overhead，其
abundance 不进入 predictor 或 acquisition。Water 与 DMSO 在缺少
处理到溶剂映射的情况下遵循现有 GOAI exact-control 实现按 context 聚合，这一限制
必须在结果中保留。

## 2. Condition-atomic split

旧实现把 `split_final` 写入 group identity，导致 126 个 `val_time` query 中有 46 个
与 train biological condition 完全相同。v2.1 使用不含 split 的全局 `condition_id`，
并强制所有 pool/evaluation condition IDs 互斥。一个 condition 只要有 train provenance，
其 oracle response、replicate count 和 measurement summaries 就只能由 `split_final=train`
treatment rows 构造；重叠 validation labels 只保留 provenance 供 audit，随后丢弃，绝不
与 train label 合并。没有 train provenance 的 validation condition 只使用自身 split rows。

- `interpolation`：仅从官方 train biological conditions 按 metadata、固定 seed
  确定性留出 20%；不读取 response。留出后每个 strain、chemical、medium、
  temperature、time level 必须仍在 candidate pool 中出现。
- `candidate pool`：官方 train conditions 去掉 interpolation holdout；这是唯一可 query
  的集合。
- `val_chem_only`、`val_strain_only`、`val_both`、`val_time`：作为辅助 OOD/外推诊断，
  先剔除与 official-train condition 重叠的记录。`val_time` 不再被描述为严格的
  condition-cold test；报告必须写明清理后的实际支持。

Protein schema 仍复用 GOAI 已冻结的规则：仅用官方 train rows 的缺失率，保留
missing rate `< 0.80` 的 4,422 个蛋白。它是 benchmark 预先固定的输出 schema，
不随 AL round 改变。

## 3. 信息边界

公开给 acquisition 的只有：candidate IDs、target-free condition descriptors、当前
labelled IDs、当前 predictor 及其 predictive uncertainty。未 query response 保存在
`RetrospectiveOracle` 内，必须通过 `reveal(ids)` 获取；evaluation IDs 永远不可 reveal。

每轮所有 target mean/scale、missing-value imputation、response SVD basis 和模型参数
都只由当轮 revealed labels 拟合。全池 response PCA 只允许作为标记清楚的 post-hoc
feasibility audit，绝不能成为 acquisition 输入或 formal rank 选择依据。

Target-free condition encoder 只对 strain、chemical、medium、temperature 四个轴做
one-hot；`pert_time` 与 `pert_time_unit` 必须换算成单一连续 minutes 列。未知 time unit
直接报错，不得按乘数 1 静默处理。

## 4. 可替换模块

1. `GroupedDataset / BenchmarkSplit`：数据 schema、candidate pool 和固定 evaluation。
2. `RetrospectiveOracle / PoolState`：reveal 权限、预算状态和不可重复 query。
3. `Predictor`：统一 `fit / predict / uncertainty` 合同。
4. `Acquisition`：统一读取无标签 context 并返回 candidate local indices。
5. `BudgetSchedule`：显式 checkpoints，不用密集 budget sweep。
6. `Evaluator`：固定 split 和固定指标面板。

第一版正式 predictor 是 response-only rank-64 low-rank dropout MLP。每轮先在 revealed
response 上拟合标准化 SVD basis，再学习 condition descriptor 到 latent response；注册
目标是带 observed-value mask 的 natural-delta reconstruction loss。训练从头开始，不
warm start。`direct` 4,422-output MLP 只用于少量 representation feasibility 对照。

第一版 acquisition 仅包含：

- `random`：均匀无放回；
- `coreset`：在固定、target-free condition descriptor 中 farthest-first；
- `uncertainty`：同一 low-rank predictor 的 MC-dropout natural-delta variance。

三种策略共享相同 initial set、budget checkpoints、model seed function、epochs 和
evaluation set。不存在 `impact` acquisition。

## 5. Compute-aware protocol

### Smoke

- seed：42；
- initial budget：32；
- checkpoints：32、64、96；
- strategies：Random、CoreSet、Uncertainty；
- 2 epochs，2 次 MC draws；
- 目的：验证 split、oracle、reveal、预算、指标、每个 fitted budget 的 standalone
  label-free receipt 和 CLI，不作科学结论。

### Formal pilot

- seed：42（仅 1 seed）；
- initial budget：128；
- acquisition batch：固定 128；
- checkpoints：128、256、512、1,024；
- strategies：Random、CoreSet、Uncertainty；
- predictor：固定 rank-64 low-rank dropout MLP；
- 每次固定 batch 后都从头重训，因此 384、640、768、896 也执行 fit，但只在预注册
  checkpoints 128、256、512、1,024 做稀疏 evaluation；
- 每次 fit 80 epochs，uncertainty 使用 8 次 MC-dropout；
- full-pool 同 backbone reference 仅用于 achievable-improvement 参照；
- direct vs low-rank 只在少量共同 nested-random budgets 上比较，不做 rank grid search。

## 6. 固定评价面板

所有指标在 matched-control natural log2-delta 空间计算。truth 有限的位置要求 prediction
也有限，不能通过输出 NaN 逃避评分。

- `delta_rmse`、`delta_mae`：响应幅度误差；
- `delta_skill_zero = 1 - SSE(model) / SSE(delta=0)`：相对“无扰动响应”的
  GOAI-AL 本地 skill，1 为完美、0 等同 zero-response、负值更差；它不是官方指标；
- `pooled_delta_pcc`：所有可观测 condition-protein positions 展平后的 PCC；
- `condition_pcc_median`：每个 condition 跨 proteins 的 PCC，再取有限值中位数；
- `protein_r2_median`：每个 protein 跨 conditions 的 R²，再取有限值中位数；同时报告
  mean、正值比例与可评价蛋白数；
- `protein_pcc_median`：辅助 pattern generalization 诊断。

主 learning curves 在固定 `interpolation` split 报告；每个清理后的 official validation
split 分开报告。若给总览，只能明确标为 macro 或 pooled local diagnostic，不能冒充
官方 GOAI score。

对高值指标 \(M\)，normalized AULC 为实际 budget 横轴上的梯形积分除以预算跨度。
budget-to-target 使用：

\[
G(b)=\frac{M(b)-M_{initial}}{M_{full}-M_{initial}},
\]

以预注册的 80% achievable improvement 为 pilot target；使用单调 envelope 和线性插值，
未达到时明确写 `not_reached`，不外推。lower-is-better 指标先统一方向。

## 7. Feasibility 与验收

Post-hoc data audit 必须量化：query/replicate/control 粒度、split overlap、缺失率、
完整五维 tensor occupancy、主要 fibers 的完整度，以及 raw/standardized response 在
rank 8/16/32/64/128 下的描述性累计能量。它只回答下一阶段是否值得进入 low-rank /
tensor 路线，不作为本轮冠军选择。

本阶段只有在以下条件同时满足时完成：condition IDs 无跨界重叠；oracle 未揭晓访问失败；
所有策略预算守恒且无重复 query；Random learning curve、三策略比较、full reference、
低秩/tensor audit 均有可复现产物；AULC 是真实梯形积分；报告明确废弃 Impact/Hit Ratio、
区分本地 proxy 与官方分，并记录数据/config/source hashes、环境、seed 和实际命令。
