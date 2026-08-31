# GOAI Active Learning Framework v2.2 Direct Semantic Multi-Seed Confirmation

状态：**预注册，尚未运行正式实验**  
预注册日期：2026-08-24 UTC  
实验 ID：`goai-al-direct-semantic-multiseed-v2.2`  
模型 ID：`GOAI-AL-V22-DIRECT-SEMANTIC-01`

## 1. 范围与停止条件

本合同是 v2.1 条件级主动学习框架完成后的单次关键结论确认，不是新的算法搜索。模型 ID 位于独立的 GOAI-AL、非 submission namespace；结果不改变 GOAI M0–M12 submission lineage 的冻结状态。本轮只比较已冻结的 Direct predictor、三种 acquisition 和两种固定表征。完成五种子 formal、独立审计、报告和台账后停止，不追加新 acquisition、rank sweep、超参数搜索或更多种子。

## 2. 数据与 condition-atomic split

- metadata SHA256：`9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef`
- proteome SHA256：`a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb`
- 数据协议：`goai-condition-atomic-v2.1`；split seed：42；缺失率阈值：严格 `< 0.80`。
- 4,920 个全局 condition；3,337 个 official-train condition；2,670 个 candidate-pool condition；667 个 interpolation condition；4,422 个蛋白。
- 清洗后的固定辅助 evaluation：`val_chem_only=503`、`val_strain_only=874`、`val_both=126`、`val_time=80`。
- candidate pool、interpolation 与四个 validation split 的 condition ID 必须两两互斥。evaluation ID 不可由 oracle reveal。

## 3. Control 合同

正式协议固定为 `pooled_exact_context_water_dmso`，`vehicle_column=null`。每个 treatment measurement 的 comparator 是八字段 exact context 内所有可用 Water/DMSO control measurements 的逐蛋白 log2 值直接等权均值；不得先按 control type 等权，也不得从 chemical、perturbation ID、source、plate、well 或其他字段推断 treatment vehicle。

released metadata 不提供 treatment-specific vehicle mapping，因此不能声称 vehicle-specific comparator。Water–DMSO sensitivity 仅是 post-hoc oracle audit，`acquisition_input=false`、`training_input=false`。

## 4. 表征合同

主表征 `semantic` 的准确含义是：pool-fitted identity one-hot、连续分钟 time，以及冻结的 target-free chemical/strain semantic blocks。消融 `identity` 是同一 identity one-hot 加连续 time，不含 semantic blocks。两者都只能读取公开 metadata 与冻结资产，必须显式声明 `response_used=false`。

五个 strain 身份是 public candidate mappings，并非 organizer-verified；该警告必须进入 manifest 和报告。DHY210 没有可接受语义，连续 block 保持零并显式标记 missing。由于本轮没有 shuffled semantic negative control，identity 与 combined 的差异只能解释为两套固定表征的预测差异，不得作 semantic causality 声明。

## 5. Predictor 与训练

所有策略和表征共享 Direct 4,422-output dropout MLP：hidden dimension 128、dropout 0.10、learning rate 0.001、AdamW weight decay 0.0002、batch size 512、target scale floor 0.05。formal 为 80 epochs、CUDA；smoke 为 2 epochs。每个 budget 从头训练，不 warm-start。所有 target statistics 和模型参数只由该轮 revealed candidate labels 拟合。

## 6. Acquisition、预算与种子

- 策略：`random`、`coreset`、MC-dropout `uncertainty`。
- formal seeds：42、43、44、45、46；initial=128、batch=128、evaluation checkpoints=128/256/512/1024、MC passes=8。
- smoke seeds：42、43；initial=32、batch=32、checkpoints=32/64/96、MC passes=2；只作诊断，不作科学判定。
- acquisition 在相邻 evaluation checkpoints 之间仍按固定 batch 稠密推进和重新训练，但只在注册 checkpoints 评分。
- 同一 run seed 的三策略共享完全相同的 initial IDs。model seed 和 acquisition seed 只依赖 `(run_seed, current_budget)`，不依赖策略或执行顺序。每个 seed×strategy 使用独立 candidate-only oracle，禁止重选和超预算。

## 7. Evaluation 与主判定

每个 checkpoint、full reference 和表征消融都对 interpolation 与四个固定 validation split 分别报告：delta RMSE/MAE、`delta_skill_zero`、pooled delta PCC、condition PCC median、protein PCC median、protein-wise R² median/mean/positive fraction及 evaluable counts。它们是本地 retrospective benchmark 指标，不是 organizer official score。

预注册主终点是 interpolation `delta_skill_zero` 对真实 budget 横轴的 normalized trapezoidal AULC。B80 使用同 seed full-pool reference、target fraction 0.80、单调 gain envelope 和区间内插；未达到时记 `null/not_reached`，禁止外推。

对 CoreSet 或 uncertainty，只有同时满足以下三项才判定优于 Random：五个 aligned seeds 的 paired AULC difference 均值 `>0`；双侧 95% t-CI 下界 `>0`；至少 4/5 seeds 同方向。否则保留 Random。不得以辅助 split、单预算或其他指标推翻该主判定。

## 8. 固定表征消融

每个 run seed 先生成一个与输入顺序无关的 nested random ID permutation。identity 与 combined semantic 必须使用相同 IDs、相同 model seed 和相同训练合同；formal 在 128、512 和 full-pool 预算比较，smoke 在 32、64 和 full-pool 比较。full semantic fit 可以复用同 seed full reference，但复用必须写入 receipt。

## 9. 信息边界与可复现性

acquisition 只能获得 candidate IDs、target-free descriptors、已揭晓 IDs、当前 predictor 和不确定性。未揭晓 labels 只存在于 `RetrospectiveOracle`；evaluation truth 仅供 scorer。每轮 receipt 不得序列化 labels、truth、predictions、model state 或 oracle values。

每个 seed 在隐藏 staging 目录完整运行，通过必需文件、schema、run identity 和 artifact SHA 校验后原子发布。resume 只跳过 hash-valid complete seeds，并从这些 seeds 重建聚合。正式运行身份必须锚定 config、规范、源码、输入、语义资产、特征矩阵、control 与 schedule；输出必须保存实际 source/config/spec snapshot。失败 staging 不得被视为完成。

## 10. 执行门与声明边界

formal 前必须完成：全套测试；真实数据无训练 load/cache/feature preflight；registry 与上下游 ledger 预注册；冻结源码/config/spec 哈希及实际 snapshot；两种子 smoke 和 exact resume 审计。若 smoke 导致任何源码或合同修改，须先追加预注册更正并更新哈希，再启动 formal。

formal 后必须完成：五个 seeds 全部 hash-valid；原始与聚合统计齐全；主判定、AULC/B80、初始集合公平、预算守恒、oracle 隔离和表征配对经独立审计；交付任务书要求的十部分技术报告；更新本地与上游 ledger。

本实验是 retrospective local proxy benchmark，不产生官方分数、submission、biological hit/discovery 或实验室因果结论。
