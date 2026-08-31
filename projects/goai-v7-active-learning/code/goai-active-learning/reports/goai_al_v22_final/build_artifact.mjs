import { execFileSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const GENERATED_AT = "2026-08-24T21:16:02Z";
const TITLE = "GOAI 主动学习框架 v2.2 技术报告";
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const sqlPath = join(here, "sources", "goai_report.sql");
const sqlite = "/home/chenyuming/miniconda3/bin/sqlite3";

function query(table) {
  const output = execFileSync(
    sqlite,
    [":memory:", ".mode json", ".read " + sqlPath, "SELECT * FROM " + table + ";"],
    { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 },
  ).trim();
  return output ? JSON.parse(output) : [];
}

const datasetNames = [
  "headline_policy",
  "headline_data",
  "checkpoint_skill",
  "policy_decision",
  "data_scope",
  "control_contract",
  "genedisco_compare",
  "architecture",
  "compute_protocol",
  "full_metrics",
  "full_representation_skill",
  "paired_semantic_delta",
  "rank_energy",
  "direct_rank64",
  "tensor_feasibility",
  "metric_definitions",
];
const datasets = Object.fromEntries(datasetNames.map((name) => [name, query(name)]));

function md(...paragraphs) {
  return paragraphs.join("\n\n");
}

function textColumn(field, label) {
  return { field, label, type: "text" };
}

function numberColumn(field, label) {
  return { field, label, format: "number" };
}

function table(id, title, subtitle, dataset, columns, sortField, density = "spacious") {
  return {
    id,
    title,
    subtitle,
    dataset,
    sourceId: "report_snapshot_sql",
    defaultSort: { field: sortField, direction: "asc" },
    density,
    layout: "full",
    columns,
  };
}

const cards = [
  {
    id: "random_aulc",
    description: "五个对齐 seeds、预算 128–1,024 的 interpolation normalized AULC；为保留的默认 comparator。",
    dataset: "headline_policy",
    sourceId: "report_snapshot_sql",
    metrics: [{ label: "Random AULC ×1,000", field: "random_aulc_x1000", format: "number" }],
  },
  {
    id: "coreset_delta",
    description: "CoreSet 相对 Random 的配对 AULC 差；方向一致但 95% CI 穿零。",
    dataset: "headline_policy",
    sourceId: "report_snapshot_sql",
    metrics: [{ label: "CoreSet ΔAULC ×1,000", field: "coreset_delta_x1000", format: "number", signed: true }],
  },
  {
    id: "uncertainty_delta",
    description: "MC-dropout uncertainty 相对 Random 的配对 AULC 差；五个 seeds 均为负向。",
    dataset: "headline_policy",
    sourceId: "report_snapshot_sql",
    metrics: [{ label: "Uncertainty ΔAULC ×1,000", field: "uncertainty_delta_x1000", format: "number", signed: true }],
  },
  {
    id: "protein_outputs",
    description: "每个 condition 的 matched-control log2-delta response 宽度。",
    dataset: "headline_data",
    sourceId: "report_snapshot_sql",
    metrics: [{ label: "Protein outputs", field: "proteins", format: "number" }],
  },
  {
    id: "evaluation_conditions",
    description: "五个 non-reveal evaluation regimes 的 condition 总数；与 candidate pool 零交集。",
    dataset: "headline_data",
    sourceId: "report_snapshot_sql",
    metrics: [
      { label: "Evaluation conditions", field: "evaluation_conditions", format: "number" },
      { label: "Pool/eval overlap", field: "pool_eval_overlap", format: "number" },
    ],
  },
];

const charts = [
  {
    id: "checkpoint_skill_chart",
    title: "Interpolation response skill by budget and acquisition",
    subtitle: "Five aligned seeds at four preregistered checkpoints; budget 128 is the shared initial set",
    type: "bar",
    dataset: "checkpoint_skill",
    sourceId: "report_snapshot_sql",
    encodings: {
      x: { field: "budget_label", type: "ordinal", label: "Experimental budget" },
      y: { field: "mean_x100", type: "quantitative", label: "delta_skill_zero × 100", format: "number" },
      color: { field: "strategy", type: "nominal", label: "Acquisition" },
      tooltip: [
        { field: "mean", type: "quantitative", label: "Mean delta_skill_zero", format: "number" },
        { field: "sample_sd", type: "quantitative", label: "Sample SD", format: "number" },
        { field: "seed_count", type: "quantitative", label: "Seeds", format: "number" },
      ],
    },
    yAxisTitle: "delta_skill_zero × 100",
    valueFormat: "number",
    layout: "full",
  },
  {
    id: "representation_chart",
    title: "Full-budget response skill by evaluation regime and representation",
    subtitle: "All 2,670 candidate conditions, five paired seeds; Combined semantics includes identity and time",
    type: "bar",
    dataset: "full_representation_skill",
    sourceId: "report_snapshot_sql",
    encodings: {
      x: { field: "split", type: "nominal", label: "Evaluation regime" },
      y: { field: "mean_x100", type: "quantitative", label: "delta_skill_zero × 100", format: "number" },
      color: { field: "representation", type: "nominal", label: "Representation" },
      tooltip: [
        { field: "mean", type: "quantitative", label: "Mean delta_skill_zero", format: "number" },
        { field: "sample_sd", type: "quantitative", label: "Sample SD", format: "number" },
        { field: "budget", type: "quantitative", label: "Budget", format: "number" },
      ],
    },
    yAxisTitle: "delta_skill_zero × 100",
    valueFormat: "number",
    layout: "full",
  },
  {
    id: "rank_energy_chart",
    title: "Cumulative response energy by rank",
    subtitle: "Official-train response only; randomized SVD seed 42; post-hoc oracle audit, not acquisition input",
    type: "bar",
    dataset: "rank_energy",
    sourceId: "report_snapshot_sql",
    encodings: {
      x: { field: "rank_label", type: "ordinal", label: "Rank" },
      y: { field: "cumulative_energy", type: "quantitative", label: "Cumulative energy", format: "percent" },
      color: { field: "variant", type: "nominal", label: "Response variant" },
    },
    yAxisTitle: "Cumulative energy",
    valueFormat: "percent",
    layout: "full",
  },
];

const tables = [
  table(
    "data_scope_table",
    "Condition-level pool and evaluation contract",
    "Condition counts after global de-duplication and removal of train-overlap validation groups",
    "data_scope",
    [
      textColumn("role", "Role / split"),
      numberColumn("conditions", "Conditions"),
      textColumn("revealable", "Oracle revealable"),
      textColumn("purpose", "Purpose"),
    ],
    "role",
  ),
  table(
    "control_contract_table",
    "Matched-control contract and sensitivity",
    "Released metadata lacks treatment-specific vehicle mapping; the comparator is pooled assay overhead",
    "control_contract",
    [
      textColumn("item", "Contract item"),
      textColumn("value", "Reviewed value"),
      textColumn("interpretation", "Interpretation"),
    ],
    "item",
  ),
  table(
    "genedisco_table",
    "GeneDisco-style active learning versus GOAI",
    "Conceptual and interface-level borrowing; no GeneDisco or DiscoBAX code graft",
    "genedisco_compare",
    [
      textColumn("dimension", "Dimension"),
      textColumn("genedisco_style", "GeneDisco-style setting"),
      textColumn("goai_framework", "GOAI implementation"),
      textColumn("implication", "Design implication"),
    ],
    "dimension",
  ),
  table(
    "architecture_table",
    "Framework modules and extension contracts",
    "Frozen source-snapshot components used by the formal controller",
    "architecture",
    [
      textColumn("module", "Module"),
      textColumn("responsibility", "Responsibility"),
      textColumn("extension_contract", "Extension contract"),
    ],
    "module",
  ),
  table(
    "protocol_table",
    "Smoke and formal compute profiles",
    "Smoke is diagnostic only; scientific decisions use the five-seed formal profile",
    "compute_protocol",
    [
      textColumn("profile", "Profile"),
      textColumn("seeds", "Seeds"),
      textColumn("budget_schedule", "Budget schedule"),
      textColumn("model_training", "Model training"),
      textColumn("purpose", "Purpose"),
    ],
    "profile",
  ),
  table(
    "full_metrics_table",
    "Full-pool combined-representation metric panel",
    "Five-seed mean ± sample SD; full reference is empirical, not a theoretical upper bound",
    "full_metrics",
    [
      textColumn("split", "Split"),
      textColumn("rmse", "RMSE"),
      textColumn("mae", "MAE"),
      textColumn("skill", "Delta skill"),
      textColumn("pooled_pcc", "Pooled PCC"),
      textColumn("condition_pcc", "Condition PCC med."),
      textColumn("protein_pcc", "Protein PCC med."),
      textColumn("protein_r2_median", "Protein R² med."),
      textColumn("protein_r2_mean", "Protein R² mean"),
      textColumn("protein_r2_positive_fraction", "Protein R² > 0 fraction"),
    ],
    "split",
    "dense",
  ),
  table(
    "policy_decision_table",
    "Preregistered acquisition decision",
    "Normalized interpolation AULC, budgets 128–1,024; paired two-sided t interval with df=4",
    "policy_decision",
    [
      textColumn("strategy", "Strategy"),
      textColumn("mean_aulc", "Mean AULC ± SD"),
      textColumn("paired_delta", "Paired Δ vs Random"),
      textColumn("ci95", "95% CI"),
      textColumn("wins_losses", "Wins / losses"),
      textColumn("preregistered_gate", "Gate"),
      textColumn("final_status", "Decision"),
    ],
    "strategy",
  ),
  table(
    "semantic_delta_table",
    "Paired combined-minus-identity response skill",
    "Five aligned seeds; 95% t intervals are descriptive outside the primary policy endpoint",
    "paired_semantic_delta",
    [
      numberColumn("budget", "Budget"),
      textColumn("split", "Split"),
      textColumn("paired_delta", "Paired Δ skill"),
      textColumn("ci95", "95% CI"),
      textColumn("wins_losses", "Wins / losses"),
      textColumn("interpretation", "Descriptive reading"),
    ],
    "budget",
    "dense",
  ),
  table(
    "direct_rank_table",
    "Direct versus rank-64 on the same nested IDs",
    "v2.1 interpolation delta_skill_zero, seed 42 only; feasibility evidence, not a five-seed comparison",
    "direct_rank64",
    [
      numberColumn("budget", "Budget"),
      textColumn("direct_skill", "Direct"),
      textColumn("rank64_skill", "Rank-64"),
      textColumn("delta_direct_minus_rank64", "Direct − rank-64"),
      textColumn("evidence_scope", "Scope"),
    ],
    "budget",
  ),
  table(
    "tensor_table",
    "Tensor occupancy and chemical-fiber completeness",
    "Five condition axes excluding time unit, which is constant in released data",
    "tensor_feasibility",
    [
      textColumn("scope", "Scope"),
      textColumn("occupied_cells", "Occupied cells"),
      textColumn("occupancy", "Occupancy"),
      textColumn("chemical_complete_fibers", "Complete chemical fibers"),
      textColumn("implication", "Implication"),
    ],
    "scope",
  ),
  table(
    "metric_table",
    "Evaluation metric contract",
    "All response metrics use the matched-control log2-delta truth mask",
    "metric_definitions",
    [
      textColumn("metric", "Metric"),
      textColumn("definition", "Definition"),
      textColumn("interpretation", "Scientific question"),
      textColumn("direction", "Preferred direction"),
    ],
    "metric",
  ),
];

const manifestSources = [
  { id: "report_snapshot_sql", label: "GOAI-AL v2.2 reviewed report snapshot SQL", path: "reports/goai_al_v22_final/sources/goai_report.sql" },
  { id: "framework_v22", label: "Frozen GOAI-AL v2.2 framework specification", path: "results/direct_multiseed_formal-20260824-v22/source_snapshot/FRAMEWORK_SPEC_V22.md" },
  { id: "formal_manifest", label: "GOAI-AL v2.2 formal root manifest", path: "results/direct_multiseed_formal-20260824-v22/manifest.json" },
  { id: "data_audit", label: "GOAI condition/control/split audit", path: "results/direct_multiseed_formal-20260824-v22/data_audit.json" },
  { id: "control_audit", label: "Water–DMSO control sensitivity audit", path: "results/direct_multiseed_formal-20260824-v22/control_vehicle_sensitivity.csv" },
  { id: "formal_audit", label: "Independent artifact and statistical audit record", path: "reports/goai_al_v22_final/AUDIT_RECORD.md" },
  { id: "v21_representation", label: "GOAI-AL v2.1 Direct versus rank-64 comparison", path: "results/pilot_v2_formal-20260824-v21/representation_metrics.csv" },
];

const blocks = [
  { id: "title", type: "markdown", body: "# " + TITLE },
  {
    id: "technical_summary",
    type: "markdown",
    body: md(
      "## 技术摘要",
      "**框架已经按任务目标实现、正式执行并完成独立审计。** Query、oracle、control、split、预算、模型、acquisition、指标、原子恢复和产物哈希均有冻结合同；五个 formal seeds 的完整网格通过复算。",
      "**当前应保留 Random。** CoreSet 的 interpolation AULC 在 5/5 seeds 上方向为正，但配对 95% CI 穿过 0，未达到预注册替换门槛；MC-dropout Uncertainty 在 5/5 seeds 上低于 Random。到预算 1,024 时，15 条 seed×policy 曲线均未达到 B80，因此不外推预算。",
      "**结构性探索有清晰边界。** Chemical/strain combined representation 在 cold-start splits 上较好，但 full-budget interpolation 略差；rank-64 只解释约一半 response energy，且同 IDs 实测弱于 Direct。下一阶段可以探索 semantic-aware diversity 与 masked tensor predictor，但不能把本轮结果解释为 semantic causality、tensor 优越或 biological discovery。",
      "> 本报告中的分数均为 retrospective local proxy；不是 organizer official score、submission、Hit Ratio、hit discovery 或实验室因果结论。",
    ),
  },
  { id: "headline_metrics", type: "metric-strip", cardIds: ["random_aulc", "coreset_delta", "uncertainty_delta", "protein_outputs", "evaluation_conditions"] },
  {
    id: "metric_key",
    type: "markdown",
    sourceId: "framework_v22",
    body: md(
      "### 读图指标键",
      "- **Budget** 是已揭晓的 biological conditions 数量；一次 query 揭晓一个 condition 的整条 4,422-protein response。\n- **delta_skill_zero** = 1 − SSE(model) / SSE(delta=0)，越高越好；0 表示与无扰动响应基线等价。\n- **Normalized AULC** 是 delta_skill_zero 按 budget 轴的梯形积分再除以预算跨度，越高表示同等实验预算下学习更快。\n- **B80** 是首次达到 same-seed initial-to-full 可实现增益 80% 的预算；未达到就报告 not_reached，禁止外推。",
    ),
  },
  {
    id: "section_1",
    type: "markdown",
    sourceId: "framework_v22",
    body: md(
      "## 1. 问题已经收敛为“选择 condition，预测整条蛋白响应”",
      "一个 query 由五个 biological axes（strain、去除浓度后的 chemical identity、medium、temperature、perturbation time）与一个 time-unit normalization key 组成。Released data 的 time unit 恒为 minute，但仍保留在 canonical key 中。一次 query 揭晓一个 4,422 维 matched-control log2-delta response 及其 observation mask，而不是一个标量。",
      "运行循环为：L_t --fit--> f_t；Q_t = A(f_t, U_t, L_t)；Y_Qt = Oracle.reveal(Q_t)；L_(t+1) = L_t ∪ Q_t。主决策问题是：在相同 condition 预算和同一 predictor 下，哪一种 acquisition 能提高 interpolation response skill 的整条 learning curve，而不是事后找到较大的变化。",
    ),
  },
  {
    id: "section_2",
    type: "markdown",
    body: md(
      "## 2. 借用 GeneDisco 的主动学习骨架，不借用其 hit 目标",
      "GOAI 保留 pool-based active learning、轮次 acquisition、固定预算和 sample-efficiency 比较这些可迁移原则；真正改变的是 query 与 response 的统计对象。这里每次实验产生一个结构化 condition 的高维 masked proteomic fiber，并且 label 必须经过 matched-control 与 replicate aggregation。",
      "因此，GeneDisco/DiscoBAX 在本项目中是概念与接口参考，而不是代码嫁接。未来方法通过 Acquisition 或 Predictor adapter 进入同一 simulator；Impact、Hit Ratio 和“变化大等于 hit”的目标不进入正式链路。",
    ),
  },
  { id: "genedisco_evidence", type: "table", tableId: "genedisco_table", layout: "full" },
  {
    id: "section_3",
    type: "markdown",
    sourceId: "data_audit",
    body: md(
      "## 3. Query、Oracle 与 Split 已做 condition-atomic 隔离",
      "全局共有 4,920 个 unique conditions。Official train 的 3,337 个 conditions 被确定性拆成 2,670 个 candidate-pool conditions 与 667 个 interpolation conditions；另有 chemical、strain、both、time 四个 non-reveal regimes。原 released split 中 46 个 train/val_time condition 重叠被显式移除，52 条 validation treatment measurements 没有进入 train oracle aggregation。修复后 pool 与全部 evaluation IDs 的交集为 0。",
      "RetrospectiveOracle 只服务 candidate IDs，evaluation truth 只进入 scorer。Acquisition 只接收 public descriptors、revealed IDs、当前 predictor/uncertainty 和 budget state；round receipts 禁止 labels、truth、predictions、model state 与 oracle values。",
      "Oracle response 的聚合顺序被冻结：先对每条 treatment measurement 计算 matched-control log2 delta，再仅使用该 condition 被保留 split 的 treatment rows，按 protein 做 mean(skipna) 聚合。一个 condition 有 1–6 条可用 matched measurements，中位数为 1；replicate count 与 observation mask 同时保留用于审计。",
    ),
  },
  { id: "data_scope_evidence", type: "table", tableId: "data_scope_table", layout: "full" },
  {
    id: "section_3_control",
    type: "markdown",
    sourceId: "report_snapshot_sql",
    body: md(
      "### Control 是冻结的 assay overhead，不是 treatment-specific vehicle",
      "默认 comparator 是八字段 exact measurement context 内所有 Water/DMSO control measurements 的逐蛋白 log2 直接等权均值；不能先按 control type 等权，也不能从 chemical、perturbation ID、source、plate 或 well 推断 treatment vehicle。7,884/7,884 treatment measurements 都有 exact-context control，但 1,478 条只能依赖 cross-split controls，其中 official-train 12 条。",
      "Released metadata 不含 treatment-specific vehicle mapping，所以报告不称其为 vehicle-matched comparator。429 个双 control contexts 的 Water–DMSO global RMS 为 0.336909，按 treatment frequency 加权为 0.338644；这只是一项 post-hoc oracle sensitivity audit，未进入 acquisition 或训练特征。",
    ),
  },
  { id: "control_evidence", type: "table", tableId: "control_contract_table", layout: "full" },
  {
    id: "section_4",
    type: "markdown",
    sourceId: "framework_v22",
    body: md(
      "## 4. Framework 以权限边界为主线，而不是以单个算法为中心",
      "GroupedDataset → Public features → PoolState → Predictor → AcquisitionContext → Oracle.reveal → Evaluator 是冻结的数据流。Response label 只在 predictor fit 与 scorer 中出现；acquisition 不能持有完整 dataset 或 oracle。",
      "新增 acquisition 只实现公共 context 到 batch IDs 的选择；新增 predictor 保持 fit/predict/uncertainty 协议。CP/Tucker 或其他 masked tensor model 应作为 Predictor adapter：输入相同 revealed IDs、response/mask，输出同一 4,422 维 response，并继续使用当前 split、预算、receipt 与 evaluator。",
    ),
  },
  { id: "architecture_evidence", type: "table", tableId: "architecture_table", layout: "full" },
  {
    id: "section_5",
    type: "markdown",
    body: md(
      "## 5. Compute-aware 协议先做门禁，再做五 seed 正式判定",
      "Smoke 仅检查真实数据加载、GPU 训练、atomic staging/resume、hash inventory 和精确输出网格；科学判定完全来自 formal profile。Formal 使用 Direct 4,422-output dropout MLP：hidden 128、dropout 0.1、AdamW learning rate 0.001、weight decay 0.0002、batch 512、target-scale floor 0.05。每个 acquired budget 从头重训，三策略在同一 run seed 共享初始 IDs，模型与 acquisition seed 只依赖 (run_seed, current_budget)。",
      "Formal 从 initial 128 开始执行 7 个 acquisition rounds，每次增加 128，因而在 128/256/384/512/640/768/896/1,024 共 8 个 budgets 从头 fit；只在预注册的 128/256/512/1,024 四个 checkpoints 做正式曲线评价。真实数据 cache 命中，cache identity key 为 9744a3a995c3be3cc212e432f8fb271478620d847a6a7e569464bf85d3685d0f。",
      "Formal 运行实际完成 150 次训练 fits（另有 5 个复用的 semantic full-reference receipts）；非复用 fit 的 train_seconds 合计约 261 秒，五个 seed-run 的墙钟和约 458 秒，fresh 启动到最后 seed 完成约 466 秒。GPU 为 A100 40GB，PyTorch deterministic algorithms 开启，cuDNN benchmark 与 TF32 关闭。最终 resume 只验证并跳过五个 hash-valid seeds，没有重写 seed payload。",
    ),
  },
  { id: "protocol_evidence", type: "table", tableId: "protocol_table", layout: "full" },
  {
    id: "section_6",
    type: "markdown",
    sourceId: "report_snapshot_sql",
    body: md(
      "## 6. Random baseline 稳定学习，但 1,024 预算仍未达到 B80",
      "Random 的 interpolation delta_skill_zero 从预算 128 的 0.0623 上升到预算 1,024 的 0.2047；同一 seed 的三策略在 128 完全相同，证明初始集合公平。下图只展示四个注册 checkpoint，因此用 grouped bars，而不把稀疏点包装成连续趋势。",
    ),
  },
  { id: "checkpoint_skill_evidence", type: "chart", chartId: "checkpoint_skill_chart", layout: "full" },
  {
    id: "section_6_interpretation",
    type: "markdown",
    sourceId: "report_snapshot_sql",
    body: md(
      "### Full reference 说明还存在可学习空间，但不是理论上界",
      "五 seed 的 combined-representation full reference 在 interpolation 上为 0.2598 ± 0.0018。B80 按“从同 seed 初始值到 same-seed full reference 的可实现增益 80%”定义；15/15 seed×policy curves 到 1,024 均为 not_reached，所以没有制造外推预算。辅助 full-reference panel 也显示 OOD 难度不均：chemical-only 与 both 的 skill 接近 0，而 time holdout 与 interpolation 更高。",
    ),
  },
  { id: "full_metrics_evidence", type: "table", tableId: "full_metrics_table", layout: "full" },
  {
    id: "section_7",
    type: "markdown",
    sourceId: "report_snapshot_sql",
    body: md(
      "## 7. CoreSet 有正向信号但未确认，Uncertainty 不应晋升",
      "Random 是均匀无放回；CoreSet 在 495 维 target-free descriptor 上做 farthest-first；Uncertainty 使用同一 Direct predictor 的 8-pass MC-dropout response variance。CoreSet 的 mean AULC 最高，且五个 seeds 均高于 Random，但配对差的 95% CI 为 [-0.001646, 0.016305]，CI 下界没有大于 0，因此不能越过预注册替换门槛。Uncertainty 的配对差为负，95% CI 全部低于 0。",
      "最终判定是 **retain Random**。CoreSet 可作为“值得复验”的候选，但不能写成统计确认优于 Random；对 Uncertainty，下一步应先评估 calibration，并比较 ensemble 等替代估计，再恢复 acquisition 比较。",
    ),
  },
  { id: "policy_evidence", type: "table", tableId: "policy_decision_table", layout: "full" },
  {
    id: "section_8_semantics",
    type: "markdown",
    sourceId: "report_snapshot_sql",
    body: md(
      "## 8. 表征、低秩与张量只提供下一阶段路线证据",
      "**Combined representation 的效应依赖 split 与预算。** 它包含 identity、continuous time 与冻结的 chemical/strain semantics。Full budget 时 chemical-only、strain-only 和 both cold-start 的 delta_skill_zero 都高于 identity+time；但 interpolation 低 0.00366，且配对 CI 完全为负。下图因此是 regime comparison，而不是“语义普遍更好”的排行榜。",
    ),
  },
  { id: "representation_evidence", type: "chart", chartId: "representation_chart", layout: "full" },
  {
    id: "section_8_semantic_limits",
    type: "markdown",
    sourceId: "report_snapshot_sql",
    body: md(
      "### Cold-start 改善不能被解释为 semantic causality",
      "预算 128 和 512 的 chemical/strain cold-start 配对差多数为正，而 interpolation 的 CI 穿零；full interpolation 则轻微但一致地负向。因为没有 shuffled-semantic negative control，且五个 strain identity 只是高置信 public candidate mappings、并非 organizer-verified，结果只能称为两套固定 representation 的预测差异。DHY210 的连续语义保持 zero/missing。",
    ),
  },
  { id: "semantic_delta_evidence", type: "table", tableId: "semantic_delta_table", layout: "full" },
  {
    id: "section_8_lowrank",
    type: "markdown",
    sourceId: "report_snapshot_sql",
    body: md(
      "### Rank-64 的压缩结构存在，但覆盖不足且实测弱于 Direct",
      "Official-train response spectrum 在 rank 64 只保留 centered response 的 53.17% 或 per-protein standardized response 的 54.23%；rank 128 也只有约 62%。这支持继续研究压缩结构，却不足以假定低秩 head 可以无损替代 Direct。",
    ),
  },
  { id: "rank_energy_evidence", type: "chart", chartId: "rank_energy_chart", layout: "full" },
  {
    id: "section_8_lowrank_interpretation",
    type: "markdown",
    sourceId: "report_snapshot_sql",
    body: md(
      "在 v2.1 同一 nested IDs、seed 42 的可比实验里，Direct 在预算 128、512 和 full 2,670 上都高于 rank-64。该单 seed 结果只用于冻结 v2.2 Direct backbone，不与 v2.2 五 seed estimates 合并。Tensor audit 则显示 official-train cell occupancy 很高，但 chemical fibers 几乎不完整；下一步若尝试 CP/Tucker，应采用 masked/inductive 形式并保持同 IDs、同预算、同 evaluator。",
    ),
  },
  { id: "direct_rank_evidence", type: "table", tableId: "direct_rank_table", layout: "full" },
  { id: "tensor_evidence", type: "table", tableId: "tensor_table", layout: "full" },
  {
    id: "section_9",
    type: "markdown",
    sourceId: "framework_v22",
    body: md(
      "## 9. 指标评价的是 response prediction 与 sample efficiency，不是假定的 hit",
      "主指标 delta_skill_zero = 1 - SSE(model) / SSE(delta=0) 使用 matched-control 无响应作为预注册 null；它是本地 normalized skill，不是 organizer official Delta Skill。Condition PCC、protein PCC/R² 和 pooled PCC 分别回答单 condition 蛋白模式、单 protein 跨 conditions 泛化与全局线性模式。AULC 必须按 budget 轴做梯形积分；B80 使用 same-seed full-reference 可实现增益并禁止外推。",
      "Impact、Hit Ratio、hit recall 与旧 hybrid 逻辑只存在于 legacy pilot_v1，没有进入 v2.1/v2.2 正式结果。大的 proteomic change 不等于 biological hit。",
    ),
  },
  { id: "metric_evidence", type: "table", tableId: "metric_table", layout: "full" },
  {
    id: "limitations",
    type: "markdown",
    sourceId: "formal_audit",
    body: md(
      "### 可信边界与尚未完成的科学证据",
      "独立 artifact audit 与统计复算均为 GO，未发现 P0/P1；但这只证明本地 retrospective benchmark、权限边界与保存的计算一致。Full reference 不是理论上界；辅助 split、representation 与多指标 CI 没有做多重比较校正；control 不是 vehicle-specific；没有 prospective wet-lab validation、organizer official scorer、submission、shuffled semantic negative control、dedicated condition/response coverage endpoint 或正式 tensor acquisition 实验。",
      "这些缺口不会推翻“保留 Random”这一预注册判定，却限定了可声称的外推范围。",
    ),
  },
  {
    id: "section_10",
    type: "markdown",
    body: md(
      "## 10. 下一阶段应围绕 cold-start、coverage 与真实实验合同推进",
      [
        "1. **冻结共同 comparator。** 保留 Random + Direct，复用相同 condition IDs、initial sets、budgets、seeds 与 response metrics。",
        "2. **优先改进 cold-start acquisition。** 比较 semantic-aware diversity/density、cluster-balanced batch 或 TypiClust 类方法，并单独预注册 chemical/strain cold-start 主终点。",
        "3. **谨慎复验 CoreSet。** 其方向值得增加 power，但正式结论仍是未确认；不做本轮 post-hoc 策略或超参数 sweep。",
        "4. **降级当前 Uncertainty。** 先评估 uncertainty calibration、deep ensemble 或更合适的 high-dimensional response uncertainty，再恢复 acquisition 比较。",
        "5. **做小规模 masked tensor predictor。** 先在 interpolation 上以 CP/Tucker adapter 与 Direct 同 IDs 比较，并按 chemical fiber coverage 分层；没有实测优势前不晋升。",
        "6. **补强实验与身份合同。** 获取显式 treatment–vehicle mapping、确认 strain identities，并加入 shuffled/blocked semantic negative control。",
      "7. **走向 prospective validation。** 在下一批真实 assay 中冻结选点清单、成本与失败规则，验证 retrospective sample-efficiency 是否能转化为 wet-lab learning gain。",
      ].join("\n"),
      "### 仍需回答的问题\n\n- Chemical/strain cold-start 的表征增益在 shuffled/blocked semantic negative control 下是否仍保留？\n- Masked CP/Tucker predictor 在同 IDs、同预算和同 evaluator 下能否稳定胜过 Direct？\n- Retrospective learning-curve 增益能否在真实 wet-lab assay 的失败率、成本和 replicate 约束下转化为 prospective gain？",
      "本阶段到此停止：框架目的已经可信完成，后续工作应作为新的预注册实验，而不是继续在当前 formal 结果上追加选择性分析。",
    ),
  },
];

const sourceDetails = [
  {
    ...manifestSources[0],
    query: {
      engine: "sqlite",
      language: "sql",
      description: "Executed bounded reporting snapshot transcribed from independently recomputed formal CSV/JSON aggregates; it contains only reviewed aggregate rows and no condition labels, predictions, or oracle responses.",
      executed_at: GENERATED_AT,
      tables_used: [
        "results/direct_multiseed_formal-20260824-v22/active_metrics.csv",
        "results/direct_multiseed_formal-20260824-v22/per_seed_curve_summary.csv",
        "results/direct_multiseed_formal-20260824-v22/paired_policy_comparisons.csv",
        "results/direct_multiseed_formal-20260824-v22/full_reference_metrics.csv",
        "results/direct_multiseed_formal-20260824-v22/representation_ablation_summary.csv",
        "results/direct_multiseed_formal-20260824-v22/data_audit.json",
        "results/direct_multiseed_formal-20260824-v22/control_vehicle_sensitivity.csv",
        "results/direct_multiseed_formal-20260824-v22/low_rank_spectrum.csv",
        "results/direct_multiseed_formal-20260824-v22/tensor_coverage.csv",
        "results/pilot_v2_formal-20260824-v21/representation_metrics.csv",
        "results/direct_multiseed_formal-20260824-v22/source_snapshot/FRAMEWORK_SPEC_V22.md",
      ],
      filters: [
        "formal seeds = 42,43,44,45,46",
        "primary policy endpoint split = interpolation",
        "primary metric = delta_skill_zero",
        "registered checkpoints = 128,256,512,1024",
        "representation contrast = combined semantics minus identity plus time",
        "legacy pilot_v1 hit/impact outputs excluded",
      ],
      metric_definitions: [
        "normalized_aulc is the trapezoidal integral of delta_skill_zero over budget divided by budget span",
        "paired policy difference is policy AULC minus same-seed Random AULC",
        "sample_sd uses n-1 denominator across five seeds",
        "paired 95% confidence intervals use two-sided t critical 2.776445 with df=4",
        "B80 is the first budget reaching 80% of same-seed initial-to-full achievable improvement and is never extrapolated",
        "combined semantics means identity plus continuous time plus frozen chemical and strain semantic blocks",
      ],
    },
  },
  {
    ...manifestSources[1],
    query: {
      engine: "file",
      language: "markdown",
      description: "Preregistered v2.2 query, control, representation, fairness, model, metric, artifact, and claim-boundary contract.",
      executed_at: "2026-08-24T20:44:44Z",
      tables_used: [manifestSources[1].path],
    },
  },
  {
    ...manifestSources[2],
    query: {
      engine: "file",
      language: "json",
      description: "Hash-valid root manifest covering run identity, environment, config/source/spec snapshot, seeds, inventory, protocol, and resume history.",
      executed_at: "2026-08-24T20:52:30Z",
      tables_used: [manifestSources[2].path],
    },
  },
  {
    ...manifestSources[3],
    query: {
      engine: "file",
      language: "json",
      description: "Condition grain, overlap repair, pool/evaluation partitions, query replicates, controls, missingness, tensor, and response-spectrum audit.",
      executed_at: "2026-08-24T20:44:44Z",
      tables_used: [manifestSources[3].path],
    },
  },
  {
    ...manifestSources[4],
    query: {
      engine: "file",
      language: "csv",
      description: "Post-hoc Water-versus-DMSO exact-context sensitivity; acquisition_input=false and training_input=false.",
      executed_at: "2026-08-24T20:44:44Z",
      tables_used: [manifestSources[4].path],
    },
  },
  {
    ...manifestSources[5],
    query: {
      engine: "file",
      language: "markdown",
      description: "Independent rehash, exact-grid, pool-boundary, resume, AULC, confidence-interval, representation, and B80 recomputation record.",
      executed_at: GENERATED_AT,
      tables_used: [manifestSources[5].path],
    },
  },
  {
    ...manifestSources[6],
    query: {
      engine: "file",
      language: "csv",
      description: "Single-seed same-nested-ID Direct and rank-64 feasibility comparison used to freeze the v2.2 Direct backbone.",
      executed_at: "2026-08-24T09:27:00Z",
      tables_used: [manifestSources[6].path],
      filters: ["seed = 42", "split = interpolation", "budgets = 128,512,2670"],
    },
  },
];

const artifact = {
  surface: "report",
  manifest: {
    version: 1,
    surface: "report",
    title: TITLE,
    description: "Condition-atomic GOAI 酵母扰动蛋白质组主动学习框架的正式多种子执行、独立审计与下一阶段决策。",
    generatedAt: GENERATED_AT,
    cards,
    charts,
    tables,
    sources: manifestSources,
    blocks,
  },
  snapshot: {
    version: 1,
    generatedAt: GENERATED_AT,
    status: "ready",
    datasets,
  },
  sources: sourceDetails,
};

const outputPath = join(here, "artifact.json");
writeFileSync(outputPath, JSON.stringify(artifact, null, 2) + "\n", "utf8");
const counts = Object.fromEntries(Object.entries(datasets).map(([key, rows]) => [key, rows.length]));
process.stdout.write(JSON.stringify({ output: outputPath, dataset_rows: counts }) + "\n");
