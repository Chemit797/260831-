-- GOAI-AL v2.2 final-report snapshot, SQLite-compatible.
-- Executed against an in-memory database on 2026-08-24 after independent
-- recomputation from the formal CSV/JSON artifacts named in artifact.json.
-- The rows below are the bounded, reviewed reporting grain; they intentionally
-- omit condition IDs, response values, predictions, and acquisition receipts.

CREATE TEMP TABLE headline_policy (
  random_aulc_x1000 REAL,
  coreset_delta_x1000 REAL,
  uncertainty_delta_x1000 REAL
);
INSERT INTO headline_policy VALUES (158.5001049416481, 7.329779867006048, -13.38261920633887);

CREATE TEMP TABLE headline_data (
  proteins INTEGER,
  evaluation_conditions INTEGER,
  pool_eval_overlap INTEGER
);
INSERT INTO headline_data VALUES (4422, 2250, 0);

CREATE TEMP TABLE checkpoint_skill (
  strategy TEXT,
  budget INTEGER,
  budget_label TEXT,
  mean REAL,
  sample_sd REAL,
  mean_x100 REAL,
  seed_count INTEGER
);
INSERT INTO checkpoint_skill VALUES
  ('Random',128,'128',0.06227716254400073,0.02202047902708675,6.227716254400073,5),
  ('Random',256,'256',0.12100729483998704,0.007266997275209,12.100729483998704,5),
  ('Random',512,'512',0.16247290895646346,0.00558610260409058,16.247290895646346,5),
  ('Random',1024,'1,024',0.20471624209508277,0.003522735946202725,20.471624209508277,5),
  ('CoreSet',128,'128',0.06227716254400073,0.02202047902708675,6.227716254400073,5),
  ('CoreSet',256,'256',0.12582631992260412,0.01346526752528422,12.582631992260412,5),
  ('CoreSet',512,'512',0.1719471473583391,0.003063267733722018,17.19471473583391,5),
  ('CoreSet',1024,'1,024',0.21254484521482767,0.004076109417689754,21.254484521482767,5),
  ('Uncertainty',128,'128',0.06227716254400073,0.02202047902708675,6.227716254400073,5),
  ('Uncertainty',256,'256',0.10414292686124202,0.006167905907310947,10.414292686124202,5),
  ('Uncertainty',512,'512',0.14094638474578552,0.003812370718619003,14.094638474578552,5),
  ('Uncertainty',1024,'1,024',0.20281513717297242,0.003863832430212846,20.281513717297242,5);

CREATE TEMP TABLE policy_decision (
  strategy TEXT,
  mean_aulc TEXT,
  paired_delta TEXT,
  ci95 TEXT,
  wins_losses TEXT,
  preregistered_gate TEXT,
  final_status TEXT
);
INSERT INTO policy_decision VALUES
  ('Random','0.158500 ± 0.004163','reference','—','—','default comparator','保留默认'),
  ('CoreSet','0.165830 ± 0.005114','+0.007330','[-0.001646, 0.016305]','5 / 0','未通过：CI 下界不大于 0','不替换 Random'),
  ('Uncertainty','0.145117 ± 0.002613','-0.013383','[-0.017284, -0.009481]','0 / 5','未通过','不替换 Random');

CREATE TEMP TABLE data_scope (
  role TEXT,
  conditions INTEGER,
  revealable TEXT,
  purpose TEXT
);
INSERT INTO data_scope VALUES
  ('Official train',3337,'部分','pool 与 interpolation 的共同母集'),
  ('Candidate pool',2670,'是','唯一允许 acquisition/query 的 conditions'),
  ('Interpolation',667,'否','official-train 内 condition-atomic 缺组合评估'),
  ('val_chem_only',503,'否','未见 chemical 的 cold-start 评估'),
  ('val_strain_only',874,'否','未见 strain 的 cold-start 评估'),
  ('val_both',126,'否','chemical 与 strain 同时未见'),
  ('val_time',80,'否','移除 46 个 train-overlap conditions 后的时间评估'),
  ('Evaluation total',2250,'否','五个评估 split 合计'),
  ('All unique conditions',4920,'—','condition-atomic 全局索引');

CREATE TEMP TABLE control_contract (
  item TEXT,
  value TEXT,
  interpretation TEXT
);
INSERT INTO control_contract VALUES
  ('Policy','pooled_exact_context_water_dmso','八字段 exact assay context 内逐 control measurement 的 log2 等权均值'),
  ('Control measurements','956','不进入 query、descriptor 或 predictor feature'),
  ('Exact control contexts','479','429 同时有 Water/DMSO；41 仅 DMSO；9 仅 Water'),
  ('Exactly matched treatments','7,884 / 7,884','7,293 双类型；450 仅 DMSO；141 仅 Water'),
  ('Cross-split-only control support','1,478 treatments','其中 official-train treatment 12 条；冻结为 assay overhead'),
  ('Treatment vehicle mapping','Unavailable','不得声称 comparator 是 vehicle-specific，也不得从其他字段推断'),
  ('Water–DMSO sensitivity','RMS 0.336909；frequency-weighted 0.338644','post-hoc oracle audit；acquisition/training input=false');

CREATE TEMP TABLE genedisco_compare (
  dimension TEXT,
  genedisco_style TEXT,
  goai_framework TEXT,
  implication TEXT
);
INSERT INTO genedisco_compare VALUES
  ('主动学习骨架','pool-based、轮次 acquisition、固定预算','保留相同抽象','可以复用接口与公平比较原则'),
  ('单次 query','一个 intervention 的标量结果','一个结构化 condition 的 4,422 维 masked response','query 成本与信息量均为 condition-level'),
  ('输入结构','主要由 intervention identity 描述','strain × chemical × medium × temperature × time','需要 target-free descriptor 与严格 split'),
  ('response 构造','通常直接使用 phenotype','matched-control log2 delta 与 replicate 聚合','control 是 assay overhead，不是 acquisition feature'),
  ('评价目标','常见 hit/discovery 或 scalar prediction','预测完整 response landscape 与 sample efficiency','不得照搬 Hit Ratio'),
  ('代码关系','参考生态','概念与接口层借鉴；无代码嫁接','新算法通过 Acquisition/Predictor adapter 接入');

CREATE TEMP TABLE architecture (
  module TEXT,
  responsibility TEXT,
  extension_contract TEXT
);
INSERT INTO architecture VALUES
  ('data.py','condition ID、control、split、cache','保持 condition-atomic 与 control contract'),
  ('semantics.py','target-free identity/chemical/strain encoder','新增资产必须 hash-pinned 且 response_used=false'),
  ('simulator.py','RetrospectiveOracle、PoolState、BudgetSchedule、receipts','禁止未 reveal labels 与 evaluation truth 进入策略'),
  ('model.py','Direct/low-rank Predictor 与不确定性','新模型实现同一 fit/predict/uncertainty 协议'),
  ('acquisition.py','Random、CoreSet、MC-dropout uncertainty','新策略只接收 immutable public context'),
  ('metrics.py','response panel、AULC、B80','保持 truth-only mask 与预注册方向'),
  ('direct_multiseed.py','公平 seeds、cold restart、atomic resume、exact-grid gate','新 runner 必须保存 source/config/spec snapshot');

CREATE TEMP TABLE compute_protocol (
  profile TEXT,
  seeds TEXT,
  budget_schedule TEXT,
  model_training TEXT,
  purpose TEXT
);
INSERT INTO compute_protocol VALUES
  ('Smoke','42, 43','initial/batch 32；checkpoints 32/64/96','2 epochs；MC 2；Direct','仅验证端到端、恢复与产物合同'),
  ('Formal','42–46','initial/batch 128；checkpoints 128/256/512/1024','80 epochs；MC 8；每 budget cold restart','预注册五 seed 策略判定');

CREATE TEMP TABLE full_metrics (
  split TEXT,
  rmse TEXT,
  mae TEXT,
  skill TEXT,
  pooled_pcc TEXT,
  condition_pcc TEXT,
  protein_pcc TEXT,
  protein_r2_median TEXT,
  protein_r2_mean TEXT,
  protein_r2_positive_fraction TEXT
);
INSERT INTO full_metrics VALUES
  ('interpolation','.3458 ± .0004','.2285 ± .0003','.2598 ± .0018','.5098 ± .0017','.3619 ± .0029','.4198 ± .0025','.1702 ± .0020','.2017 ± .0019','.9866 ± .0010'),
  ('val_chem_only','.3185 ± .0009','.2107 ± .0008','.0076 ± .0059','.1992 ± .0022','.1724 ± .0024','.1704 ± .0040','-.0052 ± .0043','-.0122 ± .0065','.4606 ± .0306'),
  ('val_strain_only','.3586 ± .0005','.2366 ± .0004','.1175 ± .0026','.3545 ± .0037','.2157 ± .0082','.2865 ± .0029','.0482 ± .0030','.0434 ± .0032','.7122 ± .0102'),
  ('val_both','.3244 ± .0007','.2111 ± .0006','.0020 ± .0042','.1681 ± .0054','.1098 ± .0078','.1245 ± .0041','-.0294 ± .0033','-.1191 ± .0035','.3499 ± .0159'),
  ('val_time','.3766 ± .0007','.2513 ± .0003','.2796 ± .0026','.5285 ± .0024','.3308 ± .0058','.4485 ± .0023','.1721 ± .0021','.1510 ± .0063','.8556 ± .0057');

CREATE TEMP TABLE full_representation_skill (
  split TEXT,
  representation TEXT,
  mean REAL,
  sample_sd REAL,
  mean_x100 REAL,
  budget INTEGER,
  seed_count INTEGER
);
INSERT INTO full_representation_skill VALUES
  ('Interpolation','Identity + time',0.26348504382872884,0.001890093103027974,26.348504382872884,2670,5),
  ('Interpolation','Combined semantics',0.2598269013902549,0.00176586461227343,25.98269013902549,2670,5),
  ('Chemical cold-start','Identity + time',-0.11357232942504011,0.011605688471162104,-11.357232942504011,2670,5),
  ('Chemical cold-start','Combined semantics',0.0075678203595553,0.005897568241458644,0.75678203595553,2670,5),
  ('Strain cold-start','Identity + time',0.06829871735736584,0.002796636281977838,6.829871735736584,2670,5),
  ('Strain cold-start','Combined semantics',0.11754323445388185,0.002623548001865272,11.754323445388185,2670,5),
  ('Both cold-start','Identity + time',-0.02739149910398884,0.003881271202877441,-2.739149910398884,2670,5),
  ('Both cold-start','Combined semantics',0.00203138944696688,0.00417749651793726,0.203138944696688,2670,5),
  ('Time holdout','Identity + time',0.27842753471744897,0.003484559751201604,27.842753471744897,2670,5),
  ('Time holdout','Combined semantics',0.2796466030503758,0.002644408962950958,27.96466030503758,2670,5);

CREATE TEMP TABLE paired_semantic_delta (
  budget INTEGER,
  split TEXT,
  paired_delta TEXT,
  ci95 TEXT,
  wins_losses TEXT,
  interpretation TEXT
);
INSERT INTO paired_semantic_delta VALUES
  (128,'interpolation','+0.002362','[-0.011964, +0.016688]','3 / 2','不确定'),
  (128,'val_chem_only','+0.102454','[+0.084152, +0.120756]','5 / 0','正向描述性证据'),
  (128,'val_strain_only','+0.022008','[+0.001307, +0.042709]','4 / 1','正向描述性证据'),
  (128,'val_both','+0.071643','[+0.034025, +0.109260]','5 / 0','正向描述性证据'),
  (128,'val_time','-0.011715','[-0.062882, +0.039452]','3 / 2','不确定'),
  (512,'interpolation','+0.005722','[-0.000458, +0.011903]','4 / 1','不确定'),
  (512,'val_chem_only','+0.082197','[+0.069155, +0.095239]','5 / 0','正向描述性证据'),
  (512,'val_strain_only','+0.046082','[+0.016616, +0.075548]','5 / 0','正向描述性证据'),
  (512,'val_both','+0.024817','[+0.001894, +0.047739]','5 / 0','正向描述性证据'),
  (512,'val_time','+0.007214','[-0.001333, +0.015762]','4 / 1','不确定'),
  (2670,'interpolation','-0.003658','[-0.005561, -0.001756]','0 / 5','负向'),
  (2670,'val_chem_only','+0.121140','[+0.104732, +0.137549]','5 / 0','正向描述性证据'),
  (2670,'val_strain_only','+0.049245','[+0.042889, +0.055600]','5 / 0','正向描述性证据'),
  (2670,'val_both','+0.029423','[+0.020492, +0.038354]','5 / 0','正向描述性证据'),
  (2670,'val_time','+0.001219','[-0.006115, +0.008553]','2 / 3','不确定');

CREATE TEMP TABLE rank_energy (
  rank INTEGER,
  rank_label TEXT,
  variant TEXT,
  cumulative_energy REAL
);
INSERT INTO rank_energy VALUES
  (8,'8','Centered response',0.2975533362630767),
  (16,'16','Centered response',0.3687735268051574),
  (32,'32','Centered response',0.4464219990035972),
  (64,'64','Centered response',0.5316853679758635),
  (128,'128','Centered response',0.6187440806661079),
  (8,'8','Per-protein standardized',0.3116906182251092),
  (16,'16','Per-protein standardized',0.386385807656376),
  (32,'32','Per-protein standardized',0.4623148961241879),
  (64,'64','Per-protein standardized',0.542291845668172),
  (128,'128','Per-protein standardized',0.6220606548833963);

CREATE TEMP TABLE direct_rank64 (
  budget INTEGER,
  direct_skill TEXT,
  rank64_skill TEXT,
  delta_direct_minus_rank64 TEXT,
  evidence_scope TEXT
);
INSERT INTO direct_rank64 VALUES
  (128,'0.063980','0.043628','+0.020352','v2.1 same IDs, seed 42'),
  (512,'0.155130','0.084643','+0.070487','v2.1 same IDs, seed 42'),
  (2670,'0.267194','0.207050','+0.060144','v2.1 same IDs, seed 42');

CREATE TEMP TABLE tensor_feasibility (
  scope TEXT,
  occupied_cells TEXT,
  occupancy TEXT,
  chemical_complete_fibers TEXT,
  implication TEXT
);
INSERT INTO tensor_feasibility VALUES
  ('Official train','3,337 / 3,552','93.95%','4 / 96 (4.17%)','总 occupancy 高，但 chemical fibers 多数不完整'),
  ('Candidate pool','2,670 / 3,552','75.17%','0 / 96 (0%)','需要 masked/inductive tensor，而非完整张量假设');

CREATE TEMP TABLE metric_definitions (
  metric TEXT,
  definition TEXT,
  interpretation TEXT,
  direction TEXT
);
INSERT INTO metric_definitions VALUES
  ('delta_skill_zero','1 - SSE(model) / SSE(delta=0)','相对无扰动响应基线的本地 normalized skill','越高越好'),
  ('delta RMSE / MAE','在 truth-observed protein positions 上的误差','response 幅度误差','越低越好'),
  ('Pooled delta PCC','展平所有 observed positions 后的相关','总体线性模式','越高越好'),
  ('Condition PCC median','每个 condition 内跨 proteins 的 PCC 中位数','单 condition 蛋白模式','越高越好'),
  ('Protein PCC / R²','每个 protein 跨 conditions 的泛化统计','protein-wise landscape','越高越好'),
  ('Normalized AULC','按 budget 轴做梯形积分并除以预算跨度','主 sample-efficiency endpoint','越高越好'),
  ('B80','达到 same-seed full reference 可实现增益 80% 的首个预算','不外推；未达到即 not_reached','越低越好'),
  ('Evaluable counts','满足样本数与非零方差条件的 axes 数量','防止 NaN/常量轴掩盖','越多越好');

-- This file has been syntax-checked with SQLite. Report packaging executes the
-- same file and queries each bounded table into artifact.json.
