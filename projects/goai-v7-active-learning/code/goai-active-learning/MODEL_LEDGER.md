# Model Ledger

## GOAI-AL-V2-PILOT-01

- Status: formal execution complete; automated validation passed
- Model ID: `GOAI-AL-V2-PILOT-01`
- Protocol: `goai-condition-atomic-v2.1`
- Intended use: retrospective comparison of random, coreset, and MC-dropout
  uncertainty acquisition for matched-control yeast-proteome response prediction
- Predictor: rank-64 response-only low-rank dropout MLP, retrained from scratch
- Registered seed: 42
- Formal budgets: initial 128; batch 128; checkpoints 128, 256, 512, 1,024
- Formal training: 80 epochs per fit; 8 MC-dropout passes
- Primary evaluation: deterministic condition-atomic interpolation holdout
- Auxiliary evaluation: each cleaned official validation split separately
- Reference: same-backbone full candidate-pool fit
- Representation check: direct versus rank-64 on nested-random 128, 512, and full
  candidate-pool IDs; no rank sweep

This is a local GOAI-AL proxy experiment, not an official GOAI model, metric,
submission, or leaderboard result. In particular, `delta_skill_zero` is a local
proxy, not an official score. This record does not by itself support biological
discovery claims.

### Completed formal execution record

- Formal attempt directory:
  `results/pilot_v2_formal-20260824-v21`
- Manifest: [`results/pilot_v2_formal-20260824-v21/manifest.json`](results/pilot_v2_formal-20260824-v21/manifest.json)
- Manifest SHA-256:
  `1c41562578da8dc4cf00b41a7265c28e0c71872159849c44ee253440056062a5`
- Exact command:

  ```text
  '/home/chenyuming/Project/Biohub - Cell Tracking During Development/.venv/bin/python3.12' -m goai_al.experiment --config configs/pilot.yaml --output-suffix formal-20260824-v21
  ```

- Started UTC: `2026-08-24T16:59:04.742492+00:00`
- Completed UTC: `2026-08-24T17:00:35.529272+00:00`
- Data provenance:
  - metadata: `/home/chenyuming/Project/go-ai/WAYB_WAYC_metadata_train_val.csv`;
    904,646 bytes; SHA-256
    `9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef`
  - proteome: `/home/chenyuming/Project/go-ai/WAYB_WAYC_proteome_raw_train_val.csv`;
    289,769,736 bytes; SHA-256
    `a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb`
- Config provenance: `/home/chenyuming/Project/active-learning/goai_active_learning/configs/pilot.yaml`;
  904 bytes; SHA-256
  `e8bd0ff11ba1b90b31d4118cac23d21e539bc788abe62150139765be9e4c3bf6`
- Source-hash inventory location: formal manifest object `hashes.source` in
  [`manifest.json`](results/pilot_v2_formal-20260824-v21/manifest.json). The eight
  inventoried source hashes are:
  - `src/goai_al/__init__.py`:
    `23f882ed4f79e17eb18c1938f8b73ef904ec966f1310de0759967a24cfd0a8dc`
  - `src/goai_al/acquisition.py`:
    `9bba057ebd26ebd30239afe638dcfd10d35a8d81f4b7bcca9383729e9639482a`
  - `src/goai_al/audit.py`:
    `19937ee52e04322b7f5da98b00d937821be973e56565ca7bfc6d80c2586f60c5`
  - `src/goai_al/data.py`:
    `7d07260a9754f34b2cca63b197e6615057f78cbf6cc3eaa866b9ebcea1045e45`
  - `src/goai_al/experiment.py`:
    `7c0745da61c59fddcc9a011265938e4330044de31d0da62f56965ed12997954c`
  - `src/goai_al/metrics.py`:
    `a798b295793ea0e24ef50f20ec0669dc3548d6be4cd6d19e2afeabc2243487d2`
  - `src/goai_al/model.py`:
    `985637b4c8a6237d414134b228420de2bae26537ab330ee793ee8a996e713201`
  - `src/goai_al/simulator.py`:
    `a6c6a02953df2740ec39d8a6e4a9d1ec62ce00bfde98fd95b2f029f57e998fc0`
- Environment: `PYTHONPATH=src`; `CUDA_VISIBLE_DEVICES=0`; Linux
  `6.8.0-137-generic-x86_64` with glibc 2.39; Python 3.12.13; NumPy 2.5.2;
  pandas 3.0.5; PyTorch 2.9.1+cu126; CUDA available with one
  `NVIDIA A100-PCIE-40GB` device.
- Seed and protocol actually executed: scientific formal mode; global seed 42;
  one isolated oracle per strategy; identical deterministic initial set;
  model seed `(global_seed * 1000003 + current_budget) mod (2^31 - 1)`;
  fresh rank-64 fit at budgets 128, 256, 384, 512, 640, 768, 896, and 1,024;
  evaluation only at registered checkpoints; target fraction 0.8; full-pool
  reference budget 2,670; direct/rank-64 checks at 128, 512, and 2,670.
- Data and split counts: 4,920 global conditions; 3,337 official-train
  conditions split into candidate pool 2,670 and interpolation 667; cleaned
  validation counts `val_chem_only=503`, `val_strain_only=874`,
  `val_both=126`, `val_time=80`; 46 overlapping `val_time` conditions removed;
  52 validation treatment rows excluded from train aggregation; train oracle
  built from exactly 5,078 train treatment rows; 7,884 released treatment rows;
  4,422 proteins; condition-protein missingness 14.1606%.
- Control counts and boundary: 1,478 treatment measurements use cross-split-only
  exact-context controls, including 12 train measurements. Water/DMSO controls
  are assay overhead only, never query candidates or predictor/acquisition input.
- Execution-table counts: [`active_metrics.csv`](results/pilot_v2_formal-20260824-v21/active_metrics.csv)
  60 rows x 24 columns; [`acquisitions.csv`](results/pilot_v2_formal-20260824-v21/acquisitions.csv)
  3,072 x 8; [`representation_metrics.csv`](results/pilot_v2_formal-20260824-v21/representation_metrics.csv)
  30 x 22; [`model_fit_receipts.csv`](results/pilot_v2_formal-20260824-v21/model_fit_receipts.csv)
  31 x 18; 24 standalone round receipts.
- Artifact inventory: 38 entries in `manifest.artifact_inventory`; all 37
  externally hashable entries were independently checked against their recorded
  byte counts and SHA-256 values with zero mismatch. `manifest.json` is the 38th,
  self-describing entry; its SHA-256 is recorded above.
- Completion status: manifest `status=complete`, `mode=formal`,
  `scientific=true`; all three strategies reached budget 1,024 with 1,024 unique
  queries each and no out-of-pool query.
- Automated validation sign-off: `PYTHONPATH=src` formal environment test run
  completed with **20 passed** (two pandas deprecation warnings); artifact
  integrity, initial-set identity, budget conservation, query uniqueness,
  acquisition-column boundary, split disjointness, and seed fairness were also
  independently checked with no failure.

### Corrected smoke execution — non-scientific

- Classification: completed corrected smoke; **non-scientific** and excluded
  from formal scientific conclusions.
- Attempt directory: `results/pilot_v2_corrected-smoke-20260824-v21`
- Manifest: [`results/pilot_v2_corrected-smoke-20260824-v21/manifest.json`](results/pilot_v2_corrected-smoke-20260824-v21/manifest.json)
- Manifest SHA-256:
  `a565303864391e42e3fb4a32d3b4bc1a67702470471126db70a58013d77f7b4e`
- Exact command:

  ```text
  '/home/chenyuming/Project/Biohub - Cell Tracking During Development/.venv/bin/python3.12' -m goai_al.experiment --config configs/pilot.yaml --smoke --output-suffix corrected-smoke-20260824-v21
  ```

- Started/completed UTC: `2026-08-24T16:56:03.937080+00:00` /
  `2026-08-24T16:57:31.414176+00:00`
- Executed protocol: seed 42; initial/batch 32; checkpoints 32, 64, 96; 2
  epochs; 2 MC passes; Random/CoreSet/Uncertainty; `scientific=false`.
- Artifacts: 23 manifest entries: 22 externally hashable artifacts independently
  matched their byte counts and SHA-256 values, plus the self-describing
  manifest. Smoke results validate mechanics only and do not fill or alter the
  formal record above.

## GOAI-AL-V22-DIRECT-SEMANTIC-01

- Status: preregistered before execution; five-seed formal and independent
  artifact/statistical audits now complete
- Parent: `GOAI-AL-V2-PILOT-01`
- Protocol/spec: [`FRAMEWORK_SPEC_V22.md`](FRAMEWORK_SPEC_V22.md)
- Scope: frozen Direct 4,422-output predictor; Random/CoreSet/MC-dropout;
  paired formal seeds 42--46; identity+time versus identity+time+target-free
  chemical/strain semantics
- Data hashes: metadata
  `9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef`;
  proteome
  `a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb`
- Pool/evaluation: 2,670 candidate conditions; 667 interpolation; auxiliary
  `503/874/126/80`; 4,422 proteins; pairwise disjoint and evaluation IDs
  non-revealable
- Control: `pooled_exact_context_water_dmso`; `vehicle_column=null`; direct
  measurement mean in log2; no vehicle inference
- Formal contract: initial/batch 128; checkpoints 128/256/512/1024; 80 epochs;
  8 MC passes. Smoke is two seeds, 32/64/96, 2 epochs and non-scientific.
- Primary gate: paired normalized interpolation `delta_skill_zero` AULC versus
  Random; mean > 0, 95% t-CI lower > 0, wins >= 4/5; multiple qualifying
  strategies are reported without a policy-vs-policy superiority claim
- Frozen hashes: config
  `ce772f6d3cb03451b55454811be3f32e8088c7c1a4e59fce63646e242c2a19d7`;
  spec `4f870cb240dcbf1124867363539d500d25c5ad2b03260cbd4fd4c38afb6962d2`;
  source inventory
  `955e42c996cf4ee0c68b00afeb1839a6823d098986ed78d6e25715cb488efd6b`
- Preflight identities: identity/semantic matrix SHA256
  `99f3af43ac59ad285644cbe3d784ca31b7d42c58bc173e2b020db38c18590823` /
  `2f2980e2a187ff0cad22ed82f0f5456706e8bced802e846ea0ee27ff8f38e4a9`;
  row-order SHA256
  `d53993b198df8a05dc4fbb1eb6bff8155ea4ee3bd3c31ee5f83caab74510a6ea`;
  control contract SHA256
  `e0c86383e991117a2783928106c33edb018769a22cc8c2932bcc31c8003d5eed`
- Planned artifacts: `results/direct_multiseed_smoke-20260824-v22` and
  `results/direct_multiseed_formal-20260824-v22`; actual source/config/spec
  snapshot and exact artifact inventory are mandatory
- Smoke receipt: completed `2026-08-24T20:33:56Z`--`20:35:56Z` on A100 GPU0;
  seeds 42/43; active/acquisition/full/ablation rows `90/576/10/60`;
  `diagnostic_only=true`. Resume skipped both seeds without rewriting them.
  Root manifest SHA256
  `fd44535f049e50f5cefda9d78b493b7bf9f27ff8ccad7c258fb1298610d62a93`;
  payload SHA256
  `ff4aade4ba0aa28ec4a1786229d8b8222089c2a33fe87c859a7afeb6e6765cb6`.
  Independent inventory/leakage/resume/determinism audit: `FORMAL GO`, no P0/P1.
- Formal execution: `2026-08-24T20:44:44Z`; seeds 42--46; fresh-to-last-seed
  465.761 s; resume 9.594 s and skipped all completed seeds. Exact grids:
  active/acquisition/full/ablation `300/15360/25/150`; 155 receipts, 150 actual fits.
- Primary AULC mean±sample SD: Random `0.158500±0.004163`; CoreSet
  `0.165830±0.005114`; Uncertainty `0.145117±0.002613`. CoreSet−Random
  `+0.00732978`, 95% t-CI `[-0.00164572,0.01630528]`, wins 5/5; CI gate fails.
  Uncertainty−Random `-0.01338262`, CI `[-0.01728392,-0.00948132]`, wins 0/5.
  Decision: **retain Random**. All 15 B80 outcomes are not reached; no extrapolation.
- Random learning curve, interpolation `delta_skill_zero`: B128
  `0.062277±0.022020`; B256 `0.121007±0.007267`; B512
  `0.162473±0.005586`; B1024 `0.204716±0.003523`; full 2670 reference
  `0.259827±0.001766`.
- Representation result is split-dependent: combined−identity full-budget skill
  `+0.121140` chemical OOD, `+0.049245` strain OOD, `+0.029423` both OOD,
  but `-0.003658` interpolation. No shuffled control; no semantic-causality claim.
- Formal root manifest SHA256
  `b51658c85c75fb1ac5548347b6fb3906d19534ffa478c4026a3fe0bfad3bf6e0`;
  payload
  `29832d7de5f51607e71a70b112b9070d9fbe10fd6e0162a926805ca3817259d3`;
  run identity
  `7e75eddbb1c563357181cd4b3aae9c794012491054dba4a4d65b278ca9abb796`;
  source snapshot payload
  `fcab6c2fc787c5cf448e7586b52971da0a5ee39c9ec85a8f8118e1df6635663f`.
- Independent audit: 178 root artifacts, 10 source snapshot files, and 29 files
  per seed all hash-valid; exact receipt/grid/stat recomputation passed; zero
  reselection, out-of-pool ID, evaluation reveal, or forbidden receipt field.
- Final technical report:
  [`reports/goai_al_v22_final/report.html`](reports/goai_al_v22_final/report.html)
  with canonical artifact
  [`artifact.json`](reports/goai_al_v22_final/artifact.json) and delivery receipt
  [`DELIVERY_RECEIPT.json`](reports/goai_al_v22_final/DELIVERY_RECEIPT.json).
  Artifact/report SHA256 are respectively
  `45914c7ea054ee312be6619da354ffd084d7b317b59a119dd21366904f3258a7` and
  `08659f859f71281a78b8caafe88c4284de8208951add26734c50c22a60aeec98`.
  Canonical validation and packaging passed; browser verification is accurately
  recorded as `structural_only` because no compatible Chromium was available.
  Independent content, scientific-number, and portable-artifact audits all
  returned `GO` with no P0/P1 findings.
- Official score/submission: none; local retrospective proxy only
- M0--M12 submission-lineage effect: none

## GOAI-AL-TRANSFER-GEOMETRY-01

- Status: completed single-seed structural research audit; local retrospective
  proxy only, with no submission or official GOAI score.
- Parent: `GOAI-AL-V22-DIRECT-SEMANTIC-01`; no change to the frozen v2.2
  runner, results, acquisition policy, or submission lineage.
- Question: whether exact knowledge transfer is supported by independent
  strain/chemical/time factors or requires pairwise relational information.
- Data: candidate-only condition-atomic matched-control response pool,
  2,670 candidate conditions from 3,337 official-train conditions and 4,422
  proteins. Metadata SHA256
  `9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef`;
  proteome SHA256
  `a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb`.
- Protocol: seed/model seed 42; 64 metadata-stratified, globally withheld
  targets; fixed 256-condition baseline with all target feature levels covered;
  six donors/arm; 80-epoch Direct identity+time MLP.  Ten relation arms per
  target include S/D/T singles, three pairs, three equal-budget 3+3 mixtures,
  and random. Target response was scored only after each fit. Bootstrap uses
  1,000 target-cluster draws; explanatory performance is leave-one-target-out.
- Completion: `2026-08-25T11:12:01Z`--`11:23:01Z` (660.447 s), A100 GPU0,
  Python 3.12.13 / PyTorch CUDA environment. The grid is complete: 640 rows
  (64x10), no target appeared in any support set, all 262-condition support
  sets were deduplicated, and all target feature-coverage flags passed.
- Results (mean equal-budget pair excess; 95% target-bootstrap CI):
  strain×chemical `0.108370 [0.062330, 0.160443]`; chemical×time
  `0.062410 [0.028378, 0.099687]`; strain×time
  `0.057144 [0.013045, 0.105483]`. In contrast, S/D/T relative-to-random
  CIs all crossed zero. Factor-only leave-one-target-out R² was `0.070295`
  (`[-0.000132, 0.126747]`); adding global pairwise terms gave `0.070271`
  (`[-0.001424, 0.125582]`), gain `-0.000024`.
- Interpretation/decision: direct pair information is detected, but neither
  a scalar/low-dimensional independent-factor Key nor a globally generalizable
  pairwise tensor Key is justified by this one panel. Do not launch the optional
  acquisition confirmation because no fixed `K*` was selected. Retain a
  target-conditioned relational-residual slot as the next research hypothesis;
  do not promote it to the frozen AL policy.
- Sensitivity: a second metadata-only donor draw retained each primary
  baseline exactly. S×D donor transfer had Spearman `0.979` and sign agreement
  `1.000` across 12 targets; other relation arms are reported in
  `DONOR_DRAW_SENSITIVITY_SUMMARY.csv` and show heterogeneous robustness.
- Artifacts: `results/transfer_geometry-20260825-v1/`; primary files SHA256:
  `TRANSFER_PROBE_RESULTS.csv`
  `5e0e198832ec0f3046e3540798918c2fef9b4ba57a843a9e93aefb6801774142`,
  `FACTOR_INTERACTION_SUMMARY.csv`
  `83b7136fa5f2e9f0d246dd4d2042e0f9d306ca0cd296b03caf39a05024bb0a06`,
  `FACTOR_DIMENSION_SUMMARY.csv`
  `77441dd6159b18ef37d3ffaecaee2ec3adac0cbf709c1f8e53ba9d5c0cdc8451`,
  and `GOAI_TRANSFER_GEOMETRY_SUMMARY.md`
  `44f5bad5e4cafff4928be455fd80039ec6acb062621799d0788150928f197e9c`.
- Source/tests: formal execution source SHA256
  `d1768496c86cfea5d7471b3f5e243c37e2e76bca76c150151a5ab4bcc27aec0b`.
  A post-run visualization-only helper added
  `donor_draw_sensitivity.png` without refitting; current source SHA256
  `588d4101232d90693fa6f573cc6445b6a50532fd651f04365227e9fcf99a51aa`.
  Final test suite: `115 passed`.
- Preserved smoke history: `results/transfer_geometry-smoke-20260825/` failed
  only while formatting its final summary (`KeyError` on a CI field) and is
  retained; `results/transfer_geometry-smoke2-20260825/` completed after the
  fix but is non-scientific and excluded from all results above.

## GOAI-AL-KEY-GEOMETRY-01

- Status: completed single-seed local retrospective Key-geometry study (v4 final
  scope correction); non-submission, non-official score, and no change to the frozen
  v2.2 runner or prior geometry results.
- Parent: `GOAI-AL-TRANSFER-GEOMETRY-01`.
- Exact change: replace relation-group averages with a metadata-only, globally
  disjoint directed donor-condition × withheld-target-condition matrix.  The shared
  256-condition baseline is required to cover all identity/time categorical levels,
  so appending a donor cannot alter feature masking.
- Planned data/scope: candidate-only 2,670 conditions from 3,337 official-train;
  4,422 proteins; metadata SHA256
  `9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef`; proteome
  SHA256 `a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb`.
  Panel is target=96, donor=148, baseline=256, selected without response values;
  Direct identity+time, model seed 42, 80 epochs.
- Validation plan: complete exact matrix from `B` and each `B∪{i}` refit, then
  response-independent five-fold held-out-entry matrix completion.  Compare ranks
  1/2/4/8/16 with factor-only, factor+compositional-pair, and general directed
  low-rank structure; never describe it as zero-shot target prediction.
- Status receipt: source/unit test mechanics complete; smoke and formal execution
  results pending.  No official score or submission.

### Correction: v1 structure optimizer rejection; exact measurements retained

- The first formal output `results/key_geometry-20260825-v1/` completed its 149
  Direct fresh fits and 14,208 exact donor×target measurements.  The exact matrix,
  panel, and fit receipts are retained; independent checks found complete unique cells,
  no proxy entries, disjoint roles, and an exact normalized-loss transfer identity.
- Its neural nonconvex structure optimizer produced a few amplitude-exploding
  predictions (high rank correlation but invalid extreme negative squared-error R²).
  That structural analysis is rejected and must not supply a Key conclusion.
- Revision v2 will reuse only the immutable exact measurement matrix and replace that
  optimizer with train-only two-way-residual masked truncated SVD plus ridge-regularized
  directed factor and pair-relation kernels.  It does not retrain a Direct predictor.

### Completion: v2 stable reanalysis

- v1 Direct measurement completed 149 fresh fits on A100 GPU0 from
  `2026-08-25T15:11:24Z` to `15:16:49Z`.  v2 completed from
  `15:29:25Z` to `15:30:13Z`, reusing exactly the same immutable matrix rather
  than retraining a predictor.  Final output:
  `results/key_geometry-20260825-v2/`; exact matrix SHA256
  `504796b0d346bf39d7e59e7c6bfe4277b7422a4263a03a64282b673665bcf2ad`.
- Integrity: 96 target / 148 donor / 256 baseline roles are disjoint; all 14,208
  exact cells are unique and finite; recomputed normalized-loss transfer identity has
  maximum error `5.38e-16`; full baseline factor coverage and all donor feature-map
  invariance checks pass.  Five preassigned held-out-entry folds contain
  2917/2831/2808/2824/2828 cells; every stored prediction's `fit_fold` label
  equals its held-out `cv_fold`, and that fold's entries were excluded from fitting.
- Held-out relation result, R² over train-only donor+target intercepts: general masked
  SVD ranks 1/2/4/8/16 = `-0.279/-0.278/-0.461/-0.525/-0.369`, with all
  target-bootstrap 95% CIs below zero.  Full-matrix 16D residual energy=0.865 is
  descriptive only, not predictive evidence.
- Structured result: factor-only `0.00115 [-0.21748,0.14572]`; factor plus
  ridge-regularized directed pair-relation blocks `0.01449 [-0.18769,0.16649]`;
  general r8 `-0.52458 [-0.72872,-0.35875]`.  Neither stable factor nor pairwise
  interaction transfer was detected, so factor coordinate dimensions 1/2/4 were not run.
- Decision: **Case E, no stable static low-dimensional geometry**.  Do not promote a
  Factor, Factor+Interaction, or General static information Key.  The one future
  research prototype is a dynamic learner-explorer, subject to independent confirmation;
  this does not modify the frozen v2.2 Random policy or any submission lineage.
- Artifacts include the requested matrix/proxy-status/rank/kernel/dimension CSVs,
  four core figures, and `GOAI_KEY_GEOMETRY_SUMMARY.md`.  Current source SHA256
  `3f9f636bfdb43ad4b5dfa7edcaf59d0000bdba0542eb3ffecba589aca7c17d46`; full tests
  `123 passed`; official score/submission: none.

### Final text correction: v3 delivery

- v2's exact-matrix, rank, kernel, dimension, and figure artifacts are valid.  Its
  summary wording briefly described the interaction point difference as a shared gain
  even though the target-bootstrap CI crossed zero.  That prose is rejected.
- `results/key_geometry-20260825-v3/` reruns the stable analysis only, from the same
  immutable v1 exact matrix, at `2026-08-25T15:34:58Z`--`15:35:46Z`; no Direct model
  was retrained.  The corrected summary says that the pair-relation interaction has
  **not** shown stable positive held-out gain.  Final summary SHA256
  `843b2b13685aa77b65bc303be83a941c0142b27ac870249259a0ad70bbc7fede`; current source
  SHA256 `495ec9a9e53282ac16147a83e0d31b0a748fed8f1a6855e3ce4cfa5ddbb1d9dc`.
- Final conclusion remains Case E / dynamic learner-explorer only; no tensor or static
  Key promotion, no v2.2 policy change, and no official score or submission.

### Final scope correction: v4 delivery

- Independent audit found no P0 target or held-out-cell leakage.  The reused immutable
  v1 exact matrix remains 14,208 unique, finite cells (SHA256
  `504796b0d346bf39d7e59e7c6bfe4277b7422a4263a03a64282b673665bcf2ad`), with transfer
  identity maximum error `5.377642775528102e-16`; all seven structural predictors
  scored every held-out edge under its matching excluded fold.
- `results/key_geometry-20260825-v4/` completed stable reanalysis only from
  `2026-08-25T15:42:36Z` to `15:43:25Z`; it did not retrain Direct.  Source SHA256
  `4e79b5497f57159f18c4322b6ca078df88e1fb2fc79d271842180e3b1e14a2ee`; rank/kernel/
  dimension/summary SHA256 `63132cce467e65305419113487a57019d385d808f6a0e2242c96492cebe914cb` /
  `3b0e8b67731981ac93d913f7218f16b95d89d6cfc4e99ab02f91fdd1135b4334` /
  `f8ac34411d9db9a0e217b1d837b701e3f441520e2439b9f24add4c5cf73aba65` /
  `737e777d84b6d8953a1bea80051c74666a823c5c09e9ea4779976a272ee18686`.
- Terminology corrected: folds are preassigned response-independent,
  stable-hash-balanced, relation-pattern-conditioned held-out entries.  The general
  comparator is iterative zero-filled masked truncated SVD and structured-kernel ridge
  strengths are fixed; negative R² only rejects stable completion by these tested
  estimators, not every conceivable static parameterization.
- Scope-correct final decision: **Case E means no stable static low-dimensional
  geometry was detected for this 256-condition shared-baseline, single-seed Direct
  identity+time, 80-epoch matrix under tested ranks/kernels.**  It does not rule out
  another support state, semantic predictor, or lower-noise measurement.  Dynamic
  learner-explorer is the next hypothesis/prototype to test, not a proven unique design.
  Full tests: `124 passed`; no policy/submission/official-score change.
