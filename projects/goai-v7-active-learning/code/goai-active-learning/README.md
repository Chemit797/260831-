# GOAI Active Learning v2 Pilot

This repository runs the frozen, condition-atomic v2 retrospective active-learning
pilot described in `FRAMEWORK_SPEC.md`. It does not copy or modify the released
GOAI data. The registered model is `GOAI-AL-V2-PILOT-01`.

## v2.2 formal follow-up

The completed five-seed follow-up is registered as
`GOAI-AL-V22-DIRECT-SEMANTIC-01` and frozen by
[`FRAMEWORK_SPEC_V22.md`](FRAMEWORK_SPEC_V22.md). Its self-contained final
technical report is
[`reports/goai_al_v22_final/report.html`](reports/goai_al_v22_final/report.html),
with the canonical source artifact and delivery receipt in the same directory.
The preregistered decision is to retain Random: CoreSet improved the mean primary
AULC in all five paired seeds, but its 95% paired confidence interval crossed
zero; Uncertainty underperformed Random. These are local retrospective proxy
results, not an official GOAI score, biological-hit claim, or submission result.

The frozen-protocol material below documents the historical v2.1 rank-64 pilot.
Use `FRAMEWORK_SPEC_V22.md`, `configs/direct_multiseed.yaml`, and the final report
for the completed v2.2 Direct experiment.

## Frozen protocol

One query is one biological condition identified by strain, chemical (without a
public concentration field), medium, temperature, perturbation time, and time
unit. A query reveals the full 4,422-protein matched-control `log2`-delta
response. Water/DMSO controls are exact-context assay overhead and are never
query candidates, predictor features, or acquisition features.

The condition ID is independent of `split_final`. Official-train conditions are
split deterministically with seed 42 into a candidate pool and a level-preserving
20% interpolation holdout. For a condition with train provenance, only train
treatment rows form its response, replicate count, and measurement summaries;
overlapping validation labels are retained only as audit provenance and then
discarded, never merged. Conditions overlapping official train are removed from
each official validation split. The interpolation holdout is the primary
learning-curve split; cleaned `val_chem_only`, `val_strain_only`, `val_both`, and
`val_time` results are reported separately.

The target-free encoder one-hots exactly four categorical axes: strain, chemical,
medium, and temperature. Perturbation time and unit are converted to one continuous
minutes column; unknown units are rejected. All released exact-context controls
remain assay overhead even when their split differs from the treatment split.

Formal mode has exactly three strategies: `random`, `coreset`, and `uncertainty`.
They share seed 42, the same deterministic initial 128 IDs, fixed 128-condition
acquisition batches, and checkpoints 128, 256, 512, and 1,024. A fresh rank-64
low-rank dropout MLP is fitted at every batch budget, including 384, 640, 768,
and 896 even though those intermediate fits are not evaluated. Its seed depends
only on global seed and current budget. Formal fits use 80 epochs and uncertainty
uses 8 MC-dropout passes. The registered objective is the rank-64 masked
natural-delta reconstruction loss. A same-backbone full-pool fit supplies the achievable-
improvement reference. Direct and rank-64 representations are compared only on
identical nested-random IDs at 128, 512, and the full pool; there is no rank sweep.

Smoke mode still runs all three strategies with seed 42, initial budget 32,
batch size 32, checkpoints 32/64/96, 2 epochs, and 2 MC passes. It is an execution
check only and is explicitly non-scientific.

## Environment and exact commands

Use the existing Biohub Python 3.12 environment. The formal config requests CUDA;
select an available GPU explicitly.

```bash
cd /home/chenyuming/Project/active-learning/goai_active_learning
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
BIOHUB_PYTHON="/home/chenyuming/Project/Biohub - Cell Tracking During Development/.venv/bin/python3.12"

"$BIOHUB_PYTHON" -m pytest

"$BIOHUB_PYTHON" -m goai_al.experiment \
  --config configs/pilot.yaml \
  --smoke \
  --output-suffix smoke-20260824-a

"$BIOHUB_PYTHON" -m goai_al.experiment \
  --config configs/pilot.yaml \
  --output-suffix formal-20260824-a
```

Instead of `--output-suffix`, pass a unique explicit directory:

```bash
"$BIOHUB_PYTHON" -m goai_al.experiment \
  --config configs/pilot.yaml \
  --smoke \
  --output-dir /absolute/path/to/a/new-attempt
```

Exactly one output selector is required. The controller refuses a nonempty
existing target, so a completed or failed attempt cannot be silently reused.

## Leakage protections

- `PoolFeatureEncoder` is fitted only on candidate-pool metadata. Its continuous
  time column remains available when unsupported categorical levels are masked.
- Each strategy owns a separate `RetrospectiveOracle` containing only candidate-
  pool responses. Labels enter a strategy only through `oracle.reveal(ids)`;
  repeated and evaluation-ID reveals fail.
- Acquisition receives only immutable public IDs, target-free descriptors,
  labelled IDs, and optional predictor uncertainty through `AcquisitionContext`.
- Target mean/scale, missing-value handling, response SVD basis, and model weights
  are fitted from that round's revealed responses only. Hidden full-pool labels
  do not enter acquisition or active-model response preprocessing.
- Full-pool reference, representation comparison, and low-rank/tensor audit are
  clearly separated reference or post-hoc outputs and never acquisition inputs.

## Output contract

Every attempt begins with `manifest.json` in `running` state and ends in
`complete` or `failed`. It records the exact command, mode, data/config/all-source
hashes, environment, seed, query/control/split contracts, protocol, and a hashed
artifact inventory. JSON, CSV, and per-round receipt writes are atomic.

The registered attempt outputs are:

- `active_metrics.csv`: exact v2 metrics at sparse checkpoints for interpolation
  and every cleaned official validation split.
- `acquisitions.csv`: public selection receipts only; it contains no oracle
  labels, response values, impact fields, or hit fields.
- `full_reference_metrics.csv`: same-backbone full-pool reference scores.
- `model_fit_receipts.csv`: fit scope, budget, seed, timing, rank, and basis hash.
- `representation_metrics.csv`: formal direct-versus-low-rank comparison; an
  empty schema-bearing file in smoke mode.
- `split_assignments.csv`: condition-level candidate/evaluation/exclusion roles.
- `analysis_summary.json`: real-spacing normalized trapezoidal AULC and the first
  interpolated budget reaching 80% achievable improvement, or explicit
  `not_reached` without extrapolation.
- `learning_curve_delta_skill_zero.png`,
  `learning_curve_condition_pcc_median.png`, and
  `learning_curve_protein_r2_median.png`: primary interpolation curves.
- `data_audit.json`, `tensor_coverage.csv`, and `low_rank_spectrum.csv`: post-hoc
  oracle audit written before experiment fitting.
- `round_receipts/<strategy>/round_*.json`: one atomic, standalone, label-free receipt
  per fitted budget, including transition/full-labelled IDs and hashes, seeds,
  checkpoint status, fit summary, timing, and current split metrics (or an empty list).

All response scores are local GOAI-AL matched-control log2-delta diagnostics
computed by `score_response`. In particular, `delta_skill_zero` is a local proxy,
not an official GOAI metric or leaderboard score. No biological-discovery claim
is supported by this pilot alone.

## Legacy warning

Files already present under `results/pilot_v1` are legacy and remain unchanged.
Their Impact, Hit Ratio/hit recall, hybrid policies, dense budget sweeps, and
split-dependent grouping are obsolete under v2 and must not be combined with or
presented as frozen v2 results.
