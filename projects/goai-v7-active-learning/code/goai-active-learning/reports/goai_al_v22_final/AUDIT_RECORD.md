# GOAI-AL v2.2 independent audit record

Audit date: 2026-08-24 UTC  
Scope: `results/direct_multiseed_formal-20260824-v22`

## Artifact integrity verdict

Verdict: **GO; P0=0, P1=0**.

- Root manifest file SHA-256: `b51658c85c75fb1ac5548347b6fb3906d19534ffa478c4026a3fe0bfad3bf6e0`.
- Root manifest payload SHA-256: `29832d7de5f51607e71a70b112b9070d9fbe10fd6e0162a926805ca3817259d3`.
- Run identity: `7e75eddbb1c563357181cd4b3aae9c794012491054dba4a4d65b278ca9abb796`.
- Source snapshot payload: `fcab6c2fc787c5cf448e7586b52971da0a5ee39c9ec85a8f8118e1df6635663f`.
- All 178 root inventory entries, 10 source-snapshot payload files, and 29 payload files per seed were rehashed successfully.
- Exact grids were independently verified: 300 active-metric rows, 25 full-reference rows, 150 ablation rows, 15,360 acquisition rows, 15 curves, 1,529 aggregate rows, 2 paired-policy rows, and 225 representation-summary rows.
- Every seed×strategy selected 1,024 unique IDs from the 2,670-condition candidate pool with no repeats and zero intersection with 2,250 evaluation IDs.
- All three strategies shared the same ordered initial 128 IDs within each seed.
- Receipt forbidden-key scan found no labels, truth, predictions, model state, impact, or oracle-response values.
- Resume skipped seeds 42–46 after hash validation and did not rewrite seed payload files.

Seed payload SHA-256 values:

- 42: `b73e5a79a284e1f8fc0c96fb836217713446245c6b2da4e7faa8c21a2cc7e01e`
- 43: `76f162f2cd9ceebc200e63381b949c70f8212be5526d78b7aff74b48559ee47b`
- 44: `2beb151bc1886b2fa0fa6feeaf82840e8e784ace9ba9d816a664576f1b86aadb`
- 45: `d601ed9dcbeb79dc40ca5b7f82f3b4bd3c70f90e37cb41890c91149e1dc25516`
- 46: `ec03e4ae1005dabd204056a0d109a3d55399561bde35e1c7bb74660c8820872e`

The only nonblocking P2 observation is a maximum `8.88e-16` last-bit decimal round-trip difference between root aggregate CSVs and concatenated seed CSVs. Identities, counts, statistics, hashes covered by the contract, and the final decision are unchanged.

## Statistical recomputation verdict

Verdict: **passed; no P0/P1 calculation error or artifact contradiction**.

- Normalized AULC values recomputed from registered budget/value pairs matched saved derivatives within `8.33e-17`.
- All 225 representation-ablation mean, sample-SD, and paired 95% t-confidence rows matched within `1e-16`.
- Active, ablation, and full-reference primary keys had zero duplicates and complete five-seed coverage.
- Full-budget combined-semantic ablation rows matched the same-seed full-reference rows exactly.
- CoreSet minus Random AULC: `+0.0073297798670060475`, 95% CI `[-0.0016457224561160994, 0.016305282190128195]`, wins/losses `5/0`; preregistered replacement gate failed because the CI lower bound was not positive.
- Uncertainty minus Random AULC: `-0.01338261920633887`, 95% CI `[-0.017283921796820444, -0.009481316615857296]`, wins/losses `0/5`.
- All 15 seed×policy B80 values were `not_reached` by budget 1,024; no extrapolation was performed.

## Claim boundary

The audit certifies a retrospective local-proxy benchmark and its saved computations. It does not certify an organizer official score, submission, wet-lab causal effect, biological hit/discovery, or general semantic-representation superiority.
