# GOAI-AL v2.2 final report source notes

## Reporting job

- Audience: technical.
- Decision: whether the implemented condition-level GOAI active-learning framework is trustworthy enough to retain a default acquisition policy and define the next experimental stage.
- Scope: v2.2 five-seed formal run, v2.2 smoke gate, v2.1 Direct-versus-rank-64 comparison, post-hoc control/low-rank/tensor audits.
- Primary endpoint: interpolation `delta_skill_zero` normalized AULC over registered budgets 128, 256, 512, and 1,024.
- Comparison basis: paired seeds 42–46 with the same initial IDs and model/acquisition seed function.
- Delivery mode: one self-contained portable HTML report generated from the canonical `artifact.json` contract.

## Required-structure mapping

The technical-report jobs map to visible report sections as follows:

- Technical summary: report block `technical_summary`.
- Definitions and measurement: sections 1, 3, and 9.
- Methodology and model design: sections 2, 4, and 5.
- Key findings with visual evidence: sections 6, 7, and 8.
- Validation, uncertainty, and limitations: sections 3, 7, 8, and 9.
- Recommendations and open questions: section 10.

The user's requested ten-part structure is preserved one-for-one as visible numbered sections.

## Chart map

| Report segment | Question | Family | Fields | Supported claim |
|---|---|---|---|---|
| Section 6 | How does response skill change at the four registered budgets? | Grouped bar | budget, strategy, five-seed mean | CoreSet is directionally highest after the shared initial budget; Uncertainty lags Random at intermediate budgets. |
| Section 8 | Does the combined representation help across evaluation regimes at full budget? | Grouped bar | split, representation, five-seed mean | Representation effects are split-dependent: cold-start improves, interpolation slightly worsens. |
| Section 8 | How much response energy is captured by low ranks? | Grouped bar | rank, centering/standardization variant, cumulative energy | Rank 64 captures only about 53–54%; compression exists but is not strong enough to justify replacing Direct. |

Four checkpoint observations are too sparse for a trend line, so the acquisition comparison uses grouped bars. Exact confidence intervals and paired decisions are shown in tables rather than encoded as decorative chart elements.

## Omitted quantitative visuals

- Split counts, protocol settings, metrics, and tensor fibers are tables because exact lookup matters more than shape.
- The Water–DMSO sensitivity audit is stated near the control contract and not charted because there is no decision threshold or comparator series.
- B80 is not plotted because all 15 seed×policy curves are `not_reached`; no extrapolated budgets are manufactured.
- No condition/response-coverage score is shown because it was not preregistered or computed independently of acquisition geometry.
- No hit/discovery visual is shown because biological hit labels do not exist and the protocol explicitly removed Impact/Hit Ratio.

## Interpretation boundaries

- All scores are retrospective local proxies, not organizer official scores or submissions.
- `delta_skill_zero` uses the matched-control no-response delta=0 baseline.
- Full reference is the same predictor fitted on all 2,670 candidate conditions; it is an empirical reference, not a theoretical upper bound.
- Semantic ablation lacks a shuffled-semantic negative control; it supports a fixed representation contrast, not semantic causality.
- Five strain mappings are high-confidence public candidates rather than organizer-verified identities; DHY210 continuous semantics are zero/missing.
- v2.1 Direct-versus-low-rank is a single-seed, same-ID feasibility comparison and is not pooled with v2.2 five-seed estimates.
- Auxiliary split and representation confidence intervals are descriptive and unadjusted for multiple comparisons; the only policy replacement gate is the preregistered interpolation AULC rule.

## Reproducibility anchors

- Formal root manifest file SHA-256: `b51658c85c75fb1ac5548347b6fb3906d19534ffa478c4026a3fe0bfad3bf6e0`.
- Formal root manifest payload SHA-256: `29832d7de5f51607e71a70b112b9070d9fbe10fd6e0162a926805ca3817259d3`.
- Formal run identity: `7e75eddbb1c563357181cd4b3aae9c794012491054dba4a4d65b278ca9abb796`.
- Source snapshot payload: `fcab6c2fc787c5cf448e7586b52971da0a5ee39c9ec85a8f8118e1df6635663f`.
- Frozen runner/config/spec SHA-256: `89f5f62aa328ca7f12e34d8f638defe279c84263ab739bf485382efb22f61b82`, `ce772f6d3cb03451b55454811be3f32e8088c7c1a4e59fce63646e242c2a19d7`, `4f870cb240dcbf1124867363539d500d25c5ad2b03260cbd4fd4c38afb6962d2`.
- Full source test gate: 104/104 passed before the immutable formal snapshot was executed.
- Independent formal audit: root inventory 178 files; exact metric/acquisition/receipt grids and all five seed payload hashes passed.
- Root aggregate CSVs may differ from seed CSV decimal round-trips by at most `8.88e-16`; recomputed statistics and decisions are unchanged.

The report intentionally does not rerun `--resume` after the final audit because doing so would legitimately update root command-history metadata.
