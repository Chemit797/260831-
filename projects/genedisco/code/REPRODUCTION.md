# DiscoBAX A100 IL-2 pilot reproduction

## Frozen contract

- Paper: ICML 2023, *DiscoBAX: Discovery of Optimal Intervention Sets in Genomic Experiment Design*.
- Declared upstream commit: `84f01283bc7f6ab5f66b5ea2a63632b401cc0402`.
- Zenodo record: `10202590`; archive size `246500391` bytes; MD5
  `9f2fb895e32c85377e4cf1b2d2658ed9`.
- Pilot: `schmidt_2021_il2`, Achilles features, seeds 1000/2000/3000,
  eight methods, batch size 32 and 25 acquisition cycles.
- Paper BAX settings: Top-K `K=2`, Levelset `c=1.5`, DiscoBAX `S=10`,
  with `20/20/20` Monte Carlo settings.

The official loop first samples 32 points and then performs 25 batches of 32,
so a completed run has selected 832 points. CUDA and CPU random number streams
are not bitwise interchangeable; this is an auditable statistical-trend pilot,
not a claim of bitwise reproduction across hardware.

## One-entry controller

Start the gated background workflow:

```bash
bash run_reproduction.sh start pilot
```

The controller creates the pinned environment, audits the exact cache, runs all
tests, validates a 24-job dry-run, runs the full-MC one-cycle DiscoBAX smoke and
only then starts the serial formal matrix. `nohup` and an independent session
keep it alive when SSH disconnects.

```bash
bash run_reproduction.sh status
bash run_reproduction.sh follow
bash run_reproduction.sh stop
bash run_reproduction.sh resume
bash run_reproduction.sh summarize
bash run_reproduction.sh doctor
```

The controller never starts the 1,500-job paper scope and always sets
`CUDA_VISIBLE_DEVICES=0`. Before each job it waits for GPU 0 to be idle for five
minutes. Once a job begins, later foreign GPU activity is recorded but does not
preempt the job.

## Storage and recovery

Because the shared server's root filesystem was nearly full during setup, the
approved default layout is:

```text
/mnt/Omics_GPU/chenyuming/DiscoBAX/pilot_il2_3seed/  # persistent results/logs
/dev/shm/chenyuming-discobax/                       # disposable env/scratch
```

The project-local `results/pilot_il2_3seed` and `logs/pilot_il2_3seed` paths are
symlinks to persistent storage. A server reboot clears `/dev/shm`; `resume`
rebuilds it and validates persistent jobs before deciding what to rerun. These
locations can be overridden with `DISCOBAX_PILOT_STORE` and
`DISCOBAX_PILOT_RUNTIME`.

Every job has immutable attempt directories. A job is promoted to FULL RUN only
after its exit code, 25 cycle artifacts, final performance row, manifest
parameters, finite metrics and log are all validated. Interrupted and failed
attempts are retained and never combined with later attempts.

## Interpretation

Final overall score is computed as
`sqrt(mean_recall_across_seeds * mean_diversity_across_seeds)`. Outputs include
raw per-seed values, population standard deviation, SEM, learning curves and a
clearly labelled paper-reference comparison. IL-2 was deliberately selected
because the paper reports a strong positive result; this three-seed pilot is not
an unbiased all-assay or statistical-significance claim.
