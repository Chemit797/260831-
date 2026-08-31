# Research continuity repository

This repository is the durable knowledge and recovery index for research previously hosted on a laboratory server.

## Required reading order

Before inspecting code, downloading artifacts, or proposing research work, read:

1. `00_请先读我_服务器迁移总蓝图.md`
2. `MIGRATION_STATUS.md`
3. `docs/01_研究总览.md`
4. `projects/goai-v7-active-learning/CURRENT_STATE.md` and `NEXT_STEPS.md`
5. `docs/06_Codex_完整归档与恢复指南.md` only when Codex recovery/history matters.
6. The relevant artifact/dataset manifest before requesting binary files.

## Current research priority

The active research thread is perturbation-based active learning on the mature GO-AI V7 baseline. GeneDisco, Biohub Kaggle, and GO-AI stage 1 are historical context unless a task explicitly concerns them.

## Migration safety rules

- Treat the server as the source until an artifact has a recorded destination, size, and SHA-256 verification.
- Never upload, print, commit, or request `auth.json`, `secrets/`, OAuth tokens, API keys, SSH keys, or Kaggle credentials.
- Do not delete source files, caches, worktrees, or old runs merely because they are large. Only delete after an explicit reviewed migration manifest authorizes it.
- Prefer structured documents, ledgers, manifests, and checksums over raw session history.
- A full Codex archive is a private encrypted evidence store, not the source of truth for research decisions.

## State of this repository

This is an in-progress migration control repository. Documentation may exist before code/data/artifacts have been uploaded. Do not infer that a manifest entry has been migrated until its status says verified.
