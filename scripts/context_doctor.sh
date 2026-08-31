#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
required=(
  "README.md"
  "00_请先读我_服务器迁移总蓝图.md"
  "AGENTS.md"
  "docs/01_研究总览.md"
  "docs/04_存储与访问说明.md"
  "projects/goai-v7-active-learning/CURRENT_STATE.md"
  "projects/goai-v7-active-learning/NEXT_STEPS.md"
  "manifests/artifacts.yaml"
)

missing=0
for relative_path in "${required[@]}"; do
  if [[ -f "${repo_root}/${relative_path}" ]]; then
    printf 'OK      %s\n' "${relative_path}"
  else
    printf 'MISSING %s\n' "${relative_path}" >&2
    missing=1
  fi
done

printf '\nRead in order:\n'
printf '  1. 00_请先读我_服务器迁移总蓝图.md\n'
printf '  2. docs/01_研究总览.md\n'
printf '  3. projects/goai-v7-active-learning/CURRENT_STATE.md\n'
printf '  4. projects/goai-v7-active-learning/NEXT_STEPS.md\n'
printf '  5. manifests/artifacts.yaml\n'

if [[ "${missing}" -ne 0 ]]; then
  printf '\nRepository context is incomplete; do not start a new experiment yet.\n' >&2
  exit 1
fi

printf '\nContext documentation is present. Verify artifact status and SHA-256 before using binary files.\n'
