#!/usr/bin/env bash
set -euo pipefail

# Create a *private, encrypted* Codex snapshot.  This script deliberately
# refuses to write an unencrypted archive and deliberately excludes all
# authentication/configuration material.  Run it only when Codex is quiet for
# the final snapshot; SQLite files are copied through sqlite3 .backup.

usage() {
  cat <<'EOF'
Usage:
  create_codex_encrypted_snapshot.sh \
    --codex-root /home/USER/.codex \
    --output-dir /safe/staging \
    --age-recipient age1... \
    [--label YYYYMMDD]

The recipient is an age PUBLIC key only. Never pass a private key, OAuth
token, API key, password, or rclone configuration to this script.
EOF
}

codex_root=""
output_dir=""
age_recipient=""
snapshot_label="$(date -u +%Y%m%d)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --codex-root) codex_root="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --age-recipient) age_recipient="$2"; shift 2 ;;
    --label) snapshot_label="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$codex_root" ]] || { printf 'Missing Codex root.\n' >&2; exit 2; }
[[ -n "$output_dir" ]] || { printf 'Missing output directory.\n' >&2; exit 2; }
[[ "$age_recipient" =~ ^age1[0-9a-z]+$ ]] || { printf 'Expected an age public recipient (age1...).\n' >&2; exit 2; }
command -v age >/dev/null || { printf 'age is required for client-side encryption.\n' >&2; exit 2; }
command -v sqlite3 >/dev/null || { printf 'sqlite3 is required for consistent database backups.\n' >&2; exit 2; }
command -v zstd >/dev/null || { printf 'zstd is required for archive compression.\n' >&2; exit 2; }

mkdir -p "$output_dir"
work_dir="$(mktemp -d "${output_dir}/codex-snapshot-${snapshot_label}.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

recovery_dir="$work_dir/recovery-kit"
live_dir="$work_dir/live-context"
provider_dir="$work_dir/provider-history"
mkdir -p "$recovery_dir" "$live_dir" "$provider_dir"

# These copied files carry useful local state but no authentication.  Do not
# copy config.toml*: server-specific relay/provider settings can contain keys.
for item in rules skills worktrees visualizations session_index.jsonl history.jsonl archived_sessions; do
  if [[ -e "$codex_root/$item" ]]; then
    rsync -a --safe-links --exclude='.git' --exclude='*.env' --exclude='*.key' --exclude='*.token' \
      "$codex_root/$item" "$recovery_dir/"
  fi
done

# Make transactionally consistent copies of state databases.  WAL/SHM files
# are intentionally not copied because .backup materializes a stable database.
for db_name in goals_1.sqlite state_5.sqlite memories_1.sqlite queue_1.sqlite; do
  if [[ -f "$codex_root/$db_name" ]]; then
    sqlite3 "$codex_root/$db_name" ".backup '$recovery_dir/$db_name'"
  fi
done

# Live history: raw sessions/attachments plus consistent index/log backups.
for item in sessions attachments archived_sessions; do
  if [[ -e "$codex_root/$item" ]]; then
    rsync -a --safe-links "$codex_root/$item" "$live_dir/"
  fi
done
for db_name in logs_2.sqlite thread_history_1.sqlite; do
  if [[ -f "$codex_root/$db_name" ]]; then
    sqlite3 "$codex_root/$db_name" ".backup '$live_dir/$db_name'"
  fi
done

# Retain provider migration histories while permanently omitting their config.
for provider_path in "$codex_root"/provider-*-backup-*; do
  [[ -d "$provider_path" ]] || continue
  rsync -a --safe-links --exclude='config.toml' --exclude='config.toml.*' --exclude='auth.json' \
    --exclude='secrets/' --exclude='cache/' --exclude='.tmp/' --exclude='tmp/' \
    "$provider_path" "$provider_dir/"
done

archive_and_encrypt() {
  local package_name="$1"
  local source_dir="$2"
  local tar_path="$output_dir/${package_name}-${snapshot_label}.tar.zst"
  local encrypted_path="${tar_path}.age"
  tar --zstd -cf "$tar_path" -C "$work_dir" "$(basename "$source_dir")"
  zstd -t "$tar_path" >/dev/null
  age -r "$age_recipient" -o "$encrypted_path" "$tar_path"
  sha256sum "$encrypted_path" > "${encrypted_path}.sha256"
  rm -f "$tar_path"
  printf '%s\n' "$encrypted_path"
}

archive_and_encrypt codex-recovery-kit "$recovery_dir"
archive_and_encrypt codex-live-context "$live_dir"
archive_and_encrypt codex-provider-history "$provider_dir"

printf 'Encrypted snapshot completed. Upload only the .age and .sha256 files; keep the age private key outside Drive.\n'
