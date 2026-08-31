#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER="$ROOT_DIR/reproduction/pilot_controller.py"
PYTHON_BIN="${DISCOBAX_SYSTEM_PYTHON:-$(command -v python3)}"
RUNTIME_CONTROL="/run/user/$(id -u)/discobax-pilot-il2"
if [[ ! -d "$(dirname "$RUNTIME_CONTROL")" || ! -w "$(dirname "$RUNTIME_CONTROL")" ]]; then
  RUNTIME_CONTROL="/tmp/discobax-pilot-il2-$(id -u)"
fi
LAUNCHER_LOG="$RUNTIME_CONTROL/launcher.log"

mkdir -p "$RUNTIME_CONTROL"

start_run() {
  local scope="${1:-pilot}"
  if [[ "$scope" != "pilot" ]]; then
    echo "ERROR: this controller only starts the 24-job 'pilot' scope." >&2
    echo "It will never implicitly start the 1,500-job paper matrix." >&2
    return 2
  fi
  export CUDA_VISIBLE_DEVICES=0
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  {
    echo
    echo "===== $(date -Is) launcher start scope=pilot ====="
  } >> "$LAUNCHER_LOG"
  nohup setsid "$PYTHON_BIN" "$CONTROLLER" run >> "$LAUNCHER_LOG" 2>&1 < /dev/null &
  local launcher_pid=$!
  sleep 2
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    # A duplicate launcher exits immediately because the live controller owns
    # the advisory lock.  Status gives the authoritative outcome either way.
    echo "Launcher exited; checking controller state."
  else
    echo "Pilot controller launched in a detached session (PID $launcher_pid)."
    echo "SSH may disconnect safely."
  fi
  status_run
}

status_run() {
  "$PYTHON_BIN" "$CONTROLLER" status
}

follow_run() {
  trap 'echo; echo "Stopped watching; the background pilot is unchanged."; exit 0' INT TERM
  while true; do
    printf '\033[2J\033[H'
    date -Is
    status_run
    echo
    echo "Refreshes every 30 seconds. Ctrl-C only closes this view."
    sleep 30
  done
}

stop_run() {
  "$PYTHON_BIN" "$CONTROLLER" stop
}

doctor_run() {
  "$PYTHON_BIN" "$CONTROLLER" doctor
}

summarize_run() {
  "$PYTHON_BIN" "$CONTROLLER" summarize
}

usage() {
  cat <<'EOF'
Usage:
  bash run_reproduction.sh start pilot
  bash run_reproduction.sh status
  bash run_reproduction.sh follow
  bash run_reproduction.sh stop
  bash run_reproduction.sh resume
  bash run_reproduction.sh summarize
  bash run_reproduction.sh doctor

With no arguments, the script starts only the 24-job pilot scope.

Optional overrides:
  DISCOBAX_PILOT_STORE    persistent results/logs root
  DISCOBAX_PILOT_RUNTIME  disposable environment/scratch root
  DISCOBAX_GPU_IDLE_SECONDS, DISCOBAX_GPU_POLL_SECONDS
  DISCOBAX_SYSTEM_PYTHON
EOF
}

command="${1:-start}"
case "$command" in
  start) start_run "${2:-pilot}" ;;
  resume) start_run pilot ;;
  status) status_run ;;
  follow) follow_run ;;
  stop) stop_run ;;
  doctor) doctor_run ;;
  summarize) summarize_run ;;
  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac
