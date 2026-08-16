#!/bin/bash
set -euo pipefail

WAIT_SERVICE="${WAIT_SERVICE:?WAIT_SERVICE is required}"
WAIT_LOG="${WAIT_LOG:?WAIT_LOG is required}"
COMPLETION_PATTERN="${COMPLETION_PATTERN:?COMPLETION_PATTERN is required}"
START_SERVICE="${START_SERVICE:?START_SERVICE is required}"
POLL_SECONDS="${POLL_SECONDS:-30}"

echo "Waiting for ${WAIT_SERVICE} to finish cleanly before starting ${START_SERVICE}."
while true; do
  status="$(supervisorctl status "$WAIT_SERVICE" 2>&1 || true)"
  state="$(awk '{print $2}' <<<"$status")"
  case "$state" in
    RUNNING|STARTING|STOPPING)
      sleep "$POLL_SECONDS"
      ;;
    EXITED|STOPPED)
      if ! grep -Fq "$COMPLETION_PATTERN" "$WAIT_LOG"; then
        echo "${WAIT_SERVICE} stopped without completion marker: ${COMPLETION_PATTERN}" >&2
        exit 3
      fi
      target_status="$(supervisorctl status "$START_SERVICE" 2>&1 || true)"
      target_state="$(awk '{print $2}' <<<"$target_status")"
      if [[ "$target_state" != "STOPPED" && "$target_state" != "EXITED" ]]; then
        echo "Refusing to start ${START_SERVICE}; current state is ${target_state}." >&2
        exit 4
      fi
      echo "Completion marker found; starting ${START_SERVICE}."
      supervisorctl start "$START_SERVICE"
      exit 0
      ;;
    *)
      echo "Unexpected supervisor state for ${WAIT_SERVICE}: ${status}" >&2
      exit 5
      ;;
  esac
done
