#!/usr/bin/env bash
# Install PyTorch for the selected hardware, then resolve framework dependencies
# from pyproject.toml while constraining the installed backend wheels.
# Usage: ./scripts/install_backend.sh [auto|cpu|mps|cuda|rocm|rocm6.4|rocm7.0] [core|vision] [--dry-run]
# PYTHON selects the interpreter; PIP_FLAGS passes optional pip flags.
set -euo pipefail

CF_BACKEND="auto"
CF_PROFILE="core"
CF_DRY_RUN=false
CF_POSITION=0
for CF_ARG in "$@"; do
  case "$CF_ARG" in
    --dry-run) CF_DRY_RUN=true ;;
    --help|-h)
      echo "usage: $0 [auto|cpu|mps|cuda|rocm|rocm6.4|rocm7.0] [core|vision] [--dry-run]"
      exit 0 ;;
    *)
      if [[ "$CF_POSITION" == 0 ]]; then CF_BACKEND="$CF_ARG";
      elif [[ "$CF_POSITION" == 1 ]]; then CF_PROFILE="$CF_ARG";
      else echo "unexpected argument: $CF_ARG" >&2; exit 2; fi
      CF_POSITION=$((CF_POSITION + 1)) ;;
  esac
done
[[ "$CF_PROFILE" == core || "$CF_PROFILE" == vision ]] || {
  echo "unknown profile: $CF_PROFILE (expected core or vision)" >&2; exit 2;
}
CF_PYTHON="${PYTHON:-python3}"
CF_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CF_ROOT"
CF_PIP_FLAGS=()
if [[ -n "${PIP_FLAGS:-}" ]]; then read -r -a CF_PIP_FLAGS <<< "$PIP_FLAGS"; fi

if [[ "$CF_BACKEND" == auto ]]; then
  if [[ "$(uname -s)" == Darwin ]]; then CF_BACKEND=mps;
  elif [[ -d /opt/rocm ]]; then CF_BACKEND=rocm;
  elif command -v nvidia-smi >/dev/null 2>&1; then CF_BACKEND=cuda;
  else CF_BACKEND=cpu; fi
fi
CF_INDEX=()
CF_EXTRA=""
case "$CF_BACKEND" in
  cpu) CF_INDEX=(--index-url https://download.pytorch.org/whl/cpu) ;;
  mps)
    [[ "$(uname -s)" == Darwin ]] || { echo "mps requires macOS" >&2; exit 2; } ;;
  cuda) CF_EXTRA=cuda ;;
  rocm|rocm6.4)
    CF_EXTRA=rocm
    CF_INDEX=(--index-url https://download.pytorch.org/whl/rocm6.4) ;;
  rocm7.0)
    CF_EXTRA=rocm
    CF_INDEX=(--pre --index-url https://download.pytorch.org/whl/nightly/rocm7.0) ;;
  *) echo "unknown backend: $CF_BACKEND" >&2; exit 2 ;;
esac
if [[ "$CF_BACKEND" == rocm* && "$(uname -s)" != Linux ]]; then
  echo "rocm requires Linux" >&2; exit 2
fi
CF_TORCH_PACKAGES=(torch)
if [[ "$CF_PROFILE" == vision ]]; then
  CF_TORCH_PACKAGES+=(torchvision)
  CF_EXTRA="${CF_EXTRA:+$CF_EXTRA,}image,detection,export"
fi
CF_SPEC="."
if [[ -n "$CF_EXTRA" ]]; then CF_SPEC=".[$CF_EXTRA]"; fi
cf_run() {
  if "$CF_DRY_RUN"; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
echo "[complexity] backend=$CF_BACKEND profile=$CF_PROFILE python=$CF_PYTHON"
cf_run "$CF_PYTHON" -m pip install ${CF_PIP_FLAGS[@]+"${CF_PIP_FLAGS[@]}"} "${CF_TORCH_PACKAGES[@]}" ${CF_INDEX[@]+"${CF_INDEX[@]}"}
if "$CF_DRY_RUN"; then
  echo "[complexity] constrain torch/torchvision to the versions installed above"
  cf_run "$CF_PYTHON" -m pip install ${CF_PIP_FLAGS[@]+"${CF_PIP_FLAGS[@]}"} -c '<backend-constraints>' -e "$CF_SPEC"
  exit 0
fi
CF_CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CF_CONSTRAINTS"' EXIT
"$CF_PYTHON" - "$CF_CONSTRAINTS" "${CF_TORCH_PACKAGES[@]}" <<'PYEOF'
import importlib.metadata
import sys
from pathlib import Path
Path(sys.argv[1]).write_text("".join(
    f"{name}=={importlib.metadata.version(name)}\n" for name in sys.argv[2:]
), encoding="utf-8")
PYEOF
cf_run "$CF_PYTHON" -m pip install ${CF_PIP_FLAGS[@]+"${CF_PIP_FLAGS[@]}"} -c "$CF_CONSTRAINTS" -e "$CF_SPEC"
"$CF_PYTHON" - <<'PYEOF'
import complexity
import torch
print(f"complexity={complexity.__version__} torch={torch.__version__}")
print(f"CUDA/ROCm available={torch.cuda.is_available()}")
print(f"MPS available={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
PYEOF
