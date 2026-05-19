#!/usr/bin/env bash
# Install complexity-framework with the correct torch backend.
#
# torch is NOT in any pyproject extra (see pyproject.toml for why). This
# script installs torch from the right index, then installs the framework
# extras separately. The result: zero risk of accidentally pulling the CUDA
# wheel on ROCm hosts or vice versa.
#
# Usage:
#   ./scripts/install_backend.sh rocm           # auto-detects rocm6.4 / rocm7.0
#   ./scripts/install_backend.sh rocm6.4
#   ./scripts/install_backend.sh rocm7.0
#   ./scripts/install_backend.sh cuda
#   ./scripts/install_backend.sh cpu
#
# Env vars:
#   PIP_FLAGS   — extra flags forwarded to pip (e.g. "--user --break-system-packages")
#                 The script auto-adds --break-system-packages on Ubuntu 24+.

set -euo pipefail

BACKEND="${1:-}"
if [[ -z "$BACKEND" ]]; then
  echo "usage: $0 {rocm | rocm6.4 | rocm7.0 | cuda | cpu}" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PIP_FLAGS="${PIP_FLAGS:-}"

# Ubuntu 24+ ships a PEP-668 marker — pip refuses to write to system Python
# without --break-system-packages. Detect and add it.
if [[ -f /etc/os-release ]]; then
  source /etc/os-release
  if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID%%.*}" -ge 24 ]]; then
    PIP_FLAGS="$PIP_FLAGS --break-system-packages"
  fi
fi

# Detect rocm subversion from /opt if --backend rocm passed without subver.
if [[ "$BACKEND" == "rocm" ]]; then
  if [[ -d /opt/rocm-7.0.0 || -d /opt/rocm-7.0 ]]; then
    # rocm7.0 wheels are nightly-only for now; rocm6.4 wheels are forward-compat
    # via stable HIP ABI. Default to 6.4 stable; user can override with rocm7.0.
    BACKEND="rocm6.4"
  elif [[ -d /opt/rocm-7.2.0 || -d /opt/rocm-7.2 ]]; then
    BACKEND="rocm6.4"  # 6.4 wheels still work; 7.2 has no stable wheels yet
  else
    BACKEND="rocm6.4"
  fi
fi

case "$BACKEND" in
  rocm6.4)
    INDEX="https://download.pytorch.org/whl/rocm6.4"
    EXTRA="rocm"
    PRE=""
    ;;
  rocm7.0)
    INDEX="https://download.pytorch.org/whl/nightly/rocm7.0"
    EXTRA="rocm"
    PRE="--pre"  # rocm7.0 is nightly-only at time of writing
    ;;
  cuda)
    # PyPI's default torch wheel IS the CUDA build, no special index needed.
    INDEX=""
    EXTRA="cuda"
    PRE=""
    ;;
  cpu)
    INDEX="https://download.pytorch.org/whl/cpu"
    EXTRA="cpu"
    PRE=""
    ;;
  *)
    echo "unknown backend: $BACKEND" >&2
    echo "expected: rocm | rocm6.4 | rocm7.0 | cuda | cpu" >&2
    exit 1
    ;;
esac

echo "[install_backend] backend=$BACKEND extra=$EXTRA"
echo "[install_backend] pip flags: $PIP_FLAGS"

# 1. Install torch from the backend-specific index.
echo ""
echo "[install_backend] step 1/2 — installing torch ($BACKEND)"
if [[ -n "$INDEX" ]]; then
  pip install $PIP_FLAGS $PRE torch --index-url "$INDEX"
else
  pip install $PIP_FLAGS torch
fi

# 2. Install the framework + backend-specific deps WITHOUT touching torch.
# --no-deps prevents pip from re-resolving torch and pulling it from PyPI.
echo ""
echo "[install_backend] step 2/2 — installing complexity-framework[$EXTRA]"
pip install $PIP_FLAGS --no-deps -e .
pip install $PIP_FLAGS -e ".[$EXTRA]" --no-deps

# 3. Install the regular non-torch deps (numpy, transformers, etc.). These
# live on PyPI normally so a plain pip install is fine.
pip install $PIP_FLAGS \
  numpy einops transformers tokenizers tiktoken datasets \
  tqdm wandb safetensors pyyaml typer

# 4. Sanity check.
echo ""
echo "[install_backend] verifying torch import + GPU detection..."
python3 - <<PYEOF
import torch
print(f"  torch={torch.__version__}")
print(f"  hip={torch.version.hip}")
print(f"  cuda={torch.version.cuda}")
print(f"  cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device_count={torch.cuda.device_count()}")
    print(f"  device_name={torch.cuda.get_device_name(0)}")
    print(f"  arch_list={torch.cuda.get_arch_list()}")
PYEOF

echo ""
echo "[install_backend] done."
