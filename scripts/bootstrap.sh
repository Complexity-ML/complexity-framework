#!/bin/bash
# bootstrap.sh — Fast setup for GPU VMs (Hyperbolic, DataCrunch, bare metal)
# Detects environment and installs PyTorch + complexity-framework.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/Complexity-ML/complexity-framework/main/scripts/bootstrap.sh | bash
#   # or after git clone:
#   bash scripts/bootstrap.sh
#
# INL / Complexity-ML — 2026

set -eo pipefail

echo "=== Complexity Framework Bootstrap ==="

# Use sudo if not root
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# ── Detect environment ────────────────────────────────────────────────
IS_CONTAINER=false
IS_VM=false
HAS_VENV=false

if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    IS_CONTAINER=true
    echo "[env] Container detected"
elif command -v systemctl &>/dev/null; then
    IS_VM=true
    echo "[env] VM detected"
else
    echo "[env] Unknown environment, treating as VM"
    IS_VM=true
fi

# ── Check GPU + Initialize CUDA driver ───────────────────────────────
# WHY: On fresh VMs, the CUDA driver is not initialized until a process
# explicitly calls it. PyTorch calls cudaGetDeviceCount() at import time,
# before the driver is ready → Error 802 (cudaErrorSystemNotReady).
# Fix: run nvidia-smi with persistence mode ON before any Python/PyTorch
# code runs. Persistence mode keeps the driver loaded between calls.
CUDA_DRIVER=""
if command -v nvidia-smi &>/dev/null; then
    # Step 1: enable persistence mode (prevents Error 802 on next boot too)
    $SUDO nvidia-smi -pm 1 2>/dev/null || nvidia-smi -pm 1 2>/dev/null || true
    # Step 2: create /dev/nvidia* device files if missing (common on fresh VMs)
    # Without these files cudaGetDeviceCount() returns Error 802
    if ! ls /dev/nvidia0 &>/dev/null; then
        echo "[gpu] /dev/nvidia* missing — installing nvidia-modprobe..."
        $SUDO apt-get install -y -qq nvidia-modprobe 2>/dev/null || true
        $SUDO nvidia-modprobe 2>/dev/null || true
        $SUDO nvidia-modprobe -u 2>/dev/null || true
    fi
    # Step 3: force driver initialization by querying GPUs
    nvidia-smi > /dev/null 2>&1 || true
    # Step 4: small sleep to let driver fully settle
    sleep 1

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
    CUDA_DRIVER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" | head -1 || echo "")
    echo "[gpu] ${GPU_COUNT}x ${GPU_NAME}"
    echo "[gpu] CUDA driver: ${CUDA_DRIVER} — persistence mode ON"
else
    echo "[gpu] No GPU detected (nvidia-smi not found)"
    echo "[gpu] Installing without CUDA support"
fi

# ── Find Python ───────────────────────────────────────────────────────
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[error] Python not found. Install python3 first."
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo "[python] $PY_VERSION"

# ── Setup pip (prefer system env, avoid venv on clusters) ────────────
PIP="$PYTHON -m pip"
HAS_VENV=false

# Use existing virtualenv/conda if active
if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_PREFIX" ]; then
    echo "[env] Using active environment: ${VIRTUAL_ENV:-$CONDA_PREFIX}"
    PIP="pip"
else
    # Test if pip works directly (prefer --break-system-packages over venv)
    if ! $PIP install --help &>/dev/null 2>&1; then
        echo "[pip] pip not available, installing..."
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-pip python3-dev 2>/dev/null || true
    fi

    # If externally managed (PEP 668), use --break-system-packages
    # This avoids venv which can break CUDA/NCCL paths on GPU clusters
    if $PIP install --dry-run setuptools 2>&1 | grep -q "externally-managed"; then
        echo "[pip] Externally managed Python — using --break-system-packages"
        PIP_EXTRA="--break-system-packages"
    fi
fi

PIP_INSTALL="$PIP install ${PIP_EXTRA:-}"

# ── Install PyTorch (nightly for best compile performance) ────────────
echo "[torch] Installing PyTorch..."

# Check if torch is already installed and recent enough
TORCH_OK=false
if $PYTHON -c "import torch; v=torch.__version__; print(v)" 2>/dev/null; then
    TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
    # Accept 2.6.x stable (pinned for FSDP1/2 compatibility with complexity-framework)
    if echo "$TORCH_VER" | grep -qE "^2\.[6-9]\.|^2\.1[0-9]\."; then
        echo "[torch] PyTorch $TORCH_VER already installed (OK)"
        TORCH_OK=true
    else
        echo "[torch] PyTorch $TORCH_VER not compatible, reinstalling 2.6.0..."
    fi
fi

if [ "$TORCH_OK" = false ]; then
    # Pin to 2.6.0 stable — compatible with complexity-framework FSDP checkpoints
    # CUDA 12.x / 13.x → cu124 (driver forward-compatible), else CPU
    CUDA_MAJOR=$(echo "${CUDA_DRIVER:-0}" | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 12 ] 2>/dev/null; then
        TORCH_CUDA="cu124"
    else
        TORCH_CUDA=""
    fi

    if [ -n "$TORCH_CUDA" ]; then
        echo "[torch] Installing PyTorch 2.6.0 (${TORCH_CUDA})..."
        $PIP_INSTALL torch==2.6.0 --index-url https://download.pytorch.org/whl/${TORCH_CUDA} || \
        $PIP_INSTALL torch==2.6.0
    else
        echo "[torch] Installing PyTorch 2.6.0 (CPU)..."
        $PIP_INSTALL torch==2.6.0
    fi

    TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
    echo "[torch] Installed PyTorch $TORCH_VER"
fi

# ── Clone and install complexity-framework ────────────────────────────
REPO_DIR=""
if [ -f "pyproject.toml" ] && grep -q "complexity-framework" pyproject.toml 2>/dev/null; then
    REPO_DIR="."
    echo "[repo] Already in complexity-framework directory"
elif [ -d "complexity-framework" ]; then
    REPO_DIR="complexity-framework"
    echo "[repo] Found existing clone"
else
    echo "[repo] Cloning complexity-framework..."
    git clone https://github.com/Complexity-ML/complexity-framework.git
    REPO_DIR="complexity-framework"
fi

cd "$REPO_DIR"
$PIP_INSTALL -e . -q
echo "[repo] complexity-framework installed"

# ── Verify ────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
$PYTHON -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}:    {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB)')
print(f'  Compile:  {hasattr(torch, \"compile\")}')
"

# ── Print launch command ──────────────────────────────────────────────
NPROC=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

echo ""
echo "=== Ready to train ==="
echo "  cd $(pwd)"
if [ "$NPROC" -gt 1 ]; then
    echo ""
    echo "  # Single node"
    echo "  torchrun --nproc_per_node=$NPROC scripts/train_400m_v1.py"
    echo ""
    echo "  # Multi-node"
    echo "  torchrun --nnodes=N --nproc_per_node=$NPROC --master_addr=<ip> --master_port=29500 --node_rank=<0..N-1> scripts/train_400m_v1.py"
else
    echo "  python scripts/train_400m_v1.py"
fi
