#!/usr/bin/env bash
# =============================================================================
# setup_env.sh
# Image: pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
# torch 2.9.1 + CUDA 12.8 da co san trong image — script chi cai PyG + deps
#
# Cach chay (khong can chmod):
#   bash rs/lightgcn_pyg/scripts/setup_env.sh
# =============================================================================

set -euo pipefail

# ─── Find repo root ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "[setup] Repo root: ${ROOT_DIR}"

# ─── [1/5] System packages ────────────────────────────────────────────────────
echo ""
echo "━━━ [1/5] System packages ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
apt-get update -qq
apt-get install -y git tmux curl gcc python3-dev cmake libomp-dev

# ─── [2/5] uv ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━ [2/5] uv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" || true
    echo "[setup] uv installed: $(uv --version)"
else
    echo "[setup] uv ok: $(uv --version)"
fi

# ─── [3/5] Verify torch (phai co san trong image) ─────────────────────────────
echo ""
echo "━━━ [3/5] Verify torch ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Tim python3 co torch (thu tung noi)
PYTHON_BIN=""
for py in python3 /usr/bin/python3 /opt/conda/bin/python3; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import torch" 2>/dev/null; then
        PYTHON_BIN="$py"
        break
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo ""
    echo "ERROR: Khong tim thay torch." >&2
    echo "  -> Dung dung image: pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime" >&2
    echo "  -> Tren RunPod: chon template 'PyTorch' hoac nhap tay image name" >&2
    exit 1
fi

TORCH_VER=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)")
CUDA_OK=$(${PYTHON_BIN} -c "import torch; print(torch.cuda.is_available())")
echo "[setup] Python : ${PYTHON_BIN}"
echo "[setup] torch  : ${TORCH_VER}"
echo "[setup] CUDA   : ${CUDA_OK}"

# Xac dinh CUDA tag tu torch version string (vd "2.9.1+cu128" -> "cu128")
CUDA_TAG=$(${PYTHON_BIN} -c "
import torch
v = torch.__version__
tag = v.split('+')[1] if '+' in v else 'cu128'
print(tag)
")
TORCH_BASE=$(${PYTHON_BIN} -c "import torch; print(torch.__version__.split('+')[0])")
echo "[setup] CUDA tag: ${CUDA_TAG}"

# ─── [4/5] torch-geometric + torch-sparse ─────────────────────────────────────
echo ""
echo "━━━ [4/5] torch-geometric + torch-sparse ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
PIP=(uv pip install --python "${PYTHON_BIN}")

TORCH_WHL_URL="${TORCH_WHL_URL:-https://data.pyg.org/whl/torch-${TORCH_BASE}+${CUDA_TAG}.html}"
echo "[setup] PyG wheel index: ${TORCH_WHL_URL}"

echo "[setup] Installing torch-geometric..."
"${PIP[@]}" torch-geometric

echo "[setup] Installing torch-sparse, torch-scatter, pyg_lib..."
"${PIP[@]}" torch-sparse torch-scatter pyg_lib \
    -f "${TORCH_WHL_URL}" \
    || echo "[setup] WARN: companion wheels failed — torch-geometric van hoat dong, chi mat toc do." >&2

# ─── [5/5] Project dependencies ───────────────────────────────────────────────
echo ""
echo "━━━ [5/5] Project dependencies ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"${PIP[@]}" \
    "pandas>=2.0.0" \
    "numpy<2.0.0" \
    scipy \
    scikit-learn \
    "transformers>=4.36.0" \
    pillow matplotlib seaborn tqdm \
    tensorboard ipykernel \
    wandb "huggingface_hub>=0.20.0"

# ─── Verify all imports ────────────────────────────────────────────────────────
echo ""
echo "━━━ Verify imports ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"${PYTHON_BIN}" - <<'PY'
checks = [
    ("torch",           lambda: __import__("torch").__version__ + " | CUDA=" + str(__import__("torch").cuda.is_available())),
    ("torch_geometric", lambda: __import__("torch_geometric").__version__),
    ("torch_sparse",    lambda: "ok"),
    ("transformers",    lambda: __import__("transformers").__version__),
    ("wandb",           lambda: __import__("wandb").__version__),
    ("tensorboard",     lambda: "ok"),
]
for name, fn in checks:
    try:
        print(f"  ✓ {name}: {fn()}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
PY

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done!"
echo ""
echo "  Neu muon cai them package sau nay (uv nhanh hon pip):"
echo "    uv pip install --python ${PYTHON_BIN} <ten-package>"
echo ""
echo "  Chay training:"
echo "    cd rs/lightgcn_pyg"
echo "    bash scripts/test/test_lightgcn.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
