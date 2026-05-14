#!/usr/bin/env bash
# Setup environment for running CombiGCN on a remote GPU pod or Docker image.
# - Installs system packages needed to compile torch extensions
# - Installs `uv` (astral) if missing
# - Installs project extras and PyG / torch-sparse (wheel URL configurable)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# Use sudo if not running as root
if [ "${EUID}" -ne 0 ]; then
  SUDO='sudo'
else
  SUDO=''
fi

echo "[setup_env] Root dir: ${ROOT_DIR}"

echo "[setup_env] Installing system packages (build-essential, cmake, libomp-dev)..."
${SUDO} apt update
${SUDO} apt install -y git tmux curl build-essential python3-dev cmake libomp-dev || {
  echo "[setup_env] apt install failed — please run the commands manually or check your environment." >&2
  exit 1
}

# Install uv (astral) into $HOME if missing
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup_env] Installing 'uv' (astral)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" || true
fi

echo "[setup_env] Installing Python project extras into system Python using 'uv'..."
cd "${ROOT_DIR}"

# Install project extras (docker extra) into image/system Python
uv pip install --system -e ".[docker]"

# Install torch-geometric and optionally PyG companion wheels if TORCH_WHL_URL is provided.
# NOTE: no hardcoded default URL is used — set TORCH_WHL_URL in your environment when needed.
DEFAULT_TORCH_WHL_URL="https://data.pyg.org/whl/torch-2.9.0+cu128.html"
TORCH_WHL_URL="${TORCH_WHL_URL:-$DEFAULT_TORCH_WHL_URL}"

echo "[setup_env] Installing torch-geometric (system pip)"
uv pip install --system torch-geometric || echo "[setup_env] torch-geometric install failed — you may need to install a specific wheel for your torch+cuda." >&2

if [ -n "${TORCH_WHL_URL}" ]; then
  echo "[setup_env] Installing PyG companion wheels from: ${TORCH_WHL_URL}"
  # install common companion wheels that often require compiled binaries
  uv pip install --system pyg_lib torch_scatter torch_sparse -f "${TORCH_WHL_URL}" || echo "[setup_env] PyG companion wheels install failed — check URL or torch version." >&2
else
  echo "[setup_env] TORCH_WHL_URL is not set — skipping install of compiled PyG companion wheels (pyg_lib, torch_scatter, torch_sparse)."
  echo "[setup_env] If you need these wheels, set TORCH_WHL_URL to the appropriate index (e.g. https://data.pyg.org/whl/torch-<version>+cu<xx>.html) and re-run this script."
fi

echo "[setup_env] Installation finished. Verifying imports..."
python3 - <<'PY'
try:
    import torch
    print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
except Exception as e:
    print('torch import failed:', e)
try:
    import torch_geometric
    print('torch_geometric ok')
except Exception as e:
    print('torch_geometric import failed:', e)
try:
    import torch_sparse
    print('torch_sparse ok')
except Exception as e:
    print('torch_sparse import failed:', e)
PY

echo "[setup_env] Done. If you're using a different torch+cuda version, set the environment variable TORCH_WHL_URL to the appropriate PyG wheel index before running this script."

echo "[setup_env] Make sure the script is executable: chmod +x scripts/setup_env.sh"
