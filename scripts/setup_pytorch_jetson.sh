#!/usr/bin/env bash
# One-time setup: PyTorch + torchvision with real GPU acceleration, in
# ml/venv, for training YOLO models on a Jetson AGX Orin (JetPack 5.1.2 /
# L4T R35.4.1) -- run directly on the Jetson, NOT in the Docker container.
#
# Why this is its own script rather than just `pip install -r
# requirements.txt`: standard PyPI `torch`/`torchvision` wheels don't
# support Tegra GPUs at all. NVIDIA publishes a `torch` wheel matched to
# each JetPack version, but does NOT publish `torchvision` wheels for
# JetPack 5.x (only 6.x) -- it has to be built from source, matched
# exactly to that torch build's ABI.
#
# Run from the repo root, on the Jetson itself:
#   ./scripts/setup_pytorch_jetson.sh
#
# Only needed on a machine you intend to *train* on. Running the finished
# perception pipeline (an already-exported .onnx model) needs none of
# this -- inference goes through a separate runtime (ONNX Runtime /
# TensorRT) with no PyTorch dependency at all.
#
# Everything installs into ml/venv (git-ignored, same convention as the
# Mac's own ml/venv) -- safe to `rm -rf ml/venv ~/ml_src` and rerun from
# scratch at any point.
#
# Known gotchas baked into this script:
#   - The NVIDIA torch wheel for this JetPack is built for Python 3.8
#     specifically (cp38 ABI tag) -- not whatever `python3` defaults to.
#   - NVIDIA's own docs say `numpy==1.26.1`, but that requires Python
#     >=3.9 and doesn't exist for cp38. The latest numpy that still
#     supports 3.8 (1.24.4) is used instead.
#   - Building torchvision needs setuptools==69.5.1 specifically (both
#     newer and the venv's ancient 44.0.0 default have been reported to
#     fail this build).
#   - FORCE_CUDA=1 must be set explicitly, or torchvision silently builds
#     CPU-only ops (defeats the whole point of doing this on the Jetson).
#     TORCH_CUDA_ARCH_LIST=8.7 targets Orin's specific compute capability.
#   - `pip install ultralytics` pulls in its own torch/torchvision by
#     default, which would silently replace the Jetson-specific builds
#     with incompatible generic ones -- installed with --no-deps and its
#     other dependencies added explicitly instead.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$REPO_ROOT/ml"
SRC="$HOME/ml_src"
TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
TORCHVISION_TAG="v0.16.1"

echo "Venv: $ML_DIR/venv"
echo "Torchvision source: $SRC/torchvision"
echo

# --- Step 0: apt prerequisites ----------------------------------------------
echo "==> Installing apt dependencies"
sudo apt-get install -y --no-install-recommends \
  python3-pip python3.8-venv libopenblas-dev libjpeg-dev zlib1g-dev \
  libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

# --- Step 1: venv (Python 3.8, locked by the NVIDIA wheel's ABI tag) -------
if [ ! -d "$ML_DIR/venv" ]; then
  echo "==> Creating ml/venv (Python 3.8)"
  python3.8 -m venv "$ML_DIR/venv"
fi
source "$ML_DIR/venv/bin/activate"
pip install --upgrade pip
pip install "numpy==1.24.4"   # NVIDIA docs say 1.26.1, but that needs Python >=3.9

# --- Step 2: NVIDIA's PyTorch wheel for this exact JetPack version ---------
echo "==> Installing NVIDIA PyTorch wheel"
pip install --no-cache "$TORCH_WHEEL_URL"

# --- Step 3: torchvision, built from source (no prebuilt wheel for JP5.x) --
mkdir -p "$SRC"
if [ ! -d "$SRC/torchvision" ]; then
  echo "==> Cloning torchvision $TORCHVISION_TAG"
  git clone --branch "$TORCHVISION_TAG" --depth 1 https://github.com/pytorch/vision.git "$SRC/torchvision"
fi
pip install "setuptools==69.5.1" wheel
echo "==> Building torchvision (slow -- real CUDA compile, ~15-20min on Orin)"
(
  cd "$SRC/torchvision"
  FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.7" python setup.py bdist_wheel
)
echo "==> Installing torchvision"
pip install "$(find "$SRC/torchvision/dist" -name '*.whl' | sort -V | tail -1)"

# --- Step 4: the rest of ml/requirements.txt, minus torch/torchvision -----
# (installed separately above -- letting pip touch them here would silently
# replace the Jetson-specific builds with incompatible generic ones)
echo "==> Installing remaining ml/requirements.txt packages"
pip install "opencv-python==4.9.0.80" onnx onnxruntime matplotlib remotezip

echo "==> Installing ultralytics (--no-deps, to avoid it pulling generic torch/torchvision)"
pip install --no-deps ultralytics
pip install nvidia-ml-py polars psutil pyyaml ultralytics-thop

# --- Verify -------------------------------------------------------------
echo
echo "==> Verifying"
python -c "
import torch, torchvision
from ultralytics import YOLO
assert torch.cuda.is_available(), 'CUDA not available -- something is wrong'
print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}')
print(f'torchvision {torchvision.__version__}')
print('ultralytics imported OK')
"

echo
echo "Done. Activate with: source $ML_DIR/venv/bin/activate"
