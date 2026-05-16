#!/usr/bin/env bash
# Build PaddlePaddle GPU wheel for NVIDIA DGX Spark (aarch64 + CUDA 13 + GB10).
# RT-DETR in main.py needs this wheel — pip 'paddlepaddle' on aarch64 is CPU-only.
#
# Usage:
#   bash scripts/build_paddle_gpu_dgx_spark.sh          # full build (~40 min)
#   FRESH_BUILD=0 bash scripts/build_paddle_gpu_dgx_spark.sh   # resume ninja only
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/rtdetr_env}"
PADDLE_SRC="${PADDLE_SRC:-$HOME/Paddle}"
PADDLE_BRANCH="${PADDLE_BRANCH:-develop}"
BUILD_DIR="${BUILD_DIR:-$PADDLE_SRC/build}"
CUDA_ARCH_BIN="${CUDA_ARCH_BIN:-12.1}"
JOBS="${JOBS:-$(nproc)}"
FRESH_BUILD="${FRESH_BUILD:-1}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

echo "[1/6] cuDNN for CMake (pip; matches torch 2.11)"
"$PIP_BIN" install "nvidia-cudnn-cu13==9.19.0.56"
SITE="$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')"
CUDNN_ROOT="$SITE/nvidia/cudnn"
export CUDNN_ROOT
echo "CUDNN_ROOT=$CUDNN_ROOT"

echo "[2/6] Keep CPU paddle installed until GPU wheel is ready"
if ! "$PYTHON_BIN" -c "import paddle" 2>/dev/null; then
  "$PIP_BIN" install "paddlepaddle==3.2.2"
fi

echo "[3/6] Paddle source (${PADDLE_BRANCH})"
if [[ ! -d "$PADDLE_SRC/.git" ]]; then
  git clone https://github.com/PaddlePaddle/Paddle.git "$PADDLE_SRC"
fi
cd "$PADDLE_SRC"
git fetch --all
git checkout "$PADDLE_BRANCH"
git submodule update --init --recursive
"$PIP_BIN" install -r "$PADDLE_SRC/python/requirements.txt"

if [[ "$FRESH_BUILD" == "1" ]] || [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
  echo "[4/6] CMake (clean)"
  if pgrep -x ninja >/dev/null 2>&1; then
    echo "ERROR: ninja is still running. Wait for it to finish or: pkill ninja"
    exit 1
  fi
  # Avoid 'rm: Directory not empty' when prior build left busy files
  if [[ -d "$BUILD_DIR" ]]; then
    find "$BUILD_DIR" -mindepth 1 -delete 2>/dev/null || rm -rf "$BUILD_DIR"/* 2>/dev/null || true
  fi
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"
  cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_GPU=ON \
    -DWITH_TESTING=OFF \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
    -DNVCC_ARCH_BIN="${CUDA_ARCH_BIN//./}" \
    -DWITH_ARM=ON \
    -DWITH_AVX=OFF \
    -DWITH_MKL=OFF \
    -DWITH_MKLDNN=OFF \
    -DWITH_NCCL=OFF \
    -DWITH_TENSORRT=OFF \
    -DCMAKE_CUDA_FLAGS="-U__ARM_NEON -DEIGEN_DONT_VECTORIZE=1" \
    -DCUDNN_ROOT="${CUDNN_ROOT}" \
    -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
    2>&1 | tee cmake_output.log
else
  echo "[4/6] CMake skipped (build.ninja exists, FRESH_BUILD=0)"
  cd "$BUILD_DIR"
fi

echo "[5/6] Ninja build (~40+ min) — log: $BUILD_DIR/build_output.log"
ninja -C "$BUILD_DIR" -j"${JOBS}" 2>&1 | tee -a "$BUILD_DIR/build_output.log"

WHEEL="$(ls -1 "$BUILD_DIR"/python/dist/paddlepaddle_gpu-*.whl 2>/dev/null | tail -1)"
if [[ -z "${WHEEL:-}" || ! -f "$WHEEL" ]]; then
  echo "ERROR: No wheel in $BUILD_DIR/python/dist/"
  echo "If ninja failed on flash_attn, edit ~/Paddle/cmake/third_party.cmake"
  echo "  (disable WITH_FLASHATTN_V3) and rerun with FRESH_BUILD=1"
  exit 1
fi

echo "[6/6] Install GPU wheel (run separately if you prefer):"
echo "  bash scripts/install_paddle_gpu_wheel.sh \"$WHEEL\""
