#!/usr/bin/env bash
# Install built paddlepaddle_gpu wheel — enables RT-DETR on GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/rtdetr_env}"
BUILD_DIR="${BUILD_DIR:-$HOME/Paddle/build}"

WHEEL="${1:-}"
if [[ -z "$WHEEL" ]]; then
  WHEEL="$(ls -1 "$BUILD_DIR"/python/dist/paddlepaddle_gpu-*.whl 2>/dev/null | tail -1)"
fi
if [[ -z "${WHEEL:-}" || ! -f "$WHEEL" ]]; then
  echo "ERROR: No paddlepaddle_gpu wheel found."
  echo "Build first (one time, ~40 min):"
  echo "  bash scripts/build_paddle_gpu_dgx_spark.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"
pip uninstall -y paddlepaddle paddlepaddle-gpu 2>/dev/null || true
pip install "$WHEEL"

python - <<'PY'
import paddle
assert paddle.device.is_compiled_with_cuda(), "Wheel is not CUDA-enabled"
paddle.set_device("gpu:0")
paddle.utils.run_check()
print("OK: Paddle GPU ready for RT-DETR")
print("paddle:", paddle.__version__, "device:", paddle.get_device())
PY

echo ""
echo "Run RT-DETR tracking on GPU:"
echo "  cd $ROOT_DIR"
echo "  REALTIME_MODE=1 PADDLE_DEVICE=gpu:0 python main.py"
