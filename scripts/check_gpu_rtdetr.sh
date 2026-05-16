#!/usr/bin/env bash
# Quick check: is the environment ready for GPU RT-DETR?
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${VENV_DIR:-$ROOT_DIR/rtdetr_env}/bin/activate"

echo "=== Platform ==="
uname -m
python --version

echo "=== PyTorch (StrongSORT Re-ID) ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== Paddle (RT-DETR) ==="
if ! python -c "import paddle" 2>/dev/null; then
  echo "paddle: NOT INSTALLED — pip install paddlepaddle==3.2.2"
  exit 1
fi
python - <<'PY'
import paddle
print("version:", paddle.__version__)
print("compiled_with_cuda:", paddle.device.is_compiled_with_cuda())
try:
    paddle.set_device("gpu:0")
    x = paddle.ones([2, 2])
    print("device:", paddle.get_device(), "tensor ok:", float(x.sum()) == 4.0)
    if paddle.device.is_compiled_with_cuda():
        print("STATUS: GPU RT-DETR ready")
    else:
        print("STATUS: CPU-only paddle — build GPU wheel:")
        print("  bash scripts/build_paddle_gpu_dgx_spark.sh")
        print("  bash scripts/install_paddle_gpu_wheel.sh")
except Exception as e:
    print("GPU test failed:", e)
PY

WHEEL=$(ls -1 "$HOME/Paddle/build/python/dist/paddlepaddle_gpu-"*.whl 2>/dev/null | tail -1)
if [[ -n "${WHEEL:-}" ]]; then
  echo "=== GPU wheel built (not installed yet?) ==="
  echo "$WHEEL"
  echo "Install: bash scripts/install_paddle_gpu_wheel.sh"
fi
