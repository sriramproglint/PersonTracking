#!/usr/bin/env bash
# One-time fix: Paddle looks for /usr/local/cuda/lib64/libcudnn.so on DGX Spark.
# Links pip-installed cuDNN into the CUDA toolkit path (requires sudo).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$ROOT/rtdetr_env}"
source "$VENV/bin/activate"

CUDNN_LIB="$("$VENV/bin/python" -c "import site, pathlib; print(pathlib.Path(site.getsitepackages()[0])/'nvidia/cudnn/lib/libcudnn.so.9')")"
CUDA_LIB64="${CUDA_LIB64:-/usr/local/cuda/lib64}"

if [[ ! -f "$CUDNN_LIB" ]]; then
  echo "Installing nvidia-cudnn-cu13..."
  pip install "nvidia-cudnn-cu13==9.19.0.56"
  CUDNN_LIB="$("$VENV/bin/python" -c "import site, pathlib; print(pathlib.Path(site.getsitepackages()[0])/'nvidia/cudnn/lib/libcudnn.so.9')")"
fi

echo "Linking $CUDNN_LIB -> $CUDA_LIB64/libcudnn.so"
sudo mkdir -p "$CUDA_LIB64"
sudo ln -sf "$CUDNN_LIB" "$CUDA_LIB64/libcudnn.so"

echo "Verify:"
"$VENV/bin/python" -c "
from person_tracking import runtime_env  # noqa: F401
import paddle
paddle.set_device('gpu:0')
x = paddle.randn([1, 3, 8, 8])
y = paddle.nn.Conv2D(3, 4, 3)(x)
print('GPU OK', y.shape)
"
