"""PaddlePaddle device setup for RT-DETR."""

from __future__ import annotations

import sys

import paddle


def gpu_install_help() -> str:
    return (
        "\nRT-DETR GPU setup:\n"
        "  pip install nvidia-cudnn-cu13==9.19.0.56\n"
        "  bash scripts/build_paddle_gpu_dgx_spark.sh\n"
        "  bash scripts/install_paddle_gpu_wheel.sh\n"
        "  bash scripts/check_gpu_rtdetr.sh\n"
    )


def _gpu_runtime_ok() -> bool:
    """True when a small GPU op succeeds (cuDNN + driver loaded)."""
    try:
        paddle.set_device("gpu:0")
        x = paddle.randn([2, 3, 8, 8])
        conv = paddle.nn.Conv2D(3, 4, 3)
        y = conv(x)
        paddle.device.cuda.synchronize()
        return y.shape == [2, 4, 6, 6]
    except Exception as exc:
        print(f"[WARN] Paddle GPU runtime check failed: {exc}")
        return False


def init_paddle(device: str, require_gpu: bool) -> tuple[str, bool]:
    """Configure Paddle device; return (device_name, cuda_compiled)."""
    cuda = paddle.device.is_compiled_with_cuda()
    dev = device.strip().lower()
    want_gpu = cuda and dev.startswith("gpu")

    if want_gpu and _gpu_runtime_ok():
        paddle.set_device(dev if ":" in dev else "gpu:0")
        return paddle.get_device(), True

    if want_gpu:
        print(
            "[WARN] Paddle GPU runtime unavailable (cuDNN). Using CPU for RT-DETR.\n"
            "  For GPU: pip install nvidia-cudnn-cu13==9.19.0.56\n"
            "           bash scripts/link_cudnn_for_paddle.sh   # once, needs sudo\n"
        )

    paddle.set_device("cpu")
    if not cuda:
        print(
            "[WARN] Paddle is CPU-only (~0.5 fps). For GPU RT-DETR on DGX Spark:\n"
            "  bash scripts/build_paddle_gpu_dgx_spark.sh\n"
            "  bash scripts/install_paddle_gpu_wheel.sh\n"
        )
        if require_gpu:
            print(gpu_install_help())
            sys.exit(1)
    if require_gpu and cuda:
        print(
            "[WARN] REQUIRE_PADDLE_GPU=1 but running on CPU. "
            "Install cuDNN: pip install nvidia-cudnn-cu13==9.19.0.56\n"
        )
        print(gpu_install_help())
        sys.exit(1)
    return paddle.get_device(), cuda and str(paddle.get_device()).startswith("gpu")
