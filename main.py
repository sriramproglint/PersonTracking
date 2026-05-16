#!/usr/bin/env python3
"""
Person tracking with RT-DETR, multi-object tracking, global IDs, and ROI dwell analytics.

Usage:
  python main.py
  REALTIME_MODE=1 VIDEO_SOURCE=video.mp4 python main.py

GPU RT-DETR on DGX Spark (aarch64): build Paddle GPU once, then run:
  bash scripts/build_paddle_gpu_dgx_spark.sh
  bash scripts/install_paddle_gpu_wheel.sh
"""

# Configure cuDNN/CUDA paths before Paddle is imported.
from person_tracking import runtime_env  # noqa: F401

from person_tracking.pipeline import run

if __name__ == "__main__":
    run()
