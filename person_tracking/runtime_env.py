"""Configure CUDA/cuDNN library paths before Paddle loads (DGX Spark / pip wheels)."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_SHIM_NAMES = (
    ("libcudnn.so", "libcudnn.so.9"),
    ("libcublas.so", "libcublas.so.13"),
    ("libcublasLt.so", "libcublasLt.so.13"),
)


def _make_shim_dir(lib_dir: Path, shim_dir: Path) -> bool:
    """Symlink versioned .so names to unversioned names Paddle/dlopen expect."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    made = False
    for link_name, target_glob in _SHIM_NAMES:
        link = shim_dir / link_name
        if link.exists():
            made = True
            continue
        matches = sorted(lib_dir.glob(target_glob))
        if not matches:
            continue
        link.symlink_to(matches[0].resolve())
        made = True
    return made


def configure_cuda_libs() -> list[str]:
    """Prepend NVIDIA pip libs + shims to LD_LIBRARY_PATH. Call before ``import paddle``."""
    sp = Path(site.getsitepackages()[0])
    lib_dirs = [
        sp / "nvidia/cudnn/lib",
        sp / "nvidia/cublas/lib",
        sp / "nvidia/cuda_runtime/lib",
        sp / "nvidia/cuda_nvrtc/lib",
        sp / "nvidia/nvjitlink/lib",
    ]
    paths: list[Path] = [p.resolve() for p in lib_dirs if p.is_dir()]
    if not paths:
        return []

    root = Path(__file__).resolve().parent.parent
    cudnn_lib = sp / "nvidia/cudnn/lib"

    # Paddle looks for $CUDA_HOME/lib64/libcudnn.so (not only LD_LIBRARY_PATH).
    cuda_home_lib64 = root / "lib64"
    if cudnn_lib.is_dir():
        _make_shim_dir(cudnn_lib, cuda_home_lib64)
    if cuda_home_lib64.is_dir() and any(cuda_home_lib64.iterdir()):
        os.environ["CUDA_HOME"] = str(root)
        paths.insert(0, cuda_home_lib64.resolve())

    shim_dir = root / ".cuda_shim"
    if cudnn_lib.is_dir():
        _make_shim_dir(cudnn_lib, shim_dir)
    if shim_dir.is_dir() and any(shim_dir.iterdir()):
        paths.insert(0, shim_dir.resolve())

    path_strs = [str(p) for p in paths]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(path_strs + ([cur] if cur else []))

    cudnn_so = cudnn_lib / "libcudnn.so.9"
    if cudnn_so.is_file():
        preload = str(cudnn_so.resolve())
        old = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = f"{preload}:{old}" if old else preload

    return path_strs


configure_cuda_libs()
