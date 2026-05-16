"""BoxMOT tracker factory."""

from __future__ import annotations

import sys

import numpy as np
import torch
import yaml

from person_tracking.config import BOXMOT_ROOT, Config

REID_TRACKERS = frozenset(
    {"strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"}
)


class CameraTracker:
    def __init__(self, inner):
        self._inner = inner

    def update(self, detections, frame):
        if not detections:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.asarray(detections, dtype=np.float32)
        out = self._inner.update(dets, np.asarray(frame))
        if out is None or len(out) == 0:
            return []
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out.tolist()


def create_tracker_stack(cfg: Config):
    if not BOXMOT_ROOT.is_dir():
        raise RuntimeError(f"BoxMOT not found: {BOXMOT_ROOT}")
    sys.path.insert(0, str(BOXMOT_ROOT))

    from boxmot.reid.core import ReID
    from boxmot.trackers.tracker_zoo import TRACKER_CONFIGS, create_tracker

    device = cfg.reid_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    reid_kw = {"device": device, "half": cfg.reid_half}
    if cfg.reid_weights:
        reid_kw["weights"] = cfg.reid_weights
    reid_model = ReID(**reid_kw).model

    yaml_path = TRACKER_CONFIGS / f"{cfg.tracker_backend}.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        args = {p: d["default"] for p, d in yaml.safe_load(f).items()}
    args["min_conf"] = cfg.conf_thres
    if cfg.tracker_backend in REID_TRACKERS:
        args["max_age"] = cfg.strongsort_max_age
        if "max_cos_dist" in args:
            args["max_cos_dist"] = cfg.strongsort_max_cos_dist

    kw = {
        "tracker_type": cfg.tracker_backend,
        "device": device,
        "half": cfg.reid_half,
        "evolve_param_dict": args,
    }
    if cfg.tracker_backend in REID_TRACKERS:
        kw["reid_model"] = reid_model

    return CameraTracker(create_tracker(**kw)), reid_model
