"""Runtime configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no")


def _apply_stable_id_defaults() -> None:
    """Conservative settings: fewer GID swaps and fewer new IDs."""
    if not _env_bool("STABLE_IDS", "1"):
        return
    for key, val in {
        "TRACKER_BACKEND": "strongsort",
        "STRONGSORT_MAX_AGE": "120",
        "REID_EVERY_N": "1",
        "GLOBAL_REID_REENTRY_THRESH": "0.58",
        "GLOBAL_REID_NEW_GID_THRESH": "0.48",
        "GLOBAL_GID_RETIRE_FRAMES": "450",
        "GLOBAL_REID_EMA_ALPHA": "0.12",
        "GLOBAL_LOST_TRACK_FRAMES": "90",
        "GLOBAL_SPATIAL_IOU_THRESH": "0.12",
        "GLOBAL_MATCH_MARGIN": "0.06",
        "BOX_SMOOTH_ALPHA": "0.45",
        "STRONGSORT_MAX_COS_DIST": "0.35",
    }.items():
        os.environ.setdefault(key, val)


def _apply_realtime_defaults() -> None:
    if not _env_bool("REALTIME_MODE"):
        return
    defaults = {
        "SHOW_PREVIEW": "0",
        "DRAW_FACE_OVERLAY": "0",
        "SAVE_ANNOTATED_VIDEO": "0",
        "PROCESS_EVERY_N": "1",
        "TRACKER_BACKEND": "bytetrack",
        "REID_EVERY_N": "2",
        "ROI_DETECT_CROP": "1",
        "IMGSZ": "512",
        "PADDLE_FP16": "1",
        "REID_HALF": "1",
        "TARGET_FPS": "0",
        "REQUIRE_PADDLE_GPU": "1",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)


_apply_realtime_defaults()

BASE_DIR = Path(__file__).resolve().parent.parent
PADDLEDET_ROOT = BASE_DIR / "PaddleDetection"
BOXMOT_ROOT = PADDLEDET_ROOT / "Yolov5_StrongSORT_OSNet"


@dataclass
class Config:
    """All tunables for a single tracking run."""

    # Paths
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    video_source: Path | None = None
    output_dir: Path | None = None

    # Detector (RT-DETR)
    conf_thres: float = 0.3
    iou_thres: float = 0.5  # NMS
    imgsz: int = 640
    paddle_device: str = "gpu:0"
    paddle_fp16: bool = False
    require_paddle_gpu: bool = True

    # Video / performance
    target_fps: float = 15.0
    process_every_n: int = 0
    annotate_every_n: int = 1
    roi_detect_crop: bool = False
    roi_crop_margin: float = 0.08

    # Tracker
    tracker_backend: str = "strongsort"
    strongsort_max_age: int = 120
    strongsort_max_cos_dist: float = 0.35
    reid_every_n: int = 1
    reid_half: bool = False
    reid_device: str = "cuda:0"
    reid_weights: str | None = None
    global_reid_reentry_thresh: float = 0.58
    global_reid_new_gid_thresh: float = 0.48
    global_reid_ema_alpha: float = 0.12
    global_gid_retire_frames: int = 450
    global_lost_track_frames: int = 90
    global_spatial_iou_thresh: float = 0.12
    global_match_margin: float = 0.06

    # Box filters / smoothing
    min_box_h_frac: float = 0.08
    min_box_w_frac: float = 0.04
    min_box_area_frac: float = 0.004
    nested_inside_thresh: float = 0.7
    box_smooth_alpha: float = 0.35
    box_interp_max_gap: int = 2
    box_match_iou: float = 0.7

    # ROI polygon (image-map coordinates)
    roi_polygon: str = "318,300,457,66,1617,68,1573,387"
    roi_ref_size: str = "1920x1080"
    roi_point_mode: str = "foot"
    draw_roi: bool = True

    # Output
    save_excel: bool = True
    save_annotated_video: bool = True
    show_preview: bool = True
    draw_face_overlay: bool = True
    face_overlay_interval: int = 3

    # Optional slow Haar visibility filter
    enable_visibility_filter: bool = False

    cam_id: str = "cam"

    @classmethod
    def from_env(cls) -> Config:
        base = BASE_DIR
        video = Path(
            os.environ.get("VIDEO_SOURCE", str(base / "div_6.mp4"))
        ).expanduser().resolve()
        out = Path(os.environ.get("OUTPUT_DIR", str(base / "output"))).resolve()
        return cls(
            base_dir=base,
            video_source=video,
            output_dir=out,
            conf_thres=float(os.environ.get("CONF_THRES", "0.3")),
            imgsz=int(os.environ.get("IMGSZ", "640")),
            paddle_device=os.environ.get("PADDLE_DEVICE", "gpu:0"),
            paddle_fp16=_env_bool("PADDLE_FP16"),
            require_paddle_gpu=_env_bool("REQUIRE_PADDLE_GPU", "0"),
            target_fps=float(os.environ.get("TARGET_FPS", "15")),
            process_every_n=int(os.environ.get("PROCESS_EVERY_N", "0")),
            annotate_every_n=max(1, int(os.environ.get("ANNOTATE_EVERY_N", "1"))),
            roi_detect_crop=_env_bool("ROI_DETECT_CROP"),
            roi_crop_margin=float(os.environ.get("ROI_CROP_MARGIN", "0.08")),
            tracker_backend=os.environ.get("TRACKER_BACKEND", "strongsort").lower(),
            strongsort_max_age=int(os.environ.get("STRONGSORT_MAX_AGE", "120")),
            strongsort_max_cos_dist=float(
                os.environ.get("STRONGSORT_MAX_COS_DIST", "0.35")
            ),
            reid_every_n=max(1, int(os.environ.get("REID_EVERY_N", "1"))),
            reid_half=_env_bool("REID_HALF"),
            reid_device=os.environ.get(
                "STRONGSORT_DEVICE",
                "cuda:0",
            ),
            reid_weights=os.environ.get("STRONGSORT_REID_WEIGHTS") or None,
            global_reid_reentry_thresh=float(
                os.environ.get("GLOBAL_REID_REENTRY_THRESH", "0.58")
            ),
            global_reid_new_gid_thresh=float(
                os.environ.get("GLOBAL_REID_NEW_GID_THRESH", "0.48")
            ),
            global_reid_ema_alpha=float(
                os.environ.get("GLOBAL_REID_EMA_ALPHA", "0.12")
            ),
            global_gid_retire_frames=int(
                os.environ.get("GLOBAL_GID_RETIRE_FRAMES", "450")
            ),
            global_lost_track_frames=int(
                os.environ.get("GLOBAL_LOST_TRACK_FRAMES", "90")
            ),
            global_spatial_iou_thresh=float(
                os.environ.get("GLOBAL_SPATIAL_IOU_THRESH", "0.12")
            ),
            global_match_margin=float(os.environ.get("GLOBAL_MATCH_MARGIN", "0.06")),
            roi_polygon=os.environ.get(
                "ROI_POLYGON",
                "318,300,457,66,1617,68,1573,387",
            ).strip(),
            roi_ref_size=os.environ.get("ROI_REF_SIZE", "1920x1080").strip(),
            roi_point_mode=os.environ.get("ROI_POINT_MODE", "foot").lower(),
            draw_roi=_env_bool("DRAW_ROI", "1"),
            save_excel=_env_bool("SAVE_EXCEL", "1"),
            save_annotated_video=_env_bool("SAVE_ANNOTATED_VIDEO", "1"),
            show_preview=_env_bool("SHOW_PREVIEW", "1"),
            draw_face_overlay=_env_bool("DRAW_FACE_OVERLAY", "1"),
            face_overlay_interval=max(
                1, int(os.environ.get("FACE_OVERLAY_INTERVAL", "3"))
            ),
            enable_visibility_filter=_env_bool("ENABLE_VISIBILITY_FILTER"),
        )

    @property
    def excel_path(self) -> Path:
        assert self.video_source and self.output_dir
        return self.output_dir / f"{self.video_source.stem}_tracking.xlsx"

    @property
    def video_out_path(self) -> Path:
        assert self.video_source and self.output_dir
        return self.output_dir / f"{self.video_source.stem}_annotated.mp4"
