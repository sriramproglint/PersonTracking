"""Video processing pipeline: RT-DETR + tracking + ROI dwell + Excel."""

from __future__ import annotations

import time

import cv2
import torch

from person_tracking.annotate import FrameAnnotator
from person_tracking.bbox import filter_person_boxes
from person_tracking.config import Config
from person_tracking.detector import RtdetrDetector
from person_tracking.identity import GlobalIdentityRegistry, ReidFeatureCache
from person_tracking.io import write_excel
from person_tracking.paddle_setup import init_paddle
from person_tracking.roi import RoiZone, ZoneDwellTracker
from person_tracking.smoothing import DetectionBBoxSmoother, TemporalBBoxSmoother, smooth_tracks
from person_tracking.tracker import create_tracker_stack
from person_tracking.visibility import FaceBoxExtractor


def _log_config(cfg: Config, paddle_dev: str, cuda: bool, frame_skip: int, source_fps: float, nframes: int):
    print(f"[INFO] RT-DETR | Paddle: {paddle_dev} (cuda={cuda}, fp16={cfg.paddle_fp16})")
    print(f"[INFO] Tracker: {cfg.tracker_backend} | Re-ID: {cfg.reid_device}")
    print(f"[INFO] Video: {cfg.video_source} | every {frame_skip} frame(s) (~{cfg.target_fps:.1f} fps)")
    if nframes > 0:
        print(f"[INFO] Frames: {nframes} @ {source_fps:.2f} fps (~{nframes / source_fps / 60:.1f} min)")
    print(f"[INFO] ROI: {cfg.roi_polygon[:60]}... | Excel: {cfg.excel_path}")
    print(
        f"[INFO] GID stability: reentry≥{cfg.global_reid_reentry_thresh}, "
        f"new_gid<{cfg.global_reid_new_gid_thresh}, retire={cfg.global_gid_retire_frames}f, "
        f"lost_buf={cfg.global_lost_track_frames}f, reid_every={cfg.reid_every_n}"
    )


def run(cfg: Config | None = None) -> None:
    if cfg is None:
        cfg = Config.from_env()

    if not cfg.video_source.is_file():
        raise FileNotFoundError(f"Video not found: {cfg.video_source}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    paddle_dev, paddle_cuda = init_paddle(cfg.paddle_device, cfg.require_paddle_gpu)
    detector = RtdetrDetector(cfg, paddle_cuda)
    tracker, reid_model = create_tracker_stack(cfg)

    roi = RoiZone(cfg.roi_polygon, cfg.roi_ref_size, cfg.roi_point_mode)
    zone = ZoneDwellTracker()
    registry = GlobalIdentityRegistry(
        reentry_thresh=cfg.global_reid_reentry_thresh,
        new_gid_thresh=cfg.global_reid_new_gid_thresh,
        ema_alpha=cfg.global_reid_ema_alpha,
        retire_frames=cfg.global_gid_retire_frames,
        lost_track_frames=cfg.global_lost_track_frames,
        spatial_iou_thresh=cfg.global_spatial_iou_thresh,
        match_margin=cfg.global_match_margin,
    )
    reid_cache = ReidFeatureCache(reid_model, cfg.reid_every_n)
    det_smoother = DetectionBBoxSmoother(cfg.box_smooth_alpha, cfg.box_interp_max_gap, cfg.box_match_iou)
    trk_smoother = TemporalBBoxSmoother(cfg.box_smooth_alpha, cfg.box_interp_max_gap)
    face = FaceBoxExtractor() if cfg.draw_face_overlay else None
    annotator = FrameAnnotator(cfg, roi, face)

    cap = cv2.VideoCapture(str(cfg.video_source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video_source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if cfg.target_fps <= 0:
        cfg.target_fps = float(source_fps)
    frame_skip = cfg.process_every_n if cfg.process_every_n > 0 else max(
        1, int(round(source_fps / cfg.target_fps))
    )
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    _log_config(cfg, paddle_dev, paddle_cuda, frame_skip, source_fps, nframes)

    gid_lifecycle: dict[int, dict] = {}
    writer = None
    frame_counter = 0
    frame_index = 0
    t0 = time.perf_counter()

    try:
        while True:
            for _ in range(frame_skip - 1):
                if not cap.grab():
                    break
            ret, frame = cap.read()
            if not ret:
                break

            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            frame_index = int(pos) if pos and pos > 0 else frame_index + frame_skip

            detections, roi_pts = detector.detect(frame, roi)
            detections = det_smoother.smooth(cfg.cam_id, detections, frame_index)

            tracks = tracker.update(detections, frame)
            tracks = smooth_tracks(cfg.cam_id, tracks, frame_index, trk_smoother)
            tracks = filter_person_boxes(
                tracks, frame.shape,
                cfg.min_box_h_frac, cfg.min_box_w_frac,
                cfg.min_box_area_frac, cfg.nested_inside_thresh,
            )

            frame_counter += 1
            if frame_counter % 30 == 0:
                elapsed = time.perf_counter() - t0
                pfps = frame_counter / elapsed if elapsed > 0 else 0
                pct = 100.0 * frame_index / nframes if nframes > 0 else 0
                print(
                    f"frame={frame_counter} src={frame_index} "
                    f"proc_fps={pfps:.1f} src_fps~{pfps * frame_skip:.1f} "
                    f"{pct:.1f}% tracks={len(tracks)}"
                )

            obs = reid_cache.observations(tracks, frame, frame_index, cfg.cam_id)
            gid_map = registry.assign(obs)

            for tr in tracks:
                gid = gid_map.get((cfg.cam_id, int(tr[4])))
                if gid is None:
                    continue
                ts = frame_index / source_fps
                fi = int(frame_index)
                if gid not in gid_lifecycle:
                    gid_lifecycle[gid] = {
                        "global_id": int(gid),
                        "id_created_frame": fi,
                        "id_created_time_sec": round(ts, 3),
                        "id_exit_frame": fi,
                        "id_exit_time_sec": round(ts, 3),
                    }
                else:
                    gid_lifecycle[gid]["id_exit_frame"] = fi
                    gid_lifecycle[gid]["id_exit_time_sec"] = round(ts, 3)

            zone.update(frame_index, tracks, gid_map, roi, roi_pts, cfg.cam_id, source_fps)

            if frame_counter % cfg.annotate_every_n == 0:
                annotator.draw(frame, tracks, gid_map, frame_index, roi_pts, cfg.cam_id)
                if cfg.save_annotated_video:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(cfg.video_out_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(cfg.target_fps),
                            (w, h),
                        )
                    writer.write(frame)
                if cfg.show_preview:
                    cv2.imshow("PersonTracking", frame)
                    if cv2.waitKey(max(1, int(1000 / cfg.target_fps))) == 27:
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if cfg.save_excel:
            write_excel(cfg, gid_lifecycle, zone, source_fps)
        if cfg.save_annotated_video and cfg.video_out_path.is_file():
            print(f"[INFO] Video: {cfg.video_out_path}")
