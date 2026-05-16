"""Polygon ROI and zone dwell-time tracking."""

from __future__ import annotations

import os

import cv2
import numpy as np


def parse_roi_polygon(raw: str) -> np.ndarray:
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) < 6 or len(parts) % 2 != 0:
        raise ValueError(
            f"ROI_POLYGON must be x,y pairs (≥3 vertices); got {len(parts)} values"
        )
    coords = np.asarray([float(v) for v in parts], dtype=np.float32).reshape(-1, 2)
    if np.max(coords) <= 1.5:
        coords[:, 0] *= float(os.environ.get("ROI_NORM_W", "1920"))
        coords[:, 1] *= float(os.environ.get("ROI_NORM_H", "1080"))
    return coords


def parse_ref_size(raw: str):
    if not raw:
        return None
    sep = "x" if "x" in raw.lower() else ","
    w_s, h_s = raw.lower().split(sep, 1) if sep == "x" else raw.split(",", 1)
    return float(w_s.strip()), float(h_s.strip())


def scale_roi_polygon(pts, frame_w, frame_h, ref_size):
    if ref_size is None:
        return pts.copy()
    ref_w, ref_h = ref_size
    if ref_w <= 0 or ref_h <= 0:
        return pts.copy()
    scaled = pts.copy()
    scaled[:, 0] *= float(frame_w) / ref_w
    scaled[:, 1] *= float(frame_h) / ref_h
    return scaled


class RoiZone:
    """Scaled polygon ROI for a fixed frame size."""

    def __init__(self, polygon_raw: str, ref_size: str, point_mode: str = "foot"):
        self._base = parse_roi_polygon(polygon_raw)
        self._ref_size = parse_ref_size(ref_size)
        self.point_mode = point_mode
        self._scaled = None
        self._for_size = None

    def for_frame(self, frame_shape):
        fh, fw = frame_shape[:2]
        key = (int(fw), int(fh))
        if self._for_size != key:
            self._scaled = scale_roi_polygon(self._base, fw, fh, self._ref_size)
            self._for_size = key
        return self._scaled

    def reference_point(self, bbox):
        x1, y1, x2, y2 = bbox[:4]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2) if self.point_mode == "center" else float(y2)
        return cx, cy

    def contains_bbox(self, bbox, roi_pts):
        px, py = self.reference_point(bbox)
        return cv2.pointPolygonTest(roi_pts, (float(px), float(py)), False) >= 0

    def crop_bounds(self, roi_pts, fw, fh, margin_frac: float):
        x1 = max(0, int(np.floor(np.min(roi_pts[:, 0]) - margin_frac * fw)))
        y1 = max(0, int(np.floor(np.min(roi_pts[:, 1]) - margin_frac * fh)))
        x2 = min(fw, int(np.ceil(np.max(roi_pts[:, 0]) + margin_frac * fw)))
        y2 = min(fh, int(np.ceil(np.max(roi_pts[:, 1]) + margin_frac * fh)))
        if x2 <= x1 + 8 or y2 <= y1 + 8:
            return 0, 0, fw, fh
        return x1, y1, x2, y2

    @staticmethod
    def draw(frame, roi_pts):
        pts = np.asarray(roi_pts, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 220, 255), lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 220, 255), 2, cv2.LINE_AA)
        return frame


def sec_to_timestamp(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60.0
    return f"{h:02d}:{m:02d}:{s:06.3f}"


class ZoneDwellTracker:
    """Per global-ID entry/exit/dwell inside the ROI (multiple visits supported)."""

    def __init__(self):
        self._visits: dict[int, list[dict]] = {}
        self._active: dict[int, int] = {}

    def update(self, frame_index, tracks, gid_map, roi_zone: RoiZone, roi_pts, cam_id, fps):
        ts = frame_index / float(fps)
        fi = int(frame_index)
        inside = set()
        for track in tracks:
            gid = gid_map.get((cam_id, int(track[4])))
            if gid is not None and roi_zone.contains_bbox(track[:4], roi_pts):
                inside.add(gid)

        for gid in inside:
            if gid not in self._active:
                visit = {
                    "global_id": int(gid),
                    "visit": len(self._visits.get(gid, [])) + 1,
                    "entry_frame": fi,
                    "entry_time_sec": round(ts, 3),
                    "exit_frame": fi,
                    "exit_time_sec": round(ts, 3),
                }
                self._visits.setdefault(gid, []).append(visit)
                self._active[gid] = len(self._visits[gid]) - 1
            v = self._visits[gid][self._active[gid]]
            v["exit_frame"] = fi
            v["exit_time_sec"] = round(ts, 3)

        for gid in list(self._active):
            if gid not in inside:
                del self._active[gid]

    def rows(self, fps: float) -> list[dict]:
        out = []
        for gid in sorted(self._visits):
            for visit in self._visits[gid]:
                entry_t = float(visit["entry_time_sec"])
                exit_t = float(visit["exit_time_sec"])
                entry_f = int(visit["entry_frame"])
                exit_f = int(visit["exit_frame"])
                out.append(
                    {
                        "global_id": int(visit["global_id"]),
                        "visit": int(visit["visit"]),
                        "entry_frame": entry_f,
                        "entry_time_sec": round(entry_t, 3),
                        "entry_time": sec_to_timestamp(entry_t),
                        "exit_frame": exit_f,
                        "exit_time_sec": round(exit_t, 3),
                        "exit_time": sec_to_timestamp(exit_t),
                        "dwell_time_sec": round(max(0.0, exit_t - entry_t), 3),
                        "dwell_frames": max(0, exit_f - entry_f + 1),
                    }
                )
        return out
