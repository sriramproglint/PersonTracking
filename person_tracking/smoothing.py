"""Temporal bounding-box smoothing for detections and tracks."""

from __future__ import annotations

import numpy as np

from person_tracking.bbox import bbox_iou_xyxy


class TemporalBBoxSmoother:
    def __init__(self, alpha=0.35, max_interp_gap=2, stale_frames=30):
        self.alpha = float(alpha)
        self.max_interp_gap = int(max_interp_gap)
        self.stale_frames = int(stale_frames)
        self._state = {}

    def _ema(self, current, previous):
        return self.alpha * current + (1.0 - self.alpha) * previous

    def update(self, key, box, frame_idx):
        box = np.asarray(box, dtype=np.float32).ravel()[:4]
        st = self._state.get(key)
        if st is None:
            smooth = box.copy()
        else:
            gap = int(frame_idx) - int(st["frame_idx"])
            if 1 < gap <= self.max_interp_gap:
                interp = st["smooth"] + (box - st["smooth"]) / float(gap)
                smooth = self._ema(interp, st["smooth"])
            else:
                smooth = self._ema(box, st["smooth"])
        self._state[key] = {"smooth": smooth.astype(np.float32), "frame_idx": int(frame_idx)}
        return smooth

    def get_smoothed(self, key):
        st = self._state.get(key)
        return None if st is None else st["smooth"]

    def prune(self, frame_idx, active_keys=None):
        active_keys = active_keys or set()
        for key, st in list(self._state.items()):
            if key not in active_keys and int(frame_idx) - int(st["frame_idx"]) > self.stale_frames:
                del self._state[key]


class DetectionBBoxSmoother:
    def __init__(self, alpha=0.35, max_interp_gap=2, match_iou=0.3):
        self.match_iou = float(match_iou)
        self._temporal = TemporalBBoxSmoother(alpha, max_interp_gap)
        self._next_slot = 0
        self._last_slots = []

    def smooth(self, cam_id, detections, frame_idx):
        if not detections:
            self._last_slots = []
            self._temporal.prune(frame_idx)
            return detections

        prev_boxes = {
            s: self._temporal.get_smoothed((cam_id, s))
            for s in self._last_slots
        }
        prev_boxes = {s: b for s, b in prev_boxes.items() if b is not None}
        slot_by_det = [-1] * len(detections)
        used = set()
        pairs = []
        for di, det in enumerate(detections):
            for slot, prev in prev_boxes.items():
                if slot in used:
                    continue
                iou = bbox_iou_xyxy(det[:4], prev)
                if iou >= self.match_iou:
                    pairs.append((iou, di, slot))
        pairs.sort(key=lambda x: -x[0])
        for _, di, slot in pairs:
            if slot_by_det[di] < 0 and slot not in used:
                slot_by_det[di] = slot
                used.add(slot)

        out, active_keys, new_slots = [], set(), []
        for di, det in enumerate(detections):
            slot = slot_by_det[di]
            if slot < 0:
                slot = self._next_slot
                self._next_slot += 1
            key = (cam_id, slot)
            smooth = self._temporal.update(key, det[:4], frame_idx)
            row = list(det)
            row[:4] = smooth.tolist()
            out.append(row)
            active_keys.add(key)
            new_slots.append(slot)
        self._last_slots = new_slots
        self._temporal.prune(frame_idx, active_keys)
        return out


def smooth_tracks(cam_id, tracks, frame_idx, smoother: TemporalBBoxSmoother):
    if not tracks:
        smoother.prune(frame_idx)
        return tracks
    out, active = [], set()
    for tr in tracks:
        tid = int(tr[4])
        key = (cam_id, tid)
        smooth = smoother.update(key, tr[:4], frame_idx)
        row = list(tr)
        row[:4] = smooth.tolist()
        out.append(row)
        active.add(key)
    smoother.prune(frame_idx, active)
    return out
