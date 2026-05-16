"""Frame annotation (boxes, labels, ROI)."""

from __future__ import annotations

import cv2

from person_tracking.roi import RoiZone


def gid_color(gid: int) -> tuple[int, int, int]:
    r = (37 * gid) % 200 + 55
    g = (17 * gid * gid) % 200 + 55
    b = (91 * gid) % 200 + 55
    return int(b), int(g), int(r)


class FrameAnnotator:
    def __init__(self, cfg, roi: RoiZone, face_extractor=None):
        self.cfg = cfg
        self.roi = roi
        self._face = face_extractor
        self._face_cache: dict[tuple, tuple] = {}

    def draw(self, frame, tracks, gid_map, frame_index, roi_pts, cam_id):
        active = set()
        for tr in tracks:
            tid = int(tr[4])
            key = (cam_id, tid)
            active.add(key)
            person = tr[:4]
            in_roi = self.roi.contains_bbox(person, roi_pts)

            if self.cfg.draw_face_overlay and self._face:
                cached = self._face_cache.get(key)
                if cached and frame_index - cached[1] < self.cfg.face_overlay_interval:
                    box = cached[0]
                else:
                    box = self._face.extract(frame, person) or person
                    self._face_cache[key] = (box, frame_index)
                x1, y1, x2, y2 = map(int, box)
            else:
                x1, y1, x2, y2 = map(int, person)

            gid = gid_map.get(key)
            color = gid_color(gid) if gid is not None else (0, 255, 0)
            if in_roi:
                color = (0, 255, 255) if gid is not None else (0, 200, 255)
            label = f"GID:{gid}" if gid is not None else f"L{tid}"
            if in_roi and gid is not None:
                label += " [ROI]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if in_roi else 2)
            cv2.putText(
                frame, label, (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        for k in [k for k in self._face_cache if k not in active]:
            del self._face_cache[k]

        if self.cfg.draw_roi:
            RoiZone.draw(frame, roi_pts)
        cv2.putText(
            frame, f"frame {frame_index}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        return frame
