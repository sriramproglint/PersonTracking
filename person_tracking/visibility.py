"""Optional Haar-based person visibility filter."""

from __future__ import annotations

import cv2
import numpy as np


def box_size_ok(frame, bbox) -> bool:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w - 1, round(x2))))
    y2 = int(max(0, min(h - 1, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return False
    return (y2 - y1) >= 28 and (x2 - x1) >= 16


def clamp_box(bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w - 1, round(x2))))
    y2 = int(max(0, min(h - 1, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class PersonVisibilityFilter:
    def __init__(self, min_aspect=0.45, max_aspect=4.5):
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        base = cv2.data.haarcascades
        self._cascades = []
        for name in (
            "haarcascade_frontalface_default.xml",
            "haarcascade_profileface.xml",
            "haarcascade_upperbody.xml",
            "haarcascade_fullbody.xml",
        ):
            clf = cv2.CascadeClassifier(base + name)
            if clf.empty():
                raise RuntimeError(f"Missing cascade: {name}")
            self._cascades.append(clf)

    def _haar(self, gray, min_w, min_h):
        gray = cv2.equalizeHist(gray)
        for cascade in self._cascades:
            if len(cascade.detectMultiScale(gray, 1.04, 2, cv2.CASCADE_SCALE_IMAGE, (max(16, min_w), max(16, min_h)))) > 0:
                return True
        return False

    def is_visible(self, frame, bbox) -> bool:
        box = clamp_box(bbox, frame.shape)
        if box is None:
            return False
        x1, y1, x2, y2 = box
        bh, bw = y2 - y1, x2 - x1
        if bh < 28 or bw < 16:
            return False
        upper = cv2.cvtColor(frame[y1 : y1 + max(1, bh // 2), x1:x2], cv2.COLOR_BGR2GRAY)
        return self._haar(upper, bw // 10, bh // 10) or (bh / max(bw, 1) >= 0.45)


class FaceBoxExtractor:
    def __init__(self, pad_frac=0.12, upper_frac=0.55):
        base = cv2.data.haarcascades
        self._face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self._profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        self.pad_frac = pad_frac
        self.upper_frac = upper_frac

    def extract(self, frame, person_bbox):
        box = clamp_box(person_bbox, frame.shape)
        if box is None:
            return None
        fh, fw = frame.shape[:2]
        px1, py1, px2, py2 = box
        sy2 = py1 + max(1, int((py2 - py1) * self.upper_frac))
        crop = cv2.cvtColor(frame[py1:sy2, px1:px2], cv2.COLOR_BGR2GRAY)
        crop = cv2.equalizeHist(crop)
        hits = []
        for c in (self._face, self._profile):
            found = c.detectMultiScale(crop, 1.05, 3, cv2.CASCADE_SCALE_IMAGE, (16, 16))
            if len(found):
                hits.extend(found.tolist())
        if not hits:
            return None
        fx, fy, w, h = max(hits, key=lambda r: r[2] * r[3])
        x1, y1 = px1 + fx, py1 + fy
        x2, y2 = x1 + w, y1 + h
        pad_x, pad_y = self.pad_frac * (x2 - x1), self.pad_frac * (y2 - y1)
        return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame.shape)
