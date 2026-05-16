"""Bounding-box utilities."""

from __future__ import annotations

import cv2
import numpy as np


def bbox_area_xyxy(box) -> float:
    x1, y1, x2, y2 = box[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def bbox_mostly_inside(inner, outer, thresh: float) -> bool:
    ix1, iy1, ix2, iy2 = inner[:4]
    ox1, oy1, ox2, oy2 = outer[:4]
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    inner_area = bbox_area_xyxy(inner)
    if inner_area <= 0.0:
        return False
    return (inter / inner_area) >= thresh


def nms_detections(detections, iou_thresh: float):
    if len(detections) <= 1:
        return detections
    boxes_xywh, scores = [], []
    for x1, y1, x2, y2, score, _cls in detections:
        boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        scores.append(float(score))
    keep = cv2.dnn.NMSBoxes(
        boxes_xywh, scores, score_threshold=0.0, nms_threshold=float(iou_thresh)
    )
    if keep is None or len(keep) == 0:
        return []
    idx = np.asarray(keep, dtype=np.int64).reshape(-1)
    return [detections[i] for i in idx]


def filter_person_boxes(
    entries,
    frame_shape,
    min_h_frac: float,
    min_w_frac: float,
    min_area_frac: float,
    nested_thresh: float,
):
    if not entries:
        return entries
    fh, fw = frame_shape[:2]
    min_h = fh * min_h_frac
    min_w = fw * min_w_frac
    min_area = fh * fw * min_area_frac

    sized = []
    for row in entries:
        x1, y1, x2, y2 = row[:4]
        bh, bw = y2 - y1, x2 - x1
        if bh < min_h or bw < min_w or bh * bw < min_area:
            continue
        sized.append(row)

    sized.sort(key=lambda r: -bbox_area_xyxy(r))
    kept = []
    for row in sized:
        if any(bbox_mostly_inside(row, other, nested_thresh) for other in kept):
            continue
        kept.append(row)
    return kept


def offset_detections(detections, ox: float, oy: float):
    if ox == 0 and oy == 0:
        return detections
    out = []
    for row in detections:
        r = list(row)
        r[0] += ox
        r[1] += oy
        r[2] += ox
        r[3] += oy
        out.append(r)
    return out
