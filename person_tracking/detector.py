"""RT-DETR person detector (PaddleDetection)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import paddle

from person_tracking.bbox import filter_person_boxes, nms_detections, offset_detections
from person_tracking.config import BASE_DIR, PADDLEDET_ROOT, Config
from person_tracking.roi import RoiZone

PERSON_CLASS = 0


def _import_ppdet():
    if PADDLEDET_ROOT.exists():
        sys.path.insert(0, str(PADDLEDET_ROOT))
    load_config = importlib.import_module("ppdet.core.workspace").load_config
    Trainer = importlib.import_module("ppdet.engine").Trainer
    return load_config, Trainer


class RtdetrDetector:
    def __init__(self, cfg: Config, paddle_cuda: bool):
        load_config, Trainer = _import_ppdet()
        roots = [
            cfg.base_dir / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
            PADDLEDET_ROOT / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        ]
        config_path = next((p for p in roots if p.exists()), None)
        if config_path is None:
            raise FileNotFoundError("RT-DETR config not found under " + str(cfg.base_dir))

        yml = load_config(str(config_path))
        yml.eval_size = [cfg.imgsz, cfg.imgsz]
        local = cfg.base_dir / "rtdetr_r50vd_6x_coco.pdparams"
        yml.weights = str(local) if local.exists() else (
            "https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams"
        )

        trainer = Trainer(yml, mode="test")
        trainer.load_weights(yml.weights)
        self.model = trainer.model
        self.model.eval()
        # FP16 can fail on some GPU stacks; enable only when PADDLE_FP16=1 and runtime supports it.
        if cfg.paddle_fp16 and paddle_cuda:
            try:
                self.model = self.model.astype("float16")
            except Exception as exc:
                print(f"[WARN] FP16 disabled: {exc}")
                cfg.paddle_fp16 = False

        self.cfg = cfg
        self._imgsz = float(cfg.imgsz)
        self._scale = np.array([[cfg.imgsz, cfg.imgsz]], dtype=np.float32)
        self._im_shape = self._scale.copy()
        self._visibility = None

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (self.cfg.imgsz, self.cfg.imgsz), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img, dtype=np.float32) * (1.0 / 255.0)
        img = img.transpose((2, 0, 1))[np.newaxis, ...]
        scale = self._scale.copy()
        scale[0, 0] = self._imgsz / h
        scale[0, 1] = self._imgsz / w
        return {
            "image": paddle.to_tensor(img),
            "im_shape": paddle.to_tensor(self._im_shape),
            "scale_factor": paddle.to_tensor(scale),
        }

    def _infer(self, inputs):
        with paddle.no_grad():
            if self.cfg.paddle_fp16 and paddle.device.is_compiled_with_cuda():
                with paddle.amp.auto_cast(enable=True, dtype="float16"):
                    return self.model(inputs)
            return self.model(inputs)

    def _parse_outputs(self, outputs, infer_shape, full_frame):
        raw = outputs["bbox"] if isinstance(outputs, dict) else outputs[0]
        ih, iw = infer_shape[:2]
        fh, fw = full_frame.shape[:2]
        dets = []
        for det in raw.numpy():
            cls, score, x1, y1, x2, y2 = det
            if max(x1, y1, x2, y2) <= 1.5:
                x1, x2 = x1 * iw, x2 * iw
                y1, y2 = y1 * ih, y2 * ih
            x1 = max(0.0, min(float(fw - 1), float(x1)))
            y1 = max(0.0, min(float(fh - 1), float(y1)))
            x2 = max(0.0, min(float(fw - 1), float(x2)))
            y2 = max(0.0, min(float(fh - 1), float(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            if int(cls) == PERSON_CLASS and score >= self.cfg.conf_thres:
                box = [x1, y1, x2, y2, float(score), 0.0]
                if self._box_allowed(full_frame, box[:4]):
                    dets.append(box)
        return dets

    def _box_allowed(self, frame, bbox) -> bool:
        from person_tracking.visibility import PersonVisibilityFilter, box_size_ok

        if self._visibility is None:
            self._visibility = (
                PersonVisibilityFilter() if self.cfg.enable_visibility_filter else False
            )
        if self._visibility is False:
            return box_size_ok(frame, bbox)
        return self._visibility.is_visible(frame, bbox)

    def detect(self, frame, roi: RoiZone):
        roi_pts = roi.for_frame(frame.shape)
        ox, oy = 0, 0
        infer = frame
        if self.cfg.roi_detect_crop:
            fh, fw = frame.shape[:2]
            x1, y1, x2, y2 = roi.crop_bounds(roi_pts, fw, fh, self.cfg.roi_crop_margin)
            ox, oy = x1, y1
            infer = frame[y1:y2, x1:x2]

        outputs = self._infer(self._preprocess(infer))
        dets = offset_detections(
            self._parse_outputs(outputs, infer.shape, frame), ox, oy
        )
        dets = filter_person_boxes(
            dets,
            frame.shape,
            self.cfg.min_box_h_frac,
            self.cfg.min_box_w_frac,
            self.cfg.min_box_area_frac,
            self.cfg.nested_inside_thresh,
        )
        return nms_detections(dets, self.cfg.iou_thres), roi_pts
