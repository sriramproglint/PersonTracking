import cv2
import os
import numpy as np
import paddle
import importlib
import sys
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
local_ppdet_root = BASE_DIR / "PaddleDetection"
if local_ppdet_root.exists():
    sys.path.insert(0, str(local_ppdet_root))

try:
    load_config = importlib.import_module("ppdet.core.workspace").load_config
    Trainer = importlib.import_module("ppdet.engine").Trainer
    ppdet_module = importlib.import_module("ppdet")
except ModuleNotFoundError as exc:
    hint = getattr(exc, "name", None)
    detail = f" (missing: {hint!r})" if hint else ""
    raise ModuleNotFoundError(
        "Could not import PaddleDetection (`ppdet`). Clone PaddleDetection next "
        "to this script, activate your venv, and run "
        "`pip install -r requirements.txt`."
        f"{detail}"
    ) from exc

# Load Paddle RT-DETR
PADDLEDET_ROOT = Path(ppdet_module.__file__).resolve().parent.parent
config_candidates = [
    BASE_DIR / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
    PADDLEDET_ROOT / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
]
config_path = next((p for p in config_candidates if p.exists()), None)
if config_path is None:
    raise FileNotFoundError(
        "RT-DETR config not found. Checked: "
        + ", ".join(str(p) for p in config_candidates)
    )

cfg = load_config(str(config_path))

local_weights = BASE_DIR / "rtdetr_r50vd_6x_coco.pdparams"
cfg.weights = (
    str(local_weights)
    if local_weights.exists()
    else "https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams"
)

trainer = Trainer(cfg, mode='test')
trainer.load_weights(cfg.weights)
model = trainer.model
model.eval()

# Multi-camera sources
def pick_existing_path(candidates):
    for cand in candidates:
        p = BASE_DIR / cand
        if p.exists():
            return str(p)
    return str(BASE_DIR / candidates[0])

sources = {
    "cam1": pick_existing_path(["cam1.mp4", "left.mp4"]),
    "cam2": pick_existing_path(["cam2.mp4", "right.mp4"]),
}

caps = {k: cv2.VideoCapture(v) for k, v in sources.items()}
for cam_id, cap in caps.items():
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source for {cam_id}: {sources[cam_id]}")

frame_counter = 0
TARGET_FPS = 15
source_fps = {
    cam_id: (cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0)
    for cam_id, cap in caps.items()
}
frame_skip = {
    cam_id: max(1, int(round(source_fps[cam_id] / TARGET_FPS)))
    for cam_id in sources
}
frame_index_by_cam = {cam_id: 0 for cam_id in sources}
ended_cams = set()
PERSON_CLASS_INDEX = 0
MODEL_INPUT_SIZE = 640

BOXMOT_ROOT = BASE_DIR / "PaddleDetection" / "Yolov5_StrongSORT_OSNet"
if not BOXMOT_ROOT.is_dir():
    raise RuntimeError(
        "StrongSORT (BoxMOT) is required but was not found at "
        f"{BOXMOT_ROOT}. Clone PaddleDetection with the "
        "Yolov5_StrongSORT_OSNet tree present."
    )
sys.path.insert(0, str(BOXMOT_ROOT))

_STRONGSORT_DEVICE = os.environ.get(
    "STRONGSORT_DEVICE",
    "cuda:0" if torch.cuda.is_available() else "cpu",
)
_STRONGSORT_REID_WEIGHTS = os.environ.get("STRONGSORT_REID_WEIGHTS")


def bbox_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def embedding_cosine_similarity(a, b):
    """Cosine similarity for L2-normalized Re-ID vectors (dot product)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b))


def global_id_bgr(gid):
    """Stable display color per global ID (BGR)."""
    r = (37 * gid) % 200 + 55
    g = (17 * gid * gid) % 200 + 55
    b = (91 * gid) % 200 + 55
    return int(b), int(g), int(r)


class GlobalIdentityRegistry:
    """Maps local StrongSORT IDs to one shared global ID using Re-ID embeddings (OSNet)."""

    def __init__(self, sim_thresh=0.52, ema_alpha=0.2):
        self.sim_thresh = sim_thresh
        self.ema_alpha = ema_alpha
        self._next_gid = 1
        self._gallery = {}

    def assign(self, observations):
        """
        observations: iterable of dict(cam_id, track_id, bbox, emb)
        emb: L2-normalized OSNet feature (same space as StrongSORT).
        Returns mapping (cam_id, track_id) -> global_id
        """
        obs_list = list(observations)
        obs_list.sort(key=lambda o: -bbox_area(o["bbox"]))
        gid_map = {}
        used_gids = set()

        for obs in obs_list:
            cam_id = obs["cam_id"]
            tid = int(obs["track_id"])
            emb = np.asarray(obs["emb"], dtype=np.float32).ravel()
            candidates = []
            for gid, pdata in self._gallery.items():
                sim = embedding_cosine_similarity(emb, pdata["emb"])
                if sim >= self.sim_thresh:
                    candidates.append((sim, gid))
            candidates.sort(key=lambda x: (-x[0], x[1]))

            chosen_gid = None
            for _sim, cand_gid in candidates:
                if cand_gid not in used_gids:
                    chosen_gid = cand_gid
                    break

            if chosen_gid is not None:
                gid = chosen_gid
                used_gids.add(gid)
                oh = self._gallery[gid]["emb"]
                merged = (1.0 - self.ema_alpha) * oh + self.ema_alpha * emb
                nrm = np.linalg.norm(merged) + 1e-12
                self._gallery[gid]["emb"] = (merged / nrm).astype(np.float32)
            else:
                gid = self._next_gid
                self._next_gid += 1
                used_gids.add(gid)
                self._gallery[gid] = {"emb": emb.copy()}

            gid_map[(cam_id, tid)] = gid

        return gid_map


class StrongSortCameraTracker:
    """
    Wraps BoxMOT StrongSort to match this script's ``update(detections, frame)``.

    ``detections`` must be Nx6 rows: ``x1,y1,x2,y2,conf,cls`` (axis-aligned).
    Returns rows ``[x1,y1,x2,y2,track_id,...]`` (track id at index 4).
    """

    def __init__(self, tracker_inner):
        self._tracker = tracker_inner

    def update(self, detections, frame):
        if not detections:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.asarray(detections, dtype=np.float32)
            if dets.ndim != 2 or dets.shape[1] != 6:
                raise ValueError(
                    "StrongSort expects detections shaped (N, 6): "
                    "x1,y1,x2,y2,conf,cls"
                )
        out = self._tracker.update(dets, np.asarray(frame))
        if out is None or len(out) == 0:
            return []
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out.tolist()


try:
    from boxmot.reid.core import ReID
    from boxmot.trackers.tracker_zoo import create_tracker
except ImportError as exc:
    raise ImportError(
        "BoxMOT StrongSORT could not be imported. From your venv run:\n"
        "  pip install 'lapx>=0.5.5,<1' 'gdown>=5.1,<6' 'opencv-python>=4.7,<5' "
        "rich click 'regex>=2024' 'ftfy>=6.1,<7' 'yacs>=0.1,<1' "
        "'huggingface-hub>=1.7' filelock pandas\n"
        f"(also ensure {BOXMOT_ROOT} is on PYTHONPATH — main.py adds it automatically)."
    ) from exc

_reid_kw = {"device": _STRONGSORT_DEVICE, "half": False}
if _STRONGSORT_REID_WEIGHTS:
    _reid_kw["weights"] = _STRONGSORT_REID_WEIGHTS
_shared_reid = ReID(**_reid_kw).model

strongsort_by_cam = {
    cam_id: StrongSortCameraTracker(
        create_tracker(
            "strongsort",
            reid_model=_shared_reid,
            device=_STRONGSORT_DEVICE,
            half=False,
        )
    )
    for cam_id in sources
}
_GLOBAL_REID_COSINE_THRESH = float(
    os.environ.get("GLOBAL_REID_COSINE_THRESH", "0.52")
)
global_registry = GlobalIdentityRegistry(
    sim_thresh=_GLOBAL_REID_COSINE_THRESH,
    ema_alpha=0.2,
)
print(
    f"[INFO] StrongSORT (BoxMOT) enabled — shared Re-ID backend, "
    f"device {_STRONGSORT_DEVICE}. "
    f"Optional: STRONGSORT_REID_WEIGHTS / STRONGSORT_DEVICE env vars."
)
print(
    f"[INFO] Same person ID across cam1/cam2: OSNet embeddings "
    f"(cosine ≥ {_GLOBAL_REID_COSINE_THRESH}; tune GLOBAL_REID_COSINE_THRESH)."
)

def preprocess(frame):
    # Align with configs/rtdetr/_base_/rtdetr_reader.yml TestReader: resize 640,
    # NormalizeImage norm_type none (scale 1/255 only). im_shape is resized HW.
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return {
        "image": paddle.to_tensor(img),
        "im_shape": paddle.to_tensor(
            np.array([[MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]], dtype="float32")
        ),
        "scale_factor": paddle.to_tensor(
            np.array([[MODEL_INPUT_SIZE / h, MODEL_INPUT_SIZE / w]], dtype="float32")
        ),
    }

while True:
    if len(ended_cams) == len(caps):
        break

    pending_observations = []
    display_batch = []

    for cam_id, cap in caps.items():
        if cam_id in ended_cams:
            continue

        ret, frame = cap.read()
        if not ret:
            ended_cams.add(cam_id)
            continue
        frame_index_by_cam[cam_id] += 1
        if frame_index_by_cam[cam_id] % frame_skip[cam_id] != 0:
            continue

        inputs = preprocess(frame)

        with paddle.no_grad():
            outputs = model(inputs)

        # Paddle RT-DETR output parsing
        # format: [class, score, x1, y1, x2, y2]
        detections = []
        raw_dets = outputs["bbox"] if isinstance(outputs, dict) else outputs[0]
        frame_h, frame_w = frame.shape[:2]
        for det in raw_dets.numpy():
            cls, score, x1, y1, x2, y2 = det
            # Some outputs are normalized [0,1], others are absolute pixels.
            if max(x1, y1, x2, y2) <= 1.5:
                x1, x2 = x1 * frame_w, x2 * frame_w
                y1, y2 = y1 * frame_h, y2 * frame_h
            x1 = max(0.0, min(float(frame_w - 1), float(x1)))
            y1 = max(0.0, min(float(frame_h - 1), float(y1)))
            x2 = max(0.0, min(float(frame_w - 1), float(x2)))
            y2 = max(0.0, min(float(frame_h - 1), float(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            if int(cls) == PERSON_CLASS_INDEX and score > 0.60:  # person class
                detections.append([x1, y1, x2, y2, float(score), 0.0])

        tracks = strongsort_by_cam[cam_id].update(detections, frame)
        frame_counter += 1
        if frame_counter % 30 == 0:
            print(f"[{cam_id}] frame={frame_counter} person_detections={len(detections)}")

        if tracks:
            xyxy = np.asarray([tr[:4] for tr in tracks], dtype=np.float32)
            embs = _shared_reid.get_features(xyxy, frame)
            embs = np.atleast_2d(np.asarray(embs, dtype=np.float32))
            for i, tr in enumerate(tracks):
                pending_observations.append(
                    {
                        "cam_id": cam_id,
                        "track_id": int(tr[4]),
                        "bbox": tr[:4],
                        "emb": embs[i].copy(),
                    }
                )
        display_batch.append((cam_id, frame, tracks))

    gid_map = global_registry.assign(pending_observations)

    for cam_id, frame, tracks in display_batch:
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            tid = int(track[4])
            gid = gid_map.get((cam_id, tid))
            color = global_id_bgr(gid) if gid is not None else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = (
                f"ID:{gid}"
                if gid is not None
                else f"{cam_id} L{tid}"
            )
            cv2.putText(
                frame,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        cv2.imshow(cam_id, frame)

    if cv2.waitKey(max(1, int(1000 / TARGET_FPS))) == 27:
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()