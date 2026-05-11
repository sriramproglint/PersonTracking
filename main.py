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
    """
    Maintains a single shared Global ID (GID) for the same person across cameras.

    Key properties:
      * Each local StrongSORT track ``(cam_id, track_id)`` is *sticky-bound*
        to a GID for the lifetime of that local track. A single low-similarity
        frame (occlusion, blur, side view) no longer triggers a new GID.
      * The gallery stores a separate running EMA embedding per camera per
        GID. Cross-view matching uses the max similarity across all stored
        per-camera views, so a person seen from very different angles in
        cam1 vs cam2 can still be merged.
      * New (not yet sticky-bound) tracks are matched to existing GIDs via
        Hungarian assignment on cosine similarity, with the constraint that
        two active local tracks in the same camera cannot share a GID.
      * Sticky bindings whose owning local track disappears for more than
        ``stale_frames`` are pruned so GIDs can be re-used.
    """

    def __init__(self, sim_thresh=0.52, ema_alpha=0.2, stale_frames=300):
        self.sim_thresh = sim_thresh
        self.ema_alpha = ema_alpha
        self.stale_frames = stale_frames
        self._next_gid = 1
        # gid -> {"views_by_cam": {cam_id: np.ndarray}, "last_seen": int}
        self._gallery = {}
        # (cam_id, track_id) -> {"gid": int, "last_seen": int}
        self._track_to_gid = {}
        self._frame = 0

    def _all_views(self, gid):
        return list(self._gallery[gid]["views_by_cam"].values())

    def _best_sim(self, emb, gid):
        sims = [embedding_cosine_similarity(emb, v) for v in self._all_views(gid)]
        return max(sims) if sims else -1.0

    def _update_views(self, gid, emb, cam_id):
        g = self._gallery[gid]
        existing = g["views_by_cam"].get(cam_id)
        if existing is None:
            g["views_by_cam"][cam_id] = emb.copy()
        else:
            merged = (1.0 - self.ema_alpha) * existing + self.ema_alpha * emb
            nrm = float(np.linalg.norm(merged)) + 1e-12
            g["views_by_cam"][cam_id] = (merged / nrm).astype(np.float32)
        g["last_seen"] = self._frame

    def _busy_gids_for_cam(self, cam_id):
        """GIDs already bound to an active sticky track in ``cam_id``."""
        return {
            info["gid"]
            for k, info in self._track_to_gid.items()
            if k[0] == cam_id
        }

    @staticmethod
    def _greedy_assignment(sim):
        rows, cols = sim.shape
        flat = [(sim[r, c], r, c) for r in range(rows) for c in range(cols)]
        flat.sort(key=lambda x: -x[0])
        used_r, used_c = set(), set()
        ri, ci = [], []
        for _, r, c in flat:
            if r in used_r or c in used_c:
                continue
            ri.append(r)
            ci.append(c)
            used_r.add(r)
            used_c.add(c)
        return np.asarray(ri, dtype=np.int64), np.asarray(ci, dtype=np.int64)

    def assign(self, observations):
        """
        observations: iterable of dict(cam_id, track_id, bbox, emb)
        emb: L2-normalized OSNet feature (same space as StrongSORT).
        Returns mapping (cam_id, track_id) -> global_id.
        """
        self._frame += 1
        gid_map = {}
        used_gids_this_frame = set()
        new_obs = []  # tracks lacking a sticky GID

        # Phase 1: keep sticky bindings for tracks we've already seen.
        for obs in observations:
            key = (obs["cam_id"], int(obs["track_id"]))
            emb = np.asarray(obs["emb"], dtype=np.float32).ravel()
            sticky = self._track_to_gid.get(key)
            if sticky is not None:
                gid = sticky["gid"]
                sticky["last_seen"] = self._frame
                self._update_views(gid, emb, obs["cam_id"])
                gid_map[key] = gid
                used_gids_this_frame.add(gid)
            else:
                new_obs.append((obs, emb))

        # Phase 2: match new tracks to existing GIDs via Hungarian assignment.
        candidate_gids = [g for g in self._gallery if g not in used_gids_this_frame]
        unassigned_rows = set(range(len(new_obs)))

        if new_obs and candidate_gids:
            rows, cols = len(new_obs), len(candidate_gids)
            sim = np.full((rows, cols), -1.0, dtype=np.float32)
            for i, (obs, emb) in enumerate(new_obs):
                cam_busy = self._busy_gids_for_cam(obs["cam_id"])
                for j, gid in enumerate(candidate_gids):
                    if gid in cam_busy:
                        # Same-camera exclusivity: two local tracks in the
                        # same camera cannot share a GID.
                        continue
                    sim[i, j] = self._best_sim(emb, gid)

            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(-sim)
            except ImportError:
                row_ind, col_ind = self._greedy_assignment(sim)

            for r, c in zip(row_ind, col_ind):
                if r >= rows or c >= cols:
                    continue
                if sim[r, c] < self.sim_thresh:
                    continue
                gid = candidate_gids[c]
                obs, emb = new_obs[r]
                key = (obs["cam_id"], int(obs["track_id"]))
                self._track_to_gid[key] = {"gid": gid, "last_seen": self._frame}
                self._update_views(gid, emb, obs["cam_id"])
                gid_map[key] = gid
                used_gids_this_frame.add(gid)
                unassigned_rows.discard(r)

        # Phase 3: anything still unmatched gets a brand-new GID.
        for r in sorted(unassigned_rows):
            obs, emb = new_obs[r]
            key = (obs["cam_id"], int(obs["track_id"]))
            gid = self._next_gid
            self._next_gid += 1
            self._gallery[gid] = {
                "views_by_cam": {obs["cam_id"]: emb.copy()},
                "last_seen": self._frame,
            }
            self._track_to_gid[key] = {"gid": gid, "last_seen": self._frame}
            gid_map[key] = gid
            used_gids_this_frame.add(gid)

        # Phase 4: prune sticky bindings whose owners have vanished.
        stale = [
            k for k, info in self._track_to_gid.items()
            if self._frame - info["last_seen"] > self.stale_frames
        ]
        for k in stale:
            del self._track_to_gid[k]

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


def _parse_zone_rect(env_value, default):
    """Parse 'x1,y1,x2,y2' in normalized [0,1] coords. Returns (x1,y1,x2,y2)."""
    raw = os.environ.get(env_value)
    if not raw:
        return default
    try:
        parts = [float(p.strip()) for p in raw.split(",")]
        if len(parts) != 4:
            raise ValueError
        x1, y1, x2, y2 = parts
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        return (x1, y1, x2, y2)
    except (ValueError, TypeError):
        print(
            f"[WARN] {env_value}={raw!r} is invalid (expected 'x1,y1,x2,y2' "
            f"in [0,1]); falling back to default {default}."
        )
        return default


# Defaults assume people walk roughly left-to-right between cameras:
#   cam1 exit  = right strip of cam1 (x in [0.70, 1.00], full height)
#   cam2 entry = left  strip of cam2 (x in [0.00, 0.30], full height)
# Override with env vars CAM1_EXIT_ZONE / CAM2_ENTRY_ZONE as 'x1,y1,x2,y2'
# in normalized [0,1] coordinates.
ZONES = {
    "cam1": {
        "label": "EXIT (-> cam2)",
        "rect_norm": _parse_zone_rect("CAM1_EXIT_ZONE", (0.00, 0.00, 0.55, 1.00)),
        "color_bgr": (0, 80, 220),   # red-ish
    },
    "cam2": {
        "label": "ENTRY (<- cam1)",
        "rect_norm": _parse_zone_rect("CAM2_ENTRY_ZONE", (0.50, 0.00, 1.00, 1.00)),
        "color_bgr": (0, 200, 80),   # green-ish
    },
}


def draw_zone(frame, zone, alpha=0.25):
    """Translucent fill + outline + label for a normalized rectangle zone."""
    h, w = frame.shape[:2]
    x1n, y1n, x2n, y2n = zone["rect_norm"]
    x1, y1 = int(round(x1n * w)), int(round(y1n * h))
    x2, y2 = int(round(x2n * w)), int(round(y2n * h))
    if x2 <= x1 or y2 <= y1:
        return
    color = zone["color_bgr"]
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
    label = zone["label"]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pad = 4
    tx, ty = x1 + pad, y1 + th + pad
    cv2.rectangle(
        frame,
        (tx - pad, ty - th - pad),
        (tx + tw + pad, ty + pad),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def bbox_in_zone(bbox, zone, frame_shape):
    """True if the bbox's bottom-center (foot point) is inside the zone."""
    h, w = frame_shape[:2]
    x1n, y1n, x2n, y2n = zone["rect_norm"]
    zx1, zy1 = x1n * w, y1n * h
    zx2, zy2 = x2n * w, y2n * h
    bx1, by1, bx2, by2 = bbox
    cx = 0.5 * (bx1 + bx2)
    cy = by2  # foot point on the ground plane
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


class HandoffEventLogger:
    """
    Detects cross-camera handoff events from per-camera local-track lifecycles
    inside each camera's handoff zone, and reports them with the assigned GID.

    Events (per camera, bidirectional):
      * ENTRY: a brand-new local track first appears inside the handoff zone
               -> the person likely arrived FROM the other camera.
      * EXIT:  a local track stops being tracked while its last known
               position was inside the handoff zone -> the person likely
               left TO the other camera.

    A track that was gone longer than ``vanish_frames`` is treated as a new
    lifecycle on next sighting, which correctly handles StrongSORT ID recycling.
    """

    def __init__(self, vanish_frames=10, entry_grace_frames=3):
        self.vanish_frames = int(vanish_frames)
        self.entry_grace_frames = int(entry_grace_frames)
        self._state = {}  # (cam_id, track_id) -> dict
        self._frame = 0

    def begin_frame(self):
        self._frame += 1
        return self._frame

    def observe(self, cam_id, track_id, gid, in_zone):
        """Record a per-track observation; returns any newly fired events."""
        key = (cam_id, int(track_id))
        st = self._state.get(key)
        if st is not None and self._frame - st["last_seen_frame"] > self.vanish_frames:
            # Track was gone long enough that we treat this sighting as a
            # fresh lifecycle (StrongSORT may recycle local track ids).
            st = None

        events = []
        if st is None:
            st = {
                "first_seen_frame": self._frame,
                "last_seen_frame": self._frame,
                "first_in_zone": bool(in_zone),
                "last_in_zone": bool(in_zone),
                "last_gid": gid,
                "entry_logged": False,
                "exit_logged": False,
            }
            self._state[key] = st
            if in_zone and gid is not None:
                events.append(("ENTRY", cam_id, gid, self._frame))
                st["entry_logged"] = True
        else:
            st["last_seen_frame"] = self._frame
            st["last_in_zone"] = bool(in_zone)
            if gid is not None:
                st["last_gid"] = gid
            # Retroactive grace window: if the very first sighting was in
            # zone but the GID hadn't been assigned yet, log ENTRY now.
            if (
                in_zone
                and gid is not None
                and not st["entry_logged"]
                and st["first_in_zone"]
                and self._frame - st["first_seen_frame"] <= self.entry_grace_frames
            ):
                events.append(("ENTRY", cam_id, gid, self._frame))
                st["entry_logged"] = True
            st["exit_logged"] = False  # alive again
        return events

    def sweep_exits(self):
        """Call after all cameras processed for the frame; logs vanished tracks."""
        events = []
        stale_keys = []
        for key, st in self._state.items():
            if st["exit_logged"]:
                if self._frame - st["last_seen_frame"] > 10 * self.vanish_frames:
                    stale_keys.append(key)
                continue
            if self._frame - st["last_seen_frame"] >= self.vanish_frames:
                if st["last_in_zone"] and st["last_gid"] is not None:
                    events.append(
                        ("EXIT", key[0], st["last_gid"], st["last_seen_frame"])
                    )
                st["exit_logged"] = True
        for k in stale_keys:
            del self._state[k]
        return events


def _print_handoff_events(events):
    for kind, cam_id, gid, frame_idx in events:
        # Pad kind to 5 chars so EXIT/ENTRY line up in the terminal.
        print(f"[HANDOFF f={frame_idx:>5}] {cam_id} {kind:<5} GID:{gid}")
print(
    f"[INFO] StrongSORT (BoxMOT) enabled — shared Re-ID backend, "
    f"device {_STRONGSORT_DEVICE}. "
    f"Optional: STRONGSORT_REID_WEIGHTS / STRONGSORT_DEVICE env vars."
)
print(
    f"[INFO] Shared GID across cam1/cam2 via OSNet Re-ID "
    f"(cosine ≥ {_GLOBAL_REID_COSINE_THRESH}; tune GLOBAL_REID_COSINE_THRESH). "
    f"Sticky per-track binding + multi-view gallery + Hungarian matching."
)
print(
    "[INFO] Handoff zones (normalized x1,y1,x2,y2): "
    f"cam1 EXIT={ZONES['cam1']['rect_norm']}, "
    f"cam2 ENTRY={ZONES['cam2']['rect_norm']}. "
    "Override with CAM1_EXIT_ZONE / CAM2_ENTRY_ZONE env vars."
)

_HANDOFF_VANISH_FRAMES = int(os.environ.get("HANDOFF_VANISH_FRAMES", "10"))
handoff_logger = HandoffEventLogger(vanish_frames=_HANDOFF_VANISH_FRAMES)
print(
    f"[INFO] Handoff logger active (vanish_frames={_HANDOFF_VANISH_FRAMES}); "
    "ENTRY/EXIT events with GID are reported to this terminal "
    "(works both cam1->cam2 and cam2->cam1)."
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
    handoff_logger.begin_frame()

    for cam_id, frame, tracks in display_batch:
        zone = ZONES.get(cam_id)
        if zone is not None:
            draw_zone(frame, zone)

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            tid = int(track[4])
            gid = gid_map.get((cam_id, tid))
            color = global_id_bgr(gid) if gid is not None else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            in_zone = zone is not None and bbox_in_zone(
                (x1, y1, x2, y2), zone, frame.shape
            )
            if zone is not None:
                _print_handoff_events(
                    handoff_logger.observe(cam_id, tid, gid, in_zone)
                )
            base = f"GID:{gid}" if gid is not None else f"{cam_id} L{tid}"
            if in_zone:
                marker = "EXIT" if cam_id == "cam1" else "ENTRY"
                label = f"{base} [{marker}]"
            else:
                label = base
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

    _print_handoff_events(handoff_logger.sweep_exits())

    if cv2.waitKey(max(1, int(1000 / TARGET_FPS))) == 27:
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()