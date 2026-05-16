"""Global person ID assignment with sticky binding and re-entry stabilization."""

from __future__ import annotations

import numpy as np

from person_tracking.bbox import bbox_iou_xyxy


def cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b))


class GlobalIdentityRegistry:
    """
    Assigns stable global IDs (GIDs) using:
      - Sticky (cam_id, track_id) -> GID for the life of a local track
      - Lost-track buffer to reconnect after StrongSORT issues a new track_id
      - Spatial + appearance gates before re-entry or new GID creation
    """

    def __init__(
        self,
        reentry_thresh=0.58,
        new_gid_thresh=0.48,
        ema_alpha=0.12,
        retire_frames=450,
        stale_frames=300,
        lost_track_frames=90,
        spatial_iou_thresh=0.12,
        match_margin=0.06,
    ):
        self.reentry_thresh = float(reentry_thresh)
        self.new_gid_thresh = float(new_gid_thresh)
        self.ema_alpha = float(ema_alpha)
        self.retire_frames = int(retire_frames)
        self.stale_frames = int(stale_frames)
        self.lost_track_frames = int(lost_track_frames)
        self.spatial_iou_thresh = float(spatial_iou_thresh)
        self.match_margin = float(match_margin)

        self._next_gid = 1
        self._retired: set[int] = set()
        self._gallery: dict[int, dict] = {}
        self._track_to_gid: dict[tuple, dict] = {}
        self._track_last: dict[tuple, dict] = {}
        self._gid_state: dict[int, dict] = {}
        self._lost: list[dict] = []
        self._frame = 0

    def _retire(self, gid: int) -> None:
        self._retired.add(gid)

    def _eligible(self, gid: int) -> bool:
        if gid in self._retired or gid not in self._gallery:
            return False
        if self._frame - self._gallery[gid]["last_seen"] > self.retire_frames:
            self._retire(gid)
            return False
        return True

    def _best_sim(self, emb, gid: int) -> float:
        views = self._gallery[gid]["views_by_cam"].values()
        sims = [cosine_sim(emb, v) for v in views]
        return max(sims) if sims else -1.0

    def _spatial_ok(self, gid: int, bbox) -> bool:
        st = self._gid_state.get(gid)
        if st is None or "bbox" not in st:
            return True
        return bbox_iou_xyxy(bbox, st["bbox"]) >= self.spatial_iou_thresh

    def _update_view(self, gid: int, emb, cam_id: str, bbox) -> None:
        g = self._gallery[gid]
        old = g["views_by_cam"].get(cam_id)
        if old is None:
            g["views_by_cam"][cam_id] = emb.copy()
        else:
            merged = (1.0 - self.ema_alpha) * old + self.ema_alpha * emb
            nrm = float(np.linalg.norm(merged)) + 1e-12
            g["views_by_cam"][cam_id] = (merged / nrm).astype(np.float32)
        g["last_seen"] = self._frame
        self._gid_state[gid] = {"bbox": list(bbox[:4]), "cam_id": cam_id, "frame": self._frame}

    def _bind(self, key, gid: int, emb, cam_id: str, bbox) -> None:
        self._track_to_gid[key] = {"gid": gid, "last_seen": self._frame}
        self._track_last[key] = {"emb": emb.copy(), "bbox": list(bbox[:4])}
        self._update_view(gid, emb, cam_id, bbox)

    def _busy_gids(self, cam_id: str) -> set[int]:
        return {v["gid"] for k, v in self._track_to_gid.items() if k[0] == cam_id}

    @staticmethod
    def _greedy(sim: np.ndarray):
        rows, cols = sim.shape
        pairs = sorted(
            [(sim[r, c], r, c) for r in range(rows) for c in range(cols)],
            reverse=True,
        )
        used_r, used_c, ri, ci = set(), set(), [], []
        for _, r, c in pairs:
            if r not in used_r and c not in used_c:
                ri.append(r)
                ci.append(c)
                used_r.add(r)
                used_c.add(c)
        return np.asarray(ri, dtype=np.int64), np.asarray(ci, dtype=np.int64)

    def _prune_lost(self) -> None:
        self._lost = [
            e
            for e in self._lost
            if self._frame - e["lost_frame"] <= self.lost_track_frames
        ]

    def _match_lost_track(self, obs, emb, busy: set[int]) -> int | None:
        """Reconnect a new local track_id to a recently lost GID (same person)."""
        bbox = obs["bbox"]
        best_gid, best_score = None, self.reentry_thresh
        for entry in self._lost:
            gid = entry["gid"]
            if gid in busy or not self._eligible(gid):
                continue
            if entry["cam_id"] != obs["cam_id"]:
                continue
            iou = bbox_iou_xyxy(bbox, entry["bbox"])
            if iou < self.spatial_iou_thresh:
                continue
            sim = cosine_sim(emb, entry["emb"])
            score = 0.5 * sim + 0.5 * iou
            if sim >= self.reentry_thresh and score > best_score:
                best_score = score
                best_gid = gid
        return best_gid

    def _pick_gallery_match(self, obs, emb, candidates: list[int], busy: set[int]):
        """Best GID from gallery with spatial gate and ambiguity margin."""
        bbox = obs["bbox"]
        scored = []
        for gid in candidates:
            if gid in busy:
                continue
            if not self._spatial_ok(gid, bbox):
                continue
            sim = self._best_sim(emb, gid)
            if sim >= self.reentry_thresh:
                scored.append((sim, gid))
        if not scored:
            return None
        scored.sort(reverse=True)
        best_sim, best_gid = scored[0]
        if len(scored) > 1 and (best_sim - scored[1][0]) < self.match_margin:
            return None
        return best_gid

    def assign(self, observations):
        self._frame += 1
        self._prune_lost()
        gid_map: dict[tuple, int] = {}
        used_gids: set[int] = set()
        new_obs = []

        for obs in observations:
            key = (obs["cam_id"], int(obs["track_id"]))
            emb = np.asarray(obs["emb"], dtype=np.float32).ravel()
            bbox = obs["bbox"]
            sticky = self._track_to_gid.get(key)
            if sticky is not None:
                gid = sticky["gid"]
                sticky["last_seen"] = self._frame
                self._update_view(gid, emb, obs["cam_id"], bbox)
                gid_map[key] = gid
                used_gids.add(gid)
            else:
                new_obs.append((obs, emb))

        busy_by_cam: dict[str, set[int]] = {}

        still_new = []
        for obs, emb in new_obs:
            key = (obs["cam_id"], int(obs["track_id"]))
            cam = obs["cam_id"]
            busy = busy_by_cam.setdefault(cam, self._busy_gids(cam))

            gid = self._match_lost_track(obs, emb, busy)
            if gid is None:
                candidates = [
                    g for g in self._gallery if g not in used_gids and self._eligible(g)
                ]
                gid = self._pick_gallery_match(obs, emb, candidates, busy)

            if gid is not None:
                self._bind(key, gid, emb, cam, obs["bbox"])
                gid_map[key] = gid
                used_gids.add(gid)
                busy.add(gid)
            else:
                still_new.append((obs, emb))

        candidates = [g for g in self._gallery if g not in used_gids and self._eligible(g)]
        unassigned = set(range(len(still_new)))

        if still_new and candidates:
            sim = np.full((len(still_new), len(candidates)), -1.0, dtype=np.float32)
            for i, (obs, emb) in enumerate(still_new):
                busy = busy_by_cam.setdefault(obs["cam_id"], self._busy_gids(obs["cam_id"]))
                for j, gid in enumerate(candidates):
                    if gid in busy:
                        continue
                    if not self._spatial_ok(gid, obs["bbox"]):
                        continue
                    sim[i, j] = self._best_sim(emb, gid)

            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(-sim)
            except ImportError:
                row_ind, col_ind = self._greedy(sim)

            for r, c in zip(row_ind, col_ind):
                if r not in unassigned or c >= len(candidates):
                    continue
                if sim[r, c] < self.reentry_thresh:
                    continue
                gid = candidates[c]
                obs, emb = still_new[r]
                key = (obs["cam_id"], int(obs["track_id"]))
                cam = obs["cam_id"]
                self._bind(key, gid, emb, cam, obs["bbox"])
                gid_map[key] = gid
                used_gids.add(gid)
                busy_by_cam.setdefault(cam, set()).add(gid)
                unassigned.discard(r)

        for r in sorted(unassigned):
            obs, emb = still_new[r]
            key = (obs["cam_id"], int(obs["track_id"]))
            cam = obs["cam_id"]
            busy = busy_by_cam.setdefault(cam, self._busy_gids(cam))

            best_gid, best_sim = None, -1.0
            for gid in self._gallery:
                if not self._eligible(gid) or gid in busy:
                    continue
                if not self._spatial_ok(gid, obs["bbox"]):
                    continue
                sim = self._best_sim(emb, gid)
                if sim > best_sim:
                    best_sim, best_gid = sim, gid

            if best_gid is not None and best_sim >= self.new_gid_thresh:
                self._bind(key, best_gid, emb, cam, obs["bbox"])
                gid_map[key] = best_gid
                used_gids.add(best_gid)
                busy.add(best_gid)
                continue

            gid = self._next_gid
            self._next_gid += 1
            self._gallery[gid] = {
                "views_by_cam": {cam: emb.copy()},
                "last_seen": self._frame,
            }
            self._bind(key, gid, emb, cam, obs["bbox"])
            gid_map[key] = gid
            used_gids.add(gid)

        # Push lost local tracks into buffer before dropping sticky map
        stale_keys = [
            k
            for k, v in self._track_to_gid.items()
            if self._frame - v["last_seen"] > self.stale_frames
        ]
        for k in stale_keys:
            meta = self._track_to_gid.pop(k)
            last = self._track_last.pop(k, None)
            if last is None:
                continue
            self._lost.append(
                {
                    "cam_id": k[0],
                    "track_id": k[1],
                    "gid": meta["gid"],
                    "emb": last["emb"],
                    "bbox": last["bbox"],
                    "lost_frame": self._frame,
                }
            )

        for gid, g in list(self._gallery.items()):
            if self._frame - g["last_seen"] > self.retire_frames:
                self._retire(gid)

        return gid_map


class ReidFeatureCache:
    """Re-ID features for global GID assignment."""

    def __init__(self, reid_model, every_n: int = 1):
        self._reid = reid_model
        self._every_n = max(1, int(every_n))
        self._cache: dict[tuple, tuple[np.ndarray, int]] = {}

    def observations(self, tracks, frame, frame_index, cam_id):
        if not tracks:
            return []
        need = [
            i
            for i, tr in enumerate(tracks)
            if (cam_id, int(tr[4])) not in self._cache
            or frame_index - self._cache[(cam_id, int(tr[4]))][1] >= self._every_n
        ]
        if need:
            xyxy = np.asarray([tracks[i][:4] for i in need], dtype=np.float32)
            embs = np.atleast_2d(
                np.asarray(self._reid.get_features(xyxy, frame), dtype=np.float32)
            )
            for j, i in enumerate(need):
                key = (cam_id, int(tracks[i][4]))
                self._cache[key] = (embs[j].copy(), int(frame_index))

        active = {(cam_id, int(tr[4])) for tr in tracks}
        for k in [k for k in self._cache if k not in active]:
            del self._cache[k]

        out = []
        for tr in tracks:
            key = (cam_id, int(tr[4]))
            if key not in self._cache:
                continue
            emb, _ = self._cache[key]
            out.append(
                {
                    "cam_id": cam_id,
                    "track_id": int(tr[4]),
                    "bbox": tr[:4],
                    "emb": emb,
                }
            )
        return out
