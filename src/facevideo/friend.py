"""Friend's head in place of the background robot helmet.

HONEST NOTE: the robot has no human face, so no face swapper can "see" it.
Here the friend's head is cut out of the photo with a neural face parser
(BiSeNet: skin, hair, hat), then

* moved with the helmet using optical-flow tracking (KLT + RANSAC similarity),
  so it follows translation, scale and roll of the helmet frame by frame;
* relit: skin colour is matched to the protagonist's skin in the same shot,
  a darkening factor is applied, and the low-frequency light/shadow of the
  helmet region is transferred onto the head every frame;
* defocused to the background depth and feather-blended.

This is a tracked 2D composite, not a generative head replacement: the head
does not turn (yaw/pitch) or speak like a real person. For that, see
"Higher realism" in README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import align
from .models import EAR_CLASSES, FACE_CLASSES, HAIR_CLASS, HAT_CLASS, SKIN_CLASS, ModelBank


@dataclass
class FriendSettings:
    enabled: bool = True
    anchor_time: float = 2.5          # seconds; frame where the box below is defined
    box: tuple = (0.112, 0.295, 0.195, 0.52)   # x, y, w, h as fractions of the frame
    start_time: float = 1.0           # tracking window
    end_time: float = 3.7
    brightness: float = 0.80          # extra darkening (background is in shade)
    skin_match: float = 0.85
    shading: float = 0.45             # how much helmet light/shadow is transferred
    defocus: float = 1.0              # blur sigma at 720p
    angle: float = -6.0               # extra roll of the head, degrees
    scale: float = 1.0                # head size relative to the box height
    fade_frames: int = 4
    smoothing_sigma: float = 1.5


@dataclass
class HeadCutout:
    rgb: np.ndarray        # (S, S, 3) BGR
    alpha: np.ndarray      # (S, S) float
    skin: np.ndarray       # (S, S) float, skin-only mask for colour statistics
    bbox: tuple            # alpha bounding box x1, y1, x2, y2 inside the cutout


def build_head(models: ModelBank, photo: np.ndarray, size: int = 512) -> HeadCutout:
    faces = models.detector.detect(photo)
    if not faces:
        raise RuntimeError("На фото друга не найдено лицо. Нужен чёткий портрет анфас.")
    kps = faces[0].kps
    # loose FFHQ crop so hair / hat fit in the frame
    M = align.estimate(kps, align.scaled_template(align.FFHQ_512, 512, 0.85))
    crop = align.warp(photo, M, 512, border=cv2.BORDER_CONSTANT)
    parse = models.parser.parse(crop)
    head = np.isin(parse, list(FACE_CLASSES) + list(EAR_CLASSES) + [HAIR_CLASS, HAT_CLASS])
    head = head.astype(np.uint8)
    # keep the biggest connected blob, fill holes
    n, lab, stats, _ = cv2.connectedComponentsWithStats(head)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        head = (lab == biggest).astype(np.uint8)
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    alpha = align.feather(head, 2, 2.0)
    ys, xs = np.where(head > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    skin = (parse == SKIN_CLASS).astype(np.float32)
    return HeadCutout(crop, alpha, skin, bbox)


# ---------------------------------------------------------------- tracking

def _box_px(box, w, h):
    x, y, bw, bh = box
    return np.array([x * w, y * h, bw * w, bh * h], np.float32)


def _klt_step(prev_gray, gray, pts):
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, winSize=(21, 21), maxLevel=3)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
    return nxt, good


def _features(gray, M_anchor_to_frame, box):
    """Good features inside the (transformed) helmet box."""
    x, y, bw, bh = box
    poly = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.float32)
    poly = cv2.transform(poly[None], M_anchor_to_frame)[0].astype(np.int32)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    return cv2.goodFeaturesToTrack(gray, 300, 0.01, 4, mask=mask, blockSize=5)


def track_helmet(grays: dict[int, np.ndarray], anchor: int, lo: int, hi: int,
                 box_px: np.ndarray, cuts: set[int]) -> dict[int, np.ndarray]:
    """Returns {frame index: 2x3 similarity transform anchor-frame -> frame}."""
    eye = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    result = {anchor: eye}
    for direction in (1, -1):
        M = eye.copy()
        pts_anchor = _features(grays[anchor], eye, box_px)
        if pts_anchor is None:
            continue
        cur_pts, ref_pts = pts_anchor.copy(), pts_anchor.copy()   # ref = same points in anchor coords
        i = anchor
        while True:
            j = i + direction
            if j < lo or j > hi or j not in grays:
                break
            if (direction == 1 and j in cuts) or (direction == -1 and i in cuts):
                break                                   # shot cut: the helmet is gone
            nxt, good = _klt_step(grays[i], grays[j], cur_pts)
            if good.sum() < 8:
                break
            cur_pts, ref_pts = nxt[good], ref_pts[good]
            M_new, inl = cv2.estimateAffinePartial2D(ref_pts, cur_pts, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.0, maxIters=2000)
            if M_new is None or inl.sum() < 6:
                break
            M = M_new.astype(np.float32)
            result[j] = M
            i = j
            if len(cur_pts) < 60:                       # replenish features
                fresh = _features(grays[j], M, box_px)
                if fresh is not None:
                    IM = cv2.invertAffineTransform(M)
                    cur_pts = np.concatenate([cur_pts, fresh])
                    ref_pts = np.concatenate([ref_pts, cv2.transform(fresh, IM)])
    return result


def _decompose(M):
    s = float(np.hypot(M[0, 0], M[1, 0]))
    a = float(np.arctan2(M[1, 0], M[0, 0]))
    return np.array([s, a, M[0, 2], M[1, 2]], np.float64)


def _recompose(p):
    s, a, tx, ty = p
    c, d = s * np.cos(a), s * np.sin(a)
    return np.array([[c, -d, tx], [d, c, ty]], np.float32)


def smooth_transforms(tr: dict[int, np.ndarray], sigma: float, box_px) -> dict[int, np.ndarray]:
    """Smooth the motion of the box centre (not raw matrix entries) over time."""
    if sigma <= 0 or len(tr) < 3:
        return tr
    from .main_face import _gaussian_smooth
    keys = sorted(tr)
    cx, cy = box_px[0] + box_px[2] / 2, box_px[1] + box_px[3] / 2
    params = []
    for k in keys:
        s, a, _, _ = _decompose(tr[k])
        center = tr[k] @ np.array([cx, cy, 1.0])
        params.append([s, a, center[0], center[1]])
    params = _gaussian_smooth(np.array(params), sigma)
    out = {}
    for k, (s, a, ccx, ccy) in zip(keys, params):
        M = _recompose((s, a, 0, 0))
        M[:, 2] = np.array([ccx, ccy]) - M[:, :2] @ np.array([cx, cy])
        out[k] = M
    return out


# ---------------------------------------------------------------- compositing

class FriendCompositor:
    def __init__(self, head: HeadCutout, cfg: FriendSettings, frame_size: tuple[int, int],
                 transforms: dict[int, np.ndarray], scene_skin: tuple | None, cut_frames: set[int]):
        self.head = head
        self.cfg = cfg
        W, H = frame_size
        self.W, self.H = W, H
        box = _box_px(cfg.box, W, H)
        # place the head so its bbox height = box height, centred on the box
        x1, y1, x2, y2 = head.bbox
        s = cfg.scale * box[3] / (y2 - y1)
        hc = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        bc = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
        R = cv2.getRotationMatrix2D((float(hc[0]), float(hc[1])), cfg.angle, s)
        R[:, 2] += bc - hc
        self.place = R.astype(np.float32)          # cutout -> anchor frame
        self.transforms = transforms
        keys = sorted(transforms)
        self.first, self.last = (keys[0], keys[-1]) if keys else (0, -1)
        self.fade_in = self.first not in cut_frames
        self.fade_out = (self.last + 1) not in cut_frames

        rgb = head.rgb
        if scene_skin is not None and head.skin.sum() > 200:
            sm, ss = align.lab_stats(rgb, head.skin)
            dm, ds = scene_skin
            rgb = align.color_transfer(rgb, sm, ss, dm, ds, cfg.skin_match)
        rgb = np.clip(rgb.astype(np.float32) * cfg.brightness, 0, 255).astype(np.uint8)
        self.rgb = rgb

    def opacity(self, i: int) -> float:
        if i not in self.transforms:
            return 0.0
        f = max(1, self.cfg.fade_frames)
        o = 1.0
        if self.fade_in:
            o = min(o, (i - self.first + 1) / f)
        if self.fade_out:
            o = min(o, (self.last - i + 1) / f)
        return float(np.clip(o, 0, 1))

    def render(self, frame: np.ndarray, i: int) -> np.ndarray:
        op = self.opacity(i)
        if op <= 0:
            return frame
        M = align.compose(self.transforms[i], self.place)       # cutout -> frame i
        H, W = frame.shape[:2]
        S = self.rgb.shape[0]
        corners = cv2.transform(np.array([[[0, 0], [S, 0], [0, S], [S, S]]], np.float32), M)[0]
        x1, y1 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
        x2, y2 = np.minimum(np.ceil(corners.max(0)).astype(int), [W, H])
        if x1 >= x2 or y1 >= y2:
            return frame
        Mr = M.copy()
        Mr[:, 2] -= (x1, y1)
        size = (int(x2 - x1), int(y2 - y1))
        fg = cv2.warpAffine(self.rgb, Mr, size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.warpAffine(self.head.alpha, Mr, size, flags=cv2.INTER_LINEAR, borderValue=0)
        roi = frame[y1:y2, x1:x2]

        fgf = fg.astype(np.float32)
        # transfer the helmet's local light / shadow (low frequency luminance) onto the head
        if self.cfg.shading > 0:
            lum = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            low = cv2.GaussianBlur(lum, (0, 0), max(4.0, min(size) * 0.12))
            ref = float((low * a).sum() / (a.sum() + 1e-6)) + 1e-3
            gain = np.clip(low / ref, 0.4, 1.8) ** self.cfg.shading
            fgf *= gain[:, :, None]
        sigma = self.cfg.defocus * H / 720.0
        if sigma > 0.2:
            fgf = cv2.GaussianBlur(fgf, (0, 0), sigma)
            a = cv2.GaussianBlur(a, (0, 0), sigma * 0.7)
        # light grain so the head has the same noise as the footage
        fgf += np.random.default_rng(i).normal(0, 1.6, fgf.shape).astype(np.float32)
        a = (a * op)[:, :, None]
        frame[y1:y2, x1:x2] = np.clip(roi * (1 - a) + fgf * a + 0.5, 0, 255).astype(np.uint8)
        return frame


def draw_box(frame: np.ndarray, cfg: FriendSettings) -> np.ndarray:
    H, W = frame.shape[:2]
    x, y, bw, bh = _box_px(cfg.box, W, H).astype(int)
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
    cv2.putText(out, "friend box", (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out
