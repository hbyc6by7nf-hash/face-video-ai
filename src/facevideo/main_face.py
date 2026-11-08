"""Protagonist face replacement: INSwapper + GFPGAN with temporal stabilisation.

Pass 1 (analyse): detect faces in every frame, pick the protagonist by ArcFace
identity (so figurines / background faces are never swapped), fill short
detection gaps (motion blur) and smooth the landmarks with a zero-phase filter
inside each shot. This removes the frame-to-frame jitter that causes flicker.

Pass 2 (render): align -> INSwapper 128 -> GFPGAN 512 -> keep low-frequency
lighting of the swap -> match skin tone/lighting to the original frame -> blend
through a BiSeNet face mask, so hands / objects in front of the face stay on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import align
from .models import FACE_CLASSES, SKIN_CLASS, ModelBank


@dataclass
class MainFaceSettings:
    similarity_threshold: float = 0.30     # ArcFace cosine to accept a face as the protagonist
    min_face_fraction: float = 0.004       # ignore faces smaller than this share of the frame
    max_gap_frames: int = 6                # interpolate detections over gaps up to this length
    smoothing_sigma: float = 1.2           # landmark smoothing, in frames (0 = off)
    enhance: bool = True                   # GFPGAN restoration
    enhance_blend: float = 0.8
    skin_match: float = 0.55               # 0 = keep source photo skin tone, 1 = full scene match
    mask_ema: float = 0.5                  # temporal smoothing of the blend mask (0 = off)
    mouth_inside: bool = True              # keep mouth interior from the swap


@dataclass
class FaceTrack:
    kps: np.ndarray                         # (N, 5, 2) smoothed landmarks
    valid: np.ndarray                       # (N,) bool
    segment: np.ndarray                     # (N,) shot index
    raw_detected: np.ndarray = field(default=None)   # (N,) bool — detected (not interpolated)
    identity: np.ndarray | None = None


def source_identity(models: ModelBank, photo: np.ndarray) -> np.ndarray:
    faces = models.detector.detect(photo)
    if not faces:
        raise RuntimeError("На вашей фотографии не найдено лицо. Нужен чёткий портрет, лицо целиком.")
    return models.embedder.embed(photo, faces[0].kps)


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    flat = pad.reshape(len(pad), -1)
    out = np.stack([np.convolve(flat[:, j], k, mode="valid") for j in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def build_track(detections: list[list], segments: np.ndarray, frame_area: float,
                cfg: MainFaceSettings) -> FaceTrack:
    """detections[i] = list of (Face with embedding). Picks the protagonist per frame."""
    n = len(detections)
    big = [[f for f in fs if f.area >= cfg.min_face_fraction * frame_area] for fs in detections]

    # Identity of the protagonist = centroid of the largest face per frame, refined twice.
    seeds = [fs[0].embedding for fs in big if fs]
    if not seeds:
        return FaceTrack(np.zeros((n, 5, 2), np.float32), np.zeros(n, bool), segments, np.zeros(n, bool))
    centroid = np.mean(seeds, 0)
    for _ in range(2):
        centroid /= np.linalg.norm(centroid)
        chosen = [max(fs, key=lambda f: f.embedding @ centroid) for fs in big if fs]
        chosen = [f for f in chosen if f.embedding @ centroid >= cfg.similarity_threshold]
        if chosen:
            centroid = np.mean([f.embedding for f in chosen], 0)
    centroid /= np.linalg.norm(centroid)

    kps = np.zeros((n, 5, 2), np.float32)
    valid = np.zeros(n, bool)
    for i, fs in enumerate(big):
        if not fs:
            continue
        best = max(fs, key=lambda f: f.embedding @ centroid)
        if best.embedding @ centroid >= cfg.similarity_threshold:
            kps[i], valid[i] = best.kps, True
    detected = valid.copy()

    # Fill short gaps inside a shot by linear interpolation.
    idx = np.where(valid)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= cfg.max_gap_frames + 1 and segments[a] == segments[b]:
            for j in range(a + 1, b):
                u = (j - a) / (b - a)
                kps[j] = kps[a] * (1 - u) + kps[b] * u
                valid[j] = True

    # Zero-phase smoothing within each continuous run (never across shot cuts).
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and valid[j + 1] and segments[j + 1] == segments[i]:
            j += 1
        # adapt smoothing to the face size: small faces jitter more in relative terms
        kps[i:j + 1] = _gaussian_smooth(kps[i:j + 1], cfg.smoothing_sigma)
        i = j + 1
    return FaceTrack(kps, valid, segments, detected, centroid)


class MainFaceRenderer:
    def __init__(self, models: ModelBank, source_embedding: np.ndarray, cfg: MainFaceSettings):
        self.m = models
        self.cfg = cfg
        self.latent = models.swapper.latent(source_embedding)
        self.valid128 = align.feather(np.pad(np.ones((120, 120), np.float32), 4), 2, 2.5)
        self.prev_mask = None
        self.prev_segment = None

    def reset(self) -> None:
        self.prev_mask = None

    def render(self, frame: np.ndarray, kps: np.ndarray, segment: int) -> np.ndarray:
        cfg = self.cfg
        if segment != self.prev_segment:
            self.reset()
            self.prev_segment = segment

        M128 = align.estimate(kps, align.ARCFACE_128)
        M512 = align.estimate(kps, align.FFHQ_512)
        crop128 = align.warp(frame, M128, 128)
        swapped128 = self.m.swapper.swap(crop128, self.latent)

        # bring the 128px swap into the 512px FFHQ crop of the original frame
        A = align.compose(M512, cv2.invertAffineTransform(M128))
        orig512 = align.warp(frame, M512, 512)
        up = cv2.warpAffine(swapped128, A, (512, 512), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(self.valid128, A, (512, 512), borderValue=0)[:, :, None]
        swapped512 = (up * valid + orig512 * (1 - valid)).astype(np.uint8)

        result = swapped512
        if cfg.enhance:
            enh = self.m.enhancer.enhance(swapped512)
            # keep GFPGAN's detail but the swap's (scene) lighting: low-frequency transfer
            ef, sf = enh.astype(np.float32), swapped512.astype(np.float32)
            low_e = cv2.GaussianBlur(ef, (0, 0), 6)
            low_s = cv2.GaussianBlur(sf, (0, 0), 6)
            detail = ef - low_e + low_s
            w = cfg.enhance_blend
            result = np.clip(detail * w + sf * (1 - w), 0, 255).astype(np.uint8)

        # face mask from the ORIGINAL frame -> occluders (hands, gun, knife...) stay in front
        parse = self.m.parser.parse(orig512)
        classes = list(FACE_CLASSES)
        if not cfg.mouth_inside:
            classes.remove(11)
        face = np.isin(parse, classes).astype(np.float32)
        skin = (parse == SKIN_CLASS).astype(np.float32)

        if cfg.skin_match > 0 and skin.sum() > 500:
            sm, ss = align.lab_stats(result, skin)
            dm, ds = align.lab_stats(orig512, skin)
            result = align.color_transfer(result, sm, ss, dm, ds, cfg.skin_match, match_std=False)

        mask = align.feather(face, 512 * 0.012, 512 * 0.018) * valid[:, :, 0]
        if self.prev_mask is not None and cfg.mask_ema > 0:
            mask = mask * (1 - cfg.mask_ema) + self.prev_mask * cfg.mask_ema
        self.prev_mask = mask
        return align.paste(frame, result, mask, M512)
