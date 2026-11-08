"""Face alignment templates, similarity transforms and blending helpers."""

from __future__ import annotations

import cv2
import numpy as np

# 5-point templates (left eye, right eye, nose tip, left/right mouth corner)
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
ARCFACE_128 = ARCFACE_112 + np.array([8.0, 0.0], np.float32)          # inswapper crop
FFHQ_512 = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813],
                     [0.50123859, 0.61331904], [0.39308822, 0.72541100],
                     [0.61150205, 0.72490465]], np.float32) * 512        # GFPGAN / BiSeNet crop


def scaled_template(template: np.ndarray, size: int, zoom: float) -> np.ndarray:
    """Template shrunk around the crop centre: zoom < 1 leaves more context around the face."""
    c = np.array([size / 2, size / 2], np.float32)
    return (template - c) * zoom + c


def estimate(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping src points onto dst; 2x3 matrix."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var = (s0 ** 2).sum() / len(src)
    scale = (S * np.diag(D)).sum() / var
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Affine composition: returns A∘B (apply B first)."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2].astype(np.float32)


def warp(image: np.ndarray, M: np.ndarray, size: int, border=cv2.BORDER_REPLICATE) -> np.ndarray:
    return cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=border)


def paste(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Alpha-blend `crop` (aligned with frame->crop transform M) back into `frame` in place.

    Works only inside the bounding box of the crop to keep it fast on large frames.
    """
    H, W = frame.shape[:2]
    h, w = crop.shape[:2]
    IM = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ IM.T
    x1, y1 = np.floor(corners.min(0)).astype(int)
    x2, y2 = np.ceil(corners.max(0)).astype(int)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x1 >= x2 or y1 >= y2:
        return frame
    IMr = IM.copy()
    IMr[:, 2] -= (x1, y1)
    size = (x2 - x1, y2 - y1)
    fg = cv2.warpAffine(crop, IMr, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(mask.astype(np.float32), IMr, size, flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)[:, :, None]
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = np.clip(roi * (1 - a) + fg.astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)
    return frame


def feather(mask: np.ndarray, erode_px: float, blur_px: float) -> np.ndarray:
    m = mask.astype(np.float32)
    if erode_px >= 1:
        k = int(erode_px) * 2 + 1
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if blur_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur_px)
    return np.clip(m, 0, 1)


def box_mask(size: int, pad: float, blur: float) -> np.ndarray:
    m = np.zeros((size, size), np.float32)
    p = int(size * pad)
    m[p:size - p, p:size - p] = 1
    return cv2.GaussianBlur(m, (0, 0), size * blur)


def lab_stats(image: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    w = weight.astype(np.float32)[:, :, None]
    s = w.sum() + 1e-6
    mean = (lab * w).sum((0, 1)) / s
    std = np.sqrt((((lab - mean) ** 2) * w).sum((0, 1)) / s) + 1e-3
    return mean, std


def color_transfer(image: np.ndarray, src_mean, src_std, dst_mean, dst_std,
                   strength: float = 1.0, match_std: bool = True) -> np.ndarray:
    """Reinhard colour transfer in Lab space with adjustable strength."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    if match_std:
        ratio = np.clip(dst_std / src_std, 0.5, 2.0)
        out = (lab - src_mean) * ratio + dst_mean
    else:
        out = lab - src_mean + dst_mean
    out = lab + (out - lab) * strength
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
