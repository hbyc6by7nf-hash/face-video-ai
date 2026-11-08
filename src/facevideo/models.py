"""Thin ONNX Runtime wrappers around the face models used by the pipeline.

All models run through onnxruntime directly, so the (hard to build on new
Python versions) `insightface` package is not required.

    SCRFD (det_10g)      face detection + 5 landmarks        InsightFace, non-commercial
    ArcFace (w600k_r50)  identity embedding                  InsightFace, non-commercial
    INSwapper 128        identity transfer onto a target     InsightFace, non-commercial
    GFPGAN 1.4           face restoration 128px -> 512px     Apache-2.0
    BiSeNet (ResNet-34)  face parsing / occlusion mask       MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from . import align

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

REQUIRED_MODELS = {
    "det_10g.onnx": "детектор лиц SCRFD (из buffalo_l)",
    "w600k_r50.onnx": "ArcFace — эмбеддинг личности (из buffalo_l)",
    "inswapper_128.onnx": "INSwapper — нейросетевая замена лица",
    "gfpgan_1.4.onnx": "GFPGAN — восстановление деталей лица",
    "bisenet_resnet_34.onnx": "BiSeNet — сегментация лица и перекрытий",
}

# CelebAMask-HQ classes produced by BiSeNet
FACE_CLASSES = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)   # skin, brows, eyes, glasses, nose, mouth, lips
SKIN_CLASS = 1
EAR_CLASSES = (7, 8, 9)
HAIR_CLASS = 17
HAT_CLASS = 18


def missing_models(model_dir: Path = MODEL_DIR) -> list[str]:
    return [name for name in REQUIRED_MODELS if not (model_dir / name).is_file()]


def pick_providers(device: str = "auto") -> list[str]:
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    wanted = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"]
    chosen = [p for p in wanted if p in available]
    if device == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError("CUDA недоступна: установите onnxruntime-gpu и драйверы NVIDIA")
    return chosen + ["CPUExecutionProvider"]


def _session(path: Path, providers: list[str]) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет модели {path.name}. Запустите: python scripts/download_models.py --accept-noncommercial"
        )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


@dataclass
class Face:
    bbox: np.ndarray       # x1, y1, x2, y2
    kps: np.ndarray        # (5, 2): left eye, right eye, nose, left mouth, right mouth
    score: float
    embedding: np.ndarray | None = None

    @property
    def area(self) -> float:
        return float(max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1]))


class FaceDetector:
    """SCRFD detector (det_10g.onnx from InsightFace buffalo_l)."""

    strides = (8, 16, 32)

    def __init__(self, path: Path, providers: list[str], size: int = 640, threshold: float = 0.5):
        self.sess = _session(path, providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size
        self.threshold = threshold
        self._centers: dict[int, np.ndarray] = {}

    def _anchor_centers(self, stride: int) -> np.ndarray:
        if stride not in self._centers:
            n = self.size // stride
            yy, xx = np.mgrid[:n, :n]
            centers = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            self._centers[stride] = np.repeat(centers, 2, axis=0)   # 2 anchors per location
        return self._centers[stride]

    def detect(self, image: np.ndarray, max_faces: int = 0) -> list[Face]:
        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (self.size, self.size),
                                     (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.input_name: blob})
        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(self.strides):
            sc = outs[i].reshape(-1)
            bb = outs[i + 3].reshape(-1, 4) * stride
            kp = outs[i + 6].reshape(-1, 10) * stride
            keep = np.where(sc >= self.threshold)[0]
            if not len(keep):
                continue
            c = self._anchor_centers(stride)[keep]
            bb, kp = bb[keep], kp[keep]
            boxes.append(np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1],
                                   c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], axis=1))
            kpss.append(np.stack([c[:, 0:1] + kp[:, 0::2], c[:, 1:2] + kp[:, 1::2]], axis=2))
            scores.append(sc[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        idx = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.threshold, 0.4)
        idx = np.array(idx).reshape(-1)
        faces = [Face(boxes[i], kpss[i], float(scores[i])) for i in idx]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces[:max_faces] if max_faces else faces


class FaceEmbedder:
    """ArcFace w600k_r50 — 512-d identity embedding (L2-normalised)."""

    def __init__(self, path: Path, providers: list[str]):
        self.sess = _session(path, providers)
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, image: np.ndarray, kps: np.ndarray) -> np.ndarray:
        M = align.estimate(kps, align.ARCFACE_112)
        crop = cv2.warpAffine(image, M, (112, 112), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.sess.run(None, {self.input_name: blob})[0].reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-8)


class FaceSwapper:
    """INSwapper 128: replaces the identity of an aligned 128x128 face crop."""

    def __init__(self, path: Path, providers: list[str]):
        import onnx
        from onnx import numpy_helper

        graph = onnx.load(str(path)).graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.sess = _session(path, providers)
        names = [i.name for i in self.sess.get_inputs()]
        self.target_name, self.source_name = names[0], names[1]

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        lat = embedding.reshape(1, -1) @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def swap(self, crop128: np.ndarray, latent: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(crop128, 1 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        out = self.sess.run(None, {self.target_name: blob, self.source_name: latent})[0][0]
        out = np.clip(out.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        return out[:, :, ::-1].copy()


class FaceEnhancer:
    """GFPGAN 1.4 on an FFHQ-aligned 512x512 crop."""

    def __init__(self, path: Path, providers: list[str]):
        self.sess = _session(path, providers)
        self.input_name = self.sess.get_inputs()[0].name

    def enhance(self, crop512: np.ndarray) -> np.ndarray:
        x = crop512[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        y = self.sess.run(None, {self.input_name: x})[0][0]
        y = np.clip((y.transpose(1, 2, 0) + 1) / 2, 0, 1) * 255.0
        return y[:, :, ::-1].round().astype(np.uint8)


class FaceParser:
    """BiSeNet face parsing (19 CelebAMask-HQ classes) on a 512x512 crop."""

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, path: Path, providers: list[str]):
        self.sess = _session(path, providers)
        self.input_name = self.sess.get_inputs()[0].name

    def parse(self, crop512: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop512, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        y = self.sess.run(None, {self.input_name: x})[0]
        return y.reshape(y.shape[-3], 512, 512).argmax(0).astype(np.uint8)


class ModelBank:
    """Lazily loads only the models a run actually needs."""

    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "auto"):
        self.dir = Path(model_dir)
        self.providers = pick_providers(device)
        self._cache: dict[str, object] = {}

    def _get(self, key, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    @property
    def detector(self) -> FaceDetector:
        return self._get("det", lambda: FaceDetector(self.dir / "det_10g.onnx", self.providers))

    @property
    def embedder(self) -> FaceEmbedder:
        return self._get("emb", lambda: FaceEmbedder(self.dir / "w600k_r50.onnx", self.providers))

    @property
    def swapper(self) -> FaceSwapper:
        return self._get("swap", lambda: FaceSwapper(self.dir / "inswapper_128.onnx", self.providers))

    @property
    def enhancer(self) -> FaceEnhancer:
        return self._get("enh", lambda: FaceEnhancer(self.dir / "gfpgan_1.4.onnx", self.providers))

    @property
    def parser(self) -> FaceParser:
        return self._get("parse", lambda: FaceParser(self.dir / "bisenet_resnet_34.onnx", self.providers))
