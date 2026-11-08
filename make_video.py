#!/usr/bin/env python3
"""Face replacement and moving helmet portrait overlay.

Main subject: InsightFace/INSwapper face transfer (optional with --skip-main-face).
Robot in the background: an animated, feathered cutout of friend's portrait.
This second effect is 2D compositing, NOT photorealistic generative replacement.
Only process likenesses with the people's permission.
"""

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

SETTINGS = {
  "template_width": 908,
  "template_height": 512,
  "friend_mode": "helmet_face",
  "friend_brightness": 0.85,
  "friend_saturation": 0.85,
  "friend_blur": 0.5,
  "friend_angle_degrees": -7.0,
  "friend_keyframes": [
    {"t": 0.00, "cx": 185, "cy": 270, "w": 132, "h": 168, "opacity": 0.00},
    {"t": 1.25, "cx": 185, "cy": 270, "w": 132, "h": 168, "opacity": 0.00},
    {"t": 1.65, "cx": 182, "cy": 285, "w": 137, "h": 174, "opacity": 0.92},
    {"t": 2.25, "cx": 187, "cy": 281, "w": 135, "h": 172, "opacity": 0.96},
    {"t": 2.80, "cx": 185, "cy": 277, "w": 132, "h": 167, "opacity": 0.96},
    {"t": 3.30, "cx": 186, "cy": 273, "w": 126, "h": 161, "opacity": 0.91},
    {"t": 3.55, "cx": 172, "cy": 282, "w": 121, "h": 158, "opacity": 0.84},
    {"t": 3.85, "cx": 140, "cy": 297, "w": 108, "h": 153, "opacity": 0.12},
    {"t": 4.00, "cx": 140, "cy": 297, "w": 108, "h": 153, "opacity": 0.00},
    {"t": 11.03, "cx": 140, "cy": 297, "w": 108, "h": 153, "opacity": 0.00}
  ]
}




def load_photo(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Не найдено {label}: {path}. Добавьте нужную фотографию в input/.")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Не удалось открыть {label}: {path}")
    return frame


def detect_photo_face(photo: np.ndarray) -> tuple[int, int, int, int]:
    """Built-in Haar detector; deliberately independent of InsightFace downloads."""
    cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(cascade_file)
    if clf.empty():
        raise RuntimeError("OpenCV face detector is missing")
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    boxes = clf.detectMultiScale(gray, 1.08, 5, minSize=(35, 35))
    if len(boxes):
        return max(boxes, key=lambda b: int(b[2]) * int(b[3]))
    # Photo should be a single headshot; center crop is a predictable fallback.
    h, w = photo.shape[:2]
    side = int(min(w, h) * 0.54)
    print("Внимание: лицо друга не обнаружено; берём центральную часть фото.", file=sys.stderr)
    return ((w - side) // 2, (h - side) // 2, side, side)


def extract_friend_portrait(photo: np.ndarray) -> np.ndarray:
    x, y, fw, fh = detect_photo_face(photo)
    h, w = photo.shape[:2]
    cx = x + fw / 2
    cy = y + fh * 0.47
    crop_w = fw * 1.40
    crop_h = fh * 1.54
    # Subpixel extraction with replicated boundaries handles faces near a photo edge.
    padded = cv2.copyMakeBorder(photo, h, h, w, w, cv2.BORDER_REPLICATE)
    return cv2.getRectSubPix(
        padded, (max(2, int(crop_w)), max(2, int(crop_h))), (cx + w, cy + h)
    )


def interpolate_keyframe(keyframes: list[dict], t: float) -> dict:
    if t <= keyframes[0]["t"]:
        return dict(keyframes[0])
    for right in range(1, len(keyframes)):
        b = keyframes[right]
        if t <= b["t"]:
            a = keyframes[right - 1]
            dt = b["t"] - a["t"]
            u = 0.0 if dt <= 0 else (t - a["t"]) / dt
            # Smooth interpolation is less jerky than abruptly moving the face crop.
            u = u * u * (3 - 2 * u)
            return {k: float(a[k]) + (float(b[k]) - float(a[k])) * u
                    for k in ("cx", "cy", "w", "h", "opacity")}
    return dict(keyframes[-1])


def place_friend_face(frame: np.ndarray, portrait: np.ndarray, config: dict, t: float) -> np.ndarray:
    pos = interpolate_keyframe(config["friend_keyframes"], t)
    opacity = float(np.clip(pos["opacity"], 0, 1))
    if opacity < 0.001:
        return frame
    fh, fw = frame.shape[:2]
    scale_x = fw / config["template_width"]
    scale_y = fh / config["template_height"]
    cx, cy = int(round(pos["cx"] * scale_x)), int(round(pos["cy"] * scale_y))
    ow = max(4, int(round(pos["w"] * scale_x)))
    oh = max(4, int(round(pos["h"] * scale_y)))

    patch = cv2.resize(portrait, (ow, oh), interpolation=cv2.INTER_AREA)
    patch = patch.astype(np.float32)
    # Gently match the reference's dark blue/cool environment.
    patch *= float(config.get("friend_brightness", 0.85))
    patch[:, :, 0] *= 1.04
    patch[:, :, 2] *= 0.96
    patch = np.uint8(np.clip(patch, 0, 255))
    sat = float(config.get("friend_saturation", 0.85))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
    patch = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blur = float(config.get("friend_blur", 0.5))
    if blur > 0:
        patch = cv2.GaussianBlur(patch, (0, 0), blur)

    # Soft oval mask fades into the helmet while leaving armor visible.
    mask = np.zeros((oh, ow), np.float32)
    cv2.ellipse(mask, (ow // 2, oh // 2), (int(ow * .47), int(oh * .47)),
                0, 0, 360, 1, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), max(2.5, min(ow, oh) * 0.070))
    angle = float(config.get("friend_angle_degrees", -7))
    rotate = cv2.getRotationMatrix2D((ow / 2.0, oh / 2.0), angle, 1.0)
    patch = cv2.warpAffine(patch, rotate, (ow, oh), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, rotate, (ow, oh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    x1, y1 = cx - ow // 2, cy - oh // 2
    x2, y2 = x1 + ow, y1 + oh
    rx1, ry1, rx2, ry2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
    if rx1 >= rx2 or ry1 >= ry2:
        return frame
    px1, py1 = rx1 - x1, ry1 - y1
    px2, py2 = px1 + rx2 - rx1, py1 + ry2 - ry1
    alpha = (mask[py1:py2, px1:px2] * opacity)[:, :, None]
    background = frame[ry1:ry2, rx1:rx2].astype(np.float32)
    foreground = patch[py1:py2, px1:px2].astype(np.float32)
    frame[ry1:ry2, rx1:rx2] = np.uint8(
        np.clip(background * (1.0 - alpha) + foreground * alpha, 0, 255)
    )
    return frame


class MainFaceSwapper:
    def __init__(self, photo: np.ndarray, model_path: Path, provider: str = "cpu"):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Не найдена модель {model_path}. Скачайте модель INSwapper по официальному источнику "
                "(убедитесь в праве использовать её). Для теста фона: --skip-main-face"
            )
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise RuntimeError("Установите pip install -r requirements.txt") from e
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if provider == "cuda" else ["CPUExecutionProvider"])
        self.analysis = FaceAnalysis(name="buffalo_l", providers=providers)
        self.analysis.prepare(ctx_id=0 if provider == "cuda" else -1, det_size=(640, 640))
        source_faces = self.analysis.get(photo)
        if not source_faces:
            raise RuntimeError("На input/me.jpg не найдено лицо. Нужна чёткая фотография анфас.")
        self.source_face = max(source_faces, key=lambda f: self._area(f.bbox))
        self.model = insightface.model_zoo.get_model(str(model_path), providers=providers)
        if self.model is None:
            raise RuntimeError(f"Не удалось загрузить модель {model_path}")

    @staticmethod
    def _area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def run(self, frame: np.ndarray) -> np.ndarray:
        faces = self.analysis.get(frame)
        if not faces:
            return frame
        h, w = frame.shape[:2]
        # Reject tiny figurine faces; the actor is the largest reasonably sized face.
        choices = [f for f in faces if self._area(f.bbox) >= (w * h * 0.007)]
        if not choices:
            return frame
        actor = max(choices, key=lambda f: self._area(f.bbox))
        return self.model.get(frame, actor, self.source_face, paste_back=True)


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("Не найден ffmpeg. Установите его и добавьте в PATH.")
    return path


def open_reference(path: Path):
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise RuntimeError(f"Невозможно прочитать видео: {path}")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(video.get(cv2.CAP_PROP_FPS))
    if not fps or not math.isfinite(fps):
        raise RuntimeError("Не удалось определить FPS видео")
    return video, width, height, fps


def main():
    parser = argparse.ArgumentParser(description="Поставить себя на место героя и друга на место робота")
    parser.add_argument("--video", type=Path, default=ROOT / "input/reference.mp4")
    parser.add_argument("--me", type=Path, default=ROOT / "input/me.jpg")
    parser.add_argument("--friend", type=Path, default=ROOT / "input/friend.jpg")
    parser.add_argument("--model", type=Path, default=ROOT / "models/inswapper_128.onnx")
    parser.add_argument("--out", type=Path, default=ROOT / "output/result.mp4")
    parser.add_argument("--preview-time", type=float, default=None,
                        help="Сохранить один JPG-кадр вместо всего ролика, например 2.5")
    parser.add_argument("--skip-main-face", action="store_true",
                        help="Быстрый тест заднего фона без AI-подмены главного лица")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    config = SETTINGS
    robot_portrait = extract_friend_portrait(load_photo(args.friend, "фотография друга"))
    swapper = None
    if not args.skip_main_face:
        my_photo = load_photo(args.me, "ваша фотография")
        swapper = MainFaceSwapper(my_photo, args.model, args.provider)
    video, width, height, fps = open_reference(args.video)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def process(frame: np.ndarray, t: float) -> np.ndarray:
        if swapper is not None:
            frame = swapper.run(frame)
        return place_friend_face(frame, robot_portrait, config, t)

    if args.preview_time is not None:
        n = max(0, int(round(args.preview_time * fps)))
        video.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = video.read()
        if not ok:
            raise RuntimeError("Нельзя получить превью: указанное время вне видео")
        frame = process(frame, n / fps)
        out = args.out if args.out.suffix.lower() in (".png", ".jpg", ".jpeg") else args.out.with_suffix(".jpg")
        if not cv2.imwrite(str(out), frame):
            raise RuntimeError(f"Не удалось записать превью: {out}")
        print(f"Тестовый кадр сохранён: {out}")
        video.release()
        return

    ffmpeg = find_ffmpeg()
    ffmpeg_cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.9g}", "-i", "pipe:0",
        "-i", str(args.video),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(args.out),
    ]
    print(f"Создаём {args.out} ({width}x{height}, {fps:g} FPS)...", flush=True)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    count = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            processed = process(frame, count / fps)
            proc.stdin.write(np.ascontiguousarray(processed).tobytes())
            count += 1
            if count % 30 == 0:
                print(f"Обработано {count} кадров ({count / fps:.1f} с)", flush=True)
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg завершился с кодом {ret}")
    except Exception:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
        raise
    finally:
        video.release()
    print(f"Готово! {args.out}")


if __name__ == "__main__":
    main()
