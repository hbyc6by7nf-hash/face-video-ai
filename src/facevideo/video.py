"""Video reading, scene-cut detection and MP4 writing with the original audio."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    fps_str: str
    frames: int
    has_audio: bool


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Не найден {name}. Установите FFmpeg и добавьте его в PATH.")
    return path


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        [ffmpeg_bin("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    streams = json.loads(out)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    num, den = (int(x) for x in v.get("r_frame_rate", "0/1").split("/"))
    fps = num / den if den else 0.0
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps:
        fps = cap.get(cv2.CAP_PROP_FPS)
        num, den = int(round(fps * 1000)), 1000
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(w, h, fps, f"{num}/{den}", frames,
                     any(s["codec_type"] == "audio" for s in streams))


def iter_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Невозможно открыть видео: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_frame(path: Path, index: int) -> np.ndarray:
    for i, frame in enumerate(iter_frames(path)):
        if i == index:
            return frame
    raise ValueError(f"Кадр {index} вне видео")


class SceneCutDetector:
    """Hard-cut detector based on HSV histogram correlation between consecutive frames."""

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.prev = None

    def __call__(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self.prev is not None:
            cut = cv2.compareHist(self.prev, hist, cv2.HISTCMP_CORREL) < self.threshold
        self.prev = hist
        return cut


class Mp4Writer:
    """Pipes BGR frames to FFmpeg (H.264) and muxes the untouched source audio."""

    def __init__(self, out: Path, source: Path, info: VideoInfo, crf: int = 16, preset: str = "medium"):
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
            "-r", info.fps_str, "-i", "pipe:0",
            "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "copy",                       # original audio, bit-exact
            "-movflags", "+faststart", str(out),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"FFmpeg завершился с кодом {self.proc.returncode}")

    def abort(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        if self.proc.poll() is None:
            self.proc.kill()
