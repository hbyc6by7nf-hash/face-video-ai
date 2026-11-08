"""End-to-end pipeline: analyse the video, then render the edited MP4."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import align, video
from .friend import (FriendCompositor, FriendSettings, build_head, draw_box, smooth_transforms,
                     track_helmet, _box_px)
from .main_face import MainFaceRenderer, MainFaceSettings, build_track, source_identity
from .models import SKIN_CLASS, ModelBank

Progress = Callable[[float, str], None]


@dataclass
class JobSettings:
    swap_main: bool = True
    main: MainFaceSettings = field(default_factory=MainFaceSettings)
    friend: FriendSettings = field(default_factory=FriendSettings)
    device: str = "auto"
    crf: int = 16


def _log(progress: Progress | None, frac: float, msg: str) -> None:
    if progress:
        progress(frac, msg)
    else:
        print(f"[{frac * 100:5.1f}%] {msg}", flush=True)


def _skin_stats(models: ModelBank, frame: np.ndarray, kps: np.ndarray):
    M = align.estimate(kps, align.FFHQ_512)
    crop = align.warp(frame, M, 512)
    skin = (models.parser.parse(crop) == SKIN_CLASS).astype(np.float32)
    return align.lab_stats(crop, skin) if skin.sum() > 500 else None


def run(video_path: Path, me_photo: Path | None, friend_photo: Path | None, out: Path,
        settings: JobSettings | None = None, models: ModelBank | None = None,
        preview_time: float | None = None, progress: Progress | None = None) -> Path:
    s = settings or JobSettings()
    models = models or ModelBank(device=s.device)
    info = video.probe(video_path)
    W, H, fps, N = info.width, info.height, info.fps, info.frames
    t0 = time.time()

    do_main = s.swap_main and me_photo is not None
    do_friend = s.friend.enabled and friend_photo is not None
    if not (do_main or do_friend):
        raise ValueError("Нечего делать: нужна ваша фотография и/или фотография друга")

    def load(p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Не удалось открыть изображение {p}")
        return img

    source_emb = source_identity(models, load(me_photo)) if do_main else None
    head = build_head(models, load(friend_photo)) if do_friend else None

    # analysis window (whole video, or ~1.5 s around the preview frame)
    if preview_time is not None:
        target = int(np.clip(round(preview_time * fps), 0, N - 1))
        lo, hi = max(0, target - int(1.5 * fps)), min(N - 1, target + int(1.5 * fps))
    else:
        target, lo, hi = None, 0, N - 1

    fcfg = s.friend
    anchor = int(np.clip(round(fcfg.anchor_time * fps), 0, N - 1))
    f_lo = int(np.clip(round(fcfg.start_time * fps), 0, N - 1))
    f_hi = int(np.clip(round(fcfg.end_time * fps), 0, N - 1))
    if preview_time is not None:            # tracking must reach from the anchor to the preview frame
        lo, hi = min(lo, anchor if do_friend else lo), max(hi, anchor if do_friend else hi)

    # ---------------- pass 1: analysis
    segments = np.zeros(N, np.int32)
    cuts: set[int] = set()
    detections: list[list] = [[] for _ in range(N)]
    grays: dict[int, np.ndarray] = {}
    anchor_frame = None
    cut_det = video.SceneCutDetector()
    seg = 0
    for i, frame in enumerate(video.iter_frames(video_path)):
        if i > hi:
            break
        if cut_det(frame):
            seg += 1
            cuts.add(i)
        segments[i] = seg
        if i < lo:
            continue
        if do_main:
            faces = models.detector.detect(frame)
            for f in faces:
                if f.area >= s.main.min_face_fraction * W * H:
                    f.embedding = models.embedder.embed(frame, f.kps)
            detections[i] = [f for f in faces if f.embedding is not None]
        if do_friend and f_lo <= i <= f_hi:
            grays[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i == anchor:
            anchor_frame = frame.copy()
        if i % 10 == 0:
            _log(progress, 0.30 * (i - lo + 1) / (hi - lo + 1), f"Анализ кадра {i + 1}/{N}")
    segments[hi + 1:] = seg + 1

    track = build_track(detections, segments, W * H, s.main) if do_main else None
    if track is not None:
        _log(progress, 0.31, f"Главный герой найден в {int(track.raw_detected[lo:hi + 1].sum())} "
                             f"кадрах, после заполнения пропусков — {int(track.valid[lo:hi + 1].sum())}")

    compositor = None
    if do_friend:
        box = _box_px(fcfg.box, W, H)
        tr = track_helmet(grays, anchor, f_lo, f_hi, box, cuts) if anchor in grays else {}
        tr = smooth_transforms(tr, fcfg.smoothing_sigma, box)
        scene_skin = None
        if anchor_frame is not None:
            faces = models.detector.detect(anchor_frame)
            if track is not None and track.valid[anchor]:
                scene_skin = _skin_stats(models, anchor_frame, track.kps[anchor])
            elif faces:
                scene_skin = _skin_stats(models, anchor_frame, faces[0].kps)
        compositor = FriendCompositor(head, fcfg, (W, H), tr, scene_skin, cuts)
        span = f"{min(tr) / fps:.2f}–{max(tr) / fps:.2f} с" if tr else "не найден"
        _log(progress, 0.32, f"Шлем отслежен: {span} ({len(tr)} кадров)")

    renderer = MainFaceRenderer(models, source_emb, s.main) if do_main else None

    def process(frame: np.ndarray, i: int) -> np.ndarray:
        if renderer is not None and track.valid[i]:
            frame = renderer.render(frame, track.kps[i], int(track.segment[i]))
        if compositor is not None:
            frame = compositor.render(frame, i)
        return frame

    # ---------------- preview: one frame (main renderer warm-up over a few previous frames)
    if target is not None:
        for i, frame in enumerate(video.iter_frames(video_path)):
            if i < target:
                continue
            frame = process(frame, i)
            out = out if out.suffix.lower() in (".jpg", ".jpeg", ".png") else out.with_suffix(".jpg")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            box_preview = out.with_name(out.stem + "_box" + out.suffix)
            if anchor_frame is not None:
                cv2.imwrite(str(box_preview), draw_box(anchor_frame, fcfg))
            _log(progress, 1.0, f"Превью сохранено: {out}")
            return out

    # ---------------- pass 2: render
    writer = video.Mp4Writer(out, video_path, info, crf=s.crf)
    try:
        for i, frame in enumerate(video.iter_frames(video_path)):
            writer.write(process(frame, i))
            if i % 5 == 0:
                el = time.time() - t0
                _log(progress, 0.33 + 0.66 * (i + 1) / N, f"Рендер кадра {i + 1}/{N} ({el:.0f} с)")
        writer.close()
    except BaseException:
        writer.abort()
        raise
    _log(progress, 1.0, f"Готово: {out} за {time.time() - t0:.0f} с")
    return out
