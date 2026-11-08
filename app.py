#!/usr/bin/env python3
"""Web interface: upload the video and two photos, press «Создать видео».

    python app.py            # opens http://127.0.0.1:7860
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402

from facevideo import video  # noqa: E402
from facevideo.friend import FriendSettings, draw_box  # noqa: E402
from facevideo.models import REQUIRED_MODELS, ModelBank, missing_models  # noqa: E402
from facevideo.pipeline import JobSettings, run  # noqa: E402

OUT_DIR = ROOT / "output"
_models: ModelBank | None = None

NOTICE = """
**Что делает программа**
* **Главный герой** — нейросетевая замена лица: детектор SCRFD → ArcFace → INSwapper 128 → GFPGAN,
  маска BiSeNet (руки/предметы перед лицом остаются сверху), сглаживание траектории лица и маски по
  времени против дрожания, подстройка цвета кожи под освещение сцены. Звук копируется без изменений.
* **Друг вместо головы робота** — у робота нет человеческого лица, поэтому это **не** нейросетевая замена,
  а отслеживаемый 2D-композит: голова друга вырезается нейросетью (лицо+волосы+шапка), следует за шлемом
  (оптический поток), перекрашивается под свет сцены. Голова не поворачивается и не говорит.

Используйте только фото людей, давших согласие. Не выдавайте результат за настоящую запись.
"""


def models() -> ModelBank:
    global _models
    if _models is None:
        _models = ModelBank()
    return _models


def _settings(enhance, skin_match, smoothing, friend_on, anchor, bx, by, bw, bh,
              f_start, f_end, brightness, angle, scale) -> JobSettings:
    s = JobSettings()
    s.main.enhance = enhance
    s.main.skin_match = skin_match
    s.main.smoothing_sigma = smoothing
    s.friend = FriendSettings(enabled=friend_on, anchor_time=anchor, box=(bx, by, bw, bh),
                              start_time=f_start, end_time=f_end, brightness=brightness,
                              angle=angle, scale=scale)
    return s


def _check(video_file, me, friend, need_me=True):
    missing = missing_models()
    if missing:
        raise gr.Error("Нет моделей: " + ", ".join(f"{m} ({REQUIRED_MODELS[m]})" for m in missing)
                       + ". Запустите: python scripts/download_models.py --accept-noncommercial")
    if not video_file:
        raise gr.Error("Загрузите оригинальное видео")
    if need_me and not me:
        raise gr.Error("Загрузите вашу фотографию")


def show_box(video_file, anchor, bx, by, bw, bh):
    if not video_file:
        return None
    info = video.probe(Path(video_file))
    frame = video.read_frame(Path(video_file), int(round(anchor * info.fps)))
    cfg = FriendSettings(anchor_time=anchor, box=(bx, by, bw, bh))
    return cv2.cvtColor(draw_box(frame, cfg), cv2.COLOR_BGR2RGB)


def preview(video_file, me, friend, t, *opts, progress=gr.Progress()):
    _check(video_file, me, friend, need_me=False)
    s = _settings(*opts)
    out = Path(tempfile.mkdtemp()) / "preview.jpg"
    run(Path(video_file), Path(me) if me else None, Path(friend) if friend else None, out, s,
        models=models(), preview_time=t, progress=lambda f, m: progress(f, desc=m))
    return cv2.cvtColor(cv2.imread(str(out)), cv2.COLOR_BGR2RGB)


def create(video_file, me, friend, *opts, progress=gr.Progress()):
    _check(video_file, me, friend)
    s = _settings(*opts)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"result_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    log: list[str] = []

    def cb(f, m):
        progress(f, desc=m)
        if not log or log[-1] != m:
            log.append(m)

    run(Path(video_file), Path(me), Path(friend) if friend else None, out, s,
        models=models(), progress=cb)
    keep = [line for line in log if not line.startswith(("Анализ кадра", "Рендер кадра"))]
    return str(out), str(out), "\n".join(keep + [log[-1]] if log else keep)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI-редактор видео: замена лиц") as ui:
        gr.Markdown("# AI-редактор видео: замена лиц")
        gr.Markdown(NOTICE)
        with gr.Row():
            video_in = gr.Video(label="1. Оригинальное видео", sources=["upload"])
            me_in = gr.Image(label="2. Ваша фотография", type="filepath", sources=["upload"])
            friend_in = gr.Image(label="3. Фотография друга", type="filepath", sources=["upload"])

        with gr.Accordion("Настройки главного лица", open=False):
            enhance = gr.Checkbox(True, label="Улучшение деталей лица (GFPGAN)")
            skin_match = gr.Slider(0, 1, 0.55, step=0.05, label="Подстройка цвета кожи под сцену")
            smoothing = gr.Slider(0, 3, 1.2, step=0.1, label="Стабилизация (сглаживание, кадров)")

        d = FriendSettings()
        with gr.Accordion("Друг вместо головы робота", open=False):
            friend_on = gr.Checkbox(True, label="Вставлять лицо друга")
            with gr.Row():
                anchor = gr.Number(d.anchor_time, label="Кадр с рамкой, с")
                f_start = gr.Number(d.start_time, label="Начало, с")
                f_end = gr.Number(d.end_time, label="Конец, с")
            with gr.Row():
                bx = gr.Slider(0, 1, d.box[0], step=0.005, label="Рамка X")
                by = gr.Slider(0, 1, d.box[1], step=0.005, label="Рамка Y")
                bw = gr.Slider(0.02, 1, d.box[2], step=0.005, label="Ширина")
                bh = gr.Slider(0.02, 1, d.box[3], step=0.005, label="Высота")
            with gr.Row():
                brightness = gr.Slider(0.3, 1.5, d.brightness, step=0.05, label="Яркость")
                angle = gr.Slider(-30, 30, d.angle, step=1, label="Наклон, °")
                scale = gr.Slider(0.5, 1.5, d.scale, step=0.05, label="Размер")
            box_btn = gr.Button("Показать рамку")
            box_img = gr.Image(label="Рамка на опорном кадре", interactive=False)

        opts = [enhance, skin_match, smoothing, friend_on, anchor, bx, by, bw, bh,
                f_start, f_end, brightness, angle, scale]
        with gr.Row():
            t_prev = gr.Number(2.5, label="Время превью, с")
            prev_btn = gr.Button("Превью кадра")
        prev_img = gr.Image(label="Превью", interactive=False)

        go = gr.Button("Создать видео", variant="primary", size="lg")
        result = gr.Video(label="Результат")
        file_out = gr.File(label="Скачать MP4")
        log = gr.Textbox(label="Журнал", lines=6)

        box_btn.click(show_box, [video_in, anchor, bx, by, bw, bh], box_img)
        prev_btn.click(preview, [video_in, me_in, friend_in, t_prev] + opts, prev_img)
        go.click(create, [video_in, me_in, friend_in] + opts, [result, file_out, log])
    return ui


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", inbrowser=True)
