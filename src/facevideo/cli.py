"""Command line entry point:  python -m facevideo.cli --help  (run from src/ or via run.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import MODEL_DIR, REQUIRED_MODELS, missing_models
from .pipeline import JobSettings, run

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AI-редактор видео: ваше лицо у героя, лицо друга вместо робота")
    p.add_argument("--video", type=Path, default=ROOT / "input/reference.mp4")
    p.add_argument("--me", type=Path, default=None, help="ваша фотография")
    p.add_argument("--friend", type=Path, default=None, help="фотография друга")
    p.add_argument("--out", type=Path, default=ROOT / "output/result.mp4")
    p.add_argument("--preview-time", type=float, default=None, help="сохранить один кадр (секунды)")
    p.add_argument("--skip-main-face", action="store_true")
    p.add_argument("--no-enhance", action="store_true", help="без GFPGAN (быстрее, менее чётко)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--friend-box", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                   help="рамка головы робота в долях кадра на момент --friend-anchor")
    p.add_argument("--friend-anchor", type=float, help="время (с) кадра, в котором задана рамка")
    p.add_argument("--friend-range", type=float, nargs=2, metavar=("START", "END"))
    p.add_argument("--friend-brightness", type=float)
    p.add_argument("--friend-angle", type=float)
    p.add_argument("--skin-match", type=float, help="0..1 подстройка цвета кожи героя под сцену")
    args = p.parse_args(argv)

    s = JobSettings(swap_main=not args.skip_main_face, device=args.device)
    s.main.enhance = not args.no_enhance
    if args.skin_match is not None:
        s.main.skin_match = args.skin_match
    f = s.friend
    if args.friend_box:
        f.box = tuple(args.friend_box)
    if args.friend_anchor is not None:
        f.anchor_time = args.friend_anchor
    if args.friend_range:
        f.start_time, f.end_time = args.friend_range
    if args.friend_brightness is not None:
        f.brightness = args.friend_brightness
    if args.friend_angle is not None:
        f.angle = args.friend_angle

    missing = missing_models()
    if missing:
        print("Не хватает моделей в", MODEL_DIR)
        for m in missing:
            print(f"  - {m}: {REQUIRED_MODELS[m]}")
        print("Скачайте: python scripts/download_models.py --accept-noncommercial")
        return 1
    run(args.video, None if args.skip_main_face else args.me, args.friend, args.out, s,
        preview_time=args.preview_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
