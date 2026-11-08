#!/usr/bin/env python3
"""Download the ONNX models used by the pipeline into ./models.

Licences (check them yourself before use — this script does not grant any rights):

* buffalo_l (SCRFD det_10g, ArcFace w600k_r50) and inswapper_128 — InsightFace.
  The InsightFace pretrained weights are released for NON-COMMERCIAL research
  use only. https://github.com/deepinsight/insightface#license
* gfpgan_1.4 — GFPGAN (Tencent ARC), Apache-2.0 code; weights distributed by
  the GFPGAN project. https://github.com/TencentARC/GFPGAN
* bisenet_resnet_34 — face parsing (CelebAMask-HQ trained), MIT-licensed
  reimplementation, ONNX export from facefusion-assets.

Usage:  python scripts/download_models.py [--accept-noncommercial]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

FF = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"
BUFFALO = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# name -> (url, sha256 or None)
FILES = {
    "inswapper_128.onnx": (FF + "inswapper_128.onnx",
        "a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7"),
    "gfpgan_1.4.onnx": (FF + "gfpgan_1.4.onnx",
        "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385"),
    "bisenet_resnet_34.onnx": (FF + "bisenet_resnet_34.onnx",
        "4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96"),
}
BUFFALO_FILES = ["det_10g.onnx", "w600k_r50.onnx"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path) -> None:
    print(f"↓ {url}", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-noncommercial", action="store_true",
                    help="confirm you have read the InsightFace non-commercial licence")
    args = ap.parse_args()
    if not args.accept_noncommercial:
        print("InsightFace models (детектор, ArcFace, INSwapper) разрешены только для\n"
              "некоммерческого исследовательского использования. Прочитайте лицензию:\n"
              "https://github.com/deepinsight/insightface#license\n"
              "Если согласны — запустите снова с флагом --accept-noncommercial")
        return 2

    MODELS.mkdir(exist_ok=True)
    for name, (url, digest) in FILES.items():
        dst = MODELS / name
        if not dst.is_file():
            fetch(url, dst)
        if digest and sha256(dst) != digest:
            print(f"! SHA-256 для {name} не совпадает с ожидаемым "
                  f"({sha256(dst)}). Файл оставлен, но проверьте источник.", file=sys.stderr)
        else:
            print(f"✓ {name}")

    if not all((MODELS / n).is_file() for n in BUFFALO_FILES):
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "buffalo_l.zip"
            fetch(BUFFALO, z)
            with zipfile.ZipFile(z) as zf:
                for member in zf.namelist():
                    base = Path(member).name
                    if base in BUFFALO_FILES:
                        with zf.open(member) as src, (MODELS / base).open("wb") as out:
                            shutil.copyfileobj(src, out)
    for n in BUFFALO_FILES:
        print(f"✓ {n}" if (MODELS / n).is_file() else f"✗ {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
