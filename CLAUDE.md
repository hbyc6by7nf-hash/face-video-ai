# Claude Code: AI face video project

Python project that replaces the protagonist's face (InsightFace INSwapper + GFPGAN, BiSeNet occlusion mask,
temporal landmark smoothing) and composites a friend's head over the background robot helmet
(optical-flow tracked 2D composite). Source audio is copied unchanged. See README.md.

Inputs (not committed): `input/reference.mp4`, the user's photo, the friend's photo. Models in `models/` (not committed).

## Commands
```
python -m pip install -r requirements.txt
python scripts/download_models.py --accept-noncommercial   # only after the user accepts the InsightFace licence
python run.py --me input/me.png --friend input/friend.png --preview-time 2.5 --out output/preview.jpg
python run.py --me input/me.png --friend input/friend.png --out output/result.mp4
python app.py   # Gradio UI
```

## Honesty/limitations
The friend's head is a tracked, relit 2D composite — it does not turn or talk; never describe it as a generative
or photorealistic replacement. Do not claim results without running the pipeline and inspecting frames.
InsightFace weights are non-commercial only. Do not invent or download personal photos.
