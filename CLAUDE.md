# Claude Code: AI face video project

This is a Python video editing project. Original input is at `input/reference.mp4`.
User supplies `input/me.jpg` and `input/friend.jpg` locally.

Goal: replace the foreground actor's face with the user's likeness using InsightFace INSwapper; insert the friend's face in place of the robot helmet on the left between approximately 1.5–3.8 seconds. Preserve source motion and audio.

## First steps
1. Read `README.md` and inspect `src/make_video.py`.
2. Check installed Python, FFmpeg, and availability/license of `models/inswapper_128.onnx`.
3. Verify the required user photos exist; do not invent or download personal photos.
4. Set up dependencies, render a preview frame at 2.5 seconds with `--skip-main-face`, allow user to adjust coordinates.
5. When model is provided, render full `output/result.mp4`.

## Commands
```
python -m pip install -r requirements-preview.txt
python src/make_video.py --friend input/friend.jpg --skip-main-face --preview-time 2.5 --out output/preview.jpg
python -m pip install -r requirements.txt
python src/make_video.py --me input/me.jpg --friend input/friend.jpg --out output/result.mp4
```

## Honesty/limitations
This is NOT a fully generative two-person video pipeline: protagonist uses a pre-trained swapper; friend uses animated 2D composite, which may not look photorealistic. No interface included. Do not claim either is accomplished without executing the pipeline and verifying output. Model licenses/permissions must be checked. If higher realism is requested, implement a separate masking and licensed video-to-video service rather than promising 2D compositing will be sufficient.
