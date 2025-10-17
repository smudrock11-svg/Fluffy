### Video outpainting options

This repo provides two ways to outpaint (expand) videos to desired dimensions:

1) Local, no-API outpainting with OpenCV (`local_outpaint.py`)
2) Veo-3.0 model outpainting via Google AI Studio (`veo_outpaint.py`)

Pick the approach that fits your needs. If you want to avoid external services entirely, use the local OpenCV option.

### Prerequisites
- Python 3.9+
- For the Google AI method only: access to Google AI Studio and a `GOOGLE_API_KEY`

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional, only for `veo_outpaint.py`
export GOOGLE_API_KEY=YOUR_KEY_HERE
```

### Local outpainting (no APIs)
- Expands canvas to `--width` x `--height`:
  - Simple modes: `blur` (default), `mirror`, `replicate`, `solid`
  - AI mode: `ai` uses MMagic (DeepFillv2) inpainting to synthesize new content in the padded regions
- Works entirely offline (no external APIs). AI mode uses local PyTorch + MMagic.

```bash
# Example: outpaint to 1080x1920 with blurred background
python local_outpaint.py \
  --input ./input.mp4 \
  --width 1080 \
  --height 1920 \
  --out ./outpainted_local.mp4

# Mirror background
python local_outpaint.py \
  --input ./input.mp4 \
  --width 1080 \
  --height 1920 \
  --mode mirror \
  --out ./outpainted_mirror.mp4

# Solid background color
python local_outpaint.py \
  --input ./input.mp4 \
  --width 1080 \
  --height 1920 \
  --mode solid \
  --color "#000000" \
  --out ./outpainted_solid.mp4

# If source is larger than target, scale it down to fit
python local_outpaint.py \
  --input ./input_4k.mp4 \
  --width 1080 \
  --height 1920 \
  --scale-to-fit \
  --out ./outpainted_scaled.mp4

# AI mode (MMagic DeepFillv2) to synthesize new borders
python local_outpaint.py \
  --input ./input.mp4 \
  --width 1080 \
  --height 1920 \
  --mode ai \
  --device auto \
  --out ./outpainted_ai.mp4
```

Notes:
- Output FPS defaults to the source FPS (or 30 if unavailable). Override via `--fps`.
- For `--mode solid`, color can be `#RRGGBB` or `R,G,B`.
- For AI mode, install dependencies:
  ```bash
  pip install torch torchvision Pillow mmengine mmcv mmagic
  ```
  GPU is recommended but not required.

### Veo-3.0 video outpainting (Google AI)
- The tool uploads your input video, requests outpainting to the given height (and optional width), and writes the result to the output path.
- Width defaults to the source video width unless overridden.

```bash
# Basic: keep source width, change only height
python veo_outpaint.py \
  --input ./input.mp4 \
  --height 1536 \
  --out ./outpainted.mp4

# With explicit width and extra hints
python veo_outpaint.py \
  --input ./input.mp4 \
  --height 1920 \
  --width 1080 \
  --fps 24 \
  --duration 8 \
  --prompt "Extend the scene vertically while preserving style and motion."
```

### What it does
- Local: pads/extends the canvas using image-based techniques only, no network calls.
- Google AI: uploads to `https://generativelanguage.googleapis.com/upload/v1beta/files`, calls `POST https://generativelanguage.googleapis.com/v1beta/models/veo-3.0:generateContent`, then saves the video.

### Notes
- Local method produces deterministic, content-preserving padding (no new content is generated).
- Google AI method can synthesize plausible new content but requires access/quota.
- Large inputs/outputs may take several minutes; timeouts are configured generously.
- For Google AI, height is required; width is optional and defaults to source width.
