### Veo-3.0 video outpainting (upload + custom height)

A minimal Python CLI that uploads a source video to Google AI Studio, calls the `veo-3.0` model to outpaint (expand) the video canvas to a custom height, and saves the generated result.

- **Real API calls**: no demo/mock/simulation
- **Upload flow**: uses `files:upload` endpoint
- **Custom height**: specify target output height in pixels

### Prerequisites
- Python 3.9+
- Access to Google AI Studio API and the `veo-3.0` model
- A valid API key in `GOOGLE_API_KEY`

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your API key (or put it in an .env loader you prefer)
export GOOGLE_API_KEY=YOUR_KEY_HERE
```

### Usage
- The tool uploads your input video, requests outpainting to the given height, and writes the result to the output path.
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
- Uploads the video to `https://generativelanguage.googleapis.com/upload/v1beta/files`
- Calls `POST https://generativelanguage.googleapis.com/v1beta/models/veo-3.0:generateContent`
- Requests video output and saves the returned video bytes to `--out`

### Notes
- You must have access and quota for `veo-3.0` in your Google AI account.
- Large inputs/outputs may take several minutes; timeouts are configured generously.
- Height is required; width is optional and defaults to source width.
