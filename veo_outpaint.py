#!/usr/bin/env python3
import os
import sys
import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import requests
import cv2

GOOGLE_AI_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
GOOGLE_AI_MODELS_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL_NAME = "veo-3.0"


def die(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def get_video_resolution(video_path: Path) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        die(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        die("Could not read video resolution (width/height <= 0)")
    return width, height


def upload_file_to_google_ai(file_path: Path, api_key: str) -> Dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        # Default to mp4 for video
        mime_type = "video/mp4"
    headers = {
        "X-Goog-Upload-Protocol": "raw",
        "X-Goog-Upload-File-Name": file_path.name,
        "Content-Type": mime_type,
        "Accept": "application/json",
    }
    url = f"{GOOGLE_AI_UPLOAD_URL}?key={api_key}"
    with open(file_path, "rb") as f:
        data = f.read()
    resp = requests.post(url, headers=headers, data=data, timeout=600)
    if resp.status_code >= 300:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        die(f"File upload failed ({resp.status_code}): {err}")
    try:
        payload = resp.json()
    except Exception as e:
        die(f"Unexpected upload response JSON parse error: {e}")
    # Expecting payload like { "file": { "name": "files/..", "uri": "...", ... } } or flat keys
    file_obj = payload.get("file", payload)
    if not isinstance(file_obj, dict):
        die(f"Unexpected upload response shape: {payload}")
    if not (file_obj.get("uri") or file_obj.get("name")):
        die(f"Upload response missing file uri/name: {file_obj}")
    return file_obj


def parse_video_from_response_json(resp_json: Dict[str, Any]) -> Optional[bytes]:
    # Try to locate inline_data of video in candidates -> content -> parts
    candidates = resp_json.get("candidates") or []
    for cand in candidates:
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            # inline_data path
            inline_data = part.get("inline_data") or part.get("inlineData")
            if inline_data and inline_data.get("mime_type", inline_data.get("mimeType", "")).startswith("video/"):
                data_b64 = inline_data.get("data")
                if data_b64:
                    try:
                        return base64.b64decode(data_b64)
                    except Exception:
                        pass
            # file_data path with direct URI
            file_data = part.get("file_data") or part.get("fileData")
            if file_data:
                # Some responses might include a direct downloadable URI
                uri = file_data.get("file_uri") or file_data.get("fileUri")
                if uri and uri.startswith("http"):
                    try:
                        res = requests.get(uri, timeout=600)
                        if res.status_code < 300:
                            return res.content
                    except Exception:
                        pass
    return None


def generate_outpaint_video(
    api_key: str,
    source_file_uri: str,
    source_mime_type: str,
    output_height: int,
    output_width: Optional[int],
    user_prompt: Optional[str],
    fps: Optional[int] = None,
    duration_seconds: Optional[float] = None,
) -> bytes:
    url = f"{GOOGLE_AI_MODELS_BASE}/{MODEL_NAME}:generateContent?key={api_key}"

    # Build instruction text
    instruction = (
        user_prompt.strip() if user_prompt else
        "Outpaint the input video to the requested output height while preserving visual fidelity, scene composition, and motion continuity."
    )
    dim_text = f"Target output dimensions: height={output_height}px" + (f", width={output_width}px" if output_width else ".")
    full_prompt = (
        f"{instruction}\n\n"
        f"- Perform vertical outpainting to expand the canvas to the target height.\n"
        f"- Maintain style, lighting, and motion consistency across newly generated regions.\n"
        f"- Avoid distorting original content; extend background and scene context naturally.\n"
        f"- {dim_text}\n"
    )

    # Generation config schema may evolve; provide broadly compatible hints
    generation_config: Dict[str, Any] = {
        "response_mime_type": "video/mp4",
    }
    # Provide size hints that many APIs accept
    video_cfg: Dict[str, Any] = {}
    if output_height:
        # Common names used by Google APIs vary; include several for forwards compatibility
        video_cfg["height_px"] = output_height
        video_cfg["height"] = output_height
    if output_width:
        video_cfg["width_px"] = output_width
        video_cfg["width"] = output_width
    if fps:
        video_cfg["frame_rate"] = fps
        video_cfg["fps"] = fps
    if duration_seconds:
        video_cfg["duration_seconds"] = duration_seconds
        video_cfg["duration"] = duration_seconds
    if video_cfg:
        generation_config["video"] = video_cfg

    body: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "file_data": {
                            "file_uri": source_file_uri,
                            "mime_type": source_mime_type,
                        }
                    },
                    {"text": full_prompt},
                ],
            }
        ],
        "generationConfig": generation_config,
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=1200)
    if resp.status_code >= 300:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        die(f"Generation failed ({resp.status_code}): {err}")

    try:
        resp_json = resp.json()
    except Exception as e:
        die(f"Failed to parse generation JSON: {e}")

    video_bytes = parse_video_from_response_json(resp_json)
    if not video_bytes:
        # Provide diagnostics on unexpected response
        sample = json.dumps(resp_json)[:2000]
        die("No video data returned by the model. Response sample: " + sample)

    return video_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Veo-3.0 video outpainting with upload and custom height")
    parser.add_argument("--input", required=True, help="Path to input video file (e.g., .mp4)")
    parser.add_argument("--height", type=int, required=True, help="Target output height in pixels")
    parser.add_argument("--width", type=int, default=None, help="Optional target output width in pixels (defaults to source width)")
    parser.add_argument("--out", required=True, help="Path to output video file (e.g., out.mp4)")
    parser.add_argument("--prompt", default=None, help="Optional additional instructions for the outpainting")
    parser.add_argument("--fps", type=int, default=None, help="Optional target FPS hint")
    parser.add_argument("--duration", type=float, default=None, help="Optional target duration seconds hint")

    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        die("GOOGLE_API_KEY environment variable is not set.")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        die(f"Input file not found: {input_path}")

    target_height = int(args.height)
    if target_height <= 0:
        die("--height must be a positive integer")

    src_width, src_height = get_video_resolution(input_path)
    output_width = args.width if args.width and args.width > 0 else src_width

    print(f"Detected source resolution: {src_width}x{src_height}")
    print(f"Requested output resolution: {output_width}x{target_height}")

    file_obj = upload_file_to_google_ai(input_path, api_key)
    file_uri = file_obj.get("uri")
    if not file_uri:
        # Some responses may return a name usable in file_data: file_uri via separate get
        name = file_obj.get("name")
        if name:
            # Construct a download URI format commonly returned by the API
            file_uri = f"https://generativelanguage.googleapis.com/v1beta/{name}:download?key={api_key}"
        else:
            die(f"Upload did not return a usable URI: {file_obj}")

    mime_type = file_obj.get("mime_type") or file_obj.get("mimeType") or mimetypes.guess_type(str(input_path))[0] or "video/mp4"

    video_bytes = generate_outpaint_video(
        api_key=api_key,
        source_file_uri=file_uri,
        source_mime_type=mime_type,
        output_height=target_height,
        output_width=output_width,
        user_prompt=args.prompt,
        fps=args.fps,
        duration_seconds=args.duration,
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(video_bytes)

    print(f"Saved outpainted video to: {out_path}")


if __name__ == "__main__":
    main()
