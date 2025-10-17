#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from diffusers import AutoPipelineForInpainting


def die(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def compute_center_padding(
    src_w: int, src_h: int, tgt_w: int, tgt_h: int
) -> Tuple[int, int, int, int, int, int]:
    if tgt_w < 1 or tgt_h < 1:
        die("Target width/height must be positive")
    left = max((tgt_w - src_w) // 2, 0)
    right = max(tgt_w - src_w - left, 0)
    top = max((tgt_h - src_h) // 2, 0)
    bottom = max(tgt_h - src_h - top, 0)
    x = left
    y = top
    return left, right, top, bottom, x, y


def ensure_valid_fps(fps: float) -> float:
    if fps and fps > 0 and np.isfinite(fps):
        return float(fps)
    return 30.0


def gaussian_kernel_for_size(width: int, height: int) -> Tuple[int, int]:
    base = max(min(width, height) // 20, 25)
    k = base if base % 2 == 1 else base + 1
    k = max(25, min(k, 151))
    return (k, k)


def build_init_and_mask(
    frame_bgr: np.ndarray,
    target_w: int,
    target_h: int,
) -> Tuple[Image.Image, Image.Image]:
    src_h, src_w = frame_bgr.shape[:2]
    left, right, top, bottom, x, y = compute_center_padding(src_w, src_h, target_w, target_h)

    # Background context: resized + strong blur
    bg = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    kx, ky = gaussian_kernel_for_size(target_w, target_h)
    bg = cv2.GaussianBlur(bg, (kx, ky), sigmaX=0, sigmaY=0)

    # Paste original frame at center (kept area)
    bg[y : y + src_h, x : x + src_w] = frame_bgr

    # Mask: white (255) where we want the model to paint; black (0) to preserve
    mask = np.full((target_h, target_w), 255, dtype=np.uint8)
    mask[y : y + src_h, x : x + src_w] = 0

    # Feather mask edges slightly to reduce seams
    feather = max(7, int(round(min(target_w, target_h) * 0.01)) | 1)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)

    init_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    init_image = Image.fromarray(init_rgb)
    mask_image = Image.fromarray(mask, mode="L")
    return init_image, mask_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AI video outpainting (local) using Stable Diffusion inpainting. "
            "Expands canvas to target width/height and synthesizes new content in the borders."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Path to output video file (.mp4 recommended)")
    parser.add_argument("--width", type=int, required=True, help="Target output width in pixels")
    parser.add_argument("--height", type=int, required=True, help="Target output height in pixels")
    parser.add_argument("--prompt", default="", help="Text prompt guiding the outpainted regions")
    parser.add_argument(
        "--negative-prompt",
        default="low quality, blurry, distorted, deformed, artifacts, watermark, text",
        help="Negative prompt to avoid undesired artifacts",
    )
    parser.add_argument(
        "--model",
        default="runwayml/stable-diffusion-inpainting",
        help="Hugging Face model id for inpainting (e.g., runwayml/stable-diffusion-inpainting)",
    )
    parser.add_argument("--steps", type=int, default=25, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run on",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 if on CUDA to reduce memory",
    )

    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        die(f"Input not found: {in_path}")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_w = int(args.width)
    target_h = int(args.height)
    if target_w <= 0 or target_h <= 0:
        die("--width and --height must be positive integers")

    # Open video
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        die(f"Failed to open video: {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = ensure_valid_fps(src_fps)

    if src_w > target_w or src_h > target_h:
        die("Target size must be greater than or equal to source size")

    # Select device and dtype
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (device == "cuda" and args.fp16) else torch.float32

    # Load pipeline
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            args.model,
            torch_dtype=dtype,
            variant="fp16" if (dtype == torch.float16) else None,
        )
    except Exception as e:
        die(f"Failed to load inpainting pipeline/model '{args.model}': {e}")

    pipe = pipe.to(device)
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass

    # Optional: disable safety checker if present
    try:
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
    except Exception:
        pass

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(args.seed))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (target_w, target_h))
    if not writer.isOpened():
        cap.release()
        die("Failed to create output video writer. Try a different path or codec.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            init_image, mask_image = build_init_and_mask(
                frame_bgr=frame,
                target_w=target_w,
                target_h=target_h,
            )

            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance),
                generator=generator,
            ).images[0]

            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            writer.write(result_bgr)
            processed += 1
    finally:
        cap.release()
        writer.release()

    print(
        f"Wrote AI-outpainted video: {out_path} | frames={processed}/{frame_count} | size={target_w}x{target_h} | fps={out_fps:.2f} | model={args.model}"
    )


if __name__ == "__main__":
    main()
