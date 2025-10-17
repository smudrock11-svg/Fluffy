#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def die(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def parse_rgb_color(color_str: str) -> Tuple[int, int, int]:
    s = color_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if "," in s:
        parts = s.split(",")
        if len(parts) != 3:
            die("--color must be in '#RRGGBB' or 'R,G,B' format")
        r, g, b = [int(x) for x in parts]
        for v in (r, g, b):
            if not (0 <= v <= 255):
                die("RGB values must be between 0 and 255")
        return (b, g, r)  # OpenCV uses BGR
    if len(s) == 6:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b, g, r)  # BGR
    die("--color must be in '#RRGGBB' or 'R,G,B' format")


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


def outpaint_frame(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    mode: str,
    color_bgr: Tuple[int, int, int],
    scale_to_fit: bool,
) -> np.ndarray:
    src_h, src_w = frame.shape[:2]

    overlay = frame
    if scale_to_fit and (src_w > target_w or src_h > target_h):
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        overlay = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        src_w, src_h = new_w, new_h

    left, right, top, bottom, x, y = compute_center_padding(src_w, src_h, target_w, target_h)

    if mode == "mirror":
        bordered = cv2.copyMakeBorder(
            overlay, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
        )
        return bordered

    if mode == "replicate":
        bordered = cv2.copyMakeBorder(
            overlay, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE
        )
        return bordered

    if mode == "solid":
        canvas = np.full((target_h, target_w, 3), color_bgr, dtype=frame.dtype)
        canvas[y : y + src_h, x : x + src_w] = overlay
        return canvas

    if mode == "blur":
        bg = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)
        kx, ky = gaussian_kernel_for_size(target_w, target_h)
        bg = cv2.GaussianBlur(bg, (kx, ky), sigmaX=0, sigmaY=0)
        bg[y : y + src_h, x : x + src_w] = overlay
        return bg

    die(f"Unsupported mode: {mode}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local video outpainting without external APIs: expand canvas to target width/height "
            "using mirror/replicate/blur/solid background modes."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Path to output video file (.mp4 recommended)")
    parser.add_argument("--width", type=int, required=True, help="Target output width in pixels")
    parser.add_argument("--height", type=int, required=True, help="Target output height in pixels")
    parser.add_argument(
        "--mode",
        choices=["mirror", "replicate", "blur", "solid"],
        default="blur",
        help="Background fill mode for outpainted regions",
    )
    parser.add_argument(
        "--color",
        default="#000000",
        help="Solid fill color for 'solid' mode. '#RRGGBB' or 'R,G,B'",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS (defaults to source FPS or 30)",
    )
    parser.add_argument(
        "--scale-to-fit",
        action="store_true",
        help=(
            "If source is larger than target, scale source down to fit instead of erroring."
        ),
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

    color_bgr = parse_rgb_color(args.color)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        die(f"Failed to open video: {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    if not args.scale_to_fit and (src_w > target_w or src_h > target_h):
        die(
            "Target size must be greater than or equal to source size unless --scale-to-fit is used"
        )

    out_fps = ensure_valid_fps(args.fps if args.fps else src_fps)

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

            result = outpaint_frame(
                frame=frame,
                target_w=target_w,
                target_h=target_h,
                mode=args.mode,
                color_bgr=color_bgr,
                scale_to_fit=args.scale_to_fit,
            )

            if result.shape[1] != target_w or result.shape[0] != target_h:
                result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            writer.write(result)
            processed += 1
    finally:
        cap.release()
        writer.release()

    print(
        f"Wrote outpainted video: {out_path} | frames={processed}/{frame_count} | mode={args.mode} | size={target_w}x{target_h} | fps={out_fps:.2f}"
    )


if __name__ == "__main__":
    main()
