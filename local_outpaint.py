#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# NOTE: AI mode dependencies (mmagic/mmedit, mmcv, mmengine, PIL, torch) are imported lazily only when used


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
    import cv2
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


def build_init_and_mask(
    frame_bgr: np.ndarray,
    target_w: int,
    target_h: int,
) -> Tuple["Image.Image", "Image.Image"]:
    import cv2
    # Lazy import PIL
    from PIL import Image

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

def load_mmagic_inpainter(
    device_pref: str,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """Load MMEditing/MMagic DeepFillv2 inpainter for AI mode.

    Downloads default config/checkpoint if not provided.
    """
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    # Lazy imports
    try:
        from mmagic.apis import init_model, inpainting_inference  # type: ignore
    except Exception:
        try:
            # Legacy package name fallback
            from mmedit.apis import init_model, inpainting_inference  # type: ignore
        except Exception:
            die(
                "AI mode requires 'mmagic' (or 'mmedit') with mmcv/mmengine.\n"
                "Install: pip install mmagic mmcv mmengine Pillow torch torchvision"
            )

    if device_pref == "auto":
        device = "cuda" if (torch is not None and getattr(torch, "cuda").is_available()) else "cpu"
    else:
        device = device_pref

    # Defaults for DeepFillv2
    if not config_path:
        config_path = (
            "https://raw.githubusercontent.com/open-mmlab/mmagic/main/configs/inpainting/deepfillv2/"
            "deepfillv2_256x256_8xb8_places.py"
        )
    if not checkpoint_path:
        checkpoint_path = (
            "https://download.openmmlab.com/mmediting/inpainting/deepfillv2/"
            "deepfillv2_256x256_8xb8_places_20200619-16250d4f.pth"
        )

    try:
        model = init_model(config_path, checkpoint_path, device=device)
    except Exception as e:
        die(
            "Failed to initialize MMagic inpainting model. Ensure mmagic/mmcv/mmengine versions are compatible.\n"
            f"Details: {e}"
        )

    def infer(img_rgb: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
        # mmagic typically expects BGR image; convert
        img_bgr = img_rgb[:, :, ::-1]
        # mask expected uint8 with 255 indicating holes
        result_bgr = inpainting_inference(model, img_bgr, mask_gray)
        # Some versions return dict or ndarray
        if isinstance(result_bgr, dict):
            # common key names
            if "pred_img" in result_bgr:
                result_bgr = result_bgr["pred_img"]
            elif "output" in result_bgr:
                result_bgr = result_bgr["output"]
        result_rgb = result_bgr[:, :, ::-1]
        return result_rgb

    return infer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local video outpainting without external APIs: expand canvas to target width/height "
            "using mirror/replicate/blur/solid backgrounds, or generate with AI inpainting."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Path to output video file (.mp4 recommended)")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target output width in pixels (default: source width)",
    )
    parser.add_argument("--height", type=int, required=True, help="Target output height in pixels")
    parser.add_argument(
        "--mode",
        choices=["mirror", "replicate", "blur", "solid", "ai"],
        default="blur",
        help="Outpainting mode: simple backgrounds or 'ai' to synthesize new content",
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

    # AI mode options (MMagic DeepFillv2)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for AI mode (MMagic)",
    )
    parser.add_argument("--ai-config", default=None, help="Override MMagic config URL/path (mode=ai)")
    parser.add_argument("--ai-checkpoint", default=None, help="Override checkpoint URL/path (mode=ai)")

    args = parser.parse_args()
    import cv2

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        die(f"Input not found: {in_path}")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_h = int(args.height)
    if target_h <= 0:
        die("--height must be a positive integer")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        die(f"Failed to open video: {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    # Resolve target width now that we know the source size
    target_w = int(args.width) if args.width is not None else src_w
    if target_w <= 0:
        die("--width must be a positive integer when specified")

    color_bgr = parse_rgb_color(args.color)

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

    # Prepare AI model only if needed
    mmagic_infer = None
    if args.mode == "ai":
        try:
            mmagic_infer = load_mmagic_inpainter(
                device_pref=args.device,
                config_path=args.ai_config,
                checkpoint_path=args.ai_checkpoint,
            )
        except Exception as e:
            cap.release()
            writer.release()
            die(str(e))

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if args.mode == "ai":
                # Build init and mask, then run MMagic inpainting (DeepFillv2)
                init_image, mask_image = build_init_and_mask(
                    frame_bgr=frame,
                    target_w=target_w,
                    target_h=target_h,
                )
                try:
                    init_np = np.array(init_image)  # RGB
                    mask_np = np.array(mask_image)  # L (0/255)
                    result_rgb = mmagic_infer(init_np, mask_np)
                except Exception as e:
                    cap.release()
                    writer.release()
                    die(f"MMagic inpainting failed: {e}")
                result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            else:
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
