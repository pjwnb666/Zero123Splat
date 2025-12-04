# split_and_flatten.py
import os, argparse
from typing import Tuple
from PIL import Image
import numpy as np
import cv2


# ---------- 基础工具 ----------
def tight_crop_rgba(img: Image.Image, thresh: int = 0) -> Image.Image:
    """基于 alpha 去掉全透明外边；thresh ∈ [0,255]"""
    if img.mode != "RGBA":
        return img
    a = np.array(img.split()[-1])
    ys, xs = np.where(a > thresh)
    if ys.size == 0 or xs.size == 0:
        return img  # 没有有效前景
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img.crop((x0, y0, x1, y1))


def estimate_bg_color_auto(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    从透明边界邻域估计一个纯色背景（用于 bg_mode='auto'）
    rgb: HxWx3, uint8; mask: HxW, {0,255}
    """
    inv = (mask == 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    border = cv2.morphologyEx(inv, cv2.MORPH_GRADIENT, k)
    if border.sum() == 0:
        return np.array([255, 255, 255], dtype=np.uint8)  # 回退白底
    ys, xs = np.where(border > 0)
    return rgb[ys, xs].mean(axis=0).astype(np.uint8)


def flatten_rgba(
    im_rgba: Image.Image,
    bg_mode: str = "auto",    # "white" | "auto" | "none"
    feather_px: int = 0,       # 边缘羽化（0关闭）
    shrink_px: int = 1,        # 边缘收缩（0关闭）
    tight_crop: bool = True,   # 最终是否裁去透明边
) -> Tuple[Image.Image, bool]:
    """
    将 RGBA 转为干净 RGB（或保持 RGBA）：
      - bg_mode='white'  合成纯白底，输出 RGB
      - bg_mode='auto'   自适应纯色底，输出 RGB
      - bg_mode='none'   不合成，保持 RGBA（但仍可做收缩/羽化/裁剪）

    返回：(PIL.Image, is_rgb)
    """
    im_rgba = im_rgba.convert("RGBA")
    arr = np.array(im_rgba)
    rgb = arr[..., :3]
    a   = arr[..., 3]
    mask = (a > 0).astype(np.uint8) * 255

    # 先收缩再羽化，消除灰黑毛边
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_px * 2 + 1, shrink_px * 2 + 1))
        mask = cv2.erode(mask, k, iterations=1)
    if feather_px > 0:
        mask_f = cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), 0)
    else:
        mask_f = mask

    if bg_mode.lower() == "none":
        # 仅更新 alpha（羽化/收缩后的），保持 RGBA
        out_a = mask_f
        out = np.dstack([rgb, out_a]).astype(np.uint8)
        out_img = Image.fromarray(out, mode="RGBA")
        if tight_crop:
            out_img = tight_crop_rgba(out_img, thresh=0)
        return out_img, False  # RGBA

    # 需要合成到纯色背景 → 输出 RGB
    if bg_mode.lower() == "white":
        bg_color = np.array([255, 255, 255], dtype=np.uint8)
    elif bg_mode.lower() == "auto":
        bg_color = estimate_bg_color_auto(rgb, mask)
    else:
        raise ValueError("bg_mode must be 'white' | 'auto' | 'none'")

    bg = np.ones_like(rgb, dtype=np.uint8) * bg_color
    alpha = (mask_f.astype(np.float32) / 255.0)[..., None]
    out = (rgb.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    out_img = Image.fromarray(out, mode="RGB")

    if tight_crop:
        ys, xs = np.where(mask > 0)
        if ys.size > 0 and xs.size > 0:
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            out_img = out_img.crop((x0, y0, x1, y1))

    return out_img, True  # RGB


# ---------- 主流程：切图 + 背景处理 ----------
def split_and_process(
    in_path: str,
    #out_dir_inject_input: str,
    out_dir_inject_images: str,
    out_dir_origin: str,
    rows: int,
    cols: int,
    crop_alpha: bool = True,
    alpha_thresh: int = 0,
    pad: int = 0,
    basename: str = "view",
    bg_mode: str = "white",    # "white" | "auto" | "none"
    feather_px: int = 2,
    shrink_px: int = 1,
    final_tight_crop: bool = True,
):
    """
    - 将拼图按 rows x cols 切块
    - 可选：先对每块去透明边（tight crop）
    - 背景处理：
        white: 合成白底 → 输出RGB
        auto : 自适应纯色底 → 输出RGB
        none : 不合成 → 输出RGBA
    - pad: 在切块后额外加全透明留白像素（仅在 bg_mode='none' 或想留边时有用）
    """
    os.makedirs(out_dir_origin, exist_ok=True)
    sheet = Image.open(in_path).convert("RGBA")
    W, H = sheet.size
    tile_w, tile_h = W // cols, H // rows
    skip_tiles = {1, 4}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            tile_idx = r * cols + c
            if tile_idx in skip_tiles:
                continue
            x0, y0 = c * tile_w, r * tile_h
            tile = sheet.crop((x0, y0, x0 + tile_w, y0 + tile_h))  # RGBA

            if crop_alpha:
                tile = tight_crop_rgba(tile, thresh=alpha_thresh)

            # 透明留白（若需要）
            if pad > 0:
                w, h = tile.size
                canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
                canvas.paste(tile, (pad, pad))
                tile = canvas

            # 背景处理
            out_img, is_rgb = flatten_rgba(
                tile,
                bg_mode=bg_mode,
                feather_px=feather_px,
                shrink_px=shrink_px,
                tight_crop=final_tight_crop,
            )
            
            # 统一保存为 .png；RGB或RGBA由 is_rgb 决定
            #out_inject_input_path = os.path.join(out_dir_inject_input, f"{basename}_{idx:02d}.png")
            out_inject_images_path = os.path.join(out_dir_inject_images, f"{basename}_{idx:02d}.png")
            out_origin_path = os.path.join(out_dir_origin, f"{basename}_{idx:02d}.png")
            #out_img.save(out_inject_input_path)
            out_img.save(out_inject_images_path)
            out_img.save(out_origin_path)
            idx += 1

    #print(f"Done. Saved {idx} tiles to {out_dir_inject_input} (bg_mode={bg_mode})")
    print(f"Done. Saved {idx} tiles to {out_dir_inject_images} (bg_mode={bg_mode})")
    print(f"Done. Saved {idx} tiles to {out_dir_origin} (bg_mode={bg_mode})")


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Split multi-view sheet and optionally flatten background.")
    ap.add_argument("--in", dest="in_path", required=True, help="输入拼图路径（PNG）")
    ap.add_argument("--out", dest="out_dir", default="views_out", help="输出目录")
    ap.add_argument("--rows", type=int, required=True, help="行数")
    ap.add_argument("--cols", type=int, required=True, help="列数")
    ap.add_argument("--no-crop", action="store_true", help="不去透明边（tight crop）")
    ap.add_argument("--alpha-thresh", type=int, default=0, help="alpha阈值(0-255)，>阈值视为前景")
    ap.add_argument("--pad", type=int, default=0, help="切块后四周额外透明留白像素")
    ap.add_argument("--basename", default="view", help="输出文件名前缀")
    ap.add_argument("--bg", dest="bg_mode", default="white", choices=["white", "auto", "none"],
                    help="背景处理模式：white=白底RGB, auto=自适应纯色底RGB, none=保留RGBA")
    ap.add_argument("--feather", type=int, default=2, help="边缘羽化像素（0关闭）")
    ap.add_argument("--shrink", type=int, default=1, help="边缘收缩像素（0关闭）")
    ap.add_argument("--final-no-crop", action="store_true", help="背景处理后不再做最终 tight crop")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_and_process(
        in_path=args.in_path,
        out_dir=args.out_dir,
        rows=args.rows,
        cols=args.cols,
        crop_alpha=(not args.no_crop),
        alpha_thresh=args.alpha_thresh,
        pad=args.pad,
        basename=args.basename,
        bg_mode=args.bg_mode,
        feather_px=args.feather,
        shrink_px=args.shrink,
        final_tight_crop=(not args.final_no_crop),
    )
