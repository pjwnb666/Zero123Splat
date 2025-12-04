# split_views.py
import os, argparse
from PIL import Image
import numpy as np

def tight_crop_rgba(img: Image.Image, thresh=0):  # 去透明边；阈值0~255
    if img.mode != "RGBA":
        return img
    a = np.array(img.split()[-1])
    ys, xs = np.where(a > thresh)
    if ys.size == 0 or xs.size == 0:
        return img  # 全透明/全不透明就不裁
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    return img.crop((x0, y0, x1, y1))

def split_grid(
    in_path, out_dir, rows, cols,
    crop_alpha=True, alpha_thresh=0,  # 去透明边
    pad=0,                            # 额外留白像素
    basename="view"
):
    os.makedirs(out_dir, exist_ok=True)
    sheet = Image.open(in_path).convert("RGBA")  # 保留透明
    W, H = sheet.size
    tile_w, tile_h = W // cols, H // rows

    idx = 0
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c*tile_w, r*tile_h
            tile = sheet.crop((x0, y0, x0+tile_w, y0+tile_h))
            if crop_alpha:
                tile = tight_crop_rgba(tile, thresh=alpha_thresh)
            if pad > 0:
                # 可选：四周加纯透明留白
                w, h = tile.size
                canvas = Image.new("RGBA", (w+2*pad, h+2*pad), (0,0,0,0))
                canvas.paste(tile, (pad, pad))
                tile = canvas
            # 命名顺序：按 行优先（top-left 到 bottom-right）
            out = os.path.join(out_dir, f"{basename}_{idx:02d}.png")
            tile.save(out)
            idx += 1
    print(f"Done. Saved {idx} tiles to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="六连图路径")
    ap.add_argument("--out", dest="out_dir", default="views_out", help="输出目录")
    ap.add_argument("--rows", type=int, required=True, help="行数")
    ap.add_argument("--cols", type=int, required=True, help="列数")
    ap.add_argument("--no-crop", action="store_true", help="不去透明边")
    ap.add_argument("--alpha-thresh", type=int, default=0, help="alpha阈值(0-255)")
    ap.add_argument("--pad", type=int, default=0, help="四周留白像素")
    ap.add_argument("--basename", default="view", help="输出文件前缀")
    args = ap.parse_args()

    split_grid(
        args.in_path, args.out_dir, args.rows, args.cols,
        crop_alpha=(not args.no_crop),
        alpha_thresh=args.alpha_thresh,
        pad=args.pad,
        basename=args.basename
    )
