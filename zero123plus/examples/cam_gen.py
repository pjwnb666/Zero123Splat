#!/usr/bin/env python3
"""
Fast-3-R → COLMAP (plus PLY) with optional APPEND mode.

Usage (新追加到现有 sparse):
python cam_gen.py \
  --image_dir /root/Zero123Splat/zero123plus/examples/views_rgb \
  --output_dir /root/Zero123Splat/outputs \
  --checkpoint_dir /root/Zero123Splat/checkpoints/Fast3R_ViT_Large_512 \
  --image_size 512 \
  --keep_percent 0.1 \
  --sparse_dir /root/Zero123Splat/outputs/sparse/0 \
  --append \
  --name_prefix ZP6_
  

Usage (写到新的 sparse_0/0):
python cam_gen.py \
  --image_dir /root/Zero123Splat/zero123plus/examples/views_rgb \
  --output_dir /path/to/outputs \
  --checkpoint_dir /path/to/checkpoint \
  --image_size 512 \
  --keep_percent 0.1 \
  --sparse_dir /root/Zero123Splat/gaussian-splatting-main/data/sparse/0
"""

import os
import struct
import torch
import numpy as np
from pathlib import Path
import argparse

from fast3r.dust3r.inference_multiview import inference
from fast3r.utils.checkpoint_utils import load_model
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

from utils.sfm_utils import init_filestructure, load_images
from scene.colmap_loader import (
    Camera as ColmapCamera,
    BaseImage as ColmapImage,
    write_cameras_binary,
    write_cameras_text,
    write_images_binary,
    write_images_text,
)
from scene.colmap_loader import Point3D, write_points3D_binary
from scene.colmap_loader import rotmat2qvec

ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ---------------- COLMAP binary IO helpers ----------------
def read_cameras_binary(bin_path: str):
    cams = {}
    with open(bin_path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id   = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width    = struct.unpack("<Q", f.read(8))[0]
            height   = struct.unpack("<Q", f.read(8))[0]
            fx, fy, cx, cy = struct.unpack("<4d", f.read(32))
            cams[cam_id] = dict(
                model_id=model_id, width=width, height=height,
                fx=fx, fy=fy, cx=cx, cy=cy
            )
    return cams

def read_images_binary(bin_path: str):
    imgs = {}
    with open(bin_path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            # name (null-terminated)
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # points2D skipped (assumed 0)
            imgs[image_id] = dict(
                qvec=qvec, tvec=tvec, camera_id=camera_id, name=name, num_points2D=num_points2D
            )
    return imgs

def write_cameras_binary_raw(bin_path: str, cams: dict):
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cam_id in sorted(cams.keys()):
            c = cams[cam_id]
            f.write(struct.pack("<i", cam_id))
            f.write(struct.pack("<i", c["model_id"]))
            f.write(struct.pack("<Q", int(c["width"])))
            f.write(struct.pack("<Q", int(c["height"])))
            f.write(struct.pack("<4d", float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"])))

def write_images_binary_raw(bin_path: str, imgs: dict):
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for image_id in sorted(imgs.keys()):
            im = imgs[image_id]
            f.write(struct.pack("<i", image_id))
            f.write(struct.pack("<4d", *[float(x) for x in im["qvec"]]))
            f.write(struct.pack("<3d", *[float(x) for x in im["tvec"]]))
            f.write(struct.pack("<i", int(im["camera_id"])))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", int(im.get("num_points2D", 0))))

# ---------------- utility ----------------
def save_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    data = np.empty(len(points), dtype=[("xyz", np.float32, 3), ("rgb", np.uint8, 3)])
    data["xyz"] = points.astype(np.float32)
    data["rgb"] = colors.astype(np.uint8)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())

def save_points3dbin(bin_path: Path, points: np.ndarray, colors: np.ndarray, error: float = 0.0):
    pts3D = {}
    for i in range(len(points)):
        pts3D[i] = Point3D(
            id=i,
            xyz=points[i].astype(np.float64),
            rgb=colors[i].astype(np.uint8),
            error=error,
            image_ids=np.empty((0,), dtype=np.int64),
            point2D_idxs=np.empty((0,), dtype=np.int64),
        )
    write_points3D_binary(pts3D, str(bin_path))
    print(f"Wrote {len(points):,} points to {bin_path}")
    return len(points)

def save_cameras_bin_txt(out_dir: Path, intrinsics: np.ndarray, width: int, height: int):
    cams = {}
    for i, K in enumerate(intrinsics, start=1):
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        cams[i] = ColmapCamera(
            id=i,
            model="PINHOLE",
            width=width,
            height=height,
            params=[fx, fy, cx, cy],
        )
    write_cameras_binary(cams, out_dir / "cameras.bin")
    write_cameras_text(cams, out_dir / "cameras.txt")
    print(f"Wrote cameras.bin / cameras.txt → {out_dir}")

def save_images_bin_txt(out_dir: Path, extrinsics_w2c: np.ndarray, image_files: list[str]):
    imgs = {}
    from scene.colmap_loader import rotmat2qvec
    for i, (w2c, path) in enumerate(zip(extrinsics_w2c, image_files), start=1):
        R = w2c[:3, :3].astype(np.float64)
        t = w2c[:3, 3].astype(np.float64)
        qvec = rotmat2qvec(R)
        imgs[i] = ColmapImage(
            id=i,
            qvec=qvec,
            tvec=t,
            camera_id=i,
            name=Path(path).name,
            xys=[],
            point3D_ids=[]
        )
    write_images_binary(imgs, out_dir / "images.bin")
    write_images_text(imgs, out_dir / "images.txt")
    print(f"Wrote images.bin / images.txt → {out_dir}")

# ---------------- main pipeline ----------------
def process_directory(
    image_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    sparse_dir: Path,
    image_size: int = 512,
    keep_percent: float = 0.1,
    append: bool = False,
    name_prefix: Path = "",
):
    assert 0 < keep_percent <= 1
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path, sparse_0_path, _ = init_filestructure(output_dir)

    # 读取 & 排序图像（顺序将贯穿整个流程）
    all_paths = sorted([str(p) for p in image_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS])
    if len(all_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f">> Loading {len(all_paths)} images from {image_dir}")
    imgs, (orgW, orgH) = load_images(all_paths, size=image_size)
    image_files = all_paths

    # 读取“真实相机”的内参（取第一个相机为基准）
    cams_real = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    if len(cams_real) == 0:
        raise FileNotFoundError(f"No camera found in {sparse_dir/'cameras.bin'}")
    cam0 = cams_real[sorted(cams_real.keys())[0]]
    fx0, fy0, cx0, cy0 = cam0["fx"], cam0["fy"], cam0["cx"], cam0["cy"]
    W0, H0 = cam0["width"], cam0["height"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, lit_module = load_model(checkpoint_dir, device=device)

    print(">> Running DUSt3R inference...")
    output_dict, _ = inference(imgs, model, device, dtype=torch.float32, verbose=True, profiling=True)

    print(">> Aligning global points...")
    lit_module.align_local_pts3d_to_global(
        output_dict["preds"], output_dict["views"], min_conf_thr_percentile=85
    )

    print(">> Estimating camera poses (DUSt3R)...")
    poses_c2w_batch, _ = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict["preds"], niter_PnP=100, focal_length_estimation_method="first_view_from_global_head"
    )
    poses_c2w = poses_c2w_batch[0]
    extrinsics_w2c = np.stack([np.linalg.inv(p) for p in poses_c2w])

    # 收集点云用于预览（非必须）
    pts, cols = [], []
    for pred, view in zip(output_dict["preds"], output_dict["views"]):
        p = pred["pts3d_in_other_view"].reshape(-1, 3)
        imgs_rgb = ((view["img"].squeeze().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        c = imgs_rgb.reshape(-1, 3)
        pts.append(p); cols.append(c)
    pts3d  = np.concatenate(pts, axis=0)
    colors = np.concatenate(cols, axis=0)
    N = len(pts3d)
    sel = np.random.choice(N, size=max(1, int(N * keep_percent)), replace=False)
    pts3d, colors = pts3d[sel], colors[sel]
    print(f">> Randomly sampled {len(pts3d):,} / {N:,} points")

    # DUSt3R 实际分辨率
    H, W = output_dict["views"][0]["img"].shape[2:]
    # 将真实 K 按比例缩放到 (W,H)
    sx, sy = W / W0, H / H0
    fx = fx0 * sx; fy = fy0 * sy; cx = cx0 * sx; cy = cy0 * sy

    # === 写出 ===
    if not append:
        # 写到 output_dir/sparse_0/0 新目录
        print("→ Writing NEW sparse (no-append) ...")
        Ks = np.stack([np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float32) for _ in image_files], axis=0)
        print(f">> Saving intrinsics (scaled K) to new sparse")
        save_cameras_bin_txt(sparse_0_path, Ks, W, H)

        print(f">> Saving extrinsics to new sparse")
        save_images_bin_txt(sparse_0_path, extrinsics_w2c, image_files)

        print(f">> Saving points3D & PLY")
        save_points3dbin(sparse_0_path / "points3D.bin", pts3d, colors)
        #save_ply(output_dir / "reconstruction.ply", pts3d, colors)
        return sparse_0_path

    # ---------- 追加模式 ----------
    print("→ APPEND mode: merge into existing cameras.bin / images.bin")
    cams_old = cams_real.copy()
    imgs_old = read_images_binary(str(sparse_dir / "images.bin"))

    # 追加camera和 images
    next_img_id = (max(imgs_old.keys()) if len(imgs_old) > 0 else 0) + 1
    next_cid    = (max(cams_old.keys()) if len(cams_old) > 0 else 0) + 1
    for w2c, path in zip(extrinsics_w2c, image_files):
        # ---- 每张图新建一个 camera ----
        cams_old[next_cid] = dict(
            model_id=1,
            width=W,
            height=H,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )
        camera_id_for_new = next_cid

        # ---- 新建 image ----
        R = w2c[:3, :3].astype(np.float64)
        t = w2c[:3, 3].astype(np.float64)
        qvec = rotmat2qvec(R)
        base = Path(path).name
        new_name = f"{name_prefix}{base}" if name_prefix else base

        # 防止重名
        existing_names = set(v["name"] for v in imgs_old.values())
        name_final = new_name
        k = 1
        while name_final in existing_names:
            name_final = f"{name_prefix}{k}_{base}"
            k += 1

        imgs_old[next_img_id] = dict(
            qvec=qvec,
            tvec=t,
            camera_id=camera_id_for_new,
            name=name_final,
            num_points2D=0
        )

        print(f">> Created new camera_id={next_cid} for image_id={next_img_id} "
            f"(W={W},H={H}, fx={fx:.2f},fy={fy:.2f},cx={cx:.2f},cy={cy:.2f})")

        # ---- 自增 ----
        next_img_id += 1
        next_cid    += 1

    # 3) 回写原位置（覆盖写入）
    write_cameras_binary_raw(str(sparse_dir / "cameras.bin"), cams_old)
    write_images_binary_raw (str(sparse_dir / "images.bin"),  imgs_old)
    cams_txt = {
    cid: ColmapCamera(
        id=cid,
        model="PINHOLE",                           # 你的 raw 里用的就是 pinhole
        width=int(c["width"]),
        height=int(c["height"]),
        params=[float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"])],
    )
    for cid, c in cams_old.items()
}


    imgs_txt = {
        iid: ColmapImage(
            id=iid,
            qvec=np.array(im["qvec"], dtype=np.float64),
            tvec=np.array(im["tvec"], dtype=np.float64),
            camera_id=int(im["camera_id"]),
            name=str(im["name"]),
            xys=[],                  # 你没写 points2D，就传空
            point3D_ids=[],
        )
        for iid, im in imgs_old.items()
    }

    # 再写文本（Path 或 str 都行）
    write_cameras_text(cams_txt, sparse_dir / "cameras.txt")
    write_images_text (imgs_txt,  sparse_dir / "images.txt")
    
    print(f"[OK] Appended {len(image_files)} images into {sparse_dir}")
    print(f"    cameras.bin: {len(cams_old)} cameras")
    print(f"    images.bin : {len(imgs_old)} images (last id={next_img_id-1})")

    # 可选：另存点云 & 预览（不影响 sparse）
    #save_ply(output_dir / "reconstruction.ply", pts3d, colors)
    #return sparse_dir

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",  required=True, type=Path, help="输入图片目录（按文件名排序）")
    #p.add_argument("--output_dir", required=True, type=Path, help="输出根目录")
    p.add_argument("--checkpoint_dir", type=Path, default=".", help="DUSt3R checkpoint 目录")
    p.add_argument("--image_size", type=int, default=512, help="DUSt3R 输入尺寸")
    p.add_argument("--keep_percent", type=float, default=0.1, help="点云随机采样比例 (0,1]")
    p.add_argument("--sparse_dir", type=Path,
                   help="原始 COLMAP 模型目录（读取/覆盖 cameras.bin & images.bin）")
    p.add_argument("--append", action="store_true", help="将新相机/图像追加写入到现有 cameras.bin/images.bin")
    p.add_argument("--name_prefix", type=str, default="ZP6_", help="追加时为新图像名加的前缀，避免重名")
    args = p.parse_args()

    out = process_directory(
        args.image_dir,
        args.output_dir,
        args.checkpoint_dir,
        args.image_size,
        args.keep_percent,
        args.sparse_dir,
        args.append,
        args.name_prefix,
    )
    print("done →", out)

if __name__ == "__main__":
    main()
