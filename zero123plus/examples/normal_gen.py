import cv2
import copy
import re
from pathlib import Path
import numpy as np
import torch
import requests
import os
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel
from .matting_postprocess import postprocess
from rembg import remove

#def rescale(single_res, input_image, ratio=0.95):
    # Rescale and recenter
    #image_arr = numpy.array(input_image)
    #ret, mask = cv2.threshold(numpy.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    #x, y, w, h = cv2.boundingRect(mask)
    #max_size = max(w, h)
    #side_len = int(max_size / ratio)
    #padded_image = numpy.zeros((side_len, side_len, 4), dtype=numpy.uint8)
    #center = side_len//2
    #padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    #rgba = Image.fromarray(padded_image).resize((single_res, single_res), Image.LANCZOS)
    #return rgba

def rescale(single_res: int, input_image: Image.Image, ratio: float = 1.0) -> Image.Image:
    """
    将前景裁剪为紧致框，居中贴到方形画布，再缩放到 single_res。
    - 兼容 RGB / RGBA 输入
    - 优先使用 alpha 作为掩码；若没有 alpha，则用灰度 + OTSU 得到掩码
    """
    # 统一转成 RGBA，避免 3 通道→4 通道的广播问题
    img_rgba = input_image.convert("RGBA")
    img_np = np.array(img_rgba)                # (H, W, 4)

    # 取得掩码：优先用 alpha；如果 alpha 全 255（无抠图），就用 OTSU 从灰度估个掩码
    alpha = img_np[..., 3]
    if np.all(alpha == 255):
        gray = cv2.cvtColor(img_np[..., :3], cv2.COLOR_RGB2GRAY)
        # 用 OTSU 自动阈值；针对浅色背景可反相
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 如果前景仍然太小或几乎全 0，则退化为全图
        if mask.sum() < 1000:
            mask = np.ones_like(gray, dtype=np.uint8) * 255
    else:
        mask = alpha

    # 紧致外接矩形
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = max(1, int(max_size / ratio))

    # 方形画布（RGBA）
    canvas = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    cx = side_len // 2

    # 贴前景到画布中心
    crop = img_np[y:y+h, x:x+w, :]             # (h, w, 4)
    ch, cw = crop.shape[:2]
    ys, xs = cx - ch // 2, cx - cw // 2
    canvas[ys:ys+ch, xs:xs+cw] = crop

    # 统一缩放到 single_res
    out = Image.fromarray(canvas).resize((single_res, single_res), Image.LANCZOS)
    return out

def _natsort_key(name: str):
    # 按自然序排序：x1, x2, x10 -> 1,2,10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

# Load the pipeline
def _first_image_in(path: Path) -> Path:
    """传入目录或图片路径；返回第一张图片的路径"""
    if path.is_file():
        return path
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not candidates:
        raise FileNotFoundError(f"No image found in: {path}")
    candidates.sort(key=lambda p: _natsort_key(p.name))
    return candidates[0]

# Load the pipeline
def run_zero123plus(zp_path : str, 
                cust_path: str,
                cn_path: str,
                cond_path: Path, 
                color_save_path: Path):

    ZP_DIR  = "/root/autodl-tmp/models/zero123plus/zero123plus-v1.2"
    CUST_DIR= "//root/autodl-tmp/models/zero123plus/zero123plus-pipeline"
    CN_DIR  = "/root/autodl-tmp/models/zero123plus/controlnet-zp12-normal-gen-v1"
    pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
        ZP_DIR, custom_pipeline=CUST_DIR,
        torch_dtype=torch.float16, local_files_only=True
    )
    normal_pipeline = copy.copy(pipeline)
    normal_pipeline.add_controlnet(ControlNetModel.from_pretrained(
        CN_DIR , torch_dtype=torch.float16, local_files_only=True
    ), conditioning_scale=1.0)
    pipeline.to("cuda:0", torch.float16)
    normal_pipeline.to("cuda:0", torch.float16)
    # Run the pipeline
    #cond = Image.open("/root/Zero123Splat/data/images/frame_000000 (1).jpg")
    first_img_path = _first_image_in(cond_path)
    cond = Image.open(first_img_path)
    # Optional: rescale input image if it occupies only a small region in input
    cond = rescale(512, cond)
    # Generate 6 images
    genimg = pipeline(
        cond,
        prompt='', guidance_scale=6, num_inference_steps=90, width=640, height=960
    ).images[0]
    # Generate normal image
    # We observe that a higher CFG scale (4) is more robust
    # but with CFG = 1 it is faster and is usually good enough for normal image
    # You can adjust to your needs
    genimg = remove(genimg)
    normalimg = normal_pipeline(
        cond, depth_image=genimg,
        prompt='', guidance_scale=6, num_inference_steps=90, width=640, height=960
    ).images[0]
    #genimg, normalimg = postprocess(genimg, normalimg)
    #genimg = remove(genimg)
    genimg.save(os.path.join(color_save_path,"colors.png"))
    colors_path = os.path.join(color_save_path,"colors.png")
    return colors_path
    #normalimg.save("normals.png")
