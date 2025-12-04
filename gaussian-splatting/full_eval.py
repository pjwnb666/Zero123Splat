#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

#python full_eval.py -m360 /root/autodl-tmp/datasets/Mipnerf --output_path /root/autodl-tmp/Mipnerf_eval
import os
from argparse import ArgumentParser
import time

import os
import time
from argparse import ArgumentParser

#mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
#mipnerf360_indoor_scenes  = ["room", "counter", "kitchen", "bonsai"]
#views_list = ["4views", "8views", "12views"]
#mipnerf360_outdoor_scenes = ["bicycle","bonsai","counter","flowers","garden","kitchen","room","stump","treehill"]
mipnerf360_outdoor_scenes = ["bicycle"]
views_list = ["4views", "8views", "12views"]
parser = ArgumentParser(description="Full evaluation for mipnerf360 (multi-views)")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval", type=str)
args, _ = parser.parse_known_args()

#all_scenes = mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes
all_scenes = mipnerf360_outdoor_scenes
scene_view_pairs = [(s, v) for s in all_scenes for v in views_list]

# 只有训练或渲染需要数据路径
if not args.skip_training or not args.skip_rendering:
    parser.add_argument("--mipnerf360", "-m360", required=True, type=str)
    args = parser.parse_args()

def pj(*xs):  # 安全拼路径并加引号
    return '"' + os.path.join(*xs) + '"'

# ---------------------------
# 1) 训练
# ---------------------------
if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 "
    t0 = time.time()
    for scene, view in scene_view_pairs:
        source_dir = os.path.join(args.mipnerf360, scene, view)           # <data>/<scene>/<view>
        model_dir  = os.path.join(args.output_path, "mipnerf360", scene, view)  # <out>/mipnerf360/<scene>/<view>
        os.makedirs(model_dir, exist_ok=True)

        cmd = f"python train.py -s {pj(source_dir)} -i images -m {pj(model_dir)} {common_args}"
        print(f"[TRAIN] {scene}/{view}\n  {cmd}")
        os.system(cmd)

    mins = (time.time() - t0) / 60.0
    print(f"[mipnerf360] training time: {mins:.2f} min ({mins/60.0:.2f} h)")

# ---------------------------
# 2) 渲染
# ---------------------------
if not args.skip_rendering:
    common_args = " --quiet --eval --skip_train"
    for scene, view in scene_view_pairs:
        source_dir = os.path.join(args.mipnerf360, scene, view)
        model_dir  = os.path.join(args.output_path, "mipnerf360", scene, view)

        # 以 30000 iter 结果渲染；如需改迭代数，改这里
        cmd = f"python render.py --iteration 30000 -s {pj(source_dir)} -m {pj(model_dir)} {common_args}"
        print(f"[RENDER] {scene}/{view}\n  {cmd}")
        os.system(cmd)

# ---------------------------
# 3) 评估（metrics.py）
# ---------------------------
if not args.skip_metrics:
    # metrics.py -m 需要传入一组包含 train/ 的模型目录；我们把每个 <out>/mipnerf360/<scene>/<view> 都塞进去
    model_roots = [os.path.join(args.output_path, "mipnerf360", scene, view) for scene, view in scene_view_pairs]
    scenes_string = " ".join('"' + p + '"' for p in model_roots)

    cmd = f"python metrics.py -m {scenes_string}"
    print(f"[METRICS]\n  {cmd}")
    os.system(cmd)


