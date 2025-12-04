#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#export PYTHONPATH=/root/Zero123Splat:$PYTHONPATH
#python full_eval_hair.py -hair /root/autodl-tmp/datasets/Hair --output_path /root/autodl-tmp/hair_eval
import os
from argparse import ArgumentParser
import time

Hair_index_start = 351
Hair_index_end = 413
hair_scenes = [str(i) for i in range(Hair_index_start,Hair_index_end+1)]

#tanks_and_temples_scenes = ["truck", "train"]
#deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--hair_dataset", "-hair",type=str,help="Root folder containing Hair subfolders (e.g. 1, 2, 3, ...)")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(hair_scenes)
#all_scenes.extend(tanks_and_temples_scenes)
#all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    #parser.add_argument('--hair_dataset', "-hair", required=True, type=str)
    #parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    #parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()


if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 "
    start_time = time.time()
    for scene in all_scenes:
        source = args.hair_dataset + "/" + scene
        os.system("python train.py -s " + source + " -i images -m " + args.output_path + "/" + scene + common_args)
    hair_timing = (time.time() - start_time)/60.0
    print(f"[Hair] training time: min ({hair_timing/60:.2f} h)")
    #for scene in tanks_and_temples_scenes:
       #source = args.tanksandtemples + "/" + scene
        #os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    #for scene in deep_blending_scenes:
    #    source = args.deepblending + "/" + scene
    #    os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in all_scenes:
        all_sources.append(args.hair_dataset + "/" + scene)
    #for scene in tanks_and_temples_scenes:
    #    all_sources.append(args.tanksandtemples + "/" + scene)
    #for scene in deep_blending_scenes:
    #    all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_test"
    for scene, source in zip(all_scenes, all_sources):
        #os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
