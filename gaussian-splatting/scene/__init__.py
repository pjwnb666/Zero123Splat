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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            os.makedirs(self.model_path, exist_ok=True)
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        #return exposure_dict

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
        
    def _known_image_names(self):
        names = set()
        for cams in self.train_cameras.values():
            for c in cams:
                names.add(c.image_name)
        for cams in self.test_cameras.values():
            for c in cams:
                names.add(c.image_name)
        return names
    
    def extend_from_source(self, source_path: str, args: ModelParams, is_test: bool=False, resolution_scales=[1.0]):
        if os.path.exists(os.path.join(source_path, "sparse")):
            # 与构造时同一回调：再次从 COLMAP 读取
            scene_info_new = sceneLoadTypeCallbacks["Colmap"](source_path, args.images, args.depths, args.eval,
                                                              args.train_test_exp)
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            scene_info_new = sceneLoadTypeCallbacks["Blender"](source_path, args.white_background, args.depths,
                                                               args.eval)
        else:
            raise RuntimeError(f"extend_from_source: Unrecognized scene type at {source_path}")

            # 只取“训练相机”或“测试相机”
        src_caminfos = scene_info_new.test_cameras if is_test else scene_info_new.train_cameras

        # 去重：基于 image_name（dataset_readers 中会设置）
        known = self._known_image_names()
        new_caminfos = []
        for ci in src_caminfos:
            # CamInfo 通常有字段 image_name；若无可 fallback 到 basename(image_path)
            name = getattr(ci, "image_name", None)
            if name is None and hasattr(ci, "image_path"):
                name = os.path.basename(ci.image_path)
                setattr(ci, "image_name", name)
            if name not in known:
                new_caminfos.append(ci)
                known.add(name)

        if len(new_caminfos) == 0:
            print("[Scene] No new cameras/images found to append.")
            return 0

        # 可选：打乱以与最初加载策略一致（多分辨率一致随机），这里保留“追加顺序”即可
        # random.shuffle(new_caminfos)

        target_dict = self.test_cameras if is_test else self.train_cameras
        for scale in resolution_scales:
            # 复用现有工厂函数构造 Camera 列表
            cam_list_new = cameraList_from_camInfos(new_caminfos, scale, args,
                                                    scene_info_new.is_nerf_synthetic, is_test)
            if scale not in target_dict:
                target_dict[scale] = []
            # 关键：顺序追加到尾部
            target_dict[scale].extend(cam_list_new)

        print(f"[Scene] Appended {len(new_caminfos)} new {'test' if is_test else 'train'} cameras from {source_path}")
        #return len(new_caminfos)
        return src_caminfos
        