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
from pipeline.system_utils import searchForMaxIteration
from deps.scene.dataset_readers import sceneLoadTypeCallbacks
from deps.scene.gaussian_model import GaussianModel
from deps.arguments import ModelParams
from pipeline.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, shuffle=True, resolution_scales=[1.0], mode='N'):
        """
        :param path: Path to colmap scene main folder.
        """
        if not mode == 'N':
            args_source_path = os.path.join(args.source_path, mode)

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args_source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args_source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args_source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args_source_path, args.white_background, args.depths, args.eval)
        else:
            print('args_source_path: ', args_source_path)
            assert False, "Could not recognize scene type!"
            
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        self.saved_scene_info = scene_info

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class StandardScene:
    def __init__(self, args : ModelParams, shuffle=True, resolution_scales=[1.0], mode='N'):
        """
        :param path: Path to colmap scene main folder.
        """
        if not mode == 'N':
            args_source_path = os.path.join(args.source_path, mode)

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args_source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args_source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args_source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args_source_path, args.white_background, args.depths, args.eval)
        else:
            print('args_source_path: ', args_source_path)
            assert False, "Could not recognize scene type!"
            
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        self.saved_scene_info = scene_info

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
