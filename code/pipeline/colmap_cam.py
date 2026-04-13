import os
import numpy as np
from PIL import Image

from deps.scene.colmap_loader import qvec2rotmat
from pipeline.graphics_utils import focal2fov
from deps.scene.dataset_readers import CameraInfo
from deps.scene.cameras import Camera as Camera_COLMAP

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=None,
                              image_path=image_path, image_name=image_name, depth_path=None,
                              width=width, height=height, is_test=False)
        cam_infos.append(cam_info)

    return cam_infos

from deps.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text
def readColmapCameraInfo(path):
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, 'images'))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    return cam_infos

def loadCam(id, cam_info):        
    image = Image.open(cam_info.image_path)
    orig_w, orig_h = image.size

    resolution = round(orig_w), round(orig_h)

    return Camera_COLMAP(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=None,
                  image_name=cam_info.image_name, uid=id, data_device='cpu',
                  train_test_exp=False, is_test_dataset=False, is_test_view=False)

def cameraList_from_camInfos(cam_infos):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(id, c))
    return camera_list

import random
def load_colmap_cameras(path, random_sample=False, sample_num=10):
    cam_infos = readColmapCameraInfo(path)
    if random_sample:
        cam_infos = random.sample(cam_infos, sample_num)
    return cameraList_from_camInfos(cam_infos)

