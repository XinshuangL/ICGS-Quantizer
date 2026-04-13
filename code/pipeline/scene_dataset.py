import os
import random

import torch
from torch.utils.data import Dataset

from pipeline.colmap_cam import load_colmap_cameras


class DatasetAB(Dataset):
    @torch.no_grad()
    def __init__(
        self,
        data_root,
        scene_num=100,
        downsample_order=2,
        block_resolution=32,
        camera_names=None,
        six_cam_start_id=150,
        fixed_AB_mode="N",
    ):
        self.data_root = data_root
        self.downsample_order = downsample_order
        self.block_resolution = block_resolution
        self.camera_names = camera_names or ["0", "1", "2", "3", "4", "5"]
        self.six_cam_start_id = six_cam_start_id
        self.fixed_AB_mode = fixed_AB_mode
        self.valid_scene_ids = [
            scene_id
            for scene_id in range(scene_num)
            if os.path.exists(f"{data_root}/scene_{scene_id}/processed_data_{downsample_order}.pt")
        ]

        print(f"# available scenes: {len(self.valid_scene_ids)}")

    def __len__(self):
        return len(self.valid_scene_ids)

    def filter_cam(self, camera_dict):
        return {camera_name: camera_dict[camera_name] for camera_name in self.camera_names}

    def choose_modes(self):
        if self.fixed_AB_mode == "N":
            source_mode = random.choice(["A", "B"])
        else:
            source_mode = self.fixed_AB_mode
        target_mode = "A" if source_mode == "B" else "B"
        return source_mode, target_mode

    def load_dino_features(self, scene_id, mode):
        dino_features = {}
        for camera_name in self.camera_names:
            camera_id = int(camera_name) + self.six_cam_start_id
            camera_str = f"{camera_id:06d}"
            dino_features[camera_name] = torch.load(
                f"{self.data_root}/scene_{scene_id}/{mode}/dino_features/{camera_str}.pt"
            )
        return dino_features

    def __getitem__(self, index):
        scene_id = self.valid_scene_ids[index]
        scene_data = torch.load(f"{self.data_root}/scene_{scene_id}/processed_data_{self.downsample_order}.pt")

        scene_path = scene_data["scene_path"]
        source_mode, target_mode = self.choose_modes()
        target_cam = self.filter_cam(scene_data["AB_data_cam"][target_mode])
        source_cam = self.filter_cam(scene_data["AB_data_cam"][source_mode])

        target_dino_features = self.load_dino_features(scene_id, target_mode)
        source_dino_features = self.load_dino_features(scene_id, source_mode)
        for camera_name in self.camera_names:
            target_cam[camera_name]["dino_features"] = target_dino_features[camera_name]
            source_cam[camera_name]["dino_features"] = source_dino_features[camera_name]

        return {
            "scene_id": scene_data["scene_id"],
            "block_resolution": self.block_resolution,
            "source_mode": source_mode,
            "target_mode": target_mode,
            "source_3dgs": scene_data["AB_data_3dgs"][source_mode],
            "target_cam": target_cam,
            "source_cam": source_cam,
            "source_viewpoint_stack": load_colmap_cameras(f"{scene_path}/{source_mode}/novel_view/"),
            "target_viewpoint_stack": load_colmap_cameras(f"{scene_path}/{target_mode}/novel_view/"),
        }


def custom_collate_fn_test(data_list):
    data = data_list[0]

    images = {}
    dino_features = {}
    camera_parameters = {}
    images_old = {}
    dino_features_old = {}
    camera_parameters_old = {}

    for camera_name in data["target_cam"]:
        target_camera = data["target_cam"][camera_name]
        source_camera = data["source_cam"][camera_name]
        images[camera_name] = torch.stack([target_camera["image"]], dim=0)
        dino_features[camera_name] = torch.stack([target_camera["dino_features"]], dim=0)
        camera_parameters[camera_name] = target_camera["camera_parameters"]
        images_old[camera_name] = torch.stack([source_camera["image"]], dim=0)
        dino_features_old[camera_name] = torch.stack([source_camera["dino_features"]], dim=0)
        camera_parameters_old[camera_name] = source_camera["camera_parameters"]

    source_3dgs = data["source_3dgs"]
    return {
        "scene_id": data["scene_id"],
        "source_mode": data["source_mode"],
        "target_mode": data["target_mode"],
        "untransparent_features": source_3dgs["untransparent_features"],
        "untransparent_scalings": source_3dgs["untransparent_scalings"],
        "untransparent_opacity_logits": source_3dgs["untransparent_opacity_logits"],
        "untransparent_rotations": source_3dgs["untransparent_rotations"],
        "block_coords_all": source_3dgs["block_coords_all"].clone(),
        "linear_block_positions": torch.tensor(source_3dgs["block_linear_positions"], dtype=torch.long),
        "dense_coords": source_3dgs["dense_coords"],
        "images": images,
        "dino_features": dino_features,
        "camera_parameters": camera_parameters,
        "images_old": images_old,
        "dino_features_old": dino_features_old,
        "camera_parameters_old": camera_parameters_old,
        "indices": source_3dgs["indices"],
        "source_viewpoint_stack": data["source_viewpoint_stack"],
        "target_viewpoint_stack": data["target_viewpoint_stack"],
    }
