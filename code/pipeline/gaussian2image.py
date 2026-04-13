import torch
from deps.scene.gaussian_model import GaussianModel
import torch.nn as nn
from deps.gaussian_renderer import render
import os
import cv2
import numpy as np
from pipeline.colmap_cam import load_colmap_cameras
import math

def create_gaussians(xyz, features_gt, scalings_gt, opacity_logits_gt, rotations_gt, cam_num=100):
    gaussians = GaussianModel(1, "default")

    gaussians.cam_num = cam_num
    assert len(features_gt.shape) == 2
    features_gt = features_gt.view(features_gt.shape[0], -1, 3)
    gaussians._features_dc = features_gt[:, :1, :]
    gaussians._features_rest = features_gt[:, 1:, :]
    gaussians._scaling = scalings_gt
    gaussians._opacity = opacity_logits_gt
    gaussians._rotation = rotations_gt

    gaussians._xyz = xyz
    gaussians.max_radii2D = torch.zeros((gaussians._opacity.shape[0]), device="cuda")
    exposure = torch.eye(3, 4, device="cuda")[None].repeat(100, 1, 1)
    gaussians._exposure = nn.Parameter(exposure.requires_grad_(False))
    return gaussians

def pred2gaussians(linear_block_positions, color_features, scalings, opacity_logits, rotations, block_resolution, sub_grid_resolution, scale_ratio=4, matched_indices=None):
    grid_resolution = block_resolution * sub_grid_resolution
    theoretical_distance = 2 / grid_resolution
    theoretical_radius = theoretical_distance / 2
    log_theoretical_radius = math.log(theoretical_radius)
    scale_bound = math.log(scale_ratio)

    scalings = scalings.clamp(log_theoretical_radius-scale_bound, log_theoretical_radius+scale_bound)

    block_is = linear_block_positions // (block_resolution ** 2)
    block_js = (linear_block_positions // block_resolution) % block_resolution
    block_ks = linear_block_positions % block_resolution
    sub_grid_is = []
    sub_grid_js = []
    sub_grid_ks = []
    for di in range(sub_grid_resolution):
        for dj in range(sub_grid_resolution):
            for dk in range(sub_grid_resolution):
                sub_grid_is.append(di)
                sub_grid_js.append(dj)
                sub_grid_ks.append(dk)
    sub_grid_is = torch.tensor(sub_grid_is, device="cuda")
    sub_grid_js = torch.tensor(sub_grid_js, device="cuda")
    sub_grid_ks = torch.tensor(sub_grid_ks, device="cuda")

    i_positions = block_is.unsqueeze(1) * sub_grid_resolution + sub_grid_is.unsqueeze(0)
    j_positions = block_js.unsqueeze(1) * sub_grid_resolution + sub_grid_js.unsqueeze(0)
    k_positions = block_ks.unsqueeze(1) * sub_grid_resolution + sub_grid_ks.unsqueeze(0)

    i_positions = i_positions.view(-1)
    j_positions = j_positions.view(-1)
    k_positions = k_positions.view(-1)

    grid_resolution = block_resolution * sub_grid_resolution
    
    xyz = torch.stack((i_positions, j_positions, k_positions), dim=-1) / (grid_resolution - 1) * 2 - 1
    if matched_indices is not None:
        xyz = xyz[matched_indices]

    gaussians = create_gaussians(xyz, color_features, scalings, opacity_logits, rotations)
    return gaussians

def save_torch_image(image, path):
    image_np = image.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(path, image_np)
    if not success:
        raise IOError(
            f"Failed to write image to '{path}'. Check that the output directory is writable and has enough free disk space."
        )
    return image_np

def render_images(gaussians, image_save_path, viewpoint_stack=None, max_cam_num=150, black_background=False):
    if viewpoint_stack is None:
        viewpoint_stack = load_colmap_cameras('standard_colmap_cameras/')
    if black_background:
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    else:
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    os.makedirs(image_save_path, exist_ok=True)
    for camera_id, viewpoint_cam in enumerate(viewpoint_stack):
        if camera_id >= max_cam_num:
            break

        image = render(viewpoint_cam, gaussians, None, background, use_trained_exp=False, separate_sh=False)["render"]
        camera_id_str = f'{camera_id:06d}'
        save_torch_image(image, f'{image_save_path}/{camera_id_str}.png')

