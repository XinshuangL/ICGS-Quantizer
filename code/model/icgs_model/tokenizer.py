import torch
import torch.nn as nn
from model.icgs_model.vector_quantize_pytorch import VectorQuantize
import numpy as np
import torch.nn.functional as F
import math
from torch.cuda import amp
from model.icgs_model.resunet import ResUNet, Downsample, Upsample, ResnetBlock, Nonlinearity, Normalize, convert_1d_to_3d, convert_3d_to_1d
from model.icgs_model.resunet_2d import ResUNet2D
from itertools import combinations
import random
from deps.scene.gaussian_model import GaussianModel
from pipeline.colmap_cam import load_colmap_cameras
from deps.gaussian_renderer import render

def update_tensor(old_tensor, indices, new_tensor):    
    updated_tensor = old_tensor.clone()
    updated_tensor[indices] = new_tensor
    return updated_tensor

def load_gaussians(checkpoint_path):
    checkpoint_items = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

    features = checkpoint_items[2]
    scaling_logits = checkpoint_items[4]
    rotation_logits = checkpoint_items[5]
    opacity_logits = checkpoint_items[6]

    features = features.view(features.shape[0], -1)
    scalings = scaling_logits.view(scaling_logits.shape[0], -1)
    opacity_logits = opacity_logits.view(opacity_logits.shape[0], -1)
    rotations = rotation_logits.view(rotation_logits.shape[0], -1)

    return features.detach(), scalings.detach(), opacity_logits.detach(), rotations.detach()

class Encoder3D(nn.Module):
    def __init__(self, input_channels, latent_channels, downsample_order=2, dropout=0.1):
        super(Encoder3D, self).__init__()

        layers = []
        next_channel = input_channels
        for cur_downsample_order in range(downsample_order):
            pre_channel = next_channel
            if cur_downsample_order < downsample_order - 1:
                next_channel = min(pre_channel * 2, latent_channels)
            else:
                next_channel = latent_channels
            layers.append(ResnetBlock(in_channels=pre_channel, out_channels=next_channel, dropout=dropout))
            layers.append(Downsample(next_channel, False))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x

class UpSampler3D(nn.Module):
    def __init__(self, latent_channels, output_channels, upsample_order=2, dropout=0.1):
        super(UpSampler3D, self).__init__()

        layers = []
        next_channel = latent_channels
        for cur_upsample_order in range(upsample_order):
            pre_channel = next_channel
            next_channel = max(pre_channel // 2, output_channels)
            layers.append(ResnetBlock(in_channels=pre_channel, out_channels=next_channel, dropout=dropout))
            layers.append(nn.ConvTranspose3d(next_channel, next_channel, 2, stride=2))
            layers.append(Normalize(next_channel))
        assert next_channel == output_channels

        self.layers = nn.Sequential(*layers)
        self.output_channels = output_channels

    def forward(self, x):
        x = self.layers(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(-1, self.output_channels)
        return x

class LinearLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LinearLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            Normalize(output_channels),
            Nonlinearity()
        )

    def forward(self, x):
        return self.layers(x)

def create_dense_from_sparse(sparse_tensor, empty_vector, non_empty_indices, dense_len, channels):
    if empty_vector == None:
        empty_vector = torch.zeros((channels), device=sparse_tensor.device, dtype=sparse_tensor.dtype)
    dense_base = empty_vector.unsqueeze(0).repeat(dense_len, 1)
    sparse_delta = sparse_tensor - empty_vector.unsqueeze(0)
    dense_delta_full = torch.zeros_like(dense_base, device=dense_base.device)
    dense_delta_full.index_add_(0, non_empty_indices.long(), sparse_delta)
    return dense_base + dense_delta_full

def number_to_2_power(number):
    n = math.ceil(math.log2(number) - 1e-2)
    return 2 ** n

class Quantizer(nn.Module):
    def __init__(self, volume_channels, base_codebook_size, vq_num=8):
        super(Quantizer, self).__init__()

        self.float_rescale_ratio = 1.0
        self.vq_list = nn.ModuleList()
        self.activated_decay = 0.99995
        self._freeze = False
        for _ in range(vq_num):
            self.vq_list.append(VectorQuantize(
                decay = self.activated_decay,
                dim = volume_channels,
                codebook_size = base_codebook_size,
                rotation_trick = False,
                threshold_ema_dead_code=-1,
            ))
        self.vq_num = vq_num

    def forward(self, x, skip_vq):
        if skip_vq:
            return x, None, None, None
        
        residual = x
        indices_list = []
        vq_loss = 0
        quantized = 0
        kmeans_inputs = []

        for vq in self.vq_list:
            residual = residual * self.float_rescale_ratio
            kmeans_inputs.append(residual.detach().cpu())
            cur_quantized, cur_vq_indices, cur_vq_loss = vq(residual, freeze_codebook=self._freeze)

            residual = residual - cur_quantized.detach()
            indices_list.append(cur_vq_indices)
            vq_loss = vq_loss + cur_vq_loss
            quantized = quantized + cur_quantized / self.float_rescale_ratio

        return quantized, indices_list, vq_loss, kmeans_inputs
    
    @torch.no_grad()
    def get_embeddings(self):
        embeddings_list = []
        for vq in self.vq_list:
            embeddings_list.append(vq._codebook.embed[0].clone().detach())
        return embeddings_list

    def decode(self, indices_list):
        quantized_out = 0
        for indices, vq in zip(indices_list, self.vq_list):
            quantized = vq.get_output_from_indices(indices)
            quantized_out = quantized_out + quantized
        return quantized_out

    def fix_codebook(self):
        self._freeze = True
        for vq in self.vq_list:
            vq._codebook.ema_update = False
    
    def activate_codebook(self):
        self._freeze = False
        for vq in self.vq_list:
            vq._codebook.ema_update = True
            vq._codebook.decay = self.activated_decay

    @torch.no_grad()
    def load_kmeans(self, which, centroids, counts=None, total_seen=None, avg_batch=None, decay=None, laplace=1.0, obs_scale=1.0):
        """
        Overwrite one residual codebook with K-Means centroids and seed EMA
        buffers as if EMA had already observed `total_seen` vectors across batches of
        size `avg_batch`.

        """
        vq = self.vq_list[which]
        cb = vq._codebook
        device, dtype = cb.embed.device, cb.embed.dtype
        K = cb.codebook_size

        C = torch.as_tensor(centroids, device=device, dtype=dtype)
        if C.ndim == 2:
            C = C.unsqueeze(0)
        cb.embed.copy_(C)
        cb.initted.copy_(torch.tensor([True], device=cb.initted.device, dtype=cb.initted.dtype))

        if counts is not None:
            cnt = torch.as_tensor(counts, device=device, dtype=cb.cluster_size.dtype)
            if cnt.ndim == 1:
                cnt = cnt.unsqueeze(0)
            denom = cnt.sum(dim=-1, keepdim=True)
            p = (cnt + laplace) / (denom + laplace * K)
        else:
            p = torch.full((1, K), 1.0 / K, device=device, dtype=cb.cluster_size.dtype)

        if decay is None:
            decay = float(cb.decay)
        else:
            decay = float(decay)

        if (total_seen is not None) and (avg_batch is not None) and (avg_batch > 0):
            T_eff = obs_scale * (float(total_seen) / float(avg_batch))
            N_total_old = (1.0 - (decay ** T_eff)) * float(avg_batch)
            N_old = p * N_total_old
        else:
            N_old = p * max(float(avg_batch or 0.0), 1.0)

        N_old = torch.clamp(N_old, min=cb.init_cluster_size)

        cb.cluster_size.copy_(N_old)
        cb.embed_avg.copy_(C * N_old[..., None])

class Image2Feature2D(nn.Module):
    def __init__(self, grid_channels, ch_mult=[1, 1, 2, 4, 8, 16], num_res_blocks=1, dropout=0.0, downsample_order=0, use_dino=True, image_resolution=256):
        super(Image2Feature2D, self).__init__()

        output_channels_1x = grid_channels // ch_mult[downsample_order]

        print('Image2Feature2D initial_channels', output_channels_1x)
        self.input_layer = nn.Conv2d(3, output_channels_1x, kernel_size=3, stride=1, padding=1)
        self.use_dino = use_dino
        self.res_unet_2d = ResUNet2D(output_channels_1x, ch_mult=ch_mult, num_res_blocks=num_res_blocks, dropout=dropout, downsample_order=downsample_order, use_dino=use_dino, image_resolution=image_resolution, resamp_with_conv=False)

        self.grid_channels = grid_channels
        self.downsample_order = downsample_order

    def forward(self, x, dino_features):
        # x should be normalized using imagenet normalization
        batch_size, _, H, W = x.shape
        H_down = H // 2**self.downsample_order
        W_down = W // 2**self.downsample_order
        x = self.input_layer(x)
        image_features = self.res_unet_2d(x, dino_features)
        return image_features.view(batch_size, self.grid_channels, H_down, W_down)

@torch.no_grad()
def ijd_grids_to_depth_image(image_points, points_depth, H, W, r):
    device = image_points.device

    d = 2 * r + 1
    yy, xx = torch.meshgrid(torch.arange(d, device=device),
                            torch.arange(d, device=device),
                            indexing='ij')
    center = r
    dist2 = (xx - center)**2 + (yy - center)**2
    disk_mask = dist2 <= r**2
    offset_y, offset_x = torch.nonzero(disk_mask, as_tuple=True)
    offset_y = offset_y - r
    offset_x = offset_x - r
    K = offset_y.size(0)

    px = image_points[:, 0].round().long().unsqueeze(1)
    py = image_points[:, 1].round().long().unsqueeze(1)
    pd = points_depth.view(-1, 1)

    x_coords = px + offset_x
    y_coords = py + offset_y
    depths = pd.expand(-1, K)

    x_flat = x_coords.reshape(-1)
    y_flat = y_coords.reshape(-1)
    d_flat = depths.reshape(-1)

    valid = (x_flat >= 0) & (x_flat < W) & (y_flat >= 0) & (y_flat < H)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    d_flat = d_flat[valid]

    flat_idx = y_flat * W + x_flat
    depth_map_flat = torch.full((H * W,), float('inf'), device=device)

    if hasattr(torch, 'scatter_reduce'):
        depth_map_flat = torch.scatter_reduce(
            depth_map_flat,
            dim=0,
            index=flat_idx,
            src=d_flat,
            reduce='amin',
            include_self=True
        )
    else:
        for idx, val in zip(flat_idx, d_flat):
            depth_map_flat[idx] = min(depth_map_flat[idx], val)

    return depth_map_flat.view(H, W)


class Extractor(nn.Module):
    def __init__(self, camera_parameters, grid_resolution=64, block_resolution=16):
        super(Extractor, self).__init__()

        K = torch.tensor(camera_parameters['K']).float()
        P = torch.tensor(camera_parameters['P']).float()

        self.K = nn.Parameter(K, requires_grad=False).cuda()
        self.P = nn.Parameter(P, requires_grad=False).cuda()
        self.grid_resolution = grid_resolution
        self.block_resolution = block_resolution
        self.image_resolution = round(float(K[0, 0]))

        self.sub_grid_resolution = self.grid_resolution // self.block_resolution
        sub_grid_is = []
        sub_grid_js = []
        sub_grid_ks = []
        for di in range(self.sub_grid_resolution):
            for dj in range(self.sub_grid_resolution):
                for dk in range(self.sub_grid_resolution):
                    sub_grid_is.append(di)
                    sub_grid_js.append(dj)
                    sub_grid_ks.append(dk)
        self.sub_grid_is = nn.Parameter(torch.tensor(sub_grid_is, device="cuda"), requires_grad=False)
        self.sub_grid_js = nn.Parameter(torch.tensor(sub_grid_js, device="cuda"), requires_grad=False)
        self.sub_grid_ks = nn.Parameter(torch.tensor(sub_grid_ks, device="cuda"), requires_grad=False)

    def project_points(self, points):
        points_homogeneous = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)], dim=1)
        camera_points = self.P @ points_homogeneous.T
        camera_points = camera_points / camera_points[3:4, :]
        image_points_homogeneous = self.K @ camera_points[:3, :]
        image_points_homogeneous = image_points_homogeneous / image_points_homogeneous[2:3, :]
        image_points = image_points_homogeneous[:2, :].T
        points_depth = camera_points[2:3, :].T

        return image_points, points_depth

    def grid_coords_to_world_positions(self, grid_coords, min_bound=-1, max_bound=1):
        batch_ids = grid_coords[:, 0]
        grid_positions = grid_coords[:, 1:]
        grid_positions = grid_positions.float()
        world_positions = grid_positions * (max_bound - min_bound) / (self.grid_resolution - 1) + min_bound
        return batch_ids, world_positions

    def forward(self, dense_coords, pixel_features):
        point_count = dense_coords.shape[0]

        batch_ids, world_positions = self.grid_coords_to_world_positions(dense_coords)
        batch_ids = batch_ids.long()
        image_points, points_depth = self.project_points(world_positions)
        
        image_is_real = image_points[:, 1]
        image_js_real = image_points[:, 0]

        image_is = torch.floor(image_is_real)
        image_js = torch.floor(image_js_real)

        image_is_plus_1 = image_is + 1
        image_js_plus_1 = image_js + 1

        image_dis = image_is_real - image_is
        image_djs = image_js_real - image_js

        image_dis = image_dis.view(point_count, 1)
        image_djs = image_djs.view(point_count, 1)

        image_is = image_is.long()
        image_js = image_js.long()
        image_is_plus_1 = image_is_plus_1.long()
        image_js_plus_1 = image_js_plus_1.long()

        image_is = image_is.clamp(0, pixel_features.shape[2] - 1)
        image_js = image_js.clamp(0, pixel_features.shape[3] - 1)
        image_is_plus_1 = image_is_plus_1.clamp(0, pixel_features.shape[2] - 1)
        image_js_plus_1 = image_js_plus_1.clamp(0, pixel_features.shape[3] - 1)

        mean_j = (1 - image_dis) * pixel_features[batch_ids, :, image_is, image_js] + image_dis * pixel_features[batch_ids, :, image_is_plus_1, image_js]
        mean_jp = (1 - image_dis) * pixel_features[batch_ids, :, image_is, image_js_plus_1] + image_dis * pixel_features[batch_ids, :, image_is_plus_1, image_js_plus_1]
        feature_bilinear = (1 - image_djs) * mean_j + image_djs * mean_jp

        return feature_bilinear, points_depth

    @torch.no_grad()
    def estimate_visibility(self, linear_block_positions):
        block_is = linear_block_positions // (self.block_resolution ** 2)
        block_js = (linear_block_positions // self.block_resolution) % self.block_resolution
        block_ks = linear_block_positions % self.block_resolution        
        i_positions = block_is.unsqueeze(1) * self.sub_grid_resolution + self.sub_grid_is.unsqueeze(0)
        j_positions = block_js.unsqueeze(1) * self.sub_grid_resolution + self.sub_grid_js.unsqueeze(0)
        k_positions = block_ks.unsqueeze(1) * self.sub_grid_resolution + self.sub_grid_ks.unsqueeze(0)
        i_positions = i_positions.view(-1)
        j_positions = j_positions.view(-1)
        k_positions = k_positions.view(-1)
        dense_coords = torch.stack((i_positions, j_positions, k_positions), dim=-1)
        dense_coords = torch.cat((torch.zeros((dense_coords.shape[0], 1), device=dense_coords.device), dense_coords), dim=1)

        _, world_positions = self.grid_coords_to_world_positions(dense_coords)
        image_points, points_depth = self.project_points(world_positions)
        depth_image = ijd_grids_to_depth_image(image_points, points_depth, self.image_resolution, self.image_resolution, 4)

        depth_image = depth_image.view(1, 1, self.image_resolution, self.image_resolution)
        points_rendered_depth, points_projected_depth = self.forward(dense_coords, depth_image)

        grid_len = 2 / (self.grid_resolution - 1)
        grid_r = grid_len / 2
        grid_visibility = points_projected_depth < points_rendered_depth + grid_r

        visible_indices = dense_coords[grid_visibility.view(-1)].long()
        grids_for_counting = torch.zeros((self.grid_resolution, self.grid_resolution, self.grid_resolution), dtype=torch.float32, device=visible_indices.device)
        grids_for_counting[visible_indices[:, 1], visible_indices[:, 2], visible_indices[:, 3]] = 1.0
        grids_for_counting = grids_for_counting.view(1, 1, self.grid_resolution, self.grid_resolution, self.grid_resolution)
        grids_for_counting = F.max_pool3d(grids_for_counting, kernel_size=self.sub_grid_resolution, stride=self.sub_grid_resolution, padding=0)
        grids_for_counting = grids_for_counting.view(self.block_resolution, self.block_resolution, self.block_resolution)
        linear_block_visibility = grids_for_counting[block_is, block_js, block_ks]
        linear_block_visibility = (linear_block_visibility > 0.5).float()
        return linear_block_visibility

class ImageFeatureEmbedding(nn.Module):
    def __init__(self, grid_channels, downsample_order=2, grid_resolution=128, block_resolution=32, image_resolution=256, image_downsample_order=0, use_dino=False, camera_names=['0', '1', '2', '3', '4', '5'], dropout=0.0):
        super(ImageFeatureEmbedding, self).__init__()
        self.image2feature2d = Image2Feature2D(grid_channels=grid_channels, downsample_order=image_downsample_order, use_dino=use_dino, image_resolution=image_resolution, dropout=dropout)
        self.camera_names = camera_names
        self.camera_num = len(camera_names)
        self.downsample_order = downsample_order
        self.grid_resolution = grid_resolution
        self.block_resolution = block_resolution

        mix_options = []
        def generate_binary_lists(n, m):
            result = []
            for ones_pos in combinations(range(n), m):
                lst = [0] * n
                for pos in ones_pos:
                    lst[pos] = 1
                result.append(lst)
            return result
        for used_camera_num in range(1, self.camera_num + 1):
            options = generate_binary_lists(self.camera_num, used_camera_num)
            options = [option for option in options if option[0] > 0]
            options = [[1 / self.camera_num / len(options), option] for option in options]
            mix_options.extend(options)
        self.mix_options = mix_options
    
    def forward(self, images, dino_features, camera_parameters, dense_coords, linear_block_positions, mix_option_id=None, visibility_random_p=0.0):
        extractor_dict = {}
        for camera_name in self.camera_names:
            extractor_dict[camera_name] = Extractor(camera_parameters[camera_name], self.grid_resolution, self.block_resolution)    

        if mix_option_id is None:
            camera_num_this_time = random.randint(1, self.camera_num)
            camera_ids_this_time = random.sample(range(self.camera_num), camera_num_this_time)
            mix_option = [1 if cur_camera_id_this_time in camera_ids_this_time else 0 for cur_camera_id_this_time in range(self.camera_num)]
        elif isinstance(mix_option_id, str) and mix_option_id == 'all':
            mix_option = [1] * self.camera_num
        elif isinstance(mix_option_id, int) and 0 <= mix_option_id < len(self.mix_options):
            mix_option = self.mix_options[mix_option_id][1]
        else:
            raise ValueError(f'Invalid mix option id: {mix_option_id}')

        selected_camera_names = [camera_name for camera_name, selected in zip(self.camera_names, mix_option) if selected == 1]

        feature_sum = 0
        count_sum = 0

        cur_image_batch = torch.cat([images[camera_name] for camera_name in selected_camera_names], dim=0)
        cur_dino_feature_batch = torch.cat([dino_features[camera_name] for camera_name in selected_camera_names], dim=0)
        cur_feature_map_batch = self.image2feature2d(cur_image_batch, cur_dino_feature_batch)

        for camera_index, camera_name in enumerate(selected_camera_names):
            cur_feature_map = cur_feature_map_batch[camera_index:camera_index+1]
            cur_dense_features, _ = extractor_dict[camera_name](dense_coords, cur_feature_map)

            with torch.no_grad():
                cur_linear_block_visibility = extractor_dict[camera_name].estimate_visibility(linear_block_positions)

                if visibility_random_p > 0.0:
                    block_num = linear_block_positions.shape[0]
                    random_visibility = torch.randint(0, 2, (block_num,), dtype=cur_linear_block_visibility.dtype, device=cur_linear_block_visibility.device)
                    random_change_mask = torch.rand((block_num,), dtype=cur_linear_block_visibility.dtype, device=cur_linear_block_visibility.device) < (visibility_random_p * 2)
                    cur_linear_block_visibility[random_change_mask] = random_visibility[random_change_mask]

                cur_linear_block_visibility = torch.tensor(cur_linear_block_visibility, dtype=cur_dense_features.dtype, device=cur_dense_features.device)
                cur_dense_coords_visibility = cur_linear_block_visibility.view(-1, 1).repeat(1, (2**self.downsample_order)**3).view(-1)
            
            feature_sum = feature_sum + cur_dense_features * cur_dense_coords_visibility.unsqueeze(1)
            count_sum = count_sum + cur_dense_coords_visibility
        
        with torch.no_grad():
            count_sum[count_sum < 0.5] = 0.1
            multiview_visibility = torch.tensor(count_sum > 0.5, dtype=cur_dense_features.dtype, device=cur_dense_features.device)

        avg_features = feature_sum / count_sum.unsqueeze(1)
        return avg_features, multiview_visibility

def create_gaussians(xyz, features_gt, scalings_gt, opacity_logits_gt, rotations_gt, view_num=150):
    gaussians = GaussianModel(1, 'default')
    gaussians.cam_num = view_num
    gaussians.active_sh_degree = 0

    gaussians._xyz = xyz.cuda()
    if len(features_gt.shape) == 2:
        features_gt = features_gt.view(features_gt.shape[0], -1, 3)
    gaussians._features_dc = features_gt[:, :1, :].cuda()
    gaussians._features_rest = features_gt[:, 1:, :].cuda()

    gaussians._scaling = scalings_gt.cuda()
    gaussians._rotation = rotations_gt.cuda()
    gaussians._opacity = opacity_logits_gt.cuda()
    gaussians.max_radii2D = torch.zeros((gaussians._opacity.shape[0]), device="cuda")

    exposure = torch.eye(3, 4, device="cuda")[None].repeat(view_num, 1, 1)
    gaussians._exposure = nn.Parameter(exposure.requires_grad_(False))
    return gaussians

def out2gaussians(geometry_out, color_out, linear_block_positions, block_resolution, sub_grid_resolution, scale_ratio=10):
    scalings_decoded_values = geometry_out[0]
    opacity_logits_decoded_values = geometry_out[1]
    rotations_decoded_values = geometry_out[2]
    colors_decoded_values = color_out[0]

    grid_resolution = block_resolution * sub_grid_resolution
    theoretical_distance = 2 / grid_resolution
    theoretical_radius = theoretical_distance / 2
    log_theoretical_radius = math.log(theoretical_radius)
    scale_bound = math.log(scale_ratio)

    scalings_decoded_values = scalings_decoded_values.clamp(log_theoretical_radius-scale_bound, log_theoretical_radius+scale_bound)

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
    
    gaussians = create_gaussians(xyz, colors_decoded_values, scalings_decoded_values, opacity_logits_decoded_values, rotations_decoded_values)
    return gaussians


class GSTokenizer(nn.Module):
    def __init__(self, volume_channels=256, codebook_size=16384, codebook_num=8, downsample_order=2, initial_channels=64, block_resolution=32, grid_resolution=128, dropout=0.1, image_resolution=256, use_dino=True, camera_names=['0', '1', '2', '3', '4', '5'], skip_vq=False):
        super(GSTokenizer, self).__init__()

        self.skip_vq = skip_vq

        self.feature_original_channels = 3
        self.scaling_original_channels = 3
        self.opacity_original_channels = 1
        self.rotation_original_channels = 4
        self.volume_channels = volume_channels

        self.block_resolution = block_resolution
        self.grid_resolution = grid_resolution
        self.sub_grid_resolution = grid_resolution // block_resolution

        self.downsample_order = downsample_order
        self.resolution_reduction_factor_max = 2 ** downsample_order
        
        theoretical_distance = 2 / grid_resolution
        theoretical_radius = theoretical_distance / 2
        log_theoretical_radius = math.log(theoretical_radius)
        self.scaling_logits_additional_bias = log_theoretical_radius

        self.feature_input_layer = LinearLayer(self.feature_original_channels, initial_channels)
        self.scaling_input_layer = LinearLayer(self.scaling_original_channels, initial_channels)
        self.rotation_input_layer = LinearLayer(self.rotation_original_channels, initial_channels)
        self.opacity_logits_input_layer = LinearLayer(self.opacity_original_channels, initial_channels)
        self.merged_input_layer = LinearLayer(initial_channels * 4, initial_channels)

        self.scene_grid_empty_vector = nn.Parameter(torch.zeros(initial_channels), requires_grad=True)
        self.initial_channels = initial_channels
        
        self.shared_encoder = Encoder3D(
            initial_channels, 
            volume_channels, 
            downsample_order,
            0.0
        )

        self.scene_block_empty_vector_input = nn.Parameter(torch.zeros(volume_channels), requires_grad=True)

        self.resunet_encoder = ResUNet(input_channels=volume_channels, block_resolution=block_resolution, ch_mult=[1, 1, 2, 4], num_res_blocks=1, resamp_with_conv=False, dropout=0.0)

        self.geometry_projection_layer = nn.Conv3d(volume_channels, volume_channels, kernel_size=3, padding=1)
        self.color_projection_layer = nn.Conv3d(volume_channels, volume_channels, kernel_size=3, padding=1)

        self.image_feature_embedding = ImageFeatureEmbedding(grid_channels=initial_channels, downsample_order=downsample_order, grid_resolution=grid_resolution, block_resolution=block_resolution, image_resolution=image_resolution, image_downsample_order=0, use_dino=use_dino, camera_names=camera_names, dropout=dropout)
        self.image_grid_feature_norm = Normalize(initial_channels)
        self.image_feature_encoder = Encoder3D(
            initial_channels, 
            volume_channels, 
            downsample_order,
            0.0
        )
        self.image_grid_empty_vector = nn.Parameter(torch.zeros(initial_channels), requires_grad=True)

        self.scene_geometry_block_empty_vector_output = nn.Parameter(torch.zeros(volume_channels), requires_grad=True)
        self.scene_color_block_empty_vector_output = nn.Parameter(torch.zeros(volume_channels), requires_grad=True)
        self.geometry_resunet_decoder = ResUNet(input_channels=volume_channels, block_resolution=block_resolution, ch_mult=[1, 1, 2, 4], num_res_blocks=1, resamp_with_conv=False, dropout=0.0)
        self.color_resunet_decoder = ResUNet(input_channels=volume_channels, block_resolution=block_resolution, ch_mult=[1, 1, 2, 4], num_res_blocks=1, resamp_with_conv=False, dropout=0.0)
        self.scene_merge_geometry_color_layer = nn.Sequential(
            nn.Linear(volume_channels + volume_channels, volume_channels),
            Normalize(volume_channels)
        )

        self.scene_image_block_empty_vector_output = nn.Parameter(torch.zeros(volume_channels*2), requires_grad=True)

        self.scene_image_resunet_decoder = ResUNet(input_channels=volume_channels*2, block_resolution=block_resolution, ch_mult=[1, 1, 2, 4], num_res_blocks=1, resamp_with_conv=False, dropout=0.0)
        self.scene_geometry_upsampler = UpSampler3D(volume_channels, initial_channels, upsample_order=2, dropout=dropout)
        self.scene_color_upsampler = UpSampler3D(volume_channels, initial_channels, upsample_order=2, dropout=dropout)
        self.scene_image_upsampler = UpSampler3D(volume_channels*2, initial_channels*2, upsample_order=2, dropout=dropout)

        self.scene_geometry_out_layer = nn.Sequential(
            Nonlinearity(),
            nn.Linear(initial_channels, self.scaling_original_channels + self.opacity_original_channels + self.rotation_original_channels)
        )
        self.scene_color_out_layer = nn.Sequential(
            Nonlinearity(),
            nn.Linear(initial_channels, self.feature_original_channels)
        )
        self.scene_image_geometry_out_layer = nn.Sequential(
            nn.Linear(initial_channels*2, initial_channels*2),
            Normalize(initial_channels*2),
            Nonlinearity(),
            nn.Linear(initial_channels*2, self.scaling_original_channels + self.opacity_original_channels + self.rotation_original_channels)
        )
        self.scene_image_color_out_layer = nn.Sequential(
            nn.Linear(initial_channels*2, initial_channels*2),
            Normalize(initial_channels*2),
            Nonlinearity(),
            nn.Linear(initial_channels*2, self.feature_original_channels)
        )

        self.image_scene_volume_linear_refine_layer = nn.Sequential(
            nn.Linear(initial_channels*3, initial_channels*3),
            Normalize(initial_channels*3),
            Nonlinearity(),
            nn.Linear(initial_channels*3, initial_channels*2),
        )

        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=0.02)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.trunc_normal_(m.weight.data, std=0.02)
                if with_bias: m.bias.data.zero_()

        self.geometry_vq = Quantizer(volume_channels, codebook_size, codebook_num)
        self.color_vq = Quantizer(volume_channels, codebook_size, codebook_num)

    def forward_geometry(self, geometry_decoded_values):
        # split
        scalings_decoded_values = geometry_decoded_values[:, :self.scaling_original_channels]
        opacity_logits_decoded_values = geometry_decoded_values[:, self.scaling_original_channels:self.scaling_original_channels + self.opacity_original_channels]
        rotations_decoded_values = geometry_decoded_values[:, self.scaling_original_channels + self.opacity_original_channels: ]

        out = [scalings_decoded_values + self.scaling_logits_additional_bias, opacity_logits_decoded_values, rotations_decoded_values]
        return out

    def forward_color(self, color_decoded_values):
        # flatten
        color_decoded_values = color_decoded_values.view(-1, self.feature_original_channels)
        return [color_decoded_values]

    def dense2sparse_final(self, dense_values, scene_num, linear_block_positions):
        channels = dense_values.shape[1]
        dense_values = dense_values.permute(0, 2, 3, 4, 1).reshape(scene_num*self.block_resolution ** 3, channels)
        return dense_values[linear_block_positions].view(-1, channels, 1, 1, 1)

    def forward(self, images, dino_features, camera_parameters, features_values, scalings_values, rotations_values, opacity_logits_values, block_coords_all, scene_num, linear_block_positions, dense_coords, mix_option_id=None, visibility_random_p=0.0):

        ####################################################################
        # scene - encode & VQ
        ####################################################################
        opacity_logits_values = F.sigmoid(opacity_logits_values)

        # input
        features_values_high_dimensional = self.feature_input_layer(features_values)
        scalings_values_high_dimensional = self.scaling_input_layer(scalings_values)
        rotations_values_high_dimensional = self.rotation_input_layer(rotations_values)
        opacity_logits_values_high_dimensional = self.opacity_logits_input_layer(opacity_logits_values)

        merged_values_high_dimensional = torch.cat([features_values_high_dimensional, scalings_values_high_dimensional, rotations_values_high_dimensional, opacity_logits_values_high_dimensional], dim=1)
        merged_values_high_dimensional = self.merged_input_layer(merged_values_high_dimensional)

        # sparse to dense: grids
        block_num = block_coords_all[:, 0].max() + 1
        single_block_len = self.resolution_reduction_factor_max ** 3
        grid_non_empty_indices = block_coords_all[:, 0] * single_block_len + block_coords_all[:, 1] * (self.resolution_reduction_factor_max ** 2) + block_coords_all[:, 2] * self.resolution_reduction_factor_max + block_coords_all[:, 3]

        dense_merged_values_high_dimensional = create_dense_from_sparse(merged_values_high_dimensional, self.scene_grid_empty_vector, grid_non_empty_indices, block_num * single_block_len, self.initial_channels)
        dense_merged_values_high_dimensional = dense_merged_values_high_dimensional.view(block_num, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.initial_channels).permute(0, 4, 1, 2, 3)

        # encode
        merged_encoded_values = self.shared_encoder(dense_merged_values_high_dimensional)

        # sparse to dense: blocks        
        merged_encoded_values_dense = create_dense_from_sparse(
            merged_encoded_values.view(block_num, self.volume_channels), 
            self.scene_block_empty_vector_input, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels
        )

        # conv residual
        merged_encoded_values_dense = merged_encoded_values_dense.view(scene_num, self.block_resolution, self.block_resolution, self.block_resolution, self.volume_channels).permute(0, 4, 1, 2, 3)
        merged_encoded_values_dense = self.resunet_encoder(merged_encoded_values_dense, is_1d=False)

        # divide to geometry and color
        geometry_values_dense = self.geometry_projection_layer(merged_encoded_values_dense)
        color_values_dense = self.color_projection_layer(merged_encoded_values_dense)

        # dense to sparse: blocks
        geometry_values_dense_1d = convert_3d_to_1d(geometry_values_dense, self.block_resolution, self.volume_channels).view(-1, self.volume_channels)
        geometry_values_dense_1d_sparse = geometry_values_dense_1d[linear_block_positions]

        color_values_dense_1d = convert_3d_to_1d(color_values_dense, self.block_resolution, self.volume_channels).view(-1, self.volume_channels)
        color_values_dense_1d_sparse = color_values_dense_1d[linear_block_positions]

        # quantize
        geometry_volume_sparse_1d, geometry_indices_list, geometry_vq_loss, geometry_cur_f_dict = self.geometry_vq(geometry_values_dense_1d_sparse, skip_vq=self.skip_vq)
        color_volume_sparse_1d, color_indices_list, color_vq_loss, color_cur_f_dict = self.color_vq(color_values_dense_1d_sparse, skip_vq=self.skip_vq)

        ####################################################################
        # image
        ####################################################################
        image_grid_features, multiview_visibility = self.image_feature_embedding(images, dino_features, camera_parameters, dense_coords, linear_block_positions, mix_option_id=mix_option_id, visibility_random_p=visibility_random_p)
        image_grid_features = image_grid_features * multiview_visibility.view(-1, 1) + self.image_grid_empty_vector.view(1, self.initial_channels) * (1 - multiview_visibility.view(-1, 1))
        image_grid_features = self.image_grid_feature_norm(image_grid_features)

        image_grid_features = image_grid_features.view(block_num, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.initial_channels).permute(0, 4, 1, 2, 3)

        image_block_features = self.image_feature_encoder(image_grid_features)

        ####################################################################
        # Scene decode - w/o image
        ####################################################################
        # sparse to dense: blocks
        geometry_volume_dense_1d = create_dense_from_sparse(
            geometry_volume_sparse_1d.view(block_num, self.volume_channels), 
            self.scene_geometry_block_empty_vector_output, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels
        )
        geometry_volume_dense_1d = geometry_volume_dense_1d.view(scene_num, self.block_resolution ** 3, self.volume_channels)

        color_volume_dense_1d = create_dense_from_sparse(
            color_volume_sparse_1d.view(block_num, self.volume_channels), 
            self.scene_color_block_empty_vector_output, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels
        )
        color_volume_dense_1d = color_volume_dense_1d.view(scene_num, self.block_resolution ** 3, self.volume_channels)

        # decode
        geometry_volume_dense_1d = self.geometry_resunet_decoder(geometry_volume_dense_1d, is_1d=True)
        scene_geometry_volume = convert_1d_to_3d(geometry_volume_dense_1d, self.block_resolution, self.volume_channels)

        color_volume_dense_1d = self.color_resunet_decoder(color_volume_dense_1d, is_1d=True)
        scene_color_volume = convert_1d_to_3d(color_volume_dense_1d, self.block_resolution, self.volume_channels)

        ####################################################################
        # Scene decode - w image
        ####################################################################
        scene_geometry_color_volume_sparse_1d = torch.cat([
            geometry_volume_sparse_1d.view(block_num, self.volume_channels).detach(), 
            color_volume_sparse_1d.view(block_num, self.volume_channels)
        ], dim=1)
        scene_geometry_color_volume_sparse_1d = self.scene_merge_geometry_color_layer(scene_geometry_color_volume_sparse_1d)
        scene_image_block_features_sparse = torch.cat([
            scene_geometry_color_volume_sparse_1d.view(block_num, self.volume_channels),
            image_block_features.view(block_num, self.volume_channels)
        ], dim=1)

        # sparse to dense: blocks
        scene_image_block_features_dense_1d = create_dense_from_sparse(
            scene_image_block_features_sparse.view(block_num, self.volume_channels*2), 
            self.scene_image_block_empty_vector_output, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels*2
        )
        scene_image_block_features_dense_1d = scene_image_block_features_dense_1d.view(scene_num, self.block_resolution ** 3, self.volume_channels*2)

        # decode
        scene_image_block_features_dense_1d = self.scene_image_resunet_decoder(scene_image_block_features_dense_1d, is_1d=True)
        scene_image_volume = convert_1d_to_3d(scene_image_block_features_dense_1d, self.block_resolution, self.volume_channels*2)

        ####################################################################
        # decode
        ####################################################################
        # dense to sparse: blocks. N, C/2C, 1, 1, 1
        scene_geometry_volume_sparse = self.dense2sparse_final(scene_geometry_volume, scene_num, linear_block_positions)
        scene_color_volume_sparse = self.dense2sparse_final(scene_color_volume, scene_num, linear_block_positions)
        scene_image_volume_sparse = self.dense2sparse_final(scene_image_volume, scene_num, linear_block_positions)

        # upsample: -> N, C'/2C', 4, 4, 4 -> N*4*4*4, C'/2C'
        scene_geometry_volume_linear = self.scene_geometry_upsampler(scene_geometry_volume_sparse)
        scene_color_volume_linear = self.scene_color_upsampler(scene_color_volume_sparse)
        scene_image_volume_linear = self.scene_image_upsampler(scene_image_volume_sparse)

        # merge refined image
        image_grid_features = image_grid_features.permute(0, 2, 3, 4, 1).reshape(block_num * self.resolution_reduction_factor_max ** 3, self.initial_channels)        
        scene_image_volume_linear_refined = scene_image_volume_linear + self.image_scene_volume_linear_refine_layer(torch.cat([
            scene_image_volume_linear,
            image_grid_features
        ], dim=1))

        scene_geometry_out_raw = self.scene_geometry_out_layer(scene_geometry_volume_linear)
        scene_color_out_raw = self.scene_color_out_layer(scene_color_volume_linear)
        scene_image_geometry_out_raw = self.scene_image_geometry_out_layer(scene_image_volume_linear)
        scene_image_color_out_raw = self.scene_image_color_out_layer(scene_image_volume_linear)
        scene_image_geometry_out_refined_raw = self.scene_image_geometry_out_layer(scene_image_volume_linear_refined)
        scene_image_color_out_refined_raw = self.scene_image_color_out_layer(scene_image_volume_linear_refined)

        # scene
        scene_geometry_out = self.forward_geometry(
            scene_geometry_out_raw
        )
        scene_color_out = self.forward_color(
            scene_color_out_raw
        )

        # scene-image
        scene_image_geometry_out = self.forward_geometry(
            scene_image_geometry_out_raw
        )
        scene_image_color_out = self.forward_color(
            scene_image_color_out_raw
        )

        # refined scene-image
        scene_image_geometry_refined_out = self.forward_geometry(
            scene_image_geometry_out_refined_raw
        )
        scene_image_color_refined_out = self.forward_color(
            scene_image_color_out_refined_raw
        )

        out = {
            'scene-geometry': scene_geometry_out,
            'scene-color': scene_color_out,
            'scene-image-geometry': scene_image_geometry_out,
            'scene-image-color': scene_image_color_out,
            'scene-image-refined-geometry': scene_image_geometry_refined_out,
            'scene-image-refined-color': scene_image_color_refined_out
        }

        return out, geometry_vq_loss, color_vq_loss, geometry_cur_f_dict, color_cur_f_dict

    def fast_forward_for_vq(self, features_values, scalings_values, rotations_values, opacity_logits_values, block_coords_all, scene_num, linear_block_positions):

        ####################################################################
        # scene - encode & VQ
        ####################################################################
        opacity_logits_values = F.sigmoid(opacity_logits_values)

        # input
        features_values_high_dimensional = self.feature_input_layer(features_values)
        scalings_values_high_dimensional = self.scaling_input_layer(scalings_values)
        rotations_values_high_dimensional = self.rotation_input_layer(rotations_values)
        opacity_logits_values_high_dimensional = self.opacity_logits_input_layer(opacity_logits_values)

        merged_values_high_dimensional = torch.cat([features_values_high_dimensional, scalings_values_high_dimensional, rotations_values_high_dimensional, opacity_logits_values_high_dimensional], dim=1)
        merged_values_high_dimensional = self.merged_input_layer(merged_values_high_dimensional)

        # sparse to dense: grids
        block_num = block_coords_all[:, 0].max() + 1
        single_block_len = self.resolution_reduction_factor_max ** 3
        grid_non_empty_indices = block_coords_all[:, 0] * single_block_len + block_coords_all[:, 1] * (self.resolution_reduction_factor_max ** 2) + block_coords_all[:, 2] * self.resolution_reduction_factor_max + block_coords_all[:, 3]

        dense_merged_values_high_dimensional = create_dense_from_sparse(merged_values_high_dimensional, self.scene_grid_empty_vector, grid_non_empty_indices, block_num * single_block_len, self.initial_channels)
        dense_merged_values_high_dimensional = dense_merged_values_high_dimensional.view(block_num, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.initial_channels).permute(0, 4, 1, 2, 3)

        # encode
        merged_encoded_values = self.shared_encoder(dense_merged_values_high_dimensional)

        # sparse to dense: blocks        
        merged_encoded_values_dense = create_dense_from_sparse(
            merged_encoded_values.view(block_num, self.volume_channels), 
            self.scene_block_empty_vector_input, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels
        )

        # conv residual
        merged_encoded_values_dense = merged_encoded_values_dense.view(scene_num, self.block_resolution, self.block_resolution, self.block_resolution, self.volume_channels).permute(0, 4, 1, 2, 3)
        merged_encoded_values_dense = self.resunet_encoder(merged_encoded_values_dense, is_1d=False)

        # divide to geometry and color
        geometry_values_dense = self.geometry_projection_layer(merged_encoded_values_dense)
        color_values_dense = self.color_projection_layer(merged_encoded_values_dense)

        # dense to sparse: blocks
        geometry_values_dense_1d = convert_3d_to_1d(geometry_values_dense, self.block_resolution, self.volume_channels).view(-1, self.volume_channels)
        geometry_values_dense_1d_sparse = geometry_values_dense_1d[linear_block_positions]

        color_values_dense_1d = convert_3d_to_1d(color_values_dense, self.block_resolution, self.volume_channels).view(-1, self.volume_channels)
        color_values_dense_1d_sparse = color_values_dense_1d[linear_block_positions]

        # quantize
        geometry_volume_sparse_1d, geometry_indices_list, geometry_vq_loss, geometry_cur_f_dict = self.geometry_vq(geometry_values_dense_1d_sparse, skip_vq=self.skip_vq)
        color_volume_sparse_1d, color_indices_list, color_vq_loss, color_cur_f_dict = self.color_vq(color_values_dense_1d_sparse, skip_vq=self.skip_vq)

        return geometry_cur_f_dict, color_cur_f_dict

    def fix_codebook(self):
        self.geometry_vq.fix_codebook()
        self.color_vq.fix_codebook()
    
    def activate_codebook(self):
        self.geometry_vq.activate_codebook()
        self.color_vq.activate_codebook()
    
    def quantize(self, features_values, scalings_values, rotations_values, opacity_logits_values, block_coords_all, scene_num, linear_block_positions):

        ####################################################################
        # scene - encode & VQ
        ####################################################################
        opacity_logits_values = F.sigmoid(opacity_logits_values)

        # input
        features_values_high_dimensional = self.feature_input_layer(features_values)
        scalings_values_high_dimensional = self.scaling_input_layer(scalings_values)
        rotations_values_high_dimensional = self.rotation_input_layer(rotations_values)
        opacity_logits_values_high_dimensional = self.opacity_logits_input_layer(opacity_logits_values)

        merged_values_high_dimensional = torch.cat([features_values_high_dimensional, scalings_values_high_dimensional, rotations_values_high_dimensional, opacity_logits_values_high_dimensional], dim=1)
        merged_values_high_dimensional = self.merged_input_layer(merged_values_high_dimensional)

        # sparse to dense: grids
        block_num = block_coords_all[:, 0].max() + 1
        single_block_len = self.resolution_reduction_factor_max ** 3
        grid_non_empty_indices = block_coords_all[:, 0] * single_block_len + block_coords_all[:, 1] * (self.resolution_reduction_factor_max ** 2) + block_coords_all[:, 2] * self.resolution_reduction_factor_max + block_coords_all[:, 3]

        dense_merged_values_high_dimensional = create_dense_from_sparse(merged_values_high_dimensional, self.scene_grid_empty_vector, grid_non_empty_indices, block_num * single_block_len, self.initial_channels)
        dense_merged_values_high_dimensional = dense_merged_values_high_dimensional.view(block_num, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.initial_channels).permute(0, 4, 1, 2, 3)

        # encode
        merged_encoded_values = self.shared_encoder(dense_merged_values_high_dimensional)

        # sparse to dense: blocks        
        merged_encoded_values_dense = create_dense_from_sparse(
            merged_encoded_values.view(block_num, self.volume_channels), 
            self.scene_block_empty_vector_input, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels
        )

        # conv residual
        merged_encoded_values_dense = merged_encoded_values_dense.view(scene_num, self.block_resolution, self.block_resolution, self.block_resolution, self.volume_channels).permute(0, 4, 1, 2, 3)
        merged_encoded_values_dense = self.resunet_encoder(merged_encoded_values_dense, is_1d=False)

        # divide to geometry and color
        geometry_values_dense = self.geometry_projection_layer(merged_encoded_values_dense)
        color_values_dense = self.color_projection_layer(merged_encoded_values_dense)

        # dense to sparse: blocks
        geometry_values_dense_1d = convert_3d_to_1d(geometry_values_dense, self.block_resolution, self.volume_channels).view(-1, self.volume_channels)
        geometry_values_dense_1d_sparse = geometry_values_dense_1d[linear_block_positions]

        color_values_dense_1d = convert_3d_to_1d(color_values_dense, self.block_resolution, self.volume_channels).view(-1, self.volume_channels)
        color_values_dense_1d_sparse = color_values_dense_1d[linear_block_positions]

        # quantize
        geometry_volume_sparse_1d, geometry_indices_list, geometry_vq_loss, geometry_cur_f_dict = self.geometry_vq(geometry_values_dense_1d_sparse, skip_vq=self.skip_vq)
        color_volume_sparse_1d, color_indices_list, color_vq_loss, color_cur_f_dict = self.color_vq(color_values_dense_1d_sparse, skip_vq=self.skip_vq)

        return geometry_indices_list, color_indices_list
    
    def linear_block_positions_to_dense_coords(self, linear_block_positions):
        cur_device = 'cuda'

        # sub_grid_resolution
        self.sub_grid_resolution = self.grid_resolution // self.block_resolution
        sub_grid_is = []
        sub_grid_js = []
        sub_grid_ks = []
        for di in range(self.sub_grid_resolution):
            for dj in range(self.sub_grid_resolution):
                for dk in range(self.sub_grid_resolution):
                    sub_grid_is.append(di)
                    sub_grid_js.append(dj)
                    sub_grid_ks.append(dk)
        
        # to tensor
        sub_grid_is = torch.tensor(sub_grid_is, device=cur_device)
        sub_grid_js = torch.tensor(sub_grid_js, device=cur_device)
        sub_grid_ks = torch.tensor(sub_grid_ks, device=cur_device)
        linear_block_positions = torch.tensor(linear_block_positions, device=cur_device)

        # linear_block_positions
        block_is = linear_block_positions // (self.block_resolution ** 2)
        block_js = (linear_block_positions // self.block_resolution) % self.block_resolution
        block_ks = linear_block_positions % self.block_resolution        
        i_positions = block_is.unsqueeze(1) * self.sub_grid_resolution + sub_grid_is.unsqueeze(0)
        j_positions = block_js.unsqueeze(1) * self.sub_grid_resolution + sub_grid_js.unsqueeze(0)
        k_positions = block_ks.unsqueeze(1) * self.sub_grid_resolution + sub_grid_ks.unsqueeze(0)
        i_positions = i_positions.view(-1)
        j_positions = j_positions.view(-1)
        k_positions = k_positions.view(-1)
        dense_coords = torch.stack((i_positions, j_positions, k_positions), dim=-1)
        dense_coords = torch.cat((torch.zeros((dense_coords.shape[0], 1), device=dense_coords.device), dense_coords), dim=1)
        return dense_coords

    def dequantize(self, geometry_indices_list, color_indices_list, scene_num, linear_block_positions, images, dino_features, camera_parameters):

        ####################################################################
        # dequantization from the indices lists
        ####################################################################
        block_num = len(linear_block_positions)
        geometry_volume_sparse_1d = self.geometry_vq.decode(geometry_indices_list)
        color_volume_sparse_1d = self.color_vq.decode(color_indices_list)
        dense_coords = self.linear_block_positions_to_dense_coords(linear_block_positions)

        ####################################################################
        # image
        ####################################################################
        image_grid_features, multiview_visibility = self.image_feature_embedding(images, dino_features, camera_parameters, dense_coords, linear_block_positions, mix_option_id='all', visibility_random_p=0.0)
        image_grid_features = image_grid_features * multiview_visibility.view(-1, 1) + self.image_grid_empty_vector.view(1, self.initial_channels) * (1 - multiview_visibility.view(-1, 1))
        image_grid_features = self.image_grid_feature_norm(image_grid_features)

        image_grid_features = image_grid_features.view(block_num, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.resolution_reduction_factor_max, self.initial_channels).permute(0, 4, 1, 2, 3)

        image_block_features = self.image_feature_encoder(image_grid_features)

        ####################################################################
        # Scene decode - w/o image
        ####################################################################
        # sparse to dense: blocks
        geometry_volume_dense_1d = create_dense_from_sparse(
            geometry_volume_sparse_1d.view(block_num, self.volume_channels), 
            self.scene_geometry_block_empty_vector_output, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels
        )
        geometry_volume_dense_1d = geometry_volume_dense_1d.view(scene_num, self.block_resolution ** 3, self.volume_channels)

        color_volume_dense_1d = create_dense_from_sparse(
            color_volume_sparse_1d.view(block_num, self.volume_channels), 
            self.scene_color_block_empty_vector_output, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels
        )
        color_volume_dense_1d = color_volume_dense_1d.view(scene_num, self.block_resolution ** 3, self.volume_channels)

        # decode
        geometry_volume_dense_1d = self.geometry_resunet_decoder(geometry_volume_dense_1d, is_1d=True)
        scene_geometry_volume = convert_1d_to_3d(geometry_volume_dense_1d, self.block_resolution, self.volume_channels)

        color_volume_dense_1d = self.color_resunet_decoder(color_volume_dense_1d, is_1d=True)
        scene_color_volume = convert_1d_to_3d(color_volume_dense_1d, self.block_resolution, self.volume_channels)

        ####################################################################
        # Scene decode - w image
        ####################################################################
        scene_geometry_color_volume_sparse_1d = torch.cat([
            geometry_volume_sparse_1d.view(block_num, self.volume_channels).detach(), 
            color_volume_sparse_1d.view(block_num, self.volume_channels)
        ], dim=1)
        scene_geometry_color_volume_sparse_1d = self.scene_merge_geometry_color_layer(scene_geometry_color_volume_sparse_1d)
        scene_image_block_features_sparse = torch.cat([
            scene_geometry_color_volume_sparse_1d.view(block_num, self.volume_channels),
            image_block_features.view(block_num, self.volume_channels)
        ], dim=1)

        # sparse to dense: blocks
        scene_image_block_features_dense_1d = create_dense_from_sparse(
            scene_image_block_features_sparse.view(block_num, self.volume_channels*2), 
            self.scene_image_block_empty_vector_output, 
            linear_block_positions, 
            scene_num * (self.block_resolution ** 3), 
            self.volume_channels*2
        )
        scene_image_block_features_dense_1d = scene_image_block_features_dense_1d.view(scene_num, self.block_resolution ** 3, self.volume_channels*2)

        # decode
        scene_image_block_features_dense_1d = self.scene_image_resunet_decoder(scene_image_block_features_dense_1d, is_1d=True)
        scene_image_volume = convert_1d_to_3d(scene_image_block_features_dense_1d, self.block_resolution, self.volume_channels*2)

        ####################################################################
        # decode
        ####################################################################
        # dense to sparse: blocks. N, C/2C, 1, 1, 1
        scene_geometry_volume_sparse = self.dense2sparse_final(scene_geometry_volume, scene_num, linear_block_positions)
        scene_color_volume_sparse = self.dense2sparse_final(scene_color_volume, scene_num, linear_block_positions)
        scene_image_volume_sparse = self.dense2sparse_final(scene_image_volume, scene_num, linear_block_positions)

        # upsample: -> N, C'/2C', 4, 4, 4 -> N*4*4*4, C'/2C'
        scene_geometry_volume_linear = self.scene_geometry_upsampler(scene_geometry_volume_sparse)
        scene_color_volume_linear = self.scene_color_upsampler(scene_color_volume_sparse)
        scene_image_volume_linear = self.scene_image_upsampler(scene_image_volume_sparse)

        # merge refined image
        image_grid_features = image_grid_features.permute(0, 2, 3, 4, 1).reshape(block_num * self.resolution_reduction_factor_max ** 3, self.initial_channels)        
        scene_image_volume_linear_refined = scene_image_volume_linear + self.image_scene_volume_linear_refine_layer(torch.cat([
            scene_image_volume_linear,
            image_grid_features
        ], dim=1))

        scene_geometry_out_raw = self.scene_geometry_out_layer(scene_geometry_volume_linear)
        scene_color_out_raw = self.scene_color_out_layer(scene_color_volume_linear)
        scene_image_geometry_out_raw = self.scene_image_geometry_out_layer(scene_image_volume_linear)
        scene_image_color_out_raw = self.scene_image_color_out_layer(scene_image_volume_linear)
        scene_image_geometry_out_refined_raw = self.scene_image_geometry_out_layer(scene_image_volume_linear_refined)
        scene_image_color_out_refined_raw = self.scene_image_color_out_layer(scene_image_volume_linear_refined)

        # scene
        scene_geometry_out = self.forward_geometry(
            scene_geometry_out_raw
        )
        scene_color_out = self.forward_color(
            scene_color_out_raw
        )

        # scene-image
        scene_image_geometry_out = self.forward_geometry(
            scene_image_geometry_out_raw
        )
        scene_image_color_out = self.forward_color(
            scene_image_color_out_raw
        )

        # refined scene-image
        scene_image_geometry_refined_out = self.forward_geometry(
            scene_image_geometry_out_refined_raw
        )
        scene_image_color_refined_out = self.forward_color(
            scene_image_color_out_refined_raw
        )

        out = {
            'scene-geometry': scene_geometry_out,
            'scene-color': scene_color_out,
            'scene-image-geometry': scene_image_geometry_out,
            'scene-image-color': scene_image_color_out,
            'scene-image-refined-geometry': scene_image_geometry_refined_out,
            'scene-image-refined-color': scene_image_color_refined_out
        }

        return out
    
    def quantize_then_dequantize(self, images, dino_features, camera_parameters, features_values, scalings_values, rotations_values, opacity_logits_values, block_coords_all, scene_num, linear_block_positions):
        block_num_1 = block_coords_all[:, 0].max() + 1
        block_num_2 = len(linear_block_positions)
        assert block_num_1 == block_num_2

        geometry_indices_list, color_indices_list = self.quantize(features_values, scalings_values, rotations_values, opacity_logits_values, block_coords_all, scene_num, linear_block_positions)
        out = self.dequantize(geometry_indices_list, color_indices_list, scene_num, linear_block_positions, images, dino_features, camera_parameters)
        return out

