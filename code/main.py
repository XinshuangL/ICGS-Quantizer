import argparse
import json
import os
import shutil

from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader
from tqdm import tqdm

from deps.lpipsPyTorch.modules.lpips import LPIPS
from model.icgs_model.tokenizer import GSTokenizer, out2gaussians
from pipeline.scene_dataset import DatasetAB, custom_collate_fn_test
from pipeline.loss_utils import _ssim, create_window, psnr
from pipeline.gaussian2image import render_images
from pipeline.general_utils import safe_state

cudnn.benchmark = False
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = False

GRID_RESOLUTION = 128
SSIM_WINDOW_SIZE = 11
DEFAULT_CAMERA_COMBOS = ["0", "01", "012", "0123", "01234", "012345"]
RENDER_SPECS = [
    ("scene", ("scene-geometry", "scene-color"), "cross"),
    ("scene_image", ("scene-image-geometry", "scene-image-color"), "cross"),
    ("scene_image_refined", ("scene-image-refined-geometry", "scene-image-refined-color"), "cross"),
    ("scene_real", ("scene-geometry", "scene-color"), "same"),
    ("scene_image_real", ("scene-image-geometry", "scene-image-color"), "same"),
    ("scene_image_refined_real", ("scene-image-refined-geometry", "scene-image-refined-color"), "same"),
]
EVAL_SPECS = [
    ("scene_real", "S2S"),
    ("scene_image_real", "S2S"),
    ("scene_image_refined_real", "S2S"),
    ("scene", "S2T"),
    ("scene_image", "S2T"),
    ("scene_image_refined", "S2T"),
]

lpips_criterion = None
ssim_window = None


def build_parser():
    parser = argparse.ArgumentParser(description="Render and evaluate the released ICGS test set.")
    parser.add_argument("--dataset-dir", default="../generated_data_test", help="Relative path to the test dataset root.")
    parser.add_argument("--checkpoint", default="model/ckpt_image_gs/stage3_fix_vq/9.pth", help="Relative path to the released test checkpoint.")
    parser.add_argument("--output-dir", default="results_ours_test", help="Relative path to the rendered test outputs.")
    parser.add_argument("--output-json", default="our_results.json", help="Relative path to the output metrics JSON.")
    parser.add_argument("--scene-num", type=int, default=100, help="Number of test scenes.")
    parser.add_argument("--view-num", type=int, default=50, help="Number of novel views per target mode.")
    parser.add_argument("--camera-combos", nargs="*", default=DEFAULT_CAMERA_COMBOS, help="Camera subsets to evaluate. Example: --camera-combos 0 01 012345")
    parser.add_argument("--ab-modes", nargs="*", default=["A", "B"], help="AB modes to evaluate. Default runs both A and B.")
    parser.add_argument("--min-free-gb", type=float, default=8.0, help="Minimum free disk space required before rendering starts.")
    return parser


def find_existing_parent(path):
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def ensure_free_space(output_dir, min_free_gb):
    probe_path = find_existing_parent(output_dir)
    _, _, free_bytes = shutil.disk_usage(probe_path)
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough free disk space for rendering. '{probe_path}' has {free_gb:.2f} GB free, "
            f"but at least {min_free_gb:.2f} GB is required."
        )


def parse_camera_combo(combo_str):
    if not combo_str:
        raise ValueError("Camera combo cannot be empty.")
    if any(ch not in "012345" for ch in combo_str):
        raise ValueError(f"Invalid camera combo '{combo_str}'. Expected digits from 0 to 5.")
    return list(combo_str)


def to_cuda(data):
    if isinstance(data, torch.Tensor):
        return data.cuda()
    if isinstance(data, dict):
        return {key: to_cuda(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_cuda(value) for value in data]
    return data


def render_output_group(outputs, specs, save_dir, scene_id, source_mode, target_mode, source_viewpoint_stack, target_viewpoint_stack, linear_block_positions, block_resolution, sub_grid_resolution):
    for folder_name, (geometry_key, color_key), mode in specs:
        gaussians = out2gaussians(
            outputs[geometry_key],
            outputs[color_key],
            linear_block_positions,
            block_resolution,
            sub_grid_resolution,
        )
        if mode == "cross":
            image_dir = f"{save_dir}/{folder_name}/scene_{scene_id}/{source_mode}2{target_mode}"
            viewpoint_stack = target_viewpoint_stack
        else:
            image_dir = f"{save_dir}/{folder_name}/scene_{scene_id}/{source_mode}2{source_mode}"
            viewpoint_stack = source_viewpoint_stack
        render_images(gaussians, image_dir, viewpoint_stack)


@torch.no_grad()
def render_predictions(tokenizer, dataloader, block_resolution, sub_grid_resolution, save_dir):
    tokenizer.eval()

    for batch in dataloader:
        batch = to_cuda(batch)
        source_mode = batch["source_mode"]
        target_mode = batch["target_mode"]
        linear_block_positions = batch["linear_block_positions"]

        cross_state_outputs = tokenizer.quantize_then_dequantize(
            batch["images"],
            batch["dino_features"],
            batch["camera_parameters"],
            batch["untransparent_features"],
            batch["untransparent_scalings"],
            batch["untransparent_rotations"],
            batch["untransparent_opacity_logits"],
            batch["block_coords_all"],
            1,
            linear_block_positions,
        )
        render_output_group(
            cross_state_outputs,
            RENDER_SPECS[:3],
            save_dir,
            batch["scene_id"],
            source_mode,
            target_mode,
            batch["source_viewpoint_stack"],
            batch["target_viewpoint_stack"],
            linear_block_positions,
            block_resolution,
            sub_grid_resolution,
        )

        same_state_outputs = tokenizer.quantize_then_dequantize(
            batch["images_old"],
            batch["dino_features_old"],
            batch["camera_parameters_old"],
            batch["untransparent_features"],
            batch["untransparent_scalings"],
            batch["untransparent_rotations"],
            batch["untransparent_opacity_logits"],
            batch["block_coords_all"],
            1,
            linear_block_positions,
        )
        render_output_group(
            same_state_outputs,
            RENDER_SPECS[3:],
            save_dir,
            batch["scene_id"],
            source_mode,
            target_mode,
            batch["source_viewpoint_stack"],
            batch["target_viewpoint_stack"],
            linear_block_positions,
            block_resolution,
            sub_grid_resolution,
        )


def load_image(path):
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3, :, :]


def init_metrics():
    global lpips_criterion
    global ssim_window

    if lpips_criterion is None:
        lpips_criterion = LPIPS("alex", "0.1").cuda()
    if ssim_window is None:
        ssim_window = create_window(SSIM_WINDOW_SIZE, 3).cuda().float()


def get_source_mode(target_mode, transfer_mode):
    if transfer_mode == "S2S":
        return target_mode
    if transfer_mode == "S2T":
        return "A" if target_mode == "B" else "B"
    raise ValueError(f"Invalid transfer mode: {transfer_mode}")


def evaluate_one_setting(pred_dir, gt_dir, scene_num, view_num, setting_name, transfer_mode, camera_combo):
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0

    for scene_idx in tqdm(range(scene_num)):
        pred_images = []
        gt_images = []
        for target_mode in ["A", "B"]:
            source_mode = get_source_mode(target_mode, transfer_mode)
            pred_image_dir = f"{pred_dir}/{camera_combo}/{setting_name}/scene_{scene_idx}/{source_mode}2{target_mode}"
            gt_image_dir = f"{gt_dir}/scene_{scene_idx}/{target_mode}/novel_view/images"
            for view_idx in range(view_num):
                pred_images.append(load_image(f"{pred_image_dir}/{view_idx:06d}.png"))
                gt_images.append(load_image(f"{gt_image_dir}/{view_idx:06d}.png"))

        pred = torch.cat(pred_images, dim=0).cuda()
        gt = torch.cat(gt_images, dim=0).cuda()
        psnr_sum += psnr(pred, gt).mean().item()
        ssim_sum += _ssim(pred, gt, ssim_window, SSIM_WINDOW_SIZE, 3, True).item()
        lpips_sum += lpips_criterion(pred, gt).mean().item() / pred.shape[0]

    return {
        "psnr": psnr_sum / scene_num,
        "ssim": ssim_sum / scene_num,
        "lpips": lpips_sum / scene_num,
    }


def evaluate_predictions(pred_dir, gt_dir, output_json, scene_num, view_num, camera_combos):
    init_metrics()
    results = {}

    for setting_name, transfer_mode in EVAL_SPECS:
        results[setting_name] = {}
        for camera_combo in camera_combos:
            metrics = evaluate_one_setting(pred_dir, gt_dir, scene_num, view_num, setting_name, transfer_mode, camera_combo)
            print(
                f"{setting_name} {camera_combo} psnr: {metrics['psnr']}, "
                f"ssim: {metrics['ssim']}, lpips: {metrics['lpips']}"
            )
            results[setting_name][camera_combo] = metrics

    with open(output_json, "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    args = build_parser().parse_args()
    safe_state(False)
    ensure_free_space(args.output_dir, args.min_free_gb)

    initial_channels = 32
    volume_channels = 128
    codebook_num = 4
    codebook_size = 1024
    sub_grid_resolution = 4
    downsample_order = 2
    block_resolution = 32
    dropout = 0.0
    image_resolution = 256
    use_dino = True

    os.makedirs(args.output_dir, exist_ok=True)

    for camera_combo in args.camera_combos:
        camera_names = parse_camera_combo(camera_combo)
        print(f"RUN camera_names={camera_combo}", flush=True)

        tokenizer = GSTokenizer(
            initial_channels=initial_channels,
            volume_channels=volume_channels,
            codebook_size=codebook_size,
            codebook_num=codebook_num,
            dropout=dropout,
            downsample_order=downsample_order,
            block_resolution=block_resolution,
            grid_resolution=GRID_RESOLUTION,
            image_resolution=image_resolution,
            use_dino=use_dino,
            camera_names=camera_names,
        ).cuda()
        tokenizer.load_state_dict(torch.load(args.checkpoint), strict=True)
        tokenizer.eval()

        for ab_mode in args.ab_modes:
            print(f"RUN fixed_AB_mode={ab_mode}", flush=True)
            dataset = DatasetAB(
                data_root=args.dataset_dir,
                scene_num=args.scene_num,
                downsample_order=downsample_order,
                block_resolution=block_resolution,
                camera_names=camera_names,
                fixed_AB_mode=ab_mode,
            )
            print("dataset length: ", len(dataset), flush=True)

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                collate_fn=custom_collate_fn_test,
                pin_memory=True,
            )
            render_predictions(tokenizer, dataloader, block_resolution, sub_grid_resolution, f"{args.output_dir}/{camera_combo}")

    evaluate_predictions(args.output_dir, args.dataset_dir, args.output_json, args.scene_num, args.view_num, args.camera_combos)
