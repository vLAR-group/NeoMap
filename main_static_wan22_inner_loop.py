import argparse
import numpy as np
import os
import random
import sys
from PIL import Image
import torch
import einops
import logging
import copy

from ops.PcdMgr import PcdMgr
from ops.cam_utils import Mcam, CamPlanner
from ops.utils.general import easy_save_video
import cv2

from tools.Wan.wan.neomap_i2v import NeoMap
from tools.Wan.wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from tools.Wan.wan.utils.prompt_extend import QwenPromptExpander
from tools.Wan.wan.utils.utils import str2bool

### vggt
from ops.vggt import ConfiguredVGGTWrapper
from neomap_release.runner import DEFAULT_CONFIG, config as load_model_config


def compute_depth_scale(our_depth_tensor, colmap_depth_array):

    # Use the first depth map from each source
    our_depth = our_depth_tensor[0, :, :, 0]

    H_target = our_depth.shape[0]
    W_target = our_depth.shape[1]

    if colmap_depth_array.ndim == 4:
        colmap_depth = colmap_depth_array[0, :, :, 0]
    else:
        colmap_depth = colmap_depth_array[0, :, :]

    if H_target != colmap_depth.shape[0] or W_target != colmap_depth.shape[1]: 
        colmap_resized = cv2.resize(colmap_depth, (W_target, H_target), interpolation=cv2.INTER_LINEAR)

    valid_mask = (
        np.isfinite(our_depth)
        & np.isfinite(colmap_resized)
        & (our_depth > 0)
        & (colmap_resized > 0)
    )
    if not np.any(valid_mask):
        raise ValueError("No overlapping valid depths found to estimate scale.")

    # Least squares estimation: our_depth ≈ scale * colmap_resized
    our_valid = our_depth[valid_mask]
    colmap_valid = colmap_resized[valid_mask]
    
    # Compute least squares scale
    numerator = np.sum(our_valid * colmap_valid)
    denominator = np.sum(colmap_valid * colmap_valid)
    
    if denominator == 0:
        raise ValueError("Cannot compute scale: denominator is zero (all colmap depths are zero)")
    
    scale = numerator / denominator
    
    return float(scale)


def get_traj_colmap(sparse_dir, num_frames=49, normalize_to_first=True):
    # This dataset-specific helper is only needed for COLMAP input, not --help
    # or the new image/depth/camera entry points.
    from colmap_camera_utils import COLMAPCameraTrajectory
    colmap_traj = COLMAPCameraTrajectory(sparse_dir, normalize_to_first=normalize_to_first)

    mcam_traj = []
    for colmap_cam in colmap_traj.cameras:
        mcam = colmap_to_mcam(colmap_cam)
        mcam_traj.append(mcam)

    if len(mcam_traj) < num_frames:
        last_mcam = mcam_traj[-1]
        pad_count = num_frames - len(mcam_traj)
        mcam_traj.extend([last_mcam.copy() for _ in range(pad_count)])

    return mcam_traj


def load_result_npz(npz_path):

    data = np.load(npz_path)
    if 'camtoworlds' in data:
        c2w_matrices = data['camtoworlds']  # shape: (N, 4, 4)
        depth_maps = data['depth_maps'] # shape: (N, H, W)
        
    elif 'extrinsics' in data:
        depth_maps = data['depth'] # shape: (N, H, W)
        camera_poses_3x4 = data['extrinsics'] # shape: (N, 3, 4) opencv world-to-camera transformation
        camera_intrinsics = data['intrinsics'] # shape: (N, 3, 3) opencv camera intrinsic matrix

        N = camera_poses_3x4.shape[0]
        camera_poses_4x4 = np.zeros((N, 4, 4), dtype=camera_poses_3x4.dtype)
        camera_poses_4x4[:, :3, :] = camera_poses_3x4
        camera_poses_4x4[:, 3, 3] = 1.0

        focal_lengths = (camera_intrinsics[:, 0, 0] + camera_intrinsics[:, 1, 1]) / 2
        depth_maps = depth_maps * (focal_lengths[:, None, None] / 300.0)
        
        # Invert to get c2w
        c2w_matrices = np.linalg.inv(camera_poses_4x4)
        
    else:
        raise ValueError(f"NPZ file must contain either 'camtoworlds' or 'extrinsics' key. "
                        f"Found keys: {list(data.keys())}")

    return depth_maps, c2w_matrices


def get_traj_npz(c2w_opencv_matrices, num_frames):

    # Get first camera pose for normalization
    first_w2c = np.linalg.inv(c2w_opencv_matrices[0])

    mcam_traj = []
    for i in range(c2w_opencv_matrices.shape[0]):
        c2w = c2w_opencv_matrices[i]
        
        # Normalize: multiply by inverse of first camera's w2c to set first camera as origin
        c2w_opencv_normalized = first_w2c @ c2w

        mcam = opencv_to_mcam(c2w_opencv_normalized)
        mcam_traj.append(mcam)

    if len(mcam_traj) < num_frames:
        last_mcam = mcam_traj[-1]
        pad_count = num_frames - len(mcam_traj)
        mcam_traj.extend([last_mcam.copy() for _ in range(pad_count)])

    return mcam_traj


def opencv_to_mcam(c2w_opencv):
    mcam = Mcam()
    
    mcam.R = c2w_opencv[:3, :3].copy()
    mcam.T = c2w_opencv[:3, 3].copy()

    # sTs
    mcam.R[[1,2],0] = -mcam.R[[1,2],0]
    mcam.R[0,1] = -mcam.R[0,1]
    mcam.R[0,2] = -mcam.R[0,2]
    mcam.T[[1,2]] = -mcam.T[[1,2]]

    return mcam


def colmap_to_mcam(colmap_cam):
    mcam = Mcam()
    
    mcam.R = colmap_cam.R.copy()
    mcam.T = colmap_cam.T.copy()

    # sTs
    mcam.R[[1,2],0] = -mcam.R[[1,2],0]
    mcam.R[0,1] = -mcam.R[0,1]
    mcam.R[0,2] = -mcam.R[0,2]
    mcam.T[[1,2]] = -mcam.T[[1,2]]

    return mcam


def load_depth_map_npz(npz_path: str) -> np.ndarray:
    """Load a depth map saved with :func:`save_depth_map_npz`."""
    with np.load(npz_path) as data:
        depth_map = data["depth_map"]
    return depth_map


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.
    """
    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    import requests
    try:
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()

        if response.status_code == 302:
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def run_vggt(input_image_path, repository_path, checkpoint_path, mask_sky,
             conf_thres=0.0, mask_black_bg=False, mask_white_bg=False):

    vggt = ConfiguredVGGTWrapper(
        repository_path=repository_path,
        checkpoint_path=checkpoint_path)
    # preprocess input
    predictions = vggt.run_vggt_init([input_image_path])

    images = predictions["images"] # (1, 3, 280, 518)

    pred_world_points = predictions["world_points_from_depth"] # (1, 280, 518, 3)
    pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0])) # (1, 280, 518)

    print(f"images.shape: {images.shape}")
    print(f"pred_world_points.shape: {pred_world_points.shape}")
    print(f"pred_world_points_conf.shape: {pred_world_points_conf.shape}")

    conf_mask = None

    if mask_sky:
        print("Masking sky from vggt predicted images...")
        import onnxruntime

        # Download skyseg.onnx if it doesn't exist
        if not os.path.exists("skyseg.onnx"):
            print("Downloading skyseg.onnx...")
            download_file_from_url(
                "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
            )

        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        
        # Get the shape of pred_world_points_conf to match
        S, H, W = pred_world_points_conf.shape
        print(f"S, H, W: {S}, {H}, {W}")
        
        sky_mask_list = []
        
        # Process each image in the predictions
        for i in range(S):
            # Extract image from predictions: images shape is [S, C, H, W]
            img_array = images[i].transpose(1, 2, 0)  # [H, W, C]
            
            # Convert to uint8 BGR format for sky segmentation
            img_bgr = (img_array * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            # Run sky segmentation directly on the image
            result_map = run_skyseg(skyseg_session, [320, 320], img_bgr)
            
            # Resize the result_map to match the depth map size
            result_map_resized = cv2.resize(result_map, (W, H))
            
            # Create binary mask: 255 = non-sky, 0 = sky
            # The model outputs low values for sky, high values for non-sky
            sky_mask = np.zeros_like(result_map_resized)
            sky_mask[result_map_resized < 32] = 255  # Non-sky regions (high values)
            
            sky_mask_list.append(sky_mask)

        # Convert list to numpy array with shape S×H×W
        sky_mask_array = np.array(sky_mask_list)

        # Apply sky mask to confidence scores
        sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
        pred_world_points_conf = pred_world_points_conf * sky_mask_binary

        conf = pred_world_points_conf.reshape(-1)
        # Convert percentage threshold to actual confidence value
        if conf_thres == 0.0:
            conf_threshold = 0.0
        else:
            conf_threshold = np.percentile(conf, conf_thres)

        conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    # Apply transformation matrix T to flip Y and Z axes
    T = np.diag([1, -1, -1])
    vertices_3d = (pred_world_points @ T).reshape(-1, 3)
    colors_rgb = einops.rearrange(predictions['images'], '1 c h w -> 1 h w c').reshape(-1, 3)
        
    # Combine into pts3d format [N, 6] with xyz and rgb
    pts3d = np.concatenate([vertices_3d, colors_rgb], axis=1)

    return predictions, vggt.avg_focal, pts3d, conf_mask


def _validate_args(args):
    # Basic check
    assert args.wan_ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    cfg = WAN_CONFIGS[args.task]

    args.num_denoising_steps = (
        args.num_denoising_steps
        if args.num_denoising_steps is not None else cfg.sample_steps)
    args.scheduler_time_shift = (
        args.scheduler_time_shift
        if args.scheduler_time_shift is not None else cfg.sample_shift)
    args.guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None else cfg.sample_guide_scale)

    args.seed = args.seed if args.seed >= 0 else random.randint(
        0, sys.maxsize)
        
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--sparse_dir', type=str, default=None)
    parser.add_argument('--npz_path', type=str, default=None)
    parser.add_argument('--num_frames', type=int, default=49)

    parser.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_CONFIG),
        help='YAML file containing VGGT and Wan checkpoint paths.')
    parser.add_argument('--output_dir', type=str, default="./results-360", help='Name of the configuration to load')

    parser.add_argument('--prompt', type=str, default="")

    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--num_denoising_steps",
        type=int, 
        default=15,
        help="Total reverse-flow steps, including the bootstrap step from 1 to tau.")
    parser.add_argument(
        "--scheduler_time_shift",
        type=float,
        default=3.5,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.98,
        help="Shifted near-noise timestamp reached by one step from pure noise.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--num_projection_iterations",
        type=int,
        default=15,
        help="Number K of AMP+PCP iterations at tau.")
    parser.add_argument(
        "--num_reanchor_steps",
        type=int,
        default=3,
        help="Number of early denoising timestamps receiving AMP-only re-anchoring.")
    parser.add_argument(
        "--amp_similarity_threshold",
        type=float,
        default=0.8,
        help="Lambda used by the AMP feature-similarity mask.")
    parser.add_argument(
        "--strict_vae_block_visibility",
        type=str2bool,
        default=True,
        help=("Mark an entire VAE compression block invisible when any "
              "pixel in that block is invisible."))
    parser.add_argument(
        "--visibility_mask_kernel",
        type=int,
        default=15,
        help="Kernel size to erode the mask.")
    parser.add_argument(
        "--use_reference_image_condition",
        type=str2bool,
        default=True,
        help="Whether to use the input condition image.")
    parser.add_argument(
        "--exclude_sky_from_geometry",
        type=str2bool,
        default=False,
        help="Whether to mask the sky.")
    parser.add_argument(
        "--visibility_mask_erode_only",
        type=str2bool,
        default=False,
        help="Whether to expand the visibility mask back.")
    
    args = parser.parse_args()

    user_config = load_model_config(args.config)
    args.vggt_checkpoint = user_config['checkpoints']['vggt']
    args.wan_ckpt_dir = user_config['checkpoints']['wan']
    args.vggt_repository = user_config['repositories']['vggt']

    if not os.path.isdir(args.vggt_repository):
        raise FileNotFoundError(
            f"VGGT repository from {args.config} does not exist: "
            f"{args.vggt_repository}")
    if not os.path.isfile(args.vggt_checkpoint):
        raise FileNotFoundError(
            f"VGGT checkpoint from {args.config} does not exist: "
            f"{args.vggt_checkpoint}")
    if not os.path.isdir(args.wan_ckpt_dir):
        raise FileNotFoundError(
            f"Wan checkpoint directory from {args.config} does not exist: "
            f"{args.wan_ckpt_dir}")

    _validate_args(args)

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    H, W = 720, 1280

    # load image
    image_names = sorted(os.listdir(args.image_dir))
    valid_num_frames = len(image_names)
    first_image_file = image_names[0]
    input_image_path = os.path.join(args.image_dir, first_image_file)
    print(f"Using input image from {input_image_path}")

    image = Image.open(input_image_path).resize((W, H))
    image = np.array(image.convert("RGB")) / 255.0  # (H, W, 3) normalize to [0, 1]
    image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to (3, H, W) tensor

    # get the camera trajectory from the colmap trajectory directory
    if args.sparse_dir:
        colmap_depth_map_path = os.path.join(args.sparse_dir, "vggt_depth_map.npz")
        gt_depth_maps = load_depth_map_npz(colmap_depth_map_path)
        traj = get_traj_colmap(args.sparse_dir, num_frames=args.num_frames, normalize_to_first=True)   
    else:
        gt_depth_maps, gt_c2w_matrices = load_result_npz(args.npz_path)
        traj = get_traj_npz(gt_c2w_matrices, args.num_frames)


    predictions, vggt_avg_focal, pm, conf_mask = run_vggt(
        input_image_path,
        repository_path=args.vggt_repository,
        checkpoint_path=args.vggt_checkpoint,
        mask_sky=args.exclude_sky_from_geometry,
        conf_thres=0.0,
        mask_black_bg=False,
        mask_white_bg=False
    )
    print(predictions['images'].shape, predictions['depth'].shape)
    vggt_H, vggt_W = predictions['images'].shape[2], predictions['images'].shape[3]
    focal_x = vggt_avg_focal * W / vggt_W
    focal_y = vggt_avg_focal * H / vggt_H

    scale = compute_depth_scale(predictions['depth'], gt_depth_maps)
    print(f"Estimated depth scale (Ours / COLMAP): {scale:.6f}")

    # align the camera trajectory scale
    for cam in traj:
        cam.set_cam(
            f = [focal_x, focal_y],
        )
        cam.H = H
        cam.W = W
        cam.c = ( cam.W//2, cam.H//2 )
        cam.T = cam.T * scale

    if conf_mask is not None:
        pcd = PcdMgr(pts3d=pm[conf_mask])
    else:
        pcd = PcdMgr(pts3d=pm)
        
    render_results = CamPlanner._render_video(pcd, traj = traj)
    render_results = torch.stack(render_results, dim=0) # [49, 720, 1280, 3]
    easy_save_video(render_results, f"./{output_dir}/render_results.mp4")

    masks = CamPlanner._render_video(pcd, traj = traj, mask = True)
    masks = torch.stack(masks, dim=0).unsqueeze(-1) # [49, 720, 1280, 1]
    easy_save_video(masks, f"./{output_dir}/masks.mp4")

    render_results = render_results.permute(0, 3, 1, 2)
    print("render_results.shape", render_results.shape)
    masks = masks.permute(0, 3, 1, 2).bool()
    print("masks.shape", masks.shape)

    render_H, render_W = render_results.shape[2], render_results.shape[3]
    print('render_H, render_W: ', render_H, render_W, render_results.shape)
    render_results = render_results.to('cuda')

    # Apply morphological operations to clean up the mask
    mask_float = masks.float()
    
    kernel_size = args.visibility_mask_kernel
    padding = (kernel_size - 1) // 2

    # Erosion: removes small isolated points
    mask_eroded = -torch.nn.functional.max_pool2d(-mask_float, kernel_size=kernel_size, stride=1, padding=padding)

    # Dilation: restores the size of remaining structures
    if not args.visibility_mask_erode_only:
        mask_opened = torch.nn.functional.max_pool2d(mask_eroded, kernel_size=kernel_size, stride=1, padding=padding)
    else:
        mask_opened = mask_eroded

    if mask_opened.shape[2:] != render_results.shape[2:]:
        print(f"Resizing mask spatial dims from {mask_opened.shape[2:]} to {render_results.shape[2:]}")
        mask_opened = torch.nn.functional.interpolate(
            mask_opened, 
            size=render_results.shape[2:],
            mode='nearest'
        )
        
    masks = mask_opened > 0.5

    white_frame = torch.ones_like(render_results)
    video_with_mask = torch.where(masks, render_results, white_frame)
    black_frame = torch.zeros_like(render_results)
    render_results = torch.where(masks, render_results, black_frame)

    easy_save_video(masks, f"./{output_dir}/masks_after_open.mp4")
    easy_save_video(video_with_mask, f"./{output_dir}/video_with_mask.mp4", value_range="0,1")


    cfg = WAN_CONFIGS[args.task]
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        print(f"offload_model is not specified, set to {args.offload_model}.")

    logging.info(f"Input prompt: {args.prompt}")
    # prompt extend
    if args.use_prompt_extend:
        img = Image.open(input_image_path).convert("RGB")
        prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=True,
                device=rank)
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang="en",
                seed=args.seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")
    
    args.prompt = args.prompt + cfg.static_scene_positive_prompt
    print(f"Final positive prompt:{args.prompt}")
    
    # Save prompt to local txt file
    prompt_save_path = os.path.join(output_dir, "final_prompt.txt")
    with open(prompt_save_path, 'w', encoding='utf-8') as f:
        f.write(args.prompt)
    print(f"Saved final prompt to {prompt_save_path}")

    print("Creating the refactored NeoMap pipeline.")
    neomap_pipeline = NeoMap(
        config=cfg,
        checkpoint_dir=args.wan_ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        seed=args.seed
    )

    condition_image = image if args.use_reference_image_condition else None
    print("Generating video with NeoMap...")
    video = neomap_pipeline(
        render_results,
        condition_image,
        masks,
        input_prompt=args.prompt,
        max_area=MAX_AREA_CONFIGS[args.size],
        tau=args.tau,
        shift=args.scheduler_time_shift,
        sampling_steps=args.num_denoising_steps,
        projection_iterations=args.num_projection_iterations,
        reanchor_steps=args.num_reanchor_steps,
        guide_scale=args.guidance_scale,
        offload_model=args.offload_model,
        similarity_threshold=args.amp_similarity_threshold,
        strict_vae_block_visibility=args.strict_vae_block_visibility)
    
    video = video[:valid_num_frames]

    # Save the generated video
    video = video.transpose(0, 1)
    easy_save_video(video, f"./{output_dir}/output.mp4", value_range="-1,1")
    print(f"Saved video to ./{output_dir}/output.mp4")
