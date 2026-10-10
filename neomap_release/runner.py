"""Lazy model loading: CPU preparation never imports Wan or VGGT."""
import gc
import json
import os
from pathlib import Path
import random
import numpy as np
from .scene import render, unproject

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'configs/neomap.example.yaml'


def config(path=DEFAULT_CONFIG):
    import yaml
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    for group in ('checkpoints', 'repositories'):
        for key, value in data[group].items():
            p = Path(os.path.expandvars(value)).expanduser()
            data[group][key] = str(p if p.is_absolute() else path.parent/p)
    return data


def reconstruct(paths, config_path=DEFAULT_CONFIG, confidence_quantile=0.1):
    import torch
    from ops.vggt import ConfiguredVGGTWrapper
    cfg = config(config_path)
    print('STAGE: VGGT reconstruction', flush=True)
    wrapper = ConfiguredVGGTWrapper(cfg['repositories']['vggt'], cfg['checkpoints']['vggt'])
    try:
        prediction = wrapper.run_vggt_init([str(p) for p in paths])
        # VGGT preprocessing determines both image coordinates and intrinsics.
        images = prediction.get('images')
        if images is None:
            images = wrapper._load_and_preprocess_images([str(p) for p in paths]).numpy()
        images = np.clip(np.moveaxis(images, 1, -1)*255, 0, 255).astype(np.uint8)
        depths = np.asarray(prediction['depth'])[..., 0].copy()
        extrinsic = np.asarray(prediction['extrinsic'])
        poses = np.repeat(np.eye(4)[None], len(paths), axis=0)
        poses[:, :3, :] = extrinsic
        poses = np.linalg.inv(poses)
        intrinsics = np.asarray(prediction['intrinsic'])
        confidence = prediction.get('depth_conf')
        points, colors = [], []
        for i in range(len(paths)):
            if confidence is not None:
                c = np.asarray(confidence[i]).squeeze()
                finite = c[np.isfinite(c)]
                if finite.size:
                    depths[i][(~np.isfinite(c)) | (c < np.quantile(finite, confidence_quantile))] = 0
            from .scene import camera
            xyz, rgb = unproject(images[i], depths[i], camera(intrinsics[i], poses[i]))
            points.append(xyz)
            colors.append(rgb)
        if not sum(map(len, points)):
            raise ValueError('VGGT produced no valid points.')
        return dict(images=images, depths=depths, intrinsics=intrinsics, poses=poses,
                    points=np.concatenate(points), colors=np.concatenate(colors))
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


def prepare(scene, traj, output, splat_radius=1, mask_kernel=3):
    import imageio.v2 as imageio
    from scipy.ndimage import binary_erosion
    if mask_kernel < 1 or mask_kernel % 2 != 1:
        raise ValueError('mask_kernel must be positive and odd.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rgbs, masks = [], []
    print('STAGE: rendering RGB and visibility priors', flush=True)
    with imageio.get_writer(str(output/'prior.mp4'), fps=traj['fps'], macro_block_size=1) as writer:
        for i, cam in enumerate(traj['target_cameras']):
            rgb, mask = render(scene, cam, traj['width'], traj['height'], splat_radius)
            if mask_kernel > 1:
                mask = binary_erosion(mask, structure=np.ones((mask_kernel, mask_kernel)), border_value=1)
            rgb[~mask] = 0
            writer.append_data(rgb)
            rgbs.append(rgb)
            masks.append(mask)
            print(f'RENDER: {i+1}/{len(traj["target_cameras"])}', flush=True)
    return np.stack(rgbs), np.stack(masks)


def generate(scene, source_index, traj, output, config_path=DEFAULT_CONFIG,
             prompt='A stationary scene. The camera moves smoothly.', seed=42,
             tau=.98, steps=15, projections=15, reanchors=3, shift=3.5,
             max_area=1280*720, offload=True, prepare_only=False,
             mask_kernel=3, splat_radius=1):
    # Reject invalid input before allocating a model.
    if not 0 < tau < 1 or shift <= 0 or steps < 2 or projections < 1 or not 0 <= reanchors <= steps-2:
        raise ValueError('Invalid tau/scheduler/projection parameters.')
    if not 0 <= seed < 2**32 or max_area < 256:
        raise ValueError('Seed must be uint32; max_area must be >=256.')
    from .scene import write_json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(prompt=prompt, seed=seed, tau=tau, steps=steps, projections=projections,
                    reanchors=reanchors, shift=shift, max_area=max_area, offload=offload,
                    prepare_only=prepare_only, mask_kernel=mask_kernel, splat_radius=splat_radius,
                    source_index=source_index, config=str(Path(config_path).resolve()),
                    scheduler='one bootstrap step 1->tau; fixed-tau projections; shifted tau->0',
                    renderer='CPU z-buffer point splats (new release renderer)')
    write_json(output/'run.json', settings)
    rgb, mask = prepare(scene, traj, output, splat_radius, mask_kernel)
    write_json(output/'visibility_stats.json', dict(frame_visible_fraction=mask.mean((1, 2)).tolist()))
    if prepare_only:
        print('DONE: prior preparation only; Wan inference was skipped', flush=True)
        return str(output/'prior.mp4')
    import torch
    import imageio.v2 as imageio
    from tools.Wan.wan.neomap_i2v import NeoMap
    from tools.Wan.wan.configs import WAN_CONFIGS
    cfg = config(config_path)
    if not Path(cfg['checkpoints']['wan']).is_dir():
        raise FileNotFoundError(cfg['checkpoints']['wan'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('STAGE: loading Wan', flush=True)
    wan_cfg = WAN_CONFIGS['i2v-A14B']
    model = NeoMap(config=wan_cfg, checkpoint_dir=cfg['checkpoints']['wan'], device_id=0,
                   convert_model_dtype=True, seed=seed, t5_cpu=True)
    final_prompt = prompt + wan_cfg.static_scene_positive_prompt
    (output/'final_prompt.txt').write_text(final_prompt)
    try:
        print('STAGE: Wan bootstrap / projection / denoising', flush=True)
        video = model(torch.from_numpy(rgb).permute(0, 3, 1, 2).float()/255,
                      torch.from_numpy(scene['images'][source_index].copy()).permute(2, 0, 1).float()/255,
                      torch.from_numpy(mask[:, None]), input_prompt=final_prompt,
                      max_area=max_area, tau=tau, shift=shift, sampling_steps=steps,
                      projection_iterations=projections, reanchor_steps=reanchors,
                      guide_scale=wan_cfg.sample_guide_scale, offload_model=offload,
                      similarity_threshold=.8, strict_vae_block_visibility=True)
        frames = ((video.detach().float().cpu().permute(1, 2, 3, 0).clamp(-1, 1)+1)*127.5).numpy().astype(np.uint8)
        if len(frames) != len(traj['target_cameras']):
            raise RuntimeError('Unexpected decoded frame count.')
        with imageio.get_writer(str(output/'output.mp4'), fps=traj['fps'], macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        print('DONE: '+str(output/'output.mp4'), flush=True)
        return str(output/'output.mp4')
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
