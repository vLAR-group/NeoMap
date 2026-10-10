"""Dataset-independent NeoMap CLI. Use --help; no model imports for CPU tests."""
import argparse
from pathlib import Path
from neomap_release.scene import (read_trajectory, explicit_scene, orbit_trajectory,
                                  load_scene, export_bundle)
from neomap_release.runner import DEFAULT_CONFIG, reconstruct, generate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['explicit', 'orbit', 'bundle'])
    p.add_argument('--image')
    p.add_argument('--depth')
    p.add_argument('--trajectory')
    p.add_argument('--bundle', help='Directory exported by the interactive UI')
    p.add_argument('--depth-scale', type=float, default=1)
    p.add_argument('--orbit-angle', type=float, default=30)
    p.add_argument('--radius-scale', type=float, default=1)
    p.add_argument('--num-frames', type=int, default=49)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--output-dir', required=True)
    p.add_argument('--prepare-only', action='store_true', help='Render priors without loading Wan; orbit still needs VGGT')
    p.add_argument('--prompt', default='A stationary scene. The camera moves smoothly.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tau', type=float, default=.98)
    p.add_argument('--steps', type=int, default=15)
    p.add_argument('--projections', type=int, default=15)
    p.add_argument('--reanchors', type=int, default=3)
    p.add_argument('--shift', type=float, default=3.5)
    p.add_argument('--max-area', type=int, default=1280*720)
    p.add_argument('--mask-kernel', type=int, default=3)
    p.add_argument('--splat-radius', type=int, choices=[0, 1, 2], default=1)
    p.add_argument('--no-offload', action='store_true')
    return p


def main():
    import json
    p = parser()
    a = p.parse_args()
    index = 0
    if a.mode == 'explicit':
        if not all((a.image, a.depth, a.trajectory)):
            p.error('explicit requires --image, --depth and --trajectory')
        traj = read_trajectory(a.trajectory)
        scene = explicit_scene(a.image, a.depth, traj, a.depth_scale)
    elif a.mode == 'orbit':
        if not a.image:
            p.error('orbit requires --image')
        from neomap_release.scene import frame_count
        frame_count(a.num_frames)
        scene = reconstruct([a.image], a.config)
        traj = orbit_trajectory(scene, angle=a.orbit_angle, frames=a.num_frames, radius_scale=a.radius_scale)
    else:
        if not a.bundle:
            p.error('bundle requires --bundle')
        directory = Path(a.bundle)
        scene = load_scene(directory/'scene.npz')
        traj = read_trajectory(directory/'cameras.json')
        index = int(json.loads((directory/'scene_metadata.json').read_text())['source_index'])
    output = Path(a.output_dir).resolve()
    export_bundle(output/'inputs', scene, index, traj)
    generate(scene, index, traj, output, config_path=a.config, prompt=a.prompt, seed=a.seed,
             tau=a.tau, steps=a.steps, projections=a.projections, reanchors=a.reanchors,
             shift=a.shift, max_area=a.max_area, offload=not a.no_offload,
             prepare_only=a.prepare_only, mask_kernel=a.mask_kernel, splat_radius=a.splat_radius)


if __name__ == '__main__':
    main()
