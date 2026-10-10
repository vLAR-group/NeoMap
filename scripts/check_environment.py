"""Check release dependencies and optional model paths without loading weights."""
import argparse
import importlib
import importlib.metadata as metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-only', action='store_true', help='Skip CUDA, FlashAttention and model-import checks')
    p.add_argument('--config', help='Also validate checkpoint/source paths (no weights are loaded)')
    a = p.parse_args()
    errors = []
    print('Python:', sys.version.split()[0])
    for module, package in [('numpy', 'numpy'), ('scipy', 'scipy'), ('PIL', 'Pillow'),
                            ('yaml', 'PyYAML'), ('imageio', 'imageio'), ('imageio_ffmpeg', 'imageio-ffmpeg'),
                            ('gradio', 'gradio'), ('plotly', 'plotly')]:
        try:
            importlib.import_module(module)
            print(f'OK {package}=={metadata.version(package)}')
        except Exception as exc:
            errors.append(f'{package}: {exc}')
    try:
        import imageio_ffmpeg
        print('FFmpeg:', imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        errors.append(f'FFmpeg: {exc}')
    if not a.cpu_only:
        try:
            import torch
            import torchvision
            import flash_attn
            from tools.Wan.wan.neomap_i2v import NeoMap
            print(f'OK torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA runtime={torch.version.cuda}')
            print(f'OK flash-attn={metadata.version("flash-attn")}, NeoMap import')
            if not torch.cuda.is_available():
                errors.append('CUDA is not available; check your driver and CUDA_VISIBLE_DEVICES.')
            else:
                print('GPU:', torch.cuda.get_device_name(0))
        except Exception as exc:
            errors.append(f'GPU/model dependencies: {exc}')
    if a.config:
        try:
            from neomap_release.runner import config
            cfg = config(a.config)
            wan = Path(cfg['checkpoints']['wan'])
            needed = ['high_noise_model', 'low_noise_model', 'google/umt5-xxl',
                      'Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth']
            for name in needed:
                if not (wan/name).exists():
                    errors.append(f'Missing Wan resource: {wan/name}')
            if not Path(cfg['checkpoints']['vggt']).is_file():
                errors.append('Missing VGGT checkpoint: '+cfg['checkpoints']['vggt'])
            repo = Path(cfg['repositories']['vggt'])
            if not (repo/'vggt').is_dir():
                errors.append(f'Missing VGGT source package: {repo}/vggt')
            elif not a.cpu_only:
                sys.path.insert(0, str(repo))
                from vggt.models.vggt import VGGT
                from vggt.utils.geometry import unproject_depth_map_to_point_map
                from vggt.utils.load_fn import load_and_preprocess_images
                from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                print('OK VGGT source imports')
        except Exception as exc:
            errors.append(f'Configuration: {exc}')
    if errors:
        print('\nCHECK FAILED:\n- '+'\n- '.join(errors))
        return 1
    print('\nCHECK PASSED (no model weights loaded; no inference performed).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
