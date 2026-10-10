# Release verification

The initial personal-data entry points were smoke-tested on 2026-09-29 in the
environment described in [ENVIRONMENT.md](ENVIRONMENT.md). Tests covered CPU
camera geometry, exact identity reprojection, z-buffer depth ordering, 5/49/81
frame paths, bundle round trips and UI camera editing/background CPU jobs.

A reduced GPU run produced a five-frame 144×96 Wan video at 8 fps. The test used
three reverse-flow steps, one projection, no reanchors and max_area=16384. This
checks execution rather than full-resolution image quality. Single-image orbit,
two-image VGGT reconstruction and VGGT reconstruction of generated frames also
completed. No 49/81-frame full-resolution release benchmark is claimed.

During publication cleanup, canonical module names and portable configuration
defaults replace the old development suffixes. Only the Wan I2V A14B runtime
components and the dataset renderer dependencies are retained. The original
projection/denoising algorithm is unchanged.

Reproduce the CPU checks with:

```bash
python -m unittest discover -s tests -v
python scripts/check_environment.py --cpu-only
python scripts/make_synthetic.py --output-dir outputs/synthetic
python main_neomap.py bundle --bundle outputs/synthetic \
  --output-dir outputs/cpu_smoke --prepare-only
```

Run the GPU/source import check with a config pointing to your installed weights:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py --config configs/neomap.local.yaml
```

Private historical server paths and logs are kept outside public documentation.
