# NeoMap release entry points

This release includes the configuration-driven research pipeline under its
canonical name, `tools/Wan/wan/neomap_i2v.py`. New CLI/UI code does not import the dataset-specific
main or `colmap_camera_utils`; no external Gene `PYTHONPATH` is required.

## Environment and weights

Use a Wan-compatible Python/CUDA environment (on this server: `conda activate Gene`).
Model, UI and CPU dependencies are listed in `requirements-release.txt`.
Set `checkpoints.wan`, `checkpoints.vggt` and `repositories.vggt` in
`configs/neomap.example.yaml`, the portable default. For private paths, copy it
to Git-ignored `configs/neomap.local.yaml` and pass it via `--config`.
Relative paths resolve against the config directory. Weights are not distributed
in this repository. VGGT code is external and must be obtained
separately on another machine. Respect upstream code and model licenses.

## Camera/depth format v1

`cameras.json` contains:

```json
{
  "version": 1,
  "coordinate_convention": "opencv",
  "pose_convention": "camera_to_world",
  "depth_convention": "camera_z",
  "width": 640, "height": 480, "fps": 8,
  "source_camera": {
    "intrinsics": [[600,0,320],[0,600,240],[0,0,1]],
    "camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  },
  "target_cameras": []
}
```

Fill `target_cameras` with 5,9,...,81 camera objects in output order (49 default).
OpenCV axes: X right, Y down, Z forward. Poses map camera points into world space.
Depth is positive camera Z, NOT Euclidean ray distance or inverse depth. Supply
float H×W `.npy` or a depth PNG with `--depth-scale` (e.g. 0.001 for mm→m).
Depth and pose translations MUST use the same units. Invalid depths: zero/NaN.
RGB, depth, width/height and intrinsics must agree; no hidden crop/resize occurs.

## CLI

Run inside this release directory (quote paths containing spaces):

```bash
CUDA_VISIBLE_DEVICES=4 python main_neomap.py explicit \
  --image input.png --depth depth.npy --trajectory cameras.json \
  --output-dir outputs/my_scene

CUDA_VISIBLE_DEVICES=4 python main_neomap.py orbit --image input.png \
  --orbit-angle 30 --radius-scale 1 --num-frames 49 \
  --output-dir outputs/orbit

python main_neomap.py bundle --bundle outputs/ui/SESSION/exports/EXPORT \
  --output-dir outputs/replay --prepare-only
```

`orbit` runs VGGT even with `--prepare-only`; explicit/bundle preparation is CPU-only.
The pivot lies on the source optical axis at the median central-region depth.
Orbit radius defaults to source-to-pivot distance; a changed radius smoothly
transitions from the initial radius, preserving the exact initial camera.
VGGT scale is arbitrary. Multi-image reconstruction uses one shared VGGT world;
the chosen source image conditions Wan. Point colors fuse by nearest-depth
z-buffer, not learned blending; inconsistent geometry can still cause ghosts.

`--tau .98 --steps 15 --projections 15 --reanchors 3 --shift 3.5` are defaults.
One bootstrap step goes directly 1→τ; fixed-τ AMP/PCP follows; a fresh shifted
schedule spans τ→0. This release does not insert user-supplied pre-τ grid nodes.
Strict VAE block visibility remains enabled. It excludes partly invisible stride
blocks, not the encoder's full receptive field.

CPU rasterization is a NEW conservative z-buffer point-splat renderer, not the
legacy CamPlanner renderer; pixel-identical legacy priors are not promised.
`--splat-radius` controls point coverage; `--mask-kernel` erodes visibility before
the original strict VAE mask processing. Default kernel is 3; use 15 for stronger
erosion. Preview black regions are unobserved, not reconstructed surfaces.

## Interactive workbench

```bash
python app_neomap.py --port 7860
# From your own computer (do not expose an unauthenticated UI publicly):
ssh -L 7860:127.0.0.1:7860 cubecloud-h200
```

Open `http://127.0.0.1:7860`. Upload 1–16 images, choose a free physical GPU,
reconstruct, and select a source image. 3D-viewer dragging only inspects the cloud.
Orbit/translation controls define the actual output camera; click Preview then
Confirm to add a keyframe. Source selection resets the path. At least two keys
are needed. Interpolation reaches all keys exactly, with smooth easing and
zero velocity at keys; it is not a constant-speed cinematic path.

Save creates a self-contained server bundle: `reference.png`, `depth.npy`,
`cameras.json`, `keyframes.json`, `scene_metadata.json`, and `scene.npz`. The camera
download button downloads only JSON; copy the full bundle when moving machines.
Generate creates an immutable snapshot of current keys and starts an independent
process. Status/log updates every 3 seconds; Cancel terminates its process group.
Jobs from this UI share a per-user GPU lock; other processes are not forcibly
stopped. A busy GPU causes a clear error rather than sharing it. The UI does not
automatically schedule retries. CPU render-only previews do not load Wan.

Load reconstructs eight evenly spaced generated video frames with VGGT, after
generation has exited. Its cloud has an independent scale/pose; it is not aligned
to the input cloud and is not ground-truth geometry. Video playback is available
without this extra reconstruction. Logs and large artifacts stay on the server.
UI job tracking is in-memory; files/logs survive restart but jobs are not reattached.
The browser receives at most 30,000 display points. Full geometry remains in the
server-side scene file. PLY is exported for external point-cloud software; the
embedded viewer uses Plotly, not a Gaussian-splat PLY loader.

## Tests

```bash
python -m unittest discover -s tests -v
python scripts/make_synthetic.py
python main_neomap.py explicit --image outputs/synthetic/reference.png \
  --depth outputs/synthetic/depth.npy --trajectory outputs/synthetic/cameras.json \
  --output-dir outputs/cpu_smoke --prepare-only
```

For a cheap actual Wan wiring test, add `--steps 3 --projections 1 --reanchors 0
--max-area 16384` and remove `--prepare-only`. This checks execution, not quality.
Full-resolution/default-step visual quality needs separate real-scene evaluation.
All `output/` and `outputs/` directories at any depth are ignored by `.gitignore`.
Machine-specific configs/scripts are ignored through `*.local.*` and private
directories. The dataset entry point remains `main_static_wan22_inner_loop.py`;
the model pipeline is `tools/Wan/wan/neomap_i2v.py`, and the wrapper is `ops/vggt.py`.
