# Environment notes

The release recipe is curated from the **existing Gene environment** used for
the tests in [TEST_REPORT.md](TEST_REPORT.md). It is not a raw `conda env export`:
training, other Wan tasks, legacy renderers and unrelated project dependencies
are intentionally excluded. The original environment is not modified.

## Installation layers

1. `environment.yml`: Python 3.11.14, pip 25.3, FFmpeg and build helpers.
2. Official PyTorch CUDA 12.1 wheels: torch 2.4.1, torchvision 0.19.1,
   torchaudio 2.4.1. Install these **before** requirements-release.txt.
3. `requirements-release.txt`: pinned direct model, geometry, video and UI
   dependencies from Gene. It is not a complete transitive lockfile.
4. FlashAttention 2.7.3: installed after torch, with build isolation disabled.
   A matching CUDA toolkit/compiler is needed if pip builds from source. CUDA
   runtime bundled in a PyTorch wheel does not supply `nvcc`.
5. VGGT source + VGGT/Wan weights: separate from the Python environment.

Use the [PyTorch version matrix](https://pytorch.org/get-started/previous-versions/)
for the specific torch/torchvision/CUDA pairing. Do not change one member alone.
The local Wan attention implementation uses FlashAttention for model inference;
CPU geometry tests do not need it.

## Scope of verification

- Observed host: Linux x86_64, NVIDIA H200, Python 3.11.14,
  torch 2.4.1+cu121, torchvision 0.19.1+cu121, torchaudio 2.4.1+cu121,
  flash-attn 2.7.3.
- Versions in requirements-release.txt were checked against installed package
  metadata. Gene contains both OpenCV wheel variants; this recipe installs only
  `opencv-python` to avoid having two distributions provide the same `cv2` module.
- Gradio 5.12.0 is used with gradio-client 1.5.4, Pydantic 2.12.5 and
  FastAPI 0.124.0. Public callback API documentation is disabled in this app to
  avoid an old Gradio boolean-schema compatibility issue. Do not remove that
  setting without testing the dependency combination.
- The tested VGGT source was an existing local checkout within another project;
  it does not expose an independently pinned upstream VGGT revision. The README
  gives the official source URL and checks the required imports. A fresh upstream
  checkout is not claimed to be byte-identical to that local copy.
- **A clean installation from scratch has not yet been tested.** No packages or
  model weights were downloaded merely to write this guide. Run the environment
  checker, `python -m pip check`, CPU tests, and a small GPU test on a new machine.
- This recipe targets the new personal-data CLI/UI, not every legacy benchmark
  script or every task in the vendored Wan repository. The legacy launchers may
  require additional dataset helpers, renderers and dependencies.

Documentation validation on 2026-09-29: both environment-checker modes passed
(the GPU/source/path check used the existing server checkpoint config), all 11
release tests passed, and the README's synthetic-data / CPU-render commands
completed. The example camera JSON and both YAML files parsed successfully.
Every pinned runtime version matched Gene; a no-network `pip install --dry-run
--no-index -r requirements-release.txt` required no changes. However, full-environment
`pip check` reported `decord 0.6.0 is not supported on this platform` in the
pre-existing shared environment. Decord is not part of this new workflow/recipe;
it was not removed or changed. These checks do not establish a clean-install test.

## Resource expectations

The supported model here is Wan2.2-I2V-A14B, **not** the smaller TI2V-5B model.
CPU offloading reduces GPU pressure but shifts model storage to host RAM. Plan
for substantial host RAM and checkpoint disk space as well as GPU memory. The
small release tests were run on H200; no minimum VRAM for other cards is claimed.
The earlier 49-frame research run is not a full-resolution qualification of the
new release renderer or UI. Start with the README's reduced smoke settings.

For low-bandwidth environments, reuse local weights, run tests on the server,
and inspect small text logs before transferring videos/point clouds. Setup is
the only stage that needs downloading source/packages/weights; the new runner
uses configured checkpoint paths instead of fetching weights automatically.
