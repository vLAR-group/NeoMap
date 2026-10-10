# Wan components for NeoMap

This directory retains the Wan I2V A14B model, VAE, text encoder, scheduler,
distributed helpers and optional prompt utilities used by NeoMap. The NeoMap
pipeline is `wan/neomap_i2v.py`.

Use the installation and inference instructions in the [NeoMap README](../../README.md).
The public entry points are `main_neomap.py`, `app_neomap.py`, and the retained
dataset entry point `main_static_wan22_inner_loop.py`.

Unused task launchers, training utilities, animation/SAM code and experimental
generation variants are not bundled in this release. For the full upstream
project, see [Wan2.2](https://github.com/Wan-Video/Wan2.2).

The original Wan copyright notices and [Apache 2.0 license](LICENSE.txt) are
retained. Third-party code keeps its upstream license.
