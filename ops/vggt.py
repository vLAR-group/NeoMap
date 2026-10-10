"""Config-driven VGGT wrapper used by the NeoMap entry points."""

import sys
from pathlib import Path

import torch


class ConfiguredVGGTWrapper:
    """Load VGGT source and weights from explicit, user-configured paths."""

    def __init__(self, repository_path, checkpoint_path, device="cuda"):
        repository_path = Path(repository_path).expanduser()
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not repository_path.is_dir():
            raise FileNotFoundError(
                f"VGGT repository does not exist: {repository_path}")
        if not (repository_path / "vggt").is_dir():
            raise FileNotFoundError(
                f"VGGT Python package is missing under: {repository_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"VGGT checkpoint does not exist: {checkpoint_path}")

        repository_string = str(repository_path)
        if repository_string not in sys.path:
            sys.path.insert(0, repository_string)

        # Imports are intentionally delayed until the configured repository has
        # been added to sys.path.
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.device = device
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16)
        self._load_and_preprocess_images = load_and_preprocess_images
        self._pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        self._unproject_depth_map_to_point_map = unproject_depth_map_to_point_map

        self.vggt = VGGT()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.vggt.load_state_dict(state_dict, strict=True)
        self.vggt.eval().to(self.device)

        self.depth = None
        self.images = None
        self.avg_focal = None

    def run_init(self, images):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
            predictions = self.vggt(images)

        extrinsic, intrinsic = self._pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        self.depth = predictions["depth"]

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)
        predictions["pose_enc_list"] = None

        predictions["world_points_from_depth"] = (
            self._unproject_depth_map_to_point_map(
                predictions["depth"],
                predictions["extrinsic"],
                predictions["intrinsic"]))
        self.avg_focal = (
            predictions["intrinsic"][0][0, 0]
            + predictions["intrinsic"][0][1, 1]) / 2.0
        return predictions

    def run_vggt_init(self, input_image_paths):
        images = self._load_and_preprocess_images(input_image_paths).to(
            self.device)
        return self.run_init(images)
