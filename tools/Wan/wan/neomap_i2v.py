# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class NeoMap:
    # NeoMap projection and denoising pipeline for Wan I2V A14B.
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        seed=42
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * config.min_step_percent)
        self.max_step = int(self.num_train_timesteps * config.max_step_percent)
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )
        self.text_encoder.model.eval()
        for param in self.text_encoder.model.parameters():
            param.requires_grad_(False)
            
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        self.vae.model.eval()
        for param in self.vae.model.parameters():
            param.requires_grad_(False)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint).eval()
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        for param in self.low_noise_model.parameters():
            param.requires_grad_(False)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint).eval()
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        for param in self.high_noise_model.parameters():
            param.requires_grad_(False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)


    @torch.no_grad()
    def predict_velocity(self,
                         latent,
                         t,
                         arg_c,
                         arg_null,
                         boundary,
                         guide_scale,
                         offload_model):

        latent_model_input = [latent.to(self.device)]
        timestep = [t]

        timestep = torch.stack(timestep).to(self.device)

        model = self._prepare_model_for_timestep(
            t, boundary, offload_model)
        sample_guide_scale = guide_scale[1] if t.item(
        ) >= boundary else guide_scale[0]

        velocity_cond = model(
            latent_model_input, t=timestep, **arg_c)[0]
        if offload_model:
            torch.cuda.empty_cache()
        velocity_uncond = model(
            latent_model_input, t=timestep, **arg_null)[0]
        if offload_model:
            torch.cuda.empty_cache()
        velocity_pred = velocity_uncond + sample_guide_scale * (
            velocity_cond - velocity_uncond)
        
        return velocity_pred

    @torch.no_grad()
    def _pixel_constrained_projection(
            self, anchored_latent, warped_prior_video, visibility_mask_px):
        """Equation (7): project onto the NVS constraint set in pixel space."""
        decoded_video = self.vae.decode([anchored_latent])[0]
        constrained_video = torch.where(
            visibility_mask_px, warped_prior_video, decoded_video)
        return self.vae.encode([constrained_video])[0]

    def _fully_visible_vae_block_masks(self, visibility_mask_px):
        """Keep a VAE compression block only when every source pixel is visible.

        Wan's causal temporal packing treats frame zero independently and then
        packs subsequent frames in groups of ``vae_stride[0]``.  Spatial blocks
        are non-overlapping ``vae_stride[1] x vae_stride[2]`` regions.

        Args:
            visibility_mask_px: Boolean tensor shaped [1, F, H, W].

        Returns:
            A pair ``(strict_pixel_mask, strict_latent_mask)``.  The pixel mask
            expands each block decision back over all pixels in that block; the
            latent mask contains one Boolean decision per compression block.
        """
        temporal_stride, height_stride, width_stride = self.vae_stride
        _, frame_num, height, width = visibility_mask_px.shape
        if (frame_num - 1) % temporal_stride != 0:
            raise ValueError(
                f"frame_num must be 1 mod {temporal_stride}, got {frame_num}")
        if height % height_stride != 0 or width % width_stride != 0:
            raise ValueError(
                "visibility mask dimensions must be divisible by the VAE "
                f"spatial stride {(height_stride, width_stride)}, got "
                f"{(height, width)}")

        mask_float = visibility_mask_px.float()
        first_block = torch.nn.functional.avg_pool2d(
            mask_float[:, :1],
            kernel_size=(height_stride, width_stride),
            stride=(height_stride, width_stride)) == 1.0

        if frame_num > 1:
            remaining_blocks = torch.nn.functional.avg_pool3d(
                mask_float[:, 1:].unsqueeze(1),
                kernel_size=(temporal_stride, height_stride, width_stride),
                stride=(temporal_stride, height_stride, width_stride)
            ).squeeze(1) == 1.0
            strict_latent_mask = torch.cat(
                [first_block, remaining_blocks], dim=1)
        else:
            strict_latent_mask = first_block

        first_pixel_block = first_block.repeat_interleave(
            height_stride, dim=2).repeat_interleave(width_stride, dim=3)
        if frame_num > 1:
            remaining_pixel_blocks = remaining_blocks.repeat_interleave(
                temporal_stride, dim=1).repeat_interleave(
                    height_stride, dim=2).repeat_interleave(
                        width_stride, dim=3)
            strict_pixel_mask = torch.cat(
                [first_pixel_block, remaining_pixel_blocks], dim=1)
        else:
            strict_pixel_mask = first_pixel_block

        return strict_pixel_mask, strict_latent_mask

    @torch.no_grad()
    def __call__(self,
                 warped_prior_video,
                 reference_image,
                 visibility_mask_px,
                 input_prompt="",
                 max_area=720 * 1280,
                 tau=0.98,
                 shift=3.5,
                 sampling_steps=15,
                 projection_iterations=15,
                 reanchor_steps=3,
                 guide_scale=3.0,
                 n_prompt="",
                 offload_model=True,
                 similarity_threshold=0.8,
                 strict_vae_block_visibility=True):

        """Generate a video with the NeoMap schedule described in the paper.

        The implementation intentionally keeps one detail from the experimental
        code: it starts at pure Gaussian noise at t=1 and performs one reverse
        flow step to ``tau``.  Alternating AMP/PCP projections then optimize the
        state at the fixed timestamp ``tau``.  Finally, a fresh shifted schedule
        advances the optimized state from ``tau`` to zero.

        ``sampling_steps`` is the total number of reverse-flow solver steps,
        including the bootstrap step from 1 to ``tau``.  Consequently the
        shifted ``tau``-to-zero schedule contains ``sampling_steps - 1`` steps.
        """

        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if shift <= 0.0:
            raise ValueError(f"shift must be positive, got {shift}")
        if sampling_steps < 2:
            raise ValueError("sampling_steps must include one bootstrap step and at least one generation step")
        if projection_iterations < 1:
            raise ValueError("projection_iterations must be at least 1")
        if reanchor_steps < 0:
            raise ValueError("reanchor_steps must be non-negative")
        if reanchor_steps > sampling_steps - 2:
            raise ValueError(
                "reanchor_steps cannot exceed the timestamps strictly below tau")

        # ====================================================================================================
        # Input Check and preprocess
        # ====================================================================================================
        frame_num = warped_prior_video.shape[0]
        assert frame_num % 4 == 1, "frame_num must be 4n+1"
        assert warped_prior_video.shape[1] == 3, "warped_prior_video must be [F, 3, h, w]"

        warped_prior_video = warped_prior_video.sub(0.5).div(0.5).to(self.device)
        
        if reference_image is not None:
            condition_image = reference_image.sub(0.5).div(0.5).to(self.device)
        else:
            condition_image = warped_prior_video[0]
            

        # ====================================================================================================
        # Calculate latent dimensions and process the image shape
        # ====================================================================================================
        h, w = condition_image.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1]) # 120
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2]) # 120
        h = lat_h * self.vae_stride[1] # 120 * 8 = 960
        w = lat_w * self.vae_stride[2] # 120 * 8 = 960
        print(f"lat_h: {lat_h}, lat_w: {lat_w}")
        print(f"h: {h}, w: {w}")
        
        # ====================================================================================================
        # Interpolate frames to valid size and VAE encode to latent space
        # ====================================================================================================
        if warped_prior_video.shape[2:] == (h, w):
            warped_prior_video = warped_prior_video.transpose(0, 1).to(self.device)
            visibility_mask_px = visibility_mask_px.bool().transpose(0, 1).to(self.device)
        else:
            warped_prior_video = torch.nn.functional.interpolate(
                warped_prior_video, size=(h, w), mode='bicubic'
            ).transpose(0, 1).to(self.device)
            visibility_mask_px = torch.nn.functional.interpolate(
                visibility_mask_px.float(), size=(h, w), mode='nearest'
            ).bool().transpose(0, 1).to(self.device)

        # ====================================================================================================
        # Prepare pixel and latent masks.  In strict mode, a compression block
        # is visible only if every pixel contributing to its stride cell is
        # visible; otherwise every pixel in that block becomes invisible.
        # ====================================================================================================
        if strict_vae_block_visibility:
            visibility_mask_px, visibility_mask_latent = (
                self._fully_visible_vae_block_masks(visibility_mask_px))
            # Newly rejected pixels are semantically invisible.  Keep the
            # conventional black representation used by the warped prior.
            warped_prior_video = torch.where(
                visibility_mask_px,
                warped_prior_video,
                torch.full_like(warped_prior_video, -1.0))
        else:
            lat_t = (frame_num - 1) // self.vae_stride[0] + 1
            mask_5d = visibility_mask_px.unsqueeze(0).float()
            mask_downsampled = torch.nn.functional.interpolate(
                mask_5d,
                size=(lat_t, lat_h, lat_w),
                mode='nearest'
            ).squeeze(0)
            visibility_mask_latent = mask_downsampled > 0.5
        
        # ====================================================================================================
        # Encode frames to latent space
        # ====================================================================================================
        prior_latent = self.vae.encode([warped_prior_video])[0]

        # ====================================================================================================
        # Pure Gaussian state at t=1.  This is the source of the bootstrap
        # reverse-flow step 1 -> tau and is not obtained by noising the prior.
        # ====================================================================================================
        initial_noise = torch.randn(
            16,                                 # VAE latent channels
            (frame_num - 1) // self.vae_stride[0] + 1,  # Temporal dimension after VAE
            lat_h,                              # Height in latent space      
            lat_w,                              # Width in latent space
            dtype=torch.float32,
            generator=self.generator,
            device=self.device) # [16, (F - 1) // 4 + 1, lat_h, lat_w] = [16, 21, 120, 120]

        # ====================================================================================================
        state_t = initial_noise

        # ====================================================================================================
        # Get guide_scale
        # ====================================================================================================
        guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
    
        # ====================================================================================================
        # Calculate max_seq_len
        # ====================================================================================================
        F = frame_num
        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # ====================================================================================================
        # Create mask for image-to-video conditioning
        # ====================================================================================================
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device) # [1, F, lat_h, lat_w] = [1, 81, 120, 120]
        msk[:, 1:] = 0 # Only first frame is conditioned
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), # [1, 4, lat_h, lat_w] = [1, 4, 120, 120] (repeat first frame 4 times)
            msk[:, 1:]                                              # [1, F-1, lat_h, lat_w] = [1, 80, 120, 120]    (frames 1-80)
        ], dim=1)                                                   # [1, F+3, lat_h, lat_w] = [1, 84, 120, 120]
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)       # [1, (F+3)//4, 4, lat_h, lat_w] = [1, 21, 4, 120, 120]
        msk = msk.transpose(1, 2)[0]                                # [4, (F+3)//4, lat_h, lat_w] = [4, 21, 120, 120]

        # ====================================================================================================
        # Preprocess prompt
        # ====================================================================================================
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # ====================================================================================================
        # VAE Encode input image as the first frame + zero future frames
        # ====================================================================================================
        # Frame 0: Input image
        # Frame 1 to F-1: Zero frames for future generations
        # encode: [3, F, h, w] -> [16, 21, lat_h, lat_w]
        y = self.vae.encode([
            torch.concat([
                    torch.nn.functional.interpolate(condition_image[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)                                                                     # [3, F-1, h, w]
            ],dim=1).to(self.device)                                                                            # [3, F, h, w]
        ])[0] # [16, 21, lat_h, lat_w] = [16, 21, 120, 120]
        y = torch.concat([msk, y]) # [4, 21, 120, 120] + [16, 21, 120, 120] = [20, 21, 120, 120]

        # ====================================================================================================
        # No-sync context manager
        # ====================================================================================================
        @contextmanager
        def noop_no_sync():
            yield
        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync', noop_no_sync)

        # ====================================================================================================
        # Denoising (Evaluation mode)
        # ====================================================================================================
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()
            
            def raw_time_from_shifted_time(shifted_time):
                """Invert s*t/(1+(s-1)*t) so tau is specified after shifting."""
                return shifted_time / (shift - (shift - 1.0) * shifted_time)

            def make_scheduler(raw_sigmas):
                scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                raw_sigmas = np.asarray(raw_sigmas, dtype=np.float32)
                scheduler.set_timesteps(
                    len(raw_sigmas),
                    device=self.device,
                    sigmas=raw_sigmas,
                    shift=shift)
                return scheduler

            def sample_standard_noise():
                return torch.randn(
                    initial_noise.shape,
                    dtype=initial_noise.dtype,
                    generator=self.generator,
                    device=self.device)

            def anchored_manifold_projection(state, timestamp, shifted_time):
                """Equations (3)-(5): Euler manifold estimate plus AMP anchoring."""
                velocity = self.predict_velocity(
                    state, timestamp, arg_c, arg_null, boundary,
                    guide_scale, offload_model)
                sigma_t = torch.as_tensor(
                    shifted_time, device=state.device, dtype=state.dtype)
                manifold_latent = state - sigma_t * velocity
                similarity = torch.nn.functional.cosine_similarity(
                    manifold_latent, prior_latent, dim=0)
                reliable_prior_mask = (
                    (similarity > similarity_threshold) & visibility_mask_latent)
                anchored_latent = torch.where(
                    reliable_prior_mask, prior_latent, manifold_latent)
                visible_count = visibility_mask_latent.float().sum().clamp_min(1.0)
                anchor_ratio = reliable_prior_mask.float().sum() / visible_count
                return anchored_latent, anchor_ratio

            def renoise_to_timestamp(data_latent, shifted_time):
                """Equation (6) with an independent Gaussian sample each call."""
                iteration_noise = sample_standard_noise()
                return ((1.0 - shifted_time) * data_latent
                        + shifted_time * iteration_noise)

            raw_tau = raw_time_from_shifted_time(tau)

            # Stage 0: preserve the implementation detail used in experiments.
            # Start from pure noise at t=1 and take one solver step to tau.
            bootstrap_scheduler = make_scheduler([1.0, raw_tau])
            t_one = bootstrap_scheduler.timesteps[0]
            bootstrap_velocity = self.predict_velocity(
                state_t, t_one, arg_c, arg_null, boundary,
                guide_scale, offload_model)
            state_tau = bootstrap_scheduler.step(
                bootstrap_velocity.unsqueeze(0),
                t_one,
                state_t.unsqueeze(0),
                return_dict=False,
                generator=self.generator)[0].squeeze(0)
            del bootstrap_scheduler

            # Stage 1: K alternating AMP/PCP projections at the fixed tau.
            tau_timestep = torch.tensor(
                int(round(tau * self.num_train_timesteps)),
                device=self.device,
                dtype=torch.int64)
            for projection_index in range(projection_iterations):
                anchored_latent, anchor_ratio = anchored_manifold_projection(
                    state_tau, tau_timestep, tau)
                pixel_constrained_latent = self._pixel_constrained_projection(
                    anchored_latent, warped_prior_video, visibility_mask_px)
                state_tau = renoise_to_timestamp(pixel_constrained_latent, tau)
                print(
                    f"Projection {projection_index + 1}/{projection_iterations}: "
                    f"anchored {anchor_ratio.item() * 100:.03f}% of visible latents")

            # Stage 2: shifted solver schedule from tau to zero.  The bootstrap
            # consumes one of the total requested reverse-flow steps.
            generation_steps = sampling_steps - 1
            raw_generation_sigmas = np.linspace(
                raw_tau, 0.0, generation_steps + 1,
                dtype=np.float32)[:-1]
            generation_scheduler = make_scheduler(raw_generation_sigmas)
            state_t = state_tau

            for denoise_index, t in enumerate(tqdm(generation_scheduler.timesteps)):
                # Re-anchor only at new timestamps strictly below tau.  PCP is
                # intentionally absent from auxiliary trajectory re-anchoring.
                if 0 < denoise_index <= reanchor_steps:
                    shifted_time = generation_scheduler.sigmas[denoise_index].to(
                        device=state_t.device, dtype=state_t.dtype)
                    anchored_latent, anchor_ratio = anchored_manifold_projection(
                        state_t, t, shifted_time)
                    state_t = renoise_to_timestamp(
                        anchored_latent, shifted_time)
                    print(
                        f"Re-anchor {denoise_index}/{reanchor_steps}: "
                        f"anchored {anchor_ratio.item() * 100:.03f}% of visible latents")

                velocity_pred = self.predict_velocity(
                    state_t, t, arg_c, arg_null, boundary,
                    guide_scale, offload_model)
                state_t = generation_scheduler.step(
                    velocity_pred.unsqueeze(0),
                    t,
                    state_t.unsqueeze(0),
                    return_dict=False,
                    generator=self.generator)[0].squeeze(0)

            x0 = [state_t]

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)
                
        del initial_noise, state_t, state_tau, x0
        del generation_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
