"""Unified full-window inference for AC-WM, FastWAM, and GigaWorld."""

from __future__ import annotations

import os

import torch
from transformers import BatchFeature

from ..modules.wam_task_attn import canonical_masked_task


CHUNKS_PER_WINDOW = 4
RGB_FRAMES_PER_CHUNK = 9
ACTIONS_PER_CHUNK = 24
LATENTS_PER_CHUNK = 2
# Four-chunk training exposes up to seven clean history latents to the fourth
# two-latent chunk (1, 3, 5, 7 history latents across the four chunks).
ACWM_HISTORY_LATENTS = 7
FULL_WINDOW_RGB_FRAMES = 33
FULL_WINDOW_LATENTS = 9


def unified_inference_backend() -> str:
    """Select the unified inference backend; cache is the deployed default."""
    five_mode = os.environ.get("WAM_FIVE_MODE_INFERENCE_BACKEND")
    unified = os.environ.get("WAM_UNIFIED_INFERENCE_BACKEND")
    if (
        five_mode is not None
        and unified is not None
        and five_mode.strip().lower() != unified.strip().lower()
    ):
        raise ValueError(
            "WAM_FIVE_MODE_INFERENCE_BACKEND and "
            "WAM_UNIFIED_INFERENCE_BACKEND must match when both are set"
        )
    value = (five_mode or unified or "cache").strip().lower()
    if value not in {"dense", "cache"}:
        raise ValueError(
            "WAM_UNIFIED_INFERENCE_BACKEND must be dense or cache, "
            f"got {value!r}"
        )
    return value


def _emit_trace(head, *, task, backend, chunk_index, phase, step_index, **tensors):
    callback = getattr(head, "_wam_inference_trace_callback", None)
    if callback is not None:
        callback(
            task=task,
            backend=backend,
            chunk_index=int(chunk_index),
            phase=phase,
            step_index=int(step_index),
            tensors=tensors,
        )


def _pad_window(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    if tensor.shape[1] <= 0 or tensor.shape[1] > target_length:
        raise ValueError(
            f"WAM inference history must contain 1..{target_length} items, "
            f"got {tensor.shape[1]}"
        )
    if tensor.shape[1] == target_length:
        return tensor
    mode = os.environ.get("WAM_DENSE_PADDING_MODE", "repeat").strip().lower()
    if mode == "repeat":
        pad = tensor[:, -1:].expand(
            -1, target_length - tensor.shape[1], *tensor.shape[2:]
        )
    elif mode == "zero":
        pad = tensor.new_zeros(
            tensor.shape[0],
            target_length - tensor.shape[1],
            *tensor.shape[2:],
        )
    else:
        raise ValueError(
            "WAM_DENSE_PADDING_MODE must be repeat or zero, "
            f"got {mode!r}"
        )
    return torch.cat([tensor, pad], dim=1)


def _acwm_decoded_to_uint8(decoded: torch.Tensor) -> torch.Tensor:
    """Quantize streamed VAE output [-1,1] [B,C,T,H,W] to uint8 [B,T,H,W,3].

    The round-trip through uint8 (rather than keeping float pixels) matches
    the training input pipeline, where every condition frame is a real
    uint8 observation before VAE encoding.
    """
    frames = decoded.clamp(-1, 1).float()
    frames = (frames + 1.0) * 127.5
    return (
        frames.round()
        .to(torch.uint8)
        .permute(0, 2, 3, 4, 1)
        .contiguous()
    )


def _full_window_forward(
    head,
    *,
    task: str,
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    video_timestep: torch.Tensor,
    action_timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: torch.Tensor | None,
    ys: torch.Tensor | None,
    action: torch.Tensor,
    state: torch.Tensor | None,
    embodiment_id: torch.Tensor,
):
    """Call the exact dense full-window mask/token-group path used in training."""
    if noisy_latents.shape != clean_latents.shape:
        raise ValueError("clean/noisy full-window latent shapes must match")
    if noisy_latents.shape[1] != FULL_WINDOW_LATENTS:
        raise ValueError(
            f"full-window inference requires {FULL_WINDOW_LATENTS} latents, "
            f"got {noisy_latents.shape[1]}"
        )
    return head.model._forward_train_masked(
        task=task,
        x=noisy_latents.transpose(1, 2),
        clean_x=clean_latents.transpose(1, 2),
        timestep=video_timestep,
        timestep_action=action_timestep,
        context=context,
        seq_len=FULL_WINDOW_LATENTS * head.model.frame_seqlen,
        y=ys,
        clip_feature=clip_feature,
        action=action,
        state=state,
        embodiment_id=embodiment_id,
        full_window_history=True,
        full_window_chunked_attention=False,
    )


def masked_episode_step(
    head,
    action_input: BatchFeature,
    *,
    task: str,
    num_inference_steps: int | None = None,
) -> BatchFeature:
    """Predict one Fast/Giga action chunk from a partial four-chunk window."""
    head.set_frozen_modules_to_eval_mode()
    canonical_task = canonical_masked_task(task)
    if canonical_task not in {
        "action_only_fast_wam",
        "action_only_gigaworld_policy",
    }:
        raise ValueError(
            "unified masked episode inference only supports FastWAM/GigaWorld"
        )
    if head.moe_routing_granularity != "token_group_level":
        raise RuntimeError(
            "unified Fast/Giga inference requires token_group_level routing; "
            "set WAM_MOE_GRANULARITY=token_group_level or use legacy inference"
        )
    if unified_inference_backend() == "cache":
        try:
            return _masked_cached_episode_step(
                head,
                action_input,
                task=canonical_task,
                num_inference_steps=num_inference_steps,
            )
        except Exception:
            reset_unified_inference(head)
            raise

    images = action_input["images"]
    episode = head._wam_masked_episode
    if episode is None:
        if images.shape[1] != 1:
            raise ValueError(
                f"{canonical_task} window seed requires 1 RGB frame, "
                f"got {images.shape[1]}"
            )
        chunk_index = 0
        history = images.detach().clone()
    else:
        if episode["task"] != canonical_task:
            raise RuntimeError("masked inference task changed without an episode reset")
        chunk_index = int(episode["chunk_index"])
        if chunk_index >= CHUNKS_PER_WINDOW:
            raise RuntimeError("masked inference window is complete; reset before reuse")
        if images.shape[1] != RGB_FRAMES_PER_CHUNK:
            raise ValueError(
                f"{canonical_task} history commit requires 9 RGB frames, "
                f"got {images.shape[1]}"
            )
        history = torch.cat(
            [episode["history_images"], images[:, 1:].detach()], dim=1
        )
    expected_history = 1 + chunk_index * (RGB_FRAMES_PER_CHUNK - 1)
    if history.shape[1] != expected_history:
        raise RuntimeError(
            f"{canonical_task} chunk {chunk_index} requires "
            f"{expected_history} history frames, got {history.shape[1]}"
        )

    full_input = dict(action_input)
    full_input["images"] = _pad_window(history, FULL_WINDOW_RGB_FRAMES)
    full_input = BatchFeature(data=full_input)
    (
        latents,
        clip_feature,
        ys,
        prompt_embs,
        num_frames,
        _height,
        _width,
    ) = head._wam_encode_obs(
        full_input, force_single_frame_first_latent=False
    )
    if num_frames != FULL_WINDOW_LATENTS:
        raise RuntimeError(
            f"unified inference RGB window encoded to {num_frames} latents"
        )

    B = latents.shape[0]
    state = action_input.state.to(
        dtype=torch.bfloat16, device=head._device
    )[:, -1:]
    full_state = torch.zeros(
        B,
        CHUNKS_PER_WINDOW,
        state.shape[-1],
        dtype=state.dtype,
        device=state.device,
    )
    full_state[:, chunk_index:chunk_index + 1] = state
    action_start = chunk_index * ACTIONS_PER_CHUNK
    action_end = action_start + ACTIONS_PER_CHUNK
    noisy_action = head._wam_fixed_inference_noise(
        (B, ACTIONS_PER_CHUNK, head.action_dim)
    )
    full_action = torch.zeros(
        B,
        CHUNKS_PER_WINDOW * ACTIONS_PER_CHUNK,
        head.action_dim,
        dtype=noisy_action.dtype,
        device=noisy_action.device,
    )
    video_timestep = torch.zeros(
        B, FULL_WINDOW_LATENTS, device=head._device
    )
    action_timestep = torch.zeros(
        B, full_action.shape[1], device=head._device
    )
    embodiment_id = action_input.embodiment_id.to(head._device)
    steps = num_inference_steps or head.num_inference_steps
    head.scheduler.set_timesteps(
        steps, denoising_strength=head.denoising_strength
    )
    timesteps = head.scheduler.timesteps.to(head._device)
    head.model._current_wam_task = canonical_task
    if head.moe_enabled:
        head._set_wam_expert_mask(
            batch_size=B,
            phase="infer",
            action_input=action_input,
            task_names=[canonical_task] * B,
        )

    with torch.amp.autocast(
        dtype=torch.bfloat16,
        device_type=torch.device(head._device).type,
    ):
        for step_index, timestep in enumerate(timesteps):
            full_action.zero_()
            full_action[:, action_start:action_end] = noisy_action
            action_timestep.zero_()
            action_timestep[:, action_start:action_end] = float(timestep)
            _video_pred, action_pred = _full_window_forward(
                head,
                task=canonical_task,
                noisy_latents=latents,
                clean_latents=latents,
                video_timestep=video_timestep,
                action_timestep=action_timestep,
                context=prompt_embs,
                clip_feature=clip_feature,
                ys=ys,
                action=full_action,
                state=full_state,
                embodiment_id=embodiment_id,
            )
            noisy_action = head.scheduler.step(
                action_pred[:, action_start:action_end],
                timestep,
                noisy_action,
            )
            _emit_trace(
                head,
                task=canonical_task,
                backend="dense33",
                chunk_index=chunk_index,
                phase="action",
                step_index=step_index,
                prediction=action_pred[:, action_start:action_end],
                sample=noisy_action,
            )

    head._wam_masked_episode = {
        "task": canonical_task,
        "chunk_index": chunk_index + 1,
        "history_images": history,
    }
    return BatchFeature(data={"action": noisy_action})


def _masked_cached_episode_step(
    head,
    action_input: BatchFeature,
    *,
    task: str,
    num_inference_steps: int | None,
) -> BatchFeature:
    """Predict one Fast/Giga chunk with exact-prefix VAE and typed DiT K/V."""
    images = action_input["images"]
    episode = head._wam_masked_episode
    if episode is None:
        if images.shape[1] != 1:
            raise ValueError(
                f"{task} cache seed requires 1 RGB frame, got {images.shape[1]}"
        )
        chunk_index = 0
        encode_conditioning = True
        stream_mode = "start"
    else:
        if episode["task"] != task:
            raise RuntimeError("masked cache task changed without an episode reset")
        chunk_index = int(episode["chunk_index"])
        if chunk_index >= CHUNKS_PER_WINDOW:
            raise RuntimeError("masked cache window is complete; reset before reuse")
        if images.shape[1] != RGB_FRAMES_PER_CHUNK:
            raise ValueError(
                f"{task} cache history commit requires 9 RGB frames, "
                f"got {images.shape[1]}"
            )
        encode_conditioning = False
        stream_mode = "commit"

    prefix_input = dict(action_input)
    prefix_input = BatchFeature(data=prefix_input)
    (
        latents,
        clip_feature,
        _ys,
        prompt_embs,
        _num_frames,
        _height,
        _width,
    ) = head._wam_encode_obs(
        prefix_input,
        force_single_frame_first_latent=False,
        idm_stream_mode=stream_mode,
        encode_conditioning=encode_conditioning,
    )
    expected_latents = 1 if chunk_index == 0 else LATENTS_PER_CHUNK
    if latents.shape[1] != expected_latents:
        raise RuntimeError(
            f"{task} cache stream encoded to {latents.shape[1]} latents; "
            f"expected {expected_latents}"
        )

    batch_size = latents.shape[0]
    state = action_input.state.to(
        dtype=torch.bfloat16, device=head._device
    )[:, -1:]
    embodiment_id = action_input.embodiment_id.to(head._device)
    head.model._current_wam_task = task
    if head.moe_enabled:
        head._set_wam_expert_mask(
            batch_size=batch_size,
            phase="infer",
            action_input=action_input,
            task_names=[task] * batch_size,
        )

    if episode is None:
        head.model.reset_idm_cache()
        context = prompt_embs
        clip = clip_feature
        head.model.masked_cache_write_clean_video(
            latents[:, :1].transpose(1, 2),
            task=task,
            abs_frame_start=0,
            context=context,
            clip_feature=clip,
            cache_chunk_id=0,
        )
        head.model.masked_cache_write_seed(
            latents[:, :1].transpose(1, 2),
            task=task,
            context=context,
            clip_feature=clip,
            state=state,
            embodiment_id=embodiment_id,
        )
    else:
        context = episode["prompt"]
        clip = episode["clip"]
        new_start = 1 + (chunk_index - 1) * LATENTS_PER_CHUNK
        head.model.masked_cache_write_clean_video(
            latents.transpose(1, 2),
            task=task,
            abs_frame_start=new_start,
            context=context,
            clip_feature=clip,
            cache_chunk_id=chunk_index,
        )

    noisy_action = head._wam_fixed_inference_noise(
        (batch_size, ACTIONS_PER_CHUNK, head.action_dim)
    )
    steps = num_inference_steps or head.num_inference_steps
    head.scheduler.set_timesteps(
        steps, denoising_strength=head.denoising_strength
    )
    timesteps = head.scheduler.timesteps.to(head._device)
    with torch.amp.autocast(
        dtype=torch.bfloat16,
        device_type=torch.device(head._device).type,
    ):
        for step_index, timestep in enumerate(timesteps):
            action_pred = head.model.masked_cached_action_chunk(
                noisy_action,
                state,
                task=task,
                chunk_index=chunk_index,
                timestep_val=timestep,
                context=context,
                clip_feature=clip,
                embodiment_id=embodiment_id,
            )
            noisy_action = head.scheduler.step(
                action_pred, timestep, noisy_action
            )
            _emit_trace(
                head,
                task=task,
                backend="cache",
                chunk_index=chunk_index,
                phase="action",
                step_index=step_index,
                prediction=action_pred,
                sample=noisy_action,
            )

    head._wam_masked_episode = {
        "task": task,
        "chunk_index": chunk_index + 1,
        "prompt": context,
        "clip": clip,
    }
    return BatchFeature(data={"action": noisy_action})


def acwm_episode_step(
    head,
    action_input: BatchFeature,
    *,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> BatchFeature:
    """Teacher-forced AC-WM chunk prediction with dense 33-frame history."""
    head.set_frozen_modules_to_eval_mode()
    if head.moe_routing_granularity != "token_group_level":
        raise RuntimeError(
            "unified AC-WM inference requires token_group_level routing; "
            "set WAM_MOE_GRANULARITY=token_group_level or use legacy inference"
        )
    images = action_input["images"]
    episode = head._wam_acwm_unified_episode
    if episode is None:
        if images.shape[1] != 1:
            raise ValueError(
                f"AC-WM window seed requires 1 RGB frame, got {images.shape[1]}"
            )
        chunk_index = 0
        history_images = images.detach().clone()
    else:
        chunk_index = int(episode["chunk_index"])
        if chunk_index >= CHUNKS_PER_WINDOW:
            raise RuntimeError("AC-WM full window is complete; reset before reuse")
        if images.shape[1] != RGB_FRAMES_PER_CHUNK:
            raise ValueError(
                "AC-WM teacher-forced history commit requires 9 RGB frames, "
                f"got {images.shape[1]}"
            )
        history_images = torch.cat(
            [episode["history_images"], images[:, 1:].detach()], dim=1
        )
    expected_history = 1 + chunk_index * (RGB_FRAMES_PER_CHUNK - 1)
    if history_images.shape[1] != expected_history:
        raise RuntimeError(
            f"AC-WM chunk {chunk_index} requires {expected_history} history "
            f"RGB frames, got {history_images.shape[1]}"
        )

    full_input = dict(action_input)
    full_input["images"] = _pad_window(
        history_images, FULL_WINDOW_RGB_FRAMES
    )
    full_input = BatchFeature(data=full_input)
    (
        clean_latents,
        clip_feature,
        ys,
        prompt_embs,
        num_frames,
        _height,
        _width,
    ) = head._wam_encode_obs(
        full_input, force_single_frame_first_latent=False
    )
    if num_frames != FULL_WINDOW_LATENTS:
        raise RuntimeError(
            f"AC-WM RGB window encoded to {num_frames} latents"
        )
    B = clean_latents.shape[0]
    action = action_input.action.to(dtype=head.dtype, device=head._device)
    if action.shape[1] != ACTIONS_PER_CHUNK:
        raise ValueError(
            f"AC-WM requires {ACTIONS_PER_CHUNK} actions, got {action.shape[1]}"
        )
    action_start = chunk_index * ACTIONS_PER_CHUNK
    action_end = action_start + ACTIONS_PER_CHUNK
    full_action = torch.zeros(
        B,
        CHUNKS_PER_WINDOW * ACTIONS_PER_CHUNK,
        action.shape[-1],
        dtype=action.dtype,
        device=action.device,
    )
    full_action[:, action_start:action_end] = action
    video_start = 1 + chunk_index * LATENTS_PER_CHUNK
    video_end = video_start + LATENTS_PER_CHUNK
    channels, height, width = clean_latents.shape[2:]
    noisy_video = head.generate_noise(
        (B, LATENTS_PER_CHUNK, channels, height, width),
        seed=int(head.seed) + chunk_index,
        device=head._device,
        dtype=head.dtype,
    )
    full_noisy = clean_latents.clone()
    full_noisy[:, video_start:video_end] = noisy_video
    video_timestep = torch.zeros(
        B, FULL_WINDOW_LATENTS, device=head._device
    )
    action_timestep = torch.zeros(
        B, full_action.shape[1], device=head._device
    )
    embodiment_id = action_input.embodiment_id.to(head._device)
    steps = num_inference_steps or head.num_inference_steps
    head.scheduler.set_timesteps(
        steps, denoising_strength=head.denoising_strength
    )
    timesteps = head.scheduler.timesteps.to(head._device)
    cfg_scale = head._wam_cfg_scale(guidance_scale)
    negative_prompt = (
        head._wam_encode_neg(full_input) if cfg_scale > 1.0 else None
    )
    if cfg_scale > 1.0 and negative_prompt is None:
        raise RuntimeError("AC-WM CFG requires a negative prompt")
    head.model._current_wam_task = "ac_wm"
    if head.moe_enabled:
        head._set_wam_expert_mask(
            batch_size=B,
            phase="infer",
            action_input=action_input,
            task_names=["ac_wm"] * B,
        )

    with torch.amp.autocast(
        dtype=torch.bfloat16,
        device_type=torch.device(head._device).type,
    ):
        for step_index, timestep in enumerate(timesteps):
            full_noisy[:, video_start:video_end] = noisy_video
            video_timestep.zero_()
            video_timestep[:, video_start:video_end] = float(timestep)

            def predict(context):
                video_pred, _action_pred = _full_window_forward(
                    head,
                    task="world_model",
                    noisy_latents=full_noisy,
                    clean_latents=clean_latents,
                    video_timestep=video_timestep,
                    action_timestep=action_timestep,
                    context=context,
                    clip_feature=clip_feature,
                    ys=ys,
                    action=full_action,
                    state=None,
                    embodiment_id=embodiment_id,
                )
                return video_pred.transpose(1, 2)[:, video_start:video_end]

            cond = predict(prompt_embs)
            if cfg_scale > 1.0:
                uncond = predict(negative_prompt)
                prediction = uncond + cfg_scale * (cond - uncond)
            else:
                prediction = cond
            noisy_video = head.scheduler.step(
                prediction, timestep, noisy_video
            )
            _emit_trace(
                head,
                task="ac_wm",
                backend="dense33",
                chunk_index=chunk_index,
                phase="video",
                step_index=step_index,
                prediction=prediction,
                sample=noisy_video,
            )

        decode_prefix = torch.cat(
            [clean_latents[:, :video_start], noisy_video], dim=1
        )
        decoded_prefix = head._acwm_decode_chunk(decode_prefix, reset=True)
        decoded_frames = RGB_FRAMES_PER_CHUNK - 1
        if decoded_prefix.shape[2] < decoded_frames:
            raise RuntimeError(
                "AC-WM dense prefix decoded fewer than eight future RGB frames"
            )
        decoded_chunk = decoded_prefix[:, :, -decoded_frames:]

    head._wam_acwm_unified_episode = {
        "chunk_index": chunk_index + 1,
        "history_images": history_images,
    }
    return BatchFeature(data={"video": noisy_video, "decoded_video": decoded_chunk})


def acwm_cached_episode_step(
    head,
    action_input: BatchFeature,
    *,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> BatchFeature:
    """Unified AC-WM rollout with a continuous bounded generated-history cache."""
    head.set_frozen_modules_to_eval_mode()
    if head.moe_routing_granularity != "token_group_level":
        raise RuntimeError("unified AC-WM cache requires token_group_level routing")
    episode = getattr(head, "_wam_acwm_reference_cache_episode", None)
    if episode is None:
        seed_input = dict(action_input)
        seed_input["images"] = seed_input["images"][:, :1]
        seed_input = BatchFeature(data=seed_input)
        seed, clip, _ys, prompt, _, _, _ = head._wam_encode_obs(
            seed_input, force_single_frame_first_latent=True
        )
        seed = seed[:, :1]
        batch_size = seed.shape[0]
        cfg_scale = head._wam_cfg_scale(guidance_scale)
        negative = head._wam_encode_neg(seed_input) if cfg_scale > 1.0 else None
        use_cfg = cfg_scale > 1.0 and negative is not None
        cache_prompt = torch.cat([prompt, negative], dim=0) if use_cfg else prompt
        cache_clip = (
            torch.cat([clip, clip], dim=0)
            if use_cfg and clip is not None
            else clip
        )
        cache_seed = torch.cat([seed, seed], dim=0) if use_cfg else seed
        window_tokens = ACWM_HISTORY_LATENTS * head.model.frame_seqlen
        embodiment_id = action_input.embodiment_id.to(head._device)
        if head.model.action_encoder.num_embodiments == 1:
            embodiment_id = torch.zeros_like(embodiment_id)
        cache_embodiment = (
            torch.cat([embodiment_id, embodiment_id], dim=0)
            if use_cfg
            else embodiment_id
        )
        head.model._current_wam_task = "ac_wm"
        if head.moe_enabled:
            head._set_wam_expert_mask(
                batch_size=cache_seed.shape[0],
                phase="infer",
                action_input=action_input,
                task_names=["ac_wm"] * cache_seed.shape[0],
            )
        head.model.reset_idm_cache()
        with torch.amp.autocast(
            dtype=torch.bfloat16,
            device_type=torch.device(head._device).type,
        ):
            head.model.masked_cache_write_clean_video(
                cache_seed.transpose(1, 2),
                task="world_model",
                abs_frame_start=0,
                context=cache_prompt,
                clip_feature=cache_clip,
                cache_chunk_id=-1,
                rolling_window_tokens=window_tokens,
            )
            head.model.masked_cache_write_seed(
                cache_seed.transpose(1, 2),
                task="world_model",
                context=cache_prompt,
                clip_feature=cache_clip,
            )
            seed_frames = _acwm_decoded_to_uint8(
                head._acwm_decode_chunk(seed, reset=True)
            )
        episode = {
            "chunk_index": 0,
            "batch_size": batch_size,
            "prompt": cache_prompt,
            "clip": cache_clip,
            "embodiment_id": cache_embodiment,
            "use_cfg": use_cfg,
            "cfg_scale": cfg_scale if use_cfg else 1.0,
            "latent_shape": tuple(seed.shape[2:]),
            "window_tokens": window_tokens,
            "clip_window_index": 0,
            "gt_seed_latent": seed.detach().clone(),
            "history_entries": [
                {
                    "latents": seed.detach().clone(),
                    "abs_frame_start": 0,
                    "cache_chunk_id": -1,
                }
            ],
            "pixel_history": [seed_frames],
        }
        head._wam_acwm_reference_cache_episode = episode

    chunk_index = int(episode["chunk_index"])
    batch_size = int(episode["batch_size"])
    # clip_gt re-anchor: the window-boundary seed request carries the real
    # observation; use it only to refresh the global appearance condition,
    # leaving the generated-history carrier untouched (no content jump).
    reanchor_mode = os.environ.get(
        "WAM_ACWM_WINDOW_REANCHOR", "clip_gt"
    ).strip().lower()
    if (
        reanchor_mode == "clip_gt"
        and chunk_index > 0
        and chunk_index % CHUNKS_PER_WINDOW == 0
    ):
        gt_frame = action_input["images"]
        if gt_frame.shape[1] != 1:
            raise ValueError(
                "clip_gt window re-anchor expects the single GT seed frame, "
                f"got {gt_frame.shape[1]} frames"
            )
        # Route through the shared obs encoder (prepare_wam_vae_video ->
        # uint8/[-1,1] normalization + resize) instead of calling
        # encode_image directly on raw uint8 frames.
        seed_input = dict(action_input)
        seed_input["images"] = seed_input["images"][:, :1]
        seed_input = BatchFeature(data=seed_input)
        with torch.amp.autocast(
            dtype=torch.bfloat16,
            device_type=torch.device(head._device).type,
        ):
            (
                _latents, _clip, _ys, _prompt,
                _num_frames, _height, _width,
            ) = head._wam_encode_obs(seed_input)
        episode["clip"] = (
            torch.cat([_clip, _clip], dim=0)
            if episode["use_cfg"]
            else _clip
        )
    action = action_input.action.to(dtype=head.dtype, device=head._device)
    if action.shape[1] != ACTIONS_PER_CHUNK:
        raise ValueError("AC-WM cache requires exactly 24 actions")
    requested_cfg = head._wam_cfg_scale(guidance_scale)
    if (requested_cfg > 1.0) != bool(episode["use_cfg"]):
        raise RuntimeError("AC-WM cache CFG mode changed without reset")

    def cache_batch(tensor):
        if tensor is None or not episode["use_cfg"]:
            return tensor
        return torch.cat([tensor, tensor], dim=0)

    channels, height, width = episode["latent_shape"]
    noisy = head.generate_noise(
        (batch_size, LATENTS_PER_CHUNK, channels, height, width),
        seed=int(head.seed) + chunk_index,
        device=head._device,
        dtype=head.dtype,
    )
    steps = num_inference_steps or head.num_inference_steps
    head.scheduler.set_timesteps(steps, denoising_strength=head.denoising_strength)
    timesteps = head.scheduler.timesteps.to(head._device)
    frame_start = 1 + chunk_index * LATENTS_PER_CHUNK
    # Training exposes action RoPE only within one four-chunk window
    # (positions 0..95). Video RoPE remains episode-absolute, while action
    # RoPE restarts at each training-window boundary.
    action_start = (chunk_index % CHUNKS_PER_WINDOW) * ACTIONS_PER_CHUNK
    with torch.amp.autocast(
        dtype=torch.bfloat16,
        device_type=torch.device(head._device).type,
    ):
        for step_index, timestep in enumerate(timesteps):
            prediction_all = head.model.acwm_cached_video_action_chunk(
                cache_batch(noisy).transpose(1, 2),
                cache_batch(action),
                abs_frame_start=frame_start,
                timestep_val=float(timestep),
                context=episode["prompt"],
                clip_feature=episode["clip"],
                embodiment_id=episode["embodiment_id"],
                action_start=action_start,
                unified_cache=True,
            ).transpose(1, 2)
            if episode["use_cfg"]:
                cond = prediction_all[:batch_size]
                uncond = prediction_all[batch_size:]
                prediction = uncond + episode["cfg_scale"] * (cond - uncond)
            else:
                prediction = prediction_all
            noisy = head.scheduler.step(prediction, timestep, noisy)
            _emit_trace(
                head,
                task="ac_wm",
                backend="cache",
                chunk_index=chunk_index,
                phase="video",
                step_index=step_index,
                prediction=prediction,
                sample=noisy,
            )
        head.model.masked_cache_write_clean_video(
            cache_batch(noisy).transpose(1, 2),
            task="world_model",
            abs_frame_start=frame_start,
            context=episode["prompt"],
            clip_feature=episode["clip"],
            cache_chunk_id=chunk_index + 1,
            rolling_window_tokens=episode["window_tokens"],
        )
        decoded = head._acwm_decode_chunk(noisy, reset=False)
        if (
            os.environ.get("WAM_ACWM_HISTORY_PIXEL_REENCODE", "0").strip()
            == "1"
        ):
            pixel_frames = _acwm_decoded_to_uint8(decoded)
            window_frames = 1 + CHUNKS_PER_WINDOW * pixel_frames.shape[1]
            buffered = torch.cat(
                episode["pixel_history"] + [pixel_frames], dim=1
            )
            episode["pixel_history"] = [buffered[:, -window_frames:]]
        episode["history_entries"].append(
            {
                "latents": noisy.detach().clone(),
                "abs_frame_start": frame_start,
                "cache_chunk_id": chunk_index + 1,
            }
        )
        # Keep the Python replay list identical to the model token-level
        # rolling window.  A two-latent chunk can straddle the cutoff (e.g.
        # 9 accumulated latents -> keep the newest 7), so dropping whole
        # entries would silently leave only six latents at a window boundary.
        history_entries = episode["history_entries"]
        total_latents = sum(entry["latents"].shape[1] for entry in history_entries)
        excess_latents = total_latents - ACWM_HISTORY_LATENTS
        while excess_latents > 0 and history_entries:
            first = history_entries[0]
            width = int(first["latents"].shape[1])
            if excess_latents >= width:
                history_entries.pop(0)
                excess_latents -= width
                continue
            first["latents"] = first["latents"][:, excess_latents:].detach().clone()
            first["abs_frame_start"] = int(first["abs_frame_start"]) + excess_latents
            excess_latents = 0
    episode["chunk_index"] = chunk_index + 1
    if episode["chunk_index"] % CHUNKS_PER_WINDOW == 0:
        # WAM_ACWM_CLIP_REFRESH controls the global appearance condition at
        # training-window boundaries.  "none" (default) anchors CLIP to the
        # episode seed frame, which removes the autoregressive brightness
        # feedback loop on long continuous rollouts; "generated" refreshes
        # CLIP from the model-generated boundary frame.  Either way the KV
        # carrier is rebuilt for the new training window below.
        if os.environ.get("WAM_ACWM_CLIP_REFRESH", "none") == "generated":
            boundary = decoded[:, :, -1:].transpose(1, 2)
            clip, _ys, _seed = head.encode_image(
                boundary, 1, boundary.shape[-2], boundary.shape[-1]
            )
            episode["clip"] = (
                torch.cat([clip, clip], dim=0)
                if episode["use_cfg"]
                else clip
            )
        head.model.reset_idm_cache()
        history_entries = episode["history_entries"]
        if not history_entries:
            raise RuntimeError("AC-WM cache boundary has no retained history")
        if (
            os.environ.get("WAM_ACWM_HISTORY_PIXEL_REENCODE", "0").strip()
            == "1"
        ):
            # Full-history pixel re-anchor: re-encode the last decoded
            # training window (1 twin frame + 4x8 chunk frames) exactly like
            # a training window and let its newest 7 latents replace the
            # retained history.  Every window boundary thus projects the
            # generated history back onto the VAE latent manifold instead of
            # carrying drifted diffusion latents forward (the decode ->
            # uint8 -> re-encode loop every block-level world model uses).
            pixel_history = torch.cat(episode["pixel_history"], dim=1)
            chunk_frames = (pixel_history.shape[1] - 1) // CHUNKS_PER_WINDOW
            frames_per_latent = chunk_frames // LATENTS_PER_CHUNK
            expected_latents = (
                1
                + (CHUNKS_PER_WINDOW * chunk_frames) // frames_per_latent
            )
            if expected_latents != ACWM_HISTORY_LATENTS + 2:
                raise RuntimeError(
                    "pixel re-anchor buffer must hold one full training "
                    f"window (1 twin + {CHUNKS_PER_WINDOW} chunks), got "
                    f"{pixel_history.shape[1]} frames"
                )
            window_frames = 1 + CHUNKS_PER_WINDOW * chunk_frames
            window_clip = pixel_history[:, -window_frames:]
            reencode_input = BatchFeature(data={"images": window_clip})
            with torch.amp.autocast(
                dtype=torch.bfloat16,
                device_type=torch.device(head._device).type,
            ):
                (
                    re_latents, _clip, _ys, _prompt,
                    _num_frames, _height, _width,
                ) = head._wam_encode_obs(
                    reencode_input, encode_conditioning=False
                )
            if re_latents.shape[1] != expected_latents:
                raise RuntimeError(
                    "pixel re-anchor expected "
                    f"{expected_latents} window latents, got "
                    f"{re_latents.shape[1]}"
                )
            first = history_entries[0]
            episode["history_entries"] = [
                {
                    "latents": re_latents[
                        :, -ACWM_HISTORY_LATENTS:
                    ].detach().clone(),
                    "abs_frame_start": int(first["abs_frame_start"]),
                    "cache_chunk_id": int(first["cache_chunk_id"]),
                }
            ]
            history_entries = episode["history_entries"]
        permanent_anchor = os.environ.get(
            "WAM_ACWM_PERMANENT_ANCHOR", "0"
        ).strip() == "1"
        if permanent_anchor:
            # Permanent anchoring (Ctrl-World style): the original GT seed
            # latent always occupies window position 0 as the spatial
            # reference; self-generated history follows at window-relative
            # positions 1+.  RoPE stays within the trained 0..8 range.
            gt_seed = episode.get("gt_seed_latent")
            if gt_seed is None:
                raise RuntimeError(
                    "permanent anchor requires a saved GT seed latent"
                )
            head.model.masked_cache_write_clean_video(
                cache_batch(gt_seed).transpose(1, 2),
                task="world_model",
                abs_frame_start=0,
                context=episode["prompt"],
                clip_feature=episode["clip"],
                cache_chunk_id=-1,
                rolling_window_tokens=episode["window_tokens"],
            )
            head.model.masked_cache_write_seed(
                cache_batch(gt_seed[:, :1]).transpose(1, 2),
                task="world_model",
                context=episode["prompt"],
                clip_feature=episode["clip"],
                abs_frame_start=0,
                cache_chunk_id=-1,
            )
            for window_pos, entry in enumerate(history_entries, start=1):
                head.model.masked_cache_write_clean_video(
                    cache_batch(entry["latents"]).transpose(1, 2),
                    task="world_model",
                    abs_frame_start=window_pos,
                    context=episode["prompt"],
                    clip_feature=episode["clip"],
                    cache_chunk_id=entry["cache_chunk_id"],
                    rolling_window_tokens=episode["window_tokens"],
                )
        else:
            # Default: the oldest self-generated latent serves as both clean
            # history and the t=0 noisy-seed carrier twin (training layout).
            first = history_entries[0]
            head.model.masked_cache_write_clean_video(
                cache_batch(first["latents"]).transpose(1, 2),
                task="world_model",
                abs_frame_start=first["abs_frame_start"],
                context=episode["prompt"],
                clip_feature=episode["clip"],
                cache_chunk_id=first["cache_chunk_id"],
                rolling_window_tokens=episode["window_tokens"],
            )
            head.model.masked_cache_write_seed(
                cache_batch(first["latents"][:, :1]).transpose(1, 2),
                task="world_model",
                context=episode["prompt"],
                clip_feature=episode["clip"],
                abs_frame_start=first["abs_frame_start"],
                cache_chunk_id=first["cache_chunk_id"],
            )
            for entry in history_entries[1:]:
                head.model.masked_cache_write_clean_video(
                    cache_batch(entry["latents"]).transpose(1, 2),
                    task="world_model",
                    abs_frame_start=entry["abs_frame_start"],
                    context=episode["prompt"],
                    clip_feature=episode["clip"],
                    cache_chunk_id=entry["cache_chunk_id"],
                    rolling_window_tokens=episode["window_tokens"],
                )
        episode["clip_window_index"] += 1
    return BatchFeature(data={"video": noisy, "decoded_video": decoded})


def joint_dense_episode_step(
    head,
    action_input: BatchFeature,
    *,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> BatchFeature:
    """Joint inference through its exact 33-frame training teacher-forcing graph."""
    from ..modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler

    head.set_frozen_modules_to_eval_mode()
    images = action_input["images"]
    episode = getattr(head, "_wam_joint_dense_episode", None)
    if episode is None:
        if images.shape[1] != 1:
            raise ValueError("Joint dense window must start from one RGB frame")
        chunk_index = 0
        history = images.detach().clone()
    else:
        chunk_index = int(episode["chunk_index"])
        if images.shape[1] != RGB_FRAMES_PER_CHUNK:
            raise ValueError("Joint dense history commit requires nine RGB frames")
        history = torch.cat([episode["history_images"], images[:, 1:].detach()], dim=1)
    if chunk_index >= CHUNKS_PER_WINDOW:
        raise RuntimeError("Joint dense window is complete; reset before reuse")

    full_input = dict(action_input)
    full_input["images"] = _pad_window(history, FULL_WINDOW_RGB_FRAMES)
    full_input = BatchFeature(data=full_input)
    clean, clip, ys, prompt, _, _, _ = head._wam_encode_obs(full_input)
    history_span = (
        clean[:, :1]
        if chunk_index == 0
        else clean[
            :,
            1 + (chunk_index - 1) * LATENTS_PER_CHUNK:
            1 + chunk_index * LATENTS_PER_CHUNK,
        ]
    )
    _emit_trace(
        head,
        task="joint_dreamzero",
        backend="dense33",
        chunk_index=chunk_index,
        phase="history_vae",
        step_index=-1,
        sample=history_span,
    )
    batch_size = clean.shape[0]
    cfg_scale = head._wam_cfg_scale(guidance_scale)
    negative = head._wam_encode_neg(full_input) if cfg_scale > 1.0 else None
    use_cfg = cfg_scale > 1.0 and negative is not None
    embodiment_id = action_input.embodiment_id.to(head._device)
    state = action_input.state.to(dtype=torch.bfloat16, device=head._device)[:, -1:]
    full_state = torch.zeros(
        batch_size,
        CHUNKS_PER_WINDOW,
        state.shape[-1],
        dtype=state.dtype,
        device=state.device,
    )
    full_state[:, chunk_index:chunk_index + 1] = state
    noisy_video = head.generate_noise(
        (
            batch_size,
            clean.shape[2],
            LATENTS_PER_CHUNK,
            clean.shape[3],
            clean.shape[4],
        ),
        seed=head.seed,
        device=head._device,
        dtype=torch.bfloat16,
    ).transpose(1, 2)
    noisy_action = head.generate_noise(
        (batch_size, ACTIONS_PER_CHUNK, head.model.action_dim),
        seed=head.seed,
        device=head._device,
        dtype=torch.bfloat16,
    )
    full_action = torch.zeros(
        batch_size,
        CHUNKS_PER_WINDOW * ACTIONS_PER_CHUNK,
        head.model.action_dim,
        dtype=noisy_action.dtype,
        device=noisy_action.device,
    )
    video_timestep = torch.zeros(
        batch_size, FULL_WINDOW_LATENTS, device=head._device, dtype=torch.int64
    )
    action_timestep = torch.zeros(
        batch_size, full_action.shape[1], device=head._device, dtype=torch.int64
    )
    video_start = 1 + chunk_index * LATENTS_PER_CHUNK
    video_end = video_start + LATENTS_PER_CHUNK
    action_start = chunk_index * ACTIONS_PER_CHUNK
    action_end = action_start + ACTIONS_PER_CHUNK
    seq_len = FULL_WINDOW_LATENTS * head.model.frame_seqlen
    head.model._current_wam_task = "joint_dreamzero"
    if head.moe_enabled:
        head._set_wam_expert_mask(
            batch_size=batch_size,
            phase="infer",
            action_input=action_input,
            task_names=["joint_dreamzero"] * batch_size,
        )

    steps = num_inference_steps or head.num_inference_steps
    video_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=head.scheduler.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    action_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=head.scheduler.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    video_scheduler.set_timesteps(steps, device=head._device, shift=head.sigma_shift)
    action_scheduler.set_timesteps(steps, device=head._device, shift=head.sigma_shift)
    if getattr(head.config, "decouple_inference_noise", False):
        final_noise = head.config.video_inference_final_noise
        sigma_max = video_scheduler.sigmas[0].item()
        video_scheduler.sigmas = (
            video_scheduler.sigmas * (sigma_max - final_noise) / sigma_max
            + final_noise
        )
        video_scheduler.timesteps = (
            video_scheduler.sigmas[:-1] * 1000
        ).to(torch.int64)

    with torch.amp.autocast(
        dtype=torch.bfloat16,
        device_type=torch.device(head._device).type,
    ):
        for step_index, (video_t, action_t) in enumerate(
            zip(video_scheduler.timesteps, action_scheduler.timesteps)
        ):
            full_noisy = clean.clone()
            full_noisy[:, video_start:video_end] = noisy_video
            full_action.zero_()
            full_action[:, action_start:action_end] = noisy_action
            video_timestep.zero_()
            action_timestep.zero_()
            video_timestep[:, video_start:video_end] = video_t
            action_timestep[:, action_start:action_end] = action_t

            def predict(context):
                video_pred, action_pred = head.model._forward_train(
                    x=full_noisy.transpose(1, 2),
                    timestep=video_timestep,
                    timestep_action=action_timestep,
                    context=context,
                    seq_len=seq_len,
                    clean_x=clean.transpose(1, 2),
                    aug_t=torch.zeros(
                        batch_size,
                        seq_len,
                        device=head._device,
                        dtype=video_timestep.dtype,
                    ),
                    y=ys,
                    clip_feature=clip,
                    action=full_action,
                    state=full_state,
                    embodiment_id=embodiment_id,
                )
                return (
                    video_pred[:, :, video_start:video_end].transpose(1, 2),
                    action_pred[:, action_start:action_end],
                )

            video_cond, action_pred = predict(prompt)
            if use_cfg:
                video_uncond, _ = predict(negative)
                video_pred = video_uncond + cfg_scale * (
                    video_cond - video_uncond
                )
            else:
                video_pred = video_cond
            noisy_video = video_scheduler.step(
                model_output=video_pred,
                timestep=video_t,
                sample=noisy_video,
                step_index=step_index,
                return_dict=False,
            )[0]
            noisy_action = action_scheduler.step(
                model_output=action_pred,
                timestep=action_t,
                sample=noisy_action,
                step_index=step_index,
                return_dict=False,
            )[0]
            _emit_trace(
                head,
                task="joint_dreamzero",
                backend="dense33",
                chunk_index=chunk_index,
                phase="joint",
                step_index=step_index,
                video_prediction=video_pred,
                action_prediction=action_pred,
                video_sample=noisy_video,
                action_sample=noisy_action,
            )

    video_output = noisy_video.transpose(1, 2)
    if chunk_index == 0:
        video_output = torch.cat(
            [clean[:, :1].transpose(1, 2), video_output], dim=2
        )
    head._wam_joint_dense_episode = {
        "chunk_index": chunk_index + 1,
        "history_images": history,
    }
    return BatchFeature(
        data={"action_pred": noisy_action, "video_pred": video_output}
    )


def idm_dense_episode_step(
    head,
    action_input: BatchFeature,
    *,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> BatchFeature:
    """IDM inference through its 33-frame LingBot teacher-forcing graph."""
    head.set_frozen_modules_to_eval_mode()
    images = action_input["images"]
    episode = getattr(head, "_wam_idm_dense_episode", None)
    if episode is None:
        if images.shape[1] != 1:
            raise ValueError("IDM dense window must start from one RGB frame")
        chunk_index = 0
        history_images = images.detach().clone()
        history_actions = []
        history_states = []
    else:
        chunk_index = int(episode["chunk_index"])
        if images.shape[1] != RGB_FRAMES_PER_CHUNK:
            raise ValueError("IDM dense history commit requires nine RGB frames")
        history_images = torch.cat(
            [episode["history_images"], images[:, 1:].detach()], dim=1
        )
        history_actions = list(episode["history_actions"])
        history_actions.append(action_input.action.detach().clone())
        history_states = list(episode["history_states"])
    if chunk_index >= CHUNKS_PER_WINDOW:
        raise RuntimeError("IDM dense window is complete; reset before reuse")

    current_state = action_input.state.to(
        dtype=torch.bfloat16, device=head._device
    )[:, -1:]
    history_states.append(current_state.detach().clone())
    full_input = dict(action_input)
    full_input["images"] = _pad_window(
        history_images, FULL_WINDOW_RGB_FRAMES
    )
    full_input = BatchFeature(data=full_input)
    clean_video, clip, _ys, prompt, _, _, _ = head._wam_encode_obs(full_input)
    batch_size = clean_video.shape[0]
    full_clean_action = torch.zeros(
        batch_size,
        CHUNKS_PER_WINDOW * ACTIONS_PER_CHUNK,
        head.action_dim,
        dtype=head.dtype,
        device=head._device,
    )
    for index, actions in enumerate(history_actions):
        start = index * ACTIONS_PER_CHUNK
        full_clean_action[:, start:start + ACTIONS_PER_CHUNK] = actions.to(
            dtype=head.dtype, device=head._device
        )
    full_state = torch.zeros(
        batch_size,
        CHUNKS_PER_WINDOW,
        current_state.shape[-1],
        dtype=current_state.dtype,
        device=current_state.device,
    )
    for index, state in enumerate(history_states):
        full_state[:, index:index + 1] = state

    cfg_scale = head._wam_cfg_scale(guidance_scale)
    negative = head._wam_encode_neg(full_input) if cfg_scale > 1.0 else None
    use_cfg = cfg_scale > 1.0 and negative is not None
    context = torch.cat([prompt, negative], dim=0) if use_cfg else prompt
    context_clip = (
        torch.cat([clip, clip], dim=0)
        if use_cfg and clip is not None
        else clip
    )
    embodiment_id = action_input.embodiment_id.to(head._device)
    if head.model.state_encoder.layer1.num_categories == 1:
        embodiment_id = torch.zeros_like(embodiment_id)

    def cfg_batch(tensor):
        return torch.cat([tensor, tensor], dim=0) if use_cfg else tensor

    video_start = 1 + chunk_index * LATENTS_PER_CHUNK
    video_end = video_start + LATENTS_PER_CHUNK
    action_start = chunk_index * ACTIONS_PER_CHUNK
    action_end = action_start + ACTIONS_PER_CHUNK
    seq_len = FULL_WINDOW_LATENTS * head.model.frame_seqlen
    zero_video_t = torch.zeros(
        batch_size, FULL_WINDOW_LATENTS, device=head._device
    )
    zero_action_t = torch.zeros(
        batch_size,
        CHUNKS_PER_WINDOW * ACTIONS_PER_CHUNK,
        device=head._device,
    )
    head.model._current_wam_task = "idm_lingbot_va"
    if head.moe_enabled:
        head._set_wam_expert_mask(
            batch_size=(2 * batch_size if use_cfg else batch_size),
            phase="infer",
            action_input=action_input,
            task_names=["idm_lingbot_va"] * (
                2 * batch_size if use_cfg else batch_size
            ),
        )
    steps = num_inference_steps or head.num_inference_steps
    head.scheduler.set_timesteps(steps, denoising_strength=head.denoising_strength)
    timesteps = head.scheduler.timesteps.to(head._device)
    channels, height, width = clean_video.shape[2:]
    noisy_video = head._wam_fixed_inference_noise(
        (batch_size, LATENTS_PER_CHUNK, channels, height, width)
    )
    shadow_action = head._wam_fixed_inference_noise(
        (batch_size, ACTIONS_PER_CHUNK, head.action_dim)
    )

    def forward_idm(noisy_v, noisy_a, clean_v, timestep):
        full_noisy_video = clean_v.clone()
        full_noisy_video[:, video_start:video_end] = noisy_v
        full_noisy_action = full_clean_action.clone()
        full_noisy_action[:, action_start:action_end] = noisy_a
        video_t = zero_video_t.clone()
        action_t = zero_action_t.clone()
        video_t[:, video_start:video_end] = float(timestep)
        action_t[:, action_start:action_end] = float(timestep)
        video_pred, action_pred = head.model._forward_train_idm(
            noisy_x=cfg_batch(full_noisy_video).transpose(1, 2),
            clean_x=cfg_batch(clean_v).transpose(1, 2),
            timestep=cfg_batch(video_t),
            clean_timestep=cfg_batch(zero_video_t),
            timestep_action=cfg_batch(action_t),
            clean_timestep_action=cfg_batch(zero_action_t),
            context=context,
            seq_len=seq_len,
            clip_feature=context_clip,
            noisy_action=cfg_batch(full_noisy_action),
            clean_action=cfg_batch(full_clean_action),
            state=cfg_batch(full_state),
            embodiment_id=cfg_batch(embodiment_id),
        )
        return (
            video_pred[:, :, video_start:video_end].transpose(1, 2),
            action_pred[:, action_start:action_end],
        )

    with torch.amp.autocast(
        dtype=torch.bfloat16,
        device_type=torch.device(head._device).type,
    ):
        for step_index, timestep in enumerate(timesteps):
            video_pred_all, action_shadow_all = forward_idm(
                noisy_video, shadow_action, clean_video, timestep
            )
            video_cond = video_pred_all[:batch_size]
            if use_cfg:
                video_uncond = video_pred_all[batch_size:]
                video_pred = video_uncond + cfg_scale * (
                    video_cond - video_uncond
                )
            else:
                video_pred = video_cond
            action_shadow = action_shadow_all[:batch_size]
            noisy_video = head.scheduler.step(
                video_pred, timestep, noisy_video
            )
            shadow_action = head.scheduler.step(
                action_shadow, timestep, shadow_action
            )
            _emit_trace(
                head,
                task="idm_lingbot_va",
                backend="dense33",
                chunk_index=chunk_index,
                phase="video",
                step_index=step_index,
                prediction=video_pred,
                sample=noisy_video,
            )

        clean_for_action = clean_video.clone()
        clean_for_action[:, video_start:video_end] = noisy_video
        noisy_action = head._wam_fixed_inference_noise(
            (batch_size, ACTIONS_PER_CHUNK, head.action_dim)
        )
        shadow_video = head._wam_fixed_inference_noise(noisy_video.shape)
        head.scheduler.set_timesteps(
            steps, denoising_strength=head.denoising_strength
        )
        action_timesteps = head.scheduler.timesteps.to(head._device)
        for step_index, timestep in enumerate(action_timesteps):
            video_shadow_all, action_pred_all = forward_idm(
                shadow_video, noisy_action, clean_for_action, timestep
            )
            video_shadow = video_shadow_all[:batch_size]
            action_pred = action_pred_all[:batch_size]
            shadow_video = head.scheduler.step(
                video_shadow, timestep, shadow_video
            )
            noisy_action = head.scheduler.step(
                action_pred, timestep, noisy_action
            )
            _emit_trace(
                head,
                task="idm_lingbot_va",
                backend="dense33",
                chunk_index=chunk_index,
                phase="action",
                step_index=step_index,
                prediction=action_pred,
                sample=noisy_action,
            )

    head._wam_idm_dense_episode = {
        "chunk_index": chunk_index + 1,
        "history_images": history_images,
        "history_actions": history_actions,
        "history_states": history_states,
    }
    return BatchFeature(
        data={
            "action": noisy_action,
            "video": noisy_video,
            "condition_video": clean_video[:, video_start - 1:video_start],
        }
    )


def reset_unified_inference(head) -> None:
    head._wam_masked_episode = None
    head._wam_acwm_unified_episode = None
    head._wam_acwm_reference_cache_episode = None
    head._wam_joint_dense_episode = None
    head._wam_idm_dense_episode = None
    head.model._current_wam_token_group_routing = None
    head.model._current_wam_task = None
    head.model._current_wam_expert_mask = None
    if hasattr(head.model, "reset_idm_cache"):
        head.model.reset_idm_cache()
    if hasattr(head.vae, "reset_streaming_encode"):
        head.vae.reset_streaming_encode()
    if hasattr(head.vae, "reset_streaming_decode"):
        head.vae.reset_streaming_decode()
