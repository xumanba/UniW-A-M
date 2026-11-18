from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps
import hashlib
import logging
import random
import time
from typing import TypeAlias, cast
import os
import re
from pathlib import Path

from accelerate import load_checkpoint_and_dispatch

from einops import rearrange
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from safetensors.torch import load_file
import json
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)

WAN_HF_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN22_HF_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"


def hf_download(filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Download a file from the specified HuggingFace repo to HF cache."""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return path


def ensure_file(path: str | None, hf_filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Return a valid local path: use `path` if it exists, otherwise download from HuggingFace."""
    if path is not None and os.path.exists(path):
        return path
    return hf_download(hf_filename, repo_id)

from torch.distributions import Beta
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import v2
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.n1_5.action_head.base_action_head import ActionHead
from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
from groot.vla.model.dreamzero.modules.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler
from groot.vla.model.dreamzero.modules.wam_moe import (
    WAM_EXPERT_NAMES,
    build_wam_expert_mask,
    inject_wam_moe_into_wan_model,
    resolve_moe_routing_granularity,
)
from groot.vla.model.dreamzero.modules.wam_task_attn import (
    WAM_MASKED_TASKS,
    canonical_masked_task,
)
from groot.vla.model.dreamzero.modules.wam_idm_cache import (
    IDMEpisodeCache,
    IDM_ACTIONS_PER_CHUNK,
    IDM_DEFAULT_WINDOW_CHUNKS,
    IDM_LATENTS_PER_CHUNK,
)
from groot.vla.data.precomputed_latents import (
    FULL_WINDOW_UNIFIED_CONSUMER,
    FULL_WINDOW_UNIFIED_CONSUMER_CODE,
    FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER,
    FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER_CODE,
    PRECOMPUTED_CONSUMER_CONTRACT_FIELD,
    V4_SINGLE_HORIZON_CONSUMER,
    V4_SINGLE_HORIZON_CONSUMER_CODE,
    load_contract,
    validate_consumer_capability,
)
from .wam_unified_inference import (
    _emit_trace as emit_wam_inference_trace,
    acwm_cached_episode_step as run_acwm_cached_episode_step,
    acwm_episode_step as run_unified_acwm_episode_step,
    idm_dense_episode_step as run_idm_dense_episode_step,
    joint_dense_episode_step as run_joint_dense_episode_step,
    masked_episode_step as run_unified_masked_episode_step,
    reset_unified_inference,
)


KVCacheType: TypeAlias = torch.Tensor

FASTWAM_CHUNKS_PER_WINDOW = 4
FASTWAM_VIDEO_FRAMES_PER_CHUNK = 9
FASTWAM_ACTIONS_PER_CHUNK = 24
ACWM_HISTORY_LATENTS = 4
ACWM_TRAINING_HISTORY_LATENTS = 4
ACWM_FUTURE_LATENTS = 2
ACWM_TOTAL_LATENTS = ACWM_TRAINING_HISTORY_LATENTS + ACWM_FUTURE_LATENTS
ACWM_INITIAL_RGB_FRAMES = 17
ACWM_INITIAL_WARMUP_LATENTS = 1
ACWM_CHUNKS_PER_CLIP = 4
ACWM_TRAINING_RGB_FRAMES = 49
ACWM_WINDOW_FIRST_RGB_INDEX = 16
ACWM_CONTINUOUS_LATENTS = 13
FULL_WINDOW_RGB_FRAMES = 33
FULL_WINDOW_LATENTS = 9
IDM_CLEAN_CONDITION_NOISE_PROBABILITY = 0.5
IDM_CLEAN_CONDITION_NOISE_MIN_RATIO = 0.5
IDM_CLEAN_CONDITION_NOISE_MAX_RATIO = 1.0


def _abort_idm_episode_on_error(method):
    """Reset partially written IDM state before re-raising an inference error."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            try:
                self.wam_idm_episode_reset()
            except Exception:
                logger.exception("Failed to reset IDM episode after inference error")
            raise

    return wrapped


def _rollout_observation_for_request(obs, request):
    """Use the latest real frame as the next window's one-frame seed."""
    images = obs.get("images") if hasattr(obs, "get") else None
    if images is None or request.expected_frames != 1 or images.shape[1] == 1:
        return obs
    if images.shape[1] < 1:
        raise ValueError("rollout observation cannot provide an empty video seed")

    data = dict(obs)
    data["images"] = images[:, -1:]
    state = data.get("state")
    if torch.is_tensor(state) and state.ndim >= 2 and state.shape[1] > 1:
        data["state"] = state[:, -1:]
    return BatchFeature(data=data) if isinstance(obs, BatchFeature) else data


def prepare_wam_vae_video(
    videos: torch.Tensor,
    *,
    normalize_video,
    dtype: torch.dtype,
    target_height: int | None = None,
    target_width: int | None = None,
) -> torch.Tensor:
    """Apply the action-head's exact RGB -> BF16 VAE input preprocessing."""
    if videos.ndim != 5:
        raise ValueError(
            f"WAM video input must be rank-5, got {tuple(videos.shape)}"
        )
    if videos.shape[-1] == 3:
        videos = rearrange(videos, "b t h w c -> b c t h w")
    elif videos.shape[1] != 3:
        raise ValueError(
            "WAM video input must be [B,T,H,W,3] or [B,3,T,H,W], got "
            f"{tuple(videos.shape)}"
        )
    if videos.dtype == torch.uint8:
        videos = videos.float().div_(255.0)
        b, c, t, h, w = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        videos = normalize_video(videos)
        videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    videos = videos.to(dtype=dtype)
    if target_height is not None and target_width is not None:
        _, _, _, h, w = videos.shape
        if (h, w) != (target_height, target_width):
            b, c, t, _, _ = videos.shape
            videos = torch.nn.functional.interpolate(
                videos.reshape(b * t, c, h, w),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, c, t, target_height, target_width)
    return videos


def resolve_v4_consumer_contract(
    task_name: str,
    *,
    unified_experiment: bool,
    marker,
    video_frames: int,
) -> str:
    """Validate the v4 batch layout for all five WAM task paradigms."""
    task = canonical_masked_task(str(task_name))
    is_fast = task == "action_only_fast_wam"
    is_giga = task == "action_only_gigaworld_policy"
    is_world = task == "world_model"
    is_joint_or_idm = not is_fast and not is_giga and not is_world
    full_window = unified_experiment or not (is_fast or is_giga or is_world)
    if torch.is_tensor(marker):
        markers = {
            int(value) for value in marker.detach().cpu().reshape(-1).tolist()
        }
    elif isinstance(marker, (list, tuple)):
        markers = {str(value) for value in marker}
    elif marker is None:
        markers = set()
    else:
        markers = {str(marker)}
    full_marker_sets = [
        {FULL_WINDOW_UNIFIED_CONSUMER_CODE},
        {FULL_WINDOW_UNIFIED_CONSUMER},
    ]
    if is_joint_or_idm and not unified_experiment:
        if markers not in (
            set(),
            {FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER_CODE},
            {FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER},
        ):
            raise RuntimeError(
                "legacy Joint/IDM full-window cache requires the "
                f"{FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER!r} marker"
            )
        if video_frames != FASTWAM_CHUNKS_PER_WINDOW:
            raise ValueError(
                "legacy Joint/IDM v4 consumer requires four RGB seed frames, "
                f"got {video_frames}"
            )
        return FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER
    if full_window:
        valid_marker_sets = [set(), *full_marker_sets]
        if not unified_experiment:
            valid_marker_sets.append({V4_SINGLE_HORIZON_CONSUMER_CODE})
        if markers not in valid_marker_sets:
            raise RuntimeError(
                "invalid full-window v4 consumer marker: "
                f"{markers!r}"
            )
        if unified_experiment and markers not in full_marker_sets:
            raise RuntimeError(
                "unified full-window v4 consumer requires the "
                f"{FULL_WINDOW_UNIFIED_CONSUMER!r} marker"
            )
        if video_frames != FULL_WINDOW_RGB_FRAMES:
            raise ValueError(
                "full-window v4 consumer requires 33 RGB frames, "
                f"got {video_frames}"
            )
        return FULL_WINDOW_UNIFIED_CONSUMER
    if markers and markers != {V4_SINGLE_HORIZON_CONSUMER_CODE}:
        raise RuntimeError(
            "single-horizon v4 consumer cannot use the full-window marker: "
            f"{markers!r}"
        )
    if video_frames != FASTWAM_CHUNKS_PER_WINDOW:
        raise ValueError(
            "single-horizon v4 consumer requires four RGB seed frames, "
            f"got {video_frames}"
        )
    return V4_SINGLE_HORIZON_CONSUMER


def validate_precomputed_v4_fields(keys) -> bool:
    """Reject a partially collated v4 cache instead of silently encoding online."""
    expected = {
        "wam_precomputed_continuous_latents",
        "wam_precomputed_chunked_latents",
    }
    present = expected.intersection(set(keys))
    if present and present != expected:
        raise RuntimeError(
            "partial v4 precomputed latent batch: expected both "
            f"{sorted(expected)}, got {sorted(present)}"
        )
    return present == expected


def use_unified_joint_consumption():
    value = os.environ.get("WAM_UNIFIED_JOINT_CONSUMPTION", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            "WAM_UNIFIED_JOINT_CONSUMPTION must be 0 or 1, "
            f"got {value!r}"
        )
    return value == "1"


def force_token_group_experiment():
    value = os.environ.get("WAM_FORCE_TOKEN_GROUP_EXPERIMENT", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            "WAM_FORCE_TOKEN_GROUP_EXPERIMENT must be 0 or 1, "
            f"got {value!r}"
        )
    return value == "1"


def use_unified_chunkwise_attention():
    value = os.environ.get("WAM_UNIFIED_CHUNKWISE_ATTENTION", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            "WAM_UNIFIED_CHUNKWISE_ATTENTION must be 0 or 1, "
            f"got {value!r}"
        )
    return value == "1"


def build_full_window_coupled_timestep_ids(
    chunk_t,
    num_video_latents,
    num_actions,
):
    """Broadcast four independent chunk timesteps over one 9/96 full window."""
    if chunk_t.ndim != 2 or chunk_t.shape[1] != FASTWAM_CHUNKS_PER_WINDOW:
        raise ValueError(
            "chunk_t must have shape [B,4], got "
            f"{tuple(chunk_t.shape)}"
        )
    if num_video_latents != FULL_WINDOW_LATENTS:
        raise ValueError(
            f"Full-window video requires {FULL_WINDOW_LATENTS} latents, got {num_video_latents}"
        )
    expected_actions = FASTWAM_CHUNKS_PER_WINDOW * FASTWAM_ACTIONS_PER_CHUNK
    if num_actions != expected_actions:
        raise ValueError(
            f"Full-window action requires {expected_actions} steps, got {num_actions}"
        )
    video_t = torch.zeros(
        chunk_t.shape[0], num_video_latents,
        dtype=chunk_t.dtype, device=chunk_t.device,
    )
    for chunk in range(FASTWAM_CHUNKS_PER_WINDOW):
        start = 1 + chunk * IDM_LATENTS_PER_CHUNK
        video_t[:, start:start + IDM_LATENTS_PER_CHUNK] = chunk_t[:, chunk:chunk + 1]
    action_t = chunk_t.repeat_interleave(FASTWAM_ACTIONS_PER_CHUNK, dim=1)
    return video_t, action_t


def add_idm_clean_video_condition_noise(
    latents,
    scheduler,
    timestep_template,
    *,
    probability=IDM_CLEAN_CONDITION_NOISE_PROBABILITY,
    min_timestep_ratio=IDM_CLEAN_CONDITION_NOISE_MIN_RATIO,
    max_timestep_ratio=IDM_CLEAN_CONDITION_NOISE_MAX_RATIO,
):
    """Apply LingBot-VA's clean-video condition augmentation.

    LingBot samples one Bernoulli decision for the whole forward.  When it is
    active, each temporal latent receives an independently sampled high-noise
    timestep, while samples in the batch share the same temporal schedule.
    """
    if latents.ndim != 5:
        raise ValueError(f"IDM clean video latents must be [B,F,C,H,W], got {tuple(latents.shape)}")
    if timestep_template.shape != latents.shape[:2]:
        raise ValueError(
            "IDM clean-video timestep template must match [B,F]: "
            f"{tuple(timestep_template.shape)} vs {tuple(latents.shape[:2])}"
        )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0,1], got {probability}")
    if not 0.0 <= min_timestep_ratio < max_timestep_ratio <= 1.0:
        raise ValueError(
            "clean-condition timestep ratios must satisfy 0 <= min < max <= 1, got "
            f"{min_timestep_ratio}/{max_timestep_ratio}"
        )

    clean_timestep = torch.zeros_like(timestep_template)
    if probability == 0.0 or (probability < 1.0 and torch.rand(1).item() >= probability):
        return latents, clean_timestep

    num_frames = latents.shape[1]
    timestep_ratios = (
        torch.rand(num_frames) * (max_timestep_ratio - min_timestep_ratio)
        + min_timestep_ratio
    )
    timestep_ids = torch.clamp(
        (timestep_ratios * scheduler.num_train_timesteps).long(),
        min=0,
        max=scheduler.num_train_timesteps - 1,
    )
    frame_timesteps = scheduler.timesteps[timestep_ids].to(
        device=latents.device,
        dtype=timestep_template.dtype,
    )
    clean_timestep = frame_timesteps.unsqueeze(0).expand(latents.shape[0], -1).clone()
    condition_noise = torch.randn_like(latents)
    noisy_condition = scheduler.add_noise(
        latents.flatten(0, 1),
        condition_noise.flatten(0, 1),
        clean_timestep.flatten(0, 1),
    ).unflatten(0, (latents.shape[0], num_frames))
    return noisy_condition, clean_timestep


def build_action_only_coupled_timestep_ids(base_t, num_video_latents, num_actions):
    """Broadcast one action-only horizon timestep over Vfut and action targets."""
    if base_t.ndim != 2 or base_t.shape[1] != 1:
        raise ValueError(f"base_t must have shape [B,1], got {tuple(base_t.shape)}")
    if num_video_latents < 1 or num_actions < 1:
        raise ValueError(
            f"Action-only training needs positive video/action lengths, got "
            f"{num_video_latents}/{num_actions}"
        )
    return (
        base_t.expand(-1, num_video_latents).clone(),
        base_t.expand(-1, num_actions).clone(),
    )


# Public compatibility name retained for existing GigaWorld tests/tools.
build_gigaworld_coupled_timestep_ids = build_action_only_coupled_timestep_ids


def expand_single_horizon_training_chunks(
    videos, actions, states, action_mask, *, precomputed_seed_only=False
):
    """Turn one joint window into independent 9-RGB/24-action samples.

    The boundary RGB frame is intentionally shared by adjacent chunks, but the
    returned B*4 samples are encoded by the VAE independently by the caller.
    """
    n_chunks = states.shape[1]
    expected_video = n_chunks if precomputed_seed_only else 8 * n_chunks + 1
    expected_actions = FASTWAM_ACTIONS_PER_CHUNK * n_chunks
    if not 1 <= n_chunks <= FASTWAM_CHUNKS_PER_WINDOW:
        raise ValueError(f"FastWAM requires 1..4 complete chunks, got {n_chunks}")
    if videos.shape[1] != expected_video or actions.shape[1] != expected_actions:
        raise ValueError(
            "FastWAM joint-training modalities disagree on chunk count: "
            f"video={videos.shape[1]} (expected {expected_video}), "
            f"action={actions.shape[1]} (expected {expected_actions}), state={n_chunks}"
        )
    if action_mask.shape[:2] != actions.shape[:2]:
        raise ValueError(
            f"FastWAM action_mask must start with {tuple(actions.shape[:2])}, "
            f"got {tuple(action_mask.shape)}"
        )
    bsz = videos.shape[0]
    if precomputed_seed_only:
        video_chunks = videos.reshape(bsz * n_chunks, 1, *videos.shape[2:])
    else:
        video_chunks = torch.stack(
            [
                videos[:, 8 * chunk : 8 * chunk + FASTWAM_VIDEO_FRAMES_PER_CHUNK]
                for chunk in range(n_chunks)
            ],
            dim=1,
        ).reshape(
            bsz * n_chunks,
            FASTWAM_VIDEO_FRAMES_PER_CHUNK,
            *videos.shape[2:],
        )
    action_chunks = actions.reshape(bsz, n_chunks, FASTWAM_ACTIONS_PER_CHUNK, *actions.shape[2:])
    action_chunks = action_chunks.reshape(bsz * n_chunks, FASTWAM_ACTIONS_PER_CHUNK, *actions.shape[2:])
    mask_chunks = action_mask.reshape(bsz, n_chunks, FASTWAM_ACTIONS_PER_CHUNK, *action_mask.shape[2:])
    mask_chunks = mask_chunks.reshape(bsz * n_chunks, FASTWAM_ACTIONS_PER_CHUNK, *action_mask.shape[2:])
    state_chunks = states.reshape(bsz * n_chunks, 1, *states.shape[2:])
    return video_chunks, action_chunks, state_chunks, mask_chunks


# Public compatibility name used by the FastWAM tests and older debug helpers.
expand_fastwam_training_chunks = expand_single_horizon_training_chunks


def expand_acwm_training_latents(continuous_latents):
    """Build four [4 clean history | 2 future] samples from one 49-frame span."""
    if continuous_latents.ndim != 5:
        raise ValueError(
            "ac_wm continuous latents must be [B,C,F,H,W], got "
            f"{tuple(continuous_latents.shape)}"
        )
    expected = ACWM_CONTINUOUS_LATENTS
    if continuous_latents.shape[2] != expected:
        raise ValueError(
            f"ac_wm needs {expected} continuous latents, got "
            f"{continuous_latents.shape[2]}"
        )
    chunks = []
    for chunk in range(FASTWAM_CHUNKS_PER_WINDOW):
        history_start = 1 + chunk * ACWM_FUTURE_LATENTS
        target_start = history_start + ACWM_TRAINING_HISTORY_LATENTS
        history = continuous_latents[:, :, history_start:target_start]
        future = continuous_latents[
            :, :, target_start:target_start + ACWM_FUTURE_LATENTS
        ]
        chunks.append(torch.cat([history, future], dim=2))
    stacked = torch.stack(chunks, dim=1)
    return stacked.reshape(
        continuous_latents.shape[0] * FASTWAM_CHUNKS_PER_WINDOW,
        continuous_latents.shape[1],
        ACWM_TOTAL_LATENTS,
        continuous_latents.shape[3],
        continuous_latents.shape[4],
    )


def apply_acwm_clean_conditions(
    clean_latents,
    noisy_latents,
    video_timestep,
    actions,
):
    """Install exact clean AC-WM history/action conditions at numeric timestep 0."""
    if clean_latents.shape != noisy_latents.shape:
        raise ValueError(
            "ac_wm clean/noisy latent shapes disagree: "
            f"{tuple(clean_latents.shape)} != {tuple(noisy_latents.shape)}"
        )
    if clean_latents.shape[1] != ACWM_TOTAL_LATENTS:
        raise ValueError(
            f"ac_wm requires {ACWM_TOTAL_LATENTS} latents, got "
            f"{clean_latents.shape[1]}"
        )
    if video_timestep.shape != clean_latents.shape[:2]:
        raise ValueError(
            "ac_wm video timestep shape must match [B,F]: "
            f"{tuple(video_timestep.shape)} != {tuple(clean_latents.shape[:2])}"
        )
    conditioned = noisy_latents.clone()
    conditioned[:, :ACWM_TRAINING_HISTORY_LATENTS] = clean_latents[
        :, :ACWM_TRAINING_HISTORY_LATENTS
    ]
    conditioned_timestep = video_timestep.clone()
    conditioned_timestep[:, :ACWM_TRAINING_HISTORY_LATENTS] = 0
    action_timestep = torch.zeros(
        actions.shape[:2],
        dtype=video_timestep.dtype,
        device=actions.device,
    )
    return conditioned, conditioned_timestep, actions, action_timestep

@dataclass
class WANPolicyHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )
    tiled: bool = field(default=True, metadata={"help": "Whether to use tiled input."})
    tile_size_height: int = field(default=34, metadata={"help": "Tile size height."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size width."})
    tile_stride_height: int = field(default=18, metadata={"help": "Tile stride height."})
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride width."})
    num_frame_per_block: int = field(default=1, metadata={"help": "Number of frames per block."})
    # Target video (H, W) for Wan22 resize. When set, videos are resized to this before VAE so latent
    # spatial size matches. Use height/width divisible by 32 for WanVideoVAE38 (16x) so latent H,W are even.
    target_video_height: int | None = field(default=None, metadata={"help": "Target video height for resize (e.g. 160 for even latent with VAE38)."})
    target_video_width: int | None = field(default=None, metadata={"help": "Target video width for resize (e.g. 320)."})

    lora_rank: int = field(default=4, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=4, metadata={"help": "LoRA alpha."})
    lora_target_modules: str = field(default="q,k,v,o,ffn.0,ffn.2")
    init_lora_weights: str = field(default="kaiming", metadata={"help": "LoRA initialization method."})
    train_architecture: str= field(default="lora", metadata={"help": "Train architecture."})
    skip_component_loading: bool = field(default=False, metadata={"help": "Construct component architectures without loading T5/CLIP/VAE/DiT weights; requires a complete full-model checkpoint immediately afterward."})

    use_gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing."})
    qformer_cfg: dict = field(default=None, metadata={"help": "Qformer configuration."})
    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    action_only_action_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Action-loss multiplier for FastWAM and GigaWorld training."},
    )
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    # High noise emphasis for BASE (coupled) training - applies Beta distribution to BOTH video and action together
    use_high_noise_emphasis: bool = field(
        default=False, metadata={"help": "Use Beta distribution for noise sampling (biases BOTH video and action towards high noise levels together)."}
    )
    high_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for high noise emphasis. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher = more high noise bias."}
    )
    # Decoupled noise sampling config for training-inference alignment
    # When enabled: video uses Beta(alpha,beta) biased towards high noise, action uses independent uniform
    decouple_video_action_noise: bool = field(
        default=False, metadata={"help": "Decouple video/action noise: video uses Beta distribution (high noise bias), action uses independent uniform."}
    )
    video_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for video noise. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher alpha = more bias to high noise."}
    )
    video_noise_beta_beta: float = field(
        default=1.0, metadata={"help": "Beta beta for video noise. Keep at 1.0."}
    )
    # Decoupled inference config - allows video to stay noisy while action fully denoises
    decouple_inference_noise: bool = field(
        default=False, metadata={"help": "Use decoupled noise schedules during inference (video stays noisy, action fully denoises)."}
    )
    video_inference_final_noise: float = field(
        default=0.8, metadata={"help": "Final noise level for video during decoupled inference (0.0-1.0). E.g., 0.8 means video ends at 80% noise."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    defer_lora_injection: bool = field(default=False, metadata={"help": "Defer LoRA injection until after loading pretrained weights."})

    vl_self_attention_cfg: dict = field(default=None)
    text_encoder_cfg: dict = field(default=None)
    image_encoder_cfg: dict = field(default=None)
    vae_cfg: dict = field(default=None)
    # UniW(A)M MoE: replace each transformer block's dense FFN with a router + 5 experts
    # (shared, vgm, vla, wm, idm). See groot/vla/model/dreamzero/modules/wam_moe.py.
    # When None or {"enabled": false}, the model stays dense.
    moe: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WANPolicyHead(ActionHead):
    config_class = WANPolicyHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: WANPolicyHeadConfig,
    ):
        super().__init__()
        self.unified_joint_consumption = use_unified_joint_consumption()
        self.force_token_group_experiment = force_token_group_experiment()
        self.unified_chunkwise_attention = use_unified_chunkwise_attention()
        if self.unified_chunkwise_attention and not self.unified_joint_consumption:
            raise ValueError(
                "WAM_UNIFIED_CHUNKWISE_ATTENTION requires "
                "WAM_UNIFIED_JOINT_CONSUMPTION=1"
            )
        self._unified_joint_consumption_ddp_validated = False
        self.tiled = config.tiled
        self.tile_size_height = config.tile_size_height
        self.tile_size_width = config.tile_size_width
        self.tile_stride_height = config.tile_stride_height
        self.tile_stride_width = config.tile_stride_width
        self.num_frame_per_block = config.num_frame_per_block
        self.hidden_size = config.hidden_size
        self.num_frames = config.num_frames
        self.text_encoder = instantiate(config.text_encoder_cfg)
        self._prompt_embedding_cache = OrderedDict()
        self._prompt_embedding_cache_size = max(
            0, int(os.environ.get("WAM_TEXT_EMBED_CACHE_SIZE", "64"))
        )
        self._prompt_embedding_cache_dir = os.environ.get(
            "WAM_TEXT_EMBED_CACHE_DIR", ""
        ).strip()
        self._prompt_embedding_cache_namespace = None
        self._prompt_embedding_cache_logged = False
        self.image_encoder = instantiate(config.image_encoder_cfg)
        self.vae = instantiate(config.vae_cfg)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.model_names = ['text_encoder']

        self.num_inference_steps = 16 
        self.seed = 1140
        self.cfg_scale = 5.0
        self.denoising_strength = 1.0
        self.sigma_shift = 5.0
        self.kv_cache1: KVCacheType | None = None
        self.kv_cache_neg: KVCacheType | None = None
        self.crossattn_cache: KVCacheType | None = None
        self.crossattn_cache_neg: KVCacheType | None = None
        self._wam_acwm_episode = None
        self._wam_masked_episode = None
        self._wam_acwm_unified_episode = None

        self.global_step = 0
        self.max_steps = 0
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.init_lora_weights = config.init_lora_weights
        self.train_architecture = config.train_architecture
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self.language = None

        self.ip_rank = 0
        self.ip_size = 1
        self.ip_group = None
        
        self._device = "cuda"
        self.dynamic_cache_schedule = os.getenv("DYNAMIC_CACHE_SCHEDULE", "False").lower() == "true"


        # Use every one of the 16 scheduler evaluations by default.  The
        # original DreamZero deployment reused cached velocity predictions on
        # half of the solver steps (8 DiT evaluations), which is a deliberate
        # latency/quality approximation.  Full 16-step evaluation is our
        # canonical open-loop and real-robot setting; NUM_DIT_STEPS remains an
        # explicit deployment override for reproducing the cached variant.
        num_dit_steps = 16
        if os.getenv("NUM_DIT_STEPS") is not None:
            num_dit_steps = int(os.getenv("NUM_DIT_STEPS"))
        if num_dit_steps == 5:
            self.dit_step_mask = [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False]
        elif num_dit_steps == 6:
            self.dit_step_mask = [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 7:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 8:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
        else:
            self.dit_step_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        assert self.dit_step_mask[0] == True, "first step must be True"

        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        if self.training:
            self.scheduler.set_timesteps(1000, training=True)
        
        
        self.input_embedding_dim = config.input_embedding_dim

        self.cpu_offload = False

        self.model = instantiate(config.diffusion_model_cfg)
        if not hasattr(self.model, "gradient_checkpointing"):
            raise AttributeError(
                "Configured diffusion model does not expose gradient_checkpointing"
            )
        checkpoint_ratio = float(os.environ.get("GRADIENT_CHECKPOINTING_RATIO", "1.0"))
        if not 0.0 <= checkpoint_ratio <= 1.0:
            raise ValueError(
                "GRADIENT_CHECKPOINTING_RATIO must be in [0,1], got "
                f"{checkpoint_ratio}"
            )
        if not self.use_gradient_checkpointing:
            checkpoint_ratio = 0.0
        self.model.gradient_checkpointing = checkpoint_ratio > 0.0
        self.model.gradient_checkpointing_ratio = checkpoint_ratio
        num_checkpoint_blocks = int(round(len(self.model.blocks) * checkpoint_ratio))
        print(
            "DreamZero activation checkpointing: "
            f"{self.model.gradient_checkpointing}; ratio={checkpoint_ratio:.3f}; "
            f"blocks={num_checkpoint_blocks}/{len(self.model.blocks)}"
        )
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        
        if not config.skip_component_loading:
            text_enc_path = ensure_file(
                self.text_encoder.text_encoder_pretrained_path,
                "models_t5_umt5-xxl-enc-bf16.pth",
            )
            self.text_encoder.load_state_dict(torch.load(text_enc_path, map_location='cpu'))

            img_enc_path = ensure_file(
                self.image_encoder.image_encoder_pretrained_path,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            )
            self.image_encoder.model.load_state_dict(
                torch.load(img_enc_path, map_location='cpu'), strict=False
            )

            # Wan2.2 (WanVideoVAE38, z_dim=48) uses Wan2.2_VAE.pth; Wan2.1 uses Wan2.1_VAE.pth
            vae_hf_filename = "Wan2.2_VAE.pth" if getattr(self.vae, "z_dim", 16) == 48 else "Wan2.1_VAE.pth"
            vae_repo_id = WAN22_HF_REPO_ID if getattr(self.vae, "z_dim", 16) == 48 else WAN_HF_REPO_ID
            vae_path = ensure_file(
                self.vae.vae_pretrained_path,
                vae_hf_filename,
                repo_id=vae_repo_id,
            )
            self.vae.model.load_state_dict(torch.load(vae_path, map_location='cpu'))

            dit_dir = self.model.diffusion_model_pretrained_path
            # Wan2.2 (in_dim=48) uses Wan2.2-TI2V-5B repo; Wan2.1 uses Wan2.1-I2V-14B-480P
            dit_repo_id = WAN22_HF_REPO_ID if getattr(self.model, "in_dim", 16) == 48 else WAN_HF_REPO_ID
            if dit_dir is None or not os.path.isdir(dit_dir):
                index_path = hf_hub_download(repo_id=dit_repo_id, filename="diffusion_pytorch_model.safetensors.index.json")
                dit_dir = os.path.dirname(index_path)
                with open(index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in set(index["weight_map"].values()):
                    hf_hub_download(repo_id=dit_repo_id, filename=shard_file)

            if dit_dir is not None:
                safetensors_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors")
                safetensors_index_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors.index.json")
                # Alternative single-file name: some Wan2.2-5B DiT checkpoints are
                # saved as model.safetensors.
                alt_safetensors_path = os.path.join(dit_dir, "model.safetensors")
                state_dict = {}

                if os.path.exists(safetensors_index_path):
                    # Handle sharded safetensors
                    print(f"Loading sharded safetensors using index: {safetensors_index_path}")

                    with open(safetensors_index_path, 'r') as f:
                        index = json.load(f)

                    # Load each shard
                    for shard_file in set(index["weight_map"].values()):
                        shard_path = os.path.join(dit_dir, shard_file)
                        print(f"Loading shard: {shard_path}")
                        shard_state_dict = load_file(shard_path)
                        state_dict.update(shard_state_dict)

                elif os.path.exists(safetensors_path):
                    # Handle single safetensors file
                    print(f"Loading weights from safetensors: {safetensors_path}")
                    state_dict = load_file(safetensors_path)

                elif os.path.exists(alt_safetensors_path):
                    print(f"Loading weights from safetensors: {alt_safetensors_path}")
                    state_dict = load_file(alt_safetensors_path)

                else:
                    raise ValueError(
                        f"No safetensors file found at {safetensors_path}, "
                        f"{safetensors_index_path} or {alt_safetensors_path}"
                    )

                # Normalize wrapper prefixes: some checkpoints save the plain Wan DiT under a
                # uniform "dit." prefix (dit.blocks.0.ffn.0.weight -> blocks.0.ffn.0.weight).
                for _prefix in ("dit.", "model."):
                    if state_dict and all(k.startswith(_prefix) for k in state_dict):
                        print(f"Stripping uniform '{_prefix}' prefix from {len(state_dict)} checkpoint keys")
                        state_dict = {k[len(_prefix):]: v for k, v in state_dict.items()}
                        break

                # A pre-expanded MoE initialization (the released `pretrained/` weights)
                # stores every block's FFN as five experts plus a router. Those modules
                # only exist after MoE injection below, so keep them aside and load them
                # there; the dense FFN keys are expected to be missing in that case.
                moe_keys = [k for k in state_dict if ".ffn.control_moe." in k]
                self._pending_moe_state_dict = {k: state_dict.pop(k) for k in moe_keys}
                if moe_keys:
                    print(f"Found {len(moe_keys)} pre-expanded MoE tensors; loading them after MoE injection")

                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if moe_keys:
                    missing_keys = [k for k in missing_keys if not re.search(r"\.ffn\.[02]\.(weight|bias)$", k)]

                if missing_keys:
                    print(f"Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

                print("Successfully loaded pretrained weights")
        else:
            print("Skipping individual component loading (loading from full pretrained model)")

        # ---- UniW(A)M MoE injection ----
        # Normally injected after loading the dense FFN weights, so each expert starts
        # as a copy of the dense FFN. With skip_component_loading=True the experts are
        # initialized structurally here and must all be replaced by a full checkpoint.
        # Env vars WAM_MOE_* override the Hydra `moe` config for quick debugging.
        moe_cfg = getattr(config, "moe", None) or {}
        self.moe_enabled = bool(moe_cfg.get("enabled", False))
        self.moe_routing_mode = os.environ.get("WAM_MOE_ROUTING_MODE") or moe_cfg.get("routing_mode", "global_task")
        requested_granularity = os.environ.get("WAM_MOE_GRANULARITY")
        if self.force_token_group_experiment:
            self.moe_routing_granularity = "token_group_level"
        elif self.unified_joint_consumption and requested_granularity is None:
            self.moe_routing_granularity = "token_group_level"
        else:
            self.moe_routing_granularity = resolve_moe_routing_granularity(
                requested_granularity
                or moe_cfg.get("routing_granularity", "sample_level")
            )
        self.moe_task = os.environ.get("WAM_MOE_TASK") or moe_cfg.get("task", "joint_dreamzero")
        # Multi-task mixed training: the task-aware dataloader uses these ratios
        # to choose one homogeneous task before loading each micro-batch.
        self.wam_task_ratios = self._parse_wam_task_ratios(os.environ.get("WAM_TASK_RATIOS", ""))
        if self.moe_enabled:
            target_blocks = moe_cfg.get("target_blocks", "all")
            target_block_indices = None if target_blocks in ("all", None, "*") else list(target_blocks)
            inject_wam_moe_into_wan_model(
                self.model,
                target_block_indices=target_block_indices,
                route_temperature=float(moe_cfg.get("route_temperature", 1.0)),
                routing_granularity=self.moe_routing_granularity,
            )
            print(
                f"[UniW(A)M MoE] enabled | routing_granularity={self.moe_routing_granularity} | "
                f"routing_mode={self.moe_routing_mode} | task={self.moe_task} | "
                f"experts={','.join(WAM_EXPERT_NAMES)} | target_blocks={target_blocks}"
            )
        pending_moe = getattr(self, "_pending_moe_state_dict", None)
        if pending_moe:
            if not self.moe_enabled:
                raise ValueError(
                    "the DiT weights contain MoE experts, but MoE is disabled; "
                    "use the MoE action-head config (wan_flow_matching_action_tf_wan22_moe)"
                )
            expected = {k for k in self.model.state_dict() if ".ffn.control_moe." in k}
            absent = sorted(expected - set(pending_moe))
            extra = sorted(set(pending_moe) - expected)
            if absent or extra:
                raise ValueError(
                    f"pre-expanded MoE weights do not match the model: "
                    f"{len(absent)} missing (e.g. {absent[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})"
                )
            self.model.load_state_dict(pending_moe, strict=False)
            print(f"Loaded {len(pending_moe)} pre-expanded MoE tensors (experts and routers)")
        self._pending_moe_state_dict = None
        # ---- End UniW(A)M MoE injection ----

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        # Video noise Beta distribution (biased towards high noise levels when enabled)
        self.video_beta_dist = Beta(config.video_noise_beta_alpha, config.video_noise_beta_beta)
        # High noise emphasis Beta distribution for coupled training (applies to both video and action)
        self.high_noise_beta_dist = Beta(config.high_noise_beta_alpha, 1.0)
        # self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.action_only_action_loss_weight = float(
            getattr(config, "action_only_action_loss_weight", 1.0)
        )
        if self.action_only_action_loss_weight < 0:
            raise ValueError("action_only_action_loss_weight must be non-negative")
        self._noise_logged = False
        self.defer_lora_injection = config.defer_lora_injection
        print("defer_lora_injection@@", self.defer_lora_injection)
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        if self.train_architecture == "lora" and not self.defer_lora_injection:
            print("Adding LoRA to model")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
        elif self.train_architecture == "lora" and self.defer_lora_injection:
            print("Deferring LoRA injection until after pretrained weights are loaded")
        else:
            self.print_trainable_params()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if not self.defer_lora_injection:
            self.print_trainable_params()


    def print_trainable_params(self):
        """Print trainable parameters of the diffusion model."""
        trainable_params = []
        total_params = 0
        trainable_total = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(name)
                trainable_total += param.numel()
                
        print(f"Total parameters in diffusion model: {total_params:,}")
        print(f"Trainable parameters in diffusion model: {trainable_total:,}")
        # print(trainable_params)


    def inject_lora_after_loading(self):
        """
        Inject LoRA adapters after pretrained weights have been loaded.
        This should be called when defer_lora_injection=True.
        """
        if self.train_architecture == "lora":
            print("Injecting LoRA after loading pretrained weights")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
            # self.model.registers.requires_grad_(True)
            # self.model.time_modality_projection.requires_grad_(True)
            
            self.text_encoder.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.print_trainable_params()
        else:
            print("LoRA injection not needed (train_architecture != 'lora')")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_diffusion_model:
                self.model.eval()
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.dtype,
                computation_device='cuda',
            ),
        )

        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        print("offloadd")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                # print("offload", module)
                                module.offload()
                    else:
                        print("tocpu")
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    print("onload")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            # print("onload", module)
                            module.onload()
                else:
                    print("togpu")
                    model.to(self._device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seqlen: int,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU KV cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        kv_cache1: KVCacheType = []
        kv_cache_neg: KVCacheType = []
        for _ in range(self.model.num_layers):
            kv_cache1.append(
                torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
            )
            kv_cache_neg.append(
                torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
            )

        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        crossattn_cache: KVCacheType = []
        crossattn_cache_neg: KVCacheType = []

        for _ in range(self.model.num_layers):
            crossattn_cache.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )
            crossattn_cache_neg.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )

        return crossattn_cache, crossattn_cache_neg
        
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def preprocess_image(self, image):
        image = (image * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
        return image

    def _ensure_prompt_embedding_cache(self):
        """Initialize cache fields for normal heads and lightweight unit fakes."""
        attrs = self.__dict__
        attrs.setdefault("_prompt_embedding_cache", OrderedDict())
        attrs.setdefault(
            "_prompt_embedding_cache_size",
            max(0, int(os.environ.get("WAM_TEXT_EMBED_CACHE_SIZE", "64"))),
        )
        attrs.setdefault(
            "_prompt_embedding_cache_dir",
            os.environ.get("WAM_TEXT_EMBED_CACHE_DIR", "").strip(),
        )
        attrs.setdefault("_prompt_embedding_cache_namespace", None)
        attrs.setdefault("_prompt_embedding_cache_logged", False)

    def _prompt_cache_namespace(self):
        """Fingerprint the frozen text encoder without hashing multi-GB weights."""
        if self._prompt_embedding_cache_namespace is not None:
            return self._prompt_embedding_cache_namespace
        encoder = self.text_encoder
        while hasattr(encoder, "_orig_mod"):
            encoder = encoder._orig_mod
        pretrained_path = str(
            getattr(encoder, "text_encoder_pretrained_path", "") or ""
        )
        path_meta = pretrained_path
        if pretrained_path:
            try:
                stat = os.stat(pretrained_path)
                path_meta += f":{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                path_meta += ":missing"
        payload = (
            "wam_prompt_embedding_cache_v1|"
            f"{path_meta}|bf16|padding_zeroed"
        )
        self._prompt_embedding_cache_namespace = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:20]
        return self._prompt_embedding_cache_namespace

    def _prompt_cache_key(self, input_ids_row, attention_mask_row):
        ids = input_ids_row.detach().to(device="cpu", dtype=torch.int64).contiguous()
        mask = attention_mask_row.detach().to(
            device="cpu", dtype=torch.uint8
        ).contiguous()
        digest = hashlib.sha256()
        digest.update(self._prompt_cache_namespace().encode("ascii"))
        digest.update(str(tuple(ids.shape)).encode("ascii"))
        digest.update(ids.numpy().tobytes())
        digest.update(mask.numpy().tobytes())
        return digest.hexdigest()

    def _prompt_cache_path(self, key):
        if not self._prompt_embedding_cache_dir:
            return None
        root = os.path.join(
            self._prompt_embedding_cache_dir,
            self._prompt_cache_namespace(),
        )
        return os.path.join(root, f"{key}.pt")

    def _remember_prompt_embedding(self, key, value):
        if self._prompt_embedding_cache_size <= 0:
            return
        self._prompt_embedding_cache[key] = value.detach()
        self._prompt_embedding_cache.move_to_end(key)
        while len(self._prompt_embedding_cache) > self._prompt_embedding_cache_size:
            self._prompt_embedding_cache.popitem(last=False)

    def _load_prompt_embedding(self, key, device):
        value = self._prompt_embedding_cache.get(key)
        if value is not None:
            self._prompt_embedding_cache.move_to_end(key)
            return value.to(device=device, dtype=torch.bfloat16)
        path = self._prompt_cache_path(key)
        if path is None or not os.path.isfile(path):
            return None
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"invalid cached tensor: {type(value)}")
            value = value.to(device=device, dtype=torch.bfloat16)
            self._remember_prompt_embedding(key, value)
            return value
        except Exception as exc:
            logger.warning("Ignoring invalid prompt cache entry %s: %s", path, exc)
            return None

    def _save_prompt_embedding(self, key, value):
        path = self._prompt_cache_path(key)
        if path is None:
            return
        root = os.path.dirname(path)
        try:
            os.makedirs(root, exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}.{random.getrandbits(32):08x}"
            torch.save(value.detach().to(device="cpu", dtype=torch.bfloat16), tmp)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Could not persist prompt embedding %s: %s", path, exc)
            try:
                if "tmp" in locals() and os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def encode_prompt(self, input_ids, attention_mask):
        """Encode each distinct final formatted prompt once and reuse it."""
        self._ensure_prompt_embedding_cache()
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Prompt ids/mask must be matching rank-2 tensors, got "
                f"{tuple(input_ids.shape)} and {tuple(attention_mask.shape)}"
            )
        if not self._prompt_embedding_cache_logged:
            logger.info(
                "Frozen text embedding cache: memory_entries=%d disk=%s",
                self._prompt_embedding_cache_size,
                self._prompt_embedding_cache_dir or "disabled",
            )
            self._prompt_embedding_cache_logged = True

        # Transfer two tiny [B, 512] tensors once, rather than synchronizing
        # device-to-host separately for every row in an expanded B*4 batch.
        ids_for_key = input_ids.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        masks_for_key = attention_mask.detach().to(
            device="cpu", dtype=torch.uint8
        ).contiguous()
        keys = [
            self._prompt_cache_key(ids_for_key[i], masks_for_key[i])
            for i in range(input_ids.shape[0])
        ]
        rows = [None] * input_ids.shape[0]
        missing = {}
        for i, key in enumerate(keys):
            cached = self._load_prompt_embedding(key, input_ids.device)
            if cached is not None:
                rows[i] = cached
            else:
                missing.setdefault(key, i)

        if missing:
            unique_keys = list(missing)
            unique_indices = torch.tensor(
                [missing[key] for key in unique_keys],
                device=input_ids.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                encoded = self.text_encoder(
                    input_ids.index_select(0, unique_indices),
                    attention_mask.index_select(0, unique_indices),
                ).clone().to(dtype=torch.bfloat16)
            unique_masks = attention_mask.index_select(0, unique_indices)
            seq_lens = unique_masks.gt(0).sum(dim=1).long()
            computed = {}
            for j, key in enumerate(unique_keys):
                encoded[j, seq_lens[j]:] = 0
                value = encoded[j].detach()
                computed[key] = value
                self._remember_prompt_embedding(key, value)
                self._save_prompt_embedding(key, value)
            for i, key in enumerate(keys):
                if rows[i] is None:
                    rows[i] = computed[key]
        return torch.stack(rows, dim=0).to(
            device=input_ids.device, dtype=torch.bfloat16
        )

    def _ensure_vae_on_device(self, ref_tensor):
        """Lazily move the VAE to the correct device/dtype on first use."""
        if not getattr(self, '_vae_device_ready', False):
            self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
            self.vae.eval()
            self._vae_device_ready = True

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        self._ensure_vae_on_device(input_video)
        with torch.no_grad():
            latents = self.vae.encode(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def encode_image(
        self, image, num_frames, height, width, *, need_seed_latent=False
    ):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            batch_size = image.shape[0]
            clip_context = self.image_encoder.encode_image(image)
            image_input = image.transpose(1, 2)
            uses_vae_image_condition = bool(
                getattr(self.model, "concat_first_frame_latent", False)
            )
            if not uses_vae_image_condition and not need_seed_latent:
                return clip_context, None, None
            if not uses_vae_image_condition:
                self._ensure_vae_on_device(image_input)
                with torch.no_grad():
                    seed = self.vae.encode(image_input)
                return clip_context, None, seed[:, :, :1]
            image_zeros = torch.zeros(batch_size, 3, num_frames-1, height, width, dtype=torch.bfloat16, device=self._device)
            self._ensure_vae_on_device(image_input)
            with torch.no_grad():
                y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
            # Build mask to match VAE output shape (VAE may use different spatial downsampling, e.g. WanVideoVAE38 uses patch_size=2 -> height/16)
            # y shape is B * 16 * (1+(T-1)/4) * H_latent * W_latent
            num_t = y.shape[2]
            h_latent, w_latent = y.shape[3], y.shape[4]
            msk = torch.zeros(batch_size, 4, num_t, h_latent, w_latent, dtype=y.dtype, device=self._device)
            msk[:, :, 0:1, :, :] = 1
            new_image = y[:, :, 0:1]
            # concat: B * (4+16) * (1+(T-1)/4) * H_latent * W_latent
            y = torch.concat([msk, y], dim=1)
        return clip_context, y, new_image
    
    def prepare_extra_input(self, latents=None):
        return {}

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming") -> nn.Module:
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_config)
        for param in model.parameters():
            param.data = param.to(torch.float32)
        return model

    @staticmethod
    def _parse_wam_task_ratios(s):
        """Parse 'task:ratio,task:ratio,...' into {task: ratio} (or None)."""
        s = (s or "").strip()
        if not s:
            return None
        out = {}
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition(":")
            k = k.strip()
            if k:
                out[k] = float(v) if v.strip() else 1.0
        return out or None

    def _action_loss_weight_for_task(self, task_name):
        task = canonical_masked_task(task_name)
        if task in {"action_only_fast_wam", "action_only_gigaworld_policy"}:
            return self.action_only_action_loss_weight
        return 1.0

    def _warn_fastwam_retrain_boundary_once(self):
        if self.moe_routing_granularity != "token_group_level":
            return
        if getattr(self, "_fastwam_retrain_warning_logged", False):
            return
        logger.warning(
            "FastWAM token-group v1 is active (up to 4 independent 9-frame/24-action/1-state "
            "horizons). Checkpoints trained with the former 96-action/4-state sample-level "
            "mixture remain loadable but require FastWAM retraining and are not a valid "
            "quality baseline for this path."
        )
        self._fastwam_retrain_warning_logged = True

    def _warn_world_model_retrain_boundary_once(self):
        if self.moe_routing_granularity != "token_group_level":
            return
        if getattr(self, "_world_model_retrain_warning_logged", False):
            return
        logger.warning(
            "world_model token-group v1 is active (independent 9-frame/24-action "
            "horizons; F/Aclean are shared-only and Vfut routes over shared+WM). Older 33-frame/96-action "
            "sample-level WM checkpoints remain loadable but require WM retraining."
        )
        self._world_model_retrain_warning_logged = True

    def _warn_joint_retrain_boundary_once(self):
        if self.moe_routing_granularity != "token_group_level":
            return
        if getattr(self, "_joint_retrain_warning_logged", False):
            return
        logger.warning(
            "joint_dreamzero token-group v1 is active: clean history/state are shared-only, "
            "noisy video routes over shared+VGM, and noisy action routes over shared+IDM. "
            "Older sample-level shared+VGM+IDM Joint checkpoints remain loadable but "
            "require Joint retraining and are not a valid quality baseline for this path."
        )
        self._joint_retrain_warning_logged = True

    def _warn_gigaworld_retrain_boundary_once(self):
        if self.moe_routing_granularity != "token_group_level":
            return
        if getattr(self, "_gigaworld_retrain_warning_logged", False):
            return
        logger.warning(
            "GigaWorld token-group v1 is active (up to 4 independent 9-frame/24-action/1-state "
            "horizons; F/S0 are shared-only, A routes over shared+VLA, and Vfut routes over "
            "shared+WM). Older 33-frame/96-action/4-state sample-level GigaWorld checkpoints "
            "remain loadable but require GigaWorld retraining and are not a valid quality baseline."
        )
        self._gigaworld_retrain_warning_logged = True

    def _warn_idm_retrain_boundary_once(self):
        if self.moe_routing_granularity != "token_group_level":
            return
        if getattr(self, "_idm_retrain_warning_logged", False):
            return
        logger.warning(
            "idm_lingbot_va token-group v1 is active: clean seed/history/state are "
            "shared-only, noisy video routes over shared+VGM, and noisy action routes "
            "over shared+IDM. Older all-token shared+VGM+IDM checkpoints remain "
            "loadable but require IDM retraining."
        )
        self._idm_retrain_warning_logged = True

    def _warn_sample_target_router_retrain_boundary_once(self):
        if self.moe_routing_granularity != "sample_level":
            return
        if getattr(self, "_sample_target_router_retrain_warning_logged", False):
            return
        logger.warning(
            "MoE1 sample-level target-only routing is active. Router pooling now "
            "uses noisy generation targets only; FastWAM/Giga pool noisy action "
            "only and deploy without future-video targets, while Joint/IDM retain "
            "their task-specific paired-target routes. Existing sample-level "
            "checkpoints remain structurally loadable but require retraining for "
            "this routing and inference contract."
        )
        self._sample_target_router_retrain_warning_logged = True

    def _warn_idm_coupled_override_once(self):
        if getattr(self, "_idm_coupled_override_logged", False):
            return
        if self.config.decouple_video_action_noise or self.config.use_high_noise_emphasis:
            logger.warning(
                "IDM ignores decouple_video_action_noise/use_high_noise_emphasis: "
                "each Vi pair and its Ai 24-step group share one uniform timestep. "
                "Other WAM tasks retain their configured noise policy."
            )
        self._idm_coupled_override_logged = True

    def _warn_gigaworld_coupled_override_once(self):
        if getattr(self, "_gigaworld_coupled_override_logged", False):
            return
        if self.config.decouple_video_action_noise or self.config.use_high_noise_emphasis:
            logger.warning(
                "GigaWorld ignores decouple_video_action_noise/use_high_noise_emphasis and "
                "uses the original coupled flow-shift=5 timestep for Vfut and A. Other WAM "
                "tasks retain their configured noise policy."
            )
        self._gigaworld_coupled_override_logged = True

    def _warn_fastwam_coupled_override_once(self):
        if getattr(self, "_fastwam_coupled_override_logged", False):
            return
        if self.config.decouple_video_action_noise or self.config.use_high_noise_emphasis:
            logger.warning(
                "FastWAM ignores decouple_video_action_noise/use_high_noise_emphasis "
                "and uses one coupled flow-shift=5 timestep for Vfut and A so its "
                "action-derived sample route has one noise stage."
            )
        self._fastwam_coupled_override_logged = True

    def _validate_unified_consumption_ddp_once(self):
        if getattr(self, "_unified_joint_consumption_ddp_validated", False):
            return
        value = int(getattr(self, "unified_joint_consumption", False))
        force_token_group = int(
            getattr(self, "force_token_group_experiment", False)
        )
        chunkwise_attention = int(
            getattr(self, "unified_chunkwise_attention", False)
        )
        granularity = getattr(
            self, "moe_routing_granularity", "sample_level"
        )
        granularity_code = {
            "sample_level": 0,
            "token_group_level": 1,
        }.get(granularity)
        if granularity_code is None:
            raise ValueError(f"Unsupported MoE routing granularity: {granularity!r}")
        if dist.is_available() and dist.is_initialized():
            encoded = (
                value + 2 * force_token_group + 4 * chunkwise_attention
                + 8 * granularity_code
            )
            probe = torch.tensor([encoded, encoded], device=self._device, dtype=torch.int32)
            low = probe[:1].clone()
            high = probe[1:].clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN)
            dist.all_reduce(high, op=dist.ReduceOp.MAX)
            if int(low.item()) != int(high.item()):
                raise RuntimeError(
                    "WAM unified-consumption experiment flags differ across DDP ranks"
                )
        logger.info(
            "WAM unified Joint-style consumption: %s; forced token-group experiment: %s; "
            "chunk-wise attention: %s; routing granularity: %s",
            "enabled" if value else "disabled",
            "enabled" if force_token_group else "disabled",
            "enabled" if chunkwise_attention else "disabled",
            granularity,
        )
        self._unified_joint_consumption_ddp_validated = True

    def _resolve_wam_task_names(self, batch_size: int, action_input: BatchFeature | None):
        """Resolve the per-sample WAM task names for this forward.

        Multi-task training requires the task-aware data path to attach one
        homogeneous ``wam_task`` before frames are loaded. Single-task training
        and inference continue to fall back to ``self.moe_task``.
        """
        if action_input is not None:
            keys = set(action_input.keys()) if hasattr(action_input, "keys") else set()
            if "wam_task" in keys:
                task_field = action_input["wam_task"]
                if task_field:
                    names = [str(t) for t in task_field]
                    if len(names) == 1 and batch_size > 1:
                        names = names * batch_size
                    if len(names) != batch_size:
                        raise ValueError(
                            f"wam_task count {len(names)} != batch size {batch_size}"
                        )
                    if len(set(names)) != 1:
                        raise ValueError(
                            f"one micro-batch must have one wam_task, got {names}"
                        )
                    if not self.wam_task_ratios:
                        from ..wam_episode import canonical_episode_mode

                        if canonical_episode_mode(
                            names[0]
                        ) != canonical_episode_mode(self.moe_task):
                            raise ValueError(
                                "explicit single-task batch label disagrees with "
                                f"configured WAM task: {names[0]!r} != "
                                f"{self.moe_task!r}"
                            )
                    return names
        if self.training and self.wam_task_ratios:
            raise KeyError(
                "WAM_TASK_RATIOS requires task-aware batches with explicit wam_task"
            )
        return [self.moe_task] * batch_size

    def _set_wam_expert_mask(
        self,
        batch_size: int,
        phase: str,
        action_input: BatchFeature | None = None,
        task_names=None,
    ):
        """Set the active-expert mask AND the current task on the DiT before forward.

        - self.model._current_wam_expert_mask: [B, num_experts] float mask (MoE).
        - self.model._current_wam_task: the batch's task name (str) that the DiT
          uses to dispatch the per-task self-attention layout. Homogeneous batch,
          so a single task name is used.
        """
        # Some IDM/joint entrypoints call specialized DiT methods directly and
        # do not pass through CausalWanModel._forward_train. Clear any previous
        # FastWAM group spec at this universal task boundary; FastWAM's masked
        # forward installs its fresh ranges immediately before the block loop.
        self.model._current_wam_token_group_routing = None
        if not getattr(self, "moe_enabled", False):
            # Even without MoE, expose the task so the DiT can dispatch self-attn.
            task_names = task_names or self._resolve_wam_task_names(batch_size, action_input)
            self.model._current_wam_task = task_names[0] if task_names else None
            return
        self._warn_sample_target_router_retrain_boundary_once()
        wam_pass = os.environ.get("WAM_MOE_PASS", "default")
        task_names = task_names or self._resolve_wam_task_names(batch_size, action_input)
        if any(
            str(name).strip().lower().replace("-", "_").replace(" ", "_")
            in {"joint_dreamzero", "joint"}
            for name in task_names
        ):
            self._warn_joint_retrain_boundary_once()
        if any(
            str(name).strip().lower().replace("-", "_").replace(" ", "_")
            in {"idm_lingbot_va", "lingbot_va", "idm"}
            for name in task_names
        ):
            self._warn_idm_retrain_boundary_once()
        # Homogeneous batch: one task drives the self-attn dispatch.
        self.model._current_wam_task = task_names[0] if task_names else self.moe_task
        mask = build_wam_expert_mask(
            task_names, phase=phase, wam_pass=wam_pass, device=self._device, dtype=torch.float32
        )
        self.model._current_wam_expert_mask = mask

    def _capture_sample_routes(self, fn):
        """Run one DiT call and return its per-MoE-block sample routes."""
        if (
            not self.moe_enabled
            or self.moe_routing_granularity != "sample_level"
        ):
            return fn(), None
        self.model._wam_sample_route_capture = {}
        try:
            output = fn()
            routes = dict(self.model._wam_sample_route_capture)
        finally:
            self.model._wam_sample_route_capture = None
        if not routes:
            raise RuntimeError("Sample-level route capture produced no block routes")
        return output, routes

    def _with_sample_route_override(self, routes, fn):
        """Run one clean cache write with captured target-derived routes."""
        if (
            routes is None
            or not self.moe_enabled
            or self.moe_routing_granularity != "sample_level"
        ):
            return fn()
        previous = getattr(self.model, "_wam_sample_route_override", None)
        self.model._wam_sample_route_override = routes
        try:
            return fn()
        finally:
            self.model._wam_sample_route_override = previous

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()
        self._validate_unified_consumption_ddp_once()

        data = action_input 
        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id
        # print("embodiment_id", embodiment_id)
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask

        state_features = action_input.state

        actions = action_input.action
        # assert the values of action is in between -1 and 1
        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"
        videos = data["images"]
        input_batch_size = videos.shape[0]
        _raw_task = str(
            self._resolve_wam_task_names(input_batch_size, action_input)[0] or ""
        ).strip().lower().replace("-", "_").replace(" ", "_")
        wam_task_name = canonical_masked_task(_raw_task)
        is_fastwam = wam_task_name == "action_only_fast_wam"
        is_world_model = wam_task_name == "world_model"
        is_gigaworld = wam_task_name == "action_only_gigaworld_policy"
        unified_experiment = getattr(self, "unified_joint_consumption", False)
        unified_joint_consumption = unified_experiment and (
            is_world_model or is_fastwam or is_gigaworld
        )
        unified_chunkwise_attention = getattr(
            self, "unified_chunkwise_attention", False
        )
        if is_fastwam:
            self._warn_fastwam_retrain_boundary_once()
            self._warn_fastwam_coupled_override_once()
        if is_world_model:
            self._warn_world_model_retrain_boundary_once()
        if is_gigaworld:
            self._warn_gigaworld_retrain_boundary_once()
            self._warn_gigaworld_coupled_override_once()

        precomputed_root = os.environ.get(
            "WAM_PRECOMPUTED_LATENTS_ROOT", ""
        ).strip()
        precomputed_acwm_root = os.environ.get(
            "WAM_PRECOMPUTED_ACWM_LATENTS_ROOT", ""
        ).strip()
        has_precomputed_v4_latents = validate_precomputed_v4_fields(data.keys())
        has_precomputed_acwm_latents = (
            "wam_precomputed_acwm_continuous_latents" in data
        )
        has_precomputed_latents = (
            has_precomputed_v4_latents or has_precomputed_acwm_latents
        )
        precomputed_consumer = data.get(PRECOMPUTED_CONSUMER_CONTRACT_FIELD)
        v4_consumer = None
        if has_precomputed_v4_latents:
            v4_consumer = resolve_v4_consumer_contract(
                wam_task_name,
                unified_experiment=unified_experiment,
                marker=precomputed_consumer,
                video_frames=videos.shape[1],
            )
        if has_precomputed_v4_latents and precomputed_root:
            cached_contract = getattr(
                self, "_wam_precomputed_contract_cache", None
            )
            if cached_contract is None or cached_contract[0] != precomputed_root:
                contract = load_contract(Path(precomputed_root))
                self._wam_precomputed_contract_cache = (
                    precomputed_root,
                    contract,
                )
            else:
                contract = cached_contract[1]
            validate_consumer_capability(
                contract,
                v4_consumer,
                require_declared=(
                    unified_experiment
                    and v4_consumer == FULL_WINDOW_UNIFIED_CONSUMER
                ),
            )
        if unified_joint_consumption and precomputed_acwm_root:
            raise RuntimeError(
                "Unified full-window consumption does not accept AC-WM v5 latents"
            )
        if is_world_model and has_precomputed_v4_latents and not unified_joint_consumption:
            raise RuntimeError(
                "AC-WM requires its 49-frame online VAE path; the v4 "
                "33-frame latent tensors do not contain the two-chunk pre-roll"
            )
        if not is_world_model and has_precomputed_acwm_latents:
            raise RuntimeError(
                "AC-WM v5 latents cannot be consumed by a non-AC-WM task"
            )
        if (
            precomputed_root
            and not has_precomputed_v4_latents
            and not (is_world_model and not unified_joint_consumption)
        ):
            raise KeyError(
                "WAM_PRECOMPUTED_LATENTS_ROOT is set but the training batch has "
                "no strict precomputed latent tensors"
            )
        if has_precomputed_v4_latents and not precomputed_root:
            raise RuntimeError(
                "training batch contains precomputed latents but "
                "WAM_PRECOMPUTED_LATENTS_ROOT is not set"
            )
        if precomputed_acwm_root and not has_precomputed_acwm_latents:
            raise KeyError(
                "WAM_PRECOMPUTED_ACWM_LATENTS_ROOT is set but the batch has "
                "no strict v5 latent tensor"
            )
        if has_precomputed_acwm_latents and not precomputed_acwm_root:
            raise RuntimeError(
                "training batch contains AC-WM v5 latents but "
                "WAM_PRECOMPUTED_ACWM_LATENTS_ROOT is not set"
            )

        # Resolve the synchronized task before any expensive encoder work. For
        # Single-horizon masked tasks unfold the joint dataloader window before VAE/CLIP
        # encoding, so every one of the four horizons has an independent codec
        # path and contributes as one B*4 sample to the mean loss.
        text = data["text"]
        text_attention_mask = data["text_attention_mask"]
        if unified_joint_consumption:
            expected_actions = (
                FASTWAM_CHUNKS_PER_WINDOW * FASTWAM_ACTIONS_PER_CHUNK
            )
            if (
                videos.shape[1] != FULL_WINDOW_RGB_FRAMES
                or actions.shape[1] != expected_actions
                or state_features.shape[1] != FASTWAM_CHUNKS_PER_WINDOW
            ):
                raise ValueError(
                    "Unified Joint-style consumption requires 33 RGB / 96 action / 4 state; "
                    f"got {videos.shape[1]}/{actions.shape[1]}/{state_features.shape[1]}"
                )
        elif is_world_model:
            n_chunks = state_features.shape[1]
            if n_chunks != FASTWAM_CHUNKS_PER_WINDOW:
                raise ValueError(
                    f"ac_wm requires {FASTWAM_CHUNKS_PER_WINDOW} chunks, got "
                    f"{n_chunks}"
                )
            expected_video_frames = (
                1 if has_precomputed_acwm_latents else ACWM_TRAINING_RGB_FRAMES
            )
            if videos.shape[1] != expected_video_frames:
                raise ValueError(
                    f"ac_wm requires {expected_video_frames} RGB frames, got "
                    f"{videos.shape[1]}"
                )
            actions = actions.reshape(
                input_batch_size * n_chunks,
                FASTWAM_ACTIONS_PER_CHUNK,
                *actions.shape[2:],
            )
            action_mask = action_mask.reshape(
                input_batch_size * n_chunks,
                FASTWAM_ACTIONS_PER_CHUNK,
                *action_mask.shape[2:],
            )
            state_features = state_features.reshape(
                input_batch_size * n_chunks, 1, *state_features.shape[2:]
            )
            repeat = n_chunks
            embodiment_id = embodiment_id.repeat_interleave(repeat, dim=0)
            has_real_action = has_real_action.repeat_interleave(repeat, dim=0)
        elif is_fastwam or is_gigaworld:
            videos, actions, state_features, action_mask = expand_single_horizon_training_chunks(
                videos,
                actions,
                state_features,
                action_mask,
                precomputed_seed_only=has_precomputed_latents,
            )
            repeat = actions.shape[0] // input_batch_size
            embodiment_id = embodiment_id.repeat_interleave(repeat, dim=0)
            has_real_action = has_real_action.repeat_interleave(repeat, dim=0)
            text = text.repeat_interleave(repeat, dim=0)
            text_attention_mask = text_attention_mask.repeat_interleave(repeat, dim=0)

        # Keep training and inference on one action-head preprocessing route.
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        videos = prepare_wam_vae_video(
            videos,
            normalize_video=self.normalize_video,
            dtype=self.dtype,
            target_height=target_h,
            target_width=target_w,
        )
        assert videos.min() >= -1.0 and videos.max() <= 1.0, (
            "videos must be in [-1,1] range"
        )
        print("videos", videos.shape)
        
        # shape of B * max_length * dim
        prompt_embs = self.encode_prompt(text, text_attention_mask)
        if is_world_model and not unified_joint_consumption:
            prompt_embs = prompt_embs.repeat_interleave(
                FASTWAM_CHUNKS_PER_WINDOW, dim=0
            )

        if has_precomputed_latents:
            if is_world_model and not unified_joint_consumption:
                stored = data["wam_precomputed_acwm_continuous_latents"]
                expected = (
                    input_batch_size,
                    48,
                    ACWM_CONTINUOUS_LATENTS,
                    10,
                    20,
                )
                if tuple(stored.shape) != expected:
                    raise ValueError(
                        f"AC-WM v5 continuous latent shape "
                        f"{tuple(stored.shape)} != {expected}"
                    )
                if stored.dtype != torch.bfloat16:
                    raise TypeError(
                        f"AC-WM v5 latents must be BF16, got {stored.dtype}"
                    )
                latents = expand_acwm_training_latents(stored)
                conditioning_num_frames = 1
            elif unified_joint_consumption:
                stored = data["wam_precomputed_continuous_latents"]
                expected = (input_batch_size, 48, 9, 10, 20)
                if tuple(stored.shape) != expected:
                    raise ValueError(
                        "Unified full-window continuous latent shape "
                        f"{tuple(stored.shape)} != {expected}"
                    )
                latents = stored
                conditioning_num_frames = FULL_WINDOW_RGB_FRAMES
            elif is_fastwam or is_gigaworld:
                stored = data["wam_precomputed_chunked_latents"]
                expected = (
                    input_batch_size,
                    FASTWAM_CHUNKS_PER_WINDOW,
                    48,
                    3,
                    10,
                    20,
                )
                if tuple(stored.shape) != expected:
                    raise ValueError(
                        f"chunked precomputed latent shape {tuple(stored.shape)} "
                        f"!= {expected}"
                    )
                latents = stored.reshape(
                    input_batch_size * FASTWAM_CHUNKS_PER_WINDOW,
                    48,
                    3,
                    10,
                    20,
                )
                conditioning_num_frames = FASTWAM_VIDEO_FRAMES_PER_CHUNK
            else:
                stored = data["wam_precomputed_continuous_latents"]
                expected = (input_batch_size, 48, 9, 10, 20)
                if tuple(stored.shape) != expected:
                    raise ValueError(
                        f"continuous precomputed latent shape {tuple(stored.shape)} "
                        f"!= {expected}"
                    )
                latents = stored
                conditioning_num_frames = 33
            if latents.dtype != torch.bfloat16:
                raise TypeError(
                    f"precomputed latents must be BF16, got {latents.dtype}"
                )
            latents = latents.to(device=self._device, dtype=self.dtype)
        else:
            latents = self._encode_masked_video_latents(
                videos,
                force_single_frame_seed=(
                    (is_fastwam or is_gigaworld)
                    and not unified_joint_consumption
                ),
            )
            if is_world_model and not unified_joint_consumption:
                latents = expand_acwm_training_latents(latents)
                conditioning_num_frames = 1
            else:
                conditioning_num_frames = videos.shape[2]

        # print("latents shape", latents.shape, self.dtype)
        _, _, _, height, width = videos.shape
        if is_world_model and not unified_joint_consumption:
            image_index = (
                0 if has_precomputed_acwm_latents
                else ACWM_WINDOW_FIRST_RGB_INDEX
            )
            image = videos[:, :, image_index:image_index + 1].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)
        if has_precomputed_latents and getattr(
            self.model, "concat_first_frame_latent", False
        ):
            raise RuntimeError(
                "strict precomputed latents currently support the Wan2.2-5B "
                "CLIP-only image condition, not legacy VAE ys conditioning"
            )

        clip_feas, ys, _ = self.encode_image(
            image, conditioning_num_frames, height, width
        )
        if is_world_model and not unified_joint_consumption:
            clip_feas = clip_feas.repeat_interleave(
                FASTWAM_CHUNKS_PER_WINDOW, dim=0
            )
            if ys is not None:
                ys = ys.repeat_interleave(
                    FASTWAM_CHUNKS_PER_WINDOW, dim=0
                )

        latents = latents.to(self._device)
        clip_feas = clip_feas.to(self._device)
        if ys is not None:
            ys = ys.to(self._device)
        prompt_embs = prompt_embs.to(self._device)

        # WAM masked-task branching: world_model / fastwam / gigaworld use the DiT's
        # per-task masked self-attn path. For these, video frame 0 is kept clean
        # (t=0) as first-frame conditioning; world_model additionally treats the
        # action as a clean condition (only video is denoised).
        is_masked_task = wam_task_name in WAM_MASKED_TASKS
        is_idm = _raw_task in ("idm_lingbot_va", "lingbot_va", "idm")
        if is_idm:
            self._warn_idm_coupled_override_once()
        if is_idm and (
            self.num_frame_per_block != IDM_LATENTS_PER_CHUNK
            or self.model.num_action_per_block != IDM_ACTIONS_PER_CHUNK
            or self.model.num_state_per_block != 1
        ):
            raise ValueError(
                "IDM requires the canonical slot state=1, latent_video=2, action=24; "
                "use the UniW(A)M MoE action-head config."
            )

        # Loss
        noise = torch.randn_like(latents)

        # specific to autoregressive 
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)
        
        # ============ VIDEO TIMESTEP SAMPLING ============
        if unified_joint_consumption:
            chunk_t = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (noise.shape[0], FASTWAM_CHUNKS_PER_WINDOW),
            )
            (
                timestep_id,
                action_only_action_timestep_id,
            ) = build_full_window_coupled_timestep_ids(
                chunk_t,
                noise.shape[1],
                actions.shape[1],
            )
            noise_mode = "FULL_WINDOW_HISTORY_COUPLED"
        elif is_fastwam or is_gigaworld:
            # The action-derived sample route is reused over every token, so
            # keep video/action at one shared noise stage.  F is pinned to t=0
            # below.  This deliberately overrides the global decoupled/high-
            # noise options for both action-only tasks.
            video_t = torch.randint(
                0, self.scheduler.num_train_timesteps, (noise.shape[0], 1)
            )
            timestep_id, action_only_action_timestep_id = build_action_only_coupled_timestep_ids(
                video_t, noise.shape[1], actions.shape[1]
            )
            noise_mode = (
                "FASTWAM_SINGLE_HORIZON_COUPLED"
                if is_fastwam
                else "GIGAWORLD_SINGLE_HORIZON_COUPLED"
            )
        elif is_world_model:
            if self.config.decouple_video_action_noise:
                ratio = self.video_beta_dist.sample([noise.shape[0], 1])
                video_t = ((1.0 - ratio) * self.scheduler.num_train_timesteps).long()
                noise_mode = "SINGLE_HORIZON_VIDEO_BETA"
            elif self.config.use_high_noise_emphasis:
                ratio = self.high_noise_beta_dist.sample([noise.shape[0], 1])
                video_t = ((1.0 - ratio) * self.scheduler.num_train_timesteps).long()
                noise_mode = "SINGLE_HORIZON_VIDEO_HIGH_NOISE"
            else:
                video_t = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], 1))
                noise_mode = "SINGLE_HORIZON_VIDEO"
            timestep_id = torch.zeros(
                noise.shape[0], noise.shape[1], dtype=torch.long
            )
            timestep_id[:, ACWM_TRAINING_HISTORY_LATENTS:] = video_t.clamp(
                0, self.scheduler.num_train_timesteps - 1
            )
        elif is_idm:
            # LingBot deployment runs video and action as two serial phases over
            # the same scheduler sequence.  Training therefore samples one
            # uniform level per chunk and broadcasts it to Vi(2)+Ai(24), even
            # when another jointly-trained task requests decoupled noise.
            timestep_id = torch.randint(
                0, self.scheduler.num_train_timesteps,
                (noise.shape[0], noise.shape[1]),
            )
            noise_mode = "IDM_COUPLED_PER_CHUNK"
        elif self.config.decouple_video_action_noise:
            # Decoupled mode: sample video from Beta distribution biased towards HIGH noise
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            # High noise emphasis mode (coupled): BOTH video and action use Beta distribution
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            # Original: uniform sampling over full range
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))
            noise_mode = "STANDARD"
        
        timestep_id_block = None
        if not is_world_model:
            timestep_id_block = timestep_id[:, 1:].reshape(
                        timestep_id.shape[0], -1, self.num_frame_per_block)
            timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
        
        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            if is_world_model:
                expected_latents = (
                    FULL_WINDOW_LATENTS
                    if unified_joint_consumption
                    else ACWM_TOTAL_LATENTS
                )
                expected_actions = (
                    FASTWAM_CHUNKS_PER_WINDOW * FASTWAM_ACTIONS_PER_CHUNK
                    if unified_joint_consumption
                    else FASTWAM_ACTIONS_PER_CHUNK
                )
                if noise.shape[1] != expected_latents or actions.shape[1] != expected_actions:
                    raise ValueError(
                        "ac_wm latent/action layout mismatch: "
                        f"actions, got latents={noise.shape[1]}, "
                        f"actions={actions.shape[1]}"
                    )
            else:
                assert actions.shape[1] / (noise.shape[1]-1) == (self.model.num_action_per_block // self.num_frame_per_block), f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
                assert (noise.shape[1]-1) / state_features.shape[1] == (self.num_frame_per_block // self.model.num_state_per_block), f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            
            # ============ ACTION TIMESTEP SAMPLING ============
            if is_world_model:
                timestep_action_id = torch.zeros(
                    actions.shape[0], actions.shape[1], dtype=torch.long
                )
                action_mode = "CLEAN_COND"
            elif is_fastwam or is_gigaworld:
                timestep_action_id = action_only_action_timestep_id
                action_mode = (
                    "FULL_WINDOW_HISTORY_COUPLED"
                    if unified_joint_consumption
                    else (
                        "FASTWAM_COUPLED_PER_HORIZON"
                        if is_fastwam
                        else "GIGAWORLD_COUPLED_PER_HORIZON"
                    )
                )
            elif is_idm:
                timestep_action_id = timestep_id_block.repeat(
                    1, 1, actions.shape[1] // (noise.shape[1] - 1)
                ).reshape(actions.shape[0], -1)
                action_mode = "IDM_COUPLED_PER_CHUNK"
            elif self.config.decouple_video_action_noise:
                # Decoupled: sample action timestep independently with full range
                timestep_action_id = torch.randint(
                    0, 
                    self.scheduler.num_train_timesteps, 
                    (actions.shape[0], actions.shape[1])
                )
                action_mode = "INDEPENDENT"
            else:
                # Original coupled: action timestep derived from video timestep
                timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
                action_mode = "COUPLED"

            # Log noise mode once
            if not self._noise_logged:
                video_mean = timestep_id.float().mean().item()
                action_mean = timestep_action_id.float().mean().item()
                if noise_mode == "DECOUPLED":
                    print(f"[NOISE] Mode={noise_mode} | Video: Beta({self.config.video_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode} Uniform mean_t={action_mean:.0f}")
                elif noise_mode == "HIGH_NOISE_EMPHASIS":
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Beta({self.config.high_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode}")
                else:
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Uniform mean_t={video_mean:.0f} | Action: {action_mode}")
                self._noise_logged = True
        else:
            noise_action = None
            timestep_action_id = None
            
        if timestep_id_block is not None:
            timestep_id_block = timestep_id_block.reshape(
                timestep_id_block.shape[0], -1
            )
            timestep_id = torch.concat(
                [timestep_id[:, :1], timestep_id_block], dim=1
            )
        # WAM masked tasks and IDM: frame 0 is the episode seed condition.
        if is_world_model:
            timestep_id = timestep_id.clone()
            if unified_joint_consumption:
                timestep_id[:, :1] = 0
            else:
                timestep_id[:, :ACWM_TRAINING_HISTORY_LATENTS] = 0
        elif is_masked_task or is_idm:
            timestep_id = timestep_id.clone()
            timestep_id[:, 0] = 0
        _, num_frames, num_channels, height, width = noise.shape
        # DiT patch_embedding uses stride (1,2,2), so sequence length is num_frames * (H//2) * (W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        seq_len = num_frames * tokens_per_frame

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)
        # Install exact clean conditions after scheduler sampling. Index 0 in the
        # training scheduler is its maximum-noise endpoint, not numeric timestep 0.
        if is_world_model and not unified_joint_consumption:
            (
                noisy_latents,
                timestep,
                noisy_actions,
                timestep_action,
            ) = apply_acwm_clean_conditions(
                latents,
                noisy_latents,
                timestep,
                actions,
            )
            training_target_action = None
        elif is_world_model:
            noisy_latents = noisy_latents.clone()
            noisy_latents[:, 0] = latents[:, 0]
            timestep = timestep.clone()
            timestep[:, 0] = 0
            noisy_actions = actions
            timestep_action = torch.zeros_like(actions[..., 0])
            training_target_action = None
        elif is_masked_task or is_idm:
            noisy_latents = noisy_latents.clone()
            noisy_latents[:, 0] = latents[:, 0]
            if unified_joint_consumption:
                timestep = timestep.clone()
                timestep[:, 0] = 0
        
        if actions.numel() > 0 and not is_world_model:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)
        elif not is_world_model:
            timestep_action = None
            noisy_actions = None
            training_target_action = None

        # UniW(A)M MoE: pick active experts for this training run/task before the DiT forward.
        self._set_wam_expert_mask(
            batch_size=noisy_latents.shape[0],
            phase="train",
            action_input=action_input,
            task_names=[_raw_task] * noisy_latents.shape[0],
        )

        # Compute loss
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            if is_idm:
                # LingBot-VA clean-video condition augmentation: one batch-level
                # Bernoulli draw, then independent high-noise timesteps per latent.
                if not getattr(self, "_idm_clean_condition_augmentation_logged", False):
                    logger.info(
                        "IDM clean-video condition augmentation: p=%.2f, timestep ratio=[%.2f, %.2f); "
                        "clean action remains at t=0",
                        IDM_CLEAN_CONDITION_NOISE_PROBABILITY,
                        IDM_CLEAN_CONDITION_NOISE_MIN_RATIO,
                        IDM_CLEAN_CONDITION_NOISE_MAX_RATIO,
                    )
                    self._idm_clean_condition_augmentation_logged = True
                clean_latents, clean_timestep = add_idm_clean_video_condition_noise(
                    latents,
                    self.scheduler,
                    timestep,
                )
                clean_timestep_action = torch.zeros_like(timestep_action)
                video_noise_pred, action_noise_pred = self.model._forward_train_idm(
                    noisy_x=noisy_latents.transpose(1, 2), clean_x=clean_latents.transpose(1, 2),
                    timestep=timestep, clean_timestep=clean_timestep,
                    timestep_action=timestep_action, clean_timestep_action=clean_timestep_action,
                    context=prompt_embs, seq_len=seq_len, y=ys, clip_feature=clip_feas,
                    noisy_action=noisy_actions, clean_action=actions,
                    state=state_features, embodiment_id=embodiment_id,
                )
            elif actions.numel() > 0:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    action=noisy_actions, timestep_action=timestep_action, 
                    clean_x=latents.transpose(1, 2),
                    wam_full_window_history=unified_joint_consumption,
                    wam_full_window_chunked_attention=unified_chunkwise_attention,
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, timestep_action=timestep_action, 
                    clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    clean_x=latents.transpose(1, 2),
                    wam_full_window_history=unified_joint_consumption,
                    wam_full_window_chunked_attention=unified_chunkwise_attention,
                )

            # Per-sample dynamics loss
            # DiT patch_embedding uses stride (1,2,2), so output spatial size can be smaller than
            # latent when H or W is odd (e.g. latent 11x20 -> model output 10x20). Crop target to match.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction='none'
            ).mean(dim=(1,3,4))  # shape: [B, ...]

            weight_dynamics = dynamics_loss_per_sample * self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
            if is_world_model:
                weighted_dynamics_loss = weight_dynamics[
                    :, (1 if unified_joint_consumption else ACWM_TRAINING_HISTORY_LATENTS):
                ].mean()
            elif is_masked_task or is_idm:
                # video frame 0 is clean conditioning (t=0) -> not a denoise target.
                weighted_dynamics_loss = weight_dynamics[:, 1:].mean()
            else:
                weighted_dynamics_loss = weight_dynamics.mean()

            # world_model: action is a clean condition -> no action loss.
            if actions.numel() > 0 and not is_world_model:
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_noise_pred.float(), training_target_action.float(), reduction='none'
                ) * action_mask  # shape: [B, ...]
                # [B] -> [B, 1, 1] so the validity flag broadcasts over both
                # the action-horizon and action-dimension axes for any batch size.
                action_loss_per_sample = (
                    has_real_action[:, None, None].float() * action_loss_per_sample
                )
                weight_action = action_loss_per_sample.mean(dim=2) * self.scheduler.training_weight(
                    timestep_action.flatten(0, 1),
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
                weighted_action_loss = weight_action.mean()
                loss = weighted_dynamics_loss + (
                    self._action_loss_weight_for_task(wam_task_name)
                    * weighted_action_loss
                )
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss
            # loss = dynamics_loss_per_sample.mean()

        # Record log
        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
        }
        # Multi-task training: also expose the losses under task-tagged keys so the
        # trainer's moving-average logger emits per-task curves
        # (e.g. task_world_model_loss_avg, task_joint_dreamzero_dynamics_loss_avg).
        if self.training and getattr(self, "wam_task_ratios", None) and _raw_task:
            output_dict[f"task_{_raw_task}_loss"] = loss.detach()
            output_dict[f"task_{_raw_task}_dynamics_loss"] = weighted_dynamics_loss.detach()
            if actions.numel() > 0 and not is_world_model:
                output_dict[f"task_{_raw_task}_action_loss"] = weighted_action_loss.detach()

        return BatchFeature(data=output_dict)

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def _wam_fixed_inference_noise(self, shape):
        """Preserve the deterministic noise contract used by real-robot WAM."""
        return self.generate_noise(
            shape,
            seed=self.seed,
            device=self._device,
            dtype=self.dtype,
        )
    
    def _get_caches(
        self, kv_caches_input: list[KVCacheType],
    ) -> list[KVCacheType]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            assert len(kv_caches_input) == 2
            if self.ip_rank == 0:
                kv_caches = [kv_caches_input[0]]
            else:
                kv_caches = [kv_caches_input[1]]
        else:
            assert len(kv_caches_input) <= 2
            kv_caches = [kv_caches_input[0]]
            if self.cfg_scale != 1.0:
                kv_caches.append(kv_caches_input[1])
        return kv_caches

    def _prepare_text_inputs(self, data: BatchFeature) -> list[tuple[torch.Tensor, torch.Tensor]]:

        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            if self.ip_rank == 0:
                text_inputs = [(data["text"], data["text_attention_mask"])]
            else:
                text_inputs = [(data["text_negative"], data["text_attention_mask_negative"])]
        else:
            text_inputs = [(data["text"], data["text_attention_mask"])]
            if self.cfg_scale != 1.0:
                text_inputs.append((data["text_negative"], data["text_attention_mask_negative"]))
        return text_inputs


    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict[str, bool | int],
        capture_routes: bool = False,
        route_overrides=None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # UniW(A)M MoE: pick active experts for inference before the DiT forward.
        self._set_wam_expert_mask(batch_size=noisy_input.shape[0], phase="infer")

        predictions = []
        captured_routes = []
        for index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[index]
            crossattn_cache = crossattn_caches[index]
            route_override = (
                route_overrides[index]
                if route_overrides is not None and index < len(route_overrides)
                else None
            )
            # trt_engine is only assigned by the deployment compile path; plain
            # instantiate() never sets it, so read defensively.
            if not kv_cache_metadata["update_kv_cache"] and getattr(self, "trt_engine", None) is not None:
                obs_noise_pred, action_noise_pred = self.trt_engine(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    context=prompt_emb,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                )
            else:
                def model_forward():
                    return self.model(
                        noisy_input,
                        timestep,
                        action=action,
                        timestep_action=timestep_action,
                        state=state,
                        embodiment_id=embodiment_id,
                        context=prompt_emb,
                        seq_len=seq_len,
                        y=y,
                        clip_feature=clip_feature,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start_frame=kv_cache_metadata["start_frame"],
                    )

                model_call = model_forward
                if route_override is not None:
                    model_call = lambda: self._with_sample_route_override(
                        route_override, model_forward
                    )
                if capture_routes:
                    (
                        (
                            obs_noise_pred,
                            action_noise_pred,
                            updated_kv_caches,
                        ),
                        routes,
                    ) = self._capture_sample_routes(model_call)
                    captured_routes.append(routes)
                else:
                    (
                        obs_noise_pred,
                        action_noise_pred,
                        updated_kv_caches,
                    ) = model_call()
                if kv_cache_metadata["update_kv_cache"]:
                    for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                        kv_cache[block_index] = updated_kv_cache.clone()
            obs_noise_pred = obs_noise_pred.clone()
            if action_noise_pred is not None:
                action_noise_pred = action_noise_pred.clone()
            else:
                action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device) # dummy action noise prediction
            predictions.append((obs_noise_pred, action_noise_pred))
        self._last_joint_route_capture = (
            captured_routes if capture_routes else None
        )
        return self._exchange_predictions(predictions)

    def _exchange_predictions(
        self,
        predictions: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size == 1:
            return predictions

        assert len(predictions) == 1
        my_predictions = list(predictions[0])

        other_predictions = [torch.empty_like(pred) for pred in my_predictions]

        send_ops = [
            dist.P2POp(op=dist.isend, tensor=pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for pred in my_predictions
        ]
        recv_ops = [
            dist.P2POp(op=dist.irecv, tensor=other_pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for other_pred in other_predictions
        ]
        ops = send_ops + recv_ops

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        output_predictions: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in range(self.ip_size)]
        output_predictions[self.ip_rank] = tuple(my_predictions)
        output_predictions[(self.ip_rank + 1) % self.ip_size] = tuple(other_predictions)
        assert all(isinstance(pred, tuple) for pred in output_predictions)
        return cast(list[tuple[torch.Tensor, torch.Tensor]], output_predictions)
    
    def should_run_model(self, index, current_timestep, prev_predictions):

        if not self.dynamic_cache_schedule:
            return self.dit_step_mask[index]

        # Always run first 2 steps to establish history
        if len(prev_predictions) < 2:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        elif self.skip_countdown == 1:
            self.skip_countdown = 0 
            return True

        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        thresholds = [0.95, 0.93]
        countdowns = [4, 2]

        for threshold, countdown in zip(thresholds, countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def _encode_masked_video_latents(self, videos, force_single_frame_seed=False):
        """Encode a video and optionally replace F with the one-frame VAE path."""
        tile_size = (self.tile_size_height, self.tile_size_width)
        tile_stride = (self.tile_stride_height, self.tile_stride_width)
        latents = self.encode_video(
            videos, self.tiled, tile_size, tile_stride
        )
        if force_single_frame_seed and videos.shape[2] > 1:
            first_latent = self.encode_video(
                videos[:, :, :1], self.tiled, tile_size, tile_stride
            )
            latents = torch.cat(
                [first_latent[:, :, :1], latents[:, :, 1:]], dim=2
            )
        return latents

    def _wam_encode_obs(
        self,
        action_input: BatchFeature,
        force_single_frame_first_latent: bool = False,
        idm_stream_mode: str | None = None,
        encode_conditioning: bool = True,
    ):
        """Shared obs encoding for WAM per-task inference (mirrors training forward).

        Returns (latents_bf_c_h_w, clip_feas, ys, prompt_embs, num_frames, H, W)
        with latents in [B, F, C, H, W] layout.  ``encode_conditioning=False``
        is the IDM commit fast path: prompt/CLIP conditioning already belongs to
        the active episode, so those three return values are ``None``.
        """
        data = action_input
        videos = data["images"]
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        videos = prepare_wam_vae_video(
            videos,
            normalize_video=self.normalize_video,
            dtype=torch.bfloat16,
            target_height=target_h,
            target_width=target_w,
        )
        prompt_embs = None
        if encode_conditioning:
            prompt_embs = self.encode_prompt(
                data["text"], data["text_attention_mask"]
            ).to(self._device)
        if idm_stream_mode is None:
            latents = self._encode_masked_video_latents(
                videos, force_single_frame_seed=force_single_frame_first_latent
            )
        elif idm_stream_mode == "start":
            self._ensure_vae_on_device(videos)
            latents = self.vae.encode_stream(videos[:, :, :1], reset=True)
        elif idm_stream_mode == "commit":
            if videos.shape[2] != 9:
                raise ValueError(
                    "IDM streaming commit requires start frame + 8 new frames; "
                    f"got {videos.shape[2]} frames"
                )
            self._ensure_vae_on_device(videos)
            latents = self.vae.encode_stream(videos[:, :, 1:], reset=False)
        else:
            raise ValueError(f"Unsupported IDM stream mode: {idm_stream_mode!r}")
        _, _, num_frames, height, width = videos.shape
        clip_feas = None
        ys = None
        if encode_conditioning:
            image = videos[:, :, :1].transpose(1, 2)
            clip_feas, ys, _ = self.encode_image(
                image, num_frames, height, width
            )
        latents = latents.to(self._device).transpose(1, 2)  # [B, F, C, H, W]
        ys = None if ys is None else ys.to(self._device)
        clip_feas = None if clip_feas is None else clip_feas.to(self._device)
        return latents, clip_feas, ys, prompt_embs, latents.shape[1], latents.shape[-2], latents.shape[-1]

    def _wam_cfg_scale(self, guidance_scale):
        """Resolve video CFG: explicit arg > env WAM_CFG_SCALE > 5.0."""
        if guidance_scale is None:
            guidance_scale = float(os.environ.get("WAM_CFG_SCALE", "5.0"))
        return float(guidance_scale)

    def _wam_encode_neg(self, action_input):
        """Encode the negative prompt for CFG (uncond branch). Returns None if the
        batch has no negative text (then CFG silently falls back to cond-only)."""
        keys = set(action_input.keys()) if hasattr(action_input, "keys") else set()
        if "text_negative" in keys and "text_attention_mask_negative" in keys:
            return self.encode_prompt(action_input["text_negative"], action_input["text_attention_mask_negative"]).to(self._device)
        return None

    def _acwm_episode_require(self):
        episode = getattr(self, "_wam_acwm_episode", None)
        if episode is None:
            raise RuntimeError("ac_wm episode cache is not initialized")
        if episode.get("invalid_reason"):
            raise RuntimeError(
                "ac_wm episode is invalid after a failed cache transition; "
                "call wam_acwm_episode_reset() before retrying. Cause: "
                f"{episode['invalid_reason']}"
            )
        return episode

    @staticmethod
    def _acwm_cache_batch(episode, tensor):
        if episode["parallel_cfg"]:
            return tensor
        return (
            torch.cat([tensor, tensor], dim=0)
            if episode["use_cfg"]
            else tensor
        )

    def _acwm_commit_history_entry(self, episode, entry):
        clean = self._acwm_cache_batch(
            episode, entry["latents"]
        ).transpose(1, 2)
        return self._with_sample_route_override(
            entry["routes"],
            lambda: self.model.acwm_cached_video_chunk(
                clean,
                abs_frame_start=int(entry["abs_frame_start"]),
                context=episode["prompt"],
                clip_feature=episode["clip"],
                cache_chunk_id=int(entry["chunk_id"]),
                window_tokens=int(episode["window_tokens"]),
            ),
        )

    def _acwm_rebuild_history_cache(self, episode):
        """Rebuild exactly the two raw history chunks under the active CLIP."""
        history = episode["raw_history"]
        if len(history) != 2:
            raise RuntimeError(
                f"ac_wm rebuild requires two history chunks, got {len(history)}"
            )
        self.model.reset_acwm_cache()
        for entry in history:
            if entry["routes"] is None:
                _output, routes = self._capture_sample_routes(
                    lambda: self._acwm_commit_history_entry(episode, entry)
                )
                entry["routes"] = routes
            else:
                self._acwm_commit_history_entry(episode, entry)
        episode["history_cache_ready"] = True

    def _acwm_decode_chunk(self, latents, *, reset):
        if not hasattr(self.vae, "decode_stream"):
            raise RuntimeError(
                "exact bounded AC-WM boundary decoding requires "
                "vae.decode_stream"
            )
        self._ensure_vae_on_device(latents)
        vae_parameter = (
            next(self.vae.parameters(), None)
            if hasattr(self.vae, "parameters")
            else None
        )
        decode_input = latents.transpose(1, 2)
        if vae_parameter is not None:
            decode_input = decode_input.to(
                device=vae_parameter.device, dtype=vae_parameter.dtype
            )
        decoded = self.vae.decode_stream(
            decode_input, reset=reset
        )
        if decoded.shape[2] <= 0:
            raise RuntimeError("AC-WM streaming VAE decoded no RGB frames")
        return decoded

    def _acwm_refresh_clip(self, episode, boundary):
        """Refresh CLIP and rebuild raw4 K/V without resetting episode time."""
        image = boundary.transpose(1, 2)
        height, width = boundary.shape[-2:]
        clip, _ys, _seed = self.encode_image(
            image, 1, height, width
        )
        episode["clip"] = self._acwm_cache_batch(episode, clip)
        episode["clip_window_index"] += 1

    @torch.no_grad()
    def wam_acwm_episode_start(
        self,
        action_input: BatchFeature,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        """Initialize one episode-persistent, four-latent ac_wm cache."""
        self.wam_acwm_episode_reset()
        cropped = dict(action_input)
        first = cropped["images"][:, :1]
        cropped["images"] = first.repeat(
            1, ACWM_INITIAL_RGB_FRAMES, 1, 1, 1
        )
        seed_input = BatchFeature(data=cropped)
        (
            latents,
            clip,
            _ys,
            prompt,
            _num_frames,
            _height,
            _width,
        ) = self._wam_encode_obs(
            seed_input, force_single_frame_first_latent=True
        )
        expected_latents = (
            ACWM_INITIAL_WARMUP_LATENTS + ACWM_HISTORY_LATENTS
        )
        if latents.shape[1] != expected_latents:
            raise RuntimeError(
                f"17-frame ac_wm initialization must encode to "
                f"{expected_latents} latents, got {latents.shape[1]}"
            )
        warmup_and_history = latents
        raw_history_latents = latents[
            :,
            ACWM_INITIAL_WARMUP_LATENTS:
            ACWM_INITIAL_WARMUP_LATENTS + ACWM_HISTORY_LATENTS,
        ]
        public_seed_latent = warmup_and_history[:, :1]
        B = raw_history_latents.shape[0]
        requested_cfg_scale = self._wam_cfg_scale(guidance_scale)
        neg_prompt = (
            self._wam_encode_neg(seed_input)
            if requested_cfg_scale > 1.0
            else None
        )
        use_cfg = requested_cfg_scale > 1.0 and neg_prompt is not None
        if requested_cfg_scale > 1.0 and not use_cfg:
            logger.warning(
                "ac_wm CFG scale %.3f requested without a negative prompt; "
                "falling back to the conditional branch.",
                requested_cfg_scale,
            )
        if self.ip_size > 1 and not use_cfg:
            raise RuntimeError(
                "two-GPU ac_wm requires CFG with a valid negative prompt"
            )

        embodiment_id = action_input.embodiment_id.to(self._device)
        num_categories = self.model.action_encoder.num_embodiments
        if num_categories == 1:
            embodiment_id = torch.zeros_like(embodiment_id)
        elif embodiment_id.numel() and (
            int(embodiment_id.min().item()) < 0
            or int(embodiment_id.max().item()) >= num_categories
        ):
            raise ValueError(
                "AC-WM embodiment ID is outside the model-local category range: "
                f"ids={embodiment_id.detach().cpu().tolist()}, "
                f"num_categories={num_categories}"
            )
        if self.ip_size > 1:
            cache_prompt = prompt if self.ip_rank == 0 else neg_prompt
            cache_clip = clip
            cache_embodiment_id = embodiment_id
        else:
            cache_prompt = (
                torch.cat([prompt, neg_prompt], dim=0) if use_cfg else prompt
            )
            cache_clip = (
                torch.cat([clip, clip], dim=0) if use_cfg else clip
            )
            cache_embodiment_id = (
                torch.cat([embodiment_id, embodiment_id], dim=0)
                if use_cfg
                else embodiment_id
            )

        self.model._current_wam_task = "ac_wm"
        if self.moe_enabled:
            self._set_wam_expert_mask(
                batch_size=cache_prompt.shape[0],
                phase="infer",
                action_input=action_input,
                task_names=["ac_wm"] * cache_prompt.shape[0],
            )
        window_tokens = ACWM_HISTORY_LATENTS * self.model.frame_seqlen
        raw_history = [
            {
                "latents": raw_history_latents[
                    :, :ACWM_FUTURE_LATENTS
                ].detach().clone(),
                "abs_frame_start": ACWM_INITIAL_WARMUP_LATENTS,
                "chunk_id": -2,
                "routes": None,
            },
            {
                "latents": raw_history_latents[
                    :, ACWM_FUTURE_LATENTS:
                ].detach().clone(),
                "abs_frame_start": (
                    ACWM_INITIAL_WARMUP_LATENTS + ACWM_FUTURE_LATENTS
                ),
                "chunk_id": -1,
                "routes": None,
            },
        ]
        episode = {
            "prompt": cache_prompt,
            "clip": cache_clip,
            "embodiment_id": cache_embodiment_id,
            "external_batch_size": B,
            "use_cfg": use_cfg,
            "parallel_cfg": self.ip_size > 1,
            "cfg_scale": requested_cfg_scale if use_cfg else 1.0,
            "latent_shape": tuple(raw_history_latents.shape[2:]),
            "seed_latent": public_seed_latent.detach().clone(),
            "seed_rgb": first.detach().clone(),
            "next_frame_start": expected_latents,
            "chunk_id": 0,
            "window_tokens": window_tokens,
            "last_routes": None,
            "raw_history": raw_history,
            "clip_window_index": 0,
            "history_cache_ready": False,
            "invalid_reason": None,
        }
        with torch.amp.autocast(
            dtype=torch.bfloat16,
            device_type=torch.device(self._device).type,
        ):
            self._acwm_rebuild_history_cache(episode)
            self._acwm_decode_chunk(
                warmup_and_history, reset=True
            )
        self._wam_acwm_episode = episode
        return BatchFeature(
            data={
                "seed_video": public_seed_latent,
                "cache_latents": ACWM_HISTORY_LATENTS,
                "next_frame_start": expected_latents,
            }
        )

    @torch.no_grad()
    def wam_acwm_episode_predict(
        self,
        action_input: BatchFeature,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        """Generate and commit one two-latent ac_wm chunk."""
        episode = self._acwm_episode_require()
        B = int(episode["external_batch_size"])
        action = action_input.action.to(
            dtype=self.dtype, device=self._device
        )
        if action.shape[1] != FASTWAM_ACTIONS_PER_CHUNK:
            raise ValueError(
                f"ac_wm requires {FASTWAM_ACTIONS_PER_CHUNK} actions, got "
                f"{action.shape[1]}"
            )
        requested_cfg_scale = self._wam_cfg_scale(guidance_scale)
        use_cfg = bool(episode["use_cfg"])
        if (requested_cfg_scale > 1.0) != use_cfg:
            raise RuntimeError(
                "ac_wm CFG mode changed after cache initialization; reset "
                "the episode first"
            )
        g = requested_cfg_scale if use_cfg else 1.0
        parallel_cfg = bool(episode["parallel_cfg"])

        def cache_batch(tensor):
            return self._acwm_cache_batch(episode, tensor)

        prompt = episode["prompt"]
        clip = episode["clip"]
        embodiment_id = episode["embodiment_id"]
        steps = num_inference_steps or self.num_inference_steps
        self.scheduler.set_timesteps(
            steps, denoising_strength=self.denoising_strength
        )
        timesteps = self.scheduler.timesteps.to(self._device)
        channels, height, width = episode["latent_shape"]
        chunk_id = int(episode["chunk_id"])
        noisy = self.generate_noise(
            (B, ACWM_FUTURE_LATENTS, channels, height, width),
            # Keep evaluation deterministic without reusing identical noise in
            # every generated chunk of the same episode.
            seed=int(self.seed) + chunk_id,
            device=self._device,
            dtype=self.dtype,
        )
        routes = None
        self.model._current_wam_task = "ac_wm"
        if self.moe_enabled:
            self._set_wam_expert_mask(
                batch_size=cache_batch(noisy).shape[0],
                phase="infer",
                action_input=action_input,
                task_names=["ac_wm"] * cache_batch(noisy).shape[0],
            )

        try:
            if not episode["history_cache_ready"]:
                with torch.amp.autocast(
                    dtype=torch.bfloat16,
                    device_type=torch.device(self._device).type,
                ):
                    self._acwm_rebuild_history_cache(episode)
            with torch.amp.autocast(
                dtype=torch.bfloat16,
                device_type=torch.device(self._device).type,
            ):
                for step_index, timestep in enumerate(timesteps):
                    def model_step():
                        return self.model.acwm_cached_video_action_chunk(
                            cache_batch(noisy).transpose(1, 2),
                            cache_batch(action),
                            abs_frame_start=int(episode["next_frame_start"]),
                            timestep_val=float(timestep),
                            context=prompt,
                            clip_feature=clip,
                            embodiment_id=embodiment_id,
                        )

                    if step_index == len(timesteps) - 1:
                        video_pred_all, routes = self._capture_sample_routes(
                            model_step
                        )
                    else:
                        video_pred_all = model_step()
                    video_pred_all = video_pred_all.transpose(1, 2)
                    if parallel_cfg:
                        exchanged = self._exchange_predictions(
                            [(
                                video_pred_all,
                                torch.zeros(
                                    (), device=video_pred_all.device,
                                    dtype=video_pred_all.dtype,
                                ),
                            )]
                        )
                        cond = exchanged[0][0]
                        uncond = exchanged[1][0]
                        prediction = uncond + g * (cond - uncond)
                    else:
                        cond = video_pred_all[:B]
                        if use_cfg:
                            uncond = video_pred_all[B:]
                            prediction = uncond + g * (cond - uncond)
                        else:
                            prediction = cond
                    noisy = self.scheduler.step(prediction, timestep, noisy)

                entry = {
                    "latents": noisy.detach().clone(),
                    "abs_frame_start": int(episode["next_frame_start"]),
                    "chunk_id": chunk_id,
                    "routes": routes,
                }
                self._acwm_commit_history_entry(episode, entry)
                episode["raw_history"].append(entry)
                episode["raw_history"] = episode["raw_history"][-2:]
                episode["history_cache_ready"] = False
                decoded_chunk = self._acwm_decode_chunk(noisy, reset=False)

            episode["last_routes"] = routes
            episode["next_frame_start"] += ACWM_FUTURE_LATENTS
            episode["chunk_id"] += 1
            if episode["chunk_id"] % ACWM_CHUNKS_PER_CLIP == 0:
                with torch.amp.autocast(
                    dtype=torch.bfloat16,
                    device_type=torch.device(self._device).type,
                ):
                    self._acwm_refresh_clip(
                        episode, decoded_chunk[:, :, -1:]
                    )
        except Exception as exc:
            episode["invalid_reason"] = f"{type(exc).__name__}: {exc}"
            self.model.reset_acwm_cache()
            if hasattr(self.vae, "reset_streaming_decode"):
                self.vae.reset_streaming_decode()
            raise RuntimeError(
                "ac_wm cache transition failed; the episode was invalidated "
                "and must be reset before retry"
            ) from exc
        return BatchFeature(
            data={
                "video": noisy,
                "decoded_video": decoded_chunk,
                "cache_latents": ACWM_HISTORY_LATENTS,
                "next_frame_start": int(episode["next_frame_start"]),
            }
        )

    def wam_acwm_episode_reset(self) -> None:
        if hasattr(self.model, "reset_acwm_cache"):
            self.model.reset_acwm_cache()
        if hasattr(self.vae, "reset_streaming_decode"):
            self.vae.reset_streaming_decode()
        self.model._wam_sample_route_capture = None
        self.model._wam_sample_route_override = None
        self.model._current_wam_task = None
        self.model._current_wam_expert_mask = None
        self.model._current_wam_token_group_routing = None
        self._wam_acwm_episode = None

    @torch.no_grad()
    def wam_masked_episode_step(
        self,
        action_input: BatchFeature,
        task: str,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        return run_unified_masked_episode_step(
            self,
            action_input,
            task=task,
            num_inference_steps=num_inference_steps,
        )

    @torch.no_grad()
    def wam_acwm_unified_episode_step(
        self,
        action_input: BatchFeature,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        return run_unified_acwm_episode_step(
            self,
            action_input,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def wam_acwm_reference_cache_episode_step(
        self,
        action_input: BatchFeature,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        try:
            return run_acwm_cached_episode_step(
                self,
                action_input,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        except Exception:
            # A failed denoising/decode/window rebuild must not leave a
            # partially written KV cache or streaming VAE state reachable by
            # the next long-lived robot request.
            reset_unified_inference(self)
            raise

    @torch.no_grad()
    def wam_joint_dense_episode_step(
        self,
        action_input: BatchFeature,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        return run_joint_dense_episode_step(
            self,
            action_input,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def wam_idm_dense_episode_step(
        self,
        action_input: BatchFeature,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        return run_idm_dense_episode_step(
            self,
            action_input,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def wam_masked_get_action(self, action_input: BatchFeature, task: str | None = None, num_inference_steps: int | None = None, guidance_scale: float | None = None, generate_video: bool = False, train_experts: bool = False) -> BatchFeature:
        """Open-loop inference for single-pass action-only masked tasks.

        - action_only_fast_wam / action_only_gigaworld_policy: condition on one
          clean RGB seed and denoise only the 24-step action. Their sample-level
          router pools noisy action tokens at both train and inference, so no
          future-video target is constructed or required during deployment.
        AC-WM/world_model is episode-only and fails below with migration
        guidance before constructing its retired single-pass path.
        """
        self.set_frozen_modules_to_eval_mode()
        task = task or self.moe_task
        norm = str(task).strip().lower().replace("-", "_").replace(" ", "_")
        canonical_task = canonical_masked_task(norm)
        is_world_model = canonical_task == "world_model"
        if is_world_model:
            raise RuntimeError(
                "world_model single-pass inference is retired; use "
                "wam_acwm_episode_step() and wam_acwm_episode_reset()"
            )
        is_fastwam = canonical_task == "action_only_fast_wam"
        is_gigaworld = canonical_task == "action_only_gigaworld_policy"
        if is_fastwam:
            self._warn_fastwam_retrain_boundary_once()
        if is_world_model:
            self._warn_world_model_retrain_boundary_once()
        if is_gigaworld:
            self._warn_gigaworld_retrain_boundary_once()
        if generate_video and (is_fastwam or is_gigaworld):
            raise ValueError(
                f"{canonical_task} is action-only and no longer generates future video"
            )
        g = self._wam_cfg_scale(guidance_scale)

        if is_fastwam or is_world_model or is_gigaworld:
            # Keep the public batch schema compatible with the joint dataset.
            # World-model inference needs its complete nine-frame target layout;
            # action-only deployment consumes exactly one clean RGB seed.
            cropped = dict(action_input)
            if "images" in cropped:
                image_count = 9 if is_world_model else 1
                cropped["images"] = cropped["images"][:, :image_count]
            if "state" in cropped:
                cropped["state"] = cropped["state"][:, :1]
            if "action" in cropped:
                cropped["action"] = cropped["action"][:, :FASTWAM_ACTIONS_PER_CHUNK]
            action_input = BatchFeature(data=cropped)

        embodiment_id = action_input.embodiment_id
        state = action_input.state.to(dtype=torch.bfloat16, device=self._device)
        latents, clip_feas, ys, prompt_embs, num_frames, Hl, Wl = self._wam_encode_obs(
            action_input,
            force_single_frame_first_latent=(
                is_world_model or is_fastwam or is_gigaworld
            ),
        )
        B = latents.shape[0]
        steps = num_inference_steps or self.num_inference_steps
        self.scheduler.set_timesteps(steps, denoising_strength=self.denoising_strength)
        timesteps = self.scheduler.timesteps.to(self._device)
        tokens_per_frame = (Hl // 2) * (Wl // 2)

        self.model._current_wam_task = norm
        if self.moe_enabled:
            self._set_wam_expert_mask(batch_size=B, phase="infer", action_input=action_input)

        if is_world_model:
            # Denoise all future video frames given clean action; frame 0 clean.
            action = action_input.action.to(dtype=self.dtype, device=self._device)
            if action.shape[1] != FASTWAM_ACTIONS_PER_CHUNK or num_frames != 3:
                raise ValueError(
                    "world_model requires one 9-RGB/24-action horizon "
                    f"(3 temporal latents); got action={action.shape[1]}, latents={num_frames}"
                )
            seq_len = num_frames * tokens_per_frame
            noisy = torch.randn_like(latents)
            noisy[:, 0] = latents[:, 0]
            timestep_action = torch.zeros(B, action.shape[1], device=self._device)
            neg_embs = self._wam_encode_neg(action_input) if g > 1.0 else None
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                for t in timesteps:
                    ts_video = torch.full((B, num_frames), float(t), device=self._device)
                    ts_video[:, 0] = 0

                    def _vpred(ctx):
                        vp, _ = self.model._forward_train_masked(
                            task=canonical_masked_task(norm), x=noisy.transpose(1, 2),
                            timestep=ts_video, timestep_action=timestep_action,
                            context=ctx, seq_len=seq_len, y=ys, clip_feature=clip_feas,
                            action=action, state=state, embodiment_id=embodiment_id,
                        )
                        return vp.transpose(1, 2)  # [B, F, C, H, W]

                    cond = _vpred(prompt_embs)
                    if g > 1.0 and neg_embs is not None:
                        # CFG (DreamZero-style): pred = uncond + g*(cond - uncond).
                        uncond = _vpred(neg_embs)
                        video_pred = uncond + g * (cond - uncond)
                    else:
                        video_pred = cond
                    noisy = self.scheduler.step(video_pred, t, noisy)
                    noisy[:, 0] = latents[:, 0]
            return BatchFeature(data={"video": noisy})

        action_dim = self.action_dim
        action_horizon = (
            FASTWAM_ACTIONS_PER_CHUNK
            if (is_fastwam or is_gigaworld)
            else (action_input.action.shape[1] if "action" in action_input.keys() else self.action_horizon)
        )

        if is_fastwam or is_gigaworld:
            if num_frames != 1:
                raise ValueError(
                    f"{canonical_task} inference requires one clean RGB seed; "
                    f"got {num_frames} latents"
                )
            first_frame = latents[:, :1]
            seq_len = tokens_per_frame
            ts_video = torch.zeros(B, 1, device=self._device)
            noisy_action = self._wam_fixed_inference_noise(
                (B, action_horizon, action_dim)
            )
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                for t in timesteps:
                    timestep_action = torch.full((B, action_horizon), float(t), device=self._device)
                    _, action_pred = self.model._forward_train_masked(
                        task=canonical_masked_task(norm), x=first_frame.transpose(1, 2),
                        timestep=ts_video, timestep_action=timestep_action,
                        context=prompt_embs, seq_len=seq_len, y=ys, clip_feature=clip_feas,
                        action=noisy_action, state=state, embodiment_id=embodiment_id,
                    )
                    noisy_action = self.scheduler.step(action_pred, t, noisy_action)
            return BatchFeature(data={"action": noisy_action})

        # Non-masked fallback retained for callers outside the canonical
        # FastWAM/Giga/WM task set.
        if train_experts and self.moe_enabled:
            self._set_wam_expert_mask(batch_size=B, phase="train", action_input=action_input)
        first_frame = latents[:, :1]  # [B, 1, C, H, W]
        seq_len = 1 * tokens_per_frame
        noisy_action = torch.randn(B, action_horizon, action_dim, device=self._device, dtype=self.dtype)
        ts_video = torch.zeros(B, 1, device=self._device)  # first frame clean
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            for t in timesteps:
                timestep_action = torch.full((B, action_horizon), float(t), device=self._device)
                _, action_pred = self.model._forward_train_masked(
                    task=canonical_masked_task(norm), x=first_frame.transpose(1, 2),
                    timestep=ts_video, timestep_action=timestep_action,
                    context=prompt_embs, seq_len=seq_len, y=ys, clip_feature=clip_feas,
                    action=noisy_action, state=state, embodiment_id=embodiment_id,
                )
                noisy_action = self.scheduler.step(action_pred, t, noisy_action)
        return BatchFeature(data={"action": noisy_action})

    @torch.no_grad()
    def wam_idm_get_action(self, action_input: BatchFeature, num_inference_steps: int | None = None, guidance_scale: float | None = None) -> BatchFeature:
        """Compatibility wrapper for one canonical two-stage IDM episode chunk.

        This deliberately reuses the persistent episode/cache implementation;
        there is no longer a second full-sequence inference path or any
        ``WAM_MOE_PASS`` expert switching.
        """
        data = {key: value for key, value in action_input.items()}
        if "images" in data:
            data["images"] = data["images"][:, :1]
        if "state" in data:
            data["state"] = data["state"][:, :1]
        if "action" in data:
            data["action"] = data["action"][:, :IDM_ACTIONS_PER_CHUNK]
        one_chunk = BatchFeature(data=data)
        self.wam_idm_episode_reset()
        try:
            self.wam_idm_episode_start(
                one_chunk, guidance_scale=guidance_scale
            )
            return self.wam_idm_episode_predict(
                one_chunk,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        finally:
            self.wam_idm_episode_reset()

    def _idm_episode_require(self):
        episode = getattr(self, "_wam_idm_episode", None)
        if episode is None or not episode["ring"].active:
            raise RuntimeError("IDM episode is not active; call wam_idm_episode_start() first")
        return episode

    @torch.no_grad()
    @_abort_idm_episode_on_error
    def wam_idm_episode_start(
        self, action_input: BatchFeature,
        window_chunks: int = IDM_DEFAULT_WINDOW_CHUNKS,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        """Start one canonical IDM episode and write only the seed latent."""
        self.set_frozen_modules_to_eval_mode()
        self._warn_idm_retrain_boundary_once()
        latents, clip, ys, prompt, _, _, _ = self._wam_encode_obs(
            action_input, idm_stream_mode="start"
        )
        ring = IDMEpisodeCache(window_chunks)
        seed = ring.start()
        # ``action_input.embodiment_id`` is the external dataset tag (for
        # example agilex_eef=34), while this DreamZero checkpoint deliberately
        # collapses the category-specific state/action projections to one local
        # embodiment.  The non-cached training and inference forwards already
        # remap that local category to zero; do the same once at persistent IDM
        # episode start so every later state/action cache write uses a valid ID.
        embodiment_id = action_input.embodiment_id.to(self._device)
        num_categories = self.model.state_encoder.layer1.num_categories
        if num_categories == 1:
            embodiment_id = torch.zeros_like(embodiment_id)
        elif embodiment_id.numel() and (
            int(embodiment_id.min().item()) < 0
            or int(embodiment_id.max().item()) >= num_categories
        ):
            raise ValueError(
                "IDM embodiment ID is outside the model-local category range: "
                f"ids={embodiment_id.detach().cpu().tolist()}, "
                f"num_categories={num_categories}"
            )
        requested_cfg_scale = self._wam_cfg_scale(guidance_scale)
        neg_prompt = (
            self._wam_encode_neg(action_input)
            if requested_cfg_scale > 1.0
            else None
        )
        use_cfg = requested_cfg_scale > 1.0 and neg_prompt is not None
        if requested_cfg_scale > 1.0 and not use_cfg:
            logger.warning(
                "IDM video CFG scale %.3f requested without a negative prompt; "
                "falling back to the conditional branch.",
                requested_cfg_scale,
            )
        if self.ip_size > 1 and not use_cfg:
            raise RuntimeError(
                "two-GPU IDM requires video CFG with a valid negative prompt; "
                "serial fallback is intentionally disabled"
            )

        external_batch_size = latents.shape[0]
        if self.ip_size > 1:
            # Match DreamZero image-parallel CFG: rank 0 owns the conditional
            # cache and rank 1 owns the negative cache.  Each GPU therefore
            # stores batch B rather than a serially duplicated batch 2B.
            cache_prompt = prompt if self.ip_rank == 0 else neg_prompt
            cache_clip = clip
            cache_seed = latents[:, :1]
            cache_embodiment_id = embodiment_id
        else:
            cache_prompt = (
                torch.cat([prompt, neg_prompt], dim=0) if use_cfg else prompt
            )
            cache_clip = torch.cat([clip, clip], dim=0) if use_cfg else clip
            cache_seed = (
                torch.cat([latents[:, :1], latents[:, :1]], dim=0)
                if use_cfg
                else latents[:, :1]
            )
            cache_embodiment_id = (
                torch.cat([embodiment_id, embodiment_id], dim=0)
                if use_cfg
                else embodiment_id
            )

        self.model._current_wam_task = "idm_lingbot_va"
        os.environ["WAM_MOE_PASS"] = "default"
        if self.moe_enabled:
            self._set_wam_expert_mask(
                batch_size=cache_seed.shape[0], phase="infer", action_input=action_input,
                task_names=["idm_lingbot_va"] * cache_seed.shape[0],
            )
        self.model.reset_idm_cache()
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            self.model.idm_cached_video_chunk(
                cache_seed.transpose(1, 2), abs_frame_start=0, timestep_val=0.0,
                context=cache_prompt, clip_feature=cache_clip, append_clean=True,
                cache_chunk_id=seed.chunk_id, is_pred=False,
                max_cache_chunks=ring.capacity + 1,
            )
        self._wam_idm_episode = {
            "ring": ring, "prompt": cache_prompt, "clip": cache_clip, "ys": ys,
            "embodiment_id": cache_embodiment_id,
            "external_batch_size": external_batch_size,
            "use_cfg": use_cfg,
            "parallel_cfg": self.ip_size > 1,
            "cfg_scale": requested_cfg_scale if use_cfg else 1.0,
            "latent_shape": tuple(latents.shape[2:]),
            "seed_latent": latents[:, :1].detach().clone(),
            "latest_condition_latent": latents[:, -1:].detach().clone(),
            "pending_state": None,
            "pending_video_routes": None,
            "pending_action_routes": None,
            "last_video_routes": None,
            "last_action_routes": None,
        }
        return BatchFeature(data={"cache_metadata": ring.snapshot()})

    @torch.no_grad()
    @_abort_idm_episode_on_error
    def wam_idm_episode_predict(
        self, action_input: BatchFeature, num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        """Predict one provisional ``state -> 2 latent -> 24 action`` chunk."""
        episode = self._idm_episode_require()
        ring = episode["ring"]
        chunk = ring.predict()
        state = action_input.state.to(dtype=torch.bfloat16, device=self._device)
        if state.shape[1] != 1:
            state = state[:, -1:]
        episode["pending_state"] = state
        B = episode["external_batch_size"]
        prompt, clip = episode["prompt"], episode["clip"]
        embodiment_id = episode["embodiment_id"]
        steps = num_inference_steps or self.num_inference_steps
        self.scheduler.set_timesteps(steps, denoising_strength=self.denoising_strength)
        timesteps = self.scheduler.timesteps.to(self._device)
        requested_cfg_scale = self._wam_cfg_scale(guidance_scale)
        use_cfg = bool(episode["use_cfg"])
        if requested_cfg_scale > 1.0 and not use_cfg:
            raise RuntimeError(
                "IDM CFG was enabled after the episode cache was created. "
                "Reset the episode and start it with the same guidance scale."
            )
        g = requested_cfg_scale if use_cfg else 1.0

        parallel_cfg = bool(episode.get("parallel_cfg", False))

        def cache_batch(tensor):
            if parallel_cfg:
                return tensor
            return torch.cat([tensor, tensor], dim=0) if use_cfg else tensor

        cache_limit = ring.cache_record_limit(chunk.chunk_id)
        # Spatial latent shape is fixed by the seed cached at start.
        latent_channels, Hl, Wl = episode["latent_shape"]
        self._with_sample_route_override(
            episode.get("last_video_routes"),
            lambda: self.model.idm_cache_write_state(
                cache_batch(state), embodiment_id, prompt, clip,
                abs_state_start=chunk.state_id,
                cache_chunk_id=chunk.chunk_id, is_pred=True,
                max_cache_chunks=cache_limit,
            ),
        )
        if parallel_cfg:
            noisy_v = self._wam_fixed_inference_noise(
                (B, IDM_LATENTS_PER_CHUNK, latent_channels, Hl, Wl)
            )
            shadow_a = self._wam_fixed_inference_noise(
                (B, IDM_ACTIONS_PER_CHUNK, self.action_dim)
            )
        else:
            noisy_v = self._wam_fixed_inference_noise(
                (B, IDM_LATENTS_PER_CHUNK, latent_channels, Hl, Wl)
            )
            shadow_a = self._wam_fixed_inference_noise(
                (B, IDM_ACTIONS_PER_CHUNK, self.action_dim)
            )
        video_routes = None
        action_routes = None
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            for step_index, t in enumerate(timesteps):
                def video_action_step():
                    return self.model.idm_cached_video_action_chunk(
                        cache_batch(noisy_v).transpose(1, 2),
                        cache_batch(shadow_a),
                        chunk.frame_start,
                        chunk.action_start,
                        float(t),
                        prompt,
                        clip,
                        embodiment_id,
                        cache_chunk_id=chunk.chunk_id,
                        is_pred=True,
                        max_cache_chunks=cache_limit,
                    )
                if step_index == len(timesteps) - 1:
                    (
                        (video_pred_all, action_shadow_pred_all),
                        video_routes,
                    ) = self._capture_sample_routes(video_action_step)
                else:
                    video_pred_all, action_shadow_pred_all = video_action_step()
                video_pred_all = video_pred_all.transpose(1, 2)
                if parallel_cfg:
                    exchanged = self._exchange_predictions(
                        [(video_pred_all, torch.zeros((), device=video_pred_all.device, dtype=video_pred_all.dtype))]
                    )
                    cond = exchanged[0][0]
                    uncond = exchanged[1][0]
                    vpred = uncond + g * (cond - uncond)
                    action_shadow_pred = action_shadow_pred_all
                else:
                    cond = video_pred_all[:B]
                    if use_cfg:
                        uncond = video_pred_all[B:]
                        vpred = uncond + g * (cond - uncond)
                    else:
                        vpred = cond
                    action_shadow_pred = action_shadow_pred_all[:B]
                noisy_v = self.scheduler.step(vpred, t, noisy_v)
                shadow_a = self.scheduler.step(
                    action_shadow_pred, t, shadow_a
                )
                emit_wam_inference_trace(
                    self,
                    task="idm_lingbot_va",
                    backend="cache",
                    chunk_index=chunk.chunk_id,
                    phase="video",
                    step_index=step_index,
                    prediction=vpred,
                    sample=noisy_v,
                )
            self._with_sample_route_override(
                video_routes,
                lambda: self.model.idm_cached_video_chunk(
                    cache_batch(noisy_v).transpose(1, 2),
                    chunk.frame_start, 0.0, prompt, clip,
                    append_clean=True, cache_chunk_id=chunk.chunk_id, is_pred=True,
                    max_cache_chunks=cache_limit,
                ),
            )
            noisy_a = torch.empty(
                B, IDM_ACTIONS_PER_CHUNK, self.action_dim,
                device=self._device, dtype=self.dtype,
            )
            if not parallel_cfg or self.ip_rank == 0:
                noisy_a = self._wam_fixed_inference_noise(noisy_a.shape)
                shadow_v = self._wam_fixed_inference_noise(noisy_v.shape)
                for step_index, t in enumerate(timesteps):
                    def action_video_step():
                        return self.model.idm_cached_video_action_chunk(
                            cache_batch(shadow_v).transpose(1, 2),
                            cache_batch(noisy_a),
                            chunk.frame_start,
                            chunk.action_start,
                            float(t),
                            prompt,
                            clip,
                            embodiment_id,
                            cache_chunk_id=chunk.chunk_id,
                            is_pred=True,
                            max_cache_chunks=cache_limit,
                            action_phase=True,
                        )
                    if step_index == len(timesteps) - 1:
                        (
                            (video_shadow_pred_all, action_pred_all),
                            action_routes,
                        ) = self._capture_sample_routes(action_video_step)
                    else:
                        video_shadow_pred_all, action_pred_all = action_video_step()
                    # LingBot/DreamZero apply CFG only to video.  In two-GPU
                    # mode rank 0 alone executes action DiT against the positive
                    # cache; rank 1 waits for the final generated action.
                    apred = action_pred_all if parallel_cfg else action_pred_all[:B]
                    vpred_shadow = (
                        video_shadow_pred_all.transpose(1, 2)
                        if parallel_cfg
                        else video_shadow_pred_all[:B].transpose(1, 2)
                    )
                    shadow_v = self.scheduler.step(
                        vpred_shadow, t, shadow_v
                    )
                    noisy_a = self.scheduler.step(apred, t, noisy_a)
                    emit_wam_inference_trace(
                        self,
                        task="idm_lingbot_va",
                        backend="cache",
                        chunk_index=chunk.chunk_id,
                        phase="action",
                        step_index=step_index,
                        prediction=apred,
                        sample=noisy_a,
                    )
            if parallel_cfg:
                dist.broadcast(noisy_a, src=0, group=self.ip_group)
            cache_action_routes = action_routes or video_routes
            self._with_sample_route_override(
                cache_action_routes,
                lambda: self.model.idm_cached_action_chunk(
                    cache_batch(noisy_a), chunk.action_start, 0.0,
                    prompt, clip, embodiment_id,
                    append_clean=True, cache_chunk_id=chunk.chunk_id, is_pred=True,
                    max_cache_chunks=cache_limit,
                ),
            )
        episode["pending_video_routes"] = video_routes
        episode["pending_action_routes"] = action_routes or video_routes
        return BatchFeature(data={
            "action": noisy_a, "video": noisy_v,
            "cache_metadata": ring.snapshot(),
        })

    @torch.no_grad()
    @_abort_idm_episode_on_error
    def wam_idm_episode_commit(self, action_input: BatchFeature) -> BatchFeature:
        """Replace the pending prediction with real 2-latent/24-action history."""
        episode = self._idm_episode_require()
        ring = episode["ring"]
        if ring.pending is None:
            raise RuntimeError("no pending IDM prediction to commit")
        pred = ring.pending
        real_latents = self._wam_encode_obs(
            action_input,
            idm_stream_mode="commit",
            encode_conditioning=False,
        )[0]
        real_actions = action_input.action.to(dtype=self.dtype, device=self._device)
        if real_latents.shape[1] != IDM_LATENTS_PER_CHUNK:
            raise ValueError("IDM commit requires exactly two real latent frames")
        if real_actions.shape[1] != IDM_ACTIONS_PER_CHUNK:
            raise ValueError("IDM commit requires exactly 24 executed actions")
        self.model.clear_idm_pred_cache()
        real = ring.commit()
        cache_limit = ring.cache_record_limit(ring.next_chunk_id)
        prompt, clip = episode["prompt"], episode["clip"]
        embodiment_id = episode["embodiment_id"]
        use_cfg = bool(episode["use_cfg"])
        parallel_cfg = bool(episode.get("parallel_cfg", False))

        def cache_batch(tensor):
            if parallel_cfg:
                return tensor
            return torch.cat([tensor, tensor], dim=0) if use_cfg else tensor

        state = episode["pending_state"]
        video_routes = episode.get("pending_video_routes")
        action_routes = episode.get("pending_action_routes")
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            self._with_sample_route_override(
                video_routes,
                lambda: self.model.idm_cache_write_state(
                    cache_batch(state), embodiment_id, prompt, clip,
                    abs_state_start=real.state_id,
                    cache_chunk_id=real.chunk_id, is_pred=False,
                    max_cache_chunks=cache_limit,
                ),
            )
            self._with_sample_route_override(
                video_routes,
                lambda: self.model.idm_cached_video_chunk(
                    cache_batch(real_latents).transpose(1, 2),
                    real.frame_start, 0.0, prompt, clip,
                    append_clean=True, cache_chunk_id=real.chunk_id, is_pred=False,
                    max_cache_chunks=cache_limit,
                ),
            )
            self._with_sample_route_override(
                action_routes,
                lambda: self.model.idm_cached_action_chunk(
                    cache_batch(real_actions), real.action_start, 0.0,
                    prompt, clip, embodiment_id,
                    append_clean=True, cache_chunk_id=real.chunk_id, is_pred=False,
                    max_cache_chunks=cache_limit,
                ),
            )
        episode["pending_state"] = None
        episode["last_video_routes"] = video_routes
        episode["last_action_routes"] = action_routes
        episode["pending_video_routes"] = None
        episode["pending_action_routes"] = None
        episode["latest_condition_latent"] = real_latents[:, -1:].detach().clone()
        return BatchFeature(data={"cache_metadata": ring.snapshot()})

    def wam_idm_episode_reset(self) -> None:
        """End the episode and clear all model/metadata cache state."""
        self.model.reset_idm_cache()
        self.model._wam_sample_route_capture = None
        self.model._wam_sample_route_override = None
        episode = getattr(self, "_wam_idm_episode", None)
        if episode is not None:
            episode["ring"].reset()
        if hasattr(self.vae, "reset_streaming_encode"):
            self.vae.reset_streaming_encode()
        self._wam_idm_episode = None
        os.environ["WAM_MOE_PASS"] = "default"

    @torch.no_grad()
    def wam_idm_get_action_cached(self, action_input: BatchFeature, num_inference_steps: int | None = None,
                                  guidance_scale: float | None = None, num_frame_per_block: int | None = None,
                                  window_blocks: int | None = None, teacher_force_history: bool = False,
                                  train_experts: bool = True) -> BatchFeature:
        """Compatibility wrapper over the canonical persistent episode API.

        The former independent 96-step implementation and expert-switch flags
        are intentionally removed.  A standard 33/96/4 teacher-forcing window
        is replayed as four calls to the same start/predict/commit lifecycle used
        by open-loop episodes and the real robot.
        """
        if num_frame_per_block not in (None, IDM_LATENTS_PER_CHUNK):
            raise ValueError("IDM compatibility API requires two latent frames per chunk")
        if not teacher_force_history:
            raise ValueError(
                "wam_idm_get_action_cached now requires teacher_force_history=True; "
                "free-running callers must use the explicit episode API"
            )
        if not train_experts:
            raise ValueError(
                "WAM_IDM_TRAIN_EXPERTS/per-pass switching was removed; "
                "routing granularity selects sample-level or token-group execution"
            )

        images = action_input.images
        actions = action_input.action
        states = action_input.state
        if images.shape[1] != 33 or actions.shape[1] != 96 or states.shape[1] != 4:
            raise ValueError(
                "IDM compatibility window must be 33 frames / 96 actions / 4 states; "
                f"got {images.shape[1]}/{actions.shape[1]}/{states.shape[1]}"
            )

        def sliced(**updates):
            data = dict(action_input)
            data.update(updates)
            return BatchFeature(data=data)

        capacity = window_blocks or IDM_DEFAULT_WINDOW_CHUNKS
        self.wam_idm_episode_start(
            sliced(images=images[:, :1], state=states[:, :1], action=actions[:, :24]),
            window_chunks=capacity, guidance_scale=guidance_scale,
        )
        videos = []
        predictions = []
        try:
            for chunk in range(4):
                a0 = chunk * IDM_ACTIONS_PER_CHUNK
                f0 = chunk * 8
                current = sliced(
                    images=images[:, f0:f0 + 1],
                    state=states[:, chunk:chunk + 1],
                    action=actions[:, a0:a0 + IDM_ACTIONS_PER_CHUNK],
                )
                out = self.wam_idm_episode_predict(
                    current,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                predictions.append(out["action"])
                videos.append(out["video"])
                if chunk < 3:
                    self.wam_idm_episode_commit(
                        sliced(
                            images=images[:, f0:f0 + 9],
                            state=states[:, chunk:chunk + 1],
                            action=actions[:, a0:a0 + IDM_ACTIONS_PER_CHUNK],
                        )
                    )
            seed = self._wam_idm_episode["seed_latent"]
            return BatchFeature(data={
                "action": torch.cat(predictions, dim=1),
                "video": torch.cat([seed] + videos, dim=1),
            })
        finally:
            self.wam_idm_episode_reset()

    @torch.no_grad()
    def wam_action_rollout(self, reset_fn, step_fn, num_chunks: int, task: str | None = None, num_inference_steps: int | None = None):
        """Compatibility callback adapter over the canonical episode runner.

        The env is injected via callbacks so this stays env/sim-agnostic:
          - reset_fn() -> action_input (BatchFeature) for the first observation
          - step_fn(action) -> action_input for the next observation, or None to stop
        Each chunk: infer an action chunk from the current observation (first frame +
        state), hand it to step_fn (the caller executes it and returns the new
        observation whose last frame becomes the next first frame). Returns the list
        of predicted action chunks.
        """
        from ..wam_episode import (
            WAMEpisodeRunner,
            unified_inference_consumption_enabled,
        )

        mode = task or "action_only_fast_wam"
        predict_chunk = (
            self.wam_masked_episode_step
            if unified_inference_consumption_enabled()
            else self.wam_masked_get_action
        )
        runner = WAMEpisodeRunner(
            mode,
            reset_fn=lambda: self.wam_episode_reset(mode),
            predict_fn=lambda obs: predict_chunk(
                obs, task=mode, num_inference_steps=num_inference_steps
            ),
        )
        obs = reset_fn()
        actions = []
        runner.reset_episode()
        try:
            for _ in range(num_chunks):
                request = runner.next_request()
                request_obs = _rollout_observation_for_request(obs, request)
                images = request_obs.get("images") if hasattr(request_obs, "get") else None
                frame_count = (
                    int(images.shape[1])
                    if images is not None
                    else request.expected_frames
                )
                out = runner.predict(request_obs, frame_count=frame_count)
                act = out["action"]
                actions.append(act)
                obs = step_fn(act)
                if obs is None:
                    break
        finally:
            runner.reset_episode()
        return actions

    def wam_episode_reset(self, runtime_mode: str | None = None) -> None:
        """Canonical reset for all deployment modes and runtime transitions."""
        if runtime_mode is not None:
            from ..wam_episode import canonical_episode_mode

            canonical_episode_mode(runtime_mode)
        self.wam_joint_episode_reset()
        self.wam_idm_episode_reset()
        self.wam_acwm_episode_reset()
        reset_unified_inference(self)

    def wam_joint_episode_reset(self) -> None:
        """Clear every persistent Joint inference condition/cache in one place."""
        for name in (
            "language",
            "kv_cache1",
            "kv_cache_neg",
            "crossattn_cache",
            "crossattn_cache_neg",
            "clip_feas",
            "ys",
        ):
            setattr(self, name, None)
        self.current_start_frame = 0
        self._joint_history_routes = None
        self._last_joint_route_capture = None
        model = getattr(self, "model", None)
        if model is not None:
            if hasattr(model, "_current_wam_expert_mask"):
                model._current_wam_expert_mask = None
            if hasattr(model, "_current_wam_task"):
                model._current_wam_task = None
            model._wam_sample_route_capture = None
            model._wam_sample_route_override = None
        os.environ["WAM_MOE_PASS"] = "default"

    @torch.no_grad()
    def wam_idm_rollout(self, reset_fn, step_fn, num_chunks: int, num_inference_steps: int | None = None):
        """Compatibility callback adapter over the persistent IDM runner.

        Same callback contract as ``wam_action_rollout``. Each chunk runs the
        two-pass inference (shared+vgm video -> shared+idm action); the caller
        executes the action and returns the next 9-frame real observation.
        """
        from ..wam_episode import WAMEpisodeRunner

        pending_action = None

        def reset_episode():
            nonlocal pending_action
            pending_action = None
            self.wam_episode_reset("idm_lingbot_va")

        def predict_one(obs):
            nonlocal pending_action
            if pending_action is None:
                self.wam_idm_episode_start(obs)
            else:
                data = dict(obs)
                data["action"] = pending_action
                obs = BatchFeature(data=data)
                self.wam_idm_episode_commit(obs)
            out = self.wam_idm_episode_predict(
                obs, num_inference_steps=num_inference_steps
            )
            pending_action = out["action"].detach().clone()
            return out

        runner = WAMEpisodeRunner(
            "idm_lingbot_va",
            reset_fn=reset_episode,
            predict_fn=predict_one,
        )
        obs = reset_fn()
        outputs = []
        runner.reset_episode()
        try:
            for _ in range(num_chunks):
                request = runner.next_request()
                request_obs = _rollout_observation_for_request(obs, request)
                images = request_obs.get("images") if hasattr(request_obs, "get") else None
                frame_count = (
                    int(images.shape[1])
                    if images is not None
                    else request.expected_frames
                )
                out = runner.predict(request_obs, frame_count=frame_count)
                outputs.append(out)
                obs = step_fn(out["action"])
                if obs is None:
                    break
        finally:
            runner.reset_episode()
        return outputs

    def lazy_joint_video_action(self, backbone_output: BatchFeature, action_input: BatchFeature, latent_video: torch.Tensor | None = None) -> BatchFeature:
        start_time = time.perf_counter()

        # Tracking time taken on GPU for various operations.
        start_text_encoder_event = torch.cuda.Event(enable_timing=True)
        end_text_encoder_event = torch.cuda.Event(enable_timing=True)
        start_image_encoder_event = torch.cuda.Event(enable_timing=True)
        end_image_encoder_event = torch.cuda.Event(enable_timing=True)
        start_vae_event = torch.cuda.Event(enable_timing=True)
        end_vae_event = torch.cuda.Event(enable_timing=True)
        start_kv_event = torch.cuda.Event(enable_timing=True)
        end_kv_event = torch.cuda.Event(enable_timing=True)
        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]

        self.set_frozen_modules_to_eval_mode()
        data = action_input 
        
        videos = data["images"]

        embodiment_id = action_input.embodiment_id
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        # Wan 5B: same as training — resize to target resolution so latent matches DiT
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
        elif videos.shape[2] == 1:
            print("videos.shape[2] == 1, reset current_start_frame to 0")
            self.current_start_frame = 0
        elif self.current_start_frame >= self.model.local_attn_size:
            print("current_start_frame >= local_attn_size, reset current_start_frame to 0")
            self.current_start_frame = 0

        if self.ip_rank == 0:
            print("videos shape", videos.shape, self.num_frames)

        start_text_encoder_event.record()

        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [self.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]

        end_text_encoder_event.record()
        
        start_image_encoder_event.record()

        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] == 4 or videos.shape[2] == 9:
            # special case for real-world eval where language is updated
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, ys, image = self.encode_image(
                image, self.num_frames, height, width, need_seed_latent=True
            )
            self.clip_feas = clip_feas.to(dtype=image.dtype)
            self.ys = None if ys is None else ys.to(dtype=image.dtype)
        
        assert self.clip_feas is not None, "clip_feas must be set"

        end_image_encoder_event.record()

        start_vae_event.record()

        if latent_video is not None and self.current_start_frame != 0:
            image = latent_video
            if self.ip_rank == 0:
                print("image shape@@", image.shape)
        elif self.current_start_frame != 0:
            # this is for real world execution
            if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
                print("no further action")
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                # Repeating videos along dim 2.
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
            
                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)
            else: 
                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)
           
            image = self.vae.encode(
                videos,
                tiled=self.tiled,
                tile_size=(self.tile_size_height, self.tile_size_width),
                tile_stride=(self.tile_stride_height, self.tile_stride_width),
            )

        end_vae_event.record()

        noise_obs = self.generate_noise((image.shape[0], image.shape[1], self.num_frame_per_block, image.shape[3], image.shape[4]), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        noise_action = self.generate_noise((image.shape[0], self.action_horizon, self.model.action_dim), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        batch_size, num_channels, num_frames, height, width = noise_obs.shape
        ######### Generate video #########
        # DiT patch_embedding uses stride (1,2,2), so tokens per frame = (H//2)*(W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        frame_seqlen = tokens_per_frame
        seq_len = num_frames * frame_seqlen

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)
        joint_chunk_index = max(
            0,
            (self.current_start_frame - 1) // self.num_frame_per_block,
        )
        emit_wam_inference_trace(
            self,
            task="joint_dreamzero",
            backend="cache",
            chunk_index=joint_chunk_index,
            phase="history_vae",
            step_index=-1,
            sample=(image[:, :1] if self.current_start_frame == 0 else image[:, -self.num_frame_per_block:]),
        )

        if self.current_start_frame == 0:
            # Reinitialize KV cache and crossattn cache for each new sequence.
            self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
                frame_seqlen=frame_seqlen,
            )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
            )

        assert self.kv_cache1 is not None
        assert self.kv_cache_neg is not None
        assert self.crossattn_cache is not None
        assert self.crossattn_cache_neg is not None
        kv_caches = self._get_caches(
            [self.kv_cache1, self.kv_cache_neg],
        )
        crossattn_caches = self._get_caches(
            [self.crossattn_cache, self.crossattn_cache_neg],
        )

        start_kv_event.record()

        if self.current_start_frame == 0:
            timestep = torch.ones([batch_size, 1], device=noise_obs.device, dtype=torch.int64) * 0
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=frame_seqlen,
                y=None if self.ys is None else self.ys[:, :, 0:1],
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=0,
                        update_kv_cache=True,
                    ),
                    route_overrides=getattr(
                        self, "_joint_history_routes", None
                    ),
                )
            self.current_start_frame += 1
            
        timestep = torch.ones([batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64) * 0

        if self.current_start_frame != 1:
            current_ref_latents = image[:, -self.num_frame_per_block:]
            if self.ys is None:
                y = None
            elif self.current_start_frame <= self.ys.shape[2]:
                y = self.ys[:, :, self.current_start_frame - self.num_frame_per_block : self.current_start_frame]
            else:
                y = self.ys[:, :, -self.num_frame_per_block:]
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=seq_len,
                y=y,
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame - self.num_frame_per_block,
                        update_kv_cache=True,
                    ),
                    route_overrides=getattr(
                        self, "_joint_history_routes", None
                    ),
                )

        end_kv_event.record()

        noisy_input = noise_obs
        noisy_input_action = noise_action

        # Step 3.1: Spatial denoising loop

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)

        # Decoupled inference: video sigmas end at video_final_noise instead of 0
        # This rescales the schedule so video still takes all denoising steps, 
        # but ends at a higher noise level (e.g., 1.0 → 0.9 → 0.8 instead of 1.0 → 0.5 → 0.0)
        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            # Rescale video sigmas: map [sigma_max, 0] -> [sigma_max, video_final_noise]
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)
            if self.ip_rank == 0:
                print(f"Decoupled inference: video sigmas {sigma_max:.3f} -> {sample_scheduler.sigmas[-1].item():.3f}")

        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        prev_predictions = [] 
        self.skip_countdown = 0
        dit_compute_steps = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            start_diffusion_events[index].record()

            # Get timesteps from respective schedulers
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]  # Already rescaled if decoupled

            # set current timestep
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * action_timestep

            # check if we need to run the DIT step
            should_run_model = self.should_run_model(
                index, current_timestep, prev_predictions
            ) or index == len(sample_scheduler.timesteps) - 1
            if should_run_model:
                dit_compute_steps += 1
                if self.ys is None:
                    y = None
                elif self.current_start_frame + self.num_frame_per_block <= self.ys.shape[2]:
                    y = self.ys[:, :, self.current_start_frame : self.current_start_frame + self.num_frame_per_block]
                else:
                    y = self.ys[:, :, -self.num_frame_per_block:]
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame,
                        update_kv_cache=False,
                    ),
                    capture_routes=(
                        index == len(sample_scheduler.timesteps) - 1
                    ),
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                if len(predictions) > 1 and self.cfg_scale > 1.0:
                    flow_pred_uncond, _flow_pred_uncond_action = predictions[1]
                    flow_pred = flow_pred_uncond + self.cfg_scale * (
                        flow_pred_cond - flow_pred_uncond
                    )
                else:
                    flow_pred = flow_pred_cond
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                max_cache_size = 2
                if len(prev_predictions) > max_cache_size:
                    prev_predictions.pop(0)

            else:
                assert len(prev_predictions) > 0, "prev_predictions must be set when skipping"
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            end_diffusion_events[index].record()

            # Video: denoising step (uses rescaled schedule if decoupled)
            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]
            
            # Action: always fully denoises with standard schedule (1000->0)
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]
            emit_wam_inference_trace(
                self,
                task="joint_dreamzero",
                backend="cache",
                chunk_index=max(0, (self.current_start_frame - 1) // self.num_frame_per_block),
                phase="joint",
                step_index=index,
                video_prediction=flow_pred.transpose(1, 2),
                action_prediction=flow_pred_cond_action,
                video_sample=noisy_input,
                action_sample=noisy_input_action,
            )

        latents = noisy_input
        latents_action = noisy_input_action
        if self._last_joint_route_capture is not None:
            self._joint_history_routes = self._last_joint_route_capture
        output = latents

        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        # Do torch.cuda.synchronize() to ensure all operations are completed before timing.
        # This isn't expected to affect inference performance since it's at the end of an inference step.
        torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        text_encoder_time = start_text_encoder_event.elapsed_time(end_text_encoder_event) / 1000
        image_encoder_time = start_image_encoder_event.elapsed_time(end_image_encoder_event) / 1000
        vae_time = start_vae_event.elapsed_time(end_vae_event) / 1000
        kv_creation_time = start_kv_event.elapsed_time(end_kv_event) / 1000
        diffusion_times = [s.elapsed_time(e) for s, e in zip(start_diffusion_events, end_diffusion_events)]
        diffusion_time = sum(diffusion_times) / 1000
        scheduler_time = total_time - kv_creation_time - diffusion_time - text_encoder_time - image_encoder_time - vae_time

        if self.ip_rank == 0:
            print(f"Time taken: Total {total_time:.2f} seconds, "
                  f"Text Encoder {text_encoder_time:.2f} seconds, "
                  f"Image Encoder {image_encoder_time:.2f} seconds, "
                  f"VAE {vae_time:.2f} seconds, "
                  f"KV Cache Creation {kv_creation_time:.2f} seconds, "
                  f"Diffusion {diffusion_time:.2f} seconds, "
                  f"DIT Compute Steps {dit_compute_steps} steps, "
                  f"Scheduler {scheduler_time:.2f} seconds")

        return BatchFeature(data={"action_pred": latents_action, "video_pred": output.transpose(1, 2)})
    
    def cache_predict_order1(self, current_timestep, timestep_1, f1, timestep_2, f2):
        h_curr = current_timestep - timestep_1
        h_past = timestep_1 - timestep_2

        v_prime = (f1 - f2) / h_past

        # Prediction 
        damping_factor = 0.25
        flow_pred = f1 + (v_prime * h_curr) * damping_factor
        return flow_pred

    def post_initialize(self):
        # Move models to the cuda device and set the dtype to bfloat16.
        print("Moving models to the cuda device and setting the dtype to bfloat16.")
        self.model.to(device=self._device, dtype=torch.bfloat16)
        self.text_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.image_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.vae.to(device=self._device, dtype=torch.bfloat16)
        import os
        ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
        LOAD_TRT_ENGINE = os.getenv("LOAD_TRT_ENGINE", None)

        # Torch compile the modules. Skip _forward_blocks: Dynamo with fullgraph can fail on
        # shape variation (e.g. x [1,50,C] vs e [1,200,C]); the block aligns e to x at runtime.
        if not ENABLE_TENSORRT:
            print("Torch compiling the TextEncoder, ImageEncoder, and VAE modules (Wan _forward_blocks not compiled).")

            self.text_encoder.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.text_encoder.forward)

            self.image_encoder.model.visual.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.image_encoder.model.visual.forward)

            self.vae.model.encode = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.vae.model.encode)
        
        self.trt_engine = None
        if LOAD_TRT_ENGINE is not None:
            raise ValueError("TensorRT deployment is outside this source release; unset LOAD_TRT_ENGINE")

    def parallelize(self, device_mesh: DeviceMesh) -> None:
        ip_mesh = device_mesh["ip"]
        self.ip_rank = ip_mesh.get_local_rank()
        self.ip_size = ip_mesh.size()
        self.ip_group = ip_mesh.get_group()

        assert self.ip_size == 1 or self.ip_size == 2, "ip_size must be 1 or 2"
        assert self.ip_rank >= 0 and self.ip_rank < self.ip_size, "ip_rank must be in [0, ip_size)"

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
