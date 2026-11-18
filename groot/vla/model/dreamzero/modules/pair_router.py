"""Plug-in Adaptive Interface Router (PAIR).

PAIR picks one of the four action-producing UniW(A)M interfaces for every
action chunk at deployment time.  The UniW(A)M backbone stays frozen: the
router is a small MLP that lives outside the model checkpoint and reads
pooled features produced by the frozen backbone itself.

Per chunk, the router consumes five inputs:

``video``
    The chunk's visible video latents (``1 + 2 * chunk_index`` latent frames)
    run through every DiT block with only the shared expert active, then
    mean-pooled over frames and patches.
``clip``
    Mean-pooled CLIP image tokens after the DiT image projection.
``state``
    The current proprioceptive state through the DiT state encoder.
``text``
    Mask-pooled umT5 text tokens after the DiT text projection.
``position``
    A one-hot of the chunk position inside the four-chunk window.

All features are taken from the same request batch that the selected
interface receives for that chunk, so training-time and deployment-time
inputs match.  Features are rounded to float16, the precision used to store
the training set.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from groot.vla.model.dreamzero.modules.wam_moe import WAM_EXPERT_NAMES
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    sinusoidal_embedding_1d,
)

# Interfaces PAIR chooses between; the order defines the router's logits.
PAIR_TASKS = (
    "joint_dreamzero",
    "idm_lingbot_va",
    "action_only_fast_wam",
    "action_only_gigaworld_policy",
)
# Pooled feature keys produced by PAIRFeatureEncoder, in router input order.
PAIR_FEATURE_KEYS = ("video", "clip", "state", "text")
# The chunk-position one-hot is appended as the last router input.
PAIR_CHUNKS_PER_WINDOW = 4
PAIR_HIDDEN_DIM = 3072
PAIR_CHECKPOINT_GLOB = "pair_router_seed*.pt"


class PAIRRouter(nn.Module):
    """Per-input LayerNorm -> concat -> 2-layer MLP -> interface logits."""

    def __init__(
        self,
        input_dims: Sequence[int] = (PAIR_HIDDEN_DIM,) * 4 + (PAIR_CHUNKS_PER_WINDOW,),
        hidden: int = 512,
        dropout: float = 0.2,
        num_tasks: int = len(PAIR_TASKS),
    ):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(int(dim)) for dim in input_dims])
        self.mlp = nn.Sequential(
            nn.Linear(sum(int(dim) for dim in input_dims), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_tasks),
        )

    def forward(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(parts) != len(self.norms):
            raise ValueError(f"router expects {len(self.norms)} inputs, got {len(parts)}")
        z = torch.cat([norm(part.float()) for norm, part in zip(self.norms, parts)], dim=-1)
        return self.mlp(z)


class PAIRRouterEnsemble(nn.Module):
    """Probability-averaged ensemble; ``forward`` returns log mean probabilities."""

    def __init__(self, routers: Sequence[PAIRRouter]):
        super().__init__()
        if not routers:
            raise ValueError("an ensemble needs at least one router")
        dims = _input_dims(routers[0])
        for router in routers[1:]:
            if _input_dims(router) != dims:
                raise ValueError(
                    f"ensemble members disagree on input widths: {dims} vs {_input_dims(router)}"
                )
        self.routers = nn.ModuleList(routers)

    def forward(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        probs = torch.stack(
            [torch.softmax(router(parts), dim=-1) for router in self.routers]
        ).mean(dim=0)
        return torch.log(probs.clamp_min(1e-12))


def _input_dims(router: PAIRRouter) -> tuple[int, ...]:
    return tuple(int(norm.weight.numel()) for norm in router.norms)


def load_pair_router(checkpoint: str | Path, device: str | torch.device = "cpu") -> PAIRRouter:
    """Load one router state dict, inferring per-input widths from it."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    count = sum(1 for key in state if key.startswith("norms.") and key.endswith(".weight"))
    if count == 0:
        raise ValueError(f"{checkpoint} is not a PAIR router state dict")
    dims = tuple(int(state[f"norms.{i}.weight"].numel()) for i in range(count))
    hidden = int(state["mlp.0.weight"].shape[0])
    num_tasks = int(state["mlp.3.weight"].shape[0])
    router = PAIRRouter(input_dims=dims, hidden=hidden, num_tasks=num_tasks)
    router.load_state_dict(state, strict=True)
    return router.to(device).eval()


def resolve_pair_checkpoints(spec: str | Path | Iterable[str | Path]) -> list[Path]:
    """Accept a file, a directory of ``pair_router_seed*.pt`` or a list of files."""
    if isinstance(spec, (str, Path)):
        text = str(spec)
        items = [item.strip() for item in text.split(",") if item.strip()]
    else:
        items = [str(item) for item in spec]
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob(PAIR_CHECKPOINT_GLOB)))
        else:
            paths.append(path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"PAIR router checkpoints not found: {missing}")
    if not paths:
        raise FileNotFoundError(f"no PAIR router checkpoints under {spec}")
    return paths


def load_pair_router_ensemble(
    spec: str | Path | Iterable[str | Path],
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load one router, or a probability-averaged ensemble of several seeds."""
    routers = [load_pair_router(path, device) for path in resolve_pair_checkpoints(spec)]
    if len(routers) == 1:
        return routers[0]
    return PAIRRouterEnsemble(routers).to(device).eval()


def position_one_hot(
    chunk_index: int | torch.Tensor,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    index = torch.as_tensor(chunk_index, device=device, dtype=torch.long).reshape(-1)
    if index.numel() == 1 and batch_size > 1:
        index = index.expand(batch_size)
    if ((index < 0) | (index >= PAIR_CHUNKS_PER_WINDOW)).any():
        raise ValueError(f"chunk_index must be in [0,{PAIR_CHUNKS_PER_WINDOW})")
    return nn.functional.one_hot(index, PAIR_CHUNKS_PER_WINDOW).float()


def router_inputs(
    features: dict[str, torch.Tensor],
    chunk_index: int | torch.Tensor,
    device: str | torch.device | None = None,
) -> list[torch.Tensor]:
    """Assemble router inputs in the training order (features + position)."""
    first = features[PAIR_FEATURE_KEYS[0]]
    device = device or first.device
    parts = [features[key].to(device=device, dtype=torch.float32) for key in PAIR_FEATURE_KEYS]
    parts.append(position_one_hot(chunk_index, first.shape[0], device))
    return parts


def pick_interfaces(
    probs: torch.Tensor,
    chunk_index: int | None = None,
    *,
    exclude: Sequence[str] = (),
    chunk0_allowed: Sequence[str] = (),
    joint_threshold: float = 0.0,
) -> torch.Tensor:
    """Turn ``[B, 4]`` router probabilities into interface indices.

    ``exclude`` removes interfaces from the choice.  ``chunk0_allowed`` names
    excluded interfaces that stay eligible on the first chunk of a window;
    this is useful for IDM-WAM in closed loop, which has no action history on
    chunk 0 but would otherwise consume actions generated by a different
    interface on later chunks.

    ``joint_threshold`` falls back to Joint-WAM unless another interface wins
    with at least that probability: ``0`` keeps the argmax, ``1`` always picks
    Joint-WAM.
    """
    if not 0.0 <= joint_threshold <= 1.0:
        raise ValueError(f"joint_threshold must be in [0,1], got {joint_threshold}")
    unknown = [name for name in (*exclude, *chunk0_allowed) if name not in PAIR_TASKS]
    if unknown:
        raise ValueError(f"unknown interfaces {unknown}; known: {PAIR_TASKS}")
    if chunk_index == 0:
        exclude = tuple(name for name in exclude if name not in chunk0_allowed)
    if len(set(exclude)) >= len(PAIR_TASKS):
        raise ValueError("cannot exclude every interface")
    joint = PAIR_TASKS.index("joint_dreamzero")
    if joint_threshold > 0 and PAIR_TASKS[joint] in exclude:
        raise ValueError("joint_threshold needs joint_dreamzero to stay eligible")
    masked = probs.clone()
    if exclude:
        masked[:, [PAIR_TASKS.index(name) for name in exclude]] = -1.0
    top = masked.argmax(dim=-1)
    top_prob = masked.max(dim=-1).values
    return torch.where(
        (top != joint) & (top_prob < joint_threshold),
        torch.full_like(top, joint),
        top,
    )


@torch.no_grad()
def select_interface(
    router: nn.Module,
    features: dict[str, torch.Tensor],
    chunk_index: int,
    *,
    exclude: Sequence[str] = (),
    chunk0_allowed: Sequence[str] = (),
    joint_threshold: float = 0.0,
) -> tuple[str, torch.Tensor]:
    """One router forward -> (interface name, probability vector) for one chunk."""
    device = next(router.parameters()).device
    probs = torch.softmax(router(router_inputs(features, chunk_index, device)), dim=-1)
    if probs.shape[0] != 1:
        raise ValueError("select_interface routes one request at a time")
    pick = pick_interfaces(
        probs, chunk_index, exclude=exclude, chunk0_allowed=chunk0_allowed,
        joint_threshold=joint_threshold,
    )
    return PAIR_TASKS[int(pick.item())], probs[0].float().cpu()


def _pad_history_images(images: torch.Tensor, target_length: int = 33) -> torch.Tensor:
    if images.ndim != 5 or images.shape[-1] != 3:
        raise ValueError("history images must be [B,T,H,W,3]")
    if not 1 <= images.shape[1] <= target_length:
        raise ValueError(f"history length must be 1..{target_length}")
    if images.shape[1] == target_length:
        return images
    return torch.cat(
        [images, images[:, -1:].expand(-1, target_length - images.shape[1], -1, -1, -1)],
        dim=1,
    )


def _fit_width(value: torch.Tensor, width: int) -> torch.Tensor:
    if value.shape[-1] == width:
        return value
    if value.shape[-1] < width:
        return nn.functional.pad(value, (0, width - value.shape[-1]))
    return value[..., :width]


class PAIRFeatureEncoder:
    """Extract PAIR router features from a frozen UniW(A)M model."""

    def __init__(self, model, probe_experts: Sequence[str] = ("shared",)):
        self.model = model
        self.head = model.action_head
        self.dit = self.head.model
        if int(getattr(self.dit, "dim", -1)) != PAIR_HIDDEN_DIM:
            raise ValueError(
                f"PAIR expects DiT hidden width {PAIR_HIDDEN_DIM}, "
                f"got {getattr(self.dit, 'dim', None)}"
            )
        unknown = [name for name in probe_experts if name not in WAM_EXPERT_NAMES]
        if unknown or not probe_experts:
            raise ValueError(f"probe experts must be a non-empty subset of {WAM_EXPERT_NAMES}")
        self.probe_experts = tuple(probe_experts)

    def _embodiment_id(self, action_input) -> torch.Tensor:
        embodiment_id = action_input.embodiment_id.to(self.head._device)
        if (
            self.dit.state_encoder.layer1.num_categories == 1
            or getattr(self.dit.action_encoder, "num_embodiments", 1) == 1
        ):
            embodiment_id = torch.zeros_like(embodiment_id)
        return embodiment_id

    def _state_feature(self, action_input, embodiment_id: torch.Tensor) -> torch.Tensor:
        state = action_input["state"][:, -1:].detach()
        state = _fit_width(
            state.to(device=self.head._device, dtype=torch.bfloat16),
            int(getattr(self.dit, "max_state_dim", state.shape[-1])),
        )
        return self.dit.state_encoder(state, embodiment_id).float()[:, 0, :]

    def _text_feature(self, prompt: torch.Tensor, action_input) -> torch.Tensor:
        tokens = self.dit.text_embedding(prompt).float()
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
        mask = action_input.get("text_attention_mask")
        if mask is None:
            mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask).to(device=tokens.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        weight = mask.to(tokens.dtype).unsqueeze(-1)
        return (
            (tokens * weight).sum(dim=1)
            / weight.sum(dim=1).clamp_min(torch.finfo(tokens.dtype).tiny)
        ).float()

    def _video_feature(
        self,
        latents: torch.Tensor,
        clip: torch.Tensor | None,
        prompt: torch.Tensor,
    ) -> torch.Tensor:
        """Observation-only pass of the visible latents through all DiT blocks.

        Uses the masked task-attention path with an all-visible mask, so every
        visible frame attends to every other.  Only ``probe_experts`` are active
        in the MoE FFNs; action and state tokens are absent.
        """
        dit = self.dit
        patch_grid = dit.patch_embedding(latents.transpose(1, 2))
        grid = torch.tensor(patch_grid.shape[2:], device=patch_grid.device, dtype=torch.long)
        tokens = patch_grid.flatten(start_dim=2).transpose(1, 2)
        frames = int(latents.shape[1])
        seq = int(tokens.shape[1])
        freqs = dit._create_freqs(grid_size=grid, start_frame=0)
        timestep = torch.zeros(tokens.shape[0], seq, device=tokens.device)
        emb = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep.flatten()).type_as(tokens)
        )
        emb = emb.unflatten(0, (tokens.shape[0], -1))
        e0 = dit.time_projection(emb).unflatten(2, (6, dit.dim))
        context = dit.text_embedding(prompt)
        if clip is not None:
            context = torch.cat([dit.img_emb(clip), context], dim=1)
        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=dit.freqs_action,
            freqs_state=dit.freqs_state,
            action_register_length=None,
            context=context,
            is_tf=False,
            task_attn_mask=torch.ones(seq, seq, dtype=torch.bool, device=tokens.device),
            task_seg=(seq, 0, 0),
            task_rope_mode="masked",
        )
        expert_mask = tokens.new_zeros(tokens.shape[0], len(WAM_EXPERT_NAMES))
        for name in self.probe_experts:
            expert_mask[:, WAM_EXPERT_NAMES.index(name)] = 1

        missing = object()
        saved = {}
        for name, value in (
            ("_current_wam_expert_mask", expert_mask),
            ("_current_wam_task", None),
            ("_current_wam_token_group_routing", None),
            ("_wam_moe_routing_granularity", "sample_level"),
        ):
            saved[name] = getattr(dit, name, missing)
            setattr(dit, name, value)
        autocast = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if tokens.is_cuda
            else nullcontext()
        )
        try:
            with autocast:
                for block in dit.blocks:
                    tokens, cache = block(tokens, **kwargs)
                    if cache is not None:
                        raise RuntimeError("PAIR probe unexpectedly produced a KV cache")
        finally:
            for name, value in saved.items():
                if value is missing:
                    try:
                        delattr(dit, name)
                    except AttributeError:
                        pass
                else:
                    setattr(dit, name, value)
        spatial = int(patch_grid.shape[3] * patch_grid.shape[4])
        return tokens.reshape(tokens.shape[0], frames, spatial, dit.dim).mean(dim=(1, 2)).float()

    @torch.no_grad()
    def __call__(self, batch: dict, chunk_index: int) -> dict[str, torch.Tensor]:
        """Features for one chunk request; returns float16 ``[B, 3072]`` vectors.

        ``batch`` is the request the interface receives for this chunk (as
        built by ``uniwam.inference.episode.BatchBuilder``): a single seed
        frame on chunk 0 and the preceding chunk's frames afterwards.
        """
        if not 0 <= int(chunk_index) < PAIR_CHUNKS_PER_WINDOW:
            raise ValueError(f"chunk_index must be in [0,{PAIR_CHUNKS_PER_WINDOW})")
        visible = 1 + 2 * int(chunk_index)
        full_input = dict(batch)
        full_input["images"] = _pad_history_images(batch["images"])
        _, action_input = self.model.prepare_input(full_input)
        latents, clip, _ys, prompt, *_ = self.head._wam_encode_obs(action_input)
        if latents.shape[1] < visible:
            raise ValueError(f"need {visible} latent frames, got {latents.shape[1]}")
        if clip is None:
            raise ValueError("PAIR requires CLIP image features")
        batch_size = latents.shape[0]
        embodiment_id = self._embodiment_id(action_input)
        clip_tokens = self.dit.img_emb(clip).float()
        if clip_tokens.ndim == 2:
            clip_tokens = clip_tokens[:, None, :]
        features = {
            "video": self._video_feature(latents[:, :visible], clip, prompt),
            "clip": clip_tokens.mean(dim=1),
            "state": self._state_feature(action_input, embodiment_id),
            "text": self._text_feature(prompt, action_input),
        }
        for key, value in features.items():
            if value.shape != (batch_size, PAIR_HIDDEN_DIM):
                raise ValueError(f"PAIR feature {key} has shape {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"PAIR feature {key} contains non-finite values")
        return {key: value.to(torch.float16) for key, value in features.items()}
