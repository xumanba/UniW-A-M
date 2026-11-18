"""Runtime UniW(A)M MoE injection for the DreamZero CausalWanModel FFNs.

This module replaces each selected transformer block's
dense FFN (``nn.Sequential(Linear, GELU, Linear)`` inside
``CausalWanAttentionBlock``) with a router + five expert FFNs:

    shared, vgm, vla, wm, idm

Injection contract:

- The base ``CausalWanModel`` is still built as a dense-FFN transformer, and the
  pretrained Wan2.2-5B weights are loaded into that dense FFN FIRST.
- ``inject_wam_moe_into_wan_model`` is then called; each expert is initialized by
  ``deepcopy(original_ffn)`` (so all experts start from the pretrained FFN) and
  the router is zero-initialized. At injection time the MoE is functionally
  equivalent to the original dense FFN.
- Each ``block.ffn`` becomes a ``WAMMoEWrapper``. It is called as ``ffn(x)`` and
  returns a single tensor, matching the dense FFN call site
  ``y = self.ffn(...)`` in ``CausalWanAttentionBlock.forward``.
- The active-expert mask is read from ``transformer._current_wam_expert_mask``
  (set by ``WANPolicyHead`` before every DiT forward). The env-var based
  fallback below is only a debug/override path.

Routing config is primarily driven by the Hydra ``moe`` block on the action head
(``routing_granularity`` / ``routing_mode`` / ``task`` /
``route_temperature`` / ``target_blocks``); the ``WAM_MOE_*`` environment
variables remain as an override for quick debugging.  ``sample_level`` is the
backward-compatible default: one route per physical batch item, shared by every
token in that item's current FFN sequence.  ``token_group_level`` explicitly
enables the newer per-token-group execution plans.
"""

import copy
import os
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from groot.vla.model.dreamzero.modules.wam_task_attn import (
    G_A,
    G_F,
    G_S,
    G_VFUT,
    build_token_group_routing_plan,
)


WAM_EXPERT_NAMES = ("shared", "vgm", "vla", "wm", "idm")
WAM_SPECIFIC_EXPERT_NAMES = WAM_EXPERT_NAMES[1:]
WAM_MOE_GRANULARITIES = ("sample_level", "token_group_level")
WAM_TASK_ALIASES = {
    "ac_wm": "embodied_world_model",
    "embodied_world_model": "embodied_world_model",
    "embodied-world-model": "embodied_world_model",
    "embodied world model": "embodied_world_model",
    "ewm": "embodied_world_model",
    "wm": "embodied_world_model",
    "world_model": "embodied_world_model",
    "world-model": "embodied_world_model",
    "action_only_fast_wam": "action_only_fast_wam",
    "action-only-fast-wam": "action_only_fast_wam",
    "action only fast wam": "action_only_fast_wam",
    "fast_wam": "action_only_fast_wam",
    "fast-wam": "action_only_fast_wam",
    "action_only_gigaworld_policy": "action_only_gigaworld_policy",
    "action-only-gigaworld-policy": "action_only_gigaworld_policy",
    "action only gigaworld policy": "action_only_gigaworld_policy",
    "gigaworld_policy": "action_only_gigaworld_policy",
    "gigaworld-policy": "action_only_gigaworld_policy",
    "idm_lingbot_va": "idm_lingbot_va",
    "idm-lingbot-va": "idm_lingbot_va",
    "idm lingbot-va": "idm_lingbot_va",
    "lingbot_va": "idm_lingbot_va",
    "lingbot-va": "idm_lingbot_va",
    "joint_dreamzero": "joint_dreamzero",
    "joint-dreamzero": "joint_dreamzero",
    "joint dreamzero": "joint_dreamzero",
    "dreamzero": "joint_dreamzero",
}
# Task -> globally available experts. In the default sample-level mode these
# are the experts executed on the complete current sequence. In explicit
# token-group mode each task's stricter plan selects subsets from this set.
WAM_TASK_EXPERTS = {
    ("embodied_world_model", "train", "default"): ("shared", "wm"),
    ("embodied_world_model", "infer", "default"): ("shared", "wm"),
    ("action_only_fast_wam", "train", "default"): ("shared", "vla", "vgm"),
    # Token-group mode prevents VGM from seeing action tokens; sample-level mode
    # intentionally executes all three experts on the complete current sample.
    ("action_only_fast_wam", "infer", "default"): ("shared", "vla", "vgm"),
    ("action_only_gigaworld_policy", "train", "default"): ("shared", "vla", "wm"),
    ("action_only_gigaworld_policy", "infer", "default"): ("shared", "vla", "wm"),
    ("idm_lingbot_va", "train", "default"): ("shared", "vgm", "idm"),
    # Both specialists are globally eligible at inference. Token-group mode
    # applies the stage spec; sample-level mode executes all three experts on
    # every current stage sequence. WAM_MOE_PASS no longer changes execution.
    ("idm_lingbot_va", "infer", "video"): ("shared", "vgm", "idm"),
    ("idm_lingbot_va", "infer", "action"): ("shared", "vgm", "idm"),
    ("idm_lingbot_va", "infer", "default"): ("shared", "vgm", "idm"),
    ("joint_dreamzero", "train", "default"): ("shared", "vgm", "idm"),
    ("joint_dreamzero", "infer", "default"): ("shared", "vgm", "idm"),
}


def _normalize_task_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower().replace("_", " ")
    normalized = " ".join(normalized.split())
    if not normalized:
        return None
    direct = value.strip().lower()
    return WAM_TASK_ALIASES.get(direct) or WAM_TASK_ALIASES.get(normalized) or direct.replace("-", "_").replace(" ", "_")


def _normalize_phase(value: Optional[str]) -> str:
    if value is None:
        return "train" if torch.is_grad_enabled() else "infer"
    value = value.strip().lower()
    if value in {"inference", "eval", "evaluation", "predict", "test"}:
        return "infer"
    if value in {"training", "train"}:
        return "train"
    return value or "infer"


def _routing_mode() -> str:
    value = os.environ.get("WAM_MOE_ROUTING_MODE", "global_task").strip().lower()
    if value in {"global", "global_task", "task"}:
        return "global_task"
    if value in {"per_sample", "sample", "mixed"}:
        return "per_sample"
    raise ValueError(f"Unsupported WAM_MOE_ROUTING_MODE={value!r}. Use global_task or per_sample.")


def normalize_moe_routing_granularity(value: str) -> str:
    """Normalize one already-selected routing granularity without env lookup."""
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sample": "sample_level",
        "sample_level": "sample_level",
        "global_sample": "sample_level",
        "token_group": "token_group_level",
        "group": "token_group_level",
        "token_group_level": "token_group_level",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported WAM_MOE_GRANULARITY={value!r}. "
            f"Use one of {WAM_MOE_GRANULARITIES}."
        )
    return resolved


def resolve_moe_routing_granularity(config_value: Optional[str] = None) -> str:
    """Resolve MoE routing granularity, with the environment taking priority.

    Checkpoints/configs created before this field existed intentionally resolve
    to ``sample_level`` so the original WAM/WorldScape behavior remains the
    compatibility baseline. Callers that already resolved this choice must use
    ``normalize_moe_routing_granularity`` so a stale environment cannot change
    the model decision during injection.
    """

    value = os.environ.get("WAM_MOE_GRANULARITY")
    if value is None:
        value = config_value or "sample_level"
    return normalize_moe_routing_granularity(value)


def _expert_names_to_indices(expert_names: Iterable[str]) -> List[int]:
    name_to_idx = {name: idx for idx, name in enumerate(WAM_EXPERT_NAMES)}
    active = {0}
    for name in expert_names:
        item = str(name).strip().lower()
        if not item:
            continue
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < len(WAM_EXPERT_NAMES):
                active.add(idx)
            continue
        if item in name_to_idx:
            active.add(name_to_idx[item])
    return sorted(active)


def _task_active_experts() -> Optional[List[int]]:
    task = _normalize_task_name(os.environ.get("WAM_MOE_TASK"))
    if task is None:
        return None
    phase = _normalize_phase(os.environ.get("WAM_MOE_PHASE"))
    wam_pass = os.environ.get("WAM_MOE_PASS", "default").strip().lower() or "default"
    experts = (
        WAM_TASK_EXPERTS.get((task, phase, wam_pass))
        or WAM_TASK_EXPERTS.get((task, phase, "default"))
        or WAM_TASK_EXPERTS.get((task, "train", "default"))
    )
    if experts is None:
        raise ValueError(
            f"Unsupported WAM_MOE_TASK={os.environ.get('WAM_MOE_TASK')!r}, "
            f"WAM_MOE_PHASE={phase!r}, WAM_MOE_PASS={wam_pass!r}."
        )
    return _expert_names_to_indices(experts)


def active_expert_indices_for_task(
    task_name: str,
    phase: Optional[str] = None,
    wam_pass: Optional[str] = None,
) -> List[int]:
    """Return active expert indices for one WAM task name."""

    task = _normalize_task_name(task_name)
    if task is None:
        return [0]
    phase_value = _normalize_phase(phase or os.environ.get("WAM_MOE_PHASE"))
    pass_value = (wam_pass or os.environ.get("WAM_MOE_PASS", "default")).strip().lower() or "default"
    experts = (
        WAM_TASK_EXPERTS.get((task, phase_value, pass_value))
        or WAM_TASK_EXPERTS.get((task, phase_value, "default"))
        or WAM_TASK_EXPERTS.get((task, "train", "default"))
    )
    if experts is None:
        raise ValueError(f"Unsupported WAM task for per-sample routing: {task_name!r}")
    return _expert_names_to_indices(experts)


def build_wam_expert_mask(
    task_names,
    *,
    phase: Optional[str] = None,
    wam_pass: Optional[str] = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Build a [B, 5] per-sample expert mask from task names."""

    if isinstance(task_names, str):
        task_names = [task_names]
    task_names = list(task_names)
    mask = torch.zeros(len(task_names), len(WAM_EXPERT_NAMES), device=device, dtype=dtype or torch.float32)
    for row, task_name in enumerate(task_names):
        for idx in active_expert_indices_for_task(task_name, phase=phase, wam_pass=wam_pass):
            mask[row, idx] = 1
    mask[:, 0] = 1
    return mask


def _parse_default_active_experts() -> List[int]:
    """Resolve active experts from WAM task mapping or a manual override.

    ``WAM_MOE_DEFAULT_ACTIVE_EXPERTS`` is an explicit debug override. If it is
    not set, ``WAM_MOE_TASK`` / ``WAM_MOE_PHASE`` / ``WAM_MOE_PASS`` implement
    the hard-router mapping requested for WAM tasks.
    """

    manual_override = os.environ.get("WAM_MOE_DEFAULT_ACTIVE_EXPERTS", "").strip()
    if not manual_override:
        task_experts = _task_active_experts()
        if task_experts is not None:
            return task_experts

    value = manual_override or "shared"
    value = value.strip().lower()
    if value in {"all", "*"}:
        return list(range(len(WAM_EXPERT_NAMES)))

    return _expert_names_to_indices(value.replace(";", ",").split(","))


class WAMControlMoE(nn.Module):
    """Five-expert WAM MoE that replaces one dense Wan FFN."""

    def __init__(self, original_ffn: nn.Module, route_temperature: float = 1.0):
        super().__init__()
        self.shared_expert = copy.deepcopy(original_ffn)
        self.vgm_expert = copy.deepcopy(original_ffn)
        self.vla_expert = copy.deepcopy(original_ffn)
        self.wm_expert = copy.deepcopy(original_ffn)
        self.idm_expert = copy.deepcopy(original_ffn)
        self.expert_names = WAM_EXPERT_NAMES

        hidden_dim = None
        ref_dtype = torch.bfloat16
        for module in original_ffn.modules():
            if isinstance(module, nn.Linear):
                hidden_dim = module.in_features
                ref_dtype = module.weight.dtype
                break
        if hidden_dim is None:
            raise ValueError("Failed to infer hidden_dim from original FFN.")

        self.router = nn.Linear(hidden_dim, len(WAM_EXPERT_NAMES), bias=False, dtype=ref_dtype)
        nn.init.zeros_(self.router.weight)
        self.route_temperature = max(float(route_temperature), 1e-6)

    def _build_allowed_expert_mask(
        self,
        batch_size: int,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if _routing_mode() == "per_sample" and wam_expert_mask is None:
            raise ValueError(
                "WAM_MOE_ROUTING_MODE=per_sample requires transformer._current_wam_expert_mask. "
                "WANPolicyHead must build a [B, 5] mask from each sample's WAM task, "
                "or switch to WAM_MOE_ROUTING_MODE=global_task."
            )
        if wam_expert_mask is not None:
            allowed_mask = wam_expert_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if allowed_mask.ndim == 1:
                if allowed_mask.numel() == len(WAM_EXPERT_NAMES):
                    allowed_mask = allowed_mask.unsqueeze(0).expand(batch_size, -1)
                elif allowed_mask.numel() == batch_size:
                    expanded = hidden_states.new_zeros(batch_size, len(WAM_EXPERT_NAMES))
                    expanded[:, 0] = 1
                    # A one-dimensional per-sample mask means "enable shared
                    # only when zero; enable all WAM experts when positive".
                    expanded[:, 1:] = allowed_mask.view(-1, 1)
                    allowed_mask = expanded
                else:
                    raise ValueError(
                        f"Unsupported wam_expert_mask shape {tuple(wam_expert_mask.shape)} "
                        f"for batch_size={batch_size}."
                    )
            elif allowed_mask.ndim == 2:
                if allowed_mask.size(0) == 1:
                    allowed_mask = allowed_mask.expand(batch_size, -1)
                if allowed_mask.shape != (batch_size, len(WAM_EXPERT_NAMES)):
                    raise ValueError(
                        f"Expected wam_expert_mask shape {(batch_size, len(WAM_EXPERT_NAMES))}, "
                        f"got {tuple(allowed_mask.shape)}."
                    )
            else:
                raise ValueError(f"Unsupported wam_expert_mask ndim={allowed_mask.ndim}.")
        else:
            allowed_mask = hidden_states.new_zeros(batch_size, len(WAM_EXPERT_NAMES))
            default_active = _parse_default_active_experts()
            allowed_mask[:, default_active] = 1

        # The shared expert is always available as the stable fallback.
        # ``expand`` returns a zero-stride view; clone before the in-place write
        # so caller-provided singleton masks are supported as documented.
        allowed_mask = allowed_mask.clone()
        allowed_mask[:, 0] = 1
        return (allowed_mask > 0).to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor] = None,
        router_extra: Optional[torch.Tensor] = None,
        token_group_routing: Optional[Dict[str, Any]] = None,
        sample_level_routing: Optional[Dict[str, Any]] = None,
        routing_override: Optional[torch.Tensor] = None,
    ):
        if token_group_routing is not None:
            return self._forward_token_groups(
                hidden_states, wam_expert_mask, router_extra, token_group_routing
            )

        batch_size = hidden_states.size(0)
        allowed_mask = self._build_allowed_expert_mask(batch_size, hidden_states, wam_expert_mask)

        if router_extra is not None:
            raise NotImplementedError("router_extra is not supported by the UniW(A)M MoE router.")

        if routing_override is not None:
            routing_weights = routing_override.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            if routing_weights.shape != allowed_mask.shape:
                raise ValueError(
                    "Sample-level routing override must have shape "
                    f"{tuple(allowed_mask.shape)}, got {tuple(routing_weights.shape)}"
                )
            if torch.any(routing_weights < 0):
                raise ValueError("Sample-level routing override cannot contain negative weights")
            routing_weights = routing_weights * allowed_mask
            normalizer = routing_weights.sum(dim=-1, keepdim=True)
            if torch.any(normalizer <= 0):
                raise ValueError("Sample-level routing override has no allowed expert mass")
            routing_weights = routing_weights / normalizer
        else:
            routing_weights = self._sample_level_route(
                hidden_states, allowed_mask, sample_level_routing
            )

        expert_modules = [
            self.shared_expert,
            self.vgm_expert,
            self.vla_expert,
            self.wm_expert,
            self.idm_expert,
        ]

        output = hidden_states.new_zeros(hidden_states.shape)
        for expert_idx, expert in enumerate(expert_modules):
            sample_indices = torch.nonzero(allowed_mask[:, expert_idx] > 0, as_tuple=True)[0]
            if sample_indices.numel() == 0:
                continue
            expert_output = expert(hidden_states.index_select(0, sample_indices))
            expert_output = expert_output * routing_weights.index_select(0, sample_indices)[:, expert_idx].view(-1, 1, 1)
            output.index_add_(
                0, sample_indices, expert_output.to(dtype=output.dtype)
            )

        return output, routing_weights

    def _masked_sample_route(
        self,
        pooled_hidden_states: torch.Tensor,
        allowed_mask: torch.Tensor,
    ) -> torch.Tensor:
        router_logits = self.router(
            pooled_hidden_states.to(self.router.weight.dtype)
        ).float()
        router_logits = router_logits.masked_fill(
            allowed_mask <= 0, torch.finfo(router_logits.dtype).min
        )
        return torch.softmax(
            router_logits / self.route_temperature, dim=-1
        ).to(pooled_hidden_states.dtype)

    @staticmethod
    def _concat_span_indices(
        spans,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        parts = []
        for span in spans:
            start, end = int(span[0]), int(span[1])
            if start < 0 or end <= start or end > seq_len:
                raise ValueError(
                    f"Invalid sample-level router span {span!r} for seq_len={seq_len}"
                )
            parts.append(torch.arange(start, end, device=device))
        if not parts:
            raise ValueError("Sample-level target router requires at least one token span")
        return torch.cat(parts)

    def _sample_level_route(
        self,
        hidden_states: torch.Tensor,
        allowed_mask: torch.Tensor,
        spec: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Return one route per physical sample from noisy target tokens only.

        Joint and IDM first route every complete chunk independently, then
        average the masked-softmax probabilities.  The resulting single route
        still mixes every allowed expert over the complete physical sequence.
        """
        if spec is None:
            return self._masked_sample_route(
                hidden_states.mean(dim=1), allowed_mask
            )

        task = str(spec.get("task", "")).strip().lower()
        stage = str(spec.get("stage", "")).strip().lower()
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        route_spans = []

        if stage == "full_window_history":
            if task in {
                "action_only_fast_wam",
                "action_only_gigaworld_policy",
            }:
                route_spans.append((tuple(spec["action_range"]),))
            elif task == "world_model":
                route_spans.append(tuple(
                    tuple(chunk["video_range"])
                    for chunk in spec.get("chunks", ())
                ))
            else:
                raise ValueError(
                    f"Unsupported full-window sample route task={task!r}"
                )
        elif task in {
            "action_only_fast_wam",
            "action_only_gigaworld_policy",
        }:
            # The action-only deployment has no future-video tokens.  Pool only
            # the noisy action target during training as well, then reuse that
            # one sample-level route over the complete training sequence.
            n_video = int(spec["n_video"])
            n_action = int(spec["n_action"])
            route_spans.append(((n_video, n_video + n_action),))
        elif task == "world_model":
            n_video = int(spec["n_video"])
            n_first = int(spec["n_first"])
            route_spans.append(((n_first, n_video),))
        elif task == "joint_dreamzero" and stage in {"train_tf", "denoise"}:
            route_spans.extend(
                (
                    tuple(chunk["video_range"]),
                    tuple(chunk["action_range"]),
                )
                for chunk in spec.get("chunks", ())
            )
        elif task == "idm_lingbot_va" and stage in {
            "train_tf",
            "video_action_denoise",
        }:
            route_spans.extend(
                (
                    tuple(chunk["video_range"]),
                    tuple(chunk["action_range"]),
                )
                for chunk in spec.get("chunks", ())
            )
        else:
            # Clean-only cache writes have no noisy target.  Until a captured
            # target route is supplied as ``routing_override``, retain the
            # original sample-level fallback for bootstrap compatibility.
            return self._masked_sample_route(
                hidden_states.mean(dim=1), allowed_mask
            )

        if not route_spans:
            raise ValueError(
                f"No noisy target chunks found for sample-level task={task} stage={stage}"
            )

        routes = []
        for spans in route_spans:
            indices = self._concat_span_indices(
                spans, seq_len=seq_len, device=device
            )
            pooled = hidden_states.index_select(1, indices).mean(dim=1)
            routes.append(self._masked_sample_route(pooled, allowed_mask))
        return torch.stack(routes, dim=1).mean(dim=1)

    def _route_group(
        self,
        hidden_states: torch.Tensor,
        pool_indices: torch.Tensor,
        allowed_indices: List[int],
    ) -> torch.Tensor:
        """Route one FastWAM token group with the existing router parameters."""
        pooled = hidden_states.index_select(1, pool_indices).mean(dim=1)
        logits = self.router(pooled.to(self.router.weight.dtype)).float()
        mask = torch.full_like(logits, torch.finfo(logits.dtype).min)
        mask[:, allowed_indices] = logits[:, allowed_indices]
        return torch.softmax(mask / self.route_temperature, dim=-1).to(hidden_states.dtype)

    def _route_from_attention_plan(
        self,
        task: str,
        query_group: int,
        hidden_states: torch.Tensor,
        group_indices: Dict[int, torch.Tensor],
    ):
        """Route one group using pooling derived from the attention allow-table."""
        config = build_token_group_routing_plan(task)[query_group]
        expert_indices = [self.expert_names.index(name) for name in config["experts"]]
        if len(expert_indices) == 1:
            weights = hidden_states.new_zeros(hidden_states.shape[0], len(self.expert_names))
            weights[:, expert_indices[0]] = 1
            return weights, expert_indices
        pool_parts = [group_indices[group_id] for group_id in config["pool_groups"]]
        pool_indices = torch.cat([part for part in pool_parts if part.numel() > 0])
        if pool_indices.numel() == 0:
            raise ValueError(f"Empty router pool for task={task} group={query_group}")
        return self._route_group(hidden_states, pool_indices, expert_indices), expert_indices

    def _forward_token_groups(
        self,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor],
        router_extra: Optional[torch.Tensor],
        spec: Dict[str, Any],
    ):
        """Token-group-level MoE for masked single-horizon or Joint layouts.

        Layout is ``[F | Vfut | A | S0]``.  Router pooling and expert execution
        are both group-local, so dropping ``Vfut`` at action-only inference is
        exact for F/A/S0 rather than an approximation of a sample-level mixture.
        """
        if router_extra is not None:
            raise NotImplementedError("router_extra is not supported by FastWAM token-group routing")
        task = str(spec.get("task", ""))
        if str(spec.get("stage", "")).startswith("masked_cache_"):
            return self._forward_masked_cache_token_groups(
                hidden_states, wam_expert_mask, spec
            )
        if str(spec.get("stage", "")) == "full_window_history":
            return self._forward_full_window_history_token_groups(
                hidden_states, wam_expert_mask, spec
            )
        if task == "joint_dreamzero":
            return self._forward_joint_token_groups(hidden_states, wam_expert_mask, spec)
        if task == "idm_lingbot_va":
            return self._forward_idm_token_groups(hidden_states, wam_expert_mask, spec)
        if task == "world_model":
            return self._forward_world_model_token_groups(hidden_states, wam_expert_mask, spec)
        if task == "action_only_gigaworld_policy":
            return self._forward_gigaworld_token_groups(hidden_states, wam_expert_mask, spec)
        if task != "action_only_fast_wam":
            raise ValueError(f"Unsupported token-group routing spec: {spec!r}")

        batch_size, seq_len, _ = hidden_states.shape
        n_video = int(spec["n_video"])
        n_first = int(spec["n_first"])
        n_action = int(spec["n_action"])
        n_state = int(spec["n_state"])
        if n_first <= 0 or n_action <= 0 or n_state != 1:
            raise ValueError(
                "FastWAM requires non-empty F/A and exactly one S0 token; "
                f"got F={n_first}, A={n_action}, S={n_state}"
            )
        if n_video + n_action + n_state != seq_len or n_first > n_video:
            raise ValueError(
                "FastWAM token-group lengths do not match the hidden sequence: "
                f"video={n_video}, first={n_first}, action={n_action}, "
                f"state={n_state}, seq={seq_len}"
            )

        # Validate the task mask when supplied. This catches an old checkpoint or
        # runtime config that accidentally disables a required specialist.
        allowed = self._build_allowed_expert_mask(batch_size, hidden_states, wam_expert_mask)
        required = torch.tensor([0, 1, 2], device=hidden_states.device)
        if not torch.all(allowed.index_select(1, required) > 0):
            raise ValueError("FastWAM token-group routing requires shared, VGM, and VLA experts")

        device = hidden_states.device
        f_idx = torch.arange(0, n_first, device=device)
        v_idx = torch.arange(n_first, n_video, device=device)
        a_idx = torch.arange(n_video, n_video + n_action, device=device)
        s_idx = torch.arange(n_video + n_action, seq_len, device=device)
        group_indices = {G_F: f_idx, G_VFUT: v_idx, G_A: a_idx, G_S: s_idx}

        # Pooling ranges and eligible experts come from the attention module's
        # single routing plan; neither F nor A can accidentally observe Vfut.
        f_weights, _ = self._route_from_attention_plan(
            task, G_F, hidden_states, group_indices
        )
        a_weights, _ = self._route_from_attention_plan(
            task, G_A, hidden_states, group_indices
        )
        v_weights = None
        if v_idx.numel() > 0:
            v_weights, _ = self._route_from_attention_plan(
                task, G_VFUT, hidden_states, group_indices
            )
        s_weights, _ = self._route_from_attention_plan(
            task, G_S, hidden_states, group_indices
        )

        output = hidden_states.new_zeros(hidden_states.shape)
        shared_out = self.shared_expert(hidden_states)

        def add_weighted(indices: torch.Tensor, values: torch.Tensor, weights: torch.Tensor, expert_idx: int):
            if indices.numel() == 0:
                return
            contribution = values * weights[:, expert_idx].view(-1, 1, 1)
            output.index_add_(
                1, indices, contribution.to(dtype=output.dtype)
            )

        add_weighted(f_idx, shared_out.index_select(1, f_idx), f_weights, 0)
        add_weighted(a_idx, shared_out.index_select(1, a_idx), a_weights, 0)
        if v_weights is not None:
            add_weighted(v_idx, shared_out.index_select(1, v_idx), v_weights, 0)
        # S0 is shared-only with fixed weight one in the centralized plan.
        add_weighted(s_idx, shared_out.index_select(1, s_idx), s_weights, 0)

        fv_idx = torch.cat([f_idx, v_idx])
        vgm_out = self.vgm_expert(hidden_states.index_select(1, fv_idx))
        add_weighted(f_idx, vgm_out[:, : f_idx.numel()], f_weights, 1)
        if v_weights is not None:
            add_weighted(v_idx, vgm_out[:, f_idx.numel() :], v_weights, 1)
        vla_out = self.vla_expert(hidden_states.index_select(1, a_idx))
        add_weighted(a_idx, vla_out, a_weights, 2)

        routing = {"F": f_weights, "Vfut": v_weights, "A": a_weights}
        return output, routing

    def _forward_full_window_history_token_groups(
        self,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor],
        spec: Dict[str, Any],
    ):
        """Token-local experts for ``[clean_V | noisy_V | A | S]``."""
        task = str(spec.get("task", ""))
        if task not in {
            "world_model",
            "action_only_fast_wam",
            "action_only_gigaworld_policy",
        }:
            raise ValueError(f"Unsupported full-window history task: {task!r}")
        batch_size, seq_len, _ = hidden_states.shape
        if int(spec.get("seq_len", -1)) != seq_len:
            raise ValueError(
                "Full-window history token-group lengths do not match: "
                f"spec={spec.get('seq_len')} hidden={seq_len}"
            )

        required_names = (
            {"shared", "wm"}
            if task == "world_model"
            else {"shared", "vla", "vgm" if task == "action_only_fast_wam" else "wm"}
        )
        allowed = self._build_allowed_expert_mask(
            batch_size, hidden_states, wam_expert_mask
        )
        required = torch.tensor(
            [self.expert_names.index(name) for name in sorted(required_names)],
            device=hidden_states.device,
        )
        if not torch.all(allowed.index_select(1, required) > 0):
            raise ValueError(
                f"Full-window {task} requires experts {sorted(required_names)}"
            )

        def indices(span):
            start, end = int(span[0]), int(span[1])
            if start < 0 or end < start or end > seq_len:
                raise ValueError(f"Invalid full-window token span {span!r}")
            return torch.arange(start, end, device=hidden_states.device)

        output = hidden_states.new_zeros(hidden_states.shape)
        shared_out = self.shared_expert(hidden_states)

        def add(idx, values, weights=None, expert_idx=0):
            if idx.numel() == 0:
                return
            if weights is not None:
                values = values * weights[:, expert_idx].view(-1, 1, 1)
            output.index_add_(1, idx, values.to(dtype=output.dtype))

        clean_idx = indices(spec["clean_range"])
        state_idx = indices(spec["state_range"])
        add(clean_idx, shared_out.index_select(1, clean_idx))
        add(state_idx, shared_out.index_select(1, state_idx))

        seed_idx = indices(spec["noisy_seed_range"])
        vgm_groups = []
        wm_groups = []
        vla_groups = []
        routing = {
            "stage": "full_window_history",
            "seed": None,
            "video": [],
            "action": [],
        }
        if task == "action_only_fast_wam":
            experts = [
                self.expert_names.index("shared"),
                self.expert_names.index("vgm"),
            ]
            seed_weights = self._route_group(hidden_states, seed_idx, experts)
            add(seed_idx, shared_out.index_select(1, seed_idx), seed_weights, 0)
            vgm_groups.append((seed_idx, seed_weights))
            routing["seed"] = seed_weights
        else:
            add(seed_idx, shared_out.index_select(1, seed_idx))

        action_all = indices(spec["action_range"])
        if task == "world_model":
            add(action_all, shared_out.index_select(1, action_all))

        for chunk in spec.get("chunks", ()):
            video_idx = indices(chunk["video_range"])
            action_idx = indices(chunk["action_range"])
            video_expert = "vgm" if task == "action_only_fast_wam" else "wm"
            video_experts = [
                self.expert_names.index("shared"),
                self.expert_names.index(video_expert),
            ]
            video_weights = self._route_group(
                hidden_states, video_idx, video_experts
            )
            add(video_idx, shared_out.index_select(1, video_idx), video_weights, 0)
            (vgm_groups if video_expert == "vgm" else wm_groups).append(
                (video_idx, video_weights)
            )
            routing["video"].append(video_weights)

            if task != "world_model":
                action_experts = [
                    self.expert_names.index("shared"),
                    self.expert_names.index("vla"),
                ]
                action_weights = self._route_group(
                    hidden_states, action_idx, action_experts
                )
                add(
                    action_idx,
                    shared_out.index_select(1, action_idx),
                    action_weights,
                    0,
                )
                vla_groups.append((action_idx, action_weights))
                routing["action"].append(action_weights)

        def run_specialist(groups, expert_name):
            if not groups:
                return
            expert_idx = self.expert_names.index(expert_name)
            expert = getattr(self, f"{expert_name}_expert")
            all_idx = torch.cat([idx for idx, _ in groups])
            values = expert(hidden_states.index_select(1, all_idx))
            cursor = 0
            for idx, weights in groups:
                width = idx.numel()
                add(idx, values[:, cursor:cursor + width], weights, expert_idx)
                cursor += width

        run_specialist(vgm_groups, "vgm")
        run_specialist(wm_groups, "wm")
        run_specialist(vla_groups, "vla")
        return output, routing

    def _forward_masked_cache_token_groups(
        self,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor],
        spec: Dict[str, Any],
    ):
        """Route a variable-length cached masked-task query by token span."""
        task = str(spec.get("task", ""))
        stage = str(spec.get("stage", ""))
        if task not in {
            "world_model",
            "action_only_fast_wam",
            "action_only_gigaworld_policy",
        }:
            raise ValueError(f"Unsupported masked-cache task: {task!r}")
        batch_size, seq_len, _ = hidden_states.shape
        if int(spec.get("seq_len", -1)) != seq_len:
            raise ValueError(
                "Masked-cache token-group length mismatch: "
                f"spec={spec.get('seq_len')} hidden={seq_len} stage={stage}"
            )

        target_specs = {
            "seed": (tuple(spec.get("seed_targets", ())), spec.get("seed_expert")),
            "video": (tuple(spec.get("video_targets", ())), spec.get("video_expert")),
            "action": (tuple(spec.get("action_targets", ())), spec.get("action_expert")),
        }
        required_names = {"shared"}
        for spans, expert_name in target_specs.values():
            if spans and expert_name:
                required_names.add(str(expert_name))
        allowed = self._build_allowed_expert_mask(
            batch_size, hidden_states, wam_expert_mask
        )
        required = torch.tensor(
            [self.expert_names.index(name) for name in sorted(required_names)],
            device=hidden_states.device,
        )
        if not torch.all(allowed.index_select(1, required) > 0):
            raise ValueError(
                f"Masked-cache {task} {stage} requires {sorted(required_names)}"
            )

        def indices(span):
            start, end = int(span[0]), int(span[1])
            if start < 0 or end < start or end > seq_len:
                raise ValueError(
                    f"Invalid masked-cache span {span!r} for seq_len={seq_len}"
                )
            return torch.arange(start, end, device=hidden_states.device)

        output = hidden_states.new_zeros(hidden_states.shape)
        shared_out = self.shared_expert(hidden_states)
        coverage = torch.zeros(
            seq_len, dtype=torch.int32, device=hidden_states.device
        )

        def add(token_idx, values, weights=None, expert_idx=0):
            if token_idx.numel() == 0:
                return
            if weights is not None:
                values = values * weights[:, expert_idx].view(-1, 1, 1)
            output.index_add_(
                1, token_idx, values.to(dtype=output.dtype)
            )

        for span in spec.get("shared_ranges", ()):
            token_idx = indices(span)
            coverage.index_add_(
                0, token_idx, torch.ones_like(token_idx, dtype=coverage.dtype)
            )
            add(token_idx, shared_out.index_select(1, token_idx))

        specialist_groups = {"vgm": [], "vla": [], "wm": []}
        routing = {"stage": stage, "seed": [], "video": [], "action": []}
        for group_name, (spans, expert_name) in target_specs.items():
            for span in spans:
                token_idx = indices(span)
                coverage.index_add_(
                    0,
                    token_idx,
                    torch.ones_like(token_idx, dtype=coverage.dtype),
                )
                if not expert_name:
                    add(token_idx, shared_out.index_select(1, token_idx))
                    continue
                expert_name = str(expert_name)
                expert_indices = [
                    self.expert_names.index("shared"),
                    self.expert_names.index(expert_name),
                ]
                weights = self._route_group(
                    hidden_states, token_idx, expert_indices
                )
                add(
                    token_idx,
                    shared_out.index_select(1, token_idx),
                    weights,
                    0,
                )
                specialist_groups[expert_name].append((token_idx, weights))
                routing[group_name].append(weights)

        if not torch.all(coverage == 1):
            raise ValueError(
                "Masked-cache routing spans must cover every token exactly once; "
                f"coverage={coverage.tolist()}"
            )

        for expert_name, groups in specialist_groups.items():
            if not groups:
                continue
            expert_idx = self.expert_names.index(expert_name)
            expert = getattr(self, f"{expert_name}_expert")
            all_idx = torch.cat([token_idx for token_idx, _ in groups])
            values = expert(hidden_states.index_select(1, all_idx))
            cursor = 0
            for token_idx, weights in groups:
                width = token_idx.numel()
                add(
                    token_idx,
                    values[:, cursor:cursor + width],
                    weights,
                    expert_idx,
                )
                cursor += width
        return output, routing

    def _forward_idm_token_groups(
        self,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor],
        spec: Dict[str, Any],
    ):
        """Target-only specialists for LingBot-style IDM teacher forcing/AR.

        Clean seed/history/state tokens are shared-only.  Noisy video targets
        route independently to shared+VGM and noisy action targets route to
        shared+IDM.  Cached stages contain only current post-attention queries,
        so their router inputs exactly match the corresponding training group.
        """
        batch_size, seq_len, _ = hidden_states.shape
        stage = str(spec.get("stage", ""))
        if int(spec.get("seq_len", -1)) != seq_len:
            raise ValueError(
                f"IDM token-group sequence mismatch: spec={spec.get('seq_len')} "
                f"hidden={seq_len} stage={stage}"
            )
        valid_stages = {
            "train_tf",
            "cache_write_seed",
            "cache_write_state",
            "cache_write_video",
            "cache_write_action",
            "video_denoise",
            "action_denoise",
            "video_action_denoise",
        }
        if stage not in valid_stages:
            raise ValueError(f"Unsupported IDM token-group stage: {stage!r}")

        required_names = {"shared"}
        if spec.get("video_targets"):
            required_names.add("vgm")
        if spec.get("action_targets"):
            required_names.add("idm")
        allowed = self._build_allowed_expert_mask(
            batch_size, hidden_states, wam_expert_mask
        )
        required = torch.tensor(
            [self.expert_names.index(name) for name in sorted(required_names)],
            device=hidden_states.device,
        )
        if not torch.all(allowed.index_select(1, required) > 0):
            raise ValueError(
                f"IDM {stage} token-group routing requires {sorted(required_names)}"
            )

        device = hidden_states.device

        def indices(span):
            start, end = int(span[0]), int(span[1])
            if start < 0 or end < start or end > seq_len:
                raise ValueError(f"Invalid IDM span {span!r} for seq_len={seq_len}")
            return torch.arange(start, end, device=device)

        output = hidden_states.new_zeros(hidden_states.shape)
        shared_out = self.shared_expert(hidden_states)

        def add(token_idx, values, weights=None, expert_idx=0):
            if token_idx.numel() == 0:
                return
            if weights is not None:
                values = values * weights[:, expert_idx].view(-1, 1, 1)
            # Autocast may return BF16 expert values even when a diagnostic
            # model keeps FP32 hidden states/buffers.  Production is normally
            # all-BF16, but scatter must remain dtype-safe in either setup.
            output.index_add_(1, token_idx, values.to(dtype=output.dtype))

        for span in spec.get("shared_ranges", ()):
            token_idx = indices(span)
            add(token_idx, shared_out.index_select(1, token_idx))

        video_groups = []
        action_groups = []
        routing = {"stage": stage, "video": [], "action": []}

        for span in spec.get("video_targets", ()):
            token_idx = indices(span)
            weights = self._route_group(
                hidden_states,
                token_idx,
                [self.expert_names.index("shared"), self.expert_names.index("vgm")],
            )
            add(token_idx, shared_out.index_select(1, token_idx), weights, 0)
            video_groups.append((token_idx, weights))
            routing["video"].append(weights)

        for span in spec.get("action_targets", ()):
            token_idx = indices(span)
            weights = self._route_group(
                hidden_states,
                token_idx,
                [self.expert_names.index("shared"), self.expert_names.index("idm")],
            )
            add(token_idx, shared_out.index_select(1, token_idx), weights, 0)
            action_groups.append((token_idx, weights))
            routing["action"].append(weights)

        def run_specialist(groups, expert_name):
            if not groups:
                return
            expert_idx = self.expert_names.index(expert_name)
            expert = getattr(self, f"{expert_name}_expert")
            all_indices = torch.cat([token_idx for token_idx, _ in groups])
            values = expert(hidden_states.index_select(1, all_indices))
            cursor = 0
            for token_idx, weights in groups:
                width = token_idx.numel()
                add(
                    token_idx,
                    values[:, cursor:cursor + width],
                    weights,
                    expert_idx,
                )
                cursor += width

        run_specialist(video_groups, "vgm")
        run_specialist(action_groups, "idm")
        return output, routing

    def _forward_gigaworld_token_groups(
        self,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor],
        spec: Dict[str, Any],
    ):
        """GigaWorld target specialists over ``[F | Vfut | A | S0]``.

        F/S0 are clean shared-only conditions. A uses shared+VLA and cannot
        attend Vfut. Vfut uses shared+WM and may attend A. When action-only
        inference drops Vfut, neither the WM router nor the WM expert executes.
        """
        task = "action_only_gigaworld_policy"
        batch_size, seq_len, _ = hidden_states.shape
        n_video = int(spec["n_video"])
        n_first = int(spec["n_first"])
        n_action = int(spec["n_action"])
        n_state = int(spec["n_state"])
        if n_first <= 0 or n_action <= 0 or n_state != 1:
            raise ValueError(
                "GigaWorld requires non-empty F/A and exactly one S0 token; "
                f"got F={n_first}, A={n_action}, S={n_state}"
            )
        if n_video + n_action + n_state != seq_len or n_first > n_video:
            raise ValueError(
                "GigaWorld token-group lengths do not match the hidden sequence: "
                f"video={n_video}, first={n_first}, action={n_action}, "
                f"state={n_state}, seq={seq_len}"
            )

        device = hidden_states.device
        f_idx = torch.arange(0, n_first, device=device)
        v_idx = torch.arange(n_first, n_video, device=device)
        a_idx = torch.arange(n_video, n_video + n_action, device=device)
        s_idx = torch.arange(n_video + n_action, seq_len, device=device)
        group_indices = {G_F: f_idx, G_VFUT: v_idx, G_A: a_idx, G_S: s_idx}

        allowed = self._build_allowed_expert_mask(batch_size, hidden_states, wam_expert_mask)
        required_names = ["shared", "vla"]
        if v_idx.numel() > 0:
            required_names.append("wm")
        required = torch.tensor(
            [self.expert_names.index(name) for name in required_names], device=device
        )
        if not torch.all(allowed.index_select(1, required) > 0):
            raise ValueError(
                f"GigaWorld token-group routing requires {required_names} experts"
            )

        f_weights, _ = self._route_from_attention_plan(task, G_F, hidden_states, group_indices)
        a_weights, _ = self._route_from_attention_plan(task, G_A, hidden_states, group_indices)
        s_weights, _ = self._route_from_attention_plan(task, G_S, hidden_states, group_indices)
        v_weights = None
        if v_idx.numel() > 0:
            v_weights, _ = self._route_from_attention_plan(
                task, G_VFUT, hidden_states, group_indices
            )

        output = hidden_states.new_zeros(hidden_states.shape)
        shared_out = self.shared_expert(hidden_states)

        def add(indices, values, weights, expert_idx):
            if indices.numel() == 0:
                return
            contribution = values * weights[:, expert_idx].view(-1, 1, 1)
            output.index_add_(
                1, indices, contribution.to(dtype=output.dtype)
            )

        add(f_idx, shared_out.index_select(1, f_idx), f_weights, 0)
        add(a_idx, shared_out.index_select(1, a_idx), a_weights, 0)
        add(s_idx, shared_out.index_select(1, s_idx), s_weights, 0)
        if v_weights is not None:
            add(v_idx, shared_out.index_select(1, v_idx), v_weights, 0)

        vla_out = self.vla_expert(hidden_states.index_select(1, a_idx))
        add(a_idx, vla_out, a_weights, self.expert_names.index("vla"))
        if v_weights is not None:
            wm_out = self.wm_expert(hidden_states.index_select(1, v_idx))
            add(v_idx, wm_out, v_weights, self.expert_names.index("wm"))

        return output, {
            "F": f_weights,
            "Vfut": v_weights,
            "A": a_weights,
            "S0": s_weights,
        }

    def _forward_joint_token_groups(
        self,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor],
        spec: Dict,
    ):
        """Token-group MoE for conservative DreamZero Joint layouts.

        The FFN input is already post self-attention and text cross-attention.
        Consequently the current ``V+A+S`` query tokens carry the cached clean
        history in both teacher-forcing training and KV-cache inference; routing
        never needs to replay historical tokens through the FFN.
        """
        batch_size, seq_len, _ = hidden_states.shape
        expected_len = int(spec.get("seq_len", -1))
        stage = str(spec.get("stage", ""))
        if expected_len != seq_len:
            raise ValueError(
                f"Joint token-group lengths do not match hidden sequence: "
                f"spec={expected_len}, hidden={seq_len}, stage={stage}"
            )
        if stage not in {"train_tf", "cache_write", "denoise"}:
            raise ValueError(f"Unsupported Joint token-group stage: {stage!r}")

        allowed = self._build_allowed_expert_mask(batch_size, hidden_states, wam_expert_mask)
        required_names = {"shared"}
        if stage != "cache_write":
            required_names.update({"vgm", "idm"})
        required = torch.tensor(
            [self.expert_names.index(name) for name in sorted(required_names)],
            device=hidden_states.device,
        )
        if not torch.all(allowed.index_select(1, required) > 0):
            raise ValueError(
                f"Joint {stage} token-group routing requires {sorted(required_names)} experts"
            )

        device = hidden_states.device

        def span_indices(span):
            start, end = (int(span[0]), int(span[1]))
            if start < 0 or end < start or end > seq_len:
                raise ValueError(f"Invalid Joint token span {span!r} for seq_len={seq_len}")
            return torch.arange(start, end, device=device)

        def concat_ranges(ranges):
            parts = [span_indices(span) for span in ranges]
            parts = [part for part in parts if part.numel() > 0]
            if not parts:
                raise ValueError(f"Empty Joint router range set for stage={stage}")
            return torch.cat(parts)

        output = hidden_states.new_zeros(hidden_states.shape)
        shared_out = self.shared_expert(hidden_states)

        def add(indices, values, weights=None, expert_idx=0):
            if indices.numel() == 0:
                return
            if weights is not None:
                values = values * weights[:, expert_idx].view(-1, 1, 1)
            # Autocast may keep the post-attention residual in FP32 while the
            # FFN returns BF16. index_add_ requires an exact dtype match.
            output.index_add_(1, indices, values.to(dtype=output.dtype))

        # Clean history and state are shared-only with a fixed weight of one.
        for span in spec.get("shared_ranges", ()):
            indices = span_indices(span)
            add(indices, shared_out.index_select(1, indices))

        if stage == "cache_write":
            return output, {"stage": stage}

        vgm_groups = []
        idm_groups = []
        routing = {"stage": stage, "Fnoisy": None, "video": [], "action": []}

        if stage == "train_tf":
            target_idx = span_indices(spec["fnoisy_range"])
            pool_idx = concat_ranges(spec["fnoisy_router_ranges"])
            expert_indices = [self.expert_names.index(name) for name in spec["fnoisy_experts"]]
            weights = self._route_group(hidden_states, pool_idx, expert_indices)
            add(target_idx, shared_out.index_select(1, target_idx), weights, 0)
            vgm_groups.append((target_idx, weights))
            routing["Fnoisy"] = weights

        for chunk in spec.get("chunks", ()):
            pool_idx = concat_ranges(chunk["router_ranges"])
            video_idx = span_indices(chunk["video_range"])
            action_idx = span_indices(chunk["action_range"])

            video_experts = [
                self.expert_names.index(name) for name in chunk["video_experts"]
            ]
            action_experts = [
                self.expert_names.index(name) for name in chunk["action_experts"]
            ]
            video_weights = self._route_group(hidden_states, pool_idx, video_experts)
            action_weights = self._route_group(hidden_states, pool_idx, action_experts)

            add(video_idx, shared_out.index_select(1, video_idx), video_weights, 0)
            add(action_idx, shared_out.index_select(1, action_idx), action_weights, 0)
            vgm_groups.append((video_idx, video_weights))
            idm_groups.append((action_idx, action_weights))
            routing["video"].append(video_weights)
            routing["action"].append(action_weights)

        def run_specialist(groups, expert_name):
            if not groups:
                return
            expert_idx = self.expert_names.index(expert_name)
            expert = getattr(self, f"{expert_name}_expert")
            all_indices = torch.cat([indices for indices, _ in groups])
            values = expert(hidden_states.index_select(1, all_indices))
            cursor = 0
            for indices, weights in groups:
                width = indices.numel()
                add(indices, values[:, cursor:cursor + width], weights, expert_idx)
                cursor += width

        run_specialist(vgm_groups, "vgm")
        run_specialist(idm_groups, "idm")
        return output, routing

    def _forward_world_model_token_groups(
        self,
        hidden_states: torch.Tensor,
        wam_expert_mask: Optional[torch.Tensor],
        spec: Dict[str, int],
    ):
        """World-model groups use separate shared+WM routes for F/Vfut/Aclean."""
        batch_size, seq_len, _ = hidden_states.shape
        n_video = int(spec["n_video"])
        n_first = int(spec["n_first"])
        n_action = int(spec["n_action"])
        n_state = int(spec["n_state"])
        if n_first <= 0 or n_action <= 0 or n_state != 0:
            raise ValueError(
                "world_model requires non-empty F/A and no state tokens; "
                f"got F={n_first}, A={n_action}, S={n_state}"
            )
        if n_video + n_action != seq_len or n_first > n_video:
            raise ValueError(
                "world_model token-group lengths do not match sequence: "
                f"video={n_video}, first={n_first}, action={n_action}, seq={seq_len}"
            )
        allowed = self._build_allowed_expert_mask(batch_size, hidden_states, wam_expert_mask)
        required = torch.tensor([0, 3], device=hidden_states.device)
        if not torch.all(allowed.index_select(1, required) > 0):
            raise ValueError("world_model token-group routing requires shared and WM experts")

        device = hidden_states.device
        f_idx = torch.arange(0, n_first, device=device)
        v_idx = torch.arange(n_first, n_video, device=device)
        a_idx = torch.arange(n_video, seq_len, device=device)
        group_indices = {G_F: f_idx, G_VFUT: v_idx, G_A: a_idx}
        f_weights, _ = self._route_from_attention_plan(
            "world_model", G_F, hidden_states, group_indices
        )
        a_weights, _ = self._route_from_attention_plan(
            "world_model", G_A, hidden_states, group_indices
        )
        v_weights = None
        if v_idx.numel() > 0:
            v_weights, _ = self._route_from_attention_plan(
                "world_model", G_VFUT, hidden_states, group_indices
            )

        output = hidden_states.new_zeros(hidden_states.shape)
        shared_out = self.shared_expert(hidden_states)

        def add(indices, values, weights, expert_idx):
            if indices.numel() == 0:
                return
            contribution = values * weights[:, expert_idx].view(-1, 1, 1)
            output.index_add_(
                1, indices, contribution.to(dtype=output.dtype)
            )

        add(f_idx, shared_out.index_select(1, f_idx), f_weights, 0)
        add(a_idx, shared_out.index_select(1, a_idx), a_weights, 0)
        if v_weights is not None:
            add(v_idx, shared_out.index_select(1, v_idx), v_weights, 0)

        # WM is a target-token specialist: conditions remain shared-only, while
        # the already-conditioned Vfut hidden states receive shared+WM capacity.
        if v_weights is not None:
            wm_out = self.wm_expert(hidden_states.index_select(1, v_idx))
            add(v_idx, wm_out, v_weights, 3)
        return output, {"F": f_weights, "Vfut": v_weights, "A": a_weights}


class WAMMoEWrapper(nn.Module):
    """Thin block.ffn replacement that reads routing state from the transformer."""

    def __init__(
        self,
        moe_module: WAMControlMoE,
        transformer_model: nn.Module,
        block_index: int = 0,
    ):
        super().__init__()
        self.control_moe = moe_module
        self.block_index = int(block_index)
        # Keep a non-registered reference to avoid recursive module ownership.
        self.transformer_model = [transformer_model]

    def forward(self, hidden_states: torch.Tensor):
        transformer = self.transformer_model[0]
        wam_expert_mask = getattr(transformer, "_current_wam_expert_mask", None)
        router_extra = getattr(transformer, "_current_wam_router_extra", None)
        current_spec = getattr(
            transformer, "_current_wam_token_group_routing", None
        )
        configured_granularity = getattr(
            transformer, "_wam_moe_routing_granularity", None
        )
        if configured_granularity is not None:
            routing_granularity = str(configured_granularity)
            if routing_granularity not in WAM_MOE_GRANULARITIES:
                raise ValueError(
                    "Transformer has invalid MoE routing granularity: "
                    f"{routing_granularity!r}"
                )
        else:
            routing_granularity = resolve_moe_routing_granularity(None)
        token_group_routing = None
        if routing_granularity == "token_group_level":
            token_group_routing = current_spec
        sample_level_routing = None
        routing_override = None
        if routing_granularity == "sample_level":
            sample_level_routing = getattr(
                transformer, "_current_wam_token_group_routing", None
            )
            overrides = getattr(
                transformer, "_wam_sample_route_override", None
            )
            if overrides is not None:
                routing_override = overrides.get(self.block_index)
                if routing_override is None:
                    raise ValueError(
                        "Missing sample-level routing override for "
                        f"MoE block {self.block_index}"
                    )
        moe_out, routing_weights = self.control_moe(
            hidden_states,
            wam_expert_mask=wam_expert_mask,
            router_extra=router_extra,
            token_group_routing=token_group_routing,
            sample_level_routing=sample_level_routing,
            routing_override=routing_override,
        )
        capture = getattr(transformer, "_wam_sample_route_capture", None)
        if routing_granularity == "sample_level" and capture is not None:
            capture[self.block_index] = routing_weights.detach().clone()
        return moe_out


def _normalize_target_block_indices(transformer_model: nn.Module, target_block_indices: Optional[Iterable[int]]):
    block_count = len(transformer_model.blocks)
    if target_block_indices is None:
        return list(range(block_count))

    normalized_indices = []
    for idx in target_block_indices:
        idx = int(idx)
        if idx < 0:
            idx += block_count
        if idx < 0 or idx >= block_count:
            raise IndexError(f"Block index {idx} is out of range for {block_count} blocks.")
        normalized_indices.append(idx)
    return list(dict.fromkeys(normalized_indices))


def inject_wam_moe_into_wan_model(
    transformer_model: nn.Module,
    target_block_indices: Optional[Iterable[int]] = None,
    route_temperature: float = 1.0,
    routing_granularity: Optional[str] = None,
):
    """Replace selected ``CausalWanModel.blocks[i].ffn`` with a WAM MoE wrapper.

    Must be called AFTER the pretrained dense FFN weights are loaded, so that the
    five experts are ``deepcopy`` of the pretrained FFN.
    """

    routing_granularity = (
        resolve_moe_routing_granularity(None)
        if routing_granularity is None
        else normalize_moe_routing_granularity(routing_granularity)
    )
    transformer_model._wam_moe_routing_granularity = routing_granularity
    target_block_indices = _normalize_target_block_indices(transformer_model, target_block_indices)
    ref_param = next(transformer_model.parameters())
    model_dtype = ref_param.dtype
    model_device = ref_param.device

    for idx in target_block_indices:
        block = transformer_model.blocks[idx]
        original_ffn = block.ffn
        # Match both dtype and device of the model so the freshly-created router
        # lands on the same device as the deepcopied experts (robust whether the
        # model is on CPU or GPU at injection time).
        moe_module = WAMControlMoE(original_ffn, route_temperature=route_temperature).to(
            device=model_device, dtype=model_dtype
        )
        block.ffn = WAMMoEWrapper(
            moe_module, transformer_model, block_index=idx
        ).to(device=model_device, dtype=model_dtype)

    print(
        "Injected UniW(A)M MoE into DreamZero CausalWanModel FFNs "
        f"({len(target_block_indices)} blocks, experts={','.join(WAM_EXPERT_NAMES)}, "
        f"routing_granularity={routing_granularity})."
    )
    return transformer_model
