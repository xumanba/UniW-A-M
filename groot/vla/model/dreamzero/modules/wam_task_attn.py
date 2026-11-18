"""Per-task self-attention connectivity for UniW(A)M MoE multi-task training.

DreamZero's `joint_dreamzero` task keeps its existing hand-rolled teacher-forcing
attention (in `wan_video_dit_action_casual_chunk.py`). The other WAM tasks each
need a DIFFERENT "who attends whom" rule over the single-stream token sequence.
This module builds, per task, a boolean self-attention mask `[L, L]`
(`True` = attention allowed) that can be fed to `F.scaled_dot_product_attention`.

Token groups (single stream, no clean/noisy duplication for the single-pass tasks):

    F     : first-frame video tokens (the conditioning frame, video[0:frame_seqlen])
    Vfut  : future video tokens (video[frame_seqlen:], the denoised target)
    A     : action tokens
    S     : state tokens

The connectivity per task:

- world_model            [F | Vfut | A(clean)]      (no S)
    F -> {F};  A -> {F, A};  Vfut -> {F, A, Vfut}
    (action is a clean condition; only video is a denoise target)
- action_only_fast_wam   [F | Vfut | A | S]
    F -> {F, S};  Vfut -> {F, Vfut, S};  A -> {F, A, S};  S -> {F, S}
    (Vfut is a pure attention SINK: it reads conditioning but nothing that the
    action can ever reach — directly or transitively — reads Vfut. This makes
    dropping the video branch at inference EXACTLY equivalent for the action,
    matching original FastWAM's `first_frame_causal` (F does not read Vfut;
    FastWAM/src/fastwam/models/wan22/wan_video_dit.py:501) and
    giga-world-policy's mask (transformer_wa_casual.py:1018). Keeping F
    independent of Vfut prevents noisy future-video information from reaching
    the action tokens through the conditioning frame.)
- action_only_gigaworld  [F | Vfut | A | S]
    F -> {F, S};  Vfut -> {F, Vfut, A, S};  A -> {F, A, S};  S -> {F, S}
    (same sink property as fastwam, but video may also attend the noisy action)

`idm_lingbot_va` uses interleaved teacher forcing and is handled separately.

Group index convention: F=0, Vfut=1, A=2, S=3.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from .wam_idm_cache import canonical_idm_frame_ids
except ImportError:  # standalone smoke-test loading
    from groot.vla.model.dreamzero.modules.wam_idm_cache import canonical_idm_frame_ids
# Group ids
G_F = 0      # first-frame video (conditioning frame)
G_VFUT = 1   # future video (denoise target)
G_A = 2      # action
G_S = 3      # state
NUM_GROUPS = 4
GROUP_NAMES = ("F", "Vfut", "A", "S")

# Tasks whose self-attention is expressed as a single per-task boolean mask here
# (single forward pass, no teacher-forcing duplication).
WAM_MASKED_TASKS = ("world_model", "action_only_fast_wam", "action_only_gigaworld_policy")

# alias -> canonical task name for the masked tasks
_TASK_ALIASES = {
    "ac_wm": "world_model",
    "world_model": "world_model",
    "wm": "world_model",
    "embodied_world_model": "world_model",
    "action_only_fast_wam": "action_only_fast_wam",
    "fast_wam": "action_only_fast_wam",
    "action_only_gigaworld_policy": "action_only_gigaworld_policy",
    "gigaworld_policy": "action_only_gigaworld_policy",
}

# Single source of truth for token-group expert eligibility. Router pooling is
# NOT duplicated here: ``build_token_group_routing_plan`` derives it directly
# from the same attention allow-table used to build the boolean attention mask.
_TOKEN_GROUP_EXPERTS = {
    "action_only_fast_wam": {
        G_F: ("shared", "vgm"),       # explicit original-FastWAM adaptation
        G_VFUT: ("shared", "vgm"),
        G_A: ("shared", "vla"),
        G_S: ("shared",),
    },
    "world_model": {
        G_F: ("shared",),
        G_VFUT: ("shared", "wm"),
        G_A: ("shared",),
    },
    "action_only_gigaworld_policy": {
        G_F: ("shared",),
        G_VFUT: ("shared", "wm"),
        G_A: ("shared", "vla"),
        G_S: ("shared",),
    },
}


def canonical_masked_task(task: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    return _TASK_ALIASES.get(str(task).strip().lower().replace("-", "_").replace(" ", "_"))


def _allow_table(task: str) -> torch.Tensor:
    """Return a [NUM_GROUPS, NUM_GROUPS] bool table: allow[q_group, k_group]."""
    allow = torch.zeros(NUM_GROUPS, NUM_GROUPS, dtype=torch.bool)

    def set_pairs(pairs):
        for q, k in pairs:
            allow[q, k] = True

    if task == "world_model":
        # F -> {F}; A -> {F, A}; Vfut -> {F, A, Vfut}
        set_pairs([
            (G_F, G_F),
            (G_A, G_F), (G_A, G_A),
            (G_VFUT, G_F), (G_VFUT, G_A), (G_VFUT, G_VFUT),
        ])
    elif task == "action_only_fast_wam":
        # Vfut = pure sink (F does NOT read Vfut); action attends first-frame +
        # state only; V/A mutually invisible. Dropping Vfut at inference is then
        # exactly equivalent for the action output (FastWAM first_frame_causal /
        # giga sink design).
        set_pairs([
            (G_F, G_F), (G_F, G_S),
            (G_VFUT, G_F), (G_VFUT, G_VFUT), (G_VFUT, G_S),
            (G_A, G_F), (G_A, G_A), (G_A, G_S),
            (G_S, G_F), (G_S, G_S),
        ])
    elif task == "action_only_gigaworld_policy":
        # like fastwam (Vfut = pure sink), but future video may also attend action
        set_pairs([
            (G_F, G_F), (G_F, G_S),
            (G_VFUT, G_F), (G_VFUT, G_VFUT), (G_VFUT, G_A), (G_VFUT, G_S),
            (G_A, G_F), (G_A, G_A), (G_A, G_S),
            (G_S, G_F), (G_S, G_S),
        ])
    else:
        raise ValueError(f"Unsupported masked WAM task: {task!r}. Expected one of {WAM_MASKED_TASKS}.")
    return allow


@lru_cache(maxsize=None)
def build_token_group_routing_plan(task: str) -> Dict[int, dict]:
    """Return group routing metadata derived from the attention specification.

    Each entry contains ``experts`` and ``pool_groups``. ``pool_groups`` is the
    exact set of key groups visible to that query group in ``_allow_table``;
    therefore attention and router context cannot drift independently.
    Single-expert groups use a fixed weight and do not invoke the router.
    """
    task = canonical_masked_task(task)
    if task not in _TOKEN_GROUP_EXPERTS:
        raise ValueError(f"Task {task!r} has no token-group routing plan")
    allow = _allow_table(task)
    plan: Dict[int, dict] = {}
    for group_id, experts in _TOKEN_GROUP_EXPERTS[task].items():
        pool_groups = tuple(torch.where(allow[group_id])[0].tolist())
        plan[group_id] = {
            "experts": tuple(experts),
            "pool_groups": pool_groups,
        }
    return plan


def is_joint_dreamzero_task(task: Optional[str]) -> bool:
    """Return whether ``task`` selects the legacy DreamZero joint path."""
    if task is None:
        return False
    normalized = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"joint_dreamzero", "joint"}


def build_idm_token_group_spec(
    *,
    stage: str,
    seq_len: Optional[int] = None,
    n_video: int = 0,
    frame_seqlen: int = 0,
    n_action: int = 0,
    n_state: int = 0,
    num_frame_per_block: int = 2,
    num_action_per_block: int = 24,
) -> Dict[str, Any]:
    """Build IDM FFN groups for teacher forcing and cached AR stages.

    Training keeps the physical layout ``[noisy_V|clean_V|noisy_A|clean_A|S]``.
    Cached stages contain only the current query tokens; their post-attention
    hidden states already carry the visible clean K/V history.  Consequently
    router pooling is always query-local and is identical between dense
    teacher forcing and cached inference.
    """
    stage = str(stage).strip().lower()
    cache_stages = {
        "cache_write_seed",
        "cache_write_state",
        "cache_write_video",
        "cache_write_action",
    }
    target_stages = {
        "video_denoise",
        "action_denoise",
        "video_action_denoise",
    }
    if stage not in {"train_tf", *cache_stages, *target_stages}:
        raise ValueError(f"Unsupported IDM token-group stage: {stage!r}")

    if stage != "train_tf":
        if seq_len is None or int(seq_len) <= 0:
            raise ValueError(f"IDM {stage} requires a positive seq_len")
        length = int(seq_len)
        spec: Dict[str, Any] = {
            "task": "idm_lingbot_va",
            "stage": stage,
            "seq_len": length,
            "shared_ranges": (),
            "video_targets": (),
            "action_targets": (),
        }
        if stage in cache_stages:
            spec["shared_ranges"] = ((0, length),)
        elif stage == "video_denoise":
            spec["video_targets"] = ((0, length),)
        elif stage == "action_denoise":
            spec["action_targets"] = ((0, length),)
        else:
            raise ValueError(
                "IDM video_action_denoise requires explicit video/action spans"
            )
        return spec

    for name, value in {
        "n_video": n_video,
        "frame_seqlen": frame_seqlen,
        "n_action": n_action,
        "n_state": n_state,
        "num_frame_per_block": num_frame_per_block,
        "num_action_per_block": num_action_per_block,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"IDM train_tf {name} must be positive, got {value}")
    if n_video % frame_seqlen:
        raise ValueError("IDM video tokens must be divisible by frame_seqlen")
    num_frames = n_video // frame_seqlen
    if num_frames != 1 + n_state * num_frame_per_block:
        raise ValueError(
            "IDM train_tf requires seed + n_state video chunks; "
            f"got frames={num_frames}, states={n_state}"
        )
    if n_action != n_state * num_action_per_block:
        raise ValueError(
            "IDM train_tf action count does not match state/chunk count; "
            f"actions={n_action}, states={n_state}"
        )

    noisy_v_start = 0
    clean_v_start = n_video
    noisy_a_start = 2 * n_video
    clean_a_start = noisy_a_start + n_action
    state_start = clean_a_start + n_action
    total = state_start + n_state
    video_width = num_frame_per_block * frame_seqlen
    seed_span = (noisy_v_start, frame_seqlen)
    video_targets = []
    action_targets = []
    chunks = []
    for index in range(n_state):
        video_span = (
            frame_seqlen + index * video_width,
            frame_seqlen + (index + 1) * video_width,
        )
        action_span = (
            noisy_a_start + index * num_action_per_block,
            noisy_a_start + (index + 1) * num_action_per_block,
        )
        video_targets.append(video_span)
        action_targets.append(action_span)
        chunks.append({
            "index": index,
            "video_range": video_span,
            "video_router_ranges": (video_span,),
            "action_range": action_span,
            "action_router_ranges": (action_span,),
        })

    return {
        "task": "idm_lingbot_va",
        "stage": "train_tf",
        "seq_len": total,
        "seed_range": seed_span,
        "clean_video_range": (clean_v_start, clean_v_start + n_video),
        "clean_action_range": (clean_a_start, clean_a_start + n_action),
        "state_range": (state_start, total),
        "shared_ranges": (
            seed_span,
            (clean_v_start, clean_v_start + n_video),
            (clean_a_start, clean_a_start + n_action),
            (state_start, total),
        ),
        "video_targets": tuple(video_targets),
        "action_targets": tuple(action_targets),
        "chunks": tuple(chunks),
    }


def build_joint_token_group_spec(
    *,
    stage: str,
    n_video: int,
    frame_seqlen: int,
    n_action: int = 0,
    n_state: int = 0,
    n_clean_video: Optional[int] = None,
    num_frame_per_block: int = 2,
    num_action_per_block: int = 24,
    num_state_per_block: int = 1,
) -> Dict[str, Any]:
    """Build the single source of truth for Joint token ranges and routing.

    ``train_tf`` describes the conservative equal-length teacher-forcing stream
    ``[clean_V | noisy_V | A | S]``.  ``cache_write`` describes a clean video
    chunk being committed to the KV cache.  ``denoise`` describes the current
    cached-inference stream ``[noisy_V | noisy_A | S]``.  All ranges are
    half-open token spans in the actual FFN sequence.

    Attention and router context intentionally differ for cached Joint: the
    attention K/V source includes clean history, while the FFN only receives the
    current post-attention query tokens.  ``attention_kv_ranges`` records the
    former for teacher-forcing validation and ``router_ranges`` records the
    latter, which exists identically in training and cached inference.
    """
    stage = str(stage).strip().lower()
    if stage not in {"train_tf", "cache_write", "denoise"}:
        raise ValueError(f"Unsupported Joint token-group stage: {stage!r}")
    for name, value in {
        "n_video": n_video,
        "frame_seqlen": frame_seqlen,
        "n_action": n_action,
        "n_state": n_state,
        "num_frame_per_block": num_frame_per_block,
        "num_action_per_block": num_action_per_block,
        "num_state_per_block": num_state_per_block,
    }.items():
        if int(value) < 0 or (name in {"frame_seqlen", "num_frame_per_block", "num_action_per_block", "num_state_per_block"} and int(value) == 0):
            raise ValueError(f"Joint {name} must be positive/non-negative, got {value}")
    if n_video <= 0 or n_video % frame_seqlen != 0:
        raise ValueError(
            f"Joint video tokens must be a positive multiple of frame_seqlen; "
            f"got video={n_video}, frame_seqlen={frame_seqlen}"
        )

    base: Dict[str, Any] = {
        "task": "joint_dreamzero",
        "stage": stage,
        "n_video": int(n_video),
        "n_action": int(n_action),
        "n_state": int(n_state),
        "frame_seqlen": int(frame_seqlen),
        "num_frame_per_block": int(num_frame_per_block),
        "num_action_per_block": int(num_action_per_block),
        "num_state_per_block": int(num_state_per_block),
    }

    if stage == "cache_write":
        if n_action != 0 or n_state != 0:
            raise ValueError("Joint cache_write accepts clean video tokens only")
        base.update({
            "seq_len": int(n_video),
            "shared_ranges": ((0, int(n_video)),),
            "chunks": (),
        })
        return base

    if stage == "denoise":
        expected_video = num_frame_per_block * frame_seqlen
        if n_video != expected_video or n_action != num_action_per_block or n_state != num_state_per_block:
            raise ValueError(
                "Joint denoise requires one canonical current chunk; "
                f"got video={n_video}/{expected_video}, action={n_action}/{num_action_per_block}, "
                f"state={n_state}/{num_state_per_block}"
            )
        v_span = (0, int(n_video))
        a_span = (int(n_video), int(n_video + n_action))
        s_span = (int(n_video + n_action), int(n_video + n_action + n_state))
        base.update({
            "seq_len": int(n_video + n_action + n_state),
            "shared_ranges": (s_span,),
            "chunks": ({
                "index": 0,
                "video_range": v_span,
                "action_range": a_span,
                "state_range": s_span,
                "attention_kv_ranges": (v_span, a_span, s_span),
                "router_ranges": (v_span, a_span, s_span),
                "video_experts": ("shared", "vgm"),
                "action_experts": ("shared", "idm"),
            },),
        })
        return base

    # Conservative DreamZero teacher forcing: clean/noisy video copies remain
    # equal length and the noisy seed F is retained as an auxiliary target.
    n_clean_video = int(n_video if n_clean_video is None else n_clean_video)
    if n_clean_video != n_video:
        raise ValueError(
            "Conservative Joint train_tf requires equal clean/noisy video lengths; "
            f"got clean={n_clean_video}, noisy={n_video}"
        )
    num_frames = n_video // frame_seqlen
    if (num_frames - 1) % num_frame_per_block != 0:
        raise ValueError(
            f"Joint video layout must be 1 + N*{num_frame_per_block} latent frames; got {num_frames}"
        )
    num_chunks = (num_frames - 1) // num_frame_per_block
    if (
        n_action != num_chunks * num_action_per_block
        or n_state != num_chunks * num_state_per_block
    ):
        raise ValueError(
            "Joint action/state registers do not match the video chunk count; "
            f"chunks={num_chunks}, action={n_action}, state={n_state}"
        )

    clean_start = 0
    noisy_start = n_clean_video
    action_start = noisy_start + n_video
    state_start = action_start + n_action
    first = frame_seqlen
    video_block = num_frame_per_block * frame_seqlen
    fnoisy_span = (noisy_start, noisy_start + first)
    chunks = []
    for i in range(num_chunks):
        v_span = (
            noisy_start + first + i * video_block,
            noisy_start + first + (i + 1) * video_block,
        )
        a_span = (
            action_start + i * num_action_per_block,
            action_start + (i + 1) * num_action_per_block,
        )
        s_span = (
            state_start + i * num_state_per_block,
            state_start + (i + 1) * num_state_per_block,
        )
        clean_history = (clean_start, clean_start + first + i * video_block)
        chunks.append({
            "index": i,
            "video_range": v_span,
            "action_range": a_span,
            "state_range": s_span,
            "attention_kv_ranges": (clean_history, v_span, a_span, s_span),
            "router_ranges": (v_span, a_span, s_span),
            "video_experts": ("shared", "vgm"),
            "action_experts": ("shared", "idm"),
        })

    base.update({
        "n_clean_video": n_clean_video,
        "seq_len": int(state_start + n_state),
        "clean_range": (clean_start, n_clean_video),
        "fnoisy_range": fnoisy_span,
        "fnoisy_router_ranges": (fnoisy_span,),
        "fnoisy_experts": ("shared", "vgm"),
        "shared_ranges": ((clean_start, n_clean_video),) + tuple(
            chunk["state_range"] for chunk in chunks
        ),
        "chunks": tuple(chunks),
    })
    return base


def build_full_window_history_spec(
    task: str,
    *,
    n_video: int,
    frame_seqlen: int,
    n_action: int,
    n_state: int,
    num_frame_per_block: int = 2,
    num_action_per_block: int = 24,
    num_state_per_block: int = 1,
) -> Dict[str, Any]:
    """Describe the opt-in Joint-style carrier for AC/Fast/Giga training."""
    task = canonical_masked_task(task)
    if task not in {
        "world_model",
        "action_only_fast_wam",
        "action_only_gigaworld_policy",
    }:
        raise ValueError(f"Unsupported full-window history task: {task!r}")
    for name, value in {
        "n_video": n_video,
        "frame_seqlen": frame_seqlen,
        "n_action": n_action,
        "n_state": n_state,
        "num_frame_per_block": num_frame_per_block,
        "num_action_per_block": num_action_per_block,
        "num_state_per_block": num_state_per_block,
    }.items():
        if int(value) < 0 or (
            name in {
                "n_video",
                "frame_seqlen",
                "n_action",
                "num_frame_per_block",
                "num_action_per_block",
                "num_state_per_block",
            }
            and int(value) == 0
        ):
            raise ValueError(f"Full-window history {name} must be positive/non-negative, got {value}")
    if n_video % frame_seqlen:
        raise ValueError("Full-window history video tokens must be divisible by frame_seqlen")
    num_frames = n_video // frame_seqlen
    if (num_frames - 1) % num_frame_per_block:
        raise ValueError(
            "Full-window history requires seed + complete video chunks; "
            f"got frames={num_frames}"
        )
    num_chunks = (num_frames - 1) // num_frame_per_block
    if n_action != num_chunks * num_action_per_block:
        raise ValueError(
            "Full-window history action count does not match video chunks; "
            f"actions={n_action}, chunks={num_chunks}"
        )
    expected_states = 0 if task == "world_model" else num_chunks * num_state_per_block
    if n_state != expected_states:
        raise ValueError(
            f"Full-window history {task} requires {expected_states} state tokens, got {n_state}"
        )

    clean_start = 0
    noisy_start = n_video
    action_start = 2 * n_video
    state_start = action_start + n_action
    total = state_start + n_state
    first = frame_seqlen
    video_block = num_frame_per_block * frame_seqlen
    chunks = []
    for index in range(num_chunks):
        chunks.append({
            "index": index,
            "history_range": (clean_start, clean_start + first + index * video_block),
            "video_range": (
                noisy_start + first + index * video_block,
                noisy_start + first + (index + 1) * video_block,
            ),
            "action_range": (
                action_start + index * num_action_per_block,
                action_start + (index + 1) * num_action_per_block,
            ),
            "state_range": (
                state_start + index * num_state_per_block,
                state_start + (index + 1) * num_state_per_block,
            ) if n_state else None,
        })
    return {
        "task": task,
        "stage": "full_window_history",
        "seq_len": total,
        "n_video": n_video,
        "n_action": n_action,
        "n_state": n_state,
        "clean_range": (clean_start, noisy_start),
        "noisy_range": (noisy_start, action_start),
        "noisy_seed_range": (noisy_start, noisy_start + first),
        "action_range": (action_start, state_start),
        "state_range": (state_start, total),
        "chunks": tuple(chunks),
    }


def build_full_window_history_mask(
    task: str,
    *,
    n_video: int,
    frame_seqlen: int,
    n_action: int,
    n_state: int,
    num_frame_per_block: int = 2,
    num_action_per_block: int = 24,
    num_state_per_block: int = 1,
    device=None,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Build the leak-free Joint-style history mask for AC/Fast/Giga."""
    spec = build_full_window_history_spec(
        task,
        n_video=n_video,
        frame_seqlen=frame_seqlen,
        n_action=n_action,
        n_state=n_state,
        num_frame_per_block=num_frame_per_block,
        num_action_per_block=num_action_per_block,
        num_state_per_block=num_state_per_block,
    )
    task = spec["task"]
    mask = torch.eye(spec["seq_len"], dtype=torch.bool, device=device)

    def allow(query_range, *key_ranges):
        if query_range is None:
            return
        q0, q1 = query_range
        for key_range in key_ranges:
            if key_range is None:
                continue
            k0, k1 = key_range
            mask[q0:q1, k0:k1] = True

    clean_start, _ = spec["clean_range"]
    noisy_seed = spec["noisy_seed_range"]
    first = frame_seqlen
    allow((clean_start, clean_start + first), (clean_start, clean_start + first))
    for chunk in spec["chunks"]:
        clean_chunk = (
            clean_start + first + chunk["index"] * num_frame_per_block * frame_seqlen,
            clean_start + first + (chunk["index"] + 1) * num_frame_per_block * frame_seqlen,
        )
        allow(clean_chunk, (clean_start, clean_chunk[1]))

    first_state = spec["chunks"][0]["state_range"] if n_state else None
    allow(noisy_seed, (clean_start, clean_start + first), noisy_seed, first_state)
    for chunk in spec["chunks"]:
        common = (chunk["history_range"], noisy_seed, chunk["state_range"])
        video_keys = (*common, chunk["video_range"])
        if task in {"world_model", "action_only_gigaworld_policy"}:
            video_keys = (*video_keys, chunk["action_range"])
        allow(chunk["video_range"], *video_keys)
        allow(
            chunk["action_range"],
            chunk["history_range"], noisy_seed,
            chunk["action_range"], chunk["state_range"],
        )
        allow(
            chunk["state_range"],
            chunk["history_range"], noisy_seed, chunk["state_range"],
        )
    return spec, mask


def build_group_ids(
    task: str,
    n_video: int,
    frame_seqlen: int,
    n_action: int,
    n_state: int,
    device=None,
) -> torch.Tensor:
    """Build a [L] long tensor mapping each token to its group id.

    Sequence layout depends on task:
      - world_model:  [F+Vfut (video) | A]           (no state)
      - fastwam/giga: [F+Vfut (video) | A | S]
    The video segment always starts with `frame_seqlen` first-frame tokens (F),
    the rest are future-video tokens (Vfut).
    """
    task = canonical_masked_task(task)
    ids: List[torch.Tensor] = []
    # video: first frame_seqlen tokens are F, the rest Vfut
    n_first = min(frame_seqlen, n_video)
    video_ids = torch.full((n_video,), G_VFUT, dtype=torch.long, device=device)
    video_ids[:n_first] = G_F
    ids.append(video_ids)
    if n_action > 0:
        ids.append(torch.full((n_action,), G_A, dtype=torch.long, device=device))
    if n_state > 0 and task != "world_model":
        ids.append(torch.full((n_state,), G_S, dtype=torch.long, device=device))
    return torch.cat(ids, dim=0)


def build_task_self_attn_mask(
    task: str,
    group_ids: torch.Tensor,
    device=None,
) -> torch.Tensor:
    """Return a boolean [L, L] mask (True = attention allowed) for the task.

    The diagonal is always allowed (every token attends itself).
    """
    task = canonical_masked_task(task)
    if task is None:
        raise ValueError("build_task_self_attn_mask requires a masked WAM task.")
    device = device if device is not None else group_ids.device
    allow = _allow_table(task).to(device)  # [G, G]
    gids = group_ids.to(device)
    # mask[i, j] = allow[gid[i], gid[j]] via gather
    mask = allow[gids][:, gids]  # [L, L] bool
    # always allow self
    L = gids.shape[0]
    mask = mask | torch.eye(L, dtype=torch.bool, device=device)
    return mask


def build_task_self_attn_mask_from_lengths(
    task: str,
    n_video: int,
    frame_seqlen: int,
    n_action: int,
    n_state: int,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience: build (group_ids, mask) from segment lengths."""
    group_ids = build_group_ids(task, n_video, frame_seqlen, n_action, n_state, device=device)
    mask = build_task_self_attn_mask(task, group_ids, device=device)
    return group_ids, mask


def build_acwm_two_history_chunk_mask(
    *,
    n_video: int,
    frame_seqlen: int,
    n_action: int,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build AC-WM mask for [H0(2) | H1(2) | future(2) | action]."""
    if frame_seqlen <= 0 or n_action <= 0:
        raise ValueError("AC-WM frame_seqlen and n_action must be positive")
    expected_video = 6 * int(frame_seqlen)
    if n_video != expected_video:
        raise ValueError(
            f"AC-WM requires six latent frames ({expected_video} tokens), "
            f"got {n_video}"
        )
    h0_end = 2 * int(frame_seqlen)
    h1_end = 4 * int(frame_seqlen)
    future_end = int(n_video)
    total = future_end + int(n_action)
    group_ids = torch.empty(total, dtype=torch.long, device=device)
    group_ids[:h1_end] = G_F
    group_ids[h1_end:future_end] = G_VFUT
    group_ids[future_end:] = G_A
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[:h0_end, :h0_end] = True
    mask[h0_end:h1_end, :h1_end] = True
    mask[h1_end:future_end, :total] = True
    mask[future_end:, :h1_end] = True
    mask[future_end:, future_end:] = True
    return group_ids, mask


# ---------------------------------------------------------------------------
# idm_lingbot_va: interleaved teacher-forcing mask (lingbot-va style).
# ---------------------------------------------------------------------------

def build_idm_teacher_forcing_mask(
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    n_action: int,
    num_action_per_block: int,
    n_state: int,
    device=None,
    window_size: Optional[int] = None,
):
    """Interleaved teacher-forcing mask for idm_lingbot_va.

    Sequence layout (single stream): [noisy_V | clean_V | noisy_A | clean_A | S].
    The canonical timeline is:
    episode seed -> current state -> 2 latent video -> 24 actions, repeated.
    ``noise_id`` = 0 for the noisy copies (denoise targets) and 1 for the clean
    copies + state (teacher-forcing conditioning).  The seed is conditioning
    only; state ``c`` is placed immediately before that chunk's future video.

    Connectivity (OR of three rules, mirroring lingbot ``_get_mask_mod``):
      - clean -> clean : block-causal (kv_fid <= q_fid)
      - noisy -> clean : strictly-earlier block-causal (kv_fid <  q_fid)  [teacher forcing]
      - noisy -> noisy : same block only (kv_fid == q_fid)
    Diagonal always allowed.

    Returns (frame_ids, noise_ids, mask, seg_lengths) where seg_lengths =
    (nV, nV, nA, nA, n_state) for [noisy_V, clean_V, noisy_A, clean_A, S].
    """
    nV = num_frames * frame_seqlen
    nA = n_action

    v_fid, a_fid, s_fid = canonical_idm_frame_ids(
        num_frames, frame_seqlen, num_frame_per_block,
        n_action, num_action_per_block, n_state, device=device,
    )

    frame_ids = torch.cat([v_fid, v_fid.clone(), a_fid, a_fid.clone(), s_fid], dim=0)
    noise_ids = torch.cat([
        torch.zeros(nV, dtype=torch.long, device=device),   # noisy_V
        torch.ones(nV, dtype=torch.long, device=device),    # clean_V
        torch.zeros(nA, dtype=torch.long, device=device),   # noisy_A
        torch.ones(nA, dtype=torch.long, device=device),    # clean_A
        torch.ones(n_state, dtype=torch.long, device=device),  # state (clean)
    ], dim=0)
    # Both copies of frame 0 are the clean episode seed.  It is visible as
    # conditioning but never treated as a future/noisy target.
    seed_tokens = frame_seqlen
    noise_ids[:seed_tokens] = 1

    L = frame_ids.shape[0]
    qf = frame_ids.view(L, 1)
    kf = frame_ids.view(1, L)
    qn = noise_ids.view(L, 1)
    kn = noise_ids.view(1, L)

    clean2clean = (qn == 1) & (kn == 1) & (kf <= qf)
    noise2clean = (qn == 0) & (kn == 1) & (kf < qf)
    noise2noise = (qn == 0) & (kn == 0) & (kf == qf)
    mask = clean2clean | noise2clean | noise2noise
    if window_size is not None:
        # lingbot block_window_mask: only attend within +/- window_size blocks.
        mask = mask & ((qf - kf).abs() <= window_size)
    mask = mask | torch.eye(L, dtype=torch.bool, device=device)

    seg_lengths = (nV, nV, nA, nA, n_state)
    return frame_ids, noise_ids, mask, seg_lengths
