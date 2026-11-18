"""Shared episode orchestration helpers without torch or hydra imports.

Used by offline evaluation utilities without loading the model.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

VALID_EVAL_TASKS = frozenset(
    {
        "ac_wm",
        "joint_dreamzero",
        "action_only_fast_wam",
        "action_only_gigaworld_policy",
        "idm_lingbot_va",
    }
)

EVAL_TASK_ALIASES = {
    "world_model": "ac_wm",
    "wm": "ac_wm",
    "embodied_world_model": "ac_wm",
}

DEFAULT_WAM_EVAL_OUTPUT_ROOT = Path(
    "./outputs/evaluation"
)

# Tasks whose episode rollout produces predicted ACTIONS compared to GT (masked
# MSE + trajectory plots). ac_wm is video-only (no action metric).
ACTION_METRIC_TASKS = frozenset(
    {
        "joint_dreamzero",
        "action_only_fast_wam",
        "action_only_gigaworld_policy",
        "idm_lingbot_va",
    }
)

JOINT_PERSISTENT_ATTRS = (
    "language",
    "current_start_frame",
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
    "clip_feas",
    "ys",
)


def load_lerobot_task_instructions(data_root: str | Path) -> dict[int, str]:
    """Load non-empty task instructions from a LeRobot dataset."""
    tasks_path = Path(data_root) / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"dataset task metadata not found: {tasks_path}")

    instructions: dict[int, str] = {}
    with open(tasks_path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            instruction = str(record.get("task", "")).strip()
            if not instruction:
                raise ValueError(
                    f"empty dataset instruction at {tasks_path}:{line_number}"
                )
            instructions[int(record["task_index"])] = instruction

    if not instructions:
        raise ValueError(f"dataset contains no task instructions: {tasks_path}")
    return instructions


def center_crop_frame(frame, scale: float = 0.95):
    """Deterministically center-crop one HWC frame using eval-transform rounding."""
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"crop scale must be in (0, 1], got {scale}")
    height, width = frame.shape[:2]
    crop_height = int(height * scale)
    crop_width = int(width * scale)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return frame[top : top + crop_height, left : left + crop_width]


def parse_eval_tasks() -> list[str]:
    """Resolve eval tasks from WAM_EVAL_TASKS or legacy WAM_MOE_TASK.

    Validates against VALID_EVAL_TASKS, deduplicates, preserves first-seen order.
    Rejects empty task lists (e.g. WAM_EVAL_TASKS=',').
    """
    raw = os.environ.get("WAM_EVAL_TASKS", "").strip()
    if raw:
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    else:
        fallback = os.environ.get("WAM_MOE_TASK", "joint_dreamzero").strip().lower()
        if not fallback:
            raise ValueError(
                "no eval tasks resolved; set WAM_EVAL_TASKS or WAM_MOE_TASK"
            )
        parts = [fallback]

    seen: set[str] = set()
    tasks: list[str] = []
    for task in parts:
        task = EVAL_TASK_ALIASES.get(task, task)
        if task not in VALID_EVAL_TASKS:
            allowed = ", ".join(sorted(VALID_EVAL_TASKS))
            raise ValueError(f"invalid eval task {task!r}; allowed: {allowed}")
        if task not in seen:
            seen.add(task)
            tasks.append(task)

    if not tasks:
        raise ValueError(
            "no eval tasks resolved; set WAM_EVAL_TASKS or WAM_MOE_TASK"
        )
    return tasks


def checkpoint_out_slug(checkpoint_path: str | None) -> str:
    if not checkpoint_path:
        return "pour_rollout"
    name = Path(checkpoint_path).name
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    if m:
        return f"pour_rollout_ckpt{m.group(1)}"
    slug = re.sub(r"[^\w.-]+", "_", name).strip("._")
    return f"pour_rollout_{slug}" if slug else "pour_rollout"


def resolve_eval_output_base() -> Path:
    """Shared external root for evaluation artifacts."""
    configured = os.environ.get("WAM_EVAL_OUTPUT_ROOT", "").strip()
    root = Path(configured).expanduser() if configured else DEFAULT_WAM_EVAL_OUTPUT_ROOT
    if not root.is_absolute():
        raise ValueError(f"WAM_EVAL_OUTPUT_ROOT must be absolute, got {str(root)!r}")
    return root


def resolve_eval_out_root(explicit_out_dir: str | None, checkpoint_path: str | None) -> Path:
    """Output root: explicit WAM_EVAL_OUT_DIR, else shared external root."""
    if explicit_out_dir:
        return Path(explicit_out_dir)
    return resolve_eval_output_base() / checkpoint_out_slug(checkpoint_path)


def resolve_hydra_output_dir(
    dreamzero_root: str | Path,
    explicit_out_dir: str | None,
    checkpoint_path: str | None,
) -> Path:
    """Hydra scratch dir under the resolved evaluation output root."""
    if explicit_out_dir:
        return Path(explicit_out_dir) / "hydra"
    return resolve_eval_out_root(None, checkpoint_path) / "hydra"


def clear_joint_persistent_state(head) -> None:
    """Reset joint DreamZero KV-cache / language state without touching model weights."""
    reset = getattr(head, "wam_joint_episode_reset", None)
    if callable(reset):
        reset()
        return
    for attr in JOINT_PERSISTENT_ATTRS:
        if not hasattr(head, attr):
            continue
        if attr == "current_start_frame":
            head.current_start_frame = 0
        else:
            setattr(head, attr, None)
    model = getattr(head, "model", None)
    if model is not None:
        if hasattr(model, "reset_idm_cache"):
            model.reset_idm_cache()
        if hasattr(model, "_current_wam_expert_mask"):
            model._current_wam_expert_mask = None


def configure_eval_task(
    head,
    task: str,
    steps: int,
    wm_cfg_scale: float,
    joint_cfg_scale: float,
) -> None:
    """Switch MoE task env + action_head runtime for one eval task."""
    task = EVAL_TASK_ALIASES.get(task, task)
    if task not in VALID_EVAL_TASKS:
        allowed = ", ".join(sorted(VALID_EVAL_TASKS))
        raise ValueError(f"invalid eval task {task!r}; allowed: {allowed}")

    os.environ["WAM_MOE_TASK"] = task
    os.environ["WAM_MOE_PHASE"] = "infer"
    os.environ["WAM_MOE_PASS"] = "default"

    head.moe_task = task
    head.num_inference_steps = steps
    head.cfg_scale = joint_cfg_scale if task == "joint_dreamzero" else wm_cfg_scale

    if head.moe_task != task:
        raise RuntimeError(f"head.moe_task={head.moe_task!r} != requested task={task!r}")
    if os.environ.get("WAM_MOE_TASK") != task:
        raise RuntimeError(
            f"WAM_MOE_TASK={os.environ.get('WAM_MOE_TASK')!r} != requested task={task!r}"
        )
    if os.environ.get("WAM_MOE_PASS") != "default":
        raise RuntimeError(
            f"WAM_MOE_PASS={os.environ.get('WAM_MOE_PASS')!r} != 'default'"
        )

    model = getattr(head, "model", None)
    if model is not None:
        model._current_wam_task = task
        if hasattr(model, "_current_wam_expert_mask"):
            model._current_wam_expert_mask = None


def collective_check_errors(dist, local_error: str | None) -> None:
    """All-gather rank errors; raise on every rank if any rank failed."""
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    failed = [e for e in errors if e]
    if failed:
        raise RuntimeError(
            "Distributed rollout failed on one or more ranks:\n" + "\n".join(failed)
        )


def merge_gathered_metrics(gathered: list) -> list[dict]:
    """Flatten per-rank metric lists and sort by episode index."""
    return sorted(
        (m for part in gathered if part for m in part),
        key=lambda m: m.get("episode", -1),
    )


def write_task_summary_metrics(summary_path: Path, all_metrics: list[dict]) -> str | None:
    """Write per-task summary JSON.

    Returns an error message on failure (for collective propagation), else None.
    """
    import traceback

    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        return None
    except Exception as exc:
        return (
            f"task summary write failed ({summary_path}): {exc}\n"
            f"{traceback.format_exc()}"
        )


def propagate_summary_write_error(dist, rank: int, summary_error: str | None) -> None:
    """Broadcast rank-0 summary write failures so no rank blocks on barrier."""
    local_error = summary_error if rank == 0 else None
    collective_check_errors(dist, local_error)


def write_eval_manifest(
    path: Path,
    *,
    checkpoint: str,
    tasks: list[str],
    episodes_spec: str,
    episode_seed: int,
    episodes: list[int],
    steps: int,
    wm_cfg_scale: float,
    idm_cfg_scale: float,
    joint_cfg_scale: float,
    max_chunks: int | None,
    world_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": checkpoint,
        "tasks": tasks,
        "episodes_spec": episodes_spec,
        "episode_seed": episode_seed,
        "episodes": episodes,
        "steps": steps,
        "cfg": {
            "ac_wm": wm_cfg_scale,
            "idm_lingbot_va": idm_cfg_scale,
            "joint_dreamzero": joint_cfg_scale,
        },
        # MAX_CHUNKS env caps both tasks equally today; record explicit per-task keys.
        "ac_wm_max_windows": max_chunks,
        "world_model_max_windows": max_chunks,
        "joint_max_chunks": max_chunks,
        "max_chunks": max_chunks,
        "_limits_note": (
            "ac_wm_max_windows caps AC-WM replan windows; joint_max_chunks caps "
            "joint DreamZero chunks. Both derive from MAX_CHUNKS (0 = full episode)."
        ),
        "world_size": world_size,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[rollout] wrote manifest -> {path}")


def resolve_episodes(spec: str, data_root: str) -> list[int]:
    """EPISODES spec -> list of episode indices.

    "all" selects every dataset episode. "random:K" samples K distinct episodes
    with EPISODE_SEED (default 0) — the draw is deterministic, so all ranks and
    both task runs see the same set. Otherwise: comma-separated explicit indices.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("EPISODES spec is empty")

    if spec.lower() == "all":
        info_path = Path(data_root) / "meta/info.json"
        total = int(json.load(open(info_path))["total_episodes"])
        if total <= 0:
            raise ValueError("EPISODES spec 'all' resolved to empty episode list")
        return list(range(total))

    if spec.lower().startswith("random:"):
        import random as _random

        k_str = spec.split(":", 1)[1].strip()
        if not k_str:
            raise ValueError(f"invalid EPISODES spec {spec!r}: missing K after random:")
        k = int(k_str)
        if k <= 0:
            raise ValueError(f"EPISODES spec {spec!r} resolved to empty episode list")
        info_path = Path(data_root) / "meta/info.json"
        total = json.load(open(info_path))["total_episodes"]
        if k > total:
            raise ValueError(
                f"EPISODES {spec!r} requests {k} episodes but dataset has {total}"
            )
        seed = int(os.environ.get("EPISODE_SEED", "0"))
        return sorted(_random.Random(seed).sample(range(total), k))

    episodes = [int(e) for e in spec.split(",") if e.strip()]
    if not episodes:
        raise ValueError(f"EPISODES spec {spec!r} resolved to empty episode list")
    return episodes


def _cli_main(argv: list[str] | None = None) -> int:
    """Small CLI for shell helpers (no torch/hydra imports)."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m uniwam.inference.orchestration hydra-output-dir ROOT [OUT_DIR] [CHECKPOINT]", file=sys.stderr)
        return 2

    cmd = args[0]
    if cmd == "hydra-output-dir":
        if len(args) < 2:
            print("hydra-output-dir requires ROOT", file=sys.stderr)
            return 2
        root = args[1]
        explicit_out = args[2] if len(args) > 2 and args[2] else None
        checkpoint = args[3] if len(args) > 3 and args[3] else None
        print(resolve_hydra_output_dir(root, explicit_out, checkpoint))
        return 0
    if cmd == "checkpoint-out-slug":
        checkpoint = args[1] if len(args) > 1 else None
        print(checkpoint_out_slug(checkpoint))
        return 0

    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())
