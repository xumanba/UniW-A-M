"""Low-overhead sampled-gradient diagnostics for WAM multi-task training.

The probe is opt-in and observes gradients only. It never changes parameters,
losses, routing, or optimizer state. For each parameter group it samples a
fixed set of coordinates, averages those gradients across distributed ranks,
and reports cosine similarity between the mean gradients of each WAM task.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist


_TASK_ALIASES = {
    "world_model": "ac_wm",
    "embodied_world_model": "ac_wm",
    "ac_wm": "ac_wm",
    "joint": "joint_dreamzero",
    "joint_dreamzero": "joint_dreamzero",
    "idm": "idm_lingbot_va",
    "idm_lingbot_va": "idm_lingbot_va",
    "fast_wam": "action_only_fast_wam",
    "action_only_fast_wam": "action_only_fast_wam",
    "gigaworld_policy": "action_only_gigaworld_policy",
    "action_only_gigaworld_policy": "action_only_gigaworld_policy",
}
_TASK_ORDER = (
    "joint_dreamzero",
    "idm_lingbot_va",
    "action_only_fast_wam",
    "action_only_gigaworld_policy",
    "ac_wm",
)
_GROUP_ORDER = (
    "shared_non_moe",
    "shared_expert",
    "vgm_expert",
    "vla_expert",
    "wm_expert",
    "idm_expert",
    "router",
    "action_state",
)


def _canonical_task(task: str) -> str:
    value = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    return _TASK_ALIASES.get(value, value or "unknown")


def _parameter_group(name: str) -> str:
    qualified_name = f".{name}."
    for expert in ("shared", "vgm", "vla", "wm", "idm"):
        if f".control_moe.{expert}_expert." in qualified_name:
            return f"{expert}_expert"
    if ".control_moe.router." in qualified_name:
        return "router"
    if any(
        token in qualified_name
        for token in (
            ".action_encoder.",
            ".action_decoder.",
            ".state_encoder.",
        )
    ):
        return "action_state"
    return "shared_non_moe"


class GradientConflictProbe:
    """Collect deterministic coordinate samples from per-task gradients."""

    def __init__(
        self,
        *,
        output_dir: str,
        sample_size: int = 16384,
        seed: int = 20260822,
        log_every: int = 10,
    ) -> None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        self.output_path = Path(output_dir) / "gradient_conflict_probe.json"
        remote_output = os.environ.get("WAM_GRADIENT_CONFLICT_OUTPUT", "").strip()
        self.remote_output_path = Path(remote_output) if remote_output else None
        self.sample_size = int(sample_size)
        self.seed = int(seed)
        self.log_every = int(log_every)
        self._handles = []
        self._current: Dict[str, torch.Tensor | None] = {}
        self._sample_counts: Dict[str, int] = {}
        self._parameter_counts: Dict[str, int] = {}
        self._gradient_sums = defaultdict(dict)
        self._step_norm_sums = defaultdict(lambda: defaultdict(float))
        self._nonzero_steps = defaultdict(lambda: defaultdict(int))
        self._task_counts = defaultdict(int)
        self._attached = False
        self._last_step = -1
        self._device_parameter = None

    @staticmethod
    def _group_seed(seed: int, group: str) -> int:
        digest = hashlib.sha256(group.encode("utf-8")).digest()
        return (seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)

    def attach(self, model: torch.nn.Module) -> None:
        if self._attached:
            raise RuntimeError("GradientConflictProbe is already attached")

        grouped = defaultdict(list)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                grouped[_parameter_group(name)].append((name, parameter))
                if self._device_parameter is None:
                    self._device_parameter = parameter
        if self._device_parameter is None:
            raise ValueError("gradient probe found no trainable parameters")

        for group in _GROUP_ORDER:
            entries = grouped.get(group, ())
            total = sum(parameter.numel() for _, parameter in entries)
            if total == 0:
                continue
            count = min(self.sample_size, total)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._group_seed(self.seed, group))
            positions = torch.randint(
                total, (count,), generator=generator, dtype=torch.int64
            ).sort().values
            self._sample_counts[group] = count
            self._parameter_counts[group] = total

            offset = 0
            for _, parameter in entries:
                end = offset + parameter.numel()
                left = int(torch.searchsorted(positions, offset, right=False))
                right = int(torch.searchsorted(positions, end, right=False))
                if right > left:
                    source = (positions[left:right] - offset).clone()
                    destination = torch.arange(left, right, dtype=torch.int64)

                    def capture(
                        gradient,
                        *,
                        group_name=group,
                        source_indices=source,
                        destination_indices=destination,
                    ):
                        sketch = self._current.get(group_name)
                        if sketch is None:
                            sketch = torch.zeros(
                                self._sample_counts[group_name],
                                device=gradient.device,
                                dtype=torch.float32,
                            )
                            self._current[group_name] = sketch
                        values = gradient.reshape(-1).index_select(
                            0, source_indices.to(gradient.device)
                        )
                        sketch.index_copy_(
                            0,
                            destination_indices.to(gradient.device),
                            values.float(),
                        )
                        return gradient

                    self._handles.append(parameter.register_hook(capture))
                offset = end

        self._attached = True

    def begin_step(self) -> None:
        if not self._attached:
            raise RuntimeError("attach(model) must be called before begin_step()")
        self._current = {group: None for group in self._sample_counts}

    def _distributed_mean(self, vector: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(vector, op=dist.ReduceOp.SUM)
            vector.div_(dist.get_world_size())
        return vector

    def end_step(self, *, task: str, step: int, ordinal: int = -1) -> dict | None:
        canonical_task = _canonical_task(task)
        device = self._device_parameter.device
        self._task_counts[canonical_task] += 1
        self._last_step = int(step)

        for group, count in self._sample_counts.items():
            vector = self._current.get(group)
            if vector is None:
                vector = torch.zeros(count, device=device, dtype=torch.float32)
            vector = self._distributed_mean(vector).detach().cpu().double()
            if group not in self._gradient_sums[canonical_task]:
                self._gradient_sums[canonical_task][group] = torch.zeros_like(vector)
            self._gradient_sums[canonical_task][group].add_(vector)
            norm = float(torch.linalg.vector_norm(vector))
            self._step_norm_sums[canonical_task][group] += norm
            if norm > 0.0:
                self._nonzero_steps[canonical_task][group] += 1

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        summary = None
        if rank == 0:
            summary = self.summary(ordinal=ordinal)
            self._write_summary(summary)
            if all(self._task_counts.get(task_name, 0) > 0 for task_name in _TASK_ORDER):
                if sum(self._task_counts.values()) % self.log_every == 0:
                    self._log_summary(summary)
        return summary

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
        denominator = float(
            torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        )
        if denominator <= 0.0:
            return None
        value = float(torch.dot(left, right) / denominator)
        return max(-1.0, min(1.0, value))

    def summary(self, *, ordinal: int = -1) -> dict:
        tasks = [task for task in _TASK_ORDER if self._task_counts.get(task, 0) > 0]
        tasks.extend(
            sorted(task for task in self._task_counts if task not in set(tasks))
        )
        group_summaries = {}
        for group in _GROUP_ORDER:
            if group not in self._sample_counts:
                continue
            means = {}
            task_stats = {}
            for task in tasks:
                count = self._task_counts[task]
                total = self._gradient_sums[task].get(group)
                if total is None:
                    total = torch.zeros(self._sample_counts[group], dtype=torch.float64)
                mean = total / count
                means[task] = mean
                task_stats[task] = {
                    "mean_gradient_norm": float(torch.linalg.vector_norm(mean)),
                    "mean_step_gradient_norm": self._step_norm_sums[task][group] / count,
                    "nonzero_step_fraction": self._nonzero_steps[task][group] / count,
                }
            cosine = {
                left: {
                    right: self._cosine(means[left], means[right])
                    for right in tasks
                }
                for left in tasks
            }
            finite_values = [
                value
                for left in tasks
                for right, value in cosine[left].items()
                if left != right and value is not None and math.isfinite(value)
            ]
            group_summaries[group] = {
                "parameter_coordinates": self._parameter_counts[group],
                "sampled_coordinates": self._sample_counts[group],
                "task_stats": task_stats,
                "pairwise_cosine": cosine,
                "off_diagonal_negative_fraction": (
                    sum(value < 0.0 for value in finite_values) / len(finite_values)
                    if finite_values
                    else None
                ),
            }

        return {
            "format": "wam_gradient_conflict_probe_v1",
            "sampling": "deterministic_uniform_coordinate_sample_with_replacement",
            "seed": self.seed,
            "sample_size_per_group": self.sample_size,
            "last_micro_step": self._last_step,
            "last_task_ordinal": int(ordinal),
            "task_counts": dict(self._task_counts),
            "groups": group_summaries,
        }

    def _write_summary(self, summary: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, self.output_path)
        if self.remote_output_path is not None:
            self.remote_output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.remote_output_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")

    @staticmethod
    def _format_metric(value: float | None) -> str:
        return "null" if value is None else f"{value:.9g}"

    def _log_summary(self, summary: dict) -> None:
        counts = ",".join(
            f"{task}:{count}" for task, count in summary["task_counts"].items()
        )
        print(
            "WAM_GRADIENT_CONFLICT_COUNTS "
            f"step={summary['last_micro_step']} counts={counts}",
            flush=True,
        )
        for group, group_summary in summary["groups"].items():
            tasks = list(group_summary["pairwise_cosine"])
            for left_index, left in enumerate(tasks):
                for right in tasks[left_index + 1 :]:
                    value = group_summary["pairwise_cosine"][left][right]
                    print(
                        "WAM_GRADIENT_CONFLICT_COSINE "
                        f"step={summary['last_micro_step']} group={group} "
                        f"left={left} right={right} "
                        f"value={self._format_metric(value)}",
                        flush=True,
                    )
            for task, task_stats in group_summary["task_stats"].items():
                print(
                    "WAM_GRADIENT_CONFLICT_TASK_STAT "
                    f"step={summary['last_micro_step']} group={group} task={task} "
                    f"mean_norm={self._format_metric(task_stats['mean_gradient_norm'])} "
                    f"mean_step_norm={self._format_metric(task_stats['mean_step_gradient_norm'])} "
                    f"nonzero_fraction={self._format_metric(task_stats['nonzero_step_fraction'])}",
                    flush=True,
                )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
