# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import ABC
import contextlib
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Optional
import warnings

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Dataset, Sampler
import transformers
from transformers import TrainerCallback, set_seed
from transformers.trainer import (
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

import groot.vla.common.utils as U
from groot.vla.data.dataset.bucketed_batch import (
    BucketedBatchIterableDataset,
    TaskAcknowledgingDataLoader,
    TaskAwareBatchDataset,
    TaskOrdinalSampler,
    TaskOrdinalState,
    canonical_wam_task_name,
    wam_benchmark_task_sequence,
)
from groot.vla.data.dataset.lerobot_sharded import ShardedLeRobotMixtureDataset
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.experiment.utils import (
    compute_grad_accum_to_match_global_bs,
    dtype_from_string,
    get_checkpoint_path,
    mprint,
    safe_save_model_for_hf_trainer,
)
from groot.vla.utils.timer import ContextTimer

# Fix resume: https://github.com/huggingface/transformers/pull/34632/files
np_core = np.core
allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
# numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
# all versions of numpy
allowlist += [type(np.dtype(np.uint32))]
torch.serialization.add_safe_globals(allowlist)

# LayerNorm classes excluded from weight decay
LAYERNORM_LAYERS = [
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LocalResponseNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
]


def _enum_value(value):
    return getattr(value, "value", str(value))


def _build_wam_effective_config(model, training_args) -> dict:
    head = model.action_head
    per_device_batch = int(training_args.per_device_train_batch_size)
    gradient_accumulation = int(training_args.gradient_accumulation_steps)
    world_size = int(training_args.world_size)
    return {
        "schema_version": 1,
        "code_commit": os.environ.get("EXPECTED_COMMIT") or None,
        "wam": {
            "unified_joint_consumption": bool(head.unified_joint_consumption),
            "unified_chunkwise_attention": bool(head.unified_chunkwise_attention),
            "force_token_group_experiment": bool(head.force_token_group_experiment),
            "moe_routing_granularity": str(head.moe_routing_granularity),
            "moe_routing_mode": str(head.moe_routing_mode),
            "task_ratios": os.environ.get("WAM_TASK_RATIOS", ""),
            "num_frames": int(head.num_frames),
            "action_horizon": int(head.action_horizon),
            "num_frame_per_block": int(head.num_frame_per_block),
            "num_action_per_block": int(head.model.num_action_per_block),
            "num_state_per_block": int(head.model.num_state_per_block),
            "action_only_action_loss_weight": float(
                head.action_only_action_loss_weight
            ),
        },
        "optimization": {
            "learning_rate": float(training_args.learning_rate),
            "lr_scheduler_type": _enum_value(training_args.lr_scheduler_type),
            "warmup_steps": int(training_args.warmup_steps),
            "warmup_ratio": float(training_args.warmup_ratio),
            "max_steps": int(training_args.max_steps),
            "save_steps": int(training_args.save_steps),
            "save_total_limit": int(training_args.save_total_limit),
            "per_device_train_batch_size": per_device_batch,
            "global_batch_size": per_device_batch
            * gradient_accumulation
            * world_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "save_only_model": bool(training_args.save_only_model),
            "keep_all_optimizer_states": os.environ.get(
                "WAM_KEEP_ALL_OPTIMIZER_STATES", "0"
            )
            == "1",
        },
    }


def _attach_wam_effective_config(model, cfg, payload: dict) -> None:
    model.config.wam_effective_config = payload
    action_head_cfg = model.config.action_head_cfg
    inner = action_head_cfg.get("config", action_head_cfg)
    moe = dict(inner.get("moe") or {})
    moe["routing_granularity"] = payload["wam"]["moe_routing_granularity"]
    moe["routing_mode"] = payload["wam"]["moe_routing_mode"]
    inner["moe"] = moe
    model.action_head.config.moe = dict(moe)
    model.action_head.config.wam_effective_config = payload

    with open_dict(cfg):
        cfg.wam_effective_config = payload
        OmegaConf.update(
            cfg,
            "action_head_cfg.config.moe.routing_granularity",
            payload["wam"]["moe_routing_granularity"],
            merge=False,
        )
        OmegaConf.update(
            cfg,
            "action_head_cfg.config.moe.routing_mode",
            payload["wam"]["moe_routing_mode"],
            merge=False,
        )


def _wam_profile_batch_metadata(inputs):
    """Return the homogeneous WAM task and logical micro-batch ordinal."""

    def first(value):
        if torch.is_tensor(value):
            return value.reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    task = first(inputs.get("wam_task", "unknown"))
    ordinal = first(inputs.get("wam_task_ordinal", -1))
    return str(task), int(ordinal)


def _wam_benchmark_batch_metadata(inputs):
    """Extract the deterministic locator carried by a benchmark batch."""

    def values(value):
        if torch.is_tensor(value):
            return value.detach().reshape(-1).tolist()
        if isinstance(value, np.ndarray):
            return value.reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            return [item for child in value for item in values(child)]
        return [value]

    required = (
        "wam_task",
        "wam_task_ordinal",
        "wam_episode_index",
        "wam_window_start",
    )
    missing = [key for key in required if key not in inputs]
    if missing:
        raise RuntimeError(
            "WAM benchmark batch is missing deterministic locator fields: "
            + ", ".join(missing)
        )
    task_values = [str(value) for value in values(inputs["wam_task"])]
    ordinal_values = [int(value) for value in values(inputs["wam_task_ordinal"])]
    episode_values = [int(value) for value in values(inputs["wam_episode_index"])]
    window_values = [int(value) for value in values(inputs["wam_window_start"])]
    if len(set(task_values)) != 1 or len(set(ordinal_values)) != 1:
        raise RuntimeError("WAM benchmark batch is not homogeneous in task/ordinal")
    if len(episode_values) != len(window_values):
        raise RuntimeError("WAM benchmark episode/window locator lengths differ")
    return {
        "task": task_values[0],
        "ordinal": ordinal_values[0],
        "episode_index": episode_values[0],
        "window_start": window_values[0],
        "episode_indices": episode_values,
        "window_starts": window_values,
    }


class WamBenchmarkStepCallback(TrainerCallback):
    """Capture full optimizer-step timings and verify A/B sample replay.

    The interval starts after the previous optimizer step and ends after the
    current optimizer step.  It therefore includes the next batch's data
    wait, forward/backward, distributed reductions, and optimizer work.  The
    next interval is anchored only after the current record's own all-reduce
    and JSONL writes, so benchmark instrumentation is not charged to the
    following step.  The model-side duration is captured separately in
    ``BaseTrainer.training_step``.
    """

    def __init__(
        self,
        trainer,
        *,
        expected_steps: int,
        warmup_steps: int,
        output_dir: str | Path,
    ):
        self.trainer = trainer
        self.expected_steps = int(expected_steps)
        self.warmup_steps = int(warmup_steps)
        self.output_dir = Path(output_dir)
        self.rank = int(os.environ.get("RANK", "0"))
        self.run_id = os.environ.get("WAM_BENCHMARK_RUN_ID", "run")
        self.local_path = self.output_dir / f"{self.run_id}.rank{self.rank}.jsonl"
        self.global_path = self.output_dir / f"{self.run_id}.global_max.jsonl"
        self.reference_path = self._resolve_reference_path()
        self.reference_records = []
        self.records = []
        self.interval_start = None

    def _resolve_reference_path(self) -> Path | None:
        raw = os.environ.get("WAM_BENCHMARK_REFERENCE_MANIFEST", "").strip()
        if raw:
            return Path(raw.replace("{rank}", str(self.rank))).expanduser()
        reference_run = os.environ.get("WAM_BENCHMARK_REFERENCE_RUN_ID", "").strip()
        if reference_run:
            return self.output_dir / f"{reference_run}.rank{self.rank}.jsonl"
        return None

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.is_file():
            raise FileNotFoundError(f"WAM benchmark reference manifest is missing: {path}")
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid benchmark manifest line {line_number}: {path}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"benchmark manifest line {line_number} is not an object: {path}")
            records.append(record)
        return records

    def on_train_begin(self, args, state, control, **kwargs):
        if int(getattr(args, "gradient_accumulation_steps", 1)) != 1:
            raise RuntimeError("WAM benchmark requires gradient_accumulation_steps=1")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.local_path.exists() and self.local_path.stat().st_size:
            raise RuntimeError(f"benchmark manifest already exists: {self.local_path}")
        if self.rank == 0 and self.global_path.exists() and self.global_path.stat().st_size:
            raise RuntimeError(f"benchmark timing file already exists: {self.global_path}")
        if self.reference_path is not None:
            self.reference_records = self._read_jsonl(self.reference_path)
            if len(self.reference_records) != self.expected_steps:
                raise RuntimeError(
                    f"reference manifest has {len(self.reference_records)} records, "
                    f"expected {self.expected_steps}: {self.reference_path}"
                )
        self.interval_start = time.perf_counter()

    def _verify_record(self, record: dict) -> None:
        if self.reference_path is None:
            return
        index = int(record["step"])
        if index >= len(self.reference_records):
            raise RuntimeError(
                f"benchmark produced step {index} beyond reference manifest "
                f"length {len(self.reference_records)}"
            )
        expected = self.reference_records[index]
        keys = (
            "step",
            "task",
            "ordinal",
            "episode_index",
            "window_start",
            "episode_indices",
            "window_starts",
        )
        mismatch = {
            key: (expected.get(key), record.get(key))
            for key in keys
            if expected.get(key) != record.get(key)
        }
        if mismatch:
            raise RuntimeError(
                "WAM benchmark reference mismatch at line "
                f"{index}: {mismatch}; reference={self.reference_path}"
            )

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.cuda.current_device())

    def on_step_end(self, args, state, control, **kwargs):
        current = getattr(self.trainer, "_wam_benchmark_current", None)
        if current is None:
            raise RuntimeError("WAM benchmark step metadata was not captured")
        self._sync_cuda()
        end = time.perf_counter()
        if self.interval_start is None:
            self.interval_start = end
        full_step_sec = end - self.interval_start
        profile_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        if profile_device is not None:
            allocated = float(torch.cuda.max_memory_allocated(profile_device))
            reserved = float(torch.cuda.max_memory_reserved(profile_device))
            total = float(torch.cuda.get_device_properties(profile_device).total_memory)
        else:
            allocated = reserved = total = 0.0
        values = torch.tensor(
            [full_step_sec, float(current["compute_sec"]), allocated, reserved, total],
            dtype=torch.float64,
            device=profile_device if profile_device is not None else "cpu",
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
        full_max, compute_max, allocated_max, reserved_max, total_max = values.detach().cpu().tolist()
        record = {
            "run_id": self.run_id,
            "rank": self.rank,
            "step": int(current["step"]),
            "global_step": int(state.global_step),
            "phase": "warmup" if int(current["step"]) < self.warmup_steps else "measured",
            "task": current["task"],
            "ordinal": int(current["ordinal"]),
            "episode_index": int(current["episode_index"]),
            "window_start": int(current["window_start"]),
            "episode_indices": [int(value) for value in current["episode_indices"]],
            "window_starts": [int(value) for value in current["window_starts"]],
            "compute_sec": float(current["compute_sec"]),
            "full_step_sec": float(full_step_sec),
            "global_max_compute_sec": float(compute_max),
            "global_max_full_step_sec": float(full_max),
            "global_max_allocated_bytes": float(allocated_max),
            "global_max_reserved_bytes": float(reserved_max),
            "global_max_total_bytes": float(total_max),
        }
        self._verify_record(record)
        with self.local_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.rank == 0:
            with self.global_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.append(record)
        # Start the next interval only after this record's distributed
        # reduction and JSONL writes.  Otherwise the next step would charge
        # benchmark instrumentation overhead to its data-wait time.
        self.interval_start = time.perf_counter()
        self.trainer._wam_benchmark_current = None

    def on_train_end(self, args, state, control, **kwargs):
        if len(self.records) != self.expected_steps:
            raise RuntimeError(
                f"WAM benchmark ended after {len(self.records)} steps, "
                f"expected {self.expected_steps}"
            )


class LossLoggerCallback(TrainerCallback):
    """Callback that writes per-step loss metrics to a JSONL file for offline analysis."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step}
        for key, value in logs.items():
            if key in {"loss", "dynamics_loss_avg", "action_loss_avg", "learning_rate"}:
                entry[key] = value
            elif key.startswith("task_") and key.endswith(
                ("_loss_avg", "_dynamics_loss_avg", "_action_loss_avg")
            ):
                entry[key] = value
        if len(entry) > 1:  # more than just "step"
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class StageReleaseReadyCallback(TrainerCallback):
    """Release a launch-owned staged checkpoint after every rank loaded it.

    ``on_train_begin`` is invoked only after model/optimizer/DeepSpeed setup.
    The shell owns deletion; this callback merely creates an atomic readiness
    marker after a distributed barrier.
    """

    def __init__(self, marker_path: str):
        self.marker_path = Path(marker_path)

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if state.is_world_process_zero:
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.marker_path.with_name(
                f".{self.marker_path.name}.tmp.{os.getpid()}"
            )
            tmp.write_text("gpu/deepspeed initialization complete\n")
            os.replace(tmp, self.marker_path)
            mprint(f"[stage] all ranks loaded checkpoint; wrote {self.marker_path}")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


class StripOldOptimizerStatesCallback(TrainerCallback):
    """Keep the DeepSpeed optimizer state (global_step*/ dirs, ~238 GB for the
    16B UniW(A)M MoE full fine-tune) only in the NEWEST checkpoint (mode="last"), or
    in NO checkpoint at all (mode="all", weights-only checkpoints everywhere).

    After every save, targeted checkpoints are stripped to weights-only
    (~44 GB): their global_step*/ dirs and the `latest` tag file are deleted.
    mode="last" keeps crash-resume possible from the newest checkpoint;
    mode="all" relies on weights-only warm starts entirely.
    Enabled via cfg.keep_optimizer_states_last_only (+ cfg.strip_all_optimizer_states).
    """

    def __init__(self, mode: str = "last"):
        assert mode in ("last", "all")
        self.mode = mode

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        out = Path(args.output_dir)
        ckpts = sorted(
            (
                d
                for d in out.glob("checkpoint-*")
                if d.is_dir() and d.name.split("-")[-1].isdigit()
            ),
            key=lambda d: int(d.name.split("-")[-1]),
        )
        targets = ckpts if self.mode == "all" else ckpts[:-1]
        for ckpt in targets:
            stripped = False
            for gs in ckpt.glob("global_step*"):
                if gs.is_dir():
                    print(f"[StripOptimizerStates] removing {gs} (weights-only checkpoint kept)")
                    shutil.rmtree(gs, ignore_errors=True)
                    stripped = True
            if stripped:
                latest_tag = ckpt / "latest"
                if latest_tag.exists():
                    latest_tag.unlink()


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self,
        run_name: str,
        exp_cfg_dir: Path | None = None,
        processor_dir: Path | None = None,
        task_ordinal_state: TaskOrdinalState | None = None,
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir
        self.task_ordinal_state = task_ordinal_state
        self._task_start_global_step = 0
        self._task_start_ordinal = int(
            os.environ.get("WAM_TASK_RESUME_MICRO_BATCH_ORDINAL", "0")
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._task_start_global_step = int(state.global_step)

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)
                    continuation_path = exp_cfg_dst / "continuation_state.json"
                    if continuation_path.is_file():
                        from groot.vla.experiment.continuation import (
                            atomic_write_json,
                        )

                        with open(
                            continuation_path, encoding="utf-8"
                        ) as stream:
                            continuation = json.load(stream)
                        continuation["task_micro_batch_ordinal"] = int(
                            self.task_ordinal_state.committed_next_ordinal
                            if self.task_ordinal_state is not None
                            else self._task_start_ordinal
                        )
                        atomic_write_json(
                            continuation_path, continuation
                        )

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class ProfCallback(transformers.TrainerCallback):
    """Callback to manage PyTorch profiler during training.

    Dynamically starts/stops the profiler within a specified session step window.
    After profiling completes, triggers optional S3 upload and removes itself.

    Args:
        profile_dir: Directory to save profile traces
        upload_callback: Optional callback to trigger S3 upload after profiling
        profile_start_step: Session step to start profiling (default: 50)
        profile_end_step: Session step to stop profiling
        warmup_steps: Number of warmup steps for profiler schedule (default: 1)
        active_steps: Number of active profiling steps (default: 5)
        trainer: Trainer instance (required for self-removal after profiling)
        record_shapes: Record tensor shapes in profiler (default: False)
        with_stack: Record Python stack traces (default: True)
        profile_memory: Record memory allocation/deallocation (default: False)
    """

    def __init__(
        self,
        profile_dir,
        upload_callback=None,
        profile_start_step=50,
        profile_end_step=55,
        warmup_steps=1,
        active_steps=5,
        trainer=None,
        record_shapes=False,
        with_stack=True,
        profile_memory=False,
    ):
        self.profile_dir = profile_dir
        self.upload_callback = upload_callback
        self.profile_start_step = profile_start_step
        self.profile_end_step = profile_end_step
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.trainer = trainer
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.upload_triggered = False
        self.starting_global_step = None
        self.session_step = 0
        self.prof = None
        self.profiling_active = False
        self.profiling_complete = False
        self.removed_from_trainer = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Remove callback after upload triggered to eliminate all overhead
        if self.profiling_complete and self.upload_triggered and not self.removed_from_trainer:
            if self.trainer is not None and hasattr(self.trainer, "callback_handler"):
                try:
                    self.trainer.callback_handler.callbacks.remove(self)
                    self.removed_from_trainer = True
                    logging.info(
                        f"Removed ProfCallback from trainer at global step {state.global_step}"
                    )
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Failed to remove ProfCallback: {e}")
            return

        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Record starting global step on first call
        if self.starting_global_step is None:
            self.starting_global_step = state.global_step

        # Calculate session step
        self.session_step = state.global_step - self.starting_global_step

        # Start profiler when we reach the profiling window
        if self.session_step == self.profile_start_step and self.prof is None:
            logging.info(
                f"Starting profiler at global step {state.global_step} (session step {self.session_step})"
            )
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=1,
                ),
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir)),
            )
            self.prof.__enter__()
            self.profiling_active = True

    def on_step_end(self, args, state, control, **kwargs):
        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Recalculate session_step to ensure accuracy
        if self.starting_global_step is not None:
            self.session_step = state.global_step - self.starting_global_step

        # Step profiler if active
        if self.profiling_active and self.prof is not None:
            self.prof.step()

        # Stop profiler when we reach the end of profiling window
        if self.session_step == self.profile_end_step and self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.profiling_active = False

            # Explicitly release profiler resources to minimize CUPTI overhead
            # Combined with TEARDOWN_CUPTI=1 env var for full cleanup
            del self.prof
            self.prof = None

            # Force CUDA synchronization to ensure profiler cleanup completes
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.profiling_complete = True
            logging.info(
                f"Profiler stopped and resources released at global step {state.global_step} "
                f"(session step {self.session_step})"
            )

            # Trigger upload if callback provided
            if self.upload_callback:
                logging.info(f"Triggering upload at global step {state.global_step}...")
                self.upload_callback()

            # Mark as ready for callback removal
            self.upload_triggered = True


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class BaseTrainer(transformers.Trainer):

    def __init__(self, **kwargs):
        # Increase the cache size limit for torch._dynamo to
        # accommodate videos with different numbers of frames.
        torch._dynamo.config.cache_size_limit = 1000

        self.compute_dtype = kwargs.pop("compute_dtype")
        self.output_dir = kwargs.pop("output_dir")
        self.timer = ContextTimer(self)

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.node_rank = int(os.environ.get("NODE_RANK", "0"))

        # Get distributed info
        self.current_step = 0
        self._wam_benchmark_current = None

        # Per-step profiling
        self.enable_profiling = kwargs.pop("enable_profiling", False)
        self.profiling_steps = kwargs.pop("profiling_steps", 5)
        # Pop new ProfCallback config options (handled in create_trainer, not here)
        kwargs.pop("enable_prof_callback", None)
        kwargs.pop("profile_start_step", None)
        kwargs.pop("profile_warmup_steps", None)
        kwargs.pop("profile_active_steps", None)
        kwargs.pop("profile_record_shapes", None)
        kwargs.pop("profile_with_stack", None)
        kwargs.pop("profile_memory", None)
        kwargs.pop("msc_profile_url", None)
        kwargs.pop("profile_delete_after_upload", None)
        if self.enable_profiling:
            # Setup profiling directories
            self.profile_dir = Path(self.output_dir) / "profiling"
            self.memory_profile_dir = self.profile_dir / "memory"
            self.torch_profile_dir = self.profile_dir / "torch"

            self.memory_profile_dir.mkdir(exist_ok=True, parents=True)
            self.torch_profile_dir.mkdir(exist_ok=True, parents=True)

            # Start recording the memory history.
            torch.cuda.memory._record_memory_history(max_entries=100000)

        super().__init__(**kwargs)

        self.loss_queues = {}
        self.loss_queue_size = 10
        self.gradient_conflict_probe = None
        if os.environ.get("WAM_GRADIENT_CONFLICT_PROBE", "0").strip() == "1":
            if int(self.args.gradient_accumulation_steps) != 1:
                raise ValueError(
                    "WAM_GRADIENT_CONFLICT_PROBE requires gradient_accumulation_steps=1"
                )
            if float(self.args.learning_rate) != 0.0:
                raise ValueError(
                    "WAM_GRADIENT_CONFLICT_PROBE requires learning_rate=0 so weights stay fixed"
                )
            from groot.vla.experiment.gradient_conflict_probe import (
                GradientConflictProbe,
            )

            self.gradient_conflict_probe = GradientConflictProbe(
                output_dir=self.args.output_dir,
                sample_size=int(
                    os.environ.get("WAM_GRADIENT_CONFLICT_SAMPLE_SIZE", "16384")
                ),
                seed=int(os.environ.get("WAM_GRADIENT_CONFLICT_SEED", "20260822")),
                log_every=int(
                    os.environ.get("WAM_GRADIENT_CONFLICT_LOG_EVERY", "10")
                ),
            )
            self.gradient_conflict_probe.attach(self.model)

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def _save_checkpoint(self, model, trial):
        """Optionally strip old optimizer states BEFORE writing the new checkpoint.

        With cfg.keep_optimizer_states_last_only, the DeepSpeed optimizer state
        (global_step*/, ~238 GB for the 16B UniW(A)M MoE) of every EXISTING checkpoint
        is deleted before the new save starts, so two full optimizer states never
        coexist on disk (transient peak drops by ~238 GB). Trade-off: while the
        new checkpoint is being written there is NO checkpoint with optimizer
        state on disk; a crash inside this window can only resume weights-only.
        The post-save StripOldOptimizerStatesCallback stays as an idempotent
        safety net for the new checkpoint's predecessors.
        """
        base_cfg = getattr(self, "base_cfg", None)
        if base_cfg is not None and base_cfg.get("keep_optimizer_states_last_only", False):
            if self.args.process_index == 0:
                out = Path(self.args.output_dir)
                for ckpt in sorted(out.glob("checkpoint-*")):
                    if not ckpt.is_dir():
                        continue
                    for gs in ckpt.glob("global_step*"):
                        if gs.is_dir():
                            print(f"[PreSaveStrip] removing {gs} before new checkpoint save")
                            shutil.rmtree(gs, ignore_errors=True)
                    latest_tag = ckpt / "latest"
                    if latest_tag.exists():
                        latest_tag.unlink()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        return super()._save_checkpoint(model, trial)

    def training_step(self, model, inputs, num_items_in_batch=None):
        enable_profile = self.enable_profiling and self.current_step % self.profiling_steps == 0
        enable_wam_benchmark = os.environ.get("WAM_BENCHMARK_MODE", "0").strip() == "1"
        enable_wam_task_profile = (
            os.environ.get("WAM_PROFILE_TASK_STEP", "0").strip() == "1"
            and torch.cuda.is_available()
        )
        profile_device = torch.cuda.current_device() if enable_wam_task_profile else None
        if enable_profile:
            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            )
        else:
            profile_context = contextlib.nullcontext()

        benchmark_cuda = enable_wam_benchmark and torch.cuda.is_available()
        if enable_wam_task_profile or benchmark_cuda:
            profile_device = torch.cuda.current_device()
            torch.cuda.synchronize(profile_device)
            torch.cuda.reset_peak_memory_stats(profile_device)
        if self.gradient_conflict_probe is not None:
            self.gradient_conflict_probe.begin_step()
        start_time = time.time()
        benchmark_compute_start = time.perf_counter() if enable_wam_benchmark else None

        with self.timer.with_label("training_step"), profile_context as prof:
            output = super().training_step(model, inputs)

        if self.gradient_conflict_probe is not None:
            task, ordinal = _wam_profile_batch_metadata(inputs)
            self.gradient_conflict_probe.end_step(
                task=task,
                step=self.current_step,
                ordinal=ordinal,
            )

        if enable_wam_task_profile or benchmark_cuda:
            torch.cuda.synchronize(profile_device)
        if enable_wam_benchmark:
            metadata = _wam_benchmark_batch_metadata(inputs)
            self._wam_benchmark_current = {
                "step": int(self.current_step),
                **metadata,
                "compute_sec": float(time.perf_counter() - benchmark_compute_start),
            }
        time_taken = time.time() - start_time
        print(
            f"Rank {self.global_rank} time taken for training_step {self.current_step}: {time_taken:.2f} seconds"
        )

        if enable_wam_task_profile:
            task, ordinal = _wam_profile_batch_metadata(inputs)
            total_memory = torch.cuda.get_device_properties(profile_device).total_memory
            global_max = torch.tensor(
                [
                    time_taken,
                    float(torch.cuda.max_memory_allocated(profile_device)),
                    float(torch.cuda.max_memory_reserved(profile_device)),
                    float(total_memory),
                ],
                device=profile_device,
                dtype=torch.float64,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX)
            if self.global_rank == 0:
                duration_sec, allocated_bytes, reserved_bytes, total_bytes = global_max.tolist()
                mib = 1024.0**2
                print(
                    "WAM_TASK_STEP_PROFILE "
                    f"micro_step={self.current_step} "
                    f"optimizer_step={self.state.global_step} "
                    f"ordinal={ordinal} "
                    f"task={task} "
                    f"duration_sec={duration_sec:.6f} "
                    f"allocated_peak_mib={allocated_bytes / mib:.2f} "
                    f"reserved_peak_mib={reserved_bytes / mib:.2f} "
                    f"total_mib={total_bytes / mib:.2f} "
                    f"allocated_pct={100.0 * allocated_bytes / total_bytes:.4f} "
                    f"reserved_pct={100.0 * reserved_bytes / total_bytes:.4f}",
                    flush=True,
                )

        if enable_profile:
            trace_path = f"{self.torch_profile_dir}/trace_rank_{self.global_rank}_step_{self.current_step}.json.gz"
            print(f"Rank {self.global_rank} exporting torch profile to {trace_path}")
            prof.export_chrome_trace(trace_path)

            snapshot_path = f"{self.memory_profile_dir}/memory_snapshot_rank_{self.global_rank}_step_{self.current_step}.pickle"
            print(f"Rank {self.global_rank} dumping memory snapshot to {snapshot_path}")
            torch.cuda.memory._dump_snapshot(snapshot_path)

        self.current_step += 1
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self.timer.with_label("model_forward"):
            outputs = model(inputs)
        ### For additional losses, track and log their moving averages
        for key, value in outputs.items():
            if key.endswith("_loss") and key != "loss":
                # Initialize queue if not exists
                if key not in self.loss_queues:
                    self.loss_queues[key] = []

                # Add current loss value to queue
                current_value = value.item() if torch.is_tensor(value) else value
                self.loss_queues[key].append(current_value)

                # Keep only last N values
                if len(self.loss_queues[key]) > self.loss_queue_size:
                    self.loss_queues[key].pop(0)

                # Log average every 10 steps
                if self.current_step % self.loss_queue_size == 0:
                    avg_loss = sum(self.loss_queues[key]) / len(self.loss_queues[key])
                    self.log({f"{key}_avg": avg_loss})

        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # DeepSpeed CPU Adam (ZeRO offload) expects 'bias_correction' in each param group.
            # HuggingFace Trainer's AdamW does not set it, causing KeyError in cpu_adam.step().
            if getattr(self.args, "deepspeed", None):
                for group in self.optimizer.param_groups:
                    group.setdefault("bias_correction", True)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):

        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.base_cfg.save_lora_only:
            # Save only the trainable parameters
            train_key = [k for k, v in self.model.named_parameters() if v.requires_grad]
            lora_state_dict = {k: v for k, v in self.model.state_dict().items() if k in train_key}
            state_dict = lora_state_dict

        if self.args.should_save:
            ret = self.model.save_pretrained(output_dir, state_dict=state_dict)

            # can separately save the VLM model for downstream evalualtion
            if self.base_cfg.save_llm:
                llm_output_dir = os.path.join(output_dir, "llm")
                self.model.backbone.model.save_pretrained(llm_output_dir)

            if self.base_cfg.save_value_model:
                assert hasattr(
                    self.model.action_head, "value_model"
                ), f"Value model not found in action head: {type(self.model.action_head)}"
                value_model_output_dir = os.path.join(output_dir, "value_model")
                self.model.action_head.value_model.save_pretrained(value_model_output_dir)

            return ret

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def num_examples(self, dataloader: DataLoader) -> int:
        """Keep Trainer sample accounting correct for pre-batched datasets."""
        dataset = getattr(dataloader, "dataset", None)
        if isinstance(
            dataset,
            (BucketedBatchIterableDataset, TaskAwareBatchDataset),
        ):
            return len(dataset.dataset)
        return super().num_examples(dataloader)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if not isinstance(train_dataset, (ShardedLeRobotMixtureDataset)):
            return super().get_train_dataloader()

        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            continuation_seed = os.environ.get("WAM_DATA_SCHEDULE_SEED")
            new_seed = (
                int(continuation_seed)
                if continuation_seed is not None
                else train_dataset.seed + curr_global_step
            )
            train_dataset.reset_seed(new_seed)
            print(f"Resetting resumed data schedule seed to {new_seed}.")

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        wam_task_ratios = getattr(data_collator, "wam_task_ratios", None)
        hydra_single_task = getattr(data_collator, "wam_task", None)
        env_single_task = os.environ.get("WAM_MOE_TASK")
        if wam_task_ratios and hydra_single_task:
            raise ValueError(
                "Hydra config cannot set both wam_task_ratios and wam_task"
            )
        if wam_task_ratios:
            wam_single_task = None
        else:
            if hydra_single_task and env_single_task and (
                canonical_wam_task_name(hydra_single_task)
                != canonical_wam_task_name(env_single_task)
            ):
                raise ValueError(
                    "WAM_MOE_TASK and Hydra wam_task disagree: "
                    f"{env_single_task!r} != {hydra_single_task!r}"
                )
            wam_single_task = hydra_single_task or env_single_task
            if not wam_single_task:
                raise ValueError(
                    "single-task WAM training requires either Hydra wam_task "
                    "or WAM_MOE_TASK"
                )
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        benchmark_mode = os.environ.get("WAM_BENCHMARK_MODE", "0").strip() == "1"
        benchmark_sequence = None
        benchmark_start_ordinal = int(
            os.environ.get("WAM_BENCHMARK_TASK_START_ORDINAL", "0")
        )
        if benchmark_mode:
            benchmark_warmup_steps = int(
                os.environ.get("WAM_BENCHMARK_WARMUP_STEPS", "20")
            )
            benchmark_measured_steps = int(
                os.environ.get("WAM_BENCHMARK_MEASURED_STEPS", "100")
            )
            raw_tasks = os.environ.get(
                "WAM_BENCHMARK_TASK_ORDER",
                "joint_dreamzero,idm_lingbot_va,action_only_fast_wam,"
                "action_only_gigaworld_policy,ac_wm",
            )
            benchmark_sequence = wam_benchmark_task_sequence(
                warmup_steps=benchmark_warmup_steps,
                measured_steps=benchmark_measured_steps,
                tasks=[task for task in raw_tasks.split(",") if task.strip()],
            )
            expected_steps = benchmark_warmup_steps + benchmark_measured_steps
            configured_total = int(
                os.environ.get("WAM_BENCHMARK_TOTAL_STEPS", str(expected_steps))
            )
            if configured_total != expected_steps:
                raise ValueError(
                    "WAM_BENCHMARK_TOTAL_STEPS must equal warmup+measured steps "
                    f"({expected_steps}), got {configured_total}"
                )
        task_aware_batching = bool(wam_task_ratios or wam_single_task or benchmark_sequence)
        dataloader_workers = self.args.dataloader_num_workers
        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": dataloader_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if task_aware_batching:
            if getattr(self.args, "include_tokens_per_second", False):
                raise RuntimeError(
                    "include_tokens_per_second is incompatible with exact "
                    "task-ordinal acknowledgement because Trainer pre-iterates "
                    "the train DataLoader before training"
                )
            resume_value = os.environ.get(
                "WAM_TASK_RESUME_MICRO_BATCH_ORDINAL"
            )
            if resume_value is None and benchmark_sequence is not None:
                resume_micro_batch_ordinal = benchmark_start_ordinal
            elif resume_value is None and curr_global_step > 0:
                raise RuntimeError(
                    "resumed task-aware training requires the exact "
                    "WAM_TASK_RESUME_MICRO_BATCH_ORDINAL saved with the "
                    "checkpoint"
                )
            else:
                resume_micro_batch_ordinal = int(resume_value or 0)
            train_dataset = TaskAwareBatchDataset(
                train_dataset,
                batch_size=self._train_batch_size,
                wam_task_ratios=wam_task_ratios,
                wam_single_task=wam_single_task,
                wam_task_sequence_seed=int(
                    os.environ.get("WAM_TASK_SEQUENCE_SEED", str(0x5741_4D00))
                ),
                wam_task_sequence=benchmark_sequence,
                wam_task_sequence_start_ordinal=benchmark_start_ordinal,
            )
            world_size = max(int(getattr(self.args, "world_size", 1)), 1)
            if benchmark_sequence is not None:
                # Bound the sampler itself so worker prefetch cannot request
                # an ordinal beyond the finite benchmark task sequence.
                task_batches_per_epoch = len(benchmark_sequence)
            else:
                rank_local_samples = int(len(train_dataset.dataset)) // world_size
                task_batches_per_epoch = rank_local_samples // self._train_batch_size
            if task_batches_per_epoch <= 0:
                raise RuntimeError(
                    "task-aware source has fewer samples than one micro-batch"
                )
            task_ordinal_state = TaskOrdinalState(
                resume_micro_batch_ordinal
            )
            self._task_ordinal_state = task_ordinal_state
            callback_handler = getattr(self, "callback_handler", None)
            for callback in getattr(callback_handler, "callbacks", ()):
                if isinstance(callback, CheckpointFormatCallback):
                    callback.task_ordinal_state = task_ordinal_state
            dataloader_params["batch_size"] = None
            dataloader_params["sampler"] = TaskOrdinalSampler(
                resume_micro_batch_ordinal,
                num_batches=task_batches_per_epoch,
                state=task_ordinal_state,
            )
            print(
                "Enabled main-process task-ordinal scheduling with "
                f"micro_batch_size={self._train_batch_size} "
                f"batches_per_iterator={task_batches_per_epoch} "
                f"resume_micro_batch={resume_micro_batch_ordinal}"
            )
        elif self._train_batch_size > 1:
            # The source stream contains variable-length samples. Pre-batch
            # inside each worker by full tensor shape, then disable DataLoader's
            # automatic batching so samples from different workers/buckets are
            # never re-packed together. Keep _train_batch_size unchanged: it is
            # the real per-device micro-batch used by Trainer and DeepSpeed.
            train_dataset = BucketedBatchIterableDataset(
                train_dataset,
                batch_size=self._train_batch_size,
            )
            dataloader_params["batch_size"] = None
            print(
                "Enabled shape-bucketed pre-batching with "
                f"micro_batch_size={self._train_batch_size}"
            )
        else:
            dataloader_params["batch_size"] = self._train_batch_size
        # persistent_workers is only valid when num_workers > 0 (PyTorch raises otherwise)
        if dataloader_workers > 0:
            dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers

        if task_aware_batching:
            return TaskAcknowledgingDataLoader(
                train_dataset,
                task_ordinal_state=task_ordinal_state,
                **dataloader_params,
            )

        return DataLoader(train_dataset, **dataloader_params)


class BaseExperiment(ABC):
    def __init__(self, cfg: DictConfig):
        # assert cfg.save_steps == 500, "save_steps must be 500 for standarized evaluation"
        assert cfg.max_steps > 0, "max_steps must be > 0 for standarized evaluation"
        # Upstream required save_total_limit >= 5 for its standardized eval sweep.
        # Our full-finetune MoE checkpoints are ~281 GB each, so small limits are
        # legitimate; only require a sane positive value.
        assert cfg.save_total_limit >= 1, "save_total_limit must be >= 1"

        if cfg.load_from_yaml is not None:
            # Override the default config with the loaded config.
            loaded_cfg = OmegaConf.load(cfg.load_from_yaml)
            cfg = loaded_cfg  # overwrite

        # Check if evaluation transforms are valid.
        assert cfg.transforms is not None, "Evaluation transforms are not provided."
        for tag, transform_cfg in cfg.transforms.items():
            try:
                # Check if the tag is a valid EmbodimentTag
                _ = EmbodimentTag(tag)
                # Check if the transform is a valid ComposedModalityTransform
                transform = instantiate(transform_cfg)
                assert isinstance(transform, ComposedModalityTransform), f"{transform=}"
            except Exception as e:
                raise ValueError(f"Evaluation transform {tag} is invalid: {e}")

        # Instantiate the training arguments.
        cfg.training_args.output_dir = cfg.training_args.output_dir.rstrip("/")
        logical_output_dir = cfg.training_args.output_dir.rstrip("/")
        cfg.training_args.run_name = logical_output_dir.split("/")[-1]
        print(f"Run name: {cfg.training_args.run_name}")
        training_args = instantiate(cfg.training_args)
        set_seed(training_args.seed)

        # Set the environment variables for wandb.
        if "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if "WANDB_RUN_ID" not in os.environ:
            runtime_id = os.environ.get("RUNTIME_ID", None)
            """If a RUNTIME_ID is available in the environment, we use it as the wandb id,
            which will allow to display the evaluation results and the training results
            in the same wandb run. Otherwise, we create a new run."""
            if runtime_id:
                os.environ["WANDB_RUN_ID"] = runtime_id
        os.environ["WANDB_DIR"] = training_args.output_dir

        # Create the experiment config directory.
        output_dir = Path(training_args.output_dir)
        exp_cfg_dir = output_dir / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        wandb_config_file = output_dir / "wandb_config.json"
        with open(wandb_config_file, "w") as f:
            json.dump(
                {
                    "project": os.environ.get("WANDB_PROJECT", ""),
                    "run_id": os.environ.get("WANDB_RUN_ID", ""),
                },
                f,
            )

        # Check if we are resuming training.
        explicit_full_resume = os.environ.get("WAM_FULL_RESUME_CHECKPOINT", "").strip()
        if explicit_full_resume:
            resume_path, continue_training = explicit_full_resume, True
            print(f"Using staged full-resume checkpoint: {resume_path}")
        else:
            resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
        if not continue_training:
            print(f"Models is ready under {training_args.output_dir}. Skip training.")
            exit(0)
        if resume_path:
            print(f"Resuming training from {resume_path}")
            resume_from_checkpoint = resume_path if explicit_full_resume else True
        else:
            # First time training.
            resume_from_checkpoint = False

        # Instantiate the model.
        model = self.create_model(cfg, training_args)

        if hasattr(model.action_head, "max_steps"):
            model.action_head.max_steps = cfg.max_steps

        # Make sure model_dtype and training_args dtype are compatible.
        compute_dtype = dtype_from_string(model.config.model_dtype)

        # Create the train dataset.
        # Dump the metadata; necessary for policy to normalize the input and unnormalize the output
        train_dataset = self.create_train_dataset(cfg, model)
        print("Using dataset:")
        print(train_dataset)
        assert (
            train_dataset.merged_metadata is not None
        ), "You must set metadata_config.merge=true in order to save the metadata."

        metadata_save_path = exp_cfg_dir / "metadata.json"
        U.json_dump(
            {k: v.model_dump(mode="json") for k, v in train_dataset.merged_metadata.items()},
            metadata_save_path,
            indent=4,
        )
        print("Successfully dumped metadata")

        val_dataset = self.create_val_dataset(cfg, model)
        data_collator = self.create_data_collator(cfg, model)
        trainer = self.create_trainer(
            cfg=cfg,
            exp_cfg_dir=exp_cfg_dir,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        required_effective_fields = (
            "unified_joint_consumption",
            "unified_chunkwise_attention",
            "force_token_group_experiment",
            "moe_routing_granularity",
        )
        if all(hasattr(model.action_head, field) for field in required_effective_fields):
            effective_config = _build_wam_effective_config(model, training_args)
            _attach_wam_effective_config(model, cfg, effective_config)
            OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)
            if training_args.process_index == 0:
                from groot.vla.experiment.checkpoint_staging import (
                    write_effective_config,
                )

                effective_path = write_effective_config(
                    training_args.output_dir, effective_config
                )
                print(f"[effective-config] wrote {effective_path}")
        self.cfg = cfg
        self.exp_cfg_dir = exp_cfg_dir
        self.training_args = training_args
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.trainer = trainer

    def create_model(self, cfg, training_args):
        from groot.vla.model.dreamzero.checkpoint_loading import (
            build_from_complete_checkpoint,
            fast_full_checkpoint_loading_enabled,
        )

        checkpoint = cfg.pretrained_model_path
        skip_components = bool(
            OmegaConf.select(
                cfg, "action_head_cfg.config.skip_component_loading", default=False
            )
        )
        fast_full_load = (
            checkpoint is not None
            and skip_components
            and fast_full_checkpoint_loading_enabled()
        )
        if fast_full_load:
            mprint(
                "Loading complete training warm-start with meta+assign "
                f"from: {checkpoint}"
            )
            model = build_from_complete_checkpoint(
                lambda: instantiate(cfg.model), checkpoint, log=mprint
            )
        else:
            model = instantiate(cfg.model)

        if checkpoint is not None and not fast_full_load:
            mprint(f"Loading pretrained weights from: {checkpoint}")
            import json, gc
            from safetensors.torch import load_file

            ckpt_dir = checkpoint
            safetensors_index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
            safetensors_path = os.path.join(ckpt_dir, "model.safetensors")

            if os.path.exists(safetensors_index_path):
                with open(safetensors_index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in sorted(set(index["weight_map"].values())):
                    shard_path = os.path.join(ckpt_dir, shard_file)
                    mprint(f"Loading shard: {shard_path}")
                    shard_state_dict = load_file(shard_path)
                    model.load_state_dict(shard_state_dict, strict=False)
                    del shard_state_dict
                    gc.collect()
            elif os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(
                    f"No weights found at '{ckpt_dir}'. "
                    "Expected 'model.safetensors' or 'model.safetensors.index.json'."
                )
            mprint("Successfully loaded pretrained weights")

        if (checkpoint is not None
                and hasattr(model, 'action_head')
                and hasattr(model.action_head, 'inject_lora_after_loading')
                and model.action_head.config.defer_lora_injection):
            model.action_head.inject_lora_after_loading()

        logical_output_dir = training_args.output_dir
        model.config.resume_path = model.config._name_or_path = logical_output_dir
        mprint(f"{model}\n")
        return model

    def create_train_dataset(self, cfg, model):
        assert torch.distributed.is_initialized()
        train_dataset = instantiate(cfg.train_dataset)
        return train_dataset

    def create_val_dataset(self, cfg, model):
        return None

    def create_data_collator(self, cfg, model):
        return instantiate(cfg.data_collator)

    def create_trainer(
        self,
        cfg,
        exp_cfg_dir,
        model,
        training_args,
        train_dataset,
        val_dataset,
        data_collator,
        compute_dtype,
    ):
        # Set the gradient accumulation steps.
        if cfg.global_batch_size is not None:
            global_bs = cfg.global_batch_size
            bs = training_args.per_device_train_batch_size
            grad_acc = compute_grad_accum_to_match_global_bs(global_bs, bs)
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_bs}, set gradient accumulation steps to {grad_acc}"
            )
        elif cfg.raise_error_if_global_batch_size_not_set:
            raise ValueError(
                "global_batch_size is not set. To ensure the scripts can be reproduced regardless of the number of nodes used, please set this."
            )
        else:
            warnings.warn(
                "global_batch_size is not set. This is fine for debugging, but please set this for real experiments."
            )

        # Instantiate the partial trainer.
        trainer_partial = instantiate(
            cfg.trainer,
            model=model,
            output_dir=training_args.output_dir,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_dtype=compute_dtype,
        )

        # Fully instantiate the trainer with dataclasses instances.
        trainer = trainer_partial(data_collator=data_collator, args=training_args)
        trainer.base_cfg = cfg
        train_dl_len = len(trainer.get_train_dataloader())
        eval_dl_len = (
            len(trainer.get_eval_dataloader()) if val_dataset is not None else "no eval dataloader"
        )

        # Save the total training steps in the config.
        with open_dict(cfg):
            cfg.total_training_steps = train_dl_len * cfg.training_args.num_train_epochs

        # Save config.
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        run_name = cfg.training_args.get("run_name", None)
        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=exp_cfg_dir)
        trainer.add_callback(ckpt_format_callback)

        loss_log_path = str(Path(training_args.output_dir) / "loss_log.jsonl")
        trainer.add_callback(LossLoggerCallback(output_path=loss_log_path))

        if os.environ.get("WAM_BENCHMARK_MODE", "0").strip() == "1":
            benchmark_warmup_steps = int(
                os.environ.get("WAM_BENCHMARK_WARMUP_STEPS", "20")
            )
            benchmark_measured_steps = int(
                os.environ.get("WAM_BENCHMARK_MEASURED_STEPS", "100")
            )
            benchmark_total_steps = int(
                os.environ.get(
                    "WAM_BENCHMARK_TOTAL_STEPS",
                    str(benchmark_warmup_steps + benchmark_measured_steps),
                )
            )
            if benchmark_total_steps != benchmark_warmup_steps + benchmark_measured_steps:
                raise ValueError(
                    "WAM_BENCHMARK_TOTAL_STEPS must equal warmup+measured steps"
                )
            benchmark_dir = os.environ.get(
                "WAM_BENCHMARK_PROFILE_DIR",
                str(Path(training_args.output_dir) / "benchmark"),
            )
            trainer.add_callback(
                WamBenchmarkStepCallback(
                    trainer,
                    expected_steps=benchmark_total_steps,
                    warmup_steps=benchmark_warmup_steps,
                    output_dir=benchmark_dir,
                )
            )
            mprint(
                "WAM benchmark capture enabled: "
                f"steps={benchmark_total_steps}, output={benchmark_dir}"
            )

        stage_ready_marker = os.environ.get("WAM_STAGE_GPU_READY_MARKER", "").strip()
        if stage_ready_marker:
            trainer.add_callback(StageReleaseReadyCallback(stage_ready_marker))
            mprint(
                "Staged-checkpoint release callback enabled; temporary weights "
                f"will be releasable at on_train_begin: {stage_ready_marker}"
            )

        # Optionally keep DeepSpeed optimizer states only in the newest checkpoint
        # (older rotated checkpoints become weights-only), or strip them from ALL
        # checkpoints (strip_all_optimizer_states). See the callback docstring.
        if cfg.get("strip_all_optimizer_states", False):
            trainer.add_callback(StripOldOptimizerStatesCallback(mode="all"))
            mprint("strip_all_optimizer_states=true: ALL checkpoints are weights-only")
        elif cfg.get("keep_optimizer_states_last_only", False):
            trainer.add_callback(StripOldOptimizerStatesCallback(mode="last"))
            mprint("keep_optimizer_states_last_only=true: older checkpoints keep weights only")

        # Add profiling callback (local profiling only, no S3 upload)
        # Local: {output_dir}/profiling/rank_{id}/*.pt.trace.json
        if cfg.trainer.get("enable_prof_callback", False):
            output_dir = Path(training_args.output_dir)
            global_rank = int(os.environ.get("RANK", "0"))

            # Get profiling configuration from trainer config
            profile_start_step = cfg.trainer.get("profile_start_step", 50)
            profile_warmup_steps = cfg.trainer.get("profile_warmup_steps", 1)
            profile_active_steps = cfg.trainer.get("profile_active_steps", 5)
            profile_record_shapes = cfg.trainer.get("profile_record_shapes", False)
            profile_with_stack = cfg.trainer.get(
                "profile_with_stack", False
            )  # Default False to match omni (stack traces add significant file size)
            profile_memory = cfg.trainer.get("profile_memory", False)

            # Calculate end step
            profile_end_step = profile_start_step + profile_warmup_steps + profile_active_steps - 1

            # Setup profile directory with rank subdirectory: {output_dir}/profiling/rank_{id}/
            profile_dir = output_dir / "profiling" / f"rank_{global_rank}"
            profile_dir.mkdir(parents=True, exist_ok=True)

            mprint(
                f"Profiling enabled: steps {profile_start_step}-{profile_end_step}, "
                f"saving to {profile_dir}"
            )

            # Add ProfCallback
            trainer.add_callback(
                ProfCallback(
                    profile_dir=profile_dir,
                    upload_callback=None,
                    profile_start_step=profile_start_step,
                    profile_end_step=profile_end_step,
                    warmup_steps=profile_warmup_steps,
                    active_steps=profile_active_steps,
                    trainer=trainer,
                    record_shapes=profile_record_shapes,
                    with_stack=profile_with_stack,
                    profile_memory=profile_memory,
                )
            )

        mprint(
            f"train dataloader length: {train_dl_len}\n"
            f"eval dataloader length: {eval_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        # Start training.
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        # Respect save_strategy=no (e.g. speed benchmarks): skip the final full
        # model dump, which writes the whole (44 GB+) model to output_dir.
        save_strategy = str(getattr(self.training_args, "save_strategy", "steps")).lower()
        if save_strategy.endswith("no"):
            return
        final_output_dir = self.training_args.output_dir
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=self.trainer, output_dir=final_output_dir
        )
