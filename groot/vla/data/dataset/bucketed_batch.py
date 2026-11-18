"""Bucketed pre-batching for IterableDataset streams with variable tensor shapes.

Wraps any ``IterableDataset`` that yields ``dict`` samples and groups them by a
full stackable signature (every ``np.ndarray`` / ``torch.Tensor`` key path,
shape, and dtype, walked recursively). Each worker independently accumulates
per-bucket buffers and yields fixed-size ``list[dict]`` micro-batches.

Designed for ``DataLoader(..., batch_size=None)`` so the loader does not re-pack
samples across buckets. Source rank/worker sharding is left entirely to the
wrapped dataset (this wrapper does not re-shard).
"""

from __future__ import annotations

import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

ShapeLeaf = Tuple[str, Tuple[int, ...], str]
ShapeSignature = Tuple[ShapeLeaf, ...]

_WAM_TASK_ALIASES = {
    "joint_dreamzero": "joint_dreamzero",
    "joint": "joint_dreamzero",
    "dreamzero": "joint_dreamzero",
    "idm_lingbot_va": "idm_lingbot_va",
    "lingbot_va": "idm_lingbot_va",
    "idm": "idm_lingbot_va",
    "action_only_fast_wam": "action_only_fast_wam",
    "fast_wam": "action_only_fast_wam",
    "fastwam": "action_only_fast_wam",
    "action_only_gigaworld_policy": "action_only_gigaworld_policy",
    "gigaworld_policy": "action_only_gigaworld_policy",
    "giga": "action_only_gigaworld_policy",
    "ac_wm": "ac_wm",
    "world_model": "ac_wm",
    "wm": "ac_wm",
    "embodied_world_model": "ac_wm",
}

WAM_BENCHMARK_DEFAULT_TASKS = (
    "joint_dreamzero",
    "idm_lingbot_va",
    "action_only_fast_wam",
    "action_only_gigaworld_policy",
    "ac_wm",
)


def wam_benchmark_task_sequence(
    *,
    warmup_steps: int = 20,
    measured_steps: int = 100,
    tasks: Sequence[str] = WAM_BENCHMARK_DEFAULT_TASKS,
) -> List[str]:
    """Build the fixed five-task sequence used by the benchmark harness.

    The task order is deliberately explicit and independent of the normal
    weighted-random scheduler.  Each phase contains an equal number of
    occurrences of every task, so the first 20 steps contain four per task
    and the measured 100 steps contain twenty per task.
    """
    warmup_steps = int(warmup_steps)
    measured_steps = int(measured_steps)
    canonical_tasks = [canonical_wam_task_name(task) for task in tasks]
    if not canonical_tasks or len(set(canonical_tasks)) != len(canonical_tasks):
        raise ValueError("benchmark task order must contain each task exactly once")
    if warmup_steps <= 0 or measured_steps <= 0:
        raise ValueError("benchmark warmup/measured steps must be positive")
    if warmup_steps % len(canonical_tasks) or measured_steps % len(canonical_tasks):
        raise ValueError(
            "benchmark warmup and measured steps must divide evenly across tasks"
        )
    return (
        canonical_tasks * (warmup_steps // len(canonical_tasks))
        + canonical_tasks * (measured_steps // len(canonical_tasks))
    )


def canonical_wam_task_name(task: str) -> str:
    key = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _WAM_TASK_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported WAM task: {task!r}") from exc


def normalize_wam_task_ratios(
    ratios: Mapping[str, float],
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for name, weight in ratios.items():
        canonical = canonical_wam_task_name(name)
        value = float(weight)
        if value < 0:
            raise ValueError("wam task ratios cannot contain negative weights")
        normalized[canonical] = normalized.get(canonical, 0.0) + value
    if not normalized or sum(normalized.values()) <= 0:
        raise ValueError("wam task ratios must contain a positive total weight")
    return normalized


def wam_task_for_micro_batch(
    ratios: Mapping[str, float],
    seed: int,
    global_micro_batch_ordinal: int,
) -> str:
    """Return the deterministic homogeneous task for one micro-batch."""
    if global_micro_batch_ordinal < 0:
        raise ValueError("global_micro_batch_ordinal must be non-negative")
    normalized = normalize_wam_task_ratios(ratios)
    names = list(normalized)
    weights = list(normalized.values())
    return random.Random(int(seed) + int(global_micro_batch_ordinal)).choices(
        names, weights=weights, k=1
    )[0]


class TaskOrdinalState:
    """Main-process acknowledgement state for consumed task micro-batches."""

    def __init__(self, committed_next_ordinal: int = 0) -> None:
        if committed_next_ordinal < 0:
            raise ValueError("committed_next_ordinal must be non-negative")
        self.committed_next_ordinal = int(committed_next_ordinal)

    def commit(self, ordinal: int) -> None:
        value = int(ordinal)
        expected = self.committed_next_ordinal
        if value != expected:
            raise RuntimeError(
                "task ordinal consumption is not contiguous: "
                f"expected {expected}, got {value}"
            )
        self.committed_next_ordinal = value + 1


class TaskOrdinalSampler(Sampler[int]):
    """Issue requests from the last *consumed* ordinal.

    ``num_batches`` is the number of requests in one finite iterator. Reusing
    the sampler after abandoning prefetched requests starts again from the
    state's committed ordinal.
    """

    def __init__(
        self,
        resume_micro_batch_ordinal: int = 0,
        *,
        num_batches: Optional[int] = None,
        state: Optional[TaskOrdinalState] = None,
    ) -> None:
        if resume_micro_batch_ordinal < 0:
            raise ValueError("resume_micro_batch_ordinal must be non-negative")
        if num_batches is not None and num_batches <= 0:
            raise ValueError("num_batches must be positive when provided")
        self.state = state or TaskOrdinalState(resume_micro_batch_ordinal)
        if self.state.committed_next_ordinal != int(resume_micro_batch_ordinal):
            raise ValueError(
                "sampler resume ordinal disagrees with acknowledgement state"
            )
        self.num_batches = None if num_batches is None else int(num_batches)

    def __iter__(self) -> Iterator[int]:
        remaining = self.num_batches
        next_request = self.state.committed_next_ordinal
        while remaining is None or remaining > 0:
            ordinal = next_request
            next_request += 1
            if remaining is not None:
                remaining -= 1
            yield ordinal

    def __len__(self) -> int:
        if self.num_batches is None:
            raise TypeError("infinite TaskOrdinalSampler has no length")
        return self.num_batches


class TaskAwareBatchDataset(Dataset):
    """Fetch one homogeneous micro-batch for an explicit task ordinal.

    The DataLoader sampler runs in the main process and sends the exact ordinal
    to ``__getitem__``. A worker therefore knows the task before fetching the
    first 33/49-frame sample and never derives consumer order from a local
    counter or worker round-robin assumptions.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int,
        *,
        wam_task_ratios: Optional[Mapping[str, float]] = None,
        wam_single_task: Optional[str] = None,
        wam_task_sequence_seed: int = 0x5741_4D00,
        wam_task_sequence: Optional[Sequence[str]] = None,
        wam_task_sequence_start_ordinal: int = 0,
    ) -> None:
        if not isinstance(dataset, IterableDataset):
            raise TypeError(
                f"dataset must be an IterableDataset, got {type(dataset).__name__}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if wam_task_ratios and wam_single_task:
            raise ValueError("configure wam_task_ratios or wam_single_task, not both")
        if not wam_task_ratios and not wam_single_task and wam_task_sequence is None:
            raise ValueError("task-aware batching requires task ratios or one task")
        if not hasattr(dataset, "set_wam_task"):
            raise TypeError(
                "task-aware batching requires the source dataset to implement "
                "set_wam_task(task)"
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.wam_task_ratios = (
            normalize_wam_task_ratios(wam_task_ratios)
            if wam_task_ratios
            else None
        )
        self.wam_single_task = (
            canonical_wam_task_name(wam_single_task)
            if wam_single_task
            else None
        )
        self.wam_task_sequence_seed = int(wam_task_sequence_seed)
        self.wam_task_sequence = (
            [canonical_wam_task_name(task) for task in wam_task_sequence]
            if wam_task_sequence is not None
            else None
        )
        if self.wam_task_sequence is not None and not self.wam_task_sequence:
            raise ValueError("wam_task_sequence must not be empty")
        self.wam_task_sequence_start_ordinal = int(wam_task_sequence_start_ordinal)
        if self.wam_task_sequence_start_ordinal < 0:
            raise ValueError("wam_task_sequence_start_ordinal must be non-negative")
        self._source: Optional[Iterator[dict]] = None
        self._buckets: Dict[Tuple[str, ShapeSignature], List[dict]] = defaultdict(list)
        self._cycle_count = 0
        self._samples_in_cycle = 0

    def task_for_ordinal(self, ordinal: int) -> str:
        if ordinal < 0:
            raise ValueError("task ordinal must be non-negative")
        if self.wam_task_sequence is not None:
            sequence_index = ordinal - self.wam_task_sequence_start_ordinal
            if sequence_index < 0 or sequence_index >= len(self.wam_task_sequence):
                raise RuntimeError(
                    "benchmark task sequence exhausted or ordinal is before its "
                    f"start: ordinal={ordinal}, start="
                    f"{self.wam_task_sequence_start_ordinal}, "
                    f"length={len(self.wam_task_sequence)}"
                )
            return self.wam_task_sequence[sequence_index]
        if self.wam_single_task is not None:
            return self.wam_single_task
        return wam_task_for_micro_batch(
            self.wam_task_ratios,
            self.wam_task_sequence_seed,
            ordinal,
        )

    def _next_source_sample(self) -> Mapping[str, Any]:
        if self._source is None:
            self._source = iter(self.dataset)
        try:
            sample = next(self._source)
            self._samples_in_cycle += 1
            return sample
        except StopIteration:
            # max_steps training may cross the finite source schedule. Cycle
            # deterministically; fail instead of spinning if one whole source
            # iterator cannot yield a usable sample.
            if self._samples_in_cycle <= 0:
                raise RuntimeError(
                    "task-aware source yielded no samples in a complete cycle"
                )
            self._cycle_count += 1
            if self._cycle_count == 1:
                warnings.warn(
                    "task-aware source exhausted before the requested "
                    "max_steps; cycling worker-local source",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._samples_in_cycle = 0
            self._source = iter(self.dataset)
            try:
                sample = next(self._source)
                self._samples_in_cycle = 1
                return sample
            except StopIteration as exc:
                raise RuntimeError(
                    "task-aware source yielded no samples in a complete cycle"
                ) from exc

    def __getitem__(self, ordinal: int) -> List[dict]:
        global_ordinal = int(ordinal)
        task = self.task_for_ordinal(global_ordinal)
        self.dataset.set_wam_task(task)
        candidate = next(
            (
                key
                for key in sorted(self._buckets, key=lambda value: repr(value))
                if key[0] == task
                and len(self._buckets[key]) >= self.batch_size
            ),
            None,
        )
        while candidate is None:
            raw_sample = self._next_source_sample()
            if not isinstance(raw_sample, Mapping):
                raise TypeError(
                    "Expected dict-like sample from source dataset, got "
                    f"{type(raw_sample).__name__}"
                )
            sample = dict(raw_sample)
            existing = sample.get("wam_task")
            if (
                existing is not None
                and canonical_wam_task_name(existing) != task
            ):
                raise ValueError(
                    f"source wam_task={existing!r} disagrees with "
                    f"scheduled task={task!r}"
                )
            sample["wam_task"] = task
            signature = shape_signature(sample)
            key = (task, signature)
            self._buckets[key].append(sample)
            if len(self._buckets[key]) >= self.batch_size:
                candidate = key

        batch = self._buckets[candidate][: self.batch_size]
        del self._buckets[candidate][: self.batch_size]
        for sample in batch:
            sample["wam_task_ordinal"] = global_ordinal
            sample["wam_task_sequence_seed"] = self.wam_task_sequence_seed
        return batch

    def __len__(self) -> int:
        return int(math.floor(len(self.dataset) / float(self.batch_size)))


class TaskAcknowledgingDataLoader(DataLoader):
    """Commit an ordinal only when its batch is delivered to the consumer."""

    def __init__(
        self,
        *args,
        task_ordinal_state: TaskOrdinalState,
        **kwargs,
    ) -> None:
        self.task_ordinal_state = task_ordinal_state
        super().__init__(*args, **kwargs)

    @staticmethod
    def _batch_ordinal(batch: Any) -> int:
        if isinstance(batch, Mapping):
            values = batch.get("wam_task_ordinal")
        elif (
            isinstance(batch, Sequence)
            and batch
            and all(isinstance(sample, Mapping) for sample in batch)
        ):
            values = [sample.get("wam_task_ordinal") for sample in batch]
        else:
            values = None
        if values is None:
            raise RuntimeError(
                "task-aware collator must preserve wam_task_ordinal"
            )
        if torch.is_tensor(values):
            flattened = values.detach().cpu().reshape(-1).tolist()
        elif isinstance(values, np.ndarray):
            flattened = values.reshape(-1).tolist()
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            flattened = list(values)
        else:
            flattened = [values]
        ordinals = {int(value) for value in flattened}
        if len(ordinals) != 1:
            raise RuntimeError(
                f"one micro-batch contains multiple task ordinals: {ordinals}"
            )
        return next(iter(ordinals))

    def __iter__(self):
        for batch in super().__iter__():
            self.task_ordinal_state.commit(self._batch_ordinal(batch))
            yield batch


def _is_stackable_array(value: Any) -> bool:
    return isinstance(value, (np.ndarray, torch.Tensor))


def collect_shape_leaves(
    value: Any,
    prefix: str = "",
) -> List[ShapeLeaf]:
    """Recursively collect (key_path, shape, dtype) for array/Tensor leaves.

    Nested ``dict`` / ``list`` / ``tuple`` containers are walked. Non-array
    scalars/strings/etc. are ignored (they do not participate in ``np.stack``).
    """
    leaves: List[ShapeLeaf] = []
    if isinstance(value, Mapping):
        for key in sorted(value.keys(), key=lambda k: str(k)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(collect_shape_leaves(value[key], child_prefix))
        return leaves
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            leaves.extend(collect_shape_leaves(child, child_prefix))
        return leaves
    if _is_stackable_array(value):
        shape = tuple(int(d) for d in value.shape)
        leaves.append((prefix, shape, str(value.dtype)))
    return leaves


def shape_signature(sample: Mapping[str, Any]) -> ShapeSignature:
    """Build a hashable full shape signature for a sample dict."""
    return tuple(collect_shape_leaves(sample))


def pad_bucket_to_batch_size(
    samples: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> Tuple[List[Mapping[str, Any]], int]:
    """Deterministically repeat samples within one bucket to fill ``batch_size``.

    Repeats cycle from the start of the remaining bucket: index ``i % n``.
    Returns ``(padded_batch, num_repeats_added)``.
    """
    if not samples:
        raise ValueError("Cannot pad an empty bucket")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    n = len(samples)
    if n > batch_size:
        raise ValueError(f"Bucket remainder length {n} exceeds batch_size {batch_size}")
    padded: List[Mapping[str, Any]] = list(samples)
    repeats = 0
    while len(padded) < batch_size:
        padded.append(samples[repeats % n])
        repeats += 1
    return padded, repeats


@dataclass
class BucketBatchStats:
    """Per-process / per-worker iteration statistics (not shared across workers)."""

    num_samples: int = 0
    num_batches_yielded: int = 0
    num_repeats: int = 0
    bucket_sample_counts: Dict[ShapeSignature, int] = field(default_factory=dict)

    def record_sample(self, signature: ShapeSignature) -> None:
        self.num_samples += 1
        self.bucket_sample_counts[signature] = self.bucket_sample_counts.get(signature, 0) + 1

    def record_batch(self, repeats: int = 0) -> None:
        self.num_batches_yielded += 1
        self.num_repeats += int(repeats)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_batches_yielded": self.num_batches_yielded,
            "num_repeats": self.num_repeats,
            "bucket_sample_counts": {
                sig: count for sig, count in self.bucket_sample_counts.items()
            },
            "num_buckets": len(self.bucket_sample_counts),
        }


class BucketedBatchIterableDataset(IterableDataset):
    """Pre-batch an ``IterableDataset`` by full stackable signature.

    Args:
        dataset: Source iterable that yields ``dict`` samples. Worker/rank
            sharding must already be handled inside ``dataset`` (this wrapper
            does not call ``get_worker_info`` for re-sharding).
        batch_size: Fixed micro-batch size; must be ``> 0``. Each yielded item
            is a ``list[dict]`` of exactly this length.

    Yields:
        ``list[dict]`` of length ``batch_size``. At stream end, each non-empty
        bucket remainder is padded by deterministic in-bucket repeats so every
        micro-batch has fixed size.

    Stats:
        After ``__iter__`` completes in the *same process*, ``last_stats`` holds
        sample/batch/repeat/bucket counts for that process. With
        ``DataLoader(num_workers>0)`` each worker has its own stats; the main
        process does not aggregate them (by design).
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int,
        *,
        wam_task_ratios: Optional[Mapping[str, float]] = None,
        wam_single_task: Optional[str] = None,
        wam_task_sequence_seed: int = 0x5741_4D00,
        resume_micro_batch_ordinal: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(dataset, IterableDataset):
            raise TypeError(
                f"dataset must be an IterableDataset, got {type(dataset).__name__}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if wam_task_ratios and wam_single_task:
            raise ValueError("configure wam_task_ratios or wam_single_task, not both")
        if wam_task_ratios or wam_single_task:
            raise ValueError(
                "BucketedBatchIterableDataset no longer supports task-aware "
                "selection; use TaskAwareBatchDataset with "
                "TaskAcknowledgingDataLoader"
            )
        self.wam_task_ratios = None
        self.wam_single_task = None
        self.wam_task_sequence_seed = int(wam_task_sequence_seed)
        self.resume_micro_batch_ordinal = int(resume_micro_batch_ordinal)
        self._last_stats: Optional[BucketBatchStats] = None

    @property
    def last_stats(self) -> Optional[BucketBatchStats]:
        """Stats from the most recent ``__iter__`` in this process, if any."""
        return self._last_stats

    def __iter__(self) -> Iterator[List[dict]]:
        stats = BucketBatchStats()
        self._last_stats = stats
        buckets: Dict[ShapeSignature, List[dict]] = defaultdict(list)

        for sample in self.dataset:
            if not isinstance(sample, Mapping):
                raise TypeError(
                    f"Expected dict-like sample from source dataset, got {type(sample).__name__}"
                )
            signature = shape_signature(sample)
            stats.record_sample(signature)
            bucket = buckets[signature]
            bucket.append(sample)
            if len(bucket) >= self.batch_size:
                batch = bucket[: self.batch_size]
                del bucket[: self.batch_size]
                stats.record_batch(repeats=0)
                yield batch

        # Flush remainders: same-bucket deterministic repeat pad only.
        for signature in list(buckets.keys()):
            remainder = buckets[signature]
            if not remainder:
                continue
            batch, repeats = pad_bucket_to_batch_size(remainder, self.batch_size)
            buckets[signature] = []
            stats.record_batch(repeats=repeats)
            yield batch

    def __len__(self) -> int:
        """Estimated micro-batch count: ``ceil(len(source) / batch_size)``.

        Semantics:
            - Uses the wrapped dataset's ``__len__`` as an estimate of how many
              *samples* a full iteration of that object represents.
            - Converts to micro-batches via ceiling division by ``batch_size``.
            - This is a planning/logging estimate, not a hard contract on how
              many batches ``__iter__`` will yield. Actual yield count can be
              higher when many incomplete buckets each pad to one full batch,
              or lower if the source skips samples. If the source length is
              global (not per-rank), this estimate inherits that same scope.
        """
        try:
            num_samples = len(self.dataset)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "Wrapped dataset has no __len__; BucketedBatchIterableDataset "
                "cannot estimate micro-batch count."
            ) from exc
        return int(math.ceil(num_samples / float(self.batch_size)))

    def __getattr__(self, name: str) -> Any:
        # Forward attributes (e.g. reset_seed) to the wrapped dataset.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.dataset, name)
