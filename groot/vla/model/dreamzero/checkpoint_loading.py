"""Memory-efficient loading for complete DreamZero/WAM safetensors checkpoints.

The normal PyTorch path creates a real FP32 model and then copies BF16 checkpoint
weights into it.  For the 16B WAM model that creates a large, avoidable FP32
allocation and an additional dtype conversion.  This module constructs parameters
on the meta device and assigns one CPU safetensors shard at a time instead.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import time
from pathlib import Path
from typing import Callable, TypeVar

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file


ModelT = TypeVar("ModelT", bound=torch.nn.Module)
LOAD_MODE_ENV = "DREAMZERO_FULL_CHECKPOINT_LOAD_MODE"


def fast_full_checkpoint_loading_enabled() -> bool:
    """Return False only for the explicit compatibility/rollback mode."""
    mode = os.environ.get(LOAD_MODE_ENV, "meta_assign").strip().lower()
    if mode not in {"meta_assign", "legacy"}:
        raise ValueError(
            f"Unsupported {LOAD_MODE_ENV}={mode!r}; expected 'meta_assign' or 'legacy'"
        )
    return mode == "meta_assign"


def _normalize_key(key: str) -> str:
    # Older PEFT-produced full checkpoints may retain this wrapper component.
    return key.replace(".base_layer.", ".")


def checkpoint_shards(checkpoint: str | os.PathLike[str]) -> tuple[list[Path], set[str] | None]:
    ckpt = Path(checkpoint)
    index_path = ckpt / "model.safetensors.index.json"
    single_path = ckpt / "model.safetensors"

    if index_path.exists():
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        declared = {_normalize_key(key) for key in weight_map}
        shards = [ckpt / name for name in sorted(set(weight_map.values()))]
    elif single_path.exists():
        declared = None
        shards = [single_path]
    else:
        declared = None
        shards = [Path(path) for path in sorted(glob.glob(str(ckpt / "*.safetensors")))]

    if not shards:
        raise FileNotFoundError(
            f"No safetensors weights found in {checkpoint!s}; expected model.safetensors "
            "or model.safetensors.index.json"
        )
    missing_files = [str(path) for path in shards if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Checkpoint index references missing shards: {missing_files}")
    return shards, declared


def load_complete_checkpoint_assign(
    model: ModelT,
    checkpoint: str | os.PathLike[str],
    *,
    log: Callable[[str], None] = print,
) -> ModelT:
    """Stream a complete checkpoint into a meta model using ``assign=True``.

    Completeness, unexpected keys, tensor shapes, and residual meta parameters are
    all checked before returning, so a partial warm-start cannot silently use this
    path.
    """
    started = time.monotonic()
    state_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    expected_state_keys = set(state_shapes)
    required_parameter_keys = {name for name, _ in model.named_parameters()}
    shards, declared_keys = checkpoint_shards(checkpoint)

    if declared_keys is not None:
        missing = required_parameter_keys - declared_keys
        unexpected = declared_keys - expected_state_keys
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint is not a complete architecture match: "
                f"missing_parameters={sorted(missing)[:20]} "
                f"unexpected_keys={sorted(unexpected)[:20]}"
            )

    loaded_keys: set[str] = set()
    for shard_number, shard_path in enumerate(shards, start=1):
        log(f"[checkpoint] assigning shard {shard_number}/{len(shards)}: {shard_path}")
        raw_state = load_file(str(shard_path), device="cpu")
        state = {_normalize_key(key): value for key, value in raw_state.items()}

        duplicates = loaded_keys.intersection(state)
        if duplicates:
            raise RuntimeError(f"Duplicate checkpoint tensor keys: {sorted(duplicates)[:20]}")
        unexpected = set(state) - expected_state_keys
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint tensor keys: {sorted(unexpected)[:20]}")
        bad_shapes = {
            key: (tuple(value.shape), state_shapes[key])
            for key, value in state.items()
            if tuple(value.shape) != state_shapes[key]
        }
        if bad_shapes:
            raise RuntimeError(f"Checkpoint tensor shape mismatch: {list(bad_shapes.items())[:20]}")

        model.load_state_dict(state, strict=False, assign=True)
        loaded_keys.update(state)
        del state, raw_state
        gc.collect()

    missing = required_parameter_keys - loaded_keys
    unexpected = loaded_keys - expected_state_keys
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint load was incomplete or architecture-mismatched: "
            f"missing_parameters={sorted(missing)[:20]} "
            f"unexpected_keys={sorted(unexpected)[:20]}"
        )

    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
    if meta_parameters:
        raise RuntimeError(f"Parameters remained on meta after checkpoint load: {meta_parameters[:20]}")

    log(
        f"[checkpoint] assigned complete checkpoint {checkpoint!s}: "
        f"{len(loaded_keys)} tensors in {time.monotonic() - started:.2f}s"
    )
    return model


def build_from_complete_checkpoint(
    factory: Callable[[], ModelT],
    checkpoint: str | os.PathLike[str],
    *,
    log: Callable[[str], None] = print,
) -> ModelT:
    """Construct ``factory`` without storage, then assign a complete checkpoint."""
    started = time.monotonic()
    with init_empty_weights(include_buffers=False):
        model = factory()
    log(f"[checkpoint] meta model constructed in {time.monotonic() - started:.2f}s")
    return load_complete_checkpoint_assign(model, checkpoint, log=log)
