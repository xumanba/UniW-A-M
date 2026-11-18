"""Launch-local data/task schedule seeds for WAM continuation runs.

Fresh dense-DiT training intentionally keeps the historical deterministic
schedule. Every weights-only or full resume gets a new data schedule, while
the task sequence seed is retained when a prior continuation state exists so
the resume ordinal continues the exact same deterministic task stream.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Any


FRESH_DATA_SEED = 42
FRESH_TASK_SEED = 0x5741_4D00
_MAX_SEED = (1 << 63) - 1


def derive_seed(root_seed: int, domain: str) -> int:
    """Derive a stable positive 63-bit seed from ``root_seed`` and ``domain``."""
    digest = hashlib.blake2b(
        f"wam-continuation-v1:{root_seed}:{domain}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & _MAX_SEED


def make_continuation_state(
    resume_mode: str,
    previous_state: dict[str, Any] | None = None,
    *,
    task_contract: dict[str, Any] | None = None,
    task_micro_batch_ordinal: int | None = None,
    root_seed_override: int | None = None,
) -> dict[str, Any]:
    mode = str(resume_mode)
    if root_seed_override is not None and (
        isinstance(root_seed_override, bool)
        or not isinstance(root_seed_override, int)
        or root_seed_override < 0
        or root_seed_override > _MAX_SEED
    ):
        raise ValueError("root_seed_override must be a non-negative 63-bit int")
    if mode == "fresh_dit" and root_seed_override is not None:
        raise ValueError("fresh_dit uses fixed historical seeds and rejects root_seed_override")
    if task_micro_batch_ordinal is not None and (
        isinstance(task_micro_batch_ordinal, bool)
        or not isinstance(task_micro_batch_ordinal, int)
        or task_micro_batch_ordinal < 0
    ):
        raise ValueError("task_micro_batch_ordinal must be a non-negative int")
    previous_ordinal = None
    if previous_state is not None and "task_micro_batch_ordinal" in previous_state:
        previous_ordinal = previous_state["task_micro_batch_ordinal"]
        if (
            isinstance(previous_ordinal, bool)
            or not isinstance(previous_ordinal, int)
            or previous_ordinal < 0
        ):
            raise ValueError(
                "previous continuation state has invalid "
                "task_micro_batch_ordinal"
            )
    ordinal = (
        previous_ordinal
        if previous_ordinal is not None
        else int(task_micro_batch_ordinal or 0)
    )
    if previous_state is not None and previous_state.get("task_contract") is not None:
        if previous_state["task_contract"] != task_contract:
            raise ValueError(
                "task ratios/order changed across continuation: "
                f"{previous_state['task_contract']!r} != {task_contract!r}"
            )
    if mode == "fresh_dit":
        return {
            "version": 2,
            "resume_mode": mode,
            "root_seed": None,
            "data_schedule_seed": FRESH_DATA_SEED,
            "task_sequence_seed": FRESH_TASK_SEED,
            "task_micro_batch_ordinal": ordinal,
            "task_contract": task_contract,
            "task_sequence_reused": False,
            "randomized": False,
        }

    root_seed = (
        secrets.randbits(63)
        if root_seed_override is None
        else root_seed_override
    )
    task_sequence_seed = derive_seed(root_seed, "task-sequence")
    task_sequence_reused = False
    if previous_state is not None and root_seed_override is None:
        previous_task_seed = previous_state.get("task_sequence_seed")
        if isinstance(previous_task_seed, bool) or not isinstance(
            previous_task_seed, int
        ):
            raise ValueError(
                "previous continuation state has invalid task_sequence_seed"
            )
        if previous_task_seed < 0 or previous_task_seed > _MAX_SEED:
            raise ValueError(
                "previous continuation task_sequence_seed is out of range"
            )
        task_sequence_seed = previous_task_seed
        task_sequence_reused = True
    return {
        "version": 2,
        "resume_mode": mode,
        "root_seed": root_seed,
        "data_schedule_seed": derive_seed(root_seed, "data-schedule"),
        "task_sequence_seed": task_sequence_seed,
        "task_micro_batch_ordinal": ordinal,
        "task_contract": task_contract,
        "task_sequence_reused": task_sequence_reused,
        "randomized": root_seed_override is None,
        "root_seed_overridden": root_seed_override is not None,
    }


def atomic_write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
