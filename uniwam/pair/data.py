"""PAIR training data: chunk features joined with per-interface chunk MAE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from groot.vla.model.dreamzero.modules.pair_router import (
    PAIR_CHUNKS_PER_WINDOW,
    PAIR_FEATURE_KEYS,
    PAIR_TASKS,
)

FAST = PAIR_TASKS.index("action_only_fast_wam")
GIGA = PAIR_TASKS.index("action_only_gigaworld_policy")
JOINT = PAIR_TASKS.index("joint_dreamzero")
SHORT_NAMES = ("Joint", "IDM", "ActionCentric", "ActionOnly")


def load_chunk_set(features: str | Path, labels: str | Path, metric: str = "mae") -> dict:
    """Join ``chunk_features.npz`` with fourway ``predictions.json`` rows.

    Returns router inputs (features + chunk-position one-hot), the per-chunk
    error matrix ``[N, 4]`` in ``PAIR_TASKS`` order, and the split names.
    """
    with np.load(features) as data:
        window_id = data["window_id"].astype(np.int64)
        chunk_index = data["chunk_index"].astype(np.int64)
        split = data["split"].astype(str)
        inputs = [data[key].astype(np.float32) for key in PAIR_FEATURE_KEYS]
    rows = json.loads(Path(labels).read_text())
    cell: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        cell.setdefault((int(row["window_id"]), int(row["chunk_index"])), {})[row["task"]] = float(row[metric])
    missing = [
        (int(w), int(c)) for w, c in zip(window_id, chunk_index)
        if set(cell.get((int(w), int(c)), {})) < set(PAIR_TASKS)
    ]
    if missing:
        raise ValueError(f"{len(missing)} chunks lack labels for all four interfaces, e.g. {missing[:3]}")
    error = np.array([[cell[(int(w), int(c))][t] for t in PAIR_TASKS] for w, c in zip(window_id, chunk_index)])
    inputs.append(np.eye(PAIR_CHUNKS_PER_WINDOW, dtype=np.float32)[chunk_index])
    return {"inputs": inputs, "error": error, "split": split,
            "window_id": window_id, "chunk_index": chunk_index}


def subset(data: dict, mask: np.ndarray) -> dict:
    return {
        "inputs": [x[mask] for x in data["inputs"]],
        "error": data["error"][mask],
        "split": data["split"][mask],
        "window_id": data["window_id"][mask],
        "chunk_index": data["chunk_index"][mask],
    }


def soft_target(error: np.ndarray, tau: float) -> np.ndarray:
    """Softmax over normalized regret: q proportional to exp(-regret / tau)."""
    span = np.clip(error.max(axis=1, keepdims=True) - error.min(axis=1, keepdims=True), 1e-12, None)
    regret = (error - error.min(axis=1, keepdims=True)) / span
    exp = np.exp(-regret / tau)
    return exp / exp.sum(axis=1, keepdims=True)


def near_tie(error: np.ndarray, eps: float) -> np.ndarray:
    """Chunks whose best and second-best interfaces differ by <= eps (relative)."""
    srt = np.sort(error, axis=1)
    return (srt[:, 1] - srt[:, 0]) / np.clip(srt[:, 0], 1e-12, None) <= eps


def tie_aware_target(error: np.ndarray, tau: float, eps: float) -> np.ndarray:
    """Soft regret target; near-tie chunks are split 50/50 between the two
    single-pass action interfaces, the cheapest choice when quality is tied."""
    target = soft_target(error, tau)
    near = near_tie(error, eps)
    target[near] = 0.0
    target[near, FAST] = 0.5
    target[near, GIGA] = 0.5
    return target


def routing_report(picks: np.ndarray, error: np.ndarray, eps: float) -> dict:
    """Realized error of the picks against fixed-interface and oracle baselines."""
    rows = np.arange(len(picks))
    winner = error.argmin(axis=1)
    near = near_tie(error, eps)
    report = {
        "chunks": int(len(picks)),
        "routed": float(error[rows, picks].mean()),
        "oracle": float(error.min(axis=1).mean()),
        "winner_accuracy": float((picks == winner).mean()),
        "decisive_accuracy": float((picks == winner)[~near].mean()) if (~near).any() else None,
        "near_tie_share": float(near.mean()),
        "fixed": {t: float(error[:, i].mean()) for i, t in enumerate(PAIR_TASKS)},
        "pick_share": {t: float((picks == i).mean()) for i, t in enumerate(PAIR_TASKS)},
        "winner_share": {t: float((winner == i).mean()) for i, t in enumerate(PAIR_TASKS)},
    }
    return report


def format_report(report: dict, metric: str = "mae") -> str:
    best_fixed = min(report["fixed"], key=report["fixed"].get)
    lines = [
        f"chunks: {report['chunks']}",
        f"  routed {metric:<4}      : {report['routed']:.6f}",
        f"  best fixed ({best_fixed}): {report['fixed'][best_fixed]:.6f}",
        f"  oracle {metric:<4}      : {report['oracle']:.6f}",
        f"  winner accuracy  : {report['winner_accuracy'] * 100:.1f}%",
    ]
    for task, short in zip(PAIR_TASKS, SHORT_NAMES):
        lines.append(
            f"  {short:<13} picked {report['pick_share'][task] * 100:5.1f}%"
            f"  wins {report['winner_share'][task] * 100:5.1f}%"
            f"  fixed {metric} {report['fixed'][task]:.6f}"
        )
    return "\n".join(lines)
