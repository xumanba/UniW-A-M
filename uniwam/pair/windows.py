#!/usr/bin/env python3
"""Sample four-chunk windows for PAIR and split them by episode.

Each window covers 4 x 24 actions and the 33 frames the interfaces see.  All
windows of one episode share a split, so validation and test episodes are
never seen during router training.  The output is a ``windows.json`` that
``uniwam.inference.fourway`` and ``uniwam.pair.features`` both consume.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from uniwam.inference.orchestration import resolve_episodes

STEPS_PER_WINDOW = 4 * 24


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True, help="Path of the windows.json to write")
    p.add_argument("--episodes", default="all",
                   help="Episode spec accepted by the inference tools (default: all)")
    p.add_argument("--windows-per-episode", type=int, default=0,
                   help="Maximum windows sampled per episode (default 0: every window)")
    p.add_argument("--stride", type=int, default=STEPS_PER_WINDOW,
                   help="Spacing between candidate window starts, in action steps")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def episode_lengths(data_root: str) -> dict[int, int]:
    lengths = {}
    with (Path(data_root) / "meta/episodes.jsonl").open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                lengths[int(row["episode_index"])] = int(row["length"])
    return lengths


def sample_windows(
    data_root: str,
    episodes: list[int],
    windows_per_episode: int,
    stride: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[dict]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions must be non-negative and sum to < 1")
    lengths = episode_lengths(data_root)
    rng = random.Random(seed)
    starts_by_episode = {}
    for ep in episodes:
        length = lengths.get(ep)
        if length is None:
            raise ValueError(f"episode {ep} is missing from meta/episodes.jsonl")
        # The last window needs frame start + 96 and actions up to start + 96.
        starts = list(range(0, length - STEPS_PER_WINDOW, stride))
        if windows_per_episode > 0 and len(starts) > windows_per_episode:
            starts = sorted(rng.sample(starts, windows_per_episode))
        if starts:
            starts_by_episode[ep] = starts
    if not starts_by_episode:
        raise ValueError("no episode contains a complete four-chunk window")

    order = sorted(starts_by_episode)
    rng.shuffle(order)
    n_test = round(len(order) * test_fraction)
    n_val = round(len(order) * val_fraction)
    split_of = {}
    for i, ep in enumerate(order):
        split_of[ep] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    windows = [
        {"episode": ep, "start": start, "split": split_of[ep]}
        for ep in sorted(starts_by_episode)
        for start in starts_by_episode[ep]
    ]
    for i, window in enumerate(windows):
        window["window_id"] = i
    return windows


def main(argv=None) -> None:
    args = parse_args(argv)
    episodes = resolve_episodes(args.episodes, args.data_root)
    windows = sample_windows(
        args.data_root, episodes, args.windows_per_episode, args.stride,
        args.val_fraction, args.test_fraction, args.seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": args.seed,
        "num_windows": len(windows),
        "windows": windows,
    }, indent=2) + "\n")
    counts = {name: sum(w["split"] == name for w in windows) for name in ("train", "val", "test")}
    print(json.dumps({"num_windows": len(windows), "split_windows": counts, "output": str(out)}))


if __name__ == "__main__":
    main()
