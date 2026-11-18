#!/usr/bin/env python3
"""Offline routing quality of a PAIR router or seed ensemble.

Routes every chunk in a feature set and scores the picks against the
interfaces' recorded chunk errors (open loop: every interface saw the same
recorded history).  No UniW(A)M forward pass is needed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from groot.vla.model.dreamzero.modules.pair_router import (
    PAIR_TASKS,
    load_pair_router_ensemble,
    pick_interfaces,
)
from uniwam.pair.data import format_report, load_chunk_set, routing_report, subset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--routers", required=True,
                   help="Router checkpoint, comma-separated list, or directory of pair_router_seed*.pt")
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--split", default="test", help="Split to score, or 'all'")
    p.add_argument("--metric", choices=("mae", "mse"), default="mae")
    p.add_argument("--exclude", default="", help="Comma-separated interfaces never selected")
    p.add_argument("--chunk0-allowed", default="",
                   help="Excluded interfaces that stay eligible on chunk 0")
    p.add_argument("--joint-threshold", type=float, default=0.0)
    p.add_argument("--tie-eps", type=float, default=0.01)
    p.add_argument("--output", help="Optional JSON report path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = lambda text: tuple(x.strip() for x in text.split(",") if x.strip())  # noqa: E731
    data = load_chunk_set(args.features, args.labels, args.metric)
    if args.split != "all":
        data = subset(data, data["split"] == args.split)
    if not len(data["error"]):
        raise ValueError(f"no chunks in split {args.split!r}")
    router = load_pair_router_ensemble(args.routers, device="cpu")
    with torch.no_grad():
        probs = torch.softmax(router([torch.from_numpy(x) for x in data["inputs"]]), dim=-1)
    picks = np.empty(len(probs), dtype=np.int64)
    for chunk in np.unique(data["chunk_index"]):
        rows = np.flatnonzero(data["chunk_index"] == chunk)
        picks[rows] = pick_interfaces(
            probs[rows], int(chunk), exclude=names(args.exclude),
            chunk0_allowed=names(args.chunk0_allowed),
            joint_threshold=args.joint_threshold,
        ).numpy()
    report = routing_report(picks, data["error"], args.tie_eps)
    report["members"] = len(getattr(router, "routers", [router]))
    report["tasks"] = list(PAIR_TASKS)
    print(format_report(report, args.metric))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
