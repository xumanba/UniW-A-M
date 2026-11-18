#!/usr/bin/env python3
"""Extract PAIR router features for every (window, chunk) in a windows.json.

For chunk ``c`` the frozen backbone sees exactly the request the interface
receives at that chunk (``uniwam.inference.fourway.make_chunk``), and the
probe keeps the first ``1 + 2c`` latent frames.  No interface is run here;
labels come from ``uniwam.inference.fourway`` on the same windows.json.

Launch with ``torchrun``; windows are sharded across ranks and merged into
``<out-dir>/chunk_features.npz`` on rank 0.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from groot.vla.model.dreamzero.modules.pair_router import (
    PAIR_CHUNKS_PER_WINDOW,
    PAIR_FEATURE_KEYS,
    PAIRFeatureEncoder,
)
from uniwam.inference.episode import BatchBuilder, EpisodeReader, WINDOW_FRAME_OFFSETS
from uniwam.inference.fourway import build_model, hydra_config, make_chunk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True, help="Complete UniW(A)M checkpoint")
    p.add_argument("--windows-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--probe-experts", default="shared",
                   help="Comma-separated MoE experts active in the video probe")
    return p.parse_args()


def init_dist() -> tuple[int, int]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        # gloo keeps the long feature pass free of NCCL watchdog timeouts.
        dist.init_process_group(backend="gloo", timeout=timedelta(hours=6))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return dist.get_rank(), dist.get_world_size()


def configure_runtime() -> None:
    for key, value in {
        "WAM_UNIFIED_INFERENCE_CONSUMPTION": "1",
        "WAM_UNIFIED_JOINT_CONSUMPTION": "1",
        "WAM_UNIFIED_CHUNKWISE_ATTENTION": "0",
        "WAM_FORCE_TOKEN_GROUP_EXPERIMENT": "1",
        "WAM_MOE_GRANULARITY": "token_group_level",
        "WAM_MOE_ROUTING_MODE": "global_task",
        "WAM_MOE_PHASE": "infer",
        "WAM_MOE_PASS": "default",
        "WAM_TEXT_EMBED_CACHE_DIR": "./.cache/text_embeddings",
    }.items():
        os.environ.setdefault(key, value)


def main() -> None:
    args = parse_args()
    configure_runtime()
    windows = json.loads(Path(args.windows_json).read_text())["windows"]
    probe_experts = tuple(name.strip() for name in args.probe_experts.split(",") if name.strip())
    out_dir = Path(args.out_dir)
    rank, world = init_dist()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps({
            "checkpoint": args.checkpoint,
            "windows_json": args.windows_json,
            "num_windows": len(windows),
            "feature_keys": list(PAIR_FEATURE_KEYS),
            "visible_latents_per_chunk": [1 + 2 * c for c in range(PAIR_CHUNKS_PER_WINDOW)],
            "probe_experts": list(probe_experts),
        }, indent=2) + "\n")
    dist.barrier()

    cfg = hydra_config(args.data_root, args.checkpoint)
    model = build_model(cfg, args.checkpoint, rank)
    encoder = PAIRFeatureEncoder(model, probe_experts=probe_experts)

    feats: dict[str, list[np.ndarray]] = {key: [] for key in PAIR_FEATURE_KEYS}
    ids: list[tuple[int, int]] = []
    my_windows = windows[rank::world]
    start_time = time.time()
    for index, window in enumerate(my_windows):
        reader = EpisodeReader(args.data_root, window["episode"])
        builder = BatchBuilder(
            tokenizer_path=cfg.tokenizer_path,
            instruction=reader.instruction,
            action_dim_padded=int(cfg.max_action_dim),
            state_dim_padded=int(cfg.max_state_dim),
        )
        start = int(window["start"])
        canvases = reader.canvases([start + o for o in WINDOW_FRAME_OFFSETS])
        for chunk_index in range(PAIR_CHUNKS_PER_WINDOW):
            batch, _gt = make_chunk(reader, builder, canvases, start, chunk_index)
            vectors = encoder(batch, chunk_index)
            for key in PAIR_FEATURE_KEYS:
                feats[key].append(vectors[key][0].cpu().numpy())
            ids.append((int(window["window_id"]), chunk_index))
        if rank == 0 and (index + 1) % 25 == 0:
            print(f"[progress] {index + 1}/{len(my_windows)} windows; "
                  f"elapsed={time.time() - start_time:.1f}s", flush=True)

    np.savez_compressed(
        out_dir / f"chunk_features.rank{rank}.npz",
        window_id=np.asarray([i[0] for i in ids], dtype=np.int64),
        chunk_index=np.asarray([i[1] for i in ids], dtype=np.int64),
        **{key: (np.stack(v) if v else np.zeros((0, 1), dtype=np.float16)) for key, v in feats.items()},
    )
    dist.barrier()
    if rank == 0:
        parts = [np.load(out_dir / f"chunk_features.rank{r}.npz") for r in range(world)]
        wid = np.concatenate([p["window_id"] for p in parts])
        cid = np.concatenate([p["chunk_index"] for p in parts])
        order = np.lexsort((cid, wid))
        splits = {int(w["window_id"]): w.get("split", "") for w in windows}
        merged = {
            "window_id": wid[order],
            "chunk_index": cid[order],
            "split": np.asarray([splits[int(w)] for w in wid[order]], dtype="<U8"),
        }
        for key in PAIR_FEATURE_KEYS:
            values = np.concatenate([p[key] for p in parts if p[key].shape[0]], axis=0)[order]
            if not np.isfinite(values.astype(np.float32)).all():
                raise RuntimeError(f"non-finite PAIR feature {key}")
            merged[key] = values
        for p in parts:
            p.close()
        expected = len(windows) * PAIR_CHUNKS_PER_WINDOW
        if len(merged["window_id"]) != expected:
            raise RuntimeError(f"expected {expected} chunk features, got {len(merged['window_id'])}")
        np.savez_compressed(out_dir / "chunk_features.npz", **merged)
        for r in range(world):
            (out_dir / f"chunk_features.rank{r}.npz").unlink()
        print(json.dumps({"done": True, "chunks": expected,
                          "output": str(out_dir / "chunk_features.npz")}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
