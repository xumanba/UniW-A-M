#!/usr/bin/env python3
"""Compare the four action-producing WAM inference paradigms on shared windows.

This is an experiment harness, intentionally separate from the repository's
episode-level evaluator.  It samples complete 4-chunk windows once, then runs
Joint, IDM, FastWAM, and GigaWorld on exactly the same (episode, start, chunk)
scenarios.  Every history image/action supplied after a seed comes from the
dataset; generated video is never fed back.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import open_dict

from groot.vla.experiment.checkpoint_staging import (
    unified_inference_expected_config,
    validate_effective_config,
)
from groot.vla.model.dreamzero.checkpoint_loading import (
    build_from_complete_checkpoint,
)
from groot.vla.model.dreamzero.wam_episode import (
    WAMEpisodeRunner,
)
from groot.vla.utils.action_args_override_utils import apply_action_overrides
from uniwam.inference.episode import (
    BatchBuilder,
    EpisodeReader,
    EEF_DIM,
    STEPS_PER_CHUNK,
    WINDOW_FRAME_OFFSETS,
    action_metrics,
    to_cuda,
)
from uniwam.inference.orchestration import configure_eval_task


TASKS = (
    "joint_dreamzero",
    "idm_lingbot_va",
    "action_only_fast_wam",
    "action_only_gigaworld_policy",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-windows", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--windows-json", help="Reuse an existing windows.json exactly")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--joint-cfg", type=float, default=5.0)
    p.add_argument("--idm-cfg", type=float, default=5.0)
    p.add_argument("--backend", choices=("cache", "dense"), default="cache")
    p.add_argument("--dry", action="store_true")
    return p.parse_args()


def init_dist() -> tuple[int, int, int]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29597")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def select_windows(data_root: str, n: int, seed: int) -> list[dict]:
    candidates: dict[int, list[int]] = {}
    with (Path(data_root) / "meta/episodes.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            ep = int(row["episode_index"])
            length = int(row["length"])
            starts = list(range(0, length - 4 * STEPS_PER_CHUNK, 4 * STEPS_PER_CHUNK))
            if starts:
                candidates[ep] = starts
    if n <= 0 or n > len(candidates):
        raise ValueError(f"requested {n} windows, but only {len(candidates)} episodes contain a complete window")
    rng = random.Random(seed)
    chosen = [
        {"episode": ep, "start": rng.choice(candidates[ep])}
        for ep in rng.sample(sorted(candidates), n)
    ]
    chosen.sort(key=lambda x: (x["episode"], x["start"]))
    for i, row in enumerate(chosen):
        row["window_id"] = i
    return chosen


def hydra_config(data_root: str, checkpoint: str):
    from uniwam.cli import compose_config
    cfg = compose_config([
        f'dataset_root="{data_root}"',
        f'pretrained_model_path="{checkpoint}"',
        '++action_head_cfg.config.skip_component_loading=true',
        'wam_task_ratios=null',
    ])
    return apply_action_overrides(cfg)


def build_model(cfg, checkpoint: str, rank: int):
    with open_dict(cfg):
        cfg.action_head_cfg.config.skip_component_loading = True
    validate_effective_config(
        checkpoint,
        unified_inference_expected_config(unified=True),
        allow_legacy=True,
    )
    model = build_from_complete_checkpoint(
        lambda: instantiate(cfg.model),
        checkpoint,
        log=lambda msg: print(f"[rank {rank}] {msg}", flush=True),
    )
    model.action_head.model = model.action_head.model.to(dtype=torch.bfloat16)
    return model.to("cuda").eval()


def make_chunk(
    reader: EpisodeReader,
    builder: BatchBuilder,
    full_canvases: np.ndarray,
    window_start: int,
    chunk_index: int,
    *,
    action_history: bool = False,
) -> tuple[dict, np.ndarray]:
    a0 = window_start + chunk_index * STEPS_PER_CHUNK
    if chunk_index == 0:
        # The first request is a single current GT seed frame.
        images = full_canvases[:1]
        action_start = a0
    else:
        # 9 frames at +0,+3,...,+24 from the preceding GT action chunk.
        images = full_canvases[(chunk_index - 1) * 8 : (chunk_index - 1) * 8 + 9]
        action_start = a0 - STEPS_PER_CHUNK if action_history else a0
    actions = reader.actions[action_start : action_start + STEPS_PER_CHUNK]
    states = reader.states[[a0]]
    return to_cuda(builder.build(images, states, actions)), reader.actions[a0 : a0 + STEPS_PER_CHUNK]


def additional_action_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    if pred.shape != gt.shape or pred.shape != (STEPS_PER_CHUNK, EEF_DIM):
        raise ValueError(f"unexpected action shapes: {pred.shape}, {gt.shape}")
    error = np.abs(pred.astype(np.float64) - gt.astype(np.float64))
    if not np.isfinite(error).all():
        raise ValueError("non-finite action error")
    return {
        "mae": float(error.mean()),
        "smooth_l1": float(np.where(error < 1.0, 0.5 * error**2, error - 0.5).mean()),
        "pred_action": pred.tolist(),
        "gt_action": gt.tolist(),
    }


@torch.no_grad()
def run_one_task(
    model,
    task: str,
    reader: EpisodeReader,
    builder: BatchBuilder,
    full_canvases: np.ndarray,
    window_start: int,
    steps: int,
    joint_cfg: float,
    idm_cfg: float,
    backend: str,
) -> list[dict]:
    head = model.action_head
    configure_eval_task(head, task, steps, idm_cfg, joint_cfg)
    os.environ["WAM_UNIFIED_INFERENCE_BACKEND"] = backend
    os.environ["WAM_FIVE_MODE_INFERENCE_BACKEND"] = backend
    head._wam_inference_trace_callback = None

    def predict(batch):
        if task == "joint_dreamzero":
            if backend == "dense":
                return head.wam_joint_dense_episode_step(
                    model.prepare_input(batch)[1],
                    num_inference_steps=steps, guidance_scale=joint_cfg,
                )
            return model.lazy_joint_video_action_causal(batch)
        if task in ("action_only_fast_wam", "action_only_gigaworld_policy"):
            return model.wam_masked_get_action(
                batch, task=task, num_inference_steps=steps, guidance_scale=1.0,
                generate_video=False, train_experts=True,
            )
        if task == "idm_lingbot_va":
            return model.wam_idm_episode_step(
                batch, teacher_force_history=True, train_experts=True,
                use_input_action_history=True, num_inference_steps=steps,
                guidance_scale=idm_cfg, window_chunks=4,
            )
        raise ValueError(task)

    runner = WAMEpisodeRunner(
        task, reset_fn=lambda: model.wam_episode_reset(task), predict_fn=predict,
    )
    rows = []
    runner.reset_episode()
    try:
        for c in range(4):
            batch, gt = make_chunk(
                reader, builder, full_canvases, window_start, c,
                action_history=task == "idm_lingbot_va",
            )
            out = runner.predict(batch, frame_count=batch["images"].shape[1])
            key = "action_pred" if task == "joint_dreamzero" else "action"
            pred = out[key][0, :, :EEF_DIM].float().cpu().numpy()
            rows.append((pred, gt))
    finally:
        runner.reset_episode()

    result = []
    for c, (pred, gt) in enumerate(rows):
        m = action_metrics(pred, gt)
        result.append({
            "task": task,
            "chunk_index": c,
            "mse": m["mse"],
            **additional_action_metrics(pred, gt),
            "mse_pos": m["mse_pos"],
            "mse_rot6d": m["mse_rot6d"],
            "mse_gripper": m["mse_gripper"],
            "pos_err_l2_mean_m": m["pos_err_l2_mean_m"],
            "history_source": "dataset_gt",
            "generated_video_history": False,
            "backend": backend,
        })
    return result


def aggregate(rows: list[dict], num_windows: int) -> dict:
    grouped: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["window_id"], row["chunk_index"]), {})[row["task"]] = row
    expected = num_windows * 4
    if len(grouped) != expected:
        raise RuntimeError(f"expected {expected} scenarios, got {len(grouped)}")
    for key, values in grouped.items():
        if set(values) != set(TASKS):
            raise RuntimeError(f"scenario {key} has tasks {sorted(values)}")
    wins = {t: 0 for t in TASKS}
    ties = 0
    scenario_rows = []
    for (window_id, chunk_index), values in sorted(grouped.items()):
        mses = {t: float(values[t]["mse"]) for t in TASKS}
        best = min(mses.values())
        winners = [t for t in TASKS if mses[t] == best]
        if len(winners) == 1:
            wins[winners[0]] += 1
            winner = winners[0]
        else:
            ties += 1
            winner = "tie"
        scenario_rows.append({
            "window_id": window_id,
            "chunk_index": chunk_index,
            **{f"mse_{t}": mses[t] for t in TASKS},
            "winner": winner,
        })
    means = {
        t: float(np.mean([r["mse"] for r in rows if r["task"] == t]))
        for t in TASKS
    }
    return {
        "num_windows": num_windows,
        "num_scenarios": len(grouped),
        "num_predictions": len(rows),
        "unique_winner_counts": wins,
        "unique_winner_share": {t: wins[t] / len(grouped) for t in TASKS},
        "tie_scenarios": ties,
        "mean_action_mse": means,
        "scenarios": scenario_rows,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WAM_UNIFIED_INFERENCE_CONSUMPTION", "1")
    os.environ.setdefault("WAM_UNIFIED_JOINT_CONSUMPTION", "1")
    os.environ.setdefault("WAM_UNIFIED_CHUNKWISE_ATTENTION", "0")
    os.environ.setdefault("WAM_FORCE_TOKEN_GROUP_EXPERIMENT", "1")
    os.environ.setdefault("WAM_MOE_GRANULARITY", "token_group_level")
    os.environ.setdefault("WAM_MOE_ROUTING_MODE", "global_task")
    os.environ.setdefault("WAM_MOE_PHASE", "infer")
    os.environ.setdefault("WAM_MOE_PASS", "default")
    os.environ["WAM_UNIFIED_INFERENCE_BACKEND"] = args.backend
    os.environ["WAM_FIVE_MODE_INFERENCE_BACKEND"] = args.backend
    os.environ.setdefault("WAM_TEXT_EMBED_CACHE_DIR", "./.cache/text_embeddings")

    windows = select_windows(args.data_root, args.num_windows, args.seed)
    if args.windows_json:
        windows = json.loads(Path(args.windows_json).read_text())["windows"]
        if len(windows) != args.num_windows:
            raise ValueError("windows-json count does not match num-windows")
    out_dir = Path(args.out_dir)
    rank, world, _ = init_dist()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "windows.json").write_text(json.dumps({
            "seed": args.seed,
            "num_windows": args.num_windows,
            "windows": windows,
        }, indent=2) + "\n")
        (out_dir / "manifest.json").write_text(json.dumps({
            "checkpoint": args.checkpoint,
            "data_root": args.data_root,
            "tasks": list(TASKS),
            "num_windows": args.num_windows,
            "num_scenarios": args.num_windows * 4,
            "num_predictions": args.num_windows * 4 * 4,
            "seed": args.seed,
            "steps": args.steps,
            "smooth_l1_beta": 1.0,
            "raw_actions_saved": True,
            "windows_json": args.windows_json,
            "backend": args.backend,
            "history_source": "dataset_gt",
            "generated_video_history": False,
            "action_mse": "mean squared error over 24 steps x 20 EEF dims",
            "tie_policy": "exact equal minima reported as tie and excluded from unique shares",
        }, indent=2) + "\n")
    dist.barrier()
    if args.dry:
        if rank == 0:
            print(json.dumps({"windows": len(windows), "first": windows[:3], "last": windows[-3:]}, indent=2))
        dist.destroy_process_group()
        return

    cfg = hydra_config(args.data_root, args.checkpoint)
    model = build_model(cfg, args.checkpoint, rank)
    local_rows = []
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
        window_start = window["start"]
        offsets = [window_start + o for o in WINDOW_FRAME_OFFSETS]
        full_canvases = reader.canvases(offsets)
        for task in TASKS:
            metrics = run_one_task(
                model, task, reader, builder, full_canvases, window["start"],
                args.steps, args.joint_cfg, args.idm_cfg, args.backend,
            )
            for m in metrics:
                local_rows.append({
                    "window_id": window["window_id"],
                    "episode": window["episode"],
                    "window_start": window["start"],
                    **m,
                })
        if rank == 0:
            print(f"[progress] {index + 1}/{len(my_windows)} windows; elapsed={time.time() - start_time:.1f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"predictions.rank{rank}.json").write_text(json.dumps(local_rows) + "\n")
    dist.barrier()
    if rank == 0:
        rows = []
        for r in range(world):
            rows.extend(json.loads((out_dir / f"predictions.rank{r}.json").read_text()))
        rows.sort(key=lambda x: (x["window_id"], x["chunk_index"], x["task"]))
        if len(rows) != args.num_windows * 4 * 4:
            raise RuntimeError(f"expected {args.num_windows * 16} predictions, got {len(rows)}")
        for row in rows:
            if row["history_source"] != "dataset_gt" or row["generated_video_history"]:
                raise RuntimeError("non-GT history marker found")
        summary = aggregate(rows, args.num_windows)
        summary["metrics"] = {"mse": aggregate(rows, args.num_windows)}
        for metric in ("mae", "smooth_l1"):
            metric_rows = [{**row, "mse": row[metric]} for row in rows]
            metric_summary = aggregate(metric_rows, args.num_windows)
            metric_summary["mean_action_" + metric] = metric_summary.pop("mean_action_mse")
            for scenario in metric_summary["scenarios"]:
                for task in TASKS:
                    scenario[metric + "_" + task] = scenario.pop("mse_" + task)
            summary["metrics"][metric] = metric_summary
        summary["smooth_l1_beta"] = 1.0
        summary["num_windows"] = args.num_windows
        (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in (
            "num_windows", "num_scenarios", "num_predictions",
            "unique_winner_counts", "unique_winner_share", "tie_scenarios",
            "mean_action_mse",
        )}, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
