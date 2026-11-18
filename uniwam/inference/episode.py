"""Episode-level rollout helpers for three-view EEF trajectories.

Tasks (WAM_EVAL_TASKS comma-separated, or legacy WAM_MOE_TASK single task):

  ac_wm          Action-conditioned WM evaluation. The default cache backend
                 denoises one two-latent chunk under 24 clean actions and
                 carries model-generated history across all windows. Set
                 WAM_UNIFIED_INFERENCE_BACKEND=dense to use the training-
                 aligned 33-frame GT-history teacher-forcing path instead.

  joint_dreamzero  Original DreamZero KV-cache chunked inference, open-loop with
                 GT observations as context: chunk c generates 2 latent video
                 frames + 24 actions; before generating chunk c (c>0) the GT
                 video of chunk c-1 (from the dataset, mimicking the real
                 observation after executing) is VAE-encoded and written into
                 the KV cache by `lazy_joint_video_action` itself. The DiT's
                 local attention window covers 4 chunks (local_attn_size=9
                 latent frames); when it is exhausted we restart the window
                 seeded with the CURRENT GT frame (same behavior as the
                 real-robot server). Output: mp4 built ONLY from the model's own
                 generated frames (GT context/re-seed frames excluded, except
                 the very first frame), action masked-MSE vs GT (same masking
                 convention as the training action loss), trajectory plots, npz.

  action_only_fast_wam / action_only_gigaworld_policy
                 Configurable-prefix action-only replan. Each four-chunk window
                 starts from one GT seed; later chunks commit the previous nine
                 GT observations to the typed KV cache. Every DiT step denoises
                 only the current noisy 24-step action target with token-group
                 routing matching training.
                 EXECUTE_ACTION_STEPS selects how many leading actions advance
                 the open-loop GT context before the next 24-step prediction.
                 WAM_ACTION_ONLY_TRAIN_EXPERTS is retained only for old command
                 compatibility; canonical FastWAM/GigaWorld already use the
                 train-aligned expert set.

  idm_lingbot_va Persistent episode-level KV-cache rollout. Every 24-step chunk
                 first generates two video latents, writes them as provisional
                 clean history, then generates 24 actions. Before the next
                 prediction, GT 8-frame observations and GT executed actions
                 replace the complete provisional chunk in-place through the
                 same atomic episode step used by the robot adapter. Token-group
                 routing selects shared+VGM for noisy video, shared+IDM for
                 noisy action, and shared-only for all clean cache writes.
                 Output: mp4 (ep*_idm_cached_rollout.mp4), action masked-MSE vs
                 GT, trajectory plots, npz. `WAM_IDM_WINDOW_BLOCKS` bounds the
                 sliding window (unset = full history).

Parallelism: episode-level data parallel. Launch single-process (one GPU) or via
torchrun --nproc_per_node N (each rank loads a model replica and processes
episodes[rank::world_size]; rank 0 gathers per-episode metrics into summary.json).
EPISODES accepts "all", an explicit comma list, or "random:K" (K distinct
episodes drawn with EPISODE_SEED, identical across ranks and across wm/joint
runs for comparability).

Provides shared adapters for offline evaluation.
"""

import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import open_dict
from tqdm import tqdm

from groot.vla.model.dreamzero.checkpoint_loading import (
    build_from_complete_checkpoint,
    fast_full_checkpoint_loading_enabled,
)
from groot.vla.model.dreamzero.action_head.wam_unified_inference import (
    unified_inference_backend,
)
from groot.vla.model.dreamzero.wam_episode import (
    WAM_JOINT_CHUNKS_PER_WINDOW as JOINT_CHUNKS_PER_WINDOW,
    WAMEpisodeRunner,
    episode_history_indices,
    unified_inference_consumption_enabled,
)
from groot.vla.experiment.checkpoint_staging import (
    unified_inference_expected_config,
    validate_effective_config,
)

# The FlowUniPC scheduler's multistep_uni_p_bh_update is torch.compile'd with
# fullgraph and recompiles across chunk shapes / step indices / orders; the
# default dynamo recompile cap aborts multi-chunk joint rollouts
# (FailOnRecompileLimitHit). Same mitigation as eval_utils/serve_dreamzero_wan22.py.
_dynamo = torch._dynamo.config
for _attr, _val in (
    ("cache_size_limit", 1000),
    ("recompile_limit", 800),
    ("accumulated_cache_size_limit", 1000),
    ("accumulated_recompile_limit", 2000),
):
    if hasattr(_dynamo, _attr):
        setattr(_dynamo, _attr, _val)

from groot.vla.utils.action_args_override_utils import apply_action_overrides
from uniwam.inference.orchestration import (
    center_crop_frame,
    clear_joint_persistent_state,
    collective_check_errors,
    configure_eval_task,
    load_lerobot_task_instructions,
    merge_gathered_metrics,
    parse_eval_tasks,
    propagate_summary_write_error,
    resolve_episodes,
    resolve_eval_out_root,
    write_eval_manifest,
    write_task_summary_metrics,
)

# ---------------------------------------------------------------------------
# Window layout constants (MUST match training: converter offsets + 24-step
# chunks, 8 frames per chunk at stride 3 = 10 Hz video, 30 Hz actions).
# ---------------------------------------------------------------------------
STEPS_PER_CHUNK = 24
FRAME_OFFSETS_PER_CHUNK = [0, 3, 6, 9, 12, 15, 18, 21]
CHUNKS_PER_WINDOW = JOINT_CHUNKS_PER_WINDOW
STEPS_PER_WINDOW = STEPS_PER_CHUNK * CHUNKS_PER_WINDOW  # 96
WINDOW_FRAME_OFFSETS = [
    c * STEPS_PER_CHUNK + o for c in range(CHUNKS_PER_WINDOW) for o in FRAME_OFFSETS_PER_CHUNK
] + [STEPS_PER_WINDOW]  # 33 frames spanning [0, 96]
FASTWAM_STEPS = 24
FASTWAM_FRAME_OFFSETS = list(range(0, FASTWAM_STEPS + 1, 3))  # 9 RGB frames
EEF_DIM = 20
VIDEO_FPS = 10  # 3x temporal subsampling of 30 Hz
_PROFILE_CALL_INDEX = 0


def timed_predict(fn, *args, **kwargs):
    global _PROFILE_CALL_INDEX
    torch.cuda.synchronize()
    start = time.perf_counter()
    profile_dir = os.environ.get("WAM_PROFILE_DIR", "").strip()
    profile_calls = int(os.environ.get("WAM_PROFILE_CALLS", "0"))
    if profile_dir and _PROFILE_CALL_INDEX < profile_calls:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            output = fn(*args, **kwargs)
        profile_root = Path(profile_dir)
        profile_root.mkdir(parents=True, exist_ok=True)
        stem = f"rank{os.environ.get('RANK', '0')}_call{_PROFILE_CALL_INDEX}"
        profiler.export_chrome_trace(str(profile_root / f"{stem}.json"))
        (profile_root / f"{stem}.txt").write_text(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=100
            )
        )
    else:
        output = fn(*args, **kwargs)
    _PROFILE_CALL_INDEX += 1
    torch.cuda.synchronize()
    return output, time.perf_counter() - start


def install_inference_trace(head):
    if os.environ.get("WAM_INFERENCE_TRACE", "0") != "1":
        head._wam_inference_trace_callback = None
        return None
    records = []

    def callback(**record):
        records.append({
            **{key: value for key, value in record.items() if key != "tensors"},
            "tensors": {
                key: value.detach().to("cpu")
                for key, value in record["tensors"].items()
            },
        })

    head._wam_inference_trace_callback = callback
    return records


def fastwam_replan_windows(
    total_valid_steps: int,
    execute_action_steps: int = FASTWAM_STEPS,
) -> list[tuple[int, int]]:
    """Return ``(start, valid)`` windows for action-only prefix replanning."""
    if total_valid_steps < 0:
        raise ValueError("total_valid_steps must be non-negative")
    if not 1 <= execute_action_steps <= FASTWAM_STEPS:
        raise ValueError(
            f"execute_action_steps must be in 1..24, got {execute_action_steps}"
        )
    return [
        (start, min(execute_action_steps, total_valid_steps - start))
        for start in range(0, total_valid_steps, execute_action_steps)
    ]


def world_model_replan_starts(total_valid_steps: int) -> list[int]:
    """Full 24-action WM horizons; incomplete tails have no training analogue."""
    if total_valid_steps < 0:
        raise ValueError("total_valid_steps must be non-negative")
    return list(range(0, total_valid_steps - FASTWAM_STEPS + 1, FASTWAM_STEPS))


def world_model_progress_total(
    data_root: str,
    episodes: list[int],
    max_windows: int | None,
) -> int:
    """Total WM windows for one rank's episode shard, from LeRobot metadata."""
    lengths: dict[int, int] = {}
    with open(Path(data_root) / "meta/episodes.jsonl") as f:
        for line in f:
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])

    missing = [ep for ep in episodes if ep not in lengths]
    if missing:
        raise ValueError(f"missing episode lengths in metadata: {missing[:20]}")

    total = 0
    for ep in episodes:
        n_windows = len(world_model_replan_starts(lengths[ep] - 1))
        total += min(n_windows, max_windows) if max_windows else n_windows
    return total


# Text templates — MUST byte-match the collate() agilex_eef branch and
# DreamTransform's negative prompt.
PROMPT_TEMPLATE = (
    "A multi-view video shows that a robot {instr} The video is split into three views: "
    "The top view shows the camera view from the robot's head, the bottom-left view shows "
    "the camera view from the left wrist, and the bottom-right view shows the camera view "
    "from the right wrist. The robot {instr}"
)
NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, "
    "painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, "
    "mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused "
    "fingers, stagnant image, cluttered background, three legs, many people in the background, "
    "walking backwards."
)


# ---------------------------------------------------------------------------
# Episode reader: parquet + 3 mp4 views, deterministic eval preprocessing
# (per-view 95% center crop then resize to 160x320, matching eval/real robot).
# ---------------------------------------------------------------------------
class EpisodeReader:
    VIEWS = ("head", "left_wrist", "right_wrist")

    def __init__(self, data_root: str, episode_index: int, view_h: int = 160, view_w: int = 320):
        import decord
        import pandas as pd

        self.view_h, self.view_w = view_h, view_w
        root = Path(data_root)
        chunk = episode_index // 1000
        self.df = pd.read_parquet(root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet")
        self.actions = np.stack(self.df["action"].values).astype(np.float32)  # [N, 20]
        self.states = np.stack(self.df["observation.state"].values).astype(np.float32)
        self.length = len(self.df)
        self.readers = {
            v: decord.VideoReader(
                str(root / f"videos/chunk-{chunk:03d}/observation.images.{v}/episode_{episode_index:06d}.mp4")
            )
            for v in self.VIEWS
        }
        task_idx = int(self.df["annotation.task"].iloc[0])
        instructions = load_lerobot_task_instructions(root)
        if task_idx not in instructions:
            raise ValueError(
                f"episode {episode_index} references missing task_index={task_idx}"
            )
        self.instruction = instructions[task_idx]

    def canvases(self, frame_indices: list[int]) -> np.ndarray:
        """DROID-style canvases for the given raw frame indices.

        Returns uint8 [T, 2*view_h, 2*view_w, 3] (RGB): head widened x2 on the
        top row, left_wrist bottom-left, right_wrist bottom-right — same layout
        as DreamTransform._prepare_video's AGILEX_EEF branch.
        """
        import cv2

        idx = [min(max(i, 0), self.length - 1) for i in frame_indices]
        views = {}
        for v in self.VIEWS:
            frames = self.readers[v].get_batch(idx).asnumpy()  # [T, H, W, 3] RGB
            views[v] = np.stack(
                [
                    cv2.resize(
                        center_crop_frame(f),
                        (self.view_w, self.view_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for f in frames
                ]
            )
        T = len(idx)
        canvas = np.zeros((T, 2 * self.view_h, 2 * self.view_w, 3), dtype=np.uint8)
        canvas[:, : self.view_h, :, :] = np.repeat(views["head"], 2, axis=2)
        canvas[:, self.view_h :, : self.view_w, :] = views["left_wrist"]
        canvas[:, self.view_h :, self.view_w :, :] = views["right_wrist"]
        return canvas


# ---------------------------------------------------------------------------
# Batch building
# ---------------------------------------------------------------------------
class BatchBuilder:
    def __init__(self, tokenizer_path: str, instruction: str, action_dim_padded: int, state_dim_padded: int):
        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

        tok = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean="whitespace")
        instr = str(instruction).lower()
        prompt = PROMPT_TEMPLATE.format(instr=instr)
        self.text, self.text_mask = tok(prompt, return_mask=True, add_special_tokens=True)
        self.text_neg, self.text_neg_mask = tok(NEGATIVE_PROMPT, return_mask=True, add_special_tokens=True)
        self.action_dim_padded = action_dim_padded
        self.state_dim_padded = state_dim_padded

    def _pad(self, x: np.ndarray, dim: int) -> torch.Tensor:
        out = np.zeros((1, x.shape[0], dim), dtype=np.float32)
        out[0, :, : x.shape[1]] = x
        return torch.from_numpy(out)

    def build(self, canvases: np.ndarray, states: np.ndarray, actions: np.ndarray | None) -> dict:
        batch = {
            "images": torch.from_numpy(canvases[None]),  # [1, T, H, W, C] uint8
            "text": self.text,
            "text_attention_mask": self.text_mask,
            "text_negative": self.text_neg,
            "text_attention_mask_negative": self.text_neg_mask,
            "state": self._pad(states, self.state_dim_padded),
            "embodiment_id": torch.tensor([34], dtype=torch.int64),  # agilex_eef marker
        }
        if actions is not None:
            batch["action"] = self._pad(actions, self.action_dim_padded)
        return batch


def to_cuda(batch: dict) -> dict:
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# VAE decode + mp4
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_latents_to_frames(head, latents_bfchw: torch.Tensor) -> np.ndarray:
    """[1, F, C, H, W] latents -> uint8 [T_px, H, W, C] (T_px = (F-1)*4 + 1)."""
    lat = latents_bfchw.transpose(1, 2).to(dtype=torch.bfloat16)  # [1, C, F, H, W]
    frames = head.vae.decode(
        lat,
        tiled=head.tiled,
        tile_size=(head.tile_size_height, head.tile_size_width),
        tile_stride=(head.tile_stride_height, head.tile_stride_width),
    )
    frames = rearrange(frames, "b c t h w -> b t h w c")[0]
    return ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)


def decoded_video_to_frames(decoded_bcthw: torch.Tensor) -> np.ndarray:
    """Already-streamed normalized RGB [1,C,T,H,W] -> uint8 [T,H,W,C]."""
    if not torch.isfinite(decoded_bcthw).all():
        raise FloatingPointError("streaming VAE decode produced NaN or Inf")
    frames = rearrange(decoded_bcthw, "b c t h w -> b t h w c")[0]
    return (
        ((frames.float() + 1) * 127.5)
        .clip(0, 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def resize_rgb_video_uint8(
    frames: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    if frames.shape[1:3] == (target_height, target_width):
        return frames
    resized = torch.nn.functional.interpolate(
        torch.from_numpy(frames).permute(0, 3, 1, 2).float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return (
        resized.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    )


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = VIDEO_FPS):
    import imageio
    import shutil
    import subprocess
    import tempfile

    def validate_mp4(candidate: Path) -> None:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt:format=duration",
                "-of",
                "json",
                str(candidate),
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                f"invalid MP4 after encoding/publish: {candidate}: "
                f"{probe.stderr.strip()}"
            )
        metadata = json.loads(probe.stdout)
        streams = metadata.get("streams") or []
        duration = float((metadata.get("format") or {}).get("duration") or 0.0)
        if not streams or streams[0].get("codec_name") != "h264" or duration <= 0:
            raise RuntimeError(f"incomplete MP4 metadata for {candidate}: {metadata}")

    path.parent.mkdir(parents=True, exist_ok=True)
    video_tmp_root = Path(os.environ.get("WAM_VIDEO_TMPDIR", "/tmp"))
    video_tmp_root.mkdir(parents=True, exist_ok=True)
    publish_tmp = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with tempfile.TemporaryDirectory(prefix="wam_mp4_", dir=video_tmp_root) as temp_dir:
            local_path = Path(temp_dir) / path.name
            imageio.mimsave(
                str(local_path),
                list(frames),
                fps=fps,
                codec="libx264",
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            validate_mp4(local_path)
            shutil.copyfile(local_path, publish_tmp)
            validate_mp4(publish_tmp)
            os.replace(publish_tmp, path)
        validate_mp4(path)
    finally:
        publish_tmp.unlink(missing_ok=True)
    print(f"[out] wrote {path} ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Metrics + trajectory visualization
# ---------------------------------------------------------------------------
POS_IDX = [0, 1, 2, 10, 11, 12]
ROT_IDX = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]
GRIP_IDX = [9, 19]
DIM_NAMES = (
    [f"L_pos_{a}" for a in "xyz"] + [f"L_rot6d_{i}" for i in range(6)] + ["L_grip"]
    + [f"R_pos_{a}" for a in "xyz"] + [f"R_rot6d_{i}" for i in range(6)] + ["R_grip"]
)


def action_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Masked MSE, same convention as the training action loss (mask = the 20
    valid dims out of the padded 32; here both arrays are already sliced to 20,
    which is equivalent): elementwise (pred-gt)^2 -> mean over dims -> mean over
    steps. Units are raw physical (m / rot6d / gripper)."""
    err2 = (pred - gt) ** 2
    per_chunk = [
        float(err2[i : i + STEPS_PER_CHUNK].mean()) for i in range(0, len(err2), STEPS_PER_CHUNK)
    ]
    return {
        "mse": float(err2.mean()),
        "mse_pos": float(err2[:, POS_IDX].mean()),
        "mse_rot6d": float(err2[:, ROT_IDX].mean()),
        "mse_gripper": float(err2[:, GRIP_IDX].mean()),
        "pos_err_l2_mean_m": float(np.linalg.norm((pred - gt)[:, POS_IDX].reshape(-1, 2, 3), axis=-1).mean()),
        "per_chunk_mse": per_chunk,
    }


def plot_trajectories(pred: np.ndarray, gt: np.ndarray, out_dir: Path, ep: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) 3D end-effector position paths, one subplot per arm.
    fig = plt.figure(figsize=(14, 6))
    for i, (arm, sl) in enumerate([("left", slice(0, 3)), ("right", slice(10, 13))]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        g, p = gt[:, sl], pred[:, sl]
        ax.plot(g[:, 0], g[:, 1], g[:, 2], "-", color="tab:blue", label="GT", linewidth=1.5)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "--", color="tab:red", label="pred", linewidth=1.5)
        ax.scatter(*g[0], color="green", s=40, label="start")
        ax.scatter(*g[-1], color="black", s=40, label="GT end")
        ax.set_title(f"ep{ep} {arm} EEF position (m)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"ep{ep}_traj3d.png", dpi=120)
    plt.close(fig)

    # (2) Per-dimension time series grid (20 panels).
    fig, axes = plt.subplots(4, 5, figsize=(22, 12), sharex=True)
    t = np.arange(len(gt))
    for d in range(EEF_DIM):
        ax = axes[d // 5, d % 5]
        ax.plot(t, gt[:, d], color="tab:blue", linewidth=1.0, label="GT")
        ax.plot(t, pred[:, d], color="tab:red", linewidth=1.0, linestyle="--", label="pred")
        ax.set_title(DIM_NAMES[d], fontsize=9)
        if d == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"ep{ep}: predicted vs GT actions (raw units, 30 Hz steps)")
    fig.tight_layout()
    fig.savefig(out_dir / f"ep{ep}_dims.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# AC-WM cache rollout: model-generated history across all windows
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_world_model_cache(
    model,
    reader: EpisodeReader,
    builder: BatchBuilder,
    out_dir: Path,
    ep: int,
    steps: int,
    cfg_scale: float,
    max_windows: int | None,
    progress: tqdm | None = None,
    write_video: bool = True,
):
    """Run AC-WM continuously, carrying generated history across windows."""
    window_starts = world_model_replan_starts(reader.length - 1)
    n_windows = len(window_starts)
    if max_windows:
        n_windows = min(n_windows, max_windows)
    if progress is None:
        print(
            f"[wm-cache] ep{ep}: {reader.length} steps -> "
            f"{n_windows} chunks of {FASTWAM_STEPS} steps"
        )

    predict_fn = lambda batch: model.wam_acwm_episode_step(
        batch,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
    )
    runner = WAMEpisodeRunner(
        "ac_wm",
        reset_fn=model.wam_acwm_episode_reset,
        predict_fn=predict_fn,
        continuous_cache=True,
    )
    chunk_times = []
    generated_chunks: list[np.ndarray] = []
    latent_values: list[torch.Tensor] = []
    decoded_values: list[torch.Tensor] = []
    seed_canvas: np.ndarray | None = None
    current_canvas: np.ndarray | None = None
    torch.cuda.reset_peak_memory_stats()
    runner.reset_episode()
    try:
        for w, a0 in enumerate(window_starts[:n_windows]):
            request = runner.next_request()
            if request.kind == "seed":
                canvases = reader.canvases([a0])
                if seed_canvas is None:
                    seed_canvas = canvases[0].copy()
            else:
                if current_canvas is None:
                    raise RuntimeError(
                        "AC-WM cache requested a generated seed before the "
                        "first predicted chunk"
                    )
                canvases = current_canvas[None]
            actions = reader.actions[a0 : a0 + FASTWAM_STEPS]
            states = reader.states[[a0]]
            batch = to_cuda(builder.build(canvases, states, actions))
            out, elapsed = timed_predict(
                runner.predict, batch, frame_count=len(canvases)
            )
            chunk_times.append(elapsed)
            if not torch.isfinite(out["video"]).all():
                raise FloatingPointError(
                    f"AC-WM cache ep{ep} chunk {w} latent contains NaN or Inf"
                )
            if not torch.isfinite(out["decoded_video"]).all():
                raise FloatingPointError(
                    f"AC-WM cache ep{ep} chunk {w} decoded RGB contains NaN or Inf"
                )
            latent_values.append(out["video"].detach().float().cpu())
            decoded_values.append(out["decoded_video"].detach().float().cpu())
            decoded = decoded_video_to_frames(out["decoded_video"])
            expected_decoded_frames = len(FRAME_OFFSETS_PER_CHUNK)
            if decoded.shape[0] != expected_decoded_frames:
                raise RuntimeError(
                    f"AC-WM cache decoded {decoded.shape[0]} frames, expected "
                    f"{expected_decoded_frames}"
                )
            generated_chunks.append(decoded)
            current_canvas = decoded[-1].copy()
            if progress is None:
                print(
                    f"[ac_wm-cache] ep{ep} chunk {w + 1}/{n_windows} done "
                    f"({1 + 8 * (w + 1)} frames)"
                )
            else:
                progress.set_postfix(
                    ep=ep, window=f"{w + 1}/{n_windows}", refresh=False
                )
                progress.update(1)

        if generated_chunks and seed_canvas is not None:
            generated = np.concatenate(generated_chunks, axis=0)
            seed = resize_rgb_video_uint8(
                seed_canvas[None], generated.shape[1], generated.shape[2]
            )
            mp4_frames = np.concatenate([seed, generated], axis=0)
            latent_tensor = torch.cat(latent_values, dim=1)
            decoded_tensor = torch.cat(decoded_values, dim=2)
            gt_indices = [
                start + offset
                for start in window_starts[:n_windows]
                for offset in range(3, FASTWAM_STEPS + 1, 3)
            ]
            gt = resize_rgb_video_uint8(
                reader.canvases(gt_indices), generated.shape[1], generated.shape[2]
            )
            mse = float(
                np.mean(
                    (generated.astype(np.float32) - gt.astype(np.float32)) ** 2
                )
            )
            psnr = (
                float("inf")
                if mse == 0.0
                else float(10.0 * math.log10((255.0**2) / mse))
            )
            numeric = {
                "latent_mean": float(latent_tensor.mean().item()),
                "latent_std": float(latent_tensor.std().item()),
                "latent_abs_max": float(latent_tensor.abs().max().item()),
                "decoded_mean": float(decoded_tensor.mean().item()),
                "decoded_std": float(decoded_tensor.std().item()),
                "decoded_abs_max": float(decoded_tensor.abs().max().item()),
                "gt_mse": mse,
                "gt_psnr": psnr,
                "frames_sha256": hashlib.sha256(generated.tobytes()).hexdigest(),
            }
        else:
            mp4_frames = []
            numeric = {}
    finally:
        runner.reset_episode()

    if write_video:
        write_mp4(
            mp4_frames,
            out_dir / f"ep{ep}_ac_wm_cache_generated_history.mp4",
        )
    return {
        "episode": ep,
        "windows": n_windows,
        "frames": len(mp4_frames),
        "backend": "cache_generated_history",
        "evaluation_protocol": "autoregressive_generated_history",
        "autonomous_rollout": True,
        "history_source": "model_generated",
        "action_condition_source": "dataset_gt",
        "cross_window_generated_history": True,
        "chunk_times_s": chunk_times,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        **numeric,
    }


# ---------------------------------------------------------------------------
# AC-WM teacher-forced dense33 evaluation, two latents per action chunk
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_world_model(model, reader: EpisodeReader, builder: BatchBuilder, out_dir: Path, ep: int,
                        steps: int, cfg_scale: float, max_windows: int | None,
                        progress: tqdm | None = None, write_video: bool = True):
    if unified_inference_backend() == "cache":
        return rollout_world_model_cache(
            model,
            reader,
            builder,
            out_dir,
            ep,
            steps,
            cfg_scale,
            max_windows,
            progress=progress,
            write_video=write_video,
        )
    if not unified_inference_consumption_enabled():
        raise ValueError(
            "AC-WM teacher-forced dense33 evaluation requires "
            "WAM_UNIFIED_INFERENCE_CONSUMPTION=1"
        )
    window_starts = world_model_replan_starts(reader.length - 1)
    n_windows = len(window_starts)
    if max_windows:
        n_windows = min(n_windows, max_windows)
    if progress is None:
        print(f"[wm] ep{ep}: {reader.length} steps -> {n_windows} windows of {FASTWAM_STEPS} steps")

    seed_canvas = reader.canvases([window_starts[0]])[0] if n_windows else None
    decoded_chunks: list[np.ndarray] = []
    latent_values: list[torch.Tensor] = []
    decoded_values: list[torch.Tensor] = []
    backend = "dense33_teacher_forced_gt_history"
    predict_fn = lambda batch: model.wam_acwm_episode_step(
        batch,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
    )
    runner = WAMEpisodeRunner(
        "ac_wm",
        reset_fn=model.wam_acwm_episode_reset,
        predict_fn=predict_fn,
    )
    chunk_times = []
    torch.cuda.reset_peak_memory_stats()
    runner.reset_episode()
    try:
        for w in range(n_windows):
            a0 = window_starts[w]
            request = runner.next_request()
            actions = reader.actions[a0 : a0 + FASTWAM_STEPS]
            states = reader.states[[a0]]
            if request.kind == "seed":
                frame_ids = [a0]
            else:
                frame_ids = episode_history_indices(
                    a0 - FASTWAM_STEPS, reader.length
                )
            canvases = reader.canvases(frame_ids)
            batch = to_cuda(builder.build(canvases, states, actions))
            out, elapsed = timed_predict(
                runner.predict, batch, frame_count=len(frame_ids)
            )
            chunk_times.append(elapsed)
            if not torch.isfinite(out["video"]).all():
                raise FloatingPointError(
                    f"AC-WM ep{ep} chunk {w} latent contains NaN or Inf"
                )
            if not torch.isfinite(out["decoded_video"]).all():
                raise FloatingPointError(
                    f"AC-WM ep{ep} chunk {w} decoded RGB contains NaN or Inf"
                )
            latent_values.append(out["video"].detach().float().cpu())
            decoded_values.append(
                out["decoded_video"].detach().float().cpu()
            )
            decoded_chunks.append(
                decoded_video_to_frames(out["decoded_video"])
            )
            frame_count = 1 + 8 * (w + 1)
            if progress is None:
                print(
                    f"[ac_wm-teacher-forced] ep{ep} window {w + 1}/{n_windows} done "
                    f"({frame_count} frames)"
                )
            else:
                progress.set_postfix(
                    ep=ep, window=f"{w + 1}/{n_windows}", refresh=False
                )
                progress.update(1)
        if decoded_chunks:
            generated = np.concatenate(decoded_chunks, axis=0)
            seed = resize_rgb_video_uint8(
                seed_canvas[None], generated.shape[1], generated.shape[2]
            )
            mp4_frames = np.concatenate([seed, generated], axis=0)
            latent_tensor = torch.cat(latent_values, dim=1)
            decoded_tensor = torch.cat(decoded_values, dim=2)
            gt_indices = [
                start + offset
                for start in window_starts[:n_windows]
                for offset in range(3, FASTWAM_STEPS + 1, 3)
            ]
            gt = reader.canvases(gt_indices)
            gt = resize_rgb_video_uint8(
                gt, generated.shape[1], generated.shape[2]
            )
            mse = float(
                np.mean(
                    (
                        generated.astype(np.float32)
                        - gt.astype(np.float32)
                    )
                    ** 2
                )
            )
            psnr = (
                float("inf")
                if mse == 0.0
                else float(10.0 * math.log10((255.0**2) / mse))
            )
            numeric = {
                "latent_mean": float(latent_tensor.mean().item()),
                "latent_std": float(latent_tensor.std().item()),
                "latent_abs_max": float(latent_tensor.abs().max().item()),
                "decoded_mean": float(decoded_tensor.mean().item()),
                "decoded_std": float(decoded_tensor.std().item()),
                "decoded_abs_max": float(decoded_tensor.abs().max().item()),
                "gt_mse": mse,
                "gt_psnr": psnr,
                "frames_sha256": hashlib.sha256(
                    generated.tobytes()
                ).hexdigest(),
            }
            print(
                f"[ac_wm-numeric] ep{ep} windows={n_windows} "
                f"latent_mean={numeric['latent_mean']:.8f} "
                f"latent_std={numeric['latent_std']:.8f} "
                f"latent_abs_max={numeric['latent_abs_max']:.8f} "
                f"decoded_mean={numeric['decoded_mean']:.8f} "
                f"decoded_std={numeric['decoded_std']:.8f} "
                f"decoded_abs_max={numeric['decoded_abs_max']:.8f} "
                f"gt_mse={numeric['gt_mse']:.8f} "
                f"gt_psnr={numeric['gt_psnr']:.8f} "
                f"frames_sha256={numeric['frames_sha256']}",
                flush=True,
            )
        else:
            mp4_frames = []
            numeric = {}
    finally:
        runner.reset_episode()

    if write_video:
        write_mp4(
            mp4_frames,
            out_dir / f"ep{ep}_ac_wm_teacher_forced_gt_history.mp4",
        )
    return {
        "episode": ep,
        "windows": n_windows,
        "frames": len(mp4_frames),
        "backend": backend,
        "evaluation_protocol": "teacher_forced_gt_history",
        "autonomous_rollout": False,
        "history_source": "dataset_gt",
        "action_condition_source": "dataset_gt",
        "cross_window_generated_history": False,
        "chunk_times_s": chunk_times,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        **numeric,
    }


# ---------------------------------------------------------------------------
# Joint rollout: DreamZero KV-cache chunks with GT video as context
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_joint(model, reader: EpisodeReader, builder: BatchBuilder, out_dir: Path, ep: int,
                  steps: int, cfg_scale: float, max_chunks: int | None):
    head = model.action_head
    n_chunks = (reader.length - 1) // STEPS_PER_CHUNK
    if max_chunks:
        n_chunks = min(n_chunks, max_chunks)
    print(f"[joint] ep{ep}: {reader.length} steps -> {n_chunks} chunks of {STEPS_PER_CHUNK} steps")

    pred_actions: list[np.ndarray] = []
    windows_latents: list[list[torch.Tensor]] = []  # per window: list of [1,F,C,H,W]
    backend = unified_inference_backend()
    if backend == "dense":
        predict_fn = lambda batch: head.wam_joint_dense_episode_step(
            model.prepare_input(batch)[1],
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        )
    else:
        predict_fn = model.lazy_joint_video_action_causal
    runner = WAMEpisodeRunner(
        "joint_dreamzero",
        reset_fn=lambda: model.wam_episode_reset("joint_dreamzero"),
        predict_fn=predict_fn,
        chunks_per_window=CHUNKS_PER_WINDOW,
    )
    chunk_times = []
    torch.cuda.reset_peak_memory_stats()
    runner.reset_episode()

    for c in range(n_chunks):
        a0 = c * STEPS_PER_CHUNK
        state = reader.states[[a0]]  # [1, 20] current observation state
        request = runner.next_request()
        if request.kind == "seed":
            # (Re)seed a window with the CURRENT GT frame (closed-loop restart
            # semantics). The public contract is exactly one seed frame.
            frame_ids = [min(a0, reader.length - 1)]
            canvases = reader.canvases(frame_ids)
            windows_latents.append([])
        else:
            # GT video of the PREVIOUS chunk = the observation after "executing"
            # it; lazy_joint_video_action VAE-encodes it and writes it into the
            # KV cache before denoising this chunk. Include the t=24 endpoint so
            # the newest frame and current state are both at time a0.
            prev0 = (c - 1) * STEPS_PER_CHUNK
            frame_ids = episode_history_indices(prev0, reader.length)
            canvases = reader.canvases(frame_ids)
        batch = to_cuda(builder.build(canvases, state, None))
        out, elapsed = timed_predict(
            runner.predict, batch, frame_count=len(frame_ids)
        )
        chunk_times.append(elapsed)
        pred_actions.append(out["action_pred"][0, :, :EEF_DIM].float().cpu().numpy())
        # video_pred: [1, C, T, H, W]; T=3 on window start (1 seed + 2 gen), else 2.
        lat = out["video_pred"].transpose(1, 2).float().cpu()  # [1, T, C, H, W]
        windows_latents[-1].append(lat)

        print(f"[joint] ep{ep} chunk {c + 1}/{n_chunks} done")

    # mp4: decode per window; keep GT frame 0 only for the very first window.
    mp4_frames: list[np.ndarray] = []
    for w, lat_list in enumerate(windows_latents):
        lat = torch.cat(lat_list, dim=1).to("cuda")  # [1, 1+2k, C, H, W]
        px = decode_latents_to_frames(head, lat)
        mp4_frames.extend(px if w == 0 else px[1:])
    write_mp4(mp4_frames, out_dir / f"ep{ep}_joint_rollout.mp4")

    # Action error (training-convention masked MSE) + plots + dump.
    pred = np.concatenate(pred_actions, axis=0)  # [n_chunks*24, 20]
    gt = reader.actions[: pred.shape[0]]
    metrics = action_metrics(pred, gt)
    metrics.update({
        "episode": ep,
        "chunks": n_chunks,
        "frames": len(mp4_frames),
        "backend": backend,
        "chunk_times_s": chunk_times,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    })
    plot_trajectories(pred, gt, out_dir, ep)
    np.savez(out_dir / f"ep{ep}_actions.npz", pred=pred, gt=gt)
    with open(out_dir / f"ep{ep}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[joint] ep{ep} action masked-MSE={metrics['mse']:.6f} "
          f"(pos {metrics['mse_pos']:.6f} | rot6d {metrics['mse_rot6d']:.6f} | "
          f"grip {metrics['mse_gripper']:.6f} | mean L2 pos err {metrics['pos_err_l2_mean_m'] * 100:.2f} cm)")
    return metrics


def _write_action_episode_outputs(pred: np.ndarray, gt: np.ndarray, out_dir: Path, ep: int,
                                   tag: str, extra: dict) -> dict:
    """Shared action-metric + trajectory dump for the action-producing tasks
    (same masked-MSE convention and plots as the joint path)."""
    metrics = action_metrics(pred, gt)
    metrics.update({"episode": ep, **extra})
    plot_trajectories(pred, gt, out_dir, ep)
    np.savez(out_dir / f"ep{ep}_actions.npz", pred=pred, gt=gt)
    with open(out_dir / f"ep{ep}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{tag}] ep{ep} action masked-MSE={metrics['mse']:.6f} "
          f"(pos {metrics['mse_pos']:.6f} | rot6d {metrics['mse_rot6d']:.6f} | "
          f"grip {metrics['mse_gripper']:.6f} | mean L2 pos err {metrics['pos_err_l2_mean_m'] * 100:.2f} cm)")
    return metrics


# ---------------------------------------------------------------------------
# Action-only rollout (fastwam / gigaworld): GT first frame as context.
# FastWAM and GigaWorld predict an independent 24-step horizon, execute only
# its configured prefix, then replan from the corresponding next GT frame.
# Canonical FastWAM/GigaWorld denoise action only from one clean RGB seed.
#   WAM_ACTION_ONLY_VIDEO=1        rejected; future-video inference was removed.
#   WAM_ACTION_ONLY_TRAIN_EXPERTS  legacy command compatibility only; it does
#                                  not change the canonical task path.
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_action_only(model, reader: EpisodeReader, builder: BatchBuilder, out_dir: Path, ep: int,
                        task: str, steps: int, max_windows: int | None,
                        execute_action_steps: int = FASTWAM_STEPS):
    if os.environ.get("WAM_ACTION_ONLY_VIDEO", "0") == "1":
        raise ValueError(
            "WAM_ACTION_ONLY_VIDEO=1 is unsupported: canonical FastWAM/GigaWorld "
            "inference no longer constructs future-video targets"
        )
    if (
        unified_inference_consumption_enabled()
        and execute_action_steps != FASTWAM_STEPS
    ):
        raise ValueError(
            "unified FastWAM/GigaWorld inference requires executing all "
            "24 actions before committing the next nine-frame history"
        )
    train_experts = os.environ.get("WAM_ACTION_ONLY_TRAIN_EXPERTS", "1") == "1"
    window_steps = execute_action_steps
    total_valid_steps = reader.length - 1
    replan_windows = fastwam_replan_windows(
        total_valid_steps, execute_action_steps
    )
    n_windows = len(replan_windows)
    if max_windows:
        n_windows = min(n_windows, max_windows)
    tag = "act-te" if train_experts else "act"
    print(f"[{tag}] ep{ep} ({task}): {reader.length} steps -> {n_windows} windows of {window_steps} steps")

    def predict_one(batch):
        return model.wam_masked_get_action(
            batch,
            task=task,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generate_video=False,
            train_experts=train_experts,
        )

    runner = WAMEpisodeRunner(
        task,
        reset_fn=lambda: model.wam_episode_reset(task),
        predict_fn=predict_one,
    )
    pred_actions: list[np.ndarray] = []
    chunk_times = []
    torch.cuda.reset_peak_memory_stats()
    runner.reset_episode()
    try:
        for w in range(n_windows):
            a0, valid_steps = replan_windows[w]
            request = runner.next_request()
            actions = reader.actions[a0 : a0 + valid_steps]  # values unused; sets schema/horizon
            if valid_steps < FASTWAM_STEPS:
                pad = np.zeros((FASTWAM_STEPS - valid_steps, actions.shape[1]), dtype=actions.dtype)
                actions = np.concatenate([actions, pad], axis=0)
            states = reader.states[[a0]]
            if request.kind == "seed":
                frame_ids = [a0]
            else:
                frame_ids = episode_history_indices(
                    a0 - FASTWAM_STEPS, reader.length
                )
            canvases = reader.canvases(frame_ids)
            batch = to_cuda(builder.build(canvases, states, actions))
            out, elapsed = timed_predict(
                runner.predict, batch, frame_count=len(frame_ids)
            )
            chunk_times.append(elapsed)
            if "video" in out:
                raise RuntimeError(f"{task} unexpectedly returned video")
            pred_actions.append(out["action"][0, :valid_steps, :EEF_DIM].float().cpu().numpy())
            print(f"[{tag}] ep{ep} window {w + 1}/{n_windows} done")
    finally:
        runner.reset_episode()

    pred = np.concatenate(pred_actions, axis=0)
    gt = reader.actions[: pred.shape[0]]
    extra = {
        "task": task,
        "windows": n_windows,
        "window_steps": window_steps,
        "video": False,
        "train_experts": train_experts,
        "backend": unified_inference_backend(),
        "chunk_times_s": chunk_times,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return _write_action_episode_outputs(pred, gt, out_dir, ep, tag, extra)


# ---------------------------------------------------------------------------
# IDM rollout: one persistent episode, 24-step teacher-forced commits.
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_idm(model, reader: EpisodeReader, builder: BatchBuilder, out_dir: Path, ep: int,
                steps: int, cfg_scale: float, max_windows: int | None):
    head = model.action_head
    wb_env = os.environ.get("WAM_IDM_WINDOW_BLOCKS", "").strip()
    window_blocks = int(wb_env) if wb_env else CHUNKS_PER_WINDOW
    replan_windows = fastwam_replan_windows(reader.length - 1)
    n_windows = len(replan_windows)
    if max_windows:
        n_windows = min(n_windows, max_windows)
    print(
        f"[idm-episode] ep{ep}: {reader.length} steps -> {n_windows} chunks of "
        f"{STEPS_PER_CHUNK} steps (persistent window_chunks={window_blocks})"
    )

    pred_actions: list[np.ndarray] = []
    mp4_frames: list[np.ndarray] = []
    backend = unified_inference_backend()
    if backend == "dense":
        predict_fn = lambda batch: head.wam_idm_dense_episode_step(
            model.prepare_input(batch)[1],
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        )
    else:
        predict_fn = lambda batch: model.wam_idm_episode_step(
            batch,
            teacher_force_history=True,
            train_experts=True,
            use_input_action_history=True,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            window_chunks=window_blocks,
        )
    runner = WAMEpisodeRunner(
        "idm_lingbot_va",
        reset_fn=lambda: model.wam_episode_reset("idm_lingbot_va"),
        predict_fn=predict_fn,
    )
    chunk_times = []
    torch.cuda.reset_peak_memory_stats()
    runner.reset_episode()
    try:
        for w, (a0, valid_steps) in enumerate(replan_windows[:n_windows]):
            request = runner.next_request()
            if request.kind == "seed":
                frame_ids = [a0]
                action_start = a0
                action_valid = valid_steps
            else:
                # Commit the observation and GT action that were actually used
                # as open-loop teacher-forced history for the previous horizon.
                previous_start = a0 - STEPS_PER_CHUNK
                frame_ids = episode_history_indices(previous_start, reader.length)
                action_start = previous_start
                action_valid = min(STEPS_PER_CHUNK, reader.length - 1 - action_start)
            actions = reader.actions[action_start : action_start + action_valid]
            if action_valid < STEPS_PER_CHUNK:
                pad = np.zeros(
                    (STEPS_PER_CHUNK - action_valid, actions.shape[1]),
                    dtype=actions.dtype,
                )
                actions = np.concatenate([actions, pad], axis=0)
            states = reader.states[[a0]]
            batch = to_cuda(builder.build(reader.canvases(frame_ids), states, actions))
            out, elapsed = timed_predict(
                runner.predict, batch, frame_count=len(frame_ids)
            )
            chunk_times.append(elapsed)
            if backend == "dense":
                condition = out["condition_video"]
            else:
                condition = head._wam_idm_episode["latest_condition_latent"]
            pred_actions.append(
                out["action"][0, :valid_steps, :EEF_DIM].float().cpu().numpy()
            )
            px = decode_latents_to_frames(
                head, torch.cat([condition, out["video"]], dim=1)
            )
            valid_video = min(8, (valid_steps + 2) // 3)
            mp4_frames.extend(
                px[: valid_video + 1] if w == 0 else px[1 : valid_video + 1]
            )
            print(
                f"[idm-episode] ep{ep} chunk {w + 1}/{n_windows} done "
                f"({len(mp4_frames)} frames)"
            )
    finally:
        runner.reset_episode()

    write_mp4(mp4_frames, out_dir / f"ep{ep}_idm_cached_rollout.mp4")
    pred = np.concatenate(pred_actions, axis=0)
    gt = reader.actions[: pred.shape[0]]
    return _write_action_episode_outputs(
        pred, gt, out_dir, ep, "idm-episode",
        {
            "chunks": n_windows,
            "frames": len(mp4_frames),
            "cached": backend == "cache",
            "backend": backend,
            "chunk_times_s": chunk_times,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_complete_eval_checkpoint(model, checkpoint: str, rank: int) -> None:
    """Stream a full-model checkpoint and reject partial/architecture-mismatched weights."""
    import glob
    from safetensors.torch import load_file

    ckpt = Path(checkpoint)
    index_path = ckpt / "model.safetensors.index.json"
    expected_state_keys = set(model.state_dict())
    required_parameter_keys = {name for name, _ in model.named_parameters()}

    declared_keys: set[str] | None = None
    if index_path.exists():
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        declared_keys = set(weight_map)
        shard_paths = [ckpt / name for name in sorted(set(weight_map.values()))]
    else:
        shard_paths = [Path(path) for path in sorted(glob.glob(str(ckpt / "*.safetensors")))]

    if not shard_paths:
        raise FileNotFoundError(f"No safetensors weights found in {checkpoint}")

    if declared_keys is not None:
        missing = required_parameter_keys - declared_keys
        unexpected = declared_keys - expected_state_keys
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint is not a complete match for the evaluation model: "
                f"missing_parameters={sorted(missing)[:20]} "
                f"unexpected_keys={sorted(unexpected)[:20]}"
            )

    loaded_keys: set[str] = set()
    for shard_path in shard_paths:
        state_dict = load_file(str(shard_path))
        loaded_keys.update(state_dict)
        model.load_state_dict(state_dict, strict=False)
        del state_dict

    missing = required_parameter_keys - loaded_keys
    unexpected = loaded_keys - expected_state_keys
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint load was incomplete or architecture-mismatched: "
            f"missing_parameters={sorted(missing)[:20]} "
            f"unexpected_keys={sorted(unexpected)[:20]}"
        )
    print(
        f"[rollout][rank {rank}] loaded complete checkpoint {checkpoint} "
        f"({len(loaded_keys)} tensors)"
    )


def main(cfg):
    if unified_inference_consumption_enabled():
        os.environ.setdefault("WAM_MOE_GRANULARITY", "token_group_level")
        os.environ.setdefault("WAM_UNIFIED_JOINT_CONSUMPTION", "1")
        if os.environ["WAM_MOE_GRANULARITY"] != "token_group_level":
            raise ValueError(
                "unified inference requires WAM_MOE_GRANULARITY=token_group_level"
            )
    cfg = apply_action_overrides(cfg)

    tasks = parse_eval_tasks()
    steps = int(os.environ.get("WAM_EVAL_STEPS", "16"))
    execute_action_steps = int(os.environ.get("EXECUTE_ACTION_STEPS", "24"))
    if not 1 <= execute_action_steps <= FASTWAM_STEPS:
        raise ValueError(
            f"EXECUTE_ACTION_STEPS must be in 1..24, got {execute_action_steps}"
        )
    wm_cfg_scale = float(
        os.environ.get(
            "ACWM_CFG_SCALE", os.environ.get("WAM_CFG_SCALE", "1.0")
        )
    )
    idm_cfg_scale = float(
        os.environ.get(
            "IDM_CFG_SCALE", os.environ.get("WAM_CFG_SCALE", "5.0")
        )
    )
    joint_cfg_scale = float(os.environ.get("JOINT_CFG_SCALE", "5.0"))
    max_chunks = int(os.environ.get("MAX_CHUNKS", "0")) or None
    data_root = os.environ.get(
        "AGILEX_EEF_DATA_ROOT", "./data/trajectories"
    )
    episodes_spec = os.environ.get("EPISODES", "random:8")
    episode_seed = int(os.environ.get("EPISODE_SEED", "0"))
    episodes = resolve_episodes(episodes_spec, data_root)
    repeats = int(os.environ.get("WAM_EVAL_REPEATS", "1"))
    if repeats <= 0:
        raise ValueError("WAM_EVAL_REPEATS must be positive")
    if not episodes:
        raise ValueError(
            f"EPISODES spec {episodes_spec!r} resolved to empty episode list"
        )
    checkpoint = cfg.pretrained_model_path or ""
    out_root = resolve_eval_out_root(os.environ.get("WAM_EVAL_OUT_DIR"), checkpoint or None)

    if os.environ.get("WAM_EVAL_DRY", "0") == "1":
        print(
            f"[dry] tasks={tasks} out_root={out_root} episodes={episodes} "
            f"steps={steps} action_only_execute_steps={execute_action_steps} "
            f"max_chunks={max_chunks} checkpoint={checkpoint!r}"
        )
        for task in tasks:
            print(f"[dry] task={task} task_dir={out_root / task}")
        reader = EpisodeReader(data_root, episodes[0])
        canv = reader.canvases(WINDOW_FRAME_OFFSETS)
        print(f"[dry] ep{episodes[0]}: len={reader.length} actions={reader.actions.shape} "
              f"canvases={canv.shape} instruction={reader.instruction!r}")
        print("[dry] OK; skipping model build.")
        return

    if checkpoint:
        effective = validate_effective_config(
            checkpoint,
            unified_inference_expected_config(
                unified=unified_inference_consumption_enabled()
            ),
            allow_legacy=os.environ.get(
                "ALLOW_LEGACY_EFFECTIVE_CONFIG", "0"
            )
            == "1",
        )
        if effective is None:
            print(
                "WARNING: legacy checkpoint has no effective_config.json; "
                "runtime routing remains explicitly forced by the launcher"
            )
        else:
            print("[effective-config] episode rollout contract validated")

    # Distributed episode-parallel setup: honor torchrun env; fall back to a
    # single-process group so the code path is identical either way.
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29593")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        # Episode runtimes are heavily imbalanced (episode length x dense33
        # VAE cost), so the end-of-run error/metric collectives can wait far
        # longer than the 10-minute c10d default while straggler ranks finish
        # their last episodes.  Allow up to 6 hours before timing out.
        from datetime import timedelta

        dist.init_process_group(
            backend="nccl", timeout=timedelta(seconds=int(
                os.environ.get("WAM_DIST_TIMEOUT_SECONDS", "21600")
            ))
        )
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    my_episodes = episodes[rank::world]
    if rank == 0:
        print(
            f"[rollout] tasks={tasks} data={data_root}\n[rollout] out={out_root}\n"
            f"[rollout] episodes(all)={episodes} world={world}"
        )
        write_eval_manifest(
            out_root / "manifest.json",
            checkpoint=checkpoint,
            tasks=tasks,
            episodes_spec=episodes_spec,
            episode_seed=episode_seed,
            episodes=episodes,
            steps=steps,
            wm_cfg_scale=wm_cfg_scale,
            idm_cfg_scale=idm_cfg_scale,
            joint_cfg_scale=joint_cfg_scale,
            max_chunks=max_chunks,
            world_size=world,
        )

    fast_full_load = bool(checkpoint) and fast_full_checkpoint_loading_enabled()
    if fast_full_load:
        with open_dict(cfg):
            cfg.action_head_cfg.config.skip_component_loading = True
        print(f"[rollout][rank {rank}] meta+assign loading {checkpoint}")
        model = build_from_complete_checkpoint(
            lambda: instantiate(cfg.model),
            checkpoint,
            log=lambda message: print(f"[rollout][rank {rank}] {message}"),
        )
    else:
        model = instantiate(cfg.model)
        if checkpoint:
            # Load on CPU before the single GPU transfer even in rollback mode.
            load_complete_eval_checkpoint(model, checkpoint, rank)
    if os.environ.get("WAM_EVAL_BF16", "1") == "1":
        model.action_head.model = model.action_head.model.to(dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    head = model.action_head

    for task in tasks:
        clear_joint_persistent_state(head)
        task_cfg_scale = (
            idm_cfg_scale if task == "idm_lingbot_va" else wm_cfg_scale
        )
        configure_eval_task(
            head, task, steps, task_cfg_scale, joint_cfg_scale
        )
        task_dir = out_root / task
        print(
            f"[rollout][rank {rank}/{world}] task={task} steps={steps} max_chunks={max_chunks} "
            f"action_only_execute_steps={execute_action_steps} -> mine={my_episodes}"
        )
        if rank == 0:
            print(f"[rollout] === task {task} -> {task_dir} ===")
            if task == "joint_dreamzero" and joint_cfg_scale == 1.0:
                print("[rollout] joint CFG disabled (cfg_scale=1.0)")

        progress = None
        if task == "ac_wm" and sys.stderr.isatty():
            progress = tqdm(
                total=world_model_progress_total(data_root, my_episodes, max_chunks),
                desc=f"WM rank {rank}/{world}",
                position=rank,
                leave=True,
                unit="window",
                dynamic_ncols=True,
                mininterval=0.5,
            )

        my_metrics: list[dict] = []
        local_error: str | None = None
        try:
            for repeat in range(repeats):
                run_task_dir = (
                    task_dir / f"repeat-{repeat}"
                    if repeats > 1
                    else task_dir
                )
                for ep in my_episodes:
                    reader = EpisodeReader(data_root, ep)
                    builder = BatchBuilder(
                        tokenizer_path=cfg.tokenizer_path,
                        instruction=reader.instruction,
                        action_dim_padded=int(cfg.max_action_dim),
                        state_dim_padded=int(cfg.max_state_dim),
                    )
                    trace_records = (
                        install_inference_trace(head)
                        if repeat == min(1, repeats - 1)
                        else None
                    )
                    if trace_records is None:
                        head._wam_inference_trace_callback = None
                    try:
                        with torch.inference_mode():
                            if task == "ac_wm":
                                m = rollout_world_model(
                                    model, reader, builder, run_task_dir, ep, steps, wm_cfg_scale,
                                    max_chunks, progress
                                )
                            elif task == "joint_dreamzero":
                                m = rollout_joint(
                                    model,
                                    reader,
                                    builder,
                                    run_task_dir,
                                    ep,
                                    steps,
                                    joint_cfg_scale,
                                    max_chunks,
                                )
                            elif task in ("action_only_fast_wam", "action_only_gigaworld_policy"):
                                m = rollout_action_only(
                                    model, reader, builder, run_task_dir, ep, task, steps,
                                    max_chunks, execute_action_steps
                                )
                            elif task == "idm_lingbot_va":
                                m = rollout_idm(
                                    model,
                                    reader,
                                    builder,
                                    run_task_dir,
                                    ep,
                                    steps,
                                    idm_cfg_scale,
                                    max_chunks,
                                )
                            else:
                                raise ValueError(f"unsupported task for episode rollout: {task}")
                    finally:
                        head._wam_inference_trace_callback = None
                    if trace_records is not None:
                        run_task_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            trace_records,
                            run_task_dir / f"ep{ep}_trace.pt",
                        )
                    m["repeat"] = repeat
                    m["warmup"] = repeats > 1 and repeat == 0
                    my_metrics.append(m)
        except Exception as exc:
            local_error = f"rank {rank} task={task}: {exc}\n{traceback.format_exc()}"
        finally:
            if progress is not None:
                progress.close()

        collective_check_errors(dist, local_error)

        gathered = [None] * world
        dist.all_gather_object(gathered, my_metrics)
        summary_error: str | None = None
        if rank == 0:
            all_metrics = merge_gathered_metrics(gathered)
            summary_path = task_dir / "summary.json"
            summary_error = write_task_summary_metrics(summary_path, all_metrics)
            if summary_error is None:
                print(
                    f"[rollout] task={task} all {len(all_metrics)} episodes done "
                    f"-> {summary_path}"
                )

        propagate_summary_write_error(dist, rank, summary_error)

        clear_joint_persistent_state(head)
        dist.barrier()


if __name__ == "__main__":
    main()
