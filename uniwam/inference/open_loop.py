"""Shared offline inference for the five UniW(A)M interfaces.

Open-loop = sample observations from an offline dataset and run inference, without
executing actions in a real environment (same notion as giga-world-policy's
open-loop client). For action tasks it reports action MSE vs the dataset's
ground-truth actions; for world_model it reports the generated video shape.

Run through the repository-root infer.py entry point and shared configuration.

Set WAM_EVAL_DRY=1 to only compose+print the config (no heavy model build) — a
cheap check that config wiring is correct.
"""

import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from groot.vla.utils.action_args_override_utils import apply_action_overrides
from groot.vla.model.dreamzero.checkpoint_loading import (
    build_from_complete_checkpoint,
    fast_full_checkpoint_loading_enabled,
)
from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    ACWM_CHUNKS_PER_CLIP,
    ACWM_WINDOW_FIRST_RGB_INDEX,
    FASTWAM_ACTIONS_PER_CHUNK,
    FULL_WINDOW_RGB_FRAMES,
)
from groot.vla.model.dreamzero.wam_episode import (
    WAMEpisodeRunner,
    unified_inference_consumption_enabled,
)
from groot.vla.model.dreamzero.action_head.wam_unified_inference import (
    unified_inference_backend,
)
from groot.vla.experiment.checkpoint_staging import (
    unified_inference_expected_config,
    validate_effective_config,
)
from uniwam.inference.orchestration import load_lerobot_task_instructions


def _first_batch(dataset, collator, batch_size):
    it = iter(dataset)
    feats = []
    for _ in range(batch_size):
        feats.append(next(it))
    return collator(feats)


def _to_cuda(batch):
    import tree

    def move(x):
        if torch.is_tensor(x):
            return x.to("cuda")
        return x

    return tree.map_structure(move, batch)


def masked_action_mse(pred, target, action_mask):
    """Compare only valid action channels, excluding model padding."""
    target = target.to(device=pred.device, dtype=pred.dtype)
    valid = action_mask.to(device=pred.device, dtype=torch.bool)
    if not valid.any():
        raise ValueError("No valid action elements available for evaluation")
    return (pred[valid] - target[valid]).square().mean()


@torch.no_grad()
def _run_acwm_open_loop_window(model, batch, steps: int, cfg_scale: float):
    expected_actions = ACWM_CHUNKS_PER_CLIP * FASTWAM_ACTIONS_PER_CHUNK
    if batch["action"].shape[1] != expected_actions:
        raise ValueError(
            f"AC-WM open-loop requires {expected_actions} actions, got "
            f"{batch['action'].shape[1]}"
        )
    unified = unified_inference_consumption_enabled()
    backend = unified_inference_backend() if unified else "legacy"
    expected_images = FULL_WINDOW_RGB_FRAMES if unified else 49
    if batch["images"].shape[1] != expected_images:
        raise ValueError(
            f"AC-WM open-loop requires exactly {expected_images} RGB frames, got "
            f"{batch['images'].shape[1]}"
        )

    latent_chunks = []
    decoded_chunks = []
    model.wam_acwm_episode_reset()
    try:
        for chunk_index in range(ACWM_CHUNKS_PER_CLIP):
            chunk_batch = dict(batch)
            if unified:
                if chunk_index == 0:
                    chunk_batch["images"] = batch["images"][:, :1]
                elif backend == "cache":
                    previous = decoded_chunks[-1][:, :, -1:]
                    chunk_batch["images"] = (
                        (previous.float().clamp(-1, 1) + 1.0) * 127.5
                    ).round().clamp(0, 255).to(torch.uint8).permute(
                        0, 2, 3, 4, 1
                    )
                else:
                    history_start = (chunk_index - 1) * 8
                    chunk_batch["images"] = batch["images"][
                        :, history_start : history_start + 9
                    ]
            else:
                seed_index = ACWM_WINDOW_FIRST_RGB_INDEX
                chunk_batch["images"] = batch["images"][
                    :, seed_index : seed_index + 1
                ]
            action_start = chunk_index * FASTWAM_ACTIONS_PER_CHUNK
            chunk_batch["action"] = batch["action"][
                :,
                action_start : action_start + FASTWAM_ACTIONS_PER_CHUNK,
            ]
            chunk = model.wam_acwm_episode_step(
                chunk_batch,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
            )
            latent_chunks.append(chunk["video"])
            decoded_chunks.append(chunk["decoded_video"])
    finally:
        model.wam_acwm_episode_reset()

    return {
        "video": torch.cat(latent_chunks, dim=1),
        "decoded_video": torch.cat(decoded_chunks, dim=2),
    }


@torch.no_grad()
def _run_masked_open_loop_window(model, batch, task: str, steps: int):
    if batch["images"].shape[1] != FULL_WINDOW_RGB_FRAMES:
        raise ValueError(
            f"{task} unified open-loop requires 33 RGB frames, got "
            f"{batch['images'].shape[1]}"
        )
    if batch["action"].shape[1] != 4 * FASTWAM_ACTIONS_PER_CHUNK:
        raise ValueError(f"{task} unified open-loop requires 96 actions")
    if batch["state"].shape[1] != 4:
        raise ValueError(f"{task} unified open-loop requires 4 states")

    runner = WAMEpisodeRunner(
        task,
        reset_fn=lambda: model.wam_episode_reset(task),
        predict_fn=lambda chunk: model.wam_masked_get_action(
            chunk,
            task=task,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generate_video=False,
            train_experts=True,
        ),
    )
    actions = []
    runner.reset_episode()
    try:
        for chunk_index in range(4):
            chunk = dict(batch)
            if chunk_index == 0:
                chunk["images"] = batch["images"][:, :1]
            else:
                start = (chunk_index - 1) * 8
                chunk["images"] = batch["images"][:, start:start + 9]
            a0 = chunk_index * FASTWAM_ACTIONS_PER_CHUNK
            chunk["action"] = batch["action"][
                :, a0:a0 + FASTWAM_ACTIONS_PER_CHUNK
            ]
            chunk["state"] = batch["state"][:, chunk_index:chunk_index + 1]
            out = runner.predict(chunk, frame_count=chunk["images"].shape[1])
            actions.append(out["action"])
    finally:
        runner.reset_episode()
    return {"action": torch.cat(actions, dim=1)}


def _validate_eval_dataset_instructions(cfg) -> None:
    data_root = (
        OmegaConf.select(cfg, "dataset_root")
        or OmegaConf.select(cfg, "droid_data_root")
    )
    if not data_root:
        raise ValueError(
            "open-loop eval could not resolve a LeRobot dataset root from config"
        )
    instructions = load_lerobot_task_instructions(str(data_root))
    print(
        f"[eval] validated {len(instructions)} non-empty dataset instruction(s) "
        f"from {data_root}"
    )


@torch.no_grad()
def _run_joint_open_loop_window(model, batch, steps: int, cfg_scale: float):
    if batch["images"].shape[1] != FULL_WINDOW_RGB_FRAMES:
        raise ValueError("Joint-WAM requires 33 RGB frames per offline window")
    if batch["state"].shape[1] != 4 or batch["action"].shape[1] != 96:
        raise ValueError("Joint-WAM requires 4 states and 96 actions")
    head = model.action_head
    head.num_inference_steps = steps
    head.cfg_scale = cfg_scale
    head.moe_task = "joint_dreamzero"
    head.model._current_wam_task = "joint_dreamzero"

    def predict(chunk):
        if unified_inference_backend() == "dense":
            return head.wam_joint_dense_episode_step(
                model.prepare_input(chunk)[1],
                num_inference_steps=steps, guidance_scale=cfg_scale,
            )
        return model.lazy_joint_video_action_causal(chunk)

    runner = WAMEpisodeRunner(
        "joint_dreamzero",
        reset_fn=lambda: model.wam_episode_reset("joint_dreamzero"),
        predict_fn=predict,
    )
    actions = []
    runner.reset_episode()
    try:
        for index in range(4):
            chunk = dict(batch)
            start = (index - 1) * 8
            chunk["images"] = batch["images"][:, :1] if index == 0 else batch["images"][:, start:start + 9]
            chunk["state"] = batch["state"][:, index:index + 1]
            chunk["action"] = batch["action"][:, index * 24:(index + 1) * 24]
            result = runner.predict(chunk, frame_count=chunk["images"].shape[1])
            actions.append(result["action_pred"])
    finally:
        runner.reset_episode()
    return {"action": torch.cat(actions, dim=1)}


def run(cfg):
    if unified_inference_consumption_enabled():
        os.environ.setdefault("WAM_MOE_GRANULARITY", "token_group_level")
        os.environ.setdefault("WAM_UNIFIED_JOINT_CONSUMPTION", "1")
        if os.environ["WAM_MOE_GRANULARITY"] != "token_group_level":
            raise ValueError(
                "unified inference requires WAM_MOE_GRANULARITY=token_group_level"
            )
    cfg = apply_action_overrides(cfg)

    task = os.environ.get("WAM_MOE_TASK", "action_only_gigaworld_policy")
    n_eval = int(os.environ.get("WAM_EVAL_NUM_SAMPLES", "4"))
    steps = int(os.environ.get("WAM_EVAL_STEPS", "16"))
    execute_action_steps = int(os.environ.get("EXECUTE_ACTION_STEPS", "24"))
    if not 1 <= execute_action_steps <= 24:
        raise ValueError(
            f"EXECUTE_ACTION_STEPS must be in 1..24, got {execute_action_steps}"
        )
    norm = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    default_cfg_scale = (
        "1.0"
        if norm in ("ac_wm", "world_model", "wm", "embodied_world_model")
        else "5.0"
    )
    cfg_scale = float(os.environ.get("WAM_CFG_SCALE", default_cfg_scale))
    print(
        f"[eval] task={task} n_eval={n_eval} steps={steps} cfg={cfg_scale} "
        f"action_only_execute_steps={execute_action_steps}"
    )

    if os.environ.get("WAM_EVAL_DRY", "0") == "1":
        # Cheap check: config composed and key inference wiring resolves.
        print("[dry] model target:", cfg.model._target_)
        print("[dry] action_head moe:", OmegaConf.select(cfg, "action_head_cfg.config.moe"))
        print("[dry] train_dataset target:", cfg.train_dataset._target_)
        print("[dry] data_collator target:", cfg.data_collator._target_)
        print("[dry] wam_task_ratios:", OmegaConf.select(cfg, "wam_task_ratios"))
        print("[dry] OK: config composes; skipping model build.")
        return

    _validate_eval_dataset_instructions(cfg)

    # Single-process distributed group (dataset build asserts dist is initialized).
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    checkpoint = cfg.pretrained_model_path
    if checkpoint:
        expected = unified_inference_expected_config(
            unified=unified_inference_consumption_enabled()
        )
        effective = validate_effective_config(
            checkpoint,
            expected,
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
            print("[effective-config] open-loop contract validated")
    fast_full_load = bool(checkpoint) and fast_full_checkpoint_loading_enabled()
    if fast_full_load:
        # The checkpoint contains all components; do not read the original Wan
        # component files during model construction.
        with open_dict(cfg):
            cfg.action_head_cfg.config.skip_component_loading = True
        print(f"[eval] meta+assign loading complete checkpoint {checkpoint}")
        model = build_from_complete_checkpoint(lambda: instantiate(cfg.model), checkpoint)
    else:
        model = instantiate(cfg.model)
        if checkpoint:
            from safetensors.torch import load_file
            import json, glob

            ckpt = checkpoint
            idx = os.path.join(ckpt, "model.safetensors.index.json")
            if os.path.exists(idx):
                with open(idx) as f:
                    weight_map = json.load(f)["weight_map"]
                for shard in sorted(set(weight_map.values())):
                    model.load_state_dict(load_file(os.path.join(ckpt, shard)), strict=False)
            else:
                for sf in glob.glob(os.path.join(ckpt, "*.safetensors")):
                    model.load_state_dict(load_file(sf), strict=False)
            print(f"[eval] legacy-loaded checkpoint {ckpt}")

    # 16B float32 MoE DiT OOMs on a single GPU; cast just the DiT to bf16 (it runs
    # under autocast(bf16) in the inference methods, so ops stay consistent). VAE/
    # T5/CLIP keep their own dtypes and run outside autocast.
    if os.environ.get("WAM_EVAL_BF16", "1") == "1":
        model.action_head.model = model.action_head.model.to(dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    dataset = instantiate(cfg.train_dataset)
    collator = instantiate(cfg.data_collator)
    batch = _to_cuda(_first_batch(dataset, collator, n_eval))

    with torch.inference_mode():
        if norm == "joint_dreamzero":
            out = _run_joint_open_loop_window(model, batch, steps, cfg_scale)
        elif norm in ("ac_wm", "world_model", "wm", "embodied_world_model"):
            out = _run_acwm_open_loop_window(
                model, batch, steps=steps, cfg_scale=cfg_scale
            )
        elif norm in ("idm_lingbot_va", "lingbot_va", "idm"):
            # IDM uses cached autoregressive inference.
            wb = int(os.environ.get("WAM_IDM_WINDOW_BLOCKS", "0")) or None
            out = model.wam_idm_get_action_cached(
                batch,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                window_blocks=wb,
                teacher_force_history=True,
            )
        elif norm in (
            "action_only_fast_wam",
            "action_only_gigaworld_policy",
        ) and unified_inference_consumption_enabled():
            out = _run_masked_open_loop_window(
                model, batch, task=task, steps=steps
            )
        else:
            out = model.wam_masked_get_action(batch, task=task, num_inference_steps=steps, guidance_scale=cfg_scale)

    if "action" in out.keys():
        pred = out["action"].float().cpu()
        print(f"[eval] predicted action shape: {tuple(pred.shape)}")
        if "action" in batch.keys():
            eval_horizon = pred.shape[1]
            pred_eval = pred[:, :eval_horizon]
            gt = batch["action"].float().cpu()[
                :, :eval_horizon, : pred.shape[2]
            ]
            valid = batch["action_mask"].cpu()[
                :, :eval_horizon, :pred.shape[2]
            ]
            mse = masked_action_mse(
                pred_eval[:, :gt.shape[1], :gt.shape[2]], gt, valid
            ).item()
            print(f"[eval] action MSE vs GT (untrained checkpoints will be large): {mse:.6f}")
    if "video" in out.keys():
        print(f"[eval] generated video latent shape: {tuple(out['video'].shape)}")
    print("[eval] done.")

