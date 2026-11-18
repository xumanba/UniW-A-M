"""Compose one shared configuration without importing the model or CUDA."""

import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "joint_dreamzero", "idm_lingbot_va", "action_only_fast_wam",
    "action_only_gigaworld_policy", "ac_wm",
)


def compose_config(overrides=(), config="unified"):
    search = f"hydra.searchpath=[file://{ROOT / 'groot/vla/configs'}]"
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name=config, overrides=[search, *overrides])


def configure_runtime(mode, task):
    # Resolve the published contract explicitly rather than inheriting a previous run.
    values = {
        "WAM_UNIFIED_JOINT_CONSUMPTION": "1",
        "WAM_UNIFIED_INFERENCE_CONSUMPTION": "1",
        "WAM_UNIFIED_CHUNKWISE_ATTENTION": "0",
        "WAM_FORCE_TOKEN_GROUP_EXPERIMENT": "1",
        "WAM_MOE_GRANULARITY": "token_group_level",
        "WAM_MOE_PHASE": mode,
        "WAM_MOE_ROUTING_MODE": "per_sample" if mode == "train" else "global_task",
        "WAM_MOE_TASK": task,
        "WAM_MOE_PASS": "default",
    }
    os.environ.update(values)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def validate_inputs(cfg, mode):
    root = Path(cfg.dataset_root)
    for name in ("info.json", "modality.json", "tasks.jsonl", "episodes.jsonl"):
        if not (root / "meta" / name).is_file():
            raise FileNotFoundError(f"Missing dataset metadata: {root / 'meta' / name}")
    if mode == "infer" and not cfg.pretrained_model_path:
        raise ValueError("Inference requires pretrained_model_path=/path/to/your/checkpoint")
    if cfg.pretrained_model_path:
        checkpoint = Path(cfg.pretrained_model_path)
        if not any((checkpoint / n).is_file() for n in (
            "model.safetensors", "model.safetensors.index.json"
        )):
            raise FileNotFoundError(f"No complete model weights under {checkpoint}")
        # Complete checkpoints contain the DiT, encoders and VAE; only the tokenizer
        # remains external. The existing loader checks shard keys and tensor shapes.
        OmegaConf.update(cfg, "action_head_cfg.config.skip_component_loading", True, force_add=True)
    else:
        for key in ("text_encoder_pretrained_path", "image_encoder_pretrained_path", "vae_pretrained_path"):
            if not Path(cfg[key]).is_file():
                raise FileNotFoundError(f"Missing {key}: {cfg[key]}")
        if not Path(cfg.dit_version).is_dir():
            raise FileNotFoundError(f"Missing DiT directory: {cfg.dit_version}")
    if not Path(cfg.tokenizer_path).is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory: {cfg.tokenizer_path}")
    if mode == "train":
        world = int(os.environ.get("WORLD_SIZE", "1"))
        micro = int(cfg.per_device_train_batch_size)
        batch = int(cfg.global_batch_size)
        if micro <= 0 or batch <= 0 or batch % (world * micro):
            raise ValueError("global_batch_size must be a positive multiple of WORLD_SIZE * per_device_train_batch_size")
        OmegaConf.update(cfg, "gradient_accumulation_steps", batch // (world * micro))


def load_config(mode):
    parser = argparse.ArgumentParser(description=f"UniW(A)M unified {mode}")
    parser.add_argument("--config", default="unified", help="Config name under configs/")
    parser.add_argument("--dry-run", action="store_true", help="Compose config without datasets, weights or CUDA")
    if mode == "infer":
        parser.add_argument("--task", choices=TASKS, default="action_only_gigaworld_policy")
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    task = getattr(args, "task", "joint_dreamzero")
    configure_runtime(mode, task)
    overrides = list(args.overrides)
    if mode == "infer":
        overrides += ["wam_task_ratios=null", f"wam_task={task}"]
    cfg = compose_config(overrides, args.config)
    os.environ["WAM_TASK_RATIOS"] = (
        ",".join(f"{name}:{ratio}" for name, ratio in cfg.wam_task_ratios.items())
        if cfg.wam_task_ratios else ""
    )
    if mode == "infer":
        cfg.train_dataset.mixture_kwargs.training = False
        for transform in cfg.transforms.values():
            transform.transforms = [
                item for item in transform.transforms
                if str(item.get("_target_", "")).rsplit(".", 1)[-1]
                not in {"VideoCrop", "VideoColorJitter", "VideoRandomGrayscale", "VideoRandomPosterize"}
            ]
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(f"[dry-run] {mode} configuration composed; no model or data loaded")
    else:
        validate_inputs(cfg, mode)
        if mode == "train":
            # BaseTrainer requires distributed initialization, including on one GPU.
            import torch
            import torch.distributed as dist
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            if not dist.is_initialized():
                dist.init_process_group("nccl")
    return cfg, args
