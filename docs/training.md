# Training Guide

This guide covers the options behind the [Training](../README.md#training)
quick start: how the model is initialized, what the default recipe is, how to
launch on one or several nodes, and what a run writes to disk.

All commands run from the checkout root. Arguments of the form `key=value` are
Hydra overrides of [`configs/unified.yaml`](../configs/unified.yaml);
`python train.py --dry-run [overrides]` prints the composed configuration
without loading data or weights.

## 1. Initialization

| Start from | Overrides | Notes |
| --- | --- | --- |
| Released UniW(A)M initialization (recommended) | `assets_root=$UNIWAM_ASSETS dit_version=./checkpoints/UniWAM/pretrained` | Every FFN is already stored as five experts plus a zero-initialized router; the expert tensors are loaded right after the MoE layers are created, and loading fails if any are missing. |
| Dense Wan2.2-TI2V-5B | `assets_root=$UNIWAM_ASSETS` | Each dense FFN is copied into the five experts when the model is built; routers start at zero. |
| A complete UniW(A)M checkpoint | `pretrained_model_path=/path/to/checkpoint tokenizer_path=$UNIWAM_ASSETS/google/umt5-xxl` | Loads all components (DiT, encoders, VAE) from the checkpoint and starts a fresh optimizer. |

Action and state adapters are newly initialized in the first two cases. The VAE,
text encoder and CLIP image encoder always come from `assets_root` unless
`vae_pretrained_path`, `text_encoder_pretrained_path` and
`image_encoder_pretrained_path` are set individually (see
[Model Assets](../README.md#model-assets)).

## 2. Default recipe

| Setting | Value | Override key |
| --- | --- | --- |
| Interfaces | Joint, IDM, Action-Centric, Action-Only, AC-WM, 0.2 each | `wam_task_ratios` |
| MoE routing | Token-group routing (CATR) | `action_head_cfg.config.moe.routing_granularity` |
| Window | 33 frames, 4 chunks of 24 actions, 4 states | `num_frames`, `action_horizon`, `max_chunk_size` |
| Resolution | 320 x 160 per view, three views | `image_resolution_width`, `image_resolution_height` |
| Action-loss weight (action-only interfaces) | 5.0 | `action_only_action_loss_weight` |
| Optimizer | AdamW, LR `1e-5`, weight decay `1e-5` | `learning_rate`, `weight_decay` |
| Schedule | 1,000 warmup steps, then constant | `training_args.warmup_steps`, `lr_scheduler_type` |
| Batch | 3 per GPU, 192 global | `per_device_train_batch_size`, `global_batch_size` |
| Length | 100,000 steps, checkpoint every 5,000 | `max_steps`, `save_steps` |
| Precision / sharding | BF16, DeepSpeed ZeRO-2 with overlap | `bf16`, `training_args.deepspeed` |

Joint-WAM and IDM-WAM are trained with video and action losses; the two
action-only interfaces keep the video loss and weight the action loss by 5;
AC-WM uses the video loss only.

## 3. Launching

Single node:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  assets_root="$UNIWAM_ASSETS" dataset_root="$UNIWAM_DATA" \
  dit_version=./checkpoints/UniWAM/pretrained \
  output_dir=./outputs/uniwam
```

Several nodes use the standard `torchrun --nnodes --node_rank --master_addr
--master_port` options with the same command on every node; see the README for
a two-node example. All nodes must see the same assets, dataset and output
directory.

Gradient accumulation is derived from the launch:
`global_batch_size / (WORLD_SIZE * per_device_train_batch_size)` must be a
positive integer, so the global batch stays 192 regardless of how many GPUs are
used. Use `training_args.deepspeed=/path/to/config.json` to supply another
DeepSpeed configuration, and `gradient_checkpointing=true` to reduce activation
memory.

## 4. Outputs

```text
outputs/uniwam/
  checkpoint-5000/
    config.json
    model.safetensors.index.json, model-*.safetensors
    experiment_cfg/            # metadata.json (normalization), effective_config.json
    ...                        # optimizer, scheduler and RNG states
  ...
```

Each `checkpoint-N/` is a complete model checkpoint that can be passed to
`infer.py` or to a new run as `pretrained_model_path`. Experiment tracking is
off by default; set `report_to=wandb wandb_project=<name>` to enable it.

## 5. Resuming

Rerunning the same command with the same `output_dir` resumes from the latest
`checkpoint-N/`, restoring the model, optimizer, scheduler and RNG states.
Passing `pretrained_model_path` instead starts a new run from the weights only;
use a new `output_dir` for it.

## 6. Troubleshooting

| Message / symptom | Cause | Fix |
| --- | --- | --- |
| `global_batch_size must be a positive multiple of ...` | Batch sizes do not divide across processes | Adjust `global_batch_size` or `per_device_train_batch_size` |
| `Missing dataset metadata: .../meta/...` | Dataset is not in the expected layout | Prepare it as described in [Data](data.md) |
| `Missing DiT directory` / `Missing *_pretrained_path` | Asset paths do not match | Check `assets_root`, `dit_version` and the individual asset paths |
| `pre-expanded MoE weights do not match the model` | `dit_version` points at an incompatible MoE checkpoint | Use the released `pretrained/` folder or a dense Wan2.2 DiT |
| CUDA out of memory | Activations exceed GPU memory | `gradient_checkpointing=true`, or lower `per_device_train_batch_size` (the global batch is kept by accumulation) |
