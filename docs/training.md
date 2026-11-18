# Unified backbone training

Run all commands from the source checkout root after editable installation.
`configs/unified.yaml` composes the shared model/data/trainer configuration.
All `key=value` arguments use Hydra override syntax. `--dry-run` only composes
configuration and loads neither the model nor the dataset.

## Initialization assets

For fresh training, `assets_root` contains a dense Wan2.2-TI2V-5B DiT and:

```text
diffusion_pytorch_model.safetensors.index.json  # or a supported single file
diffusion_pytorch_model-*.safetensors           # if sharded
Wan2.2_VAE.pth
models_t5_umt5-xxl-enc-bf16.pth
models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
google/umt5-xxl/                               # tokenizer files
```

The CLIP component is inherited from the upstream DreamZero/Wan image encoder
path; obtain it from the upstream
[Wan2.1-I2V-14B-480P assets](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
if it is not in the 5B download. `dit_version`, `tokenizer_path` and the three
`*_pretrained_path` fields can point to separate local locations. No weights
are downloaded as part of the release checks.

The dense DiT initializes all MoE FFN experts; action/state adapters are newly
initialized. This route makes fresh training possible without distributing
the source experiment's private initializer. It does not recreate that
initializer's previous training history.

## Recipe and distributed launch

The selected code version uses five equally sampled interfaces, 33 frames /
96 actions / 4 states, online VAE encoding, token-group MoE and dense unified
training attention. Optimization uses AdamW, LR `1e-5`, weight decay `1e-5`,
effective batch 192 and a 1,000-step warmup followed by a constant LR, for
100,000 steps. This explicitly preserves the supplied version's warmup;
the paper's optimization table describes the backbone schedule as constant.

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  assets_root=/path/to/assets dataset_root=/path/to/trajectories \
  output_dir=./outputs/uniwam
```

The example does not assert that eight GPUs meet the memory requirement.
Choose a topology appropriate to your hardware. Multi-node training uses
standard `torchrun --nnodes --node_rank --master_addr --master_port` options.
There is no provider-specific launcher.

`global_batch_size` must be divisible by total training processes times
`per_device_train_batch_size`. The entry point computes gradient accumulation
from these values. The default DeepSpeed configuration is ZeRO-2 overlap.
Use a relative or absolute path in `training_args.deepspeed` to supply your own
DeepSpeed config. Activation checkpointing is controlled by
`gradient_checkpointing=true` if needed for the local memory budget.

## Weights and continuation

`pretrained_model_path=/path/to/complete/checkpoint` loads all model components
from a compatible model checkpoint and starts a fresh optimizer in a fresh
output directory. A complete checkpoint contains `config.json` and either
`model.safetensors` or `model.safetensors.index.json` with all referenced shards.
The tokenizer remains a separate asset.

Full continuation uses the existing trainer's saved model/optimizer/scheduler
and RNG state from an existing `output_dir`. Task-aware data continuation also
needs the exact acknowledged task micro-batch ordinal, exposed by the original
continuation utilities in `groot/vla/experiment/continuation.py`; it is not
equivalent to starting a new weights-only run. Exact full continuation has not
been validated as part of source cleanup. Use a fresh output directory for
an independent run.

Checkpoint files remain local under `output_dir`; cloud upload callbacks and
provider credential readers were excluded. Checkpoints written by users are
ignored by Git and must stay out of the source release. Experiment tracking is
disabled by default (`report_to=none`).
