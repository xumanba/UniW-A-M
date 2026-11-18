# Inference Guide

One UniW(A)M checkpoint serves all five interfaces. This guide covers offline
evaluation with `infer.py`, the per-chunk episode API used for deployment, and
how to switch interfaces. For adaptive per-chunk interface selection, see
[PAIR](pair.md).

## 1. Checkpoint

A complete checkpoint contains `config.json`, `model.safetensors` (or
`model.safetensors.index.json` with all shards) and `experiment_cfg/`. It
includes the DiT, text/image encoders and VAE; only the umT5 tokenizer is passed
separately (`tokenizer_path`).

Checkpoints written by this code store `experiment_cfg/effective_config.json`,
which is validated against the launch configuration. The released
[`folding_clothes`](https://huggingface.co/xumanba/UniWAM/tree/main/folding_clothes)
checkpoint was saved without it; set `ALLOW_LEGACY_EFFECTIVE_CONFIG=1` when
loading it.

## 2. Offline evaluation

```bash
CUDA_VISIBLE_DEVICES=0 ALLOW_LEGACY_EFFECTIVE_CONFIG=1 \
WAM_EVAL_NUM_SAMPLES=1 WAM_EVAL_STEPS=16 \
python infer.py --task joint_dreamzero \
  pretrained_model_path=./checkpoints/UniWAM/folding_clothes \
  tokenizer_path="$UNIWAM_ASSETS/google/umt5-xxl" \
  dataset_root="$UNIWAM_DATA"
```

`infer.py` samples trajectory windows from `dataset_root`, predicts all four
chunks of each window with recorded observations as history, and prints the
action MSE over the valid (unpadded) action channels (`ac_wm` prints the shape
of the generated video latents instead). Random crop and color augmentation are
disabled.

| Variable | Default | Meaning |
| --- | --- | --- |
| `WAM_EVAL_NUM_SAMPLES` | 4 | Windows evaluated (offline batch size) |
| `WAM_EVAL_STEPS` | 16 | Denoising steps |
| `WAM_CFG_SCALE` | 5 (1 for `ac_wm`) | Classifier-free guidance scale |
| `WAM_EVAL_BF16` | 1 | Run the model in BF16 |

## 3. Interfaces

| Interface | `--task` | Inputs per chunk | Output |
| --- | --- | --- | --- |
| Joint-WAM | `joint_dreamzero` | Observations, state, instruction | Actions and future video, jointly |
| IDM-WAM | `idm_lingbot_va` | Observations, state, instruction | Future video, then actions |
| Action-Centric-WAM | `action_only_fast_wam` | Observations, state, instruction | Actions |
| Action-Only-WAM | `action_only_gigaworld_policy` | Observations, state, instruction | Actions |
| AC-WM | `ac_wm` | Observations and given actions | Future video |

AC-WM is an action-conditioned world model: it predicts video for supplied
actions and does not output executable actions.

## 4. Episode API

For closed-loop use, drive the model one action chunk at a time.
`WAMEpisodeRunner` ([`wam_episode.py`](../groot/vla/model/dreamzero/wam_episode.py))
owns the window lifecycle; requests are built with
`uniwam.inference.episode.BatchBuilder`.

A window has four chunks of 24 actions.

- **Chunk 0**: one RGB seed frame (the three views tiled into one canvas), the
  current state and the instruction.
- **Chunks 1-3**: the nine frames observed while the previous chunk was
  executed (action offsets `0, 3, ..., 24`), the current state and the
  instruction.

After chunk 3 the next request starts a new window with a seed frame. Online
integrations must pass the frames actually observed after execution.

| Interface | Model call |
| --- | --- |
| Joint-WAM | `model.lazy_joint_video_action_causal` |
| IDM-WAM | `model.wam_idm_episode_step` |
| Action-Centric-WAM / Action-Only-WAM | `model.wam_masked_get_action` |
| AC-WM | `model.wam_acwm_episode_step` (with the actions to condition on) |

By default the history of a window is kept in a KV cache across its chunks.
Set `WAM_UNIFIED_INFERENCE_BACKEND=dense` to recompute the full context every
chunk instead.

## 5. Switching interfaces

All interfaces share one set of weights, so switching needs no reload. When
switching, call `model.wam_episode_reset(<new task>)` and start the new
interface at chunk 0 from the latest observation; KV caches are never carried
across interfaces. [PAIR](pair.md) automates this choice for every chunk.

## 6. Troubleshooting

| Message / symptom | Cause | Fix |
| --- | --- | --- |
| `effective_config.json` missing or mismatched | Checkpoint saved without it (e.g. the released `folding_clothes`) | `ALLOW_LEGACY_EFFECTIVE_CONFIG=1` |
| `No complete model weights under ...` | Checkpoint folder lacks weights or shards | Download the whole checkpoint folder |
| `Missing tokenizer directory` | `tokenizer_path` is wrong | Point it at `Wan2.2-TI2V-5B/google/umt5-xxl` |
| Large action error on your own data | State/action layout differs from training | Check the vector order and coordinates in [Data](data.md) |
