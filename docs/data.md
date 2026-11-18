# Data Format

All datasets go through the same trainer and model; a dataset only needs to be
in the trajectory layout below and match a data adapter in
[`groot/vla/configs/data/dreamzero/`](../groot/vla/configs/data/dreamzero/).

## 1. Directory layout

LeRobot v2 storage with a few additional metadata files:

```text
trajectories/
  meta/
    info.json          # features, FPS and file templates
    modality.json      # names and vector slices of state/action/video keys
    embodiment.json    # embodiment tag
    stats.json         # per-key statistics
    tasks.jsonl        # task_index -> language instruction
    episodes.jsonl     # episode_index, length
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<camera-key>/episode_000000.mp4
```

## 2. Included adapter

[`trajectory.yaml`](../groot/vla/configs/data/dreamzero/trajectory.yaml) is the
default adapter (`agilex_eef` embodiment tag), used for the real-world AgileX
dual-arm tasks:

| Modality | Keys / layout |
| --- | --- |
| Video | `video.head`, `video.left_wrist`, `video.right_wrist`; 30 FPS |
| State | `left_pos` (3), `left_rot6d` (6), `left_gripper` (1), then the same for the right arm: 20 dims |
| Action | Same 20-dim layout, absolute target end-effector poses |
| Language | `annotation.task`, resolved from `meta/tasks.jsonl` |

- Rotations use the 6D representation (first two columns of the rotation
  matrix).
- State and action values are used as given (tensor conversion and
  concatenation, no extra normalization) and should lie in `[-1, 1]`; training
  samples with values outside this range are skipped.
- Vectors are zero-padded to 64 channels with validity masks inside the model.
- The three views are tiled into one canvas (head view on the top row, wrist
  views below) and resized to 320 x 160 per view.

## 3. Preparing metadata

[`tools/data/prepare_metadata.py`](../tools/data/prepare_metadata.py) reads an
existing LeRobot v2 dataset (parquet + videos) and writes the extra metadata
into a separate output copy; it does not modify parquet files or videos. For a
20-dim EEF dataset with `observation.state`, `action`, `task_index` and camera
keys ending in `head`, `left_wrist` and `right_wrist`:

```bash
EEF_KEYS='{"left_pos":[0,3],"left_rot6d":[3,9],"left_gripper":[9,10],"right_pos":[10,13],"right_rot6d":[13,19],"right_gripper":[19,20]}'

python -m tools.data.prepare_metadata \
  --dataset-path /path/to/converted_eef_data \
  --output-path "$UNIWAM_DATA" \
  --embodiment-tag agilex_eef \
  --state-keys "$EEF_KEYS" --action-keys "$EEF_KEYS" \
  --task-key task_index
```

`annotation.task` maps to the numeric `task_index` column; the instruction
strings come from `meta/tasks.jsonl`. Run `python -m tools.data.prepare_metadata
--help` for all options. The tool writes metadata only: converting joint-space
actions to end-effector poses or changing camera layouts is done beforehand.

## 4. Training windows

Each training sample is a four-chunk window: 33 video frames, 96 actions
(4 x 24) and 4 states, with frames sampled at offsets aligned to the action
steps rather than as 33 consecutive frames. Episodes shorter than one window
are skipped. Camera timestamps, FPS, action/state timing, gripper convention
and rotation representation must agree with the adapter.

## 5. Other embodiments and benchmarks

Convert observations and actions to this layout, or add a data configuration
with your own modality keys and select it with `data=dreamzero/<adapter>`;
`train.py` and `infer.py` stay the same. LIBERO and RoboTwin 2.0 conversion and
evaluation scripts will be released together with their checkpoints.
