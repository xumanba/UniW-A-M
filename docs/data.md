# Shared trajectory format

All datasets use the same trainer and inference model. Convert raw data once
into a common trajectory layout or define a data adapter in
`groot/vla/configs/data/dreamzero/`. Dataset names do not select separate
training algorithms.

```text
trajectories/
  meta/
    info.json
    modality.json
    embodiment.json
    stats.json
    tasks.jsonl
    episodes.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<camera-key>/episode_000000.mp4
```

File templates are defined by `meta/info.json`; the included adapter selects
30 FPS through `fps.agilex_eef` and source video timestamps must agree with it.
The metadata preparation
tool reads existing LeRobot v2 parquet/videos and generates the additional
metadata used by this loader:

```bash
python -m tools.data.prepare_metadata --help
```

It does not convert benchmark-specific observations or coordinate systems.
Review its options and use `--output-path` to prepare a separate dataset copy.
The model uses metadata to interpret vector slices; a correct tensor dimension
alone does not establish a correct action representation.

For an already converted 20D EEF dataset with `observation.state`, `action`,
`task_index`, a valid existing `meta/tasks.jsonl`, and camera keys ending in
`head`, `left_wrist`, `right_wrist`, prepare matching metadata explicitly:

```bash
EEF_KEYS='{"left_pos":[0,3],"left_rot6d":[3,9],"left_gripper":[9,10],"right_pos":[10,13],"right_rot6d":[13,19],"right_gripper":[19,20]}'
python -m tools.data.prepare_metadata \
  --dataset-path /path/to/converted_eef_data \
  --output-path /path/to/prepared_data \
  --embodiment-tag agilex_eef \
  --state-keys "$EEF_KEYS" --action-keys "$EEF_KEYS" \
  --task-key task_index
```

This maps `annotation.task` to the original numeric `task_index` column.
Existing task metadata supplies the actual language strings; numeric task IDs
are not language instructions. The tool does not turn joint-space actions
into EEF poses or change camera layouts.

## Included adapter

`trajectory.yaml` selects the `agilex_eef` schema tag (a format identifier) and
the existing three-camera, absolute EEF adapter:

| Modality | Keys / order |
| --- | --- |
| Video | `video.head`, `video.left_wrist`, `video.right_wrist` |
| State | `left_pos` (3), `left_rot6d` (6), `left_gripper` (1), then the same for the right arm |
| Action | The same 20-dimensional ordering, absolute target EEF poses |
| Language | `annotation.task`, resolved from task metadata |

The video keys must be present in `modality.json`; state/action names must map
to parquet vector slices with those dimensions. The original adapter's
state/action transform is tensor conversion plus concatenation, without an
extra normalization transform. The selected training contract checks valid
state/action values against the unit range; prepare values consistently or
explicitly adapt this requirement and the transforms to the dataset.
Model inputs are padded to 64 state/action channels with validity masks.

Training requires complete four-chunk windows (33 sampled video frames,
96 actions and 4 states); episodes that cannot form them are not valid training
windows. The loader uses temporally aligned frame offsets, not simply the
first 33 consecutive video frames. Camera timing, action/state timing, FPS,
gripper convention and rotation representation must agree with the adapter.

For LIBERO or RoboTwin, convert source observations/actions and metadata to
this contract, or add a matching data configuration and use
`data=dreamzero/<adapter>`. The root `train.py` and `infer.py` remain shared.
No native benchmark conversion or simulator rollout is claimed by this release.
