# Shared inference

`infer.py` loads the same model for all five interfaces. It samples an offline
trajectory window, predicts actions/video and reports tensor dimensions and
action MSE over valid, unpadded channels where applicable. This is offline evaluation against recorded
observations; it does not execute commands in a simulator or on a robot.

```bash
WAM_EVAL_NUM_SAMPLES=1 WAM_EVAL_STEPS=16 \
python infer.py --task joint_dreamzero \
  dataset_root=/path/to/trajectories \
  pretrained_model_path=/path/to/your/checkpoint
```

The pretrained model must be a compatible complete UniW(A)M checkpoint with
all components, configuration and, by default, `effective_config.json`.
For a known compatible older checkpoint without that metadata, explicitly set
`ALLOW_LEGACY_EFFECTIVE_CONFIG=1`; the implementation still validates available
routing settings. No trained policy checkpoint is bundled in this repository.

The public entry point disables random crop/color augmentation and sets the
dataset mixture to evaluation mode. For the action-only interfaces, metrics
cover the complete predicted four-chunk window. Joint predictions also include
video internally, but the compact offline entry point reports action metrics;
AC-WM reports generated latent dimensions.

## Episode API

`WAMEpisodeRunner` owns reset and window boundaries. At the beginning of a
window, supply one observed RGB seed frame. For subsequent chunks, supply
nine observation frames aligned to the previous executed 24-step action chunk
(offsets `0,3,...,24`), plus the current state and language. A window has four
chunks, each of 24 actions. Offline action inference supplies recorded history;
online integrations must supply actual observations after execution.

| Interface | Model call |
| --- | --- |
| Joint | `model.lazy_joint_video_action_causal` (cache) or `head.wam_joint_dense_episode_step` (dense) |
| IDM | `model.wam_idm_episode_step` / cached episode API |
| Action-Centric / Action-Only | `model.wam_masked_get_action` |
| AC-WM | `model.wam_acwm_episode_step` with supplied clean actions |

The task-specific helper calls share the same model, checkpoint and lifecycle
contract. `uniwam/inference/open_loop.py` provides the four-chunk offline
dispatch; `episode.py` contains longer episode evaluation and tensor adapters.
The default unified backend is cache; set
`WAM_UNIFIED_INFERENCE_BACKEND=dense` for the dense compatibility path.

Default denoising uses 16 steps. Joint uses guidance 5; IDM uses guidance 5 for
video and no action guidance; the action-only interfaces use guidance 1.
AC-WM receives actions as conditions and does not produce executable actions.

When switching interfaces, reset the old runtime cache and start the new
interface from the latest observation. Do not transfer KV caches between
interfaces. The root command evaluates the fixed interface selected by `--task`
for the entire window.
