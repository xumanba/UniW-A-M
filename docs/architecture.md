# Model and interfaces

UniW(A)M combines a shared Wan2.2-TI2V-5B diffusion Transformer with five FFN
experts per block: shared, VGM, VLA, WM and IDM. CATR selects allowed experts
by the role of each token group and learns mixing weights within that set.
Dense FFN weights initialize the experts; router weights start at zero.

| Component | Source |
| --- | --- |
| MoE experts and routing | `groot/vla/model/dreamzero/modules/wam_moe.py` |
| Interface-specific masks and token groups | `groot/vla/model/dreamzero/modules/wam_task_attn.py` |
| Shared DiT | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` |
| Flow matching and interface losses | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` |
| Unified dense/cache inference | `groot/vla/model/dreamzero/action_head/wam_unified_inference.py` |
| Episode/window lifecycle | `groot/vla/model/dreamzero/wam_episode.py` |

Each training window has 33 RGB frames, 96 actions, 4 states and a language
instruction. The VAE produces one seed latent plus four two-latent blocks.
Each block aligns to 24 action steps and one state. AC-WM uses clean actions
to predict video and does not consume state tokens.

Joint and IDM use video loss plus action loss. The two action-only training
interfaces retain video supervision and multiply action loss by 5. AC-WM uses
video loss only. The implementation retains historical task aliases and dense
compatibility code; the release entry points explicitly enable token-group
routing and unified consumption.
