from dataclasses import dataclass, field
import logging
from typing import Tuple

from hydra.utils import instantiate
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3
logger = logging.getLogger(__name__)


@dataclass
class VLAConfig(PretrainedConfig):
    model_type = "vla"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class VLA(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = VLAConfig
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: VLAConfig,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.backbone = instantiate(config.backbone_cfg)
        self.action_head = instantiate(config.action_head_cfg)
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] % self.action_horizon == 0
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):

        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def wam_masked_get_action(
        self,
        inputs: dict,
        task: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generate_video: bool = False,
        train_experts: bool = False,
    ) -> BatchFeature:
        """Open-loop inference for the two action-only masked tasks.

        AC-WM/world_model must use ``wam_acwm_episode_step`` so its raw4,
        route, CLIP-window and decoder state cannot be bypassed.
        ``train_experts`` is retained for compatibility with older callers."""
        from .modules.wam_task_attn import canonical_masked_task
        from .wam_episode import unified_inference_consumption_enabled

        resolved_task = task or self.action_head.moe_task
        if canonical_masked_task(resolved_task) == "world_model":
            raise RuntimeError(
                "world_model single-pass inference is retired; use "
                "wam_acwm_episode_step() and wam_acwm_episode_reset()"
            )
        _, action_inputs = self.prepare_input(inputs)
        if (
            unified_inference_consumption_enabled()
            and self.action_head.moe_enabled
        ):
            if generate_video:
                raise ValueError("unified Fast/Giga inference is action-only")
            return self.action_head.wam_masked_episode_step(
                action_inputs,
                task=resolved_task,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        return self.action_head.wam_masked_get_action(
            action_inputs, task=task, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generate_video=generate_video,
            train_experts=train_experts
        )

    def wam_idm_get_action(
        self,
        inputs: dict,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        """Compatibility one-chunk IDM episode (video phase, then action phase).

        Uses the same persistent cache implementation as open-loop and robot
        deployment; guidance_scale>1 applies only to the video phase.
        """
        _, action_inputs = self.prepare_input(inputs)
        return self.action_head.wam_idm_get_action(
            action_inputs, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
        )

    def wam_idm_get_action_cached(
        self,
        inputs: dict,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        num_frame_per_block: int | None = None,
        window_blocks: int | None = None,
        teacher_force_history: bool = False,
        train_experts: bool = True,
    ) -> BatchFeature:
        """Compatibility 33/96/4 replay over the persistent IDM episode API.

        ``teacher_force_history`` must be true. ``train_experts=False`` is no
        longer supported because token-group stages select experts directly.
        """
        _, action_inputs = self.prepare_input(inputs)
        return self.action_head.wam_idm_get_action_cached(
            action_inputs, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            num_frame_per_block=num_frame_per_block, window_blocks=window_blocks,
            teacher_force_history=teacher_force_history, train_experts=train_experts,
        )

    def wam_idm_episode_start(
        self,
        inputs: dict,
        window_chunks: int = 4,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        """Start the real-robot IDM cache with the episode seed observation."""
        _, action_inputs = self.prepare_input(inputs)
        start_kwargs = {"window_chunks": window_chunks}
        if guidance_scale is not None:
            start_kwargs["guidance_scale"] = guidance_scale
        return self.action_head.wam_idm_episode_start(
            action_inputs, **start_kwargs,
        )

    def wam_idm_episode_predict(
        self, inputs: dict, num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        """Predict a provisional 2-latent/24-action chunk."""
        _, action_inputs = self.prepare_input(inputs)
        return self.action_head.wam_idm_episode_predict(
            action_inputs, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def wam_idm_episode_commit(self, inputs: dict) -> BatchFeature:
        """Replace the pending prediction with real observation/action history."""
        _, action_inputs = self.prepare_input(inputs)
        return self.action_head.wam_idm_episode_commit(action_inputs)

    def wam_idm_episode_step(
        self,
        inputs: dict,
        teacher_force_history: bool = True,
        train_experts: bool = True,
        use_input_action_history: bool = False,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        window_chunks: int = 4,
    ) -> BatchFeature:
        """Real-robot episode step matching the agilex pipeline contract.

        The first call writes the seed then predicts.  Every later call replaces
        the preceding provisional chunk with the newly arrived real eight-frame
        observation and the exact normalized 24-action chunk returned previously,
        then predicts the next chunk.
        """
        if not teacher_force_history:
            raise ValueError("episode_step requires teacher_force_history=True")
        if not train_experts:
            raise ValueError("episode_step requires train_experts=True")

        try:
            _, action_inputs = self.prepare_input(inputs)
            pending_action = getattr(self, "_wam_idm_pending_action", None)
            if pending_action is None:
                start_kwargs = {"window_chunks": window_chunks}
                if guidance_scale is not None:
                    start_kwargs["guidance_scale"] = guidance_scale
                self.action_head.wam_idm_episode_start(
                    action_inputs, **start_kwargs,
                )
            else:
                # Both adapters use this exact atomic transition.  The robot commits
                # the normalized action returned on the prior request, while the
                # open-loop teacher-forcing adapter explicitly commits its GT action.
                if use_input_action_history:
                    executed_action = action_inputs.get("action")
                    if executed_action is None:
                        raise ValueError(
                            "IDM teacher-forced commit requires input action history"
                        )
                else:
                    executed_action = pending_action
                action_inputs["action"] = executed_action
                self.action_head.wam_idm_episode_commit(action_inputs)

            output = self.action_head.wam_idm_episode_predict(
                action_inputs, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            self._wam_idm_pending_action = output["action"].detach().clone()
            return output
        except Exception:
            try:
                self.wam_idm_episode_reset()
            except Exception:
                logger.exception("Failed to reset outer IDM state after inference error")
            raise

    def wam_idm_episode_reset(self) -> None:
        """Clear one episode's IDM ring and layer KV caches."""
        try:
            self.action_head.wam_idm_episode_reset()
        finally:
            self._wam_idm_pending_action = None

    def wam_acwm_episode_step(
        self,
        inputs: dict,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> BatchFeature:
        """Predict one AC-WM chunk with the selected unified backend."""
        from .wam_episode import unified_inference_consumption_enabled
        from .action_head.wam_unified_inference import unified_inference_backend

        _, action_inputs = self.prepare_input(inputs)
        if (
            unified_inference_consumption_enabled()
            and self.action_head.moe_enabled
        ):
            if unified_inference_backend() == "cache":
                return self.action_head.wam_acwm_reference_cache_episode_step(
                    action_inputs,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
            return self.action_head.wam_acwm_unified_episode_step(
                action_inputs,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        if self.action_head._wam_acwm_episode is None:
            self.action_head.wam_acwm_episode_start(
                action_inputs, guidance_scale=guidance_scale
            )
        return self.action_head.wam_acwm_episode_predict(
            action_inputs,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def wam_acwm_episode_reset(self) -> None:
        """Clear both unified and legacy AC-WM episode state."""
        self.action_head.wam_episode_reset("ac_wm")

    def wam_episode_reset(self, runtime_mode: str | None = None) -> None:
        """Clear every deployment episode state through one public facade.

        Resetting all persistent backends is intentional: a runtime mode switch
        must not leave Joint KV state or an IDM provisional chunk reachable by
        the next mode.  FastWAM/GigaWorld are stateless between horizons, but
        still use this facade so both adapters share one lifecycle contract.
        """
        self.action_head.wam_episode_reset(runtime_mode)
        self._wam_idm_pending_action = None

    def wam_joint_episode_reset(self) -> None:
        """Thin public wrapper for the canonical Joint cache reset."""
        self.action_head.wam_joint_episode_reset()

    def joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.joint_video_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action_causal(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs, latent_video=latent_video)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action_causal_gt_cond(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        action_head_outputs = self.action_head.lazy_joint_video_action_causal_gt_cond(backbone_outputs, action_inputs, latent_video=latent_video)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def lazy_joint_video_action_efficient(
        self,
        inputs: dict,
        prompt_embs: torch.Tensor | None = None,
        prompt_emb_nega: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action_efficient(backbone_outputs, action_inputs, prompt_embs=prompt_embs, prompt_emb_nega=prompt_emb_nega)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def gt_video_action_pred(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.gt_video_action_pred(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def get_language(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone.generate(backbone_inputs)
        return backbone_outputs

    def get_video(
        self,
        inputs: dict,
    ) -> BatchFeature:
        _, video_inputs = self.prepare_input(inputs)
        video_outputs = self.action_head.get_video(video_inputs)
        return video_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Non-tensor leaves (e.g. the WAM per-sample "wam_task" list[str]) pass
            # through untouched.
            if not torch.is_tensor(x):
                return x
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs


    @classmethod
    def from_pretrained_for_tuning(
        cls, 
        pretrained_model_name_or_path: str,
        config: VLAConfig = None,  # This config will now be USED
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        offload_state_dict: bool = True,
        lora_weights_path: str | None = None,
    ):
        if config is None:
            raise ValueError(
                "A `config` object must be provided to build the model structure."
            )

        import os
        import json
        import gc
        from safetensors.torch import load_file

        model = cls(config)

        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        if os.path.exists(safetensors_index_path):
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            missing_keys_accum = set()
            unexpected_keys_accum = set()
            shard_files = sorted(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                missing_keys, unexpected_keys = model.load_state_dict(shard_state_dict, strict=False)
                if missing_keys:
                    missing_keys_accum.update(missing_keys)
                if unexpected_keys:
                    unexpected_keys_accum.update(unexpected_keys)
                # Free shard immediately
                del shard_state_dict
                gc.collect()
            if missing_keys_accum:
                print(f"Missing keys when loading sharded pretrained weights: {sorted(missing_keys_accum)} ... total={len(missing_keys_accum)}")
            if unexpected_keys_accum:
                print(f"Unexpected keys when loading sharded pretrained weights: {sorted(unexpected_keys_accum)} ... total={len(unexpected_keys_accum)}")
            if not missing_keys_accum and not unexpected_keys_accum:
                print("Successfully loaded pretrained base weights (sharded)")
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict = load_file(safetensors_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                print("Successfully loaded pretrained base weights")
        else:
            raise FileNotFoundError(
                f"No weights found at '{pretrained_model_name_or_path}'. "
                "Expected 'model.safetensors' or 'model.safetensors.index.json'."
            )

        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}")
            model.load_lora_weight(lora_weights_path)
        else:
            if hasattr(model, 'action_head') and hasattr(model.action_head, 'inject_lora_after_loading') and model.action_head.config.defer_lora_injection:
                print("Injecting LoRA adapters into action_head after loading pretrained weights")
                model.action_head.inject_lora_after_loading()
        
        print(f"{cls}\n")
        return model

    @classmethod
    def load_lora(
        cls, 
        pretrained_model_name_or_path: str
    ): 
        from safetensors.torch import load_file
        import os
        import json
        print("loading lora@@@@@")

        # Check for different checkpoint formats
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
        
        state_dict = {}
        if os.path.exists(safetensors_index_path):
            # Handle sharded safetensors
            print(f"Loading sharded safetensors using index: {safetensors_index_path}")
            
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            
            # Load each shard
            for shard_file in set(index["weight_map"].values()):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
                
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict.update(load_file(safetensors_path))
        
        # Load config
        print("loading config@@")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = VLAConfig(**config_dict)
        print("loading model")

        # Disable defer_lora_injection so LoRA layers are created during init,
        # matching the PEFT key hierarchy (base_model.model.*) in the checkpoint.
        ah_cfg = config.action_head_cfg
        inner = ah_cfg.get('config', ah_cfg) if isinstance(ah_cfg.get('config'), dict) else ah_cfg
        if 'defer_lora_injection' in inner:
            inner['defer_lora_injection'] = False
            print("defer_lora_injection disabled for load_lora")
        # Enable component loading so DiT base weights are loaded from pretrained
        if 'skip_component_loading' in inner:
            inner['skip_component_loading'] = False
            print("skip_component_loading disabled for load_lora")

        # Instantiate model (LoRA layers now exist from init)
        model = cls(config)

        # Remove .base_layer from keys if present
        has_base_layer = any(".base_layer." in key for key in state_dict.keys())
        if has_base_layer:
            print("Removing '.base_layer' from state dict keys")
            state_dict = {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        print("Successfully loaded pretrained weights")

        print(f"{cls}\n")
        return model

    def load_lora_weight(self, pretrained_model_name_or_path: str):
        """Load only LoRA weights from a pretrained model without loading config."""
        from safetensors.torch import load_file
        import os
        import json
        
        print(f"Loading LoRA weights from {pretrained_model_name_or_path}")
        
        # Check for different checkpoint formats
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        state_dict = {}
        if os.path.exists(safetensors_index_path):
            # Handle sharded safetensors
            print(f"Loading sharded safetensors using index: {safetensors_index_path}")
            
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            
            # Load each shard
            for shard_file in set(index["weight_map"].values()):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
                
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict.update(load_file(safetensors_path))
        else:
            raise FileNotFoundError(f"No valid checkpoint found at {pretrained_model_name_or_path}")
        
        print("Loading LoRA weights into existing model")

        def rewrite_lora_state_dict_keys(state_dict, pattern, repl):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace(pattern, repl)
                new_state_dict[new_k] = v
            return new_state_dict

        has_target_pattern = any("action_head.model.base_model.model" in key for key in state_dict.keys())
        
        if not has_target_pattern:
            print("Rewriting LoRA state dict keys from 'action_head.model' to 'action_head.model.base_model.model'")
            state_dict = rewrite_lora_state_dict_keys(
                state_dict,
                pattern="action_head.model",
                repl="action_head.model.base_model.model",
            )
        else:
            print("State dict already has 'action_head.model.base_model.model' pattern, skipping key rewrite")
        
        # Load only the weights into the existing model
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        print("Successfully loaded LoRA state dict")
            
        if missing_keys:
            print(f"Missing keys when loading LoRA weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading LoRA weights: {unexpected_keys}")
        
        print("Successfully loaded LoRA weights")

    @classmethod
    def from_config_with_lora_weights(
        cls,
        config: VLAConfig,
        pretrained_model_path: str,
    ):
        """Create VLA model from config and then load LoRA weights from pretrained model."""
        print(f"Creating VLA model from config and loading LoRA weights from {pretrained_model_path}")
        
        # 1. Create model from config (similar to vla.yaml)
        model = cls(config)
        print("Model created from config")
        
        # 2. Load LoRA weights into the created model
        model.load_lora_weight(pretrained_model_path)
        
        return model

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        config: VLAConfig = None
    ):
        del config
        import json
        import os

        from groot.vla.model.dreamzero.checkpoint_loading import (
            build_from_complete_checkpoint,
            checkpoint_shards,
            fast_full_checkpoint_loading_enabled,
        )

        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = VLAConfig(**config_dict)
        inner = config.action_head_cfg.get("config", config.action_head_cfg)
        if "defer_lora_injection" in inner:
            inner["defer_lora_injection"] = False

        if fast_full_checkpoint_loading_enabled():
            # Complete checkpoints supply Wan/T5/CLIP/VAE as well as the action
            # head, so component loading during construction would be redundant.
            inner["skip_component_loading"] = True
            print("[checkpoint] using meta+assign full-checkpoint loader")
            return build_from_complete_checkpoint(
                lambda: cls(config), pretrained_model_name_or_path
            )

        # Explicit rollback path: DREAMZERO_FULL_CHECKPOINT_LOAD_MODE=legacy.
        from safetensors.torch import load_file

        print("[checkpoint] using legacy in-memory loader")
        state_dict = {}
        shards, _ = checkpoint_shards(pretrained_model_name_or_path)
        for shard_path in shards:
            print(f"Loading shard: {shard_path}")
            for key, value in load_file(str(shard_path)).items():
                state_dict[key.replace(".base_layer.", ".")] = value
        model = cls(config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        print("Successfully loaded pretrained weights")
        return model

    def post_initialize(self):
        self.action_head.post_initialize()

    def parallelize(self, device_mesh: DeviceMesh):
        self.action_head.parallelize(device_mesh=device_mesh)


class CotrainVLA(VLA):

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        if "cotrain" in inputs and inputs["cotrain"]:
            return self.backbone.cotrain(inputs)
        return super().forward(inputs)


def create_vla_with_pretrained_action_head(pretrained_vla_path: str, config: VLAConfig):
    # 1. Instantiate a new VLAModel
    vla = VLA(config)

    # 2. Load the pretrained VLAModel
    pretrained_vla = VLA.from_pretrained(pretrained_vla_path)

    # 3. Replace the action head in the new VLAModel with the pretrained action head
    vla.action_head = pretrained_vla.action_head

    # 4. Replace the action head config in the new VLAModel with the pretrained action head config
    vla.config.action_head_cfg = pretrained_vla.config.action_head_cfg

    # 5. Return the new VLAModel
    return vla


# register
AutoConfig.register("vla", VLAConfig)
AutoModel.register(VLAConfig, VLA)
