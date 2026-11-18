from typing import Any, TypeAlias

from groot.vla.model.dreamzero.modules.wan2_1_attention import (
    AttentionModule,
    _gpu_supports_flash_attention,
)
from groot.vla.model.n1_5.modules.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanRMSNorm,
    rope_action_apply,
    rope_masked_apply,
    rope_idm_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from groot.vla.model.dreamzero.modules.wam_task_attn import (
    WAM_MASKED_TASKS,
    build_acwm_two_history_chunk_mask,
    build_full_window_history_mask,
    build_full_window_history_spec,
    canonical_masked_task,
    build_task_self_attn_mask_from_lengths,
    build_idm_teacher_forcing_mask,
    build_idm_token_group_spec,
    build_joint_token_group_spec,
    is_joint_dreamzero_task,
)
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.nn.attention.flex_attention import BlockMask
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.distributed as dist
import os

ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"


def masked_state_timesteps(task, timestep_action, state_length):
    """Return per-state timestep modulation for one masked WAM sequence."""
    if state_length <= 0:
        return timestep_action[:, :0]
    if task in {"action_only_fast_wam", "action_only_gigaworld_policy"}:
        return torch.zeros_like(timestep_action[:, :state_length])
    stride = timestep_action.shape[1] // state_length
    return timestep_action[:, ::stride][:, :state_length]


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


def causal_rope_action_apply(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index):
    if ENABLE_TENSORRT:
        return causal_rope_action_apply_no_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)
    else:
        return causal_rope_action_apply_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)


def causal_rope_action_apply_no_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, d = x.shape
    
    # (B, seq_len, n, d) -> (B, seq_len, n, d/2, 2)
    x = x.reshape(B, seq_len, n, -1, 2)
    x_real = x[..., 0] 
    x_imag = x[..., 1] 
    
    # Split freqs into cos and sin components
    freqs = freqs.unsqueeze(0).view(1, freqs.shape[0], 1, -1, 2)
    freqs_cos = freqs[..., 0] # Shape: (1, seq_len', 1, d/2)
    freqs_sin = freqs[..., 1] # Shape: (1, seq_len', 1, d/2)
    
    #  Handle the Action/State Register Frequencies
    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        
        freqs_action_slice = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state_slice = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        
        # Combine the action/state tokens for this frame
        freqs_1d = torch.cat([freqs_action_slice, freqs_state_slice], dim=0).view(
            action_register_length, 1, -1, 2
        )
        
        # Split the new action/state frequencies
        freqs_cos_1d = freqs_1d[..., 0]
        freqs_sin_1d = freqs_1d[..., 1]

        # Append the action/state register sin/cos to the main sequence sin/cos
        freqs_cos = torch.cat([freqs_cos[0], freqs_cos_1d], dim=0).unsqueeze(0)
        freqs_sin = torch.cat([freqs_sin[0], freqs_sin_1d], dim=0).unsqueeze(0)
    
    x_real_rotated = x_real * freqs_cos - x_imag * freqs_sin
    x_imag_rotated = x_real * freqs_sin + x_imag * freqs_cos
    
    x_rotated = torch.stack((x_real_rotated, x_imag_rotated), dim=-1)
    
    return x_rotated.flatten(3)

def causal_rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)

    return x


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim)
        self.causal_attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim, causal=True)

    def _visualize_attention_mask(self, total_len, first_image_len, image_blocks_len, 
                                   action_len, state_len, num_image_blocks, 
                                   num_action_blocks, num_state_blocks,
                                   num_frame_per_block, frame_seqlen,
                                   num_action_per_block, num_state_per_block):
        """
        Create and print a visualization of the attention mask pattern.
        Returns a binary mask [total_len, total_len] where 1 = can attend, 0 = cannot attend.
        """
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Create mask tensor
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        
        # First image: self-attention only
        mask[first_image_start:first_image_end, first_image_start:first_image_end] = True
        
        # Image blocks
        for block_idx in range(num_image_blocks):
            block_start = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            
            # Attend to first image
            mask[block_start:block_end, first_image_start:first_image_end] = True
            
            # Attend to previous and current image blocks
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[block_start:block_end, image_kv_start:block_end] = True
            
            # Attend to current action block
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            mask[block_start:block_end, action_block_start:action_block_end] = True
            
            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[block_start:block_end, state_block_start:state_block_end] = True
        
        # Action blocks
        for block_idx in range(num_action_blocks):
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            
            # Attend to first image
            mask[action_block_start:action_block_end, first_image_start:first_image_end] = True
            
            # Attend to previous and current image blocks
            image_block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[action_block_start:action_block_end, image_kv_start:image_block_end] = True
            
            # Self-attention
            mask[action_block_start:action_block_end, action_block_start:action_block_end] = True
            
            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[action_block_start:action_block_end, state_block_start:state_block_end] = True
        
        # State blocks: self-attention only
        for block_idx in range(num_state_blocks):
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[state_block_start:state_block_end, state_block_start:state_block_end] = True
        
        return mask

    def _blockwise_causal_flash_attn(self, q, k, v, frame_seqlen, num_frame_per_block=1, 
                                       action_horizon=None, state_horizon=None, 
                                       num_action_per_block=None, num_state_per_block=None,
                                       visualize_mask=False):
        """
        Implement blockwise causal attention using flash_attention.
        Matches the pattern from _prepare_blockwise_causal_attn_mask:
        
        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image blocks + current action block + current state block
        - Action blocks: can attend to previous image blocks + current image block + current state block + first image
        - State blocks: conditioning only, cannot attend to anything
        
        Args:
            q, k, v: Query, key, value tensors [B, L, num_heads, head_dim]
            frame_seqlen: Number of tokens per frame
            num_frame_per_block: Number of frames per attention block
            action_horizon: Total number of action tokens (if None, no action/state tokens)
            state_horizon: Total number of state tokens (if None, no action/state tokens)
            num_action_per_block: Number of action tokens per block
            num_state_per_block: Number of state tokens per block
            visualize_mask: If True, print the attention mask pattern
        
        Returns:
            Attention output [B, L, num_heads, head_dim]
        """
        b, total_len, n, d = q.shape
        
        # Check if we have action/state tokens
        has_action_state = (action_horizon is not None and state_horizon is not None)
        
        if not has_action_state:
            # OPTIMIZED: Simple blockwise causal attention (without action/state tokens)
            num_frames = total_len // frame_seqlen
            block_size = frame_seqlen * num_frame_per_block
            num_blocks = (num_frames - 1) // num_frame_per_block
            
            # Handle edge case when sequence is too short (no blocks to process)
            if num_blocks <= 0:
                # Process entire sequence as a single block
                return self.attn(q, k, v)
            
            # OPTIMIZATION: For global attention, process all blocks in one call with causal masking
            if self.local_attn_size == -1:
                # Single flash_attention call with causal=True for all blocks at once
                # This is much faster than looping!
                return self.causal_attn(q, k, v)
            
            # With local attention, still need loop but optimize it
            # Pre-allocate output tensor
            output = torch.empty_like(q)
            
            # Pre-compute block boundaries
            block_starts = [frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            kv_starts = [max(0, end - self.local_attn_size * frame_seqlen) for end in block_ends]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                kv_start = kv_starts[block_idx]
                
                output[:, block_start:block_end] = self.attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end]
                )
            
            return output

        assert action_horizon is not None and state_horizon is not None
        assert num_action_per_block is not None and num_state_per_block is not None

        # Multi-modal structure: [first image] [image blocks] [action blocks] [state blocks]
        # Calculate block structure
        first_image_len = frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        image_blocks_len = total_len - first_image_len - action_len - state_len
        
        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block

        assert num_image_blocks == num_action_blocks == num_state_blocks
        
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Visualize attention mask if requested
        if visualize_mask:
            mask = self._visualize_attention_mask(
                total_len, first_image_len, image_blocks_len, 
                action_len, state_len, num_image_blocks,
                num_action_blocks, num_state_blocks,
                num_frame_per_block, frame_seqlen,
                num_action_per_block, num_state_per_block
            )
            
            print("\n" + "="*80)
            print("ATTENTION MASK VISUALIZATION")
            print("="*80)
            print(f"Total length: {total_len}")
            print(f"First image: [{first_image_start}:{first_image_end}] (len={first_image_len})")
            print(f"Image blocks: [{image_blocks_start}:{image_blocks_end}] (len={image_blocks_len}, num_blocks={num_image_blocks})")
            print(f"Action tokens: [{action_start}:{action_end}] (len={action_len}, num_blocks={num_action_blocks})")
            print(f"State tokens: [{state_start}:{state_end}] (len={state_len}, num_blocks={num_state_blocks})")
            print(f"Local attention size: {self.local_attn_size}")
            print("-"*80)
            
            # Print a downsampled version of the mask if it's too large
            if total_len <= 100:
                # Print full mask for small sequences
                print("Attention mask (1=can attend, 0=cannot attend):")
                print("Rows=Query tokens, Cols=Key tokens")
                for i in range(total_len):
                    row = "".join(["1" if mask[i, j] else "." for j in range(total_len)])
                    print(f"{i:4d}: {row}")
            else:
                # Print downsampled version for large sequences
                downsample = max(1, total_len // 100)
                print(f"Attention mask (downsampled by {downsample}x):")
                print("Rows=Query tokens, Cols=Key tokens (1=can attend, .=cannot attend)")
                for i in range(0, total_len, downsample):
                    row = "".join(["1" if mask[i, j] else "." for j in range(0, total_len, downsample)])
                    print(f"{i:4d}: {row}")
            
            # Save mask as image
            try:
                import cv2
                import numpy as np
                mask_np = mask.cpu().float().numpy()
                # Resize for visualization if needed
                if total_len > 1000:
                    mask_np = cv2.resize(mask_np, (1000, 1000), interpolation=cv2.INTER_NEAREST)
                mask_img = (mask_np * 255).astype(np.uint8)
                cv2.imwrite("attention_mask_blockwise_flash.png", mask_img)
                print(f"\nMask saved to: attention_mask_blockwise_flash.png")
            except Exception as e:
                print(f"Could not save mask image: {e}")
            
            print("="*80 + "\n")
        
        # OPTIMIZED: Pre-allocate output tensor and pre-compute all indices
        output = torch.empty_like(q)
        
        # Process first image (conditioning, can only self-attend)
        output[:, first_image_start:first_image_end] = self.attn(
            q[:, first_image_start:first_image_end],
            k[:, first_image_start:first_image_end],
            v[:, first_image_start:first_image_end]
        )
        
        # Pre-compute all block indices for image blocks
        image_block_starts = [image_blocks_start + i * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        image_block_ends = [image_blocks_start + (i + 1) * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        if self.local_attn_size != -1:
            image_kv_starts = [max(image_blocks_start, end - self.local_attn_size * frame_seqlen) for end in image_block_ends]
        else:
            image_kv_starts = [image_blocks_start] * num_image_blocks
        
        # Pre-compute action and state block indices
        action_block_starts = [action_start + i * num_action_per_block for i in range(num_action_blocks)]
        action_block_ends = [action_start + (i + 1) * num_action_per_block for i in range(num_action_blocks)]
        state_block_starts = [state_start + i * num_state_per_block for i in range(num_state_blocks)]
        state_block_ends = [state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)]
        
        # Process each image block
        for block_idx in range(num_image_blocks):
            block_start = image_block_starts[block_idx]
            block_end = image_block_ends[block_idx]
            image_kv_start = image_kv_starts[block_idx]
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # Build context: first image + relevant image blocks + current action + current state
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],  # First image
                k[:, image_kv_start:block_end],  # Image blocks
                k[:, action_block_start:action_block_end],  # Current action block
                k[:, state_block_start:state_block_end]  # Current state block
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, block_start:block_end] = self.attn(
                q[:, block_start:block_end], k_context, v_context
            )
        
        # Process each action block
        for block_idx in range(num_action_blocks):
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            image_block_end = image_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # Determine image context range
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            
            # Build context
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],  # First image
                k[:, image_kv_start:image_block_end],  # Image blocks
                k[:, action_block_start:action_block_end],  # Current action block
                k[:, state_block_start:state_block_end]  # Current state block
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:image_block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, action_block_start:action_block_end] = self.attn(
                q[:, action_block_start:action_block_end], k_context, v_context
            )
        
        # Process state blocks (conditioning, can only self-attend)
        for block_idx in range(num_state_blocks):
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            output[:, state_block_start:state_block_end] = self.attn(
                q[:, state_block_start:state_block_end],
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end]
            )
        
        return output

    def _process_clean_image_only(self, clean_image_q, clean_image_k, clean_image_v, clean_frames):
        """Process clean image blocks with causal attention pattern - OPTIMIZED
        
        First frame: conditioning, cannot attend to anything (self-attention only)
        Block i: attends to first frame + previous blocks (0 to i-1) + current block
        
        OPTIMIZATION: Instead of looping through blocks, we batch process them together
        by using a single flash_attention call with properly structured KV cache.
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (clean_frames - 1) // self.num_frame_per_block
        
        if num_blocks == 0:
            # Only first frame - single attention call
            return self.attn(
                clean_image_q[:, :self.frame_seqlen],
                clean_image_k[:, :self.frame_seqlen],
                clean_image_v[:, :self.frame_seqlen]
            )
        
        # Pre-allocate output tensor (avoids list append + cat overhead)
        b, total_len, n, d = clean_image_q.shape
        output = torch.empty_like(clean_image_q)
        
        # First frame: conditioning, self-attention only
        output[:, :self.frame_seqlen] = self.attn(
            clean_image_q[:, :self.frame_seqlen],
            clean_image_k[:, :self.frame_seqlen],
            clean_image_v[:, :self.frame_seqlen]
        )
        
        # OPTIMIZATION: Process all blocks together with causal masking
        # For global attention (no local_attn_size), we can process all blocks in one call
        if self.local_attn_size == -1:
            # Single attention call for all blocks!
            # Each position can attend to first_frame + everything up to itself
            blocks_q = clean_image_q[:, self.frame_seqlen:]
            blocks_k = clean_image_k  # Can attend to everything including first frame
            blocks_v = clean_image_v
            
            # Use causal masking: each block token can see first frame + all previous tokens
            output[:, self.frame_seqlen:] = self.causal_attn(
                blocks_q, blocks_k, blocks_v
            )
        else:
            # With local attention, we still need to loop but with optimizations
            # Pre-compute all block boundaries to reduce overhead
            block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                
                q_block = clean_image_q[:, block_start:block_end]
                
                # Context: first frame + recent blocks within local_attn_size
                image_kv_start = max(self.frame_seqlen, block_end - self.local_attn_size * self.frame_seqlen)
                k_context = torch.cat([
                    clean_image_k[:, :self.frame_seqlen],  # First frame
                    clean_image_k[:, image_kv_start:block_end]  # Recent blocks + current
                ], dim=1)
                v_context = torch.cat([
                    clean_image_v[:, :self.frame_seqlen],
                    clean_image_v[:, image_kv_start:block_end]
                ], dim=1)
                
                output[:, block_start:block_end] = self.attn(q_block, k_context, v_context)
        
        return output
    
    def _process_state_blocks(self, state_q, state_k, state_v, state_horizon):
        """Process state blocks: self-attention only - OPTIMIZED
        
        OPTIMIZATION: State blocks only do self-attention within each block.
        Instead of looping, we can process all blocks in a single call with block-diagonal masking,
        or even simpler: just one attention call since they're independent.
        """
        num_blocks = state_horizon // self.num_state_per_block
        
        if num_blocks == 1:
            # Single block - one attention call
            return self.attn(state_q, state_k, state_v)
        
        # OPTIMIZATION: Since each state block only attends to itself (no cross-block attention),
        # we can process all blocks in a single batched call. Flash attention will handle this
        # efficiently. The blocks are independent, so this is safe.
        # Alternative: reshape and process as separate batch items
        
        # Pre-allocate output
        output = torch.empty_like(state_q)
        
        # Process all blocks (keeping loop for now due to block-diagonal pattern)
        # This could be further optimized with custom masking
        for block_idx in range(num_blocks):
            state_block_start = block_idx * self.num_state_per_block
            state_block_end = state_block_start + self.num_state_per_block
            
            output[:, state_block_start:state_block_end] = self.attn(
                state_q[:, state_block_start:state_block_end],
                state_k[:, state_block_start:state_block_end],
                state_v[:, state_block_start:state_block_end]
            )
        
        return output
    
    def _process_noisy_image_blocks(self, noisy_image_q, noisy_image_k, noisy_image_v,
                                     clean_image_k, clean_image_v,
                                     noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                                     half_frames, action_horizon, state_horizon):
        """Process noisy image blocks with teacher forcing pattern - OPTIMIZED
        
        First frame: conditioning, cannot attend to anything (self-attention only)
        Block i: attends to action[i] + state[i] + first_clean_frame + clean_blocks[0:i] + current_noisy_block
        
        OPTIMIZATION: Pre-allocate output, pre-compute indices, reduce memory allocations
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        
        # Pre-allocate output tensor
        output = torch.empty_like(noisy_image_q)
        
        # First noisy frame: conditioning, self-attention only
        output[:, :self.frame_seqlen] = self.attn(
            noisy_image_q[:, :self.frame_seqlen],
            noisy_image_k[:, :self.frame_seqlen],
            noisy_image_v[:, :self.frame_seqlen]
        )
        
        if num_blocks == 0:
            return output
        
        # Pre-compute all block indices to reduce loop overhead
        noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        noisy_block_ends = [min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts]
        clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]
        
        # Process noisy image blocks
        for block_idx in range(num_blocks):
            noisy_start = noisy_block_starts[block_idx]
            noisy_end = noisy_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]
            
            q_block = noisy_image_q[:, noisy_start:noisy_end]
            
            # Build context: first_clean_frame + clean_blocks[0:i] + current_noisy_block + action[i] + state[i]
            k_context = torch.cat([
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end]
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end]
            ], dim=1)
            
            output[:, noisy_start:noisy_end] = self.attn(q_block, k_context, v_context)
        
        return output
    
    def _process_noisy_action_blocks(self, noisy_action_q, noisy_action_k, noisy_action_v,
                                      clean_image_k, clean_image_v,
                                      noisy_image_k, noisy_image_v,
                                      noisy_state_k, noisy_state_v,
                                      half_frames, action_horizon, state_horizon):
        """Process noisy action blocks with teacher forcing pattern - OPTIMIZED
        
        First action (for first frame): cannot attend to anything (self-attention only)
        Action block i: attends to first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] + state[i]
        
        OPTIMIZATION: Pre-allocate output, pre-compute indices, reduce memory allocations
        """
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        
        if num_blocks == 0:
            return torch.empty_like(noisy_action_q)
        
        # Pre-allocate output tensor
        output = torch.empty_like(noisy_action_q)
        
        # Pre-compute all block indices
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        clean_context_ends = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_starts = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_ends = [start + self.frame_seqlen * self.num_frame_per_block for start in noisy_image_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]
        
        # Process noisy action blocks
        for block_idx in range(num_blocks):
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            noisy_img_start = noisy_image_block_starts[block_idx]
            noisy_img_end = noisy_image_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]
            
            q_block = noisy_action_q[:, action_start:action_end]
            
            # Build context: first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] + state[i]
            k_context = torch.cat([
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_img_start:noisy_img_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end]
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_img_start:noisy_img_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end]
            ], dim=1)
            
            output[:, action_start:action_end] = self.attn(q_block, k_context, v_context)
        
        return output

    def _masked_full_attn(self, q, k, v, attn_mask):
        """Single full self-attention with a boolean [L, L] mask (True=allowed).

        q, k, v: [B, L, n, d]. attn_mask broadcasts over batch and heads. Used by
        the WAM masked tasks (world_model / fastwam / gigaworld). attn_mask=None
        means unmasked full attention (used by the idm KV-cache AR path).
        """
        dtype = torch.bfloat16
        out_dtype = q.dtype
        qt = q.transpose(1, 2).to(dtype)  # [B, n, L, d]
        kt = k.transpose(1, 2).to(dtype)
        vt = v.transpose(1, 2).to(dtype)
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask)
        return out.transpose(1, 2).to(out_dtype)  # [B, L, n, d]

    def _full_window_history_chunked_attn(self, q, k, v, spec):
        """Evaluate the full-window mask exactly, one query chunk at a time."""
        output = torch.empty_like(q)

        def attend(query_range, *key_ranges):
            if query_range is None:
                return
            q0, q1 = query_range
            ranges = [span for span in key_ranges if span is not None]
            key = torch.cat([k[:, start:end] for start, end in ranges], dim=1)
            value = torch.cat([v[:, start:end] for start, end in ranges], dim=1)
            output[:, q0:q1] = self.attn(q[:, q0:q1], key, value)

        def attend_chunks(items):
            if not items:
                return
            # FlashAttention's varlen path evaluates equal-width query chunks in
            # parallel while skipping padded KV. Other backends keep the exact
            # same computation with the simple per-chunk path.
            if (
                len(items) == 1
                or self.attn.backend not in {"FA2", "FA3"}
                or not _gpu_supports_flash_attention()
            ):
                for query_range, key_ranges in items:
                    attend(query_range, *key_ranges)
                return
            batch_size = q.shape[0]
            query_widths = [query_range[1] - query_range[0] for query_range, _ in items]
            if len(set(query_widths)) != 1:
                raise ValueError(
                    "Packed full-window attention requires equal query widths, "
                    f"got {query_widths}"
                )
            queries = []
            keys = []
            values = []
            key_lengths = []
            for query_range, key_ranges in items:
                q0, q1 = query_range
                ranges = [span for span in key_ranges if span is not None]
                key = torch.cat([k[:, start:end] for start, end in ranges], dim=1)
                value = torch.cat([v[:, start:end] for start, end in ranges], dim=1)
                queries.append(q[:, q0:q1])
                keys.append(key)
                values.append(value)
                key_lengths.extend([key.shape[1]] * batch_size)
            max_key_length = max(key.shape[1] for key in keys)

            def pad_to_max(tensor):
                padding = max_key_length - tensor.shape[1]
                return F.pad(tensor, (0, 0, 0, 0, 0, padding)) if padding else tensor

            packed = self.attn(
                torch.cat(queries, dim=0),
                torch.cat([pad_to_max(key) for key in keys], dim=0),
                torch.cat([pad_to_max(value) for value in values], dim=0),
                q_lens=torch.full(
                    (len(items) * batch_size,), query_widths[0],
                    dtype=torch.int32, device=q.device,
                ),
                k_lens=torch.tensor(
                    key_lengths, dtype=torch.int32, device=q.device,
                ),
            )
            for index, (query_range, _) in enumerate(items):
                q0, q1 = query_range
                output[:, q0:q1] = packed[
                    index * batch_size:(index + 1) * batch_size
                ]

        clean_start, _ = spec["clean_range"]
        noisy_seed = spec["noisy_seed_range"]
        seed_width = noisy_seed[1] - noisy_seed[0]
        clean_seed = (clean_start, clean_start + seed_width)
        attend(clean_seed, clean_seed)
        clean_items = []
        for chunk in spec["chunks"]:
            width = chunk["video_range"][1] - chunk["video_range"][0]
            clean_chunk = (chunk["history_range"][1], chunk["history_range"][1] + width)
            clean_items.append((clean_chunk, ((clean_start, clean_chunk[1]),)))
        attend_chunks(clean_items)

        first_state = spec["chunks"][0]["state_range"] if spec["n_state"] else None
        attend(noisy_seed, clean_seed, noisy_seed, first_state)
        video_items = []
        action_items = []
        state_items = []
        for chunk in spec["chunks"]:
            video_keys = [
                chunk["history_range"],
                noisy_seed,
                chunk["video_range"],
            ]
            if spec["task"] in {
                "world_model",
                "action_only_gigaworld_policy",
            }:
                video_keys.append(chunk["action_range"])
            video_keys.append(chunk["state_range"])
            video_items.append((chunk["video_range"], tuple(video_keys)))
            action_items.append((
                chunk["action_range"],
                (
                    chunk["history_range"], noisy_seed,
                    chunk["action_range"], chunk["state_range"],
                ),
            ))
            if chunk["state_range"] is not None:
                state_items.append((
                    chunk["state_range"],
                    (chunk["history_range"], noisy_seed, chunk["state_range"]),
                ))
        attend_chunks(video_items)
        attend_chunks(action_items)
        attend_chunks(state_items)
        return output

    # ---- idm autoregressive KV cache (lingbot-va style) ----
    # Clean-history K/V pool per layer. The current chunk's noisy tokens attend the
    # cached CLEAN history + themselves; when a chunk is finalized (clean), its K/V
    # is appended and the pool is truncated to a token window. Keys are cached
    # POST-RoPE (each token keeps its absolute position), so reuse is exact.
    def reset_idm_cache(self):
        self._idm_k = None
        self._idm_v = None
        self._idm_cache_chunks = []

    def clear_idm_pred_cache(self):
        """Drop provisional chunks without disturbing durable history."""
        chunks = getattr(self, "_idm_cache_chunks", [])
        self._idm_cache_chunks = [chunk for chunk in chunks if not chunk["is_pred"]]
        self._sync_legacy_idm_cache_view()

    def _sync_legacy_idm_cache_view(self):
        chunks = getattr(self, "_idm_cache_chunks", [])
        self._idm_k = torch.cat([c["k"] for c in chunks], dim=1) if chunks else None
        self._idm_v = torch.cat([c["v"] for c in chunks], dim=1) if chunks else None

    @staticmethod
    def _apply_pos_freqs(x, pos_freqs):
        """Apply RoPE to x [B, L, n, d] with complex freqs pos_freqs [L, 1, d/2]."""
        B, L, n, _ = x.shape
        xc = torch.view_as_complex(x.to(torch.float64).reshape(B, L, n, -1, 2))
        return torch.view_as_real(xc * pos_freqs.unsqueeze(0)).flatten(3)

    @staticmethod
    def _rope_masked_apply_batched(
        x,
        freqs,
        freqs_action,
        freqs_state,
        n_video,
        n_action,
        n_state,
    ):
        """Apply masked-task RoPE with one video position table per sample."""
        if freqs.ndim != 4:
            raise ValueError(
                "batched video RoPE must be [B,L,1,D], got "
                f"{tuple(freqs.shape)}"
            )
        batch_size, seq_len, num_heads, _ = x.shape
        freq_batch = freqs.shape[0]
        if freq_batch != batch_size:
            if freq_batch <= 0 or batch_size % freq_batch:
                raise ValueError(
                    "batched video RoPE batch does not divide query batch: "
                    f"freqs={freq_batch}, queries={batch_size}"
                )
            # CFG concatenates complete conditional/unconditional batches.
            freqs = freqs.repeat(batch_size // freq_batch, 1, 1, 1)
        parts = [freqs[:, :n_video]]
        if n_action:
            parts.append(
                freqs_action[:n_action]
                .view(1, n_action, 1, -1)
                .expand(batch_size, -1, -1, -1)
            )
        if n_state:
            parts.append(
                freqs_state[:n_state]
                .view(1, n_state, 1, -1)
                .expand(batch_size, -1, -1, -1)
            )
        freqs_full = torch.cat(parts, dim=1)
        if freqs_full.shape[1] != seq_len:
            raise ValueError(
                "masked RoPE length mismatch: "
                f"freqs={freqs_full.shape[1]}, tokens={seq_len}"
            )
        if freqs_full.is_complex():
            xc = torch.view_as_complex(
                x.to(torch.float64).reshape(
                    batch_size, seq_len, num_heads, -1, 2
                )
            )
            return torch.view_as_real(xc * freqs_full).flatten(3)
        x0, x1 = x.chunk(2, dim=-1)
        freqs_cos, freqs_sin = freqs_full.chunk(2, dim=-1)
        return torch.cat(
            [
                x0 * freqs_cos - x1 * freqs_sin,
                x1 * freqs_cos + x0 * freqs_sin,
            ],
            dim=-1,
        )

    def forward_idm_cached(
        self, x, pos_freqs, append_clean: bool, window_tokens: int = None,
        cache_chunk_id: int = None, is_pred: bool = False, max_cache_chunks: int = None,
        rolling_window_tokens: int = None,
        current_attn_mask: torch.Tensor | None = None,
        video_query_count: int = 0,
        exclude_history_tail_for_video: int = 0,
        cache_kind: str | None = None,
        visible_cache_kinds: tuple[str, ...] | None = None,
        cache_kind_order: tuple[str, ...] | None = None,
        append_token_range: tuple[int, int] | None = None,
    ):
        """Cached AR self-attention for one idm chunk.

        x: [B, L_cur, dim] tokens of the CURRENT chunk (all video, or all action).
        pos_freqs: [L_cur, 1, d/2] complex RoPE freqs at the chunk's ABSOLUTE positions.
        append_clean: if True, append this chunk's (clean) K/V to the history pool.
        rolling_window_tokens: ac_wm-only exact token window; older tokens are dropped.
        Returns the attention output [B, L_cur, dim].
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        rq = self._apply_pos_freqs(q, pos_freqs).type_as(v)
        rk = self._apply_pos_freqs(k, pos_freqs).type_as(v)

        chunks = getattr(self, "_idm_cache_chunks", [])
        visible_chunks = chunks
        if visible_cache_kinds is not None:
            visible = set(visible_cache_kinds)
            visible_chunks = [
                chunk
                for chunk in chunks
                if chunk.get("kind") in visible
            ]
            if cache_kind_order is not None:
                order = {
                    kind: index for index, kind in enumerate(cache_kind_order)
                }
                visible_chunks = sorted(
                    enumerate(visible_chunks),
                    key=lambda item: (
                        order.get(item[1].get("kind"), len(order)), item[0]
                    ),
                )
                visible_chunks = [chunk for _, chunk in visible_chunks]
            hk = (
                torch.cat([chunk["k"] for chunk in visible_chunks], dim=1)
                if visible_chunks
                else None
            )
            hv = (
                torch.cat([chunk["v"] for chunk in visible_chunks], dim=1)
                if visible_chunks
                else None
            )
        else:
            hk = getattr(self, "_idm_k", None)
            hv = getattr(self, "_idm_v", None)
        if hk is not None:
            K = torch.cat([hk, rk], dim=1)
            V = torch.cat([hv, v], dim=1)
        else:
            K, V = rk, v

        attn_mask = None
        if current_attn_mask is not None:
            if current_attn_mask.shape != (s, s):
                raise ValueError(
                    "IDM current attention mask must have shape "
                    f"{(s, s)}, got {tuple(current_attn_mask.shape)}"
                )
            history_len = 0 if hk is None else hk.shape[1]
            history_mask = torch.ones(
                s, history_len, dtype=torch.bool, device=x.device
            )
            if exclude_history_tail_for_video:
                video_query_count = int(video_query_count)
                tail = int(exclude_history_tail_for_video)
                if (
                    video_query_count <= 0
                    or video_query_count > s
                    or tail > history_len
                ):
                    raise ValueError(
                        "Invalid IDM video/history exclusion: "
                        f"video_queries={video_query_count}, tail={tail}, "
                        f"history={history_len}"
                    )
                history_mask[:video_query_count, history_len - tail:] = False
            attn_mask = torch.cat(
                [history_mask, current_attn_mask.to(device=x.device)], dim=1
            )
        out = self._masked_full_attn(rq, K, V, attn_mask=attn_mask)
        out = self.o(out.flatten(2))

        if append_clean:
            append_k, append_v = rk, v
            if append_token_range is not None:
                append_start, append_end = map(int, append_token_range)
                if not (0 <= append_start < append_end <= s):
                    raise ValueError(
                        "Invalid cached append token range: "
                        f"{append_token_range!r} for current length {s}"
                    )
                append_k = rk[:, append_start:append_end]
                append_v = v[:, append_start:append_end]
            chunk_id = int(cache_chunk_id) if cache_chunk_id is not None else (
                chunks[-1]["chunk_id"] + 1 if chunks else 0
            )
            if (
                chunks
                and chunks[-1]["chunk_id"] == chunk_id
                and chunks[-1].get("kind") == cache_kind
            ):
                chunks[-1]["k"] = torch.cat(
                    [chunks[-1]["k"], append_k], dim=1
                )
                chunks[-1]["v"] = torch.cat(
                    [chunks[-1]["v"], append_v], dim=1
                )
                chunks[-1]["is_pred"] = bool(is_pred)
            else:
                chunks.append({
                    "chunk_id": chunk_id,
                    "is_pred": bool(is_pred),
                    "kind": cache_kind,
                    "k": append_k,
                    "v": append_v,
                })
            if max_cache_chunks is not None and max_cache_chunks > 0:
                chunks[:] = chunks[-int(max_cache_chunks):]
            if (
                rolling_window_tokens is not None
                and int(rolling_window_tokens) > 0
            ):
                # AC-WM keeps a typed t=0 seed copy in addition to the clean
                # history window.  The configured limit applies to clean
                # history only; evicting seed tokens here would make the
                # rolling cache differ from the training [clean history +
                # noisy seed] carrier after the fourth chunk.
                rollable = [
                    chunk
                    for chunk in chunks
                    if chunk.get("kind") != "seed"
                ]
                excess = sum(chunk["k"].shape[1] for chunk in rollable) - int(
                    rolling_window_tokens
                )
                for chunk in rollable:
                    if excess <= 0:
                        break
                    width = chunk["k"].shape[1]
                    if excess >= width:
                        chunks[:] = [
                            candidate
                            for candidate in chunks
                            if candidate is not chunk
                        ]
                        excess -= width
                    else:
                        chunk["k"] = chunk["k"][:, excess:]
                        chunk["v"] = chunk["v"][:, excess:]
                        excess = 0
            self._idm_cache_chunks = chunks
            self._sync_legacy_idm_cache_view()
        return out

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        task_attn_mask: torch.Tensor | None = None,
        task_seg: tuple | None = None,
        task_rope_mode: str = "masked",
        task_attn_spec: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        updated_kv_cache: torch.Tensor | None = None

        # WAM masked-task path: single full attention with a per-task boolean mask.
        # "masked" layout [video|action|state] (task_seg=(n_video,n_action,n_state));
        # "idm" layout [noisy_V|clean_V|noisy_A|clean_A|S] (task_seg=(nV,nA,nS)).
        if kv_cache is None and (
            task_attn_mask is not None or task_attn_spec is not None
        ):
            if task_rope_mode in {
                "full_window_history",
                "full_window_history_chunked",
            }:
                nV, nA, nS = task_seg
                clean_q, noisy_q = q[:, :nV], q[:, nV:]
                clean_k, noisy_k = k[:, :nV], k[:, nV:]
                roped_query = torch.cat([
                    rope_masked_apply(
                        clean_q, freqs, freqs_action, freqs_state,
                        nV, 0, 0,
                    ),
                    rope_masked_apply(
                        noisy_q, freqs, freqs_action, freqs_state,
                        nV, nA, nS,
                    ),
                ], dim=1).type_as(v)
                roped_key = torch.cat([
                    rope_masked_apply(
                        clean_k, freqs, freqs_action, freqs_state,
                        nV, 0, 0,
                    ),
                    rope_masked_apply(
                        noisy_k, freqs, freqs_action, freqs_state,
                        nV, nA, nS,
                    ),
                ], dim=1).type_as(v)
            elif task_rope_mode == "idm":
                nV, nA, nS = task_seg
                roped_query = rope_idm_apply(q, freqs, freqs_action, freqs_state, nV, nA, nS).type_as(v)
                roped_key = rope_idm_apply(k, freqs, freqs_action, freqs_state, nV, nA, nS).type_as(v)
            else:
                n_video, n_action, n_state = task_seg
                if freqs.ndim == 3:
                    roped_query = rope_masked_apply(
                        q, freqs, freqs_action, freqs_state,
                        n_video, n_action, n_state,
                    ).type_as(v)
                    roped_key = rope_masked_apply(
                        k, freqs, freqs_action, freqs_state,
                        n_video, n_action, n_state,
                    ).type_as(v)
                else:
                    roped_query = self._rope_masked_apply_batched(
                        q, freqs, freqs_action, freqs_state,
                        n_video, n_action, n_state,
                    ).type_as(v)
                    roped_key = self._rope_masked_apply_batched(
                        k, freqs, freqs_action, freqs_state,
                        n_video, n_action, n_state,
                    ).type_as(v)
            if task_rope_mode == "full_window_history_chunked":
                if task_attn_spec is None:
                    raise ValueError("Chunked full-window attention requires task_attn_spec")
                x = self._full_window_history_chunked_attn(
                    roped_query, roped_key, v, task_attn_spec
                )
            else:
                x = self._masked_full_attn(roped_query, roped_key, v, task_attn_mask)
            # output projection (same tail as the other attention paths)
            x = x.flatten(2)
            x = self.o(x)
            return x, None

        if kv_cache is None:
            if is_tf:
                # Teacher forcing training.
                if action_register_length is not None:
                    q_context = q[:, :(s-action_register_length)//2]
                    k_context = k[:, :(s-action_register_length)//2]
                    q_noisy = q[:, (s-action_register_length)//2:]  
                    k_noisy = k[:, (s-action_register_length)//2:]
                else:
                    q_context = q[:, :s//2]
                    k_context = k[:, :s//2]
                    q_noisy = q[:, s//2:]
                    k_noisy = k[:, s//2:]
                roped_query = []
                roped_key = []

                # rope should be same for clean and noisy parts
                rq_context = rope_action_apply(
                    x=q_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)
                rk_context = rope_action_apply(
                    x=k_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)

                rq_noisy = rope_action_apply(
                    x=q_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                rk_noisy = rope_action_apply(
                    x=k_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                roped_query.append(rq_context)
                roped_key.append(rk_context)
                roped_query.append(rq_noisy)
                roped_key.append(rk_noisy)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
                # Calculate sequence dimensions
                half_seq_len = (s - (action_register_length if action_register_length is not None else 0)) // 2
                
                if action_register_length is not None:
                    # Teacher forcing structure:
                    # Clean half: [image tokens only]
                    # Noisy half: [image tokens][action tokens][state tokens]
                    # Causality only applies to image blocks!
                    
                    # Clean half contains ONLY image tokens
                    clean_image_seq_len = half_seq_len
                    clean_frames = clean_image_seq_len // self.frame_seqlen
                    
                    # Noisy half contains image + action + state tokens
                    noisy_image_seq_len = half_seq_len
                    noisy_frames = noisy_image_seq_len // self.frame_seqlen
                    num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
                    action_horizon = num_image_blocks * self.num_action_per_block
                    state_horizon = num_image_blocks * self.num_state_per_block

                    # One Joint layout object now owns both the legacy
                    # teacher-forcing ranges and the post-attention MoE groups.
                    # The optimized blockwise kernels below are unchanged.
                    joint_layout = build_joint_token_group_spec(
                        stage="train_tf",
                        n_video=noisy_image_seq_len,
                        n_clean_video=clean_image_seq_len,
                        frame_seqlen=self.frame_seqlen,
                        n_action=action_horizon,
                        n_state=state_horizon,
                        num_frame_per_block=self.num_frame_per_block,
                        num_action_per_block=self.num_action_per_block,
                        num_state_per_block=self.num_state_per_block,
                    )
                    
                    # Block layout must match actual register length. For 5B use 320x176 so latent frame_seqlen=55.
                    if roped_query.shape[1] != half_seq_len + noisy_image_seq_len + action_horizon + state_horizon:
                        raise ValueError(
                            "Sequence length does not match block layout. "
                            "For 5B use 320x176 (e.g. data=dreamzero/droid_relative_wan22 or image_resolution_width=320, image_resolution_height=176). "
                            f"Got noisy_frames={noisy_frames}, num_image_blocks={num_image_blocks}, "
                            f"action_register_length={action_register_length}. "
                            "Ensure (noisy_frames - 1) // num_frame_per_block >= 1 and register length equals "
                            "num_blocks * (num_action_per_block + num_state_per_block)."
                        )
                    
                    # Split clean and noisy parts
                    # Clean: [image tokens only]
                    clean_start, clean_end = joint_layout["clean_range"]
                    clean_image_q = roped_query[:, clean_start:clean_end]
                    clean_image_k = roped_key[:, clean_start:clean_end]
                    clean_image_v = v[:, clean_start:clean_end]

                    # Noisy: [image tokens][action tokens][state tokens]
                    noisy_start = joint_layout["fnoisy_range"][0]
                    noisy_end = noisy_start + noisy_image_seq_len
                    action_start = noisy_end
                    action_end = action_start + action_horizon
                    state_start = action_end
                    state_end = state_start + state_horizon
                    noisy_image_q = roped_query[:, noisy_start:noisy_end]
                    noisy_action_q = roped_query[:, action_start:action_end]
                    noisy_state_q = roped_query[:, state_start:state_end]

                    noisy_image_k = roped_key[:, noisy_start:noisy_end]
                    noisy_action_k = roped_key[:, action_start:action_end]
                    noisy_state_k = roped_key[:, state_start:state_end]

                    noisy_image_v = v[:, noisy_start:noisy_end]
                    noisy_action_v = v[:, action_start:action_end]
                    noisy_state_v = v[:, state_start:state_end]
                    
                    # ========== Process CLEAN (context) image tokens ==========
                    # Clean images: simple blockwise causal attention (no action/state)
                    clean_image_outputs = self._process_clean_image_only(
                        clean_image_q, clean_image_k, clean_image_v, clean_frames)
                    
                    # ========== Process NOISY tokens ==========
                    # Noisy image blocks: attend to previous clean image blocks + current noisy image + current noisy action + current noisy state
                    noisy_image_outputs = self._process_noisy_image_blocks(
                        noisy_image_q, noisy_image_k, noisy_image_v,
                        clean_image_k, clean_image_v,
                        noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                        noisy_frames, action_horizon, state_horizon)
                    
                    # Noisy action blocks: attend to previous clean image blocks (including first) + current noisy image + current noisy action + same state
                    noisy_action_outputs = self._process_noisy_action_blocks(
                        noisy_action_q, noisy_action_k, noisy_action_v,
                        clean_image_k, clean_image_v, 
                        noisy_image_k, noisy_image_v,
                        noisy_state_k, noisy_state_v,
                        noisy_frames, action_horizon, state_horizon)
                    
                    # Noisy state blocks: self-attention only
                    noisy_state_outputs = self._process_state_blocks(
                        noisy_state_q, noisy_state_k, noisy_state_v, state_horizon)
                    
                    # Concatenate all outputs in order: clean_img, noisy_img, noisy_act, noisy_state
                    x = torch.cat([
                        clean_image_outputs,
                        noisy_image_outputs, noisy_action_outputs, noisy_state_outputs
                    ], dim=1)
                else:
                    # No action/state tokens, fall back to simple image-only teacher forcing
                    half_frames = half_seq_len // self.frame_seqlen
                    clean_q = roped_query[:, :half_seq_len]
                    clean_k = roped_key[:, :half_seq_len]
                    clean_v = v[:, :half_seq_len]
                    noisy_q = roped_query[:, half_seq_len:]
                    noisy_k = roped_key[:, half_seq_len:]
                    noisy_v = v[:, half_seq_len:]
                    
                    # Process clean frames with blockwise causal attention
                    x_clean = self._blockwise_causal_flash_attn(
                        clean_q, clean_k, clean_v, self.frame_seqlen, self.num_frame_per_block,
                        action_horizon=None, state_horizon=None,
                        num_action_per_block=None, num_state_per_block=None,
                        visualize_mask=False)
                    
                    # Process noisy frames: attend to all clean frames + themselves
                    full_k = torch.cat([clean_k, noisy_k], dim=1)
                    full_v = torch.cat([clean_v, noisy_v], dim=1)
                    x_noisy = self.attn(noisy_q, full_k, full_v)
                    
                    x = torch.cat([x_clean, x_noisy], dim=1)

            else:
                roped_query = rope_action_apply(
                    x=q,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                roped_key = rope_action_apply(
                    x=k,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                # Calculate dynamic action and state horizons
                if action_register_length is not None:
                    chunk_size = action_register_length // (self.num_action_per_block + self.num_state_per_block)
                    action_horizon = chunk_size * self.num_action_per_block
                    state_horizon = chunk_size * self.num_state_per_block
                else:
                    action_horizon = None
                    state_horizon = None

                # Use blockwise causal flash attention without massive padding
                visualize = False
                x = self._blockwise_causal_flash_attn(
                    roped_query, roped_key, v, self.frame_seqlen, self.num_frame_per_block,
                    action_horizon=action_horizon,
                    state_horizon=state_horizon,
                    num_action_per_block=self.num_action_per_block if action_register_length else None,
                    num_state_per_block=self.num_state_per_block if action_register_length else None,
                    visualize_mask=visualize)

        else:
            action_state_index = (current_start_frame - 1) // self.num_frame_per_block

            roped_query = causal_rope_action_apply(
                x=q,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply(
                x=k,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)

            # split roped_query and roped_action_query (the last action_register_length tokens)
            roped_action_query: torch.Tensor | None = None
            roped_action_key: torch.Tensor | None = None
            action_v: torch.Tensor | None = None

            if action_register_length is not None:
                roped_action_query = roped_query[:, -action_register_length:]
                roped_query = roped_query[:, :-action_register_length]
                roped_action_key = roped_key[:, -action_register_length:]
                roped_key = roped_key[:, :-action_register_length]
                action_v = v[:, -action_register_length:]
                v = v[:, :-action_register_length]
                assert roped_action_query is not None
                assert roped_action_key is not None
                assert action_v is not None

            num_new_tokens = roped_query.shape[1]
            assert roped_key.shape[1] == num_new_tokens
            assert v.shape[1] == num_new_tokens

            # If we are using local attention and the current KV cache size is larger
            # than the local attention size, we need to truncate the KV cache

            updated_kv_cache = kv_cache
            updated_k = updated_kv_cache[0]
            updated_v = updated_kv_cache[1]
            # Assign new keys/values directly up to current_end
            new_k = torch.cat([updated_k, roped_key], dim=1)
            new_v = torch.cat([updated_v, v], dim=1)

            # We may need to truncate the KV cache if it's size is larger than the max attention size.
            new_k = new_k[:, -self.max_attention_size:]
            new_v = new_v[:, -self.max_attention_size:]

            if action_register_length is not None:
                x = self.attn(
                    torch.cat([roped_query, roped_action_query], dim=1),
                    torch.cat([new_k, roped_action_key], dim=1),
                    torch.cat([new_v, action_v], dim=1),
                )
            else:
                x = self.attn(
                    roped_query,
                    new_k,
                    new_v,
                )
            updated_kv_cache = torch.stack([new_k, new_v], dim=0)


        # output
        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        task_attn_mask: torch.Tensor | None = None,
        task_seg: tuple | None = None,
        task_rope_mode: str = "masked",
        task_attn_spec: dict | None = None,
        idm_cache_ctx: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Align modulation sequence length to x so mul/add broadcast (e.g. when F != L under compile)
        L = x.shape[1]
        aligned = []
        for part in e:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e = tuple(aligned)

        # self-attention
        if idm_cache_ctx is not None:
            # idm AR: cached self-attention over clean history + current chunk.
            y = self.self_attn.forward_idm_cached(
                x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
                pos_freqs=idm_cache_ctx["pos_freqs"],
                append_clean=idm_cache_ctx["append_clean"],
                window_tokens=idm_cache_ctx["window_tokens"],
                rolling_window_tokens=idm_cache_ctx.get(
                    "rolling_window_tokens"
                ),
                cache_chunk_id=idm_cache_ctx["cache_chunk_id"],
                is_pred=idm_cache_ctx["is_pred"],
                max_cache_chunks=idm_cache_ctx["max_cache_chunks"],
                current_attn_mask=idm_cache_ctx.get("current_attn_mask"),
                video_query_count=idm_cache_ctx.get("video_query_count", 0),
                exclude_history_tail_for_video=idm_cache_ctx.get(
                    "exclude_history_tail_for_video", 0
                ),
                cache_kind=idm_cache_ctx.get("cache_kind"),
                visible_cache_kinds=idm_cache_ctx.get("visible_cache_kinds"),
                cache_kind_order=idm_cache_ctx.get("cache_kind_order"),
                append_token_range=idm_cache_ctx.get("append_token_range"),
            )
            updated_kv_cache = None
        else:
            y, updated_kv_cache = self.self_attn(
                x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                kv_cache=kv_cache,
                is_tf=is_tf,
                current_start_frame=current_start_frame,
                task_attn_mask=task_attn_mask,
                task_seg=task_seg,
                task_rope_mode=task_rope_mode,
                task_attn_spec=task_attn_spec,
            )
        x = x + (y * e[2].squeeze(2))

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(self.norm3(x), context)
            y = self.ffn(
                (self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            )
            x = x + (y * e[5].squeeze(2))
            return x

        x = cross_attn_ffn(x, context, e)
        return x, updated_kv_cache


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        # Align modulation sequence length to x (e.g. when F != L1 under compile)
        L = x.shape[1]
        aligned = []
        for part in e:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e = tuple(aligned)
        x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 frame_seqlen=220,
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 max_chunk_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 num_frame_per_block=1, 
                 action_dim=32,
                 num_registers=8,
                 max_state_dim=64,
                 max_num_embodiments=32,
                 hidden_size=1024,
                 diffusion_model_pretrained_path=None,
                 num_action_per_block=32,
                 num_state_per_block=1,
                 concat_first_frame_latent=True):
        r"""
        Initialize the diffusion model backbone.

        Args:
            concat_first_frame_latent (`bool`, *optional*, defaults to True):
                If True, concat [x; y] before patch_embedding (14B I2V style). If False, latent only (5B pretrained style; first-frame via CLIP).
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path
        self.action_dim = action_dim
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim
        self.max_num_embodiments = max_num_embodiments
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.concat_first_frame_latent = concat_first_frame_latent

        max_num_embodiments = 1

        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, frame_seqlen,
                                    self.local_attn_size, sink_size, num_frame_per_block, qk_norm, cross_attn_norm, eps,
                                    num_action_per_block, num_state_per_block)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        
        self.freqs_action = rope_params(1024*10, d)
        self.freqs_state = rope_params(1024, d)
        self.freqs = [
            # AC-WM keeps absolute latent time across a whole generated
            # episode. 4096 covers the longest current AgileX episode while
            # preserving the exact prefix used by existing checkpoints.
            rope_params(4096, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]
        if model_type in ('i2v', 'ti2v'):
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        # The action head sets this from the root `gradient_checkpointing`
        # configuration. Keep construction side-effect free so the saved config,
        # Trainer and actual block execution cannot silently disagree.
        self.gradient_checkpointing = False
        # Fraction of Transformer blocks recomputed during backward. 1.0 keeps
        # the historical all-block checkpointing behavior; values in (0, 1)
        # checkpoint an evenly distributed subset of blocks.
        self.gradient_checkpointing_ratio = 1.0
        self.independent_first_frame = False if self.num_frame_per_block == 1 else True


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _should_checkpoint_block(self, block_index: int) -> bool:
        if not self.gradient_checkpointing:
            return False
        num_blocks = len(self.blocks)
        if num_blocks <= 0:
            return False
        ratio = float(getattr(self, "gradient_checkpointing_ratio", 1.0))
        checkpoint_count = min(num_blocks, max(0, int(round(num_blocks * ratio))))
        if checkpoint_count == 0:
            return False
        if checkpoint_count == num_blocks:
            return True
        # Bresenham-style even distribution with exactly checkpoint_count
        # selected blocks. This avoids concentrating stored activations at one
        # end of the network.
        return (
            ((block_index + 1) * checkpoint_count) // num_blocks
            > (block_index * checkpoint_count) // num_blocks
        )

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1, action_horizon=1, state_horizon=1, num_action_per_block=30, num_state_per_block=1
    ) -> BlockMask:
        """
        We will divide the token sequence into the following format:
        [first image (conditioning)] [image blocks] [action blocks] [state blocks]
        
        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image block + current action block + current state block
        - Action blocks: can attend to previous image block + current image block + current state block
        - State blocks: conditioning only, cannot attend to anything
        
        Block alignment:
        - num_image_blocks = (num_frames - 1) // num_frame_per_block
        - num_action_blocks = action_horizon // num_action_per_block  
        - num_state_blocks = state_horizon // num_state_per_block
        - num_image_blocks = num_action_blocks + 1 = num_state_blocks + 1
        """
        # Calculate block structure
        num_image_blocks = (num_frames - 1) // num_frame_per_block
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block
        
        # Verify the relationship: num_image_blocks = num_action_blocks + 1 = num_state_blocks + 1
        assert num_image_blocks == num_action_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_action_blocks}"
        assert num_image_blocks == num_state_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_state_blocks}"
        
        # Token ranges
        first_image_len = frame_seqlen  # First image (conditioning)
        image_blocks_len = num_image_blocks * num_frame_per_block * frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        total_length = first_image_len + image_blocks_len + action_len + state_len
        
        # print("total_length", total_length, first_image_len, image_blocks_len, action_len, state_len)
        # Padding to multiple of 128
        # padded_length = math.ceil(total_length / 128) * 128 - total_length
        padded_length = math.ceil((local_attn_size * frame_seqlen + (local_attn_size - 1) + 32 * (local_attn_size - 1))/128) * 128 - total_length
        total_padded_length = total_length + padded_length
        # print("total_padded_length", total_padded_length, total_length, padded_length)
        
        # Define token ranges for each modality
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Precompute block indices for each token
        block_indices = torch.zeros(total_padded_length, device=device, dtype=torch.long)
        
        # First image gets special block index -1 (conditioning, cannot attend to anything)
        block_indices[first_image_start:first_image_end] = -1
        
        # Assign block indices for image blocks (0 to num_image_blocks-1)
        for block_idx in range(num_image_blocks):
            start_idx = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            end_idx = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            block_indices[start_idx:end_idx] = block_idx
        
        # Assign block indices for action tokens (0 to num_action_blocks-1)
        for block_idx in range(num_action_blocks):
            start_idx = action_start + block_idx * num_action_per_block
            end_idx = action_start + (block_idx + 1) * num_action_per_block
            block_indices[start_idx:end_idx] = block_idx
        
        # Assign block indices for state tokens (0 to num_state_blocks-1)
        for block_idx in range(num_state_blocks):
            start_idx = state_start + block_idx * num_state_per_block
            end_idx = state_start + (block_idx + 1) * num_state_per_block
            block_indices[start_idx:end_idx] = block_idx
        
        # Padding tokens get block index of last block + 1 (won't attend to anything)
        block_indices[total_length:] = num_image_blocks
        
        def attention_mask(b, h, q_idx, kv_idx):
            # Self-attention
            self_attn = (q_idx == kv_idx)
            
            # Determine which modality q and kv belong to
            q_is_first_image = (q_idx >= first_image_start) & (q_idx < first_image_end)
            q_is_image_block = (q_idx >= image_blocks_start) & (q_idx < image_blocks_end)
            q_is_action = (q_idx >= action_start) & (q_idx < action_end)
            q_is_state = (q_idx >= state_start) & (q_idx < state_end)
            
            kv_is_first_image = (kv_idx >= first_image_start) & (kv_idx < first_image_end)
            kv_is_image_block = (kv_idx >= image_blocks_start) & (kv_idx < image_blocks_end)
            kv_is_action = (kv_idx >= action_start) & (kv_idx < action_end)
            kv_is_state = (kv_idx >= state_start) & (kv_idx < state_end)
            
            q_block = block_indices[q_idx]
            kv_block = block_indices[kv_idx]
            
            # First image query (conditioning) - cannot attend to anything
            first_image_mask = q_is_first_image & False
            
            # Image block query
            image_to_first = q_is_image_block & kv_is_first_image  # Image block to first image: always allowed
            image_to_image = q_is_image_block & kv_is_image_block & (kv_block <= q_block)  # Image block to image block: can attend to current and previous image blocks
            image_to_action = q_is_image_block & kv_is_action & (kv_block == q_block)  # Image block to action: can attend to current action block
            image_to_state = q_is_image_block & kv_is_state & (kv_block == q_block)  # Image block to state: can attend to current state block
            
            image_block_mask = image_to_first | image_to_image | image_to_action | image_to_state
            
            # Action query
            action_to_image = q_is_action & kv_is_image_block & (kv_block <= q_block)  # Action to image block: can attend to current and all previous image blocks
            action_to_action = q_is_action & kv_is_action & (kv_block == q_block)  # Action to action: only same block
            action_to_state = q_is_action & kv_is_state & (kv_block == q_block)  # Action to state: only same block
            action_to_first = q_is_action & kv_is_first_image  # Action to first image: always allowed
            
            action_mask = action_to_image | action_to_action | action_to_state | action_to_first
            
            # State query (conditioning) - cannot attend to anything
            state_mask = q_is_state & False
            
            # Combine all masks
            return self_attn | first_image_mask | image_block_mask | action_mask | state_mask
        
        block_mask = create_block_mask(
            attention_mask, B=None, H=None, 
            Q_LEN=total_padded_length,
            KV_LEN=total_padded_length, 
            _compile=False, device=device
        )
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Created blockwise causal attention mask:")
            print(f"  first_image_tokens={first_image_len} (conditioning)")
            print(f"  num_image_blocks={num_image_blocks} (blocks of {num_frame_per_block * frame_seqlen})")
            print(f"  num_action_blocks={num_action_blocks} (blocks of {num_action_per_block})")
            print(f"  num_state_blocks={num_state_blocks} (blocks of {num_state_per_block})")
            print(f"  total_length={total_length}, padded_length={padded_length}")
            print(block_mask)

            # Debug: materialize a small slice of the mask into 0/1 strings
            try:
                dense_mask = create_mask(
                    attention_mask,
                    B=None,
                    H=None,
                    Q_LEN=total_padded_length,
                    KV_LEN=total_padded_length,
                    device=device,
                )[0, 0]  # [Q, K]
                preview_q = min(979, dense_mask.shape[0])
                preview_k = min(979, dense_mask.shape[1])
                print("Block mask (preview):")
                for qi in range(preview_q):
                    row = dense_mask[qi, :preview_k].to(torch.int8).tolist()
                    print(" ".join(str(int(v)) for v in row))
            except Exception as err:
                print("[warn] Failed to materialize block mask preview:", err)
        
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(self.local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Forward pass through the diffusion model blocks.
        """
        x = x.flatten(start_dim=2).transpose(1, 2)

        B = x.shape[0]
        F = timestep.shape[1]

        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            state_features = None
            action_length = 0
            action_register_length = None

        # time embeddings: expand to exactly seq_len so e matches x (5B: frame_seqlen=50, 1 frame -> 50 tokens)
        if F <= seq_len:
            repeat = (seq_len + F - 1) // F
            timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(0, F - 1, seq_len, device=timestep.device, dtype=torch.long)
            timestep = timestep[:, indices]

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        context = self.text_embedding(context)
        
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # Joint cached inference has two FFN layouts.  Clean seed/history writes
        # contain video tokens only and therefore execute shared exclusively;
        # the denoise call contains one current V/A/S chunk.  Historical clean
        # context is already represented in the post-attention current tokens.
        token_group_spec = None
        if is_joint_dreamzero_task(getattr(self, "_current_wam_task", None)):
            if action is None:
                token_group_spec = build_joint_token_group_spec(
                    stage="cache_write",
                    n_video=seq_len,
                    frame_seqlen=self.frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                )
            else:
                token_group_spec = build_joint_token_group_spec(
                    stage="denoise",
                    n_video=seq_len,
                    frame_seqlen=self.frame_seqlen,
                    n_action=action_length,
                    n_state=state_features.shape[1],
                    num_frame_per_block=self.num_frame_per_block,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                )
        self._current_wam_token_group_routing = token_group_spec

        updated_kv_caches: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv_cache)

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches


    def _forward_inference_trt(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:


        frame_seqlen = 880
        seq_len = 2*frame_seqlen 
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])
        
        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
        ) 

        return x_video, action_noise_pred

    def _forward_inference_trt_droid(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:


        frame_seqlen = 880
        seq_len = 2*frame_seqlen 
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])
        
        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
        ) 

        return x_video, action_noise_pred


    def _forward_inference(
        self,
        x,
        timestep,
        context,
        seq_len,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
        embodiment_id=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            timestep (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            action (Tensor, *optional*):
                Action tensor of shape [B, H, D]
            state (Tensor, *optional*):
                State tensor of shape [B, H, D]
            embodiment_id (Tensor, *optional*):
                Embodiment ID tensor of shape [B]
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            clip_feature (Tensor, *optional*):
                CLIP image features for image-to-video mode
            timestep_action (Tensor, *optional*):
                Action timestep tensor of shape [B]
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """      
        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len

        # Concat [x; y] only when pretrained that way (14B). 5B uses latent only, first-frame via CLIP.
        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # embeddings
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=current_start_frame,
        )

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

        # Copy the updated KV caches back to the original KV cache.
        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        #for block_index, updated_kv_cache in enumerate(updated_kv_caches):
        #    kv_cache[block_index] = updated_kv_cache.clone()

        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred, updated_kv_caches

    def _forward_train_idm(
        self,
        noisy_x,
        clean_x,
        timestep,
        clean_timestep,
        timestep_action,
        clean_timestep_action,
        context,
        seq_len,
        y=None,
        clip_feature=None,
        noisy_action=None,
        clean_action=None,
        state=None,
        embodiment_id=None,
    ):
        """idm_lingbot_va interleaved teacher-forcing training forward.

        Sequence: [noisy_V | clean_V | noisy_A | clean_A | S]. Noisy copies are the
        denoise targets; clean copies + state are teacher-forcing conditioning
        (clean copies at t=0). A single interleaved teacher-forcing mask governs
        connectivity (see wam_task_attn.build_idm_teacher_forcing_mask). Returns
        (video_noise_pred, action_noise_pred) for the NOISY copies only.
        """
        embodiment_id = torch.tensor([0], device=noisy_x.device).repeat(noisy_x.shape[0])

        # Patchify both video copies.
        nx = self.patch_embedding(noisy_x)
        grid_size = torch.tensor(nx.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(grid_size=grid_size, start_frame=0)
        nx = nx.flatten(start_dim=2).transpose(1, 2)
        cx = self.patch_embedding(clean_x).flatten(start_dim=2).transpose(1, 2)
        assert nx.shape[1] == seq_len and cx.shape[1] == seq_len
        B = nx.shape[0]
        F = timestep.shape[1]
        num_frames = grid_size[0].item()

        # Encode both action copies + state.
        noisy_a = self.action_encoder(noisy_action, timestep_action, embodiment_id)
        clean_a = self.action_encoder(clean_action, clean_timestep_action, embodiment_id)
        n_action = noisy_a.shape[1]
        state_features = self.state_encoder(state, embodiment_id)
        n_state = state_features.shape[1]

        x = torch.cat([nx, cx, noisy_a, clean_a, state_features], dim=1)

        # Interleaved teacher-forcing mask + segment lengths.
        _, _, task_mask, seg_lengths = build_idm_teacher_forcing_mask(
            num_frames=num_frames, frame_seqlen=self.frame_seqlen,
            num_frame_per_block=self.num_frame_per_block,
            n_action=n_action, num_action_per_block=self.num_action_per_block,
            n_state=n_state, device=x.device,
        )
        nV, _, nA, _, nS = seg_lengths
        task_seg = (nV, nA, nS)

        # One spec owns the FFN token responsibilities for the same physical
        # sequence governed by the teacher-forcing mask above.  The four chunks
        # remain in one parallel forward; only expert execution becomes local.
        token_group_spec = build_idm_token_group_spec(
            stage="train_tf",
            n_video=nV,
            frame_seqlen=self.frame_seqlen,
            n_action=nA,
            n_state=nS,
            num_frame_per_block=self.num_frame_per_block,
            num_action_per_block=self.num_action_per_block,
        )

        # Time embeddings, in sequence order [noisy_V | clean_V | noisy_A | clean_A | S].
        def _video_ts(ts):
            return ts.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
        ts_noisy_v = _video_ts(timestep)
        ts_clean_v = _video_ts(clean_timestep)
        stride = timestep_action.shape[1] // n_state
        ts_state = clean_timestep_action[:, ::stride]
        timestep_full = torch.cat([ts_noisy_v, ts_clean_v, timestep_action, clean_timestep_action, ts_state], dim=1)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_full.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # Text (+ optional CLIP first-frame) via cross-attention.
        assert context.shape[1] == self.text_len
        context = self.text_embedding(context)
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        kwargs = dict(
            e=e0, freqs=freqs, freqs_action=self.freqs_action, freqs_state=self.freqs_state,
            action_register_length=2 * n_action + n_state, context=context, is_tf=False,
            task_attn_mask=task_mask, task_seg=task_seg, task_rope_mode="idm",
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                outputs, updated_kv_cache = module(*inputs, **kwargs)
                assert updated_kv_cache is None
                return outputs
            return custom_forward

        # Keep the metadata alive through backward because checkpointed FFNs
        # read it again during recomputation.
        self._current_wam_token_group_routing = token_group_spec
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self._should_checkpoint_block(block_index):
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x, **kwargs, use_reentrant=False,
                )
            else:
                x, _ = block(x, **kwargs)

        # Split outputs: noisy_V is the first nV tokens; noisy_A starts at 2*nV.
        e_video = e[:, :nV]
        x_video = self.head(x[:, :nV], e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)
        action_noise_pred = self.action_decoder(x[:, 2 * nV: 2 * nV + nA], embodiment_id)
        return video_noise_pred, action_noise_pred

    def reset_idm_cache(self):
        """Clear the per-layer idm AR clean-history K/V pools."""
        for block in self.blocks:
            block.self_attn.reset_idm_cache()

    def reset_acwm_cache(self):
        """Clear the per-layer ac_wm rolling clean-history K/V pools."""
        self.reset_idm_cache()

    def clear_idm_pred_cache(self):
        """Remove only provisional prediction chunks from every layer."""
        for block in self.blocks:
            block.self_attn.clear_idm_pred_cache()

    def _idm_cached_blocks(self, tokens, pos_freqs, timestep_full, context, clip_feature,
                           append_clean, stage, window_tokens=None, cache_chunk_id=None,
                           is_pred=False, max_cache_chunks=None,
                           rolling_window_tokens=None,
                           current_attn_mask=None, routing_spec=None,
                           video_query_count=0,
                           exclude_history_tail_for_video=0,
                           cache_kind=None, visible_cache_kinds=None,
                           cache_kind_order=None, append_token_range=None):
        """Run all blocks over ONE idm chunk with the cached AR self-attention.

        tokens: [B, L_cur, dim] embedded chunk tokens (video OR action).
        pos_freqs: [L_cur, 1, d/2] complex RoPE at the chunk's absolute positions.
        timestep_full: [B, L_cur] per-token timestep for modulation.
        Returns block output [B, L_cur, dim].
        """
        B = tokens.shape[0]
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_full.flatten()).type_as(tokens))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))
        ctx = self.text_embedding(context)
        if clip_feature is not None:
            ctx = torch.cat([self.img_emb(clip_feature), ctx], dim=1)
        idm_ctx = dict(
            pos_freqs=pos_freqs, append_clean=append_clean, window_tokens=window_tokens,
            rolling_window_tokens=rolling_window_tokens,
            cache_chunk_id=cache_chunk_id, is_pred=is_pred,
            max_cache_chunks=max_cache_chunks,
            current_attn_mask=current_attn_mask,
            video_query_count=video_query_count,
            exclude_history_tail_for_video=exclude_history_tail_for_video,
            cache_kind=cache_kind,
            visible_cache_kinds=visible_cache_kinds,
            cache_kind_order=cache_kind_order,
            append_token_range=append_token_range,
        )
        self._current_wam_token_group_routing = (
            routing_spec
            if routing_spec is not None
            else build_idm_token_group_spec(stage=stage, seq_len=tokens.shape[1])
        )
        x = tokens
        for block in self.blocks:
            x, _ = block(
                x=x, e=e0, freqs=None, freqs_action=None, freqs_state=None,
                action_register_length=None, context=ctx, idm_cache_ctx=idm_ctx,
            )
        return x

    def idm_cache_write_state(
        self, state, embodiment_id, context, clip_feature=None, window_tokens=None, abs_state_start=0,
        cache_chunk_id=None, is_pred=False, max_cache_chunks=None,
    ):
        """Run a clean state through every block and commit its per-layer K/V."""
        state_features = self.state_encoder(state, embodiment_id)  # [B, nS, dim]
        nS = state_features.shape[1]
        pos = self.freqs_state[abs_state_start:abs_state_start + nS].view(nS, 1, -1)
        timestep_full = torch.zeros(
            state_features.shape[0], nS, device=state_features.device
        )
        return self._idm_cached_blocks(
            state_features, pos, timestep_full, context, clip_feature,
            append_clean=True, stage="cache_write_state", window_tokens=window_tokens,
            cache_chunk_id=cache_chunk_id, is_pred=is_pred,
            max_cache_chunks=max_cache_chunks,
        )

    def idm_cached_video_chunk(self, noisy_chunk, abs_frame_start, timestep_val, context,
                               clip_feature, append_clean, window_tokens=None,
                               cache_chunk_id=None, is_pred=False, max_cache_chunks=None):
        """Denoise-step forward for a VIDEO chunk under the idm cache.

        noisy_chunk: [B, C_in, Fc, Hc, Wc]. Returns video velocity pred
        [B, C_out, Fc, Hc, Wc]. Set append_clean=True to write this chunk (clean)
        into the history pool.
        """
        xemb = self.patch_embedding(noisy_chunk)
        grid_size = torch.tensor(xemb.shape[2:], dtype=torch.long)
        pos_freqs = self._create_freqs(grid_size=grid_size, start_frame=abs_frame_start)
        tokens = xemb.flatten(start_dim=2).transpose(1, 2)  # [B, Lc, dim]
        B, Lc = tokens.shape[0], tokens.shape[1]
        timestep_full = torch.full((B, Lc), float(timestep_val), device=tokens.device)
        stage = (
            "cache_write_seed"
            if append_clean and abs_frame_start == 0
            else "cache_write_video"
            if append_clean
            else "video_denoise"
        )
        out = self._idm_cached_blocks(tokens, pos_freqs, timestep_full, context, clip_feature,
                                      append_clean=append_clean, stage=stage, window_tokens=window_tokens,
                                      cache_chunk_id=cache_chunk_id, is_pred=is_pred,
                                      max_cache_chunks=max_cache_chunks)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_full.flatten()).type_as(tokens))
        e = e.unflatten(dim=0, sizes=(B, -1))
        x_video = self.head(out, e.unsqueeze(2))
        return self.unpatchify(x_video, grid_size)

    def idm_cached_action_chunk(self, noisy_action, abs_action_start, timestep_val, context,
                                clip_feature, embodiment_id, append_clean, window_tokens=None,
                                cache_chunk_id=None, is_pred=False, max_cache_chunks=None):
        """Denoise-step forward for an ACTION chunk under the idm cache.

        noisy_action: [B, Ac, action_dim]. Returns action velocity [B, Ac, action_dim].
        """
        B, Ac = noisy_action.shape[0], noisy_action.shape[1]
        ts_action = torch.full((B, Ac), float(timestep_val), device=noisy_action.device)
        tokens = self.action_encoder(noisy_action, ts_action, embodiment_id)  # [B, Ac, dim]
        pos_freqs = self.freqs_action[abs_action_start: abs_action_start + Ac].view(Ac, 1, -1)
        stage = "cache_write_action" if append_clean else "action_denoise"
        out = self._idm_cached_blocks(tokens, pos_freqs, ts_action, context, clip_feature,
                                      append_clean=append_clean, stage=stage, window_tokens=window_tokens,
                                      cache_chunk_id=cache_chunk_id, is_pred=is_pred,
                                      max_cache_chunks=max_cache_chunks)
        return self.action_decoder(out, embodiment_id)

    def idm_cached_video_action_chunk(
        self,
        noisy_video,
        noisy_action,
        abs_frame_start,
        abs_action_start,
        timestep_val,
        context,
        clip_feature,
        embodiment_id,
        window_tokens=None,
        cache_chunk_id=None,
        is_pred=False,
        max_cache_chunks=None,
        action_phase=False,
    ):
        """Denoise one IDM video/action target pair in a single current query.

        Video and action tokens share the sample-level route but retain IDM's
        attention isolation: neither noisy target reads the other.  Action
        obtains the finalized current video only from the clean history cache
        during the second phase.
        """
        xemb = self.patch_embedding(noisy_video)
        grid_size = torch.tensor(xemb.shape[2:], dtype=torch.long)
        video_pos = self._create_freqs(
            grid_size=grid_size, start_frame=abs_frame_start
        )
        video_tokens = xemb.flatten(start_dim=2).transpose(1, 2)
        B, n_video = video_tokens.shape[:2]
        n_action = noisy_action.shape[1]
        ts_action = torch.full(
            (B, n_action), float(timestep_val), device=noisy_action.device
        )
        action_tokens = self.action_encoder(
            noisy_action, ts_action, embodiment_id
        )
        action_pos = self.freqs_action[
            abs_action_start: abs_action_start + n_action
        ].view(n_action, 1, -1)

        tokens = torch.cat([video_tokens, action_tokens], dim=1)
        pos_freqs = torch.cat([video_pos, action_pos], dim=0)
        ts_video = torch.full(
            (B, n_video), float(timestep_val), device=video_tokens.device
        )
        timestep_full = torch.cat([ts_video, ts_action], dim=1)
        current_mask = torch.zeros(
            n_video + n_action,
            n_video + n_action,
            dtype=torch.bool,
            device=tokens.device,
        )
        current_mask[:n_video, :n_video] = True
        current_mask[n_video:, n_video:] = True
        routing_spec = {
            "task": "idm_lingbot_va",
            "stage": "video_action_denoise",
            "seq_len": n_video + n_action,
            "shared_ranges": (),
            "video_targets": ((0, n_video),),
            "action_targets": ((n_video, n_video + n_action),),
            "chunks": ({
                "index": 0,
                "video_range": (0, n_video),
                "action_range": (n_video, n_video + n_action),
            },),
        }
        out = self._idm_cached_blocks(
            tokens,
            pos_freqs,
            timestep_full,
            context,
            clip_feature,
            append_clean=False,
            stage="video_action_denoise",
            window_tokens=window_tokens,
            cache_chunk_id=cache_chunk_id,
            is_pred=is_pred,
            max_cache_chunks=max_cache_chunks,
            current_attn_mask=current_mask,
            routing_spec=routing_spec,
            video_query_count=n_video,
            exclude_history_tail_for_video=(
                n_video if action_phase else 0
            ),
        )

        e_video = self.time_embedding(
            sinusoidal_embedding_1d(
                self.freq_dim, ts_video.flatten()
            ).type_as(video_tokens)
        ).unflatten(0, (B, n_video))
        video_pred = self.head(
            out[:, :n_video], e_video.unsqueeze(2)
        )
        video_pred = self.unpatchify(video_pred, grid_size)
        action_pred = self.action_decoder(
            out[:, n_video:], embodiment_id
        )
        return video_pred, action_pred

    def masked_cache_write_clean_video(
        self,
        clean_chunk,
        *,
        task,
        abs_frame_start,
        context,
        clip_feature,
        cache_chunk_id,
        rolling_window_tokens=None,
    ):
        """Commit one clean video prefix chunk without exposing the noisy seed."""
        xemb = self.patch_embedding(clean_chunk)
        grid_size = torch.tensor(xemb.shape[2:], dtype=torch.long)
        pos_freqs = self._create_freqs(
            grid_size=grid_size, start_frame=abs_frame_start
        )
        tokens = xemb.flatten(start_dim=2).transpose(1, 2)
        token_count = tokens.shape[1]
        timestep_full = torch.zeros(
            tokens.shape[0], token_count, device=tokens.device
        )
        routing_spec = {
            "task": task,
            "stage": "masked_cache_write_clean",
            "seq_len": token_count,
            "shared_ranges": ((0, token_count),),
        }
        current_mask = torch.ones(
            token_count,
            token_count,
            dtype=torch.bool,
            device=tokens.device,
        )
        return self._idm_cached_blocks(
            tokens,
            pos_freqs,
            timestep_full,
            context,
            clip_feature,
            append_clean=True,
            stage="masked_cache_write_clean",
            cache_chunk_id=cache_chunk_id,
            rolling_window_tokens=rolling_window_tokens,
            current_attn_mask=current_mask,
            routing_spec=routing_spec,
            cache_kind="clean",
            visible_cache_kinds=("clean",),
            cache_kind_order=("clean",),
        )

    def masked_cache_write_seed(
        self,
        seed_latent,
        *,
        task,
        context,
        clip_feature,
        state=None,
        embodiment_id=None,
        abs_frame_start=0,
        cache_chunk_id=0,
    ):
        """Commit the noisy seed K/V while using, but not retaining, S0.

        abs_frame_start is normally zero for a fresh episode.  Continuous
        AC-WM windows reuse the oldest retained latent as the local training
        seed, so its original absolute video position must be preserved.
        """
        xemb = self.patch_embedding(seed_latent)
        grid_size = torch.tensor(xemb.shape[2:], dtype=torch.long)
        video_pos = self._create_freqs(
            grid_size=grid_size, start_frame=abs_frame_start
        )
        video_tokens = xemb.flatten(start_dim=2).transpose(1, 2)
        batch_size, n_video = video_tokens.shape[:2]
        tokens = video_tokens
        positions = video_pos
        shared_ranges = []
        if state is not None:
            if embodiment_id is None:
                raise ValueError("masked seed state requires embodiment_id")
            local_id = torch.zeros_like(embodiment_id)
            state_tokens = self.state_encoder(state, local_id)
            n_state = state_tokens.shape[1]
            tokens = torch.cat([tokens, state_tokens], dim=1)
            positions = torch.cat(
                [
                    positions,
                    self.freqs_state[:n_state].view(n_state, 1, -1),
                ],
                dim=0,
            )
            shared_ranges.append((n_video, n_video + n_state))
        timestep_full = torch.zeros(
            batch_size, tokens.shape[1], device=tokens.device
        )
        seed_expert = (
            "vgm" if task == "action_only_fast_wam" else None
        )
        routing_spec = {
            "task": task,
            "stage": "masked_cache_write_seed",
            "seq_len": tokens.shape[1],
            "shared_ranges": tuple(shared_ranges),
            "seed_targets": ((0, n_video),),
            "seed_expert": seed_expert,
        }
        current_mask = torch.ones(
            tokens.shape[1],
            tokens.shape[1],
            dtype=torch.bool,
            device=tokens.device,
        )
        return self._idm_cached_blocks(
            tokens,
            positions,
            timestep_full,
            context,
            clip_feature,
            append_clean=True,
            stage="masked_cache_write_seed",
            cache_chunk_id=cache_chunk_id,
            current_attn_mask=current_mask,
            routing_spec=routing_spec,
            cache_kind="seed",
            visible_cache_kinds=("clean",),
            cache_kind_order=("clean",),
            append_token_range=(0, n_video),
        )

    def masked_cached_action_chunk(
        self,
        noisy_action,
        state,
        *,
        task,
        chunk_index,
        timestep_val,
        context,
        clip_feature,
        embodiment_id,
    ):
        """Denoise one Fast/Giga action query against typed history caches."""
        batch_size, n_action = noisy_action.shape[:2]
        local_id = torch.zeros_like(embodiment_id)
        action_timestep = torch.full(
            (batch_size, n_action),
            float(timestep_val),
            device=noisy_action.device,
        )
        action_tokens = self.action_encoder(
            noisy_action, action_timestep, local_id
        )
        state_tokens = self.state_encoder(state, local_id)
        n_state = state_tokens.shape[1]
        tokens = torch.cat([action_tokens, state_tokens], dim=1)
        action_start = int(chunk_index) * self.num_action_per_block
        action_pos = self.freqs_action[
            action_start:action_start + n_action
        ].view(n_action, 1, -1)
        state_start = int(chunk_index) * self.num_state_per_block
        state_pos = self.freqs_state[
            state_start:state_start + n_state
        ].view(n_state, 1, -1)
        positions = torch.cat([action_pos, state_pos], dim=0)
        state_timestep = torch.zeros(
            batch_size, n_state, device=noisy_action.device
        )
        timestep_full = torch.cat(
            [action_timestep, state_timestep], dim=1
        )
        current_mask = torch.zeros(
            n_action + n_state,
            n_action + n_state,
            dtype=torch.bool,
            device=tokens.device,
        )
        current_mask[:n_action, :] = True
        current_mask[n_action:, n_action:] = True
        routing_spec = {
            "task": task,
            "stage": "masked_cache_action_denoise",
            "seq_len": n_action + n_state,
            "shared_ranges": ((n_action, n_action + n_state),),
            "action_targets": ((0, n_action),),
            "action_expert": "vla",
        }
        out = self._idm_cached_blocks(
            tokens,
            positions,
            timestep_full,
            context,
            clip_feature,
            append_clean=False,
            stage="masked_cache_action_denoise",
            current_attn_mask=current_mask,
            routing_spec=routing_spec,
            visible_cache_kinds=("clean", "seed"),
            cache_kind_order=("clean", "seed"),
        )
        return self.action_decoder(out[:, :n_action], local_id)

    def acwm_cached_video_chunk(
        self,
        clean_chunk,
        abs_frame_start,
        context,
        clip_feature,
        *,
        cache_chunk_id,
        window_tokens,
    ):
        """Commit one clean ac_wm video chunk into the rolling history cache."""
        xemb = self.patch_embedding(clean_chunk)
        grid_size = torch.tensor(xemb.shape[2:], dtype=torch.long)
        pos_freqs = self._create_freqs(
            grid_size=grid_size, start_frame=abs_frame_start
        )
        tokens = xemb.flatten(start_dim=2).transpose(1, 2)
        timestep_full = torch.zeros(
            tokens.shape[0], tokens.shape[1], device=tokens.device
        )
        routing_spec = {
            "task": "world_model",
            "n_video": tokens.shape[1],
            "n_first": 0,
            "n_action": 0,
            "n_state": 0,
        }
        return self._idm_cached_blocks(
            tokens,
            pos_freqs,
            timestep_full,
            context,
            clip_feature,
            append_clean=True,
            stage="acwm_cache_write",
            rolling_window_tokens=window_tokens,
            cache_chunk_id=cache_chunk_id,
            is_pred=False,
            routing_spec=routing_spec,
        )

    def acwm_cached_video_action_chunk(
        self,
        noisy_video,
        clean_action,
        abs_frame_start,
        timestep_val,
        context,
        clip_feature,
        embodiment_id,
        *,
        action_start=0,
        unified_cache=False,
    ):
        """Denoise two ac_wm video latents under clean action and cached history."""
        xemb = self.patch_embedding(noisy_video)
        grid_size = torch.tensor(xemb.shape[2:], dtype=torch.long)
        video_pos = self._create_freqs(
            grid_size=grid_size, start_frame=abs_frame_start
        )
        video_tokens = xemb.flatten(start_dim=2).transpose(1, 2)
        B, n_video = video_tokens.shape[:2]

        n_action = clean_action.shape[1]
        action_timestep = torch.zeros(
            B, n_action, device=clean_action.device
        )
        action_tokens = self.action_encoder(
            clean_action, action_timestep, embodiment_id
        )
        action_start = int(action_start)
        action_pos = self.freqs_action[
            action_start:action_start + n_action
        ].view(n_action, 1, -1)

        tokens = torch.cat([video_tokens, action_tokens], dim=1)
        pos_freqs = torch.cat([video_pos, action_pos], dim=0)
        video_timestep = torch.full(
            (B, n_video), float(timestep_val), device=video_tokens.device
        )
        timestep_full = torch.cat(
            [video_timestep, action_timestep], dim=1
        )

        current_mask = torch.zeros(
            n_video + n_action,
            n_video + n_action,
            dtype=torch.bool,
            device=tokens.device,
        )
        current_mask[:n_video, :] = True
        current_mask[n_video:, n_video:] = True
        if unified_cache:
            routing_spec = {
                "task": "world_model",
                "stage": "masked_cache_acwm_denoise",
                "seq_len": n_video + n_action,
                "shared_ranges": ((n_video, n_video + n_action),),
                "video_targets": ((0, n_video),),
                "video_expert": "wm",
            }
        else:
            routing_spec = {
                "task": "world_model",
                "n_video": n_video,
                "n_first": 0,
                "n_action": n_action,
                "n_state": 0,
            }
        out = self._idm_cached_blocks(
            tokens,
            pos_freqs,
            timestep_full,
            context,
            clip_feature,
            append_clean=False,
            stage="acwm_denoise",
            current_attn_mask=current_mask,
            routing_spec=routing_spec,
            video_query_count=n_video,
            visible_cache_kinds=("clean", "seed") if unified_cache else None,
            cache_kind_order=("clean", "seed") if unified_cache else None,
        )

        e_video = self.time_embedding(
            sinusoidal_embedding_1d(
                self.freq_dim, video_timestep.flatten()
            ).type_as(video_tokens)
        ).unflatten(0, (B, n_video))
        video_pred = self.head(
            out[:, :n_video], e_video.unsqueeze(2)
        )
        return self.unpatchify(video_pred, grid_size)

    def _forward_train_masked(
        self,
        task,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        y=None,
        clip_feature=None,
        action=None,
        state=None,
        embodiment_id=None,
        video_start_frame=None,
        clean_x=None,
        full_window_history=False,
        full_window_chunked_attention=False,
    ):
        """WAM masked-task training forward (world_model / fastwam / gigaworld).

        Sequence: [video | action | state] (no clean/noisy teacher-forcing copy).
        First video frame is expected to be clean (t=0), set by the action head.
        world_model drops the state segment. A single per-task boolean self-attn
        mask governs connectivity; text is injected via cross-attn (unchanged).
        """
        if full_window_chunked_attention and not full_window_history:
            raise ValueError(
                "Chunk-wise full-window attention requires full_window_history=True"
            )
        # Video tokens. Legacy ac_wm uses four clean history latents followed by two
        # denoise targets; the other masked tasks keep one clean seed latent.
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        rope_start = video_start_frame if video_start_frame is not None else 0
        if task == "world_model" and self.training and not full_window_history:
            if video_start_frame is None and x.shape[0] % 4 == 0:
                relative_starts = torch.tensor(
                    [1, 3, 5, 7], device=x.device, dtype=torch.long
                ).repeat(x.shape[0] // 4)
                max_base = (
                    min(1024, int(self.freqs[0].shape[0]))
                    - int(grid_size[0])
                    - int(relative_starts.max().item())
                )
                if max_base < 0:
                    raise ValueError(
                        "video RoPE table cannot fit one AC-WM Bx4 window"
                    )
                rope_start = relative_starts + torch.randint(
                    0, max_base + 1, (1,), device=x.device
                )
            elif video_start_frame is None:
                max_start = max(
                    0,
                    min(1024, int(self.freqs[0].shape[0]))
                    - int(grid_size[0]),
                )
                if max_start:
                    rope_start = int(
                        torch.randint(
                            0, max_start + 1, (1,), device=x.device
                        ).item()
                    )
        freqs = self._create_freqs(
            grid_size=grid_size, start_frame=rope_start
        )
        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len
        if full_window_history:
            if clean_x is None:
                raise ValueError("Full-window history training requires clean_x")
            granularity = str(
                getattr(self, "_wam_moe_routing_granularity", "sample_level")
            ).strip().lower()
            if granularity not in {"sample_level", "token_group_level"}:
                raise ValueError(
                    "Full-window history requires sample_level or "
                    f"token_group_level routing, got {granularity!r}"
                )
            clean_x = self.patch_embedding(clean_x)
            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            if clean_x.shape[1] != seq_len:
                raise ValueError(
                    "Full-window clean/noisy video token lengths disagree: "
                    f"{clean_x.shape[1]} != {seq_len}"
                )
        B = x.shape[0]
        F = timestep.shape[1]
        clean_video_latents = (
            4 if task == "world_model" and F == 6 else 1
        )
        clean_video_tokens = clean_video_latents * self.frame_seqlen

        use_state = (task != "world_model") and (state is not None)

        embodiment_id = torch.tensor([0], device=x.device).repeat(B)
        assert action is not None, "WAM masked tasks require action tokens."
        action_features = self.action_encoder(action, timestep_action, embodiment_id)
        action_length = action_features.shape[1]
        if use_state:
            state_features = self.state_encoder(state, embodiment_id)
            state_length = state_features.shape[1]
            register = torch.cat([action_features, state_features], dim=1)
        else:
            state_features = None
            state_length = 0
            register = action_features
        if full_window_history:
            x = torch.cat([clean_x, x, register], dim=1)
        else:
            x = torch.cat([x, register], dim=1)

        # Per-task self-attention mask over [video | action | state].
        if full_window_history:
            full_window_kwargs = dict(
                n_video=seq_len,
                frame_seqlen=self.frame_seqlen,
                n_action=action_length,
                n_state=state_length,
                num_frame_per_block=self.num_frame_per_block,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
            )
            if full_window_chunked_attention:
                token_group_spec = build_full_window_history_spec(
                    task, **full_window_kwargs
                )
                task_mask = None
            else:
                token_group_spec, task_mask = build_full_window_history_mask(
                    task, device=x.device, **full_window_kwargs
                )
        elif task == "world_model":
            _, task_mask = build_acwm_two_history_chunk_mask(
                n_video=seq_len,
                frame_seqlen=self.frame_seqlen,
                n_action=action_length,
                device=x.device,
            )
        else:
            _, task_mask = build_task_self_attn_mask_from_lengths(
                task, n_video=seq_len, frame_seqlen=clean_video_tokens,
                n_action=action_length, n_state=state_length, device=x.device,
            )
        task_seg = (seq_len, action_length, state_length)

        # Time embeddings for [video | action | state].
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
        ts_parts = [timestep, timestep_action]
        if use_state:
            # Action-only tasks treat S0 as a fixed clean condition, exactly
            # like the clean first frame F.
            ts_parts.append(masked_state_timesteps(task, timestep_action, state_length))
        if full_window_history:
            ts_parts.insert(0, torch.zeros_like(timestep))
        timestep_full = torch.cat(ts_parts, dim=1)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_full.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # Text (+ optional CLIP first-frame) via cross-attention (unchanged).
        assert context.shape[1] == self.text_len
        context = self.text_embedding(context)
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_length + state_length,
            context=context,
            is_tf=False,
            task_attn_mask=task_mask,
            task_seg=task_seg,
            task_rope_mode=(
                (
                    "full_window_history_chunked"
                    if full_window_chunked_attention
                    else "full_window_history"
                )
                if full_window_history
                else "masked"
            ),
            task_attn_spec=(
                token_group_spec if full_window_chunked_attention else None
            ),
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                outputs, updated_kv_cache = module(*inputs, **kwargs)
                assert updated_kv_cache is None
                return outputs
            return custom_forward

        # FastWAM, world_model, and GigaWorld use token-group-level MoE. The exact
        # ranges are expressed in DiT tokens (one F latent contains
        # ``frame_seqlen`` spatial tokens). Other tasks keep sample-level routing.
        if not full_window_history:
            token_group_spec = None
        if not full_window_history and task in (
            "action_only_fast_wam",
            "world_model",
            "action_only_gigaworld_policy",
        ):
            token_group_spec = {
                "task": task,
                "n_video": seq_len,
                "n_first": clean_video_tokens,
                "n_action": action_length,
                "n_state": state_length,
            }
        # Keep this through backward because checkpoint recomputes the FFN.
        self._current_wam_token_group_routing = token_group_spec
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self._should_checkpoint_block(block_index):
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x, **kwargs, use_reentrant=False,
                )
            else:
                x, _ = block(x, **kwargs)

        # Split outputs.
        noisy_start = seq_len if full_window_history else 0
        action_start = noisy_start + seq_len
        action_noise_pred = self.action_decoder(
            x[:, action_start: action_start + action_length], embodiment_id
        )
        x_video = x[:, noisy_start:noisy_start + seq_len]
        e_video = e[:, noisy_start:noisy_start + seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)
        return video_noise_pred, action_noise_pred

    def _forward_train(
        self,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        y=None,
        clip_feature=None,
        action=None,
        state=None,
        embodiment_id=None,
        video_start_frame=None,
        wam_full_window_history=False,
        wam_full_window_chunked_attention=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # WAM masked-task dispatch: world_model / fastwam / gigaworld use a single
        # per-task masked self-attention over [video|action|state] (no clean/noisy
        # teacher-forcing duplication). joint_dreamzero keeps the path below.
        masked_task = canonical_masked_task(getattr(self, "_current_wam_task", None))
        # This assignment happens before every training/inference block path.
        # FastWAM's masked path replaces it with a concrete spec above.
        self._current_wam_token_group_routing = None
        if masked_task is not None:
            return self._forward_train_masked(
                task=masked_task, x=x, timestep=timestep, timestep_action=timestep_action,
                context=context, seq_len=seq_len, y=y, clip_feature=clip_feature,
                action=action, state=state, embodiment_id=embodiment_id,
                video_start_frame=video_start_frame,
                clean_x=clean_x,
                full_window_history=wam_full_window_history,
                full_window_chunked_attention=wam_full_window_chunked_attention,
            )

        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None

        # Concat [x; y] only when pretrained that way (14B). 5B uses latent only, first-frame via CLIP.
        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # embeddings
        x = self.patch_embedding(x)

        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=0,
        )

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len

        B = x.shape[0]
        F = timestep.shape[1]

        # time embeddings
        if action is not None:
            embodiment_id = torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            action_length = action_features.shape[1]
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            action_length = None
            state_features = None
            action_register = None
            action_register_length = None

        # time embeddings
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
        timestep_original = timestep.clone()

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        assert context.shape[1] == self.text_len
        context = self.text_embedding(context)

        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        if clean_x is not None:
            if y is not None and self.concat_first_frame_latent:
                clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
            clean_x = self.patch_embedding(clean_x)
            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] == seq_len

            x = torch.cat([clean_x, x], dim=1)

            if aug_t is None:
                aug_t = torch.zeros_like(timestep_original)
            assert aug_t is not None

            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
            e0_clean = self.time_projection(e_clean)
            e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
            e0 = torch.cat([e0_clean, e0], dim=1)

        # Joint keeps DreamZero's original equal 9-clean/9-noisy teacher-
        # forcing stream.  Only the FFN expert assignment becomes token-local.
        token_group_spec = None
        if (
            clean_x is not None
            and action is not None
            and is_joint_dreamzero_task(getattr(self, "_current_wam_task", None))
        ):
            token_group_spec = build_joint_token_group_spec(
                stage="train_tf",
                n_video=seq_len,
                n_clean_video=clean_x.shape[1],
                frame_seqlen=self.frame_seqlen,
                n_action=action_length,
                n_state=state_features.shape[1],
                num_frame_per_block=self.num_frame_per_block,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
            )
        self._current_wam_token_group_routing = token_group_spec

        # arguments
        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            context=context,
            is_tf=clean_x is not None,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                outputs, updated_kv_cache = module(*inputs, **kwargs)
                assert updated_kv_cache is None
                return outputs
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self._should_checkpoint_block(block_index):
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x, _ = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, clean_x.shape[1]:]

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_size):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Patchified features, with shape [B, L, C_out * prod(patch_size)].
            grid_size (Tensor):
                Spatial-temporal grid dimensions before patching, with shape [3]
                (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        assert x.shape[1] == math.prod(grid_size)
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int | list[int] | torch.Tensor,
    ):
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        starts = torch.as_tensor(
            start_frame, device=device, dtype=torch.long
        )
        scalar_start = starts.ndim == 0
        if starts.ndim == 0:
            starts = starts.reshape(1)
        elif starts.ndim != 1 or starts.numel() == 0:
            raise ValueError(
                "RoPE start_frame must be a scalar or non-empty 1D sequence"
            )
        if int(starts.min().item()) < 0:
            raise ValueError(
                "RoPE start_frame must be non-negative, got "
                f"{starts.detach().cpu().tolist()}"
            )
        max_end = int(starts.max().item()) + f
        if max_end > self.freqs[0].shape[0]:
            raise ValueError(
                "video RoPE table exhausted: "
                f"requested max frame {max_end}, "
                f"capacity={self.freqs[0].shape[0]}"
            )
        if h > self.freqs[1].shape[0] or w > self.freqs[2].shape[0]:
            raise ValueError(
                "spatial RoPE table exhausted: "
                f"requested HxW={h}x{w}, capacity="
                f"{self.freqs[1].shape[0]}x{self.freqs[2].shape[0]}"
            )
        batch_size = starts.numel()
        time_indices = starts[:, None] + torch.arange(f, device=device)
        freqs = torch.cat(
            [
                self.freqs[0][time_indices]
                .view(batch_size, f, 1, 1, -1)
                .expand(batch_size, f, h, w, -1),
                self.freqs[1][:h]
                .view(1, 1, h, 1, -1)
                .expand(batch_size, f, h, w, -1),
                self.freqs[2][:w]
                .view(1, 1, 1, w, -1)
                .expand(batch_size, f, h, w, -1),
            ],
            dim=-1
        ).reshape(batch_size, f * h * w, 1, -1)

        return freqs[0] if scalar_start else freqs

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
