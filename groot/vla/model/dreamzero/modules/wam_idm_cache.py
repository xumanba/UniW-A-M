"""Canonical WAM IDM timeline and episode-level chunk-ring metadata.

This module intentionally contains no model code.  It is the single authority
for absolute positions, prediction replacement, and complete-chunk FIFO
eviction, making the cache lifecycle testable without loading Wan weights.
"""

from dataclasses import dataclass
from typing import Optional


IDM_LATENTS_PER_CHUNK = 2
IDM_ACTIONS_PER_CHUNK = 24
IDM_DEFAULT_WINDOW_CHUNKS = 4


@dataclass(frozen=True)
class IDMChunk:
    chunk_id: int
    state_id: int
    frame_start: int
    action_start: int
    is_pred: bool


class IDMEpisodeCache:
    """Metadata ring for ``seed -> (state, 2 latents, 24 actions)*``.

    The seed occupies absolute frame 0.  A prediction is provisional: commit
    removes it and inserts the real chunk at the same absolute IDs.  Capacity is
    enforced in complete chunks; token-level truncation is deliberately absent.
    """

    def __init__(self, capacity: int = IDM_DEFAULT_WINDOW_CHUNKS):
        if capacity < 1:
            raise ValueError("IDM cache capacity must be positive")
        self.capacity = int(capacity)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.chunks: list[IDMChunk] = []
        self.pending: Optional[IDMChunk] = None
        self.next_chunk_id = 0

    def start(self) -> IDMChunk:
        self.reset()
        self.active = True
        seed = IDMChunk(-1, -1, 0, 0, False)
        self.chunks.append(seed)
        return seed

    def _require_active(self) -> None:
        if not self.active:
            raise RuntimeError("IDM episode is not active; call start() first")

    @staticmethod
    def positions(chunk_id: int) -> tuple[int, int, int]:
        if chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")
        return chunk_id, 1 + chunk_id * IDM_LATENTS_PER_CHUNK, chunk_id * IDM_ACTIONS_PER_CHUNK

    def predict(self) -> IDMChunk:
        self._require_active()
        if self.pending is not None:
            raise RuntimeError("commit or reset the pending IDM prediction before predict()")
        state_id, frame_start, action_start = self.positions(self.next_chunk_id)
        chunk = IDMChunk(self.next_chunk_id, state_id, frame_start, action_start, True)
        self.chunks.append(chunk)
        self.pending = chunk
        self._evict()
        return chunk

    def commit(self) -> IDMChunk:
        self._require_active()
        if self.pending is None:
            raise RuntimeError("no pending IDM prediction to commit")
        pred = self.pending
        self.chunks = [chunk for chunk in self.chunks if not chunk.is_pred]
        real = IDMChunk(pred.chunk_id, pred.state_id, pred.frame_start, pred.action_start, False)
        self.chunks.append(real)
        self.pending = None
        self.next_chunk_id += 1
        self._evict()
        return real

    def _evict(self) -> None:
        # ``capacity`` counts the rolling current+history chunk window; the
        # episode seed is an extra condition during the four trained chunks.
        # Keep seed+current/history while predicting chunk ids [0, capacity),
        # then roll to the last ``capacity`` normal chunks as one unit.
        cursor = self.pending.chunk_id if self.pending is not None else self.next_chunk_id
        limit = self.capacity + 1 if cursor < self.capacity else self.capacity
        while len(self.chunks) > limit:
            del self.chunks[0]

    def cache_record_limit(self, chunk_id: int) -> int:
        """Per-layer K/V record limit matching the metadata eviction policy."""
        return self.capacity + 1 if int(chunk_id) < self.capacity else self.capacity

    @property
    def completed_history(self) -> list[IDMChunk]:
        return [chunk for chunk in self.chunks if not chunk.is_pred and chunk.chunk_id >= 0]

    def snapshot(self) -> tuple[IDMChunk, ...]:
        return tuple(self.chunks)


def canonical_idm_frame_ids(
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    n_action: int,
    num_action_per_block: int,
    n_state: int,
    *,
    device=None,
):
    """Return video/action/state timeline IDs for a full training sequence.

    Frame 0 is the episode seed.  For chunk ``c`` the strict order is
    ``state(3c+1) -> video(3c+2) -> action(3c+3)``.
    """
    import torch

    if num_frames < 1:
        raise ValueError("IDM requires one seed frame")
    future_frames = num_frames - 1
    expected_frames = n_state * num_frame_per_block
    expected_actions = n_state * num_action_per_block
    if future_frames != expected_frames or n_action != expected_actions:
        raise ValueError(
            "non-canonical IDM sample: expected seed + "
            f"{n_state}*{num_frame_per_block} frames and "
            f"{n_state}*{num_action_per_block} actions, got "
            f"num_frames={num_frames}, n_action={n_action}"
        )

    video_frame_ids = torch.zeros(num_frames, dtype=torch.long, device=device)
    if future_frames:
        future_chunk = torch.arange(future_frames, device=device) // num_frame_per_block
        video_frame_ids[1:] = future_chunk * 3 + 2
    video_ids = video_frame_ids.repeat_interleave(frame_seqlen)

    action_chunk = torch.arange(n_action, device=device) // num_action_per_block
    action_ids = action_chunk * 3 + 3
    state_ids = torch.arange(n_state, device=device) * 3 + 1
    return video_ids, action_ids, state_ids
