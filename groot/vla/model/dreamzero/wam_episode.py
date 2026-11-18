"""Environment-agnostic lifecycle controller for WAM deployment episodes.

The runner deliberately owns only model-facing inference semantics.  Dataset
and robot adapters remain responsible for sourcing observations and deciding
which actions were actually executed, but they must satisfy the same request
contract exposed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Callable


WAM_ACTION_HORIZON = 24
WAM_HISTORY_OFFSETS = tuple(range(0, WAM_ACTION_HORIZON + 1, 3))
WAM_SEED_FRAMES = 1
WAM_HISTORY_FRAMES = len(WAM_HISTORY_OFFSETS)
WAM_JOINT_CHUNKS_PER_WINDOW = 4
WAM_ACWM_CHUNKS_PER_WINDOW = 4


_MODE_ALIASES = {
    "joint_dreamzero": "joint_dreamzero",
    "joint": "joint_dreamzero",
    "dreamzero": "joint_dreamzero",
    "idm_lingbot_va": "idm_lingbot_va",
    "lingbot_va": "idm_lingbot_va",
    "idm": "idm_lingbot_va",
    "action_only_fast_wam": "action_only_fast_wam",
    "fast_wam": "action_only_fast_wam",
    "fastwam": "action_only_fast_wam",
    "action_only_gigaworld_policy": "action_only_gigaworld_policy",
    "gigaworld_policy": "action_only_gigaworld_policy",
    "giga": "action_only_gigaworld_policy",
    "ac_wm": "ac_wm",
    "world_model": "ac_wm",
    "wm": "ac_wm",
    "embodied_world_model": "ac_wm",
}


def canonical_episode_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _MODE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported WAM episode mode: {mode!r}") from exc


def use_unified_inference_consumption() -> bool:
    """Return whether inference should consume the shared four-chunk window."""
    value = os.environ.get("WAM_UNIFIED_INFERENCE_CONSUMPTION", "1").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            "WAM_UNIFIED_INFERENCE_CONSUMPTION must be 0 or 1, "
            f"got {value!r}"
        )
    return value == "1"


def unified_inference_consumption_enabled() -> bool:
    """Compatibility alias used by inference entrypoints."""
    return use_unified_inference_consumption()


def episode_history_indices(start_step: int, episode_length: int) -> list[int]:
    """Return the shared ``start + 8x3`` observation indices for one horizon."""
    if int(episode_length) <= 0:
        raise ValueError("episode_length must be positive")
    return [
        min(int(start_step) + offset, int(episode_length) - 1)
        for offset in WAM_HISTORY_OFFSETS
    ]


@dataclass(frozen=True)
class WAMEpisodeRequest:
    mode: str
    kind: str
    expected_frames: int
    reset_before_predict: bool
    requires_history_commit: bool
    chunk_index: int
    chunk_in_window: int


class WAMEpisodeRunner:
    """Own the shared reset/request/predict lifecycle for one WAM mode.

    ``predict_fn`` is the model-facing operation.  For IDM it must be the
    canonical atomic episode step (commit prior real history when needed,
    generate video, then generate action).  For Joint and the single-horizon
    policies it is the corresponding one-chunk prediction call.

    Counters advance only after ``predict_fn`` returns successfully.
    """

    def __init__(
        self,
        mode: str,
        reset_fn: Callable[[], None],
        predict_fn: Callable[[Any], Any],
        *,
        chunks_per_window: int = WAM_JOINT_CHUNKS_PER_WINDOW,
        continuous_cache: bool = False,
    ) -> None:
        self.mode = canonical_episode_mode(mode)
        self._unified_inference_consumption = use_unified_inference_consumption()
        if continuous_cache and self.mode != "ac_wm":
            raise ValueError("continuous_cache is only valid for ac_wm")
        if int(chunks_per_window) <= 0:
            raise ValueError("chunks_per_window must be positive")
        if (
            self.mode == "ac_wm"
            and int(chunks_per_window) != WAM_ACWM_CHUNKS_PER_WINDOW
        ):
            raise ValueError(
                "ac_wm chunks_per_window is fixed at "
                f"{WAM_ACWM_CHUNKS_PER_WINDOW}"
            )
        self._reset_fn = reset_fn
        self._predict_fn = predict_fn
        self._chunks_per_window = int(chunks_per_window)
        self._continuous_cache = bool(continuous_cache)
        # AC-WM continuous rollouts re-anchor every training window.  Three
        # modes control the source of the window-boundary observation:
        #   "gt"       (default) reset the model episode and rebuild its
        #              carrier + CLIP from the real boundary frame;
        #   "clip_gt"  keep the generated-history carrier (no jump) but
        #              refresh the CLIP appearance condition from the real
        #              boundary frame (color drift cure without reseeding);
        #   "generated" fully autonomous, CLIP stays anchored as configured.
        self._window_reanchor = os.environ.get(
            "WAM_ACWM_WINDOW_REANCHOR", "clip_gt"
        ).strip().lower()
        if self._window_reanchor not in {"gt", "clip_gt", "generated"}:
            raise ValueError(
                "WAM_ACWM_WINDOW_REANCHOR must be gt, clip_gt or generated; "
                f"got {self._window_reanchor!r}"
            )
        self._completed_chunks = 0
        self._chunks_in_window = 0

    def reset_episode(self) -> None:
        self._reset_fn()
        self.reset_tracking()

    def reset_tracking(self) -> None:
        """Reset lifecycle counters after an already-completed model reset."""
        self._completed_chunks = 0
        self._chunks_in_window = 0

    def next_request(self) -> WAMEpisodeRequest:
        if self.mode == "joint_dreamzero":
            is_seed = self._chunks_in_window == 0
            return WAMEpisodeRequest(
                mode=self.mode,
                kind="seed" if is_seed else "history",
                expected_frames=WAM_SEED_FRAMES if is_seed else WAM_HISTORY_FRAMES,
                reset_before_predict=is_seed and self._completed_chunks > 0,
                requires_history_commit=False,
                chunk_index=self._completed_chunks,
                chunk_in_window=self._chunks_in_window,
            )
        if self.mode == "idm_lingbot_va":
            # IDM training sees one 33-frame window: one seed followed by four
            # independent 8-frame/24-action chunks. Deployment must therefore
            # refresh the image condition and all layer K/V at the same window
            # boundary instead of retaining the episode's first-frame CLIP
            # condition forever. The incoming nine-frame history still
            # contains the endpoint needed for the next one-frame seed.
            is_seed = self._chunks_in_window == 0
            return WAMEpisodeRequest(
                mode=self.mode,
                kind="seed" if is_seed else "history",
                expected_frames=WAM_SEED_FRAMES if is_seed else WAM_HISTORY_FRAMES,
                reset_before_predict=is_seed and self._completed_chunks > 0,
                requires_history_commit=not is_seed,
                chunk_index=self._completed_chunks,
                chunk_in_window=self._chunks_in_window,
            )
        if self.mode in (
            "action_only_fast_wam",
            "action_only_gigaworld_policy",
        ):
            if self._unified_inference_consumption:
                is_seed = self._chunks_in_window == 0
                return WAMEpisodeRequest(
                    mode=self.mode,
                    kind="seed" if is_seed else "history",
                    expected_frames=(
                        WAM_SEED_FRAMES if is_seed else WAM_HISTORY_FRAMES
                    ),
                    reset_before_predict=is_seed and self._completed_chunks > 0,
                    requires_history_commit=False,
                    chunk_index=self._completed_chunks,
                    chunk_in_window=self._chunks_in_window,
                )
            return WAMEpisodeRequest(
                mode=self.mode,
                kind="latest",
                expected_frames=WAM_SEED_FRAMES,
                reset_before_predict=False,
                requires_history_commit=False,
                chunk_index=self._completed_chunks,
                chunk_in_window=0,
            )
        if self.mode == "ac_wm":
            if self._unified_inference_consumption:
                if self._continuous_cache:
                    reanchor_seed = (
                        self._window_reanchor != "generated"
                        and self._completed_chunks > 0
                        and self._completed_chunks % self._chunks_per_window == 0
                    )
                    return WAMEpisodeRequest(
                        mode=self.mode,
                        kind=(
                            "seed"
                            if (self._completed_chunks == 0 or reanchor_seed)
                            else "generated"
                        ),
                        expected_frames=WAM_SEED_FRAMES,
                        reset_before_predict=(
                            reanchor_seed and self._window_reanchor == "gt"
                        ),
                        requires_history_commit=False,
                        chunk_index=self._completed_chunks,
                        chunk_in_window=self._chunks_in_window,
                    )
                is_seed = self._chunks_in_window == 0
                return WAMEpisodeRequest(
                    mode=self.mode,
                    kind="seed" if is_seed else "history",
                    expected_frames=(
                        WAM_SEED_FRAMES if is_seed else WAM_HISTORY_FRAMES
                    ),
                    reset_before_predict=is_seed and self._completed_chunks > 0,
                    requires_history_commit=not is_seed,
                    chunk_index=self._completed_chunks,
                    chunk_in_window=self._chunks_in_window,
                )
            return WAMEpisodeRequest(
                mode=self.mode,
                kind="latest",
                expected_frames=WAM_SEED_FRAMES,
                reset_before_predict=(
                    self._unified_inference_consumption
                    and self._chunks_in_window == 0
                    and self._completed_chunks > 0
                ),
                requires_history_commit=False,
                chunk_index=self._completed_chunks,
                chunk_in_window=self._chunks_in_window,
            )
        raise AssertionError(f"unhandled canonical mode {self.mode!r}")

    def predict(self, batch: Any, frame_count: int) -> Any:
        # Call the canonical implementation directly so compatibility wrappers
        # may expose a narrower legacy request dataclass without weakening the
        # internal mode-aware contract.
        request = WAMEpisodeRunner.next_request(self)
        if int(frame_count) != request.expected_frames:
            raise ValueError(
                f"{request.mode} {request.kind} request expects "
                f"{request.expected_frames} frame(s), got {frame_count}"
            )
        if request.reset_before_predict:
            self._reset_fn()
        result = self._predict_fn(batch)
        self._completed_chunks += 1
        if self.mode in (
            "joint_dreamzero",
            "idm_lingbot_va",
            "ac_wm",
        ) or (
            self._unified_inference_consumption
            and self.mode
            in ("action_only_fast_wam", "action_only_gigaworld_policy")
        ):
            self._chunks_in_window = (
                self._chunks_in_window + 1
            ) % self._chunks_per_window
        return result

    def state_snapshot(self) -> dict[str, Any]:
        request = WAMEpisodeRunner.next_request(self)
        return {
            "mode": self.mode,
            "completed_chunks": self._completed_chunks,
            "chunks_in_window": self._chunks_in_window,
            "chunks_per_window": (
                self._chunks_per_window
                if self.mode
                in ("joint_dreamzero", "idm_lingbot_va", "ac_wm")
                or (
                    self._unified_inference_consumption
                    and self.mode
                    in ("action_only_fast_wam", "action_only_gigaworld_policy")
                )
                else None
            ),
            "next_request": asdict(request),
        }
