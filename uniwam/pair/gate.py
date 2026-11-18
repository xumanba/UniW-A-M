"""Per-chunk interface selection for deployment.

``PAIRGate`` wraps a frozen UniW(A)M model and a trained router (or seed
ensemble).  Call it once per chunk boundary with the request batch the next
interface would receive and that request's position in its four-chunk
window::

    gate = PAIRGate(model, "outputs/pair/routers")
    task, probs = gate.route(batch, chunk_index)
    if task != active_task:
        model.wam_episode_reset(task)   # the new interface starts a fresh window
        active_task = task

``batch`` is built exactly like the offline requests
(``uniwam.inference.episode.BatchBuilder``): the seed frame on chunk 0 and
the nine frames of the preceding chunk afterwards, plus the current state and
instruction.  After an interface switch the new interface starts at chunk 0,
so the next ``chunk_index`` passed to the gate is 0 as well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch

from groot.vla.model.dreamzero.modules.pair_router import (
    PAIR_TASKS,
    PAIRFeatureEncoder,
    load_pair_router_ensemble,
    select_interface,
)


class PAIRGate:
    def __init__(
        self,
        model,
        routers: str | Path | Iterable[str | Path],
        *,
        device: str | torch.device = "cuda",
        exclude: Sequence[str] = (),
        chunk0_allowed: Sequence[str] = (),
        joint_threshold: float = 0.0,
        probe_experts: Sequence[str] = ("shared",),
    ):
        """
        Args:
            model: Frozen UniW(A)M model (``base_vla`` with its action head).
            routers: Router checkpoint, list of checkpoints, or a directory of
                ``pair_router_seed*.pt``; several seeds are averaged.
            exclude: Interfaces never selected.
            chunk0_allowed: Excluded interfaces that stay eligible on chunk 0.
                ``exclude=("idm_lingbot_va",), chunk0_allowed=("idm_lingbot_va",)``
                keeps IDM-WAM only where it needs no generated action history.
            joint_threshold: Fall back to Joint-WAM unless another interface
                wins with at least this probability.
            probe_experts: Must match the experts used for feature extraction.
        """
        self.encoder = PAIRFeatureEncoder(model, probe_experts=probe_experts)
        self.router = load_pair_router_ensemble(routers, device=device)
        self.exclude = tuple(exclude)
        self.chunk0_allowed = tuple(chunk0_allowed)
        self.joint_threshold = float(joint_threshold)

    @property
    def members(self) -> int:
        return len(getattr(self.router, "routers", [self.router]))

    @torch.no_grad()
    def route(self, batch: dict, chunk_index: int) -> tuple[str, dict[str, float]]:
        """Interface for the chunk requested by ``batch``, plus its probabilities."""
        device = next(self.router.parameters()).device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        features = self.encoder(batch, chunk_index)
        task, probs = select_interface(
            self.router, features, chunk_index,
            exclude=self.exclude, chunk0_allowed=self.chunk0_allowed,
            joint_threshold=self.joint_threshold,
        )
        return task, {name: float(p) for name, p in zip(PAIR_TASKS, probs)}
