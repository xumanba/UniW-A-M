"""Validation helpers for launch-owned training checkpoint staging."""

from __future__ import annotations

import json
from pathlib import Path


EFFECTIVE_CONFIG_RELATIVE_PATH = Path("experiment_cfg/effective_config.json")


def write_effective_config(root: str | Path, payload: dict) -> Path:
    """Atomically persist the effective runtime contract beside checkpoints."""
    path = Path(root) / EFFECTIVE_CONFIG_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_effective_config(root: str | Path) -> dict:
    path = Path(root) / EFFECTIVE_CONFIG_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint effective config is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"effective config must be a JSON object: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported effective config schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    model_config_path = Path(root) / "config.json"
    if model_config_path.is_file():
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        embedded = model_config.get("wam_effective_config")
        if embedded is not None and embedded != payload:
            raise ValueError(
                "checkpoint config.json and experiment effective config disagree"
            )
    return payload


def unified_inference_expected_config(*, unified: bool) -> dict:
    """Return the current training-side contract required by unified inference."""
    if not unified:
        return {"wam": {"unified_joint_consumption": False}}
    # Mixed training uses per-sample routing while each inference launch is
    # task-homogeneous and uses global-task selection. The semantic contract is
    # token-group granularity; the orchestration routing mode need not match.
    return {
        "wam": {
            "unified_joint_consumption": True,
            "unified_chunkwise_attention": False,
            "moe_routing_granularity": "token_group_level",
            "num_frames": 33,
            "action_horizon": 24,
            "num_frame_per_block": 2,
            "num_action_per_block": 24,
            "num_state_per_block": 1,
        }
    }


def _validate_expected_fields(actual: dict, expected: dict, prefix: str = "") -> None:
    for key, expected_value in expected.items():
        field = f"{prefix}.{key}" if prefix else key
        if key not in actual:
            raise ValueError(f"effective config is missing required field {field}")
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                raise ValueError(f"effective config field {field} must be an object")
            _validate_expected_fields(actual_value, expected_value, field)
        elif actual_value != expected_value:
            raise ValueError(
                f"effective config mismatch for {field}: "
                f"checkpoint={actual_value!r}, launch={expected_value!r}"
            )


def validate_effective_config(
    root: str | Path,
    expected: dict,
    *,
    allow_legacy: bool = False,
    require_legacy_topology: bool = False,
) -> dict | None:
    """Validate a checkpoint contract, with an explicit legacy escape hatch."""
    try:
        payload = load_effective_config(root)
    except FileNotFoundError:
        if not allow_legacy:
            raise
        topology_path = Path(root) / "experiment_cfg/training_topology.json"
        if require_legacy_topology and not topology_path.is_file():
            raise FileNotFoundError(
                f"legacy checkpoint training topology is missing: {topology_path}"
            )
        if topology_path.is_file():
            topology = json.loads(topology_path.read_text(encoding="utf-8"))
            expected_granularity = expected.get("wam", {}).get(
                "moe_routing_granularity"
            )
            if (
                expected_granularity is not None
                and topology.get("moe_routing_granularity")
                != expected_granularity
            ):
                raise ValueError(
                    "legacy checkpoint routing mismatch: "
                    f"checkpoint={topology.get('moe_routing_granularity')!r}, "
                    f"launch={expected_granularity!r}"
                )
        return None
    _validate_expected_fields(payload, expected)
    return payload


def validate_checkpoint_dir(path: str | Path) -> list[str]:
    """Return the complete safetensors file list or raise a useful error."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"staged checkpoint directory is missing: {root}")
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"staged checkpoint config is missing: {config}")

    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.is_file():
        return [single.name]
    if not index.is_file():
        raise FileNotFoundError(
            f"no model.safetensors or model.safetensors.index.json under {root}"
        )

    payload = json.loads(index.read_text(encoding="utf-8"))
    names = sorted(set(payload.get("weight_map", {}).values()))
    if not names:
        raise ValueError(f"empty or invalid weight_map in {index}")
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"staged checkpoint is missing {len(missing)} referenced shard(s): {preview}"
        )
    return names


def validate_full_resume_dir(path: str | Path) -> list[str]:
    """Validate model files plus the HF/DeepSpeed state needed for full resume."""
    root = Path(path)
    shards = validate_checkpoint_dir(root)
    trainer_state = root / "trainer_state.json"
    if not trainer_state.is_file():
        raise FileNotFoundError(f"full resume trainer state is missing: {trainer_state}")
    tags = [p for p in root.glob("global_step*") if p.is_dir()]
    if not tags:
        raise FileNotFoundError(f"full resume DeepSpeed state is missing under {root}")
    return shards


def is_safe_launch_stage(path: str | Path, stage_root: str | Path) -> bool:
    candidate = Path(path).resolve()
    root = Path(stage_root).resolve()
    return candidate.parent == root and candidate.name.startswith(".train_stage_")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    shards = (
        validate_full_resume_dir(args.checkpoint)
        if args.full
        else validate_checkpoint_dir(args.checkpoint)
    )
    print(f"validated {args.checkpoint}: {len(shards)} safetensors file(s)")
