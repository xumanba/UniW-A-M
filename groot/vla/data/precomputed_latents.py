"""Strict access to offline AgileX WAM VAE latents."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


ENV_NAME = "WAM_PRECOMPUTED_LATENTS_ROOT"
INDEX_ENV_NAME = "WAM_PRECOMPUTED_LATENTS_INDEX"
INDEX_NAME = "latent_index.sqlite3"
CONTRACT_SCOPES = {
    "agilex-wam-vae-v4": "episode",
}
# The same v4 shard contains two layouts, but the training consumers have
# different temporal contracts.  Keep the consumer name explicit at the data
# boundary so a full-window tensor cannot be silently interpreted as four
# independent single-horizon samples (or vice versa).
PRECOMPUTED_CONSUMER_CONTRACT_FIELD = "wam_precomputed_latent_contract"
V4_SINGLE_HORIZON_CONSUMER = "v4_single_horizon"
FULL_WINDOW_UNIFIED_CONSUMER = "full_window_unified"
FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER = "full_window_legacy_seed_rgb"
V4_SINGLE_HORIZON_CONSUMER_CODE = 0
FULL_WINDOW_UNIFIED_CONSUMER_CODE = 1
FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER_CODE = 2
V4_CONSUMER_CAPABILITIES = {
    FULL_WINDOW_UNIFIED_CONSUMER: {
        "rgb_carrier_frames": 33,
        "latent_key": "continuous_latents",
        "latent_shape": [48, 9, 10, 20],
    },
    FULL_WINDOW_LEGACY_SEED_RGB_CONSUMER: {
        "rgb_carrier_frames": 4,
        "latent_key": "continuous_latents",
        "latent_shape": [48, 9, 10, 20],
    },
    V4_SINGLE_HORIZON_CONSUMER: {
        "rgb_carrier_frames": 4,
        "latent_key": "chunked_latents",
        "latent_shape": [4, 48, 3, 10, 20],
    },
}
CONTINUOUS_SHAPE = (1, 48, 9, 10, 20)
CHUNKED_SHAPE = (1, 4, 48, 3, 10, 20)


def configured_root() -> Path | None:
    raw = os.environ.get(ENV_NAME, "").strip()
    return Path(raw).resolve() if raw else None


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"precomputed latent contract is missing: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return value


def validate_consumer_capability(
    contract: dict,
    consumer: str,
    *,
    require_declared: bool = True,
) -> bool:
    """Validate one v4 consumer capability without weakening legacy reads.

    Old v4 caches predate the capability block and remain readable by the
    legacy single-horizon path.  A full-window unified consumer must have the
    producer declaration, because its 33-frame/continuous interpretation is
    not inferable from the shard tensors alone.
    """
    expected = V4_CONSUMER_CAPABILITIES.get(consumer)
    if expected is None:
        raise ValueError(f"unsupported v4 consumer contract: {consumer!r}")
    capabilities = contract.get("consumer_contracts")
    if capabilities is None:
        if require_declared:
            raise ValueError(
                "v4 producer contract has no consumer_contracts capability block; "
                f"cannot use {consumer!r}"
            )
        return False
    if not isinstance(capabilities, dict):
        raise ValueError("consumer_contracts must be a JSON object")
    actual = capabilities.get(consumer)
    if not isinstance(actual, dict):
        raise ValueError(
            f"v4 producer contract is missing consumer capability {consumer!r}"
        )
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"consumer capability {consumer!r} mismatch for {key}: "
                f"{actual.get(key)!r} != {value!r}"
            )
    return True


def load_contract(root: Path) -> dict:
    contract = _read_json(root / "contract.json")
    contract_version = contract.get("contract_version")
    if contract_version not in CONTRACT_SCOPES:
        raise ValueError(
            f"unsupported latent contract_version={contract_version!r}; "
            f"expected one of {sorted(CONTRACT_SCOPES)!r}"
        )
    pixel = contract.get("pixel_transform") or {}
    scope = pixel.get("scope", "window")
    if (
        pixel.get("mode") != "seeded_train"
        or scope != CONTRACT_SCOPES[contract_version]
        or int(pixel.get("variants", -1)) != 1
    ):
        raise ValueError(
            "precomputed training requires pixel_transform mode=seeded_train, "
            f"scope={CONTRACT_SCOPES[contract_version]}, and variants=1; got "
            f"mode={pixel.get('mode')!r} scope={scope!r} "
            f"variants={pixel.get('variants')!r}"
        )
    expected_pixel = {
        "per_view_resize_hw": [160, 320],
        "mosaic": "head-width-doubled-over-left-right-wrists",
        "model_resize_hw": [160, 320],
        "normalization": "uint8->[0,1]->(x-0.5)/0.5",
        "model_resize_dtype": "bf16",
        "model_resize_device": "cuda",
    }
    for key, expected in expected_pixel.items():
        if pixel.get(key) != expected:
            raise ValueError(
                f"WAM pixel transform mismatch for {key}: "
                f"{pixel.get(key)!r} != {expected!r}"
            )
    if contract.get("vae_profile") != "wam_moe":
        raise ValueError(
            f"expected vae_profile='wam_moe', got "
            f"{contract.get('vae_profile')!r}"
        )
    if contract.get("layouts") != "both":
        raise ValueError(
            f"five-task training requires latent layouts='both', got {contract.get('layouts')!r}"
        )
    consumer_contracts = contract.get("consumer_contracts")
    if consumer_contracts is not None:
        for consumer in V4_CONSUMER_CAPABILITIES:
            validate_consumer_capability(
                contract,
                consumer,
                require_declared=True,
            )
    if contract.get("save_dtype") != "bf16":
        raise ValueError(
            f"precomputed training requires BF16 tensors, got {contract.get('save_dtype')!r}"
        )
    if contract.get("single_frame_seed_override") is not True:
        raise ValueError("chunked latent contract is missing single-frame seed override")
    temporal = contract.get("temporal") or {}
    expected_temporal = {
        "source_action_hz": 30,
        "rgb_hz": 10,
        "rgb_stride": 3,
        "window_rgb_frames": 33,
        "window_action_steps": 96,
        "chunks_per_window": 4,
        "rgb_frames_per_chunk": 9,
        "action_steps_per_chunk": 24,
        "chunk_frame_offsets": [0, 8, 16, 24],
    }
    for key, expected in expected_temporal.items():
        if temporal.get(key) != expected:
            raise ValueError(
                f"latent temporal contract mismatch for {key}: "
                f"{temporal.get(key)!r} != {expected!r}"
            )
    vae = contract.get("vae") or {}
    if vae.get("posterior") != "mean_mu_not_sample":
        raise ValueError("precomputed latents must use the VAE posterior mean")
    if vae.get("z_dim") != 48 or vae.get("tiled") is not False:
        raise ValueError("UniW(A)M MoE requires non-tiled Wan2.2 VAE z_dim=48")
    contract_id = contract.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError("latent contract has no contract_id")
    unhashed = dict(contract)
    unhashed.pop("contract_id", None)
    raw = json.dumps(
        unhashed, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    expected_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    if contract_id != expected_id:
        raise ValueError(
            f"latent contract_id={contract_id!r} does not match content hash "
            f"{expected_id!r}"
        )
    return contract


def manifest_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("manifest_m*_r*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no latent manifests found under {root}")
    return paths


def manifest_signature(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def augmentation_seed(contract: dict, episode_index: int, window_start: int) -> int:
    pixel = contract["pixel_transform"]
    value = {
        "base_seed": int(pixel["base_seed"]),
        "episode_index": int(episode_index),
        "variant": 0,
    }
    if pixel.get("scope", "window") == "window":
        value["window_start"] = int(window_start)
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16) & 0x7FFF_FFFF_FFFF_FFFF


def _index_matches(index_path: Path, contract_id: str, signature: str) -> bool:
    if not index_path.is_file():
        return False
    try:
        uri = f"file:{index_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        return (
            metadata.get("contract_id") == contract_id
            and metadata.get("manifest_signature") == signature
            and int(metadata.get("schema_version", "0")) == 1
        )
    except (OSError, sqlite3.Error, ValueError):
        return False


def configured_index_path(root: Path) -> Path:
    raw = os.environ.get(INDEX_ENV_NAME, "").strip()
    return Path(raw).resolve() if raw else root / INDEX_NAME


def build_index(root: Path, index_path: Path | None = None) -> Path:
    """Build or reuse the one shared on-disk manifest index."""
    root = root.resolve()
    contract = load_contract(root)
    paths = manifest_paths(root)
    signature = manifest_signature(paths)
    contract_id = str(contract["contract_id"])
    index_path = (
        configured_index_path(root)
        if index_path is None
        else index_path.resolve()
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_name(f".{index_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        paths = manifest_paths(root)
        signature = manifest_signature(paths)
        if _index_matches(index_path, contract_id, signature):
            return index_path

        tmp_path = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}")
        if tmp_path.exists():
            tmp_path.unlink()
        connection = sqlite3.connect(tmp_path)
        shard_paths: set[str] = set()
        row_count = 0
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=MEMORY;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE samples (
                    episode_index INTEGER NOT NULL,
                    window_start INTEGER NOT NULL,
                    shard_path TEXT NOT NULL,
                    continuous_key TEXT NOT NULL,
                    chunked_key TEXT NOT NULL,
                    augmentation_seed INTEGER NOT NULL,
                    PRIMARY KEY (episode_index, window_start)
                ) WITHOUT ROWID;
                """
            )
            pending = []
            for manifest in paths:
                with open(manifest, "r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            keys = record["tensor_keys"]
                            if record["contract_id"] != contract_id:
                                raise ValueError(
                                    f"contract_id={record['contract_id']!r} != {contract_id!r}"
                                )
                            if record.get("tensor_dtype") != "bf16":
                                raise ValueError(
                                    f"tensor_dtype={record.get('tensor_dtype')!r}"
                                )
                            seeds = record["augmentation_seeds"]
                            if not isinstance(seeds, list) or len(seeds) != 1:
                                raise ValueError(f"augmentation_seeds={seeds!r}")
                            episode_index = int(record["episode_index"])
                            window_start = int(record["window_start"])
                            seed = int(seeds[0])
                            expected_seed = augmentation_seed(
                                contract, episode_index, window_start
                            )
                            if seed != expected_seed:
                                raise ValueError(
                                    f"augmentation seed {seed} != expected {expected_seed}"
                                )
                            shard_path = str(record["shard_path"])
                            pending.append(
                                (
                                    episode_index,
                                    window_start,
                                    shard_path,
                                    str(keys["continuous_latents"]),
                                    str(keys["chunked_latents"]),
                                    seed,
                                )
                            )
                            shard_paths.add(shard_path)
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(
                                f"invalid latent manifest row {manifest}:{line_number}: {exc}"
                            ) from exc
                        if len(pending) >= 10_000:
                            try:
                                connection.executemany(
                                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)",
                                    pending,
                                )
                            except sqlite3.IntegrityError as exc:
                                raise ValueError(
                                    "duplicate (episode_index, window_start) in latent manifests"
                                ) from exc
                            row_count += len(pending)
                            pending.clear()
            if pending:
                try:
                    connection.executemany(
                        "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)", pending
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "duplicate (episode_index, window_start) in latent manifests"
                    ) from exc
                row_count += len(pending)
            if row_count == 0:
                raise ValueError(f"latent manifests under {root} contain no samples")
            missing = [
                relative for relative in sorted(shard_paths)
                if not (root / relative).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} latent shards referenced by manifests are missing; "
                    f"first={missing[0]}"
                )
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [
                    ("schema_version", "1"),
                    ("contract_id", contract_id),
                    ("manifest_signature", signature),
                    ("sample_count", str(row_count)),
                ],
            )
            connection.commit()
        except Exception:
            connection.close()
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        else:
            connection.close()
            os.replace(tmp_path, index_path)
        return index_path


@dataclass(frozen=True)
class LatentSample:
    continuous: torch.Tensor
    chunked: torch.Tensor
    augmentation_seed: int


class PrecomputedLatentStore:
    """Per-worker read-only access to the shared SQLite/safetensors dataset."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.contract = load_contract(self.root)
        self.index_path = configured_index_path(self.root)
        paths = manifest_paths(self.root)
        signature = manifest_signature(paths)
        if not _index_matches(
            self.index_path, str(self.contract["contract_id"]), signature
        ):
            raise RuntimeError(
                f"latent index is missing or stale: {self.index_path}; "
                "run vae_precoding/build_latent_index.py first"
            )
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def from_environment(cls) -> "PrecomputedLatentStore | None":
        root = configured_root()
        unified = os.environ.get("WAM_UNIFIED_JOINT_CONSUMPTION", "0").strip()
        acwm_root = os.environ.get("WAM_PRECOMPUTED_ACWM_LATENTS_ROOT", "").strip()
        if unified == "1" and acwm_root:
            raise RuntimeError(
                "unified full-window v4 cache conflicts with the AC-WM v5 root; "
                "unset WAM_PRECOMPUTED_ACWM_LATENTS_ROOT and configure a v4 root"
            )
        return None if root is None else cls(root)

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{self.index_path}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
        return self._connection

    def load(self, episode_index: int, window_start: int) -> LatentSample:
        row = self._db().execute(
            """
            SELECT shard_path, continuous_key, chunked_key, augmentation_seed
            FROM samples WHERE episode_index=? AND window_start=?
            """,
            (int(episode_index), int(window_start)),
        ).fetchone()
        if row is None:
            raise KeyError(
                "precomputed latent is missing for "
                f"episode={episode_index} window_start={window_start}"
            )
        shard_relative, continuous_key, chunked_key, saved_seed = row
        expected_seed = augmentation_seed(
            self.contract, int(episode_index), int(window_start)
        )
        if int(saved_seed) != expected_seed:
            raise ValueError(
                f"indexed augmentation seed {saved_seed} != expected {expected_seed}"
            )
        shard_path = self.root / str(shard_relative)
        if not shard_path.is_file():
            raise FileNotFoundError(f"precomputed latent shard is missing: {shard_path}")
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("contract_id") != self.contract["contract_id"]:
                    raise ValueError(
                        f"shard contract_id mismatch in {shard_path}"
                    )
                continuous = handle.get_tensor(str(continuous_key))
                chunked = handle.get_tensor(str(chunked_key))
        except Exception as exc:
            raise RuntimeError(
                f"failed to read precomputed latent shard {shard_path}: {exc}"
            ) from exc
        if tuple(continuous.shape) != CONTINUOUS_SHAPE:
            raise ValueError(
                f"continuous latent shape {tuple(continuous.shape)} != {CONTINUOUS_SHAPE}"
            )
        if tuple(chunked.shape) != CHUNKED_SHAPE:
            raise ValueError(
                f"chunked latent shape {tuple(chunked.shape)} != {CHUNKED_SHAPE}"
            )
        if continuous.dtype != torch.bfloat16 or chunked.dtype != torch.bfloat16:
            raise TypeError(
                f"precomputed latents must be BF16, got {continuous.dtype}/{chunked.dtype}"
            )
        return LatentSample(
            continuous=continuous[0].contiguous(),
            chunked=chunked[0].contiguous(),
            augmentation_seed=expected_seed,
        )
