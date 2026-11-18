"""Strict read-only access to immutable AC-WM 49-frame VAE latent caches."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open


ENV_NAME = "WAM_PRECOMPUTED_ACWM_LATENTS_ROOT"
INDEX_ENV_NAME = "WAM_PRECOMPUTED_ACWM_LATENTS_INDEX"
INDEX_NAME = "acwm_latent_index.sqlite3"
READY_NAME = "READY.json"
CONTRACT_VERSION = "agilex-wam-vae-v5-acwm49"
CONTINUOUS_SHAPE = (1, 48, 13, 10, 20)
WINDOW_FIRST_RGB_INDEX = 16
DEFAULT_AUGMENTATION_SEED = 20260720
INDEX_SCHEMA_VERSION = "2"
INDEX_INSERT_BATCH_SIZE = 8192
_MANIFEST_NAME = re.compile(r"manifest_m[^/]+_r\d+\.jsonl")


def configured_root() -> Path | None:
    raw = os.environ.get(ENV_NAME, "").strip()
    return Path(raw).resolve() if raw else None


def configured_index_path(root: Path) -> Path:
    raw = os.environ.get(INDEX_ENV_NAME, "").strip()
    return Path(raw).resolve() if raw else root / INDEX_NAME


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return value


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a full SHA256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value


def _annotation_scalar_bytes(value: object) -> bytes:
    if value is None:
        return b"N"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return b"B1" if value else b"B0"
    if isinstance(value, int):
        raw = str(value).encode("ascii")
        return b"I" + struct.pack(">Q", len(raw)) + raw
    if isinstance(value, float):
        if np.isnan(value):
            return b"Fnan"
        raw = value.hex().encode("ascii")
        return b"F" + struct.pack(">Q", len(raw)) + raw
    if isinstance(value, bytes):
        return b"Y" + struct.pack(">Q", len(value)) + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + struct.pack(">Q", len(raw)) + raw
    raise TypeError(f"unsupported annotation scalar type: {type(value)}")


def update_annotation_digest(
    digest,
    episode_index: int,
    values,
) -> None:
    digest.update(b"E" + struct.pack(">q", int(episode_index)))
    digest.update(struct.pack(">Q", len(values)))
    for row_index, value in enumerate(values):
        raw = _annotation_scalar_bytes(value)
        digest.update(struct.pack(">Q", row_index))
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)


def augmentation_seed(
    contract: dict,
    episode_index: int,
    window_start: int,
) -> int:
    raw = _canonical_json(
        {
            "base_seed": int(contract["pixel_transform"]["base_seed"]),
            "episode_index": int(episode_index),
            "window_start": int(window_start),
            "variant": 0,
        }
    )
    return (
        int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)
        & 0x7FFF_FFFF_FFFF_FFFF
    )


def load_contract(root: Path, *, require_full: bool = True) -> dict:
    contract = _read_json(root / "contract.json", "AC-WM latent contract")
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"unsupported AC-WM latent contract_version="
            f"{contract.get('contract_version')!r}; expected {CONTRACT_VERSION!r}"
        )
    expected_fields = {
        "vae_profile": "wam_moe",
        "layout": "continuous13",
        "save_dtype": "bf16",
        "tensor_shape": list(CONTINUOUS_SHAPE),
    }
    for key, expected in expected_fields.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"AC-WM latent contract mismatch for {key}: "
                f"{contract.get(key)!r} != {expected!r}"
            )
    temporal = contract.get("temporal") or {}
    expected_temporal = {
        "source_action_hz": 30,
        "rgb_hz": 10,
        "rgb_stride": 3,
        "window_rgb_frames": 49,
        "pre_roll_rgb_frames": 16,
        "current_window_rgb_frames": 33,
        "window_action_steps": 96,
        "chunks_per_window": 4,
        "rgb_frames_per_chunk": 9,
        "action_steps_per_chunk": 24,
        "window_first_rgb_index": WINDOW_FIRST_RGB_INDEX,
        "continuous_latents": 13,
        "pre_roll_left_pad": "episode_and_contiguous_annotation.task_boundary",
        "frame_indices": (
            "max(window_start-48+3*i,annotation_segment_start),i=0..15;"
            "window_start+3*i,i=0..32"
        ),
    }
    for key, expected in expected_temporal.items():
        if temporal.get(key) != expected:
            raise ValueError(
                f"AC-WM temporal contract mismatch for {key}: "
                f"{temporal.get(key)!r} != {expected!r}"
            )
    pixel = contract.get("pixel_transform") or {}
    expected_pixel = {
        "mode": "seeded_train",
        "scope": "window",
        "variants": 1,
        "base_seed": DEFAULT_AUGMENTATION_SEED,
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
                f"AC-WM pixel contract mismatch for {key}: "
                f"{pixel.get(key)!r} != {expected!r}"
            )
    for key in ("implementation_sha256", "config_sha256"):
        _validate_sha256(pixel.get(key), f"pixel_transform.{key}")
    config_payload = {
        key: pixel[key]
        for key in expected_pixel
    } | {
        "crop_scale": pixel.get("crop_scale"),
        "brightness": pixel.get("brightness"),
        "contrast": pixel.get("contrast"),
        "saturation": pixel.get("saturation"),
        "hue": pixel.get("hue"),
    }
    expected_config_hash = hashlib.sha256(
        _canonical_json(config_payload).encode("utf-8")
    ).hexdigest()
    if pixel["config_sha256"] != expected_config_hash:
        raise ValueError(
            "pixel_transform.config_sha256 does not match the transform config"
        )
    vae = contract.get("vae") or {}
    expected_vae = {
        "implementation": (
            "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE38"
        ),
        "posterior": "mean_mu_not_sample",
        "z_dim": 48,
        "dim": 160,
        "tiled": False,
    }
    for key, expected in expected_vae.items():
        if vae.get(key) != expected:
            raise ValueError(
                f"AC-WM VAE contract mismatch for {key}: "
                f"{vae.get(key)!r} != {expected!r}"
            )
    for key in ("weight_sha256", "implementation_sha256"):
        _validate_sha256(vae.get(key), f"vae.{key}")
    dataset = contract.get("dataset") or {}
    for key in (
        "info_sha256",
        "view_schema_sha256",
        "episodes_sha256",
        "annotation_task_sha256",
    ):
        _validate_sha256(dataset.get(key), f"dataset.{key}")
    if dataset.get("annotation_key") != "annotation.task":
        raise ValueError("dataset.annotation_key must be 'annotation.task'")
    coverage = contract.get("coverage") or {}
    if coverage.get("mode") not in {"full", "partial"}:
        raise ValueError("AC-WM coverage.mode must be full or partial")
    if require_full and coverage.get("mode") != "full":
        raise ValueError("partial AC-WM cache is not training-loadable")
    for key in ("expected_episode_count", "expected_window_count"):
        if not isinstance(coverage.get(key), int) or coverage[key] <= 0:
            raise ValueError(f"coverage.{key} must be positive")
    _validate_sha256(
        coverage.get("expected_identity_sha256"),
        "coverage.expected_identity_sha256",
    )

    contract_id = _validate_sha256(contract.get("contract_id"), "contract_id")
    unhashed = dict(contract)
    unhashed.pop("contract_id", None)
    expected_id = hashlib.sha256(
        _canonical_json(unhashed).encode("utf-8")
    ).hexdigest()
    if contract_id != expected_id:
        raise ValueError(
            f"AC-WM contract_id={contract_id!r} does not match full content "
            f"hash {expected_id!r}"
        )
    return contract


def _ready_signature(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ready(root: Path, contract_id: str) -> tuple[dict, str]:
    path = root / READY_NAME
    ready = _read_json(path, "AC-WM READY marker")
    if ready.get("schema_version") != 1:
        raise ValueError("unsupported AC-WM READY schema_version")
    if ready.get("contract_id") != contract_id:
        raise ValueError("AC-WM READY contract_id mismatch")
    _validate_sha256(ready.get("manifest_signature"), "READY manifest_signature")
    names = ready.get("manifest_files")
    if not isinstance(names, list) or not names:
        raise ValueError("READY manifest_files must be a non-empty list")
    if len(names) != len(set(names)):
        raise ValueError("READY manifest_files contains duplicates")
    for name in names:
        if not isinstance(name, str) or not _MANIFEST_NAME.fullmatch(name):
            raise ValueError(f"unsafe READY manifest name: {name!r}")
    count = ready.get("sample_count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("READY sample_count must be positive")
    shard_hashes = ready.get("shard_sha256")
    if not isinstance(shard_hashes, dict) or not shard_hashes:
        raise ValueError("READY shard_sha256 must be a non-empty object")
    for relative, digest in shard_hashes.items():
        resolve_shard_path(root, relative)
        _validate_sha256(digest, f"READY shard_sha256[{relative!r}]")
    _validate_sha256(
        ready.get("dataset_binding_sha256"),
        "READY dataset_binding_sha256",
    )
    return ready, _ready_signature(path)


def dataset_fingerprints(
    dataset_root: Path,
) -> tuple[str, str, str, str, int, int, str]:
    info_path = dataset_root.resolve() / "meta/info.json"
    raw = info_path.read_bytes()
    info = json.loads(raw)
    features = info.get("features") or {}
    keys = (
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    view_schema = {key: features.get(key) for key in keys}
    episodes_path = dataset_root / "meta/episodes.jsonl"
    episode_bytes = episodes_path.read_bytes()
    rows = [
        json.loads(line)
        for line in episode_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    template = str(info["data_path"])
    chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
    annotation_digest = hashlib.sha256()
    identity_digest = hashlib.sha256()
    window_count = 0
    normalized = []
    for position, row in enumerate(rows):
        episode = int(row.get("episode_index", position))
        length = int(row["length"])
        normalized.append((episode, length))
        parquet = dataset_root / template.format(
            episode_chunk=episode // chunk_size,
            episode_index=episode,
        )
        values = pq.read_table(
            parquet, columns=["annotation.task"]
        )["annotation.task"].to_numpy()
        update_annotation_digest(annotation_digest, episode, values)
    for episode, length in sorted(normalized):
        for start in range(max(0, length - 96)):
            identity_digest.update(f"{episode}:{start}\n".encode("ascii"))
            window_count += 1
    return (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(
            _canonical_json(view_schema).encode("utf-8")
        ).hexdigest(),
        hashlib.sha256(episode_bytes).hexdigest(),
        annotation_digest.hexdigest(),
        len(rows),
        window_count,
        identity_digest.hexdigest(),
    )


def manifest_paths(root: Path, ready: dict | None = None) -> list[Path]:
    if ready is None:
        contract = load_contract(root)
        ready, _ = load_ready(root, str(contract["contract_id"]))
    paths = [root / str(name) for name in ready["manifest_files"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AC-WM manifest is missing: {missing[0]}")
    return paths


def manifest_signature(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_shard_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("shard_path must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or len(pure.parts) != 2
        or pure.parts[0] != "shards"
        or pure.name in {"", ".", ".."}
    ):
        raise ValueError(
            f"shard_path must be relative to the cache shards directory: {relative!r}"
        )
    shards_root = (root / "shards").resolve()
    try:
        shards_root.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"cache shards directory escapes cache root: {root / 'shards'}"
        ) from exc
    resolved = (root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(shards_root)
    except ValueError as exc:
        raise ValueError(
            f"shard_path escapes the cache shards directory: {relative!r}"
        ) from exc
    return resolved


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ready_shards(root: Path, ready: dict) -> None:
    for relative, expected in ready["shard_sha256"].items():
        path = resolve_shard_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"AC-WM latent shard is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"AC-WM READY shard hash {actual} != {expected}: {path}"
            )


def _index_matches(
    index_path: Path,
    contract_id: str,
    ready_signature: str,
    shard_snapshot_digest: str,
) -> bool:
    if not index_path.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        return (
            metadata.get("contract_id") == contract_id
            and metadata.get("ready_signature") == ready_signature
            and metadata.get("schema_version") == INDEX_SCHEMA_VERSION
            and metadata.get("shard_snapshot_digest")
            == shard_snapshot_digest
        )
    except (OSError, sqlite3.Error):
        return False


def _iter_manifest_rows(
    root: Path,
    paths: list[Path],
    contract: dict,
):
    contract_id = str(contract["contract_id"])
    for manifest in paths:
        with open(manifest, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record["contract_id"] != contract_id:
                        raise ValueError("contract_id mismatch")
                    if record.get("tensor_dtype") != "bf16":
                        raise ValueError("tensor_dtype must be bf16")
                    if record.get("tensor_shape") != list(CONTINUOUS_SHAPE):
                        raise ValueError(
                            f"tensor_shape={record.get('tensor_shape')!r}"
                        )
                    episode_index = int(record["episode_index"])
                    window_start = int(record["window_start"])
                    seed = int(record["augmentation_seed"])
                    expected_seed = augmentation_seed(
                        contract, episode_index, window_start
                    )
                    if seed != expected_seed:
                        raise ValueError(
                            f"augmentation_seed {seed} != {expected_seed}"
                        )
                    shard_relative = str(record["shard_path"])
                    resolve_shard_path(root, shard_relative)
                    tensor_key = str(record["tensor_key"])
                    if not tensor_key:
                        raise ValueError("tensor_key is empty")
                    content_hash = _validate_sha256(
                        record["content_sha256"], "content_sha256"
                    )
                    yield (
                        episode_index,
                        window_start,
                        shard_relative,
                        tensor_key,
                        seed,
                        content_hash,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid AC-WM manifest row "
                        f"{manifest}:{line_number}: {exc}"
                    ) from exc


def build_index(root: Path, index_path: Path | None = None) -> Path:
    root = root.resolve()
    contract = load_contract(root)
    contract_id = str(contract["contract_id"])
    ready, ready_sig = load_ready(root, contract_id)
    shard_snapshot_digest = hashlib.sha256(
        _canonical_json(ready["shard_sha256"]).encode("utf-8")
    ).hexdigest()
    paths = manifest_paths(root, ready)
    signature = manifest_signature(paths)
    if signature != ready["manifest_signature"]:
        raise RuntimeError("AC-WM manifests do not match the immutable READY snapshot")
    index_path = (
        configured_index_path(root)
        if index_path is None
        else index_path.resolve()
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_name(f".{index_path.name}.lock")
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ready, ready_sig = load_ready(root, contract_id)
        shard_snapshot_digest = hashlib.sha256(
            _canonical_json(ready["shard_sha256"]).encode("utf-8")
        ).hexdigest()
        paths = manifest_paths(root, ready)
        signature = manifest_signature(paths)
        if signature != ready["manifest_signature"]:
            raise RuntimeError(
                "AC-WM manifests do not match the immutable READY snapshot"
            )
        if _index_matches(
            index_path, contract_id, ready_sig, shard_snapshot_digest
        ):
            return index_path
        validate_ready_shards(root, ready)
        tmp_path = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}")
        if tmp_path.exists():
            tmp_path.unlink()
        connection = sqlite3.connect(tmp_path)
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
                    tensor_key TEXT NOT NULL,
                    augmentation_seed INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    PRIMARY KEY (episode_index, window_start)
                ) WITHOUT ROWID;
                """
            )
            pending: list[tuple] = []
            try:
                for row in _iter_manifest_rows(root, paths, contract):
                    pending.append(row)
                    if len(pending) >= INDEX_INSERT_BATCH_SIZE:
                        connection.executemany(
                            "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)",
                            pending,
                        )
                        row_count += len(pending)
                        pending.clear()
                if pending:
                    connection.executemany(
                        "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)",
                        pending,
                    )
                    row_count += len(pending)
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "duplicate (episode_index, window_start) in AC-WM manifests"
                ) from exc
            if row_count == 0:
                raise ValueError("AC-WM latent manifests contain no samples")
            if row_count != int(ready["sample_count"]):
                raise ValueError(
                    f"READY sample_count {ready['sample_count']} != {row_count}"
                )
            connection.execute(
                "CREATE INDEX samples_by_shard ON samples(shard_path)"
            )
            referenced_shards = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT shard_path FROM samples"
                )
            }
            if referenced_shards != set(ready["shard_sha256"]):
                raise ValueError(
                    "READY shard snapshot does not exactly match manifest shards"
                )
            for relative in sorted(referenced_shards):
                shard_path = resolve_shard_path(root, relative)
                if not shard_path.is_file():
                    raise FileNotFoundError(
                        f"AC-WM latent shard is missing: {shard_path}"
                    )
                expected_keys = {
                    row[0]
                    for row in connection.execute(
                        "SELECT tensor_key FROM samples WHERE shard_path=?",
                        (relative,),
                    )
                }
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
                    if metadata.get("contract_id") != contract_id:
                        raise ValueError(
                            f"shard contract_id mismatch in {shard_path}"
                        )
                    available = set(handle.keys())
                missing_keys = expected_keys - available
                if missing_keys:
                    raise KeyError(
                        f"AC-WM shard {shard_path} is missing tensor key "
                        f"{sorted(missing_keys)[0]!r}"
                    )
            if (
                manifest_signature(paths) != signature
                or _ready_signature(root / READY_NAME) != ready_sig
            ):
                raise RuntimeError(
                    "AC-WM READY/manifests changed while the index was built"
                )
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [
                    ("schema_version", INDEX_SCHEMA_VERSION),
                    ("contract_id", contract_id),
                    ("ready_signature", ready_sig),
                    ("manifest_signature", signature),
                    ("sample_count", str(row_count)),
                    (
                        "shard_snapshot_digest",
                        hashlib.sha256(
                            _canonical_json(
                                ready["shard_sha256"]
                            ).encode("utf-8")
                        ).hexdigest(),
                    ),
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
class ACWMLatentSample:
    continuous: torch.Tensor
    augmentation_seed: int
    content_sha256: str


class PrecomputedACWMLatentStore:
    """Per-worker access with strict first-load content verification."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.contract = load_contract(self.root)
        ready, ready_sig = load_ready(
            self.root, str(self.contract["contract_id"])
        )
        binding_path = self.root / "dataset_binding.json"
        if file_sha256(binding_path) != ready["dataset_binding_sha256"]:
            raise RuntimeError("AC-WM dataset binding snapshot is stale")
        binding = _read_json(binding_path, "AC-WM dataset binding")
        if (
            binding.get("contract_id") != self.contract["contract_id"]
            or binding.get("dataset") != self.contract["dataset"]
            or binding.get("coverage") != self.contract["coverage"]
        ):
            raise ValueError("AC-WM dataset binding does not match contract")
        self.index_path = configured_index_path(self.root)
        shard_snapshot_digest = hashlib.sha256(
            _canonical_json(ready["shard_sha256"]).encode("utf-8")
        ).hexdigest()
        if not _index_matches(
            self.index_path,
            str(self.contract["contract_id"]),
            ready_sig,
            shard_snapshot_digest,
        ):
            raise RuntimeError(
                f"AC-WM latent index is missing or stale: {self.index_path}; "
                "run vae_precoding/build_acwm_latent_index.py first"
            )
        self._connection: sqlite3.Connection | None = None
        self._verified_samples: set[tuple[int, int, str]] = set()
        self._validated_dataset_roots: set[Path] = set()

    @classmethod
    def from_environment(cls) -> "PrecomputedACWMLatentStore | None":
        root = configured_root()
        return None if root is None else cls(root)

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{self.index_path}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
        return self._connection

    def validate_dataset(self, dataset_root: Path) -> None:
        dataset_root = dataset_root.resolve()
        if dataset_root in self._validated_dataset_roots:
            return
        expected = self.contract["dataset"]
        current_info_hash = hashlib.sha256(
            (dataset_root / "meta/info.json").read_bytes()
        ).hexdigest()
        if current_info_hash != expected["info_sha256"]:
            raise ValueError(
                "AC-WM cache dataset info fingerprint does not match "
                f"{dataset_root}"
            )
        episode_bytes = (dataset_root / "meta/episodes.jsonl").read_bytes()
        episodes_hash = hashlib.sha256(episode_bytes).hexdigest()
        rows = [
            json.loads(line)
            for line in episode_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        info = json.loads(
            (dataset_root / "meta/info.json").read_text()
        )
        view_hash = hashlib.sha256(
            _canonical_json(
                {
                    key: (info.get("features") or {}).get(key)
                    for key in (
                        "observation.images.head",
                        "observation.images.left_wrist",
                        "observation.images.right_wrist",
                    )
                }
            ).encode("utf-8")
        ).hexdigest()
        if view_hash != expected["view_schema_sha256"]:
            raise ValueError(
                "AC-WM cache dataset view-schema fingerprint does not match "
                f"{dataset_root}"
            )
        if episodes_hash != expected["episodes_sha256"]:
            raise ValueError("AC-WM cache episodes fingerprint mismatch")
        episode_count = len(rows)
        identity = hashlib.sha256()
        window_count = 0
        for position, row in sorted(
            enumerate(rows),
            key=lambda item: int(
                item[1].get("episode_index", item[0])
            ),
        ):
            episode = int(row.get("episode_index", position))
            for start in range(max(0, int(row["length"]) - 96)):
                identity.update(f"{episode}:{start}\n".encode("ascii"))
                window_count += 1
        identity_hash = identity.hexdigest()
        coverage = self.contract["coverage"]
        if (
            episode_count != coverage["expected_episode_count"]
            or window_count != coverage["expected_window_count"]
            or identity_hash != coverage["expected_identity_sha256"]
        ):
            raise ValueError("AC-WM cache dataset coverage fingerprint mismatch")
        self._validated_dataset_roots.add(dataset_root)

    def load(self, episode_index: int, window_start: int) -> ACWMLatentSample:
        episode_index = int(episode_index)
        window_start = int(window_start)
        row = self._db().execute(
            """
            SELECT shard_path, tensor_key, augmentation_seed, content_sha256
            FROM samples WHERE episode_index=? AND window_start=?
            """,
            (episode_index, window_start),
        ).fetchone()
        if row is None:
            raise KeyError(
                "precomputed AC-WM latent is missing for "
                f"episode={episode_index} window_start={window_start}"
            )
        shard_relative, tensor_key, saved_seed, saved_hash = row
        expected_seed = augmentation_seed(
            self.contract, episode_index, window_start
        )
        if int(saved_seed) != expected_seed:
            raise ValueError(
                f"indexed AC-WM augmentation seed {saved_seed} != {expected_seed}"
            )
        shard_path = resolve_shard_path(self.root, shard_relative)
        if not shard_path.is_file():
            raise FileNotFoundError(f"AC-WM latent shard is missing: {shard_path}")
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("contract_id") != self.contract["contract_id"]:
                    raise ValueError("shard contract_id mismatch")
                if str(tensor_key) not in handle.keys():
                    raise KeyError(f"tensor key {tensor_key!r} is missing")
                continuous = handle.get_tensor(str(tensor_key))
        except Exception as exc:
            raise RuntimeError(
                f"failed to read AC-WM latent shard {shard_path}: {exc}"
            ) from exc
        if tuple(continuous.shape) != CONTINUOUS_SHAPE:
            raise ValueError(
                f"AC-WM continuous latent shape {tuple(continuous.shape)} "
                f"!= {CONTINUOUS_SHAPE}"
            )
        if continuous.dtype != torch.bfloat16:
            raise TypeError(
                f"AC-WM precomputed latent must be BF16, got {continuous.dtype}"
            )
        verification_key = (episode_index, window_start, str(saved_hash))
        if verification_key not in self._verified_samples:
            actual_hash = tensor_sha256(continuous)
            if actual_hash != saved_hash:
                raise ValueError(
                    f"AC-WM latent content hash {actual_hash} != {saved_hash}"
                )
            self._verified_samples.add(verification_key)
        return ACWMLatentSample(
            continuous=continuous[0].contiguous(),
            augmentation_seed=expected_seed,
            content_sha256=str(saved_hash),
        )
