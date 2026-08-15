"""Shared torch-free helpers for the v1 reward pipeline (scripts 26-32).

Everything in this module runs with only the Python standard library plus
numpy. It is imported by the Phase 2-5 scripts and by their unit tests, so it
must never import torch, gymnasium, mani_skill, cv2, or any VLM client.

Path convention: new scripts resolve their roots in this order
  1. explicit CLI argument;
  2. environment variables REWARD_DATASET_ROOT / REWARD_SPLIT_DIR;
  3. the historical Windows defaults (documented, may not exist elsewhere).
"""

from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import os
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "TASK_IDS",
    "SPLITS",
    "VALID_LABELS",
    "PREFERENCE_STATES",
    "FORBIDDEN_FEATURE_EXACT",
    "FORBIDDEN_FEATURE_PATTERNS",
    "FORBIDDEN_LABEL_COLUMNS",
    "default_dataset_root",
    "default_split_dir",
    "load_csv",
    "write_csv",
    "write_json",
    "parse_numeric",
    "as_float",
    "parse_indices",
    "base_success_id",
    "clip_trajectory_id",
    "pair_group_id",
    "build_group_split_map",
    "trajectory_split_map",
    "check_forbidden_feature",
    "assert_features_allowed",
    "scan_forbidden_columns",
    "sha256_file",
    "PavCalibrator",
    "binned_reliability",
    "confidence_bucket_weight",
    "inspect_torch_checkpoint",
    "linspace_resample",
    "leakage_report",
]

TASK_IDS = ("peginsertion", "stackcube", "stackpyramid")
SPLITS = ("train", "val", "test")
VALID_LABELS = ("A>B", "B>A")
PREFERENCE_STATES = ("A>B", "B>A", "unsure")

_WINDOWS_DATASET_ROOT = r"D:\Users\User\Desktop\reward_model_dataset"
_WINDOWS_SPLIT_DIR = (
    r"D:\Users\User\Desktop\reward_model_dataset\dataset_splits"
    r"\bootstrap_v1_fusion_stereo_v1_clean"
)


def default_dataset_root() -> Path:
    return Path(os.environ.get("REWARD_DATASET_ROOT", _WINDOWS_DATASET_ROOT))


def default_split_dir() -> Path:
    return Path(os.environ.get("REWARD_SPLIT_DIR", _WINDOWS_SPLIT_DIR))


# ---------------------------------------------------------------------------
# CSV / JSON IO (schema-stable variants of the helpers used in scripts 10-25)
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write rows with a fixed column order.

    Unlike the v0 helper, callers can pin ``fieldnames`` so schemas do not
    depend on dict insertion order across runs. Unknown keys raise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_numeric(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return 1.0
    if lowered in {"false", "no"}:
        return 0.0
    try:
        parsed = float(text)
        return parsed if math.isfinite(parsed) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def as_float(value: Any, default: float = 0.0) -> float:
    parsed = parse_numeric(value)
    return parsed if math.isfinite(parsed) else default


def parse_indices(value: str) -> list[int]:
    return [int(float(item)) for item in str(value).replace(",", ";").split(";") if item.strip()]


def linspace_resample(indices: list[int], length: int) -> list[int]:
    """Uniform resampling identical to the runtime policy (linspace + round)."""
    if not indices:
        raise ValueError("indices must be non-empty")
    if length < 1:
        raise ValueError("length must be at least 1")
    if length == 1:
        return [indices[-1]]
    if len(indices) <= length:
        return list(indices)
    positions = [
        round(i * (len(indices) - 1) / (length - 1)) for i in range(length)
    ]
    return [indices[position] for position in positions]


# ---------------------------------------------------------------------------
# Identifiers and split maps (mirrors 17_build_dataset_splits.py definitions)
# ---------------------------------------------------------------------------

def clip_trajectory_id(clip_id: str) -> str:
    return re.sub(r"-C\d+$", "", str(clip_id))


def base_success_id(clip_id_or_trajectory: str) -> str:
    """Return the canonical source trajectory used by script 17.

    This deliberately mirrors ``17_build_dataset_splits.py``. Derived
    OFFSET/TRUNC/near-miss trajectory names that begin with a normal success
    id must remain in the same source group as that success trajectory.
    """
    text = clip_trajectory_id(clip_id_or_trajectory)
    match = re.match(r"^(SC|SP|PEG)-SUCC-\d+", text)
    return match.group(0) if match else text


def pair_group_id(row: dict[str, str]) -> str:
    groups = sorted({base_success_id(row["clip_a_id"]), base_success_id(row["clip_b_id"])})
    return f"{row['task_id']}::" + "+".join(groups)


def build_group_split_map(split_dir: Path) -> dict[str, str]:
    """source_group_id -> v1 split, read from the split tables themselves.

    A group appearing in more than one split file is a leakage error and
    raises immediately.
    """
    mapping: dict[str, str] = {}
    for split in SPLITS:
        for row in load_csv(Path(split_dir) / f"{split}_pairs.csv"):
            group = row.get("source_group_id") or pair_group_id(row)
            existing = mapping.get(group)
            if existing is not None and existing != split:
                raise ValueError(
                    f"source group {group!r} appears in both {existing} and {split}"
                )
            mapping[group] = split
    return mapping


def trajectory_split_map(split_dir: Path) -> dict[tuple[str, str], str]:
    """(task_id, trajectory_id) -> v1 split derived from the pair tables.

    This is the corrected replacement for the legacy per-frame ``split``
    column (audit finding F2). Every trajectory referenced by any clip in a
    split file is assigned that file's split; conflicts raise.
    """
    mapping: dict[tuple[str, str], str] = {}
    for split in SPLITS:
        for row in load_csv(Path(split_dir) / f"{split}_pairs.csv"):
            for side in ("a", "b"):
                key = (row["task_id"], clip_trajectory_id(row[f"clip_{side}_id"]))
                existing = mapping.get(key)
                if existing is not None and existing != split:
                    raise ValueError(
                        f"trajectory {key} appears in both {existing} and {split}"
                    )
                mapping[key] = split
    return mapping


def leakage_report(rows: Iterable[dict[str, str]], split_key: str = "split_v1") -> dict[str, Any]:
    """Independent recomputation of the 17_build leakage check."""
    group_to_splits: dict[str, set[str]] = collections.defaultdict(set)
    base_to_splits: dict[str, set[str]] = collections.defaultdict(set)
    n = 0
    for row in rows:
        n += 1
        split = row.get(split_key, "")
        group_to_splits[row.get("source_group_id") or pair_group_id(row)].add(split)
        for clip_key in ("clip_a_id", "clip_b_id"):
            base_to_splits[f"{row['task_id']}::{base_success_id(row[clip_key])}"].add(split)
    leaking_groups = {k: sorted(v) for k, v in group_to_splits.items() if len(v) > 1}
    leaking_bases = {k: sorted(v) for k, v in base_to_splits.items() if len(v) > 1}
    return {
        "rows": n,
        "source_group_leakage_count": len(leaking_groups),
        "base_success_id_leakage_count": len(leaking_bases),
        "source_group_leakage": leaking_groups,
        "base_success_id_leakage": leaking_bases,
    }


# ---------------------------------------------------------------------------
# Privileged-feature deny list (audit findings F4/F5/F6)
# ---------------------------------------------------------------------------

# Exact names that must never be model/runtime inputs.
FORBIDDEN_FEATURE_EXACT = frozenset(
    {
        "env_success",
        "success",
        "observed_success",
        "expected_success",
        "is_success",
        "task_success",
        "object_pose",
        "goal_pose",
        "object_position",
        "goal_position",
        "tcp_pose",
        "privileged_state",
        "frame_idx",
        "frame_index",
        "time",
        "timestep",
        "step",
        "center_time_proxy",
        "progress_rank_terminal",
        "stage_id",
        "stage_name",
        "stage_score",
        # v0 hand-designed rule outputs (21_train PHYSICAL_FEATURES)
        "stereo_end_score_proxy",
        "stereo_end_dist_m",
        "stereo_end_depth_error_m",
        "contact_end_stage_id",
        "contact_grasp_ratio",
        "contact_support_contact_ratio",
    }
)

# Substring patterns (matched case-insensitively) for whole families.
FORBIDDEN_FEATURE_PATTERNS = (
    r"env_?success",
    r"\bevaluate\b",
    r"score_proxy",
    r"label_proxy",
    r"_pose\b",
    r"^pose_",
    r"fusion_label",
    r"final_preference",
    r"candidate_label",
    r"mimo_",
    r"preference_label",
    r"supervision_bucket",
    r"progress_rank",
    r"\btime_progress\b",
)

# Columns that are labels/teacher outputs: allowed as *offline supervision or
# pass-through metadata*, forbidden as model input features anywhere.
FORBIDDEN_LABEL_COLUMNS = frozenset(
    {
        "candidate_label",
        "candidate_confidence",
        "mimo_preference",
        "mimo_confidence",
        "final_preference_label_v0",
        "final_opposite_label_v0",
        "preference_label_hint_v0",
        "fusion_label_v1",
        "fusion_bucket_v0",
        "stereo_geometry_label_proxy",
        "contact_stage_label_proxy",
    }
)


def check_forbidden_feature(name: str) -> str | None:
    """Return the reason a feature name is forbidden, or None if allowed."""
    lowered = str(name).strip().lower()
    if lowered in FORBIDDEN_FEATURE_EXACT:
        return f"exact deny-list match: {lowered}"
    for pattern in FORBIDDEN_FEATURE_PATTERNS:
        if re.search(pattern, lowered):
            return f"pattern deny-list match: {pattern}"
    return None


def assert_features_allowed(names: Iterable[str], context: str) -> None:
    violations = {name: check_forbidden_feature(name) for name in names}
    violations = {k: v for k, v in violations.items() if v}
    if violations:
        details = "; ".join(f"{k} ({v})" for k, v in sorted(violations.items()))
        raise ValueError(f"forbidden model-input feature(s) in {context}: {details}")


def scan_forbidden_columns(names: Iterable[str]) -> dict[str, str]:
    """Non-raising variant for audits/reports."""
    result: dict[str, str] = {}
    for name in names:
        reason = check_forbidden_feature(name)
        if reason:
            result[name] = reason
    return result


# ---------------------------------------------------------------------------
# Hashing / provenance
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            block = file.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Calibration (pure numpy; no sklearn dependency)
# ---------------------------------------------------------------------------

class PavCalibrator:
    """Isotonic (pool-adjacent-violators) mapping score -> P(correct).

    Fit on (score, correct) pairs from the TRAIN split only. Prediction is a
    monotone step function with linear interpolation between block centers and
    clamping at the ends. Deliberately tiny and dependency-free so it can run
    and be unit-tested in environments without sklearn.
    """

    def __init__(self) -> None:
        self.thresholds_: list[float] = []
        self.values_: list[float] = []
        self.n_fit_: int = 0

    def fit(self, scores: Iterable[float], correct: Iterable[float]) -> "PavCalibrator":
        pairs = sorted(
            (float(s), float(c))
            for s, c in zip(scores, correct)
            if math.isfinite(float(s))
        )
        if not pairs:
            raise ValueError("PavCalibrator.fit received no finite scores")
        self.n_fit_ = len(pairs)
        # Equal scores must be aggregated before PAV. Otherwise tuple sorting
        # can order tied negatives before positives and create duplicate
        # thresholds whose prediction depends on row ordering.
        tied: list[list[float]] = []
        for score, label in pairs:
            if tied and score == tied[-1][2]:
                tied[-1][0] += label
                tied[-1][1] += 1.0
            else:
                tied.append([label, 1.0, score, score])

        # blocks: [sum_y, count, min_score, max_score]
        blocks: list[list[float]] = []
        for block in tied:
            blocks.append(block)
            while len(blocks) >= 2 and (
                blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
            ):
                y2, n2, lo2, hi2 = blocks.pop()
                blocks[-1][0] += y2
                blocks[-1][1] += n2
                blocks[-1][3] = hi2
        self.thresholds_ = [0.5 * (b[2] + b[3]) for b in blocks]
        self.values_ = [b[0] / b[1] for b in blocks]
        return self

    def predict(self, scores: Iterable[float]) -> list[float]:
        if not self.thresholds_:
            raise RuntimeError("PavCalibrator used before fit")
        out: list[float] = []
        ts, vs = self.thresholds_, self.values_
        for raw in scores:
            score = float(raw)
            if not math.isfinite(score):
                out.append(float("nan"))
                continue
            if score <= ts[0]:
                out.append(vs[0])
                continue
            if score >= ts[-1]:
                out.append(vs[-1])
                continue
            hi = 1
            while ts[hi] < score:
                hi += 1
            lo = hi - 1
            span = ts[hi] - ts[lo]
            frac = 0.0 if span <= 0 else (score - ts[lo]) / span
            out.append(vs[lo] + frac * (vs[hi] - vs[lo]))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds_,
            "values": self.values_,
            "n_fit": self.n_fit_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PavCalibrator":
        calibrator = cls()
        calibrator.thresholds_ = [float(x) for x in payload["thresholds"]]
        calibrator.values_ = [float(x) for x in payload["values"]]
        calibrator.n_fit_ = int(payload.get("n_fit", 0))
        return calibrator


def binned_reliability(
    keys: Iterable[str],
    correct: Iterable[float],
    smoothing: float = 1.0,
) -> dict[str, float]:
    """Laplace-smoothed P(correct | categorical bucket), e.g. mimo high/medium/low."""
    hits: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, float] = collections.defaultdict(float)
    for key, value in zip(keys, correct):
        key = str(key).strip().lower()
        counts[key] += 1.0
        hits[key] += float(value)
    return {
        key: (hits[key] + smoothing) / (counts[key] + 2.0 * smoothing)
        for key in counts
    }


def confidence_bucket_weight(text: str) -> float:
    """Legacy high/medium/low mapping kept for parity with 24_train."""
    mapping = {"high": 1.0, "medium": 0.7, "low": 0.4}
    key = str(text).strip().lower()
    if key in mapping:
        return mapping[key]
    return max(as_float(text, 1.0), 1e-4)


# ---------------------------------------------------------------------------
# Torch-free checkpoint schema inspection (audit finding F17)
# ---------------------------------------------------------------------------

class _Stub:
    __slots__ = ("name", "args")

    def __init__(self, name: str, args: tuple | None = None) -> None:
        self.name = name
        self.args = args

    def __call__(self, *args: Any, **_: Any) -> "_Stub":
        return _Stub(self.name, args)

    def __setitem__(self, key: Any, value: Any) -> None:  # BUILD/SETITEMS on stubs
        return None

    def __setstate__(self, state: Any) -> None:
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<stub {self.name}>"


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that never resolves real classes except OrderedDict.

    Every other global becomes an inert stub whose call records its args, so
    tensor shapes can be read from ``_rebuild_tensor_v2`` argument tuples
    without importing torch or executing arbitrary constructors.
    """

    _SAFE = {("collections", "OrderedDict"): collections.OrderedDict}

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self._SAFE:
            return self._SAFE[(module, name)]
        return _Stub(f"{module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        try:
            return _Stub("storage", (str(pid[1]), pid[2], pid[4]))
        except Exception:  # noqa: BLE001 - malformed pid still yields a stub
            return _Stub("storage")


def inspect_torch_checkpoint(path: Path) -> dict[str, Any]:
    """Read a torch zip checkpoint's schema without torch and without
    executing pickled code. Returns top-level scalar metadata plus tensor
    names/shapes and the parameter count implied by the shapes."""
    with zipfile.ZipFile(Path(path)) as archive:
        candidates = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if not candidates:
            raise ValueError(f"{path} is not a torch zip checkpoint (no data.pkl)")
        payload = _SafeUnpickler(io.BytesIO(archive.read(candidates[0]))).load()
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level checkpoint object is not a dict")

    def plain(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        return repr(value)

    tensors: dict[str, list[int] | None] = {}
    param_count = 0
    state_dict = payload.get("state_dict")
    if isinstance(state_dict, dict):
        for key, value in state_dict.items():
            shape = None
            if isinstance(value, _Stub) and value.args and len(value.args) >= 3:
                maybe_shape = value.args[2]
                if isinstance(maybe_shape, tuple):
                    shape = [int(d) for d in maybe_shape]
            tensors[str(key)] = shape
            if shape:
                block = 1
                for dim in shape:
                    block *= dim
                param_count += block
    metadata = {
        key: plain(value)
        for key, value in payload.items()
        if key != "state_dict"
    }
    return {
        "metadata": metadata,
        "tensor_shapes": tensors,
        "tensor_count": len(tensors),
        "parameter_count_from_shapes": param_count,
    }
