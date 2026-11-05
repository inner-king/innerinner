# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual background swap: a Gaussian-splat USDZ takes the place of ``mesh_background_0`` in a scene JSON.

The swap is a pure function of the scene dict, so it is tested without Isaac Sim. It changes only what the spec names:
the removed objects leave both ``init_info`` and ``object_registry``, and ``gs_background`` is appended to both. Every
other entry stays as it was, because the other readers of the same scene file (policy evaluation, teleop, task
proposal, scene sampling) must see the same objects.

The pose formula is the one ``photo_metric/render_photo_view.py`` uses to place the splat in the OmniGibson renders
that were checked against the photos; the entry format follows ``7_build_og_scene_assets.py``. Only numpy and the
standard library are imported here, so the module loads in the light editor env and in every env that runs its tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1
GS_NAME = "gs_background"  # the editor and the stage-1 material code look up this name, so it cannot be a parameter
MAX_ABS_DZ_M = 0.02  # a contact correction, not a relocation: a larger shift is a bad alignment, not a fix

_SCALE_RANGE = (0.1, 10.0)
_ORTHO_TOL = 1e-6
_LAST_ROW_TOL = 1e-9
_SPEC_KEYS = frozenset(
    {"schema", "name", "usdz", "pose_json", "matrix", "dz_m", "floor_plane_visible", "remove", "kit_settings", "color"}
)
_DEFAULT_REMOVE = ("mesh_background_0",)


class SplatBackgroundError(ValueError):
    """A spec or a scene that cannot take the swap. It is raised before anything is changed."""


@dataclasses.dataclass(frozen=True)
class SplatBackgroundSpec:
    """A checked spec. ``matrix`` maps the DA3 frame to the scene frame (``M_da3_to_scene``), as a 4x4 float64 array."""

    name: str
    usdz: Path
    matrix: np.ndarray
    pose_json: Path | None
    dz_m: float
    floor_plane_visible: bool
    remove: tuple[str, ...]
    kit_settings: dict[str, Any]
    color: dict | None


def load_visual_background(path) -> SplatBackgroundSpec:
    """Reads and checks a spec file. Every refusal is a ``SplatBackgroundError``.

    Relative paths in the spec (``usdz``, ``pose_json``) are taken from the folder of the spec file, so a spec moves
    together with its splat. ``~`` is not expanded. Unknown keys are refused: a misspelt key such as ``dz`` would
    otherwise be ignored and its default kept, and the swap would run cleanly with the splat in the wrong place.
    """
    spec_path = Path(path).absolute()
    spec_dir = spec_path.parent
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # ValueError covers JSONDecodeError and UnicodeDecodeError
        raise SplatBackgroundError(f"cannot read the spec {spec_path}: {error}") from error
    if not isinstance(raw, dict):
        raise SplatBackgroundError(f"the spec {spec_path} must be a JSON object")
    unknown = sorted(set(raw) - _SPEC_KEYS)
    if unknown:
        raise SplatBackgroundError(f"unknown key(s) in the spec: {unknown}")

    # bool is an int in Python, and True == 1 would pass a plain != test, so the type is checked too
    if "schema" not in raw or type(raw["schema"]) is not int or raw["schema"] != SCHEMA:
        raise SplatBackgroundError(f"the spec needs \"schema\": {SCHEMA}, got {raw.get('schema', '<missing>')!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise SplatBackgroundError("the spec needs a non-empty string \"name\"")

    usdz_text = raw.get("usdz")
    if not isinstance(usdz_text, str) or not usdz_text:
        raise SplatBackgroundError("the spec needs a \"usdz\" path")
    usdz = _resolve(spec_dir, usdz_text)
    if usdz.suffix != ".usdz":
        raise SplatBackgroundError(f"usdz must end in .usdz: {usdz_text!r}")
    if not usdz.is_file():
        raise SplatBackgroundError(f"usdz not found: {usdz}")

    # the matrix comes from exactly one place: a file that holds M_da3_to_scene, or the spec itself
    has_pose, has_matrix = "pose_json" in raw, "matrix" in raw
    if has_pose == has_matrix:
        raise SplatBackgroundError("give exactly one of \"pose_json\" and \"matrix\" in the spec")
    pose_json: Path | None = None
    if has_pose:
        pose_text = raw["pose_json"]
        if not isinstance(pose_text, str) or not pose_text:
            raise SplatBackgroundError("\"pose_json\" must be a path")
        pose_json = _resolve(spec_dir, pose_text)
        if not pose_json.is_file():
            raise SplatBackgroundError(f"pose_json not found: {pose_json}")
        try:
            pose_doc = json.loads(pose_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SplatBackgroundError(f"cannot read pose_json {pose_json}: {error}") from error
        if not isinstance(pose_doc, dict) or "M_da3_to_scene" not in pose_doc:
            raise SplatBackgroundError(f"{pose_json} has no \"M_da3_to_scene\" matrix")
        matrix = _checked_matrix(pose_doc["M_da3_to_scene"], str(pose_json))
    else:
        matrix = _checked_matrix(raw["matrix"], "the spec's \"matrix\"")

    dz_m = raw.get("dz_m", 0.0)
    if not _is_real(dz_m) or not math.isfinite(dz_m) or abs(dz_m) > MAX_ABS_DZ_M:
        raise SplatBackgroundError(f"dz_m must be a finite number with |dz_m| <= {MAX_ABS_DZ_M} m, got {dz_m!r}")

    floor_visible = raw.get("floor_plane_visible", False)
    if not isinstance(floor_visible, bool):
        raise SplatBackgroundError(f"floor_plane_visible must be true or false, got {floor_visible!r}")

    remove_raw = raw.get("remove", list(_DEFAULT_REMOVE))
    if not isinstance(remove_raw, list) or not all(isinstance(item, str) for item in remove_raw):
        raise SplatBackgroundError("\"remove\" must be a list of object names")
    if any(item == "" for item in remove_raw):
        raise SplatBackgroundError("\"remove\" has an empty name")
    if len(set(remove_raw)) != len(remove_raw):
        raise SplatBackgroundError(f"\"remove\" repeats a name: {remove_raw}")
    if GS_NAME in remove_raw:
        raise SplatBackgroundError(f"\"remove\" must not name {GS_NAME!r}: the spec adds it")

    kit_settings = raw.get("kit_settings", {})
    if not isinstance(kit_settings, dict):
        raise SplatBackgroundError("\"kit_settings\" must be an object of Kit setting paths and values")
    for key, value in kit_settings.items():
        if not key.startswith("/"):
            raise SplatBackgroundError(f"kit_settings key {key!r} must be a settings path starting with '/'")
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SplatBackgroundError(f"kit_settings {key!r} is not finite")
        elif not isinstance(value, (bool, int, str)):
            raise SplatBackgroundError(f"kit_settings {key!r} must be a bool, number or string, not {type(value).__name__}")

    if raw.get("color") is not None:
        raise SplatBackgroundError("\"color\" is not implemented yet: the spec must say null")

    return SplatBackgroundSpec(
        name=name,
        usdz=usdz,
        matrix=matrix,
        pose_json=pose_json,
        dz_m=float(dz_m),
        floor_plane_visible=floor_visible,
        remove=tuple(remove_raw),
        kit_settings=dict(kit_settings),
        color=None,
    )


def pose_from_similarity(matrix, dz_m: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Position, xyzw quaternion and scale of a similarity transform, with ``dz_m`` added in world z.

    The formula is the one in ``photo_metric/render_photo_view.py`` 178-187: ``s = cbrt(det R)``, ``R / s`` is the
    rotation, and the translation column is the position. ``dz_m`` is a contact correction on top of that. It moves
    only z, and the matrix itself is not changed.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise SplatBackgroundError(f"pose_from_similarity needs a 4x4 matrix, got shape {m.shape}")
    scale = float(np.cbrt(np.linalg.det(m[:3, :3])))
    rot = m[:3, :3] / scale
    pos = m[:3, 3].copy()
    pos[2] += float(dz_m)
    return pos, _quat_xyzw(rot), scale


def md5_of_file(path) -> str:
    """MD5 hex digest of a file, read in 1 MiB pieces (the USDZ is 58 MB).

    This is the digest OmniGibson's ``check_hash`` computes for ``expected_file_hash``, so the value written here is
    the one the loader compares against.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gs_background_entry(usdz, scale, md5) -> dict:
    """The ``init_info`` entry of the splat, in the form ``7_build_og_scene_assets.py`` writes.

    There is no ``category`` key, as in the stage-7 entry.
    """
    s = float(scale)
    return {
        "class_module": "omnigibson.objects.usd_object",
        "class_name": "USDObject",
        "args": {
            "name": GS_NAME,
            "usd_path": str(usdz),
            "scale": [s, s, s],
            "fixed_base": True,
            "visual_only": True,
            "expected_file_hash": str(md5),
        },
    }


def gs_background_state(pos, quat_xyzw) -> dict:
    """The ``object_registry`` state of the splat: a static root link at the given pose, with zero velocities.

    ``non_kin`` is written as an empty dict because the scene files that OmniGibson saves carry it on every object
    (the knife entry of ``tofu_og_cut.edit_scene`` does too). The velocities are zero because a saved state's live
    velocities would be replayed against this pose (AGENTS.md, rule 3).
    """
    if len(pos) != 3 or len(quat_xyzw) != 4:
        raise SplatBackgroundError(f"a splat state needs 3 position and 4 quaternion values, got {len(pos)} and {len(quat_xyzw)}")
    return {
        "is_asleep": True,
        "non_kin": {},
        "root_link": {
            "pos": [float(v) for v in pos],
            "ori": [float(v) for v in quat_xyzw],
            "lin_vel": [0.0, 0.0, 0.0],
            "ang_vel": [0.0, 0.0, 0.0],
        },
    }


def swap_visual_background(scene: dict, spec: SplatBackgroundSpec) -> dict:
    """Puts the splat in place of the spec's ``remove`` objects, in ``scene`` (changed in place).

    Every check runs before the first change, so a refused swap leaves ``scene`` as it was. Each removed name must be
    in both ``init_info`` and ``object_registry``. A name in only one of them would leave the scene half changed, so the
    swap is refused instead.

    Returns:
        A record of the change that can go into ``setup.json`` as it is (plain strings, numbers, lists and dicts).
    """
    try:
        init_info = scene["objects_info"]["init_info"]
        registry = scene["state"]["registry"]["object_registry"]
    except (KeyError, TypeError) as error:
        raise SplatBackgroundError("the scene needs objects_info.init_info and state.registry.object_registry") from error
    if not isinstance(init_info, dict) or not isinstance(registry, dict):
        raise SplatBackgroundError("init_info and object_registry must be objects")
    if len(set(spec.remove)) != len(spec.remove):
        raise SplatBackgroundError(f"remove repeats a name: {list(spec.remove)}")
    if GS_NAME in init_info or GS_NAME in registry:
        raise SplatBackgroundError(f"the scene already has a {GS_NAME!r} object")
    for name in spec.remove:
        if name not in init_info or name not in registry:
            raise SplatBackgroundError(f"{name!r} must be in both init_info and object_registry of the scene")
    _checked_matrix(spec.matrix, "the spec's matrix")  # a spec built by hand is checked as well as a loaded one
    try:
        md5 = md5_of_file(spec.usdz)
    except OSError as error:
        raise SplatBackgroundError(f"cannot read the splat {spec.usdz}: {error}") from error
    pos, quat, scale = pose_from_similarity(spec.matrix, spec.dz_m)
    entry = gs_background_entry(spec.usdz, scale, md5)
    state = gs_background_state(pos, quat)

    for name in spec.remove:
        del init_info[name]
        del registry[name]
    init_info[GS_NAME] = entry
    registry[GS_NAME] = state
    return {
        "name": spec.name,
        "usdz": str(spec.usdz),
        "md5": md5,
        "removed": list(spec.remove),
        "pos": [float(v) for v in pos],
        "ori_xyzw": [float(v) for v in quat],
        "scale": float(scale),
        "dz_m": float(spec.dz_m),
        "floor_plane_visible": bool(spec.floor_plane_visible),
        "kit_settings": dict(spec.kit_settings),
    }


def _resolve(spec_dir: Path, text: str) -> Path:
    """An absolute path: a relative ``text`` is taken from ``spec_dir``, and ``..`` is collapsed (symlinks are kept)."""
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = spec_dir / candidate
    return candidate.resolve()


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _all_numbers(value) -> bool:
    if isinstance(value, list):
        return all(_all_numbers(item) for item in value)
    return _is_real(value)


def _checked_matrix(value, where: str) -> np.ndarray:
    """A 4x4 similarity transform as float64, or a ``SplatBackgroundError`` naming the test it fails."""
    if isinstance(value, np.ndarray):  # a spec built in code holds an array; the checks below are on the list form
        value = value.tolist()
    if not isinstance(value, list) or not _all_numbers(value):
        raise SplatBackgroundError(f"{where}: the matrix must be a list of 4 rows of 4 numbers")
    try:
        m = np.array(value, dtype=np.float64)
    except ValueError as error:  # a ragged list has no array form
        raise SplatBackgroundError(f"{where}: the matrix must be 4 rows of 4 numbers ({error})") from error
    if m.shape != (4, 4):
        raise SplatBackgroundError(f"{where}: the matrix must be 4x4, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise SplatBackgroundError(f"{where}: the matrix has entries that are not finite")
    if np.max(np.abs(m[3] - np.array([0.0, 0.0, 0.0, 1.0]))) > _LAST_ROW_TOL:
        raise SplatBackgroundError(f"{where}: the last row must be [0, 0, 0, 1]")
    rot = m[:3, :3]
    det = float(np.linalg.det(rot))
    # a reflection has a negative determinant, and cbrt would then give a negative scale that looks like a flip
    if not det > 0.0:
        raise SplatBackgroundError(f"{where}: the rotation part must have a positive determinant (no reflection)")
    scale = float(np.cbrt(det))
    if not _SCALE_RANGE[0] <= scale <= _SCALE_RANGE[1]:
        raise SplatBackgroundError(f"{where}: the scale {scale:.6g} is outside [{_SCALE_RANGE[0]}, {_SCALE_RANGE[1]}]")
    unit = rot / scale
    if np.max(np.abs(unit.T @ unit - np.eye(3))) > _ORTHO_TOL:
        raise SplatBackgroundError(f"{where}: the rotation part is not a similarity (A/s is not orthogonal)")
    return m


def _quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """The xyzw quaternion of a rotation matrix, with the sign ``scipy.spatial.transform.Rotation.from_matrix`` gives.

    scipy is not imported here, so the module loads without it. The branch is the one scipy uses: the largest of the
    three diagonal entries and the trace picks which component is computed first, and the sign follows from it. The
    tests check the sign against scipy as well as the rotation.
    """
    m = np.asarray(rot, dtype=np.float64)
    decision = np.array([m[0, 0], m[1, 1], m[2, 2], np.trace(m)])
    choice = int(np.argmax(decision))
    q = np.empty(4)
    if choice != 3:
        i = choice
        j = (i + 1) % 3
        k = (j + 1) % 3
        q[i] = 1.0 - decision[3] + 2.0 * m[i, i]
        q[j] = m[j, i] + m[i, j]
        q[k] = m[k, i] + m[i, k]
        q[3] = m[k, j] - m[j, k]
    else:
        q[0] = m[2, 1] - m[1, 2]
        q[1] = m[0, 2] - m[2, 0]
        q[2] = m[1, 0] - m[0, 1]
        q[3] = 1.0 + decision[3]
    return q / np.linalg.norm(q)
