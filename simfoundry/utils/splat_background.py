# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual background swap: a Gaussian-splat USDZ takes the place of ``mesh_background_0`` in a scene JSON.

The swap is a pure function of the scene dict, so it is tested without Isaac Sim. It changes only what the spec names:
the removed objects leave both ``init_info`` and ``object_registry``, and ``gs_background`` is appended to both, together
with the counter-patch boxes when the spec has any. Every other entry stays as it was, because the other readers of the
same scene file (policy evaluation, teleop, task proposal, scene sampling) must see the same objects.

The pose formula is the one ``photo_metric/render_photo_view.py`` uses to place the splat in the OmniGibson renders
that were checked against the photos; the entry format follows ``7_build_og_scene_assets.py``. Only numpy and the
standard library are imported here, so the module loads in the light editor env and in every env that runs its tests.

The optional colour matrix (``ccm``) is a second, run-time change: ``apply_splat_ccm`` writes it to the splat's own prims
once they exist. A spec without one changes nothing, in the scene or in the record.

The optional ``counter_patch`` adds opaque, visual-only boxes just below the splat's countertop, one per rectangle. They
follow the splat in both sections, so where the splat draws nothing they show instead of the sky. A spec without one
changes nothing either: no entry, and no key in the record.

A spec is checked the same way whether it comes from a file or is built in code (``_checked_spec``), and the swap runs
those checks before it reads or changes the scene.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
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
    {"schema", "name", "usdz", "pose_json", "matrix", "dz_m", "floor_plane_visible", "remove", "kit_settings", "color",
     "ccm", "counter_patch"}
)
_DEFAULT_REMOVE = ("mesh_background_0",)
_CCM_LIMIT = 4.0  # a larger entry is more likely a typo than a correction, so it is refused
CCM_ATTRIBUTES = ("omni:nurec:ccmR", "omni:nurec:ccmG", "omni:nurec:ccmB")  # the rows of the splat's colour matrix
PATCH_NAME_PREFIX = "counter_patch_"  # a box is named by this prefix and its rectangle's index
_PATCH_KEYS = frozenset({"rects", "z_top_m", "thickness_m", "rgb"})
_PATCH_MAX_RECTS = 8
_PATCH_MAX_ABS_M = 5.0  # a corner further out is more likely a typo than a patch, so it is refused
_PATCH_MAX_THICKNESS_M = 0.05
MIN_RECT_SIDE_M = 1e-3  # a box side under 1 mm is a degenerate sliver, not a patch, so it is refused
MIN_THICKNESS_M = 1e-4  # a box thinner than 0.1 mm is refused for the same reason
_LENGTH_DECIMALS = 12  # a side is rounded to a picometre before the minimum test, so that 0.401 - 0.4 is exactly 1 mm


class SplatBackgroundError(ValueError):
    """A spec or a scene that cannot take the swap. It is raised before anything is changed."""


@dataclasses.dataclass(frozen=True)
class CounterPatch:
    """A checked counter patch: one opaque box per rectangle, just under the splat's countertop.

    ``rects`` are ``(xmin, xmax, ymin, ymax)`` in the scene frame. Each box's top face is at ``z_top_m`` and it is
    ``thickness_m`` thick. ``rgb`` is its colour, each channel in [0, 1]. The boxes are visual only and add no collision.
    """

    rects: tuple[tuple[float, float, float, float], ...]
    z_top_m: float
    thickness_m: float
    rgb: tuple[float, float, float]


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
    ccm: tuple[tuple[float, ...], ...] | None = None  # the splat's 3 x 4 colour matrix; None changes nothing at run time
    counter_patch: CounterPatch | None = None  # None adds no boxes to the scene


def _checked_spec(spec: SplatBackgroundSpec) -> SplatBackgroundSpec:
    """Every field check of a spec, in one place, so a spec from a file and a spec built in code meet the same tests.

    The result is the spec with plain values: a float ``dz_m``, a tuple ``remove``, a dict ``kit_settings``, the checked
    matrix, the colour matrix as tuples, and a ``CounterPatch`` or ``None``. Every refusal is a ``SplatBackgroundError``.
    ``swap_visual_background`` calls this before it reads or changes the scene.
    """
    name = spec.name
    if not isinstance(name, str) or not name:
        raise SplatBackgroundError("the spec needs a non-empty string \"name\"")

    if not isinstance(spec.usdz, (str, os.PathLike)) or not str(spec.usdz):
        raise SplatBackgroundError("the spec needs a \"usdz\" path")
    usdz = Path(spec.usdz)
    if usdz.suffix != ".usdz":
        raise SplatBackgroundError(f"usdz must end in .usdz: {str(spec.usdz)!r}")
    if not usdz.is_file():
        raise SplatBackgroundError(f"usdz not found: {usdz}")

    pose_json = None if spec.pose_json is None else Path(spec.pose_json)
    if pose_json is not None and not pose_json.is_file():
        raise SplatBackgroundError(f"pose_json not found: {pose_json}")

    matrix = _checked_matrix(spec.matrix, "the spec's \"matrix\"")

    dz = spec.dz_m
    problem = _number_problem(dz, "dz_m")
    if problem is not None:
        raise SplatBackgroundError(problem)
    if not abs(dz) <= MAX_ABS_DZ_M:  # compared before float(), so a huge int is refused here as well
        raise SplatBackgroundError(f"dz_m is out of range: |dz_m| must be <= {MAX_ABS_DZ_M} m, got {dz!r}")

    floor_visible = spec.floor_plane_visible
    if not isinstance(floor_visible, bool):
        raise SplatBackgroundError(f"floor_plane_visible must be true or false, got {floor_visible!r}")

    remove = spec.remove
    if not isinstance(remove, (list, tuple)) or not all(isinstance(item, str) for item in remove):
        raise SplatBackgroundError("\"remove\" must be a list of object names")
    if any(item == "" for item in remove):
        raise SplatBackgroundError("\"remove\" has an empty name")
    if len(set(remove)) != len(remove):
        raise SplatBackgroundError(f"\"remove\" repeats a name: {list(remove)}")
    if GS_NAME in remove:
        raise SplatBackgroundError(f"\"remove\" must not name {GS_NAME!r}: the spec adds it")

    kit = spec.kit_settings
    if not isinstance(kit, dict):
        raise SplatBackgroundError("\"kit_settings\" must be an object of Kit setting paths and values")
    for key, value in kit.items():
        if not isinstance(key, str) or not key.startswith("/"):
            raise SplatBackgroundError(f"kit_settings key {key!r} must be a settings path starting with '/'")
        if isinstance(value, float) and not math.isfinite(value):
            raise SplatBackgroundError(f"kit_settings {key!r} is NaN or infinite, got {value!r}")
        if not isinstance(value, (bool, int, float, str)):
            raise SplatBackgroundError(
                f"kit_settings {key!r} must be a bool, number or string, not {type(value).__name__}")

    if spec.color is not None:
        raise SplatBackgroundError("\"color\" is not implemented yet: the spec must say null")

    ccm = _checked_ccm(spec.ccm, '"ccm"')
    patch = _checked_counter_patch(spec.counter_patch, '"counter_patch"')

    return dataclasses.replace(
        spec, name=name, usdz=usdz, matrix=matrix, pose_json=pose_json, dz_m=float(dz),
        floor_plane_visible=floor_visible, remove=tuple(remove), kit_settings=dict(kit), color=None, ccm=ccm,
        counter_patch=patch)


def load_visual_background(path) -> SplatBackgroundSpec:
    """Reads and checks a spec file. Every refusal is a ``SplatBackgroundError``.

    The file-level checks are here: the JSON, the schema, the unknown keys, the pose file, and the rule that gives exactly
    one matrix source. The field checks are in ``_checked_spec``, which the swap runs too.

    Relative paths in the spec (``usdz``, ``pose_json``) are taken from the folder of the spec file, so a spec moves
    together with its splat. ``~`` is not expanded. Unknown keys are refused: a misspelt key such as ``dz`` would
    otherwise be ignored and its default kept, and the swap would run cleanly with the splat in the wrong place.
    Paths are kept as spelled, symlinks included (see ``_resolve``).
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

    usdz_text = raw.get("usdz")
    if not isinstance(usdz_text, str) or not usdz_text:
        raise SplatBackgroundError("the spec needs a \"usdz\" path")

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

    return _checked_spec(SplatBackgroundSpec(
        name=raw.get("name"),
        usdz=_resolve(spec_dir, usdz_text),
        matrix=matrix,
        pose_json=pose_json,
        dz_m=raw.get("dz_m", 0.0),
        floor_plane_visible=raw.get("floor_plane_visible", False),
        remove=raw.get("remove", list(_DEFAULT_REMOVE)),
        kit_settings=raw.get("kit_settings", {}),
        color=raw.get("color"),
        ccm=raw.get("ccm"),
        counter_patch=raw.get("counter_patch"),
    ))


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


def counter_patch_entry(index: int, rect, thickness_m: float, rgb) -> dict:
    """The ``init_info`` entry of one counter-patch box: a Cube, scaled to the rectangle.

    OmniGibson's default Cube is 1 m on a side (``create_primitive_mesh`` with extents 1.0, omnigibson/utils/usd_utils.py
    1668-1710), so ``scale`` is the box's size in metres: x and y from the rectangle, z the thickness. ``size`` is not
    passed. With a custom ``scale`` OmniGibson ignores ``size`` (omnigibson/objects/primitive_object.py 150-152) and logs
    one warning for each box, which is expected. The box is visual only and fixed, so it adds nothing to collision or
    to the dynamics.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in rect)
    r, g, b = (float(v) for v in rgb)
    return {
        "class_module": "omnigibson.objects.primitive_object",
        "class_name": "PrimitiveObject",
        "args": {
            "name": f"{PATCH_NAME_PREFIX}{index}",
            "primitive_type": "Cube",
            "scale": [xmax - xmin, ymax - ymin, float(thickness_m)],
            "rgba": [r, g, b, 1.0],
            "visual_only": True,
            "fixed_base": True,
        },
    }


def counter_patch_state(rect, z_top_m: float, thickness_m: float) -> dict:
    """The ``object_registry`` state of one counter-patch box: centred on its rectangle, its top face at ``z_top_m``.

    The three keys beyond a plain splat state are the ones OmniGibson's ``PrimitiveObject`` needs for a Cube. Restoring a
    scene's initial state calls ``_load_state``, which reads ``state["size"]`` for a Cube (primitive_object.py 369-377, with
    VALID_SIZE_OBJECTS on line 20); a missing key is a KeyError. ``_dump_state`` (361-367) writes -1 for the radius and the
    height of a Cube, and the size, which is 1.0. The size setter with 1.0 leaves the geometry as it is (312-343): its scale
    factor is 1.0, because the box's size is its prim scale, not its extent.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in rect)
    pos = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, float(z_top_m) - float(thickness_m) / 2.0)
    state = gs_background_state(pos, [0.0, 0.0, 0.0, 1.0])
    state.update({"radius": -1, "height": -1, "size": 1.0})  # appended after the three keys of a plain splat state
    return state


def _taken_message(name: str, init_info: dict, registry: dict) -> str:
    """The refusal for a name the scene already holds: it names the name and the section or sections that hold it."""
    holders = [label for label, section in (("init_info", init_info), ("object_registry", registry)) if name in section]
    return f"the scene already has a {name!r} object in {' and '.join(holders)}"


def _missing_message(name: str, in_init: bool, in_registry: bool) -> str:
    """The refusal for a removed name that is not in both sections: it says which section lacks it."""
    if not in_init and not in_registry:
        return f"{name!r} is in neither init_info nor object_registry of the scene, so the spec cannot remove it"
    present, absent = ("init_info", "object_registry") if in_init else ("object_registry", "init_info")
    return f"{name!r} is in {present} but not in {absent} of the scene; a name must be in both to be removed"


def swap_visual_background(scene: dict, spec: SplatBackgroundSpec) -> dict:
    """Puts the splat in place of the spec's ``remove`` objects, in ``scene`` (changed in place).

    The spec gets ``_checked_spec`` first, the same checks a loaded spec gets, before anything in the scene is read. The
    spec's counter-patch boxes, if it has any, follow the splat in both sections. Every check runs before the first
    change, so a refused swap leaves ``scene`` as it was. Each removed name must be in both ``init_info`` and
    ``object_registry``. A name in only one of them would leave the scene half changed, so the swap is refused instead.

    Returns:
        A record of the change that can go into ``setup.json`` as it is (plain strings, numbers, lists and dicts).
    """
    spec = _checked_spec(spec)
    try:
        init_info = scene["objects_info"]["init_info"]
        registry = scene["state"]["registry"]["object_registry"]
    except (KeyError, TypeError) as error:
        raise SplatBackgroundError("the scene needs objects_info.init_info and state.registry.object_registry") from error
    if not isinstance(init_info, dict) or not isinstance(registry, dict):
        raise SplatBackgroundError("init_info and object_registry must be objects")
    if GS_NAME in init_info or GS_NAME in registry:
        raise SplatBackgroundError(_taken_message(GS_NAME, init_info, registry))
    for name in spec.remove:
        in_init, in_registry = name in init_info, name in registry
        if not (in_init and in_registry):
            raise SplatBackgroundError(_missing_message(name, in_init, in_registry))
    patch = spec.counter_patch
    patch_names = [f"{PATCH_NAME_PREFIX}{index}" for index in range(len(patch.rects))] if patch is not None else []
    for name in patch_names:
        if name in init_info or name in registry:
            raise SplatBackgroundError(_taken_message(name, init_info, registry))
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
    if patch is not None:  # the boxes come after the splat, in rectangle order, in both sections
        for index, rect in enumerate(patch.rects):
            init_info[patch_names[index]] = counter_patch_entry(index, rect, patch.thickness_m, patch.rgb)
            registry[patch_names[index]] = counter_patch_state(rect, patch.z_top_m, patch.thickness_m)
    record = {
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
    if spec.ccm is not None:  # no key at all without a matrix, so the record of a plain swap is the same as before
        record["ccm"] = [list(row) for row in spec.ccm]
    if patch is not None:  # no key at all without boxes, for the same reason
        record["counter_patch"] = {
            "names": patch_names,
            "rects": [list(rect) for rect in patch.rects],
            "z_top_m": patch.z_top_m,
            "thickness_m": patch.thickness_m,
            "rgb": list(patch.rgb),
        }
    return record


def apply_splat_ccm(root_prim, rows, make_vec=None) -> list[dict]:
    """Writes the splat's 3 x 4 colour matrix to the prims under ``root_prim`` that carry it, and returns what was written.

    ``rows[0]``, ``rows[1]`` and ``rows[2]`` go to ``omni:nurec:ccmR``, ``omni:nurec:ccmG`` and ``omni:nurec:ccmB``, each
    through ``make_vec(*row)``. The default ``make_vec`` makes a ``pxr.Gf.Vec4f``, and pxr is imported only there. Every
    check runs before the first write: a prim with some of the three attributes is refused, and so is a splat with none,
    because a matrix written in part, or not at all, would look applied when it is not.

    Returns:
        One record per attribute written: the prim's path, the attribute's name, and the value read back, as text.
    """
    if len(rows) != 3 or any(len(row) != 4 for row in rows):
        raise SplatBackgroundError("apply_splat_ccm needs 3 rows of 4 numbers")
    targets = []
    for prim in _prims_under(root_prim):
        attrs = [(name, prim.GetAttribute(name)) for name in CCM_ATTRIBUTES]
        present = [(name, attr) for name, attr in attrs if attr is not None and attr.IsValid()]
        if not present:
            continue
        if len(present) != len(CCM_ATTRIBUTES):
            absent = [name for name, attr in attrs if attr is None or not attr.IsValid()]
            raise SplatBackgroundError(f"{prim.GetPath()} carries part of the colour matrix; missing {absent}")
        targets.append((prim, present))
    if not targets:
        raise SplatBackgroundError(f"no prim under the splat carries {list(CCM_ATTRIBUTES)}: nothing would be written")
    make = _vec4f if make_vec is None else make_vec
    written = []
    for prim, present in targets:
        for (name, attr), row in zip(present, rows):
            attr.Set(make(*row))
            written.append({"prim": str(prim.GetPath()), "attr": name, "read_back": str(attr.Get())})
    return written


def _prims_under(prim):
    """``prim`` and every prim below it, each before its children, in the order the children are listed."""
    yield prim
    for child in prim.GetChildren():
        yield from _prims_under(child)


def _vec4f(*values):
    from pxr import Gf  # imported here, not at the top: the module is loaded by envs that have no pxr

    return Gf.Vec4f(*[float(v) for v in values])


def _resolve(spec_dir: Path, text: str) -> Path:
    """The path a spec gives, made absolute by joining a relative one to ``spec_dir``, and otherwise left as written.

    Nothing is resolved. A symlink keeps the name the spec gives, so the scene records that name. ``..`` is not collapsed
    as text either: after a symlink, ``link/..`` is the parent of the link's target, and collapsing it would name a file
    that the OS would not open.
    """
    candidate = Path(text)
    return candidate if candidate.is_absolute() else spec_dir / candidate


def _number_problem(value, where: str):
    """None when ``value`` is a finite real number that is not a bool; otherwise the sentence that says why it is not.

    The reasons are separate: a bool, a value that is not a number (a string, a null, a list), and NaN or an infinity. The
    range is the caller's check. A bool is named first, because ``True == 1`` would pass a plain number test.
    """
    if isinstance(value, bool):
        return f"{where} is a bool ({value!r}), not a number"
    if not isinstance(value, (int, float)):
        return f"{where} is not a number, got {value!r}"
    if isinstance(value, float) and not math.isfinite(value):  # an int is finite; math.isfinite overflows on a huge one
        return f"{where} is NaN or infinite, got {value!r}"
    return None


def _checked_matrix(value, where: str) -> np.ndarray:
    """A 4x4 similarity transform as float64, or a ``SplatBackgroundError`` that names the test it fails."""
    if isinstance(value, np.ndarray):  # a spec built in code holds an array; the checks below are on the list form
        value = value.tolist()
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise SplatBackgroundError(f"{where}: the matrix must be a list of 4 rows of 4 numbers")
    for i, row in enumerate(value):
        for j, entry in enumerate(row):
            problem = _number_problem(entry, f"{where}: matrix entry [{i}][{j}]")
            if problem is not None:
                raise SplatBackgroundError(problem)
            try:
                float(entry)
            except OverflowError as error:  # an int this large is a number, but no float holds it
                raise SplatBackgroundError(f"{where}: matrix entry [{i}][{j}] is out of range: too large for a float") from error
    m = np.array(value, dtype=np.float64)
    if np.max(np.abs(m[3] - np.array([0.0, 0.0, 0.0, 1.0]))) > _LAST_ROW_TOL:
        raise SplatBackgroundError(f"{where}: the last row must be [0, 0, 0, 1]")
    rot = m[:3, :3]
    det = float(np.linalg.det(rot))
    # a reflection has a negative determinant, and cbrt would then give a negative scale that looks like a flip
    if not det > 0.0:
        raise SplatBackgroundError(f"{where}: the rotation part must have a positive determinant (no reflection)")
    scale = float(np.cbrt(det))
    if not _SCALE_RANGE[0] <= scale <= _SCALE_RANGE[1]:
        raise SplatBackgroundError(
            f"{where}: the scale {scale:.6g} is out of range [{_SCALE_RANGE[0]}, {_SCALE_RANGE[1]}]")
    unit = rot / scale
    if np.max(np.abs(unit.T @ unit - np.eye(3))) > _ORTHO_TOL:
        raise SplatBackgroundError(f"{where}: the rotation part is not a similarity (A/s is not orthogonal)")
    return m


def _checked_ccm(value, where: str):
    """A 3 x 4 colour matrix as three tuples of floats, ``None`` for no matrix, or a refusal that names ``where``."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(
            isinstance(row, (list, tuple)) and len(row) == 4 for row in value):
        raise SplatBackgroundError(f"{where} must be null or 3 rows of 4 numbers")
    for i, row in enumerate(value):
        for j, entry in enumerate(row):
            problem = _number_problem(entry, f"{where} entry [{i}][{j}]")
            if problem is not None:
                raise SplatBackgroundError(problem)
            if not abs(entry) <= _CCM_LIMIT:  # compared before any float() call, so a huge int is refused here as well
                raise SplatBackgroundError(
                    f"{where} is out of range: each entry must satisfy |value| <= {_CCM_LIMIT}, got {entry!r}")
    return tuple(tuple(float(v) for v in row) for row in value)


def _checked_counter_patch(value, where: str):
    """A ``counter_patch`` as a ``CounterPatch`` of plain floats, ``None`` for no patch, or a refusal that names ``where``.

    ``value`` is the spec's object or a ``CounterPatch`` built in code, so a hand-built patch is checked like a loaded one.
    Each range test comes before any ``float()`` call, because a huge int overflows ``float``.
    """
    if value is None:
        return None
    if isinstance(value, CounterPatch):
        value = {"rects": value.rects, "z_top_m": value.z_top_m, "thickness_m": value.thickness_m, "rgb": value.rgb}
    if not isinstance(value, dict):
        raise SplatBackgroundError(
            f"{where} must be null or an object with \"rects\", \"z_top_m\", \"thickness_m\" and \"rgb\"")
    unknown = sorted(set(value) - _PATCH_KEYS)
    if unknown:
        raise SplatBackgroundError(f"{where} has unknown key(s): {unknown}")
    missing = sorted(_PATCH_KEYS - set(value))
    if missing:
        raise SplatBackgroundError(f"{where} is missing key(s): {missing}")

    rects = value["rects"]
    if not isinstance(rects, (list, tuple)):
        raise SplatBackgroundError(f"{where}: rects must be a list of rectangles, got {type(rects).__name__}")
    if not 1 <= len(rects) <= _PATCH_MAX_RECTS:
        raise SplatBackgroundError(f"{where}: rects has {len(rects)} rectangles; it needs 1 to {_PATCH_MAX_RECTS}")
    checked = []
    for index, rect in enumerate(rects):
        here = f"{where}: rects[{index}]"
        if not isinstance(rect, (list, tuple)):
            raise SplatBackgroundError(
                f"{here} must be a list of 4 numbers [xmin, xmax, ymin, ymax], got {type(rect).__name__}")
        if len(rect) != 4:
            raise SplatBackgroundError(f"{here} has {len(rect)} numbers; it needs 4 [xmin, xmax, ymin, ymax]")
        for entry in rect:
            problem = _number_problem(entry, here)
            if problem is not None:
                raise SplatBackgroundError(problem)
            if not abs(entry) <= _PATCH_MAX_ABS_M:  # before float(), so a huge int is refused here as well
                raise SplatBackgroundError(
                    f"{here} is out of range: each value must satisfy |value| <= {_PATCH_MAX_ABS_M} m, got {entry!r}")
        xmin, xmax, ymin, ymax = (float(v) for v in rect)
        if not (xmin < xmax and ymin < ymax):
            raise SplatBackgroundError(f"{here} needs xmin < xmax and ymin < ymax, got {list(rect)}")
        for axis, side in (("x", xmax - xmin), ("y", ymax - ymin)):
            if round(side, _LENGTH_DECIMALS) < MIN_RECT_SIDE_M:
                raise SplatBackgroundError(
                    f"{here} is too narrow in {axis}: the side is {side:g} m, and the minimum is "
                    f"{MIN_RECT_SIDE_M:g} m (1 mm)")
        checked.append((xmin, xmax, ymin, ymax))

    z_top = value["z_top_m"]
    problem = _number_problem(z_top, f"{where}: z_top_m")
    if problem is not None:
        raise SplatBackgroundError(problem)
    if not abs(z_top) <= MAX_ABS_DZ_M:
        raise SplatBackgroundError(
            f"{where}: z_top_m is out of range: |z_top_m| must be <= {MAX_ABS_DZ_M} m, got {z_top!r}")

    thickness = value["thickness_m"]
    problem = _number_problem(thickness, f"{where}: thickness_m")
    if problem is not None:
        raise SplatBackgroundError(problem)
    if not MIN_THICKNESS_M <= thickness <= _PATCH_MAX_THICKNESS_M:
        raise SplatBackgroundError(
            f"{where}: thickness_m is out of range: it must be between {MIN_THICKNESS_M:g} m and "
            f"{_PATCH_MAX_THICKNESS_M:g} m, got {thickness!r}")

    rgb = value["rgb"]
    if not isinstance(rgb, (list, tuple)):
        raise SplatBackgroundError(f"{where}: rgb must be a list of 3 numbers, got {type(rgb).__name__}")
    if len(rgb) != 3:
        raise SplatBackgroundError(f"{where}: rgb has {len(rgb)} numbers; it needs 3")
    for entry in rgb:
        problem = _number_problem(entry, f"{where}: rgb")
        if problem is not None:
            raise SplatBackgroundError(problem)
    if not all(0.0 <= entry <= 1.0 for entry in rgb):
        raise SplatBackgroundError(f"{where}: rgb is out of range: each channel must be in [0, 1], got {list(rgb)}")
    return CounterPatch(rects=tuple(checked), z_top_m=float(z_top), thickness_m=float(thickness),
                        rgb=tuple(float(v) for v in rgb))


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
