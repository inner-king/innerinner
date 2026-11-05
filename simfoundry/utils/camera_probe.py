# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only camera probe of the tofu policy runner: segmentation channels and camera poses on chosen ticks.

``--probe-out`` turns it on. It exists for the measurements of the splat-background test plan (what share of the
front and wrist frames each class takes, and where the cameras were), not as an input to any policy, so nothing here
runs when the option is off. The arrays come from the same ``get_obs()`` read that the runner already makes for the
rgb frames; the probe never reads a camera again.

Only numpy and the standard library are imported, so the module loads wherever the runner does.
"""

import json
from pathlib import Path

import numpy as np

DEFAULT_TICKS = (0, 25, 50, 75, 99)
CAMERA_ATTRIBUTES = {  # the key in the record -> the USD attribute of the camera prim
    "focal_length": "focalLength",
    "horizontal_aperture": "horizontalAperture",
    "vertical_aperture": "verticalAperture",
    "horizontal_aperture_offset": "horizontalApertureOffset",
    "vertical_aperture_offset": "verticalApertureOffset",
}


def _numpy(value):
    """An array of a tensor or of an array-like, keeping the dtype: segmentation ids must stay integers."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _names(table):
    """An id -> name table with string keys, as JSON needs it. A missing table is empty."""
    if not table:
        return {}
    return {str(key): str(name) for key, name in table.items()}


def probe_arrays(obs, info):
    """The probe's arrays and segmentation names, from one ``get_obs()`` read of one camera.

    Args:
        obs: the observation dict of that read: ``rgb`` (H x W x 3 or 4) and ``seg_instance`` are required, and
            ``seg_semantic`` is read when the read has it. A subset install has no BEHAVIOR category list, so its
            semantic channel cannot be built and the probe is run on ``seg_instance`` alone.
        info: the info dict of that read. Each segmentation channel has an id -> name (or label) table in it.

    Returns:
        ``({"rgb": uint8 (H, W, 3), "seg_instance": uint32 (H, W)[, "seg_semantic": uint32 (H, W)]},
        {"instance": {id: name}, "semantic": {id: label}})``. The ids are string keys. Without ``seg_semantic`` the
        arrays have no such key and the ``semantic`` table is empty.

    Raises:
        ValueError: ``rgb`` or ``seg_instance`` is missing, a segmentation channel whose shape is not the rgb's, or a
            negative id.
    """
    if "rgb" not in obs or "seg_instance" not in obs:
        raise ValueError(f"the probe needs rgb and seg_instance, the read has {sorted(obs)}")
    rgb = _numpy(obs["rgb"])[:, :, :3].astype(np.uint8)
    height, width = rgb.shape[:2]
    arrays = {"rgb": rgb}
    for key in ("seg_instance", "seg_semantic"):
        if key not in obs:  # seg_semantic is optional: a read without it has no semantic channel
            continue
        seg = _numpy(obs[key])
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[:, :, 0]
        if seg.shape != (height, width):
            raise ValueError(f"{key} has shape {seg.shape}, the rgb has {(height, width)}")
        if seg.size and seg.min() < 0:  # a negative id would wrap around in uint32
            raise ValueError(f"{key} has a negative id")
        arrays[key] = seg.astype(np.uint32)
    tables = info or {}
    semantic = _names(tables.get("seg_semantic")) if "seg_semantic" in arrays else {}
    seg_info = {"instance": _names(tables.get("seg_instance")), "semantic": semantic}
    return arrays, seg_info


def pose_record(position, orientation_xyzw):
    """A camera's world pose as plain lists: the position (3 numbers) and the xyzw quaternion (4 numbers)."""
    p = [float(v) for v in _numpy(position).reshape(-1)]
    q = [float(v) for v in _numpy(orientation_xyzw).reshape(-1)]
    if len(p) != 3 or len(q) != 4:
        raise ValueError(f"a pose needs 3 position and 4 orientation values, got {len(p)} and {len(q)}")
    return {"position": p, "orientation_xyzw": q}


def camera_attributes(prim):
    """The lens attributes of a camera prim as floats. An attribute the prim does not have is None.

    ``prim`` is anything with ``GetAttribute(name)`` whose result has ``IsValid()`` and ``Get()``, which a USD prim is.
    So the function runs without pxr, and a fake prim can stand in for it in the tests.
    """
    out = {}
    for key, name in CAMERA_ATTRIBUTES.items():
        attr = prim.GetAttribute(name) if prim is not None else None
        value = attr.Get() if attr is not None and attr.IsValid() else None
        out[key] = None if value is None else float(value)
    return out


def write_probe(out_dir, seed, tick, cam, arrays, pose, camera_info, seg_info):
    """Writes one camera's probe at one tick, and returns the two paths written.

    The files are ``DIR/ep{seed:05d}/t{tick:04d}_{cam}.npz``, with the arrays ``rgb`` (uint8) and ``seg_instance``
    (uint32), and ``seg_semantic`` (uint32) only when ``arrays`` has it; and the same stem with ``.json``: the pose, the
    image size (taken from the arrays), the lens attributes (null where the prim has none) and the segmentation names
    (``seg_info["semantic"]`` is empty without ``seg_semantic``).
    """
    folder = Path(out_dir) / f"ep{int(seed):05d}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"t{int(tick):04d}_{cam}"
    npz_path, json_path = folder / f"{stem}.npz", folder / f"{stem}.json"
    saved = {"rgb": arrays["rgb"], "seg_instance": arrays["seg_instance"]}
    if "seg_semantic" in arrays:
        saved["seg_semantic"] = arrays["seg_semantic"]
    np.savez(npz_path, **saved)
    height, width = arrays["rgb"].shape[:2]
    record = {
        "position": [float(v) for v in pose["position"]],
        "orientation_xyzw": [float(v) for v in pose["orientation_xyzw"]],
        "image_width": int(width),
        "image_height": int(height),
    }
    for key in CAMERA_ATTRIBUTES:
        value = camera_info.get(key)
        record[key] = None if value is None else float(value)
    semantic = _names(seg_info.get("semantic")) if "seg_semantic" in arrays else {}
    record["seg_info"] = {"instance": _names(seg_info["instance"]), "semantic": semantic}
    json_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return npz_path, json_path


class CameraProbe:
    """What the runner holds as ``run.probe`` when ``--probe-out`` is given: where to write, and on which ticks."""

    def __init__(self, out_dir, ticks=DEFAULT_TICKS):
        self.out_dir = Path(out_dir)
        self.ticks = frozenset(int(t) for t in ticks)

    def wants(self, tick):
        """Whether ``tick`` (the tick's number before it is recorded) is one of the probed ticks."""
        return int(tick) in self.ticks

    def save(self, seed, tick, cam, arrays, pose, camera_info, seg_info):
        """Writes one camera's probe if the tick is probed. Returns the paths written, or None for an unprobed tick."""
        if not self.wants(tick):
            return None
        return write_probe(self.out_dir, seed, tick, cam, arrays, pose, camera_info, seg_info)
