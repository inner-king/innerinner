# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The manifest beside an imported asset: what it was made from, and whether a new import would change it.

Stage 13 imports each object into the shared dataset, where an import under the same model name overwrites the asset
an earlier scene uses. The manifest records hashes of the inputs (the stage-11 folder's geometry and physics, the import
settings, the import code), so the next import can tell three things apart: the same asset (skipped when asked to), a
changed asset (a conflict, reported and handled by `on_conflict`), and an asset made before manifests existed (unknown
provenance). Nothing here imports OmniGibson, so the tests run without it.
"""

import hashlib
import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from simfoundry.utils.python_utils import atomic_output_path

logger = logging.getLogger(__name__)

#: The manifest's file name, beside the asset's own files.
MANIFEST_NAME = "real2sim_asset.json"
#: The values `on_conflict` accepts.
ON_CONFLICT_VALUES = ("warn", "fail", "overwrite")


class AssetConflictError(RuntimeError):
    """A re-import would replace an asset that an earlier scene uses, and `on_conflict` is `fail`."""


def _digest_of(entries):
    """sha256 of (relative name, bytes) pairs in name order. A symbolic link contributes a marker, not its target."""
    digest = hashlib.sha256()
    for name, path in sorted(entries, key=lambda item: item[0]):
        if os.path.islink(path):
            payload = b"<symbolic link>"
        else:
            with open(path, "rb") as handle:
                payload = handle.read()
        digest.update(name.encode("utf-8") + b"\0" + payload)
    return digest.hexdigest()


def directory_hashes(asset_dir):
    """Return the geometry and physics hashes of an asset folder, 64 hex digits each.

    The geometry hash covers every file under the folder, except the URDF files directly in `urdf/` and the manifest
    file at the top, each by its path relative to the folder (forward slashes) and by its bytes. The physics hash covers
    the URDF files directly in `urdf/`, by name, where the masses, friction and inertia are. Timestamps and the order in
    which the files are listed do not count. A symbolic link is not followed: its name counts, its target does not.

    Args:
        asset_dir: the asset's folder.

    Returns:
        dict: `geometry_hash` and `physics_hash`.

    Raises:
        ValueError: when `asset_dir` is not a folder.
    """
    root = Path(asset_dir)
    if not root.is_dir():
        raise ValueError(f"asset folder {root} does not exist")
    geometry, physics = [], []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                geometry.append((Path(path).relative_to(root).as_posix(), path))
        for name in filenames:
            path = os.path.join(dirpath, name)
            relative = Path(path).relative_to(root).as_posix()
            if relative == MANIFEST_NAME:
                continue
            if relative.startswith("urdf/") and relative.endswith(".urdf") and "/" not in relative[len("urdf/"):]:
                physics.append((relative, path))
            else:
                geometry.append((relative, path))
    return {"geometry_hash": _digest_of(geometry), "physics_hash": _digest_of(physics)}


def import_settings_hash(settings):
    """Return the first 16 hex digits of the sha256 of `settings` as JSON with sorted keys."""
    text = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def import_code_hash(files, versions):
    """Return the first 16 hex digits of a hash of the import code and the versions it runs under.

    A change to the import code, or to the OmniGibson it runs under, makes an asset imported before it a settings conflict
    rather than a skip: the same settings may now write different files. Not covered: a hand edit to a checkout whose git
    hash did not change, because the git hash is the only version there is.

    Args:
        files: (name, path) pairs of the code files; their bytes are hashed in name order.
        versions: a dict of version strings, hashed as JSON with sorted keys.

    Returns:
        str: 16 hex digits.
    """
    digest = hashlib.sha256()
    for name, path in sorted(files, key=lambda item: item[0]):
        with open(path, "rb") as handle:
            digest.update(name.encode("utf-8") + b"\0" + handle.read())
    digest.update(json.dumps(versions, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()[:16]


def urdf_mass_kg(urdf_path):
    """Return the total link mass of a URDF in kilograms, or None when it has none or cannot be read.

    The mass is for the line a physics change writes. It is not a hash input: the physics hash covers the URDF itself.
    """
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError):
        return None
    masses = [float(element.get("value")) for element in root.iter("mass") if element.get("value") is not None]
    return float(sum(masses)) if masses else None


def read_manifest(model_dir):
    """Return the manifest in `model_dir`, or None when there is none or it cannot be read.

    A manifest that is not a JSON object is logged as a warning and treated as absent. The asset is then imported with
    unknown provenance, which is reported, not trusted.
    """
    path = Path(model_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("asset manifest %s cannot be read (%s); the asset is treated as having no manifest", path, exc)
        return None
    if not isinstance(record, dict):
        logger.warning("asset manifest %s is not a JSON object; the asset is treated as having no manifest", path)
        return None
    return record


def write_manifest(model_dir, record):
    """Write `record` as the manifest of `model_dir`, atomically, and return its path.

    `written_at` (UTC, ISO 8601) is added unless the record has one. If the write fails, the previous manifest stays.

    Raises:
        OSError: when the folder does not exist or cannot be written.
    """
    path = Path(model_dir) / MANIFEST_NAME
    stored = dict(record)
    stored.setdefault("written_at", datetime.now(timezone.utc).isoformat())
    text = json.dumps(stored, indent=2, sort_keys=True) + "\n"
    with atomic_output_path(path) as tmp_path:
        Path(tmp_path).write_text(text, encoding="utf-8")
    return path


def remove_manifest(model_dir):
    """Delete the manifest in `model_dir` and say whether there was one. Nothing else in the folder is touched.

    An import that is about to overwrite an asset removes the manifest first. If the import then fails half way, the
    asset has no manifest, so the next run reports unknown provenance instead of trusting a manifest that describes the
    asset it replaced.

    Args:
        model_dir: the asset's folder in the dataset.

    Returns:
        bool: True when a manifest was removed, False when there was none, or the folder does not exist.
    """
    path = Path(model_dir) / MANIFEST_NAME
    if not path.is_file():
        return False
    path.unlink()
    return True


def decide_import(model_dir_exists, existing, new):
    """Decide whether an import replaces the asset in the dataset, and whether that is a conflict.

    Args:
        model_dir_exists: whether the asset's folder is in the dataset.
        existing: the manifest in that folder (a dict), or None.
        new: the manifest the import would write, with the hashes.

    Returns:
        dict: `action` ("import" or "skip"), `conflict` (None, "unknown_provenance", "geometry", "settings" or
        "physics") and `reason` (a short text).

    The checks run in the order that matters: the geometry first (the asset itself differs), then the settings and the
    import code (the same geometry imported differently), then the physics (only the masses and the friction differ).
    """
    if not model_dir_exists:
        return {"action": "import", "conflict": None, "reason": "no asset in the dataset yet"}
    if existing is None:
        return {"action": "import", "conflict": "unknown_provenance",
                "reason": "the asset folder has no manifest: it was made before manifests existed"}
    if existing.get("geometry_hash") != new["geometry_hash"]:
        return {"action": "import", "conflict": "geometry",
                "reason": f"the geometry differs (manifest {str(existing.get('geometry_hash'))[:8]}, "
                          f"new {new['geometry_hash'][:8]})"}
    if (existing.get("import_settings_hash") != new["import_settings_hash"]
            or existing.get("import_code_hash") != new["import_code_hash"]):
        return {"action": "import", "conflict": "settings",
                "reason": "the import settings or the import code differ"}
    if existing.get("physics_hash") != new["physics_hash"]:
        return {"action": "import", "conflict": "physics",
                "reason": "only the physics differ (URDF masses, friction or inertia)"}
    return {"action": "skip", "conflict": None,
            "reason": "geometry, settings, import code and physics are unchanged"}


def _kilograms(value):
    return "unknown" if value is None else f"{value:.4g} kg"


def describe_decision(decision, asset_dir, existing, new):
    """Return the text for a decision from decide_import: what the asset is, and what the import does to it.

    The geometry text names the earlier scene and the geometry hashes, and says that the earlier scene's objects that
    point at this asset change, which is the reason the conflict matters.

    Args:
        decision: the dict decide_import returned.
        asset_dir: the asset's folder in the dataset.
        existing: the manifest that was there, or None.
        new: the manifest the import would write.

    Returns:
        str: one line of text.
    """
    previous = existing or {}
    conflict = decision["conflict"]
    if conflict == "geometry":
        return (f"asset {asset_dir} was written by scene {previous.get('scene_name', '?')} "
                f"(geometry {str(previous.get('geometry_hash'))[:8]}); this import writes geometry "
                f"{new['geometry_hash'][:8]} over it, and the earlier scenes' objects that point at this asset change")
    if conflict == "settings":
        return f"the import settings or the import code changed for {asset_dir}; the asset is imported again"
    if conflict == "physics":
        return (f"physics changed for {asset_dir} (mass {_kilograms(previous.get('mass_kg'))} -> "
                f"{_kilograms(new.get('mass_kg'))}); the asset is imported again")
    if conflict == "unknown_provenance":
        return f"{asset_dir} has no manifest (made before manifests existed); it is imported over and given one"
    return f"{asset_dir}: {decision['reason']}"


def apply_conflict_policy(decision, on_conflict, describe):
    """Report a decision from decide_import, and for a geometry conflict apply `on_conflict`.

    Args:
        decision: the dict decide_import returned.
        on_conflict: "warn" (the default: a warning, and the import goes on), "fail" (AssetConflictError before the
            import), or "overwrite" (an INFO line, and the import goes on). Only a geometry conflict is subject to it.
            A settings change and an unknown provenance are always warnings; a physics change is always an INFO line.
        describe: a function from the decision to its text.

    Raises:
        ValueError: when `on_conflict` is not one of the three values, or the decision names no known conflict.
        AssetConflictError: for a geometry conflict when `on_conflict` is "fail".
    """
    if on_conflict not in ON_CONFLICT_VALUES:
        raise ValueError(f"on_conflict must be one of {ON_CONFLICT_VALUES}, got {on_conflict!r}")
    conflict = decision["conflict"]
    if conflict is None:
        return
    message = describe(decision)
    if conflict == "geometry":
        if on_conflict == "fail":
            raise AssetConflictError(message)
        if on_conflict == "warn":
            logger.warning(message)
        else:
            logger.info(message)
    elif conflict in ("settings", "unknown_provenance"):
        logger.warning(message)
    elif conflict == "physics":
        logger.info(message)
    else:
        raise ValueError(f"unknown conflict {conflict!r}")
