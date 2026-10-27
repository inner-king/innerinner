# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model names for the object assets that stage 11 writes into the shared asset dataset.

The dataset keys an asset by its model name, and stage 13 imports with --overwrite: two different inputs that get the
same name overwrite each other, and the earlier scene keeps pointing at the asset that was replaced. A name that also
carries a hash of the input makes a different input a different name. Everything here is pure: no file is written, and
the global random state is not used.
"""

import hashlib
import os
import random
import string
from pathlib import Path


def generate_seeded_random_letters(seed_value, length=6):
    """Return `length` random lowercase letters that depend only on `seed_value`.

    The letters are the ones the global seeded version gave (random.seed followed by random.choices), so a name made
    before this module existed does not change. A local random.Random does the drawing, so the global random state is
    left as it was.

    Args:
        seed_value: the seed, a string.
        length: the number of letters.

    Returns:
        str: `length` lowercase letters.
    """
    return "".join(random.Random(seed_value).choices(string.ascii_lowercase, k=length))


def content_hash8(files, scale, articulated):
    """Return the first eight hex digits of a hash of what an asset is made of.

    The hash covers the files, the scale they are written at, and whether the asset is articulated. The order of `files`
    does not count: they are sorted by relative name. A renamed file is a different input, because its relative name is
    hashed with its bytes. The scale enters as `float(scale).hex()`, so scales that differ in the last bits (1.0 and
    1.0000001) give different hashes.

    Args:
        files: (relative name, absolute path) pairs. The relative names must be unique.
        scale: the factor the files are written at.
        articulated: whether the asset is an articulated object (its files and a rigid object's are different kinds).

    Returns:
        str: eight lowercase hex digits.

    Raises:
        ValueError: when two of the files have the same relative name.
    """
    names = [name for name, _ in files]
    if len(set(names)) != len(names):
        raise ValueError(f"the relative names of the hashed files must be unique, got {sorted(names)}")
    digest = hashlib.sha256()
    for name, path in sorted(files, key=lambda item: item[0]):
        with open(path, "rb") as handle:
            digest.update(name.encode("utf-8") + b"\0" + handle.read())
    digest.update(float(scale).hex().encode("ascii"))
    digest.update(b"articulated" if articulated else b"rigid")
    return digest.hexdigest()[:8]


def articulated_input_files(results_dir, urdf_path, mesh_parts_dir):
    """Return the inputs of an articulated object as (relative name, path) pairs.

    These are every file under `mesh_parts_dir` and the URDF. Their relative names are taken from `results_dir` with
    forward slashes, so the same object hashes the same wherever its results folder is.

    Args:
        results_dir: the object's results folder; `urdf_path` and `mesh_parts_dir` lie under it.
        urdf_path: the articulated object's URDF.
        mesh_parts_dir: the folder with the object's mesh parts.

    Returns:
        list: (relative name, absolute or as-given path) pairs.
    """
    pairs = [(Path(urdf_path).relative_to(results_dir).as_posix(), str(urdf_path))]
    for dirpath, _, filenames in os.walk(mesh_parts_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            pairs.append((Path(path).relative_to(results_dir).as_posix(), path))
    return pairs


def model_name(scene_name, category, idx, content_hash=None, length=6):
    """Return the model name of object `idx` of `category` in the scene `scene_name`.

    With no content hash the seed is the scene, the category and the index, as before the content hash existed. With one,
    the hash is part of the seed: the same input gives the same name, and a different input gives a different one.

    Args:
        scene_name: the scene's name.
        category: the object's category.
        idx: the object's index in the scene.
        content_hash: the eight digits from `content_hash8`, or None.
        length: the number of letters.

    Returns:
        str: the model name.
    """
    seed = f"{scene_name}_{category}_{idx}"
    if content_hash is not None:
        seed = f"{seed}_{content_hash}"
    return generate_seeded_random_letters(seed_value=seed, length=length)
