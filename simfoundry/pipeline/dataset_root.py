# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which OmniGibson dataset root a run writes into, and whether that folder lies outside the checkout.

Stage 13 writes imported assets into a shared dataset folder. When the checkout's deps/ is a link into a shared tree
(a copied checkout), the write lands in the shared folder. This module decides whether that happened, with links
followed. Pure functions: nothing here reads the environment or writes files.
"""

from __future__ import annotations

import os

#: Where the root came from: SIMFOUNDRY_DATASET_ROOT ("explicit"), this checkout's default ("checkout"), or the
#: environment's OMNIGIBSON_DATA_PATH when it is not this checkout's default ("ambient").
SOURCES = ("explicit", "checkout", "ambient")


def source_of_dataset_root(*, explicit: bool, env_value: str | None, checkout_value: str) -> str:
    """The source of the dataset root a stage will use.

    Args:
        explicit: whether SIMFOUNDRY_DATASET_ROOT is set.
        env_value: the OMNIGIBSON_DATA_PATH the stage process has, or None when unset.
        checkout_value: this checkout's default dataset root, as the orchestrator writes it.

    Returns:
        "explicit", "checkout" when the environment holds this checkout's default, else "ambient".
    """
    if explicit:
        return "explicit"
    return "checkout" if env_value == checkout_value else "ambient"


def describe_dataset_root(dataset_path, repo_dir, source) -> dict:
    """Where the dataset folder is once links are followed, and whether that is outside the checkout.

    The check compares path components, not characters: "/a/repo2" is outside "/a/repo".

    Args:
        dataset_path: the folder the stage writes into (the dataset root and the dataset name).
        repo_dir: the checkout's root.
        source: one of SOURCES.

    Returns:
        dict: dataset_path (resolved), repo_realpath (resolved), outside_checkout (bool), source.
    """
    dataset_real = os.path.realpath(dataset_path)
    repo_real = os.path.realpath(repo_dir)
    inside = os.path.commonpath([dataset_real, repo_real]) == repo_real
    return {"dataset_path": dataset_real, "repo_realpath": repo_real, "outside_checkout": not inside, "source": source}


def check_dataset_root(description, allow_external) -> tuple[str | None, str | None]:
    """A warning, and a reason to stop, for a dataset folder outside the checkout.

    Only a checkout default that resolves outside the checkout is flagged. An explicit root is the user's choice, and an
    ambient OMNIGIBSON_DATA_PATH is one the user set for this command; the caller logs that one at INFO instead.

    Args:
        description: the dict describe_dataset_root returns.
        allow_external: s13_usd.allow_external_dataset. False turns the warning into a reason to stop.

    Returns:
        (warning, stop reason): either may be None. The stop reason is set only when allow_external is False.
    """
    if not description["outside_checkout"] or description["source"] != "checkout":
        return None, None
    warning = (
        f"The stage writes into a dataset folder outside this checkout: {description['dataset_path']} "
        f"(checkout: {description['repo_realpath']}). The checkout's deps/ may be a link into a shared tree (a copied "
        "checkout). To write elsewhere, set SIMFOUNDRY_DATASET_ROOT (run.sh --dataset-root); to stop when this "
        "happens, set s13_usd.allow_external_dataset=false."
    )
    if allow_external:
        return warning, None
    return warning, f"{warning} Refusing: s13_usd.allow_external_dataset is false."


def explicit_root_mismatch(dataset_path, explicit_root) -> str | None:
    """Why a stage that was given an explicit root would write outside it, or None when it writes inside it.

    OmniGibson reads OMNIGIBSON_DATA_PATH, not SIMFOUNDRY_DATASET_ROOT. A stage started without the orchestrator (which
    sets OMNIGIBSON_DATA_PATH to the explicit root) therefore writes into OmniGibson's own default root, and this is the
    only place that notices. Both paths are resolved with links followed, and containment is compared by path
    components: "/a/root2" is outside "/a/root".

    Args:
        dataset_path: the folder the stage writes into.
        explicit_root: the root SIMFOUNDRY_DATASET_ROOT names.

    Returns:
        None when dataset_path is inside explicit_root (the folder itself counts as inside), else a message naming both
        resolved paths and the two ways out.
    """
    root_real = os.path.realpath(explicit_root)
    dataset_real = os.path.realpath(dataset_path)
    if os.path.commonpath([dataset_real, root_real]) == root_real:
        return None
    return (
        f"SIMFOUNDRY_DATASET_ROOT is set to {root_real}, but the stage would write into {dataset_real}. "
        "OmniGibson reads OMNIGIBSON_DATA_PATH, not that variable. Run the stage through run.sh or the orchestrator, "
        "or export OMNIGIBSON_DATA_PATH to the same root."
    )
