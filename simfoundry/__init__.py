# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Set hardcoded-macros
# TODO: Make these configurable and download the checkpoints at the start
ROOT_DIR = os.path.dirname(__file__)
REPO_DIR = '/'.join(ROOT_DIR.split('/')[:-1])
CHECKPOINT_DIR = f"{REPO_DIR}/checkpoints"
ASSET_DIR = f"{REPO_DIR}/assets" # TODO: maybe a way to detect this?
CFG_DIR = f"{REPO_DIR}/scripts/cfg"
DATA_DIR = f"{REPO_DIR}/Data"


DATASET_ROOT_ENV = "SIMFOUNDRY_DATASET_ROOT"


def checkout_dataset_root() -> str:
    """This checkout's OmniGibson datasets, deps/BEHAVIOR-1K/datasets. The path may be a link out of the checkout."""
    return os.path.join(REPO_DIR, "deps", "BEHAVIOR-1K", "datasets")


def dataset_root_source() -> str:
    """Return "explicit" when SIMFOUNDRY_DATASET_ROOT names the dataset root, otherwise "checkout"."""
    return "explicit" if os.environ.get(DATASET_ROOT_ENV) else "checkout"


def get_omnigibson_data_path() -> str:
    """Return the OmniGibson dataset root that pipeline assets are written to and read from.

    SIMFOUNDRY_DATASET_ROOT (run.sh --dataset-root) names the root explicitly. It must be an existing directory: a
    missing one stops the run rather than falling back to the checkout. Without it the root is this checkout's datasets.
    The shell's OMNIGIBSON_DATA_PATH is not read here, so the stages of one run (13 writes the assets, 14 reads them)
    share one root, and a value left over from another OmniGibson project cannot split them. The reason is inferred:
    the release records none for forcing the checkout's root.
    """
    explicit = os.environ.get(DATASET_ROOT_ENV)
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{DATASET_ROOT_ENV}={explicit!r} is not a directory; create it or unset the variable")
        return path
    return checkout_dataset_root()


def configure_omnigibson_data_path(*, force: bool = True) -> str:
    """Point OmniGibson at the pipeline's dataset root before importing it.

    The root is get_omnigibson_data_path(). An explicit SIMFOUNDRY_DATASET_ROOT always goes into the environment, even
    with force=False: it is the user's choice, not a default to keep. The checkout's root replaces an existing
    OMNIGIBSON_DATA_PATH only when force is set and the checkout has datasets, as before.
    """
    dataset_root = get_omnigibson_data_path()
    explicit = dataset_root_source() == "explicit"
    if explicit or (os.path.isdir(dataset_root) and (force or not os.environ.get("OMNIGIBSON_DATA_PATH"))):
        os.environ["OMNIGIBSON_DATA_PATH"] = dataset_root
    return os.environ.get("OMNIGIBSON_DATA_PATH", dataset_root)


# For importing specific OmniGibson dependences
def import_og_dependencies():
    from simfoundry.tasks.pick_place_task import PickPlaceTask
