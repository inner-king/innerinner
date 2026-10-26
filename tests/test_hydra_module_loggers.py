# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage module loggers still print under the Hydra logging setup the stage configs select.

Hydra's built-in "disabled" hydra_logging setting has disable_existing_loggers: true, so every logger
a module created before hydra.main ran was switched off. Stage warnings and infos then reached neither
stdout nor the job log. The stage configs select keep_loggers instead, which is the same setting with
that one flag false. The logging setup only runs inside hydra.main, so these tests run small Hydra
programs in a subprocess.
"""

import os
import pathlib
import subprocess
import sys

import hydra
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "scripts" / "cfg"

# The configs that select a hydra_logging setting, which is every stage config.
STAGE_CONFIGS = [
    "real2sim_cfg", "auto_bg", "cluttered_scene", "ServeFruitsOnGreenPlate", "PutMarkerInCup",
    "PutCupInBowl", "stack_dishware", "stack_dishware_easy", "serve_banana",
]

# Module-level loggers, created before hydra.main as in the stage modules, plus a third-party one.
LOGGER_PROBE = '''\
import logging

import hydra

logger = logging.getLogger("simfoundry.pipeline.frame_selection")
third_party = logging.getLogger("some.thirdparty")


@hydra.main(config_name="real2sim_cfg", config_path="__CFG_DIR__", version_base="1.3")
def main(cfg):
    logger.info("PROBE INFO line")
    logger.warning("PROBE WARNING line")
    third_party.warning("PROBE THIRDPARTY WARNING")
    third_party.error("PROBE THIRDPARTY ERROR")


if __name__ == "__main__":
    main()
'''

# The real module, with a one-frame bundle whose only score fails the clipping gate.
FRAME_PROBE = '''\
import hydra
import numpy as np
from omegaconf import open_dict

from simfoundry.pipeline import frame_selection
from simfoundry.pipeline.frame_selection import FrameBundle, FrameScore, select_canonical_frame


def clipped_frame(position, bundle, sam3, cfg, sel_cfg):
    return FrameScore(idx=bundle.frame_ids[position], eligible=False, tilt=0.0, plane_inlier_ratio=0.9,
                      support_coverage=0.5, n_objects=2, clipped_frac=0.65,
                      reject_reason="65% of object area runs off the frame")


frame_selection.score_frame = clipped_frame


@hydra.main(config_name="real2sim_cfg", config_path="__CFG_DIR__", version_base="1.3")
def main(cfg):
    with open_dict(cfg):
        cfg.s3_ground.frame_selection.mode = "heuristic"
        cfg.s3_ground.frame_selection.write_debug_sheet = False
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    select_canonical_frame(cfg, sam3=None, bundle=FrameBundle([frame], [frame], [np.eye(3)], [None]))


if __name__ == "__main__":
    main()
'''


def _run(tmp_path, program, *overrides):
    """Run `program` under Hydra in a subprocess; return (stdout, job log text)."""
    script = tmp_path / "probe.py"
    script.write_text(program.replace("__CFG_DIR__", str(CFG_DIR)))
    run_dir = tmp_path / "run"
    env = {**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-B", str(script), f"hydra.run.dir={run_dir}", *overrides],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    # The job log is named after the script (hydra.job.name), inside the run directory.
    return proc.stdout, (run_dir / "probe.log").read_text()


def test_module_infos_and_warnings_reach_stdout_and_the_job_log(tmp_path):
    stdout, log = _run(tmp_path, LOGGER_PROBE)
    # Exactly once each: a logger that also propagates to root's handlers prints every line twice.
    for line in ("PROBE INFO line", "PROBE WARNING line"):
        assert stdout.count(line) == 1
        assert log.count(line) == 1
    # The root level still hides library warnings and still shows errors.
    for text in (stdout, log):
        assert "PROBE THIRDPARTY WARNING" not in text
        assert "PROBE THIRDPARTY ERROR" in text


def test_the_disabled_setting_silences_module_loggers_while_the_program_still_runs(tmp_path):
    # The contrast that explains the bug: under the old setting the module lines are gone, but the
    # root-logger error still shows, so the program ran and only the module loggers were switched off.
    stdout, log = _run(tmp_path, LOGGER_PROBE, "hydra/hydra_logging=disabled")
    for text in (stdout, log):
        assert "PROBE INFO line" not in text
        assert "PROBE WARNING line" not in text
        assert "PROBE THIRDPARTY ERROR" in text


def test_the_lone_candidate_warning_reaches_stdout_and_the_job_log(tmp_path):
    stdout, log = _run(tmp_path, FRAME_PROBE)
    warning = "Only one candidate frame (idx 0) was scored"
    assert stdout.count(warning) == 1
    assert log.count(warning) == 1


def _hydra_logging_overrides(name):
    config = yaml.safe_load((CFG_DIR / f"{name}.yaml").read_text()) or {}
    return [
        value
        for entry in config.get("defaults", [])
        if isinstance(entry, dict)
        for key, value in entry.items()
        if "hydra/hydra_logging" in key
    ]


@pytest.mark.parametrize("name", STAGE_CONFIGS)
def test_each_stage_config_selects_keep_loggers(name):
    assert _hydra_logging_overrides(name) == ["keep_loggers"]


def test_only_the_stage_configs_select_a_hydra_logging_setting():
    selecting = {path.stem for path in CFG_DIR.glob("*.yaml") if _hydra_logging_overrides(path.stem)}
    assert selecting == set(STAGE_CONFIGS)


def test_no_config_selects_the_disabled_setting():
    for path in CFG_DIR.glob("*.yaml"):
        assert "disabled" not in _hydra_logging_overrides(path.stem), path.name


def test_keep_loggers_is_hydras_disabled_setting_with_existing_loggers_kept():
    keep_path = CFG_DIR / "hydra" / "hydra_logging" / "keep_loggers.yaml"
    assert keep_path.read_text().splitlines()[0] == "# @package hydra.hydra_logging"
    builtin = pathlib.Path(hydra.__file__).parent / "conf" / "hydra" / "hydra_logging" / "disabled.yaml"
    keep = yaml.safe_load(keep_path.read_text())
    assert keep == {**yaml.safe_load(builtin.read_text()), "disable_existing_loggers": False}
