# SPDX-License-Identifier: Apache-2.0
"""The package docstring of simfoundry/cutting, which joins the MPM solver with runtime cutting; the modules it names
import on their own, and the two cutting scripts answer --help. CPU only, no Isaac Sim."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CUTTING = REPO / "simfoundry" / "cutting"
RUNTIME_MODULES = ["geometry", "usd_io", "rule", "knife_asset"]
CUTTING_README = re.compile(r"(?<![\w/])README\.md")  # the cutting README, not repro/README.md


def _package_doc():
    return ast.get_docstring(ast.parse((CUTTING / "__init__.py").read_text(encoding="utf-8"))) or ""


def test_the_docstring_has_the_mpm_solver_paragraph():
    paragraphs = _package_doc().split("\n\n")
    solver = [p for p in paragraphs if "MPM" in p]
    assert solver, "no paragraph of the package docstring names the MPM solver"
    assert any("tofu" in p and CUTTING_README.search(p) for p in solver), (
        "the MPM paragraph must name tofu and point at the cutting README.md")
    assert (CUTTING / "README.md").is_file()


def test_the_docstring_has_the_runtime_cutting_paragraph():
    assert any(p.startswith("Runtime cutting:") for p in _package_doc().split("\n\n"))


def test_the_docstring_lists_the_runtime_modules_once_each_in_double_backticks():
    doc = _package_doc()
    bullets = re.findall(r"^- ``(\w+)``:", doc, flags=re.MULTILINE)
    assert sorted(bullets) == sorted(RUNTIME_MODULES)
    for name in RUNTIME_MODULES:
        assert f"``{name}``" in doc
        assert (CUTTING / f"{name}.py").is_file(), f"{name}.py is named in the docstring but is not in the package"


def test_the_docstring_points_at_repro_section_15_2():
    doc = _package_doc()
    assert "repro/README.md" in doc
    assert "15.2" in doc
    repro = (REPO / "repro" / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^### 15\.2 ", repro, flags=re.MULTILINE), "repro/README.md has no section 15.2"


def _import_in_isolated_process(dotted):
    code = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"m = importlib.import_module({dotted!r})\n"
        f"assert str(m.__file__).startswith({str(REPO)!r}), m.__file__\n"
        "print('ok', m.__name__)\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=600)


MODULES = [
    pytest.param("simfoundry.cutting", id="package"),
    *[pytest.param(f"simfoundry.cutting.{p.stem}", id=p.stem) for p in sorted(CUTTING.glob("*.py"))
      if p.stem != "__init__" and p.stem != "rule"],
    pytest.param("simfoundry.cutting.rule", id="rule", marks=pytest.mark.skip(
        reason="rule imports OmniGibson and torch at the top: about 10 s here, and it only runs where OmniGibson does")),
]


@pytest.mark.parametrize("dotted", MODULES)
def test_the_cutting_modules_import_in_a_fresh_interpreter(dotted):
    proc = _import_in_isolated_process(dotted)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().startswith("ok ")


@pytest.mark.parametrize("script", ["scripts/cutting/tofu_policy_episodes.py", "scripts/cutting/tofu_og_cut.py"])
def test_the_cutting_scripts_answer_help(script):
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, str(REPO / script), "--help"], capture_output=True, text=True, env=env,
                          cwd=str(REPO), timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "usage:" in proc.stdout
