# SPDX-License-Identifier: Apache-2.0
"""Checks of the splat-background documentation against the code and against the files it names (no GPU, no simulator).

The checks are plain text and JSON reads: the required statements of docs/BACKGROUND_SWAP.md (the default, the none option, the
spec-root and scene-guard rules, and the observed evaluation of three checkpoints trained on the old background, with its
numbers), the two recipes and the none sentence of scripts/cutting/policy/README.md, the example spec in the doc against the
packaged spec, and the rule that the GitHub-bound text names no local folder, no machine, no internal name and no placeholder.
The commits the doc names are checked when this clone has them.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "BACKGROUND_SWAP.md"
POLICY_README = REPO / "scripts" / "cutting" / "policy" / "README.md"
PACKAGED_SPEC = REPO / "simfoundry" / "configs" / "visual_background" / "p1.json"
TBD_LINE = "> **TBD (owner's decision, still open):** where the two files are hosted."
REQUIRED_STATEMENTS = [
    "--visual-background none",
    "SIMFOUNDRY_VISUAL_BACKGROUND_ROOT",
    "--visual-background-root",
    "Data/visual_background",
    "scene_requires",
    "cannot verify",
    "The key's presence does not say whether the background was swapped.",
    "The `name` does:",
    "In an evaluation of three checkpoints trained on the old background",
    "all 150 episodes end in a basic and a strict success",
    "and none fails.",
    "-0.516 (interval -0.648 to -0.413)",
    "-0.166 (-0.26 to -0.09)",
    "8.13 mm against 4.71 mm",
    "+0.70 to +0.81 mm",
    "0.69 in front and 0.70 to 0.71 on the wrist",
    "The cause is not separated",
    "A control evaluation uses the same runner on the same system (Ubuntu 26.04) with the background off",
    "-0.212 (interval -0.302 to -0.138)",
    "-0.162 (interval -0.255 to -0.074)",
    "-0.189 (interval -0.320 to -0.070)",
    "4.29, 4.40 and 5.23 mm",
    "+0.57, +0.63 and +0.73 mm",
    "basic success is 50 of 50 episodes",
    "-0.208 (interval -0.297 to -0.138)",
    "-0.106 (interval -0.191 to -0.029)",
    "-0.184 (interval -0.306 to -0.072)",
    "4.45, 3.74 and 5.21 mm",
    "-0.470 (interval -0.620 to -0.353)",
    "-0.523 (interval -0.682 to -0.383)",
    "-0.554 (interval -0.681 to -0.453)",
    "7.86, 8.54 and 7.92 mm",
    "-0.004 (interval -0.037 to +0.027)",
    "-0.056 (interval -0.101 to -0.011)",
    "-0.005 (interval -0.065 to +0.055)",
    "-0.258 (interval -0.367 to -0.164)",
    "-0.361 (interval -0.485 to -0.254)",
    "-0.365 (interval -0.460 to -0.275)",
    "5.3 to 8.3 mm",
    "at most 21 mm",
    "0.8 to 1.7 mm",
    "at most 6.3 mm",
    "stays within 0.06 of the old background's b_s",
    "shows a small difference between the control and the old background (-0.056) that the data cannot tell from run-to-run variation",
    "the look and brightness of the background and the dark wrist region are not separated from each other",
    "counting the labels puts 100.0 % of them on the front counter-patch box 0",
    "an estimate from the stored poses puts 99.8 % on the front counter-patch box 0",
    "has not been ruled out",
    "The same configuration can render in one of two slightly different states: the frames differ by about 2 levels of 255, mostly at edges, with zero mean. Both states were seen on Ubuntu 24.04 and on 26.04; the cause is not known.",
    "To evaluate a checkpoint as it was trained, pass `--visual-background none`.",
]
# built from parts, so that this GitHub-bound file does not contain the forbidden strings itself
FORBIDDEN_TEXT = ["/home/" + "inner", "simfoundry" + "_logs", "super" + "2", "super" + "3", "[I" + "37", "[I" + "38", "P" + "5",
                  "cod" + "er"]
README_RECIPE_SENTENCE = re.compile(
    r"reproduce the numbers of checkpoints trained on the old background[\s\S]{0,200}?--visual-background none")


def _doc():
    return DOC.read_text(encoding="utf-8")


def _first_json_block(text):
    match = re.search(r"```json\n(.*?)\n```", text, flags=re.DOTALL)
    assert match, "the doc has no JSON example"
    return json.loads(match.group(1))


def _flat():
    """The doc with every run of whitespace as one space, so a statement that the Markdown wraps over two lines still counts."""
    return re.sub(r"\s+", " ", _doc())


@pytest.mark.parametrize("statement", REQUIRED_STATEMENTS)
def test_the_doc_states_the_default_options_guard_name_rule_and_the_observed_evaluation(statement):
    assert statement in _flat(), f"docs/BACKGROUND_SWAP.md must state: {statement!r}"


def test_the_tbd_marker_is_one_standalone_line_and_appears_once():
    assert _doc().splitlines().count(TBD_LINE) == 1, "the owner's-decision TBD sentence must appear once, alone on its line"


def test_the_doc_names_no_local_folder_no_machine_no_internal_name_and_no_placeholder():
    text = _doc()
    for forbidden in FORBIDDEN_TEXT:
        assert forbidden not in text, f"the GitHub-bound doc must not contain {forbidden!r}"
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), "the doc must not contain an e-mail address"


def test_the_example_reflectance_rects_and_thickness_are_the_packaged_spec_values():
    assert PACKAGED_SPEC.is_file(), "the packaged spec simfoundry/configs/visual_background/p1.json is missing"
    packaged = json.loads(PACKAGED_SPEC.read_text(encoding="utf-8"))["counter_patch"]
    example = _first_json_block(_doc())["counter_patch"]
    assert example["rgb"] == packaged["rgb"] == [0.1875, 0.1825, 0.185]
    assert example["rects"] == packaged["rects"]
    assert example["z_top_m"] == packaged["z_top_m"] and example["thickness_m"] == packaged["thickness_m"]


def test_the_example_ccm_and_scene_requires_are_the_packaged_values():
    packaged = json.loads(PACKAGED_SPEC.read_text(encoding="utf-8"))
    example = _first_json_block(_doc())
    assert example["ccm"] == packaged["ccm"], "the example ccm must be the packaged matrix, digit for digit"
    assert example["scene_requires"] == packaged["scene_requires"]


def test_the_example_paths_are_relative_to_the_spec_file():
    example = _first_json_block(_doc())
    assert example["usdz"] == "PD_slab4_bg.usdz" and example["pose_json"] == "align_v4b.json"


def test_the_policy_readme_has_the_default_recipe_and_the_none_recipe():
    text = POLICY_README.read_text(encoding="utf-8")
    assert "--visual-background none" in text, "the policy README must show the recipe with --visual-background none"
    assert "default background" in text, "the policy README must say which recipe is the default background"
    default_block = [block for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)
                     if "tofu_policy_episodes.py" in block and "--visual-background none" not in block]
    assert default_block, "the policy README must show the runner call with the default background"


def test_the_policy_readme_says_the_none_option_reproduces_the_old_background_numbers():
    assert README_RECIPE_SENTENCE.search(POLICY_README.read_text(encoding="utf-8")), (
        "the policy README must say, next to the recipes, that the old-background numbers need --visual-background none")


def test_the_commits_the_doc_names_are_in_this_clone():
    shas = sorted(set(re.findall(r"`([0-9a-f]{7})`", _doc())))
    assert shas, "the doc names no commit; the test expects the base commits of the probe and visual-background tests"
    for sha in shas:
        probe = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
                               capture_output=True, text=True)
        if probe.returncode != 0:
            pytest.skip(f"commit {sha} named in the doc is not in this clone")
        ancestor = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", sha, "HEAD"], capture_output=True)
        assert ancestor.returncode == 0, f"commit {sha} named in the doc is not in the history of HEAD"
