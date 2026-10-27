# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage info records the model that answered a call, not the Gemini name the config asks for.

With SIMFOUNDRY_LOCAL_BACKEND on, the pipeline still builds a Gemini client; only the client is
swapped for a local backend. These tests check that the names recorded in stage_info.json match
the names the local backend logs, and that stages 5 and 6 record them.
"""

import ast
import importlib
import io
import pathlib
import re
from types import SimpleNamespace

import pytest
from PIL import Image

from simfoundry.models import local_backends
from simfoundry.models.local_backends import served_model_name

STAGES = pathlib.Path("scripts/pipeline/A_reconstruction/stages")


class _FakeGemini:
    """Stands in for simfoundry.models.vlm.Gemini; only the model id is used."""

    def __init__(self, model):
        self.model = model


def _set_local(monkeypatch, enabled=True, image_backend="edit2511", ollama_model="qwen3.5:9b"):
    if enabled:
        monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    else:
        monkeypatch.delenv("SIMFOUNDRY_LOCAL_BACKEND", raising=False)
    monkeypatch.setattr(local_backends, "IMAGE_BACKEND", image_backend)
    monkeypatch.setattr(local_backends, "OLLAMA_MODEL", ollama_model)


# ---------------------------------------------------------------------------
# The name matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "0", "no", "off"])
def test_disabled_returns_the_gemini_name_unchanged(monkeypatch, value):
    _set_local(monkeypatch, enabled=False)
    if value is not None:
        monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", value)

    assert served_model_name("gemini-3-pro-image") == "gemini-3-pro-image"
    assert served_model_name("gemini-3-pro-preview") == "gemini-3-pro-preview"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_enabled_values_route_to_local_models(monkeypatch, value):
    _set_local(monkeypatch)
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", value)

    assert served_model_name("gemini-3-pro-image") == "Qwen-Image-Edit-2511"
    assert served_model_name("gemini-3-pro-preview") == "qwen3.5:9b"


def test_edit2511_names_the_image_model_and_ollama_for_text(monkeypatch):
    _set_local(monkeypatch, image_backend="edit2511")

    assert served_model_name("gemini-3-pro-image") == "Qwen-Image-Edit-2511"
    assert served_model_name("gemini-3-pro-preview") == "qwen3.5:9b"
    assert served_model_name("gemini-3.1-pro-preview") == "qwen3.5:9b"
    # Case does not matter: the backend dispatch lowercases the id too.
    assert served_model_name("Gemini-3-Pro-Image") == "Qwen-Image-Edit-2511"


def test_qwen21_names_the_image_model_and_leaves_text_on_ollama(monkeypatch):
    _set_local(monkeypatch, image_backend="qwen21")

    assert served_model_name("gemini-3-pro-image") == "Qwen-Image-2.1-Turbo"
    assert served_model_name("gemini-3-pro-preview") == "qwen3.5:9b"


# ---------------------------------------------------------------------------
# The call log and the recorded name come from the same place
# ---------------------------------------------------------------------------

def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_comfy(png):
    """Minimal ComfyUI: accepts one prompt, reports it finished, and serves one PNG."""

    def post(url, json=None, **kwargs):
        assert url.endswith("/prompt"), url
        return SimpleNamespace(status_code=200, json=lambda: {"prompt_id": "p"}, text="")

    def get(url, **kwargs):
        if "/history/" in url:
            finished = {"p": {"status": {"completed": True},
                              "outputs": {"out": {"images": [{"filename": "a.png"}]}}}}
            return SimpleNamespace(json=lambda: finished)
        assert "/view" in url, url
        return SimpleNamespace(raise_for_status=lambda: None, content=png)

    return SimpleNamespace(post=post, get=get, RequestException=Exception)


@pytest.mark.parametrize("image_backend", ["edit2511", "qwen21"])
def test_image_call_log_names_what_served_it(monkeypatch, capsys, image_backend):
    _set_local(monkeypatch, image_backend=image_backend)
    monkeypatch.setattr(local_backends, "COMFY_FREE_AFTER", False)
    monkeypatch.setattr(local_backends, "_comfy_upload", lambda img_bytes: "x.png")
    png = _png_bytes()
    monkeypatch.setattr(local_backends, "requests", _fake_comfy(png))

    out = local_backends.comfy_image_edit("a prompt", [png], seed=0, verbose=True)

    assert len(out) == 1
    logged = re.search(r"\[local-image\] (.+?):", capsys.readouterr().out).group(1)
    assert logged == served_model_name("gemini-3-pro-image")


def test_text_call_log_and_request_name_the_ollama_model(monkeypatch, capsys):
    _set_local(monkeypatch)
    monkeypatch.setattr(local_backends, "OLLAMA_KEEP_ALIVE", "60s")
    sent = {}

    def post(url, json=None, timeout=None, **kwargs):
        sent["body"] = json
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"message": {"content": "ok"}, "done_reason": "stop"})

    monkeypatch.setattr(local_backends, "requests",
                        SimpleNamespace(post=post, get=None, RequestException=Exception))

    text, finish = local_backends.ollama_chat("hi", [], verbose=True)

    assert (text, finish) == ("ok", "STOP")
    expected = served_model_name("gemini-3-pro-preview")
    # The model name itself contains a colon, so the group ends at the ": N image(s)" that follows it.
    logged = re.search(r"\[local-vlm\] (.+): 0 image\(s\)", capsys.readouterr().out).group(1)
    assert logged == expected
    assert sent["body"]["model"] == expected


# ---------------------------------------------------------------------------
# Stages 5 and 6 record those names
# ---------------------------------------------------------------------------

def _main_and_finalize_calls(stage_name):
    tree = ast.parse((STAGES / stage_name).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "finalize_stage"]
    definitions = sorted(
        n.lineno for n in ast.walk(main)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "served_models" for t in n.targets)
    )
    return main, calls, definitions


@pytest.mark.parametrize("stage_name, expected_calls", [
    ("5_decompose_scene.py", 4),
    ("6_upsample_object_images.py", 1),
])
def test_every_finalize_stage_call_records_served_models(stage_name, expected_calls):
    # Counting the calls stops one from silently disappearing from the check.
    _, calls, definitions = _main_and_finalize_calls(stage_name)

    assert len(calls) == expected_calls
    assert definitions, "served_models is never assigned in main()"
    for call in calls:
        assert any(isinstance(n, ast.Name) and n.id == "served_models" for n in ast.walk(call))
        # These stages need a GPU, so a use before the definition would only surface as a NameError
        # at run time; the definition has to come first.
        assert definitions[0] < call.lineno


@pytest.mark.parametrize("stage_name", ["5_decompose_scene.py", "6_upsample_object_images.py"])
def test_stage_imports_served_model_name_at_module_level(stage_name):
    # The evaluation tests below inject served_model_name themselves, so only this import keeps a
    # missing line from passing. Without it the stage fails with NameError on the GPU host.
    tree = ast.parse((STAGES / stage_name).read_text())
    imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "simfoundry.models.local_backends"
        and any(alias.name == "served_model_name" for alias in node.names)
    ]

    assert imports


def test_stage6_validity_record_stays_inside_check_valid():
    # check_valid defaults to false. If the record leaked out of its branch, validity_model would be
    # undefined under the default config and the stage would fail with NameError.
    main, _, _ = _main_and_finalize_calls("6_upsample_object_images.py")
    # served_models.update(ollama_options_record()) is unconditional (it is {} unless the Ollama options are set);
    # this test is about the keyword update that the validity branch makes
    updates = [
        n for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "update" and getattr(n.func.value, "id", None) == "served_models" and n.keywords
    ]
    assert len(updates) == 1
    guards = [
        n for n in ast.walk(main)
        if isinstance(n, ast.If) and any(sub is updates[0] for stmt in n.body for sub in ast.walk(stmt))
    ]

    assert guards, "the validity record is not inside the body of an if"
    assert all(isinstance(g.test, ast.Name) and g.test.id == "check_valid" for g in guards)


def _evaluate_recorded(stage_name, namespace):
    """Evaluate main()'s served_models literal and its validity update against fake models.

    Returns the recorded dict. Checking the values, not just the presence of the name, is what
    catches a key that records the configured alias instead of the model that answered.
    """
    main, _, definitions = _main_and_finalize_calls(stage_name)
    assign = min(
        (n for n in ast.walk(main)
         if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "served_models" for t in n.targets)),
        key=lambda n: n.lineno,
    )
    ns = dict(namespace)
    # the Ollama options are not under test here: with none set, their record is {}
    ns.setdefault("ollama_options_record", lambda: {})
    recorded = eval(compile(ast.Expression(body=assign.value), stage_name, "eval"), ns)
    ns["served_models"] = recorded
    for node in ast.walk(main):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update" and getattr(node.func.value, "id", None) == "served_models"):
            eval(compile(ast.Expression(body=node), stage_name, "eval"), ns)
    return recorded


def _stage5_namespace(removal_model, removal_model_name):
    return {
        "Gemini": _FakeGemini,
        "served_model_name": served_model_name,
        "vlm_pro": _FakeGemini("gemini-3-pro-preview"),
        "detection_model_name": "gemini-3-pro-preview",
        "removal_model": removal_model,
        "removal_model_name": removal_model_name,
    }


@pytest.mark.parametrize("enabled, expected", [
    (False, {
        "detection_model": "gemini-3-pro-preview",
        "detection_model_configured": "gemini-3-pro-preview",
        "removal_model": "gemini-3-pro-image",
        "removal_model_configured": "gemini",
    }),
    (True, {
        "detection_model": "qwen3.5:9b",
        "detection_model_configured": "gemini-3-pro-preview",
        "removal_model": "Qwen-Image-Edit-2511",
        "removal_model_configured": "gemini",
    }),
])
def test_stage5_records_the_models_that_answered(monkeypatch, enabled, expected):
    _set_local(monkeypatch, enabled=enabled)
    namespace = _stage5_namespace(_FakeGemini("gemini-3-pro-image"), "gemini")

    assert _evaluate_recorded("5_decompose_scene.py", namespace) == expected


def test_stage5_flux_removal_is_recorded_by_its_name(monkeypatch):
    _set_local(monkeypatch)
    namespace = _stage5_namespace(object(), "flux")

    recorded = _evaluate_recorded("5_decompose_scene.py", namespace)

    assert recorded["removal_model"] == "flux"
    assert recorded["removal_model_configured"] == "flux"


@pytest.mark.parametrize("enabled, expected", [
    (False, {
        "model": "gemini-3-pro-image",
        "model_configured": "gemini",
        "validity_model": "gemini-3-pro-preview",
        "validity_model_configured": "gemini",
    }),
    (True, {
        "model": "Qwen-Image-Edit-2511",
        "model_configured": "gemini",
        "validity_model": "qwen3.5:9b",
        "validity_model_configured": "gemini",
    }),
])
def test_stage6_records_the_models_that_answered(monkeypatch, enabled, expected):
    _set_local(monkeypatch, enabled=enabled)
    namespace = {
        "Gemini": _FakeGemini,
        "served_model_name": served_model_name,
        "model": _FakeGemini("gemini-3-pro-image"),
        "model_name": "gemini",
        "validity_model": _FakeGemini("gemini-3-pro-preview"),
        "validity_model_name": "gemini",
    }

    assert _evaluate_recorded("6_upsample_object_images.py", namespace) == expected


# ---------------------------------------------------------------------------
# Stages 3, 9 and 11 record the same way
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage_name, expected_calls", [
    ("3_segment_ground_plane.py", 1),
    ("9_articulate_objects.py", 1),  # the call sits in the record_stage_result closure
    ("11_make_objects_sim_ready.py", 1),
])
def test_b4_finalize_calls_record_served_models(stage_name, expected_calls):
    _, calls, definitions = _main_and_finalize_calls(stage_name)

    assert len(calls) == expected_calls
    assert definitions, "served_models is never assigned in main()"
    for call in calls:
        assert any(isinstance(n, ast.Name) and n.id == "served_models" for n in ast.walk(call))
        # A use before the definition would only surface as a NameError on the GPU host.
        assert definitions[0] < call.lineno


@pytest.mark.parametrize("stage_name", [
    "3_segment_ground_plane.py",
    "9_articulate_objects.py",
    "11_make_objects_sim_ready.py",
])
def test_b4_stage_imports_served_model_name_at_module_level(stage_name):
    tree = ast.parse((STAGES / stage_name).read_text())
    imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "simfoundry.models.local_backends"
        and any(alias.name == "served_model_name" for alias in node.names)
    ]

    assert imports


def _stage_result_additional_info(stage_name):
    """The additional_info expression of the stage's single StageResult(...) call."""
    main, _, _ = _main_and_finalize_calls(stage_name)
    results = [
        n for n in ast.walk(main)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "StageResult"
    ]
    assert len(results) == 1
    keyword = next(k for k in results[0].keywords if k.arg == "additional_info")
    return keyword.value


@pytest.mark.parametrize("enabled, expected_name", [(False, "gemini-3.1-pro-preview"), (True, "qwen3.5:9b")])
def test_stage3_records_the_detection_model_that_answers(monkeypatch, enabled, expected_name):
    _set_local(monkeypatch, enabled=enabled)
    namespace = {
        "served_model_name": served_model_name,
        "cfg": SimpleNamespace(s3_ground=SimpleNamespace(detection_model="gemini-3.1-pro-preview")),
    }

    recorded = _evaluate_recorded("3_segment_ground_plane.py", namespace)

    assert recorded == {
        "detection_model": expected_name,
        "detection_model_configured": "gemini-3.1-pro-preview",
    }


@pytest.mark.parametrize("enabled, expected_name", [(False, "gemini-3-pro-preview"), (True, "qwen3.5:9b")])
def test_stage9_records_the_classifier_that_answers(monkeypatch, enabled, expected_name):
    _set_local(monkeypatch, enabled=enabled)
    namespace = {
        "served_model_name": served_model_name,
        "vlm": _FakeGemini("gemini-3-pro-preview"),
        "cfg": SimpleNamespace(s9_articulate_objects=SimpleNamespace(vlm_model="gemini-3-pro-preview")),
    }

    recorded = _evaluate_recorded("9_articulate_objects.py", namespace)

    assert recorded == {"vlm_model": expected_name, "vlm_model_configured": "gemini-3-pro-preview"}


def test_stage9_record_stage_result_merges_served_models_first(monkeypatch):
    # record_stage_result writes {**served_models, **additional_info}: the call's own keys are added
    # after the served names, so a key the call sets wins over a recorded one.
    node = _stage_result_additional_info("9_articulate_objects.py")
    assert isinstance(node, ast.Dict)
    assert node.keys[0] is None and isinstance(node.values[0], ast.Name) and node.values[0].id == "served_models"

    recorded = {"vlm_model": "qwen3.5:9b", "vlm_model_configured": "gemini-3-pro-preview"}
    merged = eval(compile(ast.Expression(body=node), "9_articulate_objects.py", "eval"),
                  {"served_models": recorded, "additional_info": {"articulated_objects": ["cup"], "vlm_model": "x"}})

    assert merged == {"vlm_model": "x", "vlm_model_configured": "gemini-3-pro-preview", "articulated_objects": ["cup"]}


@pytest.mark.parametrize("enabled, expected_name", [(False, "gemini-3-pro-preview"), (True, "qwen3.5:9b")])
def test_stage11_records_the_vlm_that_answers(monkeypatch, enabled, expected_name):
    _set_local(monkeypatch, enabled=enabled)
    namespace = {
        "served_model_name": served_model_name,
        "vlm": _FakeGemini("gemini-3-pro-preview"),
        "vlm_model": "gemini-3-pro-preview",
    }

    recorded = _evaluate_recorded("11_make_objects_sim_ready.py", namespace)

    assert recorded == {"vlm_model": expected_name, "vlm_model_configured": "gemini-3-pro-preview"}


# ---------------------------------------------------------------------------
# The qwen21 name follows the UNET file the workflow loads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unet, expected", [
    ("qwen-image-2.1-turbo-Q8_0.gguf", "Qwen-Image-2.1-Turbo"),
    ("Qwen-Image-2.1-TURBO-Q4.gguf", "Qwen-Image-2.1-Turbo"),
    ("qwen-image-2.1-Q8_0.gguf", "Qwen-Image-2.1"),
    ("sub/dir/qwen-image-2.1-Q8_0.gguf", "Qwen-Image-2.1"),
    ("qwen-image-2.1", "Qwen-Image-2.1"),
    ("my_finetune.gguf", "my_finetune"),
    ("qwen-image-edit-2511-Q8_0.gguf", "qwen-image-edit-2511-Q8_0"),
    ("my.model.v2", "my.model.v2"),
    ("x.safetensors", "x"),
])
def test_the_qwen21_name_is_read_from_the_unet_file_name(unet, expected):
    assert local_backends._qwen21_display_name(unet) == expected


def test_the_name_follows_the_unet_the_workflow_loads_at_call_time(monkeypatch):
    _set_local(monkeypatch, image_backend="qwen21")
    monkeypatch.setattr(local_backends, "Q21_UNET", "qwen-image-2.1-turbo-Q8_0.gguf")
    assert served_model_name("gemini-3-pro-image") == "Qwen-Image-2.1-Turbo"
    monkeypatch.setattr(local_backends, "Q21_UNET", "qwen-image-2.1-Q8_0.gguf")
    assert served_model_name("gemini-3-pro-image") == "Qwen-Image-2.1"
    assert served_model_name("gemini-3-pro-preview") == "qwen3.5:9b"


def test_the_unet_named_in_the_environment_is_the_one_the_name_follows(monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    monkeypatch.setenv("SIMFOUNDRY_IMAGE_BACKEND", "qwen21")
    monkeypatch.setenv("SIMFOUNDRY_Q21_UNET", "qwen-image-2.1-Q8_0.gguf")
    try:
        importlib.reload(local_backends)
        assert local_backends.Q21_UNET == "qwen-image-2.1-Q8_0.gguf"
        assert local_backends.image_backend_name() == "Qwen-Image-2.1"
        assert local_backends.served_model_name("gemini-3-pro-image") == "Qwen-Image-2.1"
    finally:
        monkeypatch.undo()
        importlib.reload(local_backends)


def test_edit2511_keeps_its_name_whatever_unet_is_set(monkeypatch):
    _set_local(monkeypatch, image_backend="edit2511")
    monkeypatch.setattr(local_backends, "Q21_UNET", "qwen-image-2.1-Q8_0.gguf")
    assert served_model_name("gemini-3-pro-image") == "Qwen-Image-Edit-2511"


def _default_argument(value):
    """The default of an `os.environ.get(name, default)` read, also inside `int(...)`."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "int":
        value = value.args[0]
    return ast.literal_eval(value.args[1])


def test_the_code_defaults_for_the_qwen21_unet_and_steps_are_the_ones_the_readme_states():
    # The README names these defaults (repro/README.md, qwen21 paragraph and chapter 32). The module reads them from the
    # environment at import, so the defaults are read from the source, whatever the environment says.
    tree = ast.parse(pathlib.Path(local_backends.__file__).read_text())
    defaults = {node.targets[0].id: _default_argument(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("Q21_UNET", "Q21_STEPS")}
    assert defaults == {"Q21_UNET": "qwen-image-2.1-turbo-Q8_0.gguf", "Q21_STEPS": "8"}
    readme = pathlib.Path("repro/README.md").read_text()
    # The whole sentence fragment, with what is on either side of the step count: "8단계" alone occurs elsewhere in the README.
    assert "`qwen-image-2.1-turbo-Q8_0.gguf`와 8단계입니다" in readme
