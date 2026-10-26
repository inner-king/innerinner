# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ollama's num_batch and num_gpu pins: the request body, the cache key and the stage records (no GPU, no Ollama).

requests.post is replaced by a fake that records a copy of each body. The environment constants of local_backends are
set per test, so the results do not depend on the environment the tests run in.
"""

import ast
import copy
import pathlib
import subprocess

import pytest
import requests

from simfoundry.models import local_backends as L
from simfoundry.models import vlm
from simfoundry.models.remote_cache import RemoteModelCache

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGES = REPO / "scripts/pipeline/A_reconstruction/stages"
RUN_SCRIPT = REPO / "repro/run_local_pipeline.sh"

# The key of the Gemini request built by _request() below, computed before the Ollama options existed (db139be) from
# the request dict that Gemini.__call__ hashed then. Pinned as a digest, so any change to the key fails here: a changed
# key would miss every cached call without an error.
OLD_GEMINI_KEY = "1c044c1ed74739125d4100552d5fe80c0085381510883abe1f62007422701001"


@pytest.fixture(autouse=True)
def _ollama_settings(monkeypatch):
    """A fixed Ollama setup with no pinned options and local serving off; each test sets what it needs."""
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "0")
    monkeypatch.setattr(L, "OLLAMA_MODEL", "test-model")
    # Not "0": a zero keep-alive makes ollama_chat poll the server after every call.
    monkeypatch.setattr(L, "OLLAMA_KEEP_ALIVE", "60s")
    monkeypatch.setattr(L, "OLLAMA_NUM_CTX", 16384)
    monkeypatch.setattr(L, "OLLAMA_MAX_TOKENS", 4096)
    monkeypatch.setattr(L, "OLLAMA_THINK", None)
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", None)
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", None)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Fake:
    """Stands in for requests.post: records a copy of each body and answers with `replies` in turn, the last repeating."""

    def __init__(self):
        self.sent = []
        self.replies = [{"message": {"content": "ok"}, "done_reason": "stop"}]

    def post(self, url, json=None, timeout=None):
        # ollama_chat changes its body on a retry, so the copy keeps the body as it was sent
        self.sent.append(copy.deepcopy(json))
        return _Response(self.replies[min(len(self.sent), len(self.replies)) - 1])


@pytest.fixture
def fake(monkeypatch):
    fake = _Fake()
    monkeypatch.setattr(L.requests, "post", fake.post)
    return fake


def _body(options=None):
    """The body ollama_chat("hello", []) sends with the fixed settings, plus the extra options given."""
    body_options = {"temperature": 0.0, "seed": 0, "num_predict": 4096, "num_ctx": 16384}
    body_options.update(options or {})
    return {
        "model": "test-model",
        "stream": False,
        "keep_alive": "60s",
        "messages": [
            {"role": "system", "content": L.SYSTEM_PROMPT},
            {"role": "user", "content": "hello", "images": []},
        ],
        "options": body_options,
    }


# --- the request body ----------------------------------------------------------------------------------------------------

def test_with_neither_option_set_the_body_is_the_one_it_always_was(fake):
    assert L.ollama_chat("hello", []) == ("ok", "STOP")
    assert fake.sent == [_body()]


def test_num_batch_alone_adds_num_batch_as_an_integer(fake, monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    L.ollama_chat("hello", [])
    assert fake.sent == [_body({"num_batch": 1024})]


def test_num_gpu_alone_adds_num_gpu_as_an_integer(fake, monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    L.ollama_chat("hello", [])
    assert fake.sent == [_body({"num_gpu": 99})]


def test_both_options_are_added_and_nothing_else_changes(fake, monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    L.ollama_chat("hello", [])
    assert fake.sent == [_body({"num_batch": 1024, "num_gpu": 99})]


def test_zero_gpu_layers_is_a_setting_not_an_unset_one(fake, monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "0")
    L.ollama_chat("hello", [])
    assert fake.sent == [_body({"num_gpu": 0})]


def test_minus_one_gpu_layers_is_sent_as_minus_one(fake, monkeypatch):
    # -1 is Ollama's own automatic placement, and the old code sent every integer through, so it still goes through
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "-1")
    L.ollama_chat("hello", [])
    assert fake.sent == [_body({"num_gpu": -1})]


def test_an_empty_setting_counts_as_unset(fake, monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "")
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "")
    L.ollama_chat("hello", [])
    assert fake.sent == [_body()]


@pytest.mark.parametrize(("name", "attr", "raw"), [
    ("SIMFOUNDRY_OLLAMA_NUM_BATCH", "OLLAMA_NUM_BATCH", "abc"),
    ("SIMFOUNDRY_OLLAMA_NUM_BATCH", "OLLAMA_NUM_BATCH", "0"),
    ("SIMFOUNDRY_OLLAMA_NUM_BATCH", "OLLAMA_NUM_BATCH", "-1"),
    ("SIMFOUNDRY_OLLAMA_NUM_BATCH", "OLLAMA_NUM_BATCH", "-5"),
    ("SIMFOUNDRY_OLLAMA_NUM_BATCH", "OLLAMA_NUM_BATCH", "1.5"),
    ("SIMFOUNDRY_OLLAMA_NUM_GPU", "OLLAMA_NUM_GPU", "abc"),
    ("SIMFOUNDRY_OLLAMA_NUM_GPU", "OLLAMA_NUM_GPU", "-2"),
    ("SIMFOUNDRY_OLLAMA_NUM_GPU", "OLLAMA_NUM_GPU", "1.5"),
])
def test_a_bad_value_is_refused_by_name_before_any_request_is_sent(fake, monkeypatch, name, attr, raw):
    monkeypatch.setattr(L, attr, raw)
    with pytest.raises(ValueError, match=name) as info:
        L.ollama_chat("hello", [])
    assert repr(raw) in str(info.value)
    assert fake.sent == []


def test_the_retry_after_a_length_stop_keeps_the_options(fake, monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    fake.replies = [
        {"message": {"content": "long"}, "done_reason": "length"},
        {"message": {"content": "short"}, "done_reason": "stop"},
    ]
    assert L.ollama_chat("hello", []) == ("short", "STOP")
    assert fake.sent == [
        _body({"num_batch": 1024}),
        # the retry samples instead of decoding greedily, and its seed moves on (ollama_chat)
        _body({"num_batch": 1024, "temperature": 0.6, "seed": 7919}),
    ]


def test_an_http_500_is_raised_as_it_is_and_not_retried_with_other_options(monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    sent = []

    def refuse(url, json=None, timeout=None):
        # The refusal is raised unchanged: a retry with other options would hide it behind an answer made with
        # settings nobody chose.
        sent.append(copy.deepcopy(json))
        refused = requests.Response()
        refused.status_code = 500
        refused.reason = "Internal Server Error"
        refused.url = url
        return refused

    monkeypatch.setattr(L.requests, "post", refuse)
    with pytest.raises(requests.HTTPError, match="500 Server Error"):
        L.ollama_chat("hello", [])
    assert sent == [_body({"num_batch": 1024, "num_gpu": 99})]


# --- the cache key -------------------------------------------------------------------------------------------------------

def _cache_key(model, request):
    return RemoteModelCache(root_dir="unused", mode="cache").key_for(provider="gemini", model=model, request=request)


def _request(model="gemini-3-pro-preview", prompt="a prompt"):
    return vlm.gemini_cache_request(model, prompt, [], 0.0, 0, 0, 8192, ["TEXT"])


@pytest.mark.parametrize("local", ["0", "1"])
def test_with_neither_option_the_key_is_the_one_it_always_was(monkeypatch, local):
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", local)
    assert _cache_key("gemini-3-pro-preview", _request()) == OLD_GEMINI_KEY


def test_num_batch_512_1024_and_unset_give_three_different_keys(monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    keys = []
    for batch in (None, "512", "1024"):
        monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", batch)
        keys.append(_cache_key("gemini-3-pro-preview", _request()))
    assert len(set(keys)) == 3


def test_num_gpu_alone_changes_the_key(monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    assert _request()["local_backend_options"] == {"num_gpu": 99}
    assert _cache_key("gemini-3-pro-preview", _request()) != OLD_GEMINI_KEY


def test_the_options_in_the_key_are_the_integers_that_are_set(monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    assert _request()["local_backend_options"] == {"num_batch": 1024}
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    assert _request()["local_backend_options"] == {"num_batch": 1024, "num_gpu": 99}


def test_an_image_model_keeps_its_key_with_the_options_set(monkeypatch):
    model = "gemini-3-pro-image"
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    with_options = _cache_key(model, _request(model))
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "0")  # the same call with no Ollama in it: no options by construction
    assert with_options == _cache_key(model, _request(model))


def test_with_local_serving_off_the_options_stay_out_of_the_key(monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")  # set, but no Ollama answers the call
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    assert _cache_key("gemini-3-pro-preview", _request()) == OLD_GEMINI_KEY


def test_minus_one_gpu_layers_is_kept_in_the_key_and_the_record(monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "-1")
    assert _request()["local_backend_options"] == {"num_gpu": -1}
    assert _cache_key("gemini-3-pro-preview", _request()) != OLD_GEMINI_KEY
    assert L.ollama_options_record() == {"vlm_num_gpu": -1}


def test_a_string_and_an_integer_setting_give_the_same_key(monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    as_text = _cache_key("gemini-3-pro-preview", _request())
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", 1024)
    assert _cache_key("gemini-3-pro-preview", _request()) == as_text


# --- the stage records ---------------------------------------------------------------------------------------------------

def test_the_record_names_only_the_options_that_are_set(monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_LOCAL_BACKEND", "1")
    assert L.ollama_options_record() == {}
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    assert L.ollama_options_record() == {"vlm_num_batch": 1024}
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", None)
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    assert L.ollama_options_record() == {"vlm_num_gpu": 99}
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    assert L.ollama_options_record() == {"vlm_num_batch": 1024, "vlm_num_gpu": 99}


def test_the_record_is_empty_when_local_serving_is_off(monkeypatch):
    monkeypatch.setattr(L, "OLLAMA_NUM_BATCH", "1024")
    monkeypatch.setattr(L, "OLLAMA_NUM_GPU", "99")
    assert L.ollama_options_record() == {}


STAGE_FILES = [
    "3_segment_ground_plane.py",
    "5_decompose_scene.py",
    "6_upsample_object_images.py",
    "9_articulate_objects.py",
    "11_make_objects_sim_ready.py",
]


@pytest.mark.parametrize("stage", STAGE_FILES)
def test_each_vlm_stage_merges_the_options_record_right_after_its_model_names(stage):
    tree = ast.parse((STAGES / stage).read_text())
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "simfoundry.models.local_backends"]
    assert len(imports) == 1
    assert any(alias.name == "ollama_options_record" for alias in imports[0].names)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    names = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == "served_models" for t in n.targets) and isinstance(n.value, ast.Dict)]
    assert len(names) == 1
    merges = [
        n for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update"
        and getattr(n.func.value, "id", None) == "served_models" and not n.keywords and len(n.args) == 1
        and isinstance(n.args[0], ast.Call) and getattr(n.args[0].func, "id", None) == "ollama_options_record"
    ]
    assert len(merges) == 1
    assert not merges[0].args[0].args and not merges[0].args[0].keywords  # no arguments: it reads the environment
    assert merges[0].lineno > names[0].lineno


# --- the run script ------------------------------------------------------------------------------------------------------

def test_the_run_script_exports_both_pins_with_overridable_defaults():
    text = RUN_SCRIPT.read_text()
    assert 'export SIMFOUNDRY_OLLAMA_NUM_BATCH="${SIMFOUNDRY_OLLAMA_NUM_BATCH:-1024}"' in text
    assert 'export SIMFOUNDRY_OLLAMA_NUM_GPU="${SIMFOUNDRY_OLLAMA_NUM_GPU:-99}"' in text


def test_the_run_script_exports_the_image_backend_with_an_overridable_default():
    text = RUN_SCRIPT.read_text()
    assert 'export SIMFOUNDRY_IMAGE_BACKEND="${SIMFOUNDRY_IMAGE_BACKEND:-qwen21}"' in text
    assert "SIMFOUNDRY_IMAGE_BACKEND:-edit2511" not in text


def test_the_run_script_is_valid_bash():
    result = subprocess.run(["bash", "-n", str(RUN_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
