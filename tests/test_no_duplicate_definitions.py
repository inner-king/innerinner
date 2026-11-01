# SPDX-License-Identifier: Apache-2.0
"""No cutting module binds the same top-level name twice: a pasted-in copy silently wins at import. CPU only."""

import ast
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TARGETS = sorted([*(REPO / "scripts" / "cutting").glob("*.py"), *(REPO / "simfoundry" / "cutting").glob("*.py")])


def _duplicate_definitions(source: str) -> dict[str, list[int]]:
    """Names that more than one top-level ``def``, ``async def`` or ``class`` binds, with the line of each.

    Only the module's own statements count. A definition under ``if`` or ``try`` is a choice between two
    bindings, not a paste, and a method in a class body can share a name legitimately (``@property`` and
    its ``.setter``), so neither is counted.

    Args:
        source: The module's source text.

    Returns:
        Map from each duplicated name to the lines of its definitions, in order. Empty when there are none.
    """
    seen: dict[str, list[int]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen.setdefault(node.name, []).append(node.lineno)
    return {name: lines for name, lines in seen.items() if len(lines) > 1}


def test_targets_are_found():
    assert TARGETS, f"no modules found under {REPO / 'scripts' / 'cutting'} or {REPO / 'simfoundry' / 'cutting'}"


def test_targets_cover_both_cutting_folders():
    folders = {p.parent.relative_to(REPO).as_posix() for p in TARGETS}
    assert folders == {"scripts/cutting", "simfoundry/cutting"}


@pytest.mark.parametrize("path", TARGETS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_duplicate_top_level_definitions(path):
    rel = path.relative_to(REPO).as_posix()
    dups = _duplicate_definitions(path.read_text(encoding="utf-8"))
    assert not dups, "\n".join(f"{rel}: {name} ({', '.join(map(str, lines))})" for name, lines in dups.items())


def test_two_functions_with_one_name_are_reported():
    source = textwrap.dedent(
        """\
        def f():
            pass


        def f():
            pass
        """
    )
    assert _duplicate_definitions(source) == {"f": [1, 5]}


def test_two_classes_with_one_name_are_reported():
    source = "class C:\n    pass\n\n\nclass C:\n    pass\n"
    assert _duplicate_definitions(source) == {"C": [1, 5]}


def test_a_function_and_a_class_with_one_name_are_reported():
    source = "def g():\n    pass\n\n\nclass g:\n    pass\n"
    assert _duplicate_definitions(source) == {"g": [1, 5]}


def test_async_definitions_are_reported():
    source = "async def h():\n    pass\n\n\ndef h():\n    pass\n\n\nasync def k():\n    pass\n\n\nasync def k():\n    pass\n"
    assert _duplicate_definitions(source) == {"h": [1, 5], "k": [9, 13]}


def test_if_else_bindings_are_not_reported():
    source = textwrap.dedent(
        """\
        import sys

        if sys.platform == "win32":
            def f():
                return 1
        else:
            def f():
                return 2

        try:
            def g():
                return 1
        except ImportError:
            def g():
                return 2
        """
    )
    assert _duplicate_definitions(source) == {}


def test_property_and_setter_are_not_reported():
    source = textwrap.dedent(
        """\
        class C:
            @property
            def x(self):
                return 1

            @x.setter
            def x(self, value):
                pass
        """
    )
    assert _duplicate_definitions(source) == {}


def test_nested_definitions_are_not_reported():
    source = "def outer():\n    def inner():\n        pass\n\n    def inner():\n        pass\n    return inner\n"
    assert _duplicate_definitions(source) == {}


def test_assignment_and_definition_with_one_name_are_not_reported():
    assert _duplicate_definitions("f = 1\n\n\ndef f():\n    pass\n") == {}


def test_line_is_the_def_line_not_the_decorator_line():
    source = "@dec\ndef f():\n    pass\n\n\n@dec\ndef f():\n    pass\n"
    assert _duplicate_definitions(source) == {"f": [2, 7]}
