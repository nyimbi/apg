"""Wave K generated pytest scaffold coverage."""

from __future__ import annotations

import ast

from compiler.compiler import compile_apg_string


SCAFFOLD_SOURCE = """
module scaffold_probe version 1.0.0 {}

table Customer {
    name: str;
    email: str;
    visits: int;
}
"""


def _compile_scaffold_probe():
    result = compile_apg_string(SCAFFOLD_SOURCE)
    assert result.success, result.errors
    return result


def test_generated_files_include_module_test_key():
    result = _compile_scaffold_probe()

    assert "tests/test_scaffold_probe.py" in result.generated_files
    assert "tests/conftest.py" in result.generated_files


def test_generated_tests_are_valid_python():
    result = _compile_scaffold_probe()

    ast.parse(result.generated_files["tests/conftest.py"], filename="tests/conftest.py")
    ast.parse(result.generated_files["tests/test_scaffold_probe.py"], filename="tests/test_scaffold_probe.py")


def test_generated_tests_define_test_functions():
    result = _compile_scaffold_probe()
    tree = ast.parse(result.generated_files["tests/test_scaffold_probe.py"])

    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "test_smoke_livez" in names
    assert "test_crud_customer" in names
    assert "test_search_customer" in names
