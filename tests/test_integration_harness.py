"""Harness sanity tests: verify the integration/ suite is structurally valid.

These tests do NOT spawn any subprocess; they only static-check the harness files.
"""

from __future__ import annotations

import ast
from pathlib import Path


INTEGRATION_DIR = Path(__file__).parent / "integration"


def test_integration_conftest_exists():
	assert (INTEGRATION_DIR / "conftest.py").is_file()


def test_integration_tests_valid_python():
	test_files = list(INTEGRATION_DIR.glob("test_*.py"))
	assert test_files, "no test_*.py files in tests/integration/"
	for tf in test_files:
		source = tf.read_text(encoding="utf-8")
		try:
			ast.parse(source, filename=str(tf))
		except SyntaxError as exc:  # pragma: no cover
			raise AssertionError(f"{tf} has SyntaxError: {exc}") from exc


def test_full_stack_declares_expected_tests():
	source = (INTEGRATION_DIR / "test_full_stack.py").read_text(encoding="utf-8")
	expected = (
		"test_livez",
		"test_readyz_after_request",
		"test_create_and_list_product",
		"test_openapi_spec",
		"test_pagination_real_http",
		"test_search_endpoint",
		"test_csv_export",
	)
	for name in expected:
		assert f"def {name}(" in source, f"missing test: {name}"


def test_security_integration_declares_expected_tests():
	source = (INTEGRATION_DIR / "test_security_integration.py").read_text(encoding="utf-8")
	for name in ("test_rate_limiting_real_http", "test_production_rejects_default_key"):
		assert f"def {name}(" in source, f"missing test: {name}"
