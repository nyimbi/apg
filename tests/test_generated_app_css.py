"""Generated app CSS regressions."""

from __future__ import annotations

from compiler.compiler import APGCompiler


SIMPLE_SOURCE = """
module generated_app_css version 1.0.0 {}

table Customer {
    name: str;
}
"""


def _generated_app_source() -> str:
	result = APGCompiler().compile_string(SIMPLE_SOURCE, "generated_app_css")
	assert result.success, result.errors
	return result.generated_files["app.py"]


def test_dark_mode_css_present():
	app_source = _generated_app_source()

	assert "@media (prefers-color-scheme: dark)" in app_source


def test_print_css_present():
	app_source = _generated_app_source()

	assert "@media print" in app_source


def test_mobile_breakpoint_present():
	app_source = _generated_app_source()

	assert "max-width: 768px" in app_source
