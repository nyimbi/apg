"""Focused compiler baseline regressions for documented APG invocation."""

from __future__ import annotations

from click.testing import CliRunner

from cli.main import cli
from compiler.code_generator import CodeGenerator
from compiler.compiler import APGCompiler, compile_apg_string
from compiler.semantic_analyzer import SemanticError


MINIMAL_AGENT_SOURCE = """
module baseline version 1.0.0 {
	description: "Compiler baseline";
}

agent Planner {
	role: "planner";
	model: "openai:gpt-4.1-mini";
	runtime: codex;
	system: "Plan concrete work.";
}
"""


def test_documented_python_target_generates_executable_application_files():
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)

	assert result.success is True
	assert result.target_language == "python"
	assert "app.py" in result.generated_files
	assert "ai_agents.py" in result.generated_files


def test_cli_compile_default_target_writes_generated_application(tmp_path):
	source = tmp_path / "baseline.apg"
	output = tmp_path / "generated"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["compile", str(source), "--output", str(output), "--verbose"])

	assert result.exit_code == 0, result.output
	assert "Compilation successful" in result.output
	assert (output / "app.py").exists()
	assert (output / "ai_agents.py").exists()


def test_cli_doctor_recognizes_spec_parser_artifacts():
	result = CliRunner().invoke(cli, ["doctor"])

	assert result.exit_code == 0, result.output
	assert "Generated parser found" in result.output
	assert "flask-appbuilder" not in result.output
	assert "django" not in result.output


def test_cli_version_advertises_python_target_not_framework_target():
	result = CliRunner().invoke(cli, ["version"])

	assert result.exit_code == 0, result.output
	assert "Target Language: Python" in result.output
	assert "Executable Python application artifacts" in result.output
	assert "Flask-AppBuilder" not in result.output
	assert "Django" not in result.output


def test_compiler_error_rendering_handles_internal_node_less_errors():
	error = SemanticError("Unsupported target language: bad-target", None, "internal")

	assert str(error) == "unknown:0:0: internal error: Unsupported target language: bad-target"


def test_python_is_the_only_advertised_compiler_target():
	help_result = CliRunner().invoke(cli, ["compile", "--help"])

	assert help_result.exit_code == 0, help_result.output
	assert "[python]" in help_result.output
	assert "flask-appbuilder" not in help_result.output
	assert "django" not in help_result.output
	assert "fastapi" not in help_result.output
	assert APGCompiler().get_supported_targets() == ["python"]
	assert CodeGenerator.normalize_target("python") == "python"


def test_framework_names_are_not_silent_compiler_target_aliases():
	result = CliRunner().invoke(cli, [
		"compile",
		"baseline.apg",
		"--target",
		"flask-appbuilder",
	])

	assert result.exit_code != 0
	assert "Invalid value for '--target'" in result.output
