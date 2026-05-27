"""CLI surface regressions for the Python-first APG workflow."""

from __future__ import annotations

import json

from click.testing import CliRunner

from cli.main import cli


def test_cli_help_lists_supported_python_workflow_commands():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("compile", "create", "doctor", "init", "run", "validate", "version"):
        assert command in result.output
    assert "Flask-AppBuilder" not in result.output
    assert "flask_appbuilder" not in result.output


def test_cli_version_describes_python_artifacts():
    result = CliRunner().invoke(cli, ["version"])

    assert result.exit_code == 0, result.output
    assert "Target Language: Python" in result.output
    assert "Executable Python application artifacts" in result.output
    assert "Flask-AppBuilder" not in result.output


def test_cli_init_creates_python_first_project_config():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["init"])
        config = json.loads(open("apg.json", encoding="utf-8").read())
        source = open("main.apg", encoding="utf-8").read()

    assert result.exit_code == 0, result.output
    assert "generate Python artifacts" in result.output
    assert "python generated/app.py" in result.output
    assert config["target_language"] == "python"
    assert config["build"]["target_language"] == "python"
    assert "agent BasicAgent" in source
    assert "Flask-AppBuilder" not in result.output
