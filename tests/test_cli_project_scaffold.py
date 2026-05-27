"""Focused regressions for APG project scaffolding."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from compiler.parser import APGParser


CLI_PATH = Path(__file__).resolve().parents[1] / "cli.py"
SPEC = importlib.util.spec_from_file_location("apg_root_cli_scaffold", CLI_PATH)
assert SPEC is not None
assert SPEC.loader is not None
CLI_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["apg_root_cli_scaffold"] = CLI_MODULE
SPEC.loader.exec_module(CLI_MODULE)

APGCLICommands = CLI_MODULE.APGCLICommands


def test_init_project_scaffolds_executable_workflow_steps(tmp_path, monkeypatch, capsys):
	monkeypatch.chdir(tmp_path)
	cli = APGCLICommands()

	assert cli.init_project("sample_app") is True
	capsys.readouterr()

	source_file = tmp_path / "sample_app" / "src" / "app.apg"
	source = source_file.read_text()
	config = json.loads((tmp_path / "sample_app" / "apg.json").read_text())
	readme = (tmp_path / "sample_app" / "README.md").read_text()

	assert "TODO: Implement step logic" not in source
	assert "if (step_name in steps)" in source
	assert "return false;" in source
	assert config["target"] == "python"
	assert "python generated/app.py" in readme
	assert "Flask-AppBuilder" not in readme
	assert "http://localhost:8080" not in readme

	result = APGParser().parse_file(str(source_file))

	assert result["success"] is True
	assert result["errors"] == []
