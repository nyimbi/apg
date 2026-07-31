"""Regression coverage for APG CI and pytest configuration."""

from __future__ import annotations

import configparser
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_github_workflow_exists():
	workflow_path = ROOT / ".github" / "workflows" / "ci.yml"

	assert workflow_path.exists()
	workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

	assert workflow["name"] == "CI"
	assert workflow["on"] == ["push", "pull_request"]
	assert set(workflow["jobs"]) == {"test", "lint"}


def test_workflow_valid_yaml():
	workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
	workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
	assert isinstance(workflow, dict)
	assert "jobs" in workflow


def test_pytest_testpaths_configured():
	"""pyproject.toml must declare testpaths (canonical Wave V check)."""
	pyproject = ROOT / "pyproject.toml"
	assert pyproject.exists()
	try:
		import tomllib
	except ModuleNotFoundError:
		import tomli as tomllib  # type: ignore[no-redef]
	config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
	options = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
	assert "tests" in options.get("testpaths", [])


def test_pytest_config_has_testpaths():
	pytest_ini = ROOT / "pytest.ini"
	if pytest_ini.exists():
		config = configparser.ConfigParser()
		config.read(pytest_ini)
		assert config.has_option("pytest", "testpaths")
		assert "tests" in config.get("pytest", "testpaths").split()
		return

	pyproject = ROOT / "pyproject.toml"
	assert pyproject.exists()

	try:
		import tomllib
	except ModuleNotFoundError:
		tomllib = None

	assert tomllib is not None, "pyproject pytest config requires tomllib"
	config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
	options = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
	assert options.get("testpaths") == ["tests"]
