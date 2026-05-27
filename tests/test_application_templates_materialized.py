"""Executable coverage for legacy application templates."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "templates" / "application_templates"
GENERATOR_PATH = REPO_ROOT / "scripts" / "template_generation" / "create_template_structure.py"
PLACEHOLDER_MARKERS = (
	"TODO: Implement",
	"This file will contain the complete implementation",
)
TEMPLATE_VALUES = {
	"project_name": "sample_project",
	"project_description": "Sample generated project",
	"author": "APG",
	"database_url": "sqlite:///sample.db",
	"secret_key": "dev-secret",
}


def _template_dirs() -> list[Path]:
	return sorted(path.parent for path in TEMPLATE_ROOT.glob("*/*/template.json"))


def _render_template(content: str) -> str:
	for key, value in TEMPLATE_VALUES.items():
		content = content.replace(f"{{{{{key}}}}}", value)
	return content


def _load_template_generator():
	spec = importlib.util.spec_from_file_location("apg_template_structure_generator", GENERATOR_PATH)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules["apg_template_structure_generator"] = module
	spec.loader.exec_module(module)
	return module


def test_application_templates_are_materialized_and_registered():
	assert _template_dirs()

	for template_dir in _template_dirs():
		metadata = json.loads((template_dir / "template.json").read_text())
		assert metadata["target"] == "python"
		assert metadata["requirements"] == []
		registered = set(metadata["files"])
		discovered = {
			str(path.relative_to(template_dir))
			for path in template_dir.rglob("*.template")
		}

		assert registered == discovered, template_dir

		for relative_path in registered:
			content = (template_dir / relative_path).read_text()
			assert not any(marker in content for marker in PLACEHOLDER_MARKERS), relative_path
			if relative_path.endswith(".py.template"):
				compile(_render_template(content), str(template_dir / relative_path), "exec")


def test_shipping_tracker_template_generates_executable_project(tmp_path, monkeypatch):
	template_dir = TEMPLATE_ROOT / "logistics" / "shipping_tracker"
	metadata = json.loads((template_dir / "template.json").read_text())

	for relative_path in metadata["files"]:
		source = template_dir / relative_path
		target = tmp_path / relative_path.replace(".template", "")
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_text(_render_template(source.read_text()))

	monkeypatch.syspath_prepend(str(tmp_path))
	for module_name in ["app", "agents", "models", "views", "tests", "config"]:
		sys.modules.pop(module_name, None)

	app = importlib.import_module("app")
	generated_tests = importlib.import_module("tests")

	assert app.health_check()["status"] == "ready"
	assert app.health_check()["template"] == "logistics/shipping_tracker"
	assert generated_tests.smoke_test() is True


def test_template_structure_generator_emits_executable_starters(tmp_path, monkeypatch):
	generator = _load_template_generator()
	metadata = {
		"name": "Generated Monitor",
		"description": "Generated monitor starter",
		"complexity": "Intermediate",
		"domain": "Operations",
		"agents": ["MonitorAgent"],
		"digital_twins": ["MonitorTwin"],
		"features": ["Signal Tracking", "Alert Routing"],
		"databases": ["signals", "alerts"],
	}
	template_dir = tmp_path / "generated_monitor"

	generator.create_template_directories(template_dir, metadata)
	generator.create_template_json(template_dir, "ops/generated_monitor", metadata)
	generator.create_template_files(template_dir, metadata)

	metadata_file = json.loads((template_dir / "template.json").read_text())
	assert metadata_file["target"] == "python"
	assert metadata_file["requirements"] == []
	assert set(metadata_file["files"]) == {
		str(path.relative_to(template_dir))
		for path in template_dir.rglob("*.template")
	}

	for relative_path in metadata_file["files"]:
		content = (template_dir / relative_path).read_text()
		assert not any(marker in content for marker in PLACEHOLDER_MARKERS), relative_path
		if relative_path.endswith(".py.template"):
			compile(_render_template(content), str(template_dir / relative_path), "exec")

	for relative_path in metadata_file["files"]:
		source = template_dir / relative_path
		target = tmp_path / "materialized" / relative_path.replace(".template", "")
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_text(_render_template(source.read_text()))

	monkeypatch.syspath_prepend(str(tmp_path / "materialized"))
	for module_name in ["app", "agents", "models", "views", "tests", "config", "digital_twins"]:
		sys.modules.pop(module_name, None)

	app = importlib.import_module("app")
	generated_tests = importlib.import_module("tests")

	assert app.health_check()["status"] == "ready"
	assert generated_tests.smoke_test() is True
