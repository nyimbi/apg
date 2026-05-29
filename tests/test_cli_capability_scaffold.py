"""CLI coverage for APG capability scaffolding."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

from click.testing import CliRunner

from capabilities.capability_contract_registry import validate_contract_shape
from cli.main import cli
from compiler.capability_publish import build_capability_publish_report


def _load_contract(path: Path):
	spec = importlib.util.spec_from_file_location("test_scaffolded_contract", path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def _load_scaffold_module(package_dir: Path, module_name: str):
	package_name = "test_scaffolded_common_demo"
	if package_name not in sys.modules:
		package = types.ModuleType(package_name)
		package.__path__ = [str(package_dir)]
		sys.modules[package_name] = package
	spec = importlib.util.spec_from_file_location(
		f"{package_name}.{module_name}",
		package_dir / f"{module_name}.py",
	)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def test_capabilities_scaffold_creates_valid_spec_backed_package():
	runner = CliRunner()
	with runner.isolated_filesystem():
		result = runner.invoke(
			cli,
			[
				"capabilities",
				"scaffold",
				"common",
				"demo",
				"--name",
				"Demo Capacity",
				"--json",
			],
		)
		assert result.exit_code == 0, result.output
		report = json.loads(result.output)
		capability_dir = Path(report["path"])
		contract_path = capability_dir / "capability_contract.py"
		module = _load_contract(contract_path)
		contract = module.get_capability_contract("tenant-demo")
		rule_result = module.evaluate_capability_rules({"tenant_context_present": False})
		service_module = _load_scaffold_module(capability_dir, "service")
		api_module = _load_scaffold_module(capability_dir, "api")
		views_module = _load_scaffold_module(capability_dir, "views")
		service = service_module.CommonDemoService()
		record = service.create_record("demo-1", "tenant-demo", {"priority": "high"})
		updated = service.update_status("demo-1", "approved", tenant_id="tenant-demo")
		view_model = views_module.dashboard_model(service, tenant_id="tenant-demo")
		api_record = api_module.create_record({"id": "api-1", "tenant_id": "tenant-demo"})
		api_status = api_module.capability_status("tenant-demo")
		publish_report = build_capability_publish_report(capability_dir)
		package_artifacts_exist = all(
			(capability_dir / artifact).is_file()
			for artifact in ("app.py", "semantic_model.json", "package_manifest.json", "release_report.json")
		)

	assert report["format"] == "apg.capability-scaffold-report.v1"
	assert report["ok"] is True
	assert report["capability"] == "common_demo"
	assert len(report["written"]) == 13
	assert contract_path.name == "capability_contract.py"
	assert package_artifacts_exist is True
	validate_contract_shape(contract, contract_path)
	assert contract["capability"] == "common_demo"
	assert contract["display_name"] == "Demo Capacity"
	assert contract["configuration"]["tenant_id"] == "tenant-demo"
	assert rule_result["decision"] == "deny"
	assert "tenant_context_required" in rule_result["matched_rules"]
	assert record["metadata"]["priority"] == "high"
	assert updated["status"] == "approved"
	assert view_model["records"][0]["id"] == "demo-1"
	assert api_record["id"] == "api-1"
	assert api_status["record_count"] == 1
	assert publish_report["format"] == "apg.capability-publish-report.v1"
	assert publish_report["ok"] is True
	assert publish_report["capabilities"][0]["capability"] == "common_demo"
	assert publish_report["runtime_evidence"]["self_test"]["passed"] is True


def test_capabilities_scaffold_refuses_to_overwrite_without_force():
	runner = CliRunner()
	with runner.isolated_filesystem():
		first = runner.invoke(cli, ["capabilities", "scaffold", "common", "demo", "--json"])
		second = runner.invoke(cli, ["capabilities", "scaffold", "common", "demo", "--json"])
		forced = runner.invoke(cli, ["capabilities", "scaffold", "common", "demo", "--force", "--json"])

	assert first.exit_code == 0, first.output
	assert second.exit_code == 1, second.output
	second_report = json.loads(second.output)
	assert second_report["format"] == "apg.capability-scaffold-report.v1"
	assert second_report["ok"] is False
	assert second_report["skipped"]
	assert "target files already exist" in second_report["errors"][0]
	assert forced.exit_code == 0, forced.output
