"""CLI coverage for APG capability scaffolding."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from click.testing import CliRunner

from capabilities.capability_contract_registry import validate_contract_shape
from cli.main import cli


def _load_contract(path: Path):
	spec = importlib.util.spec_from_file_location("test_scaffolded_contract", path)
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

	assert report["format"] == "apg.capability-scaffold-report.v1"
	assert report["ok"] is True
	assert report["capability"] == "common_demo"
	assert len(report["written"]) == 9
	assert contract_path.name == "capability_contract.py"
	validate_contract_shape(contract, contract_path)
	assert contract["capability"] == "common_demo"
	assert contract["display_name"] == "Demo Capacity"
	assert contract["configuration"]["tenant_id"] == "tenant-demo"
	assert rule_result["decision"] == "deny"
	assert "tenant_context_required" in rule_result["matched_rules"]


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
