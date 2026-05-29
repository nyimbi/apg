"""CLI coverage for applying capability publish plans to a local catalog."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from cli.main import cli


def _scaffold_demo_package(runner: CliRunner) -> Path:
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
	return Path(json.loads(result.output)["path"])


def test_capabilities_publish_apply_dry_run_does_not_write_catalog():
	runner = CliRunner()
	with runner.isolated_filesystem():
		package_dir = _scaffold_demo_package(runner)
		catalog_path = Path("catalog/capabilities.json")
		result = runner.invoke(
			cli,
			[
				"capabilities",
				"publish-apply",
				str(package_dir),
				"--catalog",
				str(catalog_path),
				"--dry-run",
				"--json",
			],
		)
		catalog_exists = catalog_path.exists()

	assert result.exit_code == 0, result.output
	payload = json.loads(result.output)
	assert payload["format"] == "apg.capability-publish-apply-report.v1"
	assert payload["ok"] is True
	assert payload["dry_run"] is True
	assert payload["written"] is False
	assert payload["capabilities"] == ["common_demo"]
	assert catalog_exists is False


def test_capabilities_publish_apply_writes_local_catalog():
	runner = CliRunner()
	with runner.isolated_filesystem():
		package_dir = _scaffold_demo_package(runner)
		catalog_path = Path("catalog/capabilities.json")
		result = runner.invoke(
			cli,
			[
				"capabilities",
				"publish-apply",
				str(package_dir),
				"--catalog",
				str(catalog_path),
				"--json",
			],
		)
		catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
		reapply = runner.invoke(
			cli,
			[
				"capabilities",
				"publish-apply",
				str(package_dir),
				"--catalog",
				str(catalog_path),
				"--json",
			],
		)
		catalog_after_reapply = json.loads(catalog_path.read_text(encoding="utf-8"))
		catalog_report = runner.invoke(
			cli,
			["capabilities", "catalog", str(catalog_path), "--json"],
		)
		capability_report = runner.invoke(
			cli,
			["capabilities", "catalog", str(catalog_path), "--capability", "common_demo", "--json"],
		)
		missing_report = runner.invoke(
			cli,
			["capabilities", "catalog", str(catalog_path), "--capability", "missing_demo", "--json"],
		)

	assert result.exit_code == 0, result.output
	payload = json.loads(result.output)
	assert payload["ok"] is True
	assert payload["written"] is True
	assert payload["applied_count"] == 1
	assert payload["catalog_summary"]["capability_count_after"] == 1
	assert catalog["format"] == "apg.capability-catalog.v1"
	assert catalog["capabilities"]["common_demo"]["entrypoint"] == "app.py"
	assert catalog["capabilities"]["common_demo"]["rule_engine"]["type"] == "deterministic"
	assert reapply.exit_code == 0, reapply.output
	reapply_payload = json.loads(reapply.output)
	assert reapply_payload["catalog_summary"]["capability_count_before"] == 1
	assert reapply_payload["catalog_summary"]["capability_count_after"] == 1
	assert catalog_after_reapply == catalog
	assert catalog_report.exit_code == 0, catalog_report.output
	catalog_payload = json.loads(catalog_report.output)
	assert catalog_payload["format"] == "apg.capability-catalog-report.v1"
	assert catalog_payload["ok"] is True
	assert catalog_payload["capability_count"] == 1
	assert catalog_payload["records"][0]["capability"] == "common_demo"
	assert catalog_payload["records"][0]["route_count"] == 1
	assert catalog_payload["records"][0]["rule_count"] == 2
	assert capability_report.exit_code == 0, capability_report.output
	capability_payload = json.loads(capability_report.output)
	assert capability_payload["records"][0]["package"] == "common_demo"
	assert missing_report.exit_code == 1
	missing_payload = json.loads(missing_report.output)
	assert missing_payload["ok"] is False
	assert missing_payload["errors"] == ["capability not found in catalog: missing_demo"]


def test_publish_apply_catalog_feeds_lint_capability_resolution():
	runner = CliRunner()
	with runner.isolated_filesystem():
		package_dir = _scaffold_demo_package(runner)
		catalog_path = Path("catalog/capabilities.json")
		publish = runner.invoke(
			cli,
			[
				"capabilities",
				"publish-apply",
				str(package_dir),
				"--catalog",
				str(catalog_path),
				"--json",
			],
		)
		source = Path("app.apg")
		source.write_text(
			"""
module local_catalog_lint version 1.0.0 {
	description: "Local catalog lint proof";
}

capability Demo {
	contract: {
		id: common_demo,
		provides: [common_demo_records],
		configuration: {tenant_scoped: true}
	};
}
""",
			encoding="utf-8",
		)
		lint = runner.invoke(
			cli,
			["lint", str(source), "--catalog", str(catalog_path), "--json"],
		)

	assert publish.exit_code == 0, publish.output
	assert lint.exit_code == 0, lint.output
	payload = json.loads(lint.output)
	assert payload["ok"] is True
	assert payload["capability_catalog"]["catalog_kind"] == "local_catalog"
	assert payload["capability_catalog"]["matched_capabilities"] == [
		{"name": "Demo", "matched_key": "common_demo"}
	]
