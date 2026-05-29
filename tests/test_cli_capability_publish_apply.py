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
