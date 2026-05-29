"""CLI coverage for capability implementation-depth auditing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from cli.main import cli


PACKAGE_ARTIFACTS = [
	"cap_spec.md",
	"models.py",
	"service.py",
	"api.py",
	"views.py",
	"app.py",
	"semantic_model.json",
	"package_manifest.json",
	"release_report.json",
]


def _materialized_demo_package(runner: CliRunner) -> Path:
	scaffold = runner.invoke(
		cli,
		[
			"capabilities",
			"scaffold",
			"common",
			"demo",
			"--name",
			"Demo Implementation",
			"--json",
		],
	)
	assert scaffold.exit_code == 0, scaffold.output
	package_dir = Path(json.loads(scaffold.output)["path"])
	for artifact in PACKAGE_ARTIFACTS:
		path = package_dir / artifact
		if path.exists():
			path.unlink()
	tests_dir = package_dir / "tests"
	if tests_dir.exists():
		shutil.rmtree(tests_dir)
	materialize = runner.invoke(
		cli,
		[
			"capabilities",
			"materialize-packages",
			"--root",
			"capabilities",
			"--capability",
			"common_demo",
			"--json",
		],
	)
	assert materialize.exit_code == 0, materialize.output
	return package_dir


def test_capabilities_implementation_audit_reports_materialized_baseline():
	runner = CliRunner()
	with runner.isolated_filesystem():
		_materialized_demo_package(runner)
		result = runner.invoke(
			cli,
			[
				"capabilities",
				"implementation-audit",
				"--root",
				"capabilities",
				"--json",
			],
		)

	assert result.exit_code == 0, result.output
	payload = json.loads(result.output)
	assert payload["format"] == "apg.capability-implementation-audit.v1"
	assert payload["ok"] is True
	assert payload["summary"]["capability_count"] == 1
	assert payload["summary"]["materialized_baseline_count"] == 1
	assert payload["summary"]["warning_count"] == 1
	assert payload["records"][0]["implementation_level"] == "materialized_baseline"
	assert payload["records"][0]["baseline_marker_count"] >= 1


def test_capabilities_implementation_audit_strict_blocks_baseline_packages():
	runner = CliRunner()
	with runner.isolated_filesystem():
		_materialized_demo_package(runner)
		result = runner.invoke(
			cli,
			[
				"capabilities",
				"implementation-audit",
				"--root",
				"capabilities",
				"--strict",
				"--json",
			],
		)

	assert result.exit_code == 1
	payload = json.loads(result.output)
	assert payload["ok"] is False
	assert payload["summary"]["materialized_baseline_count"] == 1
	assert payload["blocking_gaps"]
