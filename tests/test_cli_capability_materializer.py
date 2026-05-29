"""CLI coverage for materializing package artifacts from capability contracts."""

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


def _scaffold_then_strip_package(runner: CliRunner) -> Path:
	result = runner.invoke(
		cli,
		[
			"capabilities",
			"scaffold",
			"common",
			"demo",
			"--name",
			"Demo Materialized",
			"--json",
		],
	)
	assert result.exit_code == 0, result.output
	package_dir = Path(json.loads(result.output)["path"])
	for artifact in PACKAGE_ARTIFACTS:
		path = package_dir / artifact
		if path.exists():
			path.unlink()
	tests_dir = package_dir / "tests"
	if tests_dir.exists():
		shutil.rmtree(tests_dir)
	return package_dir


def test_capabilities_materialize_packages_writes_publishable_artifacts():
	runner = CliRunner()
	with runner.isolated_filesystem():
		package_dir = _scaffold_then_strip_package(runner)
		dry_run = runner.invoke(
			cli,
			[
				"capabilities",
				"materialize-packages",
				"--root",
				"capabilities",
				"--capability",
				"common_demo",
				"--dry-run",
				"--json",
			],
		)
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
		publish_plan = runner.invoke(
			cli,
			["capabilities", "publish-plan", str(package_dir), "--json"],
		)
		app_exists_after_materialize = (package_dir / "app.py").exists()
		package_test_exists = (package_dir / "tests" / "test_materialized_package.py").exists()

	assert dry_run.exit_code == 0, dry_run.output
	dry_payload = json.loads(dry_run.output)
	assert dry_payload["format"] == "apg.capability-package-materialization.v1"
	assert dry_payload["dry_run"] is True
	assert dry_payload["would_write_count"] >= len(PACKAGE_ARTIFACTS)
	assert (package_dir / "app.py").exists() is False

	assert materialize.exit_code == 0, materialize.output
	payload = json.loads(materialize.output)
	assert payload["ok"] is True
	assert payload["package_count"] == 1
	assert payload["written_count"] >= len(PACKAGE_ARTIFACTS)
	assert app_exists_after_materialize is True
	assert package_test_exists is True

	assert publish_plan.exit_code == 0, publish_plan.output
	plan_payload = json.loads(publish_plan.output)
	assert plan_payload["ok"] is True
	assert plan_payload["capabilities"][0]["capability"] == "common_demo"
	assert plan_payload["runtime_evidence"]["self_test"]["passed"] is True


def test_capabilities_materialize_packages_reports_unknown_capability():
	runner = CliRunner()
	with runner.isolated_filesystem():
		_scaffold_then_strip_package(runner)
		result = runner.invoke(
			cli,
			[
				"capabilities",
				"materialize-packages",
				"--root",
				"capabilities",
				"--capability",
				"missing_demo",
				"--json",
			],
		)

	assert result.exit_code == 1
	payload = json.loads(result.output)
	assert payload["ok"] is False
	assert payload["errors"] == ["unknown capability contract: missing_demo"]
