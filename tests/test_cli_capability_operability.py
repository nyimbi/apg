"""CLI coverage for capability inspection and rule execution."""

from __future__ import annotations

import json

from click.testing import CliRunner

from cli.main import cli


def test_capabilities_inspect_exposes_contract_surfaces_as_json():
	result = CliRunner().invoke(
		cli,
		["capabilities", "inspect", "composition_events", "--tenant-id", "tenant-alpha", "--json"],
	)

	assert result.exit_code == 0, result.output
	payload = json.loads(result.output)

	assert payload["format"] == "apg.capability-inspect-report.v1"
	assert payload["ok"] is True
	assert payload["capability"] == "composition_events"
	assert payload["tenant_id"] == "tenant-alpha"
	assert payload["configuration"]["tenant_id"] == "tenant-alpha"
	assert payload["summary"]["rule_count"] >= 1
	assert payload["summary"]["route_count"] >= 1
	assert payload["summary"]["ui_shell"] == "apg_python"
	assert payload["theme"]["tokens"]["border.radius"]


def test_capabilities_evaluate_rules_runs_deterministic_engine_as_json():
	context = json.dumps({
		"tenant_context_present": False,
		"operation_type": "write",
		"policy_attached": False,
	})
	result = CliRunner().invoke(
		cli,
		[
			"capabilities",
			"evaluate-rules",
			"composition_events",
			"--context-json",
			context,
			"--json",
		],
	)

	assert result.exit_code == 0, result.output
	payload = json.loads(result.output)

	assert payload["format"] == "apg.capability-rule-evaluation-report.v1"
	assert payload["ok"] is True
	assert payload["decision"] == "deny"
	assert "tenant_context_required" in payload["matched_rules"]
	assert payload["context"]["operation_type"] == "write"


def test_capabilities_evaluate_rules_reads_context_file():
	runner = CliRunner()
	with runner.isolated_filesystem():
		with open("context.json", "w", encoding="utf-8") as handle:
			json.dump({"risk_level": "high", "review_recorded": False}, handle)
		result = runner.invoke(
			cli,
			[
				"capabilities",
				"evaluate-rules",
				"composition_events",
				"--context-file",
				"context.json",
				"--json",
			],
		)

	assert result.exit_code == 0, result.output
	payload = json.loads(result.output)
	assert payload["decision"] == "require_review"
	assert "high_risk_requires_review" in payload["matched_rules"]


def test_capabilities_evaluate_rules_rejects_invalid_context_shape():
	result = CliRunner().invoke(
		cli,
		[
			"capabilities",
			"evaluate-rules",
			"composition_events",
			"--context-json",
			"[]",
			"--json",
		],
	)

	assert result.exit_code == 1
	payload = json.loads(result.output)
	assert payload["ok"] is False
	assert payload["errors"] == ["context JSON must be an object"]
