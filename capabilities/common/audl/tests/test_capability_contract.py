"""Regression coverage for the AUDL executable capability contract."""

from .. import get_capability_info
from ..capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-a",
		{"retention": {"archive_after_days": 180}}
	)

	assert contract["capability"] == "audl"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration"]["retention"]["archive_after_days"] == 180
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"ingestion",
		"retention",
		"compliance",
		"investigations",
		"notifications",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"events",
		"timeline",
		"investigations",
		"compliance",
		"reports",
		"rules",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/api/v1/audit"
	assert contract["theme"]["tokens"]["border.radius"] == "10px"
	assert "audit_timeline" in contract["theme"]["components"]
	assert "compliance_scorecard" in contract["theme"]["components"]


def test_rule_engine_denies_unsafe_audit_operations():
	result = evaluate_capability_rules({
		"tenant_id_missing": True,
		"immutable_storage": True,
		"checksum_verified": False,
		"requested_operation": "export",
		"contains_pii": True,
		"masking_enabled": False,
		"event_severity": "critical",
		"escalation_configured": False,
		"batch_size": 20000,
		"stream_processing_enabled": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"require_tenant_context",
		"immutable_events_require_checksum",
		"regulated_exports_require_masking",
		"critical_events_require_escalation",
		"high_volume_ingestion_requires_stream_processing",
	}


def test_capability_info_includes_manifest_and_theme():
	info = get_capability_info()

	assert info["metadata"]["capability_id"] == "common/audl"
	assert info["configuration"]["tenant_id"] == "default"
	assert info["rule_engine"]["type"] == "deterministic"
	assert info["ui_manifest"]["requires_theme"] is True
	assert info["theme"]["name"] == "audl_forensics"
	assert {route["name"] for route in info["ui_manifest"]["routes"]} >= {
		"timeline",
		"investigations",
		"reports",
	}
