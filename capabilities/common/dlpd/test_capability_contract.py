"""Regression coverage for the DLPD executable capability contract."""

from capabilities.common.dlpd import register_capability
from capabilities.common.dlpd.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-dlp", {"channels": {"bulk_export_threshold_records": 5000}})

	assert contract["capability"] == "dlpd"
	assert contract["configuration"]["tenant_id"] == "tenant-dlp"
	assert contract["configuration"]["channels"]["bulk_export_threshold_records"] == 5000
	assert contract["configuration_schema"]["required"] == ["tenant_id", "data_patterns", "channels", "response", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "policies", "classifiers", "channels", "incidents", "quarantine", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/dlpd/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "incident_queue" in contract["theme"]["components"]


def test_rule_engine_enforces_dlp_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "inspect_egress",
		"egress_policy_attached": False,
		"sensitive_content_detected": True,
		"classification_label_present": False,
		"severity": "high",
		"blocked_or_quarantined": False,
		"quarantine_requested": True,
		"quarantine_encrypted": False,
		"export_record_count": 20000,
		"review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"inspection_source_requires_policy",
		"sensitive_content_requires_classification",
		"high_severity_exfiltration_requires_block",
		"quarantine_requires_encryption",
		"large_export_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "dlpd"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "dlpd_data_protection_ops"
	assert registration["ui_components"]["incidents"] == "/dlpd/incidents"
	assert "nlpc" in registration["dependencies"]
	assert "dlpd:respond" in registration["permissions"]
