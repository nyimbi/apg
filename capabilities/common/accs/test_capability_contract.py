"""Regression coverage for the ACCS executable capability contract."""

from capabilities.common.accs import register_capability
from capabilities.common.accs.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-accs", {"standards": {"default_standard": "EN-301-549"}})

	assert contract["capability"] == "accs"
	assert contract["configuration"]["tenant_id"] == "tenant-accs"
	assert contract["configuration"]["standards"]["default_standard"] == "EN-301-549"
	assert contract["configuration_schema"]["required"] == ["tenant_id", "standards", "audits", "assistive", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "audits", "findings", "remediation", "assistive", "media", "compliance", "settings"}
	assert contract["theme"]["name"] == "accs_accessibility_ops"


def test_rule_engine_enforces_accs_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "start_audit", "standard_selected": False, "violation_detected": True, "remediation_owner_assigned": False, "published_ui": True, "contrast_passed": False, "media_content_present": True, "captions_available": False, "issue_severity": "critical", "review_recorded": False})
	review_result = evaluate_capability_rules({"tenant_context_present": True, "issue_severity": "critical", "review_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "audit_requires_standard", "violation_requires_remediation_owner", "published_ui_requires_contrast", "media_requires_captions", "critical_issue_requires_review"}
	assert review_result["decision"] == "require_review"


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "accs"
	assert "nlpc" in registration["dependencies"]
	assert registration["ui_components"]["remediation"] == "/accs/remediation"
	assert "accs:remediate" in registration["permissions"]
