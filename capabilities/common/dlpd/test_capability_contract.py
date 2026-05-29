"""Regression coverage for the DLPD executable capability contract."""

import pytest

from capabilities.common.dlpd import register_capability
from capabilities.common.dlpd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.dlpd.service import DlpdService


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


def test_service_runs_sensitive_egress_quarantine_and_incident_lifecycle():
	service = DlpdService()

	classifier = service.register_classifier(
		classifier_id="cls-secrets",
		tenant_id="tenant-dlp",
		name="Secrets and PCI",
		classifier_type="built_in",
		sensitivity_label="restricted",
		pattern_keys=["secrets", "pci"],
	)
	policy = service.register_policy(
		policy_id="pol-email",
		tenant_id="tenant-dlp",
		name="Email exfiltration prevention",
		owner="security-ops",
		channels=["email", "api_export"],
		classifiers=[classifier["id"]],
		default_action="quarantine",
	)
	inspection = service.inspect_egress(
		inspection_id="insp-1",
		tenant_id="tenant-dlp",
		policy_id=policy["id"],
		channel="email",
		subject_id="user-1",
		destination="external@example.com",
		content="api_key='SECRET123456789' card 4111 1111 1111 1111",
	)

	assert inspection["decision"] == "quarantine"
	assert inspection["classification_label"] == "restricted"
	assert inspection["quarantined"] is True
	assert inspection["quarantine_id"] == "qrn-insp-1"
	assert inspection["incident_id"] == "inc-insp-1"
	assert service.list_quarantine("tenant-dlp")[0]["encrypted"] is True
	assert service.list_incidents("tenant-dlp")[0]["owner"] == "security-ops"

	resolved = service.resolve_incident("inc-insp-1", "tenant-dlp", "analyst-1", "false positive removed from export")
	assert resolved["status"] == "resolved"
	assert service.dashboard_summary("tenant-dlp")["quarantine_count"] == 1
	assert service.list_audit_events("tenant-dlp")


def test_service_enforces_dlp_guardrails():
	service = DlpdService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_policy("pol-missing-tenant", "", "Missing tenant", "owner", ["email"], [])
	with pytest.raises(PermissionError, match="incident_owner_required"):
		service.register_policy("pol-owner", "tenant-dlp", "Missing owner", "", ["email"], [])
	with pytest.raises(PermissionError, match="egress_policy_required"):
		service.register_policy("pol-policy", "tenant-dlp", "Missing egress policy", "owner", ["email"], [], egress_policy_attached=False)
	with pytest.raises(PermissionError, match="custom_pattern_review_required"):
		service.register_classifier("cls-custom", "tenant-dlp", "Unreviewed custom", "custom", "restricted", ["secrets"])

	service.register_classifier("cls-pii", "tenant-dlp", "PII", "built_in", "confidential", ["pii"])
	service.register_policy("pol-chat", "tenant-dlp", "Chat policy", "owner", ["chat"], ["cls-pii"], default_action="allow")
	with pytest.raises(PermissionError, match="channel_not_covered_by_policy"):
		service.inspect_egress(
			inspection_id="insp-wrong-channel",
			tenant_id="tenant-dlp",
			policy_id="pol-chat",
			channel="email",
			subject_id="user-1",
			destination="external@example.com",
			content="alice@example.com",
		)
	with pytest.raises(PermissionError, match="classification_label_required"):
		service.inspect_egress(
			inspection_id="insp-unlabeled",
			tenant_id="tenant-dlp",
			policy_id="pol-chat",
			channel="chat",
			subject_id="user-1",
			destination="external-room",
			content="alice@example.com",
			auto_classify=False,
		)


def test_large_export_review_and_high_severity_block_rules_are_executable():
	service = DlpdService()
	service.register_classifier("cls-secrets", "tenant-dlp", "Secrets", "built_in", "restricted", ["secrets"])
	service.register_policy(
		"pol-alert",
		"tenant-dlp",
		"Alert only",
		"owner",
		["api_export"],
		["cls-secrets"],
		default_action="alert",
	)

	with pytest.raises(PermissionError, match="high_severity_block_required"):
		service.inspect_egress(
			inspection_id="insp-alert",
			tenant_id="tenant-dlp",
			policy_id="pol-alert",
			channel="api_export",
			subject_id="user-1",
			destination="external-api",
			content="secret='SECRET123456789'",
		)

	service.register_policy(
		"pol-review",
		"tenant-dlp",
		"Review large exports",
		"owner",
		["api_export"],
		["cls-secrets"],
		default_action="allow",
	)
	review = service.inspect_egress(
		inspection_id="insp-review",
		tenant_id="tenant-dlp",
		policy_id="pol-review",
		channel="api_export",
		subject_id="user-2",
		destination="warehouse-export",
		content="ordinary account number export",
		record_count=20000,
	)
	assert review["decision"] == "require_review"
	assert review["review_required"] is True

	reviewed = service.review_export("insp-review", "tenant-dlp", "reviewer-1")
	assert reviewed["decision"] == "reviewed"
	assert reviewed["reviewed_by"] == "reviewer-1"
