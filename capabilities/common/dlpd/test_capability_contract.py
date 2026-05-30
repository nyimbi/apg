"""Regression coverage for the DLPD executable capability contract."""

import pytest

from capabilities.common.dlpd import api, register_capability
from capabilities.common.dlpd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.dlpd.service import DlpdService
from capabilities.common.dlpd.views import (
	analytics_model,
	audit_model,
	channel_monitor_model,
	classifier_workbench_model,
	dashboard_model,
	incident_queue_model,
	inspection_workbench_model,
	legal_hold_model,
	policy_console_model,
	quarantine_vault_model,
	review_queue_model,
	settings_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-dlp", {"channels": {"bulk_export_threshold_records": 5000}})

	assert contract["capability"] == "dlpd"
	assert contract["configuration"]["tenant_id"] == "tenant-dlp"
	assert contract["configuration"]["channels"]["bulk_export_threshold_records"] == 5000
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "data_patterns", "policies", "channels", "classification", "response", "quarantine", "incidents", "reviews", "security", "governance", "observability", "adapters", "ui", "theme"}
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "policies", "classifiers", "channels", "inspections", "incidents", "quarantine", "reviews", "legal_hold", "analytics", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/dlpd/api/v1"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "incident_queue" in contract["theme"]["components"]
	assert "review_queue" in contract["theme"]["components"]


def test_rule_engine_enforces_dlp_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "inspect_egress",
		"egress_policy_attached": False,
		"policy_active": False,
		"channel_covered": False,
		"destination_present": False,
		"sensitive_content_detected": True,
		"classification_label_present": False,
		"severity": "high",
		"blocked_or_quarantined": False,
		"secret_detected": True,
		"quarantine_requested": True,
		"quarantine_encrypted": False,
		"export_record_count": 20000,
		"review_recorded": False,
	})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_dlp_mutation", "event_stream": "kafka"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"inspection_source_requires_policy",
		"inspection_requires_active_policy",
		"inspection_requires_covered_channel",
		"inspection_requires_destination",
		"sensitive_content_requires_classification",
		"secret_exfiltration_requires_block",
		"high_severity_exfiltration_requires_block",
		"quarantine_requires_encryption",
		"large_export_requires_review",
	}
	assert batch_result["matched_rules"] == ["batch_dlp_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "dlpd"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "dlpd_data_protection_ops"
	assert registration["ui_components"]["incidents"] == "/dlpd/incidents"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["endpoints"]["audit"] == "/dlpd/api/v1/audit"
	assert "nlpc" in registration["dependencies"]
	assert "dlpd:respond" in registration["permissions"]
	assert "dlpd:audit" in registration["permissions"]


def test_service_runs_sensitive_egress_quarantine_and_incident_lifecycle():
	service = DlpdService()
	tenant_id = "tenant-dlp"

	classifier = service.register_classifier("cls-secrets", tenant_id, "Secrets and PCI", "built_in", "restricted", ["secrets", "pci"])
	policy = service.register_policy("pol-email", tenant_id, "Email exfiltration prevention", "security-ops", ["email", "api_export"], [classifier["id"]], default_action="quarantine")
	inspection = service.inspect_egress(
		inspection_id="insp-1",
		tenant_id=tenant_id,
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
	assert service.list_quarantine(tenant_id)[0]["encrypted"] is True
	assert service.list_incidents(tenant_id)[0]["owner"] == "security-ops"

	resolved = service.resolve_incident("inc-insp-1", tenant_id, "analyst-1", "false positive removed from export")
	assert resolved["status"] == "resolved"
	assert service.dashboard_summary(tenant_id)["quarantine_count"] == 1
	assert dashboard_model(service, tenant_id)["summary"]["inspection_count"] == 1
	assert policy_console_model(service, tenant_id)["policies"][0]["id"] == "pol-email"
	assert classifier_workbench_model(service, tenant_id)["classifiers"][0]["id"] == "cls-secrets"
	assert channel_monitor_model(service, tenant_id)["inspections"][0]["id"] == "insp-1"
	assert inspection_workbench_model(service, tenant_id)["inspections"][0]["decision"] == "quarantine"
	assert incident_queue_model(service, tenant_id)["resolved"][0]["id"] == "inc-insp-1"
	assert quarantine_vault_model(service, tenant_id)["quarantine"][0]["id"] == "qrn-insp-1"
	assert review_queue_model(service, tenant_id)["review_rules"]
	assert legal_hold_model(service, tenant_id)["legal_hold_items"][0]["id"] == "qrn-insp-1"
	assert analytics_model(service, tenant_id)["summary"]["quarantine_count"] == 1
	assert audit_model(service, tenant_id)["audit_events"]
	assert settings_model(service, tenant_id)["configuration"]["adapters"]["event_stream"] == "bytewax"


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
		service.inspect_egress("insp-wrong-channel", "tenant-dlp", "pol-chat", "email", "user-1", "external@example.com", "alice@example.com")
	with pytest.raises(PermissionError, match="classification_label_required"):
		service.inspect_egress("insp-unlabeled", "tenant-dlp", "pol-chat", "chat", "user-1", "external-room", "alice@example.com", auto_classify=False)


def test_large_export_review_and_high_severity_block_rules_are_executable():
	service = DlpdService()
	service.register_classifier("cls-secrets", "tenant-dlp", "Secrets", "built_in", "restricted", ["secrets"])
	service.register_policy("pol-alert", "tenant-dlp", "Alert only", "owner", ["api_export"], ["cls-secrets"], default_action="alert")

	with pytest.raises(PermissionError, match="secret_exfiltration_block_required"):
		service.inspect_egress("insp-alert", "tenant-dlp", "pol-alert", "api_export", "user-1", "external-api", "secret='SECRET123456789'")

	service.register_policy("pol-review", "tenant-dlp", "Review large exports", "owner", ["api_export"], ["cls-secrets"], default_action="allow")
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


def test_dlpd_runtime_isolates_same_record_ids_by_tenant():
	service = DlpdService()

	alpha = service.register_classifier("shared-classifier", "tenant-alpha", "Alpha PII", "built_in", "confidential", ["pii"])
	beta = service.register_classifier("shared-classifier", "tenant-beta", "Beta PII", "built_in", "confidential", ["pii"])

	assert alpha["tenant_id"] == "tenant-alpha"
	assert beta["tenant_id"] == "tenant-beta"
	assert service.list_classifiers("tenant-alpha") == [alpha]
	assert service.list_classifiers("tenant-beta") == [beta]

	with pytest.raises(PermissionError, match="cross_tenant_dlp_access_denied"):
		service.register_policy("pol-cross", "tenant-gamma", "Cross", "owner", ["email"], ["shared-classifier"])
		service.inspect_egress("insp-cross", "tenant-gamma", "pol-cross", "email", "user", "external", "alice@example.com")


def test_api_helpers_wrap_runtime_operations():
	tenant_id = "tenant-api-dlpd"
	classifier = api.register_classifier({"id": "api-classifier", "tenant_id": tenant_id, "name": "API Secrets", "classifier_type": "built_in", "sensitivity_label": "restricted", "pattern_keys": ["secrets"]})
	policy = api.register_policy({"id": "api-policy", "tenant_id": tenant_id, "name": "API policy", "owner": "owner", "channels": ["email"], "classifiers": [classifier["id"]], "default_action": "quarantine"})
	inspection = api.inspect_egress({"id": "api-inspection", "tenant_id": tenant_id, "policy_id": policy["id"], "channel": "email", "subject_id": "user-1", "destination": "external@example.com", "content": "api_key='SECRET123456789'"})

	assert classifier["id"] == "api-classifier"
	assert policy["id"] == "api-policy"
	assert inspection["quarantined"] is True
	assert api.capability_status(tenant_id)["inspection_count"] == 1
