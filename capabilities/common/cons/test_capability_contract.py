"""Regression coverage for the CONS executable capability contract."""

from datetime import datetime, timedelta, timezone

import pytest

from capabilities.common.cons import register_capability
from capabilities.common.cons.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.cons.service import ConsService
from capabilities.common.cons.views import analytics_model, audit_trail_model, dashboard_model, privacy_agents_model, settings_model


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-cons", {"consents": {"stale_review_days": 180}})

	assert contract["capability"] == "cons"
	assert contract["configuration"]["tenant_id"] == "tenant-cons"
	assert contract["configuration"]["consents"]["stale_review_days"] == 180
	assert contract["configuration_schema"]["required"] == ["tenant_id", "purposes", "consents", "privacy_requests", "privacy_agents", "governance", "observability", "adapters", "ui", "theme"]
	assert contract["configuration"]["privacy_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "purposes", "notices", "consents", "requests", "preferences", "agents", "analytics", "audit", "settings"}
	assert contract["theme"]["name"] == "cons_privacy_center"
	assert contract["streaming"]["processor"] == "bytewax"


def test_rule_engine_enforces_cons_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_purpose", "legal_basis_present": False, "consent_age_days": 400, "stale_consent_reviewed": False})
	privacy_request_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "process_privacy_request", "identity_verified": False})
	processing_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "process_consent_gated_data", "active_consent_present": False})
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "privacy_agent_present": True, "agent_registered": True, "agent_runtime_supported": False, "agent_role_supported": True, "agent_scope_present": True, "agent_contribution_disclosed": True})
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_privacy_mutation", "event_stream": "custom"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "purpose_requires_legal_basis", "stale_consent_requires_review"}
	assert privacy_request_result["matched_rules"] == ["privacy_request_requires_identity_verification"]
	assert processing_result["matched_rules"] == ["processing_requires_active_consent"]
	assert agent_result["matched_rules"] == ["privacy_agent_runtime_supported"]
	assert stream_result["matched_rules"] == ["batch_privacy_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "cons"
	assert "dlpd" in registration["dependencies"]
	assert "bytewax" in registration["optional_dependencies"]
	assert registration["ui_components"]["requests"] == "/cons/requests"
	assert registration["ui_components"]["agents"] == "/cons/agents"
	assert "cons:process_requests" in registration["permissions"]
	assert "cons:audit" in registration["permissions"]
	assert registration["streaming"]["processor"] == "bytewax"


def test_service_runs_privacy_lifecycle_with_consent_processing_and_request_completion():
	service = ConsService()

	notice = service.publish_notice(
		"notice-v1",
		"tenant-cons",
		"2026.1",
		"https://privacy.example/notice",
		"en",
		["marketing"],
		"privacy-owner",
	)
	purpose = service.create_purpose(
		"purpose-marketing",
		"tenant-cons",
		"Product marketing",
		"privacy-owner",
		"consent",
		"retain-24-months",
		"notice-v1",
		["email", "profile"],
	)
	agent = service.register_privacy_agent(
		"tenant-cons",
		"codex-notice",
		"Codex Notice Reviewer",
		"codex",
		"notice_reviewer",
		"Review notices, purpose links, and request evidence.",
		True,
		"policy:cons:agents",
	)
	inactive = service.change_purpose_state("tenant-cons", "purpose-marketing", False, "Pause marketing purpose for notice review.")
	active = service.change_purpose_state("tenant-cons", "purpose-marketing", True, "Resume after notice review.")
	consent = service.capture_consent(
		"consent-001",
		"tenant-cons",
		"subject-001",
		"purpose-marketing",
		"notice-v1",
		"web-form",
		"preference-center",
	)
	preferences = service.update_preferences(
		"pref-001",
		"tenant-cons",
		"subject-001",
		{"email": True, "sms": False},
		{"purpose-marketing": True},
		"subject-001",
	)
	decision = service.process_consent_gated_data(
		"decision-001",
		"tenant-cons",
		"subject-001",
		"purpose-marketing",
	)
	request = service.submit_privacy_request(
		"request-access-001",
		"tenant-cons",
		"subject-001",
		"access",
		"subject-001",
		identity_verified=True,
		evidence_reference="identity-proof:001",
	)
	completed = service.complete_privacy_request(
		"request-access-001",
		"tenant-cons",
		"privacy-ops",
		"Export delivered through secure portal.",
	)
	summary = service.dashboard_summary("tenant-cons")

	assert notice["version"] == "2026.1"
	assert purpose["legal_basis"] == "consent"
	assert agent["runtime"] == "codex"
	assert inactive["active"] is False
	assert active["active"] is True
	assert consent["status"] == "active"
	assert consent["provenance_hash"]
	assert preferences["channels"]["email"] is True
	assert decision["decision"] == "allow"
	assert decision["consent_id"] == "consent-001"
	assert request["status"] == "open"
	assert completed["status"] == "completed"
	assert summary["purpose_count"] == 1
	assert summary["active_consent_count"] == 1
	assert summary["open_request_count"] == 0
	assert summary["privacy_agent_count"] == 1
	assert summary["audit_event_count"] >= 10
	assert summary["coverage"]["posture"] == "fully_consented"
	assert len(service.list_audit_events("tenant-cons")) >= 7
	assert privacy_agents_model(service, "tenant-cons")["agents"][0]["id"] == "codex-notice"
	assert audit_trail_model(service, "tenant-cons")["events"]
	assert analytics_model(service, "tenant-cons")["streaming"]["processor"] == "bytewax"
	assert settings_model(service, "tenant-cons")["streaming"]["processor"] == "bytewax"
	assert dashboard_model(service, "tenant-cons")["streaming"]["processor"] == "bytewax"


def test_service_enforces_privacy_guardrails():
	service = ConsService()
	service.publish_notice("notice-v1", "tenant-cons", "2026.1", "https://privacy.example/notice", "en", [], "privacy-owner")

	with pytest.raises(PermissionError, match="legal_basis_required"):
		service.create_purpose(
			"purpose-no-basis",
			"tenant-cons",
			"Analytics",
			"privacy-owner",
			"",
			"retain-12-months",
			"notice-v1",
			["usage"],
		)

	with pytest.raises(PermissionError, match="purpose_owner_required"):
		service.create_purpose(
			"purpose-no-owner",
			"tenant-cons",
			"Analytics",
			"",
			"legitimate_interest",
			"retain-12-months",
			"notice-v1",
			["usage"],
		)

	service.create_purpose(
		"purpose-analytics",
		"tenant-cons",
		"Analytics",
		"privacy-owner",
		"consent",
		"retain-12-months",
		"notice-v1",
		["usage"],
	)
	with pytest.raises(KeyError, match="notice_not_found"):
		service.capture_consent(
			"consent-missing-notice",
			"tenant-cons",
			"subject-001",
			"purpose-analytics",
			"missing-notice",
			"web",
			"privacy-owner",
		)

	with pytest.raises(PermissionError, match="active_consent_required"):
		service.process_consent_gated_data("decision-denied", "tenant-cons", "subject-001", "purpose-analytics")

	with pytest.raises(PermissionError, match="identity_verification_required"):
		service.submit_privacy_request(
			"request-unverified",
			"tenant-cons",
			"subject-001",
			"delete",
			"subject-001",
			identity_verified=False,
			evidence_reference="identity-proof:001",
		)

	with pytest.raises(PermissionError, match="request_evidence_required"):
		service.submit_privacy_request(
			"request-no-evidence",
			"tenant-cons",
			"subject-001",
			"delete",
			"subject-001",
			identity_verified=True,
			evidence_reference="",
		)

	with pytest.raises(PermissionError, match="privacy_agent_runtime_not_supported"):
		service.register_privacy_agent("tenant-cons", "bad-runtime", "Bad Runtime", "custom", "notice_reviewer", "Review notice.", True)

	with pytest.raises(PermissionError, match="privacy_agent_role_not_supported"):
		service.register_privacy_agent("tenant-cons", "bad-role", "Bad Role", "codex", "owner", "Review notice.", True)

	with pytest.raises(PermissionError, match="privacy_agent_disclosure_required"):
		service.register_privacy_agent("tenant-cons", "undisclosed", "Undisclosed", "codex", "notice_reviewer", "Review notice.", False)

	with pytest.raises(PermissionError, match="cons_state_change_reason_required"):
		service.change_purpose_state("tenant-cons", "purpose-analytics", False, "")

	with pytest.raises(PermissionError, match="cons_audit_event_required"):
		service.change_purpose_state("tenant-cons", "purpose-analytics", False, "Pause without audit.", audit_recorded=False)


def test_service_withdraws_consent_and_flags_stale_consent_for_review():
	service = ConsService()
	service.publish_notice("notice-v1", "tenant-cons", "2026.1", "https://privacy.example/notice", "en", [], "privacy-owner")
	service.create_purpose(
		"purpose-marketing",
		"tenant-cons",
		"Marketing",
		"privacy-owner",
		"consent",
		"retain-24-months",
		"notice-v1",
		["email"],
	)
	stale_time = datetime.now(timezone.utc) - timedelta(days=400)
	service.capture_consent(
		"consent-stale",
		"tenant-cons",
		"subject-001",
		"purpose-marketing",
		"notice-v1",
		"web-form",
		"preference-center",
		captured_at=stale_time,
	)

	review_required = service.review_stale_consents("tenant-cons")
	withdrawn = service.withdraw_consent("consent-stale", "tenant-cons", "subject-001")

	assert [item["id"] for item in review_required] == ["consent-stale"]
	assert withdrawn["status"] == "withdrawn"
	assert service.dashboard_summary("tenant-cons")["withdrawn_consent_count"] == 1


def test_service_allows_duplicate_privacy_ids_across_tenants():
	service = ConsService()

	for tenant_id in ("tenant-a", "tenant-b"):
		service.publish_notice("notice-v1", tenant_id, "2026.1", "https://privacy.example/notice", "en", [], "privacy-owner")
		service.create_purpose("purpose-shared", tenant_id, "Shared", "privacy-owner", "consent", "retain-12-months", "notice-v1", ["email"])
		service.register_privacy_agent(tenant_id, "shared-agent", "Shared Agent", "codex", "notice_reviewer", f"Review {tenant_id}.", True)

	assert service.list_purposes("tenant-a")[0]["id"] == "purpose-shared"
	assert service.list_purposes("tenant-b")[0]["id"] == "purpose-shared"
	assert service.list_privacy_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_privacy_agents("tenant-b")[0]["id"] == "shared-agent"
