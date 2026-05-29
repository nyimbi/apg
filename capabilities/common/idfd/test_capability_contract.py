"""Regression coverage for the IDFD executable capability contract."""

import pytest

from capabilities.common.idfd import register_capability
from capabilities.common.idfd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.idfd.service import IdfdService, expires_in_days
from capabilities.common.idfd.views import (
	audit_model,
	certificate_center_model,
	dashboard_model,
	mapping_table_model,
	protocol_workbench_model,
	provider_console_model,
	session_monitor_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-sso", {"sessions": {"max_session_hours": 8}})

	assert contract["capability"] == "idfd"
	assert contract["configuration"]["tenant_id"] == "tenant-sso"
	assert contract["configuration"]["sessions"]["max_session_hours"] == 8
	assert contract["configuration_schema"]["required"] == ["tenant_id", "providers", "protocols", "sessions", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "providers", "protocols", "mappings", "sessions", "certificates", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/idfd/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "certificate_timeline" in contract["theme"]["components"]


def test_rule_engine_enforces_federation_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_provider",
		"signing_key_present": False,
		"protocol": "saml",
		"assertion_encrypted": False,
		"redirect_allowlist_configured": False,
		"session_privilege": "privileged",
		"mfa_completed": False,
		"metadata_age_hours": 72,
		"metadata_refresh_completed": False
	})
	oidc_result = evaluate_capability_rules({"tenant_context_present": True, "protocol": "oidc", "redirect_allowlist_configured": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "provider_requires_signing_key", "saml_assertion_requires_encryption", "privileged_federation_requires_mfa", "stale_metadata_requires_refresh"}
	assert oidc_result["matched_rules"] == ["oidc_client_requires_redirect_allowlist"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "idfd"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "idfd_federation_console"
	assert registration["ui_components"]["providers"] == "/idfd/providers"
	assert "mfau" in registration["dependencies"]
	assert "idfd:rotate_keys" in registration["permissions"]


def test_idfd_lifecycle_is_executable():
	service = IdfdService()

	provider = service.register_provider(
		provider_id="saml-main",
		tenant_id="tenant-sso",
		name="Corporate SAML",
		protocol="saml",
		owner_id="identity-owner",
		signing_key_id="signing-key-1",
		metadata_url="https://idp.example.test/metadata",
		assertion_encrypted=True,
		metadata_age_hours=2,
	)
	mapping = service.add_claim_mapping(
		mapping_id="map-email",
		tenant_id="tenant-sso",
		provider_id="saml-main",
		source_claim="mail",
		target_claim="email",
		transform="lowercase",
		reviewed=True,
	)
	session = service.issue_session(
		session_id="session-1",
		tenant_id="tenant-sso",
		provider_id="saml-main",
		subject_id="user-1",
		session_privilege="privileged",
		mfa_completed=True,
		risk_score=0.2,
	)
	certificate = service.register_certificate(
		certificate_id="cert-1",
		tenant_id="tenant-sso",
		provider_id="saml-main",
		key_id="signing-key-1",
		expires_at=expires_in_days(10),
	)
	report = service.health_report("health-1", "tenant-sso")
	revoked = service.revoke_session("session-1", "tenant-sso", "admin_request")

	assert provider["protocol"] == "saml"
	assert mapping["target_claim"] == "email"
	assert session["status"] == "active"
	assert certificate["active"] is True
	assert report["expiring_certificate_count"] == 1
	assert revoked["status"] == "revoked"
	assert service.dashboard_summary("tenant-sso")["provider_count"] == 1
	assert provider_console_model(service, "tenant-sso")["providers"][0]["id"] == "saml-main"
	assert protocol_workbench_model(service, "tenant-sso")["protocols"]["saml"][0]["id"] == "saml-main"
	assert mapping_table_model(service, "tenant-sso")["mappings"][0]["id"] == "map-email"
	assert session_monitor_model(service, "tenant-sso")["sessions"][0]["status"] == "revoked"
	assert certificate_center_model(service, "tenant-sso")["certificates"][0]["id"] == "cert-1"
	assert audit_model(service, "tenant-sso")["events"]
	assert dashboard_model(service, "tenant-sso")["summary"]["audit_event_count"] >= 6


def test_idfd_service_enforces_policy_guardrails():
	service = IdfdService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_provider("missing-tenant", "", "Missing Tenant", "saml", "owner", "key")

	with pytest.raises(PermissionError, match="signing_key_required"):
		service.register_provider("missing-key", "tenant-sso", "Missing Key", "saml", "owner", "")

	with pytest.raises(PermissionError, match="saml_assertion_encryption_required"):
		service.register_provider("plain-saml", "tenant-sso", "Plain SAML", "saml", "owner", "key", assertion_encrypted=False)

	with pytest.raises(PermissionError, match="redirect_allowlist_required"):
		service.register_provider("bad-oidc", "tenant-sso", "Bad OIDC", "oidc", "owner", "key", redirect_allowlist=[])

	with pytest.raises(PermissionError, match="metadata_refresh_required"):
		service.register_provider(
			"stale",
			"tenant-sso",
			"Stale Provider",
			"saml",
			"owner",
			"key",
			metadata_age_hours=72,
			metadata_refresh_completed=False,
		)

	service.register_provider("good-oidc", "tenant-sso", "Good OIDC", "oidc", "owner", "key", redirect_allowlist=["https://app.example/callback"])

	with pytest.raises(PermissionError, match="privileged_mfa_required"):
		service.issue_session("session-no-mfa", "tenant-sso", "good-oidc", "user-1", session_privilege="privileged", mfa_completed=False)

	with pytest.raises(PermissionError, match="claim_mapping_review_required"):
		service.add_claim_mapping("unreviewed", "tenant-sso", "good-oidc", "groups", "roles", reviewed=False)

	with pytest.raises(KeyError, match="provider_missing"):
		service.add_claim_mapping("wrong-tenant", "other-tenant", "good-oidc", "mail", "email")
