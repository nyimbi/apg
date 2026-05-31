"""Regression coverage for the IDFD executable capability contract."""

import pytest

from capabilities.common.idfd import api, register_capability
from capabilities.common.idfd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.idfd.service import IdfdService, expires_in_days
from capabilities.common.idfd.views import (
	audit_model,
	certificate_center_model,
	dashboard_model,
	federation_agent_roster_model,
	lifecycle_batch_model,
	mapping_table_model,
	protocol_workbench_model,
	provider_console_model,
	review_queue_model,
	risk_console_model,
	scim_directory_model,
	session_monitor_model,
	settings_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-sso", {"sessions": {"max_session_hours": 8}})

	assert contract["capability"] == "idfd"
	assert contract["configuration"]["tenant_id"] == "tenant-sso"
	assert contract["configuration"]["sessions"]["max_session_hours"] == 8
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "providers", "protocols", "claims", "sessions", "scim", "certificates", "reviews", "agents", "streaming", "security", "governance", "observability", "adapters", "ui", "theme"}
	assert len(contract["rule_engine"]["rules"]) >= 44
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "providers", "protocols", "mappings", "sessions", "certificates", "scim", "risk", "reviews", "agents", "lifecycle", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/idfd/api/v1"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "federation_agent_composition" in contract["provides"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "federation_agent_batch" in contract["streaming"]["required_operations"]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "certificate_timeline" in contract["theme"]["components"]
	assert "review_queue" in contract["theme"]["components"]
	assert "federation_agent_roster" in contract["theme"]["components"]
	assert "bytewax_lifecycle_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_federation_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_provider",
		"owner_present": False,
		"signing_key_present": False,
		"metadata_present": False,
		"metadata_signed": False,
		"protocol": "saml",
		"assertion_encrypted": False,
		"response_signature_validated": False,
		"session_privilege": "privileged",
		"mfa_completed": False,
		"metadata_age_hours": 72,
		"metadata_refresh_completed": False,
	})
	oidc_result = evaluate_capability_rules({"tenant_context_present": True, "protocol": "oidc", "redirect_allowlist_configured": False, "pkce_required": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_federation_mutation", "event_stream": "kafka"})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_federation_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "validate_idfd_lifecycle_batch", "event_stream": "kafka", "mutation_count": 1})
	empty_lifecycle_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "validate_idfd_lifecycle_batch", "event_stream": "bytewax", "mutation_count": 0})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {"tenant_context_required", "provider_requires_owner", "provider_requires_signing_key", "provider_metadata_url_required", "provider_metadata_signature_required", "saml_assertion_requires_encryption", "saml_requires_signed_response", "privileged_federation_requires_mfa", "stale_metadata_requires_refresh"}
	assert set(oidc_result["matched_rules"]) == {"oidc_client_requires_redirect_allowlist", "oidc_requires_pkce"}
	assert batch_result["matched_rules"] == ["batch_federation_mutation_requires_bytewax"]
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {"federation_agent_runtime_supported", "federation_agent_role_supported", "federation_agent_requires_scope", "federation_agent_requires_owner", "federation_agent_requires_purpose", "federation_agent_requires_contribution_disclosure", "federation_agent_privileged_role_requires_human_approval"}
	assert lifecycle_result["matched_rules"] == ["bytewax_idfd_stream_required"]
	assert empty_lifecycle_result["matched_rules"] == ["idfd_lifecycle_batch_requires_mutations"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "idfd"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "idfd_federation_console"
	assert registration["ui_components"]["providers"] == "/idfd/providers"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert registration["endpoints"]["agents"] == "/idfd/api/v1/agents"
	assert registration["endpoints"]["lifecycle"] == "/idfd/api/v1/lifecycle"
	assert registration["endpoints"]["audit"] == "/idfd/api/v1/audit"
	assert "mfau" in registration["dependencies"]
	assert "idfd:rotate_keys" in registration["permissions"]
	assert "idfd:review" in registration["permissions"]


def test_idfd_lifecycle_is_executable():
	service = IdfdService()
	tenant_id = "tenant-sso"

	provider = service.register_provider(
		provider_id="saml-main",
		tenant_id=tenant_id,
		name="Corporate SAML",
		protocol="saml",
		owner_id="identity-owner",
		signing_key_id="signing-key-1",
		metadata_url="https://idp.example.test/metadata",
		assertion_encrypted=True,
		metadata_age_hours=2,
	)
	mapping = service.add_claim_mapping("map-email", tenant_id, "saml-main", "mail", "email", "lowercase", reviewed=True)
	session = service.issue_session("session-1", tenant_id, "saml-main", "user-1", session_privilege="privileged", mfa_completed=True, risk_score=0.2)
	certificate = service.register_certificate("cert-1", tenant_id, "saml-main", "signing-key-1", expires_in_days(10))
	report = service.health_report("health-1", tenant_id)
	revoked = service.revoke_session("session-1", tenant_id, "admin_request")
	agent = service.register_federation_agent("agent-idfd", tenant_id, "Federation Steward", "codex", "provider_reviewer", "provider metadata", "identity", "review federation provider evidence")
	batch = service.validate_idfd_lifecycle_batch(tenant_id, "bytewax", 2, "federation_agent_batch", "idfd-batch-1")

	assert provider["protocol"] == "saml"
	assert mapping["target_claim"] == "email"
	assert session["status"] == "active"
	assert certificate["active"] is True
	assert report["expiring_certificate_count"] == 1
	assert revoked["status"] == "revoked"
	assert agent["status"] == "active"
	assert batch["status"] == "accepted"
	assert service.dashboard_summary(tenant_id)["provider_count"] == 1
	assert service.dashboard_summary(tenant_id)["federation_agent_count"] == 1
	assert service.dashboard_summary(tenant_id)["lifecycle_batch_count"] == 1
	assert provider_console_model(service, tenant_id)["providers"][0]["id"] == "saml-main"
	assert protocol_workbench_model(service, tenant_id)["protocols"]["saml"][0]["id"] == "saml-main"
	assert mapping_table_model(service, tenant_id)["mappings"][0]["id"] == "map-email"
	assert session_monitor_model(service, tenant_id)["sessions"][0]["status"] == "revoked"
	assert certificate_center_model(service, tenant_id)["certificates"][0]["id"] == "cert-1"
	assert scim_directory_model(service, tenant_id)["route"] == "/idfd/scim"
	assert risk_console_model(service, tenant_id)["session_count"] == 1
	assert review_queue_model(service, tenant_id)["review_rules"]
	assert federation_agent_roster_model(service, tenant_id)["agents"][0]["id"] == "agent-idfd"
	assert lifecycle_batch_model(service, tenant_id)["required_processor"] == "bytewax"
	assert settings_model(service, tenant_id)["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert audit_model(service, tenant_id)["events"]
	assert dashboard_model(service, tenant_id)["summary"]["audit_event_count"] >= 6


def test_idfd_service_enforces_policy_guardrails():
	service = IdfdService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_provider("missing-tenant", "", "Missing Tenant", "saml", "owner", "key", metadata_url="https://idp.example.test/metadata")

	with pytest.raises(PermissionError, match="signing_key_required"):
		service.register_provider("missing-key", "tenant-sso", "Missing Key", "saml", "owner", "", metadata_url="https://idp.example.test/metadata")

	with pytest.raises(PermissionError, match="saml_assertion_encryption_required"):
		service.register_provider("plain-saml", "tenant-sso", "Plain SAML", "saml", "owner", "key", metadata_url="https://idp.example.test/metadata", assertion_encrypted=False)

	with pytest.raises(PermissionError, match="redirect_allowlist_required"):
		service.register_provider("bad-oidc", "tenant-sso", "Bad OIDC", "oidc", "owner", "key", metadata_url="https://idp.example.test/metadata", redirect_allowlist=[])

	with pytest.raises(PermissionError, match="metadata_refresh_required"):
		service.register_provider("stale", "tenant-sso", "Stale Provider", "saml", "owner", "key", metadata_url="https://idp.example.test/metadata", metadata_age_hours=72, metadata_refresh_completed=False)

	service.register_provider("good-oidc", "tenant-sso", "Good OIDC", "oidc", "owner", "key", metadata_url="https://idp.example.test/metadata", redirect_allowlist=["https://app.example/callback"])

	with pytest.raises(PermissionError, match="privileged_mfa_required"):
		service.issue_session("session-no-mfa", "tenant-sso", "good-oidc", "user-1", session_privilege="privileged", mfa_completed=False)

	with pytest.raises(PermissionError, match="high_risk_reauth_required"):
		service.issue_session("session-risk", "tenant-sso", "good-oidc", "user-1", risk_score=0.91, reauth_completed=False)

	with pytest.raises(PermissionError, match="claim_mapping_review_required"):
		service.add_claim_mapping("unreviewed", "tenant-sso", "good-oidc", "groups", "roles", reviewed=False)

	with pytest.raises(PermissionError, match="cross_tenant_federation_access_denied"):
		service.add_claim_mapping("wrong-tenant", "other-tenant", "good-oidc", "mail", "email")

	with pytest.raises(PermissionError, match="unsupported_federation_agent_runtime"):
		service.register_federation_agent("agent-bad", "tenant-sso", "Bad Agent", "unknown", "provider_reviewer", "providers", "owner", "purpose")

	pending_agent = service.register_federation_agent("agent-pending", "tenant-sso", "Pending Agent", "claude_code", "session_risk_reviewer", "sessions", "owner", "purpose")
	assert pending_agent["status"] == "pending_review"

	with pytest.raises(ValueError, match="idfd_lifecycle_batch_empty"):
		service.validate_idfd_lifecycle_batch("tenant-sso", "bytewax", 0)
	with pytest.raises(ValueError, match="unsupported_idfd_lifecycle_operation"):
		service.validate_idfd_lifecycle_batch("tenant-sso", "bytewax", 1, "unknown_batch")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_idfd_lifecycle_batch("tenant-sso", "kafka", 1, "federation_agent_batch")


def test_idfd_runtime_isolates_same_record_ids_by_tenant():
	service = IdfdService()

	alpha = service.register_provider("shared-provider", "tenant-alpha", "Alpha SAML", "saml", "owner", "key-a", metadata_url="https://alpha.example/metadata")
	beta = service.register_provider("shared-provider", "tenant-beta", "Beta SAML", "saml", "owner", "key-b", metadata_url="https://beta.example/metadata")

	assert alpha["tenant_id"] == "tenant-alpha"
	assert beta["tenant_id"] == "tenant-beta"
	assert service.list_providers("tenant-alpha") == [alpha]
	assert service.list_providers("tenant-beta") == [beta]

	with pytest.raises(PermissionError, match="cross_tenant_federation_access_denied"):
		service.issue_session("session-cross", "tenant-gamma", "shared-provider", "user-1")


def test_api_helpers_wrap_runtime_operations():
	tenant_id = "tenant-api-idfd"
	provider = api.register_provider({
		"id": "api-provider",
		"tenant_id": tenant_id,
		"name": "API OIDC",
		"protocol": "oidc",
		"owner_id": "identity",
		"signing_key_id": "key-1",
		"metadata_url": "https://idp.example.test/metadata",
		"redirect_allowlist": ["https://app.example/callback"],
	})
	mapping = api.add_claim_mapping({"id": "api-map", "tenant_id": tenant_id, "provider_id": provider["id"], "source_claim": "mail", "target_claim": "email", "reviewed": True})
	session = api.issue_session({"id": "api-session", "tenant_id": tenant_id, "provider_id": provider["id"], "subject_id": "user-1"})
	agent = api.register_federation_agent({"id": "api-agent", "tenant_id": tenant_id, "name": "API Agent", "runtime": "opencode", "role": "provider_reviewer", "scope": "providers", "owner": "identity", "purpose": "inspect federation providers"})
	batch = api.validate_lifecycle_batch({"id": "api-batch", "tenant_id": tenant_id, "event_stream": "bytewax", "mutation_count": 1, "operation": "federation_agent_batch"})

	assert provider["protocol"] == "oidc"
	assert mapping["target_claim"] == "email"
	assert session["status"] == "active"
	assert agent["status"] == "active"
	assert batch["status"] == "accepted"
	assert api.capability_status(tenant_id)["provider_count"] == 1
	assert api.capability_status(tenant_id)["federation_agent_count"] == 1
	assert api.list_federation_agents(tenant_id)[0]["id"] == "api-agent"
	assert api.list_lifecycle_batches(tenant_id)[0]["id"] == "api-batch"
