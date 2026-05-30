"""Regression coverage for the MFAU executable capability contract."""

import pytest

from capabilities.common.mfau import register_capability
from capabilities.common.mfau.api import (
	bind_device_endpoint,
	create_challenge_endpoint,
	create_service,
	enroll_method_endpoint,
	health,
	register_profile_endpoint,
)
from capabilities.common.mfau.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.mfau.mfa_runtime import MfauGuardrailError, MfauService
from capabilities.common.mfau.views import (
	audit_timeline_model,
	backup_code_model,
	biometric_consent_model,
	challenge_console_model,
	dashboard_model,
	device_trust_model,
	enrollment_wizard_model,
	governance_model,
	method_registry_model,
	policy_studio_model,
	profile_registry_model,
	recovery_center_model,
	risk_console_model,
	route_manifest,
	settings_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-auth", {"risk": {"high_risk_threshold": 0.8}})

	assert contract["capability"] == "mfau"
	assert contract["configuration"]["tenant_id"] == "tenant-auth"
	assert contract["configuration"]["risk"]["high_risk_threshold"] == 0.8
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "profiles", "methods", "enrollment", "challenge", "risk", "devices", "recovery", "backup_codes", "policies", "biometrics", "security", "governance", "observability", "adapters", "ui", "theme"}
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "profiles", "methods", "enrollment", "challenges", "risk", "devices", "recovery", "backup_codes", "policies", "biometrics", "governance", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/mfau/api/v1"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "risk_meter" in contract["theme"]["components"]


def test_rule_engine_enforces_adaptive_mfa_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "recover_account",
		"verified_recovery_channel": False,
	})
	enrollment_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "enroll_method",
		"profile_present": True,
		"method_type_present": True,
		"method_type_allowed": True,
		"method_type": "biometric",
		"biometric_consent_recorded": False,
		"template_encrypted": True,
		"device_bound_method": False,
		"active_method_count": 1,
		"review_recorded": True,
		"secret_encrypted": True,
	})
	challenge_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_challenge",
		"profile_present": True,
		"active_method_present": True,
		"risk_score": 0.9,
		"risk_override_approved": False,
		"step_up_completed": False,
		"action_risk": "admin",
		"phishing_resistant_factor_present": False,
		"device_trust_score": 0.2,
		"device_review_recorded": False,
		"profile_locked": False,
	})
	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "batch_mfa_mutation",
		"event_stream": "kafka",
	})
	policy_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "change_policy",
		"audit_event_recorded": False,
		"state_change_requested": True,
	})
	device_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_challenge",
		"device_trust_score": 0.2,
		"device_review_recorded": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"recovery_requires_verified_channel",
	}
	assert enrollment_result["decision"] == "deny"
	assert enrollment_result["matched_rules"] == ["biometric_method_requires_consent"]
	assert challenge_result["decision"] == "deny"
	assert set(challenge_result["matched_rules"]) >= {
		"high_risk_requires_step_up",
		"critical_risk_requires_block",
		"admin_action_requires_phishing_resistant_factor",
		"low_trust_device_requires_review",
	}
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_mfa_mutation_requires_bytewax"]
	assert policy_result["decision"] == "deny"
	assert policy_result["matched_rules"] == ["policy_change_requires_audit", "mfa_state_change_requires_audit"]
	assert device_result["decision"] == "require_review"
	assert device_result["matched_rules"] == ["low_trust_device_requires_review"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "mfau"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mfau_adaptive_auth_console"
	assert registration["ui_components"]["enrollment"] == "/mfau/enrollment"
	assert "auth" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["endpoints"]["audit"] == "/mfau/api/v1/audit"
	assert "mfau:challenge" in registration["permissions"]
	assert "mfau:audit" in registration["permissions"]


def test_mfau_lifecycle_is_executable():
	service = MfauService()
	tenant_id = "tenant-auth"

	profile = service.create_user_profile(
		profile_id="profile-alice",
		tenant_id=tenant_id,
		user_id="alice",
		policy_id="standard-mfa",
		primary_channel="alice@example.com",
	)
	device = service.bind_device(
		device_id="device-alice",
		tenant_id=tenant_id,
		user_id="alice",
		trust_score=0.86,
	)
	method = service.enroll_method(
		method_id="method-alice-webauthn",
		tenant_id=tenant_id,
		user_id="alice",
		method_type="webauthn",
		device_id=device["id"],
		phishing_resistant=True,
	)
	risk = service.assess_risk(
		assessment_id="risk-alice-login",
		tenant_id=tenant_id,
		user_id="alice",
		risk_score=0.42,
		device_trust_score=device["trust_score"],
	)
	challenge = service.create_challenge(
		challenge_id="challenge-alice-login",
		tenant_id=tenant_id,
		user_id="alice",
		method_id=method["id"],
		assessment_id=risk["id"],
	)
	completed = service.complete_challenge(challenge["id"], tenant_id)
	recovery = service.recover_account(
		recovery_id="recovery-alice",
		tenant_id=tenant_id,
		user_id="alice",
		admin_recovery=True,
		admin_approval_recorded=True,
	)
	codes = service.generate_backup_codes(
		code_set_id="codes-alice",
		tenant_id=tenant_id,
		user_id="alice",
		code_count=3,
	)
	service.use_backup_code(
		code_set_id=codes["id"],
		tenant_id=tenant_id,
		user_id="alice",
		code_value=codes["metadata"]["codes"][0],
	)
	policy = service.create_policy(
		policy_id="policy-strong",
		tenant_id=tenant_id,
		name="Strong MFA",
	)

	assert profile["metadata"]["policy_id"] == "standard-mfa"
	assert method["metadata"]["phishing_resistant"] is True
	assert risk["status"] == "normal"
	assert completed["status"] == "completed"
	assert recovery["status"] == "approved"
	assert service.list_backup_code_sets(tenant_id)[0]["metadata"]["remaining"] == 2
	assert policy["metadata"]["name"] == "Strong MFA"

	summary = service.dashboard_summary(tenant_id)
	assert summary["profile_count"] == 1
	assert summary["active_method_count"] == 1
	assert summary["completed_challenge_count"] == 1
	assert summary["recovery_count"] == 1
	assert summary["policy_count"] == 1
	assert summary["audit_event_count"] >= 8

	assert dashboard_model(service, tenant_id)["summary"]["profile_count"] == 1
	assert profile_registry_model(service, tenant_id)["profiles"][0]["id"] == "profile-alice"
	assert method_registry_model(service, tenant_id)["methods"][0]["id"] == "method-alice-webauthn"
	assert enrollment_wizard_model(service, tenant_id)["steps"][0] == "select_user"
	assert challenge_console_model(service, tenant_id)["challenges"][0]["id"] == "challenge-alice-login"
	assert risk_console_model(service, tenant_id)["assessments"][0]["id"] == "risk-alice-login"
	assert device_trust_model(service, tenant_id)["devices"][0]["id"] == "device-alice"
	assert recovery_center_model(service, tenant_id)["recoveries"][0]["id"] == "recovery-alice"
	assert backup_code_model(service, tenant_id)["code_sets"][0]["metadata"]["remaining"] == 2
	assert policy_studio_model(service, tenant_id)["policies"][0]["id"] == "policy-strong"
	assert biometric_consent_model(service, tenant_id)["requirements"]["consent_required"] is True
	assert governance_model(service, tenant_id)["adapters"]["event_stream"] == "bytewax"
	assert audit_timeline_model(service, tenant_id)["events"]
	assert settings_model(service, tenant_id)["route_manifest"]["capability"] == "mfau"
	assert route_manifest(tenant_id)["api_prefix"] == "/mfau/api/v1"


def test_runtime_rejects_guardrail_violations():
	service = MfauService()
	tenant_id = "tenant-auth"
	service.create_user_profile("profile-bob", tenant_id, "bob", "standard-mfa", "bob@example.com")

	with pytest.raises(MfauGuardrailError) as biometric_error:
		service.enroll_method(
			method_id="method-bob-bio",
			tenant_id=tenant_id,
			user_id="bob",
			method_type="biometric",
			device_id="device-bob",
			biometric_consent_recorded=False,
		)
	assert "biometric_method_requires_consent" in biometric_error.value.result["matched_rules"]

	with pytest.raises(MfauGuardrailError) as recovery_error:
		service.recover_account(
			recovery_id="recovery-bob",
			tenant_id=tenant_id,
			user_id="bob",
			verified_recovery_channel=False,
		)
	assert recovery_error.value.result["decision"] == "deny"

	with pytest.raises(MfauGuardrailError) as policy_error:
		service.create_policy(
			policy_id="policy-no-audit",
			tenant_id=tenant_id,
			name="No Audit",
			audit_event_recorded=False,
		)
	assert "policy_change_requires_audit" in policy_error.value.result["matched_rules"]


def test_api_helpers_wrap_runtime_operations():
	service = create_service()
	tenant_id = "tenant-api"

	profile_response = register_profile_endpoint(service, {
		"profile_id": "profile-api",
		"tenant_id": tenant_id,
		"user_id": "api-user",
		"policy_id": "standard-mfa",
		"primary_channel": "api@example.com",
	})
	device_response = bind_device_endpoint(service, {
		"device_id": "device-api",
		"tenant_id": tenant_id,
		"user_id": "api-user",
		"trust_score": 0.7,
	})
	method_response = enroll_method_endpoint(service, {
		"method_id": "method-api",
		"tenant_id": tenant_id,
		"user_id": "api-user",
		"method_type": "webauthn",
		"device_id": "device-api",
	})
	risk = service.assess_risk("risk-api", tenant_id, "api-user", 0.2, 0.7)
	challenge_response = create_challenge_endpoint(service, {
		"challenge_id": "challenge-api",
		"tenant_id": tenant_id,
		"user_id": "api-user",
		"method_id": "method-api",
		"assessment_id": risk["id"],
	})

	assert profile_response["ok"] is True
	assert device_response["ok"] is True
	assert method_response["ok"] is True
	assert challenge_response["ok"] is True
	assert health(service, tenant_id)["summary"]["challenge_count"] == 1
