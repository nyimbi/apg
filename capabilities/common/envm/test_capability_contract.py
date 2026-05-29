"""Regression coverage for the ENVM executable capability contract and service."""

from __future__ import annotations

import pytest

from capabilities.common.envm import register_capability
from capabilities.common.envm.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.envm.service import EnvmService
from capabilities.common.envm.views import dashboard_model


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-env", {"drift": {"drift_threshold_percent": 10}})

	assert contract["capability"] == "envm"
	assert contract["configuration"]["tenant_id"] == "tenant-env"
	assert contract["configuration"]["drift"]["drift_threshold_percent"] == 10
	assert contract["configuration_schema"]["required"] == ["tenant_id", "environments", "promotion", "drift", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "envm_environment_ops"


def test_rule_engine_enforces_environment_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_environment", "environment_owner_assigned": False, "environment": "production", "approval_recorded": False, "secret_scope_present": True, "secret_policy_attached": False, "drift_percent": 15, "drift_review_recorded": False})
	promote_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "promote", "promotion_path_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "environment_requires_owner", "production_change_requires_approval", "secret_scope_requires_policy", "high_drift_requires_review"}
	assert promote_result["matched_rules"] == ["promotion_requires_path"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "envm"
	assert "depl" in registration["dependencies"]
	assert registration["ui_components"]["promotion"] == "/envm/promotion"
	assert "envm:promote" in registration["permissions"]


def test_environment_promotion_drift_and_secret_scope_lifecycle():
	service = EnvmService()
	dev = service.register_environment(
		environment_id="env-dev",
		tenant_id="tenant-env",
		name="Development",
		stage="development",
		region="ke-nairobi",
		owner="platform",
		configuration_source="git://config/env-dev",
		rbac_policy="rbac-dev",
		secret_scope_policy="secret-dev",
	)
	prod = service.register_environment(
		environment_id="env-prod",
		tenant_id="tenant-env",
		name="Production",
		stage="production",
		region="ke-nairobi",
		owner="operations",
		configuration_source="git://config/env-prod",
		rbac_policy="rbac-prod",
		secret_scope_policy="secret-prod",
		approval_recorded=True,
	)
	path = service.create_promotion_path(
		path_id="path-dev-prod",
		tenant_id="tenant-env",
		source_environment_id="env-dev",
		target_environment_id="env-prod",
		deployment_link="depl:release-42",
		rollback_environment_id="env-dev",
		approval_recorded=True,
	)
	run = service.run_promotion(
		run_id="promotion-1",
		tenant_id="tenant-env",
		promotion_path_id="path-dev-prod",
		requested_by="release-manager",
		artifact_ref="artifact:42",
		approval_recorded=True,
	)
	drift = service.record_drift(
		report_id="drift-1",
		tenant_id="tenant-env",
		environment_id="env-prod",
		declared_version="git:abc",
		observed_version="live:def",
		changed_items=2,
		total_items=100,
	)
	scope = service.register_secret_scope(
		scope_id="scope-prod",
		tenant_id="tenant-env",
		environment_id="env-prod",
		name="prod-db",
		policy_ref="keym-policy-prod",
		secret_refs=("keym://prod/db",),
		access_roles=("envm-admin",),
	)
	model = dashboard_model(service, "tenant-env")

	assert dev["fingerprint"] != prod["fingerprint"]
	assert prod["production_locked"] is True
	assert path["status"] == "approved"
	assert run["status"] == "promoted"
	assert drift["status"] == "minor_drift"
	assert scope["policy_ref"] == "keym-policy-prod"
	assert model["summary"]["environment_count"] == 2
	assert model["summary"]["promotion_run_count"] == 1
	assert model["summary"]["secret_scope_count"] == 1
	assert len(model["audit_events"]) == 6


def test_environment_guardrails_block_missing_policy_inputs():
	service = EnvmService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_environment(
			environment_id="env-1",
			tenant_id="",
			name="No Tenant",
			stage="development",
			region="ke-nairobi",
			owner="platform",
			configuration_source="git://config",
			rbac_policy="rbac",
			secret_scope_policy="secrets",
		)

	with pytest.raises(PermissionError, match="environment_owner_required"):
		service.register_environment(
			environment_id="env-2",
			tenant_id="tenant-env",
			name="No Owner",
			stage="development",
			region="ke-nairobi",
			owner="",
			configuration_source="git://config",
			rbac_policy="rbac",
			secret_scope_policy="secrets",
		)

	with pytest.raises(PermissionError, match="production_approval_required"):
		service.register_environment(
			environment_id="env-3",
			tenant_id="tenant-env",
			name="Production",
			stage="production",
			region="ke-nairobi",
			owner="operations",
			configuration_source="git://config",
			rbac_policy="rbac",
			secret_scope_policy="secrets",
			approval_recorded=False,
		)

	with pytest.raises(PermissionError, match="stage_policy_required"):
		service.register_environment(
			environment_id="env-4",
			tenant_id="tenant-env",
			name="Unknown",
			stage="sandbox",
			region="ke-nairobi",
			owner="platform",
			configuration_source="git://config",
			rbac_policy="rbac",
			secret_scope_policy="secrets",
		)


def test_promotion_secret_and_drift_guardrails():
	service = EnvmService()
	service.register_environment("env-dev", "tenant-env", "Development", "development", "ke-nairobi", "platform", "git://dev", "rbac-dev", "secret-dev")
	service.register_environment("env-prod", "tenant-env", "Production", "production", "ke-nairobi", "operations", "git://prod", "rbac-prod", "secret-prod")

	with pytest.raises(PermissionError, match="promotion_path_required"):
		service.create_promotion_path(
			path_id="bad-path",
			tenant_id="tenant-env",
			source_environment_id="env-dev",
			target_environment_id="env-prod",
			deployment_link="depl:1",
			rollback_environment_id="env-dev",
			approval_recorded=True,
			promotion_path_attached=False,
		)

	with pytest.raises(PermissionError, match="production_approval_required"):
		service.create_promotion_path(
			path_id="unapproved-prod",
			tenant_id="tenant-env",
			source_environment_id="env-dev",
			target_environment_id="env-prod",
			deployment_link="depl:1",
			rollback_environment_id="env-dev",
			approval_recorded=False,
		)

	with pytest.raises(PermissionError, match="secret_policy_required"):
		service.register_secret_scope(
			scope_id="bad-scope",
			tenant_id="tenant-env",
			environment_id="env-prod",
			name="bad",
			policy_ref="",
			secret_refs=("keym://prod/db",),
			access_roles=("envm-admin",),
			secret_policy_attached=False,
		)

	drift = service.record_drift(
		report_id="drift-review",
		tenant_id="tenant-env",
		environment_id="env-prod",
		declared_version="git:abc",
		observed_version="live:def",
		changed_items=12,
		total_items=100,
	)

	assert drift["status"] == "review_required"
	assert service.dashboard_summary("tenant-env")["review_required_drift_count"] == 1
