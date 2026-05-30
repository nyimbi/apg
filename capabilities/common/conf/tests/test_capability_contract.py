"""Regression coverage for the CONF executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.conf import api, register_capability, views
from capabilities.common.conf.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.conf.service import ConfService


def _ready_service() -> ConfService:
	service = ConfService()
	service.create_record(
		record_id="database-url",
		tenant_id="tenant-conf",
		key="apg.database.url",
		value="postgresql://primary",
		environment="staging",
		owner="platform-owner",
		contains_secrets=False,
		secrets_encrypted=False,
		metadata={"system": "core"},
	)
	return service


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-conf",
		{"automation": {"auto_remediation_enabled": True}}
	)

	assert contract["capability"] == "conf"
	assert contract["configuration"]["tenant_id"] == "tenant-conf"
	assert contract["configuration"]["automation"]["auto_remediation_enabled"] is True
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"gitops",
		"security",
		"automation",
		"change_management",
		"conf_agents",
		"observability",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 13
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"resources",
		"templates",
		"changes",
		"approvals",
		"policies",
		"deployments",
		"drift",
		"drift_remediation",
		"agents",
		"gitops",
		"audit",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/api/v1/config"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["conf_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "change_approval_queue" in contract["theme"]["components"]
	assert "drift_remediation_queue" in contract["theme"]["components"]
	assert "configuration_agent_roster" in contract["theme"]["components"]
	assert "configuration_audit_timeline" in contract["theme"]["components"]


def test_rule_engine_denies_unsafe_configuration_operations():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_record",
		"configuration_owner_assigned": False,
		"requested_operation": "apply",
		"validation_passed": False,
		"target_environment": "production",
		"change_approved": False,
		"contains_secrets": True,
		"secrets_encrypted": False,
		"rollback_plan_available": False,
		"drift_detected": True,
		"remediation_plan_available": False,
	})
	review_result = evaluate_capability_rules({
		"operation": "approve_change",
		"change_reviewer_same_as_requester": True,
	})
	drift_review_result = evaluate_capability_rules({
		"operation": "approve_drift_remediation",
		"drift_reviewer_same_as_detector": True,
	})
	batch_result = evaluate_capability_rules({
		"operation": "configuration_batch",
		"event_stream": "queue",
	})
	agent_result = evaluate_capability_rules({
		"operation": "register_conf_agent",
		"runtime_supported": False,
		"role_supported": False,
	})
	agent_action_result = evaluate_capability_rules({
		"operation": "conf_agent_action",
		"privileged_action": True,
		"human_approved": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"configuration_record_requires_owner",
		"validate_before_apply",
		"production_changes_require_approval",
		"encrypted_secrets_required",
		"drift_requires_remediation_plan",
		"production_deployment_requires_rollback",
	}
	assert review_result["matched_rules"] == ["change_review_requires_independent_reviewer"]
	assert drift_review_result["matched_rules"] == ["drift_review_requires_independent_reviewer"]
	assert batch_result["matched_rules"] == ["bytewax_event_stream_required"]
	assert set(agent_result["matched_rules"]) == {"conf_agent_runtime_supported", "conf_agent_role_supported"}
	assert agent_action_result["matched_rules"] == ["conf_agent_privileged_action_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "conf"
	assert "auth_rbac" in registration["dependencies"]
	assert registration["ui_components"]["changes"] == "/config/changes"
	assert registration["ui_components"]["approvals"] == "/config/approvals"
	assert registration["ui_components"]["drift_remediation"] == "/config/drift/remediation"
	assert registration["ui_components"]["agents"] == "/config/agents"
	assert registration["ui_components"]["audit"] == "/config/audit"
	assert "conf:approve" in registration["permissions"]
	assert "conf:remediate" in registration["permissions"]
	assert "conf:agent_manage" in registration["permissions"]


def test_service_promotes_configuration_change_and_reviews_drift():
	service = _ready_service()
	change = service.request_change(
		change_id="change-prod-url",
		tenant_id="tenant-conf",
		record_id="database-url",
		target_environment="production",
		requested_by="platform-owner",
		summary="Promote database URL after migration.",
		proposed_value="postgresql://primary-prod",
		validation_passed=True,
		rollback_plan="Revert apg.database.url to postgresql://primary.",
	)
	approved_change = service.decide_change(
		change_id=change["id"],
		tenant_id="tenant-conf",
		reviewer="change-manager",
		decision="approved",
		notes="Validation, maintenance window, and rollback evidence checked.",
	)
	deployment = service.deploy_change(
		deployment_id="deploy-prod-url",
		tenant_id="tenant-conf",
		change_id=approved_change["id"],
		requested_by="release-engineer",
		strategy="rolling",
		change_approved=False,
	)
	remediation = service.request_drift_remediation(
		remediation_id="drift-prod-url",
		tenant_id="tenant-conf",
		record_id="database-url",
		detected_by="drift-agent",
		drift_summary="Runtime URL differs from declared production URL.",
		remediation_plan="Re-apply approved production URL through deployment center.",
	)
	approved_remediation = service.decide_drift_remediation(
		remediation_id=remediation["id"],
		tenant_id="tenant-conf",
		reviewer="ops-reviewer",
		decision="approved",
		notes="Plan remediates only declared drift and preserves rollback.",
	)
	agent = service.register_conf_agent(
		agent_id="agent-conf",
		tenant_id="tenant-conf",
		name="Configuration Reviewer",
		runtime="codex",
		role="configuration_reviewer",
		purpose="review configuration changes",
		owner="ops-reviewer",
	)
	model = views.dashboard_model(service, "tenant-conf")

	assert deployment["status"] == "completed"
	assert deployment["applied_version"] == 2
	assert service.list_records("tenant-conf")[0]["environment"] == "production"
	assert service.list_records("tenant-conf")[0]["value"] == "postgresql://primary-prod"
	assert approved_remediation["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert service.governance_summary("tenant-conf")["audit_event_count"] >= 5
	assert model["summary"]["deployment_count"] == 1
	assert model["summary"]["drift_remediation_count"] == 1
	assert model["summary"]["agent_count"] == 1
	assert model["streaming"]["processor"] == "bytewax"


def test_service_enforces_change_deployment_and_drift_guardrails():
	service = ConfService()

	with pytest.raises(PermissionError, match="configuration_owner_required"):
		service.create_record(
			record_id="missing-owner",
			tenant_id="tenant-conf",
			key="apg.logging.level",
			value="INFO",
			environment="development",
			owner="",
		)

	with pytest.raises(PermissionError, match="secret_encryption_required"):
		service.create_record(
			record_id="plain-secret",
			tenant_id="tenant-conf",
			key="apg.database.password",
			value="not-encrypted",
			environment="development",
			owner="platform-owner",
			contains_secrets=True,
			secrets_encrypted=False,
		)

	service = _ready_service()
	with pytest.raises(PermissionError, match="validation_required"):
		service.request_change(
			change_id="invalid-change",
			tenant_id="tenant-conf",
			record_id="database-url",
			target_environment="staging",
			requested_by="platform-owner",
			summary="Invalid URL.",
			proposed_value="bad",
			validation_passed=False,
		)

	change = service.request_change(
		change_id="prod-change",
		tenant_id="tenant-conf",
		record_id="database-url",
		target_environment="production",
		requested_by="platform-owner",
		summary="Production URL update.",
		proposed_value="postgresql://primary-prod",
		validation_passed=True,
		rollback_plan="Restore previous URL.",
	)
	with pytest.raises(PermissionError, match="production_approval_required"):
		service.deploy_change(
			deployment_id="raw-approval-bypass",
			tenant_id="tenant-conf",
			change_id=change["id"],
			requested_by="platform-owner",
			change_approved=True,
		)
	with pytest.raises(PermissionError, match="independent_change_reviewer_required"):
		service.decide_change(
			change_id=change["id"],
			tenant_id="tenant-conf",
			reviewer="platform-owner",
			decision="approved",
			notes="Self approval should fail.",
		)
	with pytest.raises(ValueError, match="configuration_change_notes_required"):
		service.decide_change(
			change_id=change["id"],
			tenant_id="tenant-conf",
			reviewer="change-manager",
			decision="approved",
			notes="",
		)
	rejected = service.decide_change(
		change_id=change["id"],
		tenant_id="tenant-conf",
		reviewer="change-manager",
		decision="rejected",
		notes="Rollback window not confirmed.",
	)
	assert rejected["status"] == "rejected"
	with pytest.raises(PermissionError, match="configuration_change_rejected"):
		service.deploy_change(
			deployment_id="rejected-change",
			tenant_id="tenant-conf",
			change_id=change["id"],
			requested_by="platform-owner",
		)

	approved_change = service.request_change(
		change_id="prod-change-no-rollback",
		tenant_id="tenant-conf",
		record_id="database-url",
		target_environment="production",
		requested_by="platform-owner",
		summary="Production update without rollback.",
		proposed_value="postgresql://primary-v2",
		validation_passed=True,
	)
	service.decide_change(
		change_id=approved_change["id"],
		tenant_id="tenant-conf",
		reviewer="change-manager",
		decision="approved",
		notes="Approved but missing rollback plan.",
	)
	with pytest.raises(PermissionError, match="production_rollback_plan_required"):
		service.deploy_change(
			deployment_id="missing-rollback",
			tenant_id="tenant-conf",
			change_id=approved_change["id"],
			requested_by="release-engineer",
		)

	with pytest.raises(PermissionError, match="drift_remediation_required"):
		service.request_drift_remediation(
			remediation_id="missing-plan",
			tenant_id="tenant-conf",
			record_id="database-url",
			detected_by="drift-agent",
			drift_summary="Drift detected.",
			remediation_plan="",
		)
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-conf", 1, "queue")
	with pytest.raises(PermissionError, match="conf_agent_runtime_not_supported"):
		service.register_conf_agent(
			agent_id="bad-agent-runtime",
			tenant_id="tenant-conf",
			name="Bad Runtime",
			runtime="unsupported",
			role="configuration_reviewer",
			purpose="review configuration",
			owner="ops",
		)
	with pytest.raises(PermissionError, match="conf_agent_role_not_supported"):
		service.register_conf_agent(
			agent_id="bad-agent-role",
			tenant_id="tenant-conf",
			name="Bad Role",
			runtime="codex",
			role="unsupported",
			purpose="review configuration",
			owner="ops",
		)
	remediation = service.request_drift_remediation(
		remediation_id="self-drift-review",
		tenant_id="tenant-conf",
		record_id="database-url",
		detected_by="drift-agent",
		drift_summary="Drift detected.",
		remediation_plan="Apply declared state.",
	)
	with pytest.raises(PermissionError, match="independent_drift_reviewer_required"):
		service.decide_drift_remediation(
			remediation_id=remediation["id"],
			tenant_id="tenant-conf",
			reviewer="drift-agent",
			decision="approved",
			notes="Self review should fail.",
		)
	with pytest.raises(ValueError, match="drift_remediation_notes_required"):
		service.decide_drift_remediation(
			remediation_id=remediation["id"],
			tenant_id="tenant-conf",
			reviewer="ops-reviewer",
			decision="approved",
			notes="",
		)

	nonprod_change = service.request_change(
		change_id="staging-rejected",
		tenant_id="tenant-conf",
		record_id="database-url",
		target_environment="staging",
		requested_by="platform-owner",
		summary="Rejected staging value.",
		proposed_value="postgresql://staging-v2",
		validation_passed=True,
	)
	service.decide_change(
		change_id=nonprod_change["id"],
		tenant_id="tenant-conf",
		reviewer="change-manager",
		decision="rejected",
		notes="Staging change rejected.",
	)
	with pytest.raises(PermissionError, match="configuration_change_rejected"):
		service.deploy_change(
			deployment_id="rejected-staging",
			tenant_id="tenant-conf",
			change_id=nonprod_change["id"],
			requested_by="release-engineer",
		)


def test_tenant_local_duplicate_ids_are_isolated():
	service = ConfService()
	for tenant_id, value in (("tenant-a", "A"), ("tenant-b", "B")):
		service.create_record(
			record_id="shared-key",
			tenant_id=tenant_id,
			key="apg.shared.value",
			value=value,
			environment="development",
			owner="owner",
		)

	assert service.list_records("tenant-a")[0]["value"] == "A"
	assert service.list_records("tenant-b")[0]["value"] == "B"
	with pytest.raises(ValueError, match="duplicate configuration record"):
		service.create_record(
			record_id="shared-key",
			tenant_id="tenant-a",
			key="apg.shared.value",
			value="A2",
			environment="development",
			owner="owner",
		)


def test_api_helpers_and_view_models_share_default_state():
	api.SERVICE = ConfService()
	record = api.create_record({
		"id": "api-record",
		"tenant_id": "tenant-api",
		"key": "apg.api.timeout",
		"value": 30,
		"environment": "development",
		"owner": "api-owner",
	})
	change = api.request_change({
		"id": "api-change",
		"tenant_id": "tenant-api",
		"record_id": record["id"],
		"target_environment": "production",
		"requested_by": "api-owner",
		"summary": "Increase timeout.",
		"proposed_value": 45,
		"validation_passed": "true",
		"rollback_plan": "Restore timeout to 30.",
	})
	api.decide_change({
		"id": change["id"],
		"tenant_id": "tenant-api",
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "Timeout and rollback checked.",
	})
	api.deploy_change({
		"id": "api-deploy",
		"tenant_id": "tenant-api",
		"change_id": change["id"],
		"requested_by": "api-release",
	})
	agent = api.register_conf_agent({
		"id": "api-agent",
		"tenant_id": "tenant-api",
		"name": "API Configuration Reviewer",
		"runtime": "claude_code",
		"role": "deployment_reviewer",
		"purpose": "review API-driven deployments",
		"owner": "api-reviewer",
	})
	batch = api.validate_batch({
		"tenant_id": "tenant-api",
		"record_count": 1,
		"event_stream": "bytewax",
	})
	model = views.dashboard_model(tenant_id="tenant-api")
	queue = views.change_queue_model(tenant_id="tenant-api")
	agents = views.agent_model(tenant_id="tenant-api")
	audit = views.audit_model(tenant_id="tenant-api")

	assert api.capability_status("tenant-api")["deployment_count"] == 1
	assert api.capability_status("tenant-api")["agent_count"] == 1
	assert model["summary"]["record_count"] == 1
	assert queue["approved_changes"][0]["id"] == "api-change"
	assert agent["role"] == "deployment_reviewer"
	assert batch["processor"] == "bytewax"
	assert agents["agents"][0]["id"] == "api-agent"
	assert audit["events"]
