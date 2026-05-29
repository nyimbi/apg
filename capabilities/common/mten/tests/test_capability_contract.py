"""Regression coverage for the MTEN executable capability contract."""

from __future__ import annotations

import pytest

from .. import api_helpers, register_capability, view_models
from ..capability_contract import evaluate_capability_rules, get_capability_contract
from ..mten_runtime import MtenService


def _ready_service() -> MtenService:
	service = MtenService()
	service.register_tenant(
		target_tenant_id="tenant-alpha",
		tenant_id="platform",
		name="tenant-alpha",
		owner="tenant-owner",
		tier="enterprise",
		primary_domain="alpha.example.com",
		projected_compute_units=900,
	)
	service.activate_tenant(
		target_tenant_id="tenant-alpha",
		tenant_id="platform",
		actor="tenant-owner",
	)
	return service


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-a",
		{"resources": {"quota_alert_threshold_percent": 90}}
	)

	assert contract["capability"] == "mten"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration"]["resources"]["quota_alert_threshold_percent"] == 90
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"provisioning",
		"isolation",
		"resources",
		"orchestration",
		"governance",
		"analytics",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 10
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"tenants",
		"provisioning",
		"capacity_approvals",
		"isolation",
		"live_migrations",
		"templates",
		"analytics",
		"optimization",
		"audit",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/mten/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "12px"
	assert "tenant_health_card" in contract["theme"]["components"]
	assert "capacity_approval_queue" in contract["theme"]["components"]
	assert "isolation_incident_panel" in contract["theme"]["components"]
	assert "live_migration_runbook" in contract["theme"]["components"]


def test_rule_engine_enforces_multi_tenant_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"cross_tenant_operation": True,
		"tenant_membership_confirmed": False,
		"tenant_status": "suspended",
		"requested_operation_is_mutation": True,
		"custom_domain_requested": True,
		"dns_validated": False,
		"projected_compute_units": 1400,
		"capacity_approval_recorded": False,
		"isolation_boundary_encrypted": False,
		"isolation_breach_detected": True,
		"tenant_suspended": False,
		"requested_operation": "live_migration",
		"runbook_attached": False,
	})
	capacity_review = evaluate_capability_rules({
		"operation": "approve_capacity",
		"capacity_reviewer_same_as_requester": True,
	})
	migration_review = evaluate_capability_rules({
		"operation": "approve_live_migration",
		"migration_reviewer_same_as_requester": True,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"cross_tenant_access_requires_membership",
		"suspended_tenants_block_mutations",
		"custom_domain_requires_dns_validation",
		"capacity_overcommit_requires_review",
		"isolation_boundary_requires_encryption",
		"isolation_breach_requires_suspension",
		"live_migration_requires_runbook",
	}
	assert capacity_review["matched_rules"] == ["capacity_review_requires_independent_reviewer"]
	assert migration_review["matched_rules"] == ["live_migration_requires_independent_reviewer"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mten_control_fabric"
	assert registration["ui_components"]["capacity_approvals"] == "/mten/capacity/approvals"
	assert registration["ui_components"]["isolation"] == "/mten/isolation"
	assert registration["ui_components"]["live_migrations"] == "/mten/migrations"
	assert registration["ui_components"]["audit"] == "/mten/audit"
	assert "mten:approve_capacity" in registration["permissions"]
	assert "mten:migrate" in registration["permissions"]
	assert "auth_rbac" in registration["dependencies"]


def test_service_runs_tenant_capacity_isolation_and_migration_lifecycle():
	service = MtenService()
	approval = service.request_capacity_approval(
		approval_id="capacity-alpha",
		tenant_id="platform",
		target_tenant_id="tenant-alpha",
		requested_by="tenant-owner",
		projected_compute_units=1600,
		justification="Enterprise launch capacity forecast.",
	)
	approved_capacity = service.decide_capacity_approval(
		approval_id=approval["id"],
		tenant_id="platform",
		reviewer="capacity-reviewer",
		decision="approved",
		notes="Forecast, quota, and cost center verified.",
	)
	tenant = service.register_tenant(
		target_tenant_id="tenant-alpha",
		tenant_id="platform",
		name="tenant-alpha",
		owner="tenant-owner",
		tier="enterprise",
		primary_domain="alpha.example.com",
		custom_domain="app.alpha.example.com",
		dns_validated=True,
		projected_compute_units=1500,
		capacity_approval_id=approved_capacity["id"],
	)
	activated = service.activate_tenant(
		target_tenant_id=tenant["id"],
		tenant_id="platform",
		actor="tenant-owner",
	)
	migration = service.request_live_migration(
		migration_id="migration-alpha",
		tenant_id="platform",
		target_tenant_id=tenant["id"],
		requested_by="tenant-owner",
		source_provider="aws",
		target_provider="azure",
		runbook="Run dual-write, drain traffic, verify data checksums, cut DNS.",
	)
	approved_migration = service.decide_live_migration(
		migration_id=migration["id"],
		tenant_id="platform",
		reviewer="migration-reviewer",
		decision="approved",
		notes="Runbook and rollback window verified.",
	)
	executed = service.execute_live_migration(
		migration_id=approved_migration["id"],
		tenant_id="platform",
		actor="migration-runner",
	)
	incident = service.record_isolation_incident(
		incident_id="incident-alpha",
		tenant_id="platform",
		target_tenant_id=tenant["id"],
		detected_by="isolation-sensor",
		breach_summary="Unexpected shared cache namespace detected.",
	)
	reactivated = service.reactivate_tenant(
		target_tenant_id=tenant["id"],
		tenant_id="platform",
		actor="security-reviewer",
		evidence="Cache namespace rotated and isolation scan passed.",
	)
	model = view_models.dashboard_model(service, "platform")

	assert activated["status"] == "active"
	assert executed["status"] == "completed"
	assert incident["suspended"] is True
	assert reactivated["status"] == "active"
	assert model["summary"]["tenant_count"] == 1
	assert model["summary"]["capacity_approval_count"] == 1
	assert model["summary"]["isolation_incident_count"] == 1
	assert model["summary"]["live_migration_count"] == 1
	assert {event["event_type"] for event in model["governance_events"]} >= {
		"capacity_approval_requested",
		"capacity_approval_decided",
		"tenant_registered",
		"tenant_activated",
		"live_migration_requested",
		"live_migration_decided",
		"live_migration_executed",
		"isolation_incident_recorded",
		"tenant_reactivated",
	}


def test_service_blocks_tenant_guardrail_violations():
	service = MtenService()

	with pytest.raises(PermissionError, match="dns_validation_required"):
		service.register_tenant(
			target_tenant_id="tenant-domain",
			tenant_id="platform",
			name="tenant-domain",
			owner="tenant-owner",
			primary_domain="domain.example.com",
			custom_domain="app.domain.example.com",
			dns_validated=False,
		)

	with pytest.raises(PermissionError, match="isolation_boundary_encryption_required"):
		service.register_tenant(
			target_tenant_id="tenant-plain-boundary",
			tenant_id="platform",
			name="tenant-plain-boundary",
			owner="tenant-owner",
			primary_domain="plain.example.com",
			isolation_boundary_encrypted=False,
		)

	with pytest.raises(PermissionError, match="capacity_review_required"):
		service.register_tenant(
			target_tenant_id="tenant-overcommit",
			tenant_id="platform",
			name="tenant-overcommit",
			owner="tenant-owner",
			primary_domain="over.example.com",
			projected_compute_units=1400,
		)

	approval = service.request_capacity_approval(
		approval_id="capacity-self",
		tenant_id="platform",
		target_tenant_id="tenant-big",
		requested_by="tenant-owner",
		projected_compute_units=1500,
		justification="Large launch.",
	)
	with pytest.raises(PermissionError, match="independent_capacity_reviewer_required"):
		service.decide_capacity_approval(
			approval_id=approval["id"],
			tenant_id="platform",
			reviewer="tenant-owner",
			decision="approved",
			notes="Self review should fail.",
		)
	with pytest.raises(ValueError, match="capacity_reviewer_notes_required"):
		service.decide_capacity_approval(
			approval_id=approval["id"],
			tenant_id="platform",
			reviewer="capacity-reviewer",
			decision="approved",
			notes="",
		)
	rejected = service.decide_capacity_approval(
		approval_id=approval["id"],
		tenant_id="platform",
		reviewer="capacity-reviewer",
		decision="rejected",
		notes="Capacity not justified.",
	)
	assert rejected["status"] == "rejected"
	with pytest.raises(PermissionError, match="capacity_review_required"):
		service.register_tenant(
			target_tenant_id="tenant-big",
			tenant_id="platform",
			name="tenant-big",
			owner="tenant-owner",
			primary_domain="big.example.com",
			projected_compute_units=1500,
			capacity_approval_id=approval["id"],
		)

	service = _ready_service()
	service.record_isolation_incident(
		incident_id="incident-alpha",
		tenant_id="platform",
		target_tenant_id="tenant-alpha",
		detected_by="isolation-sensor",
		breach_summary="Boundary breach.",
	)
	with pytest.raises(PermissionError, match="tenant_suspended"):
		service.request_live_migration(
			migration_id="migration-suspended",
			tenant_id="platform",
			target_tenant_id="tenant-alpha",
			requested_by="tenant-owner",
			source_provider="aws",
			target_provider="gcp",
			runbook="Drain and migrate.",
		)
	service.reactivate_tenant(
		target_tenant_id="tenant-alpha",
		tenant_id="platform",
		actor="security-reviewer",
		evidence="Isolation verified.",
	)
	with pytest.raises(PermissionError, match="live_migration_runbook_required"):
		service.request_live_migration(
			migration_id="migration-no-runbook",
			tenant_id="platform",
			target_tenant_id="tenant-alpha",
			requested_by="tenant-owner",
			source_provider="aws",
			target_provider="gcp",
			runbook="",
		)
	migration = service.request_live_migration(
		migration_id="migration-review",
		tenant_id="platform",
		target_tenant_id="tenant-alpha",
		requested_by="tenant-owner",
		source_provider="aws",
		target_provider="gcp",
		runbook="Drain traffic and verify checksums.",
	)
	with pytest.raises(PermissionError, match="independent_migration_reviewer_required"):
		service.decide_live_migration(
			migration_id=migration["id"],
			tenant_id="platform",
			reviewer="tenant-owner",
			decision="approved",
			notes="Self review should fail.",
		)
	with pytest.raises(ValueError, match="migration_reviewer_notes_required"):
		service.decide_live_migration(
			migration_id=migration["id"],
			tenant_id="platform",
			reviewer="migration-reviewer",
			decision="approved",
			notes="",
		)
	with pytest.raises(PermissionError, match="live_migration_approval_required"):
		service.execute_live_migration(
			migration_id=migration["id"],
			tenant_id="platform",
			actor="migration-runner",
		)


def test_tenant_local_duplicate_ids_are_isolated():
	service = MtenService()
	for platform_tenant, owner in (("platform-a", "owner-a"), ("platform-b", "owner-b")):
		service.register_tenant(
			target_tenant_id="shared-tenant",
			tenant_id=platform_tenant,
			name="shared-tenant",
			owner=owner,
			primary_domain=f"{platform_tenant}.example.com",
		)

	assert service.list_tenants("platform-a")[0]["owner"] == "owner-a"
	assert service.list_tenants("platform-b")[0]["owner"] == "owner-b"
	with pytest.raises(ValueError, match="duplicate tenant"):
		service.register_tenant(
			target_tenant_id="shared-tenant",
			tenant_id="platform-a",
			name="shared-tenant",
			owner="owner-a2",
			primary_domain="again.example.com",
		)


def test_api_helpers_and_view_models_share_default_state():
	api_helpers.SERVICE = MtenService()
	tenant = api_helpers.register_tenant({
		"id": "tenant-api",
		"tenant_id": "platform-api",
		"name": "tenant-api",
		"owner": "api-owner",
		"primary_domain": "api.example.com",
	})
	api_helpers.activate_tenant({
		"id": tenant["id"],
		"tenant_id": "platform-api",
		"actor": "api-owner",
	})
	api_helpers.request_live_migration({
		"id": "migration-api",
		"tenant_id": "platform-api",
		"target_tenant_id": tenant["id"],
		"requested_by": "api-owner",
		"source_provider": "aws",
		"target_provider": "azure",
		"runbook": "Drain traffic and verify checksums.",
	})
	model = view_models.dashboard_model(tenant_id="platform-api")
	provisioning = view_models.provisioning_model(tenant_id="platform-api")
	migrations = view_models.migration_model(tenant_id="platform-api")
	governance = view_models.governance_model(tenant_id="platform-api")

	assert api_helpers.capability_status("platform-api")["tenant_count"] == 1
	assert model["summary"]["live_migration_count"] == 1
	assert provisioning["active"][0]["id"] == "tenant-api"
	assert migrations["migrations"][0]["id"] == "migration-api"
	assert governance["events"]
