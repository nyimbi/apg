"""TENS package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.tens import api, views
from capabilities.common.tens.service import TensService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("materialized_contract_tens", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "tens"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_tens", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "tens" in model["capabilities"]


def test_legacy_tenant_migration_lifecycle_executes():
	service = TensService()

	legacy = service.register_legacy_tenant(
		tenant_id="tenant-a",
		legacy_tenant_id="legacy-001",
		source_system="erp-legacy",
		owner="tenant-owner",
		compatibility_scope="finance",
		days_since_activity=30,
	)
	mapping = service.map_tenant(
		tenant_id="tenant-a",
		legacy_tenant_id=legacy["id"],
		apg_tenant_id="apg-tenant-001",
		validated_by="migration-lead",
		validation_ref="validation://mapping/1",
	)
	boundary = service.validate_access_boundary(
		tenant_id="tenant-a",
		legacy_tenant_id=legacy["id"],
		auth_boundary_ref="auth://boundary/1",
		role_mapping_ref="roles://mapping/1",
		isolation_validation_ref="isolation://tenant/1",
		privileged_review_ref="review://privileged/1",
		actor="security-lead",
	)
	migration = service.create_migration_plan(
		tenant_id="tenant-a",
		legacy_tenant_id=legacy["id"],
		mapping_id=mapping["id"],
		owner="migration-lead",
		approval_ref="approval://migration/1",
		rollback_plan_ref="rollback://tenant/1",
		post_migration_validation_ref="validation://post/1",
	)
	completed = service.complete_migration(
		tenant_id="tenant-a",
		migration_id=migration["id"],
		actor="migration-lead",
		post_migration_validation_ref="validation://post/complete",
	)
	deprecation = service.record_deprecation_plan(
		tenant_id="tenant-a",
		legacy_tenant_id=legacy["id"],
		owner="tenant-owner",
		deprecation_ref="deprecation://legacy-001",
		target_date="2026-12-31",
	)
	summary = service.dashboard_summary("tenant-a")

	assert legacy["status"] == "active"
	assert mapping["status"] == "validated"
	assert boundary["status"] == "validated"
	assert migration["status"] == "approved"
	assert completed["status"] == "completed"
	assert deprecation["status"] == "planned"
	assert summary["legacy_tenant_count"] == 1
	assert summary["mapped_tenant_count"] == 1
	assert summary["completed_migration_count"] == 1
	assert summary["deprecation_count"] == 1


def test_guardrails_require_tenant_owner_mapping_boundary_approval_and_deprecation():
	service = TensService()

	try:
		service.register_legacy_tenant("", "legacy-x", "system", "owner", "scope")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	try:
		service.register_legacy_tenant("tenant-a", "legacy-x", "system", "", "scope")
	except PermissionError as exc:
		assert str(exc) == "legacy_owner_required"
	else:
		raise AssertionError("missing owner was accepted")

	stale = service.register_legacy_tenant("tenant-a", "legacy-stale", "system", "owner", "scope", days_since_activity=240)
	assert stale["status"] == "stale"
	assert stale["required_actions"] == ["review_legacy_tenant"]

	try:
		service.map_tenant("tenant-a", stale["id"], "apg-tenant", "validator", "", mapping_validated=False)
	except PermissionError as exc:
		assert str(exc) == "mapping_validation_required"
	else:
		raise AssertionError("unvalidated mapping was accepted")

	mapping = service.map_tenant("tenant-a", stale["id"], "apg-tenant", "validator", "validation://mapping")

	try:
		service.validate_access_boundary("tenant-a", stale["id"], "", "roles://mapping", "isolation://tenant", "review://privileged", "actor")
	except PermissionError as exc:
		assert str(exc) == "auth_boundary_required"
	else:
		raise AssertionError("missing auth boundary was accepted")

	service.validate_access_boundary("tenant-a", stale["id"], "auth://boundary", "roles://mapping", "isolation://tenant", "review://privileged", "actor")

	try:
		service.create_migration_plan("tenant-a", stale["id"], mapping["id"], "owner", "", "rollback://tenant", "validation://post")
	except PermissionError as exc:
		assert str(exc) == "migration_approval_required"
	else:
		raise AssertionError("unapproved migration was accepted")

	try:
		service.record_deprecation_plan("tenant-a", stale["id"], "owner", "", "2026-12-31")
	except PermissionError as exc:
		assert str(exc) == "deprecation_plan_required"
	else:
		raise AssertionError("missing deprecation plan was accepted")


def test_api_and_view_models_expose_legacy_tenant_surfaces():
	local_service = TensService()
	api.SERVICE = local_service

	legacy = api.register_legacy_tenant({
		"tenant_id": "tenant-b",
		"legacy_tenant_id": "legacy-002",
		"source_system": "crm-legacy",
		"owner": "crm-owner",
		"compatibility_scope": "customers",
	})
	mapping = api.map_tenant({
		"tenant_id": "tenant-b",
		"legacy_tenant_id": legacy["id"],
		"apg_tenant_id": "apg-tenant-002",
		"validated_by": "migration-lead",
		"validation_ref": "validation://mapping/2",
	})
	api.validate_access_boundary({
		"tenant_id": "tenant-b",
		"legacy_tenant_id": legacy["id"],
		"auth_boundary_ref": "auth://boundary/2",
		"role_mapping_ref": "roles://mapping/2",
		"isolation_validation_ref": "isolation://tenant/2",
		"privileged_review_ref": "review://privileged/2",
		"actor": "security-lead",
	})
	migration = api.create_migration_plan({
		"tenant_id": "tenant-b",
		"legacy_tenant_id": legacy["id"],
		"mapping_id": mapping["id"],
		"owner": "migration-lead",
		"approval_ref": "approval://migration/2",
		"rollback_plan_ref": "rollback://tenant/2",
		"post_migration_validation_ref": "validation://post/2",
	})
	api.complete_migration({
		"tenant_id": "tenant-b",
		"migration_id": migration["id"],
		"actor": "migration-lead",
		"post_migration_validation_ref": "validation://post/complete/2",
	})
	api.record_deprecation_plan({
		"tenant_id": "tenant-b",
		"legacy_tenant_id": legacy["id"],
		"owner": "crm-owner",
		"deprecation_ref": "deprecation://legacy-002",
		"target_date": "2026-12-31",
	})

	status = api.capability_status("tenant-b")
	legacy_payload = api.list_tenant_legacy("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	registry = views.legacy_tenant_registry_model(local_service, "tenant-b")
	mappings = views.mapping_workbench_model(local_service, "tenant-b")
	migrations = views.migration_queue_model(local_service, "tenant-b")
	boundaries = views.boundary_review_model(local_service, "tenant-b")
	deprecations = views.deprecation_model(local_service, "tenant-b")
	audit = views.audit_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["legacy_tenant_count"] == 1
	assert legacy_payload["summary"]["completed_migration_count"] == 1
	assert dashboard["summary"]["deprecation_count"] == 1
	assert registry["route"] == "/tens/tenants"
	assert mappings["validation_required"] is True
	assert migrations["states"] == ["planned", "approved", "executing", "completed", "blocked"]
	assert boundaries["required_evidence"] == ["auth_boundary", "role_mapping", "tenant_isolation", "privileged_access_review"]
	assert deprecations["plan_required"] is True
	assert audit["events"]
	assert settings["theme"]["name"] == "tens_legacy_tenant_migration"
