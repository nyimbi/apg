"""Regression coverage for the NCOD executable capability contract."""

import pytest

from capabilities.common.ncod import register_capability
from capabilities.common.ncod.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.ncod.service import NcodService
from capabilities.common.ncod.views import (
	app_library_model,
	builder_model,
	component_catalog_model,
	connector_bindings_model,
	dashboard_model,
	page_composer_model,
	publish_center_model,
	settings_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-builder", {"apps": {"publish_approval_required": False}})

	assert contract["capability"] == "ncod"
	assert contract["configuration"]["tenant_id"] == "tenant-builder"
	assert contract["configuration"]["apps"]["publish_approval_required"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "apps", "builder", "extensions", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "apps", "builder", "pages", "components", "publishing", "connectors", "settings"}
	assert contract["ui"]["api_prefix"] == "/ncod/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "page_composer" in contract["theme"]["components"]


def test_rule_engine_enforces_no_code_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_app",
		"app_owner_assigned": False,
		"approval_recorded": False,
		"script_extension_present": True,
		"script_policy_attached": False,
		"external_connector_present": True,
		"connector_policy_attached": False,
		"production_change": True,
		"change_review_recorded": False
	})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_app", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "app_requires_owner", "script_extension_requires_policy", "external_connector_requires_policy", "production_change_requires_review"}
	assert publish_result["matched_rules"] == ["publish_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "ncod"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "ncod_app_builder"
	assert registration["ui_components"]["builder"] == "/ncod/builder"
	assert "scpt" in registration["dependencies"]
	assert "ncod:build" in registration["permissions"]


def test_ncod_lifecycle_is_executable():
	service = NcodService()
	tenant_id = "tenant-builder"

	app = service.create_app(
		app_id="field-service",
		tenant_id=tenant_id,
		name="Field Service Console",
		owner="ops-platform",
		description="Dispatch and work-order app",
		rbac_policy_ref="rbac:field-service",
		data_residency_policy_ref="residency:ke",
		accessibility_checked=True,
	)
	page = service.add_page(
		page_id="work-orders",
		tenant_id=tenant_id,
		app_id=app["id"],
		name="Work Orders",
		route="work-orders",
		layout="dashboard",
	)
	component = service.add_component(
		component_id="work-order-table",
		tenant_id=tenant_id,
		page_id=page["id"],
		component_type="table",
		name="Work Order Table",
		props={"page_size": 25},
		bindings={"rows": "work_orders"},
		accessibility_label="Work orders table",
		order=10,
	)
	data_binding = service.bind_data_source(
		binding_id="work-orders-data",
		tenant_id=tenant_id,
		app_id=app["id"],
		name="Work Orders",
		source_type="entity",
		source_ref="fsm.work_order",
		schema={"fields": ["id", "status", "assignee"]},
		policy_ref="data:work-orders",
	)
	workflow = service.attach_workflow(
		binding_id="dispatch-flow",
		tenant_id=tenant_id,
		app_id=app["id"],
		trigger="on_dispatch",
		workflow_ref="wflo:dispatch-technician",
	)
	script = service.add_script_extension(
		extension_id="validate-dispatch",
		tenant_id=tenant_id,
		app_id=app["id"],
		name="Validate Dispatch",
		hook="before_submit",
		script_ref="scpt:validate-dispatch",
		policy_ref="script-policy:dispatch",
	)
	connector = service.add_connector_binding(
		binding_id="maps-connector",
		tenant_id=tenant_id,
		app_id=app["id"],
		name="Maps",
		connector_ref="conn:maps",
		policy_ref="connector-policy:maps",
		scopes=["geocode"],
	)
	validation = service.validate_app(
		validation_id="validate-field-service",
		tenant_id=tenant_id,
		app_id=app["id"],
	)
	release = service.publish_app(
		release_id="release-field-service-prod",
		tenant_id=tenant_id,
		app_id=app["id"],
		target_environment="production",
		approval_recorded=True,
		approval_ref="approval:field-service",
		change_review_recorded=True,
	)

	assert page["route"] == "/work-orders"
	assert component["component_type"] == "table"
	assert data_binding["validated"] is True
	assert workflow["enabled"] is True
	assert script["status"] == "approved"
	assert connector["scopes"] == ["geocode"]
	assert validation["passed"] is True
	assert release["status"] == "production"
	assert service.list_apps(tenant_id)[0]["status"] == "published"

	summary = service.dashboard_summary(tenant_id)
	assert summary["app_count"] == 1
	assert summary["published_app_count"] == 1
	assert summary["component_count"] == 1
	assert summary["release_count"] == 1

	assert dashboard_model(service, tenant_id)["summary"]["app_count"] == 1
	assert app_library_model(service, tenant_id)["apps"][0]["id"] == "field-service"
	assert builder_model(service, tenant_id)["workflow_bindings"][0]["workflow_ref"] == "wflo:dispatch-technician"
	assert page_composer_model(service, tenant_id)["components"][0]["id"] == "work-order-table"
	assert component_catalog_model(service, tenant_id)["components"][0]["accessibility_label"] == "Work orders table"
	assert publish_center_model(service, tenant_id)["releases"][0]["approval_recorded"] is True
	assert connector_bindings_model(service, tenant_id)["connector_bindings"][0]["connector_ref"] == "conn:maps"
	assert settings_model(service, tenant_id)["audit_events"]


def test_ncod_service_enforces_policy_guardrails():
	service = NcodService()
	tenant_id = "tenant-guardrails"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_app(
			app_id="missing-tenant",
			tenant_id="",
			name="Missing Tenant",
			owner="builder",
		)

	with pytest.raises(PermissionError, match="app_owner_required"):
		service.create_app(
			app_id="missing-owner",
			tenant_id=tenant_id,
			name="Missing Owner",
			owner="",
		)

	app = service.create_app(
		app_id="guardrail-app",
		tenant_id=tenant_id,
		name="Guardrail App",
		owner="builder",
		rbac_policy_ref="rbac:guardrail",
		data_residency_policy_ref="residency:guardrail",
		accessibility_checked=True,
	)
	page = service.add_page(
		page_id="guardrail-page",
		tenant_id=tenant_id,
		app_id=app["id"],
		name="Guardrail Page",
		route="/guardrail",
	)

	with pytest.raises(PermissionError, match="accessibility_label_required"):
		service.add_component(
			component_id="bad-input",
			tenant_id=tenant_id,
			page_id=page["id"],
			component_type="input",
			name="Bad Input",
		)

	service.add_component(
		component_id="good-input",
		tenant_id=tenant_id,
		page_id=page["id"],
		component_type="input",
		name="Good Input",
		accessibility_label="Good input",
	)
	service.bind_data_source(
		binding_id="invalid-data",
		tenant_id=tenant_id,
		app_id=app["id"],
		name="Invalid Data",
		source_type="entity",
		source_ref="bad.source",
		schema={"bad": "schema"},
	)
	validation = service.validate_app(
		validation_id="validate-bad-data",
		tenant_id=tenant_id,
		app_id=app["id"],
	)
	assert validation["passed"] is False
	assert "data_bindings_valid" in validation["issues"]

	with pytest.raises(PermissionError, match="script_policy_required"):
		service.add_script_extension(
			extension_id="unsafe-script",
			tenant_id=tenant_id,
			app_id=app["id"],
			name="Unsafe Script",
			hook="before_submit",
			script_ref="scpt:unsafe",
			policy_ref="",
		)

	with pytest.raises(PermissionError, match="connector_policy_required"):
		service.add_connector_binding(
			binding_id="unsafe-connector",
			tenant_id=tenant_id,
			app_id=app["id"],
			name="Unsafe Connector",
			connector_ref="conn:unsafe",
			policy_ref="",
		)

	service.bind_data_source(
		binding_id="valid-data",
		tenant_id=tenant_id,
		app_id=app["id"],
		name="Valid Data",
		source_type="entity",
		source_ref="good.source",
		schema={"fields": ["id"]},
	)
	# Remove the invalid binding by replacing service state through a valid new app path.
	clean = service.create_app(
		app_id="clean-app",
		tenant_id=tenant_id,
		name="Clean App",
		owner="builder",
		rbac_policy_ref="rbac:clean",
		data_residency_policy_ref="residency:clean",
		accessibility_checked=True,
	)
	clean_page = service.add_page(
		page_id="clean-page",
		tenant_id=tenant_id,
		app_id=clean["id"],
		name="Clean Page",
		route="/clean",
	)
	service.add_component(
		component_id="clean-button",
		tenant_id=tenant_id,
		page_id=clean_page["id"],
		component_type="button",
		name="Clean Button",
		accessibility_label="Clean button",
	)
	service.validate_app(
		validation_id="validate-clean",
		tenant_id=tenant_id,
		app_id=clean["id"],
	)
	with pytest.raises(PermissionError, match="publish_approval_required"):
		service.publish_app(
			release_id="publish-no-approval",
			tenant_id=tenant_id,
			app_id=clean["id"],
			target_environment="staging",
			approval_recorded=False,
		)
	with pytest.raises(PermissionError, match="production_change_review_required"):
		service.publish_app(
			release_id="publish-no-review",
			tenant_id=tenant_id,
			app_id=clean["id"],
			target_environment="production",
			approval_recorded=True,
			change_review_recorded=False,
		)

	with pytest.raises(LookupError, match="builder_app_not_found"):
		service.add_page(
			page_id="cross-tenant",
			tenant_id="other-tenant",
			app_id=clean["id"],
			name="Cross Tenant",
			route="/cross",
		)
