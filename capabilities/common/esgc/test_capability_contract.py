"""Regression coverage for the ESGC executable capability contract and service."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.esgc import register_capability
from capabilities.common.esgc.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.esgc.service import EsgcService
from capabilities.common.esgc.views import dashboard_model, esgc_agent_model


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-esgc", {"reporting": {"target_tracking_enabled": False}})

	assert contract["capability"] == "esgc"
	assert contract["configuration"]["tenant_id"] == "tenant-esgc"
	assert contract["configuration"]["reporting"]["target_tracking_enabled"] is False
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"emissions",
		"data_sources",
		"reporting",
		"targets",
		"esgc_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["provides"] == [
		"emissions_inventory",
		"factor_library",
		"activity_emissions",
		"sustainability_reporting",
		"target_tracking",
		"esg_evidence",
		"esgc_agents",
	]
	assert contract["requires"] == ["auth", "conf", "audl", "geos", "pred", "comp"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["batch_mutation_guardrail"] == "batch_esgc_mutation_requires_bytewax"
	assert contract["configuration"]["esgc_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "emissions", "factors", "data_sources", "reports", "targets", "agents", "rules", "audit", "settings"}
	assert contract["theme"]["name"] == "esgc_sustainability_ops"


def test_rule_engine_enforces_esgc_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_inventory", "organization_owner_assigned": False, "factor_source_approved": False, "geospatial_boundary_present": False, "emission_anomaly_detected": True, "anomaly_review_recorded": False})
	report_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_report", "approval_recorded": False, "factor_source_approved": True, "geospatial_boundary_present": True})
	agent_result = evaluate_capability_rules({"esgc_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_esgc_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "inventory_requires_owner", "factor_requires_approved_source", "emission_requires_boundary", "emission_anomaly_requires_review"}
	assert report_result["matched_rules"] == ["report_requires_approval"]
	assert agent_result["decision"] == "deny"
	assert agent_result["matched_rules"] == ["esgc_agent_runtime_supported"]
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_esgc_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "esgc"
	assert "comp" in registration["dependencies"]
	assert registration["ui_components"]["reports"] == "/esgc/reports"
	assert registration["ui_components"]["agents"] == "/esgc/agents"
	assert "esgc:report" in registration["permissions"]
	assert registration["streaming"]["processor"] == "bytewax"


def test_emissions_factor_report_and_target_lifecycle():
	service = EsgcService()
	inventory = service.create_inventory(
		inventory_id="inventory-2026",
		tenant_id="tenant-esgc",
		organization="Datacraft Kenya",
		owner="sustainability-lead",
		reporting_year=2026,
		boundary_ref="boundary:operations",
		geospatial_boundary="geos:ke-operations",
		compliance_framework="GHG Protocol",
	)
	factor = service.register_factor(
		factor_id="factor-grid-ke",
		tenant_id="tenant-esgc",
		name="Kenya grid electricity",
		scope="scope_2",
		unit="kwh",
		co2e_per_unit=0.00025,
		source="national-grid-factor",
		source_evidence="audl:evidence-grid-2026",
		version="2026.1",
		approved_source=True,
	)
	activity = service.record_activity(
		activity_id="activity-jan",
		tenant_id="tenant-esgc",
		inventory_id="inventory-2026",
		factor_id="factor-grid-ke",
		activity_type="electricity",
		quantity=10000,
		unit="kwh",
		evidence_ref="iotd:meter-jan",
	)
	report = service.publish_report(
		report_id="report-q1",
		tenant_id="tenant-esgc",
		inventory_id="inventory-2026",
		report_type="quarterly_carbon",
		period="2026-Q1",
		compliance_mapping="GHG Protocol Scope 2",
		audit_evidence_ref="audl:report-q1",
		approved_by="esg-controller",
		approval_recorded=True,
	)
	target = service.create_target(
		target_id="target-2030",
		tenant_id="tenant-esgc",
		inventory_id="inventory-2026",
		name="Reduce operational emissions",
		baseline_year=2024,
		target_year=2030,
		baseline_co2e_tonnes=10.0,
		target_reduction_percent=50.0,
	)
	agent = service.register_esgc_agent(
		tenant_id="tenant-esgc",
		name="Report reviewer",
		runtime="codex",
		role="report_reviewer",
		scope="review report evidence and compliance mapping",
	)
	model = dashboard_model(service, "tenant-esgc")

	assert inventory["owner"] == "sustainability-lead"
	assert factor["approved_source"] is True
	assert activity["co2e_tonnes"] == 2.5
	assert activity["status"] == "recorded"
	assert report["total_co2e_tonnes"] == 2.5
	assert target["progress_percent"] == 100.0
	assert target["status"] == "achieved"
	assert agent["runtime"] == "codex"
	assert agent["role"] == "report_reviewer"
	assert model["summary"]["total_co2e_tonnes"] == 2.5
	assert model["summary"]["esgc_agent_count"] == 1
	assert service.validate_batch_esgc_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_esgc_mutation("other-stream")["decision"] == "deny"
	assert len(model["audit_events"]) == 6


def test_inventory_and_factor_guardrails_block_missing_governance():
	service = EsgcService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_inventory(
			inventory_id="inventory-no-tenant",
			tenant_id="",
			organization="Missing Tenant",
			owner="sustainability",
			reporting_year=2026,
			boundary_ref="boundary",
			geospatial_boundary="geos",
			compliance_framework="GHG Protocol",
		)

	with pytest.raises(PermissionError, match="organization_owner_required"):
		service.create_inventory(
			inventory_id="inventory-no-owner",
			tenant_id="tenant-esgc",
			organization="Missing Owner",
			owner="",
			reporting_year=2026,
			boundary_ref="boundary",
			geospatial_boundary="geos",
			compliance_framework="GHG Protocol",
		)

	with pytest.raises(PermissionError, match="boundary_required"):
		service.create_inventory(
			inventory_id="inventory-no-boundary",
			tenant_id="tenant-esgc",
			organization="No Boundary",
			owner="sustainability",
			reporting_year=2026,
			boundary_ref="",
			geospatial_boundary="",
			compliance_framework="GHG Protocol",
		)

	with pytest.raises(PermissionError, match="factor_source_required"):
		service.register_factor(
			factor_id="factor-unapproved",
			tenant_id="tenant-esgc",
			name="Unapproved factor",
			scope="scope_1",
			unit="litre",
			co2e_per_unit=0.0025,
			source="spreadsheet",
			source_evidence="",
			version="draft",
			approved_source=False,
		)

	with pytest.raises(PermissionError, match="esgc_agent_runtime_not_supported"):
		service.register_esgc_agent(
			tenant_id="tenant-esgc",
			name="Unsupported reviewer",
			runtime="unsupported",
			role="report_reviewer",
			scope="review reports",
		)


def test_activity_report_and_anomaly_guardrails():
	service = EsgcService()
	service.create_inventory("inventory-2026", "tenant-esgc", "Datacraft Kenya", "sustainability", 2026, "boundary", "geos", "GHG Protocol")
	service.register_factor("factor-grid", "tenant-esgc", "Grid", "scope_2", "kwh", 0.0003, "grid", "audl:grid", "2026.1", True)

	review_required = service.record_activity(
		activity_id="activity-spike",
		tenant_id="tenant-esgc",
		inventory_id="inventory-2026",
		factor_id="factor-grid",
		activity_type="electricity",
		quantity=50000,
		unit="kwh",
		evidence_ref="meter:spike",
		expected_max_quantity=10000,
	)

	assert review_required["status"] == "review_required"
	assert service.dashboard_summary("tenant-esgc")["review_required_activity_count"] == 1

	with pytest.raises(PermissionError, match="activity_unit_factor_mismatch"):
		service.record_activity(
			activity_id="activity-unit-mismatch",
			tenant_id="tenant-esgc",
			inventory_id="inventory-2026",
			factor_id="factor-grid",
			activity_type="fuel",
			quantity=10,
			unit="litre",
			evidence_ref="meter:fuel",
		)

	with pytest.raises(PermissionError, match="report_approval_required"):
		service.publish_report(
			report_id="report-unapproved",
			tenant_id="tenant-esgc",
			inventory_id="inventory-2026",
			report_type="annual",
			period="2026",
			compliance_mapping="GHG Protocol",
			audit_evidence_ref="audl:report",
			approved_by="",
			approval_recorded=False,
		)

	service.register_esgc_agent("tenant-esgc", "Activity reviewer", "codex", "activity_reviewer", "review anomalies")
	assert esgc_agent_model(service, "tenant-esgc")["esgc_agents"][0]["role"] == "activity_reviewer"


def test_lifecycle_ids_are_tenant_scoped():
	service = EsgcService()

	for tenant_id, organization, quantity in (
		("tenant-a", "Tenant A", 100.0),
		("tenant-b", "Tenant B", 250.0),
	):
		service.create_inventory("shared-inventory", tenant_id, organization, "lead", 2026, "boundary", "geo", "GHG Protocol")
		service.register_factor("shared-factor", tenant_id, "Electricity", "scope_2", "kwh", 0.001, "source", "evidence", "v1", True)
		service.record_activity("shared-activity", tenant_id, "shared-inventory", "shared-factor", "electricity", quantity, "kwh", "invoice")
		service.register_esgc_agent(tenant_id, "Reviewer", "codex", "activity_reviewer", "review anomalies", agent_id="shared-agent")

	assert service.list_inventories("tenant-a")[0]["organization"] == "Tenant A"
	assert service.list_inventories("tenant-b")[0]["organization"] == "Tenant B"
	assert service.dashboard_summary("tenant-a")["total_co2e_tonnes"] == 0.1
	assert service.dashboard_summary("tenant-b")["total_co2e_tonnes"] == 0.25
	assert service.list_esgc_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_esgc_agents("tenant-b")[0]["id"] == "shared-agent"


def test_generated_evidence_and_docs_are_current():
	app = _load_module("esgc_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["esgc"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["esgc"]["screens"]["agents"]["route"] == "/esgc/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
