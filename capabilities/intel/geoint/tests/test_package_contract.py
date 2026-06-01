"""Executable Geospatial Intelligence capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_intel_geoint", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_geoint"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "geoint_agent_workflow" in contract["provides"]
	assert "/intel-geoint/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_geoint", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "geoint_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "geoint_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "geoint_agent_action", "targeting_or_harmful_scope": True})["decision"] == "deny"


def test_service_executes_geoint_lifecycle():
	service_module = _load_module("service_intel_geoint", PACKAGE_DIR / "service.py")
	service = service_module.GeospatialIntelligenceService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "secret", "approver-1", "2026-12-31", "authority-evidence")
	area = service.record_area("area-1", "tenant-test", "Port corridor", "geojson:area", "secret", "owner-1", authority["id"], "area-evidence")
	source = service.register_source("source-1", "tenant-test", "satellite_imagery", "optical", "high", "owner-1", authority["id"], "source-evidence")
	plan = service.record_collection_plan("plan-1", "tenant-test", authority["id"], area["id"], source["id"], "catalog_query", 30, "approval-ref", "plan-evidence")
	observation = service.record_observation("obs-1", "tenant-test", plan["id"], "observation-ref", "2026-06-01T00:00:00Z", 0.91, "observation-evidence")
	feature = service.record_feature("feature-1", "tenant-test", observation["id"], "facility", "geojson:feature", 0.84, "analyst-1", "feature-evidence")
	change = service.record_change("change-1", "tenant-test", feature["id"], "construction", "high", 0.82, "analyst-1", "change-evidence")
	assessment = service.record_assessment("assessment-1", "tenant-test", change["id"], "change_report", "secret", "analyst-1", "assessment-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", assessment["id"], "watch-center", "REL TO PARTNER", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_geoint_agent("agent-1", "tenant-test", "GEOINT Agent", "codex", "feature_analyst", "feature analysis")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert area["geometry_reference"] == "geojson:area"
	assert source["sensor_type"] == "optical"
	assert plan["retention_days"] == 30
	assert observation["geospatial_accuracy_score"] == 0.91
	assert feature["feature_type"] == "facility"
	assert change["severity"] == "high"
	assert assessment["assessment_type"] == "change_report"
	assert dissemination["release_marking"] == "REL TO PARTNER"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_geoint", PACKAGE_DIR / "service.py")
	service = service_module.GeospatialIntelligenceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "secret", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "confidential", "approver-b", "2026-12-31", "evidence-b")
	service.record_area("shared-area", "tenant-a", "Area A", "geojson:a", "secret", "owner-a", tenant_a["id"], "evidence-a")
	service.record_area("shared-area", "tenant-b", "Area B", "geojson:b", "confidential", "owner-b", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["area_count"] == 1
	assert dashboard_b["area_count"] == 1
	assert service._tenant_area_or_none("shared-area", "tenant-a").name == "Area A"
	assert service._tenant_area_or_none("shared-area", "tenant-b").name == "Area B"


def test_service_guardrails_reject_invalid_geoint_actions():
	service_module = _load_module("guardrail_service_intel_geoint", PACKAGE_DIR / "service.py")
	service = service_module.GeospatialIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "secret", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_area("area", "tenant-test", "Area", "geojson", "secret", "owner", "missing-auth", "evidence")
	with pytest.raises(PermissionError, match="geometry_reference_required"):
		service.record_area("area", "tenant-test", "Area", "", "secret", "owner", authority["id"], "evidence")
	area = service.record_area("area-ok", "tenant-test", "Area", "geojson", "secret", "owner", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="sensor_type_not_supported"):
		service.register_source("source", "tenant-test", "satellite_imagery", "unknown", "high", "owner", authority["id"], "evidence")
	source = service.register_source("source-ok", "tenant-test", "satellite_imagery", "optical", "high", "owner", authority["id"], "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="area_authority_mismatch"):
		service.record_collection_plan("plan", "tenant-test", other_authority["id"], area["id"], source["id"], "catalog_query", 30, "approval", "evidence")
	with pytest.raises(PermissionError, match="retention_days_invalid"):
		service.record_collection_plan("plan", "tenant-test", authority["id"], area["id"], source["id"], "catalog_query", 0, "approval", "evidence")
	plan = service.record_collection_plan("plan-ok", "tenant-test", authority["id"], area["id"], source["id"], "catalog_query", 30, "approval", "evidence")
	with pytest.raises(PermissionError, match="geospatial_accuracy_score_invalid"):
		service.record_observation("obs", "tenant-test", plan["id"], "obs-ref", "2026-06-01T00:00:00Z", 1.2, "evidence")
	observation = service.record_observation("obs-ok", "tenant-test", plan["id"], "obs-ref", "2026-06-01T00:00:00Z", 0.9, "evidence")
	with pytest.raises(PermissionError, match="feature_type_not_supported"):
		service.record_feature("feature", "tenant-test", observation["id"], "unknown", "geojson", 0.9, "analyst", "evidence")
	feature = service.record_feature("feature-ok", "tenant-test", observation["id"], "facility", "geojson", 0.9, "analyst", "evidence")
	with pytest.raises(PermissionError, match="severity_not_supported"):
		service.record_change("change", "tenant-test", feature["id"], "construction", "unknown", 0.9, "analyst", "evidence")
	change = service.record_change("change-ok", "tenant-test", feature["id"], "construction", "high", 0.9, "analyst", "evidence")
	with pytest.raises(PermissionError, match="classification_not_supported"):
		service.record_assessment("assessment", "tenant-test", change["id"], "change_report", "unknown", "analyst", "evidence")
	assessment = service.record_assessment("assessment-ok", "tenant-test", change["id"], "change_report", "secret", "analyst", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", assessment["id"], "audience", "REL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", assessment["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="geoint_agent_runtime_not_supported"):
		service.register_geoint_agent("agent", "tenant-test", "Bad Agent", "unsupported", "feature_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="targeting_or_harmful_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, targeting_or_harmful_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_geoint", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_geoint", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_geoint", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "confidential", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	area = api.record_area({"tenant_id": "tenant-api", "area_id": "api-area", "name": "Area", "geometry_reference": "geojson", "classification": "confidential", "owner_id": "owner", "authority_id": authority["id"], "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "open_geodata", "sensor_type": "manual_survey", "resolution_class": "metadata_only", "owner_id": "owner", "authority_id": authority["id"], "evidence_reference": "evidence"})
	api.record_collection_plan({"tenant_id": "tenant-api", "plan_id": "api-plan", "authority_id": authority["id"], "area_id": area["id"], "source_id": source["id"], "collection_mode": "manual_upload", "retention_days": 7, "approval_reference": "approval", "evidence_reference": "evidence"})
	agent = api.register_geoint_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "GEOINT Agent", "runtime": "claude_code", "role": "area_planner"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.geoint_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "area_planner"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["collection_plans"][0]["id"] == "api-plan"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_geoint"]["screens"]["agents"]["route"] == "/intel-geoint/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_geoint", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_geoint"]["streaming"]["processor"] == "bytewax"
