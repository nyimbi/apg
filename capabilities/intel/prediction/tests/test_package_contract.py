"""Executable Predictive Intelligence capability package tests."""

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
	module = _load_module("contract_intel_prediction", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_prediction"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "prediction_agent_workflow" in contract["provides"]
	assert "/intel-prediction/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_prediction", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "prediction_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "prediction_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "prediction_agent_action", "unsupported_automated_decision_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "prediction_agent_action", "hallucinated_forecast_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "prediction_agent_action", "privacy_bypass_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "prediction_agent_action", "unapproved_model_deployment_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "prediction_agent_action", "autonomous_warning_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "prediction_agent_action", "autonomous_recommendation_scope": True})["decision"] == "deny"


def test_service_executes_prediction_lifecycle():
	service_module = _load_module("service_intel_prediction", PACKAGE_DIR / "service.py")
	service = service_module.PredictiveIntelligenceService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	workspace = service.record_workspace("workspace-1", "tenant-test", "threat_prediction", "Threat Prediction", "confidential", authority["id"], "workspace-evidence")
	scenario = service.record_scenario("scenario-1", "tenant-test", workspace["id"], "threat_scenario", "scenario-ref", "near_term", "owner-1", "scenario-evidence")
	indicator = service.record_indicator("indicator-1", "tenant-test", scenario["id"], "leading_indicator", "indicator-ref", 0.82, "indicator-evidence")
	model = service.record_model("model-1", "tenant-test", scenario["id"], "machine_learning", "Forecast likely threat escalation", "validation-ref", "high", "model-evidence")
	forecast = service.record_forecast("forecast-1", "tenant-test", model["id"], "probability", "forecast-ref", 0.76, "analyst-1", "forecast-evidence")
	projection = service.record_projection("projection-1", "tenant-test", forecast["id"], "threat_projection", "high", 0.71, "analyst-1", "projection-evidence")
	warning = service.record_warning("warning-1", "tenant-test", projection["id"], "early_warning", "high", "trigger-ref", "approval-ref", "warning-evidence")
	recommendation = service.record_recommendation("recommendation-1", "tenant-test", projection["id"], "mitigate", "action-ref", "approval-ref", "recommendation-evidence")
	review = service.record_review("review-1", "tenant-test", recommendation["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_prediction_agent("agent-1", "tenant-test", "Prediction Agent", "codex", "forecast_analyst", "forecast support")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert workspace["workspace_type"] == "threat_prediction"
	assert scenario["horizon"] == "near_term"
	assert indicator["indicator_type"] == "leading_indicator"
	assert model["model_type"] == "machine_learning"
	assert forecast["forecast_type"] == "probability"
	assert projection["projection_type"] == "threat_projection"
	assert warning["warning_type"] == "early_warning"
	assert recommendation["recommendation_type"] == "mitigate"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_prediction", PACKAGE_DIR / "service.py")
	service = service_module.PredictiveIntelligenceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_workspace("shared-workspace", "tenant-a", "threat_prediction", "Workspace A", "confidential", tenant_a["id"], "evidence-a")
	service.record_workspace("shared-workspace", "tenant-b", "strategic_forecast", "Workspace B", "unclassified", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["workspace_count"] == 1
	assert dashboard_b["workspace_count"] == 1
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-a").name == "Workspace A"
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-b").name == "Workspace B"


def test_service_guardrails_reject_invalid_prediction_actions():
	service_module = _load_module("guardrail_service_intel_prediction", PACKAGE_DIR / "service.py")
	service = service_module.PredictiveIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_workspace("workspace", "tenant-test", "threat_prediction", "workspace", "confidential", "missing-auth", "evidence")
	workspace = service.record_workspace("workspace-ok", "tenant-test", "threat_prediction", "workspace", "confidential", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="scenario_owner_required"):
		service.record_scenario("scenario", "tenant-test", workspace["id"], "threat_scenario", "scenario", "near_term", "", "evidence")
	with pytest.raises(PermissionError, match="prediction_horizon_not_supported"):
		service.record_scenario("scenario", "tenant-test", workspace["id"], "threat_scenario", "scenario", "never", "owner", "evidence")
	scenario = service.record_scenario("scenario-ok", "tenant-test", workspace["id"], "threat_scenario", "scenario", "near_term", "owner", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_indicator("indicator", "tenant-test", scenario["id"], "leading_indicator", "indicator", 1.8, "evidence")
	with pytest.raises(PermissionError, match="model_validation_required"):
		service.record_model("model", "tenant-test", scenario["id"], "machine_learning", "objective", "", "high", "evidence")
	model = service.record_model("model-ok", "tenant-test", scenario["id"], "machine_learning", "objective", "validation", "high", "evidence")
	with pytest.raises(PermissionError, match="forecast_type_not_supported"):
		service.record_forecast("forecast", "tenant-test", model["id"], "unknown", "forecast", 0.8, "analyst", "evidence")
	forecast = service.record_forecast("forecast-ok", "tenant-test", model["id"], "probability", "forecast", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="probability_score_invalid"):
		service.record_projection("projection", "tenant-test", forecast["id"], "threat_projection", "high", 1.8, "analyst", "evidence")
	projection = service.record_projection("projection-ok", "tenant-test", forecast["id"], "threat_projection", "high", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="warning_approval_required"):
		service.record_warning("warning", "tenant-test", projection["id"], "early_warning", "high", "trigger", "", "evidence")
	with pytest.raises(PermissionError, match="recommendation_approval_required"):
		service.record_recommendation("recommendation", "tenant-test", projection["id"], "mitigate", "action", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", projection["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="prediction_agent_runtime_not_supported"):
		service.register_prediction_agent("agent", "tenant-test", "Bad Agent", "unsupported", "forecast_analyst", "scope")
	with pytest.raises(PermissionError, match="prediction_agent_scope_required"):
		service.register_prediction_agent("agent", "tenant-test", "Prediction Agent", "codex", "forecast_analyst", "")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="unsupported_automated_decision_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unsupported_automated_decision_scope=True)
	with pytest.raises(PermissionError, match="hallucinated_forecast_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, hallucinated_forecast_scope=True)
	with pytest.raises(PermissionError, match="privacy_bypass_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, privacy_bypass_scope=True)
	with pytest.raises(PermissionError, match="unapproved_model_deployment_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_model_deployment_scope=True)
	with pytest.raises(PermissionError, match="autonomous_warning_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_warning_scope=True)
	with pytest.raises(PermissionError, match="autonomous_recommendation_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_recommendation_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_prediction", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_prediction", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_prediction", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	workspace = api.record_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "workspace_type": "strategic_forecast", "name": "Strategic Forecast", "classification": "unclassified", "authority_id": authority["id"], "evidence_reference": "evidence"})
	scenario = api.record_scenario({"tenant_id": "tenant-api", "scenario_id": "api-scenario", "workspace_id": workspace["id"], "scenario_type": "strategic_scenario", "scenario_reference": "scenario-ref", "horizon": "mid_term", "owner_id": "owner", "evidence_reference": "evidence"})
	api.record_model({"tenant_id": "tenant-api", "model_id": "api-model", "scenario_id": scenario["id"], "model_type": "statistical", "objective": "forecast demand", "validation_reference": "validation", "risk_level": "medium", "evidence_reference": "evidence"})
	agent = api.register_prediction_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Prediction Agent", "runtime": "claude_code", "role": "forecast_analyst"})
	batch = api.validate_batch({"tenant_id": "tenant-api", "item_count": 2})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.prediction_console_model(api.service(), "tenant-api")
	workbench = views.agent_workbench_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "forecast_analyst"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["scenarios"][0]["id"] == scenario["id"]
	assert workbench["agents"][0]["id"] == agent["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_prediction"]["screens"]["agents"]["route"] == "/intel-prediction/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_prediction", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_prediction"]["streaming"]["processor"] == "bytewax"
