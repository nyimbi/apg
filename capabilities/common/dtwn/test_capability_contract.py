"""Regression coverage for the DTWN executable capability contract."""

import pytest

from capabilities.common.dtwn import register_capability
from capabilities.common.dtwn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.dtwn.service import DtwnService
from capabilities.common.dtwn.views import analytics_model, audit_trail_model, dashboard_model, settings_model, twin_agents_model


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-dtwn", {"simulation": {"prediction_confidence_threshold": 0.9}})

	assert contract["capability"] == "dtwn"
	assert contract["configuration"]["tenant_id"] == "tenant-dtwn"
	assert contract["configuration"]["simulation"]["prediction_confidence_threshold"] == 0.9
	assert contract["configuration_schema"]["required"] == ["tenant_id", "twins", "telemetry", "simulation", "twin_agents", "governance", "observability", "adapters", "ui", "theme"]
	assert contract["configuration"]["twin_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["topic"] == "apg.dtwn.lifecycle"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "twins", "models", "telemetry", "simulations", "predictions", "topology", "agents", "audit", "analytics", "settings"}
	assert contract["theme"]["name"] == "dtwn_digital_twin_ops"


def test_rule_engine_enforces_dtwn_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_twin", "twin_owner_assigned": False, "telemetry_source_authenticated": False, "prediction_risk_score": 0.95, "prediction_review_recorded": False})
	simulation_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "run_simulation", "model_present": False, "telemetry_source_authenticated": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "twin_requires_owner", "telemetry_requires_authenticated_source", "high_risk_prediction_requires_review"}
	assert simulation_result["matched_rules"] == ["simulation_requires_model"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "dtwn"
	assert "iotd" in registration["dependencies"]
	assert registration["ui_components"]["topology"] == "/dtwn/topology"
	assert registration["ui_components"]["agents"] == "/dtwn/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "dtwn:simulate" in registration["permissions"]


def test_service_runs_digital_twin_lifecycle_with_simulation_and_review():
	service = DtwnService()

	pump = service.create_twin(
		twin_id="twin-pump-1",
		tenant_id="tenant-dtwn",
		asset_id="asset-pump-1",
		name="Pump 1",
		owner="operations",
		twin_type="pump",
		location={"site": "plant-a", "lat": 1.2, "lon": 36.8},
		initial_state={"temperature": 42, "vibration": 18},
	)
	tank = service.create_twin(
		twin_id="twin-tank-1",
		tenant_id="tenant-dtwn",
		asset_id="asset-tank-1",
		name="Tank 1",
		owner="operations",
		twin_type="tank",
	)
	model = service.register_simulation_model(
		model_id="model-pump-risk",
		tenant_id="tenant-dtwn",
		name="Pump risk model",
		version="1.0.0",
		owner="model-risk",
		model_type="physics_ml_hybrid",
		calibration_evidence="calibration-report-001",
		approved_by="chief-engineer",
		confidence=0.91,
	)
	agent = service.register_twin_agent(
		"tenant-dtwn",
		"codex-twin-reviewer",
		"Codex Twin Reviewer",
		"codex",
		"prediction_reviewer",
		"Review high-risk twin predictions and simulation evidence.",
		True,
		"policy:dtwn:agents:v1",
	)
	telemetry = service.ingest_telemetry(
		sample_id="tel-1",
		tenant_id="tenant-dtwn",
		twin_id=pump["id"],
		source_id="iot-gateway-1",
		source_type="iot",
		authenticated=True,
		measurements={"temperature": 88, "vibration": 64},
		geospatial_context={"site": "plant-a"},
	)
	link = service.link_topology(
		link_id="link-1",
		tenant_id="tenant-dtwn",
		source_twin_id=pump["id"],
		target_twin_id=tank["id"],
		relationship="feeds",
	)
	run = service.run_simulation(
		run_id="sim-1",
		tenant_id="tenant-dtwn",
		twin_id=pump["id"],
		model_id=model["id"],
		scenario="high load",
		environment="production",
		approved_by="shift-lead",
	)
	prediction = service.record_prediction(
		prediction_id="pred-1",
		tenant_id="tenant-dtwn",
		twin_id=pump["id"],
		model_id=model["id"],
		risk_score=0.91,
		confidence=0.86,
		horizon="48h",
		recommendation="inspect bearing assembly",
	)
	dashboard = dashboard_model(service, "tenant-dtwn")
	agents = twin_agents_model(service, "tenant-dtwn")
	analytics = analytics_model(service, "tenant-dtwn")
	audit = audit_trail_model(service, "tenant-dtwn")
	settings = settings_model("tenant-dtwn")

	assert telemetry["state_version"] != pump["state_version"]
	assert link["relationship"] == "feeds"
	assert agent["runtime"] == "codex"
	assert agent["role"] == "prediction_reviewer"
	assert run["status"] == "completed"
	assert run["outputs"]["risk_score"] > 0
	assert prediction["review_required"] is True
	assert service.dashboard_summary("tenant-dtwn")["review_required_prediction_count"] == 1
	assert service.dashboard_summary("tenant-dtwn")["twin_agent_count"] == 1
	assert dashboard["twin_agents"][0]["id"] == "codex-twin-reviewer"
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert analytics["signals"]["telemetry_per_twin"] == 0.5
	assert audit["guardrails"]
	assert settings["streaming"]["processor"] == "bytewax"

	reviewed = service.review_prediction("pred-1", "tenant-dtwn", "reliability-engineer")
	assert reviewed["status"] == "reviewed"
	assert service.list_audit_events("tenant-dtwn")


def test_service_enforces_digital_twin_guardrails():
	service = DtwnService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_twin("twin-no-tenant", "", "asset-1", "No tenant", "owner", "asset")
	with pytest.raises(PermissionError, match="twin_owner_required"):
		service.create_twin("twin-no-owner", "tenant-dtwn", "asset-1", "No owner", "", "asset")
	with pytest.raises(PermissionError, match="asset_identity_required"):
		service.create_twin("twin-no-asset", "tenant-dtwn", "", "No asset", "owner", "asset")
	with pytest.raises(PermissionError, match="calibration_evidence_required"):
		service.register_simulation_model("model-no-cal", "tenant-dtwn", "No calibration", "1", "owner", "physics", "", "approver")
	with pytest.raises(PermissionError, match="prediction_confidence_threshold"):
		service.register_simulation_model("model-low-confidence", "tenant-dtwn", "Low confidence", "1", "owner", "physics", "evidence", "approver", confidence=0.4)

	service.create_twin("twin-1", "tenant-dtwn", "asset-1", "Twin", "owner", "asset")
	service.register_simulation_model("model-draft", "tenant-dtwn", "Draft", "1", "owner", "physics", "evidence", None)
	with pytest.raises(PermissionError, match="telemetry_source_auth_required"):
		service.ingest_telemetry("tel-bad", "tenant-dtwn", "twin-1", "source-1", "iot", False, {"temperature": 12})
	with pytest.raises(PermissionError, match="telemetry_measurements_required"):
		service.ingest_telemetry("tel-empty", "tenant-dtwn", "twin-1", "source-1", "iot", True, {})
	with pytest.raises(PermissionError, match="simulation_model_required"):
		service.run_simulation("sim-draft", "tenant-dtwn", "twin-1", "model-draft", "baseline")


def test_production_simulation_requires_approval_and_high_risk_prediction_can_be_reviewed():
	service = DtwnService()
	service.create_twin("twin-1", "tenant-dtwn", "asset-1", "Twin", "owner", "asset", initial_state={"load": 90})
	service.register_simulation_model("model-1", "tenant-dtwn", "Model", "1", "owner", "physics", "evidence", "approver")

	with pytest.raises(PermissionError, match="simulation_approval_required"):
		service.run_simulation("sim-prod", "tenant-dtwn", "twin-1", "model-1", "baseline", environment="production")

	run = service.run_simulation("sim-prod-approved", "tenant-dtwn", "twin-1", "model-1", "baseline", environment="production", approved_by="approver")
	assert run["status"] == "completed"

	reviewed_prediction = service.record_prediction(
		"pred-reviewed",
		"tenant-dtwn",
		"twin-1",
		"model-1",
		risk_score=0.91,
		confidence=0.85,
		horizon="24h",
		recommendation="inspect",
		reviewed_by="reviewer",
	)
	assert reviewed_prediction["review_required"] is False
	assert reviewed_prediction["reviewed_by"] == "reviewer"


def test_twin_agents_state_changes_bytewax_and_tenant_scope():
	service = DtwnService()
	for tenant_id in ("tenant-a", "tenant-b"):
		service.create_twin("shared-twin", tenant_id, "shared-asset", "Shared Twin", "owner", "asset")
		service.register_simulation_model("shared-model", tenant_id, "Shared Model", "1", "owner", "physics", "evidence", "approver")
		service.register_twin_agent(
			tenant_id,
			"shared-agent",
			"Shared Agent",
			"codex",
			"simulation_operator",
			f"Operate simulations for {tenant_id}.",
			True,
		)

	assert len(service.list_twin_agents("tenant-a")) == 1
	assert len(service.list_twin_agents("tenant-b")) == 1
	assert service.list_twins("tenant-a")[0]["tenant_id"] == "tenant-a"

	inactive = service.change_twin_status("tenant-a", "shared-twin", "inactive", "Pause twin for maintenance.", "owner")
	assert inactive["status"] == "inactive"
	assert service.validate_batch_twin_mutation("tenant-a", "bytewax", "owner")["processor"] == "bytewax"

	with pytest.raises(PermissionError, match="dtwn_state_change_reason_required"):
		service.change_twin_status("tenant-a", "shared-twin", "inactive", "", "owner")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch_twin_mutation("tenant-a", "custom-stream", "owner")
	with pytest.raises(PermissionError, match="twin_agent_runtime_not_supported"):
		service.register_twin_agent("tenant-a", "bad-runtime", "Bad Runtime", "custom", "simulation_operator", "Operate.", True)
	with pytest.raises(PermissionError, match="twin_agent_role_not_supported"):
		service.register_twin_agent("tenant-a", "bad-role", "Bad Role", "codex", "owner", "Operate.", True)
	with pytest.raises(PermissionError, match="twin_agent_disclosure_required"):
		service.register_twin_agent("tenant-a", "undisclosed", "Undisclosed", "codex", "simulation_operator", "Operate.", False)
