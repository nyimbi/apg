"""Executable Digital Surveillance capability package tests."""

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
	module = _load_module("contract_intel_surveillance", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_surveillance"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "surveillance_agent_workflow" in contract["provides"]
	assert "/intel-surveillance/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_surveillance", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "surveillance_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "surveillance_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "surveillance_agent_action", "covert_tracking_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "surveillance_agent_action", "spyware_scope": True})["decision"] == "deny"


def test_service_executes_surveillance_lifecycle():
	service_module = _load_module("service_intel_surveillance", PACKAGE_DIR / "service.py")
	service = service_module.DigitalSurveillanceService()

	authority = service.record_authority("auth-1", "tenant-test", "security_monitoring_authority", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	program = service.record_program("program-1", "tenant-test", "facility_monitoring", "Facility watch", "high", authority["id"], "program-evidence")
	asset = service.record_asset("asset-1", "tenant-test", "facility", "facility-ref", "owner-1", authority["id"], "privacy-review", "asset-evidence")
	sensor = service.register_sensor("sensor-1", "tenant-test", "camera", asset["id"], "camera-ref", "custodian-1", "calibration-ref", "sensor-evidence")
	observation = service.record_observation("observation-1", "tenant-test", program["id"], sensor["id"], "motion", "obs-ref", "sha256:abc", "2026-06-01T00:00:00Z", 0.82, "observation-evidence")
	alert = service.record_alert("alert-1", "tenant-test", observation["id"], "intrusion", "high", 0.86, "analyst-1", "alert-evidence")
	risk = service.record_risk("risk-1", "tenant-test", alert["id"], "physical_security", "high", 0.84, "analyst-1", "risk-evidence")
	referral = service.record_referral("referral-1", "tenant-test", risk["id"], "incident_response", "security-team", "approval-ref", "referral-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", risk["id"], "security-team", "CONFIDENTIAL", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_surveillance_agent("agent-1", "tenant-test", "Surveillance Agent", "codex", "alert_analyst", "alert analysis")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "security_monitoring_authority"
	assert program["program_type"] == "facility_monitoring"
	assert asset["asset_type"] == "facility"
	assert sensor["sensor_type"] == "camera"
	assert observation["observation_type"] == "motion"
	assert alert["alert_type"] == "intrusion"
	assert risk["assessment_type"] == "physical_security"
	assert referral["referral_type"] == "incident_response"
	assert dissemination["release_marking"] == "CONFIDENTIAL"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_surveillance", PACKAGE_DIR / "service.py")
	service = service_module.DigitalSurveillanceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "security_monitoring_authority", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_program("shared-program", "tenant-a", "facility_monitoring", "Facility A", "medium", tenant_a["id"], "evidence-a")
	service.record_program("shared-program", "tenant-b", "endpoint_monitoring", "Endpoint B", "low", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["program_count"] == 1
	assert dashboard_b["program_count"] == 1
	assert service._tenant_program_or_none("shared-program", "tenant-a").name == "Facility A"
	assert service._tenant_program_or_none("shared-program", "tenant-b").name == "Endpoint B"


def test_service_guardrails_reject_invalid_surveillance_actions():
	service_module = _load_module("guardrail_service_intel_surveillance", PACKAGE_DIR / "service.py")
	service = service_module.DigitalSurveillanceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "security_monitoring_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "security_monitoring_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_program("program", "tenant-test", "facility_monitoring", "facility", "medium", "missing-auth", "evidence")
	program = service.record_program("program-ok", "tenant-test", "facility_monitoring", "facility", "medium", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="privacy_review_required"):
		service.record_asset("asset", "tenant-test", "facility", "facility-ref", "owner", authority["id"], "", "evidence")
	asset = service.record_asset("asset-ok", "tenant-test", "facility", "facility-ref", "owner", authority["id"], "privacy", "evidence")
	with pytest.raises(PermissionError, match="sensor_calibration_required"):
		service.register_sensor("sensor", "tenant-test", "camera", asset["id"], "sensor-ref", "custodian", "", "evidence")
	sensor = service.register_sensor("sensor-ok", "tenant-test", "camera", asset["id"], "sensor-ref", "custodian", "calibration", "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "confidential", "approver", "2026-12-31", "evidence")
	other_program = service.record_program("program-other", "tenant-test", "endpoint_monitoring", "endpoint", "low", other_authority["id"], "evidence")
	with pytest.raises(PermissionError, match="authority_mismatch"):
		service.record_observation("observation", "tenant-test", other_program["id"], sensor["id"], "motion", "ref", "hash", "2026-06-01", 0.8, "evidence")
	observation = service.record_observation("observation-ok", "tenant-test", program["id"], sensor["id"], "motion", "ref", "hash", "2026-06-01", 0.8, "evidence")
	with pytest.raises(PermissionError, match="alert_type_not_supported"):
		service.record_alert("alert", "tenant-test", observation["id"], "unknown", "high", 0.8, "analyst", "evidence")
	alert = service.record_alert("alert-ok", "tenant-test", observation["id"], "intrusion", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="assessment_type_not_supported"):
		service.record_risk("risk", "tenant-test", alert["id"], "unknown", "medium", 0.8, "analyst", "evidence")
	risk = service.record_risk("risk-ok", "tenant-test", alert["id"], "physical_security", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="referral_approval_required"):
		service.record_referral("referral", "tenant-test", risk["id"], "incident_response", "team", "", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", risk["id"], "team", "CONFIDENTIAL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", risk["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="surveillance_agent_runtime_not_supported"):
		service.register_surveillance_agent("agent", "tenant-test", "Bad Agent", "unsupported", "alert_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="covert_tracking_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, covert_tracking_scope=True)
	with pytest.raises(PermissionError, match="spyware_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, spyware_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_surveillance", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_surveillance", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_surveillance", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	program = api.record_program({"tenant_id": "tenant-api", "program_id": "api-program", "program_type": "facility_monitoring", "name": "Facility", "priority": "medium", "authority_id": authority["id"], "evidence_reference": "evidence"})
	asset = api.record_asset({"tenant_id": "tenant-api", "asset_id": "api-asset", "asset_type": "facility", "asset_reference": "asset-ref", "owner_id": "owner", "authority_id": authority["id"], "privacy_review_reference": "privacy", "evidence_reference": "evidence"})
	sensor = api.register_sensor({"tenant_id": "tenant-api", "sensor_id": "api-sensor", "sensor_type": "camera", "asset_id": asset["id"], "sensor_reference": "sensor-ref", "custodian_id": "custodian", "calibration_reference": "calibration", "evidence_reference": "evidence"})
	api.record_observation({"tenant_id": "tenant-api", "observation_id": "api-observation", "program_id": program["id"], "sensor_id": sensor["id"], "observation_type": "motion", "observation_reference": "obs-ref", "content_fingerprint": "hash", "observed_at": "2026-06-01", "confidence_score": 0.8, "evidence_reference": "evidence"})
	agent = api.register_surveillance_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Surveillance Agent", "runtime": "claude_code", "role": "alert_analyst"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.surveillance_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "alert_analyst"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["sensors"][0]["id"] == sensor["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_surveillance"]["screens"]["agents"]["route"] == "/intel-surveillance/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_surveillance", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_surveillance"]["streaming"]["processor"] == "bytewax"
