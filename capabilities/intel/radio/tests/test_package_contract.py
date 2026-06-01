"""Executable Radio Intelligence Listener capability package tests."""

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
	module = _load_module("contract_intel_radio", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_radio"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "radio_agent_workflow" in contract["provides"]
	assert "/intel-radio/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_radio", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "radio_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "radio_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "radio_agent_action", "transmit_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "radio_agent_action", "jamming_scope": True})["decision"] == "deny"


def test_service_executes_radio_lifecycle():
	service_module = _load_module("service_intel_radio", PACKAGE_DIR / "service.py")
	service = service_module.RadioIntelligenceListenerService()

	authority = service.record_authority("auth-1", "tenant-test", "spectrum_license", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	band = service.record_band_plan("band-1", "tenant-test", "public_safety", "Public safety band", 150.0, 170.0, authority["id"], "band-evidence")
	receiver = service.register_receiver("receiver-1", "tenant-test", "sdr", "site-ref", "custodian-1", authority["id"], "calibration-ref", "receiver-evidence")
	session = service.record_session("session-1", "tenant-test", band["id"], receiver["id"], "incident_watch", "2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z", "plan-ref", "session-evidence")
	observation = service.record_observation("observation-1", "tenant-test", session["id"], 160.25, "voice", "sha256:abc", "2026-06-01T00:05:00Z", 0.82, "observation-evidence")
	classification = service.record_classification("classification-1", "tenant-test", observation["id"], "emergency_signal", "high", 0.86, "analyst-1", "classification-evidence")
	event = service.record_event("event-1", "tenant-test", classification["id"], "public_safety_event", "high", 0.84, "analyst-1", "event-evidence")
	referral = service.record_referral("referral-1", "tenant-test", event["id"], "public_safety_notice", "dispatch", "approval-ref", "referral-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", event["id"], "public-safety-team", "CONFIDENTIAL", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_radio_agent("agent-1", "tenant-test", "Radio Agent", "codex", "signal_analyst", "signal analysis")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "spectrum_license"
	assert band["band_type"] == "public_safety"
	assert receiver["receiver_type"] == "sdr"
	assert session["session_type"] == "incident_watch"
	assert observation["frequency_mhz"] == 160.25
	assert classification["classification_type"] == "emergency_signal"
	assert event["event_type"] == "public_safety_event"
	assert referral["referral_type"] == "public_safety_notice"
	assert dissemination["release_marking"] == "CONFIDENTIAL"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_radio", PACKAGE_DIR / "service.py")
	service = service_module.RadioIntelligenceListenerService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "spectrum_license", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "public_safety_authority", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_band_plan("shared-band", "tenant-a", "aviation", "Aviation A", 118.0, 137.0, tenant_a["id"], "evidence-a")
	service.record_band_plan("shared-band", "tenant-b", "maritime", "Maritime B", 156.0, 162.0, tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["band_plan_count"] == 1
	assert dashboard_b["band_plan_count"] == 1
	assert service._tenant_band_or_none("shared-band", "tenant-a").name == "Aviation A"
	assert service._tenant_band_or_none("shared-band", "tenant-b").name == "Maritime B"


def test_service_guardrails_reject_invalid_radio_actions():
	service_module = _load_module("guardrail_service_intel_radio", PACKAGE_DIR / "service.py")
	service = service_module.RadioIntelligenceListenerService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "spectrum_license", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "spectrum_license", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="frequency_range_invalid"):
		service.record_band_plan("band", "tenant-test", "public_safety", "bad band", 170.0, 150.0, authority["id"], "evidence")
	band = service.record_band_plan("band-ok", "tenant-test", "public_safety", "band", 150.0, 170.0, authority["id"], "evidence")
	with pytest.raises(PermissionError, match="receiver_calibration_required"):
		service.register_receiver("receiver", "tenant-test", "sdr", "site", "custodian", authority["id"], "", "evidence")
	receiver = service.register_receiver("receiver-ok", "tenant-test", "sdr", "site", "custodian", authority["id"], "calibration", "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "public_safety_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	other_receiver = service.register_receiver("receiver-other", "tenant-test", "fixed_station", "site", "custodian", other_authority["id"], "calibration", "evidence")
	with pytest.raises(PermissionError, match="authority_mismatch"):
		service.record_session("session", "tenant-test", band["id"], other_receiver["id"], "incident_watch", "2026-06-01", "", "plan", "evidence")
	session = service.record_session("session-ok", "tenant-test", band["id"], receiver["id"], "incident_watch", "2026-06-01", "", "plan", "evidence")
	with pytest.raises(PermissionError, match="frequency_out_of_band"):
		service.record_observation("observation", "tenant-test", session["id"], 180.0, "voice", "hash", "2026-06-01", 0.8, "evidence")
	observation = service.record_observation("observation-ok", "tenant-test", session["id"], 160.0, "voice", "hash", "2026-06-01", 0.8, "evidence")
	with pytest.raises(PermissionError, match="classification_type_not_supported"):
		service.record_classification("classification", "tenant-test", observation["id"], "unknown", "high", 0.8, "analyst", "evidence")
	classification = service.record_classification("classification-ok", "tenant-test", observation["id"], "interference", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="event_type_not_supported"):
		service.record_event("event", "tenant-test", classification["id"], "unknown", "medium", 0.8, "analyst", "evidence")
	event = service.record_event("event-ok", "tenant-test", classification["id"], "interference_event", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="referral_approval_required"):
		service.record_referral("referral", "tenant-test", event["id"], "public_safety_notice", "team", "", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", event["id"], "team", "CONFIDENTIAL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", event["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="radio_agent_runtime_not_supported"):
		service.register_radio_agent("agent", "tenant-test", "Bad Agent", "unsupported", "signal_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="transmit_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, transmit_scope=True)
	with pytest.raises(PermissionError, match="jamming_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, jamming_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_radio", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_radio", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_radio", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "spectrum_license", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	band = api.record_band_plan({"tenant_id": "tenant-api", "band_id": "api-band", "band_type": "public_safety", "name": "Band", "frequency_min_mhz": 150.0, "frequency_max_mhz": 170.0, "authority_id": authority["id"], "evidence_reference": "evidence"})
	receiver = api.register_receiver({"tenant_id": "tenant-api", "receiver_id": "api-receiver", "receiver_type": "sdr", "site_reference": "site", "custodian_id": "custodian", "authority_id": authority["id"], "calibration_reference": "calibration", "evidence_reference": "evidence"})
	api.record_session({"tenant_id": "tenant-api", "session_id": "api-session", "band_id": band["id"], "receiver_id": receiver["id"], "session_type": "incident_watch", "started_at": "2026-06-01", "collection_plan_reference": "plan", "evidence_reference": "evidence"})
	agent = api.register_radio_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Radio Agent", "runtime": "claude_code", "role": "signal_analyst"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.radio_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "signal_analyst"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["receivers"][0]["id"] == receiver["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_radio"]["screens"]["agents"]["route"] == "/intel-radio/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_radio", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_radio"]["streaming"]["processor"] == "bytewax"
