"""Focused contract and lifecycle tests for the CKM RTC capability."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_rtc_contract_declares_lifecycle_surfaces():
	module = _load_module("ckm_rtc_contract_under_test", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "ckm_rtc"
	assert contract["display_name"] == "Real-Time Collaboration"
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["governance"]["batch_event_stream"] == "bytewax"
	assert contract["configuration"]["rtc_agents"]["supported_runtimes"] == [
		"codex",
		"claude_code",
		"opencode",
		"pi",
	]
	assert contract["provides"] == [
		"collaboration_sessions",
		"presence_awareness",
		"real_time_messaging",
		"media_collaboration",
		"decision_capture",
		"page_collaboration",
		"rtc_agents",
	]
	assert contract["requires"] == ["auth", "conf", "audl", "ckm_not"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["batch_mutation_guardrail"] == "batch_rtc_mutation_requires_bytewax"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"rooms",
		"presence",
		"messages",
		"media",
		"decisions",
		"agents",
		"analytics",
		"audit",
	}


def test_rtc_contract_rules_cover_agents_media_decisions_and_bytewax():
	module = _load_module("ckm_rtc_contract_rules_under_test", PACKAGE_DIR / "capability_contract.py")

	no_tenant = module.evaluate_capability_rules({"tenant_context_present": False})
	assert no_tenant["decision"] == "deny"
	assert "tenant_context_required" in no_tenant["matched_rules"]

	join_blocked = module.evaluate_capability_rules({
		"operation": "join_session",
		"participant_allowed": False,
	})
	assert join_blocked["decision"] == "deny"
	assert "join_requires_allowed_participant" in join_blocked["matched_rules"]

	recording_blocked = module.evaluate_capability_rules({
		"operation": "start_recording",
		"recording_consent_present": False,
	})
	assert recording_blocked["decision"] == "deny"
	assert "recording_requires_consent" in recording_blocked["matched_rules"]

	agent_runtime = module.evaluate_capability_rules({
		"rtc_agent_present": True,
		"agent_runtime_supported": False,
	})
	assert agent_runtime["decision"] == "deny"
	assert "rtc_agent_runtime_supported" in agent_runtime["matched_rules"]

	batch = module.evaluate_capability_rules({
		"requested_operation": "batch_rtc_mutation",
		"event_stream": "other_stream",
	})
	assert batch["decision"] == "deny"
	assert "batch_rtc_mutation_requires_bytewax" in batch["matched_rules"]


def test_rtc_lifecycle_service_enforces_guardrails():
	package = importlib.import_module("capabilities.ckm.rtc")
	service = package.RtcLifecycleService("tenant-test")

	agent = service.register_rtc_agent(
		name="Decision reviewer",
		runtime="codex",
		role="decision_reviewer",
		scope="review decisions and trace evidence",
	)
	assert agent["runtime"] == "codex"
	assert agent["role"] == "decision_reviewer"

	session = service.create_session(
		session_id="close-room",
		name="Close review",
		owner_id="user-cfo",
		context_ref="fin.glr/period/2026-05",
		participant_policy=["user-cfo", "user-controller"],
	)
	assert session["status"] == "active"

	participant = service.join_session("close-room", "user-controller", role="editor")
	assert participant["user_id"] == "user-controller"

	with pytest.raises(PermissionError, match="participant_not_allowed"):
		service.join_session("close-room", "user-outsider")

	presence = service.update_presence(
		session_id="close-room",
		user_id="user-controller",
		status="active",
		heartbeat_id="heartbeat-1",
		context_ref="fin.glr/journal-review",
	)
	assert presence["heartbeat_id"] == "heartbeat-1"

	message = service.post_message("close-room", "user-controller", "Variance review is ready.")
	assert message["status"] == "posted"
	assert message["decision"] == "allow"

	review_message = service.post_message(
		"close-room",
		"user-controller",
		"Sensitive variance note.",
		sensitive_content_detected=True,
	)
	assert review_message["status"] == "review_required"
	assert "sensitive_content_review_required" in review_message["reasons"]

	with pytest.raises(PermissionError, match="screen_share_permission_required"):
		service.start_screen_share("close-room", "user-controller", permission_granted=False)

	recording = service.start_recording("close-room", "user-cfo", consent_ref="consent/close-room")
	assert recording["status"] == "started"

	decision = service.capture_decision(
		session_id="close-room",
		owner_id="user-cfo",
		decision_text="Approve accrual adjustment batch A.",
		trace_ref="audit/decision/close-room/1",
	)
	assert decision["trace_ref"] == "audit/decision/close-room/1"

	assert service.validate_batch_rtc_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_rtc_mutation("other-stream")["decision"] == "deny"
	summary = service.dashboard_summary()
	assert summary["rtc_agent_count"] == 1
	assert summary["active_session_count"] == 1
	assert summary["presence_count"] == 1
	assert summary["decision_count"] == 1


def test_rtc_generated_evidence_and_docs_are_current():
	app = _load_module("ckm_rtc_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["ckm_rtc"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["ckm_rtc"]["screens"]["agents"]["route"] == "/ckm-rtc/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
