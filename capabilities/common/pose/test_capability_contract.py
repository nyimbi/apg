"""Regression coverage for the POSE executable capability contract."""

from capabilities.common.pose import register_capability
from capabilities.common.pose.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-pose", {"tracking": {"max_persons_per_frame": 12}})

	assert contract["capability"] == "pose"
	assert contract["configuration"]["tenant_id"] == "tenant-pose"
	assert contract["configuration"]["tracking"]["max_persons_per_frame"] == 12
	assert contract["configuration_schema"]["required"] == ["tenant_id", "models", "tracking", "analysis", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "estimate", "tracking", "analysis", "sessions", "models", "quality", "settings"}
	assert contract["ui"]["api_prefix"] == "/pose/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "pose_viewer" in contract["theme"]["components"]


def test_rule_engine_enforces_pose_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "analyze_pose",
		"subject_consent_recorded": False,
		"session_owner_assigned": False,
		"realtime_stream": True,
		"secure_stream": False,
		"sensitive_use": True,
		"approval_recorded": False,
		"pose_quality_score": 0.2,
		"quality_review_recorded": False
	})
	tracking_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "start_tracking", "session_owner_assigned": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "subject_consent_required", "secure_stream_required", "sensitive_use_requires_approval", "low_pose_quality_requires_review"}
	assert tracking_result["matched_rules"] == ["tracking_session_requires_owner"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "pose"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "pose_motion_intelligence"
	assert registration["ui_components"]["tracking"] == "/pose/tracking"
	assert "cvsn" in registration["dependencies"]
	assert "pose:track" in registration["permissions"]
