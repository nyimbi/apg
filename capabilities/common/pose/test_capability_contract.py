"""Regression coverage for the POSE executable capability contract."""

import pytest

from capabilities.common.pose import register_capability
from capabilities.common.pose.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.pose.service import PoseService
from capabilities.common.pose import views


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-pose", {"tracking": {"max_persons_per_frame": 12}})

	assert contract["capability"] == "pose"
	assert contract["configuration"]["tenant_id"] == "tenant-pose"
	assert contract["configuration"]["tracking"]["max_persons_per_frame"] == 12
	assert contract["configuration_schema"]["required"] == ["tenant_id", "models", "sessions", "tracking", "analysis", "pose_agents", "governance", "observability", "adapters", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 20
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "estimate", "tracking", "analysis", "reconstruction", "sessions", "models", "quality", "agents", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/pose/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "pose_viewer" in contract["theme"]["components"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert "codex" in contract["configuration"]["pose_agents"]["supported_runtimes"]


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
	agent_result = evaluate_capability_rules({
		"pose_agent_present": True,
		"agent_registered": False,
		"agent_runtime_supported": False,
		"agent_scope_present": False,
		"agent_contribution_disclosed": False,
	})
	stream_result = evaluate_capability_rules({"operation": "batch_pose_mutation", "event_stream": "memory"})
	assert set(agent_result["matched_rules"]) == {"pose_agent_requires_registration", "pose_agent_runtime_supported", "pose_agent_requires_scope", "pose_agent_requires_disclosure"}
	assert stream_result["matched_rules"] == ["batch_pose_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "pose"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "pose_motion_intelligence"
	assert registration["streaming"]["processor"] == "bytewax"
	assert registration["ui_components"]["tracking"] == "/pose/tracking"
	assert registration["ui_components"]["agents"] == "/pose/agents"
	assert "cvsn" in registration["dependencies"]
	assert "pose:track" in registration["permissions"]
	assert "pose_agents" in registration["capabilities"]


def test_pose_lifecycle_is_executable():
	service = PoseService()
	tenant_id = "tenant-pose"

	model = service.register_model(
		model_id="rtmpose",
		tenant_id=tenant_id,
		name="RTMPose",
		model_type="rtmpose",
		owner="vision-team",
		policy_ref="pose-policy:default",
		minimum_keypoint_confidence=0.7,
		edge_ready=True,
	)
	session = service.start_session(
		session_id="session-001",
		tenant_id=tenant_id,
		name="Movement Study",
		owner="coach",
		source_ref="camera:studio-a",
		model_id=model["id"],
		subject_consent_recorded=True,
		secure_stream=True,
		realtime_stream=True,
		max_persons=2,
	)
	frame = service.record_frame(
		frame_id="frame-001",
		tenant_id=tenant_id,
		session_id=session["id"],
		frame_number=1,
		occurred_at="2026-05-30T10:00:00Z",
		source_ref="frame://001",
		width=1920,
		height=1080,
	)
	estimate = service.estimate_pose(
		estimate_id="estimate-001",
		tenant_id=tenant_id,
		session_id=session["id"],
		frame_id=frame["id"],
		model_id=model["id"],
		keypoints=[
			{"name": "left_shoulder", "x": 100, "y": 120, "confidence": 0.96},
			{"name": "right_shoulder", "x": 180, "y": 122, "confidence": 0.94},
		],
		person_count=1,
	)
	analysis = service.analyze_pose(
		analysis_id="analysis-001",
		tenant_id=tenant_id,
		estimation_id=estimate["id"],
		analysis_type="biomechanical",
	)
	reconstruction = service.reconstruct_3d(
		reconstruction_id="reconstruction-001",
		tenant_id=tenant_id,
		estimation_id=estimate["id"],
		camera_calibration_ref="calibration:studio-a",
	)
	agent = service.register_pose_agent(
		agent_id="codex-pose-agent",
		tenant_id=tenant_id,
		name="Codex Pose Agent",
		runtime="codex",
		role="quality_reviewer",
		scope="quality,analysis",
		contribution_disclosed=True,
		policy_ref="agent-policy:pose",
	)
	closed = service.change_session_state(tenant_id, session["id"], "completed", "study complete")

	assert estimate["confidence"] >= 0.9
	assert analysis["metrics"]["keypoint_count"] == 2
	assert reconstruction["keypoints_3d"][0]["z"] > 0
	assert agent["runtime"] == "codex"
	assert closed["status"] == "completed"

	summary = service.dashboard_summary(tenant_id)
	assert summary["model_count"] == 1
	assert summary["session_count"] == 1
	assert summary["estimate_count"] == 1
	assert summary["analysis_count"] == 1
	assert summary["reconstruction_count"] == 1
	assert summary["agent_count"] == 1
	assert views.dashboard_model(service, tenant_id)["summary"]["estimate_count"] == 1
	assert views.estimator_model(service, tenant_id)["estimates"][0]["id"] == "estimate-001"
	assert views.tracking_console_model(service, tenant_id)["sessions"][0]["id"] == "session-001"
	assert views.analysis_workbench_model(service, tenant_id)["analyses"][0]["id"] == "analysis-001"
	assert views.reconstruction_model(service, tenant_id)["reconstructions"][0]["id"] == "reconstruction-001"
	assert views.pose_agents_model(service, tenant_id)["agents"][0]["id"] == "codex-pose-agent"
	assert views.audit_trail_model(service, tenant_id)["audit_events"]
	assert views.analytics_model(service, tenant_id)["summary"]["agent_count"] == 1


def test_pose_service_enforces_guardrails():
	service = PoseService()
	tenant_id = "tenant-pose"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_model("bad", "", "Bad", "rtmpose", "owner", "policy")

	with pytest.raises(PermissionError, match="model_policy_required"):
		service.register_model("bad-policy", tenant_id, "Bad Policy", "rtmpose", "owner", "")

	model = service.register_model("model", tenant_id, "Model", "rtmpose", "owner", "policy", minimum_keypoint_confidence=0.7)

	with pytest.raises(PermissionError, match="subject_consent_required"):
		service.start_session("session-no-consent", tenant_id, "No Consent", "owner", "camera:1", model["id"], False, True)

	with pytest.raises(PermissionError, match="secure_stream_required"):
		service.start_session("session-insecure", tenant_id, "Insecure", "owner", "camera:1", model["id"], True, False, realtime_stream=True)

	session = service.start_session("session", tenant_id, "Session", "owner", "camera:1", model["id"], True, True)

	with pytest.raises(PermissionError, match="frame_timestamp_required"):
		service.record_frame("frame-bad", tenant_id, session["id"], 1, "", "frame://bad", 640, 480)

	frame = service.record_frame("frame", tenant_id, session["id"], 1, "2026-05-30T10:00:00Z", "frame://1", 640, 480)

	with pytest.raises(PermissionError, match="pose_keypoints_required"):
		service.estimate_pose("estimate-empty", tenant_id, session["id"], frame["id"], model["id"], [])

	with pytest.raises(PermissionError, match="pose_quality_review_required"):
		service.estimate_pose(
			"estimate-low",
			tenant_id,
			session["id"],
			frame["id"],
			model["id"],
			[{"name": "nose", "x": 1, "y": 1, "confidence": 0.8}],
			quality_score=0.2,
		)

	estimate = service.estimate_pose(
		"estimate",
		tenant_id,
		session["id"],
		frame["id"],
		model["id"],
		[{"name": "nose", "x": 1, "y": 1, "confidence": 0.8}],
	)

	with pytest.raises(PermissionError, match="medical_review_required"):
		service.analyze_pose("analysis-medical", tenant_id, estimate["id"], "clinical", medical_grade=True)

	with pytest.raises(PermissionError, match="camera_calibration_required"):
		service.reconstruct_3d("reconstruction-bad", tenant_id, estimate["id"], "")

	with pytest.raises(PermissionError, match="pose_agent_disclosure_required"):
		service.register_pose_agent("agent", tenant_id, "Agent", "codex", "quality_reviewer", "quality", False)
