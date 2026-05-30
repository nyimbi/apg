"""Regression coverage for the CVSN executable capability contract."""

import pytest

from capabilities.common.cvsn import get_capability_info, register_capability
from capabilities.common.cvsn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.cvsn.cvsn_runtime import CvsnService
from capabilities.common.cvsn.view_models import (
	asset_workbench_model,
	audit_timeline_model,
	dashboard_model,
	document_processing_model,
	governance_model,
	image_analysis_model,
	model_registry_model,
	quality_inspection_model,
	review_console_model,
	safety_console_model,
	similarity_search_model,
	video_analytics_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-vision", {"processing": {"max_file_size_mb": 25}})

	assert contract["capability"] == "cvsn"
	assert contract["display_name"] == "Computer Vision"
	assert contract["configuration"]["tenant_id"] == "tenant-vision"
	assert contract["configuration"]["processing"]["max_file_size_mb"] == 25
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"processing",
		"vision_tasks",
		"ocr",
		"detection",
		"video",
		"quality",
		"safety",
		"privacy",
		"model_registry",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"assets",
		"documents",
		"images",
		"video",
		"quality",
		"safety",
		"similarity",
		"review",
		"models",
		"governance",
		"audit",
		"settings"
	}
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "cvsn_runtime.CvsnService"
	assert contract["ui"]["api_prefix"] == "/cvsn/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "quality_defect_marker" in contract["theme"]["components"]


def test_rule_engine_denies_unsafe_vision_workloads():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "run_job",
		"task_enabled": False,
		"operator_present": False,
		"processing_type": "facial_analysis",
		"consent_recorded": False,
		"anonymization_enabled": False,
		"retention_days": 90,
		"batch_size": 25,
		"async_queue_enabled": False,
	})
	stream_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_vision_events",
		"event_stream": "kafka",
	})
	bytewax_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_vision_events",
		"event_stream": "bytewax",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"task_must_be_enabled",
		"operator_required",
		"facial_analysis_requires_consent",
		"facial_analysis_requires_anonymization",
		"biometric_retention_requires_limit",
		"large_batch_requires_async_queue",
	}
	assert stream_result["matched_rules"] == ["vision_events_require_bytewax"]
	assert bytewax_result["decision"] == "allow"


def test_registration_includes_full_capability_contract():
	info = get_capability_info()
	registration = register_capability()

	assert info["configuration"]["tenant_id"] == "default"
	assert info["rule_engine"]["type"] == "deterministic"
	assert info["ui_manifest"]["requires_theme"] is True
	assert info["theme"]["name"] == "cvsn_industrial"
	assert {route["name"] for route in info["ui_manifest"]["routes"]} >= {"quality", "safety", "models"}
	assert registration["name"] == "cvsn"
	assert registration["ui_components"]["quality"] == "/cvsn/quality"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert "aicr" in registration["dependencies"]
	assert "moni" in registration["dependencies"]
	assert "cv:object_detection" in registration["permissions"]
	assert "cv:review" in registration["permissions"]


def test_cvsn_lifecycle_is_executable():
	service = CvsnService()

	asset = service.ingest_asset(
		"asset-001",
		"tenant-vision",
		"image",
		"image/png",
		2.5,
		"s3://tenant-vision/frame-001.png",
	)
	model = service.register_model(
		"model-001",
		"tenant-vision",
		"Defect Detector",
		"quality_inspection",
		"mlcm://vision/defect-detector",
		"quality-team",
		"1.0.0",
		"model-card://vision/defect-detector",
	)
	pipeline = service.register_pipeline(
		"pipe-001",
		"tenant-vision",
		"Factory Quality Pipeline",
		"quality-team",
		model["id"],
		"1.0.0",
		["quality_inspection", "object_detection"],
	)
	job = service.run_job(
		"job-001",
		"tenant-vision",
		asset["id"],
		pipeline["tasks"][0],
		"operator-1",
		inspection_plan_attached=True,
		defect_taxonomy_attached=True,
	)
	image_job = service.run_job("job-002", "tenant-vision", asset["id"], "object_detection", "operator-1")
	release = service.release_model(model["id"], "tenant-vision", True, True)

	assert asset["content_hash"]
	assert job["results"]["inspection_result"] == "pass"
	assert image_job["results"]["object_count"] == 1
	assert release["status"] == "released"
	assert service.dashboard_summary("tenant-vision") == {
		"tenant_id": "tenant-vision",
		"asset_count": 1,
		"job_count": 2,
		"document_job_count": 0,
		"image_job_count": 1,
		"video_job_count": 0,
		"quality_job_count": 1,
		"safety_job_count": 0,
		"model_count": 1,
		"released_model_count": 1,
		"pipeline_count": 1,
		"audit_event_count": 6,
	}
	assert dashboard_model(service, "tenant-vision")["summary"]["job_count"] == 2
	assert asset_workbench_model(service, "tenant-vision")["assets"][0]["id"] == "asset-001"
	assert document_processing_model(service, "tenant-vision")["ocr_jobs"] == []
	assert image_analysis_model(service, "tenant-vision")["image_jobs"][0]["id"] == "job-002"
	assert video_analytics_model(service, "tenant-vision")["video_jobs"] == []
	assert quality_inspection_model(service, "tenant-vision")["quality_jobs"][0]["id"] == "job-001"
	assert safety_console_model(service, "tenant-vision")["safety_jobs"] == []
	assert similarity_search_model(service, "tenant-vision")["search_index_adapter"] == "srch"
	assert review_console_model(service, "tenant-vision")["low_confidence_jobs"] == []
	assert model_registry_model(service, "tenant-vision")["models"][0]["status"] == "released"
	assert governance_model(service, "tenant-vision")["rules"]
	assert audit_timeline_model(service, "tenant-vision")["audit_events"]


def test_cvsn_service_enforces_policy_guardrails_before_processing():
	service = CvsnService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.ingest_asset("asset", "", "image", "image/png", 1, "s3://asset")
	with pytest.raises(PermissionError, match="asset_source_required"):
		service.ingest_asset("asset", "tenant-vision", "image", "image/png", 1, "")
	with pytest.raises(PermissionError, match="unsupported_mime_type"):
		service.ingest_asset("asset", "tenant-vision", "image", "application/pdf", 1, "s3://asset")
	with pytest.raises(PermissionError, match="asset_too_large"):
		service.ingest_asset("asset", "tenant-vision", "image", "image/png", 55, "s3://asset")

	service.ingest_asset("image", "tenant-vision", "image", "image/png", 1, "s3://image")
	service.ingest_asset("video", "tenant-vision", "video", "video/mp4", 5, "s3://video")
	task_calls: list[str] = []

	def forbidden_processing(asset, processing_type):
		task_calls.append(processing_type)
		return 0.99, {}

	service._run_processing = forbidden_processing  # type: ignore[method-assign]
	with pytest.raises(PermissionError, match="operator_required"):
		service.run_job("job-no-operator", "tenant-vision", "image", "object_detection", "")
	with pytest.raises(PermissionError, match="ocr_requires_document_or_image"):
		service.run_job("job-bad-ocr", "tenant-vision", "video", "ocr", "operator")
	with pytest.raises(PermissionError, match="video_asset_required"):
		service.run_job("job-bad-video", "tenant-vision", "image", "video_analytics", "operator")
	with pytest.raises(PermissionError, match="inspection_plan_required"):
		service.run_job("job-no-plan", "tenant-vision", "image", "quality_inspection", "operator", inspection_plan_attached=False)
	with pytest.raises(PermissionError, match="biometric_consent_required"):
		service.run_job("job-face", "tenant-vision", "image", "facial_analysis", "operator", consent_recorded=False)
	with pytest.raises(PermissionError, match="sampling_policy_required"):
		service.run_job("job-video", "tenant-vision", "video", "video_analytics", "operator", sampling_policy_attached=False)
	assert task_calls == []

	def critical_defect(asset, processing_type):
		return 0.95, {"inspection_result": "fail", "critical_defect_detected": True, "severity": "critical"}

	service._run_processing = critical_defect  # type: ignore[method-assign]
	with pytest.raises(PermissionError, match="critical_defect_alert_required"):
		service.run_job(
			"job-critical-defect",
			"tenant-vision",
			"image",
			"quality_inspection",
			"operator",
			alerting_enabled=False,
		)
	with pytest.raises(PermissionError, match="incident_acknowledgement_required"):
		service.run_job(
			"job-critical-safety",
			"tenant-vision",
			"image",
			"factory_safety",
			"operator",
			incident_acknowledged=False,
		)

	with pytest.raises(PermissionError, match="mlcm_model_ref_required"):
		service.register_model("model", "tenant-vision", "Model", "object_detection", "", "owner", "1.0.0", "card")
	with pytest.raises(PermissionError, match="model_card_required"):
		service.register_model("model", "tenant-vision", "Model", "object_detection", "mlcm://model", "owner", "1.0.0", "")
	with pytest.raises(PermissionError, match="model_version_required"):
		service.register_model("model", "tenant-vision", "Model", "object_detection", "mlcm://model", "owner", "", "card")
	service.register_model("model", "tenant-vision", "Model", "object_detection", "mlcm://model", "owner", "1.0.0", "card")
	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_pipeline("pipe", "", "Pipeline", "owner", "model", "1.0.0", ["object_detection"])
	with pytest.raises(PermissionError, match="vision_task_required"):
		service.register_pipeline("pipe", "tenant-vision", "Pipeline", "owner", "model", "1.0.0", [])
	with pytest.raises(PermissionError, match="vision_task_not_enabled"):
		service.register_pipeline("pipe", "tenant-vision", "Pipeline", "owner", "model", "1.0.0", ["not_enabled"])
	with pytest.raises(PermissionError, match="model_evaluation_required"):
		service.release_model("model", "tenant-vision", False, True)
	with pytest.raises(PermissionError, match="model_release_approval_required"):
		service.release_model("model", "tenant-vision", True, False)
