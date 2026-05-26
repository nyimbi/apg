"""Regression coverage for the CVSN executable capability contract."""

from .. import get_capability_info, register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-a")

	assert contract["capability"] == "cvsn"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"processing",
		"ocr",
		"detection",
		"safety",
		"privacy",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 5
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"documents",
		"images",
		"video",
		"quality",
		"safety",
		"models",
		"rules",
		"settings"
	}
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "vision_canvas" in contract["theme"]["components"]
	assert "safety_alert" in contract["theme"]["components"]


def test_rule_engine_denies_unsafe_vision_workloads():
	result = evaluate_capability_rules({
		"tenant_id": "",
		"tenant_id_missing": True,
		"processing_type": "facial_recognition",
		"consent_recorded": False,
		"retention_days": 90,
		"domain": "factory_safety",
		"severity": "critical",
		"alerting_enabled": False,
		"batch_size": 25,
		"async_queue_enabled": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"require_tenant_isolation",
		"biometric_processing_requires_controls",
		"biometric_retention_requires_limit",
		"factory_hazard_requires_alerting",
		"large_batch_requires_async_queue"
	}


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
	assert "aicr" in registration["dependencies"]
	assert "cv:object_detection" in registration["permissions"]
