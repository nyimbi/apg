"""Regression coverage for the IOTD executable capability contract."""

from capabilities.common.iotd import register_capability
from capabilities.common.iotd.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-iotd", {"devices": {"certificate_rotation_days": 60}})

	assert contract["capability"] == "iotd"
	assert contract["configuration"]["tenant_id"] == "tenant-iotd"
	assert contract["configuration"]["devices"]["certificate_rotation_days"] == 60
	assert contract["configuration_schema"]["required"] == ["tenant_id", "devices", "telemetry", "commands", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "devices", "telemetry", "commands", "firmware", "security", "rules", "settings"}
	assert contract["theme"]["name"] == "iotd_device_ops"


def test_rule_engine_enforces_iotd_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_device", "device_identity_present": False, "dangerous_command": True, "approval_recorded": False, "last_seen_days": 45, "stale_device_reviewed": False})
	telemetry_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "ingest_telemetry", "encryption_applied": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "device_requires_identity", "dangerous_command_requires_approval", "stale_device_requires_review"}
	assert telemetry_result["matched_rules"] == ["telemetry_requires_encryption"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "iotd"
	assert "mqeb" in registration["dependencies"]
	assert registration["ui_components"]["commands"] == "/iotd/commands"
	assert "iotd:command" in registration["permissions"]
