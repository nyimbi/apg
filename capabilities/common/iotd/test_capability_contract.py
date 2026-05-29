"""Regression coverage for the IOTD executable capability contract."""

import pytest

from capabilities.common.iotd import register_capability
from capabilities.common.iotd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.iotd.service import IotdService
from capabilities.common.iotd.views import (
	command_center_model,
	dashboard_model,
	device_console_model,
	firmware_manager_model,
	rules_model,
	security_model,
	telemetry_monitor_model,
)


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


def test_iotd_lifecycle_is_executable():
	service = IotdService()

	device = service.register_device(
		device_id="device-1",
		tenant_id="tenant-iot",
		device_key="dev-key-1",
		owner_id="ops-owner",
		certificate_id="cert-1",
		fleet_id="line-a",
	)
	telemetry = service.ingest_telemetry(
		event_id="telemetry-1",
		tenant_id="tenant-iot",
		device_id="device-1",
		schema_name="temperature",
		payload={"timestamp": "2026-05-29T12:00:00Z", "temperature": 42.5},
		encryption_applied=True,
	)
	command = service.dispatch_command(
		command_id="command-1",
		tenant_id="tenant-iot",
		device_id="device-1",
		command="restart",
		dangerous=True,
		approval_id="approval-1",
	)
	ack = service.acknowledge_command("command-1", "tenant-iot", "restarted")
	firmware = service.register_firmware(
		firmware_id="fw-1",
		tenant_id="tenant-iot",
		version="1.2.3",
		artifact_uri="s3://firmware/fw-1.bin",
		signature_id="sig-1",
	)
	deployment = service.deploy_firmware("deploy-1", "tenant-iot", "fw-1", "line-a", ["device-1"])
	report = service.health_report("health-1", "tenant-iot")

	assert device["status"] == "provisioned"
	assert telemetry["accepted"] is True
	assert command["status"] == "dispatched"
	assert ack["status"] == "acknowledged"
	assert firmware["signature_verified"] is True
	assert deployment["device_ids"] == ["device-1"]
	assert report["online_device_count"] == 1
	assert service.dashboard_summary("tenant-iot")["device_count"] == 1
	assert device_console_model(service, "tenant-iot")["devices"][0]["id"] == "device-1"
	assert telemetry_monitor_model(service, "tenant-iot")["events"][0]["schema_name"] == "temperature"
	assert command_center_model(service, "tenant-iot")["commands"][0]["status"] == "acknowledged"
	assert firmware_manager_model(service, "tenant-iot")["deployments"][0]["id"] == "deploy-1"
	assert security_model(service, "tenant-iot")["audit_events"]
	assert rules_model(service, "tenant-iot")["health_reports"][0]["id"] == "health-1"
	assert dashboard_model(service, "tenant-iot")["summary"]["audit_event_count"] >= 6


def test_iotd_service_enforces_policy_guardrails():
	service = IotdService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_device("missing-tenant", "", "device-key", "owner", "cert")

	with pytest.raises(PermissionError, match="device_identity_required"):
		service.register_device("missing-key", "tenant-iot", "", "owner", "cert")

	with pytest.raises(PermissionError, match="device_owner_required"):
		service.register_device("missing-owner", "tenant-iot", "device-key", "", "cert")

	with pytest.raises(PermissionError, match="stale_device_review_required"):
		service.register_device("stale", "tenant-iot", "device-key", "owner", "cert", last_seen_days=45, stale_device_reviewed=False)

	service.register_device("device-1", "tenant-iot", "device-key", "owner", "cert")

	with pytest.raises(PermissionError, match="telemetry_encryption_required"):
		service.ingest_telemetry("telemetry-plain", "tenant-iot", "device-1", "temperature", {"timestamp": "now"}, encryption_applied=False)

	with pytest.raises(PermissionError, match="telemetry_schema_invalid"):
		service.ingest_telemetry("telemetry-invalid", "tenant-iot", "device-1", "temperature", {"temperature": 42.5})

	with pytest.raises(PermissionError, match="command_approval_required"):
		service.dispatch_command("command-danger", "tenant-iot", "device-1", "factory_reset", dangerous=True)

	with pytest.raises(PermissionError, match="firmware_signature_required"):
		service.register_firmware("fw-unsigned", "tenant-iot", "0.0.1", "s3://fw.bin", "", firmware_signature_verified=False)

	with pytest.raises(KeyError, match="device_missing"):
		service.ingest_telemetry("wrong-tenant", "other-tenant", "device-1", "temperature", {"timestamp": "now"})
