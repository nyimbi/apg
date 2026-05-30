"""Regression coverage for the IOTD executable capability contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.iotd import register_capability
from capabilities.common.iotd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.iotd.service import IotdService
from capabilities.common.iotd.views import (
	audit_trail_model,
	command_center_model,
	dashboard_model,
	device_console_model,
	firmware_manager_model,
	health_model,
	iotd_agent_model,
	rules_model,
	security_model,
	telemetry_monitor_model,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-iotd", {"devices": {"certificate_rotation_days": 60}})

	assert contract["capability"] == "iotd"
	assert contract["configuration"]["tenant_id"] == "tenant-iotd"
	assert contract["configuration"]["devices"]["certificate_rotation_days"] == 60
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"devices",
		"telemetry",
		"commands",
		"firmware",
		"iotd_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["provides"] == [
		"device_registry",
		"telemetry_ingestion",
		"command_dispatch",
		"firmware_lifecycle",
		"device_security",
		"device_health",
		"iotd_agents",
	]
	assert contract["requires"] == ["auth", "encr", "audl", "conf"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["iotd_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "devices", "telemetry", "commands", "firmware", "agents", "health", "security", "rules", "audit", "settings"}
	assert contract["theme"]["name"] == "iotd_device_ops"


def test_rule_engine_enforces_iotd_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_device", "device_identity_present": False, "dangerous_command": True, "approval_recorded": False, "last_seen_days": 45, "stale_device_reviewed": False})
	telemetry_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "ingest_telemetry", "event_stream": "other-stream", "encryption_applied": False, "schema_valid": False})
	agent_result = evaluate_capability_rules({"iotd_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_iot_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "device_requires_identity", "dangerous_command_requires_approval", "stale_device_requires_review"}
	assert set(telemetry_result["matched_rules"]) == {"telemetry_requires_bytewax_stream", "telemetry_requires_encryption", "telemetry_requires_schema"}
	assert agent_result["decision"] == "deny"
	assert agent_result["matched_rules"] == ["iotd_agent_runtime_supported"]
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_iot_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "iotd"
	assert "auth" in registration["dependencies"]
	assert registration["ui_components"]["commands"] == "/iotd/commands"
	assert registration["ui_components"]["agents"] == "/iotd/agents"
	assert registration["streaming"]["processor"] == "bytewax"
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
	agent = service.register_iotd_agent(
		tenant_id="tenant-iot",
		name="Fleet reviewer",
		runtime="codex",
		role="fleet_operator",
		scope="review fleet telemetry and command risk",
	)

	assert device["status"] == "provisioned"
	assert telemetry["event_bus"] == "bytewax"
	assert telemetry["accepted"] is True
	assert command["status"] == "dispatched"
	assert ack["status"] == "acknowledged"
	assert firmware["signature_verified"] is True
	assert deployment["device_ids"] == ["device-1"]
	assert report["online_device_count"] == 1
	assert agent["runtime"] == "codex"
	assert agent["role"] == "fleet_operator"
	assert service.dashboard_summary("tenant-iot")["device_count"] == 1
	assert service.dashboard_summary("tenant-iot")["iotd_agent_count"] == 1
	assert service.validate_batch_iot_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_iot_mutation("other-stream")["decision"] == "deny"
	assert device_console_model(service, "tenant-iot")["devices"][0]["id"] == "device-1"
	assert telemetry_monitor_model(service, "tenant-iot")["events"][0]["schema_name"] == "temperature"
	assert command_center_model(service, "tenant-iot")["commands"][0]["status"] == "acknowledged"
	assert firmware_manager_model(service, "tenant-iot")["deployments"][0]["id"] == "deploy-1"
	assert security_model(service, "tenant-iot")["audit_events"]
	assert rules_model(service, "tenant-iot")["health_reports"][0]["id"] == "health-1"
	assert iotd_agent_model(service, "tenant-iot")["iotd_agents"][0]["role"] == "fleet_operator"
	assert audit_trail_model(service, "tenant-iot")["audit_events"]
	assert health_model(service, "tenant-iot")["summary"]["device_count"] == 1
	assert dashboard_model(service, "tenant-iot")["summary"]["audit_event_count"] >= 7


def test_iotd_service_enforces_policy_guardrails():
	service = IotdService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_device("missing-tenant", "", "device-key", "owner", "cert")

	with pytest.raises(PermissionError, match="device_identity_required"):
		service.register_device("missing-key", "tenant-iot", "", "owner", "cert")

	with pytest.raises(PermissionError, match="device_owner_required"):
		service.register_device("missing-owner", "tenant-iot", "device-key", "", "cert")
	with pytest.raises(PermissionError, match="device_certificate_required"):
		service.register_device("missing-cert", "tenant-iot", "device-key", "owner", "")

	with pytest.raises(PermissionError, match="stale_device_review_required"):
		service.register_device("stale", "tenant-iot", "device-key", "owner", "cert", last_seen_days=45, stale_device_reviewed=False)

	service.register_device("device-1", "tenant-iot", "device-key", "owner", "cert")

	with pytest.raises(PermissionError, match="telemetry_encryption_required"):
		service.ingest_telemetry("telemetry-plain", "tenant-iot", "device-1", "temperature", {"timestamp": "now"}, encryption_applied=False)
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.ingest_telemetry("telemetry-stream", "tenant-iot", "device-1", "temperature", {"timestamp": "now"}, event_bus="other-stream")

	with pytest.raises(PermissionError, match="telemetry_schema_invalid"):
		service.ingest_telemetry("telemetry-invalid", "tenant-iot", "device-1", "temperature", {"temperature": 42.5})

	with pytest.raises(PermissionError, match="command_approval_required"):
		service.dispatch_command("command-danger", "tenant-iot", "device-1", "factory_reset", dangerous=True)
	with pytest.raises(PermissionError, match="command_name_required"):
		service.dispatch_command("command-empty", "tenant-iot", "device-1", "")

	with pytest.raises(PermissionError, match="firmware_signature_required"):
		service.register_firmware("fw-unsigned", "tenant-iot", "0.0.1", "s3://fw.bin", "", firmware_signature_verified=False)
	with pytest.raises(PermissionError, match="firmware_artifact_required"):
		service.register_firmware("fw-missing-artifact", "tenant-iot", "0.0.1", "", "sig")
	with pytest.raises(PermissionError, match="deployment_devices_required"):
		service.deploy_firmware("deploy-empty", "tenant-iot", service.register_firmware("fw-ok", "tenant-iot", "1.0.0", "s3://fw.bin", "sig")["id"], "default", [])

	with pytest.raises(KeyError, match="device_missing"):
		service.ingest_telemetry("wrong-tenant", "other-tenant", "device-1", "temperature", {"timestamp": "now"})
	with pytest.raises(PermissionError, match="iotd_agent_runtime_not_supported"):
		service.register_iotd_agent("tenant-iot", "Unsupported", "unsupported", "fleet_operator", "review")


def test_lifecycle_ids_are_tenant_scoped():
	service = IotdService()

	for tenant_id, owner, temperature in (
		("tenant-a", "owner-a", 21.5),
		("tenant-b", "owner-b", 29.0),
	):
		service.register_device("shared-device", tenant_id, "device-key", owner, "cert")
		service.ingest_telemetry("shared-event", tenant_id, "shared-device", "temperature", {"timestamp": "now", "temperature": temperature})
		service.dispatch_command("shared-command", tenant_id, "shared-device", "ping")
		service.register_iotd_agent(tenant_id, "Reviewer", "codex", "fleet_operator", "review tenant fleet", agent_id="shared-agent")

	assert service.list_devices("tenant-a")[0]["owner_id"] == "owner-a"
	assert service.list_devices("tenant-b")[0]["owner_id"] == "owner-b"
	assert service.list_telemetry("tenant-a")[0]["payload"]["temperature"] == 21.5
	assert service.list_telemetry("tenant-b")[0]["payload"]["temperature"] == 29.0
	assert service.list_commands("tenant-a")[0]["id"] == "shared-command"
	assert service.list_commands("tenant-b")[0]["id"] == "shared-command"
	assert service.list_iotd_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_iotd_agents("tenant-b")[0]["id"] == "shared-agent"


def test_generated_evidence_and_docs_are_current():
	app = _load_module("iotd_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["iotd"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["iotd"]["screens"]["agents"]["route"] == "/iotd/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
