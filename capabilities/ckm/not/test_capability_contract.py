"""Focused contract and lifecycle tests for the CKM notification capability."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys

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


def test_notification_contract_declares_lifecycle_surfaces():
	module = _load_module("ckm_not_contract_under_test", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "ckm_not"
	assert contract["display_name"] == "Notification System"
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["governance"]["batch_event_stream"] == "bytewax"
	assert contract["configuration"]["notification_agents"]["supported_runtimes"] == [
		"codex",
		"claude_code",
		"opencode",
		"pi",
	]
	assert contract["provides"] == [
		"notification_delivery",
		"template_management",
		"campaign_orchestration",
		"preference_center",
		"channel_provider_registry",
		"engagement_analytics",
		"notification_agents",
	]
	assert contract["requires"] == ["auth", "conf", "encr", "audl"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["batch_mutation_guardrail"] == "batch_notification_mutation_requires_bytewax"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"templates",
		"campaigns",
		"deliveries",
		"preferences",
		"providers",
		"agents",
		"analytics",
		"audit",
	}


def test_notification_contract_rules_cover_agents_consent_and_bytewax():
	module = _load_module("ckm_not_contract_rules_under_test", PACKAGE_DIR / "capability_contract.py")

	no_tenant = module.evaluate_capability_rules({"tenant_context_present": False})
	assert no_tenant["decision"] == "deny"
	assert "tenant_context_required" in no_tenant["matched_rules"]

	no_consent = module.evaluate_capability_rules({
		"delivery_requested": True,
		"external_channel_requested": True,
		"recipient_consent_present": False,
	})
	assert no_consent["decision"] == "deny"
	assert "external_delivery_requires_consent" in no_consent["matched_rules"]

	disallowed_channel = module.evaluate_capability_rules({
		"delivery_requested": True,
		"channel_allowed_by_preference": False,
	})
	assert disallowed_channel["decision"] == "deny"
	assert "delivery_channel_must_be_allowed" in disallowed_channel["matched_rules"]

	agent_runtime = module.evaluate_capability_rules({
		"notification_agent_present": True,
		"agent_runtime_supported": False,
	})
	assert agent_runtime["decision"] == "deny"
	assert "notification_agent_runtime_supported" in agent_runtime["matched_rules"]

	batch = module.evaluate_capability_rules({
		"requested_operation": "batch_notification_mutation",
		"event_stream": "other_stream",
	})
	assert batch["decision"] == "deny"
	assert "batch_notification_mutation_requires_bytewax" in batch["matched_rules"]


def test_notification_lifecycle_service_enforces_guardrails():
	package = importlib.import_module("capabilities.ckm.not")
	service = package.NotificationLifecycleService("tenant-test")

	agent = service.register_notification_agent(
		name="Delivery reviewer",
		runtime="codex",
		role="delivery_reviewer",
		scope="review delivery exceptions",
	)
	assert agent["runtime"] == "codex"
	assert agent["role"] == "delivery_reviewer"

	provider = service.register_provider(
		provider_id="email-primary",
		name="Primary email",
		channel="email",
		secret_ref="secret/not/email-primary",
	)
	assert provider["channel"] == "email"
	assert provider["secret_ref"] == "secret/not/email-primary"

	template = service.create_template(
		template_id="case-update",
		name="Case update",
		channels=["email", "in_app"],
		content={"email": "Case {{case_id}} changed.", "in_app": "Case {{case_id}} changed."},
		variable_schema={"case_id": {"type": "string"}},
	)
	assert template["status"] == "draft"
	assert service.approve_template("case-update", reviewer_id="reviewer-1")["approved"] is True

	service.set_preference(
		recipient_id="recipient-1",
		allowed_channels=["email", "in_app"],
		consent_refs={"email": "consent-record-1"},
	)
	delivery = service.request_delivery(
		template_id="case-update",
		recipient_id="recipient-1",
		channels=["email"],
		topic="case",
	)
	assert delivery["status"] == "queued"
	assert delivery["decision"] == "allow"

	deferred = service.request_delivery(
		template_id="case-update",
		recipient_id="recipient-1",
		channels=["email"],
		topic="case",
		within_quiet_hours=True,
	)
	assert deferred["status"] == "deferred"
	assert deferred["decision"] == "allow"

	service.set_preference(
		recipient_id="recipient-3",
		allowed_channels=["email"],
	)
	disallowed = service.request_delivery(
		template_id="case-update",
		recipient_id="recipient-3",
		channels=["in_app"],
		topic="case",
	)
	assert disallowed["status"] == "blocked"
	assert "notification_channel_not_allowed" in disallowed["reasons"]

	service.set_preference(
		recipient_id="recipient-2",
		allowed_channels=["email"],
		suppressed_topics=["case"],
		consent_refs={"email": "consent-record-2"},
	)
	blocked = service.request_delivery(
		template_id="case-update",
		recipient_id="recipient-2",
		channels=["email"],
		topic="case",
	)
	assert blocked["status"] == "blocked"
	assert "recipient_suppressed" in blocked["reasons"]

	assert service.validate_batch_notification_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_notification_mutation("other-stream")["decision"] == "deny"
	assert service.dashboard_summary()["notification_agent_count"] == 1
	assert service.dashboard_summary()["provider_count"] == 1


def test_notification_generated_evidence_and_docs_are_current():
	app = _load_module("ckm_not_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["ckm_not"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["ckm_not"]["screens"]["agents"]["route"] == "/ckm-not/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
