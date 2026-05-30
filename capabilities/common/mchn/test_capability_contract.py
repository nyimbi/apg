"""Regression coverage for the MCHN executable capability contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.mchn import register_capability
from capabilities.common.mchn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.mchn.service import MchnService
from capabilities.common.mchn.views import (
	analytics_model,
	audit_trail_model,
	channel_monitor_model,
	dashboard_model,
	delivery_governance_model,
	mchn_agent_model,
	policy_model,
	render_console_model,
	route_console_model,
	template_manager_model,
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


def test_contract_exposes_configuration_rules_ui_theme_and_streaming():
	contract = get_capability_contract("tenant-output", {"channels": {"fallback_required": False}})

	assert contract["capability"] == "mchn"
	assert contract["configuration"]["tenant_id"] == "tenant-output"
	assert contract["configuration"]["channels"]["fallback_required"] is False
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"channels",
		"rendering",
		"delivery",
		"mchn_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["provides"] == [
		"channel_routing",
		"format_rendering",
		"output_templates",
		"delivery_policy",
		"delivery_receipts",
		"omnichannel_analytics",
		"mchn_agents",
	]
	assert contract["requires"] == ["ntfy", "auth", "conf", "audl"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["mchn_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "render", "templates", "routes", "channels", "agents", "analytics", "policies", "audit", "settings"}
	assert contract["theme"]["name"] == "mchn_omnichannel_output"
	assert contract["ui"]["api_prefix"] == "/mchn/api/v1"


def test_rule_engine_enforces_multichannel_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_channel",
		"channel_owner_assigned": False,
		"provider_ref_present": False,
		"sensitive_output": True,
		"output_encrypted": False,
		"channel_health": "unhealthy",
		"delivery_requested": True,
		"recipient_count": 20000,
		"delivery_review_recorded": False,
	})
	template_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_template", "template_approved": False, "template_content_present": False, "template_channel_present": False})
	delivery_result = evaluate_capability_rules({"operation": "deliver_batch", "delivery_actor_present": False, "rendered_output_present": False, "recipient_count": 0, "event_stream": "other-stream"})
	agent_result = evaluate_capability_rules({"mchn_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_output_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "channel_requires_owner", "channel_requires_provider", "sensitive_output_requires_encryption", "unhealthy_channel_blocks_delivery", "large_delivery_requires_review"}
	assert set(template_result["matched_rules"]) == {"template_requires_approval", "template_requires_content", "template_requires_channel"}
	assert set(delivery_result["matched_rules"]) == {"delivery_requires_actor", "delivery_requires_rendered_output", "delivery_requires_positive_recipients", "delivery_requires_bytewax_stream"}
	assert agent_result["decision"] == "deny"
	assert agent_result["matched_rules"] == ["mchn_agent_runtime_supported"]
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_output_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "mchn"
	assert "ntfy" in registration["dependencies"]
	assert registration["ui_components"]["render"] == "/mchn/render"
	assert registration["ui_components"]["agents"] == "/mchn/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "mchn:route" in registration["permissions"]


def test_mchn_lifecycle_is_executable():
	service = MchnService()

	email = service.create_channel(
		channel_id="channel-email",
		tenant_id="tenant-output",
		name="Email primary",
		channel_type="email",
		owner="output-team",
		provider_ref="provider://email",
	)
	sms = service.create_channel(
		channel_id="channel-sms",
		tenant_id="tenant-output",
		name="SMS fallback",
		channel_type="sms",
		owner="output-team",
		provider_ref="provider://sms",
		health="degraded",
	)
	policy = service.create_delivery_policy(
		policy_id="policy-standard",
		tenant_id="tenant-output",
		name="Standard policy",
		max_recipients=25000,
		throttle_per_minute=1000,
		requires_encryption_for_sensitive=True,
		compliance_ref="compliance://output",
	)
	template = service.publish_template(
		template_id="template-invoice",
		tenant_id="tenant-output",
		name="Invoice notice",
		channel_types=("email", "sms"),
		subject_template="Invoice $invoice_id",
		body_template="Hello $customer, invoice $invoice_id is ready.",
		locale="en",
		theme_ref="mchn_omnichannel_output",
		approved=True,
		approved_by="content-owner",
	)
	route = service.create_route(
		route_id="route-invoice",
		tenant_id="tenant-output",
		name="Invoice route",
		template_id=template["id"],
		primary_channel_id=email["id"],
		fallback_channel_ids=(sms["id"],),
		policy_id=policy["id"],
	)
	rendered = service.render_output(
		output_id="output-invoice-1",
		tenant_id="tenant-output",
		route_id=route["id"],
		recipient_ref="customer:1001",
		variables={"customer": "Ada", "invoice_id": "INV-1001"},
		output_format="html",
		sensitive_output=True,
		output_encrypted=True,
	)
	batch = service.deliver_batch(
		batch_id="batch-invoice",
		tenant_id="tenant-output",
		route_id=route["id"],
		requested_by="billing-user",
		rendered_output_ids=(rendered["id"],),
		recipient_count=12000,
		delivery_review_recorded=True,
	)
	receipt = service.record_receipt(
		receipt_id="receipt-invoice",
		tenant_id="tenant-output",
		batch_id=batch["id"],
		rendered_output_id=rendered["id"],
		delivery_state="delivered",
		provider_message_id="provider-msg-1",
	)
	agent = service.register_mchn_agent(
		tenant_id="tenant-output",
		name="Delivery reviewer",
		runtime="codex",
		role="delivery_reviewer",
		scope="review large delivery batches and channel routing",
	)

	assert rendered["subject"] == "Invoice INV-1001"
	assert rendered["body"] == "Hello Ada, invoice INV-1001 is ready."
	assert rendered["status"] == "ready"
	assert batch["status"] == "queued"
	assert receipt["delivery_state"] == "delivered"
	assert agent["runtime"] == "codex"
	assert agent["role"] == "delivery_reviewer"
	assert service.dashboard_summary("tenant-output")["large_batch_count"] == 1
	assert service.dashboard_summary("tenant-output")["mchn_agent_count"] == 1
	assert service.validate_batch_output_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_output_mutation("other-stream")["decision"] == "deny"
	assert dashboard_model(service, "tenant-output")["summary"]["rendered_output_count"] == 1
	assert dashboard_model(service, "tenant-output")["streaming"]["processor"] == "bytewax"
	assert render_console_model(service, "tenant-output")["rendered_outputs"][0]["id"] == "output-invoice-1"
	assert template_manager_model(service, "tenant-output")["templates"][0]["id"] == "template-invoice"
	assert route_console_model(service, "tenant-output")["delivery_routes"][0]["id"] == "route-invoice"
	assert channel_monitor_model(service, "tenant-output")["channels"][0]["id"] == "channel-email"
	assert analytics_model(service, "tenant-output")["summary"]["delivery_batch_count"] == 1
	assert policy_model(service, "tenant-output")["policies"][0]["id"] == "policy-standard"
	assert mchn_agent_model(service, "tenant-output")["mchn_agents"][0]["role"] == "delivery_reviewer"
	assert audit_trail_model(service, "tenant-output")["audit_events"]
	assert delivery_governance_model(service, "tenant-output")["streaming"]["processor"] == "bytewax"
	assert len(service.list_audit_events("tenant-output")) >= 8


def test_mchn_service_enforces_policy_guardrails():
	service = MchnService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_delivery_policy(
			policy_id="policy-missing-tenant",
			tenant_id="",
			name="Missing tenant",
			max_recipients=100,
			throttle_per_minute=10,
			requires_encryption_for_sensitive=True,
			compliance_ref="compliance://default",
		)

	with pytest.raises(PermissionError, match="channel_owner_required"):
		service.create_channel("channel-no-owner", "tenant-output", "No owner", "email", "", "provider://email")
	with pytest.raises(PermissionError, match="channel_provider_required"):
		service.create_channel("channel-no-provider", "tenant-output", "No provider", "email", "output-team", "")

	with pytest.raises(PermissionError, match="template_approval_required"):
		service.publish_template("template-unapproved", "tenant-output", "Unapproved", ("email",), "Subject", "Body", "en", "mchn_omnichannel_output", False, "")
	with pytest.raises(PermissionError, match="template_approver_required"):
		service.publish_template("template-no-approver", "tenant-output", "No approver", ("email",), "Subject", "Body", "en", "mchn_omnichannel_output", True, "")
	with pytest.raises(PermissionError, match="template_content_required"):
		service.publish_template("template-no-content", "tenant-output", "No content", ("email",), "", "", "en", "mchn_omnichannel_output", True, "owner")
	with pytest.raises(PermissionError, match="template_channel_required"):
		service.publish_template("template-no-channel", "tenant-output", "No channel", (), "Subject", "Body", "en", "mchn_omnichannel_output", True, "owner")

	channel = service.create_channel("channel-main", "tenant-output", "Main", "email", "output-team", "provider://email")
	unhealthy_channel = service.create_channel("channel-unhealthy", "tenant-output", "Unhealthy", "email", "output-team", "provider://email-unhealthy", health="unhealthy")
	policy = service.create_delivery_policy("policy-main", "tenant-output", "Main policy", 100, 10, True, "compliance://default")
	template = service.publish_template("template-main", "tenant-output", "Main template", ("email",), "Subject $id", "Body $id", "en", "mchn_omnichannel_output", True, "content-owner")
	route = service.create_route("route-main", "tenant-output", "Main route", template["id"], channel["id"], (), policy["id"])
	unhealthy_route = service.create_route("route-unhealthy", "tenant-output", "Unhealthy route", template["id"], unhealthy_channel["id"], (), policy["id"])
	rendered = service.render_output("output-main", "tenant-output", route["id"], "recipient:1", {"id": "1"}, "text")

	with pytest.raises(PermissionError, match="recipient_policy_required"):
		service.create_delivery_policy("policy-no-recipients", "tenant-output", "No recipients", 0, 10, True, "compliance://default")
	with pytest.raises(PermissionError, match="throttle_policy_required"):
		service.create_delivery_policy("policy-no-throttle", "tenant-output", "No throttle", 100, 0, True, "compliance://default")
	with pytest.raises(PermissionError, match="compliance_policy_required"):
		service.create_delivery_policy("policy-no-compliance", "tenant-output", "No compliance", 100, 10, True, "")
	with pytest.raises(PermissionError, match="recipient_policy_required"):
		service.render_output("output-no-recipient", "tenant-output", route["id"], "", {"id": "1"}, "text")
	with pytest.raises(PermissionError, match="output_encryption_required"):
		service.render_output("output-sensitive", "tenant-output", route["id"], "recipient:1", {"id": "1"}, "text", sensitive_output=True, output_encrypted=False)
	with pytest.raises(PermissionError, match="channel_unhealthy"):
		service.deliver_batch("batch-unhealthy", "tenant-output", unhealthy_route["id"], "output-user", (rendered["id"],), 1)
	with pytest.raises(PermissionError, match="delivery_actor_required"):
		service.deliver_batch("batch-no-actor", "tenant-output", route["id"], "", (rendered["id"],), 1)
	with pytest.raises(PermissionError, match="rendered_output_required"):
		service.deliver_batch("batch-no-output", "tenant-output", route["id"], "output-user", (), 1)
	with pytest.raises(PermissionError, match="recipient_policy_required"):
		service.deliver_batch("batch-no-recipient", "tenant-output", route["id"], "output-user", (rendered["id"],), 0)
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.deliver_batch("batch-other-stream", "tenant-output", route["id"], "output-user", (rendered["id"],), 1, event_stream="other-stream")
	with pytest.raises(PermissionError, match="large_delivery_review_required"):
		service.deliver_batch("batch-large", "tenant-output", route["id"], "output-user", (rendered["id"],), 20000, delivery_review_recorded=False)
	with pytest.raises(PermissionError, match="delivery_policy_review_required"):
		service.deliver_batch("batch-policy", "tenant-output", route["id"], "output-user", (rendered["id"],), 101, delivery_review_recorded=False)
	with pytest.raises(PermissionError, match="provider_message_required"):
		batch = service.deliver_batch("batch-ok", "tenant-output", route["id"], "output-user", (rendered["id"],), 1)
		service.record_receipt("receipt-no-provider", "tenant-output", batch["id"], rendered["id"], "delivered", "")
	with pytest.raises(PermissionError, match="mchn_agent_runtime_not_supported"):
		service.register_mchn_agent("tenant-output", "Unsupported", "unsupported", "delivery_reviewer", "review")
	with pytest.raises(PermissionError, match="mchn_agent_scope_required"):
		service.register_mchn_agent("tenant-output", "No scope", "codex", "delivery_reviewer", "")

	other_channel = service.create_channel("channel-other", "other-tenant", "Other", "email", "other-team", "provider://other")
	other_policy = service.create_delivery_policy("policy-other", "other-tenant", "Other policy", 100, 10, True, "compliance://other")
	other_template = service.publish_template("template-other", "other-tenant", "Other template", ("email",), "Other", "Other", "en", "mchn_omnichannel_output", True, "other-owner")
	other_route = service.create_route("route-other", "other-tenant", "Other route", other_template["id"], other_channel["id"], (), other_policy["id"])

	with pytest.raises(KeyError, match="rendered_output_not_found"):
		service.deliver_batch("batch-cross-tenant", "other-tenant", other_route["id"], "output-user", (rendered["id"],), 1, delivery_review_recorded=True)


def test_lifecycle_ids_are_tenant_scoped():
	service = MchnService()

	for tenant_id, owner, recipient in (
		("tenant-a", "owner-a", "recipient:a"),
		("tenant-b", "owner-b", "recipient:b"),
	):
		service.create_channel("channel-main", tenant_id, "Main", "email", owner, "provider://email")
		service.publish_template("template-main", tenant_id, "Template", ("email",), "Subject $id", "Body $id", "en", "mchn_omnichannel_output", True, owner)
		service.create_delivery_policy("policy-main", tenant_id, "Policy", 100, 10, True, "compliance://default")
		service.create_route("route-main", tenant_id, "Route", "template-main", "channel-main", (), "policy-main")
		service.render_output("output-main", tenant_id, "route-main", recipient, {"id": tenant_id}, "text")
		service.register_mchn_agent(tenant_id, "Reviewer", "codex", "delivery_reviewer", "review tenant output", agent_id="shared-agent")

	assert service.list_channels("tenant-a")[0]["owner"] == "owner-a"
	assert service.list_channels("tenant-b")[0]["owner"] == "owner-b"
	assert service.list_rendered_outputs("tenant-a")[0]["recipient_ref"] == "recipient:a"
	assert service.list_rendered_outputs("tenant-b")[0]["recipient_ref"] == "recipient:b"
	assert service.list_mchn_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_mchn_agents("tenant-b")[0]["id"] == "shared-agent"


def test_generated_evidence_and_docs_are_current():
	app = _load_module("mchn_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["mchn"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["mchn"]["screens"]["agents"]["route"] == "/mchn/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
