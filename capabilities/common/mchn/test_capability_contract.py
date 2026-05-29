"""Regression coverage for the MCHN executable capability contract."""

import pytest

from capabilities.common.mchn import register_capability
from capabilities.common.mchn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.mchn.service import MchnService
from capabilities.common.mchn.views import (
	analytics_model,
	channel_monitor_model,
	dashboard_model,
	policy_model,
	render_console_model,
	route_console_model,
	template_manager_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-output", {"channels": {"fallback_required": False}})

	assert contract["capability"] == "mchn"
	assert contract["configuration"]["tenant_id"] == "tenant-output"
	assert contract["configuration"]["channels"]["fallback_required"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "channels", "rendering", "delivery", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "mchn_omnichannel_output"
	assert contract["ui"]["api_prefix"] == "/mchn/api/v1"


def test_rule_engine_enforces_multichannel_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_channel", "channel_owner_assigned": False, "sensitive_output": True, "output_encrypted": False, "channel_health": "unhealthy", "delivery_requested": True, "recipient_count": 20000, "delivery_review_recorded": False})
	template_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_template", "template_approved": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "channel_requires_owner", "sensitive_output_requires_encryption", "unhealthy_channel_blocks_delivery", "large_delivery_requires_review"}
	assert template_result["matched_rules"] == ["template_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "mchn"
	assert "ntfy" in registration["dependencies"]
	assert registration["ui_components"]["render"] == "/mchn/render"
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

	assert rendered["subject"] == "Invoice INV-1001"
	assert rendered["body"] == "Hello Ada, invoice INV-1001 is ready."
	assert rendered["status"] == "ready"
	assert batch["status"] == "queued"
	assert receipt["delivery_state"] == "delivered"
	assert service.dashboard_summary("tenant-output")["large_batch_count"] == 1
	assert dashboard_model(service, "tenant-output")["summary"]["rendered_output_count"] == 1
	assert render_console_model(service, "tenant-output")["rendered_outputs"][0]["id"] == "output-invoice-1"
	assert template_manager_model(service, "tenant-output")["templates"][0]["id"] == "template-invoice"
	assert route_console_model(service, "tenant-output")["delivery_routes"][0]["id"] == "route-invoice"
	assert channel_monitor_model(service, "tenant-output")["channels"][0]["id"] == "channel-email"
	assert analytics_model(service, "tenant-output")["summary"]["delivery_batch_count"] == 1
	assert policy_model(service, "tenant-output")["policies"][0]["id"] == "policy-standard"
	assert len(service.list_audit_events("tenant-output")) >= 7


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
		service.create_channel(
			channel_id="channel-no-owner",
			tenant_id="tenant-output",
			name="No owner",
			channel_type="email",
			owner="",
			provider_ref="provider://email",
		)

	with pytest.raises(PermissionError, match="template_approval_required"):
		service.publish_template(
			template_id="template-unapproved",
			tenant_id="tenant-output",
			name="Unapproved",
			channel_types=("email",),
			subject_template="Subject",
			body_template="Body",
			locale="en",
			theme_ref="mchn_omnichannel_output",
			approved=False,
			approved_by="",
		)

	channel = service.create_channel(
		channel_id="channel-main",
		tenant_id="tenant-output",
		name="Main",
		channel_type="email",
		owner="output-team",
		provider_ref="provider://email",
	)
	unhealthy_channel = service.create_channel(
		channel_id="channel-unhealthy",
		tenant_id="tenant-output",
		name="Unhealthy",
		channel_type="email",
		owner="output-team",
		provider_ref="provider://email-unhealthy",
		health="unhealthy",
	)
	policy = service.create_delivery_policy(
		policy_id="policy-main",
		tenant_id="tenant-output",
		name="Main policy",
		max_recipients=100,
		throttle_per_minute=10,
		requires_encryption_for_sensitive=True,
		compliance_ref="compliance://default",
	)
	template = service.publish_template(
		template_id="template-main",
		tenant_id="tenant-output",
		name="Main template",
		channel_types=("email",),
		subject_template="Subject $id",
		body_template="Body $id",
		locale="en",
		theme_ref="mchn_omnichannel_output",
		approved=True,
		approved_by="content-owner",
	)
	route = service.create_route(
		route_id="route-main",
		tenant_id="tenant-output",
		name="Main route",
		template_id=template["id"],
		primary_channel_id=channel["id"],
		fallback_channel_ids=(),
		policy_id=policy["id"],
	)
	unhealthy_route = service.create_route(
		route_id="route-unhealthy",
		tenant_id="tenant-output",
		name="Unhealthy route",
		template_id=template["id"],
		primary_channel_id=unhealthy_channel["id"],
		fallback_channel_ids=(),
		policy_id=policy["id"],
	)
	rendered = service.render_output(
		output_id="output-main",
		tenant_id="tenant-output",
		route_id=route["id"],
		recipient_ref="recipient:1",
		variables={"id": "1"},
		output_format="text",
	)

	with pytest.raises(PermissionError, match="output_encryption_required"):
		service.render_output(
			output_id="output-sensitive",
			tenant_id="tenant-output",
			route_id=route["id"],
			recipient_ref="recipient:1",
			variables={"id": "1"},
			output_format="text",
			sensitive_output=True,
			output_encrypted=False,
		)

	with pytest.raises(PermissionError, match="channel_unhealthy"):
		service.deliver_batch(
			batch_id="batch-unhealthy",
			tenant_id="tenant-output",
			route_id=unhealthy_route["id"],
			requested_by="output-user",
			rendered_output_ids=(rendered["id"],),
			recipient_count=1,
		)

	with pytest.raises(PermissionError, match="large_delivery_review_required"):
		service.deliver_batch(
			batch_id="batch-large",
			tenant_id="tenant-output",
			route_id=route["id"],
			requested_by="output-user",
			rendered_output_ids=(rendered["id"],),
			recipient_count=20000,
			delivery_review_recorded=False,
		)

	with pytest.raises(PermissionError, match="delivery_policy_review_required"):
		service.deliver_batch(
			batch_id="batch-policy",
			tenant_id="tenant-output",
			route_id=route["id"],
			requested_by="output-user",
			rendered_output_ids=(rendered["id"],),
			recipient_count=101,
			delivery_review_recorded=False,
		)

	other_channel = service.create_channel(
		channel_id="channel-other",
		tenant_id="other-tenant",
		name="Other",
		channel_type="email",
		owner="other-team",
		provider_ref="provider://other",
	)
	other_policy = service.create_delivery_policy(
		policy_id="policy-other",
		tenant_id="other-tenant",
		name="Other policy",
		max_recipients=100,
		throttle_per_minute=10,
		requires_encryption_for_sensitive=True,
		compliance_ref="compliance://other",
	)
	other_template = service.publish_template(
		template_id="template-other",
		tenant_id="other-tenant",
		name="Other template",
		channel_types=("email",),
		subject_template="Other",
		body_template="Other",
		locale="en",
		theme_ref="mchn_omnichannel_output",
		approved=True,
		approved_by="other-owner",
	)
	other_route = service.create_route(
		route_id="route-other",
		tenant_id="other-tenant",
		name="Other route",
		template_id=other_template["id"],
		primary_channel_id=other_channel["id"],
		fallback_channel_ids=(),
		policy_id=other_policy["id"],
	)

	with pytest.raises(KeyError, match="rendered_output_not_found"):
		service.deliver_batch(
			batch_id="batch-cross-tenant",
			tenant_id="other-tenant",
			route_id=other_route["id"],
			requested_by="output-user",
			rendered_output_ids=(rendered["id"],),
			recipient_count=1,
			delivery_review_recorded=True,
		)
