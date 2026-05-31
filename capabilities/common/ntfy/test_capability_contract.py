"""Regression coverage for the NTFY executable capability contract."""

import pytest

from capabilities.common.ntfy import register_capability
from capabilities.common.ntfy import package_api
from capabilities.common.ntfy import view_models
from capabilities.common.ntfy.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.ntfy.notification_runtime import NotificationRuntime


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-notify", {"delivery": {"max_batch_size": 1000}})

	assert contract["capability"] == "ntfy"
	assert contract["configuration"]["tenant_id"] == "tenant-notify"
	assert contract["configuration"]["delivery"]["max_batch_size"] == 1000
	assert set(contract["configuration_schema"]["required"]) >= {
		"tenant_id",
		"channels",
		"delivery",
		"preferences",
		"templates",
		"campaigns",
		"security",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	}
	assert len(contract["rule_engine"]["rules"]) >= 40
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "messages", "templates", "campaigns", "preferences", "suppression", "channels", "analytics", "agents", "lifecycle", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/ntfy/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "channel_matrix" in contract["theme"]["components"]
	assert "notification_agent_roster" in contract["theme"]["components"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["agent_adapter"] == "aicr_provider_neutral_notification_agent_adapter"
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "notification_agent_batch" in contract["streaming"]["required_operations"]


def test_rule_engine_enforces_notification_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "send_message",
		"template_present": False,
		"template_approved": False,
		"message_class": "marketing",
		"recipient_opted_in": False,
		"recipient_unsubscribed": True,
		"sensitive_payload": True,
		"payload_encrypted": False,
		"provider_health": "unhealthy",
		"delivery_requested": True,
		"channel_enabled": False,
		"channel": "webhook",
		"webhook_signature_present": False,
		"event_bus_present": False,
		"audit_event_recorded": False,
		"duplicate_idempotency_key": True,
		"recipient_count": 7000,
		"batch_review_recorded": False,
	})
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_notification_mutation", "event_stream": "kafka"})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_notification_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
	})
	privileged_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_notification_agent",
		"agent_runtime_supported": True,
		"agent_role_supported": True,
		"scope_present": True,
		"owner_present": True,
		"purpose_present": True,
		"contribution_disclosed": True,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "validate_ntfy_lifecycle_batch", "event_stream": "kafka", "mutation_count": 1})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"send_requires_template",
		"approved_template_required",
		"marketing_requires_opt_in",
		"unsubscribe_blocks_marketing",
		"sensitive_payload_requires_encryption",
		"provider_health_required",
		"channel_enabled_required",
		"webhook_requires_signature",
		"delivery_requires_event_bus",
		"delivery_requires_audit",
		"duplicate_idempotency_key_blocked",
		"large_batch_requires_review",
	}
	assert stream_result["decision"] == "deny"
	assert "batch_notification_mutation_requires_bytewax" in stream_result["matched_rules"]
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"notification_agent_runtime_supported",
		"notification_agent_role_supported",
		"notification_agent_requires_scope",
		"notification_agent_requires_owner",
		"notification_agent_requires_purpose",
		"notification_agent_requires_contribution_disclosure",
	}
	assert privileged_result["decision"] == "require_review"
	assert privileged_result["matched_rules"] == ["notification_agent_privileged_role_requires_human_approval"]
	assert lifecycle_result["decision"] == "deny"
	assert lifecycle_result["matched_rules"] == ["bytewax_ntfy_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "ntfy"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "ntfy_notification_ops"
	assert registration["ui_components"]["campaigns"] == "/ntfy/campaigns"
	assert registration["ui_components"]["audit"] == "/ntfy/audit"
	assert registration["ui_components"]["agents"] == "/ntfy/agents"
	assert "mqeb" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["lifecycle_stream"] == "ntfy.lifecycle"
	assert "ntfy:send" in registration["permissions"]
	assert "ntfy:audit" in registration["permissions"]


def _build_runtime() -> NotificationRuntime:
	runtime = NotificationRuntime()
	runtime.register_channel("tenant-notify", "email", "ses-primary", "ops", fallback_channel="sms")
	runtime.register_channel("tenant-notify", "sms", "twilio-primary", "ops")
	runtime.register_preference("tenant-notify", "user-1", {"email": "user@example.com", "sms": "+15551234567"}, ["email", "sms"], opted_in=True)
	runtime.register_template("tenant-notify", "welcome", "Welcome", "marketing-owner", "en", ["email", "sms"], {"email": "Hello", "sms": "Hello"}, approved=True)
	return runtime


def test_runtime_executes_message_campaign_and_view_lifecycle():
	runtime = _build_runtime()
	delivery = runtime.send_message("tenant-notify", "welcome", "user-1", "email", message_class="marketing", idempotency_key="welcome:user-1")
	campaign = runtime.create_campaign("tenant-notify", "spring", "Spring Launch", "marketing-owner", "welcome", ["user-1"], ["email"])
	approved = runtime.approve_campaign("tenant-notify", "spring", "reviewer")
	sent = runtime.send_campaign("tenant-notify", "spring")
	agent = runtime.register_notification_agent("agent-steward", "tenant-notify", "Notification Steward", "codex", "notification_steward", "campaign:spring", "marketing-owner", "review campaign delivery", human_approval_required=True)
	batch = runtime.validate_ntfy_lifecycle_batch("tenant-notify", "bytewax", 2, "notification_agent_batch", "batch-agent")
	summary = runtime.dashboard_summary("tenant-notify")

	assert delivery["status"] == "delivered"
	assert campaign["status"] == "draft"
	assert approved["approved"] is True
	assert sent["campaign"]["status"] == "sent"
	assert agent["runtime"] == "codex"
	assert agent["status"] == "active"
	assert batch["status"] == "accepted"
	assert summary["delivery_count"] == 1
	assert summary["notification_agent_count"] == 1
	assert summary["lifecycle_batch_count"] == 1
	assert view_models.message_console_model(runtime, "tenant-notify")["deliveries"][0]["id"] == delivery["id"]
	assert view_models.campaign_console_model(runtime, "tenant-notify")["sent"][0]["id"] == "spring"
	assert view_models.notification_agent_roster_model(runtime, "tenant-notify")["active"][0]["id"] == "agent-steward"
	assert view_models.lifecycle_batch_model(runtime, "tenant-notify")["accepted"][0]["id"] == "batch-agent"
	assert view_models.audit_model(runtime, "tenant-notify")["audit_events"]


def test_runtime_enforces_consent_encryption_provider_and_idempotency_guardrails():
	runtime = _build_runtime()
	runtime.register_preference("tenant-notify", "user-2", {"email": "two@example.com"}, ["email"], opted_in=False)
	runtime.register_channel("tenant-notify", "webhook", "webhook-provider", "ops", healthy=True)

	with pytest.raises(PermissionError, match="recipient_opt_in_required"):
		runtime.send_message("tenant-notify", "welcome", "user-2", "email", message_class="marketing")
	with pytest.raises(PermissionError, match="payload_encryption_required"):
		runtime.send_message("tenant-notify", "welcome", "user-1", "email", sensitive_payload=True)
	with pytest.raises(PermissionError, match="webhook_signature_required"):
		runtime.send_message("tenant-notify", "welcome", "user-1", "webhook", webhook_signature_present=False)

	first = runtime.send_message("tenant-notify", "welcome", "user-1", "email", message_class="marketing", idempotency_key="dup-key")
	with pytest.raises(PermissionError, match="duplicate_notification_send"):
		runtime.send_message("tenant-notify", "welcome", "user-1", "email", message_class="marketing", idempotency_key="dup-key")

	unapproved_campaign = runtime.create_campaign("tenant-notify", "no-consent", "No Consent", "marketing-owner", "welcome", ["user-2"], ["email"])
	runtime.approve_campaign("tenant-notify", unapproved_campaign["id"], "reviewer")
	with pytest.raises(PermissionError, match="recipient_opt_in_required"):
		runtime.send_campaign("tenant-notify", unapproved_campaign["id"])
	bad_channel_campaign = runtime.create_campaign("tenant-notify", "bad-channel", "Bad Channel", "marketing-owner", "welcome", ["user-1"], ["push"])
	runtime.approve_campaign("tenant-notify", bad_channel_campaign["id"], "reviewer")
	with pytest.raises(PermissionError, match="channel_not_enabled"):
		runtime.send_campaign("tenant-notify", bad_channel_campaign["id"])

	assert first["status"] == "delivered"


def test_runtime_routes_large_campaigns_and_quiet_hours_to_review():
	runtime = _build_runtime()
	audience = [f"user-{index}" for index in range(6001)]
	campaign = runtime.create_campaign("tenant-notify", "large", "Large Campaign", "marketing-owner", "welcome", audience, ["email"], message_class="transactional")
	runtime.approve_campaign("tenant-notify", campaign["id"], "reviewer")
	large_result = runtime.send_campaign("tenant-notify", campaign["id"])
	quiet = runtime.send_message("tenant-notify", "welcome", "user-1", "email", message_class="marketing", quiet_hours_active=True)

	assert large_result["campaign"]["status"] == "review_required"
	assert "review_batch" in large_result["required_actions"]
	assert quiet["status"] == "review_required"
	assert "review_quiet_hours_override" in quiet["required_actions"]


def test_tenant_local_records_do_not_collide():
	runtime = NotificationRuntime()
	runtime.register_channel("tenant-alpha", "email", "alpha-provider", "alpha-owner")
	runtime.register_channel("tenant-beta", "email", "beta-provider", "beta-owner")
	runtime.register_preference("tenant-alpha", "shared-user", {"email": "alpha@example.com"}, ["email"], opted_in=True)
	runtime.register_preference("tenant-beta", "shared-user", {"email": "beta@example.com"}, ["email"], opted_in=True)
	runtime.register_template("tenant-alpha", "shared-template", "Shared", "alpha-owner", "en", ["email"], {"email": "Hello"}, approved=True)
	runtime.register_template("tenant-beta", "shared-template", "Shared", "beta-owner", "en", ["email"], {"email": "Hello"}, approved=True)
	alpha_delivery = runtime.send_message("tenant-alpha", "shared-template", "shared-user", "email", idempotency_key="same-key")
	beta_delivery = runtime.send_message("tenant-beta", "shared-template", "shared-user", "email", idempotency_key="same-key")

	assert runtime.list_channels("tenant-alpha")[0]["provider"] == "alpha-provider"
	assert runtime.list_channels("tenant-beta")[0]["provider"] == "beta-provider"
	assert runtime.list_preferences("tenant-alpha")[0]["addresses"]["email"] == "alpha@example.com"
	assert runtime.list_preferences("tenant-beta")[0]["addresses"]["email"] == "beta@example.com"
	assert alpha_delivery["tenant_id"] == "tenant-alpha"
	assert beta_delivery["tenant_id"] == "tenant-beta"


def test_runtime_and_api_enforce_notification_agent_and_lifecycle_guardrails():
	runtime = NotificationRuntime()

	with pytest.raises(PermissionError, match="unsupported_notification_agent_runtime"):
		runtime.register_notification_agent("agent-unsupported", "tenant-notify", "Unsupported", "kafka_agent", "channel_reviewer", "channel:*", "ops", "review channel health")

	with pytest.raises(PermissionError, match="notification_agent_contribution_disclosure_required"):
		runtime.register_notification_agent("agent-undisclosed", "tenant-notify", "Undisclosed", "codex", "channel_reviewer", "channel:*", "ops", "review channels", contribution_disclosed=False)

	pending = runtime.register_notification_agent(
		"agent-campaign-reviewer",
		"tenant-notify",
		"Campaign Reviewer",
		"claude_code",
		"campaign_reviewer",
		"campaign:*",
		"marketing-owner",
		"review campaign audiences",
	)

	with pytest.raises(ValueError, match="ntfy_lifecycle_batch_empty"):
		runtime.validate_ntfy_lifecycle_batch("tenant-notify", "bytewax", 0, "campaign_batch")
	with pytest.raises(ValueError, match="unsupported_ntfy_lifecycle_operation"):
		runtime.validate_ntfy_lifecycle_batch("tenant-notify", "bytewax", 1, "unknown_batch")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		runtime.validate_ntfy_lifecycle_batch("tenant-notify", "kafka", 1, "campaign_batch")

	package_api.RUNTIME = NotificationRuntime()
	api_agent = package_api.register_notification_agent({
		"id": "agent-api",
		"tenant_id": "tenant-api",
		"name": "API Agent",
		"runtime": "opencode",
		"role": "template_reviewer",
		"scope": "template:*",
		"owner": "template-owner",
		"purpose": "review template drift",
	})
	api_batch = package_api.validate_lifecycle_batch({"tenant_id": "tenant-api", "event_stream": "bytewax", "mutation_count": 2, "operation": "template_batch", "batch_id": "batch-api"})
	status = package_api.capability_status("tenant-api")
	state = package_api.notification_state("tenant-api")

	assert pending["status"] == "pending_review"
	assert pending["human_approval_required"] is False
	assert api_agent["runtime"] == "opencode"
	assert api_batch["accepted"] is True
	assert status["agents"]["first_class"] is True
	assert status["lifecycle_batch_count"] == 1
	assert state["notification_agents"][0]["id"] == "agent-api"
	assert state["lifecycle_batches"][0]["id"] == "batch-api"
