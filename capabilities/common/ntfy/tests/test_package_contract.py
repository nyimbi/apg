"""Notifications and Alerts package runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.ntfy import package_api, view_models
from capabilities.common.ntfy.notification_runtime import NotificationRuntime


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_package_contract_shape_and_entrypoint_are_publishable():
	contract_module = _load_module("ntfy_contract_runtime", PACKAGE_DIR / "capability_contract.py")
	app_module = _load_module("ntfy_app_runtime", PACKAGE_DIR / "app.py")
	contract = contract_module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	self_test = app_module.self_test()
	manifest = app_module.component_manifest()
	model = app_module.semantic_model()

	assert contract["capability"] == "ntfy"
	assert len(contract["ui"]["routes"]) >= 10
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["ntfy"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["ntfy"]["runtime"]["service"] == "notification_runtime.NotificationRuntime"


def test_package_api_executes_notification_lifecycle():
	package_api.RUNTIME = NotificationRuntime()
	package_api.register_channel({"tenant_id": "tenant-api", "channel": "email", "provider": "ses-primary", "owner": "ops", "fallback_channel": "sms"})
	package_api.register_channel({"tenant_id": "tenant-api", "channel": "sms", "provider": "sms-primary", "owner": "ops"})
	package_api.register_preference({"tenant_id": "tenant-api", "recipient_id": "user-1", "addresses": {"email": "user@example.com"}, "preferred_channels": ["email"], "opted_in": True})
	template = package_api.register_template({"tenant_id": "tenant-api", "template_id": "welcome", "name": "Welcome", "owner": "owner", "locale": "en", "channels": ["email"], "content": {"email": "Hello"}, "approved": True})
	delivery = package_api.send_message({"tenant_id": "tenant-api", "template_id": template["id"], "recipient_id": "user-1", "channel": "email", "message_class": "marketing", "idempotency_key": "api-send"})
	campaign = package_api.create_campaign({"tenant_id": "tenant-api", "campaign_id": "launch", "name": "Launch", "owner": "owner", "template_id": template["id"], "audience": ["user-1"], "channels": ["email"]})
	package_api.approve_campaign({"tenant_id": "tenant-api", "campaign_id": campaign["id"], "approved_by": "reviewer"})
	sent = package_api.send_campaign({"tenant_id": "tenant-api", "campaign_id": campaign["id"]})
	state = package_api.notification_state("tenant-api")

	assert delivery["status"] == "delivered"
	assert sent["campaign"]["status"] == "sent"
	assert state["summary"]["delivery_count"] == 1
	assert state["audit_events"]


def test_view_models_match_runtime_state():
	runtime = NotificationRuntime()
	runtime.register_channel("tenant-view", "email", "ses-primary", "ops")
	runtime.register_preference("tenant-view", "user-1", {"email": "user@example.com"}, ["email"], opted_in=True)
	runtime.register_template("tenant-view", "welcome", "Welcome", "owner", "en", ["email"], {"email": "Hello"}, approved=True)
	runtime.send_message("tenant-view", "welcome", "user-1", "email", message_class="marketing")
	runtime.create_campaign("tenant-view", "launch", "Launch", "owner", "welcome", ["user-1"], ["email"])

	dashboard = view_models.dashboard_model(runtime, "tenant-view")
	messages = view_models.message_console_model(runtime, "tenant-view")
	templates = view_models.template_studio_model(runtime, "tenant-view")
	campaigns = view_models.campaign_console_model(runtime, "tenant-view")
	preferences = view_models.preference_center_model(runtime, "tenant-view")
	channels = view_models.channel_health_model(runtime, "tenant-view")
	analytics = view_models.analytics_model(runtime, "tenant-view")
	audit = view_models.audit_model(runtime, "tenant-view")
	settings = view_models.settings_model("tenant-view")

	assert dashboard["summary"]["delivery_count"] == 1
	assert messages["deliveries"][0]["status"] == "delivered"
	assert templates["templates"][0]["approved"] is True
	assert campaigns["drafts"][0]["id"] == "launch"
	assert preferences["opted_in"][0]["recipient_id"] == "user-1"
	assert channels["channels"][0]["channel"] == "email"
	assert analytics["delivery_rate"] == 1.0
	assert audit["audit_events"]
	assert settings["theme"]["name"] == "ntfy_notification_ops"
