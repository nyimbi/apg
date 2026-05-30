"""Chat and Messaging package runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.chat import api, views
from capabilities.common.chat.service import ChatService


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
	contract_module = _load_module("chat_contract_runtime", PACKAGE_DIR / "capability_contract.py")
	app_module = _load_module("chat_app_runtime", PACKAGE_DIR / "app.py")
	contract = contract_module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	self_test = app_module.self_test()
	manifest = app_module.component_manifest()
	model = app_module.semantic_model()

	assert contract["capability"] == "chat"
	assert len(contract["ui"]["routes"]) >= 10
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["chat"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["chat"]["runtime"]["service"] == "service.ChatService"


def test_api_helpers_execute_chat_lifecycle():
	api.SERVICE = ChatService()
	room = api.create_room({"id": "api-room", "tenant_id": "tenant-api", "name": "API Room", "owner": "owner", "members": ["owner", "member"], "retention_policy": "retain-30-days"})
	message = api.send_message({"id": "api-message", "tenant_id": "tenant-api", "room_id": room["id"], "sender": "member", "body": "hello"})
	presence = api.update_presence({"tenant_id": "tenant-api", "room_id": room["id"], "user_id": "member", "status": "online", "typing": True})
	state = api.capability_status("tenant-api")

	assert room["status"] == "active"
	assert message["status"] == "delivered"
	assert presence["typing"] is True
	assert state["room_count"] == 1
	assert state["message_count"] == 1


def test_view_models_match_service_state():
	service = ChatService()
	service.create_room("view-room", "tenant-view", "View Room", "owner", ["owner", "member"], "retain-30-days")
	service.send_message("view-message", "tenant-view", "view-room", "member", "message", attachments=["report.txt"])
	service.update_presence("tenant-view", "member", "online", room_id="view-room", typing=True)

	dashboard = views.dashboard_model(service, "tenant-view")
	rooms = views.room_manager_model(service, "tenant-view")
	messages = views.message_console_model(service, "tenant-view", "view-room")
	moderation = views.moderation_queue_model(service, "tenant-view")
	agents = views.agent_participant_model("tenant-view")
	analytics = views.analytics_model(service, "tenant-view")
	audit = views.audit_model(service, "tenant-view")
	settings = views.settings_model("tenant-view")

	assert dashboard["summary"]["room_count"] == 1
	assert rooms["active"][0]["id"] == "view-room"
	assert messages["messages"][0]["id"] == "view-message"
	assert moderation["pending"] == []
	assert agents["enabled"] is True
	assert analytics["attachment_rate"] == 1.0
	assert audit["audit_events"]
	assert settings["theme"]["name"] == "chat_team_messaging"
