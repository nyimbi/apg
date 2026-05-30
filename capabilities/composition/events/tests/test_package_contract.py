"""Event capability package contract tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_composition_events"


def _load_module(name: str):
	if PACKAGE_NAME not in sys.modules:
		package = types.ModuleType(PACKAGE_NAME)
		package.__path__ = [str(PACKAGE_DIR)]
		sys.modules[PACKAGE_NAME] = package
	spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.{name}", PACKAGE_DIR / f"{name}.py")
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_and_streaming_are_valid():
	module = _load_module("capability_contract")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "composition_events"
	assert "event_agents" in contract["provides"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert any(route["path"] == "/composition-events/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_publish():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "publish_event",
		"source_capability_present": True,
		"correlation_present": True,
		"event_stream": "other",
	})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "publish_requires_bytewax" in bad_stream["matched_rules"]


def test_service_stream_schema_publish_subscription_processor_lifecycle():
	service_module = _load_module("service")
	service = service_module.CompositionEventsService()

	schema = service.register_schema(
		"order-created",
		"tenant-test",
		"OrderCreated",
		"1.0.0",
		{"type": "object", "required": ["order_id"]},
	)
	stream = service.create_stream(
		"orders",
		"tenant-test",
		"Orders",
		"owner-1",
		"orders",
		"7d",
		"tenant_id",
		pii_stream=True,
		schema_id=schema["id"],
	)
	event = asyncio.run(service.publish_event(
		stream["id"],
		"tenant-test",
		"order.created",
		{"order_id": "A-1"},
		"orders",
		"corr-1",
		"tenant-test",
	))
	subscription = service.create_subscription(
		"order-workers",
		"tenant-test",
		stream["id"],
		"consumer-owner",
		"at_least_once",
		retry_enabled=True,
		dead_letter_stream_id=stream["id"],
	)
	processor = service.register_processor(
		"orders-processor",
		"tenant-test",
		"Orders Processor",
		stream["id"],
		stateful=True,
		checkpoint_configured=True,
		reviewed_by="processor-reviewer",
	)

	assert event["bytewax"]["stream"] == stream["bytewax_stream"]
	assert subscription["status"] == "active"
	assert processor["processor_runtime"] == "bytewax"
	assert service.dashboard_summary("tenant-test")["audit_event_count"] >= 5


def test_service_enforces_event_guardrails():
	service_module = _load_module("service")
	service = service_module.CompositionEventsService()

	try:
		service.create_stream("bad", "", "Bad", "owner", "bad", "7d", "tenant_id")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	try:
		service.create_stream("pii", "tenant-test", "PII", "owner", "crm", "7d", "tenant_id", pii_stream=True)
	except PermissionError as exc:
		assert "pii_stream_requires_schema" in str(exc)
	else:
		raise AssertionError("expected schema guardrail")

	stream = service.create_stream("orders", "tenant-test", "Orders", "owner", "orders", "7d", "tenant_id")
	try:
		asyncio.run(service.publish_event(stream["id"], "tenant-test", "order.created", {}, "", "corr-1", "tenant-test"))
	except PermissionError as exc:
		assert "publish_requires_source_capability" in str(exc)
	else:
		raise AssertionError("expected source-capability guardrail")

	try:
		service.create_subscription("retry", "tenant-test", stream["id"], "owner", "at_least_once", retry_enabled=True)
	except PermissionError as exc:
		assert "retry_subscription_requires_dead_letter" in str(exc)
	else:
		raise AssertionError("expected dead-letter guardrail")

	try:
		service.register_processor("bad-processor", "tenant-test", "Bad", stream["id"], stateful=False, checkpoint_configured=False)
	except PermissionError as exc:
		assert "processor_requires_checkpoint" in str(exc)
	else:
		raise AssertionError("expected checkpoint guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.CompositionEventsService()
	agent = service.register_event_agent("tenant-test", "Stream Review", "codex", "stream_architect", "Review streams.")
	agent_result = service.validate_agent_event_action("tenant-test", agent["id"], "recommend_stream", True, True)
	batch = service.validate_batch_publish("tenant-test", 2)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_record = api_module.create_record({"id": "api-stream", "tenant_id": "tenant-api"})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["event_agent_count"] == 1
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("event_stream_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["capabilities"]["composition_events"]["streaming"]["processor"] == "bytewax"
