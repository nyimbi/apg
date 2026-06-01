"""MQEB package contract and dependency-light runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.mqeb import api, view_models
from capabilities.common.mqeb.service import MqebService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("package_contract_mqeb", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "mqeb"
	assert len(contract["ui"]["routes"]) >= 14
	assert len(contract["rule_engine"]["rules"]) >= 22
	assert contract["configuration"]["operation_governance"]["bytewax_first_runtime"] is True
	assert contract["configuration"]["operation_governance"]["broker_core_dependency_allowed"] is False
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["engine"] == "bytewax"
	assert "review_evidence" in contract["provides"]
	assert contract["review_evidence"]["pending_queues"]


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_mqeb", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()
	capability = model["capabilities"]["mqeb"]

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert len(capability["ui"]["routes"]) >= 14
	assert capability["adapters"]["preferred_stream_runtime"] == "bytewax"
	assert capability["approvals"]["priority_quota"] == "PriorityQuotaExceptionRecord"
	assert capability["approvals"]["event_agent"] == "MqebAgentRecord"
	assert capability["agents"]["mqeb_agent_contract"]["first_class"] is True
	assert capability["streaming"]["engine"] == "bytewax"
	assert "review_evidence" in capability["provides"]
	assert capability["review_evidence"]["pending_queues"]


def test_event_fabric_lifecycle_records_publish_delivery_replay_and_audit_state():
	service = MqebService()

	topic = service.create_topic(
		tenant_id="tenant-a",
		topic_id="invoice-events",
		name="Invoice Events",
		owner="finance-platform",
		classification="regulated",
		retention_days=30,
		delivery_mode="exactly_once",
		encrypted=True,
		schema_ref="schema://invoice-events/v1",
		dead_letter_topic="invoice-events.dlq",
	)
	message = service.publish_message(
		tenant_id="tenant-a",
		message_id="invoice-1",
		topic_id=topic["id"],
		producer="erp-billing",
		delivery_mode="exactly_once",
		idempotency_key="invoice-1",
		payload_size=512,
		priority_messages_per_minute=100,
	)
	subscription = service.create_subscription(
		tenant_id="tenant-a",
		subscription_id="warehouse",
		name="Warehouse Projection",
		topic_pattern="invoice-events",
		consumer="warehouse-sync",
		delivery_mode="exactly_once",
		protocol="bytewax",
		dead_letter_topic="invoice-events.dlq",
	)
	delivered = service.record_delivery_attempt("tenant-a", "delivery-1", message["id"], subscription["id"], "delivered")
	service.pause_subscription("tenant-a", subscription["id"], "ops", "Deployment window.")
	service.resume_subscription("tenant-a", subscription["id"], "ops", "Deployment complete.")
	replay = service.request_replay(
		tenant_id="tenant-a",
		replay_id="replay-1",
		topic_id=topic["id"],
		requested_by="ops",
		reason="Recover downstream projection.",
		range_start="2026-05-30T00:00:00Z",
		range_end="2026-05-30T01:00:00Z",
	)
	replay_approved = service.decide_replay("tenant-a", replay["id"], "auditor", "approved", "Evidence attached.")
	summary = service.dashboard_summary("tenant-a")

	assert message["status"] == "published"
	assert message["policy_decision"] == "allow"
	assert delivered["status"] == "delivered"
	assert delivered["policy_decision"] == "allow"
	assert replay_approved["status"] == "approved"
	assert replay_approved["policy_decision"] == "allow"
	assert summary["topic_count"] == 1
	assert summary["message_count"] == 1
	assert summary["subscription_count"] == 1
	assert summary["pending_review_count"] == 0
	assert {event["event_type"] for event in service.list_audit_events("tenant-a")} >= {
		"topic_created",
		"message_published",
		"subscription_created",
		"delivery_delivered",
		"subscription_paused",
		"subscription_resumed",
		"replay_requested",
		"replay_decided",
	}


def test_event_agents_and_bytewax_lifecycle_batches_are_first_class_state():
	service = MqebService()

	with pytest.raises(PermissionError, match="unsupported_event_agent_runtime"):
		service.register_event_agent(
			tenant_id="tenant-a",
			agent_id="bad-runtime",
			name="Bad Runtime",
			runtime="custom-runtime",
			role="replay-reviewer",
			scope="replay review",
			owner="ops",
			purpose="review replay requests",
			contribution_disclosed=True,
			human_approval_required=True,
		)
	review_agent = service.register_event_agent(
		tenant_id="tenant-a",
		agent_id="privileged",
		name="Privileged",
		runtime="codex",
		role="bytewax-topology-reviewer",
		scope="bytewax topology review",
		owner="platform",
		purpose="review stream topology changes",
		contribution_disclosed=True,
		human_approval_required=False,
	)
	assert review_agent["status"] == "pending_review"
	assert review_agent["policy_decision"] == "require_review"
	assert review_agent["review_reasons"] == ["event_agent_human_approval_required"]
	agent = service.register_event_agent(
		tenant_id="tenant-a",
		agent_id="replay-agent",
		name="Replay Agent",
		runtime="claude-code",
		role="replay-reviewer",
		scope="bounded replay review",
		owner="platform",
		purpose="review replay approvals",
		contribution_disclosed=True,
		human_approval_required=True,
	)
	batch = service.validate_event_lifecycle_batch("tenant-a", "ByteWax", 3)
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_event_lifecycle_batch("tenant-a", "custom-broker", 1)
	denied_batch = [
		item for item in service.list_lifecycle_batches("tenant-a")
		if item["status"] == "denied"
	][0]
	with pytest.raises(ValueError, match="event_lifecycle_batch_empty"):
		service.validate_event_lifecycle_batch("tenant-a", "bytewax", 0)
	summary = service.dashboard_summary("tenant-a")

	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "replay_reviewer"
	assert agent["human_approval_required"] is True
	assert batch["event_stream"] == "bytewax"
	assert batch["required_processor"] == "bytewax"
	assert batch["accepted"] is True
	assert batch["policy_decision"] == "allow"
	assert denied_batch["policy_decision"] == "deny"
	assert denied_batch["review_reasons"] == ["bytewax_event_stream_required"]
	assert summary["event_agent_count"] == 2
	assert summary["pending_event_agent_review_count"] == 1
	assert summary["lifecycle_batch_count"] == 2
	assert summary["denied_lifecycle_batch_count"] == 1
	assert summary["pending_review_count"] == 1
	assert {event["event_type"] for event in service.list_audit_events("tenant-a")} >= {
		"event_agent_registered",
		"event_lifecycle_batch_accepted",
		"event_lifecycle_batch_denied",
	}


def test_mqeb_guardrails_fail_closed():
	service = MqebService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_topic("", "events", "Events", "owner")
	with pytest.raises(ValueError, match="topic_owner_required"):
		service.create_topic("tenant-a", "events", "Events", "")
	with pytest.raises(ValueError, match="unsupported_topic_classification"):
		service.create_topic("tenant-a", "bad", "Bad", "owner", classification="secret")
	with pytest.raises(ValueError, match="topic_retention_days_required"):
		service.create_topic("tenant-a", "zero-retention", "Zero", "owner", retention_days=0)

	restricted = service.create_topic(
		"tenant-a",
		"restricted",
		"Restricted",
		"owner",
		classification="restricted",
		encrypted=False,
		schema_ref="schema://restricted/v1",
	)
	regulated = service.create_topic(
		"tenant-a",
		"regulated",
		"Regulated",
		"owner",
		classification="regulated",
		encrypted=True,
		schema_ref="",
	)
	regulated_unencrypted = service.create_topic(
		"tenant-a",
		"regulated-unencrypted",
		"Regulated Unencrypted",
		"owner",
		classification="regulated",
		encrypted=False,
		schema_ref="schema://regulated/v1",
	)
	exactly_once = service.create_topic(
		"tenant-a",
		"exactly-once",
		"Exactly Once",
		"owner",
		classification="internal",
		encrypted=True,
		dead_letter_topic="exactly-once.dlq",
		delivery_mode="exactly_once",
	)
	disabled = service.create_topic("tenant-a", "disabled", "Disabled", "owner", status="disabled")

	assert service.publish_message("tenant-a", "msg-restricted", restricted["id"], "producer", encrypted=False)["status"] == "denied"
	assert service.publish_message("tenant-a", "msg-regulated", regulated["id"], "producer")["status"] == "denied"
	assert service.publish_message("tenant-a", "msg-regulated-unencrypted", regulated_unencrypted["id"], "producer")["status"] == "denied"
	assert service.publish_message("tenant-a", "msg-exactly", exactly_once["id"], "producer", delivery_mode="exactly_once")["status"] == "denied"
	assert service.publish_message("tenant-a", "msg-disabled", disabled["id"], "producer")["status"] == "denied"
	assert service.publish_message("tenant-a", "msg-cross", exactly_once["id"], "producer", cross_tenant_publish=True)["status"] == "denied"

	quota_review = service.publish_message(
		"tenant-a",
		"msg-quota",
		exactly_once["id"],
		"producer",
		delivery_mode="exactly_once",
		idempotency_key="msg-quota",
		priority_messages_per_minute=20000,
	)
	assert quota_review["status"] == "review_required"
	assert quota_review["policy_decision"] == "require_review"
	with pytest.raises(ValueError, match="priority_exception_reason_required"):
		service.request_priority_exception("tenant-a", "quota", exactly_once["id"], "owner", "")
	exception = service.request_priority_exception("tenant-a", "quota", exactly_once["id"], "owner", "Seasonal peak.")
	with pytest.raises(PermissionError, match="independent_priority_exception_reviewer_required"):
		service.decide_priority_exception("tenant-a", exception["id"], " OWNER ", "approved", "Self review.")
	with pytest.raises(ValueError, match="review_notes_required"):
		service.decide_priority_exception("tenant-a", exception["id"], "reviewer", "approved", "")
	approved = service.decide_priority_exception("tenant-a", exception["id"], "reviewer", "approved", "Approved for migration.")
	assert approved["status"] == "approved"
	assert approved["policy_decision"] == "allow"
	allowed = service.publish_message(
		"tenant-a",
		"msg-quota-approved",
		exactly_once["id"],
		"producer",
		delivery_mode="exactly_once",
		idempotency_key="msg-quota-approved",
		priority_messages_per_minute=20000,
	)
	assert allowed["status"] == "published"

	with pytest.raises(PermissionError, match="replay_range_required"):
		service.request_replay("tenant-a", "replay", exactly_once["id"], "owner", "Need replay.", "", "")
	with pytest.raises(PermissionError, match="replay_reason_required"):
		service.request_replay("tenant-a", "replay", exactly_once["id"], "owner", "", "start", "end")
	replay = service.request_replay("tenant-a", "replay", exactly_once["id"], "owner", "Need replay.", "start", "end")
	with pytest.raises(PermissionError, match="independent_replay_reviewer_required"):
		service.decide_replay("tenant-a", replay["id"], " OWNER ", "approved", "Self review.")

	subscription = service.create_subscription("tenant-a", "sub", "Sub", "exactly-once", "consumer")
	service.pause_subscription("tenant-a", subscription["id"], "ops", "Testing pause.")
	with pytest.raises(PermissionError, match="subscription_paused"):
		service.record_delivery_attempt("tenant-a", "paused-delivery", allowed["id"], subscription["id"], "delivered")
	service.resume_subscription("tenant-a", subscription["id"], "ops", "Resume evidence.")
	with pytest.raises(ValueError, match="delivery_failure_reason_required"):
		service.record_delivery_attempt("tenant-a", "retry", allowed["id"], subscription["id"], "retry")
	dead_letter = service.record_delivery_attempt("tenant-a", "dead-letter", allowed["id"], subscription["id"], "dead_letter", 3, "Consumer failure.")
	assert dead_letter["status"] == "dead_letter"


def test_api_and_view_models_expose_event_fabric_surfaces():
	local_service = MqebService()
	api.SERVICE = local_service

	with pytest.raises(PermissionError, match="tenant_context_required"):
		api.create_topic_record({"id": "missing-tenant", "name": "Missing", "owner": "ops"})
	pending_agent = api.register_event_agent({
		"tenant_id": "tenant-b",
		"id": "privileged-string-bool",
		"name": "Privileged String Bool",
		"runtime": "codex",
		"role": "replay-reviewer",
		"scope": "replay review",
		"owner": "platform",
		"purpose": "review replay decisions",
		"contribution_disclosed": "true",
		"human_approval_required": "false",
	})
	assert pending_agent["status"] == "pending_review"
	topic = api.create_topic_record({
		"tenant_id": "tenant-b",
		"id": "orders",
		"name": "Orders",
		"owner": "commerce",
		"classification": "internal",
		"encrypted": True,
		"dead_letter_topic": "orders.dlq",
	})
	message = api.publish_message_record({
		"tenant_id": "tenant-b",
		"id": "order-1",
		"topic_id": topic["id"],
		"producer": "order-service",
		"payload_size": 256,
	})
	subscription = api.create_subscription_record({
		"tenant_id": "tenant-b",
		"id": "fulfillment",
		"name": "Fulfillment",
		"topic_pattern": "orders",
		"consumer": "fulfillment-service",
		"protocol": "bytewax",
	})
	api.record_delivery_attempt({
		"tenant_id": "tenant-b",
		"id": "delivery-1",
		"message_id": message["id"],
		"subscription_id": subscription["id"],
		"outcome": "delivered",
	})
	api.register_event_agent({
		"tenant_id": "tenant-b",
		"id": "routing-agent",
		"name": "Routing Agent",
		"runtime": "opencode",
		"role": "routing-reviewer",
		"scope": "routing changes",
		"owner": "platform",
		"purpose": "review routing rule changes",
		"contribution_disclosed": True,
	})
	api.validate_event_lifecycle_batch({
		"tenant_id": "tenant-b",
		"event_stream": "bytewax",
		"mutation_count": 2,
	})

	status = api.capability_status("tenant-b")
	fabric = api.list_event_fabric("tenant-b")
	dashboard = view_models.dashboard_model(tenant_id="tenant-b")
	topics = view_models.topic_inventory_model(tenant_id="tenant-b")
	publish = view_models.publish_workbench_model(tenant_id="tenant-b")
	subscriptions = view_models.subscription_model(tenant_id="tenant-b")
	delivery = view_models.delivery_model(tenant_id="tenant-b")
	quota = view_models.quota_exception_queue_model(tenant_id="tenant-b")
	replay = view_models.replay_console_model(tenant_id="tenant-b")
	agents = view_models.event_agent_roster_model(tenant_id="tenant-b")
	bytewax = view_models.bytewax_bridge_model(tenant_id="tenant-b")
	audit = view_models.audit_timeline_model(tenant_id="tenant-b")
	settings = view_models.settings_model("tenant-b")

	assert status["topic_count"] == 1
	assert status["event_agent_count"] == 2
	assert status["pending_event_agent_review_count"] == 1
	assert status["pending_review_count"] == 1
	assert fabric["summary"]["message_count"] == 1
	assert len(fabric["event_agents"]) == 2
	assert fabric["pending_reviews"][0]["status"] == "pending_review"
	assert dashboard["summary"]["subscription_count"] == 1
	assert dashboard["review_evidence"]["deny_behavior"] == "Denied MQEB lifecycle batches persist evidence before PermissionError"
	assert any(agent["role"] == "routing_reviewer" for agent in dashboard["event_agents"])
	assert topics["classifications"][-1] == "regulated"
	assert publish["messages"][0]["status"] == "published"
	assert subscriptions["protocols"][0] == "bytewax"
	assert delivery["delivery_attempts"][0]["outcome"] == "delivered"
	assert quota["pending"] == []
	assert replay["pending"] == []
	assert agents["supported_runtimes"][0] == "codex"
	assert agents["pending_reviews"][0]["status"] == "pending_review"
	assert any(agent["name"] == "Routing Agent" for agent in agents["event_agents"])
	assert bytewax["preferred_runtime"] == "bytewax"
	assert bytewax["lifecycle_batches"][0]["accepted"] is True
	assert audit["events"]
	assert settings["configuration"]["operation_governance"]["bytewax_first_runtime"] is True
	assert settings["agents"]["first_class"] is True
	assert settings["streaming"]["engine"] == "bytewax"
	assert settings["review_evidence"]["pending_queues"]
