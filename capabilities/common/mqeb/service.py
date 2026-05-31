#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Core Service
Main service implementation with APG integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, AsyncGenerator
from dataclasses import dataclass, field, asdict
import uuid
from uuid_extensions import uuid7str

from .models import (
	MQMessage, TopicConfiguration, Subscription, MessageEvent, BrokerNode,
	MessagePriority, DeliveryMode, ProtocolType, MessageStatus, RetryPolicy
)


TOPIC_CLASSIFICATIONS = {"public", "internal", "restricted", "regulated"}
TOPIC_STATUSES = {"active", "disabled", "deprecated"}
SUBSCRIPTION_STATUSES = {"active", "paused", "disabled"}
DELIVERY_OUTCOMES = {"delivered", "retry", "dead_letter"}
DELIVERY_MODES = {"at_most_once", "at_least_once", "exactly_once"}
PROTOCOLS = {"http_rest", "websocket", "mqtt", "amqp", "bytewax", "grpc"}


def _utc_now() -> str:
	return datetime.utcnow().isoformat() + "Z"


def _stable_id(prefix: str, *parts: object) -> str:
	payload = "|".join(str(part) for part in parts)
	return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _normalize_choice(value: str, allowed: set[str], default: str, error_prefix: str) -> str:
	normalized = str(value or default).strip().lower()
	if normalized not in allowed:
		raise ValueError(f"{error_prefix}:{value}")
	return normalized


def _required_actions(result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in result.get("actions", [])
		if action.get("required_action")
	]


@dataclass(slots=True)
class TopicRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	classification: str
	retention_days: int
	delivery_mode: str
	encrypted: bool
	schema_ref: str
	dead_letter_topic: str
	status: str = "active"
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class MessageRecord:
	id: str
	tenant_id: str
	topic_id: str
	producer: str
	priority: str
	delivery_mode: str
	status: str
	decision: str
	matched_rules: list[str]
	required_actions: list[str]
	idempotency_key: str = ""
	schema_ref: str = ""
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class SubscriptionRecord:
	id: str
	tenant_id: str
	name: str
	topic_pattern: str
	consumer: str
	delivery_mode: str
	protocol: str
	dead_letter_topic: str
	status: str = "active"
	lag_messages: int = 0
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class DeliveryAttemptRecord:
	id: str
	tenant_id: str
	message_id: str
	subscription_id: str
	outcome: str
	retry_count: int
	reason: str
	status: str
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class PriorityQuotaExceptionRecord:
	id: str
	tenant_id: str
	topic_id: str
	requested_by: str
	reason: str
	status: str = "pending"
	decision: str = ""
	reviewer: str = ""
	notes: str = ""
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class ReplayRequestRecord:
	id: str
	tenant_id: str
	topic_id: str
	requested_by: str
	reason: str
	range_start: str
	range_end: str
	status: str = "pending"
	decision: str = ""
	reviewer: str = ""
	evidence: str = ""
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class MqebAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	status: str = "active"
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class EventLifecycleBatchRecord:
	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	accepted: bool
	decision: str
	matched_rules: list[str]
	required_actions: list[str]
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class MqebAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "info"
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


class MqebService:
	"""Dependency-light MQEB event-fabric service for generated APG applications."""

	def __init__(self) -> None:
		from .capability_contract import (
			PRIVILEGED_MQEB_AGENT_ROLES,
			SUPPORTED_MQEB_AGENT_ROLES,
			SUPPORTED_MQEB_AGENT_RUNTIMES,
			evaluate_capability_rules,
			get_capability_contract,
		)

		self._evaluate_rules = evaluate_capability_rules
		self._get_contract = get_capability_contract
		self._agent_runtimes = set(SUPPORTED_MQEB_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_MQEB_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_MQEB_AGENT_ROLES)
		self.topics: dict[str, TopicRecord] = {}
		self.messages: dict[str, MessageRecord] = {}
		self.subscriptions: dict[str, SubscriptionRecord] = {}
		self.delivery_attempts: dict[str, DeliveryAttemptRecord] = {}
		self.priority_exceptions: dict[str, PriorityQuotaExceptionRecord] = {}
		self.replay_requests: dict[str, ReplayRequestRecord] = {}
		self.event_agents: dict[str, MqebAgentRecord] = {}
		self.lifecycle_batches: dict[str, EventLifecycleBatchRecord] = {}
		self.audit_events: dict[str, MqebAuditEventRecord] = {}

	def describe(self, tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		return self._get_contract(tenant_id, overrides)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return self._evaluate_rules(dict(context))

	def create_topic(
		self,
		tenant_id: str,
		topic_id: str,
		name: str,
		owner: str,
		classification: str = "internal",
		retention_days: int = 7,
		delivery_mode: str = "at_least_once",
		encrypted: bool = False,
		schema_ref: str = "",
		dead_letter_topic: str = "",
		status: str = "active",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(topic_id or "").strip():
			raise ValueError("topic_id_required")
		if not str(name or "").strip():
			raise ValueError("topic_name_required")
		if not str(owner or "").strip():
			raise ValueError("topic_owner_required")
		classification_value = _normalize_choice(classification, TOPIC_CLASSIFICATIONS, "internal", "unsupported_topic_classification")
		status_value = _normalize_choice(status, TOPIC_STATUSES, "active", "unsupported_topic_status")
		delivery_value = _normalize_choice(delivery_mode, DELIVERY_MODES, "at_least_once", "unsupported_delivery_mode")
		retention_value = int(retention_days)
		if retention_value <= 0:
			raise ValueError("topic_retention_days_required")
		record_id = _stable_id("mqeb_topic", tenant_id, topic_id)
		if record_id in self.topics:
			raise ValueError(f"topic_already_exists:{topic_id}")
		record = TopicRecord(
			id=record_id,
			tenant_id=tenant_id,
			name=str(name).strip(),
			owner=str(owner).strip(),
			classification=classification_value,
			retention_days=retention_value,
			delivery_mode=delivery_value,
			encrypted=bool(encrypted),
			schema_ref=str(schema_ref or "").strip(),
			dead_letter_topic=str(dead_letter_topic or "").strip(),
			status=status_value,
		)
		self.topics[record.id] = record
		self._record_event(tenant_id, "topic_created", record.id, f"Topic created: {record.name}", owner)
		return record.to_dict()

	def publish_message(
		self,
		tenant_id: str,
		message_id: str,
		topic_id: str,
		producer: str,
		priority: str = "normal",
		delivery_mode: str | None = None,
		encrypted: bool | None = None,
		schema_ref: str = "",
		idempotency_key: str = "",
		payload_size: int = 1,
		priority_messages_per_minute: int = 0,
		cross_tenant_publish: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(message_id or "").strip():
			raise ValueError("message_id_required")
		if not str(producer or "").strip():
			raise ValueError("message_producer_required")
		if int(payload_size) <= 0:
			raise ValueError("message_payload_required")
		topic = self._get_topic(tenant_id, topic_id)
		delivery_value = _normalize_choice(delivery_mode or topic.delivery_mode, DELIVERY_MODES, topic.delivery_mode, "unsupported_delivery_mode")
		encrypted_value = topic.encrypted if encrypted is None else bool(encrypted)
		schema_value = str(schema_ref or topic.schema_ref or "").strip()
		context = {
			"tenant_context_present": True,
			"operation": "publish",
			"topic_exists": True,
			"topic_classification": topic.classification,
			"message_encrypted": encrypted_value,
			"schema_ref_present": bool(schema_value),
			"cross_tenant_publish": bool(cross_tenant_publish),
			"delivery_mode": delivery_value,
			"dead_letter_queue_configured": bool(topic.dead_letter_topic),
			"idempotency_key_present": bool(str(idempotency_key or "").strip()),
			"topic_status": topic.status,
			"priority_messages_per_minute": int(priority_messages_per_minute),
			"quota_exception_recorded": self._priority_exception_approved(tenant_id, topic.id),
		}
		result = self.evaluate(context)
		status = {"allow": "published", "deny": "denied", "require_review": "review_required"}[result["decision"]]
		record = MessageRecord(
			id=_stable_id("mqeb_message", tenant_id, message_id),
			tenant_id=tenant_id,
			topic_id=topic.id,
			producer=str(producer).strip(),
			priority=str(priority or "normal").strip().lower(),
			delivery_mode=delivery_value,
			status=status,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			required_actions=_required_actions(result),
			idempotency_key=str(idempotency_key or "").strip(),
			schema_ref=schema_value,
		)
		self.messages[record.id] = record
		severity = "high" if status == "denied" else "medium" if status == "review_required" else "info"
		self._record_event(tenant_id, f"message_{status}", record.id, f"Message {status}: {topic.name}", producer, severity)
		return record.to_dict()

	def create_subscription(
		self,
		tenant_id: str,
		subscription_id: str,
		name: str,
		topic_pattern: str,
		consumer: str,
		delivery_mode: str = "at_least_once",
		protocol: str = "bytewax",
		dead_letter_topic: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(subscription_id or "").strip():
			raise ValueError("subscription_id_required")
		if not str(name or "").strip():
			raise ValueError("subscription_name_required")
		if not str(topic_pattern or "").strip():
			raise ValueError("subscription_topic_pattern_required")
		if not str(consumer or "").strip():
			raise ValueError("subscription_consumer_required")
		delivery_value = _normalize_choice(delivery_mode, DELIVERY_MODES, "at_least_once", "unsupported_delivery_mode")
		protocol_value = _normalize_choice(protocol, PROTOCOLS, "bytewax", "unsupported_protocol")
		if delivery_value == "exactly_once" and not str(dead_letter_topic or "").strip():
			raise PermissionError("dead_letter_queue_required")
		record_id = _stable_id("mqeb_subscription", tenant_id, subscription_id)
		if record_id in self.subscriptions:
			raise ValueError(f"subscription_already_exists:{subscription_id}")
		record = SubscriptionRecord(
			id=record_id,
			tenant_id=tenant_id,
			name=str(name).strip(),
			topic_pattern=str(topic_pattern).strip(),
			consumer=str(consumer).strip(),
			delivery_mode=delivery_value,
			protocol=protocol_value,
			dead_letter_topic=str(dead_letter_topic or "").strip(),
		)
		self.subscriptions[record.id] = record
		self._record_event(tenant_id, "subscription_created", record.id, f"Subscription created: {record.name}", consumer)
		return record.to_dict()

	def pause_subscription(self, tenant_id: str, subscription_id: str, actor: str, reason: str) -> dict[str, Any]:
		record = self._get_subscription(tenant_id, subscription_id)
		if not str(actor or "").strip():
			raise ValueError("subscription_actor_required")
		if not str(reason or "").strip():
			raise ValueError("subscription_pause_reason_required")
		record.status = "paused"
		self._record_event(tenant_id, "subscription_paused", record.id, reason, actor, "medium")
		return record.to_dict()

	def resume_subscription(self, tenant_id: str, subscription_id: str, actor: str, evidence: str) -> dict[str, Any]:
		record = self._get_subscription(tenant_id, subscription_id)
		if not str(actor or "").strip():
			raise ValueError("subscription_actor_required")
		if not str(evidence or "").strip():
			raise ValueError("subscription_resume_evidence_required")
		record.status = "active"
		self._record_event(tenant_id, "subscription_resumed", record.id, evidence, actor)
		return record.to_dict()

	def record_delivery_attempt(
		self,
		tenant_id: str,
		attempt_id: str,
		message_id: str,
		subscription_id: str,
		outcome: str,
		retry_count: int = 0,
		reason: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		message = self._get_message(tenant_id, message_id)
		subscription = self._get_subscription(tenant_id, subscription_id)
		outcome_value = _normalize_choice(outcome, DELIVERY_OUTCOMES, "delivered", "unsupported_delivery_outcome")
		context = {
			"operation": "deliver",
			"subscription_status": subscription.status,
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		if outcome_value in {"retry", "dead_letter"} and not str(reason or "").strip():
			raise ValueError("delivery_failure_reason_required")
		record = DeliveryAttemptRecord(
			id=_stable_id("mqeb_delivery", tenant_id, attempt_id),
			tenant_id=tenant_id,
			message_id=message.id,
			subscription_id=subscription.id,
			outcome=outcome_value,
			retry_count=max(0, int(retry_count)),
			reason=str(reason or "").strip(),
			status="dead_letter" if outcome_value == "dead_letter" else outcome_value,
		)
		self.delivery_attempts[record.id] = record
		subscription.lag_messages = max(0, subscription.lag_messages - 1) if outcome_value == "delivered" else subscription.lag_messages + 1
		if outcome_value == "dead_letter":
			message.status = "dead_letter"
		self._record_event(tenant_id, f"delivery_{outcome_value}", record.id, f"Delivery {outcome_value}: {subscription.name}", subscription.consumer, "medium")
		return record.to_dict()

	def request_priority_exception(self, tenant_id: str, exception_id: str, topic_id: str, requested_by: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		topic = self._get_topic(tenant_id, topic_id)
		if not str(exception_id or "").strip():
			raise ValueError("priority_exception_id_required")
		if not str(requested_by or "").strip():
			raise ValueError("priority_exception_requester_required")
		if not str(reason or "").strip():
			raise ValueError("priority_exception_reason_required")
		record_id = _stable_id("mqeb_priority_exception", tenant_id, exception_id)
		if record_id in self.priority_exceptions:
			raise ValueError(f"priority_exception_already_exists:{exception_id}")
		record = PriorityQuotaExceptionRecord(record_id, tenant_id, topic.id, str(requested_by).strip(), str(reason).strip())
		self.priority_exceptions[record.id] = record
		self._record_event(tenant_id, "priority_exception_requested", record.id, f"Priority exception requested: {topic.name}", requested_by, "medium")
		return record.to_dict()

	def decide_priority_exception(self, tenant_id: str, exception_id: str, reviewer: str, decision: str, notes: str) -> dict[str, Any]:
		record = self._get_priority_exception(tenant_id, exception_id)
		self._decide_review_record(record, "decide_priority_exception", reviewer, decision, notes, "independent_priority_exception_reviewer_required")
		self._record_event(tenant_id, "priority_exception_decided", record.id, f"Priority exception {record.status}: {record.topic_id}", reviewer, "medium")
		return record.to_dict()

	def request_replay(self, tenant_id: str, replay_id: str, topic_id: str, requested_by: str, reason: str, range_start: str, range_end: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		topic = self._get_topic(tenant_id, topic_id)
		if not str(replay_id or "").strip():
			raise ValueError("replay_id_required")
		if not str(requested_by or "").strip():
			raise ValueError("replay_requester_required")
		context = {
			"operation": "replay",
			"replay_reason_present": bool(str(reason or "").strip()),
			"replay_range_bounded": bool(str(range_start or "").strip() and str(range_end or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		record_id = _stable_id("mqeb_replay", tenant_id, replay_id)
		if record_id in self.replay_requests:
			raise ValueError(f"replay_already_exists:{replay_id}")
		record = ReplayRequestRecord(
			id=record_id,
			tenant_id=tenant_id,
			topic_id=topic.id,
			requested_by=str(requested_by).strip(),
			reason=str(reason).strip(),
			range_start=str(range_start).strip(),
			range_end=str(range_end).strip(),
		)
		self.replay_requests[record.id] = record
		self._record_event(tenant_id, "replay_requested", record.id, f"Replay requested: {topic.name}", requested_by, "medium")
		return record.to_dict()

	def decide_replay(self, tenant_id: str, replay_id: str, reviewer: str, decision: str, evidence: str) -> dict[str, Any]:
		record = self._get_replay(tenant_id, replay_id)
		self._decide_review_record(record, "decide_replay", reviewer, decision, evidence, "independent_replay_reviewer_required", notes_field="evidence")
		self._record_event(tenant_id, "replay_decided", record.id, f"Replay {record.status}: {record.topic_id}", reviewer, "medium")
		return record.to_dict()

	def register_event_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		"""Register a first-class MQEB event agent with guardrail evidence."""
		self._require_tenant(tenant_id)
		if not str(agent_id or "").strip():
			raise ValueError("event_agent_id_required")
		if not str(name or "").strip():
			raise ValueError("event_agent_name_required")
		runtime_value = self._normalize_agent_token(runtime)
		role_value = self._normalize_agent_token(role)
		context = {
			"operation": "register_event_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"agent_scope_present": bool(str(scope or "").strip()),
			"agent_owner_present": bool(str(owner or "").strip()),
			"agent_purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_agent_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		record_id = _stable_id("mqeb_agent", tenant_id, agent_id)
		if record_id in self.event_agents:
			raise ValueError(f"event_agent_already_exists:{agent_id}")
		record = MqebAgentRecord(
			id=record_id,
			tenant_id=tenant_id,
			name=str(name).strip(),
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
		)
		self.event_agents[record.id] = record
		self._record_event(tenant_id, "event_agent_registered", record.id, f"Event agent registered: {record.name}", record.owner)
		return record.to_dict()

	def validate_event_lifecycle_batch(self, tenant_id: str, event_stream: str, mutation_count: int) -> dict[str, Any]:
		"""Validate that MQEB lifecycle mutation batches flow through Bytewax."""
		self._require_tenant(tenant_id)
		mutation_value = int(mutation_count)
		if mutation_value <= 0:
			raise ValueError("event_lifecycle_batch_empty")
		stream_value = self._normalize_agent_token(event_stream)
		result = self.evaluate({
			"operation": "validate_event_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		record = EventLifecycleBatchRecord(
			id=_stable_id("mqeb_lifecycle_batch", tenant_id, stream_value, len(self.lifecycle_batches)),
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			required_actions=_required_actions(result),
			status="accepted" if accepted else "denied",
		)
		self.lifecycle_batches[record.id] = record
		severity = "info" if accepted else "high"
		self._record_event(tenant_id, f"event_lifecycle_batch_{record.status}", record.id, f"Lifecycle batch {record.status}: {stream_value}", "mqeb", severity)
		if not accepted:
			raise PermissionError(self._first_reason(result))
		return record.to_dict()

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_topic(
			tenant_id=tenant_id,
			topic_id=record_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or metadata.get("created_by") or "system"),
			classification=str(metadata.get("classification") or "internal"),
			retention_days=int(metadata.get("retention_days", 7) or 7),
			delivery_mode=str(metadata.get("delivery_mode") or "at_least_once"),
			encrypted=bool(metadata.get("encrypted", False)),
			schema_ref=str(metadata.get("schema_ref") or ""),
			dead_letter_topic=str(metadata.get("dead_letter_topic") or ""),
			status=status,
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_topics(tenant_id)

	def list_topics(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.topics, tenant_id)

	def list_messages(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.messages, tenant_id)

	def list_subscriptions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.subscriptions, tenant_id)

	def list_delivery_attempts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.delivery_attempts, tenant_id)

	def list_priority_exceptions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.priority_exceptions, tenant_id)

	def list_replay_requests(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.replay_requests, tenant_id)

	def list_event_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.event_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		messages = self.list_messages(tenant_id)
		subscriptions = self.list_subscriptions(tenant_id)
		return {
			"tenant_id": tenant_id,
			"topic_count": len(self.list_topics(tenant_id)),
			"message_count": len(messages),
			"denied_message_count": sum(1 for item in messages if item["status"] == "denied"),
			"review_required_count": sum(1 for item in messages if item["status"] == "review_required"),
			"subscription_count": len(subscriptions),
			"paused_subscription_count": sum(1 for item in subscriptions if item["status"] == "paused"),
			"dead_letter_count": sum(1 for item in self.list_delivery_attempts(tenant_id) if item["outcome"] == "dead_letter"),
			"pending_priority_exception_count": sum(1 for item in self.list_priority_exceptions(tenant_id) if item["status"] == "pending"),
			"pending_replay_count": sum(1 for item in self.list_replay_requests(tenant_id) if item["status"] == "pending"),
			"event_agent_count": len(self.list_event_agents(tenant_id)),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": sum(1 for item in self.list_lifecycle_batches(tenant_id) if not item["accepted"]),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			raise PermissionError("tenant_context_required")

	def _get_topic(self, tenant_id: str, topic_id: str) -> TopicRecord:
		record = self.topics.get(_stable_id("mqeb_topic", tenant_id, topic_id)) or self.topics.get(topic_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"topic_not_found:{topic_id}")
		return record

	def _get_message(self, tenant_id: str, message_id: str) -> MessageRecord:
		record = self.messages.get(_stable_id("mqeb_message", tenant_id, message_id)) or self.messages.get(message_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"message_not_found:{message_id}")
		return record

	def _get_subscription(self, tenant_id: str, subscription_id: str) -> SubscriptionRecord:
		record = self.subscriptions.get(_stable_id("mqeb_subscription", tenant_id, subscription_id)) or self.subscriptions.get(subscription_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"subscription_not_found:{subscription_id}")
		return record

	def _get_priority_exception(self, tenant_id: str, exception_id: str) -> PriorityQuotaExceptionRecord:
		record = self.priority_exceptions.get(_stable_id("mqeb_priority_exception", tenant_id, exception_id)) or self.priority_exceptions.get(exception_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"priority_exception_not_found:{exception_id}")
		return record

	def _get_replay(self, tenant_id: str, replay_id: str) -> ReplayRequestRecord:
		record = self.replay_requests.get(_stable_id("mqeb_replay", tenant_id, replay_id)) or self.replay_requests.get(replay_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"replay_not_found:{replay_id}")
		return record

	def _priority_exception_approved(self, tenant_id: str, topic_id: str) -> bool:
		return any(item.tenant_id == tenant_id and item.topic_id == topic_id and item.status == "approved" for item in self.priority_exceptions.values())

	def _decide_review_record(
		self,
		record: Any,
		operation: str,
		reviewer: str,
		decision: str,
		notes: str,
		self_review_reason: str,
		notes_field: str = "notes",
	) -> None:
		if record.status != "pending":
			raise ValueError("review_already_decided")
		decision_value = str(decision or "").strip().lower()
		if decision_value not in {"approved", "rejected"}:
			raise ValueError("review_decision_invalid")
		reviewer_value = str(reviewer or "").strip()
		notes_value = str(notes or "").strip()
		if not reviewer_value:
			raise ValueError("reviewer_required")
		if not notes_value:
			raise ValueError("review_notes_required")
		requester_value = str(record.requested_by or "").strip()
		result = self.evaluate({
			"operation": operation,
			"reviewer_same_as_requester": reviewer_value.casefold() == requester_value.casefold(),
			"review_notes_attached": bool(notes_value),
		})
		if result["decision"] == "deny":
			reason = self._first_reason(result)
			raise PermissionError(self_review_reason if reason == "independent_reviewer_required" else reason)
		record.status = decision_value
		record.decision = decision_value
		record.reviewer = reviewer_value
		setattr(record, notes_field, notes_value)

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, actor: str, severity: str = "info") -> dict[str, Any]:
		record = MqebAuditEventRecord(
			id=_stable_id("mqeb_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _first_reason(self, result: dict[str, Any]) -> str:
		for action in result.get("actions", []):
			if action.get("reason"):
				return str(action["reason"])
		return "message_operation_denied"

	def _normalize_agent_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])


class MQEBService:
	"""
	Core MQEB service implementation
	Provides high-performance message queuing with APG integration
	"""
	
	def __init__(self, config: Dict[str, Any] | None = None):
		self.config = config or {}
		self.running = False
		
		# Core components
		self.message_store: Dict[str, MQMessage] = {}
		self.topics: Dict[str, TopicConfiguration] = {}
		self.subscriptions: Dict[str, Subscription] = {}
		self.broker_nodes: Dict[str, BrokerNode] = {}
		
		# Message routing and processing
		self.message_queues: Dict[str, List[str]] = {}  # topic -> message_ids
		self.subscription_queues: Dict[str, List[str]] = {}  # subscription_id -> message_ids
		self.dead_letter_queues: Dict[str, List[str]] = {}
		
		# Performance tracking
		self.metrics = {
			'messages_published': 0,
			'messages_delivered': 0,
			'messages_failed': 0,
			'bytes_processed': 0,
			'active_connections': 0,
			'topics_created': 0,
			'subscriptions_created': 0
		}
		
		# Background tasks
		self._background_tasks: Set[asyncio.Task] = set()
		
		# Logging
		self.logger = logging.getLogger('mqeb.service')
	
	async def _log_audit_event(self, event_type: str, resource_id: str, action: str, 
							  user_id: str = None, details: Dict[str, Any] = None) -> None:
		"""Log audit events for compliance"""
		event = MessageEvent(
			message_id=resource_id,
			event_type=event_type,
			status="success",
			tenant_id=details.get('tenant_id', 'default') if details else 'default',
			user_id=user_id,
			metadata=details or {}
		)
		
		# In production, would persist to audit database
		self.logger.info(f"[AUDIT] {event_type}: {action} by {user_id}")
	
	async def initialize(self, config: Dict[str, Any] | None = None) -> None:
		"""Initialize MQEB service"""
		if config:
			self.config.update(config)
		
		self.logger.info("Initializing MQEB service...")
		
		# Initialize broker node
		await self._initialize_broker_node()
		
		# Initialize security and compliance engines
		await self._initialize_security_engines()
		
		# Start background tasks
		await self._start_background_tasks()
		
		# Initialize default topics
		await self._create_default_topics()
		
		self.running = True
		self.logger.info("MQEB service initialized successfully")
	
	async def shutdown(self) -> None:
		"""Shutdown MQEB service gracefully"""
		self.logger.info("Shutting down MQEB service...")
		
		self.running = False
		
		# Shutdown security engines if initialized
		if hasattr(self, 'quantum_security'):
			try:
				await self.quantum_security.shutdown()
			except Exception as e:
				self.logger.error(f"Error shutting down quantum security engine: {e}")
		
		if hasattr(self, 'compliance_governance'):
			try:
				await self.compliance_governance.shutdown()
			except Exception as e:
				self.logger.error(f"Error shutting down compliance governance engine: {e}")
		
		if hasattr(self, 'enterprise_workflow'):
			try:
				await self.enterprise_workflow.shutdown()
			except Exception as e:
				self.logger.error(f"Error shutting down enterprise workflow engine: {e}")
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		# Wait for tasks to complete
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		
		self.logger.info("MQEB service shut down")
	
	async def _initialize_broker_node(self) -> None:
		"""Initialize this broker node"""
		node = BrokerNode(
			name=f"mqeb-broker-{uuid.uuid4().hex[:8]}",
			hostname="localhost",  # Would be actual hostname
			ip_address="127.0.0.1",  # Would be actual IP
			port=8080,
			region="us-east-1",  # Would be from config
			cluster_id="default-cluster",
			protocols_enabled=[ProtocolType.HTTP_REST, ProtocolType.WEBSOCKET]
		)
		
		self.broker_nodes[node.id] = node
		self.logger.info(f"Initialized broker node: {node.name}")
	
	async def _initialize_security_engines(self) -> None:
		"""Initialize quantum security, compliance governance, and enterprise workflow engines"""
		try:
			# Try to initialize quantum security engine
			if self.config.get('quantum_security_enabled', True):
				try:
					from .quantum_security import create_quantum_security_engine
					self.quantum_security = await create_quantum_security_engine(self)
					self.logger.info("Quantum security engine initialized")
				except ImportError:
					self.logger.warning("Quantum security module not available")
				except Exception as e:
					self.logger.error(f"Failed to initialize quantum security engine: {e}")
			
			# Try to initialize compliance governance engine
			if self.config.get('compliance_governance_enabled', True):
				try:
					from .compliance_governance import create_compliance_governance_engine
					self.compliance_governance = await create_compliance_governance_engine(self)
					self.logger.info("Compliance governance engine initialized")
				except ImportError:
					self.logger.warning("Compliance governance module not available")
				except Exception as e:
					self.logger.error(f"Failed to initialize compliance governance engine: {e}")
			
			# Try to initialize enterprise workflow engine
			if self.config.get('enterprise_workflows_enabled', True):
				try:
					from .enterprise_integration import create_enterprise_workflow_engine
					self.enterprise_workflow = await create_enterprise_workflow_engine(self)
					self.logger.info("Enterprise workflow engine initialized")
				except ImportError:
					self.logger.warning("Enterprise integration module not available")
				except Exception as e:
					self.logger.error(f"Failed to initialize enterprise workflow engine: {e}")
			
		except Exception as e:
			self.logger.error(f"Error initializing security engines: {e}")
			# Continue without security engines - service should still be functional
	
	async def _start_background_tasks(self) -> None:
		"""Start background processing tasks"""
		
		# Message processing task
		task = asyncio.create_task(self._message_processing_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Metrics collection task
		task = asyncio.create_task(self._metrics_collection_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Health monitoring task
		task = asyncio.create_task(self._health_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		self.logger.info("Started background processing tasks")
	
	async def _create_default_topics(self) -> None:
		"""Create default system topics"""
		default_topics = [
			{
				'name': 'system.events',
				'description': 'System-wide events and notifications',
				'partitions': 5
			},
			{
				'name': 'user.events',
				'description': 'User activity events',
				'partitions': 10
			},
			{
				'name': 'application.logs', 
				'description': 'Application log messages',
				'partitions': 15
			},
			{
				'name': 'metrics.performance',
				'description': 'Performance and monitoring metrics',
				'partitions': 5
			}
		]
		
		for topic_spec in default_topics:
			try:
				topic_config = TopicConfiguration(
					name=topic_spec['name'],
					description=topic_spec['description'],
					partitions=topic_spec['partitions'],
					tenant_id='system',
					created_by='system'
				)
				
				await self.create_topic(topic_config)
				
			except Exception as e:
				self.logger.warning(f"Failed to create default topic {topic_spec['name']}: {e}")
	
	async def create_topic(self, topic_config: TopicConfiguration) -> str:
		"""Create a new topic"""
		
		# Validate topic doesn't already exist
		if topic_config.name in self.topics:
			raise ValueError(f"Topic {topic_config.name} already exists")
		
		# Store topic configuration
		self.topics[topic_config.name] = topic_config
		
		# Initialize topic message queue
		self.message_queues[topic_config.name] = []
		
		# Update metrics
		self.metrics['topics_created'] += 1
		
		# Log audit event
		await self._log_audit_event(
			event_type="topic_created",
			resource_id=topic_config.name,
			action="create_topic",
			user_id=topic_config.created_by,
			details={
				'topic_name': topic_config.name,
				'partitions': topic_config.partitions,
				'tenant_id': topic_config.tenant_id
			}
		)
		
		self.logger.info(f"Created topic: {topic_config.name}")
		return topic_config.name
	
	async def publish_message(self, message: MQMessage, context: Dict[str, Any] | None = None) -> str:
		"""Publish a message to a topic"""
		
		# Validate topic exists
		if message.topic not in self.topics:
			raise ValueError(f"Topic {message.topic} does not exist")
		
		# Validate message
		if message.is_expired():
			raise ValueError("Message has already expired")
		
		# Apply security and compliance checks
		if context is None:
			context = {}
		
		# Quantum security processing
		if hasattr(self, 'quantum_security'):
			try:
				security_result = await self.quantum_security.secure_message(message, context)
				if not security_result:
					raise ValueError("Message failed quantum security validation")
			except Exception as e:
				self.logger.error(f"Quantum security processing failed: {e}")
				# Continue without security - configurable behavior
		
		# Compliance and governance processing
		if hasattr(self, 'compliance_governance'):
			try:
				compliance_result = await self.compliance_governance.process_message_compliance(message, context)
				if not compliance_result['compliant']:
					self.logger.warning(f"Message {message.id} compliance violations: {compliance_result['violations']}")
					# In strict mode, could reject message here
			except Exception as e:
				self.logger.error(f"Compliance processing failed: {e}")
		
		# Store message
		self.message_store[message.id] = message
		
		# Add to topic queue
		self.message_queues[message.topic].append(message.id)
		
		# Route to subscriptions
		await self._route_message_to_subscriptions(message)
		
		# Trigger enterprise workflows if enabled
		if hasattr(self, 'enterprise_workflow') and self.enterprise_workflow.running:
			try:
				execution_id = await self.enterprise_workflow.trigger_workflow(message, context)
				if execution_id:
					self.logger.debug(f"Triggered workflow execution {execution_id} for message {message.id}")
			except Exception as e:
				self.logger.error(f"Failed to trigger workflow for message {message.id}: {e}")
		
		# Update metrics
		self.metrics['messages_published'] += 1
		self.metrics['bytes_processed'] += message.size_bytes()
		
		# Log audit event
		await self._log_audit_event(
			event_type="message_published",
			resource_id=message.id,
			action="publish_message",
			user_id=message.user_id,
			details={
				'topic': message.topic,
				'size_bytes': message.size_bytes(),
				'tenant_id': message.tenant_id
			}
		)
		
		self.logger.debug(f"Published message {message.id} to topic {message.topic}")
		return message.id
	
	async def _route_message_to_subscriptions(self, message: MQMessage) -> None:
		"""Route message to matching subscriptions"""
		
		for subscription in self.subscriptions.values():
			# Check if subscription matches message topic
			if await self._subscription_matches_topic(subscription, message.topic):
				# Check message filter
				if subscription.message_filter and not subscription.message_filter.matches(message):
					continue
				
				# Add to subscription queue
				if subscription.id not in self.subscription_queues:
					self.subscription_queues[subscription.id] = []
				
				self.subscription_queues[subscription.id].append(message.id)
				
				self.logger.debug(f"Routed message {message.id} to subscription {subscription.id}")
	
	async def _subscription_matches_topic(self, subscription: Subscription, topic: str) -> bool:
		"""Check if subscription topic pattern matches the given topic"""
		
		import fnmatch
		return fnmatch.fnmatch(topic, subscription.topic_pattern)
	
	async def create_subscription(self, subscription: Subscription) -> str:
		"""Create a new subscription"""
		
		# Validate subscription doesn't already exist
		if subscription.id in self.subscriptions:
			raise ValueError(f"Subscription {subscription.id} already exists")
		
		# Store subscription
		self.subscriptions[subscription.id] = subscription
		
		# Initialize subscription queue
		self.subscription_queues[subscription.id] = []
		
		# Update metrics
		self.metrics['subscriptions_created'] += 1
		
		# Log audit event
		await self._log_audit_event(
			event_type="subscription_created",
			resource_id=subscription.id,
			action="create_subscription",
			user_id=subscription.created_by,
			details={
				'subscription_name': subscription.name,
				'topic_pattern': subscription.topic_pattern,
				'tenant_id': subscription.tenant_id
			}
		)
		
		self.logger.info(f"Created subscription: {subscription.name}")
		return subscription.id
	
	async def consume_messages(self, subscription_id: str, max_messages: int = 10) -> List[MQMessage]:
		"""Consume messages from a subscription"""
		
		if subscription_id not in self.subscriptions:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		subscription = self.subscriptions[subscription_id]
		
		if subscription_id not in self.subscription_queues:
			return []
		
		# Get messages from subscription queue
		message_ids = self.subscription_queues[subscription_id][:max_messages]
		messages = []
		
		for message_id in message_ids:
			if message_id in self.message_store:
				message = self.message_store[message_id]
				
				# Check if message is still valid
				if not message.is_expired():
					messages.append(message)
				else:
					# Remove expired message
					self.message_store.pop(message_id, None)
		
		# Remove consumed messages from queue (for at-least-once delivery)
		if subscription.delivery_mode == DeliveryMode.AT_LEAST_ONCE:
			self.subscription_queues[subscription_id] = self.subscription_queues[subscription_id][len(messages):]
		
		# Update metrics
		self.metrics['messages_delivered'] += len(messages)
		
		self.logger.debug(f"Consumed {len(messages)} messages from subscription {subscription_id}")
		return messages
	
	async def get_topic_stats(self, topic_name: str) -> Dict[str, Any]:
		"""Get statistics for a topic"""
		
		if topic_name not in self.topics:
			raise ValueError(f"Topic {topic_name} not found")
		
		topic_config = self.topics[topic_name]
		message_queue = self.message_queues.get(topic_name, [])
		
		# Calculate message sizes
		total_size = 0
		for message_id in message_queue:
			if message_id in self.message_store:
				total_size += self.message_store[message_id].size_bytes()
		
		# Count active subscriptions
		active_subscriptions = 0
		for subscription in self.subscriptions.values():
			if await self._subscription_matches_topic(subscription, topic_name):
				active_subscriptions += 1
		
		return {
			'topic_name': topic_name,
			'partitions': topic_config.partitions,
			'replication_factor': topic_config.replication_factor,
			'total_messages': len(message_queue),
			'total_size_bytes': total_size,
			'active_subscriptions': active_subscriptions,
			'retention_ms': topic_config.retention_ms,
			'created_at': topic_config.created_at.isoformat()
		}
	
	async def get_subscription_stats(self, subscription_id: str) -> Dict[str, Any]:
		"""Get statistics for a subscription"""
		
		if subscription_id not in self.subscriptions:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		subscription = self.subscriptions[subscription_id]
		queue_size = len(self.subscription_queues.get(subscription_id, []))
		
		return {
			'subscription_id': subscription_id,
			'name': subscription.name,
			'topic_pattern': subscription.topic_pattern,
			'delivery_mode': subscription.delivery_mode.value,
			'protocol': subscription.protocol.value,
			'pending_messages': queue_size,
			'total_messages': subscription.total_messages,
			'failed_messages': subscription.failed_messages,
			'success_rate': subscription.success_rate(),
			'enabled': subscription.enabled,
			'paused': subscription.paused,
			'created_at': subscription.created_at.isoformat()
		}
	
	async def get_cluster_stats(self) -> Dict[str, Any]:
		"""Get cluster-wide statistics"""
		
		return {
			'broker_nodes': len(self.broker_nodes),
			'total_topics': len(self.topics),
			'total_subscriptions': len(self.subscriptions),
			'total_messages_stored': len(self.message_store),
			'metrics': self.metrics.copy(),
			'uptime_seconds': int((datetime.utcnow() - datetime.utcnow()).total_seconds()),  # Would be actual uptime
			'cluster_healthy': self._is_cluster_healthy()
		}
	
	def _is_cluster_healthy(self) -> bool:
		"""Check if cluster is healthy"""
		
		# Check if any broker nodes are unhealthy
		for node in self.broker_nodes.values():
			if not node.is_healthy():
				return False
		
		# Check for any critical issues
		error_rate = self.metrics.get('messages_failed', 0) / max(1, self.metrics.get('messages_published', 1))
		if error_rate > 0.05:  # 5% error rate threshold
			return False
		
		return True
	
	async def _message_processing_loop(self) -> None:
		"""Background message processing loop"""
		
		while self.running:
			try:
				# Process message delivery
				await self._process_pending_deliveries()
				
				# Clean up expired messages
				await self._cleanup_expired_messages()
				
				# Process dead letter queues
				await self._process_dead_letter_queues()
				
				# Sleep before next iteration
				await asyncio.sleep(1)
				
			except Exception as e:
				self.logger.error(f"Error in message processing loop: {e}")
				await asyncio.sleep(5)
	
	async def _process_pending_deliveries(self) -> None:
		"""Process pending message deliveries"""
		
		for subscription_id, message_ids in self.subscription_queues.items():
			if not message_ids:
				continue
			
			subscription = self.subscriptions.get(subscription_id)
			if not subscription or not subscription.enabled or subscription.paused:
				continue
			
			# Process deliveries for this subscription
			try:
				await self._deliver_messages_to_subscription(subscription, message_ids[:10])
			except Exception as e:
				self.logger.error(f"Error delivering messages to subscription {subscription_id}: {e}")
	
	async def _deliver_messages_to_subscription(self, subscription: Subscription, message_ids: List[str]) -> None:
		"""Deliver messages to a specific subscription"""
		
		messages = []
		for message_id in message_ids:
			if message_id in self.message_store:
				messages.append(self.message_store[message_id])
		
		if not messages:
			return
		
		# Simulate message delivery based on protocol
		if subscription.protocol == ProtocolType.HTTP_REST and subscription.webhook_url:
			await self._deliver_via_webhook(subscription, messages)
		elif subscription.protocol == ProtocolType.WEBSOCKET:
			await self._deliver_via_websocket(subscription, messages)
		else:
			# Fallback - mark as delivered for now
			self.logger.debug(f"Simulated delivery of {len(messages)} messages to {subscription.id}")
		
		# Update subscription statistics
		subscription.total_messages += len(messages)
		subscription.last_delivery = datetime.utcnow()
	
	async def _deliver_via_webhook(self, subscription: Subscription, messages: List[MQMessage]) -> None:
		"""Deliver messages via HTTP webhook"""
		
		# In production, would make actual HTTP requests
		self.logger.debug(f"Webhook delivery simulation: {len(messages)} messages to {subscription.webhook_url}")
		
		# Simulate success for now
		await asyncio.sleep(0.1)  # Simulate network delay
	
	async def _deliver_via_websocket(self, subscription: Subscription, messages: List[MQMessage]) -> None:
		"""Deliver messages via WebSocket"""
		
		# In production, would send via WebSocket connections
		self.logger.debug(f"WebSocket delivery simulation: {len(messages)} messages")
		
		# Simulate success for now
		await asyncio.sleep(0.01)  # Simulate minimal delay
	
	async def _cleanup_expired_messages(self) -> None:
		"""Clean up expired messages"""
		
		expired_message_ids = []
		
		for message_id, message in self.message_store.items():
			if message.is_expired():
				expired_message_ids.append(message_id)
		
		# Remove expired messages
		for message_id in expired_message_ids:
			message = self.message_store.pop(message_id, None)
			if message:
				self.logger.debug(f"Cleaned up expired message: {message_id}")
				
				# Remove from topic queues
				if message.topic in self.message_queues:
					try:
						self.message_queues[message.topic].remove(message_id)
					except ValueError:
						pass
				
				# Remove from subscription queues
				for subscription_queue in self.subscription_queues.values():
					try:
						subscription_queue.remove(message_id)
					except ValueError:
						pass
	
	async def _process_dead_letter_queues(self) -> None:
		"""Process messages in dead letter queues"""
		
		# In production, would implement retry logic and dead letter queue processing
		pass
	
	async def _metrics_collection_loop(self) -> None:
		"""Background metrics collection loop"""
		
		while self.running:
			try:
				# Update node health metrics
				for node in self.broker_nodes.values():
					node.last_heartbeat = datetime.utcnow()
					# Would update actual resource usage metrics
				
				# Calculate performance metrics
				# Would collect actual performance data
				
				await asyncio.sleep(30)  # Collect metrics every 30 seconds
				
			except Exception as e:
				self.logger.error(f"Error in metrics collection loop: {e}")
				await asyncio.sleep(60)
	
	async def _health_monitoring_loop(self) -> None:
		"""Background health monitoring loop"""
		
		while self.running:
			try:
				# Check cluster health
				if not self._is_cluster_healthy():
					self.logger.warning("Cluster health check failed")
				
				# Monitor resource usage
				# Would implement actual resource monitoring
				
				await asyncio.sleep(60)  # Check health every minute
				
			except Exception as e:
				self.logger.error(f"Error in health monitoring loop: {e}")
				await asyncio.sleep(120)


# Factory function
async def create_mqeb_service(config: Dict[str, Any] | None = None) -> MQEBService:
	"""Create and initialize MQEB service"""
	service = MQEBService(config)
	await service.initialize()
	return service


# Export main components
__all__ = [
	'DeliveryAttemptRecord',
	'MessageRecord',
	'MqebAuditEventRecord',
	'MqebService',
	'MQEBService',
	'PriorityQuotaExceptionRecord',
	'ReplayRequestRecord',
	'SubscriptionRecord',
	'TopicRecord',
	'create_mqeb_service'
]
