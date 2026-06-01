"""Dependency-light UI view models for the MQEB capability package."""

from __future__ import annotations

from . import api
from .capability_contract import get_capability_contract
from .service import MqebService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"topics": service.list_topics(tenant_id),
		"messages": service.list_messages(tenant_id),
		"subscriptions": service.list_subscriptions(tenant_id),
		"delivery_attempts": service.list_delivery_attempts(tenant_id),
		"priority_exceptions": service.list_priority_exceptions(tenant_id),
		"replay_requests": service.list_replay_requests(tenant_id),
		"event_agents": service.list_event_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
	}


def topic_inventory_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/mqeb/topics",
		"tenant_id": tenant_id,
		"topics": service.list_topics(tenant_id),
		"classifications": ["public", "internal", "restricted", "regulated"],
		"statuses": ["active", "disabled", "deprecated"],
	}


def publish_workbench_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/mqeb/publish",
		"tenant_id": tenant_id,
		"topics": service.list_topics(tenant_id),
		"messages": service.list_messages(tenant_id),
		"delivery_modes": ["at_most_once", "at_least_once", "exactly_once"],
	}


def subscription_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/mqeb/subscriptions",
		"tenant_id": tenant_id,
		"subscriptions": service.list_subscriptions(tenant_id),
		"protocols": ["bytewax", "http_rest", "websocket", "mqtt", "amqp", "grpc"],
	}


def delivery_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	attempts = service.list_delivery_attempts(tenant_id)
	return {
		"route": "/mqeb/delivery",
		"tenant_id": tenant_id,
		"delivery_attempts": attempts,
		"dead_letters": [item for item in attempts if item["outcome"] == "dead_letter"],
	}


def quota_exception_queue_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	exceptions = service.list_priority_exceptions(tenant_id)
	return {
		"route": "/mqeb/quota-exceptions",
		"tenant_id": tenant_id,
		"priority_exceptions": exceptions,
		"pending": [item for item in exceptions if item["status"] == "pending"],
	}


def replay_console_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	replays = service.list_replay_requests(tenant_id)
	return {
		"route": "/mqeb/replays",
		"tenant_id": tenant_id,
		"replay_requests": replays,
		"pending": [item for item in replays if item["status"] == "pending"],
	}


def bytewax_bridge_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	return {
		"route": "/mqeb/bytewax",
		"tenant_id": tenant_id,
		"preferred_runtime": "bytewax",
		"streaming": contract["streaming"],
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"subscriptions": [item for item in service.list_subscriptions(tenant_id) if item["protocol"] == "bytewax"],
		"adapter_status": "adapter_boundary",
	}


def event_agent_roster_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	contract = service.describe(tenant_id)
	return {
		"route": "/mqeb/agents",
		"tenant_id": tenant_id,
		"event_agents": service.list_event_agents(tenant_id),
		"pending_reviews": [
			agent for agent in service.list_event_agents(tenant_id)
			if agent["status"] == "pending_review"
		],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"guardrails": contract["agents"]["guardrails"],
	}


def audit_timeline_model(service: MqebService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or api.SERVICE
	return {
		"route": "/mqeb/audit",
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/mqeb/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
	}
