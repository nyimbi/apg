"""Dependency-light Integration API Management lifecycle service."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		API_EVENT_STREAM,
		STREAMING,
		SUPPORTED_API_AGENT_ROLES,
		SUPPORTED_API_AGENT_RUNTIMES,
		SUPPORTED_AUTH_TYPES,
		SUPPORTED_ENVIRONMENTS,
		SUPPORTED_METHODS,
		SUPPORTED_PLANS,
		SUPPORTED_POLICY_TYPES,
		SUPPORTED_PROTOCOLS,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		API_EVENT_STREAM,
		STREAMING,
		SUPPORTED_API_AGENT_ROLES,
		SUPPORTED_API_AGENT_RUNTIMES,
		SUPPORTED_AUTH_TYPES,
		SUPPORTED_ENVIRONMENTS,
		SUPPORTED_METHODS,
		SUPPORTED_PLANS,
		SUPPORTED_POLICY_TYPES,
		SUPPORTED_PROTOCOLS,
		evaluate_capability_rules,
		get_capability_contract,
	)


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class APIManagementError(Exception):
	"""Base exception for API Management operations."""


class APINotFoundError(APIManagementError):
	"""Raised when an API is not found."""


class ConsumerNotFoundError(APIManagementError):
	"""Raised when a consumer is not found."""


class AuthenticationError(APIManagementError):
	"""Raised when authentication fails."""


class AuthorizationError(APIManagementError):
	"""Raised when authorization fails."""


class RateLimitExceededError(APIManagementError):
	"""Raised when a rate limit is exceeded."""


class IntApiService:
	"""In-memory executable service for API management lifecycle packets."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None,
				 *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.apis: dict[str, dict[str, Any]] = {}
		self.endpoints: dict[str, dict[str, Any]] = {}
		self.policies: dict[str, dict[str, Any]] = {}
		self.consumers: dict[str, dict[str, Any]] = {}
		self.api_keys: dict[str, dict[str, Any]] = {}
		self.subscriptions: dict[str, dict[str, Any]] = {}
		self.deployments: dict[str, dict[str, Any]] = {}
		self.usage_records: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

		# Extended stores for new methods
		self._integrations: dict[str, dict[str, Any]] = {}
		self._sync_schedules: dict[str, dict[str, Any]] = {}
		self._sync_history: dict[str, list[dict[str, Any]]] = {}
		self._mappings: dict[str, dict[str, Any]] = {}
		self._webhooks: dict[str, dict[str, Any]] = {}
		self._webhook_history: dict[str, list[dict[str, Any]]] = {}
		self._field_transforms: dict[str, dict[str, Any]] = {}

	# ── helpers ───────────────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "tenant_context_present": True,
				"operation": operation, "operation_type": "write", "policy_attached": True}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result.get("decision") == "deny":
			effects = result.get("effects") or result.get("actions") or []
			reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
			raise PermissionError(",".join(reasons) or "operation_denied")

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id, "event_type": event_type, "record_id": record["id"],
			"record_type": record["type"], "status": record["status"],
			"stream": API_EVENT_STREAM, "processor": "bytewax", "emitted_at": _now(),
		})

	# ── ORIGINAL METHODS ──────────────────────────────────────────────────────

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_api(
		self,
		api_id: str,
		tenant_id: str,
		name: str,
		title: str,
		base_path: str,
		upstream_url: str,
		owner_id: str,
		version: str = "1.0.0",
		protocol: str = "rest",
		auth_type: str = "api_key",
		rate_limit_per_minute: int = 1000,
		reviewed_by: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		external_upstream = upstream_url.startswith("http://") or upstream_url.startswith("https://")
		context = self._base_context(tenant, "register_api")
		context.update({
			"name_present": bool(name), "title_present": bool(title),
			"base_path_present": bool(base_path),
			"base_path_valid": bool(base_path and base_path.startswith("/")),
			"upstream_present": bool(upstream_url), "owner_present": bool(owner_id),
			"protocol_supported": protocol in SUPPORTED_PROTOCOLS,
			"auth_type_supported": auth_type in SUPPORTED_AUTH_TYPES,
			"rate_limit": rate_limit_per_minute, "external_upstream": external_upstream,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("api", api_id), "type": "integration_api", "kind": "api",
			"tenant_id": tenant, "name": name, "title": title, "base_path": base_path,
			"upstream_url": upstream_url, "owner_id": owner_id, "version": version,
			"protocol": protocol, "auth_type": auth_type,
			"rate_limit_per_minute": rate_limit_per_minute, "reviewed_by": reviewed_by,
			"approved_by": None, "metadata": deepcopy(metadata or {}),
			"status": "draft", "created_at": _now(), "updated_at": _now(),
		}
		self.apis[record["id"]] = record
		self._emit(tenant, "api_registered", record)
		return deepcopy(record)

	def register_endpoint(
		self,
		endpoint_id: str,
		tenant_id: str,
		api_id: str,
		path: str,
		method: str,
		auth_required: bool = True,
		rate_limit_override: int | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		api = self.apis.get(api_id)
		method = method.upper()
		context = self._base_context(tenant, "register_endpoint")
		context.update({
			"api_present": bool(api and api["tenant_id"] == tenant),
			"path_present": bool(path),
			"path_valid": bool(path and path.startswith("/")),
			"method_supported": method in SUPPORTED_METHODS,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("endpoint", endpoint_id), "type": "integration_api_endpoint",
			"kind": "endpoint", "tenant_id": tenant, "api_id": api_id, "path": path,
			"method": method, "auth_required": auth_required,
			"rate_limit_override": rate_limit_override, "status": "active", "created_at": _now(),
		}
		self.endpoints[record["id"]] = record
		self._emit(tenant, "endpoint_registered", record)
		return deepcopy(record)

	def attach_policy(
		self,
		policy_id: str,
		tenant_id: str,
		api_id: str,
		policy_type: str,
		name: str,
		config: dict[str, Any],
		execution_order: int = 100,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		api = self.apis.get(api_id)
		context = self._base_context(tenant, "attach_policy")
		context.update({
			"api_present": bool(api and api["tenant_id"] == tenant),
			"name_present": bool(name),
			"policy_type_supported": policy_type in SUPPORTED_POLICY_TYPES,
			"config_present": bool(config), "execution_order": execution_order,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("policy", policy_id), "type": "integration_api_policy",
			"kind": "policy", "tenant_id": tenant, "api_id": api_id,
			"policy_type": policy_type, "name": name, "config": deepcopy(config),
			"execution_order": execution_order, "status": "active", "created_at": _now(),
		}
		self.policies[record["id"]] = record
		self._emit(tenant, "policy_attached", record)
		return deepcopy(record)

	def register_consumer(
		self,
		consumer_id: str,
		tenant_id: str,
		name: str,
		contact_email: str,
		owner_id: str,
		external: bool = False,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_consumer")
		context.update({
			"name_present": bool(name), "email_present": bool(contact_email),
			"email_valid": "@" in contact_email and "." in contact_email.rsplit("@", 1)[-1],
			"owner_present": bool(owner_id), "external_consumer": external,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("consumer", consumer_id), "type": "integration_api_consumer",
			"kind": "consumer", "tenant_id": tenant, "name": name, "contact_email": contact_email,
			"owner_id": owner_id, "external": external, "reviewed_by": reviewed_by,
			"status": "active", "created_at": _now(),
		}
		self.consumers[record["id"]] = record
		self._emit(tenant, "consumer_registered", record)
		return deepcopy(record)

	def issue_api_key(
		self,
		key_id: str,
		tenant_id: str,
		consumer_id: str,
		name: str,
		scopes: list[str],
		expires_on: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		consumer = self.consumers.get(consumer_id)
		context = self._base_context(tenant, "issue_api_key")
		context.update({
			"consumer_present": bool(consumer and consumer["tenant_id"] == tenant),
			"name_present": bool(name), "scope_present": bool(scopes),
			"expiration_present": bool(expires_on),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("key", key_id), "type": "integration_api_key",
			"kind": "api_key", "tenant_id": tenant, "consumer_id": consumer_id, "name": name,
			"key_prefix": f"apg_{uuid4().hex[:8]}", "scopes": list(scopes),
			"expires_on": expires_on, "status": "active", "created_at": _now(),
		}
		self.api_keys[record["id"]] = record
		self._emit(tenant, "api_key_issued", record)
		return deepcopy(record)

	def create_subscription(
		self,
		subscription_id: str,
		tenant_id: str,
		consumer_id: str,
		api_id: str,
		plan: str,
		approved_by: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		api = self.apis.get(api_id)
		consumer = self.consumers.get(consumer_id)
		context = self._base_context(tenant, "create_subscription")
		context.update({
			"consumer_present": bool(consumer and consumer["tenant_id"] == tenant),
			"api_present": bool(api and api["tenant_id"] == tenant),
			"plan_supported": plan in SUPPORTED_PLANS, "approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("subscription", subscription_id),
			"type": "integration_api_subscription", "kind": "subscription",
			"tenant_id": tenant, "consumer_id": consumer_id, "api_id": api_id,
			"plan": plan, "approved_by": approved_by, "status": "active", "created_at": _now(),
		}
		self.subscriptions[record["id"]] = record
		self._emit(tenant, "subscription_created", record)
		return deepcopy(record)

	def approve_api(self, api_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		api = self.apis.get(api_id)
		if not api or api["tenant_id"] != tenant:
			raise PermissionError("api_required")
		self._assert_rules({**self._base_context(tenant, "approve_api"),
							 "approver_present": bool(approved_by)})
		api["approved_by"] = approved_by
		api["status"] = "approved"
		api["updated_at"] = _now()
		self._emit(tenant, "api_approved", api)
		return deepcopy(api)

	def deploy_api(
		self,
		deployment_id: str,
		tenant_id: str,
		api_id: str,
		environment: str,
		gateway_route: str,
		deployed_by: str,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		api = self.apis.get(api_id)
		context = self._base_context(tenant, "deploy_api")
		context.update({
			"api_present": bool(api and api["tenant_id"] == tenant),
			"environment_supported": environment in SUPPORTED_ENVIRONMENTS,
			"route_present": bool(gateway_route), "deployer_present": bool(deployed_by),
			"production_environment": environment == "prod",
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("deployment", deployment_id),
			"type": "integration_api_deployment", "kind": "deployment",
			"tenant_id": tenant, "api_id": api_id, "environment": environment,
			"gateway_route": gateway_route, "deployed_by": deployed_by,
			"approved_by": approved_by, "status": "deployed", "created_at": _now(),
		}
		api["status"] = "deployed"
		api["updated_at"] = _now()
		self.deployments[record["id"]] = record
		self._emit(tenant, "api_deployed", record)
		return deepcopy(record)

	def record_usage(
		self,
		usage_id: str,
		tenant_id: str,
		api_id: str,
		consumer_id: str | None,
		endpoint_id: str | None,
		status_code: int,
		latency_ms: int,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		api = self.apis.get(api_id)
		context = self._base_context(tenant, "record_usage")
		context.update({
			"api_present": bool(api and api["tenant_id"] == tenant),
			"status_code_present": status_code is not None,
			"latency_ms": latency_ms, "slow_request": latency_ms >= 2000,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("usage", usage_id), "type": "integration_api_usage",
			"kind": "usage", "tenant_id": tenant, "api_id": api_id,
			"consumer_id": consumer_id, "endpoint_id": endpoint_id, "status_code": status_code,
			"latency_ms": latency_ms, "reviewed_by": reviewed_by, "status": "active",
			"created_at": _now(),
		}
		self.usage_records[record["id"]] = record
		self._emit(tenant, "usage_recorded", record)
		return deepcopy(record)

	def register_api_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_api_agent")
		context.update({
			"agent_runtime_supported": runtime in SUPPORTED_API_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_API_AGENT_ROLES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("agent"), "type": "integration_api_agent", "kind": "agent",
			"tenant_id": tenant, "name": name, "runtime": runtime, "role": role,
			"scope": scope, "status": "active", "created_at": _now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "api_agent_registered", record)
		return deepcopy(record)

	def validate_api_agent_action(self, tenant_id: str, agent_id: str, action: str,
								  privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("api_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant, "tenant_context_present": True,
			"operation": "api_agent_action", "action": action,
			"privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant, "tenant_context_present": True,
			"operation": "api_batch", "event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count,
				"processor": "bytewax", "stream": API_EVENT_STREAM}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None,
					  status: str = "draft") -> dict[str, Any]:
		data = dict(metadata or {})
		record = self.register_api(
			record_id, tenant_id,
			str(data.get("name") or data.get("api_name") or record_id),
			str(data.get("title") or data.get("api_title") or record_id),
			str(data.get("base_path") or f"/{record_id}"),
			str(data.get("upstream_url") or "internal://service"),
			str(data.get("owner_id") or "system"),
			str(data.get("version") or "1.0.0"),
			str(data.get("protocol") or "rest"),
			str(data.get("auth_type") or "api_key"),
			int(data.get("rate_limit_per_minute", 1000)),
			data.get("reviewed_by"),
			{"compatibility_status": status, **data},
		)
		record["status"] = status
		self.apis[record["id"]]["status"] = status
		return record

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		usage = self.list_records("usage_records", tenant)
		return {
			"tenant_id": tenant,
			"api_count": len(self.list_records("apis", tenant)),
			"endpoint_count": len(self.list_records("endpoints", tenant)),
			"policy_count": len(self.list_records("policies", tenant)),
			"consumer_count": len(self.list_records("consumers", tenant)),
			"api_key_count": len(self.list_records("api_keys", tenant)),
			"subscription_count": len(self.list_records("subscriptions", tenant)),
			"deployment_count": len(self.list_records("deployments", tenant)),
			"usage_record_count": len(usage),
			"slow_request_count": len([r for r in usage if r["latency_ms"] >= 2000]),
			"api_agent_count": len(self.list_records("agents", tenant)),
			"audit_event_count": len(self.audit_events(tenant)),
			"overall_status": ("attention_required"
							   if any(r["latency_ms"] >= 2000 for r in usage) else "operating"),
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	def list_records(self, collection: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if collection is None:
			return self.list_all_records(tenant)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(r) for r in store.values() if r["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(r) for r in store if r["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		for col in ["apis", "endpoints", "policies", "consumers", "api_keys",
					"subscriptions", "deployments", "usage_records", "agents"]:
			records.extend(self.list_records(col, tenant))
		return sorted(records, key=lambda r: (r["kind"], r["id"]))

	# ── INTEGRATION LIFECYCLE ─────────────────────────────────────────────────

	async def register_integration(
		self,
		name: str,
		type: str,
		source: str,
		target: str,
		config: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a new integration definition."""
		tenant = self._tenant(tenant_id)
		integration_id = self._record_id("integration")
		record: dict[str, Any] = {
			"id": integration_id, "type": "integration_definition", "kind": "integration",
			"tenant_id": tenant, "name": name, "integration_type": type,
			"source": source, "target": target, "config": deepcopy(config),
			"status": "registered", "created_at": _now(), "updated_at": _now(),
		}
		self._integrations[integration_id] = record
		self._sync_history[integration_id] = []
		self._audit_events.append({
			"tenant_id": tenant, "event_type": "integration_registered",
			"record_id": integration_id, "record_type": "integration_definition",
			"status": "registered", "stream": API_EVENT_STREAM,
			"processor": "bytewax", "emitted_at": _now(),
		})
		return deepcopy(record)

	async def activate_integration(self, integration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Activate a registered integration."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		integration["status"] = "active"
		integration["activated_at"] = _now()
		integration["updated_at"] = _now()
		return deepcopy(integration)

	async def deactivate_integration(
		self,
		integration_id: str,
		reason: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Deactivate an integration with a reason."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		integration["status"] = "inactive"
		integration["deactivation_reason"] = reason
		integration["deactivated_at"] = _now()
		integration["updated_at"] = _now()
		# Cancel any scheduled syncs
		for schedule in self._sync_schedules.values():
			if schedule.get("integration_id") == integration_id:
				schedule["status"] = "cancelled"
		return deepcopy(integration)

	async def test_integration(self, integration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Run a connectivity test for an integration."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		await asyncio.sleep(0)  # simulate async I/O
		# Basic connectivity check against config
		source_reachable = bool(integration.get("source"))
		target_reachable = bool(integration.get("target"))
		healthy = source_reachable and target_reachable
		return {
			"integration_id": integration_id, "test_status": "passed" if healthy else "failed",
			"source_reachable": source_reachable, "target_reachable": target_reachable,
			"latency_ms": 12, "tested_at": _now(),
		}

	async def clone_integration(
		self,
		integration_id: str,
		new_name: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Clone an existing integration under a new name."""
		tenant = self._tenant(tenant_id)
		src = self._integrations.get(integration_id)
		if not src or src["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		new_id = self._record_id("integration")
		clone = {**deepcopy(src), "id": new_id, "name": new_name,
				 "status": "registered", "cloned_from": integration_id,
				 "created_at": _now(), "updated_at": _now()}
		self._integrations[new_id] = clone
		self._sync_history[new_id] = []
		return deepcopy(clone)

	async def export_integration_config(self, integration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Export the full integration configuration as a portable dict."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		export = deepcopy(integration)
		export["exported_at"] = _now()
		export.pop("tenant_id", None)  # strip tenant for portability
		return export

	async def import_integration_config(
		self,
		config_dict: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Import an integration from an exported config dict."""
		tenant = self._tenant(tenant_id)
		new_id = self._record_id("integration")
		record = {
			**deepcopy(config_dict), "id": new_id, "tenant_id": tenant,
			"status": "registered", "imported_at": _now(), "updated_at": _now(),
		}
		record.pop("exported_at", None)
		self._integrations[new_id] = record
		self._sync_history[new_id] = []
		return deepcopy(record)

	# ── SYNC & PROCESSING ─────────────────────────────────────────────────────

	async def sync_now(self, integration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Trigger an immediate synchronisation run."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		if integration["status"] != "active":
			return {"success": False, "error": f"integration_not_active:{integration['status']}"}
		await asyncio.sleep(0)
		run_id = self._record_id("sync")
		run: dict[str, Any] = {
			"id": run_id, "integration_id": integration_id, "tenant_id": tenant,
			"status": "completed", "records_synced": 0, "records_failed": 0,
			"triggered_by": "manual", "started_at": _now(), "completed_at": _now(),
		}
		self._sync_history.setdefault(integration_id, []).append(run)
		integration["last_sync_at"] = _now()
		return deepcopy(run)

	async def schedule_sync(
		self,
		integration_id: str,
		cron_expr: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Schedule recurring sync using a cron expression."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		schedule_id = self._record_id("schedule")
		schedule = {
			"id": schedule_id, "integration_id": integration_id, "tenant_id": tenant,
			"cron_expr": cron_expr, "status": "active", "created_at": _now(),
		}
		self._sync_schedules[schedule_id] = schedule
		integration["sync_schedule_id"] = schedule_id
		return deepcopy(schedule)

	async def cancel_scheduled_sync(self, integration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel any active sync schedule for an integration."""
		tenant = self._tenant(tenant_id)
		cancelled = []
		for sid, schedule in self._sync_schedules.items():
			if (schedule.get("integration_id") == integration_id
					and schedule["tenant_id"] == tenant and schedule["status"] == "active"):
				schedule["status"] = "cancelled"
				schedule["cancelled_at"] = _now()
				cancelled.append(sid)
		integration = self._integrations.get(integration_id)
		if integration and integration["tenant_id"] == tenant:
			integration.pop("sync_schedule_id", None)
		return {"integration_id": integration_id, "cancelled_schedules": cancelled,
				"count": len(cancelled)}

	async def sync_history(
		self,
		integration_id: str,
		limit: int = 20,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return recent sync run history for an integration."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		history = self._sync_history.get(integration_id, [])
		return [deepcopy(h) for h in history[-limit:]]

	async def retry_failed_records(self, integration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retry all failed records from the last sync run."""
		tenant = self._tenant(tenant_id)
		history = self._sync_history.get(integration_id, [])
		if not history:
			return {"retried": 0, "error": "no_sync_history"}
		last_run = history[-1]
		failed_count = int(last_run.get("records_failed", 0))
		if failed_count == 0:
			return {"retried": 0, "message": "no_failed_records"}
		await asyncio.sleep(0)
		retry_run_id = self._record_id("sync")
		retry_run: dict[str, Any] = {
			"id": retry_run_id, "integration_id": integration_id, "tenant_id": tenant,
			"status": "completed", "records_synced": failed_count, "records_failed": 0,
			"triggered_by": "retry", "started_at": _now(), "completed_at": _now(),
		}
		history.append(retry_run)
		return {"retried": failed_count, "run_id": retry_run_id}

	async def bulk_sync(
		self,
		integration_ids: list[str],
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Trigger sync for multiple integrations concurrently."""
		results = await asyncio.gather(
			*[self.sync_now(iid, tenant_id) for iid in integration_ids],
			return_exceptions=True,
		)
		return [{"integration_id": iid,
				 "result": r if not isinstance(r, Exception) else {"error": str(r)}}
				for iid, r in zip(integration_ids, results)]

	async def sync_preview(
		self,
		integration_id: str,
		limit: int = 10,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Preview records that would be synced without actually syncing."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		# Stub preview: in production this would fetch from source adapter
		preview_records = [
			{"record_id": f"preview_{i}", "source": integration["source"],
			 "preview_only": True, "data": {}} for i in range(min(limit, 5))
		]
		return {"integration_id": integration_id, "preview_count": len(preview_records),
				"limit": limit, "records": preview_records, "ts": _now()}

	# ── TRANSFORMATION ────────────────────────────────────────────────────────

	async def create_mapping(
		self,
		name: str,
		source_schema: dict[str, Any],
		target_schema: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a field mapping between source and target schemas."""
		tenant = self._tenant(tenant_id)
		mapping_id = self._record_id("mapping")
		auto_rules: dict[str, str] = {}
		for field in source_schema:
			if field in target_schema:
				auto_rules[field] = field
		record: dict[str, Any] = {
			"id": mapping_id, "name": name, "tenant_id": tenant,
			"source_schema": deepcopy(source_schema), "target_schema": deepcopy(target_schema),
			"rules": auto_rules, "status": "active", "created_at": _now(),
		}
		self._mappings[mapping_id] = record
		return deepcopy(record)

	async def apply_mapping(
		self,
		integration_id: str,
		record: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Apply the integration's mapping to transform a single record."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		mapping_id = integration.get("mapping_id")
		if not mapping_id:
			return {"transformed": deepcopy(record), "mapping_applied": False}
		mapping = self._mappings.get(mapping_id, {})
		rules = mapping.get("rules", {})
		transforms = self._field_transforms
		transformed: dict[str, Any] = {}
		for src_field, dst_field in rules.items():
			value = record.get(src_field)
			transform_key = f"{integration_id}:{src_field}"
			if transform_key in transforms:
				t = transforms[transform_key]
				value = self._apply_transform(value, t.get("rule_type", "passthrough"), t.get("params", {}))
			transformed[dst_field] = value
		return {"transformed": transformed, "mapping_applied": True, "mapping_id": mapping_id}

	async def validate_mapping(
		self,
		mapping_id: str,
		test_record: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Validate a mapping by applying it to a test record and checking schema conformance."""
		tenant = self._tenant(tenant_id)
		mapping = self._mappings.get(mapping_id)
		if not mapping or mapping["tenant_id"] != tenant:
			raise KeyError(f"mapping_not_found:{mapping_id}")
		rules = mapping.get("rules", {})
		target_schema = mapping.get("target_schema", {})
		transformed: dict[str, Any] = {dst: test_record.get(src) for src, dst in rules.items()}
		missing = [f for f in target_schema if f not in transformed]
		extra = [f for f in transformed if f not in target_schema]
		return {
			"mapping_id": mapping_id, "valid": len(missing) == 0,
			"transformed": transformed, "missing_target_fields": missing,
			"extra_fields": extra, "test_record": test_record,
		}

	async def field_transform_rule(
		self,
		field: str,
		rule_type: str,
		params: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a transformation rule for a field."""
		tenant = self._tenant(tenant_id)
		rule_id = self._record_id("transform")
		rule = {
			"id": rule_id, "field": field, "rule_type": rule_type,
			"params": deepcopy(params), "tenant_id": tenant, "created_at": _now(),
		}
		key = f"{tenant}:{field}"
		self._field_transforms[key] = rule
		return deepcopy(rule)

	async def mapping_test(
		self,
		mapping_id: str,
		sample_data: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Apply a mapping to a batch of sample records for validation."""
		tenant = self._tenant(tenant_id)
		mapping = self._mappings.get(mapping_id)
		if not mapping or mapping["tenant_id"] != tenant:
			raise KeyError(f"mapping_not_found:{mapping_id}")
		results = []
		for sample in sample_data:
			rules = mapping.get("rules", {})
			transformed = {dst: sample.get(src) for src, dst in rules.items()}
			results.append({"input": sample, "output": transformed})
		return {"mapping_id": mapping_id, "sample_count": len(sample_data),
				"results": results, "ts": _now()}

	# ── WEBHOOKS ─────────────────────────────────────────────────────────────

	async def register_webhook(
		self,
		integration_id: str,
		url: str,
		events: list[str],
		secret: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a webhook endpoint for an integration."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		webhook_id = self._record_id("webhook")
		webhook = {
			"id": webhook_id, "integration_id": integration_id, "tenant_id": tenant,
			"url": url, "events": list(events), "secret_hash": hashlib.sha256(secret.encode()).hexdigest(),
			"status": "active", "created_at": _now(),
		}
		self._webhooks[webhook_id] = webhook
		self._webhook_history[webhook_id] = []
		return {**deepcopy(webhook), "secret_hash": webhook["secret_hash"]}  # never return secret

	async def test_webhook(self, webhook_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Send a test ping to a webhook endpoint."""
		tenant = self._tenant(tenant_id)
		webhook = self._webhooks.get(webhook_id)
		if not webhook or webhook["tenant_id"] != tenant:
			raise KeyError(f"webhook_not_found:{webhook_id}")
		await asyncio.sleep(0)
		# Simulate HTTP delivery
		delivery_id = self._record_id("delivery")
		delivery: dict[str, Any] = {
			"id": delivery_id, "webhook_id": webhook_id, "event_type": "webhook.test",
			"status_code": 200, "response_ms": 45, "delivered_at": _now(),
		}
		self._webhook_history.setdefault(webhook_id, []).append(delivery)
		return {"webhook_id": webhook_id, "test_status": "delivered",
				"delivery": delivery}

	async def webhook_history(
		self,
		webhook_id: str,
		limit: int = 20,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return delivery history for a webhook."""
		tenant = self._tenant(tenant_id)
		webhook = self._webhooks.get(webhook_id)
		if not webhook or webhook["tenant_id"] != tenant:
			raise KeyError(f"webhook_not_found:{webhook_id}")
		history = self._webhook_history.get(webhook_id, [])
		return [deepcopy(h) for h in history[-limit:]]

	async def disable_webhook(self, webhook_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Disable a webhook endpoint."""
		tenant = self._tenant(tenant_id)
		webhook = self._webhooks.get(webhook_id)
		if not webhook or webhook["tenant_id"] != tenant:
			raise KeyError(f"webhook_not_found:{webhook_id}")
		webhook["status"] = "disabled"
		webhook["disabled_at"] = _now()
		return {"webhook_id": webhook_id, "status": "disabled", "ts": _now()}

	# ── MONITORING ────────────────────────────────────────────────────────────

	async def error_report(
		self,
		integration_id: str,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Produce an error summary report for an integration."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		history = self._sync_history.get(integration_id, [])
		total_runs = len(history)
		total_failed = sum(int(r.get("records_failed", 0)) for r in history)
		failed_runs = sum(1 for r in history if r.get("status") == "failed")
		usage = self.list_records("usage_records", tenant)
		error_codes = [u["status_code"] for u in usage
					   if u.get("api_id") in {a["id"] for a in self.list_records("apis", tenant)}
					   and u["status_code"] >= 400]
		from collections import Counter
		error_dist = dict(Counter(error_codes).most_common(10))
		return {
			"integration_id": integration_id, "period": period,
			"total_sync_runs": total_runs, "failed_sync_runs": failed_runs,
			"total_failed_records": total_failed,
			"error_rate": failed_runs / total_runs if total_runs else 0.0,
			"api_error_distribution": error_dist, "ts": _now(),
		}

	async def data_quality_report(
		self,
		integration_id: str,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Report data quality metrics for an integration's sync history."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		history = self._sync_history.get(integration_id, [])
		total_synced = sum(int(r.get("records_synced", 0)) for r in history)
		total_failed = sum(int(r.get("records_failed", 0)) for r in history)
		total_processed = total_synced + total_failed
		quality_score = total_synced / total_processed if total_processed else 1.0
		return {
			"integration_id": integration_id, "period": period,
			"total_records_processed": total_processed,
			"records_synced_ok": total_synced, "records_failed": total_failed,
			"quality_score": round(quality_score, 4),
			"grade": "A" if quality_score >= 0.99 else ("B" if quality_score >= 0.95 else "C"),
			"ts": _now(),
		}

	async def integration_dashboard(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return unified integration status dashboard."""
		tenant = self._tenant(tenant_id)
		integrations = [i for i in self._integrations.values() if i["tenant_id"] == tenant]
		active = sum(1 for i in integrations if i["status"] == "active")
		inactive = sum(1 for i in integrations if i["status"] == "inactive")
		webhooks = [w for w in self._webhooks.values() if w["tenant_id"] == tenant]
		schedules = [s for s in self._sync_schedules.values() if s["tenant_id"] == tenant]
		return {
			"tenant_id": tenant,
			"total_integrations": len(integrations),
			"active_integrations": active, "inactive_integrations": inactive,
			"registered_webhooks": len(webhooks),
			"active_sync_schedules": sum(1 for s in schedules if s["status"] == "active"),
			"mappings": len(self._mappings),
			"ts": _now(),
		}

	async def api_health_check(self, integration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Check health of all APIs involved in an integration."""
		tenant = self._tenant(tenant_id)
		integration = self._integrations.get(integration_id)
		if not integration or integration["tenant_id"] != tenant:
			raise KeyError(f"integration_not_found:{integration_id}")
		await asyncio.sleep(0)
		apis = self.list_records("apis", tenant)
		health_checks = [{"api_id": a["id"], "name": a["name"],
						  "status": a["status"],
						  "healthy": a["status"] in ("active", "deployed", "approved")}
						 for a in apis]
		all_healthy = all(h["healthy"] for h in health_checks)
		return {
			"integration_id": integration_id, "overall_health": "healthy" if all_healthy else "degraded",
			"api_health_checks": health_checks, "checked_at": _now(),
		}

	# ── private helpers ───────────────────────────────────────────────────────

	@staticmethod
	def _apply_transform(value: Any, rule_type: str, params: dict[str, Any]) -> Any:
		"""Apply a single field transform rule."""
		if rule_type == "uppercase" and isinstance(value, str):
			return value.upper()
		if rule_type == "lowercase" and isinstance(value, str):
			return value.lower()
		if rule_type == "default" and value is None:
			return params.get("default")
		if rule_type == "cast":
			target_type = params.get("type", "str")
			try:
				return {"int": int, "float": float, "str": str, "bool": bool}[target_type](value)
			except (ValueError, KeyError, TypeError):
				return value
		if rule_type == "prefix" and isinstance(value, str):
			return params.get("prefix", "") + value
		if rule_type == "suffix" and isinstance(value, str):
			return value + params.get("suffix", "")
		return value  # passthrough


APILifecycleService = IntApiService
ConsumerManagementService = IntApiService
PolicyManagementService = IntApiService
AnalyticsService = IntApiService
APIService = IntApiService
