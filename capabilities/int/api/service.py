"""Dependency-light Integration API Management lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
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
except ImportError:  # pragma: no cover - supports direct file loading in tests
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

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
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

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": API_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

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
			"name_present": bool(name),
			"title_present": bool(title),
			"base_path_present": bool(base_path),
			"base_path_valid": bool(base_path and base_path.startswith("/")),
			"upstream_present": bool(upstream_url),
			"owner_present": bool(owner_id),
			"protocol_supported": protocol in SUPPORTED_PROTOCOLS,
			"auth_type_supported": auth_type in SUPPORTED_AUTH_TYPES,
			"rate_limit": rate_limit_per_minute,
			"external_upstream": external_upstream,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("api", api_id),
			"type": "integration_api",
			"kind": "api",
			"tenant_id": tenant,
			"name": name,
			"title": title,
			"base_path": base_path,
			"upstream_url": upstream_url,
			"owner_id": owner_id,
			"version": version,
			"protocol": protocol,
			"auth_type": auth_type,
			"rate_limit_per_minute": rate_limit_per_minute,
			"reviewed_by": reviewed_by,
			"approved_by": None,
			"metadata": deepcopy(metadata or {}),
			"status": "draft",
			"created_at": self._now(),
			"updated_at": self._now(),
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
			"id": self._record_id("endpoint", endpoint_id),
			"type": "integration_api_endpoint",
			"kind": "endpoint",
			"tenant_id": tenant,
			"api_id": api_id,
			"path": path,
			"method": method,
			"auth_required": auth_required,
			"rate_limit_override": rate_limit_override,
			"status": "active",
			"created_at": self._now(),
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
			"config_present": bool(config),
			"execution_order": execution_order,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("policy", policy_id),
			"type": "integration_api_policy",
			"kind": "policy",
			"tenant_id": tenant,
			"api_id": api_id,
			"policy_type": policy_type,
			"name": name,
			"config": deepcopy(config),
			"execution_order": execution_order,
			"status": "active",
			"created_at": self._now(),
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
			"name_present": bool(name),
			"email_present": bool(contact_email),
			"email_valid": "@" in contact_email and "." in contact_email.rsplit("@", 1)[-1],
			"owner_present": bool(owner_id),
			"external_consumer": external,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("consumer", consumer_id),
			"type": "integration_api_consumer",
			"kind": "consumer",
			"tenant_id": tenant,
			"name": name,
			"contact_email": contact_email,
			"owner_id": owner_id,
			"external": external,
			"reviewed_by": reviewed_by,
			"status": "active",
			"created_at": self._now(),
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
			"name_present": bool(name),
			"scope_present": bool(scopes),
			"expiration_present": bool(expires_on),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("key", key_id),
			"type": "integration_api_key",
			"kind": "api_key",
			"tenant_id": tenant,
			"consumer_id": consumer_id,
			"name": name,
			"key_prefix": f"apg_{uuid4().hex[:8]}",
			"scopes": list(scopes),
			"expires_on": expires_on,
			"status": "active",
			"created_at": self._now(),
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
			"plan_supported": plan in SUPPORTED_PLANS,
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("subscription", subscription_id),
			"type": "integration_api_subscription",
			"kind": "subscription",
			"tenant_id": tenant,
			"consumer_id": consumer_id,
			"api_id": api_id,
			"plan": plan,
			"approved_by": approved_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.subscriptions[record["id"]] = record
		self._emit(tenant, "subscription_created", record)
		return deepcopy(record)

	def approve_api(self, api_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		api = self.apis.get(api_id)
		if not api or api["tenant_id"] != tenant:
			raise PermissionError("api_required")
		self._assert_rules({
			**self._base_context(tenant, "approve_api"),
			"approver_present": bool(approved_by),
		})
		api["approved_by"] = approved_by
		api["status"] = "approved"
		api["updated_at"] = self._now()
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
			"route_present": bool(gateway_route),
			"deployer_present": bool(deployed_by),
			"production_environment": environment == "prod",
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("deployment", deployment_id),
			"type": "integration_api_deployment",
			"kind": "deployment",
			"tenant_id": tenant,
			"api_id": api_id,
			"environment": environment,
			"gateway_route": gateway_route,
			"deployed_by": deployed_by,
			"approved_by": approved_by,
			"status": "deployed",
			"created_at": self._now(),
		}
		api["status"] = "deployed"
		api["updated_at"] = self._now()
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
			"latency_ms": latency_ms,
			"slow_request": latency_ms >= 2000,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("usage", usage_id),
			"type": "integration_api_usage",
			"kind": "usage",
			"tenant_id": tenant,
			"api_id": api_id,
			"consumer_id": consumer_id,
			"endpoint_id": endpoint_id,
			"status_code": status_code,
			"latency_ms": latency_ms,
			"reviewed_by": reviewed_by,
			"status": "active",
			"created_at": self._now(),
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
			"id": self._record_id("agent"),
			"type": "integration_api_agent",
			"kind": "agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "api_agent_registered", record)
		return deepcopy(record)

	def validate_api_agent_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("api_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "api_agent_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "api_batch",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": API_EVENT_STREAM}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "draft") -> dict[str, Any]:
		data = dict(metadata or {})
		record = self.register_api(
			record_id,
			tenant_id,
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
			"slow_request_count": len([record for record in usage if record["latency_ms"] >= 2000]),
			"api_agent_count": len(self.list_records("agents", tenant)),
			"audit_event_count": len(self.audit_events(tenant)),
			"overall_status": "attention_required" if any(record["latency_ms"] >= 2000 for record in usage) else "operating",
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if collection is None:
			return self.list_all_records(tenant)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		for collection in ["apis", "endpoints", "policies", "consumers", "api_keys", "subscriptions", "deployments", "usage_records", "agents"]:
			records.extend(self.list_records(collection, tenant))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))


APILifecycleService = IntApiService
ConsumerManagementService = IntApiService
PolicyManagementService = IntApiService
AnalyticsService = IntApiService
APIService = IntApiService
