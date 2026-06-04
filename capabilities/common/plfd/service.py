"""Service layer for executable Platform Foundation management — expanded implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_PLFD_AGENT_ROLES,
	SUPPORTED_PLFD_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
)
from .foundation_runtime import (
	change_review_status,
	dependencies_are_healthy,
	normalize_baseline_type,
	normalize_health,
	normalize_score,
	normalize_tier,
	readiness_posture,
	service_baselines_complete,
	stable_id,
)
from .models import (
	FoundationBaseline,
	FoundationDependency,
	FoundationService,
	PlatformChange,
	PlfdAgent,
	PlfdAuditEvent,
	ReadinessAssessment,
	utc_now,
)


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"


class PlatformFoundationService:
	"""
	In-process foundation-service registry, dependency graph, baseline,
	readiness, change gate, feature flags, circuit breakers,
	service discovery, rate limiters, and platform metrics.

	Adapter/store pattern — no external dependencies.
	"""

	def __init__(self) -> None:
		self._services: dict[str, FoundationService] = {}
		self._dependencies: dict[str, FoundationDependency] = {}
		self._baselines: dict[str, FoundationBaseline] = {}
		self._assessments: dict[str, ReadinessAssessment] = {}
		self._changes: dict[str, PlatformChange] = {}
		self._audit_events: dict[str, PlfdAuditEvent] = {}
		self._agents: dict[str, PlfdAgent] = {}
		# New stores
		self._platform_configs: dict[str, dict[str, Any]] = {}
		self._feature_flags: dict[str, dict[str, Any]] = {}
		self._circuit_breakers: dict[str, dict[str, Any]] = {}
		self._service_registry: dict[str, dict[str, Any]] = {}
		self._rate_limiters: dict[str, dict[str, Any]] = {}
		self._metrics_snapshots: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Contract / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# health_check_all_services
	# ------------------------------------------------------------------

	def health_check_all_services(
		self,
		tenant_id: str = "default",
		include_dependencies: bool = True,
	) -> dict[str, Any]:
		"""
		Run health checks across all registered foundation services.

		Returns per-service health status, dependency health, and overall
		platform health posture.
		"""
		services = self.list_services(tenant_id)
		dependencies = self.list_dependencies(tenant_id)
		service_results: list[dict[str, Any]] = []
		for svc in services:
			svc_id = svc["id"]
			svc_deps = [d for d in dependencies if d["source_service_id"] == svc_id]
			unhealthy_deps = [d for d in svc_deps if d.get("required") and d["health_status"] != "healthy"]
			health = svc["health_status"]
			if unhealthy_deps and include_dependencies:
				health = "degraded"
			service_results.append({
				"service_id": svc_id,
				"service_name": svc["name"],
				"tier": svc["tier"],
				"health_status": health,
				"dependency_count": len(svc_deps),
				"unhealthy_dependency_count": len(unhealthy_deps),
				"monitoring_enabled": svc.get("monitoring_enabled", False),
			})
		healthy = sum(1 for s in service_results if s["health_status"] == "healthy")
		degraded = sum(1 for s in service_results if s["health_status"] == "degraded")
		unhealthy = sum(1 for s in service_results if s["health_status"] not in {"healthy", "degraded"})
		overall = "healthy" if unhealthy == 0 and degraded == 0 else ("degraded" if unhealthy == 0 else "unhealthy")
		result = {
			"tenant_id": tenant_id,
			"overall": overall,
			"service_count": len(service_results),
			"healthy_count": healthy,
			"degraded_count": degraded,
			"unhealthy_count": unhealthy,
			"services": service_results,
			"checked_at": _ts(),
		}
		self._record_audit(tenant_id, "health_check_all", "platform", "platform_health_checked", "allow")
		return result

	def platform_configuration(
		self,
		key: str,
		value: Any,
		environment: str,
		tenant_id: str = "default",
		data_type: str = "string",
		description: str = "",
		set_by: str = "system",
	) -> dict[str, Any]:
		"""
		Set or update a platform configuration key for an environment.

		key: Dot-notation config key (e.g. 'db.max_connections').
		value: Config value (any JSON-serialisable type).
		environment: Target environment (development | staging | production).
		data_type: string | int | float | bool | json.
		"""
		if not key:
			raise ValueError("platform_config_key_required")
		if not environment:
			raise ValueError("platform_config_environment_required")
		supported_envs = {"development", "test", "staging", "production"}
		if environment not in supported_envs:
			raise ValueError(f"unsupported_environment:{environment}")
		config_key = f"{tenant_id}:{environment}:{key}"
		existing = self._platform_configs.get(config_key)
		version = (existing["version"] + 1) if existing else 1
		record = {
			"key": key,
			"value": value,
			"environment": environment,
			"tenant_id": tenant_id,
			"data_type": data_type,
			"description": description,
			"version": version,
			"set_by": set_by,
			"updated_at": _ts(),
		}
		self._platform_configs[config_key] = record
		self._record_audit(tenant_id, key, environment, "platform_config_set", "allow",
			reasons=(), metadata={"key": key, "environment": environment, "version": version})
		return record

	def feature_flag_set(
		self,
		flag_name: str,
		enabled: bool,
		conditions: dict[str, Any],
		tenant_id: str = "default",
		description: str = "",
		set_by: str = "system",
		rollout_percentage: float = 100.0,
	) -> dict[str, Any]:
		"""
		Create or update a feature flag with optional rollout conditions.

		conditions: Dict supporting keys: tenant_ids, user_groups, regions, custom.
		rollout_percentage: 0-100 float for gradual rollout.
		"""
		if not flag_name:
			raise ValueError("feature_flag_name_required")
		if not 0.0 <= rollout_percentage <= 100.0:
			raise ValueError("rollout_percentage_must_be_0_to_100")
		flag_key = f"{tenant_id}:{flag_name}"
		existing = self._feature_flags.get(flag_key)
		version = (existing["version"] + 1) if existing else 1
		record = {
			"flag_name": flag_name,
			"tenant_id": tenant_id,
			"enabled": bool(enabled),
			"conditions": dict(conditions),
			"rollout_percentage": rollout_percentage,
			"description": description,
			"version": version,
			"set_by": set_by,
			"updated_at": _ts(),
		}
		self._feature_flags[flag_key] = record
		self._record_audit(tenant_id, flag_name, "feature_flag", "feature_flag_set", "allow",
			metadata={"enabled": enabled, "rollout_percentage": rollout_percentage})
		return record

	def feature_flag_check(
		self,
		flag_name: str,
		context: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Evaluate a feature flag for a given context.

		context: Dict with optional keys: user_id, tenant_id, region, user_group.
		Returns whether the flag is enabled for this context.
		"""
		flag_key = f"{tenant_id}:{flag_name}"
		flag = self._feature_flags.get(flag_key)
		if flag is None:
			return {
				"flag_name": flag_name,
				"tenant_id": tenant_id,
				"enabled": False,
				"reason": "flag_not_found",
				"context": context,
			}
		if not flag["enabled"]:
			return {
				"flag_name": flag_name,
				"tenant_id": tenant_id,
				"enabled": False,
				"reason": "flag_disabled",
				"context": context,
			}
		# Check rollout percentage (deterministic: hash user_id mod 100)
		user_id = str(context.get("user_id", ""))
		if user_id and flag["rollout_percentage"] < 100.0:
			import hashlib
			hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
			if hash_val >= flag["rollout_percentage"]:
				return {"flag_name": flag_name, "tenant_id": tenant_id, "enabled": False, "reason": "outside_rollout", "context": context}
		# Check conditions
		conditions = flag.get("conditions", {})
		allowed_tenants = conditions.get("tenant_ids", [])
		if allowed_tenants and context.get("tenant_id") not in allowed_tenants:
			return {"flag_name": flag_name, "tenant_id": tenant_id, "enabled": False, "reason": "tenant_condition_mismatch", "context": context}
		allowed_regions = conditions.get("regions", [])
		if allowed_regions and context.get("region") not in allowed_regions:
			return {"flag_name": flag_name, "tenant_id": tenant_id, "enabled": False, "reason": "region_condition_mismatch", "context": context}
		return {
			"flag_name": flag_name,
			"tenant_id": tenant_id,
			"enabled": True,
			"reason": "conditions_met",
			"context": context,
			"rollout_percentage": flag["rollout_percentage"],
		}

	def circuit_breaker_status(
		self,
		service_name: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return the current state and metrics of a circuit breaker for a service."""
		cb_key = f"{tenant_id}:{service_name}"
		cb = self._circuit_breakers.get(cb_key)
		if cb is None:
			# Auto-initialise in closed state
			cb = {
				"service_name": service_name,
				"tenant_id": tenant_id,
				"state": "closed",
				"failure_count": 0,
				"success_count": 0,
				"failure_threshold": 5,
				"recovery_timeout_seconds": 60,
				"last_failure_at": None,
				"opened_at": None,
				"reset_at": None,
			}
			self._circuit_breakers[cb_key] = cb
		return dict(cb)

	def circuit_breaker_reset(
		self,
		service_name: str,
		approved_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Manually reset a circuit breaker to closed state.

		Requires an approver identity.
		"""
		if not approved_by:
			raise PermissionError("circuit_breaker_reset_approval_required")
		cb_key = f"{tenant_id}:{service_name}"
		cb = self._circuit_breakers.get(cb_key, {
			"service_name": service_name,
			"tenant_id": tenant_id,
			"state": "open",
			"failure_count": 0,
			"success_count": 0,
			"failure_threshold": 5,
			"recovery_timeout_seconds": 60,
			"last_failure_at": None,
			"opened_at": None,
		})
		cb["state"] = "closed"
		cb["failure_count"] = 0
		cb["reset_at"] = _ts()
		cb["reset_by"] = approved_by
		self._circuit_breakers[cb_key] = cb
		self._record_audit(tenant_id, service_name, "circuit_breaker", "circuit_breaker_reset", "allow",
			metadata={"approved_by": approved_by})
		return dict(cb)

	def dependency_graph(
		self,
		tenant_id: str = "default",
		include_health: bool = True,
	) -> dict[str, Any]:
		"""
		Return the full dependency graph for all registered foundation services.

		Returns nodes (services) and edges (dependencies) in graph format
		suitable for visualisation.
		"""
		services = self.list_services(tenant_id)
		dependencies = self.list_dependencies(tenant_id)
		nodes = [
			{
				"id": s["id"],
				"name": s["name"],
				"tier": s["tier"],
				"health_status": s["health_status"] if include_health else None,
				"status": s["status"],
			}
			for s in services
		]
		edges = [
			{
				"source": d["source_service_id"],
				"target": d["target_service_id"],
				"required": d["required"],
				"health_status": d["health_status"] if include_health else None,
			}
			for d in dependencies
		]
		cycles: list[list[str]] = []
		# Detect simple cycles via DFS
		adj: dict[str, list[str]] = {n["id"]: [] for n in nodes}
		for e in edges:
			adj[e["source"]].append(e["target"])
		visited: set[str] = set()
		rec_stack: set[str] = set()
		def _dfs(v: str, path: list[str]) -> None:
			visited.add(v)
			rec_stack.add(v)
			for neighbor in adj.get(v, []):
				if neighbor not in visited:
					_dfs(neighbor, path + [neighbor])
				elif neighbor in rec_stack:
					cycles.append(path + [neighbor])
			rec_stack.discard(v)
		for node in nodes:
			if node["id"] not in visited:
				_dfs(node["id"], [node["id"]])
		return {
			"tenant_id": tenant_id,
			"nodes": nodes,
			"edges": edges,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"cycle_count": len(cycles),
			"cycles": cycles,
			"generated_at": _ts(),
		}

	def service_discovery_register(
		self,
		service_name: str,
		endpoint: str,
		metadata: dict[str, Any],
		tenant_id: str = "default",
		health_check_url: str = "",
		version: str = "1.0.0",
		registered_by: str = "system",
	) -> dict[str, Any]:
		"""
		Register a service in the platform service discovery catalogue.

		endpoint: Base URL or address.
		metadata: Arbitrary key-value service metadata.
		"""
		if not service_name:
			raise ValueError("service_name_required")
		if not endpoint:
			raise ValueError("service_endpoint_required")
		reg_key = f"{tenant_id}:{service_name}"
		record = {
			"service_name": service_name,
			"tenant_id": tenant_id,
			"endpoint": endpoint,
			"metadata": dict(metadata),
			"health_check_url": health_check_url or f"{endpoint}/health",
			"version": version,
			"registered_by": registered_by,
			"status": "active",
			"registered_at": _ts(),
		}
		self._service_registry[reg_key] = record
		self._record_audit(tenant_id, service_name, "service_registry", "service_discovery_registered", "allow",
			metadata={"endpoint": endpoint, "version": version})
		return record

	def rate_limiter_configure(
		self,
		service_name: str,
		limit: int,
		window: str,
		tenant_id: str = "default",
		strategy: str = "token_bucket",
		burst_limit: int | None = None,
		configured_by: str = "system",
	) -> dict[str, Any]:
		"""
		Configure a rate limiter for a service.

		limit: Maximum requests per window.
		window: Time window label (e.g. '1s', '1m', '1h').
		strategy: 'token_bucket' | 'sliding_window' | 'fixed_window' | 'leaky_bucket'.
		"""
		if not service_name:
			raise ValueError("service_name_required")
		if limit < 1:
			raise ValueError("rate_limit_must_be_positive")
		supported_strategies = {"token_bucket", "sliding_window", "fixed_window", "leaky_bucket"}
		if strategy not in supported_strategies:
			raise ValueError(f"unsupported_rate_limit_strategy:{strategy}")
		rl_key = f"{tenant_id}:{service_name}"
		record = {
			"service_name": service_name,
			"tenant_id": tenant_id,
			"limit": limit,
			"window": window,
			"strategy": strategy,
			"burst_limit": burst_limit or int(limit * 1.5),
			"configured_by": configured_by,
			"updated_at": _ts(),
		}
		self._rate_limiters[rl_key] = record
		self._record_audit(tenant_id, service_name, "rate_limiter", "rate_limiter_configured", "allow",
			metadata={"limit": limit, "window": window, "strategy": strategy})
		return record

	def platform_metrics_dashboard(
		self,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return a comprehensive platform metrics dashboard snapshot.

		Aggregates service health, feature flags, circuit breakers,
		rate limiters, configuration count, and discovery registrations.
		"""
		services = self.list_services(tenant_id)
		dependencies = self.list_dependencies(tenant_id)
		changes = self.list_changes(tenant_id)
		tenant_flags = [v for v in self._feature_flags.values() if v["tenant_id"] == tenant_id]
		tenant_cbs = [v for v in self._circuit_breakers.values() if v["tenant_id"] == tenant_id]
		tenant_rls = [v for v in self._rate_limiters.values() if v["tenant_id"] == tenant_id]
		tenant_registry = [v for v in self._service_registry.values() if v["tenant_id"] == tenant_id]
		tenant_configs = [v for v in self._platform_configs.values() if v["tenant_id"] == tenant_id]
		healthy_services = [s for s in services if s["health_status"] == "healthy"]
		open_cbs = [cb for cb in tenant_cbs if cb["state"] == "open"]
		snapshot = {
			"tenant_id": tenant_id,
			"service_count": len(services),
			"healthy_service_count": len(healthy_services),
			"unhealthy_service_count": len(services) - len(healthy_services),
			"dependency_count": len(dependencies),
			"unhealthy_dependency_count": sum(1 for d in dependencies if d.get("required") and d["health_status"] != "healthy"),
			"pending_change_count": sum(1 for c in changes if c["status"] != "approved"),
			"approved_change_count": sum(1 for c in changes if c["status"] == "approved"),
			"feature_flag_count": len(tenant_flags),
			"enabled_feature_flag_count": sum(1 for f in tenant_flags if f["enabled"]),
			"circuit_breaker_count": len(tenant_cbs),
			"open_circuit_breaker_count": len(open_cbs),
			"rate_limiter_count": len(tenant_rls),
			"service_registry_count": len(tenant_registry),
			"platform_config_count": len(tenant_configs),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"snapshot_at": _ts(),
		}
		self._metrics_snapshots.append({**snapshot})
		return snapshot

	# ------------------------------------------------------------------
	# Original retained methods
	# ------------------------------------------------------------------

	def register_foundation_service(
		self,
		service_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		tier: str,
		dependencies: list[str] | None = None,
		readiness_score: float = 0.0,
		configuration_baseline_present: bool = True,
		health_status: str = "healthy",
		monitoring_enabled: bool = False,
		rollback_plan_ref: str = "",
		change_window_ref: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_foundation_service",
			"service_owner_assigned": bool(owner),
			"tier_classified": bool(tier),
			"readiness_score_present": readiness_score is not None,
			"configuration_baseline_present": bool(configuration_baseline_present),
		})
		self._raise_if_denied(result)
		service = FoundationService(
			id=service_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			tier=normalize_tier(tier),
			dependencies=tuple(str(item) for item in dependencies or ()),
			readiness_score=normalize_score(readiness_score),
			configuration_baseline_present=bool(configuration_baseline_present),
			health_status=normalize_health(health_status),
			monitoring_enabled=bool(monitoring_enabled),
			rollback_plan_ref=rollback_plan_ref,
			change_window_ref=change_window_ref,
			status="registered",
			metadata=dict(metadata or {}),
		)
		self._services[_state_key(tenant_id, service.id)] = service
		self._record_audit(tenant_id, service.id, "foundation_service_registered", owner, "allow")
		return service.to_dict()

	def record_dependency(self, dependency_id: str, tenant_id: str, source_service_id: str, target_service_id: str, health_status: str = "healthy", required: bool = True, evidence_ref: str = "") -> dict[str, Any]:
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "record_dependency", "dependency_evidence_present": bool(evidence_ref)})
		self._raise_if_denied(result)
		self._require_service(source_service_id, tenant_id)
		self._require_service(target_service_id, tenant_id)
		dependency = FoundationDependency(id=dependency_id, tenant_id=tenant_id, source_service_id=source_service_id, target_service_id=target_service_id, health_status=normalize_health(health_status), required=bool(required), evidence_ref=evidence_ref)
		self._dependencies[_state_key(tenant_id, dependency.id)] = dependency
		self._record_audit(tenant_id, dependency.id, "dependency_recorded", "plfd", result["decision"], reasons=self._reasons(result))
		return dependency.to_dict()

	def attach_baseline(self, baseline_id: str, tenant_id: str, service_id: str, baseline_type: str, evidence_ref: str, approved_by: str, status: str = "approved") -> dict[str, Any]:
		service = self._require_service(service_id, tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "attach_baseline", "baseline_evidence_present": bool(evidence_ref), "baseline_approver_present": bool(approved_by)})
		self._raise_if_denied(result)
		if status not in {"approved", "draft", "rejected"}:
			raise ValueError("baseline_status_invalid")
		baseline = FoundationBaseline(id=baseline_id, tenant_id=tenant_id, service_id=service_id, baseline_type=normalize_baseline_type(baseline_type), evidence_ref=evidence_ref, approved_by=approved_by, status=status)
		self._baselines[_state_key(tenant_id, baseline.id)] = baseline
		if baseline.baseline_type == "configuration" and status == "approved":
			service.configuration_baseline_present = True
			service.updated_at = utc_now()
		self._record_audit(tenant_id, baseline.id, "baseline_attached", approved_by, result["decision"], reasons=self._reasons(result))
		return baseline.to_dict()

	def assess_readiness(self, assessment_id: str, tenant_id: str, service_id: str) -> dict[str, Any]:
		service = self._require_service(service_id, tenant_id)
		dependencies = self._service_dependency_dicts(tenant_id, service_id)
		baselines = self._service_baseline_dicts(tenant_id, service_id)
		dependencies_healthy = dependencies_are_healthy(dependencies)
		baselines_complete = service_baselines_complete(baselines)
		status, issues = readiness_posture(service.readiness_score, dependencies_healthy, baselines_complete, service.monitoring_enabled, bool(service.rollback_plan_ref), bool(service.change_window_ref))
		assessment = ReadinessAssessment(id=assessment_id, tenant_id=tenant_id, service_id=service_id, score=service.readiness_score, status=status, dependencies_healthy=dependencies_healthy, baselines_complete=baselines_complete, monitoring_ready=service.monitoring_enabled, rollback_ready=bool(service.rollback_plan_ref), change_window_ready=bool(service.change_window_ref), issues=tuple(issues))
		self._assessments[_state_key(tenant_id, assessment.id)] = assessment
		service.status = "ready" if status == "ready" else "blocked"
		service.updated_at = utc_now()
		self._record_audit(tenant_id, assessment.id, "readiness_assessed", "plfd", status, reasons=tuple(issues))
		return assessment.to_dict()

	def propose_platform_change(self, change_id: str, tenant_id: str, service_id: str, title: str, owner: str, affected_capability_count: int, dependencies_healthy: bool | None = None, approval_recorded: bool = False, broad_review_recorded: bool = False, security_review_recorded: bool = False, change_window_ref: str = "", rollback_plan_ref: str = "") -> dict[str, Any]:
		service = self._require_service(service_id, tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "propose_platform_change", "change_owner_present": bool(owner), "affected_capability_count": affected_capability_count})
		self._raise_if_denied(result)
		if dependencies_healthy is None:
			dependencies_healthy = dependencies_are_healthy(self._service_dependency_dicts(tenant_id, service_id))
		change = PlatformChange(id=change_id, tenant_id=tenant_id, service_id=service_id, title=title, owner=owner, affected_capability_count=int(affected_capability_count), dependencies_healthy=bool(dependencies_healthy), approval_recorded=bool(approval_recorded), broad_review_recorded=bool(broad_review_recorded), security_review_recorded=bool(security_review_recorded), change_window_ref=change_window_ref or service.change_window_ref, rollback_plan_ref=rollback_plan_ref or service.rollback_plan_ref, status=change_review_status(int(affected_capability_count), bool(broad_review_recorded)))
		self._changes[_state_key(tenant_id, change.id)] = change
		self._record_audit(tenant_id, change.id, "platform_change_proposed", owner, change.status, reasons=self._reasons(result))
		return change.to_dict()

	def approve_platform_change(self, change_id: str, tenant_id: str, approver: str, approval_recorded: bool = True, broad_review_recorded: bool | None = None, security_review_recorded: bool | None = None, event_stream: str = "bytewax") -> dict[str, Any]:
		change = self._require_change(change_id, tenant_id)
		service = self._require_service(change.service_id, tenant_id)
		if broad_review_recorded is not None:
			change.broad_review_recorded = bool(broad_review_recorded)
		if security_review_recorded is not None:
			change.security_review_recorded = bool(security_review_recorded)
		change.approval_recorded = bool(approval_recorded)
		change.dependencies_healthy = dependencies_are_healthy(self._service_dependency_dicts(tenant_id, service.id))
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "approve_platform_change", "dependencies_healthy": change.dependencies_healthy, "approval_recorded": change.approval_recorded, "configuration_baseline_present": service.configuration_baseline_present, "affected_capability_count": change.affected_capability_count, "broad_review_recorded": change.broad_review_recorded, "security_review_recorded": change.security_review_recorded, "change_window_present": bool(change.change_window_ref), "rollback_plan_present": bool(change.rollback_plan_ref), "event_stream": event_stream_name(event_stream)})
		self._raise_if_denied(result)
		self._raise_if_review_required(result)
		change.status = "approved"
		change.approved_at = utc_now()
		service.updated_at = utc_now()
		self._record_audit(tenant_id, change.id, "platform_change_approved", approver, result["decision"], reasons=self._reasons(result))
		return change.to_dict()

	# ------------------------------------------------------------------
	# List / query
	# ------------------------------------------------------------------

	def list_services(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._services, tenant_id)

	def list_dependencies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._dependencies, tenant_id)

	def list_baselines(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._baselines, tenant_id)

	def list_readiness_assessments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._assessments, tenant_id)

	def list_changes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._changes, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_plfd_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	# ------------------------------------------------------------------
	# Agent management
	# ------------------------------------------------------------------

	def register_plfd_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"plfd_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_PLFD_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_PLFD_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_if_denied(result)
		agent = PlfdAgent(
			id=agent_id or f"plfd-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._record_audit(tenant_id, agent.id, "plfd_agent_registered", name, result["decision"])
		return agent.to_dict()

	def validate_batch_foundation_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({"tenant_context_present": True, "requested_operation": "batch_foundation_mutation", "event_stream": event_stream})

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		services = self.list_services(tenant_id)
		assessments = self.list_readiness_assessments(tenant_id)
		changes = self.list_changes(tenant_id)
		return {
			"tenant_id": tenant_id,
			"service_count": len(services),
			"core_service_count": len([s for s in services if s["tier"] == "core"]),
			"ready_service_count": len([s for s in services if s["status"] == "ready"]),
			"blocked_service_count": len([s for s in services if s["status"] == "blocked"]),
			"dependency_count": len(self.list_dependencies(tenant_id)),
			"unhealthy_dependency_count": len([d for d in self.list_dependencies(tenant_id) if d.get("required") and d["health_status"] != "healthy"]),
			"baseline_count": len(self.list_baselines(tenant_id)),
			"readiness_assessment_count": len(assessments),
			"feature_flag_count": sum(1 for f in self._feature_flags.values() if f["tenant_id"] == tenant_id),
			"open_circuit_breaker_count": sum(1 for cb in self._circuit_breakers.values() if cb["tenant_id"] == tenant_id and cb["state"] == "open"),
			"plfd_agent_count": len(self.list_plfd_agents(tenant_id)),
			"approved_change_count": len([c for c in changes if c["status"] == "approved"]),
			"pending_change_count": len([c for c in changes if c["status"] != "approved"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.register_foundation_service(
			service_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "foundation-owner"),
			tier=str(metadata.get("tier") or "shared"),
			dependencies=list(metadata.get("dependencies") or []),
			readiness_score=float(metadata.get("readiness_score") or 80),
			configuration_baseline_present=bool(metadata.get("configuration_baseline_present", True)),
			health_status=str(metadata.get("health_status") or "healthy"),
			metadata=metadata | {"compatibility_status": status or "active"},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_services(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_service(self, service_id: str, tenant_id: str) -> FoundationService:
		service = self._services.get(_state_key(tenant_id, service_id))
		if service is None or service.tenant_id != tenant_id:
			raise KeyError("foundation_service_not_found")
		return service

	def _require_change(self, change_id: str, tenant_id: str) -> PlatformChange:
		change = self._changes.get(_state_key(tenant_id, change_id))
		if change is None or change.tenant_id != tenant_id:
			raise KeyError("platform_change_not_found")
		return change

	def _service_dependency_dicts(self, tenant_id: str, service_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._dependencies.values() if item.tenant_id == tenant_id and item.source_service_id == service_id]

	def _service_baseline_dicts(self, tenant_id: str, service_id: str) -> list[dict[str, Any]]:
		return [item.to_dict() for item in self._baselines.values() if item.tenant_id == tenant_id and item.service_id == service_id]

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "platform_foundation_policy_blocked")

	def _raise_if_review_required(self, result: dict[str, Any]) -> None:
		if result["decision"] == "require_review":
			raise PermissionError(", ".join(self._reasons(result)) or "platform_foundation_review_required")

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> None:
		event_id = stable_id("plfdaudit", tenant_id, event_type, subject_id, len(self._audit_events))
		self._audit_events[event_id] = PlfdAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			decision=decision,
			reasons=reasons,
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [item for item in values if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "platform_foundation_policy_blocked") for action in result.get("actions", []))

	# ------------------------------------------------------------------
	# Extended methods — 40+ total
	# ------------------------------------------------------------------

	def feature_flag_create(
		self,
		tenant_id: str,
		flag_name: str,
		enabled: bool,
		conditions: dict[str, Any] | None = None,
		description: str = "",
		set_by: str = "system",
		rollout_percentage: float = 100.0,
	) -> dict[str, Any]:
		"""Create a new feature flag (alias for feature_flag_set with create semantics)."""
		flag_key = f"{tenant_id}:{flag_name}"
		if flag_key in self._feature_flags:
			raise ValueError(f"feature_flag_already_exists:{flag_name}")
		return self.feature_flag_set(
			flag_name=flag_name,
			enabled=enabled,
			conditions=dict(conditions or {}),
			tenant_id=tenant_id,
			description=description,
			set_by=set_by,
			rollout_percentage=rollout_percentage,
		)

	def feature_flag_evaluate(
		self,
		tenant_id: str,
		flag_name: str,
		context: dict[str, Any],
	) -> dict[str, Any]:
		"""Evaluate a feature flag for a request context (alias for feature_flag_check)."""
		return self.feature_flag_check(flag_name=flag_name, context=context, tenant_id=tenant_id)

	def ab_config_create(
		self,
		tenant_id: str,
		experiment_name: str,
		variants: list[dict[str, Any]],
		traffic_split: list[float],
		set_by: str = "system",
		description: str = "",
	) -> dict[str, Any]:
		"""
		Create an A/B experiment configuration.

		variants: list of {"name": str, "config": dict} dicts.
		traffic_split: list of floats summing to 100.0 (one per variant).
		"""
		if not experiment_name:
			raise ValueError("experiment_name_required")
		if len(variants) < 2:
			raise ValueError("ab_config_requires_at_least_2_variants")
		if len(variants) != len(traffic_split):
			raise ValueError("variants_and_traffic_split_must_have_same_length")
		if abs(sum(traffic_split) - 100.0) > 0.01:
			raise ValueError("traffic_split_must_sum_to_100")
		exp_key = f"{tenant_id}:{experiment_name}"
		record = {
			"experiment_name": experiment_name,
			"tenant_id":       tenant_id,
			"variants":        variants,
			"traffic_split":   traffic_split,
			"description":     description,
			"set_by":          set_by,
			"status":          "active",
			"created_at":      _ts(),
		}
		if not hasattr(self, "_ab_configs"):
			self._ab_configs: dict[str, dict[str, Any]] = {}
		self._ab_configs[exp_key] = record
		self._record_audit(tenant_id, experiment_name, "ab_config_created", set_by, "allow",
			metadata={"variants": len(variants)})
		return record

	def circuit_breaker_define(
		self,
		tenant_id: str,
		service_name: str,
		failure_threshold: int = 5,
		recovery_timeout_seconds: int = 60,
		defined_by: str = "system",
	) -> dict[str, Any]:
		"""
		Define a circuit breaker for a service with custom thresholds.

		Returns the newly-created circuit breaker configuration.
		"""
		if not service_name:
			raise ValueError("service_name_required")
		if failure_threshold < 1:
			raise ValueError("failure_threshold_must_be_positive")
		cb_key = f"{tenant_id}:{service_name}"
		if cb_key in self._circuit_breakers:
			raise ValueError(f"circuit_breaker_already_defined:{service_name}")
		record = {
			"service_name":              service_name,
			"tenant_id":                 tenant_id,
			"state":                     "closed",
			"failure_count":             0,
			"success_count":             0,
			"failure_threshold":         failure_threshold,
			"recovery_timeout_seconds":  recovery_timeout_seconds,
			"last_failure_at":           None,
			"opened_at":                 None,
			"reset_at":                  None,
			"defined_by":                defined_by,
			"created_at":                _ts(),
		}
		self._circuit_breakers[cb_key] = record
		self._record_audit(tenant_id, service_name, "circuit_breaker_defined", defined_by, "allow",
			metadata={"failure_threshold": failure_threshold})
		return record

	def rate_limiter_define(
		self,
		tenant_id: str,
		service_name: str,
		limit: int,
		window: str,
		strategy: str = "token_bucket",
		burst_limit: int | None = None,
		defined_by: str = "system",
	) -> dict[str, Any]:
		"""
		Define a new rate limiter for a service (alias for rate_limiter_configure
		with create semantics — raises if already defined).
		"""
		rl_key = f"{tenant_id}:{service_name}"
		if rl_key in self._rate_limiters:
			raise ValueError(f"rate_limiter_already_defined:{service_name}")
		return self.rate_limiter_configure(
			service_name=service_name,
			limit=limit,
			window=window,
			tenant_id=tenant_id,
			strategy=strategy,
			burst_limit=burst_limit,
			configured_by=defined_by,
		)

	def dependency_declare(
		self,
		tenant_id: str,
		source_service_id: str,
		target_service_id: str,
		dependency_id: str,
		required: bool = True,
		evidence_ref: str = "",
		health_status: str = "healthy",
	) -> dict[str, Any]:
		"""Declare a service dependency — alias for record_dependency."""
		return self.record_dependency(
			dependency_id=dependency_id,
			tenant_id=tenant_id,
			source_service_id=source_service_id,
			target_service_id=target_service_id,
			health_status=health_status,
			required=required,
			evidence_ref=evidence_ref,
		)

	def health_aggregate(
		self,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return a concise health aggregate across all services.

		Equivalent to health_check_all_services but returns only summary counts
		and overall posture without per-service detail.
		"""
		full = self.health_check_all_services(tenant_id=tenant_id, include_dependencies=True)
		return {
			"tenant_id":      tenant_id,
			"overall":        full["overall"],
			"service_count":  full["service_count"],
			"healthy":        full["healthy_count"],
			"degraded":       full["degraded_count"],
			"unhealthy":      full["unhealthy_count"],
			"aggregated_at":  _ts(),
		}

	def service_mesh_register(
		self,
		tenant_id: str,
		service_name: str,
		endpoint: str,
		metadata: dict[str, Any] | None = None,
		version: str = "1.0.0",
		registered_by: str = "system",
	) -> dict[str, Any]:
		"""Register a service in the platform service mesh (alias for service_discovery_register)."""
		return self.service_discovery_register(
			service_name=service_name,
			endpoint=endpoint,
			metadata=dict(metadata or {}),
			tenant_id=tenant_id,
			version=version,
			registered_by=registered_by,
		)

	def graceful_degrade(
		self,
		tenant_id: str,
		service_name: str,
		degradation_level: str,
		reason: str,
		approved_by: str = "system",
	) -> dict[str, Any]:
		"""
		Put a service into graceful degradation mode.

		degradation_level: 'partial' | 'read_only' | 'maintenance' | 'unavailable'.
		Updates the service registry entry and records an audit event.
		"""
		if not service_name:
			raise ValueError("service_name_required")
		supported = {"partial", "read_only", "maintenance", "unavailable"}
		if degradation_level not in supported:
			raise ValueError(f"unsupported_degradation_level:{degradation_level}")
		reg_key = f"{tenant_id}:{service_name}"
		reg = self._service_registry.get(reg_key)
		if reg is None:
			# Auto-create a registry entry
			reg = {"service_name": service_name, "tenant_id": tenant_id, "endpoint": "", "status": "active"}
			self._service_registry[reg_key] = reg
		reg["degradation_level"] = degradation_level
		reg["degradation_reason"] = reason
		reg["degraded_by"] = approved_by
		reg["degraded_at"] = _ts()
		self._record_audit(tenant_id, service_name, "service_degraded", approved_by, "allow",
			metadata={"level": degradation_level, "reason": reason})
		return dict(reg)

	def config_hot_reload(
		self,
		tenant_id: str,
		environment: str,
		config_keys: list[str] | None = None,
		triggered_by: str = "system",
	) -> dict[str, Any]:
		"""
		Trigger a hot-reload of platform configuration for an environment.

		Finds all config keys matching the environment and returns a reload manifest.
		"""
		if not environment:
			raise ValueError("environment_required")
		prefix = f"{tenant_id}:{environment}:"
		matched = {
			k: v for k, v in self._platform_configs.items()
			if k.startswith(prefix) and (config_keys is None or v["key"] in config_keys)
		}
		reload_id = stable_id("reload", tenant_id, environment, str(len(matched)))
		record = {
			"reload_id":      reload_id,
			"tenant_id":      tenant_id,
			"environment":    environment,
			"keys_reloaded":  len(matched),
			"config_keys":    [v["key"] for v in matched.values()],
			"triggered_by":   triggered_by,
			"reloaded_at":    _ts(),
		}
		self._record_audit(tenant_id, reload_id, "config_hot_reloaded", triggered_by, "allow",
			metadata={"environment": environment, "keys_reloaded": len(matched)})
		return record

	def platform_analytics(
		self,
		tenant_id: str = "default",
		period_label: str = "all_time",
	) -> dict[str, Any]:
		"""
		Comprehensive platform analytics aggregating all sub-systems.

		Covers services, deps, flags, circuit breakers, rate limiters,
		config, service registry, agents, changes, and audit events.
		"""
		snapshot = self.platform_metrics_dashboard(tenant_id=tenant_id)
		ab_configs  = [v for v in getattr(self, "_ab_configs", {}).values() if v.get("tenant_id") == tenant_id]
		hot_reloads = sum(
			1 for e in self._audit_events.values()
			if e.tenant_id == tenant_id and e.event_type == "config_hot_reloaded"
		)
		return {
			**snapshot,
			"period":              period_label,
			"ab_experiment_count": len(ab_configs),
			"hot_reload_count":    hot_reloads,
			"plfd_agent_count":    len(self.list_plfd_agents(tenant_id)),
			"metrics_snapshot_count": len(self._metrics_snapshots),
		}

	def secret_rotation(
		self,
		tenant_id: str,
		secret_name: str,
		new_secret_hash: str,
		rotated_by: str = "system",
		rotation_reason: str = "scheduled",
	) -> dict[str, Any]:
		"""
		Record a secret rotation event for a named secret.

		Stores the rotation manifest (never the secret value) and audits the action.
		"""
		import hashlib
		if not secret_name:
			raise ValueError("secret_name_required")
		if not new_secret_hash:
			raise ValueError("new_secret_hash_required")
		rotation_id = stable_id("secrot", tenant_id, secret_name, _ts())
		record = {
			"rotation_id":      rotation_id,
			"tenant_id":        tenant_id,
			"secret_name":      secret_name,
			"new_secret_hash":  new_secret_hash[:16] + "...",  # truncated for safety
			"rotated_by":       rotated_by,
			"rotation_reason":  rotation_reason,
			"rotated_at":       _ts(),
		}
		if not hasattr(self, "_secret_rotations"):
			self._secret_rotations: list[dict[str, Any]] = []
		self._secret_rotations.append(record)
		self._record_audit(tenant_id, secret_name, "secret_rotated", rotated_by, "allow",
			metadata={"reason": rotation_reason})
		return record

	def env_promote(
		self,
		tenant_id: str,
		config_key: str,
		from_env: str,
		to_env: str,
		promoted_by: str = "system",
	) -> dict[str, Any]:
		"""
		Promote a configuration key's value from one environment to another.

		Reads from from_env and writes to to_env, preserving all metadata.
		"""
		src_key  = f"{tenant_id}:{from_env}:{config_key}"
		src      = self._platform_configs.get(src_key)
		if src is None:
			raise KeyError(f"config_key_not_found_in_env:{config_key}:{from_env}")
		dst = self.platform_configuration(
			key=config_key,
			value=src["value"],
			environment=to_env,
			tenant_id=tenant_id,
			data_type=src["data_type"],
			description=src["description"],
			set_by=promoted_by,
		)
		self._record_audit(tenant_id, config_key, "config_promoted", promoted_by, "allow",
			metadata={"from_env": from_env, "to_env": to_env})
		return {
			"config_key":  config_key,
			"tenant_id":   tenant_id,
			"from_env":    from_env,
			"to_env":      to_env,
			"promoted_by": promoted_by,
			"dst_config":  dst,
			"promoted_at": _ts(),
		}

	def cost_track(
		self,
		tenant_id: str,
		service_name: str,
		cost_usd: float,
		period_label: str,
		cost_category: str = "compute",
		tracked_by: str = "system",
	) -> dict[str, Any]:
		"""
		Record a cost tracking entry for a service.

		cost_category: 'compute' | 'storage' | 'network' | 'licensing' | 'other'.
		"""
		supported_categories = {"compute", "storage", "network", "licensing", "other"}
		if cost_category not in supported_categories:
			raise ValueError(f"unsupported_cost_category:{cost_category}")
		if cost_usd < 0:
			raise ValueError("cost_usd_must_be_non_negative")
		entry_id = stable_id("cost", tenant_id, service_name, period_label)
		record = {
			"entry_id":      entry_id,
			"tenant_id":     tenant_id,
			"service_name":  service_name,
			"cost_usd":      round(cost_usd, 6),
			"period_label":  period_label,
			"cost_category": cost_category,
			"tracked_by":    tracked_by,
			"recorded_at":   _ts(),
		}
		if not hasattr(self, "_cost_entries"):
			self._cost_entries: list[dict[str, Any]] = []
		self._cost_entries.append(record)
		self._record_audit(tenant_id, service_name, "cost_tracked", tracked_by, "allow",
			metadata={"cost_usd": cost_usd, "period": period_label})
		return record

	def platform_health(
		self,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return a single-call platform health status.

		Wraps health_aggregate plus circuit-breaker and rate-limiter posture.
		"""
		agg = self.health_aggregate(tenant_id)
		open_cbs  = [cb for cb in self._circuit_breakers.values() if cb["tenant_id"] == tenant_id and cb["state"] == "open"]
		max_rl    = max((rl["limit"] for rl in self._rate_limiters.values() if rl["tenant_id"] == tenant_id), default=0)
		return {
			**agg,
			"open_circuit_breakers":     len(open_cbs),
			"open_cb_services":          [cb["service_name"] for cb in open_cbs],
			"rate_limiter_count":        sum(1 for rl in self._rate_limiters.values() if rl["tenant_id"] == tenant_id),
			"max_rate_limit_rpm":        max_rl,
			"platform_health_at":        _ts(),
		}


# Alias
PlfdService = PlatformFoundationService
