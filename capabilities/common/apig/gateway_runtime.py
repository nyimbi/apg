"""Dependency-light APIG route publication runtime for package composition."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	PRIVILEGED_APIG_AGENT_ROLES,
	SUPPORTED_APIG_AGENT_ROLES,
	SUPPORTED_APIG_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	GatewayAuditEvent,
	GatewayAgentRecord,
	GatewayConsumerRecord,
	GatewayDeploymentRecord,
	GatewayLifecycleBatchRecord,
	GatewayPolicyRecord,
	GatewayQuotaReview,
	GatewayRouteRecord,
	GatewayTrafficShiftRecord,
	GatewayUpstreamRecord,
)


UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class ApigService:
	"""Tenant-scoped gateway control-plane facade for generated APG apps."""

	def __init__(self) -> None:
		self._agent_runtimes = set(SUPPORTED_APIG_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_APIG_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_APIG_AGENT_ROLES)
		self._upstreams: dict[tuple[str, str], GatewayUpstreamRecord] = {}
		self._consumers: dict[tuple[str, str], GatewayConsumerRecord] = {}
		self._routes: dict[tuple[str, str], GatewayRouteRecord] = {}
		self._quota_reviews: dict[tuple[str, str], GatewayQuotaReview] = {}
		self._policies: dict[tuple[str, str], GatewayPolicyRecord] = {}
		self._traffic_shifts: dict[tuple[str, str], GatewayTrafficShiftRecord] = {}
		self._deployments: dict[tuple[str, str], GatewayDeploymentRecord] = {}
		self._gateway_agents: dict[tuple[str, str], GatewayAgentRecord] = {}
		self._lifecycle_batches: dict[tuple[str, str], GatewayLifecycleBatchRecord] = {}
		self._events: list[GatewayAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_upstream(
		self,
		upstream_id: str,
		tenant_id: str,
		name: str,
		base_url: str,
		owner: str,
		health: str = "healthy",
		health_check_configured: bool = True,
		labels: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_upstream",
			"upstream_owner_assigned": bool(owner),
			"https_enabled": base_url.startswith("https://"),
			"health_check_configured": health_check_configured and bool(health),
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, upstream_id) in self._upstreams:
			raise ValueError(f"upstream already exists for tenant: {upstream_id}")
		if not name:
			raise ValueError("upstream name is required")
		if not (base_url.startswith("http://") or base_url.startswith("https://")):
			raise ValueError("upstream base_url must be http or https")
		record = GatewayUpstreamRecord(
			id=upstream_id,
			tenant_id=tenant_id,
			name=name,
			base_url=base_url.rstrip("/"),
			owner=owner,
			health=health,
			**_policy_kwargs(result),
			labels=dict(labels or {}),
		)
		self._upstreams[self._tenant_key(tenant_id, upstream_id)] = record
		self._record_event(
			tenant_id=tenant_id,
			event_type="upstream_registered",
			subject_id=upstream_id,
			message=f"Registered upstream {name}.",
			evidence={"base_url": record.base_url, "owner": owner, "health": health, "matched_rules": result["matched_rules"]},
			policy_result=result,
		)
		return record.model_dump(mode="json")

	def register_consumer(
		self,
		consumer_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		access_tier: str = "standard",
		identity_provider: str = "auth",
		credential_rotation_recorded: bool = True,
		rbac_approval_recorded: bool = False,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_consumer",
			"consumer_owner_assigned": bool(owner),
			"access_tier": access_tier,
			"credential_rotation_recorded": credential_rotation_recorded,
			"rbac_approval_recorded": rbac_approval_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, consumer_id) in self._consumers:
			raise ValueError(f"consumer already exists for tenant: {consumer_id}")
		if not name:
			raise ValueError("consumer name is required")
		record = GatewayConsumerRecord(
			id=consumer_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			access_tier=access_tier,
			identity_provider=identity_provider,
			credential_rotation_recorded=credential_rotation_recorded,
			rbac_approval_recorded=rbac_approval_recorded,
			status="registered",
			**_policy_kwargs(result),
		)
		self._consumers[self._tenant_key(tenant_id, consumer_id)] = record
		self._record_event(
			tenant_id=tenant_id,
			event_type="consumer_registered",
			subject_id=consumer_id,
			message=f"Registered consumer {name}.",
			evidence={"access_tier": access_tier, "matched_rules": result["matched_rules"]},
			policy_result=result,
		)
		return record.model_dump(mode="json")

	def request_route(
		self,
		route_id: str,
		tenant_id: str,
		path: str,
		methods: list[str] | tuple[str, ...],
		upstream_id: str,
		owner: str,
		route_exposure: str = "internal",
		consumer_id: str | None = None,
		auth_policy_attached: bool = True,
		threat_policy_attached: bool = True,
		mtls_enabled: bool = True,
		rate_limit_configured: bool = True,
		requested_rps_limit: int = 1000,
		wasm_filter_attached: bool = False,
		filter_signature_verified: bool = True,
		justification: str = "",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, route_id) in self._routes:
			raise ValueError(f"route already exists for tenant: {route_id}")
		upstream_registered = self._tenant_key(tenant_id, upstream_id) in self._upstreams
		normalized_methods = [method.upper() for method in methods]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_route",
			"service_registered": upstream_registered,
			"route_owner_assigned": bool(owner),
			"absolute_path": path.startswith("/"),
			"methods_present": bool(normalized_methods),
			"route_exposure": route_exposure,
			"auth_policy_attached": auth_policy_attached,
			"mtls_enabled": mtls_enabled,
			"unsafe_http_method_enabled": any(method in UNSAFE_METHODS for method in normalized_methods),
			"threat_policy_attached": threat_policy_attached,
			"rate_limit_configured": rate_limit_configured,
			"wasm_filter_attached": wasm_filter_attached,
			"filter_signature_verified": filter_signature_verified,
			"requested_rps_limit": requested_rps_limit,
			"quota_review_recorded": False,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		self._get_upstream(tenant_id, upstream_id)
		if consumer_id is not None:
			self._get_consumer(tenant_id, consumer_id)
		status = "pending_quota_review" if result["decision"] == "require_review" else "active"
		route = GatewayRouteRecord(
			id=route_id,
			tenant_id=tenant_id,
			path=path,
			methods=normalized_methods,
			upstream_id=upstream_id,
			owner=owner,
			route_exposure=route_exposure,
			consumer_id=consumer_id,
			auth_policy_attached=auth_policy_attached,
			threat_policy_attached=threat_policy_attached,
			mtls_enabled=mtls_enabled,
			rate_limit_configured=rate_limit_configured,
			requested_rps_limit=requested_rps_limit,
			wasm_filter_attached=wasm_filter_attached,
			filter_signature_verified=filter_signature_verified,
			status=status,
			**_policy_kwargs(result),
		)
		self._routes[self._tenant_key(tenant_id, route_id)] = route
		self._record_event(
			tenant_id=tenant_id,
			event_type="route_requested",
			subject_id=route_id,
			message=f"Requested route {path}.",
			evidence={"status": status, "upstream_id": upstream_id, "requested_rps_limit": requested_rps_limit, "matched_rules": result["matched_rules"]},
			policy_result=result,
		)
		if result["decision"] == "require_review":
			review = GatewayQuotaReview(
				id=f"quota:{route_id}",
				tenant_id=tenant_id,
				route_id=route_id,
				requested_rps_limit=requested_rps_limit,
				requester=owner,
				justification=justification or "High gateway quota requested.",
				matched_rules=result["matched_rules"],
				policy_decision=result["decision"],
				review_reasons=_reasons(result),
				review_evidence=_review_evidence(result),
			)
			self._quota_reviews[self._tenant_key(tenant_id, review.id)] = review
			self._record_event(
				tenant_id=tenant_id,
				event_type="quota_review_requested",
				subject_id=review.id,
				message=f"Requested quota review for {route_id}.",
				evidence={"route_id": route_id, "requested_rps_limit": requested_rps_limit},
				policy_result=result,
			)
			return {"route": route.model_dump(mode="json"), "quota_review": review.model_dump(mode="json")}
		return route.model_dump(mode="json")

	def decide_quota_review(
		self,
		review_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		review = self._quota_reviews.get(self._tenant_key(tenant_id, review_id))
		if review is None:
			raise KeyError(f"unknown quota review for tenant: {review_id}")
		if decision not in {"approved", "rejected"}:
			raise ValueError("quota review decision must be approved or rejected")
		if not reviewer:
			raise ValueError("quota review reviewer is required")
		if not notes:
			raise ValueError("quota review notes are required")
		decided = GatewayQuotaReview(
			id=review.id,
			tenant_id=review.tenant_id,
			route_id=review.route_id,
			requested_rps_limit=review.requested_rps_limit,
			requester=review.requester,
			justification=review.justification,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
			matched_rules=list(review.matched_rules),
			policy_decision=review.policy_decision,
			review_reasons=list(review.review_reasons),
			review_evidence={**review.review_evidence, "review_recorded": True},
		)
		self._quota_reviews[self._tenant_key(tenant_id, review_id)] = decided
		self._record_event(
			tenant_id=tenant_id,
			event_type="quota_review_decided",
			subject_id=review_id,
			message=f"Quota review {review_id} was {decision}.",
			evidence={"route_id": review.route_id, "reviewer": reviewer},
		)
		return decided.model_dump(mode="json")

	def activate_route(self, route_id: str, tenant_id: str) -> dict[str, Any]:
		route = self._get_route(tenant_id, route_id)
		if route.status == "pending_quota_review":
			review = self._quota_reviews.get(self._tenant_key(tenant_id, f"quota:{route_id}"))
			if review is None or review.decision != "approved":
				raise PermissionError("quota_review_required")
		activated = GatewayRouteRecord(
			id=route.id,
			tenant_id=route.tenant_id,
			path=route.path,
			methods=list(route.methods),
			upstream_id=route.upstream_id,
			owner=route.owner,
			route_exposure=route.route_exposure,
			auth_policy_attached=route.auth_policy_attached,
			threat_policy_attached=route.threat_policy_attached,
			mtls_enabled=route.mtls_enabled,
			rate_limit_configured=route.rate_limit_configured,
			requested_rps_limit=route.requested_rps_limit,
			consumer_id=route.consumer_id,
			wasm_filter_attached=route.wasm_filter_attached,
			filter_signature_verified=route.filter_signature_verified,
			status="active",
			decision=route.decision,
			matched_rules=list(route.matched_rules),
			policy_decision=route.policy_decision,
			review_reasons=list(route.review_reasons),
			review_evidence=dict(route.review_evidence),
		)
		self._routes[self._tenant_key(tenant_id, route_id)] = activated
		self._record_event(
			tenant_id=tenant_id,
			event_type="route_activated",
			subject_id=route_id,
			message=f"Activated route {route.path}.",
			evidence={"requested_rps_limit": route.requested_rps_limit},
		)
		return activated.model_dump(mode="json")

	def change_policy(
		self,
		policy_id: str,
		tenant_id: str,
		name: str,
		policy_type: str,
		actor: str,
		policy_review_recorded: bool,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "change_policy",
			"policy_review_recorded": policy_review_recorded,
		})
		record = GatewayPolicyRecord(
			id=policy_id,
			tenant_id=tenant_id,
			name=name,
			policy_type=policy_type,
			actor=actor,
			status="active" if result["decision"] == "allow" else _status_for_decision(result["decision"]),
			**_policy_kwargs(result, policy_review_recorded),
			metadata=dict(metadata or {}),
		)
		self._policies[self._tenant_key(tenant_id, policy_id)] = record
		self._record_event(
			tenant_id=tenant_id,
			event_type="policy_change_evaluated",
			subject_id=policy_id,
			message=f"Evaluated policy change {name}.",
			evidence={"decision": result["decision"], "matched_rules": result["matched_rules"]},
			policy_result=result,
			review_recorded=policy_review_recorded,
		)
		return record.model_dump(mode="json")

	def shift_traffic(
		self,
		shift_id: str,
		tenant_id: str,
		route_id: str,
		canary_percent: int,
		actor: str,
		rollback_plan_recorded: bool,
		canary_review_recorded: bool,
		rollback_plan: str | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		self._get_route(tenant_id, route_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "shift_traffic",
			"canary_percent": canary_percent,
			"rollback_plan_recorded": rollback_plan_recorded,
			"canary_review_recorded": canary_review_recorded,
		})
		record = GatewayTrafficShiftRecord(
			id=shift_id,
			tenant_id=tenant_id,
			route_id=route_id,
			canary_percent=canary_percent,
			actor=actor,
			status="active" if result["decision"] == "allow" else _status_for_decision(result["decision"]),
			**_policy_kwargs(result, canary_review_recorded),
			rollback_plan=rollback_plan,
		)
		self._traffic_shifts[self._tenant_key(tenant_id, shift_id)] = record
		self._record_event(
			tenant_id=tenant_id,
			event_type="traffic_shift_evaluated",
			subject_id=shift_id,
			message=f"Evaluated canary shift for {route_id}.",
			evidence={"decision": result["decision"], "canary_percent": canary_percent, "matched_rules": result["matched_rules"]},
			policy_result=result,
			review_recorded=canary_review_recorded,
		)
		return record.model_dump(mode="json")

	def deploy_gateway(
		self,
		deployment_id: str,
		tenant_id: str,
		environment: str,
		region: str,
		actor: str,
		observability_configured: bool,
		deployment_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		allowed_regions = set(self.describe(tenant_id)["configuration"]["edge"]["allowed_regions"])
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_gateway",
			"environment": environment,
			"allowed_region": region in allowed_regions,
			"observability_configured": observability_configured,
			"deployment_approval_recorded": deployment_approval_recorded,
		})
		record = GatewayDeploymentRecord(
			id=deployment_id,
			tenant_id=tenant_id,
			environment=environment,
			region=region,
			actor=actor,
			status="deployed" if result["decision"] == "allow" else _status_for_decision(result["decision"]),
			**_policy_kwargs(result, deployment_approval_recorded),
		)
		self._deployments[self._tenant_key(tenant_id, deployment_id)] = record
		self._record_event(
			tenant_id=tenant_id,
			event_type="deployment_evaluated",
			subject_id=deployment_id,
			message=f"Evaluated {environment} deployment in {region}.",
			evidence={"decision": result["decision"], "matched_rules": result["matched_rules"]},
			policy_result=result,
			review_recorded=deployment_approval_recorded,
		)
		return record.model_dump(mode="json")

	def retire_route(
		self,
		route_id: str,
		tenant_id: str,
		actor: str,
		impact_review_recorded: bool,
	) -> dict[str, Any]:
		route = self._get_route(tenant_id, route_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_route",
			"impact_review_recorded": impact_review_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		retired = GatewayRouteRecord(
			id=route.id,
			tenant_id=route.tenant_id,
			path=route.path,
			methods=list(route.methods),
			upstream_id=route.upstream_id,
			owner=route.owner,
			route_exposure=route.route_exposure,
			consumer_id=route.consumer_id,
			auth_policy_attached=route.auth_policy_attached,
			threat_policy_attached=route.threat_policy_attached,
			mtls_enabled=route.mtls_enabled,
			rate_limit_configured=route.rate_limit_configured,
			requested_rps_limit=route.requested_rps_limit,
			wasm_filter_attached=route.wasm_filter_attached,
			filter_signature_verified=route.filter_signature_verified,
			status="retired",
			**_policy_kwargs(result, impact_review_recorded),
		)
		self._routes[self._tenant_key(tenant_id, route_id)] = retired
		self._record_event(
			tenant_id=tenant_id,
			event_type="route_retired",
			subject_id=route_id,
			message=f"Retired route {route.path}.",
			evidence={"actor": actor, "matched_rules": result["matched_rules"]},
			policy_result=result,
			review_recorded=impact_review_recorded,
		)
		return retired.model_dump(mode="json")

	def register_gateway_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		runtime_value = _normalize_agent_token(runtime)
		role_value = _normalize_agent_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_gateway_agent",
			"unsupported_agent_runtime": runtime_value not in self._agent_runtimes,
			"unsupported_agent_role": role_value not in self._agent_roles,
			"agent_scope_present": bool(str(scope or "").strip()),
			"agent_owner_present": bool(str(owner or "").strip()),
			"agent_purpose_present": bool(str(purpose or "").strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
			"privileged_agent_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		if result["decision"] == "deny":
			self._record_event(
				tenant_id=tenant_id,
				event_type="gateway_agent_registration_denied",
				subject_id=agent_id,
				message=f"Denied gateway agent {name}.",
				evidence={"runtime": runtime_value, "role": role_value, "matched_rules": result["matched_rules"]},
				policy_result=result,
			)
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, agent_id) in self._gateway_agents:
			raise ValueError(f"gateway agent already exists for tenant: {agent_id}")
		record = GatewayAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status="active" if result["decision"] == "allow" else _status_for_decision(result["decision"]),
			**_policy_kwargs(result, bool(human_approval_required)),
		)
		self._gateway_agents[self._tenant_key(tenant_id, agent_id)] = record
		self._record_event(
			tenant_id=tenant_id,
			event_type="gateway_agent_registered",
			subject_id=agent_id,
			message=f"Registered gateway agent {name}.",
			evidence={"runtime": runtime_value, "role": role_value, "status": record.status, "matched_rules": result["matched_rules"]},
			policy_result=result,
			review_recorded=bool(human_approval_required),
		)
		return record.model_dump(mode="json")

	def validate_apig_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("apig_lifecycle_batch_empty")
		stream_value = _normalize_agent_token(event_stream)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_apig_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		record = GatewayLifecycleBatchRecord(
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			accepted=accepted,
			**_policy_kwargs(result),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_key(tenant_id, record.id)] = record
		self._record_event(
			tenant_id=tenant_id,
			event_type=f"lifecycle_batch_{record.status}",
			subject_id=record.id,
			message=f"Validated APIG lifecycle batch through {stream_value}.",
			evidence={"event_stream": stream_value, "mutation_count": mutation_count, "matched_rules": result["matched_rules"]},
			policy_result=result,
		)
		if not accepted:
			_raise_if_blocked(result)
		return record.model_dump(mode="json")

	def list_upstreams(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._upstreams, tenant_id)

	def list_consumers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._consumers, tenant_id)

	def list_routes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._routes, tenant_id)

	def list_quota_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._quota_reviews, tenant_id)

	def list_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._policies, tenant_id)

	def list_traffic_shifts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._traffic_shifts, tenant_id)

	def list_deployments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._deployments, tenant_id)

	def list_gateway_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._gateway_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._events)
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.model_dump(mode="json") for event in events]

	def gateway_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		routes = self.list_routes(tenant_id)
		reviews = self.list_quota_reviews(tenant_id)
		return {
			"tenant_id": tenant_id,
			"upstream_count": len(self.list_upstreams(tenant_id)),
			"consumer_count": len(self.list_consumers(tenant_id)),
			"route_count": len(routes),
			"active_route_count": len([route for route in routes if route["status"] == "active"]),
			"retired_route_count": len([route for route in routes if route["status"] == "retired"]),
			"pending_quota_review_count": len([review for review in reviews if review["decision"] == "pending"]),
			"public_route_count": len([route for route in routes if route["route_exposure"] == "public"]),
			"edge_filter_count": len([route for route in routes if route["wasm_filter_attached"]]),
			"policy_count": len(self.list_policies(tenant_id)),
			"traffic_shift_count": len(self.list_traffic_shifts(tenant_id)),
			"deployment_count": len(self.list_deployments(tenant_id)),
			"gateway_agent_count": len(self.list_gateway_agents(tenant_id)),
			"pending_gateway_agent_review_count": len([agent for agent in self.list_gateway_agents(tenant_id) if agent["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([batch for batch in self.list_lifecycle_batches(tenant_id) if batch["status"] == "denied"]),
			"review_count": len(self.list_pending_reviews(tenant_id)),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return gateway records awaiting quota, policy, traffic, deployment, or agent review."""
		items = (
			self.list_upstreams(tenant_id)
			+ self.list_consumers(tenant_id)
			+ self.list_routes(tenant_id)
			+ self.list_quota_reviews(tenant_id)
			+ self.list_policies(tenant_id)
			+ self.list_traffic_shifts(tenant_id)
			+ self.list_deployments(tenant_id)
			+ self.list_gateway_agents(tenant_id)
			+ self.list_lifecycle_batches(tenant_id)
		)
		return [
			record
			for record in items
			if record.get("status") in {"pending", "pending_review", "pending_quota_review", "review_required"}
			or record.get("decision") == "pending"
		]

	def list_records(self, tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
		collections = {
			"upstreams": self.list_upstreams,
			"consumers": self.list_consumers,
			"routes": self.list_routes,
			"quota_reviews": self.list_quota_reviews,
			"policies": self.list_policies,
			"traffic_shifts": self.list_traffic_shifts,
			"deployments": self.list_deployments,
			"gateway_agents": self.list_gateway_agents,
			"lifecycle_batches": self.list_lifecycle_batches,
			"audit_events": self.list_audit_events,
		}
		if record_type:
			if record_type not in collections:
				raise ValueError(f"unsupported record_type: {record_type}")
			return collections[record_type](tenant_id)
		rows: list[dict[str, Any]] = []
		for loader in collections.values():
			rows.extend(loader(tenant_id))
		return rows

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		upstream_id = str(metadata.get("upstream_id") or "manual-upstream")
		if self._tenant_key(tenant_id, upstream_id) not in self._upstreams:
			self.register_upstream(
				upstream_id=upstream_id,
				tenant_id=tenant_id,
				name=str(metadata.get("upstream_name") or upstream_id),
				base_url=str(metadata.get("base_url") or "https://example.internal"),
			owner=str(metadata.get("owner") or "operations"),
			health_check_configured=bool(metadata.get("health_check_configured", True)),
		)
		route = self.request_route(
			route_id=record_id,
			tenant_id=tenant_id,
			path=str(metadata.get("path") or f"/{record_id}"),
			methods=list(metadata.get("methods") or ["GET"]),
			upstream_id=upstream_id,
			owner=str(metadata.get("owner") or "operations"),
			route_exposure=str(metadata.get("route_exposure") or "internal"),
			consumer_id=metadata.get("consumer_id"),
			auth_policy_attached=bool(metadata.get("auth_policy_attached", True)),
			threat_policy_attached=bool(metadata.get("threat_policy_attached", True)),
			mtls_enabled=bool(metadata.get("mtls_enabled", True)),
			rate_limit_configured=bool(metadata.get("rate_limit_configured", True)),
			requested_rps_limit=int(metadata.get("requested_rps_limit") or 1000),
			wasm_filter_attached=bool(metadata.get("wasm_filter_attached", False)),
			filter_signature_verified=bool(metadata.get("filter_signature_verified", True)),
		)
		if "route" in route:
			return route["route"]
		route["status"] = status
		return route

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _enforce_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		_raise_if_blocked(result)

	def _get_upstream(self, tenant_id: str, upstream_id: str) -> GatewayUpstreamRecord:
		upstream = self._upstreams.get(self._tenant_key(tenant_id, upstream_id))
		if upstream is None:
			raise KeyError(f"unknown upstream for tenant: {upstream_id}")
		return upstream

	def _get_consumer(self, tenant_id: str, consumer_id: str) -> GatewayConsumerRecord:
		consumer = self._consumers.get(self._tenant_key(tenant_id, consumer_id))
		if consumer is None:
			raise KeyError(f"unknown consumer for tenant: {consumer_id}")
		return consumer

	def _get_route(self, tenant_id: str, route_id: str) -> GatewayRouteRecord:
		route = self._routes.get(self._tenant_key(tenant_id, route_id))
		if route is None:
			raise KeyError(f"unknown route for tenant: {route_id}")
		return route

	def _dump_tenant_records(self, records: dict[tuple[str, str], Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.model_dump(mode="json") for record in sorted(values, key=lambda item: item.id)]

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		evidence: dict[str, Any] | None = None,
		policy_result: dict[str, Any] | None = None,
		review_recorded: bool = False,
	) -> None:
		result = policy_result or _allow_result()
		self._events.append(
			GatewayAuditEvent(
				tenant_id=tenant_id,
				event_type=event_type,
				subject_id=subject_id,
				message=message,
				policy_decision=result["decision"],
				matched_rules=list(result["matched_rules"]),
				review_reasons=_reasons(result),
				review_evidence=_review_evidence(result, review_recorded),
				evidence=dict(evidence or {}),
			)
		)


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


def _reasons(result: dict[str, Any]) -> list[str]:
	return list(dict.fromkeys(
		str(action["reason"])
		for action in result.get("actions", [])
		if action.get("reason")
	))


def _review_evidence(result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
	return {
		"required_actions": list(dict.fromkeys(
			str(action.get("required_action"))
			for action in result.get("actions", [])
			if action.get("required_action")
		)),
		"reasons": _reasons(result),
		"review_recorded": bool(review_recorded),
	}


def _policy_kwargs(result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
	return {
		"decision": result["decision"],
		"matched_rules": list(result["matched_rules"]),
		"policy_decision": result["decision"],
		"review_reasons": _reasons(result),
		"review_evidence": _review_evidence(result, review_recorded),
	}


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(action.get("reason", "gateway_policy_blocked") for action in result["actions"])
	raise PermissionError(reasons or "gateway_policy_blocked")


def _normalize_agent_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _status_for_decision(decision: str) -> str:
	if decision == "require_review":
		return "pending_review"
	if decision == "deny":
		return "denied"
	return "active"


__all__ = ["ApigService"]
