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

	# ── new methods ─────────────────────────────────────────────────────────

	def api_key_create(
		self,
		key_id: str,
		tenant_id: str,
		consumer_id: str,
		owner: str,
		scopes: list[str] | None = None,
		expiry_days: int = 365,
	) -> dict[str, Any]:
		"""Issue a new API key for a consumer with scopes and TTL."""
		self._enforce_tenant(tenant_id)
		self._get_consumer(tenant_id, consumer_id)
		from datetime import datetime, timezone, timedelta
		import hashlib, secrets
		raw = secrets.token_hex(32)
		key_hash = hashlib.sha256(raw.encode()).hexdigest()
		expires_at = (datetime.now(timezone.utc) + timedelta(days=expiry_days)).isoformat()
		record = {
			"key_id": key_id,
			"tenant_id": tenant_id,
			"consumer_id": consumer_id,
			"owner": owner,
			"scopes": list(scopes or ["read"]),
			"key_hash": key_hash,
			"expires_at": expires_at,
			"status": "active",
		}
		self._policies[self._tenant_key(tenant_id, key_id)] = GatewayPolicyRecord(
			id=key_id,
			tenant_id=tenant_id,
			name=f"api_key:{consumer_id}",
			policy_type="api_key",
			actor=owner,
			status="active",
			decision="allow",
			matched_rules=[],
			policy_decision="allow",
			review_reasons=[],
			review_evidence={},
			metadata=record,
		)
		self._record_event(
			tenant_id=tenant_id,
			event_type="api_key_created",
			subject_id=key_id,
			message=f"Created API key {key_id} for consumer {consumer_id}.",
			evidence={"consumer_id": consumer_id, "scopes": record["scopes"], "expires_at": expires_at},
		)
		return record

	def api_key_revoke(
		self,
		key_id: str,
		tenant_id: str,
		actor: str,
		reason: str = "",
	) -> dict[str, Any]:
		"""Revoke an API key immediately."""
		self._enforce_tenant(tenant_id)
		policy = self._policies.get(self._tenant_key(tenant_id, key_id))
		if policy is None or policy.policy_type != "api_key":
			raise KeyError(f"unknown api_key: {key_id}")
		metadata = dict(policy.metadata)
		metadata["status"] = "revoked"
		metadata["revoked_by"] = actor
		metadata["revoke_reason"] = reason
		revoked = GatewayPolicyRecord(
			id=policy.id,
			tenant_id=policy.tenant_id,
			name=policy.name,
			policy_type=policy.policy_type,
			actor=actor,
			status="revoked",
			decision="allow",
			matched_rules=[],
			policy_decision="allow",
			review_reasons=[],
			review_evidence={},
			metadata=metadata,
		)
		self._policies[self._tenant_key(tenant_id, key_id)] = revoked
		self._record_event(
			tenant_id=tenant_id,
			event_type="api_key_revoked",
			subject_id=key_id,
			message=f"Revoked API key {key_id}.",
			evidence={"actor": actor, "reason": reason},
		)
		return metadata

	def rate_limit_apply(
		self,
		rule_id: str,
		tenant_id: str,
		route_id: str,
		requests_per_minute: int,
		actor: str,
		burst_multiplier: float = 1.5,
	) -> dict[str, Any]:
		"""Apply a rate-limiting rule to a route."""
		self._enforce_tenant(tenant_id)
		route = self._get_route(tenant_id, route_id)
		record = {
			"rule_id": rule_id,
			"tenant_id": tenant_id,
			"route_id": route_id,
			"requests_per_minute": requests_per_minute,
			"burst_limit": int(requests_per_minute * burst_multiplier),
			"actor": actor,
			"status": "active",
		}
		self._policies[self._tenant_key(tenant_id, rule_id)] = GatewayPolicyRecord(
			id=rule_id,
			tenant_id=tenant_id,
			name=f"rate_limit:{route_id}",
			policy_type="rate_limiting",
			actor=actor,
			status="active",
			decision="allow",
			matched_rules=[],
			policy_decision="allow",
			review_reasons=[],
			review_evidence={},
			metadata=record,
		)
		self._record_event(
			tenant_id=tenant_id,
			event_type="rate_limit_applied",
			subject_id=rule_id,
			message=f"Rate limit {requests_per_minute} rpm applied to route {route_id}.",
			evidence={"route_id": route_id, "requests_per_minute": requests_per_minute},
		)
		return record

	def quota_tracking(
		self,
		tenant_id: str,
		consumer_id: str,
	) -> dict[str, Any]:
		"""Return quota usage statistics for a consumer."""
		self._enforce_tenant(tenant_id)
		self._get_consumer(tenant_id, consumer_id)
		routes = self.list_routes(tenant_id)
		consumer_routes = [r for r in routes if r.get("consumer_id") == consumer_id]
		total_rps = sum(r.get("requested_rps_limit", 0) for r in consumer_routes)
		return {
			"tenant_id": tenant_id,
			"consumer_id": consumer_id,
			"route_count": len(consumer_routes),
			"total_rps_limit": total_rps,
			"routes": [r["id"] for r in consumer_routes],
		}

	def transformation_apply(
		self,
		transform_id: str,
		tenant_id: str,
		route_id: str,
		request_transforms: dict[str, Any] | None = None,
		response_transforms: dict[str, Any] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Register request/response transformation rules for a route."""
		self._enforce_tenant(tenant_id)
		self._get_route(tenant_id, route_id)
		record = {
			"transform_id": transform_id,
			"tenant_id": tenant_id,
			"route_id": route_id,
			"request_transforms": dict(request_transforms or {}),
			"response_transforms": dict(response_transforms or {}),
			"actor": actor,
			"status": "active",
		}
		self._policies[self._tenant_key(tenant_id, transform_id)] = GatewayPolicyRecord(
			id=transform_id,
			tenant_id=tenant_id,
			name=f"transform:{route_id}",
			policy_type="transformation",
			actor=actor,
			status="active",
			decision="allow",
			matched_rules=[],
			policy_decision="allow",
			review_reasons=[],
			review_evidence={},
			metadata=record,
		)
		self._record_event(
			tenant_id=tenant_id,
			event_type="transformation_applied",
			subject_id=transform_id,
			message=f"Transformation {transform_id} applied to route {route_id}.",
			evidence={"route_id": route_id},
		)
		return record

	def mock_endpoint(
		self,
		mock_id: str,
		tenant_id: str,
		path: str,
		methods: list[str],
		response_body: dict[str, Any],
		status_code: int = 200,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Register a mock endpoint that returns a canned response."""
		self._enforce_tenant(tenant_id)
		record = {
			"mock_id": mock_id,
			"tenant_id": tenant_id,
			"path": path,
			"methods": [m.upper() for m in methods],
			"response_body": response_body,
			"status_code": status_code,
			"actor": actor,
			"status": "active",
		}
		self._policies[self._tenant_key(tenant_id, mock_id)] = GatewayPolicyRecord(
			id=mock_id,
			tenant_id=tenant_id,
			name=f"mock:{path}",
			policy_type="mock",
			actor=actor,
			status="active",
			decision="allow",
			matched_rules=[],
			policy_decision="allow",
			review_reasons=[],
			review_evidence={},
			metadata=record,
		)
		self._record_event(
			tenant_id=tenant_id,
			event_type="mock_endpoint_registered",
			subject_id=mock_id,
			message=f"Mock endpoint {mock_id} registered at {path}.",
			evidence={"path": path, "status_code": status_code},
		)
		return record

	def documentation_generate(
		self,
		tenant_id: str,
		gateway_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate OpenAPI-style documentation from registered routes and upstreams."""
		routes = self.list_routes(tenant_id)
		upstreams = self.list_upstreams(tenant_id)
		paths: dict[str, Any] = {}
		for route in routes:
			path = route["path"]
			paths.setdefault(path, {})
			for method in route.get("methods", []):
				paths[path][method.lower()] = {
					"summary": f"Route to {route.get('upstream_id')}",
					"security": [{"BearerAuth": []}] if route.get("auth_policy_attached") else [],
					"x-rate-limit": route.get("requested_rps_limit"),
					"x-exposure": route.get("route_exposure"),
				}
		return {
			"tenant_id": tenant_id,
			"openapi": "3.1.0",
			"info": {"title": f"APG Gateway API (tenant={tenant_id})", "version": "1.0.0"},
			"paths": paths,
			"upstream_count": len(upstreams),
			"route_count": len(routes),
		}

	def version_manage(
		self,
		version_id: str,
		tenant_id: str,
		route_id: str,
		api_version: str,
		actor: str,
		deprecated: bool = False,
	) -> dict[str, Any]:
		"""Register an API version tag for a route."""
		self._enforce_tenant(tenant_id)
		self._get_route(tenant_id, route_id)
		record = {
			"version_id": version_id,
			"tenant_id": tenant_id,
			"route_id": route_id,
			"api_version": api_version,
			"deprecated": deprecated,
			"actor": actor,
			"status": "active",
		}
		self._policies[self._tenant_key(tenant_id, version_id)] = GatewayPolicyRecord(
			id=version_id,
			tenant_id=tenant_id,
			name=f"version:{route_id}:{api_version}",
			policy_type="versioning",
			actor=actor,
			status="active",
			decision="allow",
			matched_rules=[],
			policy_decision="allow",
			review_reasons=[],
			review_evidence={},
			metadata=record,
		)
		self._record_event(
			tenant_id=tenant_id,
			event_type="api_version_registered",
			subject_id=version_id,
			message=f"Registered version {api_version} for route {route_id}.",
			evidence={"api_version": api_version, "deprecated": deprecated},
		)
		return record

	def deprecation_notice(
		self,
		notice_id: str,
		tenant_id: str,
		route_id: str,
		sunset_date: str,
		migration_url: str,
		actor: str,
	) -> dict[str, Any]:
		"""Attach a deprecation notice to a route with a sunset date."""
		self._enforce_tenant(tenant_id)
		route = self._get_route(tenant_id, route_id)
		record = {
			"notice_id": notice_id,
			"tenant_id": tenant_id,
			"route_id": route_id,
			"route_path": route.path,
			"sunset_date": sunset_date,
			"migration_url": migration_url,
			"actor": actor,
			"status": "active",
		}
		self._policies[self._tenant_key(tenant_id, notice_id)] = GatewayPolicyRecord(
			id=notice_id,
			tenant_id=tenant_id,
			name=f"deprecation:{route_id}",
			policy_type="deprecation",
			actor=actor,
			status="active",
			decision="allow",
			matched_rules=[],
			policy_decision="allow",
			review_reasons=[],
			review_evidence={},
			metadata=record,
		)
		self._record_event(
			tenant_id=tenant_id,
			event_type="deprecation_notice_created",
			subject_id=notice_id,
			message=f"Deprecation notice for {route.path} — sunset {sunset_date}.",
			evidence={"sunset_date": sunset_date, "migration_url": migration_url},
		)
		return record

	def developer_portal(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return a developer portal payload with discoverable API catalogue."""
		routes = self.list_routes(tenant_id)
		consumers = self.list_consumers(tenant_id)
		docs = self.documentation_generate(tenant_id)
		public_routes = [r for r in routes if r.get("route_exposure") == "public"]
		return {
			"tenant_id": tenant_id,
			"portal_title": f"Developer Portal — tenant {tenant_id}",
			"public_api_count": len(public_routes),
			"consumer_count": len(consumers),
			"openapi_spec": docs,
			"public_routes": public_routes,
		}

	def usage_analytics(
		self,
		tenant_id: str,
		route_id: str | None = None,
	) -> dict[str, Any]:
		"""Return usage analytics aggregated from traffic shift and deployment records."""
		routes = self.list_routes(tenant_id)
		if route_id:
			routes = [r for r in routes if r["id"] == route_id]
		shifts = self.list_traffic_shifts(tenant_id)
		deployments = self.list_deployments(tenant_id)
		total_rps = sum(r.get("requested_rps_limit", 0) for r in routes)
		return {
			"tenant_id": tenant_id,
			"route_count": len(routes),
			"total_rps_capacity": total_rps,
			"traffic_shift_count": len(shifts),
			"deployment_count": len(deployments),
			"active_deployments": len([d for d in deployments if d.get("status") == "deployed"]),
		}

	def sla_monitoring(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return SLA compliance metrics for the gateway."""
		routes = self.list_routes(tenant_id)
		upstreams = self.list_upstreams(tenant_id)
		healthy_upstreams = [u for u in upstreams if u.get("health") == "healthy"]
		active_routes = [r for r in routes if r.get("status") == "active"]
		sla_score = (len(healthy_upstreams) / max(len(upstreams), 1)) * 100
		return {
			"tenant_id": tenant_id,
			"sla_score": round(sla_score, 1),
			"healthy_upstream_count": len(healthy_upstreams),
			"total_upstream_count": len(upstreams),
			"active_route_count": len(active_routes),
			"sla_status": "met" if sla_score >= 99.9 else "at_risk" if sla_score >= 95.0 else "breached",
		}

	def api_discovery(
		self,
		tenant_id: str,
		keyword: str = "",
	) -> list[dict[str, Any]]:
		"""Discover routes matching an optional keyword in path or owner."""
		routes = self.list_routes(tenant_id)
		if keyword:
			kw = keyword.lower()
			routes = [r for r in routes if kw in r.get("path", "").lower() or kw in r.get("owner", "").lower()]
		return routes

	def schema_validate(
		self,
		tenant_id: str,
		route_id: str,
		payload: dict[str, Any],
		schema: dict[str, Any],
	) -> dict[str, Any]:
		"""Validate a payload against a JSON schema for a route (structural check)."""
		self._enforce_tenant(tenant_id)
		self._get_route(tenant_id, route_id)
		required = schema.get("required", [])
		missing = [k for k in required if k not in payload]
		extra_props = schema.get("additionalProperties", True)
		unexpected: list[str] = []
		if not extra_props:
			allowed = set(schema.get("properties", {}).keys())
			unexpected = [k for k in payload if k not in allowed]
		valid = not missing and not unexpected
		return {
			"tenant_id": tenant_id,
			"route_id": route_id,
			"valid": valid,
			"missing_required_fields": missing,
			"unexpected_fields": unexpected,
		}

	def security_scan(
		self,
		tenant_id: str,
		route_id: str | None = None,
	) -> dict[str, Any]:
		"""Run a security posture scan across routes."""
		self._enforce_tenant(tenant_id)
		routes = self.list_routes(tenant_id)
		if route_id:
			routes = [r for r in routes if r["id"] == route_id]
		findings: list[dict[str, Any]] = []
		for r in routes:
			if not r.get("auth_policy_attached"):
				findings.append({"route_id": r["id"], "severity": "high", "issue": "no_auth_policy"})
			if not r.get("threat_policy_attached"):
				findings.append({"route_id": r["id"], "severity": "medium", "issue": "no_threat_policy"})
			if not r.get("mtls_enabled") and r.get("route_exposure") == "public":
				findings.append({"route_id": r["id"], "severity": "high", "issue": "public_route_no_mtls"})
		return {
			"tenant_id": tenant_id,
			"routes_scanned": len(routes),
			"finding_count": len(findings),
			"high_severity": len([f for f in findings if f["severity"] == "high"]),
			"medium_severity": len([f for f in findings if f["severity"] == "medium"]),
			"findings": findings,
			"status": "clean" if not findings else "issues_found",
		}

	def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return gateway service health."""
		summary = self.gateway_summary(tenant_id)
		return {
			"status": "healthy",
			"tenant_id": tenant_id,
			"upstream_count": summary["upstream_count"],
			"active_route_count": summary["active_route_count"],
			"pending_review_count": summary["pending_review_count"],
		}

	def dashboard(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return aggregated KPI dashboard."""
		summary = self.gateway_summary(tenant_id)
		sla = self.sla_monitoring(tenant_id)
		scan = self.security_scan(tenant_id)
		return {
			**summary,
			"sla": sla,
			"security_findings": scan["finding_count"],
			"health": self.health_check(tenant_id),
		}

	def export_routes(
		self,
		tenant_id: str,
		export_format: str = "json",
	) -> dict[str, Any]:
		"""Export route definitions."""
		routes = self.list_routes(tenant_id)
		if export_format == "csv":
			keys = list(routes[0].keys()) if routes else []
			lines = [",".join(keys)] + [",".join(str(r.get(k, "")) for k in keys) for r in routes]
			data = "\n".join(lines)
		else:
			import json as _json
			data = _json.dumps(routes, default=str, indent=2)
		return {"tenant_id": tenant_id, "format": export_format, "count": len(routes), "data": data}

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
