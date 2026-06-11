"""Service layer for the Zero Trust Network Access capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .zero_trust_runtime import (
	ZeroTrustAccessRequestRecord,
	ZeroTrustAgentRecord,
	ZeroTrustAuditEventRecord,
	ZeroTrustDeviceRecord,
	ZeroTrustIdentityRecord,
	ZeroTrustResourceRecord,
	ZeroTrustSessionRecord,
	ZtnaLifecycleBatchRecord,
	bounded_score,
	stable_id,
	utc_now,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class ZtnaService:
	"""Dependency-light zero-trust runtime behind the capability contract."""

	def __init__(self) -> None:
		self._identities: dict[str, ZeroTrustIdentityRecord] = {}
		self._devices: dict[str, ZeroTrustDeviceRecord] = {}
		self._resources: dict[str, ZeroTrustResourceRecord] = {}
		self._access_requests: dict[str, ZeroTrustAccessRequestRecord] = {}
		self._sessions: dict[str, ZeroTrustSessionRecord] = {}
		self._zero_trust_agents: dict[str, ZeroTrustAgentRecord] = {}
		self._lifecycle_batches: dict[str, ZtnaLifecycleBatchRecord] = {}
		self._audit_events: list[ZeroTrustAuditEventRecord] = []
		self._agent_runtimes = set(DEFAULT_CONFIGURATION["agents"]["supported_runtimes"])
		self._agent_roles = set(DEFAULT_CONFIGURATION["agents"]["supported_roles"])
		self._privileged_agent_roles = set(DEFAULT_CONFIGURATION["agents"]["privileged_roles"])
		self._lifecycle_operations = set(DEFAULT_CONFIGURATION["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_identity(
		self,
		identity_key: str,
		tenant_id: str,
		subject_id: str,
		display_name: str,
		verified: bool = False,
		privileged: bool = False,
		mfa_completed: bool = False,
		federated_provider: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_identity",
			"subject_present": bool(subject_id),
			"display_name_present": bool(display_name),
			"federated_identity": bool(federated_provider),
			"federated_provider_present": bool(federated_provider),
		})
		self._raise_if_denied(result)
		identity_id = stable_id("identity", tenant_id, identity_key)
		identity = ZeroTrustIdentityRecord(
			id=identity_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			display_name=display_name,
			verified=verified,
			privileged=privileged,
			mfa_completed=mfa_completed,
			status="verified" if verified else "pending",
			federated_provider=federated_provider,
			verified_at=utc_now() if verified else None,
			metadata=dict(metadata or {}),
		)
		self._identities[identity_id] = identity
		self._audit(tenant_id, "identity_registered", identity_id, subject_id, {"verified": verified, "privileged": privileged})
		return identity.to_dict()

	def verify_identity(self, identity_id: str, actor_id: str, mfa_completed: bool | None = None) -> dict[str, Any]:
		identity = self._get_identity(identity_id)
		identity.verified = True
		identity.status = "verified"
		identity.verified_at = utc_now()
		if mfa_completed is not None:
			identity.mfa_completed = bool(mfa_completed)
		self._audit(identity.tenant_id, "identity_verified", identity.id, actor_id, {"mfa_completed": identity.mfa_completed})
		return identity.to_dict()

	def register_device(
		self,
		device_key: str,
		tenant_id: str,
		identity_id: str,
		name: str,
		trust_score: float,
		posture_present: bool = True,
		managed: bool = False,
		attested: bool = False,
		compliant: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		identity = self._get_identity(identity_id)
		if identity.tenant_id != tenant_id:
			self._raise_cross_tenant()
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_device",
			"identity_present": bool(identity),
		})
		self._raise_if_denied(result)
		score = bounded_score(trust_score)
		status = "trusted" if posture_present and compliant and score >= 0.7 else "quarantined"
		device_id = stable_id("device", tenant_id, identity_id, device_key)
		device = ZeroTrustDeviceRecord(
			id=device_id,
			tenant_id=tenant_id,
			identity_id=identity_id,
			name=name,
			trust_score=score,
			posture_present=posture_present,
			managed=managed,
			attested=attested,
			compliant=compliant,
			status=status,
			metadata=dict(metadata or {}),
		)
		self._devices[device_id] = device
		self._audit(tenant_id, "device_registered", device_id, identity.subject_id, {"trust_score": score, "status": status})
		return device.to_dict()

	def update_device_posture(
		self,
		device_id: str,
		trust_score: float,
		posture_present: bool = True,
		compliant: bool = True,
		attested: bool | None = None,
		actor_id: str = "system",
	) -> dict[str, Any]:
		device = self._get_device(device_id)
		device.trust_score = bounded_score(trust_score)
		device.posture_present = posture_present
		device.compliant = compliant
		if attested is not None:
			device.attested = attested
		device.status = "trusted" if posture_present and compliant and device.trust_score >= 0.7 else "quarantined"
		device.last_posture_at = utc_now()
		self._audit(device.tenant_id, "device_posture_updated", device.id, actor_id, {"trust_score": device.trust_score, "status": device.status})
		return device.to_dict()

	def register_resource(
		self,
		resource_key: str,
		tenant_id: str,
		name: str,
		access_level: str = "standard",
		sensitive: bool = False,
		policy_attached: bool = False,
		policy_id: str | None = None,
		network_segment: str = "default",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_resource",
			"resource_name_present": bool(name),
			"network_segment_present": bool(network_segment),
			"sensitive_resource": sensitive,
			"microsegmentation_present": bool(network_segment),
		})
		self._raise_if_denied(result)
		resource_id = stable_id("resource", tenant_id, resource_key)
		resource = ZeroTrustResourceRecord(
			id=resource_id,
			tenant_id=tenant_id,
			name=name,
			access_level=access_level,
			sensitive=sensitive,
			policy_attached=policy_attached,
			policy_id=policy_id,
			network_segment=network_segment,
			status="active" if policy_attached else "policy_required",
			metadata=dict(metadata or {}),
		)
		self._resources[resource_id] = resource
		self._audit(tenant_id, "resource_registered", resource_id, "system", {"policy_attached": policy_attached, "access_level": access_level})
		return resource.to_dict()

	def attach_resource_policy(self, resource_id: str, policy_id: str, actor_id: str) -> dict[str, Any]:
		resource = self._get_resource(resource_id)
		result = self.evaluate({
			"tenant_context_present": bool(resource.tenant_id),
			"operation": "attach_resource_policy",
			"policy_present": bool(policy_id),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		resource.policy_attached = True
		resource.policy_id = policy_id
		resource.status = "active"
		self._audit(resource.tenant_id, "resource_policy_attached", resource.id, actor_id, {"policy_id": policy_id})
		return resource.to_dict()

	def request_access(
		self,
		identity_id: str,
		device_id: str,
		resource_id: str,
		requested_by: str,
		mfa_completed: bool | None = None,
		access_review_recorded: bool = False,
		just_in_time_approval_present: bool = False,
		least_privilege_scope_present: bool = True,
		explicit_access_decision_present: bool = True,
		access_risk_score: float | None = None,
	) -> dict[str, Any]:
		identity = self._get_identity(identity_id)
		device = self._get_device(device_id)
		resource = self._get_resource(resource_id)
		self._assert_same_tenant(identity.tenant_id, device.tenant_id, resource.tenant_id)
		mfa_ok = identity.mfa_completed if mfa_completed is None else bool(mfa_completed)
		risk = bounded_score(access_risk_score if access_risk_score is not None else self._risk_score(identity, device, resource))
		duplicate_pending_review = any(
			request.tenant_id == identity.tenant_id
			and request.identity_id == identity_id
			and request.device_id == device_id
			and request.resource_id == resource_id
			and request.status == "review_required"
			for request in self._access_requests.values()
		)
		context = {
			"operation": "request_access",
			"tenant_context_present": bool(identity.tenant_id),
			"identity_verified": identity.verified,
			"identity_status": identity.status,
			"device_posture_present": device.posture_present,
			"device_trust_score": device.trust_score,
			"device_compliant": device.compliant,
			"device_attested": device.attested,
			"managed_device": device.managed,
			"resource_policy_attached": resource.policy_attached,
			"sensitive_resource": resource.sensitive,
			"microsegmentation_present": bool(resource.network_segment),
			"access_level": resource.access_level,
			"mfa_completed": mfa_ok,
			"access_risk_score": risk,
			"access_review_recorded": access_review_recorded,
			"just_in_time_approval_present": just_in_time_approval_present,
			"least_privilege_scope_present": least_privilege_scope_present,
			"explicit_access_decision_present": explicit_access_decision_present,
			"duplicate_pending_review": duplicate_pending_review,
			"access_decision_recorded": True,
			"audit_event_recorded": True,
		}
		result = self.evaluate(context)
		deny_reasons = [action.get("reason", "access_denied") for action in result["actions"] if action.get("decision") == "deny"]
		if deny_reasons:
			raise PermissionError(", ".join(deny_reasons))
		required_actions = [action.get("required_action", "review_required") for action in result["actions"] if action.get("decision") == "require_review"]
		status = "review_required" if required_actions else "approved"
		request_id = stable_id("access", identity.tenant_id, identity_id, device_id, resource_id, len(self._access_requests) + 1)
		record = ZeroTrustAccessRequestRecord(
			id=request_id,
			tenant_id=identity.tenant_id,
			identity_id=identity_id,
			device_id=device_id,
			resource_id=resource_id,
			requested_by=requested_by,
			access_level=resource.access_level,
			risk_score=risk,
			status=status,
			required_actions=required_actions,
			matched_rules=list(result["matched_rules"]),
			decision_reasons=[action.get("reason", "review_required") for action in result["actions"]],
		)
		self._access_requests[request_id] = record
		self._audit(identity.tenant_id, "access_requested", request_id, requested_by, {"status": status, "risk_score": risk})
		return record.to_dict()

	def approve_access_request(self, request_id: str, reviewer_id: str) -> dict[str, Any]:
		request = self._get_access_request(request_id)
		if request.status not in {"review_required", "approved"}:
			raise PermissionError("access_request_not_reviewable")
		result = self.evaluate({
			"tenant_context_present": bool(request.tenant_id),
			"operation": "approve_access_request",
			"reviewer_same_as_requester": reviewer_id == request.requested_by,
			"notes_present": True,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		request.status = "approved"
		request.required_actions = []
		request.reviewed_by = reviewer_id
		request.reviewed_at = utc_now()
		self._audit(request.tenant_id, "access_request_approved", request.id, reviewer_id, {"risk_score": request.risk_score})
		return request.to_dict()

	def start_session(self, request_id: str, actor_id: str) -> dict[str, Any]:
		request = self._get_access_request(request_id)
		result = self.evaluate({
			"tenant_context_present": bool(request.tenant_id),
			"operation": "start_session",
			"access_request_approved": request.status == "approved",
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		session_id = stable_id("session", request.tenant_id, request.id)
		session = ZeroTrustSessionRecord(
			id=session_id,
			tenant_id=request.tenant_id,
			access_request_id=request.id,
			identity_id=request.identity_id,
			device_id=request.device_id,
			resource_id=request.resource_id,
			risk_score=request.risk_score,
		)
		request.status = "active"
		self._sessions[session_id] = session
		self._audit(request.tenant_id, "session_started", session_id, actor_id, {"access_request_id": request.id})
		return session.to_dict()

	def reevaluate_session(
		self,
		session_id: str,
		risk_score: float,
		identity_verified: bool = True,
		device_posture_present: bool = True,
		access_review_recorded: bool = False,
		actor_id: str = "system",
	) -> dict[str, Any]:
		session = self._get_session(session_id)
		request = self._get_access_request(session.access_request_id)
		resource = self._get_resource(session.resource_id)
		risk = bounded_score(risk_score)
		result = self.evaluate({
			"tenant_context_present": bool(session.tenant_id),
			"operation": "reevaluate_session",
			"identity_verified": identity_verified,
			"device_posture_present": device_posture_present,
			"resource_policy_attached": resource.policy_attached,
			"access_level": request.access_level,
			"mfa_completed": True,
			"access_risk_score": risk,
			"access_review_recorded": access_review_recorded,
			"continuous_verification_present": True,
		})
		session.risk_score = risk
		if result["decision"] == "deny":
			session.status = "revoked"
			session.ended_at = utc_now()
			session.reauth_required = True
		elif result["decision"] == "require_review":
			session.status = "review_required"
			session.reauth_required = True
		else:
			session.status = "active"
			session.reauth_required = False
		self._audit(session.tenant_id, "session_reevaluated", session.id, actor_id, {"decision": result["decision"], "risk_score": risk})
		return session.to_dict()

	def close_session(self, session_id: str, actor_id: str) -> dict[str, Any]:
		session = self._get_session(session_id)
		result = self.evaluate({
			"tenant_context_present": bool(session.tenant_id),
			"operation": "close_session",
			"actor_present": bool(actor_id),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		session.status = "closed"
		session.ended_at = utc_now()
		self._audit(session.tenant_id, "session_closed", session.id, actor_id, {})
		return session.to_dict()

	def register_zero_trust_agent(
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
		self._require_tenant(tenant_id)
		record_id = stable_id("ztna_agent", tenant_id, agent_id)
		if record_id in self._zero_trust_agents:
			raise ValueError(f"zero_trust_agent_already_exists:{agent_id}")
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_zero_trust_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		self._raise_if_denied(result)
		if not str(name or "").strip():
			raise ValueError("zero_trust_agent_name_required")
		agent = ZeroTrustAgentRecord(
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
			status="pending_review" if result["decision"] == "require_review" else "active",
		)
		self._zero_trust_agents[record_id] = agent
		self._audit(tenant_id, "zero_trust_agent_registered", record_id, owner, {**agent.to_dict(), "rule_decision": result["decision"]})
		return agent.to_dict()

	def validate_ztna_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "ztna_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("ztna_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_ztna_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_ztna_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
		})
		accepted = result["decision"] == "allow"
		record_id = stable_id("ztna_batch", tenant_id, batch_id or len(self._lifecycle_batches) + 1)
		record = ZtnaLifecycleBatchRecord(
			id=record_id,
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[record_id] = record
		self._audit(tenant_id, f"ztna_lifecycle_batch_{record.status}", record_id, "bytewax", record.to_dict())
		self._raise_if_denied(result)
		return record.to_dict()

	def list_identities(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in sorted(self._filter(self._identities.values(), tenant_id), key=lambda item: item.display_name)]

	def list_devices(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in sorted(self._filter(self._devices.values(), tenant_id), key=lambda item: item.name)]

	def list_resources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in sorted(self._filter(self._resources.values(), tenant_id), key=lambda item: item.name)]

	def list_access_requests(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in sorted(self._filter(self._access_requests.values(), tenant_id), key=lambda item: item.created_at)]

	def list_sessions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in sorted(self._filter(self._sessions.values(), tenant_id), key=lambda item: item.started_at)]

	def list_zero_trust_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in sorted(self._filter(self._zero_trust_agents.values(), tenant_id), key=lambda item: item.name)]

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in sorted(self._filter(self._lifecycle_batches.values(), tenant_id), key=lambda item: item.created_at)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in sorted(self._filter(self._audit_events, tenant_id), key=lambda item: item.created_at)]

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		identities = self.list_identities(tenant_id)
		devices = self.list_devices(tenant_id)
		resources = self.list_resources(tenant_id)
		requests = self.list_access_requests(tenant_id)
		sessions = self.list_sessions(tenant_id)
		return {
			"tenant_id": tenant_id,
			"identity_count": len(identities),
			"verified_identity_count": sum(1 for identity in identities if identity["verified"]),
			"device_count": len(devices),
			"trusted_device_count": sum(1 for device in devices if device["status"] == "trusted"),
			"resource_count": len(resources),
			"policy_required_resource_count": sum(1 for resource in resources if not resource["policy_attached"]),
			"access_request_count": len(requests),
			"access_review_count": sum(1 for request in requests if request["status"] == "review_required"),
			"active_session_count": sum(1 for session in sessions if session["status"] == "active"),
			"revoked_session_count": sum(1 for session in sessions if session["status"] == "revoked"),
			"zero_trust_agent_count": len(self.list_zero_trust_agents(tenant_id)),
			"pending_agent_review_count": sum(1 for item in self.list_zero_trust_agents(tenant_id) if item["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": sum(1 for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"),
		}

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		meta = dict(metadata or {})
		return self.register_resource(
			resource_key=record_id,
			tenant_id=tenant_id,
			name=str(meta.get("name") or record_id),
			access_level=str(meta.get("access_level") or "standard"),
			sensitive=bool(meta.get("sensitive", False)),
			policy_attached=status == "active" or bool(meta.get("policy_attached", False)),
			policy_id=meta.get("policy_id"),
			metadata={"compatibility_status": status, **meta},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_resources(tenant_id)

	# ── New async methods ─────────────────────────────────────────────────────

	async def async_register_identity(
		self,
		identity_key: str,
		tenant_id: str,
		subject_id: str,
		display_name: str,
		verified: bool = False,
		privileged: bool = False,
		mfa_completed: bool = False,
		federated_provider: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Async wrapper for register_identity — safe for use in async adapters and I/O contexts."""
		return self.register_identity(
			identity_key=identity_key,
			tenant_id=tenant_id,
			subject_id=subject_id,
			display_name=display_name,
			verified=verified,
			privileged=privileged,
			mfa_completed=mfa_completed,
			federated_provider=federated_provider,
			metadata=metadata,
		)

	async def async_request_access(
		self,
		identity_id: str,
		device_id: str,
		resource_id: str,
		requested_by: str,
		mfa_completed: bool | None = None,
		access_review_recorded: bool = False,
		just_in_time_approval_present: bool = False,
		least_privilege_scope_present: bool = True,
		explicit_access_decision_present: bool = True,
		access_risk_score: float | None = None,
	) -> dict[str, Any]:
		"""Async access request suitable for concurrent broker fan-out patterns."""
		return self.request_access(
			identity_id=identity_id,
			device_id=device_id,
			resource_id=resource_id,
			requested_by=requested_by,
			mfa_completed=mfa_completed,
			access_review_recorded=access_review_recorded,
			just_in_time_approval_present=just_in_time_approval_present,
			least_privilege_scope_present=least_privilege_scope_present,
			explicit_access_decision_present=explicit_access_decision_present,
			access_risk_score=access_risk_score,
		)

	async def async_reevaluate_session(
		self,
		session_id: str,
		risk_score: float,
		identity_verified: bool = True,
		device_posture_present: bool = True,
		access_review_recorded: bool = False,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Async session reevaluation — called from telemetry pipelines on posture change events."""
		return self.reevaluate_session(
			session_id=session_id,
			risk_score=risk_score,
			identity_verified=identity_verified,
			device_posture_present=device_posture_present,
			access_review_recorded=access_review_recorded,
			actor_id=actor_id,
		)

	async def async_update_device_posture(
		self,
		device_id: str,
		trust_score: float,
		posture_present: bool = True,
		compliant: bool = True,
		attested: bool | None = None,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Async posture update — suitable for continuous telemetry ingestion loops."""
		return self.update_device_posture(
			device_id=device_id,
			trust_score=trust_score,
			posture_present=posture_present,
			compliant=compliant,
			attested=attested,
			actor_id=actor_id,
		)

	async def async_bulk_reevaluate_sessions(
		self,
		tenant_id: str,
		risk_score: float = 0.5,
		actor_id: str = "system",
	) -> list[dict[str, Any]]:
		"""Re-evaluate all active sessions for a tenant in one async sweep.

		Useful when a tenant-level policy change or identity revocation requires
		all in-flight sessions to be checked immediately.
		"""
		import asyncio
		active = [s for s in self._sessions.values() if s.tenant_id == tenant_id and s.status == "active"]
		tasks = [
			self.async_reevaluate_session(
				session_id=s.id,
				risk_score=risk_score,
				actor_id=actor_id,
			)
			for s in active
		]
		if not tasks:
			return []
		return list(await asyncio.gather(*tasks, return_exceptions=False))

	async def async_evaluate_policy(
		self,
		identity_id: str,
		resource_id: str,
		action: str,
		context: dict[str, Any] | None = None,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Async policy evaluation for identity-resource-action triples.

		Resolves the identity and resource, builds a full evaluation context, runs
		the rule engine, and returns an enriched decision payload with matched
		rules and deny reasons. Does not mutate session state.
		"""
		ctx = dict(context or {})
		try:
			identity = self._get_identity(identity_id)
			resource = self._get_resource(resource_id)
		except KeyError as exc:
			return {"allowed": False, "reason": str(exc), "action": action}
		if identity.tenant_id != resource.tenant_id:
			return {
				"allowed": False,
				"reason": "cross_tenant_access_denied",
				"action": action,
			}
		ctx.update({
			"operation": "evaluate_policy",
			"tenant_context_present": bool(identity.tenant_id),
			"identity_verified": identity.verified,
			"identity_status": identity.status,
			"resource_policy_attached": resource.policy_attached,
			"sensitive_resource": resource.sensitive,
			"access_level": resource.access_level,
			"mfa_completed": identity.mfa_completed,
			"microsegmentation_present": bool(resource.network_segment),
		})
		result = self.evaluate(ctx)
		allowed = result["decision"] == "allow"
		self._audit(
			identity.tenant_id,
			"policy_evaluated",
			resource_id,
			actor_id,
			{"identity_id": identity_id, "action": action, "decision": result["decision"]},
		)
		return {
			"allowed": allowed,
			"decision": result["decision"],
			"action": action,
			"identity_id": identity_id,
			"resource_id": resource_id,
			"matched_rules": list(result.get("matched_rules", [])),
			"deny_reasons": [
				a.get("reason", "policy_denied")
				for a in result.get("actions", [])
				if a.get("decision") == "deny"
			],
			"tenant_id": identity.tenant_id,
		}

	async def async_close_session(
		self,
		session_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Async session close — for use in event-driven session lifecycle handlers."""
		return self.close_session(session_id=session_id, actor_id=actor_id)

	async def async_compliance_snapshot(
		self,
		tenant_id: str,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Async ZTNA compliance snapshot for a tenant.

		Aggregates identity verification, device posture, resource policy coverage,
		active session counts, and pending review backlog into a single dict suitable
		for export to audit dashboards or SIEM pipelines.
		"""
		summary = self.dashboard_summary(tenant_id)
		posture = self._posture_detail(tenant_id)
		self._audit(
			tenant_id,
			"compliance_snapshot_generated",
			tenant_id,
			actor_id,
			{"session_count": summary.get("active_session_count", 0)},
		)
		return {
			"tenant_id": tenant_id,
			"generated_at": utc_now(),
			"summary": summary,
			"posture": posture,
		}

	def _posture_detail(self, tenant_id: str) -> dict[str, Any]:
		"""Return a compact device posture detail dict for a tenant."""
		devices = [d for d in self._devices.values() if d.tenant_id == tenant_id]
		if not devices:
			return {"total": 0, "compliant": 0, "avg_trust": 0.0, "by_status": {}}
		by_status: dict[str, int] = {}
		for d in devices:
			by_status[d.status] = by_status.get(d.status, 0) + 1
		return {
			"total": len(devices),
			"compliant": sum(1 for d in devices if d.compliant),
			"avg_trust": round(sum(d.trust_score for d in devices) / len(devices), 4),
			"by_status": by_status,
		}

	def _risk_score(self, identity: ZeroTrustIdentityRecord, device: ZeroTrustDeviceRecord, resource: ZeroTrustResourceRecord) -> float:
		score = 1.0 - device.trust_score
		if identity.privileged:
			score += 0.2
		if resource.sensitive or resource.access_level == "privileged":
			score += 0.2
		if not device.managed:
			score += 0.1
		return bounded_score(score)

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _get_identity(self, identity_id: str) -> ZeroTrustIdentityRecord:
		try:
			return self._identities[identity_id]
		except KeyError as exc:
			raise KeyError(f"identity_not_found:{identity_id}") from exc

	def _get_device(self, device_id: str) -> ZeroTrustDeviceRecord:
		try:
			return self._devices[device_id]
		except KeyError as exc:
			raise KeyError(f"device_not_found:{device_id}") from exc

	def _get_resource(self, resource_id: str) -> ZeroTrustResourceRecord:
		try:
			return self._resources[resource_id]
		except KeyError as exc:
			raise KeyError(f"resource_not_found:{resource_id}") from exc

	def _get_access_request(self, request_id: str) -> ZeroTrustAccessRequestRecord:
		try:
			return self._access_requests[request_id]
		except KeyError as exc:
			raise KeyError(f"access_request_not_found:{request_id}") from exc

	def _get_session(self, session_id: str) -> ZeroTrustSessionRecord:
		try:
			return self._sessions[session_id]
		except KeyError as exc:
			raise KeyError(f"session_not_found:{session_id}") from exc

	@staticmethod
	def _assert_same_tenant(*tenant_ids: str) -> None:
		if len(set(tenant_ids)) != 1:
			result = evaluate_capability_rules({"tenant_context_present": True, "cross_tenant_access": True})
			raise PermissionError(", ".join(action.get("reason", "cross_tenant_zero_trust_access_denied") for action in result["actions"]))

	def _audit(self, tenant_id: str, action: str, subject_id: str, actor_id: str, details: dict[str, Any] | None = None) -> None:
		event = ZeroTrustAuditEventRecord(
			id=stable_id("audit", tenant_id, action, subject_id, len(self._audit_events) + 1),
			tenant_id=tenant_id,
			action=action,
			subject_id=subject_id,
			actor_id=actor_id,
			details=dict(details or {}),
		)
		self._audit_events.append(event)

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(action.get("reason", "zero_trust_policy_blocked") for action in result["actions"]))

	def _raise_cross_tenant(self) -> None:
		result = self.evaluate({"tenant_context_present": True, "cross_tenant_access": True})
		self._raise_if_denied(result)

	@staticmethod
	def _filter(records: Any, tenant_id: str | None) -> list[Any]:
		items = list(records)
		if tenant_id is None:
			return items
		return [record for record in items if record.tenant_id == tenant_id]


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


# ── 14 new methods ───────────────────────────────────────────────────────────

def _access_policy_evaluate(
	self: "ZtnaService",
	subject_id: str,
	resource_id: str,
	action: str,
	context: dict[str, Any],
	actor_id: str = "system",
) -> dict[str, Any]:
	"""Evaluate whether subject_id may perform action on resource_id."""
	try:
		identity = self._get_identity(subject_id)
		resource = self._get_resource(resource_id)
	except KeyError as exc:
		return {"allowed": False, "reason": str(exc)}
	allowed = identity.verified and resource.access_level != "classified"
	decision = "allow" if allowed else "deny"
	self._audit(identity.tenant_id, "policy_evaluated", resource_id, actor_id,
		{"subject_id": subject_id, "action": action, "decision": decision})
	return {
		"subject_id": subject_id,
		"resource_id": resource_id,
		"action": action,
		"decision": decision,
		"allowed": allowed,
		"context": context,
	}

ZtnaService.access_policy_evaluate = _access_policy_evaluate  # type: ignore[attr-defined]


def _device_trust_level(
	self: "ZtnaService",
	device_id: str,
) -> str:
	"""Return trust level label: high / medium / low / untrusted."""
	device = self._get_device(device_id)
	score = device.trust_score
	if score >= 0.8:
		return "high"
	if score >= 0.5:
		return "medium"
	if score >= 0.2:
		return "low"
	return "untrusted"

ZtnaService.device_trust_level = _device_trust_level  # type: ignore[attr-defined]


def _micro_segment_define(
	self: "ZtnaService",
	segment_name: str,
	resources: list[str],
	allowed_identities: list[str],
	tenant_id: str,
	actor_id: str = "system",
) -> dict[str, Any]:
	"""Define a micro-segment grouping resources and allowed identities."""
	self._require_tenant(tenant_id)
	seg_id = stable_id("mseg", tenant_id, segment_name)
	self._audit(tenant_id, "micro_segment_defined", seg_id, actor_id,
		{"resources": resources, "allowed_identities": allowed_identities})
	return {
		"segment_id": seg_id,
		"segment_name": segment_name,
		"tenant_id": tenant_id,
		"resources": resources,
		"allowed_identities": allowed_identities,
		"created_at": utc_now(),
	}

ZtnaService.micro_segment_define = _micro_segment_define  # type: ignore[attr-defined]


def _session_risk_score(
	self: "ZtnaService",
	session_id: str,
) -> float:
	"""Return the current risk score for a session (1.0 - trust_score proxy)."""
	session = self._get_session(session_id)
	return round(1.0 - bounded_score(session.trust_score), 4)

ZtnaService.session_risk_score = _session_risk_score  # type: ignore[attr-defined]


def _continuous_auth_check(
	self: "ZtnaService",
	session_id: str,
) -> bool:
	"""Return True if the session passes continuous authentication (trust >= 0.5)."""
	session = self._get_session(session_id)
	return session.trust_score >= 0.5 and session.status == "active"

ZtnaService.continuous_auth_check = _continuous_auth_check  # type: ignore[attr-defined]


def _anomaly_in_session(
	self: "ZtnaService",
	session_id: str,
	events: list[dict[str, Any]],
	actor_id: str = "system",
) -> list[dict[str, Any]]:
	"""Detect anomalous events in a session based on risk_score threshold."""
	session = self._get_session(session_id)
	anomalies: list[dict[str, Any]] = []
	for event in events:
		rs = float(event.get("risk_score", 0.0))
		if rs >= 0.7:
			anomalies.append({**event, "flagged": True, "session_id": session_id})
			self._audit(session.tenant_id, "session_anomaly_detected", session_id, actor_id,
				{"event": event, "risk_score": rs})
	return anomalies

ZtnaService.anomaly_in_session = _anomaly_in_session  # type: ignore[attr-defined]


def _terminate_risky_session(
	self: "ZtnaService",
	session_id: str,
	reason: str,
	actor_id: str = "system",
) -> dict[str, Any]:
	"""Terminate a session flagged as high-risk."""
	session = self._get_session(session_id)
	session.status = "terminated"
	session.ended_at = utc_now()
	self._audit(session.tenant_id, "risky_session_terminated", session_id, actor_id, {"reason": reason})
	return {"session_id": session_id, "status": "terminated", "reason": reason, "terminated_at": utc_now()}

ZtnaService.terminate_risky_session = _terminate_risky_session  # type: ignore[attr-defined]


def _policy_simulate(
	self: "ZtnaService",
	subject_id: str,
	resource_id: str,
	action: str,
	actor_id: str = "system",
) -> dict[str, Any]:
	"""Simulate a policy evaluation without writing audit events."""
	try:
		identity = self._get_identity(subject_id)
		resource = self._get_resource(resource_id)
	except KeyError as exc:
		return {"simulated": True, "allowed": False, "reason": str(exc)}
	allowed = identity.verified and resource.access_level != "classified"
	return {
		"simulated": True,
		"subject_id": subject_id,
		"resource_id": resource_id,
		"action": action,
		"allowed": allowed,
		"decision": "allow" if allowed else "deny",
	}

ZtnaService.policy_simulate = _policy_simulate  # type: ignore[attr-defined]


def _ztna_compliance_report(
	self: "ZtnaService",
	period: str,
	actor_id: str = "system",
) -> dict[str, Any]:
	"""Generate a ZTNA compliance report summarising posture and access decisions."""
	all_identities = list(self._identities.values())
	all_devices = list(self._devices.values())
	all_sessions = list(self._sessions.values())
	compliant_devices = sum(1 for d in all_devices if d.compliant)
	active_sessions = sum(1 for s in all_sessions if s.status == "active")
	high_trust = sum(1 for d in all_devices if d.trust_score >= 0.8)
	return {
		"period": period,
		"total_identities": len(all_identities),
		"verified_identities": sum(1 for i in all_identities if i.verified),
		"total_devices": len(all_devices),
		"compliant_devices": compliant_devices,
		"high_trust_devices": high_trust,
		"total_sessions": len(all_sessions),
		"active_sessions": active_sessions,
		"audit_events": len(self._audit_events),
		"generated_at": utc_now(),
	}

ZtnaService.ztna_compliance_report = _ztna_compliance_report  # type: ignore[attr-defined]


def _trust_score_history(
	self: "ZtnaService",
	identity_id: str,
	limit: int = 20,
) -> list[dict[str, Any]]:
	"""Return recent audit events for an identity as a trust score timeline."""
	events = [
		e.to_dict() for e in self._audit_events
		if e.subject_id == identity_id
	]
	return sorted(events, key=lambda e: e.get("id", ""), reverse=True)[:limit]

ZtnaService.trust_score_history = _trust_score_history  # type: ignore[attr-defined]


def _lateral_movement_detect(
	self: "ZtnaService",
	network_events: list[dict[str, Any]],
	threshold: int = 3,
) -> dict[str, Any]:
	"""Detect potential lateral movement: subjects accessing many resources rapidly."""
	access_counts: dict[str, set[str]] = {}
	for evt in network_events:
		subj = evt.get("subject_id", "unknown")
		res = evt.get("resource_id", "unknown")
		access_counts.setdefault(subj, set()).add(res)
	suspects: list[dict[str, Any]] = [
		{"subject_id": s, "unique_resources_accessed": len(r)}
		for s, r in access_counts.items()
		if len(r) >= threshold
	]
	return {
		"events_analysed": len(network_events),
		"threshold": threshold,
		"suspects": suspects,
		"lateral_movement_detected": len(suspects) > 0,
	}

ZtnaService.lateral_movement_detect = _lateral_movement_detect  # type: ignore[attr-defined]


def _ztna_posture_report(
	self: "ZtnaService",
	tenant_id: str,
	actor_id: str = "system",
) -> dict[str, Any]:
	"""Return a device posture summary for a tenant."""
	devices = [d for d in self._devices.values() if d.tenant_id == tenant_id]
	by_status: dict[str, int] = {}
	for d in devices:
		by_status[d.status] = by_status.get(d.status, 0) + 1
	compliant = sum(1 for d in devices if d.compliant)
	avg_trust = round(sum(d.trust_score for d in devices) / max(len(devices), 1), 3)
	return {
		"tenant_id": tenant_id,
		"total_devices": len(devices),
		"compliant_devices": compliant,
		"non_compliant_devices": len(devices) - compliant,
		"compliance_rate_pct": round(compliant / max(len(devices), 1) * 100, 1),
		"avg_trust_score": avg_trust,
		"by_status": by_status,
		"generated_at": utc_now(),
	}

ZtnaService.ztna_posture_report = _ztna_posture_report  # type: ignore[attr-defined]


def _ztna_analytics(
	self: "ZtnaService",
	period: str,
	tenant_id: str | None = None,
) -> dict[str, Any]:
	"""Return ZTNA analytics for a period."""
	identities = self._filter(self._identities.values(), tenant_id)
	devices = self._filter(self._devices.values(), tenant_id)
	sessions = self._filter(self._sessions.values(), tenant_id)
	requests = self._filter(self._access_requests.values(), tenant_id)
	approved = sum(1 for r in requests if r.status == "approved")
	return {
		"period": period,
		"tenant_id": tenant_id,
		"total_identities": len(identities),
		"total_devices": len(devices),
		"total_sessions": len(sessions),
		"total_access_requests": len(requests),
		"approved_requests": approved,
		"approval_rate_pct": round(approved / max(len(requests), 1) * 100, 1),
		"audit_events": len(self._audit_events),
		"generated_at": utc_now(),
	}

ZtnaService.ztna_analytics = _ztna_analytics  # type: ignore[attr-defined]


# ── Extended methods injected onto ZtnaService ────────────────────────────────

def access_policy_create_ztna(self, name, resource_id, actions, conditions, tenant_id="default"):
	"""ZTNA capability: access policy create ztna."""
	policy={"id":str(__import__("uuid").uuid4()),"name":name,"resource_id":resource_id,"actions":actions,"conditions":conditions,"tenant_id":tenant_id,"status":"active"}
	self._store.setdefault(f"ztna_policies:{tenant_id}",{})[policy["id"]]=policy
	return policy

ZtnaService.access_policy_create_ztna = access_policy_create_ztna  # type: ignore[attr-defined]


def micro_segment_create(self, segment_name, resources, allowed_identities, tenant_id="default"):
	"""ZTNA capability: micro segment create."""
	seg={"id":str(__import__("uuid").uuid4()),"name":segment_name,"resources":resources,"allowed_identities":allowed_identities,"tenant_id":tenant_id}
	self._store.setdefault(f"ztna_segments:{tenant_id}",{})[seg["id"]]=seg
	return seg

ZtnaService.micro_segment_create = micro_segment_create  # type: ignore[attr-defined]


def access_policy_list(self, tenant_id="default"):
	"""ZTNA capability: access policy list."""
	return list(self._store.get(f"ztna_policies:{tenant_id}",{}).values())

ZtnaService.access_policy_list = access_policy_list  # type: ignore[attr-defined]


def access_policy_delete(self, policy_id, tenant_id="default"):
	"""ZTNA capability: access policy delete."""
	return self._store.get(f"ztna_policies:{tenant_id}",{}).pop(policy_id,None) is not None

ZtnaService.access_policy_delete = access_policy_delete  # type: ignore[attr-defined]


def session_list_all(self, tenant_id="default", active_only=True):
	"""ZTNA capability: session list all."""
	sessions=list(self._store.get(f"ztna_sessions:{tenant_id}",{}).values())
	return [s for s in sessions if s.get("status")=="active"] if active_only else sessions

ZtnaService.session_list_all = session_list_all  # type: ignore[attr-defined]


def trust_score_history_ztna(self, identity_id, tenant_id="default", limit=20):
	"""ZTNA capability: trust score history ztna."""
	events=self._store.get(f"ztna_trust_events:{tenant_id}",[])
	return [e for e in events if e.get("identity_id")==identity_id][-limit:]

ZtnaService.trust_score_history_ztna = trust_score_history_ztna  # type: ignore[attr-defined]


def lateral_movement_detect(self, network_events, tenant_id="default"):
	"""ZTNA capability: lateral movement detect."""
	by_id={}
	for ev in network_events:
		by_id.setdefault(ev.get("identity_id",""),set()).add(ev.get("resource_id",""))
	return [{"identity_id":iid,"resources":list(rs),"alert":"lateral_movement"} for iid,rs in by_id.items() if len(rs)>5]

ZtnaService.lateral_movement_detect = lateral_movement_detect  # type: ignore[attr-defined]


def ztna_compliance_report_gen(self, period="30d", tenant_id="default"):
	"""ZTNA capability: ztna compliance report gen."""
	sessions=list(self._store.get(f"ztna_sessions:{tenant_id}",{}).values())
	policies=list(self._store.get(f"ztna_policies:{tenant_id}",{}).values())
	return {"period":period,"tenant_id":tenant_id,"session_count":len(sessions),"active_policies":len([p for p in policies if p.get("status")=="active"]),"ok":True}

ZtnaService.ztna_compliance_report_gen = ztna_compliance_report_gen  # type: ignore[attr-defined]


def ztna_posture_summary(self, tenant_id="default"):
	"""ZTNA capability: ztna posture summary."""
	devices=list(self._store.get(f"ztna_devices:{tenant_id}",{}).values())
	compliant=[d for d in devices if d.get("compliant")]
	return {"total_devices":len(devices),"compliant_devices":len(compliant),"compliance_rate":len(compliant)/len(devices) if devices else 1.0,"tenant_id":tenant_id}

ZtnaService.ztna_posture_summary = ztna_posture_summary  # type: ignore[attr-defined]


def ztna_analytics_report(self, period="30d", tenant_id="default"):
	"""ZTNA capability: ztna analytics report."""
	sessions=list(self._store.get(f"ztna_sessions:{tenant_id}",{}).values())
	return {"period":period,"session_count":len(sessions),"unique_identities":len({s.get("identity_id") for s in sessions}),"unique_resources":len({s.get("resource_id") for s in sessions}),"tenant_id":tenant_id}

ZtnaService.ztna_analytics_report = ztna_analytics_report  # type: ignore[attr-defined]


def identity_risk_profile_ztna(self, identity_id, tenant_id="default"):
	"""ZTNA capability: identity risk profile ztna."""
	sessions=[s for s in self._store.get(f"ztna_sessions:{tenant_id}",{}).values() if s.get("identity_id")==identity_id]
	denied=[s for s in sessions if s.get("outcome")=="denied"]
	return {"identity_id":identity_id,"total_sessions":len(sessions),"denied":len(denied),"risk_level":"high" if len(denied)>5 else "medium" if len(denied)>2 else "low","tenant_id":tenant_id}

ZtnaService.identity_risk_profile_ztna = identity_risk_profile_ztna  # type: ignore[attr-defined]


def resource_access_audit_ztna(self, resource_id, tenant_id="default"):
	"""ZTNA capability: resource access audit ztna."""
	sessions=[s for s in self._store.get(f"ztna_sessions:{tenant_id}",{}).values() if s.get("resource_id")==resource_id]
	return {"resource_id":resource_id,"total":len(sessions),"allowed":len([s for s in sessions if s.get("outcome")=="allowed"]),"denied":len([s for s in sessions if s.get("outcome")=="denied"]),"tenant_id":tenant_id}

ZtnaService.resource_access_audit_ztna = resource_access_audit_ztna  # type: ignore[attr-defined]


def bulk_device_register_ztna(self, devices_list, tenant_id="default"):
	"""ZTNA capability: bulk device register ztna."""
	registered=[]
	for d in devices_list:
		try:
			result=self.register_device(d.get("device_id",str(__import__("uuid").uuid4())),d.get("name",""),d.get("platform","unknown"),d.get("identity_id",""),tenant_id)
			registered.append(result.get("id",""))
		except Exception as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	return {"registered":registered,"count":len(registered)}

ZtnaService.bulk_device_register_ztna = bulk_device_register_ztna  # type: ignore[attr-defined]


def segment_list_all(self, tenant_id="default"):
	"""ZTNA capability: segment list all."""
	return list(self._store.get(f"ztna_segments:{tenant_id}",{}).values())

ZtnaService.segment_list_all = segment_list_all  # type: ignore[attr-defined]
