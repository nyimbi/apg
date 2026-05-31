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
