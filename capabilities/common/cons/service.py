"""Consent and privacy management service for the APG CONS capability."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .capability_contract import (
	SUPPORTED_PRIVACY_AGENT_ROLES,
	SUPPORTED_PRIVACY_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	ConsentEvent,
	PreferenceProfile,
	PrivacyAuditEvent,
	PrivacyAgent,
	PrivacyNotice,
	PrivacyPurpose,
	PrivacyRequest,
	ProcessingDecision,
	utc_now,
)
from .privacy_engine import consent_age_days, consent_coverage, request_due_at, request_sla_state, stable_digest


class ConsService:
	"""Tenant-scoped purpose registry, consent ledger, preference, and request service."""

	def __init__(self) -> None:
		self._purposes: dict[str, PrivacyPurpose] = {}
		self._notices: dict[str, PrivacyNotice] = {}
		self._consents: dict[str, ConsentEvent] = {}
		self._preferences: dict[str, PreferenceProfile] = {}
		self._requests: dict[str, PrivacyRequest] = {}
		self._processing_decisions: dict[str, ProcessingDecision] = {}
		self._agents: dict[str, PrivacyAgent] = {}
		self._audit_events: list[PrivacyAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def publish_notice(
		self,
		notice_id: str,
		tenant_id: str,
		version: str,
		url: str,
		language: str,
		purposes: list[str],
		published_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, notice_id)
		if key in self._notices:
			raise ValueError("notice_already_exists")
		notice = PrivacyNotice(
			id=notice_id,
			tenant_id=tenant_id,
			version=version,
			url=url,
			language=language,
			purposes=list(purposes),
			published_by=published_by,
		)
		self._notices[key] = notice
		self._record_audit(tenant_id, "notice_published", notice_id, published_by, notice.to_dict())
		return notice.to_dict()

	def create_purpose(
		self,
		purpose_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		legal_basis: str,
		retention_policy: str,
		notice_id: str,
		data_categories: list[str],
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_purpose",
			"legal_basis_present": bool(legal_basis),
			"purpose_owner_assigned": bool(owner),
			"retention_policy_present": bool(retention_policy),
			"notice_link_present": bool(notice_id),
		})
		self._raise_if_denied(result)
		self._require_notice(notice_id, tenant_id)
		key = self._key(tenant_id, purpose_id)
		if key in self._purposes:
			raise ValueError("purpose_already_exists")
		purpose = PrivacyPurpose(
			id=purpose_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			legal_basis=legal_basis,
			retention_policy=retention_policy,
			notice_id=notice_id,
			data_categories=list(data_categories),
		)
		self._purposes[key] = purpose
		self._record_audit(tenant_id, "purpose_created", purpose_id, owner, purpose.to_dict())
		return purpose.to_dict()

	def capture_consent(
		self,
		consent_id: str,
		tenant_id: str,
		subject_id: str,
		purpose_id: str,
		notice_id: str,
		source: str,
		captured_by: str,
		captured_at: datetime | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_purpose(purpose_id, tenant_id)
		self._require_notice(notice_id, tenant_id)
		key = self._key(tenant_id, consent_id)
		if key in self._consents:
			raise ValueError("consent_already_exists")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "capture_consent",
			"notice_present": bool(notice_id),
		})
		self._raise_if_denied(result)
		payload = {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"purpose_id": purpose_id,
			"notice_id": notice_id,
			"source": source,
			"captured_by": captured_by,
		}
		consent = ConsentEvent(
			id=consent_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			purpose_id=purpose_id,
			notice_id=notice_id,
			source=source,
			captured_by=captured_by,
			captured_at=captured_at or utc_now(),
			provenance_hash=stable_digest(payload),
		)
		self._consents[key] = consent
		self._record_audit(tenant_id, "consent_captured", consent_id, captured_by, consent.to_dict())
		return consent.to_dict()

	def withdraw_consent(self, consent_id: str, tenant_id: str, actor: str) -> dict[str, Any]:
		consent = self._require_consent(consent_id, tenant_id)
		consent.status = "withdrawn"
		consent.withdrawn_at = utc_now()
		self._record_audit(tenant_id, "consent_withdrawn", consent_id, actor, consent.to_dict())
		return consent.to_dict()

	def update_preferences(
		self,
		profile_id: str,
		tenant_id: str,
		subject_id: str,
		channels: dict[str, bool],
		purposes: dict[str, bool],
		updated_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, profile_id)
		profile = PreferenceProfile(
			id=profile_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			channels=dict(channels),
			purposes=dict(purposes),
			updated_by=updated_by,
		)
		self._preferences[key] = profile
		self._record_audit(tenant_id, "preferences_updated", profile_id, updated_by, profile.to_dict())
		return profile.to_dict()

	def process_consent_gated_data(
		self,
		decision_id: str,
		tenant_id: str,
		subject_id: str,
		purpose_id: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_purpose(purpose_id, tenant_id)
		active = self._active_consent(tenant_id, subject_id, purpose_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "process_consent_gated_data",
			"active_consent_present": active is not None,
		})
		if result["decision"] == "deny":
			decision = ProcessingDecision(
				id=decision_id,
				tenant_id=tenant_id,
				subject_id=subject_id,
				purpose_id=purpose_id,
				decision="deny",
				reason=", ".join(action.get("reason", "consent_required") for action in result["actions"]),
				consent_id=None,
			)
			self._processing_decisions[self._key(tenant_id, decision_id)] = decision
			self._record_audit(tenant_id, "processing_denied", decision_id, subject_id, decision.to_dict())
			raise PermissionError(decision.reason)
		decision = ProcessingDecision(
			id=decision_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			purpose_id=purpose_id,
			decision="allow",
			reason="active_consent_present",
			consent_id=active.id if active else None,
		)
		self._processing_decisions[self._key(tenant_id, decision_id)] = decision
		self._record_audit(tenant_id, "processing_allowed", decision_id, subject_id, decision.to_dict())
		return decision.to_dict()

	def submit_privacy_request(
		self,
		request_id: str,
		tenant_id: str,
		subject_id: str,
		request_type: str,
		submitted_by: str,
		identity_verified: bool,
		evidence_reference: str,
		submitted_at: datetime | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "process_privacy_request",
			"identity_verified": identity_verified,
			"request_evidence_present": bool(evidence_reference),
		})
		self._raise_if_denied(result)
		key = self._key(tenant_id, request_id)
		if key in self._requests:
			raise ValueError("privacy_request_already_exists")
		submitted = submitted_at or utc_now()
		request = PrivacyRequest(
			id=request_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			request_type=request_type,
			submitted_by=submitted_by,
			identity_verified=identity_verified,
			evidence_reference=evidence_reference,
			submitted_at=submitted,
			due_at=request_due_at(submitted),
		)
		self._requests[key] = request
		self._record_audit(tenant_id, "privacy_request_submitted", request_id, submitted_by, request.to_dict())
		return request.to_dict()

	def complete_privacy_request(self, request_id: str, tenant_id: str, actor: str, resolution: str) -> dict[str, Any]:
		request = self._require_request(request_id, tenant_id)
		request.status = "completed"
		request.completed_at = utc_now()
		request.resolution = resolution
		self._record_audit(tenant_id, "privacy_request_completed", request_id, actor, request.to_dict())
		return request.to_dict()

	def register_privacy_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool,
		policy_ref: str = "",
		registered: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_privacy_agent_runtime(runtime)
		normalized_role = _normalize_privacy_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": True,
			"privacy_agent_present": True,
			"agent_registered": bool(registered),
			"agent_runtime_supported": bool(normalized_runtime),
			"agent_role_supported": bool(normalized_role),
			"agent_scope_present": bool(scope.strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		self._raise_if_denied(result)
		key = self._key(tenant_id, agent_id)
		if key in self._agents:
			raise ValueError("privacy_agent_already_registered")
		agent = PrivacyAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name or agent_id,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref or None,
		)
		self._agents[key] = agent
		self._record_audit(tenant_id, "privacy_agent_registered", agent_id, agent.name, agent.to_dict())
		return agent.to_dict()

	def change_purpose_state(
		self,
		tenant_id: str,
		purpose_id: str,
		active: bool,
		reason: str,
		audit_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		purpose = self._require_purpose(purpose_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"state_change_requested": True,
			"state_change_reason_present": bool(reason.strip()),
			"audit_event_recorded": bool(audit_recorded),
		})
		self._raise_if_denied(result)
		purpose.active = bool(active)
		self._record_audit(tenant_id, "purpose_state_changed", purpose_id, purpose.owner, purpose.to_dict() | {"reason": reason})
		return purpose.to_dict()

	def review_stale_consents(self, tenant_id: str, now: datetime | None = None) -> list[dict[str, Any]]:
		self._require_tenant(tenant_id)
		review_required: list[dict[str, Any]] = []
		for consent in self._consents.values():
			if consent.tenant_id != tenant_id or consent.status != "active":
				continue
			age = consent_age_days(consent.captured_at, now)
			result = self.evaluate({
				"tenant_context_present": True,
				"consent_age_days": age,
				"stale_consent_reviewed": False,
			})
			if result["decision"] == "require_review":
				review_required.append(consent.to_dict())
				self._record_audit(tenant_id, "stale_consent_review_required", consent.id, consent.subject_id, consent.to_dict())
		return review_required

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		purposes = [purpose for purpose in self._purposes.values() if purpose.tenant_id == tenant_id]
		active_consents = [consent for consent in self._consents.values() if consent.tenant_id == tenant_id and consent.status == "active"]
		withdrawn = [consent for consent in self._consents.values() if consent.tenant_id == tenant_id and consent.status == "withdrawn"]
		requests = [request for request in self._requests.values() if request.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"purpose_count": len(purposes),
			"notice_count": len([notice for notice in self._notices.values() if notice.tenant_id == tenant_id]),
			"active_consent_count": len(active_consents),
			"withdrawn_consent_count": len(withdrawn),
			"preference_profile_count": len([profile for profile in self._preferences.values() if profile.tenant_id == tenant_id]),
			"open_request_count": len([request for request in requests if request.status == "open"]),
			"overdue_request_count": len([request for request in requests if request.status == "open" and request_sla_state(request.due_at) == "overdue"]),
			"processing_decision_count": len([decision for decision in self._processing_decisions.values() if decision.tenant_id == tenant_id]),
			"privacy_agent_count": len([agent for agent in self._agents.values() if agent.tenant_id == tenant_id]),
			"audit_event_count": len([event for event in self._audit_events if event.tenant_id == tenant_id]),
			"coverage": consent_coverage(len(active_consents), len(withdrawn), len(purposes)),
		}

	def list_purposes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._purposes.values(), tenant_id)

	def list_notices(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._notices.values(), tenant_id)

	def list_consents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._consents.values(), tenant_id)

	def list_preferences(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._preferences.values(), tenant_id)

	def list_requests(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._requests.values(), tenant_id)

	def list_processing_decisions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._processing_decisions.values(), tenant_id)

	def list_privacy_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._agents.values(), tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in sorted(events, key=lambda item: item.id)]

	def _active_consent(self, tenant_id: str, subject_id: str, purpose_id: str) -> ConsentEvent | None:
		for consent in self._consents.values():
			if consent.tenant_id == tenant_id and consent.subject_id == subject_id and consent.purpose_id == purpose_id and consent.status == "active":
				return consent
		return None

	def _record_audit(self, tenant_id: str, event_type: str, subject_id: str, actor: str, payload: dict[str, Any]) -> None:
		self._audit_events.append(PrivacyAuditEvent(
			id=f"audit-{len(self._audit_events) + 1:06d}",
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			payload_hash=stable_digest(payload),
		))

	def _require_tenant(self, tenant_id: str) -> None:
		self._raise_if_denied(self.evaluate({"tenant_context_present": bool(tenant_id)}))

	def _require_notice(self, notice_id: str, tenant_id: str) -> PrivacyNotice:
		notice = self._notices.get(self._key(tenant_id, notice_id))
		if notice is None:
			raise KeyError("notice_not_found")
		return notice

	def _require_purpose(self, purpose_id: str, tenant_id: str) -> PrivacyPurpose:
		purpose = self._purposes.get(self._key(tenant_id, purpose_id))
		if purpose is None:
			raise KeyError("purpose_not_found")
		return purpose

	def _require_consent(self, consent_id: str, tenant_id: str) -> ConsentEvent:
		consent = self._consents.get(self._key(tenant_id, consent_id))
		if consent is None:
			raise KeyError("consent_not_found")
		return consent

	def _require_request(self, request_id: str, tenant_id: str) -> PrivacyRequest:
		request = self._requests.get(self._key(tenant_id, request_id))
		if request is None:
			raise KeyError("privacy_request_not_found")
		return request

	def _tenant_sorted(self, values: Any, tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(values)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "privacy_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "privacy_policy_blocked")

	def _key(self, tenant_id: str, object_id: str) -> str:
		return f"{tenant_id}:{object_id}"


def _normalize_privacy_agent_runtime(runtime: str) -> str:
	value = (runtime or "").strip().lower()
	return value if value in SUPPORTED_PRIVACY_AGENT_RUNTIMES else ""


def _normalize_privacy_agent_role(role: str) -> str:
	value = (role or "").strip().lower()
	return value if value in SUPPORTED_PRIVACY_AGENT_ROLES else ""
