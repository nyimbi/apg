"""Consent and privacy management service for the APG CONS capability."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
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


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


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
		self._dpa_register: dict[str, dict[str, Any]] = {}
		self._breach_records: dict[str, dict[str, Any]] = {}
		self._pia_records: dict[str, dict[str, Any]] = {}
		self._portability_exports: dict[str, dict[str, Any]] = {}
		self._erasure_records: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ existing

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

	# ------------------------------------------------------------------ new methods

	def record_consent(
		self,
		tenant_id: str,
		consent_id: str,
		subject_id: str,
		purpose: str,
		legal_basis: str,
		expiry_date: str | None,
		channel: str,
		captured_by: str = "system",
	) -> dict[str, Any]:
		"""High-level consent capture from any channel with inline purpose/notice bootstrap."""
		# auto-bootstrap notice and purpose if not present
		notice_id = f"notice:{purpose}:default"
		purpose_id = f"purpose:{purpose}:default"
		if self._key(tenant_id, notice_id) not in self._notices:
			self.publish_notice(
				notice_id=notice_id,
				tenant_id=tenant_id,
				version="v1",
				url=f"https://privacy.example.com/{purpose}",
				language="en",
				purposes=[purpose],
				published_by="system",
			)
		if self._key(tenant_id, purpose_id) not in self._purposes:
			self.create_purpose(
				purpose_id=purpose_id,
				tenant_id=tenant_id,
				name=purpose,
				owner="system",
				legal_basis=legal_basis,
				retention_policy="standard",
				notice_id=notice_id,
				data_categories=["general"],
			)
		result = self.capture_consent(
			consent_id=consent_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			purpose_id=purpose_id,
			notice_id=notice_id,
			source=channel,
			captured_by=captured_by,
		)
		if expiry_date:
			result["expiry_date"] = expiry_date
		return result

	def revoke_consent(
		self,
		tenant_id: str,
		consent_id: str,
		reason: str,
		revoked_by: str = "subject",
	) -> dict[str, Any]:
		"""Revoke a consent record with a documented reason."""
		consent = self._require_consent(consent_id, tenant_id)
		assert bool(reason), "revocation reason required"
		consent.status = "withdrawn"
		consent.withdrawn_at = utc_now()
		revocation = consent.to_dict() | {"revocation_reason": reason, "revoked_by": revoked_by}
		self._record_audit(tenant_id, "consent_revoked", consent_id, revoked_by, revocation)
		return revocation

	def preference_update(
		self,
		tenant_id: str,
		subject_id: str,
		preferences: dict[str, Any],
		updated_by: str = "subject",
	) -> dict[str, Any]:
		"""Update communication and purpose preferences for a data subject."""
		channels = {k: bool(v) for k, v in preferences.get("channels", {}).items()}
		purposes = {k: bool(v) for k, v in preferences.get("purposes", {}).items()}
		profile_id = f"pref:{subject_id}"
		return self.update_preferences(
			profile_id=profile_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			channels=channels,
			purposes=purposes,
			updated_by=updated_by,
		)

	def subject_access_request(
		self,
		tenant_id: str,
		request_id: str,
		subject_id: str,
		request_type: str = "access",
		identity_verified: bool = True,
		evidence_reference: str = "id_verified",
	) -> dict[str, Any]:
		"""Submit a GDPR/POPIA data subject access request."""
		return self.submit_privacy_request(
			request_id=request_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			request_type=request_type,
			submitted_by=subject_id,
			identity_verified=identity_verified,
			evidence_reference=evidence_reference,
		)

	def right_to_erasure(
		self,
		tenant_id: str,
		erasure_id: str,
		subject_id: str,
		scope: str,
		approved_by: str,
		evidence_ref: str = "",
	) -> dict[str, Any]:
		"""Process a right-to-erasure (GDPR Art. 17) request, marking all subject data for deletion."""
		assert bool(approved_by), "approver required for erasure"
		assert scope in {"all", "marketing", "analytics", "profile"}, f"invalid scope: {scope}"
		# withdraw all active consents for this subject
		withdrawn_consents: list[str] = []
		for consent in list(self._consents.values()):
			if consent.tenant_id == tenant_id and consent.subject_id == subject_id and consent.status == "active":
				consent.status = "withdrawn"
				consent.withdrawn_at = utc_now()
				withdrawn_consents.append(consent.id)
		# mark preferences deleted
		deleted_profiles: list[str] = []
		for key, profile in list(self._preferences.items()):
			if profile.tenant_id == tenant_id and profile.subject_id == subject_id:
				del self._preferences[key]
				deleted_profiles.append(profile.id)
		record = {
			"id": erasure_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"scope": scope,
			"approved_by": approved_by,
			"evidence_ref": evidence_ref,
			"withdrawn_consents": withdrawn_consents,
			"deleted_profiles": deleted_profiles,
			"status": "completed",
			"executed_at": _utc_now_iso(),
		}
		self._erasure_records[self._key(tenant_id, erasure_id)] = record
		self._record_audit(tenant_id, "right_to_erasure_executed", erasure_id, approved_by, record)
		return record

	def data_portability_export(
		self,
		tenant_id: str,
		export_id: str,
		subject_id: str,
		format: str = "json",
		requested_by: str = "subject",
	) -> dict[str, Any]:
		"""Package all subject data for portability export (GDPR Art. 20)."""
		assert format in {"json", "csv", "xml"}, f"unsupported format: {format}"
		consents = [c.to_dict() for c in self._consents.values() if c.tenant_id == tenant_id and c.subject_id == subject_id]
		preferences = [p.to_dict() for p in self._preferences.values() if p.tenant_id == tenant_id and p.subject_id == subject_id]
		requests = [r.to_dict() for r in self._requests.values() if r.tenant_id == tenant_id and r.subject_id == subject_id]
		payload: dict[str, Any] = {
			"subject_id": subject_id,
			"consents": consents,
			"preferences": preferences,
			"privacy_requests": requests,
		}
		if format == "json":
			content = json.dumps(payload, default=str, indent=2)
		else:
			# CSV/XML: simplified single-line summary
			content = f"{format.upper()}:subject_id={subject_id},consent_count={len(consents)},pref_count={len(preferences)}"
		checksum = hashlib.sha256(content.encode()).hexdigest()
		export = {
			"id": export_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"format": format,
			"requested_by": requested_by,
			"record_count": len(consents) + len(preferences) + len(requests),
			"checksum_sha256": checksum,
			"content_preview": content[:200],
			"status": "ready",
			"generated_at": _utc_now_iso(),
		}
		self._portability_exports[self._key(tenant_id, export_id)] = export
		self._record_audit(tenant_id, "data_portability_exported", export_id, requested_by, {"subject_id": subject_id, "format": format})
		return export

	def privacy_impact_assessment(
		self,
		tenant_id: str,
		pia_id: str,
		processing_activity: str,
		risks: list[dict[str, Any]],
		mitigations: list[dict[str, Any]],
		assessed_by: str,
		dpo_reviewed: bool = False,
	) -> dict[str, Any]:
		"""Record a Privacy Impact Assessment (DPIA) for a high-risk processing activity."""
		assert bool(processing_activity), "processing_activity required"
		assert bool(assessed_by), "assessor required"
		residual_risk_scores = [float(r.get("residual_score", r.get("score", 5))) for r in risks]
		avg_residual = sum(residual_risk_scores) / len(residual_risk_scores) if residual_risk_scores else 0.0
		pia = {
			"id": pia_id,
			"tenant_id": tenant_id,
			"processing_activity": processing_activity,
			"risks": risks,
			"mitigations": mitigations,
			"risk_count": len(risks),
			"mitigation_count": len(mitigations),
			"avg_residual_risk_score": round(avg_residual, 2),
			"risk_level": "high" if avg_residual >= 7 else "medium" if avg_residual >= 4 else "low",
			"assessed_by": assessed_by,
			"dpo_reviewed": dpo_reviewed,
			"status": "approved" if dpo_reviewed else "pending_dpo_review",
			"assessed_at": _utc_now_iso(),
		}
		self._pia_records[self._key(tenant_id, pia_id)] = pia
		self._record_audit(tenant_id, "pia_recorded", pia_id, assessed_by, {"processing_activity": processing_activity, "risk_level": pia["risk_level"]})
		return pia

	def dpa_register(
		self,
		tenant_id: str,
		entry_id: str,
		processing_activity: str,
		purpose: str,
		categories: list[str],
		retention: str,
		legal_basis: str = "legitimate_interest",
		controller: str = "organization",
	) -> dict[str, Any]:
		"""Add an entry to the Records of Processing Activities (ROPA) under GDPR Art. 30."""
		assert bool(processing_activity), "processing_activity required"
		assert bool(categories), "data categories required"
		assert bool(retention), "retention period required"
		entry = {
			"id": entry_id,
			"tenant_id": tenant_id,
			"processing_activity": processing_activity,
			"purpose": purpose,
			"categories": list(categories),
			"retention": retention,
			"legal_basis": legal_basis,
			"controller": controller,
			"status": "active",
			"registered_at": _utc_now_iso(),
		}
		self._dpa_register[self._key(tenant_id, entry_id)] = entry
		self._record_audit(tenant_id, "dpa_entry_registered", entry_id, controller, entry)
		return entry

	def breach_notification(
		self,
		tenant_id: str,
		breach_id: str,
		description: str,
		dpa_notified: bool,
		subjects_notified: bool,
		severity: str = "high",
		affected_subjects: int = 0,
		notified_by: str = "dpo",
	) -> dict[str, Any]:
		"""Record a personal data breach notification under GDPR Art. 33/34."""
		assert bool(description), "breach description required"
		assert severity in {"low", "medium", "high", "critical"}, f"invalid severity: {severity}"
		record = {
			"id": breach_id,
			"tenant_id": tenant_id,
			"description": description,
			"severity": severity,
			"affected_subjects": affected_subjects,
			"dpa_notified": dpa_notified,
			"subjects_notified": subjects_notified,
			"notified_by": notified_by,
			"notification_status": "complete" if (dpa_notified and (subjects_notified or affected_subjects == 0)) else "partial",
			"reported_at": _utc_now_iso(),
		}
		self._breach_records[self._key(tenant_id, breach_id)] = record
		self._record_audit(tenant_id, "breach_notification_recorded", breach_id, notified_by, record)
		return record

	def consent_analytics(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Aggregate consent metrics for a tenant over a reporting period."""
		consents = [c for c in self._consents.values() if c.tenant_id == tenant_id]
		active = [c for c in consents if c.status == "active"]
		withdrawn = [c for c in consents if c.status == "withdrawn"]
		purposes = [p for p in self._purposes.values() if p.tenant_id == tenant_id]
		requests = [r for r in self._requests.values() if r.tenant_id == tenant_id]
		open_requests = [r for r in requests if r.status == "open"]
		overdue = [r for r in open_requests if request_sla_state(r.due_at) == "overdue"]
		# consent rate per purpose
		purpose_consent_rates: dict[str, dict[str, int]] = {}
		for c in consents:
			bucket = purpose_consent_rates.setdefault(c.purpose_id, {"active": 0, "withdrawn": 0})
			if c.status == "active":
				bucket["active"] += 1
			else:
				bucket["withdrawn"] += 1
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_consents": len(consents),
			"active_consents": len(active),
			"withdrawn_consents": len(withdrawn),
			"consent_rate_pct": round(len(active) / max(len(consents), 1) * 100, 2),
			"purpose_count": len(purposes),
			"open_requests": len(open_requests),
			"overdue_requests": len(overdue),
			"breach_count": len([b for b in self._breach_records.values() if b["tenant_id"] == tenant_id]),
			"pia_count": len([p for p in self._pia_records.values() if p["tenant_id"] == tenant_id]),
			"dpa_entry_count": len([e for e in self._dpa_register.values() if e["tenant_id"] == tenant_id]),
			"coverage": consent_coverage(len(active), len(withdrawn), len(purposes)),
			"purpose_consent_rates": purpose_consent_rates,
			"computed_at": _utc_now_iso(),
		}

	# ------------------------------------------------------------------ list / compat

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
			"breach_count": len([b for b in self._breach_records.values() if b["tenant_id"] == tenant_id]),
			"pia_count": len([p for p in self._pia_records.values() if p["tenant_id"] == tenant_id]),
			"dpa_entry_count": len([e for e in self._dpa_register.values() if e["tenant_id"] == tenant_id]),
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

	def dsar_process(
		self,
		tenant_id: str,
		request_id: str,
		subject_id: str,
		request_type: str = "access",
		identity_verified: bool = True,
		evidence_reference: str = "id_verified",
	) -> dict[str, Any]:
		"""Process a Data Subject Access Request (DSAR) — alias for subject_access_request."""
		return self.subject_access_request(
			tenant_id=tenant_id,
			request_id=request_id,
			subject_id=subject_id,
			request_type=request_type,
			identity_verified=identity_verified,
			evidence_reference=evidence_reference,
		)

	def erasure_execute(
		self,
		tenant_id: str,
		erasure_id: str,
		subject_id: str,
		scope: str,
		approved_by: str,
		evidence_ref: str = "",
	) -> dict[str, Any]:
		"""Execute right-to-erasure (GDPR Art 17) — alias for right_to_erasure."""
		return self.right_to_erasure(
			tenant_id=tenant_id,
			erasure_id=erasure_id,
			subject_id=subject_id,
			scope=scope,
			approved_by=approved_by,
			evidence_ref=evidence_ref,
		)

	def portability_export(
		self,
		tenant_id: str,
		export_id: str,
		subject_id: str,
		format: str = "json",
		requested_by: str = "subject",
	) -> dict[str, Any]:
		"""Export portable data package for a subject (GDPR Art 20)."""
		return self.data_portability_export(
			tenant_id=tenant_id,
			export_id=export_id,
			subject_id=subject_id,
			format=format,
			requested_by=requested_by,
		)

	def cookie_manage(
		self,
		tenant_id: str,
		cookie_policy_id: str,
		categories: dict[str, bool],
		subject_id: str,
		channel: str = "web",
		captured_by: str = "system",
	) -> dict[str, Any]:
		"""Record cookie consent choices per category (necessary, analytics, marketing, etc.)."""
		self._require_tenant(tenant_id)
		consent_records = []
		for category, accepted in categories.items():
			consent_id = f"{cookie_policy_id}:{subject_id}:{category}"
			if accepted:
				try:
					rec = self.record_consent(
						tenant_id=tenant_id,
						consent_id=consent_id,
						subject_id=subject_id,
						purpose=f"cookie_{category}",
						legal_basis="consent",
						expiry_date=None,
						channel=channel,
						captured_by=captured_by,
					)
				except ValueError:
					rec = {"id": consent_id, "status": "already_exists"}
			else:
				rec = {"id": consent_id, "category": category, "accepted": False, "status": "declined"}
			consent_records.append(rec)
		return {
			"cookie_policy_id": cookie_policy_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"categories": categories,
			"consent_records": consent_records,
			"recorded_at": _utc_now_iso(),
		}

	def legitimate_interest_balance(
		self,
		tenant_id: str,
		assessment_id: str,
		processing_activity: str,
		legitimate_interest: str,
		necessity_justification: str,
		balancing_test: str,
		assessed_by: str,
		dpo_reviewed: bool = False,
	) -> dict[str, Any]:
		"""Record a Legitimate Interest Assessment (LIA) under GDPR Art 6(1)(f)."""
		self._require_tenant(tenant_id)
		record = {
			"id": assessment_id,
			"tenant_id": tenant_id,
			"processing_activity": processing_activity,
			"legitimate_interest": legitimate_interest,
			"necessity_justification": necessity_justification,
			"balancing_test": balancing_test,
			"assessed_by": assessed_by,
			"dpo_reviewed": dpo_reviewed,
			"outcome": "approved" if dpo_reviewed else "pending_dpo_review",
			"assessed_at": _utc_now_iso(),
		}
		self._pia_records[self._key(tenant_id, assessment_id)] = record
		self._record_audit(tenant_id, "lia_recorded", assessment_id, assessed_by, record)
		return record

	def consent_analytics(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Aggregate consent metrics for a tenant over a reporting period."""
		consents = [c for c in self._consents.values() if c.tenant_id == tenant_id]
		active = [c for c in consents if c.status == "active"]
		withdrawn = [c for c in consents if c.status == "withdrawn"]
		purposes = [p for p in self._purposes.values() if p.tenant_id == tenant_id]
		requests = [r for r in self._requests.values() if r.tenant_id == tenant_id]
		open_requests = [r for r in requests if r.status == "open"]
		overdue = [r for r in open_requests if request_sla_state(r.due_at) == "overdue"]
		purpose_consent_rates: dict[str, dict[str, int]] = {}
		for c in consents:
			bucket = purpose_consent_rates.setdefault(c.purpose_id, {"active": 0, "withdrawn": 0})
			if c.status == "active":
				bucket["active"] += 1
			else:
				bucket["withdrawn"] += 1
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_consents": len(consents),
			"active_consents": len(active),
			"withdrawn_consents": len(withdrawn),
			"consent_rate_pct": round(len(active) / max(len(consents), 1) * 100, 2),
			"purpose_count": len(purposes),
			"open_requests": len(open_requests),
			"overdue_requests": len(overdue),
			"breach_count": len([b for b in self._breach_records.values() if b["tenant_id"] == tenant_id]),
			"pia_count": len([p for p in self._pia_records.values() if p["tenant_id"] == tenant_id]),
			"dpa_entry_count": len([e for e in self._dpa_register.values() if e["tenant_id"] == tenant_id]),
			"coverage": consent_coverage(len(active), len(withdrawn), len(purposes)),
			"purpose_consent_rates": purpose_consent_rates,
			"computed_at": _utc_now_iso(),
		}

	def privacy_notice_version(
		self,
		tenant_id: str,
		notice_id: str,
		new_version: str,
		url: str,
		language: str,
		purposes: list[str],
		published_by: str,
	) -> dict[str, Any]:
		"""Publish a new version of a privacy notice, superseding the previous one."""
		self._require_tenant(tenant_id)
		versioned_id = f"{notice_id}:{new_version}"
		return self.publish_notice(
			notice_id=versioned_id,
			tenant_id=tenant_id,
			version=new_version,
			url=url,
			language=language,
			purposes=purposes,
			published_by=published_by,
		) | {"supersedes": notice_id}

	def third_party_disclose(
		self,
		tenant_id: str,
		disclosure_id: str,
		third_party_name: str,
		purpose: str,
		data_categories: list[str],
		legal_basis: str,
		dpa_ref: str,
		disclosed_by: str,
	) -> dict[str, Any]:
		"""Record a third-party data disclosure in the ROPA register."""
		return self.dpa_register(
			tenant_id=tenant_id,
			entry_id=disclosure_id,
			processing_activity=f"third_party_disclosure:{third_party_name}",
			purpose=purpose,
			categories=data_categories,
			retention="as_per_contract",
			legal_basis=legal_basis,
			controller=disclosed_by,
		) | {"third_party_name": third_party_name, "dpa_ref": dpa_ref}

	def children_consent(
		self,
		tenant_id: str,
		consent_id: str,
		child_subject_id: str,
		guardian_subject_id: str,
		purpose: str,
		age_verified: bool,
		channel: str = "web",
		captured_by: str = "system",
	) -> dict[str, Any]:
		"""Capture parental/guardian consent for a child data subject."""
		assert age_verified, "age_verified must be True for children consent"
		result = self.record_consent(
			tenant_id=tenant_id,
			consent_id=consent_id,
			subject_id=child_subject_id,
			purpose=purpose,
			legal_basis="parental_consent",
			expiry_date=None,
			channel=channel,
			captured_by=captured_by,
		)
		result["guardian_subject_id"] = guardian_subject_id
		result["consent_type"] = "children_consent"
		result["age_verified"] = age_verified
		return result

	def breach_notify(
		self,
		tenant_id: str,
		breach_id: str,
		description: str,
		dpa_notified: bool,
		subjects_notified: bool,
		severity: str = "high",
		affected_subjects: int = 0,
		notified_by: str = "dpo",
	) -> dict[str, Any]:
		"""Record a personal data breach notification (alias for breach_notification)."""
		return self.breach_notification(
			tenant_id=tenant_id,
			breach_id=breach_id,
			description=description,
			dpa_notified=dpa_notified,
			subjects_notified=subjects_notified,
			severity=severity,
			affected_subjects=affected_subjects,
			notified_by=notified_by,
		)

	def list_breaches(self, tenant_id: str) -> list[dict[str, Any]]:
		return [b for b in self._breach_records.values() if b["tenant_id"] == tenant_id]

	def list_pia_records(self, tenant_id: str) -> list[dict[str, Any]]:
		return [p for p in self._pia_records.values() if p["tenant_id"] == tenant_id]

	def list_dpa_entries(self, tenant_id: str) -> list[dict[str, Any]]:
		return [e for e in self._dpa_register.values() if e["tenant_id"] == tenant_id]

	# ------------------------------------------------------------------ internals

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


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()
