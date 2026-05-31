"""Dependency-light BIOP runtime for generated APG applications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


SUPPORTED_MODALITIES = {"face", "voice", "fingerprint", "iris", "palm", "behavioral", "document"}


@dataclass(frozen=True)
class BiometricConsent:
	"""Tenant-scoped biometric consent evidence."""

	id: str
	tenant_id: str
	subject_id: str
	purpose: str
	modalities: tuple[str, ...]
	jurisdictions: tuple[str, ...]
	granted_by: str
	evidence: str
	status: str = "active"
	revoked_by: str = ""
	revocation_reason: str = ""
	created_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		data = asdict(self)
		data["modalities"] = list(self.modalities)
		data["jurisdictions"] = list(self.jurisdictions)
		return data


@dataclass(frozen=True)
class BiometricTemplateRecord:
	"""Encrypted biometric template metadata."""

	id: str
	tenant_id: str
	subject_id: str
	modality: str
	template_hash: str
	encrypted: bool
	quality_score: float
	consent_id: str
	retention_policy: str
	status: str = "active"
	retired_by: str = ""
	retirement_reason: str = ""
	created_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(frozen=True)
class BiometricVerificationRecord:
	"""Biometric verification decision and review state."""

	id: str
	tenant_id: str
	subject_id: str
	template_id: str
	modality: str
	requested_by: str
	match_confidence: float
	liveness_score: float
	source_jurisdiction: str
	target_jurisdiction: str
	consent_id: str
	status: str
	decision: str
	reasons: tuple[str, ...]
	privacy_review_id: str = ""
	match_review_id: str = ""
	reviewer: str = ""
	reviewer_notes: str = ""
	created_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		data = asdict(self)
		data["reasons"] = list(self.reasons)
		return data


@dataclass(frozen=True)
class BiometricReviewApproval:
	"""Independent review evidence for privacy or match decisions."""

	id: str
	tenant_id: str
	verification_id: str
	review_type: str
	requested_by: str
	justification: str
	status: str = "pending"
	decision: str = ""
	reviewer: str = ""
	notes: str = ""
	created_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(frozen=True)
class BiometricAuditEvent:
	"""BIOP governance event."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...]
	metadata: dict[str, Any]
	created_at: str

	def to_dict(self) -> dict[str, Any]:
		data = asdict(self)
		data["reasons"] = list(self.reasons)
		return data


@dataclass(frozen=True)
class BiometricAgentRecord:
	"""Provider-neutral AI agent registered for biometric governance."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	status: str = "active"
	created_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		data = asdict(self)
		data["agent_id"] = self.id
		return data


@dataclass(frozen=True)
class BiopLifecycleBatchRecord:
	"""Bytewax lifecycle batch validation evidence for biometric changes."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: tuple[str, ...] = ()
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		data = asdict(self)
		data["batch_id"] = self.id
		data["matched_rules"] = list(self.matched_rules)
		return data


class BiopService:
	"""Consent, template, verification, review, and audit lifecycle facade."""

	def __init__(self) -> None:
		self.contract = get_capability_contract()
		self._consents: dict[tuple[str, str], BiometricConsent] = {}
		self._templates: dict[tuple[str, str], BiometricTemplateRecord] = {}
		self._verifications: dict[tuple[str, str], BiometricVerificationRecord] = {}
		self._reviews: dict[tuple[str, str], BiometricReviewApproval] = {}
		self._audit_events: dict[tuple[str, str], BiometricAuditEvent] = {}
		self._biometric_agents: dict[tuple[str, str], BiometricAgentRecord] = {}
		self._lifecycle_batches: dict[tuple[str, str], BiopLifecycleBatchRecord] = {}
		self._agent_runtimes = set(self.contract["agents"]["supported_runtimes"])
		self._agent_roles = set(self.contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(self.contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(self.contract["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_consent(
		self,
		consent_id: str,
		tenant_id: str,
		subject_id: str,
		purpose: str,
		modalities: list[str] | tuple[str, ...],
		jurisdictions: list[str] | tuple[str, ...],
		granted_by: str,
		evidence: str,
	) -> dict[str, Any]:
		self._ensure_new(self._consents, tenant_id, consent_id, "consent")
		if not subject_id:
			raise ValueError("biometric_subject_required")
		if not purpose:
			raise ValueError("biometric_consent_purpose_required")
		if not granted_by:
			raise ValueError("biometric_consent_grantor_required")
		if not evidence:
			raise ValueError("biometric_consent_evidence_required")
		normalized_modalities = self._normalize_modalities(modalities)
		normalized_jurisdictions = self._normalize_jurisdictions(jurisdictions)
		consent = BiometricConsent(
			id=consent_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			purpose=purpose,
			modalities=normalized_modalities,
			jurisdictions=normalized_jurisdictions,
			granted_by=granted_by,
			evidence=evidence,
			created_at=self._now(),
		)
		self._consents[self._tenant_key(tenant_id, consent_id)] = consent
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type="consent_recorded",
			actor=granted_by,
			decision="allow",
			metadata={"consent_id": consent_id, "modalities": list(normalized_modalities)},
		)
		return consent.to_dict()

	def revoke_consent(self, consent_id: str, tenant_id: str, revoked_by: str, reason: str) -> dict[str, Any]:
		consent = self._require_consent(consent_id, tenant_id)
		if not revoked_by:
			raise ValueError("biometric_consent_revoker_required")
		if not reason:
			raise ValueError("biometric_consent_revocation_reason_required")
		revoked = replace(consent, status="revoked", revoked_by=revoked_by, revocation_reason=reason)
		self._consents[self._tenant_key(tenant_id, consent_id)] = revoked
		for key, template in list(self._templates.items()):
			if template.tenant_id == tenant_id and template.consent_id == consent_id and template.status == "active":
				self._templates[key] = replace(template, status="retired", retired_by=revoked_by, retirement_reason="consent_revoked")
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=consent.subject_id,
			event_type="consent_revoked",
			actor=revoked_by,
			decision="deny",
			reasons=("consent_revoked",),
			metadata={"consent_id": consent_id, "reason": reason},
		)
		return revoked.to_dict()

	def enroll_template(
		self,
		template_id: str,
		tenant_id: str,
		subject_id: str,
		modality: str,
		template_hash: str,
		encrypted: bool,
		quality_score: float,
		consent_id: str,
		retention_policy: str,
	) -> dict[str, Any]:
		self._ensure_new(self._templates, tenant_id, template_id, "template")
		normalized_modality = self._normalize_modality(modality)
		consent = self._active_consent(consent_id, tenant_id, subject_id, normalized_modality, "")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "store_template",
			"template_encrypted": bool(encrypted),
			"active_consent_present": consent.status == "active",
		})
		self._raise_if_denied(result)
		if not template_hash:
			raise ValueError("biometric_template_hash_required")
		if quality_score < float(self.describe(tenant_id)["configuration"]["modalities"]["quality_threshold"]):
			raise PermissionError("biometric_template_quality_too_low")
		if not retention_policy:
			raise ValueError("biometric_template_retention_policy_required")
		template = BiometricTemplateRecord(
			id=template_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			modality=normalized_modality,
			template_hash=template_hash,
			encrypted=bool(encrypted),
			quality_score=float(quality_score),
			consent_id=consent_id,
			retention_policy=retention_policy,
			created_at=self._now(),
		)
		self._templates[self._tenant_key(tenant_id, template_id)] = template
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type="template_enrolled",
			actor=consent.granted_by,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"template_id": template_id, "modality": normalized_modality, "quality_score": quality_score},
		)
		return template.to_dict()

	def retire_template(self, template_id: str, tenant_id: str, retired_by: str, reason: str) -> dict[str, Any]:
		template = self._require_template(template_id, tenant_id)
		if not retired_by:
			raise ValueError("biometric_template_retiring_actor_required")
		if not reason:
			raise ValueError("biometric_template_retirement_reason_required")
		retired = replace(template, status="retired", retired_by=retired_by, retirement_reason=reason)
		self._templates[self._tenant_key(tenant_id, template_id)] = retired
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=template.subject_id,
			event_type="template_retired",
			actor=retired_by,
			decision="deny",
			reasons=("template_retired",),
			metadata={"template_id": template_id, "reason": reason},
		)
		return retired.to_dict()

	def register_biometric_agent(
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
		self._ensure_new(self._biometric_agents, tenant_id, agent_id, "biometric agent")
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "register_biometric_agent",
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
			raise ValueError("biometric_agent_name_required")
		agent = BiometricAgentRecord(
			id=str(agent_id).strip(),
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
			created_at=self._now(),
		)
		self._biometric_agents[self._tenant_key(tenant_id, agent.id)] = agent
		self._record_audit(
			tenant_id=tenant_id,
			subject_id="biometric-agent",
			event_type="biometric_agent_registered",
			actor=owner,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"agent_id": agent.id, "runtime": runtime_value, "role": role_value},
		)
		return agent.to_dict()

	def validate_biop_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "biometric_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("biop_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_biop_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "validate_biop_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
		})
		accepted = result["decision"] == "allow"
		record = BiopLifecycleBatchRecord(
			id=batch_id or f"biopbatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			status="accepted" if accepted else "denied",
			created_at=self._now(),
		)
		self._lifecycle_batches[self._tenant_key(tenant_id, record.id)] = record
		self._record_audit(
			tenant_id=tenant_id,
			subject_id="biop-lifecycle",
			event_type=f"biop_lifecycle_batch_{record.status}",
			actor="bytewax" if accepted else stream_value,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"batch_id": record.id, "operation": operation_value, "event_stream": stream_value},
		)
		self._raise_if_denied(result)
		return record.to_dict()

	def request_verification(
		self,
		verification_id: str,
		tenant_id: str,
		subject_id: str,
		template_id: str,
		modality: str,
		requested_by: str,
		match_confidence: float,
		liveness_score: float,
		source_jurisdiction: str,
		target_jurisdiction: str,
		privacy_review_recorded: bool = False,
		human_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._ensure_new(self._verifications, tenant_id, verification_id, "verification")
		normalized_modality = self._normalize_modality(modality)
		template = self._active_template(template_id, tenant_id, subject_id, normalized_modality)
		consent = self._active_consent(template.consent_id, tenant_id, subject_id, normalized_modality, target_jurisdiction)
		cross_border = bool(source_jurisdiction and target_jurisdiction and source_jurisdiction != target_jurisdiction)
		minimum_match = float(self.describe(tenant_id)["configuration"]["modalities"]["minimum_match_confidence"])
		minimum_liveness = float(self.describe(tenant_id)["configuration"]["liveness"]["minimum_liveness_score"])
		if not requested_by:
			raise ValueError("biometric_verification_requester_required")
		if liveness_score < minimum_liveness:
			result = self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "authenticate",
				"liveness_passed": False,
				"active_template_present": True,
				"active_consent_present": True,
			})
			self._raise_if_denied(result)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "process_biometric",
			"consent_recorded": True,
			"liveness_passed": liveness_score >= minimum_liveness,
			"cross_border_processing": cross_border,
			"privacy_review_recorded": False if cross_border else True,
			"match_confidence": float(match_confidence),
			"human_review_recorded": False,
			"active_template_present": True,
			"active_consent_present": True,
		})
		status = "verified"
		decision = "allow"
		reasons = self._reasons(result)
		if cross_border:
			status = "pending_privacy_review"
			decision = "require_review"
			reasons = tuple(set(reasons + ("privacy_review_required",)))
		elif match_confidence < minimum_match:
			status = "pending_match_review"
			decision = "require_review"
			reasons = tuple(set(reasons + ("low_match_confidence",)))
		verification = BiometricVerificationRecord(
			id=verification_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			template_id=template_id,
			modality=normalized_modality,
			requested_by=requested_by,
			match_confidence=float(match_confidence),
			liveness_score=float(liveness_score),
			source_jurisdiction=source_jurisdiction,
			target_jurisdiction=target_jurisdiction,
			consent_id=consent.id,
			status=status,
			decision=decision,
			reasons=tuple(sorted(reasons)),
			created_at=self._now(),
		)
		self._verifications[self._tenant_key(tenant_id, verification_id)] = verification
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type="verification_requested",
			actor=requested_by,
			decision=decision,
			reasons=verification.reasons,
			metadata={"verification_id": verification_id, "template_id": template_id, "cross_border": cross_border},
		)
		return verification.to_dict()

	def request_privacy_review(
		self,
		review_id: str,
		tenant_id: str,
		verification_id: str,
		requested_by: str,
		justification: str,
	) -> dict[str, Any]:
		return self._request_review(review_id, tenant_id, verification_id, "privacy", requested_by, justification, "pending_privacy_review")

	def decide_privacy_review(
		self,
		review_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		return self._decide_review(review_id, tenant_id, reviewer, decision, notes, "privacy")

	def request_match_review(
		self,
		review_id: str,
		tenant_id: str,
		verification_id: str,
		requested_by: str,
		justification: str,
	) -> dict[str, Any]:
		return self._request_review(review_id, tenant_id, verification_id, "match", requested_by, justification, "pending_match_review")

	def decide_match_review(
		self,
		review_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		return self._decide_review(review_id, tenant_id, reviewer, decision, notes, "match")

	def list_consents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._consents.values(), tenant_id)

	def list_templates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._templates.values(), tenant_id)

	def list_verifications(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._verifications.values(), tenant_id)

	def list_reviews(self, tenant_id: str | None = None, review_type: str | None = None) -> list[dict[str, Any]]:
		reviews = list(self._reviews.values())
		if review_type is not None:
			reviews = [review for review in reviews if review.review_type == review_type]
		return self._list(reviews, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events.values(), tenant_id)

	def list_biometric_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._biometric_agents.values(), tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches.values(), tenant_id)

	def biometric_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		consents = self.list_consents(tenant_id)
		templates = self.list_templates(tenant_id)
		verifications = self.list_verifications(tenant_id)
		reviews = self.list_reviews(tenant_id)
		return {
			"consent_count": len(consents),
			"active_consent_count": len([item for item in consents if item["status"] == "active"]),
			"template_count": len(templates),
			"active_template_count": len([item for item in templates if item["status"] == "active"]),
			"verification_count": len(verifications),
			"verified_count": len([item for item in verifications if item["status"] == "verified"]),
			"pending_privacy_review_count": len([item for item in verifications if item["status"] == "pending_privacy_review"]),
			"pending_match_review_count": len([item for item in verifications if item["status"] == "pending_match_review"]),
			"rejected_count": len([item for item in verifications if item["status"] == "rejected"]),
			"review_count": len(reviews),
			"pending_review_count": len([item for item in reviews if item["status"] == "pending"]),
			"biometric_agent_count": len(self.list_biometric_agents(tenant_id)),
			"pending_agent_review_count": len([item for item in self.list_biometric_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _request_review(
		self,
		review_id: str,
		tenant_id: str,
		verification_id: str,
		review_type: str,
		requested_by: str,
		justification: str,
		required_status: str,
	) -> dict[str, Any]:
		verification = self._require_verification(verification_id, tenant_id)
		self._ensure_new(self._reviews, tenant_id, review_id, f"{review_type} review")
		if verification.status != required_status:
			raise ValueError(f"{review_type}_review_not_required")
		if any(
			review.tenant_id == tenant_id
			and review.verification_id == verification_id
			and review.review_type == review_type
			and review.status == "pending"
			for review in self._reviews.values()
		):
			raise ValueError(f"{review_type}_review_already_pending")
		if not requested_by:
			raise ValueError(f"{review_type}_review_requester_required")
		if not justification:
			raise ValueError(f"{review_type}_review_justification_required")
		review = BiometricReviewApproval(
			id=review_id,
			tenant_id=tenant_id,
			verification_id=verification_id,
			review_type=review_type,
			requested_by=requested_by,
			justification=justification,
			created_at=self._now(),
		)
		self._reviews[self._tenant_key(tenant_id, review_id)] = review
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=verification.subject_id,
			event_type=f"{review_type}_review_requested",
			actor=requested_by,
			decision="require_review",
			metadata={"review_id": review_id, "verification_id": verification_id},
		)
		return review.to_dict()

	def _decide_review(
		self,
		review_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
		review_type: str,
	) -> dict[str, Any]:
		review = self._require_review(review_id, tenant_id)
		verification = self._require_verification(review.verification_id, tenant_id)
		if review.review_type != review_type:
			raise ValueError(f"{review_type}_review_type_mismatch")
		if review.status != "pending":
			raise ValueError(f"{review_type}_review_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError(f"{review_type}_review_decision_invalid")
		if not reviewer:
			raise ValueError(f"{review_type}_review_reviewer_required")
		if not notes:
			raise ValueError(f"{review_type}_review_notes_required")
		if verification.status not in {"pending_privacy_review", "pending_match_review"}:
			raise ValueError(f"{review_type}_review_not_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": f"approve_{review_type}_review",
			f"{review_type}_reviewer_same_as_requester": reviewer in {review.requested_by, verification.requested_by},
		})
		self._raise_if_denied(result)
		decided = replace(review, decision=decision, reviewer=reviewer, notes=notes, status=decision)
		self._reviews[self._tenant_key(tenant_id, review_id)] = decided
		updated = self._transition_after_review(verification, decided)
		self._verifications[self._tenant_key(tenant_id, verification.id)] = updated
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=verification.subject_id,
			event_type=f"{review_type}_review_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"review_id": review_id, "verification_id": verification.id},
		)
		return updated.to_dict()

	def _transition_after_review(
		self,
		verification: BiometricVerificationRecord,
		review: BiometricReviewApproval,
	) -> BiometricVerificationRecord:
		if review.decision == "rejected":
			return replace(
				verification,
				status="rejected",
				decision="deny",
				reasons=tuple(sorted(set(verification.reasons + (f"{review.review_type}_review_rejected",)))),
				reviewer=review.reviewer,
				reviewer_notes=review.notes,
				privacy_review_id=review.id if review.review_type == "privacy" else verification.privacy_review_id,
				match_review_id=review.id if review.review_type == "match" else verification.match_review_id,
			)
		minimum_match = float(self.describe(verification.tenant_id)["configuration"]["modalities"]["minimum_match_confidence"])
		if review.review_type == "privacy" and verification.match_confidence < minimum_match:
			return replace(
				verification,
				status="pending_match_review",
				decision="require_review",
				reasons=tuple(sorted(set(verification.reasons + ("low_match_confidence",)))),
				reviewer=review.reviewer,
				reviewer_notes=review.notes,
				privacy_review_id=review.id,
			)
		return replace(
			verification,
			status="verified",
			decision="allow",
			reasons=tuple(sorted(reason for reason in verification.reasons if reason not in {"privacy_review_required", "low_match_confidence"})),
			reviewer=review.reviewer,
			reviewer_notes=review.notes,
			privacy_review_id=review.id if review.review_type == "privacy" else verification.privacy_review_id,
			match_review_id=review.id if review.review_type == "match" else verification.match_review_id,
		)

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _ensure_new(self, records: dict[tuple[str, str], Any], tenant_id: str, record_id: str, label: str) -> None:
		self._require_tenant(tenant_id)
		if not record_id:
			raise ValueError(f"{label}_id_required")
		if self._tenant_key(tenant_id, record_id) in records:
			raise ValueError(f"{label} already exists for tenant: {record_id}")

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_consent(self, consent_id: str, tenant_id: str) -> BiometricConsent:
		consent = self._consents.get(self._tenant_key(tenant_id, consent_id))
		if consent is None:
			raise KeyError(f"unknown biometric consent: {consent_id}")
		return consent

	def _active_consent(
		self,
		consent_id: str,
		tenant_id: str,
		subject_id: str,
		modality: str,
		jurisdiction: str,
	) -> BiometricConsent:
		consent = self._require_consent(consent_id, tenant_id)
		if consent.status != "active":
			raise PermissionError("biometric_consent_not_active")
		if consent.subject_id != subject_id:
			raise PermissionError("biometric_consent_subject_mismatch")
		if modality not in consent.modalities:
			raise PermissionError("biometric_consent_modality_mismatch")
		if jurisdiction and consent.jurisdictions and jurisdiction not in consent.jurisdictions:
			raise PermissionError("biometric_consent_jurisdiction_mismatch")
		return consent

	def _require_template(self, template_id: str, tenant_id: str) -> BiometricTemplateRecord:
		template = self._templates.get(self._tenant_key(tenant_id, template_id))
		if template is None:
			raise KeyError(f"unknown biometric template: {template_id}")
		return template

	def _active_template(
		self,
		template_id: str,
		tenant_id: str,
		subject_id: str,
		modality: str,
	) -> BiometricTemplateRecord:
		template = self._require_template(template_id, tenant_id)
		if template.status != "active":
			raise PermissionError("biometric_template_not_active")
		if template.subject_id != subject_id:
			raise PermissionError("biometric_template_subject_mismatch")
		if template.modality != modality:
			raise PermissionError("biometric_template_modality_mismatch")
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "active_template_present": True})
		self._raise_if_denied(result)
		return template

	def _require_verification(self, verification_id: str, tenant_id: str) -> BiometricVerificationRecord:
		verification = self._verifications.get(self._tenant_key(tenant_id, verification_id))
		if verification is None:
			raise KeyError(f"unknown biometric verification: {verification_id}")
		return verification

	def _require_review(self, review_id: str, tenant_id: str) -> BiometricReviewApproval:
		review = self._reviews.get(self._tenant_key(tenant_id, review_id))
		if review is None:
			raise KeyError(f"unknown biometric review: {review_id}")
		return review

	def _normalize_modality(self, modality: str) -> str:
		normalized = str(modality).strip().lower()
		if normalized not in SUPPORTED_MODALITIES:
			raise ValueError(f"unsupported_biometric_modality: {modality}")
		return normalized

	def _normalize_modalities(self, modalities: list[str] | tuple[str, ...]) -> tuple[str, ...]:
		normalized = tuple(self._normalize_modality(item) for item in modalities)
		if not normalized:
			raise ValueError("biometric_consent_modalities_required")
		return normalized

	def _normalize_jurisdictions(self, jurisdictions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
		normalized = tuple(str(item).strip().upper() for item in jurisdictions if str(item).strip())
		if not normalized:
			raise ValueError("biometric_consent_jurisdictions_required")
		return normalized

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> BiometricAuditEvent:
		event_id = f"audit-{len(self._audit_events) + 1:06d}"
		event = BiometricAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
			created_at=self._now(),
		)
		self._audit_events[self._tenant_key(tenant_id, event_id)] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(self._reasons(result))
			raise PermissionError(reasons or "biometric_policy_blocked")

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(
			str(action.get("reason") or action.get("required_action") or "biometric_policy_blocked")
			for action in result.get("actions", [])
		)

	def _list(self, values: Any, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
