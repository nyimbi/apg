"""Dependency-light generated-app runtime for FREC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


StoreKey = tuple[str, str]


@dataclass
class FaceRecord:
	"""Serializable facial-recognition runtime record."""

	id: str
	tenant_id: str
	kind: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FacialRecognitionAgentRecord:
	"""Provider-neutral AI-agent composition record for facial-recognition governance."""

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
	status: str
	created_at: str

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FrecLifecycleBatchRecord:
	"""Bytewax lifecycle batch validation evidence for facial-recognition changes."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: tuple[str, ...]
	status: str
	created_at: str

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


class FrecGuardrailError(ValueError):
	"""Raised when a FREC guardrail denies or requires review for an operation."""

	def __init__(self, result: dict[str, Any]):
		self.result = result
		super().__init__(f"{result['decision']}:{','.join(result['matched_rules'])}")


class FrecService:
	"""In-memory facial-recognition lifecycle facade for generated APG apps."""

	def __init__(self, tenant_id: str = "default", configuration_overrides: dict[str, Any] | None = None):
		self.contract = get_capability_contract(tenant_id, configuration_overrides)
		self.configuration = self.contract["configuration"]
		self._consents: dict[StoreKey, dict[str, Any]] = {}
		self._templates: dict[StoreKey, dict[str, Any]] = {}
		self._liveness: dict[StoreKey, dict[str, Any]] = {}
		self._verifications: dict[StoreKey, dict[str, Any]] = {}
		self._watchlists: dict[StoreKey, dict[str, Any]] = {}
		self._identifications: dict[StoreKey, dict[str, Any]] = {}
		self._reviews: dict[StoreKey, dict[str, Any]] = {}
		self._emotion_events: dict[StoreKey, dict[str, Any]] = {}
		self._facial_recognition_agents: dict[StoreKey, dict[str, Any]] = {}
		self._lifecycle_batches: dict[StoreKey, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		self._agent_runtimes = set(self.contract["agents"]["supported_runtimes"])
		self._agent_roles = set(self.contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(self.contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(self.contract["streaming"]["required_operations"])

	def describe(self) -> dict[str, Any]:
		return self.contract

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules({"tenant_context_present": True, **context})

	def record_face_consent(self, consent_id: str, tenant_id: str, subject_id: str, purpose: str, evidence: str, actor: str = "") -> dict[str, Any]:
		self._ensure_new(self._consents, tenant_id, consent_id)
		self._raise_if_denied({
			"operation": "record_face_consent",
			"tenant_context_present": bool(tenant_id),
			"subject_present": bool(subject_id),
			"purpose_present": bool(purpose),
			"evidence_present": bool(evidence),
		})
		record = FaceRecord(
			id=consent_id,
			tenant_id=tenant_id,
			kind="face_consent",
			status="active",
			metadata={"subject_id": subject_id, "purpose": purpose, "evidence": evidence, "actor": actor or subject_id},
		).as_dict()
		self._consents[self._key(tenant_id, consent_id)] = record
		self._audit(tenant_id, "face_consent_recorded", consent_id, subject_id=subject_id, actor=actor or subject_id, audit_event_recorded=True)
		return record

	def revoke_face_consent(self, consent_id: str, tenant_id: str, revoked_by: str, reason: str) -> dict[str, Any]:
		consent = self._get_tenant_record(self._consents, consent_id, tenant_id)
		if not reason:
			raise ValueError("face_consent_revocation_reason_required")
		for template in self._templates.values():
			if template["tenant_id"] == tenant_id and template["metadata"]["consent_id"] == consent_id and template["status"] == "active":
				template["status"] = "retired"
				template["metadata"]["retirement_reason"] = "consent_revoked"
		consent["status"] = "revoked"
		consent["metadata"]["revoked_by"] = revoked_by
		consent["metadata"]["revocation_reason"] = reason
		self._audit(tenant_id, "face_consent_revoked", consent_id, subject_id=consent["metadata"]["subject_id"], actor=revoked_by, audit_event_recorded=True)
		return consent

	def enroll_face(
		self,
		template_id: str,
		tenant_id: str,
		subject_id: str,
		consent_id: str,
		template_hash: str,
		face_quality: float,
		template_encrypted: bool = True,
		retention_policy: str = "face-template-365d",
		recapture_completed: bool = True,
	) -> dict[str, Any]:
		self._ensure_new(self._templates, tenant_id, template_id)
		consent = self._get_tenant_record(self._consents, consent_id, tenant_id)
		context = {
			"operation": "enroll_face",
			"tenant_context_present": bool(tenant_id),
			"consent_recorded": bool(consent),
			"active_consent_present": consent["status"] == "active",
			"template_hash_present": bool(template_hash),
			"template_encrypted": template_encrypted,
			"face_quality": float(face_quality),
			"recapture_completed": recapture_completed,
			"retention_policy_present": bool(retention_policy),
		}
		self._raise_if_review_required(context)
		record = FaceRecord(
			id=template_id,
			tenant_id=tenant_id,
			kind="face_template",
			status="active",
			metadata={
				"subject_id": subject_id,
				"consent_id": consent_id,
				"template_hash": template_hash,
				"face_quality": float(face_quality),
				"template_encrypted": template_encrypted,
				"retention_policy": retention_policy,
			},
		).as_dict()
		self._templates[self._key(tenant_id, template_id)] = record
		self._audit(tenant_id, "face_template_enrolled", template_id, subject_id=subject_id, actor=subject_id, audit_event_recorded=True)
		return record

	def retire_template(self, template_id: str, tenant_id: str, retired_by: str, reason: str) -> dict[str, Any]:
		template = self._get_tenant_record(self._templates, template_id, tenant_id)
		self._raise_if_denied({"operation": "retire_template", "tenant_context_present": bool(tenant_id), "retirement_reason_present": bool(reason)})
		template["status"] = "retired"
		template["metadata"]["retired_by"] = retired_by
		template["metadata"]["retirement_reason"] = reason
		self._audit(tenant_id, "face_template_retired", template_id, subject_id=template["metadata"]["subject_id"], actor=retired_by, audit_event_recorded=True)
		return template

	def record_liveness(
		self,
		liveness_id: str,
		tenant_id: str,
		subject_id: str,
		liveness_score: float,
		spoof_detected: bool = False,
		deepfake_detected: bool = False,
	) -> dict[str, Any]:
		self._ensure_new(self._liveness, tenant_id, liveness_id)
		context = {
			"operation": "authenticate_face",
			"tenant_context_present": bool(tenant_id),
			"liveness_passed": float(liveness_score) >= self.configuration["liveness"]["minimum_liveness_score"],
			"liveness_score": float(liveness_score),
			"spoof_detected": spoof_detected,
			"deepfake_detected": deepfake_detected,
		}
		self._raise_if_denied(context)
		record = FaceRecord(
			id=liveness_id,
			tenant_id=tenant_id,
			kind="face_liveness",
			status="passed",
			metadata={"subject_id": subject_id, "liveness_score": float(liveness_score), "spoof_detected": spoof_detected, "deepfake_detected": deepfake_detected},
		).as_dict()
		self._liveness[self._key(tenant_id, liveness_id)] = record
		self._audit(tenant_id, "face_liveness_recorded", liveness_id, subject_id=subject_id, actor=subject_id, audit_event_recorded=True)
		return record

	def verify_face(
		self,
		verification_id: str,
		tenant_id: str,
		subject_id: str,
		template_id: str,
		liveness_id: str,
		match_confidence: float,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._ensure_new(self._verifications, tenant_id, verification_id)
		template = self._get_tenant_record(self._templates, template_id, tenant_id)
		liveness = self._get_tenant_record(self._liveness, liveness_id, tenant_id)
		context = {
			"operation": "verify_face",
			"tenant_context_present": bool(tenant_id),
			"active_template_present": template["status"] == "active",
			"subject_matches_template": template["metadata"]["subject_id"] == subject_id,
			"liveness_present": bool(liveness),
			"match_confidence": float(match_confidence),
			"review_recorded": review_recorded,
		}
		result = self._raise_if_review_required(context)
		record = FaceRecord(
			id=verification_id,
			tenant_id=tenant_id,
			kind="face_verification",
			status="verified",
			metadata={"subject_id": subject_id, "template_id": template_id, "liveness_id": liveness_id, "match_confidence": float(match_confidence), "decision": result["decision"]},
		).as_dict()
		self._verifications[self._key(tenant_id, verification_id)] = record
		self._audit(tenant_id, "face_verified", verification_id, subject_id=subject_id, actor=subject_id, audit_event_recorded=True)
		return record

	def create_watchlist(self, watchlist_id: str, tenant_id: str, name: str, policy_id: str, owner: str, reason: str) -> dict[str, Any]:
		self._ensure_new(self._watchlists, tenant_id, watchlist_id)
		self._raise_if_denied({
			"operation": "create_watchlist",
			"tenant_context_present": bool(tenant_id),
			"watchlist_policy_attached": bool(policy_id),
			"owner_present": bool(owner),
			"reason_present": bool(reason),
		})
		record = FaceRecord(id=watchlist_id, tenant_id=tenant_id, kind="face_watchlist", status="active", metadata={"name": name, "policy_id": policy_id, "owner": owner, "reason": reason, "subjects": []}).as_dict()
		self._watchlists[self._key(tenant_id, watchlist_id)] = record
		self._audit(tenant_id, "watchlist_created", watchlist_id, actor=owner, audit_event_recorded=True)
		return record

	def add_watchlist_subject(self, watchlist_id: str, tenant_id: str, subject_id: str, template_id: str, added_by: str, reason: str) -> dict[str, Any]:
		watchlist = self._get_tenant_record(self._watchlists, watchlist_id, tenant_id)
		template = self._get_tenant_record(self._templates, template_id, tenant_id)
		self._raise_if_denied({"operation": "add_watchlist_subject", "tenant_context_present": bool(tenant_id), "active_template_present": template["status"] == "active", "reason_present": bool(reason)})
		watchlist["metadata"]["subjects"].append({"subject_id": subject_id, "template_id": template_id, "added_by": added_by, "reason": reason})
		self._audit(tenant_id, "watchlist_subject_added", watchlist_id, subject_id=subject_id, actor=added_by, audit_event_recorded=True)
		return watchlist

	def identify_face(self, identification_id: str, tenant_id: str, watchlist_id: str, candidate_subject_id: str, identification_confidence: float, review_recorded: bool = False) -> dict[str, Any]:
		self._ensure_new(self._identifications, tenant_id, identification_id)
		watchlist = self._get_tenant_record(self._watchlists, watchlist_id, tenant_id)
		subjects = watchlist["metadata"]["subjects"]
		watchlist_hit = any(item["subject_id"] == candidate_subject_id for item in subjects)
		context = {
			"operation": "identify_face",
			"tenant_context_present": bool(tenant_id),
			"watchlist_policy_attached": bool(watchlist["metadata"].get("policy_id")),
			"watchlist_hit": watchlist_hit,
			"identification_confidence": float(identification_confidence),
			"review_recorded": review_recorded,
		}
		self._raise_if_review_required(context)
		record = FaceRecord(
			id=identification_id,
			tenant_id=tenant_id,
			kind="face_identification",
			status="matched" if watchlist_hit else "not_matched",
			metadata={"watchlist_id": watchlist_id, "candidate_subject_id": candidate_subject_id, "identification_confidence": float(identification_confidence), "watchlist_hit": watchlist_hit},
		).as_dict()
		self._identifications[self._key(tenant_id, identification_id)] = record
		self._audit(tenant_id, "face_identified", identification_id, subject_id=candidate_subject_id, actor="system", audit_event_recorded=True)
		return record

	def request_review(self, review_id: str, tenant_id: str, subject_id: str, requested_by: str, reason: str) -> dict[str, Any]:
		self._ensure_new(self._reviews, tenant_id, review_id)
		if any(review["tenant_id"] == tenant_id and review["status"] == "pending" and review["metadata"]["subject_id"] == subject_id for review in self._reviews.values()):
			self._raise_if_denied({"operation": "request_review", "tenant_context_present": bool(tenant_id), "pending_review_exists": True})
		record = FaceRecord(id=review_id, tenant_id=tenant_id, kind="face_review", status="pending", metadata={"subject_id": subject_id, "requested_by": requested_by, "reason": reason}).as_dict()
		self._reviews[self._key(tenant_id, review_id)] = record
		self._audit(tenant_id, "face_review_requested", review_id, subject_id=subject_id, actor=requested_by, audit_event_recorded=True)
		return record

	def decide_review(self, review_id: str, tenant_id: str, reviewer: str, decision: str, notes: str) -> dict[str, Any]:
		review = self._get_tenant_record(self._reviews, review_id, tenant_id)
		self._raise_if_denied({
			"operation": "decide_review",
			"tenant_context_present": bool(tenant_id),
			"reviewer_same_as_requester": reviewer == review["metadata"]["requested_by"],
			"notes_present": bool(notes),
		})
		review["status"] = decision
		review["metadata"]["reviewer"] = reviewer
		review["metadata"]["decision"] = decision
		review["metadata"]["notes"] = notes
		self._audit(tenant_id, "face_review_decided", review_id, subject_id=review["metadata"]["subject_id"], actor=reviewer, audit_event_recorded=True)
		return review

	def analyze_emotion(self, event_id: str, tenant_id: str, subject_id: str, approved_purpose_recorded: bool, aggregate_only: bool = True) -> dict[str, Any]:
		self._ensure_new(self._emotion_events, tenant_id, event_id)
		self._raise_if_denied({
			"operation": "analyze_emotion",
			"tenant_context_present": bool(tenant_id),
			"emotion_analysis_requested": True,
			"approved_purpose_recorded": approved_purpose_recorded,
			"aggregate_only": aggregate_only,
			"individual_emotion_approval_recorded": False,
		})
		record = FaceRecord(id=event_id, tenant_id=tenant_id, kind="emotion_analysis", status="completed", metadata={"subject_id": subject_id, "aggregate_only": aggregate_only}).as_dict()
		self._emotion_events[self._key(tenant_id, event_id)] = record
		self._audit(tenant_id, "emotion_analysis_completed", event_id, subject_id=subject_id, actor="system", audit_event_recorded=True)
		return record

	def register_facial_recognition_agent(
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
		self._ensure_new(self._facial_recognition_agents, tenant_id, agent_id)
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "register_facial_recognition_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		if result["decision"] == "deny":
			raise FrecGuardrailError(result)
		if not str(name or "").strip():
			raise ValueError("facial_recognition_agent_name_required")
		record = FacialRecognitionAgentRecord(
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
			created_at=datetime.now(timezone.utc).isoformat(),
		).as_dict()
		self._facial_recognition_agents[self._key(tenant_id, record["id"])] = record
		self._audit(tenant_id, "facial_recognition_agent_registered", record["id"], actor=owner, audit_event_recorded=True, decision=result["decision"], matched_rules=result["matched_rules"])
		return record

	def validate_frec_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "facial_recognition_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		if not str(tenant_id or "").strip():
			self._raise_if_denied({"tenant_context_present": False})
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("frec_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_frec_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"operation": "validate_frec_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
		})
		accepted = result["decision"] == "allow"
		record = FrecLifecycleBatchRecord(
			id=batch_id or f"frecbatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			status="accepted" if accepted else "denied",
			created_at=datetime.now(timezone.utc).isoformat(),
		).as_dict()
		self._lifecycle_batches[self._key(tenant_id, record["id"])] = record
		self._audit(tenant_id, f"frec_lifecycle_batch_{record['status']}", record["id"], actor="bytewax" if accepted else stream_value, audit_event_recorded=True, decision=result["decision"], matched_rules=result["matched_rules"])
		if result["decision"] == "deny":
			raise FrecGuardrailError(result)
		return record

	def list_consents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._consents, tenant_id)

	def list_templates(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._templates, tenant_id)

	def list_liveness(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._liveness, tenant_id)

	def list_verifications(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._verifications, tenant_id)

	def list_watchlists(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._watchlists, tenant_id)

	def list_identifications(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._identifications, tenant_id)

	def list_reviews(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._reviews, tenant_id)

	def list_emotion_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._emotion_events, tenant_id)

	def list_facial_recognition_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._facial_recognition_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [event for event in self._audit_events if event["tenant_id"] == tenant_id]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"consent_count": len(self.list_consents(tenant_id)),
			"active_template_count": len([item for item in self.list_templates(tenant_id) if item["status"] == "active"]),
			"liveness_count": len(self.list_liveness(tenant_id)),
			"verification_count": len(self.list_verifications(tenant_id)),
			"watchlist_count": len(self.list_watchlists(tenant_id)),
			"identification_count": len(self.list_identifications(tenant_id)),
			"review_count": len(self.list_reviews(tenant_id)),
			"emotion_event_count": len(self.list_emotion_events(tenant_id)),
			"facial_recognition_agent_count": len(self.list_facial_recognition_agents(tenant_id)),
			"pending_agent_review_count": len([item for item in self.list_facial_recognition_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def package(self, tenant_id: str) -> dict[str, Any]:
		return {
			"contract": self.contract,
			"summary": self.dashboard_summary(tenant_id),
			"consents": self.list_consents(tenant_id),
			"templates": self.list_templates(tenant_id),
			"liveness": self.list_liveness(tenant_id),
			"verifications": self.list_verifications(tenant_id),
			"watchlists": self.list_watchlists(tenant_id),
			"identifications": self.list_identifications(tenant_id),
			"reviews": self.list_reviews(tenant_id),
			"emotion_events": self.list_emotion_events(tenant_id),
			"facial_recognition_agents": self.list_facial_recognition_agents(tenant_id),
			"lifecycle_batches": self.list_lifecycle_batches(tenant_id),
			"audit_events": self.list_audit_events(tenant_id),
		}

	def _tenant_records(self, records: dict[StoreKey, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [record for record in records.values() if record["tenant_id"] == tenant_id]

	def _get_tenant_record(self, records: dict[StoreKey, dict[str, Any]], record_id: str, tenant_id: str) -> dict[str, Any]:
		record = records.get(self._key(tenant_id, record_id))
		if record is None:
			cross_tenant_hit = any(candidate["id"] == record_id for candidate in records.values())
			if cross_tenant_hit:
				result = evaluate_capability_rules({"tenant_context_present": bool(tenant_id), "cross_tenant_access": True})
				raise FrecGuardrailError(result)
			raise KeyError(record_id)
		return record

	def _ensure_new(self, records: dict[StoreKey, dict[str, Any]], tenant_id: str, record_id: str) -> None:
		if not record_id:
			raise ValueError("face_record_id_required")
		if self._key(tenant_id, record_id) in records:
			raise ValueError(f"face_record_already_exists:{record_id}")

	def _key(self, tenant_id: str, record_id: str) -> StoreKey:
		return (tenant_id, record_id)

	def _raise_if_denied(self, context: dict[str, Any]) -> dict[str, Any]:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise FrecGuardrailError(result)
		return result

	def _raise_if_review_required(self, context: dict[str, Any]) -> dict[str, Any]:
		result = self._raise_if_denied(context)
		if result["decision"] == "require_review":
			raise FrecGuardrailError(result)
		return result

	def _audit(self, tenant_id: str, event_type: str, record_id: str, actor: str = "system", audit_event_recorded: bool = True, **metadata: Any) -> None:
		self._raise_if_denied({"tenant_context_present": bool(tenant_id), "state_change_requested": True, "audit_event_recorded": audit_event_recorded})
		self._audit_events.append({
			"id": f"audit_{len(self._audit_events) + 1}",
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"subject_id": metadata.get("subject_id", ""),
			"actor": actor,
			"metadata": metadata,
			"created_at": datetime.now(timezone.utc).isoformat(),
		})


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
