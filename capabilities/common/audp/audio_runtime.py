"""Dependency-light AUDP audio governance runtime for package composition."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .capability_contract import (
	SUPPORTED_AUDIO_AGENT_ROLES,
	SUPPORTED_AUDIO_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	AudioAgentRecord,
	AudioConsentRecord,
	AudioGovernanceEvent,
	AudioModelPolicyRecord,
	AudioProcessingJobRecord,
	AudioSynthesisReviewRecord,
	AudioTranscriptReviewRecord,
)


TRANSCRIPTION_REVIEW_THRESHOLD = 0.78


class AudpService:
	"""Tenant-scoped audio-processing governance facade for generated APG apps."""

	def __init__(self) -> None:
		self._consents: dict[tuple[str, str], AudioConsentRecord] = {}
		self._policies: dict[tuple[str, str], AudioModelPolicyRecord] = {}
		self._jobs: dict[tuple[str, str], AudioProcessingJobRecord] = {}
		self._transcript_reviews: dict[tuple[str, str], AudioTranscriptReviewRecord] = {}
		self._synthesis_reviews: dict[tuple[str, str], AudioSynthesisReviewRecord] = {}
		self._agents: dict[tuple[str, str], AudioAgentRecord] = {}
		self._governance_events: list[AudioGovernanceEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_consent(
		self,
		consent_id: str,
		tenant_id: str,
		consent_type: str,
		subject_id: str,
		granted_by: str,
		evidence: str,
		scope: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, consent_id) in self._consents:
			raise ValueError(f"audio consent already exists for tenant: {consent_id}")
		if consent_type not in {"recording", "voice_owner"}:
			raise ValueError("audio consent type must be recording or voice_owner")
		if not subject_id:
			raise ValueError("audio consent subject is required")
		if not granted_by:
			raise ValueError("audio consent grantor is required")
		if not evidence:
			raise ValueError("audio consent evidence is required")
		record = AudioConsentRecord(
			id=consent_id,
			tenant_id=tenant_id,
			consent_type=consent_type,
			subject_id=subject_id,
			granted_by=granted_by,
			scope=dict(scope or {}),
			evidence=evidence,
		)
		self._consents[self._tenant_key(tenant_id, consent_id)] = record
		self._record_governance(
			tenant_id=tenant_id,
			event_type="audio_consent_recorded",
			subject_id=consent_id,
			message=f"Recorded {consent_type} consent {consent_id}.",
			evidence={"subject_id": subject_id, "granted_by": granted_by},
		)
		return record.model_dump(mode="json")

	def attach_model_policy(
		self,
		policy_id: str,
		tenant_id: str,
		model_id: str,
		policy_name: str,
		allowed_operations: list[str] | tuple[str, ...],
		attached_by: str,
		risk_tier: str = "standard",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, policy_id) in self._policies:
			raise ValueError(f"audio model policy already exists for tenant: {policy_id}")
		if not model_id:
			raise ValueError("audio model ID is required")
		if not policy_name:
			raise ValueError("audio model policy name is required")
		if not allowed_operations:
			raise ValueError("audio model policy allowed operations are required")
		if not attached_by:
			raise ValueError("audio model policy attach actor is required")
		record = AudioModelPolicyRecord(
			id=policy_id,
			tenant_id=tenant_id,
			model_id=model_id,
			policy_name=policy_name,
			allowed_operations=list(allowed_operations),
			attached_by=attached_by,
			risk_tier=risk_tier,
		)
		self._policies[self._tenant_key(tenant_id, policy_id)] = record
		self._record_governance(
			tenant_id=tenant_id,
			event_type="audio_model_policy_attached",
			subject_id=policy_id,
			message=f"Attached audio model policy {policy_name}.",
			evidence={"model_id": model_id, "allowed_operations": list(allowed_operations)},
		)
		return record.model_dump(mode="json")

	def request_transcription(
		self,
		job_id: str,
		tenant_id: str,
		audio_source_id: str,
		requested_by: str,
		model_id: str,
		language_code: str = "auto",
		confidence: float = 1.0,
		retention_policy: str = "default",
		result: dict[str, Any] | None = None,
		human_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		self._ensure_new_job(tenant_id, job_id)
		self._require_recording_consent(tenant_id, audio_source_id)
		self._require_model_policy(tenant_id, model_id, "transcription")
		self._require_requester_and_retention(requested_by, retention_policy)
		rule_result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "process_recording",
			"audio_job_requested": True,
			"recording_consent_recorded": True,
			"model_invocation": True,
			"model_policy_attached": True,
			"retention_policy_present": bool(retention_policy),
			"transcription_confidence": confidence,
			"human_review_recorded": human_review_recorded,
		})
		if rule_result["decision"] == "deny":
			_raise_if_blocked(rule_result)
		status = "pending_review" if rule_result["decision"] == "require_review" else "completed"
		job = self._create_job(
			job_id=job_id,
			tenant_id=tenant_id,
			job_type="transcription",
			audio_source_id=audio_source_id,
			requested_by=requested_by,
			model_id=model_id,
			language_code=language_code,
			confidence=confidence,
			status=status,
			retention_policy=retention_policy,
			result=result or {"transcript": ""},
		)
		self._record_governance(
			tenant_id=tenant_id,
			event_type="transcription_requested",
			subject_id=job_id,
			message=f"Requested transcription {job_id}.",
			evidence={"confidence": confidence, "status": status},
		)
		if status == "pending_review":
			review = AudioTranscriptReviewRecord(
				id=f"transcript-review:{job_id}",
				tenant_id=tenant_id,
				job_id=job_id,
				confidence=confidence,
			)
			self._transcript_reviews[self._tenant_key(tenant_id, review.id)] = review
			self._record_governance(
				tenant_id=tenant_id,
				event_type="transcript_review_requested",
				subject_id=review.id,
				message=f"Requested transcript review for {job_id}.",
				evidence={"confidence": confidence},
			)
			return {"job": job.model_dump(mode="json"), "transcript_review": review.model_dump(mode="json")}
		return job.model_dump(mode="json")

	def decide_transcript_review(
		self,
		review_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		review = self._transcript_reviews.get(self._tenant_key(tenant_id, review_id))
		if review is None:
			raise KeyError(f"unknown transcript review for tenant: {review_id}")
		if decision not in {"approved", "rejected"}:
			raise ValueError("transcript review decision must be approved or rejected")
		if not reviewer:
			raise ValueError("transcript reviewer is required")
		if not notes:
			raise ValueError("transcript reviewer notes are required")
		decided = AudioTranscriptReviewRecord(
			id=review.id,
			tenant_id=review.tenant_id,
			job_id=review.job_id,
			confidence=review.confidence,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
			created_at=review.created_at,
			decided_at=datetime.utcnow(),
		)
		self._transcript_reviews[self._tenant_key(tenant_id, review_id)] = decided
		job = self._get_job(tenant_id, review.job_id)
		self._jobs[self._tenant_key(tenant_id, job.id)] = self._replace_job_status(
			job,
			"completed" if decision == "approved" else "blocked",
		)
		self._record_governance(
			tenant_id=tenant_id,
			event_type="transcript_review_decided",
			subject_id=review_id,
			message=f"Transcript review {review_id} was {decision}.",
			evidence={"reviewer": reviewer, "decision": decision},
		)
		return decided.model_dump(mode="json")

	def request_synthesis(
		self,
		job_id: str,
		tenant_id: str,
		text: str,
		requested_by: str,
		model_id: str,
		watermark_applied: bool,
		retention_policy: str = "default",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		self._ensure_new_job(tenant_id, job_id)
		self._require_model_policy(tenant_id, model_id, "synthesis")
		self._require_requester_and_retention(requested_by, retention_policy)
		if not text:
			raise ValueError("synthesis text is required")
		rule_result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"audio_job_requested": True,
			"synthetic_audio_requested": True,
			"watermark_applied": watermark_applied,
			"synthetic_release_reviewed": False,
			"model_invocation": True,
			"model_policy_attached": True,
			"retention_policy_present": bool(retention_policy),
		})
		if rule_result["decision"] == "deny":
			_raise_if_blocked(rule_result)
		job = self._create_job(
			job_id=job_id,
			tenant_id=tenant_id,
			job_type="synthesis",
			audio_source_id=f"synthetic:{job_id}",
			requested_by=requested_by,
			model_id=model_id,
			status="pending_review",
			watermark_applied=watermark_applied,
			retention_policy=retention_policy,
			result={"text": text, "watermarked": watermark_applied},
		)
		review = AudioSynthesisReviewRecord(
			id=f"synthesis-review:{job_id}",
			tenant_id=tenant_id,
			job_id=job_id,
			watermark_applied=watermark_applied,
		)
		self._synthesis_reviews[self._tenant_key(tenant_id, review.id)] = review
		self._record_governance(
			tenant_id=tenant_id,
			event_type="synthesis_requested",
			subject_id=job_id,
			message=f"Requested synthesis {job_id}.",
			evidence={"watermark_applied": watermark_applied},
		)
		self._record_governance(
			tenant_id=tenant_id,
			event_type="synthesis_review_requested",
			subject_id=review.id,
			message=f"Requested synthetic-audio release review for {job_id}.",
			evidence={"watermark_applied": watermark_applied},
		)
		return {"job": job.model_dump(mode="json"), "synthesis_review": review.model_dump(mode="json")}

	def decide_synthesis_review(
		self,
		review_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		review = self._synthesis_reviews.get(self._tenant_key(tenant_id, review_id))
		if review is None:
			raise KeyError(f"unknown synthesis review for tenant: {review_id}")
		if decision not in {"approved", "rejected"}:
			raise ValueError("synthesis review decision must be approved or rejected")
		if not reviewer:
			raise ValueError("synthesis reviewer is required")
		if not notes:
			raise ValueError("synthesis reviewer notes are required")
		decided = AudioSynthesisReviewRecord(
			id=review.id,
			tenant_id=review.tenant_id,
			job_id=review.job_id,
			watermark_applied=review.watermark_applied,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
			created_at=review.created_at,
			decided_at=datetime.utcnow(),
		)
		self._synthesis_reviews[self._tenant_key(tenant_id, review_id)] = decided
		job = self._get_job(tenant_id, review.job_id)
		self._jobs[self._tenant_key(tenant_id, job.id)] = self._replace_job_status(
			job,
			"completed" if decision == "approved" else "blocked",
		)
		self._record_governance(
			tenant_id=tenant_id,
			event_type="synthesis_review_decided",
			subject_id=review_id,
			message=f"Synthesis review {review_id} was {decision}.",
			evidence={"reviewer": reviewer, "decision": decision},
		)
		return decided.model_dump(mode="json")

	def request_voice_clone(
		self,
		job_id: str,
		tenant_id: str,
		voice_owner_id: str,
		requested_by: str,
		model_id: str,
		retention_policy: str = "default",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		self._ensure_new_job(tenant_id, job_id)
		self._require_voice_owner_consent(tenant_id, voice_owner_id)
		self._require_model_policy(tenant_id, model_id, "voice_cloning")
		self._require_requester_and_retention(requested_by, retention_policy)
		rule_result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "clone_voice",
			"audio_job_requested": True,
			"voice_owner_consent_recorded": True,
			"model_invocation": True,
			"model_policy_attached": True,
			"retention_policy_present": bool(retention_policy),
		})
		_raise_if_blocked(rule_result)
		job = self._create_job(
			job_id=job_id,
			tenant_id=tenant_id,
			job_type="voice_cloning",
			audio_source_id=voice_owner_id,
			requested_by=requested_by,
			model_id=model_id,
			status="completed",
			retention_policy=retention_policy,
			result={"voice_owner_id": voice_owner_id},
		)
		self._record_governance(
			tenant_id=tenant_id,
			event_type="voice_clone_requested",
			subject_id=job_id,
			message=f"Requested voice clone {job_id}.",
			evidence={"voice_owner_id": voice_owner_id},
		)
		return job.model_dump(mode="json")

	def request_analysis(
		self,
		job_id: str,
		tenant_id: str,
		audio_source_id: str,
		requested_by: str,
		model_id: str,
		analysis_types: list[str] | tuple[str, ...],
		retention_policy: str = "default",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		self._ensure_new_job(tenant_id, job_id)
		self._require_recording_consent(tenant_id, audio_source_id)
		self._require_model_policy(tenant_id, model_id, "analysis")
		self._require_requester_and_retention(requested_by, retention_policy)
		if not analysis_types:
			raise ValueError("analysis types are required")
		rule_result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "process_recording",
			"audio_job_requested": True,
			"recording_consent_recorded": True,
			"model_invocation": True,
			"model_policy_attached": True,
			"retention_policy_present": bool(retention_policy),
		})
		_raise_if_blocked(rule_result)
		job = self._create_job(
			job_id=job_id,
			tenant_id=tenant_id,
			job_type="analysis",
			audio_source_id=audio_source_id,
			requested_by=requested_by,
			model_id=model_id,
			status="completed",
			retention_policy=retention_policy,
			result={"analysis_types": list(analysis_types)},
		)
		self._record_governance(
			tenant_id=tenant_id,
			event_type="analysis_requested",
			subject_id=job_id,
			message=f"Requested audio analysis {job_id}.",
			evidence={"analysis_types": list(analysis_types)},
		)
		return job.model_dump(mode="json")

	def register_audio_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool,
		policy_ref: str = "",
		registered: bool = True,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		normalized_runtime = _normalize_agent_runtime(runtime)
		normalized_role = _normalize_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"audio_agent_present": True,
			"agent_registered": bool(registered),
			"agent_runtime_supported": bool(normalized_runtime),
			"agent_scope_present": bool(scope.strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		_raise_if_blocked(result)
		if self._tenant_key(tenant_id, agent_id) in self._agents:
			raise ValueError(f"audio agent already exists for tenant: {agent_id}")
		agent = AudioAgentRecord(
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
		self._agents[self._tenant_key(tenant_id, agent_id)] = agent
		self._record_governance(
			tenant_id=tenant_id,
			event_type="audio_agent_registered",
			subject_id=agent_id,
			message=f"Registered audio agent {agent.name}.",
			evidence={"runtime": normalized_runtime, "role": normalized_role, "scope": scope},
		)
		return agent.model_dump(mode="json")

	def change_job_state(
		self,
		tenant_id: str,
		job_id: str,
		status: str,
		reason: str,
		audit_recorded: bool = True,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		job = self._get_job(tenant_id, job_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"state_change_requested": True,
			"state_change_reason_present": bool(reason.strip()),
			"audit_event_recorded": bool(audit_recorded),
		})
		_raise_if_blocked(result)
		if status not in {"active", "pending_review", "blocked", "completed", "cancelled"}:
			raise ValueError("audio job status must be active, pending_review, blocked, completed, or cancelled")
		updated = self._replace_job_status(job, status)
		self._jobs[self._tenant_key(tenant_id, job_id)] = updated
		self._record_governance(
			tenant_id=tenant_id,
			event_type="audio_job_state_changed",
			subject_id=job_id,
			message=reason,
			evidence={"status": status},
		)
		return updated.model_dump(mode="json")

	def list_consents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._consents, tenant_id)

	def list_model_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._policies, tenant_id)

	def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._jobs, tenant_id)

	def list_transcript_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._transcript_reviews, tenant_id)

	def list_synthesis_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._synthesis_reviews, tenant_id)

	def list_audio_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._agents, tenant_id)

	def list_governance_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._governance_events)
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.model_dump(mode="json") for event in events]

	def audio_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		jobs = self.list_jobs(tenant_id)
		transcript_reviews = self.list_transcript_reviews(tenant_id)
		synthesis_reviews = self.list_synthesis_reviews(tenant_id)
		return {
			"tenant_id": tenant_id,
			"consent_count": len(self.list_consents(tenant_id)),
			"model_policy_count": len(self.list_model_policies(tenant_id)),
			"job_count": len(jobs),
			"transcription_count": len([job for job in jobs if job["job_type"] == "transcription"]),
			"synthesis_count": len([job for job in jobs if job["job_type"] == "synthesis"]),
			"analysis_count": len([job for job in jobs if job["job_type"] == "analysis"]),
			"agent_count": len(self.list_audio_agents(tenant_id)),
			"pending_review_count": len([review for review in transcript_reviews + synthesis_reviews if review["decision"] == "pending"]),
			"blocked_job_count": len([job for job in jobs if job["status"] == "blocked"]),
			"governance_event_count": len(self.list_governance_events(tenant_id)),
		}

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_jobs(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "completed",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		audio_source_id = str(metadata.get("audio_source_id") or record_id)
		model_id = str(metadata.get("model_id") or "manual-model")
		if status not in {"active", "pending_review", "blocked", "completed"}:
			raise ValueError("audio job status must be active, pending_review, blocked, or completed")
		record = self.request_transcription(
			job_id=record_id,
			tenant_id=tenant_id,
			audio_source_id=audio_source_id,
			requested_by=str(metadata.get("requested_by") or "operations"),
			model_id=model_id,
			language_code=str(metadata.get("language_code") or "auto"),
			confidence=float(metadata.get("confidence", 1.0)),
			retention_policy=str(metadata.get("retention_policy") or "default"),
			result=metadata,
			human_review_recorded=_coerce_bool(metadata.get("human_review_recorded", False)),
		)
		job = record["job"] if "job" in record else record
		if job["status"] == "pending_review":
			return job
		updated = self._replace_job_status(self._get_job(tenant_id, record_id), status)
		self._jobs[self._tenant_key(tenant_id, record_id)] = updated
		return updated.model_dump(mode="json")

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _enforce_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		_raise_if_blocked(result)

	def _ensure_new_job(self, tenant_id: str, job_id: str) -> None:
		if self._tenant_key(tenant_id, job_id) in self._jobs:
			raise ValueError(f"audio job already exists for tenant: {job_id}")

	def _require_requester_and_retention(self, requested_by: str, retention_policy: str) -> None:
		if not requested_by:
			raise ValueError("audio job requester is required")
		if not retention_policy:
			raise ValueError("audio retention policy is required")

	def _has_recording_consent(self, tenant_id: str, audio_source_id: str) -> bool:
		return any(
			consent.tenant_id == tenant_id
			and consent.consent_type == "recording"
			and consent.status == "active"
			and (consent.subject_id == audio_source_id or consent.scope.get("audio_source_id") == audio_source_id)
			for consent in self._consents.values()
		)

	def _require_recording_consent(self, tenant_id: str, audio_source_id: str) -> None:
		if self._has_recording_consent(tenant_id, audio_source_id):
			return
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "process_recording",
			"recording_consent_recorded": False,
		})
		_raise_if_blocked(result)

	def _require_voice_owner_consent(self, tenant_id: str, voice_owner_id: str) -> None:
		if any(
			consent.tenant_id == tenant_id
			and consent.consent_type == "voice_owner"
			and consent.status == "active"
			and consent.subject_id == voice_owner_id
			for consent in self._consents.values()
		):
			return
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "clone_voice",
			"voice_owner_consent_recorded": False,
		})
		_raise_if_blocked(result)

	def _model_policy_for(self, tenant_id: str, model_id: str, operation: str) -> AudioModelPolicyRecord | None:
		for policy in self._policies.values():
			if policy.tenant_id != tenant_id or policy.model_id != model_id:
				continue
			if operation in policy.allowed_operations or "*" in policy.allowed_operations:
				return policy
		return None

	def _require_model_policy(self, tenant_id: str, model_id: str, operation: str) -> None:
		if self._model_policy_for(tenant_id, model_id, operation) is not None:
			return
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"model_invocation": True,
			"model_policy_attached": False,
		})
		_raise_if_blocked(result)

	def _create_job(
		self,
		job_id: str,
		tenant_id: str,
		job_type: str,
		audio_source_id: str,
		requested_by: str,
		model_id: str,
		language_code: str = "auto",
		confidence: float = 1.0,
		status: str = "completed",
		watermark_applied: bool = False,
		retention_policy: str = "default",
		result: dict[str, Any] | None = None,
	) -> AudioProcessingJobRecord:
		job = AudioProcessingJobRecord(
			id=job_id,
			tenant_id=tenant_id,
			job_type=job_type,
			audio_source_id=audio_source_id,
			requested_by=requested_by,
			model_id=model_id,
			language_code=language_code,
			confidence=confidence,
			status=status,
			watermark_applied=watermark_applied,
			retention_policy=retention_policy,
			result=dict(result or {}),
		)
		self._jobs[self._tenant_key(tenant_id, job_id)] = job
		return job

	def _get_job(self, tenant_id: str, job_id: str) -> AudioProcessingJobRecord:
		job = self._jobs.get(self._tenant_key(tenant_id, job_id))
		if job is None:
			raise KeyError(f"unknown audio job for tenant: {job_id}")
		return job

	def _replace_job_status(self, job: AudioProcessingJobRecord, status: str) -> AudioProcessingJobRecord:
		return AudioProcessingJobRecord(
			id=job.id,
			tenant_id=job.tenant_id,
			job_type=job.job_type,
			audio_source_id=job.audio_source_id,
			requested_by=job.requested_by,
			model_id=job.model_id,
			language_code=job.language_code,
			confidence=job.confidence,
			status=status,
			watermark_applied=job.watermark_applied,
			retention_policy=job.retention_policy,
			result=dict(job.result),
			created_at=job.created_at,
		)

	def _dump_tenant_records(self, records: dict[tuple[str, str], Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.model_dump(mode="json") for record in sorted(values, key=lambda item: item.id)]

	def _record_governance(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		evidence: dict[str, Any] | None = None,
	) -> None:
		self._governance_events.append(
			AudioGovernanceEvent(
				tenant_id=tenant_id,
				event_type=event_type,
				subject_id=subject_id,
				message=message,
				evidence=dict(evidence or {}),
			)
		)


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(action.get("reason", "audio_policy_blocked") for action in result["actions"])
	raise PermissionError(reasons or "audio_policy_blocked")


def _coerce_bool(value: Any) -> bool:
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)


def _normalize_agent_runtime(runtime: str) -> str:
	value = (runtime or "").strip().lower()
	if value not in SUPPORTED_AUDIO_AGENT_RUNTIMES:
		raise PermissionError("audio_agent_runtime_not_supported")
	return value


def _normalize_agent_role(role: str) -> str:
	value = (role or "").strip().lower()
	if value not in SUPPORTED_AUDIO_AGENT_ROLES:
		raise PermissionError("unsupported_audio_agent_role")
	return value


__all__ = ["AudpService"]
