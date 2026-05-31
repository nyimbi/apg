"""Dependency-light IMEX lifecycle runtime for generated APG applications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	PRIVILEGED_IMEX_AGENT_ROLES,
	SUPPORTED_IMEX_AGENT_ROLES,
	SUPPORTED_IMEX_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


@dataclass(frozen=True)
class TransferEndpoint:
	id: str
	tenant_id: str
	name: str
	endpoint_type: str
	conn_binding_ref: str
	owner: str
	external: bool = False
	approved: bool = True
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class MappingProfile:
	id: str
	tenant_id: str
	name: str
	source_profile_ref: str
	mapping_ref: str
	quality_gate_ref: str
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class TransferJob:
	id: str
	tenant_id: str
	name: str
	direction: str
	source_endpoint_id: str
	destination_endpoint_id: str
	format: str
	owner: str
	environment: str
	mapping_profile_id: str
	checksum: str
	data_classification: str = "internal"
	pii_detected: bool = False
	pii_policy_ref: str = ""
	etlp_plan_ref: str = ""
	status: str = "draft"
	preview_validated: bool = False
	quality_score: float | None = None
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class TransferRun:
	id: str
	tenant_id: str
	job_id: str
	status: str
	record_count: int
	checkpoint_ref: str
	quality_score: float | None = None
	records_processed: int = 0
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class TransferArtifact:
	id: str
	tenant_id: str
	run_id: str
	artifact_ref: str
	checksum: str
	retention_policy: str
	status: str = "published"
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class ReviewRecord:
	id: str
	tenant_id: str
	subject_id: str
	review_type: str
	status: str = "pending"
	requester: str = ""
	notes: str = ""
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class TransferAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class TransferAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	status: str
	contribution_disclosed: bool
	human_approval_required: bool
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class TransferLifecycleBatchRecord:
	id: str
	tenant_id: str
	event_stream: str
	operation: str
	mutation_count: int
	status: str
	created_at: str = field(default_factory=lambda: _now())


class ImexService:
	"""Tenant-scoped import/export lifecycle facade for generated APG apps."""

	def __init__(self) -> None:
		self._agent_runtimes = set(SUPPORTED_IMEX_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_IMEX_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_IMEX_AGENT_ROLES)
		self._endpoints: dict[tuple[str, str], TransferEndpoint] = {}
		self._mappings: dict[tuple[str, str], MappingProfile] = {}
		self._jobs: dict[tuple[str, str], TransferJob] = {}
		self._runs: dict[tuple[str, str], TransferRun] = {}
		self._artifacts: dict[tuple[str, str], TransferArtifact] = {}
		self._reviews: dict[tuple[str, str], ReviewRecord] = {}
		self._transfer_agents: dict[tuple[str, str], TransferAgentRecord] = {}
		self._lifecycle_batches: dict[tuple[str, str], TransferLifecycleBatchRecord] = {}
		self._events: list[TransferAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_endpoint(
		self,
		endpoint_id: str,
		tenant_id: str,
		name: str,
		endpoint_type: str,
		conn_binding_ref: str,
		owner: str,
		external: bool = False,
		approved: bool = True,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if not owner:
			raise PermissionError("endpoint_owner_required")
		if not conn_binding_ref:
			raise PermissionError("connector_binding_required")
		if self._tenant_key(tenant_id, endpoint_id) in self._endpoints:
			raise ValueError(f"endpoint already exists for tenant: {endpoint_id}")
		record = TransferEndpoint(endpoint_id, tenant_id, name, endpoint_type, conn_binding_ref, owner, external, approved)
		self._endpoints[self._tenant_key(tenant_id, endpoint_id)] = record
		self._record_event(tenant_id, "endpoint_registered", endpoint_id, f"Registered endpoint {name}.", {"endpoint_type": endpoint_type})
		return _dump(record)

	def create_mapping_profile(
		self,
		mapping_id: str,
		tenant_id: str,
		name: str,
		source_profile_ref: str,
		mapping_ref: str,
		quality_gate_ref: str,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if not source_profile_ref:
			raise PermissionError("source_profile_required")
		if not mapping_ref:
			raise PermissionError("schema_mapping_required")
		if not quality_gate_ref:
			raise PermissionError("quality_gate_required")
		record = MappingProfile(mapping_id, tenant_id, name, source_profile_ref, mapping_ref, quality_gate_ref)
		self._mappings[self._tenant_key(tenant_id, mapping_id)] = record
		self._record_event(tenant_id, "mapping_profile_created", mapping_id, f"Created mapping profile {name}.", {})
		return _dump(record)

	def create_job(
		self,
		job_id: str,
		tenant_id: str,
		name: str,
		direction: str,
		source_endpoint_id: str,
		destination_endpoint_id: str,
		format: str,
		owner: str,
		environment: str,
		mapping_profile_id: str,
		checksum: str,
		data_classification: str = "internal",
		pii_detected: bool = False,
		pii_policy_ref: str = "",
		etlp_plan_ref: str = "",
		destination_approved: bool | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		source = self._endpoints.get(self._tenant_key(tenant_id, source_endpoint_id))
		destination = self._endpoints.get(self._tenant_key(tenant_id, destination_endpoint_id))
		mapping = self._mappings.get(self._tenant_key(tenant_id, mapping_profile_id))
		config = get_capability_contract(tenant_id)["configuration"]
		destination_is_approved = destination.approved if destination_approved is None and destination is not None else bool(destination_approved)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_job",
			"owner_assigned": bool(owner),
			"direction_present": direction in {"import", "export", "migration"},
			"source_registered": source is not None,
			"destination_registered": destination is not None,
			"format_supported": format in config["formats"]["supported_formats"],
			"source_profile_present": mapping is not None and bool(mapping.source_profile_ref),
			"checksum_present": bool(checksum),
			"mapping_present": mapping is not None and bool(mapping.mapping_ref),
			"pii_detected": pii_detected,
			"pii_policy_attached": bool(pii_policy_ref),
			"external_destination": bool(destination.external) if destination else False,
			"destination_approved": destination_is_approved,
			"direction": direction,
			"etlp_plan_present": bool(etlp_plan_ref),
			"connector_binding_present": bool(source and source.conn_binding_ref and destination and destination.conn_binding_ref),
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, job_id) in self._jobs:
			raise ValueError(f"job already exists for tenant: {job_id}")
		status = "pending_review" if result["decision"] == "require_review" else "draft"
		record = TransferJob(job_id, tenant_id, name, direction, source_endpoint_id, destination_endpoint_id, format, owner, environment, mapping_profile_id, checksum, data_classification, pii_detected, pii_policy_ref, etlp_plan_ref, status)
		self._jobs[self._tenant_key(tenant_id, job_id)] = record
		self._record_event(tenant_id, "job_created", job_id, f"Created {direction} job {name}.", {"matched_rules": result["matched_rules"], "status": status})
		if result["decision"] == "require_review":
			self._create_review(tenant_id, f"destination:{job_id}", job_id, "destination", owner, "External destination approval required.")
		return _dump(record)

	def validate_preview(self, tenant_id: str, job_id: str, quality_score: float, invalid_records: int = 0) -> dict[str, Any]:
		job = self._get_job(tenant_id, job_id)
		record = self._replace_job(job, preview_validated=True, quality_score=quality_score, status="validated")
		self._jobs[self._tenant_key(tenant_id, job_id)] = record
		self._record_event(tenant_id, "preview_validated", job_id, f"Preview validation quality score {quality_score}.", {"invalid_records": invalid_records})
		return _dump(record)

	def execute_job(
		self,
		tenant_id: str,
		job_id: str,
		run_id: str,
		record_count: int,
		approval_recorded: bool = False,
		export_encrypted: bool = True,
		monitoring_enabled: bool = True,
		checkpointing_enabled: bool = True,
		quality_review_recorded: bool = False,
		invalid_records_present: bool = False,
		quarantine_enabled: bool = True,
		capacity_review_recorded: bool = False,
	) -> dict[str, Any]:
		job = self._get_job(tenant_id, job_id)
		active_jobs = len([run for run in self._runs.values() if run.tenant_id == tenant_id and run.status == "running"])
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "execute_job",
			"preview_validated": job.preview_validated,
			"environment": job.environment,
			"approval_recorded": approval_recorded,
			"direction": job.direction,
			"data_classification": job.data_classification,
			"export_encrypted": export_encrypted,
			"record_count": record_count,
			"monitoring_enabled": monitoring_enabled,
			"checkpointing_enabled": checkpointing_enabled,
			"quality_score": job.quality_score or 0.0,
			"quality_review_recorded": quality_review_recorded,
			"invalid_records_present": invalid_records_present,
			"quarantine_enabled": quarantine_enabled,
			"active_jobs": active_jobs,
			"capacity_review_recorded": capacity_review_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, run_id) in self._runs:
			raise ValueError(f"run already exists for tenant: {run_id}")
		status = "pending_review" if result["decision"] == "require_review" else "running"
		if result["decision"] == "require_review":
			self._create_review(tenant_id, f"quality:{run_id}", job_id, "quality", job.owner, "Transfer requires review before execution.")
		run = TransferRun(run_id, tenant_id, job_id, status, record_count, checkpoint_ref=f"checkpoint:{run_id}", quality_score=job.quality_score)
		self._runs[self._tenant_key(tenant_id, run_id)] = run
		self._jobs[self._tenant_key(tenant_id, job_id)] = self._replace_job(job, status=status)
		self._record_event(tenant_id, "job_execution_requested", run_id, f"Execution decision: {status}.", {"matched_rules": result["matched_rules"]})
		return _dump(run)

	def complete_run(self, tenant_id: str, run_id: str, records_processed: int, quality_score: float, audit_evidence_present: bool = True) -> dict[str, Any]:
		run = self._get_run(tenant_id, run_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "complete_run",
			"audit_evidence_present": audit_evidence_present,
			"quality_score_present": quality_score is not None,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if run.status != "running":
			raise PermissionError("transfer_run_not_running")
		record = TransferRun(run.id, run.tenant_id, run.job_id, "completed", run.record_count, run.checkpoint_ref, quality_score, records_processed, run.created_at)
		self._runs[self._tenant_key(tenant_id, run_id)] = record
		self._jobs[self._tenant_key(tenant_id, run.job_id)] = self._replace_job(self._get_job(tenant_id, run.job_id), status="completed")
		self._record_event(tenant_id, "run_completed", run_id, f"Completed transfer run {run_id}.", {"records_processed": records_processed, "quality_score": quality_score})
		return _dump(record)

	def publish_artifact(self, tenant_id: str, artifact_id: str, run_id: str, artifact_ref: str, checksum: str, retention_policy: str) -> dict[str, Any]:
		run = self._get_run(tenant_id, run_id)
		if run.status != "completed":
			raise PermissionError("transfer_run_not_completed")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_artifact",
			"checksum_present": bool(checksum),
			"retention_policy_present": bool(retention_policy),
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = TransferArtifact(artifact_id, tenant_id, run_id, artifact_ref, checksum, retention_policy)
		self._artifacts[self._tenant_key(tenant_id, artifact_id)] = record
		self._record_event(tenant_id, "artifact_published", artifact_id, f"Published transfer artifact {artifact_ref}.", {})
		return _dump(record)

	def replay_run(self, tenant_id: str, run_id: str, replay_id: str, idempotency_key: str) -> dict[str, Any]:
		run = self._get_run(tenant_id, run_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "replay_run", "idempotency_key_present": bool(idempotency_key)})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = TransferRun(replay_id, tenant_id, run.job_id, "queued", run.record_count, checkpoint_ref=f"replay:{idempotency_key}", quality_score=run.quality_score)
		self._runs[self._tenant_key(tenant_id, replay_id)] = record
		self._record_event(tenant_id, "run_replay_queued", replay_id, f"Queued replay for {run_id}.", {"idempotency_key": idempotency_key})
		return _dump(record)

	def purge_artifact(self, tenant_id: str, artifact_id: str, actor: str, purge_review_recorded: bool) -> dict[str, Any]:
		artifact = self._artifacts.get(self._tenant_key(tenant_id, artifact_id))
		if artifact is None:
			raise KeyError(f"unknown artifact for tenant: {artifact_id}")
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "purge_artifact", "destructive": True, "purge_review_recorded": purge_review_recorded})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = TransferArtifact(artifact.id, artifact.tenant_id, artifact.run_id, artifact.artifact_ref, artifact.checksum, artifact.retention_policy, "purged", artifact.created_at)
		self._artifacts[self._tenant_key(tenant_id, artifact_id)] = record
		self._record_event(tenant_id, "artifact_purged", artifact_id, f"Purged artifact {artifact_id}.", {"actor": actor})
		return _dump(record)

	def register_transfer_agent(
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
		"""Register a governed AI/automation participant for transfer work."""
		self._enforce_tenant(tenant_id)
		normalized_runtime = _normalize_agent_token(runtime)
		normalized_role = _normalize_agent_token(role)
		privileged_role = normalized_role in self._privileged_agent_roles
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_transfer_agent",
			"agent_runtime_supported": normalized_runtime in self._agent_runtimes,
			"agent_role_supported": normalized_role in self._agent_roles,
			"scope_present": bool(scope),
			"owner_present": bool(owner),
			"purpose_present": bool(purpose),
			"contribution_disclosed": contribution_disclosed,
			"privileged_role": privileged_role,
			"human_approval_required": human_approval_required,
		})
		if result["decision"] == "deny":
			self._record_event(tenant_id, "transfer_agent_registration_denied", agent_id, f"Denied transfer agent {name}.", {"matched_rules": result["matched_rules"]})
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, agent_id) in self._transfer_agents:
			raise ValueError(f"transfer agent already exists for tenant: {agent_id}")
		record = TransferAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			owner=owner,
			purpose=purpose,
			status=_status_for_decision(result),
			contribution_disclosed=contribution_disclosed,
			human_approval_required=human_approval_required,
		)
		self._transfer_agents[self._tenant_key(tenant_id, agent_id)] = record
		self._record_event(tenant_id, "transfer_agent_registered", agent_id, f"Registered transfer agent {name}.", {"matched_rules": result["matched_rules"], "status": record.status})
		return _dump(record)

	def validate_imex_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "transfer_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		"""Validate that import/export lifecycle mutations are processed by Bytewax."""
		self._enforce_tenant(tenant_id)
		normalized_stream = _normalize_agent_token(event_stream)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_imex_lifecycle_batch",
			"event_stream": normalized_stream,
		})
		resolved_batch_id = batch_id or f"{operation}:{len(self._lifecycle_batches) + 1}"
		record = TransferLifecycleBatchRecord(
			id=resolved_batch_id,
			tenant_id=tenant_id,
			event_stream=normalized_stream,
			operation=operation,
			mutation_count=mutation_count,
			status="denied" if result["decision"] == "deny" else "accepted",
		)
		self._lifecycle_batches[self._tenant_key(tenant_id, resolved_batch_id)] = record
		self._record_event(tenant_id, "imex_lifecycle_batch_validated", resolved_batch_id, f"Validated IMEX lifecycle batch through {normalized_stream}.", {"matched_rules": result["matched_rules"], "status": record.status})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		return _dump(record)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		jobs = [job for job in self._jobs.values() if job.tenant_id == tenant_id]
		runs = [run for run in self._runs.values() if run.tenant_id == tenant_id]
		lifecycle_batches = [batch for batch in self._lifecycle_batches.values() if batch.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"endpoint_count": len([endpoint for endpoint in self._endpoints.values() if endpoint.tenant_id == tenant_id]),
			"mapping_count": len([mapping for mapping in self._mappings.values() if mapping.tenant_id == tenant_id]),
			"job_count": len(jobs),
			"active_run_count": len([run for run in runs if run.status == "running"]),
			"completed_run_count": len([run for run in runs if run.status == "completed"]),
			"artifact_count": len([artifact for artifact in self._artifacts.values() if artifact.tenant_id == tenant_id]),
			"pending_review_count": len([review for review in self._reviews.values() if review.tenant_id == tenant_id and review.status == "pending"]),
			"transfer_agent_count": len([agent for agent in self._transfer_agents.values() if agent.tenant_id == tenant_id]),
			"lifecycle_batch_count": len(lifecycle_batches),
			"denied_lifecycle_batch_count": len([batch for batch in lifecycle_batches if batch.status == "denied"]),
			"audit_event_count": len([event for event in self._events if event.tenant_id == tenant_id]),
		}

	def list_endpoints(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._endpoints, tenant_id)

	def list_mappings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._mappings, tenant_id)

	def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._jobs, tenant_id)

	def list_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._runs, tenant_id)

	def list_artifacts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._artifacts, tenant_id)

	def list_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reviews, tenant_id)

	def list_transfer_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._transfer_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [_dump(event) for event in self._events if tenant_id is None or event.tenant_id == tenant_id]

	def _create_review(self, tenant_id: str, review_id: str, subject_id: str, review_type: str, requester: str, notes: str) -> ReviewRecord:
		review = ReviewRecord(review_id, tenant_id, subject_id, review_type, requester=requester, notes=notes)
		self._reviews[self._tenant_key(tenant_id, review_id)] = review
		self._record_event(tenant_id, f"{review_type}_review_requested", review_id, f"Requested {review_type} review.", {"subject_id": subject_id})
		return review

	def _replace_job(self, job: TransferJob, **changes: Any) -> TransferJob:
		values = _dump(job)
		values.update(changes)
		return TransferJob(**values)

	def _get_job(self, tenant_id: str, job_id: str) -> TransferJob:
		record = self._jobs.get(self._tenant_key(tenant_id, job_id))
		if record is None:
			raise KeyError(f"unknown job for tenant: {job_id}")
		return record

	def _get_run(self, tenant_id: str, run_id: str) -> TransferRun:
		record = self._runs.get(self._tenant_key(tenant_id, run_id))
		if record is None:
			raise KeyError(f"unknown transfer run for tenant: {run_id}")
		return record

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, evidence: dict[str, Any] | None = None) -> None:
		self._events.append(TransferAuditEvent(f"event:{len(self._events) + 1}", tenant_id, event_type, subject_id, message, dict(evidence or {})))

	def _list(self, store: dict[tuple[str, str], Any], tenant_id: str | None) -> list[dict[str, Any]]:
		return [_dump(record) for record in store.values() if tenant_id is None or record.tenant_id == tenant_id]

	def _enforce_tenant(self, tenant_id: str) -> None:
		if not tenant_id:
			_raise_if_blocked(self.evaluate({"tenant_context_present": False}))

	def _tenant_key(self, tenant_id: str, value: str) -> tuple[str, str]:
		return tenant_id, value


def _dump(record: Any) -> dict[str, Any]:
	return asdict(record)


def _raise_if_blocked(result: dict[str, Any]) -> None:
	reasons = [action.get("reason", "imex_guardrail_failed") for action in result.get("actions", [])]
	raise PermissionError(",".join(reasons) or "imex_guardrail_failed")


def _normalize_agent_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _status_for_decision(result: dict[str, Any]) -> str:
	return "pending_review" if result["decision"] == "require_review" else "active"


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()
