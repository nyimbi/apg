"""Executable service layer for APG Scraper/Data Harvesting."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_HARVEST_AGENT_ROLES,
	SUPPORTED_HARVEST_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .harvest_runtime import (
	classify_dlp_status,
	normalize_extractor_type,
	normalize_harvest_mode,
	normalize_source_type,
	normalize_tags,
	result_retention_until,
	run_status,
	stable_id,
	summarize_decision,
	utc_now,
)
from .models import ExtractorProfile, HarvestAgent, HarvestJob, HarvestResult, HarvestRun, HarvestSource, PipelineHandoff, ScrpAuditEvent


class ScrpService:
	"""Tenant-scoped source, extractor, harvest, result, and pipeline runtime."""

	def __init__(self) -> None:
		self._sources: dict[str, HarvestSource] = {}
		self._extractors: dict[str, ExtractorProfile] = {}
		self._jobs: dict[str, HarvestJob] = {}
		self._runs: dict[str, HarvestRun] = {}
		self._results: dict[str, HarvestResult] = {}
		self._handoffs: dict[str, PipelineHandoff] = {}
		self._agents: dict[str, HarvestAgent] = {}
		self._audit_events: list[ScrpAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_source(
		self,
		tenant_id: str,
		name: str,
		source_type: str,
		endpoint: str,
		owner: str,
		terms_evidence: str,
		credential_vault_ref: str,
		rate_limit_per_minute: int,
		robots_policy_attached: bool = True,
		pii_expected: bool = False,
		pii_policy_attached: bool = False,
		sensitive_source: bool = False,
		source_review_recorded: bool = False,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not endpoint:
			raise ValueError("source_endpoint_required")
		source_type = normalize_source_type(source_type)
		policy_context = {
			"tenant_context_present": True,
			"operation": "register_source",
			"source_owner_assigned": bool(owner),
			"terms_evidence_present": bool(terms_evidence),
			"credential_vault_present": bool(credential_vault_ref),
			"rate_limit_per_minute": int(rate_limit_per_minute),
			"robots_policy_attached": bool(robots_policy_attached),
			"pii_expected": pii_expected,
			"pii_policy_attached": pii_policy_attached,
			"sensitive_source": sensitive_source,
			"source_review_recorded": source_review_recorded,
		}
		result = self.evaluate(policy_context)
		if result["decision"] == "deny" or (result["decision"] == "require_review" and not source_review_recorded):
			raise PermissionError(summarize_decision(result))
		source = HarvestSource(
			id=stable_id("src", tenant_id, name, endpoint),
			tenant_id=tenant_id,
			name=name,
			source_type=source_type,
			owner=owner,
			endpoint=endpoint,
			terms_evidence=terms_evidence,
			credential_vault_ref=credential_vault_ref,
			rate_limit_per_minute=rate_limit_per_minute,
			robots_policy_attached=robots_policy_attached,
			pii_expected=pii_expected,
			pii_policy_attached=pii_policy_attached,
			sensitive_source=sensitive_source,
			source_review_recorded=source_review_recorded,
			tags=normalize_tags(tags),
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._sources[source.id] = source
		self._record_event(tenant_id, "source_registered", source.id, f"Source {name} registered.", owner)
		return source.to_dict()

	def create_extractor_profile(
		self,
		tenant_id: str,
		name: str,
		extractor_type: str,
		owner: str,
		schema: dict[str, Any],
		output_mapping: dict[str, str] | None = None,
		incremental_cursor_field: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("extractor_owner_required")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "create_extractor",
			"schema_present": bool(schema),
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		extractor = ExtractorProfile(
			id=stable_id("ext", tenant_id, name, extractor_type),
			tenant_id=tenant_id,
			name=name,
			extractor_type=normalize_extractor_type(extractor_type),
			owner=owner,
			schema=dict(schema),
			output_mapping=dict(output_mapping or {}),
			incremental_cursor_field=incremental_cursor_field,
			created_at=utc_now(),
		)
		self._extractors[extractor.id] = extractor
		self._record_event(tenant_id, "extractor_created", extractor.id, f"Extractor {name} created.", owner)
		return extractor.to_dict()

	def create_harvest_job(
		self,
		tenant_id: str,
		name: str,
		source_id: str,
		extractor_profile_id: str,
		owner: str,
		mode: str = "incremental",
		schedule_policy_attached: bool = True,
		pipeline_target: str | None = None,
		enabled: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("harvest_job_owner_required")
		self._require_owned(self._sources, source_id, tenant_id, "source_not_found")
		self._require_owned(self._extractors, extractor_profile_id, tenant_id, "extractor_not_found")
		mode = normalize_harvest_mode(mode)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "create_harvest_job",
			"pipeline_target_present": bool(pipeline_target),
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		job = HarvestJob(
			id=stable_id("job", tenant_id, name, source_id, extractor_profile_id),
			tenant_id=tenant_id,
			name=name,
			source_id=source_id,
			extractor_profile_id=extractor_profile_id,
			owner=owner,
			mode=mode,
			schedule_policy_attached=schedule_policy_attached,
			pipeline_target=pipeline_target,
			enabled=enabled,
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._jobs[job.id] = job
		self._record_event(tenant_id, "harvest_job_created", job.id, f"Harvest job {name} created.", owner)
		return job.to_dict()

	def run_harvest(self, tenant_id: str, job_id: str, requested_by: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		job = self._require_owned(self._jobs, job_id, tenant_id, "harvest_job_not_found")
		if not job.enabled:
			raise PermissionError("harvest_job_disabled")
		source = self._require_owned(self._sources, job.source_id, tenant_id, "source_not_found")
		self._require_owned(self._extractors, job.extractor_profile_id, tenant_id, "extractor_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "run_harvest",
			"schedule_policy_attached": job.schedule_policy_attached,
			"terms_evidence_present": bool(source.terms_evidence),
			"pii_expected": source.pii_expected,
			"pii_policy_attached": source.pii_policy_attached,
			"sensitive_source": source.sensitive_source,
			"source_review_recorded": source.source_review_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		run = HarvestRun(
			id=stable_id("run", tenant_id, job_id, len(self._runs) + 1),
			tenant_id=tenant_id,
			job_id=job_id,
			source_id=job.source_id,
			extractor_profile_id=job.extractor_profile_id,
			requested_by=requested_by,
			status="running",
			dlp_status=classify_dlp_status(source.pii_expected, False),
			started_at=utc_now(),
			logs=[f"Started harvest job {job.name} for source {source.name}."],
		)
		self._runs[run.id] = run
		self._record_event(tenant_id, "harvest_run_started", run.id, f"Harvest run started for {job.name}.", requested_by)
		return run.to_dict()

	def complete_harvest_run(
		self,
		tenant_id: str,
		run_id: str,
		records_extracted: int,
		error_count: int = 0,
		dlp_scanned: bool = True,
		dlp_violations: int = 0,
		schema_valid: bool = True,
		storage_ref: str | None = None,
		logs: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		run = self._require_owned(self._runs, run_id, tenant_id, "harvest_run_not_found")
		source = self._require_owned(self._sources, run.source_id, tenant_id, "source_not_found")
		job = self._require_owned(self._jobs, run.job_id, tenant_id, "harvest_job_not_found")
		if records_extracted < 0 or error_count < 0 or dlp_violations < 0:
			raise ValueError("harvest_counts_must_be_non_negative")
		result_policy = self.evaluate({
			"tenant_context_present": True,
			"operation": "complete_harvest_run",
			"pii_expected": source.pii_expected,
			"dlp_scanned": bool(dlp_scanned),
		})
		if result_policy["decision"] != "allow":
			raise PermissionError(summarize_decision(result_policy))
		run.records_extracted = records_extracted
		run.error_count = error_count
		run.dlp_status = classify_dlp_status(source.pii_expected, dlp_scanned, dlp_violations)
		run.dlp_violations = dlp_violations
		run.status = run_status(records_extracted, error_count, run.dlp_status == "failed")
		run.logs.extend(logs or [])
		run.completed_at = utc_now()
		result = HarvestResult(
			id=stable_id("res", tenant_id, run.id, records_extracted),
			tenant_id=tenant_id,
			run_id=run.id,
			record_count=records_extracted,
			schema_valid=schema_valid,
			retention_until=result_retention_until(get_capability_contract(tenant_id)["configuration"]["extraction"]["result_retention_days"]),
			storage_ref=storage_ref or f"memory://{run.id}",
			created_at=utc_now(),
		)
		self._results[result.id] = result
		if job.pipeline_target and run.status == "succeeded":
			handoff = PipelineHandoff(
				id=stable_id("pipe", tenant_id, result.id, job.pipeline_target),
				tenant_id=tenant_id,
				result_id=result.id,
				pipeline_target=job.pipeline_target,
				status="queued",
				created_at=utc_now(),
			)
			self._handoffs[handoff.id] = handoff
		self._record_event(tenant_id, "harvest_run_completed", run.id, f"Harvest completed with status {run.status}.", run.requested_by, "warning" if run.status != "succeeded" else "info")
		return run.to_dict() | {"result": result.to_dict()}

	def register_harvest_agent(
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
		normalized_runtime = _normalize_harvest_agent_runtime(runtime)
		normalized_role = _normalize_harvest_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": True,
			"harvest_agent_present": True,
			"agent_registered": bool(registered),
			"agent_runtime_supported": bool(normalized_runtime),
			"agent_role_supported": bool(normalized_role),
			"agent_scope_present": bool(scope.strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		agent_key = stable_id("agent", tenant_id, agent_id)
		if agent_key in self._agents:
			raise ValueError("harvest_agent_already_registered")
		agent = HarvestAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name or agent_id,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref or None,
			created_at=utc_now(),
		)
		self._agents[agent_key] = agent
		self._record_event(tenant_id, "harvest_agent_registered", agent.id, f"Harvest agent {agent.name} registered.", agent.name)
		return agent.to_dict()

	def change_harvest_job_state(
		self,
		tenant_id: str,
		job_id: str,
		enabled: bool,
		reason: str,
		audit_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		job = self._require_owned(self._jobs, job_id, tenant_id, "harvest_job_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"state_change_requested": True,
			"state_change_reason_present": bool(reason.strip()),
			"audit_event_recorded": bool(audit_recorded),
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		job.enabled = bool(enabled)
		job.updated_at = utc_now()
		self._record_event(tenant_id, "harvest_job_state_changed", job.id, reason, job.owner)
		return job.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility shim for package tooling that expects create_record."""
		self._require_tenant(tenant_id)
		metadata = metadata or {}
		return self.register_source(
			tenant_id=tenant_id,
			name=record_id,
			source_type=metadata.get("source_type", "api"),
			endpoint=metadata.get("endpoint", f"https://example.invalid/{record_id}"),
			owner=metadata.get("owner", "system"),
			terms_evidence=metadata.get("terms_evidence", "internal_authorization"),
			credential_vault_ref=metadata.get("credential_vault_ref", "vault://scrp/default"),
			rate_limit_per_minute=metadata.get("rate_limit_per_minute", 60),
			pii_expected=metadata.get("pii_expected", False),
			pii_policy_attached=metadata.get("pii_policy_attached", metadata.get("pii_expected", False)),
			source_review_recorded=status == "active",
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_sources(tenant_id)

	def list_sources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sources, tenant_id)

	def list_extractors(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._extractors, tenant_id)

	def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._jobs, tenant_id)

	def list_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._runs, tenant_id)

	def list_results(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._results, tenant_id)

	def list_handoffs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._handoffs, tenant_id)

	def list_harvest_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		sources = self.list_sources(tenant_id)
		runs = self.list_runs(tenant_id)
		return {
			"source_count": len(sources),
			"sensitive_source_count": sum(1 for item in sources if item["sensitive_source"]),
			"extractor_count": len(self.list_extractors(tenant_id)),
			"job_count": len(self.list_jobs(tenant_id)),
			"run_count": len(runs),
			"succeeded_run_count": sum(1 for item in runs if item["status"] == "succeeded"),
			"blocked_run_count": sum(1 for item in runs if item["status"] == "blocked"),
			"result_count": len(self.list_results(tenant_id)),
			"pipeline_handoff_count": len(self.list_handoffs(tenant_id)),
			"agent_count": len(self.list_harvest_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not tenant_id:
			self._raise_policy({"tenant_context_present": False})

	def _require_owned(self, store: dict[str, Any], object_id: str, tenant_id: str, missing_reason: str) -> Any:
		item = store.get(object_id)
		if item is None or item.tenant_id != tenant_id:
			raise KeyError(missing_reason)
		return item

	def _raise_policy(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		raise PermissionError(summarize_decision(result))

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, actor: str, severity: str = "info") -> None:
		event = ScrpAuditEvent(
			id=stable_id("evt", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			created_at=utc_now(),
		)
		self._audit_events.append(event)

	def _list(self, store: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = store.values()
		if tenant_id is not None:
			values = [value for value in values if value.tenant_id == tenant_id]
		return [value.to_dict() for value in values]


def _normalize_harvest_agent_runtime(runtime: str) -> str:
	value = (runtime or "").strip().lower()
	return value if value in SUPPORTED_HARVEST_AGENT_RUNTIMES else ""


def _normalize_harvest_agent_role(role: str) -> str:
	value = (role or "").strip().lower()
	return value if value in SUPPORTED_HARVEST_AGENT_ROLES else ""
