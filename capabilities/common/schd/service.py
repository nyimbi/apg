"""Executable service layer for APG Scheduling and Job Orchestration."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	DEFAULT_CONFIGURATION,
	PRIVILEGED_SCHD_AGENT_ROLES,
	SUPPORTED_SCHD_AGENT_ROLES,
	SUPPORTED_SCHD_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import CalendarPolicy, JobDefinition, JobRun, ScheduleDefinition, SchdAuditEvent, SchdLifecycleBatch, SchedulerAgent, WorkerPool
from .scheduling_runtime import (
	backoff_seconds,
	next_run_hint,
	normalize_criticality,
	normalize_retry_strategy,
	normalize_tags,
	normalize_trigger_type,
	normalize_worker_state,
	run_status,
	schedule_state,
	stable_id,
	summarize_decision,
	utc_now,
)


class SchdService:
	"""Tenant-scoped calendar, worker, schedule, job, and run runtime."""

	def __init__(self) -> None:
		self._calendars: dict[str, CalendarPolicy] = {}
		self._workers: dict[str, WorkerPool] = {}
		self._jobs: dict[str, JobDefinition] = {}
		self._schedules: dict[str, ScheduleDefinition] = {}
		self._runs: dict[str, JobRun] = {}
		self._agents: dict[str, SchedulerAgent] = {}
		self._lifecycle_batches: dict[str, SchdLifecycleBatch] = {}
		self._audit_events: list[SchdAuditEvent] = []
		self._agent_runtimes = {self._normalize_token(value) for value in SUPPORTED_SCHD_AGENT_RUNTIMES}
		self._agent_roles = {self._normalize_token(value) for value in SUPPORTED_SCHD_AGENT_ROLES}
		self._privileged_agent_roles = {self._normalize_token(value) for value in PRIVILEGED_SCHD_AGENT_ROLES}
		self._lifecycle_operations = {self._normalize_token(value) for value in DEFAULT_CONFIGURATION["streaming"]["required_operations"]}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_calendar_policy(
		self,
		tenant_id: str,
		name: str,
		timezone: str,
		owner: str,
		business_days: list[str] | None = None,
		blackout_windows: list[str] | None = None,
		holiday_calendar: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not timezone:
			self._raise_policy({"tenant_context_present": True, "operation": "create_schedule", "timezone_present": False, "schedule_owner_assigned": bool(owner)})
		if not owner:
			self._raise_policy({"tenant_context_present": True, "operation": "create_schedule", "schedule_owner_assigned": False, "timezone_present": bool(timezone)})
		policy = CalendarPolicy(
			id=stable_id("cal", tenant_id, name, timezone),
			tenant_id=tenant_id,
			name=name,
			timezone=timezone,
			owner=owner,
			business_days=list(business_days or ["mon", "tue", "wed", "thu", "fri"]),
			blackout_windows=list(blackout_windows or []),
			holiday_calendar=holiday_calendar,
			created_at=utc_now(),
		)
		self._calendars[policy.id] = policy
		self._record_event(tenant_id, "calendar_policy_created", policy.id, f"Calendar policy {name} created.", owner)
		return policy.to_dict()

	def register_worker_pool(
		self,
		tenant_id: str,
		name: str,
		queue: str,
		max_concurrency: int,
		state: str = "ready",
		autoscaling_enabled: bool = False,
		heartbeat_ref: str = "local://worker-health",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "register_worker_pool",
			"worker_queue_present": bool(str(queue or "").strip()),
			"max_concurrency": int(max_concurrency),
			"health_check_attached": bool(str(heartbeat_ref or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		pool = WorkerPool(
			id=stable_id("wrk", tenant_id, name, queue),
			tenant_id=tenant_id,
			name=name,
			queue=queue,
			max_concurrency=max_concurrency,
			state=normalize_worker_state(state),
			autoscaling_enabled=autoscaling_enabled,
			heartbeat_ref=heartbeat_ref,
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._workers[pool.id] = pool
		self._record_event(tenant_id, "worker_pool_registered", pool.id, f"Worker pool {name} registered.", "system")
		return pool.to_dict()

	def change_worker_state(self, tenant_id: str, worker_pool_id: str, state: str, actor: str, reason: str = "") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		pool = self._require_owned(self._workers, worker_pool_id, tenant_id, "worker_pool_not_found")
		target_state = normalize_worker_state(state)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "change_worker_state",
			"target_worker_state": target_state,
			"state_change_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		pool.state = target_state
		pool.state_reason = reason
		pool.updated_at = utc_now()
		self._record_event(tenant_id, "worker_pool_state_changed", pool.id, f"Worker pool {pool.name} changed to {target_state}.", actor, "warning" if target_state != "ready" else "info")
		return pool.to_dict()

	def define_job(
		self,
		tenant_id: str,
		name: str,
		command: str,
		owner: str,
		criticality: str = "normal",
		expected_runtime_minutes: int = 30,
		external_job: bool = False,
		monitoring_attached: bool = False,
		approval_recorded: bool = False,
		runtime_review_recorded: bool = False,
		retry_strategy: str = "fixed",
		retry_policy_ref: str = "retry://default",
		max_attempts: int = 3,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("job_owner_required")
		if not command:
			raise ValueError("job_command_required")
		if expected_runtime_minutes <= 0:
			raise ValueError("expected_runtime_minutes_must_be_positive")
		if max_attempts <= 0:
			raise ValueError("max_attempts_must_be_positive")
		criticality = normalize_criticality(criticality)
		retry_strategy = normalize_retry_strategy(retry_strategy)
		policy_context = {
			"tenant_context_present": True,
			"job_criticality": criticality,
			"monitoring_attached": monitoring_attached,
			"external_job": external_job,
			"approval_recorded": approval_recorded,
			"expected_runtime_minutes": expected_runtime_minutes,
			"runtime_review_recorded": runtime_review_recorded,
			"operation": "define_job",
			"job_owner_assigned": bool(str(owner or "").strip()),
			"job_command_present": bool(str(command or "").strip()),
			"retry_policy_attached": bool(str(retry_policy_ref or "").strip()) and retry_strategy != "none",
		}
		result = self.evaluate(policy_context)
		if result["decision"] == "deny" or (result["decision"] == "require_review" and not runtime_review_recorded):
			raise PermissionError(summarize_decision(result))
		job = JobDefinition(
			id=stable_id("job", tenant_id, name, command),
			tenant_id=tenant_id,
			name=name,
			command=command,
			owner=owner,
			criticality=criticality,
			expected_runtime_minutes=expected_runtime_minutes,
			external_job=external_job,
			monitoring_attached=monitoring_attached,
			approval_recorded=approval_recorded,
			retry_strategy=retry_strategy,
			retry_policy_ref=retry_policy_ref,
			max_attempts=max_attempts,
			tags=normalize_tags(tags),
			created_at=utc_now(),
		)
		self._jobs[job.id] = job
		self._record_event(tenant_id, "job_defined", job.id, f"Job {name} defined.", owner)
		return job.to_dict()

	def create_schedule(
		self,
		tenant_id: str,
		name: str,
		job_id: str,
		calendar_policy_id: str,
		worker_pool_id: str,
		trigger_type: str,
		timezone: str,
		owner: str,
		interval_minutes: int | None = None,
		cron: str | None = None,
		manual_reason: str | None = None,
		event_policy_ref: str = "",
		enabled: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			self._raise_policy({"tenant_context_present": True, "operation": "create_schedule", "schedule_owner_assigned": False, "timezone_present": bool(timezone)})
		if not timezone:
			self._raise_policy({"tenant_context_present": True, "operation": "create_schedule", "schedule_owner_assigned": bool(owner), "timezone_present": False})
		job = self._require_owned(self._jobs, job_id, tenant_id, "job_not_found")
		self._require_owned(self._calendars, calendar_policy_id, tenant_id, "calendar_policy_not_found")
		worker = self._require_owned(self._workers, worker_pool_id, tenant_id, "worker_pool_not_found")
		trigger_type = normalize_trigger_type(trigger_type)
		if trigger_type == "interval" and not interval_minutes:
			raise ValueError("interval_minutes_required")
		if trigger_type == "cron" and not cron:
			raise ValueError("cron_expression_required")
		if trigger_type == "manual" and not manual_reason:
			raise PermissionError("manual_run_reason_required")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "create_schedule",
			"schedule_owner_assigned": bool(str(owner or "").strip()),
			"timezone_present": bool(str(timezone or "").strip()),
			"calendar_policy_present": bool(calendar_policy_id),
			"worker_pool_present": bool(worker_pool_id),
			"manual_trigger": trigger_type == "manual",
			"manual_reason_present": bool(manual_reason),
			"event_trigger": trigger_type == "event",
			"event_policy_attached": bool(str(event_policy_ref or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		if not job.enabled:
			raise PermissionError("job_not_runnable")
		if worker.state == "offline":
			raise PermissionError("worker_pool_not_ready")
		schedule = ScheduleDefinition(
			id=stable_id("sch", tenant_id, name, job.id, trigger_type),
			tenant_id=tenant_id,
			name=name,
			job_id=job.id,
			calendar_policy_id=calendar_policy_id,
			worker_pool_id=worker_pool_id,
			trigger_type=trigger_type,
			timezone=timezone,
			owner=owner,
			enabled=enabled,
			interval_minutes=interval_minutes,
			cron=cron,
			event_policy_ref=event_policy_ref,
			manual_reason=manual_reason,
			next_run_hint=next_run_hint(trigger_type, timezone, interval_minutes),
			state=schedule_state(enabled),
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._schedules[schedule.id] = schedule
		self._record_event(tenant_id, "schedule_created", schedule.id, f"Schedule {name} created.", owner)
		return schedule.to_dict()

	def trigger_run(
		self,
		tenant_id: str,
		schedule_id: str,
		requested_by: str,
		manual_reason: str | None = None,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		schedule = self._require_owned(self._schedules, schedule_id, tenant_id, "schedule_not_found")
		job = self._require_owned(self._jobs, schedule.job_id, tenant_id, "job_not_found")
		worker = self._require_owned(self._workers, schedule.worker_pool_id, tenant_id, "worker_pool_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "trigger_run",
			"schedule_active": schedule.enabled and schedule.state == "active",
			"worker_pool_ready": worker.state == "ready",
			"manual_trigger": schedule.trigger_type == "manual",
			"manual_reason_present": bool(str(manual_reason or schedule.manual_reason or "").strip()),
			"requested_by_present": bool(str(requested_by or "").strip()),
			"event_stream": event_stream,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		if not job.enabled:
			raise PermissionError("job_not_runnable")
		run = JobRun(
			id=stable_id("run", tenant_id, schedule_id, len(self._runs) + 1),
			tenant_id=tenant_id,
			schedule_id=schedule_id,
			job_id=schedule.job_id,
			worker_pool_id=schedule.worker_pool_id,
			requested_by=requested_by,
			event_stream=event_stream,
			status="running",
			started_at=utc_now(),
			logs=[f"Started schedule {schedule.name}."],
		)
		self._runs[run.id] = run
		self._record_event(tenant_id, "job_run_started", run.id, f"Run started for schedule {schedule.name}.", requested_by)
		return run.to_dict()

	def complete_run(
		self,
		tenant_id: str,
		run_id: str,
		records_processed: int = 0,
		error_count: int = 0,
		exit_code: int = 0,
		logs: list[str] | None = None,
		blocked_count: int = 0,
		completion_evidence_ref: str = "run://local-evidence",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		run = self._require_owned(self._runs, run_id, tenant_id, "run_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "complete_run",
			"run_counts_valid": records_processed >= 0 and error_count >= 0 and blocked_count >= 0,
			"audit_event_recorded": bool(str(completion_evidence_ref or "").strip()),
			"state_change_requested": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		job = self._require_owned(self._jobs, run.job_id, tenant_id, "job_not_found")
		run.records_processed = records_processed
		run.error_count = error_count
		run.exit_code = exit_code
		run.status = run_status(exit_code == 0 and error_count == 0, exit_code != 0 or error_count > 0, blocked_count)
		if run.status == "failed" and run.attempt >= job.max_attempts and job.dead_letter_enabled:
			run.status = "dead_lettered"
		if run.status == "failed":
			run.next_retry_seconds = backoff_seconds(job.retry_strategy, run.attempt)
		run.logs.extend(logs or [])
		run.completion_evidence_ref = completion_evidence_ref
		run.completed_at = utc_now()
		self._record_event(tenant_id, "job_run_completed", run.id, f"Run completed with status {run.status}.", run.requested_by, "warning" if run.status != "succeeded" else "info")
		return run.to_dict()

	def retry_run(self, tenant_id: str, run_id: str, requested_by: str, reason: str = "") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		run = self._require_owned(self._runs, run_id, tenant_id, "run_not_found")
		job = self._require_owned(self._jobs, run.job_id, tenant_id, "job_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "retry_run",
			"run_retryable": run.status in {"failed", "dead_lettered"} and run.attempt < job.max_attempts,
			"state_change_requested": True,
			"audit_event_recorded": bool(str(reason or "").strip()),
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		retry = JobRun(
			id=stable_id("run", tenant_id, run.schedule_id, len(self._runs) + 1),
			tenant_id=tenant_id,
			schedule_id=run.schedule_id,
			job_id=run.job_id,
			worker_pool_id=run.worker_pool_id,
			requested_by=requested_by,
			event_stream=run.event_stream,
			status="running",
			attempt=run.attempt + 1,
			parent_run_id=run.id,
			started_at=utc_now(),
			logs=[f"Retrying run {run.id}: {reason}"],
		)
		self._runs[retry.id] = retry
		self._record_event(tenant_id, "job_run_retried", retry.id, f"Retry started for run {run.id}.", requested_by)
		return retry.to_dict()

	def dead_letter_run(self, tenant_id: str, run_id: str, actor: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		run = self._require_owned(self._runs, run_id, tenant_id, "run_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "dead_letter_run",
			"dead_letter_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		run.status = "dead_lettered"
		run.dead_letter_reason = reason
		run.completed_at = utc_now()
		self._record_event(tenant_id, "job_run_dead_lettered", run.id, f"Run dead-lettered: {reason}", actor, "high")
		return run.to_dict()

	def cancel_run(self, tenant_id: str, run_id: str, actor: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		run = self._require_owned(self._runs, run_id, tenant_id, "run_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "cancel_run",
			"cancel_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		run.status = "cancelled"
		run.cancel_reason = reason
		run.completed_at = utc_now()
		self._record_event(tenant_id, "job_run_cancelled", run.id, f"Run cancelled: {reason}", actor, "warning")
		return run.to_dict()

	def pause_schedule(self, tenant_id: str, schedule_id: str, actor: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		schedule = self._require_owned(self._schedules, schedule_id, tenant_id, "schedule_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "pause_schedule",
			"pause_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		schedule.enabled = True
		schedule.state = schedule_state(True, paused=True)
		schedule.state_reason = reason
		schedule.updated_at = utc_now()
		self._record_event(tenant_id, "schedule_paused", schedule.id, f"Schedule {schedule.name} paused: {reason}", actor, "warning")
		return schedule.to_dict()

	def resume_schedule(self, tenant_id: str, schedule_id: str, actor: str, reason: str = "resume approved") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		schedule = self._require_owned(self._schedules, schedule_id, tenant_id, "schedule_not_found")
		result = self.evaluate({"tenant_context_present": True, "state_change_requested": True, "audit_event_recorded": bool(reason)})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		schedule.enabled = True
		schedule.state = schedule_state(True)
		schedule.state_reason = reason
		schedule.updated_at = utc_now()
		self._record_event(tenant_id, "schedule_resumed", schedule.id, f"Schedule {schedule.name} resumed.", actor)
		return schedule.to_dict()

	def disable_schedule(self, tenant_id: str, schedule_id: str, actor: str, reason: str = "disabled by operator") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		schedule = self._require_owned(self._schedules, schedule_id, tenant_id, "schedule_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "disable_schedule",
			"disable_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		schedule.enabled = False
		schedule.state = schedule_state(False)
		schedule.state_reason = reason
		schedule.updated_at = utc_now()
		self._record_event(tenant_id, "schedule_disabled", schedule.id, f"Schedule {schedule.name} disabled.", actor)
		return schedule.to_dict()

	def register_scheduler_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope_ref: str,
		registered_by: str,
		contribution_disclosed: bool,
		owner_ref: str = "",
		purpose: str = "",
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		owner_value = str(owner_ref or "").strip()
		purpose_value = str(purpose or "").strip()
		approval_recorded = self._coerce_bool(human_approval_required)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "register_scheduler_agent",
			"agent_id_present": bool(str(agent_id or "").strip()),
			"agent_name_present": bool(str(name or "").strip()),
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"agent_scope_present": bool(str(scope_ref or "").strip()),
			"agent_owner_present": bool(owner_value),
			"agent_purpose_present": bool(purpose_value),
			"agent_contribution_disclosed": self._coerce_bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": approval_recorded,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		if not self._scope_exists_for_tenant(tenant_id, scope_ref):
			raise KeyError("scheduler_agent_scope_not_found")
		agent = SchedulerAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope_ref=scope_ref,
			registered_by=registered_by,
			contribution_disclosed=self._coerce_bool(contribution_disclosed),
			owner_ref=owner_value,
			purpose=purpose_value,
			human_approval_required=approval_recorded,
			status="pending_review" if result["decision"] == "require_review" else "active",
			created_at=utc_now(),
		)
		self._agents[self._tenant_key(tenant_id, agent.id)] = agent
		self._record_event(tenant_id, "scheduler_agent_registered", agent.id, f"Scheduler agent {name} registered.", registered_by)
		return agent.to_dict()

	def validate_batch_mutation(self, event_stream: str) -> dict[str, Any]:
		result = self.evaluate({"tenant_context_present": True, "operation": "batch_scheduler_mutation", "event_stream": event_stream})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		return result

	def validate_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "scheduler_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		operation_value = self._normalize_token(operation)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "validate_schd_lifecycle_batch",
			"event_stream": self._normalize_token(event_stream),
			"mutation_count": int(mutation_count),
			"lifecycle_operation_supported": operation_value in self._lifecycle_operations,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy_result(result)
		record_id = batch_id or stable_id("schd_lifecycle_batch", tenant_id, operation_value, len(self._lifecycle_batches))
		batch = SchdLifecycleBatch(
			id=record_id,
			tenant_id=tenant_id,
			event_stream=self._normalize_token(event_stream),
			operation=operation_value,
			mutation_count=int(mutation_count),
			status="accepted" if result["decision"] == "allow" else "review_required",
			matched_rules=list(result["matched_rules"]),
			required_actions=self._required_actions(result),
			created_at=utc_now(),
		)
		self._lifecycle_batches[self._tenant_key(tenant_id, record_id)] = batch
		self._record_event(tenant_id, "schd_lifecycle_batch_validated", batch.id, f"SCHD lifecycle batch {batch.status}: {operation_value}", "schd")
		return batch.to_dict()

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
		owner = metadata.get("owner", "system")
		calendar = self.create_calendar_policy(tenant_id, f"{record_id}-calendar", metadata.get("timezone", "UTC"), owner)
		worker = self.register_worker_pool(tenant_id, f"{record_id}-worker", metadata.get("queue", "default"), metadata.get("max_concurrency", 1))
		job = self.define_job(tenant_id, f"{record_id}-job", metadata.get("command", "python job.py"), owner, monitoring_attached=True)
		return self.create_schedule(
			tenant_id,
			record_id,
			job["id"],
			calendar["id"],
			worker["id"],
			metadata.get("trigger_type", "interval"),
			metadata.get("timezone", "UTC"),
			owner,
			interval_minutes=metadata.get("interval_minutes", 60),
			enabled=status == "active",
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_schedules(tenant_id)

	def list_calendars(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._calendars, tenant_id)

	def list_worker_pools(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._workers, tenant_id)

	def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._jobs, tenant_id)

	def list_schedules(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._schedules, tenant_id)

	def list_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._runs, tenant_id)

	def list_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		schedules = self.list_schedules(tenant_id)
		runs = self.list_runs(tenant_id)
		workers = self.list_worker_pools(tenant_id)
		return {
			"schedule_count": len(schedules),
			"active_schedule_count": sum(1 for item in schedules if item["state"] == "active"),
			"job_count": len(self.list_jobs(tenant_id)),
			"worker_pool_count": len(workers),
			"ready_worker_pool_count": sum(1 for item in workers if item["state"] == "ready"),
			"run_count": len(runs),
			"succeeded_run_count": sum(1 for item in runs if item["status"] == "succeeded"),
			"failed_run_count": sum(1 for item in runs if item["status"] in {"failed", "dead_lettered"}),
			"calendar_count": len(self.list_calendars(tenant_id)),
			"agent_count": len(self.list_agents(tenant_id)),
			"pending_agent_review_count": sum(1 for item in self.list_agents(tenant_id) if item["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": sum(1 for item in self.list_lifecycle_batches(tenant_id) if item["status"] not in {"accepted", "review_required"}),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not tenant_id:
			self._raise_policy({"tenant_context_present": False})

	def _tenant_key(self, tenant_id: str, record_id: str) -> str:
		return f"{str(tenant_id or '').strip()}::{str(record_id or '').strip()}"

	def _normalize_token(self, value: object) -> str:
		return str(value or "").strip().lower()

	def _coerce_bool(self, value: object) -> bool:
		if isinstance(value, bool):
			return value
		if value is None:
			return False
		if isinstance(value, str):
			return value.strip().lower() in {"1", "true", "yes", "y", "on"}
		return bool(value)

	def _required_actions(self, result: dict[str, Any]) -> list[str]:
		return [
			str(action["required_action"])
			for action in result.get("actions", [])
			if action.get("required_action")
		]

	def _require_owned(self, store: dict[str, Any], object_id: str, tenant_id: str, missing_reason: str) -> Any:
		item = store.get(object_id)
		if item is None or item.tenant_id != tenant_id:
			raise KeyError(missing_reason)
		return item

	def _scope_exists_for_tenant(self, tenant_id: str, scope_ref: str) -> bool:
		for store in (self._calendars, self._workers, self._jobs, self._schedules, self._runs, self._lifecycle_batches):
			item = store.get(scope_ref)
			if item is None:
				item = store.get(self._tenant_key(tenant_id, scope_ref))
			if item is not None:
				return item.tenant_id == tenant_id
		return False

	def _raise_policy(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		self._raise_policy_result(result)

	def _raise_policy_result(self, result: dict[str, Any]) -> None:
		raise PermissionError(summarize_decision(result))

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, actor: str, severity: str = "info") -> None:
		event = SchdAuditEvent(
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
