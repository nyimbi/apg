"""Executable service layer for APG Scheduling and Job Orchestration."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import CalendarPolicy, JobDefinition, JobRun, ScheduleDefinition, SchdAuditEvent, WorkerPool
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
		self._audit_events: list[SchdAuditEvent] = []

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
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not queue:
			raise ValueError("worker_queue_required")
		if max_concurrency <= 0:
			raise ValueError("worker_capacity_must_be_positive")
		pool = WorkerPool(
			id=stable_id("wrk", tenant_id, name, queue),
			tenant_id=tenant_id,
			name=name,
			queue=queue,
			max_concurrency=max_concurrency,
			state=normalize_worker_state(state),
			autoscaling_enabled=autoscaling_enabled,
			created_at=utc_now(),
		)
		self._workers[pool.id] = pool
		self._record_event(tenant_id, "worker_pool_registered", pool.id, f"Worker pool {name} registered.", "system")
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
		enabled: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			self._raise_policy({"tenant_context_present": True, "operation": "create_schedule", "schedule_owner_assigned": False, "timezone_present": bool(timezone)})
		if not timezone:
			self._raise_policy({"tenant_context_present": True, "operation": "create_schedule", "schedule_owner_assigned": bool(owner), "timezone_present": False})
		job = self._require_owned(self._jobs, job_id, tenant_id, "job_not_found")
		self._require_owned(self._calendars, calendar_policy_id, tenant_id, "calendar_policy_not_found")
		self._require_owned(self._workers, worker_pool_id, tenant_id, "worker_pool_not_found")
		trigger_type = normalize_trigger_type(trigger_type)
		if trigger_type == "interval" and not interval_minutes:
			raise ValueError("interval_minutes_required")
		if trigger_type == "cron" and not cron:
			raise ValueError("cron_expression_required")
		if trigger_type == "manual" and not manual_reason:
			raise PermissionError("manual_run_reason_required")
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
			manual_reason=manual_reason,
			next_run_hint=next_run_hint(trigger_type, timezone, interval_minutes),
			state=schedule_state(enabled),
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._schedules[schedule.id] = schedule
		self._record_event(tenant_id, "schedule_created", schedule.id, f"Schedule {name} created.", owner)
		return schedule.to_dict()

	def trigger_run(self, tenant_id: str, schedule_id: str, requested_by: str, manual_reason: str | None = None) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		schedule = self._require_owned(self._schedules, schedule_id, tenant_id, "schedule_not_found")
		if not schedule.enabled or schedule.state != "active":
			raise PermissionError("schedule_not_runnable")
		if schedule.trigger_type == "manual" and not manual_reason:
			raise PermissionError("manual_run_reason_required")
		run = JobRun(
			id=stable_id("run", tenant_id, schedule_id, len(self._runs) + 1),
			tenant_id=tenant_id,
			schedule_id=schedule_id,
			job_id=schedule.job_id,
			worker_pool_id=schedule.worker_pool_id,
			requested_by=requested_by,
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
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		run = self._require_owned(self._runs, run_id, tenant_id, "run_not_found")
		if records_processed < 0 or error_count < 0:
			raise ValueError("run_counts_must_be_non_negative")
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
		run.completed_at = utc_now()
		self._record_event(tenant_id, "job_run_completed", run.id, f"Run completed with status {run.status}.", run.requested_by, "warning" if run.status != "succeeded" else "info")
		return run.to_dict()

	def disable_schedule(self, tenant_id: str, schedule_id: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		schedule = self._require_owned(self._schedules, schedule_id, tenant_id, "schedule_not_found")
		schedule.enabled = False
		schedule.state = schedule_state(False)
		schedule.updated_at = utc_now()
		self._record_event(tenant_id, "schedule_disabled", schedule.id, f"Schedule {schedule.name} disabled.", actor)
		return schedule.to_dict()

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
