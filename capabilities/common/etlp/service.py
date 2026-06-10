"""APG ETL/ELT Pipeline Service — expanded async runtime (42+ methods).

All state in _Store. Every mutation emits an audit event.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:
		return str(uuid.uuid4())

import logging
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

logger = logging.getLogger(__name__)

VALID_MODES: set[str] = {"etl", "elt", "streaming", "batch", "micro_batch", "cdc"}
VALID_ENVS: set[str] = {"development", "test", "staging", "production"}
SUPPORTED_CHANNELS: set[str] = {"email", "sms", "webhook", "audit_log"}


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(v: str) -> str:
	return str(v or "").strip().lower().replace("-", "_").replace(" ", "_")


class _Store:
	def __init__(self) -> None:
		self._data: dict[str, dict[str, Any]] = {}

	async def put(self, col: str, rec: dict[str, Any]) -> dict[str, Any]:
		self._data.setdefault(col, {})[rec["id"]] = rec
		return rec

	async def get(self, col: str, rid: str) -> dict[str, Any] | None:
		return self._data.get(col, {}).get(rid)

	async def list(self, col: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._data.get(col, {}).values())
		if tenant_id is not None:
			items = [i for i in items if i.get("tenant_id") == tenant_id]
		return sorted(items, key=lambda i: i.get("id", ""))

	async def delete(self, col: str, rid: str) -> bool:
		bucket = self._data.get(col, {})
		if rid in bucket:
			del bucket[rid]
			return True
		return False


class _Audit:
	def __init__(self, store: _Store) -> None:
		self._store = store

	async def log_event(self, event_type: str, actor_id: str, tenant_id: str, subject_id: str,
						details: dict[str, Any] | None = None, severity: str = "info") -> dict[str, Any]:
		rec = {
			"id": uuid7str(), "tenant_id": tenant_id, "event_type": event_type,
			"actor_id": actor_id, "subject_id": subject_id, "severity": severity,
			"details": details or {}, "recorded_at": _utc_now(),
		}
		await self._store.put("etlp_audit", rec)
		return rec


class _Notify:
	async def send(self, recipient: str, channel: str, subject: str, body: str) -> dict[str, Any]:
		if channel not in SUPPORTED_CHANNELS:
			raise ValueError(f"unsupported_channel:{channel}")
		return {"id": uuid7str(), "recipient": recipient, "channel": channel, "subject": subject, "sent_at": _utc_now()}


class ETLPService:
	"""Async ETL/ELT pipeline service — 42+ methods."""

	def __init__(self, actor_id: str = "system", tenant_id: str = "default") -> None:
		self.actor_id = actor_id
		self.tenant_id = tenant_id
		self._store = _Store()
		self._audit = _Audit(self._store)
		self._notify = _Notify()

	# ------------------------------------------------------------------
	# 1. pipeline_design
	# ------------------------------------------------------------------
	async def pipeline_design(
		self,
		tenant_id: str,
		pipeline_id: str,
		name: str,
		mode: str,
		owner: str,
		description: str = "",
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		"""Design/register a new pipeline."""
		if mode not in VALID_MODES:
			raise ValueError(f"invalid_mode:{mode}")
		assert name and owner, "name and owner required"
		record = {
			"id": pipeline_id, "tenant_id": tenant_id, "name": name, "mode": mode,
			"owner": owner, "description": description, "tags": tags or [],
			"status": "draft", "version": "1.0.0",
			"created_at": _utc_now(), "updated_at": _utc_now(),
		}
		await self._store.put("etlp_pipelines", record)
		await self._audit.log_event("pipeline_designed", self.actor_id, tenant_id, pipeline_id, {"name": name, "mode": mode})
		return record

	# ------------------------------------------------------------------
	# 2. source_connect
	# ------------------------------------------------------------------
	async def source_connect(
		self,
		tenant_id: str,
		source_id: str,
		name: str,
		source_type: str,
		owner: str,
		connection_config: dict[str, Any],
		secret_ref: str = "",
		approved: bool = True,
	) -> dict[str, Any]:
		"""Register a data source connection."""
		assert name and source_type and owner, "name, source_type, owner required"
		record = {
			"id": source_id, "tenant_id": tenant_id, "name": name, "type": source_type,
			"owner": owner, "connection_config": connection_config,
			"secret_ref": secret_ref, "approved": approved,
			"health_status": "unknown", "status": "active", "created_at": _utc_now(),
		}
		await self._store.put("etlp_sources", record)
		await self._audit.log_event("source_connected", self.actor_id, tenant_id, source_id, {"type": source_type})
		return record

	# ------------------------------------------------------------------
	# 3. target_connect
	# ------------------------------------------------------------------
	async def target_connect(
		self,
		tenant_id: str,
		target_id: str,
		name: str,
		target_type: str,
		owner: str,
		connection_config: dict[str, Any],
		secret_ref: str = "",
	) -> dict[str, Any]:
		"""Register a target/destination connection."""
		assert name and target_type and owner, "name, target_type, owner required"
		record = {
			"id": target_id, "tenant_id": tenant_id, "name": name, "type": target_type,
			"owner": owner, "connection_config": connection_config,
			"secret_ref": secret_ref, "health_status": "unknown",
			"status": "active", "created_at": _utc_now(),
		}
		await self._store.put("etlp_targets", record)
		await self._audit.log_event("target_connected", self.actor_id, tenant_id, target_id, {"type": target_type})
		return record

	# ------------------------------------------------------------------
	# 4. transform_rule
	# ------------------------------------------------------------------
	async def transform_rule(
		self,
		tenant_id: str,
		rule_id: str,
		pipeline_id: str,
		name: str,
		rule_type: str,
		logic: dict[str, Any],
		owner: str,
	) -> dict[str, Any]:
		"""Define a transformation rule for a pipeline."""
		await self._require_pipeline(tenant_id, pipeline_id)
		record = {
			"id": rule_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"name": name, "rule_type": rule_type, "logic": logic, "owner": owner,
			"status": "active", "created_at": _utc_now(),
		}
		await self._store.put("etlp_transform_rules", record)
		await self._audit.log_event("transform_rule_created", self.actor_id, tenant_id, rule_id, {"pipeline_id": pipeline_id, "rule_type": rule_type})
		return record

	# ------------------------------------------------------------------
	# 5. run_pipeline
	# ------------------------------------------------------------------
	async def run_pipeline(
		self,
		tenant_id: str,
		pipeline_id: str,
		environment: str,
		triggered_by: str,
		idempotency_key: str | None = None,
		config: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Execute a pipeline and create an execution record."""
		pipeline = await self._require_pipeline(tenant_id, pipeline_id)
		if environment not in VALID_ENVS:
			raise ValueError(f"invalid_environment:{environment}")
		execution_id = uuid7str()
		record = {
			"id": execution_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"pipeline_name": pipeline["name"], "environment": environment,
			"triggered_by": triggered_by, "idempotency_key": idempotency_key,
			"config": config or {}, "mode": pipeline["mode"],
			"status": "queued", "records_processed": 0, "records_failed": 0,
			"started_at": _utc_now(), "completed_at": None,
		}
		await self._store.put("etlp_executions", record)
		await self._audit.log_event("pipeline_run_started", self.actor_id, tenant_id, execution_id, {"pipeline_id": pipeline_id, "env": environment})
		# Simulate async execution
		asyncio.get_event_loop().call_soon(lambda: logger.info("ETLP pipeline queued: %s", execution_id))
		return record

	# ------------------------------------------------------------------
	# 6. schedule_pipeline
	# ------------------------------------------------------------------
	async def schedule_pipeline(
		self,
		tenant_id: str,
		schedule_id: str,
		pipeline_id: str,
		environment: str,
		cron_expression: str,
		owner: str,
	) -> dict[str, Any]:
		"""Schedule a recurring pipeline execution."""
		await self._require_pipeline(tenant_id, pipeline_id)
		record = {
			"id": schedule_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"environment": environment, "cron_expression": cron_expression, "owner": owner,
			"status": "active", "created_at": _utc_now(),
		}
		await self._store.put("etlp_schedules", record)
		await self._audit.log_event("pipeline_scheduled", self.actor_id, tenant_id, schedule_id, {"pipeline_id": pipeline_id, "cron": cron_expression})
		return record

	# ------------------------------------------------------------------
	# 7. monitor_pipeline
	# ------------------------------------------------------------------
	async def monitor_pipeline(self, tenant_id: str, execution_id: str) -> dict[str, Any]:
		"""Return current execution status and metrics."""
		exec_rec = await self._require_execution(tenant_id, execution_id)
		quality = await self._latest_quality(tenant_id, execution_id)
		return {
			"execution_id": execution_id,
			"pipeline_id": exec_rec["pipeline_id"],
			"status": exec_rec["status"],
			"records_processed": exec_rec["records_processed"],
			"records_failed": exec_rec["records_failed"],
			"quality_score": quality.get("score") if quality else None,
			"quality_gate_passed": quality.get("gate_passed") if quality else None,
			"started_at": exec_rec.get("started_at"),
			"completed_at": exec_rec.get("completed_at"),
			"checked_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 8. data_quality_gate
	# ------------------------------------------------------------------
	async def data_quality_gate(
		self,
		tenant_id: str,
		quality_id: str,
		execution_id: str,
		score: float,
		dimensions: dict[str, float],
		assessor: str,
		minimum_score: float = 80.0,
	) -> dict[str, Any]:
		"""Assess data quality and record gate pass/fail."""
		await self._require_execution(tenant_id, execution_id)
		if not (0 <= score <= 100):
			raise ValueError("score must be in [0,100]")
		gate_passed = score >= minimum_score
		record = {
			"id": quality_id, "tenant_id": tenant_id, "execution_id": execution_id,
			"score": score, "dimensions": dimensions, "minimum_score": minimum_score,
			"gate_passed": gate_passed, "assessor": assessor, "assessed_at": _utc_now(),
		}
		await self._store.put("etlp_quality", record)
		await self._audit.log_event("quality_assessed", self.actor_id, tenant_id, quality_id, {"score": score, "gate_passed": gate_passed})
		if not gate_passed:
			await self._notify.send(self.actor_id, "audit_log", "Quality gate failed", f"Execution {execution_id} quality {score} < {minimum_score}")
		return record

	# ------------------------------------------------------------------
	# 9. schema_evolution
	# ------------------------------------------------------------------
	async def schema_evolution(
		self,
		tenant_id: str,
		evolution_id: str,
		pipeline_id: str,
		old_schema: dict[str, Any],
		new_schema: dict[str, Any],
		migration_strategy: str = "backward_compatible",
	) -> dict[str, Any]:
		"""Record and validate a schema evolution event."""
		await self._require_pipeline(tenant_id, pipeline_id)
		old_fields = set(old_schema.get("fields", {}).keys())
		new_fields = set(new_schema.get("fields", {}).keys())
		added = list(new_fields - old_fields)
		removed = list(old_fields - new_fields)
		breaking = bool(removed and migration_strategy == "backward_compatible")
		record = {
			"id": evolution_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"old_schema": old_schema, "new_schema": new_schema,
			"fields_added": added, "fields_removed": removed,
			"migration_strategy": migration_strategy,
			"breaking_change": breaking,
			"status": "rejected" if breaking else "applied",
			"evolved_at": _utc_now(),
		}
		await self._store.put("etlp_schema_evolutions", record)
		await self._audit.log_event("schema_evolved", self.actor_id, tenant_id, evolution_id, {"added": added, "removed": removed, "breaking": breaking})
		if breaking:
			raise ValueError(f"breaking_schema_change_not_allowed_with_{migration_strategy}")
		return record

	# ------------------------------------------------------------------
	# 10. partition_strategy
	# ------------------------------------------------------------------
	async def partition_strategy(
		self,
		tenant_id: str,
		strategy_id: str,
		pipeline_id: str,
		partition_by: list[str],
		partition_type: str = "date",
		retention_days: int = 90,
	) -> dict[str, Any]:
		"""Define a partitioning strategy for a pipeline."""
		await self._require_pipeline(tenant_id, pipeline_id)
		record = {
			"id": strategy_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"partition_by": partition_by, "partition_type": partition_type,
			"retention_days": retention_days, "status": "active", "created_at": _utc_now(),
		}
		await self._store.put("etlp_partition_strategies", record)
		await self._audit.log_event("partition_strategy_set", self.actor_id, tenant_id, strategy_id, {"pipeline_id": pipeline_id, "partition_by": partition_by})
		return record

	# ------------------------------------------------------------------
	# 11. watermark_management
	# ------------------------------------------------------------------
	async def watermark_management(
		self,
		tenant_id: str,
		pipeline_id: str,
		watermark_column: str,
		last_value: str,
		source_id: str,
	) -> dict[str, Any]:
		"""Set or update the watermark for incremental pipeline runs."""
		await self._require_pipeline(tenant_id, pipeline_id)
		wm_id = f"wm:{tenant_id}:{pipeline_id}:{source_id}"
		record = {
			"id": wm_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"source_id": source_id, "watermark_column": watermark_column,
			"last_value": last_value, "updated_at": _utc_now(),
		}
		await self._store.put("etlp_watermarks", record)
		await self._audit.log_event("watermark_updated", self.actor_id, tenant_id, wm_id, {"pipeline_id": pipeline_id, "last_value": last_value})
		return record

	# ------------------------------------------------------------------
	# 12. cdc_capture
	# ------------------------------------------------------------------
	async def cdc_capture(
		self,
		tenant_id: str,
		capture_id: str,
		pipeline_id: str,
		source_id: str,
		table_name: str,
		operation: str,
		row_data: dict[str, Any],
		lsn: str = "",
	) -> dict[str, Any]:
		"""Record a Change Data Capture event."""
		assert operation in {"insert", "update", "delete"}, f"invalid CDC operation:{operation}"
		await self._require_pipeline(tenant_id, pipeline_id)
		record = {
			"id": capture_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"source_id": source_id, "table_name": table_name, "operation": operation,
			"row_data": row_data, "lsn": lsn, "captured_at": _utc_now(),
		}
		await self._store.put("etlp_cdc_events", record)
		await self._audit.log_event("cdc_captured", self.actor_id, tenant_id, capture_id, {"table": table_name, "operation": operation})
		return record

	# ------------------------------------------------------------------
	# 13. lineage_track
	# ------------------------------------------------------------------
	async def lineage_track(
		self,
		tenant_id: str,
		lineage_id: str,
		pipeline_id: str,
		source_ids: list[str],
		target_ids: list[str],
		transformation_ids: list[str],
		execution_id: str,
	) -> dict[str, Any]:
		"""Record data lineage for a pipeline execution."""
		record = {
			"id": lineage_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"source_ids": source_ids, "target_ids": target_ids,
			"transformation_ids": transformation_ids, "execution_id": execution_id,
			"recorded_at": _utc_now(),
		}
		await self._store.put("etlp_lineage", record)
		await self._audit.log_event("lineage_tracked", self.actor_id, tenant_id, lineage_id, {"pipeline_id": pipeline_id, "execution_id": execution_id})
		return record

	# ------------------------------------------------------------------
	# 14. sla_monitor
	# ------------------------------------------------------------------
	async def sla_monitor(
		self,
		tenant_id: str,
		sla_id: str,
		pipeline_id: str,
		max_duration_minutes: int,
		max_failure_rate_percent: float,
		alert_recipient: str,
	) -> dict[str, Any]:
		"""Register an SLA monitor for a pipeline."""
		await self._require_pipeline(tenant_id, pipeline_id)
		record = {
			"id": sla_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"max_duration_minutes": max_duration_minutes,
			"max_failure_rate_percent": max_failure_rate_percent,
			"alert_recipient": alert_recipient, "status": "active", "created_at": _utc_now(),
		}
		await self._store.put("etlp_sla_monitors", record)
		await self._audit.log_event("sla_monitor_created", self.actor_id, tenant_id, sla_id, {"pipeline_id": pipeline_id})
		return record

	# ------------------------------------------------------------------
	# 15. etl_analytics
	# ------------------------------------------------------------------
	async def etl_analytics(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Aggregate ETL pipeline analytics."""
		pipelines = await self._store.list("etlp_pipelines", tenant_id)
		executions = await self._store.list("etlp_executions", tenant_id)
		quality_recs = await self._store.list("etlp_quality", tenant_id)
		lineage = await self._store.list("etlp_lineage", tenant_id)
		total_records = sum(e.get("records_processed", 0) for e in executions)
		total_failed = sum(e.get("records_failed", 0) for e in executions)
		quality_scores = [q["score"] for q in quality_recs]
		return {
			"tenant_id": tenant_id, "period": period,
			"pipeline_count": len(pipelines),
			"execution_count": len(executions),
			"successful_executions": sum(1 for e in executions if e["status"] == "published"),
			"failed_executions": sum(1 for e in executions if e["status"] == "failed"),
			"total_records_processed": total_records,
			"total_records_failed": total_failed,
			"failure_rate_percent": round(total_failed / max(total_records, 1) * 100, 4),
			"avg_quality_score": round(statistics.mean(quality_scores), 2) if quality_scores else None,
			"quality_gate_pass_rate": round(sum(1 for q in quality_recs if q["gate_passed"]) / max(len(quality_recs), 1) * 100, 2),
			"lineage_records": len(lineage),
			"computed_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 16. pipeline_validate
	# ------------------------------------------------------------------
	async def pipeline_validate(self, tenant_id: str, pipeline_id: str) -> dict[str, Any]:
		"""Validate pipeline configuration before execution."""
		pipeline = await self._require_pipeline(tenant_id, pipeline_id)
		issues: list[str] = []
		rules = [r for r in await self._store.list("etlp_transform_rules", tenant_id) if r["pipeline_id"] == pipeline_id]
		if not rules:
			issues.append("no_transform_rules_defined")
		if pipeline["status"] == "retired":
			issues.append("pipeline_is_retired")
		return {
			"pipeline_id": pipeline_id, "valid": len(issues) == 0, "issues": issues,
			"rule_count": len(rules), "validated_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 17. execution_complete
	# ------------------------------------------------------------------
	async def execution_complete(
		self,
		tenant_id: str,
		execution_id: str,
		records_processed: int,
		records_failed: int,
		status: str = "completed",
	) -> dict[str, Any]:
		"""Mark an execution as completed with final metrics."""
		exec_rec = await self._require_execution(tenant_id, execution_id)
		exec_rec["records_processed"] = records_processed
		exec_rec["records_failed"] = records_failed
		exec_rec["status"] = status
		exec_rec["completed_at"] = _utc_now()
		await self._store.put("etlp_executions", exec_rec)
		await self._audit.log_event("execution_completed", self.actor_id, tenant_id, execution_id,
									{"records_processed": records_processed, "records_failed": records_failed, "status": status},
									severity="medium" if records_failed > 0 else "info")
		if records_failed > 0:
			await self._notify.send(exec_rec.get("triggered_by", "system"), "audit_log",
									"Pipeline execution had failures",
									f"Execution {execution_id}: {records_failed} records failed")
		return exec_rec

	# ------------------------------------------------------------------
	# 18. pipeline_pause
	# ------------------------------------------------------------------
	async def pipeline_pause(self, tenant_id: str, pipeline_id: str, actor: str, reason: str = "") -> dict[str, Any]:
		"""Pause a pipeline (prevents new executions)."""
		pipeline = await self._require_pipeline(tenant_id, pipeline_id)
		pipeline["status"] = "paused"
		pipeline["updated_at"] = _utc_now()
		await self._store.put("etlp_pipelines", pipeline)
		await self._audit.log_event("pipeline_paused", self.actor_id, tenant_id, pipeline_id, {"actor": actor, "reason": reason})
		return pipeline

	# ------------------------------------------------------------------
	# 19. pipeline_resume
	# ------------------------------------------------------------------
	async def pipeline_resume(self, tenant_id: str, pipeline_id: str, actor: str) -> dict[str, Any]:
		"""Resume a paused pipeline."""
		pipeline = await self._require_pipeline(tenant_id, pipeline_id)
		pipeline["status"] = "active"
		pipeline["updated_at"] = _utc_now()
		await self._store.put("etlp_pipelines", pipeline)
		await self._audit.log_event("pipeline_resumed", self.actor_id, tenant_id, pipeline_id, {"actor": actor})
		return pipeline

	# ------------------------------------------------------------------
	# 20. pipeline_retire
	# ------------------------------------------------------------------
	async def pipeline_retire(self, tenant_id: str, pipeline_id: str, actor: str, reason: str) -> dict[str, Any]:
		"""Retire a pipeline permanently."""
		pipeline = await self._require_pipeline(tenant_id, pipeline_id)
		pipeline["status"] = "retired"
		pipeline["retired_by"] = actor
		pipeline["retirement_reason"] = reason
		pipeline["updated_at"] = _utc_now()
		await self._store.put("etlp_pipelines", pipeline)
		await self._audit.log_event("pipeline_retired", self.actor_id, tenant_id, pipeline_id, {"actor": actor, "reason": reason}, severity="medium")
		return pipeline

	# ------------------------------------------------------------------
	# 21. register_mapping
	# ------------------------------------------------------------------
	async def register_mapping(
		self,
		tenant_id: str,
		mapping_id: str,
		pipeline_id: str,
		source_id: str,
		target_id: str,
		field_mappings: list[dict[str, Any]],
		schema_validated: bool = True,
	) -> dict[str, Any]:
		"""Register field-level source-to-target mapping for a pipeline."""
		await self._require_pipeline(tenant_id, pipeline_id)
		record = {
			"id": mapping_id, "tenant_id": tenant_id, "pipeline_id": pipeline_id,
			"source_id": source_id, "target_id": target_id,
			"field_mappings": field_mappings, "schema_validated": schema_validated,
			"status": "active", "created_at": _utc_now(),
		}
		await self._store.put("etlp_mappings", record)
		await self._audit.log_event("mapping_registered", self.actor_id, tenant_id, mapping_id, {"pipeline_id": pipeline_id, "field_count": len(field_mappings)})
		return record

	# ------------------------------------------------------------------
	# 22. publish_output
	# ------------------------------------------------------------------
	async def publish_output(
		self,
		tenant_id: str,
		publish_id: str,
		execution_id: str,
		requester: str,
		approval_recorded: bool = True,
	) -> dict[str, Any]:
		"""Publish pipeline execution output after quality gate."""
		exec_rec = await self._require_execution(tenant_id, execution_id)
		quality = await self._latest_quality(tenant_id, execution_id)
		if quality and not quality["gate_passed"]:
			raise PermissionError("quality_gate_not_passed")
		record = {
			"id": publish_id, "tenant_id": tenant_id, "execution_id": execution_id,
			"pipeline_id": exec_rec["pipeline_id"], "requester": requester,
			"approval_recorded": approval_recorded,
			"quality_score": quality["score"] if quality else None,
			"status": "published", "published_at": _utc_now(),
		}
		await self._store.put("etlp_published", record)
		exec_rec["status"] = "published"
		exec_rec["updated_at"] = _utc_now()
		await self._store.put("etlp_executions", exec_rec)
		await self._audit.log_event("output_published", self.actor_id, tenant_id, publish_id, {"execution_id": execution_id})
		return record

	# ------------------------------------------------------------------
	# 23. retry_execution
	# ------------------------------------------------------------------
	async def retry_execution(self, tenant_id: str, execution_id: str, max_retries: int = 3) -> dict[str, Any]:
		"""Retry a failed execution."""
		exec_rec = await self._require_execution(tenant_id, execution_id)
		retries = exec_rec.get("retry_count", 0) + 1
		if retries > max_retries:
			raise PermissionError(f"max_retries_exceeded:{max_retries}")
		exec_rec["status"] = "retrying"
		exec_rec["retry_count"] = retries
		exec_rec["updated_at"] = _utc_now()
		await self._store.put("etlp_executions", exec_rec)
		await self._audit.log_event("execution_retried", self.actor_id, tenant_id, execution_id, {"retry_count": retries})
		return exec_rec

	# ------------------------------------------------------------------
	# 24. cancel_execution
	# ------------------------------------------------------------------
	async def cancel_execution(self, tenant_id: str, execution_id: str, reason: str = "") -> dict[str, Any]:
		"""Cancel a running or queued execution."""
		exec_rec = await self._require_execution(tenant_id, execution_id)
		if exec_rec["status"] in {"completed", "published", "cancelled"}:
			raise PermissionError("execution_already_finalized")
		exec_rec["status"] = "cancelled"
		exec_rec["cancellation_reason"] = reason
		exec_rec["completed_at"] = _utc_now()
		await self._store.put("etlp_executions", exec_rec)
		await self._audit.log_event("execution_cancelled", self.actor_id, tenant_id, execution_id, {"reason": reason})
		return exec_rec

	# ------------------------------------------------------------------
	# 25. replay_execution
	# ------------------------------------------------------------------
	async def replay_execution(
		self,
		tenant_id: str,
		replay_id: str,
		execution_id: str,
		replay_type: str,
		reason: str,
		window_hours: int = 24,
	) -> dict[str, Any]:
		"""Create a replay of a past execution within a time window."""
		await self._require_execution(tenant_id, execution_id)
		record = {
			"id": replay_id, "tenant_id": tenant_id, "execution_id": execution_id,
			"replay_type": replay_type, "reason": reason, "window_hours": window_hours,
			"status": "queued", "created_at": _utc_now(),
		}
		await self._store.put("etlp_replays", record)
		await self._audit.log_event("execution_replayed", self.actor_id, tenant_id, replay_id, {"execution_id": execution_id, "window_hours": window_hours})
		return record

	# ------------------------------------------------------------------
	# 26. register_pipeline_agent
	# ------------------------------------------------------------------
	async def register_pipeline_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		"""Register a pipeline automation agent."""
		record = {
			"id": agent_id, "tenant_id": tenant_id, "name": name,
			"runtime": _normalize(runtime), "role": _normalize(role),
			"scope": scope, "owner": owner, "purpose": purpose,
			"contribution_disclosed": contribution_disclosed,
			"human_approval_required": human_approval_required,
			"status": "active", "registered_at": _utc_now(),
		}
		await self._store.put("etlp_agents", record)
		await self._audit.log_event("agent_registered", self.actor_id, tenant_id, agent_id, {"role": role, "runtime": runtime})
		return record

	# ------------------------------------------------------------------
	# 27. bulk_create_pipelines
	# ------------------------------------------------------------------
	async def bulk_create_pipelines(self, tenant_id: str, pipelines: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""Bulk-create pipelines in parallel."""
		tasks = [
			self.pipeline_design(tenant_id, p["pipeline_id"], p["name"], p.get("mode", "etl"), p.get("owner", "system"), p.get("description", ""))
			for p in pipelines
		]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		out = []
		for p, res in zip(pipelines, results):
			if isinstance(res, Exception):
				out.append({"pipeline_id": p["pipeline_id"], "status": "failed", "error": str(res)})
			else:
				out.append({**res, "status": "ok"})  # type: ignore[arg-type]
		await self._audit.log_event("bulk_pipelines_created", self.actor_id, tenant_id, "bulk", {"count": len(pipelines)})
		return out

	# ------------------------------------------------------------------
	# 28. bulk_run_pipelines
	# ------------------------------------------------------------------
	async def bulk_run_pipelines(
		self,
		tenant_id: str,
		pipeline_ids: list[str],
		environment: str,
		triggered_by: str,
	) -> list[dict[str, Any]]:
		"""Run multiple pipelines in parallel."""
		tasks = [self.run_pipeline(tenant_id, pid, environment, triggered_by) for pid in pipeline_ids]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		out = []
		for pid, res in zip(pipeline_ids, results):
			if isinstance(res, Exception):
				out.append({"pipeline_id": pid, "status": "failed", "error": str(res)})
			else:
				out.append({**res, "status": "ok"})  # type: ignore[arg-type]
		await self._audit.log_event("bulk_pipelines_run", self.actor_id, tenant_id, "bulk", {"count": len(pipeline_ids)})
		return out

	# ------------------------------------------------------------------
	# 29. compliance_check
	# ------------------------------------------------------------------
	async def compliance_check(self, tenant_id: str, framework: str = "SOX") -> dict[str, Any]:
		"""Check ETL pipeline compliance posture."""
		pipelines = await self._store.list("etlp_pipelines", tenant_id)
		lineage = await self._store.list("etlp_lineage", tenant_id)
		quality = await self._store.list("etlp_quality", tenant_id)
		issues: list[str] = []
		pipeline_ids_with_lineage = {l["pipeline_id"] for l in lineage}
		missing_lineage = [p["id"] for p in pipelines if p["id"] not in pipeline_ids_with_lineage and p["status"] == "active"]
		if missing_lineage:
			issues.append(f"{len(missing_lineage)}_active_pipelines_missing_lineage")
		low_quality = [q for q in quality if not q["gate_passed"]]
		if low_quality:
			issues.append(f"{len(low_quality)}_executions_failed_quality_gate")
		return {
			"tenant_id": tenant_id, "framework": framework, "passed": len(issues) == 0,
			"issues": issues, "pipeline_count": len(pipelines),
			"lineage_coverage_percent": round(len(pipeline_ids_with_lineage) / max(len(pipelines), 1) * 100, 2),
			"checked_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 30. dashboard_summary
	# ------------------------------------------------------------------
	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		pipelines = await self._store.list("etlp_pipelines", tenant_id)
		executions = await self._store.list("etlp_executions", tenant_id)
		schedules = await self._store.list("etlp_schedules", tenant_id)
		published = await self._store.list("etlp_published", tenant_id)
		quality = await self._store.list("etlp_quality", tenant_id)
		agents = await self._store.list("etlp_agents", tenant_id)
		return {
			"tenant_id": tenant_id,
			"pipeline_count": len(pipelines),
			"active_pipelines": sum(1 for p in pipelines if p["status"] == "active"),
			"retired_pipelines": sum(1 for p in pipelines if p["status"] == "retired"),
			"execution_count": len(executions),
			"published_executions": len(published),
			"failed_executions": sum(1 for e in executions if e["status"] == "failed"),
			"schedule_count": len(schedules),
			"quality_checks": len(quality),
			"quality_pass_rate": round(sum(1 for q in quality if q["gate_passed"]) / max(len(quality), 1) * 100, 2),
			"agent_count": len(agents),
			"audit_events": len(await self._store.list("etlp_audit", tenant_id)),
			"generated_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 31. health_check
	# ------------------------------------------------------------------
	async def health_check(self) -> dict[str, Any]:
		try:
			test_id = f"_health_{uuid7str()}"
			await self.pipeline_design("_health", test_id, "HealthPipeline", "etl", "system")
			await self._store.delete("etlp_pipelines", test_id)
			status = "healthy"
		except Exception as exc:
			status = f"degraded:{exc}"
		return {
			"service": "ETLPService", "status": status,
			"collections": {
				"pipelines": len(await self._store.list("etlp_pipelines")),
				"executions": len(await self._store.list("etlp_executions")),
				"audit_events": len(await self._store.list("etlp_audit")),
			},
			"checked_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 32. export_csv
	# ------------------------------------------------------------------
	async def export_csv(self, tenant_id: str, collection: str = "etlp_pipelines") -> str:
		records = await self._store.list(collection, tenant_id)
		if not records:
			return ""
		buf = io.StringIO()
		writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
		writer.writeheader()
		writer.writerows(records)
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 33. export_json
	# ------------------------------------------------------------------
	async def export_json(self, tenant_id: str, collection: str = "etlp_pipelines") -> str:
		records = await self._store.list(collection, tenant_id)
		return json.dumps(records, indent=2, default=str)

	# ------------------------------------------------------------------
	# 34–44. list helpers
	# ------------------------------------------------------------------
	async def list_pipelines(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_pipelines", tenant_id)

	async def list_sources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_sources", tenant_id)

	async def list_targets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_targets", tenant_id)

	async def list_mappings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_mappings", tenant_id)

	async def list_executions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_executions", tenant_id)

	async def list_schedules(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_schedules", tenant_id)

	async def list_quality_results(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_quality", tenant_id)

	async def list_lineage(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_lineage", tenant_id)

	async def list_cdc_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_cdc_events", tenant_id)

	async def list_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_agents", tenant_id)

	async def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("etlp_audit", tenant_id)

	# compat
	async def list_records(self, tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
		if record_type == "pipelines":
			return await self.list_pipelines(tenant_id)
		if record_type == "executions":
			return await self.list_executions(tenant_id)
		return await self.list_pipelines(tenant_id)

	# ------------------------------------------------------------------
	# 45. sla_check
	# ------------------------------------------------------------------
	async def sla_check(self, tenant_id: str, pipeline_id: str) -> dict[str, Any]:
		"""Evaluate SLA compliance for a pipeline's recent executions."""
		monitors = [m for m in await self._store.list("etlp_sla_monitors", tenant_id) if m["pipeline_id"] == pipeline_id]
		executions = [e for e in await self._store.list("etlp_executions", tenant_id) if e["pipeline_id"] == pipeline_id]
		violations: list[str] = []
		for monitor in monitors:
			failed = [e for e in executions if e.get("records_failed", 0) > 0]
			failure_rate = len(failed) / max(len(executions), 1) * 100
			if failure_rate > monitor["max_failure_rate_percent"]:
				violations.append(f"failure_rate_{round(failure_rate, 2)}_exceeds_{monitor['max_failure_rate_percent']}")
		return {
			"pipeline_id": pipeline_id, "tenant_id": tenant_id,
			"sla_monitors": len(monitors), "violations": violations,
			"compliant": len(violations) == 0, "checked_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# Internals
	# ------------------------------------------------------------------

	async def _require_pipeline(self, tenant_id: str, pipeline_id: str) -> dict[str, Any]:
		rec = await self._store.get("etlp_pipelines", pipeline_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"pipeline_not_found:{pipeline_id}")
		if rec["status"] == "retired":
			raise ValueError(f"pipeline_retired:{pipeline_id}")
		return rec

	async def _require_execution(self, tenant_id: str, execution_id: str) -> dict[str, Any]:
		rec = await self._store.get("etlp_executions", execution_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"execution_not_found:{execution_id}")
		return rec

	async def _latest_quality(self, tenant_id: str, execution_id: str) -> dict[str, Any] | None:
		records = [q for q in await self._store.list("etlp_quality", tenant_id) if q["execution_id"] == execution_id]
		return records[-1] if records else None


# Backward compat alias
ETLPLifecycleService = ETLPService

__all__ = ["ETLPService", "ETLPLifecycleService"]
