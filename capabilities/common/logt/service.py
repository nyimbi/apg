"""Service layer for APG Logging and Tracing."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	DEFAULT_CONFIGURATION,
	SUPPORTED_LOGT_AGENT_ROLES,
	SUPPORTED_LOGT_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
)
from .models import DiagnosticExport, DiagnosticQuery, IngestionPipeline, LogEvent, LogtAgent, LogtAuditEvent, RetentionPolicy, SpanRecord, TraceRecord
from .observability_runtime import ObservabilityRuntime


class LogtService:
	"""Structured log, trace, pipeline, retention, query, export, and audit service."""

	def __init__(self) -> None:
		self._pipelines: dict[str, IngestionPipeline] = {}
		self._logs: dict[str, LogEvent] = {}
		self._traces: dict[str, TraceRecord] = {}
		self._spans: dict[str, SpanRecord] = {}
		self._queries: dict[str, DiagnosticQuery] = {}
		self._exports: dict[str, DiagnosticExport] = {}
		self._retention_policies: dict[str, RetentionPolicy] = {}
		self._audit_events: dict[str, LogtAuditEvent] = {}
		self._agents: dict[str, LogtAgent] = {}
		self._runtime = ObservabilityRuntime()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_retention_policy(
		self,
		policy_id: str,
		tenant_id: str,
		name: str,
		log_retention_days: int,
		span_retention_days: int | None = None,
		redaction_required: bool = True,
		export_approval_required: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if log_retention_days <= 0:
			raise PermissionError("log_retention_policy_required")
		span_days = span_retention_days or int(DEFAULT_CONFIGURATION["tracing"]["span_retention_days"])
		if span_days <= 0:
			raise PermissionError("span_retention_policy_required")
		policy = RetentionPolicy(
			id=policy_id,
			tenant_id=tenant_id,
			name=name,
			log_retention_days=log_retention_days,
			span_retention_days=span_days,
			redaction_required=redaction_required,
			export_approval_required=export_approval_required,
		)
		self._retention_policies[_state_key(tenant_id, policy_id)] = policy
		self._audit(tenant_id, policy_id, "retention_policy_created", "system", "allow", metadata={"log_days": log_retention_days, "span_days": span_days})
		return policy.to_dict()

	def create_pipeline(
		self,
		pipeline_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		schema_ref: str,
		event_bus_ref: str,
		sampling_policy: str,
		retention_policy_id: str,
		status: str = "active",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_pipeline",
			"pipeline_owner_assigned": bool(owner),
			"schema_ref_present": bool(schema_ref),
			"event_stream": event_stream_name(event_bus_ref),
			"sampling_policy_present": bool(sampling_policy),
		})
		self._raise_if_denied(result)
		self._require_retention_policy(retention_policy_id, tenant_id)
		pipeline = IngestionPipeline(
			id=pipeline_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			schema_ref=schema_ref,
			event_bus_ref=event_bus_ref,
			sampling_policy=sampling_policy,
			retention_policy_id=retention_policy_id,
			status=status,
		)
		self._pipelines[_state_key(tenant_id, pipeline_id)] = pipeline
		self._audit(tenant_id, pipeline_id, "pipeline_created", owner, result["decision"], reasons=self._reasons(result))
		return pipeline.to_dict()

	def ingest_log(
		self,
		log_id: str,
		tenant_id: str,
		pipeline_id: str,
		service_name: str,
		severity: str,
		message: str,
		attributes: dict[str, Any] | None = None,
		trace_id: str = "",
		span_id: str = "",
		sensitive_log_content: bool = False,
		redaction_applied: bool = True,
	) -> dict[str, Any]:
		pipeline = self._require_pipeline(pipeline_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "ingest_log",
			"service_name_present": bool(service_name),
			"sensitive_log_content": bool(sensitive_log_content),
			"redaction_applied": bool(redaction_applied),
		})
		self._raise_if_denied(result)
		severity_value = self._runtime.normalize_severity(severity)
		log = LogEvent(
			id=log_id,
			tenant_id=tenant_id,
			pipeline_id=pipeline.id,
			service_name=service_name,
			severity=severity_value,
			message=self._runtime.redact_message(message, redaction_applied),
			attributes=dict(attributes or {}),
			trace_id=trace_id,
			span_id=span_id,
			sensitive_log_content=sensitive_log_content,
			redaction_applied=redaction_applied,
		)
		self._logs[_state_key(tenant_id, log_id)] = log
		self._audit(tenant_id, log_id, "log_ingested", pipeline.owner, result["decision"], reasons=self._reasons(result), metadata={"severity": severity_value, "service": service_name})
		return log.to_dict()

	def ingest_trace(
		self,
		trace_record_id: str,
		tenant_id: str,
		pipeline_id: str,
		trace_id: str,
		root_service: str,
		operation: str,
		trace_context: dict[str, Any],
		sampling_policy: str = "",
		status: str = "active",
	) -> dict[str, Any]:
		pipeline = self._require_pipeline(pipeline_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "ingest_trace",
			"trace_context_present": bool(trace_context),
			"trace_id_present": bool(trace_id),
		})
		self._raise_if_denied(result)
		if not root_service:
			raise PermissionError("root_service_required")
		trace = TraceRecord(
			id=trace_record_id,
			tenant_id=tenant_id,
			pipeline_id=pipeline.id,
			trace_id=trace_id,
			root_service=root_service,
			operation=operation,
			trace_context=dict(trace_context),
			sampling_policy=sampling_policy or pipeline.sampling_policy,
			status=status,
		)
		self._traces[_state_key(tenant_id, trace_record_id)] = trace
		self._audit(tenant_id, trace_record_id, "trace_ingested", pipeline.owner, result["decision"], metadata={"trace_id": trace_id, "root_service": root_service})
		return trace.to_dict()

	def record_span(
		self,
		span_record_id: str,
		tenant_id: str,
		trace_id: str,
		span_id: str,
		service_name: str,
		operation: str,
		duration_ms: float,
		parent_span_id: str = "",
		attributes: dict[str, Any] | None = None,
		error: bool = False,
	) -> dict[str, Any]:
		self._require_trace_by_trace_id(trace_id, tenant_id)
		if not span_id:
			raise PermissionError("span_id_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_span",
			"span_service_present": bool(service_name),
			"span_duration_valid": duration_ms >= 0,
		})
		self._raise_if_denied(result)
		span = SpanRecord(
			id=span_record_id,
			tenant_id=tenant_id,
			trace_id=trace_id,
			span_id=span_id,
			parent_span_id=parent_span_id,
			service_name=service_name,
			operation=operation,
			duration_ms=round(float(duration_ms), 3),
			status=self._runtime.span_status(duration_ms, error),
			attributes=dict(attributes or {}),
		)
		self._spans[_state_key(tenant_id, span_record_id)] = span
		self._audit(tenant_id, span_record_id, "span_recorded", service_name, result["decision"], metadata={"trace_id": trace_id, "duration_ms": duration_ms})
		return span.to_dict()

	def search_logs(
		self,
		query_id: str,
		tenant_id: str,
		query_text: str,
		requested_by: str,
		query_window_hours: int,
		query_review_recorded: bool = False,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "search_logs",
			"query_actor_present": bool(requested_by),
			"query_window_hours": query_window_hours,
			"query_review_recorded": bool(query_review_recorded),
		})
		self._raise_if_review_required(result, query_review_recorded)
		matches = [
			log for log in self.list_logs(tenant_id)
			if self._runtime.match_log(log, query_text)
		]
		query = DiagnosticQuery(
			id=query_id,
			tenant_id=tenant_id,
			query_text=query_text,
			requested_by=requested_by,
			query_window_hours=query_window_hours,
			query_review_recorded=query_review_recorded,
			result_count=len(matches),
			status=self._runtime.query_status(query_window_hours, query_review_recorded),
		)
		self._queries[_state_key(tenant_id, query_id)] = query
		self._audit(tenant_id, query_id, "diagnostic_query_executed", requested_by, result["decision"], reasons=self._reasons(result), metadata={"result_count": len(matches)})
		return {"query": query.to_dict(), "results": matches}

	def export_logs(
		self,
		export_id: str,
		tenant_id: str,
		export_type: str,
		requested_by: str,
		item_ids: list[str] | tuple[str, ...],
		approval_recorded: bool,
		approval_ref: str = "",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "export_logs",
			"approval_recorded": bool(approval_recorded),
			"approval_ref_present": bool(approval_ref),
		})
		self._raise_if_denied(result)
		if not requested_by:
			raise PermissionError("export_actor_required")
		for item_id in item_ids:
			if not self._diagnostic_item_belongs_to_tenant(item_id, tenant_id):
				raise KeyError("diagnostic_export_item_not_found")
		export = DiagnosticExport(
			id=export_id,
			tenant_id=tenant_id,
			export_type=export_type,
			requested_by=requested_by,
			approval_ref=approval_ref,
			item_ids=tuple(item_ids),
			status="approved",
		)
		self._exports[_state_key(tenant_id, export_id)] = export
		self._audit(tenant_id, export_id, "diagnostic_export_created", requested_by, result["decision"], metadata={"item_count": len(item_ids), "export_type": export_type})
		return export.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		policy_id = str(metadata.get("retention_policy_id") or f"retention-{record_id}")
		if _state_key(tenant_id, policy_id) not in self._retention_policies:
			self.create_retention_policy(
				policy_id=policy_id,
				tenant_id=tenant_id,
				name=str(metadata.get("retention_name") or "Default diagnostics retention"),
				log_retention_days=int(metadata.get("log_retention_days", 30)),
			)
		pipeline_id = str(metadata.get("pipeline_id") or f"pipeline-{record_id}")
		if _state_key(tenant_id, pipeline_id) not in self._pipelines:
			self.create_pipeline(
				pipeline_id=pipeline_id,
				tenant_id=tenant_id,
				name=str(metadata.get("pipeline_name") or "Default diagnostics pipeline"),
				owner=str(metadata.get("owner") or "system"),
				schema_ref=str(metadata.get("schema_ref") or "schema://diagnostics"),
				event_bus_ref=str(metadata.get("event_bus_ref") or "bytewax://diagnostics"),
				sampling_policy=str(metadata.get("sampling_policy") or "head-based"),
				retention_policy_id=policy_id,
			)
		return self.ingest_log(
			log_id=record_id,
			tenant_id=tenant_id,
			pipeline_id=pipeline_id,
			service_name=str(metadata.get("service_name") or "apg"),
			severity=str(metadata.get("severity") or "info"),
			message=str(metadata.get("message") or record_id),
			attributes=metadata,
			trace_id=str(metadata.get("trace_id") or ""),
			span_id=str(metadata.get("span_id") or ""),
			sensitive_log_content=bool(metadata.get("sensitive_log_content", False)),
			redaction_applied=bool(metadata.get("redaction_applied", True)),
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_logs(tenant_id)

	def list_pipelines(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._pipelines, tenant_id)

	def list_logs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._logs, tenant_id)

	def list_traces(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._traces, tenant_id)

	def list_spans(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._spans, tenant_id)

	def list_queries(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._queries, tenant_id)

	def list_exports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._exports, tenant_id)

	def list_retention_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._retention_policies, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def register_logt_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"logt_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_LOGT_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_LOGT_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_if_denied(result)
		agent = LogtAgent(
			id=agent_id or f"logt-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._audit(tenant_id, agent.id, "logt_agent_registered", name, result["decision"], metadata=agent.to_dict())
		return agent.to_dict()

	def validate_batch_diagnostic_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": True,
			"requested_operation": "batch_diagnostic_mutation",
			"event_stream": event_stream,
		})

	def list_logt_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def service_map(self, tenant_id: str = "default") -> dict[str, Any]:
		return self._runtime.service_map(self.list_spans(tenant_id))

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		logs = self.list_logs(tenant_id)
		spans = self.list_spans(tenant_id)
		return {
			"tenant_id": tenant_id,
			"pipeline_count": len(self.list_pipelines(tenant_id)),
			"log_count": len(logs),
			"trace_count": len(self.list_traces(tenant_id)),
			"span_count": len(spans),
			"query_count": len(self.list_queries(tenant_id)),
			"export_count": len(self.list_exports(tenant_id)),
			"retention_policy_count": len(self.list_retention_policies(tenant_id)),
			"logt_agent_count": len(self.list_logt_agents(tenant_id)),
			"error_log_count": len([log for log in logs if log["severity"] in {"error", "critical"}]),
			"slow_span_count": len([span for span in spans if span["status"] == "slow"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_retention_policy(self, policy_id: str, tenant_id: str) -> RetentionPolicy:
		policy = self._retention_policies.get(_state_key(tenant_id, policy_id))
		if policy is None or policy.tenant_id != tenant_id:
			raise KeyError("retention_policy_not_found")
		return policy

	def _require_pipeline(self, pipeline_id: str, tenant_id: str) -> IngestionPipeline:
		pipeline = self._pipelines.get(_state_key(tenant_id, pipeline_id))
		if pipeline is None or pipeline.tenant_id != tenant_id:
			raise KeyError("ingestion_pipeline_not_found")
		return pipeline

	def _require_trace_by_trace_id(self, trace_id: str, tenant_id: str) -> TraceRecord:
		for trace in self._traces.values():
			if trace.trace_id == trace_id and trace.tenant_id == tenant_id:
				return trace
		raise KeyError("trace_not_found")

	def _diagnostic_item_belongs_to_tenant(self, item_id: str, tenant_id: str) -> bool:
		for records in (self._logs, self._traces, self._spans):
			item = records.get(_state_key(tenant_id, item_id))
			if item is not None:
				return item.tenant_id == tenant_id
		return False

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "diagnostic_policy_blocked")

	def _raise_if_review_required(self, result: dict[str, Any], review_recorded: bool) -> None:
		self._raise_if_denied(result)
		if result["decision"] == "require_review" and not review_recorded:
			raise PermissionError(", ".join(self._reasons(result)) or "diagnostic_review_required")

	def _audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> None:
		event_id = self._runtime.stable_id("audit", {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"event_type": event_type,
			"actor": actor,
			"index": len(self._audit_events),
		})
		self._audit_events[event_id] = LogtAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "diagnostic_policy_blocked") for action in result.get("actions", ()))


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"
