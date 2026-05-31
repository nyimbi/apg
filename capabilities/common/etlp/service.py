#!/usr/bin/env python3
"""
APG ETLP Business Logic Service
Core pipeline orchestration and processing engine

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from uuid_extensions import uuid7str

from .capability_contract import (
	PRIVILEGED_ETLP_AGENT_ROLES,
	SUPPORTED_ETLP_AGENT_ROLES,
	SUPPORTED_ETLP_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	Pipeline, Transformation, Execution, DataSource, QualityRule, Schedule,
	PipelineStatus, ExecutionMode, TransformationType, QualityRuleType,
	PipelineMetrics, validate_pipeline_dependencies, calculate_pipeline_complexity
)


@dataclass
class ETLPPipelineRecord:
	record_id: str
	tenant_id: str
	pipeline_id: str
	name: str
	mode: str
	owner: str | None
	description: str = ""
	status: str = "draft"
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	tags: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPDatasourceRecord:
	datasource_id: str
	tenant_id: str
	name: str
	datasource_type: str
	owner: str | None
	secret_ref: str | None
	approved: bool
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPMappingRecord:
	mapping_id: str
	tenant_id: str
	pipeline_id: str
	source_datasource_id: str
	target_datasource_id: str
	field_mappings: list[dict[str, Any]]
	schema_validated: bool
	lineage_emitted: bool
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPExecutionRecord:
	execution_id: str
	tenant_id: str
	pipeline_id: str
	environment: str
	triggered_by: str
	idempotency_key: str | None
	estimated_cost: float
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	mode: str = "elt"
	records_processed: int = 0
	records_failed: int = 0
	logs: list[dict[str, Any]] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPQualityRecord:
	quality_id: str
	tenant_id: str
	execution_id: str
	pipeline_id: str
	score: float
	dimensions: dict[str, float]
	gate_passed: bool
	assessor: str
	status: str = "accepted"
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPScheduleRecord:
	schedule_id: str
	tenant_id: str
	pipeline_id: str
	environment: str
	schedule: str
	owner: str
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPPublishRecord:
	publish_id: str
	tenant_id: str
	execution_id: str
	pipeline_id: str
	requester: str
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	quality_score: float | None = None
	review_notes: str | None = None
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPReplayRecord:
	replay_id: str
	tenant_id: str
	execution_id: str
	replay_type: str
	reason: str | None
	window_hours: int
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPPipelineAgentRecord:
	agent_id: str
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
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPLifecycleBatchRecord:
	batch_id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ETLPAuditEventRecord:
	event_id: str
	tenant_id: str
	event_type: str
	subject: str
	actor: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	details: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


class ETLPLifecycleService:
	"""Dependency-light ETLP lifecycle and guardrail control plane."""

	def __init__(self, tenant_id: str = "default"):
		self.tenant_id = tenant_id
		self._agent_runtimes = set(SUPPORTED_ETLP_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_ETLP_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_ETLP_AGENT_ROLES)
		self.pipelines: dict[str, ETLPPipelineRecord] = {}
		self.datasources: dict[str, ETLPDatasourceRecord] = {}
		self.mappings: dict[str, ETLPMappingRecord] = {}
		self.executions: dict[str, ETLPExecutionRecord] = {}
		self.quality_results: dict[str, ETLPQualityRecord] = {}
		self.schedules: dict[str, ETLPScheduleRecord] = {}
		self.publish_reviews: dict[str, ETLPPublishRecord] = {}
		self.replay_requests: dict[str, ETLPReplayRecord] = {}
		self.pipeline_agents: dict[str, ETLPPipelineAgentRecord] = {}
		self.lifecycle_batches: dict[str, ETLPLifecycleBatchRecord] = {}
		self.audit_events: list[ETLPAuditEventRecord] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def register_pipeline(
		self,
		*,
		tenant_id: str,
		pipeline_id: str,
		name: str,
		mode: str,
		owner: str | None,
		description: str = "",
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> ETLPPipelineRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		mode = self._require_text(mode, "mode")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_pipeline",
			"owner_assigned": bool(str(owner or "").strip()),
			"unsupported_mode": mode not in set(self.describe(tenant_id)["configuration"]["pipelines"]["supported_modes"]),
		}
		decision = evaluate_capability_rules(context)
		record = ETLPPipelineRecord(
			record_id=uuid7str(),
			tenant_id=tenant_id,
			pipeline_id=self._require_text(pipeline_id, "pipeline_id"),
			name=self._require_text(name, "name"),
			mode=mode,
			owner=owner.strip() if isinstance(owner, str) and owner.strip() else None,
			description=description,
			status="draft" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			decision=decision["decision"],
			matched_rules=decision["matched_rules"],
			tags=list(tags or []),
			metadata=dict(metadata or {}),
		)
		self.pipelines[self._key(tenant_id, record.pipeline_id)] = record
		self._audit(tenant_id, "pipeline.registered", record.pipeline_id, record.owner or "system", decision, context)
		return record

	def register_datasource(
		self,
		*,
		tenant_id: str,
		datasource_id: str,
		name: str,
		datasource_type: str,
		owner: str | None,
		secret_ref: str | None,
		approved: bool,
		embedded_secret_present: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> ETLPDatasourceRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		supported = set(self.describe(tenant_id)["configuration"]["datasources"]["supported_types"])
		datasource_type = self._require_text(datasource_type, "datasource_type")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_datasource",
			"datasource_owner_assigned": bool(str(owner or "").strip()),
			"secret_reference_present": bool(str(secret_ref or "").strip()),
			"datasource_approved": approved,
			"unsupported_datasource_type": datasource_type not in supported,
			"embedded_secret_present": embedded_secret_present,
		}
		decision = evaluate_capability_rules(context)
		record = ETLPDatasourceRecord(
			datasource_id=self._require_text(datasource_id, "datasource_id"),
			tenant_id=tenant_id,
			name=self._require_text(name, "name"),
			datasource_type=datasource_type,
			owner=owner.strip() if isinstance(owner, str) and owner.strip() else None,
			secret_ref=secret_ref.strip() if isinstance(secret_ref, str) and secret_ref.strip() else None,
			approved=approved,
			status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			decision=decision["decision"],
			matched_rules=decision["matched_rules"],
			metadata=dict(metadata or {}),
		)
		self.datasources[self._key(tenant_id, record.datasource_id)] = record
		self._audit(tenant_id, "datasource.registered", record.datasource_id, record.owner or "system", decision, context)
		return record

	def register_mapping(
		self,
		*,
		tenant_id: str,
		mapping_id: str,
		pipeline_id: str,
		source_datasource_id: str,
		target_datasource_id: str,
		field_mappings: list[dict[str, Any]],
		schema_validated: bool,
		lineage_emitted: bool,
	) -> ETLPMappingRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		pipeline = self._require_pipeline(tenant_id, pipeline_id)
		source_registered = self._key(tenant_id, source_datasource_id) in self.datasources
		target_registered = self._key(tenant_id, target_datasource_id) in self.datasources
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_mapping",
			"source_and_target_registered": source_registered and target_registered,
			"schema_validated": schema_validated,
		}
		decision = evaluate_capability_rules(context)
		record = ETLPMappingRecord(
			mapping_id=self._require_text(mapping_id, "mapping_id"),
			tenant_id=tenant_id,
			pipeline_id=pipeline.pipeline_id,
			source_datasource_id=self._require_text(source_datasource_id, "source_datasource_id"),
			target_datasource_id=self._require_text(target_datasource_id, "target_datasource_id"),
			field_mappings=list(field_mappings),
			schema_validated=schema_validated,
			lineage_emitted=lineage_emitted,
			status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			decision=decision["decision"],
			matched_rules=decision["matched_rules"],
		)
		self.mappings[self._key(tenant_id, record.mapping_id)] = record
		self._audit(tenant_id, "mapping.registered", record.mapping_id, pipeline.owner or "system", decision, context)
		return record

	def execute_pipeline(
		self,
		*,
		tenant_id: str,
		pipeline_id: str,
		environment: str,
		triggered_by: str,
		idempotency_key: str | None,
		approval_recorded: bool = False,
		estimated_cost: float = 0.0,
		cost_review_recorded: bool = False,
	) -> ETLPExecutionRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		pipeline = self._require_pipeline(tenant_id, pipeline_id)
		has_mapping = any(mapping.tenant_id == tenant_id and mapping.pipeline_id == pipeline.pipeline_id for mapping in self.mappings.values())
		lineage_emitted = any(mapping.tenant_id == tenant_id and mapping.pipeline_id == pipeline.pipeline_id and mapping.lineage_emitted for mapping in self.mappings.values())
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "execute_pipeline",
			"owner_assigned": bool(pipeline.owner),
			"environment": self._require_text(environment, "environment"),
			"approval_recorded": approval_recorded,
			"idempotency_key_present": bool(str(idempotency_key or "").strip()),
			"transformation_present": has_mapping,
			"lineage_emitted": lineage_emitted,
			"estimated_cost": estimated_cost,
			"cost_review_recorded": cost_review_recorded,
		}
		decision = evaluate_capability_rules(context)
		record = ETLPExecutionRecord(
			execution_id=uuid7str(),
			tenant_id=tenant_id,
			pipeline_id=pipeline.pipeline_id,
			environment=environment,
			triggered_by=self._require_text(triggered_by, "triggered_by"),
			idempotency_key=idempotency_key,
			estimated_cost=estimated_cost,
			status="queued" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			decision=decision["decision"],
			matched_rules=decision["matched_rules"],
			mode=pipeline.mode,
		)
		self.executions[self._key(tenant_id, record.execution_id)] = record
		self._audit(tenant_id, "execution.requested", record.execution_id, record.triggered_by, decision, context)
		return record

	def assess_quality(
		self,
		*,
		tenant_id: str,
		execution_id: str,
		score: float,
		dimensions: dict[str, float],
		assessor: str,
	) -> ETLPQualityRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		execution = self._require_execution(tenant_id, execution_id)
		if score < 0.0 or score > 100.0 or any(value < 0.0 or value > 100.0 for value in dimensions.values()):
			raise ValueError("quality scores must be between 0 and 100")
		minimum = self.describe(tenant_id)["configuration"]["quality"]["minimum_publish_score"]
		record = ETLPQualityRecord(
			quality_id=uuid7str(),
			tenant_id=tenant_id,
			execution_id=execution.execution_id,
			pipeline_id=execution.pipeline_id,
			score=score,
			dimensions=dict(dimensions),
			gate_passed=score >= minimum,
			assessor=self._require_text(assessor, "assessor"),
		)
		self.quality_results[self._key(tenant_id, record.quality_id)] = record
		self._audit(tenant_id, "quality.assessed", record.execution_id, record.assessor, {"decision": "allow", "matched_rules": []}, asdict(record))
		return record

	def schedule_pipeline(
		self,
		*,
		tenant_id: str,
		pipeline_id: str,
		environment: str,
		schedule: str,
		owner: str,
		schedule_review_recorded: bool = False,
	) -> ETLPScheduleRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		pipeline = self._require_pipeline(tenant_id, pipeline_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "schedule_pipeline",
			"environment": self._require_text(environment, "environment"),
			"schedule_review_recorded": schedule_review_recorded,
		}
		decision = evaluate_capability_rules(context)
		record = ETLPScheduleRecord(
			schedule_id=uuid7str(),
			tenant_id=tenant_id,
			pipeline_id=pipeline.pipeline_id,
			environment=environment,
			schedule=self._require_text(schedule, "schedule"),
			owner=self._require_text(owner, "owner"),
			decision=decision["decision"],
			status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
		)
		self.schedules[self._key(tenant_id, record.schedule_id)] = record
		self._audit(tenant_id, "pipeline.scheduled", record.pipeline_id, record.owner, decision, context)
		return record

	def publish_output(
		self,
		*,
		tenant_id: str,
		execution_id: str,
		requester: str,
		publish_approval_recorded: bool,
		review_notes: str | None = None,
	) -> ETLPPublishRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		execution = self._require_execution(tenant_id, execution_id)
		quality = self._latest_quality_for_execution(tenant_id, execution.execution_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_output",
			"quality_gate_passed": bool(quality and quality.gate_passed),
			"quality_score": quality.score if quality else 0.0,
			"publish_approval_recorded": publish_approval_recorded,
		}
		decision = evaluate_capability_rules(context)
		record = ETLPPublishRecord(
			publish_id=uuid7str(),
			tenant_id=tenant_id,
			execution_id=execution.execution_id,
			pipeline_id=execution.pipeline_id,
			requester=self._require_text(requester, "requester"),
			decision=decision["decision"],
			status="published" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
			quality_score=quality.score if quality else None,
			review_notes=review_notes,
		)
		self.publish_reviews[self._key(tenant_id, record.publish_id)] = record
		if decision["decision"] == "allow":
			execution.status = "published"
			execution.updated_at = datetime.utcnow()
		self._audit(tenant_id, "output.publish_evaluated", record.execution_id, record.requester, decision, context)
		return record

	def retry_execution(self, *, tenant_id: str, execution_id: str, retry_count: int, retry_review_recorded: bool = False) -> ETLPExecutionRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		execution = self._require_execution(tenant_id, execution_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "retry_execution",
			"retry_count": retry_count,
			"retry_review_recorded": retry_review_recorded,
		}
		decision = evaluate_capability_rules(context)
		execution.decision = decision["decision"]
		execution.matched_rules = decision["matched_rules"]
		execution.status = "retrying" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"])
		execution.updated_at = datetime.utcnow()
		self._audit(tenant_id, "execution.retry_evaluated", execution.execution_id, execution.triggered_by, decision, context)
		return execution

	def replay_execution(
		self,
		*,
		tenant_id: str,
		execution_id: str,
		replay_type: str,
		reason: str | None,
		window_hours: int,
		replay_review_recorded: bool = False,
	) -> ETLPReplayRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		self._require_execution(tenant_id, execution_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "replay_execution",
			"reason_present": bool(str(reason or "").strip()),
			"replay_window_hours": window_hours,
			"replay_review_recorded": replay_review_recorded,
		}
		decision = evaluate_capability_rules(context)
		record = ETLPReplayRecord(
			replay_id=uuid7str(),
			tenant_id=tenant_id,
			execution_id=execution_id,
			replay_type=self._require_text(replay_type, "replay_type"),
			reason=str(reason or "").strip() or None,
			window_hours=window_hours,
			decision=decision["decision"],
			status="queued" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
		)
		self.replay_requests[self._key(tenant_id, record.replay_id)] = record
		self._audit(tenant_id, "execution.replay_requested", record.execution_id, "system", decision, context)
		return record

	def retire_pipeline(self, *, tenant_id: str, pipeline_id: str, actor: str, impact_review_recorded: bool) -> ETLPPipelineRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		pipeline = self._require_pipeline(tenant_id, pipeline_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_pipeline",
			"impact_review_recorded": impact_review_recorded,
		}
		decision = evaluate_capability_rules(context)
		pipeline.decision = decision["decision"]
		pipeline.matched_rules = decision["matched_rules"]
		if decision["decision"] == "allow":
			pipeline.status = "retired"
			pipeline.updated_at = datetime.utcnow()
		self._audit(tenant_id, "pipeline.retire_evaluated", pipeline.pipeline_id, self._require_text(actor, "actor"), decision, context)
		return pipeline

	def register_pipeline_agent(
		self,
		*,
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
	) -> ETLPPipelineAgentRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		agent_id = self._require_text(agent_id, "agent_id")
		name = self._require_text(name, "name")
		runtime_value = self._normalize_agent_token(runtime)
		role_value = self._normalize_agent_token(role)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_pipeline_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"agent_scope_present": bool(str(scope or "").strip()),
			"agent_owner_present": bool(str(owner or "").strip()),
			"agent_purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_agent_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		}
		decision = evaluate_capability_rules(context)
		if decision["decision"] == "deny":
			self._audit(
				tenant_id,
				"agent.registration_denied",
				agent_id,
				str(owner or "system").strip() or "system",
				decision,
				context,
			)
			raise PermissionError(self._first_reason(decision))
		record_key = self._key(tenant_id, agent_id)
		if record_key in self.pipeline_agents:
			raise ValueError(f"pipeline_agent_already_exists:{agent_id}")
		record = ETLPPipelineAgentRecord(
			agent_id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=self._require_text(scope, "scope"),
			owner=self._require_text(owner, "owner"),
			purpose=self._require_text(purpose, "purpose"),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
		)
		self.pipeline_agents[record_key] = record
		self._audit(tenant_id, "agent.registered", agent_id, record.owner, decision, asdict(record))
		return record

	def validate_etlp_lifecycle_batch(
		self,
		*,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> ETLPLifecycleBatchRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("etlp_lifecycle_batch_empty")
		stream_value = self._normalize_agent_token(event_stream)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_etlp_lifecycle_batch",
			"event_stream": stream_value,
		}
		decision = evaluate_capability_rules(context)
		accepted = decision["decision"] == "allow"
		record = ETLPLifecycleBatchRecord(
			batch_id=uuid7str(),
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			accepted=accepted,
			decision=decision["decision"],
			matched_rules=list(decision["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self.lifecycle_batches[self._key(tenant_id, record.batch_id)] = record
		self._audit(tenant_id, f"lifecycle_batch.{record.status}", stream_value, "etlp", decision, asdict(record))
		if not accepted:
			raise PermissionError(self._first_reason(decision))
		return record

	def list_records(self, tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant_id = tenant_id or self.tenant_id
		collections: dict[str, Any] = {
			"pipelines": self.pipelines.values(),
			"datasources": self.datasources.values(),
			"mappings": self.mappings.values(),
			"executions": self.executions.values(),
			"quality_results": self.quality_results.values(),
			"schedules": self.schedules.values(),
			"publish_reviews": self.publish_reviews.values(),
			"replay_requests": self.replay_requests.values(),
			"pipeline_agents": self.pipeline_agents.values(),
			"lifecycle_batches": self.lifecycle_batches.values(),
			"audit_events": self.audit_events,
		}
		if record_type:
			if record_type not in collections:
				raise ValueError(f"Unsupported record_type {record_type}")
			values = collections[record_type]
		else:
			values = []
			for collection in collections.values():
				values.extend(collection)
		return [
			asdict(record)
			for record in values
			if getattr(record, "tenant_id", None) == tenant_id
		]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant_id = tenant_id or self.tenant_id
		return {
			"tenant_id": tenant_id,
			"pipeline_count": len(self.list_records(tenant_id, "pipelines")),
			"datasource_count": len(self.list_records(tenant_id, "datasources")),
			"mapping_count": len(self.list_records(tenant_id, "mappings")),
			"execution_count": len(self.list_records(tenant_id, "executions")),
			"schedule_count": len(self.list_records(tenant_id, "schedules")),
			"published_count": sum(1 for row in self.list_records(tenant_id, "publish_reviews") if row["status"] == "published"),
			"review_count": sum(1 for record_type in ("datasources", "executions", "replay_requests") for row in self.list_records(tenant_id, record_type) if row["status"] == "pending_review"),
			"pipeline_agent_count": len(self.list_records(tenant_id, "pipeline_agents")),
			"lifecycle_batch_count": len(self.list_records(tenant_id, "lifecycle_batches")),
			"denied_lifecycle_batch_count": sum(1 for row in self.list_records(tenant_id, "lifecycle_batches") if not row["accepted"]),
			"audit_event_count": len(self.list_records(tenant_id, "audit_events")),
		}

	def _audit(self, tenant_id: str, event_type: str, subject: str, actor: str, decision: dict[str, Any], details: dict[str, Any]) -> None:
		self.audit_events.append(ETLPAuditEventRecord(
			event_id=uuid7str(),
			tenant_id=tenant_id,
			event_type=event_type,
			subject=subject,
			actor=actor,
			decision=decision["decision"],
			matched_rules=list(decision["matched_rules"]),
			details=details,
		))

	def _require_pipeline(self, tenant_id: str, pipeline_id: str) -> ETLPPipelineRecord:
		pipeline_id = self._require_text(pipeline_id, "pipeline_id")
		record = self.pipelines.get(self._key(tenant_id, pipeline_id))
		if record is None:
			raise KeyError(f"Pipeline {pipeline_id} not found for tenant {tenant_id}")
		if record.status in {"denied", "retired"}:
			raise ValueError(f"Pipeline {pipeline_id} is {record.status} and cannot continue lifecycle operations")
		return record

	def _require_execution(self, tenant_id: str, execution_id: str) -> ETLPExecutionRecord:
		execution_id = self._require_text(execution_id, "execution_id")
		record = self.executions.get(self._key(tenant_id, execution_id))
		if record is None:
			raise KeyError(f"Execution {execution_id} not found for tenant {tenant_id}")
		return record

	def _latest_quality_for_execution(self, tenant_id: str, execution_id: str) -> ETLPQualityRecord | None:
		for record in reversed(list(self.quality_results.values())):
			if record.tenant_id == tenant_id and record.execution_id == execution_id:
				return record
		return None

	@staticmethod
	def _status_for_decision(decision: str) -> str:
		if decision == "require_review":
			return "pending_review"
		if decision == "deny":
			return "denied"
		return "active"

	@staticmethod
	def _require_text(value: str, field_name: str) -> str:
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f"{field_name} is required")
		return value.strip()

	@staticmethod
	def _require_choice(value: str, field_name: str, allowed: set[str]) -> str:
		text = ETLPLifecycleService._require_text(value, field_name)
		if text not in allowed:
			raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
		return text

	@staticmethod
	def _normalize_agent_token(value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	@staticmethod
	def _first_reason(result: dict[str, Any]) -> str:
		for action in result.get("actions", []):
			if action.get("reason"):
				return str(action["reason"])
		return "etlp_operation_denied"

	@staticmethod
	def _key(tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"


class ETLPService:
	"""Main ETLP service orchestrating pipeline operations"""
	
	def __init__(self, tenant_id: str, user_id: str):
		"""Initialize ETLP service with APG context"""
		assert tenant_id, "tenant_id is required for APG multi-tenancy"
		assert user_id, "user_id is required for APG audit trail"
		
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.connector_registry = ConnectorRegistry()
		self.transformation_engine = TransformationEngine()
		self.quality_engine = QualityEngine()
		self.execution_monitor = ExecutionMonitor()
		self.ai_optimizer = AIOptimizer()
		
		# APG capability integrations - will be injected
		self.metadata_service = None
		self.aicr_service = None
		self.auth_service = None
		self.audit_service = None
		self.notification_service = None
		self.collaboration_service = None
	
	async def _log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log info message with APG context"""
		timestamp = datetime.utcnow().isoformat()
		ctx = f" | {context}" if context else ""
		print(f"[{timestamp}] ETLP INFO [{self.tenant_id}:{self.user_id}]: {message}{ctx}")
	
	async def _log_error(self, message: str, error: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log error message with APG context"""
		timestamp = datetime.utcnow().isoformat()
		ctx = f" | {context}" if context else ""
		error_details = f" | Error: {str(error)}" if error else ""
		print(f"[{timestamp}] ETLP ERROR [{self.tenant_id}:{self.user_id}]: {message}{ctx}{error_details}")
		
		# Send to APG audit service if available
		if self.audit_service:
			await self._audit_error(message, error, context)
	
	async def _log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
		"""Log warning message with APG context"""
		timestamp = datetime.utcnow().isoformat()
		ctx = f" | {context}" if context else ""
		print(f"[{timestamp}] ETLP WARN [{self.tenant_id}:{self.user_id}]: {message}{ctx}")
	
	# Pipeline Management
	
	async def create_pipeline(self, pipeline_data: Dict[str, Any]) -> Pipeline:
		"""Create new pipeline with APG integration"""
		assert pipeline_data, "Pipeline data cannot be empty"
		
		try:
			# Add APG context
			pipeline_data.update({
				"tenant_id": self.tenant_id,
				"created_by": self.user_id,
				"id": uuid7str()
			})
			
			pipeline = Pipeline(**pipeline_data)
			
			# Validate dependencies
			issues = await validate_pipeline_dependencies(pipeline)
			if issues:
				await self._log_error(f"Pipeline validation failed: {issues}")
				raise ValueError(f"Pipeline validation failed: {', '.join(issues)}")
			
			# Calculate complexity
			complexity = await calculate_pipeline_complexity(pipeline)
			await self._log_info(f"Pipeline complexity: {complexity['complexity_score']}", {"pipeline_id": pipeline.id})
			
			# Register with APG metadata service
			if self.metadata_service:
				await self._register_pipeline_metadata(pipeline)
			
			# Audit trail
			if self.audit_service:
				await self._audit_pipeline_creation(pipeline)
			
			await self._log_info(f"Pipeline created: {pipeline.name}", {"pipeline_id": pipeline.id})
			return pipeline
			
		except Exception as e:
			await self._log_error("Pipeline creation failed", e)
			raise
	
	async def update_pipeline(self, pipeline_id: str, updates: Dict[str, Any]) -> Pipeline:
		"""Update existing pipeline"""
		assert pipeline_id, "Pipeline ID is required"
		assert updates, "Update data cannot be empty"
		
		try:
			# Load current pipeline (would be from database)
			pipeline = await self._load_pipeline(pipeline_id)
			if not pipeline:
				raise ValueError(f"Pipeline {pipeline_id} not found")
			
			# Check permissions
			await self._check_pipeline_permissions(pipeline, "write")
			
			# Apply updates
			old_version = pipeline.version
			for key, value in updates.items():
				if hasattr(pipeline, key):
					setattr(pipeline, key, value)
			
			pipeline.updated_at = datetime.utcnow()
			pipeline.updated_by = self.user_id
			
			# Version management
			if "steps" in updates or "transformations" in updates:
				await self._increment_pipeline_version(pipeline)
			
			# Validate updated pipeline
			issues = await validate_pipeline_dependencies(pipeline)
			if issues:
				raise ValueError(f"Pipeline validation failed: {', '.join(issues)}")
			
			# Update metadata
			if self.metadata_service:
				await self._update_pipeline_metadata(pipeline)
			
			# Audit trail
			if self.audit_service:
				await self._audit_pipeline_update(pipeline, updates, old_version)
			
			# Notify collaborators
			if self.collaboration_service and pipeline.collaboration_enabled:
				await self._notify_pipeline_update(pipeline, updates)
			
			await self._log_info(f"Pipeline updated: {pipeline.name}", {"pipeline_id": pipeline.id})
			return pipeline
			
		except Exception as e:
			await self._log_error(f"Pipeline update failed: {pipeline_id}", e)
			raise
	
	async def delete_pipeline(self, pipeline_id: str, hard_delete: bool = False) -> bool:
		"""Delete pipeline (soft delete by default)"""
		assert pipeline_id, "Pipeline ID is required"
		
		try:
			pipeline = await self._load_pipeline(pipeline_id)
			if not pipeline:
				raise ValueError(f"Pipeline {pipeline_id} not found")
			
			await self._check_pipeline_permissions(pipeline, "delete")
			
			# Check for running executions
			if await self._has_running_executions(pipeline_id):
				raise ValueError("Cannot delete pipeline with running executions")
			
			if hard_delete:
				# Permanently delete pipeline
				await self._hard_delete_pipeline(pipeline)
			else:
				# Soft delete
				pipeline.is_deleted = True
				pipeline.deleted_at = datetime.utcnow()
				pipeline.deleted_by = self.user_id
			
			# Update metadata service
			if self.metadata_service:
				await self._remove_pipeline_metadata(pipeline)
			
			# Audit trail
			if self.audit_service:
				await self._audit_pipeline_deletion(pipeline, hard_delete)
			
			await self._log_info(f"Pipeline deleted: {pipeline.name}", {"pipeline_id": pipeline.id})
			return True
			
		except Exception as e:
			await self._log_error(f"Pipeline deletion failed: {pipeline_id}", e)
			raise
	
	async def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
		"""Retrieve pipeline by ID"""
		assert pipeline_id, "Pipeline ID is required"
		
		try:
			pipeline = await self._load_pipeline(pipeline_id)
			if pipeline:
				await self._check_pipeline_permissions(pipeline, "read")
			return pipeline
		except Exception as e:
			await self._log_error(f"Pipeline retrieval failed: {pipeline_id}", e)
			raise
	
	async def list_pipelines(self, filters: Optional[Dict[str, Any]] = None, 
							 limit: int = 100, offset: int = 0) -> List[Pipeline]:
		"""List pipelines with filtering and pagination"""
		assert limit > 0, "Limit must be positive"
		assert offset >= 0, "Offset must be non-negative"
		
		try:
			# Apply tenant filter
			filters = filters or {}
			filters["tenant_id"] = self.tenant_id
			
			# Load pipelines (would be from database)
			pipelines = await self._load_pipelines(filters, limit, offset)
			
			# Filter by permissions
			accessible_pipelines = []
			for pipeline in pipelines:
				try:
					await self._check_pipeline_permissions(pipeline, "read")
					accessible_pipelines.append(pipeline)
				except PermissionError:
					continue
			
			return accessible_pipelines
			
		except Exception as e:
			await self._log_error("Pipeline listing failed", e)
			raise
	
	# Pipeline Execution
	
	async def execute_pipeline(self, pipeline_id: str, config: Optional[Dict[str, Any]] = None,
							   execution_mode: Optional[ExecutionMode] = None) -> str:
		"""Execute pipeline and return execution ID"""
		assert pipeline_id, "Pipeline ID is required"
		
		try:
			pipeline = await self._load_pipeline(pipeline_id)
			if not pipeline:
				raise ValueError(f"Pipeline {pipeline_id} not found")
			
			await self._check_pipeline_permissions(pipeline, "execute")
			
			if pipeline.status not in [PipelineStatus.ACTIVE, PipelineStatus.DRAFT]:
				raise ValueError(f"Pipeline status {pipeline.status} does not allow execution")
			
			# Create execution record
			execution_id = uuid7str()
			execution = Execution(
				id=execution_id,
				pipeline_id=pipeline_id,
				tenant_id=self.tenant_id,
				status=PipelineStatus.RUNNING,
				execution_mode=execution_mode or pipeline.execution_mode,
				triggered_by=self.user_id,
				trigger_type="manual",
				pipeline_version=pipeline.version,
				configuration=config or {},
				started_at=datetime.utcnow()
			)
			
			# Start execution asynchronously
			asyncio.create_task(self._execute_pipeline_async(pipeline, execution))
			
			await self._log_info(f"Pipeline execution started: {pipeline.name}", 
								{"pipeline_id": pipeline_id, "execution_id": execution_id})
			return execution_id
			
		except Exception as e:
			await self._log_error(f"Pipeline execution start failed: {pipeline_id}", e)
			raise
	
	async def _execute_pipeline_async(self, pipeline: Pipeline, execution: Execution) -> None:
		"""Asynchronously execute pipeline"""
		metrics = PipelineMetrics()
		start_time = datetime.utcnow()
		
		try:
			await self._log_info(f"Executing pipeline: {pipeline.name}", {"execution_id": execution.id})
			
			# AI optimization
			if pipeline.ai_optimization_enabled and self.aicr_service:
				await self._optimize_pipeline_execution(pipeline, execution)
			
			# Process each step
			for step_index, step in enumerate(pipeline.steps):
				await self._execute_pipeline_step(pipeline, execution, step_index, step, metrics)
			
			# Apply data quality rules
			if pipeline.quality_rules:
				await self._apply_quality_rules(pipeline, execution, metrics)
			
			# Calculate final metrics
			execution.completed_at = datetime.utcnow()
			execution.duration_ms = int((execution.completed_at - start_time).total_seconds() * 1000)
			execution.status = PipelineStatus.SUCCESS
			execution.metrics = metrics.__dict__
			
			# Update lineage
			if pipeline.lineage_tracked and self.metadata_service:
				await self._update_pipeline_lineage(pipeline, execution)
			
			# Success notification
			if pipeline.alert_on_failure and self.notification_service:
				await self._send_success_notification(pipeline, execution)
			
			await self._log_info(f"Pipeline execution completed: {pipeline.name}", 
								{"execution_id": execution.id, "duration_ms": execution.duration_ms})
			
		except Exception as e:
			# Handle execution failure
			execution.status = PipelineStatus.FAILED
			execution.error_message = str(e)
			execution.error_details = {"exception_type": type(e).__name__}
			execution.stack_trace = traceback.format_exc()
			execution.completed_at = datetime.utcnow()
			execution.duration_ms = int((execution.completed_at - start_time).total_seconds() * 1000)
			
			# Failure notification
			if pipeline.alert_on_failure and self.notification_service:
				await self._send_failure_notification(pipeline, execution, e)
			
			# Auto-retry if configured
			if pipeline.retry_count > 0:
				await self._schedule_retry(pipeline, execution)
			
			await self._log_error(f"Pipeline execution failed: {pipeline.name}", e, 
								 {"execution_id": execution.id})
	
	async def _execute_pipeline_step(self, pipeline: Pipeline, execution: Execution,
									 step_index: int, step: Dict[str, Any], metrics: PipelineMetrics) -> None:
		"""Execute individual pipeline step"""
		step_start = datetime.utcnow()
		
		try:
			step_type = step.get("type", "unknown")
			await self._log_info(f"Executing step {step_index}: {step_type}", {"execution_id": execution.id})
			
			if step_type == "extract":
				await self._execute_extract_step(step, metrics)
			elif step_type == "transform":
				await self._execute_transform_step(step, metrics)
			elif step_type == "load":
				await self._execute_load_step(step, metrics)
			elif step_type == "quality_check":
				await self._execute_quality_step(step, metrics)
			else:
				await self._execute_custom_step(step, metrics)
			
			step_duration = int((datetime.utcnow() - step_start).total_seconds() * 1000)
			metrics.processing_time_ms += step_duration
			
		except Exception as e:
			await self._log_error(f"Step {step_index} failed", e, {"execution_id": execution.id})
			raise
	
	# Transformation Management
	
	async def create_transformation(self, transform_data: Dict[str, Any]) -> Transformation:
		"""Create new data transformation"""
		assert transform_data, "Transformation data cannot be empty"
		
		try:
			transform_data.update({
				"tenant_id": self.tenant_id,
				"created_by": self.user_id,
				"id": uuid7str()
			})
			
			transformation = Transformation(**transform_data)
			
			# Validate transformation logic
			await self._validate_transformation(transformation)
			
			# Register with transformation engine
			await self.transformation_engine.register_transformation(transformation)
			
			await self._log_info(f"Transformation created: {transformation.name}", 
								{"transformation_id": transformation.id})
			return transformation
			
		except Exception as e:
			await self._log_error("Transformation creation failed", e)
			raise
	
	# Data Source Management
	
	async def create_data_source(self, source_data: Dict[str, Any]) -> DataSource:
		"""Create new data source connection"""
		assert source_data, "Data source data cannot be empty"
		
		try:
			source_data.update({
				"tenant_id": self.tenant_id,
				"created_by": self.user_id,
				"id": uuid7str()
			})
			
			data_source = DataSource(**source_data)
			
			# Test connection
			is_healthy = await self._test_data_source_connection(data_source)
			data_source.is_healthy = is_healthy
			data_source.last_health_check = datetime.utcnow()
			
			# Register with connector registry
			await self.connector_registry.register_data_source(data_source)
			
			# Sync with APG metadata service
			if data_source.metadata_sync_enabled and self.metadata_service:
				await self._sync_data_source_metadata(data_source)
			
			await self._log_info(f"Data source created: {data_source.name}", 
								{"data_source_id": data_source.id, "healthy": is_healthy})
			return data_source
			
		except Exception as e:
			await self._log_error("Data source creation failed", e)
			raise
	
	async def test_data_source(self, source_id: str) -> Dict[str, Any]:
		"""Test data source connection health"""
		assert source_id, "Data source ID is required"
		
		try:
			data_source = await self._load_data_source(source_id)
			if not data_source:
				raise ValueError(f"Data source {source_id} not found")
			
			start_time = datetime.utcnow()
			is_healthy = await self._test_data_source_connection(data_source)
			response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			# Update health status
			data_source.is_healthy = is_healthy
			data_source.last_health_check = datetime.utcnow()
			
			await data_source._log_health_check(is_healthy, response_time_ms)
			
			return {
				"healthy": is_healthy,
				"response_time_ms": response_time_ms,
				"last_check": data_source.last_health_check.isoformat()
			}
			
		except Exception as e:
			await self._log_error(f"Data source test failed: {source_id}", e)
			raise
	
	# Quality Rule Management
	
	async def create_quality_rule(self, rule_data: Dict[str, Any]) -> QualityRule:
		"""Create new data quality rule"""
		assert rule_data, "Quality rule data cannot be empty"
		
		try:
			rule_data.update({
				"tenant_id": self.tenant_id,
				"created_by": self.user_id,
				"id": uuid7str()
			})
			
			quality_rule = QualityRule(**rule_data)
			
			# Validate rule logic
			await self._validate_quality_rule(quality_rule)
			
			# Register with quality engine
			await self.quality_engine.register_quality_rule(quality_rule)
			
			await self._log_info(f"Quality rule created: {quality_rule.name}", 
								{"quality_rule_id": quality_rule.id})
			return quality_rule
			
		except Exception as e:
			await self._log_error("Quality rule creation failed", e)
			raise
	
	# Execution Monitoring
	
	async def get_execution(self, execution_id: str) -> Optional[Execution]:
		"""Get execution details"""
		assert execution_id, "Execution ID is required"
		
		try:
			execution = await self._load_execution(execution_id)
			if execution and execution.tenant_id != self.tenant_id:
				return None  # Cross-tenant access denied
			return execution
		except Exception as e:
			await self._log_error(f"Execution retrieval failed: {execution_id}", e)
			raise
	
	async def list_executions(self, pipeline_id: Optional[str] = None, 
							  status: Optional[PipelineStatus] = None,
							  limit: int = 100, offset: int = 0) -> List[Execution]:
		"""List pipeline executions with filtering"""
		assert limit > 0, "Limit must be positive"
		assert offset >= 0, "Offset must be non-negative"
		
		try:
			filters = {"tenant_id": self.tenant_id}
			if pipeline_id:
				filters["pipeline_id"] = pipeline_id
			if status:
				filters["status"] = status
			
			executions = await self._load_executions(filters, limit, offset)
			return executions
			
		except Exception as e:
			await self._log_error("Execution listing failed", e)
			raise
	
	async def cancel_execution(self, execution_id: str) -> bool:
		"""Cancel running pipeline execution"""
		assert execution_id, "Execution ID is required"
		
		try:
			execution = await self._load_execution(execution_id)
			if not execution:
				raise ValueError(f"Execution {execution_id} not found")
			
			if execution.tenant_id != self.tenant_id:
				raise PermissionError("Access denied")
			
			if execution.status != PipelineStatus.RUNNING:
				raise ValueError(f"Cannot cancel execution with status {execution.status}")
			
			# Signal cancellation
			await self._signal_execution_cancellation(execution_id)
			
			# Update execution record
			execution.status = PipelineStatus.CANCELLED
			execution.completed_at = datetime.utcnow()
			execution.duration_ms = int((execution.completed_at - execution.started_at).total_seconds() * 1000)
			
			await self._log_info(f"Execution cancelled", {"execution_id": execution_id})
			return True
			
		except Exception as e:
			await self._log_error(f"Execution cancellation failed: {execution_id}", e)
			raise
	
	# AI-Powered Optimization
	
	async def optimize_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
		"""Use AI to optimize pipeline performance"""
		assert pipeline_id, "Pipeline ID is required"
		
		try:
			pipeline = await self._load_pipeline(pipeline_id)
			if not pipeline:
				raise ValueError(f"Pipeline {pipeline_id} not found")
			
			if not self.aicr_service:
				raise ValueError("AI/CR service not available")
			
			# Analyze historical performance
			executions = await self._load_executions({"pipeline_id": pipeline_id}, limit=50)
			performance_data = await self._analyze_pipeline_performance(executions)
			
			# Get AI recommendations
			recommendations = await self.ai_optimizer.generate_recommendations(pipeline, performance_data)
			
			# Apply automatic optimizations if enabled
			if pipeline.ai_optimization_enabled:
				await self._apply_ai_optimizations(pipeline, recommendations)
			
			await self._log_info(f"Pipeline optimization completed", {"pipeline_id": pipeline_id})
			return recommendations
			
		except Exception as e:
			await self._log_error(f"Pipeline optimization failed: {pipeline_id}", e)
			raise
	
	# Collaboration Features
	
	async def get_pipeline_collaborators(self, pipeline_id: str) -> List[Dict[str, Any]]:
		"""Get list of pipeline collaborators"""
		assert pipeline_id, "Pipeline ID is required"
		
		try:
			pipeline = await self._load_pipeline(pipeline_id)
			if not pipeline:
				raise ValueError(f"Pipeline {pipeline_id} not found")
			
			if self.collaboration_service:
				collaborators = await self._get_pipeline_collaborators(pipeline_id)
				return collaborators
			
			return []
			
		except Exception as e:
			await self._log_error(f"Collaborator retrieval failed: {pipeline_id}", e)
			raise
	
	# Helper methods (would interact with actual database and services)
	
	async def _load_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
		"""Load pipeline from database"""
		# Mock implementation - would query database
		return None
	
	async def _load_pipelines(self, filters: Dict[str, Any], limit: int, offset: int) -> List[Pipeline]:
		"""Load pipelines from database with filters"""
		# Mock implementation - would query database
		return []
	
	async def _load_execution(self, execution_id: str) -> Optional[Execution]:
		"""Load execution from database"""
		# Mock implementation - would query database
		return None
	
	async def _load_executions(self, filters: Dict[str, Any], limit: int, offset: int) -> List[Execution]:
		"""Load executions from database with filters"""
		# Mock implementation - would query database
		return []
	
	async def _load_data_source(self, source_id: str) -> Optional[DataSource]:
		"""Load data source from database"""
		# Mock implementation - would query database
		return None
	
	async def _check_pipeline_permissions(self, pipeline: Pipeline, action: str) -> None:
		"""Check if user has permission to perform action on pipeline"""
		if self.auth_service:
			# Use APG RBAC service
			permission = f"etlp:pipeline:{action}"
			has_permission = await self.auth_service.check_permission(self.user_id, permission, pipeline.id)
			if not has_permission:
				raise PermissionError(f"Access denied for action: {action}")
		
		# Basic ownership check
		if pipeline.tenant_id != self.tenant_id:
			raise PermissionError("Cross-tenant access denied")
	
	async def _increment_pipeline_version(self, pipeline: Pipeline) -> None:
		"""Increment pipeline version"""
		parts = pipeline.version.split('.')
		parts[2] = str(int(parts[2]) + 1)
		pipeline.version = '.'.join(parts)
	
	async def _has_running_executions(self, pipeline_id: str) -> bool:
		"""Check if pipeline has running executions"""
		executions = await self._load_executions(
			{"pipeline_id": pipeline_id, "status": PipelineStatus.RUNNING}, 
			limit=1, offset=0
		)
		return len(executions) > 0
	
	async def _test_data_source_connection(self, data_source: DataSource) -> bool:
		"""Test data source connection"""
		try:
			# Use connector registry to test connection
			connector = await self.connector_registry.get_connector(data_source.type)
			if connector:
				return await connector.test_connection(data_source)
			return False
		except Exception as e:
			await self._log_error(f"Connection test failed for {data_source.id}", e)
			return False
	
	# APG Integration helpers
	
	async def _audit_pipeline_creation(self, pipeline: Pipeline) -> None:
		"""Record pipeline creation in APG audit trail"""
		if self.audit_service:
			await self.audit_service.log_event(
				"pipeline_created",
				{"pipeline_id": pipeline.id, "name": pipeline.name},
				self.user_id
			)
	
	async def _audit_pipeline_update(self, pipeline: Pipeline, updates: Dict[str, Any], old_version: str) -> None:
		"""Record pipeline update in APG audit trail"""
		if self.audit_service:
			await self.audit_service.log_event(
				"pipeline_updated",
				{
					"pipeline_id": pipeline.id,
					"updates": updates,
					"old_version": old_version,
					"new_version": pipeline.version
				},
				self.user_id
			)
	
	async def _audit_pipeline_deletion(self, pipeline: Pipeline, hard_delete: bool) -> None:
		"""Record pipeline deletion in APG audit trail"""
		if self.audit_service:
			await self.audit_service.log_event(
				"pipeline_deleted",
				{"pipeline_id": pipeline.id, "name": pipeline.name, "hard_delete": hard_delete},
				self.user_id
			)
	
	async def _audit_error(self, message: str, error: Optional[Exception], context: Optional[Dict[str, Any]]) -> None:
		"""Record error in APG audit trail"""
		if self.audit_service:
			await self.audit_service.log_event(
				"error_occurred",
				{
					"message": message,
					"error": str(error) if error else None,
					"context": context,
					"stack_trace": traceback.format_exc() if error else None
				},
				self.user_id
			)
	
	# Additional service methods for API integration
	
	async def test_data_source_connection(self, source_id: str) -> Dict[str, Any]:
		"""Test connection to a data source
		
		Attempts to establish a connection to the specified data source and
		returns the connection status and any error details.
		
		Args:
			source_id: ID of the data source to test
			
		Returns:
			Dict containing connection test results with success status and details
			
		Raises:
			ValueError: If data source is not found
			PermissionError: If access is denied
		"""
		assert source_id, "Data source ID is required"
		
		try:
			# Load data source
			data_source = await self._load_data_source(source_id)
			if not data_source:
				raise ValueError(f"Data source {source_id} not found")
			
			# Check permissions
			if data_source.tenant_id != self.tenant_id:
				raise PermissionError("Access denied")
			
			# Test connection using connector
			connector = await self.connector_registry.get_connector(data_source.type)
			if not connector:
				return {
					"success": False,
					"error": f"No connector available for type {data_source.type}",
					"tested_at": datetime.utcnow().isoformat()
				}
			
			# Perform connection test
			connection_result = await connector.test_connection(data_source)
			
			return {
				"success": connection_result,
				"source_id": source_id,
				"source_type": str(data_source.type),
				"tested_at": datetime.utcnow().isoformat(),
				"response_time_ms": getattr(connector, 'last_response_time', None)
			}
			
		except Exception as e:
			self._log_error(f"Connection test failed for {source_id}: {str(e)}")
			return {
				"success": False,
				"error": str(e),
				"source_id": source_id,
				"tested_at": datetime.utcnow().isoformat()
			}
	
	async def save_field_mapping_configuration(self, config: Any) -> str:
		"""Save field mapping configuration to database
		
		Stores a field mapping configuration for later retrieval and execution.
		Includes validation and tenant isolation.
		
		Args:
			config: MappingConfiguration instance to save
			
		Returns:
			str: ID of the saved configuration
			
		Raises:
			ValueError: If configuration is invalid
		"""
		try:
			# Generate unique ID
			from uuid_extensions import uuid7str
			config_id = uuid7str()
			
			# Add metadata
			config_data = {
				"id": config_id,
				"tenant_id": self.tenant_id,
				"created_by": self.user_id,
				"created_at": datetime.utcnow().isoformat(),
				"configuration": config.model_dump() if hasattr(config, 'model_dump') else config
			}
			
			# Store configuration (would use actual database in production)
			# For now, store in memory registry
			if not hasattr(self, '_field_mapping_configs'):
				self._field_mapping_configs = {}
			
			self._field_mapping_configs[config_id] = config_data
			
			self._log_info(f"Saved field mapping configuration: {config_id}")
			
			return config_id
			
		except Exception as e:
			self._log_error(f"Failed to save field mapping configuration: {str(e)}")
			raise
	
	async def validate_quality_rule_logic(self, rule_id: str) -> Dict[str, Any]:
		"""Validate quality rule logic and syntax
		
		Checks if the quality rule's validation logic is syntactically correct
		and can be executed without errors.
		
		Args:
			rule_id: ID of the quality rule to validate
			
		Returns:
			Dict containing validation results
			
		Raises:
			ValueError: If rule is not found
		"""
		assert rule_id, "Quality rule ID is required"
		
		try:
			# Load quality rule
			rule = await self._load_quality_rule(rule_id)
			if not rule:
				raise ValueError(f"Quality rule {rule_id} not found")
			
			# Check permissions
			if rule.tenant_id != self.tenant_id:
				raise PermissionError("Access denied")
			
			validation_result = {
				"valid": True,
				"errors": [],
				"warnings": [],
				"rule_id": rule_id
			}
			
			# Validate condition syntax
			if rule.condition:
				try:
					# Basic syntax validation for SQL-like conditions
					condition_str = str(rule.condition)
					if not condition_str.strip():
						validation_result["errors"].append("Empty condition")
						validation_result["valid"] = False
					
					# Check for dangerous patterns
					dangerous_patterns = ['drop', 'delete', 'truncate', 'alter']
					for pattern in dangerous_patterns:
						if pattern.lower() in condition_str.lower():
							validation_result["warnings"].append(f"Potentially dangerous keyword: {pattern}")
					
				except Exception as e:
					validation_result["errors"].append(f"Condition syntax error: {str(e)}")
					validation_result["valid"] = False
			
			# Validate validation logic
			if rule.validation_logic:
				try:
					# Basic validation of validation logic structure
					if isinstance(rule.validation_logic, dict):
						required_keys = ['type', 'parameters']
						missing_keys = [key for key in required_keys if key not in rule.validation_logic]
						if missing_keys:
							validation_result["errors"].append(f"Missing required keys: {missing_keys}")
							validation_result["valid"] = False
					else:
						validation_result["warnings"].append("Validation logic should be a dictionary")
				
				except Exception as e:
					validation_result["errors"].append(f"Validation logic error: {str(e)}")
					validation_result["valid"] = False
			
			return validation_result
			
		except Exception as e:
			self._log_error(f"Quality rule validation failed for {rule_id}: {str(e)}")
			return {
				"valid": False,
				"errors": [str(e)],
				"warnings": [],
				"rule_id": rule_id
			}
	
	def _increment_version(self, version: str) -> str:
		"""Increment semantic version string
		
		Takes a semantic version (e.g., "1.2.3") and increments the patch version.
		
		Args:
			version: Current version string
			
		Returns:
			str: Incremented version string
		"""
		try:
			parts = version.split('.')
			if len(parts) != 3:
				return "1.0.1"  # Default if invalid format
			
			major, minor, patch = parts
			new_patch = str(int(patch) + 1)
			return f"{major}.{minor}.{new_patch}"
			
		except (ValueError, IndexError):
			return "1.0.1"  # Default if parsing fails
	
	async def _load_data_source(self, source_id: str) -> Optional[Any]:
		"""Load data source by ID (mock implementation)
		
		In production, this would query the database for the data source.
		
		Args:
			source_id: Data source ID to load
			
		Returns:
			DataSource instance or None if not found
		"""
		# Mock implementation - would use actual database
		return None
	
	async def _load_quality_rule(self, rule_id: str) -> Optional[Any]:
		"""Load quality rule by ID (mock implementation)
		
		In production, this would query the database for the quality rule.
		
		Args:
			rule_id: Quality rule ID to load
			
		Returns:
			QualityRule instance or None if not found
		"""
		# Mock implementation - would use actual database
		return None


# Supporting service classes

class ConnectorRegistry:
	"""Registry for data source connectors"""
	
	def __init__(self):
		self.connectors = {}
	
	async def register_data_source(self, data_source: DataSource) -> None:
		"""Register data source in the connector registry
		
		Stores the data source configuration and initializes the appropriate
		connector for future use in pipeline executions.
		
		Args:
			data_source: DataSource instance to register
			
		Raises:
			ValueError: If data source type is not supported
		"""
		try:
			# Validate data source type
			if not hasattr(data_source.type, 'value'):
				connector_type = str(data_source.type)
			else:
				connector_type = data_source.type.value
			
			if connector_type not in self.connectors:
				raise ValueError(f"Unsupported data source type: {connector_type}")
			
			# Store data source configuration
			self.data_sources[data_source.id] = {
				'config': data_source,
				'connector': self.connectors[connector_type],
				'registered_at': datetime.utcnow(),
				'health_status': 'unknown',
				'last_health_check': None
			}
			
			# Initialize connector if needed
			connector = self.connectors[connector_type]
			if hasattr(connector, 'initialize'):
				await connector.initialize(data_source)
			
			self._log_debug(f"Registered data source: {data_source.name} ({connector_type})")
			
		except Exception as e:
			self._log_error(f"Failed to register data source {data_source.name}: {str(e)}")
			raise
	
	async def get_connector(self, source_type: str) -> Optional[Any]:
		"""Get connector for source type"""
		return self.connectors.get(source_type)


class TransformationEngine:
	"""Engine for executing data transformations"""
	
	def __init__(self):
		self.registered_transformations = {}
	
	async def register_transformation(self, transformation: Transformation) -> None:
		"""Register transformation"""
		self.registered_transformations[transformation.id] = transformation
	
	async def execute_transformation(self, transform_id: str, data: Any) -> Any:
		"""Execute transformation on data"""
		transformation = self.registered_transformations.get(transform_id)
		if not transformation:
			raise ValueError(f"Transformation {transform_id} not found")
		
		# Execute transformation logic
		return await self._apply_transformation(transformation, data)
	
	async def _apply_transformation(self, transformation: Transformation, data: Any) -> Any:
		"""Apply transformation logic to data"""
		# Implementation would execute transformation
		return data


class QualityEngine:
	"""Engine for data quality validation"""
	
	def __init__(self):
		self.registered_rules = {}
	
	async def register_quality_rule(self, rule: QualityRule) -> None:
		"""Register quality rule"""
		self.registered_rules[rule.id] = rule
	
	async def validate_data(self, data: Any, rule_ids: List[str]) -> Dict[str, Any]:
		"""Validate data against quality rules"""
		results = {
			"passed": 0,
			"failed": 0,
			"violations": []
		}
		
		for rule_id in rule_ids:
			rule = self.registered_rules.get(rule_id)
			if rule and rule.enabled:
				violations = await self._apply_quality_rule(rule, data)
				if violations:
					results["failed"] += 1
					results["violations"].extend(violations)
				else:
					results["passed"] += 1
		
		return results
	
	async def _apply_quality_rule(self, rule: QualityRule, data: Any) -> List[Dict[str, Any]]:
		"""Apply quality rule to data and return violations
		
		Executes the quality rule validation logic against the provided data
		and returns a list of violations found.
		
		Args:
			rule: QualityRule to apply
			data: Data to validate (could be a record, DataFrame, etc.)
			
		Returns:
			List of violation dictionaries, empty if no violations
		"""
		violations = []
		
		try:
			# Handle different rule types
			if hasattr(rule.type, 'value'):
				rule_type = rule.type.value
			else:
				rule_type = str(rule.type)
			
			# Apply sampling if configured
			sample_rate = rule.sample_percentage / 100.0
			if sample_rate < 1.0:
				import random
				if random.random() > sample_rate:
					return violations  # Skip validation for this sample
			
			# Execute rule based on type
			if rule_type == 'not_null':
				violations.extend(await self._validate_not_null(rule, data))
			elif rule_type == 'range':
				violations.extend(await self._validate_range(rule, data))
			elif rule_type == 'format':
				violations.extend(await self._validate_format(rule, data))
			elif rule_type == 'uniqueness':
				violations.extend(await self._validate_uniqueness(rule, data))
			elif rule_type == 'custom':
				violations.extend(await self._validate_custom(rule, data))
			else:
				# Generic validation using condition
				violations.extend(await self._validate_condition(rule, data))
		
		except Exception as e:
			# Log validation error and treat as violation
			violation = {
				"rule_id": rule.id,
				"rule_name": rule.name,
				"field_name": rule.field_name,
				"violation_type": "validation_error",
				"message": f"Error executing rule: {str(e)}",
				"severity": rule.severity,
				"timestamp": datetime.utcnow().isoformat(),
				"suggested_fix": rule.suggested_fix
			}
			violations.append(violation)
		
		return violations
	
	async def _validate_not_null(self, rule: QualityRule, data: Any) -> List[Dict[str, Any]]:
		"""Validate not null constraint"""
		violations = []
		
		# Handle different data structures
		if hasattr(data, 'get'):  # Dictionary-like
			value = data.get(rule.field_name)
		elif hasattr(data, rule.field_name):  # Object with attributes
			value = getattr(data, rule.field_name)
		else:
			return violations  # Cannot validate without field
		
		if value is None or (isinstance(value, str) and value.strip() == ""):
			violations.append({
				"rule_id": rule.id,
				"rule_name": rule.name,
				"field_name": rule.field_name,
				"violation_type": "null_value",
				"message": rule.error_message or f"Field {rule.field_name} cannot be null",
				"severity": rule.severity,
				"timestamp": datetime.utcnow().isoformat(),
				"suggested_fix": rule.suggested_fix
			})
		
		return violations
	
	async def _validate_range(self, rule: QualityRule, data: Any) -> List[Dict[str, Any]]:
		"""Validate numeric range constraint"""
		violations = []
		
		try:
			# Extract field value
			if hasattr(data, 'get'):
				value = data.get(rule.field_name)
			elif hasattr(data, rule.field_name):
				value = getattr(data, rule.field_name)
			else:
				return violations
			
			if value is None:
				return violations
			
			# Get range parameters from condition
			condition = rule.condition or {}
			min_val = condition.get('min')
			max_val = condition.get('max')
			
			numeric_value = float(value)
			
			if min_val is not None and numeric_value < min_val:
				violations.append({
					"rule_id": rule.id,
					"rule_name": rule.name,
					"field_name": rule.field_name,
					"violation_type": "range_violation",
					"message": f"Value {numeric_value} is below minimum {min_val}",
					"severity": rule.severity,
					"timestamp": datetime.utcnow().isoformat(),
					"suggested_fix": rule.suggested_fix
				})
			
			if max_val is not None and numeric_value > max_val:
				violations.append({
					"rule_id": rule.id,
					"rule_name": rule.name,
					"field_name": rule.field_name,
					"violation_type": "range_violation",
					"message": f"Value {numeric_value} exceeds maximum {max_val}",
					"severity": rule.severity,
					"timestamp": datetime.utcnow().isoformat(),
					"suggested_fix": rule.suggested_fix
				})
		
		except (ValueError, TypeError):
			violations.append({
				"rule_id": rule.id,
				"rule_name": rule.name,
				"field_name": rule.field_name,
				"violation_type": "type_error",
				"message": f"Value cannot be converted to number for range validation",
				"severity": rule.severity,
				"timestamp": datetime.utcnow().isoformat(),
				"suggested_fix": rule.suggested_fix
			})
		
		return violations
	
	async def _validate_format(self, rule: QualityRule, data: Any) -> List[Dict[str, Any]]:
		"""Validate format constraint using regex"""
		violations = []
		
		try:
			# Extract field value
			if hasattr(data, 'get'):
				value = data.get(rule.field_name)
			elif hasattr(data, rule.field_name):
				value = getattr(data, rule.field_name)
			else:
				return violations
			
			if value is None:
				return violations
			
			# Get pattern from condition
			condition = rule.condition or {}
			pattern = condition.get('pattern')
			
			if pattern:
				import re
				if not re.match(pattern, str(value)):
					violations.append({
						"rule_id": rule.id,
						"rule_name": rule.name,
						"field_name": rule.field_name,
						"violation_type": "format_violation",
						"message": rule.error_message or f"Value '{value}' does not match required format",
						"severity": rule.severity,
						"timestamp": datetime.utcnow().isoformat(),
						"suggested_fix": rule.suggested_fix
					})
		
		except Exception as e:
			violations.append({
				"rule_id": rule.id,
				"rule_name": rule.name,
				"field_name": rule.field_name,
				"violation_type": "validation_error",
				"message": f"Format validation error: {str(e)}",
				"severity": rule.severity,
				"timestamp": datetime.utcnow().isoformat(),
				"suggested_fix": rule.suggested_fix
			})
		
		return violations
	
	async def _validate_uniqueness(self, rule: QualityRule, data: Any) -> List[Dict[str, Any]]:
		"""Validate uniqueness constraint"""
		# This would require access to full dataset for comparison
		# For now, return empty as this needs dataset-level validation
		return []
	
	async def _validate_custom(self, rule: QualityRule, data: Any) -> List[Dict[str, Any]]:
		"""Validate using custom logic"""
		violations = []
		
		try:
			# Execute custom validation logic
			validation_logic = rule.validation_logic or {}
			logic_type = validation_logic.get('type')
			
			if logic_type == 'python_expression':
				# Evaluate Python expression (in secure sandbox)
				expression = validation_logic.get('expression', 'True')
				field_value = data.get(rule.field_name) if hasattr(data, 'get') else getattr(data, rule.field_name, None)
				
				# Create safe evaluation context
				safe_context = {'value': field_value, 'data': data}
				
				try:
					# Evaluate expression (this should be sandboxed in production)
					result = eval(expression, {"__builtins__": {}}, safe_context)
					if not result:
						violations.append({
							"rule_id": rule.id,
							"rule_name": rule.name,
							"field_name": rule.field_name,
							"violation_type": "custom_validation",
							"message": rule.error_message or "Custom validation failed",
							"severity": rule.severity,
							"timestamp": datetime.utcnow().isoformat(),
							"suggested_fix": rule.suggested_fix
						})
				except Exception as eval_error:
					violations.append({
						"rule_id": rule.id,
						"rule_name": rule.name,
						"field_name": rule.field_name,
						"violation_type": "evaluation_error",
						"message": f"Custom validation error: {str(eval_error)}",
						"severity": rule.severity,
						"timestamp": datetime.utcnow().isoformat(),
						"suggested_fix": rule.suggested_fix
					})
		
		except Exception as e:
			violations.append({
				"rule_id": rule.id,
				"rule_name": rule.name,
				"field_name": rule.field_name,
				"violation_type": "validation_error",
				"message": f"Custom validation setup error: {str(e)}",
				"severity": rule.severity,
				"timestamp": datetime.utcnow().isoformat(),
				"suggested_fix": rule.suggested_fix
			})
		
		return violations
	
	async def _validate_condition(self, rule: QualityRule, data: Any) -> List[Dict[str, Any]]:
		"""Validate using generic condition"""
		# Generic condition-based validation
		# This would implement a SQL-like condition evaluator
		return []


class ExecutionMonitor:
	"""Monitor pipeline execution progress"""
	
	def __init__(self):
		self.active_executions = {}
	
	async def start_monitoring(self, execution_id: str) -> None:
		"""Start monitoring execution"""
		self.active_executions[execution_id] = {
			"start_time": datetime.utcnow(),
			"status": "running"
		}
	
	async def update_execution_metrics(self, execution_id: str, metrics: Dict[str, Any]) -> None:
		"""Update execution metrics"""
		if execution_id in self.active_executions:
			self.active_executions[execution_id]["metrics"] = metrics
	
	async def stop_monitoring(self, execution_id: str) -> None:
		"""Stop monitoring execution"""
		if execution_id in self.active_executions:
			del self.active_executions[execution_id]


class AIOptimizer:
	"""AI-powered pipeline optimization"""
	
	def __init__(self):
		self.optimization_history = {}
	
	async def generate_recommendations(self, pipeline: Pipeline, 
									   performance_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate AI-powered optimization recommendations"""
		recommendations = {
			"performance_improvements": [],
			"resource_optimizations": [],
			"reliability_enhancements": [],
			"cost_optimizations": []
		}
		
		# Analyze performance patterns
		if performance_data.get("avg_duration_ms", 0) > 300000:  # > 5 minutes
			recommendations["performance_improvements"].append({
				"type": "parallelization",
				"description": "Increase max_parallelism to reduce execution time",
				"impact": "high",
				"estimated_improvement": "30-50% faster execution"
			})
		
		if performance_data.get("failure_rate", 0) > 0.1:  # > 10% failure rate
			recommendations["reliability_enhancements"].append({
				"type": "retry_policy",
				"description": "Implement exponential backoff retry strategy",
				"impact": "medium",
				"estimated_improvement": "50% reduction in transient failures"
			})
		
		return recommendations
	
	async def apply_optimization(self, pipeline: Pipeline, optimization: Dict[str, Any]) -> None:
		"""Apply AI-recommended optimization to pipeline
		
		Implements the specific optimization recommendations from the AI optimizer
		by updating pipeline configuration and parameters.
		
		Args:
			pipeline: Pipeline to optimize
			optimization: Optimization recommendation details
			
		Raises:
			ValueError: If optimization type is not supported
		"""
		try:
			opt_type = optimization.get('type')
			self._log_info(f"Applying optimization {opt_type} to pipeline {pipeline.name}")
			
			if opt_type == 'parallelization':
				# Update parallelism settings
				new_parallelism = optimization.get('recommended_parallelism', pipeline.max_parallelism)
				pipeline.max_parallelism = min(new_parallelism, 100)  # Cap at 100
				pipeline.configuration['parallel_workers'] = pipeline.max_parallelism
				
			elif opt_type == 'batch_size':
				# Update batch processing settings
				new_batch_size = optimization.get('recommended_batch_size', 1000)
				pipeline.configuration['batch_size'] = new_batch_size
				
			elif opt_type == 'memory_optimization':
				# Update memory settings
				memory_settings = optimization.get('memory_settings', {})
				pipeline.configuration.update(memory_settings)
				
			elif opt_type == 'execution_order':
				# Reorder pipeline steps for optimal execution
				optimal_order = optimization.get('optimal_order', [])
				if optimal_order:
					reordered_steps = []
					for step_id in optimal_order:
						for step in pipeline.steps:
							if step.get('id') == step_id:
								reordered_steps.append(step)
								break
					pipeline.steps = reordered_steps
				
			elif opt_type == 'caching':
				# Enable caching for expensive operations
				caching_config = optimization.get('caching_config', {})
				pipeline.configuration['caching'] = caching_config
				pipeline.configuration['enable_caching'] = True
				
			elif opt_type == 'resource_allocation':
				# Update resource allocation
				resource_config = optimization.get('resource_config', {})
				pipeline.configuration['resources'] = resource_config
				
			else:
				raise ValueError(f"Unknown optimization type: {opt_type}")
			
			# Update pipeline metadata
			pipeline.updated_at = datetime.utcnow()
			pipeline.updated_by = self.user_id
			pipeline.version = self._increment_version(pipeline.version)
			
			# Log optimization applied
			self._log_info(f"Applied {opt_type} optimization to pipeline {pipeline.name}")
			
			# Update in database would go here
			# await self.database.update_pipeline(pipeline)
			
		except Exception as e:
			self._log_error(f"Failed to apply optimization {opt_type}: {str(e)}")
			raise
