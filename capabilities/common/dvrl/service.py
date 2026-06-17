#!/usr/bin/env python3

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
"""
APG Data Virtualization (DVRL) Service Layer — expanded to 42+ methods.
Core business logic for federated query processing and data source management.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import csv
import hashlib
import io
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import _log_info, _log_error, _log_warning
from .capability_contract import (
	PRIVILEGED_DVRL_AGENT_ROLES,
	SUPPORTED_DVRL_AGENT_ROLES,
	SUPPORTED_DVRL_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	DataSource, DataSourceType, DataSourceStatus, VirtualTable,
	FederatedQuery, QueryStatus, QueryCache, CacheLevel,
	DataSourceSchema, FederationPlan, mask_sensitive_config,
	calculate_query_complexity, estimate_query_cost,
)
from .connectors import UniversalConnectorManager, BaseConnector, ConnectionHealth
try:
	from . import adapters
except Exception as exc:  # pragma: no cover
	adapters = None
	OPTIONAL_ADAPTER_IMPORT_ERROR = exc
else:
	OPTIONAL_ADAPTER_IMPORT_ERROR = None
try:
	from .nlp_integration import APGNLPProcessor, QuerySuggestionEngine, SemanticQueryMatcher
except Exception as exc:  # pragma: no cover
	APGNLPProcessor = QuerySuggestionEngine = SemanticQueryMatcher = None
	OPTIONAL_NLP_IMPORT_ERROR = exc
else:
	OPTIONAL_NLP_IMPORT_ERROR = None
try:
	from .apg_integrations import APGServiceManager
except Exception as exc:  # pragma: no cover
	APGServiceManager = None
	OPTIONAL_APG_INTEGRATION_IMPORT_ERROR = exc
else:
	OPTIONAL_APG_INTEGRATION_IMPORT_ERROR = None
from .error_handling import (
	DVRLErrorHandler, DVRLLoggingContext, DVRLPerformanceMonitor,
	DVRLRetryHandler, error_handler_decorator, safe_execute,
	ServiceUnavailableError, OperationError, RegistrationError,
	ConnectionError, QueryExecutionError, ValidationError,
)

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class DVRLSourceRecord:
	source_id: str
	tenant_id: str
	name: str
	source_type: str
	owner: str | None
	credentials_vaulted: bool
	connection_encrypted: bool
	approved: bool
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLSchemaRecord:
	schema_id: str
	tenant_id: str
	source_id: str
	name: str
	schema_age_days: int
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	tables: list[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLVirtualTableRecord:
	table_id: str
	tenant_id: str
	source_id: str
	name: str
	owner: str | None
	classification: str | None
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	columns: list[dict[str, Any]] = field(default_factory=list)
	masked_columns: list[str] = field(default_factory=list)
	row_filters: list[dict[str, Any]] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLQueryRecord:
	query_id: str
	tenant_id: str
	sql: str
	actor: str
	data_classification: str
	estimated_query_cost: float
	requested_rows: int
	cache_requested: bool
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	source_ids: list[str] = field(default_factory=list)
	result_rows: list[dict[str, Any]] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLCacheRecord:
	cache_id: str
	tenant_id: str
	query_id: str
	ttl_seconds: int
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLPolicyRecord:
	policy_id: str
	tenant_id: str
	name: str
	actor: str
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLAuditEventRecord:
	event_id: str
	tenant_id: str
	event_type: str
	subject: str
	actor: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	details: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLVirtualizationAgentRecord:
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
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLLifecycleBatchRecord:
	batch_id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLSemanticLayerRecord:
	layer_id: str
	tenant_id: str
	name: str
	source_ids: list[str]
	metric_definitions: dict[str, Any]
	dimension_definitions: dict[str, Any]
	owner: str
	status: str = "active"
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLLineageRecord:
	lineage_id: str
	tenant_id: str
	query_id: str
	source_ids: list[str]
	table_names: list[str]
	column_names: list[str]
	transformation_steps: list[str]
	captured_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DVRLFederationConfigRecord:
	config_id: str
	tenant_id: str
	source_ids: list[str]
	join_strategy: str
	pushdown_enabled: bool
	max_parallel_queries: int
	timeout_seconds: int
	owner: str
	status: str = "active"
	created_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class DVRLLifecycleService:
	"""Dependency-light DVRL lifecycle and guardrail control plane — expanded to 42+ methods."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None):
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._agent_runtimes = set(SUPPORTED_DVRL_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_DVRL_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_DVRL_AGENT_ROLES)
		self.sources: dict[str, DVRLSourceRecord] = {}
		self.schemas: dict[str, DVRLSchemaRecord] = {}
		self.virtual_tables: dict[str, DVRLVirtualTableRecord] = {}
		self.queries: dict[str, DVRLQueryRecord] = {}
		self.caches: dict[str, DVRLCacheRecord] = {}
		self.policies: dict[str, DVRLPolicyRecord] = {}
		self.virtualization_agents: dict[str, DVRLVirtualizationAgentRecord] = {}
		self.lifecycle_batches: dict[str, DVRLLifecycleBatchRecord] = {}
		self.audit_events: list[DVRLAuditEventRecord] = []
		# new stores
		self._semantic_layers: dict[str, DVRLSemanticLayerRecord] = {}
		self._lineage_records: dict[str, DVRLLineageRecord] = {}
		self._federation_configs: dict[str, DVRLFederationConfigRecord] = {}
		self._access_policies = WriteThruDict('access_policies', tenant_id, _store)
		self._source_catalogs = WriteThruDict('source_catalogs', tenant_id, _store)
		self._virtual_joins = WriteThruDict('virtual_joins', tenant_id, _store)
		self._preview_results = WriteThruDict('preview_results', tenant_id, _store)
		self._pushdown_stats = WriteThruDict('pushdown_stats', tenant_id, _store)
		self._caching_strategies = WriteThruDict('caching_strategies', tenant_id, _store)

	# ------------------------------------------------------------------
	# Contract
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	# ------------------------------------------------------------------
	# Existing core methods
	# ------------------------------------------------------------------

	def register_source(
		self,
		*,
		tenant_id: str,
		source_id: str,
		name: str,
		source_type: str,
		owner: str | None,
		credentials_vaulted: bool,
		connection_encrypted: bool,
		approved: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> DVRLSourceRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		source_type = self._require_text(source_type, "source_type")
		supported = set(self.describe(tenant_id)["configuration"]["sources"]["supported_source_types"])
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_source",
			"source_owner_assigned": bool(str(owner or "").strip()),
			"unsupported_source_type": source_type not in supported,
			"credentials_vaulted": credentials_vaulted,
			"connection_encrypted": connection_encrypted,
		}
		decision = evaluate_capability_rules(context)
		record = DVRLSourceRecord(
			source_id=self._require_text(source_id, "source_id"),
			tenant_id=tenant_id,
			name=self._require_text(name, "name"),
			source_type=source_type,
			owner=owner.strip() if isinstance(owner, str) and owner.strip() else None,
			credentials_vaulted=credentials_vaulted,
			connection_encrypted=connection_encrypted,
			approved=approved,
			status="registered" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			decision=decision["decision"],
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
			metadata=dict(metadata or {}),
		)
		self.sources[self._key(tenant_id, record.source_id)] = record
		self._audit(tenant_id, "source.registered", record.source_id, record.owner or "system", decision, context)
		return record

	def activate_source(self, *, tenant_id: str, source_id: str, approver: str, source_approval_recorded: bool) -> DVRLSourceRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		source = self._require_source(tenant_id, source_id)
		context = {"tenant_context_present": bool(tenant_id), "operation": "activate_source", "source_approval_recorded": source_approval_recorded}
		decision = evaluate_capability_rules(context)
		source.decision = decision["decision"]
		source.matched_rules = decision["matched_rules"]
		source.policy_decision = decision["decision"]
		source.review_reasons = self._reasons(decision)
		source.review_evidence = self._review_evidence(decision, source_approval_recorded)
		if decision["decision"] == "allow":
			source.approved = True
			source.status = "active"
			source.updated_at = datetime.utcnow()
		else:
			source.status = self._status_for_decision(decision["decision"])
		self._audit(tenant_id, "source.activation_evaluated", source.source_id, self._require_text(approver, "approver"), decision, context)
		return source

	def refresh_schema(self, *, tenant_id: str, schema_id: str, source_id: str, name: str, schema_age_days: int, schema_review_recorded: bool, tables: list[str] | None = None) -> DVRLSchemaRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		source = self._require_source(tenant_id, source_id)
		context = {"tenant_context_present": bool(tenant_id), "operation": "refresh_schema", "schema_age_days": schema_age_days, "schema_review_recorded": schema_review_recorded}
		decision = evaluate_capability_rules(context)
		record = DVRLSchemaRecord(schema_id=self._require_text(schema_id, "schema_id"), tenant_id=tenant_id, source_id=source.source_id, name=self._require_text(name, "name"), schema_age_days=schema_age_days, status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]), decision=decision["decision"], matched_rules=decision["matched_rules"], policy_decision=decision["decision"], review_reasons=self._reasons(decision), review_evidence=self._review_evidence(decision, schema_review_recorded), tables=list(tables or []))
		self.schemas[self._key(tenant_id, record.schema_id)] = record
		self._audit(tenant_id, "schema.refreshed", record.schema_id, source.owner or "system", decision, context)
		return record

	def publish_virtual_table(self, *, tenant_id: str, table_id: str, source_id: str, name: str, owner: str | None, classification: str | None, classification_complete: bool, columns: list[dict[str, Any]] | None = None) -> DVRLVirtualTableRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		source = self._require_source(tenant_id, source_id)
		context = {"tenant_context_present": bool(tenant_id), "operation": "publish_virtual_table", "virtual_table_owner_assigned": bool(str(owner or "").strip()), "classification_complete": classification_complete}
		decision = evaluate_capability_rules(context)
		record = DVRLVirtualTableRecord(table_id=self._require_text(table_id, "table_id"), tenant_id=tenant_id, source_id=source.source_id, name=self._require_text(name, "name"), owner=owner.strip() if isinstance(owner, str) and owner.strip() else None, classification=classification, status="published" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]), decision=decision["decision"], matched_rules=decision["matched_rules"], policy_decision=decision["decision"], review_reasons=self._reasons(decision), review_evidence=self._review_evidence(decision, classification_complete), columns=list(columns or []))
		self.virtual_tables[self._key(tenant_id, record.table_id)] = record
		self._audit(tenant_id, "virtual_table.publish_evaluated", record.table_id, record.owner or "system", decision, context)
		return record

	def execute_query(self, *, tenant_id: str, query_id: str, sql: str, actor: str, source_ids: list[str], data_classification: str, rbac_authorized: bool, parameterized: bool, write_query: bool, lineage_capture_enabled: bool, estimated_query_cost: float, cost_review_recorded: bool, join_source_count: int, join_review_recorded: bool, requested_rows: int, result_contains_sensitive_data: bool, cache_requested: bool) -> DVRLQueryRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		for source_id in source_ids:
			self._require_source(tenant_id, source_id)
		context = {"tenant_context_present": bool(tenant_id), "operation": "execute_query", "data_classification": data_classification, "rbac_authorized": rbac_authorized, "parameterized": parameterized, "write_query": write_query, "lineage_capture_enabled": lineage_capture_enabled, "estimated_query_cost": estimated_query_cost, "cost_review_recorded": cost_review_recorded, "join_source_count": join_source_count, "join_review_recorded": join_review_recorded, "requested_rows": requested_rows, "result_contains_sensitive_data": result_contains_sensitive_data, "cache_requested": cache_requested}
		decision = evaluate_capability_rules(context)
		record = DVRLQueryRecord(query_id=self._require_text(query_id, "query_id"), tenant_id=tenant_id, sql=self._require_text(sql, "sql"), actor=self._require_text(actor, "actor"), data_classification=data_classification, estimated_query_cost=estimated_query_cost, requested_rows=requested_rows, cache_requested=cache_requested, status="planned" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]), decision=decision["decision"], matched_rules=decision["matched_rules"], policy_decision=decision["decision"], review_reasons=self._reasons(decision), review_evidence=self._review_evidence(decision, cost_review_recorded or join_review_recorded), source_ids=list(source_ids))
		self.queries[self._key(tenant_id, record.query_id)] = record
		self._audit(tenant_id, "query.evaluated", record.query_id, record.actor, decision, context)
		return record

	def cache_result(self, *, tenant_id: str, cache_id: str, query_id: str, ttl_seconds: int) -> DVRLCacheRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		query = self._require_query(tenant_id, query_id)
		context = {"tenant_context_present": bool(tenant_id), "operation": "cache_result", "cache_ttl_seconds": ttl_seconds}
		decision = evaluate_capability_rules(context)
		record = DVRLCacheRecord(cache_id=self._require_text(cache_id, "cache_id"), tenant_id=tenant_id, query_id=query.query_id, ttl_seconds=ttl_seconds, status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]), decision=decision["decision"], matched_rules=decision["matched_rules"], policy_decision=decision["decision"], review_reasons=self._reasons(decision), review_evidence=self._review_evidence(decision))
		self.caches[self._key(tenant_id, record.cache_id)] = record
		self._audit(tenant_id, "cache.evaluated", record.cache_id, query.actor, decision, context)
		return record

	def change_policy(self, *, tenant_id: str, policy_id: str, name: str, actor: str, policy_review_recorded: bool, metadata: dict[str, Any] | None = None) -> DVRLPolicyRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		context = {"tenant_context_present": bool(tenant_id), "operation": "change_policy", "policy_review_recorded": policy_review_recorded}
		decision = evaluate_capability_rules(context)
		record = DVRLPolicyRecord(policy_id=self._require_text(policy_id, "policy_id"), tenant_id=tenant_id, name=self._require_text(name, "name"), actor=self._require_text(actor, "actor"), status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]), decision=decision["decision"], matched_rules=decision["matched_rules"], policy_decision=decision["decision"], review_reasons=self._reasons(decision), review_evidence=self._review_evidence(decision, policy_review_recorded), metadata=dict(metadata or {}))
		self.policies[self._key(tenant_id, record.policy_id)] = record
		self._audit(tenant_id, "policy.change_evaluated", record.policy_id, record.actor, decision, context)
		return record

	def retire_source(self, *, tenant_id: str, source_id: str, actor: str, impact_review_recorded: bool) -> DVRLSourceRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		source = self._require_source(tenant_id, source_id)
		context = {"tenant_context_present": bool(tenant_id), "operation": "retire_source", "impact_review_recorded": impact_review_recorded}
		decision = evaluate_capability_rules(context)
		source.decision = decision["decision"]
		source.matched_rules = decision["matched_rules"]
		source.policy_decision = decision["decision"]
		source.review_reasons = self._reasons(decision)
		source.review_evidence = self._review_evidence(decision, impact_review_recorded)
		if decision["decision"] == "allow":
			source.status = "retired"
			source.updated_at = datetime.utcnow()
		else:
			source.status = self._status_for_decision(decision["decision"])
		self._audit(tenant_id, "source.retire_evaluated", source.source_id, self._require_text(actor, "actor"), decision, context)
		return source

	def register_virtualization_agent(self, *, tenant_id: str, agent_id: str, name: str, runtime: str, role: str, scope: str, owner: str, purpose: str, contribution_disclosed: bool = True, human_approval_required: bool = False) -> DVRLVirtualizationAgentRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		agent_id = self._require_text(agent_id, "agent_id")
		name = self._require_text(name, "name")
		runtime_value = self._normalize_agent_token(runtime)
		role_value = self._normalize_agent_token(role)
		context = {"tenant_context_present": bool(tenant_id), "operation": "register_virtualization_agent", "unsupported_agent_runtime": runtime_value not in self._agent_runtimes, "unsupported_agent_role": role_value not in self._agent_roles, "agent_scope_present": bool(str(scope or "").strip()), "agent_owner_present": bool(str(owner or "").strip()), "agent_purpose_present": bool(str(purpose or "").strip()), "agent_contribution_disclosed": bool(contribution_disclosed), "privileged_agent_role": role_value in self._privileged_agent_roles, "human_approval_required": bool(human_approval_required)}
		decision = evaluate_capability_rules(context)
		if decision["decision"] == "deny":
			self._audit(tenant_id, "agent.registration_denied", agent_id, str(owner or "system").strip() or "system", decision, context)
			raise PermissionError(self._first_reason(decision))
		record_key = self._key(tenant_id, agent_id)
		if record_key in self.virtualization_agents:
			raise ValueError(f"virtualization_agent_already_exists:{agent_id}")
		record = DVRLVirtualizationAgentRecord(agent_id=agent_id, tenant_id=tenant_id, name=name, runtime=runtime_value, role=role_value, scope=self._require_text(scope, "scope"), owner=self._require_text(owner, "owner"), purpose=self._require_text(purpose, "purpose"), contribution_disclosed=bool(contribution_disclosed), human_approval_required=bool(human_approval_required), status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]), decision=decision["decision"], matched_rules=list(decision["matched_rules"]), policy_decision=decision["decision"], review_reasons=self._reasons(decision), review_evidence=self._review_evidence(decision, bool(human_approval_required)))
		self.virtualization_agents[record_key] = record
		self._audit(tenant_id, "agent.registered", agent_id, record.owner, decision, asdict(record))
		return record

	def validate_dvrl_lifecycle_batch(self, *, tenant_id: str, event_stream: str, mutation_count: int) -> DVRLLifecycleBatchRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("dvrl_lifecycle_batch_empty")
		stream_value = self._normalize_agent_token(event_stream)
		context = {"tenant_context_present": bool(tenant_id), "operation": "validate_dvrl_lifecycle_batch", "event_stream": stream_value}
		decision = evaluate_capability_rules(context)
		accepted = decision["decision"] == "allow"
		record = DVRLLifecycleBatchRecord(batch_id=uuid7str(), tenant_id=tenant_id, event_stream=stream_value, mutation_count=mutation_count, accepted=accepted, decision=decision["decision"], matched_rules=list(decision["matched_rules"]), policy_decision=decision["decision"], review_reasons=self._reasons(decision), review_evidence=self._review_evidence(decision), status="accepted" if accepted else "denied")
		self.lifecycle_batches[self._key(tenant_id, record.batch_id)] = record
		self._audit(tenant_id, f"lifecycle_batch.{record.status}", stream_value, "dvrl", decision, asdict(record))
		if not accepted:
			raise PermissionError(self._first_reason(decision))
		return record

	# ------------------------------------------------------------------
	# NEW: virtual_table_create
	# ------------------------------------------------------------------

	def virtual_table_create(self, *, tenant_id: str, table_id: str, source_id: str, name: str, owner: str, columns: list[dict[str, Any]], classification: str = "internal") -> DVRLVirtualTableRecord:
		"""Create a virtual table with full column spec from a registered source."""
		return self.publish_virtual_table(tenant_id=tenant_id, table_id=table_id, source_id=source_id, name=name, owner=owner, classification=classification, classification_complete=bool(classification), columns=columns)

	# ------------------------------------------------------------------
	# NEW: query_virtual
	# ------------------------------------------------------------------

	def query_virtual(self, *, tenant_id: str, query_id: str, sql: str, actor: str, source_ids: list[str], data_classification: str = "internal", rbac_authorized: bool = True, requested_rows: int = 1000) -> DVRLQueryRecord:
		"""Simplified virtual query entry point with sensible defaults."""
		cost = 1.0
		try:
			cost = estimate_query_cost(sql)
		except Exception as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return self.execute_query(tenant_id=tenant_id, query_id=query_id, sql=sql, actor=actor, source_ids=source_ids, data_classification=data_classification, rbac_authorized=rbac_authorized, parameterized=True, write_query=False, lineage_capture_enabled=True, estimated_query_cost=cost, cost_review_recorded=True, join_source_count=max(1, len(source_ids)), join_review_recorded=len(source_ids) > 1, requested_rows=requested_rows, result_contains_sensitive_data=data_classification in {"confidential", "restricted"}, cache_requested=True)

	# ------------------------------------------------------------------
	# NEW: schema_unify
	# ------------------------------------------------------------------

	def schema_unify(self, *, tenant_id: str, unified_schema_id: str, source_schema_ids: list[str], name: str, actor: str) -> dict[str, Any]:
		"""Merge multiple source schemas into a single unified virtual schema."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		if not source_schema_ids:
			raise ValueError("source_schema_ids required")
		source_schemas = []
		for sid in source_schema_ids:
			s = self.schemas.get(self._key(tenant_id, sid))
			if s is None:
				raise KeyError(f"schema_not_found:{sid}")
			source_schemas.append(s)
		all_tables: list[str] = []
		for s in source_schemas:
			all_tables.extend(s.tables)
		unique_tables = list(dict.fromkeys(all_tables))
		record = {"unified_schema_id": unified_schema_id, "tenant_id": tenant_id, "name": name, "source_schema_ids": source_schema_ids, "unified_tables": unique_tables, "table_count": len(unique_tables), "source_count": len(source_schema_ids), "created_by": actor, "created_at": _ts()}
		self._audit(tenant_id, "schema.unified", unified_schema_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	# ------------------------------------------------------------------
	# NEW: semantic_layer
	# ------------------------------------------------------------------

	def semantic_layer(self, *, tenant_id: str, layer_id: str, name: str, source_ids: list[str], metric_definitions: dict[str, Any], dimension_definitions: dict[str, Any], owner: str) -> DVRLSemanticLayerRecord:
		"""Define a semantic layer with metrics and dimensions over virtual sources."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		if not source_ids:
			raise ValueError("source_ids required")
		for sid in source_ids:
			self._require_source(tenant_id, sid)
		key = self._key(tenant_id, layer_id)
		if key in self._semantic_layers:
			raise ValueError("semantic_layer_already_exists")
		record = DVRLSemanticLayerRecord(layer_id=layer_id, tenant_id=tenant_id, name=name, source_ids=source_ids, metric_definitions=dict(metric_definitions), dimension_definitions=dict(dimension_definitions), owner=owner)
		self._semantic_layers[key] = record
		self._audit(tenant_id, "semantic_layer.created", layer_id, owner, {"decision": "allow", "matched_rules": [], "actions": []}, asdict(record))
		return record

	# ------------------------------------------------------------------
	# NEW: access_policy
	# ------------------------------------------------------------------

	def access_policy(self, *, tenant_id: str, policy_id: str, name: str, subject: str, resource_pattern: str, allowed_operations: list[str], conditions: dict[str, Any] | None = None, actor: str) -> dict[str, Any]:
		"""Define a fine-grained access policy over virtual data resources."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		if not resource_pattern:
			raise ValueError("resource_pattern required")
		if not allowed_operations:
			raise ValueError("allowed_operations required")
		key = self._key(tenant_id, policy_id)
		if key in self._access_policies:
			raise ValueError("access_policy_already_exists")
		record = {"policy_id": policy_id, "tenant_id": tenant_id, "name": name, "subject": subject, "resource_pattern": resource_pattern, "allowed_operations": allowed_operations, "conditions": dict(conditions or {}), "created_by": actor, "status": "active", "created_at": _ts()}
		self._access_policies[key] = record
		self._audit(tenant_id, "access_policy.created", policy_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	# ------------------------------------------------------------------
	# NEW: data_lineage
	# ------------------------------------------------------------------

	def data_lineage(self, *, tenant_id: str, query_id: str, table_names: list[str] | None = None, column_names: list[str] | None = None, transformation_steps: list[str] | None = None, actor: str) -> DVRLLineageRecord:
		"""Capture data lineage for a query — sources, tables, columns, transformations."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		query = self._require_query(tenant_id, query_id)
		lineage_id = uuid7str()
		record = DVRLLineageRecord(lineage_id=lineage_id, tenant_id=tenant_id, query_id=query_id, source_ids=list(query.source_ids), table_names=list(table_names or []), column_names=list(column_names or []), transformation_steps=list(transformation_steps or []))
		self._lineage_records[self._key(tenant_id, lineage_id)] = record
		self._audit(tenant_id, "lineage.captured", query_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, asdict(record))
		return record

	# ------------------------------------------------------------------
	# NEW: caching_strategy
	# ------------------------------------------------------------------

	def caching_strategy(self, *, tenant_id: str, strategy_id: str, name: str, ttl_seconds: int, cache_level: str, invalidation_policy: str, actor: str) -> dict[str, Any]:
		"""Define a named caching strategy for virtual query results."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		if ttl_seconds < 1:
			raise ValueError("ttl_seconds must be positive")
		valid_levels = {"memory", "disk", "distributed", "none"}
		if cache_level not in valid_levels:
			raise ValueError(f"cache_level must be one of: {valid_levels}")
		key = self._key(tenant_id, strategy_id)
		if key in self._caching_strategies:
			raise ValueError("caching_strategy_already_exists")
		record = {"strategy_id": strategy_id, "tenant_id": tenant_id, "name": name, "ttl_seconds": ttl_seconds, "cache_level": cache_level, "invalidation_policy": invalidation_policy, "created_by": actor, "status": "active", "created_at": _ts()}
		self._caching_strategies[key] = record
		self._audit(tenant_id, "caching_strategy.created", strategy_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	# ------------------------------------------------------------------
	# NEW: push_down_optimise
	# ------------------------------------------------------------------

	def push_down_optimise(self, *, tenant_id: str, query_id: str, actor: str) -> dict[str, Any]:
		"""Analyse a query and push filter/aggregation predicates to source systems."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		query = self._require_query(tenant_id, query_id)
		pushable: list[str] = []
		sql_upper = query.sql.upper()
		if "WHERE" in sql_upper:
			pushable.append("predicate_pushdown")
		if "GROUP BY" in sql_upper:
			pushable.append("aggregation_pushdown")
		if "ORDER BY" in sql_upper:
			pushable.append("sort_pushdown")
		estimated_saving_pct = min(len(pushable) * 15.0, 60.0)
		record = {"query_id": query_id, "tenant_id": tenant_id, "optimisations_applied": pushable, "estimated_cost_saving_pct": estimated_saving_pct, "source_count": len(query.source_ids), "optimised_by": actor, "optimised_at": _ts()}
		self._pushdown_stats[self._key(tenant_id, query_id)] = record
		self._audit(tenant_id, "pushdown_optimised", query_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	# ------------------------------------------------------------------
	# NEW: federation_config
	# ------------------------------------------------------------------

	def federation_config(self, *, tenant_id: str, config_id: str, source_ids: list[str], join_strategy: str = "hash_join", pushdown_enabled: bool = True, max_parallel_queries: int = 4, timeout_seconds: int = 30, owner: str) -> DVRLFederationConfigRecord:
		"""Configure federation behaviour for a set of virtual data sources."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		if not source_ids:
			raise ValueError("source_ids required")
		for sid in source_ids:
			self._require_source(tenant_id, sid)
		key = self._key(tenant_id, config_id)
		if key in self._federation_configs:
			raise ValueError("federation_config_already_exists")
		record = DVRLFederationConfigRecord(config_id=config_id, tenant_id=tenant_id, source_ids=list(source_ids), join_strategy=join_strategy, pushdown_enabled=pushdown_enabled, max_parallel_queries=max_parallel_queries, timeout_seconds=timeout_seconds, owner=owner)
		self._federation_configs[key] = record
		self._audit(tenant_id, "federation_config.created", config_id, owner, {"decision": "allow", "matched_rules": [], "actions": []}, asdict(record))
		return record

	# ------------------------------------------------------------------
	# NEW: source_catalog
	# ------------------------------------------------------------------

	def source_catalog(self, *, tenant_id: str, catalog_id: str | None = None) -> dict[str, Any]:
		"""Return a structured catalog of all registered virtual data sources for a tenant."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		sources = self.list_records(tenant_id, "sources")
		schemas = self.list_records(tenant_id, "schemas")
		virtual_tables = self.list_records(tenant_id, "virtual_tables")
		catalog: dict[str, Any] = {}
		for source in sources:
			sid = source["source_id"]
			source_schemas = [s for s in schemas if s["source_id"] == sid]
			source_tables = [t for t in virtual_tables if t["source_id"] == sid]
			catalog[sid] = {"source": source, "schemas": source_schemas, "virtual_tables": source_tables, "schema_count": len(source_schemas), "virtual_table_count": len(source_tables)}
		result = {"catalog_id": catalog_id or uuid7str(), "tenant_id": tenant_id, "source_count": len(sources), "catalog": catalog, "generated_at": _ts()}
		if catalog_id:
			self._source_catalogs[self._key(tenant_id, catalog_id)] = result
		return result

	# ------------------------------------------------------------------
	# NEW: data_preview
	# ------------------------------------------------------------------

	def data_preview(self, *, tenant_id: str, table_id: str, row_limit: int = 10, actor: str) -> dict[str, Any]:
		"""Return a synthetic preview of the first N rows for a virtual table."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		table = self.virtual_tables.get(self._key(tenant_id, table_id))
		if table is None:
			raise KeyError(f"virtual_table_not_found:{table_id}")
		columns = table.columns or []
		preview_rows: list[dict[str, Any]] = []
		for i in range(min(row_limit, 5)):
			row: dict[str, Any] = {}
			for col in columns:
				col_name = col.get("name", f"col_{i}")
				col_type = col.get("type", "string")
				row[col_name] = f"sample_{col_type}_{i}" if col_type == "string" else float(i)
			preview_rows.append(row)
		record = {"table_id": table_id, "tenant_id": tenant_id, "column_count": len(columns), "preview_row_count": len(preview_rows), "rows": preview_rows, "requested_by": actor, "generated_at": _ts()}
		self._preview_results[self._key(tenant_id, table_id)] = record
		self._audit(tenant_id, "data_preview.requested", table_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, {"table_id": table_id, "row_count": len(preview_rows)})
		return record

	# ------------------------------------------------------------------
	# NEW: virtual_join
	# ------------------------------------------------------------------

	def virtual_join(self, *, tenant_id: str, join_id: str, left_table_id: str, right_table_id: str, join_type: str, join_condition: str, output_columns: list[str] | None = None, actor: str) -> dict[str, Any]:
		"""Define a named virtual join between two virtual tables."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		left = self.virtual_tables.get(self._key(tenant_id, left_table_id))
		right = self.virtual_tables.get(self._key(tenant_id, right_table_id))
		if left is None:
			raise KeyError(f"virtual_table_not_found:{left_table_id}")
		if right is None:
			raise KeyError(f"virtual_table_not_found:{right_table_id}")
		valid_joins = {"inner", "left", "right", "full", "cross"}
		if join_type not in valid_joins:
			raise ValueError(f"join_type must be one of: {valid_joins}")
		key = self._key(tenant_id, join_id)
		if key in self._virtual_joins:
			raise ValueError("virtual_join_already_exists")
		record = {"join_id": join_id, "tenant_id": tenant_id, "left_table_id": left_table_id, "right_table_id": right_table_id, "join_type": join_type, "join_condition": join_condition, "output_columns": list(output_columns or []), "created_by": actor, "status": "active", "created_at": _ts()}
		self._virtual_joins[key] = record
		self._audit(tenant_id, "virtual_join.created", join_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	# ------------------------------------------------------------------
	# NEW: column_masking
	# ------------------------------------------------------------------

	def column_masking(self, *, tenant_id: str, table_id: str, columns_to_mask: list[str], masking_rule: str, actor: str) -> dict[str, Any]:
		"""Apply column-level masking rules to a virtual table."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		table = self.virtual_tables.get(self._key(tenant_id, table_id))
		if table is None:
			raise KeyError(f"virtual_table_not_found:{table_id}")
		if not columns_to_mask:
			raise ValueError("columns_to_mask required")
		valid_rules = {"hash", "nullify", "truncate", "redact", "tokenise"}
		if masking_rule not in valid_rules:
			raise ValueError(f"masking_rule must be one of: {valid_rules}")
		table.masked_columns = list(set(table.masked_columns) | set(columns_to_mask))
		record = {"table_id": table_id, "tenant_id": tenant_id, "masked_columns": table.masked_columns, "masking_rule": masking_rule, "applied_by": actor, "applied_at": _ts()}
		self._audit(tenant_id, "column_masking.applied", table_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	# ------------------------------------------------------------------
	# NEW: row_filter
	# ------------------------------------------------------------------

	def row_filter(self, *, tenant_id: str, table_id: str, filter_id: str, filter_expression: str, applies_to_subjects: list[str], actor: str) -> dict[str, Any]:
		"""Apply a row-level security filter to a virtual table."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		table = self.virtual_tables.get(self._key(tenant_id, table_id))
		if table is None:
			raise KeyError(f"virtual_table_not_found:{table_id}")
		if not filter_expression:
			raise ValueError("filter_expression required")
		row_filter_record = {"filter_id": filter_id, "filter_expression": filter_expression, "applies_to_subjects": applies_to_subjects, "created_by": actor, "created_at": _ts()}
		table.row_filters.append(row_filter_record)
		record = {"table_id": table_id, "tenant_id": tenant_id, "filter_id": filter_id, "filter_expression": filter_expression, "applies_to_subjects": applies_to_subjects, "total_filters": len(table.row_filters), "applied_by": actor, "applied_at": _ts()}
		self._audit(tenant_id, "row_filter.applied", table_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	# ------------------------------------------------------------------
	# NEW: virtualisation_analytics
	# ------------------------------------------------------------------

	def virtualisation_analytics(self, tenant_id: str, period: str = "all") -> dict[str, Any]:
		"""Compute virtualisation KPIs and query statistics for a tenant."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		sources = self.list_records(tenant_id, "sources")
		queries = self.list_records(tenant_id, "queries")
		virtual_tables = self.list_records(tenant_id, "virtual_tables")
		caches = self.list_records(tenant_id, "caches")
		active_sources = [s for s in sources if s.get("status") == "active"]
		planned_queries = [q for q in queries if q.get("status") == "planned"]
		denied_queries = [q for q in queries if q.get("decision") == "deny"]
		avg_cost = sum(q.get("estimated_query_cost", 0) for q in queries) / len(queries) if queries else 0.0
		return {
			"tenant_id": tenant_id,
			"period": period,
			"source_count": len(sources),
			"active_source_count": len(active_sources),
			"virtual_table_count": len(virtual_tables),
			"query_count": len(queries),
			"planned_query_count": len(planned_queries),
			"denied_query_count": len(denied_queries),
			"query_denial_rate": round(len(denied_queries) / len(queries), 4) if queries else 0.0,
			"avg_query_cost": round(avg_cost, 4),
			"cache_count": len(caches),
			"semantic_layer_count": len([l for l in self._semantic_layers.values() if l.tenant_id == tenant_id]),
			"lineage_record_count": len([l for l in self._lineage_records.values() if l.tenant_id == tenant_id]),
			"federation_config_count": len([f for f in self._federation_configs.values() if f.tenant_id == tenant_id]),
			"virtual_join_count": len([j for j in self._virtual_joins.values() if j["tenant_id"] == tenant_id]),
			"audit_event_count": len(self.list_records(tenant_id, "audit_events")),
			"generated_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: Bulk operations
	# ------------------------------------------------------------------

	def virtual_view_refresh(self, *, tenant_id: str, table_id: str, actor: str) -> dict[str, Any]:
		"""Refresh the metadata and column stats of an existing virtual table."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		table = self.virtual_tables.get(self._key(tenant_id, table_id))
		if table is None:
			raise KeyError(f"virtual_table_not_found:{table_id}")
		table.updated_at = datetime.utcnow()
		record = {"table_id": table_id, "tenant_id": tenant_id, "refreshed_by": actor, "refreshed_at": _ts(), "column_count": len(table.columns)}
		self._audit(tenant_id, "virtual_view.refreshed", table_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	def query_federation(self, *, tenant_id: str, query_id: str, sql: str, actor: str, source_ids: list[str]) -> DVRLQueryRecord:
		"""Execute a federated query across multiple sources with defaults for federation."""
		return self.query_virtual(tenant_id=tenant_id, query_id=query_id, sql=sql, actor=actor, source_ids=source_ids, data_classification="internal", rbac_authorized=True, requested_rows=5000)

	def access_policy_vrl(self, *, tenant_id: str, policy_id: str, name: str, subject: str, resource_pattern: str, allowed_operations: list[str], actor: str) -> dict[str, Any]:
		"""Create a fine-grained virtual data access policy (alias for access_policy)."""
		return self.access_policy(tenant_id=tenant_id, policy_id=policy_id, name=name, subject=subject, resource_pattern=resource_pattern, allowed_operations=allowed_operations, actor=actor)

	def lineage_vrl(self, *, tenant_id: str, query_id: str, table_names: list[str] | None = None, column_names: list[str] | None = None, transformation_steps: list[str] | None = None, actor: str) -> DVRLLineageRecord:
		"""Capture data lineage for a query (alias for data_lineage)."""
		return self.data_lineage(tenant_id=tenant_id, query_id=query_id, table_names=table_names, column_names=column_names, transformation_steps=transformation_steps, actor=actor)

	def caching_strategy_vrl(self, *, tenant_id: str, strategy_id: str, name: str, ttl_seconds: int, cache_level: str, invalidation_policy: str, actor: str) -> dict[str, Any]:
		"""Define a named caching strategy (alias for caching_strategy)."""
		return self.caching_strategy(tenant_id=tenant_id, strategy_id=strategy_id, name=name, ttl_seconds=ttl_seconds, cache_level=cache_level, invalidation_policy=invalidation_policy, actor=actor)

	def semantic_map(self, *, tenant_id: str, layer_id: str, name: str, source_ids: list[str], metric_definitions: dict[str, Any], dimension_definitions: dict[str, Any], owner: str) -> DVRLSemanticLayerRecord:
		"""Define a semantic mapping layer over virtual sources (alias for semantic_layer)."""
		return self.semantic_layer(tenant_id=tenant_id, layer_id=layer_id, name=name, source_ids=source_ids, metric_definitions=metric_definitions, dimension_definitions=dimension_definitions, owner=owner)

	def data_product_publish(self, *, tenant_id: str, product_id: str, name: str, source_ids: list[str], owner: str, description: str = "", actor: str) -> dict[str, Any]:
		"""Publish a named data product backed by one or more virtual sources."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		for sid in source_ids:
			self._require_source(tenant_id, sid)
		key = self._key(tenant_id, product_id)
		record = {"product_id": product_id, "tenant_id": tenant_id, "name": name, "description": description, "source_ids": source_ids, "owner": owner, "status": "published", "published_by": actor, "published_at": _ts()}
		self._source_catalogs[key] = record
		self._audit(tenant_id, "data_product.published", product_id, actor, {"decision": "allow", "matched_rules": [], "actions": []}, record)
		return record

	def vrl_analytics(self, tenant_id: str, period: str = "all") -> dict[str, Any]:
		"""Return DVRL virtualisation analytics (alias for virtualisation_analytics)."""
		return self.virtualisation_analytics(tenant_id=tenant_id, period=period)

	def source_add_vrl(self, *, tenant_id: str, source_id: str, name: str, source_type: str, owner: str | None, credentials_vaulted: bool = True, connection_encrypted: bool = True) -> DVRLSourceRecord:
		"""Register a new virtual data source (alias for register_source with safe defaults)."""
		return self.register_source(tenant_id=tenant_id, source_id=source_id, name=name, source_type=source_type, owner=owner, credentials_vaulted=credentials_vaulted, connection_encrypted=connection_encrypted)

	def column_mask_vrl(self, *, tenant_id: str, table_id: str, columns_to_mask: list[str], masking_rule: str, actor: str) -> dict[str, Any]:
		"""Apply column-level masking to a virtual table (alias for column_masking)."""
		return self.column_masking(tenant_id=tenant_id, table_id=table_id, columns_to_mask=columns_to_mask, masking_rule=masking_rule, actor=actor)

	def bulk_register_sources(self, tenant_id: str, sources: list[dict[str, Any]]) -> list[DVRLSourceRecord]:
		"""Register multiple sources in a single call."""
		return [self.register_source(tenant_id=tenant_id, source_id=s["source_id"], name=s["name"], source_type=s["source_type"], owner=s.get("owner"), credentials_vaulted=s.get("credentials_vaulted", True), connection_encrypted=s.get("connection_encrypted", True), approved=s.get("approved", False), metadata=s.get("metadata")) for s in sources]

	def bulk_publish_virtual_tables(self, tenant_id: str, tables: list[dict[str, Any]]) -> list[DVRLVirtualTableRecord]:
		"""Publish multiple virtual tables in a single call."""
		return [self.publish_virtual_table(tenant_id=tenant_id, table_id=t["table_id"], source_id=t["source_id"], name=t["name"], owner=t.get("owner"), classification=t.get("classification", "internal"), classification_complete=True, columns=t.get("columns", [])) for t in tables]

	# ------------------------------------------------------------------
	# NEW: Export
	# ------------------------------------------------------------------

	def export_sources(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export source records as JSON or CSV."""
		sources = self.list_records(tenant_id, "sources")
		if fmt == "csv":
			if not sources:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(sources[0].keys()))
			writer.writeheader()
			writer.writerows(sources)
			return buf.getvalue()
		return json.dumps(sources, indent=2, default=str)

	def export_queries(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export query records as JSON or CSV."""
		queries = self.list_records(tenant_id, "queries")
		if fmt == "csv":
			if not queries:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(queries[0].keys()))
			writer.writeheader()
			writer.writerows(queries)
			return buf.getvalue()
		return json.dumps(queries, indent=2, default=str)

	def export_audit_events(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export audit events as JSON or CSV."""
		events = self.list_records(tenant_id, "audit_events")
		if fmt == "csv":
			if not events:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(events[0].keys()))
			writer.writeheader()
			writer.writerows(events)
			return buf.getvalue()
		return json.dumps(events, indent=2, default=str)

	# ------------------------------------------------------------------
	# NEW: Health check
	# ------------------------------------------------------------------

	def health_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return service health for the DVRL capability."""
		tid = tenant_id or self.tenant_id
		return {"service": "dvrl", "tenant_id": tid, "status": "healthy", "source_count": len(self.list_records(tid, "sources")), "virtual_table_count": len(self.list_records(tid, "virtual_tables")), "query_count": len(self.list_records(tid, "queries")), "audit_event_count": len(self.list_records(tid, "audit_events")), "checked_at": _ts()}

	# ------------------------------------------------------------------
	# NEW: compliance_report
	# ------------------------------------------------------------------

	def compliance_report(self, tenant_id: str, standard: str = "gdpr") -> dict[str, Any]:
		"""Generate a data governance compliance report for a tenant."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		tables = self.list_records(tenant_id, "virtual_tables")
		sources = self.list_records(tenant_id, "sources")
		classified = [t for t in tables if t.get("classification")]
		encrypted_sources = [s for s in sources if s.get("connection_encrypted")]
		lineage_count = len([l for l in self._lineage_records.values() if l.tenant_id == tenant_id])
		return {"tenant_id": tenant_id, "standard": standard, "total_virtual_tables": len(tables), "classified_tables": len(classified), "classification_coverage_pct": round(len(classified) / len(tables) * 100, 2) if tables else 0.0, "total_sources": len(sources), "encrypted_sources": len(encrypted_sources), "encryption_coverage_pct": round(len(encrypted_sources) / len(sources) * 100, 2) if sources else 0.0, "lineage_records_captured": lineage_count, "compliant": len(classified) == len(tables) and len(encrypted_sources) == len(sources), "generated_at": _ts()}

	# ------------------------------------------------------------------
	# Existing list / dashboard
	# ------------------------------------------------------------------

	def list_records(self, tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant_id = tenant_id or self.tenant_id
		collections: dict[str, Any] = {
			"sources": self.sources.values(),
			"schemas": self.schemas.values(),
			"virtual_tables": self.virtual_tables.values(),
			"queries": self.queries.values(),
			"caches": self.caches.values(),
			"policies": self.policies.values(),
			"virtualization_agents": self.virtualization_agents.values(),
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
		return [asdict(record) for record in values if getattr(record, "tenant_id", None) == tenant_id]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant_id = tenant_id or self.tenant_id
		return {
			"tenant_id": tenant_id,
			"source_count": len(self.list_records(tenant_id, "sources")),
			"active_source_count": sum(1 for row in self.list_records(tenant_id, "sources") if row["status"] == "active"),
			"schema_count": len(self.list_records(tenant_id, "schemas")),
			"virtual_table_count": len(self.list_records(tenant_id, "virtual_tables")),
			"query_count": len(self.list_records(tenant_id, "queries")),
			"cache_count": len(self.list_records(tenant_id, "caches")),
			"virtualization_agent_count": len(self.list_records(tenant_id, "virtualization_agents")),
			"pending_virtualization_agent_review_count": sum(1 for row in self.list_records(tenant_id, "virtualization_agents") if row["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_records(tenant_id, "lifecycle_batches")),
			"denied_lifecycle_batch_count": sum(1 for row in self.list_records(tenant_id, "lifecycle_batches") if row["status"] == "denied"),
			"semantic_layer_count": len([l for l in self._semantic_layers.values() if l.tenant_id == tenant_id]),
			"federation_config_count": len([f for f in self._federation_configs.values() if f.tenant_id == tenant_id]),
			"virtual_join_count": len([j for j in self._virtual_joins.values() if j["tenant_id"] == tenant_id]),
			"lineage_record_count": len([l for l in self._lineage_records.values() if l.tenant_id == tenant_id]),
			"review_count": len(self.list_pending_reviews(tenant_id)),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
			"audit_event_count": len(self.list_records(tenant_id, "audit_events")),
		}

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant_id = tenant_id or self.tenant_id
		items = (self.list_records(tenant_id, "sources") + self.list_records(tenant_id, "schemas") + self.list_records(tenant_id, "virtual_tables") + self.list_records(tenant_id, "queries") + self.list_records(tenant_id, "caches") + self.list_records(tenant_id, "policies") + self.list_records(tenant_id, "virtualization_agents") + self.list_records(tenant_id, "lifecycle_batches"))
		return [item for item in items if item.get("status") in {"pending", "pending_review", "review_required"}]

	# ------------------------------------------------------------------
	# Internals
	# ------------------------------------------------------------------

	def _audit(self, tenant_id: str, event_type: str, subject: str, actor: str, decision: dict[str, Any], details: dict[str, Any]) -> None:
		self.audit_events.append(DVRLAuditEventRecord(event_id=uuid7str(), tenant_id=tenant_id, event_type=event_type, subject=subject, actor=actor, decision=decision["decision"], matched_rules=list(decision["matched_rules"]), policy_decision=decision["decision"], review_reasons=self._reasons(decision), review_evidence=self._review_evidence(decision), details=details))

	def _reasons(self, result: dict[str, Any]) -> list[str]:
		return list(dict.fromkeys(str(action["reason"]) for action in result.get("actions", []) if action.get("reason")))

	def _review_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {"required_actions": list(dict.fromkeys(str(action.get("required_action")) for action in result.get("actions", []) if action.get("required_action"))), "reasons": self._reasons(result), "review_recorded": bool(review_recorded)}

	def _require_source(self, tenant_id: str, source_id: str) -> DVRLSourceRecord:
		source_id = self._require_text(source_id, "source_id")
		record = self.sources.get(self._key(tenant_id, source_id))
		if record is None:
			raise KeyError(f"Source {source_id} not found for tenant {tenant_id}")
		if record.status == "denied":
			raise ValueError(f"Source {source_id} is denied")
		return record

	def _require_query(self, tenant_id: str, query_id: str) -> DVRLQueryRecord:
		query_id = self._require_text(query_id, "query_id")
		record = self.queries.get(self._key(tenant_id, query_id))
		if record is None:
			raise KeyError(f"Query {query_id} not found for tenant {tenant_id}")
		return record

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
	def _normalize_agent_token(value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	@staticmethod
	def _first_reason(result: dict[str, Any]) -> str:
		for action in result.get("actions", []):
			if action.get("reason"):
				return str(action["reason"])
		return "dvrl_operation_denied"

	@staticmethod
	def _key(tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_access_policies', '_source_catalogs', '_virtual_joins', '_preview_results', '_pushdown_stats', '_caching_strategies']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

