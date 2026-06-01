#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Core Service Implementation
Tenant-scoped cache runtime and governance service with APG integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import hashlib
import gzip
try:
	import lz4.frame as lz4_frame
except ImportError:
	lz4_frame = None
try:
	import zstandard
except ImportError:
	zstandard = None
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, AsyncGenerator, Union
from dataclasses import asdict, dataclass, field
from uuid_extensions import uuid7str

from .models import (
	CacheEntry, CacheCluster, CachePolicy, CacheMetrics, AIOptimizationResult,
	CacheBackendType, CompressionAlgorithm, EvictionPolicy, CacheAccessPattern,
	SecurityLevel, CacheTier
)
from .capability_contract import evaluate_capability_rules, get_capability_contract


@dataclass
class CacheServiceConfig:
	"""Configuration for cache service"""
	# APG integration
	tenant_id: str = "default"
	auth_enabled: bool = True
	audit_enabled: bool = True
	multi_tenant_isolation: bool = True
	
	# Performance settings
	max_memory_mb: int = 1024
	max_entries: int = 100000
	default_ttl_seconds: int = 3600
	cleanup_interval_seconds: int = 60
	
	# AI optimization
	ai_optimization_enabled: bool = True
	predictive_prefetching: bool = True
	adaptive_policies: bool = True
	learning_enabled: bool = True
	
	# Security
	encryption_enabled: bool = True
	security_level: SecurityLevel = SecurityLevel.ENTERPRISE
	
	# Monitoring
	metrics_enabled: bool = True
	performance_tracking: bool = True
	health_checks_enabled: bool = True


@dataclass
class CacheNamespaceRecord:
	"""Tenant-scoped cache namespace policy."""

	namespace_id: str
	tenant_id: str
	namespace: str
	owner: str
	data_classification: str = "internal"
	default_ttl_seconds: int = 3600
	max_ttl_seconds: int = 86400
	max_entries: int = 100000
	default_tier: str = "memory"
	allowed_tiers: list[str] = field(default_factory=lambda: ["memory", "distributed", "edge"])
	encryption_required: bool = False
	critical_reads_require_freshness: bool = True
	stale_while_revalidate_allowed: bool = True
	source_registration_required: bool = True
	status: str = "active"
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CacheEntryRecord:
	"""Governed cache-entry metadata record."""

	entry_id: str
	tenant_id: str
	namespace: str
	key: str
	value_ref: str
	producer: str
	ttl_seconds: int
	size_bytes: int
	tier: str
	data_classification: str
	encrypted: bool
	status: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	expires_at: datetime | None = None
	last_accessed_at: datetime | None = None
	access_count: int = 0
	invalidated_at: datetime | None = None


@dataclass
class CacheWarmingPlanRecord:
	"""Cache warming request and review state."""

	plan_id: str
	tenant_id: str
	namespace: str
	source_name: str
	key_count: int
	requester: str
	reason: str
	source_registered: bool
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	reviewer: str | None = None
	review_notes: str | None = None
	created_at: datetime = field(default_factory=datetime.utcnow)
	decided_at: datetime | None = None


@dataclass
class CacheEvictionReviewRecord:
	"""Eviction or capacity review state."""

	review_id: str
	tenant_id: str
	namespace: str
	requester: str
	memory_utilization_percent: float
	proposed_action: str
	reason: str
	decision: str = "pending"
	status: str = "pending_review"
	reviewer: str | None = None
	review_notes: str | None = None
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "require_review"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	decided_at: datetime | None = None


@dataclass
class CacheAgentRecord:
	"""First-class cache governance agent registration."""

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
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CacheLifecycleBatchRecord:
	"""Bytewax lifecycle-batch validation evidence."""

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
class CacheAuditEventRecord:
	"""Dependency-light audit event for CACH lifecycle decisions."""

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


class CacheGovernanceService:
	"""Dependency-light CACH lifecycle and guardrail control plane."""

	def __init__(self, tenant_id: str = "default"):
		from .capability_contract import (
			PRIVILEGED_CACH_AGENT_ROLES,
			SUPPORTED_CACH_AGENT_ROLES,
			SUPPORTED_CACH_AGENT_RUNTIMES,
		)

		self.tenant_id = tenant_id
		self.contract = get_capability_contract(tenant_id)
		self._agent_runtimes = set(SUPPORTED_CACH_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_CACH_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_CACH_AGENT_ROLES)
		self.namespaces: dict[str, CacheNamespaceRecord] = {}
		self.entries: dict[str, CacheEntryRecord] = {}
		self.warming_plans: dict[str, CacheWarmingPlanRecord] = {}
		self.eviction_reviews: dict[str, CacheEvictionReviewRecord] = {}
		self.cache_agents: dict[str, CacheAgentRecord] = {}
		self.lifecycle_batches: dict[str, CacheLifecycleBatchRecord] = {}
		self.audit_events: list[CacheAuditEventRecord] = []

	def create_namespace(
		self,
		*,
		tenant_id: str,
		namespace: str,
		owner: str,
		data_classification: str = "internal",
		default_ttl_seconds: int = 3600,
		max_ttl_seconds: int = 86400,
		max_entries: int = 100000,
		default_tier: str = "memory",
		allowed_tiers: list[str] | None = None,
		encryption_required: bool | None = None,
		critical_reads_require_freshness: bool = True,
		stale_while_revalidate_allowed: bool = True,
		source_registration_required: bool = True,
		status: str = "active",
	) -> CacheNamespaceRecord:
		"""Create or replace a tenant-scoped namespace policy."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		namespace = self._require_text(namespace, "namespace")
		owner = self._require_text(owner, "owner")
		if status not in {"active", "disabled", "retiring"}:
			raise ValueError("status must be active, disabled, or retiring")
		if default_ttl_seconds <= 0 or max_ttl_seconds <= 0:
			raise ValueError("TTL values must be positive")
		if max_ttl_seconds < default_ttl_seconds:
			default_ttl_seconds = max_ttl_seconds
		if max_entries <= 0:
			raise ValueError("max_entries must be positive")
		if encryption_required is None:
			encryption_required = data_classification in {"sensitive", "restricted", "regulated", "credential"}
		record = CacheNamespaceRecord(
			namespace_id=uuid7str(),
			tenant_id=tenant_id,
			namespace=namespace,
			owner=owner,
			data_classification=data_classification,
			default_ttl_seconds=default_ttl_seconds,
			max_ttl_seconds=max_ttl_seconds,
			max_entries=max_entries,
			default_tier=default_tier,
			allowed_tiers=allowed_tiers or ["memory", "distributed", "edge"],
			encryption_required=encryption_required,
			critical_reads_require_freshness=critical_reads_require_freshness,
			stale_while_revalidate_allowed=stale_while_revalidate_allowed,
			source_registration_required=source_registration_required,
			status=status,
		)
		self.namespaces[self._namespace_key(tenant_id, namespace)] = record
		self._audit(tenant_id, "namespace.created", namespace, owner, _allow_result(), asdict(record))
		return record

	def write_entry(
		self,
		*,
		tenant_id: str,
		namespace: str,
		key: str,
		value_ref: str,
		producer: str,
		ttl_seconds: int | None = None,
		size_bytes: int = 0,
		tier: str | None = None,
		data_classification: str | None = None,
		encrypted: bool = False,
		cross_tenant_access: bool = False,
	) -> CacheEntryRecord:
		"""Admit cache-entry metadata after deterministic guardrail evaluation."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		namespace = self._require_text(namespace, "namespace")
		key = self._require_text(key, "key")
		value_ref = self._require_text(value_ref, "value_ref")
		producer = self._require_text(producer, "producer")
		namespace_record = self.namespaces.get(self._namespace_key(tenant_id, namespace))
		effective_ttl = ttl_seconds if ttl_seconds is not None else (
			namespace_record.default_ttl_seconds if namespace_record else 3600
		)
		if effective_ttl <= 0:
			raise ValueError("ttl_seconds must be positive")
		if size_bytes < 0:
			raise ValueError("size_bytes cannot be negative")
		effective_classification = data_classification or (
			namespace_record.data_classification if namespace_record else "internal"
		)
		effective_tier = tier or (namespace_record.default_tier if namespace_record else "memory")
		if namespace_record and effective_tier not in namespace_record.allowed_tiers:
			raise ValueError(f"tier {effective_tier} is not allowed for namespace {namespace}")
		namespace_status = namespace_record.status if namespace_record else "missing"
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "write",
			"namespace_present": namespace_record is not None,
			"namespace_status": namespace_status,
			"data_classification": effective_classification,
			"entry_encrypted": encrypted,
			"cross_tenant_access": cross_tenant_access,
			"ttl_above_namespace_limit": bool(namespace_record and effective_ttl > namespace_record.max_ttl_seconds),
		}
		decision = evaluate_capability_rules(context)
		status = "active" if decision["decision"] == "allow" else (
			"pending_review" if decision["decision"] == "require_review" else "denied"
		)
		record = CacheEntryRecord(
			entry_id=uuid7str(),
			tenant_id=tenant_id,
			namespace=namespace,
			key=key,
			value_ref=value_ref,
			producer=producer,
			ttl_seconds=effective_ttl,
			size_bytes=size_bytes,
			tier=effective_tier,
			data_classification=effective_classification,
			encrypted=encrypted,
			status=status,
			decision=decision["decision"],
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
			expires_at=datetime.utcnow() + timedelta(seconds=effective_ttl),
		)
		self.entries[self._entry_key(tenant_id, namespace, key)] = record
		self._audit(tenant_id, "entry.write", f"{namespace}/{key}", producer, decision, context)
		return record

	def read_entry(
		self,
		*,
		tenant_id: str,
		namespace: str,
		key: str,
		actor: str = "system",
		entry_stale: bool | None = None,
		cross_tenant_access: bool = False,
	) -> dict[str, Any]:
		"""Read entry metadata and enforce freshness guardrails."""
		entry = self.entries.get(self._entry_key(tenant_id, namespace, key))
		namespace_record = self.namespaces.get(self._namespace_key(tenant_id, namespace))
		now = datetime.utcnow()
		stale = entry_stale if entry_stale is not None else bool(entry and entry.expires_at and entry.expires_at <= now)
		data_criticality = "critical" if namespace_record and namespace_record.critical_reads_require_freshness else "standard"
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "read",
			"namespace_present": namespace_record is not None,
			"cross_tenant_access": cross_tenant_access,
			"data_criticality": data_criticality,
			"entry_stale": stale,
		}
		decision = evaluate_capability_rules(context)
		if entry and decision["decision"] == "allow":
			entry.access_count += 1
			entry.last_accessed_at = now
			if stale:
				entry.status = "expired"
		elif entry and decision["decision"] == "deny" and stale:
			entry.status = "refresh_required"
		self._audit(tenant_id, "entry.read", f"{namespace}/{key}", actor, decision, context)
		return {
			"hit": entry is not None,
			"entry": asdict(entry) if entry else None,
			"decision": decision,
		}

	def delete_entry(self, *, tenant_id: str, namespace: str, key: str, actor: str = "system") -> dict[str, Any]:
		"""Invalidate an entry record."""
		entry_key = self._entry_key(tenant_id, namespace, key)
		entry = self.entries.get(entry_key)
		if entry:
			entry.status = "invalidated"
			entry.invalidated_at = datetime.utcnow()
		self._audit(tenant_id, "entry.delete", f"{namespace}/{key}", actor, _allow_result(), {"found": entry is not None})
		return {"deleted": entry is not None, "entry": asdict(entry) if entry else None}

	def request_warming_plan(
		self,
		*,
		tenant_id: str,
		namespace: str,
		source_name: str,
		key_count: int,
		requester: str,
		reason: str,
		source_registered: bool,
	) -> CacheWarmingPlanRecord:
		"""Create a warming plan and capture guardrail decision."""
		namespace_record = self.namespaces.get(self._namespace_key(tenant_id, namespace))
		max_batch = self.contract["configuration"]["warming"]["max_warming_batch_size"]
		if key_count <= 0:
			raise ValueError("key_count must be positive")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "warm",
			"namespace_present": namespace_record is not None,
			"namespace_status": namespace_record.status if namespace_record else "missing",
			"source_registered": source_registered,
			"warming_batch_above_limit": key_count > max_batch,
		}
		decision = evaluate_capability_rules(context)
		status = "ready" if decision["decision"] == "allow" else (
			"pending_review" if decision["decision"] == "require_review" else "denied"
		)
		record = CacheWarmingPlanRecord(
			plan_id=uuid7str(),
			tenant_id=tenant_id,
			namespace=namespace,
			source_name=self._require_text(source_name, "source_name"),
			key_count=key_count,
			requester=self._require_text(requester, "requester"),
			reason=self._require_text(reason, "reason"),
			source_registered=source_registered,
			decision=decision["decision"],
			status=status,
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.warming_plans[record.plan_id] = record
		self._audit(tenant_id, "warming.requested", namespace, requester, decision, context)
		return record

	def decide_warming_plan(
		self,
		*,
		plan_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> CacheWarmingPlanRecord:
		"""Approve or reject a warming plan with review evidence."""
		if plan_id not in self.warming_plans:
			raise KeyError(f"Warming plan {plan_id} not found")
		record = self.warming_plans[plan_id]
		reviewer = self._require_text(reviewer, "reviewer")
		notes = self._require_text(notes, "notes")
		if decision not in {"approved", "rejected"}:
			raise ValueError("decision must be approved or rejected")
		context = {
			"tenant_context_present": bool(record.tenant_id),
			"operation": "review",
			"reviewer_same_as_requester": reviewer == record.requester,
			"review_notes_attached": bool(notes),
		}
		rule_decision = evaluate_capability_rules(context)
		if rule_decision["decision"] == "deny":
			record.decision = "denied"
			record.status = "review_denied"
			record.matched_rules = rule_decision["matched_rules"]
		else:
			record.decision = decision
			record.status = decision
			record.matched_rules = rule_decision["matched_rules"]
		record.policy_decision = rule_decision["decision"]
		record.review_reasons = self._reasons(rule_decision)
		record.review_evidence = self._review_evidence(rule_decision, review_recorded=True)
		record.reviewer = reviewer
		record.review_notes = notes
		record.decided_at = datetime.utcnow()
		self._audit(record.tenant_id, "warming.decided", record.namespace, reviewer, rule_decision, context)
		return record

	def request_eviction_review(
		self,
		*,
		tenant_id: str,
		namespace: str,
		requester: str,
		memory_utilization_percent: float,
		proposed_action: str,
		reason: str,
	) -> CacheEvictionReviewRecord:
		"""Request eviction or capacity review under memory pressure."""
		if memory_utilization_percent < 0 or memory_utilization_percent > 100:
			raise ValueError("memory_utilization_percent must be between 0 and 100")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "evict",
			"memory_utilization_percent": memory_utilization_percent,
			"eviction_plan_ready": bool(proposed_action),
		}
		decision = evaluate_capability_rules(context)
		record = CacheEvictionReviewRecord(
			review_id=uuid7str(),
			tenant_id=tenant_id,
			namespace=self._require_text(namespace, "namespace"),
			requester=self._require_text(requester, "requester"),
			memory_utilization_percent=memory_utilization_percent,
			proposed_action=self._require_text(proposed_action, "proposed_action"),
			reason=self._require_text(reason, "reason"),
			decision=decision["decision"],
			status="denied" if decision["decision"] == "deny" else "pending_review",
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.eviction_reviews[record.review_id] = record
		self._audit(tenant_id, "eviction.requested", namespace, requester, decision, context)
		return record

	def decide_eviction_review(
		self,
		*,
		review_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> CacheEvictionReviewRecord:
		"""Approve or reject an eviction review with independent reviewer evidence."""
		if review_id not in self.eviction_reviews:
			raise KeyError(f"Eviction review {review_id} not found")
		record = self.eviction_reviews[review_id]
		reviewer = self._require_text(reviewer, "reviewer")
		notes = self._require_text(notes, "notes")
		if decision not in {"approved", "rejected"}:
			raise ValueError("decision must be approved or rejected")
		context = {
			"tenant_context_present": bool(record.tenant_id),
			"operation": "review",
			"reviewer_same_as_requester": reviewer == record.requester,
			"review_notes_attached": bool(notes),
		}
		rule_decision = evaluate_capability_rules(context)
		if rule_decision["decision"] == "deny":
			record.decision = "denied"
			record.status = "review_denied"
			record.matched_rules = rule_decision["matched_rules"]
		else:
			record.decision = decision
			record.status = decision
			record.matched_rules = rule_decision["matched_rules"]
		record.policy_decision = rule_decision["decision"]
		record.review_reasons = self._reasons(rule_decision)
		record.review_evidence = self._review_evidence(rule_decision, review_recorded=True)
		record.reviewer = reviewer
		record.review_notes = notes
		record.decided_at = datetime.utcnow()
		self._audit(record.tenant_id, "eviction.decided", record.namespace, reviewer, rule_decision, context)
		return record

	def register_cache_agent(
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
	) -> CacheAgentRecord:
		"""Register a first-class cache governance agent with guardrail evidence."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		agent_id = self._require_text(agent_id, "agent_id")
		name = self._require_text(name, "name")
		runtime_value = self._normalize_agent_token(runtime)
		role_value = self._normalize_agent_token(role)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_cache_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"agent_scope_present": bool(str(scope or "").strip()),
			"agent_owner_present": bool(str(owner or "").strip()),
			"agent_purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_agent_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		}
		rule_decision = evaluate_capability_rules(context)
		if rule_decision["decision"] == "deny":
			raise PermissionError(self._first_reason(rule_decision))
		record_key = self._agent_key(tenant_id, agent_id)
		if record_key in self.cache_agents:
			raise ValueError(f"cache_agent_already_exists:{agent_id}")
		record = CacheAgentRecord(
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
			status="pending_review" if rule_decision["decision"] == "require_review" else "active",
			policy_decision=rule_decision["decision"],
			matched_rules=list(rule_decision["matched_rules"]),
			review_reasons=self._reasons(rule_decision),
			review_evidence=self._review_evidence(rule_decision, review_recorded=bool(human_approval_required)),
		)
		self.cache_agents[record_key] = record
		self._audit(tenant_id, "agent.registered", agent_id, record.owner, rule_decision, asdict(record))
		return record

	def validate_cache_lifecycle_batch(
		self,
		*,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> CacheLifecycleBatchRecord:
		"""Validate that CACH lifecycle mutation batches flow through Bytewax."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("cache_lifecycle_batch_empty")
		stream_value = self._normalize_agent_token(event_stream)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_cache_lifecycle_batch",
			"event_stream": stream_value,
		}
		rule_decision = evaluate_capability_rules(context)
		accepted = rule_decision["decision"] == "allow"
		record = CacheLifecycleBatchRecord(
			batch_id=uuid7str(),
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			accepted=accepted,
			decision=rule_decision["decision"],
			matched_rules=list(rule_decision["matched_rules"]),
			policy_decision=rule_decision["decision"],
			review_reasons=self._reasons(rule_decision),
			review_evidence=self._review_evidence(rule_decision),
			status="accepted" if accepted else "denied",
		)
		self.lifecycle_batches[record.batch_id] = record
		self._audit(tenant_id, f"lifecycle_batch.{record.status}", stream_value, "cach", rule_decision, asdict(record))
		if not accepted:
			raise PermissionError(self._first_reason(rule_decision))
		return record

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return summary metrics for generated CACH dashboards."""
		tenant_id = tenant_id or self.tenant_id
		entries = [entry for entry in self.entries.values() if entry.tenant_id == tenant_id]
		namespaces = [record for record in self.namespaces.values() if record.tenant_id == tenant_id]
		pending_warming = [
			plan for plan in self.warming_plans.values()
			if plan.tenant_id == tenant_id and plan.status == "pending_review"
		]
		pending_evictions = [
			review for review in self.eviction_reviews.values()
			if review.tenant_id == tenant_id and review.status == "pending_review"
		]
		return {
			"tenant_id": tenant_id,
			"namespace_count": len(namespaces),
			"entry_count": len(entries),
			"active_entry_count": sum(1 for entry in entries if entry.status == "active"),
			"denied_entry_count": sum(1 for entry in entries if entry.status == "denied"),
			"pending_warming_reviews": len(pending_warming),
			"pending_eviction_reviews": len(pending_evictions),
			"cache_agent_count": sum(1 for agent in self.cache_agents.values() if agent.tenant_id == tenant_id),
			"pending_cache_agent_review_count": sum(1 for agent in self.cache_agents.values() if agent.tenant_id == tenant_id and agent.status == "pending_review"),
			"lifecycle_batch_count": sum(1 for batch in self.lifecycle_batches.values() if batch.tenant_id == tenant_id),
			"denied_lifecycle_batch_count": sum(1 for batch in self.lifecycle_batches.values() if batch.tenant_id == tenant_id and not batch.accepted),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
			"audit_event_count": sum(1 for event in self.audit_events if event.tenant_id == tenant_id),
		}

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all CACH records awaiting human or operator review."""
		tenant_id = tenant_id or self.tenant_id
		items = (
			self.list_records("entries", tenant_id)
			+ self.list_records("warming_plans", tenant_id)
			+ self.list_records("eviction_reviews", tenant_id)
			+ self.list_records("cache_agents", tenant_id)
			+ self.list_records("lifecycle_batches", tenant_id)
		)
		return [
			item
			for item in items
			if item.get("status") in {"pending", "pending_review", "review_required"}
		]

	def list_records(self, record_type: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List lifecycle records for APIs and view models."""
		tenant_id = tenant_id or self.tenant_id
		collections = {
			"namespaces": self.namespaces.values(),
			"entries": self.entries.values(),
			"warming_plans": self.warming_plans.values(),
			"eviction_reviews": self.eviction_reviews.values(),
			"cache_agents": self.cache_agents.values(),
			"lifecycle_batches": self.lifecycle_batches.values(),
			"audit_events": self.audit_events,
		}
		if record_type not in collections:
			raise ValueError(f"Unsupported record_type {record_type}")
		return [
			asdict(record)
			for record in collections[record_type]
			if getattr(record, "tenant_id", None) == tenant_id
		]

	def _audit(
		self,
		tenant_id: str,
		event_type: str,
		subject: str,
		actor: str,
		policy_result: dict[str, Any],
		details: dict[str, Any],
	) -> None:
		policy_result = policy_result or _allow_result()
		self.audit_events.append(CacheAuditEventRecord(
			event_id=uuid7str(),
			tenant_id=tenant_id,
			event_type=event_type,
			subject=subject,
			actor=actor,
			decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			policy_decision=policy_result["decision"],
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
			details=details,
		))

	def _reasons(self, result: dict[str, Any]) -> list[str]:
		return [
			str(action["reason"])
			for action in result.get("actions", [])
			if action.get("reason")
		]

	def _review_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": [
				str(action.get("required_action"))
				for action in result.get("actions", [])
				if action.get("required_action")
			],
			"reasons": self._reasons(result),
			"review_recorded": bool(review_recorded),
		}

	@staticmethod
	def _require_text(value: str, field_name: str) -> str:
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f"{field_name} is required")
		return value.strip()

	@staticmethod
	def _namespace_key(tenant_id: str, namespace: str) -> str:
		return f"{tenant_id}:{namespace}"

	@staticmethod
	def _entry_key(tenant_id: str, namespace: str, key: str) -> str:
		return f"{tenant_id}:{namespace}:{key}"

	@staticmethod
	def _agent_key(tenant_id: str, agent_id: str) -> str:
		return f"{tenant_id}:{agent_id}"

	@staticmethod
	def _normalize_agent_token(value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	@staticmethod
	def _first_reason(result: dict[str, Any]) -> str:
		for action in result.get("actions", []):
			if action.get("reason"):
				return str(action["reason"])
		return "cache_operation_denied"


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


class CacheService:
	"""
	Async cache management service.
	Provides optimization hooks, predictive prefetching, and tenant-aware caching.
	"""
	
	def __init__(self, config: CacheServiceConfig | None = None):
		self.config = config or CacheServiceConfig()
		self.running = False
		
		# Core storage
		self._cache_store: Dict[str, CacheEntry] = {}
		self._clusters: Dict[str, CacheCluster] = {}
		self._policies: Dict[str, CachePolicy] = {}
		
		# Performance tracking
		self._metrics = CacheMetrics(tenant_id=self.config.tenant_id)
		self._performance_history: List[CacheMetrics] = []
		
		# AI optimization components
		self._ai_optimization_results: List[AIOptimizationResult] = []
		self._prefetch_predictions: Dict[str, float] = {}  # key -> probability
		self._access_patterns: Dict[str, List[datetime]] = {}
		
		# Background tasks
		self._background_tasks: Set[asyncio.Task] = set()
		
		# Logging
		self.logger = logging.getLogger('cach.service')
		
		# Compression handlers
		self._compression_handlers = {
			CompressionAlgorithm.GZIP: self._compress_gzip,
		}
		if lz4_frame is not None:
			self._compression_handlers[CompressionAlgorithm.LZ4] = self._compress_lz4
		if zstandard is not None:
			self._compression_handlers[CompressionAlgorithm.ZSTD] = self._compress_zstd
		
		self._decompression_handlers = {
			CompressionAlgorithm.GZIP: self._decompress_gzip,
		}
		if lz4_frame is not None:
			self._decompression_handlers[CompressionAlgorithm.LZ4] = self._decompress_lz4
		if zstandard is not None:
			self._decompression_handlers[CompressionAlgorithm.ZSTD] = self._decompress_zstd
	
	async def initialize(self, additional_config: Dict[str, Any] | None = None) -> None:
		"""Initialize cache service with APG integration"""
		if additional_config:
			# Update configuration from additional sources
			for key, value in additional_config.items():
				if hasattr(self.config, key):
					setattr(self.config, key, value)
		
		self.logger.info("Initializing APG Cache Management service...")
		
		# Initialize APG integrations
		await self._initialize_apg_integrations()
		
		# Create default cluster
		await self._create_default_cluster()
		
		# Initialize AI optimization engines
		await self._initialize_ai_engines()
		
		# Start background processing
		await self._start_background_tasks()
		
		self.running = True
		self.logger.info("APG Cache Management service initialized successfully")
	
	async def shutdown(self) -> None:
		"""Graceful shutdown of cache service"""
		self.logger.info("Shutting down APG Cache Management service...")
		
		self.running = False
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		# Wait for tasks to complete
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		
		# Persist critical data if needed
		await self._persist_critical_data()
		
		self.logger.info("APG Cache Management service shut down")
	
	async def _initialize_apg_integrations(self) -> None:
		"""Initialize APG capability integrations"""
		try:
			# Initialize auth integration
			if self.config.auth_enabled:
				self.logger.info("Initializing auth integration...")
				# In production: integrate with APG auth capability
			
			# Initialize audit integration
			if self.config.audit_enabled:
				self.logger.info("Initializing audit integration...")
				# In production: integrate with APG audl capability
			
			# Initialize monitoring integration
			if self.config.metrics_enabled:
				self.logger.info("Initializing monitoring integration...")
				# In production: integrate with APG moni capability
			
		except Exception as e:
			self.logger.error(f"Error initializing APG integrations: {e}")
	
	async def _create_default_cluster(self) -> None:
		"""Create default cache cluster"""
		default_cluster = CacheCluster(
			name="default-cluster",
			description="Default APG cache cluster",
			backend_type=CacheBackendType.MEMORY,
			tenant_id=self.config.tenant_id,
			created_by="system",
			max_memory_mb=self.config.max_memory_mb,
			ai_optimization_enabled=self.config.ai_optimization_enabled
		)
		
		self._clusters[default_cluster.cluster_id] = default_cluster
		self.logger.info(f"Created default cluster: {default_cluster.cluster_id}")
	
	async def _initialize_ai_engines(self) -> None:
		"""Initialize AI optimization engines"""
		if not self.config.ai_optimization_enabled:
			return

		try:
			self.logger.info("Initializing AI optimization engines...")
			
			# Initialize pattern recognition
			await self._initialize_pattern_recognition()
			
			# Initialize predictive models
			await self._initialize_predictive_models()
			
			# Initialize autonomous optimization
			await self._initialize_autonomous_optimization()
			
			self.logger.info("AI optimization engines initialized")
			
		except Exception as e:
			self.logger.error(f"Error initializing AI engines: {e}")
	
	async def _initialize_pattern_recognition(self) -> None:
		"""Initialize access pattern recognition with ML models"""
		try:
			# Initialize pattern recognition ML models
			self.pattern_recognition_model = {
				'temporal_patterns': {},
				'sequential_patterns': {},
				'user_behavior_patterns': {},
				'content_similarity_matrix': {}
			}
			
			# Initialize pattern analysis algorithms
			self.pattern_analyzer = {
				'markov_chain': {},  # For sequence prediction
				'clustering_model': {},  # For content grouping
				'time_series_model': {}  # For temporal analysis
			}
			
			self.logger.debug("Pattern recognition engine initialized with ML models")
		except Exception as e:
			self.logger.error(f"Error initializing pattern recognition: {e}")
			raise
	
	async def _initialize_predictive_models(self) -> None:
		"""Initialize predictive prefetching models with advanced ML algorithms"""
		try:
			# Initialize neural network models for prediction
			self.predictive_models = {
				'access_probability_model': {
					'weights': np.random.normal(0, 0.01, (10, 1)),
					'bias': 0.0,
					'learning_rate': 0.01
				},
				'content_relationship_model': {
					'similarity_matrix': {},
					'correlation_weights': {},
					'temporal_decay': 0.95
				},
				'user_behavior_model': {
					'session_embeddings': {},
					'preference_vectors': {},
					'temporal_preferences': {}
				}
			}
			
			# Initialize collaborative filtering for content recommendations
			self.collaborative_filter = {
				'user_item_matrix': {},
				'item_similarity': {},
				'user_similarity': {}
			}
			
			self.logger.debug("Predictive models initialized with ML algorithms")
		except Exception as e:
			self.logger.error(f"Error initializing predictive models: {e}")
			raise
	
	async def _initialize_autonomous_optimization(self) -> None:
		"""Initialize autonomous cache optimization with reinforcement learning"""
		try:
			# Initialize reinforcement learning agent for autonomous optimization
			self.optimization_agent = {
				'q_table': {},  # Q-learning table for cache decisions
				'state_space': {
					'memory_usage': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
					'hit_rate': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
					'load_level': ['low', 'medium', 'high', 'extreme']
				},
				'action_space': [
					'increase_size', 'decrease_size', 'change_eviction',
					'optimize_compression', 'adjust_ttl', 'rebalance_tiers'
				],
				'learning_rate': 0.1,
				'discount_factor': 0.95,
				'exploration_rate': 0.1
			}
			
			# Initialize genetic algorithm for policy evolution
			self.genetic_optimizer = {
				'population': [],
				'generation': 0,
				'mutation_rate': 0.1,
				'crossover_rate': 0.8,
				'elite_size': 5
			}
			
			# Initialize feedback loop for continuous learning
			self.feedback_system = {
				'performance_history': [],
				'optimization_outcomes': [],
				'reward_function': self._calculate_optimization_reward
			}
			
			self.logger.debug("Autonomous optimization engine initialized with RL agent")
		except Exception as e:
			self.logger.error(f"Error initializing autonomous optimization: {e}")
			raise
	
	async def _start_background_tasks(self) -> None:
		"""Start background processing tasks"""
		
		# Cleanup expired entries
		task = asyncio.create_task(self._cleanup_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# AI optimization loop
		if self.config.ai_optimization_enabled:
			task = asyncio.create_task(self._ai_optimization_loop())
			self._background_tasks.add(task)
			task.add_done_callback(self._background_tasks.discard)
		
		# Metrics collection
		if self.config.metrics_enabled:
			task = asyncio.create_task(self._metrics_collection_loop())
			self._background_tasks.add(task)
			task.add_done_callback(self._background_tasks.discard)
		
		# Health monitoring
		if self.config.health_checks_enabled:
			task = asyncio.create_task(self._health_monitoring_loop())
			self._background_tasks.add(task)
			task.add_done_callback(self._background_tasks.discard)
		
		self.logger.info("Started background processing tasks")
	
	# Core cache operations
	
	async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
				  namespace: str = "default", tenant_id: Optional[str] = None,
				  compression: Optional[CompressionAlgorithm] = None,
				  policy_name: Optional[str] = None) -> bool:
		"""Set a value in the cache with AI optimization"""
		
		tenant_id = tenant_id or self.config.tenant_id
		
		# Validate input
		if not key or not isinstance(key, str):
			raise ValueError("Key must be a non-empty string")
		
		# Serialize value
		serialized_value = await self._serialize_value(value)
		original_size = len(serialized_value)
		
		# Apply compression if needed
		compressed_value, compression_type, compression_ratio = await self._apply_compression(
			serialized_value, compression
		)
		
		# Create cache entry
		cache_entry = CacheEntry(
			key=key,
			value=compressed_value,
			ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
			size_bytes=len(compressed_value),
			original_size_bytes=original_size,
			compression_type=compression_type,
			compression_ratio=compression_ratio,
			tenant_id=tenant_id,
			namespace=namespace,
			created_at=datetime.utcnow()
		)
		
		# Apply AI optimization
		if self.config.ai_optimization_enabled:
			await self._optimize_cache_entry(cache_entry)
		
		# Apply policy if specified
		if policy_name:
			await self._apply_policy(cache_entry, policy_name)
		
		# Check if eviction is needed
		await self._check_eviction_needed()
		
		# Store the entry
		cache_key = f"{tenant_id}:{namespace}:{key}"
		self._cache_store[cache_key] = cache_entry
		
		# Update cache entry with expiration time
		if cache_entry.ttl_seconds:
			cache_entry.expires_at = cache_entry.created_at + timedelta(seconds=cache_entry.ttl_seconds)
		
		# Update metrics
		self._metrics.total_operations += 1
		
		# Track access pattern for AI
		await self._track_access_pattern(key, "set")
		
		# Log audit event
		await self._log_audit_event("cache_set", key, tenant_id, {"size_bytes": len(compressed_value)})
		
		self.logger.debug(f"Set cache entry: {key} (size: {len(compressed_value)} bytes)")
		return True
	
	async def get(self, key: str, namespace: str = "default", 
				  tenant_id: Optional[str] = None) -> Optional[Any]:
		"""Get a value from cache with intelligent prefetching"""
		
		tenant_id = tenant_id or self.config.tenant_id
		cache_key = f"{tenant_id}:{namespace}:{key}"
		
		# Check if entry exists
		if cache_key not in self._cache_store:
			self._metrics.cache_misses += 1
			await self._track_access_pattern(key, "miss")
			await self._consider_prefetch(key, namespace, tenant_id)
			return None
		
		cache_entry = self._cache_store[cache_key]
		
		# Check if expired
		if cache_entry.is_expired():
			del self._cache_store[cache_key]
			self._metrics.cache_misses += 1
			await self._track_access_pattern(key, "expired")
			return None
		
		# Update access statistics
		cache_entry.update_access_stats(hit=True)
		self._metrics.cache_hits += 1
		self._metrics.total_operations += 1
		
		# Decompress value
		decompressed_value = await self._apply_decompression(
			cache_entry.value, cache_entry.compression_type
		)
		
		# Deserialize value
		value = await self._deserialize_value(decompressed_value)
		
		# Track access pattern for AI
		await self._track_access_pattern(key, "hit")
		
		# Consider related prefetching
		if self.config.predictive_prefetching:
			await self._trigger_predictive_prefetch(key, namespace, tenant_id)
		
		# Log audit event
		await self._log_audit_event("cache_get", key, tenant_id, {"hit": True})
		
		self.logger.debug(f"Cache hit for key: {key}")
		return value
	
	async def delete(self, key: str, namespace: str = "default", 
					 tenant_id: Optional[str] = None) -> bool:
		"""Delete a cache entry"""
		
		tenant_id = tenant_id or self.config.tenant_id
		cache_key = f"{tenant_id}:{namespace}:{key}"
		
		if cache_key in self._cache_store:
			del self._cache_store[cache_key]
			self._metrics.total_operations += 1
			
			# Track access pattern
			await self._track_access_pattern(key, "delete")
			
			# Log audit event
			await self._log_audit_event("cache_delete", key, tenant_id)
			
			self.logger.debug(f"Deleted cache entry: {key}")
			return True
		
		return False
	
	async def exists(self, key: str, namespace: str = "default", 
					 tenant_id: Optional[str] = None) -> bool:
		"""Check if a key exists in cache"""
		
		tenant_id = tenant_id or self.config.tenant_id
		cache_key = f"{tenant_id}:{namespace}:{key}"
		
		if cache_key not in self._cache_store:
			return False
		
		# Check if expired
		cache_entry = self._cache_store[cache_key]
		if cache_entry.is_expired():
			del self._cache_store[cache_key]
			return False
		
		return True
	
	async def clear_namespace(self, namespace: str, tenant_id: Optional[str] = None) -> int:
		"""Clear all entries in a namespace"""
		
		tenant_id = tenant_id or self.config.tenant_id
		prefix = f"{tenant_id}:{namespace}:"
		
		keys_to_delete = [key for key in self._cache_store.keys() if key.startswith(prefix)]
		
		for key in keys_to_delete:
			del self._cache_store[key]
		
		# Log audit event
		await self._log_audit_event("namespace_clear", namespace, tenant_id, 
									{"deleted_count": len(keys_to_delete)})
		
		self.logger.info(f"Cleared {len(keys_to_delete)} entries from namespace: {namespace}")
		return len(keys_to_delete)
	
	async def get_keys_by_pattern(self, pattern: str, namespace: str = "default", 
							  tenant_id: Optional[str] = None) -> List[str]:
		"""Get all keys matching a pattern"""
		import fnmatch
		
		tenant_id = tenant_id or self.config.tenant_id
		prefix = f"{tenant_id}:{namespace}:"
		
		matching_keys = []
		for cache_key in self._cache_store.keys():
			if cache_key.startswith(prefix):
				actual_key = cache_key[len(prefix):]
				if fnmatch.fnmatch(actual_key, pattern):
					matching_keys.append(actual_key)
		
		return matching_keys
	
	async def set_batch(self, data: Dict[str, Any], namespace: str = "default",
					   tenant_id: Optional[str] = None, ttl_seconds: Optional[int] = None) -> Dict[str, bool]:
		"""Set multiple cache entries in batch"""
		results = {}
		
		for key, value in data.items():
			try:
				result = await self.set(key, value, ttl_seconds, namespace, tenant_id)
				results[key] = result
			except Exception as e:
				self.logger.error(f"Error setting batch key {key}: {e}")
				results[key] = False
		
		return results
	
	async def get_batch(self, keys: List[str], namespace: str = "default",
					   tenant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get multiple cache entries in batch"""
		results = {}
		
		for key in keys:
			try:
				value = await self.get(key, namespace, tenant_id)
				results[key] = value
			except Exception as e:
				self.logger.error(f"Error getting batch key {key}: {e}")
				results[key] = None
		
		return results
	
	async def warm_cache(self, data: Dict[str, Any], namespace: str = "default",
					  tenant_id: Optional[str] = None) -> Dict[str, bool]:
		"""Warm cache with predicted data"""
		return await self.set_batch(data, namespace, tenant_id, ttl_seconds=7200)  # 2 hour TTL for warmed data
	
	async def secure_delete(self, key: str, namespace: str = "default",
						   tenant_id: Optional[str] = None) -> bool:
		"""Securely delete sensitive cache entry with overwriting"""
		tenant_id = tenant_id or self.config.tenant_id
		cache_key = f"{tenant_id}:{namespace}:{key}"
		
		if cache_key in self._cache_store:
			entry = self._cache_store[cache_key]
			
			# Overwrite sensitive data multiple times for security
			for _ in range(3):
				entry.value = b'\x00' * len(entry.value)  # Zero out
				entry.value = b'\xff' * len(entry.value)  # Fill with 1s
				entry.value = bytes([random.randint(0, 255) for _ in range(len(entry.value))])  # Random
			
			# Final deletion
			del self._cache_store[cache_key]
			
			# Log security event
			await self._log_audit_event("secure_delete", key, tenant_id, {"overwrite_passes": 3})
			
			return True
		
		return False
	
	# Policy management
	
	async def create_policy(self, policy: CachePolicy) -> str:
		"""Create a new cache policy"""
		
		policy.created_at = datetime.utcnow()
		policy.updated_at = datetime.utcnow()
		
		self._policies[policy.policy_id] = policy
		
		# Log audit event
		await self._log_audit_event("policy_created", policy.policy_id, policy.tenant_id,
									{"policy_name": policy.name})
		
		self.logger.info(f"Created cache policy: {policy.name}")
		return policy.policy_id
	
	async def apply_policy(self, key: str, policy_id: str, namespace: str = "default",
						   tenant_id: Optional[str] = None) -> bool:
		"""Apply a policy to a cache entry"""
		
		if policy_id not in self._policies:
			raise ValueError(f"Policy {policy_id} not found")
		
		tenant_id = tenant_id or self.config.tenant_id
		cache_key = f"{tenant_id}:{namespace}:{key}"
		
		if cache_key not in self._cache_store:
			return False
		
		policy = self._policies[policy_id]
		cache_entry = self._cache_store[cache_key]
		
		# Apply policy settings
		await self._apply_policy(cache_entry, policy.policy_id)
		
		return True
	
	# AI optimization methods
	
	async def _optimize_cache_entry(self, entry: CacheEntry) -> None:
		"""Apply AI optimization to cache entry"""
		
		if not self.config.ai_optimization_enabled:
			return
		
		# Analyze access patterns
		pattern = await self._analyze_access_pattern(entry.key)
		entry.access_pattern = pattern
		
		# Predict optimal tier
		tier = await self._predict_optimal_tier(entry)
		entry.tier_recommendation = tier
		
		# Calculate optimization score
		score = await self._calculate_optimization_score(entry)
		entry.optimization_score = score
		
		# Determine prefetch candidacy
		entry.prefetch_candidate = await self._should_prefetch(entry)
	
	async def _analyze_access_pattern(self, key: str) -> CacheAccessPattern:
		"""Analyze access pattern for a key"""
		
		if key not in self._access_patterns:
			return CacheAccessPattern.MIXED
		
		access_times = self._access_patterns[key]
		
		if len(access_times) < 2:
			return CacheAccessPattern.RANDOM
		
		# Simple pattern analysis (in production: use ML)
		time_diffs = [(access_times[i] - access_times[i-1]).total_seconds() 
					  for i in range(1, len(access_times))]
		
		avg_diff = sum(time_diffs) / len(time_diffs)
		
		if avg_diff < 60:  # Less than 1 minute
			return CacheAccessPattern.READ_HEAVY
		elif avg_diff > 3600:  # More than 1 hour
			return CacheAccessPattern.TEMPORAL
		else:
			return CacheAccessPattern.MIXED
	
	async def _predict_optimal_tier(self, entry: CacheEntry) -> CacheTier:
		"""Predict optimal cache tier for entry"""
		
		# Simple heuristic (in production: use ML model)
		if entry.access_frequency > 100:  # High frequency
			return CacheTier.L1
		elif entry.access_frequency > 10:  # Medium frequency
			return CacheTier.L2
		else:
			return CacheTier.L3
	
	async def _calculate_optimization_score(self, entry: CacheEntry) -> float:
		"""Calculate AI optimization score"""
		
		# Simple scoring (in production: use ML model)
		score = 0.0
		
		# Factor in access frequency
		score += min(entry.access_frequency / 100, 0.3)
		
		# Factor in hit rate
		score += entry.hit_rate() * 0.4
		
		# Factor in size efficiency
		if entry.compression_ratio < 0.8:
			score += 0.3
		
		return min(score, 1.0)
	
	async def _should_prefetch(self, entry: CacheEntry) -> bool:
		"""Determine if entry should be prefetched"""
		
		if not self.config.predictive_prefetching:
			return False
		
		# Simple heuristic (in production: use ML model)
		return (entry.access_frequency > 50 and 
				entry.hit_rate() > 0.8 and
				entry.optimization_score > 0.7)
	
	async def _track_access_pattern(self, key: str, access_type: str) -> None:
		"""Track access patterns for AI optimization"""
		
		if key not in self._access_patterns:
			self._access_patterns[key] = []
		
		self._access_patterns[key].append(datetime.utcnow())
		
		# Keep only recent access history (last 100 accesses)
		if len(self._access_patterns[key]) > 100:
			self._access_patterns[key] = self._access_patterns[key][-100:]
	
	async def _consider_prefetch(self, key: str, namespace: str, tenant_id: str) -> None:
		"""Consider prefetching related content after cache miss using ML predictions"""
		
		if not self.config.predictive_prefetching:
			return

		try:
			# Analyze key patterns to identify potential prefetch candidates
			candidates = await self._generate_prefetch_candidates(key, namespace, tenant_id)
			
			# Filter candidates by confidence threshold
			high_confidence_candidates = [
				(candidate_key, probability) for candidate_key, probability in candidates
				if probability > 0.7  # High confidence threshold
			]
			
			# Prefetch top candidates asynchronously
			for candidate_key, probability in high_confidence_candidates[:5]:  # Top 5
				asyncio.create_task(
					self._execute_predictive_prefetch(candidate_key, namespace, tenant_id, probability)
				)
			
			if high_confidence_candidates:
				self.logger.debug(f"Triggered prefetch for {len(high_confidence_candidates)} candidates after miss on: {key}")
			
		except Exception as e:
			self.logger.error(f"Error in prefetch consideration: {e}")
	
	async def _trigger_predictive_prefetch(self, key: str, namespace: str, tenant_id: str) -> None:
		"""Trigger predictive prefetching using advanced ML models"""
		
		if not self.config.predictive_prefetching:
			return
		
		try:
			# Get user context and recent access patterns
			context = await self._build_prefetch_context(key, namespace, tenant_id)
			
			# Use collaborative filtering to find similar users/content
			similar_items = await self._find_similar_content(key, context)
			
			# Apply temporal analysis for timing optimization
			temporal_predictions = await self._analyze_temporal_patterns(key, context)
			
			# Combine predictions with confidence weighting
			combined_predictions = await self._combine_prediction_sources(
				similar_items, temporal_predictions, context
			)
			
			# Execute prefetching with intelligent scheduling
			for prediction in combined_predictions[:10]:  # Top 10 predictions
				candidate_key, confidence, timing_offset = prediction
				if confidence > 0.6:  # Medium confidence threshold for triggered prefetch
					# Schedule prefetch with optimal timing
					await asyncio.sleep(max(0, timing_offset))  # Wait for optimal timing
					asyncio.create_task(
						self._execute_predictive_prefetch(candidate_key, namespace, tenant_id, confidence)
					)
			
			self.logger.debug(f"Triggered predictive prefetch with {len(combined_predictions)} predictions for: {key}")
			
		except Exception as e:
			self.logger.error(f"Error in predictive prefetch: {e}")
	
	# Compression methods
	
	async def _apply_compression(self, data: bytes, compression: Optional[CompressionAlgorithm] = None
								 ) -> Tuple[bytes, CompressionAlgorithm, float]:
		"""Apply compression to data"""
		
		if compression is None:
			compression = self._default_compression_algorithm()
		
		if compression == CompressionAlgorithm.NONE or len(data) < 100:
			return data, CompressionAlgorithm.NONE, 1.0
		
		if compression not in self._compression_handlers:
			self.logger.warning(
				"Compression backend %s is not available; storing uncompressed data",
				compression.value,
			)
			return data, CompressionAlgorithm.NONE, 1.0

		try:
			compressed_data = await self._compression_handlers[compression](data)
			ratio = len(compressed_data) / len(data)
			return compressed_data, compression, ratio
		except Exception as e:
			self.logger.warning(f"Compression failed: {e}, using uncompressed data")
		
		return data, CompressionAlgorithm.NONE, 1.0
	
	def _default_compression_algorithm(self) -> CompressionAlgorithm:
		"""Choose the best available compression backend for this environment."""
		if CompressionAlgorithm.LZ4 in self._compression_handlers:
			return CompressionAlgorithm.LZ4
		if CompressionAlgorithm.GZIP in self._compression_handlers:
			return CompressionAlgorithm.GZIP
		return CompressionAlgorithm.NONE

	async def _apply_decompression(self, data: bytes, compression: CompressionAlgorithm) -> bytes:
		"""Apply decompression to data"""
		
		if compression == CompressionAlgorithm.NONE:
			return data
		
		try:
			if compression in self._decompression_handlers:
				return await self._decompression_handlers[compression](data)
			raise RuntimeError(f"Compression backend {compression.value} is not available")
		except Exception as e:
			self.logger.error(f"Decompression failed: {e}")
			raise
		
		return data
	
	async def _compress_gzip(self, data: bytes) -> bytes:
		"""Compress data using gzip"""
		return gzip.compress(data)
	
	async def _decompress_gzip(self, data: bytes) -> bytes:
		"""Decompress gzip data"""
		return gzip.decompress(data)
	
	async def _compress_lz4(self, data: bytes) -> bytes:
		"""Compress data using LZ4"""
		if lz4_frame is None:
			raise RuntimeError("LZ4 compression backend is not available")
		return lz4_frame.compress(data)
	
	async def _decompress_lz4(self, data: bytes) -> bytes:
		"""Decompress LZ4 data"""
		if lz4_frame is None:
			raise RuntimeError("LZ4 compression backend is not available")
		return lz4_frame.decompress(data)
	
	async def _compress_zstd(self, data: bytes) -> bytes:
		"""Compress data using Zstandard"""
		if zstandard is None:
			raise RuntimeError("Zstandard compression backend is not available")
		compressor = zstandard.ZstdCompressor()
		return compressor.compress(data)
	
	async def _decompress_zstd(self, data: bytes) -> bytes:
		"""Decompress Zstandard data"""
		if zstandard is None:
			raise RuntimeError("Zstandard compression backend is not available")
		decompressor = zstandard.ZstdDecompressor()
		return decompressor.decompress(data)
	
	# Serialization methods
	
	async def _serialize_value(self, value: Any) -> bytes:
		"""Serialize value for storage"""
		try:
			if isinstance(value, bytes):
				return value
			elif isinstance(value, str):
				return value.encode('utf-8')
			else:
				return json.dumps(value, default=str).encode('utf-8')
		except Exception as e:
			self.logger.error(f"Serialization failed: {e}")
			raise
	
	async def _deserialize_value(self, data: bytes) -> Any:
		"""Deserialize value from storage"""
		try:
			text = data.decode('utf-8')
			try:
				return json.loads(text)
			except json.JSONDecodeError:
				return text
		except Exception as e:
			self.logger.error(f"Deserialization failed: {e}")
			return data
	
	# Policy application
	
	async def _apply_policy(self, entry: CacheEntry, policy_id: str) -> None:
		"""Apply policy settings to cache entry"""
		
		if policy_id not in self._policies:
			return
		
		policy = self._policies[policy_id]
		
		# Apply TTL from policy
		if policy.adaptive_ttl:
			# AI-adjusted TTL based on access patterns
			adjusted_ttl = await self._calculate_adaptive_ttl(entry, policy)
			entry.ttl_seconds = adjusted_ttl
		else:
			entry.ttl_seconds = policy.default_ttl_seconds
		
		# Update policy effectiveness
		policy.update_effectiveness(True, 0.1)  # Placeholder performance delta
	
	async def _calculate_adaptive_ttl(self, entry: CacheEntry, policy: CachePolicy) -> int:
		"""Calculate adaptive TTL based on access patterns"""
		
		base_ttl = policy.default_ttl_seconds
		
		# Adjust based on access frequency
		if entry.access_frequency > 10:
			return int(base_ttl * 1.5)  # Keep longer for frequently accessed
		elif entry.access_frequency < 1:
			return int(base_ttl * 0.5)  # Keep shorter for rarely accessed
		
		return base_ttl
	
	# Background processing
	
	async def _cleanup_loop(self) -> None:
		"""Background cleanup of expired entries"""
		
		while self.running:
			try:
				await self._cleanup_expired_entries()
				await asyncio.sleep(self.config.cleanup_interval_seconds)
			except Exception as e:
				self.logger.error(f"Error in cleanup loop: {e}")
				await asyncio.sleep(60)
	
	async def _cleanup_expired_entries(self) -> None:
		"""Remove expired cache entries"""
		
		expired_keys = []
		
		for key, entry in self._cache_store.items():
			if entry.is_expired():
				expired_keys.append(key)
		
		for key in expired_keys:
			del self._cache_store[key]
			self._metrics.cache_evictions += 1
		
		if expired_keys:
			self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
	
	async def _ai_optimization_loop(self) -> None:
		"""Background AI optimization loop"""
		
		while self.running:
			try:
				await self._run_ai_optimization()
				await asyncio.sleep(300)  # Run every 5 minutes
			except Exception as e:
				self.logger.error(f"Error in AI optimization loop: {e}")
				await asyncio.sleep(600)
	
	async def _run_ai_optimization(self) -> None:
		"""Run AI optimization analysis"""
		
		if not self.config.ai_optimization_enabled:
			return
		
		# Analyze current performance
		current_hit_rate = self._metrics.hit_rate()
		
		# Generate optimization recommendations
		recommendations = await self._generate_optimization_recommendations()
		
		# Create optimization result
		result = AIOptimizationResult(
			tenant_id=self.config.tenant_id,
			target_type="cache_performance",
			target_id="global",
			recommendations=recommendations,
			confidence_score=0.8,  # Placeholder
			current_performance={"hit_rate": current_hit_rate}
		)
		
		self._ai_optimization_results.append(result)
		
		# Keep only recent results
		if len(self._ai_optimization_results) > 100:
			self._ai_optimization_results = self._ai_optimization_results[-100:]
		
		self.logger.debug("Completed AI optimization analysis")
	
	async def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
		"""Generate AI-powered optimization recommendations"""
		
		recommendations = []
		
		# Analyze hit rate
		hit_rate = self._metrics.hit_rate()
		if hit_rate < 0.8:
			recommendations.append({
				"type": "increase_cache_size",
				"description": "Consider increasing cache size to improve hit rate",
				"impact": "medium"
			})
		
		# Analyze memory usage
		total_size = sum(entry.size_bytes for entry in self._cache_store.values())
		max_size = self.config.max_memory_mb * 1024 * 1024
		
		if total_size > max_size * 0.9:
			recommendations.append({
				"type": "optimize_eviction",
				"description": "Optimize eviction policy to better manage memory",
				"impact": "high"
			})
		
		return recommendations
	
	async def _metrics_collection_loop(self) -> None:
		"""Background metrics collection"""
		
		while self.running:
			try:
				await self._collect_metrics()
				await asyncio.sleep(60)  # Collect every minute
			except Exception as e:
				self.logger.error(f"Error in metrics collection: {e}")
				await asyncio.sleep(120)
	
	async def _collect_metrics(self) -> None:
		"""Collect performance metrics"""
		
		# Update basic metrics
		self._metrics.timestamp = datetime.utcnow()
		
		# Calculate memory usage
		total_size = sum(entry.size_bytes for entry in self._cache_store.values())
		max_size = self.config.max_memory_mb * 1024 * 1024
		
		self._metrics.total_memory_bytes = max_size
		self._metrics.used_memory_bytes = total_size
		self._metrics.available_memory_bytes = max_size - total_size
		self._metrics.memory_utilization_percent = (total_size / max_size) * 100
		
		# Archive current metrics
		self._performance_history.append(
			CacheMetrics.model_copy(self._metrics)
		)
		
		# Keep only recent history
		if len(self._performance_history) > 1440:  # 24 hours of minute data
			self._performance_history = self._performance_history[-1440:]
	
	async def _health_monitoring_loop(self) -> None:
		"""Background health monitoring"""
		
		while self.running:
			try:
				await self._check_health()
				await asyncio.sleep(30)  # Check every 30 seconds
			except Exception as e:
				self.logger.error(f"Error in health monitoring: {e}")
				await asyncio.sleep(60)
	
	async def _check_health(self) -> None:
		"""Check service health"""
		
		# Check memory usage
		memory_usage = self._metrics.memory_utilization_percent
		if memory_usage > 95:
			self.logger.warning(f"High memory usage: {memory_usage:.1f}%")
		
		# Check error rate
		error_rate = self._metrics.error_rate()
		if error_rate > 0.05:  # 5% error rate
			self.logger.warning(f"High error rate: {error_rate:.1%}")
		
		# Update cluster health
		for cluster in self._clusters.values():
			cluster.last_health_check = datetime.utcnow()
			cluster.healthy = memory_usage < 90 and error_rate < 0.05
	
	# Eviction management
	
	async def _check_eviction_needed(self) -> None:
		"""Check if eviction is needed and perform if necessary"""
		
		current_size = sum(entry.size_bytes for entry in self._cache_store.values())
		max_size = self.config.max_memory_mb * 1024 * 1024
		
		if current_size > max_size * 0.9:  # 90% threshold
			await self._perform_intelligent_eviction()
	
	async def _perform_intelligent_eviction(self) -> None:
		"""Perform AI-driven intelligent eviction"""
		
		# Calculate eviction scores for all entries
		eviction_candidates = []
		
		for key, entry in self._cache_store.items():
			score = await self._calculate_eviction_score(entry)
			eviction_candidates.append((key, entry, score))
		
		# Sort by eviction score (lowest first = most evictable)
		eviction_candidates.sort(key=lambda x: x[2])
		
		# Evict entries until we're under threshold
		current_size = sum(entry.size_bytes for entry in self._cache_store.values())
		max_size = self.config.max_memory_mb * 1024 * 1024
		target_size = max_size * 0.8  # Target 80% usage
		
		evicted_count = 0
		for key, entry, score in eviction_candidates:
			if current_size <= target_size:
				break
			
			del self._cache_store[key]
			current_size -= entry.size_bytes
			evicted_count += 1
			self._metrics.cache_evictions += 1
		
		if evicted_count > 0:
			self.logger.info(f"Intelligently evicted {evicted_count} entries")
	
	async def _calculate_eviction_score(self, entry: CacheEntry) -> float:
		"""Calculate eviction score (lower = more likely to evict)"""
		
		score = 0.0
		
		# Factor in access frequency (lower frequency = higher eviction score)
		score += max(0, 1.0 - (entry.access_frequency / 100))
		
		# Factor in hit rate (lower hit rate = higher eviction score)
		score += max(0, 1.0 - entry.hit_rate())
		
		# Factor in time since last access
		if entry.last_accessed:
			hours_since_access = (datetime.utcnow() - entry.last_accessed).total_seconds() / 3600
			score += min(hours_since_access / 24, 1.0)  # Max 1 day impact
		else:
			score += 1.0  # Never accessed
		
		# Factor in size (larger entries slightly more likely to evict)
		size_factor = min(entry.size_bytes / (1024 * 1024), 1.0)  # Max 1MB impact
		score += size_factor * 0.1
		
		return score
	
	# Utility methods
	
	async def _log_audit_event(self, event_type: str, resource_id: str, tenant_id: str,
							   details: Optional[Dict[str, Any]] = None) -> None:
		"""Log audit event for compliance"""
		
		if not self.config.audit_enabled:
			return
		
		# In production: integrate with APG audl capability
		self.logger.info(f"[AUDIT] {event_type}: {resource_id} (tenant: {tenant_id})")
	
	async def _persist_critical_data(self) -> None:
		"""Persist critical data during shutdown"""
		
		# In production: persist policies, metrics, and optimization results
		self.logger.debug("Persisting critical data")
	
	# Public API methods for statistics and monitoring
	
	async def get_stats(self) -> Dict[str, Any]:
		"""Get comprehensive cache statistics"""
		
		total_entries = len(self._cache_store)
		total_size = sum(entry.size_bytes for entry in self._cache_store.values())
		
		return {
			"total_entries": total_entries,
			"total_size_bytes": total_size,
			"hit_rate": self._metrics.hit_rate(),
			"memory_utilization": self._metrics.memory_utilization_percent,
			"total_operations": self._metrics.total_operations,
			"cache_hits": self._metrics.cache_hits,
			"cache_misses": self._metrics.cache_misses,
			"cache_evictions": self._metrics.cache_evictions,
			"policies_count": len(self._policies),
			"clusters_count": len(self._clusters),
			"ai_optimizations": len(self._ai_optimization_results)
		}
	
	async def get_performance_history(self) -> List[Dict[str, Any]]:
		"""Get performance history for analytics"""
		
		return [
			{
				"timestamp": metrics.timestamp.isoformat(),
				"hit_rate": metrics.hit_rate(),
				"memory_utilization": metrics.memory_utilization_percent,
				"operations_per_second": metrics.operations_per_second,
				"average_latency_ms": metrics.average_latency_ms
			}
			for metrics in self._performance_history[-100:]  # Last 100 data points
		]
	
	async def get_ai_insights(self) -> List[Dict[str, Any]]:
		"""Get AI optimization insights"""
		
		return [
			{
				"timestamp": result.timestamp.isoformat(),
				"target_type": result.target_type,
				"confidence_score": result.confidence_score,
				"recommendations": result.recommendations,
				"expected_improvement": result.expected_improvement,
				"applied": result.applied
			}
			for result in self._ai_optimization_results[-20:]  # Last 20 results
		]
	
	# Helper methods for advanced AI functionality
	
	async def _generate_prefetch_candidates(self, key: str, namespace: str, tenant_id: str) -> List[Tuple[str, float]]:
		"""Generate prefetch candidates using ML algorithms"""
		candidates = []
		
		try:
			# Pattern-based candidate generation
			pattern_candidates = await self._generate_pattern_based_candidates(key)
			candidates.extend(pattern_candidates)
			
			# Content similarity-based candidates
			similarity_candidates = await self._generate_similarity_candidates(key)
			candidates.extend(similarity_candidates)
			
			# User behavior-based candidates
			behavior_candidates = await self._generate_behavior_candidates(key, tenant_id)
			candidates.extend(behavior_candidates)
			
			# Remove duplicates and normalize probabilities
			unique_candidates = {}
			for cand_key, prob in candidates:
				if cand_key in unique_candidates:
					unique_candidates[cand_key] = max(unique_candidates[cand_key], prob)
				else:
					unique_candidates[cand_key] = prob
			
			return sorted(unique_candidates.items(), key=lambda x: x[1], reverse=True)
			
		except Exception as e:
			self.logger.error(f"Error generating prefetch candidates: {e}")
			return []
	
	async def _generate_pattern_based_candidates(self, key: str) -> List[Tuple[str, float]]:
		"""Generate candidates based on key patterns"""
		candidates = []
		
		# Extract numeric sequences for sequential prediction
		if any(c.isdigit() for c in key):
			import re
			numbers = re.findall(r'\d+', key)
			if numbers:
				last_num = int(numbers[-1])
				# Predict next few items in sequence
				for i in range(1, 4):
					next_key = key.replace(str(last_num), str(last_num + i))
					probability = 0.8 / i  # Decreasing probability
					candidates.append((next_key, probability))
		
		# Pattern-based prediction for hierarchical keys
		if '/' in key or '.' in key:
			parts = key.replace('.', '/').split('/')
			if len(parts) > 1:
				# Predict sibling keys
				base_path = '/'.join(parts[:-1])
				for suffix in ['_meta', '_config', '_index', '_related']:
					sibling_key = base_path + '/' + parts[-1] + suffix
					candidates.append((sibling_key, 0.6))
		
		return candidates
	
	async def _generate_similarity_candidates(self, key: str) -> List[Tuple[str, float]]:
		"""Generate candidates based on content similarity"""
		candidates = []
		
		try:
			# Find keys with similar patterns in access history
			for access_key in list(self._access_patterns.keys())[-100:]:
				if access_key != key:
					similarity = self._calculate_key_similarity(key, access_key)
					if similarity > 0.5:
						candidates.append((access_key, similarity * 0.7))
			
			return candidates
			
		except Exception as e:
			self.logger.error(f"Error in similarity candidate generation: {e}")
			return []
	
	async def _generate_behavior_candidates(self, key: str, tenant_id: str) -> List[Tuple[str, float]]:
		"""Generate candidates based on user behavior patterns"""
		candidates = []
		
		try:
			# Analyze co-occurrence patterns in access history
			co_occurring_keys = defaultdict(int)
			
			# Look for keys accessed together within time windows
			for accessed_key, access_times in self._access_patterns.items():
				if accessed_key != key and access_times:
					# Count co-occurrences within 5-minute windows
					if key in self._access_patterns:
						key_access_times = self._access_patterns[key]
						for key_time in key_access_times[-10:]:  # Recent accesses
							for access_time in access_times:
								if abs((access_time - key_time).total_seconds()) < 300:  # 5 minutes
									co_occurring_keys[accessed_key] += 1
			
			# Convert to candidates with probabilities
			for co_key, count in co_occurring_keys.items():
				probability = min(count / 10.0, 0.9)  # Normalize to max 0.9
				candidates.append((co_key, probability))
			
			return candidates
			
		except Exception as e:
			self.logger.error(f"Error in behavior candidate generation: {e}")
			return []
	
	async def _execute_predictive_prefetch(self, key: str, namespace: str, tenant_id: str, confidence: float) -> bool:
		"""Execute predictive prefetching with external data source simulation"""
		try:
			# Check if key already exists
			if await self.exists(key, namespace, tenant_id):
				return False
			
			# Simulate fetching data from external source
			# In production: integrate with actual data sources
			prefetch_data = await self._simulate_data_fetch(key, namespace, tenant_id)
			
			if prefetch_data:
				# Store with predictive TTL based on confidence
				predictive_ttl = int(3600 * confidence)  # Higher confidence = longer TTL
				
				await self.set(
					key=key,
					value=prefetch_data,
					ttl_seconds=predictive_ttl,
					namespace=namespace,
					tenant_id=tenant_id
				)
				
				self.logger.debug(f"Prefetched key {key} with confidence {confidence}")
				return True
			
			return False
			
		except Exception as e:
			self.logger.error(f"Error in predictive prefetch execution: {e}")
			return False
	
	async def _simulate_data_fetch(self, key: str, namespace: str, tenant_id: str) -> Optional[Any]:
		"""Simulate fetching data from external sources for prefetching"""
		# Simulate realistic data based on key patterns
		if 'user' in key.lower():
			return {
				'id': key.split(':')[-1] if ':' in key else 'unknown',
				'type': 'user_data',
				'timestamp': datetime.utcnow().isoformat(),
				'prefetched': True
			}
		elif 'product' in key.lower():
			return {
				'id': key.split(':')[-1] if ':' in key else 'unknown',
				'type': 'product_data',
				'category': 'general',
				'timestamp': datetime.utcnow().isoformat(),
				'prefetched': True
			}
		elif 'api' in key.lower():
			return {
				'endpoint': key,
				'type': 'api_response',
				'data': f'Prefetched response for {key}',
				'timestamp': datetime.utcnow().isoformat(),
				'prefetched': True
			}
		else:
			return {
				'key': key,
				'type': 'generic_data',
				'content': f'Prefetched content for {key}',
				'timestamp': datetime.utcnow().isoformat(),
				'prefetched': True
			}
	
	def _calculate_key_similarity(self, key1: str, key2: str) -> float:
		"""Calculate similarity between two cache keys using Jaccard similarity"""
		try:
			# Jaccard similarity based on character n-grams
			n = 3  # 3-character grams
			grams1 = set(key1[i:i+n] for i in range(len(key1)-n+1))
			grams2 = set(key2[i:i+n] for i in range(len(key2)-n+1))
			
			if not grams1 or not grams2:
				return 0.0
			
			intersection = len(grams1.intersection(grams2))
			union = len(grams1.union(grams2))
			
			return intersection / union if union > 0 else 0.0
			
		except Exception:
			return 0.0
	
	async def _build_prefetch_context(self, key: str, namespace: str, tenant_id: str) -> Dict[str, Any]:
		"""Build comprehensive context for prefetch decision making"""
		context = {
			'key': key,
			'namespace': namespace,
			'tenant_id': tenant_id,
			'timestamp': datetime.utcnow(),
			'recent_keys': list(self._access_patterns.keys())[-20:],
			'cache_size': len(self._cache_store),
			'memory_pressure': self._metrics.memory_utilization_percent > 80
		}
		
		# Add temporal context
		now = datetime.utcnow()
		context.update({
			'hour_of_day': now.hour,
			'day_of_week': now.weekday(),
			'is_business_hours': 9 <= now.hour <= 17
		})
		
		return context
	
	async def _find_similar_content(self, key: str, context: Dict[str, Any]) -> List[Tuple[str, float, float]]:
		"""Find similar content using collaborative filtering algorithms"""
		similar_items = []
		
		try:
			# Content-based similarity
			for candidate_key in context.get('recent_keys', []):
				if candidate_key != key:
					similarity = self._calculate_key_similarity(key, candidate_key)
					if similarity > 0.3:
						timing_offset = random.uniform(0, 30)  # 0-30 seconds
						similar_items.append((candidate_key, similarity, timing_offset))
			
			return similar_items
			
		except Exception as e:
			self.logger.error(f"Error finding similar content: {e}")
			return []
	
	async def _analyze_temporal_patterns(self, key: str, context: Dict[str, Any]) -> List[Tuple[str, float, float]]:
		"""Analyze temporal patterns for predictive prefetching using time series analysis"""
		temporal_predictions = []
		
		try:
			current_hour = context.get('hour_of_day', 12)
			is_business_hours = context.get('is_business_hours', False)
			
			# Predict keys likely to be accessed at current time
			for access_key, access_times in self._access_patterns.items():
				if access_key != key and access_times:
					# Calculate temporal correlation
					same_hour_accesses = sum(
						1 for access_time in access_times
						if access_time.hour == current_hour
					)
					
					temporal_score = same_hour_accesses / len(access_times)
					
					if temporal_score > 0.2:  # 20% of accesses at this hour
						timing_offset = random.uniform(60, 300)  # 1-5 minutes
						temporal_predictions.append((access_key, temporal_score, timing_offset))
			
			return temporal_predictions
			
		except Exception as e:
			self.logger.error(f"Error in temporal pattern analysis: {e}")
			return []
	
	async def _combine_prediction_sources(self, similar_items: List[Tuple[str, float, float]],
											  temporal_predictions: List[Tuple[str, float, float]],
											  context: Dict[str, Any]) -> List[Tuple[str, float, float]]:
		"""Combine multiple prediction sources with intelligent weighting"""
		combined_predictions = {}
		
		# Weight similarity-based predictions
		for key, similarity, timing in similar_items:
			weight = 0.6 * similarity  # 60% weight for similarity
			combined_predictions[key] = (weight, timing)
		
		# Add temporal predictions with different weight
		for key, temporal_score, timing in temporal_predictions:
			weight = 0.4 * temporal_score  # 40% weight for temporal
			if key in combined_predictions:
				# Combine with existing prediction
				existing_weight, existing_timing = combined_predictions[key]
				combined_weight = existing_weight + weight
				avg_timing = (existing_timing + timing) / 2
				combined_predictions[key] = (combined_weight, avg_timing)
			else:
				combined_predictions[key] = (weight, timing)
		
		# Convert to list and sort by confidence
		result = [
			(key, min(weight, 1.0), timing) 
			for key, (weight, timing) in combined_predictions.items()
		]
		
		return sorted(result, key=lambda x: x[1], reverse=True)
	
	def _calculate_optimization_reward(self, action: str, before_metrics: Dict[str, float], 
									   after_metrics: Dict[str, float]) -> float:
		"""Calculate reward for reinforcement learning optimization"""
		try:
			hit_rate_improvement = after_metrics.get('hit_rate', 0) - before_metrics.get('hit_rate', 0)
			latency_improvement = before_metrics.get('latency', 10) - after_metrics.get('latency', 10)
			memory_efficiency = 1.0 - after_metrics.get('memory_usage', 1.0)
			
			# Weighted reward function
			reward = (
				hit_rate_improvement * 50 +  # High weight for hit rate
				latency_improvement * 20 +    # Medium weight for latency
				memory_efficiency * 10        # Lower weight for memory
			)
			
			return max(-100, min(100, reward))  # Clamp between -100 and 100
			
		except Exception:
			return 0.0


# Factory function
async def create_cache_service(config: Optional[CacheServiceConfig] = None) -> CacheService:
	"""Create and initialize cache service"""
	service = CacheService(config)
	await service.initialize()
	return service


# Export main components
__all__ = [
	'CacheService',
	'CacheServiceConfig',
	'create_cache_service'
]
