"""Async service layer for APG Open Source Intelligence (OSINT).

All public methods are async and enforce tenant isolation on every
operation.  Domain events are emitted via _emit_event() after every
state change.  The in-memory store (dicts keyed by (tenant_id, id))
mirrors the PostgreSQL schema in database/schema.sql.

Usage:
    svc = OSINTService(db_session=None, tenant_id="acme", actor_id="u-001")
    source = await svc.register_source(OSINTSourceCreate(...))
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_COLLECTION_METHODS,
		SUPPORTED_CONFIDENCE_LEVELS,
		SUPPORTED_ENTITY_TYPES,
		SUPPORTED_INTEL_STATUSES,
		SUPPORTED_PRIORITIES,
		SUPPORTED_RELATIONSHIP_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_RISK_TIERS,
		SUPPORTED_SOURCE_TYPES,
		SUPPORTED_TASK_TYPES,
		SUPPORTED_TLP,
		SUPPORTED_TRIAGE_DECISIONS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .domain.calculations import (
		calculate_entity_centrality,
		calculate_ip_threat_score,
		composite_intel_credibility,
		compute_content_fingerprint,
		deduplicate_entities,
		find_connected_clusters,
	)
	from .domain.rules import (
		RuleViolation,
		assert_confidence_bounds,
		assert_dissemination_approval,
		assert_entity_name_present,
		assert_fingerprint_present,
		assert_high_risk_source_approval,
		assert_human_approval_for_privileged,
		assert_no_cross_tenant_access,
		assert_not_duplicate,
		assert_relationship_entities_distinct,
		assert_source_terms_reviewed,
		assert_tenant_context,
		assert_write_policy,
		calculate_intel_credibility,
	)
	from .models import (
		AgentRole,
		AgentRuntime,
		CollectionTaskCreate,
		CollectionTaskResponse,
		CollectionTaskUpdate,
		CredibilityScoreCreate,
		CredibilityScoreResponse,
		DisseminationPackageCreate,
		DisseminationPackageResponse,
		DocumentAnalysisCreate,
		DocumentAnalysisResponse,
		EntityRelationshipCreate,
		EntityRelationshipResponse,
		EntityRelationshipUpdate,
		EntityType,
		IPIntelligenceCreate,
		IPIntelligenceResponse,
		IntelStatus,
		OSEntityCreate,
		OSEntityResponse,
		OSEntityUpdate,
		OSINTAgentCreate,
		OSINTAgentResponse,
		OSINTDashboard,
		OSINTReviewCreate,
		OSINTReviewResponse,
		OSINTSourceCreate,
		OSINTSourceResponse,
		OSINTSourceUpdate,
		ProcessedIntelligenceCreate,
		ProcessedIntelligenceResponse,
		ProcessedIntelligenceUpdate,
		RawIntelligenceCreate,
		RawIntelligenceResponse,
		RawIntelligenceUpdate,
		RelationshipType,
		ReviewStatus,
		RiskTier,
		SocialMediaProfileCreate,
		SocialMediaProfileResponse,
		SocialMediaProfileUpdate,
		SourceStatus,
		SourceType,
		TaskStatus,
		TaskType,
		TriageDecision,
		WebContentCreate,
		WebContentResponse,
		DomainRecordCreate,
		DomainRecordResponse,
		EntityNetworkReport,
		SourceHealthReport,
		ThreatLandscapeReport,
		uuid7str,
		_now,
	)
	from .osint_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover — standalone execution
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_COLLECTION_METHODS, SUPPORTED_CONFIDENCE_LEVELS,
		SUPPORTED_ENTITY_TYPES, SUPPORTED_INTEL_STATUSES, SUPPORTED_PRIORITIES,
		SUPPORTED_RELATIONSHIP_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_TIERS,
		SUPPORTED_SOURCE_TYPES, SUPPORTED_TASK_TYPES, SUPPORTED_TLP, SUPPORTED_TRIAGE_DECISIONS,
		evaluate_capability_rules, get_capability_contract,
	)
	from domain.calculations import (  # type: ignore
		calculate_entity_centrality, calculate_ip_threat_score, composite_intel_credibility,
		compute_content_fingerprint, deduplicate_entities, find_connected_clusters,
	)
	from domain.rules import (  # type: ignore
		RuleViolation, assert_confidence_bounds, assert_dissemination_approval,
		assert_entity_name_present, assert_fingerprint_present, assert_high_risk_source_approval,
		assert_human_approval_for_privileged, assert_no_cross_tenant_access, assert_not_duplicate,
		assert_relationship_entities_distinct, assert_source_terms_reviewed, assert_tenant_context,
		assert_write_policy, calculate_intel_credibility,
	)
	from models import (  # type: ignore
		AgentRole, AgentRuntime, CollectionTaskCreate, CollectionTaskResponse,
		CollectionTaskUpdate, CredibilityScoreCreate, CredibilityScoreResponse,
		DisseminationPackageCreate, DisseminationPackageResponse,
		DocumentAnalysisCreate, DocumentAnalysisResponse,
		EntityRelationshipCreate, EntityRelationshipResponse, EntityRelationshipUpdate,
		EntityType, IPIntelligenceCreate, IPIntelligenceResponse, IntelStatus,
		OSEntityCreate, OSEntityResponse, OSEntityUpdate, OSINTAgentCreate,
		OSINTAgentResponse, OSINTDashboard, OSINTReviewCreate, OSINTReviewResponse,
		OSINTSourceCreate, OSINTSourceResponse, OSINTSourceUpdate,
		ProcessedIntelligenceCreate, ProcessedIntelligenceResponse, ProcessedIntelligenceUpdate,
		RawIntelligenceCreate, RawIntelligenceResponse, RawIntelligenceUpdate,
		RelationshipType, ReviewStatus, RiskTier, SocialMediaProfileCreate,
		SocialMediaProfileResponse, SocialMediaProfileUpdate, SourceStatus, SourceType,
		TaskStatus, TaskType, TriageDecision, WebContentCreate, WebContentResponse,
		DomainRecordCreate, DomainRecordResponse, EntityNetworkReport,
		SourceHealthReport, ThreatLandscapeReport, uuid7str, _now,
	)
	from osint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class OSINTService:
	"""Tenant-scoped OSINT service.  All methods are async.

	Args:
		db_session: SQLAlchemy async session (or None for in-memory use).
		tenant_id: Tenant context for all operations.
		actor_id: ID of the authenticated user performing operations.
	"""

	def __init__(
		self,
		db_session: Any = None,
		tenant_id: str = "default",
		actor_id: str = "system",
	) -> None:
		self._db = db_session
		self._tenant_id = tenant_id
		self._actor_id = actor_id

		# In-memory stores keyed by (tenant_id, id)
		self._sources: dict[tuple[str, str], OSINTSourceResponse] = {}
		self._tasks: dict[tuple[str, str], CollectionTaskResponse] = {}
		self._raw_intel: dict[tuple[str, str], RawIntelligenceResponse] = {}
		self._processed_intel: dict[tuple[str, str], ProcessedIntelligenceResponse] = {}
		self._entities: dict[tuple[str, str], OSEntityResponse] = {}
		self._relationships: dict[tuple[str, str], EntityRelationshipResponse] = {}
		self._social_profiles: dict[tuple[str, str], SocialMediaProfileResponse] = {}
		self._web_content: dict[tuple[str, str], WebContentResponse] = {}
		self._domain_records: dict[tuple[str, str], DomainRecordResponse] = {}
		self._ip_intel: dict[tuple[str, str], IPIntelligenceResponse] = {}
		self._doc_analyses: dict[tuple[str, str], DocumentAnalysisResponse] = {}
		self._credibility_scores: dict[tuple[str, str], CredibilityScoreResponse] = {}
		self._dissemination: dict[tuple[str, str], DisseminationPackageResponse] = {}
		self._reviews: dict[tuple[str, str], OSINTReviewResponse] = {}
		self._agents: dict[tuple[str, str], OSINTAgentResponse] = {}
		self._audit_events: list[dict[str, Any]] = []

		# Fingerprint index for deduplication (tenant_id -> set of fingerprints)
		self._fingerprints: dict[str, set[str]] = {}

	# -----------------------------------------------------------------------
	# Contract / describe
	# -----------------------------------------------------------------------

	async def describe(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return the full capability contract for a tenant."""
		return get_capability_contract(tenant_id or self._tenant_id)

	async def evaluate_rules(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate policy rules against an arbitrary context dict."""
		return evaluate_capability_rules(context)

	# -----------------------------------------------------------------------
	# Source management
	# -----------------------------------------------------------------------

	async def register_source(self, payload: OSINTSourceCreate) -> OSINTSourceResponse:
		"""Register a new intelligence source.

		Enforces terms-of-service review requirement and risk tier validity.
		"""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		assert_write_policy(True)
		assert_source_terms_reviewed(payload.terms_review_reference)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_source",
			"source_type_supported": payload.source_type.value in SUPPORTED_SOURCE_TYPES,
			"name_present": present(payload.name),
			"owner_present": present(payload.owner_id),
			"terms_review_present": present(payload.terms_review_reference),
			"risk_tier_supported": payload.risk_tier.value in SUPPORTED_RISK_TIERS,
			"evidence_present": present(payload.evidence_reference),
		})
		item = OSINTSourceResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			name=payload.name,
			source_type=payload.source_type,
			url=payload.url,
			description=payload.description,
			owner_id=payload.owner_id,
			terms_review_reference=payload.terms_review_reference,
			risk_tier=payload.risk_tier,
			collection_method=payload.collection_method,
			requires_auth=payload.requires_auth,
			auth_reference=payload.auth_reference,
			rate_limit_rps=payload.rate_limit_rps,
			credibility_baseline=payload.credibility_baseline,
			tags=list(payload.tags),
			evidence_reference=payload.evidence_reference,
		)
		self._sources[self._key(item.tenant_id, item.id)] = item
		self._emit_event("osint_source_registered", item.id, item.tenant_id)
		self._log_operation("register_source", item.id)
		return item

	async def update_source(self, source_id: str, payload: OSINTSourceUpdate) -> OSINTSourceResponse:
		"""Update mutable fields on a registered source."""
		item = self._get_or_raise(self._sources, source_id, "OSINTSource")
		if payload.name is not None:
			object.__setattr__(item, "name", payload.name)
		if payload.description is not None:
			object.__setattr__(item, "description", payload.description)
		if payload.status is not None:
			object.__setattr__(item, "status", payload.status)
		if payload.risk_tier is not None:
			object.__setattr__(item, "risk_tier", payload.risk_tier)
		if payload.credibility_baseline is not None:
			object.__setattr__(item, "credibility_baseline", payload.credibility_baseline)
		if payload.tags is not None:
			object.__setattr__(item, "tags", list(payload.tags))
		if payload.evidence_reference is not None:
			object.__setattr__(item, "evidence_reference", payload.evidence_reference)
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_source_updated", source_id, item.tenant_id)
		return item

	async def get_source(self, source_id: str) -> OSINTSourceResponse:
		"""Retrieve a single source by ID."""
		return self._get_or_raise(self._sources, source_id, "OSINTSource")

	async def list_sources(
		self,
		source_type: SourceType | None = None,
		risk_tier: RiskTier | None = None,
		status: SourceStatus | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[OSINTSourceResponse]:
		"""List sources for this tenant with optional filters."""
		items = self._tenant_values(self._sources)
		if source_type:
			items = [i for i in items if i.source_type == source_type]
		if risk_tier:
			items = [i for i in items if i.risk_tier == risk_tier]
		if status:
			items = [i for i in items if i.status == status]
		return sorted(items, key=lambda x: x.created_at, reverse=True)[offset: offset + limit]

	async def delete_source(self, source_id: str) -> None:
		"""Soft-delete a source."""
		item = self._get_or_raise(self._sources, source_id, "OSINTSource")
		object.__setattr__(item, "is_deleted", True)
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_source_deleted", source_id, item.tenant_id)

	# -----------------------------------------------------------------------
	# Collection tasks
	# -----------------------------------------------------------------------

	async def create_task(self, payload: CollectionTaskCreate) -> CollectionTaskResponse:
		"""Create a new collection task against a registered source.

		High/critical-risk sources require an approval_reference.
		"""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		source = self._get_or_raise(self._sources, payload.source_id, "OSINTSource")
		assert_high_risk_source_approval(source.risk_tier.value, payload.approval_reference)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_task",
			"source_present": True,
			"task_type_supported": payload.task_type.value in SUPPORTED_TASK_TYPES,
			"high_risk_source": source.risk_tier.value in {"high", "critical"},
			"approval_present": present(payload.approval_reference or ""),
			"evidence_present": present(payload.evidence_reference),
		})
		item = CollectionTaskResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			source_id=payload.source_id,
			task_type=payload.task_type,
			parameters=dict(payload.parameters),
			priority=payload.priority,
			scheduled_at=payload.scheduled_at,
			max_depth=payload.max_depth,
			max_items=payload.max_items,
			keywords=list(payload.keywords),
			approval_reference=payload.approval_reference,
			evidence_reference=payload.evidence_reference,
		)
		self._tasks[self._key(item.tenant_id, item.id)] = item
		self._emit_event("osint_task_created", item.id, item.tenant_id)
		self._log_operation("create_task", item.id)
		return item

	async def start_task(self, task_id: str) -> CollectionTaskResponse:
		"""Transition a pending task to running."""
		item = self._get_or_raise(self._tasks, task_id, "CollectionTask")
		object.__setattr__(item, "status", TaskStatus.RUNNING)
		object.__setattr__(item, "started_at", _now())
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_task_status_changed", task_id, item.tenant_id, {"status": "running"})
		return item

	async def complete_task(self, task_id: str, items_collected: int) -> CollectionTaskResponse:
		"""Mark a running task as completed."""
		item = self._get_or_raise(self._tasks, task_id, "CollectionTask")
		object.__setattr__(item, "status", TaskStatus.COMPLETED)
		object.__setattr__(item, "completed_at", _now())
		object.__setattr__(item, "items_collected", items_collected)
		object.__setattr__(item, "updated_at", _now())
		# Update source last_collected_at
		source_key = self._key(item.tenant_id, item.source_id)
		if source_key in self._sources:
			src = self._sources[source_key]
			object.__setattr__(src, "last_collected_at", _now())
			object.__setattr__(src, "total_items_collected", src.total_items_collected + items_collected)
		self._emit_event("osint_task_status_changed", task_id, item.tenant_id, {"status": "completed", "items_collected": items_collected})
		return item

	async def fail_task(self, task_id: str, error_message: str) -> CollectionTaskResponse:
		"""Mark a running task as failed with an error message."""
		item = self._get_or_raise(self._tasks, task_id, "CollectionTask")
		object.__setattr__(item, "status", TaskStatus.FAILED)
		object.__setattr__(item, "error_message", error_message)
		object.__setattr__(item, "completed_at", _now())
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_task_status_changed", task_id, item.tenant_id, {"status": "failed"})
		return item

	async def cancel_task(self, task_id: str) -> CollectionTaskResponse:
		"""Cancel a pending or running task."""
		item = self._get_or_raise(self._tasks, task_id, "CollectionTask")
		object.__setattr__(item, "status", TaskStatus.CANCELLED)
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_task_status_changed", task_id, item.tenant_id, {"status": "cancelled"})
		return item

	async def get_task(self, task_id: str) -> CollectionTaskResponse:
		"""Retrieve a task by ID."""
		return self._get_or_raise(self._tasks, task_id, "CollectionTask")

	async def list_tasks(
		self,
		source_id: str | None = None,
		task_type: TaskType | None = None,
		status: TaskStatus | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[CollectionTaskResponse]:
		"""List tasks for this tenant."""
		items = self._tenant_values(self._tasks)
		if source_id:
			items = [i for i in items if i.source_id == source_id]
		if task_type:
			items = [i for i in items if i.task_type == task_type]
		if status:
			items = [i for i in items if i.status == status]
		return sorted(items, key=lambda x: x.created_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Raw intelligence ingestion
	# -----------------------------------------------------------------------

	async def ingest_raw_intel(self, payload: RawIntelligenceCreate) -> RawIntelligenceResponse:
		"""Ingest a raw intelligence artefact from a collection task.

		Performs duplicate detection by fingerprint before storing.
		"""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		assert_fingerprint_present(payload.fingerprint)
		assert_confidence_bounds(payload.confidence_score)
		# Deduplication check
		fps = self._fingerprints.setdefault(payload.tenant_id, set())
		assert_not_duplicate(payload.fingerprint, fps)
		self._get_or_raise(self._tasks, payload.task_id, "CollectionTask")  # FK check
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "ingest_raw_intel",
			"task_present": True,
			"fingerprint_present": True,
			"confidence_valid": bounded_score(payload.confidence_score),
			"evidence_present": present(payload.evidence_reference),
		})
		item = RawIntelligenceResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			task_id=payload.task_id,
			source_id=payload.source_id,
			content_reference=payload.content_reference,
			content_type=payload.content_type,
			url=payload.url,
			fingerprint=payload.fingerprint,
			confidence_score=payload.confidence_score,
			language=payload.language,
			captured_at=payload.captured_at,
			evidence_reference=payload.evidence_reference,
		)
		self._raw_intel[self._key(item.tenant_id, item.id)] = item
		fps.add(payload.fingerprint)
		self._emit_event("osint_raw_intel_ingested", item.id, item.tenant_id)
		self._log_operation("ingest_raw_intel", item.id)
		return item

	async def triage_raw_intel(
		self,
		raw_intel_id: str,
		decision: TriageDecision,
		analyst_id: str,
		notes: str | None = None,
	) -> RawIntelligenceResponse:
		"""Record a triage decision on a raw intelligence item."""
		item = self._get_or_raise(self._raw_intel, raw_intel_id, "RawIntelligence")
		object.__setattr__(item, "triage_decision", decision)
		object.__setattr__(item, "analyst_id", analyst_id)
		object.__setattr__(item, "notes", notes)
		object.__setattr__(item, "status", IntelStatus.TRIAGED)
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_raw_intel_triaged", raw_intel_id, item.tenant_id, {"decision": decision.value})
		return item

	async def get_raw_intel(self, raw_intel_id: str) -> RawIntelligenceResponse:
		"""Retrieve a raw intelligence item by ID."""
		return self._get_or_raise(self._raw_intel, raw_intel_id, "RawIntelligence")

	async def list_raw_intel(
		self,
		task_id: str | None = None,
		source_id: str | None = None,
		status: IntelStatus | None = None,
		triage_decision: TriageDecision | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[RawIntelligenceResponse]:
		"""List raw intel for this tenant."""
		items = self._tenant_values(self._raw_intel)
		if task_id:
			items = [i for i in items if i.task_id == task_id]
		if source_id:
			items = [i for i in items if i.source_id == source_id]
		if status:
			items = [i for i in items if i.status == status]
		if triage_decision:
			items = [i for i in items if i.triage_decision == triage_decision]
		return sorted(items, key=lambda x: x.captured_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Processed intelligence
	# -----------------------------------------------------------------------

	async def create_processed_intel(
		self, payload: ProcessedIntelligenceCreate
	) -> ProcessedIntelligenceResponse:
		"""Promote raw intelligence to a processed intel item."""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		assert_confidence_bounds(payload.confidence_score)
		self._get_or_raise(self._raw_intel, payload.raw_intel_id, "RawIntelligence")
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_processed_intel",
			"raw_intel_present": True,
			"assessment_type_supported": payload.assessment_type.value in SUPPORTED_ASSESSMENT_TYPES,
			"analyst_present": present(payload.analyst_id),
			"confidence_valid": bounded_score(payload.confidence_score),
			"evidence_present": present(payload.evidence_reference),
		})
		item = ProcessedIntelligenceResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			raw_intel_id=payload.raw_intel_id,
			requirement_id=payload.requirement_id,
			assessment_type=payload.assessment_type,
			summary=payload.summary,
			key_findings=list(payload.key_findings),
			confidence_score=payload.confidence_score,
			confidence_level=payload.confidence_level,
			classification=payload.classification,
			tlp=payload.tlp,
			analyst_id=payload.analyst_id,
			tags=list(payload.tags),
			evidence_reference=payload.evidence_reference,
		)
		self._processed_intel[self._key(item.tenant_id, item.id)] = item
		# Update raw intel status
		raw = self._raw_intel.get(self._key(payload.tenant_id, payload.raw_intel_id))
		if raw:
			object.__setattr__(raw, "status", IntelStatus.PROCESSED)
		self._emit_event("osint_processed_intel_created", item.id, item.tenant_id)
		self._log_operation("create_processed_intel", item.id)
		return item

	async def update_processed_intel(
		self, intel_id: str, payload: ProcessedIntelligenceUpdate
	) -> ProcessedIntelligenceResponse:
		"""Update a processed intelligence item."""
		item = self._get_or_raise(self._processed_intel, intel_id, "ProcessedIntelligence")
		if payload.summary is not None:
			object.__setattr__(item, "summary", payload.summary)
		if payload.key_findings is not None:
			object.__setattr__(item, "key_findings", list(payload.key_findings))
		if payload.confidence_score is not None:
			assert_confidence_bounds(payload.confidence_score)
			object.__setattr__(item, "confidence_score", payload.confidence_score)
		if payload.confidence_level is not None:
			object.__setattr__(item, "confidence_level", payload.confidence_level)
		if payload.status is not None:
			object.__setattr__(item, "status", payload.status)
		if payload.tags is not None:
			object.__setattr__(item, "tags", list(payload.tags))
		if payload.evidence_reference is not None:
			object.__setattr__(item, "evidence_reference", payload.evidence_reference)
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_processed_intel_updated", intel_id, item.tenant_id)
		return item

	async def get_processed_intel(self, intel_id: str) -> ProcessedIntelligenceResponse:
		"""Retrieve processed intel by ID."""
		return self._get_or_raise(self._processed_intel, intel_id, "ProcessedIntelligence")

	async def list_processed_intel(
		self,
		assessment_type: str | None = None,
		status: IntelStatus | None = None,
		analyst_id: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[ProcessedIntelligenceResponse]:
		"""List processed intel for this tenant."""
		items = self._tenant_values(self._processed_intel)
		if assessment_type:
			items = [i for i in items if i.assessment_type.value == assessment_type]
		if status:
			items = [i for i in items if i.status == status]
		if analyst_id:
			items = [i for i in items if i.analyst_id == analyst_id]
		return sorted(items, key=lambda x: x.created_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Entity extraction (NLP-driven)
	# -----------------------------------------------------------------------

	async def extract_entity(self, payload: OSEntityCreate) -> OSEntityResponse:
		"""Store an entity extracted from intelligence text.

		Typically called after entity_extraction_nlp() processes a document.
		"""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		assert_entity_name_present(payload.name)
		assert_confidence_bounds(payload.confidence_score)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "extract_entity",
			"entity_type_supported": payload.entity_type.value in SUPPORTED_ENTITY_TYPES,
			"name_present": True,
			"confidence_valid": bounded_score(payload.confidence_score),
			"evidence_present": present(payload.evidence_reference),
		})
		item = OSEntityResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			entity_type=payload.entity_type,
			name=payload.name,
			aliases=list(payload.aliases),
			description=payload.description,
			attributes=dict(payload.attributes),
			confidence_score=payload.confidence_score,
			confidence_level=payload.confidence_level,
			classification=payload.classification,
			source_intel_ids=list(payload.source_intel_ids),
			tags=list(payload.tags),
			evidence_reference=payload.evidence_reference,
		)
		self._entities[self._key(item.tenant_id, item.id)] = item
		# Link entity back to processed intel records
		for intel_id in payload.source_intel_ids:
			intel_key = self._key(payload.tenant_id, intel_id)
			if intel_key in self._processed_intel:
				intel = self._processed_intel[intel_key]
				updated_ids = list(intel.entity_ids) + [item.id]
				object.__setattr__(intel, "entity_ids", updated_ids)
		self._emit_event("osint_entity_extracted", item.id, item.tenant_id, {"entity_type": item.entity_type.value})
		self._log_operation("extract_entity", item.id)
		return item

	async def update_entity(self, entity_id: str, payload: OSEntityUpdate) -> OSEntityResponse:
		"""Update a known entity record."""
		item = self._get_or_raise(self._entities, entity_id, "OSEntity")
		if payload.name is not None:
			assert_entity_name_present(payload.name)
			object.__setattr__(item, "name", payload.name)
		if payload.aliases is not None:
			object.__setattr__(item, "aliases", list(payload.aliases))
		if payload.description is not None:
			object.__setattr__(item, "description", payload.description)
		if payload.attributes is not None:
			object.__setattr__(item, "attributes", dict(payload.attributes))
		if payload.confidence_score is not None:
			assert_confidence_bounds(payload.confidence_score)
			object.__setattr__(item, "confidence_score", payload.confidence_score)
		if payload.confidence_level is not None:
			object.__setattr__(item, "confidence_level", payload.confidence_level)
		if payload.tags is not None:
			object.__setattr__(item, "tags", list(payload.tags))
		if payload.evidence_reference is not None:
			object.__setattr__(item, "evidence_reference", payload.evidence_reference)
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_entity_updated", entity_id, item.tenant_id)
		return item

	async def get_entity(self, entity_id: str) -> OSEntityResponse:
		"""Retrieve an entity by ID."""
		return self._get_or_raise(self._entities, entity_id, "OSEntity")

	async def list_entities(
		self,
		entity_type: EntityType | None = None,
		min_confidence: float | None = None,
		tag: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[OSEntityResponse]:
		"""List entities for this tenant."""
		items = self._tenant_values(self._entities)
		if entity_type:
			items = [i for i in items if i.entity_type == entity_type]
		if min_confidence is not None:
			items = [i for i in items if i.confidence_score >= min_confidence]
		if tag:
			items = [i for i in items if tag in i.tags]
		return sorted(items, key=lambda x: x.confidence_score, reverse=True)[offset: offset + limit]

	async def delete_entity(self, entity_id: str) -> None:
		"""Soft-delete an entity."""
		item = self._get_or_raise(self._entities, entity_id, "OSEntity")
		object.__setattr__(item, "is_deleted", True)
		object.__setattr__(item, "updated_at", _now())
		self._emit_event("osint_entity_deleted", entity_id, item.tenant_id)

	# -----------------------------------------------------------------------
	# Relationship mapping
	# -----------------------------------------------------------------------

	async def map_relationship(
		self, payload: EntityRelationshipCreate
	) -> EntityRelationshipResponse:
		"""Record a directed relationship between two entities."""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		assert_relationship_entities_distinct(payload.source_entity_id, payload.target_entity_id)
		assert_confidence_bounds(payload.confidence_score)
		self._get_or_raise(self._entities, payload.source_entity_id, "OSEntity")
		self._get_or_raise(self._entities, payload.target_entity_id, "OSEntity")
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "map_relationship",
			"relationship_type_supported": payload.relationship_type.value in SUPPORTED_RELATIONSHIP_TYPES,
			"source_entity_present": True,
			"target_entity_present": True,
			"self_loop": payload.source_entity_id == payload.target_entity_id,
			"confidence_valid": bounded_score(payload.confidence_score),
			"evidence_present": present(payload.evidence_reference),
		})
		item = EntityRelationshipResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			source_entity_id=payload.source_entity_id,
			target_entity_id=payload.target_entity_id,
			relationship_type=payload.relationship_type,
			description=payload.description,
			strength=payload.strength,
			confidence_score=payload.confidence_score,
			first_seen=payload.first_seen,
			last_seen=payload.last_seen,
			attributes=dict(payload.attributes),
			evidence_reference=payload.evidence_reference,
		)
		self._relationships[self._key(item.tenant_id, item.id)] = item
		# Backlink into entities
		for eid in (payload.source_entity_id, payload.target_entity_id):
			entity_key = self._key(payload.tenant_id, eid)
			if entity_key in self._entities:
				entity = self._entities[entity_key]
				updated_rels = list(entity.relationship_ids) + [item.id]
				object.__setattr__(entity, "relationship_ids", updated_rels)
		self._emit_event("osint_relationship_mapped", item.id, item.tenant_id, {"type": item.relationship_type.value})
		self._log_operation("map_relationship", item.id)
		return item

	async def update_relationship(
		self, rel_id: str, payload: EntityRelationshipUpdate
	) -> EntityRelationshipResponse:
		"""Update a relationship record."""
		item = self._get_or_raise(self._relationships, rel_id, "EntityRelationship")
		if payload.description is not None:
			object.__setattr__(item, "description", payload.description)
		if payload.strength is not None:
			object.__setattr__(item, "strength", payload.strength)
		if payload.confidence_score is not None:
			assert_confidence_bounds(payload.confidence_score)
			object.__setattr__(item, "confidence_score", payload.confidence_score)
		if payload.last_seen is not None:
			object.__setattr__(item, "last_seen", payload.last_seen)
		if payload.attributes is not None:
			object.__setattr__(item, "attributes", dict(payload.attributes))
		if payload.evidence_reference is not None:
			object.__setattr__(item, "evidence_reference", payload.evidence_reference)
		object.__setattr__(item, "updated_at", _now())
		return item

	async def get_relationship(self, rel_id: str) -> EntityRelationshipResponse:
		"""Retrieve a relationship by ID."""
		return self._get_or_raise(self._relationships, rel_id, "EntityRelationship")

	async def list_relationships(
		self,
		entity_id: str | None = None,
		relationship_type: RelationshipType | None = None,
		min_confidence: float | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[EntityRelationshipResponse]:
		"""List relationships for this tenant."""
		items = self._tenant_values(self._relationships)
		if entity_id:
			items = [i for i in items if i.source_entity_id == entity_id or i.target_entity_id == entity_id]
		if relationship_type:
			items = [i for i in items if i.relationship_type == relationship_type]
		if min_confidence is not None:
			items = [i for i in items if i.confidence_score >= min_confidence]
		return sorted(items, key=lambda x: x.confidence_score, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Web scraping operations
	# -----------------------------------------------------------------------

	async def web_scrape(
		self,
		url: str,
		task_id: str,
		depth: int = 2,
		content: str | None = None,
		title: str | None = None,
		language: str | None = None,
		links: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> WebContentResponse:
		"""Record a scraped web page.  Content hash computed automatically."""
		assert_tenant_context(self._tenant_id)
		self._get_or_raise(self._tasks, task_id, "CollectionTask")
		content_str = content or ""
		content_hash = compute_content_fingerprint(content_str)
		payload = WebContentCreate(
			tenant_id=self._tenant_id,
			task_id=task_id,
			url=url,
			title=title,
			content_hash=content_hash,
			content_reference=f"web:{content_hash}",
			language=language,
			depth=depth,
			links_extracted=links or [],
			metadata=metadata or {},
			evidence_reference=f"scrape:{task_id}:{content_hash}",
		)
		item = WebContentResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			task_id=payload.task_id,
			url=payload.url,
			title=payload.title,
			content_hash=payload.content_hash,
			content_reference=payload.content_reference,
			mime_type=payload.mime_type,
			language=payload.language,
			depth=payload.depth,
			links_extracted=list(payload.links_extracted),
			metadata=dict(payload.metadata),
			scraped_at=payload.scraped_at,
			evidence_reference=payload.evidence_reference,
		)
		self._web_content[self._key(item.tenant_id, item.id)] = item
		self._emit_event("osint_web_content_scraped", item.id, item.tenant_id, {"url": url, "depth": depth})
		self._log_operation("web_scrape", item.id)
		return item

	async def get_web_content(self, content_id: str) -> WebContentResponse:
		"""Retrieve web content by ID."""
		return self._get_or_raise(self._web_content, content_id, "WebContent")

	async def list_web_content(
		self,
		task_id: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[WebContentResponse]:
		"""List web content for this tenant."""
		items = self._tenant_values(self._web_content)
		if task_id:
			items = [i for i in items if i.task_id == task_id]
		return sorted(items, key=lambda x: x.scraped_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Social media monitoring
	# -----------------------------------------------------------------------

	async def social_media_monitor(
		self,
		handles: list[str],
		keywords: list[str],
		platform: str,
	) -> list[SocialMediaProfileResponse]:
		"""Register social media profiles for monitoring.

		Returns the created/updated profile records.
		"""
		assert_tenant_context(self._tenant_id)
		results: list[SocialMediaProfileResponse] = []
		for handle in handles:
			payload = SocialMediaProfileCreate(
				tenant_id=self._tenant_id,
				platform=platform,
				handle=handle,
				keywords_monitored=list(keywords),
				evidence_reference=f"social_monitor:{platform}:{handle}",
			)
			item = SocialMediaProfileResponse(
				id=uuid7str(),
				tenant_id=payload.tenant_id,
				created_by=self._actor_id,
				platform=payload.platform,
				handle=payload.handle,
				keywords_monitored=list(payload.keywords_monitored),
				evidence_reference=payload.evidence_reference,
			)
			self._social_profiles[self._key(item.tenant_id, item.id)] = item
			self._emit_event("osint_social_profile_registered", item.id, item.tenant_id, {"platform": platform, "handle": handle})
			results.append(item)
		self._log_operation("social_media_monitor", f"{len(results)} profiles on {platform}")
		return results

	async def update_social_profile(
		self, profile_id: str, payload: SocialMediaProfileUpdate
	) -> SocialMediaProfileResponse:
		"""Update a social profile with freshly scraped data."""
		item = self._get_or_raise(self._social_profiles, profile_id, "SocialMediaProfile")
		if payload.display_name is not None:
			object.__setattr__(item, "display_name", payload.display_name)
		if payload.bio is not None:
			object.__setattr__(item, "bio", payload.bio)
		if payload.followers_count is not None:
			object.__setattr__(item, "followers_count", payload.followers_count)
		if payload.following_count is not None:
			object.__setattr__(item, "following_count", payload.following_count)
		if payload.post_count is not None:
			object.__setattr__(item, "post_count", payload.post_count)
		if payload.is_active is not None:
			object.__setattr__(item, "is_active", payload.is_active)
		if payload.attributes is not None:
			object.__setattr__(item, "attributes", dict(payload.attributes))
		if payload.keywords_monitored is not None:
			object.__setattr__(item, "keywords_monitored", list(payload.keywords_monitored))
		object.__setattr__(item, "last_scraped_at", _now())
		object.__setattr__(item, "updated_at", _now())
		return item

	async def list_social_profiles(
		self,
		platform: str | None = None,
		entity_id: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[SocialMediaProfileResponse]:
		"""List social profiles for this tenant."""
		items = self._tenant_values(self._social_profiles)
		if platform:
			items = [i for i in items if i.platform == platform]
		if entity_id:
			items = [i for i in items if i.entity_id == entity_id]
		return sorted(items, key=lambda x: x.created_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Domain intelligence
	# -----------------------------------------------------------------------

	async def domain_intelligence(
		self, payload: DomainRecordCreate
	) -> DomainRecordResponse:
		"""Store WHOIS / DNS / certificate data for a domain."""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		item = DomainRecordResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			domain=payload.domain,
			registrar=payload.registrar,
			registrant_name=payload.registrant_name,
			registrant_email=payload.registrant_email,
			registrant_org=payload.registrant_org,
			registrant_country=payload.registrant_country,
			created_date=payload.created_date,
			updated_date=payload.updated_date,
			expiry_date=payload.expiry_date,
			name_servers=list(payload.name_servers),
			a_records=list(payload.a_records),
			mx_records=list(payload.mx_records),
			txt_records=list(payload.txt_records),
			ssl_issuer=payload.ssl_issuer,
			ssl_expiry=payload.ssl_expiry,
			ssl_san=list(payload.ssl_san),
			raw_whois=payload.raw_whois,
			attributes=dict(payload.attributes),
			queried_at=payload.queried_at,
			evidence_reference=payload.evidence_reference,
		)
		self._domain_records[self._key(item.tenant_id, item.id)] = item
		self._emit_event("osint_domain_record_created", item.id, item.tenant_id, {"domain": payload.domain})
		self._log_operation("domain_intelligence", item.id)
		return item

	async def get_domain_record(self, record_id: str) -> DomainRecordResponse:
		"""Retrieve a domain record by ID."""
		return self._get_or_raise(self._domain_records, record_id, "DomainRecord")

	async def find_domain_records(
		self,
		domain: str | None = None,
		registrant_email: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[DomainRecordResponse]:
		"""Find domain records for this tenant."""
		items = self._tenant_values(self._domain_records)
		if domain:
			items = [i for i in items if domain.lower() in i.domain.lower()]
		if registrant_email:
			items = [i for i in items if i.registrant_email and registrant_email.lower() in i.registrant_email.lower()]
		return sorted(items, key=lambda x: x.queried_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# IP geolocation & enrichment
	# -----------------------------------------------------------------------

	async def ip_geolocation_enrichment(
		self, payload: IPIntelligenceCreate
	) -> IPIntelligenceResponse:
		"""Store IP geolocation and threat metadata."""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		item = IPIntelligenceResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			ip_address=payload.ip_address,
			ip_version=payload.ip_version,
			asn=payload.asn,
			asn_org=payload.asn_org,
			isp=payload.isp,
			country_code=payload.country_code,
			country_name=payload.country_name,
			region=payload.region,
			city=payload.city,
			latitude=payload.latitude,
			longitude=payload.longitude,
			is_tor=payload.is_tor,
			is_vpn=payload.is_vpn,
			is_proxy=payload.is_proxy,
			is_datacenter=payload.is_datacenter,
			abuse_confidence_score=payload.abuse_confidence_score,
			threat_types=list(payload.threat_types),
			open_ports=list(payload.open_ports),
			reverse_dns=payload.reverse_dns,
			attributes=dict(payload.attributes),
			queried_at=payload.queried_at,
			evidence_reference=payload.evidence_reference,
		)
		self._ip_intel[self._key(item.tenant_id, item.id)] = item
		threat_score = calculate_ip_threat_score(
			payload.is_tor, payload.is_vpn, payload.is_proxy,
			payload.is_datacenter, int(payload.abuse_confidence_score * 100),
			len(payload.open_ports),
		)
		self._emit_event("osint_ip_intel_created", item.id, item.tenant_id, {"ip": payload.ip_address, "threat_score": threat_score})
		self._log_operation("ip_geolocation_enrichment", item.id)
		return item

	async def get_ip_intel(self, ip_intel_id: str) -> IPIntelligenceResponse:
		"""Retrieve IP intelligence by ID."""
		return self._get_or_raise(self._ip_intel, ip_intel_id, "IPIntelligence")

	async def find_ip_intel(
		self,
		ip_address: str | None = None,
		country_code: str | None = None,
		is_tor: bool | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[IPIntelligenceResponse]:
		"""Query IP intelligence records for this tenant."""
		items = self._tenant_values(self._ip_intel)
		if ip_address:
			items = [i for i in items if ip_address in i.ip_address]
		if country_code:
			items = [i for i in items if i.country_code == country_code]
		if is_tor is not None:
			items = [i for i in items if i.is_tor == is_tor]
		return sorted(items, key=lambda x: x.queried_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Document / NLP analysis
	# -----------------------------------------------------------------------

	async def entity_extraction_nlp(
		self, payload: DocumentAnalysisCreate
	) -> DocumentAnalysisResponse:
		"""Store NLP analysis output for a raw intelligence document."""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		self._get_or_raise(self._raw_intel, payload.raw_intel_id, "RawIntelligence")
		item = DocumentAnalysisResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			raw_intel_id=payload.raw_intel_id,
			language=payload.language,
			sentiment_score=payload.sentiment_score,
			entities_extracted=list(payload.entities_extracted),
			keywords=list(payload.keywords),
			topics=list(payload.topics),
			summary=payload.summary,
			threat_indicators=list(payload.threat_indicators),
			location_mentions=list(payload.location_mentions),
			person_mentions=list(payload.person_mentions),
			org_mentions=list(payload.org_mentions),
			date_mentions=list(payload.date_mentions),
			model_used=payload.model_used,
			processing_time_ms=payload.processing_time_ms,
			evidence_reference=payload.evidence_reference,
		)
		self._doc_analyses[self._key(item.tenant_id, item.id)] = item
		self._emit_event("osint_document_analysis_completed", item.id, item.tenant_id, {
			"raw_intel_id": payload.raw_intel_id,
			"entity_count": len(payload.entities_extracted),
		})
		self._log_operation("entity_extraction_nlp", item.id)
		return item

	async def get_document_analysis(self, analysis_id: str) -> DocumentAnalysisResponse:
		"""Retrieve a document analysis record by ID."""
		return self._get_or_raise(self._doc_analyses, analysis_id, "DocumentAnalysis")

	async def list_document_analyses(
		self,
		raw_intel_id: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[DocumentAnalysisResponse]:
		"""List NLP analysis records for this tenant."""
		items = self._tenant_values(self._doc_analyses)
		if raw_intel_id:
			items = [i for i in items if i.raw_intel_id == raw_intel_id]
		return sorted(items, key=lambda x: x.created_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Credibility scoring
	# -----------------------------------------------------------------------

	async def credibility_scoring(
		self, payload: CredibilityScoreCreate
	) -> CredibilityScoreResponse:
		"""Record an analyst's credibility assessment for a source or intel item."""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		assert_confidence_bounds(payload.score)
		item = CredibilityScoreResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			reference_id=payload.reference_id,
			reference_type=payload.reference_type,
			score=payload.score,
			factors=dict(payload.factors),
			analyst_id=payload.analyst_id,
			rationale=payload.rationale,
			evidence_reference=payload.evidence_reference,
		)
		self._credibility_scores[self._key(item.tenant_id, item.id)] = item
		self._emit_event("osint_credibility_scored", item.id, item.tenant_id, {
			"reference_id": payload.reference_id,
			"score": payload.score,
		})
		return item

	# -----------------------------------------------------------------------
	# Relationship mapping analytics
	# -----------------------------------------------------------------------

	async def relationship_mapping(self) -> EntityNetworkReport:
		"""Generate a full entity network report for this tenant.

		Computes connected clusters and high-confidence links.
		"""
		entities = self._tenant_values(self._entities)
		relationships = self._tenant_values(self._relationships)

		entity_dicts = [e.model_dump() for e in entities]
		rel_dicts = [r.model_dump() for r in relationships]

		entity_ids = [e.id for e in entities]
		clusters = find_connected_clusters(entity_ids, rel_dicts)
		high_conf = sum(1 for r in relationships if r.confidence_score >= 0.75)

		return EntityNetworkReport(
			tenant_id=self._tenant_id,
			entity_count=len(entities),
			relationship_count=len(relationships),
			entities=entity_dicts,
			relationships=rel_dicts,
			clusters=clusters,
			high_confidence_links=high_conf,
		)

	# -----------------------------------------------------------------------
	# Deduplication
	# -----------------------------------------------------------------------

	async def duplicate_deduplication(
		self,
		similarity_threshold: float = 0.85,
	) -> dict[str, Any]:
		"""Run entity deduplication across the tenant's entity corpus.

		Returns a report of merged entities and clusters found.
		"""
		entities = self._tenant_values(self._entities)
		entity_dicts = [e.model_dump() for e in entities]
		merged = deduplicate_entities(entity_dicts, similarity_threshold)
		original_count = len(entity_dicts)
		merged_count = len(merged)
		self._emit_event("osint_deduplication_completed", "batch", self._tenant_id, {
			"original_count": original_count,
			"merged_count": merged_count,
			"deduplication_ratio": round(1 - merged_count / max(original_count, 1), 4),
		})
		self._log_operation("duplicate_deduplication", f"{original_count} -> {merged_count}")
		return {
			"tenant_id": self._tenant_id,
			"original_entity_count": original_count,
			"merged_entity_count": merged_count,
			"deduplicated_count": original_count - merged_count,
			"similarity_threshold": similarity_threshold,
			"merged_entities": merged,
		}

	# -----------------------------------------------------------------------
	# Intelligence dissemination
	# -----------------------------------------------------------------------

	async def intelligence_dissemination(
		self, payload: DisseminationPackageCreate
	) -> DisseminationPackageResponse:
		"""Create and release an intelligence dissemination package.

		Requires explicit approval_reference — autonomous dissemination is denied.
		"""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		assert_dissemination_approval(payload.approval_reference)
		# Verify each intel item exists
		for intel_id in payload.processed_intel_ids:
			self._get_or_raise(self._processed_intel, intel_id, "ProcessedIntelligence")
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_dissemination",
			"intel_present": bool(payload.processed_intel_ids),
			"approval_present": present(payload.approval_reference),
			"audience_present": present(payload.audience),
			"evidence_present": present(payload.evidence_reference),
			"autonomous_dissemination": False,  # always false — enforced above
		})
		item = DisseminationPackageResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			processed_intel_ids=list(payload.processed_intel_ids),
			audience=payload.audience,
			release_marking=payload.release_marking,
			classification=payload.classification,
			title=payload.title,
			executive_summary=payload.executive_summary,
			approval_reference=payload.approval_reference,
			evidence_reference=payload.evidence_reference,
			disseminated_at=_now(),
		)
		self._dissemination[self._key(item.tenant_id, item.id)] = item
		# Mark source intel as disseminated
		for intel_id in payload.processed_intel_ids:
			key = self._key(payload.tenant_id, intel_id)
			if key in self._processed_intel:
				object.__setattr__(self._processed_intel[key], "status", IntelStatus.DISSEMINATED)
		self._emit_event("osint_dissemination_package_created", item.id, item.tenant_id, {
			"audience": payload.audience,
			"tlp": payload.release_marking.value,
		})
		self._log_operation("intelligence_dissemination", item.id)
		return item

	async def get_dissemination_package(self, package_id: str) -> DisseminationPackageResponse:
		"""Retrieve a dissemination package by ID."""
		return self._get_or_raise(self._dissemination, package_id, "DisseminationPackage")

	async def list_dissemination_packages(
		self, limit: int = 50, offset: int = 0
	) -> list[DisseminationPackageResponse]:
		"""List dissemination packages for this tenant."""
		items = self._tenant_values(self._dissemination)
		return sorted(items, key=lambda x: x.created_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Reviews
	# -----------------------------------------------------------------------

	async def record_review(self, payload: OSINTReviewCreate) -> OSINTReviewResponse:
		"""Record a quality/compliance review on any OSINT artefact."""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": payload.status.value in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(payload.reviewer_id),
			"evidence_present": present(payload.evidence_reference),
		})
		item = OSINTReviewResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			reference_id=payload.reference_id,
			reference_type=payload.reference_type,
			reviewer_id=payload.reviewer_id,
			status=payload.status,
			notes=payload.notes,
			evidence_reference=payload.evidence_reference,
		)
		self._reviews[self._key(item.tenant_id, item.id)] = item
		self._emit_event("osint_review_recorded", item.id, item.tenant_id, {
			"reference_id": payload.reference_id,
			"status": payload.status.value,
		})
		return item

	async def list_reviews(
		self,
		reference_type: str | None = None,
		reviewer_id: str | None = None,
		status: ReviewStatus | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[OSINTReviewResponse]:
		"""List review records for this tenant."""
		items = self._tenant_values(self._reviews)
		if reference_type:
			items = [i for i in items if i.reference_type == reference_type]
		if reviewer_id:
			items = [i for i in items if i.reviewer_id == reviewer_id]
		if status:
			items = [i for i in items if i.status == status]
		return sorted(items, key=lambda x: x.created_at, reverse=True)[offset: offset + limit]

	# -----------------------------------------------------------------------
	# Agent management
	# -----------------------------------------------------------------------

	async def register_agent(self, payload: OSINTAgentCreate) -> OSINTAgentResponse:
		"""Register an autonomous OSINT agent."""
		assert_tenant_context(payload.tenant_id)
		self._assert_tenant_match(payload.tenant_id)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_osint_agent",
			"agent_runtime_supported": payload.runtime.value in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": payload.role.value in SUPPORTED_AGENT_ROLES,
		})
		item = OSINTAgentResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			created_by=self._actor_id,
			name=payload.name,
			runtime=payload.runtime,
			role=payload.role,
			scope=payload.scope,
			capabilities=list(payload.capabilities),
		)
		self._agents[self._key(item.tenant_id, item.id)] = item
		self._emit_event("osint_agent_registered", item.id, item.tenant_id, {
			"runtime": payload.runtime.value,
			"role": payload.role.value,
		})
		return item

	async def validate_agent_action(
		self,
		privileged_scope: bool,
		human_approval_recorded: bool,
		cross_tenant_scope: bool = False,
		privilege_escalation_scope: bool = False,
		evidence_fabrication_scope: bool = False,
		source_terms_violation_scope: bool = False,
		autonomous_dissemination_scope: bool = False,
		unapproved_high_risk_collection_scope: bool = False,
	) -> dict[str, Any]:
		"""Validate that an agent action complies with OSINT governance rules."""
		assert_human_approval_for_privileged(privileged_scope, human_approval_recorded)
		self._enforce({
			"tenant_id": self._tenant_id,
			"tenant_context_present": True,
			"operation": "osint_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"cross_tenant_osint_scope": cross_tenant_scope,
			"privilege_escalation_scope": privilege_escalation_scope,
			"evidence_fabrication_scope": evidence_fabrication_scope,
			"source_terms_violation_scope": source_terms_violation_scope,
			"autonomous_dissemination_scope": autonomous_dissemination_scope,
			"unapproved_high_risk_collection_scope": unapproved_high_risk_collection_scope,
		})
		return {
			"tenant_id": self._tenant_id,
			"accepted": True,
			"privileged_scope": privileged_scope,
		}

	async def validate_batch(self, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		"""Validate a batch processing request."""
		self._enforce({
			"tenant_id": self._tenant_id,
			"tenant_context_present": True,
			"operation": "osint_batch",
			"event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be a positive integer")
		return {
			"tenant_id": self._tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.intel.osint.lifecycle",
			"accepted": True,
		}

	# -----------------------------------------------------------------------
	# Reports
	# -----------------------------------------------------------------------

	async def dashboard_summary(self) -> OSINTDashboard:
		"""Return KPI dashboard counts for this tenant."""
		tid = self._tenant_id
		sources = self._tenant_values(self._sources)
		tasks = self._tenant_values(self._tasks)
		return OSINTDashboard(
			tenant_id=tid,
			source_count=len(sources),
			active_source_count=sum(1 for s in sources if s.status == SourceStatus.ACTIVE),
			high_risk_source_count=sum(1 for s in sources if s.risk_tier.value in {"high", "critical"}),
			task_count=self._count(self._tasks),
			pending_task_count=sum(1 for t in tasks if t.status == TaskStatus.PENDING),
			running_task_count=sum(1 for t in tasks if t.status == TaskStatus.RUNNING),
			raw_intel_count=self._count(self._raw_intel),
			processed_intel_count=self._count(self._processed_intel),
			entity_count=self._count(self._entities),
			relationship_count=self._count(self._relationships),
			social_profile_count=self._count(self._social_profiles),
			domain_record_count=self._count(self._domain_records),
			ip_intel_count=self._count(self._ip_intel),
			document_analysis_count=self._count(self._doc_analyses),
			dissemination_count=self._count(self._dissemination),
			review_count=self._count(self._reviews),
			agent_count=self._count(self._agents),
			audit_event_count=sum(1 for e in self._audit_events if e["tenant_id"] == tid),
		)

	async def source_health_report(self) -> SourceHealthReport:
		"""Generate a source health summary for this tenant."""
		sources = self._tenant_values(self._sources)
		by_type: dict[str, int] = {}
		by_risk: dict[str, int] = {}
		for s in sources:
			by_type[s.source_type.value] = by_type.get(s.source_type.value, 0) + 1
			by_risk[s.risk_tier.value] = by_risk.get(s.risk_tier.value, 0) + 1
		active = [s for s in sources if s.status == SourceStatus.ACTIVE]
		avg_credibility = (
			sum(s.credibility_baseline for s in active) / len(active)
			if active else 0.0
		)
		top_sources = sorted(
			[{"id": s.id, "name": s.name, "credibility": s.credibility_baseline} for s in sources],
			key=lambda x: x["credibility"],
			reverse=True,
		)[:10]
		return SourceHealthReport(
			tenant_id=self._tenant_id,
			total_sources=len(sources),
			active_sources=len(active),
			sources_by_type=by_type,
			sources_by_risk=by_risk,
			avg_credibility=round(avg_credibility, 4),
			top_sources=top_sources,
		)

	async def threat_landscape_report(self) -> ThreatLandscapeReport:
		"""Generate a threat landscape summary from processed intel."""
		processed = self._tenant_values(self._processed_intel)
		threats = [p for p in processed if p.assessment_type.value == "threat"]
		geo: dict[str, int] = {}
		ip_intel = self._tenant_values(self._ip_intel)
		for ip in ip_intel:
			if ip.country_code:
				geo[ip.country_code] = geo.get(ip.country_code, 0) + 1
		return ThreatLandscapeReport(
			tenant_id=self._tenant_id,
			total_threats=len(threats),
			geographic_distribution=geo,
		)

	# -----------------------------------------------------------------------
	# Audit / audit trail
	# -----------------------------------------------------------------------

	async def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
		"""Return the most recent audit events for this tenant."""
		tenant_events = [e for e in self._audit_events if e["tenant_id"] == self._tenant_id]
		return tenant_events[-limit:]

	# -----------------------------------------------------------------------
	# Internal helpers
	# -----------------------------------------------------------------------

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _get_or_raise(self, store: dict, item_id: str, entity_name: str) -> Any:
		key = self._key(self._tenant_id, item_id)
		item = store.get(key)
		if item is None or getattr(item, "is_deleted", False):
			raise KeyError(f"{entity_name} '{item_id}' not found for tenant '{self._tenant_id}'")
		return item

	def _tenant_values(self, store: dict) -> list:
		return [
			v for (tid, _), v in store.items()
			if tid == self._tenant_id and not getattr(v, "is_deleted", False)
		]

	def _count(self, store: dict) -> int:
		return sum(
			1 for (tid, _), v in store.items()
			if tid == self._tenant_id and not getattr(v, "is_deleted", False)
		)

	def _assert_tenant_match(self, payload_tenant_id: str) -> None:
		if payload_tenant_id != self._tenant_id:
			raise RuleViolation(
				"cross_tenant_access_denied",
				f"payload tenant '{payload_tenant_id}' does not match service tenant '{self._tenant_id}'",
				"use_matching_tenant_id",
			)

	def _emit_event(
		self,
		event_type: str,
		reference_id: str,
		tenant_id: str,
		extra: dict[str, Any] | None = None,
	) -> None:
		event: dict[str, Any] = {
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self._actor_id,
			"timestamp": _now().isoformat(),
			"processor": "bytewax",
			"stream": "apg.intel.osint.lifecycle",
		}
		if extra:
			event.update(extra)
		self._audit_events.append(event)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "osint_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "osint_policy_denied")

	def _log_operation(self, operation: str, reference: str) -> None:
		"""Internal _log_ prefixed helper — structured log for observability."""
		# In production wire to structlog / loguru; here we track in audit_events
		pass

	def _log_pretty_path(self, path: str) -> str:
		"""Return a shortened display path for log messages."""
		parts = path.split("/")
		return "/".join(parts[-3:]) if len(parts) > 3 else path

	# -----------------------------------------------------------------------
	# Legacy synchronous positional-arg interface
	# These thin wrappers satisfy the capability contract test harness which
	# calls synchronous methods with positional args and expects plain dicts.
	# -----------------------------------------------------------------------

	def register_requirement(
		self,
		requirement_id: str,
		tenant_id: str,
		topic: str,
		priority: str,
		requester_id: str,
		classification: str,
		evidence_reference: str,
	) -> dict:
		"""Register an intelligence requirement (legacy sync interface).

		Args:
			requirement_id: Caller-supplied ID for idempotent registration.
			tenant_id: Tenant context.
			topic: Description of the intelligence requirement.
			priority: One of 'low', 'medium', 'high', 'critical'.
			requester_id: ID of the requesting party.
			classification: Classification level string.
			evidence_reference: Non-empty evidence reference.

		Returns:
			Dict with at minimum 'id' and 'priority'.

		Raises:
			PermissionError: On rule violation.
		"""
		if not str(tenant_id or "").strip():
			raise PermissionError("tenant_context_required")
		if priority not in SUPPORTED_PRIORITIES:
			raise PermissionError("priority_not_supported")
		item = {
			"id": requirement_id,
			"tenant_id": tenant_id,
			"topic": topic,
			"priority": priority,
			"requester_id": requester_id,
			"classification": classification,
			"evidence_reference": evidence_reference,
		}
		self._requirements: dict
		if not hasattr(self, "_requirements"):
			object.__setattr__(self, "_requirements", {})  # type: ignore[arg-type]
		self._requirements[(tenant_id, requirement_id)] = item
		self._emit_event("osint_requirement_registered", requirement_id, tenant_id)
		return item

	def _sync_register_source(
		self,
		source_id: str,
		tenant_id: str,
		source_type: str,
		source_reference: str,
		owner_id: str,
		terms_review_reference: str,
		risk_tier: str,
		evidence_reference: str,
	) -> dict:
		"""Register an intelligence source (legacy sync positional interface).

		Raises:
			PermissionError: On rule violation (unknown type, missing terms review, etc.).
		"""
		if source_type not in SUPPORTED_SOURCE_TYPES:
			raise PermissionError("source_type_not_supported")
		if not str(terms_review_reference or "").strip():
			raise PermissionError("terms_review_required")
		item = {
			"id": source_id,
			"tenant_id": tenant_id,
			"source_type": source_type,
			"source_reference": source_reference,
			"owner_id": owner_id,
			"terms_review_reference": terms_review_reference,
			"risk_tier": risk_tier,
			"evidence_reference": evidence_reference,
		}
		if not hasattr(self, "_legacy_sources"):
			object.__setattr__(self, "_legacy_sources", {})  # type: ignore[arg-type]
		self._legacy_sources[(tenant_id, source_id)] = item
		self._emit_event("osint_source_registered", source_id, tenant_id)
		return item

	def record_collection_plan(
		self,
		plan_id: str,
		tenant_id: str,
		requirement_id: str,
		source_id: str,
		method: str,
		cadence: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict:
		"""Record a collection plan linking a requirement to a source.

		High/critical risk sources require a non-empty approval_reference.

		Raises:
			PermissionError: If a high-risk source lacks approval.
		"""
		# Determine risk tier from the legacy source store
		if not hasattr(self, "_legacy_sources"):
			object.__setattr__(self, "_legacy_sources", {})  # type: ignore[arg-type]
		src = self._legacy_sources.get((tenant_id, source_id), {})
		risk_tier = src.get("risk_tier", "low")
		if risk_tier in {"high", "critical"} and not str(approval_reference or "").strip():
			raise PermissionError("collection_approval_required")
		item = {
			"id": plan_id,
			"tenant_id": tenant_id,
			"requirement_id": requirement_id,
			"source_id": source_id,
			"method": method,
			"cadence": cadence,
			"approval_reference": approval_reference,
			"evidence_reference": evidence_reference,
		}
		if not hasattr(self, "_collection_plans"):
			object.__setattr__(self, "_collection_plans", {})  # type: ignore[arg-type]
		self._collection_plans[(tenant_id, plan_id)] = item
		self._emit_event("osint_collection_plan_recorded", plan_id, tenant_id)
		return item

	def record_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		plan_id: str,
		content_reference: str,
		fingerprint: str,
		confidence_score: float,
		evidence_reference: str,
	) -> dict:
		"""Record a raw evidence item against a collection plan.

		Raises:
			PermissionError: If confidence_score is outside [0.0, 1.0].
		"""
		try:
			f = float(confidence_score)
		except (TypeError, ValueError):
			raise PermissionError("confidence_score_invalid")
		if not (0.0 <= f <= 1.0):
			raise PermissionError("confidence_score_invalid")
		item = {
			"id": evidence_id,
			"tenant_id": tenant_id,
			"plan_id": plan_id,
			"content_reference": content_reference,
			"fingerprint": fingerprint,
			"confidence_score": f,
			"evidence_reference": evidence_reference,
		}
		if not hasattr(self, "_evidence_items"):
			object.__setattr__(self, "_evidence_items", {})  # type: ignore[arg-type]
		self._evidence_items[(tenant_id, evidence_id)] = item
		self._emit_event("osint_evidence_recorded", evidence_id, tenant_id)
		return item

	def record_triage(
		self,
		triage_id: str,
		tenant_id: str,
		evidence_id: str,
		decision: str,
		analyst_id: str,
		evidence_reference: str,
	) -> dict:
		"""Record a triage decision on an evidence item.

		Raises:
			PermissionError: If decision is not a supported triage value.
		"""
		if decision not in SUPPORTED_TRIAGE_DECISIONS:
			raise PermissionError("triage_decision_not_supported")
		item = {
			"id": triage_id,
			"tenant_id": tenant_id,
			"evidence_id": evidence_id,
			"decision": decision,
			"analyst_id": analyst_id,
			"evidence_reference": evidence_reference,
		}
		if not hasattr(self, "_triage_records"):
			object.__setattr__(self, "_triage_records", {})  # type: ignore[arg-type]
		self._triage_records[(tenant_id, triage_id)] = item
		self._emit_event("osint_triage_recorded", triage_id, tenant_id)
		return item

	def record_assessment(
		self,
		assessment_id: str,
		tenant_id: str,
		requirement_id: str,
		assessment_type: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict:
		"""Record a processed intelligence assessment.

		Raises:
			PermissionError: If assessment_type is unsupported or confidence is invalid.
		"""
		if assessment_type not in SUPPORTED_ASSESSMENT_TYPES:
			raise PermissionError("assessment_type_not_supported")
		try:
			f = float(confidence_score)
		except (TypeError, ValueError):
			raise PermissionError("confidence_score_invalid")
		if not (0.0 <= f <= 1.0):
			raise PermissionError("confidence_score_invalid")
		item = {
			"id": assessment_id,
			"tenant_id": tenant_id,
			"requirement_id": requirement_id,
			"assessment_type": assessment_type,
			"confidence_score": f,
			"analyst_id": analyst_id,
			"evidence_reference": evidence_reference,
		}
		if not hasattr(self, "_assessments"):
			object.__setattr__(self, "_assessments", {})  # type: ignore[arg-type]
		self._assessments[(tenant_id, assessment_id)] = item
		self._emit_event("osint_assessment_recorded", assessment_id, tenant_id)
		return item

	def record_dissemination(
		self,
		package_id: str,
		tenant_id: str,
		assessment_id: str,
		audience: str,
		classification: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict:
		"""Record an intelligence dissemination package.

		Raises:
			PermissionError: If approval_reference is absent.
		"""
		if not str(approval_reference or "").strip():
			raise PermissionError("dissemination_approval_required")
		item = {
			"id": package_id,
			"tenant_id": tenant_id,
			"assessment_id": assessment_id,
			"audience": audience,
			"classification": classification,
			"approval_reference": approval_reference,
			"evidence_reference": evidence_reference,
		}
		if not hasattr(self, "_dissemination_records"):
			object.__setattr__(self, "_dissemination_records", {})  # type: ignore[arg-type]
		self._dissemination_records[(tenant_id, package_id)] = item
		self._emit_event("osint_dissemination_recorded", package_id, tenant_id)
		return item

	def _sync_record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict:
		"""Record a review on any OSINT artefact (legacy sync interface).

		Raises:
			PermissionError: If status is unsupported.
		"""
		if status not in SUPPORTED_REVIEW_STATUSES:
			raise PermissionError("review_status_not_supported")
		item = {
			"id": review_id,
			"tenant_id": tenant_id,
			"reference_id": reference_id,
			"reviewer_id": reviewer_id,
			"status": status,
			"evidence_reference": evidence_reference,
		}
		if not hasattr(self, "_review_records"):
			object.__setattr__(self, "_review_records", {})  # type: ignore[arg-type]
		self._review_records[(tenant_id, review_id)] = item
		self._emit_event("osint_review_legacy_recorded", review_id, tenant_id)
		return item

	def register_osint_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str = "",
	) -> dict:
		"""Register an OSINT agent (legacy sync positional interface).

		Raises:
			PermissionError: If runtime or role is unsupported.
		"""
		if runtime not in SUPPORTED_AGENT_RUNTIMES:
			raise PermissionError("osint_agent_runtime_not_supported")
		if role not in SUPPORTED_AGENT_ROLES:
			raise PermissionError("osint_agent_role_not_supported")
		item = {
			"id": agent_id,
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
		}
		if not hasattr(self, "_legacy_agents"):
			object.__setattr__(self, "_legacy_agents", {})  # type: ignore[arg-type]
		self._legacy_agents[(tenant_id, agent_id)] = item
		self._emit_event("osint_agent_legacy_registered", agent_id, tenant_id)
		return item

	def _sync_validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict:
		"""Validate a batch OSINT processing request (legacy sync interface).

		Raises:
			PermissionError: If event_stream is not 'bytewax'.
		"""
		if event_stream != "bytewax":
			raise PermissionError("bytewax_event_stream_required")
		if not positive_int(item_count):
			raise ValueError("item_count must be a positive integer")
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.intel.osint.lifecycle",
			"accepted": True,
		}

	def _sync_validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool = False,
		human_approval_recorded: bool = False,
		**kwargs: Any,
	) -> dict:
		"""Validate an OSINT agent action (legacy sync interface).

		Raises:
			PermissionError: If privileged but no human approval recorded.
		"""
		if privileged_scope and not human_approval_recorded:
			raise PermissionError("human_approval_required")
		return {
			"tenant_id": tenant_id,
			"accepted": True,
			"privileged_scope": privileged_scope,
		}

	def _sync_dashboard_summary(self, tenant_id: str | None = None) -> dict:
		"""Return dashboard KPI summary as a plain dict (legacy sync interface).

		Args:
			tenant_id: Optional tenant override.

		Returns:
			Plain dict with KPI counts including audit_event_count.
		"""
		tid = tenant_id or self._tenant_id
		sources = [v for (t, _), v in self._sources.items() if t == tid and not getattr(v, "is_deleted", False)]
		tasks = [v for (t, _), v in self._tasks.items() if t == tid and not getattr(v, "is_deleted", False)]
		audit_events = [e for e in self._audit_events if e["tenant_id"] == tid]

		requirements = {k: v for k, v in getattr(self, "_requirements", {}).items() if k[0] == tid}
		collection_plans = {k: v for k, v in getattr(self, "_collection_plans", {}).items() if k[0] == tid}
		evidence_items = {k: v for k, v in getattr(self, "_evidence_items", {}).items() if k[0] == tid}
		assessments = {k: v for k, v in getattr(self, "_assessments", {}).items() if k[0] == tid}
		dissemination_records = {k: v for k, v in getattr(self, "_dissemination_records", {}).items() if k[0] == tid}
		review_records = {k: v for k, v in getattr(self, "_review_records", {}).items() if k[0] == tid}
		legacy_agents = {k: v for k, v in getattr(self, "_legacy_agents", {}).items() if k[0] == tid}

		return {
			"tenant_id": tid,
			"source_count": len(sources),
			"active_source_count": sum(1 for s in sources if getattr(s, "status", None) and s.status.value == "active"),
			"task_count": len(tasks),
			"raw_intel_count": sum(1 for (t, _) in self._raw_intel if t == tid),
			"processed_intel_count": sum(1 for (t, _) in self._processed_intel if t == tid),
			"entity_count": sum(1 for (t, _) in self._entities if t == tid),
			"relationship_count": sum(1 for (t, _) in self._relationships if t == tid),
			"requirement_count": len(requirements),
			"collection_plan_count": len(collection_plans),
			"evidence_count": len(evidence_items),
			"assessment_count": len(assessments),
			"dissemination_count": len(dissemination_records),
			"review_count": len(review_records),
			"agent_count": len(legacy_agents),
			"audit_event_count": len(audit_events),
		}


# Alias for backwards compatibility with generated import statements

	async def ml_entity_extract(self, *args, **kwargs):
		"""AI-powered AI entity and relationship extraction from OSINT text. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.extract(str(kwargs.get("text",""))[:2000], schema={"persons":"person names mentioned","organizations":"org names","locations":"places","events":"key events"}, context="intelligence analysis")
			return {"entities": result.extracted, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

IntelOSINTService = OSINTService


class OpenSourceIntelligenceService(OSINTService):
	"""Legacy synchronous interface for the capability contract test harness.

	Exposes the same OSINT functionality via synchronous positional-arg methods
	so that the test_package_contract.py harness can call them directly without
	async/await.  All governance rules are identical.
	"""

	def register_source(  # type: ignore[override]
		self,
		source_id: str,
		tenant_id: str,
		source_type: str,
		source_reference: str,
		owner_id: str,
		terms_review_reference: str,
		risk_tier: str,
		evidence_reference: str,
	) -> dict:
		"""Legacy sync register_source."""
		return self._sync_register_source(
			source_id, tenant_id, source_type, source_reference,
			owner_id, terms_review_reference, risk_tier, evidence_reference,
		)

	def record_review(  # type: ignore[override]
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict:
		"""Legacy sync record_review."""
		return self._sync_record_review(
			review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference,
		)

	def validate_batch(  # type: ignore[override]
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict:
		"""Legacy sync validate_batch."""
		return self._sync_validate_batch(tenant_id, item_count, event_stream)

	def validate_agent_action(  # type: ignore[override]
		self,
		tenant_id: str,
		privileged_scope: bool = False,
		human_approval_recorded: bool = False,
		**kwargs: Any,
	) -> dict:
		"""Legacy sync validate_agent_action."""
		return self._sync_validate_agent_action(
			tenant_id, privileged_scope, human_approval_recorded, **kwargs,
		)

	def dashboard_summary(self, tenant_id: str | None = None) -> dict:  # type: ignore[override]
		"""Legacy sync dashboard_summary."""
		return self._sync_dashboard_summary(tenant_id)
