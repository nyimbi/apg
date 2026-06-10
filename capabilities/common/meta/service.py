#!/usr/bin/env python3
"""
APG Metadata Management - Main Service
Unified metadata management service orchestrating all components

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import asdict, dataclass, field
from enum import Enum
from uuid_extensions import uuid7str

try:
	from .database import MetaDatabaseManager, create_database_manager
	from .integrations import APGMetadataIntegrationManager, create_apg_integration_manager
	from .discovery import MetadataDiscoveryService, DiscoverySchedule, create_discovery_service
	from .ai_classifier import AIClassificationEngine, create_ai_classifier
	from .lineage_engine import DataLineageEngine, LineageEdge, create_lineage_engine
	from .search_engine import MetadataSearchEngine, SearchQuery, create_search_engine
	from .connectors import ConnectorConfig
	_RUNTIME_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
	MetaDatabaseManager = APGMetadataIntegrationManager = MetadataDiscoveryService = object
	DiscoverySchedule = AIClassificationEngine = DataLineageEngine = LineageEdge = object
	MetadataSearchEngine = SearchQuery = ConnectorConfig = object
	create_database_manager = create_apg_integration_manager = None
	create_discovery_service = create_ai_classifier = create_lineage_engine = create_search_engine = None
	_RUNTIME_IMPORT_ERROR = exc

from .capability_contract import (
	PRIVILEGED_META_AGENT_ROLES,
	SUPPORTED_META_AGENT_ROLES,
	SUPPORTED_META_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


class ServiceStatus(str, Enum):
	"""Service status enumeration"""
	INITIALIZING = "initializing"
	RUNNING = "running"
	DEGRADED = "degraded"
	STOPPED = "stopped"
	ERROR = "error"


@dataclass
class ServiceHealth:
	"""Service health status"""
	service_name: str = "metadata_management"
	status: ServiceStatus = ServiceStatus.STOPPED
	uptime_seconds: float = 0.0
	last_health_check: datetime = field(default_factory=datetime.utcnow)
	
	# Component health
	database_healthy: bool = False
	discovery_healthy: bool = False
	ai_classifier_healthy: bool = False
	lineage_engine_healthy: bool = False
	search_engine_healthy: bool = False
	integrations_healthy: bool = False
	
	# Performance metrics
	total_assets: int = 0
	total_discoveries: int = 0
	total_searches: int = 0
	total_classifications: int = 0
	avg_response_time_ms: float = 0.0
	
	# Error tracking
	error_count_24h: int = 0
	last_error: Optional[str] = None
	warnings: List[str] = field(default_factory=list)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for API response"""
		return {
			"service_name": self.service_name,
			"status": self.status.value,
			"uptime_seconds": self.uptime_seconds,
			"last_health_check": self.last_health_check.isoformat(),
			"components": {
				"database": self.database_healthy,
				"discovery": self.discovery_healthy,
				"ai_classifier": self.ai_classifier_healthy,
				"lineage_engine": self.lineage_engine_healthy,
				"search_engine": self.search_engine_healthy,
				"integrations": self.integrations_healthy
			},
			"metrics": {
				"total_assets": self.total_assets,
				"total_discoveries": self.total_discoveries,
				"total_searches": self.total_searches,
				"total_classifications": self.total_classifications,
				"avg_response_time_ms": self.avg_response_time_ms
			},
			"issues": {
				"error_count_24h": self.error_count_24h,
				"last_error": self.last_error,
				"warnings": self.warnings
			}
		}


@dataclass
class MetaAssetRecord:
	"""Dependency-light metadata asset record for generated applications."""

	record_id: str
	tenant_id: str
	asset_id: str
	asset_type: str
	name: str
	business_key: str
	source_system: str
	owner: str | None
	steward: str | None
	sensitivity: str = "internal"
	status: str = "draft"
	decision: str = "allow"
	quality_score: float | None = None
	classification_id: str | None = None
	lineage_available: bool = False
	age_days: int = 0
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	tags: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaDiscoveryJobRecord:
	"""Discovery schedule or job evidence."""

	job_id: str
	tenant_id: str
	connector_type: str
	source_system: str
	schedule: str
	connector_approved: bool
	schedule_review_current: bool
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	discovered_asset_ids: list[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaClassificationRecord:
	"""Classification evidence and steward review state."""

	classification_id: str
	tenant_id: str
	asset_id: str
	label: str
	confidence: float
	classification_complete: bool
	decision: str
	status: str
	steward_review_recorded: bool = False
	steward: str | None = None
	review_notes: str | None = None
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	reviewed_at: datetime | None = None


@dataclass
class MetaLineageRecord:
	"""Lineage edge evidence."""

	lineage_id: str
	tenant_id: str
	source_asset_id: str
	target_asset_id: str
	lineage_type: str
	depth: int
	evidence: str | None
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaQualityRecord:
	"""Metadata quality assessment evidence."""

	quality_id: str
	tenant_id: str
	asset_id: str
	score: float
	dimensions: dict[str, float]
	assessor: str
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaCertificationRecord:
	"""Certification decision state."""

	certification_id: str
	tenant_id: str
	asset_id: str
	requester: str
	decision: str
	status: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	review_notes: str | None = None
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaGlossaryTermRecord:
	"""Business glossary term and ownership evidence."""

	term_id: str
	tenant_id: str
	term: str
	definition: str
	owner: str | None
	linked_asset_ids: list[str] = field(default_factory=list)
	decision: str = "allow"
	status: str = "active"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaCatalogAgentRecord:
	"""First-class metadata catalog governance agent registration."""

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
class MetaLifecycleBatchRecord:
	"""Bytewax metadata lifecycle-batch validation evidence."""

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
class MetaAuditEventRecord:
	"""Dependency-light META audit event."""

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


class MetaService:
	"""Dependency-light META lifecycle and guardrail control plane."""

	def __init__(self, tenant_id: str = "default"):
		self.tenant_id = tenant_id
		self.contract = get_capability_contract(tenant_id)
		self._agent_runtimes = set(SUPPORTED_META_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_META_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_META_AGENT_ROLES)
		self.assets: dict[str, MetaAssetRecord] = {}
		self.discovery_jobs: dict[str, MetaDiscoveryJobRecord] = {}
		self.classifications: dict[str, MetaClassificationRecord] = {}
		self.lineage: dict[str, MetaLineageRecord] = {}
		self.quality_assessments: dict[str, MetaQualityRecord] = {}
		self.certifications: dict[str, MetaCertificationRecord] = {}
		self.glossary_terms: dict[str, MetaGlossaryTermRecord] = {}
		self.catalog_agents: dict[str, MetaCatalogAgentRecord] = {}
		self.lifecycle_batches: dict[str, MetaLifecycleBatchRecord] = {}
		self.audit_events: list[MetaAuditEventRecord] = []
		self.records: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def create_record(
		self,
		*,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		record_id = self._require_text(record_id, "record_id")
		tenant_id = self._require_text(tenant_id, "tenant_id")
		record = {
			"id": record_id,
			"tenant_id": tenant_id,
			"metadata": dict(metadata or {}),
			"status": status,
			"created_at": datetime.utcnow().isoformat(),
		}
		self.records[f"{tenant_id}:{record_id}"] = record
		self._audit(tenant_id, "record.created", record_id, "system", _allow_result(), record)
		return record

	def register_asset(
		self,
		*,
		tenant_id: str,
		asset_id: str,
		asset_type: str,
		name: str,
		business_key: str,
		source_system: str,
		owner: str | None = None,
		steward: str | None = None,
		sensitivity: str = "internal",
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
		age_days: int = 0,
	) -> MetaAssetRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		asset_type = self._require_text(asset_type, "asset_type")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_asset",
			"unsupported_asset_type": asset_type not in self._supported_asset_types(tenant_id),
			"business_key_present": bool(str(business_key or "").strip()),
			"source_system_present": bool(str(source_system or "").strip()),
			"asset_sensitivity": "restricted" if sensitivity in {"restricted", "pii", "phi", "pci", "secret"} else sensitivity,
			"steward_assigned": bool(steward),
		}
		decision = evaluate_capability_rules(context)
		record = MetaAssetRecord(
			record_id=uuid7str(),
			tenant_id=tenant_id,
			asset_id=self._require_text(asset_id, "asset_id"),
			asset_type=asset_type,
			name=self._require_text(name, "name"),
			business_key=str(business_key or "").strip(),
			source_system=str(source_system or "").strip(),
			owner=owner.strip() if isinstance(owner, str) and owner.strip() else None,
			steward=steward.strip() if isinstance(steward, str) and steward.strip() else None,
			sensitivity=sensitivity,
			status="draft" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			decision=decision["decision"],
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
			tags=list(tags or []),
			metadata=dict(metadata or {}),
			age_days=age_days,
		)
		self.assets[self._asset_key(tenant_id, record.asset_id)] = record
		self._audit(tenant_id, "asset.registered", record.asset_id, record.owner or "system", decision, context)
		return record

	def schedule_discovery(
		self,
		*,
		tenant_id: str,
		connector_type: str,
		source_system: str,
		schedule: str,
		connector_approved: bool,
		schedule_review_current: bool,
	) -> MetaDiscoveryJobRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		connector_type = self._require_choice(connector_type, "connector_type", set(self.describe(tenant_id)["configuration"]["discovery"]["allowed_connector_types"]))
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "schedule_discovery",
			"connector_approved": connector_approved,
			"schedule_review_current": schedule_review_current,
		}
		decision = evaluate_capability_rules(context)
		record = MetaDiscoveryJobRecord(
			job_id=uuid7str(),
			tenant_id=tenant_id,
			connector_type=connector_type,
			source_system=self._require_text(source_system, "source_system"),
			schedule=self._require_text(schedule, "schedule"),
			connector_approved=connector_approved,
			schedule_review_current=schedule_review_current,
			decision=decision["decision"],
			status="scheduled" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.discovery_jobs[record.job_id] = record
		self._audit(tenant_id, "discovery.scheduled", record.job_id, record.source_system, decision, context)
		return record

	def record_discovery_result(self, *, job_id: str, discovered_asset_ids: list[str]) -> MetaDiscoveryJobRecord:
		if job_id not in self.discovery_jobs:
			raise KeyError(f"Discovery job {job_id} not found")
		record = self.discovery_jobs[job_id]
		if record.status != "scheduled":
			raise ValueError(f"Discovery job {job_id} is {record.status} and cannot record results")
		record.discovered_asset_ids = list(discovered_asset_ids)
		record.status = "completed"
		record.policy_decision = "allow"
		record.review_reasons = []
		record.review_evidence = self._review_evidence(_allow_result(), review_recorded=True)
		self._audit(record.tenant_id, "discovery.completed", record.job_id, record.source_system, _allow_result(), asdict(record))
		return record

	def classify_asset(
		self,
		*,
		tenant_id: str,
		asset_id: str,
		label: str,
		confidence: float,
		classification_complete: bool,
		steward_review_recorded: bool = False,
	) -> MetaClassificationRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		asset = self._require_asset(tenant_id, asset_id)
		if confidence < 0.0 or confidence > 1.0:
			raise ValueError("confidence must be between 0 and 1")
		context = {
			"tenant_context_present": bool(tenant_id),
			"asset_sensitivity": "restricted" if label in self.describe(tenant_id)["configuration"]["classification"]["sensitive_labels"] else asset.sensitivity,
			"classification_complete": classification_complete,
			"classification_confidence": confidence,
			"steward_review_recorded": steward_review_recorded,
		}
		decision = evaluate_capability_rules(context)
		record = MetaClassificationRecord(
			classification_id=uuid7str(),
			tenant_id=tenant_id,
			asset_id=asset.asset_id,
			label=self._require_text(label, "label"),
			confidence=confidence,
			classification_complete=classification_complete,
			decision=decision["decision"],
			status="accepted" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			steward_review_recorded=steward_review_recorded,
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision, steward_review_recorded),
		)
		self.classifications[record.classification_id] = record
		if decision["decision"] == "allow":
			asset.classification_id = record.classification_id
			asset.sensitivity = "restricted" if label in self.describe(tenant_id)["configuration"]["classification"]["sensitive_labels"] else asset.sensitivity
			asset.updated_at = datetime.utcnow()
		self._audit(tenant_id, "asset.classified", asset.asset_id, "classifier", decision, context)
		return record

	def review_classification(self, *, classification_id: str, steward: str, review_notes: str) -> MetaClassificationRecord:
		if classification_id not in self.classifications:
			raise KeyError(f"Classification {classification_id} not found")
		record = self.classifications[classification_id]
		context = {
			"tenant_context_present": bool(record.tenant_id),
			"operation": "review_classification",
			"review_notes_present": bool(str(review_notes or "").strip()),
		}
		decision = evaluate_capability_rules(context)
		record.steward = self._require_text(steward, "steward")
		record.review_notes = str(review_notes or "").strip() or None
		record.steward_review_recorded = decision["decision"] == "allow"
		record.decision = "reviewed" if decision["decision"] == "allow" else decision["decision"]
		record.status = "reviewed" if decision["decision"] == "allow" else "review_denied"
		record.matched_rules = decision["matched_rules"]
		record.policy_decision = decision["decision"]
		record.review_reasons = self._reasons(decision)
		record.review_evidence = self._review_evidence(decision, record.steward_review_recorded)
		record.reviewed_at = datetime.utcnow()
		self._audit(record.tenant_id, "classification.reviewed", record.classification_id, record.steward, decision, context)
		return record

	def capture_lineage(
		self,
		*,
		tenant_id: str,
		source_asset_id: str,
		target_asset_id: str,
		lineage_type: str,
		depth: int = 1,
		evidence: str | None = None,
	) -> MetaLineageRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		source_registered = self._asset_key(tenant_id, source_asset_id) in self.assets
		target_registered = self._asset_key(tenant_id, target_asset_id) in self.assets
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "capture_lineage",
			"source_and_target_registered": source_registered and target_registered,
			"lineage_depth": depth,
		}
		decision = evaluate_capability_rules(context)
		record = MetaLineageRecord(
			lineage_id=uuid7str(),
			tenant_id=tenant_id,
			source_asset_id=self._require_text(source_asset_id, "source_asset_id"),
			target_asset_id=self._require_text(target_asset_id, "target_asset_id"),
			lineage_type=self._require_text(lineage_type, "lineage_type"),
			depth=depth,
			evidence=evidence,
			decision=decision["decision"],
			status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.lineage[record.lineage_id] = record
		if decision["decision"] == "allow":
			self._require_asset(tenant_id, source_asset_id).lineage_available = True
			self._require_asset(tenant_id, target_asset_id).lineage_available = True
		self._audit(tenant_id, "lineage.captured", record.lineage_id, "system", decision, context)
		return record

	def assess_quality(
		self,
		*,
		tenant_id: str,
		asset_id: str,
		score: float,
		dimensions: dict[str, float],
		assessor: str,
	) -> MetaQualityRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		asset = self._require_asset(tenant_id, asset_id)
		if score < 0.0 or score > 100.0 or any(value < 0.0 or value > 100.0 for value in dimensions.values()):
			raise ValueError("quality scores must be between 0 and 100")
		record = MetaQualityRecord(
			quality_id=uuid7str(),
			tenant_id=tenant_id,
			asset_id=asset.asset_id,
			score=score,
			dimensions=dict(dimensions),
			assessor=self._require_text(assessor, "assessor"),
			decision="allow",
			status="accepted",
			policy_decision="allow",
			review_reasons=[],
			review_evidence=self._review_evidence(_allow_result()),
		)
		self.quality_assessments[record.quality_id] = record
		asset.quality_score = score
		asset.updated_at = datetime.utcnow()
		self._audit(tenant_id, "quality.assessed", asset.asset_id, record.assessor, _allow_result(), asdict(record))
		return record

	def request_certification(self, *, tenant_id: str, asset_id: str, requester: str, review_notes: str | None = None) -> MetaCertificationRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		asset = self._require_asset(tenant_id, asset_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "certify_asset",
			"certification_requested": True,
			"lineage_available": asset.lineage_available,
			"quality_score": asset.quality_score or 0.0,
			"asset_age_days": asset.age_days,
			"freshness_review_recorded": bool(review_notes),
		}
		decision = evaluate_capability_rules(context)
		record = MetaCertificationRecord(
			certification_id=uuid7str(),
			tenant_id=tenant_id,
			asset_id=asset.asset_id,
			requester=self._require_text(requester, "requester"),
			decision=decision["decision"],
			status="certified" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision, bool(review_notes)),
			review_notes=review_notes,
		)
		self.certifications[record.certification_id] = record
		if decision["decision"] == "allow":
			asset.status = "certified"
			asset.updated_at = datetime.utcnow()
		self._audit(tenant_id, "asset.certification_requested", asset.asset_id, record.requester, decision, context)
		return record

	def publish_asset(self, *, tenant_id: str, asset_id: str) -> MetaAssetRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		asset = self._require_asset(tenant_id, asset_id)
		classification = self._classification_for_asset(tenant_id, asset.asset_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_asset",
			"asset_owner_assigned": bool(asset.owner),
			"quality_assessment_present": asset.quality_score is not None,
			"asset_sensitivity": "restricted" if asset.sensitivity in {"restricted", "pii", "phi", "pci", "secret"} else asset.sensitivity,
			"classification_complete": bool(classification and classification.classification_complete),
			"steward_assigned": bool(asset.steward),
		}
		decision = evaluate_capability_rules(context)
		asset.decision = decision["decision"]
		asset.matched_rules = decision["matched_rules"]
		asset.policy_decision = decision["decision"]
		asset.review_reasons = self._reasons(decision)
		asset.review_evidence = self._review_evidence(decision)
		if decision["decision"] == "allow":
			asset.status = "published"
		asset.updated_at = datetime.utcnow()
		self._audit(tenant_id, "asset.publish_evaluated", asset.asset_id, asset.owner or "system", decision, context)
		return asset

	def register_glossary_term(
		self,
		*,
		tenant_id: str,
		term: str,
		definition: str,
		owner: str | None,
		linked_asset_ids: list[str] | None = None,
	) -> MetaGlossaryTermRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_glossary_term",
			"term_owner_assigned": bool(owner),
		}
		decision = evaluate_capability_rules(context)
		record = MetaGlossaryTermRecord(
			term_id=uuid7str(),
			tenant_id=tenant_id,
			term=self._require_text(term, "term"),
			definition=self._require_text(definition, "definition"),
			owner=owner.strip() if isinstance(owner, str) and owner.strip() else None,
			linked_asset_ids=list(linked_asset_ids or []),
			decision=decision["decision"],
			status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.glossary_terms[record.term_id] = record
		self._audit(tenant_id, "glossary.term.registered", record.term_id, record.owner or "system", decision, context)
		return record

	def retire_asset(self, *, tenant_id: str, asset_id: str, impact_analysis_present: bool, actor: str) -> MetaAssetRecord:
		tenant_id = self._require_text(tenant_id, "tenant_id")
		asset = self._require_asset(tenant_id, asset_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_asset",
			"impact_analysis_present": impact_analysis_present,
		}
		decision = evaluate_capability_rules(context)
		asset.decision = decision["decision"]
		asset.matched_rules = decision["matched_rules"]
		asset.policy_decision = decision["decision"]
		asset.review_reasons = self._reasons(decision)
		asset.review_evidence = self._review_evidence(decision)
		if decision["decision"] == "allow":
			asset.status = "retired"
			asset.updated_at = datetime.utcnow()
		self._audit(tenant_id, "asset.retired", asset.asset_id, self._require_text(actor, "actor"), decision, context)
		return asset

	def register_catalog_agent(
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
	) -> MetaCatalogAgentRecord:
		"""Register a first-class metadata catalog agent with guardrail evidence."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		agent_id = self._require_text(agent_id, "agent_id")
		name = self._require_text(name, "name")
		runtime_value = self._normalize_agent_token(runtime)
		role_value = self._normalize_agent_token(role)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_catalog_agent",
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
		record_key = self._agent_key(tenant_id, agent_id)
		if record_key in self.catalog_agents:
			raise ValueError(f"catalog_agent_already_exists:{agent_id}")
		record = MetaCatalogAgentRecord(
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
			status="active" if decision["decision"] == "allow" else self._status_for_decision(decision["decision"]),
			decision=decision["decision"],
			matched_rules=list(decision["matched_rules"]),
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision, bool(human_approval_required)),
		)
		self.catalog_agents[record_key] = record
		self._audit(tenant_id, "agent.registered", agent_id, record.owner, decision, asdict(record))
		return record

	def validate_meta_lifecycle_batch(
		self,
		*,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> MetaLifecycleBatchRecord:
		"""Validate that metadata lifecycle mutation batches flow through Bytewax."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("meta_lifecycle_batch_empty")
		stream_value = self._normalize_agent_token(event_stream)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_meta_lifecycle_batch",
			"event_stream": stream_value,
		}
		decision = evaluate_capability_rules(context)
		accepted = decision["decision"] == "allow"
		record = MetaLifecycleBatchRecord(
			batch_id=uuid7str(),
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			accepted=accepted,
			decision=decision["decision"],
			matched_rules=list(decision["matched_rules"]),
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
			status="accepted" if accepted else "denied",
		)
		self.lifecycle_batches[record.batch_id] = record
		self._audit(tenant_id, f"lifecycle_batch.{record.status}", stream_value, "meta", decision, asdict(record))
		if not accepted:
			raise PermissionError(self._first_reason(decision))
		return record

	def list_records(self, tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant_id = tenant_id or self.tenant_id
		collections: dict[str, Any] = {
			"assets": self.assets.values(),
			"discovery_jobs": self.discovery_jobs.values(),
			"classifications": self.classifications.values(),
			"lineage": self.lineage.values(),
			"quality_assessments": self.quality_assessments.values(),
			"certifications": self.certifications.values(),
			"glossary_terms": self.glossary_terms.values(),
			"catalog_agents": self.catalog_agents.values(),
			"lifecycle_batches": self.lifecycle_batches.values(),
			"audit_events": self.audit_events,
			"records": self.records.values(),
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
			dict(record) if isinstance(record, dict) else asdict(record)
			for record in values
			if (record.get("tenant_id") if isinstance(record, dict) else getattr(record, "tenant_id", None)) == tenant_id
		]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant_id = tenant_id or self.tenant_id
		return {
			"tenant_id": tenant_id,
			"asset_count": len(self.list_records(tenant_id, "assets")),
			"published_asset_count": sum(1 for row in self.list_records(tenant_id, "assets") if row["status"] == "published"),
			"discovery_job_count": len(self.list_records(tenant_id, "discovery_jobs")),
			"classification_review_count": sum(1 for row in self.list_records(tenant_id, "classifications") if row["status"] == "pending_review"),
			"lineage_edge_count": len(self.list_records(tenant_id, "lineage")),
			"certified_asset_count": sum(1 for row in self.list_records(tenant_id, "assets") if row["status"] == "certified"),
			"glossary_term_count": len(self.list_records(tenant_id, "glossary_terms")),
			"catalog_agent_count": len(self.list_records(tenant_id, "catalog_agents")),
			"pending_catalog_agent_review_count": sum(1 for row in self.list_records(tenant_id, "catalog_agents") if row["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_records(tenant_id, "lifecycle_batches")),
			"denied_lifecycle_batch_count": sum(1 for row in self.list_records(tenant_id, "lifecycle_batches") if not row["accepted"]),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
			"audit_event_count": len(self.list_records(tenant_id, "audit_events")),
		}

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all META records awaiting steward or human review."""
		tenant_id = tenant_id or self.tenant_id
		items = (
			self.list_records(tenant_id, "assets")
			+ self.list_records(tenant_id, "discovery_jobs")
			+ self.list_records(tenant_id, "classifications")
			+ self.list_records(tenant_id, "lineage")
			+ self.list_records(tenant_id, "quality_assessments")
			+ self.list_records(tenant_id, "certifications")
			+ self.list_records(tenant_id, "glossary_terms")
			+ self.list_records(tenant_id, "catalog_agents")
			+ self.list_records(tenant_id, "lifecycle_batches")
		)
		return [
			item
			for item in items
			if item.get("status") in {"pending", "pending_review", "review_required"}
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
		self.audit_events.append(MetaAuditEventRecord(
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
		return list(dict.fromkeys(
			str(action["reason"])
			for action in result.get("actions", [])
			if action.get("reason")
		))

	def _review_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": list(dict.fromkeys(
				str(action.get("required_action"))
				for action in result.get("actions", [])
				if action.get("required_action")
			)),
			"reasons": self._reasons(result),
			"review_recorded": bool(review_recorded),
		}

	def _supported_asset_types(self, tenant_id: str) -> set[str]:
		return set(self.describe(tenant_id)["configuration"]["catalog"]["supported_asset_types"])

	def _require_asset(self, tenant_id: str, asset_id: str) -> MetaAssetRecord:
		asset_id = self._require_text(asset_id, "asset_id")
		record = self.assets.get(self._asset_key(tenant_id, asset_id))
		if record is None:
			raise KeyError(f"Asset {asset_id} not found for tenant {tenant_id}")
		if record.status == "denied":
			raise ValueError(f"Asset {asset_id} is denied and cannot continue lifecycle operations")
		return record

	def _classification_for_asset(self, tenant_id: str, asset_id: str) -> MetaClassificationRecord | None:
		for classification in reversed(list(self.classifications.values())):
			if classification.tenant_id == tenant_id and classification.asset_id == asset_id:
				return classification
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
		text = MetaService._require_text(value, field_name)
		if text not in allowed:
			raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
		return text

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
		return "meta_operation_denied"

	@staticmethod
	def _asset_key(tenant_id: str, asset_id: str) -> str:
		return f"{tenant_id}:{asset_id}"


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


class APGMetadataService:
	"""Main APG Metadata Management Service orchestrating all components"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.service_start_time = datetime.utcnow()
		
		# Component instances
		self.db_manager: Optional[MetaDatabaseManager] = None
		self.integration_manager: Optional[APGMetadataIntegrationManager] = None
		self.discovery_service: Optional[MetadataDiscoveryService] = None
		self.ai_classifier: Optional[AIClassificationEngine] = None
		self.lineage_engine: Optional[DataLineageEngine] = None
		self.search_engine: Optional[MetadataSearchEngine] = None
		
		# Service state
		self.health = ServiceHealth()
		self.initialized = False
		
		# Performance tracking
		self.request_count = 0
		self.total_response_time = 0.0
		self.error_count_24h = 0
		
		# Background tasks
		self.health_check_task: Optional[asyncio.Task] = None
		self.maintenance_task: Optional[asyncio.Task] = None
		
		# Service configuration
		self.enable_auto_discovery = config.get('enable_auto_discovery', True)
		self.enable_ai_classification = config.get('enable_ai_classification', True)
		self.enable_lineage_tracking = config.get('enable_lineage_tracking', True)
		self.enable_advanced_search = config.get('enable_advanced_search', True)
		
		# Health check interval
		self.health_check_interval = config.get('health_check_interval_seconds', 60)
		self.maintenance_interval = config.get('maintenance_interval_seconds', 3600)  # 1 hour
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize all service components"""
		if _RUNTIME_IMPORT_ERROR is not None:
			raise ModuleNotFoundError(
				"META production runtime requires optional database/search dependencies such as asyncpg"
			) from _RUNTIME_IMPORT_ERROR
		if self.initialized:
			return {"status": "already_initialized"}
		
		self.health.status = ServiceStatus.INITIALIZING
		
		try:
			await self._log_info("Starting APG Metadata Management Service initialization")
			
			# Initialize core database manager
			await self._log_info("Initializing database manager...")
			self.db_manager = await create_database_manager(self.config.get('database', {}))
			self.health.database_healthy = True
			await self._log_info("✓ Database manager initialized")
			
			# Initialize APG integrations
			await self._log_info("Initializing APG integrations...")
			self.integration_manager = await create_apg_integration_manager(
				self.config.get('integrations', {}),
				self.db_manager
			)
			self.health.integrations_healthy = True
			await self._log_info("✓ APG integrations initialized")
			
			# Initialize discovery service
			if self.enable_auto_discovery:
				await self._log_info("Initializing discovery service...")
				self.discovery_service = await create_discovery_service(
					self.db_manager,
					self.integration_manager,
					self.config.get('discovery', {})
				)
				self.health.discovery_healthy = True
				await self._log_info("✓ Discovery service initialized")
			
			# Initialize AI classifier
			if self.enable_ai_classification:
				await self._log_info("Initializing AI classifier...")
				self.ai_classifier = await create_ai_classifier(
					self.db_manager,
					self.integration_manager,
					self.config.get('ai_classifier', {})
				)
				self.health.ai_classifier_healthy = True
				await self._log_info("✓ AI classifier initialized")
			
			# Initialize lineage engine
			if self.enable_lineage_tracking:
				await self._log_info("Initializing lineage engine...")
				self.lineage_engine = await create_lineage_engine(
					self.db_manager,
					self.integration_manager,
					self.config.get('lineage', {})
				)
				self.health.lineage_engine_healthy = True
				await self._log_info("✓ Lineage engine initialized")
			
			# Initialize search engine
			if self.enable_advanced_search:
				await self._log_info("Initializing search engine...")
				self.search_engine = await create_search_engine(
					self.db_manager,
					self.integration_manager,
					self.config.get('search', {})
				)
				self.health.search_engine_healthy = True
				await self._log_info("✓ Search engine initialized")
			
			# Start background tasks
			await self._start_background_tasks()
			
			# Update service status
			self.health.status = ServiceStatus.RUNNING
			self.health.last_health_check = datetime.utcnow()
			self.initialized = True
			
			await self._log_info("🚀 APG Metadata Management Service initialized successfully")
			
			return {
				"status": "initialized",
				"components_initialized": {
					"database_manager": True,
					"integration_manager": True,
					"discovery_service": self.enable_auto_discovery,
					"ai_classifier": self.enable_ai_classification,
					"lineage_engine": self.enable_lineage_tracking,
					"search_engine": self.enable_advanced_search
				},
				"service_capabilities": await self._get_service_capabilities(),
				"initialization_time_ms": (datetime.utcnow() - self.service_start_time).total_seconds() * 1000
			}
			
		except Exception as e:
			self.health.status = ServiceStatus.ERROR
			self.health.last_error = str(e)
			await self._log_error(f"Service initialization failed: {str(e)}")
			raise
	
	async def shutdown(self):
		"""Shutdown all service components gracefully"""
		if not self.initialized:
			return
		
		try:
			await self._log_info("Shutting down APG Metadata Management Service...")
			
			self.health.status = ServiceStatus.STOPPED
			
			# Stop background tasks
			if self.health_check_task and not self.health_check_task.done():
				self.health_check_task.cancel()
				try:
					await self.health_check_task
				except asyncio.CancelledError:
					pass
			
			if self.maintenance_task and not self.maintenance_task.done():
				self.maintenance_task.cancel()
				try:
					await self.maintenance_task
				except asyncio.CancelledError:
					pass
			
			# Shutdown components in reverse order
			if self.search_engine:
				# Search engine doesn't have explicit shutdown
				pass
			
			if self.lineage_engine:
				# Lineage engine doesn't have explicit shutdown
				pass
			
			if self.ai_classifier:
				# AI classifier doesn't have explicit shutdown
				pass
			
			if self.discovery_service:
				await self.discovery_service.shutdown()
			
			if self.integration_manager:
				await self.integration_manager.shutdown()
			
			if self.db_manager:
				await self.db_manager.close()
			
			self.initialized = False
			
			await self._log_info("✓ APG Metadata Management Service shutdown completed")
			
		except Exception as e:
			await self._log_error(f"Service shutdown failed: {str(e)}")
	
	# === Discovery Operations ===
	
	async def create_discovery_schedule(self, schedule: DiscoverySchedule) -> str:
		"""Create a new discovery schedule"""
		if not self.discovery_service:
			raise RuntimeError("Discovery service not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			schedule_id = await self.discovery_service.create_discovery_schedule(schedule)
			
			await self._track_performance(start_time)
			await self._log_info(f"Created discovery schedule: {schedule_id}")
			
			return schedule_id
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def run_discovery(self, schedule_id: str, override_config: Dict[str, Any] = None) -> str:
		"""Run a discovery job"""
		if not self.discovery_service:
			raise RuntimeError("Discovery service not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			job_id = await self.discovery_service.run_discovery_job(schedule_id, override_config)
			
			await self._track_performance(start_time)
			await self._log_info(f"Started discovery job: {job_id}")
			
			self.health.total_discoveries += 1
			
			return job_id
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def get_discovery_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
		"""Get discovery job status"""
		if not self.discovery_service:
			raise RuntimeError("Discovery service not initialized")
		
		return await self.discovery_service.get_discovery_job_status(job_id)
	
	# === AI Classification Operations ===
	
	async def classify_column_data(self,
				       column_name: str,
				       data_type: str,
				       sample_data: List[Any],
				       context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Classify column data using AI"""
		if not self.ai_classifier:
			raise RuntimeError("AI classifier not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			result = await self.ai_classifier.classify_column_data(
				column_name, data_type, sample_data, context
			)
			
			await self._track_performance(start_time)
			await self._log_info(f"Classified column '{column_name}' as '{result.classification}'")
			
			self.health.total_classifications += 1
			
			return result.to_dict()
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	# === Lineage Operations ===
	
	async def add_lineage_relationship(self, edge: LineageEdge) -> str:
		"""Add a lineage relationship"""
		if not self.lineage_engine:
			raise RuntimeError("Lineage engine not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			edge_id = await self.lineage_engine.add_lineage_relationship(edge)
			
			await self._track_performance(start_time)
			await self._log_info(f"Added lineage relationship: {edge_id}")
			
			return edge_id
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def get_lineage_path(self,
				   asset_id: str,
				   tenant_id: str,
				   direction: str = "both",
				   max_depth: int = None) -> List[Dict[str, Any]]:
		"""Get lineage paths for an asset"""
		if not self.lineage_engine:
			raise RuntimeError("Lineage engine not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			from .lineage_engine import LineageDirection
			
			lineage_direction = LineageDirection(direction)
			paths = await self.lineage_engine.get_lineage_path(
				asset_id, tenant_id, lineage_direction, max_depth
			)
			
			await self._track_performance(start_time)
			await self._log_info(f"Retrieved {len(paths)} lineage paths for asset {asset_id}")
			
			return [path.to_dict() for path in paths]
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def analyze_impact(self,
				 asset_id: str,
				 tenant_id: str,
				 change_type: str = "schema_change",
				 change_details: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Perform impact analysis"""
		if not self.lineage_engine:
			raise RuntimeError("Lineage engine not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			result = await self.lineage_engine.analyze_impact(
				asset_id, tenant_id, change_type, change_details
			)
			
			await self._track_performance(start_time)
			await self._log_info(f"Impact analysis completed for {asset_id}: {result.total_impacted_assets} assets")
			
			return result.to_dict()
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	# === Search Operations ===
	
	async def search_metadata(self, query: SearchQuery) -> Dict[str, Any]:
		"""Search metadata assets"""
		if not self.search_engine:
			raise RuntimeError("Search engine not initialized")
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			response = await self.search_engine.search(query)
			
			await self._track_performance(start_time)
			await self._log_info(f"Search completed: '{query.query_text}' -> {response.total_results} results")
			
			self.health.total_searches += 1
			
			return response.to_dict()
			
		except Exception as e:
			await self._track_error(e)
			raise
	
	# === Asset Operations ===
	
	async def get_asset(self, asset_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Get metadata asset by ID"""
		start_time = asyncio.get_event_loop().time()
		
		try:
			async with self.db_manager.get_session(tenant_id) as session:
				from sqlalchemy import select
				from .models import MetaAsset
				
				stmt = select(MetaAsset).where(
					MetaAsset.id == asset_id,
					MetaAsset.tenant_id == tenant_id,
					MetaAsset.is_deleted == False
				)
				
				result = await session.execute(stmt)
				asset = result.scalar_one_or_none()
				
				if asset:
					await self._track_performance(start_time)
					return await self._asset_to_dict(asset)
				
				return None
				
		except Exception as e:
			await self._track_error(e)
			raise
	
	async def list_assets(self,
			      tenant_id: str,
			      filters: Dict[str, Any] = None,
			      limit: int = 100,
			      offset: int = 0) -> Dict[str, Any]:
		"""List metadata assets with filtering"""
		start_time = asyncio.get_event_loop().time()
		
		try:
			async with self.db_manager.get_session(tenant_id) as session:
				from sqlalchemy import select
				from .models import MetaAsset
				
				stmt = select(MetaAsset).where(
					MetaAsset.tenant_id == tenant_id,
					MetaAsset.is_deleted == False
				)
				
				# Apply filters
				if filters:
					for field, value in filters.items():
						if hasattr(MetaAsset, field):
							attr = getattr(MetaAsset, field)
							if isinstance(value, list):
								stmt = stmt.where(attr.in_(value))
							else:
								stmt = stmt.where(attr == value)
				
				# Apply pagination
				stmt = stmt.offset(offset).limit(limit)
				
				result = await session.execute(stmt)
				assets = result.scalars().all()
				
				# Get total count for pagination
				count_stmt = select(MetaAsset.id).where(
					MetaAsset.tenant_id == tenant_id,
					MetaAsset.is_deleted == False
				)
				
				if filters:
					for field, value in filters.items():
						if hasattr(MetaAsset, field):
							attr = getattr(MetaAsset, field)
							if isinstance(value, list):
								count_stmt = count_stmt.where(attr.in_(value))
							else:
								count_stmt = count_stmt.where(attr == value)
				
				total_result = await session.execute(count_stmt)
				total_count = len(total_result.scalars().all())
				
				await self._track_performance(start_time)
				
				return {
					"assets": [await self._asset_to_dict(asset) for asset in assets],
					"pagination": {
						"offset": offset,
						"limit": limit,
						"total": total_count,
						"has_more": (offset + limit) < total_count
					}
				}
				
		except Exception as e:
			await self._track_error(e)
			raise
	
	# === Health and Monitoring ===
	
	async def get_health_status(self) -> Dict[str, Any]:
		"""Get service health status"""
		await self._update_health_status()
		return self.health.to_dict()
	
	async def get_service_metrics(self) -> Dict[str, Any]:
		"""Get service performance metrics"""
		metrics = {
			"uptime_seconds": (datetime.utcnow() - self.service_start_time).total_seconds(),
			"request_count": self.request_count,
			"avg_response_time_ms": self.total_response_time / max(self.request_count, 1),
			"error_rate": self.error_count_24h / max(self.request_count, 1),
		}
		
		# Add component-specific metrics
		if self.search_engine:
			search_metrics = await self.search_engine.get_search_analytics()
			metrics["search"] = search_metrics
		
		if self.ai_classifier:
			classifier_metrics = await self.ai_classifier.get_classification_stats()
			metrics["ai_classifier"] = classifier_metrics
		
		if self.db_manager:
			db_metrics = await self.db_manager.get_database_stats()
			metrics["database"] = db_metrics
		
		return metrics
	
	# === Internal Methods ===
	
	async def _get_service_capabilities(self) -> Dict[str, Any]:
		"""Get service capabilities"""
		return {
			"auto_discovery": self.enable_auto_discovery,
			"ai_classification": self.enable_ai_classification,
			"lineage_tracking": self.enable_lineage_tracking,
			"advanced_search": self.enable_advanced_search,
			"natural_language_queries": self.enable_advanced_search,
			"real_time_lineage": self.enable_lineage_tracking,
			"federated_learning": self.enable_ai_classification,
			"apg_integration": True,
			"multi_tenant": True,
			"graph_analytics": self.enable_lineage_tracking
		}
	
	async def _start_background_tasks(self):
		"""Start background monitoring and maintenance tasks"""
		self.health_check_task = asyncio.create_task(self._health_check_loop())
		self.maintenance_task = asyncio.create_task(self._maintenance_loop())
	
	async def _health_check_loop(self):
		"""Background health check loop"""
		while self.initialized:
			try:
				await asyncio.sleep(self.health_check_interval)
				await self._update_health_status()
			except asyncio.CancelledError:
				break
			except Exception as e:
				await self._log_error(f"Health check failed: {str(e)}")
	
	async def _maintenance_loop(self):
		"""Background maintenance loop"""
		while self.initialized:
			try:
				await asyncio.sleep(self.maintenance_interval)
				await self._run_maintenance_tasks()
			except asyncio.CancelledError:
				break
			except Exception as e:
				await self._log_error(f"Maintenance task failed: {str(e)}")
	
	async def _update_health_status(self):
		"""Update service health status"""
		try:
			self.health.last_health_check = datetime.utcnow()
			self.health.uptime_seconds = (datetime.utcnow() - self.service_start_time).total_seconds()
			
			# Check database health
			if self.db_manager:
				db_health = await self.db_manager.health_check()
				self.health.database_healthy = db_health.is_healthy
			
			# Update asset count
			if self.db_manager:
				try:
					async with self.db_manager.get_session() as session:
						from sqlalchemy import select, func
						from .models import MetaAsset
						
						stmt = select(func.count(MetaAsset.id)).where(
							MetaAsset.is_deleted == False
						)
						result = await session.execute(stmt)
						self.health.total_assets = result.scalar() or 0
				except Exception:
					pass
			
			# Calculate average response time
			if self.request_count > 0:
				self.health.avg_response_time_ms = self.total_response_time / self.request_count
			
			# Determine overall status
			component_health = [
				self.health.database_healthy,
				self.health.discovery_healthy or not self.enable_auto_discovery,
				self.health.ai_classifier_healthy or not self.enable_ai_classification,
				self.health.lineage_engine_healthy or not self.enable_lineage_tracking,
				self.health.search_engine_healthy or not self.enable_advanced_search,
				self.health.integrations_healthy
			]
			
			if all(component_health):
				self.health.status = ServiceStatus.RUNNING
			elif any(component_health):
				self.health.status = ServiceStatus.DEGRADED
			else:
				self.health.status = ServiceStatus.ERROR
			
		except Exception as e:
			self.health.status = ServiceStatus.ERROR
			self.health.last_error = str(e)
			await self._log_error(f"Health status update failed: {str(e)}")
	
	async def _run_maintenance_tasks(self):
		"""Run periodic maintenance tasks"""
		await self._log_info("Running maintenance tasks...")
		
		try:
			# Database maintenance
			if self.db_manager:
				await self.db_manager.optimize_performance()
			
			# Reset 24h error counter
			current_time = datetime.utcnow()
			if not hasattr(self, '_last_error_reset') or (current_time - self._last_error_reset).days >= 1:
				self.error_count_24h = 0
				self._last_error_reset = current_time
			
			await self._log_info("✓ Maintenance tasks completed")
			
		except Exception as e:
			await self._log_error(f"Maintenance tasks failed: {str(e)}")
	
	async def _track_performance(self, start_time: float):
		"""Track request performance"""
		response_time = (asyncio.get_event_loop().time() - start_time) * 1000
		self.request_count += 1
		self.total_response_time += response_time
	
	async def _track_error(self, error: Exception):
		"""Track error occurrence"""
		self.error_count_24h += 1
		self.health.last_error = str(error)
		await self._log_error(f"Service error: {str(error)}")
	
	async def _asset_to_dict(self, asset) -> Dict[str, Any]:
		"""Convert MetaAsset to dictionary"""
		return {
			"id": asset.id,
			"name": asset.name,
			"display_name": asset.display_name,
			"description": asset.description,
			"asset_type": asset.asset_type,
			"source_system": asset.source_system,
			"source_system_type": asset.source_system_type,
			"external_id": asset.external_id,
			"status": asset.status,
			"business_domain": asset.business_domain,
			"schema_info": asset.schema_info,
			"quality_score": asset.quality_score,
			"tags": asset.tags,
			"owner": asset.owner,
			"steward": asset.steward,
			"custom_attributes": asset.custom_attributes,
			"created_at": asset.created_at.isoformat() if asset.created_at else None,
			"updated_at": asset.updated_at.isoformat() if asset.updated_at else None,
			"created_by": asset.created_by,
			"updated_by": asset.updated_by
		}
	
	async def _log_info(self, message: str):
		"""Log info message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META SERVICE INFO: {message}")
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META SERVICE ERROR: {message}")


# Factory function for easy initialization
async def create_metadata_service(config: Dict[str, Any] = None) -> APGMetadataService:
	"""Factory function to create and initialize metadata service"""
	service = APGMetadataService(config)
	await service.initialize()
	return service


# Service singleton for global access
_metadata_service_instance: Optional[APGMetadataService] = None


async def get_metadata_service(config: Dict[str, Any] = None) -> APGMetadataService:
	"""Get or create the global metadata service instance"""
	global _metadata_service_instance
	
	if _metadata_service_instance is None:
		_metadata_service_instance = await create_metadata_service(config)
	
	return _metadata_service_instance


async def shutdown_metadata_service():
	"""Shutdown the global metadata service instance"""
	global _metadata_service_instance
	
	if _metadata_service_instance:
		await _metadata_service_instance.shutdown()
		_metadata_service_instance = None
