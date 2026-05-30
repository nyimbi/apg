"""Domain models for the APG Scraper/Data Harvesting capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class HarvestSource:
	"""Tenant-owned source with terms, credentials, rate, and compliance metadata."""

	id: str
	tenant_id: str
	name: str
	source_type: str
	owner: str
	endpoint: str
	terms_evidence: str
	credential_vault_ref: str
	rate_limit_per_minute: int
	robots_policy_attached: bool = True
	pii_expected: bool = False
	pii_policy_attached: bool = False
	sensitive_source: bool = False
	source_review_recorded: bool = False
	tags: list[str] = field(default_factory=list)
	created_at: datetime | None = None
	updated_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"source_type": self.source_type,
			"owner": self.owner,
			"endpoint": self.endpoint,
			"terms_evidence": self.terms_evidence,
			"credential_vault_ref": self.credential_vault_ref,
			"rate_limit_per_minute": self.rate_limit_per_minute,
			"robots_policy_attached": self.robots_policy_attached,
			"pii_expected": self.pii_expected,
			"pii_policy_attached": self.pii_policy_attached,
			"sensitive_source": self.sensitive_source,
			"source_review_recorded": self.source_review_recorded,
			"tags": list(self.tags),
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
		}


@dataclass
class ExtractorProfile:
	"""Schema and parser profile for harvested payloads."""

	id: str
	tenant_id: str
	name: str
	extractor_type: str
	owner: str
	schema: dict[str, Any]
	output_mapping: dict[str, str] = field(default_factory=dict)
	schema_validation_required: bool = True
	incremental_cursor_field: str | None = None
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"extractor_type": self.extractor_type,
			"owner": self.owner,
			"schema": dict(self.schema),
			"output_mapping": dict(self.output_mapping),
			"schema_validation_required": self.schema_validation_required,
			"incremental_cursor_field": self.incremental_cursor_field,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class HarvestJob:
	"""Configured harvest job binding source, extractor, schedule, and pipeline."""

	id: str
	tenant_id: str
	name: str
	source_id: str
	extractor_profile_id: str
	owner: str
	mode: str = "incremental"
	schedule_policy_attached: bool = True
	pipeline_handoff_required: bool = True
	pipeline_target: str | None = None
	enabled: bool = True
	created_at: datetime | None = None
	updated_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"source_id": self.source_id,
			"extractor_profile_id": self.extractor_profile_id,
			"owner": self.owner,
			"mode": self.mode,
			"schedule_policy_attached": self.schedule_policy_attached,
			"pipeline_handoff_required": self.pipeline_handoff_required,
			"pipeline_target": self.pipeline_target,
			"enabled": self.enabled,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
		}


@dataclass
class HarvestRun:
	"""One deterministic harvest run and compliance scan summary."""

	id: str
	tenant_id: str
	job_id: str
	source_id: str
	extractor_profile_id: str
	requested_by: str
	status: str = "queued"
	records_extracted: int = 0
	error_count: int = 0
	dlp_status: str = "pending"
	dlp_violations: int = 0
	logs: list[str] = field(default_factory=list)
	started_at: datetime | None = None
	completed_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"job_id": self.job_id,
			"source_id": self.source_id,
			"extractor_profile_id": self.extractor_profile_id,
			"requested_by": self.requested_by,
			"status": self.status,
			"records_extracted": self.records_extracted,
			"error_count": self.error_count,
			"dlp_status": self.dlp_status,
			"dlp_violations": self.dlp_violations,
			"logs": list(self.logs),
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
		}


@dataclass
class HarvestResult:
	"""Result batch metadata produced by a harvest run."""

	id: str
	tenant_id: str
	run_id: str
	record_count: int
	schema_valid: bool
	retention_until: str
	storage_ref: str
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"run_id": self.run_id,
			"record_count": self.record_count,
			"schema_valid": self.schema_valid,
			"retention_until": self.retention_until,
			"storage_ref": self.storage_ref,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class PipelineHandoff:
	"""Pipeline handoff record for ETL/Data Platform integration."""

	id: str
	tenant_id: str
	result_id: str
	pipeline_target: str
	status: str = "queued"
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"result_id": self.result_id,
			"pipeline_target": self.pipeline_target,
			"status": self.status,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class HarvestAgent:
	"""AI harvest-agent registration with runtime, scope, and disclosure."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	policy_ref: str | None = None
	status: str = "active"
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"policy_ref": self.policy_ref,
			"status": self.status,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class ScrpAuditEvent:
	"""Audit trail entry for data harvesting operations."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "info"
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"actor": self.actor,
			"severity": self.severity,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


ScrpRecord = HarvestSource
