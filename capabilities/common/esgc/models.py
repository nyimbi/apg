"""Domain models for APG ESG/Carbon Tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmissionsInventory:
	"""Tenant emissions inventory with accountable owner and reporting boundary."""

	id: str
	tenant_id: str
	organization: str
	owner: str
	reporting_year: int
	boundary_ref: str
	geospatial_boundary: str
	compliance_framework: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"organization": self.organization,
			"owner": self.owner,
			"reporting_year": self.reporting_year,
			"boundary_ref": self.boundary_ref,
			"geospatial_boundary": self.geospatial_boundary,
			"compliance_framework": self.compliance_framework,
			"status": self.status,
		}


@dataclass(frozen=True)
class EmissionFactor:
	"""Versioned emissions factor with source and evidence metadata."""

	id: str
	tenant_id: str
	name: str
	scope: str
	unit: str
	co2e_per_unit: float
	source: str
	source_evidence: str
	version: str
	approved_source: bool
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"scope": self.scope,
			"unit": self.unit,
			"co2e_per_unit": self.co2e_per_unit,
			"source": self.source,
			"source_evidence": self.source_evidence,
			"version": self.version,
			"approved_source": self.approved_source,
			"status": self.status,
		}


@dataclass(frozen=True)
class EmissionActivity:
	"""Measured activity data converted into carbon dioxide equivalent."""

	id: str
	tenant_id: str
	inventory_id: str
	factor_id: str
	activity_type: str
	scope: str
	quantity: float
	unit: str
	co2e_tonnes: float
	evidence_ref: str
	anomaly_detected: bool
	anomaly_review_recorded: bool
	status: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"inventory_id": self.inventory_id,
			"factor_id": self.factor_id,
			"activity_type": self.activity_type,
			"scope": self.scope,
			"quantity": self.quantity,
			"unit": self.unit,
			"co2e_tonnes": self.co2e_tonnes,
			"evidence_ref": self.evidence_ref,
			"anomaly_detected": self.anomaly_detected,
			"anomaly_review_recorded": self.anomaly_review_recorded,
			"status": self.status,
		}


@dataclass(frozen=True)
class SustainabilityReport:
	"""Approved ESG/carbon report with compliance mapping and evidence state."""

	id: str
	tenant_id: str
	inventory_id: str
	report_type: str
	period: str
	total_co2e_tonnes: float
	compliance_mapping: str
	audit_evidence_ref: str
	approved_by: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"inventory_id": self.inventory_id,
			"report_type": self.report_type,
			"period": self.period,
			"total_co2e_tonnes": self.total_co2e_tonnes,
			"compliance_mapping": self.compliance_mapping,
			"audit_evidence_ref": self.audit_evidence_ref,
			"approved_by": self.approved_by,
			"status": self.status,
		}


@dataclass(frozen=True)
class ReductionTarget:
	"""Carbon reduction target compared against baseline and current inventory."""

	id: str
	tenant_id: str
	inventory_id: str
	name: str
	baseline_year: int
	target_year: int
	baseline_co2e_tonnes: float
	target_reduction_percent: float
	current_co2e_tonnes: float
	progress_percent: float
	status: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"inventory_id": self.inventory_id,
			"name": self.name,
			"baseline_year": self.baseline_year,
			"target_year": self.target_year,
			"baseline_co2e_tonnes": self.baseline_co2e_tonnes,
			"target_reduction_percent": self.target_reduction_percent,
			"current_co2e_tonnes": self.current_co2e_tonnes,
			"progress_percent": self.progress_percent,
			"status": self.status,
		}


@dataclass(frozen=True)
class EsgcAgent:
	"""Registered AI agent allowed to assist ESG and carbon operations."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"

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
			"status": self.status,
		}


@dataclass(frozen=True)
class EsgcAuditEvent:
	"""Governance event emitted by ESG/carbon operations."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
		}
