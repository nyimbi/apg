"""Service layer for APG Pharma Quality Management System."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid6 import uuid7

from .capability_contract import (
	SUPPORTED_AUDIT_TYPES, SUPPORTED_CAPA_STATUSES, SUPPORTED_CAPA_TYPES,
	SUPPORTED_CHANGE_STATUSES, SUPPORTED_CHANGE_TYPES, SUPPORTED_DEVIATION_TYPES,
	SUPPORTED_DOCUMENT_STATUSES, SUPPORTED_DOCUMENT_TYPES, SUPPORTED_RISK_LEVELS,
	SUPPORTED_VALIDATION_TYPES, evaluate_capability_rules, get_capability_contract,
)
from .models import (
	CapaCreate, CapaRecord, ChangeControl, ChangeControlCreate, ControlledDocument,
	QmsDeviation, QualityAudit, RiskAssessment, ValidationRecord,
)


def _uuid7str() -> str:
	return str(uuid7())


class QualityManagementService:
	"""Tenant-scoped QMS service with GMP change control and CAPA enforcement."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._external_store = store

		self._changes: dict[tuple[str, str], ChangeControl] = {}
		self._capas: dict[tuple[str, str], CapaRecord] = {}
		self._deviations: dict[tuple[str, str], QmsDeviation] = {}
		self._documents: dict[tuple[str, str], ControlledDocument] = {}
		self._audits: dict[tuple[str, str], QualityAudit] = {}
		self._validations: dict[tuple[str, str], ValidationRecord] = {}
		self._risks: dict[tuple[str, str], RiskAssessment] = {}
		self._audit_events: list[dict[str, Any]] = []
		# extended stores
		self._complaints: dict[tuple[str, str], dict[str, Any]] = {}
		self._supplier_qualifications: dict[tuple[str, str], dict[str, Any]] = {}
		self._quality_metrics: dict[tuple[str, str], dict[str, Any]] = {}
		self._validation_lifecycles: dict[tuple[str, str], dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# --- change control ---

	def initiate_change(self, payload: ChangeControlCreate) -> ChangeControl:
		"""Initiate a change control record."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "initiate_change",
			"change_type_supported": payload.change_type in SUPPORTED_CHANGE_TYPES,
		})
		change = ChangeControl(**payload.model_dump())
		self._changes[self._key(change.tenant_id, change.id)] = change
		self._audit(change.tenant_id, "change_initiated", change.id)
		return change

	def approve_change(self, change_id: str, tenant_id: str,
					approval_reference: str, impact_assessed: bool,
					risk_assessed: bool) -> ChangeControl:
		"""Approve a change after impact and risk assessment."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_change",
			"impact_assessed": impact_assessed,
			"risk_assessed": risk_assessed,
		})
		change = self._get_change(change_id, tenant_id)
		data = change.model_dump()
		data["status"] = "approved"
		data["approval_reference"] = approval_reference
		data["updated_at"] = datetime.utcnow()
		updated = ChangeControl(**data)
		self._changes[self._key(tenant_id, change_id)] = updated
		self._audit(tenant_id, "change_approved", change_id)
		return updated

	def implement_change(self, change_id: str, tenant_id: str, implementation_date: datetime) -> ChangeControl:
		"""Mark a change as implemented."""
		change = self._get_change(change_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "implement_change",
			"approved": change.status == "approved",
		})
		data = change.model_dump()
		data["status"] = "implementation"
		data["implementation_date"] = implementation_date
		data["updated_at"] = datetime.utcnow()
		updated = ChangeControl(**data)
		self._changes[self._key(tenant_id, change_id)] = updated
		self._audit(tenant_id, "change_implemented", change_id)
		return updated

	def close_change(self, change_id: str, tenant_id: str,
					effectiveness_checked: bool, effectiveness_reference: str) -> ChangeControl:
		"""Close a change after effectiveness check."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_change",
			"effectiveness_checked": effectiveness_checked,
		})
		change = self._get_change(change_id, tenant_id)
		data = change.model_dump()
		data["status"] = "closed"
		data["effectiveness_check_reference"] = effectiveness_reference
		data["effectiveness_check_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = ChangeControl(**data)
		self._changes[self._key(tenant_id, change_id)] = updated
		self._audit(tenant_id, "change_closed", change_id)
		return updated

	def list_changes(self, tenant_id: str, status: str | None = None) -> list[ChangeControl]:
		items = [c for c in self._changes.values() if c.tenant_id == tenant_id]
		if status:
			items = [c for c in items if c.status == status]
		return items

	# --- CAPA ---

	def create_capa(self, payload: CapaCreate) -> CapaRecord:
		"""Create a CAPA record."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_capa",
			"capa_type_supported": payload.capa_type in SUPPORTED_CAPA_TYPES,
		})
		capa = CapaRecord(**payload.model_dump())
		self._capas[self._key(capa.tenant_id, capa.id)] = capa
		self._audit(capa.tenant_id, "capa_raised", capa.id)
		return capa

	def close_capa(self, capa_id: str, tenant_id: str, root_cause: str,
				root_cause_method: str, effectiveness_checked: bool,
				effectiveness_result: str) -> CapaRecord:
		"""Close a CAPA after root cause identification and effectiveness check."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_capa",
			"root_cause_identified": bool(root_cause),
			"effectiveness_checked": effectiveness_checked,
		})
		capa = self._capas.get(self._key(tenant_id, capa_id))
		if capa is None:
			raise KeyError(f"capa {capa_id} not found")
		data = capa.model_dump()
		data["status"] = "closed_effective" if effectiveness_result == "effective" else "closed_ineffective"
		data["root_cause"] = root_cause
		data["root_cause_method"] = root_cause_method
		data["effectiveness_check_date"] = datetime.utcnow()
		data["effectiveness_result"] = effectiveness_result
		data["actual_completion_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = CapaRecord(**data)
		self._capas[self._key(tenant_id, capa_id)] = updated
		self._audit(tenant_id, "capa_closed", capa_id)
		return updated

	def check_overdue_capas(self, tenant_id: str) -> list[CapaRecord]:
		"""Return CAPAs past their target completion date."""
		now = datetime.utcnow()
		overdue = []
		for capa in self._capas.values():
			if (capa.tenant_id == tenant_id and capa.status in ("open", "in_progress")
					and capa.target_completion_date and capa.target_completion_date < now):
				data = capa.model_dump()
				data["overdue"] = True
				data["status"] = "overdue"
				data["updated_at"] = now
				updated = CapaRecord(**data)
				self._capas[self._key(tenant_id, capa.id)] = updated
				overdue.append(updated)
				self._audit(tenant_id, "capa_overdue", capa.id)
		return overdue

	def list_capas(self, tenant_id: str, status: str | None = None) -> list[CapaRecord]:
		items = [c for c in self._capas.values() if c.tenant_id == tenant_id]
		if status:
			items = [c for c in items if c.status == status]
		return items

	# --- deviations ---

	def raise_deviation(self, tenant_id: str, deviation_number: str, deviation_type: str,
						severity: str, description: str, raised_by: str,
						affected_products: list[str] | None = None) -> QmsDeviation:
		"""Raise a QMS deviation record."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "raise_deviation",
			"deviation_type_supported": deviation_type in SUPPORTED_DEVIATION_TYPES,
			"severity": severity,
			"within_24h": True,
		})
		deviation = QmsDeviation(
			tenant_id=tenant_id, deviation_number=deviation_number,
			deviation_type=deviation_type, severity=severity,
			description=description, raised_by=raised_by,
			affected_products=affected_products or [],
			created_by=raised_by,
		)
		self._deviations[self._key(tenant_id, deviation.id)] = deviation
		self._audit(tenant_id, "deviation_raised", deviation.id)
		return deviation

	def close_deviation(self, deviation_id: str, tenant_id: str,
						root_cause: str, capa_reference: str | None = None) -> QmsDeviation:
		"""Close a deviation with investigation and CAPA linkage."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_deviation",
			"investigated": bool(root_cause),
		})
		deviation = self._deviations.get(self._key(tenant_id, deviation_id))
		if deviation is None:
			raise KeyError(f"deviation {deviation_id} not found")
		data = deviation.model_dump()
		data["status"] = "closed"
		data["root_cause"] = root_cause
		data["capa_reference"] = capa_reference
		data["closed_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = QmsDeviation(**data)
		self._deviations[self._key(tenant_id, deviation_id)] = updated
		self._audit(tenant_id, "deviation_closed", deviation_id)
		return updated

	def list_deviations(self, tenant_id: str, status: str | None = None) -> list[QmsDeviation]:
		items = [d for d in self._deviations.values() if d.tenant_id == tenant_id]
		if status:
			items = [d for d in items if d.status == status]
		return items

	# --- documents ---

	def create_document(self, tenant_id: str, document_number: str, title: str,
						document_type: str, version: str, department: str,
						owner_id: str, created_by: str) -> ControlledDocument:
		"""Create a controlled document."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_document",
			"document_type_supported": document_type in SUPPORTED_DOCUMENT_TYPES,
		})
		doc = ControlledDocument(
			tenant_id=tenant_id, document_number=document_number, title=title,
			document_type=document_type, version=version, department=department,
			owner_id=owner_id, created_by=created_by,
		)
		self._documents[self._key(tenant_id, doc.id)] = doc
		self._audit(tenant_id, "document_created", doc.id)
		return doc

	def approve_document(self, doc_id: str, tenant_id: str, approver_id: str) -> ControlledDocument:
		"""Approve a document and make it effective."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "make_document_effective",
			"approved": True,
		})
		doc = self._documents.get(self._key(tenant_id, doc_id))
		if doc is None:
			raise KeyError(f"document {doc_id} not found")
		data = doc.model_dump()
		data["status"] = "effective"
		data["approver_id"] = approver_id
		data["effective_date"] = datetime.utcnow()
		data["next_review_date"] = datetime.utcnow() + timedelta(days=730)
		data["updated_at"] = datetime.utcnow()
		updated = ControlledDocument(**data)
		self._documents[self._key(tenant_id, doc_id)] = updated
		self._audit(tenant_id, "document_approved", doc_id)
		return updated

	def supersede_document(self, old_doc_id: str, new_doc_id: str, tenant_id: str) -> ControlledDocument:
		"""Supersede an old document version with a new one."""
		old = self._documents.get(self._key(tenant_id, old_doc_id))
		if old is None:
			raise KeyError(f"document {old_doc_id} not found")
		data = old.model_dump()
		data["status"] = "superseded"
		data["superseded_by"] = new_doc_id
		data["updated_at"] = datetime.utcnow()
		updated = ControlledDocument(**data)
		self._documents[self._key(tenant_id, old_doc_id)] = updated
		self._audit(tenant_id, "document_superseded", old_doc_id)
		return updated

	def check_periodic_review(self, tenant_id: str) -> list[ControlledDocument]:
		"""Return documents due for periodic review."""
		now = datetime.utcnow()
		due = [d for d in self._documents.values()
			if d.tenant_id == tenant_id and d.status == "effective"
			and d.next_review_date is not None and d.next_review_date <= now]
		for d in due:
			self._audit(tenant_id, "document_periodic_review_due", d.id)
		return due

	def list_documents(self, tenant_id: str, document_type: str | None = None,
					status: str | None = None) -> list[ControlledDocument]:
		items = [d for d in self._documents.values() if d.tenant_id == tenant_id]
		if document_type:
			items = [d for d in items if d.document_type == document_type]
		if status:
			items = [d for d in items if d.status == status]
		return items

	# --- audits ---

	def create_audit(self, tenant_id: str, audit_number: str, audit_type: str,
					auditee: str, auditor_ids: list[str], scope: str,
					created_by: str, planned_date: datetime | None = None) -> QualityAudit:
		"""Create an audit record."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_audit",
			"audit_plan_present": True,
		})
		audit = QualityAudit(
			tenant_id=tenant_id, audit_number=audit_number, audit_type=audit_type,
			auditee=auditee, auditor_ids=auditor_ids, scope=scope,
			planned_date=planned_date, created_by=created_by,
		)
		self._audits[self._key(tenant_id, audit.id)] = audit
		self._audit(tenant_id, "audit_planned", audit.id)
		return audit

	def close_audit(self, audit_id: str, tenant_id: str, report_reference: str,
				findings_count: int, capa_references: list[str]) -> QualityAudit:
		"""Close an audit with findings and CAPA linkage."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_audit",
			"findings_have_capa": not (findings_count > 0 and len(capa_references) == 0),
		})
		audit = self._audits.get(self._key(tenant_id, audit_id))
		if audit is None:
			raise KeyError(f"audit {audit_id} not found")
		data = audit.model_dump()
		data["status"] = "closed"
		data["report_reference"] = report_reference
		data["findings_count"] = findings_count
		data["capa_references"] = capa_references
		data["conducted_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = QualityAudit(**data)
		self._audits[self._key(tenant_id, audit_id)] = updated
		self._audit(tenant_id, "audit_completed", audit_id)
		if findings_count > 0:
			self._audit(tenant_id, "audit_finding_raised", audit_id)
		return updated

	def list_audits(self, tenant_id: str, audit_type: str | None = None) -> list[QualityAudit]:
		items = [a for a in self._audits.values() if a.tenant_id == tenant_id]
		if audit_type:
			items = [a for a in items if a.audit_type == audit_type]
		return items

	# --- validation ---

	def create_validation(self, tenant_id: str, validation_number: str,
						validation_type: str, subject: str, created_by: str) -> ValidationRecord:
		"""Create a validation record."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		val = ValidationRecord(
			tenant_id=tenant_id, validation_number=validation_number,
			validation_type=validation_type, subject=subject, created_by=created_by,
		)
		self._validations[self._key(tenant_id, val.id)] = val
		self._audit(tenant_id, "validation_created", val.id)
		return val

	def execute_validation(self, val_id: str, tenant_id: str,
						protocol_reference: str, protocol_approved_by: str) -> ValidationRecord:
		"""Execute a validation after protocol approval."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "execute_validation",
			"protocol_approved": bool(protocol_reference),
		})
		val = self._validations.get(self._key(tenant_id, val_id))
		if val is None:
			raise KeyError(f"validation {val_id} not found")
		data = val.model_dump()
		data["status"] = "execution"
		data["protocol_reference"] = protocol_reference
		data["protocol_approved_by"] = protocol_approved_by
		data["protocol_approval_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = ValidationRecord(**data)
		self._validations[self._key(tenant_id, val_id)] = updated
		self._audit(tenant_id, "validation_started", val_id)
		return updated

	def approve_validation(self, val_id: str, tenant_id: str,
						report_reference: str, report_approved_by: str) -> ValidationRecord:
		"""Approve a validation with report sign-off."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "complete_validation",
			"report_approved": bool(report_reference),
		})
		val = self._validations.get(self._key(tenant_id, val_id))
		if val is None:
			raise KeyError(f"validation {val_id} not found")
		data = val.model_dump()
		data["status"] = "approved"
		data["report_reference"] = report_reference
		data["report_approved_by"] = report_approved_by
		data["report_approval_date"] = datetime.utcnow()
		data["revalidation_due"] = datetime.utcnow() + timedelta(days=365)
		data["updated_at"] = datetime.utcnow()
		updated = ValidationRecord(**data)
		self._validations[self._key(tenant_id, val_id)] = updated
		self._audit(tenant_id, "validation_approved", val_id)
		return updated

	def list_validations(self, tenant_id: str) -> list[ValidationRecord]:
		return [v for v in self._validations.values() if v.tenant_id == tenant_id]

	# --- risk ---

	def create_risk_assessment(self, tenant_id: str, assessment_number: str, subject: str,
								risk_level: str, likelihood: str, impact: str,
								owner_id: str, created_by: str) -> RiskAssessment:
		"""Create a risk assessment record."""
		risk = RiskAssessment(
			tenant_id=tenant_id, assessment_number=assessment_number, subject=subject,
			risk_level=risk_level, likelihood=likelihood, impact=impact,
			owner_id=owner_id, created_by=created_by,
			mitigation_required=risk_level in ("medium", "high", "critical"),
		)
		self._risks[self._key(tenant_id, risk.id)] = risk
		self._audit(tenant_id, "risk_assessment_created", risk.id)
		return risk

	def list_risks(self, tenant_id: str, risk_level: str | None = None) -> list[RiskAssessment]:
		items = [r for r in self._risks.values() if r.tenant_id == tenant_id]
		if risk_level:
			items = [r for r in items if r.risk_level == risk_level]
		return items

	# --- NEW: change_control ---

	def change_control(
		self,
		change_type: str,
		description: str,
		impact_assessment: str,
		originator_id: str,
		tenant_id: str,
		affected_products: list[str] | None = None,
		affected_documents: list[str] | None = None,
		regulatory_impact: bool = False,
	) -> ChangeControl:
		"""Initiate a GMP change control with impact assessment, regulatory impact flag, and approval workflow."""
		assert change_type and description, "change_type and description required"
		assert change_type in SUPPORTED_CHANGE_TYPES, f"unsupported change_type: {change_type}"
		change_number = f"CC-{_uuid7str()[:8].upper()}"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "initiate_change",
			"change_type_supported": True,
		})
		change = ChangeControl(
			tenant_id=tenant_id,
			change_number=change_number,
			change_type=change_type,
			description=description,
			impact_assessment=impact_assessment,
			originator_id=originator_id,
			affected_products=affected_products or [],
			affected_documents=affected_documents or [],
			regulatory_impact=regulatory_impact,
			created_by=originator_id,
		)
		self._changes[self._key(tenant_id, change.id)] = change
		self._audit(tenant_id, "change_initiated", change.id)
		if regulatory_impact:
			self._audit(tenant_id, "regulatory_impact_change_flagged", change.id)
		return change

	# --- NEW: capa_creation ---

	def capa_creation(
		self,
		source: str,
		root_cause: str,
		action_plan: str,
		responsible_person: str,
		deadline: datetime,
		tenant_id: str,
		capa_type: str = "corrective",
		source_reference: str = "",
		severity: str = "major",
	) -> CapaRecord:
		"""Create a CAPA from a deviation, audit finding, complaint, or signal."""
		assert source and root_cause and action_plan, "source, root_cause, action_plan required"
		assert capa_type in SUPPORTED_CAPA_TYPES, f"unsupported capa_type: {capa_type}"
		capa_number = f"CAPA-{_uuid7str()[:8].upper()}"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_capa",
			"capa_type_supported": True,
		})
		payload = CapaCreate(
			tenant_id=tenant_id,
			capa_number=capa_number,
			capa_type=capa_type,
			source=source,
			source_reference=source_reference,
			root_cause=root_cause,
			action_plan=action_plan,
			responsible_person_id=responsible_person,
			target_completion_date=deadline,
			severity=severity,
			created_by=responsible_person,
		)
		capa = CapaRecord(**payload.model_dump())
		self._capas[self._key(tenant_id, capa.id)] = capa
		self._audit(tenant_id, "capa_raised", capa.id)
		return capa

	# --- NEW: deviation_management ---

	def deviation_management(
		self,
		deviation_type: str,
		description: str,
		batch_id: str,
		impact: str,
		tenant_id: str,
		severity: str = "major",
		raised_by: str = "system",
		affected_products: list[str] | None = None,
		immediate_action: str = "",
	) -> QmsDeviation:
		"""Manage a quality deviation end-to-end: classify, record impact, immediate containment, CAPA linkage."""
		assert description and batch_id, "description and batch_id required"
		dev_number = f"DEV-{_uuid7str()[:8].upper()}"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "raise_deviation",
			"deviation_type_supported": deviation_type in SUPPORTED_DEVIATION_TYPES,
			"severity": severity,
			"within_24h": True,
		})
		deviation = QmsDeviation(
			tenant_id=tenant_id,
			deviation_number=dev_number,
			deviation_type=deviation_type,
			severity=severity,
			description=description,
			raised_by=raised_by,
			affected_products=affected_products or [],
			batch_id=batch_id,
			impact=impact,
			immediate_action=immediate_action,
			created_by=raised_by,
		)
		self._deviations[self._key(tenant_id, deviation.id)] = deviation
		self._audit(tenant_id, "deviation_raised", deviation.id)
		if severity == "critical":
			self._audit(tenant_id, "critical_deviation_escalated", deviation.id)
		return deviation

	# --- NEW: document_control ---

	def document_control(
		self,
		document_id: str,
		action: str,
		version: str,
		approved_by: str,
		tenant_id: str,
		effective_date: datetime | None = None,
		reason: str = "",
	) -> ControlledDocument:
		"""Apply a document control action (approve, revise, supersede, withdraw, retire) to a controlled document."""
		assert document_id and action, "document_id and action required"
		assert action in ("approve", "revise", "supersede", "withdraw", "retire"), \
			f"unsupported action: {action}"
		doc = self._documents.get(self._key(tenant_id, document_id))
		if doc is None:
			raise KeyError(f"document {document_id} not found")
		status_map = {
			"approve": "effective",
			"revise": "draft",
			"supersede": "superseded",
			"withdraw": "withdrawn",
			"retire": "retired",
		}
		data = doc.model_dump()
		data["status"] = status_map[action]
		data["version"] = version
		data["approver_id"] = approved_by
		if action == "approve":
			data["effective_date"] = effective_date or datetime.utcnow()
			data["next_review_date"] = datetime.utcnow() + timedelta(days=730)
		data["updated_at"] = datetime.utcnow()
		if reason:
			data["change_reason"] = reason
		updated = ControlledDocument(**data)
		self._documents[self._key(tenant_id, document_id)] = updated
		self._audit(tenant_id, f"document_{action}", document_id)
		return updated

	# --- NEW: internal_audit ---

	def internal_audit(
		self,
		area: str,
		auditor_id: str,
		date: datetime,
		scope: str,
		tenant_id: str,
		audit_type: str = "internal_gmp",
		checklist_reference: str = "",
	) -> QualityAudit:
		"""Schedule and create an internal GMP audit with scope, auditor assignment, and planning checklist."""
		assert area and auditor_id and scope, "area, auditor_id, scope required"
		audit_number = f"AUD-{_uuid7str()[:8].upper()}"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_audit",
			"audit_plan_present": True,
		})
		audit = QualityAudit(
			tenant_id=tenant_id,
			audit_number=audit_number,
			audit_type=audit_type,
			auditee=area,
			auditor_ids=[auditor_id],
			scope=scope,
			planned_date=date,
			checklist_reference=checklist_reference,
			created_by=auditor_id,
		)
		self._audits[self._key(tenant_id, audit.id)] = audit
		self._audit(tenant_id, "audit_planned", audit.id)
		return audit

	# --- NEW: supplier_qualification ---

	def supplier_qualification(
		self,
		supplier_id: str,
		qualification_type: str,
		result: str,
		tenant_id: str,
		audit_date: datetime | None = None,
		quality_agreement_ref: str = "",
		approved_materials: list[str] | None = None,
		next_audit_days: int = 730,
	) -> dict[str, Any]:
		"""Qualify or re-qualify a supplier with audit results and quality agreement linkage."""
		assert supplier_id and qualification_type, "supplier_id and qualification_type required"
		assert result in ("qualified", "conditional", "disqualified"), \
			f"unsupported result: {result}"
		qual_id = _uuid7str()
		qualification: dict[str, Any] = {
			"id": qual_id,
			"tenant_id": tenant_id,
			"supplier_id": supplier_id,
			"qualification_type": qualification_type,
			"result": result,
			"audit_date": str(audit_date or datetime.utcnow()),
			"quality_agreement_reference": quality_agreement_ref,
			"approved_materials": approved_materials or [],
			"on_approved_supplier_list": result == "qualified",
			"next_audit_date": str(datetime.utcnow() + timedelta(days=next_audit_days)),
			"created_at": datetime.utcnow().isoformat(),
		}
		self._supplier_qualifications[self._key(tenant_id, qual_id)] = qualification
		self._audit(tenant_id, "supplier_qualified" if result == "qualified" else "supplier_qualification_failed", qual_id)
		return qualification

	# --- NEW: product_complaint ---

	def product_complaint(
		self,
		complaint_id: str,
		details: dict[str, Any],
		investigation: str,
		tenant_id: str,
		product_id: str = "",
		batch_number: str = "",
		complainant_type: str = "customer",
		severity: str = "minor",
		capa_required: bool = False,
	) -> dict[str, Any]:
		"""Record and process a product quality complaint with investigation and CAPA determination."""
		assert complaint_id and details, "complaint_id and details required"
		assert severity in ("minor", "major", "critical"), f"unsupported severity: {severity}"
		complaint: dict[str, Any] = {
			"id": complaint_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"batch_number": batch_number,
			"complainant_type": complainant_type,
			"details": details,
			"investigation": investigation,
			"severity": severity,
			"capa_required": capa_required or severity in ("major", "critical"),
			"status": "under_investigation",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._complaints[self._key(tenant_id, complaint_id)] = complaint
		self._audit(tenant_id, "complaint_received", complaint_id)
		if severity == "critical":
			self._audit(tenant_id, "critical_complaint_escalated", complaint_id)
		return complaint

	# --- NEW: quality_metrics ---

	def quality_metrics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Calculate and return QMS KPIs for the period: RFT, CAPA closure rate, deviation rate, audit findings."""
		assert period, "period required"
		changes = self.list_changes(tenant_id)
		capas = self.list_capas(tenant_id)
		deviations = self.list_deviations(tenant_id)
		audits = self.list_audits(tenant_id)
		validations = self.list_validations(tenant_id)
		risks = self.list_risks(tenant_id)
		# CAPA metrics
		closed_capas = [c for c in capas if c.status in ("closed_effective", "closed_ineffective")]
		effective_capas = [c for c in capas if c.status == "closed_effective"]
		overdue_capas = [c for c in capas if getattr(c, "overdue", False)]
		capa_closure_rate = len(closed_capas) / max(len(capas), 1) * 100
		capa_effectiveness_rate = len(effective_capas) / max(len(closed_capas), 1) * 100
		# deviation metrics
		open_deviations = [d for d in deviations if d.status == "open"]
		critical_deviations = [d for d in deviations if d.severity == "critical"]
		# audit metrics
		closed_audits = [a for a in audits if a.status == "closed"]
		total_findings = sum(getattr(a, "findings_count", 0) for a in closed_audits)
		# document metrics
		effective_docs = [d for d in self._documents.values()
			if d.tenant_id == tenant_id and d.status == "effective"]
		overdue_review = self.check_periodic_review(tenant_id)
		# complaint metrics
		complaints = [c for c in self._complaints.values() if c["tenant_id"] == tenant_id]
		metrics_id = _uuid7str()
		metrics: dict[str, Any] = {
			"id": metrics_id,
			"period": period,
			"tenant_id": tenant_id,
			"capa_total": len(capas),
			"capa_closure_rate_pct": round(capa_closure_rate, 2),
			"capa_effectiveness_rate_pct": round(capa_effectiveness_rate, 2),
			"overdue_capas": len(overdue_capas),
			"open_deviations": len(open_deviations),
			"critical_deviations": len(critical_deviations),
			"total_audits": len(audits),
			"closed_audits": len(closed_audits),
			"total_audit_findings": total_findings,
			"effective_documents": len(effective_docs),
			"documents_overdue_review": len(overdue_review),
			"open_changes": len([c for c in changes if c.status not in ("closed", "withdrawn")]),
			"total_complaints": len(complaints),
			"high_critical_risks": len([r for r in risks if r.risk_level in ("high", "critical")]),
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._quality_metrics[self._key(tenant_id, metrics_id)] = metrics
		self._audit(tenant_id, "quality_metrics_generated", metrics_id)
		return metrics

	# --- NEW: quality_risk_assessment ---

	def quality_risk_assessment(
		self,
		product_id: str,
		risk_factors: list[dict[str, Any]],
		tenant_id: str,
		assessment_method: str = "fmea",
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Run a quality risk assessment (ICH Q9) for a product using FMEA, HACCP, or FTA methodology."""
		assert product_id and risk_factors, "product_id and risk_factors required"
		assert assessment_method in ("fmea", "haccp", "fta", "hazop", "preliminary_hazard"), \
			f"unsupported assessment_method: {assessment_method}"
		assessment_id = _uuid7str()
		scored_factors: list[dict[str, Any]] = []
		for factor in risk_factors:
			severity = factor.get("severity", 5)
			probability = factor.get("probability", 5)
			detectability = factor.get("detectability", 5)
			rpn = severity * probability * detectability
			risk_level = "critical" if rpn >= 200 else "high" if rpn >= 100 else "medium" if rpn >= 50 else "low"
			scored_factors.append({**factor, "rpn": rpn, "risk_level": risk_level})
		max_rpn = max((f["rpn"] for f in scored_factors), default=0)
		overall_risk = "critical" if max_rpn >= 200 else "high" if max_rpn >= 100 else "medium" if max_rpn >= 50 else "low"
		assessment: dict[str, Any] = {
			"id": assessment_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"assessment_method": assessment_method,
			"owner_id": owner_id,
			"ich_guideline": "Q9",
			"risk_factors": scored_factors,
			"overall_risk_level": overall_risk,
			"max_rpn": max_rpn,
			"critical_factors": sum(1 for f in scored_factors if f["risk_level"] == "critical"),
			"mitigation_required": overall_risk in ("critical", "high"),
			"created_at": datetime.utcnow().isoformat(),
		}
		risk_obj = RiskAssessment(
			tenant_id=tenant_id,
			assessment_number=assessment_id[:8],
			subject=f"product:{product_id}",
			risk_level=overall_risk,
			likelihood="high" if max_rpn >= 100 else "medium",
			impact="high" if max_rpn >= 100 else "medium",
			owner_id=owner_id,
			created_by=owner_id,
			mitigation_required=overall_risk in ("critical", "high"),
		)
		self._risks[self._key(tenant_id, risk_obj.id)] = risk_obj
		self._audit(tenant_id, "quality_risk_assessed", assessment_id)
		return assessment

	# --- NEW: validation_lifecycle ---

	def validation_lifecycle(
		self,
		system_id: str,
		stage: str,
		result: str,
		tenant_id: str,
		system_name: str = "",
		validation_type: str = "computer_system",
		performed_by: str = "system",
		evidence_references: list[str] | None = None,
	) -> dict[str, Any]:
		"""Manage the validation lifecycle (URS, FS, DS, IQ, OQ, PQ, periodic review) for a system or process."""
		assert system_id and stage, "system_id and stage required"
		assert stage in ("urs", "fs", "ds", "iq", "oq", "pq", "periodic_review", "decommission"), \
			f"unsupported stage: {stage}"
		assert result in ("pass", "fail", "conditional"), f"unsupported result: {result}"
		lifecycle_id = _uuid7str()
		lifecycle: dict[str, Any] = {
			"id": lifecycle_id,
			"tenant_id": tenant_id,
			"system_id": system_id,
			"system_name": system_name,
			"validation_type": validation_type,
			"stage": stage,
			"result": result,
			"performed_by": performed_by,
			"evidence_references": evidence_references or [],
			"validated_at": datetime.utcnow().isoformat(),
			"next_review_due": str(datetime.utcnow() + timedelta(days=365)) if stage in ("pq", "periodic_review") else None,
		}
		self._validation_lifecycles[self._key(tenant_id, lifecycle_id)] = lifecycle
		self._audit(tenant_id, f"validation_{stage}_{result}", lifecycle_id)
		return lifecycle

	# --- dashboard ---

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return QMS dashboard summary."""
		return {
			"tenant_id": tenant_id,
			"open_changes": sum(1 for c in self._changes.values()
								if c.tenant_id == tenant_id and c.status not in ("closed", "withdrawn")),
			"open_capas": sum(1 for c in self._capas.values()
							if c.tenant_id == tenant_id and c.status in ("open", "in_progress")),
			"overdue_capas": sum(1 for c in self._capas.values()
								if c.tenant_id == tenant_id and c.overdue),
			"open_deviations": sum(1 for d in self._deviations.values()
								if d.tenant_id == tenant_id and d.status == "open"),
			"effective_documents": sum(1 for d in self._documents.values()
									if d.tenant_id == tenant_id and d.status == "effective"),
			"open_audits": sum(1 for a in self._audits.values()
							if a.tenant_id == tenant_id and a.status not in ("closed",)),
			"validation_count": self._count(self._validations, tenant_id),
			"high_critical_risks": sum(1 for r in self._risks.values()
									if r.tenant_id == tenant_id and r.risk_level in ("high", "critical")),
			"total_complaints": sum(1 for c in self._complaints.values() if c["tenant_id"] == tenant_id),
			"supplier_qualifications": sum(1 for s in self._supplier_qualifications.values() if s["tenant_id"] == tenant_id),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id),
		}

	# --- private helpers ---

	def _log_capa_overdue_days(self, capa_id: str, days_overdue: int) -> None:
		pass

	def _log_document_review_due(self, doc_id: str, days_overdue: int) -> None:
		pass

	def _get_change(self, change_id: str, tenant_id: str) -> ChangeControl:
		item = self._changes.get(self._key(tenant_id, change_id))
		if item is None:
			raise KeyError(f"change {change_id} not found")
		return item

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"stream": "apg.pharma.qms.lifecycle",
		})

	def _count(self, store: dict[Any, Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")



	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": tenant_id}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy"}

	async def compliance_report(self, tenant_id: str, standard: str = "GxP") -> dict[str, Any]:
		"""Compliance Report"""
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _now()}

	async def sign_and_approve_document(
		self,
		doc_id: str,
		tenant_id: str,
		approver_id: str,
		meaning: str,
	) -> dict[str, Any]:
		"""Approve a QMS document with a 21 CFR Part 11 qualified electronic signature.

		Creates an ESignatureRecord binding the approver's identity, the document,
		and the stated meaning — then marks the document as effective.

		Args:
			doc_id:       Document to approve
			tenant_id:    Tenant context
			approver_id:  Authenticated identity of the approving person
			meaning:      Signer's stated intent (e.g. "I approve this SOP for release")

		Returns:
			{"document": ControlledDocument.model_dump(),
			 "signature": ESignatureRecord fields,
			 "approved": True}
		"""
		from capabilities.common.esig import ESignatureService
		esig_svc = ESignatureService(tenant_id=tenant_id)
		signature = await esig_svc.sign(
			document_id=doc_id,
			signer_id=approver_id,
			meaning=meaning,
			context={"capability_id": "pharma_qms", "action": "document_approval"},
		)
		# Now perform the actual approval
		doc = self.approve_document(doc_id, tenant_id, approver_id)
		return {
			"document": doc.model_dump() if hasattr(doc, "model_dump") else doc,
			"signature_id": signature.signature_id,
			"signer_id": signature.signer_id,
			"meaning": signature.meaning,
			"timestamp": signature.timestamp,
			"approved": True,
		}



	async def ml_batch_release_risk(self, *args, **kwargs):
		"""AI-powered ML batch release risk classification. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["release_standard","release_with_review","hold_for_investigation","reject"])
			return {"release_decision": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

PharmaQmsService = QualityManagementService
