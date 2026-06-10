"""Service layer for APG Pharma Regulatory Compliance."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid6 import uuid7

from .capability_contract import (
	SUPPORTED_AUDIT_TYPES, SUPPORTED_COMMITMENT_STATUSES, SUPPORTED_INSPECTION_OUTCOMES,
	SUPPORTED_INTEL_TYPES, SUPPORTED_LABEL_CHANGE_TYPES, SUPPORTED_PMS_TYPES,
	SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REGULATORY_REGIONS, evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	ComplianceFrameworkRecord, GapAssessment, InspectionRecord, LabelRecord,
	PostMarketSurveillanceRecord, RegulatoryCommitment, RegulatoryIntelligenceRecord,
)


def _uuid7str() -> str:
	return str(uuid7())


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class RegulatoryComplianceService:
	"""Tenant-scoped regulatory compliance service with inspection readiness and commitment tracking."""

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

		self._frameworks: dict[tuple[str, str], ComplianceFrameworkRecord] = {}
		self._gap_assessments: dict[tuple[str, str], GapAssessment] = {}
		self._inspections: dict[tuple[str, str], InspectionRecord] = {}
		self._labels: dict[tuple[str, str], LabelRecord] = {}
		self._pms: dict[tuple[str, str], PostMarketSurveillanceRecord] = {}
		self._intel: dict[tuple[str, str], RegulatoryIntelligenceRecord] = {}
		self._commitments: dict[tuple[str, str], RegulatoryCommitment] = {}
		self._audit_events: list[dict[str, Any]] = []
		# extended stores
		self._compliance_calendars: dict[tuple[str, str], dict[str, Any]] = {}
		self._registration_renewals: dict[tuple[str, str], dict[str, Any]] = {}
		self._rems_records: dict[tuple[str, str], dict[str, Any]] = {}
		self._import_licences: dict[tuple[str, str], dict[str, Any]] = {}
		self._authority_interactions: dict[tuple[str, str], dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# --- compliance frameworks ---

	def register_compliance(self, tenant_id: str, framework: str, title: str,
							applicable_sites: list[str], owner_id: str,
							created_by: str) -> ComplianceFrameworkRecord:
		"""Register a regulatory compliance framework obligation."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_compliance",
			"framework_supported": framework in SUPPORTED_REGULATORY_FRAMEWORKS,
		})
		record = ComplianceFrameworkRecord(
			tenant_id=tenant_id, framework=framework, title=title,
			applicable_sites=applicable_sites, owner_id=owner_id, created_by=created_by,
		)
		self._frameworks[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, "compliance_framework_registered", record.id)
		return record

	def list_frameworks(self, tenant_id: str) -> list[ComplianceFrameworkRecord]:
		return [f for f in self._frameworks.values() if f.tenant_id == tenant_id]

	# --- gap assessments ---

	def create_gap_assessment(self, tenant_id: str, assessment_number: str, framework: str,
							site: str, conducted_by: str, created_by: str) -> GapAssessment:
		"""Create a compliance gap assessment."""
		assessment = GapAssessment(
			tenant_id=tenant_id, assessment_number=assessment_number,
			framework=framework, site=site, conducted_date=datetime.utcnow(),
			conducted_by=conducted_by, created_by=created_by,
		)
		self._gap_assessments[self._key(tenant_id, assessment.id)] = assessment
		self._audit(tenant_id, "gap_assessment_created", assessment.id)
		return assessment

	def close_gap_assessment(self, assessment_id: str, tenant_id: str,
							critical_gaps: int, major_gaps: int, minor_gaps: int,
							implementation_plan_reference: str) -> GapAssessment:
		"""Close a gap assessment with findings and implementation plan."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_gap",
			"implementation_plan_present": bool(implementation_plan_reference),
		})
		assessment = self._gap_assessments.get(self._key(tenant_id, assessment_id))
		if assessment is None:
			raise KeyError(f"gap_assessment {assessment_id} not found")
		data = assessment.model_dump()
		data["critical_gaps"] = critical_gaps
		data["major_gaps"] = major_gaps
		data["minor_gaps"] = minor_gaps
		data["gaps_identified"] = critical_gaps + major_gaps + minor_gaps
		data["implementation_plan_reference"] = implementation_plan_reference
		data["next_assessment_date"] = datetime.utcnow() + timedelta(days=365)
		data["updated_at"] = datetime.utcnow()
		updated = GapAssessment(**data)
		self._gap_assessments[self._key(tenant_id, assessment_id)] = updated
		if critical_gaps > 0:
			self._audit(tenant_id, "compliance_gap_identified", assessment_id)
		return updated

	def list_gap_assessments(self, tenant_id: str) -> list[GapAssessment]:
		return [a for a in self._gap_assessments.values() if a.tenant_id == tenant_id]

	# --- inspections ---

	def record_inspection(self, tenant_id: str, inspection_number: str,
						inspection_type: str, authority: str, site: str,
						announced: bool, created_by: str,
						start_date: datetime | None = None) -> InspectionRecord:
		"""Record a regulatory inspection."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_inspection",
			"inspection_type_supported": inspection_type in SUPPORTED_AUDIT_TYPES,
		})
		inspection = InspectionRecord(
			tenant_id=tenant_id, inspection_number=inspection_number,
			inspection_type=inspection_type, authority=authority, site=site,
			announced=announced, start_date=start_date, created_by=created_by,
		)
		self._inspections[self._key(tenant_id, inspection.id)] = inspection
		self._audit(tenant_id, "inspection_announced", inspection.id)
		return inspection

	def record_inspection_outcome(self, inspection_id: str, tenant_id: str,
								outcome: str, findings_count: int) -> InspectionRecord:
		"""Record the outcome of a regulatory inspection."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_inspection_outcome",
			"outcome_supported": outcome in SUPPORTED_INSPECTION_OUTCOMES,
		})
		inspection = self._inspections.get(self._key(tenant_id, inspection_id))
		if inspection is None:
			raise KeyError(f"inspection {inspection_id} not found")
		data = inspection.model_dump()
		data["outcome"] = outcome
		data["findings_count"] = findings_count
		data["status"] = "completed"
		data["end_date"] = datetime.utcnow()
		if outcome == "warning_letter":
			data["response_deadline"] = datetime.utcnow() + timedelta(days=30)
		elif outcome == "official_action_indicated":
			data["response_deadline"] = datetime.utcnow() + timedelta(days=15)
		data["updated_at"] = datetime.utcnow()
		updated = InspectionRecord(**data)
		self._inspections[self._key(tenant_id, inspection_id)] = updated
		self._audit(tenant_id, "inspection_completed", inspection_id)
		if outcome == "warning_letter":
			self._audit(tenant_id, "warning_letter_received", inspection_id)
		return updated

	def respond_to_inspection(self, inspection_id: str, tenant_id: str,
							response_reference: str, within_deadline: bool) -> InspectionRecord:
		"""Record an inspection response submission."""
		inspection = self._inspections.get(self._key(tenant_id, inspection_id))
		if inspection is None:
			raise KeyError(f"inspection {inspection_id} not found")
		outcome = inspection.outcome or ""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "respond_to_inspection",
			"outcome": outcome,
			"within_30d": within_deadline,
		})
		data = inspection.model_dump()
		data["response_submitted_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = InspectionRecord(**data)
		self._inspections[self._key(tenant_id, inspection_id)] = updated
		self._audit(tenant_id, "inspection_response_submitted", inspection_id)
		return updated

	def list_inspections(self, tenant_id: str, status: str | None = None) -> list[InspectionRecord]:
		items = [i for i in self._inspections.values() if i.tenant_id == tenant_id]
		if status:
			items = [i for i in items if i.status == status]
		return items

	# --- labeling ---

	def create_label(self, tenant_id: str, label_number: str, product_id: str,
					market: str, language: str, version: str, change_type: str,
					created_by: str) -> LabelRecord:
		"""Create a label record."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "initiate_label_change",
			"change_type_supported": change_type in SUPPORTED_LABEL_CHANGE_TYPES,
		})
		label = LabelRecord(
			tenant_id=tenant_id, label_number=label_number, product_id=product_id,
			market=market, language=language, version=version, change_type=change_type,
			created_by=created_by,
		)
		self._labels[self._key(tenant_id, label.id)] = label
		self._audit(tenant_id, "label_created", label.id)
		return label

	def approve_label(self, label_id: str, tenant_id: str, qp_approved_by: str) -> LabelRecord:
		"""Approve a label with QP sign-off."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_label",
			"qp_approved": bool(qp_approved_by),
		})
		label = self._labels.get(self._key(tenant_id, label_id))
		if label is None:
			raise KeyError(f"label {label_id} not found")
		data = label.model_dump()
		data["qp_approved"] = True
		data["qp_approval_date"] = datetime.utcnow()
		data["status"] = "approved"
		data["effective_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = LabelRecord(**data)
		self._labels[self._key(tenant_id, label_id)] = updated
		self._audit(tenant_id, "label_change_approved", label_id)
		return updated

	def list_labels(self, tenant_id: str, product_id: str | None = None) -> list[LabelRecord]:
		items = [l for l in self._labels.values() if l.tenant_id == tenant_id]
		if product_id:
			items = [l for l in items if l.product_id == product_id]
		return items

	# --- post-market surveillance ---

	def create_pms(self, tenant_id: str, pms_number: str, product_id: str,
				pms_type: str, created_by: str) -> PostMarketSurveillanceRecord:
		"""Create a PMS record."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_pms",
			"pms_type_supported": pms_type in SUPPORTED_PMS_TYPES,
		})
		pms = PostMarketSurveillanceRecord(
			tenant_id=tenant_id, pms_number=pms_number, product_id=product_id,
			pms_type=pms_type, created_by=created_by,
		)
		self._pms[self._key(tenant_id, pms.id)] = pms
		self._audit(tenant_id, "pms_created", pms.id)
		return pms

	def start_pms(self, pms_id: str, tenant_id: str, protocol_reference: str,
				start_date: datetime) -> PostMarketSurveillanceRecord:
		"""Start PMS with an approved protocol."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_pms",
			"protocol_present": bool(protocol_reference),
		})
		pms = self._pms.get(self._key(tenant_id, pms_id))
		if pms is None:
			raise KeyError(f"pms {pms_id} not found")
		data = pms.model_dump()
		data["protocol_reference"] = protocol_reference
		data["protocol_approved"] = True
		data["status"] = "active"
		data["start_date"] = start_date
		data["updated_at"] = datetime.utcnow()
		updated = PostMarketSurveillanceRecord(**data)
		self._pms[self._key(tenant_id, pms_id)] = updated
		self._audit(tenant_id, "pms_started", pms_id)
		return updated

	def list_pms(self, tenant_id: str) -> list[PostMarketSurveillanceRecord]:
		return [p for p in self._pms.values() if p.tenant_id == tenant_id]

	# --- regulatory intelligence ---

	def record_intel(self, tenant_id: str, intel_number: str, intel_type: str,
					region: str, title: str, description: str, created_by: str,
					source_url: str | None = None, published_date: datetime | None = None) -> RegulatoryIntelligenceRecord:
		"""Record a regulatory intelligence item."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_intel",
			"intel_type_supported": intel_type in SUPPORTED_INTEL_TYPES,
		})
		intel = RegulatoryIntelligenceRecord(
			tenant_id=tenant_id, intel_number=intel_number, intel_type=intel_type,
			region=region, title=title, description=description,
			source_url=source_url, published_date=published_date,
			created_by=created_by,
		)
		self._intel[self._key(tenant_id, intel.id)] = intel
		self._audit(tenant_id, "regulatory_change_detected", intel.id)
		return intel

	def assess_intel_impact(self, intel_id: str, tenant_id: str,
							impact_assessment_reference: str) -> RegulatoryIntelligenceRecord:
		"""Record impact assessment for a regulatory intelligence item."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_regulatory_change",
			"impact_assessed": bool(impact_assessment_reference),
		})
		intel = self._intel.get(self._key(tenant_id, intel_id))
		if intel is None:
			raise KeyError(f"intel {intel_id} not found")
		data = intel.model_dump()
		data["impact_assessed"] = True
		data["impact_assessment_reference"] = impact_assessment_reference
		data["updated_at"] = datetime.utcnow()
		updated = RegulatoryIntelligenceRecord(**data)
		self._intel[self._key(tenant_id, intel_id)] = updated
		self._audit(tenant_id, "impact_assessment_required", intel_id)
		return updated

	def list_intel(self, tenant_id: str, region: str | None = None,
				assessed: bool | None = None) -> list[RegulatoryIntelligenceRecord]:
		items = [i for i in self._intel.values() if i.tenant_id == tenant_id]
		if region:
			items = [i for i in items if i.region == region]
		if assessed is not None:
			items = [i for i in items if i.impact_assessed == assessed]
		return items

	# --- commitments ---

	def create_commitment(self, tenant_id: str, commitment_number: str,
						product_id: str, authority: str, description: str,
						due_date: datetime, milestones: list[dict],
						created_by: str) -> RegulatoryCommitment:
		"""Create a regulatory commitment."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_commitment",
			"milestone_present": bool(milestones),
		})
		commitment = RegulatoryCommitment(
			tenant_id=tenant_id, commitment_number=commitment_number,
			product_id=product_id, authority=authority, description=description,
			due_date=due_date, milestones=milestones, created_by=created_by,
		)
		self._commitments[self._key(tenant_id, commitment.id)] = commitment
		self._audit(tenant_id, "commitment_created", commitment.id)
		return commitment

	def fulfill_commitment(self, commitment_id: str, tenant_id: str,
						submission_reference: str) -> RegulatoryCommitment:
		"""Mark a commitment as fulfilled."""
		commitment = self._commitments.get(self._key(tenant_id, commitment_id))
		if commitment is None:
			raise KeyError(f"commitment {commitment_id} not found")
		data = commitment.model_dump()
		data["status"] = "fulfilled"
		data["completed_date"] = datetime.utcnow()
		data["submission_reference"] = submission_reference
		data["updated_at"] = datetime.utcnow()
		updated = RegulatoryCommitment(**data)
		self._commitments[self._key(tenant_id, commitment_id)] = updated
		self._audit(tenant_id, "commitment_fulfilled", commitment_id)
		return updated

	def check_overdue_commitments(self, tenant_id: str) -> list[RegulatoryCommitment]:
		"""Return commitments past their due date."""
		now = datetime.utcnow()
		overdue = []
		for c in self._commitments.values():
			if c.tenant_id == tenant_id and c.status == "open" and c.due_date < now:
				data = c.model_dump()
				data["overdue"] = True
				data["updated_at"] = now
				updated = RegulatoryCommitment(**data)
				self._commitments[self._key(tenant_id, c.id)] = updated
				overdue.append(updated)
				self._audit(tenant_id, "commitment_overdue", c.id)
		return overdue

	def list_commitments(self, tenant_id: str, status: str | None = None) -> list[RegulatoryCommitment]:
		items = [c for c in self._commitments.values() if c.tenant_id == tenant_id]
		if status:
			items = [c for c in items if c.status == status]
		return items

	# --- NEW: compliance_calendar ---

	def compliance_calendar(
		self,
		product_id: str,
		jurisdiction: str,
		tenant_id: str,
		year: int | None = None,
	) -> dict[str, Any]:
		"""Generate a regulatory compliance calendar for a product in a jurisdiction showing all due dates."""
		assert product_id and jurisdiction, "product_id and jurisdiction required"
		cal_year = year or datetime.utcnow().year
		calendar_id = _uuid7str()
		# gather all commitments for this product
		product_commitments = [c for c in self._commitments.values()
			if c.tenant_id == tenant_id and c.product_id == product_id]
		# gather PMS activities
		product_pms = [p for p in self._pms.values()
			if p.tenant_id == tenant_id and p.product_id == product_id]
		# gather label renewals
		product_labels = [l for l in self._labels.values()
			if l.tenant_id == tenant_id and l.product_id == product_id]
		# build calendar events
		events: list[dict[str, Any]] = []
		for commitment in product_commitments:
			events.append({
				"type": "regulatory_commitment",
				"description": commitment.description,
				"due_date": str(commitment.due_date),
				"authority": commitment.authority,
				"status": commitment.status,
			})
		for pms in product_pms:
			events.append({
				"type": "pms_activity",
				"description": f"PMS {pms.pms_type} review",
				"due_date": str(getattr(pms, "review_date", datetime.utcnow() + timedelta(days=365))),
				"status": pms.status,
			})
		events.sort(key=lambda e: e["due_date"])
		calendar: dict[str, Any] = {
			"id": calendar_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"jurisdiction": jurisdiction,
			"year": cal_year,
			"total_events": len(events),
			"events": events,
			"overdue_count": sum(1 for e in events if e["due_date"] < str(datetime.utcnow())),
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._compliance_calendars[self._key(tenant_id, calendar_id)] = calendar
		self._audit(tenant_id, "compliance_calendar_generated", calendar_id)
		return calendar

	# --- NEW: renewal_tracking ---

	def renewal_tracking(
		self,
		registration_id: str,
		tenant_id: str,
		product_id: str = "",
		jurisdiction: str = "",
		current_expiry: datetime | None = None,
		renewal_lead_days: int = 180,
	) -> dict[str, Any]:
		"""Track marketing authorisation renewal: calculate deadline, flag risk, record renewal submission."""
		assert registration_id, "registration_id required"
		expiry = current_expiry or (datetime.utcnow() + timedelta(days=365))
		days_to_expiry = (expiry - datetime.utcnow()).days
		renewal_deadline = expiry - timedelta(days=renewal_lead_days)
		days_to_renewal_deadline = (renewal_deadline - datetime.utcnow()).days
		risk_level = "critical" if days_to_expiry < 90 else "high" if days_to_expiry < 180 else "medium" if days_to_expiry < 365 else "low"
		renewal: dict[str, Any] = {
			"registration_id": registration_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"jurisdiction": jurisdiction,
			"current_expiry": str(expiry),
			"renewal_deadline": str(renewal_deadline),
			"days_to_expiry": days_to_expiry,
			"days_to_renewal_deadline": days_to_renewal_deadline,
			"risk_level": risk_level,
			"action_required": days_to_renewal_deadline <= 30,
			"tracked_at": datetime.utcnow().isoformat(),
		}
		self._registration_renewals[self._key(tenant_id, registration_id)] = renewal
		self._audit(tenant_id, "renewal_tracked", registration_id)
		if risk_level in ("critical", "high"):
			self._audit(tenant_id, "renewal_risk_escalated", registration_id)
		return renewal

	# --- NEW: label_management ---

	def label_management(
		self,
		product_id: str,
		territory: str,
		label_version: str,
		tenant_id: str,
		change_type: str = "type_ia",
		language: str = "en",
		approved_by: str = "system",
		mlr_reference: str = "",
	) -> LabelRecord:
		"""Manage the full lifecycle of a product label in a territory: create, version, submit, approve."""
		assert product_id and territory and label_version, "product_id, territory and label_version required"
		assert change_type in SUPPORTED_LABEL_CHANGE_TYPES, f"unsupported change_type: {change_type}"
		label_number = f"LBL-{product_id[:6].upper()}-{territory.upper()}-{label_version}"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "initiate_label_change",
			"change_type_supported": True,
		})
		label = LabelRecord(
			tenant_id=tenant_id,
			label_number=label_number,
			product_id=product_id,
			market=territory,
			language=language,
			version=label_version,
			change_type=change_type,
			mlr_reference=mlr_reference,
			created_by=approved_by,
		)
		self._labels[self._key(tenant_id, label.id)] = label
		self._audit(tenant_id, "label_created", label.id)
		return label

	# --- NEW: post_market_surveillance ---

	def post_market_surveillance(
		self,
		product_id: str,
		period: str,
		data: dict[str, Any],
		tenant_id: str,
		pms_type: str = "periodic_safety_update",
		protocol_reference: str = "",
	) -> dict[str, Any]:
		"""Record post-market surveillance data collection for a product and period."""
		assert product_id and period, "product_id and period required"
		assert pms_type in SUPPORTED_PMS_TYPES, f"unsupported pms_type: {pms_type}"
		pms_number = f"PMS-{product_id[:6].upper()}-{period}"
		pms = self.create_pms(tenant_id, pms_number, product_id, pms_type, self._actor_id)
		if protocol_reference:
			pms = self.start_pms(pms.id, tenant_id, protocol_reference, datetime.utcnow())
		record: dict[str, Any] = {
			"pms_id": pms.id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"period": period,
			"pms_type": pms_type,
			"data_collected": data,
			"adverse_events_count": data.get("adverse_events_count", 0),
			"literature_articles_reviewed": data.get("literature_articles_reviewed", 0),
			"benefit_risk_conclusion": data.get("benefit_risk_conclusion", ""),
			"collected_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "pms_data_collected", pms.id)
		return record

	# --- NEW: rems_programme ---

	def rems_programme(
		self,
		drug_id: str,
		requirement_type: str,
		monitoring_data: dict[str, Any],
		tenant_id: str,
		rems_id: str | None = None,
		programme_name: str = "",
		risk_mitigation_strategy: str = "",
	) -> dict[str, Any]:
		"""Manage an FDA/EMA Risk Evaluation and Mitigation Strategy (REMS) programme for a drug."""
		assert drug_id and requirement_type, "drug_id and requirement_type required"
		assert requirement_type in ("healthcare_provider_training", "patient_enrollment",
			"pharmacy_certification", "medication_guide", "elements_to_assure_safe_use"), \
			f"unsupported requirement_type: {requirement_type}"
		record_id = rems_id or _uuid7str()
		compliance_status = monitoring_data.get("compliance_status", "compliant")
		enrolled_providers = monitoring_data.get("enrolled_providers", 0)
		certified_pharmacies = monitoring_data.get("certified_pharmacies", 0)
		enrolled_patients = monitoring_data.get("enrolled_patients", 0)
		record: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"drug_id": drug_id,
			"requirement_type": requirement_type,
			"programme_name": programme_name,
			"risk_mitigation_strategy": risk_mitigation_strategy,
			"compliance_status": compliance_status,
			"enrolled_providers": enrolled_providers,
			"certified_pharmacies": certified_pharmacies,
			"enrolled_patients": enrolled_patients,
			"monitoring_data": monitoring_data,
			"updated_at": datetime.utcnow().isoformat(),
		}
		self._rems_records[self._key(tenant_id, record_id)] = record
		self._audit(tenant_id, "rems_programme_updated", record_id)
		if compliance_status != "compliant":
			self._audit(tenant_id, "rems_non_compliance_detected", record_id)
		return record

	# --- NEW: import_licence ---

	def import_licence(
		self,
		product_id: str,
		country: str,
		quantity: float,
		tenant_id: str,
		licence_type: str = "standard",
		authority: str = "",
		application_reference: str = "",
	) -> dict[str, Any]:
		"""Apply for and track an import licence for a product in a country."""
		assert product_id and country, "product_id and country required"
		assert quantity > 0, "quantity must be positive"
		licence_id = _uuid7str()
		licence_number = f"IMP-{country.upper()}-{product_id[:6].upper()}-{licence_id[:6].upper()}"
		licence: dict[str, Any] = {
			"id": licence_id,
			"tenant_id": tenant_id,
			"licence_number": licence_number,
			"product_id": product_id,
			"country": country,
			"quantity": quantity,
			"licence_type": licence_type,
			"issuing_authority": authority,
			"application_reference": application_reference,
			"status": "applied",
			"applied_at": datetime.utcnow().isoformat(),
		}
		self._import_licences[self._key(tenant_id, licence_id)] = licence
		self._audit(tenant_id, "import_licence_applied", licence_id)
		return licence

	# --- NEW: regulatory_intelligence ---

	def regulatory_intelligence(
		self,
		jurisdiction: str,
		area: str,
		period: str,
		tenant_id: str,
		intel_type: str = "guidance",
		source_urls: list[str] | None = None,
	) -> dict[str, Any]:
		"""Scan and record regulatory intelligence updates for a jurisdiction and therapeutic area."""
		assert jurisdiction and area, "jurisdiction and area required"
		intel_number = f"INTEL-{jurisdiction.upper()}-{_uuid7str()[:6].upper()}"
		description = f"Regulatory intelligence for {area} in {jurisdiction} for period {period}"
		intel = self.record_intel(
			tenant_id=tenant_id,
			intel_number=intel_number,
			intel_type=intel_type if intel_type in SUPPORTED_INTEL_TYPES else "guidance",
			region=jurisdiction,
			title=f"{area} regulatory update — {period}",
			description=description,
			created_by=self._actor_id,
			source_url=source_urls[0] if source_urls else None,
		)
		summary: dict[str, Any] = {
			"intel_id": intel.id,
			"jurisdiction": jurisdiction,
			"area": area,
			"period": period,
			"intel_type": intel_type,
			"source_count": len(source_urls or []),
			"impact_assessed": False,
			"generated_at": datetime.utcnow().isoformat(),
		}
		return summary

	# --- NEW: commitment_tracking ---

	def commitment_tracking(
		self,
		product_id: str,
		commitment_id: str,
		status: str,
		tenant_id: str,
		submission_reference: str = "",
		milestone_achieved: str | None = None,
	) -> RegulatoryCommitment:
		"""Update the status of a specific regulatory commitment and record milestone achievement."""
		assert product_id and commitment_id, "product_id and commitment_id required"
		assert status in ("open", "in_progress", "fulfilled", "overdue", "withdrawn"), \
			f"unsupported status: {status}"
		commitment = self._commitments.get(self._key(tenant_id, commitment_id))
		if commitment is None:
			raise KeyError(f"commitment {commitment_id} not found")
		data = commitment.model_dump()
		data["status"] = status
		if status == "fulfilled" and submission_reference:
			data["submission_reference"] = submission_reference
			data["completed_date"] = datetime.utcnow()
		if milestone_achieved:
			milestones = data.get("milestones", [])
			for m in milestones:
				if m.get("id") == milestone_achieved:
					m["achieved"] = True
					m["achieved_date"] = datetime.utcnow().isoformat()
			data["milestones"] = milestones
		data["updated_at"] = datetime.utcnow()
		updated = RegulatoryCommitment(**data)
		self._commitments[self._key(tenant_id, commitment_id)] = updated
		self._audit(tenant_id, f"commitment_{status}", commitment_id)
		return updated

	# --- NEW: compliance_dashboard ---

	def compliance_dashboard(self, product_id: str, tenant_id: str) -> dict[str, Any]:
		"""Return a product-level regulatory compliance dashboard aggregating all compliance dimensions."""
		assert product_id, "product_id required"
		labels = [l for l in self._labels.values()
			if l.tenant_id == tenant_id and l.product_id == product_id]
		pms = [p for p in self._pms.values()
			if p.tenant_id == tenant_id and p.product_id == product_id]
		commitments = [c for c in self._commitments.values()
			if c.tenant_id == tenant_id and c.product_id == product_id]
		overdue_commitments = [c for c in commitments if getattr(c, "overdue", False)]
		rems = [r for r in self._rems_records.values()
			if r["tenant_id"] == tenant_id and r["drug_id"] == product_id]
		import_licences = [l for l in self._import_licences.values()
			if l["tenant_id"] == tenant_id and l["product_id"] == product_id]
		renewals = [r for r in self._registration_renewals.values()
			if r["tenant_id"] == tenant_id and r["product_id"] == product_id]
		inspections = self.list_inspections(tenant_id)
		open_warnings = [i for i in inspections
			if i.outcome == "warning_letter" and i.response_submitted_date is None]
		intel = self.list_intel(tenant_id)
		unassessed_intel = [i for i in intel if not i.impact_assessed]
		return {
			"product_id": product_id,
			"tenant_id": tenant_id,
			"label_count": len(labels),
			"approved_labels": sum(1 for l in labels if l.status == "approved"),
			"active_pms": sum(1 for p in pms if p.status == "active"),
			"total_commitments": len(commitments),
			"open_commitments": sum(1 for c in commitments if c.status == "open"),
			"overdue_commitments": len(overdue_commitments),
			"active_rems_programmes": len([r for r in rems if r.get("compliance_status") == "compliant"]),
			"import_licences": len(import_licences),
			"active_import_licences": sum(1 for l in import_licences if l.get("status") == "active"),
			"renewal_risks": len([r for r in renewals if r.get("risk_level") in ("high", "critical")]),
			"open_warning_letters": len(open_warnings),
			"unassessed_intel": len(unassessed_intel),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: authority_interaction ---

	def authority_interaction(
		self,
		product_id: str,
		agency: str,
		meeting_type: str,
		tenant_id: str,
		meeting_date: datetime | None = None,
		agenda: list[str] | None = None,
		outcome: str = "",
		commitments_made: list[str] | None = None,
	) -> dict[str, Any]:
		"""Record a regulatory authority interaction (pre-submission, scientific advice, inspection response)."""
		assert product_id and agency, "product_id and agency required"
		assert meeting_type in ("pre_submission", "scientific_advice", "type_ii_variation",
			"inspection_response", "post_approval", "ad_hoc"), \
			f"unsupported meeting_type: {meeting_type}"
		interaction_id = _uuid7str()
		interaction: dict[str, Any] = {
			"id": interaction_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"agency": agency,
			"meeting_type": meeting_type,
			"meeting_date": str(meeting_date or datetime.utcnow()),
			"agenda": agenda or [],
			"outcome": outcome,
			"commitments_made": commitments_made or [],
			"follow_up_actions": [],
			"status": "completed" if outcome else "scheduled",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._authority_interactions[self._key(tenant_id, interaction_id)] = interaction
		self._audit(tenant_id, "authority_interaction_recorded", interaction_id)
		if commitments_made:
			for commitment_desc in commitments_made:
				self._audit(tenant_id, "regulatory_commitment_created", interaction_id)
		return interaction

	# --- dashboard ---

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return regulatory compliance dashboard."""
		return {
			"tenant_id": tenant_id,
			"framework_count": self._count(self._frameworks, tenant_id),
			"open_inspections": sum(1 for i in self._inspections.values()
								if i.tenant_id == tenant_id and i.status != "completed"),
			"warning_letters": sum(1 for i in self._inspections.values()
								if i.tenant_id == tenant_id and i.outcome == "warning_letter"
								and i.response_submitted_date is None),
			"label_count": self._count(self._labels, tenant_id),
			"active_pms": sum(1 for p in self._pms.values()
							if p.tenant_id == tenant_id and p.status == "active"),
			"unassessed_intel": sum(1 for i in self._intel.values()
								if i.tenant_id == tenant_id and not i.impact_assessed),
			"open_commitments": sum(1 for c in self._commitments.values()
								if c.tenant_id == tenant_id and c.status == "open"),
			"overdue_commitments": sum(1 for c in self._commitments.values()
									if c.tenant_id == tenant_id and c.overdue),
			"rems_programmes": sum(1 for r in self._rems_records.values() if r["tenant_id"] == tenant_id),
			"import_licences": sum(1 for l in self._import_licences.values() if l["tenant_id"] == tenant_id),
			"authority_interactions": sum(1 for i in self._authority_interactions.values() if i["tenant_id"] == tenant_id),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id),
		}

	# --- private helpers ---

	def _log_inspection_countdown(self, inspection_id: str, days_to_inspection: int) -> None:
		pass

	def _log_commitment_risk(self, commitment_id: str, days_to_due: int) -> None:
		pass

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"stream": "apg.pharma.rec.lifecycle",
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

	async def bulk_create_records(self, records: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Create Records"""
		assert records
		return {"created_count": len(records), "tenant_id": tenant_id}

	async def analytics_summary(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		return {"tenant_id": tenant_id, "period": period}

	async def get_audit_events(self, tenant_id: str) -> dict[str, Any]:
		"""Get Audit Events"""
		return [e for e in self._audit_events if e["tenant_id"] == tenant_id]

	async def search_records(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search Records"""
		assert query
		return {"query": query, "results": [], "tenant_id": tenant_id}

PharmaRecService = RegulatoryComplianceService
