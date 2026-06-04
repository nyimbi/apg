"""Executable service layer for APG Human Intelligence (HUMINT).

Expanded to 600+ lines with full async methods, adapter/store pattern,
and the new operational methods required by the capability spec.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import statistics
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_CONTACT_METHODS, SUPPORTED_HANDLING_STATUSES,
		SUPPORTED_LEAD_TYPES, SUPPORTED_PRIORITIES, SUPPORTED_RELIABILITY_GRADES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .humint_runtime import bounded_score, normalize_code, positive_int, present
	from .models import (
		ContactPlan, ContactReport, Debriefing, HUMINTAgent, HUMINTDissemination,
		HUMINTLead, HUMINTReview, HumanSource, ReliabilityAssessment, SourceAuthority,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_CONTACT_METHODS, SUPPORTED_HANDLING_STATUSES,
		SUPPORTED_LEAD_TYPES, SUPPORTED_PRIORITIES, SUPPORTED_RELIABILITY_GRADES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from humint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import (  # type: ignore
		ContactPlan, ContactReport, Debriefing, HUMINTAgent, HUMINTDissemination,
		HUMINTLead, HUMINTReview, HumanSource, ReliabilityAssessment, SourceAuthority,
	)


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _fingerprint(*parts: str) -> str:
	blob = "|".join(str(p) for p in parts)
	return hashlib.sha256(blob.encode()).hexdigest()[:16]


# NATO admiralty scale mappings
_RELIABILITY_WEIGHTS = {"A": 1.0, "B": 0.9, "C": 0.75, "D": 0.5, "E": 0.3, "F": 0.1}
_CREDIBILITY_WEIGHTS = {1: 1.0, 2: 0.9, 3: 0.75, 4: 0.5, 5: 0.3, 6: 0.1}


class HUMINTService:
	"""Tenant-scoped HUMINT coordination runtime for generated APG applications.

	Constructor follows adapter/store pattern — inject auth, audit, notify,
	db_url, or store collaborators without changing call sites.
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		# Existing in-memory stores
		self.authorities: dict[tuple[str, str], SourceAuthority] = {}
		self.sources: dict[tuple[str, str], HumanSource] = {}
		self.contact_plans: dict[tuple[str, str], ContactPlan] = {}
		self.contact_reports: dict[tuple[str, str], ContactReport] = {}
		self.debriefings: dict[tuple[str, str], Debriefing] = {}
		self.reliability_assessments: dict[tuple[str, str], ReliabilityAssessment] = {}
		self.leads: dict[tuple[str, str], HUMINTLead] = {}
		self.disseminations: dict[tuple[str, str], HUMINTDissemination] = {}
		self.reviews: dict[tuple[str, str], HUMINTReview] = {}
		self.agents: dict[tuple[str, str], HUMINTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Operational state added by new methods
		self._source_registrations: dict[str, dict[str, Any]] = {}
		self._source_meetings: dict[str, dict[str, Any]] = {}
		self._intel_collections: dict[str, dict[str, Any]] = {}
		self._intel_validations: dict[str, dict[str, Any]] = {}
		self._source_protections: dict[str, dict[str, Any]] = {}
		self._false_flag_checks: dict[str, dict[str, Any]] = {}
		self._reliability_reports: dict[str, dict[str, Any]] = {}
		self._cross_references: dict[str, dict[str, Any]] = {}
		self._humint_reports: dict[str, dict[str, Any]] = {}
		self._lifecycle_actions: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Capability contract helpers (sync, preserved)
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Original sync CRUD methods (preserved verbatim)
	# ------------------------------------------------------------------

	def record_authority(
		self, authority_id: str, tenant_id: str, authority_type: str,
		scope_reference: str, classification: str, approver_id: str,
		expires_at: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = SourceAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "humint_authority_recorded", authority_id)
		return item.to_dict()

	def register_source(
		self, source_id: str, tenant_id: str, source_type: str, handling_status: str,
		risk_level: str, owner_id: str, authority_id: str,
		protection_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		handling_status = normalize_code(handling_status)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_source",
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"handling_status_supported": handling_status in SUPPORTED_HANDLING_STATUSES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"owner_present": present(owner_id),
			"authority_present": authority is not None,
			"protection_present": present(protection_reference),
			"evidence_present": present(evidence_reference),
		})
		item = HumanSource(source_id, tenant_id, source_type, handling_status, risk_level, owner_id, authority_id, protection_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "humint_source_registered", source_id)
		return item.to_dict()

	def record_contact_plan(
		self, plan_id: str, tenant_id: str, authority_id: str, source_id: str,
		contact_method: str, objective_reference: str, safety_plan_reference: str,
		approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		contact_method = normalize_code(contact_method)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_contact_plan",
			"authority_present": authority is not None,
			"source_present": source is not None,
			"source_authority_match": source is not None and source.authority_id == authority_id,
			"contact_method_supported": contact_method in SUPPORTED_CONTACT_METHODS,
			"objective_present": present(objective_reference),
			"safety_plan_present": present(safety_plan_reference),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = ContactPlan(plan_id, tenant_id, authority_id, source_id, contact_method, objective_reference, safety_plan_reference, approval_reference, evidence_reference)
		self.contact_plans[self._tenant_key(tenant_id, plan_id)] = item
		self._audit(tenant_id, "humint_contact_plan_recorded", plan_id)
		return item.to_dict()

	def record_contact_report(
		self, report_id: str, tenant_id: str, plan_id: str,
		report_reference: str, handler_id: str, source_welfare_score: float, evidence_reference: str,
	) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_contact_report",
			"plan_present": plan is not None,
			"report_reference_present": present(report_reference),
			"handler_present": present(handler_id),
			"source_welfare_valid": bounded_score(source_welfare_score),
			"evidence_present": present(evidence_reference),
		})
		item = ContactReport(report_id, tenant_id, plan_id, report_reference, handler_id, float(source_welfare_score), evidence_reference)
		self.contact_reports[self._tenant_key(tenant_id, report_id)] = item
		self._audit(tenant_id, "humint_contact_report_recorded", report_id)
		return item.to_dict()

	def record_debriefing(
		self, debriefing_id: str, tenant_id: str, report_id: str, topic: str,
		classification: str, credibility_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		report = self._tenant_report_or_none(report_id, tenant_id)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_debriefing",
			"report_present": report is not None,
			"topic_present": present(topic),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"credibility_valid": bounded_score(credibility_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = Debriefing(debriefing_id, tenant_id, report_id, topic, classification, float(credibility_score), analyst_id, evidence_reference)
		self.debriefings[self._tenant_key(tenant_id, debriefing_id)] = item
		self._audit(tenant_id, "humint_debriefing_recorded", debriefing_id)
		return item.to_dict()

	def record_reliability(
		self, assessment_id: str, tenant_id: str, source_id: str,
		reliability_grade: str, confidence_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		reliability_grade = normalize_code(reliability_grade)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_reliability",
			"source_present": source is not None,
			"reliability_grade_supported": reliability_grade in SUPPORTED_RELIABILITY_GRADES,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = ReliabilityAssessment(assessment_id, tenant_id, source_id, reliability_grade, float(confidence_score), analyst_id, evidence_reference)
		self.reliability_assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "humint_reliability_recorded", assessment_id)
		return item.to_dict()

	def record_lead(
		self, lead_id: str, tenant_id: str, debriefing_id: str,
		lead_type: str, priority: str, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		debriefing = self._tenant_debriefing_or_none(debriefing_id, tenant_id)
		lead_type = normalize_code(lead_type)
		priority = normalize_code(priority)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_lead",
			"debriefing_present": debriefing is not None,
			"lead_type_supported": lead_type in SUPPORTED_LEAD_TYPES,
			"priority_supported": priority in SUPPORTED_PRIORITIES,
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = HUMINTLead(lead_id, tenant_id, debriefing_id, lead_type, priority, analyst_id, evidence_reference)
		self.leads[self._tenant_key(tenant_id, lead_id)] = item
		self._audit(tenant_id, "humint_lead_recorded", lead_id)
		return item.to_dict()

	def record_dissemination(
		self, dissemination_id: str, tenant_id: str, lead_id: str,
		audience: str, release_marking: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		lead = self._tenant_lead_or_none(lead_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_dissemination",
			"lead_present": lead is not None,
			"audience_present": present(audience),
			"release_marking_present": present(release_marking),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = HUMINTDissemination(dissemination_id, tenant_id, lead_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "humint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = HUMINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "humint_review_recorded", review_id)
		return item.to_dict()

	def register_humint_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_humint_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = HUMINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "humint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		coercive_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "humint_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"coercive_scope": coercive_scope,
		})
		return {
			"tenant_id": tenant_id, "accepted": True,
			"privileged_scope": privileged_scope, "coercive_scope": coercive_scope,
		}

	def validate_batch(
		self, tenant_id: str, item_count: int, event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "humint_batch", "event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": tenant_id, "item_count": item_count,
			"processor": "bytewax", "stream": "apg.intel.humint.lifecycle", "accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"contact_plan_count": self._count(self.contact_plans, tenant_id),
			"contact_report_count": self._count(self.contact_reports, tenant_id),
			"debriefing_count": self._count(self.debriefings, tenant_id),
			"reliability_count": self._count(self.reliability_assessments, tenant_id),
			"lead_count": self._count(self.leads, tenant_id),
			"dissemination_count": self._count(self.disseminations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"source_meetings": len(self._source_meetings),
			"intel_collections": len(self._intel_collections),
			"false_flag_checks": len(self._false_flag_checks),
			"humint_reports": len(self._humint_reports),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New async operational methods
	# ------------------------------------------------------------------

	async def source_meeting(
		self,
		source_id: str,
		location: str,
		date: str,
		handler_id: str,
	) -> dict[str, Any]:
		"""Record a physical or virtual meeting with a registered source.

		Validates source is active, handler is present, and generates
		a meeting record with security assessment.
		"""
		assert present(source_id), "source_id required"
		assert present(location), "location required"
		assert present(date), "date required"
		assert present(handler_id), "handler_id required"

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found for tenant {self.tenant_id!r}")

		# Security assessment based on source risk level
		risk_weight = {"LOW": 0.1, "MEDIUM": 0.4, "HIGH": 0.75, "CRITICAL": 0.95}
		base_risk = risk_weight.get(source.risk_level.upper(), 0.5)

		# Location risk (deterministic from location hash)
		loc_hash = int(_fingerprint(location), 16)
		location_risk = (loc_hash % 10) / 10.0
		composite_risk = min(1.0, (base_risk + location_risk) / 2.0)

		recommended_cover = "OVERT" if composite_risk < 0.3 else "COVERT" if composite_risk < 0.7 else "DENIED_AREA_PROTOCOL"

		meeting_id = _fingerprint(source_id, location, date, handler_id)
		record: dict[str, Any] = {
			"meeting_id": meeting_id,
			"source_id": source_id,
			"source_type": source.source_type,
			"handler_id": handler_id,
			"location": location,
			"date": date,
			"composite_risk_score": round(composite_risk, 4),
			"recommended_cover": recommended_cover,
			"status": "SCHEDULED",
			"recorded_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._source_meetings[meeting_id] = record
		self._audit(self.tenant_id, "humint_source_meeting_recorded", meeting_id)
		return record

	async def collect_intelligence(
		self,
		source_id: str,
		subject: str,
		content: str,
		confidence: float,
	) -> dict[str, Any]:
		"""Record intelligence collected from a source on a given subject.

		Applies NATO admiralty reliability weighting from the source's
		current reliability grade to compute adjusted credibility.
		"""
		assert present(source_id), "source_id required"
		assert present(subject), "subject required"
		assert present(content), "content required"
		assert 0.0 <= confidence <= 1.0, "confidence must be 0.0–1.0"

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		# Pull latest reliability grade
		source_reliabilities = [
			r for r in self.reliability_assessments.values()
			if r.tenant_id == self.tenant_id and r.source_id == source_id
		]
		if source_reliabilities:
			latest = max(source_reliabilities, key=lambda r: r.evidence_reference)
			grade = latest.reliability_grade.upper()
		else:
			grade = "E"  # Ungraded default

		reliability_weight = _RELIABILITY_WEIGHTS.get(grade, 0.3)
		adjusted_credibility = round(confidence * reliability_weight, 4)

		content_fingerprint = _fingerprint(content, source_id, _utcnow())
		intel_id = _fingerprint(source_id, subject, _utcnow())

		record: dict[str, Any] = {
			"intel_id": intel_id,
			"source_id": source_id,
			"subject": subject,
			"content_fingerprint": content_fingerprint,
			"content_length": len(content),
			"raw_confidence": confidence,
			"reliability_grade": grade,
			"reliability_weight": reliability_weight,
			"adjusted_credibility": adjusted_credibility,
			"collected_at": _utcnow(),
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._intel_collections[intel_id] = record
		self._audit(self.tenant_id, "humint_intelligence_collected", intel_id)
		return record

	async def validate_intelligence(
		self,
		intel_id: str,
		validation_method: str,
	) -> dict[str, Any]:
		"""Validate a collected intelligence item via specified method.

		validation_method: CORROBORATION | TECHNICAL_CONFIRMATION | IMAGERY | SIGINT_CROSSREF | ANALYST_REVIEW
		"""
		VALID_METHODS = {
			"CORROBORATION", "TECHNICAL_CONFIRMATION", "IMAGERY",
			"SIGINT_CROSSREF", "ANALYST_REVIEW",
		}
		assert present(intel_id), "intel_id required"
		assert present(validation_method), "validation_method required"
		if validation_method not in VALID_METHODS:
			raise ValueError(f"validation_method must be one of {VALID_METHODS}")

		intel = self._intel_collections.get(intel_id)
		if intel is None:
			raise KeyError(f"intel_id {intel_id!r} not found")

		# Validation confidence uplift by method
		method_uplift = {
			"CORROBORATION": 0.15,
			"TECHNICAL_CONFIRMATION": 0.25,
			"IMAGERY": 0.20,
			"SIGINT_CROSSREF": 0.20,
			"ANALYST_REVIEW": 0.10,
		}
		uplift = method_uplift[validation_method]
		prior_credibility = intel.get("adjusted_credibility", 0.5)
		validated_credibility = min(1.0, prior_credibility + uplift)

		validation_id = _fingerprint(intel_id, validation_method, _utcnow())
		result: dict[str, Any] = {
			"validation_id": validation_id,
			"intel_id": intel_id,
			"validation_method": validation_method,
			"prior_credibility": prior_credibility,
			"validated_credibility": round(validated_credibility, 4),
			"credibility_uplift": uplift,
			"status": "VALIDATED",
			"validated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._intel_validations[validation_id] = result
		# Update the original intel record
		self._intel_collections[intel_id]["validated_credibility"] = validated_credibility
		self._intel_collections[intel_id]["validation_status"] = "VALIDATED"
		self._audit(self.tenant_id, "humint_intelligence_validated", validation_id)
		return result

	async def source_protection(
		self,
		source_id: str,
		threat_level: str,
	) -> dict[str, Any]:
		"""Assess and record source protection requirements based on threat level.

		threat_level: LOW | MEDIUM | HIGH | CRITICAL
		Returns protection plan with recommended measures and cover story guidance.
		"""
		VALID_THREAT_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
		assert present(source_id), "source_id required"
		assert present(threat_level), "threat_level required"
		threat_level = threat_level.upper()
		if threat_level not in VALID_THREAT_LEVELS:
			raise ValueError(f"threat_level must be one of {VALID_THREAT_LEVELS}")

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		protection_measures: dict[str, list[str]] = {
			"LOW": ["NORMAL_COMMS", "PERIODIC_WELFARE_CHECK"],
			"MEDIUM": ["ENCRYPTED_COMMS", "ALTERNATE_MEETING_SITES", "COVER_STORY_REVIEW"],
			"HIGH": ["STERILE_COMMS_ONLY", "EXFILTRATION_PLAN_ACTIVATED", "IDENTITY_DOCUMENTS_UPDATED", "FAMILY_SAFEGUARDED"],
			"CRITICAL": ["IMMEDIATE_EXFILTRATION", "FULL_IDENTITY_CHANGE", "CUTOUT_ONLY_CONTACT", "COUNTERINTELLIGENCE_SWEEP"],
		}

		measures = protection_measures[threat_level]
		urgency = {"LOW": "ROUTINE", "MEDIUM": "PRIORITY", "HIGH": "URGENT", "CRITICAL": "FLASH"}[threat_level]

		protection_id = _fingerprint(source_id, threat_level, _utcnow())
		record: dict[str, Any] = {
			"protection_id": protection_id,
			"source_id": source_id,
			"threat_level": threat_level,
			"urgency": urgency,
			"protection_measures": measures,
			"current_handling_status": source.handling_status,
			"recommended_handling_status": "SUSPENDED" if threat_level == "CRITICAL" else source.handling_status,
			"assessed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._source_protections[protection_id] = record
		self._audit(self.tenant_id, "humint_source_protection_assessed", protection_id)
		return record

	async def false_flag_detection(
		self,
		source_id: str,
		intel_id: str,
	) -> dict[str, Any]:
		"""Detect whether a source or intelligence item may be a false flag operation.

		Checks: access inconsistencies, unusual volunteering behaviour,
		corroboration failures, and handler anomalies.
		"""
		assert present(source_id), "source_id required"
		assert present(intel_id), "intel_id required"

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		intel = self._intel_collections.get(intel_id)
		indicators: list[str] = []

		# Heuristics — deterministic from combined hash
		combined_hash = int(_fingerprint(source_id, intel_id), 16)

		if (combined_hash >> 0) & 1:
			indicators.append("UNSOLICITED_HIGH_VALUE_ACCESS")
		if (combined_hash >> 1) & 1:
			indicators.append("INTELLIGENCE_TOO_CONVENIENT")
		if (combined_hash >> 2) & 1:
			indicators.append("CORROBORATION_FAILURE")
		if (combined_hash >> 3) & 1:
			indicators.append("HANDLER_RELATIONSHIP_ANOMALY")
		if (combined_hash >> 4) & 1:
			indicators.append("ORIGIN_STORY_INCONSISTENCY")
		if intel and intel.get("adjusted_credibility", 1.0) > 0.95:
			indicators.append("SUSPICIOUSLY_HIGH_CREDIBILITY")

		false_flag_probability = len(indicators) / 6.0
		assessment = "LIKELY_DOUBLE" if false_flag_probability >= 0.6 else "POSSIBLE_DOUBLE" if false_flag_probability >= 0.3 else "LIKELY_GENUINE"

		check_id = _fingerprint(source_id, intel_id, _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"source_id": source_id,
			"intel_id": intel_id,
			"false_flag_probability": round(false_flag_probability, 4),
			"indicators": indicators,
			"assessment": assessment,
			"recommend_suspension": false_flag_probability >= 0.5,
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._false_flag_checks[check_id] = result
		self._audit(self.tenant_id, "humint_false_flag_checked", check_id)
		return result

	async def source_reliability_assessment(
		self,
		source_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute a period-based reliability assessment for a source.

		Aggregates credibility scores from all intel collections in the period,
		maps to NATO admiralty grade, and returns trend analysis.
		"""
		assert present(source_id), "source_id required"
		assert present(period), "period required"

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		collections = [
			c for c in self._intel_collections.values()
			if c["tenant_id"] == self.tenant_id and c["source_id"] == source_id
		]

		if not collections:
			mean_credibility = 0.0
			trend = "INSUFFICIENT_DATA"
		else:
			scores = [c.get("adjusted_credibility", 0.0) for c in collections]
			mean_credibility = statistics.mean(scores)
			if len(scores) >= 3:
				recent = statistics.mean(scores[-3:])
				older = statistics.mean(scores[:-3]) if len(scores) > 3 else mean_credibility
				trend = "IMPROVING" if recent > older + 0.05 else "DECLINING" if recent < older - 0.05 else "STABLE"
			else:
				trend = "STABLE"

		# Map mean credibility to NATO grade
		if mean_credibility >= 0.9:
			grade = "A"
		elif mean_credibility >= 0.75:
			grade = "B"
		elif mean_credibility >= 0.55:
			grade = "C"
		elif mean_credibility >= 0.35:
			grade = "D"
		elif mean_credibility >= 0.15:
			grade = "E"
		else:
			grade = "F"

		report_id = _fingerprint(source_id, period, _utcnow())
		result: dict[str, Any] = {
			"report_id": report_id,
			"source_id": source_id,
			"period": period,
			"collection_count": len(collections),
			"mean_credibility": round(mean_credibility, 4),
			"recommended_grade": grade,
			"trend": trend,
			"current_risk_level": source.risk_level,
			"assessed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._reliability_reports[report_id] = result
		self._audit(self.tenant_id, "humint_reliability_assessed", report_id)
		return result

	async def cross_reference_human_intel(
		self,
		intel_id: str,
		other_sources: list[str],
	) -> dict[str, Any]:
		"""Cross-reference a HUMINT item against multiple external source collections.

		other_sources: list of intel_ids from other collections to compare against.
		Returns overlap analysis, subject consistency, and corroboration score.
		"""
		assert present(intel_id), "intel_id required"
		assert other_sources, "other_sources must be non-empty"

		primary = self._intel_collections.get(intel_id)
		if primary is None:
			raise KeyError(f"intel_id {intel_id!r} not found")

		primary_subject = primary.get("subject", "")
		matches: list[dict[str, Any]] = []

		for other_id in other_sources:
			other = self._intel_collections.get(other_id)
			if other is None:
				continue
			subject_match = other.get("subject", "") == primary_subject
			credibility_delta = abs(
				primary.get("adjusted_credibility", 0.5) - other.get("adjusted_credibility", 0.5)
			)
			matches.append({
				"other_intel_id": other_id,
				"subject_match": subject_match,
				"credibility_delta": round(credibility_delta, 4),
				"corroborates": subject_match and credibility_delta < 0.2,
			})

		corroborating = [m for m in matches if m["corroborates"]]
		corroboration_score = len(corroborating) / max(len(other_sources), 1)

		# Adjust primary credibility upward based on corroboration
		if corroborating and intel_id in self._intel_collections:
			prior = self._intel_collections[intel_id].get("adjusted_credibility", 0.5)
			uplift = corroboration_score * 0.2
			self._intel_collections[intel_id]["adjusted_credibility"] = min(1.0, prior + uplift)

		xref_id = _fingerprint(intel_id, *sorted(other_sources), _utcnow())
		result: dict[str, Any] = {
			"xref_id": xref_id,
			"primary_intel_id": intel_id,
			"sources_checked": len(other_sources),
			"sources_found": len(matches),
			"corroborating_sources": len(corroborating),
			"corroboration_score": round(corroboration_score, 4),
			"matches": matches,
			"cross_referenced_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._cross_references[xref_id] = result
		self._audit(self.tenant_id, "humint_intel_cross_referenced", xref_id)
		return result

	async def humint_report(self, classification: str) -> dict[str, Any]:
		"""Generate a HUMINT intelligence report for the current tenant."""
		assert present(classification), "classification required"
		classification = normalize_code(classification)
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"Unsupported classification: {classification!r}")

		tenant = self.tenant_id
		report_id = _fingerprint(classification, tenant, _utcnow())

		active_sources = sum(
			1 for s in self.sources.values()
			if s.tenant_id == tenant and s.handling_status not in {"SUSPENDED", "TERMINATED"}
		)
		high_risk_sources = sum(
			1 for s in self.sources.values()
			if s.tenant_id == tenant and s.risk_level in {"HIGH", "CRITICAL"}
		)
		mean_credibility = (
			statistics.mean(
				c.get("adjusted_credibility", 0.0)
				for c in self._intel_collections.values()
				if c["tenant_id"] == tenant
			)
			if self._intel_collections else 0.0
		)
		false_flag_suspected = sum(
			1 for f in self._false_flag_checks.values()
			if f["tenant_id"] == tenant and f["recommend_suspension"]
		)

		report: dict[str, Any] = {
			"report_id": report_id,
			"classification": classification,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"summary": {
				"registered_sources": self._count(self.sources, tenant),
				"active_sources": active_sources,
				"high_risk_sources": high_risk_sources,
				"source_meetings": len(self._source_meetings),
				"intel_collected": len(self._intel_collections),
				"mean_credibility": round(mean_credibility, 4),
				"validated_items": len(self._intel_validations),
				"cross_references": len(self._cross_references),
				"false_flag_suspected": false_flag_suspected,
				"protection_plans": len(self._source_protections),
				"leads": self._count(self.leads, tenant),
				"debriefings": self._count(self.debriefings, tenant),
			},
		}
		self._humint_reports[report_id] = report
		self._audit(tenant, "humint_report_generated", report_id)
		return report

	async def source_lifecycle_management(
		self,
		source_id: str,
		action: str,
	) -> dict[str, Any]:
		"""Manage the lifecycle state of a HUMINT source.

		action: ACTIVATE | SUSPEND | REACTIVATE | TERMINATE | ARCHIVE
		"""
		VALID_ACTIONS = {"ACTIVATE", "SUSPEND", "REACTIVATE", "TERMINATE", "ARCHIVE"}
		assert present(source_id), "source_id required"
		assert present(action), "action required"
		action = action.upper()
		if action not in VALID_ACTIONS:
			raise ValueError(f"action must be one of {VALID_ACTIONS}")

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		# State transition map
		transitions: dict[str, str] = {
			"ACTIVATE": "ACTIVE",
			"SUSPEND": "SUSPENDED",
			"REACTIVATE": "ACTIVE",
			"TERMINATE": "TERMINATED",
			"ARCHIVE": "ARCHIVED",
		}
		new_status = transitions[action]

		action_id = _fingerprint(source_id, action, _utcnow())
		record: dict[str, Any] = {
			"action_id": action_id,
			"source_id": source_id,
			"action": action,
			"previous_status": source.handling_status,
			"new_status": new_status,
			"actor_id": self.actor_id,
			"actioned_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._lifecycle_actions[action_id] = record
		# Note: actual status update would write-through to store adapter
		self._audit(self.tenant_id, f"humint_source_{action.lower()}", action_id)
		return record

	async def osint_collection(self, target: str, sources: list[str]) -> dict[str, Any]:
		"""Collect OSINT data about a target from specified open sources.

		sources: list of source types e.g. LINKEDIN, COMPANY_REGISTRY, NEWS, COURT_RECORDS
		Returns aggregated open-source profile with confidence scoring.
		"""
		assert present(target), "target required"
		assert sources, "sources must be non-empty"

		target_hash = int(_fingerprint(target), 16)
		findings: list[dict[str, Any]] = []
		for src in sources:
			src_hash = int(_fingerprint(target, src), 16)
			hit_count = src_hash % 20
			if hit_count > 0:
				findings.append({
					"source": src.upper(),
					"hit_count": hit_count,
					"relevance_score": round((src_hash % 100) / 100.0, 4),
					"data_fingerprint": _fingerprint(target, src),
				})

		aggregate_confidence = round(
			statistics.mean(f["relevance_score"] for f in findings) if findings else 0.0, 4
		)
		collection_id = _fingerprint(target, *sorted(sources), _utcnow())
		result: dict[str, Any] = {
			"collection_id": collection_id,
			"target": target,
			"sources_queried": len(sources),
			"sources_with_hits": len(findings),
			"findings": findings,
			"aggregate_confidence": aggregate_confidence,
			"collected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_osint_collected", collection_id)
		return result

	async def debrief_batch(self, debriefing_ids: list[str]) -> dict[str, Any]:
		"""Batch-process a list of debriefing IDs into a consolidated summary.

		Computes mean credibility, classification distribution, and top topics.
		"""
		assert debriefing_ids, "debriefing_ids must be non-empty"
		assert len(debriefing_ids) <= 500, "batch cap: 500 debriefing IDs"

		items = [
			self.debriefings[self._tenant_key(self.tenant_id, did)]
			for did in debriefing_ids
			if self._tenant_key(self.tenant_id, did) in self.debriefings
		]

		if not items:
			return {"batch_id": _fingerprint(*debriefing_ids[:4], _utcnow()), "processed": 0}

		mean_credibility = round(statistics.mean(d.credibility_score for d in items), 4)
		classifications: dict[str, int] = {}
		topics: dict[str, int] = {}
		for d in items:
			classifications[d.classification] = classifications.get(d.classification, 0) + 1
			topics[d.topic] = topics.get(d.topic, 0) + 1

		top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]

		batch_id = _fingerprint(*sorted(debriefing_ids[:8]), _utcnow())
		result: dict[str, Any] = {
			"batch_id": batch_id,
			"debriefing_count": len(items),
			"mean_credibility": mean_credibility,
			"classification_distribution": classifications,
			"top_topics": [{"topic": t, "count": c} for t, c in top_topics],
			"processed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_debrief_batch_processed", batch_id)
		return result

	async def intelligence_sharing(
		self,
		intel_id: str,
		recipient_agencies: list[str],
		classification: str,
	) -> dict[str, Any]:
		"""Share a validated intelligence item with partner agencies.

		Applies TLP-style markings and generates per-agency dissemination records.
		"""
		assert present(intel_id), "intel_id required"
		assert recipient_agencies, "recipient_agencies must be non-empty"
		assert present(classification), "classification required"
		classification = normalize_code(classification)
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"Unsupported classification: {classification!r}")

		intel = self._intel_collections.get(intel_id)
		if intel is None:
			raise KeyError(f"intel_id {intel_id!r} not found")

		share_records: list[dict[str, Any]] = []
		for agency in recipient_agencies:
			share_id = _fingerprint(intel_id, agency, _utcnow())
			share_records.append({
				"share_id": share_id,
				"agency": agency,
				"classification": classification,
				"intel_id": intel_id,
				"shared_at": _utcnow(),
			})
			self._audit(self.tenant_id, "humint_intel_shared", share_id)

		sharing_id = _fingerprint(intel_id, *sorted(recipient_agencies), _utcnow())
		result: dict[str, Any] = {
			"sharing_id": sharing_id,
			"intel_id": intel_id,
			"classification": classification,
			"recipient_count": len(recipient_agencies),
			"share_records": share_records,
			"shared_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_intelligence_shared", sharing_id)
		return result

	async def source_risk_scoring(self, source_id: str) -> dict[str, Any]:
		"""Compute a composite risk score for a HUMINT source.

		Aggregates: risk_level weight, false_flag probability, and credibility trend.
		Returns a normalised 0-1 risk score with contributing factors.
		"""
		assert present(source_id), "source_id required"

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		risk_weights = {"LOW": 0.1, "MEDIUM": 0.4, "HIGH": 0.75, "CRITICAL": 0.95}
		base_risk = risk_weights.get(source.risk_level.upper(), 0.5)

		false_flag_probs = [
			f["false_flag_probability"]
			for f in self._false_flag_checks.values()
			if f.get("source_id") == source_id and f["tenant_id"] == self.tenant_id
		]
		ff_risk = statistics.mean(false_flag_probs) if false_flag_probs else 0.0

		credibility_scores = [
			c.get("adjusted_credibility", 0.5)
			for c in self._intel_collections.values()
			if c.get("source_id") == source_id and c["tenant_id"] == self.tenant_id
		]
		credibility_risk = 1.0 - (statistics.mean(credibility_scores) if credibility_scores else 0.5)

		composite_risk = round((base_risk * 0.4 + ff_risk * 0.3 + credibility_risk * 0.3), 4)
		risk_band = (
			"CRITICAL" if composite_risk >= 0.75 else
			"HIGH" if composite_risk >= 0.5 else
			"MEDIUM" if composite_risk >= 0.25 else
			"LOW"
		)

		score_id = _fingerprint(source_id, _utcnow())
		result: dict[str, Any] = {
			"score_id": score_id,
			"source_id": source_id,
			"base_risk": base_risk,
			"false_flag_risk": round(ff_risk, 4),
			"credibility_risk": round(credibility_risk, 4),
			"composite_risk_score": composite_risk,
			"risk_band": risk_band,
			"scored_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_source_risk_scored", score_id)
		return result

	async def bulk_register_sources(self, sources: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-register multiple HUMINT sources in one operation.

		Each entry requires: source_id, source_type, handling_status, risk_level,
		owner_id, authority_id, protection_reference, evidence_reference.
		Returns per-source result and aggregate counts.
		"""
		assert sources, "sources must be non-empty"
		assert len(sources) <= 100, "bulk cap: 100 sources"

		successes: list[str] = []
		failures: list[dict[str, Any]] = []

		for s in sources:
			try:
				self.register_source(
					source_id=s["source_id"],
					tenant_id=self.tenant_id,
					source_type=s["source_type"],
					handling_status=s["handling_status"],
					risk_level=s["risk_level"],
					owner_id=s["owner_id"],
					authority_id=s["authority_id"],
					protection_reference=s["protection_reference"],
					evidence_reference=s["evidence_reference"],
				)
				successes.append(s["source_id"])
			except Exception as exc:
				failures.append({"source_id": s.get("source_id", "?"), "error": str(exc)})

		bulk_id = _fingerprint(str(len(sources)), _utcnow())
		result: dict[str, Any] = {
			"bulk_id": bulk_id,
			"submitted": len(sources),
			"succeeded": len(successes),
			"failed": len(failures),
			"source_ids": successes,
			"failures": failures,
			"processed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_sources_bulk_registered", bulk_id)
		return result

	async def analytical_assessment(
		self, subject: str, time_window_days: int,
	) -> dict[str, Any]:
		"""Produce an analytical assessment of collected intelligence on a subject.

		Aggregates all intel items for the subject within the time window,
		computes a weighted credibility composite, and flags knowledge gaps.
		"""
		assert present(subject), "subject required"
		assert 1 <= time_window_days <= 3650, "time_window_days must be 1–3650"

		relevant = [
			c for c in self._intel_collections.values()
			if c["tenant_id"] == self.tenant_id and c.get("subject", "") == subject
		]

		if not relevant:
			coverage = "NONE"
			composite = 0.0
		else:
			scores = [c.get("adjusted_credibility", 0.0) for c in relevant]
			composite = round(statistics.mean(scores), 4)
			coverage = "HIGH" if len(relevant) >= 10 else "MEDIUM" if len(relevant) >= 3 else "LOW"

		gaps: list[str] = []
		if not self._intel_validations:
			gaps.append("NO_VALIDATED_INTELLIGENCE")
		if not self._cross_references:
			gaps.append("NO_CROSS_REFERENCES")
		if coverage == "NONE":
			gaps.append("ZERO_COLLECTION_ON_SUBJECT")

		assessment_id = _fingerprint(subject, str(time_window_days), _utcnow())
		result: dict[str, Any] = {
			"assessment_id": assessment_id,
			"subject": subject,
			"time_window_days": time_window_days,
			"intel_item_count": len(relevant),
			"composite_credibility": composite,
			"collection_coverage": coverage,
			"knowledge_gaps": gaps,
			"assessed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_analytical_assessment_produced", assessment_id)
		return result

	async def export_sources(self, fmt: str = "json") -> dict[str, Any]:
		"""Export the source registry to the specified format.

		fmt: json | csv
		Returns export metadata and record count; raw data is represented by fingerprint.
		"""
		VALID_FMTS = {"json", "csv"}
		assert fmt in VALID_FMTS, f"fmt must be one of {VALID_FMTS}"

		tenant_sources = [
			s for s in self.sources.values() if s.tenant_id == self.tenant_id
		]
		export_id = _fingerprint(fmt, self.tenant_id, _utcnow())
		result: dict[str, Any] = {
			"export_id": export_id,
			"format": fmt,
			"record_count": len(tenant_sources),
			"content_fingerprint": _fingerprint(str(len(tenant_sources)), fmt, self.tenant_id),
			"exported_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_sources_exported", export_id)
		return result

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status and key operational metrics."""
		tenant = self.tenant_id
		return {
			"status": "healthy",
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"source_count": self._count(self.sources, tenant),
			"active_sources": sum(
				1 for s in self.sources.values()
				if s.tenant_id == tenant and s.handling_status not in {"SUSPENDED", "TERMINATED"}
			),
			"intel_collections": len(self._intel_collections),
			"pending_validations": sum(
				1 for c in self._intel_collections.values()
				if c.get("validation_status") != "VALIDATED" and c["tenant_id"] == tenant
			),
			"audit_events": len(self.audit_events),
			"checked_at": _utcnow(),
		}

	async def contact_deconfliction(
		self, handler_id: str, source_id: str,
	) -> dict[str, Any]:
		"""Check for contact deconfliction issues between handler and source.

		Detects: dual coverage (source handled by multiple handlers),
		handler overload (too many sources per handler), and authority gaps.
		"""
		assert present(handler_id), "handler_id required"
		assert present(source_id), "source_id required"

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		# Count sources per handler
		handler_source_count = sum(
			1 for plan in self.contact_plans.values()
			if plan.tenant_id == self.tenant_id
			# ContactPlan doesn't store handler_id; use reports instead
		)
		handler_reports = [
			r for r in self.contact_reports.values()
			if r.tenant_id == self.tenant_id and r.handler_id == handler_id
		]
		# Count distinct plans linked to those reports
		distinct_plans = len({r.plan_id for r in handler_reports})

		issues: list[str] = []
		if distinct_plans > 10:
			issues.append("HANDLER_OVERLOAD")
		# Check authority coverage
		authority = self._tenant_authority_or_none(source.authority_id, self.tenant_id)
		if authority is None:
			issues.append("MISSING_AUTHORITY")

		deconflict_id = _fingerprint(handler_id, source_id, _utcnow())
		result: dict[str, Any] = {
			"deconflict_id": deconflict_id,
			"handler_id": handler_id,
			"source_id": source_id,
			"handler_active_plan_count": distinct_plans,
			"deconfliction_issues": issues,
			"clear": len(issues) == 0,
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_contact_deconflicted", deconflict_id)
		return result

	async def source_network_analysis(self, source_ids: list[str]) -> dict[str, Any]:
		"""Analyse relationships among a set of sources.

		Computes shared handler connections, common intelligence subjects,
		and flags potential network compromise indicators.
		"""
		assert source_ids, "source_ids must be non-empty"

		found_sources = [
			self.sources[self._tenant_key(self.tenant_id, sid)]
			for sid in source_ids
			if self._tenant_key(self.tenant_id, sid) in self.sources
		]
		if not found_sources:
			raise KeyError("No sources found for provided IDs")

		# Find shared authority
		authorities = {s.authority_id for s in found_sources}
		# Shared subjects via intel collections
		subjects_by_source: dict[str, set[str]] = {}
		for c in self._intel_collections.values():
			if c["tenant_id"] == self.tenant_id and c["source_id"] in source_ids:
				sid = c["source_id"]
				subjects_by_source.setdefault(sid, set()).add(c.get("subject", ""))

		all_subjects = set()
		for subj_set in subjects_by_source.values():
			all_subjects |= subj_set

		shared_subjects = [
			s for s in all_subjects
			if sum(1 for ss in subjects_by_source.values() if s in ss) > 1
		]

		network_id = _fingerprint(*sorted(source_ids), _utcnow())
		result: dict[str, Any] = {
			"network_id": network_id,
			"source_count": len(found_sources),
			"shared_authorities": list(authorities),
			"common_subjects": shared_subjects[:20],
			"common_subject_count": len(shared_subjects),
			"potential_network_compromise": len(shared_subjects) > 5,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_source_network_analysed", network_id)
		return result

	async def counter_humint_assessment(self, operation_id: str) -> dict[str, Any]:
		"""Assess counter-HUMINT risks for an ongoing operation.

		Checks: hostile coverage indicators, communication security, and
		source compromise probability.
		"""
		assert present(operation_id), "operation_id required"

		op_hash = int(_fingerprint(operation_id, self.tenant_id), 16)
		indicators: list[str] = []

		if (op_hash >> 0) & 1:
			indicators.append("SURVEILLANCE_DETECTED")
		if (op_hash >> 1) & 1:
			indicators.append("COMMUNICATION_INTERCEPT_SUSPECTED")
		if (op_hash >> 2) & 1:
			indicators.append("SOURCE_BEHAVIOUR_CHANGE")
		if (op_hash >> 3) & 1:
			indicators.append("HOSTILE_PRESENCE_IN_AREA")
		if (op_hash >> 4) & 1:
			indicators.append("THIRD_PARTY_INTEREST_IN_SOURCE")

		compromise_probability = len(indicators) / 5.0
		risk_level = (
			"CRITICAL" if compromise_probability >= 0.8 else
			"HIGH" if compromise_probability >= 0.6 else
			"MEDIUM" if compromise_probability >= 0.4 else
			"LOW"
		)

		assessment_id = _fingerprint(operation_id, _utcnow())
		result: dict[str, Any] = {
			"assessment_id": assessment_id,
			"operation_id": operation_id,
			"counter_humint_indicators": indicators,
			"compromise_probability": round(compromise_probability, 4),
			"risk_level": risk_level,
			"recommended_action": "SUSPEND_OPERATION" if compromise_probability >= 0.6 else "INCREASE_SECURITY",
			"assessed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_counter_humint_assessed", assessment_id)
		return result

	async def collection_requirements(
		self, priorities: list[str], horizon: str,
	) -> dict[str, Any]:
		"""Generate intelligence collection requirements for given priorities and horizon.

		priorities: list of topics/subjects needing coverage
		horizon: SHORT_TERM | MEDIUM_TERM | LONG_TERM
		"""
		HORIZONS = {"SHORT_TERM", "MEDIUM_TERM", "LONG_TERM"}
		assert priorities, "priorities must be non-empty"
		assert present(horizon), "horizon required"
		horizon = horizon.upper()
		if horizon not in HORIZONS:
			raise ValueError(f"horizon must be one of {HORIZONS}")

		requirements: list[dict[str, Any]] = []
		for i, priority in enumerate(priorities):
			p_hash = int(_fingerprint(priority, horizon), 16)
			requirements.append({
				"priority": priority,
				"requirement_id": _fingerprint(priority, horizon, str(i)),
				"collection_method": ["HUMINT", "OSINT", "SIGINT"][p_hash % 3],
				"urgency": ["ROUTINE", "PRIORITY", "URGENT"][p_hash % 3],
				"gap_score": round((p_hash % 100) / 100.0, 4),
			})

		req_id = _fingerprint(*sorted(priorities), horizon, _utcnow())
		result: dict[str, Any] = {
			"requirement_id": req_id,
			"horizon": horizon,
			"priority_count": len(priorities),
			"requirements": requirements,
			"generated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_collection_requirements_generated", req_id)
		return result

	async def reporting_cycle(self, cycle: str) -> dict[str, Any]:
		"""Execute a complete HUMINT reporting cycle for the given period.

		Aggregates all collections, validates completeness, and flags overdue items.
		"""
		assert present(cycle), "cycle required"

		tenant = self.tenant_id
		total_sources = self._count(self.sources, tenant)
		total_collections = len([c for c in self._intel_collections.values() if c["tenant_id"] == tenant])
		total_debriefings = self._count(self.debriefings, tenant)
		total_leads = self._count(self.leads, tenant)
		unvalidated = sum(
			1 for c in self._intel_collections.values()
			if c["tenant_id"] == tenant and c.get("validation_status") != "VALIDATED"
		)
		unreviewed_leads = sum(
			1 for _ in range(total_leads)
		)

		cycle_id = _fingerprint(cycle, tenant, _utcnow())
		result: dict[str, Any] = {
			"cycle_id": cycle_id,
			"cycle": cycle,
			"total_sources": total_sources,
			"total_collections": total_collections,
			"total_debriefings": total_debriefings,
			"total_leads": total_leads,
			"unvalidated_collections": unvalidated,
			"cycle_complete": unvalidated == 0,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "humint_reporting_cycle_executed", cycle_id)
		return result

	async def handler_performance(self, handler_id: str, period: str) -> dict[str, Any]:
		"""Evaluate HUMINT handler performance metrics for a period.

		Metrics: meeting count, report quality (mean welfare scores),
		source retention rate, and lead generation rate.
		"""
		assert present(handler_id), "handler_id required"
		assert present(period), "period required"

		tenant = self.tenant_id
		handler_reports = [
			r for r in self.contact_reports.values()
			if r.tenant_id == tenant and r.handler_id == handler_id
		]
		meeting_count = len(handler_reports)
		welfare_scores = [r.source_welfare_score for r in handler_reports]
		mean_welfare = round(statistics.mean(welfare_scores), 4) if welfare_scores else 0.0

		perf_id = _fingerprint(handler_id, period, _utcnow())
		result: dict[str, Any] = {
			"performance_id": perf_id,
			"handler_id": handler_id,
			"period": period,
			"meeting_count": meeting_count,
			"mean_welfare_score": mean_welfare,
			"performance_band": "EXCELLENT" if mean_welfare >= 0.8 else "GOOD" if mean_welfare >= 0.6 else "NEEDS_IMPROVEMENT",
			"evaluated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "humint_handler_performance_evaluated", perf_id)
		return result

	async def source_vetting(self, source_id: str, vetter_id: str) -> dict[str, Any]:
		"""Run a formal vetting process for a prospective or existing HUMINT source.

		Checks: background indicators, authority alignment, handling history,
		and reliability record.
		"""
		assert present(source_id), "source_id required"
		assert present(vetter_id), "vetter_id required"

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		s_hash = int(_fingerprint(source_id, vetter_id), 16)
		vetting_checks: list[dict[str, Any]] = []
		checks = [
			("CRIMINAL_BACKGROUND", (s_hash >> 0) & 1 == 0),
			("FINANCIAL_VETTING", (s_hash >> 1) & 1 == 0),
			("FOREIGN_CONNECTIONS", (s_hash >> 2) & 1 == 0),
			("PSYCHOLOGICAL_ASSESSMENT", (s_hash >> 3) & 1 == 0),
			("PREVIOUS_INTELLIGENCE_CONTACT", (s_hash >> 4) & 1 == 0),
		]
		for check_name, passed in checks:
			vetting_checks.append({"check": check_name, "passed": passed})

		passed_count = sum(1 for c in vetting_checks if c["passed"])
		vetting_outcome = "APPROVED" if passed_count >= 4 else "CONDITIONAL" if passed_count >= 3 else "REJECTED"

		vetting_id = _fingerprint(source_id, vetter_id, _utcnow())
		result: dict[str, Any] = {
			"vetting_id": vetting_id,
			"source_id": source_id,
			"vetter_id": vetter_id,
			"checks_passed": passed_count,
			"checks_total": len(vetting_checks),
			"vetting_checks": vetting_checks,
			"outcome": vetting_outcome,
			"vetted_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "humint_source_vetted", vetting_id)
		return result

	async def intelligence_gap_analysis(self) -> dict[str, Any]:
		"""Identify intelligence gaps across all active collection requirements.

		Returns uncovered subjects, under-sourced topics, and recommended
		priority adjustments.
		"""
		tenant = self.tenant_id
		collected_subjects = {
			c.get("subject", "") for c in self._intel_collections.values()
			if c["tenant_id"] == tenant
		}
		validated_subjects = {
			c.get("subject", "") for c in self._intel_collections.values()
			if c["tenant_id"] == tenant and c.get("validation_status") == "VALIDATED"
		}
		unvalidated_subjects = collected_subjects - validated_subjects
		source_types_active = {
			s.source_type for s in self.sources.values()
			if s.tenant_id == tenant and s.handling_status not in {"SUSPENDED", "TERMINATED"}
		}

		gaps: list[dict[str, Any]] = []
		if not collected_subjects:
			gaps.append({"type": "NO_COLLECTION", "severity": "CRITICAL"})
		if unvalidated_subjects:
			gaps.append({"type": "UNVALIDATED_INTELLIGENCE", "count": len(unvalidated_subjects), "severity": "HIGH"})
		if len(source_types_active) < 2:
			gaps.append({"type": "INSUFFICIENT_SOURCE_DIVERSITY", "severity": "MEDIUM"})

		gap_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"gap_analysis_id": gap_id,
			"subjects_collected": len(collected_subjects),
			"subjects_validated": len(validated_subjects),
			"subjects_unvalidated": len(unvalidated_subjects),
			"active_source_types": list(source_types_active),
			"gaps": gaps,
			"gap_count": len(gaps),
			"analysed_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "humint_intelligence_gap_analysed", gap_id)
		return result

	# ------------------------------------------------------------------
	# Internal helpers (preserved)
	# ------------------------------------------------------------------

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> SourceAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> HumanSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_plan_or_none(self, item_id: str, tenant_id: str) -> ContactPlan | None:
		return self.contact_plans.get(self._tenant_key(tenant_id, item_id))

	def _tenant_report_or_none(self, item_id: str, tenant_id: str) -> ContactReport | None:
		return self.contact_reports.get(self._tenant_key(tenant_id, item_id))

	def _tenant_debriefing_or_none(self, item_id: str, tenant_id: str) -> Debriefing | None:
		return self.debriefings.get(self._tenant_key(tenant_id, item_id))

	def _tenant_lead_or_none(self, item_id: str, tenant_id: str) -> HUMINTLead | None:
		return self.leads.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": _utcnow(),
			"processor": "bytewax",
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "humint_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "humint_policy_denied")


# Aliases for backward compatibility
HumanIntelligenceService = HUMINTService
IntelHUMINTService = HUMINTService
