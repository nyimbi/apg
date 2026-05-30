"""Dependency-light Sustainability and ESG lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		ESG_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ESG_AGENT_ROLES,
		SUPPORTED_ESG_AGENT_RUNTIMES,
		SUPPORTED_FRAMEWORKS,
		SUPPORTED_MEASUREMENT_SOURCES,
		SUPPORTED_METRIC_TYPES,
		SUPPORTED_PILLARS,
		SUPPORTED_REPORT_TYPES,
		SUPPORTED_RISK_TIERS,
		SUPPORTED_TARGET_TYPES,
		SUPPORTED_UNITS,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		ESG_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ESG_AGENT_ROLES,
		SUPPORTED_ESG_AGENT_RUNTIMES,
		SUPPORTED_FRAMEWORKS,
		SUPPORTED_MEASUREMENT_SOURCES,
		SUPPORTED_METRIC_TYPES,
		SUPPORTED_PILLARS,
		SUPPORTED_REPORT_TYPES,
		SUPPORTED_RISK_TIERS,
		SUPPORTED_TARGET_TYPES,
		SUPPORTED_UNITS,
		evaluate_capability_rules,
		get_capability_contract,
	)


class ESGManagementError(Exception):
	"""Base exception for ESG operations."""


class ESGRecordNotFoundError(ESGManagementError):
	"""Raised when an ESG lifecycle record is not found."""


class ESGManagementLifecycleService:
	"""In-memory executable service for ESG lifecycle packets."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.profiles: dict[str, dict[str, Any]] = {}
		self.frameworks: dict[str, dict[str, Any]] = {}
		self.metrics: dict[str, dict[str, Any]] = {}
		self.measurements: dict[str, dict[str, Any]] = {}
		self.targets: dict[str, dict[str, Any]] = {}
		self.supplier_assessments: dict[str, dict[str, Any]] = {}
		self.initiatives: dict[str, dict[str, Any]] = {}
		self.risks: dict[str, dict[str, Any]] = {}
		self.reports: dict[str, dict[str, Any]] = {}
		self.stakeholders: dict[str, dict[str, Any]] = {}
		self.engagements: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "tenant_context_present": True, "operation": operation, "operation_type": "write", "policy_attached": True, "audit_enabled": True}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "record_id": record["id"], "record_type": record["type"], "status": record["status"], "stream": ESG_EVENT_STREAM, "processor": "bytewax", "emitted_at": self._now()})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_esg_profile(self, profile_id: str, tenant_id: str, name: str, industry: str, country: str, reporting_year: int | None, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_esg_profile")
		context.update({"name_present": bool(name), "industry_present": bool(industry), "country_present": bool(country), "reporting_year_present": reporting_year is not None, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("profile", profile_id), "type": "esg_profile", "kind": "profile", "tenant_id": tenant, "name": name, "industry": industry, "country": country, "reporting_year": int(reporting_year), "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.profiles[record["id"]] = record
		self._emit(tenant, "esg_profile_created", record)
		return deepcopy(record)

	def add_framework(self, framework_id: str, tenant_id: str, profile_id: str, code: str, version: str, mandatory: bool, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		profile = self._get(self.profiles, profile_id, tenant, "profile")
		context = self._base_context(tenant, "add_framework")
		context.update({"profile_present": bool(profile), "framework_supported": code in SUPPORTED_FRAMEWORKS, "version_present": bool(version), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("framework", framework_id), "type": "esg_framework", "kind": "framework", "tenant_id": tenant, "profile_id": profile_id, "code": code, "version": version, "mandatory": bool(mandatory), "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.frameworks[record["id"]] = record
		self._emit(tenant, "esg_framework_added", record)
		return deepcopy(record)

	def define_metric(self, metric_id: str, tenant_id: str, profile_id: str, pillar: str, metric_type: str, unit: str, name: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		profile = self._get(self.profiles, profile_id, tenant, "profile")
		context = self._base_context(tenant, "define_metric")
		context.update({"profile_present": bool(profile), "pillar_supported": pillar in SUPPORTED_PILLARS, "metric_type_supported": metric_type in SUPPORTED_METRIC_TYPES, "unit_supported": unit in SUPPORTED_UNITS, "name_present": bool(name), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("metric", metric_id), "type": "esg_metric", "kind": "metric", "tenant_id": tenant, "profile_id": profile_id, "pillar": pillar, "metric_type": metric_type, "unit": unit, "name": name, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.metrics[record["id"]] = record
		self._emit(tenant, "esg_metric_defined", record)
		return deepcopy(record)

	def record_measurement(self, measurement_id: str, tenant_id: str, metric_id: str, period: str, value: float | None, source: str, evidence_id: str, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		metric = self._get(self.metrics, metric_id, tenant, "metric")
		context = self._base_context(tenant, "record_measurement")
		context.update({"metric_present": bool(metric), "period_present": bool(period), "value_present": value is not None, "source_supported": source in SUPPORTED_MEASUREMENT_SOURCES, "evidence_present": bool(evidence_id), "review_required": source in {"supplier", "calculation"}, "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("measurement", measurement_id), "type": "esg_measurement", "kind": "measurement", "tenant_id": tenant, "metric_id": metric_id, "period": period, "value": float(value), "source": source, "evidence_id": evidence_id, "reviewed_by": reviewed_by, "unit": metric["unit"], "status": "recorded", "created_at": self._now()}
		self.measurements[record["id"]] = record
		self._emit(tenant, "esg_measurement_recorded", record)
		return deepcopy(record)

	def set_target(self, target_id: str, tenant_id: str, metric_id: str, target_type: str, baseline_value: float | None, target_value: float | None, due_date: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		metric = self._get(self.metrics, metric_id, tenant, "metric")
		context = self._base_context(tenant, "set_target")
		context.update({"metric_present": bool(metric), "target_type_supported": target_type in SUPPORTED_TARGET_TYPES, "baseline_present": baseline_value is not None, "target_present": target_value is not None, "due_date_present": bool(due_date), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("target", target_id), "type": "esg_target", "kind": "target", "tenant_id": tenant, "metric_id": metric_id, "target_type": target_type, "baseline_value": float(baseline_value), "target_value": float(target_value), "due_date": due_date, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.targets[record["id"]] = record
		self._emit(tenant, "esg_target_set", record)
		return deepcopy(record)

	def record_supplier_assessment(self, assessment_id: str, tenant_id: str, supplier_id: str, period: str, score: float, risk_tier: str, evidence_id: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		score_value = float(score)
		context = self._base_context(tenant, "record_supplier_assessment")
		context.update({"supplier_present": bool(supplier_id), "period_present": bool(period), "score_in_range": 0 <= score_value <= 100, "evidence_present": bool(evidence_id), "high_risk": risk_tier in {"high", "critical"}, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("supplier", assessment_id), "type": "esg_supplier_assessment", "kind": "supplier_assessment", "tenant_id": tenant, "supplier_id": supplier_id, "period": period, "score": score_value, "risk_tier": risk_tier, "evidence_id": evidence_id, "owner_id": owner_id, "status": "recorded", "created_at": self._now()}
		self.supplier_assessments[record["id"]] = record
		self._emit(tenant, "esg_supplier_assessed", record)
		return deepcopy(record)

	def record_initiative(self, initiative_id: str, tenant_id: str, profile_id: str, name: str, pillar: str, budget: float, owner_id: str, expected_impact: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		profile = self._get(self.profiles, profile_id, tenant, "profile")
		context = self._base_context(tenant, "record_initiative")
		context.update({"profile_present": bool(profile), "name_present": bool(name), "pillar_supported": pillar in SUPPORTED_PILLARS, "owner_present": bool(owner_id), "impact_present": bool(expected_impact)})
		self._assert_rules(context)
		record = {"id": self._record_id("initiative", initiative_id), "type": "esg_initiative", "kind": "initiative", "tenant_id": tenant, "profile_id": profile_id, "name": name, "pillar": pillar, "budget": float(budget), "owner_id": owner_id, "expected_impact": expected_impact, "status": "active", "created_at": self._now()}
		self.initiatives[record["id"]] = record
		self._emit(tenant, "esg_initiative_recorded", record)
		return deepcopy(record)

	def record_risk(self, risk_id: str, tenant_id: str, profile_id: str, tier: str, category: str, description: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		profile = self._get(self.profiles, profile_id, tenant, "profile")
		context = self._base_context(tenant, "record_risk")
		context.update({"profile_present": bool(profile), "risk_tier_supported": tier in SUPPORTED_RISK_TIERS, "description_present": bool(description), "high_or_critical": tier in {"high", "critical"}, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("risk", risk_id), "type": "esg_risk", "kind": "risk", "tenant_id": tenant, "profile_id": profile_id, "tier": tier, "category": category, "description": description, "owner_id": owner_id, "status": "open", "created_at": self._now()}
		self.risks[record["id"]] = record
		self._emit(tenant, "esg_risk_recorded", record)
		return deepcopy(record)

	def create_report(self, report_id: str, tenant_id: str, profile_id: str, report_type: str, period: str, framework_ids: list[str], measurement_ids: list[str], approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		profile = self._get(self.profiles, profile_id, tenant, "profile")
		frameworks = [self.frameworks.get(item) for item in framework_ids]
		measurements = [self.measurements.get(item) for item in measurement_ids]
		valid_frameworks = [item for item in frameworks if item and item["tenant_id"] == tenant]
		valid_measurements = [item for item in measurements if item and item["tenant_id"] == tenant]
		context = self._base_context(tenant, "create_report")
		context.update({"profile_present": bool(profile), "report_type_supported": report_type in SUPPORTED_REPORT_TYPES, "frameworks_present": bool(framework_ids and len(valid_frameworks) == len(framework_ids)), "measurements_present": bool(measurement_ids and len(valid_measurements) == len(measurement_ids)), "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("report", report_id), "type": "esg_report", "kind": "report", "tenant_id": tenant, "profile_id": profile_id, "report_type": report_type, "period": period, "framework_ids": list(framework_ids), "measurement_ids": list(measurement_ids), "approved_by": approved_by, "status": "approved", "created_at": self._now()}
		self.reports[record["id"]] = record
		self._emit(tenant, "esg_report_created", record)
		return deepcopy(record)

	def register_stakeholder(self, stakeholder_id: str, tenant_id: str, profile_id: str, stakeholder_type: str, name: str, channel: str, consent_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		profile = self._get(self.profiles, profile_id, tenant, "profile")
		context = self._base_context(tenant, "register_stakeholder")
		context.update({"profile_present": bool(profile), "name_present": bool(name), "consent_recorded": bool(consent_recorded)})
		self._assert_rules(context)
		record = {"id": self._record_id("stakeholder", stakeholder_id), "type": "esg_stakeholder", "kind": "stakeholder", "tenant_id": tenant, "profile_id": profile_id, "stakeholder_type": stakeholder_type, "name": name, "channel": channel, "consent_recorded": bool(consent_recorded), "status": "active", "created_at": self._now()}
		self.stakeholders[record["id"]] = record
		self._emit(tenant, "esg_stakeholder_registered", record)
		return deepcopy(record)

	def record_engagement(self, engagement_id: str, tenant_id: str, stakeholder_id: str, topic: str, channel: str, sentiment: str = "neutral", owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		stakeholder = self._get(self.stakeholders, stakeholder_id, tenant, "stakeholder")
		context = self._base_context(tenant, "record_engagement")
		context.update({"stakeholder_present": bool(stakeholder), "topic_present": bool(topic), "negative_sentiment": sentiment == "negative", "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("engagement", engagement_id), "type": "esg_engagement", "kind": "engagement", "tenant_id": tenant, "stakeholder_id": stakeholder_id, "topic": topic, "channel": channel, "sentiment": sentiment, "owner_id": owner_id, "status": "recorded", "created_at": self._now()}
		self.engagements[record["id"]] = record
		self._emit(tenant, "esg_engagement_recorded", record)
		return deepcopy(record)

	def register_esg_agent(self, tenant_id: str, name: str, runtime: str, role: str, purpose: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_esg_agent")
		context.update({"runtime_supported": runtime in SUPPORTED_ESG_AGENT_RUNTIMES, "role_supported": role in SUPPORTED_ESG_AGENT_ROLES})
		self._assert_rules(context)
		record = {"id": self._record_id("agent"), "type": "esg_agent", "kind": "agent", "tenant_id": tenant, "name": name, "runtime": runtime, "role": role, "purpose": purpose, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.agents[record["id"]] = record
		self._emit(tenant, "esg_agent_registered", record)
		return deepcopy(record)

	def validate_esg_agent_action(self, tenant_id: str, privileged_action: bool, human_approved: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return evaluate_capability_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "agent_action", "privileged_action": privileged_action, "human_approved": human_approved})

	def validate_batch(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if event_stream != "bytewax":
			self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "esg_batch", "event_stream": "queue"})
		return {"tenant_id": tenant, "record_count": int(record_count), "processor": "bytewax", "event_stream": ESG_EVENT_STREAM, "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		def count(records: dict[str, dict[str, Any]]) -> int:
			return sum(1 for record in records.values() if record["tenant_id"] == tenant)
		return {"tenant_id": tenant, "profile_count": count(self.profiles), "framework_count": count(self.frameworks), "metric_count": count(self.metrics), "measurement_count": count(self.measurements), "target_count": count(self.targets), "supplier_assessment_count": count(self.supplier_assessments), "initiative_count": count(self.initiatives), "risk_count": count(self.risks), "report_count": count(self.reports), "stakeholder_count": count(self.stakeholders), "engagement_count": count(self.engagements), "agent_count": count(self.agents), "audit_event_count": sum(1 for event in self._audit_events if event["tenant_id"] == tenant), "streaming": deepcopy(STREAMING)}

	def list_records(self, tenant_id: str, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		stores = [self.profiles, self.frameworks, self.metrics, self.measurements, self.targets, self.supplier_assessments, self.initiatives, self.risks, self.reports, self.stakeholders, self.engagements, self.agents]
		records = [record for store in stores for record in store.values() if record["tenant_id"] == tenant]
		if record_type:
			records = [record for record in records if record["type"] == record_type or record["kind"] == record_type]
		return deepcopy(records)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return deepcopy([event for event in self._audit_events if event["tenant_id"] == tenant])

	def _get(self, store: dict[str, dict[str, Any]], record_id: str, tenant_id: str, label: str) -> dict[str, Any]:
		record = store.get(record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise ESGRecordNotFoundError(f"{label}_not_found")
		return record


ESGManagementService = ESGManagementLifecycleService
ESGService = ESGManagementLifecycleService
ESGReportingService = ESGManagementLifecycleService
ESGRiskService = ESGManagementLifecycleService
