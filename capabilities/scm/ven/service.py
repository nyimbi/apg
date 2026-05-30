"""Dependency-light SCM Vendor Management lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		STREAMING,
		SUPPORTED_COMPLIANCE_STATUSES,
		SUPPORTED_PERFORMANCE_DIMENSIONS,
		SUPPORTED_RISK_TIERS,
		SUPPORTED_VENDOR_AGENT_ROLES,
		SUPPORTED_VENDOR_AGENT_RUNTIMES,
		SUPPORTED_VENDOR_TYPES,
		VENDOR_EVENT_STREAM,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		STREAMING,
		SUPPORTED_COMPLIANCE_STATUSES,
		SUPPORTED_PERFORMANCE_DIMENSIONS,
		SUPPORTED_RISK_TIERS,
		SUPPORTED_VENDOR_AGENT_ROLES,
		SUPPORTED_VENDOR_AGENT_RUNTIMES,
		SUPPORTED_VENDOR_TYPES,
		VENDOR_EVENT_STREAM,
		evaluate_capability_rules,
		get_capability_contract,
	)


class VendorManagementError(Exception):
	"""Base exception for vendor operations."""


class VendorNotFoundError(VendorManagementError):
	"""Raised when a vendor record is not found."""


class VendorManagementLifecycleService:
	"""In-memory executable service for Vendor Management lifecycle packets."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.vendors: dict[str, dict[str, Any]] = {}
		self.qualifications: dict[str, dict[str, Any]] = {}
		self.onboarding: dict[str, dict[str, Any]] = {}
		self.performance: dict[str, dict[str, Any]] = {}
		self.risks: dict[str, dict[str, Any]] = {}
		self.compliance: dict[str, dict[str, Any]] = {}
		self.contracts: dict[str, dict[str, Any]] = {}
		self.communications: dict[str, dict[str, Any]] = {}
		self.portal_users: dict[str, dict[str, Any]] = {}
		self.scorecards: dict[str, dict[str, Any]] = {}
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
		self._audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "record_id": record["id"], "record_type": record["type"], "status": record["status"], "stream": VENDOR_EVENT_STREAM, "processor": "bytewax", "emitted_at": self._now()})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_vendor(self, vendor_id: str, tenant_id: str, code: str, name: str, vendor_type: str, category: str, country: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_vendor")
		context.update({"code_present": bool(code), "name_present": bool(name), "vendor_type_supported": vendor_type in SUPPORTED_VENDOR_TYPES, "category_present": bool(category), "country_present": bool(country), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("vendor", vendor_id), "type": "vendor_profile", "kind": "vendor", "tenant_id": tenant, "code": code.upper(), "name": name, "vendor_type": vendor_type, "category": category, "country": country, "owner_id": owner_id, "stage": "prospect", "status": "active", "created_at": self._now()}
		self.vendors[record["id"]] = record
		self._emit(tenant, "vendor_created", record)
		return deepcopy(record)

	def qualify_vendor(self, qualification_id: str, tenant_id: str, vendor_id: str, criteria: list[str], qualified_by: str, score: float | None, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		score_value = float(score) if score is not None else None
		context = self._base_context(tenant, "qualify_vendor")
		context.update({"vendor_present": bool(vendor), "criteria_present": bool(criteria), "qualified_by_present": bool(qualified_by), "score_present": score is not None, "score_below_threshold": bool(score_value is not None and score_value < 70), "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("qualification", qualification_id), "type": "vendor_qualification", "kind": "qualification", "tenant_id": tenant, "vendor_id": vendor_id, "criteria": list(criteria), "qualified_by": qualified_by, "reviewed_by": reviewed_by, "score": score_value, "status": "qualified", "created_at": self._now()}
		self.qualifications[record["id"]] = record
		vendor["stage"] = "qualified"
		self._emit(tenant, "vendor_qualified", record)
		return deepcopy(record)

	def onboard_vendor(self, onboarding_id: str, tenant_id: str, vendor_id: str, checklist: list[str], owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "onboard_vendor")
		context.update({"vendor_present": bool(vendor), "checklist_present": bool(checklist), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("onboarding", onboarding_id), "type": "vendor_onboarding", "kind": "onboarding", "tenant_id": tenant, "vendor_id": vendor_id, "checklist": list(checklist), "owner_id": owner_id, "status": "complete", "created_at": self._now()}
		self.onboarding[record["id"]] = record
		vendor["stage"] = "active"
		self._emit(tenant, "vendor_onboarded", record)
		return deepcopy(record)

	def record_performance(self, performance_id: str, tenant_id: str, vendor_id: str, period: str, scores: dict[str, float], reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		values = [float(value) for value in scores.values()]
		dimensions_supported = all(dimension in SUPPORTED_PERFORMANCE_DIMENSIONS for dimension in scores)
		scores_in_range = all(0 <= value <= 100 for value in values)
		average_score = round(sum(values) / len(values), 2) if values else 0.0
		context = self._base_context(tenant, "record_performance")
		context.update({"vendor_present": bool(vendor), "period_present": bool(period), "dimensions_supported": dimensions_supported, "scores_in_range": scores_in_range, "low_score": average_score < 60, "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("performance", performance_id), "type": "vendor_performance", "kind": "performance", "tenant_id": tenant, "vendor_id": vendor_id, "period": period, "scores": dict(scores), "average_score": average_score, "reviewed_by": reviewed_by, "status": "recorded", "created_at": self._now()}
		self.performance[record["id"]] = record
		self._emit(tenant, "vendor_performance_recorded", record)
		return deepcopy(record)

	def record_risk(self, risk_id: str, tenant_id: str, vendor_id: str, risk_type: str, tier: str, description: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "record_risk")
		context.update({"vendor_present": bool(vendor), "risk_tier_supported": tier in SUPPORTED_RISK_TIERS, "description_present": bool(description), "high_or_critical": tier in {"high", "critical"}, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("risk", risk_id), "type": "vendor_risk", "kind": "risk", "tenant_id": tenant, "vendor_id": vendor_id, "risk_type": risk_type, "tier": tier, "description": description, "owner_id": owner_id, "status": "open", "created_at": self._now()}
		self.risks[record["id"]] = record
		self._emit(tenant, "vendor_risk_recorded", record)
		return deepcopy(record)

	def record_compliance(self, compliance_id: str, tenant_id: str, vendor_id: str, framework: str, status: str, evidence_id: str, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "record_compliance")
		context.update({"vendor_present": bool(vendor), "framework_present": bool(framework), "status_supported": status in SUPPORTED_COMPLIANCE_STATUSES, "evidence_present": bool(evidence_id), "review_required": status in {"non_compliant", "expired"}, "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("compliance", compliance_id), "type": "vendor_compliance", "kind": "compliance", "tenant_id": tenant, "vendor_id": vendor_id, "framework": framework, "status_value": status, "evidence_id": evidence_id, "reviewed_by": reviewed_by, "status": "recorded", "created_at": self._now()}
		self.compliance[record["id"]] = record
		self._emit(tenant, "vendor_compliance_recorded", record)
		return deepcopy(record)

	def create_contract(self, contract_id: str, tenant_id: str, vendor_id: str, value: float | None, currency: str, start_date: str, end_date: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "create_contract")
		context.update({"vendor_present": bool(vendor), "value_present": value is not None, "currency_present": bool(currency), "date_range_present": bool(start_date and end_date), "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("contract", contract_id), "type": "vendor_contract", "kind": "contract", "tenant_id": tenant, "vendor_id": vendor_id, "value": float(value), "currency": currency, "start_date": start_date, "end_date": end_date, "approved_by": approved_by, "status": "active", "created_at": self._now()}
		self.contracts[record["id"]] = record
		self._emit(tenant, "vendor_contract_created", record)
		return deepcopy(record)

	def record_communication(self, communication_id: str, tenant_id: str, vendor_id: str, channel: str, subject: str, sentiment: str = "neutral", owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "record_communication")
		context.update({"vendor_present": bool(vendor), "channel_present": bool(channel), "subject_present": bool(subject), "negative_sentiment": sentiment == "negative", "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("communication", communication_id), "type": "vendor_communication", "kind": "communication", "tenant_id": tenant, "vendor_id": vendor_id, "channel": channel, "subject": subject, "sentiment": sentiment, "owner_id": owner_id, "status": "recorded", "created_at": self._now()}
		self.communications[record["id"]] = record
		self._emit(tenant, "vendor_communication_recorded", record)
		return deepcopy(record)

	def create_portal_user(self, portal_user_id: str, tenant_id: str, vendor_id: str, email: str, role: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "create_portal_user")
		context.update({"vendor_present": bool(vendor), "email_present": bool(email), "role_present": bool(role), "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("portal", portal_user_id), "type": "vendor_portal_user", "kind": "portal_user", "tenant_id": tenant, "vendor_id": vendor_id, "email": email, "role": role, "approved_by": approved_by, "status": "active", "created_at": self._now()}
		self.portal_users[record["id"]] = record
		self._emit(tenant, "vendor_portal_user_created", record)
		return deepcopy(record)

	def create_scorecard(self, scorecard_id: str, tenant_id: str, vendor_id: str, period: str, performance_id: str, risk_id: str, compliance_ids: list[str], generated_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		performance = self.performance.get(performance_id)
		risk = self.risks.get(risk_id)
		context = self._base_context(tenant, "create_scorecard")
		context.update({"vendor_present": bool(vendor), "performance_present": bool(performance and performance["tenant_id"] == tenant), "risk_present": bool(risk and risk["tenant_id"] == tenant), "generator_present": bool(generated_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("scorecard", scorecard_id), "type": "vendor_scorecard", "kind": "scorecard", "tenant_id": tenant, "vendor_id": vendor_id, "period": period, "performance_id": performance_id, "risk_id": risk_id, "compliance_ids": list(compliance_ids), "generated_by": generated_by, "overall_score": performance["average_score"], "risk_tier": risk["tier"], "status": "published", "created_at": self._now()}
		self.scorecards[record["id"]] = record
		self._emit(tenant, "vendor_scorecard_created", record)
		return deepcopy(record)

	def register_vendor_agent(self, tenant_id: str, name: str, runtime: str, role: str, purpose: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_vendor_agent")
		context.update({"runtime_supported": runtime in SUPPORTED_VENDOR_AGENT_RUNTIMES, "role_supported": role in SUPPORTED_VENDOR_AGENT_ROLES})
		self._assert_rules(context)
		record = {"id": self._record_id("agent"), "type": "vendor_agent", "kind": "agent", "tenant_id": tenant, "name": name, "runtime": runtime, "role": role, "purpose": purpose, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.agents[record["id"]] = record
		self._emit(tenant, "vendor_agent_registered", record)
		return deepcopy(record)

	def validate_vendor_agent_action(self, tenant_id: str, privileged_action: bool, human_approved: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return evaluate_capability_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "agent_action", "privileged_action": privileged_action, "human_approved": human_approved})

	def validate_batch(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if event_stream != "bytewax":
			self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "vendor_batch", "event_stream": "queue"})
		return {"tenant_id": tenant, "record_count": int(record_count), "processor": "bytewax", "event_stream": VENDOR_EVENT_STREAM, "accepted": True}

	def create_record(self, payload: dict[str, Any]) -> dict[str, Any]:
		tenant = self._tenant(payload.get("tenant_id"))
		record = {"id": self._record_id("record", payload.get("id")), "type": payload.get("type", "vendor_record"), "kind": payload.get("kind", "generic"), "tenant_id": tenant, "status": payload.get("status", "active"), "created_at": self._now(), **payload}
		self._emit(tenant, "vendor_record_created", record)
		return deepcopy(record)

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		def count(records: dict[str, dict[str, Any]]) -> int:
			return sum(1 for record in records.values() if record["tenant_id"] == tenant)
		return {"tenant_id": tenant, "vendor_count": count(self.vendors), "qualification_count": count(self.qualifications), "onboarding_count": count(self.onboarding), "performance_count": count(self.performance), "risk_count": count(self.risks), "compliance_count": count(self.compliance), "contract_count": count(self.contracts), "communication_count": count(self.communications), "portal_user_count": count(self.portal_users), "scorecard_count": count(self.scorecards), "agent_count": count(self.agents), "audit_event_count": sum(1 for event in self._audit_events if event["tenant_id"] == tenant), "streaming": deepcopy(STREAMING)}

	def list_records(self, tenant_id: str, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		stores = [self.vendors, self.qualifications, self.onboarding, self.performance, self.risks, self.compliance, self.contracts, self.communications, self.portal_users, self.scorecards, self.agents]
		records = [record for store in stores for record in store.values() if record["tenant_id"] == tenant]
		if record_type:
			records = [record for record in records if record["type"] == record_type or record["kind"] == record_type]
		return deepcopy(records)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return deepcopy([event for event in self._audit_events if event["tenant_id"] == tenant])

	def _get_vendor(self, vendor_id: str, tenant_id: str) -> dict[str, Any]:
		vendor = self.vendors.get(vendor_id)
		if not vendor or vendor["tenant_id"] != tenant_id:
			raise VendorNotFoundError("vendor_not_found")
		return vendor


VendorManagementService = VendorManagementLifecycleService
VendorLifecycleService = VendorManagementLifecycleService
VendorRiskService = VendorManagementLifecycleService
VendorPerformanceService = VendorManagementLifecycleService
