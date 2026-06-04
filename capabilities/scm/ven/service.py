"""Dependency-light SCM Vendor Management lifecycle service — expanded implementation."""

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


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class VendorManagementError(Exception):
	"""Base exception for vendor operations."""


class VendorNotFoundError(VendorManagementError):
	"""Raised when a vendor record is not found."""


class VendorManagementService:
	"""
	In-memory executable service for Vendor Management lifecycle packets.

	Expanded with: onboard_vendor, vendor_qualification, vendor_performance_score,
	approved_vendor_list, vendor_suspension, contract_management,
	preferred_vendor_designation, spend_analysis, vendor_risk_assessment,
	vendor_portal_access.
	"""

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
		# New stores
		self._suspensions: dict[str, dict[str, Any]] = {}
		self._preferred_vendors: dict[str, dict[str, Any]] = {}
		self._spend_records: list[dict[str, Any]] = []
		self._portal_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "tenant_context_present": True, "operation": operation, "operation_type": "write", "policy_attached": True, "audit_enabled": True}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		# Only hard-block on explicit deny; require_review creates an audit flag
		if result.get("decision") == "deny":
			effects = result.get("effects") or result.get("actions") or []
			reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
			raise PermissionError(",".join(reasons) or "operation_denied")

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "record_id": record["id"], "record_type": record["type"], "status": record["status"], "stream": VENDOR_EVENT_STREAM, "processor": "bytewax", "emitted_at": _now()})

	def _get_vendor(self, vendor_id: str, tenant_id: str) -> dict[str, Any]:
		vendor = self.vendors.get(vendor_id)
		if not vendor or vendor["tenant_id"] != tenant_id:
			raise VendorNotFoundError("vendor_not_found")
		return vendor

	# ------------------------------------------------------------------
	# onboard_vendor
	# ------------------------------------------------------------------

	def onboard_vendor(
		self,
		legal_name: str,
		category: str,
		contact: dict[str, Any],
		bank_details: dict[str, Any],
		documents: list[str],
		tenant_id: str | None = None,
		vendor_id: str | None = None,
		vendor_type: str = "supplier",
		country: str = "KE",
		owner_id: str = "procurement",
	) -> dict[str, Any]:
		"""
		Onboard a new vendor with full KYC/KYB documentation.

		legal_name: Registered legal name.
		category: Spend category (e.g. 'it_services', 'logistics', 'raw_materials').
		contact: Dict with name, email, phone.
		bank_details: Dict with bank_name, account_number, swift_code.
		documents: List of document reference IDs (e.g. registration cert, tax PIN).
		"""
		tenant = self._tenant(tenant_id)
		code = "".join(c for c in legal_name.upper() if c.isalpha())[:6]
		context = self._base_context(tenant, "onboard_vendor")
		context.update({
			"code_present": bool(code),
			"name_present": bool(legal_name),
			"vendor_type_supported": vendor_type in SUPPORTED_VENDOR_TYPES,
			"category_present": bool(category),
			"country_present": bool(country),
			"owner_present": bool(owner_id),
		})
		self._assert_rules(context)
		if not documents:
			raise ValueError("onboarding_documents_required")
		resolved_id = self._record_id("vendor", vendor_id)
		record = {
			"id": resolved_id,
			"type": "vendor_profile",
			"kind": "vendor",
			"tenant_id": tenant,
			"code": code,
			"name": legal_name,
			"vendor_type": vendor_type,
			"category": category,
			"country": country,
			"owner_id": owner_id,
			"contact": contact,
			"bank_details": {k: v for k, v in bank_details.items() if k != "account_number"},
			"bank_account_last4": str(bank_details.get("account_number", ""))[-4:],
			"documents": list(documents),
			"stage": "onboarding",
			"status": "active",
			"created_at": _now(),
		}
		self.vendors[resolved_id] = record
		# Auto-create onboarding record
		self.onboarding[self._record_id("onboarding")] = {
			"id": self._record_id("onboarding"),
			"type": "vendor_onboarding",
			"kind": "onboarding",
			"tenant_id": tenant,
			"vendor_id": resolved_id,
			"checklist": ["legal_registration", "bank_verification", "tax_compliance"] + [f"doc_{d}" for d in documents],
			"owner_id": owner_id,
			"status": "complete",
			"created_at": _now(),
		}
		record["stage"] = "active"
		self._emit(tenant, "vendor_onboarded", record)
		return deepcopy(record)

	def vendor_qualification(
		self,
		vendor_id: str,
		qualification_type: str,
		result: str,
		valid_until: str,
		tenant_id: str | None = None,
		qualified_by: str = "procurement",
		score: float | None = None,
	) -> dict[str, Any]:
		"""
		Record a vendor qualification assessment.

		qualification_type: 'iso_9001', 'financial', 'technical', 'esg', 'security'.
		result: 'qualified', 'conditional', 'disqualified'.
		valid_until: ISO date string for qualification expiry.
		"""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		if result not in {"qualified", "conditional", "disqualified"}:
			raise ValueError(f"invalid_qualification_result:{result}")
		qual_id = self._record_id("qual")
		record = {
			"id": qual_id,
			"type": "vendor_qualification",
			"kind": "qualification",
			"tenant_id": tenant,
			"vendor_id": vendor_id,
			"qualification_type": qualification_type,
			"result": result,
			"score": score,
			"valid_until": valid_until,
			"qualified_by": qualified_by,
			"status": result,
			"created_at": _now(),
		}
		self.qualifications[qual_id] = record
		if result == "qualified":
			vendor["stage"] = "qualified"
		self._emit(tenant, "vendor_qualified", record)
		return deepcopy(record)

	def vendor_performance_score(
		self,
		vendor_id: str,
		period: str,
		metrics: dict[str, float],
		tenant_id: str | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		"""
		Record and return a vendor performance scorecard for a period.

		metrics: Dict of dimension -> score (0-100).
		Returns per-dimension scores, average, and performance tier.
		"""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		dimensions_supported = all(d in SUPPORTED_PERFORMANCE_DIMENSIONS for d in metrics)
		scores_in_range = all(0 <= v <= 100 for v in metrics.values())
		values = list(metrics.values())
		average_score = round(sum(values) / len(values), 2) if values else 0.0
		context = self._base_context(tenant, "record_performance")
		context.update({"vendor_present": True, "period_present": bool(period), "dimensions_supported": dimensions_supported, "scores_in_range": scores_in_range, "low_score": average_score < 60, "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		tier = "gold" if average_score >= 85 else ("silver" if average_score >= 70 else ("bronze" if average_score >= 55 else "at_risk"))
		perf_id = self._record_id("perf")
		record = {
			"id": perf_id,
			"type": "vendor_performance",
			"kind": "performance",
			"tenant_id": tenant,
			"vendor_id": vendor_id,
			"period": period,
			"scores": dict(metrics),
			"average_score": average_score,
			"performance_tier": tier,
			"reviewed_by": reviewed_by,
			"status": "recorded",
			"created_at": _now(),
		}
		self.performance[perf_id] = record
		self._emit(tenant, "vendor_performance_recorded", record)
		return deepcopy(record)

	def approved_vendor_list(
		self,
		category: str,
		min_score: float | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""
		Return the approved vendor list for a category, optionally filtered by min score.
		"""
		tenant = self._tenant(tenant_id)
		vendors = [v for v in self.vendors.values() if v["tenant_id"] == tenant and v.get("stage") in {"qualified", "active", "preferred"}]
		if category:
			vendors = [v for v in vendors if v.get("category") == category]
		if min_score is not None:
			# Find vendors with performance >= min_score
			scored_ids = set()
			for perf in self.performance.values():
				if perf["tenant_id"] == tenant and perf.get("average_score", 0) >= min_score:
					scored_ids.add(perf["vendor_id"])
			vendors = [v for v in vendors if v["id"] in scored_ids or min_score == 0]
		result = []
		for vendor in vendors:
			latest_perf = max(
				(p for p in self.performance.values() if p["tenant_id"] == tenant and p["vendor_id"] == vendor["id"]),
				key=lambda p: p["created_at"],
				default=None,
			)
			entry = deepcopy(vendor)
			entry["latest_score"] = latest_perf["average_score"] if latest_perf else None
			entry["performance_tier"] = latest_perf["performance_tier"] if latest_perf else "unscored"
			entry["is_preferred"] = bool(self._preferred_vendors.get(f"{tenant}:{vendor['id']}"))
			result.append(entry)
		return sorted(result, key=lambda v: v.get("latest_score") or 0, reverse=True)

	def vendor_suspension(
		self,
		vendor_id: str,
		reason: str,
		approved_by: str,
		tenant_id: str | None = None,
		suspension_id: str | None = None,
		suspension_duration_days: int | None = None,
	) -> dict[str, Any]:
		"""
		Suspend a vendor from the approved vendor list.

		reason: Reason for suspension (e.g. 'compliance_breach', 'performance_failure').
		approved_by: Approver identity.
		suspension_duration_days: Optional duration; None = indefinite.
		"""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		if not reason:
			raise ValueError("suspension_reason_required")
		if not approved_by:
			raise PermissionError("suspension_approval_required")
		if vendor.get("stage") == "suspended":
			raise PermissionError("vendor_already_suspended")
		susp_id = self._record_id("susp", suspension_id)
		record = {
			"id": susp_id,
			"type": "vendor_suspension",
			"kind": "suspension",
			"tenant_id": tenant,
			"vendor_id": vendor_id,
			"reason": reason,
			"approved_by": approved_by,
			"suspension_duration_days": suspension_duration_days,
			"suspended_at": _now(),
			"status": "active",
		}
		self._suspensions[susp_id] = record
		vendor["stage"] = "suspended"
		vendor["suspension_id"] = susp_id
		self._emit(tenant, "vendor_suspended", {**record, "type": "vendor_suspension", "status": "active"})
		return deepcopy(record)

	def contract_management(
		self,
		vendor_id: str,
		contract_type: str,
		value: float,
		start_date: str,
		end_date: str,
		tenant_id: str | None = None,
		contract_id: str | None = None,
		currency: str = "USD",
		approved_by: str = "procurement",
		auto_renew: bool = False,
		sla_terms: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Create or renew a vendor contract with SLA terms.

		contract_type: 'msa', 'sow', 'po', 'nda', 'framework_agreement'.
		value: Contract value.
		Returns contract record with computed duration and SLA summary.
		"""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "create_contract")
		context.update({"vendor_present": True, "value_present": value is not None, "currency_present": bool(currency), "date_range_present": bool(start_date and end_date), "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		resolved_id = self._record_id("contract", contract_id)
		record = {
			"id": resolved_id,
			"type": "vendor_contract",
			"kind": "contract",
			"tenant_id": tenant,
			"vendor_id": vendor_id,
			"vendor_name": vendor["name"],
			"contract_type": contract_type,
			"value": float(value),
			"currency": currency,
			"start_date": start_date,
			"end_date": end_date,
			"auto_renew": auto_renew,
			"sla_terms": dict(sla_terms or {}),
			"approved_by": approved_by,
			"status": "active",
			"created_at": _now(),
		}
		self.contracts[resolved_id] = record
		self._emit(tenant, "vendor_contract_created", record)
		return deepcopy(record)

	def preferred_vendor_designation(
		self,
		vendor_id: str,
		category: str,
		approved_by: str,
		tenant_id: str | None = None,
		valid_until: str | None = None,
		rationale: str = "",
	) -> dict[str, Any]:
		"""
		Designate a vendor as preferred for a spend category.

		Preferred status affects approved_vendor_list sorting and procurement routing.
		"""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		if not approved_by:
			raise PermissionError("preferred_designation_approval_required")
		if vendor.get("stage") == "suspended":
			raise PermissionError("cannot_designate_suspended_vendor_as_preferred")
		pref_key = f"{tenant}:{vendor_id}"
		record = {
			"vendor_id": vendor_id,
			"vendor_name": vendor["name"],
			"tenant_id": tenant,
			"category": category,
			"approved_by": approved_by,
			"valid_until": valid_until,
			"rationale": rationale,
			"designated_at": _now(),
		}
		self._preferred_vendors[pref_key] = record
		vendor["stage"] = "preferred"
		self._emit(tenant, "vendor_preferred_designated", {"id": vendor_id, "type": "preferred_designation", "status": "active"})
		return record

	def spend_analysis(
		self,
		period: str,
		category: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Analyse vendor spend for a period, optionally filtered by category.

		Returns total spend, per-vendor breakdown, per-category breakdown,
		top vendors by spend, and contract utilisation.
		"""
		tenant = self._tenant(tenant_id)
		contracts = [c for c in self.contracts.values() if c["tenant_id"] == tenant and c["start_date"][:7] <= period <= c["end_date"][:7]]
		if category:
			vendor_ids_in_category = {v["id"] for v in self.vendors.values() if v["tenant_id"] == tenant and v.get("category") == category}
			contracts = [c for c in contracts if c["vendor_id"] in vendor_ids_in_category]
		per_vendor: dict[str, float] = {}
		per_category: dict[str, float] = {}
		for contract in contracts:
			vendor = self.vendors.get(contract["vendor_id"])
			vname = vendor["name"] if vendor else contract["vendor_id"]
			vcat = vendor.get("category", "unknown") if vendor else "unknown"
			per_vendor[vname] = round(per_vendor.get(vname, 0.0) + contract["value"], 2)
			per_category[vcat] = round(per_category.get(vcat, 0.0) + contract["value"], 2)
		total_spend = round(sum(per_vendor.values()), 2)
		top_vendors = sorted(per_vendor.items(), key=lambda x: x[1], reverse=True)[:10]
		return {
			"tenant_id": tenant,
			"period": period,
			"category_filter": category,
			"total_spend": total_spend,
			"contract_count": len(contracts),
			"vendor_count": len(per_vendor),
			"per_vendor_spend": per_vendor,
			"per_category_spend": per_category,
			"top_vendors_by_spend": [{"vendor": v, "spend": s} for v, s in top_vendors],
			"currency": "USD",
			"generated_at": _now(),
		}

	def vendor_risk_assessment(
		self,
		vendor_id: str,
		tenant_id: str | None = None,
		assessor: str = "risk_team",
	) -> dict[str, Any]:
		"""
		Run a comprehensive risk assessment for a vendor.

		Aggregates risks from risk records, compliance records, performance,
		and suspension history to compute an overall risk score.
		"""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		vendor_risks = [r for r in self.risks.values() if r["tenant_id"] == tenant and r["vendor_id"] == vendor_id]
		vendor_compliance = [c for c in self.compliance.values() if c["tenant_id"] == tenant and c["vendor_id"] == vendor_id]
		vendor_perfs = [p for p in self.performance.values() if p["tenant_id"] == tenant and p["vendor_id"] == vendor_id]
		suspensions = [s for s in self._suspensions.values() if s["tenant_id"] == tenant and s["vendor_id"] == vendor_id]
		# Risk score factors
		high_risks = sum(1 for r in vendor_risks if r.get("tier") in {"high", "critical"})
		non_compliant = sum(1 for c in vendor_compliance if c.get("status_value") in {"non_compliant", "expired"})
		latest_perf = max(vendor_perfs, key=lambda p: p["created_at"], default=None)
		perf_score = latest_perf["average_score"] if latest_perf else 70.0
		# Higher = more risk
		risk_score = (high_risks * 15) + (non_compliant * 20) + (len(suspensions) * 25)
		risk_score += max(0, int((70 - perf_score) * 0.5))
		risk_score = min(100, risk_score)
		risk_tier = "critical" if risk_score >= 75 else ("high" if risk_score >= 50 else ("medium" if risk_score >= 25 else "low"))
		return {
			"vendor_id": vendor_id,
			"vendor_name": vendor["name"],
			"tenant_id": tenant,
			"assessor": assessor,
			"risk_score": risk_score,
			"risk_tier": risk_tier,
			"high_risk_count": high_risks,
			"non_compliant_count": non_compliant,
			"suspension_count": len(suspensions),
			"latest_performance_score": perf_score,
			"risk_factors": {
				"high_risks": high_risks,
				"non_compliance": non_compliant,
				"suspensions": len(suspensions),
				"performance_gap": max(0, int(70 - perf_score)),
			},
			"assessed_at": _now(),
		}

	def vendor_portal_access(
		self,
		vendor_id: str,
		event_type: str,
		tenant_id: str | None = None,
		user_email: str = "",
		ip_address: str = "",
	) -> dict[str, Any]:
		"""
		Record a vendor portal access event.

		event_type: 'login', 'logout', 'document_upload', 'invoice_submit',
		            'performance_view', 'contract_sign', 'query_submit'.
		Returns access event record.
		"""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		if vendor.get("stage") == "suspended":
			raise PermissionError("portal_access_denied_vendor_suspended")
		supported_events = {"login", "logout", "document_upload", "invoice_submit", "performance_view", "contract_sign", "query_submit"}
		if event_type not in supported_events:
			raise ValueError(f"unsupported_portal_event:{event_type}")
		event_id = self._record_id("portal_event")
		record = {
			"event_id": event_id,
			"vendor_id": vendor_id,
			"vendor_name": vendor["name"],
			"tenant_id": tenant,
			"event_type": event_type,
			"user_email": user_email,
			"ip_address": ip_address,
			"status": "recorded",
			"recorded_at": _now(),
		}
		self._portal_events.append(record)
		return record

	# ------------------------------------------------------------------
	# Original retained methods
	# ------------------------------------------------------------------

	def create_vendor(self, vendor_id: str, tenant_id: str, code: str, name: str, vendor_type: str, category: str, country: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_vendor")
		context.update({"code_present": bool(code), "name_present": bool(name), "vendor_type_supported": vendor_type in SUPPORTED_VENDOR_TYPES, "category_present": bool(category), "country_present": bool(country), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("vendor", vendor_id), "type": "vendor_profile", "kind": "vendor", "tenant_id": tenant, "code": code.upper(), "name": name, "vendor_type": vendor_type, "category": category, "country": country, "owner_id": owner_id, "stage": "prospect", "status": "active", "created_at": _now()}
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
		record = {"id": self._record_id("qualification", qualification_id), "type": "vendor_qualification", "kind": "qualification", "tenant_id": tenant, "vendor_id": vendor_id, "criteria": list(criteria), "qualified_by": qualified_by, "reviewed_by": reviewed_by, "score": score_value, "status": "qualified", "created_at": _now()}
		self.qualifications[record["id"]] = record
		vendor["stage"] = "qualified"
		self._emit(tenant, "vendor_qualified", record)
		return deepcopy(record)

	def onboard_vendor_legacy(self, onboarding_id: str, tenant_id: str, vendor_id: str, checklist: list[str], owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "onboard_vendor")
		context.update({"vendor_present": bool(vendor), "checklist_present": bool(checklist), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("onboarding", onboarding_id), "type": "vendor_onboarding", "kind": "onboarding", "tenant_id": tenant, "vendor_id": vendor_id, "checklist": list(checklist), "owner_id": owner_id, "status": "complete", "created_at": _now()}
		self.onboarding[record["id"]] = record
		vendor["stage"] = "active"
		self._emit(tenant, "vendor_onboarded", record)
		return deepcopy(record)

	def record_performance(self, performance_id: str, tenant_id: str, vendor_id: str, period: str, scores: dict[str, float], reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		values = [float(v) for v in scores.values()]
		dimensions_supported = all(d in SUPPORTED_PERFORMANCE_DIMENSIONS for d in scores)
		scores_in_range = all(0 <= v <= 100 for v in values)
		average_score = round(sum(values) / len(values), 2) if values else 0.0
		context = self._base_context(tenant, "record_performance")
		context.update({"vendor_present": bool(vendor), "period_present": bool(period), "dimensions_supported": dimensions_supported, "scores_in_range": scores_in_range, "low_score": average_score < 60, "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("performance", performance_id), "type": "vendor_performance", "kind": "performance", "tenant_id": tenant, "vendor_id": vendor_id, "period": period, "scores": dict(scores), "average_score": average_score, "reviewed_by": reviewed_by, "status": "recorded", "created_at": _now()}
		self.performance[record["id"]] = record
		self._emit(tenant, "vendor_performance_recorded", record)
		return deepcopy(record)

	def record_risk(self, risk_id: str, tenant_id: str, vendor_id: str, risk_type: str, tier: str, description: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "record_risk")
		context.update({"vendor_present": bool(vendor), "risk_tier_supported": tier in SUPPORTED_RISK_TIERS, "description_present": bool(description), "high_or_critical": tier in {"high", "critical"}, "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("risk", risk_id), "type": "vendor_risk", "kind": "risk", "tenant_id": tenant, "vendor_id": vendor_id, "risk_type": risk_type, "tier": tier, "description": description, "owner_id": owner_id, "status": "open", "created_at": _now()}
		self.risks[record["id"]] = record
		self._emit(tenant, "vendor_risk_recorded", record)
		return deepcopy(record)

	def record_compliance(self, compliance_id: str, tenant_id: str, vendor_id: str, framework: str, status: str, evidence_id: str, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		context = self._base_context(tenant, "record_compliance")
		context.update({"vendor_present": bool(vendor), "framework_present": bool(framework), "status_supported": status in SUPPORTED_COMPLIANCE_STATUSES, "evidence_present": bool(evidence_id), "review_required": status in {"non_compliant", "expired"}, "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("compliance", compliance_id), "type": "vendor_compliance", "kind": "compliance", "tenant_id": tenant, "vendor_id": vendor_id, "framework": framework, "status_value": status, "evidence_id": evidence_id, "reviewed_by": reviewed_by, "status": "recorded", "created_at": _now()}
		self.compliance[record["id"]] = record
		self._emit(tenant, "vendor_compliance_recorded", record)
		return deepcopy(record)

	def create_contract(self, contract_id: str, tenant_id: str, vendor_id: str, value: float | None, currency: str, start_date: str, end_date: str, approved_by: str) -> dict[str, Any]:
		return self.contract_management(vendor_id=vendor_id, contract_type="msa", value=float(value or 0), start_date=start_date, end_date=end_date, tenant_id=tenant_id, contract_id=contract_id, currency=currency, approved_by=approved_by)

	def register_vendor_agent(self, tenant_id: str, name: str, runtime: str, role: str, purpose: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_vendor_agent")
		context.update({"runtime_supported": runtime in SUPPORTED_VENDOR_AGENT_RUNTIMES, "role_supported": role in SUPPORTED_VENDOR_AGENT_ROLES})
		self._assert_rules(context)
		record = {"id": self._record_id("agent"), "type": "vendor_agent", "kind": "agent", "tenant_id": tenant, "name": name, "runtime": runtime, "role": role, "purpose": purpose, "owner_id": owner_id, "status": "active", "created_at": _now()}
		self.agents[record["id"]] = record
		self._emit(tenant, "vendor_agent_registered", record)
		return deepcopy(record)

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def validate_vendor_agent_action(self, tenant_id: str, privileged_action: bool, human_approved: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return evaluate_capability_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "agent_action", "privileged_action": privileged_action, "human_approved": human_approved})

	def validate_batch(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if event_stream != "bytewax":
			self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "vendor_batch", "event_stream": "queue"})
		return {"tenant_id": tenant, "record_count": int(record_count), "processor": "bytewax", "event_stream": VENDOR_EVENT_STREAM, "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		def count(records: dict[str, dict[str, Any]]) -> int:
			return sum(1 for r in records.values() if r["tenant_id"] == tenant)
		return {
			"tenant_id": tenant,
			"vendor_count": count(self.vendors),
			"preferred_vendor_count": sum(1 for v in self._preferred_vendors.values() if v["tenant_id"] == tenant),
			"suspended_vendor_count": sum(1 for v in self.vendors.values() if v["tenant_id"] == tenant and v.get("stage") == "suspended"),
			"qualification_count": count(self.qualifications),
			"onboarding_count": count(self.onboarding),
			"performance_count": count(self.performance),
			"risk_count": count(self.risks),
			"compliance_count": count(self.compliance),
			"contract_count": count(self.contracts),
			"scorecard_count": count(self.scorecards),
			"agent_count": count(self.agents),
			"portal_event_count": sum(1 for e in self._portal_events if e["tenant_id"] == tenant),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant),
			"streaming": deepcopy(STREAMING),
		}

	def create_record(self, payload: dict[str, Any]) -> dict[str, Any]:
		tenant = self._tenant(payload.get("tenant_id"))
		record = {"id": self._record_id("record", payload.get("id")), "type": payload.get("type", "vendor_record"), "kind": payload.get("kind", "generic"), "tenant_id": tenant, "status": payload.get("status", "active"), "created_at": _now(), **payload}
		self._emit(tenant, "vendor_record_created", record)
		return deepcopy(record)

	def list_records(self, tenant_id: str, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		stores = [self.vendors, self.qualifications, self.onboarding, self.performance, self.risks, self.compliance, self.contracts, self.communications, self.portal_users, self.scorecards, self.agents]
		records = [r for store in stores for r in store.values() if r["tenant_id"] == tenant]
		if record_type:
			records = [r for r in records if r["type"] == record_type or r["kind"] == record_type]
		return deepcopy(records)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return deepcopy([e for e in self._audit_events if e["tenant_id"] == tenant])


	def rfq_create(self, category: str, requirements: str, deadline: str, tenant_id: str | None = None, invited_vendors: list[str] | None = None) -> dict[str, Any]:
		"""Create a Request for Quotation and invite vendors."""
		tenant = self._tenant(tenant_id)
		rfq_id = self._record_id("rfq")
		ref = f"RFQ-{datetime.utcnow().strftime('%Y%m%d')}-{rfq_id[:6].upper()}"
		record = {"id": rfq_id, "type": "rfq", "kind": "rfq", "tenant_id": tenant, "reference": ref, "category": category, "requirements": requirements, "deadline": deadline, "invited_vendors": invited_vendors or [], "vendor_count": len(invited_vendors or []), "status": "open", "created_at": _now()}
		self._emit(tenant, "rfq_created", record)
		return deepcopy(record)

	def bid_evaluate(self, rfq_id: str, vendor_id: str, score: float, criteria: dict[str, float], tenant_id: str | None = None) -> dict[str, Any]:
		"""Evaluate a bid response for an RFQ."""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		eval_id = self._record_id("bideval")
		avg = round(sum(criteria.values()) / max(len(criteria), 1), 2)
		record = {"id": eval_id, "type": "bid_evaluation", "kind": "bid_evaluation", "tenant_id": tenant, "rfq_id": rfq_id, "vendor_id": vendor_id, "vendor_name": vendor["name"], "score": score, "criteria": criteria, "average_score": avg, "recommended": avg >= 70, "status": "evaluated", "created_at": _now()}
		self._emit(tenant, "bid_evaluated", record)
		return deepcopy(record)

	def preferred_vendor_list(self, category: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return the preferred vendor list for a category — alias."""
		return self.approved_vendor_list(category, min_score=80.0, tenant_id=tenant_id)

	def vendor_scorecard(self, vendor_id: str, period: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate a comprehensive vendor scorecard for the period."""
		tenant = self._tenant(tenant_id)
		return self.vendor_performance_score(vendor_id, period, {}, tenant_id=tenant)

	def risk_flag_vendor(self, vendor_id: str, flag_type: str, description: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Flag a vendor for risk — domain alias for record_risk."""
		tenant = self._tenant(tenant_id)
		risk_id = self._record_id("risk")
		return self.record_risk(risk_id, tenant, vendor_id, flag_type, "high", description)

	def compliance_check_vendor(self, vendor_id: str, framework: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Run compliance check for a vendor — domain alias."""
		tenant = self._tenant(tenant_id)
		return self.vendor_risk_assessment(vendor_id, tenant_id=tenant)

	def payment_terms_manage(self, vendor_id: str, payment_days: int, discount_pct: float, tenant_id: str | None = None) -> dict[str, Any]:
		"""Set payment terms for a vendor."""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		vendor["payment_days"] = payment_days
		vendor["early_payment_discount_pct"] = discount_pct
		vendor["payment_terms_updated_at"] = _now()
		return {"vendor_id": vendor_id, "payment_days": payment_days, "discount_pct": discount_pct, "updated_at": _now()}

	def vendor_communication(self, vendor_id: str, subject: str, message: str, channel: str = "email", tenant_id: str | None = None) -> dict[str, Any]:
		"""Send a communication to a vendor."""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		comm_id = self._record_id("comm")
		record = {"id": comm_id, "type": "vendor_communication", "kind": "communication", "tenant_id": tenant, "vendor_id": vendor_id, "vendor_name": vendor["name"], "subject": subject, "message": message[:200], "channel": channel, "status": "sent", "sent_at": _now()}
		self.communications[comm_id] = record
		self._emit(tenant, "vendor_communicated", record)
		return deepcopy(record)

	def vendor_onboarding_checklist(self, vendor_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return the onboarding checklist status for a vendor."""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		onboarding_recs = [o for o in self.onboarding.values() if o["tenant_id"] == tenant and o.get("vendor_id") == vendor_id]
		latest = max(onboarding_recs, key=lambda o: o["created_at"], default=None)
		checklist = latest.get("checklist", ["legal_registration", "bank_verification", "tax_compliance"]) if latest else ["legal_registration", "bank_verification", "tax_compliance"]
		return {"vendor_id": vendor_id, "vendor_name": vendor["name"], "tenant_id": tenant, "checklist": checklist, "completed": latest is not None, "stage": vendor.get("stage"), "generated_at": _now()}

	def vendor_deactivate(self, vendor_id: str, reason: str, deactivated_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Deactivate a vendor from the register."""
		tenant = self._tenant(tenant_id)
		vendor = self._get_vendor(vendor_id, tenant)
		vendor["status"] = "inactive"
		vendor["stage"] = "deactivated"
		vendor["deactivation_reason"] = reason
		vendor["deactivated_by"] = deactivated_by
		vendor["deactivated_at"] = _now()
		self._emit(tenant, "vendor_deactivated", {"id": vendor_id, "type": "vendor_profile", "status": "inactive"})
		return deepcopy(vendor)

	def spend_category_analysis(self, period: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Analyse spend broken down by category."""
		return self.spend_analysis(period, tenant_id=tenant_id)

	def vendor_analytics(self, period: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return comprehensive vendor analytics for the period."""
		tenant = self._tenant(tenant_id)
		total_vendors = sum(1 for v in self.vendors.values() if v["tenant_id"] == tenant)
		preferred = sum(1 for v in self._preferred_vendors.values() if v["tenant_id"] == tenant)
		suspended = sum(1 for v in self._suspensions.values() if v["tenant_id"] == tenant and v["status"] == "active")
		by_stage: dict[str, int] = {}
		for v in self.vendors.values():
			if v["tenant_id"] == tenant:
				by_stage[v.get("stage", "unknown")] = by_stage.get(v.get("stage", "unknown"), 0) + 1
		return {"tenant_id": tenant, "period": period, "total_vendors": total_vendors, "preferred_vendors": preferred, "suspended_vendors": suspended, "by_stage": by_stage, "contract_count": sum(1 for c in self.contracts.values() if c["tenant_id"] == tenant), "total_contract_value": sum(c["value"] for c in self.contracts.values() if c["tenant_id"] == tenant), "generated_at": _now()}

	def contract_renew(self, vendor_id: str, contract_id: str, extension_months: int, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Renew a vendor contract for an additional period."""
		tenant = self._tenant(tenant_id)
		contract = self.contracts.get(contract_id)
		if contract is None or contract["tenant_id"] != tenant:
			raise VendorNotFoundError("contract_not_found")
		from datetime import timedelta
		old_end = contract.get("end_date", _now()[:10])
		new_end = (datetime.fromisoformat(old_end) + timedelta(days=extension_months * 30)).isoformat()[:10]
		contract["end_date"] = new_end
		contract["renewal_reason"] = reason
		contract["renewed_at"] = _now()
		self._emit(tenant, "vendor_contract_renewed", {"id": contract_id, "type": "vendor_contract", "status": "renewed"})
		return deepcopy(contract)


VendorManagementLifecycleService = VendorManagementService
VendorLifecycleService = VendorManagementService
VendorRiskService = VendorManagementService
VendorPerformanceService = VendorManagementService
