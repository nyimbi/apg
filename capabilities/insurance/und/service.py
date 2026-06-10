"""Underwriting Engine Service (ins_und).

Risk assessment, rating engine, capacity management, reinsurance treaties, underwriting rules.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

RISK_BANDS = {"preferred": (0.0, 0.3), "standard": (0.3, 0.6), "substandard": (0.6, 0.8), "declined": (0.8, 1.0)}
TREATY_TYPES = {"quota_share", "surplus", "excess_of_loss", "stop_loss", "facultative"}
DECISIONS = {"accept", "accept_with_conditions", "refer", "decline"}

BASE_RATES: dict[str, Decimal] = {
	"motor_comprehensive": Decimal("0.04"),
	"motor_third_party": Decimal("0.012"),
	"fire_industrial": Decimal("0.002"),
	"fire_domestic": Decimal("0.0015"),
	"marine_cargo": Decimal("0.005"),
	"marine_hull": Decimal("0.008"),
	"life_whole": Decimal("0.025"),
	"life_term": Decimal("0.015"),
	"health_individual": Decimal("0.06"),
	"health_group": Decimal("0.05"),
	"travel": Decimal("0.03"),
	"engineering": Decimal("0.003"),
}

CAPACITY_LIMITS: dict[str, Decimal] = {
	"motor_comprehensive": Decimal("50000000"),
	"fire_industrial": Decimal("500000000"),
	"fire_domestic": Decimal("100000000"),
	"marine_cargo": Decimal("200000000"),
	"marine_hull": Decimal("300000000"),
	"life_whole": Decimal("20000000"),
	"life_term": Decimal("10000000"),
	"health_individual": Decimal("5000000"),
	"engineering": Decimal("1000000000"),
}


class UnderwritingEngineService:
	"""In-memory executable service for the Underwriting Engine."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.submissions: dict[str, dict[str, Any]] = {}
		self.assessments: dict[str, dict[str, Any]] = {}
		self.ratings: dict[str, dict[str, Any]] = {}
		self.capacity_checks: dict[str, dict[str, Any]] = {}
		self.treaties: dict[str, dict[str, Any]] = {}
		self.rules: dict[str, dict[str, Any]] = {}
		self.referrals: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._record_id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"entity_type": entity_type,
			"details": details or {},
			"created_at": self._now(),
		})

	def _score_to_band(self, score: float) -> str:
		for band, (lo, hi) in RISK_BANDS.items():
			if lo <= score < hi:
				return band
		return "declined"

	def _compute_risk_score(self, product_code: str, risk_attributes: dict[str, Any]) -> float:
		"""Heuristic risk scoring based on product and attributes."""
		base = 0.3
		age = risk_attributes.get("age", 35)
		if age > 60:
			base += 0.1
		elif age < 25:
			base += 0.15
		claim_history = risk_attributes.get("claim_history_count", 0)
		base += min(claim_history * 0.05, 0.25)
		vehicle_age = risk_attributes.get("vehicle_age_years", 0)
		if product_code.startswith("motor") and vehicle_age > 10:
			base += 0.1
		health_conditions = risk_attributes.get("pre_existing_conditions", 0)
		if product_code.startswith("health") or product_code.startswith("life"):
			base += health_conditions * 0.08
		return min(round(base, 4), 0.99)

	# ── Submissions ───────────────────────────────────────────────────────────

	async def submit_risk(
		self,
		tenant_id: str,
		proposer_name: str,
		proposer_id: str,
		product_code: str,
		risk_class: str,
		sum_insured: Decimal,
		submitted_by: str,
		currency: str = "KES",
		risk_attributes: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Accept a risk submission for underwriting evaluation."""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": self._record_id("sub"),
			"type": "und_submission",
			"proposer_name": proposer_name,
			"proposer_id": proposer_id,
			"product_code": product_code,
			"risk_class": risk_class,
			"sum_insured": Decimal(str(sum_insured)),
			"currency": currency,
			"risk_attributes": deepcopy(risk_attributes or {}),
			"submitted_by": submitted_by,
			"status": "submitted",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.submissions[record["id"]] = record
		self._emit(tenant, "risk_submitted", record["id"], "und_submission", {"product_code": product_code})
		_log.info("Risk submission: %s product=%s tenant=%s", record["id"], product_code, tenant)
		return deepcopy(record)

	async def get_submission(self, tenant_id: str, submission_id: str) -> dict[str, Any]:
		"""Get a risk submission."""
		tenant = self._tenant(tenant_id)
		sub = self.submissions.get(submission_id)
		if not sub or sub["tenant_id"] != tenant:
			raise KeyError(f"submission_not_found:{submission_id}")
		return deepcopy(sub)

	async def list_submissions(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		"""List submissions."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.submissions.values() if s["tenant_id"] == tenant]
		if status:
			items = [s for s in items if s["status"] == status]
		return items

	# ── Risk Assessment ───────────────────────────────────────────────────────

	async def assess_risk(self, tenant_id: str, submission_id: str, assessed_by: str | None = None) -> dict[str, Any]:
		"""Run automated risk assessment on a submission."""
		tenant = self._tenant(tenant_id)
		sub = self.submissions.get(submission_id)
		if not sub or sub["tenant_id"] != tenant:
			raise KeyError(f"submission_not_found:{submission_id}")
		score = self._compute_risk_score(sub["product_code"], sub["risk_attributes"])
		band = self._score_to_band(score)
		decision = "decline" if band == "declined" else ("refer" if band == "substandard" else "accept")
		base_rate = BASE_RATES.get(sub["product_code"], Decimal("0.05"))
		loading = Decimal("1.0")
		loading_factors: dict[str, float] = {}
		if band == "substandard":
			loading = Decimal("1.25")
			loading_factors["substandard_loading"] = 0.25
		elif band == "preferred":
			loading = Decimal("0.9")
			loading_factors["preferred_discount"] = -0.1
		recommended_premium = (sub["sum_insured"] * base_rate * loading).quantize(Decimal("0.01"))
		exclusions: list[str] = []
		age = sub["risk_attributes"].get("age", 35)
		if age > 65:
			exclusions.append("age_exclusion_over_65")
		pre_existing = sub["risk_attributes"].get("pre_existing_conditions", 0)
		if pre_existing > 0:
			exclusions.append("pre_existing_condition_exclusion")
		record: dict[str, Any] = {
			"id": self._record_id("asmnt"),
			"type": "und_assessment",
			"submission_id": submission_id,
			"risk_score": score,
			"risk_band": band,
			"recommended_premium": recommended_premium,
			"loading_factors": loading_factors,
			"exclusions": exclusions,
			"decision": decision,
			"assessed_by": assessed_by or "automated",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.assessments[record["id"]] = record
		sub["status"] = "assessed"
		sub["assessment_id"] = record["id"]
		self._emit(tenant, "risk_assessed", record["id"], "und_assessment", {"score": score, "decision": decision})
		return deepcopy(record)

	async def get_assessment(self, tenant_id: str, assessment_id: str) -> dict[str, Any]:
		"""Retrieve an assessment."""
		tenant = self._tenant(tenant_id)
		asmnt = self.assessments.get(assessment_id)
		if not asmnt or asmnt["tenant_id"] != tenant:
			raise KeyError(f"assessment_not_found:{assessment_id}")
		return deepcopy(asmnt)

	async def list_assessments(self, tenant_id: str, decision: str | None = None) -> list[dict[str, Any]]:
		"""List assessments."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.assessments.values() if a["tenant_id"] == tenant]
		if decision:
			items = [a for a in items if a["decision"] == decision]
		return items

	async def override_assessment(self, tenant_id: str, assessment_id: str, new_decision: str, override_by: str, reason: str) -> dict[str, Any]:
		"""Manual underwriter override of automated assessment."""
		tenant = self._tenant(tenant_id)
		asmnt = self.assessments.get(assessment_id)
		if not asmnt or asmnt["tenant_id"] != tenant:
			raise KeyError(f"assessment_not_found:{assessment_id}")
		if new_decision not in DECISIONS:
			raise ValueError(f"unsupported_decision:{new_decision}")
		asmnt["original_decision"] = asmnt["decision"]
		asmnt["decision"] = new_decision
		asmnt["override_by"] = override_by
		asmnt["override_reason"] = reason
		asmnt["overridden_at"] = self._now()
		self._emit(tenant, "assessment_overridden", assessment_id, "und_assessment", {"new_decision": new_decision})
		return deepcopy(asmnt)

	# ── Rating Engine ─────────────────────────────────────────────────────────

	async def rate_risk(
		self,
		tenant_id: str,
		submission_id: str,
		base_rate: Decimal,
		adjustments: dict[str, Decimal] | None = None,
		rated_by: str = "",
	) -> dict[str, Any]:
		"""Compute final premium using base rate plus adjustment factors."""
		tenant = self._tenant(tenant_id)
		sub = self.submissions.get(submission_id)
		if not sub or sub["tenant_id"] != tenant:
			raise KeyError(f"submission_not_found:{submission_id}")
		adj = adjustments or {}
		total_adj = sum(adj.values(), Decimal("0"))
		effective_rate = Decimal(str(base_rate)) + total_adj
		if effective_rate < Decimal("0"):
			effective_rate = Decimal("0")
		premium = (sub["sum_insured"] * effective_rate).quantize(Decimal("0.01"))
		record: dict[str, Any] = {
			"id": self._record_id("rate"),
			"type": "und_rating",
			"submission_id": submission_id,
			"base_rate": Decimal(str(base_rate)),
			"adjustments": deepcopy(adj),
			"effective_rate": effective_rate,
			"sum_insured": sub["sum_insured"],
			"computed_premium": premium,
			"rated_by": rated_by,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.ratings[record["id"]] = record
		sub["final_premium"] = premium
		sub["status"] = "rated"
		self._emit(tenant, "risk_rated", record["id"], "und_rating", {"premium": str(premium)})
		return deepcopy(record)

	async def list_ratings(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List rating records."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.ratings.values() if r["tenant_id"] == tenant]

	# ── Capacity Management ───────────────────────────────────────────────────

	async def check_capacity(
		self,
		tenant_id: str,
		product_code: str,
		risk_class: str,
		requested_sum_insured: Decimal,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Check available underwriting capacity for a product."""
		tenant = self._tenant(tenant_id)
		limit = CAPACITY_LIMITS.get(product_code, Decimal("10000000"))
		committed = sum(
			s["sum_insured"] for s in self.submissions.values()
			if s["tenant_id"] == tenant and s["product_code"] == product_code and s["status"] not in {"declined", "withdrawn"}
		)
		available = limit - committed
		si = Decimal(str(requested_sum_insured))
		within_capacity = si <= available
		record: dict[str, Any] = {
			"id": self._record_id("cap"),
			"type": "und_capacity_check",
			"product_code": product_code,
			"risk_class": risk_class,
			"capacity_limit": limit,
			"committed_sum_insured": committed,
			"available_capacity": available,
			"requested_sum_insured": si,
			"within_capacity": within_capacity,
			"requires_reinsurance": si > available * Decimal("0.7"),
			"currency": currency,
			"tenant_id": tenant,
			"checked_at": self._now(),
		}
		self.capacity_checks[record["id"]] = record
		self._emit(tenant, "capacity_checked", record["id"], "und_capacity_check", {"product_code": product_code})
		return deepcopy(record)

	# ── Reinsurance Treaties ──────────────────────────────────────────────────

	async def create_treaty(
		self,
		tenant_id: str,
		treaty_name: str,
		treaty_type: str,
		reinsurer: str,
		retention: Decimal,
		cession_pct: float,
		treaty_limit: Decimal,
		effective_date: str,
		expiry_date: str,
	) -> dict[str, Any]:
		"""Register a reinsurance treaty."""
		tenant = self._tenant(tenant_id)
		if treaty_type not in TREATY_TYPES:
			raise ValueError(f"unsupported_treaty_type:{treaty_type}")
		if not (0 <= cession_pct <= 1):
			raise ValueError("cession_pct_must_be_between_0_and_1")
		record: dict[str, Any] = {
			"id": self._record_id("treaty"),
			"type": "und_reinsurance_treaty",
			"treaty_name": treaty_name,
			"treaty_type": treaty_type,
			"reinsurer": reinsurer,
			"retention": Decimal(str(retention)),
			"cession_pct": cession_pct,
			"treaty_limit": Decimal(str(treaty_limit)),
			"effective_date": effective_date,
			"expiry_date": expiry_date,
			"status": "active",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.treaties[record["id"]] = record
		self._emit(tenant, "treaty_registered", record["id"], "und_reinsurance_treaty", {"treaty_name": treaty_name})
		return deepcopy(record)

	async def get_treaty(self, tenant_id: str, treaty_id: str) -> dict[str, Any]:
		"""Retrieve a treaty."""
		tenant = self._tenant(tenant_id)
		treaty = self.treaties.get(treaty_id)
		if not treaty or treaty["tenant_id"] != tenant:
			raise KeyError(f"treaty_not_found:{treaty_id}")
		return deepcopy(treaty)

	async def list_treaties(self, tenant_id: str, active_only: bool = True) -> list[dict[str, Any]]:
		"""List reinsurance treaties."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(t) for t in self.treaties.values() if t["tenant_id"] == tenant]
		if active_only:
			items = [t for t in items if t["status"] == "active"]
		return items

	async def apply_treaty_cession(self, tenant_id: str, treaty_id: str, gross_premium: Decimal) -> dict[str, Any]:
		"""Calculate treaty cession for a gross premium."""
		tenant = self._tenant(tenant_id)
		treaty = self.treaties.get(treaty_id)
		if not treaty or treaty["tenant_id"] != tenant:
			raise KeyError(f"treaty_not_found:{treaty_id}")
		gp = Decimal(str(gross_premium))
		cession = (gp * Decimal(str(treaty["cession_pct"]))).quantize(Decimal("0.01"))
		retention = gp - cession
		return {
			"treaty_id": treaty_id,
			"gross_premium": str(gp),
			"cession_pct": treaty["cession_pct"],
			"ceded_premium": str(cession),
			"retained_premium": str(retention),
			"reinsurer": treaty["reinsurer"],
			"computed_at": self._now(),
		}

	# ── Underwriting Rules ────────────────────────────────────────────────────

	async def create_rule(
		self,
		tenant_id: str,
		rule_name: str,
		product_code: str,
		condition: str,
		action: str,
		priority: int = 100,
	) -> dict[str, Any]:
		"""Create an underwriting rule."""
		tenant = self._tenant(tenant_id)
		if any(r["rule_name"] == rule_name and r["tenant_id"] == tenant for r in self.rules.values()):
			raise ValueError(f"rule_name_duplicate:{rule_name}")
		record: dict[str, Any] = {
			"id": self._record_id("rule"),
			"type": "und_rule",
			"rule_name": rule_name,
			"product_code": product_code,
			"condition": condition,
			"action": action,
			"priority": priority,
			"active": True,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.rules[record["id"]] = record
		self._emit(tenant, "rule_created", record["id"], "und_rule", {"rule_name": rule_name})
		return deepcopy(record)

	async def list_rules(self, tenant_id: str, product_code: str | None = None, active_only: bool = True) -> list[dict[str, Any]]:
		"""List underwriting rules."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.rules.values() if r["tenant_id"] == tenant]
		if active_only:
			items = [r for r in items if r["active"]]
		if product_code:
			items = [r for r in items if r["product_code"] == product_code]
		return sorted(items, key=lambda r: r["priority"])

	async def deactivate_rule(self, tenant_id: str, rule_id: str) -> dict[str, Any]:
		"""Deactivate an underwriting rule."""
		tenant = self._tenant(tenant_id)
		rule = self.rules.get(rule_id)
		if not rule or rule["tenant_id"] != tenant:
			raise KeyError(f"rule_not_found:{rule_id}")
		rule["active"] = False
		rule["deactivated_at"] = self._now()
		self._emit(tenant, "rule_deactivated", rule_id, "und_rule", {})
		return deepcopy(rule)

	async def delete_rule(self, tenant_id: str, rule_id: str) -> dict[str, Any]:
		"""Delete an underwriting rule."""
		tenant = self._tenant(tenant_id)
		rule = self.rules.get(rule_id)
		if not rule or rule["tenant_id"] != tenant:
			raise KeyError(f"rule_not_found:{rule_id}")
		del self.rules[rule_id]
		self._emit(tenant, "rule_deleted", rule_id, "und_rule", {})
		return {"id": rule_id, "status": "deleted"}

	# ── Referrals ─────────────────────────────────────────────────────────────

	async def create_referral(self, tenant_id: str, submission_id: str, reason: str, referred_to: str) -> dict[str, Any]:
		"""Create an underwriting referral for senior review."""
		tenant = self._tenant(tenant_id)
		sub = self.submissions.get(submission_id)
		if not sub or sub["tenant_id"] != tenant:
			raise KeyError(f"submission_not_found:{submission_id}")
		record: dict[str, Any] = {
			"id": self._record_id("ref"),
			"type": "und_referral",
			"submission_id": submission_id,
			"reason": reason,
			"referred_to": referred_to,
			"status": "pending",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.referrals[record["id"]] = record
		sub["status"] = "referred"
		self._emit(tenant, "risk_referred", record["id"], "und_referral", {"submission_id": submission_id})
		return deepcopy(record)

	async def resolve_referral(self, tenant_id: str, referral_id: str, decision: str, resolved_by: str, notes: str = "") -> dict[str, Any]:
		"""Resolve a senior underwriting referral."""
		tenant = self._tenant(tenant_id)
		ref = self.referrals.get(referral_id)
		if not ref or ref["tenant_id"] != tenant:
			raise KeyError(f"referral_not_found:{referral_id}")
		if decision not in DECISIONS:
			raise ValueError(f"unsupported_decision:{decision}")
		ref["decision"] = decision
		ref["resolved_by"] = resolved_by
		ref["notes"] = notes
		ref["status"] = "resolved"
		ref["resolved_at"] = self._now()
		self._emit(tenant, "referral_resolved", referral_id, "und_referral", {"decision": decision})
		return deepcopy(ref)

	async def list_referrals(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		"""List referrals."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.referrals.values() if r["tenant_id"] == tenant]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	# ── Portfolio analytics ───────────────────────────────────────────────────

	async def underwriting_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Underwriting portfolio summary."""
		tenant = self._tenant(tenant_id)
		subs = [s for s in self.submissions.values() if s["tenant_id"] == tenant]
		by_product: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for s in subs:
			by_product[s["product_code"]] = by_product.get(s["product_code"], 0) + 1
			by_status[s["status"]] = by_status.get(s["status"], 0) + 1
		return {
			"tenant_id": tenant,
			"total_submissions": len(subs),
			"by_product": by_product,
			"by_status": by_status,
			"assessments": len([a for a in self.assessments.values() if a["tenant_id"] == tenant]),
			"active_treaties": len([t for t in self.treaties.values() if t["tenant_id"] == tenant and t["status"] == "active"]),
			"active_rules": len([r for r in self.rules.values() if r["tenant_id"] == tenant and r["active"]]),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Service health status."""
		return {
			"service": "ins_und",
			"status": "healthy",
			"submission_count": len(self.submissions),
			"assessment_count": len(self.assessments),
			"treaty_count": len(self.treaties),
			"rule_count": len(self.rules),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Describe this capability."""
		return {
			"capability_id": "ins_und",
			"name": "Underwriting Engine",
			"version": "1.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"supported_products": list(BASE_RATES.keys()),
			"treaty_types": list(TREATY_TYPES),
			"decisions": list(DECISIONS),
			"risk_bands": list(RISK_BANDS.keys()),
		}

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return audit trail."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
