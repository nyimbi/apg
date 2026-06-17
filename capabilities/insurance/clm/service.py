"""Claims Management Service (ins_clm).

Handles FNOL, assessment, reserve management, payments, fraud detection, and subrogation.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

SUPPORTED_CLAIM_STATUSES = {
	"fnol", "under_assessment", "reserved", "approved", "partially_paid",
	"fully_paid", "repudiated", "withdrawn", "subrogation",
}
SUPPORTED_RESERVE_TYPES = {"outstanding", "ibnr", "allocated_loss_adjustment", "unallocated_loss_adjustment"}
SUPPORTED_PAYMENT_TYPES = {"partial", "full", "advance", "ex_gratia", "recoverable_advance"}
FRAUD_HIGH_THRESHOLD = 0.75


class ClaimsManagementService:
	"""In-memory executable service for the Claims Management lifecycle."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.claims: dict[str, dict[str, Any]] = {}
		self.reserves: dict[str, dict[str, Any]] = {}
		self.payments: dict[str, dict[str, Any]] = {}
		self.fraud_assessments: dict[str, dict[str, Any]] = {}
		self.subrogations: dict[str, dict[str, Any]] = {}
		self.assessments: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		self._claim_seq: int = 0

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _claim_number(self, tenant: str) -> str:
		self._claim_seq += 1
		year = datetime.utcnow().year
		return f"CLM-{year}-{self._claim_seq:06d}"

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

	def _get_claim(self, claim_id: str, tenant: str) -> dict[str, Any]:
		clm = self.claims.get(claim_id)
		if not clm or clm["tenant_id"] != tenant:
			raise KeyError(f"claim_not_found:{claim_id}")
		return clm

	# ── FNOL ──────────────────────────────────────────────────────────────────

	async def register_fnol(
		self,
		tenant_id: str,
		policy_id: str,
		policy_number: str,
		claimant_name: str,
		claimant_id: str,
		incident_date: str,
		incident_description: str,
		estimated_loss: Decimal,
		reported_by: str,
		currency: str = "KES",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register a First Notification of Loss."""
		tenant = self._tenant(tenant_id)
		if not incident_description:
			raise ValueError("incident_description_required")
		if Decimal(str(estimated_loss)) <= 0:
			raise ValueError("estimated_loss_must_be_positive")
		claim_number = self._claim_number(tenant)
		record: dict[str, Any] = {
			"id": self._record_id("clm"),
			"type": "ins_claim",
			"claim_number": claim_number,
			"policy_id": policy_id,
			"policy_number": policy_number,
			"claimant_name": claimant_name,
			"claimant_id": claimant_id,
			"incident_date": incident_date,
			"incident_description": incident_description,
			"estimated_loss": Decimal(str(estimated_loss)),
			"reserve_amount": Decimal("0"),
			"paid_amount": Decimal("0"),
			"currency": currency,
			"status": "fnol",
			"fraud_flag": False,
			"fraud_score": 0.0,
			"assessor_id": None,
			"reported_by": reported_by,
			"tenant_id": tenant,
			"created_at": self._now(),
			"updated_at": None,
			"metadata": deepcopy(metadata or {}),
		}
		self.claims[record["id"]] = record
		self._emit(tenant, "fnol_registered", record["id"], "ins_claim", {"claim_number": claim_number, "policy_id": policy_id})
		_log.info("FNOL registered: %s tenant=%s", claim_number, tenant)
		return deepcopy(record)

	async def get_claim(self, tenant_id: str, claim_id: str) -> dict[str, Any]:
		"""Retrieve a claim by ID."""
		tenant = self._tenant(tenant_id)
		return deepcopy(self._get_claim(claim_id, tenant))

	async def get_claim_by_number(self, tenant_id: str, claim_number: str) -> dict[str, Any]:
		"""Retrieve a claim by claim number."""
		tenant = self._tenant(tenant_id)
		clm = next((c for c in self.claims.values() if c["claim_number"] == claim_number and c["tenant_id"] == tenant), None)
		if not clm:
			raise KeyError(f"claim_not_found:{claim_number}")
		return deepcopy(clm)

	async def list_claims(self, tenant_id: str, status: str | None = None, policy_id: str | None = None) -> list[dict[str, Any]]:
		"""List claims, optionally filtered."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.claims.values() if c["tenant_id"] == tenant]
		if status:
			items = [c for c in items if c["status"] == status]
		if policy_id:
			items = [c for c in items if c["policy_id"] == policy_id]
		return items

	async def update_claim(self, tenant_id: str, claim_id: str, updates: dict[str, Any]) -> dict[str, Any]:
		"""Update claim fields."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		allowed = {"status", "assessor_id", "fraud_flag", "metadata"}
		for key, value in updates.items():
			if key not in allowed:
				raise ValueError(f"field_not_updatable:{key}")
			if key == "status" and value not in SUPPORTED_CLAIM_STATUSES:
				raise ValueError(f"unsupported_status:{value}")
			clm[key] = value
		clm["updated_at"] = self._now()
		self._emit(tenant, "claim_updated", claim_id, "ins_claim", {"fields": list(updates.keys())})
		return deepcopy(clm)

	async def delete_claim(self, tenant_id: str, claim_id: str, reason: str) -> dict[str, Any]:
		"""Withdraw (soft-delete) a claim in FNOL state."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		if clm["status"] != "fnol":
			raise PermissionError("only_fnol_claims_can_be_withdrawn")
		clm["status"] = "withdrawn"
		clm["withdrawal_reason"] = reason
		clm["withdrawn_at"] = self._now()
		self._emit(tenant, "claim_withdrawn", claim_id, "ins_claim", {"reason": reason})
		return deepcopy(clm)

	# ── Assessment ────────────────────────────────────────────────────────────

	async def assign_assessor(self, tenant_id: str, claim_id: str, assessor_id: str, assigned_by: str) -> dict[str, Any]:
		"""Assign a loss assessor to a claim."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		if clm["status"] not in {"fnol", "under_assessment"}:
			raise PermissionError("claim_cannot_be_assigned_in_current_status")
		clm["assessor_id"] = assessor_id
		clm["status"] = "under_assessment"
		clm["updated_at"] = self._now()
		rec: dict[str, Any] = {
			"id": self._record_id("asmnt"),
			"type": "ins_assessment",
			"claim_id": claim_id,
			"assessor_id": assessor_id,
			"assigned_by": assigned_by,
			"status": "in_progress",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.assessments[rec["id"]] = rec
		self._emit(tenant, "assessor_assigned", claim_id, "ins_claim", {"assessor_id": assessor_id})
		return deepcopy(rec)

	async def submit_assessment_report(
		self,
		tenant_id: str,
		claim_id: str,
		assessed_loss: Decimal,
		recommendation: str,
		findings: str,
		assessor_id: str,
	) -> dict[str, Any]:
		"""Submit assessment report for a claim."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		if clm["status"] != "under_assessment":
			raise PermissionError("claim_must_be_under_assessment")
		assessment = next((a for a in self.assessments.values() if a["claim_id"] == claim_id and a["tenant_id"] == tenant), None)
		if not assessment:
			raise KeyError("assessment_record_not_found")
		assessment["assessed_loss"] = Decimal(str(assessed_loss))
		assessment["recommendation"] = recommendation
		assessment["findings"] = findings
		assessment["status"] = "submitted"
		assessment["submitted_at"] = self._now()
		clm["estimated_loss"] = Decimal(str(assessed_loss))
		clm["updated_at"] = self._now()
		self._emit(tenant, "assessment_submitted", claim_id, "ins_claim", {"assessed_loss": str(assessed_loss)})
		return deepcopy(assessment)

	# ── Reserve Management ────────────────────────────────────────────────────

	async def set_reserve(
		self,
		tenant_id: str,
		claim_id: str,
		reserve_amount: Decimal,
		reserve_type: str = "outstanding",
		set_by: str = "",
		justification: str = "",
	) -> dict[str, Any]:
		"""Set or update a claim reserve."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		if reserve_type not in SUPPORTED_RESERVE_TYPES:
			raise ValueError(f"unsupported_reserve_type:{reserve_type}")
		old_reserve = clm["reserve_amount"]
		clm["reserve_amount"] = Decimal(str(reserve_amount))
		clm["status"] = "reserved"
		clm["updated_at"] = self._now()
		record: dict[str, Any] = {
			"id": self._record_id("res"),
			"type": "ins_reserve",
			"claim_id": claim_id,
			"reserve_type": reserve_type,
			"old_reserve": old_reserve,
			"new_reserve": Decimal(str(reserve_amount)),
			"set_by": set_by,
			"justification": justification,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.reserves[record["id"]] = record
		self._emit(tenant, "reserve_set", record["id"], "ins_reserve", {"claim_id": claim_id, "amount": str(reserve_amount)})
		return deepcopy(record)

	async def list_reserves(self, tenant_id: str, claim_id: str | None = None) -> list[dict[str, Any]]:
		"""List reserve records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.reserves.values() if r["tenant_id"] == tenant]
		if claim_id:
			items = [r for r in items if r["claim_id"] == claim_id]
		return items

	# ── Payment ───────────────────────────────────────────────────────────────

	async def process_payment(
		self,
		tenant_id: str,
		claim_id: str,
		payment_amount: Decimal,
		payment_type: str,
		payee_name: str,
		payee_account: str,
		payment_reference: str,
		authorised_by: str,
	) -> dict[str, Any]:
		"""Process a claims payment."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		if clm["status"] not in {"reserved", "approved", "partially_paid"}:
			raise PermissionError("claim_must_be_reserved_or_approved_for_payment")
		if payment_type not in SUPPORTED_PAYMENT_TYPES:
			raise ValueError(f"unsupported_payment_type:{payment_type}")
		amount = Decimal(str(payment_amount))
		if amount <= 0:
			raise ValueError("payment_amount_must_be_positive")
		if amount > clm["reserve_amount"] - clm["paid_amount"]:
			raise ValueError("payment_exceeds_outstanding_reserve")
		record: dict[str, Any] = {
			"id": self._record_id("pay"),
			"type": "ins_payment",
			"claim_id": claim_id,
			"claim_number": clm["claim_number"],
			"payment_type": payment_type,
			"payment_amount": amount,
			"payee_name": payee_name,
			"payee_account": payee_account,
			"payment_reference": payment_reference,
			"authorised_by": authorised_by,
			"status": "processed",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.payments[record["id"]] = record
		clm["paid_amount"] = clm["paid_amount"] + amount
		clm["status"] = "fully_paid" if clm["paid_amount"] >= clm["reserve_amount"] else "partially_paid"
		clm["updated_at"] = self._now()
		self._emit(tenant, "claim_payment_processed", record["id"], "ins_payment", {"claim_id": claim_id, "amount": str(amount)})
		return deepcopy(record)

	async def list_payments(self, tenant_id: str, claim_id: str | None = None) -> list[dict[str, Any]]:
		"""List payment records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.payments.values() if p["tenant_id"] == tenant]
		if claim_id:
			items = [p for p in items if p["claim_id"] == claim_id]
		return items

	# ── Fraud Detection ───────────────────────────────────────────────────────

	async def assess_fraud_risk(
		self,
		tenant_id: str,
		claim_id: str,
		fraud_score: float,
		indicators: list[str],
		assessed_by: str,
		recommendation: str,
	) -> dict[str, Any]:
		"""Record a fraud risk assessment for a claim."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		is_high_risk = fraud_score >= FRAUD_HIGH_THRESHOLD
		record: dict[str, Any] = {
			"id": self._record_id("fraud"),
			"type": "ins_fraud_assessment",
			"claim_id": claim_id,
			"fraud_score": fraud_score,
			"indicators": list(indicators),
			"assessed_by": assessed_by,
			"recommendation": recommendation,
			"high_risk": is_high_risk,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.fraud_assessments[record["id"]] = record
		if is_high_risk:
			clm["fraud_flag"] = True
		clm["fraud_score"] = fraud_score
		clm["updated_at"] = self._now()
		self._emit(tenant, "fraud_assessed", record["id"], "ins_fraud_assessment", {"claim_id": claim_id, "score": fraud_score})
		return deepcopy(record)

	async def list_fraud_assessments(self, tenant_id: str, high_risk_only: bool = False) -> list[dict[str, Any]]:
		"""List fraud assessments."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(f) for f in self.fraud_assessments.values() if f["tenant_id"] == tenant]
		if high_risk_only:
			items = [f for f in items if f["high_risk"]]
		return items

	async def flag_claim_fraud(self, tenant_id: str, claim_id: str, reason: str, flagged_by: str) -> dict[str, Any]:
		"""Manually flag a claim as fraudulent."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		clm["fraud_flag"] = True
		clm["fraud_reason"] = reason
		clm["fraud_flagged_by"] = flagged_by
		clm["fraud_flagged_at"] = self._now()
		clm["updated_at"] = self._now()
		self._emit(tenant, "claim_fraud_flagged", claim_id, "ins_claim", {"reason": reason})
		return deepcopy(clm)

	async def repudiate_claim(self, tenant_id: str, claim_id: str, reason: str, authorised_by: str) -> dict[str, Any]:
		"""Repudiate (deny) a claim."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		if clm["status"] in {"fully_paid", "repudiated", "withdrawn"}:
			raise PermissionError(f"claim_cannot_be_repudiated_in_status:{clm['status']}")
		clm["status"] = "repudiated"
		clm["repudiation_reason"] = reason
		clm["repudiated_by"] = authorised_by
		clm["repudiated_at"] = self._now()
		clm["updated_at"] = self._now()
		self._emit(tenant, "claim_repudiated", claim_id, "ins_claim", {"reason": reason})
		return deepcopy(clm)

	async def approve_claim(self, tenant_id: str, claim_id: str, approved_amount: Decimal, approved_by: str) -> dict[str, Any]:
		"""Approve a claim for payment."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		if clm["status"] not in {"reserved", "under_assessment"}:
			raise PermissionError("claim_must_be_reserved_for_approval")
		clm["status"] = "approved"
		clm["approved_amount"] = Decimal(str(approved_amount))
		clm["approved_by"] = approved_by
		clm["approved_at"] = self._now()
		clm["updated_at"] = self._now()
		self._emit(tenant, "claim_approved", claim_id, "ins_claim", {"approved_amount": str(approved_amount)})
		return deepcopy(clm)

	# ── Subrogation ───────────────────────────────────────────────────────────

	async def initiate_subrogation(
		self,
		tenant_id: str,
		claim_id: str,
		third_party_name: str,
		third_party_id: str,
		recovery_amount: Decimal,
		legal_reference: str | None = None,
	) -> dict[str, Any]:
		"""Initiate subrogation recovery against a third party."""
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)
		if clm["status"] not in {"fully_paid", "partially_paid"}:
			raise PermissionError("subrogation_requires_paid_claim")
		record: dict[str, Any] = {
			"id": self._record_id("sub"),
			"type": "ins_subrogation",
			"claim_id": claim_id,
			"claim_number": clm["claim_number"],
			"third_party_name": third_party_name,
			"third_party_id": third_party_id,
			"recovery_amount": Decimal(str(recovery_amount)),
			"recovered_amount": Decimal("0"),
			"legal_reference": legal_reference,
			"status": "initiated",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.subrogations[record["id"]] = record
		clm["status"] = "subrogation"
		clm["updated_at"] = self._now()
		self._emit(tenant, "subrogation_initiated", record["id"], "ins_subrogation", {"claim_id": claim_id})
		return deepcopy(record)

	async def record_subrogation_recovery(self, tenant_id: str, subrogation_id: str, recovered_amount: Decimal) -> dict[str, Any]:
		"""Record a subrogation recovery payment received."""
		tenant = self._tenant(tenant_id)
		sub = self.subrogations.get(subrogation_id)
		if not sub or sub["tenant_id"] != tenant:
			raise KeyError(f"subrogation_not_found:{subrogation_id}")
		sub["recovered_amount"] = sub["recovered_amount"] + Decimal(str(recovered_amount))
		sub["status"] = "recovered" if sub["recovered_amount"] >= sub["recovery_amount"] else "partial_recovery"
		sub["last_recovery_at"] = self._now()
		self._emit(tenant, "subrogation_recovery_recorded", subrogation_id, "ins_subrogation", {"amount": str(recovered_amount)})
		return deepcopy(sub)

	async def list_subrogations(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List subrogation records."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(s) for s in self.subrogations.values() if s["tenant_id"] == tenant]

	# ── Analytics & Reports ───────────────────────────────────────────────────

	async def claims_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Summary statistics for claims portfolio."""
		tenant = self._tenant(tenant_id)
		clms = [c for c in self.claims.values() if c["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		total_reserve = Decimal("0")
		total_paid = Decimal("0")
		for c in clms:
			by_status[c["status"]] = by_status.get(c["status"], 0) + 1
			total_reserve += c["reserve_amount"]
			total_paid += c["paid_amount"]
		return {
			"tenant_id": tenant,
			"total_claims": len(clms),
			"by_status": by_status,
			"total_reserve": str(total_reserve),
			"total_paid": str(total_paid),
			"fraud_flagged": sum(1 for c in clms if c.get("fraud_flag")),
			"generated_at": self._now(),
		}

	async def loss_ratio_report(self, tenant_id: str, earned_premium: Decimal) -> dict[str, Any]:
		"""Calculate loss ratio against a given earned premium figure."""
		tenant = self._tenant(tenant_id)
		total_paid = sum(c["paid_amount"] for c in self.claims.values() if c["tenant_id"] == tenant)
		ep = Decimal(str(earned_premium))
		loss_ratio = (total_paid / ep * 100).quantize(Decimal("0.01")) if ep > 0 else Decimal("0")
		return {
			"tenant_id": tenant,
			"earned_premium": str(ep),
			"incurred_losses": str(total_paid),
			"loss_ratio_pct": str(loss_ratio),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Service health status."""
		return {
			"service": "ins_clm",
			"status": "healthy",
			"claim_count": len(self.claims),
			"open_reserves": len([r for r in self.reserves.values()]),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Describe this capability."""
		return {
			"capability_id": "ins_clm",
			"name": "Claims Management",
			"version": "1.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"supported_statuses": list(SUPPORTED_CLAIM_STATUSES),
			"supported_payment_types": list(SUPPORTED_PAYMENT_TYPES),
		}

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return audit trail."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── STP (Straight-Through Processing) ────────────────────────────────────

	async def evaluate_stp_eligibility(
		self,
		tenant_id: str,
		claim_id: str,
		stp_loss_ceiling: Decimal = Decimal("50000"),
		lookback_days: int = 90,
	) -> dict[str, Any]:
		"""Evaluate whether a claim qualifies for straight-through processing.

		Returns eligibility verdict plus reason vector. If eligible, automatically
		advances the claim through reserve → approve → payment in one atomic
		transaction and emits `stp_auto_approved`.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)

		reasons: list[str] = []
		eligible = True

		if clm["status"] not in {"fnol", "under_assessment"}:
			eligible = False
			reasons.append(f"ineligible_status:{clm['status']}")

		if clm["fraud_flag"] or clm["fraud_score"] >= FRAUD_HIGH_THRESHOLD:
			eligible = False
			reasons.append("fraud_flag_or_high_score")

		loss = Decimal(str(clm["estimated_loss"]))
		if loss > stp_loss_ceiling:
			eligible = False
			reasons.append(f"loss_exceeds_ceiling:{loss}>{stp_loss_ceiling}")

		# velocity check: count other claims for same policy in lookback window
		from datetime import timedelta
		cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat(timespec="seconds") + "Z"
		prior_count = sum(
			1 for c in self.claims.values()
			if c["tenant_id"] == tenant
			and c["policy_id"] == clm["policy_id"]
			and c["id"] != claim_id
			and c.get("created_at", "") >= cutoff
		)
		if prior_count > 0:
			eligible = False
			reasons.append(f"prior_claims_in_window:{prior_count}")

		result: dict[str, Any] = {
			"claim_id": claim_id,
			"eligible": eligible,
			"reasons": reasons,
			"evaluated_at": self._now(),
		}

		if eligible:
			# auto-progress: set reserve = estimated_loss, approve, no payment disbursed yet
			clm["reserve_amount"] = loss
			clm["status"] = "approved"
			clm["approved_amount"] = loss
			clm["approved_by"] = "stp_engine"
			clm["approved_at"] = self._now()
			clm["updated_at"] = self._now()
			clm["stp_settled"] = True
			self._emit(tenant, "stp_auto_approved", claim_id, "ins_claim", {"loss": str(loss)})
			result["auto_approved_amount"] = str(loss)
			_log.info("STP auto-approved claim=%s tenant=%s amount=%s", claim_id, tenant, loss)

		return result

	# ── Claim Complexity Triage ───────────────────────────────────────────────

	async def score_claim_complexity(
		self,
		tenant_id: str,
		claim_id: str,
		injury_involved: bool = False,
		commercial_vehicle: bool = False,
		catastrophe_code: str | None = None,
	) -> dict[str, Any]:
		"""Score claim complexity at FNOL to drive adjuster routing.

		Returns a complexity_tier: simple | standard | complex | catastrophic
		and a numeric score (0-100) with feature-level explanations.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)

		score = 0
		features: dict[str, Any] = {}

		loss = Decimal(str(clm["estimated_loss"]))
		if loss < Decimal("50000"):
			features["loss_band"] = "low"
		elif loss < Decimal("500000"):
			score += 20
			features["loss_band"] = "medium"
		elif loss < Decimal("5000000"):
			score += 40
			features["loss_band"] = "high"
		else:
			score += 60
			features["loss_band"] = "very_high"

		if injury_involved:
			score += 20
			features["injury"] = True
		if commercial_vehicle:
			score += 10
			features["commercial_vehicle"] = True
		if catastrophe_code:
			score += 20
			features["catastrophe_code"] = catastrophe_code
		if clm["fraud_flag"]:
			score += 15
			features["fraud_flagged"] = True

		score = min(score, 100)

		if score < 20:
			tier = "simple"
		elif score < 50:
			tier = "standard"
		elif score < 75:
			tier = "complex"
		else:
			tier = "catastrophic"

		clm["complexity_score"] = score
		clm["complexity_tier"] = tier
		clm["updated_at"] = self._now()
		self._emit(tenant, "complexity_scored", claim_id, "ins_claim", {"tier": tier, "score": score})

		return {
			"claim_id": claim_id,
			"complexity_score": score,
			"complexity_tier": tier,
			"features": features,
			"scored_at": self._now(),
		}

	# ── Reserve Adequacy ──────────────────────────────────────────────────────

	async def check_reserve_adequacy(
		self,
		tenant_id: str,
		claim_id: str,
	) -> dict[str, Any]:
		"""Evaluate reserve adequacy based on paid trajectory.

		Projects months-to-reserve-exhaustion from recent payment run-rate.
		Returns adequacy_status: adequate | warning | critical.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)

		reserve = Decimal(str(clm["reserve_amount"]))
		paid = Decimal(str(clm["paid_amount"]))

		if reserve == Decimal("0"):
			return {
				"claim_id": claim_id,
				"adequacy_status": "no_reserve",
				"reserve_utilisation": None,
				"months_to_exhaustion": None,
				"recommended_top_up": None,
				"checked_at": self._now(),
			}

		utilisation = (paid / reserve).quantize(Decimal("0.0001"))
		outstanding = reserve - paid

		# estimate monthly burn from payment records for this claim
		claim_payments = [
			p for p in self.payments.values()
			if p["claim_id"] == claim_id and p["tenant_id"] == tenant
		]
		if len(claim_payments) >= 2:
			# sort by created_at, take last 3 to compute average monthly disbursement
			sorted_pays = sorted(claim_payments, key=lambda x: x["created_at"])[-3:]
			total_recent = sum(Decimal(str(p["payment_amount"])) for p in sorted_pays)
			avg_monthly = (total_recent / Decimal(str(len(sorted_pays)))).quantize(Decimal("0.01"))
		else:
			avg_monthly = Decimal("0")

		if avg_monthly > 0:
			months_to_exhaustion = float((outstanding / avg_monthly).quantize(Decimal("0.1")))
		else:
			months_to_exhaustion = None

		if utilisation >= Decimal("0.95"):
			adequacy_status = "critical"
		elif utilisation >= Decimal("0.85"):
			adequacy_status = "warning"
		else:
			adequacy_status = "adequate"

		# recommend top-up to restore to 120% coverage of outstanding
		recommended_top_up = None
		if adequacy_status in {"warning", "critical"}:
			recommended_top_up = str((outstanding * Decimal("1.2") - outstanding).quantize(Decimal("0.01")))

		if adequacy_status in {"warning", "critical"}:
			self._emit(tenant, "reserve_adequacy_warning", claim_id, "ins_claim", {
				"status": adequacy_status, "utilisation": str(utilisation),
			})

		return {
			"claim_id": claim_id,
			"reserve_amount": str(reserve),
			"paid_amount": str(paid),
			"outstanding": str(outstanding),
			"reserve_utilisation": str(utilisation),
			"adequacy_status": adequacy_status,
			"avg_monthly_payments": str(avg_monthly),
			"months_to_exhaustion": months_to_exhaustion,
			"recommended_top_up": recommended_top_up,
			"checked_at": self._now(),
		}

	# ── Claim Velocity Check ──────────────────────────────────────────────────

	async def check_claim_velocity(
		self,
		tenant_id: str,
		policy_id: str,
		claimant_id: str,
		window_days: int = 30,
	) -> dict[str, Any]:
		"""Detect anomalous claim submission bursts per policy or claimant.

		Returns velocity_risk_level: low | medium | high with counts and
		emits `velocity_alert` when z-score > 2.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)

		from datetime import timedelta
		cutoff = (datetime.utcnow() - timedelta(days=window_days)).isoformat(timespec="seconds") + "Z"

		policy_claims_window = [
			c for c in self.claims.values()
			if c["tenant_id"] == tenant
			and c["policy_id"] == policy_id
			and c.get("created_at", "") >= cutoff
		]
		claimant_claims_window = [
			c for c in self.claims.values()
			if c["tenant_id"] == tenant
			and c["claimant_id"] == claimant_id
			and c.get("created_at", "") >= cutoff
		]

		policy_count = len(policy_claims_window)
		claimant_count = len(claimant_claims_window)

		# simple threshold-based risk tiers (replace with z-score once baseline built)
		if policy_count >= 5 or claimant_count >= 4:
			risk_level = "high"
		elif policy_count >= 3 or claimant_count >= 2:
			risk_level = "medium"
		else:
			risk_level = "low"

		if risk_level == "high":
			self._emit(tenant, "velocity_alert", policy_id, "ins_policy", {
				"policy_count": policy_count,
				"claimant_count": claimant_count,
				"window_days": window_days,
			})
			_log.warning("Velocity alert policy=%s claimant=%s tenant=%s", policy_id, claimant_id, tenant)

		return {
			"policy_id": policy_id,
			"claimant_id": claimant_id,
			"window_days": window_days,
			"policy_claims_count": policy_count,
			"claimant_claims_count": claimant_count,
			"velocity_risk_level": risk_level,
			"checked_at": self._now(),
		}

	# ── Excess / Deductible Application ──────────────────────────────────────

	async def compute_applicable_excess(
		self,
		tenant_id: str,
		claim_id: str,
		excess_schedule: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Compute and apply policy excesses to a claim's payable amount.

		excess_schedule format: list of {type, amount, applies_when (optional predicate key)}
		Supported applies_when keys: always, young_driver, commercial, voluntary.

		Returns net_payable after stacking all applicable excesses and updates claim.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)

		if clm["status"] not in {"reserved", "approved"}:
			raise PermissionError("excess_computation_requires_reserved_or_approved_claim")

		gross_loss = Decimal(str(clm["estimated_loss"]))
		total_excess = Decimal("0")
		applied: list[dict[str, Any]] = []

		for rule in excess_schedule:
			applies_when = rule.get("applies_when", "always")
			excess_amount = Decimal(str(rule["amount"]))
			apply = False
			if applies_when == "always":
				apply = True
			elif applies_when == "young_driver" and clm.get("metadata", {}).get("young_driver"):
				apply = True
			elif applies_when == "commercial" and clm.get("metadata", {}).get("commercial_vehicle"):
				apply = True
			elif applies_when == "voluntary":
				apply = True

			if apply:
				total_excess += excess_amount
				applied.append({"type": rule.get("type", "basic"), "amount": str(excess_amount)})

		net_payable = max(Decimal("0"), gross_loss - total_excess)
		clm["excess_applied"] = str(total_excess)
		clm["net_payable_amount"] = str(net_payable)
		clm["updated_at"] = self._now()

		self._emit(tenant, "excess_computed", claim_id, "ins_claim", {
			"gross_loss": str(gross_loss),
			"total_excess": str(total_excess),
			"net_payable": str(net_payable),
		})

		return {
			"claim_id": claim_id,
			"gross_loss": str(gross_loss),
			"applied_excesses": applied,
			"total_excess": str(total_excess),
			"net_payable": str(net_payable),
			"computed_at": self._now(),
		}

	# ── Litigation Management ─────────────────────────────────────────────────

	async def open_litigation_matter(
		self,
		tenant_id: str,
		claim_id: str,
		law_firm_id: str,
		case_reference: str,
		court: str,
		first_hearing_date: str,
		litigation_reserve_uplift: Decimal = Decimal("0"),
		opened_by: str = "",
	) -> dict[str, Any]:
		"""Open a litigation matter linked to a claim.

		Uplifts the claim reserve by litigation_reserve_uplift and sets
		claim status to `litigation`. Stores matter in self.litigations.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(case_reference, "case_reference")
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)

		if not hasattr(self, "litigations"):
			self.litigations: dict[str, dict[str, Any]] = {}

		record: dict[str, Any] = {
			"id": self._record_id("lit"),
			"type": "ins_litigation",
			"claim_id": claim_id,
			"claim_number": clm["claim_number"],
			"law_firm_id": law_firm_id,
			"case_reference": case_reference,
			"court": court,
			"first_hearing_date": first_hearing_date,
			"phase": "filed",
			"events": [],
			"legal_costs": Decimal("0"),
			"status": "active",
			"opened_by": opened_by,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.litigations[record["id"]] = record

		if litigation_reserve_uplift > 0:
			uplift = Decimal(str(litigation_reserve_uplift))
			clm["reserve_amount"] = clm["reserve_amount"] + uplift

		clm["litigation_matter_id"] = record["id"]
		clm["updated_at"] = self._now()
		self._emit(tenant, "litigation_opened", record["id"], "ins_litigation", {"claim_id": claim_id, "court": court})
		_log.info("Litigation opened matter=%s claim=%s tenant=%s", record["id"], claim_id, tenant)
		return deepcopy(record)

	async def log_litigation_event(
		self,
		tenant_id: str,
		litigation_id: str,
		event_type: str,
		description: str,
		legal_cost: Decimal = Decimal("0"),
		new_phase: str | None = None,
	) -> dict[str, Any]:
		"""Record an event in the litigation matter timeline (hearing, filing, settlement)."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		if not hasattr(self, "litigations"):
			self.litigations = {}
		lit = self.litigations.get(litigation_id)
		if not lit or lit["tenant_id"] != tenant:
			raise KeyError(f"litigation_not_found:{litigation_id}")

		event: dict[str, Any] = {
			"event_type": event_type,
			"description": description,
			"legal_cost": Decimal(str(legal_cost)),
			"recorded_at": self._now(),
		}
		lit["events"].append(event)
		lit["legal_costs"] = lit["legal_costs"] + Decimal(str(legal_cost))
		if new_phase:
			lit["phase"] = new_phase
		if new_phase in {"settled", "dismissed"}:
			lit["status"] = "closed"
			lit["closed_at"] = self._now()

		self._emit(tenant, "litigation_event_logged", litigation_id, "ins_litigation", {
			"event_type": event_type, "phase": lit["phase"],
		})
		return deepcopy(lit)

	# ── Multi-Currency Support ────────────────────────────────────────────────

	async def convert_claim_currency(
		self,
		tenant_id: str,
		claim_id: str,
		target_currency: str,
		fx_rate: Decimal,
		fx_source: str,
	) -> dict[str, Any]:
		"""Record FX conversion for a claim's monetary amounts.

		Stores original values with full FX provenance and sets reporting-currency
		fields on the claim. Rate is immutable once recorded (audit compliance).
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		clm = self._get_claim(claim_id, tenant)

		if fx_rate <= Decimal("0"):
			raise ValueError("fx_rate_must_be_positive")

		rate = Decimal(str(fx_rate))
		original_currency = clm["currency"]
		if original_currency == target_currency:
			raise ValueError("source_and_target_currency_identical")

		conversion: dict[str, Any] = {
			"id": self._record_id("fx"),
			"claim_id": claim_id,
			"from_currency": original_currency,
			"to_currency": target_currency,
			"fx_rate": str(rate),
			"fx_source": fx_source,
			"original_estimated_loss": str(clm["estimated_loss"]),
			"converted_estimated_loss": str((clm["estimated_loss"] * rate).quantize(Decimal("0.01"))),
			"original_reserve": str(clm["reserve_amount"]),
			"converted_reserve": str((clm["reserve_amount"] * rate).quantize(Decimal("0.01"))),
			"original_paid": str(clm["paid_amount"]),
			"converted_paid": str((clm["paid_amount"] * rate).quantize(Decimal("0.01"))),
			"converted_at": self._now(),
			"tenant_id": tenant,
		}

		clm["fx_conversions"] = clm.get("fx_conversions", [])
		clm["fx_conversions"].append(conversion)
		clm["reporting_currency"] = target_currency
		clm["reporting_estimated_loss"] = str((clm["estimated_loss"] * rate).quantize(Decimal("0.01")))
		clm["updated_at"] = self._now()

		self._emit(tenant, "claim_currency_converted", claim_id, "ins_claim", {
			"from": original_currency, "to": target_currency, "rate": str(rate),
		})
		return deepcopy(conversion)

	# ── Regulatory Large-Loss Notification ───────────────────────────────────

	async def generate_large_loss_notifications(
		self,
		tenant_id: str,
		threshold: Decimal = Decimal("1000000"),
		lookback_hours: int = 24,
	) -> list[dict[str, Any]]:
		"""Identify claims exceeding the regulatory large-loss threshold.

		Returns structured notification records suitable for IRA Kenya Form C-4
		or equivalent regulatory submission. Claims are those created or updated
		within the lookback window with estimated_loss >= threshold.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		from datetime import timedelta
		cutoff = (datetime.utcnow() - timedelta(hours=lookback_hours)).isoformat(timespec="seconds") + "Z"
		threshold_d = Decimal(str(threshold))

		notifications: list[dict[str, Any]] = []
		for clm in self.claims.values():
			if clm["tenant_id"] != tenant:
				continue
			created = clm.get("created_at", "")
			updated = clm.get("updated_at") or ""
			if created < cutoff and updated < cutoff:
				continue
			loss = Decimal(str(clm["estimated_loss"]))
			if loss < threshold_d:
				continue
			notifications.append({
				"notification_type": "large_loss",
				"claim_number": clm["claim_number"],
				"claim_id": clm["id"],
				"policy_id": clm["policy_id"],
				"claimant_name": clm["claimant_name"],
				"estimated_loss": str(loss),
				"currency": clm["currency"],
				"incident_date": clm.get("incident_date", ""),
				"status": clm["status"],
				"fraud_flag": clm.get("fraud_flag", False),
				"tenant_id": tenant,
				"generated_at": self._now(),
			})

		_log.info(
			"Large-loss scan: %d notifications threshold=%s tenant=%s",
			len(notifications), threshold_d, tenant,
		)
		return notifications

	# ── Portfolio SLA Compliance Dashboard ───────────────────────────────────

	async def sla_compliance_dashboard(
		self,
		tenant_id: str,
		acknowledge_hours: int = 72,
		assess_days: int = 14,
		settle_days: int = 90,
	) -> dict[str, Any]:
		"""Return portfolio-level SLA compliance heat-map.

		Evaluates each open claim against configurable SLA ladders and classifies
		each as compliant, warning (>80% of SLA elapsed), or breached.
		Emits `sla_breach_detected` events for newly breached claims.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		from datetime import timedelta

		now = datetime.utcnow()
		summary: dict[str, Any] = {
			"compliant": 0, "warning": 0, "breached": 0, "by_claim": [],
		}

		for clm in self.claims.values():
			if clm["tenant_id"] != tenant:
				continue
			if clm["status"] in {"fully_paid", "repudiated", "withdrawn"}:
				continue

			created_dt = datetime.fromisoformat(clm["created_at"].rstrip("Z"))
			elapsed_hours = (now - created_dt).total_seconds() / 3600
			elapsed_days = elapsed_hours / 24

			sla_limit_days = settle_days
			sla_pct = min(elapsed_days / sla_limit_days, 1.0) if sla_limit_days > 0 else 1.0

			if sla_pct >= 1.0:
				sla_status = "breached"
				self._emit(tenant, "sla_breach_detected", clm["id"], "ins_claim", {
					"elapsed_days": round(elapsed_days, 1), "limit_days": sla_limit_days,
				})
			elif sla_pct >= 0.8:
				sla_status = "warning"
			else:
				sla_status = "compliant"

			summary[sla_status] += 1
			summary["by_claim"].append({
				"claim_id": clm["id"],
				"claim_number": clm["claim_number"],
				"status": clm["status"],
				"elapsed_days": round(elapsed_days, 1),
				"sla_pct_used": round(sla_pct * 100, 1),
				"sla_status": sla_status,
			})

		summary["total_open"] = summary["compliant"] + summary["warning"] + summary["breached"]
		summary["tenant_id"] = tenant
		summary["generated_at"] = self._now()
		return summary

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

