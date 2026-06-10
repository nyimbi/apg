"""Claims Management Service (ins_clm).

Handles FNOL, assessment, reserve management, payments, fraud detection, and subrogation.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

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

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.claims: dict[str, dict[str, Any]] = {}
		self.reserves: dict[str, dict[str, Any]] = {}
		self.payments: dict[str, dict[str, Any]] = {}
		self.fraud_assessments: dict[str, dict[str, Any]] = {}
		self.subrogations: dict[str, dict[str, Any]] = {}
		self.assessments: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
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
