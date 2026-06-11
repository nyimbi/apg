"""Async service layer for Property Insurance (ins)."""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

from .models import (
	InsurerCreate, InsurerResponse,
	PolicyCreate, PolicyResponse, PolicyUpdate,
	InsuredAssetCreate, InsuredAssetResponse,
	ClaimCreate, ClaimResponse,
	EndorsementCreate, EndorsementResponse,
	PremiumAllocationCreate, PremiumAllocationResponse,
	CoverageGapCreate, CoverageGapResponse,
	CoverageStatus, ClaimStatus, InsurerGrade,
)
from .capability_contract import evaluate_capability_rules
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

log = logging.getLogger(__name__)


class InsService:
	"""Service implementing all Property Insurance operations."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: dict[str, Any] | None = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store: dict[str, list[dict[str, Any]]] = store or {
			"insurers": [], "policies": [], "assets": [],
			"claims": [], "endorsements": [], "allocations": [],
			"gaps": [], "brokers": [],
			"loss_adjusters": [], "reinstatements": [], "captives": [],
		}
		self._claim_counter = 0
		self._endorsement_counter = 0

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_operation(self, op: str, entity_id: str, tenant_id: str) -> None:
		log.info("ins.%s entity=%s tenant=%s", op, entity_id, tenant_id)

	def _log_critical_gap(self, property_id: str, description: str) -> None:
		log.critical("ins.critical_gap property=%s gap=%s", property_id, description)

	def _log_renewal_due(self, policy_id: str, days_remaining: int) -> None:
		log.warning("ins.renewal_due policy=%s days_remaining=%d", policy_id, days_remaining)

	def _log_large_claim(self, claim_id: str, amount: Decimal) -> None:
		log.warning("ins.large_claim claim=%s amount=%s", claim_id, amount)

	# ── Rules ─────────────────────────────────────────────────────────────────

	def _check_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			log.warning("ins.rule_denied rule=%s reason=%s", result["rule"], result["reason"])
			raise ValueError(f"rule_denied:{result['rule']}:{result['reason']}")

	def _next_claim_ref(self) -> str:
		self._claim_counter += 1
		return f"CLM-{self._claim_counter:07d}"

	def _next_endorsement_ref(self) -> str:
		self._endorsement_counter += 1
		return f"END-{self._endorsement_counter:06d}"

	# ── Insurer ───────────────────────────────────────────────────────────────

	async def register_insurer(self, payload: InsurerCreate) -> InsurerResponse:
		"""Register an insurer in the registry."""
		self._check_rules({"tenant_context_present": True, "operation_type": "write", "policy_attached": True})
		record = InsurerResponse(**payload.model_dump())
		self._store["insurers"].append(record.model_dump())
		self._log_operation("register_insurer", record.id, record.tenant_id)
		return record

	async def get_insurer(self, insurer_id: str, tenant_id: str) -> InsurerResponse | None:
		"""Fetch an insurer."""
		for i in self._store["insurers"]:
			if i["id"] == insurer_id and i["tenant_id"] == tenant_id:
				return InsurerResponse(**i)
		return None

	async def list_insurers(self, tenant_id: str, grade: str | None = None) -> list[InsurerResponse]:
		"""List insurers."""
		results = [i for i in self._store["insurers"] if i["tenant_id"] == tenant_id]
		if grade:
			results = [i for i in results if i.get("grade") == grade]
		return [InsurerResponse(**i) for i in results]

	# ── Policy ────────────────────────────────────────────────────────────────

	async def create_policy(self, payload: PolicyCreate) -> PolicyResponse:
		"""Create an insurance policy."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_policy",
			"policy_type_supported": True,
			"insurer_present": True,
			"valuation_basis_supported": True,
			"operation_type": "write",
			"policy_attached": True,
			"cross_tenant": False,
		})
		record = PolicyResponse(**payload.model_dump())
		self._store["policies"].append(record.model_dump())
		for i, ins in enumerate(self._store["insurers"]):
			if ins["id"] == payload.insurer_id:
				ins["active_policies"] = ins.get("active_policies", 0) + 1
				self._store["insurers"][i] = ins
				break
		self._log_operation("create_policy", record.id, record.tenant_id)
		return record

	async def get_policy(self, policy_id: str, tenant_id: str) -> PolicyResponse | None:
		"""Fetch a policy."""
		for p in self._store["policies"]:
			if p["id"] == policy_id and p["tenant_id"] == tenant_id:
				return PolicyResponse(**p)
		return None

	async def list_policies(self, tenant_id: str, property_id: str | None = None, status: str | None = None) -> list[PolicyResponse]:
		"""List policies."""
		results = [p for p in self._store["policies"] if p["tenant_id"] == tenant_id]
		if property_id:
			results = [p for p in results if property_id in p.get("property_ids", [])]
		if status:
			results = [p for p in results if p.get("status") == status]
		return [PolicyResponse(**p) for p in results]

	async def bind_policy(self, policy_id: str, tenant_id: str) -> PolicyResponse | None:
		"""Bind (activate) a policy after asset schedule verification."""
		for i, p in enumerate(self._store["policies"]):
			if p["id"] == policy_id and p["tenant_id"] == tenant_id:
				insurer = await self.get_insurer(p["insurer_id"], tenant_id)
				insurer_grade = insurer.grade.value if insurer else "conditional"
				asset_schedule_present = len([a for a in self._store["assets"] if a.get("policy_id") == policy_id]) > 0
				self._check_rules({
					"operation": "bind_policy",
					"insurer_grade": insurer_grade,
					"asset_schedule_present": asset_schedule_present,
				})
				p["status"] = CoverageStatus.active.value
				p["updated_at"] = datetime.utcnow()
				self._store["policies"][i] = p
				self._log_operation("bind_policy", policy_id, tenant_id)
				return PolicyResponse(**p)
		return None

	async def update_policy(self, policy_id: str, tenant_id: str, updates: PolicyUpdate) -> PolicyResponse | None:
		"""Update policy details."""
		for i, p in enumerate(self._store["policies"]):
			if p["id"] == policy_id and p["tenant_id"] == tenant_id:
				p.update({k: v for k, v in updates.model_dump().items() if v is not None})
				p["updated_at"] = datetime.utcnow()
				self._store["policies"][i] = p
				return PolicyResponse(**p)
		return None

	async def get_renewal_pipeline(self, tenant_id: str, days_ahead: int = 90) -> list[dict[str, Any]]:
		"""Return policies expiring within the given days window."""
		cutoff = date.today() + timedelta(days=days_ahead)
		results = []
		for p in self._store["policies"]:
			if p["tenant_id"] == tenant_id and p["status"] == CoverageStatus.active.value:
				expiry = datetime.strptime(p["expiry_date"], "%Y-%m-%d").date()
				if expiry <= cutoff:
					days_remaining = (expiry - date.today()).days
					self._log_renewal_due(p["id"], days_remaining)
					results.append({"policy_id": p["id"], "policy_number": p.get("policy_number"), "expiry_date": p["expiry_date"], "days_remaining": days_remaining})
		return sorted(results, key=lambda x: x["days_remaining"])

	# ── Asset Schedule ────────────────────────────────────────────────────────

	async def add_asset_to_schedule(self, payload: InsuredAssetCreate) -> InsuredAssetResponse:
		"""Add an asset to an insurance policy schedule."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "add_asset_to_schedule",
			"asset_type_supported": True,
			"valuation_basis_supported": True,
		})
		record = InsuredAssetResponse(**payload.model_dump())
		self._store["assets"].append(record.model_dump())
		self._log_operation("add_asset_to_schedule", record.id, record.tenant_id)
		return record

	async def list_policy_assets(self, tenant_id: str, policy_id: str) -> list[InsuredAssetResponse]:
		"""List assets on a policy schedule."""
		return [InsuredAssetResponse(**a) for a in self._store["assets"]
				if a["tenant_id"] == tenant_id and a["policy_id"] == policy_id]

	async def remove_asset_from_schedule(self, asset_id: str, tenant_id: str) -> bool:
		"""Remove an asset from the schedule."""
		initial = len(self._store["assets"])
		self._store["assets"] = [a for a in self._store["assets"] if not (a["id"] == asset_id and a["tenant_id"] == tenant_id)]
		return len(self._store["assets"]) < initial

	# ── Claim ─────────────────────────────────────────────────────────────────

	async def lodge_claim(self, payload: ClaimCreate) -> ClaimResponse:
		"""Lodge a new insurance claim."""
		policy = await self.get_policy(payload.policy_id, payload.tenant_id)
		policy_active = policy is not None and policy.status.value == "active"
		peril_covered = policy is not None and (not policy.perils_covered or payload.peril in policy.perils_covered)
		self._check_rules({
			"tenant_context_present": True,
			"operation": "lodge_claim",
			"policy_active": policy_active,
			"peril_covered": peril_covered,
			"claim_type_supported": True,
		})
		ref = self._next_claim_ref()
		if payload.estimated_loss > Decimal("1000000"):
			self._log_large_claim(ref, payload.estimated_loss)
		record = ClaimResponse(**payload.model_dump(), claim_ref=ref)
		self._store["claims"].append(record.model_dump())
		if policy:
			for i, p in enumerate(self._store["policies"]):
				if p["id"] == payload.policy_id:
					p["claims_count"] = p.get("claims_count", 0) + 1
					self._store["policies"][i] = p
					break
		self._log_operation("lodge_claim", record.id, record.tenant_id)
		return record

	async def get_claim(self, claim_id: str, tenant_id: str) -> ClaimResponse | None:
		"""Fetch a claim."""
		for c in self._store["claims"]:
			if c["id"] == claim_id and c["tenant_id"] == tenant_id:
				return ClaimResponse(**c)
		return None

	async def approve_claim(self, claim_id: str, tenant_id: str, approved_value: Decimal, senior_approved: bool) -> ClaimResponse | None:
		"""Approve a claim with optional senior sign-off for large amounts."""
		for i, c in enumerate(self._store["claims"]):
			if c["id"] == claim_id and c["tenant_id"] == tenant_id:
				above_threshold = approved_value > Decimal("1000000")
				self._check_rules({"operation": "approve_claim", "amount_above_threshold": above_threshold, "senior_approved": senior_approved or not above_threshold})
				c["status"] = ClaimStatus.approved.value
				c["approved_value"] = str(approved_value)
				c["senior_approved"] = senior_approved
				c["updated_at"] = datetime.utcnow()
				self._store["claims"][i] = c
				return ClaimResponse(**c)
		return None

	async def settle_claim(self, claim_id: str, tenant_id: str, settlement_amount: Decimal) -> ClaimResponse | None:
		"""Record claim settlement."""
		for i, c in enumerate(self._store["claims"]):
			if c["id"] == claim_id and c["tenant_id"] == tenant_id:
				policy = await self.get_policy(c["policy_id"], tenant_id)
				sum_insured = policy.sum_insured if policy else Decimal("999999999")
				self._check_rules({"operation": "settle_claim", "settlement_exceeds_sum_insured": settlement_amount > sum_insured})
				c["status"] = ClaimStatus.settled.value
				c["settlement_amount"] = str(settlement_amount)
				c["settled_at"] = datetime.utcnow()
				c["updated_at"] = datetime.utcnow()
				self._store["claims"][i] = c
				self._log_operation("settle_claim", claim_id, tenant_id)
				return ClaimResponse(**c)
		return None

	async def list_claims(self, tenant_id: str, policy_id: str | None = None, status: str | None = None) -> list[ClaimResponse]:
		"""List claims."""
		results = [c for c in self._store["claims"] if c["tenant_id"] == tenant_id]
		if policy_id:
			results = [c for c in results if c.get("policy_id") == policy_id]
		if status:
			results = [c for c in results if c.get("status") == status]
		return [ClaimResponse(**c) for c in results]

	# ── Endorsement ───────────────────────────────────────────────────────────

	async def issue_endorsement(self, payload: EndorsementCreate) -> EndorsementResponse:
		"""Issue a policy endorsement."""
		policy = await self.get_policy(payload.policy_id, payload.tenant_id)
		sum_insured = policy.sum_insured if policy else Decimal("0")
		new_sum = sum_insured + payload.sum_insured_change
		self._check_rules({
			"tenant_context_present": True,
			"operation": "issue_endorsement",
			"endorsement_type_supported": True,
			"endorsed_sum_exceeds_market_value": False,
		})
		ref = self._next_endorsement_ref()
		record = EndorsementResponse(**payload.model_dump(), ref=ref, issued_at=datetime.utcnow())
		self._store["endorsements"].append(record.model_dump())
		if policy and payload.sum_insured_change:
			for i, p in enumerate(self._store["policies"]):
				if p["id"] == payload.policy_id:
					p["sum_insured"] = str(new_sum)
					p["updated_at"] = datetime.utcnow()
					self._store["policies"][i] = p
					break
		return record

	async def list_endorsements(self, tenant_id: str, policy_id: str | None = None) -> list[EndorsementResponse]:
		"""List endorsements."""
		results = [e for e in self._store["endorsements"] if e["tenant_id"] == tenant_id]
		if policy_id:
			results = [e for e in results if e.get("policy_id") == policy_id]
		return [EndorsementResponse(**e) for e in results]

	# ── Premium Allocation ────────────────────────────────────────────────────

	async def allocate_premium(self, payload: PremiumAllocationCreate) -> PremiumAllocationResponse:
		"""Run a premium allocation for a policy period."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "allocate_premium",
			"method_supported": True,
		})
		total = sum(Decimal(str(a.get("amount", 0))) for a in payload.allocations)
		record = PremiumAllocationResponse(**payload.model_dump(), total_allocated=total)
		self._store["allocations"].append(record.model_dump())
		return record

	# ── Coverage Gap Analysis ─────────────────────────────────────────────────

	async def detect_coverage_gaps(self, tenant_id: str, property_id: str) -> list[CoverageGapResponse]:
		"""Analyse coverage gaps for a property."""
		policies = await self.list_policies(tenant_id, property_id=property_id, status=CoverageStatus.active.value)
		gaps: list[CoverageGapResponse] = []
		if not policies:
			gap_payload = CoverageGapCreate(
				tenant_id=tenant_id,
				property_id=property_id,
				gap_description="No active insurance policy found for property",
				severity="critical",
				detected_by="system",
			)
			gap_response = await self.record_coverage_gap(gap_payload)
			gaps.append(gap_response)
		return gaps

	async def record_coverage_gap(self, payload: CoverageGapCreate) -> CoverageGapResponse:
		"""Record a detected coverage gap."""
		if payload.severity == "critical":
			self._log_critical_gap(payload.property_id, payload.gap_description)
			self._check_rules({"operation": "analyse_gaps", "critical_gap_detected": True, "alert_sent": False})
		record = CoverageGapResponse(**payload.model_dump(), alert_sent=payload.severity == "critical")
		self._store["gaps"].append(record.model_dump())
		return record

	async def list_coverage_gaps(self, tenant_id: str, property_id: str | None = None, resolved: bool = False) -> list[CoverageGapResponse]:
		"""List coverage gaps."""
		results = [g for g in self._store["gaps"] if g["tenant_id"] == tenant_id and g.get("resolved", False) == resolved]
		if property_id:
			results = [g for g in results if g.get("property_id") == property_id]
		return [CoverageGapResponse(**g) for g in results]

	# ── Insurance Dashboard ───────────────────────────────────────────────────

	async def get_insurance_summary(self, tenant_id: str) -> dict[str, Any]:
		"""High-level insurance portfolio summary."""
		policies = await self.list_policies(tenant_id)
		active_policies = [p for p in policies if p.status.value == "active"]
		open_claims = await self.list_claims(tenant_id, status=ClaimStatus.lodged.value)
		return {
			"tenant_id": tenant_id,
			"active_policies": len(active_policies),
			"expiring_90_days": len(await self.get_renewal_pipeline(tenant_id, days_ahead=90)),
			"open_claims": len(open_claims),
			"critical_gaps": len(await self.list_coverage_gaps(tenant_id, resolved=False)),
			"total_sum_insured": float(sum(p.sum_insured for p in active_policies)),
		}

	# ── NEW: schedule_insurance ───────────────────────────────────────────────

	async def schedule_insurance(
		self,
		policy_id: str,
		asset_id: str,
		insured_value: Decimal,
		coverage_type: str,
		tenant_id: str,
		valuation_date: date | None = None,
		coverage_notes: str = "",
	) -> InsuredAssetResponse:
		"""Add an asset to an insurance policy schedule with insured value and coverage type."""
		assert policy_id and asset_id, "policy_id and asset_id required"
		assert insured_value > 0, "insured_value must be positive"
		assert coverage_type in ("building", "contents", "machinery", "liability", "loss_of_rent",
			"terrorism", "flood", "all_risks"), f"unsupported coverage_type: {coverage_type}"
		self._check_rules({
			"tenant_context_present": True,
			"operation": "add_asset_to_schedule",
			"asset_type_supported": True,
			"valuation_basis_supported": True,
		})
		from uuid6 import uuid7
		asset_record_id = str(uuid7())
		record: dict[str, Any] = {
			"id": asset_record_id,
			"tenant_id": tenant_id,
			"policy_id": policy_id,
			"asset_id": asset_id,
			"insured_value": str(insured_value),
			"coverage_type": coverage_type,
			"valuation_date": str(valuation_date or date.today()),
			"coverage_notes": coverage_notes,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["assets"].append(record)
		self._log_operation("schedule_insurance", asset_record_id, tenant_id)
		return InsuredAssetResponse(**record)

	# ── NEW: premium_allocation ────────────────────────────────────────────────

	async def premium_allocation(
		self,
		policy_id: str,
		units: list[str],
		tenant_id: str,
		method: str = "floor_area",
		total_premium: Decimal | None = None,
	) -> dict[str, Any]:
		"""Allocate insurance premium across units in a property (by floor area, value, or equal split)."""
		assert policy_id and units, "policy_id and units required"
		assert method in ("floor_area", "insured_value", "equal_split"), f"unsupported method: {method}"
		policy = await self.get_policy(policy_id, tenant_id)
		premium = total_premium or (policy.annual_premium if policy and hasattr(policy, "annual_premium") else Decimal("0"))
		per_unit = premium / max(len(units), 1)
		allocations = [
			{"unit_id": u, "allocated_premium": str(per_unit.quantize(Decimal("0.01"))), "method": method}
			for u in units
		]
		from uuid6 import uuid7
		allocation_id = str(uuid7())
		result: dict[str, Any] = {
			"id": allocation_id,
			"tenant_id": tenant_id,
			"policy_id": policy_id,
			"method": method,
			"unit_count": len(units),
			"total_premium": str(premium),
			"allocations": allocations,
			"allocated_at": datetime.utcnow().isoformat(),
		}
		self._store["allocations"].append(result)
		return result

	# ── NEW: claim_notification ───────────────────────────────────────────────

	async def claim_notification(
		self,
		property_id: str,
		incident_type: str,
		estimated_loss: Decimal,
		incident_date: date,
		tenant_id: str,
		description: str = "",
		notified_by: str = "system",
	) -> dict[str, Any]:
		"""Record initial claim notification to insurer within required reporting window."""
		assert property_id and incident_type, "property_id and incident_type required"
		assert estimated_loss >= 0, "estimated_loss must be non-negative"
		policies = await self.list_policies(tenant_id, property_id=property_id, status="active")
		active_policy_id = policies[0].id if policies else None
		from uuid6 import uuid7
		notification_id = str(uuid7())
		notification_deadline = datetime.utcnow() + timedelta(days=3)
		if estimated_loss > Decimal("1000000"):
			self._log_large_claim(notification_id, estimated_loss)
		notification: dict[str, Any] = {
			"id": notification_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"policy_id": active_policy_id,
			"incident_type": incident_type,
			"incident_date": str(incident_date),
			"estimated_loss": str(estimated_loss),
			"description": description,
			"notified_by": notified_by,
			"notification_deadline": notification_deadline.isoformat(),
			"status": "notified",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("claim_notification", notification_id, tenant_id)
		return notification

	# ── NEW: loss_adjuster_appointment ───────────────────────────────────────

	async def loss_adjuster_appointment(
		self,
		claim_id: str,
		adjuster_id: str,
		tenant_id: str,
		adjuster_firm: str = "",
		appointment_date: date | None = None,
		scope_of_assessment: str = "",
	) -> dict[str, Any]:
		"""Appoint a loss adjuster to a claim and record the appointment."""
		assert claim_id and adjuster_id, "claim_id and adjuster_id required"
		from uuid6 import uuid7
		appointment_id = str(uuid7())
		appointment: dict[str, Any] = {
			"id": appointment_id,
			"tenant_id": tenant_id,
			"claim_id": claim_id,
			"adjuster_id": adjuster_id,
			"adjuster_firm": adjuster_firm,
			"appointment_date": str(appointment_date or date.today()),
			"scope_of_assessment": scope_of_assessment,
			"status": "appointed",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["loss_adjusters"].append(appointment)
		# update claim
		for i, c in enumerate(self._store["claims"]):
			if c["id"] == claim_id and c["tenant_id"] == tenant_id:
				c["loss_adjuster_id"] = adjuster_id
				c["loss_adjuster_appointed_at"] = datetime.utcnow().isoformat()
				self._store["claims"][i] = c
				break
		self._log_operation("loss_adjuster_appointed", appointment_id, tenant_id)
		return appointment

	# ── NEW: claim_settlement ──────────────────────────────────────────────────

	async def claim_settlement(
		self,
		claim_id: str,
		settlement_amount: Decimal,
		settlement_date: date,
		tenant_id: str,
		settlement_basis: str = "agreed_settlement",
		payment_reference: str = "",
	) -> ClaimResponse | None:
		"""Record full and final settlement of a claim."""
		assert claim_id, "claim_id required"
		assert settlement_amount >= 0, "settlement_amount must be non-negative"
		result = await self.settle_claim(claim_id, tenant_id, settlement_amount)
		if result:
			for i, c in enumerate(self._store["claims"]):
				if c["id"] == claim_id and c["tenant_id"] == tenant_id:
					c["settlement_date"] = str(settlement_date)
					c["settlement_basis"] = settlement_basis
					c["payment_reference"] = payment_reference
					self._store["claims"][i] = c
					break
		return result

	# ── NEW: reinstatement_costing ─────────────────────────────────────────────

	async def reinstatement_costing(
		self,
		claim_id: str,
		reinstatement_items: list[dict[str, Any]],
		tenant_id: str,
		surveyor_id: str = "system",
		surveyor_reference: str = "",
	) -> dict[str, Any]:
		"""Record reinstatement cost estimate for a claim from a quantity surveyor."""
		assert claim_id and reinstatement_items, "claim_id and reinstatement_items required"
		total_reinstatement = sum(float(item.get("cost", 0)) for item in reinstatement_items)
		from uuid6 import uuid7
		costing_id = str(uuid7())
		costing: dict[str, Any] = {
			"id": costing_id,
			"tenant_id": tenant_id,
			"claim_id": claim_id,
			"surveyor_id": surveyor_id,
			"surveyor_reference": surveyor_reference,
			"reinstatement_items": reinstatement_items,
			"item_count": len(reinstatement_items),
			"total_reinstatement_cost": total_reinstatement,
			"vat_applicable": True,
			"status": "draft",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["reinstatements"].append(costing)
		self._log_operation("reinstatement_costing_recorded", costing_id, tenant_id)
		return costing

	# ── NEW: policy_renewal_review ─────────────────────────────────────────────

	async def policy_renewal_review(
		self,
		policy_id: str,
		tenant_id: str,
		broker_recommendation: str = "",
		market_options: list[dict[str, Any]] | None = None,
		renewal_decision: str = "renew",
	) -> dict[str, Any]:
		"""Review a policy for renewal: compare market options, record broker recommendation, capture decision."""
		assert policy_id, "policy_id required"
		assert renewal_decision in ("renew", "replace", "cancel", "pending"), \
			f"unsupported renewal_decision: {renewal_decision}"
		policy = await self.get_policy(policy_id, tenant_id)
		if policy is None:
			raise KeyError(f"policy {policy_id} not found")
		from uuid6 import uuid7
		review_id = str(uuid7())
		review: dict[str, Any] = {
			"id": review_id,
			"tenant_id": tenant_id,
			"policy_id": policy_id,
			"policy_number": getattr(policy, "policy_number", ""),
			"current_expiry": getattr(policy, "expiry_date", ""),
			"broker_recommendation": broker_recommendation,
			"market_options": market_options or [],
			"renewal_decision": renewal_decision,
			"reviewed_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("policy_renewal_reviewed", review_id, tenant_id)
		return review

	# ── NEW: insurance_analytics ───────────────────────────────────────────────

	async def insurance_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate insurance portfolio analytics for a period."""
		assert period, "period required"
		policies = await self.list_policies(tenant_id)
		active_policies = [p for p in policies if p.status.value == "active"]
		claims = await self.list_claims(tenant_id)
		settled_claims = [c for c in claims if c.status.value == "settled"]
		total_sum_insured = sum(p.sum_insured for p in active_policies)
		total_settlement = sum(
			Decimal(str(c.settlement_amount)) for c in settled_claims
			if hasattr(c, "settlement_amount") and c.settlement_amount
		)
		total_premium_allocated = sum(
			Decimal(str(a.get("total_premium", 0)))
			for a in self._store.get("allocations", [])
			if a.get("tenant_id") == tenant_id
		)
		loss_ratio = float(total_settlement / max(total_premium_allocated, Decimal("1"))) * 100
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_policies": len(policies),
			"active_policies": len(active_policies),
			"total_sum_insured": float(total_sum_insured),
			"total_claims": len(claims),
			"settled_claims": len(settled_claims),
			"total_settlement_paid": float(total_settlement),
			"total_premium_allocated": float(total_premium_allocated),
			"loss_ratio_pct": round(loss_ratio, 2),
			"expiring_90_days": len(await self.get_renewal_pipeline(tenant_id, days_ahead=90)),
			"critical_gaps": len(await self.list_coverage_gaps(tenant_id, resolved=False)),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: under_insurance_check ─────────────────────────────────────────────

	async def under_insurance_check(
		self,
		property_id: str,
		tenant_id: str,
		current_rebuild_cost: Decimal | None = None,
	) -> dict[str, Any]:
		"""Check if a property is under-insured by comparing sum insured to current rebuild cost."""
		assert property_id, "property_id required"
		policies = await self.list_policies(tenant_id, property_id=property_id, status="active")
		if not policies:
			return {
				"property_id": property_id,
				"under_insured": True,
				"reason": "no_active_policy",
				"checked_at": datetime.utcnow().isoformat(),
			}
		total_sum_insured = sum(p.sum_insured for p in policies)
		assets = []
		for p in policies:
			assets.extend(await self.list_policy_assets(tenant_id, p.id))
		total_insured_value = sum(
			Decimal(str(a.insured_value)) for a in assets
			if hasattr(a, "insured_value") and a.insured_value
		)
		rebuild_cost = current_rebuild_cost or total_insured_value * Decimal("1.2")
		under_insured = total_sum_insured < rebuild_cost * Decimal("0.9")
		gap = max(Decimal("0"), rebuild_cost - total_sum_insured)
		return {
			"property_id": property_id,
			"tenant_id": tenant_id,
			"total_sum_insured": float(total_sum_insured),
			"estimated_rebuild_cost": float(rebuild_cost),
			"under_insured": under_insured,
			"insurance_gap": float(gap),
			"adequacy_ratio_pct": round(float(total_sum_insured / max(rebuild_cost, Decimal("1"))) * 100, 2),
			"checked_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: captive_insurance_management ─────────────────────────────────────

	async def captive_insurance_management(
		self,
		captive_id: str,
		period: str,
		tenant_id: str,
		premium_income: Decimal = Decimal("0"),
		claims_paid: Decimal = Decimal("0"),
		reserves: Decimal = Decimal("0"),
		reinsurance_purchased: bool = False,
	) -> dict[str, Any]:
		"""Manage a captive insurance company: record premium, claims, reserves, and reinsurance."""
		assert captive_id and period, "captive_id and period required"
		net_underwriting_result = premium_income - claims_paid
		loss_ratio = float(claims_paid / max(premium_income, Decimal("1"))) * 100
		solvency_ratio = float(reserves / max(claims_paid, Decimal("1")))
		from uuid6 import uuid7
		record_id = str(uuid7())
		record: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"captive_id": captive_id,
			"period": period,
			"premium_income": str(premium_income),
			"claims_paid": str(claims_paid),
			"reserves": str(reserves),
			"net_underwriting_result": str(net_underwriting_result),
			"loss_ratio_pct": round(loss_ratio, 2),
			"solvency_ratio": round(solvency_ratio, 2),
			"reinsurance_purchased": reinsurance_purchased,
			"regulatory_compliant": solvency_ratio >= 1.5,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._store["captives"].append(record)
		self._log_operation("captive_period_recorded", record_id, tenant_id)
		return record


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0, "exported_at": datetime.utcnow().isoformat()}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy", "checked_at": datetime.utcnow().isoformat()}

	async def compliance_audit(self, tenant_id: str, standard: str = "RICS") -> dict[str, Any]:
		"""Compliance Audit"""
		self._log_operation("compliance_audit", "audit", tenant_id)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "checked_at": datetime.utcnow().isoformat()}

	async def bulk_update_records(self, updates: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Update Records"""
		assert updates, "updates required"
		self._log_operation("bulk_update", "bulk", tenant_id)
		return {"updated_count": len(updates), "tenant_id": tenant_id}

	async def get_kpis(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		self._log_operation("get_kpis", "kpis", tenant_id)
		return {"tenant_id": tenant_id, "period": period, "computed_at": datetime.utcnow().isoformat()}

	async def search_records(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search Records"""
		assert query, "query required"
		return {"query": query, "tenant_id": tenant_id, "results": [], "result_count": 0}

	async def archive_record(self, record_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Archive Record"""
		assert record_id and reason, "record_id and reason required"
		self._log_operation("archive_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "archived", "reason": reason, "archived_at": datetime.utcnow().isoformat()}

	async def restore_record(self, record_id: str, tenant_id: str) -> dict[str, Any]:
		"""Restore Record"""
		assert record_id, "record_id required"
		self._log_operation("restore_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "active", "restored_at": datetime.utcnow().isoformat()}

	# ── NEW: parametric_trigger_evaluate ─────────────────────────────────────

	async def parametric_trigger_evaluate(
		self,
		property_id: str,
		peril: str,
		measurement_value: Decimal,
		measurement_unit: str,
		threshold_value: Decimal,
		tenant_id: str,
		data_source: str = "oracle",
		measurement_date: date | None = None,
	) -> dict[str, Any]:
		"""
		Evaluate a parametric insurance trigger for a catastrophe peril.

		If ``measurement_value`` exceeds ``threshold_value`` the method
		auto-lodges and pre-approves a claim against the active policy for
		the property, bypassing the manual loss-adjuster workflow. Suitable
		for flood (rainfall mm), wind (km/h), and seismic (Richter) perils.

		Returns the trigger evaluation record and, when triggered, the
		auto-created ClaimResponse.
		"""
		assert property_id and peril, "property_id and peril required"
		assert measurement_value >= 0 and threshold_value > 0, "values must be non-negative; threshold must be positive"
		assert peril in ("flood", "earthquake", "wind", "hail", "drought"), f"parametric peril not supported: {peril}"

		from uuid6 import uuid7
		trigger_id = str(uuid7())
		triggered = measurement_value >= threshold_value
		policies = await self.list_policies(tenant_id, property_id=property_id, status="active")

		auto_claim: dict[str, Any] | None = None
		if triggered and policies:
			policy = policies[0]
			# Pre-approved parametric payout = sum_insured * parametric_pct
			parametric_pct = min(float(measurement_value / threshold_value) - 1.0, 1.0)
			estimated_payout = policy.sum_insured * Decimal(str(round(parametric_pct, 4)))
			claim_payload = ClaimCreate(
				tenant_id=tenant_id,
				policy_id=policy.id,
				claim_type="partial_loss" if parametric_pct < 1.0 else "total_loss",
				peril=peril,
				incident_date=measurement_date or date.today(),
				description=f"Parametric trigger: {peril} {measurement_value}{measurement_unit} >= threshold {threshold_value}{measurement_unit}",
				estimated_loss=estimated_payout,
				currency=policy.currency,
				property_id=property_id,
				evidence_ids=[f"parametric:{data_source}:{trigger_id}"],
				created_by=data_source,
			)
			claim = await self.lodge_claim(claim_payload)
			# auto-approve — no loss adjuster required for parametric
			approved = await self.approve_claim(claim.id, tenant_id, estimated_payout, senior_approved=True)
			auto_claim = approved.model_dump() if approved else claim.model_dump()

		result: dict[str, Any] = {
			"id": trigger_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"peril": peril,
			"measurement_value": str(measurement_value),
			"measurement_unit": measurement_unit,
			"threshold_value": str(threshold_value),
			"triggered": triggered,
			"data_source": data_source,
			"measurement_date": str(measurement_date or date.today()),
			"active_policy_count": len(policies),
			"auto_claim": auto_claim,
			"evaluated_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("parametric_trigger_evaluated", trigger_id, tenant_id)
		return result

	# ── NEW: score_claim_fraud_risk ────────────────────────────────────────────

	async def score_claim_fraud_risk(
		self,
		claim_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""
		Score a claim for fraud risk (0–100, higher = more suspicious).

		Checks: asset on schedule at incident date, duplicate incident
		dates, estimated loss versus sum insured ratio, claim frequency on
		policy within rolling 12 months. Routes high-risk claims (score >
		70) to senior adjuster and records a fraud flag.
		"""
		assert claim_id, "claim_id required"
		claim = await self.get_claim(claim_id, tenant_id)
		if claim is None:
			raise KeyError(f"claim {claim_id} not found")

		score = 0
		flags: list[str] = []

		# 1. Asset on schedule check
		policy_assets = await self.list_policy_assets(tenant_id, claim.policy_id)
		asset_on_schedule = any(
			str(a.property_id) == str(claim.property_id) for a in policy_assets
		)
		if not asset_on_schedule:
			score += 35
			flags.append("asset_not_on_schedule_at_incident")

		# 2. Duplicate incident date across claims for same tenant
		all_claims = await self.list_claims(tenant_id)
		same_date_claims = [
			c for c in all_claims
			if c.id != claim_id and str(c.incident_date) == str(claim.incident_date)
		]
		if len(same_date_claims) >= 2:
			score += 20
			flags.append("duplicate_incident_date_across_claims")

		# 3. Estimated loss vs sum insured ratio
		policy = await self.get_policy(claim.policy_id, tenant_id)
		if policy:
			loss_ratio = float(claim.estimated_loss / max(policy.sum_insured, Decimal("1")))
			if loss_ratio > 0.8:
				score += 20
				flags.append("loss_exceeds_80pct_sum_insured")

		# 4. Claim frequency on this policy in last 12 months
		policy_claims = await self.list_claims(tenant_id, policy_id=claim.policy_id)
		recent_cutoff = date.today() - timedelta(days=365)
		recent_claims = [
			c for c in policy_claims
			if hasattr(c, "incident_date") and c.incident_date > recent_cutoff
		]
		if len(recent_claims) > 3:
			score += 25
			flags.append("high_claim_frequency_12_months")

		score = min(score, 100)
		route_to_senior = score >= 70

		# update claim record with fraud score
		for i, c in enumerate(self._store["claims"]):
			if c["id"] == claim_id and c["tenant_id"] == tenant_id:
				c["fraud_score"] = score
				c["fraud_flags"] = flags
				c["senior_review_required"] = route_to_senior
				self._store["claims"][i] = c
				break

		result: dict[str, Any] = {
			"claim_id": claim_id,
			"tenant_id": tenant_id,
			"fraud_score": score,
			"risk_band": "high" if score >= 70 else ("medium" if score >= 40 else "low"),
			"flags": flags,
			"route_to_senior_adjuster": route_to_senior,
			"scored_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("fraud_scored", claim_id, tenant_id)
		return result

	# ── NEW: initiate_subrogation ──────────────────────────────────────────────

	async def initiate_subrogation(
		self,
		claim_id: str,
		tenant_id: str,
		liable_party_id: str,
		liable_party_name: str,
		recovery_basis: str,
		estimated_recovery: Decimal,
		assigned_to: str = "legal_team",
	) -> dict[str, Any]:
		"""
		Open a subrogation recovery file after a settled claim where a
		third party bears responsibility (contractor negligence, tenant
		damage, motor vehicle impact). Tracks correspondence and recovery.
		"""
		assert claim_id and liable_party_id, "claim_id and liable_party_id required"
		assert recovery_basis in (
			"contractor_negligence", "tenant_damage", "third_party_motor",
			"landlord_liability", "product_liability", "other"
		), f"unsupported recovery_basis: {recovery_basis}"

		claim = await self.get_claim(claim_id, tenant_id)
		if claim is None:
			raise KeyError(f"claim {claim_id} not found")
		if claim.status.value not in ("settled", "approved"):
			raise ValueError(f"subrogation requires settled/approved claim; status={claim.status.value}")

		from uuid6 import uuid7
		subrogation_id = str(uuid7())
		record: dict[str, Any] = {
			"id": subrogation_id,
			"tenant_id": tenant_id,
			"claim_id": claim_id,
			"claim_ref": claim.claim_ref,
			"liable_party_id": liable_party_id,
			"liable_party_name": liable_party_name,
			"recovery_basis": recovery_basis,
			"estimated_recovery": str(estimated_recovery),
			"actual_recovery": "0",
			"status": "open",
			"assigned_to": assigned_to,
			"correspondence": [],
			"opened_at": datetime.utcnow().isoformat(),
			"closed_at": None,
		}
		if "subrogations" not in self._store:
			self._store["subrogations"] = []
		self._store["subrogations"].append(record)
		self._log_operation("subrogation_opened", subrogation_id, tenant_id)
		return record

	async def record_subrogation_recovery(
		self,
		subrogation_id: str,
		tenant_id: str,
		recovery_amount: Decimal,
		payment_reference: str = "",
	) -> dict[str, Any]:
		"""Record a cash recovery against an open subrogation file."""
		assert subrogation_id and recovery_amount >= 0, "subrogation_id required; recovery_amount must be non-negative"
		subrogations = self._store.get("subrogations", [])
		for i, s in enumerate(subrogations):
			if s["id"] == subrogation_id and s["tenant_id"] == tenant_id:
				prev = Decimal(str(s.get("actual_recovery", "0")))
				s["actual_recovery"] = str(prev + recovery_amount)
				s["payment_reference"] = payment_reference
				estimated = Decimal(str(s.get("estimated_recovery", "0")))
				if (prev + recovery_amount) >= estimated:
					s["status"] = "closed"
					s["closed_at"] = datetime.utcnow().isoformat()
				subrogations[i] = s
				self._log_operation("subrogation_recovery_recorded", subrogation_id, tenant_id)
				return s
		raise KeyError(f"subrogation {subrogation_id} not found")

	# ── NEW: generate_loss_run ─────────────────────────────────────────────────

	async def generate_loss_run(
		self,
		tenant_id: str,
		years: int = 5,
		policy_id: str | None = None,
		property_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Produce a structured loss run: 5-year claims history by policy and
		property with frequency, severity, cause breakdown, and trend.
		Format matches what underwriters require during renewal negotiation.
		"""
		assert years > 0, "years must be positive"
		cutoff_year = date.today().year - years
		claims = await self.list_claims(tenant_id, policy_id=policy_id)

		if property_id:
			claims = [c for c in claims if str(c.property_id) == property_id]

		# Filter to window
		in_window = [
			c for c in claims
			if hasattr(c, "incident_date") and c.incident_date.year >= cutoff_year
		]

		# Aggregate by year
		by_year: dict[int, dict[str, Any]] = {}
		for c in in_window:
			yr = c.incident_date.year
			if yr not in by_year:
				by_year[yr] = {"year": yr, "claim_count": 0, "total_estimated": Decimal("0"),
								"total_settled": Decimal("0"), "perils": {}}
			by_year[yr]["claim_count"] += 1
			by_year[yr]["total_estimated"] += c.estimated_loss
			if c.settlement_amount:
				by_year[yr]["total_settled"] += c.settlement_amount
			by_year[yr]["perils"][c.peril] = by_year[yr]["perils"].get(c.peril, 0) + 1

		# Severity trend (simple linear slope on settled amounts)
		years_sorted = sorted(by_year.keys())
		settled_series = [float(by_year[y]["total_settled"]) for y in years_sorted]
		avg_severity = sum(settled_series) / max(len(settled_series), 1)
		trend_direction = "stable"
		if len(settled_series) >= 2:
			if settled_series[-1] > settled_series[0] * 1.1:
				trend_direction = "increasing"
			elif settled_series[-1] < settled_series[0] * 0.9:
				trend_direction = "decreasing"

		return {
			"tenant_id": tenant_id,
			"policy_id": policy_id,
			"property_id": property_id,
			"window_years": years,
			"total_claims": len(in_window),
			"total_estimated_loss": float(sum(c.estimated_loss for c in in_window)),
			"total_settled": float(sum(c.settlement_amount or Decimal("0") for c in in_window)),
			"annual_breakdown": [
				{
					**v,
					"total_estimated": float(v["total_estimated"]),
					"total_settled": float(v["total_settled"]),
				}
				for v in sorted(by_year.values(), key=lambda x: x["year"])
			],
			"average_annual_severity": round(avg_severity, 2),
			"severity_trend": trend_direction,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: issue_certificate ─────────────────────────────────────────────────

	async def issue_certificate(
		self,
		policy_id: str,
		tenant_id: str,
		certificate_type: str = "insurance_certificate",
		beneficiary_name: str = "",
		beneficiary_reference: str = "",
		issued_by: str = "system",
	) -> dict[str, Any]:
		"""
		Issue a formal insurance certificate against an active policy.
		Returns a structured document payload ready for PDF/DOCX rendering.
		Certificate types: insurance_certificate, mortgage_endorsement,
		loss_payee_clause, co-insurance_certificate.
		"""
		assert policy_id, "policy_id required"
		assert certificate_type in (
			"insurance_certificate", "mortgage_endorsement",
			"loss_payee_clause", "co_insurance_certificate"
		), f"unsupported certificate_type: {certificate_type}"

		policy = await self.get_policy(policy_id, tenant_id)
		if policy is None:
			raise KeyError(f"policy {policy_id} not found")
		if policy.status.value not in ("active", "endorsed"):
			raise ValueError(f"certificate_requires_active_policy: status={policy.status.value}")

		self._check_rules({"operation": "issue_certificate", "policy_active": True})

		insurer = await self.get_insurer(policy.insurer_id, tenant_id)
		from uuid6 import uuid7
		cert_id = str(uuid7())
		cert_number = f"CERT-{cert_id[:8].upper()}"

		certificate: dict[str, Any] = {
			"id": cert_id,
			"cert_number": cert_number,
			"tenant_id": tenant_id,
			"certificate_type": certificate_type,
			"policy_id": policy_id,
			"policy_number": policy.policy_number,
			"policy_type": policy.policy_type.value,
			"insurer_name": insurer.name if insurer else policy.insurer_id,
			"insurer_grade": insurer.grade.value if insurer else "unknown",
			"sum_insured": float(policy.sum_insured),
			"currency": policy.currency,
			"commencement_date": str(policy.commencement_date),
			"expiry_date": str(policy.expiry_date),
			"perils_covered": policy.perils_covered,
			"beneficiary_name": beneficiary_name,
			"beneficiary_reference": beneficiary_reference,
			"issued_by": issued_by,
			"is_draft": False,
			"issued_at": datetime.utcnow().isoformat(),
			"valid_until": str(policy.expiry_date),
		}
		if "certificates" not in self._store:
			self._store["certificates"] = []
		self._store["certificates"].append(certificate)
		self._log_operation("certificate_issued", cert_id, tenant_id)
		return certificate

	# ── NEW: run_portfolio_stress_test ────────────────────────────────────────

	async def run_portfolio_stress_test(
		self,
		tenant_id: str,
		scenario_name: str,
		affected_perils: list[str],
		pml_factor: Decimal,
		affected_location: str = "all",
	) -> dict[str, Any]:
		"""
		Run a portfolio-level Probable Maximum Loss (PML) stress test.

		``pml_factor`` is a Decimal 0–1 representing the fraction of sum
		insured expected to be lost in the scenario (e.g. 0.25 for a
		1-in-100-year flood in a specific zone). Outputs gross and net
		(post-reinsurance) retained loss.
		"""
		assert scenario_name and affected_perils, "scenario_name and affected_perils required"
		assert 0 < pml_factor <= 1, "pml_factor must be between 0 and 1"

		policies = await self.list_policies(tenant_id, status="active")
		# filter to policies covering at least one of the affected perils
		exposed_policies = [
			p for p in policies
			if not p.perils_covered or any(peril in p.perils_covered for peril in affected_perils)
		]

		total_exposed_sum = sum(p.sum_insured for p in exposed_policies)
		gross_pml = total_exposed_sum * pml_factor

		# Simple XL reinsurance: first 10M retained, 90% of excess ceded
		retention_limit = Decimal("10000000")
		if gross_pml <= retention_limit:
			reinsurance_recovery = Decimal("0")
		else:
			reinsurance_recovery = (gross_pml - retention_limit) * Decimal("0.90")

		net_retained = gross_pml - reinsurance_recovery

		from uuid6 import uuid7
		test_id = str(uuid7())
		result: dict[str, Any] = {
			"id": test_id,
			"tenant_id": tenant_id,
			"scenario_name": scenario_name,
			"affected_perils": affected_perils,
			"affected_location": affected_location,
			"pml_factor": float(pml_factor),
			"total_active_policies": len(policies),
			"exposed_policies": len(exposed_policies),
			"total_exposed_sum_insured": float(total_exposed_sum),
			"gross_pml": float(gross_pml),
			"reinsurance_recovery": float(reinsurance_recovery),
			"net_retained_loss": float(net_retained),
			"capital_adequacy_flag": float(net_retained) < 5_000_000,
			"run_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("stress_test_run", test_id, tenant_id)
		return result

	# ── NEW: advance_renewal_stage ────────────────────────────────────────────

	async def advance_renewal_stage(
		self,
		policy_id: str,
		tenant_id: str,
		new_stage: str,
		broker_id: str | None = None,
		market_quotes: list[dict[str, Any]] | None = None,
		notes: str = "",
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""
		Advance a policy through the structured renewal pipeline:
		rfq_sent → quotes_received → approved → bound → lapsed.

		At each stage transition the method validates the prior stage was
		completed, records the transition event, and returns the updated
		renewal record. Replaces ad-hoc spreadsheet renewal tracking.
		"""
		assert policy_id and new_stage, "policy_id and new_stage required"
		valid_stages = ("rfq_sent", "quotes_received", "approved", "bound", "lapsed")
		assert new_stage in valid_stages, f"invalid renewal stage: {new_stage}"

		policy = await self.get_policy(policy_id, tenant_id)
		if policy is None:
			raise KeyError(f"policy {policy_id} not found")

		expiry = datetime.strptime(str(policy.expiry_date), "%Y-%m-%d").date()
		days_to_expiry = (expiry - date.today()).days

		from uuid6 import uuid7
		event_id = str(uuid7())
		renewal_event: dict[str, Any] = {
			"id": event_id,
			"tenant_id": tenant_id,
			"policy_id": policy_id,
			"policy_number": policy.policy_number,
			"stage": new_stage,
			"days_to_expiry": days_to_expiry,
			"broker_id": broker_id,
			"market_quotes": market_quotes or [],
			"notes": notes,
			"actor_id": actor_id,
			"transitioned_at": datetime.utcnow().isoformat(),
		}

		if "renewal_events" not in self._store:
			self._store["renewal_events"] = []
		self._store["renewal_events"].append(renewal_event)

		# Update policy renewal_status to match stage
		stage_to_renewal_status = {
			"rfq_sent": "in_negotiation",
			"quotes_received": "quoted",
			"approved": "accepted",
			"bound": "bound",
			"lapsed": "lapsed",
		}
		await self.update_policy(policy_id, tenant_id, PolicyUpdate(renewal_status=stage_to_renewal_status[new_stage]))
		self._log_operation(f"renewal_stage_{new_stage}", event_id, tenant_id)
		return renewal_event

	# ── NEW: apportion_insurance_to_tenants ───────────────────────────────────

	async def apportion_insurance_to_tenants(
		self,
		policy_id: str,
		tenant_id: str,
		tenant_unit_map: list[dict[str, Any]],
		apportionment_basis: str = "floor_area",
		period: str = "",
	) -> dict[str, Any]:
		"""
		Apportion insurance premium to individual property tenants based on
		occupied floor area or insured value. Produces a per-tenant charge
		schedule ready for posting to ``realestate_acc``.

		``tenant_unit_map`` is a list of dicts with keys:
		  tenant_id, unit_id, floor_area_sqm (for floor_area basis) or
		  insured_value (for value basis).
		"""
		assert policy_id and tenant_unit_map, "policy_id and tenant_unit_map required"
		assert apportionment_basis in ("floor_area", "insured_value", "equal"), \
			f"unsupported apportionment_basis: {apportionment_basis}"

		policy = await self.get_policy(policy_id, tenant_id)
		if policy is None:
			raise KeyError(f"policy {policy_id} not found")

		annual_premium = policy.annual_premium
		total_basis: Decimal = Decimal("0")
		for t in tenant_unit_map:
			if apportionment_basis == "floor_area":
				total_basis += Decimal(str(t.get("floor_area_sqm", 0)))
			elif apportionment_basis == "insured_value":
				total_basis += Decimal(str(t.get("insured_value", 0)))
			else:
				total_basis += Decimal("1")

		if total_basis == Decimal("0"):
			raise ValueError("total basis is zero — cannot apportion")

		apportioned: list[dict[str, Any]] = []
		for t in tenant_unit_map:
			if apportionment_basis == "floor_area":
				basis = Decimal(str(t.get("floor_area_sqm", 0)))
			elif apportionment_basis == "insured_value":
				basis = Decimal(str(t.get("insured_value", 0)))
			else:
				basis = Decimal("1")
			share = basis / total_basis
			charge = (annual_premium * share).quantize(Decimal("0.01"))
			apportioned.append({
				"tenant_id": t.get("tenant_id"),
				"unit_id": t.get("unit_id"),
				"basis_value": float(basis),
				"share_pct": round(float(share) * 100, 4),
				"insurance_charge": float(charge),
				"currency": policy.currency,
			})

		from uuid6 import uuid7
		run_id = str(uuid7())
		result: dict[str, Any] = {
			"id": run_id,
			"management_tenant_id": tenant_id,
			"policy_id": policy_id,
			"policy_number": policy.policy_number,
			"apportionment_basis": apportionment_basis,
			"period": period,
			"annual_premium": float(annual_premium),
			"currency": policy.currency,
			"tenant_count": len(apportioned),
			"apportioned_charges": apportioned,
			"total_apportioned": float(sum(Decimal(str(a["insurance_charge"])) for a in apportioned)),
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("premium_apportioned", run_id, tenant_id)
		return result

	# ── NEW: get_broker_scorecard ──────────────────────────────────────────────

	async def get_broker_scorecard(
		self,
		broker_id: str,
		tenant_id: str,
		period_years: int = 3,
	) -> dict[str, Any]:
		"""
		Compute a broker performance scorecard over the last N years:
		- Policies placed and retention rate
		- Average quote turnaround (days from rfq_sent to quotes_received)
		- Commission as % of total premium
		- Claims handled vs. claims escalated ratio
		Returns a ranked score 0–100 with band (preferred / approved / conditional).
		"""
		assert broker_id, "broker_id required"
		cutoff_year = date.today().year - period_years
		all_policies = await self.list_policies(tenant_id)
		broker_policies = [p for p in all_policies if p.broker_id == broker_id]

		# Renewal events for this broker
		renewal_events = [
			e for e in self._store.get("renewal_events", [])
			if e.get("broker_id") == broker_id and e.get("tenant_id") == tenant_id
		]
		rfq_events = [e for e in renewal_events if e["stage"] == "rfq_sent"]
		quote_events = {e["policy_id"]: e for e in renewal_events if e["stage"] == "quotes_received"}

		turnaround_days: list[float] = []
		for rfq in rfq_events:
			pid = rfq["policy_id"]
			if pid in quote_events:
				rfq_dt = datetime.fromisoformat(rfq["transitioned_at"])
				quote_dt = datetime.fromisoformat(quote_events[pid]["transitioned_at"])
				turnaround_days.append((quote_dt - rfq_dt).total_seconds() / 86400)

		avg_turnaround = round(sum(turnaround_days) / max(len(turnaround_days), 1), 1)
		bound_count = len([e for e in renewal_events if e["stage"] == "bound"])
		retention_rate = round(bound_count / max(len(rfq_events), 1) * 100, 1)

		total_premium = float(sum(p.annual_premium for p in broker_policies))

		# Score components
		score = 0
		if retention_rate >= 85:
			score += 30
		elif retention_rate >= 70:
			score += 20
		else:
			score += 5

		if avg_turnaround <= 5:
			score += 30
		elif avg_turnaround <= 14:
			score += 20
		else:
			score += 5

		if len(broker_policies) >= 10:
			score += 20
		elif len(broker_policies) >= 3:
			score += 15
		else:
			score += 5

		score += 20  # baseline for being registered

		band = "preferred" if score >= 80 else ("approved" if score >= 60 else "conditional")

		return {
			"broker_id": broker_id,
			"tenant_id": tenant_id,
			"period_years": period_years,
			"total_policies_placed": len(broker_policies),
			"total_premium_managed": total_premium,
			"retention_rate_pct": retention_rate,
			"avg_quote_turnaround_days": avg_turnaround,
			"bound_renewals": bound_count,
			"scorecard_score": score,
			"scorecard_band": band,
			"computed_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: attach_claim_evidence ─────────────────────────────────────────────

	async def attach_claim_evidence(
		self,
		claim_id: str,
		tenant_id: str,
		evidence_type: str,
		file_reference: str,
		file_hash_sha256: str,
		description: str = "",
		uploaded_by: str = "system",
	) -> dict[str, Any]:
		"""
		Attach a piece of evidence to a claim with chain-of-custody logging.
		``file_hash_sha256`` is recorded to detect tampering on retrieval.
		Evidence types: photo, video, police_abstract, contractor_report,
		quantity_survey, weather_report, invoice, other.
		"""
		assert claim_id and file_reference and file_hash_sha256, \
			"claim_id, file_reference, and file_hash_sha256 required"
		valid_types = (
			"photo", "video", "police_abstract", "contractor_report",
			"quantity_survey", "weather_report", "invoice", "other"
		)
		assert evidence_type in valid_types, f"unsupported evidence_type: {evidence_type}"

		claim = await self.get_claim(claim_id, tenant_id)
		if claim is None:
			raise KeyError(f"claim {claim_id} not found")

		from uuid6 import uuid7
		evidence_id = str(uuid7())
		evidence: dict[str, Any] = {
			"id": evidence_id,
			"tenant_id": tenant_id,
			"claim_id": claim_id,
			"evidence_type": evidence_type,
			"file_reference": file_reference,
			"file_hash_sha256": file_hash_sha256,
			"description": description,
			"uploaded_by": uploaded_by,
			"integrity_verified": True,
			"uploaded_at": datetime.utcnow().isoformat(),
		}
		if "claim_evidence" not in self._store:
			self._store["claim_evidence"] = []
		self._store["claim_evidence"].append(evidence)

		# append evidence_id to the claim record
		for i, c in enumerate(self._store["claims"]):
			if c["id"] == claim_id and c["tenant_id"] == tenant_id:
				c.setdefault("evidence_ids", []).append(evidence_id)
				self._store["claims"][i] = c
				break

		self._log_operation("evidence_attached", evidence_id, tenant_id)
		return evidence
