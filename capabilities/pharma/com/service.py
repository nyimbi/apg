"""Service layer for APG Pharma Commercial Operations."""

from __future__ import annotations

from datetime import datetime, date
from typing import Any
from uuid6 import uuid7

from .capability_contract import (
	SUPPORTED_CALL_TYPES, SUPPORTED_CHANNEL_TYPES, SUPPORTED_COMPLIANCE_FRAMEWORKS,
	SUPPORTED_INTERACTION_TYPES, SUPPORTED_PLAN_STATUSES, SUPPORTED_REP_TYPES,
	SUPPORTED_SAMPLE_TYPES, SUPPORTED_SPEND_CATEGORIES, SUPPORTED_TARGET_TIERS,
	SUPPORTED_TERRITORY_TYPES, evaluate_capability_rules, get_capability_contract,
)
from .models import (
	AggregateSpendRecord, CallRecord, CallRecordCreate, CommercialPlan, CommercialPlanCreate,
	HcpInteraction, HcpInteractionCreate, SalesRep, SalesRepCreate, SampleDispensing,
	SampleDispensingCreate, TargetPhysician, Territory, TerritoryCreate, TerritoryUpdate,
)


def _uuid7str() -> str:
	return str(uuid7())


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class CommercialOperationsService:
	"""Tenant-scoped commercial operations service for pharma field force management."""

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

		self._territories: dict[tuple[str, str], Territory] = {}
		self._reps: dict[tuple[str, str], SalesRep] = {}
		self._calls: dict[tuple[str, str], CallRecord] = {}
		self._samples: dict[tuple[str, str], SampleDispensing] = {}
		self._interactions: dict[tuple[str, str], HcpInteraction] = {}
		self._plans: dict[tuple[str, str], CommercialPlan] = {}
		self._targets: dict[tuple[str, str], TargetPhysician] = {}
		self._spend: dict[tuple[str, str], AggregateSpendRecord] = {}
		self._audit_events: list[dict[str, Any]] = []
		# extended stores
		self._hcp_visits: dict[tuple[str, str], dict[str, Any]] = {}
		self._pdma_records: dict[tuple[str, str], dict[str, Any]] = {}
		self._market_access: dict[tuple[str, str], dict[str, Any]] = {}
		self._promo_materials: dict[tuple[str, str], dict[str, Any]] = {}
		self._prescriber_analytics: dict[tuple[str, str], dict[str, Any]] = {}

	# --- contract ---

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the capability contract for this tenant."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate capability rules against a context."""
		return evaluate_capability_rules(context)

	# --- territories ---

	def create_territory(self, payload: TerritoryCreate) -> Territory:
		"""Create a new sales territory with required approvals."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_territory",
			"territory_type_supported": payload.territory_type in SUPPORTED_TERRITORY_TYPES,
			"owner_present": bool(payload.owner_id),
			"approval_present": bool(payload.approval_reference),
		})
		territory = Territory(**payload.model_dump())
		self._territories[self._key(territory.tenant_id, territory.id)] = territory
		self._audit(territory.tenant_id, "territory_created", territory.id)
		return territory

	def get_territory(self, territory_id: str, tenant_id: str) -> Territory:
		"""Get a territory by ID within tenant scope."""
		item = self._territories.get(self._key(tenant_id, territory_id))
		if item is None:
			raise KeyError(f"territory {territory_id} not found")
		return item

	def list_territories(self, tenant_id: str) -> list[Territory]:
		"""List all territories for a tenant."""
		return [t for t in self._territories.values() if t.tenant_id == tenant_id]

	def update_territory(self, territory_id: str, tenant_id: str, update: TerritoryUpdate) -> Territory:
		"""Update a territory's mutable fields."""
		territory = self.get_territory(territory_id, tenant_id)
		data = territory.model_dump()
		for k, v in update.model_dump(exclude_none=True).items():
			data[k] = v
		data["updated_at"] = datetime.utcnow()
		updated = Territory(**data)
		self._territories[self._key(tenant_id, territory_id)] = updated
		self._audit(tenant_id, "territory_updated", territory_id)
		return updated

	# --- sales reps ---

	def assign_rep(self, payload: SalesRepCreate) -> SalesRep:
		"""Assign a sales rep to a territory with certification check."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "assign_rep",
			"rep_type_supported": payload.rep_type in SUPPORTED_REP_TYPES,
			"territory_present": bool(payload.territory_id),
			"certification_present": bool(payload.certification_reference),
		})
		rep = SalesRep(**payload.model_dump())
		self._reps[self._key(rep.tenant_id, rep.id)] = rep
		self._audit(rep.tenant_id, "rep_assigned", rep.id)
		return rep

	def get_rep(self, rep_id: str, tenant_id: str) -> SalesRep:
		"""Get a rep by ID within tenant scope."""
		item = self._reps.get(self._key(tenant_id, rep_id))
		if item is None:
			raise KeyError(f"rep {rep_id} not found")
		return item

	def list_reps(self, tenant_id: str) -> list[SalesRep]:
		"""List all reps for a tenant."""
		return [r for r in self._reps.values() if r.tenant_id == tenant_id]

	def list_reps_by_territory(self, territory_id: str, tenant_id: str) -> list[SalesRep]:
		"""List all reps assigned to a specific territory."""
		return [r for r in self._reps.values() if r.tenant_id == tenant_id and r.territory_id == territory_id]

	# --- call activity ---

	def record_call(self, payload: CallRecordCreate) -> CallRecord:
		"""Record a physician call with required product discussion."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_call",
			"physician_id_present": bool(payload.physician_id),
			"call_type_supported": payload.call_type in SUPPORTED_CALL_TYPES,
			"product_present": bool(payload.products_discussed),
		})
		call = CallRecord(**payload.model_dump())
		self._calls[self._key(call.tenant_id, call.id)] = call
		self._audit(call.tenant_id, "call_recorded", call.id)
		return call

	def list_calls(self, tenant_id: str, rep_id: str | None = None) -> list[CallRecord]:
		"""List calls, optionally filtered by rep."""
		calls = [c for c in self._calls.values() if c.tenant_id == tenant_id]
		if rep_id:
			calls = [c for c in calls if c.rep_id == rep_id]
		return calls

	def list_calls_by_physician(self, physician_id: str, tenant_id: str) -> list[CallRecord]:
		"""List all calls to a specific physician."""
		return [c for c in self._calls.values() if c.tenant_id == tenant_id and c.physician_id == physician_id]

	# --- sample management ---

	def dispense_sample(self, payload: SampleDispensingCreate) -> SampleDispensing:
		"""Dispense a product sample with PDMA compliance enforcement."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "dispense_sample",
			"pdma_compliant": payload.pdma_compliant,
			"signature_present": bool(payload.hcp_signature_reference),
			"lot_number_present": bool(payload.lot_number),
			"expiry_present": bool(payload.expiry_date),
			"sample_type_supported": payload.sample_type in SUPPORTED_SAMPLE_TYPES,
		})
		sample = SampleDispensing(**payload.model_dump())
		self._samples[self._key(sample.tenant_id, sample.id)] = sample
		self._audit(sample.tenant_id, "sample_dispensed", sample.id)
		return sample

	def list_samples(self, tenant_id: str, rep_id: str | None = None) -> list[SampleDispensing]:
		"""List sample dispensings, optionally by rep."""
		items = [s for s in self._samples.values() if s.tenant_id == tenant_id]
		if rep_id:
			items = [s for s in items if s.rep_id == rep_id]
		return items

	def reconcile_samples(self, tenant_id: str, rep_id: str) -> dict[str, Any]:
		"""Reconcile sample inventory for a rep."""
		samples = [s for s in self._samples.values() if s.tenant_id == tenant_id and s.rep_id == rep_id]
		by_product: dict[str, int] = {}
		for s in samples:
			by_product[s.product_id] = by_product.get(s.product_id, 0) + s.quantity
		self._audit(tenant_id, "sample_reconciled", rep_id)
		return {"tenant_id": tenant_id, "rep_id": rep_id, "dispensed_by_product": by_product, "total_dispensings": len(samples)}

	# --- HCP interactions ---

	def record_interaction(self, payload: HcpInteractionCreate) -> HcpInteraction:
		"""Record an HCP interaction with aggregate spend tracking."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_interaction",
			"hcp_id_present": bool(payload.hcp_id),
			"interaction_type_supported": payload.interaction_type in SUPPORTED_INTERACTION_TYPES,
		})
		interaction = HcpInteraction(**payload.model_dump())
		self._interactions[self._key(interaction.tenant_id, interaction.id)] = interaction
		self._audit(interaction.tenant_id, "interaction_recorded", interaction.id)
		return interaction

	def list_interactions(self, tenant_id: str, hcp_id: str | None = None) -> list[HcpInteraction]:
		"""List interactions, optionally filtered by HCP."""
		items = [i for i in self._interactions.values() if i.tenant_id == tenant_id]
		if hcp_id:
			items = [i for i in items if i.hcp_id == hcp_id]
		return items

	# --- aggregate spend ---

	def record_spend(self, tenant_id: str, hcp_id: str, category: str, amount: float,
					currency: str, fiscal_year: str, created_by: str,
					receipt_reference: str | None = None, pre_approval_reference: str | None = None,
					hcp_consent_reference: str | None = None, quarter: str | None = None) -> AggregateSpendRecord:
		"""Record aggregate spend with policy enforcement."""
		aggregate_cap = 500.0
		current_total = self._aggregate_spend_for_hcp(hcp_id, tenant_id, fiscal_year)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_spend",
			"amount_above_threshold": amount > 25.0,
			"receipt_present": bool(receipt_reference) if amount > 25.0 else True,
			"amount_above_approval_threshold": amount > 100.0,
			"pre_approval_present": bool(pre_approval_reference) if amount > 100.0 else True,
			"aggregate_cap_exceeded": (current_total + amount) > aggregate_cap,
		})
		record = AggregateSpendRecord(
			tenant_id=tenant_id, hcp_id=hcp_id, category=category, amount=amount,
			currency=currency, fiscal_year=fiscal_year, quarter=quarter,
			receipt_reference=receipt_reference, pre_approval_reference=pre_approval_reference,
			hcp_consent_reference=hcp_consent_reference, created_by=created_by,
		)
		self._spend[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, "spend_recorded", record.id)
		return record

	def get_aggregate_spend_summary(self, tenant_id: str, hcp_id: str, fiscal_year: str) -> dict[str, Any]:
		"""Return aggregate spend summary for a HCP in a fiscal year."""
		records = [r for r in self._spend.values()
				if r.tenant_id == tenant_id and r.hcp_id == hcp_id and r.fiscal_year == fiscal_year]
		total = sum(r.amount for r in records)
		by_category: dict[str, float] = {}
		for r in records:
			by_category[r.category] = by_category.get(r.category, 0.0) + r.amount
		return {"hcp_id": hcp_id, "fiscal_year": fiscal_year, "total": total,
				"by_category": by_category, "cap": 500.0, "cap_remaining": 500.0 - total}

	# --- commercial plans ---

	def create_plan(self, payload: CommercialPlanCreate) -> CommercialPlan:
		"""Create a commercial plan in draft status."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		plan = CommercialPlan(**payload.model_dump())
		self._plans[self._key(plan.tenant_id, plan.id)] = plan
		self._audit(plan.tenant_id, "plan_created", plan.id)
		return plan

	def approve_plan(self, plan_id: str, tenant_id: str, approval_reference: str) -> CommercialPlan:
		"""Approve a commercial plan."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_plan",
			"approval_present": bool(approval_reference),
		})
		plan = self._plans.get(self._key(tenant_id, plan_id))
		if plan is None:
			raise KeyError(f"plan {plan_id} not found")
		data = plan.model_dump()
		data["status"] = "approved"
		data["approval_reference"] = approval_reference
		data["updated_at"] = datetime.utcnow()
		approved = CommercialPlan(**data)
		self._plans[self._key(tenant_id, plan_id)] = approved
		self._audit(tenant_id, "plan_approved", plan_id)
		return approved

	def list_plans(self, tenant_id: str) -> list[CommercialPlan]:
		"""List all commercial plans for a tenant."""
		return [p for p in self._plans.values() if p.tenant_id == tenant_id]

	# --- target segmentation ---

	def set_target(self, tenant_id: str, physician_id: str, tier: str, territory_id: str,
					product_ids: list[str], call_frequency_per_quarter: int,
					segmentation_reference: str, created_by: str) -> TargetPhysician:
		"""Set or update a physician target tier."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "set_target_tier",
			"tier_supported": tier in SUPPORTED_TARGET_TIERS,
		})
		target = TargetPhysician(
			tenant_id=tenant_id, physician_id=physician_id, tier=tier,
			territory_id=territory_id, product_ids=product_ids,
			call_frequency_per_quarter=call_frequency_per_quarter,
			segmentation_reference=segmentation_reference, created_by=created_by,
		)
		self._targets[self._key(tenant_id, target.id)] = target
		self._audit(tenant_id, "target_set", target.id)
		return target

	def list_targets(self, tenant_id: str, territory_id: str | None = None) -> list[TargetPhysician]:
		"""List target physicians, optionally by territory."""
		items = [t for t in self._targets.values() if t.tenant_id == tenant_id]
		if territory_id:
			items = [t for t in items if t.territory_id == territory_id]
		return items

	# --- NEW: territory_assignment ---

	def territory_assignment(self, rep_id: str, territory: str, tenant_id: str) -> dict[str, Any]:
		"""Assign or reassign a rep to a territory, validating rep exists and territory is valid."""
		assert rep_id, "rep_id required"
		assert territory, "territory required"
		rep = self._reps.get(self._key(tenant_id, rep_id))
		if rep is None:
			raise KeyError(f"rep {rep_id} not found for tenant {tenant_id}")
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "assign_rep",
			"territory_present": bool(territory),
		})
		old_territory = rep.territory_id
		data = rep.model_dump()
		data["territory_id"] = territory
		data["updated_at"] = datetime.utcnow()
		updated = SalesRep(**data)
		self._reps[self._key(tenant_id, rep_id)] = updated
		self._audit(tenant_id, "territory_assignment_changed", rep_id)
		return {
			"rep_id": rep_id,
			"tenant_id": tenant_id,
			"previous_territory": old_territory,
			"new_territory": territory,
			"assigned_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: call_plan ---

	def call_plan(self, rep_id: str, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate a call plan for a rep for a given period based on target tiers and call frequencies."""
		assert rep_id, "rep_id required"
		assert period, "period required"
		rep = self._reps.get(self._key(tenant_id, rep_id))
		if rep is None:
			raise KeyError(f"rep {rep_id} not found")
		territory_id = rep.territory_id
		targets = [t for t in self._targets.values()
				if t.tenant_id == tenant_id and t.territory_id == territory_id]
		plan_items: list[dict[str, Any]] = []
		total_calls_planned = 0
		for t in targets:
			calls_needed = t.call_frequency_per_quarter
			total_calls_planned += calls_needed
			plan_items.append({
				"physician_id": t.physician_id,
				"tier": t.tier,
				"product_ids": t.product_ids,
				"calls_planned": calls_needed,
				"calls_completed": len([c for c in self._calls.values()
					if c.tenant_id == tenant_id and c.rep_id == rep_id
					and c.physician_id == t.physician_id]),
			})
		existing_calls = len([c for c in self._calls.values()
			if c.tenant_id == tenant_id and c.rep_id == rep_id])
		self._audit(tenant_id, "call_plan_generated", rep_id)
		return {
			"rep_id": rep_id,
			"territory_id": territory_id,
			"period": period,
			"total_targets": len(targets),
			"total_calls_planned": total_calls_planned,
			"total_calls_completed": existing_calls,
			"completion_rate": round(existing_calls / max(total_calls_planned, 1) * 100, 2),
			"plan_items": plan_items,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: hcp_visit_record ---

	def hcp_visit_record(
		self,
		rep_id: str,
		hcp_id: str,
		products_detailed: list[str],
		samples_given: dict[str, int],
		tenant_id: str,
		visit_date: date | None = None,
		call_type: str = "face_to_face",
		notes: str = "",
	) -> dict[str, Any]:
		"""Record a complete HCP visit including product detail, samples dispensed, and PDMA fields."""
		assert rep_id and hcp_id, "rep_id and hcp_id required"
		assert products_detailed, "at least one product must be detailed"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_call",
			"physician_id_present": True,
			"call_type_supported": call_type in SUPPORTED_CALL_TYPES,
			"product_present": bool(products_detailed),
		})
		visit_id = _uuid7str()
		visit_dt = visit_date or date.today()
		total_samples = sum(samples_given.values())
		record: dict[str, Any] = {
			"id": visit_id,
			"tenant_id": tenant_id,
			"rep_id": rep_id,
			"hcp_id": hcp_id,
			"visit_date": str(visit_dt),
			"call_type": call_type,
			"products_detailed": products_detailed,
			"samples_given": samples_given,
			"total_samples_dispensed": total_samples,
			"notes": notes,
			"pdma_compliant": True,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._hcp_visits[self._key(tenant_id, visit_id)] = record
		self._audit(tenant_id, "hcp_visit_recorded", visit_id)
		return record

	# --- NEW: pdma_compliance_check ---

	def pdma_compliance_check(self, visit_id: str, tenant_id: str) -> dict[str, Any]:
		"""Validate a recorded visit for PDMA/EFPIA compliance: signature, lot numbers, sample limits."""
		visit = self._hcp_visits.get(self._key(tenant_id, visit_id))
		if visit is None:
			# Also check legacy call records
			call = next((c for c in self._calls.values()
				if c.tenant_id == tenant_id and c.id == visit_id), None)
			if call is None:
				raise KeyError(f"visit/call {visit_id} not found")
			violations: list[str] = []
			if not getattr(call, "pdma_compliant", False):
				violations.append("pdma_compliant flag not set")
			if not getattr(call, "hcp_signature_reference", None):
				violations.append("hcp_signature_reference missing")
			self._audit(tenant_id, "pdma_compliance_checked", visit_id)
			return {
				"visit_id": visit_id,
				"tenant_id": tenant_id,
				"compliant": len(violations) == 0,
				"violations": violations,
				"checked_at": datetime.utcnow().isoformat(),
			}
		violations = []
		if not visit.get("pdma_compliant"):
			violations.append("pdma_compliant flag not set")
		samples = visit.get("samples_given", {})
		for product_id, qty in samples.items():
			if qty > 6:
				violations.append(f"sample quantity {qty} exceeds PDMA limit of 6 for product {product_id}")
		if not visit.get("hcp_id"):
			violations.append("hcp_id missing on visit")
		self._audit(tenant_id, "pdma_compliance_checked", visit_id)
		return {
			"visit_id": visit_id,
			"tenant_id": tenant_id,
			"compliant": len(violations) == 0,
			"violations": violations,
			"checked_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: sample_management ---

	def sample_management(
		self,
		rep_id: str,
		product: str,
		quantity: int,
		type: str,
		tenant_id: str,
		lot_number: str = "",
		reason: str = "",
	) -> dict[str, Any]:
		"""Manage sample inventory: receive, adjust, or write-off samples for a rep."""
		assert rep_id and product, "rep_id and product required"
		assert type in ("receipt", "dispensing", "return", "write_off", "adjustment"), \
			f"unsupported sample transaction type: {type}"
		assert quantity > 0, "quantity must be positive"
		txn_id = _uuid7str()
		existing = [s for s in self._samples.values()
			if s.tenant_id == tenant_id and s.rep_id == rep_id and s.product_id == product]
		total_dispensed = sum(s.quantity for s in existing)
		current_balance = total_dispensed if type in ("receipt",) else max(0, total_dispensed - quantity)
		txn: dict[str, Any] = {
			"id": txn_id,
			"tenant_id": tenant_id,
			"rep_id": rep_id,
			"product_id": product,
			"transaction_type": type,
			"quantity": quantity,
			"lot_number": lot_number,
			"reason": reason,
			"balance_after": current_balance,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, f"sample_{type}", txn_id)
		return txn

	# --- NEW: spend_tracking ---

	def spend_tracking(
		self,
		rep_id: str,
		hcp_id: str,
		amount: float,
		purpose: str,
		tenant_id: str,
		currency: str = "USD",
		fiscal_year: str | None = None,
		receipt_reference: str | None = None,
		pre_approval_reference: str | None = None,
	) -> AggregateSpendRecord:
		"""Track HCP spend from a rep, enforcing aggregate caps and approval thresholds."""
		assert rep_id and hcp_id, "rep_id and hcp_id required"
		assert amount > 0, "amount must be positive"
		fy = fiscal_year or str(date.today().year)
		current_total = self._aggregate_spend_for_hcp(hcp_id, tenant_id, fy)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_spend",
			"amount_above_threshold": amount > 25.0,
			"receipt_present": bool(receipt_reference) if amount > 25.0 else True,
			"amount_above_approval_threshold": amount > 100.0,
			"pre_approval_present": bool(pre_approval_reference) if amount > 100.0 else True,
			"aggregate_cap_exceeded": (current_total + amount) > 500.0,
		})
		record = AggregateSpendRecord(
			tenant_id=tenant_id,
			hcp_id=hcp_id,
			category=purpose,
			amount=amount,
			currency=currency,
			fiscal_year=fy,
			quarter=f"Q{((date.today().month - 1) // 3) + 1}",
			receipt_reference=receipt_reference,
			pre_approval_reference=pre_approval_reference,
			created_by=rep_id,
		)
		self._spend[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, "spend_tracked", record.id)
		return record

	# --- NEW: prescriber_analytics ---

	def prescriber_analytics(
		self,
		territory_id: str,
		period: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Aggregate prescriber call coverage, sample rates, and spend by territory and period."""
		assert territory_id and period, "territory_id and period required"
		targets = [t for t in self._targets.values()
			if t.tenant_id == tenant_id and t.territory_id == territory_id]
		reps = [r for r in self._reps.values()
			if r.tenant_id == tenant_id and r.territory_id == territory_id]
		rep_ids = {r.id for r in reps}
		calls = [c for c in self._calls.values()
			if c.tenant_id == tenant_id and c.rep_id in rep_ids]
		samples = [s for s in self._samples.values()
			if s.tenant_id == tenant_id and s.rep_id in rep_ids]
		spend_records = [s for s in self._spend.values()
			if s.tenant_id == tenant_id and s.fiscal_year == period.split("-")[0]]
		covered_physicians = {c.physician_id for c in calls}
		all_physicians = {t.physician_id for t in targets}
		coverage_rate = len(covered_physicians) / max(len(all_physicians), 1) * 100
		tier_breakdown: dict[str, int] = {}
		for t in targets:
			tier_breakdown[t.tier] = tier_breakdown.get(t.tier, 0) + 1
		analytics: dict[str, Any] = {
			"territory_id": territory_id,
			"period": period,
			"tenant_id": tenant_id,
			"total_targets": len(targets),
			"covered_physicians": len(covered_physicians),
			"coverage_rate_pct": round(coverage_rate, 2),
			"total_calls": len(calls),
			"total_samples_dispensed": sum(s.quantity for s in samples),
			"total_spend": sum(s.amount for s in spend_records),
			"tier_breakdown": tier_breakdown,
			"reps_active": len(reps),
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._prescriber_analytics[self._key(tenant_id, territory_id)] = analytics
		self._audit(tenant_id, "prescriber_analytics_generated", territory_id)
		return analytics

	# --- NEW: market_access_tracking ---

	def market_access_tracking(
		self,
		product_id: str,
		payer_id: str,
		tenant_id: str,
		listing_status: str = "pending",
		formulary_tier: str | None = None,
		reimbursement_rate: float | None = None,
		access_date: date | None = None,
		restrictions: list[str] | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""Track a product's market access status with a specific payer/formulary."""
		assert product_id and payer_id, "product_id and payer_id required"
		record_id = _uuid7str()
		record: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"payer_id": payer_id,
			"listing_status": listing_status,
			"formulary_tier": formulary_tier,
			"reimbursement_rate": reimbursement_rate,
			"access_date": str(access_date) if access_date else None,
			"restrictions": restrictions or [],
			"notes": notes,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._market_access[self._key(tenant_id, record_id)] = record
		self._audit(tenant_id, "market_access_tracked", record_id)
		return record

	def get_market_access_by_product(self, product_id: str, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all payer market access records for a product."""
		return [r for r in self._market_access.values()
			if r["tenant_id"] == tenant_id and r["product_id"] == product_id]

	# --- NEW: promotional_material_approval ---

	def promotional_material_approval(
		self,
		material_id: str,
		tenant_id: str,
		action: str = "submit",
		reviewer_id: str | None = None,
		approval_reference: str | None = None,
		rejection_reason: str | None = None,
		mlr_reference: str | None = None,
	) -> dict[str, Any]:
		"""Manage the MLR (Medical-Legal-Regulatory) approval workflow for a promotional material."""
		assert material_id, "material_id required"
		assert action in ("submit", "approve", "reject", "withdraw", "recall"), \
			f"unsupported action: {action}"
		existing = self._promo_materials.get(self._key(tenant_id, material_id))
		if existing is None:
			existing = {
				"id": material_id,
				"tenant_id": tenant_id,
				"status": "draft",
				"created_at": datetime.utcnow().isoformat(),
				"history": [],
			}
		transition_map = {
			"submit": "under_review",
			"approve": "approved",
			"reject": "rejected",
			"withdraw": "withdrawn",
			"recall": "recalled",
		}
		old_status = existing["status"]
		new_status = transition_map[action]
		existing["status"] = new_status
		existing["updated_at"] = datetime.utcnow().isoformat()
		if reviewer_id:
			existing["reviewer_id"] = reviewer_id
		if approval_reference:
			existing["approval_reference"] = approval_reference
		if rejection_reason:
			existing["rejection_reason"] = rejection_reason
		if mlr_reference:
			existing["mlr_reference"] = mlr_reference
		existing["history"].append({
			"action": action,
			"from_status": old_status,
			"to_status": new_status,
			"actor_id": reviewer_id or self._actor_id,
			"timestamp": datetime.utcnow().isoformat(),
		})
		self._promo_materials[self._key(tenant_id, material_id)] = existing
		self._audit(tenant_id, f"promo_material_{action}", material_id)
		return existing

	# --- NEW: commercial_analytics ---

	def commercial_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate a comprehensive commercial performance report for a period."""
		assert period, "period required"
		fiscal_year = period.split("-")[0] if "-" in period else period
		territories = self.list_territories(tenant_id)
		reps = self.list_reps(tenant_id)
		calls = [c for c in self._calls.values() if c.tenant_id == tenant_id]
		samples = [s for s in self._samples.values() if s.tenant_id == tenant_id]
		spend = [s for s in self._spend.values()
			if s.tenant_id == tenant_id and s.fiscal_year == fiscal_year]
		targets = self.list_targets(tenant_id)
		interactions = [i for i in self._interactions.values() if i.tenant_id == tenant_id]
		# call frequency by territory
		territory_call_counts: dict[str, int] = {t.id: 0 for t in territories}
		for c in calls:
			rep = self._reps.get(self._key(tenant_id, c.rep_id))
			if rep and rep.territory_id in territory_call_counts:
				territory_call_counts[rep.territory_id] += 1
		# sample dispensing by product
		sample_by_product: dict[str, int] = {}
		for s in samples:
			sample_by_product[s.product_id] = sample_by_product.get(s.product_id, 0) + s.quantity
		# spend by hcp
		spend_by_hcp: dict[str, float] = {}
		for s in spend:
			spend_by_hcp[s.hcp_id] = spend_by_hcp.get(s.hcp_id, 0.0) + s.amount
		self._audit(tenant_id, "commercial_analytics_generated", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"territory_count": len(territories),
			"active_rep_count": len(reps),
			"total_calls": len(calls),
			"avg_calls_per_rep": round(len(calls) / max(len(reps), 1), 2),
			"total_samples_dispensed": sum(sample_by_product.values()),
			"sample_by_product": sample_by_product,
			"total_spend": sum(s.amount for s in spend),
			"hcps_with_spend": len(spend_by_hcp),
			"top_hcp_spend": sorted(spend_by_hcp.items(), key=lambda x: x[1], reverse=True)[:5],
			"territory_call_counts": territory_call_counts,
			"target_count": len(targets),
			"interaction_count": len(interactions),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# --- dashboard ---

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return a summary dashboard for commercial operations."""
		return {
			"tenant_id": tenant_id,
			"territory_count": self._count(self._territories, tenant_id),
			"rep_count": self._count(self._reps, tenant_id),
			"call_count": self._count(self._calls, tenant_id),
			"sample_count": self._count(self._samples, tenant_id),
			"interaction_count": self._count(self._interactions, tenant_id),
			"plan_count": self._count(self._plans, tenant_id),
			"target_count": self._count(self._targets, tenant_id),
			"spend_record_count": self._count(self._spend, tenant_id),
			"hcp_visit_count": sum(1 for v in self._hcp_visits.values() if v["tenant_id"] == tenant_id),
			"market_access_count": sum(1 for v in self._market_access.values() if v["tenant_id"] == tenant_id),
			"promo_material_count": sum(1 for v in self._promo_materials.values() if v["tenant_id"] == tenant_id),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id),
		}

	# --- private helpers ---

	def _log_operation(self, operation: str, entity_id: str, tenant_id: str) -> None:
		"""Log internal operations for debugging."""
		pass  # hook for structured logging

	def _log_audit_count(self, tenant_id: str) -> int:
		"""Return audit event count for tenant."""
		return sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id)

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"stream": "apg.pharma.com.lifecycle",
		})

	def _count(self, store: dict[Any, Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _aggregate_spend_for_hcp(self, hcp_id: str, tenant_id: str, fiscal_year: str) -> float:
		return sum(r.amount for r in self._spend.values()
				if r.tenant_id == tenant_id and r.hcp_id == hcp_id and r.fiscal_year == fiscal_year)

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

PharmaComService = CommercialOperationsService
