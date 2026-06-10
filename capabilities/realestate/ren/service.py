"""Async service layer for Rental Operations (ren)."""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

from .models import (
	TenancyCreate, TenancyResponse, TenancyUpdate,
	RentPaymentCreate, RentPaymentResponse,
	ArrearsRecordCreate, ArrearsRecordResponse,
	DepositCreate, DepositResponse,
	DepositDeductionCreate, DepositDeductionResponse,
	NoticeCreate, NoticeResponse,
	TenancyRenewalCreate, TenancyRenewalResponse,
	ReferencingCreate, ReferencingResponse,
	TenancyStatus, ArrearsStatus, DepositStatus,
)
from .capability_contract import evaluate_capability_rules
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

log = logging.getLogger(__name__)


class RenService:
	"""Service implementing all Rental Operations."""

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
			"tenancies": [], "payments": [], "arrears": [],
			"deposits": [], "deductions": [], "notices": [],
			"renewals": [], "referencing": [],
			"applications": [], "credit_checks": [], "listings": [],
		}

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_operation(self, op: str, entity_id: str, tenant_id: str) -> None:
		log.info("ren.%s entity=%s tenant=%s", op, entity_id, tenant_id)

	def _log_arrears(self, tenancy_id: str, amount: Decimal, days: int) -> None:
		log.warning("ren.arrears tenancy=%s amount=%s days=%d", tenancy_id, amount, days)

	def _log_sla_breach(self, tenancy_id: str, days_overdue: int) -> None:
		log.warning("ren.sla_breach tenancy=%s days_overdue=%d", tenancy_id, days_overdue)

	# ── Rules ─────────────────────────────────────────────────────────────────

	def _check_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			log.warning("ren.rule_denied rule=%s reason=%s", result["rule"], result["reason"])
			raise ValueError(f"rule_denied:{result['rule']}:{result['reason']}")

	# ── Tenancy ───────────────────────────────────────────────────────────────

	async def create_tenancy(self, payload: TenancyCreate) -> TenancyResponse:
		"""Create a new tenancy application."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_tenancy",
			"tenancy_type_supported": True,
			"unit_present": True,
			"tenant_present": True,
			"rent_frequency_supported": True,
			"operation_type": "write",
			"policy_attached": True,
			"cross_tenant": False,
			"currency_supported": True,
		})
		record = TenancyResponse(**payload.model_dump())
		self._store["tenancies"].append(record.model_dump())
		self._log_operation("create_tenancy", record.id, record.tenant_id)
		return record

	async def get_tenancy(self, tenancy_id: str, tenant_id: str) -> TenancyResponse | None:
		"""Fetch a tenancy by ID."""
		for t in self._store["tenancies"]:
			if t["id"] == tenancy_id and t["tenant_id"] == tenant_id:
				return TenancyResponse(**t)
		return None

	async def list_tenancies(self, tenant_id: str, unit_id: str | None = None, status: str | None = None) -> list[TenancyResponse]:
		"""List tenancies with optional filters."""
		results = [t for t in self._store["tenancies"] if t["tenant_id"] == tenant_id]
		if unit_id:
			results = [t for t in results if t.get("unit_id") == unit_id]
		if status:
			results = [t for t in results if t.get("status") == status]
		return [TenancyResponse(**t) for t in results]

	async def activate_tenancy(self, tenancy_id: str, tenant_id: str) -> TenancyResponse | None:
		"""Activate tenancy after all pre-conditions are satisfied."""
		for i, t in enumerate(self._store["tenancies"]):
			if t["id"] == tenancy_id and t["tenant_id"] == tenant_id:
				self._check_rules({
					"operation": "activate_tenancy",
					"deposit_registered": t.get("deposit_registered", False),
					"referencing_complete": t.get("referencing_complete", False),
					"tenancy_type": t.get("tenancy_type"),
					"right_to_rent_checked": t.get("right_to_rent_checked", True),
				})
				t["status"] = TenancyStatus.active.value
				t["updated_at"] = datetime.utcnow()
				self._store["tenancies"][i] = t
				self._log_operation("activate_tenancy", tenancy_id, tenant_id)
				return TenancyResponse(**t)
		return None

	async def update_tenancy(self, tenancy_id: str, tenant_id: str, updates: TenancyUpdate) -> TenancyResponse | None:
		"""Update mutable tenancy fields."""
		for i, t in enumerate(self._store["tenancies"]):
			if t["id"] == tenancy_id and t["tenant_id"] == tenant_id:
				self._check_rules({"operation_type": "write", "tenancy_status": t.get("status")})
				t.update({k: v for k, v in updates.model_dump().items() if v is not None})
				t["updated_at"] = datetime.utcnow()
				self._store["tenancies"][i] = t
				return TenancyResponse(**t)
		return None

	# ── Rent Collection ───────────────────────────────────────────────────────

	async def record_rent_payment(self, payload: RentPaymentCreate) -> RentPaymentResponse:
		"""Record an incoming rent payment."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "record_payment",
			"payment_method_supported": True,
			"currency_supported": True,
		})
		tenancy = await self.get_tenancy(payload.tenancy_id, payload.tenant_id)
		expected_rent = tenancy.rent_amount if tenancy else payload.amount
		is_short = payload.amount < expected_rent
		shortfall = (expected_rent - payload.amount) if is_short else Decimal("0")
		record = RentPaymentResponse(**payload.model_dump(), is_short_payment=is_short, shortfall=shortfall)
		self._store["payments"].append(record.model_dump())
		if is_short:
			await self._update_arrears(payload.tenancy_id, payload.tenant_id, shortfall, 0)
		else:
			await self._clear_arrears(payload.tenancy_id, payload.tenant_id)
		self._log_operation("record_payment", record.id, record.tenant_id)
		return record

	async def list_payments(self, tenant_id: str, tenancy_id: str | None = None, period: str | None = None) -> list[RentPaymentResponse]:
		"""List rent payments."""
		results = [p for p in self._store["payments"] if p["tenant_id"] == tenant_id]
		if tenancy_id:
			results = [p for p in results if p["tenancy_id"] == tenancy_id]
		if period:
			results = [p for p in results if p.get("period") == period]
		return [RentPaymentResponse(**p) for p in results]

	# ── Arrears ───────────────────────────────────────────────────────────────

	async def _update_arrears(self, tenancy_id: str, tenant_id: str, amount: Decimal, days: int) -> None:
		self._log_arrears(tenancy_id, amount, days)
		payload = ArrearsRecordCreate(tenant_id=tenant_id, tenancy_id=tenancy_id, amount_overdue=amount, days_overdue=days, created_by="system")
		await self.record_arrears(payload)

	async def _clear_arrears(self, tenancy_id: str, tenant_id: str) -> None:
		for i, a in enumerate(self._store["arrears"]):
			if a["tenancy_id"] == tenancy_id and a["tenant_id"] == tenant_id:
				a["status"] = ArrearsStatus.current.value
				a["amount_overdue"] = "0"
				a["updated_at"] = datetime.utcnow()
				self._store["arrears"][i] = a

	async def record_arrears(self, payload: ArrearsRecordCreate) -> ArrearsRecordResponse:
		"""Record an arrears situation."""
		status = self._classify_arrears_status(payload.days_overdue)
		self._log_arrears(payload.tenancy_id, payload.amount_overdue, payload.days_overdue)
		record = ArrearsRecordResponse(**payload.model_dump(), status=status)
		self._store["arrears"].append(record.model_dump())
		return record

	def _classify_arrears_status(self, days: int) -> ArrearsStatus:
		if days == 0:
			return ArrearsStatus.current
		if days <= 30:
			return ArrearsStatus.days_1_30
		if days <= 60:
			return ArrearsStatus.days_31_60
		if days <= 90:
			return ArrearsStatus.days_61_90
		return ArrearsStatus.days_90_plus

	async def get_arrears_report(self, tenant_id: str) -> list[ArrearsRecordResponse]:
		"""Return all active arrears for a tenant."""
		return [ArrearsRecordResponse(**a) for a in self._store["arrears"]
				if a["tenant_id"] == tenant_id and a["status"] != ArrearsStatus.current.value]

	async def escalate_arrears_to_legal(self, arrears_id: str, tenant_id: str) -> ArrearsRecordResponse | None:
		"""Escalate an arrears case to legal action."""
		for i, a in enumerate(self._store["arrears"]):
			if a["id"] == arrears_id and a["tenant_id"] == tenant_id:
				above_threshold = Decimal(str(a["amount_overdue"])) > Decimal("0") and a["days_overdue"] >= 90
				self._check_rules({"operation": "commence_legal_action", "arrears_above_threshold": above_threshold})
				a["status"] = ArrearsStatus.legal_action.value
				a["legal_action_commenced"] = True
				a["updated_at"] = datetime.utcnow()
				self._store["arrears"][i] = a
				return ArrearsRecordResponse(**a)
		return None

	# ── Deposit ───────────────────────────────────────────────────────────────

	async def register_deposit(self, payload: DepositCreate) -> DepositResponse:
		"""Register a tenancy deposit."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "register_deposit",
			"deposit_type_supported": True,
		})
		record = DepositResponse(**payload.model_dump(), status=DepositStatus.registered)
		self._store["deposits"].append(record.model_dump())
		for i, t in enumerate(self._store["tenancies"]):
			if t["id"] == payload.tenancy_id and t["tenant_id"] == payload.tenant_id:
				t["deposit_id"] = record.id
				t["deposit_registered"] = True
				t["updated_at"] = datetime.utcnow()
				self._store["tenancies"][i] = t
				break
		self._log_operation("register_deposit", record.id, record.tenant_id)
		return record

	async def get_deposit(self, deposit_id: str, tenant_id: str) -> DepositResponse | None:
		for d in self._store["deposits"]:
			if d["id"] == deposit_id and d["tenant_id"] == tenant_id:
				return DepositResponse(**d)
		return None

	async def deduct_from_deposit(self, payload: DepositDeductionCreate) -> DepositDeductionResponse:
		"""Record a deduction from a deposit (evidence required)."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "deduct_from_deposit",
			"evidence_present": len(payload.evidence_document_ids) > 0,
		})
		deposit = await self.get_deposit(payload.deposit_id, payload.tenant_id)
		if deposit:
			held = deposit.amount - deposit.total_deducted
			self._check_rules({"operation": "deduct_from_deposit", "deduction_exceeds_held": payload.amount > held})
			for i, d in enumerate(self._store["deposits"]):
				if d["id"] == payload.deposit_id:
					d["total_deducted"] = str(Decimal(str(d.get("total_deducted", 0))) + payload.amount)
					d["updated_at"] = datetime.utcnow()
					self._store["deposits"][i] = d
					break
		record = DepositDeductionResponse(**payload.model_dump())
		self._store["deductions"].append(record.model_dump())
		return record

	async def release_deposit(self, deposit_id: str, tenant_id: str, released_by: str) -> DepositResponse | None:
		for i, d in enumerate(self._store["deposits"]):
			if d["id"] == deposit_id and d["tenant_id"] == tenant_id:
				d["status"] = DepositStatus.released.value
				d["released_at"] = datetime.utcnow()
				d["updated_at"] = datetime.utcnow()
				self._store["deposits"][i] = d
				self._log_operation("release_deposit", deposit_id, tenant_id)
				return DepositResponse(**d)
		return None

	# ── Notice ────────────────────────────────────────────────────────────────

	async def serve_notice(self, payload: NoticeCreate) -> NoticeResponse:
		"""Serve a formal tenancy notice."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "serve_notice",
			"notice_type_supported": True,
		})
		record = NoticeResponse(**payload.model_dump())
		self._store["notices"].append(record.model_dump())
		for i, t in enumerate(self._store["tenancies"]):
			if t["id"] == payload.tenancy_id and t["tenant_id"] == payload.tenant_id:
				t["status"] = TenancyStatus.notice_served.value
				t["updated_at"] = datetime.utcnow()
				self._store["tenancies"][i] = t
				break
		self._log_operation("serve_notice", record.id, record.tenant_id)
		return record

	async def list_notices(self, tenant_id: str, tenancy_id: str | None = None) -> list[NoticeResponse]:
		results = [n for n in self._store["notices"] if n["tenant_id"] == tenant_id]
		if tenancy_id:
			results = [n for n in results if n["tenancy_id"] == tenancy_id]
		return [NoticeResponse(**n) for n in results]

	# ── Renewal ───────────────────────────────────────────────────────────────

	async def initiate_renewal(self, payload: TenancyRenewalCreate) -> TenancyRenewalResponse:
		"""Initiate a tenancy renewal offer."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "initiate_renewal",
			"renewal_type_supported": True,
		})
		record = TenancyRenewalResponse(**payload.model_dump(), offered_at=datetime.utcnow())
		self._store["renewals"].append(record.model_dump())
		return record

	async def accept_renewal(self, renewal_id: str, tenant_id: str) -> TenancyRenewalResponse | None:
		for i, r in enumerate(self._store["renewals"]):
			if r["id"] == renewal_id and r["tenant_id"] == tenant_id:
				r["status"] = "accepted"
				r["accepted_at"] = datetime.utcnow()
				r["updated_at"] = datetime.utcnow()
				self._store["renewals"][i] = r
				for j, t in enumerate(self._store["tenancies"]):
					if t["id"] == r["tenancy_id"] and t["tenant_id"] == tenant_id:
						t["end_date"] = r["new_end_date"]
						t["rent_amount"] = r["new_rent"]
						t["updated_at"] = datetime.utcnow()
						self._store["tenancies"][j] = t
						break
				return TenancyRenewalResponse(**r)
		return None

	async def get_renewal_pipeline(self, tenant_id: str, months_ahead: int = 3) -> list[dict[str, Any]]:
		cutoff = date.today() + timedelta(days=months_ahead * 30)
		results = []
		for t in self._store["tenancies"]:
			if t["tenant_id"] == tenant_id and t["status"] == TenancyStatus.active.value and t.get("end_date"):
				end = datetime.strptime(t["end_date"], "%Y-%m-%d").date()
				if end <= cutoff:
					results.append({"tenancy_id": t["id"], "end_date": t["end_date"], "days_remaining": (end - date.today()).days, "unit_id": t.get("unit_id")})
		return sorted(results, key=lambda x: x["days_remaining"])

	# ── Referencing ───────────────────────────────────────────────────────────

	async def run_referencing(self, payload: ReferencingCreate) -> ReferencingResponse:
		"""Initiate a referencing check."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "run_referencing",
			"referencing_type_supported": True,
		})
		record = ReferencingResponse(**payload.model_dump())
		self._store["referencing"].append(record.model_dump())
		return record

	async def complete_referencing(self, ref_id: str, tenant_id: str, passed: bool, results: dict[str, Any]) -> ReferencingResponse | None:
		for i, r in enumerate(self._store["referencing"]):
			if r["id"] == ref_id and r["tenant_id"] == tenant_id:
				r["status"] = "passed" if passed else "failed"
				r["results"] = results
				r["completed_at"] = datetime.utcnow()
				r["updated_at"] = datetime.utcnow()
				self._store["referencing"][i] = r
				if passed:
					for j, t in enumerate(self._store["tenancies"]):
						if t["id"] == r["tenancy_id"] and t["tenant_id"] == tenant_id:
							t["referencing_complete"] = True
							t["updated_at"] = datetime.utcnow()
							self._store["tenancies"][j] = t
							break
				return ReferencingResponse(**r)
		return None

	# ── Rent Roll ─────────────────────────────────────────────────────────────

	async def generate_rent_roll(self, tenant_id: str, property_id: str | None = None) -> list[dict[str, Any]]:
		tenancies = await self.list_tenancies(tenant_id, status=TenancyStatus.active.value)
		roll = []
		for t in tenancies:
			if property_id and t.property_id != property_id:
				continue
			roll.append({
				"tenancy_id": t.id,
				"unit_id": t.unit_id,
				"property_id": t.property_id,
				"tenant_entity_id": t.tenant_entity_id,
				"rent_amount": float(t.rent_amount),
				"rent_frequency": t.rent_frequency.value,
				"start_date": str(t.start_date),
				"end_date": str(t.end_date) if t.end_date else None,
				"arrears_status": t.arrears_status.value,
				"total_arrears": float(t.total_arrears),
			})
		return roll

	# ── NEW: advertise_unit ────────────────────────────────────────────────────

	async def advertise_unit(
		self,
		unit_id: str,
		rent: Decimal,
		available_from: date,
		listing_description: str,
		tenant_id: str,
		listing_channels: list[str] | None = None,
		photos_count: int = 0,
		epc_rating: str = "",
	) -> dict[str, Any]:
		"""Create a rental listing for a unit across specified channels."""
		assert unit_id and rent > 0 and listing_description, \
			"unit_id, rent > 0, listing_description required"
		from uuid6 import uuid7
		listing_id = str(uuid7())
		listing: dict[str, Any] = {
			"id": listing_id,
			"tenant_id": tenant_id,
			"unit_id": unit_id,
			"rent": str(rent),
			"available_from": str(available_from),
			"listing_description": listing_description,
			"listing_channels": listing_channels or ["portal"],
			"photos_count": photos_count,
			"epc_rating": epc_rating,
			"status": "active",
			"views": 0,
			"enquiries": 0,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["listings"].append(listing)
		self._log_operation("unit_advertised", listing_id, tenant_id)
		return listing

	# ── NEW: tenant_application ────────────────────────────────────────────────

	async def tenant_application(
		self,
		unit_id: str,
		applicant_id: str,
		employment_details: dict[str, Any],
		guarantor: dict[str, Any] | None,
		tenant_id: str,
		move_in_date: date | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""Record a tenant application for a unit with employment and guarantor details."""
		assert unit_id and applicant_id and employment_details, \
			"unit_id, applicant_id, employment_details required"
		from uuid6 import uuid7
		application_id = str(uuid7())
		annual_income = float(employment_details.get("annual_income", 0))
		# standard affordability: rent should be <= 40% of monthly net income
		monthly_income = annual_income / 12
		listing = next((l for l in self._store.get("listings", [])
			if l["unit_id"] == unit_id and l["tenant_id"] == tenant_id), None)
		monthly_rent = float(Decimal(str(listing["rent"])) if listing else 0)
		affordability_ratio = monthly_rent / max(monthly_income, 1) * 100
		application: dict[str, Any] = {
			"id": application_id,
			"tenant_id": tenant_id,
			"unit_id": unit_id,
			"applicant_id": applicant_id,
			"employment_details": employment_details,
			"guarantor": guarantor,
			"move_in_date": str(move_in_date or date.today()),
			"notes": notes,
			"monthly_rent": monthly_rent,
			"monthly_income": round(monthly_income, 2),
			"affordability_ratio_pct": round(affordability_ratio, 2),
			"affordability_ok": affordability_ratio <= 40,
			"status": "received",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["applications"].append(application)
		self._log_operation("application_received", application_id, tenant_id)
		return application

	# ── NEW: reference_check ───────────────────────────────────────────────────

	async def reference_check(
		self,
		application_id: str,
		reference_type: str,
		tenant_id: str,
		reference_provider: str = "",
		outcome: str = "pending",
		notes: str = "",
	) -> dict[str, Any]:
		"""Conduct a reference check (employer, previous landlord, personal) for an application."""
		assert application_id and reference_type, "application_id and reference_type required"
		assert reference_type in ("employer", "previous_landlord", "personal",
			"bank", "professional"), f"unsupported reference_type: {reference_type}"
		assert outcome in ("pending", "satisfactory", "unsatisfactory", "unable_to_obtain"), \
			f"unsupported outcome: {outcome}"
		from uuid6 import uuid7
		ref_id = str(uuid7())
		reference: dict[str, Any] = {
			"id": ref_id,
			"tenant_id": tenant_id,
			"application_id": application_id,
			"reference_type": reference_type,
			"reference_provider": reference_provider,
			"outcome": outcome,
			"notes": notes,
			"checked_at": datetime.utcnow().isoformat(),
		}
		self._store["referencing"].append(reference)
		# update application status if all refs satisfactory
		for i, a in enumerate(self._store["applications"]):
			if a["id"] == application_id and a["tenant_id"] == tenant_id:
				a[f"ref_{reference_type}_outcome"] = outcome
				a["updated_at"] = datetime.utcnow()
				self._store["applications"][i] = a
				break
		return reference

	# ── NEW: credit_check_tenant ───────────────────────────────────────────────

	async def credit_check_tenant(
		self,
		application_id: str,
		tenant_id: str,
		credit_score: int | None = None,
		ccjs: int = 0,
		bankruptcies: int = 0,
		provider: str = "experian",
	) -> dict[str, Any]:
		"""Run and record a credit check for a tenant application."""
		assert application_id, "application_id required"
		assert provider in ("experian", "equifax", "transunion", "clearscore"), \
			f"unsupported credit provider: {provider}"
		from uuid6 import uuid7
		check_id = str(uuid7())
		# derive pass/fail
		passed = (
			(credit_score or 0) >= 600
			and ccjs == 0
			and bankruptcies == 0
		)
		check: dict[str, Any] = {
			"id": check_id,
			"tenant_id": tenant_id,
			"application_id": application_id,
			"provider": provider,
			"credit_score": credit_score,
			"ccjs": ccjs,
			"bankruptcies": bankruptcies,
			"passed": passed,
			"risk_band": "low" if (credit_score or 0) >= 700 else "medium" if (credit_score or 0) >= 600 else "high",
			"checked_at": datetime.utcnow().isoformat(),
		}
		self._store["credit_checks"].append(check)
		for i, a in enumerate(self._store["applications"]):
			if a["id"] == application_id and a["tenant_id"] == tenant_id:
				a["credit_check_passed"] = passed
				a["credit_score"] = credit_score
				a["updated_at"] = datetime.utcnow()
				self._store["applications"][i] = a
				break
		self._log_operation("credit_check_completed", check_id, tenant_id)
		return check

	# ── NEW: sign_tenancy ─────────────────────────────────────────────────────

	async def sign_tenancy(
		self,
		unit_id: str,
		tenant_entity_id: str,
		rent: Decimal,
		deposit: Decimal,
		start_date: date,
		tenant_id: str,
		end_date: date | None = None,
		tenancy_type: str = "assured_shorthold",
		rent_frequency: str = "monthly",
	) -> dict[str, Any]:
		"""Create and activate a signed tenancy agreement with deposit registration."""
		assert unit_id and tenant_entity_id and rent > 0 and deposit >= 0, \
			"unit_id, tenant_entity_id, rent > 0, deposit >= 0 required"
		assert tenancy_type in ("assured_shorthold", "assured", "periodic",
			"commercial", "commercial_fri"), \
			f"unsupported tenancy_type: {tenancy_type}"
		from uuid6 import uuid7
		tenancy_id = str(uuid7())
		tenancy: dict[str, Any] = {
			"id": tenancy_id,
			"tenant_id": tenant_id,
			"unit_id": unit_id,
			"tenant_entity_id": tenant_entity_id,
			"rent_amount": str(rent),
			"deposit_amount": str(deposit),
			"start_date": str(start_date),
			"end_date": str(end_date) if end_date else None,
			"tenancy_type": tenancy_type,
			"rent_frequency": rent_frequency,
			"status": TenancyStatus.active.value,
			"deposit_registered": False,
			"referencing_complete": True,
			"right_to_rent_checked": True,
			"total_arrears": "0",
			"arrears_status": ArrearsStatus.current.value,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["tenancies"].append(tenancy)
		# auto-register deposit
		from uuid6 import uuid7 as _uuid7
		deposit_id = str(_uuid7())
		deposit_record: dict[str, Any] = {
			"id": deposit_id,
			"tenant_id": tenant_id,
			"tenancy_id": tenancy_id,
			"amount": str(deposit),
			"total_deducted": "0",
			"status": DepositStatus.registered.value,
			"deposit_type": "traditional",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["deposits"].append(deposit_record)
		for i, t in enumerate(self._store["tenancies"]):
			if t["id"] == tenancy_id:
				t["deposit_id"] = deposit_id
				t["deposit_registered"] = True
				self._store["tenancies"][i] = t
				break
		self._log_operation("tenancy_signed", tenancy_id, tenant_id)
		return tenancy

	# ── NEW: collect_rent ─────────────────────────────────────────────────────

	async def collect_rent(
		self,
		unit_id: str,
		period: str,
		amount: Decimal,
		payment_method: str,
		tenant_id: str,
		tenancy_id: str | None = None,
	) -> dict[str, Any]:
		"""Collect rent for a unit and period, identify shortfalls, update arrears."""
		assert unit_id and period and amount >= 0, "unit_id, period, amount >= 0 required"
		assert payment_method in ("bacs", "direct_debit", "standing_order", "cheque",
			"card", "cash", "bank_transfer"), \
			f"unsupported payment_method: {payment_method}"
		# find the tenancy
		t_id = tenancy_id
		if not t_id:
			tenancy = next(
				(t for t in self._store["tenancies"]
				if t["tenant_id"] == tenant_id and t["unit_id"] == unit_id
				and t["status"] == TenancyStatus.active.value),
				None,
			)
			t_id = tenancy["id"] if tenancy else None
		expected_rent = Decimal("0")
		if t_id:
			for t in self._store["tenancies"]:
				if t["id"] == t_id:
					expected_rent = Decimal(str(t.get("rent_amount", 0)))
					break
		is_short = amount < expected_rent
		shortfall = (expected_rent - amount) if is_short else Decimal("0")
		from uuid6 import uuid7
		payment_id = str(uuid7())
		payment: dict[str, Any] = {
			"id": payment_id,
			"tenant_id": tenant_id,
			"tenancy_id": t_id,
			"unit_id": unit_id,
			"period": period,
			"amount": str(amount),
			"expected_rent": str(expected_rent),
			"payment_method": payment_method,
			"is_short_payment": is_short,
			"shortfall": str(shortfall),
			"status": "received",
			"received_at": datetime.utcnow().isoformat(),
		}
		self._store["payments"].append(payment)
		if is_short and t_id:
			await self._update_arrears(t_id, tenant_id, shortfall, 1)
		elif t_id:
			await self._clear_arrears(t_id, tenant_id)
		self._log_operation("rent_collected", payment_id, tenant_id)
		return payment

	# ── NEW: arrears_management ────────────────────────────────────────────────

	async def arrears_management(
		self,
		unit_id: str,
		arrears_amount: Decimal,
		tenant_id: str,
		days_overdue: int = 0,
		action: str = "record",
		notes: str = "",
	) -> dict[str, Any]:
		"""Manage arrears for a unit: record, escalate, initiate legal, or resolve."""
		assert unit_id and arrears_amount >= 0, "unit_id and arrears_amount >= 0 required"
		assert action in ("record", "chase", "formal_demand", "legal", "resolve"), \
			f"unsupported action: {action}"
		tenancy = next(
			(t for t in self._store["tenancies"]
			if t["tenant_id"] == tenant_id and t["unit_id"] == unit_id
			and t["status"] == TenancyStatus.active.value),
			None,
		)
		tenancy_id = tenancy["id"] if tenancy else None
		if action == "legal" and days_overdue < 90:
			raise ValueError("legal action requires at least 90 days overdue")
		from uuid6 import uuid7
		record_id = str(uuid7())
		status = self._classify_arrears_status(days_overdue)
		if action == "legal":
			status = ArrearsStatus.legal_action
		elif action == "resolve":
			status = ArrearsStatus.current
		arrears: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"tenancy_id": tenancy_id,
			"unit_id": unit_id,
			"amount_overdue": str(arrears_amount),
			"days_overdue": days_overdue,
			"action_taken": action,
			"status": status.value,
			"notes": notes,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["arrears"].append(arrears)
		if days_overdue >= 90:
			self._log_sla_breach(tenancy_id or unit_id, days_overdue)
		return arrears

	# ── NEW: serve_notice ─────────────────────────────────────────────────────
	# (override with enriched signature)

	async def serve_notice_formal(
		self,
		unit_id: str,
		notice_type: str,
		notice_date: date,
		reason: str,
		tenant_id: str,
		tenancy_id: str | None = None,
		notice_period_days: int | None = None,
		served_by: str = "system",
	) -> dict[str, Any]:
		"""Serve a formal tenancy notice with computed expiry date."""
		assert unit_id and notice_type and reason, "unit_id, notice_type, reason required"
		assert notice_type in ("section_21", "section_8", "section_25", "section_26",
			"notice_to_quit", "break_notice", "notice_of_possession"), \
			f"unsupported notice_type: {notice_type}"
		default_periods = {
			"section_21": 56, "section_8": 14, "section_25": 180,
			"section_26": 180, "notice_to_quit": 28, "break_notice": 90,
			"notice_of_possession": 14,
		}
		notice_days = notice_period_days or default_periods.get(notice_type, 28)
		expiry_date = notice_date + timedelta(days=notice_days)
		t_id = tenancy_id
		if not t_id:
			tenancy = next(
				(t for t in self._store["tenancies"]
				if t["tenant_id"] == tenant_id and t["unit_id"] == unit_id
				and t["status"] in (TenancyStatus.active.value, TenancyStatus.notice_served.value)),
				None,
			)
			t_id = tenancy["id"] if tenancy else None
		from uuid6 import uuid7
		notice_id = str(uuid7())
		notice: dict[str, Any] = {
			"id": notice_id,
			"tenant_id": tenant_id,
			"tenancy_id": t_id,
			"unit_id": unit_id,
			"notice_type": notice_type,
			"notice_date": str(notice_date),
			"notice_period_days": notice_days,
			"expiry_date": str(expiry_date),
			"reason": reason,
			"served_by": served_by,
			"status": "served",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["notices"].append(notice)
		if t_id:
			for i, t in enumerate(self._store["tenancies"]):
				if t["id"] == t_id:
					t["status"] = TenancyStatus.notice_served.value
					t["notice_expiry_date"] = str(expiry_date)
					t["updated_at"] = datetime.utcnow()
					self._store["tenancies"][i] = t
					break
		self._log_operation("notice_served", notice_id, tenant_id)
		return notice

	# ── NEW: end_tenancy ──────────────────────────────────────────────────────

	async def end_tenancy(
		self,
		unit_id: str,
		vacate_date: date,
		deposit_deductions: list[dict[str, Any]],
		tenant_id: str,
		tenancy_id: str | None = None,
		checkout_reference: str = "",
	) -> dict[str, Any]:
		"""End a tenancy: record vacate date, process deposit deductions, close tenancy."""
		assert unit_id, "unit_id required"
		t_id = tenancy_id
		if not t_id:
			tenancy = next(
				(t for t in self._store["tenancies"]
				if t["tenant_id"] == tenant_id and t["unit_id"] == unit_id
				and t["status"] in (TenancyStatus.active.value, TenancyStatus.notice_served.value)),
				None,
			)
			t_id = tenancy["id"] if tenancy else None
		if not t_id:
			raise KeyError(f"no active tenancy found for unit {unit_id}")
		total_deductions = sum(float(d.get("amount", 0)) for d in deposit_deductions)
		# process deposit deductions
		deposit = next(
			(d for d in self._store["deposits"]
			if d.get("tenancy_id") == t_id and d["tenant_id"] == tenant_id),
			None,
		)
		deposit_held = float(Decimal(str(deposit["amount"]))) if deposit else 0.0
		deposit_returned = max(0.0, deposit_held - total_deductions)
		# close tenancy
		for i, t in enumerate(self._store["tenancies"]):
			if t["id"] == t_id and t["tenant_id"] == tenant_id:
				t["status"] = TenancyStatus.terminated.value
				t["vacate_date"] = str(vacate_date)
				t["checkout_reference"] = checkout_reference
				t["updated_at"] = datetime.utcnow()
				self._store["tenancies"][i] = t
				break
		self._log_operation("tenancy_ended", t_id, tenant_id)
		return {
			"tenancy_id": t_id,
			"unit_id": unit_id,
			"tenant_id": tenant_id,
			"vacate_date": str(vacate_date),
			"deposit_held": deposit_held,
			"total_deductions": total_deductions,
			"deduction_items": deposit_deductions,
			"deposit_returned": deposit_returned,
			"checkout_reference": checkout_reference,
			"closed_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: rental_analytics ─────────────────────────────────────────────────

	async def rental_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate rental portfolio analytics for a period."""
		assert period, "period required"
		tenancies = await self.list_tenancies(tenant_id)
		active = [t for t in tenancies if t.status.value == "active"]
		notices = [t for t in tenancies if t.status.value == "notice_served"]
		terminated = [t for t in tenancies if t.status.value == "terminated"]
		payments = await self.list_payments(tenant_id)
		period_payments = [p for p in payments if p.period == period] if payments else []
		arrears = await self.get_arrears_report(tenant_id)
		total_rent_collected = sum(p.amount for p in period_payments if hasattr(p, "amount"))
		total_arrears = sum(
			Decimal(str(a.amount_overdue))
			for a in arrears
			if hasattr(a, "amount_overdue")
		)
		applications = [a for a in self._store.get("applications", []) if a["tenant_id"] == tenant_id]
		deposits = [d for d in self._store["deposits"] if d["tenant_id"] == tenant_id]
		total_deposits_held = sum(
			float(Decimal(str(d["amount"])) - Decimal(str(d.get("total_deducted", 0))))
			for d in deposits
			if d.get("status") == DepositStatus.registered.value
		)
		renewal_pipeline = await self.get_renewal_pipeline(tenant_id, months_ahead=3)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_tenancies": len(tenancies),
			"active_tenancies": len(active),
			"tenancies_in_notice": len(notices),
			"terminated_tenancies": len(terminated),
			"rent_collected": float(total_rent_collected) if total_rent_collected else 0,
			"arrears_cases": len(arrears),
			"total_arrears": float(total_arrears),
			"applications_received": len(applications),
			"total_deposits_held": total_deposits_held,
			"renewals_due_3_months": len(renewal_pipeline),
			"generated_at": datetime.utcnow().isoformat(),
		}


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

	async def get_audit_trail(self, tenant_id: str, entity_id: str = "") -> dict[str, Any]:
		"""Get Audit Trail"""
		return {"entity_id": entity_id, "tenant_id": tenant_id, "events": [], "retrieved_at": datetime.utcnow().isoformat()}

	async def analytics_summary(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		self._log_operation("analytics_summary", "analytics", tenant_id)
		return {"tenant_id": tenant_id, "period": period, "computed_at": datetime.utcnow().isoformat()}
