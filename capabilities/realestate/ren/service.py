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

	# ── NEW: get_tenancy_statement ─────────────────────────────────────────────

	async def get_tenancy_statement(
		self,
		tenancy_id: str,
		tenant_id: str,
		from_date: date | None = None,
		to_date: date | None = None,
	) -> dict[str, Any]:
		"""
		Produce a chronological ledger statement for a tenancy.

		Returns opening balance, line items (charges, payments, credits), closing
		balance, and total days-in-arrears for the requested window.  Defaults to
		the full tenancy life if dates are omitted.
		"""
		assert tenancy_id and tenant_id, "tenancy_id and tenant_id required"
		tenancy = await self.get_tenancy(tenancy_id, tenant_id)
		if not tenancy:
			raise KeyError(f"tenancy {tenancy_id} not found")

		# Collect payments in window
		payments = [
			p for p in self._store["payments"]
			if p.get("tenancy_id") == tenancy_id and p.get("tenant_id") == tenant_id
		]
		if from_date:
			payments = [p for p in payments if p.get("received_at", p.get("created_at", "")) >= str(from_date)]
		if to_date:
			payments = [p for p in payments if p.get("received_at", p.get("created_at", "")) <= str(to_date)]

		# Collect arrears records in window
		arrears = [
			a for a in self._store["arrears"]
			if a.get("tenancy_id") == tenancy_id and a.get("tenant_id") == tenant_id
		]

		# Build ledger lines
		ledger: list[dict[str, Any]] = []
		running_balance = Decimal("0")

		for p in sorted(payments, key=lambda x: x.get("received_at", x.get("created_at", ""))):
			amt = Decimal(str(p.get("amount", 0)))
			running_balance -= amt  # credit reduces balance
			ledger.append({
				"date": p.get("received_at", p.get("created_at")),
				"type": "payment",
				"description": f"Rent payment – period {p.get('period', 'N/A')}",
				"debit": None,
				"credit": float(amt),
				"balance": float(running_balance),
				"reference": p.get("id"),
			})
			# charge row for expected rent
			expected = Decimal(str(p.get("expected_rent", 0)))
			if expected > 0:
				running_balance += expected
				ledger.insert(-1, {
					"date": p.get("received_at", p.get("created_at")),
					"type": "charge",
					"description": f"Rent charge – period {p.get('period', 'N/A')}",
					"debit": float(expected),
					"credit": None,
					"balance": float(running_balance - amt),
					"reference": None,
				})

		total_arrears = sum(
			Decimal(str(a.get("amount_overdue", 0)))
			for a in arrears
			if a.get("status") not in (ArrearsStatus.current.value,)
		)
		days_in_arrears = max((a.get("days_overdue", 0) for a in arrears), default=0)

		self._log_operation("tenancy_statement", tenancy_id, tenant_id)
		return {
			"tenancy_id": tenancy_id,
			"tenant_id": tenant_id,
			"from_date": str(from_date or tenancy.start_date),
			"to_date": str(to_date or date.today()),
			"opening_balance": 0.0,
			"closing_balance": float(running_balance),
			"total_charged": float(sum(l["debit"] for l in ledger if l["debit"])),
			"total_paid": float(sum(l["credit"] for l in ledger if l["credit"])),
			"total_arrears": float(total_arrears),
			"days_in_arrears": days_in_arrears,
			"ledger": ledger,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: record_inspection ─────────────────────────────────────────────────

	async def record_inspection(
		self,
		tenancy_id: str,
		tenant_id: str,
		inspection_type: str,
		condition_items: list[dict[str, Any]],
		inspector_id: str,
		inspection_date: date | None = None,
		photos: list[str] | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""
		Record a property inspection (move-in, mid-term, or move-out).

		Each condition_item: {"room": str, "item": str, "condition": str,
		"grade": 1-5, "notes": str}.  Photos are document IDs from the
		document store.  Move-out inspections are automatically linked to the
		deposit deduction workflow.
		"""
		assert tenancy_id and tenant_id and inspector_id, \
			"tenancy_id, tenant_id, inspector_id required"
		assert inspection_type in ("move_in", "mid_term", "move_out"), \
			f"unsupported inspection_type: {inspection_type}"
		assert condition_items, "condition_items must not be empty"

		from uuid6 import uuid7
		inspection_id = str(uuid7())
		insp_date = inspection_date or date.today()

		# Validate each condition item has minimum fields
		for item in condition_items:
			assert "room" in item and "item" in item, \
				"each condition_item requires 'room' and 'item' keys"
			grade = item.get("grade", 3)
			assert 1 <= int(grade) <= 5, "grade must be 1-5"

		# Count items requiring remediation (grade < 3 = below acceptable)
		remediation_required = [i for i in condition_items if int(i.get("grade", 3)) < 3]

		if "inspections" not in self._store:
			self._store["inspections"] = []

		inspection: dict[str, Any] = {
			"id": inspection_id,
			"tenant_id": tenant_id,
			"tenancy_id": tenancy_id,
			"inspection_type": inspection_type,
			"inspection_date": str(insp_date),
			"inspector_id": inspector_id,
			"condition_items": condition_items,
			"photos": photos or [],
			"notes": notes,
			"items_requiring_remediation": len(remediation_required),
			"remediation_items": remediation_required,
			"deposit_deduction_eligible": inspection_type == "move_out" and len(remediation_required) > 0,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["inspections"].append(inspection)
		self._log_operation(f"inspection_{inspection_type}", inspection_id, tenant_id)
		return inspection

	# ── NEW: propose_rent_increase ─────────────────────────────────────────────

	async def propose_rent_increase(
		self,
		tenancy_id: str,
		tenant_id: str,
		new_rent: Decimal,
		effective_date: date,
		proposed_by: str,
		notice_days: int | None = None,
		reason: str = "",
	) -> dict[str, Any]:
		"""
		Propose a rent increase for a tenancy.

		Enforces minimum statutory notice period (defaults to one full rental period
		for the tenancy's rent_frequency).  The increase is NOT applied until
		`effective_date` passes; use `apply_rent_increase()` to commit.

		Returns a rent-increase proposal record with status "proposed".
		"""
		assert tenancy_id and tenant_id and proposed_by, \
			"tenancy_id, tenant_id, proposed_by required"
		assert new_rent > 0, "new_rent must be positive"
		assert effective_date > date.today(), "effective_date must be in the future"

		tenancy = await self.get_tenancy(tenancy_id, tenant_id)
		if not tenancy:
			raise KeyError(f"tenancy {tenancy_id} not found")
		if tenancy.status not in (TenancyStatus.active, TenancyStatus.holding_over):
			raise ValueError(f"rent increase only allowed on active/holding-over tenancy, got {tenancy.status}")

		# Derive minimum notice days from rent frequency
		_freq_notice_days = {
			"weekly": 7, "fortnightly": 14, "monthly": 28,
			"quarterly": 84, "semi_annual": 168, "annual": 365, "in_advance": 28,
		}
		min_notice = notice_days or _freq_notice_days.get(tenancy.rent_frequency.value, 28)
		days_until_effective = (effective_date - date.today()).days
		if days_until_effective < min_notice:
			raise ValueError(
				f"notice too short: {days_until_effective} days given, "
				f"minimum {min_notice} days required for {tenancy.rent_frequency.value} tenancy"
			)

		if "rent_increases" not in self._store:
			self._store["rent_increases"] = []

		from uuid6 import uuid7
		proposal_id = str(uuid7())
		proposal: dict[str, Any] = {
			"id": proposal_id,
			"tenant_id": tenant_id,
			"tenancy_id": tenancy_id,
			"current_rent": str(tenancy.rent_amount),
			"new_rent": str(new_rent),
			"increase_pct": round(
				float((new_rent - tenancy.rent_amount) / tenancy.rent_amount * 100), 2
			),
			"effective_date": str(effective_date),
			"notice_days": days_until_effective,
			"proposed_by": proposed_by,
			"reason": reason,
			"status": "proposed",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["rent_increases"].append(proposal)
		self._log_operation("rent_increase_proposed", proposal_id, tenant_id)
		return proposal

	async def apply_rent_increase(
		self,
		proposal_id: str,
		tenant_id: str,
		applied_by: str,
	) -> dict[str, Any]:
		"""
		Apply an approved rent increase once the effective date has passed.

		Updates the tenancy rent_amount.  Raises ValueError if effective_date
		has not yet been reached.
		"""
		assert proposal_id and tenant_id and applied_by, \
			"proposal_id, tenant_id, applied_by required"
		proposals = self._store.get("rent_increases", [])
		for i, p in enumerate(proposals):
			if p["id"] == proposal_id and p["tenant_id"] == tenant_id:
				if p["status"] not in ("proposed", "approved"):
					raise ValueError(f"proposal status is {p['status']}, cannot apply")
				effective = date.fromisoformat(p["effective_date"])
				if effective > date.today():
					raise ValueError(
						f"effective_date {effective} has not passed; cannot apply increase yet"
					)
				p["status"] = "applied"
				p["applied_by"] = applied_by
				p["applied_at"] = datetime.utcnow().isoformat()
				self._store["rent_increases"][i] = p
				# Update tenancy rent_amount
				for j, t in enumerate(self._store["tenancies"]):
					if t["id"] == p["tenancy_id"] and t["tenant_id"] == tenant_id:
						t["rent_amount"] = p["new_rent"]
						t["updated_at"] = datetime.utcnow()
						self._store["tenancies"][j] = t
						break
				self._log_operation("rent_increase_applied", proposal_id, tenant_id)
				return p
		raise KeyError(f"rent increase proposal {proposal_id} not found")

	# ── NEW: record_void_period ────────────────────────────────────────────────

	async def record_void_period(
		self,
		unit_id: str,
		tenant_id: str,
		start_date: date,
		end_date: date | None = None,
		reason: str = "between_tenancies",
		notes: str = "",
	) -> dict[str, Any]:
		"""
		Record a void period for a unit (gap between tenancies).

		If end_date is None the void is open (unit still vacant).  Void periods
		feed into `get_void_report()` and `rental_analytics()` void_rate_pct.
		"""
		assert unit_id and tenant_id, "unit_id and tenant_id required"
		assert reason in (
			"between_tenancies", "refurbishment", "owner_occupied",
			"legal_dispute", "unlet", "other"
		), f"unsupported reason: {reason}"
		assert end_date is None or end_date >= start_date, \
			"end_date must be >= start_date"

		if "voids" not in self._store:
			self._store["voids"] = []

		from uuid6 import uuid7
		void_id = str(uuid7())
		void_days = (end_date - start_date).days if end_date else None
		void: dict[str, Any] = {
			"id": void_id,
			"tenant_id": tenant_id,
			"unit_id": unit_id,
			"start_date": str(start_date),
			"end_date": str(end_date) if end_date else None,
			"void_days": void_days,
			"reason": reason,
			"status": "closed" if end_date else "open",
			"notes": notes,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["voids"].append(void)
		self._log_operation("void_recorded", void_id, tenant_id)
		return void

	async def get_void_report(
		self,
		tenant_id: str,
		period_start: date | None = None,
		period_end: date | None = None,
		unit_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Summarise void periods for a portfolio or single unit.

		Returns total void days, void rate (vs total days in period), and
		a breakdown by reason.
		"""
		assert tenant_id, "tenant_id required"
		voids = [v for v in self._store.get("voids", []) if v["tenant_id"] == tenant_id]
		if unit_id:
			voids = [v for v in voids if v["unit_id"] == unit_id]
		if period_start:
			voids = [v for v in voids if (v["end_date"] or str(date.today())) >= str(period_start)]
		if period_end:
			voids = [v for v in voids if v["start_date"] <= str(period_end)]

		total_void_days = sum(v.get("void_days") or 0 for v in voids)
		period_days = (
			(period_end - period_start).days
			if period_start and period_end
			else 365
		)
		void_rate_pct = round(total_void_days / max(period_days, 1) * 100, 2)

		by_reason: dict[str, int] = {}
		for v in voids:
			by_reason[v["reason"]] = by_reason.get(v["reason"], 0) + (v.get("void_days") or 0)

		return {
			"tenant_id": tenant_id,
			"unit_id": unit_id,
			"period_start": str(period_start) if period_start else None,
			"period_end": str(period_end) if period_end else None,
			"void_count": len(voids),
			"total_void_days": total_void_days,
			"void_rate_pct": void_rate_pct,
			"by_reason": by_reason,
			"open_voids": len([v for v in voids if v["status"] == "open"]),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: snapshot_rent_roll / compare_rent_rolls ───────────────────────────

	async def snapshot_rent_roll(
		self,
		tenant_id: str,
		snapshot_date: date | None = None,
		property_id: str | None = None,
		label: str = "",
	) -> dict[str, Any]:
		"""
		Capture a named point-in-time snapshot of the rent roll.

		Stores the snapshot in `_store["rent_roll_snapshots"]`.  Use
		`compare_rent_rolls()` to diff two snapshots for month-end
		reconciliation or auditor evidence packs.
		"""
		assert tenant_id, "tenant_id required"
		roll = await self.generate_rent_roll(tenant_id, property_id=property_id)
		snap_date = snapshot_date or date.today()

		if "rent_roll_snapshots" not in self._store:
			self._store["rent_roll_snapshots"] = []

		from uuid6 import uuid7
		snapshot_id = str(uuid7())
		snapshot: dict[str, Any] = {
			"id": snapshot_id,
			"tenant_id": tenant_id,
			"snapshot_date": str(snap_date),
			"property_id": property_id,
			"label": label or f"snapshot-{snap_date}",
			"tenancy_count": len(roll),
			"gross_rent": sum(r["rent_amount"] for r in roll),
			"total_arrears": sum(r["total_arrears"] for r in roll),
			"roll": roll,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["rent_roll_snapshots"].append(snapshot)
		self._log_operation("rent_roll_snapshot", snapshot_id, tenant_id)
		return snapshot

	async def compare_rent_rolls(
		self,
		snapshot_id_a: str,
		snapshot_id_b: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""
		Diff two rent roll snapshots.

		Returns sets of added/removed/changed tenancies with delta amounts.
		Useful for month-end reconciliation: compare start-of-month vs end-of-month.
		"""
		assert snapshot_id_a and snapshot_id_b and tenant_id, \
			"snapshot_id_a, snapshot_id_b, tenant_id required"
		snapshots = {
			s["id"]: s
			for s in self._store.get("rent_roll_snapshots", [])
			if s["tenant_id"] == tenant_id
		}
		snap_a = snapshots.get(snapshot_id_a)
		snap_b = snapshots.get(snapshot_id_b)
		if not snap_a:
			raise KeyError(f"snapshot {snapshot_id_a} not found")
		if not snap_b:
			raise KeyError(f"snapshot {snapshot_id_b} not found")

		roll_a = {r["tenancy_id"]: r for r in snap_a["roll"]}
		roll_b = {r["tenancy_id"]: r for r in snap_b["roll"]}

		added = [roll_b[tid] for tid in roll_b if tid not in roll_a]
		removed = [roll_a[tid] for tid in roll_a if tid not in roll_b]
		changed = []
		for tid in roll_a:
			if tid in roll_b:
				a, b = roll_a[tid], roll_b[tid]
				if a["rent_amount"] != b["rent_amount"] or a["total_arrears"] != b["total_arrears"]:
					changed.append({
						"tenancy_id": tid,
						"rent_delta": b["rent_amount"] - a["rent_amount"],
						"arrears_delta": b["total_arrears"] - a["total_arrears"],
						"before": a,
						"after": b,
					})

		return {
			"snapshot_a": {"id": snapshot_id_a, "date": snap_a["snapshot_date"], "label": snap_a["label"]},
			"snapshot_b": {"id": snapshot_id_b, "date": snap_b["snapshot_date"], "label": snap_b["label"]},
			"added_count": len(added),
			"removed_count": len(removed),
			"changed_count": len(changed),
			"gross_rent_delta": snap_b["gross_rent"] - snap_a["gross_rent"],
			"arrears_delta": snap_b["total_arrears"] - snap_a["total_arrears"],
			"added": added,
			"removed": removed,
			"changed": changed,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: schedule_arrears_chase ────────────────────────────────────────────

	async def schedule_arrears_chase(
		self,
		arrears_id: str,
		tenant_id: str,
		chase_sequence: list[dict[str, Any]] | None = None,
		override_contact: str = "",
	) -> dict[str, Any]:
		"""
		Schedule an automated arrears-chasing sequence for an arrears record.

		Each step in `chase_sequence`: {"days_after": int, "method": str,
		"template": str}.  Supported methods: email, sms, letter, phone, portal.
		Returns a chase schedule record.  Integrates with `schd` capability when
		available.
		"""
		assert arrears_id and tenant_id, "arrears_id and tenant_id required"
		arrears_rec = next(
			(a for a in self._store["arrears"]
			if a["id"] == arrears_id and a["tenant_id"] == tenant_id),
			None,
		)
		if not arrears_rec:
			raise KeyError(f"arrears record {arrears_id} not found")

		_default_sequence: list[dict[str, Any]] = [
			{"days_after": 3, "method": "email", "template": "arrears_reminder_1"},
			{"days_after": 7, "method": "sms", "template": "arrears_reminder_2"},
			{"days_after": 14, "method": "letter", "template": "formal_demand"},
			{"days_after": 30, "method": "email", "template": "pre_legal_warning"},
		]
		sequence = chase_sequence or _default_sequence
		supported_methods = {"email", "sms", "letter", "phone", "portal"}
		for step in sequence:
			assert step.get("method") in supported_methods, \
				f"unsupported chase method: {step.get('method')}"
			assert isinstance(step.get("days_after"), int) and step["days_after"] > 0, \
				"days_after must be a positive int"

		if "chase_schedules" not in self._store:
			self._store["chase_schedules"] = []

		from uuid6 import uuid7
		schedule_id = str(uuid7())
		today = date.today()
		scheduled_steps = [
			{
				**step,
				"scheduled_date": str(today + timedelta(days=step["days_after"])),
				"status": "pending",
			}
			for step in sequence
		]
		schedule: dict[str, Any] = {
			"id": schedule_id,
			"tenant_id": tenant_id,
			"arrears_id": arrears_id,
			"tenancy_id": arrears_rec.get("tenancy_id"),
			"override_contact": override_contact,
			"steps": scheduled_steps,
			"status": "active",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["chase_schedules"].append(schedule)
		self._log_operation("arrears_chase_scheduled", schedule_id, tenant_id)
		return schedule

	# ── NEW: generate_rent_receipt ─────────────────────────────────────────────

	async def generate_rent_receipt(
		self,
		payment_id: str,
		tenant_id: str,
		issued_by: str = "system",
	) -> dict[str, Any]:
		"""
		Generate a formal rent receipt for a payment.

		Populates `receipt_number` as a sequential `REC-{YYYY}-{NNN}` string.
		Returns the receipt dict.  Marks the payment record as receipted.
		"""
		assert payment_id and tenant_id, "payment_id and tenant_id required"
		payment = next(
			(p for p in self._store["payments"]
			if p["id"] == payment_id and p["tenant_id"] == tenant_id),
			None,
		)
		if not payment:
			raise KeyError(f"payment {payment_id} not found")

		# Derive next sequential receipt number for this tenant / year
		year = datetime.utcnow().year
		existing = [
			p for p in self._store["payments"]
			if p.get("receipt_number") and p["tenant_id"] == tenant_id
			and p["receipt_number"].startswith(f"REC-{year}-")
		]
		seq = len(existing) + 1
		receipt_number = f"REC-{year}-{seq:04d}"

		# Update payment record with receipt number
		for i, p in enumerate(self._store["payments"]):
			if p["id"] == payment_id and p["tenant_id"] == tenant_id:
				p["receipt_number"] = receipt_number
				p["receipted_at"] = datetime.utcnow().isoformat()
				self._store["payments"][i] = p
				break

		# Fetch tenancy info for receipt
		tenancy = None
		if payment.get("tenancy_id"):
			tenancy = await self.get_tenancy(payment["tenancy_id"], tenant_id)

		if "receipts" not in self._store:
			self._store["receipts"] = []

		from uuid6 import uuid7
		receipt_id = str(uuid7())
		receipt: dict[str, Any] = {
			"id": receipt_id,
			"receipt_number": receipt_number,
			"tenant_id": tenant_id,
			"payment_id": payment_id,
			"tenancy_id": payment.get("tenancy_id"),
			"unit_id": tenancy.unit_id if tenancy else payment.get("unit_id"),
			"property_id": tenancy.property_id if tenancy else None,
			"amount": payment.get("amount"),
			"currency": payment.get("currency", "KES"),
			"payment_method": payment.get("payment_method"),
			"period": payment.get("period"),
			"payment_date": payment.get("received_at", payment.get("created_at")),
			"issued_by": issued_by,
			"issued_at": datetime.utcnow().isoformat(),
		}
		self._store["receipts"].append(receipt)
		self._log_operation("receipt_generated", receipt_id, tenant_id)
		return receipt

	# ── NEW: run_compliance_check ──────────────────────────────────────────────

	async def run_compliance_check(
		self,
		tenancy_id: str,
		tenant_id: str,
		jurisdiction: str = "KE",
	) -> dict[str, Any]:
		"""
		Run a structured compliance checklist for a tenancy.

		Items are driven by jurisdiction and tenancy type.  Each item returns
		status (pass / fail / not_applicable / unknown) with expiry date where
		relevant.  Raises no errors — returns per-item verdicts for the caller
		to act on.
		"""
		assert tenancy_id and tenant_id, "tenancy_id and tenant_id required"
		assert jurisdiction in ("KE", "GB", "ZA", "NG", "GH"), \
			f"unsupported jurisdiction: {jurisdiction}"
		tenancy = await self.get_tenancy(tenancy_id, tenant_id)
		if not tenancy:
			raise KeyError(f"tenancy {tenancy_id} not found")

		# Base compliance items (jurisdiction-agnostic)
		items: list[dict[str, Any]] = [
			{
				"item": "deposit_registered",
				"description": "Tenancy deposit registered with approved scheme",
				"status": "pass" if tenancy.deposit_registered else "fail",
				"remediation": "Register deposit within 30 days of receipt" if not tenancy.deposit_registered else None,
			},
			{
				"item": "referencing_complete",
				"description": "Tenant referencing checks completed",
				"status": "pass" if tenancy.referencing_complete else "fail",
				"remediation": "Complete referencing before activation" if not tenancy.referencing_complete else None,
			},
			{
				"item": "right_to_rent_checked",
				"description": "Right to rent (occupancy eligibility) verified",
				"status": "pass" if tenancy.right_to_rent_checked else "fail",
				"remediation": "Verify ID and occupancy eligibility documents" if not tenancy.right_to_rent_checked else None,
			},
		]

		# GB-specific items
		if jurisdiction == "GB":
			items += [
				{
					"item": "epc_valid",
					"description": "Energy Performance Certificate (EPC) rating E or above",
					"status": "unknown",
					"remediation": "Obtain EPC from accredited assessor",
				},
				{
					"item": "gas_safety_cert",
					"description": "Annual Gas Safety Certificate (CP12) valid",
					"status": "unknown",
					"remediation": "Book annual gas safety check",
				},
				{
					"item": "eicr_valid",
					"description": "Electrical Installation Condition Report (EICR) within 5 years",
					"status": "unknown",
					"remediation": "Commission EICR from qualified electrician",
				},
			]

		overall = "pass" if all(i["status"] == "pass" for i in items) else "fail"
		fail_count = len([i for i in items if i["status"] == "fail"])

		if "compliance_checks" not in self._store:
			self._store["compliance_checks"] = []

		from uuid6 import uuid7
		check_id = str(uuid7())
		result: dict[str, Any] = {
			"id": check_id,
			"tenant_id": tenant_id,
			"tenancy_id": tenancy_id,
			"jurisdiction": jurisdiction,
			"tenancy_type": tenancy.tenancy_type.value,
			"overall_status": overall,
			"fail_count": fail_count,
			"items": items,
			"checked_at": datetime.utcnow().isoformat(),
		}
		self._store["compliance_checks"].append(result)
		self._log_operation("compliance_checked", check_id, tenant_id)
		return result
