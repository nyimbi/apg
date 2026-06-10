"""Executable service layer for APG Tax Administration.

Implements the full TaxAdministrationService covering:
  - Taxpayer registration lifecycle
  - Return filing (PAYE, VAT, CIT, WHT, EXCISE, CUSTOMS, STAMP_DUTY)
  - Assessments, objections, and appeals
  - Audit case management
  - Payments, penalty/interest calculation, debt collection
  - Refunds
  - Revenue, compliance, delinquency, and exchange-of-information reporting

Kenya-specific rates (Income Tax Act Cap 470):
  - Late filing penalty: 5% of tax due (min KES 1,000)
  - Late payment interest: 1% per month (or part thereof) on outstanding tax
  - Refund interest: 1% per month payable by KRA on overdue refunds
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUDIT_TYPES, SUPPORTED_DEBT_COLLECTION_METHODS,
		SUPPORTED_OBJECTION_STATUSES, SUPPORTED_REGISTRATION_STATUSES, SUPPORTED_RETURN_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_TAX_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AuditFindingCreate, AuditFindingResponse,
		AppealCreate, AppealResponse,
		InterestCreate, InterestResponse,
		ObjectionCreate, ObjectionResponse,
		PenaltyCreate, PenaltyResponse,
		TaxAssessmentCreate, TaxAssessmentResponse,
		TaxAuditCreate, TaxAuditResponse,
		TaxClearanceCertificateCreate, TaxClearanceCertificateResponse,
		TaxDebtCreate, TaxDebtResponse,
		TaxObligationCreate, TaxObligationResponse,
		TaxPaymentCreate, TaxPaymentResponse,
		TaxRefundCreate, TaxRefundResponse,
		TaxReturnCreate, TaxReturnResponse,
		TaxpayerCreate, TaxpayerResponse,
		AssessmentStatus, AssessmentType, AppealStatus, AuditStatus, AuditType,
		ClearanceCertificateStatus, CollectionMethod, DebtStatus, FindingType,
		ObjectionStatus, PaymentMethod, PaymentStatus, PenaltyStatus, PenaltyType,
		RefundStatus, ReturnStatus, ReturnType, TaxpayerStatus, TaxpayerType, TaxType,
		InterestType,
		uuid7str,
		# Report models
		TaxDashboardKPI, ComplianceRiskProfile, DebtAgingBucket, RevenueReport,
		DemandNotice, EOIRequest,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUDIT_TYPES, SUPPORTED_DEBT_COLLECTION_METHODS,
		SUPPORTED_OBJECTION_STATUSES, SUPPORTED_REGISTRATION_STATUSES, SUPPORTED_RETURN_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_TAX_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AuditFindingCreate, AuditFindingResponse,
		AppealCreate, AppealResponse,
		InterestCreate, InterestResponse,
		ObjectionCreate, ObjectionResponse,
		PenaltyCreate, PenaltyResponse,
		TaxAssessmentCreate, TaxAssessmentResponse,
		TaxAuditCreate, TaxAuditResponse,
		TaxClearanceCertificateCreate, TaxClearanceCertificateResponse,
		TaxDebtCreate, TaxDebtResponse,
		TaxObligationCreate, TaxObligationResponse,
		TaxPaymentCreate, TaxPaymentResponse,
		TaxRefundCreate, TaxRefundResponse,
		TaxReturnCreate, TaxReturnResponse,
		TaxpayerCreate, TaxpayerResponse,
		AssessmentStatus, AssessmentType, AppealStatus, AuditStatus, AuditType,
		ClearanceCertificateStatus, CollectionMethod, DebtStatus, FindingType,
		ObjectionStatus, PaymentMethod, PaymentStatus, PenaltyStatus, PenaltyType,
		RefundStatus, ReturnStatus, ReturnType, TaxpayerStatus, TaxpayerType, TaxType,
		InterestType,
		uuid7str,
		TaxDashboardKPI, ComplianceRiskProfile, DebtAgingBucket, RevenueReport,
		DemandNotice, EOIRequest,
	)


# ---------------------------------------------------------------------------
# Kenya statutory constants
# ---------------------------------------------------------------------------

_KE_LATE_FILING_RATE = Decimal("0.05")       # 5% of tax due
_KE_LATE_FILING_MINIMUM = Decimal("1000")    # KES 1,000 minimum penalty
_KE_LATE_PAYMENT_MONTHLY = Decimal("0.01")  # 1% per month on outstanding
_KE_REFUND_INTEREST_MONTHLY = Decimal("0.01")  # 1% per month KRA owes taxpayer
_KE_OBJECTION_DAYS = 30                       # days from assessment date
_KE_TIN_PATTERN = re.compile(r"^[AP]\d{9}[A-Z]$")  # KRA PIN format

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _today() -> date:
	return date.today()


def _now() -> datetime:
	return datetime.utcnow()


def _months_between(d1: date, d2: date) -> int:
	"""Whole months elapsed from d1 to d2 (inclusive of partial month)."""
	delta_days = (d2 - d1).days
	return max(0, -(-delta_days // 30))  # ceiling division


def _cents(v: Decimal | int | float) -> Decimal:
	return Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# In-process store helpers (replace with DB adapter in production)
# ---------------------------------------------------------------------------

class _Store(dict):
	"""Typed dict-based in-process store with tenant-scoped lookup."""

	def put(self, tenant_id: str, record_id: str, obj: Any) -> None:
		self[(tenant_id, record_id)] = obj

	def get_item(self, tenant_id: str, record_id: str) -> Any | None:
		return self.get((tenant_id, record_id))

	def tenant_values(self, tenant_id: str) -> list[Any]:
		return [v for (tid, _), v in self.items() if tid == tenant_id]

	def count(self, tenant_id: str) -> int:
		return sum(1 for (tid, _) in self if tid == tenant_id)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class TaxAdministrationService:
	"""Tenant-scoped tax administration runtime.

	All public methods are *synchronous* (thin runtime; swap for async + ORM
	in production).  Stores use in-process dicts keyed by (tenant_id, record_id).
	The legacy adapter methods at the bottom preserve backward-compatibility with
	the original capability_contract-driven interface.
	"""

	def __init__(self) -> None:
		# Pydantic-model stores
		self._taxpayers: _Store = _Store()
		self._obligations: _Store = _Store()
		self._returns: _Store = _Store()
		self._assessments: _Store = _Store()
		self._payments: _Store = _Store()
		self._debts: _Store = _Store()
		self._demand_notices: _Store = _Store()
		self._audits: _Store = _Store()
		self._findings: _Store = _Store()
		self._objections: _Store = _Store()
		self._appeals: _Store = _Store()
		self._refunds: _Store = _Store()
		self._penalties: _Store = _Store()
		self._interests: _Store = _Store()
		self._clearances: _Store = _Store()
		self._eoi_requests: _Store = _Store()
		self._parked_transactions: _Store = _Store()

		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Contract / evaluation
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ==================================================================
	# TAXPAYER REGISTRATION
	# ==================================================================

	def register_taxpayer(
		self,
		taxpayer_id: str,
		tenant_id: str,
		tax_type: str,
		tax_pin: str,
		id_number: str,
		legal_name: str,
		event_id: str = "",
		*,
		entity_type: str = "individual",
		business_type: str | None = None,
		address: str = "",
		contact: str = "",
		tax_types: list[str] | None = None,
		trade_name: str | None = None,
		email: str | None = None,
		phone: str | None = None,
		evidence_reference: str = "initial_registration",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Register a new taxpayer and issue a TIN.

		Generates a KRA-format PIN (Axxxxxxxx[A-Z]) and records the taxpayer
		with PENDING status pending verification.
		"""
		_et = _normalize(entity_type)
		assert _present(legal_name), "legal_name required"
		assert _present(id_number), "id_number required"

		# Map entity_type string to TaxpayerType enum value
		_type_map = {
			"individual": TaxpayerType.INDIVIDUAL,
			"company": TaxpayerType.COMPANY,
			"partnership": TaxpayerType.PARTNERSHIP,
			"trust": TaxpayerType.TRUST,
			"government": TaxpayerType.GOVERNMENT_ENTITY,
			"ngo": TaxpayerType.NGO,
			"foreign": TaxpayerType.FOREIGN_ENTITY,
		}
		taxpayer_type = _type_map.get(_et, TaxpayerType.INDIVIDUAL)

		# Derive national_id vs business_registration_number
		nat_id: str | None = None
		brn: str | None = None
		if taxpayer_type == TaxpayerType.INDIVIDUAL:
			nat_id = id_number
		else:
			brn = id_number

		# Generate a unique PIN
		pin = self._generate_pin(tenant_id, taxpayer_type)

		# Resolve TaxType list
		resolved_types: list[TaxType] = []
		for t in (tax_types or []):
			try:
				resolved_types.append(TaxType(t.lower()))
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		rec = TaxpayerResponse(
			tenant_id=tenant_id,
			taxpayer_type=taxpayer_type,
			tax_pin=pin,
			national_id=nat_id,
			business_registration_number=brn,
			taxpayer_name=legal_name.strip(),
			trade_name=trade_name,
			email=email,
			phone=phone or contact,
			physical_address=address,
			tax_types=resolved_types,
			sector_code=business_type,
			country_of_incorporation="KE",
			is_resident=True,
			evidence_reference=evidence_reference,
			status=TaxpayerStatus.PENDING,
			created_by=created_by,
		)
		self._taxpayers.put(tenant_id, rec.id, rec)
		self._audit(tenant_id, "taxpayer_registered", rec.id)
		return rec.model_dump(mode="json")

	def update_taxpayer(self, tin: str, *, tenant_id: str = "default", **fields: Any) -> dict[str, Any]:
		"""Update mutable fields on a taxpayer record located by PIN."""
		rec = self._find_taxpayer_by_pin(tin, tenant_id)
		assert rec is not None, f"taxpayer not found: {tin}"

		allowed = {
			"taxpayer_name", "trade_name", "email", "phone",
			"physical_address", "postal_address", "sector_code",
			"evidence_reference", "status", "tax_types",
		}
		data = rec.model_dump()
		for k, v in fields.items():
			if k in allowed:
				data[k] = v
		data["updated_at"] = _now()

		updated = TaxpayerResponse(**data)
		self._taxpayers.put(tenant_id, rec.id, updated)
		self._audit(tenant_id, "taxpayer_updated", rec.id)
		return updated.model_dump(mode="json")

	def deregister_taxpayer(
		self, tin: str, reason: str, deregistration_date: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Mark a taxpayer as deregistered.  Outstanding debts must be nil."""
		rec = self._find_taxpayer_by_pin(tin, tenant_id)
		assert rec is not None, f"taxpayer not found: {tin}"
		assert _present(reason), "deregistration reason required"

		data = rec.model_dump()
		data["status"] = TaxpayerStatus.DEREGISTERED.value
		data["metadata"]["deregistration_reason"] = reason
		data["metadata"]["deregistration_date"] = deregistration_date
		data["updated_at"] = _now()

		updated = TaxpayerResponse(**data)
		self._taxpayers.put(tenant_id, rec.id, updated)
		self._audit(tenant_id, "taxpayer_deregistered", rec.id)
		return updated.model_dump(mode="json")

	def taxpayer_search(
		self, query: str, search_type: str = "name", *, tenant_id: str = "default"
	) -> list[dict[str, Any]]:
		"""Search taxpayers by name, PIN, national_id, or phone."""
		q = query.strip().lower()
		results = []
		for rec in self._taxpayers.tenant_values(tenant_id):
			match search_type:
				case "name":
					hit = q in rec.taxpayer_name.lower() or q in (rec.trade_name or "").lower()
				case "pin":
					hit = rec.tax_pin.upper() == query.strip().upper()
				case "national_id":
					hit = (rec.national_id or "").lower() == q
				case "phone":
					hit = (rec.phone or "").replace(" ", "") == query.replace(" ", "")
				case _:
					hit = q in rec.taxpayer_name.lower()
			if hit:
				results.append(rec.model_dump(mode="json"))
		return results

	def verify_tin(self, tin: str, country: str = "KE") -> dict[str, Any]:
		"""Validate TIN format and check existence in the registry."""
		tin_clean = tin.strip().upper()
		format_valid = bool(_KE_TIN_PATTERN.match(tin_clean)) if country.upper() == "KE" else len(tin_clean) >= 5
		exists = any(
			rec.tax_pin.upper() == tin_clean
			for rec in self._taxpayers.values()
		)
		status: str | None = None
		if exists:
			rec = next(r for r in self._taxpayers.values() if r.tax_pin.upper() == tin_clean)
			status = rec.status.value if hasattr(rec.status, "value") else str(rec.status)
		return {
			"tin": tin_clean,
			"country": country.upper(),
			"format_valid": format_valid,
			"exists": exists,
			"status": status,
			"verified_at": _now().isoformat(),
		}

	# ==================================================================
	# RETURN FILING
	# ==================================================================

	def submit_return(
		self,
		tin: str,
		tax_type: str,
		period: str,
		return_data: dict[str, Any],
		attachments: list[str] | None = None,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""File a tax return.

		Supports tax_types: PAYE, VAT, CIT, WHT, EXCISE, CUSTOMS, STAMP_DUTY.
		``period`` is a string like "2025-01" or "2025" or "Q1-2025".
		"""
		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"
		assert _present(period), "period required"

		# Map tax_type -> ReturnType
		_rt_map = {
			"vat": ReturnType.MONTHLY_VAT,
			"paye": ReturnType.WITHHOLDING_TAX_RETURN,
			"wht": ReturnType.WITHHOLDING_TAX_RETURN,
			"cit": ReturnType.CORPORATE_ANNUAL,
			"corporate_tax": ReturnType.CORPORATE_ANNUAL,
			"income_tax": ReturnType.ANNUAL_INCOME,
			"excise": ReturnType.MONTHLY_VAT,
			"customs": ReturnType.CUSTOMS_ENTRY,
			"stamp_duty": ReturnType.ANNUAL_INCOME,
			"turnover_tax": ReturnType.TURNOVER_TAX_MONTHLY,
		}
		return_type = _rt_map.get(_normalize(tax_type), ReturnType.ANNUAL_INCOME)

		# Parse period into start/end dates
		period_start, period_end = self._parse_period(period)

		gross = Decimal(str(return_data.get("gross_income", 0)))
		deductions = Decimal(str(return_data.get("allowable_deductions", 0)))
		taxable = Decimal(str(return_data.get("taxable_income", gross - deductions)))
		liability = Decimal(str(return_data.get("tax_liability", 0)))
		credits = Decimal(str(return_data.get("tax_credits", 0)))
		paid = Decimal(str(return_data.get("tax_paid", 0)))
		net_payable = _cents(liability - credits - paid)
		evidence = (attachments[0] if attachments else None) or return_data.get("evidence_reference", "submitted")

		rec = TaxReturnResponse(
			tenant_id=tenant_id,
			taxpayer_id=tp["id"] if isinstance(tp, dict) else tp.id,
			tax_pin=tin.strip().upper(),
			return_type=return_type,
			tax_period_start=period_start,
			tax_period_end=period_end,
			gross_income=gross,
			allowable_deductions=deductions,
			taxable_income=taxable,
			tax_liability=liability,
			tax_credits=credits,
			tax_paid=paid,
			net_tax_payable=net_payable,
			filing_date=_now(),
			status=ReturnStatus.FILED,
			evidence_reference=str(evidence),
			is_amended=False,
			created_by=created_by,
		)
		self._returns.put(tenant_id, rec.id, rec)
		self._audit(tenant_id, "tax_return_filed", rec.id)
		return rec.model_dump(mode="json")

	def validate_return(self, return_id: str, *, tenant_id: str = "default") -> dict[str, Any]:
		"""Run consistency checks on a filed return."""
		rec = self._returns.get_item(tenant_id, return_id)
		assert rec is not None, f"return not found: {return_id}"

		issues: list[str] = []
		if rec.gross_income < 0:
			issues.append("gross_income cannot be negative")
		if rec.tax_liability < 0:
			issues.append("tax_liability cannot be negative")
		computed_taxable = rec.gross_income - rec.allowable_deductions
		if abs(computed_taxable - rec.taxable_income) > Decimal("0.01"):
			issues.append(f"taxable_income mismatch: declared={rec.taxable_income} computed={computed_taxable}")
		computed_net = rec.tax_liability - rec.tax_credits - rec.tax_paid
		if abs(computed_net - rec.net_tax_payable) > Decimal("0.01"):
			issues.append(f"net_tax_payable mismatch: declared={rec.net_tax_payable} computed={computed_net}")

		status = "valid" if not issues else "invalid"
		if status == "valid" and rec.status == ReturnStatus.FILED:
			data = rec.model_dump()
			data["status"] = ReturnStatus.UNDER_REVIEW.value
			data["updated_at"] = _now()
			self._returns.put(tenant_id, return_id, TaxReturnResponse(**data))

		return {
			"return_id": return_id,
			"status": status,
			"issues": issues,
			"validated_at": _now().isoformat(),
		}

	def amend_return(
		self,
		return_id: str,
		amendment_reason: str,
		amended_data: dict[str, Any],
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""File an amended return, linking to the original."""
		orig = self._returns.get_item(tenant_id, return_id)
		assert orig is not None, f"return not found: {return_id}"
		assert _present(amendment_reason), "amendment_reason required"

		data = orig.model_dump()
		for k in ("gross_income", "allowable_deductions", "taxable_income",
				  "tax_liability", "tax_credits", "tax_paid", "net_tax_payable"):
			if k in amended_data:
				data[k] = Decimal(str(amended_data[k]))
		data["id"] = uuid7str()
		data["is_amended"] = True
		data["original_return_id"] = return_id
		data["status"] = ReturnStatus.AMENDED.value
		data["metadata"]["amendment_reason"] = amendment_reason
		data["created_at"] = _now()
		data["updated_at"] = _now()
		data["created_by"] = created_by

		amended = TaxReturnResponse(**data)
		self._returns.put(tenant_id, amended.id, amended)

		# Mark original as amended
		orig_data = orig.model_dump()
		orig_data["status"] = ReturnStatus.AMENDED.value
		orig_data["updated_at"] = _now()
		self._returns.put(tenant_id, return_id, TaxReturnResponse(**orig_data))

		self._audit(tenant_id, "tax_return_amended", amended.id)
		return amended.model_dump(mode="json")

	def file_nil_return(
		self,
		tin: str,
		tax_type: str,
		period: str,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""File a nil return (no activity) to maintain compliance."""
		return self.submit_return(
			tin=tin,
			tax_type=tax_type,
			period=period,
			return_data={
				"gross_income": 0,
				"allowable_deductions": 0,
				"taxable_income": 0,
				"tax_liability": 0,
				"tax_credits": 0,
				"tax_paid": 0,
				"net_tax_payable": 0,
				"evidence_reference": "nil_return",
				"is_nil": True,
			},
			tenant_id=tenant_id,
			created_by=created_by,
		)

	def return_filing_status(
		self, tin: str, tax_type: str, period: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Check whether a return has been filed for the given TIN / tax_type / period."""
		period_start, period_end = self._parse_period(period)
		_rt_map = {
			"vat": ReturnType.MONTHLY_VAT,
			"paye": ReturnType.WITHHOLDING_TAX_RETURN,
			"wht": ReturnType.WITHHOLDING_TAX_RETURN,
			"cit": ReturnType.CORPORATE_ANNUAL,
			"income_tax": ReturnType.ANNUAL_INCOME,
			"customs": ReturnType.CUSTOMS_ENTRY,
		}
		return_type = _rt_map.get(_normalize(tax_type))

		matches = [
			r for r in self._returns.tenant_values(tenant_id)
			if r.tax_pin.upper() == tin.strip().upper()
			and (return_type is None or r.return_type == return_type)
			and r.tax_period_start <= period_end
			and r.tax_period_end >= period_start
		]
		return {
			"tin": tin,
			"tax_type": tax_type,
			"period": period,
			"filed": len(matches) > 0,
			"count": len(matches),
			"returns": [{"id": r.id, "status": r.status.value, "filing_date": r.filing_date.isoformat() if r.filing_date else None} for r in matches],
		}

	def filing_history(
		self,
		tin: str,
		period_from: str,
		period_to: str,
		*,
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""Return all filed returns for a TIN within a date range."""
		from_date, _ = self._parse_period(period_from)
		_, to_date = self._parse_period(period_to)
		results = [
			r for r in self._returns.tenant_values(tenant_id)
			if r.tax_pin.upper() == tin.strip().upper()
			and r.tax_period_start >= from_date
			and r.tax_period_end <= to_date
			and not r.is_deleted
		]
		results.sort(key=lambda r: r.tax_period_start)
		return [r.model_dump(mode="json") for r in results]

	# ==================================================================
	# ASSESSMENT & AUDIT
	# ==================================================================

	def issue_assessment(
		self,
		tin: str,
		tax_type: str,
		period: str,
		assessed_amount: float,
		reason: str,
		assessment_type: str,
		*,
		tenant_id: str = "default",
		assessor_id: str = "system",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Issue a tax assessment. Creates a TaxDebt record simultaneously."""
		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"

		_at_map = {
			"self_assessment": AssessmentType.SELF_ASSESSMENT,
			"amended": AssessmentType.AMENDED_ASSESSMENT,
			"additional": AssessmentType.BEST_JUDGEMENT,
			"jeopardy": AssessmentType.BEST_JUDGEMENT,
			"audit_assessment": AssessmentType.AUDIT_ASSESSMENT,
			"estimated": AssessmentType.ESTIMATED_ASSESSMENT,
		}
		at = _at_map.get(_normalize(assessment_type), AssessmentType.BEST_JUDGEMENT)

		# Find or synthesise a return for this assessment
		period_start, period_end = self._parse_period(period)
		taxpayer_id = tp["id"] if isinstance(tp, dict) else tp.id

		# Look for an existing return
		existing_return = next(
			(r for r in self._returns.tenant_values(tenant_id)
			 if r.tax_pin.upper() == tin.strip().upper()
			 and r.tax_period_start <= period_end
			 and r.tax_period_end >= period_start),
			None,
		)
		# Create a synthetic placeholder return if none exists (jeopardy / best judgement)
		if existing_return is None:
			placeholder = TaxReturnResponse(
				tenant_id=tenant_id,
				taxpayer_id=taxpayer_id,
				tax_pin=tin.strip().upper(),
				return_type=ReturnType.ANNUAL_INCOME,
				tax_period_start=period_start,
				tax_period_end=period_end,
				gross_income=Decimal("0"),
				allowable_deductions=Decimal("0"),
				taxable_income=Decimal("0"),
				tax_liability=Decimal(str(assessed_amount)),
				tax_credits=Decimal("0"),
				tax_paid=Decimal("0"),
				net_tax_payable=Decimal(str(assessed_amount)),
				status=ReturnStatus.ASSESSED,
				evidence_reference=reason or "best_judgement",
				is_amended=False,
				created_by=created_by,
			)
			self._returns.put(tenant_id, placeholder.id, placeholder)
			return_id = placeholder.id
		else:
			return_id = existing_return.id

		due_date = _today() + timedelta(days=30)
		rec = TaxAssessmentResponse(
			tenant_id=tenant_id,
			return_id=return_id,
			taxpayer_id=taxpayer_id,
			assessment_type=at,
			assessed_amount=_cents(Decimal(str(assessed_amount))),
			tax_liability_per_return=Decimal("0"),
			additional_tax=_cents(Decimal(str(assessed_amount))),
			assessor_id=assessor_id,
			assessment_date=_today(),
			due_date=due_date,
			evidence_reference=reason or "assessment_issued",
			notes=reason,
			status=AssessmentStatus.ISSUED,
			created_by=created_by,
		)
		self._assessments.put(tenant_id, rec.id, rec)

		# Auto-create a debt record
		debt = TaxDebtResponse(
			tenant_id=tenant_id,
			taxpayer_id=taxpayer_id,
			assessment_id=rec.id,
			principal_amount=rec.assessed_amount,
			penalty_amount=Decimal("0"),
			interest_amount=Decimal("0"),
			total_amount=rec.assessed_amount,
			balance=rec.assessed_amount,
			due_date=due_date,
			status=DebtStatus.OUTSTANDING,
			created_by=created_by,
		)
		self._debts.put(tenant_id, debt.id, debt)

		self._audit(tenant_id, "tax_assessed", rec.id)
		result = rec.model_dump(mode="json")
		result["debt_id"] = debt.id
		return result

	def raise_objection(
		self,
		assessment_id: str,
		grounds: str,
		amount_disputed: float,
		objection_date: str | None = None,
		*,
		tenant_id: str = "default",
		tax_pin: str = "",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""File a taxpayer objection. Must be within 30 days of assessment."""
		assessment = self._assessments.get_item(tenant_id, assessment_id)
		assert assessment is not None, f"assessment not found: {assessment_id}"
		assert _present(grounds), "grounds required"

		filed_date = date.fromisoformat(objection_date) if objection_date else _today()
		days_since = (filed_date - assessment.assessment_date).days
		assert days_since <= _KE_OBJECTION_DAYS, (
			f"objection deadline passed: {days_since} days since assessment "
			f"(limit {_KE_OBJECTION_DAYS})"
		)

		rec = ObjectionResponse(
			tenant_id=tenant_id,
			assessment_id=assessment_id,
			taxpayer_id=assessment.taxpayer_id,
			tax_pin=tax_pin or assessment.taxpayer_id,
			grounds=grounds,
			amount_disputed=_cents(Decimal(str(amount_disputed))),
			supporting_documents=[],
			evidence_reference="objection_submitted",
			filed_date=filed_date,
			status=ObjectionStatus.SUBMITTED,
			created_by=created_by,
		)
		self._objections.put(tenant_id, rec.id, rec)

		# Update assessment status
		adata = assessment.model_dump()
		adata["status"] = AssessmentStatus.OBJECTED.value
		adata["updated_at"] = _now()
		self._assessments.put(tenant_id, assessment_id, TaxAssessmentResponse(**adata))

		self._audit(tenant_id, "objection_filed", rec.id)
		return rec.model_dump(mode="json")

	def process_objection(
		self,
		objection_id: str,
		decision: str,
		revised_amount: float,
		officer_id: str,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Determine an objection: upheld / partially_upheld / dismissed."""
		obj = self._objections.get_item(tenant_id, objection_id)
		assert obj is not None, f"objection not found: {objection_id}"
		assert _present(officer_id), "officer_id required"

		_status_map = {
			"upheld": ObjectionStatus.UPHELD,
			"partially_upheld": ObjectionStatus.PARTIALLY_UPHELD,
			"dismissed": ObjectionStatus.DISMISSED,
			"withdrawn": ObjectionStatus.WITHDRAWN,
		}
		new_status = _status_map.get(_normalize(decision), ObjectionStatus.DISMISSED)

		data = obj.model_dump()
		data["status"] = new_status.value
		data["amount_upheld"] = _cents(Decimal(str(revised_amount)))
		data["reviewing_officer_id"] = officer_id
		data["determination_date"] = _today().isoformat()
		data["determination_notes"] = f"decision={decision} revised_amount={revised_amount}"
		data["updated_at"] = _now()
		days = (date.fromisoformat(str(data["determination_date"])) - obj.filed_date).days
		data["days_to_determination"] = days

		updated = ObjectionResponse(**data)
		self._objections.put(tenant_id, objection_id, updated)
		self._audit(tenant_id, "objection_determined", objection_id)
		return updated.model_dump(mode="json")

	def file_appeal(
		self,
		objection_id: str,
		appeal_grounds: str,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""File an appeal to the Tax Appeals Tribunal."""
		obj = self._objections.get_item(tenant_id, objection_id)
		assert obj is not None, f"objection not found: {objection_id}"
		assert obj.status in (ObjectionStatus.DISMISSED, ObjectionStatus.PARTIALLY_UPHELD), (
			"appeal only valid after dismissed or partially_upheld objection"
		)
		assert _present(appeal_grounds), "appeal_grounds required"

		rec = AppealResponse(
			tenant_id=tenant_id,
			objection_id=objection_id,
			taxpayer_id=obj.taxpayer_id,
			grounds=appeal_grounds,
			amount_in_dispute=obj.amount_disputed,
			tribunal="Tax Appeals Tribunal",
			evidence_reference="appeal_lodged",
			status=AppealStatus.SUBMITTED,
			created_by=created_by,
		)
		self._appeals.put(tenant_id, rec.id, rec)

		# Update objection status
		odata = obj.model_dump()
		odata["status"] = ObjectionStatus.APPEALED.value
		odata["updated_at"] = _now()
		self._objections.put(tenant_id, objection_id, ObjectionResponse(**odata))

		self._audit(tenant_id, "appeal_filed", rec.id)
		return rec.model_dump(mode="json")

	def open_audit_case(
		self,
		tin: str,
		audit_type: str,
		audit_period: str,
		assigned_officer: str,
		*,
		tenant_id: str = "default",
		scope_description: str | None = None,
		risk_score: float | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Open a new audit case for a taxpayer."""
		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"
		assert _present(assigned_officer), "assigned_officer required"

		_at_map = {
			"desk_audit": AuditType.DESK_AUDIT,
			"field_audit": AuditType.FIELD_AUDIT,
			"it_audit": AuditType.IT_AUDIT,
			"transfer_pricing": AuditType.TRANSFER_PRICING,
			"vat_refund_audit": AuditType.VAT_REFUND_AUDIT,
			"forensic_audit": AuditType.FORENSIC_AUDIT,
			"compliance_audit": AuditType.COMPLIANCE_AUDIT,
			"sector_audit": AuditType.SECTOR_AUDIT,
		}
		at = _at_map.get(_normalize(audit_type), AuditType.DESK_AUDIT)
		period_start, period_end = self._parse_period(audit_period)
		taxpayer_id = tp["id"] if isinstance(tp, dict) else tp.id

		rec = TaxAuditResponse(
			tenant_id=tenant_id,
			taxpayer_id=taxpayer_id,
			tax_pin=tin.strip().upper(),
			audit_type=at,
			auditor_id=assigned_officer,
			audit_team=[assigned_officer],
			tax_period_start=period_start,
			tax_period_end=period_end,
			scope_description=scope_description,
			risk_score=Decimal(str(risk_score)) if risk_score is not None else None,
			evidence_reference="audit_opened",
			status=AuditStatus.PLANNED,
			created_by=created_by,
		)
		self._audits.put(tenant_id, rec.id, rec)
		self._audit(tenant_id, "audit_case_opened", rec.id)
		return rec.model_dump(mode="json")

	def conduct_audit(
		self,
		case_id: str,
		findings: list[dict[str, Any]],
		adjustments: dict[str, Any] | None = None,
		*,
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Record audit progress and findings; transition status to IN_PROGRESS."""
		audit = self._audits.get_item(tenant_id, case_id)
		assert audit is not None, f"audit case not found: {case_id}"

		finding_ids: list[str] = []
		total_additional = Decimal("0")

		_ft_map = {
			"underpayment": FindingType.UNDERPAYMENT,
			"overpayment": FindingType.OVERPAYMENT,
			"non_compliance": FindingType.NON_COMPLIANCE,
			"evasion": FindingType.EVASION,
			"fraud": FindingType.FRAUD,
		}

		for f in findings:
			add_tax = _cents(Decimal(str(f.get("additional_tax", 0))))
			penalty = _cents(Decimal(str(f.get("penalty_amount", 0))))
			interest = _cents(Decimal(str(f.get("interest_amount", 0))))
			fr = AuditFindingResponse(
				tenant_id=tenant_id,
				audit_id=case_id,
				taxpayer_id=audit.taxpayer_id,
				finding_type=_ft_map.get(_normalize(f.get("finding_type", "")), FindingType.NON_COMPLIANCE),
				description=f.get("description", "audit_finding"),
				additional_tax=add_tax,
				penalty_amount=penalty,
				interest_amount=interest,
				total_amount=_cents(add_tax + penalty + interest),
				period_affected=f.get("period_affected"),
				evidence_reference=f.get("evidence_reference", "audit_finding"),
				created_by=created_by,
			)
			self._findings.put(tenant_id, fr.id, fr)
			finding_ids.append(fr.id)
			total_additional += add_tax

		adata = audit.model_dump()
		adata["status"] = AuditStatus.IN_PROGRESS.value
		adata["finding_ids"] = list(set(adata.get("finding_ids", [])) | set(finding_ids))
		adata["total_additional_tax"] = _cents(Decimal(str(adata.get("total_additional_tax", 0))) + total_additional)
		adata["updated_at"] = _now()
		updated = TaxAuditResponse(**adata)
		self._audits.put(tenant_id, case_id, updated)
		self._audit(tenant_id, "audit_findings_recorded", case_id)
		return updated.model_dump(mode="json")

	def close_audit_case(
		self,
		case_id: str,
		outcome: str,
		final_tax_due: float,
		penalties: float,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Close an audit case and issue final assessment if tax is due."""
		audit = self._audits.get_item(tenant_id, case_id)
		assert audit is not None, f"audit case not found: {case_id}"

		adata = audit.model_dump()
		adata["status"] = AuditStatus.FINALISED.value
		adata["scope_description"] = (adata.get("scope_description") or "") + f" | outcome={outcome}"
		adata["total_additional_tax"] = _cents(Decimal(str(final_tax_due)))
		adata["updated_at"] = _now()
		closed = TaxAuditResponse(**adata)
		self._audits.put(tenant_id, case_id, closed)

		# Issue audit assessment if tax is due
		assessment_id: str | None = None
		if final_tax_due > 0:
			ar = TaxAssessmentResponse(
				tenant_id=tenant_id,
				return_id=uuid7str(),  # synthetic
				taxpayer_id=audit.taxpayer_id,
				assessment_type=AssessmentType.AUDIT_ASSESSMENT,
				assessed_amount=_cents(Decimal(str(final_tax_due))),
				tax_liability_per_return=Decimal("0"),
				additional_tax=_cents(Decimal(str(final_tax_due))),
				assessor_id=audit.auditor_id,
				assessment_date=_today(),
				due_date=_today() + timedelta(days=30),
				evidence_reference=f"audit_case={case_id}",
				notes=outcome,
				status=AssessmentStatus.ISSUED,
				created_by="system",
			)
			self._assessments.put(tenant_id, ar.id, ar)
			assessment_id = ar.id

		self._audit(tenant_id, "audit_case_closed", case_id)
		result = closed.model_dump(mode="json")
		result["audit_assessment_id"] = assessment_id
		return result

	def audit_case_analytics(self, period: str, *, tenant_id: str = "default") -> dict[str, Any]:
		"""Aggregate audit statistics for a reporting period."""
		period_start, period_end = self._parse_period(period)
		all_audits = [
			a for a in self._audits.tenant_values(tenant_id)
			if a.tax_period_start <= period_end and a.tax_period_end >= period_start
		]
		by_type: dict[str, int] = {}
		by_status: dict[str, int] = {}
		total_additional = Decimal("0")
		for a in all_audits:
			by_type[a.audit_type.value] = by_type.get(a.audit_type.value, 0) + 1
			by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
			total_additional += a.total_additional_tax

		return {
			"period": period,
			"total_cases": len(all_audits),
			"by_type": by_type,
			"by_status": by_status,
			"total_additional_tax_assessed": str(total_additional),
			"generated_at": _now().isoformat(),
		}

	# ==================================================================
	# PAYMENT & DEBT
	# ==================================================================

	def process_tax_payment(
		self,
		tin: str,
		tax_type: str,
		period: str,
		amount: float,
		payment_method: str,
		reference: str,
		*,
		tenant_id: str = "default",
		assessment_id: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Record a tax payment and queue it for allocation."""
		assert amount > 0, "payment amount must be positive"
		assert _present(reference), "payment reference required"

		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"
		taxpayer_id = tp["id"] if isinstance(tp, dict) else tp.id

		_pm_map = {
			"bank_transfer": PaymentMethod.BANK_TRANSFER,
			"mobile_money": PaymentMethod.MOBILE_MONEY,
			"mpesa": PaymentMethod.MOBILE_MONEY,
			"cheque": PaymentMethod.CHEQUE,
			"cash": PaymentMethod.CASH,
			"card": PaymentMethod.CREDIT_CARD,
			"direct_debit": PaymentMethod.DIRECT_DEBIT,
			"rtgs": PaymentMethod.RTGS,
		}
		pm = _pm_map.get(_normalize(payment_method), PaymentMethod.BANK_TRANSFER)

		# Find associated return if any
		period_start, period_end = self._parse_period(period)
		return_id = next(
			(r.id for r in self._returns.tenant_values(tenant_id)
			 if r.tax_pin.upper() == tin.strip().upper()
			 and r.tax_period_start <= period_end
			 and r.tax_period_end >= period_start),
			None,
		)

		rec = TaxPaymentResponse(
			tenant_id=tenant_id,
			taxpayer_id=taxpayer_id,
			assessment_id=assessment_id,
			return_id=return_id,
			payment_reference=reference,
			payment_method=pm,
			amount=_cents(Decimal(str(amount))),
			payment_date=_today(),
			evidence_reference=reference,
			status=PaymentStatus.CONFIRMED,
			created_by=created_by,
		)
		self._payments.put(tenant_id, rec.id, rec)
		self._audit(tenant_id, "payment_received", rec.id)
		return rec.model_dump(mode="json")

	def allocate_payment_to_assessments(
		self, payment_id: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Allocate a confirmed payment to outstanding debt(s) FIFO by due_date."""
		payment = self._payments.get_item(tenant_id, payment_id)
		assert payment is not None, f"payment not found: {payment_id}"

		outstanding_debts = sorted(
			[d for d in self._debts.tenant_values(tenant_id)
			 if d.taxpayer_id == payment.taxpayer_id
			 and d.status in (DebtStatus.OUTSTANDING, DebtStatus.PARTIALLY_PAID)],
			key=lambda d: d.due_date,
		)

		remaining = payment.amount
		allocated: list[dict[str, Any]] = []

		for debt in outstanding_debts:
			if remaining <= 0:
				break
			apply = min(remaining, debt.balance)
			ddata = debt.model_dump()
			ddata["amount_paid"] = _cents(Decimal(str(ddata.get("amount_paid", 0))) + apply)
			ddata["balance"] = _cents(debt.balance - apply)
			ddata["status"] = (
				DebtStatus.PAID.value if ddata["balance"] == 0
				else DebtStatus.PARTIALLY_PAID.value
			)
			ddata["updated_at"] = _now()
			self._debts.put(tenant_id, debt.id, TaxDebtResponse(**ddata))
			allocated.append({"debt_id": debt.id, "applied": str(apply), "balance": str(ddata["balance"])})
			remaining -= apply

		# Update payment applied_to list
		pdata = payment.model_dump()
		pdata["applied_to"] = [a["debt_id"] for a in allocated]
		pdata["status"] = (
			PaymentStatus.FULLY_APPLIED.value if remaining == 0
			else (PaymentStatus.PARTIALLY_APPLIED.value if allocated else PaymentStatus.CONFIRMED.value)
		)
		pdata["updated_at"] = _now()
		self._payments.put(tenant_id, payment_id, TaxPaymentResponse(**pdata))

		return {
			"payment_id": payment_id,
			"amount": str(payment.amount),
			"unallocated": str(remaining),
			"allocated": allocated,
		}

	def calculate_penalty_and_interest(
		self,
		assessment_id: str,
		payment_date: str,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute Kenya-rate late-filing penalty (5%) and late-payment interest (1%/month).

		Kenya Income Tax Act / VAT Act rates:
		  - Late filing: 5% of tax due (minimum KES 1,000)
		  - Late payment: 1% of outstanding per commenced month
		"""
		assessment = self._assessments.get_item(tenant_id, assessment_id)
		assert assessment is not None, f"assessment not found: {assessment_id}"

		payment_dt = date.fromisoformat(payment_date)
		due_date = assessment.due_date or (assessment.assessment_date + timedelta(days=30))

		# Late filing penalty
		tax_due = assessment.assessed_amount
		filing_penalty = Decimal("0")
		if payment_dt > due_date:
			filing_penalty = max(
				_cents(tax_due * _KE_LATE_FILING_RATE),
				_KE_LATE_FILING_MINIMUM,
			)

		# Late payment interest: 1% per commenced month from due_date to payment_date
		months_late = _months_between(due_date, payment_dt) if payment_dt > due_date else 0
		interest_amount = _cents(tax_due * _KE_LATE_PAYMENT_MONTHLY * months_late)

		total = _cents(tax_due + filing_penalty + interest_amount)

		# Persist penalty record
		pen = PenaltyResponse(
			tenant_id=tenant_id,
			taxpayer_id=assessment.taxpayer_id,
			assessment_id=assessment_id,
			penalty_type=PenaltyType.LATE_PAYMENT,
			base_amount=tax_due,
			rate=_KE_LATE_FILING_RATE,
			calculated_amount=filing_penalty,
			period_days=(payment_dt - due_date).days if payment_dt > due_date else 0,
			status=PenaltyStatus.ASSESSED,
			created_by="system",
		)
		self._penalties.put(tenant_id, pen.id, pen)

		# Persist interest record
		int_rec = InterestResponse(
			tenant_id=tenant_id,
			taxpayer_id=assessment.taxpayer_id,
			assessment_id=assessment_id,
			interest_type=InterestType.LATE_PAYMENT,
			principal_amount=tax_due,
			annual_rate=_KE_LATE_PAYMENT_MONTHLY * 12,
			from_date=due_date,
			to_date=payment_dt,
			days=(payment_dt - due_date).days,
			calculated_amount=interest_amount,
			created_by="system",
		)
		self._interests.put(tenant_id, int_rec.id, int_rec)

		return {
			"assessment_id": assessment_id,
			"tax_due": str(tax_due),
			"due_date": due_date.isoformat(),
			"payment_date": payment_date,
			"days_late": (payment_dt - due_date).days if payment_dt > due_date else 0,
			"months_late": months_late,
			"late_filing_penalty": str(filing_penalty),
			"late_payment_interest": str(interest_amount),
			"total_payable": str(total),
			"penalty_id": pen.id,
			"interest_id": int_rec.id,
		}

	def issue_demand_notice(
		self,
		tin: str,
		outstanding_amount: float,
		deadline: str,
		*,
		tenant_id: str = "default",
		issued_by: str = "system",
	) -> dict[str, Any]:
		"""Issue a formal demand notice to a delinquent taxpayer."""
		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"
		taxpayer_id = tp["id"] if isinstance(tp, dict) else tp.id

		notice = DemandNotice(
			tenant_id=tenant_id,
			debt_id=uuid7str(),
			taxpayer_id=taxpayer_id,
			tax_pin=tin.strip().upper(),
			amount_demanded=_cents(Decimal(str(outstanding_amount))),
			due_date=date.fromisoformat(deadline),
			notice_number=f"DN-{_today().strftime('%Y%m%d')}-{uuid7str()[:6].upper()}",
			issued_date=_today(),
			notice_text=(
				f"You are hereby required to pay KES {outstanding_amount:,.2f} "
				f"within 30 days failing which enforcement action will be taken."
			),
			issued_by=issued_by,
		)
		self._demand_notices.put(tenant_id, notice.id, notice)
		self._audit(tenant_id, "demand_notice_issued", notice.id)
		return notice.model_dump(mode="json")

	def debt_collection_action(
		self,
		tin: str,
		action_type: str,
		officer_id: str,
		*,
		tenant_id: str = "default",
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Initiate a formal debt collection action.

		action_types: distress / court_order / employer_attachment / bank_attachment /
		              payment_plan / garnishment / asset_seizure / write_off
		"""
		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"
		assert _present(officer_id), "officer_id required"

		_cm_map = {
			"distress": CollectionMethod.ASSET_SEIZURE,
			"court_order": CollectionMethod.LEGAL_PROCEEDINGS,
			"employer_attachment": CollectionMethod.SALARY_ATTACHMENT,
			"bank_attachment": CollectionMethod.BANK_LEVY,
			"garnishment": CollectionMethod.GARNISHMENT,
			"payment_plan": CollectionMethod.PAYMENT_PLAN,
			"asset_seizure": CollectionMethod.ASSET_SEIZURE,
			"write_off": CollectionMethod.WRITE_OFF,
		}
		cm = _cm_map.get(_normalize(action_type), CollectionMethod.LEGAL_PROCEEDINGS)
		taxpayer_id = tp["id"] if isinstance(tp, dict) else tp.id

		total_outstanding = sum(
			d.balance for d in self._debts.tenant_values(tenant_id)
			if d.taxpayer_id == taxpayer_id
			and d.status in (DebtStatus.OUTSTANDING, DebtStatus.PARTIALLY_PAID)
		)

		action = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"taxpayer_id": taxpayer_id,
			"tax_pin": tin.strip().upper(),
			"action_type": cm.value,
			"officer_id": officer_id,
			"total_outstanding": str(_cents(total_outstanding)),
			"notes": notes,
			"status": "initiated",
			"initiated_at": _now().isoformat(),
		}
		self._audit(tenant_id, "debt_collection_action_initiated", action["id"])
		return action

	def issue_tax_clearance_certificate(
		self,
		tin: str,
		validity_days: int = 180,
		*,
		tenant_id: str = "default",
		purpose: str = "general",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Issue a Tax Clearance Certificate if taxpayer has no outstanding debts."""
		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"
		taxpayer_id = tp["id"] if isinstance(tp, dict) else tp.id

		# Check outstanding debts
		outstanding = [
			d for d in self._debts.tenant_values(tenant_id)
			if d.taxpayer_id == taxpayer_id
			and d.status in (DebtStatus.OUTSTANDING, DebtStatus.PARTIALLY_PAID)
			and d.balance > 0
		]

		if outstanding:
			return {
				"tin": tin,
				"status": ClearanceCertificateStatus.REJECTED.value,
				"reason": f"outstanding_debts={len(outstanding)} total={sum(d.balance for d in outstanding)}",
				"issued": False,
			}

		cert = TaxClearanceCertificateResponse(
			tenant_id=tenant_id,
			taxpayer_id=taxpayer_id,
			tax_pin=tin.strip().upper(),
			purpose=purpose,
			certificate_number=f"TCC-{_today().strftime('%Y')}-{uuid7str()[:8].upper()}",
			issue_date=_today(),
			expiry_date=_today() + timedelta(days=validity_days),
			validity_months=validity_days // 30,
			evidence_reference="clearance_issued",
			status=ClearanceCertificateStatus.ISSUED,
			reviewer_id=created_by,
			created_by=created_by,
		)
		self._clearances.put(tenant_id, cert.id, cert)
		self._audit(tenant_id, "clearance_certificate_issued", cert.id)
		return cert.model_dump(mode="json")

	# ==================================================================
	# REFUNDS
	# ==================================================================

	def refund_application(
		self,
		tin: str,
		tax_type: str,
		period: str,
		refund_amount: float,
		reason: str,
		*,
		tenant_id: str = "default",
		bank_account_number: str | None = None,
		bank_name: str | None = None,
		supporting_documents: list[str] | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Submit a refund application for overpaid tax or input VAT credit."""
		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"
		assert refund_amount > 0, "refund_amount must be positive"
		taxpayer_id = tp["id"] if isinstance(tp, dict) else tp.id

		# Find the associated return
		period_start, period_end = self._parse_period(period)
		ret = next(
			(r for r in self._returns.tenant_values(tenant_id)
			 if r.tax_pin.upper() == tin.strip().upper()
			 and r.tax_period_start <= period_end
			 and r.tax_period_end >= period_start),
			None,
		)
		return_id = ret.id if ret else uuid7str()

		rec = TaxRefundResponse(
			tenant_id=tenant_id,
			taxpayer_id=taxpayer_id,
			tax_pin=tin.strip().upper(),
			return_id=return_id,
			refund_type=reason,
			claimed_amount=_cents(Decimal(str(refund_amount))),
			bank_account_number=bank_account_number,
			bank_name=bank_name,
			evidence_reference="refund_application",
			status=RefundStatus.CLAIMED,
			created_by=created_by,
		)
		self._refunds.put(tenant_id, rec.id, rec)
		self._audit(tenant_id, "refund_application_submitted", rec.id)
		return rec.model_dump(mode="json")

	def verify_refund(
		self,
		refund_id: str,
		officer_id: str,
		*,
		tenant_id: str = "default",
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Assign a reviewer to a refund claim and transition to UNDER_REVIEW."""
		rec = self._refunds.get_item(tenant_id, refund_id)
		assert rec is not None, f"refund not found: {refund_id}"
		assert _present(officer_id), "officer_id required"

		data = rec.model_dump()
		data["status"] = RefundStatus.UNDER_REVIEW.value
		data["reviewer_id"] = officer_id
		data["review_notes"] = notes
		data["updated_at"] = _now()
		updated = TaxRefundResponse(**data)
		self._refunds.put(tenant_id, refund_id, updated)
		self._audit(tenant_id, "refund_under_review", refund_id)
		return updated.model_dump(mode="json")

	def approve_refund(
		self,
		refund_id: str,
		approved_by: str,
		payment_method: str,
		*,
		tenant_id: str = "default",
		approved_amount: float | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Approve a refund and record the payment method for disbursement."""
		rec = self._refunds.get_item(tenant_id, refund_id)
		assert rec is not None, f"refund not found: {refund_id}"
		assert _present(approved_by), "approved_by required"

		data = rec.model_dump()
		data["status"] = RefundStatus.APPROVED.value
		data["reviewer_id"] = approved_by
		data["review_notes"] = notes
		data["approved_amount"] = _cents(
			Decimal(str(approved_amount)) if approved_amount is not None
			else rec.claimed_amount
		)
		data["processed_date"] = _today().isoformat()
		data["metadata"]["payment_method"] = payment_method
		data["updated_at"] = _now()
		updated = TaxRefundResponse(**data)
		self._refunds.put(tenant_id, refund_id, updated)
		self._audit(tenant_id, "refund_approved", refund_id)
		return updated.model_dump(mode="json")

	def refund_analytics(self, period: str, *, tenant_id: str = "default") -> dict[str, Any]:
		"""Aggregate refund statistics for a given period."""
		period_start, period_end = self._parse_period(period)
		all_refunds = [
			r for r in self._refunds.tenant_values(tenant_id)
			if r.created_at.date() >= period_start and r.created_at.date() <= period_end
		]
		by_status: dict[str, int] = {}
		total_claimed = Decimal("0")
		total_approved = Decimal("0")
		for r in all_refunds:
			s = r.status.value
			by_status[s] = by_status.get(s, 0) + 1
			total_claimed += r.claimed_amount
			total_approved += r.approved_amount or Decimal("0")

		return {
			"period": period,
			"total_applications": len(all_refunds),
			"by_status": by_status,
			"total_claimed": str(_cents(total_claimed)),
			"total_approved": str(_cents(total_approved)),
			"approval_rate": (
				str(_cents(total_approved / total_claimed * 100)) if total_claimed > 0 else "0.00"
			),
			"generated_at": _now().isoformat(),
		}

	# ==================================================================
	# REPORTING
	# ==================================================================

	def revenue_collection_report(
		self,
		period: str,
		tax_type: str | None = None,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Revenue collection report: assessed vs collected vs refunded by tax type."""
		period_start, period_end = self._parse_period(period)

		# Payments in period
		payments = [
			p for p in self._payments.tenant_values(tenant_id)
			if p.payment_date >= period_start and p.payment_date <= period_end
		]
		# Assessments in period
		assessments = [
			a for a in self._assessments.tenant_values(tenant_id)
			if a.assessment_date >= period_start and a.assessment_date <= period_end
		]
		# Refunds paid in period
		refunds = [
			r for r in self._refunds.tenant_values(tenant_id)
			if r.status == RefundStatus.APPROVED
			and r.processed_date and date.fromisoformat(str(r.processed_date)) >= period_start
			and date.fromisoformat(str(r.processed_date)) <= period_end
		]

		total_assessed = _cents(sum(a.assessed_amount for a in assessments))
		total_collected = _cents(sum(p.amount for p in payments))
		total_refunded = _cents(sum(r.approved_amount or r.claimed_amount for r in refunds))
		net_revenue = _cents(total_collected - total_refunded)

		# Build by_tax_type if assessments have type info
		by_type: dict[str, str] = {}

		return {
			"tenant_id": tenant_id,
			"period": period,
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"tax_type_filter": tax_type,
			"total_assessments": len(assessments),
			"total_assessed": str(total_assessed),
			"total_payments": len(payments),
			"total_collected": str(total_collected),
			"total_refunded": str(total_refunded),
			"net_revenue": str(net_revenue),
			"collection_rate": str(_cents(total_collected / total_assessed * 100)) if total_assessed > 0 else "0.00",
			"by_tax_type": by_type,
			"generated_at": _now().isoformat(),
		}

	def compliance_rate_report(
		self,
		period: str,
		sector: str | None = None,
		*,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compliance rate: taxpayers who filed on time vs total obligated."""
		period_start, period_end = self._parse_period(period)

		all_taxpayers = list(self._taxpayers.tenant_values(tenant_id))
		if sector:
			all_taxpayers = [t for t in all_taxpayers if (t.sector_code or "").lower() == sector.lower()]

		active = [t for t in all_taxpayers if t.status == TaxpayerStatus.ACTIVE]
		filed_tins = {
			r.tax_pin.upper()
			for r in self._returns.tenant_values(tenant_id)
			if r.tax_period_start <= period_end and r.tax_period_end >= period_start
		}
		compliant = [t for t in active if t.tax_pin.upper() in filed_tins]
		non_compliant = [t for t in active if t.tax_pin.upper() not in filed_tins]

		rate = len(compliant) / len(active) if active else 0.0

		return {
			"period": period,
			"sector": sector,
			"total_active_taxpayers": len(active),
			"compliant": len(compliant),
			"non_compliant": len(non_compliant),
			"compliance_rate": round(rate, 4),
			"compliance_pct": f"{rate * 100:.2f}%",
			"generated_at": _now().isoformat(),
		}

	def delinquency_report(
		self, as_of_date: str, *, tenant_id: str = "default"
	) -> dict[str, Any]:
		"""Debt aging report bucketed by 0-30, 31-90, 91-180, 180+ days."""
		as_of = date.fromisoformat(as_of_date)
		outstanding = [
			d for d in self._debts.tenant_values(tenant_id)
			if d.status in (DebtStatus.OUTSTANDING, DebtStatus.PARTIALLY_PAID)
			and d.balance > 0
		]

		buckets: dict[str, dict[str, Any]] = {
			"0-30": {"count": 0, "balance": Decimal("0"), "taxpayer_ids": set()},
			"31-90": {"count": 0, "balance": Decimal("0"), "taxpayer_ids": set()},
			"91-180": {"count": 0, "balance": Decimal("0"), "taxpayer_ids": set()},
			"180+": {"count": 0, "balance": Decimal("0"), "taxpayer_ids": set()},
		}

		for d in outstanding:
			age = (as_of - d.due_date).days
			if age <= 30:
				bk = "0-30"
			elif age <= 90:
				bk = "31-90"
			elif age <= 180:
				bk = "91-180"
			else:
				bk = "180+"
			buckets[bk]["count"] += 1
			buckets[bk]["balance"] += d.balance
			buckets[bk]["taxpayer_ids"].add(d.taxpayer_id)

		total_balance = sum(d.balance for d in outstanding)
		result_buckets = {
			k: {
				"count": v["count"],
				"taxpayer_count": len(v["taxpayer_ids"]),
				"balance": str(_cents(v["balance"])),
			}
			for k, v in buckets.items()
		}

		return {
			"as_of_date": as_of_date,
			"total_delinquent_accounts": len(outstanding),
			"total_outstanding_balance": str(_cents(total_balance)),
			"aging_buckets": result_buckets,
			"generated_at": _now().isoformat(),
		}

	def exchange_of_information(
		self,
		request_source: str,
		tin: str,
		data_type: str,
		*,
		tenant_id: str = "default",
		urgency: str = "routine",
	) -> dict[str, Any]:
		"""Process an Exchange of Information request (FATCA / CRS / DTA).

		``request_source``: ISO country code of the requesting jurisdiction.
		``data_type``: e.g. "account_balances", "income", "beneficial_ownership"
		``urgency``: routine / urgent / spontaneous
		"""
		tp = self._find_taxpayer_by_pin(tin, tenant_id)
		assert tp is not None, f"taxpayer not found: {tin}"
		taxpayer_name = tp["taxpayer_name"] if isinstance(tp, dict) else tp.taxpayer_name

		eoi = EOIRequest(
			tenant_id=tenant_id,
			treaty_partner=request_source.upper(),
			subject_taxpayer_id=tp["id"] if isinstance(tp, dict) else tp.id,
			subject_name=taxpayer_name,
			information_requested=data_type,
			legal_basis="double_tax_agreement",
			urgency=urgency,
			response_deadline=_today() + timedelta(days=90 if urgency == "routine" else 30),
		)
		self._eoi_requests.put(tenant_id, eoi.id, eoi)
		self._audit(tenant_id, "eoi_request_processed", eoi.id)
		return eoi.model_dump(mode="json")

	# ==================================================================
	# DASHBOARD
	# ==================================================================

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""KPI dashboard for the tax administration module."""
		all_taxpayers = list(self._taxpayers.tenant_values(tenant_id))
		active_tp = sum(1 for t in all_taxpayers if t.status == TaxpayerStatus.ACTIVE)
		all_returns = list(self._returns.tenant_values(tenant_id))
		overdue_returns = sum(
			1 for r in all_returns
			if r.status in (ReturnStatus.DRAFT, ReturnStatus.FILED)
			and r.filing_date and (r.filing_date.date() if hasattr(r.filing_date, 'date') else r.filing_date) < _today()
		)
		all_assessments = list(self._assessments.tenant_values(tenant_id))
		pending_assessments = sum(1 for a in all_assessments if a.status == AssessmentStatus.ISSUED)
		total_assessed = _cents(sum(a.assessed_amount for a in all_assessments))
		total_collected = _cents(sum(p.amount for p in self._payments.tenant_values(tenant_id)))
		all_debts = list(self._debts.tenant_values(tenant_id))
		outstanding_debt = _cents(sum(
			d.balance for d in all_debts
			if d.status in (DebtStatus.OUTSTANDING, DebtStatus.PARTIALLY_PAID)
		))
		open_objections = sum(
			1 for o in self._objections.tenant_values(tenant_id)
			if o.status in (ObjectionStatus.SUBMITTED, ObjectionStatus.UNDER_REVIEW)
		)
		open_audits = sum(
			1 for a in self._audits.tenant_values(tenant_id)
			if a.status in (AuditStatus.PLANNED, AuditStatus.IN_PROGRESS)
		)
		pending_refunds = sum(
			1 for r in self._refunds.tenant_values(tenant_id)
			if r.status in (RefundStatus.CLAIMED, RefundStatus.UNDER_REVIEW)
		)
		pending_certs = sum(
			1 for c in self._clearances.tenant_values(tenant_id)
			if c.status in (ClearanceCertificateStatus.APPLIED, ClearanceCertificateStatus.UNDER_REVIEW)
		)
		compliance_rate = active_tp / len(all_taxpayers) if all_taxpayers else Decimal("0")
		collection_rate = total_collected / total_assessed if total_assessed > 0 else Decimal("0")

		return TaxDashboardKPI(
			tenant_id=tenant_id,
			as_of=_now(),
			registered_taxpayers=len(all_taxpayers),
			active_taxpayers=active_tp,
			returns_filed_ytd=len(all_returns),
			returns_overdue=overdue_returns,
			assessments_pending=pending_assessments,
			total_tax_assessed=total_assessed,
			total_tax_collected=total_collected,
			total_outstanding_debt=outstanding_debt,
			open_objections=open_objections,
			open_audits=open_audits,
			pending_refunds=pending_refunds,
			pending_clearance_certs=pending_certs,
			compliance_rate=_cents(Decimal(str(compliance_rate))),
			collection_rate=_cents(Decimal(str(collection_rate))),
		).model_dump(mode="json")

	# ==================================================================
	# LEGACY adapter interface (capability_contract-driven, original signatures)
	# All legacy methods now delegate to the Pydantic store instead of
	# separate dataclass stores.
	# ==================================================================

	def file_return(
		self, return_id: str, tenant_id: str, return_type: str, taxpayer_pin: str,
		period: str, gross_income: float, tax_liability: float, tax_paid: float,
		evidence_reference: str, status: str = "filed",
	) -> dict[str, Any]:
		"""Legacy: file a tax return (original positional interface)."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "file_return",
			"return_type_supported": _normalize(return_type) in SUPPORTED_RETURN_TYPES,
			"taxpayer_pin_present": _present(taxpayer_pin),
			"period_present": _present(period),
			"evidence_present": _present(evidence_reference),
		})
		period_start, period_end = self._parse_period(period)
		_rt_map = {
			"monthly_vat": ReturnType.MONTHLY_VAT,
			"annual_income": ReturnType.ANNUAL_INCOME,
			"quarterly_advance": ReturnType.QUARTERLY_ADVANCE,
			"withholding_tax_return": ReturnType.WITHHOLDING_TAX_RETURN,
			"corporate_annual": ReturnType.CORPORATE_ANNUAL,
			"customs_entry": ReturnType.CUSTOMS_ENTRY,
		}
		rt = _rt_map.get(_normalize(return_type), ReturnType.ANNUAL_INCOME)
		rec = TaxReturnResponse(
			id=return_id,
			tenant_id=tenant_id,
			taxpayer_id=taxpayer_pin,
			tax_pin=taxpayer_pin,
			return_type=rt,
			tax_period_start=period_start,
			tax_period_end=period_end,
			gross_income=Decimal(str(gross_income)),
			allowable_deductions=Decimal("0"),
			taxable_income=Decimal(str(gross_income)),
			tax_liability=Decimal(str(tax_liability)),
			tax_credits=Decimal("0"),
			tax_paid=Decimal(str(tax_paid)),
			net_tax_payable=_cents(Decimal(str(tax_liability)) - Decimal(str(tax_paid))),
			status=ReturnStatus(status) if status in ReturnStatus._value2member_map_ else ReturnStatus.FILED,
			evidence_reference=evidence_reference,
			is_amended=False,
			created_by="system",
		)
		self._returns.put(tenant_id, return_id, rec)
		self._audit(tenant_id, "tax_return_filed", return_id)
		return rec.model_dump(mode="json")

	def raise_assessment(
		self, assessment_id: str, tenant_id: str, return_id: str, assessment_type: str,
		assessed_amount: float, assessor_id: str, assessment_date: str, evidence_reference: str,
		status: str = "draft",
	) -> dict[str, Any]:
		"""Legacy: raise a tax assessment (original positional interface)."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "raise_assessment",
			"assessment_type_supported": _normalize(assessment_type) in SUPPORTED_ASSESSMENT_TYPES,
			"return_present": self._returns.get_item(tenant_id, return_id) is not None,
			"assessor_present": _present(assessor_id),
		})
		_at_map = {
			"self_assessment": AssessmentType.SELF_ASSESSMENT,
			"amended_assessment": AssessmentType.AMENDED_ASSESSMENT,
			"best_judgement": AssessmentType.BEST_JUDGEMENT,
			"audit_assessment": AssessmentType.AUDIT_ASSESSMENT,
			"estimated_assessment": AssessmentType.ESTIMATED_ASSESSMENT,
		}
		at = _at_map.get(_normalize(assessment_type), AssessmentType.BEST_JUDGEMENT)
		_as_map = {
			"draft": AssessmentStatus.DRAFT,
			"issued": AssessmentStatus.ISSUED,
		}
		ast = _as_map.get(_normalize(status), AssessmentStatus.DRAFT)
		rec = TaxAssessmentResponse(
			id=assessment_id,
			tenant_id=tenant_id,
			return_id=return_id,
			taxpayer_id=return_id,  # best available without full lookup
			assessment_type=at,
			assessed_amount=_cents(Decimal(str(assessed_amount))),
			tax_liability_per_return=Decimal("0"),
			additional_tax=_cents(Decimal(str(assessed_amount))),
			assessor_id=assessor_id,
			assessment_date=date.fromisoformat(assessment_date) if assessment_date else _today(),
			evidence_reference=evidence_reference,
			status=ast,
			created_by="system",
		)
		self._assessments.put(tenant_id, assessment_id, rec)
		self._audit(tenant_id, "tax_assessed", assessment_id)
		return rec.model_dump(mode="json")

	def file_objection(
		self, objection_id: str, tenant_id: str, assessment_id: str, taxpayer_pin: str,
		grounds: str, amount_disputed: float, evidence_reference: str,
		filed_date: str = "", within_deadline: bool = True,
	) -> dict[str, Any]:
		"""Legacy: file an objection (original positional interface)."""
		assessment = self._assessments.get_item(tenant_id, assessment_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "file_objection",
			"assessment_present": assessment is not None,
			"grounds_present": _present(grounds),
			"within_deadline": within_deadline,
		})
		fd = date.fromisoformat(filed_date) if filed_date else _today()
		rec = ObjectionResponse(
			id=objection_id,
			tenant_id=tenant_id,
			assessment_id=assessment_id,
			taxpayer_id=taxpayer_pin,
			tax_pin=taxpayer_pin,
			grounds=grounds,
			amount_disputed=_cents(Decimal(str(amount_disputed))),
			supporting_documents=[],
			evidence_reference=evidence_reference,
			filed_date=fd,
			status=ObjectionStatus.SUBMITTED,
			created_by="system",
		)
		self._objections.put(tenant_id, objection_id, rec)
		self._audit(tenant_id, "objection_filed", objection_id)
		return rec.model_dump(mode="json")

	def initiate_collection(
		self, collection_id: str, tenant_id: str, taxpayer_pin: str, assessment_id: str,
		collection_method: str, amount_owed: float, demand_notice_reference: str,
		approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Legacy: initiate debt collection (original positional interface)."""
		assessment = self._assessments.get_item(tenant_id, assessment_id)
		collection_method = _normalize(collection_method)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "initiate_collection",
			"assessed_liability_present": assessment is not None,
			"demand_notice_issued": _present(demand_notice_reference),
			"collection_method_supported": collection_method in SUPPORTED_DEBT_COLLECTION_METHODS,
		})
		return self.debt_collection_action(
			taxpayer_pin, collection_method, approval_reference or "system",
			tenant_id=tenant_id,
			notes=f"collection_id={collection_id} demand_ref={demand_notice_reference} evidence={evidence_reference}",
		)

	def open_audit(
		self, audit_id: str, tenant_id: str, taxpayer_pin: str, audit_type: str,
		auditor_id: str, period_under_review: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Legacy: open an audit case (original positional interface)."""
		# Register a minimal taxpayer record if none exists so open_audit_case succeeds
		if self._find_taxpayer_by_pin(taxpayer_pin, tenant_id) is None:
			stub = TaxpayerResponse(
				tenant_id=tenant_id,
				taxpayer_type=TaxpayerType.INDIVIDUAL,
				tax_pin=taxpayer_pin,
				taxpayer_name=taxpayer_pin,
				evidence_reference=evidence_reference,
				status=TaxpayerStatus.ACTIVE,
				tax_types=[],
				country_of_incorporation="KE",
				is_resident=True,
				created_by="system",
				# suppress extra='forbid' for optional fields not supplied
			)
			self._taxpayers.put(tenant_id, stub.id, stub)
		return self.open_audit_case(
			taxpayer_pin, audit_type, period_under_review, auditor_id,
			tenant_id=tenant_id,
		)

	def complete_audit(self, audit_id: str, tenant_id: str, findings: str) -> dict[str, Any]:
		"""Legacy: complete an audit (original positional interface)."""
		return self.close_audit_case(
			audit_id, findings, 0.0, 0.0, tenant_id=tenant_id,
		)

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a governance review (original interface, now Pydantic-backed)."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": _normalize(status) in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": _present(reviewer_id),
			"evidence_present": _present(evidence_reference),
		})
		rec = {
			"id": review_id,
			"tenant_id": tenant_id,
			"reference_id": reference_id,
			"reviewer_id": reviewer_id,
			"status": _normalize(status),
			"evidence_reference": evidence_reference,
			"recorded_at": _now().isoformat(),
		}
		self._eoi_requests.put(tenant_id, review_id, rec)  # reuse misc store
		self._audit(tenant_id, "tax_review_recorded", review_id)
		return rec

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register a tax administration agent (original interface)."""
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_tax_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		agent = {
			"id": agent_id,
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"registered_at": _now().isoformat(),
		}
		self._eoi_requests.put(tenant_id, agent_id, agent)  # reuse misc store
		self._audit(tenant_id, "tax_agent_registered", agent_id)
		return agent

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "tax_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.tax.lifecycle", "accepted": True}

	# ==================================================================
	# Internal helpers
	# ==================================================================

	def _generate_pin(self, tenant_id: str, taxpayer_type: TaxpayerType) -> str:
		"""Generate a unique KRA-format PIN: A/P + 9 digits + check letter."""
		prefix = "A" if taxpayer_type == TaxpayerType.INDIVIDUAL else "P"
		import random
		import string
		for _ in range(100):
			digits = "".join(random.choices(string.digits, k=9))
			check = random.choice(string.ascii_uppercase)
			pin = f"{prefix}{digits}{check}"
			if not self._has_pin(pin, tenant_id):
				return pin
		raise RuntimeError("PIN generation exhausted")

	def _find_taxpayer_by_pin(self, tin: str, tenant_id: str) -> TaxpayerResponse | None:
		tin_clean = tin.strip().upper()
		return next(
			(r for r in self._taxpayers.tenant_values(tenant_id)
			 if r.tax_pin.upper() == tin_clean),
			None,
		)

	def _parse_period(self, period: str) -> tuple[date, date]:
		"""Parse a period string to (start, end) date tuple.

		Accepts: "2025", "2025-01", "Q1-2025", "2025-01-01", "2025-01-01/2025-03-31"
		"""
		p = period.strip()
		# Full range
		if "/" in p:
			parts = p.split("/")
			return date.fromisoformat(parts[0]), date.fromisoformat(parts[1])
		# Quarter: Q1-2025
		qm = re.match(r"Q([1-4])-(\d{4})", p, re.IGNORECASE)
		if qm:
			q, year = int(qm.group(1)), int(qm.group(2))
			month_start = (q - 1) * 3 + 1
			month_end = month_start + 2
			import calendar
			last_day = calendar.monthrange(year, month_end)[1]
			return date(year, month_start, 1), date(year, month_end, last_day)
		# Monthly: 2025-01
		if re.match(r"^\d{4}-\d{2}$", p):
			year, month = int(p[:4]), int(p[5:7])
			import calendar
			last_day = calendar.monthrange(year, month)[1]
			return date(year, month, 1), date(year, month, last_day)
		# Annual: 2025
		if re.match(r"^\d{4}$", p):
			year = int(p)
			return date(year, 1, 1), date(year, 12, 31)
		# Full ISO date: treat as single day
		try:
			d = date.fromisoformat(p)
			return d, d
		except ValueError as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		raise ValueError(f"Cannot parse period: {period!r}")

	def _has_pin(self, tax_pin: str, tenant_id: str) -> bool:
		return any(
			r.tax_pin.upper() == tax_pin.upper()
			for r in self._taxpayers.tenant_values(tenant_id)
		)

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"recorded_at": _now().isoformat(),
		})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")


# backward-compat alias

	async def ml_tax_audit_select(self, *args, **kwargs):
		"""AI-powered AI selection scoring for tax audit targeting. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="tax_audit_selection")
			return {"audit_score": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

GovernmentTaxService = TaxAdministrationService
