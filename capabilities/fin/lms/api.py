"""Loan Management System — REST API handlers.

url_prefix = "/api/fin/lms"

Thin handlers: parse → service call → serialise.
No business logic lives here.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

try:
	from .models import (
		AmortisationMethod, ClosureReason, DemandNoticeType, LoanStatus,
		MoratoriumType, PaymentMethod, PenaltyType, RestructureType, LoanClassification,
	)
	from .service import LoanManagementService
except ImportError:
	from models import (  # type: ignore[no-redef]
		AmortisationMethod, ClosureReason, DemandNoticeType, LoanStatus,
		MoratoriumType, PaymentMethod, PenaltyType, RestructureType, LoanClassification,
	)
	from service import LoanManagementService  # type: ignore[no-redef]


_SERVICE = LoanManagementService()


def service() -> LoanManagementService:
	"""Return process-level singleton; override in tests or DI containers."""
	return _SERVICE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _d(v: Any) -> Decimal:
	return Decimal(str(v))


def _date(v: Any) -> date:
	if isinstance(v, date):
		return v
	return date.fromisoformat(str(v))


# ── Health ────────────────────────────────────────────────────────────────────

def health() -> dict[str, Any]:
	return _SERVICE.health_check()


# ── Disbursement ──────────────────────────────────────────────────────────────

async def disburse_loan(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/disburse
	{
	  "tenant_id": "t1",
	  "disbursement_date": "2025-01-15",
	  "account_id": "ACC-001",
	  "amount": "100000.00",
	  "disbursement_ref": "REF-001"
	}
	"""
	return await service().disburse_loan(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		disbursement_date=_date(payload["disbursement_date"]),
		account_id=payload["account_id"],
		amount=_d(payload["amount"]),
		disbursement_ref=payload["disbursement_ref"],
	)


async def generate_schedule(payload: dict[str, Any]) -> list[dict[str, Any]]:
	"""
	POST /api/fin/lms/schedule/generate
	"""
	return await service().generate_amortisation_schedule(
		loan_id=payload["loan_id"],
		principal=_d(payload["principal"]),
		rate=_d(payload["rate"]),
		tenor_months=int(payload["tenor_months"]),
		method=AmortisationMethod(payload["method"]),
		first_payment_date=_date(payload["first_payment_date"]),
	)


# ── Repayment ─────────────────────────────────────────────────────────────────

async def record_repayment(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/repayments
	"""
	return await service().record_repayment(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		amount=_d(payload["amount"]),
		payment_date=_date(payload["payment_date"]),
		payment_ref=payload["payment_ref"],
		payment_method=PaymentMethod(payload["payment_method"]),
	)


# ── Arrears ───────────────────────────────────────────────────────────────────

async def calculate_arrears(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/arrears
	{"tenant_id": "t1", "loan_id": "...", "as_of_date": "2025-06-01"}
	"""
	result = await service().calculate_arrears(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		as_of_date=_date(payload["as_of_date"]),
	)
	return result.model_dump()


async def apply_penalty(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/penalty
	"""
	amount = await service().apply_penalty(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		penalty_type=PenaltyType(payload["penalty_type"]),
		as_of_date=_date(payload["as_of_date"]),
	)
	return {"penalty_applied": str(amount)}


async def batch_arrears(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/batch/arrears
	{"tenant_id": "t1", "as_of_date": "2025-06-01"}
	"""
	return await service().batch_calculate_arrears(
		tenant_id=payload["tenant_id"],
		as_of_date=_date(payload["as_of_date"]),
	)


# ── Restructuring ─────────────────────────────────────────────────────────────

async def restructure_loan(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/restructure
	"""
	return await service().restructure_loan(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		restructure_type=RestructureType(payload["restructure_type"]),
		new_terms=payload.get("new_terms", {}),
		effective_date=_date(payload["effective_date"]),
		approved_by=payload["approved_by"],
	)


async def grant_moratorium(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/moratorium
	"""
	return await service().grant_moratorium(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		from_date=_date(payload["from_date"]),
		to_date=_date(payload["to_date"]),
		moratorium_type=MoratoriumType(payload["moratorium_type"]),
		reason=payload["reason"],
		approved_by=payload["approved_by"],
		interest_accrues=bool(payload.get("interest_accrues", True)),
	)


async def reprice_loan(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/reprice
	"""
	return await service().reprice_loan(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		new_rate=_d(payload["new_rate"]),
		effective_date=_date(payload["effective_date"]),
		approved_by=payload["approved_by"],
	)


# ── Write-off / recovery ──────────────────────────────────────────────────────

async def write_off_loan(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/write-off
	"""
	return await service().write_off_loan(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		write_off_date=_date(payload["write_off_date"]),
		reason=payload["reason"],
		approved_by=payload["approved_by"],
		write_off_amount=_d(payload["write_off_amount"]),
	)


async def record_recovery(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	POST /api/fin/lms/loans/{loan_id}/recovery
	"""
	return await service().record_recovery(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		amount=_d(payload["amount"]),
		recovery_date=_date(payload["recovery_date"]),
		method=payload["method"],
	)


# ── Query endpoints ───────────────────────────────────────────────────────────

async def get_loan(tenant_id: str, loan_id: str) -> dict[str, Any]:
	"""GET /api/fin/lms/loans/{loan_id}?tenant_id=t1"""
	return await service().get_loan(tenant_id, loan_id)


async def get_loan_schedule(tenant_id: str, loan_id: str) -> list[dict[str, Any]]:
	"""GET /api/fin/lms/loans/{loan_id}/schedule?tenant_id=t1"""
	return await service().get_loan_schedule(tenant_id, loan_id)


async def get_loan_statement(payload: dict[str, Any]) -> list[dict[str, Any]]:
	"""
	GET /api/fin/lms/loans/{loan_id}/statement?tenant_id=t1&from=&to=
	"""
	return await service().get_loan_statement(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		from_date=_date(payload["from_date"]),
		to_date=_date(payload["to_date"]),
	)


async def list_loans(payload: dict[str, Any]) -> list[dict[str, Any]]:
	"""
	GET /api/fin/lms/loans?tenant_id=t1&customer_id=&status=&days_past_due_min=
	"""
	return await service().list_loans(
		tenant_id=payload["tenant_id"],
		customer_id=payload.get("customer_id"),
		status=payload.get("status"),
		days_past_due_min=int(payload["days_past_due_min"]) if payload.get("days_past_due_min") is not None else None,
	)


# ── Portfolio / classification / provision ────────────────────────────────────

async def get_portfolio_quality(payload: dict[str, Any]) -> dict[str, Any]:
	"""
	GET /api/fin/lms/portfolio/quality?tenant_id=t1&as_of_date=2025-06-01
	"""
	result = await service().get_portfolio_quality(
		tenant_id=payload["tenant_id"],
		as_of_date=_date(payload["as_of_date"]),
	)
	return result.model_dump()


async def classify_loan(tenant_id: str, loan_id: str) -> dict[str, Any]:
	"""GET /api/fin/lms/loans/{loan_id}/classification?tenant_id=t1"""
	classification = await service().classify_loan(tenant_id, loan_id)
	return {"loan_id": loan_id, "classification": classification.value}


async def calculate_required_provision(tenant_id: str, loan_id: str) -> dict[str, Any]:
	"""GET /api/fin/lms/loans/{loan_id}/provision/required?tenant_id=t1"""
	amount = await service().calculate_required_provision(tenant_id, loan_id)
	return {"loan_id": loan_id, "required_provision": str(amount)}


async def post_provision_entry(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /api/fin/lms/loans/{loan_id}/provision"""
	return await service().post_provision_entry(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		provision_amount=_d(payload["provision_amount"]),
		posting_date=_date(payload["posting_date"]),
	)


async def get_provision_report(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /api/fin/lms/provision/report?tenant_id=t1&as_of_date="""
	return await service().get_provision_report(
		tenant_id=payload["tenant_id"],
		as_of_date=_date(payload["as_of_date"]),
	)


# ── Collections / closure ─────────────────────────────────────────────────────

async def send_demand_notice(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /api/fin/lms/loans/{loan_id}/notice"""
	return await service().send_demand_notice(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		notice_type=DemandNoticeType(payload["notice_type"]),
	)


async def refer_to_collections(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /api/fin/lms/loans/{loan_id}/collections/refer"""
	return await service().refer_to_collections(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		referred_by=payload["referred_by"],
		notes=payload.get("notes", ""),
	)


async def close_loan(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /api/fin/lms/loans/{loan_id}/close"""
	return await service().close_loan(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		closure_date=_date(payload["closure_date"]),
		reason=ClosureReason(payload["reason"]),
	)


async def get_early_settlement_amount(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /api/fin/lms/loans/{loan_id}/early-settlement?tenant_id=&settlement_date="""
	return await service().get_early_settlement_amount(
		tenant_id=payload["tenant_id"],
		loan_id=payload["loan_id"],
		settlement_date=_date(payload["settlement_date"]),
	)
