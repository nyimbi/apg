"""SACCO Lending — full async service."""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_lnd"
PRODUCT_TYPES = {"development", "emergency", "school_fees", "business", "mortgage", "asset"}
INTEREST_METHODS = {"reducing_balance", "flat_rate"}
LOAN_STATUSES = {"pending", "approved", "disbursed", "active", "arrears", "written_off", "closed"}
DISBURSEMENT_METHODS = {"cash", "mpesa", "bank_transfer", "savings_account"}
PAYMENT_METHODS = {"cash", "mpesa", "bank_transfer", "cheque", "salary_deduction"}
CRB_REPORT_TYPES = {"listing", "delisting", "inquiry"}

# Credit score grade thresholds
SCORE_GRADES = [(800, "A"), (650, "B"), (500, "C"), (350, "D"), (0, "E")]


class SaccoLendingService:
	"""Async service for SACCO loan products, credit scoring, repayments, arrears, and CRB reporting."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.products: dict[str, dict[str, Any]] = {}
		self.loans: dict[str, dict[str, Any]] = {}
		self.repayments: dict[str, dict[str, Any]] = {}
		self.schedules: dict[str, list[dict[str, Any]]] = {}
		self.credit_scores: dict[str, dict[str, Any]] = {}
		self.guarantors: dict[str, dict[str, Any]] = {}
		self.crb_reports: dict[str, dict[str, Any]] = {}
		self.arrears_records: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		self._loan_counter: int = 0

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _next_loan_number(self, tenant_id: str) -> str:
		self._loan_counter += 1
		return f"LN-{tenant_id[:4].upper()}-{self._loan_counter:07d}"

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record.get("id", ""),
			"record_type": record.get("type", "loan"),
			"emitted_at": self._now(),
		})

	def _get_loan(self, loan_id: str, tenant_id: str) -> dict[str, Any]:
		loan = self.loans.get(loan_id)
		if not loan or loan["tenant_id"] != tenant_id:
			raise KeyError(f"loan_not_found: {loan_id}")
		return loan

	def _get_product(self, product_id: str, tenant_id: str) -> dict[str, Any]:
		p = self.products.get(product_id)
		if not p or p["tenant_id"] != tenant_id:
			raise KeyError(f"product_not_found: {product_id}")
		return p

	def _score_to_grade(self, score: int) -> str:
		for threshold, grade in SCORE_GRADES:
			if score >= threshold:
				return grade
		return "E"

	def _build_reducing_balance_schedule(
		self,
		principal: Decimal,
		annual_rate: Decimal,
		term_months: int,
		start_date: str,
		grace_months: int = 0,
	) -> list[dict[str, Any]]:
		"""Generate a reducing-balance amortisation schedule."""
		monthly_rate = annual_rate / Decimal("1200")
		if monthly_rate == 0:
			monthly_payment = principal / Decimal(str(term_months))
		else:
			factor = (1 + monthly_rate) ** term_months
			monthly_payment = (principal * monthly_rate * factor / (factor - 1)).quantize(Decimal("0.01"), ROUND_HALF_UP)
		balance = principal
		schedule: list[dict[str, Any]] = []
		try:
			start = date.fromisoformat(start_date)
		except Exception:
			start = date.today()
		for i in range(1, term_months + 1):
			# approximate due date as month offset
			month = start.month + i - 1
			year = start.year + month // 12
			month = month % 12 or 12
			due = date(year, month, start.day)
			if i <= grace_months:
				interest = (balance * monthly_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
				schedule.append({
					"installment_no": i,
					"due_date": due.isoformat(),
					"principal": Decimal("0"),
					"interest": interest,
					"total_due": interest,
					"balance_after": balance,
					"status": "pending",
				})
				continue
			interest = (balance * monthly_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
			principal_component = (monthly_payment - interest).quantize(Decimal("0.01"), ROUND_HALF_UP)
			if principal_component > balance:
				principal_component = balance
			balance -= principal_component
			schedule.append({
				"installment_no": i,
				"due_date": due.isoformat(),
				"principal": principal_component,
				"interest": interest,
				"total_due": principal_component + interest,
				"balance_after": max(Decimal("0"), balance),
				"status": "pending",
			})
		return schedule

	# ── Health & Describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"product_count": len(self.products),
			"loan_count": len(self.loans),
			"active_loans": sum(1 for l in self.loans.values() if l.get("status") in {"active", "disbursed"}),
			"arrears_loans": sum(1 for l in self.loans.values() if l.get("status") == "arrears"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "fintech",
			"description": "SACCO loan products, credit scoring, guarantors, repayment schedules, arrears, CRB reporting",
			"product_types": list(PRODUCT_TYPES),
			"interest_methods": list(INTEREST_METHODS),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]

	# ── Loan Products ─────────────────────────────────────────────────────────

	async def create_product(
		self,
		product_code: str,
		product_name: str,
		product_type: str,
		interest_rate_pa: float,
		min_amount: float,
		max_amount: float,
		min_term_months: int,
		max_term_months: int,
		tenant_id: str | None = None,
		interest_method: str = "reducing_balance",
		max_multiplier: float = 3.0,
		grace_period_months: int = 0,
		processing_fee_pct: float = 0.0,
		insurance_fee_pct: float = 0.0,
		min_guarantors: int = 2,
		requires_collateral: bool = False,
		description: str | None = None,
	) -> dict[str, Any]:
		"""Define a new loan product."""
		t = self._tenant(tenant_id)
		if product_type not in PRODUCT_TYPES:
			raise ValueError(f"invalid_product_type: {product_type}")
		if interest_method not in INTEREST_METHODS:
			raise ValueError(f"invalid_interest_method: {interest_method}")
		for p in self.products.values():
			if p["tenant_id"] == t and p["product_code"] == product_code:
				raise ValueError(f"product_code_exists: {product_code}")
		pid = self._record_id("lprod")
		record: dict[str, Any] = {
			"id": pid,
			"type": "sacco_loan_product",
			"tenant_id": t,
			"product_code": product_code,
			"product_name": product_name,
			"product_type": product_type,
			"interest_rate_pa": Decimal(str(interest_rate_pa)),
			"interest_method": interest_method,
			"min_amount": Decimal(str(min_amount)),
			"max_amount": Decimal(str(max_amount)),
			"min_term_months": min_term_months,
			"max_term_months": max_term_months,
			"max_multiplier": Decimal(str(max_multiplier)),
			"grace_period_months": grace_period_months,
			"processing_fee_pct": Decimal(str(processing_fee_pct)),
			"insurance_fee_pct": Decimal(str(insurance_fee_pct)),
			"min_guarantors": min_guarantors,
			"requires_collateral": requires_collateral,
			"description": description,
			"is_active": True,
			"created_at": self._now(),
		}
		self.products[pid] = record
		self._emit(t, "loan_product_created", record)
		_log.info("Loan product created: %s tenant=%s", product_code, t)
		return deepcopy(record)

	async def update_product(
		self,
		product_id: str,
		tenant_id: str | None = None,
		interest_rate_pa: float | None = None,
		max_amount: float | None = None,
		max_multiplier: float | None = None,
		is_active: bool | None = None,
		description: str | None = None,
	) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		product = self._get_product(product_id, t)
		if interest_rate_pa is not None:
			product["interest_rate_pa"] = Decimal(str(interest_rate_pa))
		if max_amount is not None:
			product["max_amount"] = Decimal(str(max_amount))
		if max_multiplier is not None:
			product["max_multiplier"] = Decimal(str(max_multiplier))
		if is_active is not None:
			product["is_active"] = is_active
		if description is not None:
			product["description"] = description
		product["updated_at"] = self._now()
		self._emit(t, "loan_product_updated", product)
		return deepcopy(product)

	async def list_products(self, tenant_id: str | None = None, product_type: str | None = None, active_only: bool = True) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.products.values() if p["tenant_id"] == t]
		if active_only:
			items = [p for p in items if p.get("is_active")]
		if product_type:
			items = [p for p in items if p.get("product_type") == product_type]
		return items

	async def get_product(self, product_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		return deepcopy(self._get_product(product_id, t))

	async def delete_product(self, product_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		product = self._get_product(product_id, t)
		active = [l for l in self.loans.values() if l["tenant_id"] == t and l["product_id"] == product_id and l["status"] not in {"closed", "written_off"}]
		if active:
			raise ValueError(f"product_has_active_loans: {len(active)}")
		product["is_active"] = False
		product["deactivated_at"] = self._now()
		self._emit(t, "loan_product_deactivated", product)
		return deepcopy(product)

	# ── Credit Scoring ────────────────────────────────────────────────────────

	async def compute_credit_score(
		self,
		member_id: str,
		savings_balance: float,
		share_capital: float,
		months_as_member: int,
		existing_loan_balance: float,
		repayment_record_pct: float,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute a SACCO credit score for a member."""
		t = self._tenant(tenant_id)
		# Score components (weighted):
		# Savings adequacy (30 pts max): savings vs share capital ratio
		savings_score = min(30, int((savings_balance / max(share_capital, 1)) * 10))
		# Membership tenure (20 pts max)
		tenure_score = min(20, months_as_member // 6)
		# Repayment history (35 pts max)
		repay_score = int(repayment_record_pct * 0.35)
		# Debt burden (15 pts max): penalise high existing debt
		debt_ratio = existing_loan_balance / max(savings_balance + share_capital, 1)
		debt_score = max(0, 15 - int(debt_ratio * 15))
		raw = savings_score + tenure_score + repay_score + debt_score  # 0–100
		score = raw * 10  # scale to 0–1000
		grade = self._score_to_grade(score)
		# Max loan = min(product_max, multiplier × savings)
		max_loan = Decimal(str(savings_balance)) * Decimal("3") if grade in {"A", "B"} else Decimal(str(savings_balance)) * Decimal("2") if grade == "C" else Decimal(str(savings_balance))
		cs_id = self._record_id("cs")
		record: dict[str, Any] = {
			"id": cs_id,
			"type": "sacco_credit_score",
			"tenant_id": t,
			"member_id": member_id,
			"score": score,
			"grade": grade,
			"max_loan_amount": max_loan,
			"factors": {
				"savings_score": savings_score,
				"tenure_score": tenure_score,
				"repayment_score": repay_score,
				"debt_score": debt_score,
			},
			"valid_until": self._now()[:10],
			"created_at": self._now(),
		}
		self.credit_scores[cs_id] = record
		self._emit(t, "credit_score_computed", record)
		return deepcopy(record)

	async def get_latest_credit_score(self, member_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
		t = self._tenant(tenant_id)
		scores = [cs for cs in self.credit_scores.values() if cs["tenant_id"] == t and cs["member_id"] == member_id]
		if not scores:
			return None
		return deepcopy(sorted(scores, key=lambda x: x["created_at"], reverse=True)[0])

	# ── Loan Applications ─────────────────────────────────────────────────────

	async def apply_for_loan(
		self,
		member_id: str,
		product_id: str,
		amount_requested: float,
		term_months: int,
		purpose: str,
		tenant_id: str | None = None,
		guarantor_ids: list[str] | None = None,
		collateral_description: str | None = None,
		collateral_value: float | None = None,
	) -> dict[str, Any]:
		"""Submit a loan application."""
		t = self._tenant(tenant_id)
		product = self._get_product(product_id, t)
		amount = Decimal(str(amount_requested))
		if amount < product["min_amount"]:
			raise ValueError(f"amount_below_minimum: {product['min_amount']}")
		if amount > product["max_amount"]:
			raise ValueError(f"amount_above_maximum: {product['max_amount']}")
		if term_months < product["min_term_months"]:
			raise ValueError(f"term_below_minimum: {product['min_term_months']}")
		if term_months > product["max_term_months"]:
			raise ValueError(f"term_above_maximum: {product['max_term_months']}")
		if product["requires_collateral"] and not collateral_description:
			raise ValueError("collateral_required_for_product")
		guarantor_list = guarantor_ids or []
		if len(guarantor_list) < product["min_guarantors"]:
			raise ValueError(f"insufficient_guarantors: need {product['min_guarantors']}, got {len(guarantor_list)}")
		loan_number = self._next_loan_number(t)
		loan_id = self._record_id("ln")
		processing_fee = (amount * product["processing_fee_pct"] / 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
		insurance_fee = (amount * product["insurance_fee_pct"] / 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
		record: dict[str, Any] = {
			"id": loan_id,
			"type": "sacco_loan",
			"tenant_id": t,
			"loan_number": loan_number,
			"member_id": member_id,
			"product_id": product_id,
			"product_code": product.get("product_code"),
			"product_name": product.get("product_name"),
			"amount_requested": amount,
			"amount_approved": None,
			"amount_disbursed": None,
			"outstanding_balance": Decimal("0"),
			"term_months_requested": term_months,
			"term_months_approved": None,
			"interest_rate_pa": product["interest_rate_pa"],
			"interest_method": product["interest_method"],
			"purpose": purpose,
			"processing_fee": processing_fee,
			"insurance_fee": insurance_fee,
			"guarantor_ids": guarantor_list,
			"collateral_description": collateral_description,
			"collateral_value": Decimal(str(collateral_value)) if collateral_value else None,
			"total_repaid": Decimal("0"),
			"arrears_days": 0,
			"arrears_amount": Decimal("0"),
			"disbursement_method": None,
			"disbursed_at": None,
			"status": "pending",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.loans[loan_id] = record
		# Register guarantors
		for gid in guarantor_list:
			gr_id = self._record_id("gr")
			self.guarantors[gr_id] = {
				"id": gr_id,
				"type": "sacco_loan_guarantor",
				"tenant_id": t,
				"loan_id": loan_id,
				"guarantor_member_id": gid,
				"status": "active",
				"created_at": self._now(),
			}
		self._emit(t, "loan_application_submitted", record)
		_log.info("Loan application submitted: %s member=%s amount=%s", loan_number, member_id, amount)
		return deepcopy(record)

	async def approve_loan(
		self,
		loan_id: str,
		approved_amount: float,
		approved_term_months: int,
		approved_by: str,
		tenant_id: str | None = None,
		approved_rate: float | None = None,
		approval_notes: str | None = None,
		conditions: list[str] | None = None,
	) -> dict[str, Any]:
		"""Approve a loan application."""
		t = self._tenant(tenant_id)
		loan = self._get_loan(loan_id, t)
		if loan["status"] != "pending":
			raise ValueError(f"cannot_approve_loan_in_status: {loan['status']}")
		loan["amount_approved"] = Decimal(str(approved_amount))
		loan["term_months_approved"] = approved_term_months
		if approved_rate is not None:
			loan["interest_rate_pa"] = Decimal(str(approved_rate))
		loan["approved_by"] = approved_by
		loan["approval_notes"] = approval_notes
		loan["conditions"] = conditions or []
		loan["approved_at"] = self._now()
		loan["status"] = "approved"
		loan["updated_at"] = self._now()
		# Build repayment schedule
		schedule = self._build_reducing_balance_schedule(
			principal=loan["amount_approved"],
			annual_rate=loan["interest_rate_pa"],
			term_months=approved_term_months,
			start_date=self._now()[:10],
			grace_months=self.products.get(loan["product_id"], {}).get("grace_period_months", 0),
		)
		self.schedules[loan_id] = schedule
		self._emit(t, "loan_approved", loan)
		return deepcopy(loan)

	async def reject_loan(self, loan_id: str, rejected_by: str, rejection_reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Reject a pending loan application."""
		t = self._tenant(tenant_id)
		loan = self._get_loan(loan_id, t)
		if loan["status"] != "pending":
			raise ValueError(f"cannot_reject_loan_in_status: {loan['status']}")
		loan["status"] = "rejected"
		loan["rejected_by"] = rejected_by
		loan["rejection_reason"] = rejection_reason
		loan["rejected_at"] = self._now()
		loan["updated_at"] = self._now()
		self._emit(t, "loan_rejected", loan)
		return deepcopy(loan)

	async def disburse_loan(
		self,
		loan_id: str,
		disbursement_method: str,
		disbursement_reference: str,
		disbursed_by: str,
		tenant_id: str | None = None,
		disbursement_account: str | None = None,
	) -> dict[str, Any]:
		"""Disburse an approved loan to the member."""
		t = self._tenant(tenant_id)
		loan = self._get_loan(loan_id, t)
		if loan["status"] != "approved":
			raise ValueError(f"cannot_disburse_loan_in_status: {loan['status']}")
		if disbursement_method not in DISBURSEMENT_METHODS:
			raise ValueError(f"invalid_disbursement_method: {disbursement_method}")
		amount = loan["amount_approved"]
		loan["amount_disbursed"] = amount
		loan["outstanding_balance"] = amount
		loan["disbursement_method"] = disbursement_method
		loan["disbursement_reference"] = disbursement_reference
		loan["disbursement_account"] = disbursement_account
		loan["disbursed_by"] = disbursed_by
		loan["disbursed_at"] = self._now()
		loan["status"] = "active"
		loan["updated_at"] = self._now()
		self._emit(t, "loan_disbursed", loan)
		_log.info("Loan disbursed: %s amount=%s method=%s", loan_id, amount, disbursement_method)
		return deepcopy(loan)

	# ── Repayments ────────────────────────────────────────────────────────────

	async def record_repayment(
		self,
		loan_id: str,
		amount: float,
		payment_reference: str,
		recorded_by: str,
		tenant_id: str | None = None,
		payment_method: str = "cash",
		payment_date: str | None = None,
	) -> dict[str, Any]:
		"""Record a loan repayment."""
		t = self._tenant(tenant_id)
		loan = self._get_loan(loan_id, t)
		if loan["status"] not in {"active", "arrears"}:
			raise ValueError(f"cannot_repay_loan_in_status: {loan['status']}")
		repayment_amount = Decimal(str(amount))
		if repayment_amount <= 0:
			raise ValueError("amount_must_be_positive")
		txn_id = self._record_id("rep")
		txn: dict[str, Any] = {
			"id": txn_id,
			"type": "sacco_loan_repayment",
			"tenant_id": t,
			"loan_id": loan_id,
			"loan_number": loan.get("loan_number"),
			"member_id": loan.get("member_id"),
			"amount": repayment_amount,
			"outstanding_before": loan["outstanding_balance"],
			"outstanding_after": max(Decimal("0"), loan["outstanding_balance"] - repayment_amount),
			"payment_reference": payment_reference,
			"payment_method": payment_method,
			"payment_date": payment_date or self._now()[:10],
			"recorded_by": recorded_by,
			"status": "completed",
			"created_at": self._now(),
		}
		self.repayments[txn_id] = txn
		loan["outstanding_balance"] = txn["outstanding_after"]
		loan["total_repaid"] = loan.get("total_repaid", Decimal("0")) + repayment_amount
		if loan["outstanding_balance"] <= 0:
			loan["status"] = "closed"
			loan["closed_at"] = self._now()
		elif loan.get("arrears_days", 0) > 0:
			# Reduce arrears amount
			loan["arrears_amount"] = max(Decimal("0"), loan.get("arrears_amount", Decimal("0")) - repayment_amount)
			if loan["arrears_amount"] == 0:
				loan["status"] = "active"
				loan["arrears_days"] = 0
		loan["updated_at"] = self._now()
		# Update schedule: mark earliest pending installment paid
		schedule = self.schedules.get(loan_id, [])
		for inst in schedule:
			if inst["status"] == "pending" and inst["total_due"] <= repayment_amount:
				inst["status"] = "paid"
				inst["paid_at"] = self._now()
				break
		self._emit(t, "loan_repayment_recorded", txn)
		return deepcopy(txn)

	async def list_repayments(self, loan_id: str | None = None, member_id: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.repayments.values() if r["tenant_id"] == t]
		if loan_id:
			items = [r for r in items if r["loan_id"] == loan_id]
		if member_id:
			items = [r for r in items if r.get("member_id") == member_id]
		return items

	async def get_repayment_schedule(self, loan_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		loan = self._get_loan(loan_id, t)
		schedule = self.schedules.get(loan_id, [])
		total_principal = sum(s["principal"] for s in schedule)
		total_interest = sum(s["interest"] for s in schedule)
		return {
			"loan_id": loan_id,
			"loan_number": loan.get("loan_number"),
			"member_id": loan.get("member_id"),
			"amount_disbursed": str(loan.get("amount_disbursed", 0)),
			"outstanding_balance": str(loan.get("outstanding_balance", 0)),
			"installments": schedule,
			"total_principal": str(total_principal),
			"total_interest": str(total_interest),
			"total_payable": str(total_principal + total_interest),
			"generated_at": self._now(),
		}

	# ── Loans listing ─────────────────────────────────────────────────────────

	async def list_loans(
		self,
		tenant_id: str | None = None,
		member_id: str | None = None,
		product_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(l) for l in self.loans.values() if l["tenant_id"] == t]
		if member_id:
			items = [l for l in items if l["member_id"] == member_id]
		if product_id:
			items = [l for l in items if l["product_id"] == product_id]
		if status:
			items = [l for l in items if l["status"] == status]
		return items

	async def get_loan(self, loan_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		return deepcopy(self._get_loan(loan_id, t))

	async def update_loan(self, loan_id: str, tenant_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		loan = self._get_loan(loan_id, t)
		allowed = {"purpose", "collateral_description", "collateral_value"}
		for k, v in kwargs.items():
			if k in allowed:
				loan[k] = v
		loan["updated_at"] = self._now()
		self._emit(t, "loan_updated", loan)
		return deepcopy(loan)

	async def delete_loan(self, loan_id: str, tenant_id: str | None = None, reason: str = "admin_cancel") -> dict[str, Any]:
		"""Cancel a pending loan application."""
		t = self._tenant(tenant_id)
		loan = self._get_loan(loan_id, t)
		if loan["status"] not in {"pending"}:
			raise ValueError(f"can_only_cancel_pending_loans")
		loan["status"] = "cancelled"
		loan["cancellation_reason"] = reason
		loan["cancelled_at"] = self._now()
		loan["updated_at"] = self._now()
		self._emit(t, "loan_cancelled", loan)
		return deepcopy(loan)

	# ── Arrears Management ────────────────────────────────────────────────────

	async def run_arrears_check(self, as_of_date: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Identify overdue loans and update arrears days."""
		t = self._tenant(tenant_id)
		try:
			check_date = date.fromisoformat(as_of_date)
		except Exception:
			check_date = date.today()
		arrears_updated = 0
		total_arrears = Decimal("0")
		for loan in self.loans.values():
			if loan["tenant_id"] != t or loan["status"] not in {"active", "arrears"}:
				continue
			schedule = self.schedules.get(loan["id"], [])
			overdue_installments = [
				s for s in schedule
				if s["status"] == "pending" and date.fromisoformat(s["due_date"]) < check_date
			]
			if not overdue_installments:
				continue
			earliest_due = min(date.fromisoformat(s["due_date"]) for s in overdue_installments)
			days_overdue = (check_date - earliest_due).days
			overdue_amount = sum(s["total_due"] for s in overdue_installments)
			loan["arrears_days"] = days_overdue
			loan["arrears_amount"] = overdue_amount
			loan["status"] = "arrears"
			loan["updated_at"] = self._now()
			arrears_rec: dict[str, Any] = {
				"id": self._record_id("arr"),
				"type": "sacco_arrears_record",
				"tenant_id": t,
				"loan_id": loan["id"],
				"loan_number": loan.get("loan_number"),
				"member_id": loan.get("member_id"),
				"arrears_days": days_overdue,
				"arrears_amount": overdue_amount,
				"overdue_installments": len(overdue_installments),
				"as_of_date": as_of_date,
				"created_at": self._now(),
			}
			self.arrears_records[arrears_rec["id"]] = arrears_rec
			arrears_updated += 1
			total_arrears += overdue_amount
		run_rec = {
			"type": "sacco_arrears_run",
			"tenant_id": t,
			"as_of_date": as_of_date,
			"loans_in_arrears": arrears_updated,
			"total_arrears_amount": str(total_arrears),
			"created_at": self._now(),
		}
		self._emit(t, "arrears_check_run", run_rec)
		return run_rec

	async def list_arrears(self, min_days: int = 1, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(a) for a in self.arrears_records.values() if a["tenant_id"] == t and a.get("arrears_days", 0) >= min_days]

	async def write_off_loan(self, loan_id: str, written_off_by: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Write off an unrecoverable loan."""
		t = self._tenant(tenant_id)
		loan = self._get_loan(loan_id, t)
		if loan["status"] not in {"active", "arrears"}:
			raise ValueError(f"cannot_write_off_loan_in_status: {loan['status']}")
		loan["status"] = "written_off"
		loan["write_off_reason"] = reason
		loan["written_off_by"] = written_off_by
		loan["written_off_at"] = self._now()
		loan["updated_at"] = self._now()
		self._emit(t, "loan_written_off", loan)
		return deepcopy(loan)

	# ── Guarantor Management ──────────────────────────────────────────────────

	async def list_guarantors(self, loan_id: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(g) for g in self.guarantors.values() if g["tenant_id"] == t]
		if loan_id:
			items = [g for g in items if g["loan_id"] == loan_id]
		return items

	async def release_guarantor(self, guarantor_id: str, released_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		gr = self.guarantors.get(guarantor_id)
		if not gr or gr["tenant_id"] != t:
			raise KeyError(f"guarantor_not_found: {guarantor_id}")
		gr["status"] = "released"
		gr["released_by"] = released_by
		gr["released_at"] = self._now()
		self._emit(t, "guarantor_released", gr)
		return deepcopy(gr)

	# ── CRB Reporting ─────────────────────────────────────────────────────────

	async def submit_crb_report(
		self,
		member_id: str,
		report_type: str,
		reason: str,
		reported_by: str,
		tenant_id: str | None = None,
		crb_reference: str | None = None,
	) -> dict[str, Any]:
		"""Submit a CRB listing, delisting, or inquiry."""
		t = self._tenant(tenant_id)
		if report_type not in CRB_REPORT_TYPES:
			raise ValueError(f"invalid_report_type: {report_type}")
		crb_id = self._record_id("crb")
		record: dict[str, Any] = {
			"id": crb_id,
			"type": "sacco_crb_report",
			"tenant_id": t,
			"member_id": member_id,
			"report_type": report_type,
			"reason": reason,
			"reported_by": reported_by,
			"crb_reference": crb_reference,
			"status": "submitted",
			"created_at": self._now(),
		}
		self.crb_reports[crb_id] = record
		self._emit(t, f"crb_{report_type}_submitted", record)
		_log.info("CRB %s submitted: member=%s", report_type, member_id)
		return deepcopy(record)

	async def list_crb_reports(self, member_id: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.crb_reports.values() if r["tenant_id"] == t]
		if member_id:
			items = [r for r in items if r["member_id"] == member_id]
		return items

	# ── Portfolio Summary ─────────────────────────────────────────────────────

	async def portfolio_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		loans = [l for l in self.loans.values() if l["tenant_id"] == t]
		by_status: dict[str, int] = {}
		total_outstanding = Decimal("0")
		total_disbursed = Decimal("0")
		total_arrears = Decimal("0")
		for loan in loans:
			by_status[loan.get("status", "unknown")] = by_status.get(loan.get("status", "unknown"), 0) + 1
			total_outstanding += loan.get("outstanding_balance", Decimal("0"))
			total_disbursed += loan.get("amount_disbursed") or Decimal("0")
			total_arrears += loan.get("arrears_amount", Decimal("0"))
		par_ratio = float(total_arrears / total_outstanding) if total_outstanding > 0 else 0.0
		return {
			"tenant_id": t,
			"total_loans": len(loans),
			"by_status": by_status,
			"total_outstanding_balance": str(total_outstanding),
			"total_disbursed": str(total_disbursed),
			"total_arrears_amount": str(total_arrears),
			"portfolio_at_risk_pct": round(par_ratio * 100, 2),
			"crb_submissions": len([r for r in self.crb_reports.values() if r["tenant_id"] == t]),
			"generated_at": self._now(),
		}

	async def export_loan_book(self, tenant_id: str | None = None, fmt: str = "json") -> dict[str, Any]:
		t = self._tenant(tenant_id)
		assert fmt in {"json", "csv", "excel"}, "fmt must be json|csv|excel"
		count = sum(1 for l in self.loans.values() if l["tenant_id"] == t)
		return {
			"tenant_id": t,
			"format": fmt,
			"record_count": count,
			"export_reference": f"loanbook-{t}-{self._now()[:10]}.{fmt}",
			"generated_at": self._now(),
		}

	async def member_loan_summary(self, member_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return all loans and repayment history for a member."""
		t = self._tenant(tenant_id)
		loans = [deepcopy(l) for l in self.loans.values() if l["tenant_id"] == t and l["member_id"] == member_id]
		repayments = [deepcopy(r) for r in self.repayments.values() if r["tenant_id"] == t and r.get("member_id") == member_id]
		guarantor_entries = [deepcopy(g) for g in self.guarantors.values() if g["tenant_id"] == t and g["guarantor_member_id"] == member_id]
		credit_score = await self.get_latest_credit_score(member_id, tenant_id=t)
		return {
			"member_id": member_id,
			"loans": loans,
			"total_outstanding": str(sum(l.get("outstanding_balance", Decimal("0")) for l in loans)),
			"total_repaid": str(sum(l.get("total_repaid", Decimal("0")) for l in loans)),
			"repayments": repayments,
			"active_guarantees": [g for g in guarantor_entries if g["status"] == "active"],
			"credit_score": credit_score,
			"generated_at": self._now(),
		}


# Alias
LendingService = SaccoLendingService
