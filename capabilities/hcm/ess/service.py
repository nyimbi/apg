"""Employee Self-Service async service."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, date
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "hcm_ess"
LEAVE_TYPES = {"annual", "sick", "maternity", "paternity", "compassionate", "unpaid", "study"}
EXPENSE_CATEGORIES = {"travel", "meals", "accommodation", "supplies", "entertainment", "other"}
BENEFIT_TYPES = {"medical", "dental", "pension", "life_insurance", "gym", "group_life"}
TRAINING_TYPES = {"internal", "external", "online", "conference", "workshop", "certification"}


class ESSService:
	"""Employee Self-Service — leave, payslips, expenses, benefits, training, personal data."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.leave_requests: dict[str, dict[str, Any]] = {}
		self.payslips: dict[str, dict[str, Any]] = {}
		self.personal_data: dict[str, dict[str, Any]] = {}
		self.expense_claims: dict[str, dict[str, Any]] = {}
		self.benefit_enrolments: dict[str, dict[str, Any]] = {}
		self.training_registrations: dict[str, dict[str, Any]] = {}
		self.leave_balances: dict[str, dict[str, float]] = {}
		self._audit_events: list[dict[str, Any]] = []

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _uid(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _emit(self, tenant_id: str, event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"id": self._uid("evt"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"payload": deepcopy(payload),
			"emitted_at": self._now(),
		})

	def _days_between(self, start: str, end: str) -> float:
		try:
			s = date.fromisoformat(start)
			e = date.fromisoformat(end)
			return max(0.0, float((e - s).days + 1))
		except Exception:
			return 1.0

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"leave_requests": len(self.leave_requests),
			"payslips": len(self.payslips),
			"expense_claims": len(self.expense_claims),
			"benefit_enrolments": len(self.benefit_enrolments),
			"training_registrations": len(self.training_registrations),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability contract."""
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "hcm",
			"version": "1.0.0",
			"description": "Employee Self-Service — leave, payslips, expenses, benefits, training",
			"leave_types": sorted(LEAVE_TYPES),
			"expense_categories": sorted(EXPENSE_CATEGORIES),
			"benefit_types": sorted(BENEFIT_TYPES),
			"training_types": sorted(TRAINING_TYPES),
		}

	# ── Audit ─────────────────────────────────────────────────────────────────

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return audit events for the tenant."""
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]

	# ── Leave Requests ────────────────────────────────────────────────────────

	async def create_leave_request(
		self,
		tenant_id: str,
		employee_id: str,
		leave_type: str,
		start_date: str,
		end_date: str,
		reason: str | None = None,
		handover_to: str | None = None,
		attachments: list[str] | None = None,
	) -> dict[str, Any]:
		"""Submit a new leave request."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		if leave_type not in LEAVE_TYPES:
			raise ValueError(f"leave_type must be one of {LEAVE_TYPES}")
		days = self._days_between(start_date, end_date)
		if days <= 0:
			raise ValueError("end_date must be on or after start_date")
		record: dict[str, Any] = {
			"id": self._uid("lv"),
			"tenant_id": t,
			"employee_id": employee_id,
			"leave_type": leave_type,
			"start_date": start_date,
			"end_date": end_date,
			"days_requested": days,
			"reason": reason,
			"handover_to": handover_to,
			"attachments": attachments or [],
			"status": "pending",
			"approved_by": None,
			"rejection_reason": None,
			"created_at": self._now(),
			"updated_at": None,
		}
		self.leave_requests[record["id"]] = record
		self._emit(t, "leave_request_created", "leave_request", record["id"], record)
		_log.info("leave_request created: %s employee=%s", record["id"], employee_id)
		return deepcopy(record)

	async def list_leave_requests(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		status: str | None = None,
		leave_type: str | None = None,
	) -> list[dict[str, Any]]:
		"""List leave requests, optionally filtered."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.leave_requests.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if status:
			items = [r for r in items if r["status"] == status]
		if leave_type:
			items = [r for r in items if r["leave_type"] == leave_type]
		return items

	async def get_leave_request(self, tenant_id: str, request_id: str) -> dict[str, Any]:
		"""Get a single leave request by ID."""
		t = self._tenant(tenant_id)
		record = self.leave_requests.get(request_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"leave_request {request_id} not found")
		return deepcopy(record)

	async def update_leave_request(
		self,
		tenant_id: str,
		request_id: str,
		**kwargs: Any,
	) -> dict[str, Any]:
		"""Update a leave request (e.g. add reason, update dates before approval)."""
		t = self._tenant(tenant_id)
		record = self.leave_requests.get(request_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"leave_request {request_id} not found")
		if record["status"] not in {"pending"}:
			raise PermissionError("only_pending_leave_requests_can_be_updated")
		allowed = {"reason", "handover_to", "attachments", "start_date", "end_date"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		if "start_date" in kwargs or "end_date" in kwargs:
			record["days_requested"] = self._days_between(record["start_date"], record["end_date"])
		record["updated_at"] = self._now()
		self._emit(t, "leave_request_updated", "leave_request", record["id"], record)
		return deepcopy(record)

	async def approve_leave_request(
		self,
		tenant_id: str,
		request_id: str,
		approved_by: str,
	) -> dict[str, Any]:
		"""Approve a pending leave request."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(approved_by, "approved_by")
		record = self.leave_requests.get(request_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"leave_request {request_id} not found")
		if record["status"] != "pending":
			raise PermissionError("only_pending_requests_can_be_approved")
		record["status"] = "approved"
		record["approved_by"] = approved_by
		record["updated_at"] = self._now()
		# Deduct from balance
		emp = record["employee_id"]
		bal = self.leave_balances.setdefault(emp, {})
		lt = record["leave_type"]
		bal[lt] = bal.get(lt, 0.0) - record["days_requested"]
		self._emit(t, "leave_request_approved", "leave_request", record["id"], record)
		return deepcopy(record)

	async def reject_leave_request(
		self,
		tenant_id: str,
		request_id: str,
		rejected_by: str,
		rejection_reason: str,
	) -> dict[str, Any]:
		"""Reject a pending leave request."""
		t = self._tenant(tenant_id)
		record = self.leave_requests.get(request_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"leave_request {request_id} not found")
		if record["status"] != "pending":
			raise PermissionError("only_pending_requests_can_be_rejected")
		record["status"] = "rejected"
		record["approved_by"] = rejected_by
		record["rejection_reason"] = rejection_reason
		record["updated_at"] = self._now()
		self._emit(t, "leave_request_rejected", "leave_request", record["id"], record)
		return deepcopy(record)

	async def cancel_leave_request(self, tenant_id: str, request_id: str) -> dict[str, Any]:
		"""Cancel a pending or approved leave request."""
		t = self._tenant(tenant_id)
		record = self.leave_requests.get(request_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"leave_request {request_id} not found")
		if record["status"] not in {"pending", "approved"}:
			raise PermissionError("cannot_cancel_in_current_state")
		record["status"] = "cancelled"
		record["updated_at"] = self._now()
		self._emit(t, "leave_request_cancelled", "leave_request", record["id"], record)
		return deepcopy(record)

	async def delete_leave_request(self, tenant_id: str, request_id: str) -> bool:
		"""Delete a leave request (only pending/draft)."""
		t = self._tenant(tenant_id)
		record = self.leave_requests.get(request_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"leave_request {request_id} not found")
		if record["status"] != "pending":
			raise PermissionError("only_pending_leave_requests_can_be_deleted")
		del self.leave_requests[request_id]
		self._emit(t, "leave_request_deleted", "leave_request", request_id, {"id": request_id})
		return True

	async def get_leave_balance(self, tenant_id: str, employee_id: str) -> dict[str, float]:
		"""Get leave balance for an employee."""
		self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		defaults: dict[str, float] = {
			"annual": 21.0, "sick": 10.0, "maternity": 90.0,
			"paternity": 14.0, "compassionate": 3.0, "unpaid": 0.0, "study": 5.0,
		}
		bal = self.leave_balances.get(employee_id, {})
		return {lt: round(defaults.get(lt, 0.0) + bal.get(lt, 0.0), 2) for lt in LEAVE_TYPES}

	# ── Payslips ──────────────────────────────────────────────────────────────

	async def generate_payslip(
		self,
		tenant_id: str,
		employee_id: str,
		period_month: int,
		period_year: int,
		gross_pay: float,
		earnings_breakdown: dict[str, float] | None = None,
		deductions_breakdown: dict[str, float] | None = None,
		currency: str = "KES",
		pay_date: str | None = None,
	) -> dict[str, Any]:
		"""Generate and store a payslip for an employee."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		earnings = earnings_breakdown or {"basic_salary": gross_pay}
		deductions = deductions_breakdown or {
			"paye": round(gross_pay * 0.25, 2),
			"nssf": min(200.0, round(gross_pay * 0.006, 2)),
			"nhif": 500.0,
		}
		total_deductions = sum(deductions.values())
		net_pay = gross_pay - total_deductions
		record: dict[str, Any] = {
			"id": self._uid("ps"),
			"tenant_id": t,
			"employee_id": employee_id,
			"period_month": period_month,
			"period_year": period_year,
			"gross_pay": gross_pay,
			"deductions": total_deductions,
			"net_pay": net_pay,
			"currency": currency,
			"earnings_breakdown": earnings,
			"deductions_breakdown": deductions,
			"pay_date": pay_date or self._now()[:10],
			"status": "issued",
			"created_at": self._now(),
		}
		self.payslips[record["id"]] = record
		self._emit(t, "payslip_generated", "payslip", record["id"], {"employee_id": employee_id, "period": f"{period_year}-{period_month:02d}"})
		return deepcopy(record)

	async def list_payslips(
		self,
		tenant_id: str,
		employee_id: str,
		year: int | None = None,
	) -> list[dict[str, Any]]:
		"""List payslips for an employee."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.payslips.values() if r["tenant_id"] == t and r["employee_id"] == employee_id]
		if year:
			items = [r for r in items if r["period_year"] == year]
		return sorted(items, key=lambda r: (r["period_year"], r["period_month"]), reverse=True)

	async def get_payslip(self, tenant_id: str, payslip_id: str) -> dict[str, Any]:
		"""Get a payslip by ID."""
		t = self._tenant(tenant_id)
		record = self.payslips.get(payslip_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"payslip {payslip_id} not found")
		return deepcopy(record)

	# ── Personal Data ─────────────────────────────────────────────────────────

	async def get_personal_data(self, tenant_id: str, employee_id: str) -> dict[str, Any]:
		"""Retrieve personal data record for employee."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		record = self.personal_data.get(f"{t}:{employee_id}")
		if not record:
			raise KeyError(f"personal_data for employee {employee_id} not found")
		return deepcopy(record)

	async def upsert_personal_data(
		self,
		tenant_id: str,
		employee_id: str,
		full_name: str,
		email: str,
		**kwargs: Any,
	) -> dict[str, Any]:
		"""Create or update personal data for an employee."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		key = f"{t}:{employee_id}"
		existing = self.personal_data.get(key, {})
		record = {
			"id": existing.get("id", self._uid("pd")),
			"tenant_id": t,
			"employee_id": employee_id,
			"full_name": full_name,
			"email": email,
			**{k: v for k, v in kwargs.items() if v is not None},
			"updated_at": self._now(),
		}
		if "created_at" not in existing:
			record["created_at"] = self._now()
		else:
			record["created_at"] = existing["created_at"]
		self.personal_data[key] = record
		self._emit(t, "personal_data_updated", "personal_data", record["id"], {"employee_id": employee_id})
		return deepcopy(record)

	async def update_personal_data(
		self,
		tenant_id: str,
		employee_id: str,
		**kwargs: Any,
	) -> dict[str, Any]:
		"""Update specific fields in personal data."""
		t = self._tenant(tenant_id)
		key = f"{t}:{employee_id}"
		record = self.personal_data.get(key)
		if not record:
			raise KeyError(f"personal_data for employee {employee_id} not found")
		allowed = {
			"phone", "emergency_contact_name", "emergency_contact_phone",
			"address_line1", "address_line2", "city", "county", "country",
			"bank_account_number", "bank_name", "bank_branch",
			"nssf_number", "nhif_number", "kra_pin",
		}
		for k, v in kwargs.items():
			if k in allowed:
				record[k] = v
		record["updated_at"] = self._now()
		self._emit(t, "personal_data_updated", "personal_data", record["id"], {"employee_id": employee_id})
		return deepcopy(record)

	# ── Expense Claims ────────────────────────────────────────────────────────

	async def create_expense_claim(
		self,
		tenant_id: str,
		employee_id: str,
		category: str,
		amount: float,
		expense_date: str,
		description: str,
		currency: str = "KES",
		receipts: list[str] | None = None,
		project_code: str | None = None,
	) -> dict[str, Any]:
		"""Submit an expense claim."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		if category not in EXPENSE_CATEGORIES:
			raise ValueError(f"category must be one of {EXPENSE_CATEGORIES}")
		if amount <= 0:
			raise ValueError("amount must be positive")
		record: dict[str, Any] = {
			"id": self._uid("ec"),
			"tenant_id": t,
			"employee_id": employee_id,
			"category": category,
			"amount": amount,
			"currency": currency,
			"expense_date": expense_date,
			"description": description,
			"receipts": receipts or [],
			"project_code": project_code,
			"status": "draft",
			"approved_by": None,
			"rejection_reason": None,
			"paid_at": None,
			"created_at": self._now(),
		}
		self.expense_claims[record["id"]] = record
		self._emit(t, "expense_claim_created", "expense_claim", record["id"], record)
		return deepcopy(record)

	async def list_expense_claims(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List expense claims."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.expense_claims.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def get_expense_claim(self, tenant_id: str, claim_id: str) -> dict[str, Any]:
		"""Get an expense claim by ID."""
		t = self._tenant(tenant_id)
		record = self.expense_claims.get(claim_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"expense_claim {claim_id} not found")
		return deepcopy(record)

	async def update_expense_claim(self, tenant_id: str, claim_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a draft expense claim."""
		t = self._tenant(tenant_id)
		record = self.expense_claims.get(claim_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"expense_claim {claim_id} not found")
		if record["status"] not in {"draft"}:
			raise PermissionError("only_draft_claims_can_be_updated")
		allowed = {"category", "amount", "currency", "expense_date", "description", "receipts", "project_code"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		self._emit(t, "expense_claim_updated", "expense_claim", record["id"], record)
		return deepcopy(record)

	async def submit_expense_claim(self, tenant_id: str, claim_id: str) -> dict[str, Any]:
		"""Submit a draft expense claim for approval."""
		t = self._tenant(tenant_id)
		record = self.expense_claims.get(claim_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"expense_claim {claim_id} not found")
		if record["status"] != "draft":
			raise PermissionError("only_draft_claims_can_be_submitted")
		record["status"] = "submitted"
		self._emit(t, "expense_claim_submitted", "expense_claim", record["id"], record)
		return deepcopy(record)

	async def approve_expense_claim(self, tenant_id: str, claim_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a submitted expense claim."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(approved_by, "approved_by")
		record = self.expense_claims.get(claim_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"expense_claim {claim_id} not found")
		if record["status"] != "submitted":
			raise PermissionError("only_submitted_claims_can_be_approved")
		record["status"] = "approved"
		record["approved_by"] = approved_by
		self._emit(t, "expense_claim_approved", "expense_claim", record["id"], record)
		return deepcopy(record)

	async def delete_expense_claim(self, tenant_id: str, claim_id: str) -> bool:
		"""Delete a draft expense claim."""
		t = self._tenant(tenant_id)
		record = self.expense_claims.get(claim_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"expense_claim {claim_id} not found")
		if record["status"] != "draft":
			raise PermissionError("only_draft_expense_claims_can_be_deleted")
		del self.expense_claims[claim_id]
		self._emit(t, "expense_claim_deleted", "expense_claim", claim_id, {"id": claim_id})
		return True

	# ── Benefits Enrolment ────────────────────────────────────────────────────

	async def enrol_benefit(
		self,
		tenant_id: str,
		employee_id: str,
		benefit_plan_id: str,
		benefit_type: str,
		effective_date: str,
		coverage_tier: str = "individual",
		dependants: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""Enrol an employee in a benefit plan."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		if benefit_type not in BENEFIT_TYPES:
			raise ValueError(f"benefit_type must be one of {BENEFIT_TYPES}")
		contribution_rates = {
			"individual": {"employee": 500.0, "employer": 2000.0},
			"family": {"employee": 1200.0, "employer": 4000.0},
			"spouse": {"employee": 800.0, "employer": 2500.0},
		}
		rates = contribution_rates.get(coverage_tier, contribution_rates["individual"])
		record: dict[str, Any] = {
			"id": self._uid("be"),
			"tenant_id": t,
			"employee_id": employee_id,
			"benefit_plan_id": benefit_plan_id,
			"benefit_type": benefit_type,
			"coverage_tier": coverage_tier,
			"effective_date": effective_date,
			"end_date": None,
			"employee_contribution": rates["employee"],
			"employer_contribution": rates["employer"],
			"dependants": dependants or [],
			"status": "active",
			"created_at": self._now(),
		}
		self.benefit_enrolments[record["id"]] = record
		self._emit(t, "benefit_enrolled", "benefit_enrolment", record["id"], record)
		return deepcopy(record)

	async def list_benefit_enrolments(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		benefit_type: str | None = None,
	) -> list[dict[str, Any]]:
		"""List benefit enrolments."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.benefit_enrolments.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if benefit_type:
			items = [r for r in items if r["benefit_type"] == benefit_type]
		return items

	async def get_benefit_enrolment(self, tenant_id: str, enrolment_id: str) -> dict[str, Any]:
		"""Get a benefit enrolment by ID."""
		t = self._tenant(tenant_id)
		record = self.benefit_enrolments.get(enrolment_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"benefit_enrolment {enrolment_id} not found")
		return deepcopy(record)

	async def update_benefit_enrolment(self, tenant_id: str, enrolment_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update benefit enrolment (coverage tier, dependants)."""
		t = self._tenant(tenant_id)
		record = self.benefit_enrolments.get(enrolment_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"benefit_enrolment {enrolment_id} not found")
		allowed = {"coverage_tier", "dependants", "end_date"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		self._emit(t, "benefit_enrolment_updated", "benefit_enrolment", record["id"], record)
		return deepcopy(record)

	async def terminate_benefit_enrolment(self, tenant_id: str, enrolment_id: str, end_date: str) -> dict[str, Any]:
		"""Terminate a benefit enrolment."""
		t = self._tenant(tenant_id)
		record = self.benefit_enrolments.get(enrolment_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"benefit_enrolment {enrolment_id} not found")
		record["status"] = "terminated"
		record["end_date"] = end_date
		self._emit(t, "benefit_enrolment_terminated", "benefit_enrolment", record["id"], record)
		return deepcopy(record)

	# ── Training Registration ─────────────────────────────────────────────────

	async def register_training(
		self,
		tenant_id: str,
		employee_id: str,
		course_id: str,
		course_name: str,
		training_type: str,
		start_date: str,
		end_date: str,
		provider: str | None = None,
		cost: float = 0.0,
		currency: str = "KES",
		justification: str | None = None,
	) -> dict[str, Any]:
		"""Register an employee for training."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		if training_type not in TRAINING_TYPES:
			raise ValueError(f"training_type must be one of {TRAINING_TYPES}")
		record: dict[str, Any] = {
			"id": self._uid("tr"),
			"tenant_id": t,
			"employee_id": employee_id,
			"course_id": course_id,
			"course_name": course_name,
			"training_type": training_type,
			"start_date": start_date,
			"end_date": end_date,
			"provider": provider,
			"cost": cost,
			"currency": currency,
			"justification": justification,
			"status": "pending",
			"approved_by": None,
			"completion_date": None,
			"certificate_url": None,
			"score": None,
			"created_at": self._now(),
		}
		self.training_registrations[record["id"]] = record
		self._emit(t, "training_registered", "training_registration", record["id"], record)
		return deepcopy(record)

	async def list_training_registrations(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List training registrations."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.training_registrations.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def get_training_registration(self, tenant_id: str, registration_id: str) -> dict[str, Any]:
		"""Get a training registration by ID."""
		t = self._tenant(tenant_id)
		record = self.training_registrations.get(registration_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"training_registration {registration_id} not found")
		return deepcopy(record)

	async def update_training_registration(self, tenant_id: str, registration_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update training registration."""
		t = self._tenant(tenant_id)
		record = self.training_registrations.get(registration_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"training_registration {registration_id} not found")
		allowed = {"status", "approved_by", "completion_date", "certificate_url", "score"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		self._emit(t, "training_registration_updated", "training_registration", record["id"], record)
		return deepcopy(record)

	async def complete_training(
		self,
		tenant_id: str,
		registration_id: str,
		completion_date: str,
		score: float | None = None,
		certificate_url: str | None = None,
	) -> dict[str, Any]:
		"""Mark training as completed."""
		t = self._tenant(tenant_id)
		record = self.training_registrations.get(registration_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"training_registration {registration_id} not found")
		record["status"] = "completed"
		record["completion_date"] = completion_date
		if score is not None:
			record["score"] = score
		if certificate_url:
			record["certificate_url"] = certificate_url
		self._emit(t, "training_completed", "training_registration", record["id"], record)
		return deepcopy(record)

	async def delete_training_registration(self, tenant_id: str, registration_id: str) -> bool:
		"""Delete a pending training registration."""
		t = self._tenant(tenant_id)
		record = self.training_registrations.get(registration_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"training_registration {registration_id} not found")
		if record["status"] != "pending":
			raise PermissionError("only_pending_registrations_can_be_deleted")
		del self.training_registrations[registration_id]
		self._emit(t, "training_registration_deleted", "training_registration", registration_id, {"id": registration_id})
		return True

	# ── Dashboard & Analytics ─────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregated ESS dashboard for a tenant."""
		t = self._tenant(tenant_id)
		pending_leaves = sum(1 for r in self.leave_requests.values() if r["tenant_id"] == t and r["status"] == "pending")
		pending_expenses = sum(1 for r in self.expense_claims.values() if r["tenant_id"] == t and r["status"] == "submitted")
		active_benefits = sum(1 for r in self.benefit_enrolments.values() if r["tenant_id"] == t and r["status"] == "active")
		pending_training = sum(1 for r in self.training_registrations.values() if r["tenant_id"] == t and r["status"] == "pending")
		return {
			"tenant_id": t,
			"leave_requests": {
				"total": sum(1 for r in self.leave_requests.values() if r["tenant_id"] == t),
				"pending": pending_leaves,
			},
			"payslips": sum(1 for r in self.payslips.values() if r["tenant_id"] == t),
			"expense_claims": {
				"total": sum(1 for r in self.expense_claims.values() if r["tenant_id"] == t),
				"pending_approval": pending_expenses,
			},
			"benefit_enrolments": {"active": active_benefits},
			"training_registrations": {
				"total": sum(1 for r in self.training_registrations.values() if r["tenant_id"] == t),
				"pending": pending_training,
			},
			"generated_at": self._now(),
		}

	async def employee_self_service_summary(self, tenant_id: str, employee_id: str) -> dict[str, Any]:
		"""Individual employee ESS summary."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		leave_requests, expense_claims, benefits, training = await asyncio.gather(
			self.list_leave_requests(t, employee_id=employee_id),
			self.list_expense_claims(t, employee_id=employee_id),
			self.list_benefit_enrolments(t, employee_id=employee_id),
			self.list_training_registrations(t, employee_id=employee_id),
			return_exceptions=True,
		)
		return {
			"employee_id": employee_id,
			"leave_balance": await self.get_leave_balance(t, employee_id),
			"leave_requests_pending": sum(1 for r in (leave_requests or []) if not isinstance(r, Exception) and r["status"] == "pending"),
			"expense_claims_pending": sum(1 for r in (expense_claims or []) if not isinstance(r, Exception) and r["status"] in {"draft", "submitted"}),
			"active_benefits": sum(1 for r in (benefits or []) if not isinstance(r, Exception) and r["status"] == "active"),
			"training_in_progress": sum(1 for r in (training or []) if not isinstance(r, Exception) and r["status"] in {"enrolled", "pending"}),
			"generated_at": self._now(),
		}
