"""SACCO Check-off Management — full async service.

Check-off is the mechanism by which employers deduct loan repayments and
savings contributions from employee salaries before payment and remit to
the SACCO.  This accounts for ~80% of SACCO loan collections in East Africa.

Architecture:
- In-memory dicts keyed by (tenant_id, record_id) — swap for DB in production.
- Idempotency on post_check_off_receipts keyed on (tenant_id, employer_id, month, year).
- All monetary arithmetic uses Python Decimal with ROUND_HALF_UP.
- _log_* helper methods for structured console/audit output.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from calendar import month_name
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

from .models import (
	CheckOffMetrics,
	CheckOffSchedule,
	CheckOffStatus,
	DeductionFrequency,
	DeductionLine,
	DeductionType,
	Employer,
	EmployerCreate,
	EmployerStatement,
	EmployerUpdate,
	GLEntry,
	MemberCheckOffEntry,
	MemberCheckOffHistory,
	MemberDeductions,
	MemberEmployerLink,
	MemberReconciliation,
	ReconciliationResult,
	RemittanceRecord,
	RemittanceStatus,
	ScheduleMemberEntry,
	StatementLine,
	UploadedDeduction,
)

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_ckf"

# GL account codes — override via service config in production
GL_CHECKOFF_RECEIVABLE = "1310"   # DR when schedule generated
GL_LOAN_LEDGER = "1410"           # CR on loan repayment receipt
GL_SAVINGS_LEDGER = "2110"        # CR on savings contribution receipt

TWO_PLACES = Decimal("0.01")


def _d(v: Any) -> Decimal:
	return Decimal(str(v)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _period_label(month: int, year: int) -> str:
	return f"{month_name[month]} {year}"


class CheckOffService:
	"""Async service managing employer check-off deductions, reconciliation and GL posting."""

	def __init__(self) -> None:
		# Core stores — keyed by id within dicts
		self._employers = WriteThruDict('employers', tenant_id, _store)
		self._links = WriteThruDict('links', tenant_id, _store)          # member-employer links
		self._schedules = WriteThruDict('schedules', tenant_id, _store)
		self._uploads: dict[str, list[dict[str, Any]]] = {}  # key: "{tid}:{eid}:{y}:{m}"
		self._reconciliations = WriteThruDict('reconciliations', tenant_id, _store)
		self._remittances = WriteThruDict('remittances', tenant_id, _store)    # key: "{tid}:{eid}:{y}:{m}"
		self._gl_entries = WriteThruList('gl_entries', tenant_id, _store)
		self._posted_keys: set[str] = set()                  # idempotency

		# Loan / savings stubs — in production delegate to lnd/dep services
		self._loan_installments: dict[str, list[dict[str, Any]]] = {}
		self._savings_contributions: dict[str, list[dict[str, Any]]] = {}
		self._member_names: dict[str, str] = {}

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_employer_event(self, tenant_id: str, employer_id: str, event: str, detail: str = "") -> None:
		_log.info("[CKF] tenant=%s employer=%s event=%s %s", tenant_id, employer_id, event, detail)

	def _log_reconciliation(self, tenant_id: str, employer_id: str, month: int, year: int,
	                         expected: Decimal, received: Decimal) -> None:
		variance = received - expected
		_log.info(
			"[CKF] reconcile tenant=%s employer=%s period=%s expected=%s received=%s variance=%s",
			tenant_id, employer_id, _period_label(month, year), expected, received, variance,
		)

	def _log_gl_post(self, tenant_id: str, employer_id: str, month: int, year: int, amount: Decimal) -> None:
		_log.info("[CKF] GL posted tenant=%s employer=%s period=%s amount=%s", tenant_id, employer_id, _period_label(month, year), amount)

	def _log_pretty_path(self, path: str) -> str:
		return path.replace("_", "/")

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _remittance_key(self, tenant_id: str, employer_id: str, year: int, month: int) -> str:
		return f"{tenant_id}:{employer_id}:{year}:{month}"

	def _get_employer_record(self, tenant_id: str, employer_id: str) -> dict[str, Any]:
		rec = self._employers.get(employer_id)
		if not rec or rec["tenant_id"] != tenant_id:
			raise KeyError(f"employer_not_found: {employer_id}")
		return rec

	def _get_active_links_for_employer(self, tenant_id: str, employer_id: str) -> list[dict[str, Any]]:
		return [
			lnk for lnk in self._links.values()
			if lnk["tenant_id"] == tenant_id
			and lnk["employer_id"] == employer_id
			and lnk["is_active"]
		]

	def _get_active_link_for_member(self, tenant_id: str, member_id: str) -> dict[str, Any] | None:
		for lnk in self._links.values():
			if lnk["tenant_id"] == tenant_id and lnk["member_id"] == member_id and lnk["is_active"]:
				return lnk
		return None

	def _due_loan_deductions(self, tenant_id: str, member_id: str) -> list[DeductionLine]:
		"""Return outstanding loan installments for this member."""
		lines: list[DeductionLine] = []
		for inst in self._loan_installments.get(f"{tenant_id}:{member_id}", []):
			if inst.get("status") in ("due", "overdue"):
				lines.append(DeductionLine(
					deduction_type=DeductionType.LOAN_PRINCIPAL,
					reference_id=inst["loan_id"],
					description=f"Loan {inst['loan_number']} installment {inst['installment_no']}",
					amount_due=_d(inst["principal_due"]),
				))
				if _d(inst.get("interest_due", 0)) > 0:
					lines.append(DeductionLine(
						deduction_type=DeductionType.LOAN_INTEREST,
						reference_id=inst["loan_id"],
						description=f"Loan {inst['loan_number']} interest",
						amount_due=_d(inst["interest_due"]),
					))
				if _d(inst.get("penalty", 0)) > 0:
					lines.append(DeductionLine(
						deduction_type=DeductionType.LOAN_PENALTY,
						reference_id=inst["loan_id"],
						description=f"Loan {inst['loan_number']} penalty",
						amount_due=_d(inst["penalty"]),
					))
		return lines

	def _due_savings_contributions(self, tenant_id: str, member_id: str) -> list[DeductionLine]:
		"""Return contractual savings contributions for this member."""
		lines: list[DeductionLine] = []
		for contrib in self._savings_contributions.get(f"{tenant_id}:{member_id}", []):
			lines.append(DeductionLine(
				deduction_type=DeductionType.SAVINGS_REGULAR,
				reference_id=contrib["product_id"],
				description=f"Savings: {contrib['product_name']}",
				amount_due=_d(contrib["monthly_amount"]),
			))
		return lines

	def _due_arrears(self, tenant_id: str, member_id: str) -> list[DeductionLine]:
		"""Return check-off arrears — months previously short-paid."""
		lines: list[DeductionLine] = []
		for rem in self._remittances.values():
			if (rem["tenant_id"] != tenant_id
				or rem.get("status") != RemittanceStatus.PARTIAL.value):
				continue
			# Find member variance in that reconciliation
			recon = self._reconciliations.get(self._remittance_key(tenant_id, rem["employer_id"], rem["payroll_year"], rem["payroll_month"]))
			if not recon:
				continue
			for mrec in recon.get("members", []):
				if mrec["member_id"] == member_id and mrec["variance"] < 0:
					lines.append(DeductionLine(
						deduction_type=DeductionType.ARREARS,
						reference_id=rem["id"],
						description=f"Arrears {_period_label(rem['payroll_month'], rem['payroll_year'])}",
						amount_due=abs(_d(mrec["variance"])),
					))
		return lines

	def _ensure_remittance(self, tenant_id: str, employer_id: str, year: int, month: int,
	                        employer_name: str) -> dict[str, Any]:
		key = self._remittance_key(tenant_id, employer_id, year, month)
		if key not in self._remittances:
			self._remittances[key] = RemittanceRecord(
				tenant_id=tenant_id,
				employer_id=employer_id,
				employer_name=employer_name,
				payroll_month=month,
				payroll_year=year,
				period_label=_period_label(month, year),
				amount_expected=Decimal("0"),
				created_at=_now(),
			).model_dump()
		return self._remittances[key]

	# ── Employer Management ───────────────────────────────────────────────────

	async def register_employer(
		self,
		tenant_id: str,
		name: str,
		registration_number: str,
		payroll_contact: str,
		remittance_account: str,
		check_off_agreement_date: str,
		deduction_frequency: DeductionFrequency = DeductionFrequency.MONTHLY,
		email: str | None = None,
		phone: str | None = None,
		address: str | None = None,
		notes: str | None = None,
	) -> Employer:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(name, "name")
		guard_non_empty_string(registration_number, "registration_number")

		# Duplicate check by registration_number within tenant
		for rec in self._employers.values():
			if rec["tenant_id"] == tenant_id and rec["registration_number"] == registration_number:
				raise ValueError(f"employer_already_registered: {registration_number}")

		emp = Employer(
			tenant_id=tenant_id,
			name=name,
			registration_number=registration_number,
			payroll_contact=payroll_contact,
			remittance_account=remittance_account,
			check_off_agreement_date=check_off_agreement_date,
			deduction_frequency=deduction_frequency,
			email=email,
			phone=phone,
			address=address,
			notes=notes,
			created_at=_now(),
			updated_at=_now(),
		)
		self._employers[emp.id] = emp.model_dump()
		self._log_employer_event(tenant_id, emp.id, "registered", emp.name)
		return emp

	async def update_employer(self, tenant_id: str, employer_id: str, updates: dict[str, Any]) -> Employer:
		guard_tenant_id(tenant_id)
		rec = self._get_employer_record(tenant_id, employer_id)
		valid = EmployerUpdate(**updates)
		patch = valid.model_dump(exclude_none=True)
		rec.update(patch)
		rec["updated_at"] = _now()
		self._log_employer_event(tenant_id, employer_id, "updated", str(list(patch.keys())))
		return Employer(**rec)

	async def deactivate_employer(self, tenant_id: str, employer_id: str, reason: str) -> Employer:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(reason, "reason")
		rec = self._get_employer_record(tenant_id, employer_id)
		rec["is_active"] = False
		rec["deactivation_reason"] = reason
		rec["deactivated_at"] = _now()
		rec["updated_at"] = _now()
		self._log_employer_event(tenant_id, employer_id, "deactivated", reason)
		return Employer(**rec)

	async def get_employer(self, tenant_id: str, employer_id: str) -> Employer:
		guard_tenant_id(tenant_id)
		rec = self._get_employer_record(tenant_id, employer_id)
		return Employer(**rec)

	async def list_employers(self, tenant_id: str, active_only: bool = True) -> list[Employer]:
		guard_tenant_id(tenant_id)
		results = []
		for rec in self._employers.values():
			if rec["tenant_id"] != tenant_id:
				continue
			if active_only and not rec["is_active"]:
				continue
			results.append(Employer(**rec))
		return sorted(results, key=lambda e: e.name)

	# ── Member ↔ Employer Links ───────────────────────────────────────────────

	async def add_member_employer_link(
		self,
		tenant_id: str,
		member_id: str,
		employer_id: str,
		employee_number: str,
		basic_salary: Decimal,
		effective_date: str,
		member_name: str = "",
	) -> MemberEmployerLink:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(member_id, "member_id")
		guard_non_empty_string(employee_number, "employee_number")
		assert basic_salary > 0, "basic_salary must be positive"

		# Validate employer exists
		self._get_employer_record(tenant_id, employer_id)

		# Deactivate any existing active link for this member
		existing = self._get_active_link_for_member(tenant_id, member_id)
		if existing:
			existing["is_active"] = False
			existing["end_date"] = effective_date
			existing["end_reason"] = "superseded_by_new_link"

		if member_name:
			self._member_names[f"{tenant_id}:{member_id}"] = member_name

		link = MemberEmployerLink(
			tenant_id=tenant_id,
			member_id=member_id,
			employer_id=employer_id,
			employee_number=employee_number,
			basic_salary=_d(basic_salary),
			effective_date=effective_date,
			created_at=_now(),
		)
		self._links[link.id] = link.model_dump()

		# Update employer member count
		rec = self._employers[employer_id]
		rec["member_count"] = len(self._get_active_links_for_employer(tenant_id, employer_id))

		_log.info("[CKF] linked member=%s employer=%s employee_no=%s", member_id, employer_id, employee_number)
		return link

	async def remove_member_employer_link(
		self,
		tenant_id: str,
		member_id: str,
		employer_id: str,
		effective_date: str,
		reason: str,
	) -> MemberEmployerLink:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(reason, "reason")
		link = self._get_active_link_for_member(tenant_id, member_id)
		if not link or link["employer_id"] != employer_id:
			raise KeyError(f"active_link_not_found: member={member_id} employer={employer_id}")
		link["is_active"] = False
		link["end_date"] = effective_date
		link["end_reason"] = reason

		rec = self._employers.get(employer_id)
		if rec:
			rec["member_count"] = len(self._get_active_links_for_employer(tenant_id, employer_id))

		_log.info("[CKF] unlinked member=%s employer=%s reason=%s", member_id, employer_id, reason)
		return MemberEmployerLink(**link)

	async def get_member_deductions(self, tenant_id: str, member_id: str) -> MemberDeductions:
		guard_tenant_id(tenant_id)
		link = self._get_active_link_for_member(tenant_id, member_id)
		if not link:
			raise KeyError(f"no_active_link: member={member_id}")

		employer = await self.get_employer(tenant_id, link["employer_id"])
		loan_lines = self._due_loan_deductions(tenant_id, member_id)
		savings_lines = self._due_savings_contributions(tenant_id, member_id)
		arrears_lines = self._due_arrears(tenant_id, member_id)

		total_loan = sum(l.amount_due for l in loan_lines)
		total_savings = sum(l.amount_due for l in savings_lines)
		total_arrears = sum(l.amount_due for l in arrears_lines)

		return MemberDeductions(
			member_id=member_id,
			employer_id=employer.id,
			employer_name=employer.name,
			employee_number=link["employee_number"],
			basic_salary=_d(link["basic_salary"]),
			loan_deductions=loan_lines,
			savings_deductions=savings_lines,
			arrears_deductions=arrears_lines,
			total_loan_deductions=_d(total_loan),
			total_savings_deductions=_d(total_savings),
			total_arrears=_d(total_arrears),
			total_deductions=_d(total_loan + total_savings + total_arrears),
		)

	# ── Schedule Generation ───────────────────────────────────────────────────

	async def generate_check_off_schedule(
		self,
		tenant_id: str,
		employer_id: str,
		payroll_month: int,
		payroll_year: int,
	) -> CheckOffSchedule:
		guard_tenant_id(tenant_id)
		assert 1 <= payroll_month <= 12, "payroll_month must be 1-12"
		assert payroll_year >= 2000, "payroll_year invalid"

		employer = await self.get_employer(tenant_id, employer_id)
		links = self._get_active_links_for_employer(tenant_id, employer_id)

		schedule_members: list[ScheduleMemberEntry] = []
		grand_loan = Decimal("0")
		grand_savings = Decimal("0")
		grand_arrears = Decimal("0")

		for lnk in links:
			mid = lnk["member_id"]
			loan_lines = self._due_loan_deductions(tenant_id, mid)
			savings_lines = self._due_savings_contributions(tenant_id, mid)
			arrears_lines = self._due_arrears(tenant_id, mid)
			all_lines = loan_lines + savings_lines + arrears_lines

			total_loan = _d(sum(l.amount_due for l in loan_lines))
			total_savings = _d(sum(l.amount_due for l in savings_lines))
			total_arrears = _d(sum(l.amount_due for l in arrears_lines))
			total = _d(total_loan + total_savings + total_arrears)

			member_name = self._member_names.get(f"{tenant_id}:{mid}", mid)
			schedule_members.append(ScheduleMemberEntry(
				member_id=mid,
				employee_number=lnk["employee_number"],
				member_name=member_name,
				basic_salary=_d(lnk["basic_salary"]),
				deductions=all_lines,
				total_loan=total_loan,
				total_savings=total_savings,
				total_arrears=total_arrears,
				total_deduction=total,
			))
			grand_loan += total_loan
			grand_savings += total_savings
			grand_arrears += total_arrears

		grand_total = _d(grand_loan + grand_savings + grand_arrears)
		schedule = CheckOffSchedule(
			tenant_id=tenant_id,
			employer_id=employer_id,
			employer_name=employer.name,
			payroll_month=payroll_month,
			payroll_year=payroll_year,
			period_label=_period_label(payroll_month, payroll_year),
			members=schedule_members,
			total_members=len(schedule_members),
			grand_total_loan=_d(grand_loan),
			grand_total_savings=_d(grand_savings),
			grand_total_arrears=_d(grand_arrears),
			grand_total=grand_total,
			status=CheckOffStatus.PENDING,
			generated_at=_now(),
		)
		self._schedules[schedule.id] = schedule.model_dump()

		# Create/update remittance record with expected amount
		rem = self._ensure_remittance(tenant_id, employer_id, payroll_year, payroll_month, employer.name)
		rem["amount_expected"] = float(grand_total)
		rem["schedule_id"] = schedule.id

		_log.info("[CKF] schedule generated employer=%s period=%s total=%s members=%d",
		          employer.name, schedule.period_label, grand_total, len(schedule_members))
		return schedule

	# ── Upload ────────────────────────────────────────────────────────────────

	async def upload_check_off_file(
		self,
		tenant_id: str,
		employer_id: str,
		payroll_month: int,
		payroll_year: int,
		deductions: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Accept employer's payroll deduction file.

		Each entry in `deductions` should match UploadedDeduction schema:
		  {member_id, amount_received, loan_deductions, savings_deductions, ...}
		"""
		guard_tenant_id(tenant_id)
		employer = await self.get_employer(tenant_id, employer_id)

		validated = [UploadedDeduction(**d).model_dump() for d in deductions]
		key = self._remittance_key(tenant_id, employer_id, payroll_year, payroll_month)
		self._uploads[key] = validated

		rem = self._ensure_remittance(tenant_id, employer_id, payroll_year, payroll_month, employer.name)
		rem["check_off_status"] = CheckOffStatus.UPLOADED.value
		rem["upload_at"] = _now()
		total_received = _d(sum(_d(d["amount_received"]) for d in validated))
		rem["amount_received"] = float(total_received)

		_log.info("[CKF] file uploaded employer=%s period=%s rows=%d total=%s",
		          employer.name, _period_label(payroll_month, payroll_year), len(validated), total_received)
		return {
			"employer_id": employer_id,
			"period": _period_label(payroll_month, payroll_year),
			"rows_accepted": len(validated),
			"total_received": str(total_received),
			"status": CheckOffStatus.UPLOADED.value,
		}

	# ── Reconciliation ────────────────────────────────────────────────────────

	async def reconcile_check_off(
		self,
		tenant_id: str,
		employer_id: str,
		payroll_month: int,
		payroll_year: int,
	) -> ReconciliationResult:
		guard_tenant_id(tenant_id)
		employer = await self.get_employer(tenant_id, employer_id)
		key = self._remittance_key(tenant_id, employer_id, payroll_year, payroll_month)

		uploaded = self._uploads.get(key, [])
		if not uploaded:
			raise ValueError(f"no_upload_found: upload file first for {_period_label(payroll_month, payroll_year)}")

		# Build expected deductions map by member
		links = self._get_active_links_for_employer(tenant_id, employer_id)
		expected_map: dict[str, Decimal] = {}
		for lnk in links:
			mid = lnk["member_id"]
			loan = _d(sum(l.amount_due for l in self._due_loan_deductions(tenant_id, mid)))
			savings = _d(sum(l.amount_due for l in self._due_savings_contributions(tenant_id, mid)))
			arrears = _d(sum(l.amount_due for l in self._due_arrears(tenant_id, mid)))
			expected_map[mid] = _d(loan + savings + arrears)

		# Build received map from upload
		received_map: dict[str, Decimal] = {}
		for entry in uploaded:
			received_map[entry["member_id"]] = _d(entry["amount_received"])

		all_members = set(expected_map) | set(received_map)
		member_results: list[MemberReconciliation] = []
		total_expected = Decimal("0")
		total_received = Decimal("0")
		short_payers: list[str] = []
		over_payers: list[str] = []

		for mid in all_members:
			exp = expected_map.get(mid, Decimal("0"))
			rec = received_map.get(mid, Decimal("0"))
			variance = _d(rec - exp)
			lnk_rec = self._get_active_link_for_member(tenant_id, mid)
			emp_no = lnk_rec["employee_number"] if lnk_rec else mid
			member_results.append(MemberReconciliation(
				member_id=mid,
				employee_number=emp_no,
				expected_total=exp,
				received_total=rec,
				variance=variance,
				is_fully_paid=variance >= 0,
			))
			total_expected += exp
			total_received += rec
			if variance < 0:
				short_payers.append(mid)
			elif variance > 0:
				over_payers.append(mid)

		total_variance = _d(total_received - total_expected)
		demand_needed = total_variance < 0

		# Allocate excess to savings if over-paid
		excess_to_savings = _d(max(Decimal("0"), total_variance))

		if demand_needed:
			status = CheckOffStatus.SHORT_PAID
		elif total_variance > 0:
			status = CheckOffStatus.OVER_PAID
		else:
			status = CheckOffStatus.RECONCILED

		result = ReconciliationResult(
			tenant_id=tenant_id,
			employer_id=employer_id,
			employer_name=employer.name,
			payroll_month=payroll_month,
			payroll_year=payroll_year,
			status=status,
			members=member_results,
			total_expected=_d(total_expected),
			total_received=_d(total_received),
			total_variance=total_variance,
			short_paying_members=short_payers,
			over_paying_members=over_payers,
			demand_notice_required=demand_needed,
			excess_to_savings=excess_to_savings,
			reconciled_at=_now(),
		)
		self._reconciliations[key] = result.model_dump()

		# Update remittance record
		rem = self._ensure_remittance(tenant_id, employer_id, payroll_year, payroll_month, employer.name)
		rem["check_off_status"] = status.value
		rem["reconciliation_id"] = result.id
		rem["reconciled_at"] = _now()
		if demand_needed:
			rem["status"] = RemittanceStatus.PARTIAL.value
		elif total_variance > 0:
			rem["status"] = RemittanceStatus.PARTIAL.value
		else:
			rem["status"] = RemittanceStatus.RECEIVED.value

		self._log_reconciliation(tenant_id, employer_id, payroll_month, payroll_year,
		                          _d(total_expected), _d(total_received))
		return result

	# ── GL Posting ────────────────────────────────────────────────────────────

	async def post_check_off_receipts(
		self,
		tenant_id: str,
		employer_id: str,
		payroll_month: int,
		payroll_year: int,
	) -> dict[str, Any]:
		"""Post GL entries for all member deductions.  IDEMPOTENT.

		DR Check-off Receivable / CR Loan Ledger  (for loan portion)
		DR Check-off Receivable / CR Savings Ledger (for savings portion)
		"""
		guard_tenant_id(tenant_id)
		post_key = self._remittance_key(tenant_id, employer_id, payroll_year, payroll_month)
		if post_key in self._posted_keys:
			_log.warning("[CKF] duplicate post attempt blocked key=%s", post_key)
			return {"status": "already_posted", "key": post_key}

		key = post_key
		recon_data = self._reconciliations.get(key)
		if not recon_data:
			raise ValueError(f"reconcile_first: no reconciliation for {_period_label(payroll_month, payroll_year)}")

		employer = await self.get_employer(tenant_id, employer_id)
		entry_date = _now()[:10]
		ref = f"CKF-{employer_id[:8]}-{payroll_year}-{payroll_month:02d}"
		entries_created = 0
		total_posted = Decimal("0")

		uploaded = self._uploads.get(key, [])
		received_by_member = {u["member_id"]: u for u in uploaded}

		for mrec in recon_data["members"]:
			mid = mrec["member_id"]
			upload_entry = received_by_member.get(mid, {})
			loan_amt = _d(upload_entry.get("loan_deductions", mrec.get("received_total", 0)))
			savings_amt = _d(upload_entry.get("savings_deductions", 0))

			if loan_amt > 0:
				gl = GLEntry(
					tenant_id=tenant_id,
					entry_date=entry_date,
					description=f"Loan repayment via check-off {_period_label(payroll_month, payroll_year)}",
					debit_account=GL_CHECKOFF_RECEIVABLE,
					credit_account=GL_LOAN_LEDGER,
					amount=loan_amt,
					member_id=mid,
					employer_id=employer_id,
					reference=ref,
					deduction_type=DeductionType.LOAN_PRINCIPAL,
					created_at=_now(),
				)
				self._gl_entries.append(gl.model_dump())
				entries_created += 1
				total_posted += loan_amt

			if savings_amt > 0:
				gl = GLEntry(
					tenant_id=tenant_id,
					entry_date=entry_date,
					description=f"Savings contribution via check-off {_period_label(payroll_month, payroll_year)}",
					debit_account=GL_CHECKOFF_RECEIVABLE,
					credit_account=GL_SAVINGS_LEDGER,
					amount=savings_amt,
					member_id=mid,
					employer_id=employer_id,
					reference=ref,
					deduction_type=DeductionType.SAVINGS_REGULAR,
					created_at=_now(),
				)
				self._gl_entries.append(gl.model_dump())
				entries_created += 1
				total_posted += savings_amt

		self._posted_keys.add(post_key)

		rem = self._ensure_remittance(tenant_id, employer_id, payroll_year, payroll_month, employer.name)
		rem["check_off_status"] = CheckOffStatus.POSTED.value
		rem["status"] = RemittanceStatus.RECEIVED.value
		rem["amount_posted"] = float(_d(total_posted))
		rem["posted_at"] = _now()

		self._log_gl_post(tenant_id, employer_id, payroll_month, payroll_year, _d(total_posted))
		return {
			"status": "posted",
			"employer_id": employer_id,
			"period": _period_label(payroll_month, payroll_year),
			"gl_entries_created": entries_created,
			"total_posted": str(_d(total_posted)),
			"reference": ref,
		}

	# ── Status & Queries ──────────────────────────────────────────────────────

	async def get_check_off_status(
		self,
		tenant_id: str,
		employer_id: str,
		payroll_month: int,
		payroll_year: int,
	) -> dict[str, Any]:
		guard_tenant_id(tenant_id)
		employer = await self.get_employer(tenant_id, employer_id)
		key = self._remittance_key(tenant_id, employer_id, payroll_year, payroll_month)
		rem = self._remittances.get(key)
		recon = self._reconciliations.get(key)
		upload = self._uploads.get(key)

		return {
			"employer_id": employer_id,
			"employer_name": employer.name,
			"period": _period_label(payroll_month, payroll_year),
			"remittance": rem,
			"reconciliation_summary": {
				"total_expected": recon["total_expected"] if recon else None,
				"total_received": recon["total_received"] if recon else None,
				"variance": recon["total_variance"] if recon else None,
				"demand_notice_required": recon["demand_notice_required"] if recon else None,
			} if recon else None,
			"upload_rows": len(upload) if upload else 0,
			"is_posted": key in self._posted_keys,
		}

	async def get_outstanding_remittances(self, tenant_id: str) -> list[dict[str, Any]]:
		guard_tenant_id(tenant_id)
		results = []
		for rem in self._remittances.values():
			if rem["tenant_id"] != tenant_id:
				continue
			if rem.get("status") in (RemittanceStatus.OUTSTANDING.value, RemittanceStatus.PARTIAL.value,
			                          RemittanceStatus.OVERDUE.value):
				results.append(rem)
		return sorted(results, key=lambda r: (r["payroll_year"], r["payroll_month"]))

	async def send_remittance_reminder(
		self,
		tenant_id: str,
		employer_id: str,
		payroll_month: int,
		payroll_year: int | None = None,
	) -> dict[str, Any]:
		guard_tenant_id(tenant_id)
		year = payroll_year or datetime.utcnow().year
		employer = await self.get_employer(tenant_id, employer_id)
		key = self._remittance_key(tenant_id, employer_id, year, payroll_month)
		rem = self._remittances.get(key)
		if not rem:
			raise KeyError(f"remittance_not_found: {_period_label(payroll_month, year)}")

		rem["reminders_sent"] = rem.get("reminders_sent", 0) + 1
		rem["last_reminder_at"] = _now()
		if rem.get("status") == RemittanceStatus.OUTSTANDING.value:
			rem["status"] = RemittanceStatus.OVERDUE.value

		_log.info("[CKF] reminder sent employer=%s period=%s count=%d",
		          employer.name, _period_label(payroll_month, year), rem["reminders_sent"])
		return {
			"employer_id": employer_id,
			"employer_name": employer.name,
			"period": _period_label(payroll_month, year),
			"reminders_sent": rem["reminders_sent"],
			"contact": employer.payroll_contact,
			"email": employer.email,
		}

	async def generate_employer_statement(
		self,
		tenant_id: str,
		employer_id: str,
		from_month: int,
		to_month: int,
		from_year: int | None = None,
		to_year: int | None = None,
	) -> EmployerStatement:
		guard_tenant_id(tenant_id)
		employer = await self.get_employer(tenant_id, employer_id)
		now_year = datetime.utcnow().year
		fy = from_year or now_year
		ty = to_year or now_year

		lines: list[StatementLine] = []
		total_exp = Decimal("0")
		total_rec = Decimal("0")

		for rem in self._remittances.values():
			if rem["tenant_id"] != tenant_id or rem["employer_id"] != employer_id:
				continue
			# Check period range
			rem_year = rem["payroll_year"]
			rem_month = rem["payroll_month"]
			if not (fy <= rem_year <= ty):
				continue
			if rem_year == fy and rem_month < from_month:
				continue
			if rem_year == ty and rem_month > to_month:
				continue
			exp = _d(rem.get("amount_expected", 0))
			rec = _d(rem.get("amount_received", 0))
			lines.append(StatementLine(
				period_label=rem["period_label"],
				payroll_month=rem_month,
				payroll_year=rem_year,
				amount_expected=exp,
				amount_received=rec,
				variance=_d(rec - exp),
				status=CheckOffStatus(rem.get("check_off_status", CheckOffStatus.PENDING.value)),
				posted_at=rem.get("posted_at"),
			))
			total_exp += exp
			total_rec += rec

		lines.sort(key=lambda l: (l.payroll_year, l.payroll_month))
		return EmployerStatement(
			employer_id=employer_id,
			employer_name=employer.name,
			tenant_id=tenant_id,
			from_period=_period_label(from_month, fy),
			to_period=_period_label(to_month, ty),
			lines=lines,
			total_expected=_d(total_exp),
			total_received=_d(total_rec),
			total_variance=_d(total_rec - total_exp),
			generated_at=_now(),
		)

	async def get_member_check_off_history(
		self,
		tenant_id: str,
		member_id: str,
		months: int = 12,
	) -> MemberCheckOffHistory:
		guard_tenant_id(tenant_id)
		link = self._get_active_link_for_member(tenant_id, member_id)
		employer_id = link["employer_id"] if link else None
		employer_name = None
		if employer_id:
			try:
				emp = await self.get_employer(tenant_id, employer_id)
				employer_name = emp.name
			except KeyError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		entries: list[MemberCheckOffEntry] = []
		total_loan = Decimal("0")
		total_savings = Decimal("0")

		for key, recon in self._reconciliations.items():
			if not key.startswith(f"{tenant_id}:"):
				continue
			for mrec in recon.get("members", []):
				if mrec["member_id"] != member_id:
					continue
				emp_id = recon["employer_id"]
				rem_key = self._remittance_key(tenant_id, emp_id, recon["payroll_year"], recon["payroll_month"])
				rem = self._remittances.get(rem_key, {})
				upload = self._uploads.get(rem_key, [])
				upload_entry = next((u for u in upload if u["member_id"] == member_id), {})
				loan_amt = _d(upload_entry.get("loan_deductions", 0))
				sav_amt = _d(upload_entry.get("savings_deductions", 0))
				status = CheckOffStatus(recon.get("status", CheckOffStatus.PENDING.value))
				try:
					emp = await self.get_employer(tenant_id, emp_id)
					e_name = emp.name
				except KeyError:
					e_name = emp_id
				entries.append(MemberCheckOffEntry(
					period_label=_period_label(recon["payroll_month"], recon["payroll_year"]),
					payroll_month=recon["payroll_month"],
					payroll_year=recon["payroll_year"],
					employer_name=e_name,
					loan_deducted=loan_amt,
					savings_deducted=sav_amt,
					total_deducted=_d(loan_amt + sav_amt),
					status=status,
				))
				total_loan += loan_amt
				total_savings += sav_amt

		entries.sort(key=lambda e: (e.payroll_year, e.payroll_month), reverse=True)
		entries = entries[:months]

		return MemberCheckOffHistory(
			member_id=member_id,
			tenant_id=tenant_id,
			employer_id=employer_id,
			employer_name=employer_name,
			entries=entries,
			total_loan_deducted=_d(total_loan),
			total_savings_deducted=_d(total_savings),
			months_covered=len(entries),
		)

	async def flag_employer_default(
		self,
		tenant_id: str,
		employer_id: str,
		defaulted_month: int,
		defaulted_year: int | None = None,
	) -> dict[str, Any]:
		guard_tenant_id(tenant_id)
		employer = await self.get_employer(tenant_id, employer_id)
		year = defaulted_year or datetime.utcnow().year
		key = self._remittance_key(tenant_id, employer_id, year, defaulted_month)
		rem = self._ensure_remittance(tenant_id, employer_id, year, defaulted_month, employer.name)
		rem["defaulted"] = True
		rem["defaulted_at"] = _now()
		rem["check_off_status"] = CheckOffStatus.DEFAULTED.value
		rem["status"] = RemittanceStatus.OVERDUE.value
		self._log_employer_event(tenant_id, employer_id, "defaulted", _period_label(defaulted_month, year))
		return {
			"employer_id": employer_id,
			"employer_name": employer.name,
			"defaulted_period": _period_label(defaulted_month, year),
			"flagged_at": rem["defaulted_at"],
		}

	async def get_check_off_metrics(self, tenant_id: str, payroll_month: int | None = None, payroll_year: int | None = None) -> CheckOffMetrics:
		guard_tenant_id(tenant_id)
		now = datetime.utcnow()
		month = payroll_month or now.month
		year = payroll_year or now.year

		employers = await self.list_employers(tenant_id, active_only=False)
		active_employers = [e for e in employers if e.is_active]
		defaulted = [e for e in employers if not e.is_active]

		total_members = sum(
			len(self._get_active_links_for_employer(tenant_id, e.id))
			for e in active_employers
		)

		total_expected = Decimal("0")
		total_collected = Decimal("0")
		short_count = 0
		over_count = 0
		compliant_count = 0

		for rem in self._remittances.values():
			if rem["tenant_id"] != tenant_id:
				continue
			if rem["payroll_year"] != year or rem["payroll_month"] != month:
				continue
			exp = _d(rem.get("amount_expected", 0))
			rec = _d(rem.get("amount_received", 0))
			total_expected += exp
			total_collected += rec
			variance = rec - exp
			if variance < 0:
				short_count += 1
			elif variance > 0:
				over_count += 1
			else:
				compliant_count += 1

		collection_rate = (
			_d(total_collected / total_expected * 100)
			if total_expected > 0 else Decimal("0")
		)
		total_rem_with_data = short_count + over_count + compliant_count
		compliance_rate = (
			_d(Decimal(compliant_count) / Decimal(total_rem_with_data) * 100)
			if total_rem_with_data > 0 else Decimal("0")
		)

		return CheckOffMetrics(
			tenant_id=tenant_id,
			period_label=_period_label(month, year),
			total_employers=len(employers),
			active_employers=len(active_employers),
			defaulted_employers=len(defaulted),
			total_members_on_checkoff=total_members,
			collection_rate_pct=collection_rate,
			compliance_rate_pct=compliance_rate,
			total_expected=_d(total_expected),
			total_collected=_d(total_collected),
			total_outstanding=_d(total_expected - total_collected),
			employers_short_paying=short_count,
			employers_over_paying=over_count,
			computed_at=_now(),
		)

	async def batch_process_all_employers(
		self,
		tenant_id: str,
		payroll_month: int,
		payroll_year: int,
	) -> dict[str, Any]:
		"""Generate schedules for all active employers in one pass."""
		guard_tenant_id(tenant_id)
		employers = await self.list_employers(tenant_id, active_only=True)
		results = []
		errors = []
		for emp in employers:
			try:
				sched = await self.generate_check_off_schedule(tenant_id, emp.id, payroll_month, payroll_year)
				results.append({
					"employer_id": emp.id,
					"employer_name": emp.name,
					"members": sched.total_members,
					"grand_total": str(sched.grand_total),
					"schedule_id": sched.id,
				})
			except Exception as exc:
				errors.append({"employer_id": emp.id, "error": str(exc)})
		return {
			"period": _period_label(payroll_month, payroll_year),
			"employers_processed": len(results),
			"employers_failed": len(errors),
			"results": results,
			"errors": errors,
		}

	# ── Stub registration (for testing without lnd/dep services) ─────────────

	def register_loan_installment(self, tenant_id: str, member_id: str, installment: dict[str, Any]) -> None:
		"""Register a loan installment stub for testing / integration."""
		key = f"{tenant_id}:{member_id}"
		self._loan_installments.setdefault(key, []).append(installment)

	def register_savings_contribution(self, tenant_id: str, member_id: str, contribution: dict[str, Any]) -> None:
		"""Register a savings contribution stub for testing / integration."""
		key = f"{tenant_id}:{member_id}"
		self._savings_contributions.setdefault(key, []).append(contribution)

	# ── Health ────────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"capability": CAPABILITY_ID,
			"status": "healthy",
			"employers": len(self._employers),
			"member_links": len(self._links),
			"schedules": len(self._schedules),
			"remittances": len(self._remittances),
			"gl_entries": len(self._gl_entries),
			"posted_periods": len(self._posted_keys),
			"checked_at": _now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_employers', '_links', '_schedules', '_reconciliations', '_remittances', '_gl_entries']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

