"""SACCO Group Lending — full async service.

Group lending (Chama loans, welfare group loans, joint liability) where the
group as a whole is collectively responsible for all borrowing.

© 2025 Datacraft — Author: Nyimbi Odero
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from copy import deepcopy
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

from .models import (
	Group, GroupMember, GroupContribution, GroupLoan, GroupRepayment,
	MerryGoRoundRound, GroupArrearsPosition, MemberArrearsPosition,
	GroupPerformanceScore, GroupSavingsSummary, GroupStatementEntry,
	MerryGoRoundResult, MerryGoRoundScheduleEntry,
	GroupType, GroupRole, ContributionType, GroupLoanStatus, MeetingFrequency,
	GroupStatus, MemberContributionLine, DisbursementInstruction, MemberRepaymentLine,
)

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_grp"

# Grade thresholds (score → grade)
SCORE_GRADES = [(90, "A"), (75, "B"), (55, "C"), (35, "D"), (0, "E")]


class GroupLendingService:
	"""Async service for SACCO group lending: Chamas, welfare groups, merry-go-rounds."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		# In-memory stores keyed by entity id
		self.groups: dict[str, Group] = {}
		self.members: dict[str, GroupMember] = {}            # id -> GroupMember
		self.contributions: dict[str, GroupContribution] = {}
		self.loans: dict[str, GroupLoan] = {}
		self.repayments: dict[str, GroupRepayment] = {}
		self.mgr_rounds: dict[str, MerryGoRoundRound] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		self._loan_counter: int = 0
		self._cache: BoundedCache = BoundedCache(max_size=512)

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _today(self) -> date:
		return datetime.utcnow().date()

	def _record_id(self, prefix: str = "rec") -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _next_loan_number(self, tenant_id: str) -> str:
		self._loan_counter += 1
		return f"GRP-LN-{tenant_id[:4].upper()}-{self._loan_counter:07d}"

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record.get("id", ""),
			"emitted_at": self._now(),
		})

	def _get_group(self, group_id: str, tenant_id: str) -> Group:
		g = self.groups.get(group_id)
		if not g or g.tenant_id != tenant_id:
			raise KeyError(f"group_not_found: {group_id}")
		return g

	def _get_loan(self, loan_id: str, tenant_id: str) -> GroupLoan:
		ln = self.loans.get(loan_id)
		if not ln or ln.tenant_id != tenant_id:
			raise KeyError(f"group_loan_not_found: {loan_id}")
		return ln

	def _group_active_members(self, group_id: str, tenant_id: str) -> list[GroupMember]:
		return [
			m for m in self.members.values()
			if m.group_id == group_id and m.tenant_id == tenant_id and m.active
		]

	def _score_to_grade(self, score: int) -> str:
		for threshold, grade in SCORE_GRADES:
			if score >= threshold:
				return grade
		return "E"

	def _decimal(self, value: Any) -> Decimal:
		return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

	# ── Group registration & management ──────────────────────────────────────

	async def register_group(
		self,
		tenant_id: str,
		name: str,
		group_type: str,
		*,
		registration_number: str | None = None,
		meeting_day: str | None = None,
		meeting_frequency: str = "MONTHLY",
		chairperson_member_id: str | None = None,
		secretary_member_id: str | None = None,
		treasurer_member_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register a new lending group (Chama, Welfare, Merry-Go-Round, Investment)."""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(name, "name")
		guard_non_empty_string(group_type, "group_type")

		group_type_enum = GroupType(group_type.upper())
		freq_enum = MeetingFrequency(meeting_frequency.upper())

		group = Group(
			tenant_id=tenant_id,
			name=name,
			group_type=group_type_enum,
			registration_number=registration_number,
			meeting_day=meeting_day,
			meeting_frequency=freq_enum,
			chairperson_member_id=chairperson_member_id,
			secretary_member_id=secretary_member_id,
			treasurer_member_id=treasurer_member_id,
			metadata=metadata or {},
		)
		self.groups[group.id] = group
		self._emit(tenant_id, "group_registered", group.model_dump())
		_log.info("Registered group %s (%s) for tenant %s", group.id, group_type, tenant_id)
		return group.model_dump()

	async def add_group_member(
		self,
		tenant_id: str,
		group_id: str,
		member_id: str,
		*,
		role: str = "MEMBER",
		joining_date: date | None = None,
		initial_contribution: Decimal | float | str = Decimal("0"),
	) -> dict[str, Any]:
		"""Add a member to a group with an initial contribution."""
		guard_tenant_id(tenant_id)
		group = self._get_group(group_id, tenant_id)
		guard_non_empty_string(member_id, "member_id")

		# Prevent duplicates
		existing = [
			m for m in self.members.values()
			if m.group_id == group_id and m.member_id == member_id and m.active
		]
		if existing:
			raise ValueError(f"member_already_in_group: {member_id}")

		role_enum = GroupRole(role.upper())
		joining = joining_date or self._today()
		init_contrib = self._decimal(initial_contribution)

		gm = GroupMember(
			tenant_id=tenant_id,
			group_id=group_id,
			member_id=member_id,
			role=role_enum,
			joining_date=joining,
			initial_contribution=init_contrib,
			total_contributions=init_contrib,
		)
		self.members[gm.id] = gm

		# Update group officers if relevant
		if role_enum == GroupRole.CHAIRPERSON:
			group.chairperson_member_id = member_id
		elif role_enum == GroupRole.SECRETARY:
			group.secretary_member_id = member_id
		elif role_enum == GroupRole.TREASURER:
			group.treasurer_member_id = member_id

		self._emit(tenant_id, "group_member_added", gm.model_dump())
		return gm.model_dump()

	async def remove_group_member(
		self,
		tenant_id: str,
		group_id: str,
		member_id: str,
		*,
		exit_date: date | None = None,
		reason: str = "",
		payout_amount: Decimal | float | str = Decimal("0"),
	) -> dict[str, Any]:
		"""Remove (exit) a member from a group."""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		gm = next(
			(m for m in self.members.values()
			 if m.group_id == group_id and m.member_id == member_id and m.active),
			None,
		)
		if not gm:
			raise KeyError(f"active_group_member_not_found: {member_id}")

		# Block exit if there's an active loan
		active_loans = [
			ln for ln in self.loans.values()
			if ln.group_id == group_id
			and ln.tenant_id == tenant_id
			and ln.status in (GroupLoanStatus.ACTIVE, GroupLoanStatus.ARREARS, GroupLoanStatus.DISBURSED)
			and member_id in ln.borrower_member_ids
		]
		if active_loans:
			raise ValueError(f"member_has_active_group_loan: settle_loan_first")

		gm.active = False
		gm.exit_date = exit_date or self._today()
		gm.exit_reason = reason
		gm.payout_amount = self._decimal(payout_amount)
		self._emit(tenant_id, "group_member_removed", gm.model_dump())
		return gm.model_dump()

	async def get_group(self, tenant_id: str, group_id: str) -> dict[str, Any]:
		"""Full group profile: members, loan history, savings totals."""
		guard_tenant_id(tenant_id)
		group = self._get_group(group_id, tenant_id)

		members = self._group_active_members(group_id, tenant_id)
		loans = [ln for ln in self.loans.values()
				 if ln.group_id == group_id and ln.tenant_id == tenant_id]
		savings = await self.get_group_savings(tenant_id, group_id)

		result = group.model_dump()
		result["members"] = [m.model_dump() for m in members]
		result["loans"] = [{"id": ln.id, "loan_number": ln.loan_number, "status": ln.status,
							"outstanding_balance": str(ln.outstanding_balance)} for ln in loans]
		result["savings"] = savings
		return result

	async def list_groups(
		self,
		tenant_id: str,
		*,
		group_type: str | None = None,
		active_only: bool = True,
	) -> list[dict[str, Any]]:
		"""List groups for a tenant, optionally filtered by type or status."""
		guard_tenant_id(tenant_id)
		results = []
		for g in self.groups.values():
			if g.tenant_id != tenant_id:
				continue
			if active_only and g.status != GroupStatus.ACTIVE:
				continue
			if group_type and g.group_type.value != group_type.upper():
				continue
			results.append(g.model_dump())
		return results

	async def update_group(
		self,
		tenant_id: str,
		group_id: str,
		updates: dict[str, Any],
	) -> dict[str, Any]:
		"""Partially update mutable group fields."""
		guard_tenant_id(tenant_id)
		group = self._get_group(group_id, tenant_id)

		allowed_fields = {
			"name", "registration_number", "meeting_day", "meeting_frequency",
			"chairperson_member_id", "secretary_member_id", "treasurer_member_id",
			"status", "metadata",
		}
		for k, v in updates.items():
			if k in allowed_fields:
				setattr(group, k, v)
		group.updated_at = self._now()
		self._emit(tenant_id, "group_updated", group.model_dump())
		return group.model_dump()

	# ── Contributions ─────────────────────────────────────────────────────────

	async def record_group_contribution(
		self,
		tenant_id: str,
		group_id: str,
		contributions: list[dict[str, Any]],
		*,
		meeting_date: date | None = None,
		contribution_type: str = "MONTHLY",
		recorded_by: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Record contributions from members at a group meeting.

		Posts each contribution to the member's sub-account within the group.
		contributions = [{"member_id": str, "amount": Decimal}, ...]
		"""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		lines: list[MemberContributionLine] = []
		total = Decimal("0")

		for c in contributions:
			member_id = c["member_id"]
			amount = self._decimal(c["amount"])
			if amount <= 0:
				raise ValueError(f"contribution_amount_must_be_positive: {member_id}")

			gm = next(
				(m for m in self.members.values()
				 if m.group_id == group_id and m.member_id == member_id and m.active),
				None,
			)
			if not gm:
				raise KeyError(f"active_member_not_in_group: {member_id}")

			gm.total_contributions += amount
			total += amount
			lines.append(MemberContributionLine(member_id=member_id, amount=amount))

		contrib = GroupContribution(
			tenant_id=tenant_id,
			group_id=group_id,
			meeting_date=meeting_date or self._today(),
			contribution_type=ContributionType(contribution_type.upper()),
			total_amount=total,
			lines=lines,
			recorded_by=recorded_by,
			notes=notes,
		)
		self.contributions[contrib.id] = contrib
		self._emit(tenant_id, "group_contribution_recorded", contrib.model_dump())
		return contrib.model_dump()

	async def get_group_savings(self, tenant_id: str, group_id: str) -> dict[str, Any]:
		"""Total savings pooled by the group and per-member breakdown."""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		members = self._group_active_members(group_id, tenant_id)
		per_member = []
		total = Decimal("0")
		for m in members:
			per_member.append({
				"member_id": m.member_id,
				"total_contributions": str(m.total_contributions),
				"joining_date": m.joining_date.isoformat(),
			})
			total += m.total_contributions

		return GroupSavingsSummary(
			group_id=group_id,
			total_savings=total,
			per_member=per_member,
		).model_dump()

	async def get_contribution_history(
		self,
		tenant_id: str,
		group_id: str,
		months: int = 12,
	) -> list[dict[str, Any]]:
		"""Contribution records for the group over the last N months."""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		today = self._today()
		from_month = today.month - months
		from_year = today.year + from_month // 12
		from_month = from_month % 12 or 12
		cutoff = date(from_year, from_month, 1)

		return [
			c.model_dump()
			for c in sorted(
				(c for c in self.contributions.values()
				 if c.group_id == group_id and c.tenant_id == tenant_id
				 and c.meeting_date >= cutoff),
				key=lambda c: c.meeting_date,
				reverse=True,
			)
		]

	async def get_contribution_compliance(
		self, tenant_id: str, group_id: str
	) -> dict[str, Any]:
		"""Per-member compliance rate: fraction of expected contribution cycles paid."""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		members = self._group_active_members(group_id, tenant_id)
		# Count total contribution sessions for the group
		sessions = set(c.meeting_date for c in self.contributions.values()
					   if c.group_id == group_id and c.tenant_id == tenant_id)
		total_sessions = len(sessions) or 1

		compliance_data = []
		for m in members:
			paid_sessions = sum(
				1 for c in self.contributions.values()
				if c.group_id == group_id and c.tenant_id == tenant_id
				and any(line.member_id == m.member_id for line in c.lines)
			)
			rate = round(100 * paid_sessions / total_sessions, 2)
			compliance_data.append({
				"member_id": m.member_id,
				"sessions_paid": paid_sessions,
				"total_sessions": total_sessions,
				"compliance_rate_pct": rate,
			})

		overall = (
			round(sum(d["compliance_rate_pct"] for d in compliance_data) / len(compliance_data), 2)
			if compliance_data else 0.0
		)
		return {
			"group_id": group_id,
			"overall_compliance_pct": overall,
			"member_compliance": compliance_data,
		}

	# ── Group loans ───────────────────────────────────────────────────────────

	async def apply_group_loan(
		self,
		tenant_id: str,
		group_id: str,
		requested_amount: Decimal | float | str,
		*,
		purpose: str,
		tenure_months: int,
		applied_by: str,
	) -> dict[str, Any]:
		"""Submit a group loan application; all active members become joint borrowers."""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)
		guard_non_empty_string(purpose, "purpose")
		guard_non_empty_string(applied_by, "applied_by")

		amount = self._decimal(requested_amount)
		if amount <= 0:
			raise ValueError("requested_amount must be positive")
		if tenure_months < 1:
			raise ValueError("tenure_months must be >= 1")

		active_members = self._group_active_members(group_id, tenant_id)
		if not active_members:
			raise ValueError("group_has_no_active_members")

		# Check no other active/pending loan exists for this group
		conflicting = [
			ln for ln in self.loans.values()
			if ln.group_id == group_id and ln.tenant_id == tenant_id
			and ln.status in (GroupLoanStatus.PENDING, GroupLoanStatus.APPROVED,
							  GroupLoanStatus.DISBURSED, GroupLoanStatus.ACTIVE,
							  GroupLoanStatus.ARREARS)
		]
		if conflicting:
			raise ValueError(f"group_has_unresolved_loan: {conflicting[0].id}")

		loan = GroupLoan(
			loan_number=self._next_loan_number(tenant_id),
			tenant_id=tenant_id,
			group_id=group_id,
			borrower_member_ids=[m.member_id for m in active_members],
			requested_amount=amount,
			purpose=purpose,
			tenure_months=tenure_months,
			applied_by=applied_by,
		)
		self.loans[loan.id] = loan
		self._emit(tenant_id, "group_loan_applied", loan.model_dump())
		_log.info("Group loan application %s for group %s, amount=%s", loan.id, group_id, amount)
		return loan.model_dump()

	async def approve_group_loan(
		self,
		tenant_id: str,
		loan_application_id: str,
		*,
		approved_amount: Decimal | float | str,
		approved_by: str,
		conditions: str | None = None,
	) -> dict[str, Any]:
		"""Approve a pending group loan application."""
		guard_tenant_id(tenant_id)
		loan = self._get_loan(loan_application_id, tenant_id)

		if loan.status != GroupLoanStatus.PENDING:
			raise ValueError(f"loan_not_pending: {loan.status}")

		amount = self._decimal(approved_amount)
		if amount <= 0:
			raise ValueError("approved_amount must be positive")

		loan.approved_amount = amount
		loan.approved_by = approved_by
		loan.approved_at = self._now()
		loan.conditions = conditions
		loan.status = GroupLoanStatus.APPROVED
		self._emit(tenant_id, "group_loan_approved", loan.model_dump())
		return loan.model_dump()

	async def disburse_group_loan(
		self,
		tenant_id: str,
		loan_id: str,
		disbursement_instructions: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Disburse loan funds to individual members per their share.

		disbursement_instructions = [{"member_id": str, "amount": Decimal, "account_id": str}, ...]
		"""
		guard_tenant_id(tenant_id)
		loan = self._get_loan(loan_id, tenant_id)

		if loan.status != GroupLoanStatus.APPROVED:
			raise ValueError(f"loan_not_approved: {loan.status}")

		instructions: list[DisbursementInstruction] = []
		total_disbursed = Decimal("0")
		member_balances: dict[str, Decimal] = {}

		for d in disbursement_instructions:
			mid = d["member_id"]
			amt = self._decimal(d["amount"])
			acct = d["account_id"]
			if mid not in loan.borrower_member_ids:
				raise ValueError(f"member_not_in_loan: {mid}")
			if amt <= 0:
				raise ValueError(f"disbursement_amount_must_be_positive: {mid}")
			instructions.append(DisbursementInstruction(member_id=mid, amount=amt, account_id=acct))
			member_balances[mid] = amt
			total_disbursed += amt

		if total_disbursed != loan.approved_amount:
			raise ValueError(
				f"disbursement_total_mismatch: expected {loan.approved_amount}, got {total_disbursed}"
			)

		loan.disbursement_instructions = instructions
		loan.disbursed_amount = total_disbursed
		loan.outstanding_balance = total_disbursed
		loan.member_balances = member_balances
		loan.member_repaid = {mid: Decimal("0") for mid in member_balances}
		loan.disbursed_at = self._now()
		loan.status = GroupLoanStatus.ACTIVE

		# Update member loan shares
		for m in self._group_active_members(loan.group_id, tenant_id):
			if m.member_id in member_balances:
				m.total_loan_share += member_balances[m.member_id]

		self._emit(tenant_id, "group_loan_disbursed", loan.model_dump())
		return loan.model_dump()

	async def record_group_repayment(
		self,
		tenant_id: str,
		loan_id: str,
		total_amount: Decimal | float | str,
		*,
		payment_date: date | None = None,
		payment_ref: str,
		member_contributions: list[dict[str, Any]],
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Record a group repayment; tracks each member's contribution to the installment."""
		guard_tenant_id(tenant_id)
		loan = self._get_loan(loan_id, tenant_id)

		if loan.status not in (GroupLoanStatus.ACTIVE, GroupLoanStatus.ARREARS):
			raise ValueError(f"loan_not_repayable: {loan.status}")

		total = self._decimal(total_amount)
		if total <= 0:
			raise ValueError("repayment_amount_must_be_positive")

		lines: list[MemberRepaymentLine] = []
		line_total = Decimal("0")

		for mc in member_contributions:
			mid = mc["member_id"]
			amt = self._decimal(mc["amount"])
			lines.append(MemberRepaymentLine(member_id=mid, amount=amt))
			# Update per-member tracking
			if mid in loan.member_repaid:
				loan.member_repaid[mid] += amt
				loan.member_balances[mid] = max(
					Decimal("0"), loan.member_balances.get(mid, Decimal("0")) - amt
				)
			line_total += amt

		# Reduce outstanding balance
		loan.outstanding_balance = max(Decimal("0"), loan.outstanding_balance - total)
		if loan.outstanding_balance == Decimal("0"):
			loan.status = GroupLoanStatus.CLOSED
			loan.closed_at = self._now()
		elif loan.status == GroupLoanStatus.ARREARS and total > 0:
			loan.status = GroupLoanStatus.ACTIVE  # Partial recovery brings out of arrears

		# Update member repaid totals in group membership
		for m in self._group_active_members(loan.group_id, tenant_id):
			contrib_amt = next(
				(line.amount for line in lines if line.member_id == m.member_id), Decimal("0")
			)
			m.total_repaid += contrib_amt

		repayment = GroupRepayment(
			tenant_id=tenant_id,
			loan_id=loan_id,
			group_id=loan.group_id,
			total_amount=total,
			payment_date=payment_date or self._today(),
			payment_ref=payment_ref,
			member_contributions=lines,
			notes=notes,
		)
		self.repayments[repayment.id] = repayment
		self._emit(tenant_id, "group_repayment_recorded", repayment.model_dump())
		return repayment.model_dump()

	async def calculate_group_loan_arrears(
		self,
		tenant_id: str,
		loan_id: str,
		as_of_date: date | None = None,
	) -> dict[str, Any]:
		"""Compute arrears position for the group loan including per-member breakdown."""
		guard_tenant_id(tenant_id)
		loan = self._get_loan(loan_id, tenant_id)

		as_of = as_of_date or self._today()
		outstanding = loan.outstanding_balance
		approved = loan.approved_amount or Decimal("0")
		total_repaid_all = sum(loan.member_repaid.values(), Decimal("0"))
		expected_repaid = approved  # Full amount expected if past tenure
		arrears = max(Decimal("0"), expected_repaid - total_repaid_all - outstanding)

		arrears_rate = (
			self._decimal(arrears / approved * 100) if approved > 0 else Decimal("0")
		)

		# Per-member positions
		member_positions: list[MemberArrearsPosition] = []
		defaulting_ids: list[str] = []

		disbursed_at_date = (
			date.fromisoformat(loan.disbursed_at[:10])
			if loan.disbursed_at else self._today()
		)
		days_elapsed = (as_of - disbursed_at_date).days
		months_elapsed = max(1, days_elapsed // 30)
		days_in_arrears = max(0, days_elapsed - loan.tenure_months * 30)

		for mid in loan.borrower_member_ids:
			share = loan.member_balances.get(mid, Decimal("0"))
			repaid = loan.member_repaid.get(mid, Decimal("0"))
			member_arrears = max(Decimal("0"), share)  # remaining balance = arrears if overdue

			# Find last repayment date for this member
			last_date: date | None = None
			for rp in self.repayments.values():
				if rp.loan_id == loan_id:
					for line in rp.member_contributions:
						if line.member_id == mid and line.amount > 0:
							if last_date is None or rp.payment_date > last_date:
								last_date = rp.payment_date

			is_defaulting = repaid == Decimal("0") and share > 0
			if is_defaulting:
				defaulting_ids.append(mid)

			member_positions.append(MemberArrearsPosition(
				member_id=mid,
				loan_share=loan.member_balances.get(mid, Decimal("0")) + repaid,
				total_repaid=repaid,
				arrears_amount=member_arrears if days_in_arrears > 0 else Decimal("0"),
				last_payment_date=last_date,
				is_defaulting=is_defaulting,
			))

		if defaulting_ids and loan.status == GroupLoanStatus.ACTIVE:
			loan.status = GroupLoanStatus.ARREARS

		return GroupArrearsPosition(
			loan_id=loan_id,
			group_id=loan.group_id,
			as_of_date=as_of,
			total_outstanding=outstanding,
			total_arrears=arrears,
			arrears_rate_pct=arrears_rate,
			days_in_arrears=days_in_arrears,
			member_positions=member_positions,
			defaulting_member_ids=defaulting_ids,
		).model_dump()

	async def get_defaulting_members(
		self, tenant_id: str, loan_id: str
	) -> list[dict[str, Any]]:
		"""Return members who have contributed nothing toward loan repayment."""
		arrears = await self.calculate_group_loan_arrears(tenant_id, loan_id)
		return [
			pos for pos in arrears["member_positions"]
			if pos["is_defaulting"]
		]

	async def trigger_joint_liability(
		self,
		tenant_id: str,
		loan_id: str,
		defaulting_member_id: str,
	) -> dict[str, Any]:
		"""Call on the group to cover a defaulting member's balance.

		Records a joint-liability event. Actual fund collection happens via
		record_group_repayment with remaining members' contributions.
		"""
		guard_tenant_id(tenant_id)
		loan = self._get_loan(loan_id, tenant_id)

		if defaulting_member_id not in loan.borrower_member_ids:
			raise ValueError(f"member_not_in_loan: {defaulting_member_id}")

		defaulting_balance = loan.member_balances.get(defaulting_member_id, Decimal("0"))
		other_members = [m for m in loan.borrower_member_ids if m != defaulting_member_id]
		share_per_member = (
			self._decimal(defaulting_balance / len(other_members))
			if other_members else Decimal("0")
		)

		event = {
			"type": "joint_liability_triggered",
			"tenant_id": tenant_id,
			"loan_id": loan_id,
			"group_id": loan.group_id,
			"defaulting_member_id": defaulting_member_id,
			"defaulting_balance": str(defaulting_balance),
			"liable_members": other_members,
			"share_per_liable_member": str(share_per_member),
			"triggered_at": self._now(),
		}
		self._audit_events.append(event)
		_log.warning(
			"Joint liability triggered on loan %s for defaulting member %s",
			loan_id, defaulting_member_id,
		)
		return event

	# ── Merry-go-round ────────────────────────────────────────────────────────

	async def set_merry_go_round_order(
		self,
		tenant_id: str,
		group_id: str,
		member_order: list[str],
	) -> dict[str, Any]:
		"""Define the rotation order for merry-go-round disbursements."""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		active_members = self._group_active_members(group_id, tenant_id)
		active_ids = {m.member_id for m in active_members}

		for mid in member_order:
			if mid not in active_ids:
				raise ValueError(f"member_not_active_in_group: {mid}")

		for m in active_members:
			if m.member_id in member_order:
				m.merry_go_round_position = member_order.index(m.member_id) + 1

		return {
			"group_id": group_id,
			"rotation_order": member_order,
			"set_at": self._now(),
		}

	async def get_merry_go_round_schedule(
		self, tenant_id: str, group_id: str
	) -> list[dict[str, Any]]:
		"""Return the merry-go-round rotation schedule with received status."""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		members = sorted(
			(m for m in self._group_active_members(group_id, tenant_id)
			 if m.merry_go_round_position is not None),
			key=lambda m: m.merry_go_round_position or 999,
		)
		schedule: list[dict[str, Any]] = []
		for m in members:
			schedule.append(MerryGoRoundScheduleEntry(
				position=m.merry_go_round_position or 0,
				member_id=m.member_id,
				has_received=m.merry_go_round_received,
			).model_dump())
		return schedule

	async def process_merry_go_round(
		self,
		tenant_id: str,
		group_id: str,
		*,
		round_date: date | None = None,
		beneficiary_member_id: str,
	) -> dict[str, Any]:
		"""Collect contributions from all active members, disburse total to beneficiary.

		Returns MerryGoRoundResult with collection details and next beneficiary.
		"""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		active_members = self._group_active_members(group_id, tenant_id)
		beneficiary_gm = next(
			(m for m in active_members if m.member_id == beneficiary_member_id), None
		)
		if not beneficiary_gm:
			raise KeyError(f"beneficiary_not_active_in_group: {beneficiary_member_id}")
		if beneficiary_gm.merry_go_round_received:
			raise ValueError(f"member_already_received_merry_go_round: {beneficiary_member_id}")

		r_date = round_date or self._today()
		contributor_lines: list[MemberContributionLine] = []
		total = Decimal("0")

		# Determine standard contribution amount (use existing contribution average per member)
		# For simplicity, use the group's most recent contribution amounts per member
		recent_contribs = sorted(
			(c for c in self.contributions.values()
			 if c.group_id == group_id and c.contribution_type == ContributionType.MERRY_GO_ROUND),
			key=lambda c: c.meeting_date,
			reverse=True,
		)
		# Fall back to MONTHLY if no MGR contributions yet
		if not recent_contribs:
			recent_contribs = sorted(
				(c for c in self.contributions.values()
				 if c.group_id == group_id),
				key=lambda c: c.meeting_date,
				reverse=True,
			)

		latest_amounts: dict[str, Decimal] = {}
		if recent_contribs:
			for line in recent_contribs[0].lines:
				latest_amounts[line.member_id] = line.amount

		for m in active_members:
			if m.member_id == beneficiary_member_id:
				continue  # Beneficiary does not contribute to own round
			amt = latest_amounts.get(m.member_id, Decimal("0"))
			if amt > 0:
				m.total_contributions += amt
				contributor_lines.append(MemberContributionLine(member_id=m.member_id, amount=amt))
				total += amt

		beneficiary_gm.merry_go_round_received = True

		# Find next beneficiary in rotation
		current_pos = beneficiary_gm.merry_go_round_position or 0
		schedule = sorted(
			(m for m in active_members if m.merry_go_round_position is not None
			 and not m.merry_go_round_received),
			key=lambda m: m.merry_go_round_position or 999,
		)
		next_beneficiary = schedule[0].member_id if schedule else None

		# Determine round number
		past_rounds = sum(
			1 for r in self.mgr_rounds.values() if r.group_id == group_id
		)

		mgr = MerryGoRoundRound(
			tenant_id=tenant_id,
			group_id=group_id,
			round_number=past_rounds + 1,
			round_date=r_date,
			beneficiary_member_id=beneficiary_member_id,
			total_collected=total,
			contributor_lines=contributor_lines,
		)
		self.mgr_rounds[mgr.id] = mgr
		self._emit(tenant_id, "merry_go_round_processed", mgr.model_dump())

		return MerryGoRoundResult(
			round_id=mgr.id,
			group_id=group_id,
			beneficiary_member_id=beneficiary_member_id,
			total_collected=total,
			round_date=r_date,
			contributor_count=len(contributor_lines),
			next_beneficiary_member_id=next_beneficiary,
		).model_dump()

	# ── Reporting & analytics ─────────────────────────────────────────────────

	async def get_group_loan(self, tenant_id: str, loan_id: str) -> dict[str, Any]:
		"""Full loan profile with per-member balance positions."""
		guard_tenant_id(tenant_id)
		loan = self._get_loan(loan_id, tenant_id)
		result = loan.model_dump()
		# Enrich with human-readable balances
		result["member_balance_summary"] = [
			{
				"member_id": mid,
				"outstanding": str(bal),
				"repaid": str(loan.member_repaid.get(mid, Decimal("0"))),
			}
			for mid, bal in loan.member_balances.items()
		]
		return result

	async def list_group_loans(
		self,
		tenant_id: str,
		*,
		group_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List group loans, optionally filtered by group or status."""
		guard_tenant_id(tenant_id)
		results = []
		for ln in self.loans.values():
			if ln.tenant_id != tenant_id:
				continue
			if group_id and ln.group_id != group_id:
				continue
			if status and ln.status.value != status.upper():
				continue
			results.append(ln.model_dump())
		return results

	async def get_group_performance_score(
		self, tenant_id: str, group_id: str
	) -> dict[str, Any]:
		"""Compute a 0-100 performance score for the group.

		Factors: repayment rate (50%), contribution compliance (50%).
		"""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		loans = [
			ln for ln in self.loans.values()
			if ln.group_id == group_id and ln.tenant_id == tenant_id
		]
		active_loans = [ln for ln in loans if ln.status in (
			GroupLoanStatus.ACTIVE, GroupLoanStatus.ARREARS
		)]

		# Repayment rate: (disbursed - outstanding) / disbursed
		total_disbursed = sum(
			(ln.disbursed_amount or Decimal("0")) for ln in loans
		)
		total_repaid = sum(
			(ln.disbursed_amount or Decimal("0")) - ln.outstanding_balance
			for ln in loans if ln.disbursed_amount
		)
		repayment_rate = (
			self._decimal(total_repaid / total_disbursed * 100)
			if total_disbursed > 0 else Decimal("100")
		)

		# Contribution compliance
		compliance = await self.get_contribution_compliance(tenant_id, group_id)
		compliance_rate = Decimal(str(compliance["overall_compliance_pct"]))

		# Weighted score
		score = int(
			Decimal("0.50") * repayment_rate + Decimal("0.50") * compliance_rate
		)
		score = max(0, min(100, score))

		savings_data = await self.get_group_savings(tenant_id, group_id)

		return GroupPerformanceScore(
			group_id=group_id,
			tenant_id=tenant_id,
			score=score,
			grade=self._score_to_grade(score),
			repayment_rate_pct=repayment_rate,
			contribution_compliance_pct=compliance_rate,
			loan_count=len(loans),
			active_loan_count=len(active_loans),
			total_saved=Decimal(str(savings_data["total_savings"])),
			computed_at=self._now(),
		).model_dump()

	async def get_group_statement(
		self,
		tenant_id: str,
		group_id: str,
		from_date: date,
		to_date: date,
	) -> list[dict[str, Any]]:
		"""Full group ledger statement between two dates."""
		guard_tenant_id(tenant_id)
		self._get_group(group_id, tenant_id)

		entries: list[GroupStatementEntry] = []
		running_balance = Decimal("0")

		# Contributions
		for c in sorted(
			(c for c in self.contributions.values()
			 if c.group_id == group_id and c.tenant_id == tenant_id
			 and from_date <= c.meeting_date <= to_date),
			key=lambda c: c.meeting_date,
		):
			running_balance += c.total_amount
			entries.append(GroupStatementEntry(
				entry_date=c.meeting_date,
				entry_type="CONTRIBUTION",
				reference=c.id,
				amount=c.total_amount,
				running_balance=running_balance,
				description=f"{c.contribution_type.value} contribution",
			))

		# Loan disbursements
		for ln in sorted(
			(ln for ln in self.loans.values()
			 if ln.group_id == group_id and ln.tenant_id == tenant_id
			 and ln.disbursed_at is not None),
			key=lambda ln: ln.disbursed_at or "",
		):
			if ln.disbursed_at:
				d = date.fromisoformat(ln.disbursed_at[:10])
				if from_date <= d <= to_date:
					running_balance -= ln.disbursed_amount or Decimal("0")
					entries.append(GroupStatementEntry(
						entry_date=d,
						entry_type="LOAN_DISBURSEMENT",
						reference=ln.loan_number,
						amount=-(ln.disbursed_amount or Decimal("0")),
						running_balance=running_balance,
						description=f"Loan disbursed: {ln.purpose}",
					))

		# Repayments
		for rp in sorted(
			(rp for rp in self.repayments.values()
			 if rp.group_id == group_id and rp.tenant_id == tenant_id
			 and from_date <= rp.payment_date <= to_date),
			key=lambda rp: rp.payment_date,
		):
			running_balance += rp.total_amount
			entries.append(GroupStatementEntry(
				entry_date=rp.payment_date,
				entry_type="REPAYMENT",
				reference=rp.payment_ref,
				amount=rp.total_amount,
				running_balance=running_balance,
				description=f"Loan repayment ref {rp.payment_ref}",
			))

		# Sort by date
		entries.sort(key=lambda e: e.entry_date)
		return [e.model_dump() for e in entries]

	# ── Health ────────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"status": "ok",
			"groups": len(self.groups),
			"members": len(self.members),
			"loans": len(self.loans),
			"contributions": len(self.contributions),
			"repayments": len(self.repayments),
			"mgr_rounds": len(self.mgr_rounds),
			"checked_at": self._now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

