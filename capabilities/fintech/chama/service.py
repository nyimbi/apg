"""Executable async service layer for APG Chama & ROSCA Engine.

Implements the full Chama/ROSCA/Table Banking domain:
  - Group lifecycle management
  - Member management with KYC hooks
  - Contribution recording with MPESA payment method support
  - ROSCA rotation calculation and payout disbursement
  - Group lending (Table Banking) with interest and guarantors
  - Treasury state machine — single source of truth for balances
  - Cycle management and completion
  - Contribution reminders via NTFY
  - Member statements (full transaction history)
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_FREQUENCIES,
		SUPPORTED_GROUP_TYPES,
		SUPPORTED_PAYMENT_METHODS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		ChContribution,
		ChContributionStatus,
		ChCycle,
		ChCycleStatus,
		ChGroup,
		ChGroupType,
		ChFrequency,
		ChLoan,
		ChLoanRepayment,
		ChLoanStatus,
		ChMeetingRecord,
		ChMeetingType,
		ChMember,
		ChPaymentMethod,
		ChPayout,
		ChPayoutStatus,
		ChTreasury,
		uuid7str,
	)
except ImportError:  # pragma: no cover — direct execution
	from capability_contract import (  # type: ignore
		SUPPORTED_FREQUENCIES,
		SUPPORTED_GROUP_TYPES,
		SUPPORTED_PAYMENT_METHODS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		ChContribution,
		ChContributionStatus,
		ChCycle,
		ChCycleStatus,
		ChGroup,
		ChGroupType,
		ChFrequency,
		ChLoan,
		ChLoanRepayment,
		ChLoanStatus,
		ChMeetingRecord,
		ChMeetingType,
		ChMember,
		ChPaymentMethod,
		ChPayout,
		ChPayoutStatus,
		ChTreasury,
		uuid7str,
	)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _d(v: Any) -> Decimal:
	"""Coerce to Decimal safely."""
	return Decimal(str(v))


def _present(v: Any) -> bool:
	if v is None:
		return False
	if isinstance(v, str):
		return bool(v.strip())
	if isinstance(v, (list, dict)):
		return bool(v)
	return True


def _cents(v: Decimal) -> Decimal:
	"""Round to 2 decimal places (KES style)."""
	return v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ChamaService:
	"""Tenant-scoped runtime for Chama/ROSCA/Table Banking operations.

	All state is held in-memory (dicts keyed on (tenant_id, id)).
	Production deployment would back this with PostgreSQL — the interface
	is intentionally identical; swap _store writes for DB calls.
	"""

	def __init__(self) -> None:
		# Primary stores
		self.groups: dict[tuple[str, str], ChGroup] = {}
		self.members: dict[tuple[str, str], ChMember] = {}
		self.contributions: dict[tuple[str, str], ChContribution] = {}
		self.payouts: dict[tuple[str, str], ChPayout] = {}
		self.cycles: dict[tuple[str, str], ChCycle] = {}
		self.loans: dict[tuple[str, str], ChLoan] = {}
		self.repayments: dict[tuple[str, str], ChLoanRepayment] = {}
		self.meetings: dict[tuple[str, str], ChMeetingRecord] = {}
		self.treasuries: dict[tuple[str, str], ChTreasury] = {}
		# Audit trail
		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Describe / evaluate
	# ------------------------------------------------------------------

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	async def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Group management
	# ------------------------------------------------------------------

	async def create_group(
		self,
		name: str,
		group_type: str,
		contribution_amount: Decimal | float | str,
		frequency: str,
		tenant_id: str,
		description: str = "",
		max_members: int = 100,
		registration_number: str | None = None,
		bank_account: str | None = None,
		mpesa_paybill: str | None = None,
	) -> dict[str, Any]:
		"""Create a new Chama, ROSCA, or Table Banking group.

		Returns the ChGroup dict. Initialises an empty treasury for the group.
		"""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_group",
			"group_type_supported": group_type.lower() in SUPPORTED_GROUP_TYPES,
			"group_name_present": _present(name),
			"contribution_amount_present": _present(contribution_amount),
			"frequency_supported": frequency.lower() in SUPPORTED_FREQUENCIES,
		})
		group = ChGroup(
			tenant_id=tenant_id,
			name=name,
			group_type=ChGroupType(group_type.lower()),
			description=description,
			contribution_amount=_d(contribution_amount),
			frequency=ChFrequency(frequency.lower()),
			max_members=max_members,
			registration_number=registration_number,
			bank_account=bank_account,
			mpesa_paybill=mpesa_paybill,
		)
		self.groups[self._key(tenant_id, group.id)] = group
		# Bootstrap treasury
		treasury = ChTreasury(tenant_id=tenant_id, group_id=group.id)
		self.treasuries[self._key(tenant_id, group.id)] = treasury
		# Auto-create first cycle
		await self._open_cycle(group.id, 1, tenant_id)
		self._audit(tenant_id, "group.created", group.id)
		return group.to_dict()

	async def add_member(
		self,
		group_id: str,
		name: str,
		phone: str,
		tenant_id: str,
		national_id: str = "",
		email: str = "",
		contribution_amount: Decimal | float | str | None = None,
	) -> dict[str, Any]:
		"""Add a member to an existing group.

		If contribution_amount is None, inherits from the group.
		Assigns next available payout_order for ROSCA groups.
		"""
		group = self._group_or_raise(group_id, tenant_id)
		assert len(group.member_ids) < group.max_members, "group is at capacity"
		assert _present(name), "name required"
		assert _present(phone), "phone required"
		amt = _d(contribution_amount) if contribution_amount is not None else group.contribution_amount
		# Determine payout order for ROSCA
		next_order = len(group.member_ids) + 1
		member = ChMember(
			tenant_id=tenant_id,
			group_id=group_id,
			name=name,
			phone=phone,
			national_id=national_id,
			email=email,
			contribution_amount=amt,
			payout_order=next_order,
		)
		self.members[self._key(tenant_id, member.id)] = member
		# Update group membership and rotation
		group.member_ids.append(member.id)
		if group.group_type in (ChGroupType.ROSCA, ChGroupType.TABLE_BANKING):
			group.payout_rotation.append(member.id)
		group.updated_at = _now()
		# Update treasury member count
		treasury = self.treasuries.get(self._key(tenant_id, group_id))
		if treasury:
			treasury.member_count = len(group.member_ids)
		self._audit(tenant_id, "member.added", member.id)
		return member.to_dict()

	# ------------------------------------------------------------------
	# Contributions
	# ------------------------------------------------------------------

	async def record_contribution(
		self,
		group_id: str,
		member_id: str,
		amount: Decimal | float | str,
		payment_method: str,
		tenant_id: str,
		payment_reference: str = "",
		mpesa_receipt: str | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""Record a member contribution for the current active cycle.

		Updates: cycle.collected_amount, treasury.total_savings,
		treasury.cash_balance, member.total_contributed.
		Publishes contribution.received event.
		"""
		group = self._group_or_raise(group_id, tenant_id)
		member = self._member_or_raise(member_id, tenant_id)
		assert member.group_id == group_id, "member does not belong to group"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_contribution",
			"member_present": True,
			"payment_method_supported": payment_method.lower() in SUPPORTED_PAYMENT_METHODS,
			"amount_positive": _d(amount) > 0,
		})
		cycle = self._active_cycle(group_id, tenant_id)
		amt = _cents(_d(amount))
		contrib = ChContribution(
			tenant_id=tenant_id,
			group_id=group_id,
			member_id=member_id,
			cycle_number=cycle.cycle_number,
			amount=amt,
			expected_amount=member.contribution_amount,
			payment_method=ChPaymentMethod(payment_method.lower()),
			payment_reference=payment_reference,
			mpesa_receipt=mpesa_receipt,
			notes=notes,
			paid_at=_now(),
		)
		self.contributions[self._key(tenant_id, contrib.id)] = contrib
		# Update cycle
		cycle.collected_amount = _cents(cycle.collected_amount + amt)
		cycle.contribution_status[member_id] = "paid"
		# Update member totals
		member.total_contributed = _cents(member.total_contributed + amt)
		# Update treasury
		treasury = self.treasuries[self._key(tenant_id, group_id)]
		treasury.total_savings = _cents(treasury.total_savings + amt)
		treasury.cash_balance = _cents(treasury.cash_balance + amt)
		treasury.as_of = _now()
		self._audit(tenant_id, "contribution.received", contrib.id)
		return contrib.to_dict()

	# ------------------------------------------------------------------
	# ROSCA rotation and payout
	# ------------------------------------------------------------------

	async def calculate_next_payout(self, group_id: str, tenant_id: str) -> dict[str, Any]:
		"""Determine which member receives the payout next in the ROSCA rotation.

		For CHAMA groups: returns None (chamas do not rotate; they vote or pro-rate).
		For TABLE_BANKING: returns the next in rotation for the savings pool payout.
		Returns member dict of the next recipient.
		"""
		group = self._group_or_raise(group_id, tenant_id)
		if group.group_type == ChGroupType.CHAMA:
			return {"group_id": group_id, "group_type": "chama", "next_recipient": None, "note": "Chama payouts are voted, not rotated"}
		rotation = group.payout_rotation
		assert rotation, "no members in payout rotation"
		idx = group.current_rotation_index % len(rotation)
		next_member_id = rotation[idx]
		member = self._member_or_raise(next_member_id, tenant_id)
		return member.to_dict() | {"rotation_index": idx, "cycle_number": group.current_cycle_number}

	async def disburse_payout(
		self,
		group_id: str,
		cycle_id: str,
		tenant_id: str,
		payment_method: str = "mpesa",
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""Disburse the cycle payout to the next member in rotation.

		Marks cycle as completed, advances rotation index, opens next cycle.
		Publishes payout.disbursed and cycle.completed events.
		"""
		group = self._group_or_raise(group_id, tenant_id)
		cycle = self.cycles.get(self._key(tenant_id, cycle_id))
		assert cycle is not None, f"cycle {cycle_id!r} not found"
		assert cycle.group_id == group_id, "cycle does not belong to group"
		assert cycle.status == ChCycleStatus.ACTIVE, "cycle is not active"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "disburse_payout",
			"cycle_present": True,
			"recipient_present": True,
		})
		# Determine recipient
		next_info = await self.calculate_next_payout(group_id, tenant_id)
		if next_info.get("next_recipient") is None and group.group_type == ChGroupType.CHAMA:
			# Chama: payout full pot to designated recipient (cycle.payout_member_id)
			assert cycle.payout_member_id, "set cycle.payout_member_id for CHAMA payout"
			recipient_id = cycle.payout_member_id
		else:
			recipient_id = next_info["id"]
		recipient = self._member_or_raise(recipient_id, tenant_id)
		treasury = self.treasuries[self._key(tenant_id, group_id)]
		payout_amount = _cents(cycle.collected_amount)
		assert treasury.cash_balance >= payout_amount, "insufficient treasury balance for payout"
		payout = ChPayout(
			tenant_id=tenant_id,
			group_id=group_id,
			cycle_id=cycle_id,
			cycle_number=cycle.cycle_number,
			recipient_member_id=recipient_id,
			amount=payout_amount,
			payment_method=ChPaymentMethod(payment_method.lower()),
			mpesa_phone=recipient.phone,
			status=ChPayoutStatus.DISBURSED,
			approved_by=approved_by,
			approved_at=_now() if approved_by else None,
			disbursed_at=_now(),
		)
		self.payouts[self._key(tenant_id, payout.id)] = payout
		# Update treasury
		treasury.total_payouts_disbursed = _cents(treasury.total_payouts_disbursed + payout_amount)
		treasury.cash_balance = _cents(treasury.cash_balance - payout_amount)
		treasury.as_of = _now()
		# Update recipient member totals
		recipient.total_received_payouts = _cents(recipient.total_received_payouts + payout_amount)
		# Complete cycle
		cycle.status = ChCycleStatus.COMPLETED
		cycle.payout_id = payout.id
		cycle.payout_amount = payout_amount
		cycle.completed_at = _now()
		cycle.payout_member_id = recipient_id
		# Advance rotation
		group.current_rotation_index = (group.current_rotation_index + 1) % max(len(group.payout_rotation), 1)
		group.total_cycles_completed += 1
		group.current_cycle_number += 1
		group.updated_at = _now()
		# Open next cycle
		await self._open_cycle(group_id, group.current_cycle_number, tenant_id)
		self._audit(tenant_id, "payout.disbursed", payout.id)
		self._audit(tenant_id, "cycle.completed", cycle_id)
		return payout.to_dict()

	# ------------------------------------------------------------------
	# Loans (Table Banking)
	# ------------------------------------------------------------------

	async def create_loan(
		self,
		group_id: str,
		member_id: str,
		amount: Decimal | float | str,
		interest_rate_monthly_pct: Decimal | float | str,
		repayment_months: int,
		guarantor_ids: list[str],
		tenant_id: str,
		notes: str = "",
		payment_method: str = "mpesa",
	) -> dict[str, Any]:
		"""Create a loan application from group treasury.

		Validates:
		- Amount does not exceed available cash balance
		- At least 2 guarantors from group members
		- All guarantors are different from the borrower

		Calculates simple interest amortisation.
		"""
		group = self._group_or_raise(group_id, tenant_id)
		member = self._member_or_raise(member_id, tenant_id)
		assert member.group_id == group_id, "member does not belong to group"
		treasury = self.treasuries[self._key(tenant_id, group_id)]
		principal = _cents(_d(amount))
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_loan",
			"amount_positive": principal > 0,
			"guarantors_present": len(guarantor_ids) >= 2,
			"loan_exceeds_treasury": principal > treasury.cash_balance,
		})
		assert repayment_months >= 1, "repayment_months must be >= 1"
		# Validate guarantors
		for gid in guarantor_ids:
			assert gid != member_id, f"borrower cannot be their own guarantor: {gid}"
			g = self.members.get(self._key(tenant_id, gid))
			assert g is not None and g.group_id == group_id, f"guarantor {gid!r} not in group"
		# Simple interest calculation
		rate = _d(interest_rate_monthly_pct) / 100
		total_interest = _cents(principal * rate * repayment_months)
		total_repayable = _cents(principal + total_interest)
		monthly_instalment = _cents(total_repayable / repayment_months)
		loan = ChLoan(
			tenant_id=tenant_id,
			group_id=group_id,
			borrower_member_id=member_id,
			principal=principal,
			interest_rate_monthly_pct=_d(interest_rate_monthly_pct),
			repayment_months=repayment_months,
			total_interest=total_interest,
			total_repayable=total_repayable,
			monthly_instalment=monthly_instalment,
			outstanding_balance=total_repayable,
			guarantor_member_ids=guarantor_ids,
			payment_method=ChPaymentMethod(payment_method.lower()),
			notes=notes,
			status=ChLoanStatus.APPROVED,
			approved_at=_now(),
		)
		self.loans[self._key(tenant_id, loan.id)] = loan
		# Update treasury — reserve cash
		treasury.cash_balance = _cents(treasury.cash_balance - principal)
		treasury.total_loans_disbursed = _cents(treasury.total_loans_disbursed + principal)
		treasury.total_loans_outstanding = _cents(treasury.total_loans_outstanding + total_repayable)
		treasury.active_loans += 1
		treasury.as_of = _now()
		# Update member outstanding
		member.total_loans_outstanding = _cents(member.total_loans_outstanding + total_repayable)
		self._audit(tenant_id, "loan.approved", loan.id)
		return loan.to_dict()

	async def record_repayment(
		self,
		loan_id: str,
		amount: Decimal | float | str,
		payment_method: str,
		tenant_id: str,
		payment_reference: str = "",
		mpesa_receipt: str | None = None,
	) -> dict[str, Any]:
		"""Record a repayment instalment against a loan.

		Applies payment to principal+interest proportionally.
		Marks loan fully_repaid when outstanding_balance reaches zero.
		Updates treasury cash_balance and interest_income.
		"""
		loan = self._loan_or_raise(loan_id, tenant_id)
		assert loan.status in (ChLoanStatus.APPROVED, ChLoanStatus.ACTIVE), \
			f"loan {loan_id!r} is not active (status={loan.status})"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_repayment",
			"amount_positive": _d(amount) > 0,
		})
		amt = _cents(_d(amount))
		# Split into principal vs interest proportionally
		if loan.outstanding_balance > 0:
			interest_fraction = loan.total_interest / loan.total_repayable if loan.total_repayable > 0 else Decimal("0")
			interest_portion = _cents(min(amt * interest_fraction, loan.total_interest - (loan.amount_repaid * interest_fraction)))
			principal_portion = _cents(amt - interest_portion)
		else:
			interest_portion = Decimal("0")
			principal_portion = Decimal("0")
		new_balance = _cents(max(loan.outstanding_balance - amt, Decimal("0")))
		repayment = ChLoanRepayment(
			tenant_id=tenant_id,
			group_id=loan.group_id,
			loan_id=loan_id,
			member_id=loan.borrower_member_id,
			amount=amt,
			principal_portion=principal_portion,
			interest_portion=interest_portion,
			payment_method=ChPaymentMethod(payment_method.lower()),
			payment_reference=payment_reference,
			mpesa_receipt=mpesa_receipt,
			balance_after=new_balance,
		)
		self.repayments[self._key(tenant_id, repayment.id)] = repayment
		# Update loan
		loan.amount_repaid = _cents(loan.amount_repaid + amt)
		loan.outstanding_balance = new_balance
		loan.status = ChLoanStatus.ACTIVE
		if new_balance == Decimal("0"):
			loan.status = ChLoanStatus.FULLY_REPAID
			loan.fully_repaid_at = _now()
		# Update treasury
		treasury = self.treasuries[self._key(tenant_id, loan.group_id)]
		treasury.cash_balance = _cents(treasury.cash_balance + amt)
		treasury.total_loans_outstanding = _cents(max(treasury.total_loans_outstanding - amt, Decimal("0")))
		treasury.total_interest_income = _cents(treasury.total_interest_income + interest_portion)
		if loan.status == ChLoanStatus.FULLY_REPAID:
			treasury.active_loans = max(treasury.active_loans - 1, 0)
		treasury.as_of = _now()
		# Update member outstanding
		member = self._member_or_raise(loan.borrower_member_id, tenant_id)
		member.total_loans_outstanding = _cents(max(member.total_loans_outstanding - amt, Decimal("0")))
		self._audit(tenant_id, "loan.repayment.received", repayment.id)
		return loan.to_dict()

	# ------------------------------------------------------------------
	# Treasury
	# ------------------------------------------------------------------

	async def get_treasury_summary(self, group_id: str, tenant_id: str) -> dict[str, Any]:
		"""Return full treasury snapshot for a group."""
		self._group_or_raise(group_id, tenant_id)
		treasury = self.treasuries.get(self._key(tenant_id, group_id))
		assert treasury is not None, f"treasury not initialised for group {group_id!r}"
		return treasury.to_dict()

	# ------------------------------------------------------------------
	# Cycle management
	# ------------------------------------------------------------------

	async def get_contribution_status(
		self, group_id: str, cycle_id: str, tenant_id: str
	) -> dict[str, Any]:
		"""Return contribution status for all members in a cycle.

		Result: {paid: [...], unpaid: [...], partial: [...], cycle: {...}}
		"""
		self._group_or_raise(group_id, tenant_id)
		cycle = self.cycles.get(self._key(tenant_id, cycle_id))
		assert cycle is not None, f"cycle {cycle_id!r} not found"
		group = self.groups[self._key(tenant_id, group_id)]
		paid: list[str] = []
		partial: list[str] = []
		unpaid: list[str] = []
		for mid in group.member_ids:
			status = cycle.contribution_status.get(mid, "pending")
			if status == "paid":
				paid.append(mid)
			elif status == "partial":
				partial.append(mid)
			else:
				unpaid.append(mid)
		completion_pct = round(len(paid) / max(len(group.member_ids), 1) * 100, 1)
		return {
			"group_id": group_id,
			"cycle_id": cycle_id,
			"cycle_number": cycle.cycle_number,
			"status": cycle.status.value,
			"paid": paid,
			"partial": partial,
			"unpaid": unpaid,
			"paid_count": len(paid),
			"unpaid_count": len(unpaid),
			"completion_pct": completion_pct,
			"collected_amount": str(cycle.collected_amount),
			"expected_amount": str(cycle.expected_amount),
			"as_of": _now(),
		}

	# ------------------------------------------------------------------
	# Reminders
	# ------------------------------------------------------------------

	async def send_contribution_reminders(
		self, group_id: str, tenant_id: str
	) -> int:
		"""Identify members who have not yet contributed this cycle and send reminders.

		In production this calls the NTFY adapter. Here we simulate and return
		the count of members notified.
		"""
		group = self._group_or_raise(group_id, tenant_id)
		cycle = self._active_cycle(group_id, tenant_id)
		notified = 0
		for mid in group.member_ids:
			status = cycle.contribution_status.get(mid, "pending")
			if status not in ("paid",):
				# Production: await ntfy_adapter.send_sms(member.phone, message)
				self._audit(tenant_id, "reminder.sent", mid)
				notified += 1
		return notified

	# ------------------------------------------------------------------
	# Member statements
	# ------------------------------------------------------------------

	async def get_member_statement(
		self, group_id: str, member_id: str, tenant_id: str
	) -> dict[str, Any]:
		"""Full transaction history for a member within a group.

		Includes: contributions, payouts received, loans, repayments.
		"""
		self._group_or_raise(group_id, tenant_id)
		member = self._member_or_raise(member_id, tenant_id)
		assert member.group_id == group_id, "member does not belong to group"
		# Contributions
		contribs = [
			c.to_dict()
			for (tid, _), c in self.contributions.items()
			if tid == tenant_id and c.group_id == group_id and c.member_id == member_id
		]
		# Payouts
		paid_out = [
			p.to_dict()
			for (tid, _), p in self.payouts.items()
			if tid == tenant_id and p.group_id == group_id and p.recipient_member_id == member_id
		]
		# Loans
		member_loans = [
			l.to_dict()
			for (tid, _), l in self.loans.items()
			if tid == tenant_id and l.group_id == group_id and l.borrower_member_id == member_id
		]
		# Repayments
		member_repayments = [
			r.to_dict()
			for (tid, _), r in self.repayments.items()
			if tid == tenant_id and r.group_id == group_id and r.member_id == member_id
		]
		total_contrib = sum(_d(c["amount"]) for c in contribs)
		total_payout = sum(_d(p["amount"]) for p in paid_out)
		total_loans = sum(_d(l["principal"]) for l in member_loans)
		total_repaid = sum(_d(r["amount"]) for r in member_repayments)
		return {
			"group_id": group_id,
			"member_id": member_id,
			"member": member.to_dict(),
			"contributions": contribs,
			"payouts_received": paid_out,
			"loans": member_loans,
			"repayments": member_repayments,
			"summary": {
				"total_contributed": str(_cents(total_contrib)),
				"total_payouts_received": str(_cents(total_payout)),
				"total_loans_disbursed": str(_cents(total_loans)),
				"total_loan_repayments": str(_cents(total_repaid)),
				"outstanding_loan_balance": str(member.total_loans_outstanding),
				"contribution_count": len(contribs),
				"payout_count": len(paid_out),
				"active_loans": sum(1 for l in member_loans if l["status"] in ("approved", "active")),
			},
			"generated_at": _now(),
		}

	# ------------------------------------------------------------------
	# Meeting records
	# ------------------------------------------------------------------

	async def record_meeting(
		self,
		group_id: str,
		tenant_id: str,
		meeting_type: str = "regular",
		venue: str = "",
		agenda: list[str] | None = None,
		resolutions: list[str] | None = None,
		minutes_text: str = "",
		members_present: int = 0,
		chairperson_id: str = "",
		secretary_id: str = "",
		recorded_by: str = "",
	) -> dict[str, Any]:
		"""Record a group meeting with minutes and resolutions."""
		group = self._group_or_raise(group_id, tenant_id)
		treasury = self.treasuries.get(self._key(tenant_id, group_id))
		balance = treasury.cash_balance if treasury else Decimal("0")
		quorum_threshold = max(len(group.member_ids) // 2 + 1, 1)
		meeting = ChMeetingRecord(
			tenant_id=tenant_id,
			group_id=group_id,
			meeting_type=ChMeetingType(meeting_type.lower()),
			venue=venue,
			total_members=len(group.member_ids),
			members_present=members_present,
			quorum_met=members_present >= quorum_threshold,
			agenda=agenda or [],
			resolutions=resolutions or [],
			minutes_text=minutes_text,
			treasury_balance_at_meeting=balance,
			chairperson_id=chairperson_id,
			secretary_id=secretary_id,
			recorded_by=recorded_by,
		)
		self.meetings[self._key(tenant_id, meeting.id)] = meeting
		self._audit(tenant_id, "meeting.recorded", meeting.id)
		return meeting.to_dict()

	# ------------------------------------------------------------------
	# Dashboard summary
	# ------------------------------------------------------------------

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""High-level tenant dashboard: active groups, total savings, active loans."""
		group_count = self._count(self.groups, tenant_id)
		member_count = self._count(self.members, tenant_id)
		contribution_count = self._count(self.contributions, tenant_id)
		payout_count = self._count(self.payouts, tenant_id)
		loan_count = self._count(self.loans, tenant_id)
		total_savings = sum(
			t.total_savings
			for (tid, _), t in self.treasuries.items()
			if tid == tenant_id
		)
		total_loans_outstanding = sum(
			t.total_loans_outstanding
			for (tid, _), t in self.treasuries.items()
			if tid == tenant_id
		)
		return {
			"tenant_id": tenant_id,
			"group_count": group_count,
			"member_count": member_count,
			"contribution_count": contribution_count,
			"payout_count": payout_count,
			"loan_count": loan_count,
			"total_savings_kes": str(_cents(total_savings)),
			"total_loans_outstanding_kes": str(_cents(total_loans_outstanding)),
			"as_of": _now(),
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _group_or_raise(self, group_id: str, tenant_id: str) -> ChGroup:
		g = self.groups.get(self._key(tenant_id, group_id))
		if g is None:
			raise KeyError(f"group {group_id!r} not found for tenant {tenant_id!r}")
		return g

	def _member_or_raise(self, member_id: str, tenant_id: str) -> ChMember:
		m = self.members.get(self._key(tenant_id, member_id))
		if m is None:
			raise KeyError(f"member {member_id!r} not found for tenant {tenant_id!r}")
		return m

	def _loan_or_raise(self, loan_id: str, tenant_id: str) -> ChLoan:
		l = self.loans.get(self._key(tenant_id, loan_id))
		if l is None:
			raise KeyError(f"loan {loan_id!r} not found for tenant {tenant_id!r}")
		return l

	def _active_cycle(self, group_id: str, tenant_id: str) -> ChCycle:
		"""Return the currently active cycle for a group."""
		for (tid, _), cycle in self.cycles.items():
			if tid == tenant_id and cycle.group_id == group_id and cycle.status == ChCycleStatus.ACTIVE:
				return cycle
		raise KeyError(f"no active cycle found for group {group_id!r}")

	async def _open_cycle(self, group_id: str, cycle_number: int, tenant_id: str) -> ChCycle:
		"""Create and store a new active cycle, initialising contribution_status for all members."""
		group = self.groups.get(self._key(tenant_id, group_id))
		member_ids = group.member_ids if group else []
		# Expected amount = sum of all member contribution amounts
		expected = sum(
			self.members[self._key(tenant_id, mid)].contribution_amount
			for mid in member_ids
			if self._key(tenant_id, mid) in self.members
		)
		cycle = ChCycle(
			tenant_id=tenant_id,
			group_id=group_id,
			cycle_number=cycle_number,
			expected_amount=_cents(_d(expected)),
			contribution_status={mid: "pending" for mid in member_ids},
		)
		self.cycles[self._key(tenant_id, cycle.id)] = cycle
		self._audit(tenant_id, "cycle.started", cycle.id)
		return cycle

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"recorded_at": _now(),
		})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for (tid, _) in store if tid == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "chama_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "chama_policy_denied")


# Canonical alias used by APG capability loader
FintechChamaService = ChamaService
