"""Domain adapters for SACCO Guarantor Management.

Bridges the GuarantorService to APG infrastructure:
- NATS event bus (audit events, notifications)
- APG mem capability (member savings/shares/status)
- APG lnd capability (loan status/DPD)
- APG ntfy capability (SMS/push notices)

All adapters degrade gracefully to no-ops when infrastructure is absent.
"""
from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Any

_log = logging.getLogger(__name__)


# ── NATS event adapter ────────────────────────────────────────────────────────

def get_audit_adapter(capability_id: str = "fintech_sacco_gua") -> Any:
	"""Return a NATS event adapter if NATS_URL is configured, else None."""
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	return None


# ── Member context adapter ────────────────────────────────────────────────────

class MemberContextAdapter:
	"""
	Fetches member savings, shares, active status, and defaulter flag
	from the APG mem (membership) and dep (deposits) capabilities.

	Falls back to returning None when those capabilities are unavailable,
	in which case the GuarantorService uses its seeded in-memory values.
	"""

	async def get_member_savings(self, tenant_id: str, member_id: str) -> Decimal | None:
		try:
			from capabilities.fintech.sacco.dep.service import SaccoDepositService
			svc = SaccoDepositService(tenant_id=tenant_id)
			summary = await svc.member_deposit_summary(member_id, tenant_id=tenant_id)
			return Decimal(str(summary.get("total_balance", "0")))
		except Exception as exc:
			_log.debug("mem_savings_unavailable member=%s: %s", member_id, exc)
			return None

	async def get_member_shares(self, tenant_id: str, member_id: str) -> Decimal | None:
		try:
			from capabilities.fintech.sacco.mem.service import SaccoMemberService
			svc = SaccoMemberService(tenant_id=tenant_id)
			member = await svc.get_member(member_id, tenant_id=tenant_id)
			return Decimal(str(member.get("share_capital", "0")))
		except Exception as exc:
			_log.debug("mem_shares_unavailable member=%s: %s", member_id, exc)
			return None

	async def is_active_member(self, tenant_id: str, member_id: str) -> bool | None:
		try:
			from capabilities.fintech.sacco.mem.service import SaccoMemberService
			svc = SaccoMemberService(tenant_id=tenant_id)
			member = await svc.get_member(member_id, tenant_id=tenant_id)
			return member.get("status") == "active"
		except Exception as exc:
			_log.debug("mem_active_unavailable member=%s: %s", member_id, exc)
			return None

	async def is_defaulter(self, tenant_id: str, member_id: str) -> bool | None:
		"""Check if member has any loan in arrears > 0 days."""
		try:
			from capabilities.fintech.sacco.lnd.service import SaccoLendingService
			svc = SaccoLendingService(tenant_id=tenant_id)
			loans = await svc.list_loans(tenant_id=tenant_id, member_id=member_id, status="arrears")
			return len(loans) > 0
		except Exception as exc:
			_log.debug("lnd_defaulter_unavailable member=%s: %s", member_id, exc)
			return None


# ── Loan context adapter ──────────────────────────────────────────────────────

class LoanContextAdapter:
	"""
	Fetches loan status and DPD from the APG lnd (lending) capability.
	"""

	async def get_loan_status(self, tenant_id: str, loan_id: str) -> str | None:
		try:
			from capabilities.fintech.sacco.lnd.service import SaccoLendingService
			svc = SaccoLendingService(tenant_id=tenant_id)
			loan = await svc.get_loan(loan_id, tenant_id=tenant_id)
			return loan.get("status")
		except Exception as exc:
			_log.debug("lnd_status_unavailable loan=%s: %s", loan_id, exc)
			return None

	async def get_loan_dpd(self, tenant_id: str, loan_id: str) -> int:
		try:
			from capabilities.fintech.sacco.lnd.service import SaccoLendingService
			svc = SaccoLendingService(tenant_id=tenant_id)
			loan = await svc.get_loan(loan_id, tenant_id=tenant_id)
			return int(loan.get("arrears_days", 0))
		except Exception as exc:
			_log.debug("lnd_dpd_unavailable loan=%s: %s", loan_id, exc)
			return 0


# ── Notification adapter ──────────────────────────────────────────────────────

class NotificationAdapter:
	"""
	Routes guarantee notices through the APG ntfy (notifications) capability.
	"""

	_TEMPLATES = {
		"warning": "Your savings are pledged as guarantee for loan {loan_id}. The loan is at risk.",
		"call_notice": "Your guaranteed savings have been applied to recover loan {loan_id} due to default.",
		"release": "Your savings pledge for loan {loan_id} has been released. Thank you.",
	}

	async def send(
		self,
		tenant_id: str,
		member_id: str,
		notice_type: str,
		loan_id: str,
		channel: str = "sms",
	) -> bool:
		template = self._TEMPLATES.get(notice_type, "Guarantee notice: {notice_type} for loan {loan_id}.")
		message = template.format(loan_id=loan_id, notice_type=notice_type)
		try:
			from capabilities.common.ntfy.service import NotificationService
			svc = NotificationService()
			await svc.send(
				tenant_id=tenant_id,
				recipient_id=member_id,
				message=message,
				channel=channel,
			)
			return True
		except Exception as exc:
			_log.debug("ntfy_unavailable member=%s notice=%s: %s", member_id, notice_type, exc)
			return False
