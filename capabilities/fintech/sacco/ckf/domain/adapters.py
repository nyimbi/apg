"""Domain adapters — bridge check-off service to lnd/dep/mem capabilities.

In production these adapters pull live data from:
  - fintech.sacco.lnd  (loan installments, arrears)
  - fintech.sacco.dep  (savings contributions)
  - fintech.sacco.mem  (member registry, employer links)

For standalone / test usage the service uses its own stubs via
register_loan_installment() and register_savings_contribution().
"""
from __future__ import annotations

from typing import Any, Protocol


class LoanServicePort(Protocol):
	"""Minimal surface needed from SACCO Lending service."""

	async def get_due_installments(self, tenant_id: str, member_id: str) -> list[dict[str, Any]]:
		"""Return installments with status in {due, overdue} for this member."""
		...


class SavingsServicePort(Protocol):
	"""Minimal surface needed from SACCO Deposits/Savings service."""

	async def get_standing_orders(self, tenant_id: str, member_id: str) -> list[dict[str, Any]]:
		"""Return active recurring savings contributions for this member."""
		...


class MemberServicePort(Protocol):
	"""Minimal surface needed from SACCO Member registry."""

	async def get_member(self, tenant_id: str, member_id: str) -> dict[str, Any]:
		"""Return member record with at least {id, full_name}."""
		...


class StubLoanAdapter:
	"""No-op adapter — service uses internal stubs."""

	async def get_due_installments(self, tenant_id: str, member_id: str) -> list[dict[str, Any]]:
		return []


class StubSavingsAdapter:
	"""No-op adapter — service uses internal stubs."""

	async def get_standing_orders(self, tenant_id: str, member_id: str) -> list[dict[str, Any]]:
		return []


class StubMemberAdapter:
	async def get_member(self, tenant_id: str, member_id: str) -> dict[str, Any]:
		return {"id": member_id, "full_name": member_id}
