"""Service layer for the Recommender Systems capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import RecsRecord


class RecsService:
	"""Dependency-light service backed by the capability contract."""

	def __init__(self) -> None:
		self._records: dict[str, RecsRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		records = list(self._records.values())
		if tenant_id is not None:
			records = [record for record in records if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(records, key=lambda item: item.id)]

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		self._enforce_write_policy(tenant_id)
		record = RecsRecord(
			id=record_id,
			tenant_id=tenant_id,
			status=status,
			metadata=dict(metadata or {}),
		)
		self._records[record_id] = record
		return record.to_dict()

	def _enforce_write_policy(self, tenant_id: str) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"risk_level": "low",
			"review_recorded": True,
		})
		if result["decision"] != "allow":
			reasons = ", ".join(action.get("reason", "capability_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "capability_policy_blocked")
