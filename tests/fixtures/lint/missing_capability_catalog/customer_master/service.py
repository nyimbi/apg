"""Service stub for customer_master (test fixture)."""
from __future__ import annotations


class CustomerMasterService:
	"""Minimal customer master service for fixture testing."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id

	def get(self, customer_id: str) -> dict:
		return {"capability": "customer_master", "id": customer_id, "tenant_id": self.tenant_id, "ok": True}
