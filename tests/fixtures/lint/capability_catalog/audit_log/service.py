"""Service stub for audit_log (test fixture)."""
from __future__ import annotations


class AuditLogService:
	"""Minimal audit log service for fixture testing."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id

	def log(self, event: str, payload: dict | None = None) -> dict:
		return {"capability": "audit_log", "event": event, "tenant_id": self.tenant_id, "ok": True}
