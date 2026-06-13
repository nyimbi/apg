"""Async service layer for APG Lot and Batch Management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import evaluate_capability_rules
	from .models import MfLmtLot, MfLmtGenealogyLink, MfLmtRecallEvent
except ImportError:
	from capability_contract import evaluate_capability_rules  # type: ignore
	from models import MfLmtLot, MfLmtGenealogyLink, MfLmtRecallEvent  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgLmtService:
	"""Lot and Batch Management service — async, in-memory."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._lots: dict[str, MfLmtLot] = {}
		self._genealogy: dict[str, MfLmtGenealogyLink] = {}
		self._recalls: dict[str, MfLmtRecallEvent] = {}

	async def create_lot(
		self,
		lot_number: str,
		item_id: str,
		item_code: str,
		quantity: float,
		lot_type: str = "production",
		uom: str = "EA",
		manufactured_date: str | None = None,
		expiry_date: str | None = None,
		shelf_life_days: int | None = None,
		work_order_id: str | None = None,
		supplier_id: str | None = None,
		supplier_lot_number: str | None = None,
		parent_lot_id: str | None = None,
		created_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> MfLmtLot:
		ctx = {"tenant_context_present": True, "operation": "create_lot", "item_present": bool(item_id)}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Lot creation denied: {decision['actions']}")

		lot = MfLmtLot(
			tenant_id=self._tenant_id,
			lot_number=lot_number,
			lot_type=lot_type,
			item_id=item_id,
			item_code=item_code,
			quantity=quantity,
			uom=uom,
			manufactured_date=manufactured_date,
			expiry_date=expiry_date,
			shelf_life_days=shelf_life_days,
			work_order_id=work_order_id,
			supplier_id=supplier_id,
			supplier_lot_number=supplier_lot_number,
			parent_lot_id=parent_lot_id,
			created_by=created_by,
			metadata=metadata or {},
		)
		self._lots[lot.id] = lot
		# Auto-link to parent if sub-lot
		if parent_lot_id and parent_lot_id in self._lots:
			await self.link_genealogy(parent_lot_id=parent_lot_id, child_lot_id=lot.id, quantity_consumed=quantity)
		return lot

	async def quarantine_lot(self, lot_id: str, reason: str) -> MfLmtLot:
		lot = self._lots.get(lot_id)
		if not lot:
			raise KeyError(f"Lot not found: {lot_id}")
		lot.status = "quarantine"
		lot.quarantine_reason = reason
		return lot

	async def release_lot(self, lot_id: str) -> MfLmtLot:
		lot = self._lots.get(lot_id)
		if not lot:
			raise KeyError(f"Lot not found: {lot_id}")
		lot.status = "available"
		lot.quarantine_reason = ""
		return lot

	async def expire_lot(self, lot_id: str) -> MfLmtLot:
		lot = self._lots.get(lot_id)
		if not lot:
			raise KeyError(f"Lot not found: {lot_id}")
		lot.status = "expired"
		return lot

	async def consume_lot(self, lot_id: str, quantity_consumed: float) -> MfLmtLot:
		lot = self._lots.get(lot_id)
		if not lot:
			raise KeyError(f"Lot not found: {lot_id}")
		lot.quantity -= quantity_consumed
		if lot.quantity <= 0:
			lot.status = "consumed"
			lot.quantity = 0.0
		return lot

	async def get_lot(self, lot_id: str) -> MfLmtLot:
		if lot_id not in self._lots:
			raise KeyError(f"Lot not found: {lot_id}")
		return self._lots[lot_id]

	async def list_lots(self, item_id: str | None = None, status: str | None = None, lot_type: str | None = None, limit: int = 100, offset: int = 0) -> list[MfLmtLot]:
		lots = list(self._lots.values())
		if item_id:
			lots = [l for l in lots if l.item_id == item_id]
		if status:
			lots = [l for l in lots if l.status == status]
		if lot_type:
			lots = [l for l in lots if l.lot_type == lot_type]
		return lots[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Genealogy
	# ------------------------------------------------------------------ #

	async def link_genealogy(self, parent_lot_id: str, child_lot_id: str, quantity_consumed: float, work_order_id: str | None = None) -> MfLmtGenealogyLink:
		link = MfLmtGenealogyLink(tenant_id=self._tenant_id, parent_lot_id=parent_lot_id, child_lot_id=child_lot_id, quantity_consumed=quantity_consumed, work_order_id=work_order_id)
		self._genealogy[link.id] = link
		return link

	async def trace_forward(self, lot_id: str) -> list[str]:
		"""Return all descendant lot IDs."""
		visited: set[str] = set()
		queue = [lot_id]
		while queue:
			current = queue.pop(0)
			for link in self._genealogy.values():
				if link.parent_lot_id == current and link.child_lot_id not in visited:
					visited.add(link.child_lot_id)
					queue.append(link.child_lot_id)
		return list(visited)

	async def trace_backward(self, lot_id: str) -> list[str]:
		"""Return all ancestor lot IDs."""
		visited: set[str] = set()
		queue = [lot_id]
		while queue:
			current = queue.pop(0)
			for link in self._genealogy.values():
				if link.child_lot_id == current and link.parent_lot_id not in visited:
					visited.add(link.parent_lot_id)
					queue.append(link.parent_lot_id)
		return list(visited)

	# ------------------------------------------------------------------ #
	# Recall
	# ------------------------------------------------------------------ #

	async def initiate_recall(self, recall_number: str, root_lot_id: str, reason: str, initiated_by: str, auto_quarantine: bool = True) -> MfLmtRecallEvent:
		ctx = {"tenant_context_present": True, "operation": "initiate_recall", "lot_present": bool(root_lot_id)}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Recall denied: {decision['actions']}")

		# Trace all affected lots
		affected = await self.trace_forward(root_lot_id)
		affected_ids = [root_lot_id] + affected

		if auto_quarantine:
			for lot_id in affected_ids:
				lot = self._lots.get(lot_id)
				if lot and lot.status not in ("expired", "consumed"):
					lot.status = "quarantine"
					lot.quarantine_reason = f"Recall {recall_number}: {reason}"

		event = MfLmtRecallEvent(tenant_id=self._tenant_id, recall_number=recall_number, root_lot_id=root_lot_id, reason=reason, affected_lot_ids=affected_ids, initiated_by=initiated_by)
		self._recalls[event.id] = event
		return event

	async def get_dashboard_summary(self) -> dict[str, Any]:
		lots = list(self._lots.values())
		return {
			"tenant_id": self._tenant_id,
			"lots": {
				"total": len(lots),
				"available": sum(1 for l in lots if l.status == "available"),
				"quarantine": sum(1 for l in lots if l.status == "quarantine"),
				"expired": sum(1 for l in lots if l.status == "expired"),
			},
			"genealogy_links": len(self._genealogy),
			"recalls": {"total": len(self._recalls), "active": sum(1 for r in self._recalls.values() if r.status in ("initiated", "in_progress"))},
		}
