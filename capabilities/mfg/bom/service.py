"""Async service layer for APG Bill of Materials."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import evaluate_capability_rules
except ImportError:
	from capability_contract import evaluate_capability_rules  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgBomService:
	"""Bill of Materials service — async, in-memory."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._boms: dict[str, dict[str, Any]] = {}
		self._bom_lines: dict[str, dict[str, Any]] = {}
		self._ecos: dict[str, dict[str, Any]] = {}

	async def create_bom(
		self,
		parent_item_id: str,
		parent_item_code: str,
		bom_type: str = "manufacturing",
		version: str = "1",
		effective_from: str | None = None,
		created_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		ctx = {"tenant_context_present": True, "operation": "create_bom", "parent_item_present": bool(parent_item_id)}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"BOM creation denied: {decision['actions']}")
		bom: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"parent_item_id": parent_item_id,
			"parent_item_code": parent_item_code,
			"bom_type": bom_type,
			"version": version,
			"effective_from": effective_from or _now(),
			"effective_to": None,
			"status": "active",
			"created_at": _now(),
			"created_by": created_by,
			"metadata": metadata or {},
		}
		self._boms[bom["id"]] = bom
		return bom

	async def add_bom_line(
		self,
		bom_id: str,
		component_item_id: str,
		component_item_code: str,
		quantity: float,
		uom: str = "EA",
		item_type: str = "buy",
		sequence: int = 10,
		notes: str = "",
	) -> dict[str, Any]:
		if bom_id not in self._boms:
			raise KeyError(f"BOM not found: {bom_id}")
		line: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"bom_id": bom_id,
			"component_item_id": component_item_id,
			"component_item_code": component_item_code,
			"quantity": quantity,
			"uom": uom,
			"item_type": item_type,
			"sequence": sequence,
			"notes": notes,
			"created_at": _now(),
		}
		self._bom_lines[line["id"]] = line
		return line

	async def get_bom(self, bom_id: str) -> dict[str, Any]:
		if bom_id not in self._boms:
			raise KeyError(f"BOM not found: {bom_id}")
		bom = dict(self._boms[bom_id])
		bom["lines"] = [l for l in self._bom_lines.values() if l["bom_id"] == bom_id]
		return bom

	async def list_boms(self, parent_item_id: str | None = None, bom_type: str | None = None) -> list[dict[str, Any]]:
		boms = list(self._boms.values())
		if parent_item_id:
			boms = [b for b in boms if b["parent_item_id"] == parent_item_id]
		if bom_type:
			boms = [b for b in boms if b["bom_type"] == bom_type]
		return boms

	async def explode_bom(self, bom_id: str, quantity: float = 1.0) -> list[dict[str, Any]]:
		"""Single-level BOM explosion. Returns component requirements."""
		bom = await self.get_bom(bom_id)
		return [
			{
				"component_item_id": line["component_item_id"],
				"component_item_code": line["component_item_code"],
				"required_quantity": line["quantity"] * quantity,
				"uom": line["uom"],
				"item_type": line["item_type"],
			}
			for line in bom.get("lines", [])
		]

	async def create_eco(
		self,
		bom_id: str,
		eco_number: str,
		description: str,
		requested_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		eco: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"bom_id": bom_id,
			"eco_number": eco_number,
			"description": description,
			"status": "draft",
			"requested_by": requested_by,
			"approver_id": None,
			"approved_at": None,
			"released_at": None,
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._ecos[eco["id"]] = eco
		return eco

	async def approve_eco(self, eco_id: str, approver_id: str) -> dict[str, Any]:
		eco = self._ecos.get(eco_id)
		if not eco:
			raise KeyError(f"ECO not found: {eco_id}")
		eco["status"] = "approved"
		eco["approver_id"] = approver_id
		eco["approved_at"] = _now()
		return eco

	async def release_eco(self, eco_id: str) -> dict[str, Any]:
		eco = self._ecos.get(eco_id)
		if not eco:
			raise KeyError(f"ECO not found: {eco_id}")
		ctx = {"tenant_context_present": True, "operation": "release_eco", "approval_present": eco.get("status") == "approved"}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"ECO release denied: {decision['actions']}")
		eco["status"] = "released"
		eco["released_at"] = _now()
		return eco

	async def list_ecos(self, bom_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		ecos = list(self._ecos.values())
		if bom_id:
			ecos = [e for e in ecos if e["bom_id"] == bom_id]
		if status:
			ecos = [e for e in ecos if e["status"] == status]
		return ecos
