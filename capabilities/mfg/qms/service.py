"""Async service layer for APG Quality Management System."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

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


class MfgQmsService:
	"""Quality Management System service — async, in-memory."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self._tenant_id = tenant_id
		_store = get_store(db_url)
		self._inspections = WriteThruDict('inspections', tenant_id, _store)
		self._ncrs = WriteThruDict('ncrs', tenant_id, _store)
		self._capas = WriteThruDict('capas', tenant_id, _store)
		self._spc_samples = WriteThruDict('spc_samples', tenant_id, _store)

	# ------------------------------------------------------------------ #
	# Inspections
	# ------------------------------------------------------------------ #

	async def create_inspection(
		self,
		item_id: str,
		item_code: str,
		inspection_type: str,
		lot_id: str | None = None,
		work_order_id: str | None = None,
		quantity_inspected: float = 0.0,
		created_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		insp: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"item_id": item_id,
			"item_code": item_code,
			"inspection_type": inspection_type,
			"lot_id": lot_id,
			"work_order_id": work_order_id,
			"quantity_inspected": quantity_inspected,
			"quantity_accepted": 0.0,
			"quantity_rejected": 0.0,
			"disposition": None,
			"status": "open",
			"inspector_id": created_by,
			"completed_at": None,
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._inspections[insp["id"]] = insp
		return insp

	async def complete_inspection(self, insp_id: str, quantity_accepted: float, quantity_rejected: float, disposition: str) -> dict[str, Any]:
		insp = self._inspections.get(insp_id)
		if not insp:
			raise KeyError(f"Inspection not found: {insp_id}")
		insp["quantity_accepted"] = quantity_accepted
		insp["quantity_rejected"] = quantity_rejected
		insp["disposition"] = disposition
		insp["status"] = "completed"
		insp["completed_at"] = _now()
		if quantity_rejected > 0:
			await self.create_ncr(item_id=insp["item_id"], item_code=insp["item_code"], quantity_defective=quantity_rejected, source_type="inspection", source_id=insp_id)
		return insp

	# ------------------------------------------------------------------ #
	# NCR
	# ------------------------------------------------------------------ #

	async def create_ncr(
		self,
		item_id: str,
		item_code: str,
		quantity_defective: float,
		source_type: str = "inspection",
		source_id: str | None = None,
		defect_description: str = "",
		severity: str = "medium",
		reported_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		ctx = {"tenant_context_present": True, "operation": "create_ncr", "item_present": bool(item_id)}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"NCR denied: {decision['actions']}")
		ncr: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"item_id": item_id,
			"item_code": item_code,
			"quantity_defective": quantity_defective,
			"source_type": source_type,
			"source_id": source_id,
			"defect_description": defect_description,
			"severity": severity,
			"status": "open",
			"disposition": None,
			"reported_by": reported_by,
			"capa_id": None,
			"closed_at": None,
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._ncrs[ncr["id"]] = ncr
		return ncr

	async def close_ncr(self, ncr_id: str, disposition: str, root_cause: str = "", closed_by: str = "system") -> dict[str, Any]:
		ncr = self._ncrs.get(ncr_id)
		if not ncr:
			raise KeyError(f"NCR not found: {ncr_id}")
		ncr["status"] = "closed"
		ncr["disposition"] = disposition
		ncr["metadata"]["root_cause"] = root_cause
		ncr["metadata"]["closed_by"] = closed_by
		ncr["closed_at"] = _now()
		return ncr

	async def list_ncrs(self, status: str | None = None, severity: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
		ncrs = list(self._ncrs.values())
		if status:
			ncrs = [n for n in ncrs if n["status"] == status]
		if severity:
			ncrs = [n for n in ncrs if n["severity"] == severity]
		return ncrs[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# CAPA
	# ------------------------------------------------------------------ #

	async def create_capa(self, ncr_id: str, capa_type: str, title: str, description: str, owner_id: str, due_date: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		capa: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"ncr_id": ncr_id,
			"capa_type": capa_type,
			"title": title,
			"description": description,
			"owner_id": owner_id,
			"due_date": due_date,
			"status": "open",
			"approval_status": None,
			"approver_id": None,
			"completed_at": None,
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._capas[capa["id"]] = capa
		# Link NCR
		ncr = self._ncrs.get(ncr_id)
		if ncr:
			ncr["capa_id"] = capa["id"]
			ncr["status"] = "capa_in_progress"
		return capa

	async def close_capa(self, capa_id: str, approver_id: str, effectiveness_notes: str = "") -> dict[str, Any]:
		capa = self._capas.get(capa_id)
		if not capa:
			raise KeyError(f"CAPA not found: {capa_id}")
		ctx = {"tenant_context_present": True, "operation": "close_capa", "approval_present": bool(approver_id)}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"CAPA close denied: {decision['actions']}")
		capa["status"] = "closed"
		capa["approval_status"] = "approved"
		capa["approver_id"] = approver_id
		capa["metadata"]["effectiveness_notes"] = effectiveness_notes
		capa["completed_at"] = _now()
		return capa

	# ------------------------------------------------------------------ #
	# SPC
	# ------------------------------------------------------------------ #

	async def add_spc_sample(self, process_id: str, characteristic: str, value: float, uom: str = "", sample_size: int = 1, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		sample: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"process_id": process_id,
			"characteristic": characteristic,
			"value": value,
			"uom": uom,
			"sample_size": sample_size,
			"sampled_at": _now(),
			"metadata": metadata or {},
		}
		self._spc_samples[sample["id"]] = sample
		return sample

	async def get_spc_summary(self, process_id: str, characteristic: str) -> dict[str, Any]:
		samples = [s for s in self._spc_samples.values() if s["process_id"] == process_id and s["characteristic"] == characteristic]
		if not samples:
			return {"process_id": process_id, "characteristic": characteristic, "count": 0}
		values = [s["value"] for s in samples]
		n = len(values)
		mean = sum(values) / n
		variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
		stddev = variance ** 0.5
		return {
			"process_id": process_id,
			"characteristic": characteristic,
			"count": n,
			"mean": round(mean, 4),
			"std_dev": round(stddev, 4),
			"min": min(values),
			"max": max(values),
			"ucl": round(mean + 3 * stddev, 4),
			"lcl": round(mean - 3 * stddev, 4),
		}

	async def get_dashboard_summary(self) -> dict[str, Any]:
		ncrs = list(self._ncrs.values())
		capas = list(self._capas.values())
		insps = list(self._inspections.values())
		return {
			"tenant_id": self._tenant_id,
			"inspections": {"total": len(insps), "open": sum(1 for i in insps if i["status"] == "open"), "completed": sum(1 for i in insps if i["status"] == "completed")},
			"ncrs": {"total": len(ncrs), "open": sum(1 for n in ncrs if n["status"] == "open"), "critical": sum(1 for n in ncrs if n["severity"] == "critical")},
			"capas": {"total": len(capas), "open": sum(1 for c in capas if c["status"] == "open"), "closed": sum(1 for c in capas if c["status"] == "closed")},
			"spc_samples": len(self._spc_samples),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_inspections', '_ncrs', '_capas', '_spc_samples']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

