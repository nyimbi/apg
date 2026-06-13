"""Async service layer for APG Shop Floor Control."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import evaluate_capability_rules
	from .models import MfSfcWorkCentre, MfSfcRouting, MfSfcOperation, MfSfcLabourRecord
except ImportError:
	from capability_contract import evaluate_capability_rules  # type: ignore
	from models import MfSfcWorkCentre, MfSfcRouting, MfSfcOperation, MfSfcLabourRecord  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgSfcService:
	"""Shop Floor Control service — async, in-memory."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._work_centres: dict[str, MfSfcWorkCentre] = {}
		self._routings: dict[str, MfSfcRouting] = {}
		self._operations: dict[str, MfSfcOperation] = {}
		self._labour_records: dict[str, MfSfcLabourRecord] = {}

	async def create_work_centre(self, code: str, name: str, wc_type: str = "machine", capacity_hours_per_day: float = 8.0, **kwargs: Any) -> MfSfcWorkCentre:
		wc = MfSfcWorkCentre(tenant_id=self._tenant_id, code=code, name=name, wc_type=wc_type, capacity_hours_per_day=capacity_hours_per_day, **kwargs)
		self._work_centres[wc.id] = wc
		return wc

	async def list_work_centres(self, wc_type: str | None = None) -> list[MfSfcWorkCentre]:
		wcs = list(self._work_centres.values())
		return [w for w in wcs if w.wc_type == wc_type] if wc_type else wcs

	async def create_routing(self, item_id: str, item_code: str, version: str = "1", **kwargs: Any) -> MfSfcRouting:
		routing = MfSfcRouting(tenant_id=self._tenant_id, item_id=item_id, item_code=item_code, version=version, **kwargs)
		self._routings[routing.id] = routing
		return routing

	async def add_operation(self, routing_id: str, sequence: int, operation_code: str, operation_name: str, work_centre_id: str, setup_time_hrs: float = 0.0, run_time_hrs: float = 0.0, **kwargs: Any) -> MfSfcOperation:
		if routing_id not in self._routings:
			raise KeyError(f"Routing not found: {routing_id}")
		op = MfSfcOperation(tenant_id=self._tenant_id, routing_id=routing_id, sequence=sequence, operation_code=operation_code, operation_name=operation_name, work_centre_id=work_centre_id, setup_time_hrs=setup_time_hrs, run_time_hrs=run_time_hrs, **kwargs)
		self._operations[op.id] = op
		return op

	async def start_operation(self, op_id: str, operator_id: str, work_order_id: str) -> MfSfcOperation:
		op = self._operations.get(op_id)
		if not op:
			raise KeyError(f"Operation not found: {op_id}")
		op.status = "in_progress"
		op.work_order_id = work_order_id
		op.operator_id = operator_id
		op.started_at = _now()
		return op

	async def complete_operation(self, op_id: str) -> MfSfcOperation:
		op = self._operations.get(op_id)
		if not op:
			raise KeyError(f"Operation not found: {op_id}")
		op.status = "completed"
		op.completed_at = _now()
		return op

	async def log_labour(self, operation_id: str, work_order_id: str, operator_id: str, hours_logged: float, labour_type: str = "direct", notes: str = "") -> MfSfcLabourRecord:
		record = MfSfcLabourRecord(tenant_id=self._tenant_id, operation_id=operation_id, work_order_id=work_order_id, operator_id=operator_id, hours_logged=hours_logged, labour_type=labour_type, notes=notes)
		self._labour_records[record.id] = record
		return record

	async def get_dispatch_list(self, work_centre_id: str) -> list[MfSfcOperation]:
		return [op for op in self._operations.values() if op.work_centre_id == work_centre_id and op.status in ("queued", "setup")]
