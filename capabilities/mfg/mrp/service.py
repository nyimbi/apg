"""Async service layer for APG Material Requirements Planning."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
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


class MfgMrpService:
	"""Material Requirements Planning service — in-memory store, async API."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self._tenant_id = tenant_id
		# keyed by id
		self._planning_runs = WriteThruDict('planning_runs', tenant_id, _store)
		self._production_orders = WriteThruDict('production_orders', tenant_id, _store)
		self._purchase_requisitions = WriteThruDict('purchase_requisitions', tenant_id, _store)
		self._exception_messages = WriteThruDict('exception_messages', tenant_id, _store)
		self._pegging_records = WriteThruDict('pegging_records', tenant_id, _store)

	# ------------------------------------------------------------------ #
	# Planning Runs
	# ------------------------------------------------------------------ #

	async def create_planning_run(
		self,
		run_type: str = "full",
		horizon: str = "week",
		horizon_days: int = 30,
		triggered_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		ctx = {"tenant_context_present": True, "operation": "run_mrp", "horizon_valid": horizon in ["day", "week", "month", "quarter"]}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"MRP planning run denied: {decision['actions']}")

		run: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"run_type": run_type,
			"horizon": horizon,
			"horizon_days": horizon_days,
			"status": "pending",
			"items_exploded": 0,
			"orders_created": 0,
			"requisitions_created": 0,
			"messages_raised": 0,
			"started_at": None,
			"completed_at": None,
			"error_message": None,
			"triggered_by": triggered_by,
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._planning_runs[run["id"]] = run
		return run

	async def execute_planning_run(self, run_id: str) -> dict[str, Any]:
		"""Simulate MRP explosion: marks run completed and returns summary."""
		run = await self.get_planning_run(run_id)
		run["status"] = "running"
		run["started_at"] = _now()

		await asyncio.sleep(0)  # yield — real impl would do BOM explosion here

		run["status"] = "completed"
		run["completed_at"] = _now()
		run["items_exploded"] = len(self._production_orders)
		run["orders_created"] = 0
		run["requisitions_created"] = 0
		run["messages_raised"] = 0
		return run

	async def get_planning_run(self, run_id: str) -> dict[str, Any]:
		if run_id not in self._planning_runs:
			raise KeyError(f"Planning run not found: {run_id}")
		return self._planning_runs[run_id]

	async def list_planning_runs(
		self,
		status: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		runs = list(self._planning_runs.values())
		if status:
			runs = [r for r in runs if r["status"] == status]
		return runs[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Production Orders
	# ------------------------------------------------------------------ #

	async def create_production_order(
		self,
		item_id: str,
		item_code: str,
		quantity: float,
		start_date: str,
		due_date: str,
		order_type: str = "planned",
		bom_id: str | None = None,
		work_center_id: str | None = None,
		source_demand_id: str | None = None,
		priority: int = 50,
		notes: str = "",
		created_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		ctx = {
			"tenant_context_present": True,
			"operation": "create_production_order",
			"item_present": bool(item_id),
			"quantity_valid": quantity > 0,
		}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Production order denied: {decision['actions']}")

		order: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"item_id": item_id,
			"item_code": item_code,
			"order_type": order_type,
			"quantity": quantity,
			"uom": "EA",
			"start_date": start_date,
			"due_date": due_date,
			"bom_id": bom_id,
			"work_center_id": work_center_id,
			"source_demand_id": source_demand_id,
			"priority": priority,
			"notes": notes,
			"released_at": None,
			"completed_at": None,
			"created_at": _now(),
			"updated_at": _now(),
			"created_by": created_by,
			"metadata": metadata or {},
		}
		self._production_orders[order["id"]] = order
		return order

	async def release_production_order(self, order_id: str, bom_id: str) -> dict[str, Any]:
		order = await self.get_production_order(order_id)
		ctx = {
			"tenant_context_present": True,
			"operation": "release_production_order",
			"bom_present": bool(bom_id),
		}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Release denied: {decision['actions']}")
		order["order_type"] = "released"
		order["bom_id"] = bom_id
		order["released_at"] = _now()
		order["updated_at"] = _now()
		return order

	async def complete_production_order(self, order_id: str, actual_qty: float | None = None) -> dict[str, Any]:
		order = await self.get_production_order(order_id)
		order["order_type"] = "completed"
		if actual_qty is not None:
			order["quantity"] = actual_qty
		order["completed_at"] = _now()
		order["updated_at"] = _now()
		return order

	async def get_production_order(self, order_id: str) -> dict[str, Any]:
		if order_id not in self._production_orders:
			raise KeyError(f"Production order not found: {order_id}")
		return self._production_orders[order_id]

	async def list_production_orders(
		self,
		order_type: str | None = None,
		item_id: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		orders = list(self._production_orders.values())
		if order_type:
			orders = [o for o in orders if o["order_type"] == order_type]
		if item_id:
			orders = [o for o in orders if o["item_id"] == item_id]
		return orders[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Purchase Requisitions
	# ------------------------------------------------------------------ #

	async def create_purchase_requisition(
		self,
		item_id: str,
		item_code: str,
		quantity: float,
		required_date: str,
		supplier_id: str | None = None,
		unit_cost: float | None = None,
		currency: str = "USD",
		source_demand_id: str | None = None,
		notes: str = "",
		created_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		ctx = {
			"tenant_context_present": True,
			"operation": "create_purchase_requisition",
			"item_present": bool(item_id),
		}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Purchase requisition denied: {decision['actions']}")

		req: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"item_id": item_id,
			"item_code": item_code,
			"status": "draft",
			"quantity": quantity,
			"uom": "EA",
			"unit_cost": unit_cost,
			"currency": currency,
			"required_date": required_date,
			"supplier_id": supplier_id,
			"source_demand_id": source_demand_id,
			"approver_id": None,
			"approved_at": None,
			"purchase_order_id": None,
			"notes": notes,
			"created_at": _now(),
			"updated_at": _now(),
			"created_by": created_by,
			"metadata": metadata or {},
		}
		self._purchase_requisitions[req["id"]] = req
		return req

	async def approve_purchase_requisition(self, req_id: str, approver_id: str) -> dict[str, Any]:
		req = await self.get_purchase_requisition(req_id)
		req["status"] = "approved"
		req["approver_id"] = approver_id
		req["approved_at"] = _now()
		req["updated_at"] = _now()
		return req

	async def get_purchase_requisition(self, req_id: str) -> dict[str, Any]:
		if req_id not in self._purchase_requisitions:
			raise KeyError(f"Purchase requisition not found: {req_id}")
		return self._purchase_requisitions[req_id]

	async def list_purchase_requisitions(
		self,
		status: str | None = None,
		item_id: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		reqs = list(self._purchase_requisitions.values())
		if status:
			reqs = [r for r in reqs if r["status"] == status]
		if item_id:
			reqs = [r for r in reqs if r["item_id"] == item_id]
		return reqs[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Exception Messages
	# ------------------------------------------------------------------ #

	async def raise_exception_message(
		self,
		message_type: str,
		item_id: str,
		item_code: str,
		description: str,
		severity: str = "medium",
		order_id: str | None = None,
		planning_run_id: str | None = None,
		suggested_action: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		msg: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"planning_run_id": planning_run_id,
			"message_type": message_type,
			"severity": severity,
			"item_id": item_id,
			"item_code": item_code,
			"order_id": order_id,
			"description": description,
			"suggested_action": suggested_action,
			"processed": False,
			"processed_at": None,
			"processed_by": None,
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._exception_messages[msg["id"]] = msg
		return msg

	async def process_exception_message(self, msg_id: str, processed_by: str) -> dict[str, Any]:
		msg = self._exception_messages.get(msg_id)
		if not msg:
			raise KeyError(f"Exception message not found: {msg_id}")
		msg["processed"] = True
		msg["processed_at"] = _now()
		msg["processed_by"] = processed_by
		return msg

	async def list_exception_messages(
		self,
		processed: bool | None = None,
		severity: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		msgs = list(self._exception_messages.values())
		if processed is not None:
			msgs = [m for m in msgs if m["processed"] == processed]
		if severity:
			msgs = [m for m in msgs if m["severity"] == severity]
		return msgs[offset : offset + limit]

	# ------------------------------------------------------------------ #
	# Pegging
	# ------------------------------------------------------------------ #

	async def create_pegging_record(
		self,
		supply_type: str,
		supply_id: str,
		demand_type: str,
		demand_id: str,
		item_id: str,
		quantity_pegged: float,
		peg_date: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		record: dict[str, Any] = {
			"id": uuid7str(),
			"tenant_id": self._tenant_id,
			"supply_type": supply_type,
			"supply_id": supply_id,
			"demand_type": demand_type,
			"demand_id": demand_id,
			"item_id": item_id,
			"quantity_pegged": quantity_pegged,
			"peg_date": peg_date,
			"created_at": _now(),
			"metadata": metadata or {},
		}
		self._pegging_records[record["id"]] = record
		return record

	async def get_pegging_for_demand(self, demand_id: str) -> list[dict[str, Any]]:
		return [r for r in self._pegging_records.values() if r["demand_id"] == demand_id]

	async def get_pegging_for_supply(self, supply_id: str) -> list[dict[str, Any]]:
		return [r for r in self._pegging_records.values() if r["supply_id"] == supply_id]

	# ------------------------------------------------------------------ #
	# Dashboard / Summary
	# ------------------------------------------------------------------ #

	async def get_dashboard_summary(self) -> dict[str, Any]:
		orders = list(self._production_orders.values())
		reqs = list(self._purchase_requisitions.values())
		msgs = list(self._exception_messages.values())

		return {
			"tenant_id": self._tenant_id,
			"production_orders": {
				"total": len(orders),
				"planned": sum(1 for o in orders if o["order_type"] == "planned"),
				"firm_planned": sum(1 for o in orders if o["order_type"] == "firm_planned"),
				"released": sum(1 for o in orders if o["order_type"] == "released"),
				"completed": sum(1 for o in orders if o["order_type"] == "completed"),
			},
			"purchase_requisitions": {
				"total": len(reqs),
				"draft": sum(1 for r in reqs if r["status"] == "draft"),
				"submitted": sum(1 for r in reqs if r["status"] == "submitted"),
				"approved": sum(1 for r in reqs if r["status"] == "approved"),
			},
			"exception_messages": {
				"total": len(msgs),
				"unprocessed": sum(1 for m in msgs if not m["processed"]),
				"critical": sum(1 for m in msgs if m["severity"] == "critical"),
				"high": sum(1 for m in msgs if m["severity"] == "high"),
			},
			"planning_runs": {
				"total": len(self._planning_runs),
				"completed": sum(1 for r in self._planning_runs.values() if r["status"] == "completed"),
			},
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_planning_runs', '_production_orders', '_purchase_requisitions', '_exception_messages', '_pegging_records']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

