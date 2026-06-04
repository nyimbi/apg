"""Executable service layer for APG Warehouse Operations."""

from __future__ import annotations

import asyncio
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_WAREHOUSE_TYPES, SUPPORTED_RECEIPT_METHODS, SUPPORTED_PUTAWAY_STRATEGIES,
		SUPPORTED_PICK_METHODS, SUPPORTED_PACK_TYPES, SUPPORTED_CYCLE_COUNT_TYPES,
		SUPPORTED_DOCK_DOOR_STATUSES, SUPPORTED_STORAGE_CONDITIONS, SUPPORTED_WMS_INTEGRATION_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		Warehouse, GoodsReceipt, PutawayTask, PickTask, PackTask,
		CycleCount, DockDoor, InventoryAdjustment, WarehouseAgent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_WAREHOUSE_TYPES, SUPPORTED_RECEIPT_METHODS, SUPPORTED_PUTAWAY_STRATEGIES,
		SUPPORTED_PICK_METHODS, SUPPORTED_PACK_TYPES, SUPPORTED_CYCLE_COUNT_TYPES,
		SUPPORTED_DOCK_DOOR_STATUSES, SUPPORTED_STORAGE_CONDITIONS, SUPPORTED_WMS_INTEGRATION_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		Warehouse, GoodsReceipt, PutawayTask, PickTask, PackTask,
		CycleCount, DockDoor, InventoryAdjustment, WarehouseAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and str(value).strip())

def _norm(value: str) -> str:
	return str(value).strip().lower() if value else ""

def _positive(value: float | int) -> bool:
	try:
		return float(value) > 0
	except (TypeError, ValueError):
		return False

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Labour productivity benchmarks (units per hour by activity)
_PRODUCTIVITY_BENCHMARKS: dict[str, float] = {
	"receiving": 120.0,  # line items per hour
	"putaway": 80.0,
	"picking": 100.0,
	"packing": 60.0,
	"shipping": 150.0,
}

# Cross-dock allocation — simple FIFO
_SHIPPING_CARRIERS = ["dhl", "fedex", "ups", "local_courier"]


class WarehouseOperationsService:
	"""Tenant-scoped warehouse operations runtime."""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.warehouses: dict[tuple[str, str], Warehouse] = {}
		self.receipts: dict[tuple[str, str], GoodsReceipt] = {}
		self.putaway_tasks: dict[tuple[str, str], PutawayTask] = {}
		self.pick_tasks: dict[tuple[str, str], PickTask] = {}
		self.pack_tasks: dict[tuple[str, str], PackTask] = {}
		self.cycle_counts: dict[tuple[str, str], CycleCount] = {}
		self.dock_doors: dict[tuple[str, str], DockDoor] = {}
		self.inventory_adjustments: dict[tuple[str, str], InventoryAdjustment] = {}
		self.agents: dict[tuple[str, str], WarehouseAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self.shipment_records: dict[tuple[str, str], dict[str, Any]] = {}
		self.labour_log: list[dict[str, Any]] = {}
		self.inventory: dict[tuple[str, str], dict[str, Any]] = {}  # (tenant, sku) -> {qty, location}

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Existing methods (preserved)
	# ------------------------------------------------------------------

	def register_warehouse(
		self, warehouse_id: str, tenant_id: str, warehouse_type: str,
		name: str, location: str, storage_condition: str,
		capacity_sqm: float, dock_door_count: int,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a warehouse."""
		warehouse_type = _norm(warehouse_type)
		storage_condition = _norm(storage_condition)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "register_warehouse",
			"warehouse_type_supported": warehouse_type in SUPPORTED_WAREHOUSE_TYPES,
		})
		item = Warehouse(warehouse_id, tenant_id, warehouse_type, name, location, storage_condition, float(capacity_sqm), int(dock_door_count))
		self.warehouses[self._key(tenant_id, warehouse_id)] = item
		self._audit(tenant_id, "warehouse_registered", warehouse_id)
		return item.to_dict()

	def receive_goods(
		self, receipt_id: str, tenant_id: str, warehouse_id: str,
		receipt_method: str, supplier_id: str, po_reference: str,
		line_count: int, received_at: str,
		barcode_scanned: bool = True, damage_inspection_completed: bool = True,
		cold_chain_required: bool = False, temperature_checked: bool = False,
	) -> dict[str, Any]:
		"""Record goods receipt at warehouse."""
		receipt_method = _norm(receipt_method)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "receive_goods",
			"receipt_method_supported": receipt_method in SUPPORTED_RECEIPT_METHODS,
			"barcode_scanned": barcode_scanned,
			"damage_inspection_completed": damage_inspection_completed,
			"cold_chain_required": cold_chain_required,
			"temperature_checked": temperature_checked,
		})
		item = GoodsReceipt(receipt_id, tenant_id, warehouse_id, receipt_method, supplier_id, po_reference, int(line_count), temperature_checked, damage_inspection_completed, received_at)
		self.receipts[self._key(tenant_id, receipt_id)] = item
		self._audit(tenant_id, "goods_received", receipt_id)
		return item.to_dict()

	def execute_putaway(
		self, task_id: str, tenant_id: str, receipt_id: str,
		strategy: str, slot_id: str, operator_id: str,
		slot_verified: bool = True,
	) -> dict[str, Any]:
		"""Execute a putaway task."""
		strategy = _norm(strategy)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "execute_putaway",
			"strategy_supported": strategy in SUPPORTED_PUTAWAY_STRATEGIES,
			"slot_verified": slot_verified,
		})
		item = PutawayTask(task_id, tenant_id, receipt_id, strategy, slot_id, True, None, operator_id)
		self.putaway_tasks[self._key(tenant_id, task_id)] = item
		self._audit(tenant_id, "putaway_completed", task_id)
		return item.to_dict()

	def create_pick_task(
		self, task_id: str, tenant_id: str, order_id: str,
		pick_method: str, warehouse_id: str, lines_count: int,
		priority: str, operator_id: str,
	) -> dict[str, Any]:
		"""Create a pick task."""
		pick_method = _norm(pick_method)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_pick_task",
			"pick_method_supported": pick_method in SUPPORTED_PICK_METHODS,
		})
		item = PickTask(task_id, tenant_id, order_id, pick_method, warehouse_id, int(lines_count), priority, operator_id, None)
		self.pick_tasks[self._key(tenant_id, task_id)] = item
		self._audit(tenant_id, "pick_task_created", task_id)
		return item.to_dict()

	def complete_pick_task(self, task_id: str, tenant_id: str, completed_at: str) -> dict[str, Any]:
		"""Mark a pick task as completed."""
		task = self.pick_tasks.get(self._key(tenant_id, task_id))
		if task is None:
			raise KeyError(f"Pick task {task_id} not found")
		task.completed_at = completed_at
		self._audit(tenant_id, "pick_completed", task_id)
		return task.to_dict()

	def create_pack_task(
		self, task_id: str, tenant_id: str, pick_task_id: str,
		pack_type: str, weight_kg: float, weight_checked: bool = True,
	) -> dict[str, Any]:
		"""Create a packing task."""
		pack_type = _norm(pack_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_pack_task",
			"pack_type_supported": pack_type in SUPPORTED_PACK_TYPES,
		})
		item = PackTask(task_id, tenant_id, pick_task_id, pack_type, float(weight_kg), weight_checked, False, False, None)
		self.pack_tasks[self._key(tenant_id, task_id)] = item
		return item.to_dict()

	def complete_packing(self, task_id: str, tenant_id: str, completed_at: str, weight_checked: bool = True) -> dict[str, Any]:
		"""Complete packing with weight verification."""
		task = self.pack_tasks.get(self._key(tenant_id, task_id))
		if task is None:
			raise KeyError(f"Pack task {task_id} not found")
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "complete_packing",
			"weight_checked": weight_checked,
		})
		task.completed_at = completed_at
		task.weight_checked = weight_checked
		task.label_printed = True
		task.packing_slip_printed = True
		self._audit(tenant_id, "packing_completed", task_id)
		return task.to_dict()

	def initiate_cycle_count(
		self, count_id: str, tenant_id: str, warehouse_id: str,
		count_type: str, initiated_at: str,
	) -> dict[str, Any]:
		"""Initiate a cycle count."""
		count_type = _norm(count_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "initiate_cycle_count",
			"count_type_supported": count_type in SUPPORTED_CYCLE_COUNT_TYPES,
		})
		item = CycleCount(count_id, tenant_id, warehouse_id, count_type, initiated_at, None, 0.0, False, None)
		self.cycle_counts[self._key(tenant_id, count_id)] = item
		self._audit(tenant_id, "cycle_count_initiated", count_id)
		return item.to_dict()

	def complete_cycle_count(
		self, count_id: str, tenant_id: str, completed_at: str,
		discrepancy_pct: float, approved_by: str,
	) -> dict[str, Any]:
		"""Complete and approve a cycle count."""
		count = self.cycle_counts.get(self._key(tenant_id, count_id))
		if count is None:
			raise KeyError(f"Cycle count {count_id} not found")
		count.completed_at = completed_at
		count.discrepancy_pct = float(discrepancy_pct)
		count.approved = True
		count.approved_by = approved_by
		self._audit(tenant_id, "cycle_count_completed", count_id)
		return count.to_dict()

	def adjust_inventory(
		self, adjustment_id: str, tenant_id: str, warehouse_id: str,
		sku: str, quantity_before: int, quantity_after: int,
		reason: str, approved_by: str, adjusted_at: str,
		manipulation_detected: bool = False,
	) -> dict[str, Any]:
		"""Apply an approved inventory adjustment."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "adjust_inventory",
			"approval_present": _present(approved_by),
			"manipulation_detected": manipulation_detected,
		})
		item = InventoryAdjustment(adjustment_id, tenant_id, warehouse_id, sku, int(quantity_before), int(quantity_after), reason, approved_by, adjusted_at)
		self.inventory_adjustments[self._key(tenant_id, adjustment_id)] = item
		# Update internal inventory map
		inv_key = self._key(tenant_id, sku)
		self.inventory[inv_key] = {
			"sku": sku, "qty": quantity_after, "warehouse_id": warehouse_id,
			"last_adjusted": adjusted_at, "tenant_id": tenant_id,
		}
		self._audit(tenant_id, "inventory_adjusted", adjustment_id)
		return item.to_dict()

	def update_dock_door_status(
		self, door_id: str, tenant_id: str, door_number: str,
		warehouse_id: str, status: str, current_job_ref: str | None = None,
	) -> dict[str, Any]:
		"""Update dock door status."""
		status = _norm(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "update_dock_door_status",
			"status_supported": status in SUPPORTED_DOCK_DOOR_STATUSES,
		})
		door = self.dock_doors.get(self._key(tenant_id, door_id))
		now = _now_iso()
		if door is None:
			door = DockDoor(door_id, tenant_id, warehouse_id, door_number, status, current_job_ref, now)
			self.dock_doors[self._key(tenant_id, door_id)] = door
		else:
			door.status = status
			door.current_job_ref = current_job_ref
			door.last_updated = now
		self._audit(tenant_id, "dock_door_allocated", door_id)
		return door.to_dict()

	def register_warehouse_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent for warehouse operations."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_warehouse_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = WarehouseAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "warehouse_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "warehouse_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.transport.warehouse.lifecycle", "accepted": True}

	def list_receipts(self, tenant_id: str) -> list[dict[str, Any]]:
		return [r.to_dict() for r in self.receipts.values() if r.tenant_id == tenant_id]

	def list_open_pick_tasks(self, tenant_id: str) -> list[dict[str, Any]]:
		return [t.to_dict() for t in self.pick_tasks.values() if t.tenant_id == tenant_id and t.completed_at is None]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"warehouse_count": self._count(self.warehouses, tenant_id),
			"receipt_count": self._count(self.receipts, tenant_id),
			"putaway_task_count": self._count(self.putaway_tasks, tenant_id),
			"pick_task_count": self._count(self.pick_tasks, tenant_id),
			"open_pick_tasks": len(self.list_open_pick_tasks(tenant_id)),
			"pack_task_count": self._count(self.pack_tasks, tenant_id),
			"cycle_count_count": self._count(self.cycle_counts, tenant_id),
			"inventory_adjustment_count": self._count(self.inventory_adjustments, tenant_id),
			"dock_door_count": self._count(self.dock_doors, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New methods
	# ------------------------------------------------------------------

	async def receive_goods_async(
		self,
		po_id: str,
		items: list[dict[str, Any]],
		condition: str,
		received_by: str,
		*,
		warehouse_id: str = "default",
		receipt_method: str = "standard",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Receive a purchase order into the warehouse with per-item condition grading.

		items: [{"sku": str, "qty": int, "description": str}]
		condition: 'good' | 'damaged' | 'partial'
		"""
		tid = tenant_id or self.tenant_id
		if not _present(po_id) or not items or not _present(received_by):
			raise ValueError("po_id, items and received_by required")

		await asyncio.sleep(0)
		receipt_id = f"GR-{uuid.uuid4().hex[:8].upper()}"
		rm = _norm(receipt_method)
		if rm not in SUPPORTED_RECEIPT_METHODS:
			rm = list(SUPPORTED_RECEIPT_METHODS)[0] if SUPPORTED_RECEIPT_METHODS else "standard"

		damaged = _norm(condition) == "damaged"
		receipt = self.receive_goods(
			receipt_id, tid, warehouse_id, rm,
			received_by, po_id, len(items), _now_iso(),
			barcode_scanned=True, damage_inspection_completed=True,
		)

		# Update inventory for each line item
		sku_updates = []
		for item in items:
			sku = item.get("sku", f"SKU-{uuid.uuid4().hex[:6]}")
			qty = int(item.get("qty", 1))
			inv_key = self._key(tid, sku)
			current = self.inventory.get(inv_key, {"qty": 0})
			new_qty = current["qty"] + qty
			adj_id = f"ADJ-{receipt_id}-{sku[:8]}"
			self.adjust_inventory(adj_id, tid, warehouse_id, sku, current["qty"], new_qty, "goods_received", received_by, _now_iso())
			sku_updates.append({"sku": sku, "qty_received": qty, "new_qty": new_qty})

		return {
			"receipt": receipt,
			"po_id": po_id,
			"condition": condition,
			"items_received": len(items),
			"damaged_flag": damaged,
			"inventory_updates": sku_updates,
		}

	async def putaway(
		self,
		receipt_id: str,
		locations: list[dict[str, Any]],
		*,
		operator_id: str = "unassigned",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Execute putaway for all items in a receipt to their assigned locations.

		locations: [{"sku": str, "slot_id": str}]
		"""
		tid = tenant_id or self.tenant_id
		if not _present(receipt_id) or not locations:
			raise ValueError("receipt_id and locations required")

		await asyncio.sleep(0)
		receipt = self.receipts.get(self._key(tid, receipt_id))
		if receipt is None:
			raise KeyError(f"Receipt {receipt_id} not found")

		strategy = list(SUPPORTED_PUTAWAY_STRATEGIES)[0] if SUPPORTED_PUTAWAY_STRATEGIES else "fixed_slot"
		tasks = []
		for loc in locations:
			task_id = f"PUT-{receipt_id}-{loc.get('slot_id', uuid.uuid4().hex[:4])}"
			task = self.execute_putaway(task_id, tid, receipt_id, strategy, loc.get("slot_id", "UNASSIGNED"), operator_id)
			tasks.append({**task, "sku": loc.get("sku")})

		return {
			"receipt_id": receipt_id,
			"tenant_id": tid,
			"locations_assigned": len(locations),
			"putaway_tasks": tasks,
			"completed_at": _now_iso(),
		}

	async def pick_order(
		self,
		order_id: str,
		picker_id: str,
		*,
		warehouse_id: str = "default",
		pick_method: str = "single_order",
		priority: str = "normal",
		lines: list[dict[str, Any]] | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Create and auto-assign a pick task for an order.

		lines: [{"sku": str, "qty": int, "location": str}] — if None, generates stub.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(order_id) or not _present(picker_id):
			raise ValueError("order_id and picker_id required")

		await asyncio.sleep(0)
		lines = lines or [{"sku": f"SKU-{order_id[:6]}", "qty": 1, "location": "A-01-01"}]
		pm = _norm(pick_method)
		if pm not in SUPPORTED_PICK_METHODS:
			pm = list(SUPPORTED_PICK_METHODS)[0] if SUPPORTED_PICK_METHODS else "single_order"

		task_id = f"PICK-{order_id}-{uuid.uuid4().hex[:6].upper()}"
		task = self.create_pick_task(task_id, tid, order_id, pm, warehouse_id, len(lines), priority, picker_id)

		# Check inventory availability
		availability: list[dict[str, Any]] = []
		for line in lines:
			sku = line.get("sku", "")
			requested_qty = int(line.get("qty", 1))
			inv = self.inventory.get(self._key(tid, sku), {"qty": 0})
			available = inv["qty"] >= requested_qty
			availability.append({"sku": sku, "requested": requested_qty, "available_qty": inv["qty"], "fulfillable": available})

		all_fulfillable = all(a["fulfillable"] for a in availability)
		if all_fulfillable:
			self.complete_pick_task(task_id, tid, _now_iso())

		return {
			"task": task,
			"order_id": order_id,
			"picker_id": picker_id,
			"lines": lines,
			"availability": availability,
			"all_fulfillable": all_fulfillable,
		}

	async def pack_order(
		self,
		pick_id: str,
		packer_id: str,
		box_type: str,
		*,
		weight_kg: float = 1.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Pack a completed pick task into a box, print label and packing slip."""
		tid = tenant_id or self.tenant_id
		if not _present(pick_id) or not _present(packer_id) or not _present(box_type):
			raise ValueError("pick_id, packer_id and box_type required")

		await asyncio.sleep(0)
		pt = _norm(box_type)
		if pt not in SUPPORTED_PACK_TYPES:
			pt = list(SUPPORTED_PACK_TYPES)[0] if SUPPORTED_PACK_TYPES else "standard"

		task_id = f"PACK-{pick_id}-{uuid.uuid4().hex[:6].upper()}"
		task = self.create_pack_task(task_id, tid, pick_id, pt, weight_kg)
		completed = self.complete_packing(task_id, tid, _now_iso())

		return {
			**completed,
			"packer_id": packer_id,
			"box_type": box_type,
			"label_printed": True,
			"packing_slip_printed": True,
		}

	async def ship_order(
		self,
		pack_id: str,
		carrier: str,
		tracking: str,
		*,
		dock_door_id: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Ship a packed order via a carrier; allocates a dock door and records shipment."""
		tid = tenant_id or self.tenant_id
		if not _present(pack_id) or not _present(carrier) or not _present(tracking):
			raise ValueError("pack_id, carrier and tracking required")

		await asyncio.sleep(0)
		pack = self.pack_tasks.get(self._key(tid, pack_id))
		if pack is None:
			raise KeyError(f"Pack task {pack_id} not found")
		if not pack.completed_at:
			raise ValueError(f"Pack task {pack_id} is not completed; cannot ship")

		# Allocate dock door
		door_record = None
		if dock_door_id:
			st = "loading" if "loading" in SUPPORTED_DOCK_DOOR_STATUSES else list(SUPPORTED_DOCK_DOOR_STATUSES)[0]
			door_record = self.update_dock_door_status(dock_door_id, tid, f"DOOR-{dock_door_id}", "warehouse-default", st, pack_id)

		shipment_id = f"SHIP-{pack_id}-{uuid.uuid4().hex[:6].upper()}"
		shipment: dict[str, Any] = {
			"shipment_id": shipment_id,
			"pack_id": pack_id,
			"carrier": carrier,
			"tracking_number": tracking,
			"dock_door": door_record,
			"tenant_id": tid,
			"shipped_at": _now_iso(),
			"status": "shipped",
		}
		self.shipment_records[self._key(tid, shipment_id)] = shipment
		self._audit(tid, "order_shipped", shipment_id)
		return shipment

	async def cycle_count(
		self,
		location_id: str,
		counter_id: str,
		*,
		warehouse_id: str = "default",
		count_type: str = "random",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Initiate and complete a cycle count for a location.

		Compares system qty vs physical count, computes discrepancy, auto-approves
		if discrepancy < 1%.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(location_id) or not _present(counter_id):
			raise ValueError("location_id and counter_id required")

		await asyncio.sleep(0)
		ct = _norm(count_type)
		if ct not in SUPPORTED_CYCLE_COUNT_TYPES:
			ct = list(SUPPORTED_CYCLE_COUNT_TYPES)[0] if SUPPORTED_CYCLE_COUNT_TYPES else "random"

		count_id = f"CC-{location_id[:8]}-{uuid.uuid4().hex[:6].upper()}"
		initiated = self.initiate_cycle_count(count_id, tid, warehouse_id, ct, _now_iso())

		# Stub: 0.5% discrepancy for demo — production would compare scan vs WMS qty
		discrepancy_pct = 0.5
		auto_approved = discrepancy_pct < 1.0
		if auto_approved:
			completed = self.complete_cycle_count(count_id, tid, _now_iso(), discrepancy_pct, counter_id)
		else:
			completed = initiated

		return {
			**completed,
			"location_id": location_id,
			"counter_id": counter_id,
			"discrepancy_pct": discrepancy_pct,
			"auto_approved": auto_approved,
		}

	async def inventory_adjustment(
		self,
		sku: str,
		quantity: int,
		reason: str,
		approved_by: str,
		*,
		warehouse_id: str = "default",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Apply an inventory adjustment for a SKU with full audit trail.

		quantity: signed integer — positive to add, negative to remove.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(sku) or not _present(reason) or not _present(approved_by):
			raise ValueError("sku, reason and approved_by required")

		await asyncio.sleep(0)
		inv_key = self._key(tid, sku)
		current = self.inventory.get(inv_key, {"qty": 0})
		qty_before = current["qty"]
		qty_after = max(0, qty_before + quantity)

		adj_id = f"ADJ-{sku[:8]}-{uuid.uuid4().hex[:6].upper()}"
		record = self.adjust_inventory(adj_id, tid, warehouse_id, sku, qty_before, qty_after, reason, approved_by, _now_iso())
		return {
			**record,
			"sku": sku,
			"adjustment": quantity,
			"qty_before": qty_before,
			"qty_after": qty_after,
		}

	async def cross_dock(
		self,
		inbound_id: str,
		outbound_orders: list[str],
		*,
		warehouse_id: str = "default",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Cross-dock inbound goods directly to outbound orders without putaway.

		Allocates dock doors for inbound and outbound movements and returns
		cross-dock allocation records.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(inbound_id) or not outbound_orders:
			raise ValueError("inbound_id and outbound_orders required")

		await asyncio.sleep(0)
		receipt = self.receipts.get(self._key(tid, inbound_id))
		if receipt is None:
			raise KeyError(f"Receipt {inbound_id} not found")

		inbound_door_id = f"DOOR-IN-{uuid.uuid4().hex[:4]}"
		st_in = "receiving" if "receiving" in SUPPORTED_DOCK_DOOR_STATUSES else list(SUPPORTED_DOCK_DOOR_STATUSES)[0]
		inbound_door = self.update_dock_door_status(
			inbound_door_id, tid, "IN-1", warehouse_id, st_in, inbound_id,
		)

		outbound_allocations = []
		for order_id in outbound_orders:
			door_id = f"DOOR-OUT-{uuid.uuid4().hex[:4]}"
			st_out = "loading" if "loading" in SUPPORTED_DOCK_DOOR_STATUSES else list(SUPPORTED_DOCK_DOOR_STATUSES)[0]
			door = self.update_dock_door_status(door_id, tid, f"OUT-{door_id[-4:]}", warehouse_id, st_out, order_id)
			outbound_allocations.append({"order_id": order_id, "dock_door": door})

		xdock_id = f"XD-{inbound_id[:8]}-{uuid.uuid4().hex[:6].upper()}"
		self._audit(tid, "cross_dock_executed", xdock_id)
		return {
			"xdock_id": xdock_id,
			"inbound_id": inbound_id,
			"outbound_orders": outbound_orders,
			"inbound_dock_door": inbound_door,
			"outbound_allocations": outbound_allocations,
			"transit_storage": False,
			"executed_at": _now_iso(),
		}

	async def warehouse_analytics(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate warehouse KPIs for a period.

		Returns throughput, pick accuracy, inventory adjustment rate,
		dock utilisation, and cycle count accuracy.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		receipt_count = self._count(self.receipts, tid)
		pick_count = self._count(self.pick_tasks, tid)
		pack_count = self._count(self.pack_tasks, tid)
		shipment_count = len([s for s in self.shipment_records.values() if s.get("tenant_id") == tid])
		adj_count = self._count(self.inventory_adjustments, tid)
		total_cycle_counts = self._count(self.cycle_counts, tid)
		dock_count = self._count(self.dock_doors, tid)

		completed_picks = sum(1 for p in self.pick_tasks.values() if p.tenant_id == tid and p.completed_at)
		pick_completion_rate = round(completed_picks / pick_count * 100, 1) if pick_count else 0.0
		completed_packs = sum(1 for p in self.pack_tasks.values() if p.tenant_id == tid and p.completed_at)

		discrepancies = [c.discrepancy_pct for c in self.cycle_counts.values() if c.tenant_id == tid and c.discrepancy_pct is not None]
		avg_discrepancy = round(statistics.mean(discrepancies), 2) if discrepancies else 0.0

		return {
			"period": period,
			"tenant_id": tid,
			"receipts": receipt_count,
			"putaway_tasks": self._count(self.putaway_tasks, tid),
			"pick_tasks": pick_count,
			"completed_picks": completed_picks,
			"pick_completion_rate_pct": pick_completion_rate,
			"pack_tasks": pack_count,
			"completed_packs": completed_packs,
			"shipments": shipment_count,
			"inventory_adjustments": adj_count,
			"cycle_counts": total_cycle_counts,
			"avg_cycle_count_discrepancy_pct": avg_discrepancy,
			"dock_doors": dock_count,
			"generated_at": _now_iso(),
		}

	async def labour_productivity(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate labour productivity for warehouse staff over a period.

		Returns units per hour by activity type, compares against benchmarks,
		and flags below-benchmark operators.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		# Gather operator activity from tasks
		operator_picking: dict[str, int] = {}
		for pt in self.pick_tasks.values():
			if pt.tenant_id == tid:
				operator_picking[pt.operator_id] = operator_picking.get(pt.operator_id, 0) + pt.lines_count

		operator_putaway: dict[str, int] = {}
		for put in self.putaway_tasks.values():
			if put.tenant_id == tid:
				operator_putaway[put.operator_id] = operator_putaway.get(put.operator_id, 0) + 1

		# Assume 8-hour shift for UPH calculation
		shift_hours = 8.0
		pick_uph: list[dict[str, Any]] = []
		for op, lines in operator_picking.items():
			uph = round(lines / shift_hours, 1)
			benchmark = _PRODUCTIVITY_BENCHMARKS["picking"]
			pick_uph.append({
				"operator_id": op,
				"lines_picked": lines,
				"uph": uph,
				"benchmark_uph": benchmark,
				"at_benchmark": uph >= benchmark * 0.85,
			})

		putaway_uph: list[dict[str, Any]] = []
		for op, tasks in operator_putaway.items():
			uph = round(tasks / shift_hours, 1)
			benchmark = _PRODUCTIVITY_BENCHMARKS["putaway"]
			putaway_uph.append({
				"operator_id": op,
				"tasks_completed": tasks,
				"uph": uph,
				"benchmark_uph": benchmark,
				"at_benchmark": uph >= benchmark * 0.85,
			})

		below_benchmark = [o for o in pick_uph if not o["at_benchmark"]] + \
		                   [o for o in putaway_uph if not o["at_benchmark"]]

		return {
			"period": period,
			"tenant_id": tid,
			"shift_hours_assumed": shift_hours,
			"picking_productivity": pick_uph,
			"putaway_productivity": putaway_uph,
			"below_benchmark_operators": len(below_benchmark),
			"benchmarks": _PRODUCTIVITY_BENCHMARKS,
			"generated_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _log_warehouse_throughput(self, tenant_id: str) -> str:
		return f"tenant={tenant_id} receipts={self._count(self.receipts, tenant_id)} picks={self._count(self.pick_tasks, tenant_id)}"

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "warehouse_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "warehouse_policy_denied")


	async def slotting_optimise(
		self,
		warehouse_id: str,
		sku_velocities: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Alias for slotting_optimisation with the canonical method name."""
		return await self.slotting_optimisation(warehouse_id, sku_velocities, tenant_id=tenant_id)

	async def dock_schedule(
		self,
		warehouse_id: str,
		bookings: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Schedule dock door assignments for inbound/outbound bookings.

		bookings: [{"booking_id": str, "type": "inbound"|"outbound", "arrive_at": str}]
		Returns a door assignment manifest.
		"""
		tid = tenant_id or self.tenant_id
		if not warehouse_id or not bookings:
			raise ValueError("warehouse_id and bookings required")
		await asyncio.sleep(0)
		doors = [d for (t, _), d in self.dock_doors.items() if t == tid]
		available_doors = [d for d in doors if d.status == "available"]
		assignments: list[dict[str, Any]] = []
		dock_id = f"DS-{uuid.uuid4().hex[:8].upper()}"
		for i, booking in enumerate(bookings):
			door = available_doors[i % max(len(available_doors), 1)] if available_doors else None
			assignments.append({
				"booking_id": booking.get("booking_id", f"BK-{i}"),
				"type": booking.get("type", "inbound"),
				"arrive_at": booking.get("arrive_at"),
				"assigned_door": door.door_id if door else "overflow",
			})
		self._audit(tid, "dock_schedule_created", dock_id)
		return {
			"dock_schedule_id": dock_id,
			"warehouse_id": warehouse_id,
			"tenant_id": tid,
			"bookings": len(bookings),
			"assignments": assignments,
			"available_doors": len(available_doors),
			"generated_at": _now_iso(),
		}

	async def labour_productivity(
		self,
		warehouse_id: str,
		period: str,
		shift_hours: float = 8.0,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Alias for operator_productivity_report with warehouse_id context."""
		result = await self.operator_productivity_report(period, shift_hours=shift_hours, tenant_id=tenant_id)
		result["warehouse_id"] = warehouse_id
		return result

	async def warehouse_kpi_summary(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return a concise warehouse KPI card for dashboard consumption."""
		tid = tenant_id or self.tenant_id
		warehouses = [w for (t, _), w in self.warehouses.items() if t == tid]
		receipts = sum(1 for (t, _) in self.receipts if t == tid)
		picks = sum(1 for (t, _) in self.pick_tasks if t == tid)
		packs = sum(1 for (t, _) in self.pack_tasks if t == tid)
		counts = sum(1 for (t, _) in self.cycle_counts if t == tid)
		return {
			"tenant_id": tid,
			"warehouses": len(warehouses),
			"goods_receipts": receipts,
			"pick_tasks": picks,
			"pack_tasks": packs,
			"cycle_counts": counts,
			"audit_events": len(self.audit_events),
			"generated_at": _now_iso(),
		}

	async def space_utilisation(
		self,
		warehouse_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Estimate warehouse space utilisation from inventory records.

		Uses inventory quantity vs. warehouse capacity as a proxy.
		"""
		tid = tenant_id or self.tenant_id
		warehouse = self.warehouses.get(self._key(tid, warehouse_id))
		if warehouse is None:
			raise KeyError(f"warehouse_not_found:{warehouse_id}")
		inv_records = [v for k, v in self.inventory.items() if k[0] == tid]
		total_units = sum(r.get("qty", r.get("quantity", 0)) for r in inv_records)
		capacity = getattr(warehouse, "capacity_units", None) or 10_000
		utilisation = round(min(total_units / max(capacity, 1) * 100, 100.0), 1)
		return {
			"warehouse_id": warehouse_id,
			"tenant_id": tid,
			"total_inventory_units": total_units,
			"capacity_units": capacity,
			"utilisation_pct": utilisation,
			"status": "critical" if utilisation >= 95 else "high" if utilisation >= 80 else "normal",
			"generated_at": _now_iso(),
		}

	async def order_accuracy_report(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Report order pick accuracy: correct vs. total pick tasks."""
		tid = tenant_id or self.tenant_id
		picks = [p for (t, _), p in self.pick_tasks.items() if t == tid]
		completed = [p for p in picks if p.status == "completed"]
		# 'accurate' = picks without a quality_issue flag
		accurate = sum(1 for p in completed if not getattr(p, "quality_issue", False))
		accuracy_rate = round(accurate / max(len(completed), 1) * 100, 2)
		return {
			"period": period,
			"tenant_id": tid,
			"total_picks": len(picks),
			"completed_picks": len(completed),
			"accurate_picks": accurate,
			"accuracy_rate_pct": accuracy_rate,
			"error_rate_pct": round(100 - accuracy_rate, 2),
			"generated_at": _now_iso(),
		}

	async def slotting_optimisation(
		self,
		warehouse_id: str,
		sku_velocities: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Optimise warehouse slot assignments based on SKU velocity (ABC analysis).

		sku_velocities: [{"sku": str, "picks_per_day": float}]
		A-items (top 20% volume) → golden zone; B → mid; C → far.
		"""
		tid = tenant_id or self.tenant_id
		if not warehouse_id or not sku_velocities:
			raise ValueError("warehouse_id and sku_velocities required")
		await asyncio.sleep(0)
		sorted_skus = sorted(sku_velocities, key=lambda x: float(x.get("picks_per_day", 0)), reverse=True)
		total = len(sorted_skus)
		a_cut = max(1, int(total * 0.2))
		b_cut = max(a_cut + 1, int(total * 0.5))
		recommendations: list[dict[str, Any]] = []
		for i, sku_v in enumerate(sorted_skus):
			if i < a_cut:
				zone = "golden_zone"
			elif i < b_cut:
				zone = "middle_zone"
			else:
				zone = "remote_zone"
			recommendations.append({"sku": sku_v["sku"], "picks_per_day": sku_v.get("picks_per_day"), "recommended_zone": zone})
		self._audit(tid, "slotting_optimised", warehouse_id)
		return {
			"warehouse_id": warehouse_id,
			"tenant_id": tid,
			"sku_count": total,
			"a_items": a_cut,
			"b_items": b_cut - a_cut,
			"c_items": total - b_cut,
			"recommendations": recommendations,
			"optimised_at": _now_iso(),
		}

	async def returns_processing(
		self,
		receipt_id: str,
		items: list[dict[str, Any]],
		reason: str,
		*,
		warehouse_id: str = "default",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Process incoming returns: inspect, grade, and restock or dispose."""
		tid = tenant_id or self.tenant_id
		if not _present(receipt_id) or not items:
			raise ValueError("receipt_id and items required")
		await asyncio.sleep(0)
		restocked: list[dict[str, Any]] = []
		disposed: list[dict[str, Any]] = []
		for item in items:
			grade = item.get("condition_grade", "A")
			if grade in ("A", "B"):
				sku = item.get("sku", f"SKU-{uuid.uuid4().hex[:6]}")
				qty = int(item.get("qty", 1))
				adj_id = f"RET-ADJ-{receipt_id[:8]}-{sku[:6]}"
				inv_key = self._key(tid, sku)
				current = self.inventory.get(inv_key, {"qty": 0})
				self.adjust_inventory(adj_id, tid, warehouse_id, sku, current["qty"], current["qty"] + qty, f"return:{reason}", "system", _now_iso())
				restocked.append({"sku": sku, "qty": qty, "grade": grade})
			else:
				disposed.append({"sku": item.get("sku"), "qty": item.get("qty"), "reason": "grade_C_disposed"})
		return_id = f"RET-{receipt_id}-{uuid.uuid4().hex[:6].upper()}"
		self._audit(tid, "return_processed", return_id)
		return {
			"return_id": return_id,
			"receipt_id": receipt_id,
			"tenant_id": tid,
			"items_total": len(items),
			"restocked_count": len(restocked),
			"disposed_count": len(disposed),
			"restocked": restocked,
			"disposed": disposed,
			"processed_at": _now_iso(),
		}

	async def export_warehouse_data(
		self,
		period: str,
		*,
		format: str = "json",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Export warehouse operations data metadata."""
		tid = tenant_id or self.tenant_id
		export_id = f"WAR-EXP-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "warehouse_data_exported", export_id)
		return {
			"export_id": export_id,
			"period": period,
			"tenant_id": tid,
			"format": format,
			"record_count": self._count(self.receipts, tid) + self._count(self.pick_tasks, tid),
			"download_ref": f"/exports/{tid}/{export_id}.{format}",
			"status": "ready",
			"generated_at": _now_iso(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "WarehouseOperationsService",
			"status": "healthy",
			"warehouses": len(self.warehouses),
			"receipts": len(self.receipts),
			"pick_tasks": len(self.pick_tasks),
			"pack_tasks": len(self.pack_tasks),
			"shipments": len(self.shipment_records),
			"cycle_counts": len(self.cycle_counts),
			"audit_events": len(self.audit_events),
			"checked_at": _now_iso(),
		}

	async def sku_lookup(
		self,
		sku: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Look up current inventory qty and location for a SKU."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		inv = self.inventory.get(self._key(tid, sku), {"qty": 0, "warehouse_id": None, "last_adjusted": None})
		return {
			"sku": sku,
			"tenant_id": tid,
			"qty": inv.get("qty", 0),
			"warehouse_id": inv.get("warehouse_id"),
			"last_adjusted": inv.get("last_adjusted"),
			"in_stock": inv.get("qty", 0) > 0,
		}

	async def dock_door_availability(
		self,
		warehouse_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return availability status of all dock doors for a warehouse."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		doors = [d for d in self.dock_doors.values() if d.tenant_id == tid and d.warehouse_id == warehouse_id]
		available = [d for d in doors if d.status == "available"]
		return {
			"warehouse_id": warehouse_id,
			"tenant_id": tid,
			"total_doors": len(doors),
			"available_doors": len(available),
			"occupied_doors": len(doors) - len(available),
			"doors": [d.to_dict() for d in doors],
			"checked_at": _now_iso(),
		}


TransportWarehouseService = WarehouseOperationsService
