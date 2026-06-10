"""Warehouse Management System async service (scm_wms)."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_wms"
BIN_TYPES = {"standard", "bulk", "cold", "hazmat", "quarantine", "oversize"}
PICK_METHODS = {"fifo", "fefo", "lifo", "zone", "wave", "batch"}
COUNT_METHODS = {"spot", "abc", "full", "zone"}
TASK_STATUSES = {"pending", "in_progress", "completed", "cancelled", "exception"}


class WarehouseManagementService:
	"""Async service for bin management, put-away rules, directed pick/pack/ship,
	cycle counting, cross-docking and slotting optimisation."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.warehouses: dict[str, dict[str, Any]] = {}
		self.bins: dict[str, dict[str, Any]] = {}
		self.inventory: dict[str, dict[str, Any]] = {}  # key = tenant:sku:bin_id
		self.putaway_tasks: dict[str, dict[str, Any]] = {}
		self.pick_tasks: dict[str, dict[str, Any]] = {}
		self.pack_tasks: dict[str, dict[str, Any]] = {}
		self.ship_tasks: dict[str, dict[str, Any]] = {}
		self.cycle_counts: dict[str, dict[str, Any]] = {}
		self.cross_docks: dict[str, dict[str, Any]] = {}
		self.slotting_optimisations: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self, tenant_id: str | None = None) -> str:
		t = tenant_id or self.tenant_id
		if not t:
			raise PermissionError("tenant_context_required")
		return t

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, status: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"status": status,
			"capability_id": CAPABILITY_ID,
			"emitted_at": self._now(),
		})

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"warehouse_count": len(self.warehouses),
			"bin_count": len(self.bins),
			"active_bins": sum(1 for b in self.bins.values() if b["status"] == "active"),
			"open_putaway": sum(1 for t in self.putaway_tasks.values() if t["status"] == "pending"),
			"open_picks": sum(1 for t in self.pick_tasks.values() if t["status"] in {"pending", "in_progress"}),
			"pending_cycles": sum(1 for c in self.cycle_counts.values() if c["status"] == "pending"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "scm",
			"version": "1.0.0",
			"description": "Bin management, put-away rules, directed pick/pack/ship, cycle counting, cross-docking, slotting optimisation",
			"bin_types": sorted(BIN_TYPES),
			"pick_methods": sorted(PICK_METHODS),
			"count_methods": sorted(COUNT_METHODS),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Warehouse management ──────────────────────────────────────────────────

	async def create_warehouse(
		self,
		name: str,
		code: str,
		address: dict[str, Any] | None = None,
		warehouse_type: str = "standard",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a warehouse."""
		tenant = self._tenant(tenant_id)
		for w in self.warehouses.values():
			if w["tenant_id"] == tenant and w["code"] == code:
				raise ValueError(f"warehouse code '{code}' already exists for tenant")
		record: dict[str, Any] = {
			"id": self._id("wh"),
			"type": "scm_wms_warehouse",
			"tenant_id": tenant,
			"name": name,
			"code": code,
			"address": deepcopy(address or {}),
			"warehouse_type": warehouse_type,
			"status": "active",
			"created_at": self._now(),
		}
		self.warehouses[record["id"]] = record
		self._emit(tenant, "warehouse_created", record["id"], "scm_wms_warehouse", "active")
		return deepcopy(record)

	async def list_warehouses(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List warehouses."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(w) for w in self.warehouses.values() if w["tenant_id"] == tenant]

	async def get_warehouse(self, warehouse_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single warehouse."""
		tenant = self._tenant(tenant_id)
		w = self.warehouses.get(warehouse_id)
		if not w or w["tenant_id"] != tenant:
			raise KeyError(f"warehouse '{warehouse_id}' not found")
		return deepcopy(w)

	# ── Bin management ────────────────────────────────────────────────────────

	async def create_bin(
		self,
		warehouse_id: str,
		aisle: str,
		bay: str,
		level: str,
		bin_code: str,
		bin_type: str = "standard",
		capacity_units: float | None = None,
		capacity_weight_kg: float | None = None,
		pick_sequence: int | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a storage bin in a warehouse."""
		tenant = self._tenant(tenant_id)
		if bin_type not in BIN_TYPES:
			raise ValueError(f"bin_type must be one of {BIN_TYPES}")
		wh = self.warehouses.get(warehouse_id)
		if not wh or wh["tenant_id"] != tenant:
			raise KeyError(f"warehouse '{warehouse_id}' not found")
		# Enforce unique bin_code per warehouse
		for b in self.bins.values():
			if b["tenant_id"] == tenant and b["warehouse_id"] == warehouse_id and b["bin_code"] == bin_code:
				raise ValueError(f"bin_code '{bin_code}' already exists in warehouse '{warehouse_id}'")
		record: dict[str, Any] = {
			"id": self._id("bin"),
			"type": "scm_wms_bin",
			"tenant_id": tenant,
			"warehouse_id": warehouse_id,
			"aisle": aisle,
			"bay": bay,
			"level": level,
			"bin_code": bin_code,
			"bin_type": bin_type,
			"capacity_units": capacity_units,
			"capacity_weight_kg": capacity_weight_kg,
			"pick_sequence": pick_sequence,
			"current_qty": 0.0,
			"status": "active",
			"created_at": self._now(),
		}
		self.bins[record["id"]] = record
		self._emit(tenant, "bin_created", record["id"], "scm_wms_bin", "active")
		return deepcopy(record)

	async def list_bins(
		self,
		warehouse_id: str | None = None,
		bin_type: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List bins with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(b) for b in self.bins.values() if b["tenant_id"] == tenant]
		if warehouse_id:
			items = [b for b in items if b["warehouse_id"] == warehouse_id]
		if bin_type:
			items = [b for b in items if b["bin_type"] == bin_type]
		return items

	async def get_bin(self, bin_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single bin."""
		tenant = self._tenant(tenant_id)
		b = self.bins.get(bin_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"bin '{bin_id}' not found")
		return deepcopy(b)

	async def update_bin(
		self,
		bin_id: str,
		updates: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update bin attributes."""
		tenant = self._tenant(tenant_id)
		b = self.bins.get(bin_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"bin '{bin_id}' not found")
		allowed = {"bin_type", "capacity_units", "capacity_weight_kg", "pick_sequence", "status"}
		for k, v in updates.items():
			if k in allowed:
				b[k] = v
		self._emit(tenant, "bin_updated", bin_id, "scm_wms_bin", b["status"])
		return deepcopy(b)

	async def delete_bin(self, bin_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Deactivate a bin."""
		tenant = self._tenant(tenant_id)
		b = self.bins.get(bin_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"bin '{bin_id}' not found")
		if b["current_qty"] > 0:
			raise ValueError("cannot deactivate a bin with stock; relocate first")
		b["status"] = "inactive"
		self._emit(tenant, "bin_deactivated", bin_id, "scm_wms_bin", "inactive")
		return deepcopy(b)

	async def suggest_putaway_bin(
		self,
		warehouse_id: str,
		sku: str,
		quantity: float,
		bin_type: str = "standard",
		tenant_id: str | None = None,
	) -> dict[str, Any] | None:
		"""Suggest the best available bin for put-away based on capacity and pick sequence."""
		tenant = self._tenant(tenant_id)
		candidates = [
			b for b in self.bins.values()
			if b["tenant_id"] == tenant
			and b["warehouse_id"] == warehouse_id
			and b["bin_type"] == bin_type
			and b["status"] == "active"
			and (b["capacity_units"] is None or (b["capacity_units"] - b["current_qty"]) >= quantity)
		]
		if not candidates:
			return None
		# Sort by pick sequence (ascending), then by current_qty (ascending = emptier first)
		candidates.sort(key=lambda x: (x["pick_sequence"] or 9999, x["current_qty"]))
		return deepcopy(candidates[0])

	# ── Put-away ──────────────────────────────────────────────────────────────

	async def create_putaway_task(
		self,
		receipt_id: str,
		sku: str,
		quantity: float,
		bin_id: str | None = None,
		assigned_to: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a put-away task for inbound goods."""
		tenant = self._tenant(tenant_id)
		suggested_bin_id = bin_id
		if not suggested_bin_id:
			# Try to auto-suggest — skip if no warehouses configured
			suggested = None
			for wh_id in [w["id"] for w in self.warehouses.values() if w["tenant_id"] == tenant]:
				suggested = await self.suggest_putaway_bin(wh_id, sku, quantity, tenant_id=tenant)
				if suggested:
					suggested_bin_id = suggested["id"]
					break
		record: dict[str, Any] = {
			"id": self._id("put"),
			"type": "scm_wms_putaway_task",
			"tenant_id": tenant,
			"receipt_id": receipt_id,
			"sku": sku,
			"quantity": quantity,
			"suggested_bin_id": suggested_bin_id,
			"confirmed_bin_id": None,
			"assigned_to": assigned_to,
			"status": "pending",
			"created_at": self._now(),
			"completed_at": None,
		}
		self.putaway_tasks[record["id"]] = record
		self._emit(tenant, "putaway_task_created", record["id"], "scm_wms_putaway_task", "pending")
		return deepcopy(record)

	async def complete_putaway_task(
		self,
		task_id: str,
		confirmed_bin_id: str,
		completed_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Confirm put-away of goods into a bin, updating inventory."""
		tenant = self._tenant(tenant_id)
		task = self.putaway_tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"putaway_task '{task_id}' not found")
		if task["status"] == "completed":
			raise ValueError("task already completed")
		bin_rec = self.bins.get(confirmed_bin_id)
		if not bin_rec or bin_rec["tenant_id"] != tenant:
			raise KeyError(f"bin '{confirmed_bin_id}' not found")
		task["confirmed_bin_id"] = confirmed_bin_id
		task["completed_by"] = completed_by
		task["status"] = "completed"
		task["completed_at"] = self._now()
		# Update bin quantity
		bin_rec["current_qty"] = bin_rec.get("current_qty", 0.0) + task["quantity"]
		# Update inventory ledger
		inv_key = f"{tenant}:{task['sku']}:{confirmed_bin_id}"
		inv = self.inventory.setdefault(inv_key, {
			"tenant_id": tenant, "sku": task["sku"], "bin_id": confirmed_bin_id, "quantity": 0.0,
		})
		inv["quantity"] += task["quantity"]
		inv["last_updated"] = self._now()
		self._emit(tenant, "putaway_completed", task_id, "scm_wms_putaway_task", "completed")
		return deepcopy(task)

	async def list_putaway_tasks(
		self,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List put-away tasks."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(t) for t in self.putaway_tasks.values() if t["tenant_id"] == tenant]
		if status:
			items = [t for t in items if t["status"] == status]
		return items

	# ── Pick ──────────────────────────────────────────────────────────────────

	async def create_pick_task(
		self,
		order_id: str,
		sku: str,
		quantity: float,
		bin_id: str,
		assigned_to: str | None = None,
		pick_method: str = "fifo",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a directed pick task."""
		tenant = self._tenant(tenant_id)
		if pick_method not in PICK_METHODS:
			raise ValueError(f"pick_method must be one of {PICK_METHODS}")
		bin_rec = self.bins.get(bin_id)
		if not bin_rec or bin_rec["tenant_id"] != tenant:
			raise KeyError(f"bin '{bin_id}' not found")
		inv_key = f"{tenant}:{sku}:{bin_id}"
		available = self.inventory.get(inv_key, {}).get("quantity", 0.0)
		if available < quantity:
			raise ValueError(f"insufficient stock in bin: available={available}, requested={quantity}")
		record: dict[str, Any] = {
			"id": self._id("pick"),
			"type": "scm_wms_pick_task",
			"tenant_id": tenant,
			"order_id": order_id,
			"sku": sku,
			"quantity": quantity,
			"picked_quantity": 0.0,
			"bin_id": bin_id,
			"assigned_to": assigned_to,
			"pick_method": pick_method,
			"status": "pending",
			"created_at": self._now(),
			"completed_at": None,
		}
		self.pick_tasks[record["id"]] = record
		self._emit(tenant, "pick_task_created", record["id"], "scm_wms_pick_task", "pending")
		return deepcopy(record)

	async def complete_pick_task(
		self,
		task_id: str,
		picked_quantity: float,
		completed_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record pick completion and deduct inventory."""
		tenant = self._tenant(tenant_id)
		task = self.pick_tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"pick_task '{task_id}' not found")
		if task["status"] == "completed":
			raise ValueError("pick task already completed")
		task["picked_quantity"] = picked_quantity
		task["completed_by"] = completed_by
		task["status"] = "completed" if picked_quantity >= task["quantity"] else "exception"
		task["completed_at"] = self._now()
		# Deduct inventory
		inv_key = f"{tenant}:{task['sku']}:{task['bin_id']}"
		inv = self.inventory.get(inv_key)
		if inv:
			inv["quantity"] = max(0.0, inv["quantity"] - picked_quantity)
			inv["last_updated"] = self._now()
		bin_rec = self.bins.get(task["bin_id"])
		if bin_rec:
			bin_rec["current_qty"] = max(0.0, bin_rec.get("current_qty", 0.0) - picked_quantity)
		self._emit(tenant, "pick_completed", task_id, "scm_wms_pick_task", task["status"])
		return deepcopy(task)

	async def list_pick_tasks(
		self,
		order_id: str | None = None,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List pick tasks."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(t) for t in self.pick_tasks.values() if t["tenant_id"] == tenant]
		if order_id:
			items = [t for t in items if t["order_id"] == order_id]
		if status:
			items = [t for t in items if t["status"] == status]
		return items

	# ── Pack ──────────────────────────────────────────────────────────────────

	async def create_pack_task(
		self,
		order_id: str,
		pick_task_ids: list[str],
		packing_station: str | None = None,
		assigned_to: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a pack task consolidating picked items."""
		tenant = self._tenant(tenant_id)
		if not pick_task_ids:
			raise ValueError("pick_task_ids must not be empty")
		record: dict[str, Any] = {
			"id": self._id("pack"),
			"type": "scm_wms_pack_task",
			"tenant_id": tenant,
			"order_id": order_id,
			"pick_task_ids": pick_task_ids,
			"packing_station": packing_station,
			"assigned_to": assigned_to,
			"cartons": [],
			"total_weight_kg": None,
			"status": "pending",
			"created_at": self._now(),
			"completed_at": None,
		}
		self.pack_tasks[record["id"]] = record
		self._emit(tenant, "pack_task_created", record["id"], "scm_wms_pack_task", "pending")
		return deepcopy(record)

	async def complete_pack_task(
		self,
		task_id: str,
		cartons: list[dict[str, Any]],
		total_weight_kg: float | None = None,
		completed_by: str = "system",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record pack completion with carton details."""
		tenant = self._tenant(tenant_id)
		task = self.pack_tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"pack_task '{task_id}' not found")
		task["cartons"] = deepcopy(cartons)
		task["total_weight_kg"] = total_weight_kg
		task["completed_by"] = completed_by
		task["status"] = "completed"
		task["completed_at"] = self._now()
		self._emit(tenant, "pack_completed", task_id, "scm_wms_pack_task", "completed")
		return deepcopy(task)

	async def list_pack_tasks(
		self,
		order_id: str | None = None,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List pack tasks."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(t) for t in self.pack_tasks.values() if t["tenant_id"] == tenant]
		if order_id:
			items = [t for t in items if t["order_id"] == order_id]
		if status:
			items = [t for t in items if t["status"] == status]
		return items

	# ── Ship ──────────────────────────────────────────────────────────────────

	async def create_ship_task(
		self,
		order_id: str,
		pack_task_id: str,
		carrier_id: str,
		destination_address: dict[str, Any],
		assigned_to: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a ship task after packing."""
		tenant = self._tenant(tenant_id)
		pack = self.pack_tasks.get(pack_task_id)
		if not pack or pack["tenant_id"] != tenant:
			raise KeyError(f"pack_task '{pack_task_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("ship"),
			"type": "scm_wms_ship_task",
			"tenant_id": tenant,
			"order_id": order_id,
			"pack_task_id": pack_task_id,
			"carrier_id": carrier_id,
			"destination_address": deepcopy(destination_address),
			"assigned_to": assigned_to,
			"tracking_number": None,
			"status": "pending",
			"created_at": self._now(),
			"dispatched_at": None,
		}
		self.ship_tasks[record["id"]] = record
		self._emit(tenant, "ship_task_created", record["id"], "scm_wms_ship_task", "pending")
		return deepcopy(record)

	async def dispatch_shipment(
		self,
		ship_task_id: str,
		tracking_number: str,
		dispatched_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark a ship task as dispatched."""
		tenant = self._tenant(tenant_id)
		task = self.ship_tasks.get(ship_task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"ship_task '{ship_task_id}' not found")
		task["tracking_number"] = tracking_number
		task["dispatched_by"] = dispatched_by
		task["status"] = "dispatched"
		task["dispatched_at"] = self._now()
		self._emit(tenant, "shipment_dispatched", ship_task_id, "scm_wms_ship_task", "dispatched")
		return deepcopy(task)

	# ── Cycle counting ────────────────────────────────────────────────────────

	async def create_cycle_count(
		self,
		warehouse_id: str,
		count_method: str = "spot",
		bin_ids: list[str] | None = None,
		assigned_to: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a cycle count job."""
		tenant = self._tenant(tenant_id)
		if count_method not in COUNT_METHODS:
			raise ValueError(f"count_method must be one of {COUNT_METHODS}")
		wh = self.warehouses.get(warehouse_id)
		if not wh or wh["tenant_id"] != tenant:
			raise KeyError(f"warehouse '{warehouse_id}' not found")
		effective_bins = bin_ids or [
			b["id"] for b in self.bins.values()
			if b["tenant_id"] == tenant and b["warehouse_id"] == warehouse_id and b["status"] == "active"
		]
		record: dict[str, Any] = {
			"id": self._id("cc"),
			"type": "scm_wms_cycle_count",
			"tenant_id": tenant,
			"warehouse_id": warehouse_id,
			"bin_ids": effective_bins,
			"count_method": count_method,
			"assigned_to": assigned_to,
			"results": [],
			"variance_items": 0,
			"status": "pending",
			"created_at": self._now(),
			"completed_at": None,
		}
		self.cycle_counts[record["id"]] = record
		self._emit(tenant, "cycle_count_created", record["id"], "scm_wms_cycle_count", "pending")
		return deepcopy(record)

	async def submit_cycle_count_results(
		self,
		count_id: str,
		results: list[dict[str, Any]],
		completed_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Submit counted quantities and compute variances."""
		tenant = self._tenant(tenant_id)
		cc = self.cycle_counts.get(count_id)
		if not cc or cc["tenant_id"] != tenant:
			raise KeyError(f"cycle_count '{count_id}' not found")
		enriched: list[dict[str, Any]] = []
		variance_count = 0
		for r in results:
			bin_id = r.get("bin_id", "")
			sku = r.get("sku", "")
			counted = float(r.get("counted_quantity", 0))
			inv_key = f"{tenant}:{sku}:{bin_id}"
			system_qty = self.inventory.get(inv_key, {}).get("quantity", 0.0)
			variance = counted - system_qty
			has_variance = abs(variance) > 0.001
			if has_variance:
				variance_count += 1
				# Adjust inventory to counted quantity
				inv = self.inventory.setdefault(inv_key, {"tenant_id": tenant, "sku": sku, "bin_id": bin_id, "quantity": 0.0})
				inv["quantity"] = counted
				inv["last_updated"] = self._now()
				bin_rec = self.bins.get(bin_id)
				if bin_rec:
					bin_rec["current_qty"] = max(0.0, bin_rec.get("current_qty", 0.0) + variance)
			enriched.append({**r, "system_quantity": system_qty, "variance": variance, "adjusted": has_variance})
		cc["results"] = enriched
		cc["variance_items"] = variance_count
		cc["completed_by"] = completed_by
		cc["status"] = "completed"
		cc["completed_at"] = self._now()
		self._emit(tenant, "cycle_count_completed", count_id, "scm_wms_cycle_count", "completed")
		return deepcopy(cc)

	async def list_cycle_counts(
		self,
		warehouse_id: str | None = None,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List cycle counts."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.cycle_counts.values() if c["tenant_id"] == tenant]
		if warehouse_id:
			items = [c for c in items if c["warehouse_id"] == warehouse_id]
		if status:
			items = [c for c in items if c["status"] == status]
		return items

	# ── Cross-docking ─────────────────────────────────────────────────────────

	async def create_cross_dock(
		self,
		inbound_shipment_id: str,
		outbound_order_id: str,
		sku: str,
		quantity: float,
		dock_door: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a cross-dock movement (inbound directly to outbound)."""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": self._id("xdock"),
			"type": "scm_wms_cross_dock",
			"tenant_id": tenant,
			"inbound_shipment_id": inbound_shipment_id,
			"outbound_order_id": outbound_order_id,
			"sku": sku,
			"quantity": quantity,
			"dock_door": dock_door,
			"status": "pending",
			"created_at": self._now(),
			"completed_at": None,
		}
		self.cross_docks[record["id"]] = record
		self._emit(tenant, "cross_dock_created", record["id"], "scm_wms_cross_dock", "pending")
		return deepcopy(record)

	async def complete_cross_dock(
		self,
		cross_dock_id: str,
		completed_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark a cross-dock as complete."""
		tenant = self._tenant(tenant_id)
		xd = self.cross_docks.get(cross_dock_id)
		if not xd or xd["tenant_id"] != tenant:
			raise KeyError(f"cross_dock '{cross_dock_id}' not found")
		xd["status"] = "completed"
		xd["completed_by"] = completed_by
		xd["completed_at"] = self._now()
		self._emit(tenant, "cross_dock_completed", cross_dock_id, "scm_wms_cross_dock", "completed")
		return deepcopy(xd)

	async def list_cross_docks(
		self,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List cross-dock records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(x) for x in self.cross_docks.values() if x["tenant_id"] == tenant]
		if status:
			items = [x for x in items if x["status"] == status]
		return items

	# ── Slotting optimisation ─────────────────────────────────────────────────

	async def run_slotting_optimisation(
		self,
		warehouse_id: str,
		optimisation_objective: str = "pick_distance",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Run slotting optimisation to minimise pick travel distance or balance bin utilisation."""
		tenant = self._tenant(tenant_id)
		wh = self.warehouses.get(warehouse_id)
		if not wh or wh["tenant_id"] != tenant:
			raise KeyError(f"warehouse '{warehouse_id}' not found")
		bins_in_wh = [b for b in self.bins.values() if b["tenant_id"] == tenant and b["warehouse_id"] == warehouse_id]
		total_bins = len(bins_in_wh)
		utilised = sum(1 for b in bins_in_wh if b["current_qty"] > 0)
		utilisation_pct = round(utilised / total_bins * 100, 1) if total_bins else 0.0
		# Simplified: assign pick_sequence based on fill rate (highest-velocity items closest to dispatch)
		suggestions: list[dict[str, Any]] = []
		for b in sorted(bins_in_wh, key=lambda x: x.get("current_qty", 0), reverse=True):
			suggestions.append({"bin_id": b["id"], "bin_code": b["bin_code"], "suggested_pick_seq": len(suggestions) + 1})
		record: dict[str, Any] = {
			"id": self._id("slot"),
			"type": "scm_wms_slotting_optimisation",
			"tenant_id": tenant,
			"warehouse_id": warehouse_id,
			"optimisation_objective": optimisation_objective,
			"total_bins": total_bins,
			"utilisation_pct": utilisation_pct,
			"suggestions": suggestions[:20],  # top 20
			"status": "completed",
			"run_at": self._now(),
		}
		self.slotting_optimisations[record["id"]] = record
		self._emit(tenant, "slotting_optimisation_completed", record["id"], "scm_wms_slotting_optimisation", "completed")
		return deepcopy(record)

	# ── Inventory queries ─────────────────────────────────────────────────────

	async def get_inventory(
		self,
		sku: str | None = None,
		warehouse_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Query current inventory levels."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(v) for v in self.inventory.values() if v["tenant_id"] == tenant]
		if sku:
			items = [i for i in items if i["sku"] == sku]
		if warehouse_id:
			bin_ids = {b["id"] for b in self.bins.values() if b["tenant_id"] == tenant and b["warehouse_id"] == warehouse_id}
			items = [i for i in items if i["bin_id"] in bin_ids]
		return items

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def warehouse_analytics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Aggregate WMS KPIs."""
		tenant = self._tenant(tenant_id)
		all_bins = [b for b in self.bins.values() if b["tenant_id"] == tenant and b["status"] == "active"]
		occupied = sum(1 for b in all_bins if b["current_qty"] > 0)
		utilisation = round(occupied / len(all_bins) * 100, 1) if all_bins else 0.0
		return {
			"tenant_id": tenant,
			"warehouses": len([w for w in self.warehouses.values() if w["tenant_id"] == tenant]),
			"total_bins": len(all_bins),
			"occupied_bins": occupied,
			"bin_utilisation_pct": utilisation,
			"pending_putaway": sum(1 for t in self.putaway_tasks.values() if t["tenant_id"] == tenant and t["status"] == "pending"),
			"open_picks": sum(1 for t in self.pick_tasks.values() if t["tenant_id"] == tenant and t["status"] in {"pending", "in_progress"}),
			"open_packs": sum(1 for t in self.pack_tasks.values() if t["tenant_id"] == tenant and t["status"] == "pending"),
			"pending_cross_docks": sum(1 for x in self.cross_docks.values() if x["tenant_id"] == tenant and x["status"] == "pending"),
			"cycle_count_variances": sum(c.get("variance_items", 0) for c in self.cycle_counts.values() if c["tenant_id"] == tenant and c["status"] == "completed"),
			"generated_at": self._now(),
		}

	async def bulk_create_bins(
		self,
		warehouse_id: str,
		bins_data: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-create multiple bins in a warehouse."""
		tenant = self._tenant(tenant_id)
		tasks = [self.create_bin(warehouse_id=warehouse_id, tenant_id=tenant, **b) for b in bins_data]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for item in raw:
			if isinstance(item, Exception):
				errors.append(str(item))
			else:
				results.append(item)
		return {"created": len(results), "failed": len(errors), "bins": results, "errors": errors}
