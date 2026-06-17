"""Organizational Management async service."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "hcm_org"
UNIT_TYPES = {"department", "division", "team", "branch", "region", "subsidiary", "company"}
POSITION_STATUSES = {"open", "filled", "frozen", "abolished"}
REPORTING_LINE_TYPES = {"direct", "dotted", "functional"}
RESTRUCTURING_STATUSES = {"draft", "proposed", "approved", "in_progress", "completed", "cancelled"}


class ORGService:
	"""Organizational Management — org chart, positions, reporting lines, headcount, restructuring."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.org_units: dict[str, dict[str, Any]] = {}
		self.positions: dict[str, dict[str, Any]] = {}
		self.reporting_lines: dict[str, dict[str, Any]] = {}
		self.headcount_plans: dict[str, dict[str, Any]] = {}
		self.restructurings: dict[str, dict[str, Any]] = {}
		self.span_of_control: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _uid(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _emit(self, tenant_id: str, event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"id": self._uid("evt"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"payload": deepcopy(payload),
			"emitted_at": self._now(),
		})

	def _compute_headcount(self, tenant_id: str, unit_id: str) -> int:
		"""Count positions in unit (and children) that are filled."""
		direct = sum(1 for p in self.positions.values() if p["tenant_id"] == tenant_id and p["org_unit_id"] == unit_id and p["status"] == "filled")
		children = [u["id"] for u in self.org_units.values() if u["tenant_id"] == tenant_id and u["parent_unit_id"] == unit_id]
		return direct + sum(self._compute_headcount(tenant_id, c) for c in children)

	def _get_ancestors(self, tenant_id: str, unit_id: str) -> list[str]:
		"""Return all ancestor unit IDs for a unit."""
		ancestors: list[str] = []
		current = unit_id
		seen: set[str] = set()
		while current:
			if current in seen:
				break
			seen.add(current)
			unit = self.org_units.get(current)
			if not unit or unit["tenant_id"] != tenant_id or not unit.get("parent_unit_id"):
				break
			ancestors.append(unit["parent_unit_id"])
			current = unit["parent_unit_id"]
		return ancestors

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"org_units": len(self.org_units),
			"positions": len(self.positions),
			"reporting_lines": len(self.reporting_lines),
			"restructurings": len(self.restructurings),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "hcm",
			"version": "1.0.0",
			"description": "Organizational Management — org chart, positions, reporting lines, headcount, restructuring",
			"unit_types": sorted(UNIT_TYPES),
			"position_statuses": sorted(POSITION_STATUSES),
			"reporting_line_types": sorted(REPORTING_LINE_TYPES),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]

	# ── Org Units ─────────────────────────────────────────────────────────────

	async def create_org_unit(
		self,
		tenant_id: str,
		name: str,
		code: str,
		unit_type: str,
		parent_unit_id: str | None = None,
		manager_employee_id: str | None = None,
		cost_centre: str | None = None,
		location: str | None = None,
		description: str | None = None,
	) -> dict[str, Any]:
		"""Create an organisational unit."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		guard_non_empty_string(code, "code")
		if unit_type not in UNIT_TYPES:
			raise ValueError(f"unit_type must be one of {UNIT_TYPES}")
		if parent_unit_id:
			parent = self.org_units.get(parent_unit_id)
			if not parent or parent["tenant_id"] != t:
				raise KeyError(f"parent_unit {parent_unit_id} not found")
		record: dict[str, Any] = {
			"id": self._uid("ou"),
			"tenant_id": t,
			"name": name,
			"code": code,
			"unit_type": unit_type,
			"parent_unit_id": parent_unit_id,
			"manager_employee_id": manager_employee_id,
			"cost_centre": cost_centre,
			"location": location,
			"description": description,
			"headcount": 0,
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.org_units[record["id"]] = record
		self._emit(t, "org_unit_created", "org_unit", record["id"], record)
		_log.info("org_unit created: %s name=%s", record["id"], name)
		return deepcopy(record)

	async def list_org_units(
		self,
		tenant_id: str,
		unit_type: str | None = None,
		parent_unit_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List org units with optional filters."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.org_units.values() if r["tenant_id"] == t]
		if unit_type:
			items = [r for r in items if r["unit_type"] == unit_type]
		if parent_unit_id:
			items = [r for r in items if r["parent_unit_id"] == parent_unit_id]
		if status:
			items = [r for r in items if r["status"] == status]
		# Attach live headcount
		for item in items:
			item["headcount"] = self._compute_headcount(t, item["id"])
		return items

	async def get_org_unit(self, tenant_id: str, unit_id: str) -> dict[str, Any]:
		"""Get org unit by ID."""
		t = self._tenant(tenant_id)
		record = self.org_units.get(unit_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"org_unit {unit_id} not found")
		result = deepcopy(record)
		result["headcount"] = self._compute_headcount(t, unit_id)
		return result

	async def update_org_unit(self, tenant_id: str, unit_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update an org unit."""
		t = self._tenant(tenant_id)
		record = self.org_units.get(unit_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"org_unit {unit_id} not found")
		allowed = {"name", "code", "manager_employee_id", "cost_centre", "location", "description", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		record["updated_at"] = self._now()
		self._emit(t, "org_unit_updated", "org_unit", record["id"], record)
		return deepcopy(record)

	async def delete_org_unit(self, tenant_id: str, unit_id: str) -> bool:
		"""Delete an org unit (only if no active positions)."""
		t = self._tenant(tenant_id)
		record = self.org_units.get(unit_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"org_unit {unit_id} not found")
		active_positions = [p for p in self.positions.values() if p["tenant_id"] == t and p["org_unit_id"] == unit_id and p["status"] != "abolished"]
		if active_positions:
			raise PermissionError("org_unit_has_active_positions")
		children = [u for u in self.org_units.values() if u["tenant_id"] == t and u["parent_unit_id"] == unit_id]
		if children:
			raise PermissionError("org_unit_has_child_units")
		del self.org_units[unit_id]
		self._emit(t, "org_unit_deleted", "org_unit", unit_id, {"id": unit_id})
		return True

	async def move_org_unit(self, tenant_id: str, unit_id: str, new_parent_id: str | None) -> dict[str, Any]:
		"""Move an org unit to a different parent."""
		t = self._tenant(tenant_id)
		record = self.org_units.get(unit_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"org_unit {unit_id} not found")
		if new_parent_id:
			if new_parent_id in self._get_ancestors(t, unit_id) + [unit_id]:
				raise ValueError("circular_hierarchy_detected")
			new_parent = self.org_units.get(new_parent_id)
			if not new_parent or new_parent["tenant_id"] != t:
				raise KeyError(f"new_parent_unit {new_parent_id} not found")
		record["parent_unit_id"] = new_parent_id
		record["updated_at"] = self._now()
		self._emit(t, "org_unit_moved", "org_unit", record["id"], {"unit_id": unit_id, "new_parent_id": new_parent_id})
		return deepcopy(record)

	async def get_org_chart(self, tenant_id: str, root_unit_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all org units as a flat list with parent references (for chart rendering)."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.org_units.values() if r["tenant_id"] == t and r["status"] == "active"]
		if root_unit_id:
			# Include root and all descendants
			def descendants(uid: str) -> set[str]:
				result = {uid}
				for u in self.org_units.values():
					if u["tenant_id"] == t and u["parent_unit_id"] == uid:
						result |= descendants(u["id"])
				return result
			ids = descendants(root_unit_id)
			items = [i for i in items if i["id"] in ids]
		for item in items:
			item["headcount"] = self._compute_headcount(t, item["id"])
		return items

	# ── Positions ─────────────────────────────────────────────────────────────

	async def create_position(
		self,
		tenant_id: str,
		title: str,
		code: str,
		org_unit_id: str,
		job_grade: str | None = None,
		reports_to_position_id: str | None = None,
		fte_count: float = 1.0,
		is_critical: bool = False,
		location: str | None = None,
		description: str | None = None,
	) -> dict[str, Any]:
		"""Create a new position."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		unit = self.org_units.get(org_unit_id)
		if not unit or unit["tenant_id"] != t:
			raise KeyError(f"org_unit {org_unit_id} not found")
		record: dict[str, Any] = {
			"id": self._uid("pos"),
			"tenant_id": t,
			"title": title,
			"code": code,
			"org_unit_id": org_unit_id,
			"job_grade": job_grade,
			"reports_to_position_id": reports_to_position_id,
			"incumbent_employee_id": None,
			"fte_count": fte_count,
			"is_critical": is_critical,
			"location": location,
			"description": description,
			"status": "open",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.positions[record["id"]] = record
		self._emit(t, "position_created", "position", record["id"], record)
		return deepcopy(record)

	async def list_positions(
		self,
		tenant_id: str,
		org_unit_id: str | None = None,
		status: str | None = None,
		is_critical: bool | None = None,
	) -> list[dict[str, Any]]:
		"""List positions."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.positions.values() if r["tenant_id"] == t]
		if org_unit_id:
			items = [r for r in items if r["org_unit_id"] == org_unit_id]
		if status:
			items = [r for r in items if r["status"] == status]
		if is_critical is not None:
			items = [r for r in items if r["is_critical"] == is_critical]
		return items

	async def get_position(self, tenant_id: str, position_id: str) -> dict[str, Any]:
		"""Get a position by ID."""
		t = self._tenant(tenant_id)
		record = self.positions.get(position_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"position {position_id} not found")
		return deepcopy(record)

	async def update_position(self, tenant_id: str, position_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a position."""
		t = self._tenant(tenant_id)
		record = self.positions.get(position_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"position {position_id} not found")
		allowed = {"title", "job_grade", "reports_to_position_id", "fte_count", "is_critical", "location", "description", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		record["updated_at"] = self._now()
		self._emit(t, "position_updated", "position", record["id"], record)
		return deepcopy(record)

	async def assign_employee_to_position(
		self,
		tenant_id: str,
		position_id: str,
		employee_id: str,
	) -> dict[str, Any]:
		"""Assign an employee to a position."""
		t = self._tenant(tenant_id)
		record = self.positions.get(position_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"position {position_id} not found")
		if record["status"] == "abolished":
			raise PermissionError("cannot_assign_to_abolished_position")
		record["incumbent_employee_id"] = employee_id
		record["status"] = "filled"
		record["updated_at"] = self._now()
		self._emit(t, "position_filled", "position", record["id"], {"position_id": position_id, "employee_id": employee_id})
		return deepcopy(record)

	async def vacate_position(self, tenant_id: str, position_id: str) -> dict[str, Any]:
		"""Remove incumbent from a position."""
		t = self._tenant(tenant_id)
		record = self.positions.get(position_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"position {position_id} not found")
		record["incumbent_employee_id"] = None
		record["status"] = "open"
		record["updated_at"] = self._now()
		self._emit(t, "position_vacated", "position", record["id"], {"position_id": position_id})
		return deepcopy(record)

	async def delete_position(self, tenant_id: str, position_id: str) -> bool:
		"""Delete (abolish) an open position."""
		t = self._tenant(tenant_id)
		record = self.positions.get(position_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"position {position_id} not found")
		if record["status"] == "filled":
			raise PermissionError("cannot_delete_filled_position")
		del self.positions[position_id]
		self._emit(t, "position_deleted", "position", position_id, {"id": position_id})
		return True

	# ── Reporting Lines ───────────────────────────────────────────────────────

	async def create_reporting_line(
		self,
		tenant_id: str,
		employee_id: str,
		manager_employee_id: str,
		effective_date: str,
		line_type: str = "direct",
		end_date: str | None = None,
	) -> dict[str, Any]:
		"""Define a reporting relationship."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		guard_non_empty_string(manager_employee_id, "manager_employee_id")
		if employee_id == manager_employee_id:
			raise ValueError("employee_cannot_report_to_themselves")
		if line_type not in REPORTING_LINE_TYPES:
			raise ValueError(f"line_type must be one of {REPORTING_LINE_TYPES}")
		record: dict[str, Any] = {
			"id": self._uid("rl"),
			"tenant_id": t,
			"employee_id": employee_id,
			"manager_employee_id": manager_employee_id,
			"line_type": line_type,
			"effective_date": effective_date,
			"end_date": end_date,
			"status": "active",
			"created_at": self._now(),
		}
		self.reporting_lines[record["id"]] = record
		self._emit(t, "reporting_line_created", "reporting_line", record["id"], record)
		return deepcopy(record)

	async def list_reporting_lines(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		manager_employee_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List reporting lines."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.reporting_lines.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if manager_employee_id:
			items = [r for r in items if r["manager_employee_id"] == manager_employee_id]
		return items

	async def get_reporting_line(self, tenant_id: str, line_id: str) -> dict[str, Any]:
		"""Get a reporting line by ID."""
		t = self._tenant(tenant_id)
		record = self.reporting_lines.get(line_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"reporting_line {line_id} not found")
		return deepcopy(record)

	async def terminate_reporting_line(self, tenant_id: str, line_id: str, end_date: str) -> dict[str, Any]:
		"""Terminate a reporting line."""
		t = self._tenant(tenant_id)
		record = self.reporting_lines.get(line_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"reporting_line {line_id} not found")
		record["end_date"] = end_date
		record["status"] = "terminated"
		self._emit(t, "reporting_line_terminated", "reporting_line", record["id"], record)
		return deepcopy(record)

	async def get_direct_reports(self, tenant_id: str, manager_employee_id: str) -> list[dict[str, Any]]:
		"""Get all direct reports for a manager."""
		t = self._tenant(tenant_id)
		return [
			deepcopy(r) for r in self.reporting_lines.values()
			if r["tenant_id"] == t and r["manager_employee_id"] == manager_employee_id
			and r["status"] == "active" and r["line_type"] == "direct"
		]

	async def compute_span_of_control(self, tenant_id: str, manager_employee_id: str) -> dict[str, Any]:
		"""Compute span-of-control metrics for a manager."""
		t = self._tenant(tenant_id)
		direct_reports = await self.get_direct_reports(t, manager_employee_id)
		total_reports = [
			r for r in self.reporting_lines.values()
			if r["tenant_id"] == t and r["manager_employee_id"] == manager_employee_id and r["status"] == "active"
		]
		result = {
			"manager_employee_id": manager_employee_id,
			"direct_report_count": len(direct_reports),
			"total_report_count": len(total_reports),
			"span_rating": "wide" if len(direct_reports) > 8 else ("narrow" if len(direct_reports) < 4 else "optimal"),
			"computed_at": self._now(),
		}
		self.span_of_control[f"{t}:{manager_employee_id}"] = result
		return result

	# ── Headcount Planning ────────────────────────────────────────────────────

	async def create_headcount_plan(
		self,
		tenant_id: str,
		org_unit_id: str,
		plan_year: int,
		planned_headcount: int,
		approved_by: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Create a headcount plan for an org unit and year."""
		t = self._tenant(tenant_id)
		unit = self.org_units.get(org_unit_id)
		if not unit or unit["tenant_id"] != t:
			raise KeyError(f"org_unit {org_unit_id} not found")
		current_hc = self._compute_headcount(t, org_unit_id)
		record: dict[str, Any] = {
			"id": self._uid("hcp"),
			"tenant_id": t,
			"org_unit_id": org_unit_id,
			"plan_year": plan_year,
			"current_headcount": current_hc,
			"planned_headcount": planned_headcount,
			"variance": planned_headcount - current_hc,
			"approved_by": approved_by,
			"notes": notes,
			"status": "draft",
			"created_at": self._now(),
		}
		self.headcount_plans[record["id"]] = record
		self._emit(t, "headcount_plan_created", "headcount_plan", record["id"], record)
		return deepcopy(record)

	async def list_headcount_plans(self, tenant_id: str, org_unit_id: str | None = None) -> list[dict[str, Any]]:
		"""List headcount plans."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.headcount_plans.values() if r["tenant_id"] == t]
		if org_unit_id:
			items = [r for r in items if r["org_unit_id"] == org_unit_id]
		return items

	async def approve_headcount_plan(self, tenant_id: str, plan_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a headcount plan."""
		t = self._tenant(tenant_id)
		record = self.headcount_plans.get(plan_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"headcount_plan {plan_id} not found")
		record["status"] = "approved"
		record["approved_by"] = approved_by
		self._emit(t, "headcount_plan_approved", "headcount_plan", record["id"], record)
		return deepcopy(record)

	# ── Restructuring ─────────────────────────────────────────────────────────

	async def create_restructuring(
		self,
		tenant_id: str,
		name: str,
		description: str,
		effective_date: str,
		initiated_by: str,
		units_affected: list[str] | None = None,
		positions_affected: list[str] | None = None,
	) -> dict[str, Any]:
		"""Initiate an organisational restructuring."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		record: dict[str, Any] = {
			"id": self._uid("rst"),
			"tenant_id": t,
			"name": name,
			"description": description,
			"effective_date": effective_date,
			"initiated_by": initiated_by,
			"units_affected": units_affected or [],
			"positions_affected": positions_affected or [],
			"status": "draft",
			"approved_by": None,
			"notes": None,
			"created_at": self._now(),
		}
		self.restructurings[record["id"]] = record
		self._emit(t, "restructuring_created", "restructuring", record["id"], record)
		return deepcopy(record)

	async def list_restructurings(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		"""List restructuring programmes."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.restructurings.values() if r["tenant_id"] == t]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def get_restructuring(self, tenant_id: str, restructuring_id: str) -> dict[str, Any]:
		"""Get a restructuring by ID."""
		t = self._tenant(tenant_id)
		record = self.restructurings.get(restructuring_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"restructuring {restructuring_id} not found")
		return deepcopy(record)

	async def update_restructuring(self, tenant_id: str, restructuring_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update restructuring status or notes."""
		t = self._tenant(tenant_id)
		record = self.restructurings.get(restructuring_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"restructuring {restructuring_id} not found")
		allowed = {"status", "approved_by", "notes", "units_affected", "positions_affected"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		self._emit(t, "restructuring_updated", "restructuring", record["id"], record)
		return deepcopy(record)

	async def delete_restructuring(self, tenant_id: str, restructuring_id: str) -> bool:
		"""Delete a draft restructuring."""
		t = self._tenant(tenant_id)
		record = self.restructurings.get(restructuring_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"restructuring {restructuring_id} not found")
		if record["status"] != "draft":
			raise PermissionError("only_draft_restructurings_can_be_deleted")
		del self.restructurings[restructuring_id]
		self._emit(t, "restructuring_deleted", "restructuring", restructuring_id, {"id": restructuring_id})
		return True

	# ── Analytics & Dashboard ─────────────────────────────────────────────────

	async def org_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Organisation-wide analytics."""
		t = self._tenant(tenant_id)
		units = [r for r in self.org_units.values() if r["tenant_id"] == t and r["status"] == "active"]
		positions = [r for r in self.positions.values() if r["tenant_id"] == t]
		critical = [p for p in positions if p["is_critical"]]
		open_critical = [p for p in critical if p["status"] == "open"]
		return {
			"tenant_id": t,
			"total_org_units": len(units),
			"total_positions": len(positions),
			"filled_positions": sum(1 for p in positions if p["status"] == "filled"),
			"open_positions": sum(1 for p in positions if p["status"] == "open"),
			"critical_positions": len(critical),
			"open_critical_positions": len(open_critical),
			"total_reporting_lines": sum(1 for r in self.reporting_lines.values() if r["tenant_id"] == t),
			"active_restructurings": sum(1 for r in self.restructurings.values() if r["tenant_id"] == t and r["status"] in {"approved", "in_progress"}),
			"generated_at": self._now(),
		}

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Dashboard summary for org management."""
		t = self._tenant(tenant_id)
		analytics, units, plans = await asyncio.gather(
			self.org_analytics(t),
			self.list_org_units(t),
			self.list_headcount_plans(t),
			return_exceptions=True,
		)
		return {
			"analytics": analytics if not isinstance(analytics, Exception) else {},
			"unit_count": len(units) if not isinstance(units, Exception) else 0,
			"headcount_plan_count": len(plans) if not isinstance(plans, Exception) else 0,
			"generated_at": self._now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

