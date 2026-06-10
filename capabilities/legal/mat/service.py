"""Matter Management — async service layer."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

MATTER_TYPES = {"litigation", "advisory", "transactional", "regulatory", "compliance", "employment", "ip", "real_estate"}
PRIORITY_LEVELS = {"low", "normal", "high", "urgent"}
MATTER_STATUSES = {"open", "active", "on_hold", "closed", "archived"}
TASK_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
DEADLINE_TYPES = {"court", "filing", "statute_of_limitations", "contractual", "regulatory", "internal"}


class MatterManagementService:
	"""In-memory async service for legal matter lifecycle management."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.matters: dict[str, dict[str, Any]] = {}
		self.tasks: dict[str, dict[str, Any]] = {}
		self.deadlines: dict[str, dict[str, Any]] = {}
		self.docket_entries: dict[str, dict[str, Any]] = {}
		self.team_assignments: dict[str, dict[str, Any]] = {}
		self.notes: dict[str, dict[str, Any]] = {}
		self.time_budgets: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self, tenant_id: str | None = None) -> str:
		val = tenant_id or self.tenant_id
		guard_tenant_id(val)
		return val

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt-"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"details": details or {},
			"created_at": self._now(),
		})

	# ── Health ──────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "leg_mat",
			"status": "healthy",
			"matter_count": len(self.matters),
			"open_tasks": sum(1 for t in self.tasks.values() if t["status"] in {"pending", "in_progress"}),
			"overdue_deadlines": sum(
				1 for d in self.deadlines.values()
				if d["status"] == "pending" and d["deadline_date"] < date.today().isoformat()
			),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability descriptor."""
		return {
			"capability_id": "leg_mat",
			"name": "Matter Management",
			"domain": "legal",
			"version": "1.0.0",
			"matter_types": sorted(MATTER_TYPES),
			"priority_levels": sorted(PRIORITY_LEVELS),
			"statuses": sorted(MATTER_STATUSES),
		}

	# ── Matters ─────────────────────────────────────────────────────────────

	async def create_matter(
		self,
		tenant_id: str,
		title: str,
		matter_type: str,
		client_id: str,
		lead_attorney_id: str,
		practice_area: str,
		jurisdiction: str,
		description: str = "",
		priority: str = "normal",
		budget: float | None = None,
		opened_date: str | None = None,
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Open a new legal matter."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		guard_non_empty_string(client_id, "client_id")
		if matter_type not in MATTER_TYPES:
			raise ValueError(f"matter_type must be one of {MATTER_TYPES}")
		if priority not in PRIORITY_LEVELS:
			raise ValueError(f"priority must be one of {PRIORITY_LEVELS}")
		record: dict[str, Any] = {
			"id": self._id("mat-"),
			"tenant_id": tenant,
			"title": title,
			"matter_type": matter_type,
			"client_id": client_id,
			"lead_attorney_id": lead_attorney_id,
			"practice_area": practice_area,
			"jurisdiction": jurisdiction,
			"description": description,
			"priority": priority,
			"budget": budget,
			"opened_date": opened_date or date.today().isoformat(),
			"closed_date": None,
			"tags": list(tags or []),
			"team_ids": [lead_attorney_id],
			"task_count": 0,
			"deadline_count": 0,
			"status": "open",
			"metadata": dict(metadata or {}),
			"created_at": self._now(),
			"updated_at": None,
		}
		self.matters[record["id"]] = record
		self._emit(tenant, "matter_created", record["id"], {"title": title, "matter_type": matter_type})
		_log.info("matter created tenant=%s id=%s type=%s", tenant, record["id"], matter_type)
		return deepcopy(record)

	async def get_matter(self, tenant_id: str, matter_id: str) -> dict[str, Any]:
		"""Retrieve a matter by ID."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		return deepcopy(matter)

	async def list_matters(
		self,
		tenant_id: str,
		status: str | None = None,
		matter_type: str | None = None,
		client_id: str | None = None,
		lead_attorney_id: str | None = None,
		priority: str | None = None,
	) -> list[dict[str, Any]]:
		"""List matters with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(m) for m in self.matters.values() if m["tenant_id"] == tenant]
		if status:
			items = [m for m in items if m["status"] == status]
		if matter_type:
			items = [m for m in items if m["matter_type"] == matter_type]
		if client_id:
			items = [m for m in items if m["client_id"] == client_id]
		if lead_attorney_id:
			items = [m for m in items if m["lead_attorney_id"] == lead_attorney_id]
		if priority:
			items = [m for m in items if m["priority"] == priority]
		return items

	async def update_matter(
		self,
		tenant_id: str,
		matter_id: str,
		**updates: Any,
	) -> dict[str, Any]:
		"""Patch allowed fields on a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		allowed = {"title", "description", "priority", "budget", "tags", "metadata", "lead_attorney_id"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				matter[k] = v
		matter["updated_at"] = self._now()
		self._emit(tenant, "matter_updated", matter_id, updates)
		return deepcopy(matter)

	async def close_matter(self, tenant_id: str, matter_id: str, closed_by: str, notes: str = "") -> dict[str, Any]:
		"""Close a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		if matter["status"] == "closed":
			raise ValueError("matter already closed")
		open_tasks = [t for t in self.tasks.values() if t["matter_id"] == matter_id and t["status"] in {"pending", "in_progress"}]
		if open_tasks:
			raise ValueError(f"cannot close matter with {len(open_tasks)} open tasks")
		matter["status"] = "closed"
		matter["closed_date"] = date.today().isoformat()
		matter["closed_by"] = closed_by
		matter["closing_notes"] = notes
		matter["updated_at"] = self._now()
		self._emit(tenant, "matter_closed", matter_id, {"closed_by": closed_by})
		return deepcopy(matter)

	async def reopen_matter(self, tenant_id: str, matter_id: str, reason: str) -> dict[str, Any]:
		"""Reopen a closed matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		matter["status"] = "active"
		matter["closed_date"] = None
		matter["reopen_reason"] = reason
		matter["updated_at"] = self._now()
		self._emit(tenant, "matter_reopened", matter_id, {"reason": reason})
		return deepcopy(matter)

	async def delete_matter(self, tenant_id: str, matter_id: str) -> dict[str, Any]:
		"""Archive (soft-delete) a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		matter["status"] = "archived"
		matter["updated_at"] = self._now()
		self._emit(tenant, "matter_archived", matter_id)
		return deepcopy(matter)

	# ── Team Assignment ──────────────────────────────────────────────────────

	async def assign_team_member(
		self,
		tenant_id: str,
		matter_id: str,
		attorney_id: str,
		role: str = "associate",
	) -> dict[str, Any]:
		"""Add an attorney to the matter team."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		if attorney_id in matter["team_ids"]:
			raise ValueError(f"attorney {attorney_id} already on matter team")
		assignment: dict[str, Any] = {
			"id": self._id("assign-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"attorney_id": attorney_id,
			"role": role,
			"status": "active",
			"assigned_at": self._now(),
		}
		self.team_assignments[assignment["id"]] = assignment
		matter["team_ids"].append(attorney_id)
		self._emit(tenant, "team_member_assigned", matter_id, {"attorney_id": attorney_id, "role": role})
		return deepcopy(assignment)

	async def remove_team_member(self, tenant_id: str, matter_id: str, attorney_id: str) -> dict[str, Any]:
		"""Remove an attorney from the matter team."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		if matter["lead_attorney_id"] == attorney_id:
			raise ValueError("cannot remove lead attorney; reassign lead first")
		matter["team_ids"] = [aid for aid in matter["team_ids"] if aid != attorney_id]
		for a in self.team_assignments.values():
			if a["matter_id"] == matter_id and a["attorney_id"] == attorney_id:
				a["status"] = "removed"
				a["removed_at"] = self._now()
		self._emit(tenant, "team_member_removed", matter_id, {"attorney_id": attorney_id})
		return deepcopy(matter)

	async def list_team_members(self, tenant_id: str, matter_id: str) -> list[dict[str, Any]]:
		"""Return current team assignments for a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		return [
			deepcopy(a) for a in self.team_assignments.values()
			if a["matter_id"] == matter_id and a["status"] == "active"
		]

	# ── Tasks ────────────────────────────────────────────────────────────────

	async def create_task(
		self,
		tenant_id: str,
		matter_id: str,
		title: str,
		assigned_to_id: str,
		due_date: str,
		description: str = "",
		priority: str = "normal",
		task_type: str = "general",
	) -> dict[str, Any]:
		"""Create a task on a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		guard_non_empty_string(title, "title")
		task: dict[str, Any] = {
			"id": self._id("task-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"title": title,
			"description": description,
			"assigned_to_id": assigned_to_id,
			"due_date": due_date,
			"priority": priority,
			"task_type": task_type,
			"status": "pending",
			"completed_at": None,
			"created_at": self._now(),
		}
		self.tasks[task["id"]] = task
		matter["task_count"] = matter.get("task_count", 0) + 1
		self._emit(tenant, "task_created", task["id"], {"matter_id": matter_id, "title": title})
		return deepcopy(task)

	async def get_task(self, tenant_id: str, task_id: str) -> dict[str, Any]:
		"""Retrieve a task."""
		tenant = self._tenant(tenant_id)
		task = self.tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"task {task_id} not found")
		return deepcopy(task)

	async def list_tasks(
		self,
		tenant_id: str,
		matter_id: str | None = None,
		assigned_to_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List tasks with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(t) for t in self.tasks.values() if t["tenant_id"] == tenant]
		if matter_id:
			items = [t for t in items if t["matter_id"] == matter_id]
		if assigned_to_id:
			items = [t for t in items if t["assigned_to_id"] == assigned_to_id]
		if status:
			items = [t for t in items if t["status"] == status]
		return items

	async def update_task(self, tenant_id: str, task_id: str, **updates: Any) -> dict[str, Any]:
		"""Update task fields."""
		tenant = self._tenant(tenant_id)
		task = self.tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"task {task_id} not found")
		allowed = {"title", "description", "assigned_to_id", "due_date", "priority"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				task[k] = v
		self._emit(tenant, "task_updated", task_id, updates)
		return deepcopy(task)

	async def complete_task(self, tenant_id: str, task_id: str, completed_by: str) -> dict[str, Any]:
		"""Mark a task as completed."""
		tenant = self._tenant(tenant_id)
		task = self.tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"task {task_id} not found")
		task["status"] = "completed"
		task["completed_at"] = self._now()
		task["completed_by"] = completed_by
		self._emit(tenant, "task_completed", task_id, {"completed_by": completed_by})
		return deepcopy(task)

	async def delete_task(self, tenant_id: str, task_id: str) -> dict[str, Any]:
		"""Cancel a task."""
		tenant = self._tenant(tenant_id)
		task = self.tasks.get(task_id)
		if not task or task["tenant_id"] != tenant:
			raise KeyError(f"task {task_id} not found")
		task["status"] = "cancelled"
		self._emit(tenant, "task_cancelled", task_id)
		return deepcopy(task)

	# ── Deadlines ────────────────────────────────────────────────────────────

	async def create_deadline(
		self,
		tenant_id: str,
		matter_id: str,
		title: str,
		deadline_date: str,
		deadline_type: str,
		description: str = "",
		reminder_days: list[int] | None = None,
	) -> dict[str, Any]:
		"""Create a deadline on a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		if deadline_type not in DEADLINE_TYPES:
			raise ValueError(f"deadline_type must be one of {DEADLINE_TYPES}")
		guard_non_empty_string(title, "title")
		deadline: dict[str, Any] = {
			"id": self._id("dl-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"title": title,
			"deadline_date": deadline_date,
			"deadline_type": deadline_type,
			"description": description,
			"reminder_days": reminder_days or [7, 1],
			"status": "pending",
			"created_at": self._now(),
		}
		self.deadlines[deadline["id"]] = deadline
		matter["deadline_count"] = matter.get("deadline_count", 0) + 1
		self._emit(tenant, "deadline_created", deadline["id"], {"matter_id": matter_id, "date": deadline_date})
		return deepcopy(deadline)

	async def get_deadline(self, tenant_id: str, deadline_id: str) -> dict[str, Any]:
		"""Retrieve a deadline."""
		tenant = self._tenant(tenant_id)
		dl = self.deadlines.get(deadline_id)
		if not dl or dl["tenant_id"] != tenant:
			raise KeyError(f"deadline {deadline_id} not found")
		return deepcopy(dl)

	async def list_deadlines(
		self,
		tenant_id: str,
		matter_id: str | None = None,
		deadline_type: str | None = None,
		overdue_only: bool = False,
	) -> list[dict[str, Any]]:
		"""List deadlines with optional filters."""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		items = [deepcopy(d) for d in self.deadlines.values() if d["tenant_id"] == tenant]
		if matter_id:
			items = [d for d in items if d["matter_id"] == matter_id]
		if deadline_type:
			items = [d for d in items if d["deadline_type"] == deadline_type]
		if overdue_only:
			items = [d for d in items if d["status"] == "pending" and d["deadline_date"] < today]
		return sorted(items, key=lambda d: d["deadline_date"])

	async def update_deadline(self, tenant_id: str, deadline_id: str, **updates: Any) -> dict[str, Any]:
		"""Update deadline fields."""
		tenant = self._tenant(tenant_id)
		dl = self.deadlines.get(deadline_id)
		if not dl or dl["tenant_id"] != tenant:
			raise KeyError(f"deadline {deadline_id} not found")
		allowed = {"title", "deadline_date", "description", "reminder_days"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				dl[k] = v
		self._emit(tenant, "deadline_updated", deadline_id, updates)
		return deepcopy(dl)

	async def acknowledge_deadline(self, tenant_id: str, deadline_id: str, acknowledged_by: str) -> dict[str, Any]:
		"""Mark a deadline as met."""
		tenant = self._tenant(tenant_id)
		dl = self.deadlines.get(deadline_id)
		if not dl or dl["tenant_id"] != tenant:
			raise KeyError(f"deadline {deadline_id} not found")
		dl["status"] = "met"
		dl["acknowledged_by"] = acknowledged_by
		dl["acknowledged_at"] = self._now()
		self._emit(tenant, "deadline_met", deadline_id, {"acknowledged_by": acknowledged_by})
		return deepcopy(dl)

	async def delete_deadline(self, tenant_id: str, deadline_id: str) -> dict[str, Any]:
		"""Remove a deadline."""
		tenant = self._tenant(tenant_id)
		dl = self.deadlines.get(deadline_id)
		if not dl or dl["tenant_id"] != tenant:
			raise KeyError(f"deadline {deadline_id} not found")
		dl["status"] = "removed"
		self._emit(tenant, "deadline_removed", deadline_id)
		return deepcopy(dl)

	# ── Court Dockets ────────────────────────────────────────────────────────

	async def create_docket_entry(
		self,
		tenant_id: str,
		matter_id: str,
		court: str,
		case_number: str,
		event_date: str,
		event_type: str,
		description: str,
		judge: str | None = None,
		courtroom: str | None = None,
	) -> dict[str, Any]:
		"""Add a court docket entry to a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		entry: dict[str, Any] = {
			"id": self._id("dkt-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"court": court,
			"case_number": case_number,
			"event_date": event_date,
			"event_type": event_type,
			"description": description,
			"judge": judge,
			"courtroom": courtroom,
			"status": "scheduled",
			"created_at": self._now(),
		}
		self.docket_entries[entry["id"]] = entry
		self._emit(tenant, "docket_entry_created", entry["id"], {"matter_id": matter_id, "court": court})
		return deepcopy(entry)

	async def get_docket_entry(self, tenant_id: str, entry_id: str) -> dict[str, Any]:
		"""Retrieve a docket entry."""
		tenant = self._tenant(tenant_id)
		entry = self.docket_entries.get(entry_id)
		if not entry or entry["tenant_id"] != tenant:
			raise KeyError(f"docket entry {entry_id} not found")
		return deepcopy(entry)

	async def list_docket_entries(
		self,
		tenant_id: str,
		matter_id: str | None = None,
		case_number: str | None = None,
	) -> list[dict[str, Any]]:
		"""List docket entries."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.docket_entries.values() if e["tenant_id"] == tenant]
		if matter_id:
			items = [e for e in items if e["matter_id"] == matter_id]
		if case_number:
			items = [e for e in items if e["case_number"] == case_number]
		return sorted(items, key=lambda e: e["event_date"])

	async def update_docket_entry(self, tenant_id: str, entry_id: str, **updates: Any) -> dict[str, Any]:
		"""Update a docket entry."""
		tenant = self._tenant(tenant_id)
		entry = self.docket_entries.get(entry_id)
		if not entry or entry["tenant_id"] != tenant:
			raise KeyError(f"docket entry {entry_id} not found")
		allowed = {"event_date", "description", "judge", "courtroom", "status"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				entry[k] = v
		self._emit(tenant, "docket_entry_updated", entry_id, updates)
		return deepcopy(entry)

	async def delete_docket_entry(self, tenant_id: str, entry_id: str) -> dict[str, Any]:
		"""Remove a docket entry."""
		tenant = self._tenant(tenant_id)
		entry = self.docket_entries.get(entry_id)
		if not entry or entry["tenant_id"] != tenant:
			raise KeyError(f"docket entry {entry_id} not found")
		entry["status"] = "cancelled"
		self._emit(tenant, "docket_entry_cancelled", entry_id)
		return deepcopy(entry)

	# ── Notes ────────────────────────────────────────────────────────────────

	async def add_note(
		self,
		tenant_id: str,
		matter_id: str,
		author_id: str,
		content: str,
		note_type: str = "general",
		is_privileged: bool = False,
	) -> dict[str, Any]:
		"""Add a note to a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		guard_non_empty_string(content, "content")
		note: dict[str, Any] = {
			"id": self._id("note-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"author_id": author_id,
			"content": content,
			"note_type": note_type,
			"is_privileged": is_privileged,
			"status": "active",
			"created_at": self._now(),
		}
		self.notes[note["id"]] = note
		self._emit(tenant, "note_added", note["id"], {"matter_id": matter_id})
		return deepcopy(note)

	async def list_notes(self, tenant_id: str, matter_id: str) -> list[dict[str, Any]]:
		"""List notes for a matter."""
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(n) for n in self.notes.values()
			if n["tenant_id"] == tenant and n["matter_id"] == matter_id and n["status"] == "active"
		]

	# ── Time Budgets ─────────────────────────────────────────────────────────

	async def set_time_budget(
		self,
		tenant_id: str,
		matter_id: str,
		total_hours: float,
		allocated_by_id: str,
	) -> dict[str, Any]:
		"""Set a time budget on a matter."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		if total_hours <= 0:
			raise ValueError("total_hours must be positive")
		budget: dict[str, Any] = {
			"id": self._id("bgt-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"total_hours": total_hours,
			"used_hours": 0.0,
			"allocated_by_id": allocated_by_id,
			"status": "active",
			"created_at": self._now(),
		}
		self.time_budgets[matter_id] = budget
		self._emit(tenant, "time_budget_set", matter_id, {"total_hours": total_hours})
		return deepcopy(budget)

	async def get_time_budget(self, tenant_id: str, matter_id: str) -> dict[str, Any] | None:
		"""Retrieve the time budget for a matter."""
		tenant = self._tenant(tenant_id)
		budget = self.time_budgets.get(matter_id)
		if budget and budget["tenant_id"] == tenant:
			return deepcopy(budget)
		return None

	# ── Analytics ────────────────────────────────────────────────────────────

	async def matter_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Return aggregate matter metrics."""
		tenant = self._tenant(tenant_id)
		matters = [m for m in self.matters.values() if m["tenant_id"] == tenant]
		tasks = [t for t in self.tasks.values() if t["tenant_id"] == tenant]
		deadlines = [d for d in self.deadlines.values() if d["tenant_id"] == tenant]
		today = date.today().isoformat()
		by_type: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for m in matters:
			by_type[m["matter_type"]] = by_type.get(m["matter_type"], 0) + 1
			by_status[m["status"]] = by_status.get(m["status"], 0) + 1
		return {
			"tenant_id": tenant,
			"total_matters": len(matters),
			"by_type": by_type,
			"by_status": by_status,
			"open_tasks": sum(1 for t in tasks if t["status"] in {"pending", "in_progress"}),
			"overdue_tasks": sum(1 for t in tasks if t["status"] == "pending" and t["due_date"] < today),
			"pending_deadlines": sum(1 for d in deadlines if d["status"] == "pending"),
			"overdue_deadlines": sum(1 for d in deadlines if d["status"] == "pending" and d["deadline_date"] < today),
			"generated_at": self._now(),
		}

	async def upcoming_deadlines(self, tenant_id: str, days_ahead: int = 14) -> list[dict[str, Any]]:
		"""Return deadlines due within the next N days."""
		tenant = self._tenant(tenant_id)
		today = date.today()
		cutoff = today.replace(day=today.day + days_ahead).isoformat() if today.day + days_ahead <= 28 else (
			date(today.year, today.month + 1 if today.month < 12 else 1, (today.day + days_ahead) % 28 or 1).isoformat()
		)
		# simpler: use string arithmetic — ISO dates sort lexically
		today_s = today.isoformat()
		items = [
			deepcopy(d) for d in self.deadlines.values()
			if d["tenant_id"] == tenant and d["status"] == "pending"
			and today_s <= d["deadline_date"]
		]
		return sorted(items, key=lambda d: d["deadline_date"])[:50]

	async def attorney_workload(self, tenant_id: str, attorney_id: str) -> dict[str, Any]:
		"""Return workload summary for an attorney."""
		tenant = self._tenant(tenant_id)
		matters = [m for m in self.matters.values() if m["tenant_id"] == tenant and attorney_id in m["team_ids"]]
		tasks = [t for t in self.tasks.values() if t["tenant_id"] == tenant and t["assigned_to_id"] == attorney_id]
		return {
			"attorney_id": attorney_id,
			"active_matters": sum(1 for m in matters if m["status"] in {"open", "active"}),
			"total_matters": len(matters),
			"pending_tasks": sum(1 for t in tasks if t["status"] in {"pending", "in_progress"}),
			"overdue_tasks": sum(1 for t in tasks if t["status"] == "pending" and t["due_date"] < date.today().isoformat()),
			"generated_at": self._now(),
		}

	async def search_matters(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
		"""Full-text search across matter titles and descriptions."""
		tenant = self._tenant(tenant_id)
		q = query.lower()
		return [
			deepcopy(m) for m in self.matters.values()
			if m["tenant_id"] == tenant and (
				q in m["title"].lower() or q in m.get("description", "").lower()
			)
		]

	async def get_audit_events(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
		"""Return audit log for the tenant."""
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		return events[-limit:]

	async def export_matter_summary(self, tenant_id: str, matter_id: str) -> dict[str, Any]:
		"""Export a full matter summary including tasks and deadlines."""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		tasks, deadlines, dockets, notes = await asyncio.gather(
			self.list_tasks(tenant_id, matter_id=matter_id),
			self.list_deadlines(tenant_id, matter_id=matter_id),
			self.list_docket_entries(tenant_id, matter_id=matter_id),
			self.list_notes(tenant_id, matter_id),
			return_exceptions=True,
		)
		return {
			"matter": deepcopy(matter),
			"tasks": tasks if isinstance(tasks, list) else [],
			"deadlines": deadlines if isinstance(deadlines, list) else [],
			"docket_entries": dockets if isinstance(dockets, list) else [],
			"notes": notes if isinstance(notes, list) else [],
			"exported_at": self._now(),
		}
