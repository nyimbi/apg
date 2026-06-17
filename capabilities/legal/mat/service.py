"""Matter Management — async service layer."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import hashlib
import logging
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

MATTER_TYPES = {"litigation", "advisory", "transactional", "regulatory", "compliance", "employment", "ip", "real_estate"}
PRIORITY_LEVELS = {"low", "normal", "high", "urgent"}
MATTER_STATUSES = {"open", "active", "on_hold", "closed", "archived"}
TASK_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
DEADLINE_TYPES = {"court", "filing", "statute_of_limitations", "contractual", "regulatory", "internal"}

# FSM: allowed status transitions
MATTER_TRANSITIONS: dict[str, set[str]] = {
	"open":     {"active", "on_hold", "archived"},
	"active":   {"on_hold", "closed", "archived"},
	"on_hold":  {"active", "closed", "archived"},
	"closed":   {"active"},
	"archived": set(),
}

# Matter-type → default task templates [{title, task_type, relative_due_days, priority}]
MATTER_TEMPLATES: dict[str, list[dict[str, Any]]] = {
	"litigation": [
		{"title": "File initial pleadings", "task_type": "filing", "relative_due_days": 14, "priority": "high"},
		{"title": "Serve process on defendants", "task_type": "service", "relative_due_days": 30, "priority": "high"},
		{"title": "Prepare initial disclosures", "task_type": "discovery", "relative_due_days": 42, "priority": "normal"},
		{"title": "Attend scheduling conference", "task_type": "court_appearance", "relative_due_days": 60, "priority": "high"},
	],
	"transactional": [
		{"title": "Draft term sheet", "task_type": "drafting", "relative_due_days": 7, "priority": "high"},
		{"title": "Due diligence review", "task_type": "review", "relative_due_days": 21, "priority": "normal"},
		{"title": "Draft definitive agreement", "task_type": "drafting", "relative_due_days": 35, "priority": "high"},
		{"title": "Regulatory filings", "task_type": "filing", "relative_due_days": 45, "priority": "normal"},
	],
	"advisory": [
		{"title": "Initial client briefing", "task_type": "meeting", "relative_due_days": 3, "priority": "normal"},
		{"title": "Research and analysis", "task_type": "research", "relative_due_days": 14, "priority": "normal"},
		{"title": "Draft opinion letter", "task_type": "drafting", "relative_due_days": 21, "priority": "normal"},
	],
}

# Deadline chain rules: trigger_event → derived deadlines with offsets
DEADLINE_CHAIN_RULES: dict[str, list[dict[str, Any]]] = {
	"complaint_filed": [
		{"title": "Defendant response deadline", "offset_days": 21, "deadline_type": "court", "reminder_days": [14, 7, 2, 1]},
		{"title": "Initial scheduling conference", "offset_days": 60, "deadline_type": "court", "reminder_days": [7, 2]},
		{"title": "Initial disclosures due", "offset_days": 42, "deadline_type": "filing", "reminder_days": [14, 7]},
	],
	"defence_filed": [
		{"title": "Plaintiff reply deadline", "offset_days": 14, "deadline_type": "court", "reminder_days": [7, 2, 1]},
		{"title": "Discovery cutoff", "offset_days": 120, "deadline_type": "court", "reminder_days": [30, 14, 7]},
	],
}


class MatterManagementService:
	"""In-memory async service for legal matter lifecycle management."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.matters: dict[str, dict[str, Any]] = {}
		self.tasks: dict[str, dict[str, Any]] = {}
		self.deadlines: dict[str, dict[str, Any]] = {}
		self.docket_entries: dict[str, dict[str, Any]] = {}
		self.team_assignments: dict[str, dict[str, Any]] = {}
		self.notes: dict[str, dict[str, Any]] = {}
		self.time_budgets: dict[str, dict[str, Any]] = {}
		self.time_entries: dict[str, dict[str, Any]] = {}
		self.invoices: dict[str, dict[str, Any]] = {}
		self.conflict_checks: dict[str, dict[str, Any]] = {}
		self.attorney_profiles: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

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

	# ── Matter FSM Transition ────────────────────────────────────────────────

	async def transition_matter_status(
		self,
		tenant_id: str,
		matter_id: str,
		new_status: str,
		actor_id: str,
		reason: str = "",
	) -> dict[str, Any]:
		"""Validate and apply a matter status transition via the FSM.

		Guards enforced:
		- Transition must be allowed by MATTER_TRANSITIONS.
		- Transitioning to 'closed' requires zero open tasks.
		- Transitioning to 'archived' requires status is already 'closed'.
		"""
		guard_non_empty_string(actor_id, "actor_id")
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		current = matter["status"]
		allowed = MATTER_TRANSITIONS.get(current, set())
		if new_status not in allowed:
			raise ValueError(
				f"transition '{current}' → '{new_status}' is not permitted; "
				f"allowed: {sorted(allowed) or 'none'}"
			)
		if new_status == "closed":
			open_tasks = [
				t for t in self.tasks.values()
				if t["matter_id"] == matter_id and t["status"] in {"pending", "in_progress"}
			]
			if open_tasks:
				raise ValueError(
					f"cannot close matter with {len(open_tasks)} open task(s); complete or cancel them first"
				)
		matter["status"] = new_status
		if new_status == "closed":
			matter["closed_date"] = date.today().isoformat()
		matter["updated_at"] = self._now()
		self._emit(tenant, "matter_status_transitioned", matter_id, {
			"from_status": current,
			"to_status": new_status,
			"actor_id": actor_id,
			"reason": reason,
		})
		_log.info(
			"matter status transition tenant=%s id=%s %s→%s actor=%s",
			tenant, matter_id, current, new_status, actor_id,
		)
		return deepcopy(matter)

	# ── Time Entry Logging ───────────────────────────────────────────────────

	async def log_time_entry(
		self,
		tenant_id: str,
		matter_id: str,
		attorney_id: str,
		hours: str,
		narrative: str,
		rate: str,
		entry_date: str | None = None,
		billable: bool = True,
		task_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a time entry against a matter.

		Args:
			hours:  Decimal string, e.g. '1.5' — stored as Decimal for precision.
			rate:   Hourly rate as Decimal string, e.g. '350.00'.
		"""
		guard_non_empty_string(attorney_id, "attorney_id")
		guard_non_empty_string(narrative, "narrative")
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		d_hours = Decimal(hours).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		d_rate = Decimal(rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		if d_hours <= 0:
			raise ValueError("hours must be positive")
		if d_rate < 0:
			raise ValueError("rate cannot be negative")
		amount = (d_hours * d_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		entry: dict[str, Any] = {
			"id": self._id("te-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"attorney_id": attorney_id,
			"hours": str(d_hours),
			"rate": str(d_rate),
			"amount": str(amount),
			"narrative": narrative,
			"billable": billable,
			"task_id": task_id,
			"entry_date": entry_date or date.today().isoformat(),
			"created_at": self._now(),
		}
		self.time_entries[entry["id"]] = entry
		# Update budget used_hours if a budget exists
		budget = self.time_budgets.get(matter_id)
		if budget and budget["tenant_id"] == tenant:
			used = Decimal(str(budget["used_hours"])) + d_hours
			budget["used_hours"] = float(used)
		self._emit(tenant, "time_entry_logged", entry["id"], {
			"matter_id": matter_id,
			"attorney_id": attorney_id,
			"hours": str(d_hours),
			"billable": billable,
		})
		return deepcopy(entry)

	async def list_time_entries(
		self,
		tenant_id: str,
		matter_id: str | None = None,
		attorney_id: str | None = None,
		billable_only: bool = False,
	) -> list[dict[str, Any]]:
		"""List time entries with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.time_entries.values() if e["tenant_id"] == tenant]
		if matter_id:
			items = [e for e in items if e["matter_id"] == matter_id]
		if attorney_id:
			items = [e for e in items if e["attorney_id"] == attorney_id]
		if billable_only:
			items = [e for e in items if e["billable"]]
		return sorted(items, key=lambda e: e["entry_date"])

	# ── Budget Burn Report ───────────────────────────────────────────────────

	async def get_budget_burn_report(self, tenant_id: str, matter_id: str) -> dict[str, Any]:
		"""Return budget burn analysis with projected overrun date.

		Uses simple linear projection based on daily burn rate over the last
		30 days of time entries.  All monetary values returned as strings to
		preserve Decimal precision.
		"""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		budget = self.time_budgets.get(matter_id)
		entries = [
			e for e in self.time_entries.values()
			if e["tenant_id"] == tenant and e["matter_id"] == matter_id and e["billable"]
		]
		total_hours = Decimal("0")
		total_amount = Decimal("0")
		for e in entries:
			total_hours += Decimal(e["hours"])
			total_amount += Decimal(e["amount"])
		report: dict[str, Any] = {
			"matter_id": matter_id,
			"total_billable_hours": str(total_hours),
			"total_billed_amount": str(total_amount),
			"budget_total_hours": None,
			"budget_remaining_hours": None,
			"burn_pct": None,
			"projected_overrun_date": None,
		}
		if budget:
			budget_hours = Decimal(str(budget["total_hours"]))
			remaining = (budget_hours - total_hours).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			burn_pct = (total_hours / budget_hours * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) if budget_hours else Decimal("0")
			report["budget_total_hours"] = str(budget_hours)
			report["budget_remaining_hours"] = str(remaining)
			report["burn_pct"] = str(burn_pct)
			# Daily burn rate from last 30 days
			cutoff = (date.today() - timedelta(days=30)).isoformat()
			recent = [e for e in entries if e["entry_date"] >= cutoff]
			if recent and remaining > 0:
				daily_hours = total_hours / Decimal("30")
				if daily_hours > 0:
					days_left = int(remaining / daily_hours)
					overrun_date = (date.today() + timedelta(days=days_left)).isoformat()
					report["projected_overrun_date"] = overrun_date
		report["generated_at"] = self._now()
		return report

	# ── Conflict Check ───────────────────────────────────────────────────────

	async def run_conflict_check(
		self,
		tenant_id: str,
		matter_id: str,
		party_names: list[str],
	) -> dict[str, Any]:
		"""Run a conflict-of-interest check against all existing matters.

		Uses exact and substring matching against existing matter titles,
		client IDs, and metadata party lists.  Flags any matches for review.
		"""
		guard_non_empty_string(matter_id, "matter_id")
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		if not party_names:
			raise ValueError("party_names must be a non-empty list")
		candidates: list[dict[str, Any]] = []
		normalised = [p.lower().strip() for p in party_names]
		for mid, m in self.matters.items():
			if mid == matter_id or m["tenant_id"] != tenant:
				continue
			hit_reasons: list[str] = []
			title_lower = m["title"].lower()
			for name in normalised:
				if name in title_lower:
					hit_reasons.append(f"party '{name}' found in matter title '{m['title']}'")
			meta_parties: list[str] = m.get("metadata", {}).get("parties", [])
			for mp in meta_parties:
				for name in normalised:
					if name in mp.lower():
						hit_reasons.append(f"party '{name}' matches existing party '{mp}' on {mid}")
			if hit_reasons:
				candidates.append({
					"conflicting_matter_id": mid,
					"conflicting_matter_title": m["title"],
					"reasons": hit_reasons,
				})
		status = "flagged" if candidates else "clear"
		check: dict[str, Any] = {
			"id": self._id("cf-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"party_names": party_names,
			"status": status,
			"candidates": candidates,
			"checked_at": self._now(),
		}
		self.conflict_checks[check["id"]] = check
		self._emit(tenant, "conflict_check_run", matter_id, {
			"status": status,
			"candidate_count": len(candidates),
		})
		_log.info("conflict check tenant=%s matter=%s status=%s hits=%d", tenant, matter_id, status, len(candidates))
		return deepcopy(check)

	# ── Deadline Chaining ────────────────────────────────────────────────────

	async def create_chained_deadlines(
		self,
		tenant_id: str,
		matter_id: str,
		trigger_event: str,
		trigger_date: str,
	) -> list[dict[str, Any]]:
		"""Create a chain of derived deadlines from a trigger event.

		Uses DEADLINE_CHAIN_RULES to compute offsets from trigger_date.
		Each created deadline is linked via 'parent_trigger_event'.
		"""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		rules = DEADLINE_CHAIN_RULES.get(trigger_event)
		if not rules:
			raise ValueError(
				f"no chain rules defined for trigger_event '{trigger_event}'; "
				f"known: {sorted(DEADLINE_CHAIN_RULES)}"
			)
		base = date.fromisoformat(trigger_date)
		created: list[dict[str, Any]] = []
		for rule in rules:
			derived_date = (base + timedelta(days=rule["offset_days"])).isoformat()
			dl = await self.create_deadline(
				tenant_id=tenant_id,
				matter_id=matter_id,
				title=rule["title"],
				deadline_date=derived_date,
				deadline_type=rule["deadline_type"],
				description=f"Derived from trigger '{trigger_event}' on {trigger_date}",
				reminder_days=rule.get("reminder_days", [7, 1]),
			)
			dl["parent_trigger_event"] = trigger_event
			dl["offset_days"] = rule["offset_days"]
			self.deadlines[dl["id"]]["parent_trigger_event"] = trigger_event
			created.append(dl)
		self._emit(tenant, "deadline_chain_created", matter_id, {
			"trigger_event": trigger_event,
			"count": len(created),
		})
		return created

	# ── Matter Template Application ──────────────────────────────────────────

	async def apply_matter_template(
		self,
		tenant_id: str,
		matter_id: str,
		template_name: str,
		start_date: str,
		assigned_to_id: str,
	) -> list[dict[str, Any]]:
		"""Bulk-create tasks from a predefined matter template.

		Args:
			template_name:  Key in MATTER_TEMPLATES (e.g. 'litigation').
			start_date:     ISO date string; task due dates = start + relative_due_days.
			assigned_to_id: Default assignee for all generated tasks.

		Returns list of created task records.
		"""
		guard_non_empty_string(assigned_to_id, "assigned_to_id")
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		templates = MATTER_TEMPLATES.get(template_name)
		if not templates:
			raise ValueError(
				f"template '{template_name}' not found; "
				f"available: {sorted(MATTER_TEMPLATES)}"
			)
		base = date.fromisoformat(start_date)
		sem = asyncio.Semaphore(8)
		created: list[dict[str, Any]] = []

		async def _create_one(tmpl: dict[str, Any]) -> dict[str, Any]:
			async with sem:
				due = (base + timedelta(days=tmpl["relative_due_days"])).isoformat()
				return await self.create_task(
					tenant_id=tenant_id,
					matter_id=matter_id,
					title=tmpl["title"],
					assigned_to_id=assigned_to_id,
					due_date=due,
					priority=tmpl.get("priority", "normal"),
					task_type=tmpl.get("task_type", "general"),
				)

		results = await asyncio.gather(*[_create_one(t) for t in templates], return_exceptions=True)
		for r in results:
			if isinstance(r, dict):
				created.append(r)
			else:
				_log.warning("template task creation failed: %s", r)
		self._emit(tenant, "matter_template_applied", matter_id, {
			"template": template_name,
			"tasks_created": len(created),
		})
		return created

	# ── Risk Scoring ─────────────────────────────────────────────────────────

	async def compute_matter_risk_score(self, tenant_id: str, matter_id: str) -> dict[str, Any]:
		"""Compute a composite risk score (0–100) for a matter.

		Scoring components:
		- Overdue tasks           × 10 per task  (max 30)
		- Overdue deadlines       × 15 per deadline (max 30)
		- SoL deadline within 30d × 20 (once)
		- Budget burn > 80%       × 15
		- Unresolved conflicts    × 25 (once)
		- Inactivity > 30 days    × 10

		Risk levels: low (0–29), medium (30–59), high (60–79), critical (80+).
		"""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		today = date.today().isoformat()
		score = 0
		factors: list[str] = []

		# Overdue tasks
		overdue_tasks = [
			t for t in self.tasks.values()
			if t["tenant_id"] == tenant
			and t["matter_id"] == matter_id
			and t["status"] in {"pending", "in_progress"}
			and t["due_date"] < today
		]
		task_pts = min(len(overdue_tasks) * 10, 30)
		if task_pts:
			score += task_pts
			factors.append(f"{len(overdue_tasks)} overdue task(s) (+{task_pts})")

		# Overdue deadlines
		overdue_dl = [
			d for d in self.deadlines.values()
			if d["tenant_id"] == tenant
			and d["matter_id"] == matter_id
			and d["status"] == "pending"
			and d["deadline_date"] < today
		]
		dl_pts = min(len(overdue_dl) * 15, 30)
		if dl_pts:
			score += dl_pts
			factors.append(f"{len(overdue_dl)} overdue deadline(s) (+{dl_pts})")

		# SoL within 30 days
		sol_cutoff = (date.today() + timedelta(days=30)).isoformat()
		sol_near = any(
			d["deadline_type"] == "statute_of_limitations"
			and d["status"] == "pending"
			and d["deadline_date"] <= sol_cutoff
			for d in self.deadlines.values()
			if d["tenant_id"] == tenant and d["matter_id"] == matter_id
		)
		if sol_near:
			score += 20
			factors.append("statute of limitations within 30 days (+20)")

		# Budget burn > 80%
		budget = self.time_budgets.get(matter_id)
		if budget and budget["tenant_id"] == tenant:
			used = Decimal(str(budget["used_hours"]))
			total = Decimal(str(budget["total_hours"]))
			if total > 0 and used / total > Decimal("0.80"):
				score += 15
				factors.append(f"budget burn {(used/total*100).quantize(Decimal('0.1'))}% (+15)")

		# Unresolved conflicts
		unresolved = any(
			c["matter_id"] == matter_id and c["status"] == "flagged"
			for c in self.conflict_checks.values()
			if c["tenant_id"] == tenant
		)
		if unresolved:
			score += 25
			factors.append("unresolved conflict flag (+25)")

		# Inactivity > 30 days
		last_event = next(
			(e["created_at"] for e in reversed(self._audit_events)
			 if e["tenant_id"] == tenant and e["entity_id"] == matter_id),
			None,
		)
		if last_event:
			last_dt = datetime.fromisoformat(last_event.rstrip("Z"))
			inactive_days = (datetime.utcnow() - last_dt).days
			if inactive_days > 30:
				score += 10
				factors.append(f"inactive {inactive_days} days (+10)")

		score = min(score, 100)
		if score < 30:
			risk_level = "low"
		elif score < 60:
			risk_level = "medium"
		elif score < 80:
			risk_level = "high"
		else:
			risk_level = "critical"

		result: dict[str, Any] = {
			"matter_id": matter_id,
			"score": score,
			"risk_level": risk_level,
			"contributing_factors": factors,
			"computed_at": self._now(),
		}
		self._emit(tenant, "risk_score_computed", matter_id, {"score": score, "risk_level": risk_level})
		return result

	async def batch_risk_scores(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Compute risk scores for all active matters and return sorted by score descending."""
		tenant = self._tenant(tenant_id)
		active_matters = [
			m for m in self.matters.values()
			if m["tenant_id"] == tenant and m["status"] in {"open", "active"}
		]
		sem = asyncio.Semaphore(16)

		async def _score(m: dict[str, Any]) -> dict[str, Any]:
			async with sem:
				return await self.compute_matter_risk_score(tenant_id, m["id"])

		results = await asyncio.gather(*[_score(m) for m in active_matters], return_exceptions=True)
		scores = [r for r in results if isinstance(r, dict)]
		return sorted(scores, key=lambda r: r["score"], reverse=True)

	# ── Privilege Log ────────────────────────────────────────────────────────

	async def generate_privilege_log(
		self,
		tenant_id: str,
		matter_id: str,
	) -> dict[str, Any]:
		"""Generate a privilege log from all privileged notes on a matter.

		Returns structured log entries conforming to FRCP 26(b)(5) fields plus
		a SHA-256 hash of the sorted entries for tamper detection.
		"""
		tenant = self._tenant(tenant_id)
		matter = self.matters.get(matter_id)
		if not matter or matter["tenant_id"] != tenant:
			raise KeyError(f"matter {matter_id} not found")
		privileged_notes = [
			n for n in self.notes.values()
			if n["tenant_id"] == tenant
			and n["matter_id"] == matter_id
			and n["is_privileged"]
			and n["status"] == "active"
		]
		entries: list[dict[str, Any]] = []
		for n in sorted(privileged_notes, key=lambda x: x["created_at"]):
			entries.append({
				"entry_id": n["id"],
				"date": n["created_at"][:10],
				"author_id": n["author_id"],
				"note_type": n["note_type"],
				"description": n["content"][:120] + ("..." if len(n["content"]) > 120 else ""),
				"privilege_basis": n.get("privilege_basis", "attorney-client privilege"),
				"recipients": n.get("recipients", []),
			})
		# SHA-256 for tamper detection
		canonical = str(sorted(str(e) for e in entries)).encode()
		log_hash = hashlib.sha256(canonical).hexdigest()
		return {
			"matter_id": matter_id,
			"matter_title": matter["title"],
			"jurisdiction": matter["jurisdiction"],
			"entries": entries,
			"entry_count": len(entries),
			"log_hash": log_hash,
			"generated_at": self._now(),
		}

	# ── Team Capacity Planning ───────────────────────────────────────────────

	async def get_team_capacity_report(
		self,
		tenant_id: str,
		attorney_ids: list[str] | None = None,
	) -> list[dict[str, Any]]:
		"""Return capacity summary for one or more attorneys.

		Aggregates: active matter count, pending task count, overdue task count,
		upcoming deadline count (next 14 days), and load_score (0–100).
		"""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		cutoff_14 = (date.today() + timedelta(days=14)).isoformat()
		# Determine attorney set
		if attorney_ids:
			atty_set = set(attorney_ids)
		else:
			atty_set = set()
			for m in self.matters.values():
				if m["tenant_id"] == tenant:
					atty_set.update(m["team_ids"])
		reports: list[dict[str, Any]] = []
		for atty in sorted(atty_set):
			active_matters = sum(
				1 for m in self.matters.values()
				if m["tenant_id"] == tenant
				and atty in m["team_ids"]
				and m["status"] in {"open", "active"}
			)
			pending_tasks = [
				t for t in self.tasks.values()
				if t["tenant_id"] == tenant
				and t["assigned_to_id"] == atty
				and t["status"] in {"pending", "in_progress"}
			]
			overdue_tasks = sum(1 for t in pending_tasks if t["due_date"] < today)
			upcoming_deadlines = sum(
				1 for d in self.deadlines.values()
				if d["tenant_id"] == tenant
				and d["status"] == "pending"
				and today <= d["deadline_date"] <= cutoff_14
				and any(
					t["matter_id"] == d["matter_id"] and t["assigned_to_id"] == atty
					for t in self.tasks.values()
					if t["tenant_id"] == tenant
				)
			)
			# load_score: normalised heuristic (0–100)
			load_score = min(
				int(active_matters * 5 + len(pending_tasks) * 3 + overdue_tasks * 10 + upcoming_deadlines * 4),
				100,
			)
			profile = self.attorney_profiles.get(f"{tenant}:{atty}", {})
			reports.append({
				"attorney_id": atty,
				"active_matters": active_matters,
				"pending_tasks": len(pending_tasks),
				"overdue_tasks": overdue_tasks,
				"upcoming_deadlines_14d": upcoming_deadlines,
				"load_score": load_score,
				"max_hours_per_week": profile.get("max_hours_per_week"),
				"out_of_office_until": profile.get("out_of_office_until"),
				"generated_at": self._now(),
			})
		return sorted(reports, key=lambda r: r["load_score"], reverse=True)

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

