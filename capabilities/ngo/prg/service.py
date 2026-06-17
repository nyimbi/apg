"""Programme & Project Monitoring Service — logframes, activities, outputs, outcomes, field data."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "ngo_prg"

SUPPORTED_STATUSES = {"planning", "active", "on_hold", "completed", "cancelled"}
SUPPORTED_OUTPUT_TYPES = {"quantitative", "qualitative", "milestone"}
SUPPORTED_DATA_TYPES = {"observation", "survey", "interview", "focus_group", "secondary"}


class ProgrammeMonitoringService:
	"""Async service for NGO programme and project monitoring."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._programmes = WriteThruDict('programmes', tenant_id, _store)
		self._logframes = WriteThruDict('logframes', tenant_id, _store)
		self._activities = WriteThruDict('activities', tenant_id, _store)
		self._outputs = WriteThruDict('outputs', tenant_id, _store)
		self._field_data = WriteThruDict('field_data', tenant_id, _store)
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	# ── helpers ───────────────────────────────────────────────────────────────

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self) -> str:
		if not self.tenant_id:
			raise PermissionError("tenant_context_required")
		return self.tenant_id

	def _emit(self, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt"),
			"tenant_id": self._tenant(),
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"emitted_at": self._now(),
		})

	def _guard_programme(self, programme_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		p = self._programmes.get(programme_id)
		if not p or p["tenant_id"] != tenant:
			raise KeyError(f"programme_not_found:{programme_id}")
		return p

	def _guard_activity(self, activity_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		a = self._activities.get(activity_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"activity_not_found:{activity_id}")
		return a

	# ── health / describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"programme_count": len(self._programmes),
			"active_programmes": sum(1 for p in self._programmes.values() if p["status"] == "active"),
			"activity_count": len(self._activities),
			"field_data_records": len(self._field_data),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "ngo",
			"version": "1.0.0",
			"description": "Logframe management, activity tracking, output/outcome recording, field data collection",
			"supported_statuses": list(SUPPORTED_STATUSES),
			"tenant_id": self.tenant_id,
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant()
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		return [deepcopy(e) for e in events[-limit:]]

	# ── programmes ────────────────────────────────────────────────────────────

	async def list_programmes(self, status: str | None = None, sector: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(p) for p in self._programmes.values() if p["tenant_id"] == tenant]
		if status:
			items = [p for p in items if p["status"] == status]
		if sector:
			items = [p for p in items if p.get("sector") == sector]
		return items

	async def get_programme(self, programme_id: str) -> dict[str, Any]:
		return deepcopy(self._guard_programme(programme_id))

	async def create_programme(
		self,
		name: str,
		code: str,
		start_date: str,
		end_date: str,
		description: str = "",
		sector: str = "",
		budget: Decimal = Decimal("0"),
		currency: str = "KES",
		lead_staff: str = "",
		geographic_focus: str = "",
	) -> dict[str, Any]:
		"""Create a new programme."""
		tenant = self._tenant()
		if not name or not code:
			raise ValueError("name_and_code_required")
		# check code uniqueness
		if any(p["code"] == code and p["tenant_id"] == tenant for p in self._programmes.values()):
			raise ValueError(f"programme_code_already_exists:{code}")
		record: dict[str, Any] = {
			"id": self._id("prg"),
			"type": "ngo_programme",
			"tenant_id": tenant,
			"name": name,
			"code": code,
			"description": description,
			"sector": sector,
			"start_date": start_date,
			"end_date": end_date,
			"budget": budget,
			"currency": currency,
			"lead_staff": lead_staff,
			"geographic_focus": geographic_focus,
			"status": "planning",
			"created_at": self._now(),
			"updated_at": None,
		}
		self._programmes[record["id"]] = record
		self._emit("programme_created", record["id"], "ngo_programme", {"name": name, "code": code})
		_log.info("Programme created: %s (%s)", record["id"], code)
		return deepcopy(record)

	async def update_programme(self, programme_id: str, **kwargs: Any) -> dict[str, Any]:
		p = self._guard_programme(programme_id)
		allowed = {"name", "description", "end_date", "budget", "status", "lead_staff", "geographic_focus"}
		if "status" in kwargs and kwargs["status"] not in SUPPORTED_STATUSES:
			raise ValueError(f"invalid_status:{kwargs['status']}")
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				p[k] = v
		p["updated_at"] = self._now()
		self._emit("programme_updated", programme_id, "ngo_programme", kwargs)
		return deepcopy(p)

	async def activate_programme(self, programme_id: str) -> dict[str, Any]:
		p = self._guard_programme(programme_id)
		if p["status"] != "planning":
			raise ValueError(f"cannot_activate_from:{p['status']}")
		p["status"] = "active"
		p["activated_at"] = self._now()
		p["updated_at"] = self._now()
		self._emit("programme_activated", programme_id, "ngo_programme")
		return deepcopy(p)

	async def close_programme(self, programme_id: str, closed_by: str) -> dict[str, Any]:
		p = self._guard_programme(programme_id)
		p["status"] = "completed"
		p["closed_by"] = closed_by
		p["closed_at"] = self._now()
		p["updated_at"] = self._now()
		self._emit("programme_closed", programme_id, "ngo_programme", {"closed_by": closed_by})
		return deepcopy(p)

	async def delete_programme(self, programme_id: str) -> dict[str, Any]:
		p = self._guard_programme(programme_id)
		if p["status"] not in {"planning", "cancelled"}:
			raise ValueError("only_planning_programmes_may_be_deleted")
		removed = self._programmes.pop(programme_id)
		self._emit("programme_deleted", programme_id, "ngo_programme")
		return deepcopy(removed)

	# ── logframes ─────────────────────────────────────────────────────────────

	async def list_logframes(self, programme_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(lf) for lf in self._logframes.values() if lf["tenant_id"] == tenant]
		if programme_id:
			items = [lf for lf in items if lf["programme_id"] == programme_id]
		return items

	async def get_logframe(self, logframe_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		lf = self._logframes.get(logframe_id)
		if not lf or lf["tenant_id"] != tenant:
			raise KeyError(f"logframe_not_found:{logframe_id}")
		return deepcopy(lf)

	async def create_logframe(
		self,
		programme_id: str,
		goal: str,
		purpose: str,
		outputs: list[str] | None = None,
		activities: list[str] | None = None,
		assumptions: list[str] | None = None,
		version: str = "1.0",
	) -> dict[str, Any]:
		"""Create a logframe for a programme."""
		self._guard_programme(programme_id)
		record: dict[str, Any] = {
			"id": self._id("lf"),
			"type": "ngo_logframe",
			"tenant_id": self._tenant(),
			"programme_id": programme_id,
			"goal": goal,
			"purpose": purpose,
			"outputs": outputs or [],
			"activities": activities or [],
			"assumptions": assumptions or [],
			"version": version,
			"status": "draft",
			"created_at": self._now(),
		}
		self._logframes[record["id"]] = record
		self._emit("logframe_created", record["id"], "ngo_logframe", {"programme_id": programme_id})
		return deepcopy(record)

	async def approve_logframe(self, logframe_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant()
		lf = self._logframes.get(logframe_id)
		if not lf or lf["tenant_id"] != tenant:
			raise KeyError(f"logframe_not_found:{logframe_id}")
		lf["status"] = "approved"
		lf["approved_by"] = approved_by
		lf["approved_at"] = self._now()
		self._emit("logframe_approved", logframe_id, "ngo_logframe", {"approved_by": approved_by})
		return deepcopy(lf)

	async def update_logframe(self, logframe_id: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant()
		lf = self._logframes.get(logframe_id)
		if not lf or lf["tenant_id"] != tenant:
			raise KeyError(f"logframe_not_found:{logframe_id}")
		allowed = {"goal", "purpose", "outputs", "activities", "assumptions", "version"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				lf[k] = v
		self._emit("logframe_updated", logframe_id, "ngo_logframe", kwargs)
		return deepcopy(lf)

	# ── activities ────────────────────────────────────────────────────────────

	async def list_activities(self, programme_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(a) for a in self._activities.values() if a["tenant_id"] == tenant]
		if programme_id:
			items = [a for a in items if a["programme_id"] == programme_id]
		if status:
			items = [a for a in items if a["status"] == status]
		return items

	async def get_activity(self, activity_id: str) -> dict[str, Any]:
		return deepcopy(self._guard_activity(activity_id))

	async def create_activity(
		self,
		programme_id: str,
		name: str,
		planned_start: str,
		planned_end: str,
		description: str = "",
		responsible_person: str = "",
		budget: Decimal = Decimal("0"),
		currency: str = "KES",
		logframe_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a project activity."""
		self._guard_programme(programme_id)
		if not name:
			raise ValueError("activity_name_required")
		record: dict[str, Any] = {
			"id": self._id("act"),
			"type": "ngo_activity",
			"tenant_id": self._tenant(),
			"programme_id": programme_id,
			"logframe_id": logframe_id,
			"name": name,
			"description": description,
			"responsible_person": responsible_person,
			"planned_start": planned_start,
			"planned_end": planned_end,
			"actual_start": None,
			"actual_end": None,
			"budget": budget,
			"currency": currency,
			"completion_pct": 0.0,
			"status": "planned",
			"created_at": self._now(),
		}
		self._activities[record["id"]] = record
		self._emit("activity_created", record["id"], "ngo_activity", {"programme_id": programme_id, "name": name})
		return deepcopy(record)

	async def update_activity(self, activity_id: str, **kwargs: Any) -> dict[str, Any]:
		a = self._guard_activity(activity_id)
		allowed = {"name", "description", "responsible_person", "planned_end", "actual_start",
				   "actual_end", "completion_pct", "status", "budget"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				a[k] = v
		if "completion_pct" in kwargs:
			a["completion_pct"] = min(100.0, max(0.0, float(kwargs["completion_pct"])))
		self._emit("activity_updated", activity_id, "ngo_activity", kwargs)
		return deepcopy(a)

	async def start_activity(self, activity_id: str) -> dict[str, Any]:
		a = self._guard_activity(activity_id)
		a["status"] = "in_progress"
		a["actual_start"] = self._now()[:10]
		self._emit("activity_started", activity_id, "ngo_activity")
		return deepcopy(a)

	async def complete_activity(self, activity_id: str) -> dict[str, Any]:
		a = self._guard_activity(activity_id)
		a["status"] = "completed"
		a["completion_pct"] = 100.0
		a["actual_end"] = self._now()[:10]
		self._emit("activity_completed", activity_id, "ngo_activity")
		return deepcopy(a)

	async def delete_activity(self, activity_id: str) -> dict[str, Any]:
		a = self._guard_activity(activity_id)
		if a["status"] not in {"planned"}:
			raise ValueError("only_planned_activities_may_be_deleted")
		removed = self._activities.pop(activity_id)
		self._emit("activity_deleted", activity_id, "ngo_activity")
		return deepcopy(removed)

	# ── outputs ───────────────────────────────────────────────────────────────

	async def list_outputs(self, activity_id: str | None = None, programme_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(o) for o in self._outputs.values() if o["tenant_id"] == tenant]
		if activity_id:
			items = [o for o in items if o["activity_id"] == activity_id]
		if programme_id:
			items = [o for o in items if o["programme_id"] == programme_id]
		return items

	async def record_output(
		self,
		activity_id: str,
		programme_id: str,
		description: str,
		target_value: float,
		reporting_date: str,
		recorded_by: str,
		achieved_value: float = 0.0,
		output_type: str = "quantitative",
		unit: str = "",
	) -> dict[str, Any]:
		"""Record an output/outcome against an activity."""
		self._guard_activity(activity_id)
		if output_type not in SUPPORTED_OUTPUT_TYPES:
			raise ValueError(f"unsupported_output_type:{output_type}")
		achievement_pct = (achieved_value / target_value * 100) if target_value else 0.0
		record: dict[str, Any] = {
			"id": self._id("out"),
			"type": "ngo_output",
			"tenant_id": self._tenant(),
			"activity_id": activity_id,
			"programme_id": programme_id,
			"output_type": output_type,
			"description": description,
			"target_value": target_value,
			"achieved_value": achieved_value,
			"unit": unit,
			"reporting_date": reporting_date,
			"recorded_by": recorded_by,
			"achievement_pct": round(achievement_pct, 2),
			"status": "recorded",
			"created_at": self._now(),
		}
		self._outputs[record["id"]] = record
		self._emit("output_recorded", record["id"], "ngo_output",
				   {"activity_id": activity_id, "achievement_pct": achievement_pct})
		return deepcopy(record)

	async def update_output(self, output_id: str, achieved_value: float) -> dict[str, Any]:
		"""Update achieved value on an output."""
		tenant = self._tenant()
		out = self._outputs.get(output_id)
		if not out or out["tenant_id"] != tenant:
			raise KeyError(f"output_not_found:{output_id}")
		out["achieved_value"] = achieved_value
		out["achievement_pct"] = round(achieved_value / out["target_value"] * 100, 2) if out["target_value"] else 0.0
		self._emit("output_updated", output_id, "ngo_output", {"achieved_value": achieved_value})
		return deepcopy(out)

	# ── field data ────────────────────────────────────────────────────────────

	async def list_field_data(self, programme_id: str | None = None, activity_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant()
		items = [deepcopy(fd) for fd in self._field_data.values() if fd["tenant_id"] == tenant]
		if programme_id:
			items = [fd for fd in items if fd["programme_id"] == programme_id]
		if activity_id:
			items = [fd for fd in items if fd.get("activity_id") == activity_id]
		return items

	async def submit_field_data(
		self,
		programme_id: str,
		collector: str,
		collection_date: str,
		data: dict[str, Any],
		activity_id: str | None = None,
		location: str = "",
		data_type: str = "observation",
		notes: str = "",
	) -> dict[str, Any]:
		"""Submit field data collection record."""
		self._guard_programme(programme_id)
		if data_type not in SUPPORTED_DATA_TYPES:
			raise ValueError(f"unsupported_data_type:{data_type}")
		record: dict[str, Any] = {
			"id": self._id("fd"),
			"type": "ngo_field_data",
			"tenant_id": self._tenant(),
			"programme_id": programme_id,
			"activity_id": activity_id,
			"collector": collector,
			"collection_date": collection_date,
			"location": location,
			"data_type": data_type,
			"data": deepcopy(data),
			"notes": notes,
			"verified": False,
			"created_at": self._now(),
		}
		self._field_data[record["id"]] = record
		self._emit("field_data_submitted", record["id"], "ngo_field_data",
				   {"programme_id": programme_id, "collector": collector})
		return deepcopy(record)

	async def verify_field_data(self, field_data_id: str, verified_by: str) -> dict[str, Any]:
		"""Mark field data as verified."""
		tenant = self._tenant()
		fd = self._field_data.get(field_data_id)
		if not fd or fd["tenant_id"] != tenant:
			raise KeyError(f"field_data_not_found:{field_data_id}")
		fd["verified"] = True
		fd["verified_by"] = verified_by
		fd["verified_at"] = self._now()
		self._emit("field_data_verified", field_data_id, "ngo_field_data", {"verified_by": verified_by})
		return deepcopy(fd)

	# ── analytics ─────────────────────────────────────────────────────────────

	async def programme_progress_report(self, programme_id: str) -> dict[str, Any]:
		"""Generate a progress report for a programme."""
		programme = self._guard_programme(programme_id)
		activities = [a for a in self._activities.values() if a["programme_id"] == programme_id]
		outputs = [o for o in self._outputs.values() if o["programme_id"] == programme_id]
		completed = [a for a in activities if a["status"] == "completed"]
		avg_completion = sum(a["completion_pct"] for a in activities) / len(activities) if activities else 0.0
		avg_achievement = sum(o["achievement_pct"] for o in outputs) / len(outputs) if outputs else 0.0
		return {
			"programme_id": programme_id,
			"programme_name": programme["name"],
			"status": programme["status"],
			"total_activities": len(activities),
			"completed_activities": len(completed),
			"avg_completion_pct": round(avg_completion, 2),
			"total_outputs": len(outputs),
			"avg_achievement_pct": round(avg_achievement, 2),
			"field_data_records": len([fd for fd in self._field_data.values() if fd["programme_id"] == programme_id]),
			"generated_at": self._now(),
		}

	async def portfolio_overview(self) -> dict[str, Any]:
		"""Return portfolio overview across all programmes."""
		tenant = self._tenant()
		programmes = [p for p in self._programmes.values() if p["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		for p in programmes:
			by_status[p["status"]] = by_status.get(p["status"], 0) + 1
		return {
			"tenant_id": tenant,
			"total_programmes": len(programmes),
			"by_status": by_status,
			"total_activities": sum(1 for a in self._activities.values() if a["tenant_id"] == tenant),
			"generated_at": self._now(),
		}

	async def activity_gantt_data(self, programme_id: str) -> list[dict[str, Any]]:
		"""Return activity data suitable for Gantt rendering."""
		self._guard_programme(programme_id)
		return [
			{
				"id": a["id"],
				"name": a["name"],
				"planned_start": a["planned_start"],
				"planned_end": a["planned_end"],
				"actual_start": a.get("actual_start"),
				"actual_end": a.get("actual_end"),
				"completion_pct": a["completion_pct"],
				"status": a["status"],
				"responsible_person": a.get("responsible_person", ""),
			}
			for a in self._activities.values()
			if a["programme_id"] == programme_id
		]

	async def bulk_create_activities(self, programme_id: str, activities: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create activities for a programme."""
		tasks = [
			self.create_activity(
				programme_id=programme_id,
				name=a["name"],
				planned_start=a["planned_start"],
				planned_end=a["planned_end"],
				description=a.get("description", ""),
				responsible_person=a.get("responsible_person", ""),
				budget=Decimal(str(a.get("budget", 0))),
				currency=a.get("currency", "KES"),
				logframe_id=a.get("logframe_id"),
			)
			for a in activities
		]
		outcomes = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for act, outcome in zip(activities, outcomes):
			if isinstance(outcome, Exception):
				errors.append({"input": act, "error": str(outcome)})
			else:
				results.append(outcome)
		return {"created": len(results), "failed": len(errors), "activities": results, "errors": errors}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_programmes', '_logframes', '_activities', '_outputs', '_field_data', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

