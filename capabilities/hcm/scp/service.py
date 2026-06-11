"""Succession Planning async service."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "hcm_scp"
READINESS_LEVELS = ["developing", "ready_in_1_year", "ready_now"]
SCENARIO_TYPES = {"planned", "emergency", "voluntary", "retirement"}
IMPACT_LEVELS = {"low", "medium", "high", "critical"}


def _nine_box_quadrant(performance: float, potential: float) -> str:
	"""Map (performance, potential) axes (each 1–3) to a nine-box quadrant label."""
	p_band = "low" if performance <= 1.5 else ("medium" if performance <= 2.5 else "high")
	q_band = "low" if potential <= 1.5 else ("medium" if potential <= 2.5 else "high")
	mapping = {
		("high", "high"): "star",
		("high", "medium"): "high_performer",
		("high", "low"): "solid_contributor",
		("medium", "high"): "high_potential",
		("medium", "medium"): "core_employee",
		("medium", "low"): "inconsistent_player",
		("low", "high"): "enigma",
		("low", "medium"): "average_performer",
		("low", "low"): "underperformer",
	}
	return mapping.get((p_band, q_band), "unknown")


class SCPService:
	"""Succession Planning — talent pools, readiness assessments, nine-box, scenarios, critical roles."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.talent_pools: dict[str, dict[str, Any]] = {}
		self.pool_members: dict[str, dict[str, Any]] = {}  # keyed by pool_id:employee_id
		self.readiness_assessments: dict[str, dict[str, Any]] = {}
		self.nine_box_entries: dict[str, dict[str, Any]] = {}
		self.succession_scenarios: dict[str, dict[str, Any]] = {}
		self.critical_roles: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

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

	def _readiness_index(self, level: str) -> int:
		try:
			return READINESS_LEVELS.index(level)
		except ValueError:
			return -1

	def _pool_member_count(self, tenant_id: str, pool_id: str) -> int:
		return sum(1 for m in self.pool_members.values() if m["tenant_id"] == tenant_id and m["pool_id"] == pool_id and m["status"] == "active")

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"talent_pools": len(self.talent_pools),
			"readiness_assessments": len(self.readiness_assessments),
			"nine_box_entries": len(self.nine_box_entries),
			"succession_scenarios": len(self.succession_scenarios),
			"critical_roles": len(self.critical_roles),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "hcm",
			"version": "1.0.0",
			"description": "Succession Planning — talent pools, readiness, nine-box, scenarios, critical roles",
			"readiness_levels": READINESS_LEVELS,
			"scenario_types": sorted(SCENARIO_TYPES),
			"impact_levels": sorted(IMPACT_LEVELS),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]

	# ── Talent Pools ──────────────────────────────────────────────────────────

	async def create_talent_pool(
		self,
		tenant_id: str,
		name: str,
		description: str | None = None,
		target_roles: list[str] | None = None,
		min_readiness_level: str = "developing",
	) -> dict[str, Any]:
		"""Create a talent pool."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		if min_readiness_level not in READINESS_LEVELS:
			raise ValueError(f"min_readiness_level must be one of {READINESS_LEVELS}")
		record: dict[str, Any] = {
			"id": self._uid("tp"),
			"tenant_id": t,
			"name": name,
			"description": description,
			"target_roles": target_roles or [],
			"min_readiness_level": min_readiness_level,
			"member_count": 0,
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.talent_pools[record["id"]] = record
		self._emit(t, "talent_pool_created", "talent_pool", record["id"], record)
		_log.info("talent_pool created: %s name=%s", record["id"], name)
		return deepcopy(record)

	async def list_talent_pools(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		"""List talent pools."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.talent_pools.values() if r["tenant_id"] == t]
		if status:
			items = [r for r in items if r["status"] == status]
		for item in items:
			item["member_count"] = self._pool_member_count(t, item["id"])
		return items

	async def get_talent_pool(self, tenant_id: str, pool_id: str) -> dict[str, Any]:
		"""Get a talent pool by ID."""
		t = self._tenant(tenant_id)
		record = self.talent_pools.get(pool_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"talent_pool {pool_id} not found")
		result = deepcopy(record)
		result["member_count"] = self._pool_member_count(t, pool_id)
		return result

	async def update_talent_pool(self, tenant_id: str, pool_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a talent pool."""
		t = self._tenant(tenant_id)
		record = self.talent_pools.get(pool_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"talent_pool {pool_id} not found")
		allowed = {"name", "description", "target_roles", "min_readiness_level", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		record["updated_at"] = self._now()
		self._emit(t, "talent_pool_updated", "talent_pool", record["id"], record)
		return deepcopy(record)

	async def delete_talent_pool(self, tenant_id: str, pool_id: str) -> bool:
		"""Delete a talent pool (only if empty)."""
		t = self._tenant(tenant_id)
		record = self.talent_pools.get(pool_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"talent_pool {pool_id} not found")
		if self._pool_member_count(t, pool_id) > 0:
			raise PermissionError("talent_pool_has_active_members")
		del self.talent_pools[pool_id]
		self._emit(t, "talent_pool_deleted", "talent_pool", pool_id, {"id": pool_id})
		return True

	async def add_to_talent_pool(
		self,
		tenant_id: str,
		pool_id: str,
		employee_id: str,
		readiness_level: str,
		added_by: str,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Add an employee to a talent pool."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		pool = self.talent_pools.get(pool_id)
		if not pool or pool["tenant_id"] != t:
			raise KeyError(f"talent_pool {pool_id} not found")
		if readiness_level not in READINESS_LEVELS:
			raise ValueError(f"readiness_level must be one of {READINESS_LEVELS}")
		if self._readiness_index(readiness_level) < self._readiness_index(pool["min_readiness_level"]):
			raise PermissionError("employee_does_not_meet_min_readiness_level")
		key = f"{pool_id}:{employee_id}"
		member: dict[str, Any] = {
			"id": self._uid("pm"),
			"tenant_id": t,
			"pool_id": pool_id,
			"employee_id": employee_id,
			"readiness_level": readiness_level,
			"added_by": added_by,
			"notes": notes,
			"status": "active",
			"added_at": self._now(),
		}
		self.pool_members[key] = member
		self._emit(t, "talent_pool_member_added", "pool_member", member["id"], member)
		return deepcopy(member)

	async def list_talent_pool_members(self, tenant_id: str, pool_id: str) -> list[dict[str, Any]]:
		"""List members of a talent pool."""
		t = self._tenant(tenant_id)
		return [deepcopy(m) for m in self.pool_members.values() if m["tenant_id"] == t and m["pool_id"] == pool_id and m["status"] == "active"]

	async def remove_from_talent_pool(self, tenant_id: str, pool_id: str, employee_id: str) -> bool:
		"""Remove an employee from a talent pool."""
		t = self._tenant(tenant_id)
		key = f"{pool_id}:{employee_id}"
		member = self.pool_members.get(key)
		if not member or member["tenant_id"] != t:
			raise KeyError(f"employee {employee_id} not in talent_pool {pool_id}")
		member["status"] = "removed"
		self._emit(t, "talent_pool_member_removed", "pool_member", member["id"], {"pool_id": pool_id, "employee_id": employee_id})
		return True

	# ── Readiness Assessments ─────────────────────────────────────────────────

	async def create_readiness_assessment(
		self,
		tenant_id: str,
		employee_id: str,
		target_role_id: str,
		readiness_level: str,
		performance_rating: float,
		potential_rating: float,
		assessed_by: str,
		development_needs: list[str] | None = None,
		risks: list[str] | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Assess an employee's readiness for a target role."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		if readiness_level not in READINESS_LEVELS:
			raise ValueError(f"readiness_level must be one of {READINESS_LEVELS}")
		if not (1.0 <= performance_rating <= 5.0):
			raise ValueError("performance_rating must be between 1.0 and 5.0")
		if not (1.0 <= potential_rating <= 5.0):
			raise ValueError("potential_rating must be between 1.0 and 5.0")
		# Map to nine-box axes (1-5 -> 1-3)
		p_axis = min(3.0, max(1.0, (performance_rating - 1) * (3.0 / 4.0) + 1.0))
		q_axis = min(3.0, max(1.0, (potential_rating - 1) * (3.0 / 4.0) + 1.0))
		quadrant = _nine_box_quadrant(p_axis, q_axis)
		record: dict[str, Any] = {
			"id": self._uid("ra"),
			"tenant_id": t,
			"employee_id": employee_id,
			"target_role_id": target_role_id,
			"readiness_level": readiness_level,
			"performance_rating": performance_rating,
			"potential_rating": potential_rating,
			"nine_box_quadrant": quadrant,
			"assessed_by": assessed_by,
			"development_needs": development_needs or [],
			"risks": risks or [],
			"notes": notes,
			"status": "current",
			"assessed_at": self._now(),
		}
		self.readiness_assessments[record["id"]] = record
		self._emit(t, "readiness_assessment_created", "readiness_assessment", record["id"], record)
		return deepcopy(record)

	async def list_readiness_assessments(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		target_role_id: str | None = None,
		readiness_level: str | None = None,
	) -> list[dict[str, Any]]:
		"""List readiness assessments."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.readiness_assessments.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if target_role_id:
			items = [r for r in items if r["target_role_id"] == target_role_id]
		if readiness_level:
			items = [r for r in items if r["readiness_level"] == readiness_level]
		return items

	async def get_readiness_assessment(self, tenant_id: str, assessment_id: str) -> dict[str, Any]:
		"""Get a readiness assessment by ID."""
		t = self._tenant(tenant_id)
		record = self.readiness_assessments.get(assessment_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"readiness_assessment {assessment_id} not found")
		return deepcopy(record)

	async def update_readiness_assessment(self, tenant_id: str, assessment_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a readiness assessment."""
		t = self._tenant(tenant_id)
		record = self.readiness_assessments.get(assessment_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"readiness_assessment {assessment_id} not found")
		allowed = {"readiness_level", "performance_rating", "potential_rating", "development_needs", "risks", "notes", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		self._emit(t, "readiness_assessment_updated", "readiness_assessment", record["id"], record)
		return deepcopy(record)

	async def delete_readiness_assessment(self, tenant_id: str, assessment_id: str) -> bool:
		"""Delete a readiness assessment."""
		t = self._tenant(tenant_id)
		record = self.readiness_assessments.get(assessment_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"readiness_assessment {assessment_id} not found")
		del self.readiness_assessments[assessment_id]
		self._emit(t, "readiness_assessment_deleted", "readiness_assessment", assessment_id, {"id": assessment_id})
		return True

	# ── Nine-Box Grid ─────────────────────────────────────────────────────────

	async def place_on_nine_box(
		self,
		tenant_id: str,
		employee_id: str,
		performance_axis: float,
		potential_axis: float,
		review_cycle: str,
		reviewer_id: str,
		label: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Place an employee on the nine-box grid for a review cycle."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		if not (1.0 <= performance_axis <= 3.0):
			raise ValueError("performance_axis must be between 1.0 and 3.0")
		if not (1.0 <= potential_axis <= 3.0):
			raise ValueError("potential_axis must be between 1.0 and 3.0")
		quadrant = _nine_box_quadrant(performance_axis, potential_axis)
		record: dict[str, Any] = {
			"id": self._uid("nb"),
			"tenant_id": t,
			"employee_id": employee_id,
			"performance_axis": performance_axis,
			"potential_axis": potential_axis,
			"quadrant": quadrant,
			"review_cycle": review_cycle,
			"reviewer_id": reviewer_id,
			"label": label,
			"notes": notes,
			"created_at": self._now(),
		}
		self.nine_box_entries[record["id"]] = record
		self._emit(t, "nine_box_placed", "nine_box_entry", record["id"], record)
		return deepcopy(record)

	async def list_nine_box_entries(
		self,
		tenant_id: str,
		review_cycle: str | None = None,
		employee_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List nine-box entries."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.nine_box_entries.values() if r["tenant_id"] == t]
		if review_cycle:
			items = [r for r in items if r["review_cycle"] == review_cycle]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		return items

	async def get_nine_box_grid(self, tenant_id: str, review_cycle: str) -> dict[str, list[dict[str, Any]]]:
		"""Return nine-box entries grouped by quadrant for a review cycle."""
		t = self._tenant(tenant_id)
		entries = await self.list_nine_box_entries(t, review_cycle=review_cycle)
		grid: dict[str, list[dict[str, Any]]] = {}
		for entry in entries:
			q = entry["quadrant"]
			grid.setdefault(q, []).append(entry)
		return grid

	# ── Succession Scenarios ──────────────────────────────────────────────────

	async def create_succession_scenario(
		self,
		tenant_id: str,
		role_id: str,
		role_title: str,
		incumbent_employee_id: str | None = None,
		scenario_type: str = "planned",
		successors: list[dict[str, Any]] | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Create a succession scenario for a role."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(role_id, "role_id")
		if scenario_type not in SCENARIO_TYPES:
			raise ValueError(f"scenario_type must be one of {SCENARIO_TYPES}")
		# Validate successors have required fields
		validated_successors = []
		for i, s in enumerate(successors or []):
			if not s.get("employee_id"):
				raise ValueError(f"successor[{i}] missing employee_id")
			validated_successors.append({
				"employee_id": s["employee_id"],
				"readiness": s.get("readiness", "developing"),
				"rank": s.get("rank", i + 1),
				"notes": s.get("notes"),
			})
		record: dict[str, Any] = {
			"id": self._uid("ss"),
			"tenant_id": t,
			"role_id": role_id,
			"role_title": role_title,
			"incumbent_employee_id": incumbent_employee_id,
			"scenario_type": scenario_type,
			"successors": validated_successors,
			"notes": notes,
			"status": "draft",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.succession_scenarios[record["id"]] = record
		self._emit(t, "succession_scenario_created", "succession_scenario", record["id"], record)
		return deepcopy(record)

	async def list_succession_scenarios(
		self,
		tenant_id: str,
		role_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List succession scenarios."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.succession_scenarios.values() if r["tenant_id"] == t]
		if role_id:
			items = [r for r in items if r["role_id"] == role_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def get_succession_scenario(self, tenant_id: str, scenario_id: str) -> dict[str, Any]:
		"""Get a succession scenario by ID."""
		t = self._tenant(tenant_id)
		record = self.succession_scenarios.get(scenario_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"succession_scenario {scenario_id} not found")
		return deepcopy(record)

	async def update_succession_scenario(self, tenant_id: str, scenario_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a succession scenario."""
		t = self._tenant(tenant_id)
		record = self.succession_scenarios.get(scenario_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"succession_scenario {scenario_id} not found")
		allowed = {"successors", "notes", "status", "incumbent_employee_id"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		record["updated_at"] = self._now()
		self._emit(t, "succession_scenario_updated", "succession_scenario", record["id"], record)
		return deepcopy(record)

	async def activate_succession_scenario(self, tenant_id: str, scenario_id: str, approved_by: str) -> dict[str, Any]:
		"""Activate a draft succession scenario."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(approved_by, "approved_by")
		record = self.succession_scenarios.get(scenario_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"succession_scenario {scenario_id} not found")
		if record["status"] != "draft":
			raise PermissionError("only_draft_scenarios_can_be_activated")
		record["status"] = "active"
		record["approved_by"] = approved_by
		record["updated_at"] = self._now()
		self._emit(t, "succession_scenario_activated", "succession_scenario", record["id"], record)
		return deepcopy(record)

	async def delete_succession_scenario(self, tenant_id: str, scenario_id: str) -> bool:
		"""Delete a draft succession scenario."""
		t = self._tenant(tenant_id)
		record = self.succession_scenarios.get(scenario_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"succession_scenario {scenario_id} not found")
		if record["status"] != "draft":
			raise PermissionError("only_draft_scenarios_can_be_deleted")
		del self.succession_scenarios[scenario_id]
		self._emit(t, "succession_scenario_deleted", "succession_scenario", scenario_id, {"id": scenario_id})
		return True

	# ── Critical Roles ────────────────────────────────────────────────────────

	async def identify_critical_role(
		self,
		tenant_id: str,
		role_id: str,
		role_title: str,
		rationale: str,
		impact_if_vacant: str,
		identified_by: str,
		time_to_fill_estimate_days: int = 90,
	) -> dict[str, Any]:
		"""Flag a role as critical."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(role_id, "role_id")
		if impact_if_vacant not in IMPACT_LEVELS:
			raise ValueError(f"impact_if_vacant must be one of {IMPACT_LEVELS}")
		# Count successors across active scenarios for this role
		successor_count = sum(
			len(s["successors"]) for s in self.succession_scenarios.values()
			if s["tenant_id"] == t and s["role_id"] == role_id and s["status"] == "active"
		)
		record: dict[str, Any] = {
			"id": self._uid("cr"),
			"tenant_id": t,
			"role_id": role_id,
			"role_title": role_title,
			"rationale": rationale,
			"impact_if_vacant": impact_if_vacant,
			"time_to_fill_estimate_days": time_to_fill_estimate_days,
			"successor_count": successor_count,
			"identified_by": identified_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.critical_roles[record["id"]] = record
		self._emit(t, "critical_role_identified", "critical_role", record["id"], record)
		return deepcopy(record)

	async def list_critical_roles(
		self,
		tenant_id: str,
		impact_if_vacant: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List critical roles."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.critical_roles.values() if r["tenant_id"] == t]
		if impact_if_vacant:
			items = [r for r in items if r["impact_if_vacant"] == impact_if_vacant]
		if status:
			items = [r for r in items if r["status"] == status]
		# Refresh successor counts
		for item in items:
			item["successor_count"] = sum(
				len(s["successors"]) for s in self.succession_scenarios.values()
				if s["tenant_id"] == t and s["role_id"] == item["role_id"] and s["status"] == "active"
			)
		return items

	async def get_critical_role(self, tenant_id: str, role_entry_id: str) -> dict[str, Any]:
		"""Get a critical role entry by ID."""
		t = self._tenant(tenant_id)
		record = self.critical_roles.get(role_entry_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"critical_role {role_entry_id} not found")
		return deepcopy(record)

	async def update_critical_role(self, tenant_id: str, role_entry_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a critical role entry."""
		t = self._tenant(tenant_id)
		record = self.critical_roles.get(role_entry_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"critical_role {role_entry_id} not found")
		allowed = {"rationale", "impact_if_vacant", "time_to_fill_estimate_days", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		self._emit(t, "critical_role_updated", "critical_role", record["id"], record)
		return deepcopy(record)

	async def delete_critical_role(self, tenant_id: str, role_entry_id: str) -> bool:
		"""Remove a critical role designation."""
		t = self._tenant(tenant_id)
		record = self.critical_roles.get(role_entry_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"critical_role {role_entry_id} not found")
		del self.critical_roles[role_entry_id]
		self._emit(t, "critical_role_removed", "critical_role", role_entry_id, {"id": role_entry_id})
		return True

	# ── Analytics & Reports ───────────────────────────────────────────────────

	async def succession_coverage_report(self, tenant_id: str) -> dict[str, Any]:
		"""Report on succession coverage for critical roles."""
		t = self._tenant(tenant_id)
		critical = await self.list_critical_roles(t)
		covered = [r for r in critical if r["successor_count"] > 0]
		uncovered = [r for r in critical if r["successor_count"] == 0]
		return {
			"tenant_id": t,
			"total_critical_roles": len(critical),
			"covered_roles": len(covered),
			"uncovered_roles": len(uncovered),
			"coverage_pct": round(len(covered) / len(critical) * 100, 1) if critical else 0.0,
			"uncovered_details": [{"role_id": r["role_id"], "title": r["role_title"], "impact": r["impact_if_vacant"]} for r in uncovered],
			"generated_at": self._now(),
		}

	async def talent_pool_readiness_report(self, tenant_id: str) -> dict[str, Any]:
		"""Readiness distribution across all talent pools."""
		t = self._tenant(tenant_id)
		pools = await self.list_talent_pools(t)
		result = []
		for pool in pools:
			members = await self.list_talent_pool_members(t, pool["id"])
			by_readiness: dict[str, int] = {}
			for m in members:
				by_readiness[m["readiness_level"]] = by_readiness.get(m["readiness_level"], 0) + 1
			result.append({
				"pool_id": pool["id"],
				"pool_name": pool["name"],
				"total_members": len(members),
				"by_readiness": by_readiness,
				"ready_now_count": by_readiness.get("ready_now", 0),
			})
		return {"pools": result, "generated_at": self._now()}

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Succession planning dashboard."""
		t = self._tenant(tenant_id)
		coverage, pool_report = await asyncio.gather(
			self.succession_coverage_report(t),
			self.talent_pool_readiness_report(t),
			return_exceptions=True,
		)
		return {
			"tenant_id": t,
			"talent_pools": sum(1 for r in self.talent_pools.values() if r["tenant_id"] == t and r["status"] == "active"),
			"readiness_assessments": sum(1 for r in self.readiness_assessments.values() if r["tenant_id"] == t),
			"nine_box_entries": sum(1 for r in self.nine_box_entries.values() if r["tenant_id"] == t),
			"active_scenarios": sum(1 for r in self.succession_scenarios.values() if r["tenant_id"] == t and r["status"] == "active"),
			"critical_roles": sum(1 for r in self.critical_roles.values() if r["tenant_id"] == t and r["status"] == "active"),
			"succession_coverage": coverage if not isinstance(coverage, Exception) else {},
			"talent_readiness": pool_report if not isinstance(pool_report, Exception) else {},
			"generated_at": self._now(),
		}

	# ── Succession Depth Scoring ──────────────────────────────────────────────

	async def succession_depth_score(self, tenant_id: str, role_id: str) -> dict[str, Any]:
		"""Compute a 0–10 succession depth score for a role across active scenarios.

		Weights: ready_now * 3, ready_in_1_year * 1.5, developing * 0.5.
		Normalised to 10 based on an ideal slate of 3 ready_now successors (score=9).
		"""
		t = self._tenant(tenant_id)
		guard_non_empty_string(role_id, "role_id")
		scenarios = [
			s for s in self.succession_scenarios.values()
			if s["tenant_id"] == t and s["role_id"] == role_id and s["status"] == "active"
		]
		counts: dict[str, int] = {"ready_now": 0, "ready_in_1_year": 0, "developing": 0}
		for scenario in scenarios:
			for successor in scenario.get("successors", []):
				level = successor.get("readiness", "developing")
				if level in counts:
					counts[level] += 1
		raw = counts["ready_now"] * 3 + counts["ready_in_1_year"] * 1.5 + counts["developing"] * 0.5
		score = round(min(raw / 9.0 * 10, 10.0), 2)
		return {
			"role_id": role_id,
			"tenant_id": t,
			"score": score,
			"ready_now": counts["ready_now"],
			"ready_in_1_year": counts["ready_in_1_year"],
			"developing": counts["developing"],
			"total_successors": sum(counts.values()),
			"risk_tier": "low" if score >= 7 else ("medium" if score >= 4 else "high"),
			"computed_at": self._now(),
		}

	# ── Bench Strength Index ──────────────────────────────────────────────────

	async def bench_strength_index(self, tenant_id: str, pool_id: str | None = None) -> dict[str, Any]:
		"""Compute Bench Strength Index (BSI) across all or a specific talent pool.

		BSI = (ready_now + 0.5 * ready_in_1_year) / total_members * 100.
		Returns 0.0 when the pool is empty.
		"""
		t = self._tenant(tenant_id)
		members = [
			m for m in self.pool_members.values()
			if m["tenant_id"] == t and m["status"] == "active"
			and (pool_id is None or m["pool_id"] == pool_id)
		]
		ready_now = sum(1 for m in members if m["readiness_level"] == "ready_now")
		ready_in_1 = sum(1 for m in members if m["readiness_level"] == "ready_in_1_year")
		developing = sum(1 for m in members if m["readiness_level"] == "developing")
		total = len(members)
		bsi = round((ready_now + 0.5 * ready_in_1) / total * 100, 1) if total else 0.0
		return {
			"tenant_id": t,
			"pool_id": pool_id,
			"bsi": bsi,
			"ready_now": ready_now,
			"ready_in_1_year": ready_in_1,
			"developing": developing,
			"total_members": total,
			"grade": "A" if bsi >= 70 else ("B" if bsi >= 50 else ("C" if bsi >= 30 else "D")),
			"computed_at": self._now(),
		}

	# ── Nine-Box Movement History ─────────────────────────────────────────────

	async def get_nine_box_movement_history(self, tenant_id: str, employee_id: str) -> dict[str, Any]:
		"""Return the chronological nine-box placement history and movement vectors for an employee."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		entries = sorted(
			[
				deepcopy(e) for e in self.nine_box_entries.values()
				if e["tenant_id"] == t and e["employee_id"] == employee_id
			],
			key=lambda e: e["created_at"],
		)
		movements: list[dict[str, Any]] = []
		for i in range(1, len(entries)):
			prev, curr = entries[i - 1], entries[i]
			changed = prev["quadrant"] != curr["quadrant"]
			movements.append({
				"from_cycle": prev["review_cycle"],
				"to_cycle": curr["review_cycle"],
				"from_quadrant": prev["quadrant"],
				"to_quadrant": curr["quadrant"],
				"performance_delta": round(curr["performance_axis"] - prev["performance_axis"], 2),
				"potential_delta": round(curr["potential_axis"] - prev["potential_axis"], 2),
				"quadrant_changed": changed,
			})
		return {
			"employee_id": employee_id,
			"tenant_id": t,
			"placements": entries,
			"movements": movements,
			"current_quadrant": entries[-1]["quadrant"] if entries else None,
			"total_placements": len(entries),
		}

	# ── Scenario Simulation ───────────────────────────────────────────────────

	async def simulate_vacancy(self, tenant_id: str, role_id: str, incumbent_employee_id: str) -> dict[str, Any]:
		"""Simulate sudden vacancy of a role — returns coverage snapshot without persisting changes.

		Models emergency scenario: removes the incumbent from all successor slates for the role
		and re-scores depth/coverage.  No state mutation occurs.
		"""
		t = self._tenant(tenant_id)
		guard_non_empty_string(role_id, "role_id")
		guard_non_empty_string(incumbent_employee_id, "incumbent_employee_id")
		# Deep-copy relevant scenarios to avoid mutation
		scenarios = [
			deepcopy(s) for s in self.succession_scenarios.values()
			if s["tenant_id"] == t and s["role_id"] == role_id and s["status"] == "active"
		]
		# Strip incumbent from successor lists
		for scenario in scenarios:
			scenario["successors"] = [
				s for s in scenario.get("successors", [])
				if s["employee_id"] != incumbent_employee_id
			]
		counts: dict[str, int] = {"ready_now": 0, "ready_in_1_year": 0, "developing": 0}
		for scenario in scenarios:
			for successor in scenario["successors"]:
				level = successor.get("readiness", "developing")
				if level in counts:
					counts[level] += 1
		raw = counts["ready_now"] * 3 + counts["ready_in_1_year"] * 1.5 + counts["developing"] * 0.5
		score = round(min(raw / 9.0 * 10, 10.0), 2)
		return {
			"simulation": "vacancy",
			"role_id": role_id,
			"tenant_id": t,
			"incumbent_removed": incumbent_employee_id,
			"remaining_successors": sum(len(s["successors"]) for s in scenarios),
			"depth_score_post_vacancy": score,
			"ready_now": counts["ready_now"],
			"ready_in_1_year": counts["ready_in_1_year"],
			"developing": counts["developing"],
			"risk_tier": "low" if score >= 7 else ("medium" if score >= 4 else "high"),
			"simulated_at": self._now(),
		}

	# ── Overdue Reviews ───────────────────────────────────────────────────────

	async def get_overdue_reviews(self, tenant_id: str, as_of: str | None = None) -> dict[str, Any]:
		"""Return succession scenarios and critical roles past their review_due_date.

		``as_of`` is an ISO-8601 date string; defaults to today.  Records missing
		``review_due_date`` are excluded (they are not yet subject to cadence enforcement).
		"""
		t = self._tenant(tenant_id)
		cutoff = as_of or self._now()[:10]  # YYYY-MM-DD
		overdue_scenarios: list[dict[str, Any]] = []
		for s in self.succession_scenarios.values():
			if s["tenant_id"] != t:
				continue
			due = s.get("review_due_date")
			if due and due < cutoff and s["status"] == "active":
				overdue_scenarios.append(deepcopy(s))
		overdue_roles: list[dict[str, Any]] = []
		for r in self.critical_roles.values():
			if r["tenant_id"] != t:
				continue
			due = r.get("review_due_date")
			if due and due < cutoff and r["status"] == "active":
				overdue_roles.append(deepcopy(r))
		return {
			"tenant_id": t,
			"as_of": cutoff,
			"overdue_scenarios": overdue_scenarios,
			"overdue_critical_roles": overdue_roles,
			"total_overdue": len(overdue_scenarios) + len(overdue_roles),
		}

	# ── Bulk Assessment Import ────────────────────────────────────────────────

	async def bulk_create_readiness_assessments(
		self,
		tenant_id: str,
		assessments: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Validate and commit multiple readiness assessments in one call.

		Each item in ``assessments`` must conform to the same field contract as
		``create_readiness_assessment``.  Returns a batch result with per-record
		success/failure; partial failures do not roll back successful records.
		"""
		t = self._tenant(tenant_id)
		results: list[dict[str, Any]] = []
		required = {"employee_id", "target_role_id", "readiness_level", "performance_rating", "potential_rating", "assessed_by"}
		succeeded = 0
		failed = 0
		for i, item in enumerate(assessments):
			missing = required - item.keys()
			if missing:
				results.append({"index": i, "status": "error", "error": f"missing fields: {sorted(missing)}"})
				failed += 1
				continue
			try:
				record = await self.create_readiness_assessment(
					tenant_id=t,
					employee_id=item["employee_id"],
					target_role_id=item["target_role_id"],
					readiness_level=item["readiness_level"],
					performance_rating=item["performance_rating"],
					potential_rating=item["potential_rating"],
					assessed_by=item["assessed_by"],
					development_needs=item.get("development_needs"),
					risks=item.get("risks"),
					notes=item.get("notes"),
				)
				results.append({"index": i, "status": "ok", "id": record["id"]})
				succeeded += 1
			except Exception as exc:
				results.append({"index": i, "status": "error", "error": str(exc)})
				failed += 1
		_log.info("bulk_create_readiness_assessments: %d ok, %d failed", succeeded, failed)
		return {
			"tenant_id": t,
			"total": len(assessments),
			"succeeded": succeeded,
			"failed": failed,
			"results": results,
		}

	# ── Retention Risk Alerts ─────────────────────────────────────────────────

	async def get_retention_risk_alerts(
		self,
		tenant_id: str,
		stale_months_threshold: int = 18,
		depth_score_threshold: float = 3.0,
	) -> dict[str, Any]:
		"""Surface retention and succession risk alerts.

		Emits alerts for:
		  (a) ready_now pool members with tenure > stale_months_threshold months
			  (high retention risk — stuck without progression).
		  (b) critical roles whose succession depth score < depth_score_threshold.
		  (c) nine-box stars not reassessed within 12 months.

		Thresholds are configurable per call.
		"""
		t = self._tenant(tenant_id)
		now_str = self._now()
		now_dt = datetime.fromisoformat(now_str.rstrip("Z"))

		alerts: list[dict[str, Any]] = []

		# (a) Stale ready_now pool members
		for m in self.pool_members.values():
			if m["tenant_id"] != t or m["status"] != "active" or m["readiness_level"] != "ready_now":
				continue
			added_dt = datetime.fromisoformat(m["added_at"].rstrip("Z"))
			months_in_pool = (now_dt - added_dt).days / 30.44
			if months_in_pool > stale_months_threshold:
				alerts.append({
					"alert_type": "stale_ready_now_successor",
					"severity": "high",
					"employee_id": m["employee_id"],
					"pool_id": m["pool_id"],
					"months_in_pool": round(months_in_pool, 1),
					"message": f"Employee {m['employee_id']} has been 'ready_now' for {round(months_in_pool, 1)} months without progression.",
				})

		# (b) Critical roles with low depth score
		for r in self.critical_roles.values():
			if r["tenant_id"] != t or r["status"] != "active":
				continue
			depth = await self.succession_depth_score(t, r["role_id"])
			if depth["score"] < depth_score_threshold:
				alerts.append({
					"alert_type": "low_succession_depth",
					"severity": "critical" if r["impact_if_vacant"] == "critical" else "high",
					"role_id": r["role_id"],
					"role_title": r["role_title"],
					"depth_score": depth["score"],
					"threshold": depth_score_threshold,
					"message": f"Critical role '{r['role_title']}' has succession depth {depth['score']} < threshold {depth_score_threshold}.",
				})

		# (c) Nine-box stars not reassessed in 12 months
		star_last_seen: dict[str, datetime] = {}
		for e in self.nine_box_entries.values():
			if e["tenant_id"] != t or e["quadrant"] != "star":
				continue
			emp = e["employee_id"]
			entry_dt = datetime.fromisoformat(e["created_at"].rstrip("Z"))
			if emp not in star_last_seen or entry_dt > star_last_seen[emp]:
				star_last_seen[emp] = entry_dt
		for emp_id, last_dt in star_last_seen.items():
			months_ago = (now_dt - last_dt).days / 30.44
			if months_ago > 12:
				alerts.append({
					"alert_type": "star_not_reassessed",
					"severity": "medium",
					"employee_id": emp_id,
					"last_assessed_months_ago": round(months_ago, 1),
					"message": f"Nine-box star {emp_id} has not been reassessed for {round(months_ago, 1)} months.",
				})

		self._emit(t, "retention_risk_alerts_generated", "alerts", "batch", {"count": len(alerts)})
		return {
			"tenant_id": t,
			"total_alerts": len(alerts),
			"critical": sum(1 for a in alerts if a["severity"] == "critical"),
			"high": sum(1 for a in alerts if a["severity"] == "high"),
			"medium": sum(1 for a in alerts if a["severity"] == "medium"),
			"alerts": alerts,
			"generated_at": now_str,
		}

	# ── Role Risk Registry ────────────────────────────────────────────────────

	async def role_risk_registry(self, tenant_id: str) -> dict[str, Any]:
		"""Produce a prioritised role risk registry for all active critical roles.

		Composite risk score combines:
		  - impact_if_vacant weight (critical=4, high=3, medium=2, low=1)
		  - succession depth score contribution (inverse: 10 - depth_score)
		  - time_to_fill normalised contribution (days / 365 * 3, capped at 3)

		Roles are sorted descending by composite risk score.
		"""
		t = self._tenant(tenant_id)
		impact_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
		registry: list[dict[str, Any]] = []

		roles = await self.list_critical_roles(t, status="active")
		depth_tasks = [self.succession_depth_score(t, r["role_id"]) for r in roles]
		depths = await asyncio.gather(*depth_tasks, return_exceptions=True)

		for role, depth in zip(roles, depths):
			impact_w = impact_weights.get(role["impact_if_vacant"], 1)
			depth_w = 10 - depth["score"]  # higher when fewer successors
			fill_w = min(role.get("time_to_fill_estimate_days", 90) / 365 * 3, 3.0)
			composite = round(impact_w * 0.4 + depth_w * 0.4 + fill_w * 0.2, 2)
			registry.append({
				"role_id": role["role_id"],
				"role_title": role["role_title"],
				"impact_if_vacant": role["impact_if_vacant"],
				"succession_depth_score": depth["score"],
				"time_to_fill_days": role.get("time_to_fill_estimate_days", 90),
				"composite_risk_score": composite,
				"risk_tier": "critical" if composite >= 6 else ("high" if composite >= 4 else ("medium" if composite >= 2 else "low")),
			})

		registry.sort(key=lambda r: r["composite_risk_score"], reverse=True)
		return {
			"tenant_id": t,
			"total_critical_roles": len(registry),
			"registry": registry,
			"generated_at": self._now(),
		}
