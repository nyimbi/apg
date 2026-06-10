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
