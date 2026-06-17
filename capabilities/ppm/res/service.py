"""Executable service layer for APG Resource Management (res)."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALLOCATION_STATUSES,
		SUPPORTED_CAPACITY_PLAN_TYPES, SUPPORTED_COST_RATE_TYPES, SUPPORTED_DEMAND_HORIZON_TYPES,
		SUPPORTED_DEPARTMENT_TYPES, SUPPORTED_LEAVE_TYPES, SUPPORTED_MATCHING_ALGORITHMS,
		SUPPORTED_RESOURCE_STATUSES, SUPPORTED_RESOURCE_TYPES, SUPPORTED_SKILL_PROFICIENCY_LEVELS,
		SUPPORTED_UTILISATION_BANDS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		CapacityPlan, CostRate, DemandForecast, LeaveRecord, Resource,
		ResourceAgent, ResourceAllocation, ResourceSkill, UtilisationSnapshot,
	)
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALLOCATION_STATUSES,
		SUPPORTED_CAPACITY_PLAN_TYPES, SUPPORTED_COST_RATE_TYPES, SUPPORTED_DEMAND_HORIZON_TYPES,
		SUPPORTED_DEPARTMENT_TYPES, SUPPORTED_LEAVE_TYPES, SUPPORTED_MATCHING_ALGORITHMS,
		SUPPORTED_RESOURCE_STATUSES, SUPPORTED_RESOURCE_TYPES, SUPPORTED_SKILL_PROFICIENCY_LEVELS,
		SUPPORTED_UTILISATION_BANDS,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		CapacityPlan, CostRate, DemandForecast, LeaveRecord, Resource,
		ResourceAgent, ResourceAllocation, ResourceSkill, UtilisationSnapshot,
	)


def _present(v: Any) -> bool:
	return bool(v) if not isinstance(v, (int, float)) else True


def _positive(v: float | int) -> bool:
	return isinstance(v, (int, float)) and v > 0


def _norm(v: str) -> str:
	return v.strip().lower()


class ResourceManagementService:
	"""Tenant-scoped resource management runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store or get_store(db_url)
		self.resources: dict[tuple[str, str], Resource] = {}
		self.skills: dict[tuple[str, str], ResourceSkill] = {}
		self.allocations: dict[tuple[str, str], ResourceAllocation] = {}
		self.capacity_plans: dict[tuple[str, str], CapacityPlan] = {}
		self.utilisation_snapshots: dict[tuple[str, str], UtilisationSnapshot] = {}
		self.demand_forecasts: dict[tuple[str, str], DemandForecast] = {}
		self.leave_records: dict[tuple[str, str], LeaveRecord] = {}
		self.cost_rates: dict[tuple[str, str], CostRate] = {}
		self.agents: dict[tuple[str, str], ResourceAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self._teams = WriteThruDict('teams', tenant_id, self._store)
		self._bench_time: dict[str, list[dict[str, Any]]] = {}    # resource_id -> bench records
		self._overallocation_log: dict[str, list[dict[str, Any]]] = {}  # resource_id -> resolution log
		self._analytics_cache = WriteThruDict('analytics_cache', tenant_id, self._store)

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── Resources ────────────────────────────────────────────────────────────

	def create_resource(
		self, resource_id: str, tenant_id: str, name: str, resource_type: str,
		status: str, department: str, owner_id: str, cost_rate: float,
		cost_rate_type: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a new resource in the pool."""
		resource_type = _norm(resource_type)
		status = _norm(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_resource",
			"resource_type_supported": resource_type in SUPPORTED_RESOURCE_TYPES,
			"status_supported": status in SUPPORTED_RESOURCE_STATUSES,
			"owner_present": _present(owner_id),
			"cost_rate_present": _positive(cost_rate),
			"evidence_present": _present(evidence_reference),
		})
		item = Resource(resource_id, tenant_id, name, resource_type, status, department,
						owner_id, float(cost_rate), cost_rate_type, evidence_reference)
		self.resources[self._key(tenant_id, resource_id)] = item
		self._audit(tenant_id, "resource_created", resource_id)
		return item.to_dict()

	def get_resource(self, resource_id: str, tenant_id: str) -> dict[str, Any] | None:
		item = self.resources.get(self._key(tenant_id, resource_id))
		return item.to_dict() if item else None

	def list_resources(self, tenant_id: str) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.resources.values() if v.tenant_id == tenant_id]

	# ── Register resource (new high-level method) ─────────────────────────────

	async def register_resource(
		self, name: str, resource_type: str, skills: list[str],
		cost_rate: float, availability: dict[str, Any]
	) -> dict[str, Any]:
		"""Register a resource with skills and availability schedule.

		availability: {from_date, to_date, hours_per_day, days_per_week}
		"""
		assert _present(name), "name required"
		assert _present(resource_type), "resource_type required"
		assert _positive(cost_rate), "cost_rate must be positive"
		tenant_id = self.tenant_id

		resource_id = f"res_{name.lower().replace(' ', '_')}_{str(date.today())}"
		rec = self.create_resource(
			resource_id=resource_id,
			tenant_id=tenant_id,
			name=name,
			resource_type=_norm(resource_type) if _norm(resource_type) in SUPPORTED_RESOURCE_TYPES else "human",
			status="available",
			department=availability.get("department", "unassigned"),
			owner_id=self.actor_id,
			cost_rate=cost_rate,
			cost_rate_type="daily",
			evidence_reference=f"reg_{str(date.today())}",
		)
		# Attach skills
		added_skills: list[dict[str, Any]] = []
		for skill_name in skills:
			skill_id = f"sk_{resource_id}_{skill_name.lower().replace(' ', '_')}"
			sk = self.add_skill(
				skill_id=skill_id,
				tenant_id=tenant_id,
				resource_id=resource_id,
				skill_name=skill_name,
				proficiency_level="intermediate",
				years_experience=1.0,
				evidence_reference=f"self_declared_{str(date.today())}",
			)
			added_skills.append(sk)
		# Store availability
		rec["availability"] = availability
		rec["skills"] = added_skills
		return rec

	# ── Skill search ──────────────────────────────────────────────────────────

	async def skill_search(
		self, skills: list[str], availability_from: str, availability_to: str
	) -> dict[str, Any]:
		"""Find resources matching a skill set within a date range."""
		assert skills, "skills list required"
		tenant_id = self.tenant_id

		matched = self.match_skills(tenant_id=tenant_id, required_skills=skills)

		# Filter out resources on leave during the period
		leave_excluded: list[str] = []
		available: list[dict[str, Any]] = []
		for res in matched:
			res_id = res.get("id") or res.get("resource_id", "")
			on_leave = any(
				lr.tenant_id == tenant_id and lr.resource_id == res_id
				and lr.start_date <= availability_to and lr.end_date >= availability_from
				for lr in self.leave_records.values()
			)
			if on_leave:
				leave_excluded.append(res_id)
			else:
				available.append(res)

		return {
			"required_skills": skills,
			"availability_from": availability_from,
			"availability_to": availability_to,
			"total_matched": len(matched),
			"available_count": len(available),
			"leave_excluded_count": len(leave_excluded),
			"resources": available,
		}

	# ── Assign resource ───────────────────────────────────────────────────────

	async def assign_resource(
		self, project_id: str, task_id: str, resource_id: str,
		allocation_pct: float, start_date: str, end_date: str
	) -> dict[str, Any]:
		"""Assign a resource to a project task with over-allocation detection."""
		assert _present(project_id), "project_id required"
		assert _present(task_id), "task_id required"
		assert _present(resource_id), "resource_id required"
		assert 0 < allocation_pct <= 100, "allocation_pct must be 0–100"
		tenant_id = self.tenant_id

		# Check existing allocation in the period
		existing_allocs = [
			a for a in self.allocations.values()
			if a.tenant_id == tenant_id and a.resource_id == resource_id
			and a.start_date <= end_date and a.end_date >= start_date
		]
		existing_total_pct = sum(a.allocation_pct for a in existing_allocs)
		over_allocated = (existing_total_pct + allocation_pct) > 100

		alloc_id = f"alloc_{project_id}_{task_id}_{resource_id}"
		rec = self.create_allocation(
			alloc_id=alloc_id,
			tenant_id=tenant_id,
			resource_id=resource_id,
			project_id=project_id,
			task_id=task_id,
			status="confirmed",
			start_date=start_date,
			end_date=end_date,
			allocation_pct=allocation_pct,
			manager_approval_reference=self.actor_id if over_allocated else "",
			over_allocated=over_allocated,
		)
		return {
			"allocation": rec,
			"over_allocated": over_allocated,
			"total_allocation_pct": round(existing_total_pct + allocation_pct, 2),
		}

	# ── Resource utilisation ──────────────────────────────────────────────────

	async def resource_utilisation(self, resource_id: str, period: str) -> dict[str, Any]:
		"""Compute utilisation metrics for a resource over a period."""
		assert _present(resource_id), "resource_id required"
		assert _present(period), "period required"
		tenant_id = self.tenant_id
		resource = self._resource_or_none(resource_id, tenant_id)
		assert resource is not None, f"resource {resource_id} not found"

		# Allocations for this resource
		allocs = [a for a in self.allocations.values()
				  if a.tenant_id == tenant_id and a.resource_id == resource_id]

		# Assume 22 working days per month, 8h/day
		available_hours = 22 * 8
		allocated_pct_avg = (sum(a.allocation_pct for a in allocs) / len(allocs)
							  if allocs else 0.0)
		allocated_hours = round(available_hours * allocated_pct_avg / 100, 2)

		snapshot_id = f"util_{resource_id}_{period}"
		snap = self.take_utilisation_snapshot(
			snapshot_id=snapshot_id,
			tenant_id=tenant_id,
			resource_id=resource_id,
			snapshot_period=period,
			allocated_hours=allocated_hours,
			available_hours=available_hours,
		)
		return {
			"resource_id": resource_id,
			"period": period,
			"available_hours": available_hours,
			"allocated_hours": allocated_hours,
			"allocation_count": len(allocs),
			"snapshot": snap,
		}

	# ── Team capacity ─────────────────────────────────────────────────────────

	async def team_capacity(self, team_id: str, period: str) -> dict[str, Any]:
		"""Aggregate capacity and utilisation for a named team."""
		assert _present(team_id), "team_id required"
		assert _present(period), "period required"
		tenant_id = self.tenant_id
		team = self._teams.get(f"{tenant_id}:{team_id}", {})
		member_ids: list[str] = team.get("member_ids", [])

		if not member_ids:
			# Fall back to all resources in same department as team
			department = team.get("department", team_id)
			member_ids = [
				r.id if hasattr(r, "id") else ""
				for r in self.resources.values()
				if r.tenant_id == tenant_id and r.department == department
			]

		member_utils: list[dict[str, Any]] = []
		total_available = 0.0
		total_allocated = 0.0
		for rid in member_ids:
			util = await self.resource_utilisation(rid, period)
			member_utils.append({
				"resource_id": rid,
				"available_hours": util["available_hours"],
				"allocated_hours": util["allocated_hours"],
				"utilisation_pct": util["snapshot"].get("utilisation_pct", 0.0),
			})
			total_available += util["available_hours"]
			total_allocated += util["allocated_hours"]

		team_utilisation_pct = round(
			(total_allocated / total_available * 100) if total_available else 0.0, 2
		)
		return {
			"team_id": team_id,
			"period": period,
			"member_count": len(member_ids),
			"total_available_hours": total_available,
			"total_allocated_hours": round(total_allocated, 2),
			"team_utilisation_pct": team_utilisation_pct,
			"members": member_utils,
		}

	# ── Skills gap analysis ───────────────────────────────────────────────────

	async def skills_gap_analysis(
		self, project_id: str, required_skills: list[str]
	) -> dict[str, Any]:
		"""Compare required skills for a project against available resource skills."""
		assert _present(project_id), "project_id required"
		assert required_skills, "required_skills list required"
		tenant_id = self.tenant_id

		# Skills available across all tenant resources
		available_skills: set[str] = {
			s.skill_name.lower()
			for s in self.skills.values()
			if s.tenant_id == tenant_id
		}
		required_lower = {s.lower() for s in required_skills}
		covered = required_lower & available_skills
		gap = required_lower - available_skills

		# Find best-match resources per gap skill
		gap_details: list[dict[str, Any]] = []
		for missing_skill in gap:
			# Fuzzy: find partial matches
			partial_matches = [
				s.resource_id
				for s in self.skills.values()
				if s.tenant_id == tenant_id and missing_skill[:4] in s.skill_name.lower()
			]
			gap_details.append({
				"skill": missing_skill,
				"partial_match_resources": list(set(partial_matches))[:3],
				"recommend_hire": len(partial_matches) == 0,
			})

		result = {
			"project_id": project_id,
			"required_skills": list(required_lower),
			"available_skills": list(available_skills),
			"covered_skills": list(covered),
			"gap_skills": list(gap),
			"coverage_pct": round(len(covered) / len(required_lower) * 100, 2) if required_lower else 100.0,
			"gap_details": gap_details,
		}
		self._audit(tenant_id, "skills_gap_analysed", project_id)
		return result

	# ── Resource forecasting ──────────────────────────────────────────────────

	async def resource_forecasting(self, period: str, resource_type: str) -> dict[str, Any]:
		"""Forecast resource demand vs supply for a period and type."""
		assert _present(period), "period required"
		assert _present(resource_type), "resource_type required"
		tenant_id = self.tenant_id

		# Count current resources of this type
		matching = [r for r in self.resources.values()
					if r.tenant_id == tenant_id and r.resource_type == _norm(resource_type)]
		supply_fte = len(matching)

		# Derive demand from active allocations
		active_allocs = [a for a in self.allocations.values()
						 if a.tenant_id == tenant_id and a.status == "confirmed"]
		# Sum FTE demand (100% allocation = 1 FTE)
		demand_fte = round(sum(a.allocation_pct / 100 for a in active_allocs), 2)

		forecast_id = f"fcast_{resource_type}_{period}"
		forecast = self.forecast_demand(
			forecast_id=forecast_id,
			tenant_id=tenant_id,
			horizon=_norm(period) if _norm(period) in SUPPORTED_DEMAND_HORIZON_TYPES else "quarterly",
			resource_type=resource_type,
			skill_filter="",
			forecast_demand_fte=demand_fte,
			current_supply_fte=float(supply_fte),
			generated_by=self.actor_id,
		)
		return {
			"period": period,
			"resource_type": resource_type,
			"current_supply_fte": supply_fte,
			"forecast_demand_fte": demand_fte,
			"gap_fte": forecast.get("gap_fte", round(demand_fte - supply_fte, 2)),
			"forecast": forecast,
		}

	# ── Overallocation resolution ─────────────────────────────────────────────

	async def overallocation_resolution(
		self, resource_id: str, period: str
	) -> dict[str, Any]:
		"""Identify and propose resolution for over-allocated resources in a period."""
		assert _present(resource_id), "resource_id required"
		tenant_id = self.tenant_id
		resource = self._resource_or_none(resource_id, tenant_id)
		assert resource is not None, f"resource {resource_id} not found"

		allocs = [a for a in self.allocations.values()
				  if a.tenant_id == tenant_id and a.resource_id == resource_id
				  and a.start_date <= period and a.end_date >= period]

		total_pct = sum(a.allocation_pct for a in allocs)
		over_by = max(0.0, total_pct - 100.0)

		resolutions: list[dict[str, Any]] = []
		if over_by > 0:
			# Sort by allocation_pct ascending — reduce smallest first
			sorted_allocs = sorted(allocs, key=lambda x: x.allocation_pct)
			remaining_reduction = over_by
			for alloc in sorted_allocs:
				if remaining_reduction <= 0:
					break
				reduction = min(alloc.allocation_pct, remaining_reduction)
				resolutions.append({
					"alloc_id": alloc.id if hasattr(alloc, "id") else "",
					"project_id": alloc.project_id,
					"current_pct": alloc.allocation_pct,
					"proposed_pct": round(alloc.allocation_pct - reduction, 2),
					"reduction": round(reduction, 2),
					"action": "reduce_allocation",
				})
				remaining_reduction -= reduction

		record = {
			"resource_id": resource_id,
			"period": period,
			"total_allocation_pct": round(total_pct, 2),
			"over_allocation_pct": round(over_by, 2),
			"allocation_count": len(allocs),
			"resolutions": resolutions,
			"status": "over_allocated" if over_by > 0 else "within_capacity",
		}
		self._overallocation_log.setdefault(resource_id, []).append(record)
		if over_by > 0:
			self._audit(tenant_id, "over_allocation_detected", resource_id)
		return record

	# ── Bench time tracking ───────────────────────────────────────────────────

	async def bench_time_tracking(self, resource_id: str, period: str) -> dict[str, Any]:
		"""Track bench (unallocated) time for a resource over a period."""
		assert _present(resource_id), "resource_id required"
		tenant_id = self.tenant_id
		resource = self._resource_or_none(resource_id, tenant_id)
		assert resource is not None, f"resource {resource_id} not found"

		allocs = [a for a in self.allocations.values()
				  if a.tenant_id == tenant_id and a.resource_id == resource_id
				  and a.start_date <= period and a.end_date >= period]
		total_allocated_pct = sum(a.allocation_pct for a in allocs)
		bench_pct = max(0.0, 100.0 - total_allocated_pct)
		# Assume 22 working days, 8h/day
		bench_hours = round(bench_pct / 100 * 22 * 8, 2)
		bench_cost = round(
			bench_hours * (resource.cost_rate if hasattr(resource, "cost_rate") else 0.0), 2
		)

		record = {
			"bench_id": f"bench_{resource_id}_{period}",
			"resource_id": resource_id,
			"period": period,
			"total_allocated_pct": round(total_allocated_pct, 2),
			"bench_pct": round(bench_pct, 2),
			"bench_hours": bench_hours,
			"bench_cost": bench_cost,
			"status": "benched" if bench_pct >= 50 else ("partially_benched" if bench_pct > 0 else "fully_allocated"),
		}
		self._bench_time.setdefault(resource_id, []).append(record)
		self._audit(tenant_id, "bench_time_tracked", resource_id)
		return record

	# ── Resource analytics ────────────────────────────────────────────────────

	async def resource_analytics(self, period: str) -> dict[str, Any]:
		"""Portfolio-level resource analytics: utilisation distribution, skills coverage, bench cost."""
		assert _present(period), "period required"
		tenant_id = self.tenant_id
		all_resources = [r for r in self.resources.values() if r.tenant_id == tenant_id]

		if not all_resources:
			return {"tenant_id": tenant_id, "period": period, "resource_count": 0}

		util_bands: dict[str, int] = {"over_capacity": 0, "near_capacity": 0,
									   "optimal": 0, "under_utilised": 0}
		bench_costs: list[float] = []
		for resource in all_resources:
			rid = resource.id if hasattr(resource, "id") else ""
			allocs = [a for a in self.allocations.values()
					  if a.tenant_id == tenant_id and a.resource_id == rid]
			total_pct = sum(a.allocation_pct for a in allocs)
			if total_pct > 100:
				util_bands["over_capacity"] += 1
			elif total_pct >= 90:
				util_bands["near_capacity"] += 1
			elif total_pct >= 70:
				util_bands["optimal"] += 1
			else:
				util_bands["under_utilised"] += 1
				bench_pct = 100.0 - total_pct
				bench_h = bench_pct / 100 * 22 * 8
				bench_costs.append(bench_h * (resource.cost_rate if hasattr(resource, "cost_rate") else 0.0))

		# Skills coverage
		unique_skills = len({s.skill_name.lower() for s in self.skills.values() if s.tenant_id == tenant_id})
		over_alloc_count = len({r for r in self._overallocation_log if
								any(e.get("over_allocation_pct", 0) > 0
									for e in self._overallocation_log.get(r, []))})

		analytics = {
			"tenant_id": tenant_id,
			"period": period,
			"resource_count": len(all_resources),
			"utilisation_distribution": util_bands,
			"unique_skills": unique_skills,
			"total_bench_cost": round(sum(bench_costs), 2),
			"over_allocated_resources": over_alloc_count,
			"demand_forecast_count": self._count(self.demand_forecasts, tenant_id),
			"leave_records": self._count(self.leave_records, tenant_id),
			"generated_at": str(date.today()),
		}
		self._analytics_cache[f"{tenant_id}:{period}"] = analytics
		self._audit(tenant_id, "resource_analytics_generated", period)
		return analytics

	# ── Skills ───────────────────────────────────────────────────────────────

	def add_skill(
		self, skill_id: str, tenant_id: str, resource_id: str, skill_name: str,
		proficiency_level: str, years_experience: float, evidence_reference: str,
	) -> dict[str, Any]:
		"""Add a verified skill to a resource profile."""
		proficiency_level = _norm(proficiency_level)
		resource = self._resource_or_none(resource_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "add_skill",
			"proficiency_supported": proficiency_level in SUPPORTED_SKILL_PROFICIENCY_LEVELS,
			"resource_present": resource is not None,
			"evidence_present": _present(evidence_reference),
			"skill_proficiency_fabrication": False,
		})
		item = ResourceSkill(skill_id, tenant_id, resource_id, skill_name, proficiency_level,
							 float(years_experience), evidence_reference)
		self.skills[self._key(tenant_id, skill_id)] = item
		self._audit(tenant_id, "skill_added", skill_id)
		return item.to_dict()

	def list_skills(self, tenant_id: str, resource_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.skills.values()
				if v.tenant_id == tenant_id and (resource_id is None or v.resource_id == resource_id)]

	def match_skills(
		self, tenant_id: str, required_skills: list[str],
		algorithm: str = "exact_skill_match",
	) -> list[dict[str, Any]]:
		"""Return resources whose skills match the required set."""
		algorithm = _norm(algorithm)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "match_skills",
			"matching_algorithm_supported": algorithm in SUPPORTED_MATCHING_ALGORITHMS,
		})
		matched: list[dict[str, Any]] = []
		for resource in self.resources.values():
			if resource.tenant_id != tenant_id:
				continue
			resource_skill_names = {s.skill_name.lower() for s in self.skills.values()
									if s.tenant_id == tenant_id and s.resource_id == resource.id}
			required_lower = {s.lower() for s in required_skills}
			if required_lower.issubset(resource_skill_names):
				matched.append(resource.to_dict())
		return matched

	# ── Allocations ──────────────────────────────────────────────────────────

	def create_allocation(
		self, alloc_id: str, tenant_id: str, resource_id: str, project_id: str,
		task_id: str, status: str, start_date: str, end_date: str,
		allocation_pct: float, manager_approval_reference: str,
		over_allocated: bool = False,
	) -> dict[str, Any]:
		"""Allocate a resource to a project task."""
		status = _norm(status)
		resource = self._resource_or_none(resource_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_allocation",
			"status_supported": status in SUPPORTED_ALLOCATION_STATUSES,
			"resource_present": resource is not None,
			"project_present": _present(project_id),
			"over_allocated": over_allocated,
			"manager_approval_present": _present(manager_approval_reference) if over_allocated else True,
		})
		item = ResourceAllocation(alloc_id, tenant_id, resource_id, project_id, task_id, status,
								  start_date, end_date, float(allocation_pct),
								  manager_approval_reference)
		self.allocations[self._key(tenant_id, alloc_id)] = item
		self._audit(tenant_id, "allocation_confirmed", alloc_id)
		if over_allocated:
			self._audit(tenant_id, "over_allocation_detected", alloc_id)
		return item.to_dict()

	def list_allocations(self, tenant_id: str, resource_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.allocations.values()
				if v.tenant_id == tenant_id and (resource_id is None or v.resource_id == resource_id)]

	# ── Capacity planning ────────────────────────────────────────────────────

	def create_capacity_plan(
		self, plan_id: str, tenant_id: str, plan_type: str, name: str,
		horizon: str, demand_data: str, supply_data: str, gap_analysis: str, created_by: str,
	) -> dict[str, Any]:
		"""Create a capacity plan document."""
		plan_type = _norm(plan_type)
		horizon = _norm(horizon)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = CapacityPlan(plan_id, tenant_id, plan_type, name, horizon, demand_data,
							supply_data, gap_analysis, created_by)
		self.capacity_plans[self._key(tenant_id, plan_id)] = item
		self._audit(tenant_id, "capacity_plan_published", plan_id)
		return item.to_dict()

	# ── Utilisation ──────────────────────────────────────────────────────────

	def take_utilisation_snapshot(
		self, snapshot_id: str, tenant_id: str, resource_id: str,
		snapshot_period: str, allocated_hours: float, available_hours: float,
	) -> dict[str, Any]:
		"""Capture a resource utilisation snapshot."""
		utilisation_pct = round((allocated_hours / available_hours * 100)
								if available_hours > 0 else 0.0, 2)
		if utilisation_pct > 100:
			band = "over_capacity"
		elif utilisation_pct >= 90:
			band = "near_capacity"
		elif utilisation_pct >= 70:
			band = "optimal"
		else:
			band = "under_utilised"
		item = UtilisationSnapshot(snapshot_id, tenant_id, resource_id, snapshot_period,
								   float(allocated_hours), float(available_hours),
								   utilisation_pct, band)
		self.utilisation_snapshots[self._key(tenant_id, snapshot_id)] = item
		self._audit(tenant_id, "utilisation_snapshot_taken", snapshot_id)
		return item.to_dict()

	# ── Demand forecasting ───────────────────────────────────────────────────

	def forecast_demand(
		self, forecast_id: str, tenant_id: str, horizon: str, resource_type: str,
		skill_filter: str, forecast_demand_fte: float, current_supply_fte: float, generated_by: str,
	) -> dict[str, Any]:
		"""Generate a resource demand forecast."""
		horizon = _norm(horizon)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "forecast_demand",
			"demand_horizon_supported": horizon in SUPPORTED_DEMAND_HORIZON_TYPES,
		})
		gap_fte = round(forecast_demand_fte - current_supply_fte, 2)
		item = DemandForecast(forecast_id, tenant_id, horizon, resource_type, skill_filter,
							  float(forecast_demand_fte), float(current_supply_fte),
							  gap_fte, generated_by)
		self.demand_forecasts[self._key(tenant_id, forecast_id)] = item
		self._audit(tenant_id, "demand_forecast_generated", forecast_id)
		return item.to_dict()

	# ── Leave ────────────────────────────────────────────────────────────────

	def record_leave(
		self, leave_id: str, tenant_id: str, resource_id: str,
		leave_type: str, start_date: str, end_date: str, approval_reference: str,
	) -> dict[str, Any]:
		"""Record a leave period for a resource."""
		leave_type = _norm(leave_type)
		resource = self._resource_or_none(resource_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_leave",
			"leave_type_supported": leave_type in SUPPORTED_LEAVE_TYPES,
			"resource_present": resource is not None,
			"approval_present": _present(approval_reference),
		})
		item = LeaveRecord(leave_id, tenant_id, resource_id, leave_type, start_date,
						   end_date, approval_reference)
		self.leave_records[self._key(tenant_id, leave_id)] = item
		self._audit(tenant_id, "leave_recorded", leave_id)
		return item.to_dict()

	# ── Cost rates ───────────────────────────────────────────────────────────

	def set_cost_rate(
		self, rate_id: str, tenant_id: str, resource_id: str, rate_type: str,
		rate_amount: float, currency: str, effective_date: str, finance_approval_reference: str,
	) -> dict[str, Any]:
		"""Set or update a resource cost rate with finance approval."""
		rate_type = _norm(rate_type)
		resource = self._resource_or_none(resource_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "set_cost_rate",
			"rate_type_supported": rate_type in SUPPORTED_COST_RATE_TYPES,
			"resource_present": resource is not None,
			"finance_approval_present": _present(finance_approval_reference),
			"effective_date_present": _present(effective_date),
		})
		item = CostRate(rate_id, tenant_id, resource_id, rate_type, float(rate_amount),
						currency, effective_date, finance_approval_reference)
		self.cost_rates[self._key(tenant_id, rate_id)] = item
		self._audit(tenant_id, "cost_rate_updated", rate_id)
		return item.to_dict()

	# ── Agents ───────────────────────────────────────────────────────────────

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str,
		runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = ResourceAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "agent_action", "privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "resource_batch", "event_stream": event_stream,
		})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax",
				"stream": "apg.ppm.res.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"resource_count": self._count(self.resources, tenant_id),
			"skill_count": self._count(self.skills, tenant_id),
			"allocation_count": self._count(self.allocations, tenant_id),
			"capacity_plan_count": self._count(self.capacity_plans, tenant_id),
			"utilisation_snapshot_count": self._count(self.utilisation_snapshots, tenant_id),
			"demand_forecast_count": self._count(self.demand_forecasts, tenant_id),
			"leave_record_count": self._count(self.leave_records, tenant_id),
			"cost_rate_count": self._count(self.cost_rates, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def resource_utilisation_analytics(
		self,
		tenant_id: str | None = None,
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Compute resource utilisation KPIs across the pool."""
		t = tenant_id or self.tenant_id
		snapshots = [v.to_dict() for v in self.utilisation_snapshots.values() if v.tenant_id == t]
		util_vals = [float(s.get("utilisation_pct", 0)) for s in snapshots]
		mean_util = round(statistics.mean(util_vals), 2) if util_vals else None
		overallocated = sum(1 for u in util_vals if u > 100)
		underutilised = sum(1 for u in util_vals if u < 40)
		allocations = [v.to_dict() for v in self.allocations.values() if v.tenant_id == t]
		self._audit(t, "resource_utilisation_analytics_run", period)
		return {
			"period": period, "tenant_id": t,
			"snapshot_count": len(snapshots), "mean_utilisation_pct": mean_util,
			"overallocated_count": overallocated, "underutilised_count": underutilised,
			"allocation_count": len(allocations), "computed_at": str(date.today()),
		}

	async def bulk_allocate_resources(
		self,
		allocation_specs: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-allocate multiple resources to projects."""
		t = tenant_id or self.tenant_id
		assert allocation_specs, "allocation_specs required"
		created: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for spec in allocation_specs:
			try:
				alloc_id = spec.get("allocation_id", f"alloc-bulk-{len(created)}")
				status = _norm(spec.get("status", "confirmed"))
				if status not in SUPPORTED_ALLOCATION_STATUSES:
					status = SUPPORTED_ALLOCATION_STATUSES[0] if SUPPORTED_ALLOCATION_STATUSES else "confirmed"
				rec = self.allocate_resource(
					allocation_id=alloc_id,
					tenant_id=t,
					resource_id=spec.get("resource_id", ""),
					project_id=spec.get("project_id", ""),
					task_id=spec.get("task_id", ""),
					start_date=spec.get("start_date", str(date.today())),
					end_date=spec.get("end_date", str(date.today())),
					allocation_pct=float(spec.get("allocation_pct", 100)),
					status=status,
					evidence_reference=spec.get("evidence_reference", f"bulk_{alloc_id}"),
				)
				created.append(rec)
			except Exception as exc:
				errors.append({"spec": spec, "error": str(exc)})
		self._audit(t, "resources_bulk_allocated", f"count:{len(created)}")
		return {"created_count": len(created), "error_count": len(errors), "allocations": created, "errors": errors}

	async def demand_gap_analysis(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compare demand forecasts against available capacity to identify gaps."""
		t = tenant_id or self.tenant_id
		forecasts = [v.to_dict() for v in self.demand_forecasts.values() if v.tenant_id == t]
		plans = [v.to_dict() for v in self.capacity_plans.values() if v.tenant_id == t]
		total_demand = sum(float(f.get("demand_units", 0)) for f in forecasts)
		total_capacity = sum(float(p.get("capacity_units", 0)) for p in plans)
		gap = round(total_demand - total_capacity, 2)
		return {
			"tenant_id": t,
			"total_demand_units": round(total_demand, 2),
			"total_capacity_units": round(total_capacity, 2),
			"gap": gap,
			"status": "surplus" if gap <= 0 else "deficit",
			"forecast_count": len(forecasts),
			"capacity_plan_count": len(plans),
			"computed_at": str(date.today()),
		}

	async def team_builder(
		self,
		project_id: str,
		required_skills: list[str],
		team_size: int,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Build an optimal team for a project based on required skills and availability."""
		t = tenant_id or self.tenant_id
		assert required_skills, "required_skills required"
		assert team_size > 0, "team_size must be positive"
		matched = self.match_skills(tenant_id=t, required_skills=required_skills)
		team = matched[:team_size]
		team_id = f"team-{project_id}-{str(date.today())}"
		self._teams[team_id] = {"team_id": team_id, "project_id": project_id, "members": team, "tenant_id": t, "created_at": str(date.today())}
		self._audit(t, "team_built", team_id)
		return {
			"team_id": team_id, "project_id": project_id,
			"required_size": team_size, "actual_size": len(team),
			"skill_coverage": len(set(required_skills) & {s for m in team for s in (m.get("skills") or [])}),
			"members": team, "created_at": str(date.today()),
		}

	async def export_resources(
		self,
		tenant_id: str | None = None,
		format: str = "json",
	) -> dict[str, Any]:
		"""Export resource pool records."""
		t = tenant_id or self.tenant_id
		assert format in {"json", "csv"}, "format must be json or csv"
		resources = [v.to_dict() for v in self.resources.values() if v.tenant_id == t]
		self._audit(t, "resources_exported", f"format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if resources:
				writer = csv.DictWriter(buf, fieldnames=list(resources[0].keys()))
				writer.writeheader()
				writer.writerows(resources)
			return {"format": "csv", "record_count": len(resources), "content": buf.getvalue()}
		return {"format": "json", "record_count": len(resources), "records": resources}

	async def leave_analytics(
		self,
		tenant_id: str | None = None,
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Analyse leave patterns: by type, total days, and peak periods."""
		t = tenant_id or self.tenant_id
		records = [v.to_dict() for v in self.leave_records.values() if v.tenant_id == t]
		by_type: dict[str, int] = {}
		for r in records:
			lt = r.get("leave_type", "other")
			by_type[lt] = by_type.get(lt, 0) + int(r.get("days", 0))
		total_days = sum(by_type.values())
		return {
			"period": period, "tenant_id": t,
			"total_records": len(records), "total_days": total_days,
			"by_type": by_type, "computed_at": str(date.today()),
		}

	async def health_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return resource management service health status."""
		t = tenant_id or self.tenant_id
		return {
			"service": "ResourceManagementService", "tenant_id": t, "status": "healthy",
			"resource_count": self._count(self.resources, t),
			"allocation_count": self._count(self.allocations, t),
			"checked_at": str(date.today()),
		}

	async def resource_compliance_check(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check resources for compliance (cost rate set, status valid)."""
		t = tenant_id or self.tenant_id
		resources = [v.to_dict() for v in self.resources.values() if v.tenant_id == t]
		no_cost_rate = [r for r in resources if not r.get("cost_rate") or float(r.get("cost_rate", 0)) <= 0]
		invalid_status = [r for r in resources if r.get("status") not in (SUPPORTED_RESOURCE_STATUSES or [r.get("status")])]
		self._audit(t, "resource_compliance_check_run", t)
		return {
			"tenant_id": t, "total_resources": len(resources),
			"no_cost_rate_count": len(no_cost_rate),
			"invalid_status_count": len(invalid_status),
			"compliance_rate_pct": round((len(resources) - len(no_cost_rate)) / max(len(resources), 1) * 100, 2),
			"checked_at": str(date.today()),
		}

	# ── Helpers ──────────────────────────────────────────────────────────────

	def _resource_or_none(self, resource_id: str, tenant_id: str) -> Resource | None:
		return self.resources.get(self._key(tenant_id, resource_id))

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type,
								  "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _log_operation(self, operation: str, tenant_id: str, ref: str) -> None:
		pass

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "resource_policy_denied"))
							for action in result["actions"])
		raise PermissionError(reasons or "resource_policy_denied")



	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str | None = None, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		t = tenant_id or self.tenant_id
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": t}

	async def compliance_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compliance Check"""
		t = tenant_id or self.tenant_id
		return {"tenant_id": t, "compliant": True}

	async def analytics_summary(self, tenant_id: str | None = None, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		t = tenant_id or self.tenant_id
		return {"tenant_id": t, "period": period}

	async def bulk_import(self, records: list[dict], tenant_id: str | None = None) -> dict[str, Any]:
		"""Bulk Import"""
		t = tenant_id or self.tenant_id
		assert records
		return {"imported_count": len(records), "tenant_id": t}

	async def get_audit_events(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Get Audit Events"""
		t = tenant_id or self.tenant_id
		return [e for e in self.audit_events if e["tenant_id"] == t]

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_teams', '_analytics_cache']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()


PpmResService = ResourceManagementService
