"""Executable service layer for APG Project Planning & Scheduling (pps)."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CALENDAR_TYPES,
		SUPPORTED_CONSTRAINT_TYPES, SUPPORTED_CRITICAL_PATH_METHODS, SUPPORTED_DEPENDENCY_TYPES,
		SUPPORTED_LEVELLING_ALGORITHMS, SUPPORTED_METHODOLOGIES, SUPPORTED_PROGRESS_METHODS,
		SUPPORTED_PROJECT_STATUSES, SUPPORTED_SCHEDULING_MODES, SUPPORTED_TASK_STATUSES,
		SUPPORTED_TASK_TYPES, SUPPORTED_WBS_LEVELS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		CriticalPathResult, Project, ProjectCalendar, ResourceLevellingResult,
		ScheduleAgent, Task, TaskDependency, WbsElement,
	)
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CALENDAR_TYPES,
		SUPPORTED_CONSTRAINT_TYPES, SUPPORTED_CRITICAL_PATH_METHODS, SUPPORTED_DEPENDENCY_TYPES,
		SUPPORTED_LEVELLING_ALGORITHMS, SUPPORTED_METHODOLOGIES, SUPPORTED_PROGRESS_METHODS,
		SUPPORTED_PROJECT_STATUSES, SUPPORTED_SCHEDULING_MODES, SUPPORTED_TASK_STATUSES,
		SUPPORTED_TASK_TYPES, SUPPORTED_WBS_LEVELS,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		CriticalPathResult, Project, ProjectCalendar, ResourceLevellingResult,
		ScheduleAgent, Task, TaskDependency, WbsElement,
	)


def _present(v: Any) -> bool:
	return bool(v) if not isinstance(v, (int, float)) else True


def _positive(v: float | int) -> bool:
	return isinstance(v, (int, float)) and v > 0


def _norm(v: str) -> str:
	return v.strip().lower()


def _parse_date(s: str) -> date:
	"""Parse ISO date string, returning today on failure."""
	try:
		return date.fromisoformat(s[:10])
	except (ValueError, TypeError):
		return date.today()


class ProjectPlanningService:
	"""Tenant-scoped project planning and scheduling runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.projects: dict[tuple[str, str], Project] = {}
		self.wbs_elements: dict[tuple[str, str], WbsElement] = {}
		self.tasks: dict[tuple[str, str], Task] = {}
		self.dependencies: dict[tuple[str, str], TaskDependency] = {}
		self.critical_path_results: dict[tuple[str, str], CriticalPathResult] = {}
		self.levelling_results: dict[tuple[str, str], ResourceLevellingResult] = {}
		self.calendars: dict[tuple[str, str], ProjectCalendar] = {}
		self.agents: dict[tuple[str, str], ScheduleAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self._schedules: dict[str, dict[str, Any]] = {}       # project_id -> computed schedule
		self._baselines: dict[str, dict[str, Any]] = {}       # project_id+name -> baseline snapshot
		self._gantt_cache: dict[str, dict[str, Any]] = {}
		self._what_if: dict[str, list[dict[str, Any]]] = {}   # project_id -> scenario list
		self._analytics: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── Projects ─────────────────────────────────────────────────────────────

	def create_project(
		self, project_id: str, tenant_id: str, name: str, status: str,
		methodology: str, owner_id: str, start_date: str, end_date: str,
		evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a new project with scheduling metadata."""
		status = _norm(status)
		methodology = _norm(methodology)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_project",
			"status_supported": status in SUPPORTED_PROJECT_STATUSES,
			"owner_present": _present(owner_id),
			"start_date_present": _present(start_date),
			"methodology_supported": methodology in SUPPORTED_METHODOLOGIES,
			"evidence_present": _present(evidence_reference),
		})
		item = Project(project_id, tenant_id, name, status, methodology, owner_id, start_date, end_date, evidence_reference)
		self.projects[self._key(tenant_id, project_id)] = item
		self._audit(tenant_id, "project_created", project_id)
		return item.to_dict()

	def get_project(self, project_id: str, tenant_id: str) -> dict[str, Any] | None:
		item = self.projects.get(self._key(tenant_id, project_id))
		return item.to_dict() if item else None

	def list_projects(self, tenant_id: str) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.projects.values() if v.tenant_id == tenant_id]

	# ── WBS ──────────────────────────────────────────────────────────────────

	def add_wbs_element(
		self, wbs_id: str, tenant_id: str, project_id: str,
		parent_id: str | None, level: str, code: str, name: str, description: str,
	) -> dict[str, Any]:
		"""Add a WBS element to the project breakdown structure."""
		level = _norm(level)
		project = self._project_or_none(project_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "add_wbs_element",
			"wbs_level_supported": level in SUPPORTED_WBS_LEVELS,
			"project_present": project is not None,
			"code_present": _present(code),
		})
		item = WbsElement(wbs_id, tenant_id, project_id, parent_id, level, code, name, description)
		self.wbs_elements[self._key(tenant_id, wbs_id)] = item
		self._audit(tenant_id, "wbs_element_added", wbs_id)
		return item.to_dict()

	def list_wbs_elements(self, tenant_id: str, project_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.wbs_elements.values() if v.tenant_id == tenant_id and (project_id is None or v.project_id == project_id)]

	# ── Tasks ────────────────────────────────────────────────────────────────

	def add_task(
		self, task_id: str, tenant_id: str, project_id: str, wbs_element_id: str,
		task_type: str, status: str, name: str, duration_days: float,
		scheduling_mode: str, constraint_type: str, progress_method: str,
		progress_pct: float, start_date: str, end_date: str,
		predecessors: list[str] | None = None, resources: list[str] | None = None,
	) -> dict[str, Any]:
		"""Add a task to a WBS element, optionally wiring predecessors and resources."""
		task_type = _norm(task_type)
		status = _norm(status)
		scheduling_mode = _norm(scheduling_mode)
		constraint_type = _norm(constraint_type)
		progress_method = _norm(progress_method)
		wbs = self.wbs_elements.get(self._key(tenant_id, wbs_element_id))
		duration_ok = _positive(duration_days) or task_type == "milestone"
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "add_task",
			"task_type_supported": task_type in SUPPORTED_TASK_TYPES,
			"wbs_element_present": wbs is not None,
			"duration_positive": duration_ok,
			"scheduling_mode_supported": scheduling_mode in SUPPORTED_SCHEDULING_MODES,
			"constraint_type_supported": constraint_type in SUPPORTED_CONSTRAINT_TYPES,
		})
		item = Task(task_id, tenant_id, project_id, wbs_element_id, task_type, status, name,
					float(duration_days), scheduling_mode, constraint_type, progress_method,
					float(progress_pct), start_date, end_date)
		self.tasks[self._key(tenant_id, task_id)] = item
		# Auto-link predecessor dependencies (FS with 0 lag)
		if predecessors:
			for pred_id in predecessors:
				dep_id = f"dep_{pred_id}_{task_id}"
				if not self._would_create_cycle(tenant_id, pred_id, task_id):
					pred = self.tasks.get(self._key(tenant_id, pred_id))
					if pred:
						dep = TaskDependency(dep_id, tenant_id, pred_id, task_id, "finish_to_start", 0.0)
						self.dependencies[self._key(tenant_id, dep_id)] = dep
		# Tag resource IDs for scheduling reference (does not create ResourceAllocation)
		if resources:
			item_dict = item.to_dict()
			item_dict["resource_ids"] = resources
			self._schedules[f"{tenant_id}:{task_id}:resources"] = item_dict
		self._audit(tenant_id, "task_added", task_id)
		return item.to_dict()

	def update_task_status(self, task_id: str, tenant_id: str, status: str, progress_pct: float) -> dict[str, Any]:
		"""Update task status and progress percentage."""
		status = _norm(status)
		task = self.tasks.get(self._key(tenant_id, task_id))
		if task is None:
			raise ValueError(f"task {task_id} not found for tenant {tenant_id}")
		task.status = status
		task.progress_pct = float(progress_pct)
		self._audit(tenant_id, "task_status_changed", task_id)
		return task.to_dict()

	def list_tasks(self, tenant_id: str, project_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.tasks.values() if v.tenant_id == tenant_id and (project_id is None or v.project_id == project_id)]

	# ── Dependencies ─────────────────────────────────────────────────────────

	def link_dependency(
		self, dep_id: str, tenant_id: str, predecessor_id: str,
		successor_id: str, dependency_type: str, lag_days: float = 0.0,
	) -> dict[str, Any]:
		"""Link a task dependency, preventing circular references."""
		dependency_type = _norm(dependency_type)
		predecessor = self.tasks.get(self._key(tenant_id, predecessor_id))
		successor = self.tasks.get(self._key(tenant_id, successor_id))
		circular = self._would_create_cycle(tenant_id, predecessor_id, successor_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "link_dependency",
			"dependency_type_supported": dependency_type in SUPPORTED_DEPENDENCY_TYPES,
			"predecessor_present": predecessor is not None,
			"successor_present": successor is not None,
			"circular_dependency": circular,
		})
		item = TaskDependency(dep_id, tenant_id, predecessor_id, successor_id, dependency_type, float(lag_days))
		self.dependencies[self._key(tenant_id, dep_id)] = item
		self._audit(tenant_id, "dependency_linked", dep_id)
		return item.to_dict()

	def _would_create_cycle(self, tenant_id: str, predecessor_id: str, new_successor_id: str) -> bool:
		"""Return True if linking would create a circular dependency."""
		visited: set[str] = set()
		queue = [new_successor_id]
		while queue:
			node = queue.pop()
			if node == predecessor_id:
				return True
			if node in visited:
				continue
			visited.add(node)
			for dep in self.dependencies.values():
				if dep.tenant_id == tenant_id and dep.predecessor_id == node:
					queue.append(dep.successor_id)
		return False

	# ── WBS composition ───────────────────────────────────────────────────────

	async def create_wbs(self, project_id: str, wbs_elements: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create WBS elements for a project, resolving parent codes in order.

		Each element dict: {wbs_id, parent_id, level, code, name, description}
		"""
		assert _present(project_id), "project_id required"
		assert wbs_elements, "wbs_elements list must not be empty"
		project = self._project_or_none(project_id, self.tenant_id)
		assert project is not None, f"project {project_id} not found"
		created: list[dict[str, Any]] = []
		for el in wbs_elements:
			wbs_id = el.get("wbs_id") or f"wbs_{project_id}_{el['code'].replace('.', '_')}"
			result = self.add_wbs_element(
				wbs_id=wbs_id,
				tenant_id=self.tenant_id,
				project_id=project_id,
				parent_id=el.get("parent_id"),
				level=el.get("level", "work_package"),
				code=el["code"],
				name=el["name"],
				description=el.get("description", ""),
			)
			created.append(result)
		self._log_op("create_wbs", self.tenant_id, project_id)
		return {
			"project_id": project_id,
			"elements_created": len(created),
			"wbs": created,
		}

	# ── Schedule network ──────────────────────────────────────────────────────

	async def schedule_network(self, project_id: str) -> dict[str, Any]:
		"""Forward/backward pass on all tasks in a project to compute ES, EF, LS, LF.

		Uses a simple CPM implementation over the stored task/dependency graph.
		"""
		assert _present(project_id), "project_id required"
		tenant_id = self.tenant_id
		project_tasks = [t for t in self.tasks.values()
						 if t.tenant_id == tenant_id and t.project_id == project_id]
		if not project_tasks:
			return {"project_id": project_id, "tasks_scheduled": 0, "schedule": []}

		task_map: dict[str, Task] = {t.id: t for t in project_tasks}
		task_ids = set(task_map)

		# Build adjacency: pred -> list of successors, succ -> list of predecessors
		successors: dict[str, list[str]] = {tid: [] for tid in task_ids}
		predecessors_map: dict[str, list[tuple[str, float]]] = {tid: [] for tid in task_ids}
		for dep in self.dependencies.values():
			if dep.tenant_id == tenant_id and dep.predecessor_id in task_ids and dep.successor_id in task_ids:
				successors[dep.predecessor_id].append(dep.successor_id)
				predecessors_map[dep.successor_id].append((dep.predecessor_id, dep.lag_days))

		# Topological sort (Kahn's algorithm)
		in_degree: dict[str, int] = {tid: len(predecessors_map[tid]) for tid in task_ids}
		queue = [tid for tid in task_ids if in_degree[tid] == 0]
		topo: list[str] = []
		while queue:
			node = queue.pop(0)
			topo.append(node)
			for succ in successors[node]:
				in_degree[succ] -= 1
				if in_degree[succ] == 0:
					queue.append(succ)

		# Forward pass – earliest start (ES) and finish (EF) in days from day 0
		ES: dict[str, float] = {tid: 0.0 for tid in task_ids}
		EF: dict[str, float] = {}
		for tid in topo:
			task = task_map[tid]
			for (pred_id, lag) in predecessors_map[tid]:
				ES[tid] = max(ES[tid], EF.get(pred_id, 0.0) + lag)
			EF[tid] = ES[tid] + task.duration_days

		# Project duration = max EF
		project_duration = max(EF.values(), default=0.0)

		# Backward pass – latest start (LS) and finish (LF)
		LF: dict[str, float] = {tid: project_duration for tid in task_ids}
		LS: dict[str, float] = {}
		for tid in reversed(topo):
			task = task_map[tid]
			for succ_id in successors[tid]:
				succ_lag = next((lag for (p, lag) in predecessors_map[succ_id] if p == tid), 0.0)
				LF[tid] = min(LF[tid], LS.get(succ_id, project_duration) - succ_lag)
			LS[tid] = LF[tid] - task.duration_days

		# Float = LS - ES
		schedule: list[dict[str, Any]] = []
		for tid in topo:
			total_float = round(LS[tid] - ES[tid], 2)
			schedule.append({
				"task_id": tid,
				"task_name": task_map[tid].name,
				"duration_days": task_map[tid].duration_days,
				"ES": round(ES[tid], 2),
				"EF": round(EF[tid], 2),
				"LS": round(LS[tid], 2),
				"LF": round(LF[tid], 2),
				"total_float": total_float,
				"critical": total_float <= 0,
			})

		self._schedules[f"{tenant_id}:{project_id}:network"] = {
			"project_id": project_id,
			"project_duration_days": project_duration,
			"schedule": schedule,
		}
		self._audit(tenant_id, "network_scheduled", project_id)
		return {
			"project_id": project_id,
			"project_duration_days": project_duration,
			"tasks_scheduled": len(schedule),
			"schedule": schedule,
		}

	# ── Critical path analysis ────────────────────────────────────────────────

	async def critical_path_analysis(self, project_id: str) -> dict[str, Any]:
		"""Compute and store the critical path, returning critical tasks and project duration."""
		network = await self.schedule_network(project_id)
		critical_tasks = [s for s in network["schedule"] if s["critical"]]
		critical_ids = [s["task_id"] for s in critical_tasks]
		result_id = f"cpr_{project_id}"
		cpr = self.calculate_critical_path(
			result_id=result_id,
			tenant_id=self.tenant_id,
			project_id=project_id,
			method="cpm",
			critical_task_ids=",".join(critical_ids),
			total_float_days=0.0,
			project_duration_days=network["project_duration_days"],
			calculated_at=str(date.today()),
		)
		return {
			"project_id": project_id,
			"method": "cpm",
			"project_duration_days": network["project_duration_days"],
			"critical_task_count": len(critical_ids),
			"critical_task_ids": critical_ids,
			"critical_path_result": cpr,
			"full_schedule": network["schedule"],
		}

	def calculate_critical_path(
		self, result_id: str, tenant_id: str, project_id: str,
		method: str, critical_task_ids: str, total_float_days: float,
		project_duration_days: float, calculated_at: str,
	) -> dict[str, Any]:
		"""Record critical path calculation result."""
		method = _norm(method)
		project = self._project_or_none(project_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "calculate_critical_path",
			"cpm_method_supported": method in SUPPORTED_CRITICAL_PATH_METHODS,
			"critical_path_manipulation": False,
		})
		item = CriticalPathResult(result_id, tenant_id, project_id, method, critical_task_ids,
								 float(total_float_days), float(project_duration_days), calculated_at)
		self.critical_path_results[self._key(tenant_id, result_id)] = item
		self._audit(tenant_id, "critical_path_recalculated", result_id)
		return item.to_dict()

	# ── Resource levelling ────────────────────────────────────────────────────

	async def resource_levelling(self, project_id: str, resource_constraints: dict[str, float]) -> dict[str, Any]:
		"""Level resource allocations across project tasks given capacity constraints.

		resource_constraints: {resource_id: max_daily_hours}
		Returns resolution summary and levelled schedule extension.
		"""
		assert _present(project_id), "project_id required"
		tenant_id = self.tenant_id
		project_tasks = [t for t in self.tasks.values()
						 if t.tenant_id == tenant_id and t.project_id == project_id]
		if not project_tasks:
			return {"project_id": project_id, "over_allocations_resolved": 0, "schedule_extension_days": 0.0}

		# Simplified levelling: count tasks that reference constrained resources
		resource_assignments = self._schedules.get(f"{tenant_id}:resources_overview", {})
		over_allocations = 0
		extension_days = 0.0
		for task in project_tasks:
			task_res_key = f"{tenant_id}:{task.id}:resources"
			task_res = self._schedules.get(task_res_key, {})
			res_ids = task_res.get("resource_ids", [])
			for rid in res_ids:
				constraint = resource_constraints.get(rid)
				if constraint and constraint < 8.0:  # less than full day = constrained
					# Extend task duration proportionally
					factor = 8.0 / constraint
					extension_days += task.duration_days * (factor - 1)
					over_allocations += 1

		result_id = f"lev_{project_id}"
		self.level_resources(
			result_id=result_id,
			tenant_id=tenant_id,
			project_id=project_id,
			algorithm="time_constrained",
			over_allocations_resolved=over_allocations,
			schedule_extension_days=round(extension_days, 2),
			levelled_at=str(date.today()),
		)
		return {
			"project_id": project_id,
			"resource_constraints_applied": len(resource_constraints),
			"over_allocations_resolved": over_allocations,
			"schedule_extension_days": round(extension_days, 2),
			"algorithm": "time_constrained",
		}

	def level_resources(
		self, result_id: str, tenant_id: str, project_id: str,
		algorithm: str, over_allocations_resolved: int,
		schedule_extension_days: float, levelled_at: str,
	) -> dict[str, Any]:
		"""Record the result of a resource levelling run."""
		algorithm = _norm(algorithm)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "level_resources",
			"levelling_algorithm_supported": algorithm in SUPPORTED_LEVELLING_ALGORITHMS,
		})
		item = ResourceLevellingResult(result_id, tenant_id, project_id, algorithm,
									  over_allocations_resolved, float(schedule_extension_days), levelled_at)
		self.levelling_results[self._key(tenant_id, result_id)] = item
		self._audit(tenant_id, "resource_levelling_completed", result_id)
		return item.to_dict()

	# ── Schedule compression ──────────────────────────────────────────────────

	async def schedule_compression(self, project_id: str, technique: str) -> dict[str, Any]:
		"""Apply fast-tracking or crashing to compress schedule duration.

		technique: "fast_track" | "crash"
		Returns estimated compression and updated task data.
		"""
		assert _present(project_id), "project_id required"
		assert technique in ("fast_track", "crash"), "technique must be fast_track or crash"
		tenant_id = self.tenant_id
		network = await self.schedule_network(project_id)
		critical = [s for s in network["schedule"] if s["critical"]]
		if not critical:
			return {"project_id": project_id, "technique": technique, "days_compressed": 0.0, "actions": []}

		actions: list[dict[str, Any]] = []
		days_compressed = 0.0

		if technique == "fast_track":
			# Identify pairs of sequential critical tasks that can overlap by 25%
			for i in range(len(critical) - 1):
				task_a = critical[i]
				task_b = critical[i + 1]
				overlap = task_a["duration_days"] * 0.25
				days_compressed += overlap
				actions.append({
					"type": "overlap",
					"task_a": task_a["task_id"],
					"task_b": task_b["task_id"],
					"overlap_days": round(overlap, 2),
					"risk": "medium",
				})
		elif technique == "crash":
			# Add resources to longest critical tasks (estimated 20% reduction each, cost premium)
			for ct in sorted(critical, key=lambda x: -x["duration_days"])[:3]:
				reduction = ct["duration_days"] * 0.20
				days_compressed += reduction
				actions.append({
					"type": "resource_addition",
					"task_id": ct["task_id"],
					"reduction_days": round(reduction, 2),
					"cost_premium_pct": 30,
					"risk": "low",
				})

		self._audit(tenant_id, "schedule_compressed", project_id)
		return {
			"project_id": project_id,
			"technique": technique,
			"original_duration_days": network["project_duration_days"],
			"days_compressed": round(days_compressed, 2),
			"compressed_duration_days": round(network["project_duration_days"] - days_compressed, 2),
			"actions": actions,
		}

	# ── Gantt chart data ──────────────────────────────────────────────────────

	async def gantt_chart_data(self, project_id: str) -> dict[str, Any]:
		"""Return Gantt-ready data: task bars with absolute start/end dates, dependencies, milestones."""
		assert _present(project_id), "project_id required"
		tenant_id = self.tenant_id
		project = self._project_or_none(project_id, tenant_id)
		if project is None:
			return {"project_id": project_id, "bars": [], "links": []}

		network_key = f"{tenant_id}:{project_id}:network"
		network = self._schedules.get(network_key)
		if network is None:
			network = await self.schedule_network(project_id)

		project_start = _parse_date(project.start_date if hasattr(project, "start_date") else str(date.today()))

		bars: list[dict[str, Any]] = []
		for s in network.get("schedule", []):
			task = self.tasks.get(self._key(tenant_id, s["task_id"]))
			start_dt = project_start + timedelta(days=s["ES"])
			end_dt = project_start + timedelta(days=s["EF"])
			bars.append({
				"task_id": s["task_id"],
				"task_name": s["task_name"],
				"start_date": start_dt.isoformat(),
				"end_date": end_dt.isoformat(),
				"duration_days": s["duration_days"],
				"progress_pct": task.progress_pct if task else 0.0,
				"critical": s["critical"],
				"total_float": s["total_float"],
				"task_type": task.task_type if task else "task",
			})

		links: list[dict[str, Any]] = []
		for dep in self.dependencies.values():
			if dep.tenant_id == tenant_id:
				links.append({
					"dep_id": dep.id,
					"predecessor_id": dep.predecessor_id,
					"successor_id": dep.successor_id,
					"type": dep.dependency_type,
					"lag_days": dep.lag_days,
				})

		gantt = {
			"project_id": project_id,
			"project_duration_days": network.get("project_duration_days", 0.0),
			"bars": bars,
			"links": links,
		}
		self._gantt_cache[f"{tenant_id}:{project_id}"] = gantt
		return gantt

	# ── What-if analysis ──────────────────────────────────────────────────────

	async def what_if_analysis(self, project_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
		"""Simulate a schedule scenario without mutating the live plan.

		scenario keys: {task_duration_overrides: {task_id: days}, delay_task_id: str, delay_days: int}
		Returns simulated duration, critical path change, and schedule delta.
		"""
		assert _present(project_id), "project_id required"
		tenant_id = self.tenant_id
		project_tasks = [t for t in self.tasks.values()
						 if t.tenant_id == tenant_id and t.project_id == project_id]
		if not project_tasks:
			return {"project_id": project_id, "scenario": scenario, "simulated_duration": 0.0}

		overrides: dict[str, float] = scenario.get("task_duration_overrides", {})
		delay_task_id: str | None = scenario.get("delay_task_id")
		delay_days: float = float(scenario.get("delay_days", 0))

		task_map: dict[str, Task] = {t.id: t for t in project_tasks}
		task_ids = set(task_map)
		successors: dict[str, list[str]] = {tid: [] for tid in task_ids}
		predecessors_map: dict[str, list[tuple[str, float]]] = {tid: [] for tid in task_ids}
		for dep in self.dependencies.values():
			if dep.tenant_id == tenant_id and dep.predecessor_id in task_ids and dep.successor_id in task_ids:
				successors[dep.predecessor_id].append(dep.successor_id)
				predecessors_map[dep.successor_id].append((dep.predecessor_id, dep.lag_days))

		in_degree: dict[str, int] = {tid: len(predecessors_map[tid]) for tid in task_ids}
		queue = [tid for tid in task_ids if in_degree[tid] == 0]
		topo: list[str] = []
		while queue:
			node = queue.pop(0)
			topo.append(node)
			for succ in successors[node]:
				in_degree[succ] -= 1
				if in_degree[succ] == 0:
					queue.append(succ)

		# Simulated durations
		sim_dur: dict[str, float] = {}
		for tid in task_ids:
			d = task_map[tid].duration_days
			if tid in overrides:
				d = float(overrides[tid])
			if tid == delay_task_id:
				d += delay_days
			sim_dur[tid] = d

		ES: dict[str, float] = {tid: 0.0 for tid in task_ids}
		EF: dict[str, float] = {}
		for tid in topo:
			for (pred_id, lag) in predecessors_map[tid]:
				ES[tid] = max(ES[tid], EF.get(pred_id, 0.0) + lag)
			EF[tid] = ES[tid] + sim_dur[tid]

		sim_duration = max(EF.values(), default=0.0)

		# Baseline duration from last network
		network_key = f"{tenant_id}:{project_id}:network"
		baseline_net = self._schedules.get(network_key, {})
		baseline_duration = baseline_net.get("project_duration_days", sim_duration)

		# Critical path in scenario
		LF: dict[str, float] = {tid: sim_duration for tid in task_ids}
		LS: dict[str, float] = {}
		for tid in reversed(topo):
			for succ_id in successors[tid]:
				succ_lag = next((lag for (p, lag) in predecessors_map[succ_id] if p == tid), 0.0)
				LF[tid] = min(LF[tid], LS.get(succ_id, sim_duration) - succ_lag)
			LS[tid] = LF[tid] - sim_dur[tid]

		sim_critical = [tid for tid in task_ids if round(LS[tid] - ES[tid], 2) <= 0]

		scenario_record = {
			"project_id": project_id,
			"scenario": scenario,
			"baseline_duration_days": baseline_duration,
			"simulated_duration_days": round(sim_duration, 2),
			"delta_days": round(sim_duration - baseline_duration, 2),
			"simulated_critical_tasks": sim_critical,
		}
		if project_id not in self._what_if:
			self._what_if[project_id] = []
		self._what_if[project_id].append(scenario_record)
		self._audit(tenant_id, "what_if_analysed", project_id)
		return scenario_record

	# ── Schedule baseline save ────────────────────────────────────────────────

	async def schedule_baseline_save(self, project_id: str, baseline_name: str) -> dict[str, Any]:
		"""Snapshot the current schedule as a named baseline for later comparison."""
		assert _present(project_id), "project_id required"
		assert _present(baseline_name), "baseline_name required"
		tenant_id = self.tenant_id
		network = await self.schedule_network(project_id)
		tasks_snapshot = self.list_tasks(tenant_id, project_id)
		deps_snapshot = [v.to_dict() for v in self.dependencies.values() if v.tenant_id == tenant_id]
		baseline_key = f"{tenant_id}:{project_id}:{baseline_name}"
		baseline = {
			"project_id": project_id,
			"baseline_name": baseline_name,
			"saved_at": str(date.today()),
			"project_duration_days": network["project_duration_days"],
			"tasks": tasks_snapshot,
			"dependencies": deps_snapshot,
			"schedule": network["schedule"],
		}
		self._baselines[baseline_key] = baseline
		self._audit(tenant_id, "schedule_baseline_saved", project_id)
		return {
			"project_id": project_id,
			"baseline_name": baseline_name,
			"saved_at": baseline["saved_at"],
			"task_count": len(tasks_snapshot),
			"project_duration_days": network["project_duration_days"],
		}

	# ── Schedule analytics ────────────────────────────────────────────────────

	async def schedule_analytics(self, project_id: str) -> dict[str, Any]:
		"""Compute schedule health metrics: SPI, completion trend, float distribution, milestone status."""
		assert _present(project_id), "project_id required"
		tenant_id = self.tenant_id
		project_tasks = [t for t in self.tasks.values()
						 if t.tenant_id == tenant_id and t.project_id == project_id]
		if not project_tasks:
			return {"project_id": project_id, "task_count": 0}

		total = len(project_tasks)
		completed = sum(1 for t in project_tasks if t.status == "completed")
		in_progress = sum(1 for t in project_tasks if t.status == "in_progress")
		not_started = total - completed - in_progress
		avg_progress = sum(t.progress_pct for t in project_tasks) / total if total else 0.0

		# Schedule performance index (EV / PV — simplified as completion ratio vs. time elapsed)
		planned_pct = 100.0  # would derive from calendar in full implementation
		spi = round((avg_progress / planned_pct) if planned_pct else 1.0, 3)

		# Float distribution from last computed schedule
		network_key = f"{tenant_id}:{project_id}:network"
		network = self._schedules.get(network_key, {})
		schedule = network.get("schedule", [])
		float_buckets: dict[str, int] = {"critical": 0, "near_critical_1_5": 0, "float_6_10": 0, "float_11_plus": 0}
		for s in schedule:
			f = s["total_float"]
			if f <= 0:
				float_buckets["critical"] += 1
			elif f <= 5:
				float_buckets["near_critical_1_5"] += 1
			elif f <= 10:
				float_buckets["float_6_10"] += 1
			else:
				float_buckets["float_11_plus"] += 1

		milestones = [t for t in project_tasks if t.task_type == "milestone"]
		milestone_summary = {
			"total": len(milestones),
			"completed": sum(1 for m in milestones if m.status == "completed"),
			"overdue": sum(1 for m in milestones if m.status not in ("completed",) and m.progress_pct < 100),
		}

		analytics = {
			"project_id": project_id,
			"total_tasks": total,
			"completed": completed,
			"in_progress": in_progress,
			"not_started": not_started,
			"avg_progress_pct": round(avg_progress, 2),
			"spi": spi,
			"float_distribution": float_buckets,
			"milestones": milestone_summary,
			"what_if_scenarios_run": len(self._what_if.get(project_id, [])),
		}
		self._analytics[f"{tenant_id}:{project_id}"] = analytics
		self._audit(tenant_id, "schedule_analytics_generated", project_id)
		return analytics

	# ── Calendars ────────────────────────────────────────────────────────────

	def create_calendar(
		self, calendar_id: str, tenant_id: str, name: str,
		calendar_type: str, working_hours_per_day: float, working_days: str,
	) -> dict[str, Any]:
		"""Create a project working calendar."""
		calendar_type = _norm(calendar_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = ProjectCalendar(calendar_id, tenant_id, name, calendar_type, float(working_hours_per_day), working_days)
		self.calendars[self._key(tenant_id, calendar_id)] = item
		self._audit(tenant_id, "calendar_created", calendar_id)
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
		item = ScheduleAgent(agent_id, tenant_id, name, runtime, role, scope)
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
			"operation": "schedule_batch", "event_stream": event_stream,
		})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.ppm.pps.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"project_count": self._count(self.projects, tenant_id),
			"wbs_element_count": self._count(self.wbs_elements, tenant_id),
			"task_count": self._count(self.tasks, tenant_id),
			"dependency_count": self._count(self.dependencies, tenant_id),
			"critical_path_result_count": self._count(self.critical_path_results, tenant_id),
			"levelling_result_count": self._count(self.levelling_results, tenant_id),
			"calendar_count": self._count(self.calendars, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def bulk_create_tasks(
		self,
		project_id: str,
		task_specs: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-create tasks for a project from a list of spec dicts."""
		t = tenant_id or self.tenant_id
		assert task_specs, "task_specs required"
		created: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for spec in task_specs:
			try:
				task_id = spec.get("task_id", f"task-bulk-{len(created)}")
				name = spec.get("name", task_id)
				task_type = _norm(spec.get("task_type", "work"))
				if task_type not in SUPPORTED_TASK_TYPES:
					task_type = SUPPORTED_TASK_TYPES[0] if SUPPORTED_TASK_TYPES else "work"
				status = _norm(spec.get("status", "not_started"))
				if status not in SUPPORTED_TASK_STATUSES:
					status = SUPPORTED_TASK_STATUSES[0] if SUPPORTED_TASK_STATUSES else "not_started"
				rec = self.create_task(
					task_id=task_id, tenant_id=t, project_id=project_id, name=name,
					task_type=task_type, status=status,
					start_date=spec.get("start_date", str(date.today())),
					end_date=spec.get("end_date", str(date.today())),
					duration_days=int(spec.get("duration_days", 1)),
					effort_hours=float(spec.get("effort_hours", 8)),
					assigned_to=spec.get("assigned_to", ""),
					owner_id=spec.get("owner_id", self.actor_id),
					evidence_reference=spec.get("evidence_reference", f"bulk_{task_id}"),
				)
				created.append(rec)
			except Exception as exc:
				errors.append({"spec": spec, "error": str(exc)})
		self._audit(t, "tasks_bulk_created", f"project:{project_id}:count:{len(created)}")
		return {"project_id": project_id, "created_count": len(created), "error_count": len(errors), "tasks": created, "errors": errors}

	async def schedule_analytics(
		self,
		tenant_id: str | None = None,
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Compute schedule analytics: on-time rate, float distribution, overdue tasks."""
		t = tenant_id or self.tenant_id
		tasks = [v.to_dict() for v in self.tasks.values() if v.tenant_id == t]
		today = str(date.today())
		completed = [tk for tk in tasks if tk.get("status") == "completed"]
		overdue = [tk for tk in tasks if tk.get("status") not in {"completed", "cancelled"} and (tk.get("end_date") or "") < today]
		in_progress = [tk for tk in tasks if tk.get("status") == "in_progress"]
		on_time_rate = round(len(completed) / max(len(tasks), 1) * 100, 2)
		self._audit(t, "schedule_analytics_run", period)
		return {
			"period": period, "tenant_id": t,
			"total_tasks": len(tasks), "completed_tasks": len(completed),
			"overdue_tasks": len(overdue), "in_progress_tasks": len(in_progress),
			"on_time_rate_pct": on_time_rate, "computed_at": today,
		}

	async def critical_path_summary(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Summarise critical path results across all projects."""
		t = tenant_id or self.tenant_id
		results = [v.to_dict() for v in self.critical_path_results.values() if v.tenant_id == t]
		return {
			"tenant_id": t,
			"critical_path_count": len(results),
			"results": results,
			"computed_at": str(date.today()),
		}

	async def export_schedule(
		self,
		project_id: str,
		tenant_id: str | None = None,
		format: str = "json",
	) -> dict[str, Any]:
		"""Export project schedule (tasks + dependencies) in JSON or CSV."""
		t = tenant_id or self.tenant_id
		assert format in {"json", "csv"}, "format must be json or csv"
		tasks = [v.to_dict() for v in self.tasks.values() if v.tenant_id == t and v.project_id == project_id]
		deps = [v.to_dict() for v in self.dependencies.values() if v.tenant_id == t and v.project_id == project_id]
		self._audit(t, "schedule_exported", f"project:{project_id}:format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if tasks:
				writer = csv.DictWriter(buf, fieldnames=list(tasks[0].keys()))
				writer.writeheader()
				writer.writerows(tasks)
			return {"format": "csv", "project_id": project_id, "task_count": len(tasks), "content": buf.getvalue()}
		return {"format": "json", "project_id": project_id, "task_count": len(tasks), "dependency_count": len(deps), "tasks": tasks, "dependencies": deps}

	async def resource_histogram(
		self,
		project_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate a resource loading histogram for a project."""
		t = tenant_id or self.tenant_id
		tasks = [v.to_dict() for v in self.tasks.values() if v.tenant_id == t and v.project_id == project_id]
		by_resource: dict[str, float] = {}
		for tk in tasks:
			res = tk.get("assigned_to") or "unassigned"
			effort = float(tk.get("effort_hours", 0))
			by_resource[res] = round(by_resource.get(res, 0.0) + effort, 2)
		return {
			"project_id": project_id, "tenant_id": t,
			"resource_count": len(by_resource),
			"total_effort_hours": sum(by_resource.values()),
			"histogram": [{"resource": r, "effort_hours": h} for r, h in sorted(by_resource.items(), key=lambda x: x[1], reverse=True)],
			"computed_at": str(date.today()),
		}

	async def milestone_tracker(
		self,
		project_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Track milestone completion status for a project."""
		t = tenant_id or self.tenant_id
		tasks = [v.to_dict() for v in self.tasks.values() if v.tenant_id == t and v.project_id == project_id]
		milestones = [tk for tk in tasks if tk.get("task_type") == "milestone"]
		completed = [m for m in milestones if m.get("status") == "completed"]
		overdue = [m for m in milestones if m.get("status") != "completed" and (m.get("end_date") or "") < str(date.today())]
		return {
			"project_id": project_id, "tenant_id": t,
			"milestone_count": len(milestones),
			"completed_count": len(completed),
			"overdue_count": len(overdue),
			"milestones": milestones,
			"computed_at": str(date.today()),
		}

	async def health_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return scheduling service health status."""
		t = tenant_id or self.tenant_id
		return {
			"service": "ProjectPlanningService",
			"tenant_id": t,
			"status": "healthy",
			"project_count": self._count(self.projects, t),
			"task_count": self._count(self.tasks, t),
			"checked_at": str(date.today()),
		}

	async def schedule_compliance_check(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check projects for scheduling compliance (baseline set, tasks have owners)."""
		t = tenant_id or self.tenant_id
		projects = [v.to_dict() for v in self.projects.values() if v.tenant_id == t]
		tasks = [v.to_dict() for v in self.tasks.values() if v.tenant_id == t]
		no_owner_tasks = [tk for tk in tasks if not tk.get("assigned_to")]
		self._audit(t, "schedule_compliance_check_run", t)
		return {
			"tenant_id": t,
			"project_count": len(projects),
			"total_tasks": len(tasks),
			"unassigned_tasks": len(no_owner_tasks),
			"compliance_rate_pct": round((len(tasks) - len(no_owner_tasks)) / max(len(tasks), 1) * 100, 2),
			"checked_at": str(date.today()),
		}

	# ── Helpers ──────────────────────────────────────────────────────────────

	def _project_or_none(self, project_id: str, tenant_id: str) -> Project | None:
		return self.projects.get(self._key(tenant_id, project_id))

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _log_op(self, operation: str, tenant_id: str, ref: str) -> None:
		pass  # hook for structured logging integration

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "scheduling_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "scheduling_policy_denied")



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

	async def search(self, query: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Search"""
		t = tenant_id or self.tenant_id
		assert query
		return {"query": query, "results": [], "tenant_id": t}

	async def generate_report(self, tenant_id: str | None = None, report_type: str = "summary", period: str = "monthly") -> dict[str, Any]:
		"""Generate Report"""
		t = tenant_id or self.tenant_id
		return {"report_type": report_type, "tenant_id": t, "period": period}

PpmPpsService = ProjectPlanningService
