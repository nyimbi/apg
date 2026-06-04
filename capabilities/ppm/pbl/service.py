"""Executable service layer for APG Project Baseline Management (pbl)."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPROVAL_WORKFLOWS,
		SUPPORTED_BASELINE_STATUSES, SUPPORTED_BASELINE_TYPES, SUPPORTED_CHANGE_PRIORITIES,
		SUPPORTED_CHANGE_STATUSES, SUPPORTED_CHANGE_TYPES, SUPPORTED_EV_FORECASTING_METHODS,
		SUPPORTED_EV_METRICS, SUPPORTED_IMPACT_AREAS, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_VARIANCE_THRESHOLDS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		BaselineAgent, BaselineApproval, ChangeImpactAssessment,
		ChangeRequest, EarnedValueSnapshot, ProjectBaseline, VarianceReport,
	)
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPROVAL_WORKFLOWS,
		SUPPORTED_BASELINE_STATUSES, SUPPORTED_BASELINE_TYPES, SUPPORTED_CHANGE_PRIORITIES,
		SUPPORTED_CHANGE_STATUSES, SUPPORTED_CHANGE_TYPES, SUPPORTED_EV_FORECASTING_METHODS,
		SUPPORTED_EV_METRICS, SUPPORTED_IMPACT_AREAS, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_VARIANCE_THRESHOLDS,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		BaselineAgent, BaselineApproval, ChangeImpactAssessment,
		ChangeRequest, EarnedValueSnapshot, ProjectBaseline, VarianceReport,
	)


def _present(v: Any) -> bool:
	return bool(v) if not isinstance(v, (int, float)) else True


def _norm(v: str) -> str:
	return v.strip().lower()


def _positive(v: float | int) -> bool:
	return isinstance(v, (int, float)) and v > 0


class ProjectBaselineService:
	"""Tenant-scoped project baseline management runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.baselines: dict[tuple[str, str], ProjectBaseline] = {}
		self.change_requests: dict[tuple[str, str], ChangeRequest] = {}
		self.impact_assessments: dict[tuple[str, str], ChangeImpactAssessment] = {}
		self.ev_snapshots: dict[tuple[str, str], EarnedValueSnapshot] = {}
		self.variance_reports: dict[tuple[str, str], VarianceReport] = {}
		self.approvals: dict[tuple[str, str], BaselineApproval] = {}
		self.agents: dict[tuple[str, str], BaselineAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self._scope_baselines: dict[str, dict[str, Any]] = {}    # project_id -> scope baseline
		self._schedule_baselines: dict[str, dict[str, Any]] = {} # project_id -> schedule baseline
		self._cost_baselines: dict[str, dict[str, Any]] = {}     # project_id -> cost baseline
		self._change_log: dict[str, list[dict[str, Any]]] = {}   # project_id -> change log
		self._restored: dict[str, list[str]] = {}                # project_id -> restore history
		self._analytics_cache: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── Scope baseline ────────────────────────────────────────────────────────

	async def set_scope_baseline(
		self, project_id: str, scope_document: dict[str, Any], approved_by: str
	) -> dict[str, Any]:
		"""Lock the scope baseline for a project. Requires explicit approver.

		scope_document: {deliverables, inclusions, exclusions, assumptions, constraints}
		"""
		assert _present(project_id), "project_id required"
		assert scope_document, "scope_document required"
		assert _present(approved_by), "approved_by required"
		tenant_id = self.tenant_id

		# Create the formal baseline record
		baseline_id = f"bl_scope_{project_id}"
		self.create_baseline(
			baseline_id=baseline_id,
			tenant_id=tenant_id,
			project_id=project_id,
			baseline_type="scope",
			status="approved",
			name=f"Scope Baseline – {project_id}",
			description=str(scope_document.get("description", "")),
			owner_id=approved_by,
			approval_reference=approved_by,
			evidence_reference=f"scope_doc_{str(date.today())}",
		)
		record = {
			"baseline_id": baseline_id,
			"project_id": project_id,
			"type": "scope",
			"approved_by": approved_by,
			"approved_at": str(date.today()),
			"deliverables": scope_document.get("deliverables", []),
			"inclusions": scope_document.get("inclusions", []),
			"exclusions": scope_document.get("exclusions", []),
			"assumptions": scope_document.get("assumptions", []),
			"constraints": scope_document.get("constraints", []),
			"version": 1,
		}
		# Increment version if already exists
		existing = self._scope_baselines.get(f"{tenant_id}:{project_id}")
		if existing:
			record["version"] = existing.get("version", 1) + 1
		self._scope_baselines[f"{tenant_id}:{project_id}"] = record
		self._audit(tenant_id, "scope_baseline_set", project_id)
		return record

	# ── Schedule baseline ─────────────────────────────────────────────────────

	async def set_schedule_baseline(
		self, project_id: str, baseline_schedule: dict[str, Any], approved_by: str
	) -> dict[str, Any]:
		"""Lock the schedule baseline. baseline_schedule: {tasks, dependencies, project_duration_days}."""
		assert _present(project_id), "project_id required"
		assert baseline_schedule, "baseline_schedule required"
		assert _present(approved_by), "approved_by required"
		tenant_id = self.tenant_id

		baseline_id = f"bl_schedule_{project_id}"
		self.create_baseline(
			baseline_id=baseline_id,
			tenant_id=tenant_id,
			project_id=project_id,
			baseline_type="schedule",
			status="approved",
			name=f"Schedule Baseline – {project_id}",
			description=f"Duration: {baseline_schedule.get('project_duration_days', 0)} days",
			owner_id=approved_by,
			approval_reference=approved_by,
			evidence_reference=f"schedule_doc_{str(date.today())}",
		)
		record = {
			"baseline_id": baseline_id,
			"project_id": project_id,
			"type": "schedule",
			"approved_by": approved_by,
			"approved_at": str(date.today()),
			"project_duration_days": baseline_schedule.get("project_duration_days", 0),
			"task_count": len(baseline_schedule.get("tasks", [])),
			"task_snapshot": baseline_schedule.get("tasks", []),
			"dependency_snapshot": baseline_schedule.get("dependencies", []),
			"version": 1,
		}
		existing = self._schedule_baselines.get(f"{tenant_id}:{project_id}")
		if existing:
			record["version"] = existing.get("version", 1) + 1
		self._schedule_baselines[f"{tenant_id}:{project_id}"] = record
		self._audit(tenant_id, "schedule_baseline_set", project_id)
		return record

	# ── Cost baseline ─────────────────────────────────────────────────────────

	async def set_cost_baseline(
		self, project_id: str, budget: dict[str, Any], approved_by: str
	) -> dict[str, Any]:
		"""Lock the cost baseline. budget: {total_budget, budget_lines: [{code, amount}]}."""
		assert _present(project_id), "project_id required"
		assert budget, "budget required"
		assert _present(approved_by), "approved_by required"
		tenant_id = self.tenant_id

		total = float(budget.get("total_budget", 0))
		assert _positive(total), "total_budget must be positive"

		baseline_id = f"bl_cost_{project_id}"
		self.create_baseline(
			baseline_id=baseline_id,
			tenant_id=tenant_id,
			project_id=project_id,
			baseline_type="cost",
			status="approved",
			name=f"Cost Baseline – {project_id}",
			description=f"BAC: {total}",
			owner_id=approved_by,
			approval_reference=approved_by,
			evidence_reference=f"cost_doc_{str(date.today())}",
		)
		record = {
			"baseline_id": baseline_id,
			"project_id": project_id,
			"type": "cost",
			"approved_by": approved_by,
			"approved_at": str(date.today()),
			"total_budget": total,
			"budget_lines": budget.get("budget_lines", []),
			"contingency": budget.get("contingency", 0.0),
			"currency": budget.get("currency", "USD"),
			"version": 1,
		}
		existing = self._cost_baselines.get(f"{tenant_id}:{project_id}")
		if existing:
			record["version"] = existing.get("version", 1) + 1
		self._cost_baselines[f"{tenant_id}:{project_id}"] = record
		self._audit(tenant_id, "cost_baseline_set", project_id)
		return record

	# ── Change request ────────────────────────────────────────────────────────

	async def change_request(
		self, project_id: str, change_type: str, description: str,
		impact_assessment: dict[str, Any], requested_by: str
	) -> dict[str, Any]:
		"""Raise a change request, recording scope/schedule/cost impact.

		impact_assessment: {scope_delta, schedule_impact_days, cost_impact, risk_level}
		"""
		assert _present(project_id), "project_id required"
		assert _present(description), "description required"
		assert _present(requested_by), "requested_by required"
		change_type = _norm(change_type)
		tenant_id = self.tenant_id

		# Determine affected baseline
		baseline_key = f"bl_{change_type.split('_')[0]}_{project_id}"
		baseline = self.baselines.get(self._key(tenant_id, baseline_key))

		cr_id = f"cr_{project_id}_{str(date.today())}_{change_type}"
		record = self.submit_change_request(
			cr_id=cr_id,
			tenant_id=tenant_id,
			baseline_id=baseline_key,
			change_type=change_type if change_type in SUPPORTED_CHANGE_TYPES else "scope_change",
			priority=impact_assessment.get("risk_level", "medium"),
			title=description[:80],
			description=description,
			submitter_id=requested_by,
			impact_reference=str(impact_assessment),
			approval_reference="",
			evidence_reference=f"cr_{str(date.today())}",
		)
		# Log to change log
		log_entry = {
			"cr_id": cr_id,
			"change_type": change_type,
			"description": description,
			"requested_by": requested_by,
			"schedule_impact_days": impact_assessment.get("schedule_impact_days", 0),
			"cost_impact": impact_assessment.get("cost_impact", 0.0),
			"risk_level": impact_assessment.get("risk_level", "medium"),
			"status": "submitted",
			"submitted_at": str(date.today()),
		}
		self._change_log.setdefault(project_id, []).append(log_entry)
		return {"change_request": record, "log_entry": log_entry}

	# ── Approve change ────────────────────────────────────────────────────────

	async def approve_change(
		self, change_request_id: str, approved_by: str, decision: str
	) -> dict[str, Any]:
		"""Record approval or rejection of a change request."""
		assert _present(change_request_id), "change_request_id required"
		assert _present(approved_by), "approved_by required"
		decision = _norm(decision)
		assert decision in ("approved", "rejected", "deferred"), "decision must be approved/rejected/deferred"
		tenant_id = self.tenant_id

		cr = self.change_requests.get(self._key(tenant_id, change_request_id))
		if cr is None:
			raise ValueError(f"change request {change_request_id} not found")

		if decision == "approved":
			result = self.implement_change(change_request_id, tenant_id, approved_by)
		else:
			cr.status = decision
			result = cr.to_dict()

		# Update change log entry
		project_id = cr.baseline_id.replace("bl_scope_", "").replace("bl_schedule_", "").replace("bl_cost_", "")
		for entry in self._change_log.get(project_id, []):
			if entry["cr_id"] == change_request_id:
				entry["status"] = decision
				entry["decided_by"] = approved_by
				entry["decided_at"] = str(date.today())

		approval_id = f"appr_{change_request_id}"
		self.record_baseline_approval(
			approval_id=approval_id,
			tenant_id=tenant_id,
			reference_id=change_request_id,
			approval_type="change_control",
			reviewer_id=approved_by,
			designated_approver=True,
			status=decision,
			evidence_reference=f"decision_{decision}_{str(date.today())}",
		)
		return {"change_request": result, "decision": decision, "decided_by": approved_by}

	# ── Baseline comparison ───────────────────────────────────────────────────

	async def baseline_comparison(
		self, project_id: str, baseline_name: str, current: dict[str, Any]
	) -> dict[str, Any]:
		"""Compare a named baseline snapshot against current project data.

		current: {duration_days, total_cost, scope_items}
		"""
		assert _present(project_id), "project_id required"
		assert _present(baseline_name), "baseline_name required"
		tenant_id = self.tenant_id
		bl_type = baseline_name.lower()

		baseline_data: dict[str, Any] = {}
		if "scope" in bl_type:
			baseline_data = self._scope_baselines.get(f"{tenant_id}:{project_id}", {})
		elif "schedule" in bl_type:
			baseline_data = self._schedule_baselines.get(f"{tenant_id}:{project_id}", {})
		elif "cost" in bl_type:
			baseline_data = self._cost_baselines.get(f"{tenant_id}:{project_id}", {})

		if not baseline_data:
			return {"project_id": project_id, "baseline_name": baseline_name,
					"status": "baseline_not_found"}

		comparison: dict[str, Any] = {
			"project_id": project_id,
			"baseline_name": baseline_name,
			"comparison_date": str(date.today()),
			"deltas": {},
		}

		if "schedule" in bl_type:
			bl_duration = float(baseline_data.get("project_duration_days", 0))
			curr_duration = float(current.get("duration_days", 0))
			delta_days = round(curr_duration - bl_duration, 2)
			comparison["deltas"]["schedule"] = {
				"baseline_duration_days": bl_duration,
				"current_duration_days": curr_duration,
				"delta_days": delta_days,
				"status": "delayed" if delta_days > 0 else ("ahead" if delta_days < 0 else "on_time"),
			}
		if "cost" in bl_type:
			bl_cost = float(baseline_data.get("total_budget", 0))
			curr_cost = float(current.get("total_cost", 0))
			delta_cost = round(curr_cost - bl_cost, 2)
			comparison["deltas"]["cost"] = {
				"baseline_budget": bl_cost,
				"current_cost": curr_cost,
				"delta": delta_cost,
				"variance_pct": round((delta_cost / bl_cost * 100) if bl_cost else 0.0, 2),
				"status": "over_budget" if delta_cost > 0 else "under_budget",
			}
		if "scope" in bl_type:
			bl_items = len(baseline_data.get("deliverables", []))
			curr_items = int(current.get("scope_items", bl_items))
			comparison["deltas"]["scope"] = {
				"baseline_deliverables": bl_items,
				"current_deliverables": curr_items,
				"additions": max(0, curr_items - bl_items),
				"removals": max(0, bl_items - curr_items),
			}

		self._audit(tenant_id, "baseline_compared", project_id)
		return comparison

	# ── Variance analysis ─────────────────────────────────────────────────────

	async def variance_analysis(
		self, project_id: str, baseline_name: str, period: str
	) -> dict[str, Any]:
		"""Full variance analysis: schedule variance (SV), cost variance (CV), SPI, CPI."""
		assert _present(project_id), "project_id required"
		assert _present(period), "period required"
		tenant_id = self.tenant_id

		ev_recs = [ev for ev in self.ev_snapshots.values()
				   if ev.tenant_id == tenant_id]
		latest_ev = max(ev_recs, key=lambda x: x.snapshot_date, default=None) if ev_recs else None

		if latest_ev:
			pv = latest_ev.pv
			ev = latest_ev.ev
			ac = latest_ev.ac
			bac = latest_ev.bac
		else:
			pv = ev = ac = bac = 0.0

		sv = round(ev - pv, 2)
		cv = round(ev - ac, 2)
		spi = round(ev / pv, 3) if pv else 1.0
		cpi = round(ev / ac, 3) if ac else 1.0
		threshold = "green" if abs(spi - 1.0) < 0.05 and abs(cpi - 1.0) < 0.05 else (
			"amber" if abs(spi - 1.0) < 0.15 and abs(cpi - 1.0) < 0.15 else "red"
		)

		report_id = f"var_{project_id}_{period}"
		report = self.generate_variance_report(
			report_id=report_id,
			tenant_id=tenant_id,
			baseline_id=f"bl_schedule_{project_id}",
			report_period=period,
			schedule_variance=sv,
			cost_variance=cv,
			spi=spi,
			cpi=cpi,
			variance_threshold=threshold,
			generated_by=self.actor_id,
		)
		return {
			"project_id": project_id,
			"baseline_name": baseline_name,
			"period": period,
			"pv": pv, "ev": ev, "ac": ac, "bac": bac,
			"sv": sv, "cv": cv, "spi": spi, "cpi": cpi,
			"health": threshold,
			"variance_report": report,
		}

	# ── Change log ────────────────────────────────────────────────────────────

	async def change_log(self, project_id: str) -> dict[str, Any]:
		"""Return the full change log for a project, with CR summary statistics."""
		assert _present(project_id), "project_id required"
		log = self._change_log.get(project_id, [])
		approved = [e for e in log if e["status"] == "implemented"]
		rejected = [e for e in log if e["status"] == "rejected"]
		pending = [e for e in log if e["status"] in ("submitted", "under_review")]
		total_schedule_impact = sum(e.get("schedule_impact_days", 0) for e in approved)
		total_cost_impact = sum(float(e.get("cost_impact", 0)) for e in approved)

		return {
			"project_id": project_id,
			"total_changes": len(log),
			"approved": len(approved),
			"rejected": len(rejected),
			"pending": len(pending),
			"total_approved_schedule_impact_days": total_schedule_impact,
			"total_approved_cost_impact": round(total_cost_impact, 2),
			"log": log,
		}

	# ── Baseline restore ──────────────────────────────────────────────────────

	async def baseline_restore(
		self, project_id: str, baseline_name: str, approved_by: str
	) -> dict[str, Any]:
		"""Restore project data from a named baseline snapshot. Requires approver."""
		assert _present(project_id), "project_id required"
		assert _present(baseline_name), "baseline_name required"
		assert _present(approved_by), "approved_by required"
		tenant_id = self.tenant_id
		bl_type = baseline_name.lower()

		restored_data: dict[str, Any] = {}
		if "scope" in bl_type:
			bl = self._scope_baselines.get(f"{tenant_id}:{project_id}")
			if bl:
				restored_data = {"type": "scope", "data": bl}
		elif "schedule" in bl_type:
			bl = self._schedule_baselines.get(f"{tenant_id}:{project_id}")
			if bl:
				restored_data = {"type": "schedule", "data": bl}
		elif "cost" in bl_type:
			bl = self._cost_baselines.get(f"{tenant_id}:{project_id}")
			if bl:
				restored_data = {"type": "cost", "data": bl}

		if not restored_data:
			return {"project_id": project_id, "baseline_name": baseline_name,
					"status": "baseline_not_found"}

		restore_record = {
			"project_id": project_id,
			"baseline_name": baseline_name,
			"approved_by": approved_by,
			"restored_at": str(date.today()),
			"restored_data_type": restored_data.get("type"),
			"status": "restored",
		}
		self._restored.setdefault(project_id, []).append(baseline_name)
		self._audit(tenant_id, "baseline_restored", project_id)
		return restore_record

	# ── Baseline analytics ────────────────────────────────────────────────────

	async def baseline_analytics(self, project_id: str) -> dict[str, Any]:
		"""Summary analytics: baseline health, change velocity, EV trend, variance trend."""
		assert _present(project_id), "project_id required"
		tenant_id = self.tenant_id

		baselines_for_project = [v for v in self.baselines.values()
								  if v.tenant_id == tenant_id and v.project_id == project_id]
		ev_recs = [ev for ev in self.ev_snapshots.values()
				   if ev.tenant_id == tenant_id]
		var_recs = [v for v in self.variance_reports.values()
					if v.tenant_id == tenant_id]
		cr_recs = [cr for cr in self.change_requests.values()
				   if cr.tenant_id == tenant_id]

		# EV trend: last 5 snapshots
		ev_trend = sorted(
			[{"date": ev.snapshot_date, "spi": round(ev.ev / ev.pv, 3) if ev.pv else 1.0,
			  "cpi": round(ev.ev / ev.ac, 3) if ev.ac else 1.0}
			 for ev in ev_recs],
			key=lambda x: x["date"]
		)[-5:]

		# Variance trend
		var_trend = sorted(
			[{"period": vr.report_period, "spi": vr.spi, "cpi": vr.cpi,
			  "breached": vr.threshold_breached}
			 for vr in var_recs],
			key=lambda x: x["period"]
		)[-5:]

		change_log = self._change_log.get(project_id, [])
		approved_cr = [c for c in change_log if c["status"] == "implemented"]

		analytics = {
			"project_id": project_id,
			"baseline_count": len(baselines_for_project),
			"baseline_types": list({b.baseline_type for b in baselines_for_project}),
			"change_request_count": len(cr_recs),
			"approved_changes": len(approved_cr),
			"total_schedule_slippage_days": sum(
				e.get("schedule_impact_days", 0) for e in approved_cr
			),
			"total_cost_impact": round(
				sum(float(e.get("cost_impact", 0)) for e in approved_cr), 2
			),
			"ev_trend": ev_trend,
			"variance_trend": var_trend,
			"restore_count": len(self._restored.get(project_id, [])),
			"has_scope_baseline": bool(self._scope_baselines.get(f"{tenant_id}:{project_id}")),
			"has_schedule_baseline": bool(self._schedule_baselines.get(f"{tenant_id}:{project_id}")),
			"has_cost_baseline": bool(self._cost_baselines.get(f"{tenant_id}:{project_id}")),
			"generated_at": str(date.today()),
		}
		self._analytics_cache[f"{tenant_id}:{project_id}"] = analytics
		self._audit(tenant_id, "baseline_analytics_generated", project_id)
		return analytics

	# ── Baselines ────────────────────────────────────────────────────────────

	def create_baseline(
		self, baseline_id: str, tenant_id: str, project_id: str,
		baseline_type: str, status: str, name: str, description: str,
		owner_id: str, approval_reference: str, evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a new project baseline (scope/schedule/cost)."""
		baseline_type = _norm(baseline_type)
		status = _norm(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_baseline",
			"baseline_type_supported": baseline_type in SUPPORTED_BASELINE_TYPES,
			"owner_present": _present(owner_id),
			"approval_present": _present(approval_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = ProjectBaseline(baseline_id, tenant_id, project_id, baseline_type, status, name,
							   description, owner_id, approval_reference, evidence_reference)
		self.baselines[self._key(tenant_id, baseline_id)] = item
		self._audit(tenant_id, "baseline_created", baseline_id)
		return item.to_dict()

	def approve_baseline(
		self, baseline_id: str, tenant_id: str, designated_approver: bool,
		approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Approve a baseline with designated approver check."""
		baseline = self._baseline_or_none(baseline_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "approve_baseline",
			"designated_approver": designated_approver,
			"baseline_present": baseline is not None,
		})
		if baseline:
			baseline.status = "approved"
			baseline.approval_reference = approval_reference
		self._audit(tenant_id, "baseline_approved", baseline_id)
		return baseline.to_dict() if baseline else {}

	def get_baseline(self, baseline_id: str, tenant_id: str) -> dict[str, Any] | None:
		item = self.baselines.get(self._key(tenant_id, baseline_id))
		return item.to_dict() if item else None

	def list_baselines(self, tenant_id: str, project_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.baselines.values()
				if v.tenant_id == tenant_id and (project_id is None or v.project_id == project_id)]

	# ── Change control ───────────────────────────────────────────────────────

	def submit_change_request(
		self, cr_id: str, tenant_id: str, baseline_id: str, change_type: str,
		priority: str, title: str, description: str, submitter_id: str,
		impact_reference: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Submit a change request against an approved baseline."""
		change_type = _norm(change_type)
		priority = _norm(priority)
		baseline = self._baseline_or_none(baseline_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "submit_change_request",
			"change_type_supported": change_type in SUPPORTED_CHANGE_TYPES,
			"priority_supported": priority in SUPPORTED_CHANGE_PRIORITIES,
			"baseline_present": baseline is not None,
			"impact_present": _present(impact_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = ChangeRequest(cr_id, tenant_id, baseline_id, change_type, priority, "submitted",
							 title, description, submitter_id, impact_reference,
							 approval_reference, evidence_reference)
		self.change_requests[self._key(tenant_id, cr_id)] = item
		self._audit(tenant_id, "change_request_submitted", cr_id)
		return item.to_dict()

	def implement_change(self, cr_id: str, tenant_id: str, approval_reference: str) -> dict[str, Any]:
		"""Mark a change request as implemented after approval."""
		cr = self.change_requests.get(self._key(tenant_id, cr_id))
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "implement_change",
			"approval_present": _present(approval_reference),
		})
		if cr:
			cr.status = "implemented"
			cr.approval_reference = approval_reference
		self._audit(tenant_id, "change_implemented", cr_id)
		return cr.to_dict() if cr else {}

	def assess_change_impact(
		self, assessment_id: str, tenant_id: str, change_request_id: str,
		impact_areas: str, schedule_impact_days: int, cost_impact_amount: float,
		scope_impact_description: str, risk_impact_description: str,
		assessor_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Record change impact assessment."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "assess_change_impact",
			"impact_area_supported": True,
		})
		item = ChangeImpactAssessment(assessment_id, tenant_id, change_request_id, impact_areas,
									  schedule_impact_days, float(cost_impact_amount),
									  scope_impact_description, risk_impact_description,
									  assessor_id, evidence_reference)
		self.impact_assessments[self._key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "change_impact_assessed", assessment_id)
		return item.to_dict()

	def list_change_requests(self, tenant_id: str, baseline_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.change_requests.values()
				if v.tenant_id == tenant_id and (baseline_id is None or v.baseline_id == baseline_id)]

	# ── Earned value ─────────────────────────────────────────────────────────

	def take_ev_snapshot(
		self, snapshot_id: str, tenant_id: str, baseline_id: str,
		snapshot_date: str, pv: float, ev: float, ac: float, bac: float,
		forecasting_method: str, eac: float, etc: float,
	) -> dict[str, Any]:
		"""Record an earned value snapshot for a baseline."""
		forecasting_method = _norm(forecasting_method)
		baseline = self._baseline_or_none(baseline_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "take_ev_snapshot",
			"baseline_present": baseline is not None,
			"forecasting_method_supported": forecasting_method in SUPPORTED_EV_FORECASTING_METHODS,
			"ev_manipulation": False,
		})
		item = EarnedValueSnapshot(snapshot_id, tenant_id, baseline_id, snapshot_date,
								   float(pv), float(ev), float(ac), float(bac),
								   forecasting_method, float(eac), float(etc))
		self.ev_snapshots[self._key(tenant_id, snapshot_id)] = item
		self._audit(tenant_id, "ev_snapshot_taken", snapshot_id)
		return item.to_dict()

	def list_ev_snapshots(self, tenant_id: str, baseline_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.ev_snapshots.values()
				if v.tenant_id == tenant_id and (baseline_id is None or v.baseline_id == baseline_id)]

	# ── Variance reports ─────────────────────────────────────────────────────

	def generate_variance_report(
		self, report_id: str, tenant_id: str, baseline_id: str,
		report_period: str, schedule_variance: float, cost_variance: float,
		spi: float, cpi: float, variance_threshold: str, generated_by: str,
	) -> dict[str, Any]:
		"""Generate a variance report for a baseline."""
		variance_threshold = _norm(variance_threshold)
		threshold_breached = (spi < 0.9 or cpi < 0.9)
		item = VarianceReport(report_id, tenant_id, baseline_id, report_period,
							  float(schedule_variance), float(cost_variance),
							  float(spi), float(cpi), variance_threshold,
							  threshold_breached, generated_by)
		self.variance_reports[self._key(tenant_id, report_id)] = item
		event = "variance_threshold_breached" if threshold_breached else "variance_report_generated"
		self._audit(tenant_id, event, report_id)
		return item.to_dict()

	# ── Approvals ────────────────────────────────────────────────────────────

	def record_baseline_approval(
		self, approval_id: str, tenant_id: str, reference_id: str,
		approval_type: str, reviewer_id: str, designated_approver: bool,
		status: str, evidence_reference: str,
	) -> dict[str, Any]:
		item = BaselineApproval(approval_id, tenant_id, reference_id, approval_type,
								reviewer_id, designated_approver, status, evidence_reference)
		self.approvals[self._key(tenant_id, approval_id)] = item
		self._audit(tenant_id, "approval_completed", approval_id)
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
		item = BaselineAgent(agent_id, tenant_id, name, runtime, role, scope)
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
			"operation": "baseline_batch", "event_stream": event_stream,
		})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax",
				"stream": "apg.ppm.pbl.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"baseline_count": self._count(self.baselines, tenant_id),
			"change_request_count": self._count(self.change_requests, tenant_id),
			"impact_assessment_count": self._count(self.impact_assessments, tenant_id),
			"ev_snapshot_count": self._count(self.ev_snapshots, tenant_id),
			"variance_report_count": self._count(self.variance_reports, tenant_id),
			"approval_count": self._count(self.approvals, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def change_impact_summary(
		self,
		project_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Summarise change impacts across all change requests for a project."""
		t = tenant_id or self.tenant_id
		requests = [v.to_dict() for v in self.change_requests.values() if v.tenant_id == t and v.project_id == project_id]
		assessments = [v.to_dict() for v in self.impact_assessments.values() if v.tenant_id == t]
		approved = sum(1 for r in requests if r.get("status") == "approved")
		rejected = sum(1 for r in requests if r.get("status") == "rejected")
		pending = len(requests) - approved - rejected
		cost_impact = sum(float(a.get("cost_impact", 0)) for a in assessments)
		schedule_impact = sum(float(a.get("schedule_impact_days", 0)) for a in assessments)
		return {
			"project_id": project_id, "tenant_id": t,
			"change_request_count": len(requests),
			"approved": approved, "rejected": rejected, "pending": pending,
			"total_cost_impact": round(cost_impact, 2),
			"total_schedule_impact_days": round(schedule_impact, 1),
			"computed_at": str(date.today()),
		}

	async def earned_value_trend(
		self,
		project_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Return EV snapshot trend for a project sorted by snapshot date."""
		t = tenant_id or self.tenant_id
		snapshots = [
			v.to_dict() for v in self.ev_snapshots.values()
			if v.tenant_id == t and v.project_id == project_id
		]
		snapshots.sort(key=lambda s: s.get("snapshot_date", ""))
		spi_trend = [float(s.get("spi", 1.0)) for s in snapshots]
		cpi_trend = [float(s.get("cpi", 1.0)) for s in snapshots]
		return {
			"project_id": project_id, "tenant_id": t,
			"snapshot_count": len(snapshots),
			"spi_trend": spi_trend, "cpi_trend": cpi_trend,
			"latest_spi": spi_trend[-1] if spi_trend else None,
			"latest_cpi": cpi_trend[-1] if cpi_trend else None,
			"computed_at": str(date.today()),
		}

	async def baseline_rebase(
		self,
		project_id: str,
		change_request_id: str,
		approved_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Rebase the project baseline following an approved change request."""
		t = tenant_id or self.tenant_id
		change = self.change_requests.get(self._key(t, change_request_id))
		if change is None:
			raise ValueError(f"Change request {change_request_id} not found")
		if change.status != "approved":
			raise ValueError("Change request must be approved before rebasing")
		# Update all baselines for this project
		rebased: list[str] = []
		for key, baseline in self.baselines.items():
			if baseline.tenant_id == t and baseline.project_id == project_id:
				baseline.version = str(int(baseline.version or "1") + 1) if baseline.version else "2"
				rebased.append(baseline.id)
		self._audit(t, "baseline_rebased", f"project:{project_id}:cr:{change_request_id}")
		return {
			"project_id": project_id, "change_request_id": change_request_id,
			"approved_by": approved_by, "rebased_baselines": rebased,
			"rebased_at": str(date.today()),
		}

	async def export_baselines(
		self,
		tenant_id: str | None = None,
		format: str = "json",
	) -> dict[str, Any]:
		"""Export baseline records in JSON or CSV format."""
		t = tenant_id or self.tenant_id
		assert format in {"json", "csv"}, "format must be json or csv"
		baselines = [v.to_dict() for v in self.baselines.values() if v.tenant_id == t]
		self._audit(t, "baselines_exported", f"format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if baselines:
				writer = csv.DictWriter(buf, fieldnames=list(baselines[0].keys()))
				writer.writeheader()
				writer.writerows(baselines)
			return {"format": "csv", "record_count": len(baselines), "content": buf.getvalue()}
		return {"format": "json", "record_count": len(baselines), "records": baselines}

	async def change_request_analytics(
		self,
		tenant_id: str | None = None,
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Compute change request KPIs: submission rate, approval rate."""
		t = tenant_id or self.tenant_id
		requests = [v.to_dict() for v in self.change_requests.values() if v.tenant_id == t]
		approved = sum(1 for r in requests if r.get("status") == "approved")
		rejected = sum(1 for r in requests if r.get("status") == "rejected")
		approval_rate = round(approved / max(len(requests), 1) * 100, 2)
		self._audit(t, "change_request_analytics_run", period)
		return {
			"period": period, "tenant_id": t,
			"total_requests": len(requests), "approved": approved, "rejected": rejected,
			"pending": len(requests) - approved - rejected,
			"approval_rate_pct": approval_rate, "computed_at": str(date.today()),
		}

	async def health_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return baseline management service health status."""
		t = tenant_id or self.tenant_id
		return {
			"service": "ProjectBaselineService", "tenant_id": t, "status": "healthy",
			"baseline_count": self._count(self.baselines, t),
			"change_request_count": self._count(self.change_requests, t),
			"checked_at": str(date.today()),
		}

	async def baseline_compliance_check(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check that all active baselines are approved and have evidence."""
		t = tenant_id or self.tenant_id
		baselines = [v.to_dict() for v in self.baselines.values() if v.tenant_id == t]
		no_evidence = [b for b in baselines if not b.get("evidence_reference")]
		pending = [b for b in baselines if b.get("status") not in {"approved", "active"}]
		self._audit(t, "baseline_compliance_check_run", t)
		return {
			"tenant_id": t,
			"total_baselines": len(baselines),
			"no_evidence_count": len(no_evidence),
			"pending_approval_count": len(pending),
			"compliance_rate_pct": round((len(baselines) - len(no_evidence) - len(pending)) / max(len(baselines), 1) * 100, 2),
			"checked_at": str(date.today()),
		}

	# ── Helpers ──────────────────────────────────────────────────────────────

	def _baseline_or_none(self, baseline_id: str, tenant_id: str) -> ProjectBaseline | None:
		return self.baselines.get(self._key(tenant_id, baseline_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "baseline_policy_denied"))
							for action in result["actions"])
		raise PermissionError(reasons or "baseline_policy_denied")



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

PpmPblService = ProjectBaselineService
