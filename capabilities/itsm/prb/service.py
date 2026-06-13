"""Executable service layer for APG ITSM Problem Management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_FIX_TYPES, SUPPORTED_PROBLEM_STATUSES,
		SUPPORTED_RCA_METHODS, SUPPORTED_WORKAROUND_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import ItKnownError, ItProblem, ItRootCauseAnalysis, ItWorkaround
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_FIX_TYPES, SUPPORTED_PROBLEM_STATUSES,
		SUPPORTED_RCA_METHODS, SUPPORTED_WORKAROUND_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import ItKnownError, ItProblem, ItRootCauseAnalysis, ItWorkaround  # type: ignore

try:
	from uuid6 import uuid7
	def _uuid7() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid
	def _uuid7() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _present(v: Any) -> bool:
	if v is None:
		return False
	if isinstance(v, str):
		return bool(v.strip())
	return True


class ProblemManagementService:
	"""Tenant-scoped Problem Management runtime for APG ITSM."""

	def __init__(self) -> None:
		self._problems: dict[tuple[str, str], ItProblem] = {}
		self._known_errors: dict[tuple[str, str], ItKnownError] = {}
		self._rcas: dict[tuple[str, str], ItRootCauseAnalysis] = {}
		self._workarounds: dict[tuple[str, str], ItWorkaround] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Problem Lifecycle
	# ------------------------------------------------------------------

	def create_problem(
		self,
		tenant_id: str,
		title: str,
		description: str = "",
		*,
		prb_id: str | None = None,
		category: str = "other",
		priority: str = "P3",
		affected_service: str | None = None,
		affected_ci_id: str | None = None,
		linked_incident_ids: list[str] | None = None,
		owner_id: str | None = None,
		team_id: str | None = None,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "create_problem",
			"title_present": _present(title),
		})
		prb = ItProblem(
			id=prb_id or _uuid7(),
			tenant_id=tenant_id,
			title=title,
			description=description,
			category=category,
			priority=priority,
			affected_service=affected_service,
			affected_ci_id=affected_ci_id,
			linked_incident_ids=linked_incident_ids or [],
			owner_id=owner_id,
			team_id=team_id,
			tags=tags or [],
		)
		self._problems[(tenant_id, prb.id)] = prb
		self._audit(tenant_id, "problem_created", prb.id)
		return prb.model_dump()

	def link_incident(self, tenant_id: str, problem_id: str, incident_id: str) -> dict[str, Any]:
		prb = self._get_prb_or_raise(tenant_id, problem_id)
		if incident_id not in prb.linked_incident_ids:
			prb.linked_incident_ids.append(incident_id)
			prb.version += 1
		self._audit(tenant_id, "incident_linked_to_problem", f"{problem_id}/{incident_id}")
		return {"problem_id": problem_id, "linked_incident_ids": list(prb.linked_incident_ids)}

	def update_problem_status(
		self,
		tenant_id: str,
		problem_id: str,
		new_status: str,
		updated_by: str,
		notes: str = "",
	) -> dict[str, Any]:
		assert new_status in SUPPORTED_PROBLEM_STATUSES, f"unsupported status {new_status!r}"
		prb = self._get_prb_or_raise(tenant_id, problem_id)
		prb.status = new_status
		prb.version += 1
		if new_status == "resolved":
			prb.resolved_at = _now()
		elif new_status == "closed":
			prb.closed_at = _now()
		self._audit(tenant_id, "problem_status_updated", problem_id)
		return prb.model_dump()

	def resolve_problem(
		self,
		tenant_id: str,
		problem_id: str,
		resolved_by: str,
		resolution_notes: str,
		fix_type: str,
		change_ticket_id: str | None = None,
	) -> dict[str, Any]:
		assert fix_type in SUPPORTED_FIX_TYPES, f"unsupported fix type {fix_type!r}"
		prb = self._get_prb_or_raise(tenant_id, problem_id)
		prb.status = "resolved"
		prb.resolved_at = _now()
		prb.resolution_notes = resolution_notes
		prb.fix_type = fix_type
		prb.fix_applied_at = _now()
		prb.change_ticket_id = change_ticket_id
		prb.version += 1
		self._audit(tenant_id, "problem_resolved", problem_id)
		return prb.model_dump()

	def get_problem(self, tenant_id: str, problem_id: str) -> dict[str, Any]:
		return self._get_prb_or_raise(tenant_id, problem_id).model_dump()

	def list_problems(
		self,
		tenant_id: str,
		*,
		status: str | None = None,
		priority: str | None = None,
		category: str | None = None,
	) -> list[dict[str, Any]]:
		results: list[dict[str, Any]] = []
		for (tid, _), prb in self._problems.items():
			if tid != tenant_id:
				continue
			if status and prb.status != status:
				continue
			if priority and prb.priority != priority:
				continue
			if category and prb.category != category:
				continue
			results.append(prb.model_dump())
		return results

	# ------------------------------------------------------------------
	# Root Cause Analysis
	# ------------------------------------------------------------------

	def start_rca(
		self,
		tenant_id: str,
		problem_id: str,
		method: str,
		conducted_by: str,
		*,
		rca_id: str | None = None,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "record_rca",
			"method_supported": method in SUPPORTED_RCA_METHODS,
		})
		_ = self._get_prb_or_raise(tenant_id, problem_id)
		rca = ItRootCauseAnalysis(
			id=rca_id or _uuid7(),
			tenant_id=tenant_id,
			problem_id=problem_id,
			method=method,
			conducted_by=conducted_by,
		)
		self._rcas[(tenant_id, rca.id)] = rca
		self._audit(tenant_id, "rca_started", rca.id)
		return rca.model_dump()

	def update_rca(
		self,
		tenant_id: str,
		rca_id: str,
		*,
		why_chain: list[str] | None = None,
		fishbone_causes: dict[str, list[str]] | None = None,
		root_cause: str | None = None,
		contributing_factors: list[str] | None = None,
		recommendations: list[str] | None = None,
		findings: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		rca = self._rcas.get((tenant_id, rca_id))
		if rca is None:
			raise KeyError(f"RCA {rca_id!r} not found")
		if why_chain is not None:
			rca.why_chain = why_chain
		if fishbone_causes is not None:
			rca.fishbone_causes = fishbone_causes
		if root_cause is not None:
			rca.root_cause = root_cause
		if contributing_factors is not None:
			rca.contributing_factors = contributing_factors
		if recommendations is not None:
			rca.recommendations = recommendations
		if findings is not None:
			rca.findings.update(findings)
		self._audit(tenant_id, "rca_updated", rca_id)
		return rca.model_dump()

	def complete_rca(
		self,
		tenant_id: str,
		rca_id: str,
		root_cause: str,
		recommendations: list[str],
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		rca = self._rcas.get((tenant_id, rca_id))
		if rca is None:
			raise KeyError(f"RCA {rca_id!r} not found")
		rca.root_cause = root_cause
		rca.recommendations = recommendations
		rca.status = "approved" if reviewed_by else "under_review"
		rca.completed_at = _now()
		rca.reviewed_by = reviewed_by
		if reviewed_by:
			rca.approved_at = _now()
		# Back-link to problem
		prb = self._problems.get((tenant_id, rca.problem_id))
		if prb:
			prb.rca_id = rca_id
			prb.root_cause_summary = root_cause
			prb.status = "root_cause_identified"
		self._audit(tenant_id, "rca_completed", rca_id)
		return rca.model_dump()

	def get_rca(self, tenant_id: str, rca_id: str) -> dict[str, Any]:
		rca = self._rcas.get((tenant_id, rca_id))
		if rca is None:
			raise KeyError(f"RCA {rca_id!r} not found")
		return rca.model_dump()

	# ------------------------------------------------------------------
	# Known Error Database
	# ------------------------------------------------------------------

	def register_known_error(
		self,
		tenant_id: str,
		problem_id: str,
		title: str,
		description: str,
		workaround: str,
		workaround_type: str,
		*,
		ke_id: str | None = None,
		workaround_steps: list[str] | None = None,
		affected_services: list[str] | None = None,
		affected_ci_ids: list[str] | None = None,
		symptom_description: str = "",
		search_keywords: list[str] | None = None,
		permanent_fix_available: bool = False,
		permanent_fix_eta: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "register_known_error",
			"workaround_present": _present(workaround),
		})
		ke = ItKnownError(
			id=ke_id or _uuid7(),
			tenant_id=tenant_id,
			problem_id=problem_id,
			title=title,
			description=description,
			workaround=workaround,
			workaround_type=workaround_type if workaround_type in SUPPORTED_WORKAROUND_TYPES else "manual",
			workaround_steps=workaround_steps or [],
			affected_services=affected_services or [],
			affected_ci_ids=affected_ci_ids or [],
			symptom_description=symptom_description,
			search_keywords=search_keywords or [],
			permanent_fix_available=permanent_fix_available,
			permanent_fix_eta=permanent_fix_eta,
			created_by=created_by,
		)
		self._known_errors[(tenant_id, ke.id)] = ke
		# Update problem status
		prb = self._problems.get((tenant_id, problem_id))
		if prb:
			prb.known_error_id = ke.id
			prb.status = "known_error"
		self._audit(tenant_id, "known_error_registered", ke.id)
		return ke.model_dump()

	def apply_workaround(
		self,
		tenant_id: str,
		incident_id: str,
		workaround_description: str,
		workaround_type: str,
		applied_by: str,
		*,
		problem_id: str | None = None,
		known_error_id: str | None = None,
		effectiveness: str | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		wa = ItWorkaround(
			tenant_id=tenant_id,
			incident_id=incident_id,
			problem_id=problem_id,
			known_error_id=known_error_id,
			workaround_description=workaround_description,
			workaround_type=workaround_type,
			applied_by=applied_by,
			effectiveness=effectiveness,
			notes=notes,
		)
		self._workarounds[(tenant_id, wa.id)] = wa
		# Increment usage count on KEDB entry
		if known_error_id:
			ke = self._known_errors.get((tenant_id, known_error_id))
			if ke:
				ke.usage_count += 1
		self._audit(tenant_id, "workaround_applied", wa.id)
		return wa.model_dump()

	def search_kedb(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
		"""Search KEDB by title, description, symptom, and keywords."""
		q = query.lower()
		results: list[dict[str, Any]] = []
		for (tid, _), ke in self._known_errors.items():
			if tid != tenant_id or ke.status != "active":
				continue
			haystack = " ".join([
				ke.title, ke.description, ke.symptom_description,
				" ".join(ke.search_keywords), " ".join(ke.affected_services),
			]).lower()
			if q in haystack:
				results.append(ke.model_dump())
		return sorted(results, key=lambda r: -r["usage_count"])

	def list_known_errors(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		results: list[dict[str, Any]] = []
		for (tid, _), ke in self._known_errors.items():
			if tid != tenant_id:
				continue
			if status and ke.status != status:
				continue
			results.append(ke.model_dump())
		return sorted(results, key=lambda r: -r["usage_count"])

	# ------------------------------------------------------------------
	# Analytics
	# ------------------------------------------------------------------

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		by_status: dict[str, int] = {s: 0 for s in SUPPORTED_PROBLEM_STATUSES}
		total = 0
		for (tid, _), prb in self._problems.items():
			if tid != tenant_id:
				continue
			total += 1
			by_status[prb.status] = by_status.get(prb.status, 0) + 1
		return {
			"tenant_id": tenant_id,
			"total_problems": total,
			"by_status": by_status,
			"kedb_entries": sum(1 for (t, _) in self._known_errors if t == tenant_id),
			"rca_count": sum(1 for (t, _) in self._rcas if t == tenant_id),
			"workarounds_applied": sum(1 for (t, _) in self._workarounds if t == tenant_id),
			"as_of": _now(),
		}

	def recurring_problem_report(self, tenant_id: str, threshold: int = 2) -> list[dict[str, Any]]:
		"""Problems with ≥ threshold linked incidents — recurring issue candidates."""
		results: list[dict[str, Any]] = []
		for (tid, _), prb in self._problems.items():
			if tid != tenant_id:
				continue
			if len(prb.linked_incident_ids) >= threshold:
				results.append({
					"problem_id": prb.id,
					"title": prb.title,
					"status": prb.status,
					"incident_count": len(prb.linked_incident_ids),
					"has_known_error": prb.known_error_id is not None,
					"has_rca": prb.rca_id is not None,
				})
		return sorted(results, key=lambda r: -r["incident_count"])

	# ------------------------------------------------------------------
	# Private
	# ------------------------------------------------------------------

	def _get_prb_or_raise(self, tenant_id: str, problem_id: str) -> ItProblem:
		prb = self._problems.get((tenant_id, problem_id))
		if prb is None:
			raise KeyError(f"problem {problem_id!r} not found for tenant {tenant_id!r}")
		return prb

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "ts": _now()})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", "prb_policy_denied") for a in result["actions"])
		raise PermissionError(reasons or "prb_policy_denied")


ItsmPrbService = ProblemManagementService
