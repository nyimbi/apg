"""Dependency-light Risk and Compliance Management lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		RCM_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ASSESSMENT_RESULTS,
		SUPPORTED_CONTROL_TYPES,
		SUPPORTED_EXCEPTION_TYPES,
		SUPPORTED_ISSUE_SEVERITIES,
		SUPPORTED_RCM_AGENT_ROLES,
		SUPPORTED_RCM_AGENT_RUNTIMES,
		SUPPORTED_RISK_CATEGORIES,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		RCM_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ASSESSMENT_RESULTS,
		SUPPORTED_CONTROL_TYPES,
		SUPPORTED_EXCEPTION_TYPES,
		SUPPORTED_ISSUE_SEVERITIES,
		SUPPORTED_RCM_AGENT_ROLES,
		SUPPORTED_RCM_AGENT_RUNTIMES,
		SUPPORTED_RISK_CATEGORIES,
		evaluate_capability_rules,
		get_capability_contract,
	)


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class GrcRcmService:
	"""In-memory executable service for the RCM lifecycle packet."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.risks: dict[str, dict[str, Any]] = {}
		self.controls: dict[str, dict[str, Any]] = {}
		self.obligations: dict[str, dict[str, Any]] = {}
		self.assessments: dict[str, dict[str, Any]] = {}
		self.evidence: dict[str, dict[str, Any]] = {}
		self.issues: dict[str, dict[str, Any]] = {}
		self.governance_decisions: dict[str, dict[str, Any]] = {}
		self.exceptions: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

		# Extended stores for new methods
		self._risk_treatments: dict[str, dict[str, Any]] = {}
		self._risk_appetite: dict[str, Any] = {}
		self._emerging_risks: dict[str, dict[str, Any]] = {}
		self._review_schedules: dict[str, dict[str, Any]] = {}
		self._obligations_register: dict[str, dict[str, Any]] = {}
		self._compliance_calendars: dict[str, dict[str, Any]] = {}
		self._regulatory_monitors: dict[str, dict[str, Any]] = {}
		self._audit_plans: dict[str, dict[str, Any]] = {}
		self._audit_engagements: dict[str, dict[str, Any]] = {}
		self._audit_findings: dict[str, dict[str, Any]] = {}
		self._management_responses: dict[str, dict[str, Any]] = {}
		self._kris: dict[str, dict[str, Any]] = {}

	# ── helpers ───────────────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result.get("decision") == "deny":
			effects = result.get("effects") or result.get("actions") or []
			reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
			raise PermissionError(",".join(reasons) or "operation_denied")

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "tenant_context_present": True,
				"operation": operation, "operation_type": "write", "policy_attached": True}

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id, "event_type": event_type, "record_id": record["id"],
			"record_type": record["type"], "status": record["status"],
			"stream": RCM_EVENT_STREAM, "processor": "bytewax", "emitted_at": _now(),
		})

	@staticmethod
	def _risk_level(residual_score: Decimal) -> str:
		if residual_score >= Decimal("0.75"):
			return "critical"
		if residual_score >= Decimal("0.45"):
			return "high"
		if residual_score >= Decimal("0.20"):
			return "medium"
		return "low"

	# ── ORIGINAL METHODS ──────────────────────────────────────────────────────

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_risk(
		self,
		risk_id: str,
		tenant_id: str,
		title: str,
		category: str,
		owner_id: str,
		likelihood: float,
		impact: float,
		reviewed_by: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		residual_score = Decimal(str(likelihood)) * Decimal(str(impact))
		risk_level = self._risk_level(residual_score)
		context = self._base_context(tenant, "register_risk")
		context.update({
			"title_present": bool(title), "owner_present": bool(owner_id),
			"risk_category_supported": category in SUPPORTED_RISK_CATEGORIES,
			"likelihood_in_range": 0 <= likelihood <= 1, "impact_in_range": 0 <= impact <= 1,
			"high_risk": risk_level in {"high", "critical"}, "review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("risk", risk_id), "type": "rcm_risk", "kind": "risk",
			"tenant_id": tenant, "title": title, "category": category, "owner_id": owner_id,
			"likelihood": likelihood, "impact": impact, "residual_score": str(residual_score),
			"risk_level": risk_level, "reviewed_by": reviewed_by,
			"metadata": deepcopy(metadata or {}), "status": "active", "created_at": _now(),
		}
		self.risks[record["id"]] = record
		self._emit(tenant, "risk_registered", record)
		return deepcopy(record)

	def register_control(
		self,
		control_id: str,
		tenant_id: str,
		name: str,
		owner_id: str,
		control_type: str,
		mapped_risk_ids: list[str],
		test_frequency_days: int = 90,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		risks_present = bool(mapped_risk_ids) and all(
			self.risks.get(rid, {}).get("tenant_id") == tenant for rid in mapped_risk_ids)
		context = self._base_context(tenant, "register_control")
		context.update({
			"name_present": bool(name), "owner_present": bool(owner_id),
			"control_type_supported": control_type in SUPPORTED_CONTROL_TYPES,
			"mapped_risk_present": risks_present, "test_frequency_days": test_frequency_days,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("control", control_id), "type": "rcm_control", "kind": "control",
			"tenant_id": tenant, "name": name, "owner_id": owner_id, "control_type": control_type,
			"mapped_risk_ids": list(mapped_risk_ids), "test_frequency_days": test_frequency_days,
			"last_assessment_result": None, "status": "active", "created_at": _now(),
		}
		self.controls[record["id"]] = record
		self._emit(tenant, "control_registered", record)
		return deepcopy(record)

	def register_obligation(
		self,
		obligation_id: str,
		tenant_id: str,
		framework: str,
		requirement: str,
		owner_id: str,
		jurisdiction: str,
		due_date: str,
		mapped_control_ids: list[str],
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		controls_present = bool(mapped_control_ids) and all(
			self.controls.get(cid, {}).get("tenant_id") == tenant for cid in mapped_control_ids)
		context = self._base_context(tenant, "register_obligation")
		context.update({
			"framework_present": bool(framework), "requirement_present": bool(requirement),
			"owner_present": bool(owner_id), "jurisdiction_present": bool(jurisdiction),
			"due_date_present": bool(due_date), "mapped_control_present": controls_present,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("obligation", obligation_id), "type": "rcm_obligation",
			"kind": "obligation", "tenant_id": tenant, "framework": framework,
			"requirement": requirement, "owner_id": owner_id, "jurisdiction": jurisdiction,
			"due_date": due_date, "mapped_control_ids": list(mapped_control_ids),
			"status": "active", "created_at": _now(),
		}
		self.obligations[record["id"]] = record
		self._emit(tenant, "obligation_registered", record)
		return deepcopy(record)

	def assess_control(
		self,
		assessment_id: str,
		tenant_id: str,
		control_id: str,
		assessor_id: str,
		result: str,
		evidence_ids: list[str] | None = None,
		findings: list[str] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		control = self.controls.get(control_id)
		failed = result in {"partially_effective", "ineffective"}
		context = self._base_context(tenant, "assess_control")
		context.update({
			"control_present": bool(control and control["tenant_id"] == tenant),
			"assessor_present": bool(assessor_id),
			"assessment_result_supported": result in SUPPORTED_ASSESSMENT_RESULTS,
			"failed_assessment": failed, "evidence_present": bool(evidence_ids),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("assessment", assessment_id), "type": "rcm_control_assessment",
			"kind": "assessment", "tenant_id": tenant, "control_id": control_id,
			"assessor_id": assessor_id, "result": result,
			"evidence_ids": list(evidence_ids or []), "findings": list(findings or []),
			"status": result, "created_at": _now(),
		}
		self.assessments[record["id"]] = record
		control["last_assessment_result"] = result
		self._emit(tenant, "control_assessed", record)
		return deepcopy(record)

	def collect_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		source: str,
		linked_record_type: str,
		linked_record_id: str,
		encrypted: bool = True,
		retention_days: int = 2555,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		linked = self._linked_record_exists(linked_record_type, linked_record_id, tenant)
		context = self._base_context(tenant, "collect_evidence")
		context.update({
			"source_present": bool(source), "linked_record_present": linked,
			"encrypted": encrypted, "retention_days": retention_days,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("evidence", evidence_id), "type": "rcm_evidence",
			"kind": "evidence", "tenant_id": tenant, "source": source,
			"linked_record_type": linked_record_type, "linked_record_id": linked_record_id,
			"encrypted": encrypted, "retention_days": retention_days,
			"status": "active", "created_at": _now(),
		}
		self.evidence[record["id"]] = record
		self._emit(tenant, "evidence_collected", record)
		return deepcopy(record)

	def open_issue(
		self,
		issue_id: str,
		tenant_id: str,
		title: str,
		severity: str,
		owner_id: str,
		remediation_plan: str,
		linked_assessment_id: str | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "open_issue")
		context.update({
			"title_present": bool(title),
			"issue_severity_supported": severity in SUPPORTED_ISSUE_SEVERITIES,
			"owner_present": bool(owner_id), "remediation_plan_present": bool(remediation_plan),
			"high_severity": severity in {"high", "critical"}, "review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		if linked_assessment_id and self.assessments.get(linked_assessment_id, {}).get("tenant_id") != tenant:
			raise PermissionError("assessment_required")
		record = {
			"id": self._record_id("issue", issue_id), "type": "rcm_issue", "kind": "issue",
			"tenant_id": tenant, "title": title, "severity": severity, "owner_id": owner_id,
			"remediation_plan": remediation_plan, "linked_assessment_id": linked_assessment_id,
			"reviewed_by": reviewed_by, "status": "open", "created_at": _now(),
		}
		self.issues[record["id"]] = record
		self._emit(tenant, "issue_opened", record)
		return deepcopy(record)

	def remediate_issue(self, issue_id: str, tenant_id: str, remediation_evidence_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		issue = self.issues.get(issue_id)
		evidence = self.evidence.get(remediation_evidence_id)
		context = self._base_context(tenant, "remediate_issue")
		context.update({
			"issue_present": bool(issue and issue["tenant_id"] == tenant),
			"remediation_evidence_present": bool(evidence and evidence["tenant_id"] == tenant),
		})
		self._assert_rules(context)
		issue["remediation_evidence_id"] = remediation_evidence_id
		issue["status"] = "remediated"
		issue["remediated_at"] = _now()
		self._emit(tenant, "issue_remediated", issue)
		return deepcopy(issue)

	def record_governance_decision(
		self,
		decision_id: str,
		tenant_id: str,
		title: str,
		approver_id: str,
		rationale: str,
		related_risk_ids: list[str],
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		high_risk = any(self.risks.get(rid, {}).get("risk_level") in {"high", "critical"}
						for rid in related_risk_ids)
		context = self._base_context(tenant, "record_governance_decision")
		context.update({
			"title_present": bool(title), "approver_present": bool(approver_id),
			"rationale_present": bool(rationale), "high_risk": high_risk,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		if not all(self.risks.get(rid, {}).get("tenant_id") == tenant for rid in related_risk_ids):
			raise PermissionError("decision_risk_missing")
		record = {
			"id": self._record_id("decision", decision_id), "type": "rcm_governance_decision",
			"kind": "governance_decision", "tenant_id": tenant, "title": title,
			"approver_id": approver_id, "rationale": rationale,
			"related_risk_ids": list(related_risk_ids), "reviewed_by": reviewed_by,
			"status": "approved", "created_at": _now(),
		}
		self.governance_decisions[record["id"]] = record
		self._emit(tenant, "governance_decision_recorded", record)
		return deepcopy(record)

	def register_exception(
		self,
		exception_id: str,
		tenant_id: str,
		exception_type: str,
		linked_risk_id: str,
		expiration_date: str,
		approved_by: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_exception")
		context.update({
			"exception_type_supported": exception_type in SUPPORTED_EXCEPTION_TYPES,
			"expiration_present": bool(expiration_date), "approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		if self.risks.get(linked_risk_id, {}).get("tenant_id") != tenant:
			raise PermissionError("exception_risk_missing")
		record = {
			"id": self._record_id("exception", exception_id), "type": "rcm_exception",
			"kind": "exception", "tenant_id": tenant, "exception_type": exception_type,
			"linked_risk_id": linked_risk_id, "expiration_date": expiration_date,
			"approved_by": approved_by, "status": "approved", "created_at": _now(),
		}
		self.exceptions[record["id"]] = record
		self._emit(tenant, "exception_registered", record)
		return deepcopy(record)

	def register_rcm_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_rcm_agent")
		context.update({
			"agent_runtime_supported": runtime in SUPPORTED_RCM_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_RCM_AGENT_ROLES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("agent"), "type": "rcm_agent", "kind": "agent",
			"tenant_id": tenant, "name": name, "runtime": runtime, "role": role,
			"scope": scope, "status": "active", "created_at": _now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "rcm_agent_registered", record)
		return deepcopy(record)

	def validate_rcm_agent_action(self, tenant_id: str, agent_id: str, action: str,
								  privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("rcm_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant, "tenant_context_present": True,
			"operation": "rcm_agent_action", "action": action,
			"privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant, "tenant_context_present": True,
			"operation": "rcm_batch", "event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count,
				"processor": "bytewax", "stream": RCM_EVENT_STREAM}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None,
					  status: str = "active") -> dict[str, Any]:
		data = dict(metadata or {})
		return self.register_risk(
			record_id, tenant_id,
			str(data.get("title") or data.get("name") or record_id),
			str(data.get("category") or "operational"),
			str(data.get("owner_id") or "system"),
			float(data.get("likelihood", data.get("probability", 0.2))),
			float(data.get("impact", 0.2)),
			data.get("reviewed_by"),
			{"compatibility_status": status, **data},
		)

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		risks = self.list_records("risks", tenant)
		issues = self.list_records("issues", tenant)
		return {
			"tenant_id": tenant,
			"risk_count": len(risks),
			"high_risk_count": len([r for r in risks if r["risk_level"] in {"high", "critical"}]),
			"control_count": len(self.list_records("controls", tenant)),
			"obligation_count": len(self.list_records("obligations", tenant)),
			"assessment_count": len(self.list_records("assessments", tenant)),
			"evidence_count": len(self.list_records("evidence", tenant)),
			"open_issue_count": len([i for i in issues if i["status"] == "open"]),
			"governance_decision_count": len(self.list_records("governance_decisions", tenant)),
			"exception_count": len(self.list_records("exceptions", tenant)),
			"rcm_agent_count": len(self.list_records("agents", tenant)),
			"audit_event_count": len(self.audit_events(tenant)),
			"overall_status": ("attention_required"
							   if issues or any(r["risk_level"] in {"high", "critical"} for r in risks)
							   else "operating"),
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	def list_records(self, collection: str, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(r) for r in store.values() if r["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(r) for r in store if r["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		for col in ["risks", "controls", "obligations", "assessments", "evidence", "issues",
					"governance_decisions", "exceptions", "agents"]:
			records.extend(self.list_records(col, tenant))
		return sorted(records, key=lambda r: (r["kind"], r["id"]))

	def _linked_record_exists(self, linked_record_type: str, linked_record_id: str, tenant_id: str) -> bool:
		col = {"risk": self.risks, "control": self.controls, "obligation": self.obligations,
			   "assessment": self.assessments, "issue": self.issues}.get(linked_record_type)
		return bool(col and col.get(linked_record_id, {}).get("tenant_id") == tenant_id)

	# ── RISK MANAGEMENT ───────────────────────────────────────────────────────

	async def risk_register_entry(
		self,
		entity_id: str,
		name: str,
		category: str,
		description: str,
		owner: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Add an entry to the risk register with full description."""
		tenant = self._tenant(tenant_id)
		risk_id = self._record_id("risk")
		record = self.register_risk(
			risk_id=risk_id, tenant_id=tenant, title=name, category=category,
			owner_id=owner, likelihood=0.3, impact=0.3,
			metadata={"entity_id": entity_id, "description": description},
		)
		return record

	async def assess_risk(
		self,
		risk_id: str,
		likelihood: float,
		impact: float,
		velocity: float = 0.5,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update likelihood, impact, and velocity for an existing risk."""
		tenant = self._tenant(tenant_id)
		risk = self.risks.get(risk_id)
		if not risk or risk["tenant_id"] != tenant:
			raise KeyError(f"risk_not_found:{risk_id}")
		assert 0.0 <= likelihood <= 1.0 and 0.0 <= impact <= 1.0 and 0.0 <= velocity <= 1.0
		residual = Decimal(str(likelihood)) * Decimal(str(impact))
		risk.update({
			"likelihood": likelihood, "impact": impact, "velocity": velocity,
			"residual_score": str(residual), "risk_level": self._risk_level(residual),
			"assessed_at": _now(),
		})
		self._emit(tenant, "risk_assessed", risk)
		return deepcopy(risk)

	async def risk_heat_map(self, entity_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Build a likelihood × impact heat-map for all risks linked to entity."""
		tenant = self._tenant(tenant_id)
		entity_risks = [r for r in self.risks.values()
						if r["tenant_id"] == tenant and r.get("metadata", {}).get("entity_id") == entity_id]
		grid: dict[str, list[str]] = {
			"critical": [], "high": [], "medium": [], "low": [],
		}
		for r in entity_risks:
			grid[r["risk_level"]].append(r["id"])
		return {"entity_id": entity_id, "heat_map": grid,
				"risk_count": len(entity_risks), "ts": _now()}

	async def risk_treatment(
		self,
		risk_id: str,
		type: str,
		actions: list[str],
		owner: str,
		deadline: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a treatment plan for a risk (accept/mitigate/transfer/avoid)."""
		tenant = self._tenant(tenant_id)
		risk = self.risks.get(risk_id)
		if not risk or risk["tenant_id"] != tenant:
			raise KeyError(f"risk_not_found:{risk_id}")
		treatment_id = self._record_id("treatment")
		treatment = {
			"id": treatment_id, "type": "rcm_risk_treatment", "kind": "risk_treatment",
			"tenant_id": tenant, "risk_id": risk_id, "treatment_type": type,
			"actions": list(actions), "owner": owner, "deadline": deadline,
			"progress": 0, "status": "active", "created_at": _now(),
		}
		self._risk_treatments[treatment_id] = treatment
		risk["treatment_id"] = treatment_id
		self._emit(tenant, "risk_treatment_created", treatment)
		return deepcopy(treatment)

	async def update_risk_treatment(
		self,
		treatment_id: str,
		progress: int,
		notes: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update progress and notes on an existing risk treatment."""
		tenant = self._tenant(tenant_id)
		treatment = self._risk_treatments.get(treatment_id)
		if not treatment or treatment["tenant_id"] != tenant:
			raise KeyError(f"treatment_not_found:{treatment_id}")
		assert 0 <= progress <= 100
		treatment["progress"] = progress
		treatment["notes"] = notes
		treatment["updated_at"] = _now()
		if progress == 100:
			treatment["status"] = "completed"
		self._emit(tenant, "risk_treatment_updated", treatment)
		return deepcopy(treatment)

	async def risk_appetite_check(
		self,
		proposed_action: str,
		score: float,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check if a proposed action's risk score falls within appetite limits."""
		tenant = self._tenant(tenant_id)
		appetite = self._risk_appetite.get(tenant, {"max_acceptable_score": 0.4})
		max_score = float(appetite.get("max_acceptable_score", 0.4))
		within_appetite = score <= max_score
		return {"proposed_action": proposed_action, "score": score,
				"max_acceptable_score": max_score, "within_appetite": within_appetite,
				"recommendation": "proceed" if within_appetite else "escalate_or_reject",
				"tenant_id": tenant}

	async def emerging_risk_register(
		self,
		name: str,
		horizon_months: int,
		tenant_id: str | None = None,
		description: str = "",
		category: str = "emerging",
	) -> dict[str, Any]:
		"""Register an emerging/horizon risk for monitoring."""
		tenant = self._tenant(tenant_id)
		risk_id = self._record_id("emerging_risk")
		horizon_date = (datetime.utcnow() + timedelta(days=horizon_months * 30)).strftime("%Y-%m-%d")
		record: dict[str, Any] = {
			"id": risk_id, "type": "rcm_emerging_risk", "kind": "emerging_risk",
			"tenant_id": tenant, "name": name, "description": description,
			"category": category, "horizon_months": horizon_months,
			"horizon_date": horizon_date, "status": "monitoring", "created_at": _now(),
		}
		self._emerging_risks[risk_id] = record
		self._audit_events.append({
			"tenant_id": tenant, "event_type": "emerging_risk_registered",
			"record_id": risk_id, "record_type": "rcm_emerging_risk",
			"status": "monitoring", "stream": RCM_EVENT_STREAM,
			"processor": "bytewax", "emitted_at": _now(),
		})
		return deepcopy(record)

	async def risk_review_cycle(
		self,
		entity_id: str,
		frequency: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Schedule a recurring risk review cycle for an entity."""
		tenant = self._tenant(tenant_id)
		freq_map = {"monthly": 30, "quarterly": 90, "biannual": 180, "annual": 365}
		days = freq_map.get(frequency, 90)
		next_review = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
		schedule_id = self._record_id("schedule")
		schedule = {
			"id": schedule_id, "entity_id": entity_id, "tenant_id": tenant,
			"frequency": frequency, "interval_days": days, "next_review_date": next_review,
			"status": "active", "created_at": _now(),
		}
		self._review_schedules[schedule_id] = schedule
		return deepcopy(schedule)

	# ── COMPLIANCE ────────────────────────────────────────────────────────────

	async def obligation_register(
		self,
		regulation: str,
		entity_id: str,
		obligations: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register all obligations for a regulation against an entity."""
		tenant = self._tenant(tenant_id)
		registered: list[dict[str, Any]] = []
		for ob in obligations:
			ob_id = self._record_id("ob")
			record: dict[str, Any] = {
				"id": ob_id, "type": "rcm_obligation_register", "kind": "obligation_register",
				"tenant_id": tenant, "regulation": regulation, "entity_id": entity_id,
				"name": ob.get("name", ""), "description": ob.get("description", ""),
				"due_date": ob.get("due_date", ""), "owner": ob.get("owner", "system"),
				"status": "active", "created_at": _now(),
			}
			self._obligations_register[ob_id] = record
			registered.append(deepcopy(record))
		return {"regulation": regulation, "entity_id": entity_id,
				"obligations_registered": len(registered), "obligations": registered}

	async def compliance_gap_analysis(
		self,
		entity_id: str,
		regulation: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Identify compliance gaps between obligations and controls."""
		tenant = self._tenant(tenant_id)
		obligations = [o for o in self._obligations_register.values()
					   if o["tenant_id"] == tenant and o.get("regulation") == regulation
					   and o.get("entity_id") == entity_id]
		controls = self.list_records("controls", tenant)
		mapped_control_ids: set[str] = set()
		for ob in self.obligations.values():
			if ob["tenant_id"] == tenant:
				mapped_control_ids.update(ob.get("mapped_control_ids", []))
		gaps = []
		for ob in obligations:
			matched = [c for c in controls if c["id"] in mapped_control_ids]
			if not matched:
				gaps.append({"obligation_id": ob["id"], "name": ob["name"], "gap": "no_mapped_control"})
		gap_score = 1.0 - (len(gaps) / len(obligations)) if obligations else 1.0
		return {"entity_id": entity_id, "regulation": regulation,
				"obligation_count": len(obligations), "gap_count": len(gaps),
				"compliance_score": round(gap_score, 3), "gaps": gaps, "ts": _now()}

	async def control_assessment(
		self,
		control_id: str,
		rating: str,
		evidence: list[str],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Perform a lightweight control assessment with rating and evidence."""
		tenant = self._tenant(tenant_id)
		control = self.controls.get(control_id)
		if not control or control["tenant_id"] != tenant:
			raise KeyError(f"control_not_found:{control_id}")
		assessment_id = self._record_id("assessment")
		return self.assess_control(
			assessment_id=assessment_id, tenant_id=tenant, control_id=control_id,
			assessor_id=self.user_id or "system", result=rating, evidence_ids=evidence,
		)

	async def compliance_calendar(
		self,
		entity_id: str,
		year: int,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate a compliance calendar of due obligations for the year."""
		tenant = self._tenant(tenant_id)
		cal_id = f"{entity_id}:{year}"
		cal_obligations = [o for o in self.obligations.values()
						   if o["tenant_id"] == tenant and o.get("due_date", "").startswith(str(year))]
		calendar: dict[str, list[dict[str, Any]]] = {}
		for ob in cal_obligations:
			month = ob.get("due_date", "")[:7]
			calendar.setdefault(month, []).append({
				"obligation_id": ob["id"], "framework": ob.get("framework"),
				"requirement": ob.get("requirement"), "due_date": ob.get("due_date"),
			})
		self._compliance_calendars[cal_id] = {"entity_id": entity_id, "year": year,
											   "calendar": calendar, "tenant_id": tenant}
		return {"entity_id": entity_id, "year": year, "months": len(calendar),
				"obligation_count": len(cal_obligations), "calendar": calendar}

	async def compliance_score(
		self,
		entity_id: str,
		framework: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute an overall compliance score for entity against framework."""
		tenant = self._tenant(tenant_id)
		obligations = [o for o in self.obligations.values()
					   if o["tenant_id"] == tenant and o.get("framework") == framework]
		controls = self.list_records("controls", tenant)
		effective_controls = [c for c in controls if c.get("last_assessment_result") == "effective"]
		score = len(effective_controls) / len(controls) if controls else 0.0
		return {"entity_id": entity_id, "framework": framework,
				"total_obligations": len(obligations), "total_controls": len(controls),
				"effective_controls": len(effective_controls),
				"compliance_score": round(score, 3), "grade": self._grade(score), "ts": _now()}

	async def compliance_dashboard(self, entity_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return unified compliance dashboard for an entity."""
		tenant = self._tenant(tenant_id)
		risks = [r for r in self.risks.values() if r["tenant_id"] == tenant]
		controls = self.list_records("controls", tenant)
		issues = [i for i in self.issues.values() if i["tenant_id"] == tenant and i["status"] == "open"]
		exceptions = [e for e in self.exceptions.values() if e["tenant_id"] == tenant]
		return {
			"entity_id": entity_id, "tenant_id": tenant,
			"risk_count": len(risks),
			"critical_risk_count": sum(1 for r in risks if r["risk_level"] == "critical"),
			"control_count": len(controls),
			"effective_control_count": sum(1 for c in controls if c.get("last_assessment_result") == "effective"),
			"open_issue_count": len(issues),
			"active_exception_count": len([e for e in exceptions if e["status"] == "approved"]),
			"overall_health": "red" if issues else ("amber" if any(r["risk_level"] == "critical" for r in risks) else "green"),
			"ts": _now(),
		}

	async def regulatory_change_monitor(
		self,
		jurisdictions: list[str],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register jurisdictions for regulatory change monitoring."""
		tenant = self._tenant(tenant_id)
		monitor_id = self._record_id("monitor")
		monitor = {
			"id": monitor_id, "tenant_id": tenant, "jurisdictions": list(jurisdictions),
			"status": "active", "last_checked": _now(), "created_at": _now(),
			"changes_detected": [],  # populated by external feed integration
		}
		self._regulatory_monitors[monitor_id] = monitor
		return deepcopy(monitor)

	async def compliance_evidence_collect(
		self,
		obligation_id: str,
		evidence: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Attach evidence to a compliance obligation."""
		tenant = self._tenant(tenant_id)
		ev_id = self._record_id("evidence")
		return self.collect_evidence(
			evidence_id=ev_id, tenant_id=tenant, source=str(evidence.get("source", "manual")),
			linked_record_type="obligation", linked_record_id=obligation_id,
			encrypted=bool(evidence.get("encrypted", True)),
			retention_days=int(evidence.get("retention_days", 2555)),
		)

	# ── AUDIT ─────────────────────────────────────────────────────────────────

	async def audit_plan_create(
		self,
		entity_id: str,
		year: int,
		areas: list[str],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create an annual audit plan for an entity."""
		tenant = self._tenant(tenant_id)
		plan_id = self._record_id("audit_plan")
		plan = {
			"id": plan_id, "type": "rcm_audit_plan", "kind": "audit_plan",
			"tenant_id": tenant, "entity_id": entity_id, "year": year, "areas": list(areas),
			"engagement_ids": [], "status": "draft", "created_at": _now(),
		}
		self._audit_plans[plan_id] = plan
		self._audit_events.append({"tenant_id": tenant, "event_type": "audit_plan_created",
								   "record_id": plan_id, "record_type": "rcm_audit_plan",
								   "status": "draft", "stream": RCM_EVENT_STREAM,
								   "processor": "bytewax", "emitted_at": _now()})
		return deepcopy(plan)

	async def audit_engagement(
		self,
		plan_id: str,
		area: str,
		objectives: list[str],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create an audit engagement under a plan."""
		tenant = self._tenant(tenant_id)
		plan = self._audit_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"audit_plan_not_found:{plan_id}")
		engagement_id = self._record_id("engagement")
		engagement = {
			"id": engagement_id, "type": "rcm_audit_engagement", "kind": "audit_engagement",
			"tenant_id": tenant, "plan_id": plan_id, "area": area,
			"objectives": list(objectives), "finding_ids": [], "status": "planned",
			"created_at": _now(),
		}
		self._audit_engagements[engagement_id] = engagement
		plan["engagement_ids"].append(engagement_id)
		return deepcopy(engagement)

	async def audit_finding(
		self,
		engagement_id: str,
		type: str,
		severity: str,
		description: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record an audit finding for an engagement."""
		tenant = self._tenant(tenant_id)
		engagement = self._audit_engagements.get(engagement_id)
		if not engagement or engagement["tenant_id"] != tenant:
			raise KeyError(f"engagement_not_found:{engagement_id}")
		finding_id = self._record_id("finding")
		finding = {
			"id": finding_id, "type": "rcm_audit_finding", "kind": "audit_finding",
			"tenant_id": tenant, "engagement_id": engagement_id,
			"finding_type": type, "severity": severity, "description": description,
			"management_response_id": None, "status": "open", "created_at": _now(),
		}
		self._audit_findings[finding_id] = finding
		engagement["finding_ids"].append(finding_id)
		return deepcopy(finding)

	async def management_response(
		self,
		finding_id: str,
		response: str,
		actions: list[str],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record management response and action plan for an audit finding."""
		tenant = self._tenant(tenant_id)
		finding = self._audit_findings.get(finding_id)
		if not finding or finding["tenant_id"] != tenant:
			raise KeyError(f"finding_not_found:{finding_id}")
		response_id = self._record_id("mgmt_response")
		mr = {
			"id": response_id, "finding_id": finding_id, "tenant_id": tenant,
			"response": response, "actions": list(actions),
			"status": "pending", "created_at": _now(),
		}
		self._management_responses[response_id] = mr
		finding["management_response_id"] = response_id
		finding["status"] = "management_responded"
		return deepcopy(mr)

	async def follow_up_status(self, finding_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return current follow-up status for an audit finding."""
		tenant = self._tenant(tenant_id)
		finding = self._audit_findings.get(finding_id)
		if not finding or finding["tenant_id"] != tenant:
			raise KeyError(f"finding_not_found:{finding_id}")
		mr_id = finding.get("management_response_id")
		mr = self._management_responses.get(mr_id) if mr_id else None
		return {"finding_id": finding_id, "finding_status": finding["status"],
				"severity": finding["severity"],
				"management_response": deepcopy(mr) if mr else None,
				"ts": _now()}

	async def audit_report(self, engagement_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate an audit report for a completed engagement."""
		tenant = self._tenant(tenant_id)
		engagement = self._audit_engagements.get(engagement_id)
		if not engagement or engagement["tenant_id"] != tenant:
			raise KeyError(f"engagement_not_found:{engagement_id}")
		findings = [self._audit_findings[fid] for fid in engagement.get("finding_ids", [])
					if fid in self._audit_findings]
		severity_summary: dict[str, int] = {}
		for f in findings:
			severity_summary[f["severity"]] = severity_summary.get(f["severity"], 0) + 1
		return {
			"engagement_id": engagement_id, "area": engagement["area"],
			"objectives": engagement["objectives"], "finding_count": len(findings),
			"severity_summary": severity_summary,
			"open_findings": sum(1 for f in findings if f["status"] == "open"),
			"responded_findings": sum(1 for f in findings if f.get("management_response_id")),
			"report_generated_at": _now(), "tenant_id": tenant,
		}

	# ── REPORTING ─────────────────────────────────────────────────────────────

	async def board_risk_report(self, entity_id: str, period: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate a board-level risk report."""
		tenant = self._tenant(tenant_id)
		risks = [r for r in self.risks.values() if r["tenant_id"] == tenant]
		controls = self.list_records("controls", tenant)
		issues = [i for i in self.issues.values() if i["tenant_id"] == tenant]
		by_level: dict[str, int] = {}
		for r in risks:
			by_level[r["risk_level"]] = by_level.get(r["risk_level"], 0) + 1
		return {
			"report_type": "board_risk_report", "entity_id": entity_id, "period": period,
			"tenant_id": tenant, "risk_summary": by_level,
			"total_risks": len(risks), "total_controls": len(controls),
			"open_issues": len([i for i in issues if i["status"] == "open"]),
			"generated_at": _now(),
		}

	async def regulatory_submission(
		self,
		entity_id: str,
		report_type: str,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Prepare a regulatory submission package."""
		tenant = self._tenant(tenant_id)
		submission_id = self._record_id("submission")
		risks = [r for r in self.risks.values() if r["tenant_id"] == tenant]
		evidence_count = len([e for e in self.evidence.values() if e["tenant_id"] == tenant])
		return {
			"submission_id": submission_id, "entity_id": entity_id,
			"report_type": report_type, "period": period, "tenant_id": tenant,
			"risk_count": len(risks), "evidence_items": evidence_count,
			"status": "prepared", "prepared_at": _now(),
		}

	async def kri_monitor(self, kri_id: str, current_value: float, tenant_id: str | None = None) -> dict[str, Any]:
		"""Update the current value of a Key Risk Indicator."""
		tenant = self._tenant(tenant_id)
		kri = self._kris.get(kri_id)
		if kri is None:
			kri = {"id": kri_id, "tenant_id": tenant, "history": [], "status": "active"}
			self._kris[kri_id] = kri
		if kri["tenant_id"] != tenant:
			raise PermissionError("kri_tenant_mismatch")
		kri["current_value"] = current_value
		kri["last_updated"] = _now()
		kri["history"].append({"value": current_value, "ts": _now()})
		threshold = float(kri.get("threshold", 0.8))
		breached = current_value > threshold
		return {"kri_id": kri_id, "current_value": current_value, "threshold": threshold,
				"breached": breached, "ts": _now()}

	async def kri_breach_alert(self, kri_id: str, value: float, tenant_id: str | None = None) -> dict[str, Any]:
		"""Raise a KRI breach alert when value exceeds threshold."""
		tenant = self._tenant(tenant_id)
		kri = self._kris.get(kri_id, {})
		threshold = float(kri.get("threshold", 0.8))
		breached = value > threshold
		alert = {
			"alert_id": self._record_id("alert"), "kri_id": kri_id,
			"value": value, "threshold": threshold, "breached": breached,
			"severity": "critical" if value > threshold * 1.5 else "high",
			"tenant_id": tenant, "raised_at": _now(),
		}
		self._audit_events.append({
			"tenant_id": tenant, "event_type": "kri_breach_alert",
			"record_id": alert["alert_id"], "record_type": "rcm_kri_alert",
			"status": "active", "stream": RCM_EVENT_STREAM,
			"processor": "bytewax", "emitted_at": _now(),
		})
		return alert

	async def risk_analytics(self, entity_id: str, period: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return risk trend analytics for an entity."""
		tenant = self._tenant(tenant_id)
		risks = [r for r in self.risks.values() if r["tenant_id"] == tenant
				 and r.get("metadata", {}).get("entity_id") == entity_id]
		avg_likelihood = sum(r["likelihood"] for r in risks) / len(risks) if risks else 0.0
		avg_impact = sum(r["impact"] for r in risks) / len(risks) if risks else 0.0
		avg_residual = avg_likelihood * avg_impact
		by_category: dict[str, int] = {}
		for r in risks:
			by_category[r["category"]] = by_category.get(r["category"], 0) + 1
		return {
			"entity_id": entity_id, "period": period, "tenant_id": tenant,
			"risk_count": len(risks), "avg_likelihood": round(avg_likelihood, 3),
			"avg_impact": round(avg_impact, 3), "avg_residual_score": round(avg_residual, 3),
			"by_category": by_category, "ts": _now(),
		}

	# ── private helpers ───────────────────────────────────────────────────────

	@staticmethod
	def _grade(score: float) -> str:
		if score >= 0.9:
			return "A"
		if score >= 0.75:
			return "B"
		if score >= 0.6:
			return "C"
		if score >= 0.4:
			return "D"
		return "F"


RCMService = GrcRcmService
