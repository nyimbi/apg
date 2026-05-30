"""Dependency-light Risk and Compliance Management lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
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
except ImportError:  # pragma: no cover - supports direct file loading in tests
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

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": RCM_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

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
			"title_present": bool(title),
			"owner_present": bool(owner_id),
			"risk_category_supported": category in SUPPORTED_RISK_CATEGORIES,
			"likelihood_in_range": 0 <= likelihood <= 1,
			"impact_in_range": 0 <= impact <= 1,
			"high_risk": risk_level in {"high", "critical"},
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("risk", risk_id),
			"type": "rcm_risk",
			"kind": "risk",
			"tenant_id": tenant,
			"title": title,
			"category": category,
			"owner_id": owner_id,
			"likelihood": likelihood,
			"impact": impact,
			"residual_score": str(residual_score),
			"risk_level": risk_level,
			"reviewed_by": reviewed_by,
			"metadata": deepcopy(metadata or {}),
			"status": "active",
			"created_at": self._now(),
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
		risks_present = bool(mapped_risk_ids) and all(self.risks.get(risk_id, {}).get("tenant_id") == tenant for risk_id in mapped_risk_ids)
		context = self._base_context(tenant, "register_control")
		context.update({
			"name_present": bool(name),
			"owner_present": bool(owner_id),
			"control_type_supported": control_type in SUPPORTED_CONTROL_TYPES,
			"mapped_risk_present": risks_present,
			"test_frequency_days": test_frequency_days,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("control", control_id),
			"type": "rcm_control",
			"kind": "control",
			"tenant_id": tenant,
			"name": name,
			"owner_id": owner_id,
			"control_type": control_type,
			"mapped_risk_ids": list(mapped_risk_ids),
			"test_frequency_days": test_frequency_days,
			"last_assessment_result": None,
			"status": "active",
			"created_at": self._now(),
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
		controls_present = bool(mapped_control_ids) and all(self.controls.get(control_id, {}).get("tenant_id") == tenant for control_id in mapped_control_ids)
		context = self._base_context(tenant, "register_obligation")
		context.update({
			"framework_present": bool(framework),
			"requirement_present": bool(requirement),
			"owner_present": bool(owner_id),
			"jurisdiction_present": bool(jurisdiction),
			"due_date_present": bool(due_date),
			"mapped_control_present": controls_present,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("obligation", obligation_id),
			"type": "rcm_obligation",
			"kind": "obligation",
			"tenant_id": tenant,
			"framework": framework,
			"requirement": requirement,
			"owner_id": owner_id,
			"jurisdiction": jurisdiction,
			"due_date": due_date,
			"mapped_control_ids": list(mapped_control_ids),
			"status": "active",
			"created_at": self._now(),
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
			"failed_assessment": failed,
			"evidence_present": bool(evidence_ids),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("assessment", assessment_id),
			"type": "rcm_control_assessment",
			"kind": "assessment",
			"tenant_id": tenant,
			"control_id": control_id,
			"assessor_id": assessor_id,
			"result": result,
			"evidence_ids": list(evidence_ids or []),
			"findings": list(findings or []),
			"status": result,
			"created_at": self._now(),
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
			"source_present": bool(source),
			"linked_record_present": linked,
			"encrypted": encrypted,
			"retention_days": retention_days,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("evidence", evidence_id),
			"type": "rcm_evidence",
			"kind": "evidence",
			"tenant_id": tenant,
			"source": source,
			"linked_record_type": linked_record_type,
			"linked_record_id": linked_record_id,
			"encrypted": encrypted,
			"retention_days": retention_days,
			"status": "active",
			"created_at": self._now(),
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
			"owner_present": bool(owner_id),
			"remediation_plan_present": bool(remediation_plan),
			"high_severity": severity in {"high", "critical"},
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		if linked_assessment_id and self.assessments.get(linked_assessment_id, {}).get("tenant_id") != tenant:
			raise PermissionError("assessment_required")
		record = {
			"id": self._record_id("issue", issue_id),
			"type": "rcm_issue",
			"kind": "issue",
			"tenant_id": tenant,
			"title": title,
			"severity": severity,
			"owner_id": owner_id,
			"remediation_plan": remediation_plan,
			"linked_assessment_id": linked_assessment_id,
			"reviewed_by": reviewed_by,
			"status": "open",
			"created_at": self._now(),
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
		issue["remediated_at"] = self._now()
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
		high_risk = any(self.risks.get(risk_id, {}).get("risk_level") in {"high", "critical"} for risk_id in related_risk_ids)
		context = self._base_context(tenant, "record_governance_decision")
		context.update({
			"title_present": bool(title),
			"approver_present": bool(approver_id),
			"rationale_present": bool(rationale),
			"high_risk": high_risk,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		if not all(self.risks.get(risk_id, {}).get("tenant_id") == tenant for risk_id in related_risk_ids):
			raise PermissionError("decision_risk_missing")
		record = {
			"id": self._record_id("decision", decision_id),
			"type": "rcm_governance_decision",
			"kind": "governance_decision",
			"tenant_id": tenant,
			"title": title,
			"approver_id": approver_id,
			"rationale": rationale,
			"related_risk_ids": list(related_risk_ids),
			"reviewed_by": reviewed_by,
			"status": "approved",
			"created_at": self._now(),
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
			"expiration_present": bool(expiration_date),
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		if self.risks.get(linked_risk_id, {}).get("tenant_id") != tenant:
			raise PermissionError("exception_risk_missing")
		record = {
			"id": self._record_id("exception", exception_id),
			"type": "rcm_exception",
			"kind": "exception",
			"tenant_id": tenant,
			"exception_type": exception_type,
			"linked_risk_id": linked_risk_id,
			"expiration_date": expiration_date,
			"approved_by": approved_by,
			"status": "approved",
			"created_at": self._now(),
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
			"id": self._record_id("agent"),
			"type": "rcm_agent",
			"kind": "agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "rcm_agent_registered", record)
		return deepcopy(record)

	def validate_rcm_agent_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("rcm_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "rcm_agent_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "rcm_batch",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": RCM_EVENT_STREAM}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		data = dict(metadata or {})
		return self.register_risk(
			record_id,
			tenant_id,
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
			"high_risk_count": len([risk for risk in risks if risk["risk_level"] in {"high", "critical"}]),
			"control_count": len(self.list_records("controls", tenant)),
			"obligation_count": len(self.list_records("obligations", tenant)),
			"assessment_count": len(self.list_records("assessments", tenant)),
			"evidence_count": len(self.list_records("evidence", tenant)),
			"open_issue_count": len([issue for issue in issues if issue["status"] == "open"]),
			"governance_decision_count": len(self.list_records("governance_decisions", tenant)),
			"exception_count": len(self.list_records("exceptions", tenant)),
			"rcm_agent_count": len(self.list_records("agents", tenant)),
			"audit_event_count": len(self.audit_events(tenant)),
			"overall_status": "attention_required" if issues or any(risk["risk_level"] in {"high", "critical"} for risk in risks) else "operating",
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		for collection in ["risks", "controls", "obligations", "assessments", "evidence", "issues", "governance_decisions", "exceptions", "agents"]:
			records.extend(self.list_records(collection, tenant))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	def _linked_record_exists(self, linked_record_type: str, linked_record_id: str, tenant_id: str) -> bool:
		collection = {
			"risk": self.risks,
			"control": self.controls,
			"obligation": self.obligations,
			"assessment": self.assessments,
			"issue": self.issues,
		}.get(linked_record_type)
		return bool(collection and collection.get(linked_record_id, {}).get("tenant_id") == tenant_id)

	@staticmethod
	def _risk_level(residual_score: Decimal) -> str:
		if residual_score >= Decimal("0.75"):
			return "critical"
		if residual_score >= Decimal("0.45"):
			return "high"
		if residual_score >= Decimal("0.20"):
			return "medium"
		return "low"


RCMService = GrcRcmService
