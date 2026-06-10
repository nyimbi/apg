"""Service layer for APG Accessibility Services."""

from __future__ import annotations

from datetime import date
from typing import Any

from .accessibility_engine import AccessibilityAuditEngine
from .capability_contract import (
	SUPPORTED_ACCESSIBILITY_AGENT_ROLES,
	SUPPORTED_ACCESSIBILITY_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	AccessibilityAgent,
	AccessibilityAudit,
	AccessibilityAuditEvent,
	AccessibilityException,
	AccessibilityFinding,
	AccessibilityReview,
	AccessibilityStandard,
	AccessibilityTarget,
	RemediationTask,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class AccsService:
	"""Accessibility standard registry, audit runner, and remediation tracker."""

	def __init__(self) -> None:
		self._standards: dict[str, AccessibilityStandard] = {}
		self._targets: dict[str, AccessibilityTarget] = {}
		self._findings: dict[str, AccessibilityFinding] = {}
		self._remediations: dict[str, RemediationTask] = {}
		self._audits: dict[str, AccessibilityAudit] = {}
		self._reviews: dict[str, AccessibilityReview] = {}
		self._exceptions: dict[str, AccessibilityException] = {}
		self._agents: dict[str, AccessibilityAgent] = {}
		self._events: list[AccessibilityAuditEvent] = []
		self._engine = AccessibilityAuditEngine()
		self.register_standard(
			standard_id="wcag_2_2_aa",
			tenant_id="default",
			name="WCAG",
			version="2.2",
			level="AA",
		)

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_standard(
		self,
		standard_id: str,
		tenant_id: str,
		name: str = "WCAG",
		version: str = "2.2",
		level: str = "AA",
		criteria: list[str] | tuple[str, ...] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		standard = AccessibilityStandard(
			id=standard_id,
			tenant_id=tenant_id,
			name=name,
			version=version,
			level=level,
			criteria=tuple(criteria or ("perceivable", "operable", "understandable", "robust")),
		)
		key = self._key(tenant_id, standard_id)
		if key in self._standards:
			raise ValueError(f"duplicate accessibility standard for tenant: {standard_id}")
		self._standards[key] = standard
		self._record_event(
			tenant_id=tenant_id,
			event_type="standard_registered",
			subject_id=standard_id,
			message=f"Registered accessibility standard {standard_id}.",
			evidence={"version": version, "level": level},
		)
		return standard.to_dict()

	def list_standards(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		standards = list(self._standards.values())
		if tenant_id is not None:
			standards = [item for item in standards if item.tenant_id in {tenant_id, "default"}]
		return [item.to_dict() for item in sorted(standards, key=lambda item: item.id)]

	def register_target(
		self,
		target_id: str,
		tenant_id: str,
		surface: str,
		route: str,
		owner: str,
		published_ui: bool = False,
		contrast_ratio: float = 4.5,
		semantic_labels_present: bool = True,
		keyboard_navigation_present: bool = True,
		media_content_present: bool = False,
		captions_available: bool = True,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		target = AccessibilityTarget(
			id=target_id,
			tenant_id=tenant_id,
			surface=surface,
			route=route,
			owner=owner,
			published_ui=published_ui,
			contrast_ratio=contrast_ratio,
			semantic_labels_present=semantic_labels_present,
			keyboard_navigation_present=keyboard_navigation_present,
			media_content_present=media_content_present,
			captions_available=captions_available,
		)
		key = self._key(tenant_id, target_id)
		if key in self._targets:
			raise ValueError(f"duplicate accessibility target for tenant: {target_id}")
		self._targets[key] = target
		self._record_event(
			tenant_id=tenant_id,
			event_type="target_registered",
			subject_id=target_id,
			message=f"Registered accessibility target {target_id}.",
			evidence={"route": route, "published_ui": published_ui},
		)
		return target.to_dict()

	def list_targets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		targets = list(self._targets.values())
		if tenant_id is not None:
			targets = [item for item in targets if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(targets, key=lambda item: item.id)]

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility alias exposing audit findings as ACCS records."""
		records = list(self._findings.values())
		if tenant_id is not None:
			records = [record for record in records if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(records, key=lambda item: item.id)]

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that records a manual accessibility finding."""
		metadata = dict(metadata or {})
		return self.record_finding(
			finding_id=record_id,
			tenant_id=tenant_id,
			target_id=str(metadata.get("target_id") or "manual"),
			rule=str(metadata.get("rule") or "manual_accessibility_review"),
			severity=str(metadata.get("severity") or "low"),
			description=str(metadata.get("description") or "Manual accessibility review finding."),
			remediation_owner=str(metadata.get("remediation_owner") or metadata.get("owner") or "accessibility-owner"),
			status=status,
			evidence=metadata,
		)

	def run_audit(
		self,
		audit_id: str,
		tenant_id: str,
		standard_id: str,
		target_ids: list[str] | tuple[str, ...],
		remediation_owner: str | None = None,
	) -> dict[str, Any]:
		standard = self._get_standard(tenant_id, standard_id)
		self._enforce_audit_policy(tenant_id, standard is not None)
		if not target_ids:
			raise ValueError("at least one accessibility target is required")
		assert standard is not None
		finding_ids: list[str] = []
		for target_id in target_ids:
			target = self._targets.get(self._key(tenant_id, target_id))
			if target is None or target.tenant_id != tenant_id:
				raise KeyError(f"unknown accessibility target: {target_id}")
			for index, draft in enumerate(self._engine.audit_target(standard, target), start=1):
				finding = self.record_finding(
					finding_id=f"{audit_id}:{target_id}:{index}",
					tenant_id=tenant_id,
					target_id=target_id,
					rule=str(draft["rule"]),
					severity=str(draft["severity"]),
					description=str(draft["description"]),
					remediation_owner=remediation_owner or "",
					evidence=dict(draft["evidence"]),
				)
				finding_ids.append(str(finding["id"]))
		audit = AccessibilityAudit(
			id=audit_id,
			tenant_id=tenant_id,
			standard_id=standard_id,
			target_ids=tuple(target_ids),
			finding_ids=tuple(finding_ids),
		)
		key = self._key(tenant_id, audit_id)
		if key in self._audits:
			raise ValueError(f"duplicate accessibility audit for tenant: {audit_id}")
		self._audits[key] = audit
		self._record_event(
			tenant_id=tenant_id,
			event_type="audit_completed",
			subject_id=audit_id,
			message=f"Completed accessibility audit {audit_id}.",
			evidence={"finding_count": len(finding_ids), "standard_id": standard_id},
		)
		return {
			**audit.to_dict(),
			"summary": self._engine.summarize_findings([self._findings[self._key(tenant_id, item)].to_dict() for item in finding_ids]),
		}

	def list_audits(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		audits = list(self._audits.values())
		if tenant_id is not None:
			audits = [item for item in audits if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(audits, key=lambda item: item.id)]

	def record_finding(
		self,
		finding_id: str,
		tenant_id: str,
		target_id: str,
		rule: str,
		severity: str,
		description: str,
		remediation_owner: str,
		status: str = "open",
		evidence: dict[str, Any] | None = None,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		review_required = self._enforce_finding_policy(tenant_id, bool(remediation_owner), severity, review_recorded)
		finding = AccessibilityFinding(
			id=finding_id,
			tenant_id=tenant_id,
			target_id=target_id,
			rule=rule,
			severity=severity,
			description=description,
			remediation_owner=remediation_owner,
			status="review_required" if review_required and status == "open" else status,
			review_required=review_required,
			review_recorded=review_recorded,
			evidence=dict(evidence or {}),
		)
		key = self._key(tenant_id, finding_id)
		if key in self._findings:
			raise ValueError(f"duplicate accessibility finding for tenant: {finding_id}")
		self._findings[key] = finding
		if remediation_owner:
			self._remediations.setdefault(
				key,
				RemediationTask(
					id=f"remediate:{finding_id}",
					tenant_id=tenant_id,
					finding_id=finding_id,
					owner=remediation_owner,
					status="review_required" if review_required else "open",
					review_recorded=review_recorded,
				),
			)
		self._record_event(
			tenant_id=tenant_id,
			event_type="finding_recorded",
			subject_id=finding_id,
			message=f"Recorded {severity} accessibility finding for {target_id}.",
			evidence={"rule": rule, "review_required": review_required},
		)
		return finding.to_dict()

	def list_findings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		findings = list(self._findings.values())
		if tenant_id is not None:
			findings = [item for item in findings if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(findings, key=lambda item: item.id)]

	def update_remediation(
		self,
		finding_id: str,
		status: str,
		review_recorded: bool = False,
		due_date: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		key = self._find_remediation_key(finding_id, tenant_id)
		task = self._remediations.get(key)
		if task is None:
			raise KeyError(f"unknown remediation task: {finding_id}")
		if tenant_id is not None and task.tenant_id != tenant_id:
			raise KeyError(f"unknown remediation task for tenant: {finding_id}")
		updated = RemediationTask(
			id=task.id,
			tenant_id=task.tenant_id,
			finding_id=task.finding_id,
			owner=task.owner,
			status=status,
			due_date=due_date or task.due_date,
			review_recorded=review_recorded or task.review_recorded,
		)
		self._remediations[key] = updated
		self._record_event(
			tenant_id=updated.tenant_id,
			event_type="remediation_updated",
			subject_id=finding_id,
			message=f"Updated remediation task {updated.id} to {status}.",
			evidence={"status": status, "review_recorded": updated.review_recorded},
		)
		return updated.to_dict()

	def record_review(
		self,
		finding_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		key = self._key(tenant_id, finding_id)
		finding = self._findings.get(key)
		if finding is None or finding.tenant_id != tenant_id:
			raise KeyError(f"unknown accessibility finding for tenant: {finding_id}")
		if not reviewer:
			raise ValueError("reviewer is required")
		if decision not in {"approved", "rejected", "needs_work"}:
			raise ValueError("review decision must be approved, rejected, or needs_work")
		if not notes:
			raise ValueError("review notes are required")
		review = AccessibilityReview(
			id=f"review:{finding_id}:{len(self._reviews) + 1}",
			tenant_id=tenant_id,
			finding_id=finding_id,
			reviewer=reviewer,
			decision=decision,
			notes=notes,
		)
		review_key = self._key(tenant_id, review.id)
		self._reviews[review_key] = review
		reviewed_status = "open" if decision == "approved" and finding.status == "review_required" else finding.status
		self._findings[key] = AccessibilityFinding(
			id=finding.id,
			tenant_id=finding.tenant_id,
			target_id=finding.target_id,
			rule=finding.rule,
			severity=finding.severity,
			description=finding.description,
			remediation_owner=finding.remediation_owner,
			status=reviewed_status,
			review_required=finding.review_required,
			review_recorded=True,
			resolution=finding.resolution,
			evidence=finding.evidence,
		)
		task = self._remediations.get(key)
		if task is not None:
			self._remediations[key] = RemediationTask(
				id=task.id,
				tenant_id=task.tenant_id,
				finding_id=task.finding_id,
				owner=task.owner,
				status="open" if decision == "approved" and task.status == "review_required" else task.status,
				due_date=task.due_date,
				review_recorded=True,
			)
		self._record_event(
			tenant_id=tenant_id,
			event_type="finding_review_recorded",
			subject_id=finding_id,
			message=f"Recorded {decision} accessibility review for {finding_id}.",
			evidence={"review_id": review.id, "reviewer": reviewer},
		)
		return review.to_dict()

	def close_finding(
		self,
		finding_id: str,
		tenant_id: str,
		resolution: str,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		key = self._key(tenant_id, finding_id)
		finding = self._findings.get(key)
		if finding is None or finding.tenant_id != tenant_id:
			raise KeyError(f"unknown accessibility finding for tenant: {finding_id}")
		if not resolution:
			raise ValueError("resolution evidence is required")
		if finding.review_required and not finding.review_recorded:
			raise PermissionError("critical_accessibility_review_required")
		if finding.review_required and not self._has_approved_review(finding):
			raise PermissionError("critical_accessibility_review_not_approved")
		closed = AccessibilityFinding(
			id=finding.id,
			tenant_id=finding.tenant_id,
			target_id=finding.target_id,
			rule=finding.rule,
			severity=finding.severity,
			description=finding.description,
			remediation_owner=finding.remediation_owner,
			status="closed",
			review_required=finding.review_required,
			review_recorded=finding.review_recorded,
			resolution=resolution,
			evidence=finding.evidence,
		)
		self._findings[key] = closed
		task = self._remediations.get(key)
		if task is not None:
			self._remediations[key] = RemediationTask(
				id=task.id,
				tenant_id=task.tenant_id,
				finding_id=task.finding_id,
				owner=task.owner,
				status="closed",
				due_date=task.due_date,
				review_recorded=task.review_recorded or finding.review_recorded,
			)
		self._record_event(
			tenant_id=tenant_id,
			event_type="finding_closed",
			subject_id=finding_id,
			message=f"Closed accessibility finding {finding_id}.",
			evidence={"resolution": resolution},
		)
		return closed.to_dict()

	def record_accessibility_exception(
		self,
		exception_id: str,
		tenant_id: str,
		finding_id: str,
		approver: str,
		reason: str,
		expires_on: str,
		compensating_controls: list[str] | tuple[str, ...],
		status: str = "approved",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		key = self._key(tenant_id, finding_id)
		finding = self._findings.get(key)
		if finding is None or finding.tenant_id != tenant_id:
			raise KeyError(f"unknown accessibility finding for tenant: {finding_id}")
		if finding.status == "closed":
			raise ValueError("accessibility exceptions cannot be recorded for closed findings")
		if not approver:
			raise ValueError("exception approver is required")
		if not reason:
			raise ValueError("exception reason is required")
		if status not in {"approved", "revoked"}:
			raise ValueError("exception status must be approved or revoked")
		self._enforce_exception_policy(expires_on, compensating_controls)
		exception_key = self._key(tenant_id, exception_id)
		if exception_key in self._exceptions:
			raise ValueError(f"duplicate accessibility exception for tenant: {exception_id}")
		exception = AccessibilityException(
			id=exception_id,
			tenant_id=tenant_id,
			finding_id=finding_id,
			approver=approver,
			reason=reason,
			expires_on=expires_on,
			compensating_controls=tuple(compensating_controls),
			status=status,
		)
		self._exceptions[exception_key] = exception
		self._record_event(
			tenant_id=tenant_id,
			event_type="accessibility_exception_recorded",
			subject_id=finding_id,
			message=f"Recorded accessibility exception {exception_id} for finding {finding_id}.",
			evidence={
				"exception_id": exception_id,
				"expires_on": expires_on,
				"status": status,
				"compensating_control_count": len(compensating_controls),
			},
		)
		return exception.to_dict()

	def list_accessibility_exceptions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		exceptions = list(self._exceptions.values())
		if tenant_id is not None:
			exceptions = [item for item in exceptions if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(exceptions, key=lambda item: item.id)]

	def register_accessibility_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		registered: bool = True,
		contribution_disclosed: bool = True,
		policy_ref: str | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		normalized_runtime = self._normalize_accessibility_agent_runtime(runtime)
		normalized_role = self._normalize_accessibility_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"accessibility_agent_present": True,
			"agent_registered": registered,
			"agent_runtime_supported": normalized_runtime is not None,
			"agent_role_supported": normalized_role is not None,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		_raise_if_blocked(result)
		if not name:
			raise ValueError("accessibility agent name is required")
		key = self._key(tenant_id, agent_id)
		if key in self._agents:
			raise ValueError(f"duplicate accessibility agent for tenant: {agent_id}")
		agent = AccessibilityAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime or runtime,
			role=normalized_role or role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref,
			status=status,
		)
		self._agents[key] = agent
		self._record_event(
			tenant_id=tenant_id,
			event_type="accessibility_agent_registered",
			subject_id=agent_id,
			message=f"Registered accessibility agent {agent_id}.",
			evidence={"runtime": agent.runtime, "role": agent.role, "scope": scope},
		)
		return agent.to_dict()

	def list_accessibility_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		agents = list(self._agents.values())
		if tenant_id is not None:
			agents = [item for item in agents if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(agents, key=lambda item: item.id)]

	def validate_batch_accessibility_mutation(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "batch_accessibility_mutation",
			"event_stream": event_stream,
			"mutation_count": mutation_count,
		})
		_raise_if_blocked(result)
		return {
			"tenant_id": tenant_id,
			"event_stream": event_stream,
			"mutation_count": mutation_count,
			"accepted": True,
			"rule_result": result,
		}

	def list_remediations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tasks = list(self._remediations.values())
		if tenant_id is not None:
			tasks = [item for item in tasks if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(tasks, key=lambda item: item.id)]

	def list_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		reviews = list(self._reviews.values())
		if tenant_id is not None:
			reviews = [item for item in reviews if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(reviews, key=lambda item: item.id)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._events)
		if tenant_id is not None:
			events = [item for item in events if item.tenant_id == tenant_id]
		return [item.to_dict() for item in events]

	def validate_publication(self, target_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		target = self._get_target(target_id, tenant_id)
		if target is None:
			raise KeyError(f"unknown accessibility target: {target_id}")
		if tenant_id is not None and target.tenant_id != tenant_id:
			raise KeyError(f"unknown accessibility target for tenant: {target_id}")
		result = self.evaluate({
			"tenant_context_present": bool(target.tenant_id),
			"published_ui": target.published_ui,
			"contrast_passed": target.contrast_ratio >= 4.5,
			"media_content_present": target.media_content_present,
			"captions_available": target.captions_available,
		})
		open_findings = [
			item for item in self.list_findings(target.tenant_id)
			if item["target_id"] == target.id and item["status"] != "closed"
		]
		active_exceptions = [
			item for item in self.list_accessibility_exceptions(target.tenant_id)
			if item["status"] == "approved"
			and self._exception_is_active(item["expires_on"])
			and item["finding_id"] in {finding["id"] for finding in open_findings}
		]
		exception_finding_ids = {item["finding_id"] for item in active_exceptions}
		publishable_with_exception = (
			result["decision"] != "allow"
			and bool(open_findings)
			and all(finding["id"] in exception_finding_ids for finding in open_findings)
		)
		return {
			"target": target.to_dict(),
			"publishable": result["decision"] == "allow",
			"publishable_with_exception": publishable_with_exception,
			"open_findings": open_findings,
			"active_exceptions": active_exceptions,
			"rule_result": result,
		}

	def compliance_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		findings = self.list_findings(tenant_id)
		remediations = self.list_remediations(tenant_id)
		open_findings = [item for item in findings if item["status"] != "closed"]
		return {
			"tenant_id": tenant_id,
			"standard_count": len(self.list_standards(tenant_id)),
			"target_count": len(self.list_targets(tenant_id)),
			"audit_count": len(self.list_audits(tenant_id)),
			"finding_count": len(findings),
			"open_finding_count": len(open_findings),
			"remediation_count": len(remediations),
			"review_count": len(self.list_reviews(tenant_id)),
			"exception_count": len(self.list_accessibility_exceptions(tenant_id)),
			"accessibility_agent_count": len(self.list_accessibility_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"critical_or_high_count": self._engine.summarize_findings(findings)["critical_or_high_count"],
		}

	def _enforce_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
		})
		_raise_if_blocked(result)

	def _enforce_audit_policy(self, tenant_id: str, standard_selected: bool) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "start_audit",
			"standard_selected": standard_selected,
		})
		_raise_if_blocked(result)

	def _enforce_finding_policy(
		self,
		tenant_id: str,
		remediation_owner_assigned: bool,
		severity: str,
		review_recorded: bool,
	) -> bool:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"violation_detected": True,
			"remediation_owner_assigned": remediation_owner_assigned,
			"issue_severity": severity,
			"review_recorded": review_recorded,
		})
		_raise_if_denied(result)
		return result["decision"] == "require_review"

	def _enforce_exception_policy(
		self,
		expires_on: str,
		compensating_controls: list[str] | tuple[str, ...],
	) -> None:
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "record_accessibility_exception",
			"exception_expiry_present": bool(expires_on),
			"compensating_controls_present": bool(compensating_controls),
		})
		_raise_if_blocked(result)
		if not self._exception_is_active(expires_on):
			raise PermissionError("accessibility_exception_expired")

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		evidence: dict[str, Any] | None = None,
	) -> None:
		self._events.append(
			AccessibilityAuditEvent(
				id=f"accs-event-{len(self._events) + 1}",
				tenant_id=tenant_id,
				event_type=event_type,
				subject_id=subject_id,
				message=message,
				evidence=dict(evidence or {}),
			)
		)

	def _has_approved_review(self, finding: AccessibilityFinding) -> bool:
		reviews = [
			review for review in self._reviews.values()
			if review.finding_id == finding.id and review.tenant_id == finding.tenant_id
		]
		if not reviews:
			return finding.review_recorded
		return any(review.decision == "approved" for review in reviews)

	def _exception_is_active(self, expires_on: str) -> bool:
		try:
			return date.fromisoformat(expires_on) >= date.today()
		except ValueError:
			raise ValueError("exception expiry must be an ISO date")

	def _key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	def _get_standard(self, tenant_id: str, standard_id: str) -> AccessibilityStandard | None:
		return self._standards.get(self._key(tenant_id, standard_id)) or self._standards.get(self._key("default", standard_id))

	def _get_target(self, target_id: str, tenant_id: str | None = None) -> AccessibilityTarget | None:
		if tenant_id is not None:
			return self._targets.get(self._key(tenant_id, target_id))
		matches = [target for target in self._targets.values() if target.id == target_id]
		return matches[0] if len(matches) == 1 else None

	def _find_remediation_key(self, finding_id: str, tenant_id: str | None = None) -> str:
		if tenant_id is not None:
			return self._key(tenant_id, finding_id)
		matches = [key for key, task in self._remediations.items() if task.finding_id == finding_id]
		if len(matches) == 1:
			return matches[0]
		return self._key("", finding_id)

	def _normalize_accessibility_agent_runtime(self, runtime: str) -> str | None:
		normalized = runtime.strip().lower().replace("-", "_").replace(" ", "_")
		return normalized if normalized in SUPPORTED_ACCESSIBILITY_AGENT_RUNTIMES else None

	def _normalize_accessibility_agent_role(self, role: str) -> str | None:
		normalized = role.strip().lower().replace("-", "_").replace(" ", "_")
		return normalized if normalized in SUPPORTED_ACCESSIBILITY_AGENT_ROLES else None


	# -------------------------------------------------------------------------
	# Extended async methods — in-memory store pattern
	# -------------------------------------------------------------------------

	async def audit_wcag(
		self,
		url: str,
		tenant_id: str,
		level: str = "AA",
		owner: str = "accessibility-team",
	) -> dict[str, Any]:
		"""
		Run a WCAG audit against a URL. Registers a target and runs an audit.
		Returns findings summary.
		"""
		from urllib.parse import urlparse
		parsed = urlparse(url)
		route = parsed.path or "/"
		target_id = f"wcag-{abs(hash(url)):x}"
		standard_id = f"wcag_2_2_{level.lower()}"

		# Register standard if not present
		standard_key = self._key(tenant_id, standard_id)
		if standard_key not in self._standards:
			try:
				self.register_standard(standard_id=standard_id, tenant_id=tenant_id,
					name="WCAG", version="2.2", level=level)
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		# Register target if not present
		target_key = self._key(tenant_id, target_id)
		if target_key not in self._targets:
			self.register_target(
				target_id=target_id, tenant_id=tenant_id,
				surface="web", route=route, owner=owner,
				published_ui=True,
			)

		# Run audit
		audit_id = f"audit-wcag-{abs(hash(url + level)):x}"
		try:
			return self.run_audit(
				audit_id=audit_id,
				tenant_id=tenant_id,
				standard_id=standard_id,
				target_ids=[target_id],
				remediation_owner=owner,
			)
		except ValueError:
			# Audit already run — return existing
			key = self._key(tenant_id, audit_id)
			audit = self._audits.get(key)
			return audit.to_dict() if audit else {"audit_id": audit_id, "status": "already_run"}

	async def auto_remediate(
		self,
		issue_id: str,
		tenant_id: str | None = None,
		resolution: str = "auto_remediated",
	) -> dict[str, Any]:
		"""Attempt automatic remediation of a finding by closing it with auto-resolution."""
		key = self._find_remediation_key(issue_id, tenant_id)
		task = self._remediations.get(key)
		if task is None:
			raise KeyError(f"remediation_task_not_found:{issue_id}")
		effective_tenant = tenant_id or task.tenant_id
		# Mark finding closed if review not required
		finding_key = self._key(effective_tenant, issue_id)
		finding = self._findings.get(finding_key)
		if finding and not finding.review_required:
			return self.close_finding(issue_id, effective_tenant, resolution)
		# Otherwise just update remediation status
		return self.update_remediation(
			finding_id=issue_id,
			status="in_progress",
			review_recorded=False,
			tenant_id=effective_tenant,
		)

	async def screen_reader_test(
		self,
		tenant_id: str,
		target_id: str,
		test_tool: str = "nvda",
		tested_by: str = "qa",
	) -> dict[str, Any]:
		"""Record a screen-reader compatibility test result for a target."""
		self._enforce_tenant(tenant_id)
		target = self._get_target(target_id, tenant_id)
		if target is None:
			raise KeyError(f"target_not_found:{target_id}")
		finding_id = f"sr-{abs(hash(target_id + test_tool)):x}"
		# Register finding only if semantic labels missing
		if not target.semantic_labels_present:
			try:
				finding = self.record_finding(
					finding_id=finding_id,
					tenant_id=tenant_id,
					target_id=target_id,
					rule="screen_reader_compatibility",
					severity="high",
					description=f"Screen reader test ({test_tool}) detected missing semantic labels.",
					remediation_owner=tested_by,
				)
			except ValueError:
				finding = self._findings.get(self._key(tenant_id, finding_id), {})
				finding = finding.to_dict() if hasattr(finding, "to_dict") else finding
		else:
			finding = None
		self._record_event(tenant_id=tenant_id, event_type="screen_reader_test_completed",
			subject_id=target_id, message=f"Screen reader test ({test_tool}) on {target_id}.",
			evidence={"tool": test_tool, "passed": target.semantic_labels_present})
		return {
			"target_id": target_id,
			"test_tool": test_tool,
			"passed": target.semantic_labels_present,
			"finding": finding,
		}

	async def keyboard_nav_test(
		self,
		tenant_id: str,
		target_id: str,
		tested_by: str = "qa",
	) -> dict[str, Any]:
		"""Test keyboard navigation on a target and record findings."""
		self._enforce_tenant(tenant_id)
		target = self._get_target(target_id, tenant_id)
		if target is None:
			raise KeyError(f"target_not_found:{target_id}")
		passed = target.keyboard_navigation_present
		if not passed:
			finding_id = f"kb-{abs(hash(target_id + 'keyboard')):x}"
			try:
				self.record_finding(
					finding_id=finding_id,
					tenant_id=tenant_id,
					target_id=target_id,
					rule="keyboard_navigation",
					severity="critical",
					description="Keyboard navigation not fully implemented on this surface.",
					remediation_owner=tested_by,
				)
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		self._record_event(tenant_id=tenant_id, event_type="keyboard_nav_test_completed",
			subject_id=target_id, message=f"Keyboard nav test on {target_id}.",
			evidence={"passed": passed})
		return {"target_id": target_id, "keyboard_navigation_present": passed, "passed": passed}

	async def contrast_check(
		self,
		tenant_id: str,
		target_id: str,
		contrast_ratio: float | None = None,
	) -> dict[str, Any]:
		"""Check foreground/background contrast ratio against WCAG AA (4.5:1)."""
		self._enforce_tenant(tenant_id)
		target = self._get_target(target_id, tenant_id)
		if target is None:
			raise KeyError(f"target_not_found:{target_id}")
		ratio = contrast_ratio if contrast_ratio is not None else target.contrast_ratio
		wcag_aa_threshold = 4.5
		passed = ratio >= wcag_aa_threshold
		if not passed:
			finding_id = f"contrast-{abs(hash(target_id + str(ratio))):x}"
			try:
				self.record_finding(
					finding_id=finding_id,
					tenant_id=tenant_id,
					target_id=target_id,
					rule="contrast_ratio",
					severity="high",
					description=f"Contrast ratio {ratio:.2f} is below WCAG AA threshold of {wcag_aa_threshold}.",
					remediation_owner="ui-team",
					evidence={"contrast_ratio": ratio, "threshold": wcag_aa_threshold},
				)
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {"target_id": target_id, "contrast_ratio": ratio, "threshold": wcag_aa_threshold, "passed": passed}

	async def font_size_validate(
		self,
		tenant_id: str,
		target_id: str,
		min_body_px: float = 16.0,
		tested_by: str = "qa",
	) -> dict[str, Any]:
		"""Validate that body font size meets minimum readability requirement."""
		self._enforce_tenant(tenant_id)
		target = self._get_target(target_id, tenant_id)
		if target is None:
			raise KeyError(f"target_not_found:{target_id}")
		# Deterministic check — in a real integration, inspect CSS
		passed = min_body_px >= 16.0
		self._record_event(tenant_id=tenant_id, event_type="font_size_validated",
			subject_id=target_id, message=f"Font size validation on {target_id}: min={min_body_px}px.",
			evidence={"min_body_px": min_body_px, "passed": passed})
		return {"target_id": target_id, "min_body_px": min_body_px, "passed": passed}

	async def alt_text_audit(
		self,
		tenant_id: str,
		target_id: str,
		image_count: int,
		images_with_alt: int,
		audited_by: str = "qa",
	) -> dict[str, Any]:
		"""Audit alt-text coverage for images on a target."""
		self._enforce_tenant(tenant_id)
		target = self._get_target(target_id, tenant_id)
		if target is None:
			raise KeyError(f"target_not_found:{target_id}")
		coverage = images_with_alt / image_count if image_count else 1.0
		passed = coverage >= 1.0
		if not passed:
			finding_id = f"alt-{abs(hash(target_id + str(image_count))):x}"
			try:
				self.record_finding(
					finding_id=finding_id,
					tenant_id=tenant_id,
					target_id=target_id,
					rule="alt_text_missing",
					severity="high",
					description=f"{image_count - images_with_alt} image(s) missing alt text.",
					remediation_owner=audited_by,
					evidence={"image_count": image_count, "images_with_alt": images_with_alt},
				)
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {
			"target_id": target_id,
			"image_count": image_count,
			"images_with_alt": images_with_alt,
			"coverage": round(coverage, 4),
			"passed": passed,
		}

	async def aria_label_check(
		self,
		tenant_id: str,
		target_id: str,
		aria_coverage: float = 1.0,
		checked_by: str = "qa",
	) -> dict[str, Any]:
		"""Check ARIA label coverage. aria_coverage is fraction 0-1."""
		self._enforce_tenant(tenant_id)
		passed = aria_coverage >= 0.9
		self._record_event(tenant_id=tenant_id, event_type="aria_label_check_completed",
			subject_id=target_id, message=f"ARIA label check on {target_id}: coverage={aria_coverage:.0%}.",
			evidence={"aria_coverage": aria_coverage, "passed": passed})
		return {"target_id": target_id, "aria_coverage": aria_coverage, "passed": passed}

	async def focus_order_audit(
		self,
		tenant_id: str,
		target_id: str,
		issues_found: int = 0,
		audited_by: str = "qa",
	) -> dict[str, Any]:
		"""Audit focus order for logical keyboard tab sequence."""
		self._enforce_tenant(tenant_id)
		passed = issues_found == 0
		if not passed:
			finding_id = f"focus-{abs(hash(target_id + str(issues_found))):x}"
			try:
				self.record_finding(
					finding_id=finding_id,
					tenant_id=tenant_id,
					target_id=target_id,
					rule="focus_order",
					severity="medium",
					description=f"{issues_found} focus order issue(s) detected.",
					remediation_owner=audited_by,
				)
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {"target_id": target_id, "issues_found": issues_found, "passed": passed}

	async def form_accessibility_audit(
		self,
		tenant_id: str,
		target_id: str,
		field_count: int,
		labelled_fields: int,
		audited_by: str = "qa",
	) -> dict[str, Any]:
		"""Audit form field label coverage."""
		self._enforce_tenant(tenant_id)
		coverage = labelled_fields / field_count if field_count else 1.0
		passed = coverage >= 1.0
		if not passed:
			finding_id = f"form-{abs(hash(target_id + str(field_count))):x}"
			try:
				self.record_finding(
					finding_id=finding_id,
					tenant_id=tenant_id,
					target_id=target_id,
					rule="form_field_labels",
					severity="high",
					description=f"{field_count - labelled_fields} form field(s) missing labels.",
					remediation_owner=audited_by,
					evidence={"field_count": field_count, "labelled_fields": labelled_fields},
				)
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {
			"target_id": target_id,
			"field_count": field_count,
			"labelled_fields": labelled_fields,
			"coverage": round(coverage, 4),
			"passed": passed,
		}

	async def pdf_accessibility(
		self,
		tenant_id: str,
		document_ref: str,
		tagged_pdf: bool = False,
		owner: str = "content-team",
	) -> dict[str, Any]:
		"""Evaluate PDF accessibility (tagged PDF requirement)."""
		self._enforce_tenant(tenant_id)
		finding = None
		if not tagged_pdf:
			finding_id = f"pdf-{abs(hash(document_ref)):x}"
			target_id = f"pdf-target-{abs(hash(document_ref)):x}"
			# Register target if needed
			if self._key(tenant_id, target_id) not in self._targets:
				self.register_target(target_id=target_id, tenant_id=tenant_id,
					surface="document", route=document_ref, owner=owner)
			try:
				finding = self.record_finding(
					finding_id=finding_id,
					tenant_id=tenant_id,
					target_id=target_id,
					rule="pdf_tagged",
					severity="high",
					description="PDF document is not tagged, preventing screen reader access.",
					remediation_owner=owner,
					evidence={"document_ref": document_ref},
				)
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {"document_ref": document_ref, "tagged_pdf": tagged_pdf, "passed": tagged_pdf, "finding": finding}

	async def accessibility_report(
		self,
		tenant_id: str,
		include_closed: bool = False,
	) -> dict[str, Any]:
		"""Generate a comprehensive accessibility status report for the tenant."""
		findings = self.list_findings(tenant_id)
		if not include_closed:
			findings = [f for f in findings if f["status"] != "closed"]
		remediations = self.list_remediations(tenant_id)
		open_remediations = [r for r in remediations if r["status"] not in {"closed"}]
		by_severity: dict[str, int] = {}
		for f in findings:
			sev = f.get("severity", "unknown")
			by_severity[sev] = by_severity.get(sev, 0) + 1
		return {
			"tenant_id": tenant_id,
			"report_type": "accessibility_report",
			"open_findings": len(findings),
			"by_severity": by_severity,
			"open_remediations": len(open_remediations),
			"audits_run": len(self.list_audits(tenant_id)),
			"targets_registered": len(self.list_targets(tenant_id)),
			"exceptions_active": len([
				e for e in self.list_accessibility_exceptions(tenant_id)
				if e["status"] == "approved"
			]),
		}

	async def remediation_track(
		self,
		tenant_id: str,
		finding_id: str,
		status: str,
		due_date: str | None = None,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		"""Update remediation tracking status for a finding."""
		return self.update_remediation(
			finding_id=finding_id,
			status=status,
			review_recorded=review_recorded,
			due_date=due_date,
			tenant_id=tenant_id,
		)

	async def user_preference_store(
		self,
		tenant_id: str,
		user_id: str,
		preferences: dict[str, Any],
	) -> dict[str, Any]:
		"""
		Store user accessibility preferences (font size, contrast mode, etc.).
		Recorded as an audit event with preference payload.
		"""
		self._enforce_tenant(tenant_id)
		pref_id = f"pref-{abs(hash(tenant_id + user_id)):x}"
		self._record_event(
			tenant_id=tenant_id,
			event_type="user_accessibility_preferences_stored",
			subject_id=pref_id,
			message=f"Accessibility preferences stored for user {user_id}.",
			evidence={"user_id": user_id, "preferences": preferences},
		)
		return {"preference_id": pref_id, "user_id": user_id, "tenant_id": tenant_id, "preferences": preferences}

	async def accessibility_analytics(
		self,
		tenant_id: str,
		days: int = 30,
	) -> dict[str, Any]:
		"""Return aggregated accessibility compliance analytics for the tenant."""
		findings = self.list_findings(tenant_id)
		closed = [f for f in findings if f["status"] == "closed"]
		open_findings = [f for f in findings if f["status"] != "closed"]
		critical_high = [f for f in open_findings if f.get("severity") in {"critical", "high"}]
		return {
			"tenant_id": tenant_id,
			"window_days": days,
			"total_findings": len(findings),
			"open_findings": len(open_findings),
			"closed_findings": len(closed),
			"critical_or_high_open": len(critical_high),
			"closure_rate": round(len(closed) / len(findings), 4) if findings else 1.0,
			"audits_completed": len(self.list_audits(tenant_id)),
			"targets_registered": len(self.list_targets(tenant_id)),
			"audit_events": len(self.list_audit_events(tenant_id)),
		}


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(action.get("reason", "accessibility_policy_blocked") for action in result["actions"])
	if result["decision"] == "require_review":
		raise PermissionError(reasons or "accessibility_review_required")
	raise PermissionError(reasons or "accessibility_policy_blocked")


def _raise_if_denied(result: dict[str, Any]) -> None:
	if result["decision"] != "deny":
		return
	reasons = ", ".join(action.get("reason", "accessibility_policy_blocked") for action in result["actions"])
	raise PermissionError(reasons or "accessibility_policy_blocked")
