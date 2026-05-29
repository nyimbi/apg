"""Service layer for APG Accessibility Services."""

from __future__ import annotations

from typing import Any

from .accessibility_engine import AccessibilityAuditEngine
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	AccessibilityAudit,
	AccessibilityAuditEvent,
	AccessibilityFinding,
	AccessibilityReview,
	AccessibilityStandard,
	AccessibilityTarget,
	RemediationTask,
)


class AccsService:
	"""Accessibility standard registry, audit runner, and remediation tracker."""

	def __init__(self) -> None:
		self._standards: dict[str, AccessibilityStandard] = {}
		self._targets: dict[str, AccessibilityTarget] = {}
		self._findings: dict[str, AccessibilityFinding] = {}
		self._remediations: dict[str, RemediationTask] = {}
		self._audits: dict[str, AccessibilityAudit] = {}
		self._reviews: dict[str, AccessibilityReview] = {}
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
		self._standards[standard_id] = standard
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
		self._targets[target_id] = target
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
		self._enforce_audit_policy(tenant_id, bool(standard_id and standard_id in self._standards))
		if not target_ids:
			raise ValueError("at least one accessibility target is required")
		standard = self._standards[standard_id]
		finding_ids: list[str] = []
		for target_id in target_ids:
			target = self._targets.get(target_id)
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
		self._audits[audit_id] = audit
		return {
			**audit.to_dict(),
			"summary": self._engine.summarize_findings([self._findings[item].to_dict() for item in finding_ids]),
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
		self._findings[finding_id] = finding
		if remediation_owner:
			self._remediations.setdefault(
				finding_id,
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
		task = self._remediations.get(finding_id)
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
		self._remediations[finding_id] = updated
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
		finding = self._findings.get(finding_id)
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
		self._reviews[review.id] = review
		reviewed_status = "open" if decision == "approved" and finding.status == "review_required" else finding.status
		self._findings[finding_id] = AccessibilityFinding(
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
		task = self._remediations.get(finding_id)
		if task is not None:
			self._remediations[finding_id] = RemediationTask(
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
		finding = self._findings.get(finding_id)
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
		self._findings[finding_id] = closed
		task = self._remediations.get(finding_id)
		if task is not None:
			self._remediations[finding_id] = RemediationTask(
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
		target = self._targets.get(target_id)
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
		return {
			"target": target.to_dict(),
			"publishable": result["decision"] == "allow",
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
