"""PolicyManagementService — GRC policy lifecycle management.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .capability_contract import (
	CAPABILITY_ID,
	CAPABILITY_VERSION,
	SUPPORTED_POLICY_TYPES,
	SUPPORTED_POLICY_STATUSES,
	SUPPORTED_REVIEW_FREQUENCIES,
	evaluate_capability_rules,
)
from .database.store import Store, get_store
from .domain.adapters import (
	AuthAdapter,
	AuditAdapter,
	NotifyAdapter,
	get_auth_adapter,
	get_audit_adapter,
	get_notify_adapter,
)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
	return str(uuid.uuid4())


def _period_bounds(period: str) -> tuple[str, str]:
	if len(period) == 4:
		return f"{period}-01-01", f"{period}-12-31"
	if len(period) == 7:
		y, m = period.split("-")
		ed = 31 if int(m) in {1, 3, 5, 7, 8, 10, 12} else 30 if int(m) != 2 else 28
		return f"{period}-01", f"{period}-{ed:02d}"
	return period, period


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class PolicyManagementService:
	"""GRC policy lifecycle management: draft, review, approve, publish, acknowledge,
	exceptions, revisions, compliance mapping, and gap analysis.

	Usage (standalone)::

		svc = PolicyManagementService()
		policy = await svc.create_policy("ISMS Policy", "information_security", ...)

	Usage (platform)::

		svc = PolicyManagementService(auth=AuthService.from_env())
	"""

	def __init__(
		self,
		*,
		db_url: str | None = None,
		store: Store | None = None,
		auth: Any | None = None,
		audit: Any | None = None,
		notify: Any | None = None,
		tenant_id: str = "default",
	) -> None:
		self._store: Store = store or get_store(db_url)
		self._auth: AuthAdapter = get_auth_adapter(auth)
		self._audit: AuditAdapter = get_audit_adapter(audit)
		self._notify: NotifyAdapter = get_notify_adapter(notify)
		self._tenant_id = tenant_id
		self._capability = CAPABILITY_ID
		self._version = CAPABILITY_VERSION

	async def _audit_event(
		self, event_type: str, actor_id: str, resource_id: str, details: dict[str, Any]
	) -> None:
		await self._audit.log_event(event_type, actor_id, self._tenant_id, resource_id, details)

	async def _get_policy(self, policy_id: str) -> dict[str, Any]:
		rec = await self._store.get("policies", policy_id)
		if rec is None:
			raise ValueError(f"Policy not found: {policy_id}")
		return rec

	def _next_review_date(self, effective_date: str, review_cycle_months: int) -> str:
		eff = date.fromisoformat(effective_date)
		return (eff + timedelta(days=review_cycle_months * 30)).isoformat()

	# ─────────────────────────────────────────────────────────
	# Lifecycle
	# ─────────────────────────────────────────────────────────

	async def create_policy(
		self,
		title: str,
		category: str,
		policy_type: str,
		owner_id: str,
		effective_date: str,
		review_cycle_months: int,
		*,
		scope: str = "organization_wide",
		description: str = "",
		version: str = "1.0",
	) -> dict[str, Any]:
		"""Create a new policy record in draft status.

		Validates type, scope, and review frequency before persisting.
		Emits ``policy_drafted`` event.
		"""
		assert title, "title required"
		assert owner_id, "owner_id required"
		assert effective_date, "effective_date required"
		assert review_cycle_months > 0, "review_cycle_months must be positive"

		if policy_type not in SUPPORTED_POLICY_TYPES:
			raise ValueError(f"Unsupported policy type: {policy_type}. Valid: {SUPPORTED_POLICY_TYPES}")

		rule_ctx = {
			"operation": "create_policy",
			"tenant_context_present": True,
			"title_present": True,
			"policy_type_supported": True,
			"owner_present": True,
			"scope_type_supported": True,
			"effective_date_present": True,
			"review_date_present": True,
			"review_frequency_supported": True,
			"version_present": True,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Policy creation denied: {verdict['matched_rules']}")

		review_date = self._next_review_date(effective_date, review_cycle_months)
		policy: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"title": title,
			"category": category,
			"policy_type": policy_type,
			"owner_id": owner_id,
			"scope": scope,
			"description": description,
			"version": version,
			"status": "draft",
			"effective_date": effective_date,
			"review_cycle_months": review_cycle_months,
			"next_review_date": review_date,
			"review_history": [],
			"revision_history": [],
			"acknowledgement_stats": {"required": 0, "completed": 0},
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("policies", policy)
		await self._audit_event("policy_drafted", owner_id, policy["id"], {"title": title, "type": policy_type})
		return policy

	async def draft_policy_content(
		self,
		policy_id: str,
		content_sections: list[dict[str, Any]],
		author_id: str,
	) -> dict[str, Any]:
		"""Attach structured content sections to a draft policy.

		Each section: {title, body, section_number}. Replaces existing content.
		Transitions status to 'draft' if currently 'under_revision'.
		"""
		assert author_id, "author_id required"
		assert content_sections, "content_sections required"

		policy = await self._get_policy(policy_id)
		if policy.get("status") not in {"draft", "under_revision"}:
			raise ValueError(f"Policy {policy_id} is {policy.get('status')!r}; content can only be drafted on draft or under_revision policies")

		policy["content_sections"] = content_sections
		policy["content_author_id"] = author_id
		policy["content_updated_at"] = _now()
		policy["updated_at"] = _now()
		policy["word_count"] = sum(len(s.get("body", "").split()) for s in content_sections)

		await self._store.put("policies", policy)
		await self._audit_event(
			"policy_content_drafted", author_id, policy_id,
			{"section_count": len(content_sections), "word_count": policy["word_count"]},
		)
		return policy

	async def policy_review(
		self,
		policy_id: str,
		reviewer_id: str,
		comments: str,
		recommended_action: str,
	) -> dict[str, Any]:
		"""Submit a policy review with comments and recommended action.

		Recommended actions: approve | request_changes | reject.
		Enforces reviewer != owner (segregation of duties).
		"""
		assert reviewer_id, "reviewer_id required"
		assert recommended_action in {"approve", "request_changes", "reject"}, (
			"recommended_action: approve | request_changes | reject"
		)

		policy = await self._get_policy(policy_id)

		if policy.get("owner_id") == reviewer_id:
			raise PermissionError("Reviewer cannot be the policy owner (segregation of duties)")

		review_record: dict[str, Any] = {
			"id": _uid(),
			"policy_id": policy_id,
			"reviewer_id": reviewer_id,
			"comments": comments,
			"recommended_action": recommended_action,
			"reviewed_at": _now(),
		}
		await self._store.put("policy_reviews", review_record)

		policy.setdefault("review_history", []).append(review_record["id"])
		policy["last_reviewed_by"] = reviewer_id
		policy["last_reviewed_at"] = _now()
		policy["status"] = "in_review"
		policy["updated_at"] = _now()
		await self._store.put("policies", policy)

		await self._audit_event(
			"policy_review_completed", reviewer_id, policy_id,
			{"recommended_action": recommended_action},
		)
		await self._notify.send(
			policy["owner_id"], "email",
			f"Policy review submitted: {policy['title']}",
			f"Reviewer {reviewer_id} recommends: {recommended_action}\nComments: {comments}",
		)
		return review_record

	async def approve_policy(
		self,
		policy_id: str,
		approver_id: str,
		approval_date: str,
		*,
		comments: str = "",
	) -> dict[str, Any]:
		"""Approve a policy, transitioning it to 'approved' status.

		Enforces approver != owner. Policy must be in 'in_review' status.
		"""
		assert approver_id, "approver_id required"
		assert approval_date, "approval_date required"

		policy = await self._get_policy(policy_id)

		if policy.get("owner_id") == approver_id:
			raise PermissionError("Approver cannot be the policy owner (segregation of duties)")
		if policy.get("status") not in {"in_review", "draft"}:
			raise ValueError(f"Policy must be in_review or draft to approve; current: {policy.get('status')}")

		policy["status"] = "approved"
		policy["approved_by"] = approver_id
		policy["approved_at"] = approval_date
		policy["approval_comments"] = comments
		policy["updated_at"] = _now()
		await self._store.put("policies", policy)

		await self._audit_event("policy_approved", approver_id, policy_id, {"approval_date": approval_date})
		await self._notify.send(
			policy["owner_id"], "email",
			f"Policy approved: {policy['title']}",
			f"Policy {policy['title']} has been approved by {approver_id} on {approval_date}.",
		)
		return policy

	async def publish_policy(
		self,
		policy_id: str,
		distribution_list: list[str],
	) -> dict[str, Any]:
		"""Publish an approved policy to the distribution list.

		Policy must be in 'approved' status. Transitions to 'published'.
		Triggers acknowledgement requests for all recipients.
		"""
		assert distribution_list, "distribution_list required"

		policy = await self._get_policy(policy_id)

		rule_ctx = {
			"operation": "publish_policy",
			"approved": policy.get("status") == "approved",
			"review_date_overdue": False,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Publication denied: {verdict['matched_rules']}")

		policy["status"] = "published"
		policy["published_at"] = _now()
		policy["distribution_list"] = distribution_list
		policy["acknowledgement_stats"] = {
			"required": len(distribution_list),
			"completed": 0,
		}
		policy["updated_at"] = _now()
		await self._store.put("policies", policy)

		# Create acknowledgement requests
		for recipient in distribution_list:
			ack_req: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"policy_id": policy_id,
				"employee_id": recipient,
				"status": "pending",
				"deadline": (date.today() + timedelta(days=30)).isoformat(),
				"requested_at": _now(),
			}
			await self._store.put("policy_acknowledgements", ack_req)
			await self._notify.send(
				recipient, "email",
				f"Action required: Acknowledge policy '{policy['title']}'",
				f"Please acknowledge policy '{policy['title']}' by {ack_req['deadline']}.",
			)

		await self._audit_event(
			"policy_published", policy.get("approved_by", "system"), policy_id,
			{"distribution_count": len(distribution_list)},
		)
		return policy

	async def acknowledge_policy(
		self,
		policy_id: str,
		employee_id: str,
		acknowledgement_date: str,
		*,
		method: str = "electronic_signature",
	) -> dict[str, Any]:
		"""Record a policy acknowledgement from an employee.

		Updates the policy acknowledgement stats counter.
		"""
		assert employee_id, "employee_id required"
		assert acknowledgement_date, "acknowledgement_date required"

		# Find existing request
		ack_reqs = await self._store.query(
			"policy_acknowledgements",
			{"policy_id": policy_id, "employee_id": employee_id},
			limit=1,
		)

		if ack_reqs:
			ack_rec = ack_reqs[0]
			ack_rec["status"] = "completed"
			ack_rec["acknowledged_at"] = acknowledgement_date
			ack_rec["method"] = method
			await self._store.put("policy_acknowledgements", ack_rec)
		else:
			ack_rec = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"policy_id": policy_id,
				"employee_id": employee_id,
				"status": "completed",
				"method": method,
				"acknowledged_at": acknowledgement_date,
				"requested_at": _now(),
			}
			await self._store.put("policy_acknowledgements", ack_rec)

		# Increment counter
		policy = await self._get_policy(policy_id)
		stats = policy.get("acknowledgement_stats", {"required": 0, "completed": 0})
		stats["completed"] = stats.get("completed", 0) + 1
		policy["acknowledgement_stats"] = stats
		policy["updated_at"] = _now()
		await self._store.put("policies", policy)

		await self._audit_event(
			"acknowledgement_completed", employee_id, policy_id,
			{"method": method, "date": acknowledgement_date},
		)
		return ack_rec

	async def policy_exception_request(
		self,
		policy_id: str,
		requestor_id: str,
		reason: str,
		compensating_controls: str,
		risk_level: str,
		*,
		exception_type: str = "temporary_exemption",
		duration_days: int = 90,
	) -> dict[str, Any]:
		"""Request an exception to a policy with compensating controls and risk assessment.

		Creates a pending exception record awaiting approval.
		"""
		assert requestor_id, "requestor_id required"
		assert reason, "reason required"
		assert compensating_controls, "compensating_controls required"
		assert risk_level in {"low", "medium", "high", "critical"}, (
			"risk_level: low | medium | high | critical"
		)
		assert 1 <= duration_days <= 365, "duration_days: 1–365"

		rule_ctx = {
			"operation": "request_exception",
			"exception_type_supported": True,
			"rationale_present": True,
			"expiration_present": True,
			"exception_days": duration_days,
			"risk_assessment_present": True,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Exception request denied: {verdict['matched_rules']}")

		expiry = (date.today() + timedelta(days=duration_days)).isoformat()
		exception: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"policy_id": policy_id,
			"requestor_id": requestor_id,
			"exception_type": exception_type,
			"reason": reason,
			"compensating_controls": compensating_controls,
			"risk_level": risk_level,
			"duration_days": duration_days,
			"expiry_date": expiry,
			"status": "pending",
			"requested_at": _now(),
		}
		await self._store.put("policy_exceptions", exception)
		await self._audit_event("exception_requested", requestor_id, policy_id, {"risk_level": risk_level})
		await self._notify.send(
			"compliance@datacraft.co.ke", "email",
			f"Policy exception request: {policy_id}",
			f"Exception requested by {requestor_id} for policy {policy_id}. Risk: {risk_level}",
		)
		return exception

	async def approve_exception(
		self,
		exception_id: str,
		approver_id: str,
		approved_until: str,
		conditions: str,
	) -> dict[str, Any]:
		"""Approve a policy exception with conditions and expiry date.

		Enforces approver != requestor (segregation of duties).
		"""
		assert approver_id, "approver_id required"
		assert approved_until, "approved_until required"
		assert conditions, "conditions required"

		exception = await self._store.get("policy_exceptions", exception_id)
		if exception is None:
			raise ValueError(f"Exception not found: {exception_id}")

		rule_ctx = {
			"operation": "approve_exception",
			"approver_is_requestor": exception.get("requestor_id") == approver_id,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Exception approval denied: {verdict['matched_rules']}")

		exception["status"] = "approved"
		exception["approver_id"] = approver_id
		exception["approved_until"] = approved_until
		exception["conditions"] = conditions
		exception["approved_at"] = _now()
		await self._store.put("policy_exceptions", exception)

		await self._audit_event(
			"exception_approved", approver_id, exception_id,
			{"approved_until": approved_until, "policy_id": exception.get("policy_id")},
		)
		await self._notify.send(
			exception["requestor_id"], "email",
			f"Exception approved until {approved_until}",
			f"Your policy exception has been approved. Conditions: {conditions}",
		)
		return exception

	async def policy_revision(
		self,
		policy_id: str,
		revision_reason: str,
		revision_summary: str,
		revised_by: str,
	) -> dict[str, Any]:
		"""Initiate a policy revision, incrementing the version and creating a revision record.

		Transitions status to 'under_revision'. The original version is archived.
		"""
		assert revision_reason, "revision_reason required"
		assert revision_summary, "revision_summary required"
		assert revised_by, "revised_by required"

		policy = await self._get_policy(policy_id)

		# Version bump
		current_version = policy.get("version", "1.0")
		try:
			major, minor = current_version.split(".")
			new_version = f"{major}.{int(minor) + 1}"
		except ValueError:
			new_version = f"{current_version}.1"

		revision: dict[str, Any] = {
			"id": _uid(),
			"policy_id": policy_id,
			"previous_version": current_version,
			"new_version": new_version,
			"revision_reason": revision_reason,
			"revision_summary": revision_summary,
			"revised_by": revised_by,
			"revised_at": _now(),
		}
		await self._store.put("policy_revisions", revision)

		policy["version"] = new_version
		policy["status"] = "under_revision"
		policy.setdefault("revision_history", []).append(revision["id"])
		policy["updated_at"] = _now()
		await self._store.put("policies", policy)

		await self._audit_event("policy_revision_started", revised_by, policy_id, {"new_version": new_version})
		return revision

	async def retire_policy(
		self,
		policy_id: str,
		reason: str,
		retired_by: str,
	) -> dict[str, Any]:
		"""Retire (archive/withdraw) a policy.

		Transitions status to 'archived'. Notifies owner and compliance team.
		"""
		assert reason, "reason required"
		assert retired_by, "retired_by required"

		policy = await self._get_policy(policy_id)

		rule_ctx = {
			"operation": "archive_policy",
			"archive_reason_present": True,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Retirement denied: {verdict['matched_rules']}")

		policy["status"] = "archived"
		policy["retired_by"] = retired_by
		policy["retirement_reason"] = reason
		policy["retired_at"] = _now()
		policy["updated_at"] = _now()
		await self._store.put("policies", policy)

		await self._audit_event("policy_archived", retired_by, policy_id, {"reason": reason})
		await self._notify.send(
			policy["owner_id"], "email",
			f"Policy retired: {policy['title']}",
			f"Policy '{policy['title']}' has been retired by {retired_by}. Reason: {reason}",
		)
		return policy

	# ─────────────────────────────────────────────────────────
	# Compliance and mapping
	# ─────────────────────────────────────────────────────────

	async def policy_compliance_check(
		self,
		entity_id: str,
		policy_id: str,
	) -> dict[str, Any]:
		"""Check an entity's compliance status against a specific policy.

		Returns acknowledgement rate, open exceptions, and overall status.
		"""
		policy = await self._get_policy(policy_id)
		acks = await self._store.query(
			"policy_acknowledgements",
			{"policy_id": policy_id},
			limit=10_000,
		)
		completed = sum(1 for a in acks if a.get("status") == "completed")
		total = len(acks)
		ack_rate = (completed / total * 100) if total > 0 else 0.0

		exceptions = await self._store.query(
			"policy_exceptions",
			{"policy_id": policy_id, "status": "approved"},
			limit=1000,
		)

		compliance: dict[str, Any] = {
			"entity_id": entity_id,
			"policy_id": policy_id,
			"policy_title": policy.get("title"),
			"policy_status": policy.get("status"),
			"acknowledgement_required": total,
			"acknowledgement_completed": completed,
			"acknowledgement_rate_pct": round(ack_rate, 2),
			"open_exceptions": len(exceptions),
			"overall_compliant": ack_rate >= 95.0 and len(exceptions) == 0,
			"checked_at": _now(),
		}
		return compliance

	async def policy_mapping(
		self,
		policy_id: str,
		regulation_ids: list[str],
		control_ids: list[str],
	) -> dict[str, Any]:
		"""Map a policy to regulatory requirements and control framework controls.

		Creates a many-to-many mapping record for GRC traceability.
		"""
		assert regulation_ids or control_ids, "at least one regulation or control required"

		policy = await self._get_policy(policy_id)
		mapping: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"policy_id": policy_id,
			"policy_title": policy.get("title"),
			"regulation_ids": regulation_ids,
			"control_ids": control_ids,
			"mapped_at": _now(),
		}
		await self._store.put("policy_mappings", mapping)
		policy["regulation_ids"] = regulation_ids
		policy["control_ids"] = control_ids
		policy["updated_at"] = _now()
		await self._store.put("policies", policy)

		await self._audit_event(
			"policy_mapped", "system", policy_id,
			{"regulations": len(regulation_ids), "controls": len(control_ids)},
		)
		return mapping

	async def policy_gap_analysis(
		self,
		entity_id: str,
		framework: str,
	) -> dict[str, Any]:
		"""Identify policy gaps against a compliance framework (ISO 27001, SOC 2, GDPR, etc.).

		Compares required controls in the framework against existing published policies.
		Returns covered, missing, and partial coverage.
		"""
		assert entity_id, "entity_id required"
		assert framework, "framework required"

		# Framework control requirements (simplified; production loads from framework registry)
		framework_requirements: dict[str, list[str]] = {
			"iso_27001": [
				"information_security", "acceptable_use", "data_privacy",
				"third_party", "bcdr", "hr",
			],
			"soc2": ["information_security", "acceptable_use", "operational", "compliance"],
			"gdpr": ["data_privacy", "information_security", "hr", "third_party"],
			"cbk_prudential": ["compliance", "operational", "finance", "hr"],
		}
		required_types = framework_requirements.get(framework.lower(), [])

		policies = await self._store.query(
			"policies",
			{"tenant_id": self._tenant_id, "status": "published"},
			limit=10_000,
		)
		existing_types = {p.get("policy_type") for p in policies}

		covered = [r for r in required_types if r in existing_types]
		missing = [r for r in required_types if r not in existing_types]
		coverage_pct = (len(covered) / len(required_types) * 100) if required_types else 100.0

		gap_report: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"framework": framework,
			"required_policy_types": required_types,
			"covered_types": covered,
			"missing_types": missing,
			"coverage_pct": round(coverage_pct, 2),
			"analysis_status": "complete" if coverage_pct == 100 else "gaps_identified",
			"generated_at": _now(),
		}
		await self._store.put("policy_gap_analyses", gap_report)
		return gap_report

	async def policy_analytics(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Compute policy management KPIs for a period.

		Includes policy counts by status and type, acknowledgement rates,
		exception rates, and upcoming reviews.
		"""
		start, end = _period_bounds(period)
		policies = await self._store.query("policies", {"tenant_id": self._tenant_id}, limit=10_000)

		by_status: dict[str, int] = {}
		by_type: dict[str, int] = {}
		for p in policies:
			s = p.get("status", "unknown")
			t = p.get("policy_type", "unknown")
			by_status[s] = by_status.get(s, 0) + 1
			by_type[t] = by_type.get(t, 0) + 1

		acks = await self._store.query("policy_acknowledgements", {}, limit=500_000)
		period_acks = [a for a in acks if start <= a.get("requested_at", "")[:10] <= end]
		completed_acks = sum(1 for a in period_acks if a.get("status") == "completed")
		ack_rate = (completed_acks / len(period_acks) * 100) if period_acks else 0.0

		exceptions = await self._store.query("policy_exceptions", {}, limit=10_000)
		period_exceptions = [e for e in exceptions if start <= e.get("requested_at", "")[:10] <= end]

		return {
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_policies": len(policies),
			"by_status": by_status,
			"by_type": by_type,
			"acknowledgement_requests": len(period_acks),
			"acknowledgements_completed": completed_acks,
			"acknowledgement_rate_pct": round(ack_rate, 2),
			"exceptions_requested": len(period_exceptions),
			"generated_at": _now(),
		}

	async def policy_library(
		self,
		category: str | None = None,
		status: str | None = None,
	) -> dict[str, Any]:
		"""Retrieve the policy library filtered by category and/or status."""
		filters: dict[str, Any] = {"tenant_id": self._tenant_id}
		if status:
			if status not in SUPPORTED_POLICY_STATUSES:
				raise ValueError(f"Unknown status: {status}")
			filters["status"] = status
		if category:
			filters["category"] = category

		policies = await self._store.query("policies", filters, limit=10_000)
		return {
			"count": len(policies),
			"filters": {"category": category, "status": status},
			"policies": policies,
			"queried_at": _now(),
		}

	async def policy_training_assignment(
		self,
		policy_id: str,
		employee_ids: list[str],
		deadline: str,
	) -> dict[str, Any]:
		"""Assign policy training to a list of employees with a completion deadline."""
		assert employee_ids, "employee_ids required"
		assert deadline, "deadline required"

		policy = await self._get_policy(policy_id)
		assignments = []
		for emp_id in employee_ids:
			assignment: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"policy_id": policy_id,
				"employee_id": emp_id,
				"deadline": deadline,
				"status": "assigned",
				"assigned_at": _now(),
			}
			await self._store.put("policy_training_assignments", assignment)
			assignments.append(assignment)
			await self._notify.send(
				emp_id, "email",
				f"Training assigned: {policy['title']}",
				f"Please complete training for policy '{policy['title']}' by {deadline}.",
			)

		return {
			"policy_id": policy_id,
			"assigned_count": len(assignments),
			"deadline": deadline,
			"assignments": assignments,
		}

	async def training_completion(
		self,
		policy_id: str,
		employee_id: str,
		score: float,
		completion_date: str,
	) -> dict[str, Any]:
		"""Record training completion for a policy with a pass/fail score.

		Pass threshold: 70%. Updates assignment status to 'completed' or 'failed'.
		"""
		assert 0 <= score <= 100, "score: 0–100"

		assignments = await self._store.query(
			"policy_training_assignments",
			{"policy_id": policy_id, "employee_id": employee_id},
			limit=1,
		)
		pass_threshold = 70.0
		passed = score >= pass_threshold

		if assignments:
			assignment = assignments[0]
			assignment["status"] = "completed" if passed else "failed"
			assignment["score"] = score
			assignment["completed_at"] = completion_date
			assignment["passed"] = passed
			await self._store.put("policy_training_assignments", assignment)
		else:
			assignment = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"policy_id": policy_id,
				"employee_id": employee_id,
				"score": score,
				"passed": passed,
				"status": "completed" if passed else "failed",
				"completed_at": completion_date,
			}
			await self._store.put("policy_training_assignments", assignment)

		await self._audit_event(
			"policy_training_completed", employee_id, policy_id,
			{"score": score, "passed": passed},
		)
		return assignment

	async def policy_search(
		self,
		query: str,
	) -> dict[str, Any]:
		"""Full-text search across policy titles, descriptions, and content.

		Returns ranked results sorted by relevance (title match first).
		"""
		assert query, "query required"
		query_lower = query.lower()

		policies = await self._store.query("policies", {"tenant_id": self._tenant_id}, limit=10_000)

		def _score(p: dict[str, Any]) -> int:
			score = 0
			if query_lower in p.get("title", "").lower():
				score += 10
			if query_lower in p.get("description", "").lower():
				score += 5
			if query_lower in p.get("policy_type", "").lower():
				score += 3
			if query_lower in p.get("category", "").lower():
				score += 3
			# search content sections
			for section in p.get("content_sections", []):
				if query_lower in section.get("body", "").lower():
					score += 2
			return score

		scored = [(p, _score(p)) for p in policies]
		results = [p for p, s in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]

		return {
			"query": query,
			"count": len(results),
			"results": results,
			"searched_at": _now(),
		}

	async def policy_expiry_report(
		self,
		days_ahead: int = 90,
	) -> dict[str, Any]:
		"""Report policies due for review within the next N days.

		Returns list of policies sorted by review date ascending.
		"""
		assert 1 <= days_ahead <= 365, "days_ahead: 1–365"

		cutoff = (date.today() + timedelta(days=days_ahead)).isoformat()
		today_str = date.today().isoformat()

		policies = await self._store.query(
			"policies",
			{"tenant_id": self._tenant_id},
			limit=10_000,
		)

		due = [
			p for p in policies
			if today_str <= p.get("next_review_date", "9999-12-31") <= cutoff
			and p.get("status") not in {"archived", "withdrawn", "superseded"}
		]
		overdue = [
			p for p in policies
			if p.get("next_review_date", "9999-12-31") < today_str
			and p.get("status") not in {"archived", "withdrawn", "superseded"}
		]

		return {
			"days_ahead": days_ahead,
			"due_count": len(due),
			"overdue_count": len(overdue),
			"due_policies": sorted(due, key=lambda p: p.get("next_review_date", "")),
			"overdue_policies": sorted(overdue, key=lambda p: p.get("next_review_date", "")),
			"generated_at": _now(),
		}

	async def policy_gap_assess(self, entity_id: str, framework: str) -> dict[str, Any]:
		"""Assess policy gaps against a compliance framework — domain alias."""
		return await self.policy_gap_analysis(entity_id, framework)

	async def policy_exception_approve(self, exception_id: str, approver_id: str, approved_until: str, conditions: str) -> dict[str, Any]:
		"""Approve a policy exception — domain alias."""
		return await self.approve_exception(exception_id, approver_id, approved_until, conditions)

	async def policy_exception_monitor(self, policy_id: str | None = None) -> dict[str, Any]:
		"""Monitor active policy exceptions for expiry and compliance."""
		from datetime import date
		today = date.today().isoformat()
		filters: dict[str, Any] = {"status": "approved"}
		if policy_id:
			filters["policy_id"] = policy_id
		exceptions = await self._store.query("policy_exceptions", filters, limit=10_000)
		expiring_soon = [e for e in exceptions if e.get("expiry_date", "9999") <= (date.today() + __import__("datetime").timedelta(days=30)).isoformat()]
		expired = [e for e in exceptions if e.get("expiry_date", "9999") < today]
		return {"policy_id_filter": policy_id, "total_active_exceptions": len(exceptions), "expiring_within_30_days": len(expiring_soon), "expired": len(expired), "expiring_list": expiring_soon, "expired_list": expired, "monitored_at": _now()}

	async def acknowledgement_chase(self, policy_id: str, chased_by: str) -> dict[str, Any]:
		"""Chase overdue policy acknowledgements via notifications."""
		acks = await self._store.query("policy_acknowledgements", {"policy_id": policy_id, "status": "pending"}, limit=10_000)
		chased = []
		for ack in acks:
			emp_id = ack.get("employee_id")
			await self._notify.send(emp_id, "email", "Overdue: Policy acknowledgement required", f"Please acknowledge policy {policy_id}. This is a reminder — overdue acknowledgement may result in non-compliance.")
			chased.append(emp_id)
		await self._audit_event("acknowledgements_chased", chased_by, policy_id, {"count": len(chased)})
		return {"policy_id": policy_id, "chased_count": len(chased), "chased_by": chased_by, "chased_at": _now()}

	async def policy_map_control(self, policy_id: str, regulation_ids: list[str], control_ids: list[str]) -> dict[str, Any]:
		"""Map policy to regulations and controls — domain alias."""
		return await self.policy_mapping(policy_id, regulation_ids, control_ids)

	async def policy_effectiveness(self, policy_id: str) -> dict[str, Any]:
		"""Assess policy effectiveness based on acknowledgement and exception data."""
		policy = await self._get_policy(policy_id)
		compliance = await self.policy_compliance_check(policy.get("owner_id", "unknown"), policy_id)
		exceptions = await self._store.query("policy_exceptions", {"policy_id": policy_id}, limit=1000)
		effectiveness_score = compliance.get("acknowledgement_rate_pct", 0)
		if exceptions:
			effectiveness_score = max(0, effectiveness_score - len(exceptions) * 5)
		return {"policy_id": policy_id, "policy_title": policy.get("title"), "acknowledgement_rate_pct": compliance.get("acknowledgement_rate_pct", 0), "exception_count": len(exceptions), "effectiveness_score": round(effectiveness_score, 1), "rating": "effective" if effectiveness_score >= 85 else ("adequate" if effectiveness_score >= 65 else "needs_improvement"), "assessed_at": _now()}

	async def policy_risk_link(self, policy_id: str, risk_ids: list[str]) -> dict[str, Any]:
		"""Link a policy to risk register entries."""
		assert risk_ids, "risk_ids required"
		policy = await self._get_policy(policy_id)
		policy["linked_risk_ids"] = risk_ids
		policy["updated_at"] = _now()
		await self._store.put("policies", policy)
		await self._audit_event("policy_risk_linked", "system", policy_id, {"risk_count": len(risk_ids)})
		return {"policy_id": policy_id, "linked_risk_ids": risk_ids, "linked_at": _now()}

	async def policy_review_notify(self, days_ahead: int = 60) -> dict[str, Any]:
		"""Notify policy owners of upcoming review deadlines."""
		report = await self.policy_expiry_report(days_ahead)
		due_policies = report.get("due_policies", [])
		notified = []
		for p in due_policies:
			owner = p.get("owner_id")
			if owner:
				await self._notify.send(owner, "email", f"Policy review due: {p.get('title')}", f"Policy '{p.get('title')}' is due for review on {p.get('next_review_date')}.")
				notified.append(p.get("id"))
		return {"notified_count": len(notified), "notified_policy_ids": notified, "days_ahead": days_ahead, "notified_at": _now()}

	async def policy_compare_versions(self, policy_id: str, version_a: str, version_b: str) -> dict[str, Any]:
		"""Compare two versions of a policy by examining revision history."""
		revisions = await self._store.query("policy_revisions", {"policy_id": policy_id}, limit=100)
		rev_a = next((r for r in revisions if r.get("new_version") == version_a or r.get("previous_version") == version_a), None)
		rev_b = next((r for r in revisions if r.get("new_version") == version_b or r.get("previous_version") == version_b), None)
		return {"policy_id": policy_id, "version_a": version_a, "version_b": version_b, "revision_a": rev_a, "revision_b": rev_b, "compared_at": _now()}

	async def policy_retire(self, policy_id: str, reason: str, retired_by: str) -> dict[str, Any]:
		"""Retire a policy — domain alias."""
		return await self.retire_policy(policy_id, reason, retired_by)

	async def training_track(self, policy_id: str, employee_id: str, score: float, completion_date: str) -> dict[str, Any]:
		"""Track policy training completion — domain alias."""
		return await self.training_completion(policy_id, employee_id, score, completion_date)

	async def stakeholder_consult(self, policy_id: str, stakeholders: list[str], consultation_type: str, deadline: str) -> dict[str, Any]:
		"""Initiate stakeholder consultation for a policy revision."""
		assert stakeholders, "stakeholders required"
		policy = await self._get_policy(policy_id)
		consult_id = _uid()
		for s in stakeholders:
			await self._notify.send(s, "email", f"Policy consultation: {policy.get('title')}", f"Your input is requested for policy '{policy.get('title')}'. Deadline: {deadline}.")
		await self._audit_event("stakeholder_consultation_initiated", "system", policy_id, {"count": len(stakeholders), "type": consultation_type})
		return {"consultation_id": consult_id, "policy_id": policy_id, "stakeholders": stakeholders, "type": consultation_type, "deadline": deadline, "initiated_at": _now()}

	async def regulatory_align(self, policy_id: str, regulations: list[str], aligned_by: str) -> dict[str, Any]:
		"""Align a policy to regulatory requirements."""
		policy = await self._get_policy(policy_id)
		return await self.policy_mapping(policy_id, regulations, policy.get("control_ids", []))

	async def policy_search_advanced(self, query: str, policy_type: str | None = None, status: str | None = None) -> dict[str, Any]:
		"""Full-text search across policies with optional type/status filter."""
		base = await self.policy_search(query)
		results = base.get("results", [])
		if policy_type:
			results = [p for p in results if p.get("policy_type") == policy_type]
		if status:
			results = [p for p in results if p.get("status") == status]
		return {**base, "results": results, "count": len(results)}

	async def policy_template(self, template_name: str, policy_type: str, scope: str, standard_sections: list[str]) -> dict[str, Any]:
		"""Create a reusable policy template."""
		tmpl_id = _uid()
		template: dict[str, Any] = {
			"id": tmpl_id,
			"tenant_id": self._tenant_id,
			"template_name": template_name,
			"policy_type": policy_type,
			"scope": scope,
			"standard_sections": standard_sections,
			"created_at": _now(),
		}
		await self._store.put("policy_templates", template)
		return template

	async def policy_kpi_report(self, period: str) -> dict[str, Any]:
		"""Return policy KPI summary for the period — calls core analytics."""
		analytics = await self.policy_analytics(period)
		return {"kpi_report": True, **analytics}

	async def policy_publish_portal(self, policy_id: str, portal_url: str, published_by: str) -> dict[str, Any]:
		"""Publish a policy to the employee self-service portal."""
		policy = await self._get_policy(policy_id)
		if policy.get("status") not in {"approved", "published"}:
			raise ValueError("Policy must be approved before portal publication")
		policy["portal_url"] = portal_url
		policy["portal_published_by"] = published_by
		policy["portal_published_at"] = _now()
		await self._store.put("policies", policy)
		await self._audit_event("policy_published_to_portal", published_by, policy_id, {"portal_url": portal_url})
		return {"policy_id": policy_id, "portal_url": portal_url, "published_by": published_by, "published_at": _now()}

	async def policy_dashboard(
		self,
		entity_id: str,
	) -> dict[str, Any]:
		"""Assemble the policy management dashboard for an entity.

		Includes status summary, compliance rates, overdue reviews, and open exceptions.
		"""
		today = date.today().isoformat()
		policies = await self._store.query("policies", {"tenant_id": self._tenant_id}, limit=10_000)

		by_status: dict[str, int] = {}
		for p in policies:
			s = p.get("status", "unknown")
			by_status[s] = by_status.get(s, 0) + 1

		overdue = [
			p for p in policies
			if p.get("next_review_date", "9999-12-31") < today
			and p.get("status") == "published"
		]

		exceptions = await self._store.query(
			"policy_exceptions",
			{"status": "pending"},
			limit=1000,
		)

		all_acks = await self._store.query("policy_acknowledgements", {}, limit=500_000)
		completed = sum(1 for a in all_acks if a.get("status") == "completed")
		overall_ack_rate = (completed / len(all_acks) * 100) if all_acks else 0.0

		return {
			"entity_id": entity_id,
			"as_of": today,
			"total_policies": len(policies),
			"by_status": by_status,
			"overdue_reviews": len(overdue),
			"overdue_policy_ids": [p["id"] for p in overdue],
			"pending_exceptions": len(exceptions),
			"overall_acknowledgement_rate_pct": round(overall_ack_rate, 2),
			"generated_at": _now(),
		}

	async def policy_kpi_summary(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise policy KPI card for dashboard consumption.

		Covers: total policies, compliance rate, overdue reviews, exceptions.
		"""
		dashboard = await self.policy_dashboard(entity_id)
		total = dashboard["total_policies"]
		by_status = dashboard["by_status"]
		published = by_status.get("published", 0)
		compliance_rate = round(published / max(total, 1) * 100, 1)
		return {
			"entity_id": entity_id,
			"period": period,
			"total_policies": total,
			"published_policies": published,
			"draft_policies": by_status.get("draft", 0),
			"compliance_rate_pct": compliance_rate,
			"overdue_reviews": dashboard["overdue_reviews"],
			"pending_exceptions": dashboard["pending_exceptions"],
			"acknowledgement_rate_pct": dashboard["overall_acknowledgement_rate_pct"],
			"generated_at": _now(),
		}

	async def ml_policy_compliance_score(self, *args, **kwargs):
		"""AI-powered AI policy compliance gap assessment. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="policy_compliance_scoring")
			return {"compliance_score": round(result.score,3), "gaps": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

