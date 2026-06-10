"""IncidentComplianceService — GRC incident and compliance management.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .capability_contract import (
	CAPABILITY_ID,
	CAPABILITY_VERSION,
	SUPPORTED_INCIDENT_TYPES,
	SUPPORTED_INCIDENT_SEVERITIES,
	SUPPORTED_INCIDENT_STATUSES,
	SUPPORTED_REGULATORY_WINDOWS,
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


class IncidentComplianceService:
	"""GRC incident lifecycle, compliance testing, corrective actions, regulatory
	notifications, BCP activation, and post-incident review management.

	Usage (standalone)::

		svc = IncidentComplianceService()
		inc = await svc.report_incident("ENT-1", "security_breach", "Phishing attack", ...)

	Usage (platform)::

		svc = IncidentComplianceService(auth=AuthService.from_env())
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

	async def _get_incident(self, incident_id: str) -> dict[str, Any]:
		rec = await self._store.get("incidents", incident_id)
		if rec is None:
			raise ValueError(f"Incident not found: {incident_id}")
		return rec

	# ─────────────────────────────────────────────────────────
	# Incident lifecycle
	# ─────────────────────────────────────────────────────────

	async def report_incident(
		self,
		entity_id: str,
		incident_type: str,
		description: str,
		severity: str,
		affected_systems: list[str],
		reported_by: str,
		*,
		title: str | None = None,
		detection_time: str | None = None,
	) -> dict[str, Any]:
		"""Report a new security or compliance incident.

		Validates type and severity, assigns detection time, and creates the
		incident record. Critical incidents trigger immediate triage notifications.
		Emits ``incident_reported`` event.
		"""
		assert entity_id, "entity_id required"
		assert description, "description required"
		assert reported_by, "reported_by required"

		if incident_type not in SUPPORTED_INCIDENT_TYPES:
			raise ValueError(f"Unsupported incident type: {incident_type}. Valid: {SUPPORTED_INCIDENT_TYPES}")
		if severity not in SUPPORTED_INCIDENT_SEVERITIES:
			raise ValueError(f"Unsupported severity: {severity}. Valid: {SUPPORTED_INCIDENT_SEVERITIES}")

		rule_ctx = {
			"operation": "report_incident",
			"tenant_context_present": True,
			"title_present": bool(title),
			"incident_type_supported": True,
			"incident_severity_supported": True,
			"reporter_present": True,
			"owner_present": True,
			"detection_time_present": True,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Incident report denied: {verdict['matched_rules']}")

		incident: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"title": title or f"{incident_type.replace('_', ' ').title()} — {_now()[:10]}",
			"incident_type": incident_type,
			"description": description,
			"severity": severity,
			"affected_systems": affected_systems,
			"reported_by": reported_by,
			"owner_id": reported_by,
			"detection_time": detection_time or _now(),
			"status": "new",
			"timeline": [{"timestamp": _now(), "event": "incident_reported", "actor": reported_by}],
			"root_cause": None,
			"lessons_learned": None,
			"regulatory_breach": False,
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("incidents", incident)

		await self._audit_event(
			"incident_reported", reported_by, incident["id"],
			{"incident_type": incident_type, "severity": severity},
		)

		# Critical incidents: immediate notification
		if severity == "critical":
			await self._notify.send(
				"incident-response@datacraft.co.ke", "email",
				f"CRITICAL INCIDENT: {incident['title']}",
				f"Critical incident reported by {reported_by}.\nType: {incident_type}\nSystems: {affected_systems}\nDescription: {description}",
			)
			await self._notify.send(
				"ciso@datacraft.co.ke", "email",
				f"CRITICAL INCIDENT ALERT: {incident['title']}",
				f"Immediate action required. Incident ID: {incident['id']}",
			)

		return incident

	async def incident_triage(
		self,
		incident_id: str,
		incident_commander_id: str,
		priority: str,
		initial_response: str,
	) -> dict[str, Any]:
		"""Triage an incident: assign commander, set priority, and record initial response.

		Priority: P1 (critical) | P2 (high) | P3 (medium) | P4 (low).
		Transitions status to 'triaged'.
		"""
		assert incident_commander_id, "incident_commander_id required"
		assert priority in {"P1", "P2", "P3", "P4"}, "priority: P1 | P2 | P3 | P4"
		assert initial_response, "initial_response required"

		incident = await self._get_incident(incident_id)
		if incident.get("status") == "closed":
			raise ValueError("Cannot triage a closed incident")

		incident["status"] = "triaged"
		incident["incident_commander_id"] = incident_commander_id
		incident["priority"] = priority
		incident["initial_response"] = initial_response
		incident["triaged_at"] = _now()
		incident["owner_id"] = incident_commander_id
		incident.setdefault("timeline", []).append({
			"timestamp": _now(),
			"event": "incident_triaged",
			"actor": incident_commander_id,
			"detail": f"Priority: {priority}",
		})
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)

		await self._audit_event(
			"incident_triaged", incident_commander_id, incident_id,
			{"priority": priority},
		)
		return incident

	async def incident_investigation(
		self,
		incident_id: str,
		findings: str,
		evidence: list[dict[str, Any]],
		investigator_id: str,
	) -> dict[str, Any]:
		"""Record investigation findings and evidence for an incident.

		Evidence items: {type, description, hash, collected_at}.
		Transitions status to 'in_investigation'.
		"""
		assert findings, "findings required"
		assert investigator_id, "investigator_id required"

		incident = await self._get_incident(incident_id)

		# Store evidence records
		evidence_ids = []
		for ev in evidence:
			ev_rec: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"incident_id": incident_id,
				"evidence_type": ev.get("type", "document"),
				"description": ev.get("description", ""),
				"hash": ev.get("hash", ""),
				"collected_at": ev.get("collected_at", _now()),
				"collected_by": investigator_id,
				"encrypted": True,
				"chain_of_custody": [{"actor": investigator_id, "action": "collected", "timestamp": _now()}],
				"retention_days": 365,
			}
			await self._store.put("incident_evidence", ev_rec)
			evidence_ids.append(ev_rec["id"])

		incident["investigation_findings"] = findings
		incident["evidence_ids"] = incident.get("evidence_ids", []) + evidence_ids
		incident["investigator_id"] = investigator_id
		incident["status"] = "in_investigation"
		incident.setdefault("timeline", []).append({
			"timestamp": _now(),
			"event": "investigation_started",
			"actor": investigator_id,
		})
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)

		await self._audit_event(
			"incident_investigated", investigator_id, incident_id,
			{"evidence_count": len(evidence_ids)},
		)
		return incident

	async def root_cause_analysis(
		self,
		incident_id: str,
		rca_method: str,
		root_causes: list[str],
		contributing_factors: list[str],
	) -> dict[str, Any]:
		"""Record root cause analysis for an incident.

		Supported RCA methods: 5_whys | fishbone | fault_tree | timeline | bow_tie.
		"""
		assert rca_method in {"5_whys", "fishbone", "fault_tree", "timeline", "bow_tie"}, (
			"rca_method: 5_whys | fishbone | fault_tree | timeline | bow_tie"
		)
		assert root_causes, "root_causes required"

		incident = await self._get_incident(incident_id)

		rca: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"rca_method": rca_method,
			"root_causes": root_causes,
			"contributing_factors": contributing_factors,
			"completed_at": _now(),
		}
		await self._store.put("incident_rcas", rca)

		incident["root_cause"] = root_causes[0] if root_causes else None
		incident["root_cause_analysis_id"] = rca["id"]
		incident.setdefault("timeline", []).append({
			"timestamp": _now(),
			"event": "rca_completed",
			"detail": f"Method: {rca_method}, {len(root_causes)} root causes identified",
		})
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)

		await self._audit_event(
			"incident_rca_completed", "system", incident_id,
			{"method": rca_method, "root_cause_count": len(root_causes)},
		)
		return rca

	async def corrective_action(
		self,
		incident_id: str,
		action_type: str,
		description: str,
		owner_id: str,
		deadline: str,
	) -> dict[str, Any]:
		"""Create a corrective or preventive action linked to an incident.

		Action types: corrective | preventive | systemic | immediate.
		"""
		assert action_type in {"corrective", "preventive", "systemic", "immediate"}, (
			"action_type: corrective | preventive | systemic | immediate"
		)
		assert description, "description required"
		assert owner_id, "owner_id required"
		assert deadline, "deadline required"

		incident = await self._get_incident(incident_id)

		action: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"action_type": action_type,
			"description": description,
			"owner_id": owner_id,
			"deadline": deadline,
			"progress_pct": 0.0,
			"status": "open",
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("corrective_actions", action)

		incident.setdefault("corrective_action_ids", []).append(action["id"])
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)

		await self._notify.send(
			owner_id, "email",
			f"Corrective action assigned: {incident['title']}",
			f"You have been assigned a {action_type} action for incident {incident_id}. Deadline: {deadline}\n{description}",
		)
		await self._audit_event(
			"corrective_action_created", owner_id, incident_id,
			{"action_type": action_type, "deadline": deadline},
		)
		return action

	async def corrective_action_update(
		self,
		action_id: str,
		progress_pct: float,
		notes: str,
		updated_by: str,
	) -> dict[str, Any]:
		"""Update progress on a corrective action."""
		assert 0 <= progress_pct <= 100, "progress_pct: 0–100"
		assert updated_by, "updated_by required"

		action = await self._store.get("corrective_actions", action_id)
		if action is None:
			raise ValueError(f"Corrective action not found: {action_id}")

		action["progress_pct"] = progress_pct
		action["last_notes"] = notes
		action["last_updated_by"] = updated_by
		action["status"] = "closed" if progress_pct >= 100 else "open"
		action["updated_at"] = _now()
		await self._store.put("corrective_actions", action)

		await self._audit_event(
			"corrective_action_updated", updated_by, action_id,
			{"progress_pct": progress_pct, "status": action["status"]},
		)
		return action

	async def close_incident(
		self,
		incident_id: str,
		resolution: str,
		lessons_learned: str,
		closed_by: str,
	) -> dict[str, Any]:
		"""Close a resolved incident with resolution summary and lessons learned.

		Requires root cause to be recorded. High/critical incidents require
		a post-incident review before closure.
		"""
		assert resolution, "resolution required"
		assert lessons_learned, "lessons_learned required"
		assert closed_by, "closed_by required"

		incident = await self._get_incident(incident_id)

		if not incident.get("root_cause"):
			raise ValueError("Root cause must be recorded before closing an incident")

		rule_ctx = {
			"operation": "close_incident",
			"root_cause_present": True,
			"high_or_critical_incident": incident.get("severity") in {"high", "critical"},
			"post_review_recorded": bool(incident.get("post_incident_review_id")),
			"regulatory_breach": incident.get("regulatory_breach", False),
			"notification_sent": bool(incident.get("regulatory_notification_sent")),
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Incident closure denied: {verdict['matched_rules']}")

		incident["status"] = "closed"
		incident["resolution"] = resolution
		incident["lessons_learned"] = lessons_learned
		incident["closed_by"] = closed_by
		incident["closed_at"] = _now()
		incident.setdefault("timeline", []).append({
			"timestamp": _now(),
			"event": "incident_closed",
			"actor": closed_by,
		})
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)

		# Add to lessons learned library
		lesson: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"incident_type": incident.get("incident_type"),
			"severity": incident.get("severity"),
			"lessons_learned": lessons_learned,
			"created_at": _now(),
		}
		await self._store.put("lessons_learned", lesson)

		await self._audit_event("incident_closed", closed_by, incident_id, {"resolution": resolution[:100]})
		return incident

	async def regulatory_notification(
		self,
		incident_id: str,
		regulator: str,
		notification_type: str,
		deadline: str,
	) -> dict[str, Any]:
		"""Send a regulatory notification for a confirmed breach incident.

		Validates the notification window (GDPR: 72h, PCI DSS: 24h) and logs dispatch.
		"""
		assert regulator, "regulator required"
		assert notification_type in {"initial", "update", "final"}, (
			"notification_type: initial | update | final"
		)

		incident = await self._get_incident(incident_id)
		detection_time = incident.get("detection_time", _now())
		hours_elapsed = (
			datetime.now(timezone.utc) - datetime.fromisoformat(detection_time)
		).total_seconds() / 3600

		# Check regulatory window
		framework = regulator.lower()
		window_hours = SUPPORTED_REGULATORY_WINDOWS.get(framework, SUPPORTED_REGULATORY_WINDOWS["default"])
		window_exceeded = hours_elapsed > window_hours

		notification: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"regulator": regulator,
			"notification_type": notification_type,
			"deadline": deadline,
			"hours_elapsed_since_detection": round(hours_elapsed, 2),
			"window_hours": window_hours,
			"window_exceeded": window_exceeded,
			"status": "sent",
			"sent_at": _now(),
		}
		await self._store.put("regulatory_notifications", notification)

		incident["regulatory_notification_sent"] = True
		incident["regulatory_breach"] = True
		incident.setdefault("regulatory_notifications", []).append(notification["id"])
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)

		await self._notify.send(
			f"compliance@datacraft.co.ke", "email",
			f"Regulatory notification sent to {regulator}",
			f"Incident {incident_id}: {notification_type} notification sent to {regulator}. Window exceeded: {window_exceeded}",
		)
		await self._audit_event(
			"regulatory_notification_sent", "compliance", incident_id,
			{"regulator": regulator, "window_exceeded": window_exceeded},
		)
		return notification

	# ─────────────────────────────────────────────────────────
	# Compliance testing
	# ─────────────────────────────────────────────────────────

	async def compliance_test(
		self,
		entity_id: str,
		control_id: str,
		test_type: str,
		test_date: str,
		result: str,
		tester_id: str,
	) -> dict[str, Any]:
		"""Record a compliance control test result.

		Test types: design | operating_effectiveness | walkthrough | inquiry.
		Results: pass | fail | partial.
		"""
		assert test_type in {"design", "operating_effectiveness", "walkthrough", "inquiry"}, (
			"test_type: design | operating_effectiveness | walkthrough | inquiry"
		)
		assert result in {"pass", "fail", "partial"}, "result: pass | fail | partial"
		assert tester_id, "tester_id required"

		test_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"control_id": control_id,
			"test_type": test_type,
			"test_date": test_date,
			"result": result,
			"tester_id": tester_id,
			"status": "completed",
			"created_at": _now(),
		}
		await self._store.put("compliance_tests", test_rec)

		if result == "fail":
			await self._notify.send(
				"compliance@datacraft.co.ke", "email",
				f"Compliance test FAILED: control {control_id}",
				f"Control {control_id} failed {test_type} test on {test_date}. Tester: {tester_id}",
			)

		await self._audit_event(
			"compliance_test_completed", tester_id, control_id,
			{"test_type": test_type, "result": result, "test_date": test_date},
		)
		return test_rec

	async def compliance_deficiency(
		self,
		control_id: str,
		deficiency_type: str,
		severity: str,
		identified_by: str,
	) -> dict[str, Any]:
		"""Record a compliance deficiency for a control.

		Deficiency types: design_gap | operating_ineffectiveness | absent_control.
		Severity: observation | significant | material_weakness.
		"""
		assert deficiency_type in {
			"design_gap", "operating_ineffectiveness", "absent_control"
		}, "deficiency_type: design_gap | operating_ineffectiveness | absent_control"
		assert severity in {"observation", "significant", "material_weakness"}, (
			"severity: observation | significant | material_weakness"
		)
		assert identified_by, "identified_by required"

		deficiency: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"control_id": control_id,
			"deficiency_type": deficiency_type,
			"severity": severity,
			"identified_by": identified_by,
			"status": "open",
			"created_at": _now(),
		}
		await self._store.put("compliance_deficiencies", deficiency)

		if severity == "material_weakness":
			await self._notify.send(
				"cfo@datacraft.co.ke", "email",
				f"Material weakness: control {control_id}",
				f"A material weakness has been identified for control {control_id}. Immediate remediation required.",
			)

		await self._audit_event(
			"compliance_deficiency_identified", identified_by, control_id,
			{"deficiency_type": deficiency_type, "severity": severity},
		)
		return deficiency

	async def remediation_plan(
		self,
		deficiency_id: str,
		remediation_actions: list[dict[str, Any]],
		deadline: str,
		owner_id: str,
	) -> dict[str, Any]:
		"""Create a remediation plan for a compliance deficiency."""
		assert remediation_actions, "remediation_actions required"
		assert deadline, "deadline required"
		assert owner_id, "owner_id required"

		deficiency = await self._store.get("compliance_deficiencies", deficiency_id)
		if deficiency is None:
			raise ValueError(f"Deficiency not found: {deficiency_id}")

		plan: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"deficiency_id": deficiency_id,
			"control_id": deficiency.get("control_id"),
			"remediation_actions": remediation_actions,
			"deadline": deadline,
			"owner_id": owner_id,
			"progress_pct": 0.0,
			"status": "active",
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("remediation_plans", plan)

		deficiency["remediation_plan_id"] = plan["id"]
		deficiency["status"] = "in_remediation"
		await self._store.put("compliance_deficiencies", deficiency)

		await self._audit_event(
			"remediation_plan_created", owner_id, deficiency_id,
			{"deadline": deadline, "action_count": len(remediation_actions)},
		)
		return plan

	async def compliance_calendar(
		self,
		entity_id: str,
		year: int,
	) -> dict[str, Any]:
		"""Generate a compliance calendar for an entity for a given year.

		Lists all scheduled compliance tests, reviews, and regulatory filings.
		"""
		assert entity_id, "entity_id required"
		assert 2020 <= year <= 2099, "year: 2020–2099"

		tests = await self._store.query(
			"compliance_tests",
			{"entity_id": entity_id},
			limit=10_000,
		)
		year_tests = [t for t in tests if t.get("test_date", "")[:4] == str(year)]

		quarterly_events = []
		for q in range(1, 5):
			sm = (q - 1) * 3 + 1
			quarterly_events.append({
				"quarter": f"Q{q}",
				"period": f"{year}-Q{q}",
				"events": [
					f"Quarterly compliance review — Q{q} {year}",
					f"KRI dashboard review — Q{q} {year}",
				],
			})

		return {
			"entity_id": entity_id,
			"year": year,
			"test_count": len(year_tests),
			"quarterly_events": quarterly_events,
			"annual_events": [
				f"Annual compliance assessment — {year}",
				f"Regulatory capital report — {year}",
				f"External audit coordination — {year}",
			],
			"generated_at": _now(),
		}

	async def compliance_score(
		self,
		entity_id: str,
		framework: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute an overall compliance score (0–100) for an entity against a framework.

		Score = (passed controls / total tested controls) × 100.
		"""
		start, end = _period_bounds(period)
		tests = await self._store.query(
			"compliance_tests",
			{"entity_id": entity_id},
			limit=10_000,
		)
		period_tests = [t for t in tests if start <= t.get("test_date", "")[:10] <= end]

		if not period_tests:
			return {
				"entity_id": entity_id,
				"framework": framework,
				"period": period,
				"score": 0.0,
				"tested_controls": 0,
				"passed": 0,
				"failed": 0,
				"message": "No tests found for period",
			}

		passed = sum(1 for t in period_tests if t.get("result") == "pass")
		failed = sum(1 for t in period_tests if t.get("result") == "fail")
		partial = sum(1 for t in period_tests if t.get("result") == "partial")
		score = ((passed + partial * 0.5) / len(period_tests)) * 100

		result: dict[str, Any] = {
			"entity_id": entity_id,
			"framework": framework,
			"period": period,
			"period_start": start,
			"period_end": end,
			"tested_controls": len(period_tests),
			"passed": passed,
			"partial": partial,
			"failed": failed,
			"score": round(score, 2),
			"rating": "excellent" if score >= 90 else "satisfactory" if score >= 70 else "needs_improvement",
			"generated_at": _now(),
		}
		await self._store.put("compliance_scores", result)
		return result

	async def incident_analytics(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute incident management analytics for a period."""
		start, end = _period_bounds(period)
		incidents = await self._store.query(
			"incidents",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)
		period_incs = [i for i in incidents if start <= i.get("created_at", "")[:10] <= end]

		by_type: dict[str, int] = {}
		by_severity: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for inc in period_incs:
			t = inc.get("incident_type", "unknown")
			s = inc.get("severity", "unknown")
			st = inc.get("status", "unknown")
			by_type[t] = by_type.get(t, 0) + 1
			by_severity[s] = by_severity.get(s, 0) + 1
			by_status[st] = by_status.get(st, 0) + 1

		closed = [i for i in period_incs if i.get("status") == "closed"]
		mttr_hours: list[float] = []
		for inc in closed:
			created = inc.get("created_at", _now())
			closed_at = inc.get("closed_at", _now())
			try:
				delta = datetime.fromisoformat(closed_at) - datetime.fromisoformat(created)
				mttr_hours.append(delta.total_seconds() / 3600)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		avg_mttr = sum(mttr_hours) / len(mttr_hours) if mttr_hours else 0.0

		return {
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_incidents": len(period_incs),
			"by_type": by_type,
			"by_severity": by_severity,
			"by_status": by_status,
			"closed_incidents": len(closed),
			"avg_mttr_hours": round(avg_mttr, 2),
			"generated_at": _now(),
		}

	async def compliance_dashboard(
		self,
		entity_id: str,
	) -> dict[str, Any]:
		"""Assemble the compliance management dashboard for an entity."""
		today = date.today().isoformat()
		incidents = await self._store.query(
			"incidents",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)
		open_incidents = [i for i in incidents if i.get("status") not in {"closed", "false_positive"}]
		critical_open = [i for i in open_incidents if i.get("severity") == "critical"]

		deficiencies = await self._store.query(
			"compliance_deficiencies",
			{"status": "open"},
			limit=1000,
		)
		actions = await self._store.query(
			"corrective_actions",
			{"status": "open"},
			limit=1000,
		)
		overdue_actions = [a for a in actions if a.get("deadline", "9999") < today]

		tests = await self._store.query("compliance_tests", {"entity_id": entity_id}, limit=10_000)
		recent_tests = [t for t in tests if t.get("test_date", "")[:7] == today[:7]]
		recent_pass_rate = (
			sum(1 for t in recent_tests if t.get("result") == "pass") / len(recent_tests) * 100
			if recent_tests else 0.0
		)

		return {
			"entity_id": entity_id,
			"as_of": today,
			"open_incidents": len(open_incidents),
			"critical_open_incidents": len(critical_open),
			"open_deficiencies": len(deficiencies),
			"open_corrective_actions": len(actions),
			"overdue_actions": len(overdue_actions),
			"this_month_pass_rate_pct": round(recent_pass_rate, 2),
			"generated_at": _now(),
		}

	async def lessons_learned_library(
		self,
		entity_id: str,
		incident_type: str | None = None,
	) -> dict[str, Any]:
		"""Retrieve the lessons learned library, optionally filtered by incident type."""
		lessons = await self._store.query(
			"lessons_learned",
			{"tenant_id": self._tenant_id},
			limit=10_000,
		)
		if incident_type:
			lessons = [l for l in lessons if l.get("incident_type") == incident_type]

		return {
			"entity_id": entity_id,
			"incident_type_filter": incident_type,
			"count": len(lessons),
			"lessons": lessons,
			"queried_at": _now(),
		}

	async def insurance_claim_trigger(
		self,
		incident_id: str,
		policy_id: str,
		estimated_loss: float,
	) -> dict[str, Any]:
		"""Trigger an insurance claim submission for an incident.

		Creates a claim record with estimated loss and required documentation checklist.
		"""
		assert policy_id, "policy_id required"
		assert estimated_loss >= 0, "estimated_loss must be non-negative"

		incident = await self._get_incident(incident_id)

		claim: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"insurance_policy_id": policy_id,
			"estimated_loss": estimated_loss,
			"incident_type": incident.get("incident_type"),
			"severity": incident.get("severity"),
			"documentation_checklist": [
				"Incident report",
				"Root cause analysis",
				"Financial loss evidence",
				"Timeline of events",
				"Remediation actions taken",
			],
			"status": "draft",
			"created_at": _now(),
		}
		await self._store.put("insurance_claims", claim)
		await self._notify.send(
			"finance@datacraft.co.ke", "email",
			f"Insurance claim initiated: incident {incident_id}",
			f"Estimated loss: KES {estimated_loss:,.2f}. Policy: {policy_id}. Please complete the claim documentation.",
		)
		await self._audit_event(
			"insurance_claim_triggered", "system", incident_id,
			{"policy_id": policy_id, "estimated_loss": estimated_loss},
		)
		return claim

	async def business_continuity_activation(
		self,
		incident_id: str,
		bcp_plan_id: str,
		activator_id: str,
	) -> dict[str, Any]:
		"""Activate a Business Continuity Plan for a major incident.

		Creates an activation record with timeline and notifies BCP team leads.
		"""
		assert bcp_plan_id, "bcp_plan_id required"
		assert activator_id, "activator_id required"

		incident = await self._get_incident(incident_id)
		bcp_plan = await self._store.get("bcp_plans", bcp_plan_id)
		plan_name = bcp_plan.get("name", bcp_plan_id) if bcp_plan else bcp_plan_id

		activation: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"bcp_plan_id": bcp_plan_id,
			"plan_name": plan_name,
			"activator_id": activator_id,
			"activation_time": _now(),
			"status": "active",
			"recovery_objectives": bcp_plan.get("recovery_objectives", {}) if bcp_plan else {},
			"timeline": [{"timestamp": _now(), "event": "bcp_activated", "actor": activator_id}],
		}
		await self._store.put("bcp_activations", activation)

		incident.setdefault("bcp_activations", []).append(activation["id"])
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)

		await self._notify.send(
			"bcp-team@datacraft.co.ke", "email",
			f"BCP ACTIVATED: {plan_name}",
			f"BCP plan '{plan_name}' activated by {activator_id} for incident {incident_id}.",
		)
		await self._audit_event(
			"bcp_activated", activator_id, incident_id,
			{"bcp_plan_id": bcp_plan_id, "plan_name": plan_name},
		)
		return activation

	async def post_incident_review(
		self,
		incident_id: str,
		review_date: str,
		reviewers: list[str],
		actions: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Conduct a post-incident review (PIR) and record findings and actions.

		Required for all high and critical incidents before closure.
		Each action: {description, owner_id, deadline}.
		"""
		assert review_date, "review_date required"
		assert reviewers, "reviewers required"
		assert actions, "actions required"

		incident = await self._get_incident(incident_id)

		pir: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"incident_title": incident.get("title"),
			"review_date": review_date,
			"reviewers": reviewers,
			"actions": actions,
			"action_count": len(actions),
			"status": "completed",
			"created_at": _now(),
		}
		await self._store.put("post_incident_reviews", pir)

		incident["post_incident_review_id"] = pir["id"]
		incident.setdefault("timeline", []).append({
			"timestamp": _now(),
			"event": "post_incident_review_completed",
			"detail": f"Reviewers: {reviewers}, Actions: {len(actions)}",
		})
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)

		await self._audit_event(
			"post_incident_review_completed", reviewers[0] if reviewers else "system", incident_id,
			{"review_date": review_date, "action_count": len(actions)},
		)
		return pir

	async def incident_categorise(self, incident_id: str, category: str, sub_category: str, categorised_by: str) -> dict[str, Any]:
		"""Categorise an incident with primary and sub-category."""
		incident = await self._get_incident(incident_id)
		incident["category"] = category
		incident["sub_category"] = sub_category
		incident["categorised_by"] = categorised_by
		incident["categorised_at"] = _now()
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)
		await self._audit_event("incident_categorised", categorised_by, incident_id, {"category": category, "sub_category": sub_category})
		return incident

	async def incident_escalate(self, incident_id: str, escalated_to: str, reason: str) -> dict[str, Any]:
		"""Escalate an incident to a higher authority."""
		incident = await self._get_incident(incident_id)
		incident["escalated_to"] = escalated_to
		incident["escalation_reason"] = reason
		incident["escalated_at"] = _now()
		incident["status"] = "escalated"
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)
		await self._notify.send(escalated_to, "email", f"Incident escalated: {incident.get('title')}", f"Incident {incident_id} escalated. Reason: {reason}")
		await self._audit_event("incident_escalated", "system", incident_id, {"escalated_to": escalated_to})
		return incident

	async def notification_send_icm(self, incident_id: str, recipients: list[str], message: str, channel: str = "email") -> dict[str, Any]:
		"""Send notifications about an incident to a list of recipients."""
		assert recipients, "recipients required"
		assert message, "message required"
		incident = await self._get_incident(incident_id)
		notif_id = _uid()
		for r in recipients:
			await self._notify.send(r, channel, f"Incident notification: {incident.get('title')}", message)
		await self._audit_event("incident_notification_sent", "system", incident_id, {"recipient_count": len(recipients), "channel": channel})
		return {"notification_id": notif_id, "incident_id": incident_id, "recipients": recipients, "channel": channel, "sent_at": _now()}

	async def investigation_assign(self, incident_id: str, investigator_id: str, scope: str) -> dict[str, Any]:
		"""Assign an investigator to an incident."""
		incident = await self._get_incident(incident_id)
		incident["investigator_id"] = investigator_id
		incident["investigation_scope"] = scope
		incident["status"] = "in_investigation"
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)
		await self._notify.send(investigator_id, "email", f"Investigation assigned: {incident.get('title')}", f"You have been assigned to investigate incident {incident_id}. Scope: {scope}")
		await self._audit_event("investigation_assigned", investigator_id, incident_id, {"scope": scope})
		return incident

	async def root_cause_confirm(self, incident_id: str, root_cause: str, confirmed_by: str) -> dict[str, Any]:
		"""Confirm the root cause of an incident."""
		incident = await self._get_incident(incident_id)
		incident["root_cause"] = root_cause
		incident["root_cause_confirmed_by"] = confirmed_by
		incident["root_cause_confirmed_at"] = _now()
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)
		await self._audit_event("root_cause_confirmed", confirmed_by, incident_id, {"root_cause": root_cause[:100]})
		return incident

	async def corrective_action_verify(self, action_id: str, verified_by: str, verification_notes: str) -> dict[str, Any]:
		"""Verify completion of a corrective action."""
		action = await self._store.get("corrective_actions", action_id)
		if action is None:
			raise ValueError(f"Corrective action not found: {action_id}")
		action["verified_by"] = verified_by
		action["verification_notes"] = verification_notes
		action["verified_at"] = _now()
		action["status"] = "verified"
		action["updated_at"] = _now()
		await self._store.put("corrective_actions", action)
		await self._audit_event("corrective_action_verified", verified_by, action_id, {})
		return action

	async def preventive_action_plan(self, incident_id: str, actions: list[dict[str, Any]], owner_id: str, deadline: str) -> dict[str, Any]:
		"""Create a preventive action plan to avoid recurrence."""
		assert actions, "actions required"
		plan_id = _uid()
		plan: dict[str, Any] = {
			"id": plan_id,
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"actions": actions,
			"action_count": len(actions),
			"owner_id": owner_id,
			"deadline": deadline,
			"status": "active",
			"created_at": _now(),
		}
		await self._store.put("preventive_action_plans", plan)
		await self._audit_event("preventive_action_plan_created", owner_id, incident_id, {"action_count": len(actions), "deadline": deadline})
		return plan

	async def lessons_learned_capture(self, incident_id: str, lessons: list[str], captured_by: str) -> dict[str, Any]:
		"""Capture lessons learned from a closed incident."""
		incident = await self._get_incident(incident_id)
		ll_id = _uid()
		record: dict[str, Any] = {
			"id": ll_id,
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"incident_type": incident.get("incident_type"),
			"lessons_learned": lessons,
			"captured_by": captured_by,
			"captured_at": _now(),
		}
		await self._store.put("lessons_learned", record)
		await self._audit_event("lessons_learned_captured", captured_by, incident_id, {"lesson_count": len(lessons)})
		return record

	async def regulatory_notify(self, incident_id: str, regulator: str, notification_type: str, deadline: str) -> dict[str, Any]:
		"""Send regulatory notification — domain alias."""
		return await self.regulatory_notification(incident_id, regulator, notification_type, deadline)

	async def insurance_notify(self, incident_id: str, policy_id: str, estimated_loss: float) -> dict[str, Any]:
		"""Notify insurer about an incident — domain alias."""
		return await self.insurance_claim_trigger(incident_id, policy_id, estimated_loss)

	async def bcp_activate(self, incident_id: str, bcp_plan_id: str, activator_id: str) -> dict[str, Any]:
		"""Activate BCP for an incident — domain alias."""
		return await self.business_continuity_activation(incident_id, bcp_plan_id, activator_id)

	async def communication_log(self, incident_id: str, message: str, channel: str, communicated_by: str) -> dict[str, Any]:
		"""Log a communication event for an incident."""
		incident = await self._get_incident(incident_id)
		log_id = _uid()
		entry: dict[str, Any] = {
			"id": log_id,
			"tenant_id": self._tenant_id,
			"incident_id": incident_id,
			"message": message,
			"channel": channel,
			"communicated_by": communicated_by,
			"logged_at": _now(),
		}
		await self._store.put("incident_communication_logs", entry)
		await self._audit_event("incident_communication_logged", communicated_by, incident_id, {"channel": channel})
		return entry

	async def incident_reopen(self, incident_id: str, reason: str, reopened_by: str) -> dict[str, Any]:
		"""Reopen a closed incident for further investigation."""
		incident = await self._get_incident(incident_id)
		if incident.get("status") != "closed":
			raise ValueError("Only closed incidents can be reopened")
		incident["status"] = "in_investigation"
		incident["reopen_reason"] = reason
		incident["reopened_by"] = reopened_by
		incident["reopened_at"] = _now()
		incident["updated_at"] = _now()
		await self._store.put("incidents", incident)
		await self._audit_event("incident_reopened", reopened_by, incident_id, {"reason": reason})
		return incident

	async def incident_metrics(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Return incident metrics for the period — alias for incident_analytics."""
		return await self.incident_analytics(entity_id, period)

	async def compliance_monitor(self, entity_id: str, framework: str, period: str) -> dict[str, Any]:
		"""Monitor compliance posture against a framework — alias for compliance_score."""
		return await self.compliance_score(entity_id, framework, period)

	async def compliance_evidence(self, control_id: str, evidence_items: list[dict[str, Any]], submitted_by: str) -> dict[str, Any]:
		"""Submit evidence of compliance for a control."""
		assert evidence_items, "evidence_items required"
		sub_id = _uid()
		submission: dict[str, Any] = {
			"id": sub_id,
			"tenant_id": self._tenant_id,
			"control_id": control_id,
			"evidence_items": evidence_items,
			"item_count": len(evidence_items),
			"submitted_by": submitted_by,
			"submitted_at": _now(),
			"status": "submitted",
		}
		await self._store.put("compliance_evidence_submissions", submission)
		await self._audit_event("compliance_evidence_submitted", submitted_by, control_id, {"item_count": len(evidence_items)})
		return submission

	async def icm_analytics(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Return ICM analytics — alias for incident_analytics."""
		return await self.incident_analytics(entity_id, period)

	async def regulatory_reporting_icm(
		self,
		period: str,
		jurisdiction: str,
	) -> dict[str, Any]:
		"""Generate a regulatory incident report for a jurisdiction and period.

		Includes incident counts by type/severity, regulatory notifications sent,
		and compliance test pass rates.
		"""
		assert jurisdiction, "jurisdiction required"
		start, end = _period_bounds(period)

		incidents = await self._store.query(
			"incidents",
			{"tenant_id": self._tenant_id},
			limit=10_000,
		)
		period_incs = [i for i in incidents if start <= i.get("created_at", "")[:10] <= end]
		reg_notifications = await self._store.query("regulatory_notifications", {}, limit=10_000)
		period_notifs = [n for n in reg_notifications if start <= n.get("sent_at", "")[:10] <= end]

		report: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"jurisdiction": jurisdiction,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_incidents": len(period_incs),
			"critical_incidents": sum(1 for i in period_incs if i.get("severity") == "critical"),
			"regulatory_notifications_sent": len(period_notifs),
			"window_exceeded_count": sum(1 for n in period_notifs if n.get("window_exceeded")),
			"by_type": {
				t: sum(1 for i in period_incs if i.get("incident_type") == t)
				for t in SUPPORTED_INCIDENT_TYPES
			},
			"generated_at": _now(),
			"status": "draft",
		}
		await self._store.put("icm_regulatory_reports", report)
		await self._audit_event(
			"regulatory_report_generated", "system", "icm",
			{"jurisdiction": jurisdiction, "period": period},
		)
		return report

	async def incident_kpi_summary(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise incident KPI card for dashboard consumption.

		Covers: total / critical incidents, MTTR, SLA breach rate, resolution rate.
		"""
		start, end = _period_bounds(period)
		incidents = await self._store.query("incidents", {"tenant_id": self._tenant_id}, limit=10_000)
		period_incs = [i for i in incidents if start <= i.get("created_at", "")[:10] <= end]
		resolved = [i for i in period_incs if i.get("status") == "resolved"]
		critical = [i for i in period_incs if i.get("severity") == "critical"]
		resolution_rate = round(len(resolved) / max(len(period_incs), 1) * 100, 1)
		mttr_hours: float = 0.0
		for i in resolved:
			try:
				from datetime import datetime as _dt
				created = _dt.fromisoformat(i["created_at"])
				closed = _dt.fromisoformat(i["resolved_at"])
				mttr_hours += (closed - created).total_seconds() / 3600
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		avg_mttr = round(mttr_hours / max(len(resolved), 1), 2)
		sla_breached = sum(1 for i in period_incs if i.get("sla_breached"))
		sla_breach_rate = round(sla_breached / max(len(period_incs), 1) * 100, 1)
		return {
			"entity_id": entity_id,
			"period": period,
			"total_incidents": len(period_incs),
			"critical_incidents": len(critical),
			"resolved_incidents": len(resolved),
			"resolution_rate_pct": resolution_rate,
			"avg_mttr_hours": avg_mttr,
			"sla_breach_rate_pct": sla_breach_rate,
			"generated_at": _now(),
		}

	async def ml_incident_severity(self, *args, **kwargs):
		"""AI-powered AI-powered incident severity classification. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["low","medium","high","critical"])
			return {"severity": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

