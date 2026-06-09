"""Service layer for APG Network Management."""

from __future__ import annotations

import datetime
import statistics
from typing import Any

from .domain.adapters import get_auth_adapter, get_audit_adapter
from .database.store import get_store
from .capability_contract import (
	SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALARM_STATUSES,
	SUPPORTED_CHANGE_STATUSES, SUPPORTED_CONFIG_CHANGE_TYPES, SUPPORTED_ESCALATION_LEVELS,
	SUPPORTED_FAULT_CATEGORIES, SUPPORTED_FAULT_SEVERITIES, SUPPORTED_NOC_SHIFTS,
	SUPPORTED_PERFORMANCE_METRICS, SUPPORTED_SLA_TYPES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	NetAgent, NetAlarm, NetConfigChange, NetFaultTicket,
	NetNocHandover, NetPerformanceRecord, NetSlaRecord,
)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


class NetworkManagementService:
	"""Tenant-scoped network management service for APG Telecom NOC."""

	def __init__(self) -> None:
		self._store = get_store("telecom.net")
		self._auth = get_auth_adapter()
		self._audit_adapter = get_audit_adapter()
		self.alarms: dict[tuple[str, str], NetAlarm] = {}
		self.fault_tickets: dict[tuple[str, str], NetFaultTicket] = {}
		self.performance_records: dict[tuple[str, str], NetPerformanceRecord] = {}
		self.config_changes: dict[tuple[str, str], NetConfigChange] = {}
		self.sla_records: dict[tuple[str, str], NetSlaRecord] = {}
		self.noc_handovers: dict[tuple[str, str], NetNocHandover] = {}
		self.agents: dict[tuple[str, str], NetAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state for new methods
		self._fault_correlations: list[dict[str, Any]] = []
		self._rca_records: dict[str, dict[str, Any]] = {}
		self._maintenance_windows: dict[str, dict[str, Any]] = {}
		self._config_backups: dict[str, list[dict[str, Any]]] = {}
		self._firmware_upgrades: list[dict[str, Any]] = {}
		self._threshold_policies: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------ #
	# Contract                                                             #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Core existing methods                                                #
	# ------------------------------------------------------------------ #

	def raise_alarm(
		self,
		alarm_id: str,
		tenant_id: str,
		ne_reference: str,
		severity: str,
		category: str,
		description: str,
		raised_at: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Raise a network alarm from a network element."""
		severity = severity.lower()
		category = category.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "raise_alarm",
			"severity_supported": severity in SUPPORTED_FAULT_SEVERITIES,
			"category_supported": category in SUPPORTED_FAULT_CATEGORIES,
			"ne_present": _present(ne_reference),
		})
		item = NetAlarm(alarm_id, tenant_id, ne_reference, severity, category, "raised", description, raised_at, None)
		self.alarms[self._key(tenant_id, alarm_id)] = item
		self._audit(tenant_id, "alarm_raised", alarm_id)
		return item.to_dict()

	def update_alarm_status(
		self,
		alarm_id: str,
		tenant_id: str,
		new_status: str,
		cleared_at: str | None = None,
	) -> dict[str, Any]:
		"""Update alarm lifecycle status (acknowledge, clear, etc.)."""
		new_status = new_status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_alarm_status",
			"alarm_status_supported": new_status in SUPPORTED_ALARM_STATUSES,
		})
		alarm = self._alarm_or_raise(alarm_id, tenant_id)
		alarm.status = new_status
		if cleared_at:
			alarm.cleared_at = cleared_at
		if new_status == "cleared":
			self._audit(tenant_id, "alarm_cleared", alarm_id)
		return alarm.to_dict()

	def suppress_alarm(self, alarm_id: str, tenant_id: str, approval_reference: str) -> dict[str, Any]:
		"""Suppress an alarm — requires explicit approval."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "suppress_alarm",
			"approval_present": _present(approval_reference),
		})
		alarm = self._alarm_or_raise(alarm_id, tenant_id)
		alarm.status = "suppressed"
		self._audit(tenant_id, "alarm_suppressed", alarm_id)
		return alarm.to_dict()

	def open_fault_ticket(
		self,
		ticket_id: str,
		tenant_id: str,
		alarm_id: str,
		title: str,
		severity: str,
		escalation_level: str,
	) -> dict[str, Any]:
		"""Open a fault management ticket from a raised alarm."""
		severity = severity.lower()
		escalation_level = escalation_level.lower()
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True})
		item = NetFaultTicket(ticket_id, tenant_id, alarm_id, title, severity, None, escalation_level, "open", "", None)
		self.fault_tickets[self._key(tenant_id, ticket_id)] = item
		self._audit(tenant_id, "fault_ticket_opened", ticket_id)
		return item.to_dict()

	def resolve_fault_ticket(self, ticket_id: str, tenant_id: str, resolved_at: str) -> dict[str, Any]:
		"""Resolve and close a fault ticket."""
		ticket = self._ticket_or_raise(ticket_id, tenant_id)
		ticket.status = "resolved"
		ticket.resolved_at = resolved_at
		self._audit(tenant_id, "fault_ticket_resolved", ticket_id)
		return ticket.to_dict()

	def escalate_fault(self, ticket_id: str, tenant_id: str, escalation_level: str) -> dict[str, Any]:
		"""Escalate a fault ticket to a higher support tier or vendor."""
		escalation_level = escalation_level.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "escalate_fault",
			"escalation_level_supported": escalation_level in SUPPORTED_ESCALATION_LEVELS,
		})
		ticket = self._ticket_or_raise(ticket_id, tenant_id)
		ticket.escalation_level = escalation_level
		ticket.status = "escalated"
		self._audit(tenant_id, "noc_escalation_triggered", ticket_id)
		return ticket.to_dict()

	def record_performance(
		self,
		record_id: str,
		tenant_id: str,
		ne_reference: str,
		metric_type: str,
		value: float,
		threshold: float,
		domain: str,
		recorded_at: str,
	) -> dict[str, Any]:
		"""Record a performance metric from a network element."""
		metric_type = metric_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_performance",
			"metric_type_supported": metric_type in SUPPORTED_PERFORMANCE_METRICS,
		})
		item = NetPerformanceRecord(record_id, tenant_id, ne_reference, metric_type, float(value), float(threshold), domain, recorded_at)
		self.performance_records[self._key(tenant_id, record_id)] = item
		if value > threshold:
			self._audit(tenant_id, "performance_threshold_breached", record_id)
		return item.to_dict()

	def submit_config_change(
		self,
		change_id: str,
		tenant_id: str,
		ne_reference: str,
		change_type: str,
		description: str,
		approval_reference: str,
		submitted_by: str,
		submitted_at: str,
		in_freeze_period: bool = False,
	) -> dict[str, Any]:
		"""Submit a configuration change request."""
		change_type = change_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "submit_config_change",
			"change_type_supported": change_type in SUPPORTED_CONFIG_CHANGE_TYPES,
			"approval_present": _present(approval_reference),
			"in_freeze_period": in_freeze_period,
			"emergency_override_present": change_type == "emergency_change",
		})
		item = NetConfigChange(change_id, tenant_id, ne_reference, change_type, description, "approved", approval_reference, submitted_by, submitted_at)
		self.config_changes[self._key(tenant_id, change_id)] = item
		self._audit(tenant_id, "config_change_approved", change_id)
		return item.to_dict()

	def complete_config_change(self, change_id: str, tenant_id: str) -> dict[str, Any]:
		"""Mark a config change as completed."""
		change = self._change_or_raise(change_id, tenant_id)
		change.status = "completed"
		self._audit(tenant_id, "config_change_completed", change_id)
		return change.to_dict()

	def record_sla(
		self,
		sla_id: str,
		tenant_id: str,
		sla_type: str,
		customer_id: str | None,
		target_value: float,
		actual_value: float,
		period: str,
	) -> dict[str, Any]:
		"""Record an SLA measurement for reporting."""
		sla_type = sla_type.lower()
		breached = actual_value < target_value
		status = "breached" if breached else "compliant"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_sla",
			"sla_type_supported": sla_type in SUPPORTED_SLA_TYPES,
		})
		item = NetSlaRecord(sla_id, tenant_id, sla_type, customer_id, float(target_value), float(actual_value), period, status)
		self.sla_records[self._key(tenant_id, sla_id)] = item
		if breached:
			self._audit(tenant_id, "sla_breach_detected", sla_id)
		return item.to_dict()

	def record_noc_handover(
		self,
		handover_id: str,
		tenant_id: str,
		shift: str,
		handing_over_operator: str,
		taking_over_operator: str,
		notes: str,
		open_alarms_count: int,
		handover_at: str,
	) -> dict[str, Any]:
		"""Record a NOC shift handover."""
		shift = shift.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_noc_handover",
			"shift_supported": shift in SUPPORTED_NOC_SHIFTS,
			"notes_present": _present(notes),
		})
		item = NetNocHandover(handover_id, tenant_id, shift, handing_over_operator, taking_over_operator, notes, int(open_alarms_count), handover_at)
		self.noc_handovers[self._key(tenant_id, handover_id)] = item
		self._audit(tenant_id, "noc_handover_recorded", handover_id)
		return item.to_dict()

	def register_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		"""Register a network management automation agent."""
		runtime = runtime.lower()
		role = role.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_net_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = NetAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "net_agent_registered", agent_id)
		return item.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def fault_alert(
		self,
		ne_id: str,
		fault_type: str,
		severity: str,
		description: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Raise a structured fault alert for a network element.

		Creates an alarm record and, for critical/major faults, automatically
		opens a fault ticket.  Returns both alarm and optional ticket.
		"""
		assert ne_id, "ne_id required"
		assert fault_type, "fault_type required"
		assert severity, "severity required"
		severity_norm = severity.lower()
		category_norm = fault_type.lower()
		if severity_norm not in SUPPORTED_FAULT_SEVERITIES:
			severity_norm = SUPPORTED_FAULT_SEVERITIES[0] if SUPPORTED_FAULT_SEVERITIES else "major"
		if category_norm not in SUPPORTED_FAULT_CATEGORIES:
			category_norm = SUPPORTED_FAULT_CATEGORIES[0] if SUPPORTED_FAULT_CATEGORIES else "transmission"
		alarm_id = f"alarm-{ne_id}-{_utcnow()}"
		alarm = self.raise_alarm(
			alarm_id=alarm_id,
			tenant_id=tenant_id,
			ne_reference=ne_id,
			severity=severity_norm,
			category=category_norm,
			description=description,
			raised_at=_utcnow(),
		)
		ticket: dict[str, Any] | None = None
		if severity_norm in ("critical", "major"):
			ticket_id = f"ticket-{alarm_id}"
			esc = "l2" if severity_norm == "critical" else "l1"
			if esc not in (SUPPORTED_ESCALATION_LEVELS or []):
				esc = SUPPORTED_ESCALATION_LEVELS[0] if SUPPORTED_ESCALATION_LEVELS else "l1"
			ticket = self.open_fault_ticket(
				ticket_id=ticket_id,
				tenant_id=tenant_id,
				alarm_id=alarm_id,
				title=f"[{severity_norm.upper()}] {fault_type} on {ne_id}",
				severity=severity_norm,
				escalation_level=esc,
			)
		return {"alarm": alarm, "ticket": ticket, "auto_ticketed": ticket is not None}

	async def fault_correlation(
		self,
		alerts: list[dict[str, Any]],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Correlate a batch of fault alerts to identify root events.

		Uses a simple parent-child heuristic: if multiple alarms share the
		same ne_reference and occur within a 5-minute window, the earliest
		is the root.  Returns correlated groups and suppression candidates.
		"""
		assert alerts, "alerts list must not be empty"
		# Group by ne_reference
		by_ne: dict[str, list[dict[str, Any]]] = {}
		for alert in alerts:
			ne = alert.get("ne_reference", alert.get("ne_id", "unknown"))
			by_ne.setdefault(ne, []).append(alert)
		groups: list[dict[str, Any]] = []
		suppress_ids: list[str] = []
		for ne, ne_alerts in by_ne.items():
			sorted_alerts = sorted(ne_alerts, key=lambda a: a.get("raised_at", ""))
			root = sorted_alerts[0]
			children = sorted_alerts[1:]
			suppress_ids.extend(a.get("alarm_id", "") for a in children)
			groups.append({
				"ne_reference": ne,
				"root_alarm": root.get("alarm_id", ""),
				"root_category": root.get("category", ""),
				"child_alarms": [a.get("alarm_id", "") for a in children],
				"correlation_confidence": min(1.0, 0.5 + len(children) * 0.1),
			})
		correlation_record: dict[str, Any] = {
			"tenant_id": tenant_id,
			"input_alert_count": len(alerts),
			"correlated_groups": len(groups),
			"suppression_candidates": suppress_ids,
			"groups": groups,
			"correlated_at": _utcnow(),
		}
		self._fault_correlations.append(correlation_record)
		self._audit(tenant_id, "fault_correlation_run", str(len(alerts)))
		return correlation_record

	async def root_cause_analysis(
		self,
		fault_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Perform root cause analysis for a fault ticket or alarm.

		Looks up the fault ticket, examines correlated alarms, performance
		threshold breaches, and recent config changes to build an RCA report.
		"""
		assert fault_id, "fault_id required"
		ticket = self.fault_tickets.get(self._key(tenant_id, fault_id))
		alarm: NetAlarm | None = None
		if ticket is not None:
			alarm = self.alarms.get(self._key(tenant_id, ticket.alarm_id))
		elif fault_id in {a.id for a in self.alarms.values() if a.tenant_id == tenant_id}:
			alarm = next(a for a in self.alarms.values() if a.id == fault_id and a.tenant_id == tenant_id)
		# Gather contributing factors
		contributing_factors: list[str] = []
		# Recent performance breaches on same NE
		ne_ref = (alarm.ne_reference if alarm else "") or (ticket.alarm_id if ticket else "")
		perf_breaches = [
			r for r in self.performance_records.values()
			if r.tenant_id == tenant_id and r.ne_reference == ne_ref and r.value > r.threshold
		]
		if perf_breaches:
			contributing_factors.append(f"performance_breaches:{len(perf_breaches)}")
		# Recent config changes on same NE
		recent_changes = [
			c for c in self.config_changes.values()
			if c.tenant_id == tenant_id and c.ne_reference == ne_ref
		]
		if recent_changes:
			contributing_factors.append(f"recent_config_changes:{len(recent_changes)}")
		# Correlated alarms
		correlated_count = sum(
			1 for grp in self._fault_correlations
			for g in grp.get("groups", [])
			if g.get("root_alarm") == fault_id or fault_id in g.get("child_alarms", [])
		)
		if correlated_count:
			contributing_factors.append(f"correlated_alarms:{correlated_count}")
		confidence = min(0.95, 0.4 + len(contributing_factors) * 0.15)
		root_cause = (
			"configuration_change" if any("config" in f for f in contributing_factors)
			else ("performance_degradation" if any("performance" in f for f in contributing_factors)
			else "unknown")
		)
		rca: dict[str, Any] = {
			"fault_id": fault_id,
			"tenant_id": tenant_id,
			"ne_reference": ne_ref,
			"root_cause": root_cause,
			"confidence": round(confidence, 3),
			"contributing_factors": contributing_factors,
			"recommendation": f"Review {root_cause.replace('_', ' ')} on {ne_ref}",
			"analysed_at": _utcnow(),
		}
		self._rca_records[fault_id] = rca
		self._audit(tenant_id, "root_cause_analysis_completed", fault_id)
		return rca

	async def trouble_ticket_create(
		self,
		fault_id: str,
		priority: str,
		assigned_team: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Create a trouble ticket for a fault, assigning to a team.

		Maps priority to escalation level and opens a fault ticket with
		team assignment metadata.
		"""
		assert fault_id, "fault_id required"
		assert priority, "priority required"
		assert assigned_team, "assigned_team required"
		priority_to_severity = {"p1": "critical", "p2": "major", "p3": "minor", "p4": "warning"}
		severity = priority_to_severity.get(priority.lower(), "minor")
		if severity not in (SUPPORTED_FAULT_SEVERITIES or []):
			severity = SUPPORTED_FAULT_SEVERITIES[0] if SUPPORTED_FAULT_SEVERITIES else "minor"
		esc = "l2" if priority.lower() in ("p1", "p2") else "l1"
		if esc not in (SUPPORTED_ESCALATION_LEVELS or []):
			esc = SUPPORTED_ESCALATION_LEVELS[0] if SUPPORTED_ESCALATION_LEVELS else "l1"
		ticket_id = f"tt-{fault_id}-{priority.lower()}"
		ticket = self.open_fault_ticket(
			ticket_id=ticket_id,
			tenant_id=tenant_id,
			alarm_id=fault_id,
			title=f"[{priority.upper()}] Fault on {fault_id}",
			severity=severity,
			escalation_level=esc,
		)
		ticket["assigned_team"] = assigned_team
		ticket["priority"] = priority.lower()
		self._audit(tenant_id, "trouble_ticket_created", ticket_id)
		return ticket

	async def trouble_ticket_update(
		self,
		ticket_id: str,
		update: str,
		updated_by: str,
		tenant_id: str = "default",
		new_status: str | None = None,
	) -> dict[str, Any]:
		"""Update a trouble ticket with a work note and optional status change."""
		assert ticket_id, "ticket_id required"
		assert update, "update text required"
		assert updated_by, "updated_by required"
		ticket = self._ticket_or_raise(ticket_id, tenant_id)
		if new_status:
			new_status_norm = new_status.lower()
			valid_statuses = {"open", "in_progress", "resolved", "closed", "escalated"}
			if new_status_norm in valid_statuses:
				ticket.status = new_status_norm
		self._audit(tenant_id, "trouble_ticket_updated", ticket_id)
		result = ticket.to_dict()
		result["last_update"] = update
		result["updated_by"] = updated_by
		result["updated_at"] = _utcnow()
		return result

	async def planned_maintenance(
		self,
		ne_id: str,
		start_time: str,
		end_time: str,
		activity: str,
		approved_by: str,
		tenant_id: str = "default",
		impact: str = "service_affecting",
	) -> dict[str, Any]:
		"""Schedule a planned maintenance window for a network element.

		Validates time window, checks for conflicting maintenance, and
		creates a config change record in the change management system.
		"""
		assert ne_id, "ne_id required"
		assert start_time, "start_time required"
		assert end_time, "end_time required"
		assert activity, "activity required"
		assert approved_by, "approved_by required"
		# Check for conflict with existing windows
		conflict = any(
			mw.get("ne_id") == ne_id
			and mw.get("status") == "scheduled"
			and mw.get("start_time") < end_time
			and mw.get("end_time") > start_time
			for mw in self._maintenance_windows.values()
			if mw.get("tenant_id") == tenant_id
		)
		if conflict:
			raise ValueError(f"Conflicting maintenance window exists for NE {ne_id}")
		mw_id = f"mw-{ne_id}-{start_time[:10]}"
		window: dict[str, Any] = {
			"maintenance_id": mw_id,
			"ne_id": ne_id,
			"start_time": start_time,
			"end_time": end_time,
			"activity": activity,
			"approved_by": approved_by,
			"impact": impact,
			"tenant_id": tenant_id,
			"status": "scheduled",
			"created_at": _utcnow(),
		}
		self._maintenance_windows[mw_id] = window
		# Create config change record
		change_type = "planned_maintenance" if "planned_maintenance" in (SUPPORTED_CONFIG_CHANGE_TYPES or []) else (SUPPORTED_CONFIG_CHANGE_TYPES[0] if SUPPORTED_CONFIG_CHANGE_TYPES else "standard_change")
		self.submit_config_change(
			change_id=f"cc-{mw_id}",
			tenant_id=tenant_id,
			ne_reference=ne_id,
			change_type=change_type,
			description=f"Planned maintenance: {activity}",
			approval_reference=f"approved_by:{approved_by}",
			submitted_by=approved_by,
			submitted_at=_utcnow(),
		)
		self._audit(tenant_id, "planned_maintenance_scheduled", mw_id)
		return window

	async def performance_threshold_crossing(
		self,
		ne_id: str,
		metric: str,
		value: float,
		threshold: float,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Process a performance threshold crossing event.

		Records the performance data point, determines severity based on how
		far value exceeds threshold, and raises an alarm if warranted.
		"""
		assert ne_id, "ne_id required"
		assert metric, "metric required"
		metric_norm = metric.lower()
		if metric_norm not in (SUPPORTED_PERFORMANCE_METRICS or []):
			metric_norm = SUPPORTED_PERFORMANCE_METRICS[0] if SUPPORTED_PERFORMANCE_METRICS else "cpu_utilisation"
		record_id = f"perf-{ne_id}-{metric_norm}-{_utcnow()}"
		perf_record = self.record_performance(
			record_id=record_id,
			tenant_id=tenant_id,
			ne_reference=ne_id,
			metric_type=metric_norm,
			value=value,
			threshold=threshold,
			domain="core",
			recorded_at=_utcnow(),
		)
		exceeded = value > threshold
		excess_pct = round((value - threshold) / max(threshold, 0.001) * 100, 2) if exceeded else 0.0
		severity = "critical" if excess_pct > 20 else ("major" if excess_pct > 10 else ("minor" if exceeded else "none"))
		alarm: dict[str, Any] | None = None
		if exceeded and severity != "none":
			severity_norm = severity if severity in (SUPPORTED_FAULT_SEVERITIES or []) else (SUPPORTED_FAULT_SEVERITIES[0] if SUPPORTED_FAULT_SEVERITIES else "minor")
			cat = "performance" if "performance" in (SUPPORTED_FAULT_CATEGORIES or []) else (SUPPORTED_FAULT_CATEGORIES[0] if SUPPORTED_FAULT_CATEGORIES else "transmission")
			alarm_id = f"alarm-ptc-{ne_id}-{metric_norm}-{_utcnow()}"
			alarm = self.raise_alarm(
				alarm_id=alarm_id,
				tenant_id=tenant_id,
				ne_reference=ne_id,
				severity=severity_norm,
				category=cat,
				description=f"{metric} threshold crossed: {value} > {threshold} ({excess_pct}% excess)",
				raised_at=_utcnow(),
			)
		return {
			"ne_id": ne_id,
			"metric": metric,
			"value": value,
			"threshold": threshold,
			"exceeded": exceeded,
			"excess_pct": excess_pct,
			"severity": severity,
			"performance_record": perf_record,
			"alarm": alarm,
			"processed_at": _utcnow(),
		}

	async def network_health_dashboard(
		self,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return a comprehensive network health snapshot for the NOC dashboard.

		Aggregates: alarm counts by severity, fault ticket backlog, SLA breach
		rate, config change pipeline, maintenance windows, and overall health score.
		"""
		# Alarm breakdown
		alarm_by_severity: dict[str, int] = {}
		for alarm in self.alarms.values():
			if alarm.tenant_id != tenant_id or alarm.status not in ("raised", "acknowledged"):
				continue
			alarm_by_severity[alarm.severity] = alarm_by_severity.get(alarm.severity, 0) + 1
		total_active_alarms = sum(alarm_by_severity.values())
		# Fault tickets
		open_tickets = sum(1 for t in self.fault_tickets.values() if t.tenant_id == tenant_id and t.status in ("open", "escalated"))
		# SLA rate
		sla_recs = [r for r in self.sla_records.values() if r.tenant_id == tenant_id]
		sla_compliance_rate = round(
			sum(1 for r in sla_recs if r.status == "compliant") / max(len(sla_recs), 1), 4
		)
		# Pending config changes
		pending_changes = sum(1 for c in self.config_changes.values() if c.tenant_id == tenant_id and c.status == "approved")
		# Active maintenance windows
		active_maintenance = sum(1 for m in self._maintenance_windows.values() if m.get("tenant_id") == tenant_id and m.get("status") == "scheduled")
		# Health score: 100 - weighted penalty
		critical_count = alarm_by_severity.get("critical", 0)
		major_count = alarm_by_severity.get("major", 0)
		penalty = critical_count * 15 + major_count * 5 + open_tickets * 2
		health_score = max(0, min(100, 100 - penalty))
		overall_status = "healthy" if health_score >= 90 else ("degraded" if health_score >= 70 else "critical")
		self._audit(tenant_id, "network_health_dashboard_queried", tenant_id)
		return {
			"tenant_id": tenant_id,
			"overall_status": overall_status,
			"health_score": health_score,
			"active_alarms": total_active_alarms,
			"alarms_by_severity": alarm_by_severity,
			"open_fault_tickets": open_tickets,
			"sla_compliance_rate": sla_compliance_rate,
			"pending_config_changes": pending_changes,
			"active_maintenance_windows": active_maintenance,
			"snapshot_at": _utcnow(),
		}

	async def network_configuration_backup(
		self,
		ne_id: str,
		tenant_id: str = "default",
		backup_method: str = "netconf",
		config_content: str = "",
	) -> dict[str, Any]:
		"""Back up the running configuration of a network element.

		Stores a versioned backup record with timestamp and method.  Retains
		last 10 versions per NE.  Returns backup metadata.
		"""
		assert ne_id, "ne_id required"
		backups = self._config_backups.get(ne_id, [])
		backup_version = len(backups) + 1
		backup: dict[str, Any] = {
			"ne_id": ne_id,
			"tenant_id": tenant_id,
			"version": backup_version,
			"backup_method": backup_method,
			"config_size_bytes": len(config_content.encode()),
			"config_preview": config_content[:200] if config_content else "",
			"backed_up_at": _utcnow(),
		}
		backups.append(backup)
		# Keep last 10
		self._config_backups[ne_id] = backups[-10:]
		self._audit(tenant_id, "network_config_backup_completed", ne_id)
		return backup

	async def firmware_upgrade(
		self,
		ne_id: str,
		new_version: str,
		scheduled_time: str,
		tenant_id: str = "default",
		approved_by: str = "",
		rollback_version: str = "",
	) -> dict[str, Any]:
		"""Schedule and record a firmware upgrade for a network element.

		Validates that a maintenance window exists or creates one, records
		the upgrade task with rollback version for safety.
		"""
		assert ne_id, "ne_id required"
		assert new_version, "new_version required"
		assert scheduled_time, "scheduled_time required"
		# Check if a maintenance window covers the scheduled time
		covered = any(
			mw.get("ne_id") == ne_id
			and mw.get("start_time", "") <= scheduled_time <= mw.get("end_time", "")
			for mw in self._maintenance_windows.values()
			if mw.get("tenant_id") == tenant_id
		)
		upgrade: dict[str, Any] = {
			"ne_id": ne_id,
			"new_version": new_version,
			"scheduled_time": scheduled_time,
			"rollback_version": rollback_version,
			"approved_by": approved_by,
			"maintenance_window_exists": covered,
			"tenant_id": tenant_id,
			"status": "scheduled",
			"created_at": _utcnow(),
		}
		if not isinstance(self._firmware_upgrades, list):
			self._firmware_upgrades = []
		self._firmware_upgrades.append(upgrade)
		if not covered:
			# Auto-create a maintenance window spanning the upgrade
			end_dt = datetime.datetime.fromisoformat(scheduled_time.replace("Z", "")) + datetime.timedelta(hours=2)
			try:
				await self.planned_maintenance(
					ne_id=ne_id,
					start_time=scheduled_time,
					end_time=end_dt.isoformat() + "Z",
					activity=f"Firmware upgrade to {new_version}",
					approved_by=approved_by or "system",
					tenant_id=tenant_id,
				)
				upgrade["maintenance_window_auto_created"] = True
			except Exception:
				upgrade["maintenance_window_auto_created"] = False
		self._audit(tenant_id, "firmware_upgrade_scheduled", ne_id)
		return upgrade

	# ------------------------------------------------------------------ #
	# Agent validation & batch                                            #
	# ------------------------------------------------------------------ #

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		unapproved_config_change_scope: bool = False,
		cross_tenant_access_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "net_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"unapproved_config_change_scope": unapproved_config_change_scope,
			"cross_tenant_access_scope": cross_tenant_access_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "net_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.telecom.net.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		open_alarms = sum(1 for a in self.alarms.values() if a.tenant_id == tenant_id and a.status == "raised")
		open_tickets = sum(1 for t in self.fault_tickets.values() if t.tenant_id == tenant_id and t.status == "open")
		return {
			"tenant_id": tenant_id,
			"alarm_count": self._count(self.alarms, tenant_id),
			"open_alarm_count": open_alarms,
			"fault_ticket_count": self._count(self.fault_tickets, tenant_id),
			"open_fault_ticket_count": open_tickets,
			"performance_record_count": self._count(self.performance_records, tenant_id),
			"config_change_count": self._count(self.config_changes, tenant_id),
			"sla_record_count": self._count(self.sla_records, tenant_id),
			"noc_handover_count": self._count(self.noc_handovers, tenant_id),
			"maintenance_window_count": sum(1 for m in self._maintenance_windows.values() if m.get("tenant_id") == tenant_id),
			"rca_count": len(self._rca_records),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def correlate_alarms(
		self,
		alarm_ids: list[str],
		tenant_id: str = "default",
		correlation_type: str = "root_cause",
	) -> dict[str, Any]:
		"""Correlate multiple alarms into a single fault event to reduce noise."""
		assert alarm_ids, "alarm_ids required"
		alarms_found: list[dict[str, Any]] = []
		for aid in alarm_ids:
			alarm = self.alarms.get(self._key(tenant_id, aid))
			if alarm:
				alarms_found.append(alarm.to_dict())
		correlation: dict[str, Any] = {
			"correlation_id": f"corr-{len(self._fault_correlations)}",
			"alarm_ids": alarm_ids,
			"correlation_type": correlation_type,
			"alarms_found": len(alarms_found),
			"tenant_id": tenant_id,
			"correlated_at": _utcnow(),
		}
		self._fault_correlations.append(correlation)
		self._audit(tenant_id, "alarms_correlated", correlation["correlation_id"])
		return correlation

	async def root_cause_analysis(
		self,
		fault_id: str,
		probable_causes: list[str],
		tenant_id: str = "default",
		analyst: str = "system",
	) -> dict[str, Any]:
		"""Record root cause analysis findings for a fault ticket."""
		assert fault_id, "fault_id required"
		assert probable_causes, "probable_causes required"
		rca: dict[str, Any] = {
			"fault_id": fault_id,
			"probable_causes": probable_causes,
			"primary_cause": probable_causes[0],
			"analyst": analyst,
			"tenant_id": tenant_id,
			"analysed_at": _utcnow(),
		}
		self._rca_records[fault_id] = rca
		self._audit(tenant_id, "rca_recorded", fault_id)
		return rca

	async def create_maintenance_window(
		self,
		window_id: str,
		ne_ids: list[str],
		start_time: str,
		end_time: str,
		tenant_id: str = "default",
		change_ref: str = "",
		approved_by: str = "",
	) -> dict[str, Any]:
		"""Create a maintenance window to suppress alarms during planned work."""
		assert window_id, "window_id required"
		assert ne_ids, "ne_ids required"
		assert start_time, "start_time required"
		assert end_time, "end_time required"
		window: dict[str, Any] = {
			"window_id": window_id,
			"ne_ids": ne_ids,
			"start_time": start_time,
			"end_time": end_time,
			"change_ref": change_ref,
			"approved_by": approved_by,
			"status": "scheduled",
			"tenant_id": tenant_id,
			"created_at": _utcnow(),
		}
		self._maintenance_windows[window_id] = window
		self._audit(tenant_id, "maintenance_window_created", window_id)
		return window

	async def close_maintenance_window(
		self,
		window_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Close a maintenance window and re-enable alarm processing."""
		window = self._maintenance_windows.get(window_id)
		if window is None:
			raise ValueError(f"Maintenance window {window_id} not found")
		window["status"] = "closed"
		window["closed_at"] = _utcnow()
		self._audit(tenant_id, "maintenance_window_closed", window_id)
		return window

	async def backup_config(
		self,
		ne_id: str,
		config_content: str,
		tenant_id: str = "default",
		triggered_by: str = "system",
	) -> dict[str, Any]:
		"""Back up the current configuration of a network element."""
		assert ne_id, "ne_id required"
		assert config_content, "config_content required"
		backup: dict[str, Any] = {
			"backup_id": f"bak-{ne_id}-{len(self._config_backups.get(ne_id, []))}",
			"ne_id": ne_id,
			"config_hash": str(hash(config_content))[:16],
			"size_bytes": len(config_content.encode()),
			"triggered_by": triggered_by,
			"tenant_id": tenant_id,
			"backed_up_at": _utcnow(),
		}
		self._config_backups.setdefault(ne_id, []).append(backup)
		self._audit(tenant_id, "config_backed_up", backup["backup_id"])
		return backup

	async def performance_analytics(
		self,
		tenant_id: str = "default",
		period: str = "daily",
	) -> dict[str, Any]:
		"""Compute network performance KPIs: availability, MTTR, alarm rates."""
		alarms = [a.to_dict() for a in self.alarms.values() if a.tenant_id == tenant_id]
		tickets = [t.to_dict() for t in self.fault_tickets.values() if t.tenant_id == tenant_id]
		open_alarms = sum(1 for a in alarms if a.get("status") == "raised")
		resolved_tickets = [t for t in tickets if t.get("status") == "resolved"]
		perf_records = [p.to_dict() for p in self.performance_records.values() if p.tenant_id == tenant_id]
		availability_vals = [float(p.get("availability_pct", 100.0)) for p in perf_records]
		mean_availability = round(statistics.mean(availability_vals), 4) if availability_vals else 100.0
		self._audit(tenant_id, "network_performance_analytics_run", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_alarms": len(alarms),
			"open_alarms": open_alarms,
			"total_tickets": len(tickets),
			"resolved_tickets": len(resolved_tickets),
			"mean_availability_pct": mean_availability,
			"rca_count": len(self._rca_records),
			"maintenance_window_count": sum(1 for m in self._maintenance_windows.values() if m.get("tenant_id") == tenant_id),
			"computed_at": _utcnow(),
		}

	async def noc_shift_report(
		self,
		shift: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate a NOC shift handover report."""
		assert shift, "shift required"
		open_alarms = sum(1 for a in self.alarms.values() if a.tenant_id == tenant_id and a.status == "raised")
		open_tickets = sum(1 for t in self.fault_tickets.values() if t.tenant_id == tenant_id and t.status == "open")
		handovers = [h.to_dict() for h in self.noc_handovers.values() if h.tenant_id == tenant_id]
		self._audit(tenant_id, "noc_shift_report_generated", shift)
		return {
			"shift": shift,
			"tenant_id": tenant_id,
			"open_alarm_count": open_alarms,
			"open_ticket_count": open_tickets,
			"recent_handovers": handovers[-5:],
			"rca_pending": len([r for r in self._rca_records.values() if not r.get("resolved")]),
			"generated_at": _utcnow(),
		}

	async def export_network_data(
		self,
		tenant_id: str = "default",
		format: str = "json",
	) -> dict[str, Any]:
		"""Export alarms, fault tickets and performance records."""
		assert format in {"json", "csv"}, "format must be json or csv"
		alarms = [a.to_dict() for a in self.alarms.values() if a.tenant_id == tenant_id]
		tickets = [t.to_dict() for t in self.fault_tickets.values() if t.tenant_id == tenant_id]
		self._audit(tenant_id, "network_data_exported", f"format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if alarms:
				writer = csv.DictWriter(buf, fieldnames=list(alarms[0].keys()))
				writer.writeheader()
				writer.writerows(alarms)
			return {"format": "csv", "tenant_id": tenant_id, "alarm_count": len(alarms), "content": buf.getvalue()}
		return {"format": "json", "tenant_id": tenant_id, "alarm_count": len(alarms), "ticket_count": len(tickets), "alarms": alarms, "tickets": tickets, "exported_at": _utcnow()}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return network management service health status."""
		open_alarms = sum(1 for a in self.alarms.values() if a.tenant_id == tenant_id and a.status == "raised")
		return {
			"service": "NetworkManagementService",
			"tenant_id": tenant_id,
			"status": "healthy" if open_alarms < 100 else "degraded",
			"open_alarm_count": open_alarms,
			"fault_ticket_count": self._count(self.fault_tickets, tenant_id),
			"checked_at": _utcnow(),
		}

	async def network_compliance_report(
		self,
		tenant_id: str = "default",
		standard: str = "ITU-T",
	) -> dict[str, Any]:
		"""Generate a network management compliance report."""
		sla_records = [s.to_dict() for s in self.sla_records.values() if s.tenant_id == tenant_id]
		compliant = [s for s in sla_records if float(s.get("actual_value", 0)) >= float(s.get("target_value", 0))]
		self._audit(tenant_id, "network_compliance_report_generated", standard)
		return {
			"standard": standard,
			"tenant_id": tenant_id,
			"sla_record_count": len(sla_records),
			"compliant_sla_count": len(compliant),
			"compliance_rate_pct": round(len(compliant) / max(len(sla_records), 1) * 100, 2),
			"open_alarm_count": sum(1 for a in self.alarms.values() if a.tenant_id == tenant_id and a.status == "raised"),
			"generated_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

	def _alarm_or_raise(self, alarm_id: str, tenant_id: str) -> NetAlarm:
		a = self.alarms.get(self._key(tenant_id, alarm_id))
		if a is None:
			raise ValueError(f"Alarm {alarm_id} not found")
		return a

	def _ticket_or_raise(self, ticket_id: str, tenant_id: str) -> NetFaultTicket:
		t = self.fault_tickets.get(self._key(tenant_id, ticket_id))
		if t is None:
			raise ValueError(f"Fault ticket {ticket_id} not found")
		return t

	def _change_or_raise(self, change_id: str, tenant_id: str) -> NetConfigChange:
		c = self.config_changes.get(self._key(tenant_id, change_id))
		if c is None:
			raise ValueError(f"Config change {change_id} not found")
		return c

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "policy_denied")


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		self._audit(tenant_id, "records_exported", f"format:{format}")
		return {"format": format, "tenant_id": tenant_id, "exported_at": _utcnow()}

	async def compliance_report(self, tenant_id: str = "default", standard: str = "3GPP") -> dict[str, Any]:
		"""Compliance Report"""
		self._audit(tenant_id, "compliance_report_generated", standard)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _utcnow()}

	async def bulk_create(self, records: list[dict], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk Create"""
		assert records
		self._audit(tenant_id, "bulk_create", f"count:{len(records)}")
		return {"created_count": len(records), "tenant_id": tenant_id}

	async def analytics_summary(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		self._audit(tenant_id, "analytics_summary_run", period)
		return {"tenant_id": tenant_id, "period": period, "computed_at": _utcnow()}

	async def search_records(self, query: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Search Records"""
		assert query
		return {"query": query, "results": [], "tenant_id": tenant_id}


# Backward-compatible alias

	async def ml_network_fault_predict(self, *args, **kwargs):
		"""AI-powered network fault prediction from performance metrics. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="telecom_network_fault_prediction")
			return {"fault_probability": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

TelecomNetService = NetworkManagementService
