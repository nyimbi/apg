"""Service layer for the Security Operations capability."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import statistics
from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_SEOP_AGENT_ROLES,
	SUPPORTED_SEOP_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .ops_runtime import (
	DetectionRecord,
	IncidentRecord,
	OpsAuditEventRecord,
	PlaybookRecord,
	PostureControlRecord,
	ResponseActionRecord,
	SeopAgentRecord,
	normalize_confidence,
	normalize_severity,
	response_required_actions,
	stable_id,
	utc_now,
)


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class SeopService:
	"""Deterministic Security Operations service for APG composition."""

	def __init__(self, db_url: str | None = None) -> None:
		self.detections: dict[str, DetectionRecord] = {}
		self.incidents: dict[str, IncidentRecord] = {}
		self.playbooks: dict[str, PlaybookRecord] = {}
		self.responses: dict[str, ResponseActionRecord] = {}
		self.posture_controls: dict[str, PostureControlRecord] = {}
		self.audit_events: dict[str, OpsAuditEventRecord] = {}
		self.seop_agents: dict[str, SeopAgentRecord] = {}
		# new collections
		_store = get_store(db_url)
		self._soc_alerts = WriteThruDict('soc_alerts', tenant_id, _store)
		self._threat_hunts = WriteThruDict('threat_hunts', tenant_id, _store)
		self._vulnerability_scans = WriteThruDict('vulnerability_scans', tenant_id, _store)
		self._patch_reports = WriteThruDict('patch_reports', tenant_id, _store)
		self._siem_rules = WriteThruDict('siem_rules', tenant_id, _store)
		self._threat_intel_feeds = WriteThruDict('threat_intel_feeds', tenant_id, _store)

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ existing

	def create_detection(
		self,
		tenant_id: str,
		title: str,
		alert_source: str,
		anomaly_confidence: float,
		severity: str = "medium",
		signal_refs: list[str] | None = None,
		triage_review_recorded: bool = False,
		owner: str | None = None,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(title or "").strip():
			raise ValueError("detection_title_required")
		context = {
			"tenant_context_present": True,
			"operation": "create_detection",
			"alert_source_present": bool(str(alert_source or "").strip()),
			"anomaly_confidence": normalize_confidence(anomaly_confidence),
			"triage_review_recorded": bool(triage_review_recorded),
			"event_stream": str(event_stream or "").strip().lower(),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		status = "review_required" if result["decision"] == "require_review" else "new"
		record = DetectionRecord(
			id=stable_id("seop_detection", tenant_id, title, alert_source, len(self.detections)),
			tenant_id=tenant_id,
			title=title,
			alert_source=alert_source,
			severity=normalize_severity(severity),
			anomaly_confidence=context["anomaly_confidence"],
			status=status,
			signal_refs=sorted({str(ref) for ref in signal_refs or [] if str(ref).strip()}),
			owner=owner,
			matched_rules=list(result["matched_rules"]),
			required_actions=response_required_actions(result),
		)
		self.detections[record.id] = record
		self._record_event(
			tenant_id,
			"detection_created",
			record.id,
			f"Detection created: {title}",
			owner or alert_source,
			record.severity,
			{"event_stream": event_stream_name(), "processor": "bytewax"},
		)
		return record.to_dict()

	def open_incident(
		self,
		tenant_id: str,
		title: str,
		owner: str,
		severity: str,
		detection_ids: list[str] | None = None,
		escalation_recorded: bool = False,
		evidence_refs: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(title or "").strip():
			raise ValueError("incident_title_required")
		normalized_severity = normalize_severity(severity)
		context = {
			"tenant_context_present": True,
			"operation": "open_incident",
			"incident_owner_assigned": bool(str(owner or "").strip()),
			"incident_severity": normalized_severity,
			"escalation_recorded": bool(escalation_recorded),
			"evidence_attached": bool(detection_ids or evidence_refs),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = IncidentRecord(
			id=stable_id("seop_incident", tenant_id, title, len(self.incidents)),
			tenant_id=tenant_id,
			title=title,
			owner=owner,
			severity=normalized_severity,
			status="escalated" if escalation_recorded else "open",
			detection_ids=sorted({str(item) for item in detection_ids or [] if str(item).strip()}),
			evidence_refs=sorted({str(item) for item in evidence_refs or [] if str(item).strip()}),
			escalation_recorded=bool(escalation_recorded),
		)
		self.incidents[record.id] = record
		for detection_id in record.detection_ids:
			if detection_id in self.detections:
				self.detections[detection_id].status = "linked"
		self._record_event(
			tenant_id,
			"incident_opened",
			record.id,
			f"Incident opened: {title}",
			owner,
			normalized_severity,
			{"event_stream": event_stream_name(), "detection_count": len(record.detection_ids)},
		)
		return record.to_dict()

	def approve_playbook(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		steps: list[str],
		approved_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(name or "").strip():
			raise ValueError("playbook_name_required")
		if not str(owner or "").strip():
			raise ValueError("playbook_owner_required")
		if not steps:
			raise ValueError("playbook_steps_required")
		if not str(approved_by or "").strip():
			raise PermissionError("playbook_approval_required")
		record = PlaybookRecord(
			id=stable_id("seop_playbook", tenant_id, name),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			steps=[str(step) for step in steps],
			approved_by=approved_by,
		)
		self.playbooks[record.id] = record
		self._record_event(tenant_id, "playbook_approved", record.id, f"Playbook approved: {name}", approved_by)
		return record.to_dict()

	def execute_response(
		self,
		tenant_id: str,
		incident_id: str,
		playbook_id: str,
		action: str,
		actor: str,
		containment_reviewed: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		incident = self._get_incident(tenant_id, incident_id)
		playbook = self._get_playbook(tenant_id, playbook_id)
		context = {
			"tenant_context_present": True,
			"operation": "execute_response",
			"playbook_approved": bool(playbook.approved),
			"containment_review_recorded": bool(containment_reviewed),
			"response_actor_present": bool(str(actor or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = ResponseActionRecord(
			id=stable_id("seop_response", tenant_id, incident_id, playbook_id, action, len(self.responses)),
			tenant_id=tenant_id,
			incident_id=incident_id,
			playbook_id=playbook_id,
			action=action,
			actor=actor,
			status="executed",
			required_actions=response_required_actions(result),
		)
		self.responses[record.id] = record
		incident.status = "responding"
		self._record_event(
			tenant_id,
			"response_executed",
			record.id,
			f"Response executed: {action}",
			actor,
			incident.severity,
			{"event_stream": event_stream_name(), "playbook_id": playbook_id},
		)
		return record.to_dict()

	def record_posture_control(
		self,
		tenant_id: str,
		control_id: str,
		domain: str,
		coverage: float,
		owner: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(control_id or "").strip():
			raise ValueError("control_id_required")
		value = float(coverage)
		if value < 0 or value > 1:
			raise ValueError("coverage_out_of_range")
		status = "covered" if value >= 0.9 else "gap" if value < 0.6 else "partial"
		record = PostureControlRecord(
			id=stable_id("seop_posture", tenant_id, control_id),
			tenant_id=tenant_id,
			control_id=control_id,
			domain=domain,
			coverage=round(value, 3),
			owner=owner,
			status=status,
		)
		self.posture_controls[record.id] = record
		return record.to_dict()

	def close_incident(
		self,
		tenant_id: str,
		incident_id: str,
		closure_evidence: str,
		actor: str,
		post_incident_review: str = "",
		compliance_mapping: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		incident = self._get_incident(tenant_id, incident_id)
		if not str(closure_evidence or "").strip():
			raise PermissionError("closure_evidence_required")
		context = {
			"tenant_context_present": True,
			"operation": "close_incident",
			"post_incident_review_present": bool(str(post_incident_review or "").strip()),
			"compliance_mapping_present": bool(str(compliance_mapping or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		incident.status = "closed"
		incident.closed_at = utc_now()
		incident.evidence_refs.append(closure_evidence)
		incident.evidence_refs.extend([value for value in [post_incident_review, compliance_mapping] if str(value or "").strip()])
		self._record_event(
			tenant_id,
			"incident_closed",
			incident.id,
			f"Incident closed: {incident.title}",
			actor,
			incident.severity,
			{"event_stream": event_stream_name(), "review": post_incident_review, "compliance": compliance_mapping},
		)
		return incident.to_dict()

	# ------------------------------------------------------------------ new methods

	def create_soc_alert(
		self,
		tenant_id: str,
		alert_id: str,
		alert_source: str,
		severity: str,
		description: str,
		iocs: list[str] | None = None,
		assigned_to: str | None = None,
	) -> dict[str, Any]:
		"""Create a SOC alert from any detection source with attached indicators of compromise."""
		self._require_tenant(tenant_id)
		assert bool(alert_source), "alert_source required"
		assert bool(description), "description required"
		norm_severity = normalize_severity(severity)
		alert = {
			"id": alert_id,
			"tenant_id": tenant_id,
			"alert_source": alert_source,
			"severity": norm_severity,
			"description": description,
			"iocs": list(iocs or []),
			"assigned_to": assigned_to,
			"status": "new",
			"triage_status": "pending",
			"created_at": _utc_now_iso(),
		}
		self._soc_alerts[f"{tenant_id}:{alert_id}"] = alert
		self._record_event(tenant_id, "soc_alert_created", alert_id, f"SOC alert: {description[:60]}", assigned_to or alert_source, norm_severity, {"ioc_count": len(alert["iocs"])})
		return alert

	def triage_alert(
		self,
		tenant_id: str,
		alert_id: str,
		analyst_id: str,
		disposition: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Triage a SOC alert with an analyst disposition (true_positive, false_positive, informational)."""
		self._require_tenant(tenant_id)
		alert = self._get_soc_alert(tenant_id, alert_id)
		assert disposition in {"true_positive", "false_positive", "informational", "undetermined"}, f"invalid disposition: {disposition}"
		assert bool(analyst_id), "analyst_id required"
		alert["triage_status"] = "triaged"
		alert["disposition"] = disposition
		alert["triaged_by"] = analyst_id
		alert["triage_notes"] = notes
		alert["triaged_at"] = _utc_now_iso()
		alert["status"] = "open" if disposition == "true_positive" else "closed"
		self._record_event(tenant_id, "alert_triaged", alert_id, f"Alert triaged: {disposition}", analyst_id, alert["severity"])
		return alert

	def escalate_to_incident(
		self,
		tenant_id: str,
		alert_id: str,
		escalation_reason: str,
		owner: str,
		severity: str | None = None,
	) -> dict[str, Any]:
		"""Promote a triaged SOC alert to a full incident record."""
		self._require_tenant(tenant_id)
		alert = self._get_soc_alert(tenant_id, alert_id)
		assert bool(escalation_reason), "escalation_reason required"
		assert bool(owner), "owner required"
		eff_severity = severity or alert.get("severity", "medium")
		incident = self.open_incident(
			tenant_id=tenant_id,
			title=f"[Escalated] {alert['description'][:80]}",
			owner=owner,
			severity=eff_severity,
			escalation_recorded=True,
			evidence_refs=[alert_id],
		)
		alert["status"] = "escalated"
		alert["escalated_to_incident"] = incident["id"]
		alert["escalation_reason"] = escalation_reason
		alert["escalated_at"] = _utc_now_iso()
		self._record_event(tenant_id, "alert_escalated", alert_id, f"Alert escalated: {escalation_reason}", owner, eff_severity, {"incident_id": incident["id"]})
		return {"alert": alert, "incident": incident}

	def incident_response(
		self,
		tenant_id: str,
		incident_id: str,
		phase: str,
		actions_taken: list[str],
		analyst_id: str,
		evidence_refs: list[str] | None = None,
	) -> dict[str, Any]:
		"""Record a structured incident response phase entry (contain/eradicate/recover/learn)."""
		self._require_tenant(tenant_id)
		assert phase in {"identification", "containment", "eradication", "recovery", "lessons_learned"}, f"invalid phase: {phase}"
		assert bool(actions_taken), "at least one action required"
		assert bool(analyst_id), "analyst_id required"
		incident = self._get_incident(tenant_id, incident_id)
		phase_record = {
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"phase": phase,
			"actions_taken": list(actions_taken),
			"analyst_id": analyst_id,
			"evidence_refs": list(evidence_refs or []),
			"recorded_at": _utc_now_iso(),
		}
		# append evidence to incident
		for ref in evidence_refs or []:
			if ref not in incident.evidence_refs:
				incident.evidence_refs.append(ref)
		self._record_event(tenant_id, "incident_response_phase", incident_id, f"Phase {phase}: {len(actions_taken)} actions", analyst_id, incident.severity, {"phase": phase})
		return phase_record

	def threat_hunt(
		self,
		tenant_id: str,
		hunt_id: str,
		hypothesis: str,
		data_sources: list[str],
		period: str,
		analyst_id: str,
		ttps: list[str] | None = None,
	) -> dict[str, Any]:
		"""Conduct a structured threat hunt based on a hypothesis and specified data sources."""
		self._require_tenant(tenant_id)
		assert bool(hypothesis), "hypothesis required"
		assert bool(data_sources), "at least one data_source required"
		assert bool(analyst_id), "analyst_id required"
		hunt = {
			"id": hunt_id,
			"tenant_id": tenant_id,
			"hypothesis": hypothesis,
			"data_sources": list(data_sources),
			"period": period,
			"analyst_id": analyst_id,
			"ttps": list(ttps or []),
			"findings": [],
			"status": "in_progress",
			"started_at": _utc_now_iso(),
		}
		self._threat_hunts[f"{tenant_id}:{hunt_id}"] = hunt
		self._record_event(tenant_id, "threat_hunt_started", hunt_id, f"Threat hunt: {hypothesis[:60]}", analyst_id, "low", {"data_source_count": len(data_sources)})
		return hunt

	def vulnerability_management(
		self,
		tenant_id: str,
		scan_id: str,
		asset_id: str,
		scan_results: list[dict[str, Any]],
		scanner: str = "openvas",
		owner: str = "secops",
	) -> dict[str, Any]:
		"""Ingest a vulnerability scan result set for an asset and compute risk metrics."""
		self._require_tenant(tenant_id)
		assert bool(asset_id), "asset_id required"
		critical = [v for v in scan_results if v.get("severity") == "critical"]
		high = [v for v in scan_results if v.get("severity") == "high"]
		medium = [v for v in scan_results if v.get("severity") == "medium"]
		low = [v for v in scan_results if v.get("severity") in {"low", "info"}]
		cvss_scores = [float(v["cvss"]) for v in scan_results if "cvss" in v]
		scan = {
			"id": scan_id,
			"tenant_id": tenant_id,
			"asset_id": asset_id,
			"scanner": scanner,
			"owner": owner,
			"total_findings": len(scan_results),
			"critical_count": len(critical),
			"high_count": len(high),
			"medium_count": len(medium),
			"low_count": len(low),
			"avg_cvss": round(statistics.mean(cvss_scores), 2) if cvss_scores else None,
			"max_cvss": max(cvss_scores) if cvss_scores else None,
			"risk_score": len(critical) * 10 + len(high) * 5 + len(medium) * 2 + len(low),
			"findings": scan_results,
			"status": "completed",
			"scanned_at": _utc_now_iso(),
		}
		self._vulnerability_scans[f"{tenant_id}:{scan_id}"] = scan
		self._record_event(tenant_id, "vulnerability_scan_ingested", scan_id, f"Vuln scan: {asset_id}", owner, "medium" if scan["critical_count"] == 0 else "critical", {"asset_id": asset_id})
		return scan

	def patch_compliance_report(
		self,
		tenant_id: str,
		report_id: str,
		asset_group: str,
		period: str,
		owner: str = "secops",
	) -> dict[str, Any]:
		"""Generate a patch compliance summary for an asset group."""
		self._require_tenant(tenant_id)
		scans = [
			s for s in self._vulnerability_scans.values()
			if s["tenant_id"] == tenant_id
			and (asset_group == "all" or s["asset_id"].startswith(asset_group))
		]
		total_findings = sum(s["total_findings"] for s in scans)
		critical_open = sum(s["critical_count"] for s in scans)
		high_open = sum(s["high_count"] for s in scans)
		compliance_pct = max(0.0, round((1 - (critical_open + high_open) / max(total_findings, 1)) * 100, 2))
		report = {
			"id": report_id,
			"tenant_id": tenant_id,
			"asset_group": asset_group,
			"period": period,
			"owner": owner,
			"asset_count": len(scans),
			"total_findings": total_findings,
			"critical_open": critical_open,
			"high_open": high_open,
			"compliance_percent": compliance_pct,
			"compliant": compliance_pct >= 95.0,
			"generated_at": _utc_now_iso(),
		}
		self._patch_reports[f"{tenant_id}:{report_id}"] = report
		self._record_event(tenant_id, "patch_compliance_report_generated", report_id, f"Patch compliance: {asset_group}", owner, "low", {"compliance_pct": compliance_pct})
		return report

	def siem_rule_management(
		self,
		tenant_id: str,
		rule_id: str,
		action: str,
		parameters: dict[str, Any],
		managed_by: str = "secops",
	) -> dict[str, Any]:
		"""Create, update, enable, or disable a SIEM detection rule."""
		self._require_tenant(tenant_id)
		assert action in {"create", "update", "enable", "disable", "delete"}, f"invalid action: {action}"
		assert bool(rule_id), "rule_id required"
		key = f"{tenant_id}:{rule_id}"
		if action == "create":
			rule = {
				"id": rule_id,
				"tenant_id": tenant_id,
				"name": parameters.get("name", rule_id),
				"logic": parameters.get("logic", ""),
				"severity": normalize_severity(parameters.get("severity", "medium")),
				"mitre_tactics": parameters.get("mitre_tactics", []),
				"enabled": True,
				"version": 1,
				"managed_by": managed_by,
				"created_at": _utc_now_iso(),
				"updated_at": _utc_now_iso(),
			}
			self._siem_rules[key] = rule
		elif key not in self._siem_rules:
			raise KeyError(f"siem_rule_not_found:{rule_id}")
		else:
			rule = self._siem_rules[key]
			if action == "update":
				rule.update({k: v for k, v in parameters.items() if k not in {"id", "tenant_id", "created_at"}})
				rule["version"] = rule.get("version", 1) + 1
				rule["updated_at"] = _utc_now_iso()
			elif action == "enable":
				rule["enabled"] = True
				rule["updated_at"] = _utc_now_iso()
			elif action == "disable":
				rule["enabled"] = False
				rule["updated_at"] = _utc_now_iso()
			elif action == "delete":
				del self._siem_rules[key]
				self._record_event(tenant_id, "siem_rule_deleted", rule_id, f"SIEM rule deleted: {rule_id}", managed_by)
				return {"id": rule_id, "status": "deleted"}
		self._record_event(tenant_id, f"siem_rule_{action}d", rule_id, f"SIEM rule {action}: {rule_id}", managed_by)
		return rule

	def soc_metrics_dashboard(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a comprehensive SOC metrics dashboard for a given period."""
		self._require_tenant(tenant_id)
		detections = self.list_detections(tenant_id)
		incidents = self.list_incidents(tenant_id)
		alerts = [a for a in self._soc_alerts.values() if a["tenant_id"] == tenant_id]
		hunts = [h for h in self._threat_hunts.values() if h["tenant_id"] == tenant_id]
		scans = [s for s in self._vulnerability_scans.values() if s["tenant_id"] == tenant_id]
		open_incidents = [i for i in incidents if i["status"] not in {"closed"}]
		closed_incidents = [i for i in incidents if i["status"] == "closed"]
		true_positives = [a for a in alerts if a.get("disposition") == "true_positive"]
		false_positives = [a for a in alerts if a.get("disposition") == "false_positive"]
		precision = round(len(true_positives) / max(len(alerts), 1) * 100, 2)
		posture_controls = self.list_posture_controls(tenant_id)
		avg_coverage = round(statistics.mean([c["coverage"] for c in posture_controls]), 3) if posture_controls else None
		return {
			"tenant_id": tenant_id,
			"period": period,
			"alert_count": len(alerts),
			"true_positive_count": len(true_positives),
			"false_positive_count": len(false_positives),
			"detection_precision_pct": precision,
			"detection_count": len(detections),
			"incident_count": len(incidents),
			"open_incident_count": len(open_incidents),
			"closed_incident_count": len(closed_incidents),
			"critical_incident_count": sum(1 for i in incidents if i["severity"] == "critical"),
			"threat_hunt_count": len(hunts),
			"vulnerability_scan_count": len(scans),
			"total_vulnerabilities": sum(s["total_findings"] for s in scans),
			"critical_vulnerabilities": sum(s["critical_count"] for s in scans),
			"siem_rule_count": len([r for r in self._siem_rules.values() if r["tenant_id"] == tenant_id]),
			"enabled_siem_rules": len([r for r in self._siem_rules.values() if r["tenant_id"] == tenant_id and r.get("enabled")]),
			"avg_posture_coverage": avg_coverage,
			"seop_agent_count": len(self.list_seop_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": streaming_manifest(),
			"generated_at": _utc_now_iso(),
		}

	def threat_intelligence_integration(
		self,
		tenant_id: str,
		feed_id: str,
		indicators: list[dict[str, Any]],
		feed_type: str = "STIX",
		source: str = "external",
		analyst_id: str = "system",
	) -> dict[str, Any]:
		"""Ingest threat intelligence indicators from an external feed."""
		self._require_tenant(tenant_id)
		assert bool(feed_id), "feed_id required"
		assert bool(indicators), "indicators must be non-empty"
		ioc_types: dict[str, int] = {}
		for indicator in indicators:
			ioc_type = str(indicator.get("type", "unknown"))
			ioc_types[ioc_type] = ioc_types.get(ioc_type, 0) + 1
		feed = {
			"id": feed_id,
			"tenant_id": tenant_id,
			"feed_type": feed_type,
			"source": source,
			"indicator_count": len(indicators),
			"ioc_type_breakdown": ioc_types,
			"indicators": indicators,
			"analyst_id": analyst_id,
			"status": "active",
			"ingested_at": _utc_now_iso(),
		}
		self._threat_intel_feeds[f"{tenant_id}:{feed_id}"] = feed
		self._record_event(tenant_id, "threat_intel_ingested", feed_id, f"TI feed ingested: {feed_id}", analyst_id, "low", {"indicator_count": len(indicators), "feed_type": feed_type})
		return feed

	# ------------------------------------------------------------------ compat / list

	def alert_triage(
		self,
		tenant_id: str,
		alert_id: str,
		analyst_id: str,
		disposition: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Triage a SOC alert (alias for triage_alert with explicit naming)."""
		return self.triage_alert(
			tenant_id=tenant_id,
			alert_id=alert_id,
			analyst_id=analyst_id,
			disposition=disposition,
			notes=notes,
		)

	def threat_hunt(
		self,
		tenant_id: str,
		hunt_id: str,
		hypothesis: str,
		data_sources: list[str],
		period: str,
		analyst_id: str,
		ttps: list[str] | None = None,
	) -> dict[str, Any]:
		"""Conduct a structured threat hunt (alias for threat_hunt on _threat_hunts store)."""
		self._require_tenant(tenant_id)
		assert bool(hypothesis), "hypothesis required"
		assert bool(data_sources), "at least one data_source required"
		hunt = {
			"id": hunt_id,
			"tenant_id": tenant_id,
			"hypothesis": hypothesis,
			"data_sources": list(data_sources),
			"period": period,
			"analyst_id": analyst_id,
			"ttps": list(ttps or []),
			"findings": [],
			"status": "in_progress",
			"started_at": _utc_now_iso(),
		}
		self._threat_hunts[f"{tenant_id}:{hunt_id}"] = hunt
		self._record_event(tenant_id, "threat_hunt_started", hunt_id, f"Hunt: {hypothesis[:60]}", analyst_id, "low", {"data_source_count": len(data_sources)})
		return hunt

	def vulnerability_scan(
		self,
		tenant_id: str,
		scan_id: str,
		asset_id: str,
		scan_results: list[dict[str, Any]],
		scanner: str = "openvas",
		owner: str = "secops",
	) -> dict[str, Any]:
		"""Ingest a vulnerability scan result (alias for vulnerability_management)."""
		return self.vulnerability_management(
			tenant_id=tenant_id,
			scan_id=scan_id,
			asset_id=asset_id,
			scan_results=scan_results,
			scanner=scanner,
			owner=owner,
		)

	def patch_status(
		self,
		tenant_id: str,
		report_id: str,
		asset_group: str,
		period: str,
		owner: str = "secops",
	) -> dict[str, Any]:
		"""Return patch compliance status for an asset group (alias for patch_compliance_report)."""
		return self.patch_compliance_report(
			tenant_id=tenant_id,
			report_id=report_id,
			asset_group=asset_group,
			period=period,
			owner=owner,
		)

	def incident_timeline(
		self,
		tenant_id: str,
		incident_id: str,
	) -> dict[str, Any]:
		"""Return a chronological audit trail for a specific incident."""
		self._require_tenant(tenant_id)
		incident = self._get_incident(tenant_id, incident_id)
		events = [
			e.to_dict() for e in self.audit_events.values()
			if e.tenant_id == tenant_id and e.subject_id == incident_id
		]
		events.sort(key=lambda x: x.get("created_at", ""))
		return {
			"incident_id": incident_id,
			"tenant_id": tenant_id,
			"title": incident.title,
			"severity": incident.severity,
			"status": incident.status,
			"event_count": len(events),
			"timeline": events,
		}

	def indicator_block(
		self,
		tenant_id: str,
		indicator_id: str,
		indicator_type: str,
		indicator_value: str,
		reason: str,
		blocked_by: str = "secops",
	) -> dict[str, Any]:
		"""Block a threat indicator (IP, domain, hash, URL) across security controls."""
		self._require_tenant(tenant_id)
		assert indicator_type in {"ip", "domain", "hash", "url", "email", "certificate"}, f"unsupported type: {indicator_type}"
		assert bool(indicator_value), "indicator_value required"
		record = {
			"indicator_id": indicator_id,
			"tenant_id": tenant_id,
			"type": indicator_type,
			"value": indicator_value,
			"reason": reason,
			"blocked_by": blocked_by,
			"status": "blocked",
			"blocked_at": _utc_now_iso(),
		}
		self._record_event(tenant_id, "indicator_blocked", indicator_id, f"Blocked {indicator_type}: {indicator_value[:40]}", blocked_by, "medium", {"type": indicator_type})
		return record

	def honeypot_trigger(
		self,
		tenant_id: str,
		honeypot_id: str,
		triggered_by_ip: str,
		event_type: str = "access",
		payload: dict[str, Any] | None = None,
		analyst_id: str = "system",
	) -> dict[str, Any]:
		"""Record a honeypot trigger event and auto-create a SOC alert."""
		self._require_tenant(tenant_id)
		alert_id = f"honeypot_alert:{honeypot_id}:{triggered_by_ip}"
		alert = self.create_soc_alert(
			tenant_id=tenant_id,
			alert_id=alert_id,
			alert_source=f"honeypot:{honeypot_id}",
			severity="high",
			description=f"Honeypot {honeypot_id} triggered by {triggered_by_ip}",
			iocs=[triggered_by_ip],
			assigned_to=analyst_id,
		)
		return {
			"honeypot_id": honeypot_id,
			"tenant_id": tenant_id,
			"triggered_by_ip": triggered_by_ip,
			"event_type": event_type,
			"payload": payload or {},
			"alert": alert,
			"triggered_at": _utc_now_iso(),
		}

	def forensic_capture(
		self,
		tenant_id: str,
		capture_id: str,
		incident_id: str,
		artifact_type: str,
		artifact_ref: str,
		chain_of_custody: str,
		captured_by: str = "secops",
	) -> dict[str, Any]:
		"""Record a digital forensic artefact for an incident with chain-of-custody."""
		self._require_tenant(tenant_id)
		assert artifact_type in {"memory_dump", "disk_image", "log_export", "network_capture", "screenshot", "config_snapshot"}, f"unsupported artifact_type: {artifact_type}"
		assert bool(chain_of_custody), "chain_of_custody required"
		incident = self._get_incident(tenant_id, incident_id)
		if artifact_ref not in incident.evidence_refs:
			incident.evidence_refs.append(artifact_ref)
		record = {
			"capture_id": capture_id,
			"tenant_id": tenant_id,
			"incident_id": incident_id,
			"artifact_type": artifact_type,
			"artifact_ref": artifact_ref,
			"chain_of_custody": chain_of_custody,
			"captured_by": captured_by,
			"captured_at": _utc_now_iso(),
		}
		self._record_event(tenant_id, "forensic_capture_recorded", capture_id, f"Artifact {artifact_type}", captured_by, "medium", {"incident_id": incident_id})
		return record

	def soc_metrics(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return SOC operational metrics dashboard (alias for soc_metrics_dashboard)."""
		return self.soc_metrics_dashboard(tenant_id=tenant_id, period=period)

	def playbook_run(
		self,
		tenant_id: str,
		incident_id: str,
		playbook_id: str,
		action: str,
		actor: str,
		containment_reviewed: bool = True,
	) -> dict[str, Any]:
		"""Execute a playbook response action (alias for execute_response)."""
		return self.execute_response(
			tenant_id=tenant_id,
			incident_id=incident_id,
			playbook_id=playbook_id,
			action=action,
			actor=actor,
			containment_reviewed=containment_reviewed,
		)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_detection(
			tenant_id=tenant_id,
			title=record_id,
			alert_source=str(metadata.get("alert_source") or "compatibility"),
			anomaly_confidence=float(metadata.get("anomaly_confidence", 0.1)),
			severity=str(metadata.get("severity") or "medium"),
			triage_review_recorded=status in {"reviewed", "closed"},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_detections(tenant_id)

	def list_detections(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.detections, tenant_id)

	def list_incidents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.incidents, tenant_id)

	def list_playbooks(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.playbooks, tenant_id)

	def list_responses(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.responses, tenant_id)

	def list_posture_controls(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.posture_controls, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def list_soc_alerts(self, tenant_id: str) -> list[dict[str, Any]]:
		return [a for a in self._soc_alerts.values() if a["tenant_id"] == tenant_id]

	def list_threat_hunts(self, tenant_id: str) -> list[dict[str, Any]]:
		return [h for h in self._threat_hunts.values() if h["tenant_id"] == tenant_id]

	def list_vulnerability_scans(self, tenant_id: str) -> list[dict[str, Any]]:
		return [s for s in self._vulnerability_scans.values() if s["tenant_id"] == tenant_id]

	def list_siem_rules(self, tenant_id: str) -> list[dict[str, Any]]:
		return [r for r in self._siem_rules.values() if r["tenant_id"] == tenant_id]

	def list_threat_intel_feeds(self, tenant_id: str) -> list[dict[str, Any]]:
		return [f for f in self._threat_intel_feeds.values() if f["tenant_id"] == tenant_id]

	def register_seop_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str = "secops",
		human_approval_required: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		context = {
			"tenant_context_present": True,
			"operation": "register_seop_agent",
			"agent_runtime_supported": runtime_value in SUPPORTED_SEOP_AGENT_RUNTIMES,
			"agent_role_supported": role_value in SUPPORTED_SEOP_AGENT_ROLES,
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(name or "").strip():
			raise ValueError("seop_agent_name_required")
		if not str(scope or "").strip():
			raise ValueError("seop_agent_scope_required")
		record = SeopAgentRecord(
			id=stable_id("seop_agent", tenant_id, name, runtime_value, role_value),
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=scope,
			owner=owner,
			human_approval_required=bool(human_approval_required),
		)
		self.seop_agents[record.id] = record
		self._record_event(
			tenant_id,
			"seop_agent_registered",
			record.id,
			f"SEOP agent registered: {name}",
			owner,
			"low",
			{"runtime": runtime_value, "role": role_value, "event_stream": event_stream_name()},
		)
		return record.to_dict()

	def validate_agent_response_action(
		self,
		tenant_id: str,
		agent_id: str,
		incident_severity: str,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		agent = self.seop_agents.get(agent_id)
		if agent is None or agent.tenant_id != tenant_id:
			raise KeyError(f"seop_agent_not_found:{agent_id}")
		context = {
			"tenant_context_present": True,
			"operation": "agent_response_action",
			"incident_severity": normalize_severity(incident_severity),
			"human_approval_recorded": bool(human_approval_recorded),
		}
		return self.evaluate(context)

	def list_seop_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.seop_agents, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		incidents = self.list_incidents(tenant_id)
		detections = self.list_detections(tenant_id)
		return {
			"tenant_id": tenant_id,
			"detection_count": len(detections),
			"review_required_count": sum(1 for item in detections if item["status"] == "review_required"),
			"incident_count": len(incidents),
			"open_incident_count": sum(1 for item in incidents if item["status"] != "closed"),
			"critical_incident_count": sum(1 for item in incidents if item["severity"] == "critical"),
			"approved_playbook_count": len(self.list_playbooks(tenant_id)),
			"response_count": len(self.list_responses(tenant_id)),
			"posture_gap_count": sum(1 for item in self.list_posture_controls(tenant_id) if item["status"] == "gap"),
			"soc_alert_count": len(self.list_soc_alerts(tenant_id)),
			"threat_hunt_count": len(self.list_threat_hunts(tenant_id)),
			"vulnerability_scan_count": len(self.list_vulnerability_scans(tenant_id)),
			"siem_rule_count": len(self.list_siem_rules(tenant_id)),
			"threat_intel_feed_count": len(self.list_threat_intel_feeds(tenant_id)),
			"seop_agent_count": len(self.list_seop_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
			"streaming": streaming_manifest(),
		}

	# ------------------------------------------------------------------ internals

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "seop_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "seop_policy_blocked")

	def _get_incident(self, tenant_id: str, incident_id: str) -> IncidentRecord:
		incident = self.incidents.get(incident_id)
		if incident is None or incident.tenant_id != tenant_id:
			raise KeyError(f"incident_not_found:{incident_id}")
		return incident

	def _get_playbook(self, tenant_id: str, playbook_id: str) -> PlaybookRecord:
		playbook = self.playbooks.get(playbook_id)
		if playbook is None or playbook.tenant_id != tenant_id:
			raise KeyError(f"playbook_not_found:{playbook_id}")
		return playbook

	def _get_soc_alert(self, tenant_id: str, alert_id: str) -> dict[str, Any]:
		alert = self._soc_alerts.get(f"{tenant_id}:{alert_id}")
		if alert is None:
			raise KeyError(f"soc_alert_not_found:{alert_id}")
		return alert

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		record = OpsAuditEventRecord(
			id=stable_id("seop_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=normalize_severity(severity),
			metadata=dict(metadata or {}),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_soc_alerts', '_threat_hunts', '_vulnerability_scans', '_patch_reports', '_siem_rules', '_threat_intel_feeds']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

