"""
Compliance management service — extended methods for APG COMP capability.

Adds 16 new methods to reach 42+ total:
	obligation_register, control_map, gap_assess, evidence_collect,
	risk_score, policy_publish, training_assign, audit_schedule,
	finding_report, remediation_track, regulatory_change_alert,
	compliance_dashboard, iso_27001_checklist, gdpr_dpia, soc2_evidence,
	health_check, bulk_create_controls, bulk_close_findings,
	export_compliance_data, export_framework_csv

These are implemented as async methods on CompServiceExtended which
extends CompService.  Import and use CompServiceExtended instead of
CompService to get the full 42+ method surface.

© 2025 Datacraft · www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from typing import Any

from .compliance_engine import stable_digest
from .models import (
	ComplianceControl,
	ComplianceFinding,
	EvidenceRecord,
	utc_now,
)
from .service import CompService


class CompServiceExtended(CompService):
	"""CompService + 16 domain-specific async methods (42+ total)."""

	# ------------------------------------------------------------------
	# Extended state stores
	# ------------------------------------------------------------------

	def __init__(self) -> None:
		super().__init__()
		self._obligations: dict[str, dict[str, Any]] = {}		# obligation_id → record
		self._control_maps: dict[str, dict[str, Any]] = {}		# map_id → record
		self._gap_assessments: dict[str, dict[str, Any]] = {}	# gap_id → record
		self._risk_scores: dict[str, dict[str, Any]] = {}		# score_id → record
		self._policies: dict[str, dict[str, Any]] = {}			# policy_id → record
		self._training_assignments: dict[str, dict[str, Any]] = {}
		self._audit_schedules: dict[str, dict[str, Any]] = {}
		self._regulatory_changes: dict[str, dict[str, Any]] = {}
		self._dpia_records: dict[str, dict[str, Any]] = {}
		self._soc2_evidence: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------ 1
	async def obligation_register(
		self,
		obligation_id: str,
		tenant_id: str,
		framework_id: str,
		title: str,
		description: str,
		owner: str,
		due_date: datetime | None = None,
		regulation_ref: str = "",
	) -> dict[str, Any]:
		"""Register a compliance obligation against a framework."""
		assert obligation_id and tenant_id and framework_id and title and owner, "All fields required"
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		key = self._key(tenant_id, obligation_id)
		if key in self._obligations:
			raise ValueError(f"obligation_already_exists:{obligation_id}")
		record: dict[str, Any] = {
			"id": obligation_id,
			"tenant_id": tenant_id,
			"framework_id": framework_id,
			"title": title,
			"description": description,
			"owner": owner,
			"regulation_ref": regulation_ref,
			"due_date": due_date.isoformat() if due_date else None,
			"status": "open",
			"created_at": utc_now().isoformat(),
		}
		self._obligations[key] = record
		self._record_audit(tenant_id, "obligation_registered", obligation_id, owner, record)
		return record

	# ------------------------------------------------------------------ 2
	async def control_map(
		self,
		map_id: str,
		tenant_id: str,
		source_framework_id: str,
		target_framework_id: str,
		mappings: list[dict[str, str]],
		mapped_by: str,
	) -> dict[str, Any]:
		"""Map controls between two compliance frameworks."""
		assert map_id and tenant_id and source_framework_id and target_framework_id and mapped_by
		self._require_tenant(tenant_id)
		self._require_framework(source_framework_id, tenant_id)
		self._require_framework(target_framework_id, tenant_id)
		key = self._key(tenant_id, map_id)
		record: dict[str, Any] = {
			"id": map_id,
			"tenant_id": tenant_id,
			"source_framework_id": source_framework_id,
			"target_framework_id": target_framework_id,
			"mappings": list(mappings),
			"mapping_count": len(mappings),
			"mapped_by": mapped_by,
			"created_at": utc_now().isoformat(),
		}
		self._control_maps[key] = record
		self._record_audit(tenant_id, "control_map_created", map_id, mapped_by, record)
		return record

	# ------------------------------------------------------------------ 3
	async def gap_assess(
		self,
		gap_id: str,
		tenant_id: str,
		framework_id: str,
		assessed_by: str,
		now: datetime | None = None,
	) -> dict[str, Any]:
		"""Assess control gaps for a framework relative to current evidence."""
		assert gap_id and tenant_id and framework_id and assessed_by
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		controls = [
			c for c in self._controls.values()
			if c.tenant_id == tenant_id and c.framework_id == framework_id
		]
		assessed_ids = {a.control_id for a in self._assessments.values() if a.tenant_id == tenant_id}
		gaps = [c.id for c in controls if c.id not in assessed_ids]
		open_findings = [
			f.id for f in self._findings.values()
			if f.tenant_id == tenant_id and f.status == "open"
		]
		score = max(0.0, 1.0 - (len(gaps) + len(open_findings)) / max(1, len(controls))) * 100
		key = self._key(tenant_id, gap_id)
		record: dict[str, Any] = {
			"id": gap_id,
			"tenant_id": tenant_id,
			"framework_id": framework_id,
			"assessed_by": assessed_by,
			"assessed_at": (now or utc_now()).isoformat(),
			"total_controls": len(controls),
			"assessed_controls": len(assessed_ids & {c.id for c in controls}),
			"gap_control_ids": gaps,
			"open_finding_ids": open_findings,
			"compliance_score_pct": round(score, 2),
		}
		self._gap_assessments[key] = record
		self._record_audit(tenant_id, "gap_assessment_completed", gap_id, assessed_by, record)
		return record

	# ------------------------------------------------------------------ 4
	async def evidence_collect(
		self,
		evidence_id: str,
		tenant_id: str,
		control_id: str,
		source: str,
		collected_by: str,
		content_hash: str,
		encrypted: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Collect and store evidence for a control with integrity hash."""
		assert evidence_id and tenant_id and control_id and source and collected_by and content_hash
		self._require_tenant(tenant_id)
		self._require_control(control_id, tenant_id)
		return self.record_evidence(
			evidence_id=evidence_id,
			tenant_id=tenant_id,
			control_id=control_id,
			source=source,
			collected_by=collected_by,
			encrypted=encrypted,
			immutable_reference=stable_digest({"hash": content_hash, "id": evidence_id}),
			collected_at=utc_now(),
			metadata=metadata or {},
		)

	# ------------------------------------------------------------------ 5
	async def risk_score(
		self,
		score_id: str,
		tenant_id: str,
		scope: str,
		scored_by: str,
		now: datetime | None = None,
	) -> dict[str, Any]:
		"""Compute an aggregate compliance risk score across all controls."""
		assert score_id and tenant_id and scope and scored_by
		self._require_tenant(tenant_id)
		controls = [c for c in self._controls.values() if c.tenant_id == tenant_id]
		findings = [f for f in self._findings.values() if f.tenant_id == tenant_id and f.status == "open"]
		critical = sum(1 for f in findings if f.severity == "critical")
		high = sum(1 for f in findings if f.severity == "high")
		medium = sum(1 for f in findings if f.severity == "medium")
		raw = (critical * 10 + high * 5 + medium * 2) / max(1, len(controls))
		risk_level = "critical" if raw > 5 else "high" if raw > 2 else "medium" if raw > 0.5 else "low"
		key = self._key(tenant_id, score_id)
		record: dict[str, Any] = {
			"id": score_id,
			"tenant_id": tenant_id,
			"scope": scope,
			"scored_by": scored_by,
			"scored_at": (now or utc_now()).isoformat(),
			"control_count": len(controls),
			"open_findings": len(findings),
			"critical_findings": critical,
			"high_findings": high,
			"medium_findings": medium,
			"risk_score": round(min(raw, 10.0), 3),
			"risk_level": risk_level,
		}
		self._risk_scores[key] = record
		self._record_audit(tenant_id, "risk_scored", score_id, scored_by, record)
		return record

	# ------------------------------------------------------------------ 6
	async def policy_publish(
		self,
		policy_id: str,
		tenant_id: str,
		framework_id: str,
		title: str,
		content: str,
		owner: str,
		version: str = "1.0",
	) -> dict[str, Any]:
		"""Publish a compliance policy document linked to a framework."""
		assert policy_id and tenant_id and framework_id and title and content and owner
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		key = self._key(tenant_id, policy_id)
		if key in self._policies:
			raise ValueError(f"policy_already_exists:{policy_id}")
		record: dict[str, Any] = {
			"id": policy_id,
			"tenant_id": tenant_id,
			"framework_id": framework_id,
			"title": title,
			"content_hash": stable_digest({"content": content}),
			"owner": owner,
			"version": version,
			"status": "published",
			"published_at": utc_now().isoformat(),
		}
		self._policies[key] = record
		self._record_audit(tenant_id, "policy_published", policy_id, owner, record)
		return record

	# ------------------------------------------------------------------ 7
	async def training_assign(
		self,
		assignment_id: str,
		tenant_id: str,
		control_id: str,
		assignee: str,
		training_name: str,
		due_days: int = 30,
		assigned_by: str = "system",
	) -> dict[str, Any]:
		"""Assign compliance training to a user for a specific control."""
		assert assignment_id and tenant_id and control_id and assignee and training_name
		self._require_tenant(tenant_id)
		self._require_control(control_id, tenant_id)
		due_at = utc_now() + timedelta(days=due_days)
		key = self._key(tenant_id, assignment_id)
		record: dict[str, Any] = {
			"id": assignment_id,
			"tenant_id": tenant_id,
			"control_id": control_id,
			"assignee": assignee,
			"training_name": training_name,
			"assigned_by": assigned_by,
			"due_at": due_at.isoformat(),
			"status": "pending",
			"assigned_at": utc_now().isoformat(),
		}
		self._training_assignments[key] = record
		self._record_audit(tenant_id, "training_assigned", assignment_id, assigned_by, record)
		return record

	# ------------------------------------------------------------------ 8
	async def audit_schedule(
		self,
		schedule_id: str,
		tenant_id: str,
		framework_id: str,
		auditor: str,
		scheduled_date: datetime,
		audit_type: str = "internal",
	) -> dict[str, Any]:
		"""Schedule a compliance audit for a framework."""
		assert schedule_id and tenant_id and framework_id and auditor
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		key = self._key(tenant_id, schedule_id)
		if key in self._audit_schedules:
			raise ValueError(f"audit_schedule_already_exists:{schedule_id}")
		record: dict[str, Any] = {
			"id": schedule_id,
			"tenant_id": tenant_id,
			"framework_id": framework_id,
			"auditor": auditor,
			"audit_type": audit_type,
			"scheduled_date": scheduled_date.isoformat(),
			"status": "scheduled",
			"created_at": utc_now().isoformat(),
		}
		self._audit_schedules[key] = record
		self._record_audit(tenant_id, "audit_scheduled", schedule_id, auditor, record)
		return record

	# ------------------------------------------------------------------ 9
	async def finding_report(
		self,
		tenant_id: str,
		framework_id: str | None = None,
		severity_filter: list[str] | None = None,
	) -> dict[str, Any]:
		"""Generate a structured findings report, optionally filtered."""
		self._require_tenant(tenant_id)
		findings = [f for f in self._findings.values() if f.tenant_id == tenant_id]
		if framework_id:
			ctrl_ids = {
				c.id for c in self._controls.values()
				if c.tenant_id == tenant_id and c.framework_id == framework_id
			}
			findings = [f for f in findings if f.control_id in ctrl_ids]
		if severity_filter:
			findings = [f for f in findings if f.severity in severity_filter]
		by_severity: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for f in findings:
			by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
			by_status[f.status] = by_status.get(f.status, 0) + 1
		return {
			"tenant_id": tenant_id,
			"framework_id": framework_id,
			"severity_filter": severity_filter,
			"total_findings": len(findings),
			"by_severity": by_severity,
			"by_status": by_status,
			"items": [f.to_dict() for f in sorted(findings, key=lambda x: x.id)],
			"generated_at": utc_now().isoformat(),
		}

	# ------------------------------------------------------------------ 10
	async def remediation_track(
		self,
		tenant_id: str,
		finding_id: str,
		update: str,
		updated_by: str,
		new_status: str | None = None,
	) -> dict[str, Any]:
		"""Update remediation progress on an open finding."""
		assert finding_id and update and updated_by
		finding = self._require_finding(finding_id, tenant_id)
		finding.remediation_plan = f"{finding.remediation_plan}\n{utc_now().isoformat()} [{updated_by}]: {update}".strip()
		if new_status and new_status in {"open", "in_progress", "resolved", "accepted"}:
			finding.status = new_status
		self._record_audit(tenant_id, "remediation_tracked", finding_id, updated_by, finding.to_dict() | {"update": update})
		return finding.to_dict()

	# ------------------------------------------------------------------ 11
	async def regulatory_change_alert(
		self,
		alert_id: str,
		tenant_id: str,
		regulation: str,
		summary: str,
		effective_date: datetime,
		impact_level: str = "medium",
		raised_by: str = "system",
	) -> dict[str, Any]:
		"""Record an incoming regulatory change that may require control updates."""
		assert alert_id and tenant_id and regulation and summary
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, alert_id)
		record: dict[str, Any] = {
			"id": alert_id,
			"tenant_id": tenant_id,
			"regulation": regulation,
			"summary": summary,
			"effective_date": effective_date.isoformat(),
			"impact_level": impact_level,
			"raised_by": raised_by,
			"status": "open",
			"created_at": utc_now().isoformat(),
		}
		self._regulatory_changes[key] = record
		self._record_audit(tenant_id, "regulatory_change_alerted", alert_id, raised_by, record)
		return record

	# ------------------------------------------------------------------ 12
	async def compliance_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Return enriched KPI dashboard including risk, gaps, and schedule data."""
		base = self.dashboard_summary(tenant_id)
		# regulatory alerts
		reg_alerts = [r for r in self._regulatory_changes.values() if r["tenant_id"] == tenant_id]
		open_alerts = [r for r in reg_alerts if r["status"] == "open"]
		# gap assessments — latest per framework
		gaps = [g for g in self._gap_assessments.values() if g["tenant_id"] == tenant_id]
		avg_score = (sum(g["compliance_score_pct"] for g in gaps) / len(gaps)) if gaps else None
		# upcoming audits
		schedules = [s for s in self._audit_schedules.values() if s["tenant_id"] == tenant_id and s["status"] == "scheduled"]
		# training assignments overdue
		overdue_training = [
			t for t in self._training_assignments.values()
			if t["tenant_id"] == tenant_id
			and t["status"] == "pending"
			and datetime.fromisoformat(t["due_at"]) < utc_now()
		]
		return {
			**base,
			"avg_compliance_score_pct": round(avg_score, 2) if avg_score is not None else None,
			"gap_assessment_count": len(gaps),
			"regulatory_alert_count": len(reg_alerts),
			"open_regulatory_alert_count": len(open_alerts),
			"upcoming_audit_count": len(schedules),
			"overdue_training_count": len(overdue_training),
			"risk_score_count": len([r for r in self._risk_scores.values() if r["tenant_id"] == tenant_id]),
			"policy_count": len([p for p in self._policies.values() if p["tenant_id"] == tenant_id]),
		}

	# ------------------------------------------------------------------ 13
	async def iso_27001_checklist(self, tenant_id: str) -> dict[str, Any]:
		"""Evaluate ISO 27001 readiness against stored controls and evidence."""
		self._require_tenant(tenant_id)
		iso_domains = [
			"A.5 Information security policies",
			"A.6 Organization of information security",
			"A.7 Human resource security",
			"A.8 Asset management",
			"A.9 Access control",
			"A.10 Cryptography",
			"A.11 Physical and environmental security",
			"A.12 Operations security",
			"A.13 Communications security",
			"A.14 System acquisition, development and maintenance",
			"A.15 Supplier relationships",
			"A.16 Information security incident management",
			"A.17 Business continuity management",
			"A.18 Compliance",
		]
		controls = [c for c in self._controls.values() if c.tenant_id == tenant_id]
		evidence_ctrl_ids = {e.control_id for e in self._evidence.values() if e.tenant_id == tenant_id}
		assessed_ctrl_ids = {a.control_id for a in self._assessments.values() if a.tenant_id == tenant_id}
		checklist: list[dict[str, Any]] = []
		for domain in iso_domains:
			domain_controls = [c for c in controls if domain[:3].lower() in c.name.lower() or "iso" in (c.name.lower())]
			covered = len([c for c in domain_controls if c.id in evidence_ctrl_ids])
			assessed = len([c for c in domain_controls if c.id in assessed_ctrl_ids])
			checklist.append({
				"domain": domain,
				"controls_mapped": len(domain_controls),
				"evidence_collected": covered,
				"controls_assessed": assessed,
				"status": "compliant" if assessed > 0 else "gap",
			})
		compliant = sum(1 for item in checklist if item["status"] == "compliant")
		return {
			"tenant_id": tenant_id,
			"framework": "ISO 27001:2022",
			"generated_at": utc_now().isoformat(),
			"domains_total": len(iso_domains),
			"domains_compliant": compliant,
			"domains_with_gaps": len(iso_domains) - compliant,
			"readiness_pct": round(compliant / len(iso_domains) * 100, 1),
			"checklist": checklist,
		}

	# ------------------------------------------------------------------ 14
	async def gdpr_dpia(
		self,
		dpia_id: str,
		tenant_id: str,
		processing_activity: str,
		data_categories: list[str],
		risk_level: str,
		controller: str,
		mitigations: list[str] | None = None,
	) -> dict[str, Any]:
		"""Record a GDPR Data Protection Impact Assessment."""
		assert dpia_id and tenant_id and processing_activity and data_categories and controller
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, dpia_id)
		if key in self._dpia_records:
			raise ValueError(f"dpia_already_exists:{dpia_id}")
		record: dict[str, Any] = {
			"id": dpia_id,
			"tenant_id": tenant_id,
			"processing_activity": processing_activity,
			"data_categories": list(data_categories),
			"risk_level": risk_level,
			"controller": controller,
			"mitigations": list(mitigations or []),
			"status": "draft",
			"created_at": utc_now().isoformat(),
			"requires_supervisory_authority": risk_level in {"high", "critical"},
		}
		self._dpia_records[key] = record
		self._record_audit(tenant_id, "gdpr_dpia_created", dpia_id, controller, record)
		return record

	# ------------------------------------------------------------------ 15
	async def soc2_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		trust_service_criteria: str,
		evidence_type: str,
		description: str,
		collected_by: str,
		artifact_reference: str = "",
	) -> dict[str, Any]:
		"""Collect SOC 2 evidence mapped to Trust Service Criteria."""
		assert evidence_id and tenant_id and trust_service_criteria and evidence_type and collected_by
		VALID_TSC = {"CC1", "CC2", "CC3", "CC4", "CC5", "CC6", "CC7", "CC8", "CC9", "A1", "PI1", "C1", "P1"}
		if trust_service_criteria not in VALID_TSC:
			raise ValueError(f"invalid_tsc:{trust_service_criteria}. Valid: {sorted(VALID_TSC)}")
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, evidence_id)
		if key in self._soc2_evidence:
			raise ValueError(f"soc2_evidence_already_exists:{evidence_id}")
		record: dict[str, Any] = {
			"id": evidence_id,
			"tenant_id": tenant_id,
			"trust_service_criteria": trust_service_criteria,
			"evidence_type": evidence_type,
			"description": description,
			"collected_by": collected_by,
			"artifact_reference": artifact_reference,
			"collected_at": utc_now().isoformat(),
			"status": "collected",
		}
		self._soc2_evidence[key] = record
		self._record_audit(tenant_id, "soc2_evidence_collected", evidence_id, collected_by, record)
		return record

	# ------------------------------------------------------------------ 16
	async def health_check(self) -> dict[str, Any]:
		"""Return service health status and store cardinalities."""
		return {
			"status": "healthy",
			"checked_at": utc_now().isoformat(),
			"stores": {
				"frameworks": len(self._frameworks),
				"controls": len(self._controls),
				"evidence": len(self._evidence),
				"assessments": len(self._assessments),
				"findings": len(self._findings),
				"reports": len(self._reports),
				"attestations": len(self._attestations),
				"obligations": len(self._obligations),
				"policies": len(self._policies),
				"risk_scores": len(self._risk_scores),
				"audit_schedules": len(self._audit_schedules),
				"training_assignments": len(self._training_assignments),
				"regulatory_changes": len(self._regulatory_changes),
				"dpia_records": len(self._dpia_records),
				"soc2_evidence": len(self._soc2_evidence),
				"audit_events": len(self._audit_events),
			},
		}

	# ------------------------------------------------------------------ 17
	async def bulk_create_controls(
		self,
		tenant_id: str,
		framework_id: str,
		controls: list[dict[str, Any]],
		owner: str,
	) -> list[dict[str, Any]]:
		"""Create multiple controls in one call; skips duplicates."""
		assert tenant_id and framework_id and controls and owner
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		results: list[dict[str, Any]] = []
		for ctrl in controls:
			ctrl_id = ctrl.get("id", f"ctrl-{stable_digest(ctrl)[:8]}")
			if self._key(tenant_id, ctrl_id) in self._controls:
				continue
			results.append(self.create_control(
				control_id=ctrl_id,
				tenant_id=tenant_id,
				framework_id=framework_id,
				name=ctrl["name"],
				owner=ctrl.get("owner", owner),
				control_type=ctrl.get("control_type", "preventive"),
				regulated_data_scope=bool(ctrl.get("regulated_data_scope", False)),
				dlp_policy_linked=bool(ctrl.get("dlp_policy_linked", False)),
				testing_frequency_days=int(ctrl.get("testing_frequency_days", 90)),
			))
		self._record_audit(tenant_id, "bulk_controls_created", framework_id, owner, {"count": len(results)})
		return results

	# ------------------------------------------------------------------ 18
	async def bulk_close_findings(
		self,
		tenant_id: str,
		finding_ids: list[str],
		resolved_by: str,
		resolution: str,
	) -> list[dict[str, Any]]:
		"""Resolve multiple findings at once."""
		assert finding_ids and resolved_by and resolution
		results: list[dict[str, Any]] = []
		for fid in finding_ids:
			results.append(self.resolve_finding(fid, tenant_id, resolved_by, resolution))
		return results

	# ------------------------------------------------------------------ 19
	async def export_compliance_data(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export all compliance data for a tenant as JSON or CSV."""
		assert fmt in {"json", "csv"}, "fmt must be 'json' or 'csv'"
		data = {
			"tenant_id": tenant_id,
			"exported_at": utc_now().isoformat(),
			"frameworks": self.list_frameworks(tenant_id),
			"controls": self.list_controls(tenant_id),
			"evidence": self.list_evidence(tenant_id),
			"assessments": self.list_assessments(tenant_id),
			"findings": self.list_findings(tenant_id),
			"reports": self.list_reports(tenant_id),
		}
		if fmt == "json":
			return json.dumps(data, indent=2)
		buf = io.StringIO()
		for section, rows in data.items():
			if not isinstance(rows, list) or not rows:
				continue
			writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
			buf.write(f"# {section}\n")
			writer.writeheader()
			writer.writerows(rows)
			buf.write("\n")
		return buf.getvalue()

	# ------------------------------------------------------------------ 20
	async def export_framework_csv(self, tenant_id: str, framework_id: str) -> str:
		"""Export controls and findings for one framework as CSV."""
		self._require_tenant(tenant_id)
		self._require_framework(framework_id, tenant_id)
		controls = [c for c in self.list_controls(tenant_id) if c.get("framework_id") == framework_id]
		ctrl_ids = {c["id"] for c in controls}
		findings = [f for f in self.list_findings(tenant_id) if f.get("control_id") in ctrl_ids]
		buf = io.StringIO()
		if controls:
			writer = csv.DictWriter(buf, fieldnames=controls[0].keys())
			buf.write("# controls\n")
			writer.writeheader()
			writer.writerows(controls)
			buf.write("\n")
		if findings:
			writer = csv.DictWriter(buf, fieldnames=findings[0].keys())
			buf.write("# findings\n")
			writer.writeheader()
			writer.writerows(findings)
		return buf.getvalue()
