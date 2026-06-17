"""Legal Compliance Management — async service layer."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import csv
import hashlib
import io
import json
import logging
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

RISK_LEVELS = {"low", "medium", "high", "critical"}
REQUIREMENT_STATUSES = {"active", "compliant", "non_compliant", "exempted", "archived"}
BREACH_SEVERITIES = {"low", "medium", "high", "critical"}
BREACH_STATUSES = {"open", "investigating", "remediated", "reported", "closed"}
EVIDENCE_TYPES = {"document", "screenshot", "log", "certificate", "attestation", "audit_report", "policy"}
CATEGORIES = {
	"data_privacy", "financial", "employment", "environmental", "corporate",
	"health_safety", "anti_bribery", "aml", "sanctions", "consumer_protection",
}


class LegalComplianceService:
	"""In-memory async service for regulatory compliance management."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.requirements: dict[str, dict[str, Any]] = {}
		self.calendar_entries: dict[str, dict[str, Any]] = {}
		self.evidence: dict[str, dict[str, Any]] = {}
		self.breaches: dict[str, dict[str, Any]] = {}
		self.assessments: dict[str, dict[str, Any]] = {}
		self.remediation_plans: dict[str, dict[str, Any]] = {}
		self.notifications: dict[str, dict[str, Any]] = {}
		self.regulator_comms: dict[str, dict[str, Any]] = {}  # I11: regulator communication log
		self.cost_entries: dict[str, dict[str, Any]] = {}     # I12: compliance cost tracking
		self.attestations: dict[str, dict[str, Any]] = {}     # I15: attestation workflow
		self._score_snapshots = WriteThruDict('score_snapshots', tenant_id, _store) # I4: compliance trend history
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}{uuid4().hex[:12]}"

	def _tenant(self, tenant_id: str | None = None) -> str:
		val = tenant_id or self.tenant_id
		guard_tenant_id(val)
		return val

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, details: dict[str, Any] | None = None) -> None:
		# I13 (security): SHA-256 chain hash over previous event content — tamper-evident audit trail
		prev_hash = self._audit_events[-1].get("chain_hash", "") if self._audit_events else ""
		event: dict[str, Any] = {
			"id": self._id("evt-"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"details": details or {},
			"created_at": self._now(),
		}
		chain_input = prev_hash + json.dumps(event, sort_keys=True, default=str)
		event["chain_hash"] = hashlib.sha256(chain_input.encode()).hexdigest()
		self._audit_events.append(event)

	# ── Health & Describe ────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		today = date.today().isoformat()
		return {
			"service": "leg_cpl",
			"status": "healthy",
			"requirement_count": len(self.requirements),
			"open_breaches": sum(1 for b in self.breaches.values() if b["status"] in {"open", "investigating"}),
			"critical_requirements": sum(
				1 for r in self.requirements.values()
				if r["risk_level"] == "critical" and r["status"] != "compliant"
			),
			"overdue_calendar": sum(
				1 for c in self.calendar_entries.values()
				if c["status"] == "pending" and c["scheduled_date"] < today
			),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "leg_cpl",
			"name": "Legal Compliance Management",
			"domain": "legal",
			"version": "1.0.0",
			"risk_levels": sorted(RISK_LEVELS),
			"categories": sorted(CATEGORIES),
			"breach_severities": sorted(BREACH_SEVERITIES),
		}

	# ── Requirements ─────────────────────────────────────────────────────────

	async def create_requirement(
		self,
		tenant_id: str,
		title: str,
		description: str,
		regulation: str,
		jurisdiction: str,
		category: str,
		frequency: str,
		owner_id: str,
		due_date: str | None = None,
		risk_level: str = "medium",
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register a compliance requirement."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		guard_non_empty_string(regulation, "regulation")
		if risk_level not in RISK_LEVELS:
			raise ValueError(f"risk_level must be one of {RISK_LEVELS}")
		record: dict[str, Any] = {
			"id": self._id("cpl-"),
			"tenant_id": tenant,
			"title": title,
			"description": description,
			"regulation": regulation,
			"jurisdiction": jurisdiction,
			"category": category,
			"frequency": frequency,
			"due_date": due_date,
			"owner_id": owner_id,
			"risk_level": risk_level,
			"status": "active",
			"evidence_count": 0,
			"tags": list(tags or []),
			"metadata": dict(metadata or {}),
			"created_at": self._now(),
			"updated_at": None,
			"last_assessed_at": None,
		}
		self.requirements[record["id"]] = record
		self._emit(tenant, "requirement_created", record["id"], {"title": title, "regulation": regulation})
		_log.info("compliance requirement created tenant=%s id=%s reg=%s", tenant, record["id"], regulation)
		return deepcopy(record)

	async def get_requirement(self, tenant_id: str, requirement_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		return deepcopy(r)

	async def list_requirements(
		self,
		tenant_id: str,
		regulation: str | None = None,
		jurisdiction: str | None = None,
		category: str | None = None,
		status: str | None = None,
		risk_level: str | None = None,
		owner_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List compliance requirements with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.requirements.values() if r["tenant_id"] == tenant]
		if regulation:
			items = [r for r in items if r["regulation"] == regulation]
		if jurisdiction:
			items = [r for r in items if r["jurisdiction"] == jurisdiction]
		if category:
			items = [r for r in items if r["category"] == category]
		if status:
			items = [r for r in items if r["status"] == status]
		if risk_level:
			items = [r for r in items if r["risk_level"] == risk_level]
		if owner_id:
			items = [r for r in items if r["owner_id"] == owner_id]
		return items

	async def update_requirement(self, tenant_id: str, requirement_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		allowed = {"title", "description", "owner_id", "due_date", "risk_level", "tags", "metadata"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				r[k] = v
		r["updated_at"] = self._now()
		self._emit(tenant, "requirement_updated", requirement_id, updates)
		return deepcopy(r)

	async def mark_compliant(self, tenant_id: str, requirement_id: str, assessed_by: str) -> dict[str, Any]:
		"""Mark a requirement as compliant."""
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		r["status"] = "compliant"
		r["last_assessed_at"] = self._now()
		r["last_assessed_by"] = assessed_by
		r["updated_at"] = self._now()
		self._emit(tenant, "requirement_compliant", requirement_id, {"assessed_by": assessed_by})
		return deepcopy(r)

	async def flag_non_compliant(self, tenant_id: str, requirement_id: str, reason: str) -> dict[str, Any]:
		"""Flag a requirement as non-compliant."""
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		r["status"] = "non_compliant"
		r["non_compliance_reason"] = reason
		r["updated_at"] = self._now()
		self._emit(tenant, "requirement_non_compliant", requirement_id, {"reason": reason})
		return deepcopy(r)

	async def delete_requirement(self, tenant_id: str, requirement_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		r["status"] = "archived"
		r["updated_at"] = self._now()
		self._emit(tenant, "requirement_archived", requirement_id)
		return deepcopy(r)

	# ── Compliance Calendar ──────────────────────────────────────────────────

	async def create_calendar_entry(
		self,
		tenant_id: str,
		requirement_id: str,
		scheduled_date: str,
		title: str,
		assigned_to_id: str,
		description: str = "",
		reminder_days: list[int] | None = None,
	) -> dict[str, Any]:
		"""Add a calendar entry for a compliance activity."""
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		entry: dict[str, Any] = {
			"id": self._id("cal-"),
			"tenant_id": tenant,
			"requirement_id": requirement_id,
			"scheduled_date": scheduled_date,
			"title": title,
			"description": description,
			"assigned_to_id": assigned_to_id,
			"reminder_days": reminder_days or [14, 7, 1],
			"status": "pending",
			"completed_at": None,
			"created_at": self._now(),
		}
		self.calendar_entries[entry["id"]] = entry
		self._emit(tenant, "calendar_entry_created", entry["id"], {"requirement_id": requirement_id, "date": scheduled_date})
		return deepcopy(entry)

	async def get_calendar_entry(self, tenant_id: str, entry_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		e = self.calendar_entries.get(entry_id)
		if not e or e["tenant_id"] != tenant:
			raise KeyError(f"calendar entry {entry_id} not found")
		return deepcopy(e)

	async def list_calendar_entries(
		self,
		tenant_id: str,
		requirement_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.calendar_entries.values() if e["tenant_id"] == tenant]
		if requirement_id:
			items = [e for e in items if e["requirement_id"] == requirement_id]
		if status:
			items = [e for e in items if e["status"] == status]
		return sorted(items, key=lambda e: e["scheduled_date"])

	async def update_calendar_entry(self, tenant_id: str, entry_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		e = self.calendar_entries.get(entry_id)
		if not e or e["tenant_id"] != tenant:
			raise KeyError(f"calendar entry {entry_id} not found")
		allowed = {"scheduled_date", "title", "description", "assigned_to_id", "reminder_days"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				e[k] = v
		self._emit(tenant, "calendar_entry_updated", entry_id, updates)
		return deepcopy(e)

	async def complete_calendar_entry(self, tenant_id: str, entry_id: str, completed_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		e = self.calendar_entries.get(entry_id)
		if not e or e["tenant_id"] != tenant:
			raise KeyError(f"calendar entry {entry_id} not found")
		e["status"] = "completed"
		e["completed_at"] = self._now()
		e["completed_by"] = completed_by
		self._emit(tenant, "calendar_entry_completed", entry_id)
		return deepcopy(e)

	async def delete_calendar_entry(self, tenant_id: str, entry_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		e = self.calendar_entries.get(entry_id)
		if not e or e["tenant_id"] != tenant:
			raise KeyError(f"calendar entry {entry_id} not found")
		e["status"] = "cancelled"
		self._emit(tenant, "calendar_entry_cancelled", entry_id)
		return deepcopy(e)

	# ── Evidence ─────────────────────────────────────────────────────────────

	async def create_evidence(
		self,
		tenant_id: str,
		requirement_id: str,
		evidence_type: str,
		title: str,
		description: str,
		collected_by_id: str,
		collection_date: str,
		file_reference: str = "",
		valid_until: str | None = None,
	) -> dict[str, Any]:
		"""Attach evidence to a compliance requirement."""
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		if evidence_type not in EVIDENCE_TYPES:
			raise ValueError(f"evidence_type must be one of {EVIDENCE_TYPES}")
		ev: dict[str, Any] = {
			"id": self._id("ev-"),
			"tenant_id": tenant,
			"requirement_id": requirement_id,
			"evidence_type": evidence_type,
			"title": title,
			"description": description,
			"file_reference": file_reference,
			"collected_by_id": collected_by_id,
			"collection_date": collection_date,
			"valid_until": valid_until,
			"status": "active",
			"custody_chain": [  # I5: chain-of-custody for regulatory enforceability
				{"actor_id": collected_by_id, "action": "created", "timestamp": self._now(), "field_delta": {}}
			],
			"created_at": self._now(),
		}
		self.evidence[ev["id"]] = ev
		r["evidence_count"] = r.get("evidence_count", 0) + 1
		self._emit(tenant, "evidence_created", ev["id"], {"requirement_id": requirement_id, "type": evidence_type})
		return deepcopy(ev)

	async def get_evidence(self, tenant_id: str, evidence_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ev = self.evidence.get(evidence_id)
		if not ev or ev["tenant_id"] != tenant:
			raise KeyError(f"evidence {evidence_id} not found")
		return deepcopy(ev)

	async def list_evidence(self, tenant_id: str, requirement_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(e) for e in self.evidence.values()
			if e["tenant_id"] == tenant and e["requirement_id"] == requirement_id
		]

	async def update_evidence(self, tenant_id: str, evidence_id: str, actor_id: str = "system", **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ev = self.evidence.get(evidence_id)
		if not ev or ev["tenant_id"] != tenant:
			raise KeyError(f"evidence {evidence_id} not found")
		allowed = {"title", "description", "valid_until", "file_reference"}
		delta: dict[str, Any] = {}
		for k, v in updates.items():
			if k in allowed and v is not None:
				delta[k] = {"old": ev.get(k), "new": v}
				ev[k] = v
		# I5: append mutation to chain-of-custody so every evidence change is traceable
		ev.setdefault("custody_chain", []).append(
			{"actor_id": actor_id, "action": "updated", "timestamp": self._now(), "field_delta": delta}
		)
		self._emit(tenant, "evidence_updated", evidence_id, updates)
		return deepcopy(ev)

	async def delete_evidence(self, tenant_id: str, evidence_id: str, actor_id: str = "system") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ev = self.evidence.get(evidence_id)
		if not ev or ev["tenant_id"] != tenant:
			raise KeyError(f"evidence {evidence_id} not found")
		ev["status"] = "archived"
		# I5: record archival in chain-of-custody
		ev.setdefault("custody_chain", []).append(
			{"actor_id": actor_id, "action": "archived", "timestamp": self._now(), "field_delta": {}}
		)
		self._emit(tenant, "evidence_archived", evidence_id)
		return deepcopy(ev)

	async def get_evidence_chain(self, tenant_id: str, evidence_id: str) -> list[dict[str, Any]]:
		"""I5: Return the immutable chain-of-custody log for an evidence item.

		Enables legal teams to demonstrate evidence integrity in enforcement proceedings.
		"""
		tenant = self._tenant(tenant_id)
		ev = self.evidence.get(evidence_id)
		if not ev or ev["tenant_id"] != tenant:
			raise KeyError(f"evidence {evidence_id} not found")
		return deepcopy(ev.get("custody_chain", []))

	# ── Breach Reporting ─────────────────────────────────────────────────────

	async def create_breach(
		self,
		tenant_id: str,
		requirement_id: str,
		title: str,
		description: str,
		severity: str,
		discovered_by_id: str,
		discovery_date: str,
		affected_records: int = 0,
		notification_required: bool = False,
		notification_deadline: str | None = None,
	) -> dict[str, Any]:
		"""Report a compliance breach."""
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		if severity not in BREACH_SEVERITIES:
			raise ValueError(f"severity must be one of {BREACH_SEVERITIES}")
		guard_non_empty_string(title, "title")
		breach: dict[str, Any] = {
			"id": self._id("brch-"),
			"tenant_id": tenant,
			"requirement_id": requirement_id,
			"title": title,
			"description": description,
			"severity": severity,
			"discovered_by_id": discovered_by_id,
			"discovery_date": discovery_date,
			"affected_records": affected_records,
			"notification_required": notification_required,
			"notification_deadline": notification_deadline,
			"status": "open",
			"remediated_at": None,
			"reported_at": None,
			"created_at": self._now(),
		}
		# I8: compute 72-hour regulatory notification SLA at creation time (GDPR Art.33 / Kenya DPA s.43)
		if notification_required:
			try:
				disc_dt = datetime.fromisoformat(discovery_date)
			except ValueError:
				disc_dt = datetime.utcnow()
			breach["notification_sla_expires_at"] = (disc_dt + timedelta(hours=72)).isoformat(timespec="seconds") + "Z"
		else:
			breach["notification_sla_expires_at"] = None

		self.breaches[breach["id"]] = breach
		# Auto-flag requirement as non-compliant
		r["status"] = "non_compliant"
		r["updated_at"] = self._now()
		self._emit(tenant, "breach_reported", breach["id"], {
			"requirement_id": requirement_id,
			"severity": severity,
			"notification_required": notification_required,
		})

		# I7: auto-generate structured remediation plan (collapses 5-day manual effort to minutes)
		sla_hours = {"critical": 4, "high": 24, "medium": 72, "low": 168}
		base_hours = sla_hours.get(severity, 72)
		plan: dict[str, Any] = {
			"id": self._id("rpl-"),
			"tenant_id": tenant,
			"breach_id": breach["id"],
			"requirement_id": requirement_id,
			"status": "active",
			"auto_generated": True,
			"milestones": [
				{"step": 1, "title": "Notify DPO / Compliance Officer", "sla_hours": 1, "status": "pending"},
				{"step": 2, "title": "Assess scope and affected records", "sla_hours": base_hours // 4 or 1, "status": "pending"},
				{"step": 3, "title": "Contain breach and prevent further exposure", "sla_hours": base_hours // 2 or 2, "status": "pending"},
				{"step": 4, "title": "Notify regulator if required", "sla_hours": 72, "status": "pending" if notification_required else "n/a"},
				{"step": 5, "title": "Implement corrective controls", "sla_hours": base_hours, "status": "pending"},
				{"step": 6, "title": "Post-incident review and lessons learned", "sla_hours": base_hours * 2, "status": "pending"},
			],
			"created_at": self._now(),
		}
		self.remediation_plans[plan["id"]] = plan
		breach["remediation_plan_id"] = plan["id"]

		_log.warning("compliance breach reported tenant=%s id=%s severity=%s", tenant, breach["id"], severity)
		return deepcopy(breach)

	async def get_breach(self, tenant_id: str, breach_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		b = self.breaches.get(breach_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"breach {breach_id} not found")
		return deepcopy(b)

	async def list_breaches(
		self,
		tenant_id: str,
		requirement_id: str | None = None,
		severity: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(b) for b in self.breaches.values() if b["tenant_id"] == tenant]
		if requirement_id:
			items = [b for b in items if b["requirement_id"] == requirement_id]
		if severity:
			items = [b for b in items if b["severity"] == severity]
		if status:
			items = [b for b in items if b["status"] == status]
		return items

	async def update_breach(self, tenant_id: str, breach_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		b = self.breaches.get(breach_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"breach {breach_id} not found")
		allowed = {"title", "description", "affected_records", "notification_deadline"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				b[k] = v
		self._emit(tenant, "breach_updated", breach_id, updates)
		return deepcopy(b)

	async def investigate_breach(self, tenant_id: str, breach_id: str, assigned_to: str) -> dict[str, Any]:
		"""Open an investigation on a breach."""
		tenant = self._tenant(tenant_id)
		b = self.breaches.get(breach_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"breach {breach_id} not found")
		b["status"] = "investigating"
		b["investigating_assigned_to"] = assigned_to
		b["investigation_started_at"] = self._now()
		self._emit(tenant, "breach_investigation_started", breach_id)
		return deepcopy(b)

	async def remediate_breach(self, tenant_id: str, breach_id: str, remediation_notes: str) -> dict[str, Any]:
		"""Mark a breach as remediated."""
		tenant = self._tenant(tenant_id)
		b = self.breaches.get(breach_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"breach {breach_id} not found")
		b["status"] = "remediated"
		b["remediation_notes"] = remediation_notes
		b["remediated_at"] = self._now()
		self._emit(tenant, "breach_remediated", breach_id)
		return deepcopy(b)

	async def report_breach_to_regulator(
		self,
		tenant_id: str,
		breach_id: str,
		regulator: str,
		reference_number: str,
		reported_by: str,
	) -> dict[str, Any]:
		"""Record regulatory breach notification."""
		tenant = self._tenant(tenant_id)
		b = self.breaches.get(breach_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"breach {breach_id} not found")
		b["status"] = "reported"
		b["regulator"] = regulator
		b["regulatory_reference"] = reference_number
		b["reported_by"] = reported_by
		b["reported_at"] = self._now()
		self._emit(tenant, "breach_reported_to_regulator", breach_id, {"regulator": regulator})
		return deepcopy(b)

	async def close_breach(self, tenant_id: str, breach_id: str, closure_notes: str) -> dict[str, Any]:
		"""Close a breach."""
		tenant = self._tenant(tenant_id)
		b = self.breaches.get(breach_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"breach {breach_id} not found")
		b["status"] = "closed"
		b["closure_notes"] = closure_notes
		b["closed_at"] = self._now()
		self._emit(tenant, "breach_closed", breach_id)
		return deepcopy(b)

	async def delete_breach(self, tenant_id: str, breach_id: str) -> dict[str, Any]:
		return await self.close_breach(tenant_id, breach_id, closure_notes="archived")

	# ── Analytics ────────────────────────────────────────────────────────────

	async def compliance_dashboard(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		reqs = [r for r in self.requirements.values() if r["tenant_id"] == tenant]
		breaches = [b for b in self.breaches.values() if b["tenant_id"] == tenant]
		today = date.today().isoformat()
		by_status: dict[str, int] = {}
		by_risk: dict[str, int] = {}
		by_category: dict[str, int] = {}
		for r in reqs:
			by_status[r["status"]] = by_status.get(r["status"], 0) + 1
			by_risk[r["risk_level"]] = by_risk.get(r["risk_level"], 0) + 1
			by_category[r["category"]] = by_category.get(r["category"], 0) + 1
		compliance_rate = (
			round(100 * by_status.get("compliant", 0) / len(reqs), 1) if reqs else 0.0
		)
		return {
			"tenant_id": tenant,
			"total_requirements": len(reqs),
			"compliance_rate_pct": compliance_rate,
			"by_status": by_status,
			"by_risk_level": by_risk,
			"by_category": by_category,
			"open_breaches": sum(1 for b in breaches if b["status"] in {"open", "investigating"}),
			"critical_breaches": sum(1 for b in breaches if b["severity"] == "critical" and b["status"] != "closed"),
			"overdue_calendar": sum(
				1 for c in self.calendar_entries.values()
				if c["tenant_id"] == tenant and c["status"] == "pending" and c["scheduled_date"] < today
			),
			"generated_at": self._now(),
		}

	async def risk_register(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all non-compliant and high-risk requirements sorted by risk."""
		tenant = self._tenant(tenant_id)
		risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
		items = [
			deepcopy(r) for r in self.requirements.values()
			if r["tenant_id"] == tenant and r["status"] in {"non_compliant", "active"}
		]
		return sorted(items, key=lambda r: risk_order.get(r["risk_level"], 99))

	async def get_audit_events(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		return events[-limit:]

	async def verify_audit_chain(self, tenant_id: str) -> dict[str, Any]:
		"""Verify the SHA-256 chain-hash integrity of the audit trail.

		Detects any tampering by re-computing each event's expected hash.
		Returns {valid: bool, checked: int, first_broken_index: int | None}.
		"""
		tenant = self._tenant(tenant_id)
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		prev_hash = ""
		for i, evt in enumerate(events):
			evt_body = {k: v for k, v in evt.items() if k != "chain_hash"}
			chain_input = prev_hash + json.dumps(evt_body, sort_keys=True, default=str)
			expected = hashlib.sha256(chain_input.encode()).hexdigest()
			if evt.get("chain_hash") != expected:
				return {"valid": False, "checked": i + 1, "first_broken_index": i}
			prev_hash = evt["chain_hash"]
		return {"valid": True, "checked": len(events), "first_broken_index": None}

	# ── I3: Regulatory Penalty Exposure Calculator ───────────────────────────

	async def calculate_penalty_exposure(
		self,
		tenant_id: str,
		annual_turnover: Decimal,
		currency: str = "USD",
	) -> dict[str, Any]:
		"""I3: Convert non-compliant requirements into board-ready financial risk figures.

		Maps each non-compliant requirement's regulation to a penalty schedule and returns
		per-requirement and aggregate maximum exposure. Turns a risk conversation from
		qualitative to quantitative — replicates LogicGate's premium calculator feature.
		"""
		tenant = self._tenant(tenant_id)
		if annual_turnover <= Decimal("0"):
			raise ValueError("annual_turnover must be positive")

		# Simplified penalty schedule: {regulation_prefix: (pct_of_turnover, fixed_cap_usd)}
		_PENALTY_SCHEDULE: dict[str, tuple[Decimal, Decimal]] = {
			"GDPR":    (Decimal("0.04"), Decimal("20_000_000")),
			"DPA":     (Decimal("0.04"), Decimal("5_000_000")),
			"PCIDSS":  (Decimal("0.00"), Decimal("500_000")),
			"HIPAA":   (Decimal("0.00"), Decimal("1_900_000")),
			"AML":     (Decimal("0.10"), Decimal("10_000_000")),
			"DEFAULT": (Decimal("0.02"), Decimal("1_000_000")),
		}

		non_compliant = [
			r for r in self.requirements.values()
			if r["tenant_id"] == tenant and r["status"] == "non_compliant"
		]

		line_items: list[dict[str, Any]] = []
		total_max = Decimal("0")

		for r in non_compliant:
			reg_key = r["regulation"].upper().split()[0] if r.get("regulation") else "DEFAULT"
			pct, cap = _PENALTY_SCHEDULE.get(reg_key, _PENALTY_SCHEDULE["DEFAULT"])
			pct_amount = (annual_turnover * pct).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			max_exposure = min(pct_amount, cap) if pct_amount > Decimal("0") else cap
			# Likely exposure: 30% of max for active, 80% for confirmed non-compliant
			likely_multiplier = Decimal("0.80") if r["status"] == "non_compliant" else Decimal("0.30")
			likely_exposure = (max_exposure * likely_multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			total_max += max_exposure
			line_items.append({
				"requirement_id": r["id"],
				"title": r["title"],
				"regulation": r["regulation"],
				"risk_level": r["risk_level"],
				"max_exposure": str(max_exposure),
				"likely_exposure": str(likely_exposure),
				"currency": currency,
			})

		return {
			"tenant_id": tenant,
			"currency": currency,
			"annual_turnover": str(annual_turnover),
			"non_compliant_count": len(non_compliant),
			"aggregate_max_exposure": str(total_max),
			"line_items": sorted(line_items, key=lambda x: Decimal(x["max_exposure"]), reverse=True),
			"calculated_at": self._now(),
		}

	# ── I4: Compliance Score Trend History ───────────────────────────────────

	async def record_compliance_snapshot(self, tenant_id: str) -> dict[str, Any]:
		"""I4: Persist today's compliance score for trend reporting.

		Call daily (e.g., via a cron job) to build the historical dataset boards demand.
		Drata and Vanta use weekly trend charts as their primary retention hook.
		"""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		reqs = [r for r in self.requirements.values() if r["tenant_id"] == tenant]
		compliant = sum(1 for r in reqs if r["status"] == "compliant")
		rate = round(100 * compliant / len(reqs), 1) if reqs else 0.0
		open_breaches = sum(1 for b in self.breaches.values() if b["tenant_id"] == tenant and b["status"] in {"open", "investigating"})
		critical = sum(1 for r in reqs if r["risk_level"] == "critical" and r["status"] != "compliant")
		snap = {
			"date": today,
			"tenant_id": tenant,
			"compliance_rate_pct": rate,
			"compliant_count": compliant,
			"total_requirements": len(reqs),
			"open_breaches": open_breaches,
			"critical_non_compliant": critical,
			"recorded_at": self._now(),
		}
		self._score_snapshots[f"{tenant}:{today}"] = snap
		return deepcopy(snap)

	async def get_compliance_trend(self, tenant_id: str, days: int = 90) -> dict[str, Any]:
		"""I4: Return time-series compliance score data for board trend charts.

		Returns snapshots within the requested window with delta and direction indicators.
		"""
		tenant = self._tenant(tenant_id)
		if days < 1:
			raise ValueError("days must be >= 1")
		cutoff = (date.today() - timedelta(days=days)).isoformat()
		snaps = sorted(
			[s for k, s in self._score_snapshots.items() if k.startswith(f"{tenant}:") and s["date"] >= cutoff],
			key=lambda s: s["date"],
		)
		# Compute delta vs previous snapshot
		for i, s in enumerate(snaps):
			if i > 0:
				s["delta_pct"] = round(s["compliance_rate_pct"] - snaps[i - 1]["compliance_rate_pct"], 1)
				s["direction"] = "up" if s["delta_pct"] > 0 else ("down" if s["delta_pct"] < 0 else "flat")
			else:
				s["delta_pct"] = None
				s["direction"] = "baseline"
		return {
			"tenant_id": tenant,
			"days": days,
			"snapshot_count": len(snaps),
			"snapshots": snaps,
			"latest_rate_pct": snaps[-1]["compliance_rate_pct"] if snaps else None,
		}

	# ── I8: Breach Notification SLA Countdown ────────────────────────────────

	async def get_breach_sla_status(self, tenant_id: str, breach_id: str) -> dict[str, Any]:
		"""I8: Real-time countdown to GDPR/DPA 72-hour regulatory notification deadline.

		Missing the notification SLA triggers fines that dwarf the original breach penalty.
		Returns hours_remaining, is_overdue, and sla_status: green | amber | red.
		"""
		tenant = self._tenant(tenant_id)
		b = self.breaches.get(breach_id)
		if not b or b["tenant_id"] != tenant:
			raise KeyError(f"breach {breach_id} not found")
		if not b.get("notification_sla_expires_at"):
			return {
				"breach_id": breach_id,
				"notification_required": False,
				"sla_status": "n/a",
				"hours_remaining": None,
				"is_overdue": False,
			}
		expires_at = datetime.fromisoformat(b["notification_sla_expires_at"].rstrip("Z"))
		now = datetime.utcnow()
		diff_hours = (expires_at - now).total_seconds() / 3600
		hours_remaining = round(diff_hours, 1)
		is_overdue = hours_remaining < 0
		if is_overdue:
			sla_status = "red"
		elif hours_remaining <= 24:
			sla_status = "amber"
		else:
			sla_status = "green"
		return {
			"breach_id": breach_id,
			"notification_required": True,
			"sla_expires_at": b["notification_sla_expires_at"],
			"hours_remaining": hours_remaining,
			"is_overdue": is_overdue,
			"sla_status": sla_status,
			"notification_filed": b.get("reported_at") is not None,
			"checked_at": self._now(),
		}

	# ── I10: Evidence Expiry and Gap Analysis ────────────────────────────────

	async def get_evidence_gap_report(self, tenant_id: str) -> dict[str, Any]:
		"""I10: Continuous audit-readiness scan — finds requirements with missing or expiring evidence.

		Drata and Qualys use this pattern to guarantee audit-readiness 365 days a year
		rather than scrambling on audit day.
		"""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		horizon = (date.today() + timedelta(days=30)).isoformat()

		reqs = [r for r in self.requirements.values() if r["tenant_id"] == tenant and r["status"] != "archived"]
		evidence_by_req: dict[str, list[dict[str, Any]]] = {}
		for ev in self.evidence.values():
			if ev["tenant_id"] == tenant:
				evidence_by_req.setdefault(ev["requirement_id"], []).append(ev)

		items: list[dict[str, Any]] = []
		total_gaps = 0
		for r in reqs:
			evs = evidence_by_req.get(r["id"], [])
			active_evs = [e for e in evs if e["status"] == "active"]
			valid_evs = [e for e in active_evs if not e.get("valid_until") or e["valid_until"] >= today]
			expired_evs = [e for e in active_evs if e.get("valid_until") and e["valid_until"] < today]
			expiring_soon = [e["id"] for e in valid_evs if e.get("valid_until") and e["valid_until"] <= horizon]
			has_valid = len(valid_evs) > 0
			if not has_valid:
				total_gaps += 1
			items.append({
				"requirement_id": r["id"],
				"title": r["title"],
				"risk_level": r["risk_level"],
				"has_valid_evidence": has_valid,
				"valid_evidence_count": len(valid_evs),
				"expired_count": len(expired_evs),
				"expiring_in_30d": expiring_soon,
				"gap": not has_valid,
			})

		items.sort(key=lambda x: (not x["gap"], x["risk_level"] == "low", x["title"]))
		return {
			"tenant_id": tenant,
			"total_requirements": len(reqs),
			"requirements_with_gaps": total_gaps,
			"gap_rate_pct": round(100 * total_gaps / len(reqs), 1) if reqs else 0.0,
			"items": items,
			"generated_at": self._now(),
		}

	# ── I11: Regulator Communication Log ─────────────────────────────────────

	async def log_regulator_communication(
		self,
		tenant_id: str,
		entity_id: str,
		regulator: str,
		direction: str,
		summary: str,
		medium: str = "email",
		reference: str = "",
		actor_id: str = "",
	) -> dict[str, Any]:
		"""I11: Log correspondence with a regulator for litigation-ready audit records.

		Every FCA, DPA, or CBK communication must be preserved with metadata.
		Aderant and iManage are built around this concept; ad-hoc email folders don't
		survive staff turnover and fail discovery requests.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(regulator, "regulator")
		guard_non_empty_string(summary, "summary")
		if direction not in {"inbound", "outbound"}:
			raise ValueError("direction must be 'inbound' or 'outbound'")
		comm: dict[str, Any] = {
			"id": self._id("comm-"),
			"tenant_id": tenant,
			"entity_id": entity_id,
			"regulator": regulator,
			"direction": direction,
			"summary": summary,
			"medium": medium,
			"reference": reference,
			"actor_id": actor_id,
			"logged_at": self._now(),
		}
		self.regulator_comms[comm["id"]] = comm
		self._emit(tenant, "regulator_comm_logged", comm["id"], {"regulator": regulator, "direction": direction})
		_log.info("regulator comm logged tenant=%s regulator=%s direction=%s", tenant, regulator, direction)
		return deepcopy(comm)

	async def list_regulator_comms(
		self,
		tenant_id: str,
		entity_id: str | None = None,
		regulator: str | None = None,
	) -> list[dict[str, Any]]:
		"""I11: Return chronological regulator communication log for an entity or across all entities."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.regulator_comms.values() if c["tenant_id"] == tenant]
		if entity_id:
			items = [c for c in items if c["entity_id"] == entity_id]
		if regulator:
			items = [c for c in items if c["regulator"] == regulator]
		return sorted(items, key=lambda c: c["logged_at"])

	# ── I12: Compliance Cost Tracking ────────────────────────────────────────

	async def log_compliance_cost(
		self,
		tenant_id: str,
		requirement_id: str,
		amount: Decimal,
		currency: str,
		cost_type: str,
		period: str,
		recorded_by: str,
	) -> dict[str, Any]:
		"""I12: Track auditor fees, tool licenses, and staff time per requirement.

		CFOs demand ROI on compliance spend; 80% of companies track this in spreadsheets.
		MetricStream and Riskonnect embed cost modules — this brings that to APG at no
		additional per-seat cost.
		"""
		tenant = self._tenant(tenant_id)
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		if amount <= Decimal("0"):
			raise ValueError("amount must be positive")
		guard_non_empty_string(currency, "currency")
		entry: dict[str, Any] = {
			"id": self._id("cost-"),
			"tenant_id": tenant,
			"requirement_id": requirement_id,
			"regulation": r["regulation"],
			"category": r["category"],
			"amount": str(amount),
			"currency": currency,
			"cost_type": cost_type,
			"period": period,
			"recorded_by": recorded_by,
			"recorded_at": self._now(),
		}
		self.cost_entries[entry["id"]] = entry
		self._emit(tenant, "compliance_cost_logged", entry["id"], {"amount": str(amount), "currency": currency})
		return deepcopy(entry)

	async def get_compliance_cost_summary(self, tenant_id: str, currency: str | None = None) -> dict[str, Any]:
		"""I12: Return per-regulation and per-category compliance cost totals as Decimal.

		Provides legal ops with a clear budget defence and identifies highest-cost regulations.
		"""
		tenant = self._tenant(tenant_id)
		entries = [e for e in self.cost_entries.values() if e["tenant_id"] == tenant]
		if currency:
			entries = [e for e in entries if e["currency"] == currency]

		by_regulation: dict[str, Decimal] = {}
		by_category: dict[str, Decimal] = {}
		total = Decimal("0")

		for e in entries:
			amt = Decimal(e["amount"])
			reg = e.get("regulation", "unknown")
			cat = e.get("category", "unknown")
			by_regulation[reg] = by_regulation.get(reg, Decimal("0")) + amt
			by_category[cat] = by_category.get(cat, Decimal("0")) + amt
			total += amt

		return {
			"tenant_id": tenant,
			"currency_filter": currency,
			"total": str(total),
			"by_regulation": {k: str(v) for k, v in sorted(by_regulation.items(), key=lambda x: x[1], reverse=True)},
			"by_category": {k: str(v) for k, v in sorted(by_category.items(), key=lambda x: x[1], reverse=True)},
			"entry_count": len(entries),
			"generated_at": self._now(),
		}

	# ── I13: Owner Workload Balancing ─────────────────────────────────────────

	async def get_owner_workload(self, tenant_id: str) -> dict[str, Any]:
		"""I13: Surface per-owner compliance workload to prevent single points of failure.

		Without workload visibility, overburdened owners miss deadlines silently.
		Navex Global and SAI360 both surface per-owner workload — this closes the
		people-management gap in compliance operations.
		"""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		reqs = [r for r in self.requirements.values() if r["tenant_id"] == tenant]
		breaches = [b for b in self.breaches.values() if b["tenant_id"] == tenant]
		calendars = [c for c in self.calendar_entries.values() if c["tenant_id"] == tenant]

		owners: dict[str, dict[str, Any]] = {}

		for r in reqs:
			owner = r.get("owner_id", "unassigned")
			if owner not in owners:
				owners[owner] = {
					"owner_id": owner,
					"active_requirements": 0,
					"non_compliant": 0,
					"overdue_calendar": 0,
					"open_breaches": 0,
					"compliance_rate_pct": 0.0,
					"_compliant": 0,
					"_total": 0,
				}
			o = owners[owner]
			o["_total"] += 1
			if r["status"] == "active":
				o["active_requirements"] += 1
			if r["status"] == "non_compliant":
				o["non_compliant"] += 1
			if r["status"] == "compliant":
				o["_compliant"] += 1

		for c in calendars:
			r = self.requirements.get(c.get("requirement_id", ""))
			owner = r.get("owner_id", "unassigned") if r else c.get("assigned_to_id", "unassigned")
			if owner in owners and c["status"] == "pending" and c["scheduled_date"] < today:
				owners[owner]["overdue_calendar"] += 1

		# Associate breaches with requirement owners
		for b in breaches:
			if b["status"] not in {"open", "investigating"}:
				continue
			r = self.requirements.get(b.get("requirement_id", ""))
			owner = r.get("owner_id", "unassigned") if r else "unassigned"
			if owner in owners:
				owners[owner]["open_breaches"] += 1

		# Compute compliance rate per owner
		for o in owners.values():
			if o["_total"] > 0:
				o["compliance_rate_pct"] = round(100 * o["_compliant"] / o["_total"], 1)
			del o["_compliant"]
			del o["_total"]

		return {
			"tenant_id": tenant,
			"owner_count": len(owners),
			"workloads": sorted(owners.values(), key=lambda o: o["non_compliant"], reverse=True),
			"generated_at": self._now(),
		}

	async def reassign_requirement(
		self,
		tenant_id: str,
		requirement_id: str,
		new_owner_id: str,
		reason: str = "",
		reassign_calendar: bool = True,
	) -> dict[str, Any]:
		"""I13: Transfer requirement ownership with full audit trail.

		Ensures accountability during staff transitions and prevents compliance gaps
		from orphaned requirements.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(new_owner_id, "new_owner_id")
		r = self.requirements.get(requirement_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"requirement {requirement_id} not found")
		old_owner = r.get("owner_id")
		r["owner_id"] = new_owner_id
		r["updated_at"] = self._now()
		if reassign_calendar:
			for c in self.calendar_entries.values():
				if c["tenant_id"] == tenant and c["requirement_id"] == requirement_id and c["status"] == "pending":
					c["assigned_to_id"] = new_owner_id
		self._emit(tenant, "requirement_reassigned", requirement_id, {
			"old_owner": old_owner,
			"new_owner": new_owner_id,
			"reason": reason,
		})
		_log.info("requirement reassigned tenant=%s id=%s from=%s to=%s", tenant, requirement_id, old_owner, new_owner_id)
		return deepcopy(r)

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_score_snapshots', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

