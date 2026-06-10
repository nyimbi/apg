"""Legal Compliance Management — async service layer."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import date, datetime
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

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.requirements: dict[str, dict[str, Any]] = {}
		self.calendar_entries: dict[str, dict[str, Any]] = {}
		self.evidence: dict[str, dict[str, Any]] = {}
		self.breaches: dict[str, dict[str, Any]] = {}
		self.assessments: dict[str, dict[str, Any]] = {}
		self.remediation_plans: dict[str, dict[str, Any]] = {}
		self.notifications: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}{uuid4().hex[:12]}"

	def _tenant(self, tenant_id: str | None = None) -> str:
		val = tenant_id or self.tenant_id
		guard_tenant_id(val)
		return val

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt-"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"details": details or {},
			"created_at": self._now(),
		})

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

	async def update_evidence(self, tenant_id: str, evidence_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ev = self.evidence.get(evidence_id)
		if not ev or ev["tenant_id"] != tenant:
			raise KeyError(f"evidence {evidence_id} not found")
		allowed = {"title", "description", "valid_until", "file_reference"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				ev[k] = v
		self._emit(tenant, "evidence_updated", evidence_id, updates)
		return deepcopy(ev)

	async def delete_evidence(self, tenant_id: str, evidence_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ev = self.evidence.get(evidence_id)
		if not ev or ev["tenant_id"] != tenant:
			raise KeyError(f"evidence {evidence_id} not found")
		ev["status"] = "archived"
		self._emit(tenant, "evidence_archived", evidence_id)
		return deepcopy(ev)

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
		self.breaches[breach["id"]] = breach
		# Auto-flag requirement as non-compliant
		r["status"] = "non_compliant"
		r["updated_at"] = self._now()
		self._emit(tenant, "breach_reported", breach["id"], {
			"requirement_id": requirement_id,
			"severity": severity,
			"notification_required": notification_required,
		})
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
