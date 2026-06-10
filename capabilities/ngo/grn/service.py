"""Grant Management Service — grant pipeline, proposals, budgets, disbursements, compliance."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "ngo_grn"

SUPPORTED_GRANT_STATUSES = {"pipeline", "proposal", "active", "suspended", "closed", "cancelled"}
SUPPORTED_REPORT_TYPES = {"narrative", "financial", "audit", "annual", "mid_term", "final"}
SUPPORTED_SEVERITY = {"low", "medium", "high", "critical"}
SUPPORTED_PAYMENT_METHODS = {"bank_transfer", "cheque", "mpesa", "swift", "eft"}


class GrantManagementService:
	"""Async service for NGO grant lifecycle management."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._grants: dict[str, dict[str, Any]] = {}
		self._proposals: dict[str, dict[str, Any]] = {}
		self._budget_lines: dict[str, dict[str, Any]] = {}
		self._disbursements: dict[str, dict[str, Any]] = {}
		self._compliance_reports: dict[str, dict[str, Any]] = {}
		self._audit_findings: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	# ── helpers ──────────────────────────────────────────────────────────────

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self) -> str:
		if not self.tenant_id:
			raise PermissionError("tenant_context_required")
		return self.tenant_id

	def _emit(self, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt"),
			"tenant_id": self._tenant(),
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"emitted_at": self._now(),
		})

	def _guard_grant(self, grant_id: str) -> dict[str, Any]:
		tenant = self._tenant()
		grant = self._grants.get(grant_id)
		if not grant or grant["tenant_id"] != tenant:
			raise KeyError(f"grant_not_found:{grant_id}")
		return grant

	# ── health / describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"grant_count": len(self._grants),
			"active_grants": sum(1 for g in self._grants.values() if g["status"] == "active"),
			"pending_disbursements": sum(1 for d in self._disbursements.values() if d["status"] == "pending"),
			"open_findings": sum(1 for f in self._audit_findings.values() if f["status"] == "open"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability descriptor."""
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "ngo",
			"version": "1.0.0",
			"description": "Grant pipeline, proposal management, budget tracking, disbursement, compliance reporting, audits",
			"supported_statuses": list(SUPPORTED_GRANT_STATUSES),
			"supported_report_types": list(SUPPORTED_REPORT_TYPES),
			"tenant_id": self.tenant_id,
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		"""Return recent audit events for the tenant."""
		tenant = self._tenant()
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		return [deepcopy(e) for e in events[-limit:]]

	# ── grants ────────────────────────────────────────────────────────────────

	async def list_grants(self, status: str | None = None, sector: str | None = None) -> list[dict[str, Any]]:
		"""List all grants for the tenant, optionally filtered."""
		tenant = self._tenant()
		items = [deepcopy(g) for g in self._grants.values() if g["tenant_id"] == tenant]
		if status:
			items = [g for g in items if g["status"] == status]
		if sector:
			items = [g for g in items if g.get("sector") == sector]
		return items

	async def get_grant(self, grant_id: str) -> dict[str, Any]:
		"""Retrieve a single grant by ID."""
		return deepcopy(self._guard_grant(grant_id))

	async def create_grant(
		self,
		title: str,
		donor_reference: str,
		amount: Decimal,
		start_date: str,
		end_date: str,
		currency: str = "KES",
		sector: str = "",
		country: str = "KE",
		programme_id: str | None = None,
		contact_person: str = "",
		notes: str = "",
	) -> dict[str, Any]:
		"""Create a new grant record in pipeline status."""
		tenant = self._tenant()
		if not title:
			raise ValueError("title_required")
		if not donor_reference:
			raise ValueError("donor_reference_required")
		if amount <= 0:
			raise ValueError("amount_must_be_positive")
		record: dict[str, Any] = {
			"id": self._id("grn"),
			"type": "ngo_grant",
			"tenant_id": tenant,
			"title": title,
			"donor_reference": donor_reference,
			"currency": currency,
			"amount": amount,
			"disbursed_amount": Decimal("0"),
			"start_date": start_date,
			"end_date": end_date,
			"sector": sector,
			"country": country,
			"programme_id": programme_id,
			"contact_person": contact_person,
			"notes": notes,
			"status": "pipeline",
			"created_at": self._now(),
			"updated_at": None,
		}
		self._grants[record["id"]] = record
		self._emit("grant_created", record["id"], "ngo_grant", {"title": title, "amount": str(amount)})
		_log.info("Grant created: %s (%s)", record["id"], title)
		return deepcopy(record)

	async def update_grant(self, grant_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update mutable fields on a grant."""
		grant = self._guard_grant(grant_id)
		allowed = {"title", "amount", "end_date", "status", "contact_person", "notes", "programme_id"}
		if "status" in kwargs and kwargs["status"] not in SUPPORTED_GRANT_STATUSES:
			raise ValueError(f"invalid_status:{kwargs['status']}")
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				grant[k] = v
		grant["updated_at"] = self._now()
		self._emit("grant_updated", grant_id, "ngo_grant", kwargs)
		return deepcopy(grant)

	async def activate_grant(self, grant_id: str, approved_by: str) -> dict[str, Any]:
		"""Transition a grant from pipeline/proposal to active."""
		grant = self._guard_grant(grant_id)
		if grant["status"] not in {"pipeline", "proposal"}:
			raise ValueError(f"cannot_activate_from:{grant['status']}")
		if not approved_by:
			raise ValueError("approved_by_required")
		grant["status"] = "active"
		grant["approved_by"] = approved_by
		grant["activated_at"] = self._now()
		grant["updated_at"] = self._now()
		self._emit("grant_activated", grant_id, "ngo_grant", {"approved_by": approved_by})
		return deepcopy(grant)

	async def close_grant(self, grant_id: str, closed_by: str, reason: str = "") -> dict[str, Any]:
		"""Close a grant and record final status."""
		grant = self._guard_grant(grant_id)
		grant["status"] = "closed"
		grant["closed_by"] = closed_by
		grant["close_reason"] = reason
		grant["closed_at"] = self._now()
		grant["updated_at"] = self._now()
		self._emit("grant_closed", grant_id, "ngo_grant", {"closed_by": closed_by, "reason": reason})
		return deepcopy(grant)

	async def delete_grant(self, grant_id: str) -> dict[str, Any]:
		"""Delete a pipeline-stage grant (not yet active)."""
		grant = self._guard_grant(grant_id)
		if grant["status"] not in {"pipeline", "cancelled"}:
			raise ValueError("only_pipeline_or_cancelled_grants_may_be_deleted")
		removed = self._grants.pop(grant_id)
		self._emit("grant_deleted", grant_id, "ngo_grant")
		return deepcopy(removed)

	# ── proposals ────────────────────────────────────────────────────────────

	async def list_proposals(self, grant_id: str | None = None) -> list[dict[str, Any]]:
		"""List proposals, optionally filtered by grant."""
		tenant = self._tenant()
		items = [deepcopy(p) for p in self._proposals.values() if p["tenant_id"] == tenant]
		if grant_id:
			items = [p for p in items if p["grant_id"] == grant_id]
		return items

	async def get_proposal(self, proposal_id: str) -> dict[str, Any]:
		"""Retrieve a proposal by ID."""
		tenant = self._tenant()
		proposal = self._proposals.get(proposal_id)
		if not proposal or proposal["tenant_id"] != tenant:
			raise KeyError(f"proposal_not_found:{proposal_id}")
		return deepcopy(proposal)

	async def create_proposal(
		self,
		grant_id: str,
		title: str,
		narrative: str,
		budget: Decimal,
		submitted_by: str,
		deadline: str,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Submit a grant proposal."""
		self._guard_grant(grant_id)
		if not submitted_by:
			raise ValueError("submitted_by_required")
		record: dict[str, Any] = {
			"id": self._id("prop"),
			"type": "ngo_grant_proposal",
			"tenant_id": self._tenant(),
			"grant_id": grant_id,
			"title": title,
			"narrative": narrative,
			"budget": budget,
			"currency": currency,
			"submitted_by": submitted_by,
			"deadline": deadline,
			"status": "submitted",
			"created_at": self._now(),
		}
		self._proposals[record["id"]] = record
		self._emit("proposal_submitted", record["id"], "ngo_grant_proposal", {"grant_id": grant_id})
		return deepcopy(record)

	async def approve_proposal(self, proposal_id: str, reviewed_by: str) -> dict[str, Any]:
		"""Approve a submitted proposal."""
		tenant = self._tenant()
		proposal = self._proposals.get(proposal_id)
		if not proposal or proposal["tenant_id"] != tenant:
			raise KeyError(f"proposal_not_found:{proposal_id}")
		proposal["status"] = "approved"
		proposal["reviewed_by"] = reviewed_by
		proposal["reviewed_at"] = self._now()
		self._emit("proposal_approved", proposal_id, "ngo_grant_proposal", {"reviewed_by": reviewed_by})
		return deepcopy(proposal)

	async def reject_proposal(self, proposal_id: str, reviewed_by: str, reason: str) -> dict[str, Any]:
		"""Reject a submitted proposal with reason."""
		tenant = self._tenant()
		proposal = self._proposals.get(proposal_id)
		if not proposal or proposal["tenant_id"] != tenant:
			raise KeyError(f"proposal_not_found:{proposal_id}")
		proposal["status"] = "rejected"
		proposal["reviewed_by"] = reviewed_by
		proposal["rejection_reason"] = reason
		proposal["reviewed_at"] = self._now()
		self._emit("proposal_rejected", proposal_id, "ngo_grant_proposal", {"reason": reason})
		return deepcopy(proposal)

	# ── budget lines ─────────────────────────────────────────────────────────

	async def list_budget_lines(self, grant_id: str) -> list[dict[str, Any]]:
		"""List budget lines for a grant."""
		self._guard_grant(grant_id)
		return [deepcopy(b) for b in self._budget_lines.values() if b["grant_id"] == grant_id]

	async def create_budget_line(
		self,
		grant_id: str,
		category: str,
		description: str,
		amount: Decimal,
		currency: str = "KES",
		period: str = "",
	) -> dict[str, Any]:
		"""Add a budget line to a grant."""
		self._guard_grant(grant_id)
		record: dict[str, Any] = {
			"id": self._id("bln"),
			"type": "ngo_budget_line",
			"tenant_id": self._tenant(),
			"grant_id": grant_id,
			"category": category,
			"description": description,
			"amount": amount,
			"spent_amount": Decimal("0"),
			"currency": currency,
			"period": period,
			"status": "active",
			"created_at": self._now(),
		}
		self._budget_lines[record["id"]] = record
		self._emit("budget_line_created", record["id"], "ngo_budget_line", {"grant_id": grant_id, "category": category})
		return deepcopy(record)

	async def update_budget_line_spent(self, line_id: str, spent_amount: Decimal) -> dict[str, Any]:
		"""Update the spent amount on a budget line."""
		tenant = self._tenant()
		line = self._budget_lines.get(line_id)
		if not line or line["tenant_id"] != tenant:
			raise KeyError(f"budget_line_not_found:{line_id}")
		if spent_amount > line["amount"]:
			raise ValueError("spent_exceeds_budget_line")
		line["spent_amount"] = spent_amount
		self._emit("budget_line_updated", line_id, "ngo_budget_line", {"spent_amount": str(spent_amount)})
		return deepcopy(line)

	async def get_budget_utilisation(self, grant_id: str) -> dict[str, Any]:
		"""Calculate budget utilisation summary for a grant."""
		grant = self._guard_grant(grant_id)
		lines = [b for b in self._budget_lines.values() if b["grant_id"] == grant_id]
		total_budget = sum(b["amount"] for b in lines)
		total_spent = sum(b["spent_amount"] for b in lines)
		utilisation_pct = float(total_spent / total_budget * 100) if total_budget else 0.0
		return {
			"grant_id": grant_id,
			"total_budget": total_budget,
			"total_spent": total_spent,
			"remaining": total_budget - total_spent,
			"utilisation_pct": round(utilisation_pct, 2),
			"line_count": len(lines),
			"currency": grant.get("currency", "KES"),
			"generated_at": self._now(),
		}

	# ── disbursements ─────────────────────────────────────────────────────────

	async def list_disbursements(self, grant_id: str | None = None) -> list[dict[str, Any]]:
		"""List disbursements, optionally filtered by grant."""
		tenant = self._tenant()
		items = [deepcopy(d) for d in self._disbursements.values() if d["tenant_id"] == tenant]
		if grant_id:
			items = [d for d in items if d["grant_id"] == grant_id]
		return items

	async def get_disbursement(self, disbursement_id: str) -> dict[str, Any]:
		"""Retrieve a disbursement by ID."""
		tenant = self._tenant()
		d = self._disbursements.get(disbursement_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"disbursement_not_found:{disbursement_id}")
		return deepcopy(d)

	async def create_disbursement(
		self,
		grant_id: str,
		amount: Decimal,
		disbursement_date: str,
		reference: str,
		approved_by: str,
		currency: str = "KES",
		payment_method: str = "bank_transfer",
		notes: str = "",
	) -> dict[str, Any]:
		"""Record a grant disbursement."""
		grant = self._guard_grant(grant_id)
		if grant["status"] != "active":
			raise ValueError("disbursements_require_active_grant")
		if payment_method not in SUPPORTED_PAYMENT_METHODS:
			raise ValueError(f"unsupported_payment_method:{payment_method}")
		if not approved_by:
			raise ValueError("approved_by_required")
		remaining = grant["amount"] - grant["disbursed_amount"]
		if amount > remaining:
			raise ValueError(f"disbursement_exceeds_remaining_balance:{remaining}")
		record: dict[str, Any] = {
			"id": self._id("dis"),
			"type": "ngo_disbursement",
			"tenant_id": self._tenant(),
			"grant_id": grant_id,
			"amount": amount,
			"currency": currency,
			"disbursement_date": disbursement_date,
			"reference": reference,
			"payment_method": payment_method,
			"approved_by": approved_by,
			"notes": notes,
			"status": "pending",
			"created_at": self._now(),
		}
		self._disbursements[record["id"]] = record
		grant["disbursed_amount"] += amount
		self._emit("disbursement_created", record["id"], "ngo_disbursement", {"grant_id": grant_id, "amount": str(amount)})
		return deepcopy(record)

	async def confirm_disbursement(self, disbursement_id: str, confirmed_by: str) -> dict[str, Any]:
		"""Confirm a pending disbursement."""
		tenant = self._tenant()
		d = self._disbursements.get(disbursement_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"disbursement_not_found:{disbursement_id}")
		if d["status"] != "pending":
			raise ValueError(f"cannot_confirm_{d['status']}_disbursement")
		d["status"] = "confirmed"
		d["confirmed_by"] = confirmed_by
		d["confirmed_at"] = self._now()
		self._emit("disbursement_confirmed", disbursement_id, "ngo_disbursement", {"confirmed_by": confirmed_by})
		return deepcopy(d)

	# ── compliance reports ────────────────────────────────────────────────────

	async def list_compliance_reports(self, grant_id: str | None = None) -> list[dict[str, Any]]:
		"""List compliance reports for the tenant."""
		tenant = self._tenant()
		items = [deepcopy(r) for r in self._compliance_reports.values() if r["tenant_id"] == tenant]
		if grant_id:
			items = [r for r in items if r["grant_id"] == grant_id]
		return items

	async def create_compliance_report(
		self,
		grant_id: str,
		report_type: str,
		period_start: str,
		period_end: str,
		submitted_by: str,
		narrative: str = "",
		attachments: list[str] | None = None,
	) -> dict[str, Any]:
		"""Submit a compliance report for a grant."""
		self._guard_grant(grant_id)
		if report_type not in SUPPORTED_REPORT_TYPES:
			raise ValueError(f"unsupported_report_type:{report_type}")
		record: dict[str, Any] = {
			"id": self._id("rpt"),
			"type": "ngo_compliance_report",
			"tenant_id": self._tenant(),
			"grant_id": grant_id,
			"report_type": report_type,
			"period_start": period_start,
			"period_end": period_end,
			"submitted_by": submitted_by,
			"narrative": narrative,
			"attachments": attachments or [],
			"status": "submitted",
			"created_at": self._now(),
		}
		self._compliance_reports[record["id"]] = record
		self._emit("compliance_report_submitted", record["id"], "ngo_compliance_report", {"grant_id": grant_id, "type": report_type})
		return deepcopy(record)

	async def approve_compliance_report(self, report_id: str, reviewed_by: str) -> dict[str, Any]:
		"""Approve a submitted compliance report."""
		tenant = self._tenant()
		report = self._compliance_reports.get(report_id)
		if not report or report["tenant_id"] != tenant:
			raise KeyError(f"compliance_report_not_found:{report_id}")
		report["status"] = "approved"
		report["reviewed_by"] = reviewed_by
		report["reviewed_at"] = self._now()
		self._emit("compliance_report_approved", report_id, "ngo_compliance_report", {"reviewed_by": reviewed_by})
		return deepcopy(report)

	# ── audit findings ────────────────────────────────────────────────────────

	async def list_audit_findings(self, grant_id: str | None = None) -> list[dict[str, Any]]:
		"""List audit findings for the tenant."""
		tenant = self._tenant()
		items = [deepcopy(f) for f in self._audit_findings.values() if f["tenant_id"] == tenant]
		if grant_id:
			items = [f for f in items if f["grant_id"] == grant_id]
		return items

	async def create_audit_finding(
		self,
		grant_id: str,
		finding_type: str,
		description: str,
		auditor: str,
		audit_date: str,
		severity: str = "medium",
		recommendations: str = "",
	) -> dict[str, Any]:
		"""Record an audit finding against a grant."""
		self._guard_grant(grant_id)
		if severity not in SUPPORTED_SEVERITY:
			raise ValueError(f"invalid_severity:{severity}")
		record: dict[str, Any] = {
			"id": self._id("aud"),
			"type": "ngo_audit_finding",
			"tenant_id": self._tenant(),
			"grant_id": grant_id,
			"finding_type": finding_type,
			"severity": severity,
			"description": description,
			"auditor": auditor,
			"audit_date": audit_date,
			"recommendations": recommendations,
			"status": "open",
			"created_at": self._now(),
		}
		self._audit_findings[record["id"]] = record
		self._emit("audit_finding_created", record["id"], "ngo_audit_finding", {"severity": severity, "grant_id": grant_id})
		_log.warning("Audit finding created: %s severity=%s grant=%s", record["id"], severity, grant_id)
		return deepcopy(record)

	async def resolve_audit_finding(self, finding_id: str, resolved_by: str, resolution_notes: str) -> dict[str, Any]:
		"""Mark an audit finding as resolved."""
		tenant = self._tenant()
		finding = self._audit_findings.get(finding_id)
		if not finding or finding["tenant_id"] != tenant:
			raise KeyError(f"audit_finding_not_found:{finding_id}")
		finding["status"] = "resolved"
		finding["resolved_by"] = resolved_by
		finding["resolution_notes"] = resolution_notes
		finding["resolved_at"] = self._now()
		self._emit("audit_finding_resolved", finding_id, "ngo_audit_finding", {"resolved_by": resolved_by})
		return deepcopy(finding)

	# ── analytics / reporting ─────────────────────────────────────────────────

	async def grant_portfolio_summary(self) -> dict[str, Any]:
		"""Return portfolio-level summary across all grants."""
		tenant = self._tenant()
		grants = [g for g in self._grants.values() if g["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		for g in grants:
			by_status[g["status"]] = by_status.get(g["status"], 0) + 1
		total_value = sum(g["amount"] for g in grants)
		total_disbursed = sum(g["disbursed_amount"] for g in grants)
		return {
			"tenant_id": tenant,
			"total_grants": len(grants),
			"by_status": by_status,
			"total_value": total_value,
			"total_disbursed": total_disbursed,
			"utilisation_pct": round(float(total_disbursed / total_value * 100), 2) if total_value else 0.0,
			"open_findings": sum(1 for f in self._audit_findings.values() if f["tenant_id"] == tenant and f["status"] == "open"),
			"generated_at": self._now(),
		}

	async def generate_donor_report(self, grant_id: str) -> dict[str, Any]:
		"""Generate a structured donor-facing report for a grant."""
		grant = self._guard_grant(grant_id)
		disbursements = [d for d in self._disbursements.values() if d["grant_id"] == grant_id]
		reports = [r for r in self._compliance_reports.values() if r["grant_id"] == grant_id]
		return {
			"grant_id": grant_id,
			"title": grant["title"],
			"donor_reference": grant["donor_reference"],
			"amount": grant["amount"],
			"disbursed_amount": grant["disbursed_amount"],
			"currency": grant["currency"],
			"status": grant["status"],
			"disbursement_count": len(disbursements),
			"compliance_report_count": len(reports),
			"period": f"{grant['start_date']} – {grant['end_date']}",
			"generated_at": self._now(),
		}

	async def check_grant_compliance_status(self, grant_id: str) -> dict[str, Any]:
		"""Return compliance status for a grant."""
		grant = self._guard_grant(grant_id)
		findings = [f for f in self._audit_findings.values() if f["grant_id"] == grant_id]
		open_critical = [f for f in findings if f["status"] == "open" and f["severity"] == "critical"]
		reports = [r for r in self._compliance_reports.values() if r["grant_id"] == grant_id]
		return {
			"grant_id": grant_id,
			"compliant": len(open_critical) == 0,
			"open_findings": len([f for f in findings if f["status"] == "open"]),
			"critical_findings": len(open_critical),
			"submitted_reports": len(reports),
			"approved_reports": len([r for r in reports if r["status"] == "approved"]),
			"checked_at": self._now(),
		}

	async def bulk_create_budget_lines(self, grant_id: str, lines: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create budget lines for a grant."""
		results, errors = [], []
		tasks = [
			self.create_budget_line(
				grant_id=grant_id,
				category=line.get("category", "general"),
				description=line.get("description", ""),
				amount=Decimal(str(line.get("amount", 0))),
				currency=line.get("currency", "KES"),
				period=line.get("period", ""),
			)
			for line in lines
		]
		outcomes = await asyncio.gather(*tasks, return_exceptions=True)
		for line, outcome in zip(lines, outcomes):
			if isinstance(outcome, Exception):
				errors.append({"input": line, "error": str(outcome)})
			else:
				results.append(outcome)
		return {"created": len(results), "failed": len(errors), "lines": results, "errors": errors}

	async def get_disbursement_schedule(self, grant_id: str) -> dict[str, Any]:
		"""Return projected disbursement schedule vs actuals."""
		grant = self._guard_grant(grant_id)
		disbursements = [d for d in self._disbursements.values() if d["grant_id"] == grant_id]
		confirmed = [d for d in disbursements if d["status"] == "confirmed"]
		pending = [d for d in disbursements if d["status"] == "pending"]
		return {
			"grant_id": grant_id,
			"total_amount": grant["amount"],
			"disbursed_amount": grant["disbursed_amount"],
			"remaining_amount": grant["amount"] - grant["disbursed_amount"],
			"confirmed_disbursements": len(confirmed),
			"pending_disbursements": len(pending),
			"confirmed_total": sum(d["amount"] for d in confirmed),
			"pending_total": sum(d["amount"] for d in pending),
			"currency": grant["currency"],
			"generated_at": self._now(),
		}
