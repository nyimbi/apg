"""Supplier Relationship Management async service (scm_srm)."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_srm"
SUPPLIER_CATEGORIES = {"raw_material", "packaging", "services", "technology", "logistics", "equipment", "consumables"}
RISK_LEVELS = {"low", "medium", "high", "critical"}
RISK_CATEGORIES = {"financial", "geopolitical", "operational", "compliance", "esg", "concentration"}
MESSAGE_TYPES = {"general", "forecast_share", "po_update", "complaint", "escalation", "nda", "performance_review"}
SUPPLIER_STATUSES = {"active", "pending_approval", "probation", "suspended", "blacklisted", "inactive"}


class SupplierRelationshipService:
	"""Async service for supplier scorecard, risk assessment, collaboration portal,
	performance reviews and preferred supplier status management."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.suppliers: dict[str, dict[str, Any]] = {}
		self.scorecards: dict[str, dict[str, Any]] = {}
		self.risk_assessments: dict[str, dict[str, Any]] = {}
		self.collaboration_messages: dict[str, dict[str, Any]] = {}
		self.performance_reviews: dict[str, dict[str, Any]] = {}
		self.preferred_supplier_records: dict[str, dict[str, Any]] = {}
		self.supplier_certifications: dict[str, dict[str, Any]] = {}
		self.escalations: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self, tenant_id: str | None = None) -> str:
		t = tenant_id or self.tenant_id
		if not t:
			raise PermissionError("tenant_context_required")
		return t

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, status: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"status": status,
			"capability_id": CAPABILITY_ID,
			"emitted_at": self._now(),
		})

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"supplier_count": len(self.suppliers),
			"active_suppliers": sum(1 for s in self.suppliers.values() if s["status"] == "active"),
			"preferred_suppliers": sum(1 for s in self.suppliers.values() if s.get("preferred")),
			"open_risks": sum(1 for r in self.risk_assessments.values() if r["status"] == "open"),
			"unread_messages": sum(1 for m in self.collaboration_messages.values() if m["status"] == "sent"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "scm",
			"version": "1.0.0",
			"description": "Supplier scorecard, risk assessment, collaboration portal, performance reviews, preferred supplier status",
			"supplier_categories": sorted(SUPPLIER_CATEGORIES),
			"risk_levels": sorted(RISK_LEVELS),
			"risk_categories": sorted(RISK_CATEGORIES),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Supplier CRUD ─────────────────────────────────────────────────────────

	async def create_supplier(
		self,
		name: str,
		supplier_code: str,
		country: str,
		category: str,
		contact_email: str | None = None,
		contact_phone: str | None = None,
		payment_terms: str = "NET30",
		currency: str = "USD",
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a new supplier."""
		tenant = self._tenant(tenant_id)
		if category not in SUPPLIER_CATEGORIES:
			raise ValueError(f"category must be one of {SUPPLIER_CATEGORIES}")
		for s in self.suppliers.values():
			if s["tenant_id"] == tenant and s["supplier_code"] == supplier_code:
				raise ValueError(f"supplier_code '{supplier_code}' already exists for tenant")
		record: dict[str, Any] = {
			"id": self._id("supp"),
			"type": "scm_srm_supplier",
			"tenant_id": tenant,
			"name": name,
			"supplier_code": supplier_code,
			"country": country,
			"category": category,
			"contact_email": contact_email,
			"contact_phone": contact_phone,
			"payment_terms": payment_terms,
			"currency": currency,
			"preferred": False,
			"risk_level": "low",
			"overall_score": None,
			"notes": notes,
			"status": "pending_approval",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.suppliers[record["id"]] = record
		self._emit(tenant, "supplier_created", record["id"], "scm_srm_supplier", "pending_approval")
		return deepcopy(record)

	async def list_suppliers(
		self,
		status: str | None = None,
		category: str | None = None,
		preferred_only: bool = False,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List suppliers with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.suppliers.values() if s["tenant_id"] == tenant]
		if status:
			items = [s for s in items if s["status"] == status]
		if category:
			items = [s for s in items if s["category"] == category]
		if preferred_only:
			items = [s for s in items if s.get("preferred")]
		return items

	async def get_supplier(self, supplier_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single supplier."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		return deepcopy(s)

	async def update_supplier(
		self,
		supplier_id: str,
		updates: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update supplier details."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		allowed = {"name", "contact_email", "contact_phone", "payment_terms", "status", "notes"}
		for k, v in updates.items():
			if k in allowed:
				s[k] = v
		s["updated_at"] = self._now()
		self._emit(tenant, "supplier_updated", supplier_id, "scm_srm_supplier", s["status"])
		return deepcopy(s)

	async def delete_supplier(self, supplier_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Deactivate a supplier."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		s["status"] = "inactive"
		s["updated_at"] = self._now()
		self._emit(tenant, "supplier_deactivated", supplier_id, "scm_srm_supplier", "inactive")
		return deepcopy(s)

	async def approve_supplier(self, supplier_id: str, approved_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Approve a pending supplier for trading."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		if s["status"] != "pending_approval":
			raise ValueError("only pending_approval suppliers can be approved")
		s["status"] = "active"
		s["approved_by"] = approved_by
		s["approved_at"] = self._now()
		s["updated_at"] = self._now()
		self._emit(tenant, "supplier_approved", supplier_id, "scm_srm_supplier", "active")
		return deepcopy(s)

	async def suspend_supplier(self, supplier_id: str, reason: str, suspended_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Suspend an active supplier."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		s["status"] = "suspended"
		s["suspension_reason"] = reason
		s["suspended_by"] = suspended_by
		s["suspended_at"] = self._now()
		s["updated_at"] = self._now()
		self._emit(tenant, "supplier_suspended", supplier_id, "scm_srm_supplier", "suspended")
		return deepcopy(s)

	# ── Preferred supplier status ─────────────────────────────────────────────

	async def set_preferred_status(
		self,
		supplier_id: str,
		preferred: bool,
		reason: str,
		set_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Grant or revoke preferred supplier status."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		s["preferred"] = preferred
		s["preferred_reason"] = reason
		s["preferred_set_by"] = set_by
		s["preferred_set_at"] = self._now()
		s["updated_at"] = self._now()
		event = "supplier_preferred_granted" if preferred else "supplier_preferred_revoked"
		self._emit(tenant, event, supplier_id, "scm_srm_supplier", s["status"])
		return deepcopy(s)

	# ── Scorecard ─────────────────────────────────────────────────────────────

	async def create_scorecard(
		self,
		supplier_id: str,
		period: str,
		quality_score: float,
		delivery_score: float,
		responsiveness_score: float,
		cost_score: float,
		sustainability_score: float | None = None,
		reviewed_by: str = "system",
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a supplier scorecard for a period."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		scores = [quality_score, delivery_score, responsiveness_score, cost_score]
		if sustainability_score is not None:
			scores.append(sustainability_score)
		for sc in scores:
			if not 0 <= sc <= 10:
				raise ValueError("all scores must be between 0 and 10")
		overall = round(sum(scores) / len(scores), 2)
		record: dict[str, Any] = {
			"id": self._id("sc"),
			"type": "scm_srm_scorecard",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"period": period,
			"quality_score": quality_score,
			"delivery_score": delivery_score,
			"responsiveness_score": responsiveness_score,
			"cost_score": cost_score,
			"sustainability_score": sustainability_score,
			"overall_score": overall,
			"reviewed_by": reviewed_by,
			"notes": notes,
			"status": "completed",
			"created_at": self._now(),
		}
		self.scorecards[record["id"]] = record
		# Update supplier's overall score
		s["overall_score"] = overall
		s["updated_at"] = self._now()
		self._emit(tenant, "scorecard_created", record["id"], "scm_srm_scorecard", "completed")
		return deepcopy(record)

	async def list_scorecards(
		self,
		supplier_id: str | None = None,
		period: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List scorecards."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.scorecards.values() if s["tenant_id"] == tenant]
		if supplier_id:
			items = [s for s in items if s["supplier_id"] == supplier_id]
		if period:
			items = [s for s in items if s["period"] == period]
		return items

	# ── Risk assessment ───────────────────────────────────────────────────────

	async def create_risk_assessment(
		self,
		supplier_id: str,
		risk_category: str,
		risk_level: str,
		description: str,
		assessed_by: str,
		mitigation_plan: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a risk assessment for a supplier."""
		tenant = self._tenant(tenant_id)
		if risk_category not in RISK_CATEGORIES:
			raise ValueError(f"risk_category must be one of {RISK_CATEGORIES}")
		if risk_level not in RISK_LEVELS:
			raise ValueError(f"risk_level must be one of {RISK_LEVELS}")
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("risk"),
			"type": "scm_srm_risk_assessment",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"risk_category": risk_category,
			"risk_level": risk_level,
			"description": description,
			"mitigation_plan": mitigation_plan,
			"assessed_by": assessed_by,
			"status": "open",
			"created_at": self._now(),
			"reviewed_at": None,
		}
		self.risk_assessments[record["id"]] = record
		# Escalate supplier risk level
		level_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
		if level_order.get(risk_level, 0) > level_order.get(s["risk_level"], 0):
			s["risk_level"] = risk_level
			s["updated_at"] = self._now()
		self._emit(tenant, "risk_assessment_created", record["id"], "scm_srm_risk_assessment", "open")
		return deepcopy(record)

	async def review_risk_assessment(
		self,
		assessment_id: str,
		reviewed_by: str,
		outcome: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark a risk assessment as reviewed."""
		tenant = self._tenant(tenant_id)
		r = self.risk_assessments.get(assessment_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"risk_assessment '{assessment_id}' not found")
		r["status"] = "reviewed"
		r["reviewed_by"] = reviewed_by
		r["outcome"] = outcome
		r["reviewed_at"] = self._now()
		self._emit(tenant, "risk_assessment_reviewed", assessment_id, "scm_srm_risk_assessment", "reviewed")
		return deepcopy(r)

	async def list_risk_assessments(
		self,
		supplier_id: str | None = None,
		risk_level: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List risk assessments."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.risk_assessments.values() if r["tenant_id"] == tenant]
		if supplier_id:
			items = [r for r in items if r["supplier_id"] == supplier_id]
		if risk_level:
			items = [r for r in items if r["risk_level"] == risk_level]
		return items

	# ── Collaboration portal ──────────────────────────────────────────────────

	async def send_collaboration_message(
		self,
		supplier_id: str,
		subject: str,
		body: str,
		message_type: str = "general",
		attachments: list[str] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Send a message to a supplier via the collaboration portal."""
		tenant = self._tenant(tenant_id)
		if message_type not in MESSAGE_TYPES:
			raise ValueError(f"message_type must be one of {MESSAGE_TYPES}")
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("msg"),
			"type": "scm_srm_collaboration_message",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"subject": subject,
			"body": body,
			"message_type": message_type,
			"attachments": attachments or [],
			"status": "sent",
			"sent_at": self._now(),
		}
		self.collaboration_messages[record["id"]] = record
		self._emit(tenant, "collaboration_message_sent", record["id"], "scm_srm_collaboration_message", "sent")
		return deepcopy(record)

	async def list_collaboration_messages(
		self,
		supplier_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List collaboration messages."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(m) for m in self.collaboration_messages.values() if m["tenant_id"] == tenant]
		if supplier_id:
			items = [m for m in items if m["supplier_id"] == supplier_id]
		return items

	# ── Performance reviews ───────────────────────────────────────────────────

	async def create_performance_review(
		self,
		supplier_id: str,
		review_period: str,
		reviewer: str,
		summary: str,
		action_items: list[str] | None = None,
		next_review_date: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a formal supplier performance review."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("prev"),
			"type": "scm_srm_performance_review",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"review_period": review_period,
			"reviewer": reviewer,
			"summary": summary,
			"action_items": action_items or [],
			"next_review_date": next_review_date,
			"status": "completed",
			"created_at": self._now(),
		}
		self.performance_reviews[record["id"]] = record
		self._emit(tenant, "performance_review_completed", record["id"], "scm_srm_performance_review", "completed")
		return deepcopy(record)

	async def list_performance_reviews(
		self,
		supplier_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List performance reviews."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.performance_reviews.values() if r["tenant_id"] == tenant]
		if supplier_id:
			items = [r for r in items if r["supplier_id"] == supplier_id]
		return items

	# ── Supplier certification ────────────────────────────────────────────────

	async def add_certification(
		self,
		supplier_id: str,
		cert_type: str,
		cert_number: str,
		issuing_body: str,
		valid_from: str,
		valid_until: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a supplier certification (ISO, halal, organic, etc.)."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("cert"),
			"type": "scm_srm_certification",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"cert_type": cert_type,
			"cert_number": cert_number,
			"issuing_body": issuing_body,
			"valid_from": valid_from,
			"valid_until": valid_until,
			"status": "active",
			"created_at": self._now(),
		}
		self.supplier_certifications[record["id"]] = record
		self._emit(tenant, "certification_added", record["id"], "scm_srm_certification", "active")
		return deepcopy(record)

	async def list_certifications(
		self,
		supplier_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List supplier certifications."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.supplier_certifications.values() if c["tenant_id"] == tenant]
		if supplier_id:
			items = [c for c in items if c["supplier_id"] == supplier_id]
		return items

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def supplier_analytics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Aggregate supplier portfolio analytics."""
		tenant = self._tenant(tenant_id)
		all_suppliers = [s for s in self.suppliers.values() if s["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		by_category: dict[str, int] = {}
		by_risk: dict[str, int] = {}
		for s in all_suppliers:
			by_status[s["status"]] = by_status.get(s["status"], 0) + 1
			by_category[s["category"]] = by_category.get(s["category"], 0) + 1
			by_risk[s["risk_level"]] = by_risk.get(s["risk_level"], 0) + 1
		scored = [s for s in all_suppliers if s.get("overall_score") is not None]
		avg_score = round(sum(s["overall_score"] for s in scored) / len(scored), 2) if scored else None
		return {
			"tenant_id": tenant,
			"total_suppliers": len(all_suppliers),
			"by_status": by_status,
			"by_category": by_category,
			"by_risk_level": by_risk,
			"preferred_count": sum(1 for s in all_suppliers if s.get("preferred")),
			"avg_score": avg_score,
			"open_risks": sum(1 for r in self.risk_assessments.values() if r["tenant_id"] == tenant and r["status"] == "open"),
			"certifications": len([c for c in self.supplier_certifications.values() if c["tenant_id"] == tenant and c["status"] == "active"]),
			"generated_at": self._now(),
		}

	async def bulk_create_suppliers(
		self,
		suppliers_data: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-register multiple suppliers."""
		tenant = self._tenant(tenant_id)
		tasks = [self.create_supplier(tenant_id=tenant, **s) for s in suppliers_data]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for item in raw:
			if isinstance(item, Exception):
				errors.append(str(item))
			else:
				results.append(item)
		return {"created": len(results), "failed": len(errors), "suppliers": results, "errors": errors}
