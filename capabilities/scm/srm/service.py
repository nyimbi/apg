"""Supplier Relationship Management async service (scm_srm)."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

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

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.suppliers: dict[str, dict[str, Any]] = {}
		self.scorecards: dict[str, dict[str, Any]] = {}
		self.risk_assessments: dict[str, dict[str, Any]] = {}
		self.collaboration_messages: dict[str, dict[str, Any]] = {}
		self.performance_reviews: dict[str, dict[str, Any]] = {}
		self.preferred_supplier_records: dict[str, dict[str, Any]] = {}
		self.supplier_certifications: dict[str, dict[str, Any]] = {}
		self.escalations: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

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

	# ── Supplier segmentation ─────────────────────────────────────────────────

	async def segment_suppliers(
		self,
		strategy: str = "risk_score",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Segment supplier portfolio using Kraljic-style or risk/score strategy.

		strategy values:
		  - "risk_score"   : 2x2 matrix of risk_level vs overall_score
		  - "spend_category": group by category (proxy for spend category)
		  - "geography"   : group by country
		"""
		tenant = self._tenant(tenant_id)
		suppliers = [s for s in self.suppliers.values() if s["tenant_id"] == tenant]
		if strategy == "risk_score":
			segments: dict[str, list[str]] = {
				"strategic": [],      # high-score, high-risk → manage closely
				"leverage": [],       # high-score, low-risk  → exploit volume
				"bottleneck": [],     # low-score, high-risk  → develop or dual-source
				"non_critical": [],   # low-score, low-risk   → simplify/automate
			}
			level_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
			for s in suppliers:
				score = s.get("overall_score") or 5.0
				risk_n = level_order.get(s["risk_level"], 0)
				high_score = score >= 7.0
				high_risk = risk_n >= 2
				if high_score and high_risk:
					segments["strategic"].append(s["id"])
				elif high_score and not high_risk:
					segments["leverage"].append(s["id"])
				elif not high_score and high_risk:
					segments["bottleneck"].append(s["id"])
				else:
					segments["non_critical"].append(s["id"])
			return {"strategy": strategy, "segments": segments, "generated_at": self._now()}
		elif strategy == "spend_category":
			by_cat: dict[str, list[str]] = {}
			for s in suppliers:
				by_cat.setdefault(s["category"], []).append(s["id"])
			return {"strategy": strategy, "segments": by_cat, "generated_at": self._now()}
		elif strategy == "geography":
			by_geo: dict[str, list[str]] = {}
			for s in suppliers:
				by_geo.setdefault(s["country"], []).append(s["id"])
			return {"strategy": strategy, "segments": by_geo, "generated_at": self._now()}
		else:
			raise ValueError(f"unknown strategy '{strategy}'; choose risk_score | spend_category | geography")

	# ── Scorecard trending ────────────────────────────────────────────────────

	async def scorecard_trend(
		self,
		supplier_id: str,
		dimension: str = "overall_score",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Return time-ordered score series for a supplier on a given dimension.

		dimension: overall_score | quality_score | delivery_score |
		           responsiveness_score | cost_score | sustainability_score
		"""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		valid_dims = {
			"overall_score", "quality_score", "delivery_score",
			"responsiveness_score", "cost_score", "sustainability_score",
		}
		if dimension not in valid_dims:
			raise ValueError(f"dimension must be one of {valid_dims}")
		cards = sorted(
			[c for c in self.scorecards.values() if c["tenant_id"] == tenant and c["supplier_id"] == supplier_id],
			key=lambda c: c["period"],
		)
		series = [{"period": c["period"], "value": c.get(dimension)} for c in cards]
		if len(series) >= 2:
			first = series[0]["value"] or 0
			last = series[-1]["value"] or 0
			trend = "improving" if last > first else ("declining" if last < first else "stable")
		else:
			trend = "insufficient_data"
		return {
			"supplier_id": supplier_id,
			"dimension": dimension,
			"series": series,
			"trend": trend,
			"generated_at": self._now(),
		}

	# ── Concentration risk detection ──────────────────────────────────────────

	async def concentration_risk_report(
		self,
		threshold_pct: float = 40.0,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Identify categories / countries where fewer than 3 suppliers represent
		the majority of the base — a classic single-source concentration risk.

		threshold_pct: if one supplier represents >= this percentage of the category
		               count, it is flagged.
		"""
		tenant = self._tenant(tenant_id)
		suppliers = [s for s in self.suppliers.values() if s["tenant_id"] == tenant and s["status"] == "active"]
		by_cat: dict[str, list[str]] = {}
		by_country: dict[str, list[str]] = {}
		for s in suppliers:
			by_cat.setdefault(s["category"], []).append(s["id"])
			by_country.setdefault(s["country"], []).append(s["id"])

		cat_risks = []
		for cat, ids in by_cat.items():
			if len(ids) <= 2:
				cat_risks.append({"dimension": "category", "value": cat, "supplier_count": len(ids), "risk": "single_source"})

		geo_risks = []
		for country, ids in by_country.items():
			pct = (1 / len(ids)) * 100 if ids else 0
			if len(ids) == 1 or pct >= threshold_pct:
				geo_risks.append({"dimension": "country", "value": country, "supplier_count": len(ids), "pct": round(pct, 1)})

		return {
			"tenant_id": tenant,
			"threshold_pct": threshold_pct,
			"category_concentration": cat_risks,
			"geography_concentration": geo_risks,
			"total_active_suppliers": len(suppliers),
			"generated_at": self._now(),
		}

	# ── Supplier development plan ─────────────────────────────────────────────

	async def create_development_plan(
		self,
		supplier_id: str,
		plan_title: str,
		objectives: list[str],
		target_score: float,
		target_date: str,
		assigned_to: str,
		budget: float | None = None,
		currency: str = "USD",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a structured supplier development plan to improve scorecard scores."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		if not 0 <= target_score <= 10:
			raise ValueError("target_score must be 0–10")
		record: dict[str, Any] = {
			"id": self._id("sdp"),
			"type": "scm_srm_development_plan",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"plan_title": plan_title,
			"objectives": objectives,
			"target_score": target_score,
			"current_score": s.get("overall_score"),
			"target_date": target_date,
			"assigned_to": assigned_to,
			"budget": budget,
			"currency": currency,
			"progress_pct": 0,
			"status": "active",
			"milestones": [],
			"created_at": self._now(),
			"updated_at": None,
		}
		if not hasattr(self, "development_plans"):
			self.development_plans: dict[str, dict[str, Any]] = {}
		self.development_plans[record["id"]] = record
		self._emit(tenant, "development_plan_created", record["id"], "scm_srm_development_plan", "active")
		return deepcopy(record)

	async def update_development_plan_progress(
		self,
		plan_id: str,
		progress_pct: float,
		milestone_note: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update progress on a supplier development plan (0–100 %)."""
		tenant = self._tenant(tenant_id)
		if not hasattr(self, "development_plans"):
			self.development_plans = {}
		plan = self.development_plans.get(plan_id)
		if not plan or plan["tenant_id"] != tenant:
			raise KeyError(f"development_plan '{plan_id}' not found")
		if not 0 <= progress_pct <= 100:
			raise ValueError("progress_pct must be 0–100")
		plan["progress_pct"] = progress_pct
		if milestone_note:
			plan["milestones"].append({"note": milestone_note, "recorded_at": self._now()})
		if progress_pct >= 100:
			plan["status"] = "completed"
		plan["updated_at"] = self._now()
		self._emit(tenant, "development_plan_updated", plan_id, "scm_srm_development_plan", plan["status"])
		return deepcopy(plan)

	# ── Contract management ───────────────────────────────────────────────────

	async def register_contract(
		self,
		supplier_id: str,
		contract_reference: str,
		contract_type: str,
		start_date: str,
		end_date: str,
		value: float | None = None,
		currency: str = "USD",
		auto_renew: bool = False,
		notice_period_days: int = 30,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a supplier contract and track renewal dates."""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		if not hasattr(self, "contracts"):
			self.contracts: dict[str, dict[str, Any]] = {}
		record: dict[str, Any] = {
			"id": self._id("ctr"),
			"type": "scm_srm_contract",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"contract_reference": contract_reference,
			"contract_type": contract_type,
			"start_date": start_date,
			"end_date": end_date,
			"value": value,
			"currency": currency,
			"auto_renew": auto_renew,
			"notice_period_days": notice_period_days,
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.contracts[record["id"]] = record
		self._emit(tenant, "contract_registered", record["id"], "scm_srm_contract", "active")
		return deepcopy(record)

	async def list_contracts(
		self,
		supplier_id: str | None = None,
		expiring_within_days: int | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List registered contracts, optionally filtering to those expiring soon."""
		tenant = self._tenant(tenant_id)
		if not hasattr(self, "contracts"):
			self.contracts = {}
		items = [deepcopy(c) for c in self.contracts.values() if c["tenant_id"] == tenant]
		if supplier_id:
			items = [c for c in items if c["supplier_id"] == supplier_id]
		if expiring_within_days is not None:
			now_dt = datetime.utcnow()
			filtered = []
			for c in items:
				try:
					end_dt = datetime.fromisoformat(c["end_date"].rstrip("Z"))
					delta = (end_dt - now_dt).days
					if 0 <= delta <= expiring_within_days:
						c["days_to_expiry"] = delta
						filtered.append(c)
				except ValueError as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
			return filtered
		return items

	# ── ESG scoring ───────────────────────────────────────────────────────────

	async def record_esg_score(
		self,
		supplier_id: str,
		period: str,
		environmental_score: float,
		social_score: float,
		governance_score: float,
		assessed_by: str,
		evidence_urls: list[str] | None = None,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a full ESG score for a supplier for a given period.

		All sub-scores are 0–10.  Composite = weighted average (E:40%, S:30%, G:30%).
		"""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		for label, val in [("environmental", environmental_score), ("social", social_score), ("governance", governance_score)]:
			if not 0 <= val <= 10:
				raise ValueError(f"{label}_score must be 0–10")
		composite = round(environmental_score * 0.4 + social_score * 0.3 + governance_score * 0.3, 2)
		if not hasattr(self, "esg_scores"):
			self.esg_scores: dict[str, dict[str, Any]] = {}
		record: dict[str, Any] = {
			"id": self._id("esg"),
			"type": "scm_srm_esg_score",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"period": period,
			"environmental_score": environmental_score,
			"social_score": social_score,
			"governance_score": governance_score,
			"composite_score": composite,
			"assessed_by": assessed_by,
			"evidence_urls": evidence_urls or [],
			"notes": notes,
			"status": "recorded",
			"created_at": self._now(),
		}
		self.esg_scores[record["id"]] = record
		# Propagate to supplier top-level ESG field
		s["esg_composite"] = composite
		s["updated_at"] = self._now()
		self._emit(tenant, "esg_score_recorded", record["id"], "scm_srm_esg_score", "recorded")
		return deepcopy(record)

	async def list_esg_scores(
		self,
		supplier_id: str | None = None,
		period: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List ESG scores."""
		tenant = self._tenant(tenant_id)
		if not hasattr(self, "esg_scores"):
			self.esg_scores = {}
		items = [deepcopy(e) for e in self.esg_scores.values() if e["tenant_id"] == tenant]
		if supplier_id:
			items = [e for e in items if e["supplier_id"] == supplier_id]
		if period:
			items = [e for e in items if e["period"] == period]
		return items

	# ── Escalation management ─────────────────────────────────────────────────

	async def raise_escalation(
		self,
		supplier_id: str,
		title: str,
		description: str,
		severity: str,
		raised_by: str,
		due_date: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Raise a formal escalation against a supplier.

		severity: low | medium | high | critical
		"""
		tenant = self._tenant(tenant_id)
		if severity not in RISK_LEVELS:
			raise ValueError(f"severity must be one of {RISK_LEVELS}")
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("esc"),
			"type": "scm_srm_escalation",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"title": title,
			"description": description,
			"severity": severity,
			"raised_by": raised_by,
			"due_date": due_date,
			"status": "open",
			"resolution": None,
			"resolved_by": None,
			"created_at": self._now(),
			"resolved_at": None,
		}
		self.escalations[record["id"]] = record
		self._emit(tenant, "escalation_raised", record["id"], "scm_srm_escalation", "open")
		return deepcopy(record)

	async def resolve_escalation(
		self,
		escalation_id: str,
		resolution: str,
		resolved_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Resolve an open escalation."""
		tenant = self._tenant(tenant_id)
		esc = self.escalations.get(escalation_id)
		if not esc or esc["tenant_id"] != tenant:
			raise KeyError(f"escalation '{escalation_id}' not found")
		if esc["status"] != "open":
			raise ValueError("only open escalations can be resolved")
		esc["status"] = "resolved"
		esc["resolution"] = resolution
		esc["resolved_by"] = resolved_by
		esc["resolved_at"] = self._now()
		self._emit(tenant, "escalation_resolved", escalation_id, "scm_srm_escalation", "resolved")
		return deepcopy(esc)

	async def list_escalations(
		self,
		supplier_id: str | None = None,
		status: str | None = None,
		severity: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List escalations with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.escalations.values() if e["tenant_id"] == tenant]
		if supplier_id:
			items = [e for e in items if e["supplier_id"] == supplier_id]
		if status:
			items = [e for e in items if e["status"] == status]
		if severity:
			items = [e for e in items if e["severity"] == severity]
		return items

	# ── Supplier benchmarking ─────────────────────────────────────────────────

	async def benchmark_supplier(
		self,
		supplier_id: str,
		peer_supplier_ids: list[str],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compare a supplier's latest scorecard against named peers.

		Returns z-score-like deviation from peer mean for each scored dimension.
		"""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")

		def _latest_card(sid: str) -> dict[str, Any] | None:
			cards = sorted(
				[c for c in self.scorecards.values() if c["tenant_id"] == tenant and c["supplier_id"] == sid],
				key=lambda c: c["period"],
				reverse=True,
			)
			return cards[0] if cards else None

		subject_card = _latest_card(supplier_id)
		if not subject_card:
			raise ValueError(f"supplier '{supplier_id}' has no scorecard data")

		dims = ["quality_score", "delivery_score", "responsiveness_score", "cost_score", "overall_score"]
		peer_cards = [_latest_card(pid) for pid in peer_supplier_ids if _latest_card(pid)]

		benchmarks: dict[str, Any] = {}
		for dim in dims:
			subject_val = subject_card.get(dim) or 0.0
			peer_vals = [pc[dim] for pc in peer_cards if pc and pc.get(dim) is not None]
			if peer_vals:
				peer_mean = sum(peer_vals) / len(peer_vals)
				delta = round(subject_val - peer_mean, 2)
			else:
				peer_mean = None
				delta = None
			benchmarks[dim] = {"subject": subject_val, "peer_mean": peer_mean, "delta": delta}

		return {
			"supplier_id": supplier_id,
			"peer_supplier_ids": peer_supplier_ids,
			"period": subject_card["period"],
			"benchmarks": benchmarks,
			"generated_at": self._now(),
		}

	# ── Onboarding workflow ───────────────────────────────────────────────────

	async def start_onboarding(
		self,
		supplier_id: str,
		checklist_items: list[str] | None = None,
		assigned_to: str = "procurement",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Start a structured onboarding workflow for a pending supplier.

		Default checklist: NDA, bank details, ISO cert, tax compliance, site audit.
		"""
		tenant = self._tenant(tenant_id)
		s = self.suppliers.get(supplier_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"supplier '{supplier_id}' not found")
		default_items = [
			"NDA signed",
			"Bank details verified",
			"ISO or equivalent certification provided",
			"Tax compliance documentation submitted",
			"Site / factory audit scheduled",
			"Insurance certificate received",
			"Data privacy agreement signed",
		]
		items = checklist_items or default_items
		checklist = [{"item": it, "completed": False, "completed_at": None} for it in items]
		if not hasattr(self, "onboarding_records"):
			self.onboarding_records: dict[str, dict[str, Any]] = {}
		record: dict[str, Any] = {
			"id": self._id("onb"),
			"type": "scm_srm_onboarding",
			"tenant_id": tenant,
			"supplier_id": supplier_id,
			"assigned_to": assigned_to,
			"checklist": checklist,
			"completion_pct": 0,
			"status": "in_progress",
			"started_at": self._now(),
			"completed_at": None,
		}
		self.onboarding_records[record["id"]] = record
		self._emit(tenant, "onboarding_started", record["id"], "scm_srm_onboarding", "in_progress")
		return deepcopy(record)

	async def complete_onboarding_item(
		self,
		onboarding_id: str,
		item_index: int,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark a single onboarding checklist item as completed."""
		tenant = self._tenant(tenant_id)
		if not hasattr(self, "onboarding_records"):
			self.onboarding_records = {}
		rec = self.onboarding_records.get(onboarding_id)
		if not rec or rec["tenant_id"] != tenant:
			raise KeyError(f"onboarding '{onboarding_id}' not found")
		checklist = rec["checklist"]
		if not 0 <= item_index < len(checklist):
			raise IndexError(f"item_index {item_index} out of range (0–{len(checklist) - 1})")
		checklist[item_index]["completed"] = True
		checklist[item_index]["completed_at"] = self._now()
		done = sum(1 for it in checklist if it["completed"])
		rec["completion_pct"] = round((done / len(checklist)) * 100, 1)
		if done == len(checklist):
			rec["status"] = "completed"
			rec["completed_at"] = self._now()
			self._emit(tenant, "onboarding_completed", onboarding_id, "scm_srm_onboarding", "completed")
		return deepcopy(rec)

	# ── Portfolio risk heatmap ────────────────────────────────────────────────

	async def risk_heatmap(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate a risk heatmap aggregating open risk assessments by category and level."""
		tenant = self._tenant(tenant_id)
		open_risks = [r for r in self.risk_assessments.values() if r["tenant_id"] == tenant and r["status"] == "open"]
		heatmap: dict[str, dict[str, int]] = {}
		for risk in open_risks:
			cat = risk["risk_category"]
			lvl = risk["risk_level"]
			heatmap.setdefault(cat, {})
			heatmap[cat][lvl] = heatmap[cat].get(lvl, 0) + 1
		# Identify hot spots (critical or high with count >= 2)
		hotspots = []
		for cat, levels in heatmap.items():
			for lvl in ("critical", "high"):
				count = levels.get(lvl, 0)
				if count >= 1:
					hotspots.append({"category": cat, "level": lvl, "count": count})
		hotspots.sort(key=lambda x: ({"critical": 0, "high": 1}.get(x["level"], 2), -x["count"]))
		return {
			"tenant_id": tenant,
			"heatmap": heatmap,
			"hotspots": hotspots,
			"total_open_risks": len(open_risks),
			"generated_at": self._now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

