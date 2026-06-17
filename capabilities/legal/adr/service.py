"""ADR / Dispute Resolution — async service layer."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
from copy import deepcopy
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CASE_TYPES = {"arbitration", "mediation", "conciliation", "expert_determination", "adjudication"}
CASE_STATUSES = {
	"filed", "notice_served", "panel_constituted", "preliminary_conference",
	"hearings", "deliberation", "award_rendered", "enforcement", "closed", "settled",
}
NEUTRAL_ROLES = {"sole_arbitrator", "presiding_arbitrator", "co_arbitrator", "mediator", "conciliator", "expert"}
PROCEEDING_TYPES = {"hearing", "conference", "document_submission", "inspection", "site_visit", "expert_session"}
AWARD_TYPES = {"final", "partial", "interim", "consent", "default", "default_award"}
ENFORCEMENT_STATUSES = {"pending", "filed", "recognized", "enforced", "challenged", "rejected"}


class ADRDisputeResolutionService:
	"""In-memory async service for arbitration and alternative dispute resolution."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.cases: dict[str, dict[str, Any]] = {}
		self.neutrals: dict[str, dict[str, Any]] = {}
		self.proceedings: dict[str, dict[str, Any]] = {}
		self.awards: dict[str, dict[str, Any]] = {}
		self.settlements: dict[str, dict[str, Any]] = {}
		self.submissions: dict[str, dict[str, Any]] = {}
		self.enforcement_actions: dict[str, dict[str, Any]] = {}
		self._case_sequence: int = 2000
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
		self._audit_events.append({
			"id": self._id("evt-"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"details": details or {},
			"created_at": self._now(),
		})

	def _next_case_number(self, case_type: str) -> str:
		self._case_sequence += 1
		prefix = {"arbitration": "ARB", "mediation": "MED", "conciliation": "CON"}.get(case_type, "ADR")
		year = datetime.utcnow().year
		return f"{prefix}-{year}-{self._case_sequence:04d}"

	# ── Health & Describe ────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "leg_adr",
			"status": "healthy",
			"case_count": len(self.cases),
			"active_cases": sum(1 for c in self.cases.values() if c["status"] not in {"closed", "settled"}),
			"pending_awards": sum(1 for a in self.awards.values() if a["status"] == "rendered"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "leg_adr",
			"name": "ADR / Dispute Resolution",
			"domain": "legal",
			"version": "1.0.0",
			"case_types": sorted(CASE_TYPES),
			"statuses": sorted(CASE_STATUSES),
			"neutral_roles": sorted(NEUTRAL_ROLES),
			"award_types": sorted(AWARD_TYPES),
		}

	# ── Cases ────────────────────────────────────────────────────────────────

	async def create_case(
		self,
		tenant_id: str,
		title: str,
		case_type: str,
		claimant_id: str,
		respondent_id: str,
		seat: str,
		counsel_ids: list[str] | None = None,
		claim_amount: float | None = None,
		currency: str = "KES",
		governing_law: str = "",
		rules: str = "",
		description: str = "",
		filed_date: str = "",
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""File a new ADR case."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		guard_non_empty_string(claimant_id, "claimant_id")
		guard_non_empty_string(respondent_id, "respondent_id")
		if case_type not in CASE_TYPES:
			raise ValueError(f"case_type must be one of {CASE_TYPES}")
		if claimant_id == respondent_id:
			raise ValueError("claimant and respondent must be different parties")
		case_number = self._next_case_number(case_type)
		record: dict[str, Any] = {
			"id": self._id("adr-"),
			"tenant_id": tenant,
			"title": title,
			"case_number": case_number,
			"case_type": case_type,
			"claimant_id": claimant_id,
			"respondent_id": respondent_id,
			"counsel_ids": list(counsel_ids or []),
			"claim_amount": claim_amount,
			"currency": currency,
			"seat": seat,
			"governing_law": governing_law,
			"rules": rules,
			"description": description,
			"filed_date": filed_date or date.today().isoformat(),
			"status": "filed",
			"arbitrator_ids": [],
			"mediator_id": None,
			"proceeding_count": 0,
			"tags": list(tags or []),
			"metadata": dict(metadata or {}),
			"created_at": self._now(),
			"updated_at": None,
		}
		self.cases[record["id"]] = record
		self._emit(tenant, "adr_case_filed", record["id"], {
			"case_number": case_number, "type": case_type, "seat": seat,
		})
		_log.info("ADR case filed tenant=%s id=%s number=%s", tenant, record["id"], case_number)
		return deepcopy(record)

	async def get_case(self, tenant_id: str, case_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		c = self.cases.get(case_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"case {case_id} not found")
		return deepcopy(c)

	async def list_cases(
		self,
		tenant_id: str,
		case_type: str | None = None,
		status: str | None = None,
		claimant_id: str | None = None,
		respondent_id: str | None = None,
		seat: str | None = None,
	) -> list[dict[str, Any]]:
		"""List ADR cases with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.cases.values() if c["tenant_id"] == tenant]
		if case_type:
			items = [c for c in items if c["case_type"] == case_type]
		if status:
			items = [c for c in items if c["status"] == status]
		if claimant_id:
			items = [c for c in items if c["claimant_id"] == claimant_id]
		if respondent_id:
			items = [c for c in items if c["respondent_id"] == respondent_id]
		if seat:
			items = [c for c in items if c["seat"] == seat]
		return items

	async def update_case(self, tenant_id: str, case_id: str, **updates: Any) -> dict[str, Any]:
		"""Update case fields."""
		tenant = self._tenant(tenant_id)
		c = self.cases.get(case_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"case {case_id} not found")
		allowed = {"title", "description", "claim_amount", "tags", "metadata", "counsel_ids"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				c[k] = v
		c["updated_at"] = self._now()
		self._emit(tenant, "adr_case_updated", case_id, updates)
		return deepcopy(c)

	async def advance_case_status(self, tenant_id: str, case_id: str, new_status: str, notes: str = "") -> dict[str, Any]:
		"""Advance the case to the next procedural status."""
		tenant = self._tenant(tenant_id)
		c = self.cases.get(case_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"case {case_id} not found")
		if new_status not in CASE_STATUSES:
			raise ValueError(f"status must be one of {CASE_STATUSES}")
		c["status"] = new_status
		c["updated_at"] = self._now()
		if notes:
			c["status_notes"] = notes
		self._emit(tenant, "adr_case_status_advanced", case_id, {"new_status": new_status})
		return deepcopy(c)

	async def delete_case(self, tenant_id: str, case_id: str) -> dict[str, Any]:
		"""Close/archive a case."""
		tenant = self._tenant(tenant_id)
		c = self.cases.get(case_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"case {case_id} not found")
		c["status"] = "closed"
		c["updated_at"] = self._now()
		self._emit(tenant, "adr_case_closed", case_id)
		return deepcopy(c)

	# ── Neutrals (Arbitrators / Mediators) ───────────────────────────────────

	async def appoint_neutral(
		self,
		tenant_id: str,
		case_id: str,
		neutral_id: str,
		role: str,
		appointed_by: str,
		appointment_date: str,
		fee_rate: float = 0.0,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Appoint an arbitrator or mediator to a case."""
		tenant = self._tenant(tenant_id)
		c = self.cases.get(case_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"case {case_id} not found")
		if role not in NEUTRAL_ROLES:
			raise ValueError(f"role must be one of {NEUTRAL_ROLES}")
		# Check for conflicts
		existing = [
			n for n in self.neutrals.values()
			if n["case_id"] == case_id and n["neutral_id"] == neutral_id and n["status"] == "active"
		]
		if existing:
			raise ValueError(f"neutral {neutral_id} already appointed to this case")
		neutral: dict[str, Any] = {
			"id": self._id("ntrl-"),
			"tenant_id": tenant,
			"case_id": case_id,
			"neutral_id": neutral_id,
			"role": role,
			"appointed_by": appointed_by,
			"appointment_date": appointment_date,
			"fee_rate": fee_rate,
			"currency": currency,
			"status": "active",
			"challenged": False,
			"created_at": self._now(),
		}
		self.neutrals[neutral["id"]] = neutral
		if role in {"sole_arbitrator", "presiding_arbitrator", "co_arbitrator"}:
			c["arbitrator_ids"].append(neutral_id)
			c["status"] = "panel_constituted"
		elif role == "mediator":
			c["mediator_id"] = neutral_id
		c["updated_at"] = self._now()
		self._emit(tenant, "neutral_appointed", neutral["id"], {"case_id": case_id, "role": role})
		return deepcopy(neutral)

	async def get_neutral(self, tenant_id: str, neutral_record_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		n = self.neutrals.get(neutral_record_id)
		if not n or n["tenant_id"] != tenant:
			raise KeyError(f"neutral {neutral_record_id} not found")
		return deepcopy(n)

	async def list_neutrals(self, tenant_id: str, case_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(n) for n in self.neutrals.values()
			if n["tenant_id"] == tenant and n["case_id"] == case_id
		]

	async def challenge_neutral(self, tenant_id: str, neutral_record_id: str, reason: str, challenged_by: str) -> dict[str, Any]:
		"""Record a challenge to a neutral's appointment."""
		tenant = self._tenant(tenant_id)
		n = self.neutrals.get(neutral_record_id)
		if not n or n["tenant_id"] != tenant:
			raise KeyError(f"neutral {neutral_record_id} not found")
		n["challenged"] = True
		n["challenge_reason"] = reason
		n["challenged_by"] = challenged_by
		n["challenged_at"] = self._now()
		n["status"] = "challenged"
		self._emit(tenant, "neutral_challenged", neutral_record_id, {"reason": reason})
		return deepcopy(n)

	async def remove_neutral(self, tenant_id: str, neutral_record_id: str, reason: str) -> dict[str, Any]:
		"""Remove a neutral from a case."""
		tenant = self._tenant(tenant_id)
		n = self.neutrals.get(neutral_record_id)
		if not n or n["tenant_id"] != tenant:
			raise KeyError(f"neutral {neutral_record_id} not found")
		n["status"] = "removed"
		n["removal_reason"] = reason
		n["removed_at"] = self._now()
		c = self.cases.get(n["case_id"])
		if c and n["neutral_id"] in c.get("arbitrator_ids", []):
			c["arbitrator_ids"] = [aid for aid in c["arbitrator_ids"] if aid != n["neutral_id"]]
		self._emit(tenant, "neutral_removed", neutral_record_id)
		return deepcopy(n)

	# ── Proceedings ───────────────────────────────────────────────────────────

	async def create_proceeding(
		self,
		tenant_id: str,
		case_id: str,
		proceeding_type: str,
		scheduled_date: str,
		venue: str,
		description: str,
		presided_by_id: str = "",
		duration_hours: float = 0.0,
	) -> dict[str, Any]:
		"""Schedule a proceeding for a case."""
		tenant = self._tenant(tenant_id)
		c = self.cases.get(case_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"case {case_id} not found")
		if proceeding_type not in PROCEEDING_TYPES:
			raise ValueError(f"proceeding_type must be one of {PROCEEDING_TYPES}")
		proc: dict[str, Any] = {
			"id": self._id("proc-"),
			"tenant_id": tenant,
			"case_id": case_id,
			"proceeding_type": proceeding_type,
			"scheduled_date": scheduled_date,
			"actual_date": None,
			"venue": venue,
			"description": description,
			"presided_by_id": presided_by_id,
			"duration_hours": duration_hours,
			"status": "scheduled",
			"minutes_reference": None,
			"created_at": self._now(),
		}
		self.proceedings[proc["id"]] = proc
		c["proceeding_count"] = c.get("proceeding_count", 0) + 1
		if c["status"] == "panel_constituted":
			c["status"] = "preliminary_conference"
		c["updated_at"] = self._now()
		self._emit(tenant, "proceeding_scheduled", proc["id"], {"case_id": case_id, "type": proceeding_type})
		return deepcopy(proc)

	async def get_proceeding(self, tenant_id: str, proceeding_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		p = self.proceedings.get(proceeding_id)
		if not p or p["tenant_id"] != tenant:
			raise KeyError(f"proceeding {proceeding_id} not found")
		return deepcopy(p)

	async def list_proceedings(self, tenant_id: str, case_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return sorted(
			[deepcopy(p) for p in self.proceedings.values() if p["tenant_id"] == tenant and p["case_id"] == case_id],
			key=lambda p: p["scheduled_date"],
		)

	async def update_proceeding(self, tenant_id: str, proceeding_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		p = self.proceedings.get(proceeding_id)
		if not p or p["tenant_id"] != tenant:
			raise KeyError(f"proceeding {proceeding_id} not found")
		allowed = {"scheduled_date", "venue", "description", "duration_hours", "presided_by_id"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				p[k] = v
		self._emit(tenant, "proceeding_updated", proceeding_id, updates)
		return deepcopy(p)

	async def conclude_proceeding(
		self,
		tenant_id: str,
		proceeding_id: str,
		actual_date: str,
		minutes_reference: str = "",
		duration_hours: float = 0.0,
	) -> dict[str, Any]:
		"""Mark a proceeding as completed."""
		tenant = self._tenant(tenant_id)
		p = self.proceedings.get(proceeding_id)
		if not p or p["tenant_id"] != tenant:
			raise KeyError(f"proceeding {proceeding_id} not found")
		p["status"] = "completed"
		p["actual_date"] = actual_date
		p["minutes_reference"] = minutes_reference
		if duration_hours:
			p["duration_hours"] = duration_hours
		self._emit(tenant, "proceeding_completed", proceeding_id)
		return deepcopy(p)

	async def delete_proceeding(self, tenant_id: str, proceeding_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		p = self.proceedings.get(proceeding_id)
		if not p or p["tenant_id"] != tenant:
			raise KeyError(f"proceeding {proceeding_id} not found")
		p["status"] = "cancelled"
		self._emit(tenant, "proceeding_cancelled", proceeding_id)
		return deepcopy(p)

	# ── Awards ────────────────────────────────────────────────────────────────

	async def create_award(
		self,
		tenant_id: str,
		case_id: str,
		award_type: str,
		award_date: str,
		awarded_to_id: str,
		summary: str,
		award_amount: float | None = None,
		currency: str = "KES",
		interest_rate: float = 0.0,
		costs_awarded: float = 0.0,
		full_text_reference: str = "",
	) -> dict[str, Any]:
		"""Record an arbitral award."""
		tenant = self._tenant(tenant_id)
		c = self.cases.get(case_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"case {case_id} not found")
		if award_type not in AWARD_TYPES:
			raise ValueError(f"award_type must be one of {AWARD_TYPES}")
		guard_non_empty_string(summary, "summary")
		award: dict[str, Any] = {
			"id": self._id("awd-"),
			"tenant_id": tenant,
			"case_id": case_id,
			"award_type": award_type,
			"award_date": award_date,
			"awarded_to_id": awarded_to_id,
			"award_amount": award_amount,
			"currency": currency,
			"interest_rate": interest_rate,
			"costs_awarded": costs_awarded,
			"summary": summary,
			"full_text_reference": full_text_reference,
			"status": "rendered",
			"enforcement_status": None,
			"created_at": self._now(),
		}
		self.awards[award["id"]] = award
		if award_type == "final":
			c["status"] = "award_rendered"
			c["updated_at"] = self._now()
		self._emit(tenant, "award_rendered", award["id"], {"case_id": case_id, "type": award_type})
		_log.info("arbitral award rendered tenant=%s id=%s case=%s", tenant, award["id"], case_id)
		return deepcopy(award)

	async def get_award(self, tenant_id: str, award_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		a = self.awards.get(award_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"award {award_id} not found")
		return deepcopy(a)

	async def list_awards(self, tenant_id: str, case_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.awards.values() if a["tenant_id"] == tenant]
		if case_id:
			items = [a for a in items if a["case_id"] == case_id]
		return items

	async def update_award(self, tenant_id: str, award_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		a = self.awards.get(award_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"award {award_id} not found")
		allowed = {"summary", "full_text_reference"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				a[k] = v
		self._emit(tenant, "award_updated", award_id, updates)
		return deepcopy(a)

	async def challenge_award(self, tenant_id: str, award_id: str, basis: str, filed_by: str) -> dict[str, Any]:
		"""File a challenge/set-aside application against an award."""
		tenant = self._tenant(tenant_id)
		a = self.awards.get(award_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"award {award_id} not found")
		a["status"] = "challenged"
		a["challenge_basis"] = basis
		a["challenged_by"] = filed_by
		a["challenged_at"] = self._now()
		self._emit(tenant, "award_challenged", award_id, {"basis": basis})
		return deepcopy(a)

	async def delete_award(self, tenant_id: str, award_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		a = self.awards.get(award_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"award {award_id} not found")
		a["status"] = "set_aside"
		self._emit(tenant, "award_set_aside", award_id)
		return deepcopy(a)

	# ── Settlements ───────────────────────────────────────────────────────────

	async def create_settlement(
		self,
		tenant_id: str,
		case_id: str,
		settlement_date: str,
		settlement_amount: float,
		terms_summary: str,
		signed_by_claimant_id: str,
		signed_by_respondent_id: str,
		currency: str = "KES",
		confidentiality_clause: bool = True,
	) -> dict[str, Any]:
		"""Record a negotiated settlement."""
		tenant = self._tenant(tenant_id)
		c = self.cases.get(case_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"case {case_id} not found")
		guard_non_empty_string(terms_summary, "terms_summary")
		settlement: dict[str, Any] = {
			"id": self._id("stl-"),
			"tenant_id": tenant,
			"case_id": case_id,
			"settlement_date": settlement_date,
			"settlement_amount": settlement_amount,
			"currency": currency,
			"terms_summary": terms_summary,
			"signed_by_claimant_id": signed_by_claimant_id,
			"signed_by_respondent_id": signed_by_respondent_id,
			"confidentiality_clause": confidentiality_clause,
			"status": "executed",
			"created_at": self._now(),
		}
		self.settlements[settlement["id"]] = settlement
		c["status"] = "settled"
		c["updated_at"] = self._now()
		self._emit(tenant, "case_settled", settlement["id"], {"case_id": case_id, "amount": settlement_amount})
		return deepcopy(settlement)

	async def get_settlement(self, tenant_id: str, settlement_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		s = self.settlements.get(settlement_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"settlement {settlement_id} not found")
		return deepcopy(s)

	async def list_settlements(self, tenant_id: str, case_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.settlements.values() if s["tenant_id"] == tenant]
		if case_id:
			items = [s for s in items if s["case_id"] == case_id]
		return items

	async def update_settlement(self, tenant_id: str, settlement_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		s = self.settlements.get(settlement_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"settlement {settlement_id} not found")
		allowed = {"terms_summary", "settlement_amount"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				s[k] = v
		self._emit(tenant, "settlement_updated", settlement_id, updates)
		return deepcopy(s)

	async def delete_settlement(self, tenant_id: str, settlement_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		s = self.settlements.get(settlement_id)
		if not s or s["tenant_id"] != tenant:
			raise KeyError(f"settlement {settlement_id} not found")
		s["status"] = "voided"
		self._emit(tenant, "settlement_voided", settlement_id)
		return deepcopy(s)

	# ── Enforcement ───────────────────────────────────────────────────────────

	async def file_enforcement_action(
		self,
		tenant_id: str,
		award_id: str,
		jurisdiction: str,
		filed_by_id: str,
		filing_date: str,
		court_reference: str = "",
	) -> dict[str, Any]:
		"""File an enforcement action for an award."""
		tenant = self._tenant(tenant_id)
		award = self.awards.get(award_id)
		if not award or award["tenant_id"] != tenant:
			raise KeyError(f"award {award_id} not found")
		action: dict[str, Any] = {
			"id": self._id("enf-"),
			"tenant_id": tenant,
			"award_id": award_id,
			"jurisdiction": jurisdiction,
			"filed_by_id": filed_by_id,
			"filing_date": filing_date,
			"court_reference": court_reference,
			"status": "filed",
			"created_at": self._now(),
		}
		self.enforcement_actions[action["id"]] = action
		award["enforcement_status"] = "filed"
		self._emit(tenant, "enforcement_action_filed", action["id"], {"award_id": award_id})
		return deepcopy(action)

	async def update_enforcement_status(
		self,
		tenant_id: str,
		enforcement_id: str,
		new_status: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Update the enforcement status."""
		tenant = self._tenant(tenant_id)
		action = self.enforcement_actions.get(enforcement_id)
		if not action or action["tenant_id"] != tenant:
			raise KeyError(f"enforcement action {enforcement_id} not found")
		if new_status not in ENFORCEMENT_STATUSES:
			raise ValueError(f"status must be one of {ENFORCEMENT_STATUSES}")
		action["status"] = new_status
		if notes:
			action["notes"] = notes
		action["updated_at"] = self._now()
		award = self.awards.get(action["award_id"])
		if award:
			award["enforcement_status"] = new_status
		self._emit(tenant, "enforcement_status_updated", enforcement_id, {"status": new_status})
		return deepcopy(action)

	# ── Analytics ────────────────────────────────────────────────────────────

	async def adr_dashboard(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		cases = [c for c in self.cases.values() if c["tenant_id"] == tenant]
		by_type: dict[str, int] = {}
		by_status: dict[str, int] = {}
		total_claims = 0.0
		for c in cases:
			by_type[c["case_type"]] = by_type.get(c["case_type"], 0) + 1
			by_status[c["status"]] = by_status.get(c["status"], 0) + 1
			if c.get("claim_amount"):
				total_claims += c["claim_amount"]
		awards = [a for a in self.awards.values() if a["tenant_id"] == tenant]
		total_awarded = sum(
			a.get("award_amount", 0) or 0 for a in awards if a["status"] in {"rendered", "enforced"}
		)
		return {
			"tenant_id": tenant,
			"total_cases": len(cases),
			"active_cases": sum(1 for c in cases if c["status"] not in {"closed", "settled"}),
			"by_type": by_type,
			"by_status": by_status,
			"total_claim_value": total_claims,
			"total_awarded_value": total_awarded,
			"settlements": len([s for s in self.settlements.values() if s["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	async def get_audit_events(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		return events[-limit:]

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

