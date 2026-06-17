"""Entity & Corporate Secretary — async service layer."""
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

ENTITY_TYPES = {"limited_company", "llp", "branch", "holding", "subsidiary", "ngo", "trust", "partnership"}
DIRECTOR_ROLES = {"director", "chairperson", "secretary", "ceo", "cfo", "md", "independent_director"}
FILING_TYPES = {
	"annual_return", "change_of_directors", "change_of_address", "share_allotment",
	"share_transfer", "special_resolution", "charges_registration", "winding_up",
}
SHARE_CLASSES = {"ordinary", "preference", "redeemable", "deferred", "non_voting"}


class EntityCorporateSecretaryService:
	"""In-memory async service for entity registry and corporate secretarial work."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.entities: dict[str, dict[str, Any]] = {}
		self.directors: dict[str, dict[str, Any]] = {}
		self.shareholders: dict[str, dict[str, Any]] = {}
		self.filings: dict[str, dict[str, Any]] = {}
		self.resolutions: dict[str, dict[str, Any]] = {}
		self.meetings: dict[str, dict[str, Any]] = {}
		self.charges: dict[str, dict[str, Any]] = {}
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

	# ── Health ───────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "leg_ent",
			"status": "healthy",
			"entity_count": len(self.entities),
			"pending_filings": sum(1 for f in self.filings.values() if f["status"] == "pending"),
			"overdue_filings": sum(
				1 for f in self.filings.values()
				if f["status"] == "pending" and f["due_date"] < date.today().isoformat()
			),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "leg_ent",
			"name": "Entity & Corporate Secretary",
			"domain": "legal",
			"version": "1.0.0",
			"entity_types": sorted(ENTITY_TYPES),
			"director_roles": sorted(DIRECTOR_ROLES),
			"filing_types": sorted(FILING_TYPES),
		}

	# ── Entities ─────────────────────────────────────────────────────────────

	async def create_entity(
		self,
		tenant_id: str,
		legal_name: str,
		entity_type: str,
		registration_number: str,
		jurisdiction: str,
		incorporation_date: str,
		registered_address: str,
		business_address: str = "",
		tax_pin: str = "",
		vat_number: str = "",
		financial_year_end: str = "12-31",
		description: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register a new legal entity."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(legal_name, "legal_name")
		guard_non_empty_string(registration_number, "registration_number")
		if entity_type not in ENTITY_TYPES:
			raise ValueError(f"entity_type must be one of {ENTITY_TYPES}")
		# Check for duplicate registration number
		existing = next(
			(e for e in self.entities.values()
			 if e["tenant_id"] == tenant and e["registration_number"] == registration_number),
			None,
		)
		if existing:
			raise ValueError(f"entity with registration_number {registration_number} already exists")
		record: dict[str, Any] = {
			"id": self._id("ent-"),
			"tenant_id": tenant,
			"legal_name": legal_name,
			"entity_type": entity_type,
			"registration_number": registration_number,
			"jurisdiction": jurisdiction,
			"incorporation_date": incorporation_date,
			"registered_address": registered_address,
			"business_address": business_address,
			"tax_pin": tax_pin,
			"vat_number": vat_number,
			"financial_year_end": financial_year_end,
			"description": description,
			"status": "active",
			"director_count": 0,
			"shareholder_count": 0,
			"metadata": dict(metadata or {}),
			"created_at": self._now(),
			"updated_at": None,
		}
		self.entities[record["id"]] = record
		self._emit(tenant, "entity_created", record["id"], {"legal_name": legal_name, "type": entity_type})
		_log.info("entity created tenant=%s id=%s reg=%s", tenant, record["id"], registration_number)
		return deepcopy(record)

	async def get_entity(self, tenant_id: str, entity_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ent = self.entities.get(entity_id)
		if not ent or ent["tenant_id"] != tenant:
			raise KeyError(f"entity {entity_id} not found")
		return deepcopy(ent)

	async def list_entities(
		self,
		tenant_id: str,
		entity_type: str | None = None,
		jurisdiction: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.entities.values() if e["tenant_id"] == tenant]
		if entity_type:
			items = [e for e in items if e["entity_type"] == entity_type]
		if jurisdiction:
			items = [e for e in items if e["jurisdiction"] == jurisdiction]
		if status:
			items = [e for e in items if e["status"] == status]
		return items

	async def update_entity(self, tenant_id: str, entity_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		ent = self.entities.get(entity_id)
		if not ent or ent["tenant_id"] != tenant:
			raise KeyError(f"entity {entity_id} not found")
		allowed = {
			"legal_name", "registered_address", "business_address", "tax_pin",
			"vat_number", "financial_year_end", "description", "metadata",
		}
		for k, v in updates.items():
			if k in allowed and v is not None:
				ent[k] = v
		ent["updated_at"] = self._now()
		self._emit(tenant, "entity_updated", entity_id, updates)
		return deepcopy(ent)

	async def deactivate_entity(self, tenant_id: str, entity_id: str, reason: str) -> dict[str, Any]:
		"""Deactivate (strike off / dissolve) an entity."""
		tenant = self._tenant(tenant_id)
		ent = self.entities.get(entity_id)
		if not ent or ent["tenant_id"] != tenant:
			raise KeyError(f"entity {entity_id} not found")
		ent["status"] = "inactive"
		ent["deactivation_reason"] = reason
		ent["deactivated_at"] = self._now()
		ent["updated_at"] = self._now()
		self._emit(tenant, "entity_deactivated", entity_id, {"reason": reason})
		return deepcopy(ent)

	async def delete_entity(self, tenant_id: str, entity_id: str) -> dict[str, Any]:
		"""Archive an entity record."""
		return await self.deactivate_entity(tenant_id, entity_id, reason="archived")

	# ── Directors ─────────────────────────────────────────────────────────────

	async def appoint_director(
		self,
		tenant_id: str,
		entity_id: str,
		full_name: str,
		id_number: str,
		nationality: str,
		appointment_date: str,
		role: str = "director",
		address: str = "",
		email: str = "",
	) -> dict[str, Any]:
		"""Appoint a director to an entity."""
		tenant = self._tenant(tenant_id)
		ent = self.entities.get(entity_id)
		if not ent or ent["tenant_id"] != tenant:
			raise KeyError(f"entity {entity_id} not found")
		if role not in DIRECTOR_ROLES:
			raise ValueError(f"role must be one of {DIRECTOR_ROLES}")
		director: dict[str, Any] = {
			"id": self._id("dir-"),
			"tenant_id": tenant,
			"entity_id": entity_id,
			"full_name": full_name,
			"id_number": id_number,
			"nationality": nationality,
			"appointment_date": appointment_date,
			"cessation_date": None,
			"role": role,
			"address": address,
			"email": email,
			"status": "active",
			"created_at": self._now(),
		}
		self.directors[director["id"]] = director
		ent["director_count"] = len([
			d for d in self.directors.values()
			if d["entity_id"] == entity_id and d["status"] == "active"
		])
		self._emit(tenant, "director_appointed", director["id"], {"entity_id": entity_id, "role": role})
		return deepcopy(director)

	async def get_director(self, tenant_id: str, director_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		d = self.directors.get(director_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"director {director_id} not found")
		return deepcopy(d)

	async def list_directors(self, tenant_id: str, entity_id: str, active_only: bool = True) -> list[dict[str, Any]]:
		"""List directors of an entity."""
		tenant = self._tenant(tenant_id)
		items = [
			deepcopy(d) for d in self.directors.values()
			if d["tenant_id"] == tenant and d["entity_id"] == entity_id
		]
		if active_only:
			items = [d for d in items if d["status"] == "active"]
		return items

	async def update_director(self, tenant_id: str, director_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		d = self.directors.get(director_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"director {director_id} not found")
		allowed = {"address", "email", "role"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				d[k] = v
		self._emit(tenant, "director_updated", director_id, updates)
		return deepcopy(d)

	async def remove_director(self, tenant_id: str, director_id: str, cessation_date: str) -> dict[str, Any]:
		"""Record cessation of a director."""
		tenant = self._tenant(tenant_id)
		d = self.directors.get(director_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"director {director_id} not found")
		entity_id = d["entity_id"]
		active_count = sum(
			1 for od in self.directors.values()
			if od["entity_id"] == entity_id and od["status"] == "active" and od["id"] != director_id
		)
		if active_count < 1:
			raise ValueError("entity must have at least one director")
		d["status"] = "ceased"
		d["cessation_date"] = cessation_date
		ent = self.entities.get(entity_id)
		if ent:
			ent["director_count"] = active_count
		self._emit(tenant, "director_removed", director_id, {"cessation_date": cessation_date})
		return deepcopy(d)

	# ── Shareholders / Share Register ────────────────────────────────────────

	async def register_shareholder(
		self,
		tenant_id: str,
		entity_id: str,
		full_name: str,
		id_number: str,
		share_class: str,
		shares_held: int,
		nominal_value: float,
		consideration_paid: float,
		nationality: str = "",
	) -> dict[str, Any]:
		"""Add a shareholder to the share register."""
		tenant = self._tenant(tenant_id)
		ent = self.entities.get(entity_id)
		if not ent or ent["tenant_id"] != tenant:
			raise KeyError(f"entity {entity_id} not found")
		if share_class not in SHARE_CLASSES:
			raise ValueError(f"share_class must be one of {SHARE_CLASSES}")
		if shares_held <= 0:
			raise ValueError("shares_held must be positive")
		sh: dict[str, Any] = {
			"id": self._id("shr-"),
			"tenant_id": tenant,
			"entity_id": entity_id,
			"full_name": full_name,
			"id_number": id_number,
			"share_class": share_class,
			"shares_held": shares_held,
			"nominal_value": nominal_value,
			"consideration_paid": consideration_paid,
			"nationality": nationality,
			"status": "active",
			"created_at": self._now(),
		}
		self.shareholders[sh["id"]] = sh
		ent["shareholder_count"] = len([
			s for s in self.shareholders.values()
			if s["entity_id"] == entity_id and s["status"] == "active"
		])
		self._emit(tenant, "shareholder_registered", sh["id"], {"entity_id": entity_id, "shares": shares_held})
		return deepcopy(sh)

	async def get_shareholder(self, tenant_id: str, shareholder_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		sh = self.shareholders.get(shareholder_id)
		if not sh or sh["tenant_id"] != tenant:
			raise KeyError(f"shareholder {shareholder_id} not found")
		return deepcopy(sh)

	async def list_shareholders(self, tenant_id: str, entity_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(s) for s in self.shareholders.values()
			if s["tenant_id"] == tenant and s["entity_id"] == entity_id
		]

	async def update_shareholder(self, tenant_id: str, shareholder_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		sh = self.shareholders.get(shareholder_id)
		if not sh or sh["tenant_id"] != tenant:
			raise KeyError(f"shareholder {shareholder_id} not found")
		allowed = {"shares_held", "consideration_paid"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				sh[k] = v
		self._emit(tenant, "shareholder_updated", shareholder_id, updates)
		return deepcopy(sh)

	async def transfer_shares(
		self,
		tenant_id: str,
		from_shareholder_id: str,
		to_full_name: str,
		to_id_number: str,
		shares_transferred: int,
		transfer_date: str,
		consideration: float,
	) -> dict[str, Any]:
		"""Record a share transfer."""
		tenant = self._tenant(tenant_id)
		from_sh = self.shareholders.get(from_shareholder_id)
		if not from_sh or from_sh["tenant_id"] != tenant:
			raise KeyError(f"shareholder {from_shareholder_id} not found")
		if from_sh["shares_held"] < shares_transferred:
			raise ValueError("insufficient shares for transfer")
		from_sh["shares_held"] -= shares_transferred
		entity_id = from_sh["entity_id"]
		# Register new/existing transferee
		new_sh = await self.register_shareholder(
			tenant_id, entity_id, to_full_name, to_id_number,
			from_sh["share_class"], shares_transferred,
			from_sh["nominal_value"], consideration,
		)
		self._emit(tenant, "shares_transferred", from_shareholder_id, {
			"to": to_id_number, "shares": shares_transferred, "date": transfer_date,
		})
		return {"from": deepcopy(from_sh), "to": new_sh, "transfer_date": transfer_date}

	# ── Statutory Filings ────────────────────────────────────────────────────

	async def create_filing(
		self,
		tenant_id: str,
		entity_id: str,
		filing_type: str,
		due_date: str,
		filing_period: str = "",
		filed_by_id: str = "",
		reference_number: str = "",
		notes: str = "",
	) -> dict[str, Any]:
		"""Schedule a statutory filing."""
		tenant = self._tenant(tenant_id)
		ent = self.entities.get(entity_id)
		if not ent or ent["tenant_id"] != tenant:
			raise KeyError(f"entity {entity_id} not found")
		if filing_type not in FILING_TYPES:
			raise ValueError(f"filing_type must be one of {FILING_TYPES}")
		filing: dict[str, Any] = {
			"id": self._id("fil-"),
			"tenant_id": tenant,
			"entity_id": entity_id,
			"filing_type": filing_type,
			"due_date": due_date,
			"filing_period": filing_period,
			"filed_by_id": filed_by_id,
			"reference_number": reference_number,
			"notes": notes,
			"status": "pending",
			"filed_at": None,
			"created_at": self._now(),
		}
		self.filings[filing["id"]] = filing
		self._emit(tenant, "filing_scheduled", filing["id"], {"entity_id": entity_id, "type": filing_type})
		return deepcopy(filing)

	async def get_filing(self, tenant_id: str, filing_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		f = self.filings.get(filing_id)
		if not f or f["tenant_id"] != tenant:
			raise KeyError(f"filing {filing_id} not found")
		return deepcopy(f)

	async def list_filings(
		self,
		tenant_id: str,
		entity_id: str | None = None,
		filing_type: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(f) for f in self.filings.values() if f["tenant_id"] == tenant]
		if entity_id:
			items = [f for f in items if f["entity_id"] == entity_id]
		if filing_type:
			items = [f for f in items if f["filing_type"] == filing_type]
		if status:
			items = [f for f in items if f["status"] == status]
		return sorted(items, key=lambda f: f["due_date"])

	async def update_filing(self, tenant_id: str, filing_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		f = self.filings.get(filing_id)
		if not f or f["tenant_id"] != tenant:
			raise KeyError(f"filing {filing_id} not found")
		allowed = {"due_date", "notes", "filed_by_id"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				f[k] = v
		self._emit(tenant, "filing_updated", filing_id, updates)
		return deepcopy(f)

	async def complete_filing(
		self,
		tenant_id: str,
		filing_id: str,
		reference_number: str,
		filed_by_id: str,
	) -> dict[str, Any]:
		"""Mark a filing as completed."""
		tenant = self._tenant(tenant_id)
		f = self.filings.get(filing_id)
		if not f or f["tenant_id"] != tenant:
			raise KeyError(f"filing {filing_id} not found")
		f["status"] = "filed"
		f["reference_number"] = reference_number
		f["filed_by_id"] = filed_by_id
		f["filed_at"] = self._now()
		self._emit(tenant, "filing_completed", filing_id, {"reference": reference_number})
		return deepcopy(f)

	async def delete_filing(self, tenant_id: str, filing_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		f = self.filings.get(filing_id)
		if not f or f["tenant_id"] != tenant:
			raise KeyError(f"filing {filing_id} not found")
		f["status"] = "cancelled"
		self._emit(tenant, "filing_cancelled", filing_id)
		return deepcopy(f)

	# ── Board Resolutions ────────────────────────────────────────────────────

	async def create_board_resolution(
		self,
		tenant_id: str,
		entity_id: str,
		resolution_number: str,
		resolution_date: str,
		resolution_type: str,
		subject: str,
		body: str,
		passed_by: list[str] | None = None,
	) -> dict[str, Any]:
		"""Record a board or shareholder resolution."""
		tenant = self._tenant(tenant_id)
		ent = self.entities.get(entity_id)
		if not ent or ent["tenant_id"] != tenant:
			raise KeyError(f"entity {entity_id} not found")
		resolution: dict[str, Any] = {
			"id": self._id("res-"),
			"tenant_id": tenant,
			"entity_id": entity_id,
			"resolution_number": resolution_number,
			"resolution_date": resolution_date,
			"resolution_type": resolution_type,
			"subject": subject,
			"body": body,
			"passed_by": list(passed_by or []),
			"status": "passed",
			"created_at": self._now(),
		}
		self.resolutions[resolution["id"]] = resolution
		self._emit(tenant, "resolution_created", resolution["id"], {"entity_id": entity_id, "type": resolution_type})
		return deepcopy(resolution)

	async def get_board_resolution(self, tenant_id: str, resolution_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		r = self.resolutions.get(resolution_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"resolution {resolution_id} not found")
		return deepcopy(r)

	async def list_board_resolutions(self, tenant_id: str, entity_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(r) for r in self.resolutions.values()
			if r["tenant_id"] == tenant and r["entity_id"] == entity_id
		]

	# ── Analytics ────────────────────────────────────────────────────────────

	async def corporate_dashboard(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		entities = [e for e in self.entities.values() if e["tenant_id"] == tenant]
		filings = [f for f in self.filings.values() if f["tenant_id"] == tenant]
		today = date.today().isoformat()
		by_type: dict[str, int] = {}
		for e in entities:
			by_type[e["entity_type"]] = by_type.get(e["entity_type"], 0) + 1
		return {
			"tenant_id": tenant,
			"total_entities": len(entities),
			"active_entities": sum(1 for e in entities if e["status"] == "active"),
			"by_type": by_type,
			"pending_filings": sum(1 for f in filings if f["status"] == "pending"),
			"overdue_filings": sum(1 for f in filings if f["status"] == "pending" and f["due_date"] < today),
			"total_directors": len([d for d in self.directors.values() if d["tenant_id"] == tenant and d["status"] == "active"]),
			"total_shareholders": len([s for s in self.shareholders.values() if s["tenant_id"] == tenant and s["status"] == "active"]),
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

