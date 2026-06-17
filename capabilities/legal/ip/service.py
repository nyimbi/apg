"""Intellectual Property Registry — async service layer."""
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

ASSET_TYPES = {"patent", "trademark", "copyright", "trade_secret", "design", "domain", "plant_variety"}
ASSET_STATUSES = {"pending", "registered", "lapsed", "abandoned", "licensed", "assigned", "expired"}
LICENSE_TYPES = {"exclusive", "non_exclusive", "sole", "sublicense", "compulsory"}
ROYALTY_BASES = {"revenue", "unit", "fixed", "tiered"}


class IntellectualPropertyService:
	"""In-memory async service for IP portfolio management."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.assets: dict[str, dict[str, Any]] = {}
		self.renewals: dict[str, dict[str, Any]] = {}
		self.licenses: dict[str, dict[str, Any]] = {}
		self.royalties: dict[str, dict[str, Any]] = {}
		self.oppositions: dict[str, dict[str, Any]] = {}
		self.assignments: dict[str, dict[str, Any]] = {}
		self.watches: dict[str, dict[str, Any]] = {}
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

	def _compute_renewal_due(self, asset: dict[str, Any]) -> str | None:
		"""Compute renewal due date as 30 days before expiry."""
		if not asset.get("expiry_date"):
			return None
		exp = asset["expiry_date"]
		# Simplistic: subtract ~30 days via string
		try:
			d = date.fromisoformat(exp)
			renewal = d.replace(day=max(1, d.day - 30)) if d.day > 30 else d.replace(month=max(1, d.month - 1))
			return renewal.isoformat()
		except Exception:
			return exp

	# ── Health & Describe ────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		today = date.today().isoformat()
		return {
			"service": "leg_ip",
			"status": "healthy",
			"asset_count": len(self.assets),
			"active_licenses": sum(1 for l in self.licenses.values() if l["status"] == "active"),
			"expiring_soon": sum(
				1 for a in self.assets.values()
				if a.get("expiry_date") and a["expiry_date"] >= today and a["status"] in {"registered"}
			),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "leg_ip",
			"name": "Intellectual Property Registry",
			"domain": "legal",
			"version": "1.0.0",
			"asset_types": sorted(ASSET_TYPES),
			"statuses": sorted(ASSET_STATUSES),
			"license_types": sorted(LICENSE_TYPES),
		}

	# ── IP Assets ────────────────────────────────────────────────────────────

	async def create_asset(
		self,
		tenant_id: str,
		title: str,
		asset_type: str,
		owner_id: str,
		jurisdiction: str,
		registration_number: str = "",
		application_number: str = "",
		filing_date: str = "",
		registration_date: str = "",
		expiry_date: str | None = None,
		classes: list[str] | None = None,
		description: str = "",
		inventors: list[str] | None = None,
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register an IP asset."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		if asset_type not in ASSET_TYPES:
			raise ValueError(f"asset_type must be one of {ASSET_TYPES}")
		record: dict[str, Any] = {
			"id": self._id("ip-"),
			"tenant_id": tenant,
			"title": title,
			"asset_type": asset_type,
			"owner_id": owner_id,
			"registration_number": registration_number,
			"application_number": application_number,
			"filing_date": filing_date,
			"registration_date": registration_date,
			"expiry_date": expiry_date,
			"jurisdiction": jurisdiction,
			"classes": list(classes or []),
			"description": description,
			"inventors": list(inventors or []),
			"status": "registered" if registration_number else "pending",
			"renewal_due_date": None,
			"license_count": 0,
			"tags": list(tags or []),
			"metadata": dict(metadata or {}),
			"created_at": self._now(),
			"updated_at": None,
		}
		record["renewal_due_date"] = self._compute_renewal_due(record)
		self.assets[record["id"]] = record
		self._emit(tenant, "ip_asset_created", record["id"], {"title": title, "type": asset_type})
		_log.info("IP asset created tenant=%s id=%s type=%s", tenant, record["id"], asset_type)
		return deepcopy(record)

	async def get_asset(self, tenant_id: str, asset_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		a = self.assets.get(asset_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"asset {asset_id} not found")
		return deepcopy(a)

	async def list_assets(
		self,
		tenant_id: str,
		asset_type: str | None = None,
		owner_id: str | None = None,
		jurisdiction: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.assets.values() if a["tenant_id"] == tenant]
		if asset_type:
			items = [a for a in items if a["asset_type"] == asset_type]
		if owner_id:
			items = [a for a in items if a["owner_id"] == owner_id]
		if jurisdiction:
			items = [a for a in items if a["jurisdiction"] == jurisdiction]
		if status:
			items = [a for a in items if a["status"] == status]
		return items

	async def update_asset(self, tenant_id: str, asset_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		a = self.assets.get(asset_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"asset {asset_id} not found")
		allowed = {
			"title", "description", "registration_number", "registration_date",
			"expiry_date", "classes", "tags", "metadata",
		}
		for k, v in updates.items():
			if k in allowed and v is not None:
				a[k] = v
		a["renewal_due_date"] = self._compute_renewal_due(a)
		a["updated_at"] = self._now()
		self._emit(tenant, "ip_asset_updated", asset_id, updates)
		return deepcopy(a)

	async def register_asset(
		self,
		tenant_id: str,
		asset_id: str,
		registration_number: str,
		registration_date: str,
		expiry_date: str | None = None,
	) -> dict[str, Any]:
		"""Record official registration of a pending asset."""
		tenant = self._tenant(tenant_id)
		a = self.assets.get(asset_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"asset {asset_id} not found")
		a["registration_number"] = registration_number
		a["registration_date"] = registration_date
		if expiry_date:
			a["expiry_date"] = expiry_date
		a["status"] = "registered"
		a["renewal_due_date"] = self._compute_renewal_due(a)
		a["updated_at"] = self._now()
		self._emit(tenant, "ip_asset_registered", asset_id, {"reg_number": registration_number})
		return deepcopy(a)

	async def abandon_asset(self, tenant_id: str, asset_id: str, reason: str) -> dict[str, Any]:
		"""Mark an asset as abandoned."""
		tenant = self._tenant(tenant_id)
		a = self.assets.get(asset_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"asset {asset_id} not found")
		a["status"] = "abandoned"
		a["abandon_reason"] = reason
		a["abandoned_at"] = self._now()
		a["updated_at"] = self._now()
		self._emit(tenant, "ip_asset_abandoned", asset_id, {"reason": reason})
		return deepcopy(a)

	async def delete_asset(self, tenant_id: str, asset_id: str) -> dict[str, Any]:
		return await self.abandon_asset(tenant_id, asset_id, reason="archived")

	# ── Renewals ─────────────────────────────────────────────────────────────

	async def create_renewal(
		self,
		tenant_id: str,
		asset_id: str,
		renewal_date: str,
		renewal_fee: float,
		submitted_by_id: str,
		currency: str = "KES",
		official_fee: float = 0.0,
		agent_fee: float = 0.0,
		reference_number: str = "",
		notes: str = "",
	) -> dict[str, Any]:
		"""Record an IP renewal filing."""
		tenant = self._tenant(tenant_id)
		a = self.assets.get(asset_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"asset {asset_id} not found")
		renewal: dict[str, Any] = {
			"id": self._id("rnw-"),
			"tenant_id": tenant,
			"asset_id": asset_id,
			"renewal_date": renewal_date,
			"renewal_fee": renewal_fee,
			"currency": currency,
			"official_fee": official_fee,
			"agent_fee": agent_fee,
			"submitted_by_id": submitted_by_id,
			"reference_number": reference_number,
			"notes": notes,
			"status": "submitted",
			"new_expiry_date": None,
			"created_at": self._now(),
		}
		self.renewals[renewal["id"]] = renewal
		self._emit(tenant, "ip_renewal_created", renewal["id"], {"asset_id": asset_id})
		return deepcopy(renewal)

	async def get_renewal(self, tenant_id: str, renewal_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		r = self.renewals.get(renewal_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"renewal {renewal_id} not found")
		return deepcopy(r)

	async def list_renewals(self, tenant_id: str, asset_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.renewals.values() if r["tenant_id"] == tenant]
		if asset_id:
			items = [r for r in items if r["asset_id"] == asset_id]
		return items

	async def confirm_renewal(
		self,
		tenant_id: str,
		renewal_id: str,
		new_expiry_date: str,
		confirmed_by_id: str,
	) -> dict[str, Any]:
		"""Confirm a renewal was accepted by the registry."""
		tenant = self._tenant(tenant_id)
		r = self.renewals.get(renewal_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"renewal {renewal_id} not found")
		r["status"] = "confirmed"
		r["new_expiry_date"] = new_expiry_date
		r["confirmed_by_id"] = confirmed_by_id
		r["confirmed_at"] = self._now()
		# Update asset expiry
		a = self.assets.get(r["asset_id"])
		if a:
			a["expiry_date"] = new_expiry_date
			a["renewal_due_date"] = self._compute_renewal_due(a)
			a["updated_at"] = self._now()
		self._emit(tenant, "ip_renewal_confirmed", renewal_id)
		return deepcopy(r)

	async def delete_renewal(self, tenant_id: str, renewal_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		r = self.renewals.get(renewal_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"renewal {renewal_id} not found")
		r["status"] = "cancelled"
		self._emit(tenant, "ip_renewal_cancelled", renewal_id)
		return deepcopy(r)

	# ── Licenses ─────────────────────────────────────────────────────────────

	async def create_license(
		self,
		tenant_id: str,
		asset_id: str,
		licensee_id: str,
		license_type: str,
		territory: str,
		start_date: str,
		end_date: str | None = None,
		royalty_rate: float = 0.0,
		royalty_base: str = "revenue",
		upfront_fee: float = 0.0,
		currency: str = "KES",
		restrictions: str = "",
	) -> dict[str, Any]:
		"""Grant a license on an IP asset."""
		tenant = self._tenant(tenant_id)
		a = self.assets.get(asset_id)
		if not a or a["tenant_id"] != tenant:
			raise KeyError(f"asset {asset_id} not found")
		if a["status"] not in {"registered"}:
			raise ValueError("can only license registered assets")
		if license_type not in LICENSE_TYPES:
			raise ValueError(f"license_type must be one of {LICENSE_TYPES}")
		if license_type == "exclusive":
			existing_exclusive = [
				l for l in self.licenses.values()
				if l["asset_id"] == asset_id and l["license_type"] == "exclusive" and l["status"] == "active"
			]
			if existing_exclusive:
				raise ValueError("asset already has an exclusive license")
		lic: dict[str, Any] = {
			"id": self._id("lic-"),
			"tenant_id": tenant,
			"asset_id": asset_id,
			"licensee_id": licensee_id,
			"license_type": license_type,
			"territory": territory,
			"start_date": start_date,
			"end_date": end_date,
			"royalty_rate": royalty_rate,
			"royalty_base": royalty_base,
			"upfront_fee": upfront_fee,
			"currency": currency,
			"restrictions": restrictions,
			"status": "active",
			"created_at": self._now(),
		}
		self.licenses[lic["id"]] = lic
		a["license_count"] = len([
			l for l in self.licenses.values()
			if l["asset_id"] == asset_id and l["status"] == "active"
		])
		self._emit(tenant, "ip_license_created", lic["id"], {"asset_id": asset_id, "type": license_type})
		return deepcopy(lic)

	async def get_license(self, tenant_id: str, license_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		l = self.licenses.get(license_id)
		if not l or l["tenant_id"] != tenant:
			raise KeyError(f"license {license_id} not found")
		return deepcopy(l)

	async def list_licenses(self, tenant_id: str, asset_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(l) for l in self.licenses.values() if l["tenant_id"] == tenant]
		if asset_id:
			items = [l for l in items if l["asset_id"] == asset_id]
		return items

	async def terminate_license(self, tenant_id: str, license_id: str, reason: str) -> dict[str, Any]:
		"""Terminate an IP license."""
		tenant = self._tenant(tenant_id)
		l = self.licenses.get(license_id)
		if not l or l["tenant_id"] != tenant:
			raise KeyError(f"license {license_id} not found")
		l["status"] = "terminated"
		l["termination_reason"] = reason
		l["terminated_at"] = self._now()
		a = self.assets.get(l["asset_id"])
		if a:
			a["license_count"] = max(0, a.get("license_count", 1) - 1)
		self._emit(tenant, "ip_license_terminated", license_id)
		return deepcopy(l)

	async def delete_license(self, tenant_id: str, license_id: str) -> dict[str, Any]:
		return await self.terminate_license(tenant_id, license_id, reason="archived")

	# ── Royalties ────────────────────────────────────────────────────────────

	async def record_royalty(
		self,
		tenant_id: str,
		license_id: str,
		period: str,
		base_amount: float,
		submitted_by_id: str,
		currency: str = "KES",
		notes: str = "",
	) -> dict[str, Any]:
		"""Record a royalty payment."""
		tenant = self._tenant(tenant_id)
		lic = self.licenses.get(license_id)
		if not lic or lic["tenant_id"] != tenant:
			raise KeyError(f"license {license_id} not found")
		royalty_amount = round(base_amount * lic["royalty_rate"] / 100, 2)
		royalty: dict[str, Any] = {
			"id": self._id("roy-"),
			"tenant_id": tenant,
			"license_id": license_id,
			"period": period,
			"base_amount": base_amount,
			"royalty_amount": royalty_amount,
			"currency": currency,
			"submitted_by_id": submitted_by_id,
			"notes": notes,
			"status": "pending",
			"paid_at": None,
			"created_at": self._now(),
		}
		self.royalties[royalty["id"]] = royalty
		self._emit(tenant, "royalty_recorded", royalty["id"], {"license_id": license_id, "period": period})
		return deepcopy(royalty)

	async def get_royalty(self, tenant_id: str, royalty_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		r = self.royalties.get(royalty_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"royalty {royalty_id} not found")
		return deepcopy(r)

	async def list_royalties(self, tenant_id: str, license_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.royalties.values() if r["tenant_id"] == tenant]
		if license_id:
			items = [r for r in items if r["license_id"] == license_id]
		return items

	async def pay_royalty(self, tenant_id: str, royalty_id: str, payment_reference: str) -> dict[str, Any]:
		"""Mark a royalty as paid."""
		tenant = self._tenant(tenant_id)
		r = self.royalties.get(royalty_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"royalty {royalty_id} not found")
		r["status"] = "paid"
		r["paid_at"] = self._now()
		r["payment_reference"] = payment_reference
		self._emit(tenant, "royalty_paid", royalty_id)
		return deepcopy(r)

	async def delete_royalty(self, tenant_id: str, royalty_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		r = self.royalties.get(royalty_id)
		if not r or r["tenant_id"] != tenant:
			raise KeyError(f"royalty {royalty_id} not found")
		r["status"] = "cancelled"
		self._emit(tenant, "royalty_cancelled", royalty_id)
		return deepcopy(r)

	# ── Analytics ────────────────────────────────────────────────────────────

	async def portfolio_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return portfolio metrics."""
		tenant = self._tenant(tenant_id)
		assets = [a for a in self.assets.values() if a["tenant_id"] == tenant]
		by_type: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for a in assets:
			by_type[a["asset_type"]] = by_type.get(a["asset_type"], 0) + 1
			by_status[a["status"]] = by_status.get(a["status"], 0) + 1
		total_royalties = sum(r["royalty_amount"] for r in self.royalties.values() if r["tenant_id"] == tenant and r["status"] == "paid")
		return {
			"tenant_id": tenant,
			"total_assets": len(assets),
			"by_type": by_type,
			"by_status": by_status,
			"active_licenses": sum(1 for l in self.licenses.values() if l["tenant_id"] == tenant and l["status"] == "active"),
			"total_royalties_paid": total_royalties,
			"generated_at": self._now(),
		}

	async def expiring_assets(self, tenant_id: str, days_ahead: int = 90) -> list[dict[str, Any]]:
		"""Return registered assets expiring in the next N days."""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		items = [
			deepcopy(a) for a in self.assets.values()
			if a["tenant_id"] == tenant
			and a["status"] == "registered"
			and a.get("expiry_date")
			and a["expiry_date"] >= today
		]
		return sorted(items, key=lambda a: a["expiry_date"])[:50]

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

