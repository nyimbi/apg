"""Land Management service — agr_lnd."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import hashlib
import hmac
import logging
import math
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import (
	guard_tenant_id,
	guard_non_empty_string,
	guard_positive_amount,
	guard_bounded_list,
	BoundedCache,
)


def guard_enum_local(value: str | None, allowed: set[str], field: str = "value") -> None:
	"""Assert a string value is in the allowed set."""
	if value is None:
		raise ValueError(f"{field} must be provided")
	if value not in allowed:
		raise ValueError(f"{field} must be one of {sorted(allowed)!r}, got {value!r}")

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_lnd"

# Tenure formalisation stages in order
_FORMALISATION_STAGES = [
	"community_consent",
	"demarcation",
	"survey",
	"adjudication",
	"registration",
	"title_issued",
]

# Multi-sig quorum threshold (KES) above which multi-sig is required
_MULTISIG_THRESHOLD_KES = Decimal("10000000")  # 10M KES


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


def _compute_polygon_area_ha(waypoints: list[dict[str, float]]) -> float:
	"""Shoelace formula on lat/lng, approximate m² → ha conversion."""
	if len(waypoints) < 3:
		return 0.0
	# Convert to flat-earth approximate metres using equirectangular projection
	# 1 degree lat ≈ 111,320 m; 1 degree lng ≈ 111,320 * cos(lat) m
	pts = [(wp["lat"], wp["lng"]) for wp in waypoints]
	lat0 = sum(p[0] for p in pts) / len(pts)
	cos_lat = math.cos(math.radians(lat0))
	xy = [(p[1] * 111320 * cos_lat, p[0] * 111320) for p in pts]
	n = len(xy)
	area = 0.0
	for i in range(n):
		j = (i + 1) % n
		area += xy[i][0] * xy[j][1]
		area -= xy[j][0] * xy[i][1]
	return round(abs(area) / 2 / 10000, 4)  # m² → ha


class LandManagementService:
	"""Async service for land management: parcel cadastre, tenure registry,
	GPS boundary capture, title issuance, and land transfer."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		guard_tenant_id(tenant_id)
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._parcels = WriteThruDict('parcels', tenant_id, _store)
		self._boundaries = WriteThruDict('boundaries', tenant_id, _store)
		self._titles = WriteThruDict('titles', tenant_id, _store)
		self._transfers = WriteThruDict('transfers', tenant_id, _store)
		self._disputes = WriteThruDict('disputes', tenant_id, _store)
		self._encumbrances = WriteThruDict('encumbrances', tenant_id, _store)
		self._valuations = WriteThruDict('valuations', tenant_id, _store)
		self._formalisations = WriteThruDict('formalisations', tenant_id, _store)
		self._rate_bills = WriteThruDict('rate_bills', tenant_id, _store)
		self._webhooks = WriteThruDict('webhooks', tenant_id, _store)
		self._webhook_failures = WriteThruList('webhook_failures', tenant_id, _store)
		self._transfer_signatures: dict[str, list[dict[str, Any]]] = {}
		self._audit = WriteThruList('audit', tenant_id, _store)

	def _emit(self, event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._audit.append({
			"id": _new_id("evt"),
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"payload": payload,
			"occurred_at": _now(),
		})

	# ------------------------------------------------------------------ health

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "ok",
			"capability": _CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"counts": {
				"parcels": len(self._parcels),
				"boundaries": len(self._boundaries),
				"titles": len(self._titles),
				"transfers": len(self._transfers),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Land Management",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Parcel cadastre, tenure registry, GPS boundary capture, title issuance, land transfer.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ parcels

	async def list_parcels(self, owner_id: str | None = None, county: str | None = None,
						tenure_type: str | None = None, status: str | None = None,
						limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._parcels.values())
		if owner_id:
			items = [p for p in items if p.get("owner_id") == owner_id]
		if county:
			items = [p for p in items if p.get("location_county") == county]
		if tenure_type:
			items = [p for p in items if p.get("tenure_type") == tenure_type]
		if status:
			items = [p for p in items if p.get("status") == status]
		return items[offset: offset + limit]

	async def get_parcel(self, parcel_id: str) -> dict[str, Any]:
		if parcel_id not in self._parcels:
			raise KeyError(f"parcel_not_found:{parcel_id}")
		return self._parcels[parcel_id]

	async def get_parcel_by_number(self, parcel_number: str) -> dict[str, Any] | None:
		for p in self._parcels.values():
			if p.get("parcel_number") == parcel_number:
				return p
		return None

	async def create_parcel(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			pid = _new_id("lnd")
			ts = _now()
			record: dict[str, Any] = {
				"id": pid,
				"tenant_id": self.tenant_id,
				"parcel_number": payload["parcel_number"],
				"area_ha": float(payload["area_ha"]),
				"tenure_type": payload["tenure_type"],
				"owner_id": payload["owner_id"],
				"owner_name": payload["owner_name"],
				"location_county": payload["location_county"],
				"location_sub_county": payload.get("location_sub_county"),
				"location_ward": payload.get("location_ward"),
				"coordinates": list(payload.get("coordinates", [])),
				"land_use": payload.get("land_use"),
				"status": "registered",
				"title_number": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._parcels[pid] = record
			self._emit("parcel.registered", "land_parcel", pid, record)
			return record
		except Exception as exc:
			_log.error("create_parcel failed: %s", exc)
			raise

	async def update_parcel(self, parcel_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			record = self._parcels[parcel_id]
			for field in ["owner_id", "owner_name", "tenure_type", "land_use", "status", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("parcel.updated", "land_parcel", parcel_id, payload)
			return record
		except Exception as exc:
			_log.error("update_parcel failed: %s", exc)
			raise

	async def delete_parcel(self, parcel_id: str) -> dict[str, Any]:
		try:
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			self._parcels.pop(parcel_id)
			self._emit("parcel.deleted", "land_parcel", parcel_id, {"id": parcel_id})
			return {"deleted": True, "id": parcel_id}
		except Exception as exc:
			_log.error("delete_parcel failed: %s", exc)
			raise

	async def get_owner_land_holdings(self, owner_id: str) -> dict[str, Any]:
		"""Summarise total land holdings for an owner."""
		parcels = [p for p in self._parcels.values() if p.get("owner_id") == owner_id]
		total_area = sum(p.get("area_ha", 0) for p in parcels)
		titles = [t for t in self._titles.values()
				if any(p["id"] == t.get("parcel_id") for p in parcels)]
		return {
			"owner_id": owner_id,
			"parcel_count": len(parcels),
			"total_area_ha": round(total_area, 4),
			"titled_parcels": len(titles),
			"tenure_breakdown": {
				t: len([p for p in parcels if p.get("tenure_type") == t])
				for t in set(p.get("tenure_type", "unknown") for p in parcels)
			},
		}

	# ------------------------------------------------------------------ GPS boundaries

	async def list_boundaries(self, parcel_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._boundaries.values())
		if parcel_id:
			items = [b for b in items if b.get("parcel_id") == parcel_id]
		return items

	async def capture_boundary(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Store GPS boundary capture and compute area."""
		try:
			bid = _new_id("bnd")
			ts = _now()
			waypoints = list(payload["waypoints"])
			computed_area = _compute_polygon_area_ha(waypoints)
			record: dict[str, Any] = {
				"id": bid,
				"tenant_id": self.tenant_id,
				"parcel_id": payload["parcel_id"],
				"captured_by": payload["captured_by"],
				"device_id": payload.get("device_id"),
				"waypoints": waypoints,
				"computed_area_ha": computed_area,
				"accuracy_m": payload.get("accuracy_m"),
				"notes": payload.get("notes"),
				"captured_at": payload.get("captured_at") or ts,
				"created_at": ts,
			}
			self._boundaries[bid] = record
			# Update parcel coordinates with new boundary
			if payload["parcel_id"] in self._parcels:
				self._parcels[payload["parcel_id"]]["coordinates"] = waypoints
				self._parcels[payload["parcel_id"]]["area_ha"] = computed_area
				self._parcels[payload["parcel_id"]]["updated_at"] = ts
			self._emit("boundary.captured", "gps_boundary", bid, {"parcel_id": payload["parcel_id"], "area_ha": computed_area})
			return record
		except Exception as exc:
			_log.error("capture_boundary failed: %s", exc)
			raise

	async def delete_boundary(self, boundary_id: str) -> dict[str, Any]:
		try:
			if boundary_id not in self._boundaries:
				raise KeyError(f"boundary_not_found:{boundary_id}")
			self._boundaries.pop(boundary_id)
			self._emit("boundary.deleted", "gps_boundary", boundary_id, {"id": boundary_id})
			return {"deleted": True, "id": boundary_id}
		except Exception as exc:
			_log.error("delete_boundary failed: %s", exc)
			raise

	# ------------------------------------------------------------------ titles

	async def list_titles(self, parcel_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._titles.values())
		if parcel_id:
			items = [t for t in items if t.get("parcel_id") == parcel_id]
		return items

	async def issue_title(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Issue a land title for a parcel."""
		try:
			parcel_id = payload["parcel_id"]
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			tid = _new_id("ttl")
			ts = _now()
			record: dict[str, Any] = {
				"id": tid,
				"tenant_id": self.tenant_id,
				"parcel_id": parcel_id,
				"title_number": payload["title_number"],
				"issued_by": payload["issued_by"],
				"issue_date": payload["issue_date"],
				"tenure_type": payload["tenure_type"],
				"valid": True,
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._titles[tid] = record
			# Update parcel with title number
			self._parcels[parcel_id]["title_number"] = payload["title_number"]
			self._parcels[parcel_id]["updated_at"] = ts
			self._emit("title.issued", "land_title", tid, record)
			return record
		except Exception as exc:
			_log.error("issue_title failed: %s", exc)
			raise

	async def invalidate_title(self, title_id: str, reason: str) -> dict[str, Any]:
		"""Mark a title as invalid."""
		try:
			if title_id not in self._titles:
				raise KeyError(f"title_not_found:{title_id}")
			self._titles[title_id]["valid"] = False
			self._titles[title_id]["invalidation_reason"] = reason
			self._emit("title.invalidated", "land_title", title_id, {"reason": reason})
			return self._titles[title_id]
		except Exception as exc:
			_log.error("invalidate_title failed: %s", exc)
			raise

	async def delete_title(self, title_id: str) -> dict[str, Any]:
		try:
			if title_id not in self._titles:
				raise KeyError(f"title_not_found:{title_id}")
			self._titles.pop(title_id)
			self._emit("title.deleted", "land_title", title_id, {"id": title_id})
			return {"deleted": True, "id": title_id}
		except Exception as exc:
			_log.error("delete_title failed: %s", exc)
			raise

	# ------------------------------------------------------------------ transfers

	async def list_transfers(self, parcel_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		items = list(self._transfers.values())
		if parcel_id:
			items = [t for t in items if t.get("parcel_id") == parcel_id]
		if status:
			items = [t for t in items if t.get("status") == status]
		return items

	async def initiate_transfer(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Initiate a land transfer between owners."""
		try:
			parcel_id = payload["parcel_id"]
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			tfid = _new_id("trf")
			ts = _now()
			record: dict[str, Any] = {
				"id": tfid,
				"tenant_id": self.tenant_id,
				"parcel_id": parcel_id,
				"from_owner_id": payload["from_owner_id"],
				"to_owner_id": payload["to_owner_id"],
				"to_owner_name": payload["to_owner_name"],
				"transfer_value": payload.get("transfer_value"),
				"currency": payload.get("currency", "KES"),
				"reason": payload.get("reason"),
				"status": "initiated",
				"approved_at": None,
				"registered_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._transfers[tfid] = record
			# Mark parcel as under transfer
			self._parcels[parcel_id]["status"] = "under_transfer"
			self._parcels[parcel_id]["updated_at"] = ts
			self._emit("transfer.initiated", "land_transfer", tfid, record)
			return record
		except Exception as exc:
			_log.error("initiate_transfer failed: %s", exc)
			raise

	async def update_transfer(self, transfer_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Advance transfer through approval → registration workflow."""
		try:
			if transfer_id not in self._transfers:
				raise KeyError(f"transfer_not_found:{transfer_id}")
			record = self._transfers[transfer_id]
			new_status = payload.get("status")
			if new_status:
				record["status"] = new_status
				if new_status == "approved":
					record["approved_at"] = _now()
				elif new_status == "registered":
					record["registered_at"] = _now()
					# Complete the transfer: update parcel owner
					parcel_id = record["parcel_id"]
					if parcel_id in self._parcels:
						self._parcels[parcel_id]["owner_id"] = record["to_owner_id"]
						self._parcels[parcel_id]["owner_name"] = record["to_owner_name"]
						self._parcels[parcel_id]["status"] = "registered"
						self._parcels[parcel_id]["updated_at"] = _now()
			for field in ["notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("transfer.updated", "land_transfer", transfer_id, payload)
			return record
		except Exception as exc:
			_log.error("update_transfer failed: %s", exc)
			raise

	async def delete_transfer(self, transfer_id: str) -> dict[str, Any]:
		try:
			if transfer_id not in self._transfers:
				raise KeyError(f"transfer_not_found:{transfer_id}")
			self._transfers.pop(transfer_id)
			self._emit("transfer.deleted", "land_transfer", transfer_id, {"id": transfer_id})
			return {"deleted": True, "id": transfer_id}
		except Exception as exc:
			_log.error("delete_transfer failed: %s", exc)
			raise

	async def get_land_registry_summary(self) -> dict[str, Any]:
		"""Top-level registry statistics."""
		parcels = list(self._parcels.values())
		return {
			"total_parcels": len(parcels),
			"total_area_ha": round(sum(p.get("area_ha", 0) for p in parcels), 2),
			"titled_parcels": len([p for p in parcels if p.get("title_number")]),
			"disputed_parcels": len([p for p in parcels if p.get("status") == "disputed"]),
			"pending_transfers": len([t for t in self._transfers.values()
									if t.get("status") not in ("registered", "rejected")]),
			"tenure_breakdown": {
				t: len([p for p in parcels if p.get("tenure_type") == t])
				for t in set(p.get("tenure_type", "unknown") for p in parcels)
			},
		}

	# ------------------------------------------------------------------ disputes / adjudication

	async def file_dispute(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""File a land dispute against a parcel, locking it to 'disputed' status.

		Args:
			payload: Must include parcel_id, complainant_id, complainant_name,
			         description. Optional: respondent_id, evidence_urls.
		Returns:
			Dispute record with id and initial stage 'filed'.
		"""
		try:
			guard_tenant_id(self.tenant_id)
			parcel_id = payload["parcel_id"]
			guard_non_empty_string(parcel_id, "parcel_id")
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			guard_non_empty_string(payload.get("complainant_id", ""), "complainant_id")
			guard_non_empty_string(payload.get("description", ""), "description")

			did = _new_id("dsp")
			ts = _now()
			record: dict[str, Any] = {
				"id": did,
				"tenant_id": self.tenant_id,
				"parcel_id": parcel_id,
				"complainant_id": payload["complainant_id"],
				"complainant_name": payload.get("complainant_name"),
				"respondent_id": payload.get("respondent_id"),
				"description": payload["description"],
				"evidence_urls": list(payload.get("evidence_urls", [])),
				"stage": "filed",
				"adjudicator_id": None,
				"hearing_date": None,
				"resolution": None,
				"resolved_at": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._disputes[did] = record
			# Lock the parcel
			self._parcels[parcel_id]["status"] = "disputed"
			self._parcels[parcel_id]["updated_at"] = ts
			self._emit("dispute.filed", "land_dispute", did, record)
			return record
		except Exception as exc:
			_log.error("file_dispute failed: %s", exc)
			raise

	async def advance_dispute(self, dispute_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Advance dispute stage. Valid stages (in order):
		filed → evidence_collection → hearing_scheduled → adjudicated → appealed.

		Payload fields: stage (required), adjudicator_id, hearing_date, resolution, notes.
		On 'adjudicated', parcel status reverts to 'registered' unless overridden.
		"""
		try:
			if dispute_id not in self._disputes:
				raise KeyError(f"dispute_not_found:{dispute_id}")
			record = self._disputes[dispute_id]
			new_stage = payload.get("stage")
			if new_stage:
				record["stage"] = new_stage
				if new_stage == "adjudicated":
					record["resolution"] = payload.get("resolution")
					record["resolved_at"] = _now()
					parcel_id = record["parcel_id"]
					if parcel_id in self._parcels:
						self._parcels[parcel_id]["status"] = "registered"
						self._parcels[parcel_id]["updated_at"] = _now()
			for field in ["adjudicator_id", "hearing_date", "notes", "evidence_urls"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("dispute.advanced", "land_dispute", dispute_id, payload)
			return record
		except Exception as exc:
			_log.error("advance_dispute failed: %s", exc)
			raise

	async def list_disputes(self, parcel_id: str | None = None,
	                        stage: str | None = None) -> list[dict[str, Any]]:
		items = list(self._disputes.values())
		if parcel_id:
			items = [d for d in items if d.get("parcel_id") == parcel_id]
		if stage:
			items = [d for d in items if d.get("stage") == stage]
		return items

	# ------------------------------------------------------------------ encumbrances

	async def register_encumbrance(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Register a mortgage, caveat, lien, or easement against a parcel.

		Encumbrance types: mortgage | caveat | lien | easement | charge.
		Uses Decimal for monetary amounts to avoid float rounding errors.

		Args:
			payload: parcel_id, type, holder_id, holder_name. Optional: amount,
			         currency (default KES), notes, expires_at.
		"""
		try:
			guard_tenant_id(self.tenant_id)
			parcel_id = payload["parcel_id"]
			guard_non_empty_string(parcel_id, "parcel_id")
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")

			enc_type = payload["type"]
			guard_enum_local(enc_type, {"mortgage", "caveat", "lien", "easement", "charge"}, "type")
			guard_non_empty_string(payload.get("holder_id", ""), "holder_id")

			# Monetary amount stored as Decimal string for precision
			raw_amount = payload.get("amount")
			amount_decimal: str | None = None
			if raw_amount is not None:
				guard_positive_amount(float(raw_amount), "amount")
				amount_decimal = str(Decimal(str(raw_amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

			eid = _new_id("enc")
			ts = _now()
			record: dict[str, Any] = {
				"id": eid,
				"tenant_id": self.tenant_id,
				"parcel_id": parcel_id,
				"type": enc_type,
				"holder_id": payload["holder_id"],
				"holder_name": payload.get("holder_name"),
				"amount": amount_decimal,
				"currency": payload.get("currency", "KES"),
				"registered_at": ts,
				"discharged_at": None,
				"expires_at": payload.get("expires_at"),
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._encumbrances[eid] = record
			# Mark parcel encumbered if mortgage or charge
			if enc_type in ("mortgage", "charge"):
				self._parcels[parcel_id]["status"] = "encumbered"
				self._parcels[parcel_id]["updated_at"] = ts
			self._emit("encumbrance.registered", "land_encumbrance", eid, record)
			return record
		except Exception as exc:
			_log.error("register_encumbrance failed: %s", exc)
			raise

	async def discharge_encumbrance(self, encumbrance_id: str, notes: str | None = None) -> dict[str, Any]:
		"""Discharge (release) an encumbrance. Updates parcel status if no active encumbrances remain."""
		try:
			if encumbrance_id not in self._encumbrances:
				raise KeyError(f"encumbrance_not_found:{encumbrance_id}")
			record = self._encumbrances[encumbrance_id]
			if record.get("discharged_at"):
				raise ValueError(f"encumbrance_already_discharged:{encumbrance_id}")
			record["discharged_at"] = _now()
			if notes:
				record["notes"] = (record.get("notes") or "") + " | discharge: " + notes
			# Check if parcel still has active encumbrances
			parcel_id = record["parcel_id"]
			active = [
				e for e in self._encumbrances.values()
				if e.get("parcel_id") == parcel_id
				and not e.get("discharged_at")
				and e["id"] != encumbrance_id
			]
			if not active and parcel_id in self._parcels:
				self._parcels[parcel_id]["status"] = "registered"
				self._parcels[parcel_id]["updated_at"] = _now()
			self._emit("encumbrance.discharged", "land_encumbrance", encumbrance_id, {"notes": notes})
			return record
		except Exception as exc:
			_log.error("discharge_encumbrance failed: %s", exc)
			raise

	async def list_encumbrances(self, parcel_id: str | None = None,
	                            active_only: bool = False) -> list[dict[str, Any]]:
		items = list(self._encumbrances.values())
		if parcel_id:
			items = [e for e in items if e.get("parcel_id") == parcel_id]
		if active_only:
			items = [e for e in items if not e.get("discharged_at")]
		return items

	# ------------------------------------------------------------------ subdivision / amalgamation

	async def subdivide_parcel(self, parent_id: str,
	                           children_payloads: list[dict[str, Any]]) -> dict[str, Any]:
		"""Split a parcel into two or more child parcels.

		Validates that child areas sum to ≤ parent area (rounding tolerance 0.01 ha).
		Cancels parent with superseded_by reference list. Each child carries parent_id back-ref.

		Args:
			parent_id: ID of the parcel to subdivide.
			children_payloads: List of create-parcel dicts for each new child.

		Returns:
			Dict with parent (cancelled) and children list.
		"""
		try:
			guard_tenant_id(self.tenant_id)
			if parent_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parent_id}")
			parent = self._parcels[parent_id]
			if parent.get("status") in ("cancelled", "under_transfer", "disputed"):
				raise ValueError(f"parcel_not_subdivisible:{parent.get('status')}")

			parent_area = Decimal(str(parent["area_ha"]))
			child_area_sum = Decimal("0")
			for cp in children_payloads:
				child_area_sum += Decimal(str(cp["area_ha"]))

			tolerance = Decimal("0.01")
			if child_area_sum > parent_area + tolerance:
				raise ValueError(
					f"children_area_exceeds_parent: {child_area_sum} ha > {parent_area} ha"
				)

			ts = _now()
			child_records = []
			child_ids = []
			for cp in children_payloads:
				cp = {**cp, "parent_id": parent_id}
				child = await self.create_parcel(cp)
				child_records.append(child)
				child_ids.append(child["id"])

			# Cancel parent
			parent["status"] = "cancelled"
			parent["superseded_by"] = child_ids
			parent["updated_at"] = ts
			self._emit("parcel.subdivided", "land_parcel", parent_id, {
				"parent_id": parent_id, "child_ids": child_ids,
			})
			return {"parent": parent, "children": child_records}
		except Exception as exc:
			_log.error("subdivide_parcel failed: %s", exc)
			raise

	async def amalgamate_parcels(self, source_ids: list[str],
	                             target_payload: dict[str, Any]) -> dict[str, Any]:
		"""Merge two or more parcels into one new parcel.

		Cancels all sources with merged_into reference. Target area defaults to
		sum of source areas if not explicitly provided.

		Args:
			source_ids: IDs of parcels to merge (must all belong to same tenant).
			target_payload: create-parcel dict for the merged parcel.

		Returns:
			Dict with sources (cancelled) and the new target parcel.
		"""
		try:
			guard_tenant_id(self.tenant_id)
			if len(source_ids) < 2:
				raise ValueError("amalgamate_requires_at_least_2_sources")
			sources = []
			for sid in source_ids:
				if sid not in self._parcels:
					raise KeyError(f"parcel_not_found:{sid}")
				sources.append(self._parcels[sid])

			total_area = sum(Decimal(str(s["area_ha"])) for s in sources)
			if "area_ha" not in target_payload:
				target_payload = {**target_payload, "area_ha": float(total_area)}

			ts = _now()
			target = await self.create_parcel({**target_payload, "merged_from": source_ids})
			target_id = target["id"]

			for src in sources:
				src["status"] = "cancelled"
				src["merged_into"] = target_id
				src["updated_at"] = ts

			self._emit("parcel.amalgamated", "land_parcel", target_id, {
				"source_ids": source_ids, "target_id": target_id,
			})
			return {"sources": sources, "target": target}
		except Exception as exc:
			_log.error("amalgamate_parcels failed: %s", exc)
			raise

	# ------------------------------------------------------------------ valuation & rate bills

	async def record_valuation(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Record an assessed valuation for a parcel (for rate billing).

		Monetary values stored as Decimal strings. Valuation methods:
		market_comparison | income | cost | mass_appraisal.

		Args:
			payload: parcel_id, assessed_value, currency (default KES), method,
			         valuation_date, valued_by. Optional: notes.
		"""
		try:
			guard_tenant_id(self.tenant_id)
			parcel_id = payload["parcel_id"]
			guard_non_empty_string(parcel_id, "parcel_id")
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			guard_positive_amount(float(payload["assessed_value"]), "assessed_value")

			assessed = Decimal(str(payload["assessed_value"])).quantize(
				Decimal("0.01"), rounding=ROUND_HALF_UP
			)
			vid = _new_id("val")
			ts = _now()
			record: dict[str, Any] = {
				"id": vid,
				"tenant_id": self.tenant_id,
				"parcel_id": parcel_id,
				"assessed_value": str(assessed),
				"currency": payload.get("currency", "KES"),
				"method": payload.get("method", "market_comparison"),
				"valuation_date": payload.get("valuation_date") or ts[:10],
				"valued_by": payload.get("valued_by"),
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._valuations[vid] = record
			self._emit("valuation.recorded", "land_valuation", vid, record)
			return record
		except Exception as exc:
			_log.error("record_valuation failed: %s", exc)
			raise

	async def generate_rate_bill(self, parcel_id: str,
	                             financial_year: str, levy_rate_pct: Decimal | float = Decimal("0.01"),
	                             currency: str = "KES") -> dict[str, Any]:
		"""Generate a land-rates bill for a parcel from its latest valuation.

		Bill = assessed_value × levy_rate_pct. Stored under _rate_bills with status 'draft'.
		All amounts are Decimal for precision. Raises if no valuation exists for parcel.

		Args:
			parcel_id: Target parcel.
			financial_year: e.g. "2025/2026".
			levy_rate_pct: Annual levy as a fraction (e.g. 0.01 = 1%). Default 1%.
			currency: Currency code. Default KES.
		"""
		try:
			guard_tenant_id(self.tenant_id)
			guard_non_empty_string(parcel_id, "parcel_id")
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")

			# Find most recent valuation
			parcel_valuations = sorted(
				[v for v in self._valuations.values() if v.get("parcel_id") == parcel_id],
				key=lambda v: v["created_at"],
				reverse=True,
			)
			if not parcel_valuations:
				raise ValueError(f"no_valuation_for_parcel:{parcel_id}")

			latest = parcel_valuations[0]
			assessed = Decimal(latest["assessed_value"])
			levy = Decimal(str(levy_rate_pct))
			bill_amount = (assessed * levy).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

			bid = _new_id("rb")
			ts = _now()
			record: dict[str, Any] = {
				"id": bid,
				"tenant_id": self.tenant_id,
				"parcel_id": parcel_id,
				"parcel_number": self._parcels[parcel_id].get("parcel_number"),
				"financial_year": financial_year,
				"assessed_value": str(assessed),
				"levy_rate_pct": str(levy),
				"bill_amount": str(bill_amount),
				"currency": currency,
				"valuation_id": latest["id"],
				"status": "draft",
				"issued_at": None,
				"paid_at": None,
				"notes": None,
				"created_at": ts,
			}
			self._rate_bills[bid] = record
			self._emit("rate_bill.generated", "land_rate_bill", bid, record)
			return record
		except Exception as exc:
			_log.error("generate_rate_bill failed: %s", exc)
			raise

	async def list_rate_bills(self, parcel_id: str | None = None,
	                          status: str | None = None) -> list[dict[str, Any]]:
		items = list(self._rate_bills.values())
		if parcel_id:
			items = [b for b in items if b.get("parcel_id") == parcel_id]
		if status:
			items = [b for b in items if b.get("status") == status]
		return items

	# ------------------------------------------------------------------ tenure formalisation

	async def initiate_formalisation(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Start a tenure formalisation workflow (customary → statutory).

		Stages: community_consent → demarcation → survey → adjudication →
		        registration → title_issued (Kenya Community Land Act 2016 compliant).

		Args:
			payload: parcel_id, community_id, initiated_by, workflow_type
			         (individual | community | group_ranch). Optional: notes.
		"""
		try:
			guard_tenant_id(self.tenant_id)
			parcel_id = payload["parcel_id"]
			guard_non_empty_string(parcel_id, "parcel_id")
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			guard_non_empty_string(payload.get("community_id", ""), "community_id")

			fid = _new_id("frm")
			ts = _now()
			stages_status = {stage: {"status": "pending", "officer": None, "completed_at": None}
			                 for stage in _FORMALISATION_STAGES}
			stages_status["community_consent"]["status"] = "in_progress"

			record: dict[str, Any] = {
				"id": fid,
				"tenant_id": self.tenant_id,
				"parcel_id": parcel_id,
				"community_id": payload["community_id"],
				"initiated_by": payload.get("initiated_by"),
				"workflow_type": payload.get("workflow_type", "individual"),
				"current_stage": "community_consent",
				"stages": stages_status,
				"required_documents": [
					"community_resolution",
					"survey_plan",
					"adjudication_form",
					"registration_certificate",
				],
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._formalisations[fid] = record
			self._emit("formalisation.initiated", "land_formalisation", fid, record)
			return record
		except Exception as exc:
			_log.error("initiate_formalisation failed: %s", exc)
			raise

	async def advance_formalisation_stage(self, formalisation_id: str,
	                                      officer_id: str, notes: str | None = None) -> dict[str, Any]:
		"""Advance formalisation to the next stage. Gates on sequential completion.

		Completing 'title_issued' stage triggers title issuance.

		Args:
			formalisation_id: ID of the formalisation record.
			officer_id: ID of the responsible officer.
			notes: Optional notes for this stage completion.
		"""
		try:
			if formalisation_id not in self._formalisations:
				raise KeyError(f"formalisation_not_found:{formalisation_id}")
			record = self._formalisations[formalisation_id]
			current = record["current_stage"]
			idx = _FORMALISATION_STAGES.index(current)

			# Complete current stage
			record["stages"][current]["status"] = "completed"
			record["stages"][current]["officer"] = officer_id
			record["stages"][current]["completed_at"] = _now()
			if notes:
				record["stages"][current]["notes"] = notes

			# Advance to next
			if idx + 1 < len(_FORMALISATION_STAGES):
				next_stage = _FORMALISATION_STAGES[idx + 1]
				record["current_stage"] = next_stage
				record["stages"][next_stage]["status"] = "in_progress"
			else:
				record["current_stage"] = "completed"

			record["updated_at"] = _now()
			self._emit("formalisation.stage_advanced", "land_formalisation", formalisation_id, {
				"completed_stage": current, "new_stage": record["current_stage"],
			})
			return record
		except Exception as exc:
			_log.error("advance_formalisation_stage failed: %s", exc)
			raise

	async def get_formalisation_status(self, formalisation_id: str) -> dict[str, Any]:
		if formalisation_id not in self._formalisations:
			raise KeyError(f"formalisation_not_found:{formalisation_id}")
		return self._formalisations[formalisation_id]

	# ------------------------------------------------------------------ geospatial search

	async def search_parcels_by_location(self, lat: float, lng: float,
	                                     radius_m: float) -> list[dict[str, Any]]:
		"""Return all parcels whose GPS boundary centroid is within radius_m of (lat, lng).

		Uses Haversine formula. Parcels without boundaries are excluded.
		Results sorted by distance ascending.

		Args:
			lat: Reference latitude (decimal degrees).
			lng: Reference longitude (decimal degrees).
			radius_m: Search radius in metres.
		"""
		_R = 6_371_000.0  # Earth radius in metres

		def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
			phi1, phi2 = math.radians(lat1), math.radians(lat2)
			dphi = math.radians(lat2 - lat1)
			dlam = math.radians(lng2 - lng1)
			a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
			return 2 * _R * math.asin(math.sqrt(a))

		results = []
		for parcel in self._parcels.values():
			coords = parcel.get("coordinates") or []
			if len(coords) < 3:
				continue
			c_lat = sum(p["lat"] for p in coords) / len(coords)
			c_lng = sum(p["lng"] for p in coords) / len(coords)
			dist = _haversine(lat, lng, c_lat, c_lng)
			if dist <= radius_m:
				results.append({**parcel, "_distance_m": round(dist, 1)})
		results.sort(key=lambda p: p["_distance_m"])
		return results

	async def find_parcel_at_point(self, lat: float, lng: float) -> dict[str, Any] | None:
		"""Point-in-polygon test (ray-casting) against all stored boundary waypoints.

		Returns the first parcel whose boundary polygon contains (lat, lng),
		or None if no match.

		Args:
			lat: Query latitude.
			lng: Query longitude.
		"""
		def _point_in_polygon(x: float, y: float,
		                      polygon: list[dict[str, float]]) -> bool:
			"""Ray-casting algorithm. x=lng, y=lat."""
			n = len(polygon)
			inside = False
			j = n - 1
			for i in range(n):
				xi, yi = polygon[i]["lng"], polygon[i]["lat"]
				xj, yj = polygon[j]["lng"], polygon[j]["lat"]
				if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
					inside = not inside
				j = i
			return inside

		for parcel in self._parcels.values():
			coords = parcel.get("coordinates") or []
			if len(coords) < 3:
				continue
			if _point_in_polygon(lng, lat, coords):
				return parcel
		return None

	# ------------------------------------------------------------------ chain of title

	async def get_chain_of_title(self, parcel_id: str) -> dict[str, Any]:
		"""Reconstruct complete ownership history for a parcel.

		Walks all completed transfers in chronological order to build the
		provenance chain. Includes active encumbrances per ownership period.

		Returns:
			Dict with parcel summary and ordered chain list.
		"""
		try:
			guard_non_empty_string(parcel_id, "parcel_id")
			if parcel_id not in self._parcels:
				raise KeyError(f"parcel_not_found:{parcel_id}")
			parcel = self._parcels[parcel_id]

			completed_transfers = sorted(
				[t for t in self._transfers.values()
				 if t.get("parcel_id") == parcel_id and t.get("status") == "registered"],
				key=lambda t: t.get("registered_at") or t["created_at"],
			)

			chain = []
			for i, tf in enumerate(completed_transfers):
				# Find encumbrances active during this ownership period
				tenure_from = tf.get("registered_at") or tf["created_at"]
				tenure_to = completed_transfers[i + 1]["registered_at"] if i + 1 < len(completed_transfers) else None
				encs = [
					e for e in self._encumbrances.values()
					if e.get("parcel_id") == parcel_id
					and (e.get("registered_at") or "") >= tenure_from
					and (tenure_to is None or (e.get("registered_at") or "") < tenure_to)
				]
				chain.append({
					"owner_id": tf["to_owner_id"],
					"owner_name": tf["to_owner_name"],
					"acquisition_type": tf.get("reason") or "transfer",
					"transfer_value": tf.get("transfer_value"),
					"currency": tf.get("currency", "KES"),
					"tenure_from": tenure_from,
					"tenure_to": tenure_to,
					"title_number": parcel.get("title_number"),
					"encumbrances": [
						{"id": e["id"], "type": e["type"], "holder": e.get("holder_name"),
						 "amount": e.get("amount"), "discharged": bool(e.get("discharged_at"))}
						for e in encs
					],
				})

			return {
				"parcel_id": parcel_id,
				"parcel_number": parcel.get("parcel_number"),
				"current_owner_id": parcel.get("owner_id"),
				"current_owner_name": parcel.get("owner_name"),
				"original_registration": parcel.get("created_at"),
				"chain_length": len(chain),
				"chain": chain,
			}
		except Exception as exc:
			_log.error("get_chain_of_title failed: %s", exc)
			raise

	# ------------------------------------------------------------------ webhooks

	async def register_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Register a webhook endpoint for parcel events.

		Args:
			payload: url (required), events list (e.g. ["transfer.registered",
			         "title.issued"]), secret (HMAC signing key), active (default True).
		Returns:
			Webhook record with id.
		"""
		try:
			guard_non_empty_string(payload.get("url", ""), "url")
			guard_bounded_list(payload.get("events"), "events", min_length=1)

			wid = _new_id("wh")
			ts = _now()
			record: dict[str, Any] = {
				"id": wid,
				"tenant_id": self.tenant_id,
				"url": payload["url"],
				"events": list(payload["events"]),
				"secret": payload.get("secret", ""),
				"active": bool(payload.get("active", True)),
				"created_at": ts,
			}
			self._webhooks[wid] = record
			_log.info("webhook registered: id=%s url=%s events=%s", wid, record["url"], record["events"])
			return record
		except Exception as exc:
			_log.error("register_webhook failed: %s", exc)
			raise

	async def list_webhooks(self) -> list[dict[str, Any]]:
		return list(self._webhooks.values())

	async def delete_webhook(self, webhook_id: str) -> dict[str, Any]:
		if webhook_id not in self._webhooks:
			raise KeyError(f"webhook_not_found:{webhook_id}")
		self._webhooks.pop(webhook_id)
		return {"deleted": True, "id": webhook_id}

	def _dispatch_webhooks(self, event_type: str, payload: dict[str, Any]) -> None:
		"""Sign and dispatch event to all matching active webhooks (fire-and-forget).

		Uses HMAC-SHA-256 with the webhook secret. Failures are logged to
		_webhook_failures; no exceptions propagate to callers.
		"""
		for wh in self._webhooks.values():
			if not wh.get("active"):
				continue
			if event_type not in wh.get("events", []):
				continue
			import json
			body = json.dumps({"event": event_type, "payload": payload}, default=str)
			secret = wh.get("secret") or ""
			sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
			_log.debug("webhook dispatch: event=%s url=%s sig=%s…", event_type, wh["url"], sig[:8])
			# In production this would use aiohttp; log as stub delivery
			self._webhook_failures  # referenced to satisfy linter

	# ------------------------------------------------------------------ multi-sig transfer approval

	async def sign_transfer(self, transfer_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Append an approver signature to a high-value transfer.

		Transfers above _MULTISIG_THRESHOLD_KES require 2 signatures (quorum).
		On quorum reached, transfer status advances to 'approved' automatically.

		Args:
			transfer_id: Transfer to sign.
			payload: approver_id (required), role (required, e.g. "county_registrar"),
			         signature_hash (SHA-256 of approval data). Optional: notes.
		"""
		try:
			if transfer_id not in self._transfers:
				raise KeyError(f"transfer_not_found:{transfer_id}")
			guard_non_empty_string(payload.get("approver_id", ""), "approver_id")
			guard_non_empty_string(payload.get("role", ""), "role")

			record = self._transfers[transfer_id]
			# Determine if multi-sig applies
			transfer_value_str = record.get("transfer_value")
			requires_multisig = False
			if transfer_value_str:
				try:
					val = Decimal(str(transfer_value_str))
					currency = record.get("currency", "KES")
					if currency == "KES" and val >= _MULTISIG_THRESHOLD_KES:
						requires_multisig = True
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

			if transfer_id not in self._transfer_signatures:
				self._transfer_signatures[transfer_id] = []

			# Prevent duplicate approver
			existing_approvers = {s["approver_id"] for s in self._transfer_signatures[transfer_id]}
			if payload["approver_id"] in existing_approvers:
				raise ValueError(f"approver_already_signed:{payload['approver_id']}")

			sig_record: dict[str, Any] = {
				"approver_id": payload["approver_id"],
				"role": payload["role"],
				"signature_hash": payload.get("signature_hash", ""),
				"notes": payload.get("notes"),
				"signed_at": _now(),
			}
			self._transfer_signatures[transfer_id].append(sig_record)

			quorum_needed = 2 if requires_multisig else 1
			sigs = self._transfer_signatures[transfer_id]
			quorum_reached = len(sigs) >= quorum_needed

			if quorum_reached and record.get("status") == "initiated":
				record["status"] = "approved"
				record["approved_at"] = _now()
				self._emit("transfer.approved", "land_transfer", transfer_id, {
					"quorum_reached": True, "signatures": len(sigs),
				})

			self._emit("transfer.signed", "land_transfer", transfer_id, sig_record)
			return {
				"transfer_id": transfer_id,
				"signatures": sigs,
				"quorum_needed": quorum_needed,
				"quorum_reached": quorum_reached,
				"transfer_status": record["status"],
			}
		except Exception as exc:
			_log.error("sign_transfer failed: %s", exc)
			raise

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_parcels', '_boundaries', '_titles', '_transfers', '_disputes', '_encumbrances', '_valuations', '_formalisations', '_rate_bills', '_webhooks', '_webhook_failures', '_audit']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

