"""Land Management service — agr_lnd."""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_lnd"


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

	def __init__(self, tenant_id: str = "default") -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		self._parcels: dict[str, dict[str, Any]] = {}
		self._boundaries: dict[str, dict[str, Any]] = {}
		self._titles: dict[str, dict[str, Any]] = {}
		self._transfers: dict[str, dict[str, Any]] = {}
		self._audit: list[dict[str, Any]] = []

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
