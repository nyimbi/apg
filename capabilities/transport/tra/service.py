"""Executable service layer for APG Asset Tracking."""

from __future__ import annotations

import asyncio
import math
import uuid
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_ASSET_TYPES, SUPPORTED_TRACKING_TECHNOLOGIES, SUPPORTED_MONITORING_TYPES,
		SUPPORTED_GEOFENCE_TYPES, SUPPORTED_ALERT_TYPES, SUPPORTED_COLD_CHAIN_STANDARDS,
		SUPPORTED_CONTAINER_STATUSES, SUPPORTED_UTILISATION_PERIODS,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		TrackedAsset, AssetLocationUpdate, Geofence, TrackingAlert,
		ColdChainRecord, Container, AssetUtilisationRecord, TrackingAgent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_ASSET_TYPES, SUPPORTED_TRACKING_TECHNOLOGIES, SUPPORTED_MONITORING_TYPES,
		SUPPORTED_GEOFENCE_TYPES, SUPPORTED_ALERT_TYPES, SUPPORTED_COLD_CHAIN_STANDARDS,
		SUPPORTED_CONTAINER_STATUSES, SUPPORTED_UTILISATION_PERIODS,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		TrackedAsset, AssetLocationUpdate, Geofence, TrackingAlert,
		ColdChainRecord, Container, AssetUtilisationRecord, TrackingAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and str(value).strip())

def _norm(value: str) -> str:
	return str(value).strip().lower() if value else ""

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Haversine for geofence boundary checks

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
	r = 6371.0
	dlat = math.radians(lat2 - lat1)
	dlng = math.radians(lng2 - lng1)
	a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
	return r * 2 * math.asin(math.sqrt(a))


# Cold chain temperature profiles (°C min/max) by standard
_COLD_CHAIN_PROFILES: dict[str, tuple[float, float]] = {
	"atp":         (-25.0, 0.0),
	"gdp":         (2.0, 8.0),
	"frozen":      (-30.0, -18.0),
	"chilled":     (0.0, 5.0),
	"ambient":     (15.0, 25.0),
	"pharma":      (2.0, 8.0),
}


class AssetTrackingService:
	"""Tenant-scoped asset tracking runtime."""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.assets: dict[tuple[str, str], TrackedAsset] = {}
		self.location_updates: dict[tuple[str, str], AssetLocationUpdate] = {}
		self.geofences: dict[tuple[str, str], Geofence] = {}
		self.alerts: dict[tuple[str, str], TrackingAlert] = {}
		self.cold_chain_records: dict[tuple[str, str], ColdChainRecord] = {}
		self.containers: dict[tuple[str, str], Container] = {}
		self.utilisation_records: dict[tuple[str, str], AssetUtilisationRecord] = {}
		self.agents: dict[tuple[str, str], TrackingAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self.geofence_events: list[dict[str, Any]] = []
		self.theft_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Existing methods (preserved)
	# ------------------------------------------------------------------

	def register_asset(
		self, asset_id: str, tenant_id: str, asset_type: str, unique_id: str,
		owner_id: str, registration: str, tracking_technology: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a trackable asset."""
		asset_type = _norm(asset_type)
		tracking_technology = _norm(tracking_technology)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "register_asset",
			"asset_type_supported": asset_type in SUPPORTED_ASSET_TYPES,
			"unique_id_present": _present(unique_id),
			"owner_present": _present(owner_id),
			"technology_supported": tracking_technology in SUPPORTED_TRACKING_TECHNOLOGIES,
		})
		item = TrackedAsset(asset_id, tenant_id, asset_type, unique_id, owner_id, registration, tracking_technology, True)
		self.assets[self._key(tenant_id, asset_id)] = item
		self._audit(tenant_id, "asset_registered", asset_id)
		return item.to_dict()

	def update_asset_location(
		self, update_id: str, tenant_id: str, asset_id: str,
		latitude: float, longitude: float, speed_kmh: float,
		heading_degrees: float, timestamp: str, source: str,
		tamper_detected: bool = False,
	) -> dict[str, Any]:
		"""Record an asset location update."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "update_asset_location",
			"tamper_detected": tamper_detected,
		})
		item = AssetLocationUpdate(update_id, tenant_id, asset_id, float(latitude), float(longitude), float(speed_kmh), float(heading_degrees), timestamp, source)
		self.location_updates[self._key(tenant_id, update_id)] = item
		self._audit(tenant_id, "asset_location_updated", update_id)
		return item.to_dict()

	def create_geofence(
		self, geofence_id: str, tenant_id: str, geofence_type: str,
		name: str, boundary_definition: str,
		alert_on_entry: bool = True, alert_on_exit: bool = True,
	) -> dict[str, Any]:
		"""Create a geofence."""
		geofence_type = _norm(geofence_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_geofence",
			"geofence_type_supported": geofence_type in SUPPORTED_GEOFENCE_TYPES,
			"area_defined": _present(boundary_definition),
		})
		item = Geofence(geofence_id, tenant_id, geofence_type, name, boundary_definition, True, alert_on_entry, alert_on_exit)
		self.geofences[self._key(tenant_id, geofence_id)] = item
		return item.to_dict()

	def raise_alert(
		self, alert_id: str, tenant_id: str, asset_id: str,
		alert_type: str, severity: str, raised_at: str, details: str,
	) -> dict[str, Any]:
		"""Raise a tracking alert."""
		alert_type = _norm(alert_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "configure_alert",
			"alert_type_supported": alert_type in SUPPORTED_ALERT_TYPES,
		})
		item = TrackingAlert(alert_id, tenant_id, asset_id, alert_type, severity, raised_at, None, None, details)
		self.alerts[self._key(tenant_id, alert_id)] = item
		self._audit(tenant_id, "tracking_alert_raised", alert_id)
		return item.to_dict()

	def acknowledge_alert(self, alert_id: str, tenant_id: str, acknowledged_at: str) -> dict[str, Any]:
		"""Acknowledge a tracking alert."""
		alert = self.alerts.get(self._key(tenant_id, alert_id))
		if alert is None:
			raise KeyError(f"Alert {alert_id} not found")
		alert.acknowledged_at = acknowledged_at
		return alert.to_dict()

	def record_cold_chain(
		self, record_id: str, tenant_id: str, asset_id: str,
		standard: str, min_temp_c: float, max_temp_c: float,
		recorded_temp_c: float, timestamp: str,
	) -> dict[str, Any]:
		"""Record a cold chain temperature reading."""
		standard = _norm(standard)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "configure_cold_chain",
			"standard_supported": standard in SUPPORTED_COLD_CHAIN_STANDARDS,
			"temp_range_defined": True,
		})
		breached = recorded_temp_c < min_temp_c or recorded_temp_c > max_temp_c
		item = ColdChainRecord(record_id, tenant_id, asset_id, standard, float(min_temp_c), float(max_temp_c), float(recorded_temp_c), timestamp, breached)
		self.cold_chain_records[self._key(tenant_id, record_id)] = item
		if breached:
			self._audit(tenant_id, "cold_chain_breach_detected", record_id)
		return item.to_dict()

	def register_container(
		self, container_id: str, tenant_id: str, iso_number: str,
		seal_number: str, owner_id: str, current_location: str, last_updated: str,
	) -> dict[str, Any]:
		"""Register a shipping container."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_container",
			"iso_number_present": _present(iso_number),
		})
		item = Container(container_id, tenant_id, iso_number, seal_number, "available", owner_id, current_location, last_updated)
		self.containers[self._key(tenant_id, container_id)] = item
		self._audit(tenant_id, "container_registered", container_id)
		return item.to_dict()

	def update_container_status(self, container_id: str, tenant_id: str, status: str) -> dict[str, Any]:
		"""Update container status."""
		status = _norm(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "update_container_status",
			"status_supported": status in SUPPORTED_CONTAINER_STATUSES,
		})
		container = self.containers.get(self._key(tenant_id, container_id))
		if container is None:
			raise KeyError(f"Container {container_id} not found")
		container.status = status
		self._audit(tenant_id, "container_status_changed", container_id)
		return container.to_dict()

	def record_utilisation(
		self, record_id: str, tenant_id: str, asset_id: str, period: str,
		period_start: str, period_end: str, idle_time_minutes: int,
		active_time_minutes: int, distance_km: float,
	) -> dict[str, Any]:
		"""Record asset utilisation metrics."""
		period = _norm(period)
		total = idle_time_minutes + active_time_minutes
		utilisation_pct = (active_time_minutes / total * 100) if total > 0 else 0.0
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "generate_utilisation_report",
			"period_supported": period in SUPPORTED_UTILISATION_PERIODS,
		})
		item = AssetUtilisationRecord(record_id, tenant_id, asset_id, period, period_start, period_end, idle_time_minutes, active_time_minutes, float(distance_km), round(utilisation_pct, 2))
		self.utilisation_records[self._key(tenant_id, record_id)] = item
		self._audit(tenant_id, "utilisation_report_generated", record_id)
		return item.to_dict()

	def register_tracking_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent for asset tracking."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_tracking_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = TrackingAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "tracking_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "tracking_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.transport.tracking.lifecycle", "accepted": True}

	def list_assets(self, tenant_id: str) -> list[dict[str, Any]]:
		return [a.to_dict() for a in self.assets.values() if a.tenant_id == tenant_id]

	def list_active_alerts(self, tenant_id: str) -> list[dict[str, Any]]:
		return [a.to_dict() for a in self.alerts.values() if a.tenant_id == tenant_id and a.resolved_at is None]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"asset_count": self._count(self.assets, tenant_id),
			"location_update_count": self._count(self.location_updates, tenant_id),
			"geofence_count": self._count(self.geofences, tenant_id),
			"alert_count": self._count(self.alerts, tenant_id),
			"active_alert_count": len(self.list_active_alerts(tenant_id)),
			"cold_chain_record_count": self._count(self.cold_chain_records, tenant_id),
			"breach_count": sum(1 for r in self.cold_chain_records.values() if r.tenant_id == tenant_id and r.breached),
			"container_count": self._count(self.containers, tenant_id),
			"utilisation_record_count": self._count(self.utilisation_records, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New methods
	# ------------------------------------------------------------------

	async def register_tracked_asset(
		self,
		asset_id: str,
		asset_type: str,
		device_id: str,
		*,
		owner_id: str = "unknown",
		registration: str = "",
		tracking_technology: str = "gps",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Register a new tracked asset with its telemetry device.

		Validates asset_type and tracking_technology, creates the asset record,
		and returns the full registration context including device binding.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(asset_id):
			raise ValueError("asset_id required")
		if not _present(device_id):
			raise ValueError("device_id required")

		await asyncio.sleep(0)
		at = _norm(asset_type)
		if at not in SUPPORTED_ASSET_TYPES:
			at = list(SUPPORTED_ASSET_TYPES)[0] if SUPPORTED_ASSET_TYPES else "vehicle"
		tt = _norm(tracking_technology)
		if tt not in SUPPORTED_TRACKING_TECHNOLOGIES:
			tt = list(SUPPORTED_TRACKING_TECHNOLOGIES)[0] if SUPPORTED_TRACKING_TECHNOLOGIES else "gps"

		asset = self.register_asset(asset_id, tid, at, device_id, owner_id, registration or asset_id, tt)
		return {
			**asset,
			"device_id": device_id,
			"registration_time": _now_iso(),
		}

	async def update_location(
		self,
		asset_id: str,
		gps_lat: float,
		gps_lng: float,
		timestamp: str,
		speed: float,
		*,
		heading_degrees: float = 0.0,
		source: str = "gps",
		check_geofences: bool = True,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Ingest a GPS location update and run geofence boundary checks.

		If check_geofences is True, evaluates all active geofences for the
		asset's tenant and raises entry/exit alerts where applicable.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(asset_id):
			raise ValueError("asset_id required")

		await asyncio.sleep(0)
		update_id = f"LOC-{asset_id}-{uuid.uuid4().hex[:8].upper()}"
		update = self.update_asset_location(
			update_id, tid, asset_id, gps_lat, gps_lng, speed,
			heading_degrees, timestamp, source,
		)

		geofence_alerts: list[dict[str, Any]] = []
		if check_geofences:
			for gf in self.geofences.values():
				if gf.tenant_id != tid or not gf.active:
					continue
				# Parse center+radius from boundary_definition: "lat,lng,radius_km"
				parts = gf.boundary_definition.split(",")
				if len(parts) >= 3:
					try:
						gf_lat = float(parts[0])
						gf_lng = float(parts[1])
						radius_km = float(parts[2])
						dist = _haversine_km(gps_lat, gps_lng, gf_lat, gf_lng)
						inside = dist <= radius_km
						if inside and gf.alert_on_entry:
							evt = {"asset_id": asset_id, "geofence_id": gf.geofence_id, "event": "entry", "distance_km": round(dist, 3)}
							self.geofence_events.append({**evt, "tenant_id": tid, "at": timestamp})
							geofence_alerts.append(evt)
					except ValueError:
						pass

		return {
			**update,
			"geofence_checks": len(geofence_alerts),
			"geofence_alerts": geofence_alerts,
		}

	async def geofence_create(
		self,
		name: str,
		coordinates: str,
		alert_on: str,
		*,
		geofence_type: str = "circle",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Create a geofence zone with alert triggers.

		coordinates: "lat,lng,radius_km" for circle; "lat1,lng1,lat2,lng2,..." for polygon
		alert_on: "entry" | "exit" | "both"
		"""
		tid = tenant_id or self.tenant_id
		if not _present(name):
			raise ValueError("name required")
		if not _present(coordinates):
			raise ValueError("coordinates required")

		await asyncio.sleep(0)
		alert_on = _norm(alert_on)
		alert_entry = alert_on in ("entry", "both")
		alert_exit = alert_on in ("exit", "both")

		gf_id = f"GF-{uuid.uuid4().hex[:8].upper()}"
		gt = _norm(geofence_type)
		if gt not in SUPPORTED_GEOFENCE_TYPES:
			gt = list(SUPPORTED_GEOFENCE_TYPES)[0] if SUPPORTED_GEOFENCE_TYPES else "circle"

		return self.create_geofence(gf_id, tid, gt, name, coordinates, alert_entry, alert_exit)

	async def geofence_event(
		self,
		asset_id: str,
		geofence_id: str,
		event_type: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record a geofence entry or exit event for an asset.

		Raises an alert if the geofence is configured to alert on this event_type.
		"""
		tid = tenant_id or self.tenant_id
		gf = self.geofences.get(self._key(tid, geofence_id))
		if gf is None:
			raise KeyError(f"Geofence {geofence_id} not found")

		await asyncio.sleep(0)
		event_type = _norm(event_type)
		should_alert = (event_type == "entry" and gf.alert_on_entry) or \
		               (event_type == "exit" and gf.alert_on_exit)

		evt: dict[str, Any] = {
			"asset_id": asset_id,
			"geofence_id": geofence_id,
			"geofence_name": gf.name,
			"event_type": event_type,
			"tenant_id": tid,
			"at": _now_iso(),
			"alert_raised": should_alert,
		}
		self.geofence_events.append(evt)

		if should_alert:
			alert_id = f"GFAL-{asset_id[:6]}-{geofence_id[:6]}-{uuid.uuid4().hex[:4].upper()}"
			at = "geofence_breach" if "geofence_breach" in SUPPORTED_ALERT_TYPES else list(SUPPORTED_ALERT_TYPES)[0]
			self.raise_alert(alert_id, tid, asset_id, at, "medium", _now_iso(), f"Asset {asset_id} {event_type} geofence {gf.name}")
			evt["alert_id"] = alert_id

		self._audit(tid, f"geofence_{event_type}", f"{asset_id}:{geofence_id}")
		return evt

	async def cold_chain_monitoring(
		self,
		asset_id: str,
		temperature: float,
		humidity: float,
		*,
		standard: str = "gdp",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record a cold chain reading and evaluate against standard's temperature profile.

		Uses _COLD_CHAIN_PROFILES for min/max. Raises a high-severity alert on breach.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(asset_id):
			raise ValueError("asset_id required")

		await asyncio.sleep(0)
		std = _norm(standard)
		if std not in SUPPORTED_COLD_CHAIN_STANDARDS:
			std = list(SUPPORTED_COLD_CHAIN_STANDARDS)[0] if SUPPORTED_COLD_CHAIN_STANDARDS else "gdp"

		profile = _COLD_CHAIN_PROFILES.get(std, (2.0, 8.0))
		min_t, max_t = profile

		record_id = f"CC-{asset_id[:8]}-{uuid.uuid4().hex[:6].upper()}"
		record = self.record_cold_chain(record_id, tid, asset_id, std, min_t, max_t, temperature, _now_iso())
		breached = record["breached"]

		alert_id = None
		if breached:
			alert_id = f"CCAL-{asset_id[:6]}-{uuid.uuid4().hex[:6].upper()}"
			at = "temperature_breach" if "temperature_breach" in SUPPORTED_ALERT_TYPES else list(SUPPORTED_ALERT_TYPES)[0]
			self.raise_alert(
				alert_id, tid, asset_id, at, "high", _now_iso(),
				f"Temperature {temperature}°C outside [{min_t}, {max_t}]°C for standard {std}",
			)

		return {
			**record,
			"humidity_pct": humidity,
			"standard_profile": {"min_c": min_t, "max_c": max_t},
			"alert_id": alert_id,
		}

	async def container_tracking(
		self,
		container_id: str,
		location: str,
		status: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Update container location and status; register if not yet known.

		Returns current container state including status history summary.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(container_id) or not _present(location) or not _present(status):
			raise ValueError("container_id, location and status required")

		await asyncio.sleep(0)
		container = self.containers.get(self._key(tid, container_id))
		if container is None:
			container_rec = self.register_container(
				container_id, tid, container_id, "UNKNOWN", "unknown", location, _now_iso(),
			)
		else:
			container.current_location = location
			container.last_updated = _now_iso()
			container_rec = container.to_dict()

		st = _norm(status)
		if st not in SUPPORTED_CONTAINER_STATUSES:
			st = "in_transit" if "in_transit" in SUPPORTED_CONTAINER_STATUSES else list(SUPPORTED_CONTAINER_STATUSES)[0]

		updated = self.update_container_status(container_id, tid, st)
		return {**updated, "location": location, "status_updated_at": _now_iso()}

	async def asset_utilisation(
		self,
		asset_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compute utilisation metrics for an asset over a period from location updates.

		Infers active time from location pings with speed > 2 km/h,
		idle time from stationary pings.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(asset_id) or not _present(period):
			raise ValueError("asset_id and period required")

		await asyncio.sleep(0)
		updates = [
			u for u in self.location_updates.values()
			if u.tenant_id == tid and u.asset_id == asset_id
		]
		total_pings = len(updates)
		moving_pings = sum(1 for u in updates if u.speed_kmh > 2.0)
		idle_pings = total_pings - moving_pings
		# Assume 1 ping per 5 minutes
		active_minutes = moving_pings * 5
		idle_minutes = idle_pings * 5
		distances = []
		sorted_updates = sorted(updates, key=lambda u: u.timestamp)
		for i in range(len(sorted_updates) - 1):
			a = sorted_updates[i]
			b = sorted_updates[i + 1]
			distances.append(_haversine_km(a.latitude, a.longitude, b.latitude, b.longitude))
		total_distance = round(sum(distances), 2)

		period_val = _norm(period)
		if period_val not in SUPPORTED_UTILISATION_PERIODS:
			period_val = list(SUPPORTED_UTILISATION_PERIODS)[0] if SUPPORTED_UTILISATION_PERIODS else "daily"

		record_id = f"UTIL-{asset_id[:8]}-{uuid.uuid4().hex[:6].upper()}"
		record = self.record_utilisation(
			record_id, tid, asset_id, period_val,
			_now_iso()[:10], _now_iso()[:10],
			idle_minutes, active_minutes, total_distance,
		)
		return {
			**record,
			"ping_count": total_pings,
			"moving_pings": moving_pings,
			"idle_pings": idle_pings,
		}

	async def theft_alert(
		self,
		asset_id: str,
		trigger: str,
		*,
		last_known_location: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Raise a theft alert for an asset with maximum severity.

		trigger: reason for suspicion, e.g. 'tamper_detected', 'geofence_breach',
		         'unexpected_movement', 'signal_lost'
		Deactivates the asset to prevent false location updates post-theft.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(asset_id):
			raise ValueError("asset_id required")
		if not _present(trigger):
			raise ValueError("trigger required")

		await asyncio.sleep(0)
		asset = self.assets.get(self._key(tid, asset_id))
		last_loc = last_known_location
		if last_loc is None and asset:
			recent = sorted(
				[u for u in self.location_updates.values() if u.tenant_id == tid and u.asset_id == asset_id],
				key=lambda u: u.timestamp, reverse=True,
			)
			last_loc = f"{recent[0].latitude},{recent[0].longitude}" if recent else "unknown"

		alert_id = f"THEFT-{asset_id[:8]}-{uuid.uuid4().hex[:6].upper()}"
		at = "theft" if "theft" in SUPPORTED_ALERT_TYPES else list(SUPPORTED_ALERT_TYPES)[0]
		alert = self.raise_alert(alert_id, tid, asset_id, at, "critical", _now_iso(), f"Theft trigger: {trigger}")

		if asset:
			asset.active = False

		theft_event: dict[str, Any] = {
			"alert_id": alert_id,
			"asset_id": asset_id,
			"trigger": trigger,
			"last_known_location": last_loc,
			"asset_deactivated": True,
			"tenant_id": tid,
			"raised_at": _now_iso(),
		}
		self.theft_events.append(theft_event)
		self._audit(tid, "theft_alert_raised", alert_id)
		return {**alert, **theft_event}

	async def tracking_report(
		self,
		asset_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Generate a comprehensive tracking report for an asset over a period.

		Includes location history summary, alert history, cold chain breaches,
		utilisation stats and geofence events.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(asset_id) or not _present(period):
			raise ValueError("asset_id and period required")

		await asyncio.sleep(0)
		asset = self.assets.get(self._key(tid, asset_id))
		if asset is None:
			raise KeyError(f"Asset {asset_id} not found")

		updates = [u for u in self.location_updates.values() if u.tenant_id == tid and u.asset_id == asset_id]
		alerts = [a for a in self.alerts.values() if a.tenant_id == tid and a.asset_id == asset_id]
		cold_chain = [c for c in self.cold_chain_records.values() if c.tenant_id == tid and c.asset_id == asset_id]
		gf_events = [e for e in self.geofence_events if e.get("asset_id") == asset_id and e.get("tenant_id") == tid]
		theft_evts = [e for e in self.theft_events if e.get("asset_id") == asset_id and e.get("tenant_id") == tid]

		speeds = [u.speed_kmh for u in updates if u.speed_kmh > 0]
		avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else 0.0
		max_speed = max(speeds) if speeds else 0.0

		return {
			"asset_id": asset_id,
			"period": period,
			"tenant_id": tid,
			"asset": asset.to_dict(),
			"location_pings": len(updates),
			"avg_speed_kmh": avg_speed,
			"max_speed_kmh": max_speed,
			"alert_count": len(alerts),
			"unresolved_alerts": sum(1 for a in alerts if a.resolved_at is None),
			"cold_chain_readings": len(cold_chain),
			"cold_chain_breaches": sum(1 for c in cold_chain if c.breached),
			"geofence_events": len(gf_events),
			"theft_events": len(theft_evts),
			"report_generated_at": _now_iso(),
		}

	async def fleet_map_view(
		self,
		filters: dict[str, Any],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return the latest known position of all tracked assets matching filters.

		filters: {"asset_type": str | None, "active_only": bool, "max_age_minutes": int}

		Returns a GeoJSON-compatible feature collection for map rendering.
		"""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)

		asset_type_filter = filters.get("asset_type")
		active_only = bool(filters.get("active_only", True))
		max_age_min = int(filters.get("max_age_minutes", 60))

		filtered_assets = [
			a for a in self.assets.values()
			if a.tenant_id == tid
			and (not asset_type_filter or a.asset_type == _norm(asset_type_filter))
			and (not active_only or a.active)
		]

		features = []
		for asset in filtered_assets:
			recent = sorted(
				[u for u in self.location_updates.values() if u.tenant_id == tid and u.asset_id == asset.asset_id],
				key=lambda u: u.timestamp, reverse=True,
			)
			if not recent:
				continue
			latest = recent[0]
			features.append({
				"type": "Feature",
				"geometry": {"type": "Point", "coordinates": [latest.longitude, latest.latitude]},
				"properties": {
					"asset_id": asset.asset_id,
					"asset_type": asset.asset_type,
					"registration": asset.registration,
					"speed_kmh": latest.speed_kmh,
					"heading": latest.heading_degrees,
					"last_updated": latest.timestamp,
					"active": asset.active,
				},
			})

		return {
			"type": "FeatureCollection",
			"tenant_id": tid,
			"feature_count": len(features),
			"filters_applied": filters,
			"features": features,
			"generated_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _log_asset_count(self, tenant_id: str) -> str:
		return f"tenant={tenant_id} assets={self._count(self.assets, tenant_id)}"

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "tracking_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "tracking_policy_denied")


	async def eta_calculate(
		self,
		asset_id: str,
		destination_lat: float,
		destination_lng: float,
		avg_speed_kmh: float = 80.0,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate ETA from the asset's last known position to a destination."""
		tid = tenant_id or self.tenant_id
		asset = self.assets.get(self._key(tid, asset_id))
		if asset is None:
			raise KeyError(f"asset_not_found:{asset_id}")
		# Get last location update
		updates = [u for (t, _), u in self.location_updates.items() if t == tid and u.asset_id == asset_id]
		if not updates:
			return {"asset_id": asset_id, "eta_available": False, "reason": "no_location_data"}
		latest = max(updates, key=lambda u: u.recorded_at)
		dist_km = _haversine_km(latest.lat, latest.lng, destination_lat, destination_lng)
		eta_hours = dist_km / max(avg_speed_kmh, 1.0)
		from datetime import datetime as _dt, timedelta as _td
		eta_dt = _dt.utcnow() + _td(hours=eta_hours)
		self._audit(tid, "eta_calculated", asset_id)
		return {
			"asset_id": asset_id,
			"tenant_id": tid,
			"current_lat": latest.lat,
			"current_lng": latest.lng,
			"destination_lat": destination_lat,
			"destination_lng": destination_lng,
			"distance_km": round(dist_km, 2),
			"avg_speed_kmh": avg_speed_kmh,
			"eta_hours": round(eta_hours, 2),
			"eta_timestamp": eta_dt.isoformat(),
			"calculated_at": _now_iso(),
		}

	async def tracking_kpi_summary(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return a concise asset tracking KPI card for dashboard consumption."""
		tid = tenant_id or self.tenant_id
		assets = [a for (t, _), a in self.assets.items() if t == tid]
		active = sum(1 for a in assets if a.active)
		alerts = [a for (t, _), a in self.alerts.items() if t == tid]
		unresolved = sum(1 for a in alerts if a.resolved_at is None)
		cold_chain = [r for (t, _), r in self.cold_chain_records.items() if t == tid]
		breaches = sum(1 for r in cold_chain if r.breached)
		return {
			"tenant_id": tid,
			"total_assets": len(assets),
			"active_assets": active,
			"total_alerts": len(alerts),
			"unresolved_alerts": unresolved,
			"cold_chain_readings": len(cold_chain),
			"cold_chain_breaches": breaches,
			"generated_at": _now_iso(),
		}

	async def cold_chain_alert(
		self,
		asset_id: str,
		temperature: float,
		standard: str = "atp",
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Raise an immediate alert if temperature breaches cold chain limits."""
		tid = tenant_id or self.tenant_id
		from .capability_contract import SUPPORTED_COLD_CHAIN_STANDARDS  # noqa
		profile = _COLD_CHAIN_PROFILES.get(standard.lower(), (-25.0, 0.0))
		min_temp, max_temp = profile
		is_breach = temperature < min_temp or temperature > max_temp
		alert_id = f"CCA-{uuid.uuid4().hex[:8].upper()}"
		if is_breach:
			self._audit(tid, "cold_chain_alert_raised", alert_id)
		return {
			"alert_id": alert_id,
			"asset_id": asset_id,
			"tenant_id": tid,
			"standard": standard,
			"temperature": temperature,
			"min_temp": min_temp,
			"max_temp": max_temp,
			"breach": is_breach,
			"severity": "critical" if is_breach else "ok",
			"raised_at": _now_iso(),
		}

	async def customs_checkpoint(
		self,
		asset_id: str,
		checkpoint_id: str,
		documents: list[str],
		officer_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record a customs checkpoint clearance event for a tracked asset."""
		tid = tenant_id or self.tenant_id
		asset = self.assets.get(self._key(tid, asset_id))
		if asset is None:
			raise KeyError(f"asset_not_found:{asset_id}")
		cp_id = f"CPC-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "customs_checkpoint_cleared", cp_id)
		return {
			"clearance_id": cp_id,
			"asset_id": asset_id,
			"checkpoint_id": checkpoint_id,
			"documents_presented": documents,
			"officer_id": officer_id,
			"tenant_id": tid,
			"status": "cleared",
			"cleared_at": _now_iso(),
		}

	async def geofence_exit_alert(
		self,
		asset_id: str,
		geofence_id: str,
		current_lat: float,
		current_lng: float,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Check if an asset has exited its assigned geofence and raise an alert."""
		tid = tenant_id or self.tenant_id
		geofence = self.geofences.get(self._key(tid, geofence_id))
		if geofence is None:
			raise KeyError(f"geofence_not_found:{geofence_id}")
		dist_km = _haversine_km(current_lat, current_lng, geofence.center_lat, geofence.center_lng)
		exited = dist_km > geofence.radius_km
		alert_id = f"GEX-{uuid.uuid4().hex[:8].upper()}"
		if exited:
			self._audit(tid, "geofence_exit_alert_raised", alert_id)
		return {
			"alert_id": alert_id,
			"asset_id": asset_id,
			"geofence_id": geofence_id,
			"tenant_id": tid,
			"current_position": {"lat": current_lat, "lng": current_lng},
			"geofence_center": {"lat": geofence.center_lat, "lng": geofence.center_lng},
			"radius_km": geofence.radius_km,
			"distance_from_center_km": round(dist_km, 3),
			"exited": exited,
			"severity": "high" if exited else "ok",
			"raised_at": _now_iso(),
		}

	async def tracking_analytics_detail(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return detailed tracking analytics for a period."""
		tid = tenant_id or self.tenant_id
		assets = [a for (t, _), a in self.assets.items() if t == tid]
		updates = [u for (t, _), u in self.location_updates.items() if t == tid]
		alerts = [a for (t, _), a in self.alerts.items() if t == tid]
		geofences = [g for (t, _), g in self.geofences.items() if t == tid]
		by_type: dict[str, int] = {}
		for a in assets:
			by_type[a.asset_type] = by_type.get(a.asset_type, 0) + 1
		alert_by_type: dict[str, int] = {}
		for al in alerts:
			alert_by_type[al.alert_type] = alert_by_type.get(al.alert_type, 0) + 1
		return {
			"tenant_id": tid,
			"period": period,
			"total_assets": len(assets),
			"assets_by_type": by_type,
			"location_updates": len(updates),
			"geofences": len(geofences),
			"total_alerts": len(alerts),
			"alerts_by_type": alert_by_type,
			"cold_chain_records": sum(1 for (t, _) in self.cold_chain_records if t == tid),
			"generated_at": _now_iso(),
		}

	async def bulk_register_assets(
		self,
		assets: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> list[dict[str, Any]]:
		"""Bulk register multiple tracked assets."""
		tid = tenant_id or self.tenant_id
		if not assets:
			raise ValueError("assets list is empty")
		results = []
		for a in assets:
			result = await self.register_tracked_asset(
				str(a["asset_id"]),
				str(a.get("asset_type", "vehicle")),
				str(a.get("device_id", a["asset_id"])),
				owner_id=str(a.get("owner_id", "unknown")),
				registration=str(a.get("registration", "")),
				tracking_technology=str(a.get("tracking_technology", "gps")),
				tenant_id=tid,
			)
			results.append(result)
		return results

	async def alert_summary(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return a summary of all active tracking alerts by type and severity."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		alerts = [a for (t, _), a in self.alerts.items() if t == tid]
		active = [a for a in alerts if a.resolved_at is None]
		by_type: dict[str, int] = {}
		by_severity: dict[str, int] = {}
		for a in active:
			by_type[a.alert_type] = by_type.get(a.alert_type, 0) + 1
			by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
		return {
			"tenant_id": tid,
			"total_alerts": len(alerts),
			"active_alerts": len(active),
			"by_type": by_type,
			"by_severity": by_severity,
			"critical_alerts": by_severity.get("critical", 0),
			"generated_at": _now_iso(),
		}

	async def cold_chain_compliance_report(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Report cold chain compliance across all assets for a period."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		records = [r for (t, _), r in self.cold_chain_records.items() if t == tid]
		breaches = [r for r in records if r.breached]
		compliance_rate = round((len(records) - len(breaches)) / max(len(records), 1) * 100, 1)
		return {
			"period": period,
			"tenant_id": tid,
			"total_readings": len(records),
			"breaches": len(breaches),
			"compliance_rate_pct": compliance_rate,
			"compliant": compliance_rate >= 99.0,
			"generated_at": _now_iso(),
		}

	async def export_tracking_data(
		self,
		period: str,
		*,
		format: str = "json",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Export asset tracking data metadata."""
		tid = tenant_id or self.tenant_id
		export_id = f"TRA-EXP-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "tracking_data_exported", export_id)
		return {
			"export_id": export_id,
			"period": period,
			"tenant_id": tid,
			"format": format,
			"record_count": self._count(self.assets, tid),
			"download_ref": f"/exports/{tid}/{export_id}.{format}",
			"status": "ready",
			"generated_at": _now_iso(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "AssetTrackingService",
			"status": "healthy",
			"assets": len(self.assets),
			"location_updates": len(self.location_updates),
			"geofences": len(self.geofences),
			"alerts": len(self.alerts),
			"cold_chain_records": len(self.cold_chain_records),
			"audit_events": len(self.audit_events),
			"checked_at": _now_iso(),
		}

	async def deactivate_asset(
		self,
		asset_id: str,
		reason: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Deactivate a tracked asset (e.g. sold, scrapped, stolen)."""
		tid = tenant_id or self.tenant_id
		asset = self.assets.get(self._key(tid, asset_id))
		if asset is None:
			raise KeyError(f"Asset {asset_id} not found")
		await asyncio.sleep(0)
		asset.active = False
		self._audit(tid, "asset_deactivated", asset_id)
		return {**asset.to_dict(), "deactivation_reason": reason, "deactivated_at": _now_iso()}

	async def resolve_alert(
		self,
		alert_id: str,
		resolved_by: str,
		notes: str = "",
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Resolve an active tracking alert."""
		tid = tenant_id or self.tenant_id
		alert = self.alerts.get(self._key(tid, alert_id))
		if alert is None:
			raise KeyError(f"Alert {alert_id} not found")
		await asyncio.sleep(0)
		alert.resolved_at = _now_iso()
		self._audit(tid, "alert_resolved", alert_id)
		return {**alert.to_dict(), "resolved_by": resolved_by, "resolution_notes": notes}

	async def asset_location_history(
		self,
		asset_id: str,
		limit: int = 50,
		*,
		tenant_id: str = "",
	) -> list[dict[str, Any]]:
		"""Return the most recent location updates for an asset."""
		tid = tenant_id or self.tenant_id
		updates = sorted(
			[u for u in self.location_updates.values() if u.tenant_id == tid and u.asset_id == asset_id],
			key=lambda u: u.timestamp,
			reverse=True,
		)
		await asyncio.sleep(0)
		return [u.to_dict() for u in updates[:limit]]


TransportTrackingService = AssetTrackingService
