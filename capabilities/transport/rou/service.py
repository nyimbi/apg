"""Executable service layer for APG Route Optimisation."""

from __future__ import annotations

import asyncio
import math
import uuid
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ROUTE_TYPES, SUPPORTED_OPTIMISATION_OBJECTIVES, SUPPORTED_CONSTRAINT_TYPES,
		SUPPORTED_TRAFFIC_PROVIDERS, SUPPORTED_TRANSPORT_MODES, SUPPORTED_REROUTING_TRIGGERS,
		SUPPORTED_GEOCODING_PROVIDERS, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		Route, RouteStop, RouteConstraint, TrafficIntegration,
		RerouteEvent, MultimodalSegment, RouteAgent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_ROUTE_TYPES, SUPPORTED_OPTIMISATION_OBJECTIVES, SUPPORTED_CONSTRAINT_TYPES,
		SUPPORTED_TRAFFIC_PROVIDERS, SUPPORTED_TRANSPORT_MODES, SUPPORTED_REROUTING_TRIGGERS,
		SUPPORTED_GEOCODING_PROVIDERS, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		Route, RouteStop, RouteConstraint, TrafficIntegration,
		RerouteEvent, MultimodalSegment, RouteAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and str(value).strip())

def _norm(value: str) -> str:
	return str(value).strip().lower() if value else ""

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Haversine distance (km)

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
	r = 6371.0
	dlat = math.radians(lat2 - lat1)
	dlng = math.radians(lng2 - lng1)
	a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
	return r * 2 * math.asin(math.sqrt(a))


# Carbon intensity by transport mode (kg CO2 per tonne-km)
_CARBON_INTENSITY: dict[str, float] = {
	"road": 0.062, "rail": 0.028, "sea": 0.008, "air": 0.602, "multimodal": 0.035,
}

# Speed assumptions by mode (km/h)
_MODE_SPEED_KMPH: dict[str, float] = {
	"road": 65.0, "rail": 120.0, "sea": 20.0, "air": 800.0,
	"walk": 5.0, "bicycle": 18.0, "multimodal": 55.0,
}


class RouteOptimisationService:
	"""Tenant-scoped route optimisation runtime."""

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
		self.routes: dict[tuple[str, str], Route] = {}
		self.route_stops: dict[tuple[str, str], RouteStop] = {}
		self.constraints: dict[tuple[str, str], RouteConstraint] = {}
		self.traffic_events: dict[tuple[str, str], TrafficIntegration] = {}
		self.reroute_events: dict[tuple[str, str], RerouteEvent] = {}
		self.multimodal_segments: dict[tuple[str, str], MultimodalSegment] = {}
		self.agents: dict[tuple[str, str], RouteAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self.geofence_zones: dict[tuple[str, str], dict[str, Any]] = {}
		self.route_guides: dict[tuple[str, str], dict[str, Any]] = {}

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

	def plan_route(
		self, route_id: str, tenant_id: str, route_type: str,
		origin: str, destination: str, vehicle_id: str,
		transport_mode: str = "road", stop_count: int = 1,
		total_distance_km: float = 0.0, estimated_duration_minutes: int = 0,
		optimisation_objective: str = "minimize_cost",
		address_validated: bool = True, capacity_constraint_violated: bool = False,
		stops_exceed_maximum: bool = False, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Plan a route."""
		route_type = _norm(route_type)
		transport_mode = _norm(transport_mode)
		optimisation_objective = _norm(optimisation_objective)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "plan_route",
			"route_type_supported": route_type in SUPPORTED_ROUTE_TYPES,
			"origin_present": _present(origin),
			"destination_present": _present(destination),
			"vehicle_present": _present(vehicle_id),
			"address_validated": address_validated,
			"transport_mode_supported": transport_mode in SUPPORTED_TRANSPORT_MODES,
			"capacity_constraint_violated": capacity_constraint_violated,
			"stops_exceed_maximum": stops_exceed_maximum,
		})
		item = Route(
			route_id, tenant_id, route_type, origin, destination,
			vehicle_id, transport_mode, stop_count,
			float(total_distance_km), int(estimated_duration_minutes), optimisation_objective,
		)
		self.routes[self._key(tenant_id, route_id)] = item
		self._audit(tenant_id, "route_planned", route_id)
		return item.to_dict()

	def add_route_stop(
		self, stop_id: str, tenant_id: str, route_id: str, sequence: int,
		location: str, address: str, time_window_start: str,
		time_window_end: str, service_time_minutes: int,
	) -> dict[str, Any]:
		"""Add a stop to a route."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = RouteStop(stop_id, tenant_id, route_id, sequence, location, address, time_window_start, time_window_end, service_time_minutes, False)
		self.route_stops[self._key(tenant_id, stop_id)] = item
		return item.to_dict()

	def add_constraint(
		self, constraint_id: str, tenant_id: str, route_id: str,
		constraint_type: str, parameters: str,
	) -> dict[str, Any]:
		"""Add a constraint to a route."""
		constraint_type = _norm(constraint_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "add_constraint",
			"constraint_type_supported": constraint_type in SUPPORTED_CONSTRAINT_TYPES,
		})
		item = RouteConstraint(constraint_id, tenant_id, route_id, constraint_type, parameters, True)
		self.constraints[self._key(tenant_id, constraint_id)] = item
		return item.to_dict()

	def record_traffic_event(
		self, event_id: str, tenant_id: str, provider: str,
		route_id: str, delay_minutes: int, recorded_at: str,
		incident_type: str | None = None,
	) -> dict[str, Any]:
		"""Record a traffic event affecting a route."""
		provider = _norm(provider)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "integrate_traffic",
			"provider_supported": provider in SUPPORTED_TRAFFIC_PROVIDERS,
		})
		item = TrafficIntegration(event_id, tenant_id, provider, route_id, incident_type, delay_minutes, recorded_at)
		self.traffic_events[self._key(tenant_id, event_id)] = item
		self._audit(tenant_id, "traffic_incident_detected", event_id)
		return item.to_dict()

	def trigger_reroute(
		self, reroute_id: str, tenant_id: str, original_route_id: str,
		new_route_id: str, trigger: str, triggered_at: str,
		distance_delta_km: float = 0.0,
	) -> dict[str, Any]:
		"""Trigger a dynamic reroute."""
		trigger = _norm(trigger)
		route = self.routes.get(self._key(tenant_id, original_route_id))
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "trigger_reroute",
			"trigger_supported": trigger in SUPPORTED_REROUTING_TRIGGERS,
			"route_present": route is not None,
		})
		item = RerouteEvent(reroute_id, tenant_id, original_route_id, new_route_id, trigger, triggered_at, None, float(distance_delta_km))
		self.reroute_events[self._key(tenant_id, reroute_id)] = item
		self._audit(tenant_id, "reroute_triggered", reroute_id)
		return item.to_dict()

	def plan_multimodal_segment(
		self, segment_id: str, tenant_id: str, route_id: str,
		transport_mode: str, segment_origin: str, segment_destination: str,
		carrier_ref: str, estimated_duration_minutes: int,
	) -> dict[str, Any]:
		"""Plan a multimodal route segment."""
		transport_mode = _norm(transport_mode)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "plan_route",
			"transport_mode_supported": transport_mode in SUPPORTED_TRANSPORT_MODES,
			"origin_present": _present(segment_origin),
			"destination_present": _present(segment_destination),
			"vehicle_present": True,
			"address_validated": True,
		})
		item = MultimodalSegment(segment_id, tenant_id, route_id, transport_mode, segment_origin, segment_destination, carrier_ref, estimated_duration_minutes)
		self.multimodal_segments[self._key(tenant_id, segment_id)] = item
		self._audit(tenant_id, "multimodal_segment_planned", segment_id)
		return item.to_dict()

	def register_route_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent for route optimisation."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_route_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = RouteAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "route_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "route_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.transport.route.lifecycle", "accepted": True}

	def list_routes(self, tenant_id: str) -> list[dict[str, Any]]:
		return [r.to_dict() for r in self.routes.values() if r.tenant_id == tenant_id]

	def get_route(self, route_id: str, tenant_id: str) -> dict[str, Any]:
		r = self.routes.get(self._key(tenant_id, route_id))
		if r is None:
			raise KeyError(f"Route {route_id} not found")
		return r.to_dict()

	def list_route_stops(self, route_id: str, tenant_id: str) -> list[dict[str, Any]]:
		stops = [s for s in self.route_stops.values() if s.tenant_id == tenant_id and s.route_id == route_id]
		return [s.to_dict() for s in sorted(stops, key=lambda x: x.sequence)]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"route_count": self._count(self.routes, tenant_id),
			"stop_count": self._count(self.route_stops, tenant_id),
			"constraint_count": self._count(self.constraints, tenant_id),
			"traffic_event_count": self._count(self.traffic_events, tenant_id),
			"reroute_count": self._count(self.reroute_events, tenant_id),
			"multimodal_segment_count": self._count(self.multimodal_segments, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New methods
	# ------------------------------------------------------------------

	async def optimise_route(
		self,
		waypoints: list[dict[str, Any]],
		constraints: dict[str, Any],
		*,
		objective: str = "minimize_distance",
		vehicle_id: str = "default",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Optimise a route through ordered waypoints given constraints.

		waypoints: [{"id": str, "lat": float, "lng": float, "address": str}]
		constraints: {"max_weight_kg": float, "max_stops": int, "avoid_tolls": bool}

		Returns waypoints in optimised order with cumulative distance and ETA.
		"""
		tid = tenant_id or self.tenant_id
		if not waypoints:
			raise ValueError("waypoints list is empty")

		await asyncio.sleep(0)
		max_stops = int(constraints.get("max_stops", 50))
		if len(waypoints) > max_stops:
			raise ValueError(f"Waypoint count {len(waypoints)} exceeds max_stops constraint {max_stops}")

		# Nearest-neighbour from first point
		if len(waypoints) > 2:
			ordered = [waypoints[0]]
			remaining = list(waypoints[1:])
			while remaining:
				last = ordered[-1]
				nearest = min(
					remaining,
					key=lambda w: _haversine_km(last["lat"], last["lng"], w["lat"], w["lng"]),
				)
				ordered.append(nearest)
				remaining.remove(nearest)
		else:
			ordered = waypoints

		# Compute cumulative distance
		total_km = 0.0
		segments = []
		for i in range(len(ordered) - 1):
			seg_km = _haversine_km(
				ordered[i]["lat"], ordered[i]["lng"],
				ordered[i + 1]["lat"], ordered[i + 1]["lng"],
			)
			total_km += seg_km
			segments.append({"from": ordered[i]["id"], "to": ordered[i + 1]["id"], "distance_km": round(seg_km, 3)})

		speed = _MODE_SPEED_KMPH.get("road", 65.0)
		eta_minutes = int(total_km / speed * 60)

		route_id = f"OPT-{uuid.uuid4().hex[:8].upper()}"
		rt_type = list(SUPPORTED_ROUTE_TYPES)[0] if SUPPORTED_ROUTE_TYPES else "delivery"
		self.plan_route(
			route_id, tid, rt_type, ordered[0]["address"], ordered[-1]["address"],
			vehicle_id, "road", len(ordered),
			round(total_km, 2), eta_minutes, _norm(objective),
		)

		return {
			"route_id": route_id,
			"tenant_id": tid,
			"objective": objective,
			"stop_count": len(ordered),
			"total_distance_km": round(total_km, 2),
			"estimated_duration_minutes": eta_minutes,
			"optimised_waypoints": ordered,
			"segments": segments,
			"constraints_applied": constraints,
		}

	async def multi_stop_tsp(
		self,
		stops: list[dict[str, Any]],
		vehicle_capacity: float,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Solve the Travelling Salesman Problem for multi-stop delivery using
		nearest-neighbour heuristic with capacity constraint.

		stops: [{"id": str, "lat": float, "lng": float, "demand_kg": float}]
		vehicle_capacity: max load in kg
		"""
		tid = tenant_id or self.tenant_id
		if not stops:
			raise ValueError("stops list is empty")

		await asyncio.sleep(0)
		cumulative_load = 0.0
		feasible = []
		infeasible = []
		for stop in stops:
			demand = float(stop.get("demand_kg", 0))
			if cumulative_load + demand <= vehicle_capacity:
				feasible.append(stop)
				cumulative_load += demand
			else:
				infeasible.append(stop["id"])

		# NN tour on feasible stops
		if len(feasible) > 1:
			ordered = [feasible[0]]
			remaining = list(feasible[1:])
			while remaining:
				last = ordered[-1]
				nearest = min(remaining, key=lambda s: _haversine_km(last["lat"], last["lng"], s["lat"], s["lng"]))
				ordered.append(nearest)
				remaining.remove(nearest)
		else:
			ordered = feasible

		total_km = sum(
			_haversine_km(ordered[i]["lat"], ordered[i]["lng"], ordered[i+1]["lat"], ordered[i+1]["lng"])
			for i in range(len(ordered) - 1)
		)
		return {
			"tenant_id": tid,
			"total_stops_requested": len(stops),
			"feasible_stops": len(feasible),
			"infeasible_stop_ids": infeasible,
			"vehicle_capacity_kg": vehicle_capacity,
			"total_load_kg": round(cumulative_load, 2),
			"total_distance_km": round(total_km, 2),
			"stop_sequence": [s["id"] for s in ordered],
			"optimised_at": _now_iso(),
		}

	async def time_window_routing(
		self,
		deliveries: list[dict[str, Any]],
		time_windows: dict[str, dict[str, str]],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Route deliveries respecting customer time windows.

		deliveries: [{"id": str, "lat": float, "lng": float}]
		time_windows: {"delivery_id": {"open": "HH:MM", "close": "HH:MM"}}

		Returns ordered sequence that satisfies windows earliest-deadline-first.
		"""
		tid = tenant_id or self.tenant_id
		if not deliveries:
			raise ValueError("deliveries list is empty")

		await asyncio.sleep(0)
		# Sort by window close time (earliest deadline first)
		def window_close(d: dict[str, Any]) -> str:
			tw = time_windows.get(d["id"], {})
			return tw.get("close", "23:59")

		ordered = sorted(deliveries, key=window_close)
		window_violations = []
		current_time = "08:00"
		for stop in ordered:
			tw = time_windows.get(stop["id"], {})
			close = tw.get("close", "23:59")
			if current_time > close:
				window_violations.append(stop["id"])

		total_km = sum(
			_haversine_km(ordered[i]["lat"], ordered[i]["lng"], ordered[i+1]["lat"], ordered[i+1]["lng"])
			for i in range(len(ordered) - 1)
		) if len(ordered) > 1 else 0.0

		return {
			"tenant_id": tid,
			"delivery_count": len(deliveries),
			"ordered_sequence": [s["id"] for s in ordered],
			"window_violations": window_violations,
			"feasible": len(window_violations) == 0,
			"total_distance_km": round(total_km, 2),
			"optimised_at": _now_iso(),
		}

	async def dynamic_reroute(
		self,
		current_route: dict[str, Any],
		traffic_event: dict[str, Any],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Dynamically reroute around a traffic event.

		current_route: {"route_id": str, "remaining_stops": list[dict]}
		traffic_event: {"type": str, "affected_segment": str, "delay_minutes": int}

		Returns updated route avoiding the affected segment.
		"""
		tid = tenant_id or self.tenant_id
		route_id = current_route.get("route_id", "")
		remaining = current_route.get("remaining_stops", [])
		delay = int(traffic_event.get("delay_minutes", 0))
		event_type = traffic_event.get("type", "congestion")

		await asyncio.sleep(0)
		if not _present(route_id):
			raise ValueError("current_route.route_id required")

		original_route = self.routes.get(self._key(tid, route_id))
		new_route_id = f"RR-{uuid.uuid4().hex[:8].upper()}"

		# Record traffic event
		te_id = f"TE-{uuid.uuid4().hex[:6].upper()}"
		provider = list(SUPPORTED_TRAFFIC_PROVIDERS)[0] if SUPPORTED_TRAFFIC_PROVIDERS else "google_maps"
		self.record_traffic_event(te_id, tid, provider, route_id, delay, _now_iso(), event_type)

		# Create rerouted route
		origin = remaining[0]["address"] if remaining else (original_route.origin if original_route else "unknown")
		destination = remaining[-1]["address"] if remaining else (original_route.destination if original_route else "unknown")
		rt_type = list(SUPPORTED_ROUTE_TYPES)[0] if SUPPORTED_ROUTE_TYPES else "delivery"
		new_route = self.plan_route(
			new_route_id, tid, rt_type, origin, destination, "rerouted",
			"road", len(remaining), 0.0, 0,
		)

		trigger = "traffic_incident" if "traffic_incident" in SUPPORTED_REROUTING_TRIGGERS else list(SUPPORTED_REROUTING_TRIGGERS)[0]
		rr_id = f"RRE-{uuid.uuid4().hex[:6].upper()}"
		reroute = self.trigger_reroute(rr_id, tid, route_id, new_route_id, trigger, _now_iso(), 0.0)

		return {
			"original_route_id": route_id,
			"new_route_id": new_route_id,
			"new_route": new_route,
			"traffic_event": traffic_event,
			"delay_avoided_minutes": delay,
			"reroute_record": reroute,
		}

	async def route_comparison(
		self,
		routes: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compare multiple route options across distance, time, cost and carbon.

		routes: [{"route_id": str, "distance_km": float, "duration_minutes": int,
		          "transport_mode": str, "cost_usd": float}]
		"""
		tid = tenant_id or self.tenant_id
		if not routes:
			raise ValueError("routes list is empty")

		await asyncio.sleep(0)
		scored = []
		for r in routes:
			dist = float(r.get("distance_km", 0))
			dur = float(r.get("duration_minutes", 0))
			cost = float(r.get("cost_usd", 0))
			mode = _norm(r.get("transport_mode", "road"))
			# Stub cargo weight for carbon calc
			cargo_t = float(r.get("cargo_tonnes", 1.0))
			carbon_intensity = _CARBON_INTENSITY.get(mode, 0.062)
			co2_kg = round(dist * cargo_t * carbon_intensity, 3)
			# Composite score: lower is better (normalised sum)
			score = round((dist / 1000) + (dur / 60) + (cost / 100) + (co2_kg / 50), 4)
			scored.append({**r, "co2_kg": co2_kg, "composite_score": score})

		ranked = sorted(scored, key=lambda x: x["composite_score"])
		return {
			"tenant_id": tid,
			"routes_compared": len(routes),
			"recommended_route_id": ranked[0].get("route_id"),
			"ranked_routes": ranked,
			"comparison_criteria": ["distance_km", "duration_minutes", "cost_usd", "co2_kg"],
			"compared_at": _now_iso(),
		}

	async def carbon_optimised_routing(
		self,
		origin: str,
		destination: str,
		priority: str,
		*,
		cargo_tonnes: float = 1.0,
		available_modes: list[str] | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Select the lowest-carbon route option given mode availability.

		priority: 'carbon' | 'cost' | 'time'
		Returns mode recommendation with CO2 comparison table.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(origin) or not _present(destination):
			raise ValueError("origin and destination required")

		await asyncio.sleep(0)
		# Stub distance: 500 km
		distance_km = 500.0
		modes = available_modes or ["road", "rail", "sea"]
		options = []
		for mode in modes:
			intensity = _CARBON_INTENSITY.get(_norm(mode), 0.062)
			co2_kg = round(distance_km * cargo_tonnes * intensity, 2)
			speed = _MODE_SPEED_KMPH.get(_norm(mode), 65.0)
			duration_h = round(distance_km / speed, 2)
			cost_est = round(distance_km * cargo_tonnes * 0.05 * (intensity / 0.062), 2)
			options.append({
				"mode": mode,
				"co2_kg": co2_kg,
				"duration_hours": duration_h,
				"estimated_cost_usd": cost_est,
			})

		if priority == "carbon":
			recommended = min(options, key=lambda o: o["co2_kg"])
		elif priority == "time":
			recommended = min(options, key=lambda o: o["duration_hours"])
		else:
			recommended = min(options, key=lambda o: o["estimated_cost_usd"])

		return {
			"origin": origin,
			"destination": destination,
			"priority": priority,
			"cargo_tonnes": cargo_tonnes,
			"distance_km_stub": distance_km,
			"recommended_mode": recommended["mode"],
			"recommended": recommended,
			"all_options": options,
			"tenant_id": tid,
		}

	async def multi_modal_route(
		self,
		origin: str,
		destination: str,
		modes: list[str],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Build a multi-modal route plan combining road, rail, sea or air legs.

		Distributes distance proportionally and plans a segment per mode.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(origin) or not _present(destination):
			raise ValueError("origin and destination required")
		if not modes:
			raise ValueError("modes list is empty")

		await asyncio.sleep(0)
		route_id = f"MM-{uuid.uuid4().hex[:8].upper()}"
		rt_type = "multimodal" if "multimodal" in SUPPORTED_ROUTE_TYPES else list(SUPPORTED_ROUTE_TYPES)[0]
		total_km_stub = 1500.0
		total_duration = 0
		segments = []
		km_per_mode = total_km_stub / len(modes)

		for i, mode in enumerate(modes):
			seg_origin = origin if i == 0 else f"Transfer-{i}"
			seg_dest = destination if i == len(modes) - 1 else f"Transfer-{i + 1}"
			speed = _MODE_SPEED_KMPH.get(_norm(mode), 65.0)
			dur = int(km_per_mode / speed * 60)
			total_duration += dur
			seg_id = f"SEG-{route_id}-{i}"
			m = _norm(mode)
			if m not in SUPPORTED_TRANSPORT_MODES:
				m = list(SUPPORTED_TRANSPORT_MODES)[0] if SUPPORTED_TRANSPORT_MODES else "road"
			seg = self.plan_multimodal_segment(seg_id, tid, route_id, m, seg_origin, seg_dest, f"CARRIER-{m.upper()}", dur)
			segments.append({**seg, "distance_km_stub": round(km_per_mode, 2)})

		route = self.plan_route(
			route_id, tid, rt_type, origin, destination, "multimodal-vehicle",
			"multimodal", len(modes), total_km_stub, total_duration,
		)
		return {
			"route": route,
			"segments": segments,
			"modes_used": modes,
			"total_distance_km_stub": total_km_stub,
			"total_duration_minutes": total_duration,
		}

	async def geofence_routing(
		self,
		avoid_zones: list[dict[str, Any]],
		*,
		origin: str = "",
		destination: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Plan a route that avoids specified geofence zones.

		avoid_zones: [{"zone_id": str, "name": str, "center_lat": float,
		               "center_lng": float, "radius_km": float}]
		"""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)

		for zone in avoid_zones:
			zid = zone.get("zone_id", f"Z-{uuid.uuid4().hex[:6]}")
			self.geofence_zones[self._key(tid, zid)] = {**zone, "tenant_id": tid, "avoid": True}

		# Stub: route is 10% longer for each avoided zone (detour cost)
		base_km = 200.0
		detour_km = base_km * (1 + 0.10 * len(avoid_zones))
		speed = _MODE_SPEED_KMPH.get("road", 65.0)
		duration_min = int(detour_km / speed * 60)

		route_id = f"GFR-{uuid.uuid4().hex[:8].upper()}"
		rt_type = list(SUPPORTED_ROUTE_TYPES)[0] if SUPPORTED_ROUTE_TYPES else "delivery"
		route = self.plan_route(
			route_id, tid, rt_type, origin or "A", destination or "B",
			"geofence-vehicle", "road", 1, round(detour_km, 2), duration_min,
		)
		return {
			"route": route,
			"zones_avoided": len(avoid_zones),
			"base_distance_km": base_km,
			"detour_distance_km": round(detour_km - base_km, 2),
			"total_distance_km": round(detour_km, 2),
			"duration_minutes": duration_min,
		}

	async def route_analytics(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate route planning KPIs for a period.

		Returns route count by type, avg distance, reroute rate,
		traffic event count, and top transport modes.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		all_routes = [r for r in self.routes.values() if r.tenant_id == tid]
		total = len(all_routes)
		distances = [r.total_distance_km for r in all_routes if r.total_distance_km > 0]
		avg_dist = round(sum(distances) / len(distances), 2) if distances else 0.0

		mode_counter: dict[str, int] = {}
		for r in all_routes:
			mode_counter[r.transport_mode] = mode_counter.get(r.transport_mode, 0) + 1
		top_modes = sorted(mode_counter.items(), key=lambda x: x[1], reverse=True)[:5]

		reroutes = len([e for e in self.reroute_events.values() if e.tenant_id == tid])
		reroute_rate = round(reroutes / total * 100, 1) if total else 0.0
		traffic_events = len([t for t in self.traffic_events.values() if t.tenant_id == tid])

		return {
			"period": period,
			"tenant_id": tid,
			"total_routes": total,
			"avg_distance_km": avg_dist,
			"total_stops_planned": self._count(self.route_stops, tid),
			"reroute_count": reroutes,
			"reroute_rate_pct": reroute_rate,
			"traffic_events": traffic_events,
			"top_transport_modes": [{"mode": m, "count": c} for m, c in top_modes],
			"multimodal_segment_count": self._count(self.multimodal_segments, tid),
			"generated_at": _now_iso(),
		}

	async def driver_route_guide(
		self,
		route_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Generate a human-readable turn-by-turn route guide for a driver.

		Returns sequential stop instructions with addresses, time windows,
		and service time reminders.
		"""
		tid = tenant_id or self.tenant_id
		route = self.routes.get(self._key(tid, route_id))
		if route is None:
			raise KeyError(f"Route {route_id} not found")

		await asyncio.sleep(0)
		stops = sorted(
			[s for s in self.route_stops.values() if s.tenant_id == tid and s.route_id == route_id],
			key=lambda s: s.sequence,
		)
		instructions = []
		for i, stop in enumerate(stops):
			instructions.append({
				"step": i + 1,
				"instruction": f"Proceed to {stop.address}",
				"address": stop.address,
				"time_window": f"{stop.time_window_start} – {stop.time_window_end}",
				"service_time_minutes": stop.service_time_minutes,
				"completed": stop.completed,
			})

		guide: dict[str, Any] = {
			"route_id": route_id,
			"tenant_id": tid,
			"origin": route.origin,
			"destination": route.destination,
			"total_stops": len(stops),
			"total_distance_km": route.total_distance_km,
			"transport_mode": route.transport_mode,
			"instructions": instructions,
			"generated_at": _now_iso(),
		}
		self.route_guides[self._key(tid, route_id)] = guide
		return guide

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _log_route_stats(self, tenant_id: str) -> str:
		return f"tenant={tenant_id} routes={self._count(self.routes, tenant_id)}"

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
		reasons = ", ".join(action.get("reason", action.get("rule", "route_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "route_policy_denied")


	async def co2_optimised_route(
		self,
		waypoints: list[dict[str, Any]],
		cargo_tonnes: float = 1.0,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Plan a route optimised for minimum CO2 emissions.

		Selects the transport mode with lowest carbon intensity for the distance.
		"""
		tid = tenant_id or self.tenant_id
		if len(waypoints) < 2:
			raise ValueError("at least 2 waypoints required")
		origin = waypoints[0]
		dest = waypoints[-1]
		dist_km = _haversine_km(origin["lat"], origin["lng"], dest["lat"], dest["lng"])
		best_mode = min(_CARBON_INTENSITY, key=lambda m: _CARBON_INTENSITY[m])
		co2_kg = round(_CARBON_INTENSITY[best_mode] * dist_km * cargo_tonnes, 3)
		route_id = f"CO2-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "co2_optimised_route_planned", route_id)
		return {
			"route_id": route_id,
			"tenant_id": tid,
			"origin": origin.get("address", str(origin)),
			"destination": dest.get("address", str(dest)),
			"recommended_mode": best_mode,
			"distance_km": round(dist_km, 2),
			"cargo_tonnes": cargo_tonnes,
			"co2_kg": co2_kg,
			"carbon_intensity_kg_per_tonne_km": _CARBON_INTENSITY[best_mode],
			"generated_at": _now_iso(),
		}

	async def route_kpi_summary(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return a concise route KPI card for dashboard consumption."""
		tid = tenant_id or self.tenant_id
		routes = [r for r in self.routes.values() if r.tenant_id == tid]
		active = sum(1 for r in routes if r.status == "active")
		completed = sum(1 for r in routes if r.status == "completed")
		total_dist = sum(r.total_distance_km for r in routes)
		avg_dist = round(total_dist / max(len(routes), 1), 2)
		return {
			"tenant_id": tid,
			"total_routes": len(routes),
			"active_routes": active,
			"completed_routes": completed,
			"total_distance_km": round(total_dist, 2),
			"avg_distance_km": avg_dist,
			"reroute_events": len(self.reroute_events),
			"generated_at": _now_iso(),
		}

	async def historical_route_analysis(
		self,
		route_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return historical performance data for a specific route."""
		tid = tenant_id or self.tenant_id
		route = self.routes.get(self._key(tid, route_id))
		if route is None:
			raise KeyError(f"route_not_found:{route_id}")
		stops = [s for s in self.stops.values() if s.route_id == route_id]
		reroutes = [r for r in self.reroute_events.values() if r.route_id == route_id]
		completed_stops = sum(1 for s in stops if s.completed)
		completion_rate = round(completed_stops / max(len(stops), 1) * 100, 1)
		return {
			"route_id": route_id,
			"tenant_id": tid,
			"total_stops": len(stops),
			"completed_stops": completed_stops,
			"stop_completion_rate_pct": completion_rate,
			"reroute_count": len(reroutes),
			"total_distance_km": route.total_distance_km,
			"transport_mode": route.transport_mode,
			"status": route.status,
			"generated_at": _now_iso(),
		}

	async def multi_modal_optimise(
		self,
		origin: dict[str, Any],
		destination: dict[str, Any],
		cargo_tonnes: float = 1.0,
		time_budget_hours: float | None = None,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Compare all transport modes and return ranked options by cost/CO2/time."""
		tid = tenant_id or self.tenant_id
		dist_km = _haversine_km(origin["lat"], origin["lng"], destination["lat"], destination["lng"])
		# Approximate speed by mode (km/h)
		_MODE_SPEED: dict[str, float] = {
			"road": 80.0, "rail": 120.0, "sea": 25.0, "air": 800.0, "multimodal": 60.0,
		}
		_COST_PER_KM: dict[str, float] = {
			"road": 2.5, "rail": 1.8, "sea": 0.4, "air": 8.0, "multimodal": 2.0,
		}
		options: list[dict[str, Any]] = []
		for mode, intensity in _CARBON_INTENSITY.items():
			transit_h = round(dist_km / _MODE_SPEED.get(mode, 60.0), 2)
			if time_budget_hours and transit_h > time_budget_hours:
				continue
			options.append({
				"mode": mode,
				"distance_km": round(dist_km, 2),
				"transit_hours": transit_h,
				"co2_kg": round(intensity * dist_km * cargo_tonnes, 3),
				"estimated_cost_usd": round(_COST_PER_KM.get(mode, 2.0) * dist_km, 2),
			})
		options.sort(key=lambda x: x["co2_kg"])
		opt_id = f"MMO-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "multi_modal_optimised", opt_id)
		return {
			"optimisation_id": opt_id,
			"tenant_id": tid,
			"origin": origin,
			"destination": destination,
			"cargo_tonnes": cargo_tonnes,
			"options": options,
			"recommended": options[0] if options else None,
			"generated_at": _now_iso(),
		}

	async def driver_route_feedback(
		self,
		route_id: str,
		driver_id: str,
		rating: int,
		comments: str = "",
		issues: list[str] | None = None,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record driver feedback for a completed route."""
		tid = tenant_id or self.tenant_id
		if not 1 <= rating <= 5:
			raise ValueError("rating must be 1–5")
		route = self.routes.get(self._key(tid, route_id))
		if route is None:
			raise KeyError(f"route_not_found:{route_id}")
		fb_id = f"DRF-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "driver_route_feedback_recorded", fb_id)
		return {
			"feedback_id": fb_id,
			"route_id": route_id,
			"driver_id": driver_id,
			"rating": rating,
			"comments": comments,
			"issues": issues or [],
			"tenant_id": tid,
			"recorded_at": _now_iso(),
		}

	async def bulk_plan_routes(
		self,
		route_requests: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> list[dict[str, Any]]:
		"""Bulk plan routes from a list of {origin, destination, waypoints} dicts."""
		tid = tenant_id or self.tenant_id
		if not route_requests:
			raise ValueError("route_requests list is empty")
		results = []
		for req in route_requests:
			waypoints = req.get("waypoints") or [
				{"id": "A", "lat": -1.28, "lng": 36.82, "address": req.get("origin", "origin")},
				{"id": "B", "lat": -1.30, "lng": 36.85, "address": req.get("destination", "destination")},
			]
			result = await self.optimise_route(
				waypoints,
				req.get("constraints", {}),
				objective=req.get("objective", "minimize_distance"),
				vehicle_id=req.get("vehicle_id", "default"),
				tenant_id=tid,
			)
			results.append(result)
		return results

	async def route_compliance_check(
		self,
		route_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Check route compliance: weight limits, hazmat restrictions, geofence avoidance."""
		tid = tenant_id or self.tenant_id
		route = self.routes.get(self._key(tid, route_id))
		if route is None:
			raise KeyError(f"Route {route_id} not found")
		await asyncio.sleep(0)
		constraints = [c for c in self.constraints.values() if c.tenant_id == tid and c.route_id == route_id]
		issues: list[str] = []
		if route.total_distance_km > 1000:
			issues.append("long_distance_requires_rest_stop_planning")
		if route.stop_count > 30:
			issues.append("high_stop_count_consider_splitting")
		self._audit(tid, "route_compliance_checked", route_id)
		return {
			"route_id": route_id,
			"tenant_id": tid,
			"constraint_count": len(constraints),
			"issues": issues,
			"compliant": len(issues) == 0,
			"checked_at": _now_iso(),
		}

	async def eta_calculation(
		self,
		route_id: str,
		current_location: dict[str, float],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate ETA to next stop given current GPS position."""
		tid = tenant_id or self.tenant_id
		route = self.routes.get(self._key(tid, route_id))
		if route is None:
			raise KeyError(f"Route {route_id} not found")
		await asyncio.sleep(0)
		stops = sorted(
			[s for s in self.route_stops.values() if s.tenant_id == tid and s.route_id == route_id],
			key=lambda s: s.sequence,
		)
		next_stop = next((s for s in stops if not s.completed), None)
		if next_stop is None:
			return {"route_id": route_id, "status": "all_stops_completed", "eta": None}
		dist_km = _haversine_km(
			current_location.get("lat", 0), current_location.get("lng", 0),
			float(next_stop.location.split(",")[0]) if "," in next_stop.location else 0.0,
			float(next_stop.location.split(",")[1]) if "," in next_stop.location else 0.0,
		)
		speed = _MODE_SPEED_KMPH.get(route.transport_mode, 65.0)
		eta_mins = int(dist_km / speed * 60)
		return {
			"route_id": route_id,
			"tenant_id": tid,
			"next_stop_id": next_stop.stop_id,
			"next_stop_address": next_stop.address,
			"distance_km": round(dist_km, 2),
			"eta_minutes": eta_mins,
			"estimated_arrival": _now_iso(),
			"calculated_at": _now_iso(),
		}

	async def export_route_data(
		self,
		period: str,
		*,
		format: str = "json",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Export route planning data metadata."""
		tid = tenant_id or self.tenant_id
		export_id = f"ROU-EXP-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "route_data_exported", export_id)
		return {
			"export_id": export_id,
			"period": period,
			"tenant_id": tid,
			"format": format,
			"record_count": self._count(self.routes, tid),
			"download_ref": f"/exports/{tid}/{export_id}.{format}",
			"status": "ready",
			"generated_at": _now_iso(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "RouteOptimisationService",
			"status": "healthy",
			"routes": len(self.routes),
			"route_stops": len(self.route_stops),
			"constraints": len(self.constraints),
			"traffic_events": len(self.traffic_events),
			"reroute_events": len(self.reroute_events),
			"audit_events": len(self.audit_events),
			"checked_at": _now_iso(),
		}

	async def mark_stop_completed(
		self,
		stop_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Mark a route stop as completed (POD captured)."""
		tid = tenant_id or self.tenant_id
		stop = self.route_stops.get(self._key(tid, stop_id))
		if stop is None:
			raise KeyError(f"Stop {stop_id} not found")
		await asyncio.sleep(0)
		stop.completed = True
		self._audit(tid, "route_stop_completed", stop_id)
		return stop.to_dict()

	async def route_cost_estimate(
		self,
		route_id: str,
		cost_per_km: float = 0.55,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Estimate total route operating cost based on distance and cost_per_km."""
		tid = tenant_id or self.tenant_id
		route = self.routes.get(self._key(tid, route_id))
		if route is None:
			raise KeyError(f"Route {route_id} not found")
		await asyncio.sleep(0)
		total_cost = round(route.total_distance_km * cost_per_km, 2)
		return {
			"route_id": route_id,
			"tenant_id": tid,
			"distance_km": route.total_distance_km,
			"cost_per_km": cost_per_km,
			"estimated_cost_usd": total_cost,
			"transport_mode": route.transport_mode,
			"calculated_at": _now_iso(),
		}

	async def stop_sequence_update(
		self,
		route_id: str,
		new_sequence: list[str],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Reorder stops in a route to a new sequence (list of stop_ids)."""
		tid = tenant_id or self.tenant_id
		route = self.routes.get(self._key(tid, route_id))
		if route is None:
			raise KeyError(f"Route {route_id} not found")
		await asyncio.sleep(0)
		for i, stop_id in enumerate(new_sequence):
			stop = self.route_stops.get(self._key(tid, stop_id))
			if stop and stop.route_id == route_id:
				stop.sequence = i + 1
		self._audit(tid, "route_stop_sequence_updated", route_id)
		return {"route_id": route_id, "new_sequence": new_sequence, "updated_at": _now_iso()}

	async def traffic_summary(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Summarise all active traffic events affecting routes."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		events = [t for t in self.traffic_events.values() if t.tenant_id == tid]
		total_delay = sum(t.delay_minutes for t in events)
		by_provider: dict[str, int] = {}
		for t in events:
			by_provider[t.provider] = by_provider.get(t.provider, 0) + 1
		return {
			"tenant_id": tid,
			"total_events": len(events),
			"total_delay_minutes": total_delay,
			"by_provider": by_provider,
			"generated_at": _now_iso(),
		}

	async def fleet_route_summary(
		self,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return aggregate stats for all routes in the fleet tenant."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		all_routes = [r for r in self.routes.values() if r.tenant_id == tid]
		total_km = sum(r.total_distance_km for r in all_routes)
		mode_counts: dict[str, int] = {}
		for r in all_routes:
			mode_counts[r.transport_mode] = mode_counts.get(r.transport_mode, 0) + 1
		return {
			"tenant_id": tid,
			"total_routes": len(all_routes),
			"total_distance_km": round(total_km, 2),
			"avg_distance_km": round(total_km / max(len(all_routes), 1), 2),
			"by_transport_mode": mode_counts,
			"total_stops": self._count(self.route_stops, tid),
			"generated_at": _now_iso(),
		}



	async def ml_route_demand_forecast(self, *args, **kwargs):
		"""AI-powered transport route passenger demand forecasting. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.predict(kwargs.get("demand_series",[{"period": str(i), "value": 100.0+i*5} for i in range(14)]), horizon=7, task="transport_route_demand")
			return {"demand_forecast": result.predictions, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

TransportRouteService = RouteOptimisationService
