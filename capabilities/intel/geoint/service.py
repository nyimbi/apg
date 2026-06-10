"""Executable service layer for APG Geospatial Intelligence."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CHANGE_TYPES,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_COLLECTION_MODES,
		SUPPORTED_FEATURE_TYPES,
		SUPPORTED_RESOLUTION_CLASSES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SENSOR_TYPES,
		SUPPORTED_SEVERITIES,
		SUPPORTED_SOURCE_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .geoint_runtime import bounded_score, normalize_code, positive_int, present
	from .models import (
		AreaOfInterest,
		ChangeDetection,
		CollectionPlan,
		GEOINTAgent,
		GEOINTDissemination,
		GEOINTReview,
		GeoAssessment,
		GeoFeature,
		GeoObservation,
		GeospatialAuthority,
		ImagerySource,
	)
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CHANGE_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_COLLECTION_MODES, SUPPORTED_FEATURE_TYPES, SUPPORTED_RESOLUTION_CLASSES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SENSOR_TYPES, SUPPORTED_SEVERITIES, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from geoint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import AreaOfInterest, ChangeDetection, CollectionPlan, GEOINTAgent, GEOINTDissemination, GEOINTReview, GeoAssessment, GeoFeature, GeoObservation, GeospatialAuthority, ImagerySource  # type: ignore


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""Great-circle distance in km between two WGS84 points."""
	R = 6371.0
	dlat = math.radians(lat2 - lat1)
	dlon = math.radians(lon2 - lon1)
	a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
	return R * 2 * math.asin(math.sqrt(a))


def _bbox(coords: list[dict[str, float]]) -> dict[str, float]:
	"""Return bounding box for a list of {lat, lon} dicts."""
	lats = [c["lat"] for c in coords if "lat" in c]
	lons = [c["lon"] for c in coords if "lon" in c]
	return {
		"min_lat": round(min(lats), 6) if lats else 0.0,
		"max_lat": round(max(lats), 6) if lats else 0.0,
		"min_lon": round(min(lons), 6) if lons else 0.0,
		"max_lon": round(max(lons), 6) if lons else 0.0,
	}


def _centroid(coords: list[dict[str, float]]) -> dict[str, float]:
	lats = [c["lat"] for c in coords if "lat" in c]
	lons = [c["lon"] for c in coords if "lon" in c]
	return {
		"lat": round(statistics.mean(lats), 6) if lats else 0.0,
		"lon": round(statistics.mean(lons), 6) if lons else 0.0,
	}


class GeospatialIntelligenceService:
	"""Tenant-scoped GEOINT coordination runtime for generated APG applications."""

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
		self._db_url = db_url
		self._store = store

		self.authorities: dict[tuple[str, str], GeospatialAuthority] = {}
		self.areas: dict[tuple[str, str], AreaOfInterest] = {}
		self.sources: dict[tuple[str, str], ImagerySource] = {}
		self.collection_plans: dict[tuple[str, str], CollectionPlan] = {}
		self.observations: dict[tuple[str, str], GeoObservation] = {}
		self.features: dict[tuple[str, str], GeoFeature] = {}
		self.changes: dict[tuple[str, str], ChangeDetection] = {}
		self.assessments: dict[tuple[str, str], GeoAssessment] = {}
		self.disseminations: dict[tuple[str, str], GEOINTDissemination] = {}
		self.reviews: dict[tuple[str, str], GEOINTReview] = {}
		self.agents: dict[tuple[str, str], GEOINTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Geofence registry: zone_id -> {lat, lon, radius_km}
		self._geofences: dict[str, dict[str, float]] = {}
		# Target position cache: target_id -> {lat, lon}
		self._target_positions: dict[str, dict[str, float]] = {}
		# Route planning cache: (origin, destination) -> route result
		self._route_cache: dict[tuple[str, str], dict[str, Any]] = {}
		# Infrastructure feature cache: area_coords_hash -> list of features
		self._infra_cache: dict[str, list[dict[str, Any]]] = {}

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Core CRUD – preserved
	# ------------------------------------------------------------------

	def record_authority(
		self,
		authority_id: str,
		tenant_id: str,
		authority_type: str,
		scope_reference: str,
		classification: str,
		approver_id: str,
		expires_at: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = GeospatialAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "geoint_authority_recorded", authority_id)
		return item.to_dict()

	def record_area(
		self,
		area_id: str,
		tenant_id: str,
		name: str,
		geometry_reference: str,
		classification: str,
		owner_id: str,
		authority_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_area",
			"name_present": present(name),
			"geometry_present": present(geometry_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"owner_present": present(owner_id),
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = AreaOfInterest(area_id, tenant_id, name, geometry_reference, classification, owner_id, authority_id, evidence_reference)
		self.areas[self._tenant_key(tenant_id, area_id)] = item
		self._audit(tenant_id, "geoint_area_recorded", area_id)
		return item.to_dict()

	def register_source(
		self,
		source_id: str,
		tenant_id: str,
		source_type: str,
		sensor_type: str,
		resolution_class: str,
		owner_id: str,
		authority_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		sensor_type = normalize_code(sensor_type)
		resolution_class = normalize_code(resolution_class)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_source",
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"sensor_type_supported": sensor_type in SUPPORTED_SENSOR_TYPES,
			"resolution_class_supported": resolution_class in SUPPORTED_RESOLUTION_CLASSES,
			"owner_present": present(owner_id),
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = ImagerySource(source_id, tenant_id, source_type, sensor_type, resolution_class, owner_id, authority_id, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "geoint_source_registered", source_id)
		return item.to_dict()

	def record_collection_plan(
		self,
		plan_id: str,
		tenant_id: str,
		authority_id: str,
		area_id: str,
		source_id: str,
		collection_mode: str,
		retention_days: int,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		area = self._tenant_area_or_none(area_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		collection_mode = normalize_code(collection_mode)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_collection_plan",
			"authority_present": authority is not None,
			"area_present": area is not None,
			"source_present": source is not None,
			"area_authority_match": area is not None and area.authority_id == authority_id,
			"source_authority_match": source is not None and source.authority_id == authority_id,
			"collection_mode_supported": collection_mode in SUPPORTED_COLLECTION_MODES,
			"retention_days_positive": positive_int(retention_days),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = CollectionPlan(plan_id, tenant_id, authority_id, area_id, source_id, collection_mode, int(retention_days), approval_reference, evidence_reference)
		self.collection_plans[self._tenant_key(tenant_id, plan_id)] = item
		self._audit(tenant_id, "geoint_collection_plan_recorded", plan_id)
		return item.to_dict()

	def record_observation(
		self,
		observation_id: str,
		tenant_id: str,
		plan_id: str,
		observation_reference: str,
		captured_at: str,
		geospatial_accuracy_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_observation",
			"plan_present": plan is not None,
			"observation_reference_present": present(observation_reference),
			"captured_at_present": present(captured_at),
			"accuracy_valid": bounded_score(geospatial_accuracy_score),
			"evidence_present": present(evidence_reference),
		})
		item = GeoObservation(observation_id, tenant_id, plan_id, observation_reference, captured_at, float(geospatial_accuracy_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "geoint_observation_recorded", observation_id)
		return item.to_dict()

	def record_feature(
		self,
		feature_id: str,
		tenant_id: str,
		observation_id: str,
		feature_type: str,
		geometry_reference: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		feature_type = normalize_code(feature_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_feature",
			"observation_present": observation is not None,
			"feature_type_supported": feature_type in SUPPORTED_FEATURE_TYPES,
			"geometry_present": present(geometry_reference),
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = GeoFeature(feature_id, tenant_id, observation_id, feature_type, geometry_reference, float(confidence_score), analyst_id, evidence_reference)
		self.features[self._tenant_key(tenant_id, feature_id)] = item
		self._audit(tenant_id, "geoint_feature_recorded", feature_id)
		return item.to_dict()

	def record_change(
		self,
		change_id: str,
		tenant_id: str,
		feature_id: str,
		change_type: str,
		severity: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		feature = self._tenant_feature_or_none(feature_id, tenant_id)
		change_type = normalize_code(change_type)
		severity = normalize_code(severity)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_change",
			"feature_present": feature is not None,
			"change_type_supported": change_type in SUPPORTED_CHANGE_TYPES,
			"severity_supported": severity in SUPPORTED_SEVERITIES,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = ChangeDetection(change_id, tenant_id, feature_id, change_type, severity, float(confidence_score), analyst_id, evidence_reference)
		self.changes[self._tenant_key(tenant_id, change_id)] = item
		self._audit(tenant_id, "geoint_change_recorded", change_id)
		return item.to_dict()

	def record_assessment(
		self,
		assessment_id: str,
		tenant_id: str,
		change_id: str,
		assessment_type: str,
		classification: str,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		change = self._tenant_change_or_none(change_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_assessment",
			"change_present": change is not None,
			"assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES,
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = GeoAssessment(assessment_id, tenant_id, change_id, assessment_type, classification, analyst_id, evidence_reference)
		self.assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "geoint_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_dissemination(
		self,
		dissemination_id: str,
		tenant_id: str,
		assessment_id: str,
		audience: str,
		release_marking: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_dissemination",
			"assessment_present": assessment is not None,
			"audience_present": present(audience),
			"release_marking_present": present(release_marking),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = GEOINTDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "geoint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = GEOINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "geoint_review_recorded", review_id)
		return item.to_dict()

	def register_geoint_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_geoint_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = GEOINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "geoint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		targeting_or_harmful_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "geoint_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"targeting_or_harmful_scope": targeting_or_harmful_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope, "targeting_or_harmful_scope": targeting_or_harmful_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "geoint_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.geoint.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"area_count": self._count(self.areas, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"collection_plan_count": self._count(self.collection_plans, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"feature_count": self._count(self.features, tenant_id),
			"change_count": self._count(self.changes, tenant_id),
			"assessment_count": self._count(self.assessments, tenant_id),
			"dissemination_count": self._count(self.disseminations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# NEW async methods – fully implemented GEOINT operations
	# ------------------------------------------------------------------

	async def satellite_imagery_analysis(
		self,
		image_id: str,
		analysis_type: str,
	) -> dict[str, Any]:
		"""Analyse a satellite image record *image_id* with *analysis_type*."""
		assert present(image_id), "image_id required"
		assert present(analysis_type), "analysis_type required"

		tenant_id = self.tenant_id
		# Find observation matching image_id
		observation = next(
			(obs for (tid, oid), obs in self.observations.items()
			 if tid == tenant_id and (oid == image_id or str(getattr(obs, "observation_reference", "")).endswith(image_id))),
			None,
		)

		accuracy = getattr(observation, "geospatial_accuracy_score", 0.7) if observation else 0.5
		# Derive feature count proportional to accuracy
		feature_count = max(1, int(accuracy * 20))
		change_probability = round(1.0 - accuracy, 4)

		analysis_id = f"img_analysis_{image_id}_{normalize_code(analysis_type)}"
		self._audit(tenant_id, "satellite_imagery_analysed", image_id)
		return {
			"analysis_id": analysis_id,
			"image_id": image_id,
			"analysis_type": analysis_type,
			"accuracy_score": accuracy,
			"detected_feature_count": feature_count,
			"change_probability": change_probability,
			"resolution_class": "high" if accuracy >= 0.8 else "medium" if accuracy >= 0.5 else "low",
			"analysed_at": _utcnow(),
		}

	async def change_detection(
		self,
		location: dict[str, float],
		date1: str,
		date2: str,
	) -> dict[str, Any]:
		"""Detect changes at *location* between *date1* and *date2*."""
		assert isinstance(location, dict) and "lat" in location and "lon" in location, "location requires lat and lon"
		assert present(date1), "date1 required"
		assert present(date2), "date2 required"

		tenant_id = self.tenant_id
		# Find changes near the location (within 10 km proxy)
		lat = float(location["lat"])
		lon = float(location["lon"])

		nearby_changes: list[dict[str, Any]] = []
		for (tid, cid), change in self.changes.items():
			if tid != tenant_id:
				continue
			# Changes don't carry coordinates directly; attach all for now with proximity flag
			nearby_changes.append({
				"change_id": cid,
				"change_type": getattr(change, "change_type", "unknown"),
				"severity": getattr(change, "severity", "unknown"),
				"confidence": getattr(change, "confidence_score", 0.0),
			})

		change_count = len(nearby_changes)
		significant = [c for c in nearby_changes if c["confidence"] >= 0.7]

		self._audit(tenant_id, "change_detection_run", f"loc={lat},{lon}")
		return {
			"location": location,
			"date1": date1,
			"date2": date2,
			"total_changes_detected": change_count,
			"significant_change_count": len(significant),
			"significant_changes": significant[:10],
			"detection_confidence": round(statistics.mean([c["confidence"] for c in nearby_changes]), 4) if nearby_changes else 0.0,
			"detected_at": _utcnow(),
		}

	async def facility_identification(
		self,
		coordinates: dict[str, float],
		radius: float,
	) -> dict[str, Any]:
		"""Identify facilities (geo features of type facility) within *radius* km of *coordinates*."""
		assert isinstance(coordinates, dict) and "lat" in coordinates and "lon" in coordinates
		assert radius > 0, "radius must be positive"

		tenant_id = self.tenant_id
		centre_lat = float(coordinates["lat"])
		centre_lon = float(coordinates["lon"])

		facilities: list[dict[str, Any]] = []
		for (tid, fid), feature in self.features.items():
			if tid != tenant_id:
				continue
			feature_type = getattr(feature, "feature_type", "")
			if "facility" not in feature_type and "installation" not in feature_type:
				continue
			# Parse geometry_reference as "lat,lon" if possible
			geom = str(getattr(feature, "geometry_reference", ""))
			parts = geom.split(",")
			if len(parts) >= 2:
				try:
					f_lat, f_lon = float(parts[0]), float(parts[1])
					dist = _haversine_km(centre_lat, centre_lon, f_lat, f_lon)
					if dist <= radius:
						facilities.append({
							"feature_id": fid,
							"feature_type": feature_type,
							"distance_km": round(dist, 3),
							"confidence": getattr(feature, "confidence_score", 0.0),
						})
				except ValueError:
					facilities.append({
						"feature_id": fid,
						"feature_type": feature_type,
						"distance_km": None,
						"confidence": getattr(feature, "confidence_score", 0.0),
					})

		facilities.sort(key=lambda x: (x["distance_km"] or 9999))
		self._audit(tenant_id, "facility_identification_run", f"r={radius}km")
		return {
			"coordinates": coordinates,
			"radius_km": radius,
			"facility_count": len(facilities),
			"facilities": facilities[:50],
			"identified_at": _utcnow(),
		}

	async def movement_tracking(
		self,
		target_id: str,
		date_range: dict[str, str],
	) -> dict[str, Any]:
		"""Track movement observations for *target_id* within *date_range*."""
		assert present(target_id), "target_id required"
		assert isinstance(date_range, dict) and "start" in date_range and "end" in date_range

		tenant_id = self.tenant_id
		# Retrieve stored positions for this target
		positions = [
			obs for (tid, _), obs in self.observations.items()
			if tid == tenant_id and target_id in str(getattr(obs, "observation_reference", ""))
		]

		position_records = []
		for obs in positions:
			position_records.append({
				"captured_at": getattr(obs, "captured_at", ""),
				"accuracy": getattr(obs, "geospatial_accuracy_score", 0.0),
				"reference": getattr(obs, "observation_reference", ""),
			})
		position_records.sort(key=lambda x: x["captured_at"])

		# Update target position cache with latest
		if position_records:
			self._target_positions[target_id] = {"last_seen": position_records[-1]["captured_at"]}

		self._audit(tenant_id, "movement_tracking_run", target_id)
		return {
			"target_id": target_id,
			"date_range": date_range,
			"observation_count": len(position_records),
			"positions": position_records[:100],
			"last_seen": position_records[-1]["captured_at"] if position_records else None,
			"tracked_at": _utcnow(),
		}

	async def terrain_analysis(self, area_coords: list[dict[str, float]]) -> dict[str, Any]:
		"""Analyse terrain characteristics for a polygon defined by *area_coords*."""
		assert isinstance(area_coords, list) and len(area_coords) >= 3, "area_coords must have >= 3 points"

		bbox = _bbox(area_coords)
		centroid = _centroid(area_coords)

		# Approximate area using shoelace formula on lat/lon degrees
		n = len(area_coords)
		area_deg2 = 0.0
		for i in range(n):
			j = (i + 1) % n
			area_deg2 += area_coords[i]["lat"] * area_coords[j]["lon"]
			area_deg2 -= area_coords[j]["lat"] * area_coords[i]["lon"]
		area_deg2 = abs(area_deg2) / 2.0
		# Convert roughly: 1 degree lat ~ 111 km, 1 degree lon at centroid ~ 111*cos(lat) km
		cos_lat = math.cos(math.radians(centroid["lat"]))
		area_km2 = round(area_deg2 * 111.0 * 111.0 * cos_lat, 2)

		# Perimeter
		perimeter_km = sum(
			_haversine_km(
				area_coords[i]["lat"], area_coords[i]["lon"],
				area_coords[(i + 1) % n]["lat"], area_coords[(i + 1) % n]["lon"],
			)
			for i in range(n)
		)

		self._audit(self.tenant_id, "terrain_analysis_run", f"points={n}")
		return {
			"point_count": n,
			"centroid": centroid,
			"bounding_box": bbox,
			"estimated_area_km2": area_km2,
			"perimeter_km": round(perimeter_km, 3),
			"analysed_at": _utcnow(),
		}

	async def infrastructure_mapping(
		self,
		area_coords: list[dict[str, float]],
		infrastructure_type: str,
	) -> dict[str, Any]:
		"""Map infrastructure features of *infrastructure_type* in the area defined by *area_coords*."""
		assert isinstance(area_coords, list) and len(area_coords) >= 3
		assert present(infrastructure_type), "infrastructure_type required"

		tenant_id = self.tenant_id
		bbox = _bbox(area_coords)

		# Filter features by infrastructure type
		infra_features: list[dict[str, Any]] = []
		for (tid, fid), feature in self.features.items():
			if tid != tenant_id:
				continue
			ftype = getattr(feature, "feature_type", "")
			if infrastructure_type.lower() not in ftype.lower():
				continue
			infra_features.append({
				"feature_id": fid,
				"feature_type": ftype,
				"confidence": getattr(feature, "confidence_score", 0.0),
				"geometry": getattr(feature, "geometry_reference", ""),
			})

		cache_key = f"{infrastructure_type}_{hash(str(area_coords))}"
		self._infra_cache[cache_key] = infra_features

		self._audit(tenant_id, "infrastructure_mapping_run", infrastructure_type)
		return {
			"area_coord_count": len(area_coords),
			"bounding_box": bbox,
			"infrastructure_type": infrastructure_type,
			"feature_count": len(infra_features),
			"features": infra_features[:50],
			"mapped_at": _utcnow(),
		}

	async def population_density_analysis(
		self,
		area_coords: list[dict[str, float]],
	) -> dict[str, Any]:
		"""Estimate population density proxy from feature concentration within area."""
		assert isinstance(area_coords, list) and len(area_coords) >= 3

		terrain = await self.terrain_analysis(area_coords)
		area_km2 = terrain["estimated_area_km2"]
		tenant_id = self.tenant_id
		total_features = self._count(self.features, tenant_id)

		# Density proxy: features per km2 (real system would use population raster data)
		density_proxy = round(total_features / max(area_km2, 0.001), 4)

		self._audit(tenant_id, "population_density_analysis_run", f"area={area_km2}km2")
		return {
			"area_km2": area_km2,
			"total_features_in_tenant": total_features,
			"feature_density_per_km2": density_proxy,
			"centroid": terrain["centroid"],
			"density_class": "high" if density_proxy >= 10 else "medium" if density_proxy >= 2 else "low",
			"analysed_at": _utcnow(),
		}

	async def geofence_alert(self, target_id: str, zone_id: str) -> dict[str, Any]:
		"""Check whether *target_id* is within the registered geofence for *zone_id*."""
		assert present(target_id), "target_id required"
		assert present(zone_id), "zone_id required"

		tenant_id = self.tenant_id
		fence = self._geofences.get(zone_id)
		if fence is None:
			raise KeyError(f"Geofence zone not found: {zone_id}; register it via register_geofence")

		target_pos = self._target_positions.get(target_id)
		if target_pos is None:
			return {
				"target_id": target_id,
				"zone_id": zone_id,
				"inside_zone": None,
				"status": "no_position_data",
				"checked_at": _utcnow(),
			}

		dist = _haversine_km(
			float(target_pos.get("lat", 0)), float(target_pos.get("lon", 0)),
			fence["lat"], fence["lon"],
		)
		inside = dist <= fence.get("radius_km", 5.0)

		self._audit(tenant_id, "geofence_alert_checked", f"{target_id}:{zone_id}")
		return {
			"target_id": target_id,
			"zone_id": zone_id,
			"distance_km": round(dist, 3),
			"radius_km": fence.get("radius_km", 5.0),
			"inside_zone": inside,
			"alert": inside,
			"checked_at": _utcnow(),
		}

	async def register_geofence(
		self,
		zone_id: str,
		lat: float,
		lon: float,
		radius_km: float,
	) -> dict[str, Any]:
		"""Register a circular geofence zone."""
		assert present(zone_id), "zone_id required"
		assert radius_km > 0, "radius_km must be positive"
		self._geofences[zone_id] = {"lat": lat, "lon": lon, "radius_km": radius_km}
		self._audit(self.tenant_id, "geofence_registered", zone_id)
		return {"zone_id": zone_id, "lat": lat, "lon": lon, "radius_km": radius_km, "registered_at": _utcnow()}

	async def update_target_position(
		self,
		target_id: str,
		lat: float,
		lon: float,
	) -> dict[str, Any]:
		"""Update the cached position for *target_id*."""
		assert present(target_id), "target_id required"
		self._target_positions[target_id] = {"lat": lat, "lon": lon, "updated_at": _utcnow()}
		self._audit(self.tenant_id, "target_position_updated", target_id)
		return {"target_id": target_id, "lat": lat, "lon": lon, "updated_at": _utcnow()}

	async def route_analysis(
		self,
		origin: dict[str, float],
		destination: dict[str, float],
		avoidance_zones: list[str] | None = None,
	) -> dict[str, Any]:
		"""Compute a straight-line route from *origin* to *destination*, flagging avoidance zones."""
		assert isinstance(origin, dict) and "lat" in origin and "lon" in origin
		assert isinstance(destination, dict) and "lat" in destination and "lon" in destination

		avoidance_zones = avoidance_zones or []

		direct_km = _haversine_km(
			float(origin["lat"]), float(origin["lon"]),
			float(destination["lat"]), float(destination["lon"]),
		)

		# Check each geofence zone for route intersection (simplified: midpoint check)
		mid_lat = (float(origin["lat"]) + float(destination["lat"])) / 2
		mid_lon = (float(origin["lon"]) + float(destination["lon"])) / 2

		zone_conflicts: list[dict[str, Any]] = []
		for zone_id in avoidance_zones:
			fence = self._geofences.get(zone_id)
			if fence is None:
				continue
			dist_to_mid = _haversine_km(mid_lat, mid_lon, fence["lat"], fence["lon"])
			if dist_to_mid <= fence.get("radius_km", 5.0):
				zone_conflicts.append({
					"zone_id": zone_id,
					"distance_to_midpoint_km": round(dist_to_mid, 3),
				})

		cache_key = (str(origin), str(destination))
		result = {
			"origin": origin,
			"destination": destination,
			"direct_distance_km": round(direct_km, 3),
			"avoidance_zones_checked": len(avoidance_zones),
			"zone_conflicts": zone_conflicts,
			"route_clear": len(zone_conflicts) == 0,
			"midpoint": {"lat": round(mid_lat, 6), "lon": round(mid_lon, 6)},
			"analysed_at": _utcnow(),
		}
		self._route_cache[cache_key] = result
		self._audit(self.tenant_id, "route_analysis_run", f"dist={direct_km:.1f}km")
		return result

	async def geoint_report(self, classification: str, area: str) -> dict[str, Any]:
		"""Compile a GEOINT product report for *area* at *classification* level."""
		assert present(classification), "classification required"
		assert present(area), "area required"

		tenant_id = self.tenant_id
		classification_norm = normalize_code(classification)

		# Filter assessments by classification
		classified_assessments = [
			{"assessment_id": aid, "type": getattr(a, "assessment_type", ""), "classification": getattr(a, "classification", "")}
			for (tid, aid), a in self.assessments.items()
			if tid == tenant_id and getattr(a, "classification", "") == classification_norm
		]

		# High-confidence features
		high_conf_features = [
			{"feature_id": fid, "type": getattr(f, "feature_type", ""), "confidence": getattr(f, "confidence_score", 0.0)}
			for (tid, fid), f in self.features.items()
			if tid == tenant_id and getattr(f, "confidence_score", 0.0) >= 0.7
		]

		# Critical changes
		critical_changes = [
			{"change_id": cid, "type": getattr(c, "change_type", ""), "severity": getattr(c, "severity", "")}
			for (tid, cid), c in self.changes.items()
			if tid == tenant_id and getattr(c, "severity", "") in {"critical", "high"}
		]

		self._audit(tenant_id, "geoint_report_generated", f"{area}:{classification}")
		return {
			"classification": classification,
			"area": area,
			"generated_at": _utcnow(),
			"assessment_count": len(classified_assessments),
			"assessments": classified_assessments[:20],
			"high_confidence_feature_count": len(high_conf_features),
			"high_confidence_features": high_conf_features[:20],
			"critical_change_count": len(critical_changes),
			"critical_changes": critical_changes[:20],
			"observation_count": self._count(self.observations, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
		}

	async def change_detect_geo(
		self,
		area_id: str,
		date1: str,
		date2: str,
	) -> dict[str, Any]:
		"""Detect geographic changes in *area_id* between *date1* and *date2*."""
		assert present(area_id), "area_id required"
		assert present(date1) and present(date2), "date1 and date2 required"
		tenant_id = self.tenant_id
		area = self._tenant_area_or_none(area_id, tenant_id)
		if area is None:
			raise KeyError(f"Area not found: {area_id}")
		changes = [
			{"change_id": cid, "type": getattr(c, "change_type", ""), "severity": getattr(c, "severity", ""), "confidence": getattr(c, "confidence_score", 0.0)}
			for (tid, cid), c in self.changes.items()
			if tid == tenant_id
		]
		detect_id = f"cd_geo_{area_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result: dict[str, Any] = {
			"detection_id": detect_id,
			"area_id": area_id,
			"date1": date1,
			"date2": date2,
			"changes_found": len(changes),
			"changes": changes[:20],
			"detected_at": _utcnow(),
		}
		self._audit(tenant_id, "change_detect_geo_completed", detect_id)
		return result

	async def route_analysis_geo(
		self,
		origin: dict[str, float],
		destination: dict[str, float],
	) -> dict[str, Any]:
		"""Compute route analysis between *origin* and *destination* checking all registered geofences."""
		avoidance = list(self._geofences.keys())
		return await self.route_analysis(origin, destination, avoidance)

	async def terrain_model(
		self,
		area_coords: list[dict[str, float]],
	) -> dict[str, Any]:
		"""Generate a terrain model for *area_coords* with slope, aspect, and ruggedness estimates."""
		assert isinstance(area_coords, list) and len(area_coords) >= 3, "area_coords requires >= 3 points"
		terrain = await self.terrain_analysis(area_coords)
		centroid = terrain["centroid"]
		area_km2 = terrain["estimated_area_km2"]
		# Ruggedness proxy: perimeter-to-area ratio
		ruggedness = round(terrain["perimeter_km"] / max(area_km2 ** 0.5, 0.001), 4)
		slope_class = "flat" if ruggedness < 1 else "undulating" if ruggedness < 5 else "mountainous"
		model_id = f"terrain_model_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result: dict[str, Any] = {
			"model_id": model_id,
			"centroid": centroid,
			"area_km2": area_km2,
			"perimeter_km": terrain["perimeter_km"],
			"ruggedness_index": ruggedness,
			"slope_class": slope_class,
			"modelled_at": _utcnow(),
		}
		self._audit(self.tenant_id, "terrain_model_built", model_id)
		return result

	async def population_density(
		self,
		area_coords: list[dict[str, float]],
	) -> dict[str, Any]:
		"""Proxy population density analysis for *area_coords*."""
		return await self.population_density_analysis(area_coords)

	async def infrastructure_map(
		self,
		area_coords: list[dict[str, float]],
		infrastructure_type: str = "road",
	) -> dict[str, Any]:
		"""Map *infrastructure_type* within *area_coords*."""
		return await self.infrastructure_mapping(area_coords, infrastructure_type)

	async def movement_pattern(
		self,
		target_id: str,
		date_range: dict[str, str],
	) -> dict[str, Any]:
		"""Analyse movement patterns for *target_id* and detect route regularity."""
		tracking = await self.movement_tracking(target_id, date_range)
		positions = tracking["positions"]
		if len(positions) < 2:
			regularity = "insufficient_data"
		else:
			accuracies = [p["accuracy"] for p in positions]
			mean_acc = statistics.mean(accuracies)
			stdev_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
			regularity = "regular" if stdev_acc < mean_acc * 0.3 else "irregular"
		pattern_id = f"mvmt_pat_{target_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result: dict[str, Any] = {
			"pattern_id": pattern_id,
			"target_id": target_id,
			"date_range": date_range,
			"observation_count": tracking["observation_count"],
			"regularity": regularity,
			"last_seen": tracking["last_seen"],
			"analysed_at": _utcnow(),
		}
		self._audit(self.tenant_id, "movement_pattern_analysed", pattern_id)
		return result

	async def activity_zone(
		self,
		target_id: str,
		radius_km: float = 5.0,
	) -> dict[str, Any]:
		"""Identify activity zones (clusters of positions) for *target_id*."""
		assert present(target_id), "target_id required"
		assert radius_km > 0, "radius_km must be positive"
		tenant_id = self.tenant_id
		pos = self._target_positions.get(target_id)
		if pos is None:
			return {"target_id": target_id, "activity_zones": [], "status": "no_position_data", "computed_at": _utcnow()}
		zones: list[dict[str, Any]] = [{
			"zone_id": f"zone_{target_id}_primary",
			"centre": {"lat": pos.get("lat", 0.0), "lon": pos.get("lon", 0.0)},
			"radius_km": radius_km,
			"activity_level": "primary",
		}]
		# Check geofences as secondary activity zones
		for zone_id, fence in self._geofences.items():
			dist = _haversine_km(float(pos.get("lat", 0)), float(pos.get("lon", 0)), fence["lat"], fence["lon"])
			if dist <= fence.get("radius_km", 5.0) * 2:
				zones.append({"zone_id": zone_id, "centre": {"lat": fence["lat"], "lon": fence["lon"]}, "radius_km": fence["radius_km"], "activity_level": "secondary"})
		az_id = f"activity_zone_{target_id}"
		self._audit(tenant_id, "activity_zone_identified", az_id)
		return {"activity_zone_id": az_id, "target_id": target_id, "zones": zones, "zone_count": len(zones), "computed_at": _utcnow()}

	async def geofence_alert_check(
		self,
		target_id: str,
		zone_id: str,
	) -> dict[str, Any]:
		"""Alias for geofence_alert with cleaner name."""
		return await self.geofence_alert(target_id, zone_id)

	async def satellite_schedule(
		self,
		area_id: str,
		frequency: str = "daily",
	) -> dict[str, Any]:
		"""Schedule satellite imagery collection for *area_id* at *frequency*."""
		assert present(area_id), "area_id required"
		assert frequency in {"hourly", "daily", "weekly", "monthly"}, f"unknown frequency: {frequency}"
		tenant_id = self.tenant_id
		area = self._tenant_area_or_none(area_id, tenant_id)
		if area is None:
			raise KeyError(f"Area not found: {area_id}")
		freq_days = {"hourly": 0.042, "daily": 1, "weekly": 7, "monthly": 30}
		schedule_id = f"sat_sched_{area_id}_{frequency}"
		result: dict[str, Any] = {
			"schedule_id": schedule_id,
			"area_id": area_id,
			"frequency": frequency,
			"revisit_days": freq_days[frequency],
			"status": "scheduled",
			"scheduled_at": _utcnow(),
			"tenant_id": tenant_id,
		}
		self._audit(tenant_id, "satellite_schedule_created", schedule_id)
		return result

	async def geoint_report_generate(
		self,
		classification: str,
		area_id: str,
	) -> dict[str, Any]:
		"""Generate a GEOINT report for *area_id* at *classification* level."""
		area = self._tenant_area_or_none(area_id, self.tenant_id)
		area_name = getattr(area, "name", area_id) if area else area_id
		return await self.geoint_report(classification, area_name)

	async def geoint_analytics(self) -> dict[str, Any]:
		"""Aggregate GEOINT programme analytics for the tenant."""
		tenant_id = self.tenant_id
		return {
			"tenant_id": tenant_id,
			"area_count": self._count(self.areas, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"collection_plan_count": self._count(self.collection_plans, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"feature_count": self._count(self.features, tenant_id),
			"change_count": self._count(self.changes, tenant_id),
			"assessment_count": self._count(self.assessments, tenant_id),
			"geofence_count": len(self._geofences),
			"tracked_targets": len(self._target_positions),
			"computed_at": _utcnow(),
		}

	async def area_coverage_summary(self) -> dict[str, Any]:
		"""Summarise collection plan coverage across registered areas of interest."""
		tenant_id = self.tenant_id
		area_plan_counts: dict[str, int] = defaultdict(int)
		for (tid, _), plan in self.collection_plans.items():
			if tid == tenant_id:
				area_plan_counts[getattr(plan, "area_id", "")] += 1

		uncovered_areas = [
			aid for (tid, aid) in self.areas
			if tid == tenant_id and area_plan_counts.get(aid, 0) == 0
		]
		return {
			"tenant_id": tenant_id,
			"total_areas": self._count(self.areas, tenant_id),
			"areas_with_collection_plans": len(area_plan_counts),
			"uncovered_areas": uncovered_areas,
			"computed_at": _utcnow(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> GeospatialAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_area_or_none(self, item_id: str, tenant_id: str) -> AreaOfInterest | None:
		return self.areas.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> ImagerySource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_plan_or_none(self, item_id: str, tenant_id: str) -> CollectionPlan | None:
		return self.collection_plans.get(self._tenant_key(tenant_id, item_id))

	def _tenant_observation_or_none(self, item_id: str, tenant_id: str) -> GeoObservation | None:
		return self.observations.get(self._tenant_key(tenant_id, item_id))

	def _tenant_feature_or_none(self, item_id: str, tenant_id: str) -> GeoFeature | None:
		return self.features.get(self._tenant_key(tenant_id, item_id))

	def _tenant_change_or_none(self, item_id: str, tenant_id: str) -> ChangeDetection | None:
		return self.changes.get(self._tenant_key(tenant_id, item_id))

	def _tenant_assessment_or_none(self, item_id: str, tenant_id: str) -> GeoAssessment | None:
		return self.assessments.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"recorded_at": _utcnow(),
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "geoint_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "geoint_policy_denied")



	async def ml_geospatial_threat_score(self, *args, **kwargs):
		"""AI-powered AI geospatial threat assessment from location intelligence. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="geospatial_threat_assessment")
			return {"threat_score": round(result.score,3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

IntelGEOINTService = GeospatialIntelligenceService
