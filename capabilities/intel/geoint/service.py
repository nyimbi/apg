"""Executable service layer for APG Geospatial Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CHANGE_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_COLLECTION_MODES, SUPPORTED_FEATURE_TYPES, SUPPORTED_RESOLUTION_CLASSES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SENSOR_TYPES, SUPPORTED_SEVERITIES, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract
	from .geoint_runtime import bounded_score, normalize_code, positive_int, present
	from .models import AreaOfInterest, ChangeDetection, CollectionPlan, GEOINTAgent, GEOINTDissemination, GEOINTReview, GeoAssessment, GeoFeature, GeoObservation, GeospatialAuthority, ImagerySource
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CHANGE_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_COLLECTION_MODES, SUPPORTED_FEATURE_TYPES, SUPPORTED_RESOLUTION_CLASSES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SENSOR_TYPES, SUPPORTED_SEVERITIES, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from geoint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import AreaOfInterest, ChangeDetection, CollectionPlan, GEOINTAgent, GEOINTDissemination, GEOINTReview, GeoAssessment, GeoFeature, GeoObservation, GeospatialAuthority, ImagerySource  # type: ignore


class GeospatialIntelligenceService:
	"""Tenant-scoped GEOINT coordination runtime for generated APG applications."""

	def __init__(self) -> None:
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

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = GeospatialAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "geoint_authority_recorded", authority_id)
		return item.to_dict()

	def record_area(self, area_id: str, tenant_id: str, name: str, geometry_reference: str, classification: str, owner_id: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_area", "name_present": present(name), "geometry_present": present(geometry_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "owner_present": present(owner_id), "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = AreaOfInterest(area_id, tenant_id, name, geometry_reference, classification, owner_id, authority_id, evidence_reference)
		self.areas[self._tenant_key(tenant_id, area_id)] = item
		self._audit(tenant_id, "geoint_area_recorded", area_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, sensor_type: str, resolution_class: str, owner_id: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		sensor_type = normalize_code(sensor_type)
		resolution_class = normalize_code(resolution_class)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "sensor_type_supported": sensor_type in SUPPORTED_SENSOR_TYPES, "resolution_class_supported": resolution_class in SUPPORTED_RESOLUTION_CLASSES, "owner_present": present(owner_id), "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = ImagerySource(source_id, tenant_id, source_type, sensor_type, resolution_class, owner_id, authority_id, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "geoint_source_registered", source_id)
		return item.to_dict()

	def record_collection_plan(self, plan_id: str, tenant_id: str, authority_id: str, area_id: str, source_id: str, collection_mode: str, retention_days: int, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		area = self._tenant_area_or_none(area_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		collection_mode = normalize_code(collection_mode)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_collection_plan", "authority_present": authority is not None, "area_present": area is not None, "source_present": source is not None, "area_authority_match": area is not None and area.authority_id == authority_id, "source_authority_match": source is not None and source.authority_id == authority_id, "collection_mode_supported": collection_mode in SUPPORTED_COLLECTION_MODES, "retention_days_positive": positive_int(retention_days), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = CollectionPlan(plan_id, tenant_id, authority_id, area_id, source_id, collection_mode, int(retention_days), approval_reference, evidence_reference)
		self.collection_plans[self._tenant_key(tenant_id, plan_id)] = item
		self._audit(tenant_id, "geoint_collection_plan_recorded", plan_id)
		return item.to_dict()

	def record_observation(self, observation_id: str, tenant_id: str, plan_id: str, observation_reference: str, captured_at: str, geospatial_accuracy_score: float, evidence_reference: str) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_observation", "plan_present": plan is not None, "observation_reference_present": present(observation_reference), "captured_at_present": present(captured_at), "accuracy_valid": bounded_score(geospatial_accuracy_score), "evidence_present": present(evidence_reference)})
		item = GeoObservation(observation_id, tenant_id, plan_id, observation_reference, captured_at, float(geospatial_accuracy_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "geoint_observation_recorded", observation_id)
		return item.to_dict()

	def record_feature(self, feature_id: str, tenant_id: str, observation_id: str, feature_type: str, geometry_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		feature_type = normalize_code(feature_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_feature", "observation_present": observation is not None, "feature_type_supported": feature_type in SUPPORTED_FEATURE_TYPES, "geometry_present": present(geometry_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = GeoFeature(feature_id, tenant_id, observation_id, feature_type, geometry_reference, float(confidence_score), analyst_id, evidence_reference)
		self.features[self._tenant_key(tenant_id, feature_id)] = item
		self._audit(tenant_id, "geoint_feature_recorded", feature_id)
		return item.to_dict()

	def record_change(self, change_id: str, tenant_id: str, feature_id: str, change_type: str, severity: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		feature = self._tenant_feature_or_none(feature_id, tenant_id)
		change_type = normalize_code(change_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_change", "feature_present": feature is not None, "change_type_supported": change_type in SUPPORTED_CHANGE_TYPES, "severity_supported": severity in SUPPORTED_SEVERITIES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = ChangeDetection(change_id, tenant_id, feature_id, change_type, severity, float(confidence_score), analyst_id, evidence_reference)
		self.changes[self._tenant_key(tenant_id, change_id)] = item
		self._audit(tenant_id, "geoint_change_recorded", change_id)
		return item.to_dict()

	def record_assessment(self, assessment_id: str, tenant_id: str, change_id: str, assessment_type: str, classification: str, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		change = self._tenant_change_or_none(change_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_assessment", "change_present": change is not None, "assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES, "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = GeoAssessment(assessment_id, tenant_id, change_id, assessment_type, classification, analyst_id, evidence_reference)
		self.assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "geoint_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = GEOINTDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "geoint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = GEOINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "geoint_review_recorded", review_id)
		return item.to_dict()

	def register_geoint_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_geoint_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = GEOINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "geoint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, targeting_or_harmful_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "geoint_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "targeting_or_harmful_scope": targeting_or_harmful_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope, "targeting_or_harmful_scope": targeting_or_harmful_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "geoint_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.geoint.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "area_count": self._count(self.areas, tenant_id), "source_count": self._count(self.sources, tenant_id), "collection_plan_count": self._count(self.collection_plans, tenant_id), "observation_count": self._count(self.observations, tenant_id), "feature_count": self._count(self.features, tenant_id), "change_count": self._count(self.changes, tenant_id), "assessment_count": self._count(self.assessments, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

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
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "geoint_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "geoint_policy_denied")


IntelGEOINTService = GeospatialIntelligenceService
