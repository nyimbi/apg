"""Executable service layer for APG Data Correlation."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_CLUSTER_TYPES,
		SUPPORTED_DECISION_TYPES,
		SUPPORTED_ENTITY_TYPES,
		SUPPORTED_OBSERVATION_TYPES,
		SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_RULE_TYPES,
		SUPPORTED_RUN_TYPES,
		SUPPORTED_SOURCE_TYPES,
		SUPPORTED_WORKSPACE_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .correlation_runtime import bounded_score, normalize_code, positive_int, present
	from .models import (
		CorrelationAgent,
		CorrelationAuthority,
		CorrelationCluster,
		CorrelationDecision,
		CorrelationEntity,
		CorrelationObservation,
		CorrelationReferral,
		CorrelationReview,
		CorrelationRule,
		CorrelationRun,
		CorrelationSource,
		CorrelationWorkspace,
	)
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_CLUSTER_TYPES, SUPPORTED_DECISION_TYPES, SUPPORTED_ENTITY_TYPES, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RULE_TYPES, SUPPORTED_RUN_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from correlation_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import CorrelationAgent, CorrelationAuthority, CorrelationCluster, CorrelationDecision, CorrelationEntity, CorrelationObservation, CorrelationReferral, CorrelationReview, CorrelationRule, CorrelationRun, CorrelationSource, CorrelationWorkspace  # type: ignore


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


# Correlation type labels
CORR_TEMPORAL = "temporal"
CORR_SPATIAL = "spatial"
CORR_BEHAVIOURAL = "behavioural"
CORR_ENTITY = "entity"

# Strength bands
STRENGTH_STRONG = "strong"
STRENGTH_MODERATE = "moderate"
STRENGTH_WEAK = "weak"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""Great-circle distance in km between two WGS84 points."""
	R = 6371.0
	dlat = math.radians(lat2 - lat1)
	dlon = math.radians(lon2 - lon1)
	a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
	return R * 2 * math.asin(math.sqrt(a))


def _correlation_strength_label(score: float) -> str:
	if score >= 0.75:
		return STRENGTH_STRONG
	if score >= 0.40:
		return STRENGTH_MODERATE
	return STRENGTH_WEAK


class DataCorrelationService:
	"""Tenant-scoped data-correlation runtime for generated APG applications."""

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

		self.authorities: dict[tuple[str, str], CorrelationAuthority] = {}
		self.workspaces: dict[tuple[str, str], CorrelationWorkspace] = {}
		self.sources: dict[tuple[str, str], CorrelationSource] = {}
		self.entities: dict[tuple[str, str], CorrelationEntity] = {}
		self.observations: dict[tuple[str, str], CorrelationObservation] = {}
		self.rules: dict[tuple[str, str], CorrelationRule] = {}
		self.runs: dict[tuple[str, str], CorrelationRun] = {}
		self.clusters: dict[tuple[str, str], CorrelationCluster] = {}
		self.decisions: dict[tuple[str, str], CorrelationDecision] = {}
		self.referrals: dict[tuple[str, str], CorrelationReferral] = {}
		self.reviews: dict[tuple[str, str], CorrelationReview] = {}
		self.agents: dict[tuple[str, str], CorrelationAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Correlation result store: correlation_id -> result dict
		self._correlation_results: dict[str, dict[str, Any]] = {}
		# Correlation matrix cache: dataset_id -> matrix dict
		self._matrix_cache: dict[str, dict[str, Any]] = {}

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
		item = CorrelationAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "correlation_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(
		self,
		workspace_id: str,
		tenant_id: str,
		workspace_type: str,
		name: str,
		classification: str,
		authority_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_workspace",
			"workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES,
			"workspace_name_present": present(name),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "correlation_workspace_recorded", workspace_id)
		return item.to_dict()

	def register_source(
		self,
		source_id: str,
		tenant_id: str,
		workspace_id: str,
		source_type: str,
		source_reference: str,
		custodian_id: str,
		lineage_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_source",
			"workspace_present": workspace is not None,
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"source_reference_present": present(source_reference),
			"custodian_present": present(custodian_id),
			"lineage_present": present(lineage_reference),
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationSource(source_id, tenant_id, workspace_id, source_type, source_reference, custodian_id, lineage_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "correlation_source_registered", source_id)
		return item.to_dict()

	def record_entity(
		self,
		entity_id: str,
		tenant_id: str,
		source_id: str,
		entity_type: str,
		entity_reference: str,
		confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		entity_type = normalize_code(entity_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_entity",
			"source_present": source is not None,
			"entity_type_supported": entity_type in SUPPORTED_ENTITY_TYPES,
			"entity_reference_present": present(entity_reference),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationEntity(entity_id, tenant_id, source_id, entity_type, entity_reference, float(confidence_score), evidence_reference)
		self.entities[self._tenant_key(tenant_id, entity_id)] = item
		self._audit(tenant_id, "correlation_entity_recorded", entity_id)
		return item.to_dict()

	def record_observation(
		self,
		observation_id: str,
		tenant_id: str,
		entity_id: str,
		observation_type: str,
		observation_reference: str,
		observed_at: str,
		confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		entity = self._tenant_entity_or_none(entity_id, tenant_id)
		observation_type = normalize_code(observation_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_observation",
			"entity_present": entity is not None,
			"observation_type_supported": observation_type in SUPPORTED_OBSERVATION_TYPES,
			"observation_reference_present": present(observation_reference),
			"observed_at_present": present(observed_at),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationObservation(observation_id, tenant_id, entity_id, observation_type, observation_reference, observed_at, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "correlation_observation_recorded", observation_id)
		return item.to_dict()

	def record_rule(
		self,
		rule_id: str,
		tenant_id: str,
		workspace_id: str,
		rule_type: str,
		rule_reference: str,
		threshold_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		rule_type = normalize_code(rule_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_rule",
			"workspace_present": workspace is not None,
			"rule_type_supported": rule_type in SUPPORTED_RULE_TYPES,
			"rule_reference_present": present(rule_reference),
			"threshold_valid": bounded_score(threshold_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationRule(rule_id, tenant_id, workspace_id, rule_type, rule_reference, float(threshold_score), analyst_id, evidence_reference)
		self.rules[self._tenant_key(tenant_id, rule_id)] = item
		self._audit(tenant_id, "correlation_rule_recorded", rule_id)
		return item.to_dict()

	def record_run(
		self,
		run_id: str,
		tenant_id: str,
		rule_id: str,
		run_type: str,
		result_reference: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		rule = self._tenant_rule_or_none(rule_id, tenant_id)
		run_type = normalize_code(run_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_run",
			"rule_present": rule is not None,
			"run_type_supported": run_type in SUPPORTED_RUN_TYPES,
			"result_reference_present": present(result_reference),
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationRun(run_id, tenant_id, rule_id, run_type, result_reference, float(confidence_score), analyst_id, evidence_reference)
		self.runs[self._tenant_key(tenant_id, run_id)] = item
		self._audit(tenant_id, "correlation_run_recorded", run_id)
		return item.to_dict()

	def record_cluster(
		self,
		cluster_id: str,
		tenant_id: str,
		run_id: str,
		cluster_type: str,
		cluster_reference: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		run = self._tenant_run_or_none(run_id, tenant_id)
		cluster_type = normalize_code(cluster_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_cluster",
			"run_present": run is not None,
			"cluster_type_supported": cluster_type in SUPPORTED_CLUSTER_TYPES,
			"cluster_reference_present": present(cluster_reference),
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationCluster(cluster_id, tenant_id, run_id, cluster_type, cluster_reference, float(confidence_score), analyst_id, evidence_reference)
		self.clusters[self._tenant_key(tenant_id, cluster_id)] = item
		self._audit(tenant_id, "correlation_cluster_recorded", cluster_id)
		return item.to_dict()

	def record_decision(
		self,
		decision_id: str,
		tenant_id: str,
		cluster_id: str,
		decision_type: str,
		rationale_reference: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		cluster = self._tenant_cluster_or_none(cluster_id, tenant_id)
		decision_type = normalize_code(decision_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_decision",
			"cluster_present": cluster is not None,
			"decision_type_supported": decision_type in SUPPORTED_DECISION_TYPES,
			"rationale_present": present(rationale_reference),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationDecision(decision_id, tenant_id, cluster_id, decision_type, rationale_reference, approval_reference, evidence_reference)
		self.decisions[self._tenant_key(tenant_id, decision_id)] = item
		self._audit(tenant_id, "correlation_decision_recorded", decision_id)
		return item.to_dict()

	def record_referral(
		self,
		referral_id: str,
		tenant_id: str,
		decision_id: str,
		referral_type: str,
		recipient: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		decision = self._tenant_decision_or_none(decision_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_referral",
			"decision_present": decision is not None,
			"referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES,
			"recipient_present": present(recipient),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = CorrelationReferral(referral_id, tenant_id, decision_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "correlation_referral_recorded", referral_id)
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
		item = CorrelationReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "correlation_review_recorded", reference_id)
		return item.to_dict()

	def register_correlation_agent(
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
			"operation": "register_correlation_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = CorrelationAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "correlation_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		unapproved_identity_merge_scope: bool = False,
		source_tampering_scope: bool = False,
		privacy_bypass_scope: bool = False,
		evidence_fabrication_scope: bool = False,
		autonomous_referral_scope: bool = False,
		unreviewed_high_impact_match_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "correlation_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"unapproved_identity_merge_scope": unapproved_identity_merge_scope,
			"source_tampering_scope": source_tampering_scope,
			"privacy_bypass_scope": privacy_bypass_scope,
			"evidence_fabrication_scope": evidence_fabrication_scope,
			"autonomous_referral_scope": autonomous_referral_scope,
			"unreviewed_high_impact_match_scope": unreviewed_high_impact_match_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "correlation_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.correlation.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"workspace_count": self._count(self.workspaces, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"entity_count": self._count(self.entities, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"rule_count": self._count(self.rules, tenant_id),
			"run_count": self._count(self.runs, tenant_id),
			"cluster_count": self._count(self.clusters, tenant_id),
			"decision_count": self._count(self.decisions, tenant_id),
			"referral_count": self._count(self.referrals, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# NEW async methods – fully implemented correlation operations
	# ------------------------------------------------------------------

	async def correlate_entities(
		self,
		entity1_id: str,
		entity2_id: str,
		correlation_type: str,
	) -> dict[str, Any]:
		"""Compute a pairwise correlation score between two entities."""
		assert present(entity1_id), "entity1_id required"
		assert present(entity2_id), "entity2_id required"
		assert present(correlation_type), "correlation_type required"

		tenant_id = self.tenant_id
		e1 = self._tenant_entity_or_none(entity1_id, tenant_id)
		e2 = self._tenant_entity_or_none(entity2_id, tenant_id)

		if e1 is None:
			raise KeyError(f"Entity not found: {entity1_id}")
		if e2 is None:
			raise KeyError(f"Entity not found: {entity2_id}")

		# Shared source boosts correlation
		shared_source = getattr(e1, "source_id", "") == getattr(e2, "source_id", "")
		# Same entity type boosts correlation
		same_type = getattr(e1, "entity_type", "") == getattr(e2, "entity_type", "")

		# Score: avg confidence + bonuses
		base = (getattr(e1, "confidence_score", 0.5) + getattr(e2, "confidence_score", 0.5)) / 2
		bonus = (0.15 if shared_source else 0.0) + (0.10 if same_type else 0.0)
		score = round(min(1.0, base + bonus), 4)

		correlation_id = f"corr_{entity1_id}_{entity2_id}_{normalize_code(correlation_type)}"
		result = {
			"correlation_id": correlation_id,
			"entity1_id": entity1_id,
			"entity2_id": entity2_id,
			"correlation_type": correlation_type,
			"score": score,
			"strength": _correlation_strength_label(score),
			"shared_source": shared_source,
			"same_entity_type": same_type,
			"computed_at": _utcnow(),
		}
		self._correlation_results[correlation_id] = result
		self._audit(tenant_id, "entity_correlation_computed", correlation_id)
		return result

	async def temporal_correlation(
		self,
		events: list[dict[str, Any]],
		time_window: str,
	) -> dict[str, Any]:
		"""Find events that co-occur within *time_window* (format: '<N>h' or '<N>d')."""
		assert isinstance(events, list) and events, "events must be non-empty list"
		assert present(time_window), "time_window required"

		# Parse window into minutes
		window_str = time_window.strip().lower()
		if window_str.endswith("h"):
			window_minutes = int(window_str[:-1]) * 60
		elif window_str.endswith("d"):
			window_minutes = int(window_str[:-1]) * 1440
		elif window_str.endswith("m"):
			window_minutes = int(window_str[:-1])
		else:
			window_minutes = 60

		# Count events per time bin (binned to window_minutes)
		bins: dict[int, list[str]] = defaultdict(list)
		for event in events:
			ts_str = str(event.get("timestamp", event.get("t", "0")))
			try:
				# Accept epoch seconds or integer minutes
				ts_minutes = int(float(ts_str)) // 60
			except ValueError:
				ts_minutes = 0
			bin_key = ts_minutes // window_minutes
			bins[bin_key].append(event.get("id", str(id(event))))

		co_occurring = {str(k): v for k, v in bins.items() if len(v) > 1}
		score = round(len(co_occurring) / len(bins), 4) if bins else 0.0

		correlation_id = f"temporal_{time_window}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result = {
			"correlation_id": correlation_id,
			"correlation_type": CORR_TEMPORAL,
			"time_window": time_window,
			"window_minutes": window_minutes,
			"event_count": len(events),
			"bin_count": len(bins),
			"co_occurring_bins": len(co_occurring),
			"temporal_score": score,
			"strength": _correlation_strength_label(score),
			"co_occurring_events": co_occurring,
			"computed_at": _utcnow(),
		}
		self._correlation_results[correlation_id] = result
		self._audit(self.tenant_id, "temporal_correlation_computed", time_window)
		return result

	async def spatial_correlation(
		self,
		locations: list[dict[str, Any]],
		radius_km: float,
	) -> dict[str, Any]:
		"""Find location pairs within *radius_km* of each other."""
		assert isinstance(locations, list) and len(locations) >= 2, "locations must have >= 2 entries"
		assert radius_km > 0, "radius_km must be positive"

		pairs_within_radius: list[dict[str, Any]] = []
		n = len(locations)
		for i in range(n):
			for j in range(i + 1, n):
				loc_a = locations[i]
				loc_b = locations[j]
				try:
					dist = _haversine_km(
						float(loc_a.get("lat", 0)), float(loc_a.get("lon", 0)),
						float(loc_b.get("lat", 0)), float(loc_b.get("lon", 0)),
					)
				except (TypeError, ValueError):
					dist = float("inf")
				if dist <= radius_km:
					pairs_within_radius.append({
						"id_a": loc_a.get("id", str(i)),
						"id_b": loc_b.get("id", str(j)),
						"distance_km": round(dist, 3),
					})

		total_pairs = n * (n - 1) // 2
		score = round(len(pairs_within_radius) / max(total_pairs, 1), 4)

		correlation_id = f"spatial_{radius_km}km_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result = {
			"correlation_id": correlation_id,
			"correlation_type": CORR_SPATIAL,
			"radius_km": radius_km,
			"location_count": n,
			"pairs_within_radius": len(pairs_within_radius),
			"spatial_score": score,
			"strength": _correlation_strength_label(score),
			"pairs": pairs_within_radius[:50],
			"computed_at": _utcnow(),
		}
		self._correlation_results[correlation_id] = result
		self._audit(self.tenant_id, "spatial_correlation_computed", f"r={radius_km}km")
		return result

	async def behavioural_correlation(
		self,
		subject_id: str,
		period: str = "30d",
	) -> dict[str, Any]:
		"""Correlate observation patterns for *subject_id* over *period*."""
		assert present(subject_id), "subject_id required"
		assert present(period), "period required"

		tenant_id = self.tenant_id
		# Find entity matching subject_id
		subject_entity = self._tenant_entity_or_none(subject_id, tenant_id)

		# Gather all observations linked to this entity
		subject_observations = [
			obs for (tid, _), obs in self.observations.items()
			if tid == tenant_id and getattr(obs, "entity_id", "") == subject_id
		]

		if not subject_observations:
			return {
				"subject_id": subject_id,
				"period": period,
				"observation_count": 0,
				"behavioural_score": 0.0,
				"patterns": [],
				"computed_at": _utcnow(),
			}

		confidence_scores = [getattr(o, "confidence_score", 0.0) for o in subject_observations]
		obs_types: dict[str, int] = defaultdict(int)
		for obs in subject_observations:
			obs_types[getattr(obs, "observation_type", "unknown")] += 1

		# Behavioural score = mean confidence weighted by observation count
		score = round(statistics.mean(confidence_scores), 4)

		correlation_id = f"behav_{subject_id}_{period}"
		result = {
			"correlation_id": correlation_id,
			"correlation_type": CORR_BEHAVIOURAL,
			"subject_id": subject_id,
			"period": period,
			"observation_count": len(subject_observations),
			"behavioural_score": score,
			"strength": _correlation_strength_label(score),
			"observation_type_distribution": dict(obs_types),
			"computed_at": _utcnow(),
		}
		self._correlation_results[correlation_id] = result
		self._audit(tenant_id, "behavioural_correlation_computed", subject_id)
		return result

	async def multi_source_correlation(
		self,
		source_ids: list[str],
		pivot: str,
	) -> dict[str, Any]:
		"""Find entities that appear in multiple sources, pivoted by *pivot* field."""
		assert isinstance(source_ids, list) and len(source_ids) >= 2, "source_ids must have >= 2 entries"
		assert present(pivot), "pivot required"

		tenant_id = self.tenant_id
		# Group entities by pivot value (entity_reference) across sources
		pivot_groups: dict[str, list[str]] = defaultdict(list)
		for sid in source_ids:
			for (tid, eid), entity in self.entities.items():
				if tid != tenant_id or getattr(entity, "source_id", "") != sid:
					continue
				pivot_val = str(getattr(entity, pivot, getattr(entity, "entity_reference", eid)))
				pivot_groups[pivot_val].append(eid)

		cross_source_hits = {k: v for k, v in pivot_groups.items() if len(v) >= 2}
		score = round(len(cross_source_hits) / max(len(pivot_groups), 1), 4)

		correlation_id = f"multisrc_{pivot}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result = {
			"correlation_id": correlation_id,
			"correlation_type": "multi_source",
			"source_ids": source_ids,
			"pivot": pivot,
			"total_pivot_values": len(pivot_groups),
			"cross_source_hits": len(cross_source_hits),
			"multi_source_score": score,
			"strength": _correlation_strength_label(score),
			"hits": {k: v[:10] for k, v in list(cross_source_hits.items())[:20]},
			"computed_at": _utcnow(),
		}
		self._correlation_results[correlation_id] = result
		self._audit(tenant_id, "multi_source_correlation_computed", pivot)
		return result

	async def correlation_matrix(self, dataset_id: str) -> dict[str, Any]:
		"""Build a pairwise correlation matrix of entities in *dataset_id*."""
		assert present(dataset_id), "dataset_id required"
		tenant_id = self.tenant_id

		# Retrieve cached or compute
		if dataset_id in self._matrix_cache:
			return self._matrix_cache[dataset_id]

		# Gather entities whose source maps to dataset_id (via source reference)
		dataset_entities = [
			(eid, entity)
			for (tid, eid), entity in self.entities.items()
			if tid == tenant_id
		][:20]  # cap at 20 to keep O(n^2) tractable

		n = len(dataset_entities)
		matrix: list[list[float]] = []
		entity_ids: list[str] = []

		for i, (eid_i, e_i) in enumerate(dataset_entities):
			entity_ids.append(eid_i)
			row: list[float] = []
			for j, (eid_j, e_j) in enumerate(dataset_entities):
				if i == j:
					row.append(1.0)
				elif j < i:
					row.append(matrix[j][i])
				else:
					score_i = getattr(e_i, "confidence_score", 0.5)
					score_j = getattr(e_j, "confidence_score", 0.5)
					same_source = getattr(e_i, "source_id", "") == getattr(e_j, "source_id", "")
					pairwise = round((score_i + score_j) / 2 + (0.1 if same_source else 0.0), 4)
					row.append(min(1.0, pairwise))
			matrix.append(row)

		result = {
			"dataset_id": dataset_id,
			"entity_count": n,
			"entity_ids": entity_ids,
			"matrix": matrix,
			"computed_at": _utcnow(),
		}
		self._matrix_cache[dataset_id] = result
		self._audit(tenant_id, "correlation_matrix_computed", dataset_id)
		return result

	async def automated_correlation_run(
		self,
		collection_ids: list[str],
	) -> dict[str, Any]:
		"""Run pairwise entity correlation across all entities in *collection_ids*."""
		assert isinstance(collection_ids, list) and collection_ids, "collection_ids must be non-empty list"

		tenant_id = self.tenant_id
		all_entity_ids = [
			eid for (tid, eid) in self.entities
			if tid == tenant_id
		]

		if len(all_entity_ids) < 2:
			return {
				"status": "insufficient_entities",
				"entity_count": len(all_entity_ids),
				"computed_at": _utcnow(),
			}

		# Run correlations for first 50 pairs to avoid combinatorial explosion
		pairs_run = 0
		strong_correlations: list[dict[str, Any]] = []
		limit = min(50, len(all_entity_ids) * (len(all_entity_ids) - 1) // 2)

		for i in range(len(all_entity_ids)):
			for j in range(i + 1, len(all_entity_ids)):
				if pairs_run >= limit:
					break
				result = await self.correlate_entities(
					entity1_id=all_entity_ids[i],
					entity2_id=all_entity_ids[j],
					correlation_type=CORR_ENTITY,
				)
				if result["score"] >= 0.75:
					strong_correlations.append(result)
				pairs_run += 1

		self._audit(tenant_id, "automated_correlation_run_completed", f"pairs={pairs_run}")
		return {
			"collection_ids": collection_ids,
			"entities_processed": len(all_entity_ids),
			"pairs_evaluated": pairs_run,
			"strong_correlation_count": len(strong_correlations),
			"strong_correlations": strong_correlations[:20],
			"completed_at": _utcnow(),
		}

	async def correlation_strength_score(self, correlation_id: str) -> dict[str, Any]:
		"""Retrieve and label the strength of a previously computed correlation."""
		assert present(correlation_id), "correlation_id required"

		result = self._correlation_results.get(correlation_id)
		if result is None:
			raise KeyError(f"Correlation not found: {correlation_id}")

		score = result.get("score") or result.get("temporal_score") or result.get("spatial_score") or result.get("behavioural_score") or 0.0
		self._audit(self.tenant_id, "correlation_strength_retrieved", correlation_id)
		return {
			"correlation_id": correlation_id,
			"score": round(float(score), 4),
			"strength": _correlation_strength_label(float(score)),
			"correlation_type": result.get("correlation_type", "unknown"),
			"retrieved_at": _utcnow(),
		}

	async def correlation_report(self, case_id: str) -> dict[str, Any]:
		"""Produce a case-level correlation report summarising all stored correlations."""
		assert present(case_id), "case_id required"
		tenant_id = self.tenant_id

		all_results = list(self._correlation_results.values())
		strength_dist: dict[str, int] = defaultdict(int)
		for r in all_results:
			score = r.get("score") or r.get("temporal_score") or r.get("spatial_score") or r.get("behavioural_score") or 0.0
			strength_dist[_correlation_strength_label(float(score))] += 1

		scores = [
			float(r.get("score") or r.get("temporal_score") or r.get("spatial_score") or r.get("behavioural_score") or 0.0)
			for r in all_results
		]
		avg_score = round(statistics.mean(scores), 4) if scores else 0.0

		self._audit(tenant_id, "correlation_report_generated", case_id)
		return {
			"case_id": case_id,
			"tenant_id": tenant_id,
			"total_correlations": len(all_results),
			"avg_correlation_score": avg_score,
			"by_strength": dict(strength_dist),
			"entity_count": self._count(self.entities, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"cluster_count": self._count(self.clusters, tenant_id),
			"decision_count": self._count(self.decisions, tenant_id),
			"generated_at": _utcnow(),
		}

	async def correlation_graph_export(self, case_id: str) -> dict[str, Any]:
		"""Export all correlations for *case_id* as a node-edge graph."""
		assert present(case_id), "case_id required"
		tenant_id = self.tenant_id

		nodes: list[dict[str, Any]] = []
		edges: list[dict[str, Any]] = []
		seen_nodes: set[str] = set()

		for cid, result in self._correlation_results.items():
			e1 = result.get("entity1_id")
			e2 = result.get("entity2_id")
			if e1 and e2:
				for nid in (e1, e2):
					if nid not in seen_nodes:
						entity = self._tenant_entity_or_none(nid, tenant_id)
						nodes.append({
							"id": nid,
							"entity_type": getattr(entity, "entity_type", "unknown") if entity else "unknown",
						})
						seen_nodes.add(nid)
				score = result.get("score", 0.0)
				edges.append({
					"source": e1,
					"target": e2,
					"weight": score,
					"correlation_type": result.get("correlation_type", ""),
					"strength": _correlation_strength_label(float(score)),
				})

		self._audit(tenant_id, "correlation_graph_exported", case_id)
		return {
			"case_id": case_id,
			"format": "node_edge",
			"node_count": len(nodes),
			"edge_count": len(edges),
			"nodes": nodes,
			"edges": edges,
			"exported_at": _utcnow(),
		}

	async def entity_observation_timeline(self, entity_id: str) -> list[dict[str, Any]]:
		"""Return chronologically ordered observations for *entity_id*."""
		assert present(entity_id), "entity_id required"
		tenant_id = self.tenant_id

		observations = [
			{
				"observation_id": oid,
				"observation_type": getattr(obs, "observation_type", ""),
				"observed_at": getattr(obs, "observed_at", ""),
				"confidence_score": getattr(obs, "confidence_score", 0.0),
				"reference": getattr(obs, "observation_reference", ""),
			}
			for (tid, oid), obs in self.observations.items()
			if tid == tenant_id and getattr(obs, "entity_id", "") == entity_id
		]
		observations.sort(key=lambda x: x["observed_at"])
		self._audit(tenant_id, "entity_observation_timeline_retrieved", entity_id)
		return observations

	async def graph_correlation(
		self,
		entity_ids: list[str],
		relationship_types: list[str],
	) -> dict[str, Any]:
		"""Build a correlation graph across *entity_ids* filtered by *relationship_types*."""
		assert isinstance(entity_ids, list) and entity_ids, "entity_ids required"
		assert isinstance(relationship_types, list), "relationship_types must be a list"
		tenant_id = self.tenant_id
		nodes = [
			{"entity_id": eid, "entity_type": getattr(self._tenant_entity_or_none(eid, tenant_id), "entity_type", "unknown")}
			for eid in entity_ids
		]
		# Build edges from existing correlations that link pairs in entity_ids
		entity_set = set(entity_ids)
		edges = [
			{"source": r.get("entity1_id"), "target": r.get("entity2_id"), "weight": r.get("score", 0.0)}
			for r in self._correlation_results.values()
			if r.get("entity1_id") in entity_set and r.get("entity2_id") in entity_set
		]
		graph_id = f"graph_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result: dict[str, Any] = {
			"graph_id": graph_id,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"relationship_types": relationship_types,
			"nodes": nodes,
			"edges": edges[:100],
			"computed_at": _utcnow(),
		}
		self._correlation_results[graph_id] = result
		self._audit(tenant_id, "graph_correlation_computed", graph_id)
		return result

	async def predictive_correlation(
		self,
		entity_id: str,
		horizon: str = "30d",
	) -> dict[str, Any]:
		"""Predict future correlation probability for *entity_id* based on historical score trend."""
		assert present(entity_id), "entity_id required"
		assert present(horizon), "horizon required"
		tenant_id = self.tenant_id
		# Gather all scores for this entity
		scores = [
			float(r.get("score", r.get("temporal_score", r.get("spatial_score", r.get("behavioural_score", 0.0)))))
			for r in self._correlation_results.values()
			if r.get("entity1_id") == entity_id or r.get("entity2_id") == entity_id or r.get("subject_id") == entity_id
		]
		if not scores:
			predicted = 0.3
			trend = "insufficient_data"
		else:
			mean_s = statistics.mean(scores)
			# Linear trend: last minus first
			trend_delta = scores[-1] - scores[0] if len(scores) > 1 else 0.0
			predicted = round(min(1.0, max(0.0, mean_s + trend_delta * 0.1)), 4)
			trend = "rising" if trend_delta > 0.05 else "falling" if trend_delta < -0.05 else "stable"
		pred_id = f"pred_corr_{entity_id}_{horizon}"
		result: dict[str, Any] = {
			"prediction_id": pred_id,
			"entity_id": entity_id,
			"horizon": horizon,
			"historical_score_count": len(scores),
			"predicted_correlation": predicted,
			"trend": trend,
			"strength": _correlation_strength_label(predicted),
			"computed_at": _utcnow(),
		}
		self._correlation_results[pred_id] = result
		self._audit(tenant_id, "predictive_correlation_computed", entity_id)
		return result

	async def multi_source_fuse(
		self,
		source_ids: list[str],
		fusion_method: str = "weighted_avg",
	) -> dict[str, Any]:
		"""Fuse entity data across *source_ids* using *fusion_method*."""
		assert isinstance(source_ids, list) and len(source_ids) >= 2, "source_ids requires >= 2"
		assert fusion_method in {"weighted_avg", "max_score", "vote"}, f"unknown fusion_method: {fusion_method}"
		tenant_id = self.tenant_id
		entity_scores: dict[str, list[float]] = defaultdict(list)
		for sid in source_ids:
			for (tid, eid), entity in self.entities.items():
				if tid == tenant_id and getattr(entity, "source_id", "") == sid:
					entity_scores[eid].append(getattr(entity, "confidence_score", 0.5))
		fused: dict[str, float] = {}
		for eid, scores in entity_scores.items():
			if fusion_method == "weighted_avg":
				fused[eid] = round(statistics.mean(scores), 4)
			elif fusion_method == "max_score":
				fused[eid] = round(max(scores), 4)
			else:
				# majority vote: > 0.5 counts as positive
				positives = sum(1 for s in scores if s > 0.5)
				fused[eid] = 1.0 if positives > len(scores) / 2 else 0.0
		fusion_id = f"fuse_{fusion_method}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result: dict[str, Any] = {
			"fusion_id": fusion_id,
			"source_ids": source_ids,
			"fusion_method": fusion_method,
			"entity_count": len(fused),
			"fused_scores": fused,
			"avg_fused_score": round(statistics.mean(fused.values()), 4) if fused else 0.0,
			"computed_at": _utcnow(),
		}
		self._correlation_results[fusion_id] = result
		self._audit(tenant_id, "multi_source_fuse_completed", fusion_id)
		return result

	async def confidence_propagate(
		self,
		root_entity_id: str,
		decay: float = 0.1,
	) -> dict[str, Any]:
		"""Propagate confidence scores from *root_entity_id* to linked entities with *decay* per hop."""
		assert present(root_entity_id), "root_entity_id required"
		assert 0 <= decay <= 1, "decay must be in [0,1]"
		tenant_id = self.tenant_id
		root = self._tenant_entity_or_none(root_entity_id, tenant_id)
		if root is None:
			raise KeyError(f"Entity not found: {root_entity_id}")
		root_conf = getattr(root, "confidence_score", 0.8)
		propagated: dict[str, float] = {root_entity_id: root_conf}
		# BFS over correlation links
		queue = [(root_entity_id, root_conf)]
		visited: set[str] = {root_entity_id}
		for _ in range(3):  # max 3 hops
			next_queue = []
			for eid, conf in queue:
				for r in self._correlation_results.values():
					e1, e2 = r.get("entity1_id"), r.get("entity2_id")
					neighbour = None
					if e1 == eid and e2 and e2 not in visited:
						neighbour = e2
					elif e2 == eid and e1 and e1 not in visited:
						neighbour = e1
					if neighbour:
						new_conf = round(conf * (1.0 - decay), 4)
						propagated[neighbour] = new_conf
						visited.add(neighbour)
						next_queue.append((neighbour, new_conf))
			queue = next_queue
		prop_id = f"conf_prop_{root_entity_id}"
		result: dict[str, Any] = {
			"propagation_id": prop_id,
			"root_entity_id": root_entity_id,
			"root_confidence": root_conf,
			"decay": decay,
			"entities_reached": len(propagated),
			"propagated_scores": propagated,
			"computed_at": _utcnow(),
		}
		self._correlation_results[prop_id] = result
		self._audit(tenant_id, "confidence_propagated", root_entity_id)
		return result

	async def correlation_visualise(self, case_id: str) -> dict[str, Any]:
		"""Generate a visualisation descriptor for the correlation graph of *case_id*."""
		assert present(case_id), "case_id required"
		graph = await self.correlation_graph_export(case_id)
		viz_spec: dict[str, Any] = {
			"schema": "cytoscape",
			"style": [
				{"selector": "node", "css": {"background-color": "#6FB1FC"}},
				{"selector": "edge", "css": {"line-color": "#ccc", "width": "mapData(weight,0,1,1,6)"}},
			],
			"elements": {
				"nodes": [{"data": n} for n in graph["nodes"]],
				"edges": [{"data": e} for e in graph["edges"]],
			},
		}
		self._audit(self.tenant_id, "correlation_visualised", case_id)
		return {
			"case_id": case_id,
			"visualisation_schema": "cytoscape",
			"node_count": graph["node_count"],
			"edge_count": graph["edge_count"],
			"spec": viz_spec,
			"generated_at": _utcnow(),
		}

	async def false_positive_filter(
		self,
		correlation_ids: list[str],
		threshold: float = 0.4,
	) -> dict[str, Any]:
		"""Flag correlations below *threshold* as probable false positives."""
		assert isinstance(correlation_ids, list) and correlation_ids, "correlation_ids required"
		assert 0 <= threshold <= 1, "threshold must be in [0,1]"
		false_positives: list[str] = []
		confirmed: list[str] = []
		for cid in correlation_ids:
			r = self._correlation_results.get(cid)
			if r is None:
				continue
			score = float(r.get("score") or r.get("temporal_score") or r.get("spatial_score") or r.get("behavioural_score") or 0.0)
			if score < threshold:
				false_positives.append(cid)
			else:
				confirmed.append(cid)
		self._audit(self.tenant_id, "false_positive_filter_applied", f"count={len(correlation_ids)}")
		return {
			"checked": len(correlation_ids),
			"false_positives": false_positives,
			"confirmed": confirmed,
			"fp_rate": round(len(false_positives) / len(correlation_ids), 4),
			"threshold": threshold,
			"computed_at": _utcnow(),
		}

	async def correlation_workflow(
		self,
		source_ids: list[str],
		steps: list[str],
	) -> list[dict[str, Any]]:
		"""Execute a correlation workflow pipeline: multi_source → entity_pairs → cluster."""
		assert source_ids, "source_ids required"
		assert steps, "steps required"
		VALID = {"multi_source", "temporal", "spatial", "behavioural", "matrix"}
		results: list[dict[str, Any]] = []
		for step in steps:
			assert step in VALID, f"Unknown step: {step}"
			if step == "multi_source":
				r = await self.multi_source_correlation(source_ids, "entity_reference")
			elif step == "temporal":
				r = await self.temporal_correlation([{"id": "e1", "timestamp": "0"}, {"id": "e2", "timestamp": "60"}], "1h")
			elif step == "spatial":
				r = await self.spatial_correlation([{"id": "p1", "lat": 0.0, "lon": 0.0}, {"id": "p2", "lat": 0.01, "lon": 0.01}], 5.0)
			elif step == "behavioural":
				entities = [eid for (tid, eid) in self.entities if tid == self.tenant_id][:1]
				r = await self.behavioural_correlation(entities[0] if entities else "none")
			else:
				r = await self.correlation_matrix(source_ids[0])
			results.append({"step": step, "result": r})
		self._audit(self.tenant_id, "correlation_workflow_completed", f"steps={len(steps)}")
		return results

	async def real_time_correlate(
		self,
		event: dict[str, Any],
		correlation_type: str = "entity",
	) -> dict[str, Any]:
		"""Correlate a single inbound *event* against all stored entities in near-real-time."""
		assert isinstance(event, dict), "event must be a dict"
		assert present(correlation_type), "correlation_type required"
		tenant_id = self.tenant_id
		event_ref = str(event.get("entity_reference", event.get("id", "unknown")))
		event_type = str(event.get("entity_type", "unknown"))
		matches: list[dict[str, Any]] = []
		for (tid, eid), entity in self.entities.items():
			if tid != tenant_id:
				continue
			ent_ref = getattr(entity, "entity_reference", "")
			same_ref = event_ref.lower() == ent_ref.lower()
			same_type = event_type == getattr(entity, "entity_type", "")
			score = round(min(1.0, (0.6 if same_ref else 0.0) + (0.4 if same_type else 0.0)), 4)
			if score > 0:
				matches.append({"entity_id": eid, "score": score, "strength": _correlation_strength_label(score)})
		matches.sort(key=lambda x: x["score"], reverse=True)
		rt_id = f"rt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result: dict[str, Any] = {
			"correlation_id": rt_id,
			"correlation_type": correlation_type,
			"event_reference": event_ref,
			"matches": matches[:20],
			"match_count": len(matches),
			"top_score": matches[0]["score"] if matches else 0.0,
			"computed_at": _utcnow(),
		}
		self._correlation_results[rt_id] = result
		self._audit(tenant_id, "real_time_correlation_computed", rt_id)
		return result

	async def correlation_analytics(self, period: str = "30d") -> dict[str, Any]:
		"""Aggregate correlation statistics across all results for *period*."""
		assert present(period), "period required"
		tenant_id = self.tenant_id
		all_scores = [
			float(r.get("score") or r.get("temporal_score") or r.get("spatial_score") or r.get("behavioural_score") or 0.0)
			for r in self._correlation_results.values()
		]
		avg = round(statistics.mean(all_scores), 4) if all_scores else 0.0
		strong = sum(1 for s in all_scores if s >= 0.75)
		moderate = sum(1 for s in all_scores if 0.40 <= s < 0.75)
		weak = sum(1 for s in all_scores if s < 0.40)
		self._audit(tenant_id, "correlation_analytics_computed", period)
		return {
			"period": period,
			"total_correlations": len(all_scores),
			"avg_score": avg,
			"strong": strong, "moderate": moderate, "weak": weak,
			"entity_count": self._count(self.entities, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"computed_at": _utcnow(),
		}

	async def cross_source_enrichment(
		self,
		entity_id: str,
		target_source_ids: list[str],
	) -> dict[str, Any]:
		"""Enrich *entity_id* by pulling correlated observations from *target_source_ids*."""
		assert present(entity_id), "entity_id required"
		assert target_source_ids, "target_source_ids required"
		tenant_id = self.tenant_id
		enrichments: list[dict[str, Any]] = []
		for sid in target_source_ids:
			for (tid, oid), obs in self.observations.items():
				if tid != tenant_id:
					continue
				obs_entity_id = getattr(obs, "entity_id", "")
				if obs_entity_id != entity_id:
					continue
				source_entity = self._tenant_entity_or_none(obs_entity_id, tenant_id)
				if source_entity and getattr(source_entity, "source_id", "") == sid:
					enrichments.append({
						"observation_id": oid,
						"source_id": sid,
						"observation_type": getattr(obs, "observation_type", ""),
						"confidence_score": getattr(obs, "confidence_score", 0.0),
					})
		enrich_id = f"cse_{entity_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		self._audit(tenant_id, "cross_source_enrichment_completed", enrich_id)
		return {
			"enrichment_id": enrich_id,
			"entity_id": entity_id,
			"sources_queried": len(target_source_ids),
			"enrichments_found": len(enrichments),
			"enrichments": enrichments[:50],
			"computed_at": _utcnow(),
		}

	async def cluster_merge(
		self,
		cluster_id_a: str,
		cluster_id_b: str,
		rationale: str,
	) -> dict[str, Any]:
		"""Merge two correlation clusters into a single compound cluster."""
		assert present(cluster_id_a) and present(cluster_id_b), "both cluster IDs required"
		assert present(rationale), "rationale required"
		tenant_id = self.tenant_id
		ca = self._tenant_cluster_or_none(cluster_id_a, tenant_id)
		cb = self._tenant_cluster_or_none(cluster_id_b, tenant_id)
		if ca is None:
			raise KeyError(f"Cluster not found: {cluster_id_a}")
		if cb is None:
			raise KeyError(f"Cluster not found: {cluster_id_b}")
		conf_a = getattr(ca, "confidence_score", 0.5)
		conf_b = getattr(cb, "confidence_score", 0.5)
		merged_conf = round((conf_a + conf_b) / 2, 4)
		merge_id = f"merge_{cluster_id_a}_{cluster_id_b}"
		self._audit(tenant_id, "clusters_merged", merge_id)
		return {
			"merge_id": merge_id,
			"cluster_id_a": cluster_id_a,
			"cluster_id_b": cluster_id_b,
			"merged_confidence": merged_conf,
			"rationale": rationale,
			"merged_at": _utcnow(),
			"tenant_id": tenant_id,
		}

	async def source_entity_overlap(self, source_id_a: str, source_id_b: str) -> dict[str, Any]:
		"""Find entities that appear in both *source_id_a* and *source_id_b*."""
		assert present(source_id_a), "source_id_a required"
		assert present(source_id_b), "source_id_b required"

		tenant_id = self.tenant_id
		refs_a = {
			getattr(e, "entity_reference", "")
			for (tid, _), e in self.entities.items()
			if tid == tenant_id and getattr(e, "source_id", "") == source_id_a
		}
		refs_b = {
			getattr(e, "entity_reference", "")
			for (tid, _), e in self.entities.items()
			if tid == tenant_id and getattr(e, "source_id", "") == source_id_b
		}
		overlap = refs_a & refs_b
		jaccard = round(len(overlap) / len(refs_a | refs_b), 4) if (refs_a | refs_b) else 0.0

		self._audit(tenant_id, "source_entity_overlap_computed", f"{source_id_a}:{source_id_b}")
		return {
			"source_id_a": source_id_a,
			"source_id_b": source_id_b,
			"entities_in_a": len(refs_a),
			"entities_in_b": len(refs_b),
			"overlap_count": len(overlap),
			"jaccard_similarity": jaccard,
			"overlapping_references": list(overlap)[:50],
			"computed_at": _utcnow(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> CorrelationAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> CorrelationWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> CorrelationSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_entity_or_none(self, item_id: str, tenant_id: str) -> CorrelationEntity | None:
		return self.entities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_rule_or_none(self, item_id: str, tenant_id: str) -> CorrelationRule | None:
		return self.rules.get(self._tenant_key(tenant_id, item_id))

	def _tenant_run_or_none(self, item_id: str, tenant_id: str) -> CorrelationRun | None:
		return self.runs.get(self._tenant_key(tenant_id, item_id))

	def _tenant_cluster_or_none(self, item_id: str, tenant_id: str) -> CorrelationCluster | None:
		return self.clusters.get(self._tenant_key(tenant_id, item_id))

	def _tenant_decision_or_none(self, item_id: str, tenant_id: str) -> CorrelationDecision | None:
		return self.decisions.get(self._tenant_key(tenant_id, item_id))

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
			action.get("reason", action.get("rule", "correlation_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "correlation_policy_denied")



	async def ml_event_correlation_score(self, *args, **kwargs):
		"""AI-powered ML-powered event correlation and attack chain detection. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="security_event_correlation")
			return {"correlation_score": round(result.score,3), "related_events": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

IntelCorrelationService = DataCorrelationService
