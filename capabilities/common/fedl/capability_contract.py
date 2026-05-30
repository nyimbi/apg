"""Executable capability contract for APG Federated Learning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"federation": {
		"coordinator_enabled": True,
		"participant_attestation_required": True,
		"participant_contract_required": True,
		"minimum_participants": 3,
		"max_participants_per_federation": 1000,
	},
	"participants": {
		"attestation_required": True,
		"contract_required": True,
		"data_residency_required": True,
		"compute_profile_required": True,
		"heartbeat_required": True,
	},
	"privacy": {
		"secure_aggregation_required": True,
		"differential_privacy_enabled": True,
		"max_privacy_epsilon": 8.0,
		"privacy_review_required_above_epsilon": 8.0,
		"budget_ledger_required": True,
	},
	"training": {
		"round_approval_required": True,
		"model_update_validation": True,
		"poisoning_detection_enabled": True,
		"minimum_quality_score": 0.75,
		"all_participant_updates_required": True,
	},
	"aggregation": {
		"secure_aggregation_required": True,
		"quarantine_poisoned_updates": True,
		"aggregate_digest_required": True,
		"lineage_required": True,
	},
	"model_release": {
		"mlcm_registration_required": True,
		"release_approval_required": True,
		"privacy_review_required": True,
		"artifact_lineage_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"auth_required": True,
		"data_residency_required": True,
		"audit_rounds": True,
		"participant_contract_required": True,
		"cross_tenant_participation_allowed": False,
		"federation_retirement_review_required": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
		"participant_heartbeat_required": True,
	},
	"adapters": {
		"generated_app_runtime": "service.FedlService",
		"production_runtime": "service.FedlService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"ai_core": "aicr",
		"model_lifecycle": "mlcm",
		"encryption": "encr",
		"tenant_service": "mten",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"metrics_sink": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_federation_console": True,
		"enable_participant_map": True,
		"enable_attestation_center": True,
		"enable_round_monitor": True,
		"enable_update_queue": True,
		"enable_aggregation_console": True,
		"enable_privacy_budget": True,
		"enable_security_console": True,
		"enable_model_registry": True,
		"enable_release_console": True,
		"enable_audit_timeline": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "fedl_privacy_mesh", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"federation",
		"participants",
		"privacy",
		"training",
		"aggregation",
		"model_release",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"federation",
		"participants",
		"privacy",
		"training",
		"aggregation",
		"model_release",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All federated learning operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "federation_requires_coordinator", "description": "Federations require a coordinator.", "condition": {"operation": "create_federation", "coordinator_present": False}, "effect": {"decision": "deny", "reason": "coordinator_required", "required_action": "assign_federation_coordinator"}},
	{"name": "federation_requires_model_family", "description": "Federations require model-family metadata.", "condition": {"operation": "create_federation", "model_family_present": False}, "effect": {"decision": "deny", "reason": "model_family_required", "required_action": "attach_model_family"}},
	{"name": "federation_requires_objective_metric", "description": "Federations require an objective metric.", "condition": {"operation": "create_federation", "objective_metric_present": False}, "effect": {"decision": "deny", "reason": "objective_metric_required", "required_action": "attach_objective_metric"}},
	{"name": "federation_requires_data_residency", "description": "Federations require data-residency regions.", "condition": {"operation": "create_federation", "data_residency_present": False}, "effect": {"decision": "deny", "reason": "data_residency_required", "required_action": "attach_data_residency_regions"}},
	{"name": "privacy_budget_must_be_positive", "description": "Federation privacy budgets must be positive.", "condition": {"operation": "create_federation", "privacy_budget_positive": False}, "effect": {"decision": "deny", "reason": "privacy_budget_required", "required_action": "set_privacy_epsilon_limit"}},
	{"name": "participant_requires_attestation", "description": "Participants require attestation before joining.", "condition": {"operation": "join_federation", "participant_attested": False}, "effect": {"decision": "deny", "reason": "participant_attestation_required", "required_action": "complete_participant_attestation"}},
	{"name": "participant_requires_contract", "description": "Participants require contract evidence.", "condition": {"operation": "join_federation", "contract_ref_present": False}, "effect": {"decision": "deny", "reason": "participant_contract_required", "required_action": "attach_participant_contract"}},
	{"name": "participant_region_must_be_allowed", "description": "Participant regions must match federation residency rules.", "condition": {"operation": "join_federation", "region_allowed": False}, "effect": {"decision": "deny", "reason": "data_residency_required", "required_action": "choose_allowed_region"}},
	{"name": "participant_requires_compute_profile", "description": "Participants require compute-profile metadata.", "condition": {"operation": "join_federation", "compute_profile_present": False}, "effect": {"decision": "require_review", "reason": "compute_profile_required", "required_action": "attach_compute_profile"}},
	{"name": "round_requires_minimum_participants", "description": "Training rounds require enough participants.", "condition": {"participant_count_lt": 3, "operation": "start_round"}, "effect": {"decision": "deny", "reason": "minimum_participants_required", "required_action": "add_participants"}},
	{"name": "round_requires_approval", "description": "Training rounds require approval evidence.", "condition": {"operation": "start_round", "approval_ref_present": False}, "effect": {"decision": "deny", "reason": "round_approval_required", "required_action": "record_round_approval"}},
	{"name": "round_requires_secure_aggregation", "description": "Training rounds require secure aggregation.", "condition": {"operation": "start_round", "secure_aggregation_enabled": False}, "effect": {"decision": "deny", "reason": "secure_aggregation_required", "required_action": "enable_secure_aggregation"}},
	{"name": "privacy_budget_requires_review", "description": "High privacy budget requires review.", "condition": {"privacy_epsilon_gt": 8.0, "privacy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "privacy_budget_review_required", "required_action": "record_privacy_review"}},
	{"name": "round_budget_must_fit_federation_limit", "description": "Round privacy budget cannot exceed federation limit.", "condition": {"operation": "start_round", "privacy_within_limit": False}, "effect": {"decision": "deny", "reason": "privacy_budget_exceeds_federation_limit", "required_action": "lower_round_privacy_epsilon"}},
	{"name": "update_requires_round_running", "description": "Updates require a running round.", "condition": {"operation": "submit_update", "round_running": False}, "effect": {"decision": "deny", "reason": "round_not_running", "required_action": "start_or_choose_running_round"}},
	{"name": "update_requires_round_participant", "description": "Updates require a participant assigned to the round.", "condition": {"operation": "submit_update", "participant_in_round": False}, "effect": {"decision": "deny", "reason": "participant_not_in_round", "required_action": "assign_participant_to_round"}},
	{"name": "update_requires_sample_count", "description": "Participant updates require positive sample counts.", "condition": {"operation": "submit_update", "sample_count_positive": False}, "effect": {"decision": "deny", "reason": "sample_count_required", "required_action": "attach_sample_count"}},
	{"name": "update_quality_score_must_be_in_range", "description": "Update quality scores must be in range.", "condition": {"operation": "submit_update", "quality_score_in_range": False}, "effect": {"decision": "deny", "reason": "quality_score_out_of_range", "required_action": "submit_quality_score_between_zero_and_one"}},
	{"name": "low_quality_update_requires_review", "description": "Low-quality updates require review before aggregation.", "condition": {"operation": "submit_update", "quality_score_lt": 0.75, "quality_review_recorded": False}, "effect": {"decision": "require_review", "reason": "update_quality_review_required", "required_action": "record_update_quality_review"}},
	{"name": "secure_aggregation_required", "description": "Federated updates require secure aggregation.", "condition": {"secure_aggregation_enabled": False, "operation": "aggregate_updates"}, "effect": {"decision": "deny", "reason": "secure_aggregation_required", "required_action": "enable_secure_aggregation"}},
	{"name": "aggregation_requires_complete_updates", "description": "Aggregation requires accepted updates from all round participants.", "condition": {"operation": "aggregate_updates", "participant_updates_complete": False}, "effect": {"decision": "deny", "reason": "participant_updates_incomplete", "required_action": "collect_remaining_updates"}},
	{"name": "poisoning_signal_blocks_round", "description": "Poisoning signals block model aggregation.", "condition": {"poisoning_signal_detected": True, "operation": "aggregate_updates"}, "effect": {"decision": "deny", "reason": "poisoning_signal_detected", "required_action": "quarantine_suspicious_update"}},
	{"name": "aggregation_requires_digest", "description": "Aggregation results require aggregate digest evidence.", "condition": {"operation": "aggregate_updates", "aggregate_digest_present": False}, "effect": {"decision": "deny", "reason": "aggregate_digest_required", "required_action": "record_aggregate_digest"}},
	{"name": "model_release_requires_approval", "description": "Federated model release requires approval.", "condition": {"operation": "release_model", "release_approval_recorded": False}, "effect": {"decision": "deny", "reason": "model_release_approval_required", "required_action": "record_model_release_approval"}},
	{"name": "model_release_requires_mlcm_registration", "description": "Federated models must be released through MLCM linkage.", "condition": {"operation": "release_model", "mlcm_model_ref_present": False}, "effect": {"decision": "deny", "reason": "mlcm_model_ref_required", "required_action": "link_mlcm_model"}},
	{"name": "model_release_requires_privacy_review", "description": "Federated model release requires privacy review evidence.", "condition": {"operation": "release_model", "privacy_review_recorded": False}, "effect": {"decision": "deny", "reason": "model_release_privacy_review_required", "required_action": "record_release_privacy_review"}},
	{"name": "federation_retirement_requires_review", "description": "Federation retirement requires impact review.", "condition": {"operation": "retire_federation", "impact_review_recorded": False}, "effect": {"decision": "deny", "reason": "federation_retirement_review_required", "required_action": "record_retirement_impact"}},
	{"name": "cross_tenant_participation_denied", "description": "Cross-tenant federation participation is denied by default.", "condition": {"cross_tenant_participation": True}, "effect": {"decision": "deny", "reason": "cross_tenant_participation_denied", "required_action": "use_tenant_scoped_federation"}},
	{"name": "bytewax_stream_required_for_round_events", "description": "Federated round event streams must use Bytewax.", "condition": {"operation": "configure_round_events", "event_stream": "kafka"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/fedl/dashboard", "component": "FEDLDashboard", "permission": "fedl:view", "nav_group": "Overview"},
	{"name": "federations", "path": "/fedl/federations", "component": "FederationConsole", "permission": "fedl:manage_federations", "nav_group": "Federations"},
	{"name": "participants", "path": "/fedl/participants", "component": "ParticipantMap", "permission": "fedl:view_participants", "nav_group": "Federations"},
	{"name": "attestation", "path": "/fedl/attestation", "component": "ParticipantAttestation", "permission": "fedl:manage_federations", "nav_group": "Federations"},
	{"name": "rounds", "path": "/fedl/rounds", "component": "TrainingRoundMonitor", "permission": "fedl:run_rounds", "nav_group": "Training"},
	{"name": "updates", "path": "/fedl/updates", "component": "ModelUpdateQueue", "permission": "fedl:run_rounds", "nav_group": "Training"},
	{"name": "aggregation", "path": "/fedl/aggregation", "component": "SecureAggregationConsole", "permission": "fedl:run_rounds", "nav_group": "Training"},
	{"name": "privacy", "path": "/fedl/privacy", "component": "PrivacyBudgetConsole", "permission": "fedl:manage_privacy", "nav_group": "Governance"},
	{"name": "security", "path": "/fedl/security", "component": "PoisoningDefense", "permission": "fedl:manage_security", "nav_group": "Governance"},
	{"name": "models", "path": "/fedl/models", "component": "FederatedModelRegistry", "permission": "fedl:view_models", "nav_group": "Models"},
	{"name": "release", "path": "/fedl/release", "component": "ModelReleaseConsole", "permission": "fedl:release_models", "nav_group": "Models"},
	{"name": "audit", "path": "/fedl/audit", "component": "FEDLAuditTimeline", "permission": "fedl:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/fedl/settings", "component": "FEDLSettings", "permission": "fedl:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "fedl_privacy_mesh",
	"tokens": {
		"color.primary": "#1E5F74",
		"color.accent": "#9B5DE5",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"participant_node_card": {"icon": "nodes", "status_indicator": "attestation-pill", "risk_style": "privacy-band"},
		"training_round_timeline": {"visual": "round-checkpoints", "highlight": "aggregation-chip"},
		"privacy_budget_meter": {"visual": "segmented-meter", "threshold_style": "epsilon-bands"},
		"federation_topology": {"visual": "privacy-mesh", "edge_style": "secure-channel-line"},
		"attestation_panel": {"icon": "shield-check", "status_indicator": "contract-chip", "risk_style": "residency-band"},
		"update_queue": {"visual": "update-ledger", "status_style": "quality-chip"},
		"aggregation_console": {"visual": "secure-merge", "highlight": "digest-chip"},
		"release_console": {"visual": "model-release-ladder", "status_style": "approval-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable FEDL capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "fedl",
		"display_name": "Federated Learning",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "__init__.py",
			"api_prefix": "/fedl/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
