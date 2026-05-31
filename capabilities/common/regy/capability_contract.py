"""Executable capability contract for APG API/Service Registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_REGY_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_REGY_AGENT_ROLES = [
	"registration_reviewer",
	"contract_reviewer",
	"discovery_reviewer",
	"health_reviewer",
	"gateway_sync_reviewer",
	"owner_transfer_reviewer",
	"retirement_reviewer",
	"catalog_steward",
]
PRIVILEGED_REGY_AGENT_ROLES = [
	"registration_reviewer",
	"contract_reviewer",
	"discovery_reviewer",
	"health_reviewer",
	"gateway_sync_reviewer",
	"owner_transfer_reviewer",
	"retirement_reviewer",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped REGY configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"registration": {
			"owner_required": True,
			"health_endpoint_required": True,
			"api_version_required": True,
			"contract_schema_required": True,
			"allowed_protocols": ["http", "https", "grpc", "websocket", "amqp", "mqtt"],
			"max_services_per_tenant": 10000,
		},
		"instances": {
			"owner_required": True,
			"endpoint_required": True,
			"health_probe_required": True,
			"region_required": True,
			"weight_required": True,
			"allowed_regions": ["local", "edge-africa", "edge-eu", "edge-east", "edge-west"],
		},
		"contracts": {
			"schema_required": True,
			"version_required": True,
			"breaking_change_review_required": True,
			"deprecation_plan_required": True,
			"migration_notes_required": True,
		},
		"discovery": {
			"service_discovery_enabled": True,
			"cache_ttl_seconds": 60,
			"prefer_healthy_instances": True,
			"cross_tenant_discovery_allowed": False,
			"max_results_without_review": 1000,
		},
		"health": {
			"active_health_checks_enabled": True,
			"default_interval_seconds": 30,
			"minimum_interval_seconds": 5,
			"failure_threshold": 3,
			"degraded_blocks_gateway_publish": True,
		},
		"routing": {
			"gateway_sync_enabled": True,
			"load_balancing_metadata_required": True,
			"supported_strategies": ["round_robin", "weighted", "least_latency", "failover"],
			"circuit_breaking_enabled": True,
			"publish_requires_healthy_instance": True,
		},
		"governance": {
			"require_tenant_context": True,
			"audit_registration_events": True,
			"duplicate_service_names_blocked": True,
			"production_registration_review_required": True,
			"owner_transfer_review_required": True,
			"retirement_impact_review_required": True,
		},
		"observability": {
			"metrics_required": True,
			"traces_required_for_production": True,
			"audit_events_required": True,
			"lineage_capture_required": True,
		},
		"adapters": {
			"production_runtime": "service.ServiceRegistryService",
			"generated_app_runtime": "registry_runtime.RegistryService",
			"http_api": "api.registry_bp",
			"service_discovery": "conf",
			"gateway_sync": "apig",
			"metrics_sink": "moni",
			"audit_sink": "audl",
			"auth_provider": "auth",
			"event_stream": "bytewax",
			"cache_store": "cach",
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_REGY_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_REGY_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_REGY_AGENT_ROLES,
			"scope_required": True,
			"owner_required": True,
			"purpose_required": True,
			"contribution_disclosure_required": True,
			"human_approval_required_for_privileged_roles": True,
		},
		"streaming": {
			"engine": "bytewax",
			"required_processor": "bytewax",
			"lifecycle_stream": "regy.lifecycle",
			"watermark": "event_time",
			"operations": [
				"service_batch",
				"instance_batch",
				"version_batch",
				"discovery_batch",
				"gateway_publication_batch",
				"review_batch",
				"registry_agent_batch",
			],
			"topics": [
				"regy.services",
				"regy.instances",
				"regy.versions",
				"regy.discovery",
				"regy.gateway_publications",
				"regy.reviews",
				"regy.agents",
			],
		},
		"ui": {
			"enable_service_catalog": True,
			"enable_registration_console": True,
			"enable_discovery_console": True,
			"enable_instance_manager": True,
			"enable_health_dashboard": True,
			"enable_version_manager": True,
			"enable_contract_review": True,
			"enable_gateway_sync": True,
			"enable_retirement_reviews": True,
			"enable_audit_timeline": True,
			"enable_registry_agent_roster": True,
			"enable_lifecycle_batch_monitor": True,
			"enable_settings": True,
		},
		"theme": {
			"default_theme": "regy_service_catalog",
			"allow_tenant_overrides": True,
		},
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"registration",
			"instances",
			"contracts",
			"discovery",
			"health",
			"routing",
			"governance",
			"observability",
			"adapters",
			"agents",
			"streaming",
			"ui",
			"theme",
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"registration": {"type": "object"},
			"instances": {"type": "object"},
			"contracts": {"type": "object"},
			"discovery": {"type": "object"},
			"health": {"type": "object"},
			"routing": {"type": "object"},
			"governance": {"type": "object"},
			"observability": {"type": "object"},
			"adapters": {"type": "object"},
			"agents": {"type": "object"},
			"streaming": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"},
		},
	})

	def for_tenant(self, tenant_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Return configuration with tenant-specific overrides applied."""
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id is required"
		merged = _deep_copy(self.defaults)
		merged["tenant_id"] = tenant_id
		if overrides:
			_deep_merge(merged, overrides)
		return merged


@dataclass(frozen=True)
class CapabilityRule:
	"""REGY policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic REGY rule engine for registry control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching registry governance rules."""
		assert isinstance(context, dict), "context must be a dictionary"
		matched: list[str] = []
		actions: list[dict[str, Any]] = []
		decision = "allow"
		for rule in self.rules:
			if _matches(rule.condition, context):
				matched.append(rule.name)
				actions.append(rule.effect)
				if rule.effect.get("decision") == "deny":
					decision = "deny"
				elif rule.effect.get("decision") == "require_review" and decision != "deny":
					decision = "require_review"
		return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


@dataclass(frozen=True)
class CapabilityUIRoute:
	"""UI route exposed by REGY."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for REGY UI surfaces."""

	name: str = "regy_service_catalog"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#1F5E5A",
		"color.accent": "#C86F2D",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8F7",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"service_catalog_row": {"icon": "network", "status_indicator": "health-pill", "risk_style": "version-band"},
		"registration_console": {"visual": "service-form", "status_indicator": "guardrail-chip"},
		"discovery_result_card": {"visual": "instance-stack", "highlight": "endpoint-chip"},
		"instance_manager": {"visual": "endpoint-table", "status_indicator": "region-pill"},
		"health_check_timeline": {"visual": "probe-timeline", "status_style": "failure-threshold"},
		"version_compatibility_panel": {"visual": "version-matrix", "highlight": "breaking-change-chip"},
		"contract_review_queue": {"visual": "review-list", "highlight": "schema-chip"},
		"gateway_sync_panel": {"visual": "gateway-link", "status_indicator": "publish-chip"},
		"retirement_review_panel": {"visual": "impact-list", "status_indicator": "retirement-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "decision-pill"},
		"registry_agent_roster": {"visual": "agent-roster", "status_indicator": "approval-chip"},
		"bytewax_lifecycle_panel": {"visual": "stream-batches", "status_indicator": "processor-chip"},
	})


def default_rules() -> list[CapabilityRule]:
	"""Default REGY rules available to every tenant."""
	return [
		CapabilityRule("tenant_context_required", "All registry operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("service_registration_requires_owner", "Service registration requires an accountable owner.", {"operation": "register_service", "owner_assigned": False}, {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"}),
		CapabilityRule("service_registration_requires_health_endpoint", "Service registration requires health endpoint metadata.", {"operation": "register_service", "health_endpoint_present": False}, {"decision": "deny", "reason": "health_endpoint_required", "required_action": "attach_health_endpoint"}),
		CapabilityRule("service_registration_requires_api_version", "Service registration requires an API version.", {"operation": "register_service", "api_version_present": False}, {"decision": "deny", "reason": "api_version_required", "required_action": "declare_api_version"}),
		CapabilityRule("service_registration_requires_contract_schema", "Service registration requires API or service contract schema evidence.", {"operation": "register_service", "contract_schema_present": False}, {"decision": "deny", "reason": "contract_schema_required", "required_action": "attach_contract_schema"}),
		CapabilityRule("duplicate_service_name_blocked", "Duplicate service names are blocked within tenant scope.", {"operation": "register_service", "duplicate_service_name": True}, {"decision": "deny", "reason": "duplicate_service_name", "required_action": "choose_unique_service_name"}),
		CapabilityRule("production_registration_requires_review", "Production service registration requires review evidence.", {"operation": "register_service", "environment": "production", "production_review_recorded": False}, {"decision": "require_review", "reason": "production_registration_review_required", "required_action": "record_production_registration_review"}),
		CapabilityRule("instance_requires_endpoint", "Service instances require endpoint metadata.", {"operation": "register_instance", "endpoint_present": False}, {"decision": "deny", "reason": "instance_endpoint_required", "required_action": "attach_instance_endpoint"}),
		CapabilityRule("instance_requires_health_probe", "Service instances require health probe evidence.", {"operation": "register_instance", "health_probe_present": False}, {"decision": "deny", "reason": "instance_health_probe_required", "required_action": "attach_instance_health_probe"}),
		CapabilityRule("instance_requires_allowed_region", "Service instances must target an allowed region.", {"operation": "register_instance", "allowed_region": False}, {"decision": "deny", "reason": "allowed_region_required", "required_action": "choose_allowed_region"}),
		CapabilityRule("instance_requires_positive_weight", "Load-balancing weights must be positive.", {"operation": "register_instance", "positive_weight": False}, {"decision": "deny", "reason": "positive_weight_required", "required_action": "set_positive_instance_weight"}),
		CapabilityRule("discovery_cross_tenant_denied", "Cross-tenant discovery is denied by default.", {"operation": "discover_services", "cross_tenant_discovery": True}, {"decision": "deny", "reason": "cross_tenant_discovery_denied", "required_action": "use_tenant_scoped_discovery"}),
		CapabilityRule("discovery_high_result_limit_requires_review", "Large discovery result limits require review.", {"operation": "discover_services", "requested_result_limit_gt": 1000, "discovery_review_recorded": False}, {"decision": "require_review", "reason": "discovery_limit_review_required", "required_action": "record_discovery_limit_review"}),
		CapabilityRule("gateway_publish_requires_registered_service", "Gateway publication requires a registered service.", {"operation": "publish_to_gateway", "service_registered": False}, {"decision": "deny", "reason": "registered_service_required", "required_action": "register_service_first"}),
		CapabilityRule("gateway_publish_requires_reviewed_service", "Gateway publication requires completed service review.", {"operation": "publish_to_gateway", "service_review_complete": False}, {"decision": "deny", "reason": "service_review_required", "required_action": "complete_service_review"}),
		CapabilityRule("gateway_publish_requires_healthy_instance", "Gateway publication requires at least one healthy instance.", {"operation": "publish_to_gateway", "healthy_instance_present": False}, {"decision": "deny", "reason": "healthy_instance_required", "required_action": "restore_or_register_healthy_instance"}),
		CapabilityRule("gateway_publish_requires_routing_metadata", "Gateway publication requires load-balancing and route metadata.", {"operation": "publish_to_gateway", "routing_metadata_present": False}, {"decision": "deny", "reason": "routing_metadata_required", "required_action": "attach_routing_metadata"}),
		CapabilityRule("breaking_change_requires_review", "Breaking API changes require compatibility review.", {"operation": "record_version", "breaking_change_detected": True, "compatibility_review_recorded": False}, {"decision": "require_review", "reason": "compatibility_review_required", "required_action": "record_compatibility_review"}),
		CapabilityRule("deprecation_requires_migration_notes", "Deprecated versions require migration notes.", {"operation": "deprecate_version", "migration_notes_present": False}, {"decision": "deny", "reason": "migration_notes_required", "required_action": "attach_migration_notes"}),
		CapabilityRule("deprecation_requires_future_eol", "Deprecated versions require a future end-of-life date.", {"operation": "deprecate_version", "future_eol_date": False}, {"decision": "deny", "reason": "future_eol_required", "required_action": "choose_future_eol_date"}),
		CapabilityRule("health_override_requires_incident_reference", "Manual health overrides require an incident or change reference.", {"operation": "override_health", "incident_reference_present": False}, {"decision": "deny", "reason": "incident_reference_required", "required_action": "attach_incident_reference"}),
		CapabilityRule("owner_transfer_requires_review", "Service ownership transfer requires review.", {"operation": "transfer_owner", "owner_transfer_review_recorded": False}, {"decision": "require_review", "reason": "owner_transfer_review_required", "required_action": "record_owner_transfer_review"}),
		CapabilityRule("service_retirement_requires_impact_review", "Retiring a service requires impact review.", {"operation": "retire_service", "impact_review_recorded": False}, {"decision": "deny", "reason": "impact_review_required", "required_action": "record_retirement_impact_review"}),
		CapabilityRule("service_retirement_requires_gateway_unpublish", "Retiring a service requires gateway unpublish evidence.", {"operation": "retire_service", "gateway_unpublish_recorded": False}, {"decision": "deny", "reason": "gateway_unpublish_required", "required_action": "record_gateway_unpublish"}),
		CapabilityRule("production_requires_tracing", "Production services require trace propagation evidence.", {"operation": "register_service", "environment": "production", "trace_propagation_configured": False}, {"decision": "deny", "reason": "trace_propagation_required", "required_action": "configure_trace_propagation"}),
		CapabilityRule("registry_agent_runtime_supported", "Registry agents must use a supported runtime adapter.", {"operation": "register_registry_agent", "agent_runtime_supported": False}, {"decision": "deny", "reason": "unsupported_registry_agent_runtime", "required_action": "choose_supported_registry_agent_runtime"}),
		CapabilityRule("registry_agent_role_supported", "Registry agents must use a supported registry role.", {"operation": "register_registry_agent", "agent_role_supported": False}, {"decision": "deny", "reason": "unsupported_registry_agent_role", "required_action": "choose_supported_registry_agent_role"}),
		CapabilityRule("registry_agent_requires_scope", "Registry agents require an explicit bounded scope.", {"operation": "register_registry_agent", "scope_present": False}, {"decision": "deny", "reason": "registry_agent_scope_required", "required_action": "declare_registry_agent_scope"}),
		CapabilityRule("registry_agent_requires_owner", "Registry agents require an accountable owner.", {"operation": "register_registry_agent", "owner_present": False}, {"decision": "deny", "reason": "registry_agent_owner_required", "required_action": "assign_registry_agent_owner"}),
		CapabilityRule("registry_agent_requires_purpose", "Registry agents require a documented purpose.", {"operation": "register_registry_agent", "purpose_present": False}, {"decision": "deny", "reason": "registry_agent_purpose_required", "required_action": "document_registry_agent_purpose"}),
		CapabilityRule("registry_agent_requires_contribution_disclosure", "Registry agents must disclose machine-authored registry contributions.", {"operation": "register_registry_agent", "contribution_disclosed": False}, {"decision": "deny", "reason": "registry_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}),
		CapabilityRule("registry_agent_privileged_role_requires_human_approval", "Privileged registry-agent roles require human approval evidence.", {"operation": "register_registry_agent", "privileged_role": True, "human_approval_required": False}, {"decision": "require_review", "reason": "registry_agent_human_approval_required", "required_action": "record_human_registry_agent_approval"}),
		CapabilityRule("bytewax_regy_stream_required", "REGY lifecycle batches must be routed through Bytewax.", {"operation": "validate_regy_lifecycle_batch", "event_stream_ne": "bytewax"}, {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_regy_lifecycle_batch_to_bytewax"}),
	]


def ui_manifest() -> dict[str, Any]:
	"""Return REGY UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/regy/dashboard", "RegistryDashboard", "registry:view_statistics", "Overview"),
		CapabilityUIRoute("services", "/regy/services", "ServiceCatalog", "registry:list_services", "Catalog"),
		CapabilityUIRoute("register", "/regy/register", "ServiceRegistrationConsole", "registry:register_service", "Catalog"),
		CapabilityUIRoute("instances", "/regy/instances", "ServiceInstanceManager", "registry:update_service", "Catalog"),
		CapabilityUIRoute("discovery", "/regy/discovery", "DiscoveryConsole", "registry:discover_services", "Discovery"),
		CapabilityUIRoute("health", "/regy/health", "ServiceHealthDashboard", "registry:view_health", "Reliability"),
		CapabilityUIRoute("versions", "/regy/versions", "ServiceVersionManager", "registry:update_service", "Governance"),
		CapabilityUIRoute("contracts", "/regy/contracts", "ContractReviewQueue", "registry:update_service", "Governance"),
		CapabilityUIRoute("gateway_sync", "/regy/gateway-sync", "GatewaySyncPanel", "registry:update_service", "Integration"),
		CapabilityUIRoute("retirements", "/regy/retirements", "RetirementReviewPanel", "registry:deregister_service", "Governance"),
		CapabilityUIRoute("audit", "/regy/audit", "RegistryAuditTimeline", "registry:view_events", "Governance"),
		CapabilityUIRoute("agents", "/regy/agents", "RegistryAgentRoster", "registry:update_service", "Governance"),
		CapabilityUIRoute("lifecycle", "/regy/lifecycle", "RegistryLifecycleBatchMonitor", "registry:view_events", "Operations"),
		CapabilityUIRoute("settings", "/regy/settings", "RegistrySettings", "registry:update_service", "Administration"),
	]
	return {"shell": "apg_python", "view_module": "view_models.py", "api_prefix": "/api/regy/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class registry-agent composition metadata."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_REGY_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_REGY_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_REGY_AGENT_ROLES),
		"requires": ["scope", "owner", "purpose", "contribution_disclosure"],
		"approval": "human approval is required before privileged registry agents mutate registry state",
		"external_runtimes_are_adapters": True,
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return REGY lifecycle streaming metadata."""
	return {
		"engine": "bytewax",
		"required_processor": "bytewax",
		"lifecycle_stream": "regy.lifecycle",
		"watermark": "event_time",
		"operations": [
			"service_batch",
			"instance_batch",
			"version_batch",
			"discovery_batch",
			"gateway_publication_batch",
			"review_batch",
			"registry_agent_batch",
		],
		"topics": [
			"regy.services",
			"regy.instances",
			"regy.versions",
			"regy.discovery",
			"regy.gateway_publications",
			"regy.reviews",
			"regy.agents",
		],
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable REGY capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "regy",
		"display_name": "API/Service Registry",
		"provides": ["service_registry", "service_discovery", "registry_agent_composition"],
		"requires": ["apig", "auth", "conf"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"ui": ui_manifest(),
		"theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components},
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default REGY rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_copy(value: dict[str, Any]) -> dict[str, Any]:
	copied: dict[str, Any] = {}
	for key, item in value.items():
		if isinstance(item, dict):
			copied[key] = _deep_copy(item)
		elif isinstance(item, list):
			copied[key] = list(item)
		else:
			copied[key] = item
	return copied


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
