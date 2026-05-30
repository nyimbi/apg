"""Executable capability contract for APG Security Operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_SEOP_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_SEOP_AGENT_ROLES = [
	"detection_reviewer",
	"incident_commander",
	"response_reviewer",
	"playbook_author",
	"posture_reviewer",
	"compliance_reviewer",
]
SEOP_EVENT_STREAM = "apg.seop.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"detection": {
		"alert_source_required": True,
		"anomaly_context_required": True,
		"correlation_enabled": True,
		"confidence_threshold": 0.7,
		"deduplicate_signals": True,
	},
	"incidents": {
		"incident_owner_required": True,
		"severity_required": True,
		"evidence_required": True,
		"critical_escalation_required": True,
		"post_incident_review_required": True,
	},
	"response": {
		"playbook_approval_required": True,
		"containment_review_required": True,
		"isolation_authorization_required": True,
		"closure_evidence_required": True,
		"response_actor_required": True,
	},
	"seop_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_SEOP_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_SEOP_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_severity": "medium",
		"disclose_agent_recommendations": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_response_actions": True,
		"post_incident_review_required": True,
		"compliance_mapping_required": True,
	},
	"observability": {
		"event_stream": SEOP_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_detection_events": True,
		"emit_response_events": True,
	},
	"adapters": {
		"event_stream": "bytewax",
		"siem": "adapter",
		"soar": "adapter",
		"edr": "adapter",
		"ticketing": "adapter",
		"threat_intel": "adapter",
	},
	"ui": {
		"enable_detection_console": True,
		"enable_incident_queue": True,
		"enable_playbook_manager": True,
		"enable_posture_dashboard": True,
		"enable_agent_workbench": True,
		"enable_audit_trail": True,
	},
	"theme": {"default_theme": "seop_security_ops", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"detection",
		"incidents",
		"response",
		"seop_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"detection",
			"incidents",
			"response",
			"seop_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{
		"name": "tenant_context_required",
		"description": "All security operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	{
		"name": "detection_requires_alert_source",
		"description": "Detections require a trusted alert source.",
		"condition": {"operation": "create_detection", "alert_source_present": False},
		"effect": {"decision": "deny", "reason": "alert_source_required", "required_action": "attach_alert_source"},
	},
	{
		"name": "detection_requires_bytewax_stream",
		"description": "Detection lifecycle events must be emitted through Bytewax.",
		"condition": {"operation": "create_detection", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_detection_lifecycle_to_bytewax"},
	},
	{
		"name": "incident_requires_owner",
		"description": "Security incidents require an accountable owner.",
		"condition": {"operation": "open_incident", "incident_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "incident_owner_required", "required_action": "assign_incident_owner"},
	},
	{
		"name": "incident_requires_evidence",
		"description": "Incidents require evidence references before they can enter the queue.",
		"condition": {"operation": "open_incident", "evidence_attached": False},
		"effect": {"decision": "deny", "reason": "incident_evidence_required", "required_action": "attach_incident_evidence"},
	},
	{
		"name": "critical_incident_requires_escalation",
		"description": "Critical incidents require escalation.",
		"condition": {"incident_severity": "critical", "escalation_recorded": False},
		"effect": {"decision": "deny", "reason": "critical_escalation_required", "required_action": "escalate_incident"},
	},
	{
		"name": "response_requires_playbook_approval",
		"description": "Response actions require approved playbooks.",
		"condition": {"operation": "execute_response", "playbook_approved": False},
		"effect": {"decision": "deny", "reason": "playbook_approval_required", "required_action": "approve_response_playbook"},
	},
	{
		"name": "response_requires_actor",
		"description": "Response actions require an accountable human or delegated agent actor.",
		"condition": {"operation": "execute_response", "response_actor_present": False},
		"effect": {"decision": "deny", "reason": "response_actor_required", "required_action": "attach_response_actor"},
	},
	{
		"name": "response_requires_containment_review",
		"description": "Containment actions require review evidence.",
		"condition": {"operation": "execute_response", "containment_review_recorded": False},
		"effect": {"decision": "deny", "reason": "containment_review_required", "required_action": "record_containment_review"},
	},
	{
		"name": "high_confidence_anomaly_requires_review",
		"description": "High-confidence anomalies require review.",
		"condition": {"anomaly_confidence_gt": 0.9, "triage_review_recorded": False},
		"effect": {"decision": "require_review", "reason": "anomaly_triage_review_required", "required_action": "review_anomaly"},
	},
	{
		"name": "closure_requires_post_incident_review",
		"description": "Closed incidents require a post-incident review reference.",
		"condition": {"operation": "close_incident", "post_incident_review_present": False},
		"effect": {"decision": "deny", "reason": "post_incident_review_required", "required_action": "attach_post_incident_review"},
	},
	{
		"name": "closure_requires_compliance_mapping",
		"description": "Closed incidents require compliance impact mapping.",
		"condition": {"operation": "close_incident", "compliance_mapping_present": False},
		"effect": {"decision": "deny", "reason": "compliance_mapping_required", "required_action": "attach_compliance_mapping"},
	},
	{
		"name": "seop_agent_runtime_supported",
		"description": "Security operations agents must use an approved runtime.",
		"condition": {"operation": "register_seop_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "seop_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "seop_agent_role_supported",
		"description": "Security operations agents must use an approved role.",
		"condition": {"operation": "register_seop_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "seop_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "critical_agent_action_requires_human_approval",
		"description": "Critical response actions proposed by agents require human approval.",
		"condition": {"operation": "agent_response_action", "incident_severity": "critical", "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/seop/dashboard", "component": "SEOPDashboard", "permission": "seop:view", "nav_group": "Overview"},
	{"name": "detections", "path": "/seop/detections", "component": "DetectionConsole", "permission": "seop:triage", "nav_group": "Detection"},
	{"name": "incidents", "path": "/seop/incidents", "component": "IncidentQueue", "permission": "seop:respond", "nav_group": "Incidents"},
	{"name": "triage", "path": "/seop/triage", "component": "ThreatTriage", "permission": "seop:triage", "nav_group": "Detection"},
	{"name": "playbooks", "path": "/seop/playbooks", "component": "PlaybookManager", "permission": "seop:manage_playbooks", "nav_group": "Response"},
	{"name": "responses", "path": "/seop/responses", "component": "ResponseActions", "permission": "seop:respond", "nav_group": "Response"},
	{"name": "posture", "path": "/seop/posture", "component": "SecurityPosture", "permission": "seop:view", "nav_group": "Operations"},
	{"name": "agents", "path": "/seop/agents", "component": "SEOPAgentWorkbench", "permission": "seop:admin", "nav_group": "Automation"},
	{"name": "audit", "path": "/seop/audit", "component": "SEOPAuditTrail", "permission": "seop:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/seop/settings", "component": "SEOPSettings", "permission": "seop:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "seop_security_ops",
	"tokens": {
		"color.primary": "#2A4365",
		"color.accent": "#C53030",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"detection_card": {"icon": "shield-alert", "status_indicator": "severity-pill", "risk_style": "threat-band"},
		"incident_queue": {"visual": "priority-list", "highlight": "sla-chip"},
		"playbook_manager": {"visual": "action-matrix", "status_style": "approval-chip"},
		"posture_panel": {"visual": "control-grid", "status_style": "coverage-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
		"audit_timeline": {"visual": "event-stream", "status_style": "severity-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": SEOP_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"detection_created",
			"incident_opened",
			"playbook_approved",
			"response_executed",
			"incident_closed",
			"seop_agent_registered",
		],
		"states": ["new", "review_required", "open", "escalated", "responding", "contained", "closed"],
		"guardrails": [
			"detection_requires_bytewax_stream",
			"critical_agent_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return SEOP_EVENT_STREAM


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "seop",
		"display_name": "Security Operations",
		"version": "1.0.0",
		"provides": [
			"detection_pipeline",
			"incident_response",
			"threat_triage",
			"response_playbooks",
			"security_posture",
			"seop_agents",
		],
		"requires": ["secu", "anom", "moni", "logt", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/seop/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
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
		if key.endswith("_lte"):
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
