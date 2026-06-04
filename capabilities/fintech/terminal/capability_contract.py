"""Executable capability contract for APG Terminal Management System."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_terminal"
CAPABILITY_NAME = "Terminal Management System"
CAPABILITY_VERSION = "1.1.0"
TERMINAL_EVENT_STREAM = "apg.fintech.terminal.lifecycle"

SUPPORTED_TERMINAL_TYPES = [
	"pos", "atm", "mpos", "android_pos", "web_pos", "kiosk",
	"unattended", "soft_pos", "tap_on_phone", "agent_terminal",
]
SUPPORTED_NETWORK_TYPES = [
	"ethernet", "wifi", "gprs", "lte", "satellite", "dialup", "bluetooth",
]
SUPPORTED_COMMUNICATION_PROTOCOLS = [
	"iso8583", "iso20022", "rest", "soap", "mqtt", "tcp_ip",
]
SUPPORTED_CERTIFICATE_TYPES = [
	"ssl_tls", "client_cert", "code_signing", "device_identity",
]
SUPPORTED_KEY_TYPES = [
	"master_key", "session_key", "pin_encryption_key",
	"mac_key", "data_encryption_key", "key_encryption_key",
]
SUPPORTED_PARAMETER_TYPES = [
	"aid_list", "public_key", "terminal_config", "floor_limit",
	"bin_range", "acquirer_config", "network_config",
]
SUPPORTED_DEPLOYMENT_ENVIRONMENTS = [
	"production", "staging", "uat", "development",
]
SUPPORTED_STATUSES = [
	"active", "inactive", "suspended", "decommissioned",
	"key_injection_pending", "configuration_pending", "maintenance",
]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"terminal_ops_reviewer", "key_injection_reviewer", "deployment_reviewer",
	"maintenance_reviewer", "compliance_reviewer", "security_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"terminals": {
		"supported_types": SUPPORTED_TERMINAL_TYPES,
		"supported_statuses": SUPPORTED_STATUSES,
		"supported_environments": SUPPORTED_DEPLOYMENT_ENVIRONMENTS,
		"serial_number_required": True,
		"merchant_required": True,
		"location_required": True,
		"model_required": True,
	},
	"keys": {
		"supported_key_types": SUPPORTED_KEY_TYPES,
		"hsm_required": True,
		"dual_control_required": True,
		"key_injection_audit_required": True,
		"key_rotation_days": 90,
		"key_expiry_warning_days": 14,
	},
	"parameters": {
		"supported_parameter_types": SUPPORTED_PARAMETER_TYPES,
		"version_tracking_required": True,
		"deployment_approval_required": True,
		"rollback_supported": True,
	},
	"communication": {
		"supported_protocols": SUPPORTED_COMMUNICATION_PROTOCOLS,
		"supported_network_types": SUPPORTED_NETWORK_TYPES,
		"heartbeat_interval_seconds": 60,
		"session_timeout_seconds": 300,
		"tls_required": True,
		"certificate_pinning_required": True,
	},
	"certificates": {
		"supported_types": SUPPORTED_CERTIFICATE_TYPES,
		"expiry_warning_days": 30,
		"auto_renewal_supported": True,
		"ca_validation_required": True,
	},
	"compliance": {
		"pci_dss_required": True,
		"tamper_detection_required": True,
		"software_integrity_check_required": True,
		"mpesa_certified_required_for_mobile_money": True,
		"cbk_type_approval_required": True,
	},
	"mobile_money": {
		"mpesa_sdk_supported": True,
		"airtel_money_sdk_supported": True,
		"equitel_sdk_supported": True,
		"tkash_sdk_supported": True,
		"ussd_push_supported": True,
		"mpesa_shortcode_registration_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_terminal_events": True,
		"segregation_of_duties": True,
		"pci_p2pe_compliance_required": True,
	},
	"observability": {
		"event_stream": TERMINAL_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_lifecycle_events": True,
		"emit_key_events": True,
		"emit_parameter_events": True,
		"emit_compliance_events": True,
		"emit_health_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"keys": "keym",
		"encryption": "encr",
		"hsm": "hsms",
		"switch": "fintech_switch",
		"payments": "fintech_payments",
		"merchants": "fintech_merchants",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_terminals": True,
		"enable_key_management": True,
		"enable_parameters": True,
		"enable_certificates": True,
		"enable_compliance": True,
		"enable_mobile_money": True,
		"enable_agents": True,
	},
	"theme": {
		"default_theme": "fintech_terminal_control",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"terminal_lifecycle_management",
	"terminal_key_injection_workflow",
	"terminal_parameter_deployment",
	"terminal_certificate_management",
	"terminal_health_monitoring",
	"pci_dss_compliance_tracking",
	"mobile_money_sdk_deployment",
	"terminal_agent_workflow",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"keym",
	"encr",
	"keym",
	"fintech_switch",
	"fintech_payments",
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-terminal/dashboard", "component": "TerminalDashboard", "permission": "fintech_terminal:view", "nav_group": "Overview"},
	{"name": "terminals", "path": "/fintech-terminal/terminals", "component": "TerminalWorkbench", "permission": "fintech_terminal:manage", "nav_group": "Terminals"},
	{"name": "key_management", "path": "/fintech-terminal/keys", "component": "TerminalKeyConsole", "permission": "fintech_terminal:manage_keys", "nav_group": "Security"},
	{"name": "parameters", "path": "/fintech-terminal/parameters", "component": "ParameterDeploymentConsole", "permission": "fintech_terminal:deploy_parameters", "nav_group": "Configuration"},
	{"name": "certificates", "path": "/fintech-terminal/certificates", "component": "CertificateWorkbench", "permission": "fintech_terminal:manage_certificates", "nav_group": "Security"},
	{"name": "compliance", "path": "/fintech-terminal/compliance", "component": "TerminalComplianceConsole", "permission": "fintech_terminal:compliance", "nav_group": "Compliance"},
	{"name": "mobile_money", "path": "/fintech-terminal/mobile-money", "component": "MobileMoneyTerminalConsole", "permission": "fintech_terminal:mobile_money", "nav_group": "Mobile Money"},
	{"name": "health", "path": "/fintech-terminal/health", "component": "TerminalHealthMonitor", "permission": "fintech_terminal:monitor", "nav_group": "Operations"},
	{"name": "deployments", "path": "/fintech-terminal/deployments", "component": "TerminalDeploymentWorkbench", "permission": "fintech_terminal:deploy", "nav_group": "Deployments"},
	{"name": "agents", "path": "/fintech-terminal/agents", "component": "TerminalAgentWorkbench", "permission": "fintech_terminal:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-terminal/settings", "component": "TerminalSettings", "permission": "fintech_terminal:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_terminal_control",
	"tokens": {
		"color.primary": "#1E3A5F",
		"color.accent": "#0F766E",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F0F4F8",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"terminals": {"icon": "monitor", "status_indicator": "terminal-status-chip"},
		"key_management": {"icon": "key-round", "status_indicator": "key-expiry-chip"},
		"parameters": {"icon": "sliders", "status_indicator": "param-version-chip"},
		"certificates": {"icon": "shield-check", "status_indicator": "cert-expiry-chip"},
		"compliance": {"icon": "clipboard-check", "status_indicator": "pci-status-chip"},
		"mobile_money": {"icon": "smartphone", "status_indicator": "sdk-version-chip"},
		"health": {"visual": "health-grid", "status_style": "health-band"},
		"deployments": {"visual": "deploy-timeline", "status_style": "deploy-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TERMINAL_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"terminal_registered",
		"terminal_deployed",
		"terminal_suspended",
		"terminal_decommissioned",
		"terminal_key_injected",
		"terminal_key_rotated",
		"terminal_parameters_deployed",
		"terminal_parameters_rolled_back",
		"terminal_certificate_installed",
		"terminal_certificate_renewed",
		"terminal_tamper_detected",
		"terminal_pci_audit_completed",
		"terminal_mobile_money_sdk_installed",
		"terminal_heartbeat_missed",
		"terminal_health_alert_raised",
		"terminal_agent_registered",
	],
	"guardrails": [
		"terminal_batch_requires_bytewax",
		"terminal_event_requires_bytewax",
		"terminal_key_operation_requires_hsm",
		"privileged_terminal_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	# Governance
	{"name": "tenant_context_required", "description": "Terminal operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "terminal_write_requires_policy", "description": "Terminal writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "terminal_policy_required", "required_action": "attach_terminal_policy"}},
	{"name": "cross_tenant_access_denied", "description": "Terminal resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Terminal lifecycle
	{"name": "terminal_type_supported", "description": "Terminal type must be supported.", "condition": {"operation": "register_terminal", "terminal_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_terminal_type", "required_action": "use_supported_terminal_type"}},
	{"name": "terminal_serial_required", "description": "Terminals require a unique serial number.", "condition": {"operation": "register_terminal", "serial_number_present": False}, "effect": {"decision": "deny", "reason": "serial_number_required", "required_action": "provide_serial_number"}},
	{"name": "terminal_merchant_required", "description": "Terminals must be assigned to a merchant.", "condition": {"operation": "register_terminal", "merchant_present": False}, "effect": {"decision": "deny", "reason": "merchant_required", "required_action": "assign_merchant"}},
	{"name": "terminal_location_required", "description": "Terminals must have a location record.", "condition": {"operation": "register_terminal", "location_present": False}, "effect": {"decision": "deny", "reason": "location_required", "required_action": "record_location"}},
	{"name": "terminal_deploy_requires_key_injection", "description": "Terminals must complete key injection before deployment.", "condition": {"operation": "deploy_terminal", "key_injection_complete": False}, "effect": {"decision": "deny", "reason": "key_injection_required", "required_action": "complete_key_injection"}},
	{"name": "terminal_deploy_requires_parameters", "description": "Terminals must have parameters deployed before activation.", "condition": {"operation": "deploy_terminal", "parameters_deployed": False}, "effect": {"decision": "deny", "reason": "parameter_deployment_required", "required_action": "deploy_parameters"}},
	{"name": "terminal_decommission_requires_approval", "description": "Terminal decommissioning requires reviewer approval.", "condition": {"operation": "decommission_terminal", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "decommission_approval_required", "required_action": "record_decommission_approval"}},
	{"name": "suspended_terminal_denies_transactions", "description": "Suspended terminals cannot process transactions.", "condition": {"operation": "process_transaction", "terminal_status": "suspended"}, "effect": {"decision": "deny", "reason": "terminal_suspended", "required_action": "reinstate_terminal"}},

	# Key management
	{"name": "key_injection_requires_hsm", "description": "Key injection operations require HSM.", "condition": {"operation": "inject_key", "hsm_present": False}, "effect": {"decision": "deny", "reason": "hsm_required", "required_action": "route_to_hsm"}},
	{"name": "key_injection_requires_dual_control", "description": "Key injection requires dual-control evidence.", "condition": {"operation": "inject_key", "dual_control_recorded": False}, "effect": {"decision": "deny", "reason": "dual_control_required", "required_action": "record_dual_control"}},
	{"name": "key_injection_requires_audit", "description": "Key injection must be audited.", "condition": {"operation": "inject_key", "audit_evidence_present": False}, "effect": {"decision": "deny", "reason": "key_injection_audit_required", "required_action": "record_audit_evidence"}},
	{"name": "expired_key_denies_transactions", "description": "Terminals with expired keys cannot process transactions.", "condition": {"operation": "process_transaction", "terminal_key_expired": True}, "effect": {"decision": "deny", "reason": "terminal_key_expired", "required_action": "rotate_terminal_key"}},
	{"name": "key_rotation_requires_approval", "description": "Key rotation requires approval.", "condition": {"operation": "rotate_key", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "key_rotation_approval_required", "required_action": "record_rotation_approval"}},

	# Parameter management
	{"name": "parameter_type_supported", "description": "Parameter type must be supported.", "condition": {"operation": "deploy_parameters", "parameter_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_parameter_type", "required_action": "use_supported_parameter_type"}},
	{"name": "parameter_deployment_requires_approval", "description": "Parameter deployments require approval evidence.", "condition": {"operation": "deploy_parameters", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "parameter_deployment_approval_required", "required_action": "record_deployment_approval"}},
	{"name": "parameter_version_required", "description": "Parameter deployments require a version identifier.", "condition": {"operation": "deploy_parameters", "version_present": False}, "effect": {"decision": "deny", "reason": "parameter_version_required", "required_action": "set_parameter_version"}},
	{"name": "parameter_rollback_requires_reason", "description": "Parameter rollbacks require a recorded reason.", "condition": {"operation": "rollback_parameters", "reason_present": False}, "effect": {"decision": "deny", "reason": "rollback_reason_required", "required_action": "record_rollback_reason"}},

	# Certificate management
	{"name": "certificate_type_supported", "description": "Certificate type must be supported.", "condition": {"operation": "install_certificate", "certificate_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_certificate_type", "required_action": "use_supported_certificate_type"}},
	{"name": "certificate_ca_validation_required", "description": "Certificates must pass CA validation.", "condition": {"operation": "install_certificate", "ca_validated": False}, "effect": {"decision": "deny", "reason": "ca_validation_required", "required_action": "validate_certificate_ca"}},
	{"name": "expired_certificate_denies_connection", "description": "Terminals with expired TLS certificates cannot connect.", "condition": {"operation": "terminal_connect", "certificate_expired": True}, "effect": {"decision": "deny", "reason": "certificate_expired", "required_action": "renew_certificate"}},

	# Compliance — PCI DSS / CBK
	{"name": "pci_dss_required", "description": "Deployed terminals must have PCI DSS compliance evidence.", "condition": {"operation": "deploy_terminal", "pci_dss_compliant": False}, "effect": {"decision": "deny", "reason": "pci_dss_compliance_required", "required_action": "complete_pci_dss_certification"}},
	{"name": "tamper_detection_required", "description": "Terminals must have tamper detection enabled.", "condition": {"operation": "deploy_terminal", "tamper_detection_enabled": False}, "effect": {"decision": "deny", "reason": "tamper_detection_required", "required_action": "enable_tamper_detection"}},
	{"name": "tamper_event_suspends_terminal", "description": "Tamper detection events suspend the terminal immediately.", "condition": {"operation": "process_transaction", "tamper_detected": True}, "effect": {"decision": "deny", "reason": "terminal_tampered", "required_action": "suspend_terminal_and_investigate"}},
	{"name": "cbk_type_approval_required", "description": "Terminals must have CBK type approval for Kenya deployment.", "condition": {"operation": "deploy_terminal", "deployment_country": "KE", "cbk_type_approved": False}, "effect": {"decision": "deny", "reason": "cbk_type_approval_required", "required_action": "obtain_cbk_type_approval"}},
	{"name": "software_integrity_required", "description": "Terminal software integrity check must pass before deployment.", "condition": {"operation": "deploy_terminal", "software_integrity_verified": False}, "effect": {"decision": "deny", "reason": "software_integrity_check_required", "required_action": "verify_software_integrity"}},

	# Mobile money compliance
	{"name": "mpesa_shortcode_registration_required", "description": "M-Pesa mobile money terminal requires registered shortcode.", "condition": {"operation": "enable_mpesa_on_terminal", "mpesa_shortcode_registered": False}, "effect": {"decision": "deny", "reason": "mpesa_shortcode_registration_required", "required_action": "register_mpesa_shortcode"}},
	{"name": "mpesa_certified_terminal_required", "description": "M-Pesa transactions require a certified terminal.", "condition": {"operation": "mpesa_transaction", "mpesa_certified": False}, "effect": {"decision": "deny", "reason": "mpesa_certified_terminal_required", "required_action": "obtain_mpesa_certification"}},
	{"name": "mobile_money_sdk_version_required", "description": "Mobile money SDK must be at the approved version.", "condition": {"operation": "process_mobile_money", "sdk_version_approved": False}, "effect": {"decision": "deny", "reason": "approved_sdk_version_required", "required_action": "update_mobile_money_sdk"}},

	# Streaming
	{"name": "terminal_batch_requires_bytewax", "description": "Terminal batches require Bytewax.", "condition": {"operation": "terminal_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_required", "required_action": "route_to_bytewax"}},
	{"name": "terminal_event_requires_bytewax", "description": "Terminal events require Bytewax.", "condition": {"operation": "terminal_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_required", "required_action": "route_to_bytewax"}},

	# Agents
	{"name": "terminal_agent_runtime_supported", "description": "Terminal agents must use a supported runtime.", "condition": {"operation": "register_terminal_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "terminal_agent_role_supported", "description": "Terminal agents must use a supported role.", "condition": {"operation": "register_terminal_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_terminal_agent_action_requires_human_approval", "description": "Privileged terminal-agent actions require human approval.", "condition": {"operation": "terminal_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": list(DEFAULT_CONFIGURATION),
		"properties": {
			key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"
		} | {"tenant_id": {"type": "string", "minLength": 1}},
	}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if context.get(key[:-4]) is None or context[key[:-4]] > expected:
				return False
			continue
		if key.endswith("_lt"):
			if context.get(key[:-3]) is None or context[key[:-3]] >= expected:
				return False
			continue
		if key.endswith("_gt"):
			if context.get(key[:-3]) is None or context[key[:-3]] <= expected:
				return False
			continue
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the executable APG Terminal Management System capability contract."""
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	if overrides:
		for key, value in overrides.items():
			if isinstance(value, dict) and isinstance(configuration.get(key), dict):
				configuration[key].update(value)
			else:
				configuration[key] = value
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/fintech-terminal/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic terminal management guardrails."""
	contract = get_capability_contract(str(context.get("tenant_id") or "default"))
	matched = [
		rule for rule in contract["rule_engine"]["rules"]
		if _matches_condition(rule["condition"], context)
	]
	decision = "allow"
	for rule in matched:
		rule_decision = rule["effect"]["decision"]
		if rule_decision == "deny":
			decision = "deny"
			break
		if rule_decision == "require_review" and decision == "allow":
			decision = "require_review"
	return {
		"decision": decision,
		"matched_rules": [rule["name"] for rule in matched],
		"actions": [rule["effect"] for rule in matched],
		"effects": [rule["effect"] for rule in matched],
	}
