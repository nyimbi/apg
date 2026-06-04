"""Executable capability contract for APG Payment Switch."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_switch"
CAPABILITY_NAME = "Payment Switch"
CAPABILITY_VERSION = "1.1.0"
SWITCH_EVENT_STREAM = "apg.fintech.switch.lifecycle"

# ISO 8583 message types
SUPPORTED_MESSAGE_TYPES = [
	"0100", "0110", "0200", "0210", "0220", "0230",
	"0400", "0410", "0420", "0430", "0800", "0810",
	"0820", "0830",
]
SUPPORTED_TRANSACTION_TYPES = [
	"purchase", "cash_advance", "balance_inquiry", "mini_statement",
	"funds_transfer", "reversal", "refund", "pre_authorization",
	"pre_auth_completion", "void", "pin_change", "key_exchange",
]
SUPPORTED_NETWORKS = [
	"visa", "mastercard", "amex", "mpesa", "airtel_money", "mtn_momo",
	"equitel", "tkash", "pesalink", "rtgs", "eft", "interbank",
	"swift", "sepa", "iso20022",
]
SUPPORTED_CHANNEL_TYPES = [
	"atm", "pos", "ecommerce", "mobile", "ussd", "agent",
	"branch", "internet_banking", "qr", "nfc", "api",
]
SUPPORTED_ROUTING_ALGORITHMS = [
	"least_cost", "priority", "round_robin", "failover",
	"load_balanced", "rule_based", "intelligent",
]
SUPPORTED_SECURITY_ALGORITHMS = [
	"3des", "aes128", "aes256", "rsa2048", "rsa4096",
]
SUPPORTED_PIN_BLOCK_FORMATS = ["iso0", "iso1", "iso3", "iso4", "ansi"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"switch_ops_reviewer", "routing_reviewer", "security_reviewer",
	"settlement_reviewer", "network_reviewer", "incident_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"routing": {
		"supported_networks": SUPPORTED_NETWORKS,
		"supported_algorithms": SUPPORTED_ROUTING_ALGORITHMS,
		"fallback_network_required": True,
		"routing_table_version_required": True,
		"max_hops": 3,
		"timeout_ms": 30000,
	},
	"transactions": {
		"supported_message_types": SUPPORTED_MESSAGE_TYPES,
		"supported_types": SUPPORTED_TRANSACTION_TYPES,
		"idempotency_required": True,
		"stan_unique_required": True,
		"rrn_unique_required": True,
		"high_value_threshold_kes": 1000000,
		"velocity_window_seconds": 60,
		"velocity_count_threshold": 10,
	},
	"channels": {
		"supported_types": SUPPORTED_CHANNEL_TYPES,
		"channel_key_required": True,
		"channel_policy_required": True,
		"mpesa_stk_push_supported": True,
		"mpesa_b2c_supported": True,
		"mpesa_b2b_supported": True,
		"ussd_session_timeout_seconds": 180,
	},
	"security": {
		"supported_algorithms": SUPPORTED_SECURITY_ALGORITHMS,
		"supported_pin_block_formats": SUPPORTED_PIN_BLOCK_FORMATS,
		"zone_key_required": True,
		"pin_block_format_required": True,
		"mac_verification_required": True,
		"key_expiry_check_required": True,
		"key_rotation_days": 30,
		"hsm_required": True,
	},
	"settlement": {
		"net_settlement_supported": True,
		"gross_settlement_supported": True,
		"settlement_file_required": True,
		"reconciliation_required": True,
		"variance_review_threshold_kes": 1000,
	},
	"mobile_money": {
		"mpesa_daraja_api_version": "v2",
		"mpesa_shortcode_required": True,
		"mpesa_passkey_required": True,
		"mpesa_consumer_key_required": True,
		"airtel_money_supported": True,
		"mtn_momo_supported": True,
		"equitel_supported": True,
		"tkash_supported": True,
		"pesalink_supported": True,
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
		"audit_switch_events": True,
		"segregation_of_duties": True,
		"key_ceremony_audit_required": True,
	},
	"observability": {
		"event_stream": SWITCH_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_routing_events": True,
		"emit_transaction_events": True,
		"emit_security_events": True,
		"emit_settlement_events": True,
		"emit_channel_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"keys": "keym",
		"encryption": "encr",
		"hsm": "hsms",
		"payments": "fintech_payments",
		"gateway": "fintech_gateway",
		"aml": "fintech_aml",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_routing": True,
		"enable_transactions": True,
		"enable_channels": True,
		"enable_security": True,
		"enable_settlement": True,
		"enable_mobile_money": True,
		"enable_agents": True,
	},
	"theme": {
		"default_theme": "fintech_switch_control",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"iso8583_message_switching",
	"payment_routing_engine",
	"channel_key_management",
	"pin_block_translation",
	"mac_generation_verification",
	"mobile_money_switching",
	"ussd_session_management",
	"switch_settlement_reconciliation",
	"network_interface_management",
	"switch_agent_workflow",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"keym",
	"encr",
	"keym",
	"fintech_payments",
	"fintech_gateway",
	"fintech_aml",
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-switch/dashboard", "component": "SwitchDashboard", "permission": "fintech_switch:view", "nav_group": "Overview"},
	{"name": "routing", "path": "/fintech-switch/routing", "component": "RoutingTableConsole", "permission": "fintech_switch:manage_routing", "nav_group": "Routing"},
	{"name": "transactions", "path": "/fintech-switch/transactions", "component": "TransactionMonitor", "permission": "fintech_switch:monitor", "nav_group": "Transactions"},
	{"name": "channels", "path": "/fintech-switch/channels", "component": "ChannelWorkbench", "permission": "fintech_switch:manage_channels", "nav_group": "Channels"},
	{"name": "security", "path": "/fintech-switch/security", "component": "SwitchSecurityConsole", "permission": "fintech_switch:manage_keys", "nav_group": "Security"},
	{"name": "mobile_money", "path": "/fintech-switch/mobile-money", "component": "MobileMoneyConsole", "permission": "fintech_switch:mobile_money", "nav_group": "Mobile Money"},
	{"name": "settlement", "path": "/fintech-switch/settlement", "component": "SwitchSettlementConsole", "permission": "fintech_switch:settle", "nav_group": "Settlement"},
	{"name": "networks", "path": "/fintech-switch/networks", "component": "NetworkInterfaceWorkbench", "permission": "fintech_switch:manage_networks", "nav_group": "Networks"},
	{"name": "incidents", "path": "/fintech-switch/incidents", "component": "SwitchIncidentWorkbench", "permission": "fintech_switch:incidents", "nav_group": "Operations"},
	{"name": "agents", "path": "/fintech-switch/agents", "component": "SwitchAgentWorkbench", "permission": "fintech_switch:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-switch/settings", "component": "SwitchSettings", "permission": "fintech_switch:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_switch_control",
	"tokens": {
		"color.primary": "#1E293B",
		"color.accent": "#0F766E",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F1F5F9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "6px",
		"density": "compact",
	},
	"components": {
		"routing": {"icon": "git-fork", "status_indicator": "route-chip"},
		"transactions": {"icon": "activity", "visual": "transaction-timeline", "status_style": "txn-chip"},
		"channels": {"icon": "plug", "status_indicator": "channel-pill"},
		"security": {"icon": "shield-lock", "status_indicator": "key-status-chip"},
		"mobile_money": {"icon": "smartphone", "status_indicator": "mm-provider-chip"},
		"settlement": {"visual": "settlement-grid", "status_style": "variance-chip"},
		"networks": {"icon": "network", "status_indicator": "network-health-chip"},
		"incidents": {"icon": "alert-triangle", "status_style": "incident-severity-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": SWITCH_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"switch_transaction_received",
		"switch_transaction_routed",
		"switch_transaction_authorized",
		"switch_transaction_reversed",
		"switch_channel_registered",
		"switch_channel_key_exchanged",
		"switch_pin_block_translated",
		"switch_mac_verified",
		"switch_mobile_money_initiated",
		"switch_ussd_session_started",
		"switch_ussd_session_completed",
		"switch_settlement_file_generated",
		"switch_reconciliation_completed",
		"switch_network_interface_registered",
		"switch_key_rotation_completed",
		"switch_incident_opened",
		"switch_agent_registered",
	],
	"guardrails": [
		"switch_batch_requires_bytewax",
		"switch_event_requires_bytewax",
		"switch_key_operation_requires_hsm",
		"privileged_switch_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	# Governance
	{"name": "tenant_context_required", "description": "Switch operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "switch_write_requires_policy", "description": "Switch configuration writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "switch_policy_required", "required_action": "attach_switch_policy"}},
	{"name": "cross_tenant_access_denied", "description": "Switch resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Switch privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Transaction processing
	{"name": "transaction_message_type_supported", "description": "ISO 8583 message type must be in supported set.", "condition": {"operation": "process_transaction", "message_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_message_type", "required_action": "use_supported_message_type"}},
	{"name": "transaction_type_supported", "description": "Transaction type must be supported.", "condition": {"operation": "process_transaction", "transaction_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_transaction_type", "required_action": "use_supported_transaction_type"}},
	{"name": "transaction_stan_required", "description": "All transactions require a unique STAN.", "condition": {"operation": "process_transaction", "stan_present": False}, "effect": {"decision": "deny", "reason": "stan_required", "required_action": "generate_stan"}},
	{"name": "transaction_stan_unique", "description": "STAN must be unique within the processing window.", "condition": {"operation": "process_transaction", "stan_duplicate": True}, "effect": {"decision": "deny", "reason": "duplicate_stan", "required_action": "generate_unique_stan"}},
	{"name": "transaction_rrn_required", "description": "All transactions require a Retrieval Reference Number.", "condition": {"operation": "process_transaction", "rrn_present": False}, "effect": {"decision": "deny", "reason": "rrn_required", "required_action": "generate_rrn"}},
	{"name": "transaction_amount_positive", "description": "Transaction amount must be positive.", "condition": {"operation": "process_transaction", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "non_positive_amount", "required_action": "set_positive_amount"}},
	{"name": "high_value_transaction_requires_approval", "description": "High-value switch transactions require approval evidence.", "condition": {"operation": "process_transaction", "high_value": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "high_value_approval_required", "required_action": "record_approval"}},
	{"name": "velocity_breach_requires_review", "description": "Velocity threshold breaches require AML review.", "condition": {"operation": "process_transaction", "velocity_breach": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "velocity_review_required", "required_action": "record_aml_review"}},
	{"name": "reversal_requires_original_transaction", "description": "Reversals require the original transaction reference.", "condition": {"operation": "reverse_transaction", "original_transaction_present": False}, "effect": {"decision": "deny", "reason": "original_transaction_required", "required_action": "attach_original_transaction"}},

	# Routing
	{"name": "routing_network_supported", "description": "Routing target network must be supported.", "condition": {"operation": "route_transaction", "network_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_network", "required_action": "select_supported_network"}},
	{"name": "routing_table_version_required", "description": "Routing decisions require a versioned routing table.", "condition": {"operation": "route_transaction", "routing_table_version_present": False}, "effect": {"decision": "deny", "reason": "routing_table_version_required", "required_action": "attach_routing_table_version"}},
	{"name": "routing_fallback_required", "description": "Routing configurations require a fallback network.", "condition": {"operation": "configure_routing", "fallback_network_present": False}, "effect": {"decision": "deny", "reason": "fallback_network_required", "required_action": "configure_fallback_network"}},

	# Channel management
	{"name": "channel_type_supported", "description": "Channel type must be supported.", "condition": {"operation": "register_channel", "channel_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_channel_type", "required_action": "select_supported_channel_type"}},
	{"name": "channel_key_required", "description": "Channels require an associated encryption key.", "condition": {"operation": "register_channel", "channel_key_present": False}, "effect": {"decision": "deny", "reason": "channel_key_required", "required_action": "assign_channel_key"}},
	{"name": "channel_policy_required", "description": "Channels require an associated policy.", "condition": {"operation": "register_channel", "channel_policy_present": False}, "effect": {"decision": "deny", "reason": "channel_policy_required", "required_action": "attach_channel_policy"}},

	# Security / HSM / keys
	{"name": "key_operation_requires_hsm", "description": "Key generation and loading operations require HSM.", "condition": {"operation": "key_operation", "hsm_present": False}, "effect": {"decision": "deny", "reason": "hsm_required", "required_action": "route_to_hsm"}},
	{"name": "zone_key_required_for_pin", "description": "PIN block translation requires zone key.", "condition": {"operation": "translate_pin_block", "zone_key_present": False}, "effect": {"decision": "deny", "reason": "zone_key_required", "required_action": "load_zone_key"}},
	{"name": "pin_block_format_supported", "description": "PIN block format must be in supported set.", "condition": {"operation": "translate_pin_block", "pin_block_format_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_pin_block_format", "required_action": "use_supported_pin_block_format"}},
	{"name": "mac_verification_required", "description": "Switch messages require MAC verification.", "condition": {"operation": "process_transaction", "mac_verified": False}, "effect": {"decision": "deny", "reason": "mac_verification_required", "required_action": "verify_mac"}},
	{"name": "expired_key_denies_processing", "description": "Transactions using expired keys are denied.", "condition": {"operation": "process_transaction", "key_expired": True}, "effect": {"decision": "deny", "reason": "key_expired", "required_action": "rotate_key"}},
	{"name": "key_ceremony_requires_audit", "description": "Key ceremonies require dual-control audit evidence.", "condition": {"operation": "key_ceremony", "audit_evidence_present": False}, "effect": {"decision": "deny", "reason": "key_ceremony_audit_required", "required_action": "record_key_ceremony_audit"}},

	# Mobile money — M-Pesa and others
	{"name": "mpesa_shortcode_required", "description": "M-Pesa transactions require a registered shortcode.", "condition": {"operation": "mpesa_transaction", "mpesa_shortcode_present": False}, "effect": {"decision": "deny", "reason": "mpesa_shortcode_required", "required_action": "register_mpesa_shortcode"}},
	{"name": "mpesa_passkey_required", "description": "M-Pesa STK push requires a passkey.", "condition": {"operation": "mpesa_stk_push", "mpesa_passkey_present": False}, "effect": {"decision": "deny", "reason": "mpesa_passkey_required", "required_action": "configure_mpesa_passkey"}},
	{"name": "mpesa_consumer_key_required", "description": "M-Pesa Daraja API requires consumer key.", "condition": {"operation": "mpesa_api_call", "mpesa_consumer_key_present": False}, "effect": {"decision": "deny", "reason": "mpesa_consumer_key_required", "required_action": "configure_mpesa_consumer_key"}},
	{"name": "mpesa_b2c_requires_approval", "description": "M-Pesa B2C disbursements above threshold require approval.", "condition": {"operation": "mpesa_b2c", "high_value": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "mpesa_b2c_approval_required", "required_action": "record_mpesa_b2c_approval"}},
	{"name": "mobile_money_kyc_required", "description": "Mobile money transactions require linked KYC.", "condition": {"operation": "mobile_money_transaction", "kyc_present": False}, "effect": {"decision": "deny", "reason": "mobile_money_kyc_required", "required_action": "attach_kyc_profile"}},
	{"name": "ussd_session_timeout_enforced", "description": "USSD sessions older than timeout are denied.", "condition": {"operation": "ussd_session", "session_expired": True}, "effect": {"decision": "deny", "reason": "ussd_session_expired", "required_action": "restart_ussd_session"}},

	# Settlement
	{"name": "settlement_file_required", "description": "Settlement reconciliation requires settlement file.", "condition": {"operation": "reconcile_settlement", "settlement_file_present": False}, "effect": {"decision": "deny", "reason": "settlement_file_required", "required_action": "attach_settlement_file"}},
	{"name": "settlement_variance_requires_review", "description": "Settlement variance above threshold requires review.", "condition": {"operation": "reconcile_settlement", "variance_exceeds_threshold": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "settlement_variance_review_required", "required_action": "record_variance_review"}},

	# Streaming
	{"name": "switch_batch_requires_bytewax", "description": "Switch transaction batches require Bytewax.", "condition": {"operation": "switch_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_required", "required_action": "route_to_bytewax"}},
	{"name": "switch_event_requires_bytewax", "description": "Switch events require Bytewax.", "condition": {"operation": "switch_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_required", "required_action": "route_to_bytewax"}},

	# Agents
	{"name": "switch_agent_runtime_supported", "description": "Switch agents must use a supported runtime.", "condition": {"operation": "register_switch_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "switch_agent_role_supported", "description": "Switch agents must use a supported role.", "condition": {"operation": "register_switch_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_switch_agent_action_requires_human_approval", "description": "Privileged switch-agent actions require human approval.", "condition": {"operation": "switch_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
	"""Return the executable APG Payment Switch capability contract."""
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
			"api_prefix": "/fintech-switch/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic payment switch guardrails."""
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
