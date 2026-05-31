#!/usr/bin/env python3
"""
APG Key Management Capability
AI-powered quantum-safe enterprise key management platform

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from typing import Dict, List, Any
from datetime import datetime

from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)

# APG Capability Metadata for Composition Engine Registration
CAPABILITY_METADATA = {
	"name": "keym",
	"display_name": "Key Management",
	"description": "AI-powered quantum-safe enterprise key management platform",
	"version": "1.0.0",
	"category": "security",
	"tags": ["cryptography", "security", "compliance", "automation", "ai", "quantum-safe"],
	
	"composition": {
		"load_order": 5,
		"dependencies": ["auth", "audl", "secu", "mten", "conf"],
		"optional_dependencies": ["aicr", "pred", "anom", "ntfy", "moni", "apig"],
		
		"export_functions": [
			"create_key",
			"rotate_key", 
			"retrieve_key",
			"delete_key",
			"encrypt_data",
			"decrypt_data",
			"sign_data",
			"verify_signature",
			"manage_policy",
			"audit_access",
			"generate_key_pair",
			"import_key",
			"export_key",
			"backup_key",
			"restore_key",
			"list_keys",
			"get_key_metadata",
			"update_key_metadata",
			"get_key_usage_stats",
			"validate_key_policy"
		],
		
		"event_handlers": {
			"key.created": "handle_key_creation",
			"key.rotated": "handle_key_rotation", 
			"key.accessed": "handle_key_access",
			"key.deleted": "handle_key_deletion",
			"policy.violated": "handle_policy_violation",
			"threat.detected": "handle_security_threat",
			"compliance.violated": "handle_compliance_violation",
			"key.expired": "handle_key_expiration",
			"key.compromised": "handle_key_compromise",
			"hsm.connected": "handle_hsm_connection"
		},
		
		"api_endpoints": {
			"base_path": "/api/v1/keym",
			"endpoints": [
				{"path": "/keys", "methods": ["GET", "POST"]},
				{"path": "/keys/{key_id}", "methods": ["GET", "PUT", "DELETE"]},
				{"path": "/keys/{key_id}/rotate", "methods": ["POST"]},
				{"path": "/keys/{key_id}/encrypt", "methods": ["POST"]},
				{"path": "/keys/{key_id}/decrypt", "methods": ["POST"]},
				{"path": "/keys/{key_id}/sign", "methods": ["POST"]},
				{"path": "/keys/{key_id}/verify", "methods": ["POST"]},
				{"path": "/policies", "methods": ["GET", "POST"]},
				{"path": "/policies/{policy_id}", "methods": ["GET", "PUT", "DELETE"]},
				{"path": "/audit", "methods": ["GET"]},
				{"path": "/health", "methods": ["GET"]},
				{"path": "/metrics", "methods": ["GET"]}
			]
		},
		
		"ui_components": {
			"dashboard": "KeyManagementDashboard",
			"key_inventory": "KeyInventoryView",
			"security_analytics": "SecurityAnalyticsView", 
			"compliance_dashboard": "ComplianceDashboard",
			"policy_manager": "PolicyManagerView",
			"audit_logs": "AuditLogsView"
		},
		
		"permissions": [
			"keym.create_key",
			"keym.read_key", 
			"keym.update_key",
			"keym.delete_key",
			"keym.rotate_key",
			"keym.encrypt_decrypt",
			"keym.sign_verify",
			"keym.manage_policies",
			"keym.view_audit_logs",
			"keym.manage_hsm",
			"keym.export_key",
			"keym.approve_export",
			"keym.approve_rotation_exception",
			"keym.respond_compromise",
			"keym.admin"
		],
		
		"config_schema": {
			"encryption": {
				"default_algorithm": "AES-256-GCM",
				"key_derivation": "PBKDF2",
				"hsm_integration": True,
				"quantum_safe": True
			},
			"security": {
				"key_rotation_interval_days": 90,
				"max_failed_attempts": 3,
				"session_timeout_minutes": 30,
				"require_mfa": True
			},
			"compliance": {
				"frameworks": ["FIPS_140_2", "GDPR", "HIPAA", "PCI_DSS"],
				"audit_retention_days": 2555,  # 7 years
				"immutable_audit": True
			},
			"performance": {
				"cache_enabled": True,
				"cache_ttl_seconds": 300,
				"connection_pool_size": 20,
				"max_concurrent_operations": 1000
			}
		}
	},
	
	"apg_integration": {
		"auth_integration": {
			"capability": "auth",
			"features": ["rbac", "mfa", "sso", "session_management"]
		},
		"audit_integration": {
			"capability": "audl", 
			"features": ["comprehensive_logging", "compliance_reporting", "tamper_evident"]
		},
		"security_integration": {
			"capability": "secu",
			"features": ["threat_intelligence", "vulnerability_scanning", "incident_response"]
		},
		"ai_integration": {
			"capabilities": ["aicr", "pred", "anom"],
			"features": ["intelligent_automation", "predictive_analytics", "anomaly_detection"]
		}
	},
	
	"business_metrics": {
		"key_performance_indicators": [
			"key_operations_per_second",
			"key_rotation_compliance_rate", 
			"security_incidents_prevented",
			"compliance_violations",
			"mean_time_to_detect_threats",
			"mean_time_to_remediate",
			"developer_onboarding_time",
			"cost_per_key_operation"
		],
		"success_targets": {
			"availability": 99.99,
			"key_ops_latency_ms": 100,
			"threat_detection_accuracy": 95.0,
			"compliance_automation": 90.0,
			"developer_satisfaction": 4.5
		}
	},
	
	"created_at": datetime.utcnow(),
	"last_updated": datetime.utcnow(),
	"maintainer": "Datacraft Security Team",
	"support_contact": "keym-support@datacraft.co.ke"
}

# APG Capability Health Check
async def health_check() -> Dict[str, Any]:
	"""Health check for APG composition engine"""
	return {
		"status": "healthy",
		"capability": "keym",
		"version": CAPABILITY_METADATA["version"],
		"dependencies_status": "all_healthy",
		"last_check": datetime.utcnow()
	}


def register_capability() -> Dict[str, Any]:
	"""Register key management with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "keym",
		"aliases": ["key_management", "kms", "vault"],
		"display_name": CAPABILITY_METADATA["display_name"],
		"description": CAPABILITY_METADATA["description"],
		"version": CAPABILITY_METADATA["version"],
		"dependencies": CAPABILITY_METADATA["composition"]["dependencies"],
		"optional_dependencies": CAPABILITY_METADATA["composition"]["optional_dependencies"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"key_lifecycle": "Create, rotate, disable, back up, and restore tenant keys",
			"key_policy_governance": "Attach policy and approval controls to key operations",
			"hsm_attestation": "Enforce software or hardware HSM attestation where required",
			"cryptographic_operations": "Coordinate encrypt, decrypt, sign, and verify workflows",
			"export_dual_control": "Govern key export through package-backed dual-control approvals",
			"rotation_exception_review": "Govern overdue rotation exceptions with independent review",
			"compromise_response": "Block compromised keys until rotation evidence is recorded",
			"key_agent_composition": "Register accountable AI agents for key governance workflows",
			"bytewax_lifecycle_streaming": "Require Bytewax for key lifecycle batch mutations",
			"audit_and_compliance": "Expose immutable key access audit and compliance surfaces",
			"capability_rules": "Evaluate deterministic key-governance rules",
			"visual_theming": "Apply vault-console theme tokens and components"
		},
		"endpoints": {
			"keys": "/keym/api/v1/keys",
			"lifecycle": "/keym/api/v1/lifecycle",
			"export_approvals": "/keym/api/v1/export-approvals",
			"rotation_exceptions": "/keym/api/v1/rotation-exceptions",
			"compromise": "/keym/api/v1/compromise",
			"policies": "/keym/api/v1/policies",
			"hsm": "/keym/api/v1/hsm",
			"agents": "/keym/api/v1/agents",
			"audit": "/keym/api/v1/audit",
			"analytics": "/keym/api/v1/analytics"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"permissions": CAPABILITY_METADATA["composition"]["permissions"]
	}

# APG Capability Information
def get_capability_info() -> Dict[str, Any]:
	"""Get capability information for APG marketplace"""
	return {
		"metadata": CAPABILITY_METADATA,
		"contract": get_capability_contract(),
		"features": [
			"AI-powered key lifecycle management",
			"Quantum-safe cryptography support", 
			"Multi-cloud key federation",
			"Hardware security module integration",
			"Behavioral analytics and anomaly detection",
			"Automated compliance management",
			"Real-time threat intelligence",
			"Developer-first SDK experience",
			"Visual key management workflows",
			"Enterprise-grade audit trails"
		],
		"integrations": [
			"AWS KMS", "Azure Key Vault", "GCP KMS",
			"Thales HSM", "SafeNet HSM", "AWS CloudHSM",
			"Active Directory", "LDAP", "SAML SSO",
			"Splunk", "ELK Stack", "Prometheus"
		],
		"compliance_frameworks": [
			"FIPS 140-2", "Common Criteria", "GDPR", 
			"HIPAA", "PCI DSS", "SOX", "ISO 27001"
		]
	}

# APG Capability Initialization
async def initialize_capability(config: Dict[str, Any] | None = None) -> bool:
	"""Initialize key management capability with APG integration"""
	try:
		# Import service layer for initialization
		from .service import KeyManagementService
		
		# Initialize with APG configuration
		service = KeyManagementService()
		await service.initialize(config or {})
		
		return True
	except Exception as e:
		# Log error through APG logging infrastructure
		print(f"Failed to initialize keym capability: {e}")
		return False

# Export capability metadata for APG composition engine
__all__ = [
	"CAPABILITY_METADATA",
	"register_capability",
	"health_check", 
	"get_capability_info",
	"initialize_capability",
	"get_capability_contract",
	"evaluate_capability_rules"
]
