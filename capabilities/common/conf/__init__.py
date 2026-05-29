"""
APG Configuration Management Capability - Revolutionary Infrastructure Automation

AI-native configuration management system providing 10x improvement over industry
leaders through predictive intelligence, universal abstraction, and autonomous operations.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from .service import ConfService, RevolutionaryConfigurationManager, create_configuration_manager, get_config_manager
from .models import (
	# Core Models
	CMResource, CMTemplate, CMPolicy, CMEnvironment, CMDeployment,
	ConfigurationDSL, ValidationResult, ExecutionResult, AIInsight, CMMetrics,
	
	# Enums
	ResourceState, DeploymentStatus, PolicyAction, ResourceType,
	PolicyType, CloudProvider, ComplianceFramework,
	
	# Validators
	validate_resource_name, validate_tenant_id, validate_configuration_spec
)
from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)


def register_capability() -> dict:
	"""Register the configuration management capability with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "conf",
		"display_name": "Configuration Management",
		"description": "AI-native configuration governance, deployment, and drift remediation",
		"version": "1.0.0",
		"dependencies": [
			"auth_rbac",
			"audit_compliance",
			"ai_orchestration",
			"notification_engine"
		],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"resources": "Manage declarative infrastructure resources",
			"templates": "Compose reusable configuration templates",
			"policies": "Enforce configuration governance and compliance",
			"change_approvals": "Review and approve governed configuration promotions",
			"deployments": "Coordinate controlled rollout workflows",
			"gitops": "Manage GitOps-driven configuration promotion",
			"drift_management": "Detect and review configuration drift remediation",
			"audit_events": "Expose configuration governance evidence",
			"capability_rules": "Evaluate deterministic capability-specific rules",
			"visual_theming": "Apply tenant-aware configuration workspace theming"
		},
		"endpoints": {
			"resources": "/api/v1/config/resources",
			"templates": "/api/v1/config/templates",
			"changes": "/api/v1/config/changes",
			"approvals": "/api/v1/config/approvals",
			"deployments": "/api/v1/config/deployments",
			"drift": "/api/v1/config/drift",
			"drift_remediation": "/api/v1/config/drift/remediation",
			"audit": "/api/v1/config/audit",
			"insights": "/api/v1/config/insights",
			"metrics": "/api/v1/config/metrics"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": [
			"conf:view",
			"conf:create",
			"conf:edit",
			"conf:deploy",
			"conf:approve",
			"conf:remediate",
			"conf:admin"
		]
	}

# Export main components
__all__ = [
	# Service Layer
	"ConfService",
	"RevolutionaryConfigurationManager",
	"create_configuration_manager",
	"get_config_manager",
	"register_capability",
	"get_capability_contract",
	"evaluate_capability_rules",
	
	# Core Models
	"CMResource", "CMTemplate", "CMPolicy", "CMEnvironment", "CMDeployment",
	"ConfigurationDSL", "ValidationResult", "ExecutionResult", "AIInsight", "CMMetrics",
	
	# Enums
	"ResourceState", "DeploymentStatus", "PolicyAction", "ResourceType",
	"PolicyType", "CloudProvider", "ComplianceFramework",
	
	# Validators
	"validate_resource_name", "validate_tenant_id", "validate_configuration_spec"
]
