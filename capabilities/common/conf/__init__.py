"""
APG Configuration Management Capability - Revolutionary Infrastructure Automation

AI-native configuration management system providing 10x improvement over industry
leaders through predictive intelligence, universal abstraction, and autonomous operations.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from .service import RevolutionaryConfigurationManager, create_configuration_manager, get_config_manager
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

# Export main components
__all__ = [
	# Service Layer
	"RevolutionaryConfigurationManager",
	"create_configuration_manager",
	"get_config_manager",
	
	# Core Models
	"CMResource", "CMTemplate", "CMPolicy", "CMEnvironment", "CMDeployment",
	"ConfigurationDSL", "ValidationResult", "ExecutionResult", "AIInsight", "CMMetrics",
	
	# Enums
	"ResourceState", "DeploymentStatus", "PolicyAction", "ResourceType",
	"PolicyType", "CloudProvider", "ComplianceFramework",
	
	# Validators
	"validate_resource_name", "validate_tenant_id", "validate_configuration_spec"
]
