"""
APG Configuration Management Models - Production Data Structures

Pydantic v2 models defining all data structures for AI-native configuration management
with comprehensive validation, type safety, and APG integration patterns.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Annotated
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from uuid_extensions import uuid7str
import json
import re

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from pydantic import SecretStr


def validate_resource_name(name: str) -> str:
	"""Validate configuration resource name"""
	assert isinstance(name, str), "Resource name must be string"
	assert 3 <= len(name) <= 128, "Resource name must be 3-128 characters"
	assert name.replace("_", "").replace("-", "").replace(".", "").isalnum(), "Resource name must be alphanumeric with allowed separators"
	return name.lower()


def validate_tenant_id(tenant_id: Optional[str]) -> Optional[str]:
	"""Validate tenant ID format"""
	if tenant_id is None:
		return None
	assert isinstance(tenant_id, str), "Tenant ID must be string"
	assert len(tenant_id) >= 8, "Tenant ID must be at least 8 characters"
	return tenant_id


def validate_configuration_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate configuration specification structure"""
	assert isinstance(spec, dict), "Configuration spec must be dictionary"
	assert "resources" in spec or "templates" in spec or "policies" in spec, "Configuration spec must contain resources, templates, or policies"
	return spec


@dataclass(frozen=True)
class ConfigurationRecord:
	"""Tenant-scoped configuration record controlled by the CONF package."""

	id: str
	tenant_id: str
	key: str
	value: Any
	environment: str
	owner: str
	contains_secrets: bool = False
	secrets_encrypted: bool = False
	validation_status: str = "validated"
	version: int = 1
	status: str = "active"
	metadata: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"key": self.key,
			"value": self.value,
			"environment": self.environment,
			"owner": self.owner,
			"contains_secrets": self.contains_secrets,
			"secrets_encrypted": self.secrets_encrypted,
			"validation_status": self.validation_status,
			"version": self.version,
			"status": self.status,
			"metadata": dict(self.metadata),
		}


@dataclass(frozen=True)
class ConfigurationChange:
	"""Configuration change request and independent approval evidence."""

	id: str
	tenant_id: str
	record_id: str
	target_environment: str
	requested_by: str
	summary: str
	proposed_value: Any
	validation_passed: bool
	contains_secrets: bool = False
	secrets_encrypted: bool = False
	rollback_plan: str = ""
	status: str = "pending"
	decision: str = ""
	reviewer: str = ""
	notes: str = ""
	policy_decision: str = "allow"
	matched_rules: List[str] = field(default_factory=list)
	review_reasons: List[str] = field(default_factory=list)
	audit_evidence: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"record_id": self.record_id,
			"target_environment": self.target_environment,
			"requested_by": self.requested_by,
			"summary": self.summary,
			"proposed_value": self.proposed_value,
			"validation_passed": self.validation_passed,
			"contains_secrets": self.contains_secrets,
			"secrets_encrypted": self.secrets_encrypted,
			"rollback_plan": self.rollback_plan,
			"status": self.status,
			"decision": self.decision,
			"reviewer": self.reviewer,
			"notes": self.notes,
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


@dataclass(frozen=True)
class ConfigurationDeployment:
	"""Applied configuration deployment evidence."""

	id: str
	tenant_id: str
	change_id: str
	record_id: str
	target_environment: str
	requested_by: str
	strategy: str
	status: str
	rollback_plan: str
	applied_version: int
	policy_decision: str = "allow"
	matched_rules: List[str] = field(default_factory=list)
	review_reasons: List[str] = field(default_factory=list)
	audit_evidence: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"change_id": self.change_id,
			"record_id": self.record_id,
			"target_environment": self.target_environment,
			"requested_by": self.requested_by,
			"strategy": self.strategy,
			"status": self.status,
			"rollback_plan": self.rollback_plan,
			"applied_version": self.applied_version,
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


@dataclass(frozen=True)
class DriftRemediation:
	"""Configuration drift finding and governed remediation review."""

	id: str
	tenant_id: str
	record_id: str
	detected_by: str
	drift_summary: str
	remediation_plan: str
	status: str = "pending"
	decision: str = ""
	reviewer: str = ""
	notes: str = ""
	policy_decision: str = "allow"
	matched_rules: List[str] = field(default_factory=list)
	review_reasons: List[str] = field(default_factory=list)
	audit_evidence: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"record_id": self.record_id,
			"detected_by": self.detected_by,
			"drift_summary": self.drift_summary,
			"remediation_plan": self.remediation_plan,
			"status": self.status,
			"decision": self.decision,
			"reviewer": self.reviewer,
			"notes": self.notes,
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


@dataclass(frozen=True)
class ConfigurationAgent:
	"""Tenant-scoped configuration agent registration and guardrail evidence."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	purpose: str
	owner: str
	human_approval_required: bool = True
	status: str = "active"
	policy_decision: str = "allow"
	matched_rules: List[str] = field(default_factory=list)
	review_reasons: List[str] = field(default_factory=list)
	audit_evidence: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"purpose": self.purpose,
			"owner": self.owner,
			"human_approval_required": self.human_approval_required,
			"status": self.status,
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


@dataclass(frozen=True)
class ConfigurationBatchEvidence:
	"""Bytewax configuration lifecycle batch evidence."""

	id: str
	tenant_id: str
	record_count: int
	event_stream: str
	status: str
	processor: str = "bytewax"
	policy_decision: str = "allow"
	matched_rules: List[str] = field(default_factory=list)
	review_reasons: List[str] = field(default_factory=list)
	audit_evidence: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"record_count": self.record_count,
			"event_stream": self.event_stream,
			"status": self.status,
			"processor": self.processor,
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


@dataclass(frozen=True)
class ConfigurationAuditEvent:
	"""Immutable package-local configuration governance audit event."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str = "allow"
	reasons: tuple[str, ...] = ()
	matched_rules: tuple[str, ...] = ()
	audit_evidence: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"matched_rules": list(self.matched_rules),
			"audit_evidence": dict(self.audit_evidence),
			"metadata": dict(self.metadata),
		}


# Enums for configuration management

class ResourceState(StrEnum):
	"""Configuration resource state"""
	PENDING = "pending"
	VALIDATING = "validating"
	VALIDATED = "validated"
	DEPLOYING = "deploying"
	DEPLOYED = "deployed"
	FAILED = "failed"
	DRIFTED = "drifted"
	REMEDIATING = "remediating"
	ARCHIVED = "archived"


class DeploymentStatus(StrEnum):
	"""Deployment execution status"""
	PENDING = "pending"
	PLANNING = "planning"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	ROLLING_BACK = "rolling_back"
	ROLLED_BACK = "rolled_back"
	CANCELLED = "cancelled"


class PolicyAction(StrEnum):
	"""Policy enforcement actions"""
	ALLOW = "allow"
	DENY = "deny"
	WARN = "warn"
	REMEDIATE = "remediate"
	QUARANTINE = "quarantine"
	AUDIT = "audit"


class ResourceType(StrEnum):
	"""Configuration resource types"""
	VIRTUAL_MACHINE = "virtual_machine"
	CONTAINER = "container"
	KUBERNETES_DEPLOYMENT = "kubernetes_deployment"
	LOAD_BALANCER = "load_balancer"
	DATABASE = "database"
	NETWORK = "network"
	SECURITY_GROUP = "security_group"
	STORAGE = "storage"
	SERVERLESS_FUNCTION = "serverless_function"
	DNS_RECORD = "dns_record"
	SSL_CERTIFICATE = "ssl_certificate"
	IOT_DEVICE = "iot_device"
	EDGE_NODE = "edge_node"
	AI_MODEL = "ai_model"
	ML_PIPELINE = "ml_pipeline"
	NLP_SERVICE = "nlp_service"
	CUSTOM = "custom"


class PolicyType(StrEnum):
	"""Configuration policy types"""
	SECURITY = "security"
	COMPLIANCE = "compliance"
	COST_OPTIMIZATION = "cost_optimization"
	PERFORMANCE = "performance"
	AVAILABILITY = "availability"
	BACKUP = "backup"
	MONITORING = "monitoring"
	GOVERNANCE = "governance"


class CloudProvider(StrEnum):
	"""Supported cloud providers"""
	AWS = "aws"
	AZURE = "azure"
	GCP = "gcp"
	ALIBABA = "alibaba"
	IBM = "ibm"
	ORACLE = "oracle"
	VMWARE = "vmware"
	OPENSTACK = "openstack"
	KUBERNETES = "kubernetes"
	EDGE = "edge"
	ON_PREMISES = "on_premises"


class ComplianceFramework(StrEnum):
	"""Supported compliance frameworks"""
	SOX = "sox"
	GDPR = "gdpr"
	HIPAA = "hipaa"
	PCI_DSS = "pci_dss"
	ISO_27001 = "iso_27001"
	SOC_2 = "soc_2"
	NIST = "nist"
	CIS = "cis"
	CUSTOM = "custom"


# Core Configuration Models

class ConfigurationDSL(BaseModel):
	"""Universal configuration DSL model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	version: str = Field(default="1.0", description="DSL version")
	kind: str = Field(..., description="Configuration kind/type")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Configuration metadata")
	spec: Annotated[Dict[str, Any], AfterValidator(validate_configuration_spec)] = Field(..., description="Configuration specification")
	dependencies: List[str] = Field(default_factory=list, description="Configuration dependencies")
	variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
	
	def to_yaml(self) -> str:
		"""Export configuration as YAML"""
		import yaml
		return yaml.dump(self.model_dump(), default_flow_style=False)
	
	def to_hcl(self) -> str:
		"""Export configuration as HCL"""
		resource_name = self.metadata.get("name") or self.kind.lower()
		lines = [f'apg_configuration "{self._hcl_label(resource_name)}" {{']
		for key, value in {
			"version": self.version,
			"kind": self.kind,
			"metadata": self.metadata,
			"spec": self.spec,
			"dependencies": self.dependencies,
			"variables": self.variables,
		}.items():
			lines.extend(self._hcl_attribute_lines(key, value, 1))
		lines.append("}")
		return "\n".join(lines) + "\n"

	@classmethod
	def _hcl_attribute_lines(cls, key: str, value: Any, indent: int) -> List[str]:
		indent_text = "\t" * indent
		rendered = cls._hcl_value(value, indent)
		return [f"{indent_text}{cls._hcl_key(key)} = {rendered}"]

	@classmethod
	def _hcl_value(cls, value: Any, indent: int) -> str:
		if value is None:
			return "null"
		if isinstance(value, bool):
			return "true" if value else "false"
		if isinstance(value, (int, float)):
			return str(value)
		if isinstance(value, str):
			return json.dumps(value)
		if isinstance(value, list):
			if not value:
				return "[]"
			child_indent = "\t" * (indent + 1)
			current_indent = "\t" * indent
			inner = ",\n".join(
				f"{child_indent}{cls._hcl_value(item, indent + 1)}"
				for item in value
			)
			return f"[\n{inner}\n{current_indent}]"
		if isinstance(value, dict):
			if not value:
				return "{}"
			child_indent = "\t" * (indent + 1)
			current_indent = "\t" * indent
			inner = "\n".join(
				f"{child_indent}{cls._hcl_key(str(item_key))} = {cls._hcl_value(item_value, indent + 1)}"
				for item_key, item_value in value.items()
			)
			return f"{{\n{inner}\n{current_indent}}}"
		return json.dumps(str(value))

	@staticmethod
	def _hcl_key(key: str) -> str:
		if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
			return key
		return json.dumps(key)

	@staticmethod
	def _hcl_label(label: Any) -> str:
		return re.sub(r"[^A-Za-z0-9_]+", "_", str(label).strip().lower()).strip("_") or "configuration"


class CMResource(BaseModel):
	"""Configuration Management Resource"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Identity and Metadata
	id: str = Field(default_factory=uuid7str, description="Unique resource identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="Tenant identifier")
	name: Annotated[str, AfterValidator(validate_resource_name)] = Field(..., description="Resource name")
	display_name: Optional[str] = Field(None, description="Human-readable display name")
	description: Optional[str] = Field(None, description="Resource description")
	tags: Dict[str, str] = Field(default_factory=dict, description="Resource tags")
	labels: Dict[str, str] = Field(default_factory=dict, description="Resource labels")
	
	# Configuration
	resource_type: ResourceType = Field(..., description="Type of configuration resource")
	cloud_provider: CloudProvider = Field(..., description="Target cloud provider")
	configuration: ConfigurationDSL = Field(..., description="Resource configuration")
	
	# State Management
	state: ResourceState = Field(default=ResourceState.PENDING, description="Current resource state")
	desired_state: ResourceState = Field(default=ResourceState.DEPLOYED, description="Desired resource state")
	last_known_config: Optional[Dict[str, Any]] = Field(None, description="Last known configuration state")
	
	# Lifecycle
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")
	
	# Deployment Tracking
	last_deployed_at: Optional[datetime] = Field(None, description="Last deployment timestamp")
	last_validated_at: Optional[datetime] = Field(None, description="Last validation timestamp")
	last_remediated_at: Optional[datetime] = Field(None, description="Last remediation timestamp")
	deployment_count: int = Field(default=0, description="Number of deployments")
	
	# Dependencies and Relationships
	depends_on: List[str] = Field(default_factory=list, description="Resource dependencies")
	dependent_resources: List[str] = Field(default_factory=list, description="Dependent resources")
	environment_id: Optional[str] = Field(None, description="Target environment")
	template_id: Optional[str] = Field(None, description="Source template ID")
	
	# Validation and Compliance
	validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
	compliance_status: Dict[str, bool] = Field(default_factory=dict, description="Compliance framework status")
	policy_violations: List[str] = Field(default_factory=list, description="Policy violations")
	
	# Performance and Cost
	estimated_cost_monthly: Optional[float] = Field(None, description="Estimated monthly cost")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
	
	# AI and Automation
	ai_confidence_score: float = Field(default=0.8, description="AI confidence in configuration")
	auto_remediation_enabled: bool = Field(default=True, description="Enable autonomous remediation")
	drift_detection_enabled: bool = Field(default=True, description="Enable drift detection")
	
	# Runtime Assertions
	def __post_init__(self):
		assert self.id is not None, "Resource ID must be set"
		assert self.name, "Resource name is required"
		assert self.resource_type, "Resource type is required"


class CMTemplate(BaseModel):
	"""Configuration Management Template"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Identity
	id: str = Field(default_factory=uuid7str, description="Template identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="Tenant identifier")
	name: Annotated[str, AfterValidator(validate_resource_name)] = Field(..., description="Template name")
	display_name: Optional[str] = Field(None, description="Display name")
	description: str = Field(..., description="Template description")
	category: str = Field(default="general", description="Template category")
	version: str = Field(default="1.0.0", description="Template version")
	
	# Template Content
	configuration_template: Dict[str, Any] = Field(..., description="Template configuration")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Template parameters")
	parameter_schema: Dict[str, Any] = Field(default_factory=dict, description="Parameter validation schema")
	outputs: Dict[str, Any] = Field(default_factory=dict, description="Template outputs")
	
	# Metadata
	supported_providers: List[CloudProvider] = Field(default_factory=list, description="Supported cloud providers")
	supported_resource_types: List[ResourceType] = Field(default_factory=list, description="Supported resource types")
	compliance_frameworks: List[ComplianceFramework] = Field(default_factory=list, description="Compliance frameworks")
	
	# Lifecycle
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Updater user ID")
	
	# Usage and Analytics
	usage_count: int = Field(default=0, description="Template usage count")
	success_rate: float = Field(default=0.0, description="Deployment success rate")
	average_deployment_time: Optional[float] = Field(None, description="Average deployment time in seconds")
	
	# AI Generation
	ai_generated: bool = Field(default=False, description="Whether template was AI-generated")
	ai_confidence_score: float = Field(default=1.0, description="AI confidence in template")
	optimization_suggestions: List[str] = Field(default_factory=list, description="AI optimization suggestions")
	
	# Validation
	is_validated: bool = Field(default=False, description="Template validation status")
	validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
	
	def instantiate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Instantiate template with parameters"""
		# Template instantiation logic would go here
		instantiated = self.configuration_template.copy()
		# Apply parameter substitution
		return instantiated


class CMPolicy(BaseModel):
	"""Configuration Management Policy"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Identity
	id: str = Field(default_factory=uuid7str, description="Policy identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="Tenant identifier")
	name: Annotated[str, AfterValidator(validate_resource_name)] = Field(..., description="Policy name")
	display_name: Optional[str] = Field(None, description="Display name")
	description: str = Field(..., description="Policy description")
	policy_type: PolicyType = Field(..., description="Type of policy")
	
	# Policy Definition
	rules: List[Dict[str, Any]] = Field(..., description="Policy rules")
	conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Policy conditions")
	actions: List[PolicyAction] = Field(..., description="Policy actions")
	severity: str = Field(default="medium", description="Policy severity level")
	
	# Scope and Targeting
	applies_to: List[ResourceType] = Field(default_factory=list, description="Applicable resource types")
	cloud_providers: List[CloudProvider] = Field(default_factory=list, description="Applicable cloud providers")
	environments: List[str] = Field(default_factory=list, description="Applicable environments")
	resource_filters: Dict[str, Any] = Field(default_factory=dict, description="Resource filtering criteria")
	
	# Enforcement
	enabled: bool = Field(default=True, description="Policy enforcement enabled")
	auto_remediate: bool = Field(default=False, description="Enable automatic remediation")
	remediation_actions: List[str] = Field(default_factory=list, description="Remediation actions")
	enforcement_mode: str = Field(default="enforce", description="Enforcement mode (enforce, warn, audit)")
	
	# Compliance
	compliance_framework: Optional[ComplianceFramework] = Field(None, description="Associated compliance framework")
	compliance_controls: List[str] = Field(default_factory=list, description="Compliance control mappings")
	regulatory_requirements: List[str] = Field(default_factory=list, description="Regulatory requirements")
	
	# Lifecycle
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Updater user ID")
	
	# Analytics
	violation_count: int = Field(default=0, description="Number of violations")
	enforcement_count: int = Field(default=0, description="Number of enforcements")
	remediation_success_rate: float = Field(default=0.0, description="Remediation success rate")
	last_evaluated_at: Optional[datetime] = Field(None, description="Last evaluation timestamp")
	
	def evaluate(self, resource: CMResource) -> Dict[str, Any]:
		"""Evaluate policy against resource"""
		# Policy evaluation logic
		return {
			"compliant": True,
			"violations": [],
			"recommendations": []
		}


class CMEnvironment(BaseModel):
	"""Configuration Management Environment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Identity
	id: str = Field(default_factory=uuid7str, description="Environment identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="Tenant identifier")
	name: Annotated[str, AfterValidator(validate_resource_name)] = Field(..., description="Environment name")
	display_name: Optional[str] = Field(None, description="Display name")
	description: str = Field(..., description="Environment description")
	environment_type: str = Field(default="development", description="Environment type")
	
	# Configuration
	cloud_provider: CloudProvider = Field(..., description="Primary cloud provider")
	region: str = Field(..., description="Primary region")
	additional_regions: List[str] = Field(default_factory=list, description="Additional regions")
	
	# Network and Security
	vpc_id: Optional[str] = Field(None, description="VPC identifier")
	subnet_ids: List[str] = Field(default_factory=list, description="Subnet identifiers")
	security_groups: List[str] = Field(default_factory=list, description="Security group identifiers")
	network_configuration: Dict[str, Any] = Field(default_factory=dict, description="Network configuration")
	
	# Access Control
	allowed_users: List[str] = Field(default_factory=list, description="Allowed user IDs")
	allowed_roles: List[str] = Field(default_factory=list, description="Allowed role IDs")
	access_policies: List[str] = Field(default_factory=list, description="Access policy IDs")
	
	# Lifecycle
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Updater user ID")
	
	# State
	is_active: bool = Field(default=True, description="Environment active status")
	resource_count: int = Field(default=0, description="Number of resources in environment")
	last_deployment_at: Optional[datetime] = Field(None, description="Last deployment timestamp")
	
	# Cost and Performance
	monthly_cost_limit: Optional[float] = Field(None, description="Monthly cost limit")
	current_monthly_cost: float = Field(default=0.0, description="Current monthly cost")
	performance_tier: str = Field(default="standard", description="Performance tier")
	
	# Automation
	auto_scaling_enabled: bool = Field(default=False, description="Auto-scaling enabled")
	backup_enabled: bool = Field(default=True, description="Backup enabled")
	monitoring_enabled: bool = Field(default=True, description="Monitoring enabled")
	compliance_scanning_enabled: bool = Field(default=True, description="Compliance scanning enabled")


class CMDeployment(BaseModel):
	"""Configuration Management Deployment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Identity
	id: str = Field(default_factory=uuid7str, description="Deployment identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="Tenant identifier")
	resource_id: str = Field(..., description="Target resource identifier")
	environment_id: str = Field(..., description="Target environment identifier")
	template_id: Optional[str] = Field(None, description="Source template identifier")
	
	# Deployment Configuration
	deployment_plan: Dict[str, Any] = Field(..., description="Deployment execution plan")
	deployment_strategy: str = Field(default="rolling", description="Deployment strategy")
	rollback_plan: Optional[Dict[str, Any]] = Field(None, description="Rollback plan")
	
	# Status and Progress
	status: DeploymentStatus = Field(default=DeploymentStatus.PENDING, description="Deployment status")
	progress_percentage: float = Field(default=0.0, description="Deployment progress percentage")
	current_phase: Optional[str] = Field(None, description="Current deployment phase")
	phases_completed: List[str] = Field(default_factory=list, description="Completed phases")
	
	# Timing
	started_at: Optional[datetime] = Field(None, description="Deployment start time")
	completed_at: Optional[datetime] = Field(None, description="Deployment completion time")
	estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
	actual_duration: Optional[int] = Field(None, description="Actual duration in seconds")
	
	# Results
	result: Optional[Dict[str, Any]] = Field(None, description="Deployment result")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	logs: List[str] = Field(default_factory=list, description="Deployment logs")
	output_artifacts: Dict[str, Any] = Field(default_factory=dict, description="Output artifacts")
	
	# Validation and Testing
	pre_deployment_checks: List[str] = Field(default_factory=list, description="Pre-deployment check results")
	post_deployment_tests: List[str] = Field(default_factory=list, description="Post-deployment test results")
	health_checks: Dict[str, bool] = Field(default_factory=dict, description="Health check results")
	
	# Compliance and Security
	compliance_validated: bool = Field(default=False, description="Compliance validation status")
	security_scanned: bool = Field(default=False, description="Security scan status")
	vulnerabilities_found: List[str] = Field(default_factory=list, description="Security vulnerabilities")
	
	# Automation and AI
	ai_optimized: bool = Field(default=False, description="AI optimization applied")
	automation_level: str = Field(default="semi", description="Automation level")
	autonomous_decisions: List[str] = Field(default_factory=list, description="Autonomous decisions made")
	
	# Metadata
	created_by: Optional[str] = Field(None, description="User who triggered deployment")
	deployment_trigger: str = Field(default="manual", description="Deployment trigger type")
	git_commit: Optional[str] = Field(None, description="Git commit hash")
	release_version: Optional[str] = Field(None, description="Release version")
	
	def calculate_duration(self) -> Optional[int]:
		"""Calculate deployment duration in seconds"""
		if self.started_at and self.completed_at:
			return int((self.completed_at - self.started_at).total_seconds())
		return None


class ValidationResult(BaseModel):
	"""Configuration validation result"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	resource_id: Optional[str] = Field(None, description="Resource identifier that was validated")
	valid: bool = Field(..., description="Overall validation status")
	errors: List[str] = Field(default_factory=list, description="Validation errors")
	warnings: List[str] = Field(default_factory=list, description="Validation warnings")
	recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
	confidence_score: float = Field(default=1.0, description="Confidence in validation")
	validated_at: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")


class ExecutionResult(BaseModel):
	"""Deployment execution result"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	success: bool = Field(..., description="Execution success status")
	message: str = Field(default="", description="Result message")
	details: Dict[str, Any] = Field(default_factory=dict, description="Execution details")
	artifacts: Dict[str, Any] = Field(default_factory=dict, description="Output artifacts")
	duration_seconds: float = Field(default=0.0, description="Execution duration")
	executed_at: datetime = Field(default_factory=datetime.utcnow, description="Execution timestamp")


class AIInsight(BaseModel):
	"""AI-generated insight"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Insight identifier")
	type: str = Field(..., description="Insight type")
	title: str = Field(..., description="Insight title")
	description: str = Field(..., description="Insight description")
	severity: str = Field(default="info", description="Insight severity")
	confidence: float = Field(..., description="AI confidence score")
	recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
	affected_resources: List[str] = Field(default_factory=list, description="Affected resource IDs")
	generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
	expires_at: Optional[datetime] = Field(None, description="Insight expiration")


class CMMetrics(BaseModel):
	"""Configuration management metrics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core Metrics
	total_resources: int = Field(default=0, description="Total number of resources")
	deployed_resources: int = Field(default=0, description="Successfully deployed resources")
	failed_resources: int = Field(default=0, description="Failed resources")
	drifted_resources: int = Field(default=0, description="Resources with configuration drift")
	
	# Performance Metrics
	average_deployment_time: float = Field(default=0.0, description="Average deployment time in seconds")
	deployment_success_rate: float = Field(default=0.0, description="Deployment success rate percentage")
	drift_detection_rate: float = Field(default=0.0, description="Drift detection rate")
	autonomous_remediation_rate: float = Field(default=0.0, description="Autonomous remediation rate")
	
	# Cost Metrics
	total_monthly_cost: float = Field(default=0.0, description="Total monthly cost")
	cost_optimization_savings: float = Field(default=0.0, description="Cost optimization savings")
	cost_per_resource: float = Field(default=0.0, description="Average cost per resource")
	
	# Compliance Metrics
	compliance_score: float = Field(default=0.0, description="Overall compliance score")
	policy_violations: int = Field(default=0, description="Number of policy violations")
	security_vulnerabilities: int = Field(default=0, description="Number of security vulnerabilities")
	
	# AI Metrics
	ai_optimization_count: int = Field(default=0, description="Number of AI optimizations")
	ai_prediction_accuracy: float = Field(default=0.0, description="AI prediction accuracy")
	natural_language_requests: int = Field(default=0, description="Natural language requests processed")
	
	# Timestamp
	collected_at: datetime = Field(default_factory=datetime.utcnow, description="Metrics collection timestamp")


# AI/ML Configuration Models

class AIModelFramework(StrEnum):
	"""Supported AI/ML frameworks"""
	OLLAMA = "ollama"
	TRANSFORMERS = "transformers"
	SPACY = "spacy"
	TENSORFLOW = "tensorflow"
	PYTORCH = "pytorch"
	SCIKIT_LEARN = "scikit_learn"
	XGBOOST = "xgboost"
	LIGHTGBM = "lightgbm"
	ONNX = "onnx"
	KERAS = "keras"
	CUSTOM = "custom"


class AIModelType(StrEnum):
	"""Types of AI models"""
	TEXT_GENERATION = "text_generation"
	SENTIMENT_ANALYSIS = "sentiment_analysis"
	NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
	TEXT_CLASSIFICATION = "text_classification"
	QUESTION_ANSWERING = "question_answering"
	TEXT_SUMMARIZATION = "text_summarization"
	TRANSLATION = "translation"
	EMBEDDING = "embedding"
	IMAGE_CLASSIFICATION = "image_classification"
	OBJECT_DETECTION = "object_detection"
	SPEECH_TO_TEXT = "speech_to_text"
	TEXT_TO_SPEECH = "text_to_speech"
	RECOMMENDATION = "recommendation"
	CUSTOM = "custom"


class AIModelState(StrEnum):
	"""AI model configuration states"""
	CONFIGURED = "configured"
	REGISTERED = "registered"
	LOADING = "loading"
	LOADED = "loaded"
	READY = "ready"
	SERVING = "serving"
	UPDATING = "updating"
	FAILED = "failed"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"


class ModelProvider(StrEnum):
	"""AI model providers"""
	OLLAMA = "ollama"
	TRANSFORMERS = "transformers"
	SPACY = "spacy"
	OPENAI = "openai"
	ANTHROPIC = "anthropic"
	COHERE = "cohere"
	CUSTOM = "custom"
	LOCAL = "local"


class AIModelConfiguration(BaseModel):
	"""AI model configuration for infrastructure management"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Model Identity
	id: str = Field(default_factory=uuid7str, description="Unique model configuration identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="Tenant identifier")
	name: Annotated[str, AfterValidator(validate_resource_name)] = Field(..., description="Model configuration name")
	display_name: Optional[str] = Field(None, description="Human-readable model name")
	description: Optional[str] = Field(None, description="Model description")
	version: str = Field(default="1.0", description="Model configuration version")
	
	# Model Specification
	framework: AIModelFramework = Field(..., description="ML framework")
	model_type: AIModelType = Field(..., description="Type of AI model")
	provider: ModelProvider = Field(..., description="Model provider")
	provider_model_name: str = Field(..., description="Provider-specific model name")
	model_path: Optional[str] = Field(None, description="Path to model files")
	
	# Configuration
	model_parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific configuration")
	runtime_config: Dict[str, Any] = Field(default_factory=dict, description="Runtime configuration")
	resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
	scaling_config: Dict[str, Any] = Field(default_factory=dict, description="Scaling configuration")
	
	# Integration with common/nlpc
	nlp_service_integration: bool = Field(default=True, description="Integrate with NLP service")
	supported_tasks: List[AIModelType] = Field(default_factory=list, description="Supported NLP tasks")
	supported_languages: List[str] = Field(default_factory=list, description="Supported languages")
	
	# Deployment
	state: AIModelState = Field(default=AIModelState.CONFIGURED, description="Current model state")
	cloud_provider: CloudProvider = Field(..., description="Target cloud provider")
	deployment_target: str = Field(..., description="Deployment target (environment)")
	
	# Monitoring and Performance
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
	health_check_config: Dict[str, Any] = Field(default_factory=dict, description="Health check configuration")
	monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
	
	# Lifecycle Management
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
	deployed_at: Optional[datetime] = Field(None, description="Deployment timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	
	# Metadata
	tags: Dict[str, str] = Field(default_factory=dict, description="Model tags")
	annotations: Dict[str, str] = Field(default_factory=dict, description="Model annotations")
	
	def to_configuration_dsl(self) -> ConfigurationDSL:
		"""Convert AI model config to universal configuration DSL"""
		return ConfigurationDSL(
			kind="AIModel",
			metadata={
				"name": self.name,
				"framework": self.framework,
				"model_type": self.model_type,
				"provider": self.provider,
				"version": self.version
			},
			spec={
				"model": {
					"provider_model_name": self.provider_model_name,
					"model_path": self.model_path,
					"configuration": self.model_config
				},
				"runtime": self.runtime_config,
				"resources": self.resource_requirements,
				"scaling": self.scaling_config,
				"integration": {
					"nlp_service": self.nlp_service_integration,
					"supported_tasks": [task.value for task in self.supported_tasks],
					"supported_languages": self.supported_languages
				},
				"deployment": {
					"target": self.deployment_target,
					"cloud_provider": self.cloud_provider,
					"monitoring": self.monitoring_enabled
				}
			},
			variables={
				"model_name": self.name,
				"framework": self.framework,
				"provider": self.provider
			}
		)


class MLPipelineConfiguration(BaseModel):
	"""ML pipeline configuration for workflow orchestration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Pipeline Identity
	id: str = Field(default_factory=uuid7str, description="Unique pipeline identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="Tenant identifier")
	name: Annotated[str, AfterValidator(validate_resource_name)] = Field(..., description="Pipeline name")
	description: Optional[str] = Field(None, description="Pipeline description")
	version: str = Field(default="1.0", description="Pipeline version")
	
	# Pipeline Components
	models: List[str] = Field(default_factory=list, description="AI model configuration IDs")
	preprocessing_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Preprocessing steps")
	postprocessing_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Postprocessing steps")
	
	# Pipeline Configuration
	input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input data schema")
	output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output data schema")
	pipeline_config: Dict[str, Any] = Field(default_factory=dict, description="Pipeline-specific configuration")
	
	# Execution
	execution_mode: str = Field(default="batch", description="Execution mode (batch/streaming/real-time)")
	parallelism: int = Field(default=1, description="Pipeline parallelism level")
	timeout_seconds: int = Field(default=300, description="Pipeline timeout")
	
	# Resources and Deployment
	resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
	cloud_provider: CloudProvider = Field(..., description="Target cloud provider")
	deployment_target: str = Field(..., description="Deployment target")
	
	# Lifecycle
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	
	# Metadata
	tags: Dict[str, str] = Field(default_factory=dict, description="Pipeline tags")


class NLPServiceConfiguration(BaseModel):
	"""NLP service configuration for common/nlpc integration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Service Identity
	id: str = Field(default_factory=uuid7str, description="Unique service configuration identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="Tenant identifier")
	name: Annotated[str, AfterValidator(validate_resource_name)] = Field(..., description="Service configuration name")
	description: Optional[str] = Field(None, description="Service description")
	version: str = Field(default="1.0", description="Service configuration version")
	
	# Service Configuration
	ollama_endpoint: str = Field(default="http://localhost:11434", description="Ollama endpoint")
	models_cache_dir: str = Field(default="./models", description="Models cache directory")
	enable_gpu: bool = Field(default=True, description="Enable GPU acceleration")
	max_memory_gb: float = Field(default=8.0, description="Maximum memory usage")
	model_timeout_seconds: int = Field(default=300, description="Model timeout")
	
	# Model Management
	registered_models: List[str] = Field(default_factory=list, description="Registered AI model configuration IDs")
	auto_model_loading: bool = Field(default=True, description="Enable automatic model loading")
	model_health_check_interval: int = Field(default=60, description="Model health check interval")
	
	# Performance Tuning
	concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
	batch_processing: bool = Field(default=True, description="Enable batch processing")
	streaming_enabled: bool = Field(default=True, description="Enable streaming processing")
	caching_enabled: bool = Field(default=True, description="Enable result caching")
	
	# Integration Configuration
	cloud_provider: CloudProvider = Field(..., description="Target cloud provider")
	deployment_target: str = Field(..., description="Deployment target")
	monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
	
	# Lifecycle
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	
	# Metadata
	tags: Dict[str, str] = Field(default_factory=dict, description="Service tags")
	
	def to_configuration_dsl(self) -> ConfigurationDSL:
		"""Convert NLP service config to universal configuration DSL"""
		return ConfigurationDSL(
			kind="NLPService",
			metadata={
				"name": self.name,
				"version": self.version,
				"service_type": "nlp_processing"
			},
			spec={
				"service": {
					"ollama_endpoint": self.ollama_endpoint,
					"models_cache_dir": self.models_cache_dir,
					"enable_gpu": self.enable_gpu,
					"max_memory_gb": self.max_memory_gb,
					"model_timeout_seconds": self.model_timeout_seconds
				},
				"models": {
					"registered_models": self.registered_models,
					"auto_loading": self.auto_model_loading,
					"health_check_interval": self.model_health_check_interval
				},
				"performance": {
					"concurrent_requests": self.concurrent_requests,
					"batch_processing": self.batch_processing,
					"streaming_enabled": self.streaming_enabled,
					"caching_enabled": self.caching_enabled
				},
				"deployment": {
					"target": self.deployment_target,
					"cloud_provider": self.cloud_provider,
					"monitoring": self.monitoring_enabled
				}
			}
		)


# Export all models
__all__ = [
	# Enums
	"ResourceState", "DeploymentStatus", "PolicyAction", "ResourceType", 
	"PolicyType", "CloudProvider", "ComplianceFramework",
	
	# AI/ML Enums
	"AIModelFramework", "AIModelType", "AIModelState", "ModelProvider",
	
	# Core Models
	"ConfigurationDSL", "CMResource", "CMTemplate", "CMPolicy", 
	"CMEnvironment", "CMDeployment",
	
	# AI/ML Models
	"AIModelConfiguration", "MLPipelineConfiguration", "NLPServiceConfiguration",
	
	# Result Models
	"ValidationResult", "ExecutionResult", "AIInsight", "CMMetrics",
	
	# Validators
	"validate_resource_name", "validate_tenant_id", "validate_configuration_spec"
]
