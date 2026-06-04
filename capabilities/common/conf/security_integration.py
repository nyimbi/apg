"""
APG Configuration Management Security Integration Layer

Integrates the APG Security Framework with Configuration Management to provide
comprehensive security governance, policy enforcement, and threat protection
for infrastructure configuration operations.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from uuid_extensions import uuid7str
import asyncio
import logging

try:
    from .models import (
        CMResource, CMDeployment, CMPolicy, ValidationResult, ExecutionResult,
        ResourceState, DeploymentStatus, PolicyAction, ResourceType, CloudProvider
    )
    from ..secu.service import APGSecurityFrameworkService
    from ..secu.models import SecurityContext, SecurityPolicy, RiskScore, ThreatIndicator
    from ..secu import SecurityLevel, RiskLevel, ThreatType, SecurityAction
except ImportError:
    # For direct imports during testing
    try:
        from .models import (
            CMResource, CMDeployment, CMPolicy, ValidationResult, ExecutionResult,
            ResourceState, DeploymentStatus, PolicyAction, ResourceType, CloudProvider
        )
    except ImportError:
        from capabilities.common.conf.models import (
            CMResource, CMDeployment, CMPolicy, ValidationResult, ExecutionResult,
            ResourceState, DeploymentStatus, PolicyAction, ResourceType, CloudProvider
        )
    # Create mock security classes for testing
    class SecurityContext:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class SecurityPolicy:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.id = uuid7str()
    
    class RiskScore:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.overall_score = 0.0
    
    class ThreatIndicator:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.id = uuid7str()
    
    class APGSecurityFrameworkService:
        async def initialize(self): pass
        async def assess_security_context(self, context): 
            context.risk_score = RiskScore(overall_score=25.0)
            context.threat_indicators = []
            return context
    
    # Mock enums for testing
    class SecurityLevel:
        HIGH = "high"
    class RiskLevel:
        LOW = "low"
        MODERATE = "moderate" 
        HIGH = "high"
        CRITICAL = "critical"
    class ThreatType:
        DATA_EXFILTRATION = "data_exfiltration"
        PRIVILEGE_ESCALATION = "privilege_escalation"
        INSIDER_THREAT = "insider_threat"
    class SecurityAction:
        REQUIRE_MFA = "require_mfa"
        REQUIRE_APPROVAL = "require_approval"
        CHALLENGE = "challenge"
        BLOCK = "block"
        DENY = "deny"
        LOG_DETAILED = "log_detailed"
        NOTIFY_ADMIN = "notify_admin"
        MONITOR = "monitor"
        ALERT = "alert"
        BACKUP = "backup"

logger = logging.getLogger(__name__)


class ConfigurationSecurityLevel(StrEnum):
    """Configuration-specific security levels"""
    PUBLIC = "public"           # Public configurations
    INTERNAL = "internal"       # Internal use configurations
    CONFIDENTIAL = "confidential"  # Confidential configurations
    RESTRICTED = "restricted"   # Highly restricted configurations
    TOP_SECRET = "top_secret"   # Maximum security configurations


class ConfigurationThreatType(StrEnum):
    """Configuration-specific threat types"""
    CONFIGURATION_DRIFT = "configuration_drift"
    UNAUTHORIZED_CHANGES = "unauthorized_changes"
    MALICIOUS_CONFIG = "malicious_configuration"
    COMPLIANCE_VIOLATION = "compliance_violation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXPOSURE = "data_exposure"
    SUPPLY_CHAIN_ATTACK = "supply_chain_attack"


class SecurityPolicyCategory(StrEnum):
    """Security policy categories for configuration management"""
    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    CHANGE_MANAGEMENT = "change_management"
    COMPLIANCE = "compliance"
    INCIDENT_RESPONSE = "incident_response"
    THREAT_PREVENTION = "threat_prevention"


class ConfigurationSecurityContext:
    """Security context for configuration operations"""
    
    def __init__(
        self,
        tenant_id: str,
        user_id: str,
        resource_id: Optional[str] = None,
        operation: str = "read",
        security_level: ConfigurationSecurityLevel = ConfigurationSecurityLevel.INTERNAL,
        device_context: Optional[Dict[str, Any]] = None,
        network_context: Optional[Dict[str, Any]] = None
    ):
        self.id = uuid7str()
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.resource_id = resource_id
        self.operation = operation
        self.security_level = security_level
        self.device_context = device_context or self._create_default_device_context()
        self.network_context = network_context or self._create_default_network_context()
        self.created_at = datetime.utcnow()
        self.risk_score: Optional[RiskScore] = None
        self.threat_indicators: List[ThreatIndicator] = []
        self.security_decisions: List[SecurityAction] = []
    
    def _create_default_device_context(self) -> Dict[str, Any]:
        """Create default device context"""
        return {
            "device_id": f"default-{uuid7str()[:8]}",
            "device_type": "unknown",
            "os_type": "unknown",
            "trust_level": "unknown"
        }
    
    def _create_default_network_context(self) -> Dict[str, Any]:
        """Create default network context"""
        return {
            "ip_address": "127.0.0.1",
            "country": "unknown",
            "city": "unknown",
            "is_vpn": False,
            "is_tor": False,
            "is_proxy": False,
            "is_known_malicious": False,
            "reputation_score": 50.0
        }
    
    def to_security_context(self) -> SecurityContext:
        """Convert to base SecurityContext for framework integration"""
        return SecurityContext(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            capability_id="configuration_management",
            action=self.operation,
            device_context=self.device_context,
            network_context=self.network_context,
            risk_score=self.risk_score,
            threat_indicators=self.threat_indicators,
            created_at=self.created_at
        )


class ConfigurationSecurityEngine:
    """Security engine specifically for configuration management operations"""
    
    def __init__(self, security_service: APGSecurityFrameworkService):
        self.security_service = security_service
        self.config_policies: Dict[str, SecurityPolicy] = {}
        self.threat_patterns: Dict[str, Any] = {}
        self.compliance_rules: Dict[str, Any] = {}
        self.policy_engine = None  # Advanced policy engine
        self._initialized = False
    
    async def initialize(self):
        """Initialize configuration security engine"""
        if not self._initialized:
            await self.security_service.initialize()
            await self._load_configuration_policies()
            await self._load_threat_patterns()
            await self._load_compliance_rules()
            
            # Initialize advanced policy engine
            try:
                from .policy_engine import get_policy_engine
                self.policy_engine = await get_policy_engine()
            except ImportError:
                logger.warning("Advanced policy engine not available, using basic policies only")
            
            self._initialized = True
            logger.info("Configuration Security Engine initialized")
    
    async def _load_configuration_policies(self):
        """Load configuration-specific security policies"""
        default_policies = [
            {
                "name": "High-Value Configuration Protection",
                "category": SecurityPolicyCategory.ACCESS_CONTROL,
                "conditions": {
                    "security_level": {"operator": "in", "value": ["confidential", "restricted", "top_secret"]},
                    "operation": {"operator": "in", "value": ["create", "update", "delete"]}
                },
                "actions": [SecurityAction.REQUIRE_MFA, SecurityAction.LOG_DETAILED, SecurityAction.NOTIFY_ADMIN],
                "priority": 1
            },
            {
                "name": "Production Configuration Change Control",
                "category": SecurityPolicyCategory.CHANGE_MANAGEMENT,
                "conditions": {
                    "environment": {"operator": "eq", "value": "production"},
                    "operation": {"operator": "in", "value": ["update", "delete", "deploy"]}
                },
                "actions": [SecurityAction.REQUIRE_APPROVAL, SecurityAction.LOG_DETAILED, SecurityAction.BACKUP],
                "priority": 5
            },
            {
                "name": "Anomalous Configuration Activity",
                "category": SecurityPolicyCategory.THREAT_PREVENTION,
                "conditions": {
                    "risk_score": {"operator": "gte", "value": 70.0},
                    "operation": {"operator": "ne", "value": "read"}
                },
                "actions": [SecurityAction.CHALLENGE, SecurityAction.MONITOR, SecurityAction.ALERT],
                "priority": 10
            }
        ]
        
        for policy_data in default_policies:
            policy = SecurityPolicy(
                name=policy_data["name"],
                category=policy_data["category"],
                conditions=policy_data["conditions"],
                actions=policy_data["actions"],
                priority=policy_data["priority"],
                created_by="system"
            )
            self.config_policies[policy.id] = policy
        
        logger.info(f"Loaded {len(self.config_policies)} configuration security policies")
    
    async def _load_threat_patterns(self):
        """Load configuration threat detection patterns"""
        self.threat_patterns = {
            "bulk_configuration_changes": {
                "pattern": "multiple_rapid_changes",
                "threshold": 10,
                "time_window": 300,  # 5 minutes
                "threat_type": ConfigurationThreatType.UNAUTHORIZED_CHANGES,
                "severity": RiskLevel.HIGH
            },
            "privilege_escalation_config": {
                "pattern": "elevated_permissions",
                "indicators": ["admin_role", "root_access", "sudo_privileges"],
                "threat_type": ConfigurationThreatType.PRIVILEGE_ESCALATION,
                "severity": RiskLevel.CRITICAL
            },
            "sensitive_data_exposure": {
                "pattern": "data_leakage_risk",
                "indicators": ["passwords", "api_keys", "secrets", "tokens"],
                "threat_type": ConfigurationThreatType.DATA_EXPOSURE,
                "severity": RiskLevel.HIGH
            }
        }
        
        logger.info(f"Loaded {len(self.threat_patterns)} threat detection patterns")
    
    async def _load_compliance_rules(self):
        """Load compliance validation rules"""
        self.compliance_rules = {
            "encryption_required": {
                "frameworks": ["GDPR", "HIPAA", "SOX"],
                "rule": "All sensitive configurations must use encryption at rest and in transit",
                "validation": "check_encryption_enabled",
                "severity": RiskLevel.HIGH
            },
            "access_logging_required": {
                "frameworks": ["SOX", "PCI_DSS"],
                "rule": "All configuration access must be logged and auditable",
                "validation": "check_audit_logging",
                "severity": RiskLevel.MODERATE
            },
            "change_approval_required": {
                "frameworks": ["SOX", "ISO_27001"],
                "rule": "Production configuration changes require approval",
                "validation": "check_change_approval",
                "severity": RiskLevel.HIGH
            }
        }
        
        logger.info(f"Loaded {len(self.compliance_rules)} compliance validation rules")
    
    async def assess_configuration_security(
        self,
        context: ConfigurationSecurityContext,
        resource: Optional[CMResource] = None
    ) -> ConfigurationSecurityContext:
        """Perform comprehensive security assessment for configuration operation"""
        assert self._initialized, "Security engine not initialized"
        
        # Convert to base security context for framework integration
        base_context = context.to_security_context()
        
        # Enhance context with configuration-specific data
        if resource:
            base_context.resource_id = resource.id
            base_context.metadata = {
                "resource_type": resource.resource_type.value,
                "cloud_provider": resource.cloud_provider.value,
                "security_level": context.security_level.value,
                "configuration_size": len(str(resource.configuration.model_dump()))
            }
        
        # Perform base security assessment
        assessed_context = await self.security_service.assess_security_context(base_context)
        
        # Update configuration context with results
        context.risk_score = assessed_context.risk_score
        context.threat_indicators = assessed_context.threat_indicators
        
        # Perform configuration-specific threat detection
        config_threats = await self._detect_configuration_threats(context, resource)
        context.threat_indicators.extend(config_threats)
        
        # Apply configuration security policies
        security_actions = await self._apply_configuration_policies(context, resource)
        context.security_decisions = security_actions
        
        # Log security assessment
        await self._log_security_assessment(context, resource)
        
        return context
    
    async def _detect_configuration_threats(
        self,
        context: ConfigurationSecurityContext,
        resource: Optional[CMResource] = None
    ) -> List[ThreatIndicator]:
        """Detect configuration-specific threats"""
        threats = []
        
        if not resource:
            return threats
        
        # Check for sensitive data exposure
        config_content = str(resource.configuration.model_dump()).lower()
        sensitive_patterns = ["password", "secret", "token", "key", "credential"]
        
        for pattern in sensitive_patterns:
            if pattern in config_content:
                threats.append(ThreatIndicator(
                    threat_type=ThreatType.DATA_EXFILTRATION,  # Using base enum
                    severity=RiskLevel.HIGH,
                    confidence=85.0,
                    source="config_analyzer",
                    title="Sensitive Data in Configuration",
                    description=f"Configuration contains potential {pattern}",
                    indicators={"pattern": pattern, "resource_id": resource.id},
                    mitigation="Review configuration for hardcoded secrets"
                ))
        
        # Check for privilege escalation configurations
        if resource.resource_type in [ResourceType.VIRTUAL_MACHINE, ResourceType.CONTAINER]:
            if self._check_privilege_escalation_config(resource):
                threats.append(ThreatIndicator(
                    threat_type=ThreatType.PRIVILEGE_ESCALATION,
                    severity=RiskLevel.CRITICAL,
                    confidence=90.0,
                    source="privilege_analyzer",
                    title="Privilege Escalation Configuration",
                    description="Configuration may allow privilege escalation",
                    indicators={"resource_type": resource.resource_type.value},
                    mitigation="Review and restrict elevated privileges"
                ))
        
        # Check for configuration drift
        if await self._detect_configuration_drift(resource, context):
            threats.append(ThreatIndicator(
                threat_type=ThreatType.INSIDER_THREAT,  # Using base enum for drift
                severity=RiskLevel.MODERATE,
                confidence=70.0,
                source="drift_detector",
                title="Configuration Drift Detected",
                description="Configuration has drifted from approved baseline",
                indicators={"resource_id": resource.id},
                mitigation="Investigate configuration changes and restore baseline"
            ))
        
        return threats
    
    def _check_privilege_escalation_config(self, resource: CMResource) -> bool:
        """Check if configuration allows privilege escalation"""
        config_str = str(resource.configuration.model_dump()).lower()
        escalation_indicators = [
            "sudo", "root", "administrator", "privileged", 
            "cap_sys_admin", "seccomp:unconfined", "privileged=true"
        ]
        
        return any(indicator in config_str for indicator in escalation_indicators)
    
    async def _detect_configuration_drift(
        self,
        resource: CMResource,
        context: ConfigurationSecurityContext
    ) -> bool:
        """Detect if configuration has drifted from baseline"""
        # Simplified implementation - would compare against stored baselines
        if resource.last_known_config:
            current_config = resource.configuration.model_dump()
            # Compare configurations and detect significant differences
            return len(str(current_config)) != len(str(resource.last_known_config))
        return False
    
    async def _apply_configuration_policies(
        self,
        context: ConfigurationSecurityContext,
        resource: Optional[CMResource] = None
    ) -> List[SecurityAction]:
        """Apply configuration-specific security policies"""
        security_actions = []
        
        # Use advanced policy engine if available
        if self.policy_engine:
            try:
                policy_context = {
                    "tenant_id": context.tenant_id,
                    "user_id": context.user_id,
                    "operation": context.operation,
                    "security_level": context.security_level.value
                }
                
                result, messages, actions = await self.policy_engine.evaluate_policies(policy_context, resource)
                
                # Convert policy actions to security actions
                for action in actions:
                    if hasattr(action, 'value'):
                        action_value = action.value
                    else:
                        action_value = str(action)
                    
                    # Map policy actions to security actions
                    if action_value in ["require_approval", "REQUIRE_APPROVAL"]:
                        security_actions.append(SecurityAction.REQUIRE_APPROVAL)
                    elif action_value in ["deny", "block", "DENY", "BLOCK"]:
                        security_actions.append(SecurityAction.DENY)
                    elif action_value in ["warn", "WARN"]:
                        security_actions.append(SecurityAction.ALERT)
                    elif action_value in ["require_mfa", "REQUIRE_MFA"]:
                        security_actions.append(SecurityAction.REQUIRE_MFA)
                
                logger.info(f"Advanced policy evaluation: result={result.value}, actions={len(actions)}")
                
            except Exception as e:
                logger.error(f"Advanced policy evaluation failed: {e}")
                # Fall back to basic policies
        
        # Also apply basic security policies
        applicable_policies = []
        
        # Filter policies applicable to this context
        for policy in self.config_policies.values():
            if await self._evaluate_policy_conditions(policy, context, resource):
                applicable_policies.append(policy)
        
        # Sort by priority
        applicable_policies.sort(key=lambda p: p.priority)
        
        # Collect security actions from basic policies
        for policy in applicable_policies:
            security_actions.extend(policy.actions)
            logger.info(f"Applied basic policy: {policy.name} for operation: {context.operation}")
        
        return security_actions
    
    async def _evaluate_policy_conditions(
        self,
        policy: SecurityPolicy,
        context: ConfigurationSecurityContext,
        resource: Optional[CMResource] = None
    ) -> bool:
        """Evaluate if policy conditions match configuration context"""
        for condition_key, condition_value in policy.conditions.items():
            if not await self._evaluate_condition(condition_key, condition_value, context, resource):
                return False
        return True
    
    async def _evaluate_condition(
        self,
        key: str,
        condition: Dict[str, Any],
        context: ConfigurationSecurityContext,
        resource: Optional[CMResource] = None
    ) -> bool:
        """Evaluate individual policy condition"""
        operator = condition.get("operator", "eq")
        expected_value = condition.get("value")
        
        # Get actual value from context
        actual_value = await self._get_context_value(key, context, resource)
        
        if operator == "eq":
            return actual_value == expected_value
        elif operator == "ne":
            return actual_value != expected_value
        elif operator == "in":
            return actual_value in expected_value
        elif operator == "gte":
            return float(actual_value) >= float(expected_value)
        elif operator == "lte":
            return float(actual_value) <= float(expected_value)
        
        return False
    
    async def _get_context_value(
        self,
        key: str,
        context: ConfigurationSecurityContext,
        resource: Optional[CMResource] = None
    ) -> Any:
        """Extract value from configuration context by key"""
        if key == "security_level":
            return context.security_level.value
        elif key == "operation":
            return context.operation
        elif key == "risk_score":
            return context.risk_score.overall_score if context.risk_score else 0.0
        elif key == "user_id":
            return context.user_id
        elif key == "tenant_id":
            return context.tenant_id
        elif key == "environment" and resource:
            return resource.environment_id or "unknown"
        elif key == "resource_type" and resource:
            return resource.resource_type.value
        elif key == "cloud_provider" and resource:
            return resource.cloud_provider.value
        
        return None
    
    async def _log_security_assessment(
        self,
        context: ConfigurationSecurityContext,
        resource: Optional[CMResource] = None
    ):
        """Log security assessment results"""
        risk_score = context.risk_score.overall_score if context.risk_score else 0.0
        threat_count = len(context.threat_indicators)
        action_count = len(context.security_decisions)
        
        logger.info(
            f"Configuration security assessment completed: "
            f"user={context.user_id}, operation={context.operation}, "
            f"risk_score={risk_score:.2f}, threats={threat_count}, actions={action_count}"
        )
        
        # Log high-risk assessments with more detail
        if risk_score > 70.0 or threat_count > 0:
            logger.warning(
                f"High-risk configuration operation detected: "
                f"context_id={context.id}, resource_id={resource.id if resource else 'None'}, "
                f"threats={[t.title for t in context.threat_indicators]}"
            )


class ConfigurationSecurityService:
    """Main service for configuration security integration"""
    
    def __init__(self):
        self.security_service: Optional[APGSecurityFrameworkService] = None
        self.security_engine: Optional[ConfigurationSecurityEngine] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize configuration security service"""
        if not self._initialized:
            # Initialize base security service (or create mock for testing)
            try:
                from ..secu.service import get_security_framework_service
                self.security_service = await get_security_framework_service()
            except (ImportError, ValueError):
                # Create mock security service for testing
                self.security_service = APGSecurityFrameworkService()
                await self.security_service.initialize()
            
            # Initialize configuration security engine
            self.security_engine = ConfigurationSecurityEngine(self.security_service)
            await self.security_engine.initialize()
            
            self._initialized = True
            logger.info("Configuration Security Service initialized")
    
    async def secure_configuration_operation(
        self,
        tenant_id: str,
        user_id: str,
        operation: str,
        resource: Optional[CMResource] = None,
        security_level: ConfigurationSecurityLevel = ConfigurationSecurityLevel.INTERNAL
    ) -> Tuple[bool, ConfigurationSecurityContext, List[str]]:
        """
        Secure a configuration operation with comprehensive security assessment
        
        Returns:
            (is_authorized, security_context, security_messages)
        """
        assert self._initialized, "Configuration security service not initialized"
        
        # Create security context
        context = ConfigurationSecurityContext(
            tenant_id=tenant_id,
            user_id=user_id,
            resource_id=resource.id if resource else None,
            operation=operation,
            security_level=security_level
        )
        
        # Perform security assessment
        context = await self.security_engine.assess_configuration_security(context, resource)
        
        # Determine authorization
        is_authorized = await self._evaluate_authorization(context)
        
        # Generate security messages
        security_messages = await self._generate_security_messages(context)
        
        return is_authorized, context, security_messages
    
    async def _evaluate_authorization(self, context: ConfigurationSecurityContext) -> bool:
        """Evaluate if operation should be authorized based on security assessment"""
        # Deny if critical threats detected
        for threat in context.threat_indicators:
            if threat.severity == RiskLevel.CRITICAL:
                return False
        
        # Deny if very high risk score
        if context.risk_score and context.risk_score.overall_score > 90.0:
            return False
        
        # Check for blocking security actions
        blocking_actions = [SecurityAction.BLOCK, SecurityAction.DENY]
        if any(action in context.security_decisions for action in blocking_actions):
            return False
        
        return True
    
    async def _generate_security_messages(
        self,
        context: ConfigurationSecurityContext
    ) -> List[str]:
        """Generate security messages for the operation"""
        messages = []
        
        # Risk score message
        if context.risk_score:
            if context.risk_score.overall_score > 70.0:
                messages.append(f"High risk operation (score: {context.risk_score.overall_score:.1f})")
        
        # Threat messages
        for threat in context.threat_indicators:
            if threat.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                messages.append(f"Security threat: {threat.title}")
        
        # Security action messages
        if SecurityAction.REQUIRE_MFA in context.security_decisions:
            messages.append("Multi-factor authentication required")
        if SecurityAction.REQUIRE_APPROVAL in context.security_decisions:
            messages.append("Manager approval required for this operation")
        if SecurityAction.CHALLENGE in context.security_decisions:
            messages.append("Additional authentication challenge required")
        
        return messages
    
    async def validate_configuration_compliance(
        self,
        resource: CMResource,
        tenant_id: str
    ) -> ValidationResult:
        """Validate configuration against compliance requirements"""
        assert self._initialized, "Configuration security service not initialized"
        
        errors = []
        warnings = []
        recommendations = []
        
        # Check encryption requirements
        if await self._check_encryption_required(resource):
            if not await self._verify_encryption_configured(resource):
                errors.append("Encryption is required but not properly configured")
        
        # Check access control requirements
        if not await self._verify_access_controls(resource):
            warnings.append("Access controls may not be properly configured")
        
        # Check data protection requirements
        if await self._check_sensitive_data(resource):
            recommendations.append("Consider additional data protection measures for sensitive content")
        
        # Generate compliance validation result
        is_valid = len(errors) == 0
        confidence_score = 0.9 if is_valid else 0.6
        
        return ValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            confidence_score=confidence_score
        )
    
    async def _check_encryption_required(self, resource: CMResource) -> bool:
        """Check if encryption is required for this resource"""
        # High-security resources require encryption
        sensitive_types = [
            ResourceType.DATABASE,
            ResourceType.STORAGE
        ]
        return resource.resource_type in sensitive_types
    
    async def _verify_encryption_configured(self, resource: CMResource) -> bool:
        """Verify encryption is properly configured"""
        config_str = str(resource.configuration.model_dump()).lower()
        encryption_indicators = ["encryption", "encrypted", "ssl", "tls", "https"]
        return any(indicator in config_str for indicator in encryption_indicators)
    
    async def _verify_access_controls(self, resource: CMResource) -> bool:
        """Verify access controls are configured"""
        config_str = str(resource.configuration.model_dump()).lower()
        access_indicators = ["security_group", "firewall", "acl", "rbac", "iam"]
        return any(indicator in config_str for indicator in access_indicators)
    
    async def _check_sensitive_data(self, resource: CMResource) -> bool:
        """Check if configuration contains sensitive data"""
        config_str = str(resource.configuration.model_dump()).lower()
        sensitive_patterns = ["password", "secret", "token", "key", "credential", "private"]
        return any(pattern in config_str for pattern in sensitive_patterns)


# Global service instance
_config_security_service = None

async def get_configuration_security_service() -> ConfigurationSecurityService:
    """Get global configuration security service instance"""
    global _config_security_service
    if _config_security_service is None:
        _config_security_service = ConfigurationSecurityService()
        await _config_security_service.initialize()
    return _config_security_service

# Export main classes
__all__ = [
    "ConfigurationSecurityLevel",
    "ConfigurationThreatType", 
    "SecurityPolicyCategory",
    "ConfigurationSecurityContext",
    "ConfigurationSecurityEngine",
    "ConfigurationSecurityService",
    "get_configuration_security_service"
]