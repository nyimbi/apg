"""
APG Configuration Management Policy Engine

Advanced policy engine for configuration governance, providing dynamic policy
management, rule evaluation, and automated compliance enforcement.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from uuid_extensions import uuid7str
import asyncio
import logging
import json
from dataclasses import dataclass, field

try:
    from .models import (
        CMResource, CMPolicy, PolicyAction, ResourceType, 
        CloudProvider, ValidationResult
    )
    from .security_integration import (
        ConfigurationSecurityLevel, ConfigurationThreatType,
        SecurityPolicyCategory
    )
except ImportError:
    # For direct imports during testing
    from models import (
        CMResource, CMPolicy, PolicyAction, ResourceType, 
        CloudProvider, ValidationResult
    )
    from security_integration import (
        ConfigurationSecurityLevel, ConfigurationThreatType,
        SecurityPolicyCategory
    )

logger = logging.getLogger(__name__)


class PolicyScope(StrEnum):
    """Policy application scope"""
    GLOBAL = "global"           # Apply to all resources
    TENANT = "tenant"           # Apply to specific tenant
    ENVIRONMENT = "environment" # Apply to specific environment
    RESOURCE_TYPE = "resource_type"  # Apply to specific resource type
    USER = "user"               # Apply to specific user
    ROLE = "role"               # Apply to specific role


class PolicyTrigger(StrEnum):
    """Policy evaluation triggers"""
    CREATE = "create"           # Trigger on resource creation
    UPDATE = "update"           # Trigger on resource update
    DELETE = "delete"           # Trigger on resource deletion
    DEPLOY = "deploy"           # Trigger on deployment
    ACCESS = "access"           # Trigger on resource access
    SCHEDULE = "schedule"       # Trigger on schedule
    DRIFT = "drift"             # Trigger on configuration drift


class PolicyEvaluationResult(StrEnum):
    """Policy evaluation results"""
    ALLOW = "allow"             # Allow operation
    DENY = "deny"               # Deny operation
    WARN = "warn"               # Allow with warning
    AUDIT = "audit"             # Audit the operation
    QUARANTINE = "quarantine"   # Quarantine the resource


@dataclass
class PolicyRule:
    """Individual policy rule definition"""
    id: str = field(default_factory=uuid7str)
    name: str = ""
    description: str = ""
    condition: Dict[str, Any] = field(default_factory=dict)
    action: PolicyAction = PolicyAction.WARN
    severity: str = "medium"
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule against context"""
        return self._evaluate_condition(self.condition, context)
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Recursively evaluate condition against context"""
        if not condition:
            return True
        
        operator = condition.get("operator", "and")
        
        if operator == "and":
            conditions = condition.get("conditions", [])
            return all(self._evaluate_condition(c, context) for c in conditions)
        
        elif operator == "or":
            conditions = condition.get("conditions", [])
            return any(self._evaluate_condition(c, context) for c in conditions)
        
        elif operator == "not":
            inner_condition = condition.get("condition", {})
            return not self._evaluate_condition(inner_condition, context)
        
        elif operator in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "regex"]:
            field = condition.get("field")
            value = condition.get("value")
            
            if field not in context:
                return False
            
            context_value = context[field]
            return self._compare_values(context_value, value, operator)
        
        return False
    
    def _compare_values(self, context_value: Any, expected_value: Any, operator: str) -> bool:
        """Compare values using specified operator"""
        try:
            if operator == "eq":
                return context_value == expected_value
            elif operator == "ne":
                return context_value != expected_value
            elif operator == "gt":
                return float(context_value) > float(expected_value)
            elif operator == "gte":
                return float(context_value) >= float(expected_value)
            elif operator == "lt":
                return float(context_value) < float(expected_value)
            elif operator == "lte":
                return float(context_value) <= float(expected_value)
            elif operator == "in":
                return context_value in expected_value
            elif operator == "contains":
                return str(expected_value) in str(context_value)
            elif operator == "regex":
                import re
                return bool(re.search(str(expected_value), str(context_value)))
        except (ValueError, TypeError):
            return False
        
        return False


@dataclass
class PolicyDefinition:
    """Complete policy definition with metadata"""
    id: str = field(default_factory=uuid7str)
    name: str = ""
    description: str = ""
    category: SecurityPolicyCategory = SecurityPolicyCategory.ACCESS_CONTROL
    scope: PolicyScope = PolicyScope.GLOBAL
    triggers: List[PolicyTrigger] = field(default_factory=list)
    rules: List[PolicyRule] = field(default_factory=list)
    
    # Targeting
    tenant_id: Optional[str] = None
    environment_id: Optional[str] = None
    resource_types: List[ResourceType] = field(default_factory=list)
    cloud_providers: List[CloudProvider] = field(default_factory=list)
    user_roles: List[str] = field(default_factory=list)
    
    # Metadata
    version: str = "1.0"
    author: str = "system"
    enabled: bool = True
    priority: int = 50  # Lower number = higher priority
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # Enforcement
    enforcement_mode: str = "enforce"  # enforce, warn, audit
    auto_remediation: bool = False
    notification_channels: List[str] = field(default_factory=list)
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if policy applies to the given context"""
        if not self.enabled:
            return False
        
        # Check tenant scope
        if self.tenant_id and context.get("tenant_id") != self.tenant_id:
            return False
        
        # Check environment scope
        if self.environment_id and context.get("environment_id") != self.environment_id:
            return False
        
        # Check resource type scope
        if self.resource_types and context.get("resource_type") not in [rt.value for rt in self.resource_types]:
            return False
        
        # Check cloud provider scope
        if self.cloud_providers and context.get("cloud_provider") not in [cp.value for cp in self.cloud_providers]:
            return False
        
        # Check user role scope
        if self.user_roles and not any(role in context.get("user_roles", []) for role in self.user_roles):
            return False
        
        # Check triggers
        if self.triggers and context.get("operation") not in [t.value for t in self.triggers]:
            return False
        
        return True
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[PolicyEvaluationResult, List[str], List[PolicyAction]]:
        """Evaluate policy against context and return result"""
        if not self.is_applicable(context):
            return PolicyEvaluationResult.ALLOW, [], []
        
        messages = []
        actions = []
        results = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if rule.evaluate(context):
                results.append(True)
                actions.append(rule.action)
                messages.append(f"Policy rule triggered: {rule.name}")
                
                # Log rule activation
                logger.info(f"Policy rule activated: policy={self.name}, rule={rule.name}, context={context.get('operation', 'unknown')}")
            else:
                results.append(False)
        
        # Determine overall result
        if not results:
            return PolicyEvaluationResult.ALLOW, messages, actions
        
        # If any rule triggered and enforcement is strict
        if any(results):
            if self.enforcement_mode == "enforce":
                if PolicyAction.DENY in actions:
                    return PolicyEvaluationResult.DENY, messages, actions
                elif PolicyAction.QUARANTINE in actions:
                    return PolicyEvaluationResult.QUARANTINE, messages, actions
                else:
                    return PolicyEvaluationResult.WARN, messages, actions
            elif self.enforcement_mode == "warn":
                return PolicyEvaluationResult.WARN, messages, actions
            else:  # audit mode
                return PolicyEvaluationResult.AUDIT, messages, actions
        
        return PolicyEvaluationResult.ALLOW, messages, actions


class ConfigurationPolicyEngine:
    """Advanced policy engine for configuration governance"""
    
    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        self.policies: Dict[str, PolicyDefinition] = {}
        self.policy_cache: Dict[str, Any] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        self._initialized = False
    
    async def initialize(self):
        """Initialize policy engine with default policies"""
        if not self._initialized:
            await self._load_default_policies()
            self._initialized = True
            logger.info(f"Configuration Policy Engine initialized with {len(self.policies)} policies")
    
    async def _load_default_policies(self):
        """Load default configuration governance policies"""
        default_policies = [
            # Production Protection Policy
            PolicyDefinition(
                name="Production Configuration Protection",
                description="Protect production configurations from unauthorized changes",
                category=SecurityPolicyCategory.CHANGE_MANAGEMENT,
                scope=PolicyScope.ENVIRONMENT,
                triggers=[PolicyTrigger.UPDATE, PolicyTrigger.DELETE, PolicyTrigger.DEPLOY],
                rules=[
                    PolicyRule(
                        name="Production Environment Check",
                        description="Require approval for production changes",
                        condition={
                            "operator": "and",
                            "conditions": [
                                {"field": "environment_type", "operator": "eq", "value": "production"},
                                {"field": "operation", "operator": "in", "value": ["update", "delete", "deploy"]}
                            ]
                        },
                        action=PolicyAction.DENY,
                        severity="high"
                    )
                ],
                priority=1,
                enforcement_mode="enforce"
            ),
            
            # Security Configuration Policy
            PolicyDefinition(
                name="Security Configuration Standards",
                description="Enforce security standards for all configurations",
                category=SecurityPolicyCategory.DATA_PROTECTION,
                scope=PolicyScope.GLOBAL,
                triggers=[PolicyTrigger.CREATE, PolicyTrigger.UPDATE],
                rules=[
                    PolicyRule(
                        name="Hardcoded Secrets Detection",
                        description="Detect and prevent hardcoded secrets",
                        condition={
                            "operator": "or",
                            "conditions": [
                                {"field": "configuration_content", "operator": "contains", "value": "password"},
                                {"field": "configuration_content", "operator": "contains", "value": "secret"},
                                {"field": "configuration_content", "operator": "contains", "value": "token"},
                                {"field": "configuration_content", "operator": "regex", "value": r"[A-Za-z0-9]{32,}"}
                            ]
                        },
                        action=PolicyAction.WARN,
                        severity="high"
                    ),
                    PolicyRule(
                        name="Encryption Requirement",
                        description="Require encryption for sensitive resources",
                        condition={
                            "operator": "and",
                            "conditions": [
                                {"field": "resource_type", "operator": "in", "value": ["database", "storage"]},
                                {"field": "configuration_content", "operator": "not", "condition": {"field": "configuration_content", "operator": "contains", "value": "encryption"}}
                            ]
                        },
                        action=PolicyAction.WARN,
                        severity="medium"
                    )
                ],
                priority=10,
                enforcement_mode="warn"
            ),
            
            # Privilege Escalation Prevention
            PolicyDefinition(
                name="Privilege Escalation Prevention",
                description="Prevent configurations that allow privilege escalation",
                category=SecurityPolicyCategory.ACCESS_CONTROL,
                scope=PolicyScope.GLOBAL,
                triggers=[PolicyTrigger.CREATE, PolicyTrigger.UPDATE],
                rules=[
                    PolicyRule(
                        name="Elevated Privileges Detection",
                        description="Detect configurations with elevated privileges",
                        condition={
                            "operator": "or",
                            "conditions": [
                                {"field": "configuration_content", "operator": "contains", "value": "sudo"},
                                {"field": "configuration_content", "operator": "contains", "value": "root"},
                                {"field": "configuration_content", "operator": "contains", "value": "administrator"},
                                {"field": "configuration_content", "operator": "contains", "value": "privileged=true"}
                            ]
                        },
                        action=PolicyAction.DENY,
                        severity="critical"
                    )
                ],
                priority=5,
                enforcement_mode="enforce"
            ),
            
            # Compliance Policy
            PolicyDefinition(
                name="Regulatory Compliance Standards",
                description="Ensure configurations meet regulatory compliance requirements",
                category=SecurityPolicyCategory.COMPLIANCE,
                scope=PolicyScope.GLOBAL,
                triggers=[PolicyTrigger.CREATE, PolicyTrigger.UPDATE, PolicyTrigger.DEPLOY],
                rules=[
                    PolicyRule(
                        name="Data Classification Check",
                        description="Ensure proper data classification",
                        condition={
                            "operator": "and",
                            "conditions": [
                                {"field": "security_level", "operator": "in", "value": ["confidential", "restricted"]},
                                {"field": "data_classification", "operator": "eq", "value": None}
                            ]
                        },
                        action=PolicyAction.WARN,
                        severity="medium"
                    ),
                    PolicyRule(
                        name="Audit Logging Requirement",
                        description="Require audit logging for compliance",
                        condition={
                            "operator": "and",
                            "conditions": [
                                {"field": "resource_type", "operator": "in", "value": ["database", "storage", "virtual_machine"]},
                                {"field": "configuration_content", "operator": "not", "condition": {"field": "configuration_content", "operator": "contains", "value": "audit"}}
                            ]
                        },
                        action=PolicyAction.WARN,
                        severity="medium"
                    )
                ],
                priority=20,
                enforcement_mode="warn"
            )
        ]
        
        for policy in default_policies:
            self.policies[policy.id] = policy
        
        logger.info(f"Loaded {len(default_policies)} default governance policies")
    
    async def add_policy(self, policy: PolicyDefinition) -> str:
        """Add a new policy to the engine"""
        policy.updated_at = datetime.utcnow()
        self.policies[policy.id] = policy
        
        # Clear relevant cache
        self._clear_policy_cache(policy)
        
        logger.info(f"Policy added: {policy.name} (ID: {policy.id})")
        return policy.id
    
    async def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing policy"""
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        
        # Update policy attributes
        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        policy.updated_at = datetime.utcnow()
        
        # Clear relevant cache
        self._clear_policy_cache(policy)
        
        logger.info(f"Policy updated: {policy.name} (ID: {policy_id})")
        return True
    
    async def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy from the engine"""
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        del self.policies[policy_id]
        
        # Clear relevant cache
        self._clear_policy_cache(policy)
        
        logger.info(f"Policy removed: {policy.name} (ID: {policy_id})")
        return True
    
    async def evaluate_policies(
        self,
        context: Dict[str, Any],
        resource: Optional[CMResource] = None
    ) -> Tuple[PolicyEvaluationResult, List[str], List[PolicyAction]]:
        """Evaluate all applicable policies against the given context"""
        assert self._initialized, "Policy engine not initialized"
        
        # Enhance context with resource information
        if resource:
            context.update({
                "resource_id": resource.id,
                "resource_type": resource.resource_type.value,
                "cloud_provider": resource.cloud_provider.value,
                "configuration_content": str(resource.configuration.model_dump()).lower(),
                "security_level": getattr(resource, "security_level", "internal"),
                "environment_id": resource.environment_id
            })
        
        # Find applicable policies
        applicable_policies = [
            policy for policy in self.policies.values() 
            if policy.is_applicable(context)
        ]
        
        # Sort by priority (lower number = higher priority)
        applicable_policies.sort(key=lambda p: p.priority)
        
        # Evaluate policies
        overall_result = PolicyEvaluationResult.ALLOW
        all_messages = []
        all_actions = []
        
        for policy in applicable_policies:
            result, messages, actions = policy.evaluate(context)
            
            all_messages.extend(messages)
            all_actions.extend(actions)
            
            # Determine most restrictive result
            if result == PolicyEvaluationResult.DENY:
                overall_result = PolicyEvaluationResult.DENY
            elif result == PolicyEvaluationResult.QUARANTINE and overall_result == PolicyEvaluationResult.ALLOW:
                overall_result = PolicyEvaluationResult.QUARANTINE
            elif result == PolicyEvaluationResult.WARN and overall_result == PolicyEvaluationResult.ALLOW:
                overall_result = PolicyEvaluationResult.WARN
            elif result == PolicyEvaluationResult.AUDIT and overall_result == PolicyEvaluationResult.ALLOW:
                overall_result = PolicyEvaluationResult.AUDIT
        
        # Record evaluation history
        evaluation_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "applicable_policies": len(applicable_policies),
            "result": overall_result.value,
            "messages": all_messages,
            "actions": [action.value if hasattr(action, 'value') else str(action) for action in all_actions]
        }
        self.evaluation_history.append(evaluation_record)
        
        # Keep history limited
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-500:]
        
        logger.info(f"Policy evaluation completed: result={overall_result.value}, policies={len(applicable_policies)}, messages={len(all_messages)}")
        
        return overall_result, all_messages, all_actions
    
    async def get_policy_violations(
        self,
        resource: CMResource,
        operation: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """Get detailed policy violations for a resource"""
        context = {
            "tenant_id": resource.tenant_id,
            "operation": operation,
            "resource_id": resource.id,
            "resource_type": resource.resource_type.value,
            "cloud_provider": resource.cloud_provider.value,
            "configuration_content": str(resource.configuration.model_dump()).lower(),
            "environment_id": resource.environment_id
        }
        
        violations = []
        
        for policy in self.policies.values():
            if not policy.is_applicable(context):
                continue
            
            result, messages, actions = policy.evaluate(context)
            
            if result in [PolicyEvaluationResult.DENY, PolicyEvaluationResult.QUARANTINE]:
                violations.append({
                    "policy_id": policy.id,
                    "policy_name": policy.name,
                    "category": policy.category.value,
                    "severity": policy.priority,
                    "result": result.value,
                    "messages": messages,
                    "actions": [action.value if hasattr(action, 'value') else str(action) for action in actions],
                    "enforcement_mode": policy.enforcement_mode
                })
        
        return violations
    
    async def generate_compliance_report(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate compliance report based on policy evaluations"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Filter evaluation history by date range and tenant
        relevant_evaluations = [
            eval_record for eval_record in self.evaluation_history
            if start_date <= datetime.fromisoformat(eval_record["timestamp"]) <= end_date
            and (not tenant_id or eval_record["context"].get("tenant_id") == tenant_id)
        ]
        
        # Analyze evaluations
        total_evaluations = len(relevant_evaluations)
        violations = [e for e in relevant_evaluations if e["result"] != "allow"]
        
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "tenant_id": tenant_id,
            "summary": {
                "total_evaluations": total_evaluations,
                "violations": len(violations),
                "compliance_rate": ((total_evaluations - len(violations)) / total_evaluations * 100) if total_evaluations > 0 else 100.0
            },
            "violations_by_category": {},
            "most_triggered_policies": {},
            "enforcement_actions": {}
        }
        
        # Analyze violations by category
        for violation in violations:
            # This would require storing more detailed violation information
            pass
        
        return report
    
    def _clear_policy_cache(self, policy: PolicyDefinition):
        """Clear policy-related cache entries"""
        # Implementation would clear relevant cache entries
        pass
    
    async def get_policy_metrics(self) -> Dict[str, Any]:
        """Get policy engine metrics"""
        return {
            "total_policies": len(self.policies),
            "active_policies": len([p for p in self.policies.values() if p.enabled]),
            "policies_by_category": {
                category.value: len([p for p in self.policies.values() if p.category == category])
                for category in SecurityPolicyCategory
            },
            "evaluation_history_size": len(self.evaluation_history),
            "last_evaluation": self.evaluation_history[-1]["timestamp"] if self.evaluation_history else None
        }


# Global policy engine instance
_policy_engine = None

async def get_policy_engine(tenant_id: Optional[str] = None) -> ConfigurationPolicyEngine:
    """Get global policy engine instance"""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = ConfigurationPolicyEngine(tenant_id)
        await _policy_engine.initialize()
    return _policy_engine

# Export main classes
__all__ = [
    "PolicyScope",
    "PolicyTrigger", 
    "PolicyEvaluationResult",
    "PolicyRule",
    "PolicyDefinition",
    "ConfigurationPolicyEngine",
    "get_policy_engine"
]