"""
Authentication & Role-Based Access Control (RBAC) Capability

Comprehensive enterprise authentication, authorization, and role-based access control
with advanced ABAC (Attribute-Based Access Control) support, multi-tenant architecture,
and GDPR compliance features.
"""

from .models import *
from .abac_models import *
from .abac_service import ABACService, get_abac_service, create_default_attributes
from .exceptions import *

__version__ = "1.0.0"
__capability_code__ = "AUTH_RBAC"
__capability_name__ = "Authentication & Role-Based Access Control"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_authentication",
	"enforces_authorization", 
	"integrates_with_auth",
	"security_audited",
	"role_based_access",
	"multi_tenant_secure",
	"abac_policy_enabled",
	"gdpr_compliant_auth",
	"session_managed",
	"permission_controlled"
]

# Primary capability interfaces for other capabilities to use
__primary_interfaces__ = [
	"ARUser",
	"ARRole", 
	"ARPermission",
	"ARUserSession",
	"ABACService",
	"authorize",
	"authenticate",
	"check_permission",
	"get_user_roles",
	"create_policy",
	"evaluate_policy"
]

# Event types emitted by this capability
__event_types__ = [
	"user.login",
	"user.logout", 
	"user.mfa_enabled",
	"user.password_changed",
	"user.account_locked",
	"user.account_unlocked",
	"role.assigned",
	"role.revoked",
	"permission.granted",
	"permission.denied",
	"policy.evaluated",
	"session.created",
	"session.expired",
	"security.suspicious_activity",
	"compliance.violation"
]

# Configuration requirements
__configuration_schema__ = {
	"authentication": {
		"password_policy": {
			"min_length": {"type": "integer", "default": 8},
			"require_uppercase": {"type": "boolean", "default": True},
			"require_lowercase": {"type": "boolean", "default": True},
			"require_numbers": {"type": "boolean", "default": True},
			"require_symbols": {"type": "boolean", "default": False},
			"history_count": {"type": "integer", "default": 12}
		},
		"session_management": {
			"session_timeout_minutes": {"type": "integer", "default": 60},
			"max_concurrent_sessions": {"type": "integer", "default": 5},
			"remember_me_days": {"type": "integer", "default": 30}
		},
		"mfa": {
			"require_for_admin": {"type": "boolean", "default": True},
			"require_for_high_security": {"type": "boolean", "default": True},
			"totp_issuer": {"type": "string", "default": "APG Enterprise"}
		}
	},
	"authorization": {
		"rbac": {
			"enable_role_hierarchy": {"type": "boolean", "default": True},
			"max_role_depth": {"type": "integer", "default": 10},
			"default_user_role": {"type": "string", "default": "user"}
		},
		"abac": {
			"enable_abac": {"type": "boolean", "default": True},
			"policy_cache_ttl": {"type": "integer", "default": 300},
			"max_evaluation_time_ms": {"type": "integer", "default": 5000}
		}
	},
	"security": {
		"account_lockout": {
			"max_failed_attempts": {"type": "integer", "default": 5},
			"lockout_duration_minutes": {"type": "integer", "default": 30},
			"progressive_lockout": {"type": "boolean", "default": True}
		},
		"rate_limiting": {
			"login_attempts_per_minute": {"type": "integer", "default": 10},
			"api_requests_per_hour": {"type": "integer", "default": 1000}
		}
	},
	"compliance": {
		"gdpr": {
			"data_retention_days": {"type": "integer", "default": 2555},
			"consent_required": {"type": "boolean", "default": True},
			"right_to_erasure": {"type": "boolean", "default": True}
		}
	}
}

# Dependencies on other capabilities
__capability_dependencies__ = [
	{
		"capability": "profile_management",
		"version": ">=1.0.0",
		"required": True,
		"integration_points": ["user_profiles", "consent_management"]
	},
	{
		"capability": "audit_logging", 
		"version": ">=1.0.0",
		"required": True,
		"integration_points": ["security_events", "access_logs"]
	},
	{
		"capability": "notification_engine",
		"version": ">=1.0.0", 
		"required": False,
		"integration_points": ["security_alerts", "password_reset"]
	}
]

# Export main interfaces
__all__ = [
	# Core Models
	"ARUser",
	"ARRole",
	"ARPermission", 
	"ARUserRole",
	"ARUserPermission",
	"ARRolePermission",
	"ARUserSession",
	"ARSessionActivity",
	"ARLoginAttempt",
	
	# ABAC Models
	"ARAttribute",
	"ARPolicy",
	"ARPolicyRule", 
	"ARPolicyCondition",
	"ARPolicySet",
	"ARPolicySetMapping",
	"ARAccessRequest",
	
	# Services
	"ABACService",
	"get_abac_service",
	"create_default_attributes",
	
	# Data Structures
	"AccessRequestContext",
	"PolicyDecision",
	"AuthorizationDecision",
	"DecisionType",
	"PolicyEffect",
	"PolicyAlgorithm",
	"AttributeType",
	
	# Exceptions
	"AuthRBACError",
	"AuthenticationError",
	"AuthorizationError",
	"PolicyEvaluationError",
	"InvalidCredentialsError",
	"AccountLockedError",
	"MFARequiredError",
	"PermissionDeniedError",
	"RoleRequiredError",
	"SessionExpiredError",
	"UserNotFoundError",
	"get_http_status_for_exception",
	"create_error_response"
]