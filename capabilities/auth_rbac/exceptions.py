"""
Authentication & RBAC Custom Exceptions

Comprehensive exception classes for authentication, authorization,
and ABAC policy evaluation with detailed error information.
"""

from typing import Dict, Any, Optional


class AuthRBACError(Exception):
	"""Base exception for all Authentication & RBAC errors"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message)
		self.message = message
		self.error_code = error_code or 'AUTH_RBAC_ERROR'
		self.details = details or {}
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'error': self.__class__.__name__,
			'message': self.message,
			'error_code': self.error_code,
			'details': self.details
		}


class AuthenticationError(AuthRBACError):
	"""Errors related to user authentication"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message, error_code or 'AUTHENTICATION_ERROR', details)


class InvalidCredentialsError(AuthenticationError):
	"""Invalid username/password or other credentials"""
	
	def __init__(self, message: str = "Invalid credentials", details: Dict[str, Any] = None):
		super().__init__(message, 'INVALID_CREDENTIALS', details)


class AccountLockedError(AuthenticationError):
	"""Account is locked due to failed login attempts"""
	
	def __init__(self, message: str = "Account is locked", locked_until: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'ACCOUNT_LOCKED', details)
		if locked_until:
			self.details['locked_until'] = locked_until


class AccountInactiveError(AuthenticationError):
	"""Account is inactive or disabled"""
	
	def __init__(self, message: str = "Account is inactive", details: Dict[str, Any] = None):
		super().__init__(message, 'ACCOUNT_INACTIVE', details)


class MFARequiredError(AuthenticationError):
	"""Multi-factor authentication is required"""
	
	def __init__(self, message: str = "Multi-factor authentication required", mfa_methods: list = None, details: Dict[str, Any] = None):
		super().__init__(message, 'MFA_REQUIRED', details)
		if mfa_methods:
			self.details['available_mfa_methods'] = mfa_methods


class MFAInvalidError(AuthenticationError):
	"""Invalid MFA token or code"""
	
	def __init__(self, message: str = "Invalid MFA token", details: Dict[str, Any] = None):
		super().__init__(message, 'MFA_INVALID', details)


class SessionExpiredError(AuthenticationError):
	"""User session has expired"""
	
	def __init__(self, message: str = "Session has expired", details: Dict[str, Any] = None):
		super().__init__(message, 'SESSION_EXPIRED', details)


class SessionInvalidError(AuthenticationError):
	"""Invalid or corrupted session"""
	
	def __init__(self, message: str = "Invalid session", details: Dict[str, Any] = None):
		super().__init__(message, 'SESSION_INVALID', details)


class TokenExpiredError(AuthenticationError):
	"""JWT or other token has expired"""
	
	def __init__(self, message: str = "Token has expired", token_type: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'TOKEN_EXPIRED', details)
		if token_type:
			self.details['token_type'] = token_type


class TokenInvalidError(AuthenticationError):
	"""Invalid or malformed token"""
	
	def __init__(self, message: str = "Invalid token", token_type: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'TOKEN_INVALID', details)
		if token_type:
			self.details['token_type'] = token_type


class AuthorizationError(AuthRBACError):
	"""Errors related to user authorization and permissions"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message, error_code or 'AUTHORIZATION_ERROR', details)


class PermissionDeniedError(AuthorizationError):
	"""User lacks required permissions"""
	
	def __init__(self, message: str = "Permission denied", required_permission: str = None, 
				 user_permissions: list = None, details: Dict[str, Any] = None):
		super().__init__(message, 'PERMISSION_DENIED', details)
		if required_permission:
			self.details['required_permission'] = required_permission
		if user_permissions:
			self.details['user_permissions'] = user_permissions


class RoleRequiredError(AuthorizationError):
	"""User lacks required role"""
	
	def __init__(self, message: str = "Required role not found", required_role: str = None, 
				 user_roles: list = None, details: Dict[str, Any] = None):
		super().__init__(message, 'ROLE_REQUIRED', details)
		if required_role:
			self.details['required_role'] = required_role
		if user_roles:
			self.details['user_roles'] = user_roles


class SecurityLevelInsufficientError(AuthorizationError):
	"""User security level is insufficient"""
	
	def __init__(self, message: str = "Insufficient security level", required_level: str = None, 
				 user_level: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'SECURITY_LEVEL_INSUFFICIENT', details)
		if required_level:
			self.details['required_level'] = required_level
		if user_level:
			self.details['user_level'] = user_level


class ResourceAccessDeniedError(AuthorizationError):
	"""Access denied to specific resource"""
	
	def __init__(self, message: str = "Resource access denied", resource_type: str = None, 
				 resource_id: str = None, action: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'RESOURCE_ACCESS_DENIED', details)
		if resource_type:
			self.details['resource_type'] = resource_type
		if resource_id:
			self.details['resource_id'] = resource_id
		if action:
			self.details['action'] = action


class TenantAccessDeniedError(AuthorizationError):
	"""Access denied due to tenant restrictions"""
	
	def __init__(self, message: str = "Tenant access denied", user_tenant: str = None, 
				 required_tenant: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'TENANT_ACCESS_DENIED', details)
		if user_tenant:
			self.details['user_tenant'] = user_tenant
		if required_tenant:
			self.details['required_tenant'] = required_tenant


class PolicyEvaluationError(AuthRBACError):
	"""Errors related to ABAC policy evaluation"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message, error_code or 'POLICY_EVALUATION_ERROR', details)


class PolicyNotFoundError(PolicyEvaluationError):
	"""Policy not found"""
	
	def __init__(self, message: str = "Policy not found", policy_id: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'POLICY_NOT_FOUND', details)
		if policy_id:
			self.details['policy_id'] = policy_id


class PolicyInvalidError(PolicyEvaluationError):
	"""Policy configuration is invalid"""
	
	def __init__(self, message: str = "Invalid policy configuration", policy_id: str = None, 
				 validation_errors: list = None, details: Dict[str, Any] = None):
		super().__init__(message, 'POLICY_INVALID', details)
		if policy_id:
			self.details['policy_id'] = policy_id
		if validation_errors:
			self.details['validation_errors'] = validation_errors


class AttributeNotFoundError(PolicyEvaluationError):
	"""Required attribute not found"""
	
	def __init__(self, message: str = "Attribute not found", attribute_name: str = None, 
				 attribute_category: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'ATTRIBUTE_NOT_FOUND', details)
		if attribute_name:
			self.details['attribute_name'] = attribute_name
		if attribute_category:
			self.details['attribute_category'] = attribute_category


class AttributeValidationError(PolicyEvaluationError):
	"""Attribute value validation failed"""
	
	def __init__(self, message: str = "Attribute validation failed", attribute_name: str = None, 
				 attribute_value: Any = None, validation_rule: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'ATTRIBUTE_VALIDATION_ERROR', details)
		if attribute_name:
			self.details['attribute_name'] = attribute_name
		if attribute_value is not None:
			self.details['attribute_value'] = attribute_value
		if validation_rule:
			self.details['validation_rule'] = validation_rule


class ConditionEvaluationError(PolicyEvaluationError):
	"""Policy condition evaluation failed"""
	
	def __init__(self, message: str = "Condition evaluation failed", condition_id: str = None, 
				 condition_name: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'CONDITION_EVALUATION_ERROR', details)
		if condition_id:
			self.details['condition_id'] = condition_id
		if condition_name:
			self.details['condition_name'] = condition_name


class RuleEvaluationError(PolicyEvaluationError):
	"""Policy rule evaluation failed"""
	
	def __init__(self, message: str = "Rule evaluation failed", rule_id: str = None, 
				 rule_name: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'RULE_EVALUATION_ERROR', details)
		if rule_id:
			self.details['rule_id'] = rule_id
		if rule_name:
			self.details['rule_name'] = rule_name


class PolicyTimeoutError(PolicyEvaluationError):
	"""Policy evaluation timed out"""
	
	def __init__(self, message: str = "Policy evaluation timeout", timeout_ms: int = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'POLICY_TIMEOUT', details)
		if timeout_ms:
			self.details['timeout_ms'] = timeout_ms


class UserManagementError(AuthRBACError):
	"""Errors related to user management operations"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message, error_code or 'USER_MANAGEMENT_ERROR', details)


class UserNotFoundError(UserManagementError):
	"""User not found"""
	
	def __init__(self, message: str = "User not found", user_id: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'USER_NOT_FOUND', details)
		if user_id:
			self.details['user_id'] = user_id


class UserAlreadyExistsError(UserManagementError):
	"""User already exists"""
	
	def __init__(self, message: str = "User already exists", email: str = None, username: str = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'USER_ALREADY_EXISTS', details)
		if email:
			self.details['email'] = email
		if username:
			self.details['username'] = username


class PasswordPolicyError(UserManagementError):
	"""Password does not meet policy requirements"""
	
	def __init__(self, message: str = "Password policy violation", policy_violations: list = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'PASSWORD_POLICY_ERROR', details)
		if policy_violations:
			self.details['policy_violations'] = policy_violations


class PasswordHistoryError(UserManagementError):
	"""Password was used recently"""
	
	def __init__(self, message: str = "Password recently used", details: Dict[str, Any] = None):
		super().__init__(message, 'PASSWORD_HISTORY_ERROR', details)


class RoleManagementError(AuthRBACError):
	"""Errors related to role management operations"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message, error_code or 'ROLE_MANAGEMENT_ERROR', details)


class RoleNotFoundError(RoleManagementError):
	"""Role not found"""
	
	def __init__(self, message: str = "Role not found", role_id: str = None, role_name: str = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'ROLE_NOT_FOUND', details)
		if role_id:
			self.details['role_id'] = role_id
		if role_name:
			self.details['role_name'] = role_name


class RoleAlreadyExistsError(RoleManagementError):
	"""Role already exists"""
	
	def __init__(self, message: str = "Role already exists", role_name: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'ROLE_ALREADY_EXISTS', details)
		if role_name:
			self.details['role_name'] = role_name


class RoleAssignmentError(RoleManagementError):
	"""Error assigning role to user"""
	
	def __init__(self, message: str = "Role assignment failed", user_id: str = None, role_id: str = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'ROLE_ASSIGNMENT_ERROR', details)
		if user_id:
			self.details['user_id'] = user_id
		if role_id:
			self.details['role_id'] = role_id


class RoleHierarchyError(RoleManagementError):
	"""Error in role hierarchy configuration"""
	
	def __init__(self, message: str = "Role hierarchy error", details: Dict[str, Any] = None):
		super().__init__(message, 'ROLE_HIERARCHY_ERROR', details)


class CircularRoleError(RoleHierarchyError):
	"""Circular reference in role hierarchy"""
	
	def __init__(self, message: str = "Circular role reference detected", role_id: str = None, 
				 parent_role_id: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'CIRCULAR_ROLE_ERROR', details)
		if role_id:
			self.details['role_id'] = role_id
		if parent_role_id:
			self.details['parent_role_id'] = parent_role_id


class PermissionManagementError(AuthRBACError):
	"""Errors related to permission management operations"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message, error_code or 'PERMISSION_MANAGEMENT_ERROR', details)


class PermissionNotFoundError(PermissionManagementError):
	"""Permission not found"""
	
	def __init__(self, message: str = "Permission not found", permission_id: str = None, 
				 resource_type: str = None, action: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'PERMISSION_NOT_FOUND', details)
		if permission_id:
			self.details['permission_id'] = permission_id
		if resource_type:
			self.details['resource_type'] = resource_type
		if action:
			self.details['action'] = action


class PermissionAlreadyExistsError(PermissionManagementError):
	"""Permission already exists"""
	
	def __init__(self, message: str = "Permission already exists", resource_type: str = None, 
				 resource_name: str = None, action: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'PERMISSION_ALREADY_EXISTS', details)
		if resource_type:
			self.details['resource_type'] = resource_type
		if resource_name:
			self.details['resource_name'] = resource_name
		if action:
			self.details['action'] = action


class SessionManagementError(AuthRBACError):
	"""Errors related to session management"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message, error_code or 'SESSION_MANAGEMENT_ERROR', details)


class MaxSessionsExceededError(SessionManagementError):
	"""Maximum concurrent sessions exceeded"""
	
	def __init__(self, message: str = "Maximum sessions exceeded", max_sessions: int = None, 
				 current_sessions: int = None, details: Dict[str, Any] = None):
		super().__init__(message, 'MAX_SESSIONS_EXCEEDED', details)
		if max_sessions:
			self.details['max_sessions'] = max_sessions
		if current_sessions:
			self.details['current_sessions'] = current_sessions


class SuspiciousActivityError(AuthRBACError):
	"""Suspicious activity detected"""
	
	def __init__(self, message: str = "Suspicious activity detected", activity_type: str = None, 
				 risk_score: float = None, details: Dict[str, Any] = None):
		super().__init__(message, 'SUSPICIOUS_ACTIVITY', details)
		if activity_type:
			self.details['activity_type'] = activity_type
		if risk_score:
			self.details['risk_score'] = risk_score


class SecurityPolicyViolationError(AuthRBACError):
	"""Security policy violation"""
	
	def __init__(self, message: str = "Security policy violation", policy_name: str = None, 
				 violation_type: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'SECURITY_POLICY_VIOLATION', details)
		if policy_name:
			self.details['policy_name'] = policy_name
		if violation_type:
			self.details['violation_type'] = violation_type


class ComplianceViolationError(AuthRBACError):
	"""Compliance requirement violation"""
	
	def __init__(self, message: str = "Compliance violation", compliance_framework: str = None, 
				 requirement: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'COMPLIANCE_VIOLATION', details)
		if compliance_framework:
			self.details['compliance_framework'] = compliance_framework
		if requirement:
			self.details['requirement'] = requirement


class GDPRViolationError(ComplianceViolationError):
	"""GDPR compliance violation"""
	
	def __init__(self, message: str = "GDPR violation", gdpr_article: str = None, 
				 data_subject_rights: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'GDPR_VIOLATION', details)
		self.details['compliance_framework'] = 'GDPR'
		if gdpr_article:
			self.details['gdpr_article'] = gdpr_article
		if data_subject_rights:
			self.details['data_subject_rights'] = data_subject_rights


class DataRetentionError(AuthRBACError):
	"""Data retention policy violation"""
	
	def __init__(self, message: str = "Data retention violation", retention_period: int = None, 
				 data_age: int = None, details: Dict[str, Any] = None):
		super().__init__(message, 'DATA_RETENTION_ERROR', details)
		if retention_period:
			self.details['retention_period'] = retention_period
		if data_age:
			self.details['data_age'] = data_age


class EncryptionError(AuthRBACError):
	"""Encryption/decryption errors"""
	
	def __init__(self, message: str = "Encryption error", operation: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'ENCRYPTION_ERROR', details)
		if operation:
			self.details['operation'] = operation


class ConfigurationError(AuthRBACError):
	"""Authentication/authorization configuration errors"""
	
	def __init__(self, message: str = "Configuration error", config_key: str = None, 
				 config_value: Any = None, details: Dict[str, Any] = None):
		super().__init__(message, 'CONFIGURATION_ERROR', details)
		if config_key:
			self.details['config_key'] = config_key
		if config_value is not None:
			self.details['config_value'] = config_value


class IntegrationError(AuthRBACError):
	"""External system integration errors"""
	
	def __init__(self, message: str = "Integration error", system_name: str = None, 
				 operation: str = None, details: Dict[str, Any] = None):
		super().__init__(message, 'INTEGRATION_ERROR', details)
		if system_name:
			self.details['system_name'] = system_name
		if operation:
			self.details['operation'] = operation


class LDAPIntegrationError(IntegrationError):
	"""LDAP/Active Directory integration errors"""
	
	def __init__(self, message: str = "LDAP integration error", ldap_server: str = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'LDAP_INTEGRATION_ERROR', details)
		self.details['system_name'] = 'LDAP'
		if ldap_server:
			self.details['ldap_server'] = ldap_server


class SAMLIntegrationError(IntegrationError):
	"""SAML SSO integration errors"""
	
	def __init__(self, message: str = "SAML integration error", idp_name: str = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'SAML_INTEGRATION_ERROR', details)
		self.details['system_name'] = 'SAML'
		if idp_name:
			self.details['idp_name'] = idp_name


class OAuthIntegrationError(IntegrationError):
	"""OAuth/OpenID Connect integration errors"""
	
	def __init__(self, message: str = "OAuth integration error", provider_name: str = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'OAUTH_INTEGRATION_ERROR', details)
		self.details['system_name'] = 'OAuth'
		if provider_name:
			self.details['provider_name'] = provider_name


class RateLimitExceededError(AuthRBACError):
	"""Rate limit exceeded"""
	
	def __init__(self, message: str = "Rate limit exceeded", limit: int = None, 
				 window_seconds: int = None, retry_after: int = None, details: Dict[str, Any] = None):
		super().__init__(message, 'RATE_LIMIT_EXCEEDED', details)
		if limit:
			self.details['limit'] = limit
		if window_seconds:
			self.details['window_seconds'] = window_seconds
		if retry_after:
			self.details['retry_after'] = retry_after


class CacheError(AuthRBACError):
	"""Cache operation errors"""
	
	def __init__(self, message: str = "Cache error", operation: str = None, cache_key: str = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'CACHE_ERROR', details)
		if operation:
			self.details['operation'] = operation
		if cache_key:
			self.details['cache_key'] = cache_key


class DatabaseError(AuthRBACError):
	"""Database operation errors"""
	
	def __init__(self, message: str = "Database error", operation: str = None, table: str = None, 
				 details: Dict[str, Any] = None):
		super().__init__(message, 'DATABASE_ERROR', details)
		if operation:
			self.details['operation'] = operation
		if table:
			self.details['table'] = table


# Exception mapping for HTTP status codes
EXCEPTION_HTTP_STATUS_MAP = {
	# Authentication errors (401 Unauthorized)
	InvalidCredentialsError: 401,
	MFARequiredError: 401,
	MFAInvalidError: 401,
	SessionExpiredError: 401,
	SessionInvalidError: 401,
	TokenExpiredError: 401,
	TokenInvalidError: 401,
	
	# Authorization errors (403 Forbidden)
	PermissionDeniedError: 403,
	RoleRequiredError: 403,
	SecurityLevelInsufficientError: 403,
	ResourceAccessDeniedError: 403,
	TenantAccessDeniedError: 403,
	
	# Client errors (400 Bad Request)
	PasswordPolicyError: 400,
	PasswordHistoryError: 400,
	AttributeValidationError: 400,
	PolicyInvalidError: 400,
	
	# Conflict errors (409 Conflict)
	UserAlreadyExistsError: 409,
	RoleAlreadyExistsError: 409,
	PermissionAlreadyExistsError: 409,
	CircularRoleError: 409,
	
	# Not found errors (404 Not Found)
	UserNotFoundError: 404,
	RoleNotFoundError: 404,
	PermissionNotFoundError: 404,
	PolicyNotFoundError: 404,
	AttributeNotFoundError: 404,
	
	# Account locked (423 Locked)
	AccountLockedError: 423,
	
	# Too many requests (429 Too Many Requests)
	RateLimitExceededError: 429,
	MaxSessionsExceededError: 429,
	
	# Server errors (500 Internal Server Error)
	PolicyEvaluationError: 500,
	ConditionEvaluationError: 500,
	RuleEvaluationError: 500,
	EncryptionError: 500,
	DatabaseError: 500,
	CacheError: 500,
	
	# Service unavailable (503 Service Unavailable)
	IntegrationError: 503,
	LDAPIntegrationError: 503,
	SAMLIntegrationError: 503,
	OAuthIntegrationError: 503,
	
	# Gateway timeout (504 Gateway Timeout)
	PolicyTimeoutError: 504
}


def get_http_status_for_exception(exception: Exception) -> int:
	"""Get appropriate HTTP status code for exception"""
	return EXCEPTION_HTTP_STATUS_MAP.get(type(exception), 500)


def create_error_response(exception: AuthRBACError, include_details: bool = False) -> Dict[str, Any]:
	"""Create standardized error response from exception"""
	response = {
		'success': False,
		'error': {
			'type': exception.__class__.__name__,
			'message': exception.message,
			'code': exception.error_code
		}
	}
	
	if include_details and exception.details:
		response['error']['details'] = exception.details
	
	return response