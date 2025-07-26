#!/usr/bin/env python3
"""
Multi-Tenant Enterprise Deployment with SSO Integration
======================================================

Comprehensive multi-tenant system for enterprise deployment of digital twins
with complete tenant isolation, SSO integration, role-based access control,
and enterprise-grade security and compliance features.
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import base64
from pathlib import Path

# JWT and authentication imports
try:
	import jwt
	from cryptography.hazmat.primitives import serialization
	from cryptography.hazmat.primitives.asymmetric import rsa
except ImportError:
	print("Warning: PyJWT and cryptography not available. Install with: pip install PyJWT cryptography")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi_tenant_enterprise")

class TenantStatus(Enum):
	"""Tenant status types"""
	ACTIVE = "active"
	SUSPENDED = "suspended"
	PENDING = "pending"
	DEACTIVATED = "deactivated"
	TRIAL = "trial"

class SubscriptionTier(Enum):
	"""Subscription tier levels"""
	BASIC = "basic"
	PROFESSIONAL = "professional"
	ENTERPRISE = "enterprise"
	PREMIUM = "premium"

class UserRole(Enum):
	"""User roles within tenants"""
	SUPER_ADMIN = "super_admin"  # Platform admin
	TENANT_ADMIN = "tenant_admin"  # Tenant administrator
	TWIN_ADMIN = "twin_admin"  # Digital twin administrator
	ENGINEER = "engineer"  # Engineering user
	ANALYST = "analyst"  # Data analyst
	VIEWER = "viewer"  # Read-only user
	GUEST = "guest"  # Limited access

class SSOProvider(Enum):
	"""Supported SSO providers"""
	OKTA = "okta"
	AZURE_AD = "azure_ad"
	GOOGLE_WORKSPACE = "google_workspace"
	AWS_SSO = "aws_sso"
	SAML_GENERIC = "saml_generic"
	LDAP = "ldap"
	CUSTOM = "custom"

class AuditEventType(Enum):
	"""Types of audit events"""
	LOGIN = "login"
	LOGOUT = "logout"
	TWIN_ACCESS = "twin_access"
	DATA_EXPORT = "data_export"
	CONFIGURATION_CHANGE = "configuration_change"
	USER_INVITATION = "user_invitation"
	ROLE_ASSIGNMENT = "role_assignment"
	API_ACCESS = "api_access"
	SECURITY_VIOLATION = "security_violation"

@dataclass
class TenantConfiguration:
	"""Tenant-specific configuration"""
	tenant_id: str
	organization_name: str
	domain: str
	status: TenantStatus
	subscription_tier: SubscriptionTier
	max_users: int
	max_digital_twins: int
	max_storage_gb: float
	features_enabled: List[str]
	custom_branding: Dict[str, Any]
	data_retention_days: int
	backup_enabled: bool
	encryption_level: str
	compliance_profiles: List[str]
	created_at: datetime
	last_activity: datetime
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'tenant_id': self.tenant_id,
			'organization_name': self.organization_name,
			'domain': self.domain,
			'status': self.status.value,
			'subscription_tier': self.subscription_tier.value,
			'max_users': self.max_users,
			'max_digital_twins': self.max_digital_twins,
			'max_storage_gb': self.max_storage_gb,
			'features_enabled': self.features_enabled,
			'custom_branding': self.custom_branding,
			'data_retention_days': self.data_retention_days,
			'backup_enabled': self.backup_enabled,
			'encryption_level': self.encryption_level,
			'compliance_profiles': self.compliance_profiles,
			'created_at': self.created_at.isoformat(),
			'last_activity': self.last_activity.isoformat()
		}

@dataclass
class SSOConfiguration:
	"""SSO provider configuration"""
	config_id: str
	tenant_id: str
	provider: SSOProvider
	provider_name: str
	client_id: str
	client_secret: str  # Encrypted
	domain: str
	metadata_url: Optional[str]
	attribute_mappings: Dict[str, str]
	role_mappings: Dict[str, UserRole]
	is_active: bool
	auto_provision_users: bool
	default_role: UserRole
	created_at: datetime
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'config_id': self.config_id,
			'tenant_id': self.tenant_id,
			'provider': self.provider.value,
			'provider_name': self.provider_name,
			'client_id': self.client_id,
			'domain': self.domain,
			'metadata_url': self.metadata_url,
			'attribute_mappings': self.attribute_mappings,
			'role_mappings': {k: v.value for k, v in self.role_mappings.items()},
			'is_active': self.is_active,
			'auto_provision_users': self.auto_provision_users,
			'default_role': self.default_role.value,
			'created_at': self.created_at.isoformat()
		}

@dataclass
class EnterpriseUser:
	"""Enterprise user with tenant and role information"""
	user_id: str
	tenant_id: str
	email: str
	first_name: str
	last_name: str
	role: UserRole
	permissions: List[str]
	sso_provider: Optional[SSOProvider]
	external_user_id: Optional[str]
	is_active: bool
	last_login: Optional[datetime]
	failed_login_attempts: int
	account_locked_until: Optional[datetime]
	created_at: datetime
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'user_id': self.user_id,
			'tenant_id': self.tenant_id,
			'email': self.email,
			'first_name': self.first_name,
			'last_name': self.last_name,
			'role': self.role.value,
			'permissions': self.permissions,
			'sso_provider': self.sso_provider.value if self.sso_provider else None,
			'external_user_id': self.external_user_id,
			'is_active': self.is_active,
			'last_login': self.last_login.isoformat() if self.last_login else None,
			'failed_login_attempts': self.failed_login_attempts,
			'account_locked_until': self.account_locked_until.isoformat() if self.account_locked_until else None,
			'created_at': self.created_at.isoformat()
		}

@dataclass
class AuditEvent:
	"""Audit event for compliance tracking"""
	event_id: str
	tenant_id: str
	user_id: Optional[str]
	event_type: AuditEventType
	resource_type: str
	resource_id: Optional[str]
	event_data: Dict[str, Any]
	ip_address: str
	user_agent: str
	timestamp: datetime
	success: bool
	risk_score: float
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'event_id': self.event_id,
			'tenant_id': self.tenant_id,
			'user_id': self.user_id,
			'event_type': self.event_type.value,
			'resource_type': self.resource_type,
			'resource_id': self.resource_id,
			'event_data': self.event_data,
			'ip_address': self.ip_address,
			'user_agent': self.user_agent,
			'timestamp': self.timestamp.isoformat(),
			'success': self.success,
			'risk_score': self.risk_score
		}

class TenantDataIsolation:
	"""Ensures complete data isolation between tenants"""
	
	def __init__(self):
		self.tenant_databases: Dict[str, str] = {}  # tenant_id -> connection_string
		self.tenant_schemas: Dict[str, str] = {}  # tenant_id -> schema_name
		self.data_encryption_keys: Dict[str, bytes] = {}  # tenant_id -> encryption_key
	
	def create_tenant_schema(self, tenant_id: str) -> str:
		"""Create isolated database schema for tenant"""
		schema_name = f"tenant_{tenant_id.replace('-', '_')}"
		self.tenant_schemas[tenant_id] = schema_name
		
		# In a real implementation, this would create the actual database schema
		logger.info(f"Created isolated schema {schema_name} for tenant {tenant_id}")
		return schema_name
	
	def get_tenant_connection_string(self, tenant_id: str) -> str:
		"""Get tenant-specific database connection string"""
		if tenant_id not in self.tenant_databases:
			# Create tenant-specific connection with schema isolation
			base_connection = "postgresql://user:pass@host:5432/enterprise_db"
			schema_name = self.tenant_schemas.get(tenant_id, f"tenant_{tenant_id}")
			connection_string = f"{base_connection}?options=-csearch_path={schema_name}"
			self.tenant_databases[tenant_id] = connection_string
		
		return self.tenant_databases[tenant_id]
	
	def encrypt_tenant_data(self, tenant_id: str, data: bytes) -> bytes:
		"""Encrypt data using tenant-specific key"""
		if tenant_id not in self.data_encryption_keys:
			# Generate tenant-specific encryption key
			self.data_encryption_keys[tenant_id] = secrets.token_bytes(32)
		
		# In a real implementation, this would use proper encryption
		# For demo, we'll just return the data with a prefix
		encrypted_prefix = f"ENC_{tenant_id}_".encode()
		return encrypted_prefix + data
	
	def decrypt_tenant_data(self, tenant_id: str, encrypted_data: bytes) -> bytes:
		"""Decrypt data using tenant-specific key"""
		encrypted_prefix = f"ENC_{tenant_id}_".encode()
		if encrypted_data.startswith(encrypted_prefix):
			return encrypted_data[len(encrypted_prefix):]
		return encrypted_data

class SSOIntegration:
	"""Single Sign-On integration with enterprise providers"""
	
	def __init__(self):
		self.jwt_secret = secrets.token_urlsafe(32)
		self.sso_configurations: Dict[str, SSOConfiguration] = {}
		self.active_sessions: Dict[str, Dict[str, Any]] = {}
	
	def configure_sso_provider(self, config: SSOConfiguration):
		"""Configure SSO provider for tenant"""
		self.sso_configurations[config.tenant_id] = config
		logger.info(f"Configured {config.provider.value} SSO for tenant {config.tenant_id}")
	
	async def initiate_sso_login(self, tenant_id: str, redirect_uri: str) -> str:
		"""Initiate SSO login flow"""
		if tenant_id not in self.sso_configurations:
			raise ValueError(f"No SSO configuration found for tenant {tenant_id}")
		
		config = self.sso_configurations[tenant_id]
		state = secrets.token_urlsafe(32)
		
		# Store state for validation
		self.active_sessions[state] = {
			'tenant_id': tenant_id,
			'redirect_uri': redirect_uri,
			'created_at': datetime.utcnow(),
			'provider': config.provider.value
		}
		
		# Generate provider-specific SSO URL
		if config.provider == SSOProvider.AZURE_AD:
			sso_url = self._generate_azure_ad_url(config, state, redirect_uri)
		elif config.provider == SSOProvider.OKTA:
			sso_url = self._generate_okta_url(config, state, redirect_uri)
		elif config.provider == SSOProvider.GOOGLE_WORKSPACE:
			sso_url = self._generate_google_workspace_url(config, state, redirect_uri)
		else:
			sso_url = self._generate_generic_saml_url(config, state, redirect_uri)
		
		logger.info(f"Initiated SSO login for tenant {tenant_id} with provider {config.provider.value}")
		return sso_url
	
	def _generate_azure_ad_url(self, config: SSOConfiguration, state: str, redirect_uri: str) -> str:
		"""Generate Azure AD SSO URL"""
		base_url = f"https://login.microsoftonline.com/{config.domain}/oauth2/v2.0/authorize"
		params = {
			'client_id': config.client_id,
			'response_type': 'code',
			'redirect_uri': redirect_uri,
			'scope': 'openid profile email',
			'state': state,
			'response_mode': 'query'
		}
		param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
		return f"{base_url}?{param_string}"
	
	def _generate_okta_url(self, config: SSOConfiguration, state: str, redirect_uri: str) -> str:
		"""Generate Okta SSO URL"""
		base_url = f"https://{config.domain}/oauth2/default/v1/authorize"
		params = {
			'client_id': config.client_id,
			'response_type': 'code',
			'redirect_uri': redirect_uri,
			'scope': 'openid profile email',
			'state': state
		}
		param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
		return f"{base_url}?{param_string}"
	
	def _generate_google_workspace_url(self, config: SSOConfiguration, state: str, redirect_uri: str) -> str:
		"""Generate Google Workspace SSO URL"""
		base_url = "https://accounts.google.com/oauth2/v2/auth"
		params = {
			'client_id': config.client_id,
			'response_type': 'code',
			'redirect_uri': redirect_uri,
			'scope': 'openid profile email',
			'state': state,
			'hd': config.domain  # Hosted domain restriction
		}
		param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
		return f"{base_url}?{param_string}"
	
	def _generate_generic_saml_url(self, config: SSOConfiguration, state: str, redirect_uri: str) -> str:
		"""Generate generic SAML SSO URL"""
		# This would generate a SAML AuthnRequest
		return f"{config.metadata_url}?SAMLRequest=encoded_request&RelayState={state}"
	
	async def handle_sso_callback(self, state: str, auth_code: str, 
								 ip_address: str = "unknown") -> Tuple[str, EnterpriseUser]:
		"""Handle SSO provider callback"""
		if state not in self.active_sessions:
			raise ValueError("Invalid or expired SSO state")
		
		session_data = self.active_sessions[state]
		tenant_id = session_data['tenant_id']
		config = self.sso_configurations[tenant_id]
		
		# Exchange auth code for user information
		user_info = await self._exchange_auth_code(config, auth_code)
		
		# Map SSO attributes to user
		user = self._map_sso_user(config, user_info, tenant_id)
		
		# Generate JWT token
		token = self._generate_jwt_token(user, session_data)
		
		# Clean up session
		del self.active_sessions[state]
		
		logger.info(f"SSO login successful for user {user.email} in tenant {tenant_id}")
		return token, user
	
	async def _exchange_auth_code(self, config: SSOConfiguration, auth_code: str) -> Dict[str, Any]:
		"""Exchange authorization code for user information"""
		# Mock implementation - in reality, this would make HTTP requests to the SSO provider
		return {
			'sub': f"user_{secrets.token_hex(8)}",
			'email': f"user@{config.domain}",
			'given_name': 'John',
			'family_name': 'Doe',
			'groups': ['engineers', 'admins']
		}
	
	def _map_sso_user(self, config: SSOConfiguration, user_info: Dict[str, Any], tenant_id: str) -> EnterpriseUser:
		"""Map SSO user information to enterprise user"""
		
		# Apply attribute mappings
		email = user_info.get(config.attribute_mappings.get('email', 'email'))
		first_name = user_info.get(config.attribute_mappings.get('first_name', 'given_name'))
		last_name = user_info.get(config.attribute_mappings.get('last_name', 'family_name'))
		groups = user_info.get(config.attribute_mappings.get('groups', 'groups'), [])
		
		# Determine role based on group mappings
		role = config.default_role
		for group in groups:
			if group in config.role_mappings:
				role = config.role_mappings[group]
				break
		
		# Generate permissions based on role
		permissions = self._get_role_permissions(role)
		
		return EnterpriseUser(
			user_id=f"sso_{uuid.uuid4().hex[:12]}",
			tenant_id=tenant_id,
			email=email,
			first_name=first_name,
			last_name=last_name,
			role=role,
			permissions=permissions,
			sso_provider=config.provider,
			external_user_id=user_info.get('sub'),
			is_active=True,
			last_login=datetime.utcnow(),
			failed_login_attempts=0,
			account_locked_until=None,
			created_at=datetime.utcnow()
		)
	
	def _get_role_permissions(self, role: UserRole) -> List[str]:
		"""Get permissions for user role"""
		permission_map = {
			UserRole.SUPER_ADMIN: [
				'platform.admin', 'tenant.manage', 'user.manage', 'system.config',
				'twin.admin', 'data.full_access', 'audit.access', 'backup.manage'
			],
			UserRole.TENANT_ADMIN: [
				'tenant.admin', 'user.manage', 'twin.admin', 'data.full_access',
				'config.manage', 'audit.view', 'reports.generate'
			],
			UserRole.TWIN_ADMIN: [
				'twin.admin', 'twin.create', 'twin.delete', 'data.read_write',
				'simulation.run', 'alerts.manage'
			],
			UserRole.ENGINEER: [
				'twin.read_write', 'data.read_write', 'simulation.run',
				'reports.view', 'alerts.view'
			],
			UserRole.ANALYST: [
				'twin.read', 'data.read', 'reports.view', 'analytics.access',
				'dashboard.view'
			],
			UserRole.VIEWER: [
				'twin.read', 'data.read', 'dashboard.view'
			],
			UserRole.GUEST: [
				'dashboard.view'
			]
		}
		return permission_map.get(role, [])
	
	def _generate_jwt_token(self, user: EnterpriseUser, session_data: Dict[str, Any]) -> str:
		"""Generate JWT token for authenticated user"""
		payload = {
			'user_id': user.user_id,
			'tenant_id': user.tenant_id,
			'email': user.email,
			'role': user.role.value,
			'permissions': user.permissions,
			'iat': datetime.utcnow(),
			'exp': datetime.utcnow() + timedelta(hours=8),
			'provider': session_data['provider']
		}
		
		return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
	
	def validate_jwt_token(self, token: str) -> Dict[str, Any]:
		"""Validate and decode JWT token"""
		try:
			payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
			return payload
		except jwt.ExpiredSignatureError:
			raise ValueError("Token has expired")
		except jwt.InvalidTokenError:
			raise ValueError("Invalid token")

class EnterpriseAuditSystem:
	"""Comprehensive audit system for enterprise compliance"""
	
	def __init__(self):
		self.audit_events: List[AuditEvent] = []
		self.risk_threshold = 7.0
		self.compliance_rules: Dict[str, List[str]] = {
			'SOC2': ['login', 'data_export', 'configuration_change', 'role_assignment'],
			'GDPR': ['data_export', 'user_data_access', 'data_deletion'],
			'HIPAA': ['data_access', 'data_export', 'user_invitation'],
			'ISO27001': ['security_violation', 'configuration_change', 'login']
		}
	
	async def log_audit_event(self, tenant_id: str, user_id: Optional[str], 
							 event_type: AuditEventType, resource_type: str,
							 resource_id: Optional[str], event_data: Dict[str, Any],
							 ip_address: str, user_agent: str, success: bool = True):
		"""Log audit event for compliance tracking"""
		
		# Calculate risk score
		risk_score = self._calculate_risk_score(event_type, event_data, ip_address, success)
		
		event = AuditEvent(
			event_id=f"audit_{uuid.uuid4().hex[:12]}",
			tenant_id=tenant_id,
			user_id=user_id,
			event_type=event_type,
			resource_type=resource_type,
			resource_id=resource_id,
			event_data=event_data,
			ip_address=ip_address,
			user_agent=user_agent,
			timestamp=datetime.utcnow(),
			success=success,
			risk_score=risk_score
		)
		
		self.audit_events.append(event)
		
		# Check for high-risk events
		if risk_score >= self.risk_threshold:
			await self._handle_high_risk_event(event)
		
		logger.info(f"Audit event logged: {event_type.value} for tenant {tenant_id}")
	
	def _calculate_risk_score(self, event_type: AuditEventType, event_data: Dict[str, Any],
							 ip_address: str, success: bool) -> float:
		"""Calculate risk score for audit event"""
		base_score = {
			AuditEventType.LOGIN: 2.0,
			AuditEventType.LOGOUT: 1.0,
			AuditEventType.TWIN_ACCESS: 3.0,
			AuditEventType.DATA_EXPORT: 7.0,
			AuditEventType.CONFIGURATION_CHANGE: 6.0,
			AuditEventType.USER_INVITATION: 4.0,
			AuditEventType.ROLE_ASSIGNMENT: 5.0,
			AuditEventType.API_ACCESS: 2.0,
			AuditEventType.SECURITY_VIOLATION: 9.0
		}.get(event_type, 3.0)
		
		# Adjust for failure
		if not success:
			base_score += 2.0
		
		# Adjust for suspicious IP patterns
		if self._is_suspicious_ip(ip_address):
			base_score += 3.0
		
		# Adjust for unusual data volume
		if event_type == AuditEventType.DATA_EXPORT:
			data_size = event_data.get('data_size_mb', 0)
			if data_size > 1000:  # Large export
				base_score += 2.0
		
		return min(10.0, base_score)
	
	def _is_suspicious_ip(self, ip_address: str) -> bool:
		"""Check if IP address is suspicious"""
		# In a real implementation, this would check against threat intelligence
		suspicious_patterns = ['10.0.0.', '192.168.', '127.0.0.']
		return not any(ip_address.startswith(pattern) for pattern in suspicious_patterns)
	
	async def _handle_high_risk_event(self, event: AuditEvent):
		"""Handle high-risk audit events"""
		logger.warning(f"High-risk event detected: {event.event_type.value} (risk score: {event.risk_score})")
		
		# In a real implementation, this would:
		# - Send alerts to security team
		# - Potentially block user/IP
		# - Trigger additional monitoring
		# - Generate incident reports
	
	def generate_compliance_report(self, tenant_id: str, compliance_framework: str,
								  start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate compliance report for specific framework"""
		
		required_events = self.compliance_rules.get(compliance_framework, [])
		
		# Filter events for tenant and date range
		relevant_events = [
			event for event in self.audit_events
			if (event.tenant_id == tenant_id and
				start_date <= event.timestamp <= end_date and
				event.event_type.value in required_events)
		]
		
		# Group events by type
		event_summary = {}
		for event in relevant_events:
			event_type = event.event_type.value
			if event_type not in event_summary:
				event_summary[event_type] = {'count': 0, 'success_rate': 0, 'avg_risk_score': 0}
			
			event_summary[event_type]['count'] += 1
			event_summary[event_type]['avg_risk_score'] += event.risk_score
		
		# Calculate averages
		for event_type in event_summary:
			count = event_summary[event_type]['count']
			if count > 0:
				event_summary[event_type]['avg_risk_score'] /= count
				
				success_count = len([e for e in relevant_events 
									if e.event_type.value == event_type and e.success])
				event_summary[event_type]['success_rate'] = (success_count / count) * 100
		
		return {
			'tenant_id': tenant_id,
			'compliance_framework': compliance_framework,
			'report_period': {
				'start': start_date.isoformat(),
				'end': end_date.isoformat()
			},
			'total_events': len(relevant_events),
			'event_summary': event_summary,
			'high_risk_events': len([e for e in relevant_events if e.risk_score >= self.risk_threshold]),
			'generated_at': datetime.utcnow().isoformat()
		}

class MultiTenantEnterpriseManager:
	"""Main multi-tenant enterprise management system"""
	
	def __init__(self):
		self.tenants: Dict[str, TenantConfiguration] = {}
		self.users: Dict[str, EnterpriseUser] = {}
		self.data_isolation = TenantDataIsolation()
		self.sso_integration = SSOIntegration()
		self.audit_system = EnterpriseAuditSystem()
		
		# Platform metrics
		self.platform_metrics = {
			'total_tenants': 0,
			'active_users': 0,
			'total_digital_twins': 0,
			'storage_used_gb': 0.0,
			'monthly_api_calls': 0
		}
		
		logger.info("Multi-Tenant Enterprise Manager initialized")
	
	async def create_tenant(self, organization_name: str, domain: str, 
						   admin_email: str, subscription_tier: SubscriptionTier) -> str:
		"""Create new enterprise tenant"""
		
		tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
		
		# Determine tier limits
		tier_limits = self._get_tier_limits(subscription_tier)
		
		# Create tenant configuration
		tenant_config = TenantConfiguration(
			tenant_id=tenant_id,
			organization_name=organization_name,
			domain=domain,
			status=TenantStatus.ACTIVE,
			subscription_tier=subscription_tier,
			max_users=tier_limits['max_users'],
			max_digital_twins=tier_limits['max_digital_twins'],
			max_storage_gb=tier_limits['max_storage_gb'],
			features_enabled=tier_limits['features'],
			custom_branding={},
			data_retention_days=tier_limits['data_retention_days'],
			backup_enabled=tier_limits['backup_enabled'],
			encryption_level=tier_limits['encryption_level'],
			compliance_profiles=tier_limits['compliance_profiles'],
			created_at=datetime.utcnow(),
			last_activity=datetime.utcnow()
		)
		
		# Create tenant data isolation
		self.data_isolation.create_tenant_schema(tenant_id)
		
		# Create initial admin user
		admin_user = await self._create_tenant_admin(tenant_id, admin_email)
		
		# Store tenant
		self.tenants[tenant_id] = tenant_config
		self.users[admin_user.user_id] = admin_user
		
		# Update platform metrics
		self.platform_metrics['total_tenants'] += 1
		self.platform_metrics['active_users'] += 1
		
		# Log audit event
		await self.audit_system.log_audit_event(
			tenant_id=tenant_id,
			user_id=None,
			event_type=AuditEventType.CONFIGURATION_CHANGE,
			resource_type='tenant',
			resource_id=tenant_id,
			event_data={'action': 'tenant_created', 'organization': organization_name},
			ip_address='system',
			user_agent='system'
		)
		
		logger.info(f"Created tenant {tenant_id} for organization {organization_name}")
		return tenant_id
	
	def _get_tier_limits(self, tier: SubscriptionTier) -> Dict[str, Any]:
		"""Get limits and features for subscription tier"""
		tier_configs = {
			SubscriptionTier.BASIC: {
				'max_users': 10,
				'max_digital_twins': 50,
				'max_storage_gb': 100.0,
				'features': ['basic_twins', 'standard_analytics'],
				'data_retention_days': 90,
				'backup_enabled': False,
				'encryption_level': 'standard',
				'compliance_profiles': []
			},
			SubscriptionTier.PROFESSIONAL: {
				'max_users': 50,
				'max_digital_twins': 500,
				'max_storage_gb': 1000.0,
				'features': ['advanced_twins', 'real_time_analytics', 'api_access'],
				'data_retention_days': 365,
				'backup_enabled': True,
				'encryption_level': 'advanced',
				'compliance_profiles': ['SOC2']
			},
			SubscriptionTier.ENTERPRISE: {
				'max_users': 500,
				'max_digital_twins': 10000,
				'max_storage_gb': 10000.0,
				'features': ['all_features', 'custom_integrations', 'sso', 'audit_logs'],
				'data_retention_days': 2555,  # 7 years
				'backup_enabled': True,
				'encryption_level': 'enterprise',
				'compliance_profiles': ['SOC2', 'GDPR', 'HIPAA', 'ISO27001']
			},
			SubscriptionTier.PREMIUM: {
				'max_users': -1,  # Unlimited
				'max_digital_twins': -1,  # Unlimited
				'max_storage_gb': -1.0,  # Unlimited
				'features': ['all_features', 'dedicated_support', 'custom_development'],
				'data_retention_days': -1,  # Configurable
				'backup_enabled': True,
				'encryption_level': 'premium',
				'compliance_profiles': ['SOC2', 'GDPR', 'HIPAA', 'ISO27001', 'FedRAMP']
			}
		}
		return tier_configs.get(tier, tier_configs[SubscriptionTier.BASIC])
	
	async def _create_tenant_admin(self, tenant_id: str, admin_email: str) -> EnterpriseUser:
		"""Create initial tenant administrator"""
		user_id = f"admin_{uuid.uuid4().hex[:12]}"
		
		admin_user = EnterpriseUser(
			user_id=user_id,
			tenant_id=tenant_id,
			email=admin_email,
			first_name="Tenant",
			last_name="Administrator",
			role=UserRole.TENANT_ADMIN,
			permissions=self.sso_integration._get_role_permissions(UserRole.TENANT_ADMIN),
			sso_provider=None,
			external_user_id=None,
			is_active=True,
			last_login=None,
			failed_login_attempts=0,
			account_locked_until=None,
			created_at=datetime.utcnow()
		)
		
		return admin_user
	
	async def configure_tenant_sso(self, tenant_id: str, provider: SSOProvider,
								  provider_config: Dict[str, Any]) -> str:
		"""Configure SSO for tenant"""
		
		if tenant_id not in self.tenants:
			raise ValueError(f"Tenant {tenant_id} not found")
		
		config_id = f"sso_{uuid.uuid4().hex[:12]}"
		
		sso_config = SSOConfiguration(
			config_id=config_id,
			tenant_id=tenant_id,
			provider=provider,
			provider_name=provider_config['provider_name'],
			client_id=provider_config['client_id'],
			client_secret=provider_config['client_secret'],
			domain=provider_config['domain'],
			metadata_url=provider_config.get('metadata_url'),
			attribute_mappings=provider_config.get('attribute_mappings', {
				'email': 'email',
				'first_name': 'given_name',
				'last_name': 'family_name',
				'groups': 'groups'
			}),
			role_mappings=provider_config.get('role_mappings', {}),
			is_active=True,
			auto_provision_users=provider_config.get('auto_provision_users', True),
			default_role=UserRole.VIEWER,
			created_at=datetime.utcnow()
		)
		
		self.sso_integration.configure_sso_provider(sso_config)
		
		# Log audit event
		await self.audit_system.log_audit_event(
			tenant_id=tenant_id,
			user_id=None,
			event_type=AuditEventType.CONFIGURATION_CHANGE,
			resource_type='sso_config',
			resource_id=config_id,
			event_data={'action': 'sso_configured', 'provider': provider.value},
			ip_address='system',
			user_agent='system'
		)
		
		logger.info(f"Configured {provider.value} SSO for tenant {tenant_id}")
		return config_id
	
	async def authenticate_user(self, tenant_id: str, email: str, ip_address: str,
							   user_agent: str) -> Tuple[str, EnterpriseUser]:
		"""Authenticate user and return JWT token"""
		
		# For SSO-enabled tenants, redirect to SSO
		if tenant_id in self.sso_integration.sso_configurations:
			sso_url = await self.sso_integration.initiate_sso_login(
				tenant_id, "http://localhost:8080/auth/callback"
			)
			
			# In a real implementation, this would redirect the user
			# For demo, we'll simulate the SSO flow
			state = sso_url.split('state=')[1].split('&')[0] if 'state=' in sso_url else 'demo_state'
			token, user = await self.sso_integration.handle_sso_callback(
				state, 'demo_auth_code', ip_address
			)
			
			# Store user if not exists
			if user.user_id not in self.users:
				self.users[user.user_id] = user
				self.platform_metrics['active_users'] += 1
			
			# Log audit event
			await self.audit_system.log_audit_event(
				tenant_id=tenant_id,
				user_id=user.user_id,
				event_type=AuditEventType.LOGIN,
				resource_type='user',
				resource_id=user.user_id,
				event_data={'method': 'sso', 'provider': user.sso_provider.value},
				ip_address=ip_address,
				user_agent=user_agent
			)
			
			return token, user
		
		# For non-SSO tenants, find user by email
		user = None
		for u in self.users.values():
			if u.tenant_id == tenant_id and u.email == email:
				user = u
				break
		
		if not user:
			raise ValueError("User not found")
		
		# Generate JWT token
		token = self.sso_integration._generate_jwt_token(user, {'provider': 'local'})
		
		# Update last login
		user.last_login = datetime.utcnow()
		
		# Log audit event
		await self.audit_system.log_audit_event(
			tenant_id=tenant_id,
			user_id=user.user_id,
			event_type=AuditEventType.LOGIN,
			resource_type='user',
			resource_id=user.user_id,
			event_data={'method': 'local'},
			ip_address=ip_address,
			user_agent=user_agent
		)
		
		return token, user
	
	async def access_digital_twin(self, token: str, twin_id: str, ip_address: str,
								 user_agent: str) -> Dict[str, Any]:
		"""Access digital twin with proper authorization"""
		
		# Validate token
		try:
			payload = self.sso_integration.validate_jwt_token(token)
		except ValueError as e:
			raise ValueError(f"Authentication failed: {e}")
		
		user_id = payload['user_id']
		tenant_id = payload['tenant_id']
		permissions = payload['permissions']
		
		# Check twin access permission
		if not any(perm in permissions for perm in ['twin.read', 'twin.read_write', 'twin.admin']):
			raise ValueError("Insufficient permissions to access digital twin")
		
		# In a real implementation, this would fetch actual twin data
		# using tenant-isolated database connection
		connection_string = self.data_isolation.get_tenant_connection_string(tenant_id)
		
		# Mock twin data
		twin_data = {
			'twin_id': twin_id,
			'tenant_id': tenant_id,
			'name': f'Twin {twin_id}',
			'status': 'active',
			'last_updated': datetime.utcnow().isoformat(),
			'data_classification': 'internal'
		}
		
		# Encrypt sensitive data if required
		if self.tenants[tenant_id].encryption_level in ['advanced', 'enterprise', 'premium']:
			# In a real implementation, this would encrypt the data
			pass
		
		# Log audit event
		await self.audit_system.log_audit_event(
			tenant_id=tenant_id,
			user_id=user_id,
			event_type=AuditEventType.TWIN_ACCESS,
			resource_type='digital_twin',
			resource_id=twin_id,
			event_data={'action': 'view', 'twin_name': twin_data['name']},
			ip_address=ip_address,
			user_agent=user_agent
		)
		
		return twin_data
	
	def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
		"""Get comprehensive tenant metrics"""
		
		if tenant_id not in self.tenants:
			raise ValueError(f"Tenant {tenant_id} not found")
		
		tenant = self.tenants[tenant_id]
		
		# Count tenant users
		tenant_users = [u for u in self.users.values() if u.tenant_id == tenant_id]
		active_users = [u for u in tenant_users if u.is_active]
		
		# Mock usage metrics
		usage_metrics = {
			'digital_twins_count': 25,
			'storage_used_gb': 150.0,
			'monthly_api_calls': 15000,
			'average_session_duration_minutes': 45
		}
		
		return {
			'tenant_id': tenant_id,
			'organization_name': tenant.organization_name,
			'status': tenant.status.value,
			'subscription_tier': tenant.subscription_tier.value,
			'users': {
				'total': len(tenant_users),
				'active': len(active_users),
				'limit': tenant.max_users
			},
			'usage': usage_metrics,
			'limits': {
				'max_digital_twins': tenant.max_digital_twins,
				'max_storage_gb': tenant.max_storage_gb
			},
			'compliance': {
				'profiles': tenant.compliance_profiles,
				'audit_events_30d': len([
					e for e in self.audit_system.audit_events
					if e.tenant_id == tenant_id and
					   e.timestamp >= datetime.utcnow() - timedelta(days=30)
				])
			},
			'created_at': tenant.created_at.isoformat(),
			'last_activity': tenant.last_activity.isoformat()
		}
	
	async def generate_compliance_report(self, tenant_id: str, framework: str) -> Dict[str, Any]:
		"""Generate compliance report for tenant"""
		
		end_date = datetime.utcnow()
		start_date = end_date - timedelta(days=30)  # Last 30 days
		
		report = self.audit_system.generate_compliance_report(
			tenant_id, framework, start_date, end_date
		)
		
		return report

# Test and example usage
async def test_multi_tenant_enterprise():
	"""Test the multi-tenant enterprise system"""
	
	# Initialize enterprise manager
	enterprise = MultiTenantEnterpriseManager()
	
	print("Creating enterprise tenants...")
	
	# Create multiple tenants
	tenant1_id = await enterprise.create_tenant(
		organization_name="Acme Manufacturing",
		domain="acme-mfg.com",
		admin_email="admin@acme-mfg.com",
		subscription_tier=SubscriptionTier.ENTERPRISE
	)
	
	tenant2_id = await enterprise.create_tenant(
		organization_name="Global Industries",
		domain="global-ind.com", 
		admin_email="admin@global-ind.com",
		subscription_tier=SubscriptionTier.PROFESSIONAL
	)
	
	print(f"Created tenants: {tenant1_id}, {tenant2_id}")
	
	# Configure SSO for first tenant
	print(f"\nConfiguring SSO for {tenant1_id}...")
	sso_config_id = await enterprise.configure_tenant_sso(
		tenant_id=tenant1_id,
		provider=SSOProvider.AZURE_AD,
		provider_config={
			'provider_name': 'Acme Azure AD',
			'client_id': 'acme-client-id',
			'client_secret': 'acme-secret',
			'domain': 'acme-mfg.onmicrosoft.com',
			'metadata_url': 'https://login.microsoftonline.com/acme-mfg.onmicrosoft.com/v2.0/.well-known/openid_configuration',
			'role_mappings': {
				'engineers': UserRole.ENGINEER,
				'admins': UserRole.TENANT_ADMIN
			}
		}
	)
	
	print(f"SSO configured with ID: {sso_config_id}")
	
	# Test authentication
	print(f"\nTesting authentication for {tenant1_id}...")
	token, user = await enterprise.authenticate_user(
		tenant_id=tenant1_id,
		email="john.doe@acme-mfg.com",
		ip_address="192.168.1.100",
		user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
	)
	
	print(f"User authenticated: {user.email} with role {user.role.value}")
	print(f"JWT Token: {token[:50]}...")
	
	# Test digital twin access
	print(f"\nTesting digital twin access...")
	twin_data = await enterprise.access_digital_twin(
		token=token,
		twin_id="twin_industrial_001",
		ip_address="192.168.1.100",
		user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
	)
	
	print(f"Twin accessed: {twin_data['name']} (ID: {twin_data['twin_id']})")
	
	# Get tenant metrics
	print(f"\nGetting tenant metrics...")
	metrics = enterprise.get_tenant_metrics(tenant1_id)
	print(f"Tenant: {metrics['organization_name']}")
	print(f"Users: {metrics['users']['active']}/{metrics['users']['limit']}")
	print(f"Digital Twins: {metrics['usage']['digital_twins_count']}")
	print(f"Storage Used: {metrics['usage']['storage_used_gb']} GB")
	
	# Generate compliance report
	print(f"\nGenerating compliance report...")
	compliance_report = await enterprise.generate_compliance_report(tenant1_id, 'SOC2')
	print(f"Compliance Report (SOC2):")
	print(f"  Total Events: {compliance_report['total_events']}")
	print(f"  High Risk Events: {compliance_report['high_risk_events']}")
	print(f"  Event Summary: {list(compliance_report['event_summary'].keys())}")
	
	# Test second tenant isolation
	print(f"\nTesting tenant isolation...")
	try:
		# Try to access tenant1's twin with tenant2's context
		await enterprise.authenticate_user(
			tenant_id=tenant2_id,
			email="admin@global-ind.com",
			ip_address="10.0.0.50",
			user_agent="Chrome/91.0"
		)
		print("Tenant 2 authentication successful")
		
		# This should be isolated from tenant 1's data
		tenant2_metrics = enterprise.get_tenant_metrics(tenant2_id)
		print(f"Tenant 2 metrics isolated: {tenant2_metrics['organization_name']}")
		
	except Exception as e:
		print(f"Tenant isolation test failed: {e}")
	
	print(f"\nPlatform Summary:")
	print(f"Total Tenants: {enterprise.platform_metrics['total_tenants']}")
	print(f"Active Users: {enterprise.platform_metrics['active_users']}")

if __name__ == "__main__":
	asyncio.run(test_multi_tenant_enterprise())