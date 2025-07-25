"""
Multi-Tenant Enterprise Models

Database models for enterprise-grade multi-tenant deployment with complete tenant isolation,
SSO integration, role-based access control, and comprehensive audit capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class MTETenant(Model, AuditMixin, BaseMixin):
	"""
	Enterprise tenant configuration and management.
	
	Represents an isolated tenant organization with its own configuration,
	limits, and enterprise-grade features.
	"""
	__tablename__ = 'mte_tenant'
	
	# Identity
	tenant_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	
	# Organization Information
	organization_name = Column(String(200), nullable=False, index=True)
	domain = Column(String(100), nullable=False, unique=True, index=True)
	subdomain = Column(String(50), nullable=True)  # Custom subdomain
	logo_url = Column(String(500), nullable=True)
	
	# Status and Lifecycle
	status = Column(String(20), default='active', index=True)  # active, suspended, pending, deactivated, trial
	subscription_tier = Column(String(20), default='basic', index=True)  # basic, professional, enterprise, premium
	billing_status = Column(String(20), default='active')  # active, overdue, suspended
	
	# Limits and Quotas
	max_users = Column(Integer, default=10)
	max_digital_twins = Column(Integer, default=50)
	max_storage_gb = Column(Float, default=100.0)
	max_api_calls_per_month = Column(Integer, default=10000)
	max_concurrent_sessions = Column(Integer, default=5)
	
	# Current Usage
	current_users = Column(Integer, default=0)
	current_digital_twins = Column(Integer, default=0)
	current_storage_gb = Column(Float, default=0.0)
	current_api_calls_month = Column(Integer, default=0)
	
	# Features and Configuration
	features_enabled = Column(JSON, default=list)  # List of enabled features
	custom_branding = Column(JSON, default=dict)  # Custom branding configuration
	configuration = Column(JSON, default=dict)  # Tenant-specific configuration
	
	# Data Management
	data_retention_days = Column(Integer, default=365)
	backup_enabled = Column(Boolean, default=True)
	encryption_level = Column(String(20), default='standard')  # standard, advanced, enterprise, premium
	geographic_region = Column(String(50), default='us-east-1')
	
	# Compliance and Security
	compliance_profiles = Column(JSON, default=list)  # SOC2, GDPR, HIPAA, etc.
	security_policies = Column(JSON, default=dict)  # Security policy configuration
	audit_retention_days = Column(Integer, default=2555)  # 7 years default
	
	# Contact and Billing
	primary_contact_email = Column(String(255), nullable=False)
	billing_contact_email = Column(String(255), nullable=True)
	support_tier = Column(String(20), default='standard')  # standard, premium, enterprise
	
	# Activity Tracking
	last_activity = Column(DateTime, default=datetime.utcnow)
	last_login = Column(DateTime, nullable=True)
	trial_end_date = Column(DateTime, nullable=True)
	
	# Relationships
	users = relationship("MTEUser", back_populates="tenant")
	sso_configurations = relationship("MTESSOConfiguration", back_populates="tenant")
	audit_events = relationship("MTEAuditEvent", back_populates="tenant")
	
	def __repr__(self):
		return f"<MTETenant {self.organization_name}>"
	
	def is_active(self) -> bool:
		"""Check if tenant is active and not suspended"""
		return self.status == 'active' and self.billing_status != 'suspended'
	
	def is_within_limits(self) -> Dict[str, bool]:
		"""Check if tenant is within usage limits"""
		return {
			'users': self.max_users == -1 or self.current_users <= self.max_users,
			'digital_twins': self.max_digital_twins == -1 or self.current_digital_twins <= self.max_digital_twins,
			'storage': self.max_storage_gb == -1 or self.current_storage_gb <= self.max_storage_gb,
			'api_calls': self.max_api_calls_per_month == -1 or self.current_api_calls_month <= self.max_api_calls_per_month
		}
	
	def calculate_usage_percentage(self) -> Dict[str, float]:
		"""Calculate usage percentages for each quota"""
		return {
			'users': (self.current_users / self.max_users * 100) if self.max_users > 0 else 0,
			'digital_twins': (self.current_digital_twins / self.max_digital_twins * 100) if self.max_digital_twins > 0 else 0,
			'storage': (self.current_storage_gb / self.max_storage_gb * 100) if self.max_storage_gb > 0 else 0,
			'api_calls': (self.current_api_calls_month / self.max_api_calls_per_month * 100) if self.max_api_calls_per_month > 0 else 0
		}
	
	def get_feature_access(self, feature: str) -> bool:
		"""Check if tenant has access to specific feature"""
		return feature in self.features_enabled
	
	def update_activity(self):
		"""Update last activity timestamp"""
		self.last_activity = datetime.utcnow()


class MTEUser(Model, AuditMixin, BaseMixin):
	"""
	Enterprise user with tenant association and role-based access.
	
	Represents users within tenant organizations with comprehensive
	access control and audit capabilities.
	"""
	__tablename__ = 'mte_user'
	
	# Identity
	user_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), ForeignKey('mte_tenant.tenant_id'), nullable=False, index=True)
	
	# User Information
	email = Column(String(255), nullable=False, index=True)
	first_name = Column(String(100), nullable=False)
	last_name = Column(String(100), nullable=False)
	display_name = Column(String(200), nullable=True)
	avatar_url = Column(String(500), nullable=True)
	
	# Authentication
	password_hash = Column(String(255), nullable=True)  # For local auth
	external_user_id = Column(String(255), nullable=True)  # For SSO
	sso_provider = Column(String(50), nullable=True)  # okta, azure_ad, google_workspace, etc.
	
	# Role and Permissions
	role = Column(String(50), default='viewer', index=True)  # super_admin, tenant_admin, twin_admin, engineer, analyst, viewer, guest
	permissions = Column(JSON, default=list)  # List of specific permissions
	groups = Column(JSON, default=list)  # User groups for role mapping
	
	# Status and Security
	is_active = Column(Boolean, default=True)
	is_verified = Column(Boolean, default=False)
	email_verified_at = Column(DateTime, nullable=True)
	
	# Login Security
	last_login = Column(DateTime, nullable=True)
	last_login_ip = Column(String(45), nullable=True)  # IPv6 support
	failed_login_attempts = Column(Integer, default=0)
	account_locked_until = Column(DateTime, nullable=True)
	force_password_change = Column(Boolean, default=False)
	
	# Session Management
	current_session_id = Column(String(255), nullable=True)
	max_concurrent_sessions = Column(Integer, default=3)
	session_timeout_minutes = Column(Integer, default=480)  # 8 hours
	
	# Preferences and Settings
	timezone = Column(String(50), default='UTC')
	language = Column(String(10), default='en')
	notification_preferences = Column(JSON, default=dict)
	ui_preferences = Column(JSON, default=dict)
	
	# Activity Tracking
	last_activity = Column(DateTime, nullable=True)
	total_login_count = Column(Integer, default=0)
	invitation_sent_at = Column(DateTime, nullable=True)
	invitation_accepted_at = Column(DateTime, nullable=True)
	
	# Relationships
	tenant = relationship("MTETenant", back_populates="users")
	audit_events = relationship("MTEAuditEvent", back_populates="user")
	
	def __repr__(self):
		return f"<MTEUser {self.email}>"
	
	def get_full_name(self) -> str:
		"""Get user's full name"""
		return f"{self.first_name} {self.last_name}".strip()
	
	def is_account_locked(self) -> bool:
		"""Check if account is currently locked"""
		if not self.account_locked_until:
			return False
		return datetime.utcnow() < self.account_locked_until
	
	def can_login(self) -> bool:
		"""Check if user can login"""
		return (
			self.is_active and
			not self.is_account_locked() and
			self.tenant.is_active()
		)
	
	def has_permission(self, permission: str) -> bool:
		"""Check if user has specific permission"""
		return permission in self.permissions
	
	def has_role(self, role: str) -> bool:
		"""Check if user has specific role"""
		return self.role == role
	
	def record_login_attempt(self, success: bool, ip_address: str):
		"""Record login attempt"""
		if success:
			self.last_login = datetime.utcnow()
			self.last_login_ip = ip_address
			self.failed_login_attempts = 0
			self.total_login_count += 1
			if self.account_locked_until:
				self.account_locked_until = None
		else:
			self.failed_login_attempts += 1
			if self.failed_login_attempts >= 5:  # Lock after 5 failed attempts
				self.account_locked_until = datetime.utcnow() + timedelta(minutes=30)
	
	def update_activity(self):
		"""Update last activity timestamp"""
		self.last_activity = datetime.utcnow()


class MTESSOConfiguration(Model, AuditMixin, BaseMixin):
	"""
	Single Sign-On configuration for tenant integration.
	
	Manages SSO provider configurations for enterprise authentication
	with support for multiple providers per tenant.
	"""
	__tablename__ = 'mte_sso_configuration'
	
	# Identity
	config_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), ForeignKey('mte_tenant.tenant_id'), nullable=False, index=True)
	
	# Provider Information
	provider = Column(String(50), nullable=False, index=True)  # okta, azure_ad, google_workspace, aws_sso, saml_generic, ldap
	provider_name = Column(String(200), nullable=False)  # Display name
	provider_domain = Column(String(200), nullable=False)
	
	# Configuration
	client_id = Column(String(255), nullable=False)
	client_secret_encrypted = Column(Text, nullable=False)  # Encrypted client secret
	metadata_url = Column(String(500), nullable=True)
	issuer_url = Column(String(500), nullable=True)
	authorization_url = Column(String(500), nullable=True)
	token_url = Column(String(500), nullable=True)
	userinfo_url = Column(String(500), nullable=True)
	
	# Mapping Configuration
	attribute_mappings = Column(JSON, default=dict)  # Map SSO attributes to user fields
	role_mappings = Column(JSON, default=dict)  # Map SSO groups/roles to platform roles
	group_attribute_name = Column(String(100), default='groups')
	
	# Behavior Settings
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)  # Default SSO provider for tenant
	auto_provision_users = Column(Boolean, default=True)
	auto_update_users = Column(Boolean, default=True)
	default_role = Column(String(50), default='viewer')
	
	# Security Settings
	require_ssl = Column(Boolean, default=True)
	verify_ssl_certificates = Column(Boolean, default=True)
	signature_algorithm = Column(String(50), default='RS256')
	
	# SAML Specific Settings
	saml_entity_id = Column(String(500), nullable=True)
	saml_acs_url = Column(String(500), nullable=True)
	saml_sls_url = Column(String(500), nullable=True)
	saml_certificate = Column(Text, nullable=True)
	
	# Usage Statistics
	total_logins = Column(Integer, default=0)
	last_used = Column(DateTime, nullable=True)
	configuration_tested_at = Column(DateTime, nullable=True)
	configuration_test_result = Column(Boolean, nullable=True)
	
	# Relationships
	tenant = relationship("MTETenant", back_populates="sso_configurations")
	
	def __repr__(self):
		return f"<MTESSOConfiguration {self.provider_name}>"
	
	def is_configured(self) -> bool:
		"""Check if SSO configuration is complete"""
		required_fields = [self.client_id, self.provider_domain]
		return all(field for field in required_fields)
	
	def record_login(self):
		"""Record successful SSO login"""
		self.total_logins += 1
		self.last_used = datetime.utcnow()
	
	def test_configuration(self) -> bool:
		"""Test SSO configuration (placeholder)"""
		# In real implementation, this would test connectivity to SSO provider
		self.configuration_tested_at = datetime.utcnow()
		self.configuration_test_result = True
		return True


class MTEAuditEvent(Model, AuditMixin, BaseMixin):
	"""
	Comprehensive audit event logging for enterprise compliance.
	
	Tracks all user activities, system events, and security-related
	actions for compliance and security monitoring.
	"""
	__tablename__ = 'mte_audit_event'
	
	# Identity
	event_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), ForeignKey('mte_tenant.tenant_id'), nullable=False, index=True)
	user_id = Column(String(36), ForeignKey('mte_user.user_id'), nullable=True, index=True)
	
	# Event Classification
	event_type = Column(String(50), nullable=False, index=True)  # login, logout, twin_access, data_export, etc.
	event_category = Column(String(50), nullable=False, index=True)  # authentication, authorization, data, system, security
	severity = Column(String(20), default='info', index=True)  # info, warning, error, critical
	
	# Event Details
	event_name = Column(String(200), nullable=False)
	event_description = Column(Text, nullable=True)
	resource_type = Column(String(100), nullable=True)  # digital_twin, user, configuration, etc.
	resource_id = Column(String(255), nullable=True)
	resource_name = Column(String(200), nullable=True)
	
	# Event Data
	event_data = Column(JSON, default=dict)  # Detailed event information
	previous_values = Column(JSON, default=dict)  # Previous values for change events
	new_values = Column(JSON, default=dict)  # New values for change events
	
	# Context Information
	ip_address = Column(String(45), nullable=True)  # IPv6 support
	user_agent = Column(Text, nullable=True)
	session_id = Column(String(255), nullable=True)
	request_id = Column(String(255), nullable=True)
	
	# Location and Device
	geographic_location = Column(String(200), nullable=True)
	device_type = Column(String(50), nullable=True)
	device_os = Column(String(100), nullable=True)
	browser = Column(String(100), nullable=True)
	
	# Result and Impact
	success = Column(Boolean, default=True)
	error_code = Column(String(50), nullable=True)
	error_message = Column(Text, nullable=True)
	response_time_ms = Column(Integer, nullable=True)
	
	# Risk Assessment
	risk_score = Column(Float, default=0.0)  # 0-10 risk score
	risk_factors = Column(JSON, default=list)  # List of risk factors
	anomaly_score = Column(Float, nullable=True)  # ML-based anomaly detection score
	
	# Compliance and Retention
	compliance_frameworks = Column(JSON, default=list)  # Which frameworks this event applies to
	retention_period_days = Column(Integer, default=2555)  # How long to retain this event
	tamper_proof_hash = Column(String(255), nullable=True)  # For audit trail integrity
	
	# Timeline
	timestamp = Column(DateTime, default=datetime.utcnow, index=True)
	processed_at = Column(DateTime, nullable=True)
	archived_at = Column(DateTime, nullable=True)
	
	# Relationships
	tenant = relationship("MTETenant", back_populates="audit_events")
	user = relationship("MTEUser", back_populates="audit_events")
	
	def __repr__(self):
		return f"<MTEAuditEvent {self.event_name}>"
	
	def calculate_risk_score(self) -> float:
		"""Calculate risk score based on event characteristics"""
		base_scores = {
			'login': 2.0,
			'logout': 1.0,
			'twin_access': 3.0,
			'data_export': 7.0,
			'configuration_change': 6.0,
			'user_invitation': 4.0,
			'role_assignment': 5.0,
			'api_access': 2.0,
			'security_violation': 9.0,
			'data_deletion': 8.0,
			'admin_action': 6.0
		}
		
		base_score = base_scores.get(self.event_type, 3.0)
		
		# Adjust for failure
		if not self.success:
			base_score += 2.0
		
		# Adjust for unusual timing (outside business hours)
		if self.timestamp.hour < 8 or self.timestamp.hour > 18:
			base_score += 1.0
		
		# Adjust for high-value resources
		if self.resource_type in ['configuration', 'user_management', 'security_settings']:
			base_score += 1.5
		
		self.risk_score = min(10.0, base_score)
		return self.risk_score
	
	def is_high_risk(self) -> bool:
		"""Check if event is high risk"""
		return self.risk_score >= 7.0
	
	def should_alert(self) -> bool:
		"""Check if event should trigger alerts"""
		return (
			self.is_high_risk() or
			not self.success and self.event_type in ['login', 'api_access', 'configuration_change'] or
			self.severity in ['error', 'critical']
		)
	
	def generate_tamper_proof_hash(self) -> str:
		"""Generate tamper-proof hash for audit trail integrity"""
		import hashlib
		data_to_hash = f"{self.event_id}{self.timestamp}{self.event_type}{self.user_id}{self.event_data}"
		self.tamper_proof_hash = hashlib.sha256(data_to_hash.encode()).hexdigest()
		return self.tamper_proof_hash


class MTETenantUsage(Model, AuditMixin, BaseMixin):
	"""
	Tenant usage tracking and billing metrics.
	
	Tracks detailed usage metrics for billing, reporting,
	and resource management purposes.
	"""
	__tablename__ = 'mte_tenant_usage'
	
	# Identity
	usage_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), ForeignKey('mte_tenant.tenant_id'), nullable=False, index=True)
	
	# Time Period
	usage_date = Column(DateTime, nullable=False, index=True)
	usage_month = Column(String(7), nullable=False, index=True)  # YYYY-MM format
	billing_period_start = Column(DateTime, nullable=False)
	billing_period_end = Column(DateTime, nullable=False)
	
	# User Metrics
	active_users = Column(Integer, default=0)
	total_user_sessions = Column(Integer, default=0)
	average_session_duration_minutes = Column(Float, default=0.0)
	
	# Digital Twin Metrics
	active_digital_twins = Column(Integer, default=0)
	twin_operations_count = Column(Integer, default=0)
	twin_data_points_processed = Column(Integer, default=0)
	
	# API Usage
	api_calls_total = Column(Integer, default=0)
	api_calls_successful = Column(Integer, default=0)
	api_calls_failed = Column(Integer, default=0)
	api_response_time_avg_ms = Column(Float, default=0.0)
	
	# Storage Usage
	storage_used_gb = Column(Float, default=0.0)
	storage_peak_gb = Column(Float, default=0.0)
	backup_storage_gb = Column(Float, default=0.0)
	
	# Compute Usage
	compute_hours = Column(Float, default=0.0)
	cpu_hours = Column(Float, default=0.0)
	memory_gb_hours = Column(Float, default=0.0)
	gpu_hours = Column(Float, default=0.0)
	
	# Data Transfer
	data_ingress_gb = Column(Float, default=0.0)
	data_egress_gb = Column(Float, default=0.0)
	
	# Feature Usage
	advanced_analytics_runs = Column(Integer, default=0)
	ml_model_executions = Column(Integer, default=0)
	simulation_runs = Column(Integer, default=0)
	report_generations = Column(Integer, default=0)
	
	# Support Usage
	support_tickets_created = Column(Integer, default=0)
	support_tickets_resolved = Column(Integer, default=0)
	support_response_time_avg_hours = Column(Float, default=0.0)
	
	# Billing
	calculated_cost = Column(Float, default=0.0)
	cost_breakdown = Column(JSON, default=dict)  # Detailed cost breakdown
	overage_charges = Column(Float, default=0.0)
	
	def __repr__(self):
		return f"<MTETenantUsage {self.tenant_id} - {self.usage_month}>"
	
	def calculate_total_cost(self) -> float:
		"""Calculate total cost for the usage period"""
		# This would include complex pricing logic based on tier, usage, etc.
		base_cost = 0.0
		
		# API call costs
		api_cost = max(0, self.api_calls_total - 10000) * 0.001  # $0.001 per call over 10k
		
		# Storage costs
		storage_cost = max(0, self.storage_used_gb - 100) * 0.10  # $0.10 per GB over 100GB
		
		# Compute costs
		compute_cost = self.compute_hours * 0.05  # $0.05 per compute hour
		
		self.calculated_cost = base_cost + api_cost + storage_cost + compute_cost
		return self.calculated_cost
	
	def is_over_limits(self, tenant_limits: Dict[str, Any]) -> Dict[str, bool]:
		"""Check if usage exceeds tenant limits"""
		return {
			'api_calls': self.api_calls_total > tenant_limits.get('max_api_calls_per_month', 10000),
			'storage': self.storage_used_gb > tenant_limits.get('max_storage_gb', 100),
			'active_users': self.active_users > tenant_limits.get('max_users', 10)
		}


class MTEComplianceReport(Model, AuditMixin, BaseMixin):
	"""
	Generated compliance reports for enterprise tenants.
	
	Stores generated compliance reports for various frameworks
	and regulatory requirements.
	"""
	__tablename__ = 'mte_compliance_report'
	
	# Identity
	report_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), ForeignKey('mte_tenant.tenant_id'), nullable=False, index=True)
	
	# Report Details
	report_name = Column(String(200), nullable=False)
	compliance_framework = Column(String(50), nullable=False, index=True)  # SOC2, GDPR, HIPAA, ISO27001, etc.
	report_type = Column(String(50), default='standard')  # standard, detailed, executive
	
	# Report Period
	period_start = Column(DateTime, nullable=False)
	period_end = Column(DateTime, nullable=False)
	generated_at = Column(DateTime, default=datetime.utcnow)
	
	# Report Content
	executive_summary = Column(Text, nullable=True)
	report_data = Column(JSON, default=dict)  # Full report data
	metrics_summary = Column(JSON, default=dict)  # Key metrics summary
	findings = Column(JSON, default=list)  # Compliance findings
	recommendations = Column(JSON, default=list)  # Recommendations
	
	# Status and Validation
	status = Column(String(20), default='generated', index=True)  # generated, reviewed, approved, archived
	validation_status = Column(String(20), default='pending')  # pending, validated, failed
	reviewer_notes = Column(Text, nullable=True)
	
	# Access and Distribution
	generated_by = Column(String(36), nullable=False)
	access_level = Column(String(20), default='internal')  # internal, external, public
	download_count = Column(Integer, default=0)
	last_accessed = Column(DateTime, nullable=True)
	
	# File Storage
	report_file_path = Column(String(500), nullable=True)  # Path to generated report file
	report_file_format = Column(String(10), default='pdf')  # pdf, docx, xlsx
	report_file_size_bytes = Column(Integer, nullable=True)
	
	def __repr__(self):
		return f"<MTEComplianceReport {self.compliance_framework} - {self.report_name}>"
	
	def is_current(self) -> bool:
		"""Check if report is current (within last 30 days)"""
		return (datetime.utcnow() - self.generated_at).days <= 30
	
	def record_access(self):
		"""Record report access"""
		self.download_count += 1
		self.last_accessed = datetime.utcnow()