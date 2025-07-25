"""
Authentication & RBAC Models

Comprehensive database models for enterprise authentication, authorization,
and role-based access control with multi-tenant support and GDPR compliance.
"""

import uuid
import hashlib
import secrets
import pyotp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, Float, JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property
from flask_appbuilder import Model
from werkzeug.security import generate_password_hash, check_password_hash

from ..common.base import BaseCapabilityModel, AuditMixin, BaseMixin

# UUID7-like string generator for consistent ID generation
def uuid7str() -> str:
	"""Generate UUID7-like string for consistent ID generation"""
	return str(uuid.uuid4())


class ARUser(Model, AuditMixin, BaseMixin):
	"""
	Enhanced user model with comprehensive authentication capabilities.
	
	Supports multi-modal authentication, MFA, session management,
	and comprehensive security controls with GDPR compliance.
	"""
	
	__tablename__ = 'ar_user'
	
	# Primary Identity
	user_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	username = Column(String(100), nullable=True, index=True)
	email = Column(String(255), nullable=False, index=True)
	email_verified = Column(Boolean, default=False, nullable=False)
	email_verification_token = Column(String(100), nullable=True)
	email_verification_expires = Column(DateTime, nullable=True)
	
	# Authentication Credentials
	password_hash = Column(String(255), nullable=True)
	password_salt = Column(String(64), nullable=True)
	password_changed_at = Column(DateTime, nullable=True)
	password_history = Column(JSON, default=list)  # List of previous password hashes
	password_reset_token = Column(String(100), nullable=True)
	password_reset_expires = Column(DateTime, nullable=True)
	
	# Account Security
	failed_login_attempts = Column(Integer, default=0, nullable=False)
	account_locked_until = Column(DateTime, nullable=True)
	security_questions = Column(JSON, default=list)  # List of security question/answer pairs
	
	# Multi-Factor Authentication
	mfa_enabled = Column(Boolean, default=False, nullable=False)
	mfa_secret = Column(String(32), nullable=True)  # TOTP secret key
	mfa_backup_codes = Column(JSON, default=list)  # List of backup codes
	mfa_phone = Column(String(20), nullable=True)  # Phone number for SMS MFA
	mfa_phone_verified = Column(Boolean, default=False)
	
	# Session Management
	max_concurrent_sessions = Column(Integer, default=5, nullable=False)
	current_session_count = Column(Integer, default=0, nullable=False)
	last_login_at = Column(DateTime, nullable=True)
	last_login_ip = Column(String(45), nullable=True)
	last_activity_at = Column(DateTime, nullable=True)
	timezone = Column(String(50), default='UTC')
	
	# Account Status
	is_active = Column(Boolean, default=True, nullable=False)
	is_verified = Column(Boolean, default=False, nullable=False)
	is_system_user = Column(Boolean, default=False, nullable=False)
	account_type = Column(String(20), default='user', nullable=False)  # user, service, admin, system
	activation_token = Column(String(100), nullable=True)
	activation_expires = Column(DateTime, nullable=True)
	
	# Security Configuration
	security_level = Column(String(20), default='standard', nullable=False)  # basic, standard, high, critical
	require_mfa = Column(Boolean, default=False, nullable=False)
	require_password_change = Column(Boolean, default=False, nullable=False)
	allowed_ip_ranges = Column(JSON, default=list)  # List of allowed IP ranges
	blocked_ip_ranges = Column(JSON, default=list)  # List of blocked IP ranges
	device_trust_enabled = Column(Boolean, default=True, nullable=False)
	
	# Privacy & GDPR
	gdpr_consent_given = Column(Boolean, default=False)
	gdpr_consent_date = Column(DateTime, nullable=True)
	data_retention_period = Column(Integer, default=2555)  # Days (7 years default)
	anonymization_scheduled = Column(DateTime, nullable=True)
	
	# API Access
	api_key_hash = Column(String(255), nullable=True)
	api_key_created_at = Column(DateTime, nullable=True)
	api_key_last_used = Column(DateTime, nullable=True)
	api_rate_limit = Column(Integer, default=1000)  # Requests per hour
	api_requests_today = Column(Integer, default=0)
	
	# Metadata
	profile_data = Column(JSON, default=dict)  # Additional profile information
	preferences = Column(JSON, default=dict)  # User preferences
	security_metadata = Column(JSON, default=dict)  # Security-related metadata
	
	# Relationships
	roles = relationship("ARUserRole", back_populates="user", cascade="all, delete-orphan")
	permissions = relationship("ARUserPermission", back_populates="user", cascade="all, delete-orphan")
	sessions = relationship("ARUserSession", back_populates="user", cascade="all, delete-orphan")
	login_attempts = relationship("ARLoginAttempt", back_populates="user", cascade="all, delete-orphan")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'email', name='uq_ar_user_tenant_email'),
		UniqueConstraint('tenant_id', 'username', name='uq_ar_user_tenant_username'),
		Index('ix_ar_user_tenant_active', 'tenant_id', 'is_active'),
		Index('ix_ar_user_security_level', 'security_level'),
		Index('ix_ar_user_last_activity', 'last_activity_at'),
	)
	
	@validates('email')
	def validate_email(self, key, address):
		"""Validate email format"""
		import re
		if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', address):
			raise ValueError("Invalid email format")
		return address.lower()
	
	@validates('security_level')
	def validate_security_level(self, key, level):
		"""Validate security level"""
		allowed_levels = ['basic', 'standard', 'high', 'critical']
		if level not in allowed_levels:
			raise ValueError(f"Security level must be one of: {allowed_levels}")
		return level
	
	@validates('account_type')
	def validate_account_type(self, key, account_type):
		"""Validate account type"""
		allowed_types = ['user', 'service', 'admin', 'system']
		if account_type not in allowed_types:
			raise ValueError(f"Account type must be one of: {allowed_types}")
		return account_type
	
	def set_password(self, password: str) -> None:
		"""Set user password with secure hashing"""
		# Generate salt
		self.password_salt = secrets.token_hex(32)
		
		# Hash password with salt
		self.password_hash = generate_password_hash(f"{password}{self.password_salt}")
		self.password_changed_at = datetime.utcnow()
		
		# Add to password history (keep last 12 passwords)
		if not self.password_history:
			self.password_history = []
		
		self.password_history.append({
			'hash': self.password_hash,
			'changed_at': self.password_changed_at.isoformat()
		})
		
		# Keep only last 12 passwords
		self.password_history = self.password_history[-12:]
		
		# Reset failed attempts
		self.failed_login_attempts = 0
		self.account_locked_until = None
	
	def check_password(self, password: str) -> bool:
		"""Check password against stored hash"""
		if not self.password_hash or not self.password_salt:
			return False
		
		return check_password_hash(self.password_hash, f"{password}{self.password_salt}")
	
	def is_password_in_history(self, password: str) -> bool:
		"""Check if password was used before"""
		if not self.password_history:
			return False
		
		password_with_salt = f"{password}{self.password_salt}"
		
		for hist_entry in self.password_history:
			if check_password_hash(hist_entry['hash'], password_with_salt):
				return True
		
		return False
	
	def setup_mfa(self) -> str:
		"""Setup TOTP-based MFA and return QR code URL"""
		if not self.mfa_secret:
			self.mfa_secret = pyotp.random_base32()
		
		# Generate backup codes
		self.mfa_backup_codes = [secrets.token_hex(4) for _ in range(10)]
		
		# Return QR code URL for app setup
		totp = pyotp.TOTP(self.mfa_secret)
		return totp.provisioning_uri(
			name=self.email,
			issuer_name="APG Enterprise"
		)
	
	def verify_mfa_token(self, token: str) -> bool:
		"""Verify TOTP token or backup code"""
		if not self.mfa_enabled or not self.mfa_secret:
			return False
		
		# Check TOTP token
		totp = pyotp.TOTP(self.mfa_secret)
		if totp.verify(token, valid_window=1):
			return True
		
		# Check backup codes
		if token in self.mfa_backup_codes:
			self.mfa_backup_codes.remove(token)
			return True
		
		return False
	
	def generate_api_key(self) -> str:
		"""Generate new API key for user"""
		api_key = secrets.token_urlsafe(32)
		self.api_key_hash = generate_password_hash(api_key)
		self.api_key_created_at = datetime.utcnow()
		return api_key
	
	def verify_api_key(self, api_key: str) -> bool:
		"""Verify API key"""
		if not self.api_key_hash:
			return False
		
		if check_password_hash(self.api_key_hash, api_key):
			self.api_key_last_used = datetime.utcnow()
			return True
		
		return False
	
	def is_account_locked(self) -> bool:
		"""Check if account is currently locked"""
		if not self.account_locked_until:
			return False
		
		return datetime.utcnow() < self.account_locked_until
	
	def lock_account(self, duration_minutes: int = 30) -> None:
		"""Lock account for specified duration"""
		self.account_locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
	
	def unlock_account(self) -> None:
		"""Unlock account and reset failed attempts"""
		self.account_locked_until = None
		self.failed_login_attempts = 0
	
	def record_failed_login(self) -> None:
		"""Record failed login attempt and lock if necessary"""
		self.failed_login_attempts += 1
		
		# Progressive lockout policy
		if self.failed_login_attempts >= 10:
			self.lock_account(120)  # 2 hours
		elif self.failed_login_attempts >= 5:
			self.lock_account(30)   # 30 minutes
		elif self.failed_login_attempts >= 3:
			self.lock_account(5)    # 5 minutes
	
	def get_effective_permissions(self) -> List[Dict[str, Any]]:
		"""Get all effective permissions from roles and direct assignments"""
		permissions = []
		
		# Get permissions from roles
		for user_role in self.roles:
			if user_role.is_active():
				role_permissions = user_role.role.get_all_permissions()
				permissions.extend(role_permissions)
		
		# Get direct permissions
		for user_perm in self.permissions:
			if user_perm.is_active():
				permissions.append(user_perm.permission.to_dict())
		
		# Remove duplicates
		seen = set()
		unique_permissions = []
		for perm in permissions:
			perm_key = f"{perm['resource_type']}:{perm['resource_name']}:{perm['action']}"
			if perm_key not in seen:
				seen.add(perm_key)
				unique_permissions.append(perm)
		
		return unique_permissions
	
	def has_permission(self, resource_type: str, resource_name: str, action: str) -> bool:
		"""Check if user has specific permission"""
		permissions = self.get_effective_permissions()
		
		for perm in permissions:
			if (perm['resource_type'] == resource_type and 
				perm['resource_name'] == resource_name and 
				perm['action'] == action):
				return True
		
		return False
	
	def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
		"""Convert user to dictionary representation"""
		data = {
			'user_id': self.user_id,
			'tenant_id': self.tenant_id,
			'username': self.username,
			'email': self.email,
			'email_verified': self.email_verified,
			'is_active': self.is_active,
			'is_verified': self.is_verified,
			'account_type': self.account_type,
			'security_level': self.security_level,
			'mfa_enabled': self.mfa_enabled,
			'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
			'last_activity_at': self.last_activity_at.isoformat() if self.last_activity_at else None,
			'created_on': self.created_on.isoformat() if self.created_on else None,
			'profile_data': self.profile_data,
			'preferences': self.preferences
		}
		
		if include_sensitive:
			data.update({
				'failed_login_attempts': self.failed_login_attempts,
				'account_locked_until': self.account_locked_until.isoformat() if self.account_locked_until else None,
				'require_mfa': self.require_mfa,
				'allowed_ip_ranges': self.allowed_ip_ranges,
				'security_metadata': self.security_metadata,
				'gdpr_consent_given': self.gdpr_consent_given,
				'gdpr_consent_date': self.gdpr_consent_date.isoformat() if self.gdpr_consent_date else None
			})
		
		return data


class ARRole(Model, AuditMixin, BaseMixin):
	"""
	Hierarchical role system with permission inheritance.
	
	Supports parent-child relationships, dynamic role assignment,
	and comprehensive permission management.
	"""
	
	__tablename__ = 'ar_role'
	
	# Identity
	role_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	name = Column(String(100), nullable=False)
	display_name = Column(String(200), nullable=True)
	description = Column(Text, nullable=True)
	
	# Hierarchy
	parent_role_id = Column(String(36), ForeignKey('ar_role.role_id'), nullable=True, index=True)
	parent_role = relationship("ARRole", remote_side="ARRole.role_id", back_populates="child_roles")
	child_roles = relationship("ARRole", back_populates="parent_role")
	hierarchy_level = Column(Integer, default=0, nullable=False)  # 0 = root level
	
	# Role Configuration
	is_system_role = Column(Boolean, default=False, nullable=False)
	is_assignable = Column(Boolean, default=True, nullable=False)
	is_default_role = Column(Boolean, default=False, nullable=False)
	max_users = Column(Integer, nullable=True)
	priority = Column(Integer, default=100, nullable=False)  # Higher = more important
	
	# Auto-assignment Rules
	auto_assign_conditions = Column(JSON, default=dict)  # Conditions for automatic assignment
	assignment_approval_required = Column(Boolean, default=False)
	
	# Time-based Constraints
	valid_from = Column(DateTime, nullable=True)
	valid_until = Column(DateTime, nullable=True)
	
	# Role Metadata
	role_metadata = Column(JSON, default=dict)
	tags = Column(JSON, default=list)
	
	# Relationships
	users = relationship("ARUserRole", back_populates="role", cascade="all, delete-orphan")
	permissions = relationship("ARRolePermission", back_populates="role", cascade="all, delete-orphan")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'name', name='uq_ar_role_tenant_name'),
		Index('ix_ar_role_tenant_assignable', 'tenant_id', 'is_assignable'),
		Index('ix_ar_role_hierarchy', 'hierarchy_level'),
		Index('ix_ar_role_priority', 'priority'),
	)
	
	@validates('name')
	def validate_name(self, key, name):
		"""Validate role name"""
		if not name or len(name.strip()) < 2:
			raise ValueError("Role name must be at least 2 characters")
		return name.strip()
	
	def is_active(self) -> bool:
		"""Check if role is currently active"""
		now = datetime.utcnow()
		
		if self.valid_from and now < self.valid_from:
			return False
		
		if self.valid_until and now > self.valid_until:
			return False
		
		return True
	
	def get_all_permissions(self) -> List[Dict[str, Any]]:
		"""Get all permissions including inherited from parent roles"""
		permissions = []
		
		# Get direct permissions
		for role_perm in self.permissions:
			if role_perm.is_active():
				permissions.append(role_perm.permission.to_dict())
		
		# Get inherited permissions from parent roles
		if self.parent_role:
			parent_permissions = self.parent_role.get_all_permissions()
			permissions.extend(parent_permissions)
		
		# Remove duplicates while preserving most specific permissions
		seen = set()
		unique_permissions = []
		for perm in permissions:
			perm_key = f"{perm['resource_type']}:{perm['resource_name']}:{perm['action']}"
			if perm_key not in seen:
				seen.add(perm_key)
				unique_permissions.append(perm)
		
		return unique_permissions
	
	def has_permission(self, resource_type: str, resource_name: str, action: str) -> bool:
		"""Check if role has specific permission"""
		permissions = self.get_all_permissions()
		
		for perm in permissions:
			if (perm['resource_type'] == resource_type and 
				perm['resource_name'] == resource_name and 
				perm['action'] == action):
				return True
		
		return False
	
	def get_hierarchy_path(self) -> List[str]:
		"""Get full hierarchy path from root to this role"""
		path = [self.name]
		
		current_role = self.parent_role
		while current_role:
			path.insert(0, current_role.name)
			current_role = current_role.parent_role
		
		return path
	
	def can_assign_to_user(self, user: ARUser) -> Tuple[bool, str]:
		"""Check if role can be assigned to user"""
		if not self.is_assignable:
			return False, "Role is not assignable"
		
		if not self.is_active():
			return False, "Role is not currently active"
		
		if self.max_users:
			current_users = len([ur for ur in self.users if ur.is_active()])
			if current_users >= self.max_users:
				return False, f"Role has reached maximum user limit ({self.max_users})"
		
		# Check auto-assignment conditions
		if self.auto_assign_conditions:
			# Implement condition checking logic here
			pass
		
		return True, "Role can be assigned"
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert role to dictionary representation"""
		return {
			'role_id': self.role_id,
			'tenant_id': self.tenant_id,
			'name': self.name,
			'display_name': self.display_name,
			'description': self.description,
			'parent_role_id': self.parent_role_id,
			'hierarchy_level': self.hierarchy_level,
			'is_system_role': self.is_system_role,
			'is_assignable': self.is_assignable,
			'is_default_role': self.is_default_role,
			'max_users': self.max_users,
			'priority': self.priority,
			'valid_from': self.valid_from.isoformat() if self.valid_from else None,
			'valid_until': self.valid_until.isoformat() if self.valid_until else None,
			'hierarchy_path': self.get_hierarchy_path(),
			'is_active': self.is_active(),
			'role_metadata': self.role_metadata,
			'tags': self.tags,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARPermission(Model, AuditMixin, BaseMixin):
	"""
	Granular permission system with resource-level access control.
	
	Supports fine-grained permissions with scope conditions,
	field restrictions, and dynamic policy evaluation.
	"""
	
	__tablename__ = 'ar_permission'
	
	# Identity
	permission_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Permission Definition
	resource_type = Column(String(100), nullable=False, index=True)  # capability, model, api, endpoint
	resource_name = Column(String(200), nullable=False, index=True)  # specific resource identifier
	action = Column(String(50), nullable=False, index=True)  # create, read, update, delete, execute, etc.
	
	# Permission Details
	display_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	permission_level = Column(String(20), default='standard', nullable=False)  # basic, standard, advanced, system
	
	# Scope and Constraints
	scope_conditions = Column(JSON, default=dict)  # Conditions for permission applicability
	field_restrictions = Column(JSON, default=list)  # Field-level access restrictions
	resource_filters = Column(JSON, default=dict)  # Dynamic resource filtering conditions
	
	# Permission Configuration
	is_system_permission = Column(Boolean, default=False, nullable=False)
	requires_approval = Column(Boolean, default=False, nullable=False)
	max_grant_duration = Column(Integer, nullable=True)  # Maximum grant duration in seconds
	auto_expire_after = Column(Integer, nullable=True)  # Auto-expire after N days of inactivity
	
	# Risk Assessment
	risk_level = Column(String(20), default='low', nullable=False)  # low, medium, high, critical
	compliance_tags = Column(JSON, default=list)  # Compliance-related tags
	
	# Permission Metadata
	permission_metadata = Column(JSON, default=dict)
	usage_analytics = Column(JSON, default=dict)  # Usage tracking data
	
	# Relationships
	roles = relationship("ARRolePermission", back_populates="permission", cascade="all, delete-orphan")
	users = relationship("ARUserPermission", back_populates="permission", cascade="all, delete-orphan")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'resource_type', 'resource_name', 'action', name='uq_ar_permission_resource_action'),
		Index('ix_ar_permission_tenant_resource', 'tenant_id', 'resource_type', 'resource_name'),
		Index('ix_ar_permission_level', 'permission_level'),
		Index('ix_ar_permission_risk', 'risk_level'),
	)
	
	@validates('action')
	def validate_action(self, key, action):
		"""Validate permission action"""
		allowed_actions = [
			'create', 'read', 'update', 'delete', 'execute', 'approve', 'reject',
			'export', 'import', 'share', 'publish', 'archive', 'restore', 'admin'
		]
		if action not in allowed_actions:
			raise ValueError(f"Action must be one of: {allowed_actions}")
		return action
	
	@validates('permission_level')
	def validate_permission_level(self, key, level):
		"""Validate permission level"""
		allowed_levels = ['basic', 'standard', 'advanced', 'system']
		if level not in allowed_levels:
			raise ValueError(f"Permission level must be one of: {allowed_levels}")
		return level
	
	@validates('risk_level')
	def validate_risk_level(self, key, level):
		"""Validate risk level"""
		allowed_levels = ['low', 'medium', 'high', 'critical']
		if level not in allowed_levels:
			raise ValueError(f"Risk level must be one of: {allowed_levels}")
		return level
	
	def evaluate_scope_conditions(self, context: Dict[str, Any]) -> bool:
		"""Evaluate scope conditions against provided context"""
		if not self.scope_conditions:
			return True
		
		# Implement condition evaluation logic
		# This would typically use a rule engine or expression evaluator
		
		for condition_key, condition_value in self.scope_conditions.items():
			context_value = context.get(condition_key)
			
			if isinstance(condition_value, dict):
				# Handle complex conditions (operators, ranges, etc.)
				operator = condition_value.get('operator', 'equals')
				expected_value = condition_value.get('value')
				
				if operator == 'equals' and context_value != expected_value:
					return False
				elif operator == 'in' and context_value not in expected_value:
					return False
				elif operator == 'not_in' and context_value in expected_value:
					return False
				elif operator == 'greater_than' and context_value <= expected_value:
					return False
				elif operator == 'less_than' and context_value >= expected_value:
					return False
			else:
				# Simple equality check
				if context_value != condition_value:
					return False
		
		return True
	
	def get_filtered_fields(self, requested_fields: List[str]) -> List[str]:
		"""Apply field restrictions to requested fields"""
		if not self.field_restrictions:
			return requested_fields
		
		# Remove restricted fields
		allowed_fields = []
		for field in requested_fields:
			if field not in self.field_restrictions:
				allowed_fields.append(field)
		
		return allowed_fields
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert permission to dictionary representation"""
		return {
			'permission_id': self.permission_id,
			'tenant_id': self.tenant_id,
			'resource_type': self.resource_type,
			'resource_name': self.resource_name,
			'action': self.action,
			'display_name': self.display_name,
			'description': self.description,
			'permission_level': self.permission_level,
			'scope_conditions': self.scope_conditions,
			'field_restrictions': self.field_restrictions,
			'resource_filters': self.resource_filters,
			'is_system_permission': self.is_system_permission,
			'requires_approval': self.requires_approval,
			'max_grant_duration': self.max_grant_duration,
			'risk_level': self.risk_level,
			'compliance_tags': self.compliance_tags,
			'permission_metadata': self.permission_metadata,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARUserRole(Model, AuditMixin, BaseMixin):
	"""
	User-Role assignment with time-based constraints and approval workflows.
	"""
	
	__tablename__ = 'ar_user_role'
	
	# Identity
	assignment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, primary_key=True)
	user_id = Column(String(36), ForeignKey('ar_user.user_id'), nullable=False, index=True)
	role_id = Column(String(36), ForeignKey('ar_role.role_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Assignment Details
	assigned_by = Column(String(36), nullable=False)  # User ID who assigned the role
	assigned_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	assignment_reason = Column(Text, nullable=True)
	
	# Time Constraints
	effective_from = Column(DateTime, nullable=True)
	effective_until = Column(DateTime, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True, nullable=False)
	requires_approval = Column(Boolean, default=False)
	approval_status = Column(String(20), default='approved')  # pending, approved, rejected
	approved_by = Column(String(36), nullable=True)
	approved_at = Column(DateTime, nullable=True)
	
	# Delegation
	is_delegated = Column(Boolean, default=False)
	delegated_by = Column(String(36), nullable=True)
	delegation_expires = Column(DateTime, nullable=True)
	
	# Metadata
	assignment_metadata = Column(JSON, default=dict)
	
	# Relationships
	user = relationship("ARUser", back_populates="roles")
	role = relationship("ARRole", back_populates="users")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('user_id', 'role_id', name='uq_ar_user_role'),
		Index('ix_ar_user_role_tenant', 'tenant_id'),
		Index('ix_ar_user_role_effective', 'effective_from', 'effective_until'),
		Index('ix_ar_user_role_approval', 'approval_status'),
	)
	
	def is_active(self) -> bool:
		"""Check if role assignment is currently active"""
		if not self.is_active:
			return False
		
		if self.approval_status != 'approved':
			return False
		
		now = datetime.utcnow()
		
		if self.effective_from and now < self.effective_from:
			return False
		
		if self.effective_until and now > self.effective_until:
			return False
		
		if self.is_delegated and self.delegation_expires and now > self.delegation_expires:
			return False
		
		return True
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert assignment to dictionary representation"""
		return {
			'assignment_id': self.assignment_id,
			'user_id': self.user_id,
			'role_id': self.role_id,
			'tenant_id': self.tenant_id,
			'assigned_by': self.assigned_by,
			'assigned_at': self.assigned_at.isoformat(),
			'assignment_reason': self.assignment_reason,
			'effective_from': self.effective_from.isoformat() if self.effective_from else None,
			'effective_until': self.effective_until.isoformat() if self.effective_until else None,
			'is_active': self.is_active,
			'approval_status': self.approval_status,
			'approved_by': self.approved_by,
			'approved_at': self.approved_at.isoformat() if self.approved_at else None,
			'is_delegated': self.is_delegated,
			'delegated_by': self.delegated_by,
			'delegation_expires': self.delegation_expires.isoformat() if self.delegation_expires else None,
			'assignment_metadata': self.assignment_metadata,
			'role_name': self.role.name if self.role else None,
			'role_display_name': self.role.display_name if self.role else None
		}


class ARUserPermission(Model, AuditMixin, BaseMixin):
	"""
	Direct user-permission assignment for exceptional access grants.
	"""
	
	__tablename__ = 'ar_user_permission'
	
	# Identity
	assignment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, primary_key=True)
	user_id = Column(String(36), ForeignKey('ar_user.user_id'), nullable=False, index=True)
	permission_id = Column(String(36), ForeignKey('ar_permission.permission_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Assignment Details
	granted_by = Column(String(36), nullable=False)
	granted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	grant_reason = Column(Text, nullable=True)
	
	# Time Constraints
	effective_from = Column(DateTime, nullable=True)
	effective_until = Column(DateTime, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True, nullable=False)
	requires_approval = Column(Boolean, default=False)
	approval_status = Column(String(20), default='approved')
	approved_by = Column(String(36), nullable=True)
	approved_at = Column(DateTime, nullable=True)
	
	# Usage Tracking
	last_used = Column(DateTime, nullable=True)
	usage_count = Column(Integer, default=0)
	
	# Metadata
	assignment_metadata = Column(JSON, default=dict)
	
	# Relationships
	user = relationship("ARUser", back_populates="permissions")
	permission = relationship("ARPermission", back_populates="users")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('user_id', 'permission_id', name='uq_ar_user_permission'),
		Index('ix_ar_user_permission_tenant', 'tenant_id'),
		Index('ix_ar_user_permission_effective', 'effective_from', 'effective_until'),
		Index('ix_ar_user_permission_approval', 'approval_status'),
	)
	
	def is_active(self) -> bool:
		"""Check if permission assignment is currently active"""
		if not self.is_active:
			return False
		
		if self.approval_status != 'approved':
			return False
		
		now = datetime.utcnow()
		
		if self.effective_from and now < self.effective_from:
			return False
		
		if self.effective_until and now > self.effective_until:
			return False
		
		return True
	
	def record_usage(self) -> None:
		"""Record permission usage"""
		self.last_used = datetime.utcnow()
		self.usage_count += 1
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert assignment to dictionary representation"""
		return {
			'assignment_id': self.assignment_id,
			'user_id': self.user_id,
			'permission_id': self.permission_id,
			'tenant_id': self.tenant_id,
			'granted_by': self.granted_by,
			'granted_at': self.granted_at.isoformat(),
			'grant_reason': self.grant_reason,
			'effective_from': self.effective_from.isoformat() if self.effective_from else None,
			'effective_until': self.effective_until.isoformat() if self.effective_until else None,
			'is_active': self.is_active,
			'approval_status': self.approval_status,
			'last_used': self.last_used.isoformat() if self.last_used else None,
			'usage_count': self.usage_count,
			'assignment_metadata': self.assignment_metadata,
			'permission_display_name': self.permission.display_name if self.permission else None
		}


class ARRolePermission(Model, AuditMixin, BaseMixin):
	"""
	Role-Permission assignment for role-based access control.
	"""
	
	__tablename__ = 'ar_role_permission'
	
	# Identity
	assignment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, primary_key=True)
	role_id = Column(String(36), ForeignKey('ar_role.role_id'), nullable=False, index=True)
	permission_id = Column(String(36), ForeignKey('ar_permission.permission_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Assignment Details
	assigned_by = Column(String(36), nullable=False)
	assigned_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	assignment_reason = Column(Text, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True, nullable=False)
	
	# Metadata
	assignment_metadata = Column(JSON, default=dict)
	
	# Relationships
	role = relationship("ARRole", back_populates="permissions")
	permission = relationship("ARPermission", back_populates="roles")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('role_id', 'permission_id', name='uq_ar_role_permission'),
		Index('ix_ar_role_permission_tenant', 'tenant_id'),
	)
	
	def is_active(self) -> bool:
		"""Check if permission assignment is currently active"""
		return self.is_active
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert assignment to dictionary representation"""
		return {
			'assignment_id': self.assignment_id,
			'role_id': self.role_id,
			'permission_id': self.permission_id,
			'tenant_id': self.tenant_id,
			'assigned_by': self.assigned_by,
			'assigned_at': self.assigned_at.isoformat(),
			'assignment_reason': self.assignment_reason,
			'is_active': self.is_active,
			'assignment_metadata': self.assignment_metadata,
			'role_name': self.role.name if self.role else None,
			'permission_display_name': self.permission.display_name if self.permission else None
		}


class ARUserSession(Model, AuditMixin, BaseMixin):
	"""
	Comprehensive session tracking and management.
	"""
	
	__tablename__ = 'ar_user_session'
	
	# Identity
	session_id = Column(String(128), primary_key=True)
	user_id = Column(String(36), ForeignKey('ar_user.user_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Token Management
	jwt_token_id = Column(String(64), unique=True, nullable=False, index=True)
	refresh_token_id = Column(String(64), unique=True, nullable=True, index=True)
	token_family = Column(String(64), nullable=True, index=True)  # For token rotation
	
	# Device Information
	device_fingerprint = Column(String(128), nullable=True, index=True)
	user_agent = Column(Text, nullable=True)
	device_type = Column(String(50), nullable=True)  # desktop, mobile, tablet, api
	device_name = Column(String(200), nullable=True)
	browser_name = Column(String(100), nullable=True)
	browser_version = Column(String(50), nullable=True)
	os_name = Column(String(100), nullable=True)
	os_version = Column(String(50), nullable=True)
	
	# Network Information
	ip_address = Column(String(45), nullable=False, index=True)
	ip_country = Column(String(10), nullable=True)
	ip_region = Column(String(100), nullable=True)
	ip_city = Column(String(100), nullable=True)
	ip_isp = Column(String(200), nullable=True)
	
	# Session Lifecycle
	login_method = Column(String(50), nullable=False)  # password, mfa, sso, api_key, social
	login_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	last_activity_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	expires_at = Column(DateTime, nullable=False, index=True)
	logout_at = Column(DateTime, nullable=True)
	logout_reason = Column(String(50), nullable=True)  # user, timeout, admin, security, system
	
	# Security Information
	is_trusted_device = Column(Boolean, default=False, nullable=False)
	trust_established_at = Column(DateTime, nullable=True)
	security_warnings = Column(JSON, default=list)
	anomaly_score = Column(Float, default=0.0)
	risk_factors = Column(JSON, default=list)
	
	# Session Analytics
	request_count = Column(Integer, default=0)
	data_transferred = Column(Integer, default=0)  # Bytes
	features_accessed = Column(JSON, default=list)
	
	# Metadata
	session_metadata = Column(JSON, default=dict)
	
	# Relationships
	user = relationship("ARUser", back_populates="sessions")
	activities = relationship("ARSessionActivity", back_populates="session", cascade="all, delete-orphan")
	
	# Table constraints
	__table_args__ = (
		Index('ix_ar_user_session_user_active', 'user_id', 'logout_at'),
		Index('ix_ar_user_session_tenant_active', 'tenant_id', 'logout_at'),
		Index('ix_ar_user_session_device', 'device_fingerprint'),
		Index('ix_ar_user_session_ip', 'ip_address'),
		Index('ix_ar_user_session_expires', 'expires_at'),
	)
	
	def is_active(self) -> bool:
		"""Check if session is currently active"""
		if self.logout_at:
			return False
		
		if datetime.utcnow() > self.expires_at:
			return False
		
		return True
	
	def extend_session(self, duration_minutes: int = 60) -> None:
		"""Extend session expiration"""
		self.expires_at = datetime.utcnow() + timedelta(minutes=duration_minutes)
		self.last_activity_at = datetime.utcnow()
	
	def terminate_session(self, reason: str = 'user') -> None:
		"""Terminate session"""
		self.logout_at = datetime.utcnow()
		self.logout_reason = reason
	
	def update_activity(self) -> None:
		"""Update last activity timestamp"""
		self.last_activity_at = datetime.utcnow()
		self.request_count += 1
	
	def add_security_warning(self, warning: str, severity: str = 'medium') -> None:
		"""Add security warning to session"""
		if not self.security_warnings:
			self.security_warnings = []
		
		self.security_warnings.append({
			'warning': warning,
			'severity': severity,
			'timestamp': datetime.utcnow().isoformat()
		})
		
		# Keep only last 10 warnings
		self.security_warnings = self.security_warnings[-10:]
	
	def calculate_anomaly_score(self) -> float:
		"""Calculate session anomaly score based on various factors"""
		score = 0.0
		
		# Time-based anomalies
		now = datetime.utcnow()
		login_hour = self.login_at.hour
		
		# Unusual login hours (weighted)
		if login_hour < 6 or login_hour > 22:
			score += 0.2
		
		# Location-based anomalies
		if self.user and self.user.last_login_ip != self.ip_address:
			score += 0.3
		
		# Device-based anomalies
		if not self.is_trusted_device:
			score += 0.1
		
		# Activity-based anomalies
		session_duration = (now - self.login_at).total_seconds() / 3600  # hours
		if session_duration > 12:  # Very long session
			score += 0.2
		
		if self.request_count > 10000:  # Very high request count
			score += 0.3
		
		# Risk factors
		if self.risk_factors:
			score += len(self.risk_factors) * 0.1
		
		# Cap at 1.0
		self.anomaly_score = min(score, 1.0)
		return self.anomaly_score
	
	def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
		"""Convert session to dictionary representation"""
		data = {
			'session_id': self.session_id,
			'user_id': self.user_id,
			'tenant_id': self.tenant_id,
			'device_type': self.device_type,
			'device_name': self.device_name,
			'browser_name': self.browser_name,
			'browser_version': self.browser_version,
			'os_name': self.os_name,
			'os_version': self.os_version,
			'ip_country': self.ip_country,
			'ip_city': self.ip_city,
			'login_method': self.login_method,
			'login_at': self.login_at.isoformat(),
			'last_activity_at': self.last_activity_at.isoformat(),
			'expires_at': self.expires_at.isoformat(),
			'logout_at': self.logout_at.isoformat() if self.logout_at else None,
			'logout_reason': self.logout_reason,
			'is_trusted_device': self.is_trusted_device,
			'is_active': self.is_active(),
			'request_count': self.request_count,
			'anomaly_score': self.anomaly_score
		}
		
		if include_sensitive:
			data.update({
				'jwt_token_id': self.jwt_token_id,
				'device_fingerprint': self.device_fingerprint,
				'ip_address': self.ip_address,
				'user_agent': self.user_agent,
				'security_warnings': self.security_warnings,
				'risk_factors': self.risk_factors,
				'session_metadata': self.session_metadata
			})
		
		return data


class ARSessionActivity(Model, AuditMixin, BaseMixin):
	"""
	Detailed session activity tracking for security and analytics.
	"""
	
	__tablename__ = 'ar_session_activity'
	
	# Identity
	activity_id = Column(String(36), unique=True, nullable=False, default=uuid7str, primary_key=True)
	session_id = Column(String(128), ForeignKey('ar_user_session.session_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Activity Details
	activity_type = Column(String(50), nullable=False, index=True)  # request, permission_check, feature_access
	activity_name = Column(String(200), nullable=False)
	resource_type = Column(String(100), nullable=True)
	resource_id = Column(String(200), nullable=True)
	
	# Request Information
	http_method = Column(String(10), nullable=True)
	endpoint = Column(String(500), nullable=True)
	status_code = Column(Integer, nullable=True)
	response_time_ms = Column(Integer, nullable=True)
	
	# Activity Metadata
	activity_data = Column(JSON, default=dict)
	
	# Relationships
	session = relationship("ARUserSession", back_populates="activities")
	
	# Table constraints
	__table_args__ = (
		Index('ix_ar_session_activity_type', 'activity_type'),
		Index('ix_ar_session_activity_created', 'created_on'),
	)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert activity to dictionary representation"""
		return {
			'activity_id': self.activity_id,
			'session_id': self.session_id,
			'tenant_id': self.tenant_id,
			'activity_type': self.activity_type,
			'activity_name': self.activity_name,
			'resource_type': self.resource_type,
			'resource_id': self.resource_id,
			'http_method': self.http_method,
			'endpoint': self.endpoint,
			'status_code': self.status_code,
			'response_time_ms': self.response_time_ms,
			'activity_data': self.activity_data,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARLoginAttempt(Model, AuditMixin, BaseMixin):
	"""
	Login attempt tracking for security monitoring and analysis.
	"""
	
	__tablename__ = 'ar_login_attempt'
	
	# Identity
	attempt_id = Column(String(36), unique=True, nullable=False, default=uuid7str, primary_key=True)
	user_id = Column(String(36), ForeignKey('ar_user.user_id'), nullable=True, index=True)  # Null for unknown users
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Attempt Details
	email = Column(String(255), nullable=False, index=True)
	username = Column(String(100), nullable=True)
	attempt_method = Column(String(50), nullable=False)  # password, mfa, sso, api_key
	success = Column(Boolean, nullable=False, index=True)
	failure_reason = Column(String(100), nullable=True)
	
	# Device and Network
	ip_address = Column(String(45), nullable=False, index=True)
	user_agent = Column(Text, nullable=True)
	device_fingerprint = Column(String(128), nullable=True)
	
	# Security Information
	is_suspicious = Column(Boolean, default=False, index=True)
	threat_indicators = Column(JSON, default=list)
	geo_location = Column(JSON, default=dict)
	
	# Metadata
	attempt_metadata = Column(JSON, default=dict)
	
	# Relationships
	user = relationship("ARUser", back_populates="login_attempts")
	
	# Table constraints
	__table_args__ = (
		Index('ix_ar_login_attempt_email_success', 'email', 'success'),
		Index('ix_ar_login_attempt_ip_suspicious', 'ip_address', 'is_suspicious'),
		Index('ix_ar_login_attempt_created', 'created_on'),
	)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert login attempt to dictionary representation"""
		return {
			'attempt_id': self.attempt_id,
			'user_id': self.user_id,
			'tenant_id': self.tenant_id,
			'email': self.email,
			'username': self.username,
			'attempt_method': self.attempt_method,
			'success': self.success,
			'failure_reason': self.failure_reason,
			'ip_address': self.ip_address,
			'is_suspicious': self.is_suspicious,
			'threat_indicators': self.threat_indicators,
			'geo_location': self.geo_location,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}