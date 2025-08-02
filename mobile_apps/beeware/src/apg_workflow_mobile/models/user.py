"""
User model and related data structures

© 2025 Datacraft. All rights reserved.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str


class UserRole(str, Enum):
	"""User role enumeration"""
	ADMIN = "admin"
	MANAGER = "manager"
	DEVELOPER = "developer"
	ANALYST = "analyst"
	USER = "user"
	GUEST = "guest"


@dataclass
class BiometricConfig:
	"""Biometric authentication configuration"""
	is_enabled: bool = False
	available_methods: List[str] = field(default_factory=list)
	selected_method: Optional[str] = None
	fallback_enabled: bool = True
	last_enrolled: Optional[datetime] = None


class User(BaseModel):
	"""User model"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	username: str = Field(..., min_length=3, max_length=50)
	email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
	first_name: str = Field(..., min_length=1, max_length=50)
	last_name: str = Field(..., min_length=1, max_length=50)
	full_name: Optional[str] = None
	role: UserRole = UserRole.USER
	tenant_id: str = Field(..., min_length=1)
	tenant_name: Optional[str] = None
	
	# Profile information
	avatar_url: Optional[str] = None
	phone: Optional[str] = None
	department: Optional[str] = None
	job_title: Optional[str] = None
	manager_id: Optional[str] = None
	location: Optional[str] = None
	timezone: str = "UTC"
	language: str = "en"
	
	# Authentication
	is_active: bool = True
	is_verified: bool = False
	last_login: Optional[datetime] = None
	password_changed_at: Optional[datetime] = None
	must_change_password: bool = False
	two_factor_enabled: bool = False
	
	# Biometric authentication
	biometric_config: BiometricConfig = field(default_factory=BiometricConfig)
	
	# Permissions and capabilities
	permissions: List[str] = Field(default_factory=list)
	groups: List[str] = Field(default_factory=list)
	
	# Workflow-specific
	workflow_quota: int = 100
	workflow_count: int = 0
	can_create_workflows: bool = True
	can_assign_tasks: bool = True
	can_approve_workflows: bool = False
	
	# Preferences
	preferences: Dict[str, Any] = Field(default_factory=dict)
	notification_settings: Dict[str, bool] = Field(
		default_factory=lambda: {
			"email_notifications": True,
			"push_notifications": True,
			"workflow_updates": True,
			"task_assignments": True,
			"system_alerts": True,
		}
	)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: Optional[str] = None
	updated_by: Optional[str] = None
	
	def __post_init__(self):
		"""Post-initialization processing"""
		if not self.full_name:
			self.full_name = f"{self.first_name} {self.last_name}"
	
	@property
	def display_name(self) -> str:
		"""Get user's display name"""
		return self.full_name or f"{self.first_name} {self.last_name}"
	
	@property
	def initials(self) -> str:
		"""Get user's initials"""
		return f"{self.first_name[0]}{self.last_name[0]}".upper()
	
	@property
	def is_admin(self) -> bool:
		"""Check if user is admin"""
		return self.role == UserRole.ADMIN
	
	@property
	def is_manager(self) -> bool:
		"""Check if user is manager or admin"""
		return self.role in [UserRole.ADMIN, UserRole.MANAGER]
	
	@property
	def can_manage_users(self) -> bool:
		"""Check if user can manage other users"""
		return "manage_users" in self.permissions or self.is_admin
	
	@property
	def can_view_analytics(self) -> bool:
		"""Check if user can view analytics"""
		return "view_analytics" in self.permissions or self.is_manager
	
	def has_permission(self, permission: str) -> bool:
		"""Check if user has specific permission"""
		return permission in self.permissions or self.is_admin
	
	def is_in_group(self, group: str) -> bool:
		"""Check if user is in specific group"""
		return group in self.groups
	
	def can_access_workflow(self, workflow_permissions: List[str]) -> bool:
		"""Check if user can access workflow based on permissions"""
		if self.is_admin:
			return True
		
		return any(perm in self.permissions for perm in workflow_permissions)
	
	def update_last_login(self):
		"""Update last login timestamp"""
		self.last_login = datetime.utcnow()
		self.updated_at = datetime.utcnow()
	
	def enable_biometric(self, method: str):
		"""Enable biometric authentication"""
		self.biometric_config.is_enabled = True
		self.biometric_config.selected_method = method
		self.biometric_config.last_enrolled = datetime.utcnow()
		self.updated_at = datetime.utcnow()
	
	def disable_biometric(self):
		"""Disable biometric authentication"""
		self.biometric_config.is_enabled = False
		self.biometric_config.selected_method = None
		self.updated_at = datetime.utcnow()
	
	def update_preferences(self, preferences: Dict[str, Any]):
		"""Update user preferences"""
		self.preferences.update(preferences)
		self.updated_at = datetime.utcnow()
	
	def update_notification_settings(self, settings: Dict[str, bool]):
		"""Update notification settings"""
		self.notification_settings.update(settings)
		self.updated_at = datetime.utcnow()
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert user to dictionary"""
		return {
			"id": self.id,
			"username": self.username,
			"email": self.email,
			"first_name": self.first_name,
			"last_name": self.last_name,
			"full_name": self.full_name,
			"display_name": self.display_name,
			"initials": self.initials,
			"role": self.role.value,
			"tenant_id": self.tenant_id,
			"tenant_name": self.tenant_name,
			"avatar_url": self.avatar_url,
			"phone": self.phone,
			"department": self.department,
			"job_title": self.job_title,
			"location": self.location,
			"timezone": self.timezone,
			"language": self.language,
			"is_active": self.is_active,
			"is_verified": self.is_verified,
			"last_login": self.last_login.isoformat() if self.last_login else None,
			"permissions": self.permissions,
			"groups": self.groups,
			"preferences": self.preferences,
			"notification_settings": self.notification_settings,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat(),
		}
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "User":
		"""Create user from dictionary"""
		# Handle datetime fields
		if "last_login" in data and data["last_login"]:
			data["last_login"] = datetime.fromisoformat(data["last_login"])
		if "created_at" in data:
			data["created_at"] = datetime.fromisoformat(data["created_at"])
		if "updated_at" in data:
			data["updated_at"] = datetime.fromisoformat(data["updated_at"])
		
		# Handle role enum
		if "role" in data and isinstance(data["role"], str):
			data["role"] = UserRole(data["role"])
		
		return cls(**data)


@dataclass
class AuthToken:
	"""Authentication token data"""
	access_token: str
	refresh_token: str
	token_type: str = "Bearer"
	expires_in: int = 3600
	expires_at: Optional[datetime] = None
	scope: Optional[str] = None
	
	def __post_init__(self):
		if not self.expires_at and self.expires_in:
			self.expires_at = datetime.utcnow().timestamp() + self.expires_in
	
	@property
	def is_expired(self) -> bool:
		"""Check if token is expired"""
		if not self.expires_at:
			return False
		return datetime.utcnow().timestamp() >= self.expires_at
	
	@property
	def expires_soon(self, threshold: int = 300) -> bool:
		"""Check if token expires within threshold seconds"""
		if not self.expires_at:
			return False
		return datetime.utcnow().timestamp() >= (self.expires_at - threshold)


@dataclass
class LoginCredentials:
	"""Login credentials"""
	username: str
	password: str
	tenant_id: str
	remember_me: bool = False
	biometric_signature: Optional[str] = None


@dataclass 
class AuthResponse:
	"""Authentication response"""
	user: User
	token: AuthToken
	permissions: List[str]
	message: Optional[str] = None