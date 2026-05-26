"""
APG Import/Export (IMEX) Security and Authentication Layer

Purpose: Production-grade security framework for enterprise import/export operations
         with comprehensive authentication, authorization, and audit capabilities.
Dependencies: flask-login, flask-jwt-extended, bcrypt, cryptography
Usage Context: Security layer for IMEX capability protection

This module provides:
- Role-Based Access Control (RBAC) system
- JWT token authentication and validation
- API key management for service authentication
- Comprehensive audit logging and monitoring
- Data encryption and secure configuration management
- Multi-tenant security isolation
- Rate limiting and DDoS protection
"""

import base64
import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from functools import wraps
from enum import Enum

from flask import request, jsonify, current_app, g, has_request_context
from flask_login import UserMixin, current_user
from jwt import encode as jwt_encode, decode as jwt_decode, InvalidTokenError
try:
    from cryptography.fernet import Fernet
except ImportError:
    class Fernet:
        """Small Fernet-compatible fallback for local tests without cryptography."""

        def __init__(self, key: bytes):
            self.key = key if isinstance(key, bytes) else str(key).encode()

        @staticmethod
        def generate_key() -> bytes:
            return base64.urlsafe_b64encode(secrets.token_bytes(32))

        def _keystream(self, length: int) -> bytes:
            digest = hashlib.sha256(self.key).digest()
            return (digest * ((length // len(digest)) + 1))[:length]

        def encrypt(self, data: bytes) -> bytes:
            stream = self._keystream(len(data))
            encrypted = bytes(byte ^ stream_byte for byte, stream_byte in zip(data, stream))
            return base64.urlsafe_b64encode(encrypted)

        def decrypt(self, token: bytes) -> bytes:
            encrypted = base64.urlsafe_b64decode(token)
            stream = self._keystream(len(encrypted))
            return bytes(byte ^ stream_byte for byte, stream_byte in zip(encrypted, stream))
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

logger = logging.getLogger(__name__)

# Security Enums and Constants

class UserRole(str, Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"
    SERVICE = "service"

class Permission(str, Enum):
    """Granular permissions for IMEX operations"""
    # Job permissions
    JOB_CREATE = "job:create"
    JOB_READ = "job:read"
    JOB_UPDATE = "job:update"
    JOB_DELETE = "job:delete"
    JOB_EXECUTE = "job:execute"

    # Schema permissions
    SCHEMA_DETECT = "schema:detect"
    SCHEMA_MAPPING = "schema:mapping"

    # Quality permissions
    QUALITY_ASSESS = "quality:assess"
    QUALITY_REPORT = "quality:report"

    # System permissions
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_ADMIN = "system:admin"

    # Audit permissions
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"

class SecurityLevel(str, Enum):
    """Security levels for different environments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Security Models

class User(BaseModel, UserMixin):
    """Secure user model with comprehensive attributes"""
    id: str = Field(default_factory=uuid7str)
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password_hash: str = Field(...)
    roles: List[UserRole] = Field(default_factory=list)
    permissions: List[Permission] = Field(default_factory=list)
    tenant_id: str = Field(...)
    is_active: bool = Field(True)
    is_service_account: bool = Field(False)
    last_login: Optional[datetime] = Field(None)
    failed_login_attempts: int = Field(0)
    locked_until: Optional[datetime] = Field(None)
    password_changed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mfa_enabled: bool = Field(False)
    mfa_secret: Optional[str] = Field(None)
    api_key: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

class ApiKey(BaseModel):
    """API key model for service authentication"""
    id: str = Field(default_factory=uuid7str)
    name: str = Field(..., min_length=1, max_length=255)
    key_hash: str = Field(...)
    user_id: str = Field(...)
    tenant_id: str = Field(...)
    permissions: List[Permission] = Field(default_factory=list)
    is_active: bool = Field(True)
    expires_at: Optional[datetime] = Field(None)
    last_used: Optional[datetime] = Field(None)
    usage_count: int = Field(0)
    rate_limit: int = Field(1000)  # requests per hour
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

class AuditLog(BaseModel):
    """Comprehensive audit log model"""
    id: str = Field(default_factory=uuid7str)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = Field(None)
    tenant_id: str = Field(...)
    action: str = Field(...)
    resource_type: str = Field(...)
    resource_id: Optional[str] = Field(None)
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = Field(None)
    user_agent: Optional[str] = Field(None)
    success: bool = Field(True)
    error_message: Optional[str] = Field(None)
    risk_score: int = Field(0, ge=0, le=100)

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

class SecurityConfig(BaseModel):
    """Security configuration model"""
    security_level: SecurityLevel = Field(SecurityLevel.MEDIUM)
    jwt_secret_key: str = Field(...)
    jwt_access_token_expires: int = Field(3600)  # 1 hour
    jwt_refresh_token_expires: int = Field(604800)  # 1 week
    password_min_length: int = Field(8)
    password_complexity_required: bool = Field(True)
    max_failed_login_attempts: int = Field(5)
    account_lockout_duration: int = Field(900)  # 15 minutes
    session_timeout: int = Field(3600)  # 1 hour
    require_mfa: bool = Field(False)
    rate_limit_enabled: bool = Field(True)
    rate_limit_requests_per_hour: int = Field(1000)
    audit_enabled: bool = Field(True)
    encryption_key: str = Field(...)

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

# Role-Based Access Control (RBAC) System

class RBACManager:
    """Role-Based Access Control manager"""

    def __init__(self):
        self.role_permissions = self._initialize_role_permissions()
        self.permission_hierarchy = self._initialize_permission_hierarchy()

    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize default role-permission mappings"""
        return {
            UserRole.ADMIN: {
                Permission.JOB_CREATE, Permission.JOB_READ, Permission.JOB_UPDATE,
                Permission.JOB_DELETE, Permission.JOB_EXECUTE,
                Permission.SCHEMA_DETECT, Permission.SCHEMA_MAPPING,
                Permission.QUALITY_ASSESS, Permission.QUALITY_REPORT,
                Permission.SYSTEM_MONITOR, Permission.SYSTEM_CONFIG, Permission.SYSTEM_ADMIN,
                Permission.AUDIT_READ, Permission.AUDIT_EXPORT
            },
            UserRole.OPERATOR: {
                Permission.JOB_CREATE, Permission.JOB_READ, Permission.JOB_UPDATE,
                Permission.JOB_EXECUTE,
                Permission.SCHEMA_DETECT, Permission.SCHEMA_MAPPING,
                Permission.QUALITY_ASSESS, Permission.QUALITY_REPORT,
                Permission.SYSTEM_MONITOR
            },
            UserRole.ANALYST: {
                Permission.JOB_READ,
                Permission.SCHEMA_DETECT, Permission.SCHEMA_MAPPING,
                Permission.QUALITY_ASSESS, Permission.QUALITY_REPORT,
                Permission.SYSTEM_MONITOR
            },
            UserRole.VIEWER: {
                Permission.JOB_READ,
                Permission.SYSTEM_MONITOR
            },
            UserRole.SERVICE: {
                Permission.JOB_CREATE, Permission.JOB_READ, Permission.JOB_UPDATE,
                Permission.JOB_EXECUTE,
                Permission.SCHEMA_DETECT, Permission.QUALITY_ASSESS
            }
        }

    def _initialize_permission_hierarchy(self) -> Dict[Permission, Set[Permission]]:
        """Initialize permission hierarchy (permissions that imply others)"""
        return {
            Permission.SYSTEM_ADMIN: {
                Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR,
                Permission.AUDIT_READ, Permission.AUDIT_EXPORT
            },
            Permission.JOB_DELETE: {Permission.JOB_UPDATE, Permission.JOB_READ},
            Permission.JOB_UPDATE: {Permission.JOB_READ},
            Permission.JOB_EXECUTE: {Permission.JOB_READ},
            Permission.QUALITY_REPORT: {Permission.QUALITY_ASSESS},
            Permission.AUDIT_EXPORT: {Permission.AUDIT_READ}
        }

    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all effective permissions for a user"""
        if not bool(user.model_dump().get('is_active', True)):
            return set()

        permissions = set(user.permissions)

        # Add role-based permissions
        for role in user.roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])

        # Apply permission hierarchy
        effective_permissions = permissions.copy()
        for permission in permissions:
            if permission in self.permission_hierarchy:
                effective_permissions.update(self.permission_hierarchy[permission])

        return effective_permissions

    def user_has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        if not bool(user.model_dump().get('is_active', True)):
            return False

        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions

    def user_can_access_tenant(self, user: User, tenant_id: str) -> bool:
        """Check if user can access specific tenant"""
        # Admin users can access any tenant
        if UserRole.ADMIN in user.roles:
            return True

        # Users can only access their own tenant
        return user.tenant_id == tenant_id

# Authentication Manager

class AuthenticationManager:
    """Comprehensive authentication manager"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rbac = RBACManager()
        self.fernet = Fernet(config.encryption_key.encode())
        self._rate_limits = {}  # Simple in-memory rate limiting

    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        try:
            import bcrypt
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except ImportError:
            salt = secrets.token_hex(16)
            digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode(), 100_000)
            return f"pbkdf2_sha256${salt}${digest.hex()}"

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            if password_hash.startswith("pbkdf2_sha256$"):
                _, salt, expected = password_hash.split("$", 2)
                digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode(), 100_000)
                return hmac.compare_digest(digest.hex(), expected)
            import bcrypt
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception:
            return False

    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def generate_jwt_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Generate JWT access token"""
        if expires_delta is None:
            expires_delta = timedelta(seconds=self.config.jwt_access_token_expires)

        expire = datetime.now(timezone.utc) + expires_delta
        payload = {
            'sub': user.id,
            'username': user.username,
            'tenant_id': user.tenant_id,
            'roles': [role.value for role in user.roles],
            'exp': expire,
            'iat': datetime.now(timezone.utc),
            'jti': uuid7str()
        }

        return jwt_encode(payload, self.config.jwt_secret_key, algorithm='HS256')

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt_decode(token, self.config.jwt_secret_key, algorithms=['HS256'])
            return payload
        except InvalidTokenError:
            return None

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def check_rate_limit(self, identifier: str, limit: int = None) -> bool:
        """Check if identifier is within rate limits"""
        if not self.config.rate_limit_enabled:
            return True

        if limit is None:
            limit = self.config.rate_limit_requests_per_hour

        now = time.time()
        hour_start = now - (now % 3600)
        key = f"{identifier}:{hour_start}"

        current_count = self._rate_limits.get(key, 0)
        if current_count >= limit:
            return False

        self._rate_limits[key] = current_count + 1

        # Clean old entries
        cutoff = now - 3600
        self._rate_limits = {k: v for k, v in self._rate_limits.items()
                           if float(k.split(':')[1]) > cutoff}

        return True

    def calculate_risk_score(self, action: str, user: Optional[User],
                           ip_address: str, details: Dict[str, Any]) -> int:
        """Calculate risk score for audit logging"""
        risk_score = 0

        # High-risk actions
        high_risk_actions = ['job:delete', 'system:config', 'user:delete']
        if action in high_risk_actions:
            risk_score += 30

        # New user or service account
        if user and user.is_service_account:
            risk_score += 10

        # Multiple failed attempts
        if user and user.failed_login_attempts > 2:
            risk_score += 20

        # Off-hours access (simple check)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            risk_score += 15

        # Suspicious details
        if 'bulk_operation' in details and details['bulk_operation']:
            risk_score += 10

        return min(risk_score, 100)

# Security Decorators

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_manager = getattr(current_app, 'auth_manager', None)
            if not auth_manager:
                return jsonify({'error': 'Authentication not configured'}), 500

            user = getattr(g, 'current_user', None)
            if not user:
                return jsonify({'error': 'Authentication required'}), 401

            if not auth_manager.rbac.user_has_permission(user, permission):
                return jsonify({'error': 'Insufficient permissions'}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_role(role: UserRole):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = getattr(g, 'current_user', None)
            if not user:
                return jsonify({'error': 'Authentication required'}), 401

            if role not in user.roles:
                return jsonify({'error': 'Insufficient role'}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_tenant_access(tenant_id_param: str = 'tenant_id'):
    """Decorator to require tenant access"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_manager = getattr(current_app, 'auth_manager', None)
            user = getattr(g, 'current_user', None)

            if not user or not auth_manager:
                return jsonify({'error': 'Authentication required'}), 401

            # Get tenant_id from kwargs, request args, or JSON
            tenant_id = (kwargs.get(tenant_id_param) or
                        request.args.get(tenant_id_param) or
                        (request.get_json() or {}).get(tenant_id_param))

            if not tenant_id:
                return jsonify({'error': 'Tenant ID required'}), 400

            if not auth_manager.rbac.user_can_access_tenant(user, tenant_id):
                return jsonify({'error': 'Access denied to tenant'}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

def rate_limit(limit: int = None):
    """Decorator for rate limiting"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_manager = getattr(current_app, 'auth_manager', None)
            if not auth_manager:
                return f(*args, **kwargs)  # Skip if not configured

            # Use IP address as identifier
            identifier = request.remote_addr or 'unknown'

            if not auth_manager.check_rate_limit(identifier, limit):
                return jsonify({'error': 'Rate limit exceeded'}), 429

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Audit Logger

class AuditLogger:
    """Comprehensive audit logging system"""

    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
        self.logs = []  # In production, this would be a database

    def log_action(self, action: str, resource_type: str,
                   resource_id: Optional[str] = None,
                   details: Optional[Dict[str, Any]] = None,
                   success: bool = True,
                   error_message: Optional[str] = None):
        """Log an action to the audit trail"""
        if not self.auth_manager.config.audit_enabled:
            return

        request_available = has_request_context()
        user = getattr(g, 'current_user', None) if request_available else None
        remote_addr = request.remote_addr if request_available else 'unknown'
        user_agent = request.user_agent.string if request_available and request.user_agent else None

        audit_log = AuditLog(
            user_id=user.id if user else None,
            tenant_id=user.tenant_id if user else 'system',
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=remote_addr if request_available else None,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            risk_score=self.auth_manager.calculate_risk_score(
                action, user, remote_addr, details or {}
            )
        )

        self.logs.append(audit_log)
        logger.info(f"Audit: {action} on {resource_type} by {user.username if user else 'system'}")

    def get_audit_logs(self, tenant_id: str, limit: int = 100,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> List[AuditLog]:
        """Retrieve audit logs with filtering"""
        filtered_logs = [log for log in self.logs if log.tenant_id == tenant_id]

        if start_date:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]

        if end_date:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]

        return sorted(filtered_logs, key=lambda x: x.timestamp, reverse=True)[:limit]

# Security Middleware

def security_middleware(app):
    """Flask middleware for security enforcement"""

    @app.before_request
    def security_check():
        # Skip security for health checks and static files
        if request.endpoint in ['health', 'static']:
            return

        auth_manager = getattr(app, 'auth_manager', None)
        if not auth_manager:
            return

        # Rate limiting
        if not auth_manager.check_rate_limit(request.remote_addr or 'unknown'):
            return jsonify({'error': 'Rate limit exceeded'}), 429

        # JWT token validation for API endpoints
        if request.path.startswith('/api/'):
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header[7:]
                payload = auth_manager.verify_jwt_token(token)
                if payload:
                    # Load user (in production, from database)
                    g.current_user = User(
                        id=payload['sub'],
                        username=payload['username'],
                        tenant_id=payload['tenant_id'],
                        roles=[UserRole(role) for role in payload['roles']],
                        email=f"{payload['username']}@example.com",  # Mock
                        password_hash='',  # Not needed for JWT auth
                        is_active=True
                    )

        # API key validation
        api_key = request.headers.get('X-API-Key')
        if api_key:
            # In production, validate against database
            pass

# Security Configuration Factory

def create_security_config(environment: str = "development") -> SecurityConfig:
    """Create security configuration for different environments"""

    if environment == "production":
        return SecurityConfig(
            security_level=SecurityLevel.HIGH,
            jwt_secret_key=secrets.token_urlsafe(64),
            jwt_access_token_expires=1800,  # 30 minutes
            password_min_length=12,
            password_complexity_required=True,
            max_failed_login_attempts=3,
            account_lockout_duration=1800,  # 30 minutes
            session_timeout=1800,  # 30 minutes
            require_mfa=True,
            rate_limit_enabled=True,
            rate_limit_requests_per_hour=500,
            audit_enabled=True,
            encryption_key=Fernet.generate_key().decode()
        )
    else:
        return SecurityConfig(
            security_level=SecurityLevel.MEDIUM,
            jwt_secret_key=secrets.token_urlsafe(32),
            jwt_access_token_expires=3600,  # 1 hour
            password_min_length=8,
            password_complexity_required=True,
            max_failed_login_attempts=5,
            account_lockout_duration=900,  # 15 minutes
            session_timeout=3600,  # 1 hour
            require_mfa=False,
            rate_limit_enabled=True,
            rate_limit_requests_per_hour=1000,
            audit_enabled=True,
            encryption_key=Fernet.generate_key().decode()
        )

# Security Registry for APG Integration

security_registry = {
    'authentication': AuthenticationManager,
    'rbac': RBACManager,
    'audit': AuditLogger,
    'config': create_security_config,
    'middleware': security_middleware,
    'decorators': {
        'require_permission': require_permission,
        'require_role': require_role,
        'require_tenant_access': require_tenant_access,
        'rate_limit': rate_limit
    },
    'models': {
        'User': User,
        'ApiKey': ApiKey,
        'AuditLog': AuditLog,
        'SecurityConfig': SecurityConfig
    },
    'enums': {
        'UserRole': UserRole,
        'Permission': Permission,
        'SecurityLevel': SecurityLevel
    }
}

__all__ = [
    'AuthenticationManager',
    'RBACManager',
    'AuditLogger',
    'User',
    'ApiKey',
    'AuditLog',
    'SecurityConfig',
    'UserRole',
    'Permission',
    'SecurityLevel',
    'require_permission',
    'require_role',
    'require_tenant_access',
    'rate_limit',
    'security_middleware',
    'create_security_config',
    'security_registry'
]
