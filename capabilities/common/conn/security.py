"""
APG Connection Management Security Module
Comprehensive security controls, authentication, authorization, and data protection

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import hashlib
import secrets
import jwt
try:
    import bcrypt
except ImportError:
    class _BcryptCompat:
        """Local password hashing fallback for test/offline environments."""

        @staticmethod
        def gensalt() -> bytes:
            return secrets.token_hex(16).encode("utf-8")

        @staticmethod
        def hashpw(password: bytes, salt: bytes) -> bytes:
            digest = hashlib.pbkdf2_hmac("sha256", password, salt, 100000)
            return b"compat$" + salt + b"$" + digest.hex().encode("utf-8")

        @staticmethod
        def checkpw(password: bytes, hashed: bytes) -> bool:
            try:
                _, salt, expected = hashed.split(b"$", 2)
            except ValueError:
                return False
            actual = hashlib.pbkdf2_hmac("sha256", password, salt, 100000).hex().encode("utf-8")
            return secrets.compare_digest(actual, expected)

    bcrypt = _BcryptCompat()
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from functools import wraps
import re
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from .error_handling import APGError, AuthenticationError, AuthorizationError, ErrorContext

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AccessAction(str, Enum):
    """Types of access actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """Types of resources that can be secured"""
    CONNECTION = "connection"
    FLOW = "flow"
    SCHEMA = "schema"
    DATA = "data"
    CONFIGURATION = "configuration"
    MONITORING = "monitoring"
    COMPOSITION = "composition"
    SYSTEM = "system"


@dataclass
class Permission:
    """Individual permission definition"""
    resource_type: ResourceType
    action: AccessAction
    resource_id: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Role:
    """User role definition"""
    name: str
    description: str
    permissions: List[Permission] = field(default_factory=list)
    inherits_from: List[str] = field(default_factory=list)


@dataclass
class User:
    """User identity and authentication info"""
    user_id: str
    username: str
    email: str
    tenant_id: str
    roles: List[str] = field(default_factory=list)
    is_active: bool = True
    is_admin: bool = False
    password_hash: Optional[str] = None
    api_key_hash: Optional[str] = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityContext:
    """Security context for requests"""
    user: User
    tenant_id: str
    session_id: str
    ip_address: str
    user_agent: str
    request_id: str
    authenticated_at: datetime
    expires_at: datetime
    permissions: Set[str] = field(default_factory=set)


@dataclass
class AuditEvent:
    """Security audit event"""
    event_id: str
    timestamp: datetime
    user_id: str
    tenant_id: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class PasswordPolicy:
    """Password policy enforcement"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_length = self.config.get('min_length', 12)
        self.require_uppercase = self.config.get('require_uppercase', True)
        self.require_lowercase = self.config.get('require_lowercase', True)
        self.require_digits = self.config.get('require_digits', True)
        self.require_symbols = self.config.get('require_symbols', True)
        self.max_age_days = self.config.get('max_age_days', 90)
        self.history_count = self.config.get('history_count', 12)

    def validate_password(self, password: str, user: User = None) -> List[str]:
        """Validate password against policy"""
        errors = []

        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")

        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if self.require_digits and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if self.require_symbols and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")

        # Check for common patterns
        if self._contains_common_patterns(password):
            errors.append("Password contains common patterns and is not secure")

        # Check against user information if provided
        if user and self._contains_user_info(password, user):
            errors.append("Password must not contain personal information")

        return errors

    def _contains_common_patterns(self, password: str) -> bool:
        """Check for common password patterns"""
        common_patterns = [
            r'123456', r'password', r'qwerty', r'admin', r'letmein',
            r'welcome', r'monkey', r'dragon', r'master', r'shadow'
        ]

        password_lower = password.lower()
        return any(re.search(pattern, password_lower) for pattern in common_patterns)

    def _contains_user_info(self, password: str, user: User) -> bool:
        """Check if password contains user information"""
        password_lower = password.lower()

        # Check username
        if user.username.lower() in password_lower:
            return True

        # Check email parts
        if '@' in user.email:
            email_parts = user.email.lower().split('@')
            if any(part in password_lower for part in email_parts):
                return True

        return False


class EncryptionManager:
    """Handles encryption and decryption of sensitive data"""

    def __init__(self, key: bytes = None):
        if key is None:
            # Generate a key from environment or create new one
            key_b64 = os.environ.get('APG_ENCRYPTION_KEY')
            if key_b64:
                self.key = base64.urlsafe_b64decode(key_b64)
            else:
                # Generate new key (should be stored securely in production)
                password = os.environ.get('APG_MASTER_PASSWORD', 'default-dev-password').encode()
                salt = os.environ.get('APG_ENCRYPTION_SALT', 'default-salt').encode()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                self.key = kdf.derive(password)
        else:
            self.key = key

        self.cipher = Fernet(base64.urlsafe_b64encode(self.key))

    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = self.cipher.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            raise APGError(
                message=f"Failed to decrypt data: {str(e)}",
                context=ErrorContext(tenant_id="unknown", operation="decrypt_data"),
                cause=e
            )

    def encrypt_connection_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in connection configuration"""
        sensitive_fields = ['password', 'api_key', 'secret', 'token', 'private_key']
        encrypted_config = config.copy()

        for field in sensitive_fields:
            if field in encrypted_config:
                encrypted_config[field] = self.encrypt(str(encrypted_config[field]))

        return encrypted_config

    def decrypt_connection_config(self, encrypted_config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in connection configuration"""
        sensitive_fields = ['password', 'api_key', 'secret', 'token', 'private_key']
        decrypted_config = encrypted_config.copy()

        for field in sensitive_fields:
            if field in decrypted_config:
                try:
                    decrypted_config[field] = self.decrypt(decrypted_config[field])
                except Exception:
                    # If decryption fails, assume it's already decrypted
                    pass

        return decrypted_config


class AuthenticationManager:
    """Handles user authentication and session management"""

    def __init__(self, encryption_manager: EncryptionManager = None):
        self.encryption_manager = encryption_manager or EncryptionManager()
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, SecurityContext] = {}
        self.password_policy = PasswordPolicy()
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 30
        self.session_timeout_hours = 8
        self.jwt_secret = os.environ.get('APG_JWT_SECRET', 'dev-secret-key')

    def create_user(self, username: str, email: str, password: str, tenant_id: str,
                   roles: List[str] = None, is_admin: bool = False) -> User:
        """Create a new user"""

        # Validate password
        validation_errors = self.password_policy.validate_password(password)
        if validation_errors:
            raise AuthenticationError(
                message=f"Password validation failed: {'; '.join(validation_errors)}",
                context=ErrorContext(tenant_id=tenant_id, operation="create_user")
            )

        # Check if user already exists
        if any(u.username == username for u in self.users.values()):
            raise AuthenticationError(
                message=f"Username '{username}' already exists",
                context=ErrorContext(tenant_id=tenant_id, operation="create_user")
            )

        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create user
        user_id = secrets.token_urlsafe(16)
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            tenant_id=tenant_id,
            roles=roles or [],
            is_admin=is_admin,
            password_hash=password_hash.decode('utf-8')
        )

        self.users[user_id] = user
        logger.info(f"Created user: {username} (tenant: {tenant_id})")

        return user

    def authenticate_user(self, username: str, password: str, ip_address: str = None,
                         user_agent: str = None) -> SecurityContext:
        """Authenticate user and create security context"""

        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user:
            raise AuthenticationError(
                message="Invalid username or password",
                context=ErrorContext(tenant_id="unknown", operation="authenticate_user")
            )

        # Check if account is locked
        if user.account_locked_until and datetime.now(timezone.utc) < user.account_locked_until:
            raise AuthenticationError(
                message=f"Account is locked until {user.account_locked_until}",
                context=ErrorContext(tenant_id=user.tenant_id, user_id=user.user_id, operation="authenticate_user")
            )

        # Verify password
        if not user.password_hash or not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            # Increment failed attempts
            user.failed_login_attempts += 1

            if user.failed_login_attempts >= self.max_login_attempts:
                # Lock account
                user.account_locked_until = datetime.now(timezone.utc) + timedelta(minutes=self.lockout_duration_minutes)
                logger.warning(f"Account locked for user {username} due to too many failed attempts")

            raise AuthenticationError(
                message="Invalid username or password",
                context=ErrorContext(tenant_id=user.tenant_id, user_id=user.user_id, operation="authenticate_user")
            )

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.account_locked_until = None
        user.last_login = datetime.now(timezone.utc)

        # Create session
        return self._create_session(user, ip_address, user_agent)

    def authenticate_api_key(self, api_key: str, ip_address: str = None) -> SecurityContext:
        """Authenticate using API key"""

        # Hash the provided API key to compare with stored hashes
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Find user by API key hash
        user = None
        for u in self.users.values():
            if u.api_key_hash == api_key_hash:
                user = u
                break

        if not user or not user.is_active:
            raise AuthenticationError(
                message="Invalid API key",
                context=ErrorContext(tenant_id="unknown", operation="authenticate_api_key")
            )

        # Create session for API key authentication
        return self._create_session(user, ip_address, "API Client")

    def _create_session(self, user: User, ip_address: str = None, user_agent: str = None) -> SecurityContext:
        """Create security context and session"""

        session_id = secrets.token_urlsafe(32)
        request_id = secrets.token_urlsafe(16)

        context = SecurityContext(
            user=user,
            tenant_id=user.tenant_id,
            session_id=session_id,
            ip_address=ip_address or "unknown",
            user_agent=user_agent or "unknown",
            request_id=request_id,
            authenticated_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=self.session_timeout_hours)
        )

        self.sessions[session_id] = context
        return context

    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate session and return security context"""

        context = self.sessions.get(session_id)
        if not context:
            return None

        # Check if session is expired
        if datetime.now(timezone.utc) > context.expires_at:
            del self.sessions[session_id]
            return None

        # Check if user is still active
        if not context.user.is_active:
            del self.sessions[session_id]
            return None

        return context

    def generate_jwt_token(self, user: User, expires_hours: int = 24) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'tenant_id': user.tenant_id,
            'roles': user.roles,
            'is_admin': user.is_admin,
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        }

        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError(
                message="Token has expired",
                context=ErrorContext(tenant_id="unknown", operation="validate_jwt_token")
            )
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(
                message=f"Invalid token: {str(e)}",
                context=ErrorContext(tenant_id="unknown", operation="validate_jwt_token"),
                cause=e
            )

    def logout_user(self, session_id: str):
        """Logout user and invalidate session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def generate_api_key(self, user: User) -> str:
        """Generate API key for user"""
        api_key = secrets.token_urlsafe(32)
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user.api_key_hash = api_key_hash

        logger.info(f"Generated API key for user {user.username}")
        return api_key


class AuthorizationManager:
    """Handles role-based access control and permissions"""

    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.audit_events: List[AuditEvent] = []
        self._setup_default_roles()

    def _setup_default_roles(self):
        """Setup default roles and permissions"""

        # Viewer role - read-only access
        viewer_role = Role(
            name="viewer",
            description="Read-only access to connections and flows",
            permissions=[
                Permission(ResourceType.CONNECTION, AccessAction.READ),
                Permission(ResourceType.FLOW, AccessAction.READ),
                Permission(ResourceType.SCHEMA, AccessAction.READ),
                Permission(ResourceType.MONITORING, AccessAction.READ),
            ]
        )

        # Operator role - can create and manage connections and flows
        operator_role = Role(
            name="operator",
            description="Can create and manage connections and flows",
            permissions=[
                Permission(ResourceType.CONNECTION, AccessAction.CREATE),
                Permission(ResourceType.CONNECTION, AccessAction.READ),
                Permission(ResourceType.CONNECTION, AccessAction.UPDATE),
                Permission(ResourceType.FLOW, AccessAction.CREATE),
                Permission(ResourceType.FLOW, AccessAction.READ),
                Permission(ResourceType.FLOW, AccessAction.UPDATE),
                Permission(ResourceType.FLOW, AccessAction.EXECUTE),
                Permission(ResourceType.SCHEMA, AccessAction.READ),
                Permission(ResourceType.MONITORING, AccessAction.READ),
            ],
            inherits_from=["viewer"]
        )

        # Admin role - full access
        admin_role = Role(
            name="admin",
            description="Full administrative access",
            permissions=[
                Permission(ResourceType.CONNECTION, AccessAction.ADMIN),
                Permission(ResourceType.FLOW, AccessAction.ADMIN),
                Permission(ResourceType.SCHEMA, AccessAction.ADMIN),
                Permission(ResourceType.DATA, AccessAction.ADMIN),
                Permission(ResourceType.CONFIGURATION, AccessAction.ADMIN),
                Permission(ResourceType.MONITORING, AccessAction.ADMIN),
                Permission(ResourceType.COMPOSITION, AccessAction.ADMIN),
                Permission(ResourceType.SYSTEM, AccessAction.ADMIN),
            ],
            inherits_from=["operator"]
        )

        # Store roles
        self.roles["viewer"] = viewer_role
        self.roles["operator"] = operator_role
        self.roles["admin"] = admin_role

    def has_permission(self, context: SecurityContext, resource_type: ResourceType,
                      action: AccessAction, resource_id: str = None) -> bool:
        """Check if user has permission for specific action"""

        # Admin users have all permissions
        if context.user.is_admin:
            return True

        # Check user roles
        user_permissions = self._get_user_permissions(context.user)

        # Check for exact permission match
        permission_key = f"{resource_type.value}:{action.value}"
        if resource_id:
            permission_key += f":{resource_id}"

        if permission_key in user_permissions:
            return True

        # Check for wildcard permissions
        wildcard_key = f"{resource_type.value}:{action.value}:*"
        if wildcard_key in user_permissions:
            return True

        # Check for admin permission on resource type
        admin_key = f"{resource_type.value}:{AccessAction.ADMIN.value}"
        if admin_key in user_permissions:
            return True

        return False

    def _get_user_permissions(self, user: User) -> Set[str]:
        """Get all permissions for a user including inherited roles"""
        permissions = set()

        def add_role_permissions(role_name: str, visited: Set[str] = None):
            if visited is None:
                visited = set()

            if role_name in visited or role_name not in self.roles:
                return

            visited.add(role_name)
            role = self.roles[role_name]

            # Add direct permissions
            for permission in role.permissions:
                perm_key = f"{permission.resource_type.value}:{permission.action.value}"
                if permission.resource_id:
                    perm_key += f":{permission.resource_id}"
                permissions.add(perm_key)

            # Add inherited permissions
            for inherited_role in role.inherits_from:
                add_role_permissions(inherited_role, visited)

        # Process all user roles
        for role_name in user.roles:
            add_role_permissions(role_name)

        return permissions

    def require_permission(self, resource_type: ResourceType, action: AccessAction,
                          resource_id: str = None):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract security context from arguments
                context = None
                for arg in args:
                    if isinstance(arg, SecurityContext):
                        context = arg
                        break

                if not context:
                    raise AuthorizationError(
                        message="No security context provided",
                        context=ErrorContext(tenant_id="unknown", operation=func.__name__)
                    )

                # Check permission
                if not self.has_permission(context, resource_type, action, resource_id):
                    self._audit_access_denied(context, resource_type, action, resource_id)
                    raise AuthorizationError(
                        message=f"Access denied: {action.value} on {resource_type.value}",
                        required_permission=f"{resource_type.value}:{action.value}",
                        context=ErrorContext(
                            tenant_id=context.tenant_id,
                            user_id=context.user.user_id,
                            operation=func.__name__
                        )
                    )

                # Audit successful access
                self._audit_access_granted(context, resource_type, action, resource_id)

                return func(*args, **kwargs)

            return wrapper
        return decorator

    def _audit_access_granted(self, context: SecurityContext, resource_type: ResourceType,
                            action: AccessAction, resource_id: str = None):
        """Audit successful access"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=datetime.now(timezone.utc),
            user_id=context.user.user_id,
            tenant_id=context.tenant_id,
            action=f"{action.value}_{resource_type.value}",
            resource_type=resource_type.value,
            resource_id=resource_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            success=True,
            details={
                'session_id': context.session_id,
                'request_id': context.request_id
            }
        )

        self.audit_events.append(event)
        logger.info(f"Access granted: {context.user.username} -> {action.value} {resource_type.value}")

    def _audit_access_denied(self, context: SecurityContext, resource_type: ResourceType,
                           action: AccessAction, resource_id: str = None):
        """Audit denied access"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=datetime.now(timezone.utc),
            user_id=context.user.user_id,
            tenant_id=context.tenant_id,
            action=f"{action.value}_{resource_type.value}",
            resource_type=resource_type.value,
            resource_id=resource_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            success=False,
            details={
                'session_id': context.session_id,
                'request_id': context.request_id,
                'reason': 'insufficient_permissions'
            }
        )

        self.audit_events.append(event)
        logger.warning(f"Access denied: {context.user.username} -> {action.value} {resource_type.value}")

    def get_audit_events(self, tenant_id: str = None, user_id: str = None,
                        limit: int = 100) -> List[AuditEvent]:
        """Get audit events with optional filtering"""
        events = self.audit_events

        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]

        if user_id:
            events = [e for e in events if e.user_id == user_id]

        # Sort by timestamp (most recent first) and limit
        events = sorted(events, key=lambda x: x.timestamp, reverse=True)
        return events[:limit]


# Global instances
encryption_manager = EncryptionManager()
auth_manager = AuthenticationManager(encryption_manager)
authz_manager = AuthorizationManager()


# Convenience decorators and functions
def require_authentication(func):
    """Decorator to require user authentication"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # This would typically extract session from Flask request
        # For now, we'll check if first argument is a SecurityContext
        if args and isinstance(args[0], SecurityContext):
            return func(*args, **kwargs)
        else:
            raise AuthenticationError(
                message="Authentication required",
                context=ErrorContext(tenant_id="unknown", operation=func.__name__)
            )
    return wrapper


def require_admin(func):
    """Decorator to require admin access"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        context = None
        for arg in args:
            if isinstance(arg, SecurityContext):
                context = arg
                break

        if not context:
            raise AuthenticationError(
                message="No security context provided",
                context=ErrorContext(tenant_id="unknown", operation=func.__name__)
            )

        if not context.user.is_admin:
            raise AuthorizationError(
                message="Admin access required",
                context=ErrorContext(
                    tenant_id=context.tenant_id,
                    user_id=context.user.user_id,
                    operation=func.__name__
                )
            )

        return func(*args, **kwargs)
    return wrapper


def require_tenant_access(func):
    """Decorator to ensure user can only access their tenant's resources"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        context = None
        tenant_id = kwargs.get('tenant_id')

        for arg in args:
            if isinstance(arg, SecurityContext):
                context = arg
                break

        if not context:
            raise AuthenticationError(
                message="No security context provided",
                context=ErrorContext(tenant_id="unknown", operation=func.__name__)
            )

        # Admin users can access any tenant
        if context.user.is_admin:
            return func(*args, **kwargs)

        # Regular users can only access their own tenant
        if tenant_id and tenant_id != context.tenant_id:
            raise AuthorizationError(
                message="Access denied: Cannot access resources from different tenant",
                context=ErrorContext(
                    tenant_id=context.tenant_id,
                    user_id=context.user.user_id,
                    operation=func.__name__
                )
            )

        return func(*args, **kwargs)
    return wrapper


# Utility functions
def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for secure storage"""
    return hashlib.sha256(data.encode()).hexdigest()


def mask_sensitive_string(value: str, visible_chars: int = 4) -> str:
    """Mask sensitive strings for display"""
    if len(value) <= visible_chars:
        return '*' * len(value)
    return value[:visible_chars] + '*' * (len(value) - visible_chars)


def classify_data_sensitivity(data: Dict[str, Any]) -> SecurityLevel:
    """Classify data sensitivity level"""
    sensitive_patterns = [
        r'password', r'secret', r'key', r'token', r'credential',
        r'ssn', r'social.*security', r'credit.*card', r'account.*number',
        r'private', r'confidential', r'restricted'
    ]

    data_str = json.dumps(data).lower()

    for pattern in sensitive_patterns:
        if re.search(pattern, data_str):
            return SecurityLevel.CONFIDENTIAL

    return SecurityLevel.INTERNAL


def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from logs"""
    sensitive_keys = [
        'password', 'secret', 'key', 'token', 'credential',
        'private_key', 'api_key', 'auth_token'
    ]

    sanitized = {}
    for key, value in data.items():
        if key.lower() in sensitive_keys:
            sanitized[key] = mask_sensitive_string(str(value))
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        else:
            sanitized[key] = value

    return sanitized
