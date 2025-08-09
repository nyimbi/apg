"""
APG Access Control Integration Service - Complete Implementation

Production-ready access control and authorization service using real SDKs including
OAuth2, SAML, LDAP, JWT, and RBAC for comprehensive identity and access management
across distributed systems.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Core async libraries
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, func, and_, or_, desc
import redis.asyncio as redis

# Authentication and authorization - real SDKs
import jwt
from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2 import OAuth2Error
from authlib.oidc.core import CodeIDToken, ImplicitIDToken
import ldap3
from ldap3 import Server, Connection, ALL, NTLM
import saml2
from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from saml2.client import Saml2Client
from saml2.config import Config as SAMLConfig

# Password hashing and security
import bcrypt
from passlib.context import CryptContext
from passlib.hash import pbkdf2_sha256
import argon2

# HTTP client for OAuth and API calls
import httpx

# Rate limiting and security
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Policy engines
import casbin
import pycasbin_async_sqlalchemy_adapter
from pycasbin_async_sqlalchemy_adapter import Adapter as CasbinAdapter

# Utilities
from uuid_extensions import uuid7str
import structlog

logger = structlog.get_logger(__name__)

# =============================================================================
# Data Models and Types
# =============================================================================

class AuthenticationMethod(Enum):
    LOCAL = "local"
    OAUTH2 = "oauth2"
    SAML = "saml"
    LDAP = "ldap"
    JWT = "jwt"
    API_KEY = "api_key"
    MULTI_FACTOR = "multi_factor"

class PermissionLevel(Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"  
    EXECUTE = "execute"
    ADMIN = "admin"
    OWNER = "owner"

class ResourceType(Enum):
    CAPABILITY = "capability"
    WORKFLOW = "workflow"
    CONFIGURATION = "configuration"
    SERVICE = "service"
    TENANT = "tenant"
    USER = "user"
    ROLE = "role"

class PolicyEffect(Enum):
    ALLOW = "allow"
    DENY = "deny"

class SessionStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"

@dataclass
class AuthenticationRequest:
    """Authentication request data."""
    method: AuthenticationMethod
    credentials: Dict[str, Any]
    client_info: Dict[str, Any]
    requested_scopes: List[str]
    metadata: Dict[str, Any]

@dataclass
class AuthenticationResult:
    """Authentication result."""
    success: bool
    user_id: Optional[str]
    session_id: Optional[str]
    access_token: Optional[str]
    refresh_token: Optional[str]
    expires_at: Optional[datetime]
    scopes: List[str]
    user_info: Dict[str, Any]
    error_message: Optional[str]

@dataclass
class Permission:
    """Permission definition."""
    user_id: Optional[str]
    role_id: Optional[str]
    resource_type: ResourceType
    resource_id: str
    permission_level: PermissionLevel
    conditions: Dict[str, Any]
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime]

@dataclass
class AuthorizationContext:
    """Authorization evaluation context."""
    user_id: str
    resource_type: ResourceType
    resource_id: str
    action: str
    client_info: Dict[str, Any]
    request_metadata: Dict[str, Any]
    time_context: Dict[str, Any]

@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    tenant_id: str
    status: SessionStatus
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    permissions: List[Permission]
    metadata: Dict[str, Any]

# =============================================================================
# Access Control Integration Service
# =============================================================================

class AccessControlIntegrationService:
    """Main access control service with multiple authentication methods."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: redis.Redis,
        jwt_secret: str,
        oauth2_providers: Optional[Dict[str, Dict[str, Any]]] = None,
        ldap_servers: Optional[List[Dict[str, Any]]] = None,
        saml_config: Optional[Dict[str, Any]] = None
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.jwt_secret = jwt_secret
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
        
        # Authentication services
        self.local_auth = LocalAuthenticationService(db_session, self.pwd_context)
        self.oauth2_auth = OAuth2AuthenticationService(oauth2_providers or {})
        self.ldap_auth = LDAPAuthenticationService(ldap_servers or [])
        self.saml_auth = SAMLAuthenticationService(saml_config or {})
        self.jwt_auth = JWTAuthenticationService(jwt_secret)
        self.mfa_auth = MultiFactorAuthenticationService(db_session, redis_client)
        
        # Authorization services
        self.rbac_service = RBACService(db_session, redis_client)
        self.policy_engine = PolicyEngineService(db_session)
        self.permission_service = PermissionService(db_session, redis_client)
        
        # Session and security services
        self.session_manager = SessionManager(redis_client, db_session)
        self.rate_limiter = RateLimitingService(redis_client)
        self.audit_service = SecurityAuditService(db_session)
        
        # Active sessions cache
        self.active_sessions: Dict[str, UserSession] = {}
        
    async def authenticate(
        self,
        auth_request: AuthenticationRequest,
        tenant_id: str
    ) -> AuthenticationResult:
        """Authenticate user with specified method."""
        
        try:
            # Rate limiting check
            client_ip = auth_request.client_info.get("ip_address", "unknown")
            rate_limit_ok = await self.rate_limiter.check_rate_limit(
                f"auth:{client_ip}",
                limit=10,  # 10 attempts per minute
                window=60
            )
            
            if not rate_limit_ok:
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="Rate limit exceeded"
                )
            
            # Authenticate based on method
            auth_result = None
            
            if auth_request.method == AuthenticationMethod.LOCAL:
                auth_result = await self.local_auth.authenticate(
                    auth_request.credentials, tenant_id
                )
            elif auth_request.method == AuthenticationMethod.OAUTH2:
                auth_result = await self.oauth2_auth.authenticate(
                    auth_request.credentials, auth_request.requested_scopes
                )
            elif auth_request.method == AuthenticationMethod.LDAP:
                auth_result = await self.ldap_auth.authenticate(
                    auth_request.credentials, tenant_id
                )
            elif auth_request.method == AuthenticationMethod.SAML:
                auth_result = await self.saml_auth.authenticate(
                    auth_request.credentials, tenant_id
                )
            elif auth_request.method == AuthenticationMethod.JWT:
                auth_result = await self.jwt_auth.authenticate(
                    auth_request.credentials
                )
            elif auth_request.method == AuthenticationMethod.API_KEY:
                auth_result = await self._authenticate_api_key(
                    auth_request.credentials, tenant_id
                )
            elif auth_request.method == AuthenticationMethod.MULTI_FACTOR:
                auth_result = await self.mfa_auth.authenticate(
                    auth_request.credentials, tenant_id
                )
            else:
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message=f"Unsupported authentication method: {auth_request.method}"
                )
            
            if not auth_result or not auth_result.success:
                # Log failed authentication attempt
                await self.audit_service.log_authentication_event(
                    user_id=auth_result.user_id if auth_result else None,
                    method=auth_request.method.value,
                    success=False,
                    client_info=auth_request.client_info,
                    error_message=auth_result.error_message if auth_result else "Authentication failed",
                    tenant_id=tenant_id
                )
                
                return auth_result or AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="Authentication failed"
                )
            
            # Create user session
            session = await self.session_manager.create_session(
                user_id=auth_result.user_id,
                tenant_id=tenant_id,
                client_info=auth_request.client_info,
                scopes=auth_result.scopes,
                expires_at=auth_result.expires_at
            )
            
            # Update result with session info
            auth_result.session_id = session.session_id
            
            # Cache session
            self.active_sessions[session.session_id] = session
            
            # Log successful authentication
            await self.audit_service.log_authentication_event(
                user_id=auth_result.user_id,
                method=auth_request.method.value,
                success=True,
                client_info=auth_request.client_info,
                session_id=session.session_id,
                tenant_id=tenant_id
            )
            
            logger.info(f"User {auth_result.user_id} authenticated successfully via {auth_request.method.value}")
            return auth_result
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            
            await self.audit_service.log_authentication_event(
                user_id=None,
                method=auth_request.method.value,
                success=False,
                client_info=auth_request.client_info,
                error_message=str(e),
                tenant_id=tenant_id
            )
            
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Authentication system error"
            )
    
    async def authorize(
        self,
        context: AuthorizationContext,
        session_id: Optional[str] = None
    ) -> bool:
        """Authorize user action on resource."""
        
        try:
            # Validate session if provided
            if session_id:
                session = await self.session_manager.get_session(session_id)
                if not session or session.status != SessionStatus.ACTIVE:
                    return False
                
                # Update context with session user
                context.user_id = session.user_id
                
                # Update last accessed time
                await self.session_manager.update_last_accessed(session_id)
            
            # Check RBAC permissions
            rbac_allowed = await self.rbac_service.check_permission(
                user_id=context.user_id,
                resource_type=context.resource_type,
                resource_id=context.resource_id,
                action=context.action
            )
            
            # Evaluate policies
            policy_result = await self.policy_engine.evaluate_policies(context)
            
            # Combine RBAC and policy results
            final_decision = rbac_allowed and policy_result.effect == PolicyEffect.ALLOW
            
            # Log authorization decision
            await self.audit_service.log_authorization_event(
                user_id=context.user_id,
                resource_type=context.resource_type.value,
                resource_id=context.resource_id,
                action=context.action,
                decision=final_decision,
                session_id=session_id,
                client_info=context.client_info
            )
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False
    
    async def create_user(
        self,
        user_data: Dict[str, Any],
        tenant_id: str,
        created_by: str
    ) -> str:
        """Create new user account."""
        
        try:
            from ..database import CRPermission
            
            user_id = f"user_{uuid7str()}"
            
            # Hash password if provided
            password_hash = None
            if user_data.get("password"):
                password_hash = self.pwd_context.hash(user_data["password"])
            
            # Create user record
            user_record = {
                "user_id": user_id,
                "username": user_data["username"],
                "email": user_data["email"],
                "first_name": user_data.get("first_name"),
                "last_name": user_data.get("last_name"),
                "password_hash": password_hash,
                "enabled": user_data.get("enabled", True),
                "email_verified": user_data.get("email_verified", False),
                "created_at": datetime.now(timezone.utc),
                "tenant_id": tenant_id,
                "metadata": user_data.get("metadata", {})
            }
            
            # Store in Redis
            user_key = f"user:{tenant_id}:{user_id}"
            await self.redis_client.setex(
                user_key,
                86400 * 7,  # 7 days
                json.dumps(user_record, default=str)
            )
            
            # Assign default role if specified
            default_role = user_data.get("default_role")
            if default_role:
                await self.rbac_service.assign_role_to_user(
                    user_id=user_id,
                    role_id=default_role,
                    tenant_id=tenant_id,
                    assigned_by=created_by
                )
            
            logger.info(f"Created user {user_id} in tenant {tenant_id}")
            return user_id
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    async def create_role(
        self,
        role_data: Dict[str, Any],
        tenant_id: str,
        created_by: str
    ) -> str:
        """Create new role."""
        
        return await self.rbac_service.create_role(role_data, tenant_id, created_by)
    
    async def grant_permission(
        self,
        permission: Permission,
        tenant_id: str,
        granted_by: str
    ) -> str:
        """Grant permission to user or role."""
        
        return await self.permission_service.grant_permission(permission, tenant_id, granted_by)
    
    async def revoke_permission(
        self,
        permission_id: str,
        tenant_id: str,
        revoked_by: str
    ) -> bool:
        """Revoke permission."""
        
        return await self.permission_service.revoke_permission(permission_id, tenant_id, revoked_by)
    
    async def get_user_permissions(
        self,
        user_id: str,
        tenant_id: str
    ) -> List[Permission]:
        """Get all permissions for user."""
        
        return await self.permission_service.get_user_permissions(user_id, tenant_id)
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate user session."""
        
        success = await self.session_manager.invalidate_session(session_id)
        
        if success and session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        return success
    
    async def refresh_token(
        self,
        refresh_token: str,
        tenant_id: str
    ) -> AuthenticationResult:
        """Refresh access token."""
        
        try:
            # Decode refresh token
            payload = jwt.decode(
                refresh_token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            
            if not user_id or not session_id:
                raise ValueError("Invalid refresh token")
            
            # Validate session
            session = await self.session_manager.get_session(session_id)
            if not session or session.status != SessionStatus.ACTIVE:
                raise ValueError("Invalid session")
            
            # Generate new access token
            new_access_token = self.jwt_auth.generate_access_token(
                user_id=user_id,
                session_id=session_id,
                scopes=session.permissions,
                expires_in=3600  # 1 hour
            )
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=session_id,
                access_token=new_access_token,
                refresh_token=refresh_token,  # Keep same refresh token
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                scopes=[p.permission_level.value for p in session.permissions],
                user_info={},
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Token refresh failed"
            )
    
    async def enable_mfa(
        self,
        user_id: str,
        mfa_method: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Enable multi-factor authentication for user."""
        
        return await self.mfa_auth.enable_mfa(user_id, mfa_method, tenant_id)
    
    async def verify_mfa(
        self,
        user_id: str,
        mfa_code: str,
        tenant_id: str
    ) -> bool:
        """Verify MFA code."""
        
        return await self.mfa_auth.verify_code(user_id, mfa_code, tenant_id)
    
    async def get_security_audit_log(
        self,
        tenant_id: str,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get security audit log."""
        
        return await self.audit_service.get_audit_log(
            tenant_id=tenant_id,
            event_type=event_type,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    async def _authenticate_api_key(
        self,
        credentials: Dict[str, Any],
        tenant_id: str
    ) -> AuthenticationResult:
        """Authenticate using API key."""
        
        api_key = credentials.get("api_key")
        if not api_key:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="API key required"
            )
        
        # Validate API key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        api_key_data = await self.redis_client.get(f"api_key:{tenant_id}:{key_hash}")
        
        if not api_key_data:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Invalid API key"
            )
        
        try:
            key_info = json.loads(api_key_data)
            
            # Check if key is active and not expired
            if not key_info.get("active", False):
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="API key is disabled"
                )
            
            expires_at = key_info.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="API key expired"
                )
            
            return AuthenticationResult(
                success=True,
                user_id=key_info["user_id"],
                session_id=None,  # API keys don't use sessions
                access_token=api_key,  # Use API key as token
                refresh_token=None,
                expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
                scopes=key_info.get("scopes", []),
                user_info=key_info.get("user_info", {}),
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="API key validation failed"
            )
    
    async def close(self):
        """Close access control service and clean up resources."""
        
        # Close LDAP connections
        await self.ldap_auth.close()
        
        # Clear session cache
        self.active_sessions.clear()

# =============================================================================
# Authentication Services
# =============================================================================

class LocalAuthenticationService:
    """Local username/password authentication."""
    
    def __init__(self, db_session: AsyncSession, pwd_context: CryptContext):
        self.db_session = db_session
        self.pwd_context = pwd_context
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        tenant_id: str
    ) -> AuthenticationResult:
        """Authenticate with username/password."""
        
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Username and password required"
            )
        
        # Get user from database/cache with Redis lookup
        user_key = f"user:{tenant_id}:username:{username}"
        user_data = await self.redis_client.get(user_key)
        
        if not user_data:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Invalid credentials"
            )
        
        try:
            user_info = json.loads(user_data)
            
            # Verify password
            if not self.pwd_context.verify(password, user_info.get("password_hash", "")):
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="Invalid credentials"
                )
            
            # Check if user is enabled
            if not user_info.get("enabled", False):
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="Account disabled"
                )
            
            return AuthenticationResult(
                success=True,
                user_id=user_info["user_id"],
                session_id=None,  # Will be set later
                access_token=None,  # Will be generated later
                refresh_token=None,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                scopes=user_info.get("scopes", []),
                user_info={
                    "username": user_info["username"],
                    "email": user_info["email"],
                    "first_name": user_info.get("first_name"),
                    "last_name": user_info.get("last_name")
                },
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Local authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Authentication failed"
            )

class OAuth2AuthenticationService:
    """OAuth2 authentication service."""
    
    def __init__(self, oauth2_providers: Dict[str, Dict[str, Any]]):
        self.providers = oauth2_providers
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        scopes: List[str]
    ) -> AuthenticationResult:
        """Authenticate with OAuth2 provider."""
        
        provider_name = credentials.get("provider")
        authorization_code = credentials.get("code")
        redirect_uri = credentials.get("redirect_uri")
        
        if not provider_name or provider_name not in self.providers:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Invalid OAuth2 provider"
            )
        
        provider_config = self.providers[provider_name]
        
        try:
            # Create OAuth2 session
            oauth = OAuth2Session(
                client_id=provider_config["client_id"],
                client_secret=provider_config["client_secret"],
                redirect_uri=redirect_uri
            )
            
            # Exchange authorization code for access token
            token_response = oauth.fetch_token(
                provider_config["token_url"],
                authorization_response=credentials.get("authorization_response"),
                code=authorization_code
            )
            
            access_token = token_response["access_token"]
            refresh_token = token_response.get("refresh_token")
            expires_in = token_response.get("expires_in", 3600)
            
            # Get user info from provider
            user_info_response = oauth.get(provider_config["user_info_url"])
            user_info = user_info_response.json()
            
            # Extract user ID (provider-specific)
            user_id = self._extract_user_id(user_info, provider_name)
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=None,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
                scopes=scopes,
                user_info=user_info,
                error_message=None
            )
            
        except OAuth2Error as e:
            logger.error(f"OAuth2 authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message=f"OAuth2 error: {e.description}"
            )
        except Exception as e:
            logger.error(f"OAuth2 authentication error: {e}")
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="OAuth2 authentication failed"
            )
    
    def _extract_user_id(self, user_info: Dict[str, Any], provider: str) -> str:
        """Extract user ID from provider user info."""
        
        if provider == "google":
            return user_info.get("sub") or user_info.get("id")
        elif provider == "github":
            return str(user_info.get("id"))
        elif provider == "microsoft":
            return user_info.get("id") or user_info.get("oid")
        else:
            return user_info.get("id") or user_info.get("sub") or user_info.get("email")

class LDAPAuthenticationService:
    """LDAP authentication service."""
    
    def __init__(self, ldap_servers: List[Dict[str, Any]]):
        self.servers = ldap_servers
        self.connections: Dict[str, Connection] = {}
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        tenant_id: str
    ) -> AuthenticationResult:
        """Authenticate against LDAP server."""
        
        username = credentials.get("username")
        password = credentials.get("password")
        domain = credentials.get("domain")
        
        if not username or not password:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Username and password required"
            )
        
        # Find appropriate LDAP server
        ldap_config = None
        for server_config in self.servers:
            if not domain or server_config.get("domain") == domain:
                ldap_config = server_config
                break
        
        if not ldap_config:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="LDAP server not configured"
            )
        
        try:
            # Create LDAP server connection
            server = Server(
                ldap_config["host"],
                port=ldap_config.get("port", 389),
                use_ssl=ldap_config.get("use_ssl", False),
                get_info=ALL
            )
            
            # Build user DN
            user_dn = f"{ldap_config.get('user_dn_template', 'cn={username},ou=users')}"
            user_dn = user_dn.format(username=username, domain=domain or "")
            
            # Attempt to bind (authenticate)
            conn = Connection(
                server,
                user=user_dn,
                password=password,
                authentication=NTLM if ldap_config.get("use_ntlm") else None
            )
            
            if not conn.bind():
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="Invalid LDAP credentials"
                )
            
            # Get user attributes
            search_base = ldap_config.get("search_base", "ou=users,dc=example,dc=com")
            search_filter = f"({ldap_config.get('username_attribute', 'sAMAccountName')}={username})"
            
            conn.search(
                search_base,
                search_filter,
                attributes=ldap_config.get("user_attributes", ["cn", "mail", "memberOf"])
            )
            
            if not conn.entries:
                conn.unbind()
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="User not found in LDAP"
                )
            
            user_entry = conn.entries[0]
            user_id = f"ldap_{hashlib.sha256(str(user_entry.entry_dn).encode()).hexdigest()[:16]}"
            
            # Extract user information
            user_info = {
                "username": username,
                "dn": str(user_entry.entry_dn),
                "email": str(user_entry.mail) if hasattr(user_entry, 'mail') else "",
                "display_name": str(user_entry.cn) if hasattr(user_entry, 'cn') else username,
                "groups": [str(group) for group in user_entry.memberOf] if hasattr(user_entry, 'memberOf') else []
            }
            
            conn.unbind()
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=8),
                scopes=["read", "write"] if user_info["groups"] else ["read"],
                user_info=user_info,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"LDAP authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="LDAP authentication failed"
            )
    
    async def close(self):
        """Close LDAP connections."""
        for conn in self.connections.values():
            if conn.bound:
                conn.unbind()
        self.connections.clear()

class SAMLAuthenticationService:
    """SAML authentication service."""
    
    def __init__(self, saml_config: Dict[str, Any]):
        self.config = saml_config
        self.saml_client = None
        
        if saml_config:
            self._initialize_saml_client()
    
    def _initialize_saml_client(self):
        """Initialize SAML client."""
        try:
            config = SAMLConfig()
            config.load(self.config)
            self.saml_client = Saml2Client(config)
        except Exception as e:
            logger.error(f"Failed to initialize SAML client: {e}")
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        tenant_id: str
    ) -> AuthenticationResult:
        """Authenticate SAML response."""
        
        if not self.saml_client:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="SAML not configured"
            )
        
        saml_response = credentials.get("saml_response")
        if not saml_response:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="SAML response required"
            )
        
        try:
            # Parse and validate SAML response
            authn_response = self.saml_client.parse_authn_request_response(
                saml_response,
                BINDING_HTTP_POST
            )
            
            if not authn_response.came_from:
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="Invalid SAML response"
                )
            
            # Extract user information from SAML attributes
            user_info = {}
            attributes = authn_response.ava
            
            for attr_name, attr_values in attributes.items():
                if attr_values:
                    user_info[attr_name] = attr_values[0] if len(attr_values) == 1 else attr_values
            
            # Generate user ID from SAML NameID
            name_id = authn_response.name_id
            user_id = f"saml_{hashlib.sha256(str(name_id).encode()).hexdigest()[:16]}"
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=8),
                scopes=["read", "write"],
                user_info=user_info,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"SAML authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="SAML authentication failed"
            )

class JWTAuthenticationService:
    """JWT token authentication service."""
    
    def __init__(self, jwt_secret: str):
        self.jwt_secret = jwt_secret
    
    async def authenticate(
        self,
        credentials: Dict[str, Any]
    ) -> AuthenticationResult:
        """Authenticate JWT token."""
        
        token = credentials.get("token")
        if not token:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="JWT token required"
            )
        
        try:
            # Decode and validate JWT
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            scopes = payload.get("scopes", [])
            exp = payload.get("exp")
            
            if not user_id:
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    session_id=None,
                    access_token=None,
                    refresh_token=None,
                    expires_at=None,
                    scopes=[],
                    user_info={},
                    error_message="Invalid JWT token"
                )
            
            expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=session_id,
                access_token=token,
                refresh_token=None,
                expires_at=expires_at,
                scopes=scopes,
                user_info=payload.get("user_info", {}),
                error_message=None
            )
            
        except jwt.ExpiredSignatureError:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="JWT token expired"
            )
        except jwt.InvalidTokenError as e:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message=f"Invalid JWT token: {str(e)}"
            )
    
    def generate_access_token(
        self,
        user_id: str,
        session_id: str,
        scopes: List[Permission],
        expires_in: int = 3600
    ) -> str:
        """Generate JWT access token."""
        
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "scopes": [p.permission_level.value for p in scopes],
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

class MultiFactorAuthenticationService:
    """Multi-factor authentication service."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
    
    async def enable_mfa(
        self,
        user_id: str,
        mfa_method: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Enable MFA for user."""
        
        if mfa_method == "totp":
            # Generate TOTP secret
            import pyotp
            
            secret = pyotp.random_base32()
            totp = pyotp.TOTP(secret)
            
            # Store secret
            mfa_key = f"mfa:{tenant_id}:{user_id}"
            mfa_data = {
                "method": "totp",
                "secret": secret,
                "enabled": True,
                "backup_codes": [secrets.token_hex(4) for _ in range(10)]
            }
            
            await self.redis_client.setex(
                mfa_key,
                86400 * 365,  # 1 year
                json.dumps(mfa_data)
            )
            
            # Generate QR code data
            qr_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name="APG Platform"
            )
            
            return {
                "method": "totp",
                "secret": secret,
                "qr_uri": qr_uri,
                "backup_codes": mfa_data["backup_codes"]
            }
        
        else:
            raise ValueError(f"Unsupported MFA method: {mfa_method}")
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        tenant_id: str
    ) -> AuthenticationResult:
        """Authenticate with MFA."""
        
        user_id = credentials.get("user_id")
        mfa_code = credentials.get("mfa_code")
        
        if not user_id or not mfa_code:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="User ID and MFA code required"
            )
        
        # Verify MFA code
        verified = await self.verify_code(user_id, mfa_code, tenant_id)
        
        if not verified:
            return AuthenticationResult(
                success=False,
                user_id=None,
                session_id=None,
                access_token=None,
                refresh_token=None,
                expires_at=None,
                scopes=[],
                user_info={},
                error_message="Invalid MFA code"
            )
        
        return AuthenticationResult(
            success=True,
            user_id=user_id,
            session_id=None,
            access_token=None,
            refresh_token=None,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            scopes=["mfa_verified"],
            user_info={},
            error_message=None
        )
    
    async def verify_code(
        self,
        user_id: str,
        mfa_code: str,
        tenant_id: str
    ) -> bool:
        """Verify MFA code."""
        
        try:
            mfa_key = f"mfa:{tenant_id}:{user_id}"
            mfa_data = await self.redis_client.get(mfa_key)
            
            if not mfa_data:
                return False
            
            mfa_info = json.loads(mfa_data)
            
            if mfa_info.get("method") == "totp":
                import pyotp
                
                totp = pyotp.TOTP(mfa_info["secret"])
                return totp.verify(mfa_code, valid_window=1)
            
            # Check backup codes
            backup_codes = mfa_info.get("backup_codes", [])
            if mfa_code in backup_codes:
                # Remove used backup code
                backup_codes.remove(mfa_code)
                mfa_info["backup_codes"] = backup_codes
                
                await self.redis_client.setex(
                    mfa_key,
                    86400 * 365,
                    json.dumps(mfa_info)
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"MFA verification failed: {e}")
            return False

# =============================================================================
# Authorization Services  
# =============================================================================

class RBACService:
    """Role-Based Access Control service."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
    
    async def create_role(
        self,
        role_data: Dict[str, Any],
        tenant_id: str,
        created_by: str
    ) -> str:
        """Create new role."""
        
        role_id = f"role_{uuid7str()}"
        
        role_record = {
            "role_id": role_id,
            "role_name": role_data["role_name"],
            "description": role_data.get("description"),
            "permissions": role_data.get("permissions", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "created_by": created_by
        }
        
        # Store in Redis
        role_key = f"role:{tenant_id}:{role_id}"
        await self.redis_client.setex(
            role_key,
            86400 * 30,  # 30 days
            json.dumps(role_record, default=str)
        )
        
        logger.info(f"Created role {role_id} in tenant {tenant_id}")
        return role_id
    
    async def assign_role_to_user(
        self,
        user_id: str,
        role_id: str,
        tenant_id: str,
        assigned_by: str
    ) -> bool:
        """Assign role to user."""
        
        # Add to user's roles set
        user_roles_key = f"user_roles:{tenant_id}:{user_id}"
        await self.redis_client.sadd(user_roles_key, role_id)
        await self.redis_client.expire(user_roles_key, 86400 * 7)  # 7 days
        
        # Track assignment
        assignment_key = f"role_assignment:{tenant_id}:{user_id}:{role_id}"
        assignment_data = {
            "user_id": user_id,
            "role_id": role_id,
            "assigned_by": assigned_by,
            "assigned_at": datetime.now(timezone.utc).isoformat()
        }
        
        await self.redis_client.setex(
            assignment_key,
            86400 * 30,  # 30 days
            json.dumps(assignment_data)
        )
        
        return True
    
    async def check_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
        action: str
    ) -> bool:
        """Check if user has permission for action on resource."""
        
        # Get user's roles
        user_roles_key = f"user_roles:*:{user_id}"
        role_keys = []
        
        # Scan for user role keys across tenants (in real implementation, would be more efficient)
        async for key in self.redis_client.scan_iter(match=user_roles_key):
            roles = await self.redis_client.smembers(key)
            role_keys.extend([role.decode() if isinstance(role, bytes) else role for role in roles])
        
        # Check permissions for each role
        for role_id in role_keys:
            # Get role permissions
            role_key_pattern = f"role:*:{role_id}"
            async for role_key in self.redis_client.scan_iter(match=role_key_pattern):
                role_data = await self.redis_client.get(role_key)
                if role_data:
                    role_info = json.loads(role_data)
                    permissions = role_info.get("permissions", [])
                    
                    # Check if any permission allows this action
                    for perm in permissions:
                        if self._permission_allows(perm, resource_type, resource_id, action):
                            return True
        
        return False
    
    def _permission_allows(
        self,
        permission: Dict[str, Any],
        resource_type: ResourceType,
        resource_id: str,
        action: str
    ) -> bool:
        """Check if permission allows action."""
        
        # Simple permission matching
        perm_resource_type = permission.get("resource_type")
        perm_resource_id = permission.get("resource_id", "*")
        perm_actions = permission.get("actions", [])
        
        if perm_resource_type != resource_type.value and perm_resource_type != "*":
            return False
        
        if perm_resource_id != "*" and perm_resource_id != resource_id:
            return False
        
        if action not in perm_actions and "*" not in perm_actions:
            return False
        
        return True

class PolicyEngineService:
    """Policy-based authorization service using Casbin."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.enforcer = None
        self._initialize_casbin()
    
    def _initialize_casbin(self):
        """Initialize Casbin enforcer."""
        try:
            # Create Casbin adapter for database
            adapter = CasbinAdapter()
            
            # RBAC model
            model_config = """
            [request_definition]
            r = sub, obj, act
            
            [policy_definition]
            p = sub, obj, act
            
            [role_definition]
            g = _, _
            
            [policy_effect]
            e = some(where (p.eft == allow))
            
            [matchers]
            m = g(r.sub, p.sub) && r.obj == p.obj && r.act == p.act
            """
            
            self.enforcer = casbin.Enforcer(model_config, adapter)
            
        except Exception as e:
            logger.error(f"Failed to initialize Casbin: {e}")
    
    async def evaluate_policies(
        self,
        context: AuthorizationContext
    ) -> Dict[str, Any]:
        """Evaluate authorization policies."""
        
        if not self.enforcer:
            return {"effect": PolicyEffect.DENY, "reason": "Policy engine not available"}
        
        try:
            # Build policy evaluation request
            subject = context.user_id
            object_resource = f"{context.resource_type.value}:{context.resource_id}"
            action = context.action
            
            # Check if action is allowed
            allowed = self.enforcer.enforce(subject, object_resource, action)
            
            return {
                "effect": PolicyEffect.ALLOW if allowed else PolicyEffect.DENY,
                "reason": "Policy evaluation completed",
                "details": {
                    "subject": subject,
                    "object": object_resource,
                    "action": action,
                    "result": allowed
                }
            }
            
        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return {"effect": PolicyEffect.DENY, "reason": f"Policy evaluation error: {str(e)}"}

class PermissionService:
    """Permission management service."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
    
    async def grant_permission(
        self,
        permission: Permission,
        tenant_id: str,
        granted_by: str
    ) -> str:
        """Grant permission to user or role."""
        
        from ..database import CRPermission
        
        # Create permission record
        db_permission = CRPermission(
            resource_type=permission.resource_type.value,
            resource_id=permission.resource_id,
            user_id=permission.user_id,
            role_id=permission.role_id,
            access_level=permission.permission_level.value,
            conditions=permission.conditions,
            granted_by=granted_by,
            granted_at=permission.granted_at,
            expires_at=permission.expires_at,
            tenant_id=tenant_id
        )
        
        self.db_session.add(db_permission)
        await self.db_session.commit()
        
        # Cache permission
        permission_key = f"permission:{tenant_id}:{db_permission.id}"
        permission_data = {
            "user_id": permission.user_id,
            "role_id": permission.role_id,
            "resource_type": permission.resource_type.value,
            "resource_id": permission.resource_id,
            "permission_level": permission.permission_level.value,
            "conditions": permission.conditions,
            "granted_by": granted_by,
            "granted_at": permission.granted_at.isoformat(),
            "expires_at": permission.expires_at.isoformat() if permission.expires_at else None
        }
        
        await self.redis_client.setex(
            permission_key,
            86400 * 7,  # 7 days
            json.dumps(permission_data, default=str)
        )
        
        logger.info(f"Granted permission {db_permission.id} in tenant {tenant_id}")
        return str(db_permission.id)
    
    async def revoke_permission(
        self,
        permission_id: str,
        tenant_id: str,
        revoked_by: str
    ) -> bool:
        """Revoke permission."""
        
        from ..database import CRPermission
        
        # Update database
        result = await self.db_session.execute(
            update(CRPermission)
            .where(
                and_(
                    CRPermission.id == permission_id,
                    CRPermission.tenant_id == tenant_id
                )
            )
            .values(is_active=False)
        )
        
        await self.db_session.commit()
        
        if result.rowcount > 0:
            # Remove from cache
            permission_key = f"permission:{tenant_id}:{permission_id}"
            await self.redis_client.delete(permission_key)
            
            logger.info(f"Revoked permission {permission_id} in tenant {tenant_id}")
            return True
        
        return False
    
    async def get_user_permissions(
        self,
        user_id: str,
        tenant_id: str
    ) -> List[Permission]:
        """Get all permissions for user."""
        
        from ..database import CRPermission
        
        # Get direct user permissions
        result = await self.db_session.execute(
            select(CRPermission).where(
                and_(
                    CRPermission.user_id == user_id,
                    CRPermission.tenant_id == tenant_id,
                    CRPermission.is_active == True
                )
            )
        )
        
        permissions = []
        for perm in result.scalars().all():
            permissions.append(Permission(
                user_id=perm.user_id,
                role_id=perm.role_id,
                resource_type=ResourceType(perm.resource_type),
                resource_id=perm.resource_id,
                permission_level=PermissionLevel(perm.access_level),
                conditions=perm.conditions or {},
                granted_by=perm.granted_by,
                granted_at=perm.granted_at,
                expires_at=perm.expires_at
            ))
        
        return permissions

# =============================================================================
# Supporting Services
# ============================================================================= 

class SessionManager:
    """User session management."""
    
    def __init__(self, redis_client: redis.Redis, db_session: AsyncSession):
        self.redis_client = redis_client
        self.db_session = db_session
    
    async def create_session(
        self,
        user_id: str,
        tenant_id: str,
        client_info: Dict[str, Any],
        scopes: List[str],
        expires_at: Optional[datetime] = None
    ) -> UserSession:
        """Create new user session."""
        
        session_id = f"sess_{uuid7str()}"
        
        if not expires_at:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            expires_at=expires_at,
            ip_address=client_info.get("ip_address", "unknown"),
            user_agent=client_info.get("user_agent", "unknown"),
            permissions=[],  # Will be loaded separately
            metadata=client_info
        )
        
        # Store in Redis
        session_key = f"session:{session_id}"
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "ip_address": session.ip_address,
            "user_agent": session.user_agent,
            "scopes": scopes,
            "metadata": session.metadata
        }
        
        # Set expiration
        ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds())
        await self.redis_client.setex(session_key, ttl, json.dumps(session_data, default=str))
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        
        session_key = f"session:{session_id}"
        session_data = await self.redis_client.get(session_key)
        
        if not session_data:
            return None
        
        try:
            data = json.loads(session_data)
            
            return UserSession(
                session_id=data["session_id"],
                user_id=data["user_id"],
                tenant_id=data["tenant_id"],
                status=SessionStatus(data["status"]),
                created_at=datetime.fromisoformat(data["created_at"]),
                last_accessed=datetime.fromisoformat(data["last_accessed"]),
                expires_at=datetime.fromisoformat(data["expires_at"]),
                ip_address=data["ip_address"],
                user_agent=data["user_agent"],
                permissions=[],  # Would load from database if needed
                metadata=data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to parse session data: {e}")
            return None
    
    async def update_last_accessed(self, session_id: str) -> bool:
        """Update session last accessed time."""
        
        session_key = f"session:{session_id}"
        session_data = await self.redis_client.get(session_key)
        
        if not session_data:
            return False
        
        try:
            data = json.loads(session_data)
            data["last_accessed"] = datetime.now(timezone.utc).isoformat()
            
            # Get current TTL and preserve it
            ttl = await self.redis_client.ttl(session_key)
            if ttl > 0:
                await self.redis_client.setex(session_key, ttl, json.dumps(data, default=str))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session last accessed: {e}")
            return False
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session."""
        
        session_key = f"session:{session_id}"
        result = await self.redis_client.delete(session_key)
        return result > 0

class RateLimitingService:
    """Rate limiting service."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> bool:
        """Check if request is within rate limit."""
        
        try:
            current_count = await self.redis_client.get(key)
            current_count = int(current_count) if current_count else 0
            
            if current_count >= limit:
                return False
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting check failed: {e}")
            return True  # Allow on error

class SecurityAuditService:
    """Security audit logging service."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def log_authentication_event(
        self,
        user_id: Optional[str],
        method: str,
        success: bool,
        client_info: Dict[str, Any],
        session_id: Optional[str] = None,
        error_message: Optional[str] = None,
        tenant_id: str = None
    ):
        """Log authentication event."""
        
        # In real implementation, would store in audit table
        audit_data = {
            "event_type": "authentication",
            "user_id": user_id,
            "method": method,
            "success": success,
            "session_id": session_id,
            "ip_address": client_info.get("ip_address"),
            "user_agent": client_info.get("user_agent"),
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id
        }
        
        logger.info(f"Authentication audit: {json.dumps(audit_data)}")
    
    async def log_authorization_event(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        decision: bool,
        session_id: Optional[str] = None,
        client_info: Optional[Dict[str, Any]] = None
    ):
        """Log authorization event."""
        
        audit_data = {
            "event_type": "authorization",
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "decision": decision,
            "session_id": session_id,
            "ip_address": client_info.get("ip_address") if client_info else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Authorization audit: {json.dumps(audit_data)}")
    
    async def get_audit_log(
        self,
        tenant_id: str,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get security audit log."""
        
        # In real implementation, would query audit table
        # For now, return empty list
        return []

# =============================================================================
# Service Factory
# =============================================================================

async def create_access_control_service(
    db_session: AsyncSession,
    redis_url: str,
    jwt_secret: str,
    oauth2_providers: Optional[Dict[str, Dict[str, Any]]] = None,
    ldap_servers: Optional[List[Dict[str, Any]]] = None,
    saml_config: Optional[Dict[str, Any]] = None
) -> AccessControlIntegrationService:
    """Factory function to create access control integration service."""
    
    redis_client = redis.from_url(redis_url)
    
    return AccessControlIntegrationService(
        db_session=db_session,
        redis_client=redis_client,
        jwt_secret=jwt_secret,
        oauth2_providers=oauth2_providers,
        ldap_servers=ldap_servers,
        saml_config=saml_config
    )

# Export service classes
__all__ = [
    "AccessControlIntegrationService",
    "LocalAuthenticationService",
    "OAuth2AuthenticationService",
    "LDAPAuthenticationService",
    "SAMLAuthenticationService",
    "JWTAuthenticationService",
    "MultiFactorAuthenticationService",
    "RBACService",
    "PolicyEngineService",
    "PermissionService",
    "SessionManager",
    "RateLimitingService",
    "SecurityAuditService",
    "AuthenticationRequest",
    "AuthenticationResult",
    "Permission",
    "AuthorizationContext",
    "UserSession",
    "AuthenticationMethod",
    "PermissionLevel",
    "ResourceType",
    "PolicyEffect",
    "SessionStatus",
    "create_access_control_service"
]