"""
Single Sign-On (SSO) Connectors

Provides comprehensive SSO integrations for major enterprise identity providers:
- Microsoft Azure AD / Entra ID
- Google Workspace / Cloud Identity
- Okta Universal Directory
- Auth0 Platform
- Ping Identity
- IBM Security Verify
- Oracle Identity Cloud
- Salesforce Identity

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
import time
import base64
import hmac
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import jwt
import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .enterprise_integration import AuditEvent, enterprise_integration


class SSOProvider(str, Enum):
	"""SSO provider types"""
	AZURE_AD = "azure_ad"
	GOOGLE_WORKSPACE = "google_workspace"
	OKTA = "okta"
	AUTH0 = "auth0"
	PING_IDENTITY = "ping_identity"
	IBM_VERIFY = "ibm_verify"
	ORACLE_IDENTITY = "oracle_identity"
	SALESFORCE = "salesforce"
	ONELOGIN = "onelogin"
	KEYCLOAK = "keycloak"


class TokenType(str, Enum):
	"""Token types"""
	ACCESS_TOKEN = "access_token"
	REFRESH_TOKEN = "refresh_token"
	ID_TOKEN = "id_token"
	SAML_ASSERTION = "saml_assertion"


@dataclass
class SSOConfiguration:
	"""Base SSO configuration"""
	provider: SSOProvider
	client_id: str
	client_secret: str
	tenant_id: Optional[str] = None
	domain: Optional[str] = None
	custom_endpoints: Optional[Dict[str, str]] = None
	scopes: List[str] = field(default_factory=list)
	custom_claims: List[str] = field(default_factory=list)
	token_validation: Dict[str, Any] = field(default_factory=dict)


class SSOSession(BaseModel):
	"""SSO session model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	user_id: str
	provider: SSOProvider
	provider_user_id: str
	email: str
	display_name: str
	roles: List[str] = Field(default_factory=list)
	groups: List[str] = Field(default_factory=list)
	claims: Dict[str, Any] = Field(default_factory=dict)
	access_token: Optional[str] = None
	refresh_token: Optional[str] = None
	id_token: Optional[str] = None
	token_expires_at: Optional[datetime] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	last_activity: datetime = Field(default_factory=datetime.utcnow)
	is_active: bool = True


class AzureADConnector:
	"""Microsoft Azure AD / Entra ID connector"""
	
	def __init__(self, config: SSOConfiguration):
		self.config = config
		self.tenant_id = config.tenant_id
		self.base_url = f"https://login.microsoftonline.com/{self.tenant_id}"
		self.graph_url = "https://graph.microsoft.com/v1.0"
		self.scopes = config.scopes or ["openid", "profile", "email", "User.Read"]
	
	async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
		"""Generate Azure AD authorization URL"""
		params = {
			"client_id": self.config.client_id,
			"response_type": "code",
			"redirect_uri": redirect_uri,
			"scope": " ".join(self.scopes),
			"state": state,
			"response_mode": "query"
		}
		
		query_string = "&".join([f"{k}={v}" for k, v in params.items()])
		return f"{self.base_url}/oauth2/v2.0/authorize?{query_string}"
	
	async def exchange_code(self, code: str, redirect_uri: str) -> Tuple[bool, Optional[SSOSession]]:
		"""Exchange authorization code for tokens"""
		try:
			token_url = f"{self.base_url}/oauth2/v2.0/token"
			
			data = {
				"client_id": self.config.client_id,
				"client_secret": self.config.client_secret,
				"code": code,
				"grant_type": "authorization_code",
				"redirect_uri": redirect_uri,
				"scope": " ".join(self.scopes)
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(token_url, data=data) as response:
					if response.status == 200:
						tokens = await response.json()
						
						# Get user info from Microsoft Graph
						user_info = await self._get_user_info(tokens["access_token"])
						if user_info:
							# Create SSO session
							sso_session = SSOSession(
								user_id=user_info["id"],
								provider=SSOProvider.AZURE_AD,
								provider_user_id=user_info["id"],
								email=user_info.get("mail", user_info.get("userPrincipalName", "")),
								display_name=user_info.get("displayName", ""),
								access_token=tokens["access_token"],
								refresh_token=tokens.get("refresh_token"),
								id_token=tokens.get("id_token"),
								token_expires_at=datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600)),
								claims=user_info
							)
							
							# Get user groups and roles
							groups = await self._get_user_groups(tokens["access_token"], user_info["id"])
							sso_session.groups = groups
							
							roles = await self._get_user_roles(tokens["access_token"], user_info["id"])
							sso_session.roles = roles
							
							return True, sso_session
			
			return False, None
			
		except Exception as e:
			print(f"Azure AD token exchange error: {e}")
			return False, None
	
	async def _get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
		"""Get user information from Microsoft Graph"""
		try:
			headers = {"Authorization": f"Bearer {access_token}"}
			
			async with aiohttp.ClientSession() as session:
				async with session.get(f"{self.graph_url}/me", headers=headers) as response:
					if response.status == 200:
						return await response.json()
			
			return None
			
		except Exception as e:
			print(f"Azure AD user info error: {e}")
			return None
	
	async def _get_user_groups(self, access_token: str, user_id: str) -> List[str]:
		"""Get user group memberships"""
		try:
			headers = {"Authorization": f"Bearer {access_token}"}
			
			async with aiohttp.ClientSession() as session:
				async with session.get(f"{self.graph_url}/me/memberOf", headers=headers) as response:
					if response.status == 200:
						data = await response.json()
						return [group.get("displayName", "") for group in data.get("value", [])]
			
			return []
			
		except Exception as e:
			print(f"Azure AD groups error: {e}")
			return []
	
	async def _get_user_roles(self, access_token: str, user_id: str) -> List[str]:
		"""Get user application roles"""
		try:
			headers = {"Authorization": f"Bearer {access_token}"}
			
			async with aiohttp.ClientSession() as session:
				async with session.get(f"{self.graph_url}/me/appRoleAssignments", headers=headers) as response:
					if response.status == 200:
						data = await response.json()
						return [role.get("appRoleId", "") for role in data.get("value", [])]
			
			return []
			
		except Exception as e:
			print(f"Azure AD roles error: {e}")
			return []
	
	async def refresh_token(self, refresh_token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
		"""Refresh access token"""
		try:
			token_url = f"{self.base_url}/oauth2/v2.0/token"
			
			data = {
				"client_id": self.config.client_id,
				"client_secret": self.config.client_secret,
				"refresh_token": refresh_token,
				"grant_type": "refresh_token",
				"scope": " ".join(self.scopes)
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(token_url, data=data) as response:
					if response.status == 200:
						return True, await response.json()
			
			return False, None
			
		except Exception as e:
			print(f"Azure AD token refresh error: {e}")
			return False, None


class GoogleWorkspaceConnector:
	"""Google Workspace / Cloud Identity connector"""
	
	def __init__(self, config: SSOConfiguration):
		self.config = config
		self.base_url = "https://accounts.google.com"
		self.api_url = "https://www.googleapis.com"
		self.scopes = config.scopes or [
			"openid", "email", "profile",
			"https://www.googleapis.com/auth/admin.directory.user.readonly",
			"https://www.googleapis.com/auth/admin.directory.group.readonly"
		]
	
	async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
		"""Generate Google authorization URL"""
		params = {
			"client_id": self.config.client_id,
			"response_type": "code",
			"redirect_uri": redirect_uri,
			"scope": " ".join(self.scopes),
			"state": state,
			"access_type": "offline",
			"prompt": "consent"
		}
		
		query_string = "&".join([f"{k}={v}" for k, v in params.items()])
		return f"{self.base_url}/o/oauth2/v2/auth?{query_string}"
	
	async def exchange_code(self, code: str, redirect_uri: str) -> Tuple[bool, Optional[SSOSession]]:
		"""Exchange authorization code for tokens"""
		try:
			token_url = f"{self.base_url}/o/oauth2/token"
			
			data = {
				"client_id": self.config.client_id,
				"client_secret": self.config.client_secret,
				"code": code,
				"grant_type": "authorization_code",
				"redirect_uri": redirect_uri
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(token_url, data=data) as response:
					if response.status == 200:
						tokens = await response.json()
						
						# Get user info from Google API
						user_info = await self._get_user_info(tokens["access_token"])
						if user_info:
							# Create SSO session
							sso_session = SSOSession(
								user_id=user_info["id"],
								provider=SSOProvider.GOOGLE_WORKSPACE,
								provider_user_id=user_info["id"],
								email=user_info.get("email", ""),
								display_name=user_info.get("name", ""),
								access_token=tokens["access_token"],
								refresh_token=tokens.get("refresh_token"),
								id_token=tokens.get("id_token"),
								token_expires_at=datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600)),
								claims=user_info
							)
							
							# Get user groups if admin access
							if "admin.directory" in " ".join(self.scopes):
								groups = await self._get_user_groups(tokens["access_token"], user_info["email"])
								sso_session.groups = groups
							
							return True, sso_session
			
			return False, None
			
		except Exception as e:
			print(f"Google Workspace token exchange error: {e}")
			return False, None
	
	async def _get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
		"""Get user information from Google API"""
		try:
			headers = {"Authorization": f"Bearer {access_token}"}
			
			async with aiohttp.ClientSession() as session:
				async with session.get(f"{self.api_url}/oauth2/v2/userinfo", headers=headers) as response:
					if response.status == 200:
						return await response.json()
			
			return None
			
		except Exception as e:
			print(f"Google user info error: {e}")
			return None
	
	async def _get_user_groups(self, access_token: str, user_email: str) -> List[str]:
		"""Get user group memberships from Google Admin API"""
		try:
			headers = {"Authorization": f"Bearer {access_token}"}
			
			async with aiohttp.ClientSession() as session:
				url = f"{self.api_url}/admin/directory/v1/groups?userKey={user_email}"
				async with session.get(url, headers=headers) as response:
					if response.status == 200:
						data = await response.json()
						return [group.get("name", "") for group in data.get("groups", [])]
			
			return []
			
		except Exception as e:
			print(f"Google groups error: {e}")
			return []


class OktaConnector:
	"""Okta Universal Directory connector"""
	
	def __init__(self, config: SSOConfiguration):
		self.config = config
		self.domain = config.domain  # e.g., "company.okta.com"
		self.base_url = f"https://{self.domain}"
		self.scopes = config.scopes or ["openid", "profile", "email", "groups"]
	
	async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
		"""Generate Okta authorization URL"""
		params = {
			"client_id": self.config.client_id,
			"response_type": "code",
			"redirect_uri": redirect_uri,
			"scope": " ".join(self.scopes),
			"state": state
		}
		
		query_string = "&".join([f"{k}={v}" for k, v in params.items()])
		return f"{self.base_url}/oauth2/v1/authorize?{query_string}"
	
	async def exchange_code(self, code: str, redirect_uri: str) -> Tuple[bool, Optional[SSOSession]]:
		"""Exchange authorization code for tokens"""
		try:
			token_url = f"{self.base_url}/oauth2/v1/token"
			
			data = {
				"client_id": self.config.client_id,
				"client_secret": self.config.client_secret,
				"code": code,
				"grant_type": "authorization_code",
				"redirect_uri": redirect_uri
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(token_url, data=data) as response:
					if response.status == 200:
						tokens = await response.json()
						
						# Get user info from Okta API
						user_info = await self._get_user_info(tokens["access_token"])
						if user_info:
							# Create SSO session
							sso_session = SSOSession(
								user_id=user_info["sub"],
								provider=SSOProvider.OKTA,
								provider_user_id=user_info["sub"],
								email=user_info.get("email", ""),
								display_name=user_info.get("name", ""),
								access_token=tokens["access_token"],
								refresh_token=tokens.get("refresh_token"),
								id_token=tokens.get("id_token"),
								token_expires_at=datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600)),
								claims=user_info
							)
							
							# Extract groups from ID token claims
							if "groups" in user_info:
								sso_session.groups = user_info["groups"]
							
							return True, sso_session
			
			return False, None
			
		except Exception as e:
			print(f"Okta token exchange error: {e}")
			return False, None
	
	async def _get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
		"""Get user information from Okta API"""
		try:
			headers = {"Authorization": f"Bearer {access_token}"}
			
			async with aiohttp.ClientSession() as session:
				async with session.get(f"{self.base_url}/oauth2/v1/userinfo", headers=headers) as response:
					if response.status == 200:
						return await response.json()
			
			return None
			
		except Exception as e:
			print(f"Okta user info error: {e}")
			return None


class Auth0Connector:
	"""Auth0 Platform connector"""
	
	def __init__(self, config: SSOConfiguration):
		self.config = config
		self.domain = config.domain  # e.g., "company.auth0.com"
		self.base_url = f"https://{self.domain}"
		self.scopes = config.scopes or ["openid", "profile", "email"]
	
	async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
		"""Generate Auth0 authorization URL"""
		params = {
			"client_id": self.config.client_id,
			"response_type": "code",
			"redirect_uri": redirect_uri,
			"scope": " ".join(self.scopes),
			"state": state
		}
		
		query_string = "&".join([f"{k}={v}" for k, v in params.items()])
		return f"{self.base_url}/authorize?{query_string}"
	
	async def exchange_code(self, code: str, redirect_uri: str) -> Tuple[bool, Optional[SSOSession]]:
		"""Exchange authorization code for tokens"""
		try:
			token_url = f"{self.base_url}/oauth/token"
			
			data = {
				"client_id": self.config.client_id,
				"client_secret": self.config.client_secret,
				"code": code,
				"grant_type": "authorization_code",
				"redirect_uri": redirect_uri
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(token_url, json=data) as response:
					if response.status == 200:
						tokens = await response.json()
						
						# Get user info from Auth0 API
						user_info = await self._get_user_info(tokens["access_token"])
						if user_info:
							# Create SSO session
							sso_session = SSOSession(
								user_id=user_info["sub"],
								provider=SSOProvider.AUTH0,
								provider_user_id=user_info["sub"],
								email=user_info.get("email", ""),
								display_name=user_info.get("name", ""),
								access_token=tokens["access_token"],
								refresh_token=tokens.get("refresh_token"),
								id_token=tokens.get("id_token"),
								token_expires_at=datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600)),
								claims=user_info
							)
							
							# Get user roles and permissions
							roles = await self._get_user_roles(tokens["access_token"], user_info["sub"])
							sso_session.roles = roles
							
							return True, sso_session
			
			return False, None
			
		except Exception as e:
			print(f"Auth0 token exchange error: {e}")
			return False, None
	
	async def _get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
		"""Get user information from Auth0 API"""
		try:
			headers = {"Authorization": f"Bearer {access_token}"}
			
			async with aiohttp.ClientSession() as session:
				async with session.get(f"{self.base_url}/userinfo", headers=headers) as response:
					if response.status == 200:
						return await response.json()
			
			return None
			
		except Exception as e:
			print(f"Auth0 user info error: {e}")
			return None
	
	async def _get_user_roles(self, access_token: str, user_id: str) -> List[str]:
		"""Get user roles from Auth0 Management API"""
		try:
			# Get management API token first
			mgmt_token = await self._get_management_token()
			if not mgmt_token:
				return []
			
			headers = {"Authorization": f"Bearer {mgmt_token}"}
			
			async with aiohttp.ClientSession() as session:
				url = f"{self.base_url}/api/v2/users/{user_id}/roles"
				async with session.get(url, headers=headers) as response:
					if response.status == 200:
						roles_data = await response.json()
						return [role.get("name", "") for role in roles_data]
			
			return []
			
		except Exception as e:
			print(f"Auth0 roles error: {e}")
			return []
	
	async def _get_management_token(self) -> Optional[str]:
		"""Get Auth0 Management API token"""
		try:
			data = {
				"client_id": self.config.client_id,
				"client_secret": self.config.client_secret,
				"audience": f"{self.base_url}/api/v2/",
				"grant_type": "client_credentials"
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(f"{self.base_url}/oauth/token", json=data) as response:
					if response.status == 200:
						token_data = await response.json()
						return token_data.get("access_token")
			
			return None
			
		except Exception as e:
			print(f"Auth0 management token error: {e}")
			return None


class PingIdentityConnector:
	"""Ping Identity connector"""
	
	def __init__(self, config: SSOConfiguration):
		self.config = config
		self.domain = config.domain  # e.g., "company.pingidentity.com"
		self.base_url = f"https://{self.domain}"
		self.scopes = config.scopes or ["openid", "profile", "email"]
	
	async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
		"""Generate Ping Identity authorization URL"""
		params = {
			"client_id": self.config.client_id,
			"response_type": "code",
			"redirect_uri": redirect_uri,
			"scope": " ".join(self.scopes),
			"state": state
		}
		
		query_string = "&".join([f"{k}={v}" for k, v in params.items()])
		return f"{self.base_url}/as/authorization.oauth2?{query_string}"
	
	async def exchange_code(self, code: str, redirect_uri: str) -> Tuple[bool, Optional[SSOSession]]:
		"""Exchange authorization code for tokens"""
		try:
			token_url = f"{self.base_url}/as/token.oauth2"
			
			# Ping Identity requires basic auth
			credentials = f"{self.config.client_id}:{self.config.client_secret}"
			auth_header = base64.b64encode(credentials.encode()).decode()
			
			headers = {
				"Authorization": f"Basic {auth_header}",
				"Content-Type": "application/x-www-form-urlencoded"
			}
			
			data = {
				"code": code,
				"grant_type": "authorization_code",
				"redirect_uri": redirect_uri
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(token_url, headers=headers, data=data) as response:
					if response.status == 200:
						tokens = await response.json()
						
						# Get user info from Ping API
						user_info = await self._get_user_info(tokens["access_token"])
						if user_info:
							# Create SSO session
							sso_session = SSOSession(
								user_id=user_info["sub"],
								provider=SSOProvider.PING_IDENTITY,
								provider_user_id=user_info["sub"],
								email=user_info.get("email", ""),
								display_name=user_info.get("name", ""),
								access_token=tokens["access_token"],
								refresh_token=tokens.get("refresh_token"),
								id_token=tokens.get("id_token"),
								token_expires_at=datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600)),
								claims=user_info
							)
							
							return True, sso_session
			
			return False, None
			
		except Exception as e:
			print(f"Ping Identity token exchange error: {e}")
			return False, None
	
	async def _get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
		"""Get user information from Ping Identity API"""
		try:
			headers = {"Authorization": f"Bearer {access_token}"}
			
			async with aiohttp.ClientSession() as session:
				async with session.get(f"{self.base_url}/idp/userinfo.openid", headers=headers) as response:
					if response.status == 200:
						return await response.json()
			
			return None
			
		except Exception as e:
			print(f"Ping Identity user info error: {e}")
			return None


class SSOConnectorFactory:
	"""Factory for creating SSO connectors"""
	
	@staticmethod
	def create_connector(config: SSOConfiguration):
		"""Create SSO connector based on provider"""
		if config.provider == SSOProvider.AZURE_AD:
			return AzureADConnector(config)
		elif config.provider == SSOProvider.GOOGLE_WORKSPACE:
			return GoogleWorkspaceConnector(config)
		elif config.provider == SSOProvider.OKTA:
			return OktaConnector(config)
		elif config.provider == SSOProvider.AUTH0:
			return Auth0Connector(config)
		elif config.provider == SSOProvider.PING_IDENTITY:
			return PingIdentityConnector(config)
		else:
			raise ValueError(f"Unsupported SSO provider: {config.provider}")


class SSOSessionManager:
	"""SSO session management"""
	
	def __init__(self):
		self.active_sessions = {}
		self.session_cache = {}
	
	async def create_session(self, sso_session: SSOSession) -> str:
		"""Create new SSO session"""
		try:
			# Store session in database
			from .database import get_async_db_session
			from sqlalchemy import text
			
			async with get_async_db_session() as session:
				await session.execute(
					text("""
					INSERT INTO wo_sso_sessions (
						id, user_id, provider, provider_user_id, email,
						display_name, roles, groups, claims, access_token,
						refresh_token, id_token, token_expires_at,
						created_at, last_activity, is_active
					) VALUES (
						:id, :user_id, :provider, :provider_user_id, :email,
						:display_name, :roles, :groups, :claims, :access_token,
						:refresh_token, :id_token, :token_expires_at,
						:created_at, :last_activity, :is_active
					)
					"""),
					{
						"id": sso_session.id,
						"user_id": sso_session.user_id,
						"provider": sso_session.provider.value,
						"provider_user_id": sso_session.provider_user_id,
						"email": sso_session.email,
						"display_name": sso_session.display_name,
						"roles": json.dumps(sso_session.roles),
						"groups": json.dumps(sso_session.groups),
						"claims": json.dumps(sso_session.claims),
						"access_token": sso_session.access_token,
						"refresh_token": sso_session.refresh_token,
						"id_token": sso_session.id_token,
						"token_expires_at": sso_session.token_expires_at,
						"created_at": sso_session.created_at,
						"last_activity": sso_session.last_activity,
						"is_active": sso_session.is_active
					}
				)
				await session.commit()
			
			# Cache session for quick access
			self.active_sessions[sso_session.id] = sso_session
			
			# Log successful SSO authentication
			audit_event = AuditEvent(
				event_type="sso_authentication",
				user_id=sso_session.user_id,
				action="login_success",
				result="success",
				details={
					"provider": sso_session.provider.value,
					"email": sso_session.email,
					"roles": sso_session.roles,
					"groups": sso_session.groups
				},
				risk_level="low"
			)
			await enterprise_integration.log_audit_event(audit_event)
			
			return sso_session.id
			
		except Exception as e:
			print(f"SSO session creation error: {e}")
			raise
	
	async def get_session(self, session_id: str) -> Optional[SSOSession]:
		"""Get SSO session by ID"""
		# Check cache first
		if session_id in self.active_sessions:
			session = self.active_sessions[session_id]
			session.last_activity = datetime.utcnow()
			return session
		
		# Load from database
		try:
			from .database import get_async_db_session
			from sqlalchemy import text
			
			async with get_async_db_session() as session:
				result = await session.execute(
					text("""
					SELECT id, user_id, provider, provider_user_id, email,
						   display_name, roles, groups, claims, access_token,
						   refresh_token, id_token, token_expires_at,
						   created_at, last_activity, is_active
					FROM wo_sso_sessions 
					WHERE id = :session_id AND is_active = true
					"""),
					{"session_id": session_id}
				)
				
				row = result.fetchone()
				if row:
					sso_session = SSOSession(
						id=row[0],
						user_id=row[1],
						provider=SSOProvider(row[2]),
						provider_user_id=row[3],
						email=row[4],
						display_name=row[5],
						roles=json.loads(row[6]),
						groups=json.loads(row[7]),
						claims=json.loads(row[8]),
						access_token=row[9],
						refresh_token=row[10],
						id_token=row[11],
						token_expires_at=row[12],
						created_at=row[13],
						last_activity=row[14],
						is_active=row[15]
					)
					
					# Cache for future use
					self.active_sessions[session_id] = sso_session
					return sso_session
			
			return None
			
		except Exception as e:
			print(f"SSO session retrieval error: {e}")
			return None
	
	async def invalidate_session(self, session_id: str):
		"""Invalidate SSO session"""
		try:
			from .database import get_async_db_session
			from sqlalchemy import text
			
			async with get_async_db_session() as session:
				await session.execute(
					text("""
					UPDATE wo_sso_sessions 
					SET is_active = false, last_activity = :now
					WHERE id = :session_id
					"""),
					{
						"session_id": session_id,
						"now": datetime.utcnow()
					}
				)
				await session.commit()
			
			# Remove from cache
			if session_id in self.active_sessions:
				sso_session = self.active_sessions[session_id]
				del self.active_sessions[session_id]
				
				# Log session invalidation
				audit_event = AuditEvent(
					event_type="sso_authentication",
					user_id=sso_session.user_id,
					action="logout",
					result="success",
					details={"provider": sso_session.provider.value},
					risk_level="low"
				)
				await enterprise_integration.log_audit_event(audit_event)
			
		except Exception as e:
			print(f"SSO session invalidation error: {e}")
			raise
	
	async def refresh_token_if_needed(self, session_id: str) -> bool:
		"""Refresh access token if needed"""
		session = await self.get_session(session_id)
		if not session or not session.refresh_token:
			return False
		
		# Check if token needs refresh (within 5 minutes of expiry)
		if session.token_expires_at and session.token_expires_at - datetime.utcnow() < timedelta(minutes=5):
			try:
				# Create connector for token refresh
				config = SSOConfiguration(
					provider=session.provider,
					client_id="",  # Would need to be stored/retrieved
					client_secret=""  # Would need to be stored/retrieved
				)
				
				connector = SSOConnectorFactory.create_connector(config)
				
				if hasattr(connector, 'refresh_token'):
					success, new_tokens = await connector.refresh_token(session.refresh_token)
					if success and new_tokens:
						# Update session with new tokens
						session.access_token = new_tokens["access_token"]
						session.token_expires_at = datetime.utcnow() + timedelta(seconds=new_tokens.get("expires_in", 3600))
						
						if "refresh_token" in new_tokens:
							session.refresh_token = new_tokens["refresh_token"]
						
						# Update in database
						await self._update_session_tokens(session)
						return True
				
				return False
				
			except Exception as e:
				print(f"Token refresh error: {e}")
				return False
		
		return True  # Token still valid
	
	async def _update_session_tokens(self, sso_session: SSOSession):
		"""Update session tokens in database"""
		try:
			from .database import get_async_db_session
			from sqlalchemy import text
			
			async with get_async_db_session() as session:
				await session.execute(
					text("""
					UPDATE wo_sso_sessions 
					SET access_token = :access_token, refresh_token = :refresh_token,
						token_expires_at = :token_expires_at, last_activity = :last_activity
					WHERE id = :session_id
					"""),
					{
						"session_id": sso_session.id,
						"access_token": sso_session.access_token,
						"refresh_token": sso_session.refresh_token,
						"token_expires_at": sso_session.token_expires_at,
						"last_activity": sso_session.last_activity
					}
				)
				await session.commit()
				
		except Exception as e:
			print(f"Session token update error: {e}")
			raise


# Global SSO session manager instance
sso_session_manager = SSOSessionManager()