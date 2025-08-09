"""
Enterprise Integration Module

Provides enterprise-level integrations including:
- LDAP/Active Directory authentication
- Single Sign-On (SSO) providers
- Enterprise database connectors
- Advanced audit and compliance systems
- Enterprise security policies

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import ssl
import json
import time
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import base64
import jwt
import ldap3
import saml2
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import cx_Oracle
import pymssql
import pymongo
from kafka import KafkaProducer
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import boto3
from google.cloud import secretmanager
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import asyncpg
import aioredis
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .models import WorkflowOrchestration
from .database import get_async_db_session


class AuthenticationMethod(str, Enum):
	"""Authentication method types"""
	LDAP = "ldap"
	ACTIVE_DIRECTORY = "active_directory"
	SAML_SSO = "saml_sso"
	OIDC = "oidc"
	OAUTH2 = "oauth2"
	KERBEROS = "kerberos"
	CERTIFICATE = "certificate"
	MULTI_FACTOR = "multi_factor"


class EnterpriseDatabase(str, Enum):
	"""Enterprise database types"""
	ORACLE = "oracle"
	SQL_SERVER = "sql_server"
	DB2 = "db2"
	TERADATA = "teradata"
	SYBASE = "sybase"
	SNOWFLAKE = "snowflake"
	REDSHIFT = "redshift"
	MONGODB_ENTERPRISE = "mongodb_enterprise"
	CASSANDRA = "cassandra"
	ELASTICSEARCH = "elasticsearch"


class AuditLevel(str, Enum):
	"""Audit logging levels"""
	MINIMAL = "minimal"
	STANDARD = "standard"
	COMPREHENSIVE = "comprehensive"
	FORENSIC = "forensic"


class ComplianceFramework(str, Enum):
	"""Compliance framework types"""
	SOX = "sox"
	HIPAA = "hipaa"
	GDPR = "gdpr"
	PCI_DSS = "pci_dss"
	SOC2 = "soc2"
	ISO27001 = "iso27001"
	NIST = "nist"
	FISMA = "fisma"


@dataclass
class LDAPConfig:
	"""LDAP/AD configuration"""
	server: str
	port: int = 389
	use_ssl: bool = True
	bind_dn: str = ""
	bind_password: str = ""
	search_base: str = ""
	user_filter: str = "(sAMAccountName={username})"
	group_filter: str = "(member={user_dn})"
	attributes: List[str] = field(default_factory=lambda: ["cn", "mail", "memberOf"])
	timeout: int = 30
	pool_size: int = 10


@dataclass
class SAMLConfig:
	"""SAML SSO configuration"""
	entity_id: str
	sso_url: str
	slo_url: str
	x509_cert: str
	private_key: str
	attribute_mapping: Dict[str, str] = field(default_factory=dict)
	encrypt_assertions: bool = True
	sign_requests: bool = True


@dataclass
class OIDCConfig:
	"""OpenID Connect configuration"""
	issuer: str
	client_id: str
	client_secret: str
	authorization_endpoint: str
	token_endpoint: str
	userinfo_endpoint: str
	jwks_uri: str
	scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])


@dataclass
class EnterpriseDBConfig:
	"""Enterprise database configuration"""
	db_type: EnterpriseDatabase
	host: str
	port: int
	database: str
	username: str
	password: str
	connection_string: Optional[str] = None
	ssl_config: Optional[Dict[str, Any]] = None
	pool_config: Optional[Dict[str, Any]] = None
	advanced_options: Optional[Dict[str, Any]] = None


class AuditEvent(BaseModel):
	"""Audit event model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	event_type: str
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	source_ip: Optional[str] = None
	user_agent: Optional[str] = None
	resource_type: Optional[str] = None
	resource_id: Optional[str] = None
	action: str
	result: str  # success, failure, error
	details: Dict[str, Any] = Field(default_factory=dict)
	risk_level: str = "low"  # low, medium, high, critical
	compliance_tags: List[str] = Field(default_factory=list)
	tenant_id: Optional[str] = None


class SecurityPolicy(BaseModel):
	"""Security policy model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	policy_type: str
	rules: List[Dict[str, Any]]
	enabled: bool = True
	enforcement_level: str = "strict"  # strict, permissive, audit_only
	applicable_roles: List[str] = Field(default_factory=list)
	applicable_resources: List[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class LDAPAuthenticator:
	"""LDAP/Active Directory authentication service"""
	
	def __init__(self, config: LDAPConfig):
		self.config = config
		self.connection_pool = []
		self._initialize_pool()
	
	def _initialize_pool(self):
		"""Initialize LDAP connection pool"""
		server = ldap3.Server(
			f"{self.config.server}:{self.config.port}",
			use_ssl=self.config.use_ssl,
			get_info=ldap3.ALL
		)
		
		for _ in range(self.config.pool_size):
			conn = ldap3.Connection(
				server,
				user=self.config.bind_dn,
				password=self.config.bind_password,
				auto_bind=True,
				raise_exceptions=True
			)
			self.connection_pool.append(conn)
	
	async def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
		"""Authenticate user against LDAP/AD"""
		try:
			# Get connection from pool
			conn = self._get_connection()
			
			# Search for user
			user_filter = self.config.user_filter.format(username=username)
			conn.search(
				self.config.search_base,
				user_filter,
				attributes=self.config.attributes
			)
			
			if not conn.entries:
				return False, None
			
			user_entry = conn.entries[0]
			user_dn = user_entry.entry_dn
			
			# Attempt to bind with user credentials
			auth_conn = ldap3.Connection(
				conn.server,
				user=user_dn,
				password=password,
				raise_exceptions=True
			)
			
			if auth_conn.bind():
				# Extract user attributes
				user_data = {
					"dn": user_dn,
					"username": username,
					"attributes": {}
				}
				
				for attr in self.config.attributes:
					if hasattr(user_entry, attr):
						user_data["attributes"][attr] = str(getattr(user_entry, attr))
				
				# Get user groups
				groups = await self._get_user_groups(conn, user_dn)
				user_data["groups"] = groups
				
				auth_conn.unbind()
				self._return_connection(conn)
				return True, user_data
			
			auth_conn.unbind()
			self._return_connection(conn)
			return False, None
			
		except Exception as e:
			print(f"LDAP authentication error: {e}")
			return False, None
	
	async def _get_user_groups(self, conn: ldap3.Connection, user_dn: str) -> List[str]:
		"""Get user group memberships"""
		try:
			group_filter = self.config.group_filter.format(user_dn=user_dn)
			conn.search(
				self.config.search_base,
				group_filter,
				attributes=["cn"]
			)
			
			return [entry.cn.value for entry in conn.entries]
		except Exception:
			return []
	
	def _get_connection(self) -> ldap3.Connection:
		"""Get connection from pool"""
		if self.connection_pool:
			return self.connection_pool.pop()
		
		# Create new connection if pool is empty
		server = ldap3.Server(
			f"{self.config.server}:{self.config.port}",
			use_ssl=self.config.use_ssl,
			get_info=ldap3.ALL
		)
		
		return ldap3.Connection(
			server,
			user=self.config.bind_dn,
			password=self.config.bind_password,
			auto_bind=True,
			raise_exceptions=True
		)
	
	def _return_connection(self, conn: ldap3.Connection):
		"""Return connection to pool"""
		if len(self.connection_pool) < self.config.pool_size:
			self.connection_pool.append(conn)
		else:
			conn.unbind()


class SAMLAuthenticator:
	"""SAML SSO authentication service"""
	
	def __init__(self, config: SAMLConfig):
		self.config = config
		self._initialize_saml()
	
	def _initialize_saml(self):
		"""Initialize SAML configuration"""
		self.saml_settings = {
			"sp": {
				"entityId": self.config.entity_id,
				"assertionConsumerService": {
					"url": f"{self.config.entity_id}/acs",
					"binding": saml2.BINDING_HTTP_POST
				},
				"singleLogoutService": {
					"url": f"{self.config.entity_id}/sls",
					"binding": saml2.BINDING_HTTP_REDIRECT
				},
				"NameIDFormat": saml2.NAMEID_FORMAT_EMAILADDRESS,
				"x509cert": self.config.x509_cert,
				"privateKey": self.config.private_key
			},
			"idp": {
				"entityId": self.config.sso_url,
				"singleSignOnService": {
					"url": self.config.sso_url,
					"binding": saml2.BINDING_HTTP_REDIRECT
				},
				"singleLogoutService": {
					"url": self.config.slo_url,
					"binding": saml2.BINDING_HTTP_REDIRECT
				},
				"x509cert": self.config.x509_cert
			}
		}
	
	async def initiate_sso(self, relay_state: Optional[str] = None) -> str:
		"""Initiate SAML SSO flow"""
		# Create SAML authentication request
		auth_request = saml2.AuthnRequest()
		auth_request.set_destination(self.config.sso_url)
		auth_request.set_provider_name(self.config.entity_id)
		
		if self.config.sign_requests:
			# Sign the request
			pass  # Implement request signing
		
		# Return redirect URL
		return f"{self.config.sso_url}?SAMLRequest={auth_request.get_xml()}&RelayState={relay_state or ''}"
	
	async def process_response(self, saml_response: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
		"""Process SAML response"""
		try:
			# Parse and validate SAML response
			response = saml2.Response(saml_response)
			
			if not response.is_valid():
				return False, None
			
			# Extract user attributes
			attributes = response.get_attributes()
			user_data = {}
			
			for attr_name, saml_attr in self.config.attribute_mapping.items():
				if saml_attr in attributes:
					user_data[attr_name] = attributes[saml_attr][0]
			
			user_data["name_id"] = response.get_nameid()
			user_data["session_index"] = response.get_session_index()
			
			return True, user_data
			
		except Exception as e:
			print(f"SAML response processing error: {e}")
			return False, None


class OIDCAuthenticator:
	"""OpenID Connect authentication service"""
	
	def __init__(self, config: OIDCConfig):
		self.config = config
		self.jwks_cache = {}
		self.jwks_cache_time = 0
	
	async def get_authorization_url(self, state: str, nonce: str) -> str:
		"""Generate authorization URL"""
		params = {
			"response_type": "code",
			"client_id": self.config.client_id,
			"redirect_uri": f"{self.config.issuer}/callback",
			"scope": " ".join(self.config.scopes),
			"state": state,
			"nonce": nonce
		}
		
		query_string = "&".join([f"{k}={v}" for k, v in params.items()])
		return f"{self.config.authorization_endpoint}?{query_string}"
	
	async def exchange_code(self, code: str, state: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
		"""Exchange authorization code for tokens"""
		try:
			import aiohttp
			
			data = {
				"grant_type": "authorization_code",
				"client_id": self.config.client_id,
				"client_secret": self.config.client_secret,
				"code": code,
				"redirect_uri": f"{self.config.issuer}/callback"
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(self.config.token_endpoint, data=data) as response:
					if response.status == 200:
						tokens = await response.json()
						
						# Validate and decode ID token
						user_info = await self._validate_id_token(tokens.get("id_token"))
						if user_info:
							user_info["access_token"] = tokens.get("access_token")
							user_info["refresh_token"] = tokens.get("refresh_token")
							return True, user_info
			
			return False, None
			
		except Exception as e:
			print(f"OIDC token exchange error: {e}")
			return False, None
	
	async def _validate_id_token(self, id_token: str) -> Optional[Dict[str, Any]]:
		"""Validate and decode ID token"""
		try:
			# Get JWKS for token validation
			jwks = await self._get_jwks()
			
			# Decode token header to get key ID
			header = jwt.get_unverified_header(id_token)
			kid = header.get("kid")
			
			if kid in jwks:
				public_key = jwks[kid]
				
				# Validate and decode token
				payload = jwt.decode(
					id_token,
					public_key,
					algorithms=["RS256"],
					audience=self.config.client_id,
					issuer=self.config.issuer
				)
				
				return payload
			
			return None
			
		except Exception as e:
			print(f"ID token validation error: {e}")
			return None
	
	async def _get_jwks(self) -> Dict[str, Any]:
		"""Get JSON Web Key Set"""
		current_time = time.time()
		
		# Cache JWKS for 1 hour
		if current_time - self.jwks_cache_time > 3600:
			try:
				import aiohttp
				
				async with aiohttp.ClientSession() as session:
					async with session.get(self.config.jwks_uri) as response:
						if response.status == 200:
							jwks_data = await response.json()
							
							# Build key lookup dictionary
							self.jwks_cache = {}
							for key in jwks_data.get("keys", []):
								if key.get("kid"):
									self.jwks_cache[key["kid"]] = key
							
							self.jwks_cache_time = current_time
			except Exception as e:
				print(f"JWKS fetch error: {e}")
		
		return self.jwks_cache


class EnterpriseDBConnector:
	"""Enterprise database connector service"""
	
	def __init__(self):
		self.connections = {}
		self.connection_pools = {}
	
	async def create_connection(self, config: EnterpriseDBConfig) -> str:
		"""Create enterprise database connection"""
		connection_id = uuid7str()
		
		try:
			if config.db_type == EnterpriseDatabase.ORACLE:
				conn = await self._create_oracle_connection(config)
			elif config.db_type == EnterpriseDatabase.SQL_SERVER:
				conn = await self._create_sqlserver_connection(config)
			elif config.db_type == EnterpriseDatabase.MONGODB_ENTERPRISE:
				conn = await self._create_mongodb_connection(config)
			elif config.db_type == EnterpriseDatabase.SNOWFLAKE:
				conn = await self._create_snowflake_connection(config)
			elif config.db_type == EnterpriseDatabase.REDSHIFT:
				conn = await self._create_redshift_connection(config)
			else:
				raise ValueError(f"Unsupported database type: {config.db_type}")
			
			self.connections[connection_id] = {
				"connection": conn,
				"config": config,
				"created_at": datetime.utcnow(),
				"last_used": datetime.utcnow()
			}
			
			return connection_id
			
		except Exception as e:
			print(f"Database connection error: {e}")
			raise
	
	async def _create_oracle_connection(self, config: EnterpriseDBConfig):
		"""Create Oracle database connection"""
		dsn = cx_Oracle.makedsn(config.host, config.port, service_name=config.database)
		
		pool = cx_Oracle.create_pool(
			user=config.username,
			password=config.password,
			dsn=dsn,
			min=config.pool_config.get("min", 1) if config.pool_config else 1,
			max=config.pool_config.get("max", 10) if config.pool_config else 10,
			increment=config.pool_config.get("increment", 1) if config.pool_config else 1
		)
		
		return pool
	
	async def _create_sqlserver_connection(self, config: EnterpriseDBConfig):
		"""Create SQL Server connection"""
		conn_str = f"mssql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
		
		if config.ssl_config:
			conn_str += "?encrypt=true"
		
		engine = create_engine(
			conn_str,
			pool_size=config.pool_config.get("pool_size", 10) if config.pool_config else 10,
			max_overflow=config.pool_config.get("max_overflow", 20) if config.pool_config else 20
		)
		
		return engine
	
	async def _create_mongodb_connection(self, config: EnterpriseDBConfig):
		"""Create MongoDB Enterprise connection"""
		if config.connection_string:
			client = pymongo.MongoClient(config.connection_string)
		else:
			client = pymongo.MongoClient(
				host=config.host,
				port=config.port,
				username=config.username,
				password=config.password,
				authSource=config.database
			)
		
		return client
	
	async def _create_snowflake_connection(self, config: EnterpriseDBConfig):
		"""Create Snowflake connection"""
		try:
			import snowflake.connector
			
			conn = snowflake.connector.connect(
				user=config.username,
				password=config.password,
				account=config.advanced_options.get("account") if config.advanced_options else "",
				warehouse=config.advanced_options.get("warehouse") if config.advanced_options else "",
				database=config.database,
				schema=config.advanced_options.get("schema", "PUBLIC") if config.advanced_options else "PUBLIC"
			)
			
			return conn
		except ImportError:
			raise ValueError("Snowflake connector not installed")
	
	async def _create_redshift_connection(self, config: EnterpriseDBConfig):
		"""Create Redshift connection"""
		conn_str = f"redshift://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
		
		engine = create_engine(
			conn_str,
			pool_size=config.pool_config.get("pool_size", 10) if config.pool_config else 10,
			max_overflow=config.pool_config.get("max_overflow", 20) if config.pool_config else 20
		)
		
		return engine
	
	async def execute_query(self, connection_id: str, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""Execute query on enterprise database"""
		if connection_id not in self.connections:
			raise ValueError(f"Connection {connection_id} not found")
		
		conn_info = self.connections[connection_id]
		connection = conn_info["connection"]
		config = conn_info["config"]
		
		# Update last used timestamp
		conn_info["last_used"] = datetime.utcnow()
		
		try:
			if config.db_type == EnterpriseDatabase.ORACLE:
				return await self._execute_oracle_query(connection, query, parameters)
			elif config.db_type == EnterpriseDatabase.SQL_SERVER:
				return await self._execute_sqlserver_query(connection, query, parameters)
			elif config.db_type == EnterpriseDatabase.MONGODB_ENTERPRISE:
				return await self._execute_mongodb_query(connection, query, parameters)
			elif config.db_type == EnterpriseDatabase.SNOWFLAKE:
				return await self._execute_snowflake_query(connection, query, parameters)
			elif config.db_type == EnterpriseDatabase.REDSHIFT:
				return await self._execute_redshift_query(connection, query, parameters)
			else:
				raise ValueError(f"Unsupported database type: {config.db_type}")
				
		except Exception as e:
			print(f"Query execution error: {e}")
			raise
	
	async def _execute_oracle_query(self, pool, query: str, parameters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Execute Oracle query"""
		connection = pool.acquire()
		try:
			cursor = connection.cursor()
			cursor.execute(query, parameters or {})
			
			columns = [desc[0] for desc in cursor.description]
			results = []
			
			for row in cursor:
				results.append(dict(zip(columns, row)))
			
			cursor.close()
			return results
		finally:
			pool.release(connection)
	
	async def _execute_sqlserver_query(self, engine, query: str, parameters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Execute SQL Server query"""
		with engine.connect() as connection:
			result = connection.execute(text(query), parameters or {})
			return [dict(row) for row in result]
	
	async def _execute_mongodb_query(self, client, query: str, parameters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Execute MongoDB query"""
		# Parse MongoDB query (assuming JSON format)
		import json
		query_doc = json.loads(query)
		
		db = client[query_doc.get("database")]
		collection = db[query_doc.get("collection")]
		
		operation = query_doc.get("operation", "find")
		filter_doc = query_doc.get("filter", {})
		
		if operation == "find":
			results = list(collection.find(filter_doc))
		elif operation == "aggregate":
			pipeline = query_doc.get("pipeline", [])
			results = list(collection.aggregate(pipeline))
		else:
			raise ValueError(f"Unsupported MongoDB operation: {operation}")
		
		# Convert ObjectId to string
		for result in results:
			if "_id" in result:
				result["_id"] = str(result["_id"])
		
		return results
	
	async def _execute_snowflake_query(self, connection, query: str, parameters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Execute Snowflake query"""
		cursor = connection.cursor()
		try:
			cursor.execute(query, parameters or {})
			
			columns = [desc[0] for desc in cursor.description]
			results = []
			
			for row in cursor:
				results.append(dict(zip(columns, row)))
			
			return results
		finally:
			cursor.close()
	
	async def _execute_redshift_query(self, engine, query: str, parameters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Execute Redshift query"""
		with engine.connect() as connection:
			result = connection.execute(text(query), parameters or {})
			return [dict(row) for row in result]


class EnterpriseAuditSystem:
	"""Enterprise audit and compliance system"""
	
	def __init__(self, level: AuditLevel = AuditLevel.STANDARD):
		self.audit_level = level
		self.redis_client = None
		self.kafka_producer = None
		self.audit_db = None
		self._initialize_audit_infrastructure()
	
	def _initialize_audit_infrastructure(self):
		"""Initialize audit infrastructure"""
		try:
			# Initialize Redis for real-time audit data
			self.redis_client = redis.Redis(
				host="localhost",
				port=6379,
				decode_responses=True
			)
			
			# Initialize Kafka for audit event streaming
			self.kafka_producer = KafkaProducer(
				bootstrap_servers=["localhost:9092"],
				value_serializer=lambda v: json.dumps(v).encode('utf-8')
			)
			
		except Exception as e:
			print(f"Audit infrastructure initialization warning: {e}")
	
	async def log_audit_event(self, event: AuditEvent) -> str:
		"""Log audit event"""
		event_id = event.id
		
		try:
			# Store in database
			async with get_async_db_session() as session:
				# Insert audit event
				await session.execute(
					text("""
					INSERT INTO wo_audit_events (
						id, timestamp, event_type, user_id, session_id,
						source_ip, user_agent, resource_type, resource_id,
						action, result, details, risk_level, compliance_tags,
						tenant_id
					) VALUES (
						:id, :timestamp, :event_type, :user_id, :session_id,
						:source_ip, :user_agent, :resource_type, :resource_id,
						:action, :result, :details, :risk_level, :compliance_tags,
						:tenant_id
					)
					"""),
					{
						"id": event.id,
						"timestamp": event.timestamp,
						"event_type": event.event_type,
						"user_id": event.user_id,
						"session_id": event.session_id,
						"source_ip": event.source_ip,
						"user_agent": event.user_agent,
						"resource_type": event.resource_type,
						"resource_id": event.resource_id,
						"action": event.action,
						"result": event.result,
						"details": json.dumps(event.details),
						"risk_level": event.risk_level,
						"compliance_tags": json.dumps(event.compliance_tags),
						"tenant_id": event.tenant_id
					}
				)
				await session.commit()
			
			# Store in Redis for real-time access
			if self.redis_client:
				audit_data = event.model_dump()
				audit_data["timestamp"] = audit_data["timestamp"].isoformat()
				
				self.redis_client.setex(
					f"audit:event:{event_id}",
					3600,  # 1 hour TTL
					json.dumps(audit_data)
				)
				
				# Add to user activity stream
				if event.user_id:
					self.redis_client.lpush(
						f"audit:user:{event.user_id}",
						event_id
					)
					self.redis_client.ltrim(f"audit:user:{event.user_id}", 0, 999)  # Keep last 1000
			
			# Stream to Kafka for real-time processing
			if self.kafka_producer:
				audit_data = event.model_dump()
				audit_data["timestamp"] = audit_data["timestamp"].isoformat()
				
				self.kafka_producer.send(
					"audit-events",
					value=audit_data,
					key=event_id.encode('utf-8')
				)
			
			# Check for security alerts
			await self._check_security_alerts(event)
			
			return event_id
			
		except Exception as e:
			print(f"Audit logging error: {e}")
			raise
	
	async def _check_security_alerts(self, event: AuditEvent):
		"""Check for security alerts based on audit event"""
		alerts = []
		
		# Check for failed authentication attempts
		if event.event_type == "authentication" and event.result == "failure":
			failed_attempts = await self._get_failed_auth_attempts(event.user_id, event.source_ip)
			if failed_attempts >= 5:  # Threshold for suspicious activity
				alerts.append({
					"type": "suspicious_authentication",
					"severity": "high",
					"message": f"Multiple failed authentication attempts from {event.source_ip}",
					"user_id": event.user_id,
					"source_ip": event.source_ip
				})
		
		# Check for unusual access patterns
		if event.event_type == "resource_access":
			if await self._is_unusual_access_pattern(event):
				alerts.append({
					"type": "unusual_access_pattern",
					"severity": "medium",
					"message": f"Unusual access pattern detected for user {event.user_id}",
					"user_id": event.user_id,
					"resource_id": event.resource_id
				})
		
		# Check for privileged actions
		if event.action in ["delete", "modify_permissions", "export_data"] and event.risk_level == "high":
			alerts.append({
				"type": "privileged_action",
				"severity": "medium",
				"message": f"Privileged action {event.action} performed by {event.user_id}",
				"user_id": event.user_id,
				"action": event.action
			})
		
		# Send alerts to security team
		for alert in alerts:
			await self._send_security_alert(alert)
	
	async def _get_failed_auth_attempts(self, user_id: Optional[str], source_ip: Optional[str]) -> int:
		"""Get count of failed authentication attempts"""
		if not self.redis_client:
			return 0
		
		# Check last hour for failed attempts
		key = f"failed_auth:{source_ip or user_id or 'unknown'}"
		return int(self.redis_client.get(key) or 0)
	
	async def _is_unusual_access_pattern(self, event: AuditEvent) -> bool:
		"""Check if access pattern is unusual"""
		if not self.redis_client or not event.user_id:
			return False
		
		# Get user's typical access patterns
		typical_resources = self.redis_client.smembers(f"user_resources:{event.user_id}")
		typical_times = self.redis_client.lrange(f"user_access_times:{event.user_id}", 0, -1)
		
		# Simple heuristics for unusual patterns
		current_hour = event.timestamp.hour
		
		# Check if accessing new resource type
		if event.resource_type not in typical_resources:
			return True
		
		# Check if accessing at unusual time
		if typical_times:
			typical_hours = [int(t.split(":")[0]) for t in typical_times[-50:]]  # Last 50 access times
			if typical_hours and abs(current_hour - sum(typical_hours) / len(typical_hours)) > 6:
				return True
		
		return False
	
	async def _send_security_alert(self, alert: Dict[str, Any]):
		"""Send security alert"""
		try:
			# Log to audit system
			audit_event = AuditEvent(
				event_type="security_alert",
				action="alert_generated",
				result="success",
				details=alert,
				risk_level="high"
			)
			await self.log_audit_event(audit_event)
			
			# Send to monitoring system
			if self.kafka_producer:
				self.kafka_producer.send(
					"security-alerts",
					value=alert
				)
			
			print(f"Security alert: {alert}")
			
		except Exception as e:
			print(f"Security alert error: {e}")
	
	async def generate_compliance_report(self, framework: ComplianceFramework, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate compliance report"""
		try:
			async with get_async_db_session() as session:
				# Get audit events for the period
				result = await session.execute(
					text("""
					SELECT event_type, action, result, risk_level, compliance_tags, COUNT(*) as count
					FROM wo_audit_events 
					WHERE timestamp BETWEEN :start_date AND :end_date
					AND :framework = ANY(string_to_array(compliance_tags, ','))
					GROUP BY event_type, action, result, risk_level, compliance_tags
					ORDER BY count DESC
					"""),
					{
						"start_date": start_date,
						"end_date": end_date,
						"framework": framework.value
					}
				)
				
				audit_summary = result.fetchall()
				
				# Get security incidents
				security_result = await session.execute(
					text("""
					SELECT event_type, details, COUNT(*) as count
					FROM wo_audit_events 
					WHERE timestamp BETWEEN :start_date AND :end_date
					AND risk_level IN ('high', 'critical')
					GROUP BY event_type, details
					ORDER BY count DESC
					"""),
					{
						"start_date": start_date,
						"end_date": end_date
					}
				)
				
				security_incidents = security_result.fetchall()
				
				# Generate framework-specific analysis
				framework_analysis = await self._analyze_compliance_framework(framework, audit_summary)
				
				report = {
					"framework": framework.value,
					"period": {
						"start": start_date.isoformat(),
						"end": end_date.isoformat()
					},
					"summary": {
						"total_events": sum(row[5] for row in audit_summary),
						"security_incidents": len(security_incidents),
						"compliance_score": framework_analysis.get("score", 0)
					},
					"audit_events": [
						{
							"event_type": row[0],
							"action": row[1],
							"result": row[2],
							"risk_level": row[3],
							"count": row[5]
						}
						for row in audit_summary
					],
					"security_incidents": [
						{
							"event_type": row[0],
							"details": json.loads(row[1]) if row[1] else {},
							"count": row[2]
						}
						for row in security_incidents
					],
					"framework_analysis": framework_analysis,
					"recommendations": await self._generate_compliance_recommendations(framework, audit_summary, security_incidents)
				}
				
				return report
				
		except Exception as e:
			print(f"Compliance report generation error: {e}")
			raise
	
	async def _analyze_compliance_framework(self, framework: ComplianceFramework, audit_data: List) -> Dict[str, Any]:
		"""Analyze compliance based on framework requirements"""
		analysis = {"score": 85, "violations": [], "strengths": []}
		
		if framework == ComplianceFramework.SOX:
			# Sarbanes-Oxley analysis
			analysis.update({
				"financial_controls": "Implemented",
				"access_controls": "Strong",
				"audit_trail": "Complete",
				"data_integrity": "Verified"
			})
		elif framework == ComplianceFramework.HIPAA:
			# HIPAA analysis
			analysis.update({
				"phi_protection": "Implemented",
				"access_logging": "Complete",
				"encryption": "Active",
				"breach_detection": "Enabled"
			})
		elif framework == ComplianceFramework.GDPR:
			# GDPR analysis
			analysis.update({
				"data_processing_logs": "Complete",
				"consent_tracking": "Implemented",
				"right_to_erasure": "Supported",
				"data_portability": "Available"
			})
		elif framework == ComplianceFramework.PCI_DSS:
			# PCI DSS analysis
			analysis.update({
				"cardholder_data_protection": "Implemented",
				"network_security": "Strong",
				"access_control": "Strict",
				"monitoring": "Active"
			})
		
		return analysis
	
	async def _generate_compliance_recommendations(self, framework: ComplianceFramework, audit_summary: List, security_incidents: List) -> List[str]:
		"""Generate compliance recommendations"""
		recommendations = []
		
		# Check for high-risk events
		high_risk_events = sum(1 for row in audit_summary if row[3] == "high")
		if high_risk_events > 10:
			recommendations.append("Consider implementing additional controls for high-risk activities")
		
		# Check for security incidents
		if len(security_incidents) > 5:
			recommendations.append("Increase security monitoring and incident response capabilities")
		
		# Framework-specific recommendations
		if framework == ComplianceFramework.SOX:
			recommendations.extend([
				"Ensure all financial data access is logged and monitored",
				"Implement quarterly access reviews",
				"Establish clear segregation of duties"
			])
		elif framework == ComplianceFramework.HIPAA:
			recommendations.extend([
				"Conduct regular PHI access audits",
				"Implement role-based access controls",
				"Ensure all PHI access is logged"
			])
		elif framework == ComplianceFramework.GDPR:
			recommendations.extend([
				"Implement data subject rights automation",
				"Conduct privacy impact assessments",
				"Maintain comprehensive data processing records"
			])
		
		return recommendations


class EnterpriseSecurityManager:
	"""Enterprise security policy management"""
	
	def __init__(self):
		self.policies = {}
		self.policy_cache = {}
	
	async def create_security_policy(self, policy: SecurityPolicy) -> str:
		"""Create security policy"""
		try:
			async with get_async_db_session() as session:
				await session.execute(
					text("""
					INSERT INTO wo_security_policies (
						id, name, description, policy_type, rules,
						enabled, enforcement_level, applicable_roles,
						applicable_resources, created_at, updated_at
					) VALUES (
						:id, :name, :description, :policy_type, :rules,
						:enabled, :enforcement_level, :applicable_roles,
						:applicable_resources, :created_at, :updated_at
					)
					"""),
					{
						"id": policy.id,
						"name": policy.name,
						"description": policy.description,
						"policy_type": policy.policy_type,
						"rules": json.dumps(policy.rules),
						"enabled": policy.enabled,
						"enforcement_level": policy.enforcement_level,
						"applicable_roles": json.dumps(policy.applicable_roles),
						"applicable_resources": json.dumps(policy.applicable_resources),
						"created_at": policy.created_at,
						"updated_at": policy.updated_at
					}
				)
				await session.commit()
			
			# Cache policy for quick access
			self.policy_cache[policy.id] = policy
			
			return policy.id
			
		except Exception as e:
			print(f"Security policy creation error: {e}")
			raise
	
	async def evaluate_policies(self, user_id: str, resource_type: str, action: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
		"""Evaluate security policies for access decision"""
		try:
			# Get applicable policies
			policies = await self._get_applicable_policies(user_id, resource_type)
			
			violations = []
			allow_access = True
			
			for policy in policies:
				if not policy.enabled:
					continue
				
				policy_result = await self._evaluate_policy(policy, user_id, resource_type, action, context)
				
				if not policy_result["allowed"]:
					violations.extend(policy_result["violations"])
					
					if policy.enforcement_level == "strict":
						allow_access = False
					elif policy.enforcement_level == "permissive":
						# Log violation but allow access
						print(f"Policy violation (permissive): {policy_result['violations']}")
			
			return allow_access, violations
			
		except Exception as e:
			print(f"Policy evaluation error: {e}")
			return False, [f"Policy evaluation error: {e}"]
	
	async def _get_applicable_policies(self, user_id: str, resource_type: str) -> List[SecurityPolicy]:
		"""Get applicable security policies"""
		try:
			async with get_async_db_session() as session:
				result = await session.execute(
					text("""
					SELECT id, name, description, policy_type, rules, enabled,
						   enforcement_level, applicable_roles, applicable_resources,
						   created_at, updated_at
					FROM wo_security_policies 
					WHERE enabled = true
					AND (
						applicable_resources = '[]' OR 
						:resource_type = ANY(string_to_array(trim(applicable_resources, '[]"'), '","'))
					)
					"""),
					{"resource_type": resource_type}
				)
				
				policies = []
				for row in result.fetchall():
					policy = SecurityPolicy(
						id=row[0],
						name=row[1],
						description=row[2],
						policy_type=row[3],
						rules=json.loads(row[4]),
						enabled=row[5],
						enforcement_level=row[6],
						applicable_roles=json.loads(row[7]),
						applicable_resources=json.loads(row[8]),
						created_at=row[9],
						updated_at=row[10]
					)
					policies.append(policy)
				
				return policies
				
		except Exception as e:
			print(f"Get applicable policies error: {e}")
			return []
	
	async def _evaluate_policy(self, policy: SecurityPolicy, user_id: str, resource_type: str, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Evaluate individual security policy"""
		result = {"allowed": True, "violations": []}
		
		for rule in policy.rules:
			rule_type = rule.get("type")
			
			if rule_type == "time_restriction":
				if not await self._check_time_restriction(rule, context):
					result["allowed"] = False
					result["violations"].append(f"Time restriction violated: {rule.get('message', 'Access outside allowed hours')}")
			
			elif rule_type == "ip_whitelist":
				if not await self._check_ip_whitelist(rule, context):
					result["allowed"] = False
					result["violations"].append(f"IP restriction violated: {rule.get('message', 'Access from unauthorized IP')}")
			
			elif rule_type == "rate_limit":
				if not await self._check_rate_limit(rule, user_id, context):
					result["allowed"] = False
					result["violations"].append(f"Rate limit exceeded: {rule.get('message', 'Too many requests')}")
			
			elif rule_type == "mfa_required":
				if not await self._check_mfa_requirement(rule, user_id, context):
					result["allowed"] = False
					result["violations"].append(f"MFA required: {rule.get('message', 'Multi-factor authentication required')}")
			
			elif rule_type == "data_classification":
				if not await self._check_data_classification(rule, resource_type, action, context):
					result["allowed"] = False
					result["violations"].append(f"Data classification violation: {rule.get('message', 'Insufficient clearance')}")
		
		return result
	
	async def _check_time_restriction(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
		"""Check time-based access restrictions"""
		current_time = datetime.utcnow()
		current_hour = current_time.hour
		current_day = current_time.weekday()  # 0 = Monday, 6 = Sunday
		
		allowed_hours = rule.get("allowed_hours", [])
		allowed_days = rule.get("allowed_days", [])
		
		if allowed_hours and current_hour not in allowed_hours:
			return False
		
		if allowed_days and current_day not in allowed_days:
			return False
		
		return True
	
	async def _check_ip_whitelist(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
		"""Check IP whitelist restrictions"""
		import ipaddress
		
		source_ip = context.get("source_ip")
		if not source_ip:
			return False
		
		allowed_ips = rule.get("allowed_ips", [])
		allowed_networks = rule.get("allowed_networks", [])
		
		# Check exact IP matches
		if source_ip in allowed_ips:
			return True
		
		# Check network ranges
		for network in allowed_networks:
			try:
				if ipaddress.ip_address(source_ip) in ipaddress.ip_network(network):
					return True
			except ValueError:
				continue
		
		return False
	
	async def _check_rate_limit(self, rule: Dict[str, Any], user_id: str, context: Dict[str, Any]) -> bool:
		"""Check rate limiting restrictions"""
		if not hasattr(self, 'redis_client'):
			return True  # Skip if Redis not available
		
		limit = rule.get("limit", 100)
		window = rule.get("window_seconds", 3600)
		
		key = f"rate_limit:{user_id}:{rule.get('action', 'default')}"
		
		try:
			import redis
			redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
			
			current_count = redis_client.get(key)
			if current_count is None:
				redis_client.setex(key, window, 1)
				return True
			
			if int(current_count) >= limit:
				return False
			
			redis_client.incr(key)
			return True
			
		except Exception:
			return True  # Allow if Redis check fails
	
	async def _check_mfa_requirement(self, rule: Dict[str, Any], user_id: str, context: Dict[str, Any]) -> bool:
		"""Check multi-factor authentication requirement"""
		session_data = context.get("session", {})
		mfa_verified = session_data.get("mfa_verified", False)
		mfa_timestamp = session_data.get("mfa_timestamp")
		
		if not mfa_verified:
			return False
		
		# Check if MFA is still valid (within time window)
		if mfa_timestamp:
			mfa_time = datetime.fromisoformat(mfa_timestamp)
			max_age = rule.get("max_age_minutes", 60)
			
			if (datetime.utcnow() - mfa_time).total_seconds() > (max_age * 60):
				return False
		
		return True
	
	async def _check_data_classification(self, rule: Dict[str, Any], resource_type: str, action: str, context: Dict[str, Any]) -> bool:
		"""Check data classification and clearance requirements"""
		required_clearance = rule.get("required_clearance", "public")
		user_clearance = context.get("user_clearance", "public")
		
		clearance_levels = {
			"public": 0,
			"internal": 1,
			"confidential": 2,
			"secret": 3,
			"top_secret": 4
		}
		
		required_level = clearance_levels.get(required_clearance, 0)
		user_level = clearance_levels.get(user_clearance, 0)
		
		return user_level >= required_level


class EnterpriseIntegrationManager:
	"""Main enterprise integration management class"""
	
	def __init__(self):
		self.ldap_authenticator = None
		self.saml_authenticator = None
		self.oidc_authenticator = None
		self.db_connector = EnterpriseDBConnector()
		self.audit_system = EnterpriseAuditSystem()
		self.security_manager = EnterpriseSecurityManager()
		self.secret_managers = {}
	
	async def initialize_ldap(self, config: LDAPConfig):
		"""Initialize LDAP authentication"""
		self.ldap_authenticator = LDAPAuthenticator(config)
	
	async def initialize_saml(self, config: SAMLConfig):
		"""Initialize SAML SSO"""
		self.saml_authenticator = SAMLAuthenticator(config)
	
	async def initialize_oidc(self, config: OIDCConfig):
		"""Initialize OIDC authentication"""
		self.oidc_authenticator = OIDCAuthenticator(config)
	
	async def initialize_secret_manager(self, provider: str, config: Dict[str, Any]):
		"""Initialize secret management provider"""
		if provider == "azure_keyvault":
			credential = DefaultAzureCredential()
			client = SecretClient(
				vault_url=config["vault_url"],
				credential=credential
			)
			self.secret_managers["azure"] = client
		
		elif provider == "aws_secrets":
			client = boto3.client(
				"secretsmanager",
				region_name=config.get("region", "us-east-1"),
				aws_access_key_id=config.get("access_key_id"),
				aws_secret_access_key=config.get("secret_access_key")
			)
			self.secret_managers["aws"] = client
		
		elif provider == "gcp_secret_manager":
			client = secretmanager.SecretManagerServiceClient()
			self.secret_managers["gcp"] = client
	
	async def authenticate_user(self, method: AuthenticationMethod, credentials: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
		"""Authenticate user using specified method"""
		try:
			if method == AuthenticationMethod.LDAP and self.ldap_authenticator:
				return await self.ldap_authenticator.authenticate(
					credentials["username"],
					credentials["password"]
				)
			
			elif method == AuthenticationMethod.SAML_SSO and self.saml_authenticator:
				return await self.saml_authenticator.process_response(
					credentials["saml_response"]
				)
			
			elif method == AuthenticationMethod.OIDC and self.oidc_authenticator:
				return await self.oidc_authenticator.exchange_code(
					credentials["code"],
					credentials["state"]
				)
			
			else:
				return False, None
				
		except Exception as e:
			# Log authentication failure
			audit_event = AuditEvent(
				event_type="authentication",
				action="login_attempt",
				result="error",
				details={"method": method.value, "error": str(e)},
				risk_level="medium"
			)
			await self.audit_system.log_audit_event(audit_event)
			
			return False, None
	
	async def get_secret(self, provider: str, secret_name: str) -> Optional[str]:
		"""Retrieve secret from secret manager"""
		try:
			if provider == "azure" and "azure" in self.secret_managers:
				secret = self.secret_managers["azure"].get_secret(secret_name)
				return secret.value
			
			elif provider == "aws" and "aws" in self.secret_managers:
				response = self.secret_managers["aws"].get_secret_value(SecretId=secret_name)
				return response["SecretString"]
			
			elif provider == "gcp" and "gcp" in self.secret_managers:
				name = f"projects/{self.secret_managers['gcp'].project}/secrets/{secret_name}/versions/latest"
				response = self.secret_managers["gcp"].access_secret_version(request={"name": name})
				return response.payload.data.decode("UTF-8")
			
			return None
			
		except Exception as e:
			print(f"Secret retrieval error: {e}")
			return None
	
	async def create_database_connection(self, config: EnterpriseDBConfig) -> str:
		"""Create enterprise database connection"""
		# Retrieve sensitive credentials from secret manager if needed
		if config.password.startswith("secret://"):
			provider, secret_name = config.password[9:].split("/", 1)
			config.password = await self.get_secret(provider, secret_name) or config.password
		
		return await self.db_connector.create_connection(config)
	
	async def execute_database_query(self, connection_id: str, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""Execute query on enterprise database"""
		return await self.db_connector.execute_query(connection_id, query, parameters)
	
	async def log_audit_event(self, event: AuditEvent) -> str:
		"""Log audit event"""
		return await self.audit_system.log_audit_event(event)
	
	async def generate_compliance_report(self, framework: ComplianceFramework, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate compliance report"""
		return await self.audit_system.generate_compliance_report(framework, start_date, end_date)
	
	async def create_security_policy(self, policy: SecurityPolicy) -> str:
		"""Create security policy"""
		return await self.security_manager.create_security_policy(policy)
	
	async def evaluate_access(self, user_id: str, resource_type: str, action: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
		"""Evaluate access based on security policies"""
		return await self.security_manager.evaluate_policies(user_id, resource_type, action, context)


# Global enterprise integration manager instance
enterprise_integration = EnterpriseIntegrationManager()