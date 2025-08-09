"""
APG Customer Relationship Management - Comprehensive Audit Logging

Enterprise-grade audit logging system for compliance, security monitoring,
and forensic analysis with real-time alerting and data retention policies.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator
import ipaddress

import asyncpg
import redis.asyncio as redis
from cryptography.fernet import Fernet


logger = logging.getLogger(__name__)


# ================================
# Enums and Constants
# ================================

class AuditEventType(Enum):
	# Data Operations
	CREATE = "create"
	READ = "read"
	UPDATE = "update"
	DELETE = "delete"
	EXPORT = "export"
	IMPORT = "import"
	BULK_OPERATION = "bulk_operation"
	
	# Authentication & Authorization
	LOGIN = "login"
	LOGOUT = "logout"
	AUTH_FAILURE = "auth_failure"
	PERMISSION_GRANT = "permission_grant"
	PERMISSION_REVOKE = "permission_revoke"
	ROLE_ASSIGNMENT = "role_assignment"
	ROLE_REMOVAL = "role_removal"
	
	# System Events
	SYSTEM_START = "system_start"
	SYSTEM_SHUTDOWN = "system_shutdown"
	CONFIG_CHANGE = "config_change"
	BACKUP_CREATE = "backup_create"
	BACKUP_RESTORE = "backup_restore"
	
	# Security Events
	SUSPICIOUS_ACTIVITY = "suspicious_activity"
	BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
	UNAUTHORIZED_ACCESS = "unauthorized_access"
	DATA_BREACH = "data_breach"
	PRIVILEGE_ESCALATION = "privilege_escalation"
	
	# Compliance Events
	DATA_ACCESS = "data_access"
	DATA_MODIFICATION = "data_modification"
	DATA_RETENTION = "data_retention"
	PRIVACY_REQUEST = "privacy_request"
	CONSENT_CHANGE = "consent_change"
	
	# API Events
	API_CALL = "api_call"
	API_ERROR = "api_error"
	RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
	INTEGRATION_EVENT = "integration_event"


class AuditSeverity(Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class DataClassification(Enum):
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"


class ComplianceFramework(Enum):
	GDPR = "gdpr"
	CCPA = "ccpa"
	HIPAA = "hipaa"
	SOX = "sox"
	PCI_DSS = "pci_dss"
	ISO27001 = "iso27001"
	SOC2 = "soc2"


class AuditStatus(Enum):
	ACTIVE = "active"
	ARCHIVED = "archived"
	DELETED = "deleted"
	QUARANTINED = "quarantined"


# ================================
# Pydantic Models
# ================================

class AuditEntry(BaseModel):
	"""Comprehensive audit entry model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: AuditEventType
	severity: AuditSeverity = AuditSeverity.MEDIUM
	
	# Event Details
	event_name: str
	event_description: Optional[str] = None
	event_category: str
	event_subcategory: Optional[str] = None
	
	# Actor Information
	user_id: Optional[str] = None
	username: Optional[str] = None
	user_role: Optional[str] = None
	session_id: Optional[str] = None
	impersonated_by: Optional[str] = None
	
	# Source Information
	source_ip: str
	source_location: Dict[str, Any] = Field(default_factory=dict)
	user_agent: Optional[str] = None
	device_fingerprint: Optional[str] = None
	
	# Target Information
	resource_type: str
	resource_id: Optional[str] = None
	resource_name: Optional[str] = None
	parent_resource_id: Optional[str] = None
	
	# Data Information
	old_values: Dict[str, Any] = Field(default_factory=dict)
	new_values: Dict[str, Any] = Field(default_factory=dict)
	changed_fields: List[str] = Field(default_factory=list)
	data_classification: DataClassification = DataClassification.INTERNAL
	data_size_bytes: int = 0
	
	# Operation Details
	operation_id: Optional[str] = None
	batch_id: Optional[str] = None
	transaction_id: Optional[str] = None
	correlation_id: Optional[str] = None
	
	# Result Information
	success: bool = True
	error_code: Optional[str] = None
	error_message: Optional[str] = None
	http_status_code: Optional[int] = None
	
	# Performance Metrics
	execution_time_ms: Optional[Decimal] = None
	response_size_bytes: Optional[int] = None
	
	# Compliance Information
	compliance_frameworks: List[ComplianceFramework] = Field(default_factory=list)
	legal_hold: bool = False
	retention_period_days: int = 2555  # 7 years default
	
	# Security Context
	risk_score: Decimal = Decimal('0.0')
	threat_indicators: List[str] = Field(default_factory=list)
	security_context: Dict[str, Any] = Field(default_factory=dict)
	
	# Timestamps
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	event_start_time: Optional[datetime] = None
	event_end_time: Optional[datetime] = None
	
	# System Information
	application_name: str = "APG_CRM"
	application_version: str = "1.0.0"
	environment: str = "production"
	
	# Audit Trail
	chain_hash: Optional[str] = None
	previous_hash: Optional[str] = None
	digital_signature: Optional[str] = None
	
	# Status and Lifecycle
	status: AuditStatus = AuditStatus.ACTIVE
	archived_at: Optional[datetime] = None
	
	# Additional Context
	business_context: Dict[str, Any] = Field(default_factory=dict)
	technical_context: Dict[str, Any] = Field(default_factory=dict)
	tags: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditQuery(BaseModel):
	"""Audit query model for searching audit logs"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	tenant_id: str
	start_date: Optional[datetime] = None
	end_date: Optional[datetime] = None
	event_types: Optional[List[AuditEventType]] = None
	severity: Optional[List[AuditSeverity]] = None
	user_ids: Optional[List[str]] = None
	resource_types: Optional[List[str]] = None
	resource_ids: Optional[List[str]] = None
	success: Optional[bool] = None
	compliance_frameworks: Optional[List[ComplianceFramework]] = None
	ip_addresses: Optional[List[str]] = None
	risk_score_min: Optional[Decimal] = None
	risk_score_max: Optional[Decimal] = None
	tags: Optional[List[str]] = None
	text_search: Optional[str] = None
	limit: int = 100
	offset: int = 0
	sort_by: str = "timestamp"
	sort_order: str = "desc"


class AuditReport(BaseModel):
	"""Audit report model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	report_name: str
	report_type: str
	query_parameters: Dict[str, Any]
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	generated_by: str
	total_entries: int
	entries: List[AuditEntry]
	summary_statistics: Dict[str, Any] = Field(default_factory=dict)
	compliance_status: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditAlert(BaseModel):
	"""Audit alert model for security and compliance monitoring"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	alert_name: str
	alert_type: str
	severity: AuditSeverity
	description: str
	conditions: Dict[str, Any]
	triggered_by: str  # audit entry ID
	triggered_at: datetime = Field(default_factory=datetime.utcnow)
	acknowledged: bool = False
	acknowledged_by: Optional[str] = None
	acknowledged_at: Optional[datetime] = None
	resolved: bool = False
	resolved_by: Optional[str] = None
	resolved_at: Optional[datetime] = None
	actions_taken: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)


# ================================
# Comprehensive Audit Logger
# ================================

class ComprehensiveAuditLogger:
	"""Enterprise-grade audit logging system"""
	
	def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis = None):
		self.db_pool = db_pool
		self.redis_client = redis_client
		self.encryption_key = self._generate_encryption_key()
		self.fernet = Fernet(self.encryption_key)
		self._initialized = False
		self._alert_rules = {}
		self._retention_policies = {}
	
	def _generate_encryption_key(self) -> bytes:
		"""Generate encryption key for sensitive audit data"""
		# In production, this should be managed by a key management service
		return Fernet.generate_key()
	
	async def initialize(self):
		"""Initialize the audit logging system"""
		try:
			if self._initialized:
				return
			
			logger.info("📋 Initializing Comprehensive Audit Logger...")
			
			# Validate database connection
			async with self.db_pool.acquire() as conn:
				await conn.execute("SELECT 1")
			
			# Initialize Redis if available
			if self.redis_client:
				await self.redis_client.ping()
				logger.info("📦 Redis connection established for audit caching")
			
			# Load alert rules and retention policies
			await self._load_alert_rules()
			await self._load_retention_policies()
			
			# Start background tasks
			asyncio.create_task(self._background_archival_task())
			asyncio.create_task(self._background_cleanup_task())
			
			self._initialized = True
			logger.info("✅ Comprehensive Audit Logger initialized successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to initialize audit logger: {str(e)}")
			raise
	
	async def _load_alert_rules(self):
		"""Load audit alert rules from database"""
		try:
			async with self.db_pool.acquire() as conn:
				rules = await conn.fetch("""
					SELECT * FROM crm_audit_alert_rules 
					WHERE is_active = true
				""")
				
				for rule in rules:
					self._alert_rules[rule['id']] = {
						'name': rule['rule_name'],
						'conditions': json.loads(rule['conditions']),
						'severity': rule['severity'],
						'actions': json.loads(rule['actions'])
					}
				
				logger.info(f"📋 Loaded {len(self._alert_rules)} audit alert rules")
				
		except Exception as e:
			logger.error(f"Error loading alert rules: {str(e)}")
	
	async def _load_retention_policies(self):
		"""Load data retention policies from database"""
		try:
			async with self.db_pool.acquire() as conn:
				policies = await conn.fetch("""
					SELECT * FROM crm_audit_retention_policies 
					WHERE is_active = true
				""")
				
				for policy in policies:
					self._retention_policies[policy['resource_type']] = {
						'retention_days': policy['retention_days'],
						'archive_after_days': policy['archive_after_days'],
						'compliance_frameworks': json.loads(policy['compliance_frameworks'])
					}
				
				logger.info(f"📋 Loaded {len(self._retention_policies)} retention policies")
				
		except Exception as e:
			logger.error(f"Error loading retention policies: {str(e)}")
	
	def _calculate_hash(self, data: str) -> str:
		"""Calculate SHA-256 hash for audit trail integrity"""
		return hashlib.sha256(data.encode()).hexdigest()
	
	def _build_audit_chain(self, entry: AuditEntry, previous_hash: str = None) -> str:
		"""Build cryptographic audit chain"""
		chain_data = f"{entry.id}:{entry.timestamp.isoformat()}:{entry.event_type.value}:{previous_hash or ''}"
		return self._calculate_hash(chain_data)
	
	async def _get_last_audit_hash(self, tenant_id: str) -> Optional[str]:
		"""Get the hash of the last audit entry for chain integrity"""
		try:
			async with self.db_pool.acquire() as conn:
				result = await conn.fetchrow("""
					SELECT chain_hash FROM crm_audit_entries 
					WHERE tenant_id = $1 
					ORDER BY timestamp DESC 
					LIMIT 1
				""", tenant_id)
				
				return result['chain_hash'] if result else None
				
		except Exception as e:
			logger.error(f"Error getting last audit hash: {str(e)}")
			return None
	
	async def log_audit_event(
		self,
		tenant_id: str,
		event_type: AuditEventType,
		event_name: str,
		resource_type: str,
		user_id: str = None,
		resource_id: str = None,
		old_values: Dict[str, Any] = None,
		new_values: Dict[str, Any] = None,
		source_ip: str = "system",
		user_agent: str = None,
		severity: AuditSeverity = AuditSeverity.MEDIUM,
		**kwargs
	) -> AuditEntry:
		"""Log a comprehensive audit event"""
		try:
			# Get previous hash for chain integrity
			previous_hash = await self._get_last_audit_hash(tenant_id)
			
			# Create audit entry
			entry_data = {
				'tenant_id': tenant_id,
				'event_type': event_type,
				'event_name': event_name,
				'resource_type': resource_type,
				'user_id': user_id,
				'resource_id': resource_id,
				'old_values': old_values or {},
				'new_values': new_values or {},
				'source_ip': source_ip,
				'user_agent': user_agent,
				'severity': severity,
				'previous_hash': previous_hash,
				**kwargs
			}
			
			# Calculate changed fields
			if old_values and new_values:
				entry_data['changed_fields'] = [
					field for field in new_values.keys()
					if old_values.get(field) != new_values.get(field)
				]
			
			# Calculate data size
			entry_data['data_size_bytes'] = len(json.dumps(new_values or {}))
			
			# Apply retention policy
			retention_policy = self._retention_policies.get(
				resource_type, {'retention_days': 2555}
			)
			entry_data['retention_period_days'] = retention_policy['retention_days']
			
			# Determine compliance frameworks
			if 'compliance_frameworks' not in entry_data:
				entry_data['compliance_frameworks'] = retention_policy.get(
					'compliance_frameworks', []
				)
			
			audit_entry = AuditEntry(**entry_data)
			
			# Build audit chain
			audit_entry.chain_hash = self._build_audit_chain(audit_entry, previous_hash)
			
			# Calculate risk score
			audit_entry.risk_score = self._calculate_risk_score(audit_entry)
			
			# Store in database
			await self._store_audit_entry(audit_entry)
			
			# Cache recent entries in Redis
			if self.redis_client:
				await self._cache_audit_entry(audit_entry)
			
			# Check for alert conditions
			await self._check_alert_conditions(audit_entry)
			
			logger.debug(f"📋 Audit event logged: {event_name} ({event_type.value})")
			return audit_entry
			
		except Exception as e:
			logger.error(f"Failed to log audit event: {str(e)}")
			raise
	
	def _calculate_risk_score(self, entry: AuditEntry) -> Decimal:
		"""Calculate risk score for audit entry"""
		risk_score = Decimal('0.0')
		
		# Base risk by event type
		risk_by_type = {
			AuditEventType.DELETE: Decimal('20.0'),
			AuditEventType.EXPORT: Decimal('15.0'),
			AuditEventType.BULK_OPERATION: Decimal('25.0'),
			AuditEventType.AUTH_FAILURE: Decimal('30.0'),
			AuditEventType.SUSPICIOUS_ACTIVITY: Decimal('80.0'),
			AuditEventType.UNAUTHORIZED_ACCESS: Decimal('90.0'),
			AuditEventType.DATA_BREACH: Decimal('100.0'),
			AuditEventType.PRIVILEGE_ESCALATION: Decimal('95.0'),
		}
		
		risk_score += risk_by_type.get(entry.event_type, Decimal('5.0'))
		
		# Risk by data classification
		classification_risk = {
			DataClassification.PUBLIC: Decimal('0.0'),
			DataClassification.INTERNAL: Decimal('5.0'),
			DataClassification.CONFIDENTIAL: Decimal('15.0'),
			DataClassification.RESTRICTED: Decimal('25.0'),
			DataClassification.TOP_SECRET: Decimal('40.0'),
		}
		
		risk_score += classification_risk.get(entry.data_classification, Decimal('10.0'))
		
		# Risk by failure
		if not entry.success:
			risk_score += Decimal('20.0')
		
		# Risk by unusual IP
		if self._is_unusual_ip(entry.source_ip):
			risk_score += Decimal('15.0')
		
		# Risk by off-hours access
		if self._is_off_hours(entry.timestamp):
			risk_score += Decimal('10.0')
		
		return min(risk_score, Decimal('100.0'))
	
	def _is_unusual_ip(self, ip_address: str) -> bool:
		"""Check if IP address is unusual (simplified implementation)"""
		try:
			ip = ipaddress.ip_address(ip_address)
			# Consider private IPs as less risky
			return not ip.is_private
		except Exception:
			return True
	
	def _is_off_hours(self, timestamp: datetime) -> bool:
		"""Check if timestamp is during off-hours"""
		hour = timestamp.hour
		# Consider 6 PM to 6 AM as off-hours
		return hour >= 18 or hour <= 6
	
	async def _store_audit_entry(self, entry: AuditEntry):
		"""Store audit entry in database"""
		try:
			# Encrypt sensitive data
			old_values_encrypted = self.fernet.encrypt(
				json.dumps(entry.old_values).encode()
			).decode() if entry.old_values else None
			
			new_values_encrypted = self.fernet.encrypt(
				json.dumps(entry.new_values).encode()
			).decode() if entry.new_values else None
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_audit_entries (
						id, tenant_id, event_type, severity, event_name, event_description,
						event_category, event_subcategory, user_id, username, user_role,
						session_id, source_ip, source_location, user_agent, device_fingerprint,
						resource_type, resource_id, resource_name, parent_resource_id,
						old_values, new_values, changed_fields, data_classification,
						data_size_bytes, operation_id, batch_id, transaction_id,
						correlation_id, success, error_code, error_message,
						http_status_code, execution_time_ms, response_size_bytes,
						compliance_frameworks, legal_hold, retention_period_days,
						risk_score, threat_indicators, security_context, timestamp,
						event_start_time, event_end_time, application_name,
						application_version, environment, chain_hash, previous_hash,
						status, business_context, technical_context, tags, metadata
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
						$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
						$29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41,
						$42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55
					)
				""",
				entry.id, entry.tenant_id, entry.event_type.value, entry.severity.value,
				entry.event_name, entry.event_description, entry.event_category,
				entry.event_subcategory, entry.user_id, entry.username, entry.user_role,
				entry.session_id, entry.source_ip, json.dumps(entry.source_location),
				entry.user_agent, entry.device_fingerprint, entry.resource_type,
				entry.resource_id, entry.resource_name, entry.parent_resource_id,
				old_values_encrypted, new_values_encrypted, json.dumps(entry.changed_fields),
				entry.data_classification.value, entry.data_size_bytes, entry.operation_id,
				entry.batch_id, entry.transaction_id, entry.correlation_id, entry.success,
				entry.error_code, entry.error_message, entry.http_status_code,
				entry.execution_time_ms, entry.response_size_bytes,
				json.dumps([f.value for f in entry.compliance_frameworks]), entry.legal_hold,
				entry.retention_period_days, entry.risk_score, json.dumps(entry.threat_indicators),
				json.dumps(entry.security_context), entry.timestamp, entry.event_start_time,
				entry.event_end_time, entry.application_name, entry.application_version,
				entry.environment, entry.chain_hash, entry.previous_hash, entry.status.value,
				json.dumps(entry.business_context), json.dumps(entry.technical_context),
				json.dumps(entry.tags), json.dumps(entry.metadata))
			
		except Exception as e:
			logger.error(f"Error storing audit entry: {str(e)}")
			raise
	
	async def _cache_audit_entry(self, entry: AuditEntry):
		"""Cache recent audit entry in Redis"""
		try:
			if not self.redis_client:
				return
			
			# Cache entry for quick access
			cache_key = f"audit:recent:{entry.tenant_id}:{entry.id}"
			cache_data = {
				'id': entry.id,
				'event_type': entry.event_type.value,
				'event_name': entry.event_name,
				'user_id': entry.user_id,
				'resource_type': entry.resource_type,
				'resource_id': entry.resource_id,
				'timestamp': entry.timestamp.isoformat(),
				'severity': entry.severity.value,
				'success': entry.success
			}
			
			await self.redis_client.setex(
				cache_key,
				timedelta(hours=24),
				json.dumps(cache_data)
			)
			
			# Add to recent events list
			recent_key = f"audit:recent_list:{entry.tenant_id}"
			await self.redis_client.lpush(recent_key, entry.id)
			await self.redis_client.ltrim(recent_key, 0, 999)  # Keep last 1000
			await self.redis_client.expire(recent_key, timedelta(hours=24))
			
		except Exception as e:
			logger.error(f"Error caching audit entry: {str(e)}")
	
	async def _check_alert_conditions(self, entry: AuditEntry):
		"""Check if audit entry triggers any alerts"""
		try:
			for rule_id, rule in self._alert_rules.items():
				if self._evaluate_alert_condition(entry, rule['conditions']):
					await self._trigger_alert(entry, rule)
					
		except Exception as e:
			logger.error(f"Error checking alert conditions: {str(e)}")
	
	def _evaluate_alert_condition(self, entry: AuditEntry, conditions: Dict[str, Any]) -> bool:
		"""Evaluate if entry matches alert conditions"""
		try:
			# Check event type
			if 'event_types' in conditions:
				if entry.event_type.value not in conditions['event_types']:
					return False
			
			# Check severity
			if 'min_severity' in conditions:
				min_severity = AuditSeverity(conditions['min_severity'])
				severity_order = [s.value for s in AuditSeverity]
				if severity_order.index(entry.severity.value) < severity_order.index(min_severity.value):
					return False
			
			# Check risk score
			if 'min_risk_score' in conditions:
				if entry.risk_score < Decimal(str(conditions['min_risk_score'])):
					return False
			
			# Check failure condition
			if 'only_failures' in conditions and conditions['only_failures']:
				if entry.success:
					return False
			
			# Check resource type
			if 'resource_types' in conditions:
				if entry.resource_type not in conditions['resource_types']:
					return False
			
			return True
			
		except Exception as e:
			logger.error(f"Error evaluating alert condition: {str(e)}")
			return False
	
	async def _trigger_alert(self, entry: AuditEntry, rule: Dict[str, Any]):
		"""Trigger audit alert"""
		try:
			alert = AuditAlert(
				tenant_id=entry.tenant_id,
				alert_name=rule['name'],
				alert_type="audit_rule",
				severity=AuditSeverity(rule['severity']),
				description=f"Alert triggered by audit event: {entry.event_name}",
				conditions=rule,
				triggered_by=entry.id
			)
			
			# Store alert
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_audit_alerts (
						id, tenant_id, alert_name, alert_type, severity,
						description, conditions, triggered_by, triggered_at,
						acknowledged, resolved, actions_taken, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
				""",
				alert.id, alert.tenant_id, alert.alert_name, alert.alert_type,
				alert.severity.value, alert.description, json.dumps(alert.conditions),
				alert.triggered_by, alert.triggered_at, alert.acknowledged,
				alert.resolved, json.dumps(alert.actions_taken), json.dumps(alert.metadata))
			
			# Execute alert actions
			await self._execute_alert_actions(alert, rule.get('actions', []))
			
			logger.warning(f"🚨 Audit alert triggered: {alert.alert_name}")
			
		except Exception as e:
			logger.error(f"Error triggering alert: {str(e)}")
	
	async def _execute_alert_actions(self, alert: AuditAlert, actions: List[str]):
		"""Execute alert actions (notifications, etc.)"""
		try:
			for action in actions:
				if action == "log_warning":
					logger.warning(f"🚨 Security Alert: {alert.description}")
				elif action == "notify_admin":
					# Implement admin notification
					logger.info(f"📧 Admin notification sent for alert: {alert.alert_name}")
				elif action == "block_ip":
					# Implement IP blocking
					logger.info(f"🚫 IP blocking requested for alert: {alert.alert_name}")
				elif action == "lock_account":
					# Implement account locking
					logger.info(f"🔒 Account locking requested for alert: {alert.alert_name}")
				
		except Exception as e:
			logger.error(f"Error executing alert actions: {str(e)}")
	
	async def query_audit_logs(self, query: AuditQuery) -> List[AuditEntry]:
		"""Query audit logs with filters"""
		try:
			# Build dynamic query
			where_conditions = ["tenant_id = $1"]
			params = [query.tenant_id]
			param_count = 1
			
			if query.start_date:
				param_count += 1
				where_conditions.append(f"timestamp >= ${param_count}")
				params.append(query.start_date)
			
			if query.end_date:
				param_count += 1
				where_conditions.append(f"timestamp <= ${param_count}")
				params.append(query.end_date)
			
			if query.event_types:
				param_count += 1
				where_conditions.append(f"event_type = ANY(${param_count})")
				params.append([et.value for et in query.event_types])
			
			if query.severity:
				param_count += 1
				where_conditions.append(f"severity = ANY(${param_count})")
				params.append([s.value for s in query.severity])
			
			if query.user_ids:
				param_count += 1
				where_conditions.append(f"user_id = ANY(${param_count})")
				params.append(query.user_ids)
			
			if query.resource_types:
				param_count += 1
				where_conditions.append(f"resource_type = ANY(${param_count})")
				params.append(query.resource_types)
			
			if query.resource_ids:
				param_count += 1
				where_conditions.append(f"resource_id = ANY(${param_count})")
				params.append(query.resource_ids)
			
			if query.success is not None:
				param_count += 1
				where_conditions.append(f"success = ${param_count}")
				params.append(query.success)
			
			if query.risk_score_min:
				param_count += 1
				where_conditions.append(f"risk_score >= ${param_count}")
				params.append(query.risk_score_min)
			
			if query.risk_score_max:
				param_count += 1
				where_conditions.append(f"risk_score <= ${param_count}")
				params.append(query.risk_score_max)
			
			if query.text_search:
				param_count += 1
				where_conditions.append(f"(event_name ILIKE ${param_count} OR event_description ILIKE ${param_count})")
				search_term = f"%{query.text_search}%"
				params.extend([search_term, search_term])
			
			# Build final query
			where_clause = " AND ".join(where_conditions)
			order_clause = f"ORDER BY {query.sort_by} {query.sort_order.upper()}"
			limit_clause = f"LIMIT {query.limit} OFFSET {query.offset}"
			
			sql = f"""
				SELECT * FROM crm_audit_entries 
				WHERE {where_clause} 
				{order_clause} 
				{limit_clause}
			"""
			
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch(sql, *params)
				
				entries = []
				for row in rows:
					# Decrypt sensitive data
					old_values = {}
					new_values = {}
					
					if row['old_values']:
						old_values = json.loads(
							self.fernet.decrypt(row['old_values'].encode()).decode()
						)
					
					if row['new_values']:
						new_values = json.loads(
							self.fernet.decrypt(row['new_values'].encode()).decode()
						)
					
					entry_data = dict(row)
					entry_data['old_values'] = old_values
					entry_data['new_values'] = new_values
					entry_data['event_type'] = AuditEventType(row['event_type'])
					entry_data['severity'] = AuditSeverity(row['severity'])
					entry_data['data_classification'] = DataClassification(row['data_classification'])
					entry_data['status'] = AuditStatus(row['status'])
					entry_data['compliance_frameworks'] = [
						ComplianceFramework(f) for f in json.loads(row['compliance_frameworks'] or '[]')
					]
					entry_data['changed_fields'] = json.loads(row['changed_fields'] or '[]')
					entry_data['source_location'] = json.loads(row['source_location'] or '{}')
					entry_data['threat_indicators'] = json.loads(row['threat_indicators'] or '[]')
					entry_data['security_context'] = json.loads(row['security_context'] or '{}')
					entry_data['business_context'] = json.loads(row['business_context'] or '{}')
					entry_data['technical_context'] = json.loads(row['technical_context'] or '{}')
					entry_data['tags'] = json.loads(row['tags'] or '[]')
					entry_data['metadata'] = json.loads(row['metadata'] or '{}')
					
					entries.append(AuditEntry(**entry_data))
				
				return entries
				
		except Exception as e:
			logger.error(f"Error querying audit logs: {str(e)}")
			raise
	
	async def generate_audit_report(
		self,
		tenant_id: str,
		report_name: str,
		report_type: str,
		query: AuditQuery,
		generated_by: str
	) -> AuditReport:
		"""Generate comprehensive audit report"""
		try:
			# Query audit entries
			entries = await self.query_audit_logs(query)
			
			# Generate summary statistics
			summary_stats = self._generate_summary_statistics(entries)
			
			# Generate compliance status
			compliance_status = self._generate_compliance_status(entries)
			
			report = AuditReport(
				tenant_id=tenant_id,
				report_name=report_name,
				report_type=report_type,
				query_parameters=query.model_dump(),
				generated_by=generated_by,
				total_entries=len(entries),
				entries=entries,
				summary_statistics=summary_stats,
				compliance_status=compliance_status
			)
			
			# Store report
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_audit_reports (
						id, tenant_id, report_name, report_type, query_parameters,
						generated_at, generated_by, total_entries, summary_statistics,
						compliance_status, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
				""",
				report.id, report.tenant_id, report.report_name, report.report_type,
				json.dumps(report.query_parameters), report.generated_at,
				report.generated_by, report.total_entries,
				json.dumps(report.summary_statistics), json.dumps(report.compliance_status),
				json.dumps(report.metadata))
			
			logger.info(f"📊 Audit report generated: {report_name}")
			return report
			
		except Exception as e:
			logger.error(f"Error generating audit report: {str(e)}")
			raise
	
	def _generate_summary_statistics(self, entries: List[AuditEntry]) -> Dict[str, Any]:
		"""Generate summary statistics for audit entries"""
		if not entries:
			return {}
		
		stats = {
			'total_entries': len(entries),
			'event_types': {},
			'severity_distribution': {},
			'success_rate': 0,
			'average_risk_score': 0,
			'top_users': {},
			'top_resources': {},
			'time_range': {
				'start': min(entry.timestamp for entry in entries).isoformat(),
				'end': max(entry.timestamp for entry in entries).isoformat()
			}
		}
		
		# Event type distribution
		for entry in entries:
			event_type = entry.event_type.value
			stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
		
		# Severity distribution
		for entry in entries:
			severity = entry.severity.value
			stats['severity_distribution'][severity] = stats['severity_distribution'].get(severity, 0) + 1
		
		# Success rate
		successful_entries = sum(1 for entry in entries if entry.success)
		stats['success_rate'] = (successful_entries / len(entries)) * 100
		
		# Average risk score
		total_risk = sum(entry.risk_score for entry in entries)
		stats['average_risk_score'] = float(total_risk / len(entries))
		
		# Top users
		user_counts = {}
		for entry in entries:
			if entry.user_id:
				user_counts[entry.user_id] = user_counts.get(entry.user_id, 0) + 1
		
		stats['top_users'] = dict(sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10])
		
		# Top resources
		resource_counts = {}
		for entry in entries:
			resource_key = f"{entry.resource_type}:{entry.resource_id or 'unknown'}"
			resource_counts[resource_key] = resource_counts.get(resource_key, 0) + 1
		
		stats['top_resources'] = dict(sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10])
		
		return stats
	
	def _generate_compliance_status(self, entries: List[AuditEntry]) -> Dict[str, Any]:
		"""Generate compliance status summary"""
		compliance_status = {
			'frameworks': {},
			'data_classifications': {},
			'retention_compliance': {},
			'security_incidents': 0,
			'privacy_events': 0
		}
		
		# Framework coverage
		for entry in entries:
			for framework in entry.compliance_frameworks:
				framework_name = framework.value
				if framework_name not in compliance_status['frameworks']:
					compliance_status['frameworks'][framework_name] = {
						'total_events': 0,
						'security_events': 0,
						'compliance_score': 100
					}
				
				compliance_status['frameworks'][framework_name]['total_events'] += 1
				
				if entry.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
					compliance_status['frameworks'][framework_name]['security_events'] += 1
		
		# Data classification distribution
		for entry in entries:
			classification = entry.data_classification.value
			compliance_status['data_classifications'][classification] = \
				compliance_status['data_classifications'].get(classification, 0) + 1
		
		# Security incidents
		compliance_status['security_incidents'] = sum(
			1 for entry in entries
			if entry.event_type in [
				AuditEventType.SUSPICIOUS_ACTIVITY,
				AuditEventType.UNAUTHORIZED_ACCESS,
				AuditEventType.DATA_BREACH
			]
		)
		
		# Privacy events
		compliance_status['privacy_events'] = sum(
			1 for entry in entries
			if entry.event_type in [
				AuditEventType.DATA_ACCESS,
				AuditEventType.PRIVACY_REQUEST,
				AuditEventType.CONSENT_CHANGE
			]
		)
		
		return compliance_status
	
	async def _background_archival_task(self):
		"""Background task to archive old audit entries"""
		while True:
			try:
				await asyncio.sleep(3600)  # Run every hour
				
				# Archive entries older than 90 days
				cutoff_date = datetime.utcnow() - timedelta(days=90)
				
				async with self.db_pool.acquire() as conn:
					result = await conn.execute("""
						UPDATE crm_audit_entries 
						SET status = 'archived', archived_at = NOW()
						WHERE timestamp < $1 AND status = 'active'
					""", cutoff_date)
					
					if result != "UPDATE 0":
						logger.info(f"📦 Archived old audit entries: {result}")
				
			except Exception as e:
				logger.error(f"Error in archival task: {str(e)}")
	
	async def _background_cleanup_task(self):
		"""Background task to clean up expired audit entries"""
		while True:
			try:
				await asyncio.sleep(86400)  # Run daily
				
				# Delete entries past their retention period
				async with self.db_pool.acquire() as conn:
					result = await conn.execute("""
						DELETE FROM crm_audit_entries 
						WHERE timestamp < NOW() - INTERVAL '1 day' * retention_period_days
						AND status = 'archived'
						AND legal_hold = false
					""")
					
					if result != "DELETE 0":
						logger.info(f"🗑️ Cleaned up expired audit entries: {result}")
				
			except Exception as e:
				logger.error(f"Error in cleanup task: {str(e)}")
	
	async def verify_audit_integrity(self, tenant_id: str, start_date: datetime = None) -> Dict[str, Any]:
		"""Verify audit log integrity using cryptographic chain"""
		try:
			query = AuditQuery(
				tenant_id=tenant_id,
				start_date=start_date,
				sort_by="timestamp",
				sort_order="asc",
				limit=10000
			)
			
			entries = await self.query_audit_logs(query)
			
			integrity_results = {
				'verified': True,
				'total_entries': len(entries),
				'chain_breaks': [],
				'hash_mismatches': [],
				'verification_timestamp': datetime.utcnow().isoformat()
			}
			
			previous_hash = None
			for i, entry in enumerate(entries):
				# Verify chain hash
				expected_hash = self._build_audit_chain(entry, previous_hash)
				
				if entry.chain_hash != expected_hash:
					integrity_results['verified'] = False
					integrity_results['hash_mismatches'].append({
						'entry_id': entry.id,
						'expected_hash': expected_hash,
						'actual_hash': entry.chain_hash,
						'timestamp': entry.timestamp.isoformat()
					})
				
				# Verify previous hash chain
				if i > 0 and entry.previous_hash != entries[i-1].chain_hash:
					integrity_results['verified'] = False
					integrity_results['chain_breaks'].append({
						'entry_id': entry.id,
						'expected_previous': entries[i-1].chain_hash,
						'actual_previous': entry.previous_hash,
						'timestamp': entry.timestamp.isoformat()
					})
				
				previous_hash = entry.chain_hash
			
			logger.info(f"🔐 Audit integrity verification: {'PASSED' if integrity_results['verified'] else 'FAILED'}")
			return integrity_results
			
		except Exception as e:
			logger.error(f"Error verifying audit integrity: {str(e)}")
			raise
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform health check"""
		try:
			health_status = {
				'status': 'healthy',
				'timestamp': datetime.utcnow().isoformat(),
				'components': {}
			}
			
			# Check database connection
			try:
				async with self.db_pool.acquire() as conn:
					result = await conn.fetchrow("SELECT COUNT(*) as count FROM crm_audit_entries LIMIT 1")
				health_status['components']['database'] = 'healthy'
				health_status['components']['total_audit_entries'] = result['count']
			except Exception as e:
				health_status['components']['database'] = f'unhealthy: {str(e)}'
				health_status['status'] = 'degraded'
			
			# Check Redis connection
			if self.redis_client:
				try:
					await self.redis_client.ping()
					health_status['components']['redis'] = 'healthy'
				except Exception as e:
					health_status['components']['redis'] = f'unhealthy: {str(e)}'
					health_status['status'] = 'degraded'
			
			return health_status
			
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}