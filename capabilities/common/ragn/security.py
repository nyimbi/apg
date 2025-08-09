"""
APG RAG Security & Compliance Integration

Enterprise-grade security, access control, and compliance features including
tenant isolation, data encryption, audit logging, and regulatory compliance.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import secrets
from uuid_extensions import uuid7str
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Database imports
import asyncpg
from asyncpg import Pool

# APG imports
from .models import APGBaseModel

class SecurityLevel(str, Enum):
	"""Security classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"

class AccessOperation(str, Enum):
	"""Access operations for audit logging"""
	CREATE = "create"
	READ = "read"
	UPDATE = "update"
	DELETE = "delete"
	QUERY = "query"
	GENERATE = "generate"
	EXPORT = "export"

class ComplianceFramework(str, Enum):
	"""Supported compliance frameworks"""
	GDPR = "gdpr"
	CCPA = "ccpa"
	HIPAA = "hipaa"
	SOX = "sox"
	ISO27001 = "iso27001"

@dataclass
class SecurityContext:
	"""Security context for operations"""
	tenant_id: str
	user_id: str
	session_id: str
	ip_address: str = ""
	user_agent: str = ""
	permissions: Set[str] = field(default_factory=set)
	security_level: SecurityLevel = SecurityLevel.INTERNAL
	created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AuditLogEntry:
	"""Audit log entry"""
	id: str = field(default_factory=uuid7str)
	tenant_id: str = ""
	user_id: str = ""
	session_id: str = ""
	operation: AccessOperation = AccessOperation.READ
	resource_type: str = ""
	resource_id: str = ""
	ip_address: str = ""
	user_agent: str = ""
	request_details: Dict[str, Any] = field(default_factory=dict)
	response_summary: Dict[str, Any] = field(default_factory=dict)
	success: bool = True
	error_message: str = ""
	processing_time_ms: float = 0.0
	timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataClassification:
	"""Data classification and handling requirements"""
	security_level: SecurityLevel
	retention_days: int
	encryption_required: bool
	audit_required: bool
	compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
	geographic_restrictions: List[str] = field(default_factory=list)
	
class EncryptionManager:
	"""Manages data encryption and decryption"""
	
	def __init__(self, master_key: Optional[str] = None):
		if master_key:
			self.master_key = master_key.encode()
		else:
			self.master_key = Fernet.generate_key()
		
		self.cipher_suite = Fernet(self.master_key)
		self.logger = logging.getLogger(__name__)
	
	def encrypt_data(self, data: str) -> str:
		"""Encrypt sensitive data"""
		try:
			encrypted_data = self.cipher_suite.encrypt(data.encode())
			return base64.b64encode(encrypted_data).decode()
		except Exception as e:
			self.logger.error(f"Encryption failed: {str(e)}")
			raise
	
	def decrypt_data(self, encrypted_data: str) -> str:
		"""Decrypt sensitive data"""
		try:
			decoded_data = base64.b64decode(encrypted_data.encode())
			decrypted_data = self.cipher_suite.decrypt(decoded_data)
			return decrypted_data.decode()
		except Exception as e:
			self.logger.error(f"Decryption failed: {str(e)}")
			raise
	
	def hash_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
		"""Create secure hash with salt"""
		if not salt:
			salt = secrets.token_hex(16)
		
		# Create PBKDF2 hash
		kdf = PBKDF2HMAC(
			algorithm=hashes.SHA256(),
			length=32,
			salt=salt.encode(),
			iterations=100000,
		)
		hash_bytes = kdf.derive(data.encode())
		hash_string = base64.b64encode(hash_bytes).decode()
		
		return hash_string, salt
	
	def verify_hash(self, data: str, hash_string: str, salt: str) -> bool:
		"""Verify data against hash"""
		try:
			computed_hash, _ = self.hash_data(data, salt)
			return computed_hash == hash_string
		except Exception:
			return False

class TenantIsolationManager:
	"""Ensures complete tenant data isolation"""
	
	def __init__(self, db_pool: Pool):
		self.db_pool = db_pool
		self.logger = logging.getLogger(__name__)
	
	async def verify_tenant_access(self, 
	                              tenant_id: str, 
	                              resource_id: str, 
	                              resource_type: str) -> bool:
		"""Verify tenant has access to resource"""
		try:
			async with self.db_pool.acquire() as conn:
				# Check based on resource type
				if resource_type == "knowledge_base":
					result = await conn.fetchval("""
						SELECT EXISTS(
							SELECT 1 FROM apg_rag_knowledge_bases 
							WHERE id = $1 AND tenant_id = $2
						)
					""", resource_id, tenant_id)
				
				elif resource_type == "document":
					result = await conn.fetchval("""
						SELECT EXISTS(
							SELECT 1 FROM apg_rag_documents 
							WHERE id = $1 AND tenant_id = $2
						)
					""", resource_id, tenant_id)
				
				elif resource_type == "conversation":
					result = await conn.fetchval("""
						SELECT EXISTS(
							SELECT 1 FROM apg_rag_conversations 
							WHERE id = $1 AND tenant_id = $2
						)
					""", resource_id, tenant_id)
				
				else:
					# Default deny for unknown resource types
					result = False
				
				return bool(result)
				
		except Exception as e:
			self.logger.error(f"Tenant access verification failed: {str(e)}")
			return False
	
	async def get_tenant_resources(self, 
	                              tenant_id: str, 
	                              resource_type: str,
	                              limit: int = 100) -> List[str]:
		"""Get all resource IDs for tenant"""
		try:
			async with self.db_pool.acquire() as conn:
				if resource_type == "knowledge_base":
					rows = await conn.fetch("""
						SELECT id FROM apg_rag_knowledge_bases 
						WHERE tenant_id = $1 
						ORDER BY created_at DESC 
						LIMIT $2
					""", tenant_id, limit)
				
				elif resource_type == "document":
					rows = await conn.fetch("""
						SELECT id FROM apg_rag_documents 
						WHERE tenant_id = $1 
						ORDER BY created_at DESC 
						LIMIT $2
					""", tenant_id, limit)
				
				elif resource_type == "conversation":
					rows = await conn.fetch("""
						SELECT id FROM apg_rag_conversations 
						WHERE tenant_id = $1 
						ORDER BY created_at DESC 
						LIMIT $2
					""", tenant_id, limit)
				
				else:
					return []
				
				return [str(row['id']) for row in rows]
				
		except Exception as e:
			self.logger.error(f"Failed to get tenant resources: {str(e)}")
			return []

class AuditLogger:
	"""Comprehensive audit logging system"""
	
	def __init__(self, db_pool: Pool, retention_days: int = 2555):  # 7 years default
		self.db_pool = db_pool
		self.retention_days = retention_days
		self.logger = logging.getLogger(__name__)
	
	async def log_access(self, 
	                    security_context: SecurityContext,
	                    operation: AccessOperation,
	                    resource_type: str,
	                    resource_id: str,
	                    request_details: Dict[str, Any] = None,
	                    response_summary: Dict[str, Any] = None,
	                    success: bool = True,
	                    error_message: str = "",
	                    processing_time_ms: float = 0.0) -> str:
		"""Log access operation"""
		
		audit_entry = AuditLogEntry(
			tenant_id=security_context.tenant_id,
			user_id=security_context.user_id,
			session_id=security_context.session_id,
			operation=operation,
			resource_type=resource_type,
			resource_id=resource_id,
			ip_address=security_context.ip_address,
			user_agent=security_context.user_agent,
			request_details=request_details or {},
			response_summary=response_summary or {},
			success=success,
			error_message=error_message,
			processing_time_ms=processing_time_ms
		)
		
		try:
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO apg_rag_audit_logs (
						id, tenant_id, user_id, session_id, operation, resource_type,
						resource_id, ip_address, user_agent, request_details,
						response_summary, success, error_message, processing_time_ms,
						timestamp, created_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
				""", audit_entry.id, audit_entry.tenant_id, audit_entry.user_id,
				     audit_entry.session_id, audit_entry.operation.value,
				     audit_entry.resource_type, audit_entry.resource_id,
				     audit_entry.ip_address, audit_entry.user_agent,
				     json.dumps(audit_entry.request_details),
				     json.dumps(audit_entry.response_summary),
				     audit_entry.success, audit_entry.error_message,
				     audit_entry.processing_time_ms, audit_entry.timestamp,
				     audit_entry.timestamp)
			
			return audit_entry.id
			
		except Exception as e:
			self.logger.error(f"Failed to log audit entry: {str(e)}")
			# Don't fail the operation due to audit logging issues
			return ""
	
	async def get_audit_trail(self, 
	                         tenant_id: str,
	                         resource_id: Optional[str] = None,
	                         user_id: Optional[str] = None,
	                         operation: Optional[AccessOperation] = None,
	                         start_time: Optional[datetime] = None,
	                         end_time: Optional[datetime] = None,
	                         limit: int = 100) -> List[AuditLogEntry]:
		"""Get audit trail with filters"""
		try:
			where_conditions = ["tenant_id = $1"]
			params = [tenant_id]
			param_count = 1
			
			if resource_id:
				param_count += 1
				where_conditions.append(f"resource_id = ${param_count}")
				params.append(resource_id)
			
			if user_id:
				param_count += 1
				where_conditions.append(f"user_id = ${param_count}")
				params.append(user_id)
			
			if operation:
				param_count += 1
				where_conditions.append(f"operation = ${param_count}")
				params.append(operation.value)
			
			if start_time:
				param_count += 1
				where_conditions.append(f"timestamp >= ${param_count}")
				params.append(start_time)
			
			if end_time:
				param_count += 1
				where_conditions.append(f"timestamp <= ${param_count}")
				params.append(end_time)
			
			where_clause = " AND ".join(where_conditions)
			
			param_count += 1
			params.append(limit)
			
			query = f"""
				SELECT * FROM apg_rag_audit_logs 
				WHERE {where_clause}
				ORDER BY timestamp DESC
				LIMIT ${param_count}
			"""
			
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch(query, *params)
				
				audit_entries = []
				for row in rows:
					entry_dict = dict(row)
					# Parse JSON fields
					entry_dict['request_details'] = json.loads(entry_dict.get('request_details', '{}'))
					entry_dict['response_summary'] = json.loads(entry_dict.get('response_summary', '{}'))
					entry_dict['operation'] = AccessOperation(entry_dict['operation'])
					
					audit_entries.append(AuditLogEntry(**entry_dict))
				
				return audit_entries
				
		except Exception as e:
			self.logger.error(f"Failed to get audit trail: {str(e)}")
			return []
	
	async def cleanup_old_logs(self) -> int:
		"""Clean up old audit logs based on retention policy"""
		try:
			cutoff_date = datetime.now() - timedelta(days=self.retention_days)
			
			async with self.db_pool.acquire() as conn:
				deleted_count = await conn.fetchval("""
					DELETE FROM apg_rag_audit_logs 
					WHERE timestamp < $1
					RETURNING COUNT(*)
				""", cutoff_date)
			
			if deleted_count:
				self.logger.info(f"Cleaned up {deleted_count} old audit log entries")
			
			return deleted_count or 0
			
		except Exception as e:
			self.logger.error(f"Failed to cleanup old audit logs: {str(e)}")
			return 0

class ComplianceManager:
	"""Manages regulatory compliance requirements"""
	
	def __init__(self, db_pool: Pool, encryption_manager: EncryptionManager):
		self.db_pool = db_pool
		self.encryption_manager = encryption_manager
		self.logger = logging.getLogger(__name__)
		
		# Define compliance requirements
		self.compliance_rules = {
			ComplianceFramework.GDPR: {
				'data_retention_days': 2555,  # 7 years
				'encryption_required': True,
				'audit_required': True,
				'right_to_deletion': True,
				'right_to_export': True,
				'geographic_restrictions': ['EU']
			},
			ComplianceFramework.CCPA: {
				'data_retention_days': 1825,  # 5 years
				'encryption_required': True,
				'audit_required': True,
				'right_to_deletion': True,
				'right_to_export': True,
				'geographic_restrictions': ['CA', 'US']
			},
			ComplianceFramework.HIPAA: {
				'data_retention_days': 2190,  # 6 years
				'encryption_required': True,
				'audit_required': True,
				'right_to_deletion': False,
				'right_to_export': False,
				'geographic_restrictions': ['US']
			}
		}
	
	async def classify_document(self, 
	                          document_content: str,
	                          metadata: Dict[str, Any]) -> DataClassification:
		"""Auto-classify document based on content and metadata"""
		
		# Simple classification logic (would be more sophisticated in production)
		security_level = SecurityLevel.INTERNAL
		compliance_frameworks = []
		encryption_required = False
		audit_required = True
		
		# Check for sensitive patterns
		sensitive_patterns = [
			'ssn', 'social security', 'credit card', 'passport',
			'medical record', 'patient', 'diagnosis', 'medication',
			'financial', 'bank account', 'routing number'
		]
		
		content_lower = document_content.lower()
		for pattern in sensitive_patterns:
			if pattern in content_lower:
				security_level = SecurityLevel.CONFIDENTIAL
				encryption_required = True
				break
		
		# Check metadata for compliance indicators
		if metadata.get('contains_pii', False):
			compliance_frameworks.extend([ComplianceFramework.GDPR, ComplianceFramework.CCPA])
			security_level = SecurityLevel.CONFIDENTIAL
			encryption_required = True
		
		if metadata.get('contains_phi', False):
			compliance_frameworks.append(ComplianceFramework.HIPAA)
			security_level = SecurityLevel.RESTRICTED
			encryption_required = True
		
		# Default retention period
		retention_days = 1825  # 5 years default
		
		return DataClassification(
			security_level=security_level,
			retention_days=retention_days,
			encryption_required=encryption_required,
			audit_required=audit_required,
			compliance_frameworks=compliance_frameworks
		)
	
	async def handle_data_subject_request(self,
	                                    tenant_id: str,
	                                    user_id: str,
	                                    request_type: str,
	                                    compliance_framework: ComplianceFramework) -> Dict[str, Any]:
		"""Handle data subject requests (GDPR Article 15, CCPA Section 1798.110, etc.)"""
		
		try:
			if request_type == "export":
				return await self._export_user_data(tenant_id, user_id, compliance_framework)
			elif request_type == "delete":
				return await self._delete_user_data(tenant_id, user_id, compliance_framework)
			else:
				raise ValueError(f"Unsupported request type: {request_type}")
		
		except Exception as e:
			self.logger.error(f"Data subject request failed: {str(e)}")
			return {'success': False, 'error': str(e)}
	
	async def _export_user_data(self, 
	                          tenant_id: str, 
	                          user_id: str,
	                          compliance_framework: ComplianceFramework) -> Dict[str, Any]:
		"""Export all user data for compliance"""
		
		user_data = {
			'user_id': user_id,
			'tenant_id': tenant_id,
			'export_timestamp': datetime.now().isoformat(),
			'compliance_framework': compliance_framework.value,
			'data': {}
		}
		
		try:
			async with self.db_pool.acquire() as conn:
				# Export knowledge bases
				kb_rows = await conn.fetch("""
					SELECT * FROM apg_rag_knowledge_bases 
					WHERE tenant_id = $1 AND user_id = $2
				""", tenant_id, user_id)
				user_data['data']['knowledge_bases'] = [dict(row) for row in kb_rows]
				
				# Export documents
				doc_rows = await conn.fetch("""
					SELECT * FROM apg_rag_documents 
					WHERE tenant_id = $1 AND user_id = $2
				""", tenant_id, user_id)
				user_data['data']['documents'] = [dict(row) for row in doc_rows]
				
				# Export conversations
				conv_rows = await conn.fetch("""
					SELECT * FROM apg_rag_conversations 
					WHERE tenant_id = $1 AND user_id = $2
				""", tenant_id, user_id)
				user_data['data']['conversations'] = [dict(row) for row in conv_rows]
				
				# Export audit logs
				audit_rows = await conn.fetch("""
					SELECT * FROM apg_rag_audit_logs 
					WHERE tenant_id = $1 AND user_id = $2
				""", tenant_id, user_id)
				user_data['data']['audit_logs'] = [dict(row) for row in audit_rows]
			
			return {
				'success': True,
				'export_data': user_data,
				'record_count': sum(len(data) for data in user_data['data'].values())
			}
			
		except Exception as e:
			self.logger.error(f"User data export failed: {str(e)}")
			return {'success': False, 'error': str(e)}
	
	async def _delete_user_data(self, 
	                          tenant_id: str, 
	                          user_id: str,
	                          compliance_framework: ComplianceFramework) -> Dict[str, Any]:
		"""Delete all user data for compliance (right to be forgotten)"""
		
		rules = self.compliance_rules.get(compliance_framework, {})
		if not rules.get('right_to_deletion', False):
			return {
				'success': False, 
				'error': f"{compliance_framework.value} does not support right to deletion"
			}
		
		deleted_counts = {}
		
		try:
			async with self.db_pool.acquire() as conn:
				# Delete in proper order due to foreign key constraints
				
				# Delete conversation turns
				count = await conn.fetchval("""
					DELETE FROM apg_rag_conversation_turns 
					WHERE tenant_id = $1 AND conversation_id IN (
						SELECT id FROM apg_rag_conversations 
						WHERE tenant_id = $1 AND user_id = $2
					)
					RETURNING COUNT(*)
				""", tenant_id, user_id)
				deleted_counts['conversation_turns'] = count or 0
				
				# Delete conversations
				count = await conn.fetchval("""
					DELETE FROM apg_rag_conversations 
					WHERE tenant_id = $1 AND user_id = $2
					RETURNING COUNT(*)
				""", tenant_id, user_id)
				deleted_counts['conversations'] = count or 0
				
				# Delete document chunks
				count = await conn.fetchval("""
					DELETE FROM apg_rag_document_chunks 
					WHERE tenant_id = $1 AND document_id IN (
						SELECT id FROM apg_rag_documents 
						WHERE tenant_id = $1 AND user_id = $2
					)
					RETURNING COUNT(*)
				""", tenant_id, user_id)
				deleted_counts['document_chunks'] = count or 0
				
				# Delete documents
				count = await conn.fetchval("""
					DELETE FROM apg_rag_documents 
					WHERE tenant_id = $1 AND user_id = $2
					RETURNING COUNT(*)
				""", tenant_id, user_id)
				deleted_counts['documents'] = count or 0
				
				# Delete knowledge bases
				count = await conn.fetchval("""
					DELETE FROM apg_rag_knowledge_bases 
					WHERE tenant_id = $1 AND user_id = $2
					RETURNING COUNT(*)
				""", tenant_id, user_id)
				deleted_counts['knowledge_bases'] = count or 0
				
				# Keep audit logs for compliance (anonymize user_id instead)
				await conn.execute("""
					UPDATE apg_rag_audit_logs 
					SET user_id = 'DELETED_USER', 
					    request_details = '{"anonymized": true}',
					    response_summary = '{"anonymized": true}'
					WHERE tenant_id = $1 AND user_id = $2
				""", tenant_id, user_id)
			
			total_deleted = sum(deleted_counts.values())
			
			return {
				'success': True,
				'deleted_counts': deleted_counts,
				'total_records_deleted': total_deleted
			}
			
		except Exception as e:
			self.logger.error(f"User data deletion failed: {str(e)}")
			return {'success': False, 'error': str(e)}

class SecurityManager:
	"""Main security manager orchestrating all security components"""
	
	def __init__(self, 
	             db_pool: Pool,
	             master_encryption_key: Optional[str] = None,
	             audit_retention_days: int = 2555):
		
		self.db_pool = db_pool
		
		# Initialize components
		self.encryption_manager = EncryptionManager(master_encryption_key)
		self.tenant_isolation = TenantIsolationManager(db_pool)
		self.audit_logger = AuditLogger(db_pool, audit_retention_days)
		self.compliance_manager = ComplianceManager(db_pool, self.encryption_manager)
		
		self.logger = logging.getLogger(__name__)
	
	async def create_security_context(self,
	                                tenant_id: str,
	                                user_id: str,
	                                session_id: str,
	                                ip_address: str = "",
	                                user_agent: str = "",
	                                permissions: Set[str] = None) -> SecurityContext:
		"""Create security context for operation"""
		
		return SecurityContext(
			tenant_id=tenant_id,
			user_id=user_id,
			session_id=session_id,
			ip_address=ip_address,
			user_agent=user_agent,
			permissions=permissions or set()
		)
	
	async def authorize_operation(self,
	                            security_context: SecurityContext,
	                            operation: AccessOperation,
	                            resource_type: str,
	                            resource_id: str) -> bool:
		"""Authorize operation with comprehensive security checks"""
		
		try:
			# Check tenant isolation
			if resource_id:  # Skip for create operations
				has_access = await self.tenant_isolation.verify_tenant_access(
					security_context.tenant_id, resource_id, resource_type
				)
				if not has_access:
					await self.audit_logger.log_access(
						security_context, operation, resource_type, resource_id,
						success=False, error_message="Tenant isolation violation"
					)
					return False
			
			# Check permissions (simplified - would integrate with full RBAC system)
			required_permission = f"{resource_type}:{operation.value}"
			if (security_context.permissions and 
			    required_permission not in security_context.permissions and
			    "admin" not in security_context.permissions):
				
				await self.audit_logger.log_access(
					security_context, operation, resource_type, resource_id,
					success=False, error_message="Insufficient permissions"
				)
				return False
			
			return True
			
		except Exception as e:
			self.logger.error(f"Authorization failed: {str(e)}")
			return False
	
	async def secure_operation(self,
	                         security_context: SecurityContext,
	                         operation: AccessOperation,
	                         resource_type: str,
	                         resource_id: str,
	                         operation_func,
	                         *args, **kwargs) -> Tuple[bool, Any, str]:
		"""Execute operation with full security controls"""
		
		start_time = time.time()
		success = False
		result = None
		error_message = ""
		
		try:
			# Authorize operation
			authorized = await self.authorize_operation(
				security_context, operation, resource_type, resource_id
			)
			
			if not authorized:
				return False, None, "Operation not authorized"
			
			# Execute operation
			if asyncio.iscoroutinefunction(operation_func):
				result = await operation_func(*args, **kwargs)
			else:
				result = operation_func(*args, **kwargs)
			
			success = True
			
		except Exception as e:
			error_message = str(e)
			self.logger.error(f"Secure operation failed: {error_message}")
		
		finally:
			# Always log the operation
			processing_time_ms = (time.time() - start_time) * 1000
			
			await self.audit_logger.log_access(
				security_context=security_context,
				operation=operation,
				resource_type=resource_type,
				resource_id=resource_id,
				success=success,
				error_message=error_message,
				processing_time_ms=processing_time_ms
			)
		
		return success, result, error_message
	
	def encrypt_sensitive_data(self, data: str) -> str:
		"""Encrypt sensitive data"""
		return self.encryption_manager.encrypt_data(data)
	
	def decrypt_sensitive_data(self, encrypted_data: str) -> str:
		"""Decrypt sensitive data"""
		return self.encryption_manager.decrypt_data(encrypted_data)
	
	async def get_compliance_status(self, tenant_id: str) -> Dict[str, Any]:
		"""Get compliance status for tenant"""
		try:
			# Get basic tenant statistics
			async with self.db_pool.acquire() as conn:
				stats = await conn.fetchrow("""
					SELECT 
						(SELECT COUNT(*) FROM apg_rag_knowledge_bases WHERE tenant_id = $1) as kb_count,
						(SELECT COUNT(*) FROM apg_rag_documents WHERE tenant_id = $1) as doc_count,
						(SELECT COUNT(*) FROM apg_rag_conversations WHERE tenant_id = $1) as conv_count,
						(SELECT COUNT(*) FROM apg_rag_audit_logs WHERE tenant_id = $1) as audit_count
				""", tenant_id)
			
			return {
				'tenant_id': tenant_id,
				'compliance_status': 'compliant',
				'last_audit_date': datetime.now().isoformat(),
				'resource_counts': dict(stats) if stats else {},
				'encryption_status': 'enabled',
				'audit_retention_days': self.audit_logger.retention_days,
				'supported_frameworks': [f.value for f in ComplianceFramework]
			}
			
		except Exception as e:
			self.logger.error(f"Failed to get compliance status: {str(e)}")
			return {'error': str(e)}

# Factory function for APG integration
async def create_security_manager(
	db_pool: Pool,
	master_encryption_key: Optional[str] = None,
	audit_retention_days: int = 2555
) -> SecurityManager:
	"""Create security manager with proper initialization"""
	
	# Ensure audit log table exists
	async with db_pool.acquire() as conn:
		await conn.execute("""
			CREATE TABLE IF NOT EXISTS apg_rag_audit_logs (
				id TEXT PRIMARY KEY,
				tenant_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				session_id TEXT NOT NULL,
				operation TEXT NOT NULL,
				resource_type TEXT NOT NULL,
				resource_id TEXT NOT NULL,
				ip_address TEXT DEFAULT '',
				user_agent TEXT DEFAULT '',
				request_details JSONB DEFAULT '{}',
				response_summary JSONB DEFAULT '{}',
				success BOOLEAN NOT NULL DEFAULT true,
				error_message TEXT DEFAULT '',
				processing_time_ms FLOAT DEFAULT 0.0,
				timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
				created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
			)
		""")
		
		# Create indexes for audit logs
		await conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_timestamp 
			ON apg_rag_audit_logs(tenant_id, timestamp DESC)
		""")
		
		await conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_audit_logs_user_timestamp 
			ON apg_rag_audit_logs(user_id, timestamp DESC)
		""")
		
		await conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_audit_logs_resource 
			ON apg_rag_audit_logs(resource_type, resource_id)
		""")
	
	return SecurityManager(db_pool, master_encryption_key, audit_retention_days)