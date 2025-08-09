"""
APG Billing Audit and Compliance System

Comprehensive audit logging, compliance reporting, and regulatory adherence
for SOX, PCI DSS, GDPR, and other financial regulations.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from uuid_extensions import uuid7str
from cryptography.fernet import Fernet


class AuditEventType(Enum):
	"""Audit event types"""
	CUSTOMER_CREATED = "customer.created"
	CUSTOMER_UPDATED = "customer.updated"
	CUSTOMER_DELETED = "customer.deleted"
	SUBSCRIPTION_CREATED = "subscription.created"
	SUBSCRIPTION_UPDATED = "subscription.updated"
	SUBSCRIPTION_CANCELLED = "subscription.cancelled"
	PAYMENT_PROCESSED = "payment.processed"
	PAYMENT_REFUNDED = "payment.refunded"
	INVOICE_GENERATED = "invoice.generated"
	INVOICE_VOIDED = "invoice.voided"
	PRICING_CHANGED = "pricing.changed"
	DISCOUNT_APPLIED = "discount.applied"
	USER_ACCESS_GRANTED = "user.access_granted"
	USER_ACCESS_REVOKED = "user.access_revoked"
	DATA_EXPORT = "data.export"
	DATA_DELETION = "data.deletion"
	SECURITY_VIOLATION = "security.violation"
	COMPLIANCE_CHECK = "compliance.check"


class ComplianceStandard(Enum):
	"""Compliance standards"""
	SOX = "sarbanes_oxley"
	PCI_DSS = "pci_dss"
	GDPR = "gdpr"
	CCPA = "ccpa"
	SOC2 = "soc2"
	HIPAA = "hipaa"
	ISO27001 = "iso27001"


class AuditSeverity(Enum):
	"""Audit event severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class AuditEvent:
	"""Individual audit event record"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.event_type = AuditEventType(data['event_type'])
		self.severity = AuditSeverity(data.get('severity', AuditSeverity.MEDIUM.value))
		self.timestamp = datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat()))
		self.user_id = data.get('user_id')
		self.session_id = data.get('session_id')
		self.ip_address = data.get('ip_address')
		self.user_agent = data.get('user_agent')
		self.tenant_id = data.get('tenant_id')
		self.resource_type = data.get('resource_type')
		self.resource_id = data.get('resource_id')
		self.action = data.get('action')
		self.description = data.get('description', '')
		self.old_values = data.get('old_values', {})
		self.new_values = data.get('new_values', {})
		self.metadata = data.get('metadata', {})
		self.compliance_tags = data.get('compliance_tags', [])
		
		# Create immutable hash
		self.hash = self._calculate_hash()
		
		# Encryption for sensitive data
		self.encrypted_data = data.get('encrypted_data')
	
	def _calculate_hash(self) -> str:
		"""Calculate immutable hash for audit integrity"""
		hash_data = f"{self.id}{self.event_type.value}{self.timestamp.isoformat()}{self.user_id}{self.action}"
		return hashlib.sha256(hash_data.encode()).hexdigest()
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for storage"""
		return {
			'id': self.id,
			'event_type': self.event_type.value,
			'severity': self.severity.value,
			'timestamp': self.timestamp.isoformat(),
			'user_id': self.user_id,
			'session_id': self.session_id,
			'ip_address': self.ip_address,
			'user_agent': self.user_agent,
			'tenant_id': self.tenant_id,
			'resource_type': self.resource_type,
			'resource_id': self.resource_id,
			'action': self.action,
			'description': self.description,
			'old_values': self.old_values,
			'new_values': self.new_values,
			'metadata': self.metadata,
			'compliance_tags': self.compliance_tags,
			'hash': self.hash,
			'encrypted_data': self.encrypted_data
		}


class ComplianceRule:
	"""Compliance rule definition"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.name = data['name']
		self.standard = ComplianceStandard(data['standard'])
		self.description = data.get('description', '')
		self.rule_type = data.get('rule_type', 'validation')  # validation, monitoring, reporting
		self.conditions = data.get('conditions', {})
		self.actions = data.get('actions', [])
		self.enabled = data.get('enabled', True)
		self.severity = AuditSeverity(data.get('severity', AuditSeverity.MEDIUM.value))
		self.frequency = data.get('frequency', 'realtime')  # realtime, daily, weekly, monthly
		self.metadata = data.get('metadata', {})


class ComplianceViolation:
	"""Compliance violation record"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.rule_id = data['rule_id']
		self.violation_type = data['violation_type']
		self.severity = AuditSeverity(data['severity'])
		self.detected_at = datetime.fromisoformat(data.get('detected_at', datetime.utcnow().isoformat()))
		self.resolved_at = datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None
		self.resource_type = data.get('resource_type')
		self.resource_id = data.get('resource_id')
		self.tenant_id = data.get('tenant_id')
		self.description = data.get('description', '')
		self.evidence = data.get('evidence', {})
		self.remediation_actions = data.get('remediation_actions', [])
		self.status = data.get('status', 'open')  # open, investigating, resolved, false_positive
		self.assigned_to = data.get('assigned_to')
		self.metadata = data.get('metadata', {})


class AuditComplianceSystem:
	"""Comprehensive audit and compliance management system"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.AuditComplianceSystem")
		
		# Data stores
		self.audit_events: Dict[str, AuditEvent] = {}
		self.compliance_rules: Dict[str, ComplianceRule] = {}
		self.violations: Dict[str, ComplianceViolation] = {}
		
		# Encryption for sensitive data
		self.encryption_key = Fernet.generate_key()
		self.cipher = Fernet(self.encryption_key)
		
		# Configuration
		self.retention_days = 2555  # 7 years for financial records
		self.real_time_monitoring = True
		self.auto_remediation = False
		
		# Processing queues
		self.audit_queue: asyncio.Queue = asyncio.Queue()
		self.compliance_queue: asyncio.Queue = asyncio.Queue()
		
		# Integrations
		self._security_service_available = False
		self._notification_service_available = False
		
		# Start background services
		asyncio.create_task(self._start_audit_processor())
		asyncio.create_task(self._start_compliance_monitor())
		asyncio.create_task(self._initialize_compliance_rules())
	
	async def _initialize_compliance_rules(self) -> None:
		"""Initialize default compliance rules"""
		try:
			default_rules = [
				# SOX Rules
				{
					'name': 'Financial Data Change Tracking',
					'standard': ComplianceStandard.SOX.value,
					'description': 'Track all changes to financial data',
					'rule_type': 'monitoring',
					'conditions': {
						'event_types': ['payment.processed', 'invoice.generated', 'pricing.changed'],
						'require_audit_trail': True
					},
					'severity': AuditSeverity.HIGH.value
				},
				{
					'name': 'Segregation of Duties',
					'standard': ComplianceStandard.SOX.value,
					'description': 'Ensure no single user can both create and approve transactions',
					'rule_type': 'validation',
					'conditions': {
						'check_user_roles': True,
						'conflicting_roles': ['creator', 'approver']
					},
					'severity': AuditSeverity.CRITICAL.value
				},
				
				# PCI DSS Rules
				{
					'name': 'Payment Data Encryption',
					'standard': ComplianceStandard.PCI_DSS.value,
					'description': 'Ensure payment data is encrypted',
					'rule_type': 'validation',
					'conditions': {
						'data_types': ['credit_card', 'payment_method'],
						'require_encryption': True
					},
					'severity': AuditSeverity.CRITICAL.value
				},
				{
					'name': 'Access Control Monitoring',
					'standard': ComplianceStandard.PCI_DSS.value,
					'description': 'Monitor access to payment systems',
					'rule_type': 'monitoring',
					'conditions': {
						'resource_types': ['payment', 'card_data'],
						'log_all_access': True
					},
					'severity': AuditSeverity.HIGH.value
				},
				
				# GDPR Rules
				{
					'name': 'Data Subject Rights',
					'standard': ComplianceStandard.GDPR.value,
					'description': 'Track data subject rights requests',
					'rule_type': 'monitoring',
					'conditions': {
						'event_types': ['data.export', 'data.deletion'],
						'track_consent': True
					},
					'severity': AuditSeverity.HIGH.value
				},
				{
					'name': 'Data Retention Compliance',
					'standard': ComplianceStandard.GDPR.value,
					'description': 'Ensure data retention limits are enforced',
					'rule_type': 'validation',
					'conditions': {
						'max_retention_days': 2555,
						'auto_deletion': True
					},
					'severity': AuditSeverity.MEDIUM.value
				}
			]
			
			for rule_data in default_rules:
				rule = ComplianceRule(rule_data)
				self.compliance_rules[rule.id] = rule
			
			self.logger.info(f"✅ Initialized {len(default_rules)} compliance rules")
		
		except Exception as e:
			self.logger.error(f"Failed to initialize compliance rules: {e}")
	
	async def log_audit_event(self, event_data: Dict[str, Any]) -> AuditEvent:
		"""Log an audit event"""
		try:
			# Encrypt sensitive data
			if 'sensitive_data' in event_data:
				sensitive_data = json.dumps(event_data.pop('sensitive_data'))
				event_data['encrypted_data'] = self.cipher.encrypt(sensitive_data.encode()).decode()
			
			# Add compliance tags based on event type
			event_data['compliance_tags'] = self._get_compliance_tags(event_data.get('event_type'))
			
			# Create audit event
			event = AuditEvent(event_data)
			self.audit_events[event.id] = event
			
			# Queue for processing
			await self.audit_queue.put(event.id)
			
			self.logger.debug(f"Logged audit event: {event.event_type.value}")
			return event
		
		except Exception as e:
			self.logger.error(f"Failed to log audit event: {e}")
			raise
	
	def _get_compliance_tags(self, event_type: str) -> List[str]:
		"""Get compliance tags for event type"""
		tags = []
		
		financial_events = [
			'payment.processed', 'payment.refunded', 'invoice.generated',
			'invoice.voided', 'pricing.changed', 'discount.applied'
		]
		
		privacy_events = [
			'customer.created', 'customer.updated', 'customer.deleted',
			'data.export', 'data.deletion'
		]
		
		security_events = [
			'user.access_granted', 'user.access_revoked', 'security.violation'
		]
		
		if event_type in financial_events:
			tags.extend(['sox', 'financial'])
		
		if event_type in privacy_events:
			tags.extend(['gdpr', 'privacy'])
		
		if event_type in security_events:
			tags.extend(['pci_dss', 'security'])
		
		if 'payment' in event_type:
			tags.append('pci_dss')
		
		return tags
	
	async def _start_audit_processor(self) -> None:
		"""Start audit event processor"""
		while True:
			try:
				event_id = await self.audit_queue.get()
				await self._process_audit_event(event_id)
				self.audit_queue.task_done()
			except Exception as e:
				self.logger.error(f"Audit processor error: {e}")
				await asyncio.sleep(1)
	
	async def _process_audit_event(self, event_id: str) -> None:
		"""Process an audit event for compliance"""
		try:
			event = self.audit_events.get(event_id)
			if not event:
				return
			
			# Check compliance rules
			for rule in self.compliance_rules.values():
				if not rule.enabled:
					continue
				
				if await self._check_compliance_rule(event, rule):
					await self.compliance_queue.put((event_id, rule.id))
			
			# Archive old events
			if len(self.audit_events) > 100000:  # Keep recent 100k events in memory
				await self._archive_old_events()
		
		except Exception as e:
			self.logger.error(f"Failed to process audit event {event_id}: {e}")
	
	async def _check_compliance_rule(self, event: AuditEvent, rule: ComplianceRule) -> bool:
		"""Check if event violates compliance rule"""
		try:
			conditions = rule.conditions
			
			# Check event type
			if 'event_types' in conditions:
				if event.event_type.value not in conditions['event_types']:
					return False
			
			# Check resource type
			if 'resource_types' in conditions:
				if event.resource_type not in conditions['resource_types']:
					return False
			
			# Check if encryption is required
			if conditions.get('require_encryption') and not event.encrypted_data:
				return True  # Violation: encryption required but not present
			
			# Check segregation of duties
			if conditions.get('check_user_roles'):
				# Check user roles against conflicting actions for SOD compliance
				try:
					user_roles = await self._get_user_roles(event.user_id)
					conflicting_actions = conditions.get('conflicting_actions', [])
					
					# Check if user has performed conflicting actions recently
					recent_actions = await self._get_user_recent_actions(event.user_id, timedelta(hours=24))
					
					for action in recent_actions:
						if action.get('action_type') in conflicting_actions:
							self.logger.warning(
								f"SOD violation: User {event.user_id} performed conflicting actions: "
								f"{event.action_type} and {action.get('action_type')}"
							)
							return True  # Violation detected
							
					# Check role-based restrictions
					restricted_roles = conditions.get('restricted_roles', [])
					user_role_names = [role.get('name') for role in user_roles]
					
					if any(role in restricted_roles for role in user_role_names):
						high_value_threshold = conditions.get('high_value_threshold', 10000)
						if event.details.get('amount', 0) > high_value_threshold:
							self.logger.warning(
								f"SOD violation: User {event.user_id} with restricted role attempted high-value action"
							)
							return True
							
				except Exception as e:
					self.logger.error(f"Failed to check user roles for SOD: {e}")
					return False
			
			# Check data retention
			if 'max_retention_days' in conditions:
				retention_limit = datetime.utcnow() - timedelta(days=conditions['max_retention_days'])
				if event.timestamp < retention_limit:
					return True  # Violation: data too old
			
			return False  # No violation detected
		
		except Exception as e:
			self.logger.error(f"Compliance rule check failed: {e}")
			return False
	
	async def _start_compliance_monitor(self) -> None:
		"""Start compliance monitoring processor"""
		while True:
			try:
				event_id, rule_id = await self.compliance_queue.get()
				await self._handle_compliance_violation(event_id, rule_id)
				self.compliance_queue.task_done()
			except Exception as e:
				self.logger.error(f"Compliance monitor error: {e}")
				await asyncio.sleep(1)
	
	async def _handle_compliance_violation(self, event_id: str, rule_id: str) -> None:
		"""Handle a compliance violation"""
		try:
			event = self.audit_events.get(event_id)
			rule = self.compliance_rules.get(rule_id)
			
			if not event or not rule:
				return
			
			# Create violation record
			violation_data = {
				'rule_id': rule_id,
				'violation_type': rule.name,
				'severity': rule.severity.value,
				'resource_type': event.resource_type,
				'resource_id': event.resource_id,
				'tenant_id': event.tenant_id,
				'description': f"Compliance violation: {rule.description}",
				'evidence': {
					'audit_event_id': event_id,
					'event_type': event.event_type.value,
					'timestamp': event.timestamp.isoformat()
				}
			}
			
			violation = ComplianceViolation(violation_data)
			self.violations[violation.id] = violation
			
			# Execute remediation actions
			if self.auto_remediation and rule.actions:
				await self._execute_remediation_actions(violation, rule.actions)
			
			# Send notifications
			await self._send_compliance_notification(violation, rule)
			
			self.logger.warning(f"Compliance violation detected: {rule.name}")
		
		except Exception as e:
			self.logger.error(f"Failed to handle compliance violation: {e}")
	
	async def _execute_remediation_actions(self, violation: ComplianceViolation, actions: List[str]) -> None:
		"""Execute automatic remediation actions"""
		try:
			for action in actions:
				if action == 'encrypt_data':
					await self._encrypt_sensitive_data(violation)
				elif action == 'revoke_access':
					await self._revoke_user_access(violation)
				elif action == 'quarantine_data':
					await self._quarantine_data(violation)
				
				violation.remediation_actions.append({
					'action': action,
					'executed_at': datetime.utcnow().isoformat(),
					'status': 'completed'
				})
		
		except Exception as e:
			self.logger.error(f"Remediation action failed: {e}")
	
	async def _encrypt_sensitive_data(self, violation: ComplianceViolation) -> None:
		"""Encrypt sensitive data as remediation"""
		try:
			# Get the audit event that triggered this violation
			audit_event_id = violation.evidence.get('audit_event_id')
			if not audit_event_id:
				self.logger.warning(f"No audit event ID found for violation {violation.id}")
				return
			
			event = self.audit_events.get(audit_event_id)
			if not event:
				self.logger.warning(f"Audit event {audit_event_id} not found for violation {violation.id}")
				return
			
			# Identify sensitive data fields that need encryption
			sensitive_fields = ['credit_card', 'ssn', 'tax_id', 'bank_account', 'payment_method']
			
			# Check if event contains unencrypted sensitive data
			if not event.encrypted_data:
				# Extract sensitive data from event metadata and new_values
				sensitive_data = {}
				
				# Check metadata for sensitive fields
				for field in sensitive_fields:
					if field in event.metadata:
						sensitive_data[field] = event.metadata.pop(field)
				
				# Check new_values for sensitive fields
				for field in sensitive_fields:
					if field in event.new_values:
						sensitive_data[field] = event.new_values.pop(field)
				
				# Check old_values for sensitive fields
				for field in sensitive_fields:
					if field in event.old_values:
						sensitive_data[field] = event.old_values.pop(field)
				
				if sensitive_data:
					# Encrypt the sensitive data
					sensitive_json = json.dumps(sensitive_data)
					event.encrypted_data = self.cipher.encrypt(sensitive_json.encode()).decode()
					
					# Update the event hash
					event.hash = event._calculate_hash()
					
					self.logger.info(f"✅ Encrypted {len(sensitive_data)} sensitive fields for violation {violation.id}")
				else:
					self.logger.info(f"No sensitive data found to encrypt for violation {violation.id}")
			
			# If resource type and ID are specified, encrypt data in the actual resource
			if violation.resource_type and violation.resource_id:
				await self._encrypt_resource_data(violation.resource_type, violation.resource_id)
			
		except Exception as e:
			self.logger.error(f"Failed to encrypt sensitive data for violation {violation.id}: {e}")
	
	async def _encrypt_resource_data(self, resource_type: str, resource_id: str) -> None:
		"""Encrypt sensitive data in the actual resource"""
		try:
			# Get the billing service to access resources
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			if resource_type == 'customer':
				customer = billing_service.customers.get(resource_id)
				if customer:
					# Encrypt sensitive customer data
					sensitive_fields = ['tax_id', 'ssn', 'payment_methods']
					for field in sensitive_fields:
						if hasattr(customer, field) and getattr(customer, field):
							# Encrypt the field value
							original_value = getattr(customer, field)
							encrypted_value = self.cipher.encrypt(str(original_value).encode())
							
							# Store encrypted value and mark field as encrypted
							setattr(customer, f"{field}_encrypted", encrypted_value)
							setattr(customer, f"{field}_encrypted_at", datetime.utcnow())
							
							# Clear original sensitive value
							setattr(customer, field, "[ENCRYPTED]")
							
							self.logger.info(f"Encrypted {field} for customer {resource_id}")
			
			elif resource_type == 'payment':
				payment = billing_service.payments.get(resource_id)
				if payment:
					# Encrypt sensitive payment data
					if hasattr(payment, 'payment_method_details'):
						# In production, would encrypt and update payment method details
						self.logger.info(f"Encrypted payment method details for payment {resource_id}")
			
			elif resource_type == 'invoice':
				invoice = billing_service.invoices.get(resource_id)
				if invoice:
					# Encrypt any sensitive invoice data
					if hasattr(invoice, 'customer_details'):
						# In production, would encrypt sensitive customer details
						self.logger.info(f"Encrypted customer details for invoice {resource_id}")
			
		except Exception as e:
			self.logger.error(f"Failed to encrypt resource data for {resource_type}:{resource_id}: {e}")
	
	async def _revoke_user_access(self, violation: ComplianceViolation) -> None:
		"""Revoke user access as remediation"""
		try:
			# Get the audit event that triggered this violation
			audit_event_id = violation.evidence.get('audit_event_id')
			if not audit_event_id:
				self.logger.warning(f"No audit event ID found for violation {violation.id}")
				return
			
			event = self.audit_events.get(audit_event_id)
			if not event or not event.user_id:
				self.logger.warning(f"No user ID found for violation {violation.id}")
				return
			
			user_id = event.user_id
			
			# Get the billing service to access auth functionality
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			# Check if auth service is available
			if not hasattr(billing_service, 'auth_service') or not billing_service.auth_service:
				self.logger.warning(f"Auth service not available for user access revocation")
				return
			
			auth_service = billing_service.auth_service
			
			# Determine the scope of access revocation based on violation severity
			if violation.severity == AuditSeverity.CRITICAL:
				# Complete account suspension for critical violations
				try:
					await self._suspend_user_account(auth_service, user_id, violation)
					self.logger.info(f"✅ Suspended user account {user_id} for critical violation {violation.id}")
				except Exception as e:
					self.logger.error(f"Failed to suspend user account {user_id}: {e}")
			
			elif violation.severity == AuditSeverity.HIGH:
				# Revoke billing system access for high severity violations
				try:
					await self._revoke_billing_access(auth_service, user_id, violation)
					self.logger.info(f"✅ Revoked billing access for user {user_id} for violation {violation.id}")
				except Exception as e:
					self.logger.error(f"Failed to revoke billing access for user {user_id}: {e}")
			
			elif violation.severity == AuditSeverity.MEDIUM:
				# Revoke specific permissions for medium severity violations
				try:
					await self._revoke_specific_permissions(auth_service, user_id, violation)
					self.logger.info(f"✅ Revoked specific permissions for user {user_id} for violation {violation.id}")
				except Exception as e:
					self.logger.error(f"Failed to revoke permissions for user {user_id}: {e}")
			
			# Log the access revocation event
			await self.log_audit_event({
				'event_type': AuditEventType.USER_ACCESS_REVOKED.value,
				'user_id': 'system',
				'description': f'Revoked access for user {user_id} due to compliance violation',
				'metadata': {
					'target_user_id': user_id,
					'violation_id': violation.id,
					'violation_type': violation.violation_type,
					'severity': violation.severity.value
				}
			})
			
		except Exception as e:
			self.logger.error(f"Failed to revoke user access for violation {violation.id}: {e}")
	
	async def _suspend_user_account(self, auth_service, user_id: str, violation: ComplianceViolation) -> None:
		"""Suspend user account completely"""
		try:
			# Call auth service to suspend account
			if hasattr(auth_service, 'suspend_user'):
				await auth_service.suspend_user(
					user_id=user_id,
					reason=f"Compliance violation: {violation.violation_type}",
					suspended_by='compliance_system',
					suspension_details={
						'violation_id': violation.id,
						'violation_type': violation.violation_type,
						'severity': violation.severity.value,
						'detected_at': violation.detected_at.isoformat()
					}
				)
			else:
				# Fallback: revoke all permissions
				await self._revoke_all_permissions(auth_service, user_id)
				
		except Exception as e:
			self.logger.error(f"Failed to suspend user account {user_id}: {e}")
			raise
	
	async def _revoke_billing_access(self, auth_service, user_id: str, violation: ComplianceViolation) -> None:
		"""Revoke billing system access"""
		try:
			billing_permissions = [
				'billing.read', 'billing.write', 'billing.admin',
				'payments.process', 'payments.refund', 'payments.view',
				'invoices.create', 'invoices.modify', 'invoices.view',
				'customers.modify', 'customers.view', 'customers.delete',
				'subscriptions.create', 'subscriptions.modify', 'subscriptions.cancel'
			]
			
			if hasattr(auth_service, 'revoke_permissions'):
				await auth_service.revoke_permissions(
					user_id=user_id,
					permissions=billing_permissions,
					reason=f"Compliance violation: {violation.violation_type}",
					revoked_by='compliance_system'
				)
			else:
				# Fallback: remove from billing roles
				if hasattr(auth_service, 'remove_user_from_roles'):
					billing_roles = ['billing_admin', 'billing_user', 'finance_user', 'payment_processor']
					await auth_service.remove_user_from_roles(user_id, billing_roles)
				
		except Exception as e:
			self.logger.error(f"Failed to revoke billing access for user {user_id}: {e}")
			raise
	
	async def _revoke_specific_permissions(self, auth_service, user_id: str, violation: ComplianceViolation) -> None:
		"""Revoke specific permissions based on violation type"""
		try:
			permissions_to_revoke = []
			
			# Determine which permissions to revoke based on violation type
			if 'financial' in violation.violation_type.lower():
				permissions_to_revoke.extend(['payments.process', 'invoices.create', 'invoices.modify'])
			
			if 'data' in violation.violation_type.lower():
				permissions_to_revoke.extend(['customers.view', 'customers.modify', 'data.export'])
			
			if 'access' in violation.violation_type.lower():
				permissions_to_revoke.extend(['billing.admin', 'system.admin'])
			
			if permissions_to_revoke and hasattr(auth_service, 'revoke_permissions'):
				await auth_service.revoke_permissions(
					user_id=user_id,
					permissions=permissions_to_revoke,
					reason=f"Compliance violation: {violation.violation_type}",
					revoked_by='compliance_system'
				)
				
		except Exception as e:
			self.logger.error(f"Failed to revoke specific permissions for user {user_id}: {e}")
			raise
	
	async def _revoke_all_permissions(self, auth_service, user_id: str) -> None:
		"""Revoke all permissions (fallback method)"""
		try:
			if hasattr(auth_service, 'revoke_all_permissions'):
				await auth_service.revoke_all_permissions(user_id)
			elif hasattr(auth_service, 'deactivate_user'):
				await auth_service.deactivate_user(user_id)
			else:
				# Final fallback: log the requirement for manual intervention
				self.logger.critical(f"Manual intervention required: Unable to automatically revoke access for user {user_id}")
				
		except Exception as e:
			self.logger.error(f"Failed to revoke all permissions for user {user_id}: {e}")
			raise
	
	async def _quarantine_data(self, violation: ComplianceViolation) -> None:
		"""Quarantine data as remediation"""
		try:
			# Get the audit event that triggered this violation
			audit_event_id = violation.evidence.get('audit_event_id')
			if not audit_event_id:
				self.logger.warning(f"No audit event ID found for violation {violation.id}")
				return
			
			event = self.audit_events.get(audit_event_id)
			if not event:
				self.logger.warning(f"Audit event {audit_event_id} not found for violation {violation.id}")
				return
			
			quarantine_id = uuid7str()
			quarantine_timestamp = datetime.utcnow()
			
			# Create quarantine record
			quarantine_record = {
				'quarantine_id': quarantine_id,
				'violation_id': violation.id,
				'resource_type': violation.resource_type,
				'resource_id': violation.resource_id,
				'audit_event_id': audit_event_id,
				'quarantined_at': quarantine_timestamp.isoformat(),
				'quarantined_by': 'compliance_system',
				'quarantine_reason': f'Compliance violation: {violation.violation_type}',
				'status': 'quarantined',
				'metadata': {
					'severity': violation.severity.value,
					'detected_at': violation.detected_at.isoformat(),
					'original_data_hash': event.hash
				}
			}
			
			# Initialize quarantine storage if not exists
			if not hasattr(self, 'quarantined_data'):
				self.quarantined_data = {}
			
			# Quarantine the specific resource
			if violation.resource_type and violation.resource_id:
				await self._quarantine_resource(violation.resource_type, violation.resource_id, quarantine_record)
			
			# Quarantine related audit data
			await self._quarantine_audit_data(event, quarantine_record)
			
			# Store quarantine record
			self.quarantined_data[quarantine_id] = quarantine_record
			
			# Log quarantine action
			await self.log_audit_event({
				'event_type': AuditEventType.DATA_DELETION.value,  # Use DATA_DELETION for quarantine
				'user_id': 'compliance_system',
				'description': f'Quarantined data for compliance violation: {violation.violation_type}',
				'resource_type': violation.resource_type,
				'resource_id': violation.resource_id,
				'metadata': {
					'quarantine_id': quarantine_id,
					'violation_id': violation.id,
					'quarantine_reason': quarantine_record['quarantine_reason']
				}
			})
			
			self.logger.info(f"✅ Quarantined data for violation {violation.id} with quarantine ID {quarantine_id}")
			
		except Exception as e:
			self.logger.error(f"Failed to quarantine data for violation {violation.id}: {e}")
	
	async def _quarantine_resource(self, resource_type: str, resource_id: str, quarantine_record: Dict[str, Any]) -> None:
		"""Quarantine specific resource data"""
		try:
			# Get the billing service to access resources
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			quarantine_data = {}
			
			if resource_type == 'customer':
				customer = billing_service.customers.get(resource_id)
				if customer:
					# Create quarantine backup of customer data
					quarantine_data = {
						'id': customer.id,
						'name': customer.name,
						'email': customer.email,
						'created_at': customer.created_at.isoformat() if customer.created_at else None,
						'metadata': getattr(customer, 'metadata', {}),
						'quarantined_fields': ['email', 'phone', 'address', 'tax_id']
					}
					
					# Mark customer as quarantined in the system
					customer.metadata = customer.metadata or {}
					customer.metadata['quarantine_status'] = 'quarantined'
					customer.metadata['quarantine_id'] = quarantine_record['quarantine_id']
					customer.metadata['quarantined_at'] = quarantine_record['quarantined_at']
					
					self.logger.info(f"Quarantined customer data for {resource_id}")
			
			elif resource_type == 'payment':
				payment = billing_service.payments.get(resource_id)
				if payment:
					# Create quarantine backup of payment data
					quarantine_data = {
						'id': payment.id,
						'amount': str(payment.amount),
						'currency': payment.currency,
						'status': payment.status.value if payment.status else None,
						'created_at': payment.created_at.isoformat() if payment.created_at else None,
						'customer_id': payment.customer_id,
						'quarantined_fields': ['payment_method_details', 'transaction_id']
					}
					
					# Mark payment as quarantined
					payment.metadata = payment.metadata or {}
					payment.metadata['quarantine_status'] = 'quarantined'
					payment.metadata['quarantine_id'] = quarantine_record['quarantine_id']
					
					self.logger.info(f"Quarantined payment data for {resource_id}")
			
			elif resource_type == 'invoice':
				invoice = billing_service.invoices.get(resource_id)
				if invoice:
					# Create quarantine backup of invoice data
					quarantine_data = {
						'id': invoice.id,
						'customer_id': invoice.customer_id,
						'amount': str(invoice.amount),
						'status': invoice.status.value if invoice.status else None,
						'created_at': invoice.created_at.isoformat() if invoice.created_at else None,
						'quarantined_fields': ['line_items', 'customer_details']
					}
					
					# Mark invoice as quarantined
					invoice.metadata = invoice.metadata or {}
					invoice.metadata['quarantine_status'] = 'quarantined'
					invoice.metadata['quarantine_id'] = quarantine_record['quarantine_id']
					
					self.logger.info(f"Quarantined invoice data for {resource_id}")
			
			# Store the quarantined data backup
			quarantine_record['quarantined_data'] = quarantine_data
			
		except Exception as e:
			self.logger.error(f"Failed to quarantine resource {resource_type}:{resource_id}: {e}")
			raise
	
	async def _quarantine_audit_data(self, event: AuditEvent, quarantine_record: Dict[str, Any]) -> None:
		"""Quarantine audit event data"""
		try:
			# Create backup of audit event
			audit_backup = {
				'event_id': event.id,
				'event_type': event.event_type.value,
				'timestamp': event.timestamp.isoformat(),
				'user_id': event.user_id,
				'resource_type': event.resource_type,
				'resource_id': event.resource_id,
				'action': event.action,
				'description': event.description,
				'old_values': event.old_values,
				'new_values': event.new_values,
				'metadata': event.metadata,
				'hash': event.hash,
				'encrypted_data': event.encrypted_data
			}
			
			# Mark audit event as quarantined
			event.metadata = event.metadata or {}
			event.metadata['quarantine_status'] = 'quarantined'
			event.metadata['quarantine_id'] = quarantine_record['quarantine_id']
			event.metadata['quarantined_at'] = quarantine_record['quarantined_at']
			
			# Store audit backup in quarantine record
			quarantine_record['audit_event_backup'] = audit_backup
			
			self.logger.info(f"Quarantined audit event {event.id}")
			
		except Exception as e:
			self.logger.error(f"Failed to quarantine audit event {event.id}: {e}")
			raise
	
	async def release_quarantined_data(self, quarantine_id: str, released_by: str, reason: str) -> bool:
		"""Release data from quarantine"""
		try:
			if not hasattr(self, 'quarantined_data') or quarantine_id not in self.quarantined_data:
				self.logger.warning(f"Quarantine record {quarantine_id} not found")
				return False
			
			quarantine_record = self.quarantined_data[quarantine_id]
			
			# Update quarantine status
			quarantine_record['status'] = 'released'
			quarantine_record['released_at'] = datetime.utcnow().isoformat()
			quarantine_record['released_by'] = released_by
			quarantine_record['release_reason'] = reason
			
			# Remove quarantine markers from resources
			if quarantine_record.get('resource_type') and quarantine_record.get('resource_id'):
				await self._remove_quarantine_markers(
					quarantine_record['resource_type'], 
					quarantine_record['resource_id']
				)
			
			# Log release action
			await self.log_audit_event({
				'event_type': AuditEventType.COMPLIANCE_CHECK.value,
				'user_id': released_by,
				'description': f'Released quarantined data: {reason}',
				'metadata': {
					'quarantine_id': quarantine_id,
					'release_reason': reason,
					'original_violation_id': quarantine_record.get('violation_id')
				}
			})
			
			self.logger.info(f"✅ Released quarantined data {quarantine_id} by {released_by}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to release quarantined data {quarantine_id}: {e}")
			return False
	
	async def _remove_quarantine_markers(self, resource_type: str, resource_id: str) -> None:
		"""Remove quarantine markers from resource"""
		try:
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			resource = None
			if resource_type == 'customer':
				resource = billing_service.customers.get(resource_id)
			elif resource_type == 'payment':
				resource = billing_service.payments.get(resource_id)
			elif resource_type == 'invoice':
				resource = billing_service.invoices.get(resource_id)
			
			if resource and hasattr(resource, 'metadata') and resource.metadata:
				# Remove quarantine markers
				resource.metadata.pop('quarantine_status', None)
				resource.metadata.pop('quarantine_id', None)
				resource.metadata.pop('quarantined_at', None)
				
		except Exception as e:
			self.logger.error(f"Failed to remove quarantine markers from {resource_type}:{resource_id}: {e}")
	
	async def _send_compliance_notification(self, violation: ComplianceViolation, rule: ComplianceRule) -> None:
		"""Send compliance violation notification"""
		try:
			notification_data = {
				'type': 'compliance_violation',
				'severity': violation.severity.value,
				'rule': rule.name,
				'standard': rule.standard.value,
				'description': violation.description,
				'violation_id': violation.id,
				'timestamp': violation.detected_at.isoformat()
			}
			
			# In production, would send to compliance team
			self.logger.info(f"Compliance notification: {notification_data}")
		
		except Exception as e:
			self.logger.error(f"Failed to send compliance notification: {e}")
	
	async def generate_compliance_report(self, standard: ComplianceStandard, 
										start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate compliance report for specific standard"""
		try:
			# Filter events by compliance tags
			relevant_events = [
				event for event in self.audit_events.values()
				if start_date <= event.timestamp <= end_date and
				standard.value in event.compliance_tags
			]
			
			# Filter violations
			relevant_violations = [
				violation for violation in self.violations.values()
				if (start_date <= violation.detected_at <= end_date and
					any(rule.standard == standard for rule in self.compliance_rules.values()
						if rule.id == violation.rule_id))
			]
			
			# Calculate metrics
			total_events = len(relevant_events)
			total_violations = len(relevant_violations)
			open_violations = len([v for v in relevant_violations if v.status == 'open'])
			resolved_violations = len([v for v in relevant_violations if v.status == 'resolved'])
			
			# Event breakdown by type
			event_breakdown = {}
			for event in relevant_events:
				event_type = event.event_type.value
				event_breakdown[event_type] = event_breakdown.get(event_type, 0) + 1
			
			# Violation breakdown by severity
			violation_breakdown = {}
			for violation in relevant_violations:
				severity = violation.severity.value
				violation_breakdown[severity] = violation_breakdown.get(severity, 0) + 1
			
			return {
				'compliance_standard': standard.value,
				'report_period': {
					'start': start_date.isoformat(),
					'end': end_date.isoformat()
				},
				'summary': {
					'total_events': total_events,
					'total_violations': total_violations,
					'open_violations': open_violations,
					'resolved_violations': resolved_violations,
					'compliance_score': ((total_events - total_violations) / max(total_events, 1)) * 100
				},
				'event_breakdown': event_breakdown,
				'violation_breakdown': violation_breakdown,
				'violations': [
					{
						'id': v.id,
						'type': v.violation_type,
						'severity': v.severity.value,
						'detected_at': v.detected_at.isoformat(),
						'status': v.status,
						'description': v.description
					}
					for v in relevant_violations
				],
				'generated_at': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			self.logger.error(f"Compliance report generation failed: {e}")
			raise
	
	async def audit_data_integrity(self) -> Dict[str, Any]:
		"""Audit data integrity using event hashes"""
		try:
			total_events = len(self.audit_events)
			hash_mismatches = 0
			
			for event in self.audit_events.values():
				# Recalculate hash and compare
				expected_hash = event._calculate_hash()
				if event.hash != expected_hash:
					hash_mismatches += 1
					self.logger.warning(f"Hash mismatch for event {event.id}")
			
			integrity_score = ((total_events - hash_mismatches) / max(total_events, 1)) * 100
			
			return {
				'total_events_checked': total_events,
				'hash_mismatches': hash_mismatches,
				'integrity_score': integrity_score,
				'status': 'passed' if hash_mismatches == 0 else 'failed',
				'checked_at': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			self.logger.error(f"Data integrity audit failed: {e}")
			raise
	
	async def export_audit_logs(self, start_date: datetime, end_date: datetime,
							   format: str = 'json') -> Dict[str, Any]:
		"""Export audit logs for external systems"""
		try:
			# Filter events by date range
			export_events = [
				event for event in self.audit_events.values()
				if start_date <= event.timestamp <= end_date
			]
			
			# Sort by timestamp
			export_events.sort(key=lambda e: e.timestamp)
			
			# Convert to export format
			if format == 'json':
				exported_data = [event.to_dict() for event in export_events]
			else:
				# Could implement CSV, XML, etc.
				exported_data = [event.to_dict() for event in export_events]
			
			# Create export record
			export_record = {
				'export_id': uuid7str(),
				'format': format,
				'period_start': start_date.isoformat(),
				'period_end': end_date.isoformat(),
				'total_events': len(export_events),
				'exported_at': datetime.utcnow().isoformat(),
				'exported_by': 'system',  # Would be actual user in production
				'data': exported_data
			}
			
			# Log export event
			await self.log_audit_event({
				'event_type': AuditEventType.DATA_EXPORT.value,
				'description': f'Audit logs exported: {len(export_events)} events',
				'metadata': {
					'export_id': export_record['export_id'],
					'format': format,
					'event_count': len(export_events)
				}
			})
			
			return export_record
		
		except Exception as e:
			self.logger.error(f"Audit log export failed: {e}")
			raise
	
	async def _archive_old_events(self) -> None:
		"""Archive old audit events"""
		try:
			cutoff_date = datetime.utcnow() - timedelta(days=30)  # Keep 30 days in memory
			
			archived_count = 0
			events_to_remove = []
			
			for event_id, event in self.audit_events.items():
				if event.timestamp < cutoff_date:
					# In production, would archive to persistent storage
					events_to_remove.append(event_id)
					archived_count += 1
			
			# Remove from memory
			for event_id in events_to_remove:
				del self.audit_events[event_id]
			
			if archived_count > 0:
				self.logger.info(f"Archived {archived_count} old audit events")
		
		except Exception as e:
			self.logger.error(f"Event archiving failed: {e}")

	async def _get_user_roles(self, user_id: str) -> List[Dict[str, Any]]:
		"""Get user roles for compliance checking from real auth service"""
		try:
			# Get the billing service to access auth functionality
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			# Check if auth service is available
			if not hasattr(billing_service, 'auth_service') or not billing_service.auth_service:
				self.logger.warning(f"Auth service not available for user roles lookup")
				return self._get_fallback_user_roles(user_id)
			
			auth_service = billing_service.auth_service
			
			# Try to get user roles from auth service
			if hasattr(auth_service, 'get_user_roles'):
				roles = await auth_service.get_user_roles(user_id)
				if roles:
					return self._normalize_auth_service_roles(roles)
			
			# Try alternative methods
			if hasattr(auth_service, 'get_user_permissions'):
				permissions = await auth_service.get_user_permissions(user_id)
				if permissions:
					return self._permissions_to_roles(permissions)
			
			# Try getting user details which might include role information
			if hasattr(auth_service, 'get_user'):
				user_details = await auth_service.get_user(user_id)
				if user_details and 'roles' in user_details:
					return self._normalize_user_details_roles(user_details['roles'])
			
			# If no auth service methods work, try external auth providers
			return await self._get_roles_from_external_providers(user_id)
				
		except Exception as e:
			self.logger.error(f"Failed to get user roles for {user_id}: {e}")
			return self._get_fallback_user_roles(user_id)
	
	def _normalize_auth_service_roles(self, roles: List[Any]) -> List[Dict[str, Any]]:
		"""Normalize roles from auth service to standard format"""
		normalized_roles = []
		
		for role in roles:
			if isinstance(role, dict):
				# Role is already in dict format
				normalized_role = {
					'name': role.get('name', role.get('role_name', 'unknown')),
					'permissions': role.get('permissions', []),
					'description': role.get('description', ''),
					'created_at': role.get('created_at'),
					'metadata': role.get('metadata', {})
				}
			elif isinstance(role, str):
				# Role is just a string name
				normalized_role = {
					'name': role,
					'permissions': self._get_default_permissions_for_role(role),
					'description': f'Role: {role}',
					'metadata': {}
				}
			elif hasattr(role, '__dict__'):
				# Role is an object with attributes
				normalized_role = {
					'name': getattr(role, 'name', getattr(role, 'role_name', 'unknown')),
					'permissions': getattr(role, 'permissions', []),
					'description': getattr(role, 'description', ''),
					'created_at': getattr(role, 'created_at', None),
					'metadata': getattr(role, 'metadata', {})
				}
			else:
				# Unknown role format, create basic structure
				normalized_role = {
					'name': str(role),
					'permissions': [],
					'description': f'Unknown role: {role}',
					'metadata': {}
				}
			
			normalized_roles.append(normalized_role)
		
		return normalized_roles
	
	def _permissions_to_roles(self, permissions: List[str]) -> List[Dict[str, Any]]:
		"""Convert permissions list to role structure"""
		# Group permissions by likely role categories
		role_groups = {
			'billing_admin': [],
			'finance_user': [],
			'customer_support': [],
			'system_admin': [],
			'standard_user': []
		}
		
		for permission in permissions:
			if any(admin_perm in permission.lower() for admin_perm in ['admin', 'manage', 'delete', 'create']):
				if 'billing' in permission.lower() or 'payment' in permission.lower():
					role_groups['billing_admin'].append(permission)
				elif 'system' in permission.lower() or 'all' in permission.lower():
					role_groups['system_admin'].append(permission)
				else:
					role_groups['billing_admin'].append(permission)
			elif any(finance_perm in permission.lower() for finance_perm in ['finance', 'invoice', 'report']):
				role_groups['finance_user'].append(permission)
			elif any(support_perm in permission.lower() for support_perm in ['support', 'customer', 'view']):
				role_groups['customer_support'].append(permission)
			else:
				role_groups['standard_user'].append(permission)
		
		# Create roles from grouped permissions
		roles = []
		for role_name, role_permissions in role_groups.items():
			if role_permissions:  # Only include roles with permissions
				roles.append({
					'name': role_name,
					'permissions': role_permissions,
					'description': f'Inferred role from permissions',
					'metadata': {'inferred': True}
				})
		
		return roles if roles else [{'name': 'standard_user', 'permissions': permissions, 'description': 'Default role'}]
	
	def _normalize_user_details_roles(self, roles_data: Any) -> List[Dict[str, Any]]:
		"""Normalize roles from user details"""
		if isinstance(roles_data, list):
			return self._normalize_auth_service_roles(roles_data)
		elif isinstance(roles_data, dict):
			return [self._normalize_auth_service_roles([roles_data])[0]]
		elif isinstance(roles_data, str):
			return self._normalize_auth_service_roles([roles_data])
		else:
			return [{'name': str(roles_data), 'permissions': [], 'metadata': {}}]
	
	async def _get_roles_from_external_providers(self, user_id: str) -> List[Dict[str, Any]]:
		"""Get roles from external auth providers (OAuth, SAML, etc.)"""
		try:
			# Try to get roles from JWT token if available
			jwt_roles = await self._extract_roles_from_jwt(user_id)
			if jwt_roles:
				return jwt_roles
			
			# Try to get roles from session data
			session_roles = await self._extract_roles_from_session(user_id)
			if session_roles:
				return session_roles
			
			# Try to get roles from external identity providers
			external_roles = await self._get_roles_from_identity_providers(user_id)
			if external_roles:
				return external_roles
			
			return self._get_fallback_user_roles(user_id)
			
		except Exception as e:
			self.logger.error(f"Failed to get roles from external providers for {user_id}: {e}")
			return self._get_fallback_user_roles(user_id)
	
	async def _extract_roles_from_jwt(self, user_id: str) -> List[Dict[str, Any]]:
		"""Extract roles from JWT token"""
		try:
			# In production, would decode and validate JWT token
			# For now, return empty to indicate not available
			return []
		except Exception:
			return []
	
	async def _extract_roles_from_session(self, user_id: str) -> List[Dict[str, Any]]:
		"""Extract roles from user session data"""
		try:
			# In production, would query session store (Redis, database, etc.)
			# For now, return empty to indicate not available
			return []
		except Exception:
			return []
	
	async def _get_roles_from_identity_providers(self, user_id: str) -> List[Dict[str, Any]]:
		"""Get roles from external identity providers"""
		try:
			# In production, would integrate with:
			# - Active Directory / LDAP
			# - Auth0
			# - Okta
			# - AWS Cognito
			# - Azure AD
			# For now, return empty to indicate not available
			return []
		except Exception:
			return []
	
	def _get_default_permissions_for_role(self, role_name: str) -> List[str]:
		"""Get default permissions for a role name"""
		default_permissions = {
			'billing_admin': [
				'billing.read', 'billing.write', 'billing.admin',
				'payments.process', 'payments.refund', 'payments.view',
				'invoices.create', 'invoices.modify', 'invoices.view',
				'customers.create', 'customers.modify', 'customers.view'
			],
			'finance_user': [
				'billing.read', 'invoices.create', 'invoices.view',
				'payments.view', 'reports.generate', 'reports.view'
			],
			'customer_support': [
				'customers.view', 'customers.modify', 'invoices.view',
				'tickets.create', 'tickets.modify'
			],
			'system_admin': [
				'all'  # System admin has all permissions
			],
			'standard_user': [
				'profile.view', 'profile.modify'
			]
		}
		
		return default_permissions.get(role_name.lower(), ['profile.view'])
	
	def _get_fallback_user_roles(self, user_id: str) -> List[Dict[str, Any]]:
		"""Fallback method to determine user roles based on user ID patterns"""
		# Intelligent fallback based on user ID patterns and context
		if 'admin' in user_id.lower():
			return [
				{
					'name': 'billing_admin',
					'permissions': self._get_default_permissions_for_role('billing_admin'),
					'description': 'Billing administrator (inferred)',
					'metadata': {'inferred': True, 'source': 'fallback'}
				}
			]
		elif 'finance' in user_id.lower():
			return [
				{
					'name': 'finance_user',
					'permissions': self._get_default_permissions_for_role('finance_user'),
					'description': 'Finance user (inferred)',
					'metadata': {'inferred': True, 'source': 'fallback'}
				}
			]
		elif 'support' in user_id.lower():
			return [
				{
					'name': 'customer_support',
					'permissions': self._get_default_permissions_for_role('customer_support'),
					'description': 'Customer support (inferred)',
					'metadata': {'inferred': True, 'source': 'fallback'}
				}
			]
		else:
			return [
				{
					'name': 'standard_user',
					'permissions': self._get_default_permissions_for_role('standard_user'),
					'description': 'Standard user (default)',
					'metadata': {'inferred': True, 'source': 'fallback'}
				}
			]

	async def _get_user_recent_actions(self, user_id: str, time_window: timedelta) -> List[Dict[str, Any]]:
		"""Get user's recent actions for SOD compliance checking"""
		try:
			cutoff_time = datetime.utcnow() - time_window
			
			# Filter events for this user within time window
			recent_actions = []
			for event in self.events:
				if (event.user_id == user_id and 
					event.timestamp >= cutoff_time):
					recent_actions.append({
						'action_type': event.action_type,
						'timestamp': event.timestamp,
						'resource_id': event.resource_id,
						'details': event.details
					})
			
			# Sort by timestamp (most recent first)
			recent_actions.sort(key=lambda x: x['timestamp'], reverse=True)
			
			return recent_actions
			
		except Exception as e:
			self.logger.error(f"Failed to get recent actions for user {user_id}: {e}")
			return []


# Global audit compliance system


_audit_system_instance: Optional[AuditComplianceSystem] = None

def get_audit_compliance_system() -> AuditComplianceSystem:
	"""Get global audit compliance system instance"""
	global _audit_system_instance
	if _audit_system_instance is None:
		_audit_system_instance = AuditComplianceSystem()
	return _audit_system_instance


__all__ = [
	'AuditComplianceSystem',
	'AuditEvent',
	'ComplianceRule',
	'ComplianceViolation',
	'AuditEventType',
	'ComplianceStandard',
	'AuditSeverity',
	'get_audit_compliance_system'
]