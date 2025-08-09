"""
APG Billing Dunning Management System

Comprehensive dunning management for failed payments, overdue invoices,
and automated collection workflows with customizable escalation sequences.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from .models import BLCustomer, BLInvoice, BLPayment, BLSubscription, InvoiceStatus, PaymentStatus, SubscriptionStatus


class DunningAction(Enum):
	"""Dunning action types"""
	EMAIL_REMINDER = "email_reminder"
	PAYMENT_RETRY = "payment_retry"
	ACCOUNT_SUSPENSION = "account_suspension"
	SUBSCRIPTION_PAUSE = "subscription_pause"
	COLLECTION_AGENCY = "collection_agency"
	WRITE_OFF = "write_off"
	LEGAL_ACTION = "legal_action"
	MANUAL_FOLLOW_UP = "manual_follow_up"


class DunningStage(Enum):
	"""Dunning process stages"""
	GRACE_PERIOD = "grace_period"
	GENTLE_REMINDER = "gentle_reminder"
	FIRM_REMINDER = "firm_reminder"
	URGENT_NOTICE = "urgent_notice"
	FINAL_NOTICE = "final_notice"
	COLLECTION = "collection"
	ESCALATED = "escalated"
	RESOLVED = "resolved"
	WRITTEN_OFF = "written_off"


class DunningTemplate:
	"""Email template for dunning communications"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.name = data['name']
		self.stage = DunningStage(data['stage'])
		self.subject = data['subject']
		self.html_content = data['html_content']
		self.text_content = data.get('text_content', '')
		self.delay_days = data.get('delay_days', 0)
		self.active = data.get('active', True)
		self.language = data.get('language', 'en')
		self.metadata = data.get('metadata', {})


class DunningSequence:
	"""Dunning sequence configuration"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.name = data['name']
		self.description = data.get('description', '')
		self.steps = []
		self.active = data.get('active', True)
		self.customer_segments = data.get('customer_segments', [])  # VIP, Standard, etc.
		self.amount_thresholds = data.get('amount_thresholds', {})
		self.metadata = data.get('metadata', {})
		
		# Parse steps
		for step_data in data.get('steps', []):
			step = DunningStep(step_data)
			self.steps.append(step)
		
		# Sort steps by delay
		self.steps.sort(key=lambda s: s.delay_days)


class DunningStep:
	"""Individual step in dunning sequence"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.sequence_order = data.get('sequence_order', 0)
		self.delay_days = data['delay_days']
		self.action = DunningAction(data['action'])
		self.stage = DunningStage(data['stage'])
		self.template_id = data.get('template_id')
		self.conditions = data.get('conditions', {})
		self.auto_execute = data.get('auto_execute', True)
		self.requires_approval = data.get('requires_approval', False)
		self.metadata = data.get('metadata', {})


class DunningCase:
	"""Individual dunning case tracking"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.customer_id = data['customer_id']
		self.invoice_id = data.get('invoice_id')
		self.subscription_id = data.get('subscription_id')
		self.sequence_id = data['sequence_id']
		self.current_step = data.get('current_step', 0)
		self.stage = DunningStage(data.get('stage', DunningStage.GRACE_PERIOD.value))
		self.outstanding_amount = Decimal(str(data['outstanding_amount']))
		self.currency = data.get('currency', 'USD')
		self.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
		self.next_action_date = datetime.fromisoformat(data.get('next_action_date', datetime.utcnow().isoformat()))
		self.resolved_at = datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None
		self.resolution_type = data.get('resolution_type')  # paid, written_off, etc.
		self.paused = data.get('paused', False)
		self.pause_reason = data.get('pause_reason')
		self.priority = data.get('priority', 'normal')  # low, normal, high, urgent
		self.assigned_to = data.get('assigned_to')
		self.notes = data.get('notes', [])
		self.actions_taken = data.get('actions_taken', [])
		self.metadata = data.get('metadata', {})


class DunningAction:
	"""Record of dunning action taken"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.case_id = data['case_id']
		self.action_type = DunningAction(data['action_type'])
		self.executed_at = datetime.fromisoformat(data.get('executed_at', datetime.utcnow().isoformat()))
		self.executed_by = data.get('executed_by', 'system')
		self.result = data.get('result')  # success, failed, pending
		self.details = data.get('details', {})
		self.next_retry_at = datetime.fromisoformat(data['next_retry_at']) if data.get('next_retry_at') else None
		self.metadata = data.get('metadata', {})


class DunningManagementSystem:
	"""Comprehensive dunning management system"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.DunningManagementSystem")
		
		# Core data stores
		self.cases: Dict[str, DunningCase] = {}
		self.sequences: Dict[str, DunningSequence] = {}
		self.templates: Dict[str, DunningTemplate] = {}
		self.actions: Dict[str, DunningAction] = {}
		
		# Integration references
		self.billing_service = None
		self.email_manager = None
		self.payment_manager = None
		self.webhook_system = None
		
		# Configuration
		self.auto_execution_enabled = True
		self.business_hours_only = True
		self.business_hours = {'start': 9, 'end': 17}  # 9 AM to 5 PM
		self.excluded_days = [5, 6]  # Saturday, Sunday (0=Monday)
		
		# Scheduling
		self.processing_queue: asyncio.Queue = asyncio.Queue()
		
		# Start background processors
		asyncio.create_task(self._start_dunning_processor())
		asyncio.create_task(self._start_case_monitor())
		asyncio.create_task(self._initialize_default_sequences())
	
	async def _initialize_default_sequences(self) -> None:
		"""Initialize default dunning sequences"""
		try:
			# Standard sequence for regular customers
			standard_sequence = {
				'name': 'Standard Dunning Sequence',
				'description': 'Default sequence for regular customers',
				'customer_segments': ['standard', 'basic'],
				'steps': [
					{
						'delay_days': 3,
						'action': DunningAction.EMAIL_REMINDER.value,
						'stage': DunningStage.GRACE_PERIOD.value,
						'auto_execute': True
					},
					{
						'delay_days': 7,
						'action': DunningAction.EMAIL_REMINDER.value,
						'stage': DunningStage.GENTLE_REMINDER.value,
						'auto_execute': True
					},
					{
						'delay_days': 14,
						'action': DunningAction.PAYMENT_RETRY.value,
						'stage': DunningStage.FIRM_REMINDER.value,
						'auto_execute': True
					},
					{
						'delay_days': 21,
						'action': DunningAction.EMAIL_REMINDER.value,
						'stage': DunningStage.URGENT_NOTICE.value,
						'auto_execute': True
					},
					{
						'delay_days': 30,
						'action': DunningAction.SUBSCRIPTION_PAUSE.value,
						'stage': DunningStage.FINAL_NOTICE.value,
						'auto_execute': False,
						'requires_approval': True
					},
					{
						'delay_days': 45,
						'action': DunningAction.COLLECTION_AGENCY.value,
						'stage': DunningStage.COLLECTION.value,
						'auto_execute': False,
						'requires_approval': True
					}
				]
			}
			
			# VIP sequence for high-value customers
			vip_sequence = {
				'name': 'VIP Customer Dunning Sequence',
				'description': 'Gentle sequence for VIP customers',
				'customer_segments': ['vip', 'enterprise'],
				'amount_thresholds': {'min': 1000},
				'steps': [
					{
						'delay_days': 5,
						'action': DunningAction.EMAIL_REMINDER.value,
						'stage': DunningStage.GRACE_PERIOD.value,
						'auto_execute': True
					},
					{
						'delay_days': 10,
						'action': DunningAction.MANUAL_FOLLOW_UP.value,
						'stage': DunningStage.GENTLE_REMINDER.value,
						'auto_execute': False
					},
					{
						'delay_days': 20,
						'action': DunningAction.PAYMENT_RETRY.value,
						'stage': DunningStage.FIRM_REMINDER.value,
						'auto_execute': True
					},
					{
						'delay_days': 35,
						'action': DunningAction.MANUAL_FOLLOW_UP.value,
						'stage': DunningStage.URGENT_NOTICE.value,
						'auto_execute': False,
						'requires_approval': True
					}
				]
			}
			
			# Add sequences
			self.add_dunning_sequence(standard_sequence)
			self.add_dunning_sequence(vip_sequence)
			
			# Initialize default templates
			await self._initialize_default_templates()
			
			self.logger.info("✅ Default dunning sequences and templates initialized")
		
		except Exception as e:
			self.logger.error(f"Failed to initialize default sequences: {e}")
	
	async def _initialize_default_templates(self) -> None:
		"""Initialize default email templates"""
		templates = [
			{
				'name': 'Grace Period Reminder',
				'stage': DunningStage.GRACE_PERIOD.value,
				'subject': 'Payment Reminder - Invoice {{invoice_number}}',
				'html_content': '''
				<h2>Payment Reminder</h2>
				<p>Dear {{customer_name}},</p>
				<p>This is a friendly reminder that payment for invoice {{invoice_number}} was due on {{due_date}}.</p>
				<p><strong>Amount Due: {{currency}} {{amount_due}}</strong></p>
				<p>Please make payment at your earliest convenience to avoid any service interruption.</p>
				<p>Thank you for your business!</p>
				'''
			},
			{
				'name': 'Gentle Reminder',
				'stage': DunningStage.GENTLE_REMINDER.value,
				'subject': 'Overdue Payment Notice - Invoice {{invoice_number}}',
				'html_content': '''
				<h2>Overdue Payment Notice</h2>
				<p>Dear {{customer_name}},</p>
				<p>Our records show that payment for invoice {{invoice_number}} is now overdue.</p>
				<p><strong>Amount Due: {{currency}} {{amount_due}}</strong></p>
				<p><strong>Days Overdue: {{days_overdue}}</strong></p>
				<p>Please remit payment immediately to avoid service disruption.</p>
				'''
			},
			{
				'name': 'Final Notice',
				'stage': DunningStage.FINAL_NOTICE.value,
				'subject': 'FINAL NOTICE - Account Suspension Pending',
				'html_content': '''
				<h2 style="color: red;">FINAL NOTICE</h2>
				<p>Dear {{customer_name}},</p>
				<p>This is your final notice regarding overdue invoice {{invoice_number}}.</p>
				<p><strong>Amount Due: {{currency}} {{amount_due}}</strong></p>
				<p><strong>Days Overdue: {{days_overdue}}</strong></p>
				<p style="color: red;"><strong>Your account will be suspended in 7 days if payment is not received.</strong></p>
				<p>Contact us immediately to resolve this matter.</p>
				'''
			}
		]
		
		for template_data in templates:
			template = DunningTemplate(template_data)
			self.templates[template.id] = template
	
	def add_dunning_sequence(self, sequence_data: Dict[str, Any]) -> DunningSequence:
		"""Add a new dunning sequence"""
		sequence = DunningSequence(sequence_data)
		self.sequences[sequence.id] = sequence
		self.logger.info(f"Added dunning sequence: {sequence.name}")
		return sequence
	
	def add_dunning_template(self, template_data: Dict[str, Any]) -> DunningTemplate:
		"""Add a new email template"""
		template = DunningTemplate(template_data)
		self.templates[template.id] = template
		self.logger.info(f"Added dunning template: {template.name}")
		return template
	
	async def create_dunning_case(self, customer_id: str, invoice_id: str = None, 
								 subscription_id: str = None, outstanding_amount: Decimal = None) -> DunningCase:
		"""Create a new dunning case"""
		try:
			# Determine appropriate sequence
			sequence = await self._select_dunning_sequence(customer_id, outstanding_amount)
			
			if not sequence:
				raise ValueError("No appropriate dunning sequence found")
			
			# Calculate outstanding amount if not provided
			if outstanding_amount is None and invoice_id:
				# Get real outstanding amount from billing service
				try:
					from .service import get_billing_service
					billing_service = get_billing_service()
					invoice = billing_service.invoices.get(invoice_id)
					if invoice:
						outstanding_amount = invoice.amount_due
					else:
						self.logger.warning(f"Invoice {invoice_id} not found for dunning case")
						outstanding_amount = Decimal('0')
				except Exception as e:
					self.logger.error(f"Failed to get outstanding amount for invoice {invoice_id}: {e}")
					outstanding_amount = Decimal('0')
			
			case_data = {
				'customer_id': customer_id,
				'invoice_id': invoice_id,
				'subscription_id': subscription_id,
				'sequence_id': sequence.id,
				'outstanding_amount': outstanding_amount or Decimal('0'),
				'stage': DunningStage.GRACE_PERIOD.value
			}
			
			case = DunningCase(case_data)
			self.cases[case.id] = case
			
			# Schedule first action
			await self._schedule_next_action(case)
			
			self.logger.info(f"Created dunning case {case.id} for customer {customer_id}")
			return case
		
		except Exception as e:
			self.logger.error(f"Failed to create dunning case: {e}")
			raise
	
	async def _select_dunning_sequence(self, customer_id: str, amount: Decimal = None) -> Optional[DunningSequence]:
		"""Select appropriate dunning sequence for customer"""
		# Get real customer segment from billing service
		customer_segment = 'standard'  # Default fallback
		
		try:
			from .service import get_billing_service
			billing_service = get_billing_service()
			customer = billing_service.customers.get(customer_id)
			if customer:
				# Check customer tier/segment
				customer_segment = getattr(customer, 'tier', 'standard')
				
				# Adjust segment based on customer value
				if customer_segment == 'standard':
					# Check if customer should be VIP based on revenue
					customer_subscriptions = [
						sub for sub in billing_service.subscriptions.values()
						if sub.customer_id == customer_id
					]
					
					total_mrr = sum(
						getattr(sub, 'mrr', Decimal('0')) 
						for sub in customer_subscriptions
					)
					
					# Promote to VIP if high MRR
					if total_mrr >= Decimal('1000'):
						customer_segment = 'vip'
					elif total_mrr >= Decimal('500'):
						customer_segment = 'enterprise'
				
		except Exception as e:
			self.logger.warning(f"Failed to get customer segment for {customer_id}: {e}")
			customer_segment = 'standard'
		
		# Find matching sequence
		for sequence in self.sequences.values():
			if not sequence.active:
				continue
			
			# Check customer segment
			if sequence.customer_segments and customer_segment not in sequence.customer_segments:
				continue
			
			# Check amount thresholds
			if amount and sequence.amount_thresholds:
				min_amount = sequence.amount_thresholds.get('min')
				max_amount = sequence.amount_thresholds.get('max')
				
				if min_amount and amount < min_amount:
					continue
				if max_amount and amount > max_amount:
					continue
			
			return sequence
		
		# Return first active sequence as fallback
		active_sequences = [s for s in self.sequences.values() if s.active]
		return active_sequences[0] if active_sequences else None
	
	async def _schedule_next_action(self, case: DunningCase) -> None:
		"""Schedule the next action for a dunning case"""
		try:
			sequence = self.sequences.get(case.sequence_id)
			if not sequence or case.current_step >= len(sequence.steps):
				return
			
			next_step = sequence.steps[case.current_step]
			
			# Calculate next action date
			next_action_date = case.created_at + timedelta(days=next_step.delay_days)
			
			# Adjust for business hours if needed
			if self.business_hours_only:
				next_action_date = self._adjust_for_business_hours(next_action_date)
			
			case.next_action_date = next_action_date
			
			# Add to processing queue if due
			if next_action_date <= datetime.utcnow():
				await self.processing_queue.put(case.id)
			
			self.logger.debug(f"Scheduled next action for case {case.id} at {next_action_date}")
		
		except Exception as e:
			self.logger.error(f"Failed to schedule next action: {e}")
	
	def _adjust_for_business_hours(self, target_date: datetime) -> datetime:
		"""Adjust date to fall within business hours"""
		# Skip weekends
		while target_date.weekday() in self.excluded_days:
			target_date += timedelta(days=1)
		
		# Adjust time to business hours
		if target_date.hour < self.business_hours['start']:
			target_date = target_date.replace(hour=self.business_hours['start'], minute=0, second=0)
		elif target_date.hour >= self.business_hours['end']:
			target_date = target_date.replace(hour=self.business_hours['start'], minute=0, second=0)
			target_date += timedelta(days=1)
			# Check if new day is weekend
			if target_date.weekday() in self.excluded_days:
				target_date = self._adjust_for_business_hours(target_date)
		
		return target_date
	
	async def _start_dunning_processor(self) -> None:
		"""Start background dunning processor"""
		while True:
			try:
				# Wait for case to process
				case_id = await self.processing_queue.get()
				await self._process_dunning_case(case_id)
				self.processing_queue.task_done()
			except Exception as e:
				self.logger.error(f"Dunning processor error: {e}")
				await asyncio.sleep(1)
	
	async def _start_case_monitor(self) -> None:
		"""Monitor cases for due actions"""
		while True:
			try:
				now = datetime.utcnow()
				
				# Check all active cases
				for case in self.cases.values():
					if (not case.resolved_at and not case.paused and 
						case.next_action_date <= now):
						await self.processing_queue.put(case.id)
				
				# Sleep for 5 minutes
				await asyncio.sleep(300)
			
			except Exception as e:
				self.logger.error(f"Case monitor error: {e}")
				await asyncio.sleep(60)
	
	async def _process_dunning_case(self, case_id: str) -> None:
		"""Process a dunning case"""
		try:
			case = self.cases.get(case_id)
			if not case or case.resolved_at or case.paused:
				return
			
			sequence = self.sequences.get(case.sequence_id)
			if not sequence or case.current_step >= len(sequence.steps):
				return
			
			step = sequence.steps[case.current_step]
			
			# Check if action requires approval
			if step.requires_approval:
				await self._request_approval(case, step)
				return
			
			# Execute action if auto-execution enabled
			if step.auto_execute and self.auto_execution_enabled:
				await self._execute_dunning_action(case, step)
			
			# Advance to next step
			case.current_step += 1
			case.stage = step.stage
			
			# Schedule next action if more steps
			if case.current_step < len(sequence.steps):
				await self._schedule_next_action(case)
			
			self.logger.info(f"Processed dunning case {case_id}, step {case.current_step}")
		
		except Exception as e:
			self.logger.error(f"Failed to process dunning case {case_id}: {e}")
	
	async def _execute_dunning_action(self, case: DunningCase, step: DunningStep) -> None:
		"""Execute a dunning action"""
		try:
			action_record = {
				'case_id': case.id,
				'action_type': step.action.value,
				'executed_by': 'system',
				'details': {'step_id': step.id}
			}
			
			if step.action == DunningAction.EMAIL_REMINDER:
				result = await self._send_dunning_email(case, step)
				action_record['result'] = 'success' if result else 'failed'
			
			elif step.action == DunningAction.PAYMENT_RETRY:
				result = await self._retry_payment(case)
				action_record['result'] = 'success' if result else 'failed'
			
			elif step.action == DunningAction.SUBSCRIPTION_PAUSE:
				result = await self._pause_subscription(case)
				action_record['result'] = 'success' if result else 'failed'
			
			elif step.action == DunningAction.ACCOUNT_SUSPENSION:
				result = await self._suspend_account(case)
				action_record['result'] = 'success' if result else 'failed'
			
			else:
				action_record['result'] = 'pending'
				action_record['details']['note'] = 'Manual action required'
			
			# Record action
			action = DunningAction(action_record)
			self.actions[action.id] = action
			case.actions_taken.append(action.id)
			
		except Exception as e:
			self.logger.error(f"Failed to execute dunning action: {e}")
	
	async def _send_dunning_email(self, case: DunningCase, step: DunningStep) -> bool:
		"""Send dunning email"""
		try:
			# Get template
			template = None
			if step.template_id:
				template = self.templates.get(step.template_id)
			else:
				# Find template by stage
				for t in self.templates.values():
					if t.stage == step.stage:
						template = t
						break
			
			if not template:
				self.logger.warning(f"No template found for dunning step {step.id}")
				return False
			
			# Get customer and invoice data for template variables
			customer_data, invoice_data = await self._get_email_template_data(case)
			
			if not customer_data:
				self.logger.error(f"Could not get customer data for case {case.id}")
				return False
			
			# Render template with data
			rendered_email = await self._render_email_template(template, customer_data, invoice_data, case)
			
			# Send email using real email service
			return await self._send_email_via_service(customer_data['email'], rendered_email, case)
		
		except Exception as e:
			self.logger.error(f"Failed to send dunning email: {e}")
			return False
	
	async def _get_email_template_data(self, case: DunningCase) -> Tuple[Optional[Dict], Optional[Dict]]:
		"""Get customer and invoice data for email templates"""
		try:
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			# Get customer data
			customer = billing_service.customers.get(case.customer_id)
			if not customer:
				return None, None
			
			customer_data = {
				'customer_id': customer.id,
				'customer_name': getattr(customer, 'name', 'Valued Customer'),
				'email': getattr(customer, 'email', ''),
				'company': getattr(customer, 'company', ''),
				'tier': getattr(customer, 'tier', 'standard')
			}
			
			# Get invoice data if available
			invoice_data = None
			if case.invoice_id:
				invoice = billing_service.invoices.get(case.invoice_id)
				if invoice:
					invoice_data = {
						'invoice_number': getattr(invoice, 'invoice_number', case.invoice_id),
						'amount_due': str(case.outstanding_amount),
						'currency': case.currency,
						'due_date': invoice.due_date.strftime('%Y-%m-%d') if hasattr(invoice, 'due_date') and invoice.due_date else 'N/A',
						'invoice_date': invoice.invoice_date.strftime('%Y-%m-%d') if hasattr(invoice, 'invoice_date') else 'N/A',
						'days_overdue': (datetime.utcnow().date() - invoice.due_date.date()).days if hasattr(invoice, 'due_date') and invoice.due_date else 0
					}
			
			return customer_data, invoice_data
			
		except Exception as e:
			self.logger.error(f"Failed to get email template data: {e}")
			return None, None
	
	async def _render_email_template(self, template: DunningTemplate, customer_data: Dict, 
									invoice_data: Optional[Dict], case: DunningCase) -> Dict[str, str]:
		"""Render email template with customer and invoice data"""
		try:
			# Prepare template variables
			template_vars = {
				'customer_name': customer_data.get('customer_name', 'Valued Customer'),
				'customer_id': customer_data.get('customer_id', ''),
				'company': customer_data.get('company', ''),
				'outstanding_amount': str(case.outstanding_amount),
				'currency': case.currency,
				'case_id': case.id,
				'stage': case.stage.value,
				'created_date': case.created_at.strftime('%Y-%m-%d')
			}
			
			# Add invoice-specific variables if available
			if invoice_data:
				template_vars.update(invoice_data)
			else:
				# Fallback values for when no invoice is available
				template_vars.update({
					'invoice_number': 'N/A',
					'amount_due': str(case.outstanding_amount),
					'due_date': 'N/A',
					'days_overdue': 0
				})
			
			# Simple template variable substitution
			rendered_subject = template.subject
			rendered_html = template.html_content
			rendered_text = template.text_content or self._html_to_text(template.html_content)
			
			# Replace template variables
			for var_name, var_value in template_vars.items():
				placeholder = f"{{{{{var_name}}}}}"
				rendered_subject = rendered_subject.replace(placeholder, str(var_value))
				rendered_html = rendered_html.replace(placeholder, str(var_value))
				rendered_text = rendered_text.replace(placeholder, str(var_value))
			
			return {
				'subject': rendered_subject,
				'html_content': rendered_html,
				'text_content': rendered_text
			}
			
		except Exception as e:
			self.logger.error(f"Template rendering failed: {e}")
			return {
				'subject': f"Payment Reminder - Account {case.customer_id}",
				'html_content': f"<p>Please settle your outstanding balance of {case.currency} {case.outstanding_amount}</p>",
				'text_content': f"Please settle your outstanding balance of {case.currency} {case.outstanding_amount}"
			}
	
	def _html_to_text(self, html_content: str) -> str:
		"""Convert HTML to plain text"""
		try:
			import re
			# Remove HTML tags
			text = re.sub(r'<[^>]+>', '', html_content)
			# Clean up whitespace
			text = re.sub(r'\s+', ' ', text).strip()
			return text
		except Exception:
			return html_content
	
	async def _send_email_via_service(self, recipient_email: str, rendered_email: Dict[str, str], case: DunningCase) -> bool:
		"""Send email using real email service"""
		try:
			import os
			import aiohttp
			import base64
			
			# Try SendGrid first
			sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
			if sendgrid_api_key:
				return await self._send_via_sendgrid(sendgrid_api_key, recipient_email, rendered_email, case)
			
			# Try Amazon SES
			aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
			aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
			aws_region = os.getenv('AWS_REGION', 'us-east-1')
			
			if aws_access_key and aws_secret_key:
				return await self._send_via_ses(aws_access_key, aws_secret_key, aws_region, recipient_email, rendered_email, case)
			
			# Try SMTP as fallback
			smtp_host = os.getenv('SMTP_HOST')
			smtp_port = int(os.getenv('SMTP_PORT', '587'))
			smtp_user = os.getenv('SMTP_USER')
			smtp_password = os.getenv('SMTP_PASSWORD')
			
			if smtp_host and smtp_user and smtp_password:
				return await self._send_via_smtp(smtp_host, smtp_port, smtp_user, smtp_password, recipient_email, rendered_email, case)
			
			# Log if no email service is configured
			self.logger.warning("No email service configured, simulating email send")
			self.logger.info(f"SIMULATED EMAIL SEND - To: {recipient_email}, Subject: {rendered_email['subject']}")
			return True
			
		except Exception as e:
			self.logger.error(f"Email sending failed: {e}")
			return False
	
	async def _send_via_sendgrid(self, api_key: str, recipient_email: str, rendered_email: Dict[str, str], case: DunningCase) -> bool:
		"""Send email via SendGrid"""
		try:
			import aiohttp
			
			url = "https://api.sendgrid.com/v3/mail/send"
			
			from_email = os.getenv('SENDGRID_FROM_EMAIL', 'noreply@datacraft.co.ke')
			from_name = os.getenv('SENDGRID_FROM_NAME', 'APG Billing')
			
			payload = {
				"personalizations": [{
					"to": [{"email": recipient_email}],
					"subject": rendered_email['subject']
				}],
				"from": {
					"email": from_email,
					"name": from_name
				},
				"content": [
					{
						"type": "text/plain",
						"value": rendered_email['text_content']
					},
					{
						"type": "text/html",
						"value": rendered_email['html_content']
					}
				],
				"custom_args": {
					"dunning_case_id": case.id,
					"customer_id": case.customer_id,
					"stage": case.stage.value
				}
			}
			
			headers = {
				"Authorization": f"Bearer {api_key}",
				"Content-Type": "application/json"
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(url, json=payload, headers=headers) as response:
					if response.status == 202:
						self.logger.info(f"Dunning email sent via SendGrid for case {case.id}")
						return True
					else:
						error_text = await response.text()
						self.logger.error(f"SendGrid email failed: {response.status} - {error_text}")
						return False
		
		except Exception as e:
			self.logger.error(f"SendGrid email sending failed: {e}")
			return False
	
	async def _send_via_ses(self, access_key: str, secret_key: str, region: str, recipient_email: str, 
						   rendered_email: Dict[str, str], case: DunningCase) -> bool:
		"""Send email via Amazon SES"""
		try:
			import boto3
			from botocore.exceptions import ClientError
			
			# Create SES client
			ses_client = boto3.client(
				'ses',
				aws_access_key_id=access_key,
				aws_secret_access_key=secret_key,
				region_name=region
			)
			
			from_email = os.getenv('SES_FROM_EMAIL', 'noreply@datacraft.co.ke')
			
			# Send email
			response = ses_client.send_email(
				Destination={'ToAddresses': [recipient_email]},
				Message={
					'Body': {
						'Html': {'Charset': 'UTF-8', 'Data': rendered_email['html_content']},
						'Text': {'Charset': 'UTF-8', 'Data': rendered_email['text_content']}
					},
					'Subject': {'Charset': 'UTF-8', 'Data': rendered_email['subject']}
				},
				Source=from_email,
				Tags=[
					{'Name': 'DunningCaseId', 'Value': case.id},
					{'Name': 'CustomerId', 'Value': case.customer_id},
					{'Name': 'Stage', 'Value': case.stage.value}
				]
			)
			
			self.logger.info(f"Dunning email sent via SES for case {case.id}, MessageId: {response['MessageId']}")
			return True
			
		except ClientError as e:
			self.logger.error(f"SES email sending failed: {e}")
			return False
		except Exception as e:
			self.logger.error(f"SES email sending failed: {e}")
			return False
	
	async def _send_via_smtp(self, smtp_host: str, smtp_port: int, smtp_user: str, smtp_password: str,
							recipient_email: str, rendered_email: Dict[str, str], case: DunningCase) -> bool:
		"""Send email via SMTP"""
		try:
			import aiosmtplib
			from email.mime.text import MIMEText
			from email.mime.multipart import MIMEMultipart
			
			# Create message
			message = MIMEMultipart('alternative')
			message['Subject'] = rendered_email['subject']
			message['From'] = smtp_user
			message['To'] = recipient_email
			
			# Add custom headers
			message['X-Dunning-Case-Id'] = case.id
			message['X-Customer-Id'] = case.customer_id
			message['X-Dunning-Stage'] = case.stage.value
			
			# Add content
			text_part = MIMEText(rendered_email['text_content'], 'plain')
			html_part = MIMEText(rendered_email['html_content'], 'html')
			
			message.attach(text_part)
			message.attach(html_part)
			
			# Send email
			await aiosmtplib.send(
				message,
				hostname=smtp_host,
				port=smtp_port,
				start_tls=True,
				username=smtp_user,
				password=smtp_password
			)
			
			self.logger.info(f"Dunning email sent via SMTP for case {case.id}")
			return True
			
		except Exception as e:
			self.logger.error(f"SMTP email sending failed: {e}")
			return False
	
	async def _retry_payment(self, case: DunningCase) -> bool:
		"""Retry payment for case"""
		try:
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			# Get the invoice and customer
			invoice = billing_service.invoices.get(case.invoice_id) if case.invoice_id else None
			customer = billing_service.customers.get(case.customer_id)
			
			if not customer:
				self.logger.error(f"Customer {case.customer_id} not found for payment retry")
				return False
			
			# Find the customer's payment methods
			payment_methods = await self._get_customer_payment_methods(customer)
			
			if not payment_methods:
				self.logger.warning(f"No payment methods found for customer {case.customer_id}")
				return False
			
			# Try each payment method until one succeeds
			for payment_method in payment_methods:
				try:
					# Attempt payment with this method
					payment_result = await self._attempt_payment_with_method(
						case, customer, invoice, payment_method
					)
					
					if payment_result['success']:
						# Payment succeeded - update case and invoice
						await self._handle_successful_payment_retry(case, payment_result)
						self.logger.info(f"Payment retry successful for case {case.id} using {payment_method['type']}")
						return True
					else:
						# Log failure and try next method
						self.logger.warning(f"Payment retry failed with {payment_method['type']}: {payment_result.get('error')}")
						
						# Update payment method if it's invalid
						if payment_result.get('payment_method_invalid'):
							await self._mark_payment_method_invalid(customer, payment_method)
				
				except Exception as method_error:
					self.logger.error(f"Payment method {payment_method['type']} failed: {method_error}")
					continue
			
			# All payment methods failed
			self.logger.error(f"All payment methods failed for case {case.id}")
			await self._handle_failed_payment_retry(case)
			return False
			
		except Exception as e:
			self.logger.error(f"Payment retry failed: {e}")
			return False
	
	async def _get_customer_payment_methods(self, customer) -> List[Dict[str, Any]]:
		"""Get customer's available payment methods"""
		try:
			payment_methods = []
			
			# Get payment methods from customer metadata
			if hasattr(customer, 'metadata') and customer.metadata:
				stored_methods = customer.metadata.get('payment_methods', [])
				
				for method in stored_methods:
					if method.get('active', True) and not method.get('invalid', False):
						payment_methods.append(method)
			
			# Sort by preference (primary first, then by creation date)
			payment_methods.sort(key=lambda m: (not m.get('is_primary', False), m.get('created_at', '')))
			
			return payment_methods
			
		except Exception as e:
			self.logger.error(f"Failed to get payment methods: {e}")
			return []
	
	async def _attempt_payment_with_method(self, case: DunningCase, customer, invoice, payment_method: Dict[str, Any]) -> Dict[str, Any]:
		"""Attempt payment with a specific payment method"""
		try:
			from .models import BLPayment, PaymentStatus
			from .payment_processors import get_payment_processor_manager
			
			payment_manager = get_payment_processor_manager()
			
			# Prepare payment data
			payment_data = {
				'customer_id': case.customer_id,
				'amount': case.outstanding_amount,
				'currency': case.currency,
				'payment_method': payment_method['type'],
				'invoice_id': case.invoice_id,
				'description': f'Dunning retry for case {case.id}',
				'metadata': {
					'dunning_case_id': case.id,
					'retry_attempt': True,
					'payment_method_id': payment_method.get('id'),
					'original_due_date': invoice.due_date.isoformat() if invoice and hasattr(invoice, 'due_date') else None
				}
			}
			
			# Add payment method specific data
			if payment_method['type'] == 'stripe':
				payment_data['stripe_payment_method_id'] = payment_method.get('stripe_payment_method_id')
				payment_data['stripe_customer_id'] = payment_method.get('stripe_customer_id')
			elif payment_method['type'] == 'paypal':
				payment_data['paypal_billing_agreement_id'] = payment_method.get('paypal_billing_agreement_id')
			elif payment_method['type'].startswith('card'):
				payment_data['card_token'] = payment_method.get('card_token')
				payment_data['gateway_customer_id'] = payment_method.get('gateway_customer_id')
			
			# Process payment through payment manager
			payment_result = await payment_manager.process_payment(payment_data)
			
			if payment_result.get('success'):
				# Create payment record
				payment_record = BLPayment({
					'id': payment_result.get('payment_id', uuid7str()),
					'customer_id': case.customer_id,
					'amount': case.outstanding_amount,
					'currency': case.currency,
					'payment_method': payment_method['type'],
					'status': PaymentStatus.COMPLETED.value,
					'transaction_type': 'dunning_retry',
					'invoice_id': case.invoice_id,
					'created_at': datetime.utcnow().isoformat(),
					'metadata': {
						'dunning_case_id': case.id,
						'payment_method_id': payment_method.get('id'),
						'processor_response': payment_result.get('processor_response'),
						'retry_attempt': True
					}
				})
				
				# Store payment record
				from .service import get_billing_service
				billing_service = get_billing_service()
				billing_service.payments[payment_record.id] = payment_record
				
				return {
					'success': True,
					'payment_id': payment_record.id,
					'payment_method': payment_method,
					'amount_paid': case.outstanding_amount
				}
			else:
				# Check if payment method is invalid
				error_code = payment_result.get('error_code', '')
				payment_method_invalid = error_code in [
					'card_declined_permanently', 'invalid_payment_method', 
					'payment_method_unavailable', 'card_expired'
				]
				
				return {
					'success': False,
					'error': payment_result.get('error', 'Payment failed'),
					'error_code': error_code,
					'payment_method_invalid': payment_method_invalid,
					'processor_response': payment_result.get('processor_response')
				}
			
		except Exception as e:
			self.logger.error(f"Payment attempt failed: {e}")
			return {
				'success': False,
				'error': str(e),
				'payment_method_invalid': False
			}
	
	async def _handle_successful_payment_retry(self, case: DunningCase, payment_result: Dict[str, Any]) -> None:
		"""Handle successful payment retry"""
		try:
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			# Update invoice if exists
			if case.invoice_id:
				invoice = billing_service.invoices.get(case.invoice_id)
				if invoice:
					# Mark invoice as paid
					from .models import InvoiceStatus
					invoice.status = InvoiceStatus.PAID
					invoice.amount_due = max(Decimal('0'), invoice.amount_due - payment_result['amount_paid'])
					invoice.paid_at = datetime.utcnow()
					
					# Add payment metadata
					if not invoice.metadata:
						invoice.metadata = {}
					invoice.metadata['dunning_retry_successful'] = {
						'case_id': case.id,
						'payment_id': payment_result['payment_id'],
						'retry_date': datetime.utcnow().isoformat(),
						'payment_method': payment_result['payment_method']['type']
					}
			
			# Resolve the dunning case
			await self.resolve_case(case.id, 'paid', f"Payment successful via {payment_result['payment_method']['type']}")
			
			# Send success notification
			await self._send_payment_success_notification(case, payment_result)
			
		except Exception as e:
			self.logger.error(f"Failed to handle successful payment retry: {e}")
	
	async def _handle_failed_payment_retry(self, case: DunningCase) -> None:
		"""Handle failed payment retry"""
		try:
			# Update case metadata
			if not case.metadata:
				case.metadata = {}
			
			retry_attempts = case.metadata.get('retry_attempts', 0) + 1
			case.metadata['retry_attempts'] = retry_attempts
			case.metadata['last_retry_attempt'] = datetime.utcnow().isoformat()
			case.metadata['all_payment_methods_failed'] = True
			
			# Log the failure
			case.notes.append({
				'timestamp': datetime.utcnow().isoformat(),
				'author': 'system',
				'content': f'Payment retry failed - all payment methods exhausted (attempt #{retry_attempts})'
			})
			
			# Schedule more aggressive action if too many failures
			if retry_attempts >= 3:
				case.notes.append({
					'timestamp': datetime.utcnow().isoformat(),
					'author': 'system',
					'content': 'Escalating case due to repeated payment failures'
				})
				
				# Skip to more aggressive dunning stage
				case.stage = DunningStage.URGENT_NOTICE
			
		except Exception as e:
			self.logger.error(f"Failed to handle payment retry failure: {e}")
	
	async def _mark_payment_method_invalid(self, customer, payment_method: Dict[str, Any]) -> None:
		"""Mark a payment method as invalid"""
		try:
			if hasattr(customer, 'metadata') and customer.metadata:
				stored_methods = customer.metadata.get('payment_methods', [])
				
				for method in stored_methods:
					if method.get('id') == payment_method.get('id'):
						method['invalid'] = True
						method['invalid_date'] = datetime.utcnow().isoformat()
						method['active'] = False
						break
				
				customer.metadata['payment_methods'] = stored_methods
			
			self.logger.info(f"Marked payment method {payment_method.get('id')} as invalid for customer {customer.id}")
			
		except Exception as e:
			self.logger.error(f"Failed to mark payment method invalid: {e}")
	
	async def _send_payment_success_notification(self, case: DunningCase, payment_result: Dict[str, Any]) -> None:
		"""Send notification for successful payment"""
		try:
			from .service import get_billing_service
			billing_service = get_billing_service()
			
			customer = billing_service.customers.get(case.customer_id)
			if not customer or not hasattr(customer, 'email'):
				return
			
			# Simple success email
			success_email = {
				'subject': f'Payment Received - Thank You',
				'html_content': f'''
				<h2>Payment Received</h2>
				<p>Dear {getattr(customer, "name", "Valued Customer")},</p>
				<p>Thank you! We have successfully processed your payment.</p>
				<p><strong>Amount Paid:</strong> {case.currency} {payment_result["amount_paid"]}</p>
				<p><strong>Payment Method:</strong> {payment_result["payment_method"]["type"].title()}</p>
				<p>Your account is now current. Thank you for your business!</p>
				''',
				'text_content': f'''
				Payment Received - Thank You
				
				Dear {getattr(customer, "name", "Valued Customer")},
				
				Thank you! We have successfully processed your payment.
				Amount Paid: {case.currency} {payment_result["amount_paid"]}
				Payment Method: {payment_result["payment_method"]["type"].title()}
				
				Your account is now current. Thank you for your business!
				'''
			}
			
			await self._send_email_via_service(customer.email, success_email, case)
			
		except Exception as e:
			self.logger.error(f"Failed to send payment success notification: {e}")
	
	async def _pause_subscription(self, case: DunningCase) -> bool:
		"""Pause subscription"""
		try:
			# In production, would pause subscription in billing service
			self.logger.info(f"Pausing subscription for case {case.id}")
			return True
		except Exception as e:
			self.logger.error(f"Subscription pause failed: {e}")
			return False
	
	async def _suspend_account(self, case: DunningCase) -> bool:
		"""Suspend customer account"""
		try:
			# In production, would suspend account
			self.logger.info(f"Suspending account for case {case.id}")
			return True
		except Exception as e:
			self.logger.error(f"Account suspension failed: {e}")
			return False
	
	async def _request_approval(self, case: DunningCase, step: DunningStep) -> None:
		"""Request approval for manual action"""
		try:
			# In production, would create approval request
			self.logger.info(f"Approval requested for case {case.id}, action {step.action.value}")
			
			# Pause case pending approval
			case.paused = True
			case.pause_reason = f"Pending approval for {step.action.value}"
		
		except Exception as e:
			self.logger.error(f"Failed to request approval: {e}")
	
	async def resolve_case(self, case_id: str, resolution_type: str, notes: str = None) -> bool:
		"""Resolve a dunning case"""
		try:
			case = self.cases.get(case_id)
			if not case:
				return False
			
			case.resolved_at = datetime.utcnow()
			case.resolution_type = resolution_type
			case.stage = DunningStage.RESOLVED
			
			if notes:
				case.notes.append({
					'timestamp': datetime.utcnow().isoformat(),
					'author': 'system',
					'content': notes
				})
			
			self.logger.info(f"Resolved dunning case {case_id}: {resolution_type}")
			return True
		
		except Exception as e:
			self.logger.error(f"Failed to resolve case {case_id}: {e}")
			return False
	
	async def get_dunning_analytics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Get dunning analytics for period"""
		try:
			# Filter cases by date range
			period_cases = [
				case for case in self.cases.values()
				if start_date <= case.created_at <= end_date
			]
			
			# Calculate metrics
			total_cases = len(period_cases)
			resolved_cases = len([c for c in period_cases if c.resolved_at])
			active_cases = total_cases - resolved_cases
			
			# Resolution breakdown
			resolution_types = {}
			for case in period_cases:
				if case.resolution_type:
					resolution_types[case.resolution_type] = resolution_types.get(case.resolution_type, 0) + 1
			
			# Stage distribution
			stage_distribution = {}
			for case in period_cases:
				stage = case.stage.value
				stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
			
			# Recovery metrics
			total_outstanding = sum(case.outstanding_amount for case in period_cases)
			recovered_amount = sum(
				case.outstanding_amount for case in period_cases 
				if case.resolution_type == 'paid'
			)
			recovery_rate = (recovered_amount / total_outstanding * 100) if total_outstanding > 0 else 0
			
			return {
				'period_start': start_date.isoformat(),
				'period_end': end_date.isoformat(),
				'total_cases': total_cases,
				'active_cases': active_cases,
				'resolved_cases': resolved_cases,
				'resolution_rate': (resolved_cases / total_cases * 100) if total_cases > 0 else 0,
				'recovery_rate': float(recovery_rate),
				'total_outstanding': str(total_outstanding),
				'recovered_amount': str(recovered_amount),
				'resolution_types': resolution_types,
				'stage_distribution': stage_distribution
			}
		
		except Exception as e:
			self.logger.error(f"Dunning analytics failed: {e}")
			raise


# Global dunning management system
_dunning_system_instance: Optional[DunningManagementSystem] = None

def get_dunning_management_system() -> DunningManagementSystem:
	"""Get global dunning management system instance"""
	global _dunning_system_instance
	if _dunning_system_instance is None:
		_dunning_system_instance = DunningManagementSystem()
	return _dunning_system_instance


__all__ = [
	'DunningManagementSystem',
	'DunningCase',
	'DunningSequence',
	'DunningStep',
	'DunningTemplate',
	'DunningAction',
	'DunningStage',
	'get_dunning_management_system'
]