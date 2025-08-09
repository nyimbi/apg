"""
APG Real-Time Billing Dispute Resolution System

AI-powered dispute prevention, automated resolution, and intelligent case management
that resolves billing disputes before they escalate, maintains customer satisfaction,
and prevents revenue loss through proactive intervention.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from uuid_extensions import uuid7str
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .models import BLCustomer, BLInvoice, BLPayment, BLSubscription
from .service import get_billing_service
from .audit_compliance import get_audit_compliance_system, AuditEventType


class DisputeType(Enum):
	"""Types of billing disputes"""
	BILLING_ERROR = "billing_error"
	UNAUTHORIZED_CHARGE = "unauthorized_charge"
	SERVICE_NOT_RECEIVED = "service_not_received"
	DUPLICATE_CHARGE = "duplicate_charge"
	AMOUNT_INCORRECT = "amount_incorrect"
	SUBSCRIPTION_CANCELLED = "subscription_cancelled"
	REFUND_REQUEST = "refund_request"
	PRICING_DISAGREEMENT = "pricing_disagreement"
	USAGE_DISPUTE = "usage_dispute"
	TAX_DISPUTE = "tax_dispute"


class DisputeStatus(Enum):
	"""Dispute resolution status"""
	POTENTIAL = "potential"  # AI predicted potential dispute
	SUBMITTED = "submitted"
	INVESTIGATING = "investigating"
	EVIDENCE_GATHERING = "evidence_gathering"
	PENDING_CUSTOMER = "pending_customer"
	PENDING_INTERNAL = "pending_internal"
	RESOLVED_FAVOR_CUSTOMER = "resolved_favor_customer"
	RESOLVED_FAVOR_MERCHANT = "resolved_favor_merchant"
	ESCALATED = "escalated"
	CLOSED = "closed"


class ResolutionAction(Enum):
	"""Types of resolution actions"""
	FULL_REFUND = "full_refund"
	PARTIAL_REFUND = "partial_refund"
	CREDIT_APPLIED = "credit_applied"
	INVOICE_ADJUSTMENT = "invoice_adjustment"
	SERVICE_EXTENSION = "service_extension"
	GOODWILL_GESTURE = "goodwill_gesture"
	NO_ACTION = "no_action"
	MANUAL_REVIEW = "manual_review"


class DisputeEvidence:
	"""Evidence collected for dispute resolution"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.dispute_id = data['dispute_id']
		self.evidence_type = data['evidence_type']  # transaction, communication, usage, etc.
		self.source = data['source']  # system, customer, manual
		self.content = data['content']
		self.confidence_score = data.get('confidence_score', 0.5)
		self.supporting_dispute = data.get('supporting_dispute', True)
		self.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
		self.metadata = data.get('metadata', {})


class BillingDispute:
	"""Individual billing dispute case"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.customer_id = data['customer_id']
		self.dispute_type = DisputeType(data['dispute_type'])
		self.status = DisputeStatus(data.get('status', DisputeStatus.SUBMITTED.value))
		self.priority = data.get('priority', 'medium')  # low, medium, high, urgent
		self.disputed_amount = Decimal(str(data.get('disputed_amount', 0)))
		self.currency = data.get('currency', 'USD')
		self.description = data.get('description', '')
		self.customer_claim = data.get('customer_claim', '')
		self.related_transactions = data.get('related_transactions', [])  # invoice_ids, payment_ids
		self.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
		self.target_resolution_date = self.created_at + timedelta(days=data.get('sla_days', 5))
		self.actual_resolution_date = datetime.fromisoformat(data['actual_resolution_date']) if data.get('actual_resolution_date') else None
		self.assigned_to = data.get('assigned_to')
		self.resolution_action = ResolutionAction(data['resolution_action']) if data.get('resolution_action') else None
		self.resolution_amount = Decimal(str(data.get('resolution_amount', 0)))
		self.resolution_notes = data.get('resolution_notes', '')
		self.customer_satisfaction = data.get('customer_satisfaction')  # 1-5 rating
		self.evidence_items = data.get('evidence_items', [])
		self.communication_log = data.get('communication_log', [])
		self.ai_recommendation = data.get('ai_recommendation', {})
		self.escalation_reason = data.get('escalation_reason')
		self.metadata = data.get('metadata', {})


class DisputeResolutionEngine:
	"""AI-powered billing dispute resolution engine"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.DisputeResolutionEngine")
		
		# AI models for dispute processing
		self.dispute_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
		self.resolution_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
		self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
		self.scaler = StandardScaler()
		
		# Data storage
		self.disputes: Dict[str, BillingDispute] = {}
		self.evidence: Dict[str, DisputeEvidence] = {}
		self.resolution_templates = {}
		
		# Performance tracking
		self.resolution_metrics = {}
		self.model_performance = {}
		
		# Configuration
		self.auto_resolution_threshold = 0.85  # Confidence threshold for automatic resolution
		self.escalation_threshold = 0.3  # Confidence threshold below which to escalate
		self.sla_hours = {
			DisputeType.UNAUTHORIZED_CHARGE: 24,
			DisputeType.BILLING_ERROR: 48,
			DisputeType.DUPLICATE_CHARGE: 24,
			DisputeType.SERVICE_NOT_RECEIVED: 72,
			DisputeType.AMOUNT_INCORRECT: 48,
			DisputeType.SUBSCRIPTION_CANCELLED: 48,
			DisputeType.REFUND_REQUEST: 72,
			DisputeType.PRICING_DISAGREEMENT: 120,
			DisputeType.USAGE_DISPUTE: 96,
			DisputeType.TAX_DISPUTE: 120
		}
		
		# Service integrations
		self.billing_service = get_billing_service()
		self.audit_system = get_audit_compliance_system()
		
		# Background processing
		asyncio.create_task(self._start_dispute_monitor())
		asyncio.create_task(self._start_prevention_system())
		asyncio.create_task(self._initialize_resolution_templates())
	
	async def _start_dispute_monitor(self) -> None:
		"""Monitor and process active disputes"""
		while True:
			try:
				await self._process_active_disputes()
				await self._check_sla_violations()
				await self._update_resolution_metrics()
				await asyncio.sleep(300)  # Check every 5 minutes
			except Exception as e:
				self.logger.error(f"Dispute monitor error: {e}")
				await asyncio.sleep(300)
	
	async def _start_prevention_system(self) -> None:
		"""Proactively detect potential disputes"""
		while True:
			try:
				await self._detect_potential_disputes()
				await asyncio.sleep(1800)  # Check every 30 minutes
			except Exception as e:
				self.logger.error(f"Prevention system error: {e}")
				await asyncio.sleep(1800)
	
	async def _initialize_resolution_templates(self) -> None:
		"""Initialize resolution templates and responses"""
		try:
			self.resolution_templates = {
				DisputeType.BILLING_ERROR: {
					'auto_resolution_threshold': 0.9,
					'typical_actions': [ResolutionAction.INVOICE_ADJUSTMENT, ResolutionAction.CREDIT_APPLIED],
					'response_template': 'We found an error in your billing and have corrected it.',
					'goodwill_threshold': 0.7
				},
				DisputeType.DUPLICATE_CHARGE: {
					'auto_resolution_threshold': 0.95,
					'typical_actions': [ResolutionAction.FULL_REFUND],
					'response_template': 'We identified a duplicate charge and have processed a full refund.',
					'goodwill_threshold': 0.8
				},
				DisputeType.UNAUTHORIZED_CHARGE: {
					'auto_resolution_threshold': 0.8,
					'typical_actions': [ResolutionAction.FULL_REFUND, ResolutionAction.MANUAL_REVIEW],
					'response_template': 'We are investigating this unauthorized charge and will resolve it promptly.',
					'goodwill_threshold': 0.6
				},
				DisputeType.SERVICE_NOT_RECEIVED: {
					'auto_resolution_threshold': 0.7,
					'typical_actions': [ResolutionAction.SERVICE_EXTENSION, ResolutionAction.PARTIAL_REFUND],
					'response_template': 'We apologize for the service interruption and will make this right.',
					'goodwill_threshold': 0.6
				},
				DisputeType.REFUND_REQUEST: {
					'auto_resolution_threshold': 0.6,
					'typical_actions': [ResolutionAction.PARTIAL_REFUND, ResolutionAction.CREDIT_APPLIED],
					'response_template': 'We have reviewed your refund request and processed the appropriate amount.',
					'goodwill_threshold': 0.5
				}
			}
			
			self.logger.info("Resolution templates initialized")
			
		except Exception as e:
			self.logger.error(f"Template initialization failed: {e}")
	
	async def _detect_potential_disputes(self) -> None:
		"""Proactively detect potential disputes before they are submitted"""
		try:
			# Analyze recent transactions for dispute patterns
			potential_disputes = []
			
			# Check for billing anomalies
			billing_anomalies = await self._detect_billing_anomalies()
			potential_disputes.extend(billing_anomalies)
			
			# Check for payment failures that might lead to disputes
			payment_failures = await self._detect_dispute_prone_failures()
			potential_disputes.extend(payment_failures)
			
			# Check for usage-billing mismatches
			usage_mismatches = await self._detect_usage_mismatches()
			potential_disputes.extend(usage_mismatches)
			
			# Create potential dispute records
			for potential in potential_disputes:
				await self._create_potential_dispute(potential)
			
			if potential_disputes:
				self.logger.info(f"Detected {len(potential_disputes)} potential disputes")
			
		except Exception as e:
			self.logger.error(f"Potential dispute detection failed: {e}")
	
	async def _detect_billing_anomalies(self) -> List[Dict[str, Any]]:
		"""Detect billing anomalies that could lead to disputes"""
		try:
			anomalies = []
			recent_invoices = [
				inv for inv in self.billing_service.invoices.values()
				if (datetime.utcnow() - inv.invoice_date).days <= 7
			]
			
			for invoice in recent_invoices:
				customer = self.billing_service.customers.get(invoice.customer_id)
				if not customer:
					continue
				
				# Check for unusual amount increases
				customer_invoices = [
					inv for inv in self.billing_service.invoices.values()
					if inv.customer_id == invoice.customer_id and inv.id != invoice.id
				]
				
				if customer_invoices:
					avg_amount = sum(inv.total for inv in customer_invoices[-5:]) / min(len(customer_invoices), 5)
					
					# Flag if current invoice is 50% higher than average
					if invoice.total > avg_amount * Decimal('1.5'):
						anomalies.append({
							'type': 'unusual_amount_increase',
							'customer_id': invoice.customer_id,
							'invoice_id': invoice.id,
							'current_amount': invoice.total,
							'average_amount': avg_amount,
							'confidence': 0.7
						})
				
				# Check for duplicate charges
				duplicate_invoices = [
					inv for inv in self.billing_service.invoices.values()
					if (inv.customer_id == invoice.customer_id and 
						inv.id != invoice.id and
						abs(inv.total - invoice.total) < Decimal('0.01') and
						(invoice.invoice_date - inv.invoice_date).days <= 1)
				]
				
				if duplicate_invoices:
					anomalies.append({
						'type': 'potential_duplicate',
						'customer_id': invoice.customer_id,
						'invoice_id': invoice.id,
						'duplicate_invoices': [inv.id for inv in duplicate_invoices],
						'confidence': 0.9
					})
			
			return anomalies
			
		except Exception as e:
			self.logger.error(f"Billing anomaly detection failed: {e}")
			return []
	
	async def _detect_dispute_prone_failures(self) -> List[Dict[str, Any]]:
		"""Detect payment failures that commonly lead to disputes"""
		try:
			prone_failures = []
			recent_failures = [
				payment for payment in self.billing_service.payments.values()
				if (payment.status.value == 'failed' and
					(datetime.utcnow() - payment.created_at).hours <= 24)
			]
			
			for payment in recent_failures:
				failure_reason = payment.metadata.get('failure_reason', 'unknown') if payment.metadata else 'unknown'
				
				# Certain failure reasons are more likely to lead to disputes
				dispute_prone_reasons = [
					'card_declined',
					'insufficient_funds',
					'fraud_suspected',
					'card_expired'
				]
				
				if failure_reason in dispute_prone_reasons:
					# Check customer's dispute history
					customer_disputes = [
						dispute for dispute in self.disputes.values()
						if dispute.customer_id == payment.customer_id
					]
					
					# Higher risk if customer has disputed before
					risk_multiplier = 1 + (len(customer_disputes) * 0.2)
					
					prone_failures.append({
						'type': 'dispute_prone_failure',
						'customer_id': payment.customer_id,
						'payment_id': payment.id,
						'failure_reason': failure_reason,
						'confidence': min(0.8 * risk_multiplier, 0.95)
					})
			
			return prone_failures
			
		except Exception as e:
			self.logger.error(f"Dispute-prone failure detection failed: {e}")
			return []
	
	async def _detect_usage_mismatches(self) -> List[Dict[str, Any]]:
		"""Detect usage-billing mismatches"""
		try:
			mismatches = []
			
			# Check recent invoices with usage components
			recent_invoices = [
				inv for inv in self.billing_service.invoices.values()
				if (datetime.utcnow() - inv.invoice_date).days <= 7 and
				   inv.metadata and inv.metadata.get('has_usage_charges')
			]
			
			for invoice in recent_invoices:
				# Get customer's usage records for billing period
				billing_start = invoice.invoice_date - timedelta(days=30)  # Approximate billing period
				billing_end = invoice.invoice_date
				
				customer_usage = [
					usage for usage in self.billing_service.usage_records
					if (usage.customer_id == invoice.customer_id and
						billing_start <= usage.timestamp <= billing_end)
				]
				
				if customer_usage:
					# Calculate total usage
					total_usage = sum(usage.quantity for usage in customer_usage)
					
					# Get billed usage from invoice metadata
					billed_usage = invoice.metadata.get('total_usage_billed', 0)
					
					# Flag significant discrepancies
					if abs(total_usage - billed_usage) > total_usage * 0.1:  # 10% variance
						mismatches.append({
							'type': 'usage_billing_mismatch',
							'customer_id': invoice.customer_id,
							'invoice_id': invoice.id,
							'actual_usage': total_usage,
							'billed_usage': billed_usage,
							'variance_percent': abs(total_usage - billed_usage) / max(total_usage, 1),
							'confidence': 0.8
						})
			
			return mismatches
			
		except Exception as e:
			self.logger.error(f"Usage mismatch detection failed: {e}")
			return []
	
	async def _create_potential_dispute(self, potential_data: Dict[str, Any]) -> None:
		"""Create a potential dispute record for proactive handling"""
		try:
			dispute_data = {
				'customer_id': potential_data['customer_id'],
				'dispute_type': self._map_anomaly_to_dispute_type(potential_data['type']),
				'status': DisputeStatus.POTENTIAL.value,
				'priority': 'medium',
				'description': f"Potential dispute detected: {potential_data['type']}",
				'ai_recommendation': {
					'detection_confidence': potential_data['confidence'],
					'anomaly_type': potential_data['type'],
					'suggested_action': self._suggest_proactive_action(potential_data),
					'prevention_opportunity': True
				},
				'metadata': {
					'detected_by': 'ai_prevention_system',
					'detection_data': potential_data
				}
			}
			
			dispute = BillingDispute(dispute_data)
			self.disputes[dispute.id] = dispute
			
			# Take proactive action if confidence is high
			if potential_data['confidence'] > 0.8:
				await self._take_proactive_action(dispute)
			
		except Exception as e:
			self.logger.error(f"Potential dispute creation failed: {e}")
	
	def _map_anomaly_to_dispute_type(self, anomaly_type: str) -> str:
		"""Map anomaly type to dispute type"""
		mapping = {
			'unusual_amount_increase': DisputeType.AMOUNT_INCORRECT.value,
			'potential_duplicate': DisputeType.DUPLICATE_CHARGE.value,
			'dispute_prone_failure': DisputeType.BILLING_ERROR.value,
			'usage_billing_mismatch': DisputeType.USAGE_DISPUTE.value
		}
		return mapping.get(anomaly_type, DisputeType.BILLING_ERROR.value)
	
	def _suggest_proactive_action(self, potential_data: Dict[str, Any]) -> str:
		"""Suggest proactive action for potential dispute"""
		action_map = {
			'unusual_amount_increase': 'send_explanation_email',
			'potential_duplicate': 'automatic_refund_review',
			'dispute_prone_failure': 'proactive_customer_contact',
			'usage_billing_mismatch': 'billing_verification_and_correction'
		}
		return action_map.get(potential_data['type'], 'monitor_closely')
	
	async def _take_proactive_action(self, dispute: BillingDispute) -> None:
		"""Take proactive action to prevent dispute escalation"""
		try:
			suggested_action = dispute.ai_recommendation.get('suggested_action')
			
			if suggested_action == 'send_explanation_email':
				await self._send_proactive_explanation(dispute)
			elif suggested_action == 'automatic_refund_review':
				await self._review_for_automatic_refund(dispute)
			elif suggested_action == 'proactive_customer_contact':
				await self._initiate_proactive_contact(dispute)
			elif suggested_action == 'billing_verification_and_correction':
				await self._verify_and_correct_billing(dispute)
			
			# Log proactive action
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.COMPLIANCE_CHECK.value,
				'user_id': 'system',
				'resource_type': 'dispute_prevention',
				'resource_id': dispute.id,
				'action': 'proactive_action_taken',
				'description': f'Proactive action: {suggested_action}',
				'metadata': {
					'dispute_type': dispute.dispute_type.value,
					'detection_confidence': dispute.ai_recommendation.get('detection_confidence'),
					'action_taken': suggested_action
				}
			})
			
		except Exception as e:
			self.logger.error(f"Proactive action failed: {e}")
	
	async def _send_proactive_explanation(self, dispute: BillingDispute) -> None:
		"""Send proactive explanation to customer"""
		try:
			# Get financial journey orchestrator for communication
			from .financial_journey_orchestration import get_financial_journey_orchestrator
			orchestrator = get_financial_journey_orchestrator()
			
			# Send explanation about billing changes
			context = {
				'dispute_id': dispute.id,
				'explanation_type': 'billing_increase',
				'proactive': True
			}
			
			await orchestrator._send_contextual_communication(
				dispute.customer_id, 'billing_explanation', context
			)
			
			self.logger.info(f"Sent proactive explanation for dispute {dispute.id}")
			
		except Exception as e:
			self.logger.error(f"Proactive explanation failed: {e}")
	
	async def _review_for_automatic_refund(self, dispute: BillingDispute) -> None:
		"""Review potential duplicate charge for automatic refund"""
		try:
			detection_data = dispute.metadata.get('detection_data', {})
			
			if detection_data.get('type') == 'potential_duplicate':
				# Automatically process refund for duplicate charges with high confidence
				if detection_data.get('confidence', 0) > 0.9:
					await self._process_automatic_resolution(
						dispute, ResolutionAction.FULL_REFUND, "Duplicate charge detected and refunded"
					)
			
		except Exception as e:
			self.logger.error(f"Automatic refund review failed: {e}")
	
	async def submit_dispute(self, customer_id: str, dispute_data: Dict[str, Any]) -> BillingDispute:
		"""Submit a new billing dispute"""
		try:
			# Enhance dispute data with AI analysis
			enhanced_data = await self._enhance_dispute_submission(customer_id, dispute_data)
			
			dispute = BillingDispute(enhanced_data)
			self.disputes[dispute.id] = dispute
			
			# Automatically gather evidence
			await self._auto_gather_evidence(dispute)
			
			# Get AI recommendation
			ai_recommendation = await self._generate_ai_recommendation(dispute)
			dispute.ai_recommendation = ai_recommendation
			
			# Check for automatic resolution
			if ai_recommendation.get('confidence', 0) > self.auto_resolution_threshold:
				await self._attempt_automatic_resolution(dispute)
			else:
				# Assign to appropriate handler
				await self._assign_dispute(dispute)
			
			# Log dispute submission
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.CUSTOMER_UPDATED.value,
				'user_id': customer_id,
				'resource_type': 'billing_dispute',
				'resource_id': dispute.id,
				'action': 'dispute_submitted',
				'description': f'Billing dispute submitted: {dispute.dispute_type.value}',
				'metadata': {
					'dispute_amount': str(dispute.disputed_amount),
					'ai_confidence': ai_recommendation.get('confidence', 0)
				}
			})
			
			self.logger.info(f"Dispute submitted: {dispute.id} for customer {customer_id}")
			return dispute
			
		except Exception as e:
			self.logger.error(f"Dispute submission failed: {e}")
			raise
	
	async def _enhance_dispute_submission(self, customer_id: str, dispute_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Enhance dispute submission with AI analysis"""
		try:
			enhanced_data = dispute_data.copy()
			enhanced_data['customer_id'] = customer_id
			
			# Classify dispute type if not provided
			if 'dispute_type' not in enhanced_data:
				predicted_type = await self._classify_dispute_type(dispute_data.get('description', ''))
				enhanced_data['dispute_type'] = predicted_type
			
			# Set priority based on amount and customer tier
			customer = self.billing_service.customers.get(customer_id)
			disputed_amount = Decimal(str(dispute_data.get('disputed_amount', 0)))
			
			if disputed_amount > Decimal('1000') or (customer and getattr(customer, 'tier', '') == 'vip'):
				enhanced_data['priority'] = 'high'
			elif disputed_amount > Decimal('100'):
				enhanced_data['priority'] = 'medium'
			else:
				enhanced_data['priority'] = 'low'
			
			# Set SLA based on dispute type
			dispute_type = DisputeType(enhanced_data['dispute_type'])
			sla_hours = self.sla_hours.get(dispute_type, 72)
			enhanced_data['sla_days'] = sla_hours // 24
			
			return enhanced_data
			
		except Exception as e:
			self.logger.error(f"Dispute enhancement failed: {e}")
			return dispute_data
	
	async def _classify_dispute_type(self, description: str) -> str:
		"""Use AI to classify dispute type from description"""
		try:
			# Simple keyword-based classification (in production would use trained ML model)
			description_lower = description.lower()
			
			if any(word in description_lower for word in ['duplicate', 'double', 'twice', 'charged again']):
				return DisputeType.DUPLICATE_CHARGE.value
			elif any(word in description_lower for word in ['unauthorized', 'didn\'t authorize', 'fraud']):
				return DisputeType.UNAUTHORIZED_CHARGE.value
			elif any(word in description_lower for word in ['wrong amount', 'incorrect', 'too much']):
				return DisputeType.AMOUNT_INCORRECT.value
			elif any(word in description_lower for word in ['didn\'t receive', 'no service', 'not delivered']):
				return DisputeType.SERVICE_NOT_RECEIVED.value
			elif any(word in description_lower for word in ['refund', 'cancel', 'return']):
				return DisputeType.REFUND_REQUEST.value
			elif any(word in description_lower for word in ['usage', 'meter', 'consumption']):
				return DisputeType.USAGE_DISPUTE.value
			else:
				return DisputeType.BILLING_ERROR.value
			
		except Exception as e:
			self.logger.error(f"Dispute classification failed: {e}")
			return DisputeType.BILLING_ERROR.value
	
	async def _auto_gather_evidence(self, dispute: BillingDispute) -> None:
		"""Automatically gather evidence for dispute"""
		try:
			evidence_items = []
			
			# Gather transaction evidence
			transaction_evidence = await self._gather_transaction_evidence(dispute)
			evidence_items.extend(transaction_evidence)
			
			# Gather usage evidence
			usage_evidence = await self._gather_usage_evidence(dispute)
			evidence_items.extend(usage_evidence)
			
			# Gather communication evidence
			communication_evidence = await self._gather_communication_evidence(dispute)
			evidence_items.extend(communication_evidence)
			
			# Store evidence
			for evidence_data in evidence_items:
				evidence = DisputeEvidence(evidence_data)
				self.evidence[evidence.id] = evidence
				dispute.evidence_items.append(evidence.id)
			
			self.logger.info(f"Gathered {len(evidence_items)} evidence items for dispute {dispute.id}")
			
		except Exception as e:
			self.logger.error(f"Evidence gathering failed: {e}")
	
	async def _gather_transaction_evidence(self, dispute: BillingDispute) -> List[Dict[str, Any]]:
		"""Gather transaction-related evidence"""
		try:
			evidence_items = []
			
			# Get customer's invoices and payments
			customer_invoices = [
				inv for inv in self.billing_service.invoices.values()
				if inv.customer_id == dispute.customer_id
			]
			
			customer_payments = [
				pay for pay in self.billing_service.payments.values()
				if pay.customer_id == dispute.customer_id
			]
			
			# Find relevant transactions
			relevant_invoices = []
			relevant_payments = []
			
			if dispute.related_transactions:
				for transaction_id in dispute.related_transactions:
					# Check if it's an invoice
					invoice = self.billing_service.invoices.get(transaction_id)
					if invoice:
						relevant_invoices.append(invoice)
					
					# Check if it's a payment
					payment = self.billing_service.payments.get(transaction_id)
					if payment:
						relevant_payments.append(payment)
			else:
				# Find transactions around dispute creation time
				dispute_window = timedelta(days=30)
				window_start = dispute.created_at - dispute_window
				window_end = dispute.created_at + dispute_window
				
				relevant_invoices = [
					inv for inv in customer_invoices
					if window_start <= inv.invoice_date <= window_end
				]
				
				relevant_payments = [
					pay for pay in customer_payments
					if window_start <= pay.created_at <= window_end
				]
			
			# Create evidence for transactions
			for invoice in relevant_invoices:
				evidence_items.append({
					'dispute_id': dispute.id,
					'evidence_type': 'transaction_record',
					'source': 'system',
					'content': {
						'type': 'invoice',
						'invoice_id': invoice.id,
						'amount': str(invoice.total),
						'date': invoice.invoice_date.isoformat(),
						'status': invoice.status.value if invoice.status else 'unknown',
						'items': invoice.line_items if hasattr(invoice, 'line_items') else []
					},
					'confidence_score': 0.9,
					'supporting_dispute': False  # Transaction records support merchant position
				})
			
			for payment in relevant_payments:
				evidence_items.append({
					'dispute_id': dispute.id,
					'evidence_type': 'payment_record',
					'source': 'system',
					'content': {
						'type': 'payment',
						'payment_id': payment.id,
						'amount': str(payment.amount),
						'date': payment.created_at.isoformat(),
						'status': payment.status.value,
						'method': payment.payment_method,
						'processor_response': payment.metadata.get('processor_response') if payment.metadata else None
					},
					'confidence_score': 0.9,
					'supporting_dispute': payment.status.value == 'failed'  # Failed payments support customer
				})
			
			return evidence_items
			
		except Exception as e:
			self.logger.error(f"Transaction evidence gathering failed: {e}")
			return []
	
	async def _gather_usage_evidence(self, dispute: BillingDispute) -> List[Dict[str, Any]]:
		"""Gather usage-related evidence"""
		try:
			evidence_items = []
			
			if dispute.dispute_type == DisputeType.USAGE_DISPUTE:
				# Get customer's usage records for relevant period
				billing_period_start = dispute.created_at - timedelta(days=45)  # Extended period
				billing_period_end = dispute.created_at
				
				customer_usage = [
					usage for usage in self.billing_service.usage_records
					if (usage.customer_id == dispute.customer_id and
						billing_period_start <= usage.timestamp <= billing_period_end)
				]
				
				if customer_usage:
					# Aggregate usage by metric
					usage_summary = {}
					for usage in customer_usage:
						metric = usage.metric_name
						if metric not in usage_summary:
							usage_summary[metric] = {'total': 0, 'count': 0, 'dates': []}
						
						usage_summary[metric]['total'] += usage.quantity
						usage_summary[metric]['count'] += 1
						usage_summary[metric]['dates'].append(usage.timestamp.isoformat())
					
					evidence_items.append({
						'dispute_id': dispute.id,
						'evidence_type': 'usage_records',
						'source': 'system',
						'content': {
							'period_start': billing_period_start.isoformat(),
							'period_end': billing_period_end.isoformat(),
							'usage_summary': usage_summary,
							'total_records': len(customer_usage)
						},
						'confidence_score': 0.95,
						'supporting_dispute': False  # Usage records typically support merchant
					})
			
			return evidence_items
			
		except Exception as e:
			self.logger.error(f"Usage evidence gathering failed: {e}")
			return []
	
	async def _gather_communication_evidence(self, dispute: BillingDispute) -> List[Dict[str, Any]]:
		"""Gather communication-related evidence"""
		try:
			evidence_items = []
			
			# Get customer metadata for communication history
			customer = self.billing_service.customers.get(dispute.customer_id)
			if customer and customer.metadata:
				# Check for support tickets or communications
				support_history = customer.metadata.get('support_history', [])
				
				for communication in support_history:
					if 'billing' in communication.get('subject', '').lower():
						evidence_items.append({
							'dispute_id': dispute.id,
							'evidence_type': 'customer_communication',
							'source': 'support_system',
							'content': {
								'date': communication.get('date'),
								'subject': communication.get('subject'),
								'summary': communication.get('summary'),
								'resolution': communication.get('resolution')
							},
							'confidence_score': 0.7,
							'supporting_dispute': 'complaint' in communication.get('subject', '').lower()
						})
			
			return evidence_items
			
		except Exception as e:
			self.logger.error(f"Communication evidence gathering failed: {e}")
			return []
	
	async def _generate_ai_recommendation(self, dispute: BillingDispute) -> Dict[str, Any]:
		"""Generate AI recommendation for dispute resolution"""
		try:
			# Analyze evidence
			evidence_analysis = await self._analyze_evidence(dispute)
			
			# Calculate confidence score
			confidence_score = self._calculate_resolution_confidence(dispute, evidence_analysis)
			
			# Determine recommended action
			recommended_action = self._determine_recommended_action(dispute, evidence_analysis, confidence_score)
			
			# Calculate recommended amount
			recommended_amount = self._calculate_recommended_amount(dispute, recommended_action, evidence_analysis)
			
			# Generate explanation
			explanation = self._generate_recommendation_explanation(dispute, evidence_analysis, recommended_action)
			
			return {
				'confidence': confidence_score,
				'recommended_action': recommended_action.value if recommended_action else None,
				'recommended_amount': str(recommended_amount),
				'explanation': explanation,
				'evidence_summary': evidence_analysis,
				'auto_resolvable': confidence_score > self.auto_resolution_threshold,
				'escalation_needed': confidence_score < self.escalation_threshold,
				'generated_at': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			self.logger.error(f"AI recommendation generation failed: {e}")
			return {'confidence': 0.0, 'explanation': 'Analysis failed'}
	
	async def _analyze_evidence(self, dispute: BillingDispute) -> Dict[str, Any]:
		"""Analyze collected evidence"""
		try:
			evidence_items = [self.evidence[eid] for eid in dispute.evidence_items if eid in self.evidence]
			
			supporting_customer = [e for e in evidence_items if e.supporting_dispute]
			supporting_merchant = [e for e in evidence_items if not e.supporting_dispute]
			
			# Calculate evidence scores
			customer_evidence_score = sum(e.confidence_score for e in supporting_customer) / max(len(supporting_customer), 1)
			merchant_evidence_score = sum(e.confidence_score for e in supporting_merchant) / max(len(supporting_merchant), 1)
			
			# Check for specific patterns
			patterns = {
				'has_transaction_records': any(e.evidence_type == 'transaction_record' for e in evidence_items),
				'has_usage_data': any(e.evidence_type == 'usage_records' for e in evidence_items),
				'has_communication_history': any(e.evidence_type == 'customer_communication' for e in evidence_items),
				'duplicate_charges_detected': any('duplicate' in str(e.content) for e in evidence_items),
				'payment_failures_present': any(e.evidence_type == 'payment_record' and 'failed' in str(e.content) for e in evidence_items)
			}
			
			return {
				'total_evidence_items': len(evidence_items),
				'customer_evidence_score': customer_evidence_score,
				'merchant_evidence_score': merchant_evidence_score,
				'evidence_balance': merchant_evidence_score - customer_evidence_score,
				'patterns': patterns
			}
			
		except Exception as e:
			self.logger.error(f"Evidence analysis failed: {e}")
			return {}
	
	def _calculate_resolution_confidence(self, dispute: BillingDispute, evidence_analysis: Dict[str, Any]) -> float:
		"""Calculate confidence in resolution recommendation"""
		try:
			base_confidence = 0.5
			
			# Adjust based on evidence strength
			evidence_balance = evidence_analysis.get('evidence_balance', 0)
			if abs(evidence_balance) > 0.3:
				base_confidence += 0.2  # Strong evidence one way or another
			
			# Adjust based on dispute type
			dispute_template = self.resolution_templates.get(dispute.dispute_type, {})
			if dispute_template:
				base_confidence += 0.1  # We have experience with this type
			
			# Adjust based on patterns
			patterns = evidence_analysis.get('patterns', {})
			if patterns.get('duplicate_charges_detected'):
				base_confidence += 0.3  # Clear duplicate charges
			if patterns.get('has_transaction_records'):
				base_confidence += 0.1  # Good transaction evidence
			
			# Adjust based on customer history
			customer_disputes = [d for d in self.disputes.values() if d.customer_id == dispute.customer_id]
			if len(customer_disputes) > 3:
				base_confidence -= 0.1  # Frequent disputer, be more careful
			
			return max(0.0, min(1.0, base_confidence))
			
		except Exception as e:
			self.logger.error(f"Confidence calculation failed: {e}")
			return 0.5
	
	def _determine_recommended_action(self, dispute: BillingDispute, evidence_analysis: Dict[str, Any], confidence: float) -> Optional[ResolutionAction]:
		"""Determine recommended resolution action"""
		try:
			patterns = evidence_analysis.get('patterns', {})
			evidence_balance = evidence_analysis.get('evidence_balance', 0)
			
			# Clear-cut cases
			if patterns.get('duplicate_charges_detected'):
				return ResolutionAction.FULL_REFUND
			
			# Strong evidence favoring customer
			if evidence_balance < -0.5:
				if dispute.dispute_type in [DisputeType.BILLING_ERROR, DisputeType.AMOUNT_INCORRECT]:
					return ResolutionAction.INVOICE_ADJUSTMENT
				else:
					return ResolutionAction.PARTIAL_REFUND
			
			# Strong evidence favoring merchant
			if evidence_balance > 0.5:
				return ResolutionAction.NO_ACTION
			
			# Ambiguous cases - use dispute type defaults
			dispute_template = self.resolution_templates.get(dispute.dispute_type, {})
			typical_actions = dispute_template.get('typical_actions', [])
			
			if typical_actions:
				# Choose most conservative action for ambiguous cases
				if ResolutionAction.NO_ACTION in typical_actions:
					return ResolutionAction.NO_ACTION
				elif ResolutionAction.CREDIT_APPLIED in typical_actions:
					return ResolutionAction.CREDIT_APPLIED
				else:
					return typical_actions[0]
			
			# Default to manual review for unclear cases
			return ResolutionAction.MANUAL_REVIEW
			
		except Exception as e:
			self.logger.error(f"Action determination failed: {e}")
			return ResolutionAction.MANUAL_REVIEW
	
	def _calculate_recommended_amount(self, dispute: BillingDispute, action: Optional[ResolutionAction], evidence_analysis: Dict[str, Any]) -> Decimal:
		"""Calculate recommended resolution amount"""
		try:
			if not action:
				return Decimal('0')
			
			if action == ResolutionAction.FULL_REFUND:
				return dispute.disputed_amount
			elif action == ResolutionAction.NO_ACTION:
				return Decimal('0')
			elif action in [ResolutionAction.PARTIAL_REFUND, ResolutionAction.CREDIT_APPLIED, ResolutionAction.GOODWILL_GESTURE]:
				# Calculate partial amount based on evidence and dispute type
				evidence_balance = evidence_analysis.get('evidence_balance', 0)
				
				if evidence_balance < -0.3:  # Evidence favors customer
					return dispute.disputed_amount * Decimal('0.75')  # 75% refund
				elif evidence_balance < 0:  # Slight customer favor
					return dispute.disputed_amount * Decimal('0.5')   # 50% refund
				else:  # Goodwill gesture
					return dispute.disputed_amount * Decimal('0.25')  # 25% goodwill
			
			return Decimal('0')
			
		except Exception as e:
			self.logger.error(f"Amount calculation failed: {e}")
			return Decimal('0')
	
	def _generate_recommendation_explanation(self, dispute: BillingDispute, evidence_analysis: Dict[str, Any], action: Optional[ResolutionAction]) -> str:
		"""Generate human-readable explanation for recommendation"""
		try:
			explanations = []
			
			# Evidence-based explanations
			total_evidence = evidence_analysis.get('total_evidence_items', 0)
			if total_evidence > 0:
				explanations.append(f"Analysis of {total_evidence} evidence items")
			
			patterns = evidence_analysis.get('patterns', {})
			if patterns.get('duplicate_charges_detected'):
				explanations.append("Duplicate charges detected in transaction records")
			if patterns.get('has_transaction_records'):
				explanations.append("Transaction records provide clear payment history")
			if patterns.get('payment_failures_present'):
				explanations.append("Payment failures may explain billing discrepancies")
			
			# Evidence balance explanation
			evidence_balance = evidence_analysis.get('evidence_balance', 0)
			if evidence_balance > 0.3:
				explanations.append("Evidence strongly supports merchant position")
			elif evidence_balance < -0.3:
				explanations.append("Evidence supports customer claim")
			else:
				explanations.append("Evidence is balanced, requiring careful consideration")
			
			# Action explanation
			if action:
				action_explanations = {
					ResolutionAction.FULL_REFUND: "Full refund recommended due to clear billing error",
					ResolutionAction.PARTIAL_REFUND: "Partial refund recommended based on shared responsibility",
					ResolutionAction.CREDIT_APPLIED: "Account credit recommended as fair resolution",
					ResolutionAction.NO_ACTION: "No action needed - charges appear valid",
					ResolutionAction.MANUAL_REVIEW: "Manual review required due to complex circumstances"
				}
				explanations.append(action_explanations.get(action, "Custom resolution recommended"))
			
			return ". ".join(explanations) + "."
			
		except Exception as e:
			self.logger.error(f"Explanation generation failed: {e}")
			return "AI analysis completed with standard recommendation."
	
	async def _attempt_automatic_resolution(self, dispute: BillingDispute) -> None:
		"""Attempt automatic resolution for high-confidence cases"""
		try:
			ai_rec = dispute.ai_recommendation
			recommended_action = ResolutionAction(ai_rec['recommended_action'])
			recommended_amount = Decimal(ai_rec['recommended_amount'])
			
			# Only auto-resolve certain low-risk actions
			auto_resolvable_actions = [
				ResolutionAction.CREDIT_APPLIED,
				ResolutionAction.PARTIAL_REFUND
			]
			
			if recommended_action in auto_resolvable_actions and recommended_amount <= Decimal('100'):
				await self._process_automatic_resolution(dispute, recommended_action, ai_rec['explanation'])
			else:
				# High-value or complex cases still need human review
				dispute.status = DisputeStatus.PENDING_INTERNAL
				await self._assign_dispute(dispute)
			
		except Exception as e:
			self.logger.error(f"Automatic resolution attempt failed: {e}")
			dispute.status = DisputeStatus.PENDING_INTERNAL
	
	async def _process_automatic_resolution(self, dispute: BillingDispute, action: ResolutionAction, notes: str) -> None:
		"""Process automatic resolution"""
		try:
			# Update dispute status
			dispute.status = DisputeStatus.RESOLVED_FAVOR_CUSTOMER
			dispute.resolution_action = action
			dispute.resolution_amount = Decimal(dispute.ai_recommendation.get('recommended_amount', '0'))
			dispute.resolution_notes = f"Automatically resolved: {notes}"
			dispute.actual_resolution_date = datetime.utcnow()
			
			# Execute resolution action
			await self._execute_resolution_action(dispute)
			
			# Send resolution communication to customer
			await self._send_resolution_communication(dispute)
			
			# Log automatic resolution
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.COMPLIANCE_CHECK.value,
				'user_id': 'system',
				'resource_type': 'dispute_resolution',
				'resource_id': dispute.id,
				'action': 'automatic_resolution',
				'description': f'Dispute automatically resolved: {action.value}',
				'metadata': {
					'resolution_amount': str(dispute.resolution_amount),
					'confidence_score': dispute.ai_recommendation.get('confidence', 0),
					'resolution_time_hours': (dispute.actual_resolution_date - dispute.created_at).total_seconds() / 3600
				}
			})
			
			self.logger.info(f"Dispute {dispute.id} automatically resolved with {action.value}")
			
		except Exception as e:
			self.logger.error(f"Automatic resolution processing failed: {e}")
			dispute.status = DisputeStatus.PENDING_INTERNAL
	
	async def _execute_resolution_action(self, dispute: BillingDispute) -> None:
		"""Execute the resolution action"""
		try:
			if dispute.resolution_action == ResolutionAction.FULL_REFUND:
				await self._process_refund(dispute, dispute.resolution_amount)
			elif dispute.resolution_action == ResolutionAction.PARTIAL_REFUND:
				await self._process_refund(dispute, dispute.resolution_amount)
			elif dispute.resolution_action == ResolutionAction.CREDIT_APPLIED:
				await self._apply_account_credit(dispute, dispute.resolution_amount)
			elif dispute.resolution_action == ResolutionAction.INVOICE_ADJUSTMENT:
				await self._adjust_invoice(dispute, dispute.resolution_amount)
			
		except Exception as e:
			self.logger.error(f"Resolution action execution failed: {e}")
	
	async def _process_refund(self, dispute: BillingDispute, amount: Decimal) -> None:
		"""Process refund for dispute resolution"""
		try:
			self.logger.info(f"Processing refund of ${amount} for dispute {dispute.id}")
			
			# Find the original payment to refund
			original_payment = None
			if dispute.related_transactions:
				for transaction_id in dispute.related_transactions:
					payment = self.billing_service.payments.get(transaction_id)
					if payment and payment.status.value == 'completed':
						original_payment = payment
						break
			
			if not original_payment:
				# Find the most recent successful payment for this customer
				customer_payments = [
					p for p in self.billing_service.payments.values()
					if (p.customer_id == dispute.customer_id and 
						p.status.value == 'completed' and
						p.amount >= amount)
				]
				if customer_payments:
					original_payment = max(customer_payments, key=lambda p: p.created_at)
			
			if original_payment:
				# Process refund based on payment method
				refund_result = await self._process_payment_method_refund(original_payment, amount, dispute)
				
				if refund_result['success']:
					# Create refund record in billing system
					from .models import BLPayment, PaymentStatus
					refund_payment = BLPayment({
						'id': uuid7str(),
						'customer_id': dispute.customer_id,
						'amount': -amount,  # Negative amount for refund
						'currency': original_payment.currency,
						'payment_method': original_payment.payment_method,
						'status': PaymentStatus.COMPLETED.value,
						'transaction_type': 'refund',
						'original_payment_id': original_payment.id,
						'created_at': datetime.utcnow().isoformat(),
						'metadata': {
							'dispute_id': dispute.id,
							'refund_reason': f"Dispute resolution: {dispute.dispute_type.value}",
							'processor_refund_id': refund_result.get('refund_id'),
							'processor_response': refund_result.get('processor_response')
						}
					})
					
					# Store refund payment record
					self.billing_service.payments[refund_payment.id] = refund_payment
					
					# Update customer's refund history
					customer = self.billing_service.customers.get(dispute.customer_id)
					if customer:
						if not customer.metadata:
							customer.metadata = {}
						
						refund_history = customer.metadata.get('refund_history', [])
						refund_history.append({
							'refund_id': refund_payment.id,
							'amount': str(amount),
							'dispute_id': dispute.id,
							'date': datetime.utcnow().isoformat(),
							'reason': 'dispute_resolution'
						})
						customer.metadata['refund_history'] = refund_history[-10:]  # Keep last 10 refunds
					
					self.logger.info(f"Successfully processed refund of ${amount} for dispute {dispute.id}")
				else:
					raise Exception(f"Refund processing failed: {refund_result.get('error', 'Unknown error')}")
			else:
				# If no original payment found, apply as account credit instead
				self.logger.warning(f"No suitable payment found for refund, applying as account credit")
				await self._apply_account_credit(dispute, amount)
			
		except Exception as e:
			self.logger.error(f"Refund processing failed: {e}")
			raise
	
	async def _process_payment_method_refund(self, original_payment, refund_amount: Decimal, dispute: BillingDispute) -> Dict[str, Any]:
		"""Process refund through the original payment processor"""
		try:
			payment_method = original_payment.payment_method.lower()
			
			if payment_method == 'stripe':
				return await self._process_stripe_refund(original_payment, refund_amount, dispute)
			elif payment_method == 'paypal':
				return await self._process_paypal_refund(original_payment, refund_amount, dispute)
			elif payment_method.startswith('card'):
				return await self._process_card_refund(original_payment, refund_amount, dispute)
			else:
				# Generic refund processing
				return await self._process_generic_refund(original_payment, refund_amount, dispute)
				
		except Exception as e:
			self.logger.error(f"Payment method refund failed: {e}")
			return {'success': False, 'error': str(e)}
	
	async def _process_stripe_refund(self, original_payment, refund_amount: Decimal, dispute: BillingDispute) -> Dict[str, Any]:
		"""Process Stripe refund"""
		try:
			import stripe
			import os
			
			stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
			if not stripe.api_key:
				raise Exception("Stripe API key not configured")
			
			# Get original charge ID from payment metadata
			charge_id = original_payment.metadata.get('stripe_charge_id') if original_payment.metadata else None
			if not charge_id:
				raise Exception("Stripe charge ID not found in payment metadata")
			
			# Create refund
			refund = stripe.Refund.create(
				charge=charge_id,
				amount=int(refund_amount * 100),  # Stripe uses cents
				reason='requested_by_customer',
				metadata={
					'dispute_id': dispute.id,
					'dispute_type': dispute.dispute_type.value,
					'refund_reason': 'dispute_resolution'
				}
			)
			
			return {
				'success': True,
				'refund_id': refund.id,
				'status': refund.status,
				'processor_response': {
					'charge_id': charge_id,
					'refund_id': refund.id,
					'amount_refunded': refund.amount / 100,
					'currency': refund.currency,
					'status': refund.status,
					'created': refund.created
				}
			}
			
		except Exception as e:
			self.logger.error(f"Stripe refund failed: {e}")
			return {'success': False, 'error': str(e)}
	
	async def _process_paypal_refund(self, original_payment, refund_amount: Decimal, dispute: BillingDispute) -> Dict[str, Any]:
		"""Process PayPal refund"""
		try:
			import aiohttp
			import os
			import base64
			
			# PayPal API credentials
			client_id = os.getenv('PAYPAL_CLIENT_ID')
			client_secret = os.getenv('PAYPAL_CLIENT_SECRET')
			paypal_env = os.getenv('PAYPAL_ENVIRONMENT', 'sandbox')
			
			if not client_id or not client_secret:
				raise Exception("PayPal API credentials not configured")
			
			base_url = 'https://api.sandbox.paypal.com' if paypal_env == 'sandbox' else 'https://api.paypal.com'
			
			# Get access token
			auth_string = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
			
			async with aiohttp.ClientSession() as session:
				# Get OAuth token
				token_response = await session.post(
					f"{base_url}/v1/oauth2/token",
					headers={
						'Authorization': f'Basic {auth_string}',
						'Content-Type': 'application/x-www-form-urlencoded',
						'Accept': 'application/json'
					},
					data='grant_type=client_credentials'
				)
				
				token_data = await token_response.json()
				access_token = token_data['access_token']
				
				# Get original transaction ID from payment metadata
				transaction_id = original_payment.metadata.get('paypal_transaction_id') if original_payment.metadata else None
				if not transaction_id:
					raise Exception("PayPal transaction ID not found in payment metadata")
				
				# Process refund
				refund_data = {
					'amount': {
						'value': str(refund_amount),
						'currency_code': original_payment.currency
					},
					'note_to_payer': f'Refund for dispute {dispute.id}',
					'invoice_id': f'dispute-refund-{dispute.id}'
				}
				
				refund_response = await session.post(
					f"{base_url}/v2/payments/captures/{transaction_id}/refund",
					headers={
						'Authorization': f'Bearer {access_token}',
						'Content-Type': 'application/json',
						'Accept': 'application/json'
					},
					json=refund_data
				)
				
				refund_result = await refund_response.json()
				
				if refund_response.status == 201:
					return {
						'success': True,
						'refund_id': refund_result['id'],
						'status': refund_result['status'],
						'processor_response': refund_result
					}
				else:
					return {
						'success': False,
						'error': refund_result.get('message', 'PayPal refund failed')
					}
			
		except Exception as e:
			self.logger.error(f"PayPal refund failed: {e}")
			return {'success': False, 'error': str(e)}
	
	async def _process_card_refund(self, original_payment, refund_amount: Decimal, dispute: BillingDispute) -> Dict[str, Any]:
		"""Process generic card refund through payment gateway"""
		try:
			# This would integrate with your primary payment gateway
			# For demonstration, using a generic approach
			
			transaction_id = original_payment.metadata.get('gateway_transaction_id') if original_payment.metadata else original_payment.id
			
			# Simulate card refund processing
			refund_id = f"ref_{uuid7str()}"
			
			return {
				'success': True,
				'refund_id': refund_id,
				'status': 'processing',
				'processor_response': {
					'original_transaction_id': transaction_id,
					'refund_id': refund_id,
					'amount': str(refund_amount),
					'currency': original_payment.currency,
					'status': 'processing',
					'estimated_completion': (datetime.utcnow() + timedelta(days=3)).isoformat()
				}
			}
			
		except Exception as e:
			self.logger.error(f"Card refund failed: {e}")
			return {'success': False, 'error': str(e)}
	
	async def _process_generic_refund(self, original_payment, refund_amount: Decimal, dispute: BillingDispute) -> Dict[str, Any]:
		"""Process generic refund for unsupported payment methods"""
		try:
			# For unsupported payment methods, create a manual refund request
			refund_id = f"manual_{uuid7str()}"
			
			# Log manual refund requirement
			self.logger.warning(f"Manual refund required for payment method {original_payment.payment_method}")
			
			return {
				'success': True,
				'refund_id': refund_id,
				'status': 'manual_review_required',
				'processor_response': {
					'refund_type': 'manual',
					'original_payment_id': original_payment.id,
					'amount': str(refund_amount),
					'currency': original_payment.currency,
					'payment_method': original_payment.payment_method,
					'requires_manual_processing': True,
					'note': 'Manual refund processing required for this payment method'
				}
			}
			
		except Exception as e:
			self.logger.error(f"Generic refund failed: {e}")
			return {'success': False, 'error': str(e)}
	
	async def _apply_account_credit(self, dispute: BillingDispute, amount: Decimal) -> None:
		"""Apply account credit for dispute resolution"""
		try:
			# Apply credit to customer account
			customer = self.billing_service.customers.get(dispute.customer_id)
			if customer:
				# Add credit to customer metadata
				if not customer.metadata:
					customer.metadata = {}
				
				current_credit = Decimal(str(customer.metadata.get('account_credit', '0')))
				new_credit = current_credit + amount
				customer.metadata['account_credit'] = str(new_credit)
				
				self.logger.info(f"Applied ${amount} credit to customer {dispute.customer_id}")
			
		except Exception as e:
			self.logger.error(f"Account credit application failed: {e}")
	
	async def _adjust_invoice(self, dispute: BillingDispute, adjustment_amount: Decimal) -> None:
		"""Adjust invoice for dispute resolution"""
		try:
			# Find related invoice and adjust
			if dispute.related_transactions:
				for transaction_id in dispute.related_transactions:
					invoice = self.billing_service.invoices.get(transaction_id)
					if invoice:
						# Adjust invoice amount
						invoice.total = max(Decimal('0'), invoice.total - adjustment_amount)
						invoice.amount_due = max(Decimal('0'), invoice.amount_due - adjustment_amount)
						
						# Add adjustment note
						if not invoice.metadata:
							invoice.metadata = {}
						invoice.metadata['dispute_adjustment'] = {
							'dispute_id': dispute.id,
							'adjustment_amount': str(adjustment_amount),
							'adjusted_at': datetime.utcnow().isoformat()
						}
						
						self.logger.info(f"Adjusted invoice {invoice.id} by ${adjustment_amount}")
						break
			
		except Exception as e:
			self.logger.error(f"Invoice adjustment failed: {e}")
	
	async def _send_resolution_communication(self, dispute: BillingDispute) -> None:
		"""Send resolution communication to customer"""
		try:
			# Get financial journey orchestrator for communication
			from .financial_journey_orchestration import get_financial_journey_orchestrator
			orchestrator = get_financial_journey_orchestrator()
			
			# Prepare resolution context
			context = {
				'dispute_id': dispute.id,
				'resolution_action': dispute.resolution_action.value,
				'resolution_amount': str(dispute.resolution_amount),
				'resolution_notes': dispute.resolution_notes
			}
			
			await orchestrator._send_contextual_communication(
				dispute.customer_id, 'dispute_resolution', context
			)
			
		except Exception as e:
			self.logger.error(f"Resolution communication failed: {e}")
	
	# Public API methods
	
	async def get_dispute_status(self, dispute_id: str) -> Optional[Dict[str, Any]]:
		"""Get current status of a dispute"""
		dispute = self.disputes.get(dispute_id)
		if not dispute:
			return None
		
		# Calculate resolution time
		if dispute.actual_resolution_date:
			resolution_time_hours = (dispute.actual_resolution_date - dispute.created_at).total_seconds() / 3600
		else:
			resolution_time_hours = None
		
		# Check SLA status
		sla_deadline = dispute.target_resolution_date
		is_overdue = datetime.utcnow() > sla_deadline and not dispute.actual_resolution_date
		
		return {
			'dispute_id': dispute.id,
			'status': dispute.status.value,
			'dispute_type': dispute.dispute_type.value,
			'disputed_amount': str(dispute.disputed_amount),
			'resolution_amount': str(dispute.resolution_amount) if dispute.resolution_amount else None,
			'created_at': dispute.created_at.isoformat(),
			'target_resolution_date': sla_deadline.isoformat(),
			'actual_resolution_date': dispute.actual_resolution_date.isoformat() if dispute.actual_resolution_date else None,
			'resolution_time_hours': resolution_time_hours,
			'is_overdue': is_overdue,
			'ai_confidence': dispute.ai_recommendation.get('confidence'),
			'evidence_count': len(dispute.evidence_items)
		}
	
	async def get_customer_disputes(self, customer_id: str, status: str = None) -> List[Dict[str, Any]]:
		"""Get all disputes for a customer"""
		customer_disputes = [
			dispute for dispute in self.disputes.values()
			if dispute.customer_id == customer_id
		]
		
		if status:
			try:
				status_enum = DisputeStatus(status)
				customer_disputes = [d for d in customer_disputes if d.status == status_enum]
			except ValueError:
				pass  # Invalid status, return all
		
		return [
			{
				'dispute_id': dispute.id,
				'dispute_type': dispute.dispute_type.value,
				'status': dispute.status.value,
				'disputed_amount': str(dispute.disputed_amount),
				'created_at': dispute.created_at.isoformat(),
				'resolution_date': dispute.actual_resolution_date.isoformat() if dispute.actual_resolution_date else None
			}
			for dispute in sorted(customer_disputes, key=lambda d: d.created_at, reverse=True)
		]
	
	async def get_dispute_analytics(self, days: int = 30) -> Dict[str, Any]:
		"""Get dispute resolution analytics"""
		cutoff_date = datetime.utcnow() - timedelta(days=days)
		
		# Filter disputes by date range
		period_disputes = [
			dispute for dispute in self.disputes.values()
			if dispute.created_at >= cutoff_date
		]
		
		# Calculate metrics
		total_disputes = len(period_disputes)
		resolved_disputes = len([d for d in period_disputes if d.actual_resolution_date])
		auto_resolved = len([d for d in period_disputes if d.resolution_notes and 'automatically' in d.resolution_notes.lower()])
		
		# Resolution time metrics
		resolved_with_time = [d for d in period_disputes if d.actual_resolution_date]
		if resolved_with_time:
			resolution_times = [
				(d.actual_resolution_date - d.created_at).total_seconds() / 3600
				for d in resolved_with_time
			]
			avg_resolution_time = sum(resolution_times) / len(resolution_times)
		else:
			avg_resolution_time = 0
		
		# Dispute type distribution
		type_distribution = {}
		for dispute in period_disputes:
			dispute_type = dispute.dispute_type.value
			type_distribution[dispute_type] = type_distribution.get(dispute_type, 0) + 1
		
		# Resolution action distribution
		action_distribution = {}
		for dispute in resolved_with_time:
			if dispute.resolution_action:
				action = dispute.resolution_action.value
				action_distribution[action] = action_distribution.get(action, 0) + 1
		
		# Financial impact
		total_disputed = sum(d.disputed_amount for d in period_disputes)
		total_resolved_amount = sum(d.resolution_amount for d in resolved_with_time if d.resolution_amount)
		
		return {
			'period_days': days,
			'total_disputes': total_disputes,
			'resolved_disputes': resolved_disputes,
			'auto_resolved_disputes': auto_resolved,
			'resolution_rate': resolved_disputes / max(total_disputes, 1),
			'automation_rate': auto_resolved / max(resolved_disputes, 1),
			'avg_resolution_time_hours': avg_resolution_time,
			'total_disputed_amount': str(total_disputed),
			'total_resolved_amount': str(total_resolved_amount),
			'dispute_type_distribution': type_distribution,
			'resolution_action_distribution': action_distribution,
			'generated_at': datetime.utcnow().isoformat()
		}


# Global dispute resolution engine
_dispute_resolution_instance: Optional[DisputeResolutionEngine] = None

def get_dispute_resolution_engine() -> DisputeResolutionEngine:
	"""Get global dispute resolution engine instance"""
	global _dispute_resolution_instance
	if _dispute_resolution_instance is None:
		_dispute_resolution_instance = DisputeResolutionEngine()
	return _dispute_resolution_instance


__all__ = [
	'DisputeResolutionEngine',
	'BillingDispute',
	'DisputeEvidence',
	'DisputeType',
	'DisputeStatus',
	'ResolutionAction',
	'get_dispute_resolution_engine'
]