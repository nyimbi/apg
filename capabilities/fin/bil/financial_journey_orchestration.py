"""
APG Unified Customer Financial Journey Orchestration

End-to-end orchestration of every customer financial interaction with intelligent
automation, contextual communications, and proactive customer success interventions
that create seamless, delightful billing experiences.

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

from .models import BLCustomer, BLSubscription, BLPayment, BLInvoice
from .service import get_billing_service
from .predictive_billing_ai import get_predictive_billing_ai
from .audit_compliance import get_audit_compliance_system, AuditEventType


class JourneyStage(Enum):
	"""Customer financial journey stages"""
	DISCOVERY = "discovery"
	TRIAL = "trial"
	CONVERSION = "conversion"
	ONBOARDING = "onboarding"
	ACTIVE = "active"
	EXPANSION = "expansion"
	RETENTION = "retention"
	WIN_BACK = "win_back"
	CHURNED = "churned"


class TouchpointType(Enum):
	"""Types of financial touchpoints"""
	PAYMENT_SUCCESS = "payment_success"
	PAYMENT_FAILURE = "payment_failure"
	INVOICE_GENERATED = "invoice_generated"
	TRIAL_EXPIRY = "trial_expiry"
	SUBSCRIPTION_CHANGE = "subscription_change"
	USAGE_THRESHOLD = "usage_threshold"
	PRICE_CHANGE = "price_change"
	DUNNING_SEQUENCE = "dunning_sequence"
	SUPPORT_INTERACTION = "support_interaction"
	RENEWAL_UPCOMING = "renewal_upcoming"


class CommunicationChannel(Enum):
	"""Communication channels"""
	EMAIL = "email"
	SMS = "sms"
	PUSH_NOTIFICATION = "push_notification"
	IN_APP = "in_app"
	PHONE_CALL = "phone_call"
	CHAT = "chat"
	WEBHOOK = "webhook"


class JourneyTouchpoint:
	"""Individual customer touchpoint in financial journey"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.customer_id = data['customer_id']
		self.journey_stage = JourneyStage(data['journey_stage'])
		self.touchpoint_type = TouchpointType(data['touchpoint_type'])
		self.timestamp = datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat()))
		self.context = data.get('context', {})
		self.customer_sentiment = data.get('customer_sentiment', 'neutral')  # positive, neutral, negative
		self.automation_level = data.get('automation_level', 'manual')  # automated, semi_automated, manual
		self.outcome = data.get('outcome')  # success, pending, failure
		self.next_actions = data.get('next_actions', [])
		self.metadata = data.get('metadata', {})


class JourneyOrchestration:
	"""Orchestrated sequence of touchpoints and actions"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.customer_id = data['customer_id']
		self.orchestration_type = data['orchestration_type']  # onboarding, retention, recovery, etc.
		self.current_stage = JourneyStage(data.get('current_stage', JourneyStage.ACTIVE.value))
		self.priority = data.get('priority', 'medium')  # low, medium, high, urgent
		self.start_date = datetime.fromisoformat(data.get('start_date', datetime.utcnow().isoformat()))
		self.target_completion = datetime.fromisoformat(data['target_completion'])
		self.status = data.get('status', 'active')  # active, paused, completed, cancelled
		self.success_criteria = data.get('success_criteria', {})
		self.touchpoints = data.get('touchpoints', [])
		self.automated_actions = data.get('automated_actions', [])
		self.human_interventions = data.get('human_interventions', [])
		self.results = data.get('results', {})
		self.metadata = data.get('metadata', {})


class SmartPaymentRouter:
	"""Intelligent payment routing and optimization"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.SmartPaymentRouter")
		self.routing_rules = {}
		self.payment_preferences = {}
		self.failure_patterns = {}
	
	async def route_payment(self, customer_id: str, payment_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Route payment using intelligent optimization"""
		try:
			# Get customer payment history and preferences
			customer_profile = await self._get_customer_payment_profile(customer_id)
			
			# Analyze optimal routing
			routing_decision = await self._calculate_optimal_routing(customer_profile, payment_data)
			
			# Apply fraud prevention
			fraud_check = await self._perform_fraud_check(customer_id, payment_data, routing_decision)
			
			if fraud_check['risk_score'] > 0.8:
				return {
					'action': 'block',
					'reason': 'high_fraud_risk',
					'alternative_actions': ['manual_review', 'additional_verification']
				}
			
			# Determine retry strategy
			retry_strategy = await self._optimize_retry_strategy(customer_profile, payment_data)
			
			return {
				'action': 'process',
				'payment_processor': routing_decision['processor'],
				'payment_method': routing_decision['method'],
				'retry_strategy': retry_strategy,
				'confidence_score': routing_decision['confidence'],
				'expected_success_rate': routing_decision['success_probability']
			}
			
		except Exception as e:
			self.logger.error(f"Payment routing failed: {e}")
			return {'action': 'fallback', 'processor': 'default'}
	
	async def _get_customer_payment_profile(self, customer_id: str) -> Dict[str, Any]:
		"""Get comprehensive customer payment profile"""
		billing_service = get_billing_service()
		
		# Get payment history
		customer_payments = [
			p for p in billing_service.payments.values()
			if p.customer_id == customer_id
		]
		
		# Analyze success patterns
		successful_payments = [p for p in customer_payments if p.status.value == 'succeeded']
		failed_payments = [p for p in customer_payments if p.status.value == 'failed']
		
		# Calculate success rates by method and processor
		method_success = {}
		processor_success = {}
		
		for payment in customer_payments:
			method = payment.payment_method or 'unknown'
			processor = payment.metadata.get('processor', 'unknown') if payment.metadata else 'unknown'
			
			if method not in method_success:
				method_success[method] = {'total': 0, 'successful': 0}
			if processor not in processor_success:
				processor_success[processor] = {'total': 0, 'successful': 0}
			
			method_success[method]['total'] += 1
			processor_success[processor]['total'] += 1
			
			if payment.status.value == 'succeeded':
				method_success[method]['successful'] += 1
				processor_success[processor]['successful'] += 1
		
		# Calculate preferred timing
		successful_hours = [p.created_at.hour for p in successful_payments]
		preferred_hour = max(set(successful_hours), key=successful_hours.count) if successful_hours else 12
		
		return {
			'total_payments': len(customer_payments),
			'success_rate': len(successful_payments) / max(len(customer_payments), 1),
			'method_preferences': {
				method: data['successful'] / max(data['total'], 1)
				for method, data in method_success.items()
			},
			'processor_preferences': {
				processor: data['successful'] / max(data['total'], 1)
				for processor, data in processor_success.items()
			},
			'preferred_hour': preferred_hour,
			'recent_failures': len([p for p in failed_payments if (datetime.utcnow() - p.created_at).days <= 30]),
			'consecutive_failures': self._count_consecutive_failures(customer_payments)
		}
	
	def _count_consecutive_failures(self, payments: List) -> int:
		"""Count consecutive recent failures"""
		if not payments:
			return 0
		
		sorted_payments = sorted(payments, key=lambda p: p.created_at, reverse=True)
		consecutive = 0
		
		for payment in sorted_payments:
			if payment.status.value == 'failed':
				consecutive += 1
			else:
				break
		
		return consecutive
	
	async def _calculate_optimal_routing(self, customer_profile: Dict[str, Any], payment_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Calculate optimal payment routing"""
		try:
			# Default routing decision
			routing = {
				'processor': 'stripe',
				'method': 'card',
				'confidence': 0.5,
				'success_probability': 0.85
			}
			
			# Use customer preferences if available
			method_prefs = customer_profile.get('method_preferences', {})
			processor_prefs = customer_profile.get('processor_preferences', {})
			
			if method_prefs:
				best_method = max(method_prefs.items(), key=lambda x: x[1])
				if best_method[1] > 0.8:  # High success rate
					routing['method'] = best_method[0]
					routing['confidence'] += 0.2
			
			if processor_prefs:
				best_processor = max(processor_prefs.items(), key=lambda x: x[1])
				if best_processor[1] > 0.8:
					routing['processor'] = best_processor[0]
					routing['confidence'] += 0.2
			
			# Adjust for payment amount
			amount = Decimal(str(payment_data.get('amount', 0)))
			if amount > Decimal('1000'):
				# High-value payments prefer more reliable processors
				routing['processor'] = 'adyen'  # Enterprise processor
				routing['confidence'] += 0.1
			
			# Adjust for consecutive failures
			if customer_profile.get('consecutive_failures', 0) > 2:
				# Try alternative processor/method
				routing['processor'] = 'paypal'
				routing['method'] = 'wallet'
				routing['confidence'] -= 0.1
			
			# Time-based optimization
			current_hour = datetime.utcnow().hour
			preferred_hour = customer_profile.get('preferred_hour', 12)
			
			if abs(current_hour - preferred_hour) > 6:
				routing['confidence'] -= 0.1
			
			# Cap confidence and calculate success probability
			routing['confidence'] = max(0.0, min(1.0, routing['confidence']))
			routing['success_probability'] = 0.7 + (routing['confidence'] * 0.2)
			
			return routing
			
		except Exception as e:
			self.logger.error(f"Routing calculation failed: {e}")
			return {'processor': 'stripe', 'method': 'card', 'confidence': 0.5, 'success_probability': 0.8}
	
	async def _perform_fraud_check(self, customer_id: str, payment_data: Dict[str, Any], routing: Dict[str, Any]) -> Dict[str, Any]:
		"""Perform intelligent fraud detection"""
		try:
			risk_score = 0.0
			risk_factors = []
			
			billing_service = get_billing_service()
			customer = billing_service.customers.get(customer_id)
			
			# Check payment amount against customer history
			amount = Decimal(str(payment_data.get('amount', 0)))
			customer_payments = [
				p for p in billing_service.payments.values()
				if p.customer_id == customer_id and p.status.value == 'succeeded'
			]
			
			if customer_payments:
				avg_payment = sum(p.amount for p in customer_payments) / len(customer_payments)
				if amount > avg_payment * 5:  # 5x larger than average
					risk_score += 0.3
					risk_factors.append('unusually_large_amount')
			else:
				# New customer with large payment
				if amount > Decimal('500'):
					risk_score += 0.4
					risk_factors.append('new_customer_large_payment')
			
			# Check customer age
			if customer:
				customer_age = (datetime.utcnow() - customer.created_at).days
				if customer_age < 7:  # Very new customer
					risk_score += 0.2
					risk_factors.append('new_customer')
			
			# Check for rapid payment attempts
			recent_payments = [
				p for p in billing_service.payments.values()
				if p.customer_id == customer_id and (datetime.utcnow() - p.created_at).seconds < 3600
			]
			
			if len(recent_payments) > 3:
				risk_score += 0.3
				risk_factors.append('rapid_payment_attempts')
			
			# Check payment method changes
			payment_method = payment_data.get('payment_method')
			if payment_method:
				recent_methods = [
					p.payment_method for p in customer_payments[-5:]  # Last 5 payments
					if p.payment_method
				]
				
				if payment_method not in recent_methods and recent_methods:
					risk_score += 0.1
					risk_factors.append('new_payment_method')
			
			return {
				'risk_score': min(risk_score, 1.0),
				'risk_factors': risk_factors,
				'recommendation': 'approve' if risk_score < 0.5 else 'review' if risk_score < 0.8 else 'block'
			}
			
		except Exception as e:
			self.logger.error(f"Fraud check failed: {e}")
			return {'risk_score': 0.0, 'risk_factors': [], 'recommendation': 'approve'}
	
	async def _optimize_retry_strategy(self, customer_profile: Dict[str, Any], payment_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Optimize payment retry strategy"""
		try:
			base_strategy = {
				'max_retries': 3,
				'retry_intervals': [1, 24, 72],  # hours
				'escalation_rules': ['email_notification', 'dunning_sequence'],
				'alternative_methods': []
			}
			
			# Adjust based on customer success rate
			success_rate = customer_profile.get('success_rate', 0.8)
			
			if success_rate > 0.9:
				# High success rate customers get more aggressive retries
				base_strategy['max_retries'] = 5
				base_strategy['retry_intervals'] = [0.5, 6, 24, 48, 96]
			elif success_rate < 0.5:
				# Low success rate customers get gentler approach
				base_strategy['max_retries'] = 2
				base_strategy['retry_intervals'] = [6, 48]
				base_strategy['escalation_rules'].append('manual_outreach')
			
			# Add alternative methods based on customer preferences
			method_prefs = customer_profile.get('method_preferences', {})
			current_method = payment_data.get('payment_method', 'card')
			
			for method, success_rate in method_prefs.items():
				if method != current_method and success_rate > 0.7:
					base_strategy['alternative_methods'].append(method)
			
			# Adjust timing based on customer preferences
			preferred_hour = customer_profile.get('preferred_hour', 12)
			base_strategy['preferred_retry_hour'] = preferred_hour
			
			return base_strategy
			
		except Exception as e:
			self.logger.error(f"Retry strategy optimization failed: {e}")
			return {'max_retries': 3, 'retry_intervals': [1, 24, 72]}


class FinancialJourneyOrchestrator:
	"""Unified customer financial journey orchestration engine"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.FinancialJourneyOrchestrator")
		
		# Core components
		self.payment_router = SmartPaymentRouter()
		
		# Data storage
		self.touchpoints: Dict[str, JourneyTouchpoint] = {}
		self.orchestrations: Dict[str, JourneyOrchestration] = {}
		self.customer_journeys: Dict[str, List[str]] = {}  # customer_id -> touchpoint_ids
		
		# Communication templates
		self.communication_templates = {}
		
		# Customer health scoring
		self.customer_health_scores: Dict[str, float] = {}
		
		# Service integrations
		self.billing_service = get_billing_service()
		self.predictive_ai = get_predictive_billing_ai()
		self.audit_system = get_audit_compliance_system()
		
		# Background processing
		asyncio.create_task(self._start_journey_monitor())
		asyncio.create_task(self._start_health_scoring())
		asyncio.create_task(self._initialize_communication_templates())
	
	async def _start_journey_monitor(self) -> None:
		"""Monitor and orchestrate customer journeys"""
		while True:
			try:
				await self._process_journey_triggers()
				await self._execute_scheduled_actions()
				await self._update_journey_stages()
				await asyncio.sleep(300)  # Check every 5 minutes
			except Exception as e:
				self.logger.error(f"Journey monitor error: {e}")
				await asyncio.sleep(300)
	
	async def _start_health_scoring(self) -> None:
		"""Continuously update customer health scores"""
		while True:
			try:
				await self._update_customer_health_scores()
				await asyncio.sleep(3600)  # Update hourly
			except Exception as e:
				self.logger.error(f"Health scoring error: {e}")
				await asyncio.sleep(3600)
	
	async def _initialize_communication_templates(self) -> None:
		"""Initialize communication templates"""
		try:
			self.communication_templates = {
				'payment_success': {
					'subject': 'Payment Confirmed - Thank You!',
					'template': '''
					Hi {{customer_name}},
					
					Great news! Your payment of {{amount}} has been successfully processed.
					
					{{contextual_message}}
					
					Next payment: {{next_payment_date}}
					
					Questions? Reply to this email or visit our support center.
					''',
					'channels': [CommunicationChannel.EMAIL.value]
				},
				'payment_failure': {
					'subject': 'Payment Update Needed',
					'template': '''
					Hi {{customer_name}},
					
					We couldn't process your payment of {{amount}}. This happens sometimes with {{failure_reason}}.
					
					{{personalized_solution}}
					
					Update your payment method: {{payment_link}}
					
					We're here to help: {{support_contact}}
					''',
					'channels': [CommunicationChannel.EMAIL.value, CommunicationChannel.SMS.value]
				},
				'trial_ending': {
					'subject': 'Your trial ends in {{days_left}} days',
					'template': '''
					Hi {{customer_name}},
					
					You've been getting great value from {{service_name}}! Your trial ends in {{days_left}} days.
					
					{{usage_highlights}}
					
					Continue your journey: {{conversion_link}}
					
					Questions about plans? {{consultation_link}}
					''',
					'channels': [CommunicationChannel.EMAIL.value, CommunicationChannel.IN_APP.value]
				},
				'renewal_upcoming': {
					'subject': 'Your subscription renews soon',
					'template': '''
					Hi {{customer_name}},
					
					Your {{plan_name}} subscription renews on {{renewal_date}} for {{amount}}.
					
					{{renewal_benefits}}
					
					Manage your subscription: {{account_link}}
					''',
					'channels': [CommunicationChannel.EMAIL.value]
				}
			}
			
			self.logger.info("Communication templates initialized")
			
		except Exception as e:
			self.logger.error(f"Template initialization failed: {e}")
	
	async def _process_journey_triggers(self) -> None:
		"""Process triggers that initiate journey orchestrations"""
		try:
			# Check for payment events
			await self._check_payment_triggers()
			
			# Check for subscription events
			await self._check_subscription_triggers()
			
			# Check for usage thresholds
			await self._check_usage_triggers()
			
			# Check for predictive alerts
			await self._check_predictive_triggers()
			
		except Exception as e:
			self.logger.error(f"Journey trigger processing failed: {e}")
	
	async def _check_payment_triggers(self) -> None:
		"""Check for payment-related journey triggers"""
		try:
			# Get recent payments
			recent_payments = [
				p for p in self.billing_service.payments.values()
				if (datetime.utcnow() - p.created_at).seconds < 3600  # Last hour
			]
			
			for payment in recent_payments:
				# Check if we've already processed this payment
				existing_touchpoint = any(
					tp for tp in self.touchpoints.values()
					if (tp.customer_id == payment.customer_id and 
						tp.metadata.get('payment_id') == payment.id)
				)
				
				if existing_touchpoint:
					continue
				
				if payment.status.value == 'succeeded':
					await self._trigger_payment_success_journey(payment)
				elif payment.status.value == 'failed':
					await self._trigger_payment_failure_journey(payment)
		
		except Exception as e:
			self.logger.error(f"Payment trigger check failed: {e}")
	
	async def _trigger_payment_success_journey(self, payment) -> None:
		"""Trigger journey for successful payment"""
		try:
			customer = self.billing_service.customers.get(payment.customer_id)
			if not customer:
				return
			
			# Create touchpoint
			touchpoint_data = {
				'customer_id': payment.customer_id,
				'journey_stage': JourneyStage.ACTIVE.value,
				'touchpoint_type': TouchpointType.PAYMENT_SUCCESS.value,
				'context': {
					'payment_amount': str(payment.amount),
					'payment_method': payment.payment_method,
					'currency': payment.currency.value if payment.currency else 'USD'
				},
				'customer_sentiment': 'positive',
				'automation_level': 'automated',
				'outcome': 'success',
				'metadata': {
					'payment_id': payment.id,
					'processed_by': 'journey_orchestrator'
				}
			}
			
			touchpoint = JourneyTouchpoint(touchpoint_data)
			self.touchpoints[touchpoint.id] = touchpoint
			
			# Add to customer journey
			if payment.customer_id not in self.customer_journeys:
				self.customer_journeys[payment.customer_id] = []
			self.customer_journeys[payment.customer_id].append(touchpoint.id)
			
			# Send personalized confirmation
			await self._send_contextual_communication(
				payment.customer_id, 'payment_success', touchpoint.context
			)
			
			# Check for upsell opportunities
			await self._evaluate_upsell_opportunity(payment.customer_id, payment)
			
			self.logger.info(f"Payment success journey triggered for customer {payment.customer_id}")
			
		except Exception as e:
			self.logger.error(f"Payment success journey failed: {e}")
	
	async def _trigger_payment_failure_journey(self, payment) -> None:
		"""Trigger recovery journey for failed payment"""
		try:
			# Create touchpoint
			touchpoint_data = {
				'customer_id': payment.customer_id,
				'journey_stage': JourneyStage.RETENTION.value,
				'touchpoint_type': TouchpointType.PAYMENT_FAILURE.value,
				'context': {
					'payment_amount': str(payment.amount),
					'failure_reason': payment.metadata.get('failure_reason', 'unknown') if payment.metadata else 'unknown',
					'retry_count': payment.metadata.get('retry_count', 0) if payment.metadata else 0
				},
				'customer_sentiment': 'negative',
				'automation_level': 'semi_automated',
				'outcome': 'pending',
				'metadata': {
					'payment_id': payment.id,
					'requires_intervention': True
				}
			}
			
			touchpoint = JourneyTouchpoint(touchpoint_data)
			self.touchpoints[touchpoint.id] = touchpoint
			
			# Add to customer journey
			if payment.customer_id not in self.customer_journeys:
				self.customer_journeys[payment.customer_id] = []
			self.customer_journeys[payment.customer_id].append(touchpoint.id)
			
			# Get intelligent retry strategy
			customer_profile = await self.payment_router._get_customer_payment_profile(payment.customer_id)
			retry_strategy = await self.payment_router._optimize_retry_strategy(
				customer_profile, {'amount': payment.amount, 'payment_method': payment.payment_method}
			)
			
			# Schedule intelligent retry
			await self._schedule_intelligent_retry(payment, retry_strategy)
			
			# Send personalized recovery communication
			await self._send_contextual_communication(
				payment.customer_id, 'payment_failure', touchpoint.context
			)
			
			self.logger.info(f"Payment failure journey triggered for customer {payment.customer_id}")
			
		except Exception as e:
			self.logger.error(f"Payment failure journey failed: {e}")
	
	async def _check_subscription_triggers(self) -> None:
		"""Check for subscription-related triggers"""
		try:
			now = datetime.utcnow()
			
			# Check for trial endings
			for subscription in self.billing_service.subscriptions.values():
				if subscription.trial_end and subscription.status.value == 'trialing':
					days_until_end = (subscription.trial_end - now).days
					
					if days_until_end in [7, 3, 1]:  # Alert at 7, 3, and 1 day(s) before trial end
						await self._trigger_trial_ending_journey(subscription, days_until_end)
				
				# Check for upcoming renewals
				if subscription.current_period_end:
					days_until_renewal = (subscription.current_period_end - now).days
					
					if days_until_renewal in [7, 1]:  # Alert 7 and 1 day before renewal
						await self._trigger_renewal_upcoming_journey(subscription, days_until_renewal)
		
		except Exception as e:
			self.logger.error(f"Subscription trigger check failed: {e}")
	
	async def _trigger_trial_ending_journey(self, subscription, days_left: int) -> None:
		"""Trigger journey for trial ending"""
		try:
			# Check if we've already sent this alert
			existing_alert = any(
				tp for tp in self.touchpoints.values()
				if (tp.customer_id == subscription.customer_id and 
					tp.touchpoint_type == TouchpointType.TRIAL_EXPIRY and
					tp.metadata.get('days_left') == days_left)
			)
			
			if existing_alert:
				return
			
			# Create touchpoint
			touchpoint_data = {
				'customer_id': subscription.customer_id,
				'journey_stage': JourneyStage.CONVERSION.value,
				'touchpoint_type': TouchpointType.TRIAL_EXPIRY.value,
				'context': {
					'days_left': days_left,
					'subscription_id': subscription.id,
					'plan_name': subscription.plan_id  # Would get actual plan name
				},
				'automation_level': 'automated',
				'metadata': {
					'days_left': days_left,
					'urgency': 'high' if days_left <= 1 else 'medium'
				}
			}
			
			touchpoint = JourneyTouchpoint(touchpoint_data)
			self.touchpoints[touchpoint.id] = touchpoint
			
			# Add to customer journey
			if subscription.customer_id not in self.customer_journeys:
				self.customer_journeys[subscription.customer_id] = []
			self.customer_journeys[subscription.customer_id].append(touchpoint.id)
			
			# Send trial ending communication
			await self._send_contextual_communication(
				subscription.customer_id, 'trial_ending', touchpoint.context
			)
			
		except Exception as e:
			self.logger.error(f"Trial ending journey failed: {e}")
	
	async def _send_contextual_communication(self, customer_id: str, template_type: str, context: Dict[str, Any]) -> None:
		"""Send contextual communication to customer"""
		try:
			template = self.communication_templates.get(template_type)
			if not template:
				self.logger.warning(f"No template found for {template_type}")
				return
			
			customer = self.billing_service.customers.get(customer_id)
			if not customer:
				return
			
			# Build context with customer data
			full_context = {
				'customer_name': getattr(customer, 'name', 'Valued Customer'),
				'customer_email': getattr(customer, 'email', ''),
				**context
			}
			
			# Personalize message content
			personalized_content = await self._personalize_message(customer_id, template, full_context)
			
			# Send via appropriate channels
			for channel in template['channels']:
				await self._send_via_channel(customer_id, channel, personalized_content)
			
			# Log communication
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.CUSTOMER_UPDATED.value,
				'user_id': 'system',
				'resource_type': 'customer_communication',
				'resource_id': customer_id,
				'action': 'communication_sent',
				'description': f'Sent {template_type} communication',
				'metadata': {
					'template_type': template_type,
					'channels': template['channels'],
					'context': context
				}
			})
			
		except Exception as e:
			self.logger.error(f"Contextual communication failed: {e}")
	
	async def _personalize_message(self, customer_id: str, template: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Personalize message content using customer intelligence"""
		try:
			# Get customer health score and preferences
			health_score = self.customer_health_scores.get(customer_id, 0.5)
			
			# Adjust tone based on customer health
			if health_score > 0.8:
				tone = 'appreciative'
			elif health_score > 0.5:
				tone = 'supportive'
			else:
				tone = 'concerned'
			
			# Get customer preferences from metadata
			customer = self.billing_service.customers.get(customer_id)
			preferences = {}
			if customer and customer.metadata:
				preferences = customer.metadata.get('communication_preferences', {})
			
			# Apply personalization
			personalized_template = template['template']
			
			# Replace placeholders
			for key, value in context.items():
				placeholder = '{{' + key + '}}'
				personalized_template = personalized_template.replace(placeholder, str(value))
			
			# Add contextual messaging based on customer state
			contextual_message = await self._generate_contextual_message(customer_id, context, tone)
			personalized_template = personalized_template.replace('{{contextual_message}}', contextual_message)
			
			return {
				'subject': template['subject'],
				'content': personalized_template,
				'tone': tone,
				'personalization_applied': True
			}
			
		except Exception as e:
			self.logger.error(f"Message personalization failed: {e}")
			return {'subject': template['subject'], 'content': template['template']}
	
	async def _generate_contextual_message(self, customer_id: str, context: Dict[str, Any], tone: str) -> str:
		"""Generate contextual message based on customer situation"""
		try:
			# Get customer predictions and insights
			predictions = await self.predictive_ai.get_customer_predictions(customer_id)
			
			messages = []
			
			# Add usage insights
			if 'payment_amount' in context:
				messages.append("Your account is in good standing.")
			
			# Add predictions-based messaging
			for prediction in predictions[:2]:  # Top 2 predictions
				if prediction.risk_score > 0.7:
					if prediction.prediction_type.value == 'payment_failure':
						messages.append("We're monitoring your account to ensure smooth payments.")
					elif prediction.prediction_type.value == 'churn_risk':
						messages.append("We're here to help you get the most value from our service.")
			
			# Add value reinforcement
			if tone == 'appreciative':
				messages.append("Thank you for being a valued customer!")
			elif tone == 'supportive':
				messages.append("We're here to support your success.")
			
			return ' '.join(messages) if messages else "We appreciate your business."
			
		except Exception as e:
			self.logger.error(f"Contextual message generation failed: {e}")
			return ""
	
	async def _send_via_channel(self, customer_id: str, channel: str, content: Dict[str, Any]) -> None:
		"""Send communication via specific channel"""
		try:
			# In production, this would integrate with actual communication services
			self.logger.info(f"Sending {channel} to customer {customer_id}: {content['subject']}")
			
			# Send via different communication channels
			if channel == CommunicationChannel.EMAIL.value:
				# Integrate with email service
				try:
					from .email_services import get_billing_email_manager
					email_manager = get_billing_email_manager()
					
					await email_manager.send_custom_email(
						customer_id=customer_id,
						subject=content['subject'],
						html_content=content.get('html_content', content.get('content', '')),
						text_content=content.get('text_content', content.get('content', ''))
					)
					self.logger.info(f"✅ Email sent to customer {customer_id}")
					
				except Exception as e:
					self.logger.warning(f"Email delivery failed: {e}")
					
			elif channel == CommunicationChannel.SMS.value:
				# Integrate with SMS service (Twilio, AWS SNS, AfricaIsTalking)
				try:
					sms_content = content.get('sms_content', content.get('content', ''))
					
					# Get customer phone number
					customer_phone = await self._get_customer_phone_number(customer_id)
					if not customer_phone:
						self.logger.warning(f"No phone number available for customer {customer_id}")
						return
					
					# Send SMS via real service
					sms_result = await self._send_sms_via_service(customer_phone, sms_content, customer_id, content)
					
					# Record SMS delivery
					delivery_record = {
						'channel': 'sms',
						'customer_id': customer_id,
						'phone_number': customer_phone,
						'content': sms_content,
						'sent_at': datetime.utcnow().isoformat(),
						'status': 'delivered' if sms_result['success'] else 'failed',
						'provider': sms_result.get('provider'),
						'message_id': sms_result.get('message_id'),
						'error': sms_result.get('error') if not sms_result['success'] else None,
						'cost': sms_result.get('cost'),
						'segments': sms_result.get('segments', 1)
					}
					
					# Store delivery record
					await self._store_communication_record(delivery_record)
					
				except Exception as e:
					self.logger.warning(f"SMS delivery failed: {e}")
					
			elif channel == CommunicationChannel.PUSH_NOTIFICATION.value:
				# Integrate with push notification service (Firebase, AWS SNS, etc.)
				try:
					push_content = {
						'title': content['subject'],
						'body': content.get('push_content', content.get('content', ''))[:200],
						'data': content.get('metadata', {})
					}
					
					# Get customer device tokens
					device_tokens = await self._get_customer_device_tokens(customer_id)
					if not device_tokens:
						self.logger.warning(f"No device tokens available for customer {customer_id}")
						return
					
					# Send push notifications via real service
					push_result = await self._send_push_via_service(device_tokens, push_content, customer_id, content)
					
					# Record push delivery
					delivery_record = {
						'channel': 'push',
						'customer_id': customer_id,
						'content': push_content,
						'sent_at': datetime.utcnow().isoformat(),
						'status': 'delivered' if push_result['success'] else 'failed',
						'provider': push_result.get('provider'),
						'message_id': push_result.get('message_id'),
						'delivered_count': push_result.get('delivered_count', 0),
						'failed_count': push_result.get('failed_count', 0),
						'error': push_result.get('error') if not push_result['success'] else None
					}
					
					# Store delivery record
					await self._store_communication_record(delivery_record)
					
				except Exception as e:
					self.logger.warning(f"Push notification delivery failed: {e}")
					
			elif channel == CommunicationChannel.IN_APP.value:
				# Show in-app notification
				try:
					in_app_content = {
						'title': content['subject'],
						'message': content.get('in_app_content', content.get('content', '')),
						'type': content.get('notification_type', 'info'),
						'priority': content.get('priority', 'normal'),
						'action_url': content.get('action_url'),
						'created_at': datetime.utcnow().isoformat(),
						'read': False,
						'metadata': content.get('metadata', {})
					}
					
					# Store in-app notification in database
					notification_result = await self._store_in_app_notification(customer_id, in_app_content, content)
					
					# Record delivery
					delivery_record = {
						'channel': 'in_app',
						'customer_id': customer_id,
						'content': in_app_content,
						'sent_at': datetime.utcnow().isoformat(),
						'status': 'delivered' if notification_result['success'] else 'failed',
						'notification_id': notification_result.get('notification_id'),
						'error': notification_result.get('error') if not notification_result['success'] else None
					}
					
					# Store delivery record
					await self._store_communication_record(delivery_record)
					
				except Exception as e:
					self.logger.warning(f"In-app notification creation failed: {e}")
			
		except Exception as e:
			self.logger.error(f"Channel communication failed for {channel}: {e}")
	
	async def _update_customer_health_scores(self) -> None:
		"""Update customer health scores based on financial behavior"""
		try:
			for customer_id in self.billing_service.customers.keys():
				health_score = await self._calculate_customer_health_score(customer_id)
				self.customer_health_scores[customer_id] = health_score
			
			self.logger.info(f"Updated health scores for {len(self.customer_health_scores)} customers")
			
		except Exception as e:
			self.logger.error(f"Health score update failed: {e}")
	
	async def _calculate_customer_health_score(self, customer_id: str) -> float:
		"""Calculate comprehensive customer health score"""
		try:
			score = 0.5  # Base score
			
			# Payment health (40% weight)
			payment_health = await self._calculate_payment_health(customer_id)
			score += payment_health * 0.4
			
			# Engagement health (30% weight)
			engagement_health = await self._calculate_engagement_health(customer_id)
			score += engagement_health * 0.3
			
			# Growth health (20% weight)
			growth_health = await self._calculate_growth_health(customer_id)
			score += growth_health * 0.2
			
			# Support health (10% weight)
			support_health = await self._calculate_support_health(customer_id)
			score += support_health * 0.1
			
			return max(0.0, min(1.0, score))
			
		except Exception as e:
			self.logger.error(f"Health score calculation failed for {customer_id}: {e}")
			return 0.5
	
	async def _calculate_payment_health(self, customer_id: str) -> float:
		"""Calculate payment health component"""
		try:
			customer_payments = [
				p for p in self.billing_service.payments.values()
				if p.customer_id == customer_id
			]
			
			if not customer_payments:
				return 0.0
			
			# Recent payment success rate
			recent_payments = [
				p for p in customer_payments
				if (datetime.utcnow() - p.created_at).days <= 90
			]
			
			success_rate = len([p for p in recent_payments if p.status.value == 'succeeded']) / len(recent_payments)
			
			# Days since last successful payment
			successful_payments = [p for p in customer_payments if p.status.value == 'succeeded']
			if successful_payments:
				last_success = max(p.created_at for p in successful_payments)
				days_since = (datetime.utcnow() - last_success).days
				recency_score = max(0, 1 - (days_since / 60))  # Decay over 60 days
			else:
				recency_score = 0
			
			# Payment consistency
			consistency_score = 1 - min(1, len([p for p in recent_payments if p.status.value == 'failed']) / 10)
			
			return (success_rate * 0.5 + recency_score * 0.3 + consistency_score * 0.2)
			
		except Exception as e:
			self.logger.error(f"Payment health calculation failed: {e}")
			return 0.5
	
	async def _calculate_engagement_health(self, customer_id: str) -> float:
		"""Calculate engagement health component"""
		try:
			customer = self.billing_service.customers.get(customer_id)
			if not customer or not customer.metadata:
				return 0.5
			
			metadata = customer.metadata
			
			# Login frequency
			last_login = metadata.get('last_login_date')
			if last_login:
				if isinstance(last_login, str):
					last_login_date = datetime.fromisoformat(last_login)
				else:
					last_login_date = last_login
				
				days_since_login = (datetime.utcnow() - last_login_date).days
				login_score = max(0, 1 - (days_since_login / 30))  # Decay over 30 days
			else:
				login_score = 0.2
			
			# Feature usage
			features_used = metadata.get('features_used_count', 0)
			feature_score = min(1, features_used / 10)  # Scale to 10 features
			
			# Session frequency
			monthly_sessions = metadata.get('monthly_sessions', 0)
			session_score = min(1, monthly_sessions / 20)  # Scale to 20 sessions
			
			return (login_score * 0.5 + feature_score * 0.3 + session_score * 0.2)
			
		except Exception as e:
			self.logger.error(f"Engagement health calculation failed: {e}")
			return 0.5
	
	async def _calculate_growth_health(self, customer_id: str) -> float:
		"""Calculate growth health component"""
		try:
			# Get customer subscriptions
			customer_subscriptions = [
				s for s in self.billing_service.subscriptions.values()
				if s.customer_id == customer_id
			]
			
			if not customer_subscriptions:
				return 0.0
			
			# Revenue growth
			total_mrr = sum(getattr(s, 'mrr', Decimal('0')) for s in customer_subscriptions)
			mrr_score = min(1, float(total_mrr) / 500)  # Scale to $500 MRR
			
			# Subscription duration
			oldest_sub = min(customer_subscriptions, key=lambda s: s.created_at)
			duration_months = (datetime.utcnow() - oldest_sub.created_at).days / 30
			duration_score = min(1, duration_months / 12)  # Scale to 12 months
			
			# Upgrade activity
			upgrades = len([s for s in customer_subscriptions if s.metadata and s.metadata.get('is_upgrade')])
			upgrade_score = min(1, upgrades / 2)  # Scale to 2 upgrades
			
			return (mrr_score * 0.5 + duration_score * 0.3 + upgrade_score * 0.2)
			
		except Exception as e:
			self.logger.error(f"Growth health calculation failed: {e}")
			return 0.5
	
	async def _calculate_support_health(self, customer_id: str) -> float:
		"""Calculate support health component"""
		try:
			customer = self.billing_service.customers.get(customer_id)
			if not customer or not customer.metadata:
				return 0.8  # Assume good if no data
			
			metadata = customer.metadata
			
			# Support ticket metrics
			recent_tickets = metadata.get('recent_support_tickets', 0)
			complaint_tickets = metadata.get('complaint_tickets', 0)
			satisfaction_score = metadata.get('satisfaction_score', 4.0)  # 1-5 scale
			
			# Calculate component scores
			ticket_volume_score = max(0, 1 - (recent_tickets / 10))  # Penalize high ticket volume
			complaint_score = max(0, 1 - (complaint_tickets / 3))  # Penalize complaints
			satisfaction_norm = (satisfaction_score - 1) / 4  # Normalize 1-5 to 0-1
			
			return (ticket_volume_score * 0.3 + complaint_score * 0.3 + satisfaction_norm * 0.4)
			
		except Exception as e:
			self.logger.error(f"Support health calculation failed: {e}")
			return 0.8
	
	async def _schedule_intelligent_retry(self, payment, retry_strategy: Dict[str, Any]) -> None:
		"""Schedule intelligent payment retry"""
		try:
			max_retries = retry_strategy.get('max_retries', 3)
			retry_intervals = retry_strategy.get('retry_intervals', [1, 24, 72])
			
			# Get current retry count
			current_retries = payment.metadata.get('retry_count', 0) if payment.metadata else 0
			
			if current_retries >= max_retries:
				# Escalate to dunning
				await self._escalate_to_dunning(payment)
				return
			
			# Schedule next retry
			if current_retries < len(retry_intervals):
				next_retry_hours = retry_intervals[current_retries]
				next_retry_time = datetime.utcnow() + timedelta(hours=next_retry_hours)
				
				# In production, would schedule actual retry
				self.logger.info(f"Scheduled retry for payment {payment.id} at {next_retry_time}")
		
		except Exception as e:
			self.logger.error(f"Retry scheduling failed: {e}")
	
	async def _escalate_to_dunning(self, payment) -> None:
		"""Escalate failed payment to dunning process"""
		try:
			# Get dunning system
			from .dunning_management import get_dunning_management_system
			dunning_system = get_dunning_management_system()
			
			# Create dunning case
			case = await dunning_system.create_dunning_case(
				customer_id=payment.customer_id,
				invoice_id=payment.invoice_id,
				outstanding_amount=payment.amount
			)
			
			self.logger.info(f"Escalated payment {payment.id} to dunning case {case.id}")
			
		except Exception as e:
			self.logger.error(f"Dunning escalation failed: {e}")
	
	# Public API methods
	
	async def get_customer_journey(self, customer_id: str, days: int = 30) -> List[JourneyTouchpoint]:
		"""Get customer's financial journey touchpoints"""
		touchpoint_ids = self.customer_journeys.get(customer_id, [])
		touchpoints = [self.touchpoints[tid] for tid in touchpoint_ids if tid in self.touchpoints]
		
		# Filter by date range
		cutoff_date = datetime.utcnow() - timedelta(days=days)
		recent_touchpoints = [tp for tp in touchpoints if tp.timestamp >= cutoff_date]
		
		return sorted(recent_touchpoints, key=lambda tp: tp.timestamp, reverse=True)
	
	async def get_customer_health_score(self, customer_id: str) -> Dict[str, Any]:
		"""Get detailed customer health score"""
		overall_score = self.customer_health_scores.get(customer_id, 0.5)
		
		# Get component scores
		payment_health = await self._calculate_payment_health(customer_id)
		engagement_health = await self._calculate_engagement_health(customer_id)
		growth_health = await self._calculate_growth_health(customer_id)
		support_health = await self._calculate_support_health(customer_id)
		
		# Determine health status
		if overall_score >= 0.8:
			status = 'excellent'
		elif overall_score >= 0.6:
			status = 'good'
		elif overall_score >= 0.4:
			status = 'fair'
		else:
			status = 'poor'
		
		return {
			'customer_id': customer_id,
			'overall_score': overall_score,
			'status': status,
			'components': {
				'payment_health': payment_health,
				'engagement_health': engagement_health,
				'growth_health': growth_health,
				'support_health': support_health
			},
			'calculated_at': datetime.utcnow().isoformat()
		}
	
	async def trigger_manual_intervention(self, customer_id: str, intervention_type: str, context: Dict[str, Any]) -> bool:
		"""Trigger manual intervention in customer journey"""
		try:
			# Create intervention touchpoint
			touchpoint_data = {
				'customer_id': customer_id,
				'journey_stage': JourneyStage.RETENTION.value,
				'touchpoint_type': TouchpointType.SUPPORT_INTERACTION.value,
				'context': context,
				'automation_level': 'manual',
				'metadata': {
					'intervention_type': intervention_type,
					'triggered_by': 'manual'
				}
			}
			
			touchpoint = JourneyTouchpoint(touchpoint_data)
			self.touchpoints[touchpoint.id] = touchpoint
			
			# Add to customer journey
			if customer_id not in self.customer_journeys:
				self.customer_journeys[customer_id] = []
			self.customer_journeys[customer_id].append(touchpoint.id)
			
			# Log intervention
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.USER_ACCESS_GRANTED.value,
				'user_id': 'system',
				'resource_type': 'customer_intervention',
				'resource_id': customer_id,
				'action': 'manual_intervention_triggered',
				'description': f'Manual intervention: {intervention_type}',
				'metadata': {
					'intervention_type': intervention_type,
					'context': context
				}
			})
			
			return True
			
		except Exception as e:
			self.logger.error(f"Manual intervention failed: {e}")
			return False
	
	async def get_journey_analytics(self, days: int = 30) -> Dict[str, Any]:
		"""Get journey orchestration analytics"""
		cutoff_date = datetime.utcnow() - timedelta(days=days)
		
		# Filter recent touchpoints
		recent_touchpoints = [
			tp for tp in self.touchpoints.values()
			if tp.timestamp >= cutoff_date
		]
		
		# Calculate metrics
		total_touchpoints = len(recent_touchpoints)
		automated_touchpoints = len([tp for tp in recent_touchpoints if tp.automation_level == 'automated'])
		successful_outcomes = len([tp for tp in recent_touchpoints if tp.outcome == 'success'])
		
		# Touchpoint distribution
		touchpoint_types = {}
		for tp in recent_touchpoints:
			tp_type = tp.touchpoint_type.value
			touchpoint_types[tp_type] = touchpoint_types.get(tp_type, 0) + 1
		
		# Journey stage distribution
		stage_distribution = {}
		for tp in recent_touchpoints:
			stage = tp.journey_stage.value
			stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
		
		# Health score distribution
		health_scores = list(self.customer_health_scores.values())
		avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 0.5
		
		return {
			'period_days': days,
			'total_touchpoints': total_touchpoints,
			'automation_rate': automated_touchpoints / max(total_touchpoints, 1),
			'success_rate': successful_outcomes / max(total_touchpoints, 1),
			'avg_customer_health': avg_health_score,
			'touchpoint_distribution': touchpoint_types,
			'stage_distribution': stage_distribution,
			'customers_monitored': len(self.customer_health_scores),
			'generated_at': datetime.utcnow().isoformat()
		}
	
	async def _get_customer_phone_number(self, customer_id: str) -> Optional[str]:
		"""Get customer phone number for SMS"""
		try:
			customer = self.billing_service.customers.get(customer_id)
			if not customer:
				return None
			
			# Check primary phone number
			phone = getattr(customer, 'phone', None)
			if phone:
				# Normalize phone number
				return self._normalize_phone_number(phone)
			
			# Check metadata for alternative numbers
			if hasattr(customer, 'metadata') and customer.metadata:
				alt_phone = customer.metadata.get('mobile_phone') or customer.metadata.get('sms_phone')
				if alt_phone:
					return self._normalize_phone_number(alt_phone)
			
			return None
			
		except Exception as e:
			self.logger.error(f"Failed to get phone number for customer {customer_id}: {e}")
			return None
	
	def _normalize_phone_number(self, phone: str) -> str:
		"""Normalize phone number to international format"""
		try:
			import re
			
			# Remove all non-digit characters
			digits_only = re.sub(r'[^\d]', '', phone)
			
			# Handle different country formats - prioritize Kenya and Africa
			if digits_only.startswith('254'):  # Kenya
				return f"+{digits_only}"
			elif digits_only.startswith('0') and len(digits_only) == 10:  # Local Kenya format
				return f"+254{digits_only[1:]}"
			elif digits_only.startswith('1') and len(digits_only) == 11:  # US/Canada
				return f"+{digits_only}"
			elif len(digits_only) == 10:  # Assume US domestic
				return f"+1{digits_only}"
			elif len(digits_only) >= 7:  # International format without +
				return f"+{digits_only}"
			else:
				return phone  # Return as-is if can't normalize
				
		except Exception:
			return phone
	
	async def _send_sms_via_service(self, phone_number: str, message: str, 
								   customer_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send SMS via available service providers"""
		try:
			import os
			
			# Truncate message to SMS limits
			if len(message) > 1600:  # Max for concatenated SMS
				message = message[:1597] + "..."
			
			# Determine optimal provider based on phone number
			provider = self._select_sms_provider(phone_number)
			
			if provider == 'africaistalking':
				return await self._send_via_africaistalking(phone_number, message, customer_id, content)
			elif provider == 'rapidpro':
				return await self._send_via_rapidpro(phone_number, message, customer_id, content)
			elif provider == 'twilio':
				return await self._send_via_twilio(phone_number, message, customer_id, content)
			elif provider == 'aws_sns':
				return await self._send_via_aws_sns(phone_number, message, customer_id, content)
			else:
				# Fallback simulation
				self.logger.warning("No SMS provider configured, simulating SMS send")
				self.logger.info(f"SIMULATED SMS - To: {phone_number}, Message: {message[:50]}...")
				return {
					'success': True,
					'provider': 'simulation',
					'message_id': f"sim_{uuid7str()}",
					'segments': len(message) // 160 + 1
				}
				
		except Exception as e:
			self.logger.error(f"SMS sending failed: {e}")
			return {'success': False, 'error': str(e), 'provider': 'unknown'}
	
	def _select_sms_provider(self, phone_number: str) -> str:
		"""Select best SMS provider based on phone number and availability"""
		import os
		
		# African numbers - prefer AfricaIsTalking
		if phone_number.startswith('+254') or phone_number.startswith('+255') or phone_number.startswith('+256'):
			if os.getenv('AFRICAISTALKING_API_KEY'):
				return 'africaistalking'
		
		# RapidPro for NGO/humanitarian contexts
		if os.getenv('RAPIDPRO_API_TOKEN'):
			return 'rapidpro'
		
		# Twilio for global coverage
		if os.getenv('TWILIO_ACCOUNT_SID') and os.getenv('TWILIO_AUTH_TOKEN'):
			return 'twilio'
		
		# AWS SNS as fallback
		if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
			return 'aws_sns'
		
		return 'simulation'
	
	async def _send_via_africaistalking(self, phone_number: str, message: str, 
									   customer_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send SMS via Africa's Talking"""
		try:
			import os
			import aiohttp
			
			api_key = os.getenv('AFRICAISTALKING_API_KEY')
			username = os.getenv('AFRICAISTALKING_USERNAME', 'sandbox')
			sender_id = os.getenv('AFRICAISTALKING_SENDER_ID', 'APG')
			
			if not api_key:
				raise Exception("Africa's Talking API key not configured")
			
			url = "https://api.africastalking.com/version1/messaging"
			
			headers = {
				'apiKey': api_key,
				'Content-Type': 'application/x-www-form-urlencoded',
				'Accept': 'application/json'
			}
			
			data = {
				'username': username,
				'to': phone_number,
				'message': message,
				'from': sender_id
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(url, headers=headers, data=data) as response:
					result = await response.json()
					
					if response.status == 201 and result.get('SMSMessageData'):
						sms_data = result['SMSMessageData']
						recipients = sms_data.get('Recipients', [])
						
						if recipients and recipients[0].get('status') == 'Success':
							return {
								'success': True,
								'provider': 'africaistalking',
								'message_id': recipients[0].get('messageId'),
								'cost': recipients[0].get('cost'),
								'segments': len(message) // 160 + 1,
								'status': 'sent'
							}
						else:
							error_msg = recipients[0].get('status', 'Unknown error') if recipients else 'No recipients'
							return {
								'success': False,
								'provider': 'africaistalking',
								'error': error_msg
							}
					else:
						return {
							'success': False,
							'provider': 'africaistalking',
							'error': result.get('SMSMessageData', {}).get('Message', 'API request failed')
						}
			
		except Exception as e:
			self.logger.error(f"Africa's Talking SMS failed: {e}")
			return {'success': False, 'provider': 'africaistalking', 'error': str(e)}
	
	async def _send_via_rapidpro(self, phone_number: str, message: str, 
								customer_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send SMS via RapidPro"""
		try:
			import os
			import aiohttp
			
			api_token = os.getenv('RAPIDPRO_API_TOKEN')
			rapidpro_url = os.getenv('RAPIDPRO_URL', 'https://app.rapidpro.io')
			
			if not api_token:
				raise Exception("RapidPro API token not configured")
			
			url = f"{rapidpro_url}/api/v2/broadcasts.json"
			
			headers = {
				'Authorization': f'Token {api_token}',
				'Content-Type': 'application/json'
			}
			
			payload = {
				'urns': [f'tel:{phone_number}'],
				'text': message
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(url, headers=headers, json=payload) as response:
					if response.status == 201:
						result = await response.json()
						return {
							'success': True,
							'provider': 'rapidpro',
							'message_id': result.get('id'),
							'broadcast_id': result.get('id'),
							'segments': len(message) // 160 + 1,
							'status': 'queued'
						}
					else:
						error_text = await response.text()
						return {
							'success': False,
							'provider': 'rapidpro',
							'error': f"HTTP {response.status}: {error_text}"
						}
			
		except Exception as e:
			self.logger.error(f"RapidPro SMS failed: {e}")
			return {'success': False, 'provider': 'rapidpro', 'error': str(e)}
	
	async def _send_via_twilio(self, phone_number: str, message: str, 
							  customer_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send SMS via Twilio"""
		try:
			import os
			from twilio.rest import Client
			
			account_sid = os.getenv('TWILIO_ACCOUNT_SID')
			auth_token = os.getenv('TWILIO_AUTH_TOKEN')
			from_number = os.getenv('TWILIO_PHONE_NUMBER')
			
			if not all([account_sid, auth_token, from_number]):
				raise Exception("Twilio credentials not configured")
			
			client = Client(account_sid, auth_token)
			
			twilio_message = client.messages.create(
				body=message,
				from_=from_number,
				to=phone_number
			)
			
			return {
				'success': True,
				'provider': 'twilio',
				'message_id': twilio_message.sid,
				'status': twilio_message.status,
				'segments': twilio_message.num_segments,
				'price': twilio_message.price,
				'price_unit': twilio_message.price_unit
			}
			
		except Exception as e:
			self.logger.error(f"Twilio SMS failed: {e}")
			return {'success': False, 'provider': 'twilio', 'error': str(e)}
	
	async def _send_via_aws_sns(self, phone_number: str, message: str, 
							   customer_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send SMS via AWS SNS"""
		try:
			import os
			import boto3
			
			aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
			aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
			aws_region = os.getenv('AWS_REGION', 'us-east-1')
			
			if not all([aws_access_key, aws_secret_key]):
				raise Exception("AWS credentials not configured")
			
			sns = boto3.client(
				'sns',
				aws_access_key_id=aws_access_key,
				aws_secret_access_key=aws_secret_key,
				region_name=aws_region
			)
			
			response = sns.publish(
				PhoneNumber=phone_number,
				Message=message,
				MessageAttributes={
					'AWS.SNS.SMS.SMSType': {
						'DataType': 'String',
						'StringValue': 'Transactional'
					},
					'customer_id': {
						'DataType': 'String',
						'StringValue': customer_id
					}
				}
			)
			
			return {
				'success': True,
				'provider': 'aws_sns',
				'message_id': response['MessageId'],
				'segments': len(message) // 160 + 1,
				'status': 'sent'
			}
			
		except Exception as e:
			self.logger.error(f"AWS SNS SMS failed: {e}")
			return {'success': False, 'provider': 'aws_sns', 'error': str(e)}
	
	async def _store_communication_record(self, delivery_record: Dict[str, Any]) -> None:
		"""Store communication delivery record"""
		try:
			# Store in billing service
			if not hasattr(self.billing_service, 'communication_records'):
				self.billing_service.communication_records = {}
			
			record_id = uuid7str()
			delivery_record['id'] = record_id
			self.billing_service.communication_records[record_id] = delivery_record
			
			# Update customer communication history
			customer_id = delivery_record['customer_id']
			customer = self.billing_service.customers.get(customer_id)
			if customer:
				if not hasattr(customer, 'metadata'):
					customer.metadata = {}
				
				comm_history = customer.metadata.get('communication_history', [])
				comm_history.append({
					'record_id': record_id,
					'channel': delivery_record['channel'],
					'sent_at': delivery_record['sent_at'],
					'status': delivery_record['status'],
					'provider': delivery_record.get('provider')
				})
				
				# Keep last 50 communications
				customer.metadata['communication_history'] = comm_history[-50:]
			
		except Exception as e:
			self.logger.error(f"Failed to store communication record: {e}")
	
	async def _get_customer_device_tokens(self, customer_id: str) -> List[Dict[str, Any]]:
		"""Get customer device tokens for push notifications"""
		try:
			customer = self.billing_service.customers.get(customer_id)
			if not customer:
				return []
			
			device_tokens = []
			
			# Get device tokens from customer metadata
			if hasattr(customer, 'metadata') and customer.metadata:
				stored_devices = customer.metadata.get('device_tokens', [])
				
				for device in stored_devices:
					if device.get('active', True) and device.get('token'):
						device_tokens.append(device)
			
			return device_tokens
			
		except Exception as e:
			self.logger.error(f"Failed to get device tokens for customer {customer_id}: {e}")
			return []
	
	async def _send_push_via_service(self, device_tokens: List[Dict[str, Any]], 
									push_content: Dict[str, Any], customer_id: str, 
									content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send push notifications via available service providers"""
		try:
			import os
			
			# Determine optimal provider
			provider = self._select_push_provider()
			
			if provider == 'firebase':
				return await self._send_push_via_firebase(device_tokens, push_content, customer_id, content)
			elif provider == 'aws_sns':
				return await self._send_push_via_aws_sns_mobile(device_tokens, push_content, customer_id, content)
			elif provider == 'apns':
				return await self._send_push_via_apns(device_tokens, push_content, customer_id, content)
			else:
				# Fallback simulation
				self.logger.warning("No push notification provider configured, simulating push send")
				self.logger.info(f"SIMULATED PUSH - To: {len(device_tokens)} devices, Title: {push_content['title']}")
				return {
					'success': True,
					'provider': 'simulation',
					'message_id': f"push_sim_{uuid7str()}",
					'delivered_count': len(device_tokens),
					'failed_count': 0
				}
				
		except Exception as e:
			self.logger.error(f"Push notification sending failed: {e}")
			return {
				'success': False, 
				'error': str(e), 
				'provider': 'unknown',
				'delivered_count': 0,
				'failed_count': len(device_tokens)
			}
	
	def _select_push_provider(self) -> str:
		"""Select best push notification provider based on availability"""
		import os
		
		# Firebase Cloud Messaging (cross-platform)
		if os.getenv('FIREBASE_CREDENTIALS_PATH') or os.getenv('FIREBASE_CREDENTIALS_JSON'):
			return 'firebase'
		
		# AWS SNS Mobile Push
		if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
			return 'aws_sns'
		
		# Apple Push Notification Service (iOS only)
		if os.getenv('APNS_KEY_PATH') or os.getenv('APNS_PRIVATE_KEY'):
			return 'apns'
		
		return 'simulation'
	
	async def _send_push_via_firebase(self, device_tokens: List[Dict[str, Any]], 
									 push_content: Dict[str, Any], customer_id: str, 
									 content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send push notifications via Firebase Cloud Messaging"""
		try:
			import os
			import json
			from firebase_admin import credentials, messaging
			import firebase_admin
			
			# Initialize Firebase if not already done
			if not firebase_admin._apps:
				# Try credentials from file
				cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
				if cred_path:
					cred = credentials.Certificate(cred_path)
				else:
					# Try credentials from environment variable (JSON string)
					cred_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
					if cred_json:
						cred_dict = json.loads(cred_json)
						cred = credentials.Certificate(cred_dict)
					else:
						raise Exception("Firebase credentials not configured")
				
				firebase_admin.initialize_app(cred)
			
			delivered_count = 0
			failed_count = 0
			message_id = None
			
			# Prepare notification
			notification = messaging.Notification(
				title=push_content['title'],
				body=push_content['body']
			)
			
			# Prepare data payload
			data_payload = {
				'customer_id': customer_id,
				'type': 'billing_notification',
				**push_content.get('data', {})
			}
			
			# Convert all data values to strings (FCM requirement)
			data_payload = {k: str(v) for k, v in data_payload.items()}
			
			# Group tokens by platform for optimized sending
			tokens_by_platform = {}
			for device in device_tokens:
				platform = device.get('platform', 'unknown')
				if platform not in tokens_by_platform:
					tokens_by_platform[platform] = []
				tokens_by_platform[platform].append(device['token'])
			
			# Send to each platform group
			for platform, tokens in tokens_by_platform.items():
				try:
					# Configure platform-specific options
					android_config = messaging.AndroidConfig(
						priority='high',
						notification=messaging.AndroidNotification(
							channel_id='billing_notifications',
							priority='high'
						)
					) if platform == 'android' else None
					
					apns_config = messaging.APNSConfig(
						payload=messaging.APNSPayload(
							aps=messaging.Aps(
								alert=messaging.ApsAlert(
									title=push_content['title'],
									body=push_content['body']
								),
								badge=1,
								sound='default'
							)
						)
					) if platform == 'ios' else None
					
					# Create message
					message = messaging.MulticastMessage(
						notification=notification,
						data=data_payload,
						tokens=tokens,
						android=android_config,
						apns=apns_config
					)
					
					# Send message
					response = messaging.send_multicast(message)
					
					delivered_count += response.success_count
					failed_count += response.failure_count
					
					if response.responses:
						message_id = response.responses[0].message_id if response.responses[0].success else None
					
					# Log individual failures for debugging
					for idx, resp in enumerate(response.responses):
						if not resp.success:
							self.logger.warning(f"Push failed for token {tokens[idx][:10]}...: {resp.exception}")
				
				except Exception as platform_error:
					self.logger.error(f"Firebase push failed for platform {platform}: {platform_error}")
					failed_count += len(tokens)
			
			return {
				'success': delivered_count > 0,
				'provider': 'firebase',
				'message_id': message_id,
				'delivered_count': delivered_count,
				'failed_count': failed_count,
				'status': 'sent'
			}
			
		except Exception as e:
			self.logger.error(f"Firebase push notification failed: {e}")
			return {
				'success': False, 
				'provider': 'firebase', 
				'error': str(e),
				'delivered_count': 0,
				'failed_count': len(device_tokens)
			}
	
	async def _send_push_via_aws_sns_mobile(self, device_tokens: List[Dict[str, Any]], 
										   push_content: Dict[str, Any], customer_id: str, 
										   content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send push notifications via AWS SNS Mobile Push"""
		try:
			import os
			import boto3
			import json
			
			aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
			aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
			aws_region = os.getenv('AWS_REGION', 'us-east-1')
			
			if not all([aws_access_key, aws_secret_key]):
				raise Exception("AWS credentials not configured")
			
			sns = boto3.client(
				'sns',
				aws_access_key_id=aws_access_key,
				aws_secret_access_key=aws_secret_key,
				region_name=aws_region
			)
			
			delivered_count = 0
			failed_count = 0
			message_id = None
			
			# Prepare message payload for different platforms
			message_payload = {
				'default': push_content['body'],
				'GCM': json.dumps({
					'notification': {
						'title': push_content['title'],
						'body': push_content['body']
					},
					'data': {
						'customer_id': customer_id,
						**push_content.get('data', {})
					}
				}),
				'APNS': json.dumps({
					'aps': {
						'alert': {
							'title': push_content['title'],
							'body': push_content['body']
						},
						'badge': 1,
						'sound': 'default'
					},
					'customer_id': customer_id,
					**push_content.get('data', {})
				})
			}
			
			# Send to each device endpoint
			for device in device_tokens:
				try:
					endpoint_arn = device.get('endpoint_arn')
					if not endpoint_arn:
						self.logger.warning(f"No endpoint ARN for device {device.get('device_id', 'unknown')}")
						failed_count += 1
						continue
					
					response = sns.publish(
						TargetArn=endpoint_arn,
						Message=json.dumps(message_payload),
						MessageStructure='json',
						MessageAttributes={
							'customer_id': {
								'DataType': 'String',
								'StringValue': customer_id
							}
						}
					)
					
					delivered_count += 1
					if not message_id:  # Store first successful message ID
						message_id = response['MessageId']
				
				except Exception as device_error:
					self.logger.warning(f"AWS SNS push failed for device {device.get('device_id', 'unknown')}: {device_error}")
					failed_count += 1
			
			return {
				'success': delivered_count > 0,
				'provider': 'aws_sns',
				'message_id': message_id,
				'delivered_count': delivered_count,
				'failed_count': failed_count,
				'status': 'sent'
			}
			
		except Exception as e:
			self.logger.error(f"AWS SNS push notification failed: {e}")
			return {
				'success': False, 
				'provider': 'aws_sns', 
				'error': str(e),
				'delivered_count': 0,
				'failed_count': len(device_tokens)
			}
	
	async def _send_push_via_apns(self, device_tokens: List[Dict[str, Any]], 
								 push_content: Dict[str, Any], customer_id: str, 
								 content: Dict[str, Any]) -> Dict[str, Any]:
		"""Send push notifications via Apple Push Notification Service (APNS)"""
		try:
			import os
			from aioapns import APNs, NotificationRequest, PushType
			
			# Get APNS credentials
			key_path = os.getenv('APNS_KEY_PATH')
			private_key = os.getenv('APNS_PRIVATE_KEY')
			key_id = os.getenv('APNS_KEY_ID')
			team_id = os.getenv('APNS_TEAM_ID')
			bundle_id = os.getenv('APNS_BUNDLE_ID', 'com.datacraft.apg')
			
			if not all([key_id, team_id, bundle_id]) or not (key_path or private_key):
				raise Exception("APNS credentials not configured")
			
			# Initialize APNS client
			if key_path:
				apns = APNs(
					key=key_path,
					key_id=key_id,
					team_id=team_id,
					topic=bundle_id,
					use_sandbox=os.getenv('APNS_USE_SANDBOX', 'false').lower() == 'true'
				)
			else:
				apns = APNs(
					private_key=private_key,
					key_id=key_id,
					team_id=team_id,
					topic=bundle_id,
					use_sandbox=os.getenv('APNS_USE_SANDBOX', 'false').lower() == 'true'
				)
			
			delivered_count = 0
			failed_count = 0
			
			# Filter iOS devices only
			ios_devices = [device for device in device_tokens if device.get('platform') == 'ios']
			
			if not ios_devices:
				return {
					'success': False,
					'provider': 'apns',
					'error': 'No iOS devices found',
					'delivered_count': 0,
					'failed_count': len(device_tokens)
				}
			
			# Send to each iOS device
			for device in ios_devices:
				try:
					device_token = device.get('token')
					if not device_token:
						failed_count += 1
						continue
					
					# Create notification request
					request = NotificationRequest(
						device_token=device_token,
						message={
							'aps': {
								'alert': {
									'title': push_content['title'],
									'body': push_content['body']
								},
								'badge': 1,
								'sound': 'default'
							},
							'customer_id': customer_id,
							**push_content.get('data', {})
						},
						push_type=PushType.ALERT
					)
					
					# Send notification
					await apns.send_notification(request)
					delivered_count += 1
				
				except Exception as device_error:
					self.logger.warning(f"APNS push failed for device {device.get('device_id', 'unknown')}: {device_error}")
					failed_count += 1
			
			return {
				'success': delivered_count > 0,
				'provider': 'apns',
				'message_id': f"apns_{uuid7str()}",
				'delivered_count': delivered_count,
				'failed_count': failed_count,
				'status': 'sent'
			}
			
		except Exception as e:
			self.logger.error(f"APNS push notification failed: {e}")
			return {
				'success': False, 
				'provider': 'apns', 
				'error': str(e),
				'delivered_count': 0,
				'failed_count': len(device_tokens)
			}
	
	async def _store_in_app_notification(self, customer_id: str, notification_content: Dict[str, Any], 
										original_content: Dict[str, Any]) -> Dict[str, Any]:
		"""Store in-app notification in database"""
		try:
			notification_id = uuid7str()
			
			# Create comprehensive notification record
			notification_record = {
				'id': notification_id,
				'customer_id': customer_id,
				'title': notification_content['title'],
				'message': notification_content['message'],
				'type': notification_content['type'],
				'priority': notification_content['priority'],
				'action_url': notification_content.get('action_url'),
				'created_at': notification_content['created_at'],
				'read': False,
				'read_at': None,
				'dismissed': False,
				'dismissed_at': None,
				'expires_at': self._calculate_notification_expiry(notification_content),
				'metadata': {
					'source': 'financial_journey_orchestrator',
					'original_content_type': original_content.get('template_type', 'system'),
					'channel': 'in_app',
					**notification_content.get('metadata', {})
				},
				'actions': self._generate_notification_actions(notification_content, original_content),
				'tags': self._generate_notification_tags(notification_content, original_content)
			}
			
			# Store in billing service notifications
			if not hasattr(self.billing_service, 'in_app_notifications'):
				self.billing_service.in_app_notifications = {}
			
			self.billing_service.in_app_notifications[notification_id] = notification_record
			
			# Update customer's notification inbox
			await self._update_customer_notification_inbox(customer_id, notification_record)
			
			# Check for notification limits and cleanup
			await self._cleanup_old_notifications(customer_id)
			
			self.logger.info(f"📥 In-app notification stored: {notification_id} for customer {customer_id}")
			
			return {
				'success': True,
				'notification_id': notification_id,
				'expires_at': notification_record['expires_at']
			}
			
		except Exception as e:
			self.logger.error(f"Failed to store in-app notification: {e}")
			return {
				'success': False,
				'error': str(e)
			}
	
	def _calculate_notification_expiry(self, notification_content: Dict[str, Any]) -> str:
		"""Calculate when the notification should expire"""
		try:
			notification_type = notification_content.get('type', 'info')
			priority = notification_content.get('priority', 'normal')
			
			# Different expiry times based on type and priority
			expiry_days = {
				('urgent', 'high'): 7,      # Urgent notifications last 1 week
				('urgent', 'normal'): 5,    # Urgent normal priority
				('warning', 'high'): 14,    # Warning high priority lasts 2 weeks
				('warning', 'normal'): 10,  # Warning normal priority
				('info', 'high'): 21,       # Info high priority lasts 3 weeks
				('info', 'normal'): 30,     # Info normal priority lasts 1 month
				('success', 'high'): 7,     # Success messages last 1 week
				('success', 'normal'): 3,   # Success normal last 3 days
			}
			
			days = expiry_days.get((notification_type, priority), 30)  # Default 30 days
			
			expiry_date = datetime.utcnow() + timedelta(days=days)
			return expiry_date.isoformat()
			
		except Exception:
			# Fallback to 30 days
			return (datetime.utcnow() + timedelta(days=30)).isoformat()
	
	def _generate_notification_actions(self, notification_content: Dict[str, Any], 
									  original_content: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate actionable buttons/links for the notification"""
		try:
			actions = []
			
			# Primary action from action_url
			if notification_content.get('action_url'):
				actions.append({
					'type': 'primary',
					'label': original_content.get('action_label', 'View Details'),
					'url': notification_content['action_url'],
					'method': 'GET'
				})
			
			# Type-specific actions
			notification_type = notification_content.get('type', 'info')
			
			if notification_type == 'payment_reminder':
				actions.extend([
					{
						'type': 'primary',
						'label': 'Pay Now',
						'url': '/billing/pay',
						'method': 'GET'
					},
					{
						'type': 'secondary',
						'label': 'Update Payment Method',
						'url': '/billing/payment-methods',
						'method': 'GET'
					}
				])
			elif notification_type == 'subscription_expiring':
				actions.extend([
					{
						'type': 'primary',
						'label': 'Renew Subscription',
						'url': '/billing/renew',
						'method': 'GET'
					},
					{
						'type': 'secondary',
						'label': 'View Plans',
						'url': '/billing/plans',
						'method': 'GET'
					}
				])
			elif notification_type == 'invoice_available':
				actions.append({
					'type': 'primary',
					'label': 'View Invoice',
					'url': f"/billing/invoices/{original_content.get('invoice_id', '')}",
					'method': 'GET'
				})
			
			# Always add dismiss action
			actions.append({
				'type': 'dismiss',
				'label': 'Dismiss',
				'action': 'dismiss_notification',
				'method': 'POST'
			})
			
			return actions[:4]  # Limit to 4 actions max
			
		except Exception as e:
			self.logger.warning(f"Failed to generate notification actions: {e}")
			return [{'type': 'dismiss', 'label': 'Dismiss', 'action': 'dismiss_notification'}]
	
	def _generate_notification_tags(self, notification_content: Dict[str, Any], 
								   original_content: Dict[str, Any]) -> List[str]:
		"""Generate searchable tags for the notification"""
		try:
			tags = []
			
			# Type-based tags
			notification_type = notification_content.get('type', 'info')
			tags.append(notification_type)
			
			# Priority-based tags
			priority = notification_content.get('priority', 'normal')
			if priority == 'high':
				tags.append('high_priority')
			elif priority == 'urgent':
				tags.extend(['urgent', 'high_priority'])
			
			# Content-based tags
			title_lower = notification_content.get('title', '').lower()
			message_lower = notification_content.get('message', '').lower()
			
			if any(word in title_lower or word in message_lower for word in ['payment', 'pay', 'invoice']):
				tags.append('billing')
			if any(word in title_lower or word in message_lower for word in ['subscription', 'plan', 'renew']):
				tags.append('subscription')
			if any(word in title_lower or word in message_lower for word in ['trial', 'expire', 'ending']):
				tags.append('trial')
			if any(word in title_lower or word in message_lower for word in ['failed', 'declined', 'error']):
				tags.append('error')
			if any(word in title_lower or word in message_lower for word in ['success', 'completed', 'confirmed']):
				tags.append('success')
			
			# Template-based tags
			template_type = original_content.get('template_type')
			if template_type:
				tags.append(f"template_{template_type}")
			
			return list(set(tags))  # Remove duplicates
			
		except Exception as e:
			self.logger.warning(f"Failed to generate notification tags: {e}")
			return ['general']
	
	async def _update_customer_notification_inbox(self, customer_id: str, notification_record: Dict[str, Any]) -> None:
		"""Update customer's notification inbox metadata"""
		try:
			customer = self.billing_service.customers.get(customer_id)
			if not customer:
				return
			
			if not hasattr(customer, 'metadata'):
				customer.metadata = {}
			
			# Update notification inbox summary
			inbox_summary = customer.metadata.get('notification_inbox', {
				'total_count': 0,
				'unread_count': 0,
				'last_notification_at': None,
				'high_priority_count': 0,
				'recent_notifications': []
			})
			
			# Update counts
			inbox_summary['total_count'] += 1
			inbox_summary['unread_count'] += 1
			inbox_summary['last_notification_at'] = notification_record['created_at']
			
			if notification_record['priority'] in ['high', 'urgent']:
				inbox_summary['high_priority_count'] += 1
			
			# Add to recent notifications (keep last 10)
			recent_notifications = inbox_summary.get('recent_notifications', [])
			recent_notifications.insert(0, {
				'id': notification_record['id'],
				'title': notification_record['title'],
				'type': notification_record['type'],
				'priority': notification_record['priority'],
				'created_at': notification_record['created_at'],
				'read': False
			})
			inbox_summary['recent_notifications'] = recent_notifications[:10]
			
			customer.metadata['notification_inbox'] = inbox_summary
			
		except Exception as e:
			self.logger.error(f"Failed to update customer notification inbox: {e}")
	
	async def _cleanup_old_notifications(self, customer_id: str) -> None:
		"""Clean up old and expired notifications for a customer"""
		try:
			if not hasattr(self.billing_service, 'in_app_notifications'):
				return
			
			current_time = datetime.utcnow()
			notifications_to_remove = []
			
			# Find customer's notifications
			customer_notifications = [
				(nid, notification) for nid, notification in self.billing_service.in_app_notifications.items()
				if notification['customer_id'] == customer_id
			]
			
			# Check for expired notifications
			for notification_id, notification in customer_notifications:
				expires_at = datetime.fromisoformat(notification['expires_at'])
				if current_time > expires_at:
					notifications_to_remove.append(notification_id)
			
			# Check notification count limits (keep last 100 per customer)
			if len(customer_notifications) > 100:
				# Sort by created_at and remove oldest
				sorted_notifications = sorted(
					customer_notifications, 
					key=lambda x: datetime.fromisoformat(x[1]['created_at']),
					reverse=True
				)
				
				for notification_id, _ in sorted_notifications[100:]:  # Remove beyond 100
					notifications_to_remove.append(notification_id)
			
			# Remove notifications
			for notification_id in notifications_to_remove:
				if notification_id in self.billing_service.in_app_notifications:
					del self.billing_service.in_app_notifications[notification_id]
			
			if notifications_to_remove:
				self.logger.info(f"Cleaned up {len(notifications_to_remove)} old notifications for customer {customer_id}")
			
		except Exception as e:
			self.logger.error(f"Failed to cleanup old notifications: {e}")


# Global financial journey orchestrator
_journey_orchestrator_instance: Optional[FinancialJourneyOrchestrator] = None

def get_financial_journey_orchestrator() -> FinancialJourneyOrchestrator:
	"""Get global financial journey orchestrator instance"""
	global _journey_orchestrator_instance
	if _journey_orchestrator_instance is None:
		_journey_orchestrator_instance = FinancialJourneyOrchestrator()
	return _journey_orchestrator_instance


__all__ = [
	'FinancialJourneyOrchestrator',
	'SmartPaymentRouter',
	'JourneyTouchpoint',
	'JourneyOrchestration',
	'JourneyStage',
	'TouchpointType',
	'CommunicationChannel',
	'get_financial_journey_orchestrator'
]