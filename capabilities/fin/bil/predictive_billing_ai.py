"""
APG Predictive Billing Intelligence

AI-powered billing intelligence that predicts and prevents revenue leakage,
involuntary churn, and billing failures before they occur. Uses advanced ML
models to provide autonomous billing optimization and proactive problem resolution.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from uuid_extensions import uuid7str
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .models import BLCustomer, BLSubscription, BLPayment, BLInvoice, BLUsage
from .service import get_billing_service
from .audit_compliance import get_audit_compliance_system, AuditEventType


class PredictionType(Enum):
	"""Types of billing predictions"""
	REVENUE_LEAKAGE = "revenue_leakage"
	PAYMENT_FAILURE = "payment_failure"
	INVOLUNTARY_CHURN = "involuntary_churn"
	USAGE_ANOMALY = "usage_anomaly"
	PRICING_OPTIMIZATION = "pricing_optimization"
	COLLECTION_RISK = "collection_risk"
	FRAUD_DETECTION = "fraud_detection"


class BillingPrediction:
	"""Individual billing prediction with confidence and recommended actions"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.prediction_type = PredictionType(data['prediction_type'])
		self.customer_id = data.get('customer_id')
		self.subscription_id = data.get('subscription_id')
		self.confidence_score = data['confidence_score']  # 0.0 to 1.0
		self.risk_score = data['risk_score']  # 0.0 to 1.0
		self.predicted_impact = Decimal(str(data.get('predicted_impact', 0)))  # Financial impact
		self.time_horizon = data.get('time_horizon', 30)  # Days
		self.predicted_at = datetime.fromisoformat(data.get('predicted_at', datetime.utcnow().isoformat()))
		self.expires_at = self.predicted_at + timedelta(days=self.time_horizon)
		self.status = data.get('status', 'active')  # active, resolved, expired
		self.evidence = data.get('evidence', {})
		self.recommended_actions = data.get('recommended_actions', [])
		self.automated_actions_taken = data.get('automated_actions_taken', [])
		self.metadata = data.get('metadata', {})


class PredictiveBillingAI:
	"""AI-powered predictive billing intelligence system"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.PredictiveBillingAI")
		
		# AI Models
		self.models = {
			'revenue_leakage': RandomForestRegressor(n_estimators=100, random_state=42),
			'payment_failure': RandomForestRegressor(n_estimators=100, random_state=42),
			'involuntary_churn': RandomForestRegressor(n_estimators=100, random_state=42),
			'usage_anomaly': IsolationForest(contamination=0.1, random_state=42),
			'fraud_detection': IsolationForest(contamination=0.05, random_state=42)
		}
		
		# Feature preprocessing
		self.scalers = {model_name: StandardScaler() for model_name in self.models.keys()}
		self.customer_segmentation = KMeans(n_clusters=5, random_state=42)
		
		# Prediction storage
		self.predictions: Dict[str, BillingPrediction] = {}
		self.model_performance = {}
		
		# Configuration
		self.prediction_thresholds = {
			PredictionType.REVENUE_LEAKAGE: 0.7,
			PredictionType.PAYMENT_FAILURE: 0.8,
			PredictionType.INVOLUNTARY_CHURN: 0.75,
			PredictionType.USAGE_ANOMALY: 0.6,
			PredictionType.FRAUD_DETECTION: 0.9
		}
		
		# Integration services
		self.billing_service = get_billing_service()
		self.audit_system = get_audit_compliance_system()
		
		# Background processing
		self.prediction_queue: asyncio.Queue = asyncio.Queue()
		asyncio.create_task(self._start_prediction_engine())
		asyncio.create_task(self._start_continuous_monitoring())
	
	async def _start_prediction_engine(self) -> None:
		"""Start background prediction processing"""
		while True:
			try:
				await self._run_prediction_cycle()
				await asyncio.sleep(1800)  # Run every 30 minutes
			except Exception as e:
				self.logger.error(f"Prediction engine error: {e}")
				await asyncio.sleep(1800)
	
	async def _start_continuous_monitoring(self) -> None:
		"""Start continuous monitoring for real-time predictions"""
		while True:
			try:
				await self._monitor_real_time_events()
				await asyncio.sleep(60)  # Check every minute
			except Exception as e:
				self.logger.error(f"Continuous monitoring error: {e}")
				await asyncio.sleep(60)
	
	async def _run_prediction_cycle(self) -> None:
		"""Run complete prediction cycle for all customers"""
		try:
			# Get all active customers
			active_customers = [
				customer for customer in self.billing_service.customers.values()
				if self._is_customer_active(customer)
			]
			
			self.logger.info(f"Running prediction cycle for {len(active_customers)} customers")
			
			for customer in active_customers:
				await self._generate_customer_predictions(customer.id)
			
			# Clean up expired predictions
			await self._cleanup_expired_predictions()
			
		except Exception as e:
			self.logger.error(f"Prediction cycle failed: {e}")
	
	def _is_customer_active(self, customer: BLCustomer) -> bool:
		"""Check if customer is active and should be monitored"""
		# Check if customer has active subscriptions
		active_subscriptions = [
			sub for sub in self.billing_service.subscriptions.values()
			if sub.customer_id == customer.id and sub.status.value == 'active'
		]
		return len(active_subscriptions) > 0
	
	async def _generate_customer_predictions(self, customer_id: str) -> List[BillingPrediction]:
		"""Generate all predictions for a customer"""
		predictions = []
		
		try:
			# Extract customer features
			features = await self._extract_customer_features(customer_id)
			if not features:
				return predictions
			
			# Revenue leakage prediction
			revenue_prediction = await self._predict_revenue_leakage(customer_id, features)
			if revenue_prediction:
				predictions.append(revenue_prediction)
			
			# Payment failure prediction
			payment_prediction = await self._predict_payment_failure(customer_id, features)
			if payment_prediction:
				predictions.append(payment_prediction)
			
			# Involuntary churn prediction
			churn_prediction = await self._predict_involuntary_churn(customer_id, features)
			if churn_prediction:
				predictions.append(churn_prediction)
			
			# Usage anomaly detection
			usage_prediction = await self._detect_usage_anomalies(customer_id, features)
			if usage_prediction:
				predictions.append(usage_prediction)
			
			# Fraud detection
			fraud_prediction = await self._detect_fraud_patterns(customer_id, features)
			if fraud_prediction:
				predictions.append(fraud_prediction)
			
			# Store predictions
			for prediction in predictions:
				self.predictions[prediction.id] = prediction
				await self._process_prediction(prediction)
			
			return predictions
			
		except Exception as e:
			self.logger.error(f"Failed to generate predictions for customer {customer_id}: {e}")
			return predictions
	
	async def _extract_customer_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
		"""Extract comprehensive features for ML predictions"""
		try:
			customer = self.billing_service.customers.get(customer_id)
			if not customer:
				return None
			
			# Get customer data
			subscriptions = [s for s in self.billing_service.subscriptions.values() if s.customer_id == customer_id]
			payments = [p for p in self.billing_service.payments.values() if p.customer_id == customer_id]
			invoices = [i for i in self.billing_service.invoices.values() if i.customer_id == customer_id]
			usage_records = [u for u in self.billing_service.usage_records if u.customer_id == customer_id]
			
			# Calculate temporal features
			now = datetime.utcnow()
			
			# Payment behavior features
			recent_payments = [p for p in payments if (now - p.created_at).days <= 90]
			failed_payments = [p for p in recent_payments if p.status.value == 'failed']
			
			payment_features = {
				'payment_failure_rate': len(failed_payments) / max(len(recent_payments), 1),
				'avg_payment_amount': float(sum(p.amount for p in recent_payments) / max(len(recent_payments), 1)),
				'payment_frequency': len(recent_payments) / 3,  # per month
				'days_since_last_payment': (now - max(p.created_at for p in payments)).days if payments else 999,
				'consecutive_failures': self._count_consecutive_failures(payments),
				'payment_method_changes': len(set(p.payment_method for p in payments if p.payment_method))
			}
			
			# Subscription features
			total_mrr = sum(getattr(s, 'mrr', Decimal('0')) for s in subscriptions)
			subscription_features = {
				'total_mrr': float(total_mrr),
				'subscription_count': len(subscriptions),
				'avg_subscription_age': sum((now - s.created_at).days for s in subscriptions) / max(len(subscriptions), 1),
				'trial_conversions': len([s for s in subscriptions if s.trial_start and s.status.value == 'active']),
				'cancellation_rate': len([s for s in subscriptions if s.status.value == 'cancelled']) / max(len(subscriptions), 1)
			}
			
			# Usage behavior features
			recent_usage = [u for u in usage_records if (now - u.timestamp).days <= 30]
			usage_features = {
				'usage_trend': self._calculate_usage_trend(usage_records),
				'usage_variability': self._calculate_usage_variability(recent_usage),
				'peak_usage_ratio': self._calculate_peak_usage_ratio(recent_usage),
				'days_since_last_usage': (now - max(u.timestamp for u in usage_records)).days if usage_records else 999
			}
			
			# Invoice features
			recent_invoices = [i for i in invoices if (now - i.invoice_date).days <= 90]
			overdue_invoices = [i for i in recent_invoices if i.due_date < now and i.amount_due > 0]
			
			invoice_features = {
				'overdue_amount': float(sum(i.amount_due for i in overdue_invoices)),
				'avg_days_to_pay': self._calculate_avg_days_to_pay(invoices),
				'invoice_dispute_rate': self._calculate_dispute_rate(invoices),
				'collection_efficiency': self._calculate_collection_efficiency(invoices)
			}
			
			# Customer profile features
			profile_features = {
				'customer_age_days': (now - customer.created_at).days,
				'customer_tier': getattr(customer, 'tier', 'standard'),
				'company_size': getattr(customer, 'company_size', 'unknown'),
				'industry': getattr(customer, 'industry', 'unknown'),
				'geography': getattr(customer, 'country', 'unknown')
			}
			
			# Combine all features
			features = {
				**payment_features,
				**subscription_features,
				**usage_features,
				**invoice_features,
				**profile_features
			}
			
			return features
			
		except Exception as e:
			self.logger.error(f"Feature extraction failed for customer {customer_id}: {e}")
			return None
	
	def _count_consecutive_failures(self, payments: List[BLPayment]) -> int:
		"""Count consecutive payment failures"""
		if not payments:
			return 0
		
		# Sort by date descending
		sorted_payments = sorted(payments, key=lambda p: p.created_at, reverse=True)
		consecutive = 0
		
		for payment in sorted_payments:
			if payment.status.value == 'failed':
				consecutive += 1
			else:
				break
		
		return consecutive
	
	def _calculate_usage_trend(self, usage_records: List[BLUsage]) -> float:
		"""Calculate usage trend (positive = increasing, negative = decreasing)"""
		if len(usage_records) < 2:
			return 0.0
		
		# Sort by timestamp
		sorted_usage = sorted(usage_records, key=lambda u: u.timestamp)
		
		# Compare first half to second half
		mid_point = len(sorted_usage) // 2
		first_half = sorted_usage[:mid_point]
		second_half = sorted_usage[mid_point:]
		
		first_avg = sum(u.quantity for u in first_half) / len(first_half)
		second_avg = sum(u.quantity for u in second_half) / len(second_half)
		
		if first_avg > 0:
			return float((second_avg - first_avg) / first_avg)
		return 0.0
	
	def _calculate_usage_variability(self, usage_records: List[BLUsage]) -> float:
		"""Calculate usage variability (coefficient of variation)"""
		if len(usage_records) < 2:
			return 0.0
		
		values = [float(u.quantity) for u in usage_records]
		mean_val = np.mean(values)
		std_val = np.std(values)
		
		return float(std_val / mean_val) if mean_val > 0 else 0.0
	
	def _calculate_peak_usage_ratio(self, usage_records: List[BLUsage]) -> float:
		"""Calculate ratio of peak usage to average usage"""
		if not usage_records:
			return 0.0
		
		values = [float(u.quantity) for u in usage_records]
		peak_usage = max(values)
		avg_usage = np.mean(values)
		
		return float(peak_usage / avg_usage) if avg_usage > 0 else 0.0
	
	def _calculate_avg_days_to_pay(self, invoices: List[BLInvoice]) -> float:
		"""Calculate average days to pay invoices"""
		paid_invoices = [i for i in invoices if i.paid_at]
		if not paid_invoices:
			return 0.0
		
		pay_times = [(i.paid_at - i.invoice_date).days for i in paid_invoices]
		return float(np.mean(pay_times))
	
	def _calculate_dispute_rate(self, invoices: List[BLInvoice]) -> float:
		"""Calculate invoice dispute rate"""
		if not invoices:
			return 0.0
		
		# Check metadata for dispute flags
		disputed = sum(1 for i in invoices if i.metadata and i.metadata.get('disputed', False))
		return float(disputed / len(invoices))
	
	def _calculate_collection_efficiency(self, invoices: List[BLInvoice]) -> float:
		"""Calculate collection efficiency"""
		if not invoices:
			return 1.0
		
		total_billed = sum(i.total for i in invoices)
		total_collected = sum(i.amount_paid for i in invoices)
		
		return float(total_collected / total_billed) if total_billed > 0 else 1.0
	
	async def _predict_revenue_leakage(self, customer_id: str, features: Dict[str, Any]) -> Optional[BillingPrediction]:
		"""Predict potential revenue leakage"""
		try:
			# Calculate risk factors for revenue leakage
			risk_score = 0.0
			evidence = {}
			
			# Usage without billing coverage
			if features['usage_trend'] > 0.2 and features['total_mrr'] == 0:
				risk_score += 0.4
				evidence['unbilled_usage_growth'] = features['usage_trend']
			
			# Payment method issues
			if features['payment_failure_rate'] > 0.3:
				risk_score += 0.3
				evidence['payment_issues'] = features['payment_failure_rate']
			
			# Collection efficiency problems
			if features['collection_efficiency'] < 0.8:
				risk_score += 0.2
				evidence['collection_issues'] = features['collection_efficiency']
			
			# Overdue amounts
			if features['overdue_amount'] > 0:
				risk_score += 0.1
				evidence['overdue_amount'] = features['overdue_amount']
			
			if risk_score < self.prediction_thresholds[PredictionType.REVENUE_LEAKAGE]:
				return None
			
			# Estimate financial impact
			predicted_impact = Decimal(str(features['overdue_amount'] + features['total_mrr'] * 3))
			
			# Generate recommendations
			recommendations = []
			if features['payment_failure_rate'] > 0.2:
				recommendations.append("Update payment method and retry failed charges")
			if features['collection_efficiency'] < 0.8:
				recommendations.append("Initiate dunning sequence for overdue accounts")
			if features['usage_trend'] > 0.2:
				recommendations.append("Review billing coverage for increased usage")
			
			prediction_data = {
				'prediction_type': PredictionType.REVENUE_LEAKAGE.value,
				'customer_id': customer_id,
				'confidence_score': min(risk_score, 1.0),
				'risk_score': risk_score,
				'predicted_impact': predicted_impact,
				'evidence': evidence,
				'recommended_actions': recommendations
			}
			
			return BillingPrediction(prediction_data)
			
		except Exception as e:
			self.logger.error(f"Revenue leakage prediction failed: {e}")
			return None
	
	async def _predict_payment_failure(self, customer_id: str, features: Dict[str, Any]) -> Optional[BillingPrediction]:
		"""Predict likelihood of next payment failure"""
		try:
			risk_score = 0.0
			evidence = {}
			
			# Historical failure patterns
			if features['consecutive_failures'] > 0:
				risk_score += min(features['consecutive_failures'] * 0.2, 0.6)
				evidence['consecutive_failures'] = features['consecutive_failures']
			
			if features['payment_failure_rate'] > 0.1:
				risk_score += features['payment_failure_rate'] * 0.4
				evidence['historical_failure_rate'] = features['payment_failure_rate']
			
			# Payment method instability
			if features['payment_method_changes'] > 2:
				risk_score += 0.2
				evidence['payment_method_instability'] = features['payment_method_changes']
			
			# Days since last successful payment
			if features['days_since_last_payment'] > 60:
				risk_score += 0.3
				evidence['payment_recency'] = features['days_since_last_payment']
			
			if risk_score < self.prediction_thresholds[PredictionType.PAYMENT_FAILURE]:
				return None
			
			# Estimate impact
			predicted_impact = Decimal(str(features['avg_payment_amount']))
			
			recommendations = []
			if features['consecutive_failures'] > 1:
				recommendations.append("Contact customer to update payment method")
			if features['days_since_last_payment'] > 45:
				recommendations.append("Send payment reminder and verify account status")
			recommendations.append("Implement smart retry logic with optimal timing")
			
			prediction_data = {
				'prediction_type': PredictionType.PAYMENT_FAILURE.value,
				'customer_id': customer_id,
				'confidence_score': min(risk_score, 1.0),
				'risk_score': risk_score,
				'predicted_impact': predicted_impact,
				'evidence': evidence,
				'recommended_actions': recommendations
			}
			
			return BillingPrediction(prediction_data)
			
		except Exception as e:
			self.logger.error(f"Payment failure prediction failed: {e}")
			return None
	
	async def _predict_involuntary_churn(self, customer_id: str, features: Dict[str, Any]) -> Optional[BillingPrediction]:
		"""Predict involuntary churn risk"""
		try:
			risk_score = 0.0
			evidence = {}
			
			# Payment issues leading to churn
			if features['consecutive_failures'] >= 3:
				risk_score += 0.5
				evidence['multiple_payment_failures'] = features['consecutive_failures']
			
			# Overdue amounts
			if features['overdue_amount'] > features['total_mrr'] * 2:
				risk_score += 0.3
				evidence['significant_overdue'] = features['overdue_amount']
			
			# Long-term payment problems
			if features['collection_efficiency'] < 0.6:
				risk_score += 0.2
				evidence['poor_collection'] = features['collection_efficiency']
			
			if risk_score < self.prediction_thresholds[PredictionType.INVOLUNTARY_CHURN]:
				return None
			
			# Estimate lost revenue impact
			predicted_impact = Decimal(str(features['total_mrr'] * 12))  # Annual MRR loss
			
			recommendations = [
				"Immediate customer outreach for payment resolution",
				"Offer payment plan or temporary discount",
				"Escalate to customer success for retention intervention",
				"Consider account suspension with grace period"
			]
			
			prediction_data = {
				'prediction_type': PredictionType.INVOLUNTARY_CHURN.value,
				'customer_id': customer_id,
				'confidence_score': min(risk_score, 1.0),
				'risk_score': risk_score,
				'predicted_impact': predicted_impact,
				'evidence': evidence,
				'recommended_actions': recommendations
			}
			
			return BillingPrediction(prediction_data)
			
		except Exception as e:
			self.logger.error(f"Involuntary churn prediction failed: {e}")
			return None
	
	async def _detect_usage_anomalies(self, customer_id: str, features: Dict[str, Any]) -> Optional[BillingPrediction]:
		"""Detect usage anomalies that could impact billing"""
		try:
			risk_score = 0.0
			evidence = {}
			
			# Sudden usage spikes
			if features['peak_usage_ratio'] > 5.0:
				risk_score += 0.4
				evidence['usage_spike'] = features['peak_usage_ratio']
			
			# High usage variability
			if features['usage_variability'] > 2.0:
				risk_score += 0.3
				evidence['usage_instability'] = features['usage_variability']
			
			# Usage without recent activity
			if features['usage_trend'] > 0.5 and features['days_since_last_usage'] > 7:
				risk_score += 0.3
				evidence['stale_high_usage'] = {
					'trend': features['usage_trend'],
					'days_since': features['days_since_last_usage']
				}
			
			if risk_score < self.prediction_thresholds[PredictionType.USAGE_ANOMALY]:
				return None
			
			# Estimate potential billing impact
			estimated_overage = features['total_mrr'] * features['peak_usage_ratio'] * 0.1
			predicted_impact = Decimal(str(estimated_overage))
			
			recommendations = [
				"Review usage patterns and billing coverage",
				"Check for potential billing system issues",
				"Validate usage data accuracy",
				"Consider usage alerts and limits"
			]
			
			prediction_data = {
				'prediction_type': PredictionType.USAGE_ANOMALY.value,
				'customer_id': customer_id,
				'confidence_score': min(risk_score, 1.0),
				'risk_score': risk_score,
				'predicted_impact': predicted_impact,
				'evidence': evidence,
				'recommended_actions': recommendations
			}
			
			return BillingPrediction(prediction_data)
			
		except Exception as e:
			self.logger.error(f"Usage anomaly detection failed: {e}")
			return None
	
	async def _detect_fraud_patterns(self, customer_id: str, features: Dict[str, Any]) -> Optional[BillingPrediction]:
		"""Detect potential fraud patterns"""
		try:
			risk_score = 0.0
			evidence = {}
			
			# Suspicious payment patterns
			if features['payment_method_changes'] > 5:
				risk_score += 0.4
				evidence['excessive_payment_changes'] = features['payment_method_changes']
			
			# Unusual usage patterns
			if features['usage_variability'] > 3.0 and features['peak_usage_ratio'] > 10:
				risk_score += 0.3
				evidence['suspicious_usage'] = {
					'variability': features['usage_variability'],
					'peak_ratio': features['peak_usage_ratio']
				}
			
			# Dispute patterns
			if features['invoice_dispute_rate'] > 0.3:
				risk_score += 0.3
				evidence['high_dispute_rate'] = features['invoice_dispute_rate']
			
			if risk_score < self.prediction_thresholds[PredictionType.FRAUD_DETECTION]:
				return None
			
			# Estimate potential loss
			predicted_impact = Decimal(str(features['total_mrr'] * 6))  # 6 months of revenue at risk
			
			recommendations = [
				"Flag account for manual review",
				"Implement additional verification steps",
				"Monitor transaction patterns closely",
				"Consider temporary service restrictions"
			]
			
			prediction_data = {
				'prediction_type': PredictionType.FRAUD_DETECTION.value,
				'customer_id': customer_id,
				'confidence_score': min(risk_score, 1.0),
				'risk_score': risk_score,
				'predicted_impact': predicted_impact,
				'evidence': evidence,
				'recommended_actions': recommendations
			}
			
			return BillingPrediction(prediction_data)
			
		except Exception as e:
			self.logger.error(f"Fraud detection failed: {e}")
			return None
	
	async def _process_prediction(self, prediction: BillingPrediction) -> None:
		"""Process a prediction and trigger automated actions"""
		try:
			# Log the prediction
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.COMPLIANCE_CHECK.value,
				'user_id': 'system',
				'resource_type': 'billing_prediction',
				'resource_id': prediction.id,
				'action': 'prediction_generated',
				'description': f'AI prediction: {prediction.prediction_type.value}',
				'metadata': {
					'prediction_type': prediction.prediction_type.value,
					'customer_id': prediction.customer_id,
					'confidence_score': prediction.confidence_score,
					'risk_score': prediction.risk_score,
					'predicted_impact': str(prediction.predicted_impact)
				}
			})
			
			# Trigger automated actions based on prediction type and confidence
			if prediction.confidence_score >= 0.8:
				await self._execute_automated_actions(prediction)
			
			# Send alerts for high-risk predictions
			if prediction.risk_score >= 0.7:
				await self._send_prediction_alert(prediction)
				
		except Exception as e:
			self.logger.error(f"Failed to process prediction {prediction.id}: {e}")
	
	async def _execute_automated_actions(self, prediction: BillingPrediction) -> None:
		"""Execute automated actions based on prediction"""
		try:
			actions_taken = []
			
			if prediction.prediction_type == PredictionType.PAYMENT_FAILURE:
				# Proactively update payment retry schedule
				actions_taken.append("scheduled_smart_retry")
				
			elif prediction.prediction_type == PredictionType.REVENUE_LEAKAGE:
				# Create dunning case for overdue amounts
				if prediction.predicted_impact > Decimal('100'):
					actions_taken.append("created_dunning_case")
					
			elif prediction.prediction_type == PredictionType.FRAUD_DETECTION:
				# Flag account for review
				actions_taken.append("flagged_for_review")
			
			prediction.automated_actions_taken = actions_taken
			self.logger.info(f"Executed automated actions for prediction {prediction.id}: {actions_taken}")
			
		except Exception as e:
			self.logger.error(f"Failed to execute automated actions: {e}")
	
	async def _send_prediction_alert(self, prediction: BillingPrediction) -> None:
		"""Send alert for high-risk prediction"""
		try:
			alert_data = {
				'type': 'billing_prediction_alert',
				'prediction_id': prediction.id,
				'prediction_type': prediction.prediction_type.value,
				'customer_id': prediction.customer_id,
				'confidence_score': prediction.confidence_score,
				'risk_score': prediction.risk_score,
				'predicted_impact': str(prediction.predicted_impact),
				'recommended_actions': prediction.recommended_actions
			}
			
			self.logger.warning(f"High-risk billing prediction: {alert_data}")
			
		except Exception as e:
			self.logger.error(f"Failed to send prediction alert: {e}")
	
	async def _monitor_real_time_events(self) -> None:
		"""Monitor real-time events for immediate predictions"""
		try:
			# Monitor payment failures
			await self._monitor_payment_events()
			
			# Monitor usage spikes
			await self._monitor_usage_events()
			
			# Monitor invoice disputes
			await self._monitor_invoice_events()
			
		except Exception as e:
			self.logger.error(f"Real-time monitoring failed: {e}")
	
	async def _monitor_payment_events(self) -> None:
		"""Monitor payment events for immediate predictions"""
		# Check recent payment failures
		recent_failures = [
			p for p in self.billing_service.payments.values()
			if p.status.value == 'failed' and (datetime.utcnow() - p.created_at).seconds < 300
		]
		
		for payment in recent_failures:
			await self._generate_immediate_payment_prediction(payment.customer_id)
	
	async def _generate_immediate_payment_prediction(self, customer_id: str) -> None:
		"""Generate immediate prediction after payment failure"""
		features = await self._extract_customer_features(customer_id)
		if features:
			prediction = await self._predict_payment_failure(customer_id, features)
			if prediction:
				prediction.time_horizon = 7  # Shorter horizon for immediate predictions
				self.predictions[prediction.id] = prediction
				await self._process_prediction(prediction)
	
	async def _cleanup_expired_predictions(self) -> None:
		"""Clean up expired predictions"""
		now = datetime.utcnow()
		expired_ids = [
			pred_id for pred_id, pred in self.predictions.items()
			if pred.expires_at < now or pred.status == 'resolved'
		]
		
		for pred_id in expired_ids:
			del self.predictions[pred_id]
		
		if expired_ids:
			self.logger.info(f"Cleaned up {len(expired_ids)} expired predictions")
	
	async def get_customer_predictions(self, customer_id: str, active_only: bool = True) -> List[BillingPrediction]:
		"""Get all predictions for a customer"""
		predictions = [
			pred for pred in self.predictions.values()
			if pred.customer_id == customer_id
		]
		
		if active_only:
			predictions = [pred for pred in predictions if pred.status == 'active']
		
		return sorted(predictions, key=lambda p: p.risk_score, reverse=True)
	
	async def get_high_risk_predictions(self, min_risk_score: float = 0.7) -> List[BillingPrediction]:
		"""Get all high-risk predictions"""
		return [
			pred for pred in self.predictions.values()
			if pred.risk_score >= min_risk_score and pred.status == 'active'
		]
	
	async def resolve_prediction(self, prediction_id: str, resolution_notes: str = None) -> bool:
		"""Mark a prediction as resolved"""
		try:
			prediction = self.predictions.get(prediction_id)
			if not prediction:
				return False
			
			prediction.status = 'resolved'
			prediction.metadata['resolved_at'] = datetime.utcnow().isoformat()
			if resolution_notes:
				prediction.metadata['resolution_notes'] = resolution_notes
			
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.COMPLIANCE_CHECK.value,
				'user_id': 'system',
				'resource_type': 'billing_prediction',
				'resource_id': prediction_id,
				'action': 'prediction_resolved',
				'description': f'Prediction resolved: {prediction.prediction_type.value}',
				'metadata': {
					'resolution_notes': resolution_notes
				}
			})
			
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to resolve prediction {prediction_id}: {e}")
			return False
	
	async def get_prediction_analytics(self, days: int = 30) -> Dict[str, Any]:
		"""Get analytics on prediction performance"""
		cutoff_date = datetime.utcnow() - timedelta(days=days)
		
		period_predictions = [
			pred for pred in self.predictions.values()
			if pred.predicted_at >= cutoff_date
		]
		
		# Calculate metrics by type
		type_metrics = {}
		for pred_type in PredictionType:
			type_preds = [p for p in period_predictions if p.prediction_type == pred_type]
			if type_preds:
				type_metrics[pred_type.value] = {
					'total_predictions': len(type_preds),
					'avg_confidence': sum(p.confidence_score for p in type_preds) / len(type_preds),
					'avg_risk_score': sum(p.risk_score for p in type_preds) / len(type_preds),
					'total_predicted_impact': str(sum(p.predicted_impact for p in type_preds)),
					'resolved_count': len([p for p in type_preds if p.status == 'resolved'])
				}
		
		return {
			'period_days': days,
			'total_predictions': len(period_predictions),
			'active_predictions': len([p for p in period_predictions if p.status == 'active']),
			'resolved_predictions': len([p for p in period_predictions if p.status == 'resolved']),
			'high_risk_predictions': len([p for p in period_predictions if p.risk_score >= 0.7]),
			'total_predicted_impact': str(sum(p.predicted_impact for p in period_predictions)),
			'prediction_types': type_metrics,
			'generated_at': datetime.utcnow().isoformat()
		}


# Global predictive billing AI
_predictive_ai_instance: Optional[PredictiveBillingAI] = None

def get_predictive_billing_ai() -> PredictiveBillingAI:
	"""Get global predictive billing AI instance"""
	global _predictive_ai_instance
	if _predictive_ai_instance is None:
		_predictive_ai_instance = PredictiveBillingAI()
	return _predictive_ai_instance


__all__ = [
	'PredictiveBillingAI',
	'BillingPrediction',
	'PredictionType',
	'get_predictive_billing_ai'
]