"""
APG Comprehensive Billing Analytics and Insights Platform

Real-time billing intelligence platform that provides actionable insights
across revenue operations, customer health, and business growth with 
predictive analytics and automated optimization recommendations.

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
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from uuid_extensions import uuid7str
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .models import BLCustomer, BLSubscription, BLPayment, BLInvoice, BLRevenue
from .service import get_billing_service
from .predictive_billing_ai import get_predictive_billing_ai, PredictionType
from .revenue_optimization import get_revenue_optimization_engine
from .audit_compliance import get_audit_compliance_system, AuditEventType


class AnalyticsScope(Enum):
	"""Analytics scope and granularity"""
	REAL_TIME = "real_time"
	HOURLY = "hourly" 
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	YEARLY = "yearly"


class MetricType(Enum):
	"""Types of billing metrics"""
	REVENUE = "revenue"
	GROWTH = "growth"
	CHURN = "churn"
	CUSTOMER_HEALTH = "customer_health"
	PAYMENT_SUCCESS = "payment_success"
	BILLING_EFFICIENCY = "billing_efficiency"
	COHORT_ANALYSIS = "cohort_analysis"
	PREDICTIVE_INSIGHTS = "predictive_insights"


class InsightPriority(Enum):
	"""Priority levels for insights"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFORMATIONAL = "informational"


class BillingInsight:
	"""Individual billing insight with context and recommendations"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.metric_type = MetricType(data['metric_type'])
		self.priority = InsightPriority(data['priority'])
		self.title = data['title']
		self.description = data['description']
		self.current_value = data['current_value']
		self.previous_value = data.get('previous_value')
		self.benchmark_value = data.get('benchmark_value')
		self.trend_direction = data.get('trend_direction', 'stable')  # up, down, stable
		self.percentage_change = data.get('percentage_change', 0.0)
		self.recommendations = data.get('recommendations', [])
		self.impact_score = data.get('impact_score', 0.0)  # 0-100
		self.confidence_level = data.get('confidence_level', 0.8)  # 0-1
		self.data_sources = data.get('data_sources', [])
		self.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
		self.expires_at = self.created_at + timedelta(hours=data.get('ttl_hours', 24))
		self.metadata = data.get('metadata', {})


class RevenueMetrics:
	"""Real-time revenue metrics calculation"""
	
	def __init__(self, billing_service):
		self.billing_service = billing_service
		self.logger = logging.getLogger(f"{__name__}.RevenueMetrics")
	
	async def calculate_mrr_metrics(self, as_of_date: datetime = None) -> Dict[str, Any]:
		"""Calculate comprehensive MRR metrics"""
		if as_of_date is None:
			as_of_date = datetime.utcnow()
		
		# Get active subscriptions
		active_subscriptions = [
			sub for sub in self.billing_service.subscriptions.values()
			if sub.status.value == 'active' and sub.created_at <= as_of_date
		]
		
		# Calculate MRR components
		current_mrr = sum(getattr(sub, 'mrr', Decimal('0')) for sub in active_subscriptions)
		
		# New MRR (subscriptions started this month)
		month_start = as_of_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
		new_subscriptions = [
			sub for sub in active_subscriptions
			if sub.created_at >= month_start
		]
		new_mrr = sum(getattr(sub, 'mrr', Decimal('0')) for sub in new_subscriptions)
		
		# Expansion MRR (upgrades this month)
		expansion_mrr = await self._calculate_expansion_mrr(month_start, as_of_date)
		
		# Contraction MRR (downgrades this month)
		contraction_mrr = await self._calculate_contraction_mrr(month_start, as_of_date)
		
		# Churned MRR (cancellations this month)
		churned_subscriptions = [
			sub for sub in self.billing_service.subscriptions.values()
			if sub.status.value == 'cancelled' and 
			   sub.cancelled_at and month_start <= sub.cancelled_at <= as_of_date
		]
		churned_mrr = sum(getattr(sub, 'mrr', Decimal('0')) for sub in churned_subscriptions)
		
		# Calculate previous month for comparison
		prev_month_start = month_start - timedelta(days=1)
		prev_month_start = prev_month_start.replace(day=1)
		prev_month_end = month_start - timedelta(seconds=1)
		previous_mrr = await self._calculate_historical_mrr(prev_month_end)
		
		# Growth calculations
		net_new_mrr = new_mrr + expansion_mrr - contraction_mrr - churned_mrr
		mrr_growth_rate = float((current_mrr - previous_mrr) / previous_mrr * 100) if previous_mrr > 0 else 0
		
		return {
			'current_mrr': str(current_mrr),
			'previous_mrr': str(previous_mrr),
			'new_mrr': str(new_mrr),
			'expansion_mrr': str(expansion_mrr),
			'contraction_mrr': str(contraction_mrr),
			'churned_mrr': str(churned_mrr),
			'net_new_mrr': str(net_new_mrr),
			'mrr_growth_rate': mrr_growth_rate,
			'active_subscriptions': len(active_subscriptions),
			'new_subscriptions': len(new_subscriptions),
			'churned_subscriptions': len(churned_subscriptions),
			'calculated_at': as_of_date.isoformat(),
			'period': f"{month_start.isoformat()[:10]} to {as_of_date.isoformat()[:10]}"
		}
	
	async def _calculate_expansion_mrr(self, start_date: datetime, end_date: datetime) -> Decimal:
		"""Calculate expansion MRR from subscription upgrades"""
		# In a real implementation, this would track subscription plan changes
		# For now, we'll estimate based on payment increases
		return Decimal('0')
	
	async def _calculate_contraction_mrr(self, start_date: datetime, end_date: datetime) -> Decimal:
		"""Calculate contraction MRR from subscription downgrades"""
		# In a real implementation, this would track subscription plan downgrades
		return Decimal('0')
	
	async def _calculate_historical_mrr(self, as_of_date: datetime) -> Decimal:
		"""Calculate historical MRR as of a specific date"""
		# Get subscriptions that were active at the historical date
		historical_subscriptions = [
			sub for sub in self.billing_service.subscriptions.values()
			if sub.created_at <= as_of_date and 
			   (sub.cancelled_at is None or sub.cancelled_at > as_of_date)
		]
		
		return sum(getattr(sub, 'mrr', Decimal('0')) for sub in historical_subscriptions)


class CustomerHealthAnalytics:
	"""Advanced customer health scoring and segmentation"""
	
	def __init__(self, billing_service):
		self.billing_service = billing_service
		self.logger = logging.getLogger(f"{__name__}.CustomerHealthAnalytics")
		self.segmentation_model = KMeans(n_clusters=5, random_state=42)
		self.scaler = StandardScaler()
	
	async def calculate_customer_health_scores(self) -> Dict[str, Any]:
		"""Calculate comprehensive customer health scores"""
		health_scores = {}
		segment_analysis = {}
		
		# Get all active customers
		active_customers = [
			customer for customer in self.billing_service.customers.values()
			if self._is_customer_active(customer)
		]
		
		# Calculate individual health scores
		customer_features = []
		customer_ids = []
		
		for customer in active_customers:
			features = await self._extract_health_features(customer.id)
			if features:
				health_score = self._calculate_individual_health_score(features)
				health_scores[customer.id] = {
					'customer_id': customer.id,
					'health_score': health_score,
					'health_category': self._categorize_health_score(health_score),
					'risk_factors': self._identify_risk_factors(features),
					'opportunities': self._identify_opportunities(features),
					'features': features
				}
				
				# Collect features for segmentation
				feature_vector = [
					features['payment_success_rate'],
					features['usage_trend'],
					features['support_ticket_rate'],
					features['feature_adoption_score'],
					features['engagement_score']
				]
				customer_features.append(feature_vector)
				customer_ids.append(customer.id)
		
		# Perform customer segmentation
		if len(customer_features) >= 5:  # Minimum for clustering
			try:
				normalized_features = self.scaler.fit_transform(customer_features)
				segments = self.segmentation_model.fit_predict(normalized_features)
				
				# Analyze segments
				for i, customer_id in enumerate(customer_ids):
					segment = segments[i]
					health_scores[customer_id]['segment'] = f"segment_{segment}"
				
				segment_analysis = self._analyze_customer_segments(customer_ids, segments, health_scores)
			
			except Exception as e:
				self.logger.warning(f"Customer segmentation failed: {e}")
		
		# Calculate aggregate metrics
		health_values = [score['health_score'] for score in health_scores.values()]
		avg_health_score = np.mean(health_values) if health_values else 0
		
		health_distribution = {
			'excellent': len([s for s in health_values if s >= 80]),
			'good': len([s for s in health_values if 60 <= s < 80]),
			'at_risk': len([s for s in health_values if 40 <= s < 60]),
			'critical': len([s for s in health_values if s < 40])
		}
		
		return {
			'customer_health_scores': health_scores,
			'segment_analysis': segment_analysis,
			'average_health_score': avg_health_score,
			'health_distribution': health_distribution,
			'total_customers_analyzed': len(health_scores),
			'calculated_at': datetime.utcnow().isoformat()
		}
	
	def _is_customer_active(self, customer) -> bool:
		"""Check if customer has active subscriptions"""
		active_subscriptions = [
			sub for sub in self.billing_service.subscriptions.values()
			if sub.customer_id == customer.id and sub.status.value == 'active'
		]
		return len(active_subscriptions) > 0
	
	async def _extract_health_features(self, customer_id: str) -> Optional[Dict[str, float]]:
		"""Extract features for customer health scoring"""
		try:
			# Get customer data
			customer_payments = [
				p for p in self.billing_service.payments.values()
				if p.customer_id == customer_id
			]
			
			customer_subscriptions = [
				s for s in self.billing_service.subscriptions.values()
				if s.customer_id == customer_id
			]
			
			# Calculate payment success rate
			recent_payments = [
				p for p in customer_payments
				if (datetime.utcnow() - p.created_at).days <= 90
			]
			successful_payments = [p for p in recent_payments if p.status.value == 'succeeded']
			payment_success_rate = len(successful_payments) / max(len(recent_payments), 1) * 100
			
			# Calculate usage trend (simplified)
			usage_trend = 50.0  # Baseline - would integrate with actual usage data
			
			# Support ticket rate (would integrate with support system)
			support_ticket_rate = 10.0  # Baseline
			
			# Feature adoption score (would integrate with product analytics)
			feature_adoption_score = 70.0  # Baseline
			
			# Engagement score based on subscription activity
			engagement_score = 60.0
			if customer_subscriptions:
				avg_subscription_age = sum(
					(datetime.utcnow() - sub.created_at).days 
					for sub in customer_subscriptions
				) / len(customer_subscriptions)
				
				# Longer tenure = higher engagement (up to a point)
				engagement_score = min(80, 40 + avg_subscription_age / 10)
			
			return {
				'payment_success_rate': payment_success_rate,
				'usage_trend': usage_trend,
				'support_ticket_rate': support_ticket_rate,
				'feature_adoption_score': feature_adoption_score,
				'engagement_score': engagement_score
			}
			
		except Exception as e:
			self.logger.error(f"Failed to extract health features for customer {customer_id}: {e}")
			return None
	
	def _calculate_individual_health_score(self, features: Dict[str, float]) -> float:
		"""Calculate individual customer health score"""
		# Weighted combination of features
		weights = {
			'payment_success_rate': 0.3,
			'usage_trend': 0.2,
			'support_ticket_rate': -0.1,  # Negative weight
			'feature_adoption_score': 0.25,
			'engagement_score': 0.25
		}
		
		score = 0.0
		for feature, value in features.items():
			weight = weights.get(feature, 0)
			if feature == 'support_ticket_rate':
				# Invert support ticket rate (lower is better)
				normalized_value = max(0, 100 - value)
			else:
				normalized_value = value
			
			score += weight * normalized_value
		
		return max(0, min(100, score))
	
	def _categorize_health_score(self, score: float) -> str:
		"""Categorize health score into buckets"""
		if score >= 80:
			return "excellent"
		elif score >= 60:
			return "good"
		elif score >= 40:
			return "at_risk"
		else:
			return "critical"
	
	def _identify_risk_factors(self, features: Dict[str, float]) -> List[str]:
		"""Identify risk factors from customer features"""
		risk_factors = []
		
		if features['payment_success_rate'] < 80:
			risk_factors.append("Low payment success rate")
		
		if features['usage_trend'] < 30:
			risk_factors.append("Declining usage")
		
		if features['support_ticket_rate'] > 20:
			risk_factors.append("High support demand")
		
		if features['feature_adoption_score'] < 40:
			risk_factors.append("Low feature adoption")
		
		if features['engagement_score'] < 40:
			risk_factors.append("Low engagement")
		
		return risk_factors
	
	def _identify_opportunities(self, features: Dict[str, float]) -> List[str]:
		"""Identify growth opportunities from customer features"""
		opportunities = []
		
		if features['feature_adoption_score'] < 70:
			opportunities.append("Feature adoption training")
		
		if features['usage_trend'] > 70:
			opportunities.append("Upsell opportunity")
		
		if features['payment_success_rate'] > 95:
			opportunities.append("Potential advocate")
		
		return opportunities
	
	def _analyze_customer_segments(self, customer_ids: List[str], segments: np.ndarray, 
								 health_scores: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze customer segments"""
		segment_analysis = {}
		
		for segment_id in np.unique(segments):
			segment_customers = [
				customer_ids[i] for i, seg in enumerate(segments) if seg == segment_id
			]
			
			segment_health_scores = [
				health_scores[customer_id]['health_score'] 
				for customer_id in segment_customers
			]
			
			segment_analysis[f"segment_{segment_id}"] = {
				'customer_count': len(segment_customers),
				'avg_health_score': np.mean(segment_health_scores),
				'health_range': {
					'min': np.min(segment_health_scores),
					'max': np.max(segment_health_scores)
				},
				'characteristics': self._characterize_segment(segment_customers, health_scores)
			}
		
		return segment_analysis
	
	def _characterize_segment(self, customer_ids: List[str], health_scores: Dict[str, Any]) -> Dict[str, Any]:
		"""Characterize a customer segment"""
		# Analyze common characteristics
		risk_factors = []
		opportunities = []
		
		for customer_id in customer_ids:
			customer_data = health_scores[customer_id]
			risk_factors.extend(customer_data['risk_factors'])
			opportunities.extend(customer_data['opportunities'])
		
		# Count most common factors
		from collections import Counter
		risk_counter = Counter(risk_factors)
		opportunity_counter = Counter(opportunities)
		
		return {
			'common_risk_factors': dict(risk_counter.most_common(3)),
			'common_opportunities': dict(opportunity_counter.most_common(3))
		}


class ComprehensiveBillingAnalytics:
	"""Main analytics engine providing comprehensive billing insights"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.ComprehensiveBillingAnalytics")
		self.billing_service = get_billing_service()
		self.predictive_ai = get_predictive_billing_ai()
		self.revenue_optimizer = get_revenue_optimization_engine()
		self.audit_system = get_audit_compliance_system()
		
		# Analytics components
		self.revenue_metrics = RevenueMetrics(self.billing_service)
		self.customer_health = CustomerHealthAnalytics(self.billing_service)
		
		# Insights storage
		self.insights: Dict[str, BillingInsight] = {}
		
		# Background processing
		asyncio.create_task(self._start_analytics_engine())
	
	async def _start_analytics_engine(self) -> None:
		"""Start background analytics processing"""
		while True:
			try:
				await self._generate_real_time_insights()
				await asyncio.sleep(300)  # Run every 5 minutes
			except Exception as e:
				self.logger.error(f"Analytics engine error: {e}")
				await asyncio.sleep(300)
	
	async def _generate_real_time_insights(self) -> None:
		"""Generate real-time billing insights"""
		try:
			# Generate MRR insights
			await self._generate_mrr_insights()
			
			# Generate customer health insights
			await self._generate_customer_health_insights()
			
			# Generate predictive insights
			await self._generate_predictive_insights()
			
			# Generate payment performance insights
			await self._generate_payment_insights()
			
			# Clean up expired insights
			self._cleanup_expired_insights()
			
		except Exception as e:
			self.logger.error(f"Real-time insights generation failed: {e}")
	
	async def _generate_mrr_insights(self) -> None:
		"""Generate MRR-related insights"""
		try:
			mrr_data = await self.revenue_metrics.calculate_mrr_metrics()
			
			# MRR growth insight
			growth_rate = mrr_data['mrr_growth_rate']
			
			if growth_rate > 20:
				priority = InsightPriority.HIGH
				title = "Exceptional MRR Growth"
				description = f"MRR is growing at {growth_rate:.1f}% month-over-month, significantly above industry benchmarks."
				recommendations = [
					"Scale successful acquisition channels",
					"Maintain current customer success initiatives",
					"Consider expanding to new market segments"
				]
				impact_score = 90.0
			elif growth_rate > 10:
				priority = InsightPriority.MEDIUM
				title = "Strong MRR Growth"
				description = f"MRR growth of {growth_rate:.1f}% is healthy and sustainable."
				recommendations = [
					"Continue current growth strategies",
					"Monitor for sustainable scaling opportunities"
				]
				impact_score = 70.0
			elif growth_rate < 0:
				priority = InsightPriority.CRITICAL
				title = "MRR Decline Alert"
				description = f"MRR is declining at {abs(growth_rate):.1f}% month-over-month. Immediate action required."
				recommendations = [
					"Investigate churn causes immediately",
					"Implement retention campaigns",
					"Review pricing and value proposition",
					"Conduct customer interviews"
				]
				impact_score = 95.0
			else:
				return  # No significant insight needed
			
			insight_data = {
				'metric_type': MetricType.REVENUE.value,
				'priority': priority.value,
				'title': title,
				'description': description,
				'current_value': growth_rate,
				'benchmark_value': 10.0,  # Industry benchmark
				'trend_direction': 'up' if growth_rate > 0 else 'down',
				'percentage_change': growth_rate,
				'recommendations': recommendations,
				'impact_score': impact_score,
				'confidence_level': 0.9,
				'data_sources': ['subscription_data', 'payment_data'],
				'metadata': mrr_data
			}
			
			insight = BillingInsight(insight_data)
			self.insights[f"mrr_growth_{datetime.utcnow().strftime('%Y%m%d')}"] = insight
			
		except Exception as e:
			self.logger.error(f"MRR insights generation failed: {e}")
	
	async def _generate_customer_health_insights(self) -> None:
		"""Generate customer health insights"""
		try:
			health_data = await self.customer_health.calculate_customer_health_scores()
			
			avg_health = health_data['average_health_score']
			critical_customers = health_data['health_distribution']['critical']
			at_risk_customers = health_data['health_distribution']['at_risk']
			total_customers = health_data['total_customers_analyzed']
			
			# Critical health insight
			if critical_customers > 0:
				critical_percentage = (critical_customers / total_customers) * 100
				
				insight_data = {
					'metric_type': MetricType.CUSTOMER_HEALTH.value,
					'priority': InsightPriority.CRITICAL.value,
					'title': f"{critical_customers} Customers at Critical Health Risk",
					'description': f"{critical_percentage:.1f}% of customers have critical health scores and are at immediate churn risk.",
					'current_value': critical_customers,
					'benchmark_value': 0,
					'trend_direction': 'down',
					'recommendations': [
						"Immediate outreach to critical health customers",
						"Implement emergency retention campaigns",
						"Provide dedicated customer success support",
						"Investigate common failure patterns"
					],
					'impact_score': 85.0,
					'confidence_level': 0.95,
					'data_sources': ['customer_health_analysis'],
					'metadata': {
						'critical_customers': critical_customers,
						'critical_percentage': critical_percentage,
						'avg_health_score': avg_health
					}
				}
				
				insight = BillingInsight(insight_data)
				self.insights[f"critical_health_{datetime.utcnow().strftime('%Y%m%d')}"] = insight
			
			# At-risk customers insight
			if at_risk_customers > total_customers * 0.15:  # More than 15%
				at_risk_percentage = (at_risk_customers / total_customers) * 100
				
				insight_data = {
					'metric_type': MetricType.CUSTOMER_HEALTH.value,
					'priority': InsightPriority.HIGH.value,
					'title': f"High Number of At-Risk Customers ({at_risk_customers})",
					'description': f"{at_risk_percentage:.1f}% of customers are at-risk, above the healthy threshold of 15%.",
					'current_value': at_risk_percentage,
					'benchmark_value': 15.0,
					'trend_direction': 'up',
					'recommendations': [
						"Proactive outreach to at-risk customers",
						"Analyze common patterns in at-risk segment",
						"Implement early warning systems",
						"Enhance onboarding and engagement programs"
					],
					'impact_score': 75.0,
					'confidence_level': 0.85,
					'data_sources': ['customer_health_analysis']
				}
				
				insight = BillingInsight(insight_data)
				self.insights[f"at_risk_health_{datetime.utcnow().strftime('%Y%m%d')}"] = insight
			
		except Exception as e:
			self.logger.error(f"Customer health insights generation failed: {e}")
	
	async def _generate_predictive_insights(self) -> None:
		"""Generate insights from predictive AI"""
		try:
			# Get high-risk predictions
			high_risk_predictions = await self.predictive_ai.get_high_risk_predictions(min_risk_score=0.8)
			
			if not high_risk_predictions:
				return
			
			# Group by prediction type
			prediction_groups = {}
			for prediction in high_risk_predictions:
				pred_type = prediction.prediction_type.value
				if pred_type not in prediction_groups:
					prediction_groups[pred_type] = []
				prediction_groups[pred_type].append(prediction)
			
			# Generate insights for each prediction type
			for pred_type, predictions in prediction_groups.items():
				total_impact = sum(pred.predicted_impact for pred in predictions)
				avg_confidence = sum(pred.confidence_score for pred in predictions) / len(predictions)
				
				if pred_type == PredictionType.INVOLUNTARY_CHURN.value:
					insight_data = {
						'metric_type': MetricType.CHURN.value,
						'priority': InsightPriority.CRITICAL.value,
						'title': f"Imminent Churn Risk: {len(predictions)} Customers",
						'description': f"AI predicts {len(predictions)} customers at high risk of involuntary churn with potential revenue impact of ${total_impact}.",
						'current_value': len(predictions),
						'predicted_impact': str(total_impact),
						'trend_direction': 'up',
						'recommendations': [
							"Immediate customer success intervention",
							"Payment method update campaigns",
							"Proactive billing communication",
							"Consider payment plan options"
						],
						'impact_score': 90.0,
						'confidence_level': avg_confidence,
						'data_sources': ['predictive_ai', 'churn_model'],
						'metadata': {
							'prediction_count': len(predictions),
							'total_predicted_impact': str(total_impact),
							'avg_confidence': avg_confidence
						}
					}
				
				elif pred_type == PredictionType.REVENUE_LEAKAGE.value:
					insight_data = {
						'metric_type': MetricType.REVENUE.value,
						'priority': InsightPriority.HIGH.value,
						'title': f"Revenue Leakage Detected: ${total_impact}",
						'description': f"AI identifies potential revenue leakage affecting {len(predictions)} accounts with ${total_impact} at risk.",
						'current_value': float(total_impact),
						'trend_direction': 'down',
						'recommendations': [
							"Audit billing coverage for high-usage accounts",
							"Update payment methods for failed charges",
							"Implement dunning management",
							"Review pricing model effectiveness"
						],
						'impact_score': 85.0,
						'confidence_level': avg_confidence,
						'data_sources': ['predictive_ai', 'revenue_model']
					}
				
				else:
					continue  # Skip other prediction types for now
				
				insight = BillingInsight(insight_data)
				self.insights[f"ai_prediction_{pred_type}_{datetime.utcnow().strftime('%Y%m%d%H')}"] = insight
				
		except Exception as e:
			self.logger.error(f"Predictive insights generation failed: {e}")
	
	async def _generate_payment_insights(self) -> None:
		"""Generate payment performance insights"""
		try:
			# Calculate payment success rate
			recent_payments = [
				p for p in self.billing_service.payments.values()
				if (datetime.utcnow() - p.created_at).days <= 7
			]
			
			if not recent_payments:
				return
			
			successful_payments = [p for p in recent_payments if p.status.value == 'succeeded']
			success_rate = len(successful_payments) / len(recent_payments) * 100
			
			# Payment success insight
			if success_rate < 90:
				priority = InsightPriority.HIGH if success_rate < 85 else InsightPriority.MEDIUM
				
				insight_data = {
					'metric_type': MetricType.PAYMENT_SUCCESS.value,
					'priority': priority.value,
					'title': f"Payment Success Rate Below Target: {success_rate:.1f}%",
					'description': f"Weekly payment success rate of {success_rate:.1f}% is below the target of 95%.",
					'current_value': success_rate,
					'benchmark_value': 95.0,
					'trend_direction': 'down',
					'recommendations': [
						"Analyze payment failure patterns",
						"Update retry logic and timing",
						"Proactive payment method validation",
						"Implement smart routing by payment type"
					],
					'impact_score': 80.0,
					'confidence_level': 0.9,
					'data_sources': ['payment_data'],
					'metadata': {
						'total_payments': len(recent_payments),
						'successful_payments': len(successful_payments),
						'failed_payments': len(recent_payments) - len(successful_payments)
					}
				}
				
				insight = BillingInsight(insight_data)
				self.insights[f"payment_success_{datetime.utcnow().strftime('%Y%m%d')}"] = insight
			
		except Exception as e:
			self.logger.error(f"Payment insights generation failed: {e}")
	
	def _cleanup_expired_insights(self) -> None:
		"""Clean up expired insights"""
		now = datetime.utcnow()
		expired_keys = [
			key for key, insight in self.insights.items()
			if insight.expires_at < now
		]
		
		for key in expired_keys:
			del self.insights[key]
		
		if expired_keys:
			self.logger.info(f"Cleaned up {len(expired_keys)} expired insights")
	
	async def get_analytics_dashboard(self, scope: AnalyticsScope = AnalyticsScope.DAILY) -> Dict[str, Any]:
		"""Get comprehensive analytics dashboard"""
		try:
			# Calculate time range based on scope
			end_date = datetime.utcnow()
			if scope == AnalyticsScope.DAILY:
				start_date = end_date - timedelta(days=1)
			elif scope == AnalyticsScope.WEEKLY:
				start_date = end_date - timedelta(days=7)
			elif scope == AnalyticsScope.MONTHLY:
				start_date = end_date - timedelta(days=30)
			else:
				start_date = end_date - timedelta(days=1)
			
			# Get core metrics
			mrr_metrics = await self.revenue_metrics.calculate_mrr_metrics()
			customer_health = await self.customer_health.calculate_customer_health_scores()
			
			# Get current insights
			current_insights = [
				{
					'id': insight.id,
					'metric_type': insight.metric_type.value,
					'priority': insight.priority.value,
					'title': insight.title,
					'description': insight.description,
					'current_value': insight.current_value,
					'trend_direction': insight.trend_direction,
					'impact_score': insight.impact_score,
					'recommendations': insight.recommendations[:3],  # Top 3 recommendations
					'created_at': insight.created_at.isoformat()
				}
				for insight in self.insights.values()
			]
			
			# Sort insights by priority and impact
			priority_order = {InsightPriority.CRITICAL: 0, InsightPriority.HIGH: 1, 
							InsightPriority.MEDIUM: 2, InsightPriority.LOW: 3}
			current_insights.sort(
				key=lambda x: (priority_order.get(InsightPriority(x['priority']), 4), -x['impact_score'])
			)
			
			# Calculate key performance indicators
			kpis = await self._calculate_kpis(start_date, end_date)
			
			return {
				'scope': scope.value,
				'period': {
					'start_date': start_date.isoformat(),
					'end_date': end_date.isoformat()
				},
				'kpis': kpis,
				'mrr_metrics': mrr_metrics,
				'customer_health': {
					'average_score': customer_health['average_health_score'],
					'distribution': customer_health['health_distribution'],
					'total_customers': customer_health['total_customers_analyzed']
				},
				'insights': current_insights[:10],  # Top 10 insights
				'insights_summary': {
					'total_insights': len(current_insights),
					'critical_insights': len([i for i in current_insights if i['priority'] == 'critical']),
					'high_priority_insights': len([i for i in current_insights if i['priority'] == 'high'])
				},
				'generated_at': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			self.logger.error(f"Analytics dashboard generation failed: {e}")
			raise
	
	async def _calculate_kpis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Calculate key performance indicators"""
		try:
			# Revenue KPIs
			period_payments = [
				p for p in self.billing_service.payments.values()
				if start_date <= p.created_at <= end_date and p.status.value == 'succeeded'
			]
			
			period_revenue = sum(p.amount for p in period_payments)
			
			# Customer KPIs
			new_customers = [
				c for c in self.billing_service.customers.values()
				if start_date <= c.created_at <= end_date
			]
			
			# Churn KPIs
			churned_subscriptions = [
				s for s in self.billing_service.subscriptions.values()
				if s.cancelled_at and start_date <= s.cancelled_at <= end_date
			]
			
			# Payment KPIs
			all_period_payments = [
				p for p in self.billing_service.payments.values()
				if start_date <= p.created_at <= end_date
			]
			
			payment_success_rate = (
				len(period_payments) / len(all_period_payments) * 100
				if all_period_payments else 0
			)
			
			return {
				'revenue': {
					'total_revenue': str(period_revenue),
					'revenue_per_customer': str(period_revenue / max(len(new_customers), 1)),
					'payment_count': len(period_payments)
				},
				'customers': {
					'new_customers': len(new_customers),
					'churned_customers': len(churned_subscriptions),
					'net_customer_growth': len(new_customers) - len(churned_subscriptions)
				},
				'payments': {
					'success_rate': payment_success_rate,
					'total_attempts': len(all_period_payments),
					'successful_payments': len(period_payments)
				}
			}
			
		except Exception as e:
			self.logger.error(f"KPI calculation failed: {e}")
			return {}
	
	async def get_insights_by_priority(self, priority: InsightPriority = None) -> List[Dict[str, Any]]:
		"""Get insights filtered by priority"""
		insights = list(self.insights.values())
		
		if priority:
			insights = [i for i in insights if i.priority == priority]
		
		# Sort by impact score and creation time
		insights.sort(key=lambda x: (-x.impact_score, -x.created_at.timestamp()))
		
		return [
			{
				'id': insight.id,
				'metric_type': insight.metric_type.value,
				'priority': insight.priority.value,
				'title': insight.title,
				'description': insight.description,
				'current_value': insight.current_value,
				'trend_direction': insight.trend_direction,
				'percentage_change': insight.percentage_change,
				'impact_score': insight.impact_score,
				'confidence_level': insight.confidence_level,
				'recommendations': insight.recommendations,
				'created_at': insight.created_at.isoformat(),
				'metadata': insight.metadata
			}
			for insight in insights
		]
	
	async def export_analytics_report(self, format: str = 'json', 
									 scope: AnalyticsScope = AnalyticsScope.MONTHLY) -> Dict[str, Any]:
		"""Export comprehensive analytics report"""
		try:
			dashboard_data = await self.get_analytics_dashboard(scope)
			insights_data = await self.get_insights_by_priority()
			
			# Get predictive analytics
			predictive_analytics = await self.predictive_ai.get_prediction_analytics(
				days=30 if scope == AnalyticsScope.MONTHLY else 7
			)
			
			report = {
				'report_type': 'comprehensive_billing_analytics',
				'generated_at': datetime.utcnow().isoformat(),
				'scope': scope.value,
				'dashboard': dashboard_data,
				'insights': insights_data,
				'predictive_analytics': predictive_analytics,
				'metadata': {
					'total_insights': len(insights_data),
					'data_sources': [
						'billing_system',
						'predictive_ai',
						'customer_health_analytics',
						'revenue_optimization'
					],
					'export_format': format
				}
			}
			
			# Log export event
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.DATA_EXPORT.value,
				'user_id': 'system',
				'resource_type': 'analytics_report',
				'resource_id': f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
				'action': 'analytics_export',
				'description': f'Comprehensive analytics report exported ({scope.value})',
				'metadata': {
					'scope': scope.value,
					'format': format,
					'insights_count': len(insights_data)
				}
			})
			
			return report
			
		except Exception as e:
			self.logger.error(f"Analytics report export failed: {e}")
			raise


# Global analytics engine
_analytics_engine_instance: Optional[ComprehensiveBillingAnalytics] = None

def get_comprehensive_billing_analytics() -> ComprehensiveBillingAnalytics:
	"""Get global comprehensive billing analytics instance"""
	global _analytics_engine_instance
	if _analytics_engine_instance is None:
		_analytics_engine_instance = ComprehensiveBillingAnalytics()
	return _analytics_engine_instance


__all__ = [
	'ComprehensiveBillingAnalytics',
	'BillingInsight',
	'AnalyticsScope',
	'MetricType',
	'InsightPriority',
	'RevenueMetrics',
	'CustomerHealthAnalytics',
	'get_comprehensive_billing_analytics'
]