"""
APG Intelligent Revenue Optimization Engine

AI-powered revenue optimization with dynamic pricing, automatic A/B testing,
elasticity analysis, and personalized pricing strategies that maximize revenue
while maintaining customer satisfaction and competitive positioning.

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import optimize
import scipy.stats as stats

from .models import BLCustomer, BLSubscription, BLPlan, BLPricingRule
from .service import get_billing_service
from .audit_compliance import get_audit_compliance_system, AuditEventType


class OptimizationType(Enum):
	"""Types of revenue optimization"""
	PRICE_ELASTICITY = "price_elasticity"
	CUSTOMER_LIFETIME_VALUE = "customer_lifetime_value"
	CONVERSION_RATE = "conversion_rate"
	CHURN_REDUCTION = "churn_reduction"
	UPSELL_OPPORTUNITY = "upsell_opportunity"
	COMPETITIVE_POSITIONING = "competitive_positioning"
	SEASONAL_ADJUSTMENT = "seasonal_adjustment"


class PricingStrategy(Enum):
	"""Pricing strategies"""
	VALUE_BASED = "value_based"
	COMPETITIVE = "competitive"
	PENETRATION = "penetration"
	PREMIUM = "premium"
	DYNAMIC = "dynamic"
	PERSONALIZED = "personalized"
	FREEMIUM = "freemium"


class OptimizationExperiment:
	"""A/B testing experiment for pricing optimization"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.name = data['name']
		self.optimization_type = OptimizationType(data['optimization_type'])
		self.strategy = PricingStrategy(data['strategy'])
		self.control_group_size = data.get('control_group_size', 0.5)
		self.test_group_size = data.get('test_group_size', 0.5)
		self.target_metric = data['target_metric']  # revenue, conversion, retention, etc.
		self.hypothesis = data['hypothesis']
		self.start_date = datetime.fromisoformat(data.get('start_date', datetime.utcnow().isoformat()))
		self.end_date = datetime.fromisoformat(data['end_date'])
		self.status = data.get('status', 'draft')  # draft, running, completed, cancelled
		self.confidence_level = data.get('confidence_level', 0.95)
		self.minimum_effect_size = data.get('minimum_effect_size', 0.05)
		self.control_settings = data.get('control_settings', {})
		self.test_settings = data.get('test_settings', {})
		self.participants = data.get('participants', [])
		self.results = data.get('results', {})
		self.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
		self.metadata = data.get('metadata', {})


class PriceOptimization:
	"""Price optimization recommendation"""
	
	def __init__(self, data: Dict[str, Any]):
		self.id = data.get('id', uuid7str())
		self.plan_id = data.get('plan_id')
		self.customer_segment = data.get('customer_segment')
		self.current_price = Decimal(str(data['current_price']))
		self.recommended_price = Decimal(str(data['recommended_price']))
		self.price_change_percent = data['price_change_percent']
		self.confidence_score = data['confidence_score']
		self.expected_revenue_impact = Decimal(str(data.get('expected_revenue_impact', 0)))
		self.expected_conversion_impact = data.get('expected_conversion_impact', 0.0)
		self.expected_churn_impact = data.get('expected_churn_impact', 0.0)
		self.reasoning = data.get('reasoning', [])
		self.market_factors = data.get('market_factors', {})
		self.customer_factors = data.get('customer_factors', {})
		self.competitive_factors = data.get('competitive_factors', {})
		self.implementation_priority = data.get('implementation_priority', 'medium')
		self.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
		self.expires_at = self.created_at + timedelta(days=data.get('validity_days', 30))


class RevenueOptimizationEngine:
	"""Intelligent revenue optimization engine"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.RevenueOptimizationEngine")
		
		# ML Models for optimization
		self.models = {
			'price_elasticity': ElasticNet(alpha=0.1, random_state=42),
			'ltv_prediction': RandomForestRegressor(n_estimators=100, random_state=42),
			'churn_probability': GradientBoostingRegressor(n_estimators=100, random_state=42),
			'conversion_rate': RandomForestRegressor(n_estimators=100, random_state=42)
		}
		
		# Feature preprocessing
		self.scalers = {model_name: StandardScaler() for model_name in self.models.keys()}
		
		# Data storage
		self.experiments: Dict[str, OptimizationExperiment] = {}
		self.optimizations: Dict[str, PriceOptimization] = {}
		self.elasticity_models: Dict[str, Any] = {}
		
		# Configuration
		self.optimization_config = {
			'min_price_change': 0.05,  # Minimum 5% price change
			'max_price_change': 0.50,  # Maximum 50% price change
			'confidence_threshold': 0.7,
			'experiment_duration_days': 30,
			'min_sample_size': 100
		}
		
		# Market intelligence
		self.market_data = {}
		self.competitive_intelligence = {}
		
		# Service integrations
		self.billing_service = get_billing_service()
		self.audit_system = get_audit_compliance_system()
		
		# Background processing
		asyncio.create_task(self._start_optimization_engine())
		asyncio.create_task(self._start_experiment_monitor())
	
	async def _start_optimization_engine(self) -> None:
		"""Start background optimization engine"""
		while True:
			try:
				await self._run_optimization_cycle()
				await asyncio.sleep(86400)  # Run daily
			except Exception as e:
				self.logger.error(f"Optimization engine error: {e}")
				await asyncio.sleep(86400)
	
	async def _start_experiment_monitor(self) -> None:
		"""Monitor running experiments"""
		while True:
			try:
				await self._monitor_experiments()
				await asyncio.sleep(3600)  # Check hourly
			except Exception as e:
				self.logger.error(f"Experiment monitor error: {e}")
				await asyncio.sleep(3600)
	
	async def _run_optimization_cycle(self) -> None:
		"""Run complete optimization cycle"""
		try:
			self.logger.info("Starting revenue optimization cycle")
			
			# Update market intelligence
			await self._update_market_intelligence()
			
			# Train optimization models
			await self._train_optimization_models()
			
			# Generate price optimizations
			await self._generate_price_optimizations()
			
			# Evaluate experiment opportunities
			await self._evaluate_experiment_opportunities()
			
			self.logger.info("Revenue optimization cycle completed")
			
		except Exception as e:
			self.logger.error(f"Optimization cycle failed: {e}")
	
	async def _update_market_intelligence(self) -> None:
		"""Update market intelligence and competitive data"""
		try:
			# Analyze internal data patterns
			await self._analyze_internal_patterns()
			
			# Update competitive intelligence
			await self._update_competitive_intelligence()
			
			# Analyze seasonal patterns
			await self._analyze_seasonal_patterns()
			
		except Exception as e:
			self.logger.error(f"Market intelligence update failed: {e}")
	
	async def _analyze_internal_patterns(self) -> None:
		"""Analyze internal pricing and conversion patterns"""
		try:
			# Analyze conversion rates by price points
			conversion_data = await self._analyze_conversion_rates()
			
			# Analyze customer lifetime value patterns
			ltv_data = await self._analyze_ltv_patterns()
			
			# Analyze churn patterns by pricing
			churn_data = await self._analyze_churn_patterns()
			
			self.market_data['internal_patterns'] = {
				'conversion_rates': conversion_data,
				'ltv_patterns': ltv_data,
				'churn_patterns': churn_data,
				'updated_at': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			self.logger.error(f"Internal pattern analysis failed: {e}")
	
	async def _analyze_conversion_rates(self) -> Dict[str, Any]:
		"""Analyze conversion rates by price points and customer segments"""
		try:
			conversion_data = {}
			
			# Get all plans and their conversion metrics
			for plan in self.billing_service.plans.values():
				plan_subscriptions = [
					sub for sub in self.billing_service.subscriptions.values()
					if sub.plan_id == plan.id
				]
				
				if not plan_subscriptions:
					continue
				
				# Calculate conversion rate (simplified - would need trial/lead data)
				conversion_rate = len([s for s in plan_subscriptions if s.status.value == 'active']) / len(plan_subscriptions)
				
				# Calculate average time to conversion
				active_subs = [s for s in plan_subscriptions if s.status.value == 'active']
				avg_time_to_convert = sum(
					(s.created_at - (s.trial_start or s.created_at)).days 
					for s in active_subs if s.trial_start
				) / max(len(active_subs), 1)
				
				conversion_data[plan.id] = {
					'plan_name': plan.name,
					'price': str(plan.price),
					'conversion_rate': conversion_rate,
					'avg_time_to_convert': avg_time_to_convert,
					'total_conversions': len(active_subs),
					'sample_size': len(plan_subscriptions)
				}
			
			return conversion_data
			
		except Exception as e:
			self.logger.error(f"Conversion rate analysis failed: {e}")
			return {}
	
	async def _analyze_ltv_patterns(self) -> Dict[str, Any]:
		"""Analyze LTV patterns by plan and customer characteristics"""
		try:
			ltv_data = {}
			
			for plan in self.billing_service.plans.values():
				plan_customers = []
				
				# Get customers with this plan
				for subscription in self.billing_service.subscriptions.values():
					if subscription.plan_id == plan.id:
						customer = self.billing_service.customers.get(subscription.customer_id)
						if customer:
							# Calculate LTV
							customer_payments = [
								p for p in self.billing_service.payments.values()
								if p.customer_id == customer.id and p.status.value == 'succeeded'
							]
							
							ltv = sum(p.amount for p in customer_payments)
							subscription_months = max(1, (datetime.utcnow() - subscription.created_at).days / 30)
							
							plan_customers.append({
								'customer_id': customer.id,
								'ltv': float(ltv),
								'subscription_months': subscription_months,
								'monthly_value': float(ltv / subscription_months),
								'customer_tier': getattr(customer, 'tier', 'standard'),
								'company_size': getattr(customer, 'company_size', 'unknown')
							})
				
				if plan_customers:
					ltv_data[plan.id] = {
						'plan_name': plan.name,
						'price': str(plan.price),
						'avg_ltv': np.mean([c['ltv'] for c in plan_customers]),
						'median_ltv': np.median([c['ltv'] for c in plan_customers]),
						'avg_monthly_value': np.mean([c['monthly_value'] for c in plan_customers]),
						'ltv_to_price_ratio': np.mean([c['ltv'] for c in plan_customers]) / float(plan.price),
						'customer_count': len(plan_customers),
						'ltv_by_tier': self._group_ltv_by_attribute(plan_customers, 'customer_tier'),
						'ltv_by_size': self._group_ltv_by_attribute(plan_customers, 'company_size')
					}
			
			return ltv_data
			
		except Exception as e:
			self.logger.error(f"LTV pattern analysis failed: {e}")
			return {}
	
	def _group_ltv_by_attribute(self, customers: List[Dict], attribute: str) -> Dict[str, float]:
		"""Group LTV by customer attribute"""
		groups = {}
		for customer in customers:
			attr_value = customer.get(attribute, 'unknown')
			if attr_value not in groups:
				groups[attr_value] = []
			groups[attr_value].append(customer['ltv'])
		
		return {
			attr_value: np.mean(ltvs) 
			for attr_value, ltvs in groups.items()
		}
	
	async def _analyze_churn_patterns(self) -> Dict[str, Any]:
		"""Analyze churn patterns by pricing and plan characteristics"""
		try:
			churn_data = {}
			
			for plan in self.billing_service.plans.values():
				plan_subscriptions = [
					sub for sub in self.billing_service.subscriptions.values()
					if sub.plan_id == plan.id
				]
				
				if not plan_subscriptions:
					continue
				
				# Calculate churn metrics
				total_subs = len(plan_subscriptions)
				churned_subs = len([s for s in plan_subscriptions if s.status.value in ['cancelled', 'expired']])
				churn_rate = churned_subs / total_subs if total_subs > 0 else 0
				
				# Calculate average subscription duration
				ended_subs = [s for s in plan_subscriptions if s.ended_at]
				avg_duration = np.mean([
					(s.ended_at - s.created_at).days 
					for s in ended_subs
				]) if ended_subs else 0
				
				# Analyze churn reasons (from metadata)
				churn_reasons = {}
				for sub in [s for s in plan_subscriptions if s.status.value == 'cancelled']:
					reason = sub.metadata.get('cancellation_reason', 'unknown') if sub.metadata else 'unknown'
					churn_reasons[reason] = churn_reasons.get(reason, 0) + 1
				
				churn_data[plan.id] = {
					'plan_name': plan.name,
					'price': str(plan.price),
					'churn_rate': churn_rate,
					'avg_duration_days': avg_duration,
					'total_subscriptions': total_subs,
					'churned_count': churned_subs,
					'churn_reasons': churn_reasons
				}
			
			return churn_data
			
		except Exception as e:
			self.logger.error(f"Churn pattern analysis failed: {e}")
			return {}
	
	async def _update_competitive_intelligence(self) -> None:
		"""Update competitive pricing intelligence"""
		try:
			# In production, this would integrate with competitive intelligence APIs
			# For now, we'll use placeholder data with realistic competitive scenarios
			
			self.competitive_intelligence = {
				'market_position': 'mid_market',
				'price_compared_to_competitors': {
					'premium_competitor': 1.2,  # 20% higher than us
					'budget_competitor': 0.8,   # 20% lower than us
					'direct_competitor': 1.05   # 5% higher than us
				},
				'market_trends': {
					'overall_pricing_trend': 'increasing',
					'price_change_velocity': 0.02,  # 2% per quarter
					'market_saturation': 0.6,
					'new_entrant_pressure': 'medium'
				},
				'feature_comparison': {
					'feature_parity_score': 0.85,
					'unique_value_props': ['ai_optimization', 'predictive_billing'],
					'competitive_gaps': ['enterprise_reporting', 'advanced_analytics']
				},
				'updated_at': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			self.logger.error(f"Competitive intelligence update failed: {e}")
	
	async def _analyze_seasonal_patterns(self) -> None:
		"""Analyze seasonal patterns in revenue and conversions"""
		try:
			# Analyze revenue by month/quarter
			monthly_revenue = {}
			quarterly_revenue = {}
			
			for payment in self.billing_service.payments.values():
				if payment.status.value == 'succeeded':
					month_key = payment.created_at.strftime('%Y-%m')
					quarter_key = f"{payment.created_at.year}-Q{(payment.created_at.month-1)//3 + 1}"
					
					monthly_revenue[month_key] = monthly_revenue.get(month_key, Decimal('0')) + payment.amount
					quarterly_revenue[quarter_key] = quarterly_revenue.get(quarter_key, Decimal('0')) + payment.amount
			
			# Calculate seasonal indices
			if len(monthly_revenue) >= 12:
				seasonal_indices = self._calculate_seasonal_indices(monthly_revenue)
			else:
				seasonal_indices = {}
			
			self.market_data['seasonal_patterns'] = {
				'monthly_revenue': {k: str(v) for k, v in monthly_revenue.items()},
				'quarterly_revenue': {k: str(v) for k, v in quarterly_revenue.items()},
				'seasonal_indices': seasonal_indices,
				'peak_season': max(seasonal_indices.items(), key=lambda x: x[1])[0] if seasonal_indices else None,
				'low_season': min(seasonal_indices.items(), key=lambda x: x[1])[0] if seasonal_indices else None
			}
			
		except Exception as e:
			self.logger.error(f"Seasonal pattern analysis failed: {e}")
	
	def _calculate_seasonal_indices(self, monthly_data: Dict[str, Decimal]) -> Dict[str, float]:
		"""Calculate seasonal indices for monthly data"""
		try:
			# Convert to list and calculate 12-month moving average
			sorted_months = sorted(monthly_data.keys())
			values = [float(monthly_data[month]) for month in sorted_months]
			
			if len(values) < 12:
				return {}
			
			# Calculate seasonal indices for each month (1-12)
			seasonal_indices = {}
			for month_num in range(1, 13):
				month_values = [values[i] for i in range(len(values)) if (i % 12) == (month_num - 1)]
				if month_values:
					avg_for_month = np.mean(month_values)
					overall_avg = np.mean(values)
					seasonal_indices[f"month_{month_num}"] = avg_for_month / overall_avg if overall_avg > 0 else 1.0
			
			return seasonal_indices
			
		except Exception as e:
			self.logger.error(f"Seasonal index calculation failed: {e}")
			return {}
	
	async def _train_optimization_models(self) -> None:
		"""Train ML models for optimization"""
		try:
			# Prepare training data
			training_data = await self._prepare_training_data()
			
			if not training_data or len(training_data) < 50:
				self.logger.warning("Insufficient data for model training")
				return
			
			df = pd.DataFrame(training_data)
			
			# Train price elasticity model
			await self._train_elasticity_model(df)
			
			# Train LTV prediction model
			await self._train_ltv_model(df)
			
			# Train conversion rate model
			await self._train_conversion_model(df)
			
			self.logger.info("Optimization models trained successfully")
			
		except Exception as e:
			self.logger.error(f"Model training failed: {e}")
	
	async def _prepare_training_data(self) -> List[Dict[str, Any]]:
		"""Prepare training data for optimization models"""
		try:
			training_data = []
			
			# Get subscription data with outcomes
			for subscription in self.billing_service.subscriptions.values():
				customer = self.billing_service.customers.get(subscription.customer_id)
				plan = self.billing_service.plans.get(subscription.plan_id)
				
				if not customer or not plan:
					continue
				
				# Calculate outcomes
				customer_payments = [
					p for p in self.billing_service.payments.values()
					if p.customer_id == customer.id and p.status.value == 'succeeded'
				]
				
				ltv = sum(p.amount for p in customer_payments)
				subscription_duration = (datetime.utcnow() - subscription.created_at).days
				is_churned = subscription.status.value in ['cancelled', 'expired']
				
				# Prepare features
				features = {
					'plan_price': float(plan.price),
					'customer_tier': customer.tier if hasattr(customer, 'tier') else 'standard',
					'company_size': customer.company_size if hasattr(customer, 'company_size') else 'unknown',
					'subscription_duration': subscription_duration,
					'ltv': float(ltv),
					'is_churned': is_churned,
					'trial_used': subscription.trial_start is not None,
					'payment_failures': len([p for p in customer_payments if p.status.value == 'failed']),
					'signup_month': subscription.created_at.month,
					'signup_year': subscription.created_at.year
				}
				
				training_data.append(features)
			
			return training_data
			
		except Exception as e:
			self.logger.error(f"Training data preparation failed: {e}")
			return []
	
	async def _train_elasticity_model(self, df: pd.DataFrame) -> None:
		"""Train price elasticity model"""
		try:
			# Prepare features for elasticity modeling
			features = ['customer_tier', 'company_size', 'trial_used', 'signup_month']
			
			# Encode categorical variables
			df_encoded = pd.get_dummies(df[features + ['plan_price', 'ltv']])
			
			X = df_encoded.drop(['ltv'], axis=1)
			y = df_encoded['ltv']
			
			if len(X) < 30:
				return
			
			# Split and train
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
			
			# Scale features
			X_train_scaled = self.scalers['price_elasticity'].fit_transform(X_train)
			X_test_scaled = self.scalers['price_elasticity'].transform(X_test)
			
			# Train model
			self.models['price_elasticity'].fit(X_train_scaled, y_train)
			
			# Calculate elasticity coefficients
			if hasattr(self.models['price_elasticity'], 'coef_'):
				price_coef_idx = [i for i, col in enumerate(X.columns) if 'plan_price' in col]
				if price_coef_idx:
					elasticity_coef = self.models['price_elasticity'].coef_[price_coef_idx[0]]
					self.elasticity_models['global_elasticity'] = elasticity_coef
			
		except Exception as e:
			self.logger.error(f"Elasticity model training failed: {e}")
	
	async def _train_ltv_model(self, df: pd.DataFrame) -> None:
		"""Train LTV prediction model"""
		try:
			features = ['plan_price', 'customer_tier', 'company_size', 'trial_used', 'signup_month']
			
			# Encode categorical variables
			df_encoded = pd.get_dummies(df[features + ['ltv']])
			
			X = df_encoded.drop(['ltv'], axis=1)
			y = df_encoded['ltv']
			
			if len(X) < 30:
				return
			
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
			
			# Train model
			self.models['ltv_prediction'].fit(X_train, y_train)
			
		except Exception as e:
			self.logger.error(f"LTV model training failed: {e}")
	
	async def _train_conversion_model(self, df: pd.DataFrame) -> None:
		"""Train conversion rate model"""
		try:
			# For conversion, we'll use subscription success as proxy
			df['converted'] = (~df['is_churned']).astype(int)
			
			features = ['plan_price', 'customer_tier', 'company_size', 'trial_used', 'signup_month']
			df_encoded = pd.get_dummies(df[features + ['converted']])
			
			X = df_encoded.drop(['converted'], axis=1)
			y = df_encoded['converted']
			
			if len(X) < 30:
				return
			
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
			
			# Train model
			self.models['conversion_rate'].fit(X_train, y_train)
			
		except Exception as e:
			self.logger.error(f"Conversion model training failed: {e}")
	
	async def _generate_price_optimizations(self) -> None:
		"""Generate price optimization recommendations"""
		try:
			for plan in self.billing_service.plans.values():
				optimization = await self._optimize_plan_pricing(plan)
				if optimization:
					self.optimizations[optimization.id] = optimization
			
			self.logger.info(f"Generated {len(self.optimizations)} price optimizations")
			
		except Exception as e:
			self.logger.error(f"Price optimization generation failed: {e}")
	
	async def _optimize_plan_pricing(self, plan: BLPlan) -> Optional[PriceOptimization]:
		"""Optimize pricing for a specific plan"""
		try:
			current_price = plan.price
			
			# Calculate optimal price using multiple approaches
			elasticity_price = await self._calculate_elasticity_optimal_price(plan)
			ltv_price = await self._calculate_ltv_optimal_price(plan)
			competitive_price = await self._calculate_competitive_optimal_price(plan)
			
			# Weight the different approaches
			price_recommendations = []
			if elasticity_price:
				price_recommendations.append(('elasticity', elasticity_price, 0.4))
			if ltv_price:
				price_recommendations.append(('ltv', ltv_price, 0.4))
			if competitive_price:
				price_recommendations.append(('competitive', competitive_price, 0.2))
			
			if not price_recommendations:
				return None
			
			# Calculate weighted average
			total_weight = sum(weight for _, _, weight in price_recommendations)
			recommended_price = sum(
				price * weight for _, price, weight in price_recommendations
			) / total_weight
			
			# Apply constraints
			min_price = current_price * (1 - self.optimization_config['max_price_change'])
			max_price = current_price * (1 + self.optimization_config['max_price_change'])
			recommended_price = max(min_price, min(max_price, recommended_price))
			
			# Calculate price change percentage
			price_change_percent = float((recommended_price - current_price) / current_price)
			
			# Skip if change is too small
			if abs(price_change_percent) < self.optimization_config['min_price_change']:
				return None
			
			# Calculate expected impacts
			expected_impacts = await self._calculate_expected_impacts(plan, recommended_price)
			
			# Generate reasoning
			reasoning = self._generate_optimization_reasoning(
				plan, current_price, recommended_price, price_recommendations, expected_impacts
			)
			
			optimization_data = {
				'plan_id': plan.id,
				'current_price': current_price,
				'recommended_price': recommended_price,
				'price_change_percent': price_change_percent,
				'confidence_score': self._calculate_confidence_score(price_recommendations, expected_impacts),
				'expected_revenue_impact': expected_impacts['revenue_impact'],
				'expected_conversion_impact': expected_impacts['conversion_impact'],
				'expected_churn_impact': expected_impacts['churn_impact'],
				'reasoning': reasoning,
				'market_factors': self.competitive_intelligence,
				'implementation_priority': self._calculate_implementation_priority(expected_impacts)
			}
			
			return PriceOptimization(optimization_data)
			
		except Exception as e:
			self.logger.error(f"Plan pricing optimization failed for {plan.id}: {e}")
			return None
	
	async def _calculate_elasticity_optimal_price(self, plan: BLPlan) -> Optional[Decimal]:
		"""Calculate optimal price based on price elasticity"""
		try:
			# Get elasticity from stored models or calculate
			elasticity = self.elasticity_models.get('global_elasticity', -1.5)  # Default elasticity
			
			# Calculate optimal price using elasticity theory
			# For elastic demand (elasticity < -1), optimal price is where MR = MC
			# Simplified: optimal price = current_price * (1 + 1/elasticity)
			
			if elasticity < -1:  # Elastic demand
				optimal_multiplier = 1 + (1 / elasticity)
				optimal_price = plan.price * Decimal(str(optimal_multiplier))
			else:  # Inelastic demand
				# For inelastic demand, can increase price
				optimal_price = plan.price * Decimal('1.1')  # 10% increase
			
			return optimal_price
			
		except Exception as e:
			self.logger.error(f"Elasticity price calculation failed: {e}")
			return None
	
	async def _calculate_ltv_optimal_price(self, plan: BLPlan) -> Optional[Decimal]:
		"""Calculate optimal price based on LTV maximization"""
		try:
			# Use LTV model to find price that maximizes LTV
			plan_data = self.market_data.get('internal_patterns', {}).get('ltv_patterns', {}).get(plan.id)
			
			if not plan_data:
				return None
			
			current_ltv_ratio = plan_data.get('ltv_to_price_ratio', 3.0)
			
			# If LTV ratio is high, can increase price
			if current_ltv_ratio > 5.0:
				return plan.price * Decimal('1.15')  # 15% increase
			elif current_ltv_ratio > 3.0:
				return plan.price * Decimal('1.08')  # 8% increase
			elif current_ltv_ratio < 2.0:
				return plan.price * Decimal('0.95')  # 5% decrease
			
			return plan.price  # No change recommended
			
		except Exception as e:
			self.logger.error(f"LTV price calculation failed: {e}")
			return None
	
	async def _calculate_competitive_optimal_price(self, plan: BLPlan) -> Optional[Decimal]:
		"""Calculate optimal price based on competitive positioning"""
		try:
			competitive_data = self.competitive_intelligence.get('price_compared_to_competitors', {})
			
			if not competitive_data:
				return None
			
			# Average competitor multiplier
			avg_competitor_multiplier = np.mean(list(competitive_data.values()))
			
			# If we're significantly below market, can increase
			if avg_competitor_multiplier > 1.15:
				return plan.price * Decimal('1.10')
			elif avg_competitor_multiplier < 0.9:
				return plan.price * Decimal('0.95')
			
			return plan.price
			
		except Exception as e:
			self.logger.error(f"Competitive price calculation failed: {e}")
			return None
	
	async def _calculate_expected_impacts(self, plan: BLPlan, new_price: Decimal) -> Dict[str, Any]:
		"""Calculate expected impacts of price change"""
		try:
			price_change_ratio = float(new_price / plan.price)
			
			# Use elasticity to estimate impacts
			elasticity = self.elasticity_models.get('global_elasticity', -1.5)
			
			# Revenue impact = (1 + elasticity * price_change_percent)
			price_change_percent = price_change_ratio - 1
			revenue_multiplier = 1 + (elasticity * price_change_percent)
			
			# Get current metrics
			plan_data = self.market_data.get('internal_patterns', {})
			conversion_data = plan_data.get('conversion_rates', {}).get(plan.id, {})
			churn_data = plan_data.get('churn_patterns', {}).get(plan.id, {})
			
			current_revenue = float(plan.price) * conversion_data.get('total_conversions', 10)
			expected_revenue = current_revenue * revenue_multiplier
			revenue_impact = expected_revenue - current_revenue
			
			# Conversion impact (inverse relationship with price)
			conversion_impact = elasticity * price_change_percent * 0.5  # Dampened effect
			
			# Churn impact (higher prices may increase churn)
			base_churn_rate = churn_data.get('churn_rate', 0.1)
			churn_impact = price_change_percent * 0.3 * base_churn_rate  # 30% of price change affects churn
			
			return {
				'revenue_impact': Decimal(str(revenue_impact)),
				'conversion_impact': conversion_impact,
				'churn_impact': churn_impact,
				'price_change_ratio': price_change_ratio
			}
			
		except Exception as e:
			self.logger.error(f"Impact calculation failed: {e}")
			return {
				'revenue_impact': Decimal('0'),
				'conversion_impact': 0.0,
				'churn_impact': 0.0,
				'price_change_ratio': 1.0
			}
	
	def _generate_optimization_reasoning(self, plan: BLPlan, current_price: Decimal, 
		recommended_price: Decimal, price_recommendations: List[Tuple], 
		expected_impacts: Dict[str, Any]) -> List[str]:
		"""Generate human-readable reasoning for optimization"""
		reasoning = []
		
		price_change_percent = float((recommended_price - current_price) / current_price * 100)
		
		# Price direction reasoning
		if price_change_percent > 0:
			reasoning.append(f"Recommended {price_change_percent:.1f}% price increase to ${recommended_price}")
		else:
			reasoning.append(f"Recommended {abs(price_change_percent):.1f}% price decrease to ${recommended_price}")
		
		# Method-specific reasoning
		for method, price, weight in price_recommendations:
			if method == 'elasticity':
				reasoning.append(f"Price elasticity analysis suggests ${price:.2f} optimal price")
			elif method == 'ltv':
				reasoning.append(f"LTV optimization indicates ${price:.2f} maximizes customer lifetime value")
			elif method == 'competitive':
				reasoning.append(f"Competitive analysis supports ${price:.2f} positioning")
		
		# Impact reasoning
		revenue_impact = expected_impacts['revenue_impact']
		if revenue_impact > 0:
			reasoning.append(f"Expected revenue increase of ${revenue_impact:.2f}")
		else:
			reasoning.append(f"Expected revenue impact of ${revenue_impact:.2f}")
		
		# Market context
		market_trends = self.competitive_intelligence.get('market_trends', {})
		if market_trends.get('overall_pricing_trend') == 'increasing':
			reasoning.append("Market pricing trend supports upward adjustment")
		
		return reasoning
	
	def _calculate_confidence_score(self, price_recommendations: List[Tuple], 
		expected_impacts: Dict[str, Any]) -> float:
		"""Calculate confidence score for optimization"""
		base_confidence = 0.5
		
		# More methods = higher confidence
		method_confidence = len(price_recommendations) * 0.15
		
		# Lower variability in recommendations = higher confidence
		if len(price_recommendations) > 1:
			prices = [price for _, price, _ in price_recommendations]
			price_std = np.std(prices)
			price_mean = np.mean(prices)
			variability_penalty = (price_std / price_mean) * 0.2 if price_mean > 0 else 0.2
		else:
			variability_penalty = 0.1
		
		# Positive revenue impact = higher confidence
		revenue_impact = float(expected_impacts['revenue_impact'])
		if revenue_impact > 0:
			impact_bonus = 0.2
		else:
			impact_bonus = -0.1
		
		confidence = base_confidence + method_confidence - variability_penalty + impact_bonus
		return max(0.0, min(1.0, confidence))
	
	def _calculate_implementation_priority(self, expected_impacts: Dict[str, Any]) -> str:
		"""Calculate implementation priority"""
		revenue_impact = float(expected_impacts['revenue_impact'])
		
		if revenue_impact > 1000:
			return 'high'
		elif revenue_impact > 100:
			return 'medium'
		else:
			return 'low'
	
	async def _evaluate_experiment_opportunities(self) -> None:
		"""Evaluate opportunities for A/B testing experiments"""
		try:
			# Look for plans with uncertain optimizations
			for optimization in self.optimizations.values():
				if (optimization.confidence_score < 0.8 and 
					abs(optimization.price_change_percent) > 0.1):
					
					experiment = await self._create_pricing_experiment(optimization)
					if experiment:
						self.experiments[experiment.id] = experiment
			
		except Exception as e:
			self.logger.error(f"Experiment evaluation failed: {e}")
	
	async def _create_pricing_experiment(self, optimization: PriceOptimization) -> Optional[OptimizationExperiment]:
		"""Create A/B testing experiment for price optimization"""
		try:
			plan = self.billing_service.plans.get(optimization.plan_id)
			if not plan:
				return None
			
			experiment_data = {
				'name': f"Price Test: {plan.name} - {optimization.price_change_percent:.1%} change",
				'optimization_type': OptimizationType.PRICE_ELASTICITY.value,
				'strategy': PricingStrategy.DYNAMIC.value,
				'target_metric': 'revenue',
				'hypothesis': f"Changing price by {optimization.price_change_percent:.1%} will increase revenue",
				'end_date': (datetime.utcnow() + timedelta(days=self.optimization_config['experiment_duration_days'])).isoformat(),
				'control_settings': {'price': str(optimization.current_price)},
				'test_settings': {'price': str(optimization.recommended_price)},
				'minimum_effect_size': 0.05
			}
			
			return OptimizationExperiment(experiment_data)
			
		except Exception as e:
			self.logger.error(f"Experiment creation failed: {e}")
			return None
	
	async def _monitor_experiments(self) -> None:
		"""Monitor running experiments"""
		try:
			for experiment in self.experiments.values():
				if experiment.status == 'running':
					await self._check_experiment_completion(experiment)
					await self._update_experiment_results(experiment)
		
		except Exception as e:
			self.logger.error(f"Experiment monitoring failed: {e}")
	
	async def _check_experiment_completion(self, experiment: OptimizationExperiment) -> None:
		"""Check if experiment should be completed"""
		try:
			now = datetime.utcnow()
			
			# Check if experiment duration reached
			if now >= experiment.end_date:
				await self._complete_experiment(experiment)
			
			# Check for statistical significance
			if len(experiment.participants) >= self.optimization_config['min_sample_size']:
				significance = await self._check_statistical_significance(experiment)
				if significance and significance['is_significant']:
					await self._complete_experiment(experiment)
		
		except Exception as e:
			self.logger.error(f"Experiment completion check failed: {e}")
	
	async def _check_statistical_significance(self, experiment: OptimizationExperiment) -> Optional[Dict[str, Any]]:
		"""Check statistical significance of experiment results"""
		try:
			# Simplified statistical test - in production would use proper A/B testing
			control_metrics = experiment.results.get('control_group', {})
			test_metrics = experiment.results.get('test_group', {})
			
			if not control_metrics or not test_metrics:
				return None
			
			# Perform t-test on revenue per customer
			control_revenue = control_metrics.get('avg_revenue_per_customer', 0)
			test_revenue = test_metrics.get('avg_revenue_per_customer', 0)
			
			# Simplified significance test
			effect_size = abs(test_revenue - control_revenue) / max(control_revenue, 1)
			is_significant = effect_size >= experiment.minimum_effect_size
			
			return {
				'is_significant': is_significant,
				'effect_size': effect_size,
				'control_revenue': control_revenue,
				'test_revenue': test_revenue,
				'improvement': (test_revenue - control_revenue) / max(control_revenue, 1)
			}
			
		except Exception as e:
			self.logger.error(f"Statistical significance check failed: {e}")
			return None
	
	async def _complete_experiment(self, experiment: OptimizationExperiment) -> None:
		"""Complete an experiment and analyze results"""
		try:
			experiment.status = 'completed'
			
			# Calculate final results
			final_results = await self._calculate_final_experiment_results(experiment)
			experiment.results['final_analysis'] = final_results
			
			# Log completion
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.COMPLIANCE_CHECK.value,
				'user_id': 'system',
				'resource_type': 'pricing_experiment',
				'resource_id': experiment.id,
				'action': 'experiment_completed',
				'description': f'Pricing experiment completed: {experiment.name}',
				'metadata': {
					'results': final_results,
					'duration_days': (datetime.utcnow() - experiment.start_date).days
				}
			})
			
			self.logger.info(f"Experiment completed: {experiment.name}")
			
		except Exception as e:
			self.logger.error(f"Experiment completion failed: {e}")
	
	async def _calculate_final_experiment_results(self, experiment: OptimizationExperiment) -> Dict[str, Any]:
		"""Calculate final results for completed experiment"""
		try:
			control_metrics = experiment.results.get('control_group', {})
			test_metrics = experiment.results.get('test_group', {})
			
			if not control_metrics or not test_metrics:
				return {'status': 'insufficient_data'}
			
			# Calculate key metrics
			control_revenue = control_metrics.get('total_revenue', 0)
			test_revenue = test_metrics.get('total_revenue', 0)
			
			revenue_improvement = (test_revenue - control_revenue) / max(control_revenue, 1)
			
			# Determine winner
			winner = 'test' if test_revenue > control_revenue else 'control'
			
			# Calculate confidence
			confidence = min(0.95, abs(revenue_improvement) * 10)  # Simplified confidence
			
			return {
				'status': 'completed',
				'winner': winner,
				'revenue_improvement': revenue_improvement,
				'confidence_level': confidence,
				'control_revenue': control_revenue,
				'test_revenue': test_revenue,
				'participants': len(experiment.participants),
				'recommendation': 'implement' if winner == 'test' and confidence > 0.8 else 'reject'
			}
			
		except Exception as e:
			self.logger.error(f"Final results calculation failed: {e}")
			return {'status': 'error'}
	
	# Public API methods
	
	async def get_optimization_recommendations(self, plan_id: str = None) -> List[PriceOptimization]:
		"""Get current optimization recommendations"""
		optimizations = list(self.optimizations.values())
		
		if plan_id:
			optimizations = [opt for opt in optimizations if opt.plan_id == plan_id]
		
		# Filter active recommendations
		now = datetime.utcnow()
		active_optimizations = [opt for opt in optimizations if opt.expires_at > now]
		
		# Sort by expected revenue impact
		return sorted(active_optimizations, key=lambda x: x.expected_revenue_impact, reverse=True)
	
	async def create_pricing_experiment(self, optimization_id: str, experiment_config: Dict[str, Any]) -> OptimizationExperiment:
		"""Create a new pricing experiment"""
		optimization = self.optimizations.get(optimization_id)
		if not optimization:
			raise ValueError(f"Optimization {optimization_id} not found")
		
		experiment_data = {
			**experiment_config,
			'optimization_type': OptimizationType.PRICE_ELASTICITY.value,
			'strategy': PricingStrategy.DYNAMIC.value
		}
		
		experiment = OptimizationExperiment(experiment_data)
		self.experiments[experiment.id] = experiment
		
		return experiment
	
	async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
		"""Get results for a specific experiment"""
		experiment = self.experiments.get(experiment_id)
		if not experiment:
			raise ValueError(f"Experiment {experiment_id} not found")
		
		return experiment.results
	
	async def get_revenue_analytics(self, days: int = 30) -> Dict[str, Any]:
		"""Get revenue optimization analytics"""
		cutoff_date = datetime.utcnow() - timedelta(days=days)
		
		# Calculate metrics
		total_optimizations = len(self.optimizations)
		active_optimizations = len([
			opt for opt in self.optimizations.values() 
			if opt.expires_at > datetime.utcnow()
		])
		
		total_expected_impact = sum(
			opt.expected_revenue_impact for opt in self.optimizations.values()
		)
		
		completed_experiments = len([
			exp for exp in self.experiments.values() 
			if exp.status == 'completed'
		])
		
		return {
			'period_days': days,
			'total_optimizations': total_optimizations,
			'active_optimizations': active_optimizations,
			'total_expected_revenue_impact': str(total_expected_impact),
			'completed_experiments': completed_experiments,
			'market_intelligence': self.competitive_intelligence,
			'seasonal_patterns': self.market_data.get('seasonal_patterns', {}),
			'generated_at': datetime.utcnow().isoformat()
		}


# Global revenue optimization engine
_revenue_optimization_instance: Optional[RevenueOptimizationEngine] = None

def get_revenue_optimization_engine() -> RevenueOptimizationEngine:
	"""Get global revenue optimization engine instance"""
	global _revenue_optimization_instance
	if _revenue_optimization_instance is None:
		_revenue_optimization_instance = RevenueOptimizationEngine()
	return _revenue_optimization_instance


__all__ = [
	'RevenueOptimizationEngine',
	'PriceOptimization',
	'OptimizationExperiment',
	'OptimizationType',
	'PricingStrategy',
	'get_revenue_optimization_engine'
]