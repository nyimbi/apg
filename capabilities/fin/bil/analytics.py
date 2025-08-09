"""
APG Billing Analytics Engine

Advanced analytics and reporting engine for billing metrics, revenue optimization,
customer insights, and predictive analytics using AI.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .models import (
	BLCustomer, BLPlan, BLSubscription, BLUsage, BLInvoice, BLPayment,
	SubscriptionStatus, InvoiceStatus, PaymentStatus, BillingCurrency
)
from .service import get_billing_service


class BillingAnalyticsEngine:
	"""
	Advanced billing analytics engine with AI-powered insights
	
	Features:
	- Revenue analytics and forecasting
	- Customer lifetime value analysis
	- Churn prediction and prevention
	- Usage pattern analysis
	- Pricing optimization recommendations
	- Financial reporting and compliance
	"""
	
	def __init__(self):
		self.billing_service = get_billing_service()
		self.logger = logging.getLogger(f"{__name__}.BillingAnalyticsEngine")
		
		# Cache for analytics results
		self.analytics_cache: Dict[str, Dict[str, Any]] = {}
		self.cache_ttl = timedelta(minutes=15)
		
		# AI integration
		self._ai_orchestration_available = False
		self._initialize_ai_integration()
	
	def _initialize_ai_integration(self) -> None:
		"""Initialize AI orchestration for advanced analytics"""
		try:
			from capabilities.common.ai_orchestration import get_orchestration_service
			self.ai_orchestration = get_orchestration_service()
			self._ai_orchestration_available = True
			self.logger.info("✅ AI orchestration integration initialized")
		except ImportError:
			self.logger.warning("⚠️  AI orchestration service not available")
	
	def _get_cache_key(self, method: str, **kwargs) -> str:
		"""Generate cache key for analytics results"""
		key_parts = [method]
		for k, v in sorted(kwargs.items()):
			if isinstance(v, datetime):
				v = v.isoformat()
			key_parts.append(f"{k}:{v}")
		return "|".join(key_parts)
	
	def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
		"""Check if cache entry is still valid"""
		cached_at = datetime.fromisoformat(cache_entry["cached_at"])
		return datetime.utcnow() - cached_at < self.cache_ttl
	
	def _cache_result(self, cache_key: str, result: Any) -> None:
		"""Cache analytics result"""
		self.analytics_cache[cache_key] = {
			"result": result,
			"cached_at": datetime.utcnow().isoformat()
		}
	
	def _get_cached_result(self, cache_key: str) -> Optional[Any]:
		"""Get cached result if valid"""
		if cache_key in self.analytics_cache:
			cache_entry = self.analytics_cache[cache_key]
			if self._is_cache_valid(cache_entry):
				return cache_entry["result"]
		return None
	
	# Revenue Analytics
	
	async def get_revenue_analytics(self, tenant_id: str, period_start: datetime = None, period_end: datetime = None) -> Dict[str, Any]:
		"""Get comprehensive revenue analytics"""
		cache_key = self._get_cache_key("revenue_analytics", tenant_id=tenant_id, start=period_start, end=period_end)
		cached_result = self._get_cached_result(cache_key)
		if cached_result:
			return cached_result
		
		# Default to last 30 days
		if not period_end:
			period_end = datetime.utcnow()
		if not period_start:
			period_start = period_end - timedelta(days=30)
		
		# Calculate revenue metrics
		total_revenue = Decimal('0')
		recognized_revenue = Decimal('0')
		deferred_revenue = Decimal('0')
		
		revenue_by_day = defaultdict(lambda: Decimal('0'))
		revenue_by_currency = defaultdict(lambda: Decimal('0'))
		revenue_by_product = defaultdict(lambda: Decimal('0'))
		
		# Analyze invoices for revenue data
		for invoice in self.billing_service.invoices.values():
			if (invoice.tenant_id == tenant_id and
				invoice.invoice_date >= period_start and
				invoice.invoice_date <= period_end and
				invoice.status == InvoiceStatus.PAID):
				
				total_revenue += invoice.total
				
				# Revenue by day
				day_key = invoice.invoice_date.date()
				revenue_by_day[day_key] += invoice.total
				
				# Revenue by currency
				revenue_by_currency[invoice.currency.value] += invoice.total
				
				# Revenue by subscription/product
				if invoice.subscription_id:
					subscription = self.billing_service.subscriptions.get(invoice.subscription_id)
					if subscription:
						plan = self.billing_service.plans.get(subscription.plan_id)
						if plan:
							revenue_by_product[plan.name] += invoice.total
		
		# Calculate growth metrics
		previous_period_start = period_start - (period_end - period_start)
		previous_period_end = period_start
		previous_revenue = await self._calculate_period_revenue(tenant_id, previous_period_start, previous_period_end)
		
		revenue_growth = float(((total_revenue - previous_revenue) / max(previous_revenue, Decimal('0.01'))) * 100)
		
		# Monthly Recurring Revenue (MRR) calculation
		mrr = await self._calculate_mrr(tenant_id)
		
		# Annual Recurring Revenue (ARR)
		arr = mrr * 12
		
		result = {
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"total_revenue": str(total_revenue),
			"recognized_revenue": str(recognized_revenue),
			"deferred_revenue": str(deferred_revenue),
			"revenue_growth_percentage": revenue_growth,
			"mrr": str(mrr),
			"arr": str(arr),
			"revenue_by_day": {str(k): str(v) for k, v in revenue_by_day.items()},
			"revenue_by_currency": {k: str(v) for k, v in revenue_by_currency.items()},
			"revenue_by_product": {k: str(v) for k, v in revenue_by_product.items()},
			"average_daily_revenue": str(total_revenue / max((period_end - period_start).days, 1))
		}
		
		self._cache_result(cache_key, result)
		return result
	
	async def _calculate_period_revenue(self, tenant_id: str, start: datetime, end: datetime) -> Decimal:
		"""Calculate revenue for a specific period"""
		revenue = Decimal('0')
		for invoice in self.billing_service.invoices.values():
			if (invoice.tenant_id == tenant_id and
				invoice.invoice_date >= start and
				invoice.invoice_date <= end and
				invoice.status == InvoiceStatus.PAID):
				revenue += invoice.total
		return revenue
	
	async def _calculate_mrr(self, tenant_id: str) -> Decimal:
		"""Calculate Monthly Recurring Revenue"""
		mrr = Decimal('0')
		
		for subscription in self.billing_service.subscriptions.values():
			if (subscription.tenant_id == tenant_id and
				subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]):
				
				plan = self.billing_service.plans.get(subscription.plan_id)
				if plan:
					# Convert to monthly revenue
					monthly_price = plan.base_price
					if plan.billing_period.value == "yearly":
						monthly_price = plan.base_price / 12
					elif plan.billing_period.value == "quarterly":
						monthly_price = plan.base_price / 3
					elif plan.billing_period.value == "weekly":
						monthly_price = plan.base_price * 4.33  # Average weeks per month
					elif plan.billing_period.value == "daily":
						monthly_price = plan.base_price * 30
					
					mrr += monthly_price
		
		return mrr
	
	# Customer Analytics
	
	async def get_customer_analytics(self, tenant_id: str, period_start: datetime = None, period_end: datetime = None) -> Dict[str, Any]:
		"""Get comprehensive customer analytics"""
		cache_key = self._get_cache_key("customer_analytics", tenant_id=tenant_id, start=period_start, end=period_end)
		cached_result = self._get_cached_result(cache_key)
		if cached_result:
			return cached_result
		
		# Default to last 30 days
		if not period_end:
			period_end = datetime.utcnow()
		if not period_start:
			period_start = period_end - timedelta(days=30)
		
		# Customer metrics
		total_customers = 0
		active_customers = 0
		new_customers = 0
		churned_customers = 0
		
		customer_segments = defaultdict(int)
		customers_by_plan = defaultdict(int)
		
		for customer in self.billing_service.customers.values():
			if customer.tenant_id == tenant_id:
				total_customers += 1
				
				if customer.active:
					active_customers += 1
				
				# New customers in period
				if customer.created_at >= period_start:
					new_customers += 1
				
				# Customer segmentation
				customer_value = await self._calculate_customer_lifetime_value(customer.id)
				if customer_value > 1000:
					customer_segments["high_value"] += 1
				elif customer_value > 100:
					customer_segments["medium_value"] += 1
				else:
					customer_segments["low_value"] += 1
		
		# Subscription distribution
		for subscription in self.billing_service.subscriptions.values():
			if subscription.tenant_id == tenant_id:
				plan = self.billing_service.plans.get(subscription.plan_id)
				if plan:
					customers_by_plan[plan.name] += 1
				
				# Churned customers (cancelled in period)
				if (subscription.cancelled_at and
					subscription.cancelled_at >= period_start and
					subscription.cancelled_at <= period_end):
					churned_customers += 1
		
		# Calculate churn rate
		churn_rate = (churned_customers / max(total_customers, 1)) * 100
		
		# Customer acquisition cost (simplified)
		cac = await self._estimate_customer_acquisition_cost(tenant_id, period_start, period_end)
		
		result = {
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"total_customers": total_customers,
			"active_customers": active_customers,
			"new_customers": new_customers,
			"churned_customers": churned_customers,
			"churn_rate_percentage": churn_rate,
			"customer_acquisition_cost": str(cac),
			"customer_segments": dict(customer_segments),
			"customers_by_plan": dict(customers_by_plan),
			"customer_growth_rate": ((new_customers - churned_customers) / max(total_customers - new_customers, 1)) * 100
		}
		
		self._cache_result(cache_key, result)
		return result
	
	async def _calculate_customer_lifetime_value(self, customer_id: str) -> Decimal:
		"""Calculate Customer Lifetime Value (CLV)"""
		total_paid = Decimal('0')
		
		for invoice in self.billing_service.invoices.values():
			if (invoice.customer_id == customer_id and
				invoice.status == InvoiceStatus.PAID):
				total_paid += invoice.total
		
		return total_paid
	
	async def _estimate_customer_acquisition_cost(self, tenant_id: str, period_start: datetime, period_end: datetime) -> Decimal:
		"""Comprehensive Customer Acquisition Cost calculation"""
		try:
			# Get new customers for the period
			new_customers = 0
			for customer in self.billing_service.customers.values():
				if (customer.tenant_id == tenant_id and
					customer.created_at >= period_start and
					customer.created_at <= period_end):
					new_customers += 1
			
			if new_customers == 0:
				return Decimal('0')
			
			# Calculate comprehensive acquisition costs
			period_revenue = await self._calculate_period_revenue(tenant_id, period_start, period_end)
			
			# Marketing costs breakdown
			marketing_spend = period_revenue * Decimal('0.15')  # 15% of revenue to marketing
			sales_team_costs = period_revenue * Decimal('0.12')  # 12% for sales team
			marketing_tools_costs = period_revenue * Decimal('0.03')  # 3% for tools (CRM, analytics, etc.)
			content_creation_costs = period_revenue * Decimal('0.02')  # 2% for content creation
			advertising_costs = marketing_spend * Decimal('0.6')  # 60% of marketing for ads
			events_costs = marketing_spend * Decimal('0.15')  # 15% for events/conferences
			
			# Operational overhead for acquisition
			overhead_costs = (marketing_spend + sales_team_costs) * Decimal('0.1')  # 10% overhead
			
			# Technology costs for acquisition (attribution tools, landing pages, etc.)
			tech_costs = period_revenue * Decimal('0.01')  # 1% for acquisition tech stack
			
			# Total acquisition investment
			total_acquisition_costs = (
				marketing_spend + sales_team_costs + marketing_tools_costs + 
				content_creation_costs + overhead_costs + tech_costs
			)
			
			# Calculate blended CAC
			blended_cac = total_acquisition_costs / new_customers
			
			# Calculate CAC by channel (simulated)
			channel_breakdown = {
				'organic': blended_cac * Decimal('0.3'),  # 30% lower cost
				'paid_search': blended_cac * Decimal('1.2'),  # 20% higher cost
				'social_media': blended_cac * Decimal('0.9'),  # 10% lower cost
				'referral': blended_cac * Decimal('0.4'),  # 60% lower cost
				'content_marketing': blended_cac * Decimal('0.7'),  # 30% lower cost
				'events': blended_cac * Decimal('1.8'),  # 80% higher cost
				'partnerships': blended_cac * Decimal('0.6')  # 40% lower cost
			}
			
			# Store detailed CAC breakdown for analytics
			self._store_cac_breakdown(tenant_id, period_start, period_end, {
				'blended_cac': float(blended_cac),
				'total_acquisition_costs': float(total_acquisition_costs),
				'new_customers': new_customers,
				'channel_breakdown': {k: float(v) for k, v in channel_breakdown.items()},
				'cost_components': {
					'marketing_spend': float(marketing_spend),
					'sales_team_costs': float(sales_team_costs),
					'tools_and_tech': float(marketing_tools_costs + tech_costs),
					'content_creation': float(content_creation_costs),
					'overhead': float(overhead_costs)
				}
			})
			
			return blended_cac
			
		except Exception as e:
			self.logger.error(f"Failed to calculate CAC: {e}")
			# Fallback to simple calculation
			period_revenue = await self._calculate_period_revenue(tenant_id, period_start, period_end)
			marketing_spend = period_revenue * Decimal('0.20')  # 20% of revenue
			return marketing_spend / max(new_customers, 1)

	def _store_cac_breakdown(self, tenant_id: str, period_start: datetime, period_end: datetime, cac_data: Dict[str, Any]) -> None:
		"""Store detailed CAC breakdown for advanced analytics"""
		try:
			# In a real implementation, this would store to a time-series database
			cache_key = f"cac_breakdown_{tenant_id}_{period_start.date()}_{period_end.date()}"
			self.analytics_cache[cache_key] = {
				'timestamp': datetime.utcnow().isoformat(),
				'data': cac_data
			}
			self.logger.info(f"Stored CAC breakdown for {tenant_id}: ${cac_data['blended_cac']:.2f}")
		except Exception as e:
			self.logger.error(f"Failed to store CAC breakdown: {e}")
	
	# Subscription Analytics
	
	async def get_subscription_analytics(self, tenant_id: str) -> Dict[str, Any]:
		"""Get subscription analytics and metrics"""
		cache_key = self._get_cache_key("subscription_analytics", tenant_id=tenant_id)
		cached_result = self._get_cached_result(cache_key)
		if cached_result:
			return cached_result
		
		# Subscription status distribution
		status_distribution = defaultdict(int)
		plan_distribution = defaultdict(int)
		
		# Subscription lifecycle metrics
		total_subscriptions = 0
		trial_conversions = 0
		total_trials = 0
		
		for subscription in self.billing_service.subscriptions.values():
			if subscription.tenant_id == tenant_id:
				total_subscriptions += 1
				status_distribution[subscription.status.value] += 1
				
				# Plan distribution
				plan = self.billing_service.plans.get(subscription.plan_id)
				if plan:
					plan_distribution[plan.name] += 1
				
				# Trial metrics
				if subscription.trial_start:
					total_trials += 1
					if subscription.status == SubscriptionStatus.ACTIVE:
						trial_conversions += 1
		
		# Trial conversion rate
		trial_conversion_rate = (trial_conversions / max(total_trials, 1)) * 100
		
		# Average subscription duration
		avg_duration = await self._calculate_average_subscription_duration(tenant_id)
		
		result = {
			"total_subscriptions": total_subscriptions,
			"status_distribution": dict(status_distribution),
			"plan_distribution": dict(plan_distribution),
			"trial_conversion_rate": trial_conversion_rate,
			"average_subscription_duration_days": avg_duration,
			"active_subscription_percentage": (status_distribution["active"] / max(total_subscriptions, 1)) * 100
		}
		
		self._cache_result(cache_key, result)
		return result
	
	async def _calculate_average_subscription_duration(self, tenant_id: str) -> float:
		"""Calculate average subscription duration in days"""
		total_duration = 0
		completed_subscriptions = 0
		
		for subscription in self.billing_service.subscriptions.values():
			if subscription.tenant_id == tenant_id:
				if subscription.cancelled_at:
					duration = (subscription.cancelled_at - subscription.created_at).days
					total_duration += duration
					completed_subscriptions += 1
		
		if completed_subscriptions > 0:
			return total_duration / completed_subscriptions
		return 0.0
	
	# Usage Analytics
	
	async def get_usage_analytics(self, tenant_id: str, period_start: datetime = None, period_end: datetime = None) -> Dict[str, Any]:
		"""Get usage analytics and patterns"""
		cache_key = self._get_cache_key("usage_analytics", tenant_id=tenant_id, start=period_start, end=period_end)
		cached_result = self._get_cached_result(cache_key)
		if cached_result:
			return cached_result
		
		# Default to last 30 days
		if not period_end:
			period_end = datetime.utcnow()
		if not period_start:
			period_start = period_end - timedelta(days=30)
		
		# Usage metrics by metric type
		usage_by_metric = defaultdict(lambda: {"total": Decimal('0'), "count": 0, "avg": Decimal('0')})
		usage_by_customer = defaultdict(lambda: Decimal('0'))
		usage_trends = defaultdict(lambda: defaultdict(lambda: Decimal('0')))
		
		for usage in self.billing_service.usage_records:
			if (usage.tenant_id == tenant_id and
				usage.timestamp >= period_start and
				usage.timestamp <= period_end):
				
				# By metric
				metric_stats = usage_by_metric[usage.metric_name]
				metric_stats["total"] += usage.quantity
				metric_stats["count"] += 1
				metric_stats["avg"] = metric_stats["total"] / metric_stats["count"]
				
				# By customer
				usage_by_customer[usage.customer_id] += usage.quantity
				
				# Trends by day
				day_key = usage.timestamp.date()
				usage_trends[usage.metric_name][day_key] += usage.quantity
		
		# Top usage metrics
		top_metrics = sorted(usage_by_metric.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
		
		# Top customers by usage
		top_customers = sorted(usage_by_customer.items(), key=lambda x: x[1], reverse=True)[:10]
		
		result = {
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"usage_by_metric": {k: {
				"total": str(v["total"]),
				"count": v["count"],
				"average": str(v["avg"])
			} for k, v in usage_by_metric.items()},
			"top_metrics": [(metric, str(stats["total"])) for metric, stats in top_metrics],
			"top_customers_by_usage": [(customer_id, str(usage)) for customer_id, usage in top_customers],
			"total_usage_records": len([u for u in self.billing_service.usage_records 
				if u.tenant_id == tenant_id and period_start <= u.timestamp <= period_end])
		}
		
		self._cache_result(cache_key, result)
		return result
	
	# AI-Powered Analytics
	
	async def get_predictive_analytics(self, tenant_id: str) -> Dict[str, Any]:
		"""Get AI-powered predictive analytics"""
		if not self._ai_orchestration_available:
			return {"error": "AI orchestration not available"}
		
		cache_key = self._get_cache_key("predictive_analytics", tenant_id=tenant_id)
		cached_result = self._get_cached_result(cache_key)
		if cached_result:
			return cached_result
		
		try:
			# Prepare data for AI analysis
			analytics_data = {
				"revenue": await self.get_revenue_analytics(tenant_id),
				"customers": await self.get_customer_analytics(tenant_id),
				"subscriptions": await self.get_subscription_analytics(tenant_id),
				"usage": await self.get_usage_analytics(tenant_id)
			}
			
			# Submit AI task for predictive analysis
			task_definition = {
				"type": "predictive_analytics",
				"input": {
					"data": analytics_data,
					"analysis_type": "billing_prediction",
					"predictions": [
						"revenue_forecast_30_days",
						"churn_risk_customers",
						"upsell_opportunities",
						"usage_anomalies"
					]
				},
				"model": "llama3.2:3b"
			}
			
			task_id = await self.ai_orchestration.submit_task(task_definition)
			
			# Wait for analysis (simplified - in production, this would be asynchronous)
			await asyncio.sleep(2)
			task_status = await self.ai_orchestration.get_task_status(task_id)
			
			if task_status["status"] == "completed":
				result = task_status.get("result", {})
				self._cache_result(cache_key, result)
				return result
			else:
				return {"status": "processing", "task_id": task_id}
		
		except Exception as e:
			self.logger.error(f"Predictive analytics failed: {e}")
			return {"error": "Predictive analytics failed"}
	
	async def get_churn_prediction(self, tenant_id: str) -> Dict[str, Any]:
		"""Get churn prediction for customers"""
		cache_key = self._get_cache_key("churn_prediction", tenant_id=tenant_id)
		cached_result = self._get_cached_result(cache_key)
		if cached_result:
			return cached_result
		
		# Simplified churn prediction based on usage patterns and payment history
		at_risk_customers = []
		
		for customer in self.billing_service.customers.values():
			if customer.tenant_id == tenant_id and customer.active:
				churn_score = await self._calculate_churn_score(customer.id)
				
				if churn_score > 0.7:  # High risk threshold
					at_risk_customers.append({
						"customer_id": customer.id,
						"customer_name": customer.name,
						"churn_score": churn_score,
						"risk_level": "high" if churn_score > 0.8 else "medium",
						"recommended_actions": await self._get_retention_recommendations(customer.id)
					})
		
		# Sort by churn score (highest risk first)
		at_risk_customers.sort(key=lambda x: x["churn_score"], reverse=True)
		
		result = {
			"total_customers_analyzed": len([c for c in self.billing_service.customers.values() if c.tenant_id == tenant_id]),
			"at_risk_customers": at_risk_customers[:20],  # Top 20 at-risk customers
			"high_risk_count": len([c for c in at_risk_customers if c["risk_level"] == "high"]),
			"medium_risk_count": len([c for c in at_risk_customers if c["risk_level"] == "medium"]),
			"analysis_date": datetime.utcnow().isoformat()
		}
		
		self._cache_result(cache_key, result)
		return result
	
	async def _calculate_churn_score(self, customer_id: str) -> float:
		"""Calculate churn probability score for a customer"""
		score = 0.0
		
		# Payment history (30% weight)
		failed_payments = sum(1 for p in self.billing_service.payments.values()
			if p.customer_id == customer_id and p.status == PaymentStatus.FAILED)
		total_payments = sum(1 for p in self.billing_service.payments.values()
			if p.customer_id == customer_id)
		
		if total_payments > 0:
			payment_failure_rate = failed_payments / total_payments
			score += payment_failure_rate * 0.3
		
		# Usage decline (25% weight)
		recent_usage = sum(u.quantity for u in self.billing_service.usage_records
			if u.customer_id == customer_id and u.timestamp > datetime.utcnow() - timedelta(days=30))
		
		older_usage = sum(u.quantity for u in self.billing_service.usage_records
			if u.customer_id == customer_id and 
			datetime.utcnow() - timedelta(days=60) < u.timestamp <= datetime.utcnow() - timedelta(days=30))
		
		if older_usage > 0:
			usage_decline = max(0, 1 - (float(recent_usage) / float(older_usage)))
			score += usage_decline * 0.25
		
		# Support tickets / complaints (20% weight)
		support_score = 0.0
		try:
			# Try to get support ticket data from customer metadata
			customer = self.billing_service.customers.get(customer_id)
			if customer and hasattr(customer, 'metadata'):
				metadata = customer.metadata or {}
				
				# Check for support metrics in metadata
				recent_tickets = metadata.get('recent_support_tickets', 0)
				complaint_tickets = metadata.get('complaint_tickets', 0)
				satisfaction_score = metadata.get('satisfaction_score', 5.0)  # 1-5 scale
				
				# More tickets = higher churn risk
				if recent_tickets > 5:
					support_score += 0.15
				elif recent_tickets > 2:
					support_score += 0.1
				elif recent_tickets > 0:
					support_score += 0.05
				
				# Complaints = higher churn risk
				if complaint_tickets > 0:
					support_score += min(complaint_tickets * 0.05, 0.1)
				
				# Low satisfaction = higher churn risk
				if satisfaction_score < 3.0:
					support_score += 0.1
				elif satisfaction_score < 4.0:
					support_score += 0.05
				
		except Exception:
			# Fallback if support data not available
			support_score = 0.05
		
		score += support_score
		
		# Subscription age (15% weight) - newer customers more likely to churn
		customer_subscriptions = [s for s in self.billing_service.subscriptions.values() if s.customer_id == customer_id]
		if customer_subscriptions:
			oldest_subscription = min(customer_subscriptions, key=lambda s: s.created_at)
			subscription_age_days = (datetime.utcnow() - oldest_subscription.created_at).days
			
			# Higher churn risk for very new (< 30 days) customers
			if subscription_age_days < 30:
				score += 0.15
			elif subscription_age_days < 90:
				score += 0.1
		
		# Engagement (10% weight) - based on login frequency, feature usage, etc.
		engagement_score = 0.0
		try:
			# Get engagement data from customer metadata and user activity
			customer = self.billing_service.customers.get(customer_id)
			if customer and hasattr(customer, 'metadata'):
				metadata = customer.metadata or {}
				
				# Check last login
				last_login = metadata.get('last_login_date')
				if last_login:
					try:
						if isinstance(last_login, str):
							last_login_date = datetime.fromisoformat(last_login)
						else:
							last_login_date = last_login
						
						days_since_login = (datetime.utcnow() - last_login_date).days
						
						# More days since login = higher churn risk
						if days_since_login > 30:
							engagement_score += 0.08
						elif days_since_login > 14:
							engagement_score += 0.05
						elif days_since_login > 7:
							engagement_score += 0.02
					except Exception:
						engagement_score += 0.05  # Unknown login = moderate risk
				else:
					engagement_score += 0.08  # No login data = high risk
				
				# Check feature usage metrics
				monthly_sessions = metadata.get('monthly_sessions', 0)
				features_used = metadata.get('features_used_count', 0)
				
				# Low usage = higher churn risk
				if monthly_sessions < 5:
					engagement_score += 0.02
				if features_used < 3:
					engagement_score += 0.02
				
		except Exception:
			# Fallback if engagement data not available
			engagement_score = 0.05
		
		score += engagement_score
		
		return min(score, 1.0)  # Cap at 1.0
	
	async def _get_retention_recommendations(self, customer_id: str) -> List[str]:
		"""Get retention recommendations for at-risk customer"""
		recommendations = []
		
		# Analyze customer data to provide targeted recommendations
		customer = self.billing_service.customers.get(customer_id)
		if not customer:
			return recommendations
		
		# Check for failed payments
		recent_failed_payments = [p for p in self.billing_service.payments.values()
			if p.customer_id == customer_id and p.status == PaymentStatus.FAILED and
			p.created_at > datetime.utcnow() - timedelta(days=30)]
		
		if recent_failed_payments:
			recommendations.append("Contact customer about payment method update")
			recommendations.append("Offer payment plan or billing assistance")
		
		# Check usage patterns
		recent_usage = [u for u in self.billing_service.usage_records
			if u.customer_id == customer_id and u.timestamp > datetime.utcnow() - timedelta(days=30)]
		
		if not recent_usage:
			recommendations.append("Provide onboarding assistance and training")
			recommendations.append("Share relevant use cases and success stories")
		
		# General retention tactics
		recommendations.extend([
			"Schedule check-in call with customer success team",
			"Offer discount or promotional pricing",
			"Provide additional support and resources"
		])
		
		return recommendations[:5]  # Return top 5 recommendations
	
	# Financial Reporting
	
	async def generate_financial_report(self, tenant_id: str, report_type: str, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
		"""Generate comprehensive financial reports"""
		cache_key = self._get_cache_key("financial_report", tenant_id=tenant_id, type=report_type, start=period_start, end=period_end)
		cached_result = self._get_cached_result(cache_key)
		if cached_result:
			return cached_result
		
		if report_type == "revenue_recognition":
			result = await self._generate_revenue_recognition_report(tenant_id, period_start, period_end)
		elif report_type == "subscription_summary":
			result = await self._generate_subscription_summary_report(tenant_id, period_start, period_end)
		elif report_type == "customer_aging":
			result = await self._generate_customer_aging_report(tenant_id, period_end)
		else:
			result = {"error": f"Unknown report type: {report_type}"}
		
		if "error" not in result:
			self._cache_result(cache_key, result)
		
		return result
	
	async def _generate_revenue_recognition_report(self, tenant_id: str, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
		"""Generate revenue recognition report"""
		# Implementation would include proper revenue recognition logic
		# This is a simplified version
		
		total_invoiced = Decimal('0')
		total_collected = Decimal('0')
		deferred_revenue = Decimal('0')
		
		for invoice in self.billing_service.invoices.values():
			if (invoice.tenant_id == tenant_id and
				invoice.invoice_date >= period_start and
				invoice.invoice_date <= period_end):
				
				total_invoiced += invoice.total
				
				if invoice.status == InvoiceStatus.PAID:
					total_collected += invoice.amount_paid
		
		return {
			"report_type": "revenue_recognition",
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"total_invoiced": str(total_invoiced),
			"total_collected": str(total_collected),
			"deferred_revenue": str(deferred_revenue),
			"collection_rate": float((total_collected / max(total_invoiced, Decimal('0.01'))) * 100),
			"generated_at": datetime.utcnow().isoformat()
		}
	
	async def _generate_subscription_summary_report(self, tenant_id: str, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
		"""Generate subscription summary report"""
		new_subscriptions = 0
		cancelled_subscriptions = 0
		active_subscriptions = 0
		
		for subscription in self.billing_service.subscriptions.values():
			if subscription.tenant_id == tenant_id:
				if subscription.created_at >= period_start and subscription.created_at <= period_end:
					new_subscriptions += 1
				
				if (subscription.cancelled_at and
					subscription.cancelled_at >= period_start and
					subscription.cancelled_at <= period_end):
					cancelled_subscriptions += 1
				
				if subscription.status == SubscriptionStatus.ACTIVE:
					active_subscriptions += 1
		
		return {
			"report_type": "subscription_summary",
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"new_subscriptions": new_subscriptions,
			"cancelled_subscriptions": cancelled_subscriptions,
			"net_subscription_growth": new_subscriptions - cancelled_subscriptions,
			"active_subscriptions_end_of_period": active_subscriptions,
			"generated_at": datetime.utcnow().isoformat()
		}
	
	async def _generate_customer_aging_report(self, tenant_id: str, as_of_date: datetime) -> Dict[str, Any]:
		"""Generate customer aging report for outstanding invoices"""
		aging_buckets = {
			"current": Decimal('0'),
			"1_30_days": Decimal('0'),
			"31_60_days": Decimal('0'),
			"61_90_days": Decimal('0'),
			"over_90_days": Decimal('0')
		}
		
		customer_details = []
		
		for invoice in self.billing_service.invoices.values():
			if (invoice.tenant_id == tenant_id and
				invoice.amount_due > 0 and
				invoice.status in [InvoiceStatus.PENDING, InvoiceStatus.OVERDUE]):
				
				days_overdue = (as_of_date - invoice.due_date).days
				
				customer = self.billing_service.customers.get(invoice.customer_id)
				customer_name = customer.name if customer else "Unknown"
				
				customer_details.append({
					"customer_id": invoice.customer_id,
					"customer_name": customer_name,
					"invoice_id": invoice.id,
					"invoice_number": invoice.invoice_number,
					"amount_due": str(invoice.amount_due),
					"due_date": invoice.due_date.isoformat(),
					"days_overdue": days_overdue
				})
				
				# Categorize by aging bucket
				if days_overdue <= 0:
					aging_buckets["current"] += invoice.amount_due
				elif days_overdue <= 30:
					aging_buckets["1_30_days"] += invoice.amount_due
				elif days_overdue <= 60:
					aging_buckets["31_60_days"] += invoice.amount_due
				elif days_overdue <= 90:
					aging_buckets["61_90_days"] += invoice.amount_due
				else:
					aging_buckets["over_90_days"] += invoice.amount_due
		
		total_outstanding = sum(aging_buckets.values())
		
		return {
			"report_type": "customer_aging",
			"as_of_date": as_of_date.isoformat(),
			"aging_summary": {k: str(v) for k, v in aging_buckets.items()},
			"total_outstanding": str(total_outstanding),
			"customer_details": sorted(customer_details, key=lambda x: x["days_overdue"], reverse=True),
			"generated_at": datetime.utcnow().isoformat()
		}


# Global analytics engine instance
_analytics_engine_instance: Optional[BillingAnalyticsEngine] = None

def get_billing_analytics_engine() -> BillingAnalyticsEngine:
	"""Get global billing analytics engine instance"""
	global _analytics_engine_instance
	if _analytics_engine_instance is None:
		_analytics_engine_instance = BillingAnalyticsEngine()
	return _analytics_engine_instance


__all__ = [
	"BillingAnalyticsEngine",
	"get_billing_analytics_engine"
]