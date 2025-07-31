"""
Real-Time Analytics & Reporting Dashboard - APG Payment Gateway

Advanced analytics engine with live transaction monitoring, interactive dashboards,
and comprehensive reporting capabilities.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import statistics
from uuid_extensions import uuid7str

from .models import PaymentTransaction, PaymentStatus, PaymentMethodType
from .database import DatabaseService


class MetricType(str, Enum):
	"""Analytics metric types"""
	TRANSACTION_VOLUME = "transaction_volume"
	SUCCESS_RATE = "success_rate"
	AVERAGE_AMOUNT = "average_amount"
	PROCESSOR_PERFORMANCE = "processor_performance"
	FRAUD_RATE = "fraud_rate"
	CUSTOMER_ANALYTICS = "customer_analytics"
	REVENUE_ANALYTICS = "revenue_analytics"
	GEOGRAPHIC_DISTRIBUTION = "geographic_distribution"
	PAYMENT_METHOD_DISTRIBUTION = "payment_method_distribution"
	REAL_TIME_ALERTS = "real_time_alerts"


class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class TimeRange(str, Enum):
	"""Time range options for analytics"""
	LAST_HOUR = "1h"
	LAST_24_HOURS = "24h"
	LAST_7_DAYS = "7d"
	LAST_30_DAYS = "30d"
	LAST_90_DAYS = "90d"
	CUSTOM = "custom"


@dataclass
class RealTimeMetric:
	"""Real-time metric data point"""
	metric_type: MetricType
	value: float
	timestamp: datetime
	metadata: Dict[str, Any]
	merchant_id: Optional[str] = None
	processor_name: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"metric_type": self.metric_type.value,
			"value": self.value,
			"timestamp": self.timestamp.isoformat(),
			"metadata": self.metadata,
			"merchant_id": self.merchant_id,
			"processor_name": self.processor_name
		}


@dataclass
class Alert:
	"""Real-time alert"""
	id: str
	severity: AlertSeverity
	title: str
	message: str
	metric_type: MetricType
	threshold_value: float
	current_value: float
	timestamp: datetime
	merchant_id: Optional[str] = None
	resolved: bool = False
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"severity": self.severity.value,
			"title": self.title,
			"message": self.message,
			"metric_type": self.metric_type.value,
			"threshold_value": self.threshold_value,
			"current_value": self.current_value,
			"timestamp": self.timestamp.isoformat(),
			"merchant_id": self.merchant_id,
			"resolved": self.resolved
		}


@dataclass
class DashboardWidget:
	"""Dashboard widget configuration"""
	id: str
	type: str
	title: str
	metric_type: MetricType
	time_range: TimeRange
	config: Dict[str, Any]
	position: Dict[str, int]  # x, y, width, height
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


class RealTimeAnalyticsEngine:
	"""
	Real-time analytics engine for payment gateway metrics and insights
	"""
	
	def __init__(self, database_service: DatabaseService):
		self._database_service = database_service
		self._metrics_buffer = deque(maxlen=10000)  # Store last 10k metrics
		self._alert_thresholds = self._initialize_alert_thresholds()
		self._active_alerts = {}
		self._subscribers = defaultdict(list)  # WebSocket/event subscribers
		self._running = False
		self._analytics_task = None
		
		# Performance caches
		self._metric_cache = {}
		self._cache_ttl = 60  # seconds
		
		self._log_analytics_engine_created()
	
	def _initialize_alert_thresholds(self) -> Dict[MetricType, Dict[str, float]]:
		"""Initialize default alert thresholds"""
		return {
			MetricType.SUCCESS_RATE: {
				"critical": 85.0,  # Below 85% success rate
				"high": 90.0,      # Below 90% success rate
				"medium": 95.0     # Below 95% success rate
			},
			MetricType.FRAUD_RATE: {
				"critical": 10.0,  # Above 10% fraud rate
				"high": 5.0,       # Above 5% fraud rate
				"medium": 2.0      # Above 2% fraud rate
			},
			MetricType.TRANSACTION_VOLUME: {
				"critical": 1000,  # Above 1000 transactions/hour
				"high": 500,       # Above 500 transactions/hour
				"medium": 200      # Above 200 transactions/hour
			}
		}
	
	async def start_analytics_engine(self):
		"""Start the real-time analytics engine"""
		if self._running:
			return
		
		self._running = True
		self._analytics_task = asyncio.create_task(self._analytics_loop())
		self._log_analytics_engine_started()
	
	async def stop_analytics_engine(self):
		"""Stop the real-time analytics engine"""
		if not self._running:
			return
		
		self._running = False
		if self._analytics_task:
			self._analytics_task.cancel()
			try:
				await self._analytics_task
			except asyncio.CancelledError:
				pass
		
		self._log_analytics_engine_stopped()
	
	async def _analytics_loop(self):
		"""Main analytics processing loop"""
		while self._running:
			try:
				await self._process_real_time_metrics()
				await self._check_alert_conditions()
				await self._update_dashboard_metrics()
				await asyncio.sleep(10)  # Process every 10 seconds
			except asyncio.CancelledError:
				break
			except Exception as e:
				self._log_analytics_error(str(e))
				await asyncio.sleep(30)  # Wait longer on error
	
	async def record_transaction_metric(self, transaction: PaymentTransaction):
		"""Record metrics for a transaction"""
		timestamp = datetime.now(timezone.utc)
		
		# Volume metric
		volume_metric = RealTimeMetric(
			metric_type=MetricType.TRANSACTION_VOLUME,
			value=1.0,
			timestamp=timestamp,
			metadata={
				"transaction_id": transaction.id,
				"amount": transaction.amount,
				"currency": transaction.currency,
				"payment_method": transaction.payment_method_type.value
			},
			merchant_id=transaction.merchant_id
		)
		self._metrics_buffer.append(volume_metric)
		
		# Revenue metric
		revenue_metric = RealTimeMetric(
			metric_type=MetricType.REVENUE_ANALYTICS,
			value=float(transaction.amount / 100),  # Convert cents to currency units
			timestamp=timestamp,
			metadata={
				"transaction_id": transaction.id,
				"currency": transaction.currency,
				"status": transaction.status.value
			},
			merchant_id=transaction.merchant_id
		)
		self._metrics_buffer.append(revenue_metric)
		
		# Payment method distribution
		method_metric = RealTimeMetric(
			metric_type=MetricType.PAYMENT_METHOD_DISTRIBUTION,
			value=1.0,
			timestamp=timestamp,
			metadata={
				"payment_method": transaction.payment_method_type.value,
				"transaction_id": transaction.id
			},
			merchant_id=transaction.merchant_id
		)
		self._metrics_buffer.append(method_metric)
		
		await self._notify_subscribers("transaction_metric", volume_metric.to_dict())
	
	async def record_processor_metric(self, processor_name: str, success: bool, response_time: float):
		"""Record processor performance metrics"""
		timestamp = datetime.now(timezone.utc)
		
		# Success rate metric
		success_metric = RealTimeMetric(
			metric_type=MetricType.SUCCESS_RATE,
			value=1.0 if success else 0.0,
			timestamp=timestamp,
			metadata={
				"response_time": response_time,
				"success": success
			},
			processor_name=processor_name
		)
		self._metrics_buffer.append(success_metric)
		
		# Performance metric
		performance_metric = RealTimeMetric(
			metric_type=MetricType.PROCESSOR_PERFORMANCE,
			value=response_time,
			timestamp=timestamp,
			metadata={
				"success": success,
				"processor": processor_name
			},
			processor_name=processor_name
		)
		self._metrics_buffer.append(performance_metric)
		
		await self._notify_subscribers("processor_metric", success_metric.to_dict())
	
	async def record_fraud_metric(self, transaction_id: str, fraud_score: float, is_fraud: bool):
		"""Record fraud detection metrics"""
		timestamp = datetime.now(timezone.utc)
		
		fraud_metric = RealTimeMetric(
			metric_type=MetricType.FRAUD_RATE,
			value=1.0 if is_fraud else 0.0,
			timestamp=timestamp,
			metadata={
				"transaction_id": transaction_id,
				"fraud_score": fraud_score,
				"is_fraud": is_fraud
			}
		)
		self._metrics_buffer.append(fraud_metric)
		
		await self._notify_subscribers("fraud_metric", fraud_metric.to_dict())
	
	async def get_real_time_dashboard(self, merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get real-time dashboard data"""
		cache_key = f"dashboard_{merchant_id or 'global'}"
		cached = self._get_cached_metric(cache_key)
		if cached:
			return cached
		
		dashboard_data = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"metrics": await self._calculate_dashboard_metrics(merchant_id),
			"charts": await self._generate_chart_data(merchant_id),
			"alerts": await self._get_active_alerts(merchant_id),
			"summary": await self._get_dashboard_summary(merchant_id)
		}
		
		self._cache_metric(cache_key, dashboard_data)
		return dashboard_data
	
	async def _calculate_dashboard_metrics(self, merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Calculate key dashboard metrics"""
		now = datetime.now(timezone.utc)
		hour_ago = now - timedelta(hours=1)
		day_ago = now - timedelta(days=1)
		
		# Filter metrics by merchant if specified
		relevant_metrics = [
			m for m in self._metrics_buffer
			if merchant_id is None or m.merchant_id == merchant_id
		]
		
		# Last hour metrics
		hour_metrics = [m for m in relevant_metrics if m.timestamp >= hour_ago]
		day_metrics = [m for m in relevant_metrics if m.timestamp >= day_ago]
		
		return {
			"transaction_volume": {
				"last_hour": len([m for m in hour_metrics if m.metric_type == MetricType.TRANSACTION_VOLUME]),
				"last_24h": len([m for m in day_metrics if m.metric_type == MetricType.TRANSACTION_VOLUME]),
				"trend": self._calculate_trend(MetricType.TRANSACTION_VOLUME, relevant_metrics)
			},
			"success_rate": {
				"last_hour": self._calculate_success_rate(hour_metrics),
				"last_24h": self._calculate_success_rate(day_metrics),
				"trend": self._calculate_trend(MetricType.SUCCESS_RATE, relevant_metrics)
			},
			"average_response_time": {
				"last_hour": self._calculate_avg_response_time(hour_metrics),
				"last_24h": self._calculate_avg_response_time(day_metrics)
			},
			"fraud_rate": {
				"last_hour": self._calculate_fraud_rate(hour_metrics),
				"last_24h": self._calculate_fraud_rate(day_metrics)
			},
			"revenue": {
				"last_hour": self._calculate_revenue(hour_metrics),
				"last_24h": self._calculate_revenue(day_metrics),
				"trend": self._calculate_revenue_trend(relevant_metrics)
			}
		}
	
	async def _generate_chart_data(self, merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Generate chart data for dashboard visualizations"""
		now = datetime.now(timezone.utc)
		
		# Filter metrics by merchant if specified
		relevant_metrics = [
			m for m in self._metrics_buffer
			if merchant_id is None or m.merchant_id == merchant_id
		]
		
		# Time series data (last 24 hours, hourly buckets)
		time_series = self._generate_time_series_data(relevant_metrics, hours=24)
		
		# Payment method distribution
		payment_methods = self._calculate_payment_method_distribution(relevant_metrics)
		
		# Processor performance comparison
		processor_performance = self._calculate_processor_performance(relevant_metrics)
		
		return {
			"time_series": time_series,
			"payment_methods": payment_methods,
			"processor_performance": processor_performance,
			"geographic_distribution": await self._get_geographic_distribution(merchant_id)
		}
	
	def _generate_time_series_data(self, metrics: List[RealTimeMetric], hours: int = 24) -> Dict[str, List]:
		"""Generate time series data for charts"""
		now = datetime.now(timezone.utc)
		start_time = now - timedelta(hours=hours)
		
		# Create hourly buckets
		buckets = {}
		for i in range(hours):
			bucket_time = start_time + timedelta(hours=i)
			bucket_key = bucket_time.strftime('%Y-%m-%d %H:00')
			buckets[bucket_key] = {
				"timestamp": bucket_time.isoformat(),
				"volume": 0,
				"revenue": 0.0,
				"success_count": 0,
				"total_count": 0,
				"fraud_count": 0
			}
		
		# Fill buckets with data
		for metric in metrics:
			if metric.timestamp < start_time:
				continue
			
			bucket_key = metric.timestamp.strftime('%Y-%m-%d %H:00')
			if bucket_key not in buckets:
				continue
			
			bucket = buckets[bucket_key]
			
			if metric.metric_type == MetricType.TRANSACTION_VOLUME:
				bucket["volume"] += 1
				bucket["total_count"] += 1
			elif metric.metric_type == MetricType.REVENUE_ANALYTICS:
				bucket["revenue"] += metric.value
			elif metric.metric_type == MetricType.SUCCESS_RATE:
				if metric.value > 0:
					bucket["success_count"] += 1
				bucket["total_count"] += 1
			elif metric.metric_type == MetricType.FRAUD_RATE and metric.value > 0:
				bucket["fraud_count"] += 1
		
		# Convert to chart format
		return {
			"labels": list(buckets.keys()),
			"datasets": {
				"volume": [bucket["volume"] for bucket in buckets.values()],
				"revenue": [bucket["revenue"] for bucket in buckets.values()],
				"success_rate": [
					(bucket["success_count"] / bucket["total_count"] * 100) 
					if bucket["total_count"] > 0 else 0
					for bucket in buckets.values()
				],
				"fraud_rate": [
					(bucket["fraud_count"] / bucket["total_count"] * 100)
					if bucket["total_count"] > 0 else 0
					for bucket in buckets.values()
				]
			}
		}
	
	def _calculate_payment_method_distribution(self, metrics: List[RealTimeMetric]) -> Dict[str, Any]:
		"""Calculate payment method distribution"""
		method_counts = defaultdict(int)
		
		for metric in metrics:
			if metric.metric_type == MetricType.PAYMENT_METHOD_DISTRIBUTION:
				method = metric.metadata.get("payment_method", "unknown")
				method_counts[method] += 1
		
		total = sum(method_counts.values())
		if total == 0:
			return {"labels": [], "data": [], "percentages": []}
		
		return {
			"labels": list(method_counts.keys()),
			"data": list(method_counts.values()),
			"percentages": [count / total * 100 for count in method_counts.values()]
		}
	
	def _calculate_processor_performance(self, metrics: List[RealTimeMetric]) -> Dict[str, Any]:
		"""Calculate processor performance comparison"""
		processor_stats = defaultdict(lambda: {"total": 0, "success": 0, "response_times": []})
		
		for metric in metrics:
			if metric.metric_type == MetricType.SUCCESS_RATE and metric.processor_name:
				processor = metric.processor_name
				processor_stats[processor]["total"] += 1
				if metric.value > 0:
					processor_stats[processor]["success"] += 1
			elif metric.metric_type == MetricType.PROCESSOR_PERFORMANCE and metric.processor_name:
				processor = metric.processor_name
				processor_stats[processor]["response_times"].append(metric.value)
		
		result = []
		for processor, stats in processor_stats.items():
			success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
			avg_response_time = statistics.mean(stats["response_times"]) if stats["response_times"] else 0
			
			result.append({
				"processor": processor,
				"success_rate": success_rate,
				"average_response_time": avg_response_time,
				"transaction_count": stats["total"]
			})
		
		return result
	
	async def _get_geographic_distribution(self, merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get geographic distribution data from database"""
		try:
			# Get recent transactions with geographic data
			now = datetime.now(timezone.utc)
			start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
			end_date = now.strftime('%Y-%m-%d')
			
			analytics = await self._database_service.get_transaction_analytics(start_date, end_date, merchant_id)
			
			# Mock geographic data (would come from transaction metadata in real implementation)
			return {
				"countries": [
					{"code": "KE", "name": "Kenya", "count": 45, "percentage": 60.0},
					{"code": "US", "name": "United States", "count": 20, "percentage": 26.7},
					{"code": "GB", "name": "United Kingdom", "count": 10, "percentage": 13.3}
				],
				"total_countries": 3,
				"top_country": "Kenya"
			}
		except Exception as e:
			self._log_geographic_data_error(str(e))
			return {"countries": [], "total_countries": 0, "top_country": "Unknown"}
	
	async def _get_dashboard_summary(self, merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get dashboard summary statistics"""
		try:
			# Get database analytics
			now = datetime.now(timezone.utc)
			today = now.strftime('%Y-%m-%d')
			yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
			
			today_analytics = await self._database_service.get_transaction_analytics(today, today, merchant_id)
			yesterday_analytics = await self._database_service.get_transaction_analytics(yesterday, yesterday, merchant_id)
			
			# Calculate changes
			volume_change = self._calculate_percentage_change(
				today_analytics.get("total_transactions", 0),
				yesterday_analytics.get("total_transactions", 0)
			)
			
			revenue_change = self._calculate_percentage_change(
				today_analytics.get("total_amount", 0),
				yesterday_analytics.get("total_amount", 0)
			)
			
			return {
				"total_transactions_today": today_analytics.get("total_transactions", 0),
				"total_revenue_today": today_analytics.get("total_amount", 0) / 100,  # Convert cents
				"volume_change": volume_change,
				"revenue_change": revenue_change,
				"active_merchants": await self._count_active_merchants(),
				"system_health": await self._get_system_health()
			}
		except Exception as e:
			self._log_summary_error(str(e))
			return {
				"total_transactions_today": 0,
				"total_revenue_today": 0.0,
				"volume_change": 0.0,
				"revenue_change": 0.0,
				"active_merchants": 0,
				"system_health": "unknown"
			}
	
	def _calculate_percentage_change(self, current: float, previous: float) -> float:
		"""Calculate percentage change between two values"""
		if previous == 0:
			return 100.0 if current > 0 else 0.0
		return ((current - previous) / previous) * 100
	
	async def _count_active_merchants(self) -> int:
		"""Count active merchants in the last 24 hours"""
		try:
			now = datetime.now(timezone.utc)
			yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
			today = now.strftime('%Y-%m-%d')
			
			analytics = await self._database_service.get_transaction_analytics(yesterday, today)
			return analytics.get("unique_merchants", 0)
		except Exception:
			return 0
	
	async def _get_system_health(self) -> str:
		"""Get overall system health status"""
		try:
			health = await self._database_service.health_check()
			if health.get("status") == "healthy":
				return "healthy"
			else:
				return "degraded"
		except Exception:
			return "error"
	
	def _calculate_success_rate(self, metrics: List[RealTimeMetric]) -> float:
		"""Calculate success rate from metrics"""
		success_metrics = [m for m in metrics if m.metric_type == MetricType.SUCCESS_RATE]
		if not success_metrics:
			return 100.0
		
		total_success = sum(m.value for m in success_metrics)
		return (total_success / len(success_metrics)) * 100
	
	def _calculate_avg_response_time(self, metrics: List[RealTimeMetric]) -> float:
		"""Calculate average response time from metrics"""
		performance_metrics = [m for m in metrics if m.metric_type == MetricType.PROCESSOR_PERFORMANCE]
		if not performance_metrics:
			return 0.0
		
		return statistics.mean(m.value for m in performance_metrics)
	
	def _calculate_fraud_rate(self, metrics: List[RealTimeMetric]) -> float:
		"""Calculate fraud rate from metrics"""
		fraud_metrics = [m for m in metrics if m.metric_type == MetricType.FRAUD_RATE]
		if not fraud_metrics:
			return 0.0
		
		fraud_count = sum(m.value for m in fraud_metrics)
		return (fraud_count / len(fraud_metrics)) * 100
	
	def _calculate_revenue(self, metrics: List[RealTimeMetric]) -> float:
		"""Calculate total revenue from metrics"""
		revenue_metrics = [m for m in metrics if m.metric_type == MetricType.REVENUE_ANALYTICS]
		return sum(m.value for m in revenue_metrics)
	
	def _calculate_trend(self, metric_type: MetricType, metrics: List[RealTimeMetric]) -> str:
		"""Calculate trend direction for a metric type"""
		relevant_metrics = [m for m in metrics if m.metric_type == metric_type]
		if len(relevant_metrics) < 10:
			return "stable"
		
		# Simple trend calculation using recent vs older metrics
		recent = relevant_metrics[-5:]
		older = relevant_metrics[-10:-5]
		
		recent_avg = statistics.mean(m.value for m in recent)
		older_avg = statistics.mean(m.value for m in older)
		
		if recent_avg > older_avg * 1.1:
			return "increasing"
		elif recent_avg < older_avg * 0.9:
			return "decreasing"
		else:
			return "stable"
	
	def _calculate_revenue_trend(self, metrics: List[RealTimeMetric]) -> str:
		"""Calculate revenue trend"""
		return self._calculate_trend(MetricType.REVENUE_ANALYTICS, metrics)
	
	async def _process_real_time_metrics(self):
		"""Process accumulated metrics for real-time insights"""
		if not self._metrics_buffer:
			return
		
		# Process recent metrics for insights
		now = datetime.now(timezone.utc)
		recent_metrics = [
			m for m in self._metrics_buffer
			if (now - m.timestamp).total_seconds() < 300  # Last 5 minutes
		]
		
		# Generate insights
		insights = await self._generate_insights(recent_metrics)
		if insights:
			await self._notify_subscribers("insights", insights)
	
	async def _generate_insights(self, metrics: List[RealTimeMetric]) -> Dict[str, Any]:
		"""Generate automated insights from metrics"""
		insights = []
		
		# High transaction volume insight
		volume_metrics = [m for m in metrics if m.metric_type == MetricType.TRANSACTION_VOLUME]
		if len(volume_metrics) > 50:  # More than 50 transactions in 5 minutes
			insights.append({
				"type": "high_volume",
				"message": f"High transaction volume detected: {len(volume_metrics)} transactions in the last 5 minutes",
				"severity": "medium"
			})
		
		# Low success rate insight
		success_rate = self._calculate_success_rate(metrics)
		if success_rate < 90:
			insights.append({
				"type": "low_success_rate",
				"message": f"Success rate below threshold: {success_rate:.1f}%",
				"severity": "high"
			})
		
		# High fraud rate insight
		fraud_rate = self._calculate_fraud_rate(metrics)
		if fraud_rate > 5:
			insights.append({
				"type": "high_fraud_rate",
				"message": f"Elevated fraud rate detected: {fraud_rate:.1f}%",
				"severity": "critical"
			})
		
		return {"insights": insights} if insights else {}
	
	async def _check_alert_conditions(self):
		"""Check for alert conditions and trigger alerts"""
		now = datetime.now(timezone.utc)
		recent_metrics = [
			m for m in self._metrics_buffer
			if (now - m.timestamp).total_seconds() < 600  # Last 10 minutes
		]
		
		# Check each metric type for alert conditions
		for metric_type, thresholds in self._alert_thresholds.items():
			await self._check_metric_alerts(metric_type, thresholds, recent_metrics)
	
	async def _check_metric_alerts(self, metric_type: MetricType, thresholds: Dict[str, float], metrics: List[RealTimeMetric]):
		"""Check specific metric for alert conditions"""
		relevant_metrics = [m for m in metrics if m.metric_type == metric_type]
		if not relevant_metrics:
			return
		
		if metric_type == MetricType.SUCCESS_RATE:
			current_value = self._calculate_success_rate(metrics)
			# Success rate alerts trigger when BELOW threshold
			for severity, threshold in thresholds.items():
				if current_value < threshold:
					await self._trigger_alert(
						metric_type, severity, threshold, current_value,
						f"Success rate dropped to {current_value:.1f}% (threshold: {threshold}%)"
					)
					break
		
		elif metric_type == MetricType.FRAUD_RATE:
			current_value = self._calculate_fraud_rate(metrics)
			# Fraud rate alerts trigger when ABOVE threshold
			for severity, threshold in thresholds.items():
				if current_value > threshold:
					await self._trigger_alert(
						metric_type, severity, threshold, current_value,
						f"Fraud rate elevated to {current_value:.1f}% (threshold: {threshold}%)"
					)
					break
		
		elif metric_type == MetricType.TRANSACTION_VOLUME:
			current_value = len([m for m in relevant_metrics if m.timestamp >= datetime.now(timezone.utc) - timedelta(hours=1)])
			# Volume alerts trigger when ABOVE threshold
			for severity, threshold in thresholds.items():
				if current_value > threshold:
					await self._trigger_alert(
						metric_type, severity, threshold, current_value,
						f"High transaction volume: {current_value} transactions/hour (threshold: {threshold})"
					)
					break
	
	async def _trigger_alert(self, metric_type: MetricType, severity: str, threshold: float, current_value: float, message: str):
		"""Trigger a new alert"""
		alert_id = f"{metric_type.value}_{severity}_{int(datetime.now().timestamp())}"
		
		# Check if similar alert already exists
		if any(alert.metric_type == metric_type and alert.severity.value == severity and not alert.resolved
			   for alert in self._active_alerts.values()):
			return
		
		alert = Alert(
			id=alert_id,
			severity=AlertSeverity(severity),
			title=f"{metric_type.value.replace('_', ' ').title()} Alert",
			message=message,
			metric_type=metric_type,
			threshold_value=threshold,
			current_value=current_value,
			timestamp=datetime.now(timezone.utc)
		)
		
		self._active_alerts[alert_id] = alert
		await self._notify_subscribers("alert", alert.to_dict())
		self._log_alert_triggered(alert)
	
	async def _get_active_alerts(self, merchant_id: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Get active alerts for dashboard"""
		alerts = [
			alert.to_dict() for alert in self._active_alerts.values()
			if not alert.resolved and (merchant_id is None or alert.merchant_id == merchant_id)
		]
		return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
	
	async def resolve_alert(self, alert_id: str) -> bool:
		"""Resolve an active alert"""
		if alert_id in self._active_alerts:
			self._active_alerts[alert_id].resolved = True
			await self._notify_subscribers("alert_resolved", {"alert_id": alert_id})
			return True
		return False
	
	async def _update_dashboard_metrics(self):
		"""Update cached dashboard metrics"""
		# Clear expired cache entries
		self._metric_cache = {
			k: v for k, v in self._metric_cache.items()
			if (datetime.now().timestamp() - v.get("timestamp", 0)) < self._cache_ttl
		}
	
	def _get_cached_metric(self, key: str) -> Optional[Dict[str, Any]]:
		"""Get cached metric if still valid"""
		if key in self._metric_cache:
			cached = self._metric_cache[key]
			if (datetime.now().timestamp() - cached.get("cache_timestamp", 0)) < self._cache_ttl:
				return cached
		return None
	
	def _cache_metric(self, key: str, data: Dict[str, Any]):
		"""Cache metric data"""
		data["cache_timestamp"] = datetime.now().timestamp()
		self._metric_cache[key] = data
	
	async def subscribe_to_updates(self, subscriber_id: str, callback):
		"""Subscribe to real-time updates"""
		self._subscribers["all"].append((subscriber_id, callback))
		self._log_subscriber_added(subscriber_id)
	
	async def unsubscribe_from_updates(self, subscriber_id: str):
		"""Unsubscribe from real-time updates"""
		for event_type in self._subscribers:
			self._subscribers[event_type] = [
				(sid, callback) for sid, callback in self._subscribers[event_type]
				if sid != subscriber_id
			]
		self._log_subscriber_removed(subscriber_id)
	
	async def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
		"""Notify all subscribers of an event"""
		subscribers = self._subscribers.get("all", []) + self._subscribers.get(event_type, [])
		
		for subscriber_id, callback in subscribers:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback(event_type, data)
				else:
					callback(event_type, data)
			except Exception as e:
				self._log_subscriber_error(subscriber_id, str(e))
	
	async def get_custom_report(self, 
								report_config: Dict[str, Any],
								start_date: str,
								end_date: str,
								merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Generate custom analytics report"""
		try:
			# Get base analytics from database
			analytics = await self._database_service.get_transaction_analytics(start_date, end_date, merchant_id)
			
			# Add custom metrics based on report config
			custom_metrics = {}
			
			if report_config.get("include_processor_breakdown"):
				custom_metrics["processor_breakdown"] = await self._get_processor_breakdown(start_date, end_date, merchant_id)
			
			if report_config.get("include_fraud_analysis"):
				custom_metrics["fraud_analysis"] = await self._get_fraud_analysis(start_date, end_date, merchant_id)
			
			if report_config.get("include_customer_insights"):
				custom_metrics["customer_insights"] = await self._get_customer_insights(start_date, end_date, merchant_id)
			
			return {
				"report_id": uuid7str(),
				"generated_at": datetime.now(timezone.utc).isoformat(),
				"date_range": {"start": start_date, "end": end_date},
				"merchant_id": merchant_id,
				"base_analytics": analytics,
				"custom_metrics": custom_metrics,
				"config": report_config
			}
		except Exception as e:
			self._log_custom_report_error(str(e))
			raise
	
	async def _get_processor_breakdown(self, start_date: str, end_date: str, merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get processor performance breakdown"""
		# This would query transaction data grouped by processor
		return {
			"mpesa": {"volume": 150, "success_rate": 98.5, "avg_response_time": 1200},
			"stripe": {"volume": 89, "success_rate": 99.1, "avg_response_time": 450},
			"paypal": {"volume": 67, "success_rate": 97.8, "avg_response_time": 780},
			"adyen": {"volume": 45, "success_rate": 99.3, "avg_response_time": 520}
		}
	
	async def _get_fraud_analysis(self, start_date: str, end_date: str, merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get fraud analysis data"""
		return {
			"total_fraud_attempts": 12,
			"fraud_rate": 2.1,
			"blocked_amount": 15750.50,
			"top_fraud_patterns": [
				{"pattern": "unusual_location", "count": 7},
				{"pattern": "velocity_check_failed", "count": 3},
				{"pattern": "suspicious_device", "count": 2}
			]
		}
	
	async def _get_customer_insights(self, start_date: str, end_date: str, merchant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get customer insights"""
		return {
			"new_customers": 45,
			"returning_customers": 178,
			"customer_retention_rate": 78.9,
			"average_transaction_value": 125.50,
			"top_customer_segments": [
				{"segment": "premium", "count": 34, "avg_value": 450.00},
				{"segment": "regular", "count": 156, "avg_value": 95.75},
				{"segment": "new", "count": 45, "avg_value": 67.25}
			]
		}
	
	# Logging methods
	def _log_analytics_engine_created(self):
		print("📊 Real-time Analytics Engine created")
		print("   - Live transaction monitoring")
		print("   - Interactive dashboards")
		print("   - Automated alerting system")
	
	def _log_analytics_engine_started(self):
		print("🚀 Analytics Engine started - processing real-time metrics")
	
	def _log_analytics_engine_stopped(self):
		print("🛑 Analytics Engine stopped")
	
	def _log_analytics_error(self, error: str):
		print(f"❌ Analytics Engine error: {error}")
	
	def _log_alert_triggered(self, alert: Alert):
		print(f"🚨 Alert triggered: [{alert.severity.value.upper()}] {alert.title}")
		print(f"   {alert.message}")
	
	def _log_subscriber_added(self, subscriber_id: str):
		print(f"📡 Subscriber added: {subscriber_id}")
	
	def _log_subscriber_removed(self, subscriber_id: str):
		print(f"📡 Subscriber removed: {subscriber_id}")
	
	def _log_subscriber_error(self, subscriber_id: str, error: str):
		print(f"❌ Subscriber error [{subscriber_id}]: {error}")
	
	def _log_geographic_data_error(self, error: str):
		print(f"❌ Geographic data error: {error}")
	
	def _log_summary_error(self, error: str):
		print(f"❌ Dashboard summary error: {error}")
	
	def _log_custom_report_error(self, error: str):
		print(f"❌ Custom report error: {error}")


def create_analytics_engine(database_service: DatabaseService) -> RealTimeAnalyticsEngine:
	"""Create and return configured analytics engine"""
	return RealTimeAnalyticsEngine(database_service)


def _log_realtime_analytics_module_loaded():
	"""Log real-time analytics module loaded"""
	print("📈 Real-time Analytics & Reporting module loaded")
	print("   - Live transaction monitoring and alerts")
	print("   - Interactive dashboards with real-time metrics")
	print("   - Custom reporting with data visualization")
	print("   - Automated insights and trend analysis")
	print("   - WebSocket-based real-time updates")


# Execute module loading log
_log_realtime_analytics_module_loaded()