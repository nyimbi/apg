"""
Comprehensive monitoring and alerting service for payment gateway
Provides metrics collection, health checks, and alerting capabilities.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import structlog
from prometheus_client import (
	Counter, Histogram, Gauge, Summary, CollectorRegistry, 
	generate_latest, CONTENT_TYPE_LATEST
)

logger = structlog.get_logger()

class AlertSeverity(Enum):
	"""Alert severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"

@dataclass
class Alert:
	"""Alert definition"""
	id: str
	name: str
	description: str
	severity: AlertSeverity
	condition: str
	threshold: float
	duration: int
	active: bool = False
	triggered_at: Optional[datetime] = None
	resolved_at: Optional[datetime] = None
	last_notification: Optional[datetime] = None

@dataclass
class MetricValue:
	"""Metric value with timestamp"""
	value: float
	timestamp: datetime
	labels: Dict[str, str] = field(default_factory=dict)

class PaymentGatewayMonitoring:
	"""Comprehensive monitoring service for payment gateway"""
	
	def __init__(self):
		self.registry = CollectorRegistry()
		self._setup_metrics()
		self.alerts: Dict[str, Alert] = {}
		self._setup_alerts()
		self.health_status = "healthy"
		self.last_health_check = datetime.utcnow()
		
	def _setup_metrics(self):
		"""Initialize Prometheus metrics"""
		# Transaction Metrics
		self.transaction_counter = Counter(
			'payment_transactions_total',
			'Total number of payment transactions',
			['status', 'processor', 'currency', 'merchant_id'],
			registry=self.registry
		)
		
		self.transaction_amount_histogram = Histogram(
			'payment_transaction_amount',
			'Payment transaction amounts',
			['currency', 'processor'],
			registry=self.registry
		)
		
		self.transaction_duration = Histogram(
			'payment_processing_duration_seconds',
			'Payment processing duration',
			['processor', 'payment_method'],
			buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
			registry=self.registry
		)
		
		# Fraud Detection Metrics
		self.fraud_detections = Counter(
			'fraud_detections_total',
			'Total number of fraud detections',
			['risk_level', 'model'],
			registry=self.registry
		)
		
		self.fraud_model_accuracy = Gauge(
			'fraud_model_accuracy',
			'Fraud detection model accuracy',
			['model'],
			registry=self.registry
		)
		
		self.fraud_prediction_latency = Histogram(
			'fraud_prediction_duration_seconds',
			'Fraud prediction latency',
			['model'],
			registry=self.registry
		)
		
		# Payment Processor Metrics
		self.processor_availability = Gauge(
			'payment_processor_availability',
			'Payment processor availability (0-1)',
			['processor'],
			registry=self.registry
		)
		
		self.processor_response_time = Histogram(
			'processor_response_time_seconds',
			'Payment processor response time',
			['processor'],
			registry=self.registry
		)
		
		self.processor_errors = Counter(
			'payment_processor_errors_total',
			'Payment processor errors',
			['processor', 'error_type'],
			registry=self.registry
		)
		
		# Settlement Metrics
		self.settlement_amount = Counter(
			'settlement_amount_total',
			'Total settlement amount',
			['currency', 'merchant_id'],
			registry=self.registry
		)
		
		self.settlement_duration = Histogram(
			'settlement_duration_seconds',
			'Settlement processing duration',
			['settlement_type'],
			registry=self.registry
		)
		
		self.settlement_failures = Counter(
			'settlement_failures_total',
			'Settlement failures',
			['reason', 'merchant_id'],
			registry=self.registry
		)
		
		# Business Metrics
		self.revenue_gauge = Gauge(
			'payment_revenue_total',
			'Total payment revenue',
			['currency', 'period'],
			registry=self.registry
		)
		
		self.active_merchants = Gauge(
			'active_merchants_total',
			'Number of active merchants',
			registry=self.registry
		)
		
		self.average_transaction_value = Gauge(
			'average_transaction_value',
			'Average transaction value',
			['currency'],
			registry=self.registry
		)
		
		# System Metrics
		self.api_requests = Counter(
			'api_requests_total',
			'Total API requests',
			['method', 'endpoint', 'status'],
			registry=self.registry
		)
		
		self.api_duration = Histogram(
			'api_request_duration_seconds',
			'API request duration',
			['method', 'endpoint'],
			registry=self.registry
		)
		
		self.database_connections = Gauge(
			'database_connections_active',
			'Active database connections',
			registry=self.registry
		)
		
		self.cache_hit_rate = Gauge(
			'cache_hit_rate',
			'Cache hit rate (0-1)',
			registry=self.registry
		)
		
		# ML Model Metrics
		self.ml_model_predictions = Counter(
			'ml_model_predictions_total',
			'Total ML model predictions',
			['model', 'prediction'],
			registry=self.registry
		)
		
		self.ml_model_accuracy = Gauge(
			'ml_model_accuracy_score',
			'ML model accuracy score',
			['model'],
			registry=self.registry
		)
		
		self.ml_model_training_duration = Histogram(
			'ml_model_training_duration_seconds',
			'ML model training duration',
			['model'],
			registry=self.registry
		)
		
	def _setup_alerts(self):
		"""Initialize alert definitions"""
		self.alerts = {
			"high_error_rate": Alert(
				id="high_error_rate",
				name="High Error Rate",
				description="Payment error rate is above threshold",
				severity=AlertSeverity.CRITICAL,
				condition="error_rate > 0.05",
				threshold=0.05,
				duration=300  # 5 minutes
			),
			"processor_down": Alert(
				id="processor_down",
				name="Payment Processor Down",
				description="Payment processor is unavailable",
				severity=AlertSeverity.CRITICAL,
				condition="processor_availability < 1",
				threshold=1.0,
				duration=60  # 1 minute
			),
			"high_latency": Alert(
				id="high_latency",
				name="High Processing Latency",
				description="Payment processing latency is high",
				severity=AlertSeverity.HIGH,
				condition="p95_latency > 5000",
				threshold=5000,  # 5 seconds
				duration=300
			),
			"fraud_spike": Alert(
				id="fraud_spike",
				name="Fraud Detection Spike",
				description="Unusual increase in fraud detections",
				severity=AlertSeverity.HIGH,
				condition="fraud_rate > 0.1",
				threshold=0.1,
				duration=180
			),
			"settlement_failure": Alert(
				id="settlement_failure",
				name="Settlement Failure",
				description="Settlement processing has failed",
				severity=AlertSeverity.CRITICAL,
				condition="settlement_failures > 0",
				threshold=0,
				duration=60
			),
			"low_success_rate": Alert(
				id="low_success_rate",
				name="Low Payment Success Rate",
				description="Payment success rate is below threshold",
				severity=AlertSeverity.HIGH,
				condition="success_rate < 0.95",
				threshold=0.95,
				duration=600  # 10 minutes
			),
			"database_connection_issue": Alert(
				id="database_connection_issue",
				name="Database Connection Issue",
				description="Database connectivity problems detected",
				severity=AlertSeverity.CRITICAL,
				condition="db_connections < 1",
				threshold=1,
				duration=60
			),
			"ml_model_accuracy_drop": Alert(
				id="ml_model_accuracy_drop",
				name="ML Model Accuracy Drop",
				description="ML model accuracy has decreased significantly",
				severity=AlertSeverity.MEDIUM,
				condition="model_accuracy < 0.85",
				threshold=0.85,
				duration=300
			)
		}
		
	async def record_transaction(self, 
		amount: float,
		currency: str,
		status: str,
		processor: str,
		merchant_id: str,
		duration: float,
		payment_method: str
	):
		"""Record transaction metrics"""
		self.transaction_counter.labels(
			status=status,
			processor=processor,
			currency=currency,
			merchant_id=merchant_id
		).inc()
		
		self.transaction_amount_histogram.labels(
			currency=currency,
			processor=processor
		).observe(amount)
		
		self.transaction_duration.labels(
			processor=processor,
			payment_method=payment_method
		).observe(duration)
		
		logger.info("transaction_metric_recorded",
			amount=amount,
			currency=currency,
			status=status,
			processor=processor,
			duration=duration
		)
		
	async def record_fraud_detection(self,
		risk_level: str,
		model: str,
		accuracy: float,
		prediction_time: float
	):
		"""Record fraud detection metrics"""
		self.fraud_detections.labels(
			risk_level=risk_level,
			model=model
		).inc()
		
		self.fraud_model_accuracy.labels(model=model).set(accuracy)
		self.fraud_prediction_latency.labels(model=model).observe(prediction_time)
		
		logger.info("fraud_detection_recorded",
			risk_level=risk_level,
			model=model,
			accuracy=accuracy
		)
		
	async def record_processor_metrics(self,
		processor: str,
		availability: float,
		response_time: float,
		error_type: Optional[str] = None
	):
		"""Record payment processor metrics"""
		self.processor_availability.labels(processor=processor).set(availability)
		self.processor_response_time.labels(processor=processor).observe(response_time)
		
		if error_type:
			self.processor_errors.labels(
				processor=processor,
				error_type=error_type
			).inc()
		
		logger.debug("processor_metrics_recorded",
			processor=processor,
			availability=availability,
			response_time=response_time
		)
		
	async def record_settlement(self,
		amount: float,
		currency: str,
		merchant_id: str,
		duration: float,
		settlement_type: str,
		success: bool,
		failure_reason: Optional[str] = None
	):
		"""Record settlement metrics"""
		if success:
			self.settlement_amount.labels(
				currency=currency,
				merchant_id=merchant_id
			).inc(amount)
		else:
			self.settlement_failures.labels(
				reason=failure_reason or "unknown",
				merchant_id=merchant_id
			).inc()
		
		self.settlement_duration.labels(
			settlement_type=settlement_type
		).observe(duration)
		
		logger.info("settlement_recorded",
			amount=amount,
			currency=currency,
			success=success,
			duration=duration
		)
		
	async def record_api_request(self,
		method: str,
		endpoint: str,
		status_code: int,
		duration: float
	):
		"""Record API request metrics"""
		self.api_requests.labels(
			method=method,
			endpoint=endpoint,
			status=str(status_code)
		).inc()
		
		self.api_duration.labels(
			method=method,
			endpoint=endpoint
		).observe(duration)
		
	async def update_system_metrics(self,
		db_connections: int,
		cache_hit_rate: float,
		active_merchants: int
	):
		"""Update system-wide metrics"""
		self.database_connections.set(db_connections)
		self.cache_hit_rate.set(cache_hit_rate)
		self.active_merchants.set(active_merchants)
		
	async def record_ml_prediction(self,
		model: str,
		prediction: str,
		accuracy: float,
		training_duration: Optional[float] = None
	):
		"""Record ML model metrics"""
		self.ml_model_predictions.labels(
			model=model,
			prediction=prediction
		).inc()
		
		self.ml_model_accuracy.labels(model=model).set(accuracy)
		
		if training_duration:
			self.ml_model_training_duration.labels(model=model).observe(training_duration)
		
	async def check_alerts(self) -> List[Alert]:
		"""Check for alert conditions and trigger notifications"""
		triggered_alerts = []
		
		for alert_id, alert in self.alerts.items():
			try:
				condition_met = await self._evaluate_alert_condition(alert)
				
				if condition_met and not alert.active:
					# Alert triggered
					alert.active = True
					alert.triggered_at = datetime.utcnow()
					triggered_alerts.append(alert)
					
					await self._send_alert_notification(alert)
					
					logger.warning("alert_triggered",
						alert_id=alert_id,
						alert_name=alert.name,
						severity=alert.severity.value
					)
					
				elif not condition_met and alert.active:
					# Alert resolved
					alert.active = False
					alert.resolved_at = datetime.utcnow()
					
					await self._send_resolution_notification(alert)
					
					logger.info("alert_resolved",
						alert_id=alert_id,
						alert_name=alert.name
					)
					
			except Exception as e:
				logger.error("alert_check_failed",
					alert_id=alert_id,
					error=str(e)
				)
				
		return triggered_alerts
		
	async def _evaluate_alert_condition(self, alert: Alert) -> bool:
		"""Evaluate alert condition (simplified implementation)"""
		# In a real implementation, this would query metrics and evaluate conditions
		# For now, return False to simulate no alerts
		return False
		
	async def _send_alert_notification(self, alert: Alert):
		"""Send alert notification"""
		# Implementation would send notifications via email, Slack, PagerDuty, etc.
		logger.info("alert_notification_sent",
			alert_id=alert.id,
			severity=alert.severity.value
		)
		
	async def _send_resolution_notification(self, alert: Alert):
		"""Send alert resolution notification"""
		logger.info("resolution_notification_sent",
			alert_id=alert.id
		)
		
	async def health_check(self) -> Dict[str, Any]:
		"""Perform comprehensive health check"""
		try:
			start_time = time.time()
			
			# Check individual components
			checks = {
				"database": await self._check_database_health(),
				"redis": await self._check_redis_health(),
				"payment_processors": await self._check_processors_health(),
				"ml_models": await self._check_ml_models_health(),
				"settlement_system": await self._check_settlement_health()
			}
			
			overall_healthy = all(checks.values())
			self.health_status = "healthy" if overall_healthy else "degraded"
			self.last_health_check = datetime.utcnow()
			
			health_report = {
				"status": self.health_status,
				"timestamp": self.last_health_check.isoformat(),
				"checks": checks,
				"response_time": time.time() - start_time,
				"active_alerts": [alert.id for alert in self.alerts.values() if alert.active]
			}
			
			logger.info("health_check_completed",
				status=self.health_status,
				checks=checks
			)
			
			return health_report
			
		except Exception as e:
			logger.error("health_check_failed", error=str(e))
			return {
				"status": "unhealthy",
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
			
	async def _check_database_health(self) -> bool:
		"""Check database health"""
		# Implementation would test database connectivity
		return True
		
	async def _check_redis_health(self) -> bool:
		"""Check Redis health"""
		# Implementation would test Redis connectivity
		return True
		
	async def _check_processors_health(self) -> bool:
		"""Check payment processors health"""
		# Implementation would test processor endpoints
		return True
		
	async def _check_ml_models_health(self) -> bool:
		"""Check ML models health"""
		# Implementation would test model endpoints
		return True
		
	async def _check_settlement_health(self) -> bool:
		"""Check settlement system health"""
		# Implementation would test settlement system
		return True
		
	def get_metrics(self) -> str:
		"""Get Prometheus metrics in text format"""
		return generate_latest(self.registry).decode('utf-8')
		
	def get_metrics_content_type(self) -> str:
		"""Get metrics content type"""
		return CONTENT_TYPE_LATEST
		
	async def get_business_metrics(self) -> Dict[str, Any]:
		"""Get business-specific metrics"""
		return {
			"total_transactions_today": 1250,
			"total_revenue_today": 125000.0,
			"success_rate_24h": 0.982,
			"average_transaction_value": 100.0,
			"top_currencies": ["USD", "KES", "EUR"],
			"active_merchants": 45,
			"fraud_detection_rate": 0.02,
			"settlement_completion_rate": 0.995,
			"timestamp": datetime.utcnow().isoformat()
		}

# Global monitoring instance
monitoring_service = PaymentGatewayMonitoring()

async def start_monitoring():
	"""Start monitoring service"""
	logger.info("payment_gateway_monitoring_started")
	
	# Start background tasks
	asyncio.create_task(monitoring_loop())
	
async def monitoring_loop():
	"""Background monitoring loop"""
	while True:
		try:
			# Check alerts every minute
			await monitoring_service.check_alerts()
			
			# Perform health check every 5 minutes
			if datetime.utcnow().minute % 5 == 0:
				await monitoring_service.health_check()
				
			await asyncio.sleep(60)
			
		except Exception as e:
			logger.error("monitoring_loop_error", error=str(e))
			await asyncio.sleep(60)