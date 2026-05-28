"""
Comprehensive monitoring and alerting service for payment gateway
Provides metrics collection, health checks, and alerting capabilities.
"""

from __future__ import annotations

import asyncio
import operator
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
try:
	import structlog
	logger = structlog.get_logger()
except ImportError:  # pragma: no cover - exercised in dependency-light tests
	import logging

	class _StandardLoggerAdapter:
		"""Small adapter that accepts structlog-style keyword fields."""

		def __init__(self, name: str):
			self._logger = logging.getLogger(name)

		def debug(self, event: str, **fields: Any) -> None:
			self._logger.debug("%s %s", event, fields)

		def info(self, event: str, **fields: Any) -> None:
			self._logger.info("%s %s", event, fields)

		def warning(self, event: str, **fields: Any) -> None:
			self._logger.warning("%s %s", event, fields)

		def error(self, event: str, **fields: Any) -> None:
			self._logger.error("%s %s", event, fields)

	logger = _StandardLoggerAdapter(__name__)

try:
	from prometheus_client import (
		Counter, Histogram, Gauge, Summary, CollectorRegistry,
		generate_latest, CONTENT_TYPE_LATEST
	)
except ImportError:  # pragma: no cover - exercised in dependency-light tests
	CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

	@dataclass
	class _Sample:
		name: str
		labels: Dict[str, str]
		value: float

	class CollectorRegistry:
		"""Minimal in-process registry compatible with the methods used here."""

		def __init__(self):
			self._collectors: List[Any] = []

		def register(self, collector: Any) -> None:
			self._collectors.append(collector)

	class _MetricFamily:
		def __init__(self, name: str, samples: List[_Sample]):
			self.name = name
			self.samples = samples

	class _LabelledMetric:
		def __init__(self, metric: "_Metric", labels: Dict[str, str]):
			self._metric = metric
			self._labels = labels

		def inc(self, amount: float = 1.0) -> None:
			self._metric._inc(self._labels, amount)

		def set(self, value: float) -> None:
			self._metric._set(self._labels, value)

		def observe(self, value: float) -> None:
			self._metric._observe(self._labels, value)

	class _Metric:
		def __init__(self, name: str, documentation: str, labelnames: Optional[List[str]] = None,
					 registry: Optional[CollectorRegistry] = None, **_: Any):
			self._name = name
			self._documentation = documentation
			self._labelnames = list(labelnames or [])
			self._values: Dict[tuple[tuple[str, str], ...], float] = {}
			if registry is not None:
				registry.register(self)

		def labels(self, **labels: str) -> _LabelledMetric:
			for label in self._labelnames:
				labels.setdefault(label, "")
			return _LabelledMetric(self, labels)

		def _key(self, labels: Dict[str, str]) -> tuple[tuple[str, str], ...]:
			return tuple(sorted((str(k), str(v)) for k, v in labels.items()))

		def _labels_from_key(self, key: tuple[tuple[str, str], ...]) -> Dict[str, str]:
			return dict(key)

		def inc(self, amount: float = 1.0) -> None:
			self._inc({}, amount)

		def set(self, value: float) -> None:
			self._set({}, value)

		def observe(self, value: float) -> None:
			self._observe({}, value)

		def _inc(self, labels: Dict[str, str], amount: float = 1.0) -> None:
			key = self._key(labels)
			self._values[key] = self._values.get(key, 0.0) + float(amount)

		def _set(self, labels: Dict[str, str], value: float) -> None:
			self._values[self._key(labels)] = float(value)

		def _observe(self, labels: Dict[str, str], value: float) -> None:
			self._inc(labels, value)

		def collect(self) -> List[_MetricFamily]:
			samples = [
				_Sample(self._name, self._labels_from_key(key), value)
				for key, value in self._values.items()
			]
			return [_MetricFamily(self._name, samples)]

	class Counter(_Metric):
		pass

	class Gauge(_Metric):
		pass

	class Summary(_Metric):
		pass

	class Histogram(_Metric):
		def __init__(self, name: str, documentation: str, labelnames: Optional[List[str]] = None,
					 buckets: Optional[List[float]] = None, registry: Optional[CollectorRegistry] = None,
					 **kwargs: Any):
			super().__init__(name, documentation, labelnames, registry, **kwargs)
			self._buckets = sorted(float(bucket) for bucket in (buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]))
			self._observations: Dict[tuple[tuple[str, str], ...], List[float]] = {}

		def _observe(self, labels: Dict[str, str], value: float) -> None:
			key = self._key(labels)
			self._observations.setdefault(key, []).append(float(value))

		def collect(self) -> List[_MetricFamily]:
			samples: List[_Sample] = []
			for key, observations in self._observations.items():
				labels = self._labels_from_key(key)
				for bucket in self._buckets:
					bucket_labels = {**labels, "le": str(bucket)}
					samples.append(_Sample(
						f"{self._name}_bucket",
						bucket_labels,
						float(sum(1 for value in observations if value <= bucket))
					))
				samples.append(_Sample(f"{self._name}_count", labels, float(len(observations))))
				samples.append(_Sample(f"{self._name}_sum", labels, float(sum(observations))))
			return [_MetricFamily(self._name, samples)]

	def generate_latest(registry: CollectorRegistry) -> bytes:
		lines: List[str] = []
		for collector in registry._collectors:
			for family in collector.collect():
				for sample in family.samples:
					labels = ",".join(f'{key}="{value}"' for key, value in sorted(sample.labels.items()))
					label_text = f"{{{labels}}}" if labels else ""
					lines.append(f"{sample.name}{label_text} {sample.value}")
		return ("\n".join(lines) + "\n").encode("utf-8")

_ALERT_CONDITION_PATTERN = re.compile(
	r"^\s*(?P<metric>[a-zA-Z_][a-zA-Z0-9_]*)\s*"
	r"(?P<operator>>=|<=|==|!=|>|<)\s*"
	r"(?P<threshold>-?\d+(?:\.\d+)?)\s*$"
)

_ALERT_OPERATORS = {
	">": operator.gt,
	">=": operator.ge,
	"<": operator.lt,
	"<=": operator.le,
	"==": operator.eq,
	"!=": operator.ne,
}

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
		"""Evaluate alert condition against the current in-process metric samples."""
		match = _ALERT_CONDITION_PATTERN.match(alert.condition)
		if not match:
			logger.warning("alert_condition_parse_failed",
				alert_id=alert.id,
				condition=alert.condition
			)
			return False

		metric_name = match.group("metric")
		operator_symbol = match.group("operator")
		threshold = float(match.group("threshold"))
		current_value = self._get_alert_metric_value(metric_name, alert)
		if current_value is None:
			logger.warning("alert_metric_unavailable",
				alert_id=alert.id,
				metric=metric_name
			)
			return False

		return _ALERT_OPERATORS[operator_symbol](current_value, threshold)

	def _get_alert_metric_value(self, metric_name: str, alert: Alert) -> Optional[float]:
		"""Return the current value for a supported alert condition metric."""
		transaction_counts = self._transaction_status_counts()
		total_transactions = sum(transaction_counts.values())

		if metric_name == "error_rate":
			if total_transactions == 0:
				return 0.0
			return self._failed_transaction_count(transaction_counts) / total_transactions

		if metric_name == "success_rate":
			if total_transactions == 0:
				return 1.0
			return self._successful_transaction_count(transaction_counts) / total_transactions

		if metric_name == "processor_availability":
			return self._min_sample_value(self.processor_availability, "payment_processor_availability", default=alert.threshold)

		if metric_name == "p95_latency":
			return self._histogram_quantile_ms(self.transaction_duration, "payment_processing_duration_seconds", 0.95)

		if metric_name == "fraud_rate":
			fraud_detections = self._sum_sample_values(self.fraud_detections, "fraud_detections_total")
			if total_transactions == 0:
				return 1.0 if fraud_detections > 0 else 0.0
			return fraud_detections / total_transactions

		if metric_name == "settlement_failures":
			return self._sum_sample_values(self.settlement_failures, "settlement_failures_total")

		if metric_name == "db_connections":
			return self._sum_sample_values(self.database_connections, "database_connections_active", default=alert.threshold)

		if metric_name == "model_accuracy":
			return self._min_sample_value(self.ml_model_accuracy, "ml_model_accuracy_score", default=1.0)

		return None

	def _transaction_status_counts(self) -> Dict[str, float]:
		"""Collect transaction counts by status from the Prometheus counter."""
		counts: Dict[str, float] = {}
		for sample in self._metric_samples(self.transaction_counter):
			if sample.name != "payment_transactions_total":
				continue
			status = str(sample.labels.get("status", "")).lower()
			counts[status] = counts.get(status, 0.0) + float(sample.value)
		return counts

	def _successful_transaction_count(self, counts: Dict[str, float]) -> float:
		"""Return transaction count for statuses treated as successful."""
		success_statuses = {
			"success", "succeeded", "successful", "approved", "authorized",
			"captured", "completed", "settled"
		}
		return sum(value for status, value in counts.items() if status in success_statuses)

	def _failed_transaction_count(self, counts: Dict[str, float]) -> float:
		"""Return transaction count for statuses treated as failed."""
		return sum(counts.values()) - self._successful_transaction_count(counts)

	def _sum_sample_values(self, metric: Any, sample_name: str, default: float = 0.0) -> float:
		"""Sum current samples for a metric, returning a default if no samples exist."""
		values = [float(sample.value) for sample in self._metric_samples(metric) if sample.name == sample_name]
		return sum(values) if values else default

	def _min_sample_value(self, metric: Any, sample_name: str, default: float) -> float:
		"""Return the minimum labelled sample value for availability/accuracy gauges."""
		values = [float(sample.value) for sample in self._metric_samples(metric) if sample.name == sample_name]
		return min(values) if values else default

	def _histogram_quantile_ms(self, metric: Any, metric_name: str, quantile: float) -> float:
		"""Approximate a histogram quantile using cumulative bucket samples."""
		buckets_by_labelset: Dict[tuple[tuple[str, str], ...], List[tuple[float, float]]] = {}
		for sample in self._metric_samples(metric):
			if sample.name != f"{metric_name}_bucket":
				continue
			labels = dict(sample.labels)
			bucket_limit = labels.pop("le", None)
			if bucket_limit is None:
				continue
			try:
				limit = float(bucket_limit)
			except ValueError:
				continue
			labelset = tuple(sorted((str(key), str(value)) for key, value in labels.items()))
			buckets_by_labelset.setdefault(labelset, []).append((limit, float(sample.value)))

		quantiles: List[float] = []
		for buckets in buckets_by_labelset.values():
			ordered = sorted(buckets, key=lambda item: item[0])
			total = ordered[-1][1] if ordered else 0.0
			if total <= 0:
				continue
			target = total * quantile
			for upper_bound_seconds, cumulative_count in ordered:
				if cumulative_count >= target:
					quantiles.append(upper_bound_seconds * 1000.0)
					break

		return max(quantiles) if quantiles else 0.0

	def _metric_samples(self, metric: Any) -> List[Any]:
		"""Return all samples exposed by a prometheus-client compatible metric."""
		samples: List[Any] = []
		for family in metric.collect():
			samples.extend(family.samples)
		return samples

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
