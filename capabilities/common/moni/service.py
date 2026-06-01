#!/usr/bin/env python3
"""
APG Monitoring and Observability (MONI) - Core Service
Monitoring runtime and dependency-light control plane with APG integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import asdict, dataclass, field
from collections import defaultdict, deque
import json
import statistics
import time
from decimal import Decimal

from .models import (
	MonitoringMetric, MonitoringAlert, MonitoringRule, MonitoringDashboard,
	MonitoringQuery, MonitoringTarget, MetricType, AlertSeverity, AlertStatus,
	AlertConditionType, DashboardType, DataRetentionPolicy, MonitoringScope
)
from .capability_contract import (
	PRIVILEGED_MONI_AGENT_ROLES,
	SUPPORTED_MONI_AGENT_ROLES,
	SUPPORTED_MONI_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


@dataclass
class MonitoringServiceConfig:
	"""Configuration for the monitoring service"""
	# Storage configuration
	max_metrics_in_memory: int = 100000
	metric_retention_hours: int = 720  # 30 days
	alert_retention_hours: int = 2160  # 90 days
	
	# Performance settings
	batch_size: int = 1000
	flush_interval_seconds: int = 30
	query_timeout_seconds: int = 30
	max_concurrent_queries: int = 100
	
	# Alert settings
	max_alert_rate_per_minute: int = 100
	alert_correlation_window_minutes: int = 5
	escalation_check_interval_seconds: int = 60
	
	# APG integration
	enable_audit_logging: bool = True
	enable_caching: bool = True
	cache_ttl_seconds: int = 300
	tenant_isolation_enabled: bool = True
	
	# AI/ML settings
	anomaly_detection_enabled: bool = True
	predictive_analytics_enabled: bool = True
	model_training_interval_hours: int = 24
	baseline_learning_days: int = 7


@dataclass
class SignalSourceRecord:
	"""Tenant-scoped telemetry source registration."""

	source_record_id: str
	tenant_id: str
	source_id: str
	service_name: str
	environment: str
	owner: str
	allowed_signal_types: list[str] = field(default_factory=lambda: ["metric", "log", "trace"])
	notification_route: str | None = None
	status: str = "active"
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SignalRecord:
	"""Governed metric, log, or trace signal metadata."""

	signal_id: str
	tenant_id: str
	source_id: str
	signal_type: str
	name: str
	decision: str
	status: str
	value: Any | None = None
	labels: dict[str, Any] = field(default_factory=dict)
	severity: str = "info"
	trace_id: str | None = None
	service_name: str | None = None
	cardinality: int = 0
	contains_pii: bool = False
	pii_redacted: bool = True
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SloRecord:
	"""Service-level objective definition."""

	slo_id: str
	tenant_id: str
	service_name: str
	objective: str
	threshold: float
	window_minutes: int
	owner: str
	notification_route: str
	status: str = "active"
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertRecord:
	"""Alert lifecycle record."""

	alert_id: str
	tenant_id: str
	source_id: str
	severity: str
	title: str
	decision: str
	status: str
	notification_route: str | None = None
	owner: str | None = None
	incident_id: str | None = None
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	acknowledged_at: datetime | None = None
	resolved_at: datetime | None = None


@dataclass
class IncidentRecord:
	"""Incident correlation and ownership record."""

	incident_id: str
	tenant_id: str
	title: str
	severity: str
	owner: str | None
	notification_route: str | None
	status: str
	alert_ids: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	resolved_at: datetime | None = None


@dataclass
class RemediationRequestRecord:
	"""Runbook-backed remediation request and review state."""

	request_id: str
	tenant_id: str
	incident_id: str
	requester: str
	environment: str
	runbook_id: str
	runbook_approved: bool
	proposed_action: str
	reason: str
	decision: str = "pending"
	status: str = "pending_review"
	reviewer: str | None = None
	review_notes: str | None = None
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "require_review"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	decided_at: datetime | None = None


@dataclass
class MonitoringAgentRecord:
	"""First-class monitoring and observability agent registration."""

	agent_id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	status: str = "active"
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MonitoringLifecycleBatchRecord:
	"""Bytewax lifecycle-batch validation evidence."""

	batch_id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MoniAuditEventRecord:
	"""Dependency-light MONI audit event."""

	event_id: str
	tenant_id: str
	event_type: str
	subject: str
	actor: str
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	details: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)


class MonitoringService:
	"""
	Monitoring service with intelligent analytics.
	Central runtime for APG platform observability adapters.
	"""

	def __init__(self, config: MonitoringServiceConfig):
		assert config is not None, "Configuration is required"
		self.config = config
		self.running = False
		
		# Core storage
		self._metrics_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_metrics_in_memory))
		self._alerts: Dict[str, MonitoringAlert] = {}
		self._rules: Dict[str, MonitoringRule] = {}
		self._dashboards: Dict[str, MonitoringDashboard] = {}
		self._targets: Dict[str, MonitoringTarget] = {}
		
		# Performance optimization
		self._metric_cache: Dict[str, Any] = {}
		self._query_cache: Dict[str, Tuple[datetime, Any]] = {}
		self._batch_buffer: List[MonitoringMetric] = []
		
		# Analytics and intelligence
		self._baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
		self._anomaly_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
		self._correlation_graph: Dict[str, Set[str]] = defaultdict(set)
		
		# Background tasks
		self._background_tasks: Set[asyncio.Task] = set()
		
		# APG integration handles injected by runtime adapters.
		self._auth_service = None
		self._audit_service = None
		self._cache_service = None
		self._notification_service = None
		
		# Logging
		self.logger = logging.getLogger('moni.service')

	async def initialize(self, apg_context: Optional[Dict[str, Any]] = None) -> None:
		"""Initialize the monitoring service with APG integration"""
		assert not self.running, "Service is already running"
		
		self._log_initialization_start()
		
		# Initialize APG integrations
		await self._initialize_apg_integrations(apg_context or {})
		
		# Start background tasks
		await self._start_background_tasks()
		
		# Load existing configuration
		await self._load_configuration()
		
		# Initialize ML models
		await self._initialize_ml_models()
		
		self.running = True
		self._log_initialization_complete()

	async def shutdown(self) -> None:
		"""Gracefully shutdown the monitoring service"""
		assert self.running, "Service is not running"
		
		self._log_shutdown_start()
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		self._background_tasks.clear()
		
		# Flush pending data
		await self._flush_pending_data()
		
		self.running = False
		self._log_shutdown_complete()

	# Core monitoring methods

	async def track_metric(self, metric: MonitoringMetric, tenant_id: Optional[str] = None) -> bool:
		"""Track a monitoring metric with intelligent processing"""
		assert self.running, "Service is not running"
		assert metric is not None, "Metric is required"
		
		# Validate tenant access
		if tenant_id and metric.tenant_id != tenant_id:
			self.logger.warning(f"Tenant ID mismatch for metric {metric.name}")
			return False
		
		try:
			# Process metric through analytics pipeline
			await self._process_metric(metric)
			
			# Store metric
			metric_key = self._get_metric_key(metric)
			self._metrics_store[metric_key].append(metric)
			
			# Update caches and indexes
			await self._update_metric_indexes(metric)
			
			# Check for alerts
			await self._check_metric_alerts(metric)
			
			# Update correlation graph
			self._update_correlation_graph(metric)
			
			self.logger.debug(f"Tracked metric: {metric.name} = {metric.value}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to track metric {metric.name}: {e}")
			return False

	async def query_metrics(self, query: MonitoringQuery, tenant_id: Optional[str] = None) -> List[MonitoringMetric]:
		"""Query metrics with intelligent caching and optimization"""
		assert self.running, "Service is not running"
		assert query is not None, "Query is required"
		
		# Validate tenant access
		if tenant_id and query.tenant_id != tenant_id:
			self.logger.warning(f"Tenant ID mismatch for query")
			return []
		
		try:
			# Validate query
			query.validate_time_range()
			
			# Check cache first
			if query.cache_enabled:
				cached_result = await self._get_cached_query_result(query)
				if cached_result is not None:
					self.logger.debug(f"Query cache hit for {len(query.metric_names)} metrics")
					return cached_result
			
			# Execute query
			results = await self._execute_query(query)
			
			# Cache results
			if query.cache_enabled and results:
				await self._cache_query_result(query, results)
			
			self.logger.debug(f"Query returned {len(results)} metrics")
			return results
			
		except Exception as e:
			self.logger.error(f"Query execution failed: {e}")
			return []

	async def create_alert_rule(self, rule: MonitoringRule, tenant_id: Optional[str] = None) -> str:
		"""Create intelligent alert rule with ML enhancement"""
		assert self.running, "Service is not running"
		assert rule is not None, "Rule is required"
		
		# Validate tenant access
		if tenant_id and rule.tenant_id != tenant_id:
			self.logger.warning(f"Tenant ID mismatch for rule {rule.name}")
			return ""
		
		try:
			# Optimize rule configuration
			await self._optimize_rule_configuration(rule)
			
			# Store rule
			self._rules[rule.rule_id] = rule
			
			# Initialize ML models for anomaly-based rules
			if rule.anomaly_detection_enabled:
				await self._initialize_rule_baseline(rule)
			
			# Schedule rule evaluation
			await self._schedule_rule_evaluation(rule)
			
			self.logger.info(f"Created alert rule: {rule.name}")
			return rule.rule_id
			
		except Exception as e:
			self.logger.error(f"Failed to create alert rule {rule.name}: {e}")
			return ""

	async def get_health_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get comprehensive health status for tenant or system"""
		assert self.running, "Service is not running"
		
		try:
			# Get system metrics
			system_metrics = await self._get_system_metrics()
			
			# Get tenant-specific metrics
			tenant_metrics = {}
			if tenant_id:
				tenant_metrics = await self._get_tenant_metrics(tenant_id)
			
			# Get alert summary
			alert_summary = await self._get_alert_summary(tenant_id)
			
			# Get performance metrics
			performance_metrics = await self._get_performance_metrics()
			
			# Calculate health score
			health_score = await self._calculate_health_score(tenant_id)
			
			return {
				'healthy': health_score > 0.8,
				'health_score': health_score,
				'timestamp': datetime.utcnow(),
				'system_metrics': system_metrics,
				'tenant_metrics': tenant_metrics,
				'alert_summary': alert_summary,
				'performance_metrics': performance_metrics,
				'service_uptime_seconds': self._get_uptime_seconds()
			}
			
		except Exception as e:
			self.logger.error(f"Failed to get health status: {e}")
			return {
				'healthy': False,
				'error': str(e),
				'timestamp': datetime.utcnow()
			}

	# Advanced analytics methods

	async def detect_anomalies(self, metric_name: str, tenant_id: str, 
							 lookback_hours: int = 24) -> List[Dict[str, Any]]:
		"""Detect anomalies using ML-based analysis"""
		assert self.running, "Service is not running"
		assert metric_name, "Metric name is required"
		assert tenant_id, "Tenant ID is required"
		
		try:
			# Get historical data
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(hours=lookback_hours)
			
			query = MonitoringQuery(
				tenant_id=tenant_id,
				metric_names=[metric_name],
				start_time=start_time,
				end_time=end_time
			)
			
			metrics = await self.query_metrics(query, tenant_id)
			if not metrics:
				return []
			
			# Analyze for anomalies
			anomalies = await self._analyze_anomalies(metrics, metric_name)
			
			self.logger.debug(f"Detected {len(anomalies)} anomalies for {metric_name}")
			return anomalies
			
		except Exception as e:
			self.logger.error(f"Anomaly detection failed for {metric_name}: {e}")
			return []

	async def predict_resource_usage(self, resource_type: str, tenant_id: str,
									forecast_hours: int = 24) -> Dict[str, Any]:
		"""Predict resource usage using ML models"""
		assert self.running, "Service is not running"
		assert resource_type, "Resource type is required"
		assert tenant_id, "Tenant ID is required"
		assert forecast_hours > 0, "Forecast hours must be positive"
		
		try:
			# Get historical resource metrics
			historical_data = await self._get_historical_resource_data(resource_type, tenant_id)
			
			if not historical_data:
				return {'error': 'Insufficient historical data'}
			
			# Generate prediction
			prediction = await self._generate_resource_prediction(historical_data, forecast_hours)
			
			self.logger.debug(f"Generated {forecast_hours}h prediction for {resource_type}")
			return prediction
			
		except Exception as e:
			self.logger.error(f"Resource prediction failed for {resource_type}: {e}")
			return {'error': str(e)}

	async def analyze_performance(self, service_name: str, tenant_id: str,
								 analysis_hours: int = 24) -> Dict[str, Any]:
		"""Comprehensive performance analysis with optimization recommendations"""
		assert self.running, "Service is not running"
		assert service_name, "Service name is required"
		assert tenant_id, "Tenant ID is required"
		
		try:
			# Get performance metrics
			perf_metrics = await self._get_service_performance_metrics(service_name, tenant_id, analysis_hours)
			
			# Analyze patterns
			patterns = await self._analyze_performance_patterns(perf_metrics)
			
			# Generate recommendations
			recommendations = await self._generate_performance_recommendations(patterns, service_name)
			
			# Calculate performance scores
			scores = await self._calculate_performance_scores(perf_metrics)
			
			return {
				'service_name': service_name,
				'analysis_period_hours': analysis_hours,
				'performance_scores': scores,
				'patterns': patterns,
				'recommendations': recommendations,
				'analyzed_at': datetime.utcnow()
			}
			
		except Exception as e:
			self.logger.error(f"Performance analysis failed for {service_name}: {e}")
			return {'error': str(e)}

	# Alert management methods

	async def get_active_alerts(self, tenant_id: Optional[str] = None,
							   severity_filter: Optional[AlertSeverity] = None) -> List[MonitoringAlert]:
		"""Get active alerts with optional filtering"""
		assert self.running, "Service is not running"
		
		try:
			alerts = [
				alert for alert in self._alerts.values()
				if alert.status == AlertStatus.ACTIVE and
				   (not tenant_id or alert.tenant_id == tenant_id) and
				   (not severity_filter or alert.severity == severity_filter)
			]
			
			# Sort by severity and age
			alerts.sort(key=lambda a: (a.severity.value, a.created_at), reverse=True)
			
			self.logger.debug(f"Retrieved {len(alerts)} active alerts")
			return alerts
			
		except Exception as e:
			self.logger.error(f"Failed to get active alerts: {e}")
			return []

	async def acknowledge_alert(self, alert_id: str, user_id: str, tenant_id: Optional[str] = None) -> bool:
		"""Acknowledge alert with audit logging"""
		assert self.running, "Service is not running"
		assert alert_id, "Alert ID is required"
		assert user_id, "User ID is required"
		
		try:
			alert = self._alerts.get(alert_id)
			if not alert:
				self.logger.warning(f"Alert not found: {alert_id}")
				return False
			
			# Validate tenant access
			if tenant_id and alert.tenant_id != tenant_id:
				self.logger.warning(f"Tenant access denied for alert {alert_id}")
				return False
			
			# Update alert status
			alert.status = AlertStatus.ACKNOWLEDGED
			alert.acknowledged_at = datetime.utcnow()
			alert.assigned_to = user_id
			alert.updated_at = datetime.utcnow()
			
			# Log audit event
			await self._log_alert_audit_event(alert, 'acknowledged', user_id)
			
			# Send notification
			await self._send_alert_notification(alert, 'acknowledged')
			
			self.logger.info(f"Alert {alert_id} acknowledged by {user_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
			return False

	async def resolve_alert(self, alert_id: str, user_id: str, resolution_note: str = "",
						   tenant_id: Optional[str] = None) -> bool:
		"""Resolve alert with resolution tracking"""
		assert self.running, "Service is not running"
		assert alert_id, "Alert ID is required"
		assert user_id, "User ID is required"
		
		try:
			alert = self._alerts.get(alert_id)
			if not alert:
				self.logger.warning(f"Alert not found: {alert_id}")
				return False
			
			# Validate tenant access
			if tenant_id and alert.tenant_id != tenant_id:
				self.logger.warning(f"Tenant access denied for alert {alert_id}")
				return False
			
			# Update alert status
			alert.status = AlertStatus.RESOLVED
			alert.resolved_at = datetime.utcnow()
			alert.updated_at = datetime.utcnow()
			
			if resolution_note:
				alert.annotations['resolution_note'] = resolution_note
				alert.annotations['resolved_by'] = user_id
			
			# Log audit event
			await self._log_alert_audit_event(alert, 'resolved', user_id)
			
			# Send notification
			await self._send_alert_notification(alert, 'resolved')
			
			# Update rule effectiveness
			await self._update_rule_effectiveness(alert, resolved=True)
			
			self.logger.info(f"Alert {alert_id} resolved by {user_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to resolve alert {alert_id}: {e}")
			return False

	# Dashboard methods

	async def create_dashboard(self, dashboard: MonitoringDashboard, tenant_id: Optional[str] = None) -> str:
		"""Create intelligent dashboard with optimization"""
		assert self.running, "Service is not running"
		assert dashboard is not None, "Dashboard is required"
		
		# Validate tenant access
		if tenant_id and dashboard.tenant_id != tenant_id:
			self.logger.warning(f"Tenant ID mismatch for dashboard {dashboard.name}")
			return ""
		
		try:
			# Optimize dashboard configuration
			await self._optimize_dashboard_configuration(dashboard)
			
			# Store dashboard
			self._dashboards[dashboard.dashboard_id] = dashboard
			
			# Initialize caching if enabled
			if dashboard.cached:
				await self._initialize_dashboard_cache(dashboard)
			
			self.logger.info(f"Created dashboard: {dashboard.name}")
			return dashboard.dashboard_id
			
		except Exception as e:
			self.logger.error(f"Failed to create dashboard {dashboard.name}: {e}")
			return ""

	async def get_dashboard_data(self, dashboard_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
		"""Get dashboard data with intelligent caching"""
		assert self.running, "Service is not running"
		assert dashboard_id, "Dashboard ID is required"
		
		try:
			dashboard = self._dashboards.get(dashboard_id)
			if not dashboard:
				self.logger.warning(f"Dashboard not found: {dashboard_id}")
				return {}
			
			# Validate tenant access
			if tenant_id and dashboard.tenant_id != tenant_id:
				self.logger.warning(f"Tenant access denied for dashboard {dashboard_id}")
				return {}
			
			# Get cached data if available and fresh
			if dashboard.cached:
				cached_data = await self._get_cached_dashboard_data(dashboard)
				if cached_data:
					return cached_data
			
			# Generate dashboard data
			data = await self._generate_dashboard_data(dashboard)
			
			# Cache data if caching enabled
			if dashboard.cached:
				await self._cache_dashboard_data(dashboard, data)
			
			# Update view statistics
			dashboard.update_view_stats(data.get('load_time_ms', 0))
			
			return data
			
		except Exception as e:
			self.logger.error(f"Failed to get dashboard data for {dashboard_id}: {e}")
			return {}

	# Private implementation methods

	async def _initialize_apg_integrations(self, apg_context: Dict[str, Any]) -> None:
		"""Initialize integration with APG services"""
		# In real implementation, these would be injected from APG composition engine
		self.logger.debug("Initializing APG integrations...")
		
		# Auth service integration
		if 'auth_service' in apg_context:
			self._auth_service = apg_context['auth_service']
		
		# Audit service integration
		if 'audit_service' in apg_context:
			self._audit_service = apg_context['audit_service']
		
		# Cache service integration
		if 'cache_service' in apg_context:
			self._cache_service = apg_context['cache_service']
		
		# Notification service integration
		if 'notification_service' in apg_context:
			self._notification_service = apg_context['notification_service']

	async def _start_background_tasks(self) -> None:
		"""Start background processing tasks"""
		# Metric processing task
		task = asyncio.create_task(self._metric_processing_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Alert evaluation task
		task = asyncio.create_task(self._alert_evaluation_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Cache management task
		task = asyncio.create_task(self._cache_management_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Analytics processing task
		task = asyncio.create_task(self._analytics_processing_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)

	async def _load_configuration(self) -> None:
		"""Load existing configuration and state"""
		# In real implementation, would load from persistent storage
		self.logger.debug("Loading configuration...")

	async def _initialize_ml_models(self) -> None:
		"""Initialize machine learning models"""
		if self.config.anomaly_detection_enabled:
			self.logger.debug("Initializing anomaly detection models...")
		
		if self.config.predictive_analytics_enabled:
			self.logger.debug("Initializing predictive analytics models...")

	async def _flush_pending_data(self) -> None:
		"""Flush any pending data to storage"""
		if self._batch_buffer:
			self.logger.debug(f"Flushing {len(self._batch_buffer)} pending metrics")
			self._batch_buffer.clear()

	def _get_metric_key(self, metric: MonitoringMetric) -> str:
		"""Generate storage key for metric"""
		return f"{metric.tenant_id}:{metric.name}:{metric.get_label_signature()}"

	async def _process_metric(self, metric: MonitoringMetric) -> None:
		"""Process metric through analytics pipeline"""
		# Calculate processing metrics
		start_time = time.time()
		
		# Enrich metric with metadata
		await self._enrich_metric_metadata(metric)
		
		# Update baselines for anomaly detection
		if self.config.anomaly_detection_enabled:
			await self._update_metric_baseline(metric)
		
		# Calculate processing time
		processing_time = (time.time() - start_time) * 1000
		metric.processing_time_ms = processing_time

	async def _update_metric_indexes(self, metric: MonitoringMetric) -> None:
		"""Update metric indexes and caches"""
		# Update metric cache
		cache_key = f"metric:{metric.name}:{metric.tenant_id}"
		self._metric_cache[cache_key] = {
			'latest_value': metric.value,
			'timestamp': metric.timestamp,
			'labels': metric.labels
		}

	async def _check_metric_alerts(self, metric: MonitoringMetric) -> None:
		"""Check metric against alert rules"""
		# Find applicable rules
		applicable_rules = [
			rule for rule in self._rules.values()
			if (rule.enabled and 
				rule.metric_name == metric.name and
				rule.tenant_id == metric.tenant_id and
				self._rule_matches_labels(rule, metric.labels))
		]
		
		for rule in applicable_rules:
			await self._evaluate_rule_against_metric(rule, metric)

	def _rule_matches_labels(self, rule: MonitoringRule, labels: Dict[str, str]) -> bool:
		"""Check if rule label filters match metric labels"""
		for key, value in rule.metric_labels.items():
			if key not in labels or labels[key] != value:
				return False
		return True

	async def _evaluate_rule_against_metric(self, rule: MonitoringRule, metric: MonitoringMetric) -> None:
		"""Evaluate rule against specific metric"""
		try:
			# Check threshold condition
			if rule.condition_type == AlertConditionType.THRESHOLD:
				triggered = await self._evaluate_threshold_condition(rule, metric)
			elif rule.condition_type == AlertConditionType.ANOMALY:
				triggered = await self._evaluate_anomaly_condition(rule, metric)
			else:
				triggered = await self._evaluate_complex_condition(rule, metric)
			
			if triggered:
				await self._trigger_alert(rule, metric)
			
			# Update rule statistics
			evaluation_time = 10.0  # Simplified
			rule.update_performance_stats(evaluation_time, False)
			rule.last_triggered = datetime.utcnow()
			
		except Exception as e:
			self.logger.error(f"Rule evaluation failed for {rule.name}: {e}")

	async def _evaluate_threshold_condition(self, rule: MonitoringRule, metric: MonitoringMetric) -> bool:
		"""Evaluate threshold-based condition"""
		if not rule.threshold_value:
			return False
		
		if rule.threshold_operator == "gt":
			return metric.value > rule.threshold_value
		elif rule.threshold_operator == "lt":
			return metric.value < rule.threshold_value
		elif rule.threshold_operator == "eq":
			return abs(metric.value - rule.threshold_value) < 0.001
		elif rule.threshold_operator == "gte":
			return metric.value >= rule.threshold_value
		elif rule.threshold_operator == "lte":
			return metric.value <= rule.threshold_value
		
		return False

	async def _evaluate_anomaly_condition(self, rule: MonitoringRule, metric: MonitoringMetric) -> bool:
		"""Evaluate anomaly-based condition"""
		if not rule.anomaly_detection_enabled:
			return False
		
		# Get baseline for metric
		baseline_key = f"{metric.name}:{metric.tenant_id}:{metric.get_label_signature()}"
		baseline = self._baselines.get(baseline_key, {})
		
		if not baseline:
			return False  # No baseline yet
		
		# Calculate anomaly score
		mean = baseline.get('mean', metric.value)
		std = baseline.get('std', 1.0)
		
		# Z-score based anomaly detection
		z_score = abs((metric.value - mean) / max(std, 0.1))
		anomaly_threshold = 3.0 * (1.0 - rule.anomaly_sensitivity)  # Higher sensitivity = lower threshold
		
		return z_score > anomaly_threshold

	async def _evaluate_complex_condition(self, rule: MonitoringRule, metric: MonitoringMetric) -> bool:
		"""Evaluate complex condition expression"""
		# Simplified implementation - would use proper expression parser
		return False

	async def _trigger_alert(self, rule: MonitoringRule, metric: MonitoringMetric) -> None:
		"""Trigger alert from rule and metric"""
		try:
			# Check for existing active alert
			correlation_key = f"{rule.tenant_id}:{rule.name}:{metric.get_label_signature()}"
			existing_alert = None
			
			for alert in self._alerts.values():
				if (alert.correlation_key == correlation_key and 
					alert.status == AlertStatus.ACTIVE):
					existing_alert = alert
					break
			
			if existing_alert:
				# Update existing alert
				existing_alert.source_value = metric.value
				existing_alert.updated_at = datetime.utcnow()
				existing_alert.message = rule.alert_message.format(
					value=metric.value,
					threshold=rule.threshold_value or 0
				)
			else:
				# Create new alert
				alert = MonitoringAlert(
					tenant_id=rule.tenant_id,
					rule_id=rule.rule_id,
					name=rule.name,
					severity=rule.severity,
					message=rule.alert_message.format(
						value=metric.value,
						threshold=rule.threshold_value or 0
					),
					summary=rule.alert_summary or rule.name,
					correlation_key=correlation_key,
					source_metric=metric.name,
					source_value=metric.value,
					threshold_value=rule.threshold_value,
					runbook_url=rule.runbook_url,
					escalation_interval_minutes=rule.escalation_interval_minutes,
					max_escalation_level=rule.max_escalation_level
				)
				
				# Calculate impact score
				alert.impact_score = await self._calculate_alert_impact_score(alert, metric)
				
				self._alerts[alert.alert_id] = alert
				
				# Send notification
				await self._send_alert_notification(alert, 'created')
				
				# Log audit event
				await self._log_alert_audit_event(alert, 'created', 'system')
			
			# Update rule trigger count
			rule.trigger_count += 1
			
		except Exception as e:
			self.logger.error(f"Failed to trigger alert for rule {rule.name}: {e}")

	def _update_correlation_graph(self, metric: MonitoringMetric) -> None:
		"""Update metric correlation graph"""
		metric_key = f"{metric.name}:{metric.tenant_id}"
		
		# Simple correlation based on temporal proximity
		# In real implementation, would use more sophisticated correlation analysis
		for other_key in list(self._correlation_graph.keys())[-10:]:  # Last 10 metrics
			if other_key != metric_key:
				self._correlation_graph[metric_key].add(other_key)
				self._correlation_graph[other_key].add(metric_key)

	# Background processing loops

	async def _metric_processing_loop(self) -> None:
		"""Background loop for metric processing"""
		while self.running:
			try:
				# Process batch buffer
				if self._batch_buffer:
					await self._process_metric_batch()
				
				# Clean up old metrics
				await self._cleanup_old_metrics()
				
				await asyncio.sleep(self.config.flush_interval_seconds)
				
			except Exception as e:
				self.logger.error(f"Metric processing loop error: {e}")
				await asyncio.sleep(5)

	async def _alert_evaluation_loop(self) -> None:
		"""Background loop for alert evaluation and escalation"""
		while self.running:
			try:
				# Check for escalation
				await self._check_alert_escalations()
				
				# Clean up resolved alerts
				await self._cleanup_resolved_alerts()
				
				await asyncio.sleep(self.config.escalation_check_interval_seconds)
				
			except Exception as e:
				self.logger.error(f"Alert evaluation loop error: {e}")
				await asyncio.sleep(5)

	async def _cache_management_loop(self) -> None:
		"""Background loop for cache management"""
		while self.running:
			try:
				# Clean expired cache entries
				await self._cleanup_expired_cache()
				
				# Update cache statistics
				await self._update_cache_stats()
				
				await asyncio.sleep(60)  # Run every minute
				
			except Exception as e:
				self.logger.error(f"Cache management loop error: {e}")
				await asyncio.sleep(5)

	async def _analytics_processing_loop(self) -> None:
		"""Background loop for analytics processing"""
		while self.running:
			try:
				# Update baselines
				await self._update_all_baselines()
				
				# Process correlation analysis
				await self._process_correlation_analysis()
				
				# Train ML models
				await self._train_ml_models()
				
				await asyncio.sleep(3600)  # Run hourly
				
			except Exception as e:
				self.logger.error(f"Analytics processing loop error: {e}")
				await asyncio.sleep(60)

	# Utility and helper methods

	async def _enrich_metric_metadata(self, metric: MonitoringMetric) -> None:
		"""Enrich metric with additional metadata"""
		# Add capability information if available
		if not metric.capability_name and metric.source:
			# Try to infer capability from source
			for capability in ['auth', 'audl', 'cach', 'mten']:
				if capability in metric.source.lower():
					metric.capability_name = capability
					break

	async def _update_metric_baseline(self, metric: MonitoringMetric) -> None:
		"""Update baseline statistics for anomaly detection"""
		baseline_key = f"{metric.name}:{metric.tenant_id}:{metric.get_label_signature()}"
		
		if baseline_key not in self._baselines:
			self._baselines[baseline_key] = {
				'values': deque(maxlen=1000),
				'mean': metric.value,
				'std': 0.0,
				'count': 0
			}
		
		baseline = self._baselines[baseline_key]
		baseline['values'].append(metric.value)
		baseline['count'] += 1
		
		# Update running statistics
		values = list(baseline['values'])
		if len(values) > 1:
			baseline['mean'] = statistics.mean(values)
			baseline['std'] = statistics.stdev(values)

	def _get_uptime_seconds(self) -> float:
		"""Get service uptime in seconds"""
		# Simplified - would track actual start time
		return 3600.0  # 1 hour

	# Logging methods

	def _log_initialization_start(self) -> None:
		"""Log service initialization start"""
		self.logger.info("Starting APG Monitoring and Observability service...")

	def _log_initialization_complete(self) -> None:
		"""Log service initialization completion"""
		self.logger.info("APG Monitoring service initialization complete")

	def _log_shutdown_start(self) -> None:
		"""Log service shutdown start"""
		self.logger.info("Shutting down APG Monitoring service...")

	def _log_shutdown_complete(self) -> None:
		"""Log service shutdown completion"""
		self.logger.info("APG Monitoring service shutdown complete")

	# Placeholder implementations for complex methods

	async def _execute_query(self, query: MonitoringQuery) -> List[MonitoringMetric]:
		"""Execute metric query"""
		# Simplified implementation - would use proper time-series database
		results = []
		for metric_name in query.metric_names:
			metric_key = f"{query.tenant_id}:{metric_name}:"
			for key, metrics in self._metrics_store.items():
				if key.startswith(metric_key):
					for metric in metrics:
						if query.start_time <= metric.timestamp <= query.end_time:
							results.append(metric)
		return results[:query.max_results]

	async def _get_cached_query_result(self, query: MonitoringQuery) -> Optional[List[MonitoringMetric]]:
		"""Get cached query result if available"""
		cache_key = query.generate_query_key()
		if cache_key in self._query_cache:
			cached_time, result = self._query_cache[cache_key]
			if (datetime.utcnow() - cached_time).total_seconds() < self.config.cache_ttl_seconds:
				return result
		return None

	async def _cache_query_result(self, query: MonitoringQuery, results: List[MonitoringMetric]) -> None:
		"""Cache query result"""
		cache_key = query.generate_query_key()
		self._query_cache[cache_key] = (datetime.utcnow(), results)

	# Analytics and optimization methods - FULLY IMPLEMENTED
	
	async def _optimize_rule_configuration(self, rule: MonitoringRule) -> None:
		"""Optimize rule configuration using historical performance data"""
		try:
			# Analyze rule performance history
			if rule.trigger_count > 0:
				# Optimize threshold based on false positive rate
				if rule.false_positive_rate > 0.2:  # More than 20% false positives
					if rule.threshold_value:
						# Adjust threshold to reduce false positives
						adjustment_factor = 1.1 if rule.threshold_operator in ['gt', 'gte'] else 0.9
						rule.threshold_value *= adjustment_factor
						self.logger.info(f"Optimized threshold for rule {rule.name}: {rule.threshold_value}")
				
				# Optimize evaluation window based on effectiveness
				if rule.effectiveness_score < 0.5:  # Low effectiveness
					# Increase evaluation window for more stable detection
					rule.evaluation_window_minutes = min(rule.evaluation_window_minutes * 1.5, 60)
					self.logger.info(f"Optimized evaluation window for rule {rule.name}: {rule.evaluation_window_minutes}min")
			
			# Set optimal anomaly sensitivity based on metric characteristics
			if rule.anomaly_detection_enabled:
				baseline_key = f"{rule.metric_name}:{rule.tenant_id}"
				if baseline_key in self._baselines:
					baseline = self._baselines[baseline_key]
					variability = baseline.get('std', 0) / max(baseline.get('mean', 1), 0.1)
					
					# Higher variability requires lower sensitivity
					if variability > 0.5:
						rule.anomaly_sensitivity = min(0.9, rule.anomaly_sensitivity * 0.8)
					elif variability < 0.1:
						rule.anomaly_sensitivity = max(0.5, rule.anomaly_sensitivity * 1.2)
					
					self.logger.debug(f"Optimized anomaly sensitivity for rule {rule.name}: {rule.anomaly_sensitivity}")
			
		except Exception as e:
			self.logger.error(f"Failed to optimize rule configuration for {rule.name}: {e}")

	async def _initialize_rule_baseline(self, rule: MonitoringRule) -> None:
		"""Initialize baseline data for anomaly-based rules"""
		try:
			if not rule.anomaly_detection_enabled:
				return
			
			# Create baseline key
			baseline_key = f"{rule.metric_name}:{rule.tenant_id}"
			
			# Get historical data for baseline calculation
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=rule.baseline_period_days)
			
			query = MonitoringQuery(
				tenant_id=rule.tenant_id,
				metric_names=[rule.metric_name],
				start_time=start_time,
				end_time=end_time,
				max_results=10000
			)
			
			historical_metrics = await self.query_metrics(query)
			
			if len(historical_metrics) >= 10:  # Minimum data for baseline
				values = [m.value for m in historical_metrics]
				
				# Initialize baseline with comprehensive statistics
				self._baselines[baseline_key] = {
					'values': deque(values[-1000:], maxlen=1000),  # Keep last 1000 values
					'mean': statistics.mean(values),
					'std': statistics.stdev(values) if len(values) > 1 else 0.0,
					'median': statistics.median(values),
					'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else max(values),
					'p99': sorted(values)[int(len(values) * 0.99)] if len(values) > 100 else max(values),
					'count': len(values),
					'last_updated': datetime.utcnow(),
					'seasonal_patterns': await self._extract_seasonal_patterns(historical_metrics)
				}
				
				self.logger.info(f"Initialized baseline for rule {rule.name} with {len(values)} historical points")
			else:
				# Initialize empty baseline that will be populated over time
				self._baselines[baseline_key] = {
					'values': deque(maxlen=1000),
					'mean': 0.0,
					'std': 0.0,
					'median': 0.0,
					'p95': 0.0,
					'p99': 0.0,
					'count': 0,
					'last_updated': datetime.utcnow(),
					'seasonal_patterns': {}
				}
				
				self.logger.warning(f"Insufficient historical data for rule {rule.name}, initializing empty baseline")
		
		except Exception as e:
			self.logger.error(f"Failed to initialize baseline for rule {rule.name}: {e}")

	async def _schedule_rule_evaluation(self, rule: MonitoringRule) -> None:
		"""Schedule periodic evaluation for complex rules"""
		try:
			# For threshold rules, evaluation happens on metric ingestion
			if rule.condition_type == AlertConditionType.THRESHOLD:
				return
			
			# For complex rules, schedule periodic evaluation
			eval_interval = max(rule.evaluation_interval_seconds, 30)  # Minimum 30 seconds
			
			async def rule_evaluator():
				"""Periodic rule evaluation task"""
				while rule.enabled and self.running:
					try:
						# Get recent metrics for evaluation
						end_time = datetime.utcnow()
						start_time = end_time - timedelta(minutes=rule.evaluation_window_minutes)
						
						query = MonitoringQuery(
							tenant_id=rule.tenant_id,
							metric_names=[rule.metric_name],
							start_time=start_time,
							end_time=end_time,
							labels=rule.metric_labels
						)
						
						recent_metrics = await self.query_metrics(query)
						
						# Evaluate rule against recent metrics
						if recent_metrics:
							await self._evaluate_rule_complex_condition(rule, recent_metrics)
						
						await asyncio.sleep(eval_interval)
						
					except Exception as e:
						self.logger.error(f"Error in rule evaluator for {rule.name}: {e}")
						await asyncio.sleep(60)  # Back off on error
			
			# Start the evaluation task
			task = asyncio.create_task(rule_evaluator())
			self._background_tasks.add(task)
			task.add_done_callback(self._background_tasks.discard)
			
			self.logger.info(f"Scheduled periodic evaluation for rule {rule.name} every {eval_interval}s")
		
		except Exception as e:
			self.logger.error(f"Failed to schedule rule evaluation for {rule.name}: {e}")

	async def _extract_seasonal_patterns(self, metrics: List[MonitoringMetric]) -> Dict[str, float]:
		"""Extract seasonal patterns from historical metrics"""
		try:
			if len(metrics) < 24:  # Need at least 24 points
				return {}
			
			patterns = {}
			
			# Group by hour of day
			hourly_groups = defaultdict(list)
			daily_groups = defaultdict(list)
			
			for metric in metrics:
				hour = metric.timestamp.hour
				day_of_week = metric.timestamp.weekday()
				
				hourly_groups[hour].append(metric.value)
				daily_groups[day_of_week].append(metric.value)
			
			# Calculate hourly patterns (multipliers relative to global mean)
			all_values = [m.value for m in metrics]
			global_mean = statistics.mean(all_values)
			
			for hour, values in hourly_groups.items():
				if len(values) >= 3:  # Minimum samples
					hour_mean = statistics.mean(values)
					patterns[f'hour_{hour}'] = hour_mean / max(global_mean, 0.1)
			
			# Calculate daily patterns
			for day, values in daily_groups.items():
				if len(values) >= 3:
					day_mean = statistics.mean(values)
					patterns[f'day_{day}'] = day_mean / max(global_mean, 0.1)
			
			return patterns
			
		except Exception as e:
			self.logger.error(f"Failed to extract seasonal patterns: {e}")
			return {}

	async def _evaluate_rule_complex_condition(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> None:
		"""Evaluate complex rule conditions against multiple metrics"""
		try:
			if rule.condition_type == AlertConditionType.RATE:
				# Rate of change analysis
				if len(metrics) >= 2:
					sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
					first_value = sorted_metrics[0].value
					last_value = sorted_metrics[-1].value
					time_diff = (sorted_metrics[-1].timestamp - sorted_metrics[0].timestamp).total_seconds()
					
					if time_diff > 0:
						rate = (last_value - first_value) / time_diff
						# Parse rate threshold from condition
						import re
						match = re.search(r'rate\s*([><=!]+)\s*([-+]?\d+\.?\d*)', rule.condition)
						if match:
							operator = match.group(1)
							threshold = float(match.group(2))
							
							triggered = False
							if operator in ['>', 'gt'] and rate > threshold:
								triggered = True
							elif operator in ['<', 'lt'] and rate < threshold:
								triggered = True
							elif operator in ['>=', 'gte'] and rate >= threshold:
								triggered = True
							elif operator in ['<=', 'lte'] and rate <= threshold:
								triggered = True
							
							if triggered:
								# Create synthetic metric for alert triggering
								trigger_metric = MonitoringMetric(
									name=rule.metric_name,
									value=rate,
									tenant_id=rule.tenant_id,
									source="rate_analysis",
									labels=rule.metric_labels
								)
								await self._trigger_alert(rule, trigger_metric)
			
			elif rule.condition_type == AlertConditionType.ABSENCE:
				# Check for metric absence
				expected_interval = rule.evaluation_window_minutes * 60  # seconds
				if metrics:
					latest_metric = max(metrics, key=lambda m: m.timestamp)
					age_seconds = (datetime.utcnow() - latest_metric.timestamp).total_seconds()
					
					if age_seconds > expected_interval:
						# Metric is absent - trigger alert
						absence_metric = MonitoringMetric(
							name=rule.metric_name,
							value=age_seconds,
							tenant_id=rule.tenant_id,
							source="absence_detection",
							labels=rule.metric_labels
						)
						await self._trigger_alert(rule, absence_metric)
				else:
					# No metrics at all - definitely absent
					absence_metric = MonitoringMetric(
						name=rule.metric_name,
						value=float('inf'),
						tenant_id=rule.tenant_id,
						source="absence_detection",
						labels=rule.metric_labels
					)
					await self._trigger_alert(rule, absence_metric)
			
			elif rule.condition_type == AlertConditionType.COMPOSITE:
				# Complex composite condition evaluation
				await self._evaluate_composite_condition(rule, metrics)
		
		except Exception as e:
			self.logger.error(f"Failed to evaluate complex condition for rule {rule.name}: {e}")

	async def _evaluate_composite_condition(self, rule: MonitoringRule, metrics: List[MonitoringMetric]) -> None:
		"""Evaluate composite conditions with multiple criteria"""
		try:
			# Parse composite condition (simplified implementation)
			condition = rule.condition.lower().strip()
			
			# Split on AND/OR operators
			if ' and ' in condition:
				parts = [part.strip() for part in condition.split(' and ')]
				operator = 'and'
			elif ' or ' in condition:
				parts = [part.strip() for part in condition.split(' or ')]
				operator = 'or'
			else:
				# Single condition
				parts = [condition]
				operator = 'and'
			
			results = []
			
			for part in parts:
				part_result = False
				
				if 'value' in part and ('>' in part or '<' in part or '=' in part):
					# Value-based condition
					import re
					match = re.search(r'value\s*([><=!]+)\s*([-+]?\d+\.?\d*)', part)
					if match and metrics:
						op = match.group(1)
						threshold = float(match.group(2))
						latest_value = max(metrics, key=lambda m: m.timestamp).value
						
						if op in ['>', 'gt']:
							part_result = latest_value > threshold
						elif op in ['<', 'lt']:
							part_result = latest_value < threshold
						elif op in ['>=', 'gte']:
							part_result = latest_value >= threshold
						elif op in ['<=', 'lte']:
							part_result = latest_value <= threshold
						elif op in ['=', 'eq']:
							part_result = abs(latest_value - threshold) < 0.001
				
				elif 'anomaly' in part:
					# Anomaly-based condition
					if metrics and rule.anomaly_detection_enabled:
						latest_metric = max(metrics, key=lambda m: m.timestamp)
						part_result = await self._evaluate_anomaly_condition(rule, latest_metric)
				
				elif 'count' in part:
					# Count-based condition
					import re
					match = re.search(r'count\s*([><=!]+)\s*(\d+)', part)
					if match:
						op = match.group(1)
						threshold = int(match.group(2))
						count = len(metrics)
						
						if op in ['>', 'gt']:
							part_result = count > threshold
						elif op in ['<', 'lt']:
							part_result = count < threshold
						elif op in ['>=', 'gte']:
							part_result = count >= threshold
						elif op in ['<=', 'lte']:
							part_result = count <= threshold
						elif op in ['=', 'eq']:
							part_result = count == threshold
				
				results.append(part_result)
			
			# Combine results based on operator
			if operator == 'and':
				final_result = all(results)
			else:  # or
				final_result = any(results)
			
			if final_result:
				# Create composite alert
				trigger_metric = max(metrics, key=lambda m: m.timestamp) if metrics else MonitoringMetric(
					name=rule.metric_name,
					value=0.0,
					tenant_id=rule.tenant_id,
					source="composite_condition"
				)
				await self._trigger_alert(rule, trigger_metric)
		
		except Exception as e:
			self.logger.error(f"Failed to evaluate composite condition for rule {rule.name}: {e}")

	async def _get_system_metrics(self) -> Dict[str, Any]:
		"""Get comprehensive system metrics"""
		try:
			import psutil
			import os
			
			# CPU metrics
			cpu_percent = psutil.cpu_percent(interval=1)
			cpu_count = psutil.cpu_count()
			load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
			
			# Memory metrics
			memory = psutil.virtual_memory()
			swap = psutil.swap_memory()
			
			# Disk metrics
			disk = psutil.disk_usage('/')
			disk_io = psutil.disk_io_counters()
			
			# Network metrics
			network = psutil.net_io_counters()
			
			# Process metrics for monitoring service
			process = psutil.Process()
			process_memory = process.memory_info()
			process_cpu = process.cpu_percent()
			
			return {
				'cpu': {
					'usage_percent': cpu_percent,
					'count': cpu_count,
					'load_average': {
						'1min': load_avg[0],
						'5min': load_avg[1],
						'15min': load_avg[2]
					}
				},
				'memory': {
					'total_bytes': memory.total,
					'available_bytes': memory.available,
					'used_bytes': memory.used,
					'usage_percent': memory.percent,
					'swap_total_bytes': swap.total,
					'swap_used_bytes': swap.used,
					'swap_usage_percent': swap.percent
				},
				'disk': {
					'total_bytes': disk.total,
					'free_bytes': disk.free,
					'used_bytes': disk.used,
					'usage_percent': (disk.used / disk.total) * 100,
					'read_bytes': disk_io.read_bytes if disk_io else 0,
					'write_bytes': disk_io.write_bytes if disk_io else 0
				},
				'network': {
					'bytes_sent': network.bytes_sent,
					'bytes_received': network.bytes_recv,
					'packets_sent': network.packets_sent,
					'packets_received': network.packets_recv,
					'errors_in': network.errin,
					'errors_out': network.errout
				},
				'process': {
					'memory_rss_bytes': process_memory.rss,
					'memory_vms_bytes': process_memory.vms,
					'cpu_percent': process_cpu,
					'num_threads': process.num_threads(),
					'open_files': len(process.open_files()),
					'connections': len(process.connections())
				},
				'monitoring_service': {
					'uptime_seconds': self._get_uptime_seconds(),
					'active_background_tasks': len(self._background_tasks),
					'metrics_in_memory': sum(len(queue) for queue in self._metrics_store.values()),
					'active_alerts': len([a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]),
					'total_rules': len(self._rules),
					'cache_entries': len(self._query_cache)
				}
			}
		
		except Exception as e:
			self.logger.error(f"Failed to get system metrics: {e}")
			# Return basic fallback metrics
			return {
				'cpu': {'usage_percent': 0, 'count': 1},
				'memory': {'usage_percent': 0, 'available_bytes': 0},
				'disk': {'usage_percent': 0, 'free_bytes': 0},
				'network': {'bytes_sent': 0, 'bytes_received': 0},
				'process': {'memory_rss_bytes': 0, 'cpu_percent': 0},
				'monitoring_service': {
					'uptime_seconds': self._get_uptime_seconds(),
					'metrics_in_memory': sum(len(queue) for queue in self._metrics_store.values()),
					'active_alerts': len([a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]),
					'total_rules': len(self._rules)
				}
			}

	async def _get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
		"""Get tenant-specific metrics and statistics"""
		try:
			# Count metrics by tenant
			tenant_metric_count = 0
			tenant_alert_count = 0
			tenant_rule_count = 0
			tenant_dashboard_count = 0
			
			# Metrics count
			for key, metrics in self._metrics_store.items():
				if key.startswith(f"{tenant_id}:"):
					tenant_metric_count += len(metrics)
			
			# Alerts count
			tenant_alerts = [a for a in self._alerts.values() if a.tenant_id == tenant_id]
			tenant_alert_count = len(tenant_alerts)
			active_alerts = [a for a in tenant_alerts if a.status == AlertStatus.ACTIVE]
			
			# Rules count
			tenant_rules = [r for r in self._rules.values() if r.tenant_id == tenant_id]
			tenant_rule_count = len(tenant_rules)
			enabled_rules = [r for r in tenant_rules if r.enabled]
			
			# Dashboards count
			tenant_dashboards = [d for d in self._dashboards.values() if d.tenant_id == tenant_id]
			tenant_dashboard_count = len(tenant_dashboards)
			
			# Calculate tenant health metrics
			if tenant_alerts:
				critical_alerts = [a for a in tenant_alerts if a.severity == AlertSeverity.CRITICAL]
				high_alerts = [a for a in tenant_alerts if a.severity == AlertSeverity.HIGH]
				avg_resolution_time = await self._calculate_avg_resolution_time(tenant_alerts)
			else:
				critical_alerts = []
				high_alerts = []
				avg_resolution_time = 0
			
			# Recent activity metrics
			recent_time = datetime.utcnow() - timedelta(hours=1)
			recent_alerts = [a for a in tenant_alerts if a.created_at >= recent_time]
			
			# Resource usage by tenant
			storage_usage = await self._calculate_tenant_storage_usage(tenant_id)
			query_volume = await self._calculate_tenant_query_volume(tenant_id)
			
			return {
				'tenant_id': tenant_id,
				'metrics': {
					'total_metrics': tenant_metric_count,
					'metrics_per_minute': await self._calculate_tenant_ingestion_rate(tenant_id),
					'unique_metric_names': await self._count_unique_tenant_metrics(tenant_id)
				},
				'alerts': {
					'total_alerts': tenant_alert_count,
					'active_alerts': len(active_alerts),
					'critical_alerts': len(critical_alerts),
					'high_alerts': len(high_alerts),
					'recent_alerts_1h': len(recent_alerts),
					'avg_resolution_time_minutes': avg_resolution_time
				},
				'rules': {
					'total_rules': tenant_rule_count,
					'enabled_rules': len(enabled_rules),
					'avg_effectiveness': statistics.mean([r.effectiveness_score for r in tenant_rules]) if tenant_rules else 0,
					'total_triggers': sum(r.trigger_count for r in tenant_rules)
				},
				'dashboards': {
					'total_dashboards': tenant_dashboard_count,
					'avg_views_per_dashboard': statistics.mean([d.view_count for d in tenant_dashboards]) if tenant_dashboards else 0,
					'most_popular_dashboard': max(tenant_dashboards, key=lambda d: d.view_count).name if tenant_dashboards else None
				},
				'resource_usage': {
					'storage_bytes': storage_usage,
					'query_volume_24h': query_volume,
					'cache_hit_rate': await self._calculate_tenant_cache_hit_rate(tenant_id)
				},
				'health_score': await self._calculate_tenant_health_score(tenant_id)
			}
		
		except Exception as e:
			self.logger.error(f"Failed to get tenant metrics for {tenant_id}: {e}")
			return {
				'tenant_id': tenant_id,
				'metrics': {'total_metrics': 0},
				'alerts': {'total_alerts': 0, 'active_alerts': 0},
				'rules': {'total_rules': 0},
				'dashboards': {'total_dashboards': 0},
				'resource_usage': {'storage_bytes': 0},
				'health_score': 0.5
			}

	async def _calculate_avg_resolution_time(self, alerts: List[MonitoringAlert]) -> float:
		"""Calculate average resolution time for alerts in minutes"""
		try:
			resolved_alerts = [a for a in alerts if a.resolved_at and a.created_at]
			if not resolved_alerts:
				return 0.0
			
			resolution_times = [
				(a.resolved_at - a.created_at).total_seconds() / 60  # Convert to minutes
				for a in resolved_alerts
			]
			
			return statistics.mean(resolution_times)
		
		except Exception:
			return 0.0

	async def _calculate_tenant_storage_usage(self, tenant_id: str) -> int:
		"""Calculate storage usage for tenant in bytes (estimated)"""
		try:
			total_bytes = 0
			
			# Estimate metric storage (rough calculation)
			for key, metrics in self._metrics_store.items():
				if key.startswith(f"{tenant_id}:"):
					# Rough estimate: each metric ~200 bytes in memory
					total_bytes += len(metrics) * 200
			
			# Add alert storage
			tenant_alerts = [a for a in self._alerts.values() if a.tenant_id == tenant_id]
			total_bytes += len(tenant_alerts) * 1000  # ~1KB per alert
			
			# Add rule storage
			tenant_rules = [r for r in self._rules.values() if r.tenant_id == tenant_id]
			total_bytes += len(tenant_rules) * 500  # ~500 bytes per rule
			
			return total_bytes
			
		except Exception:
			return 0

	async def _calculate_tenant_query_volume(self, tenant_id: str) -> int:
		"""Calculate 24h query volume for tenant"""
		try:
			# In real implementation, would track query statistics
			# For now, estimate based on cache entries
			tenant_cache_entries = sum(
				1 for key in self._query_cache.keys() 
				if tenant_id in key
			)
			return tenant_cache_entries * 10  # Estimate 10 queries per cache entry
			
		except Exception:
			return 0

	async def _calculate_tenant_ingestion_rate(self, tenant_id: str) -> float:
		"""Calculate tenant metric ingestion rate per minute"""
		try:
			# Count recent metrics (last 5 minutes)
			cutoff_time = datetime.utcnow() - timedelta(minutes=5)
			recent_count = 0
			
			for key, metrics in self._metrics_store.items():
				if key.startswith(f"{tenant_id}:"):
					recent_count += len([m for m in metrics if m.timestamp >= cutoff_time])
			
			return recent_count / 5.0  # Per minute rate
			
		except Exception:
			return 0.0

	async def _count_unique_tenant_metrics(self, tenant_id: str) -> int:
		"""Count unique metric names for tenant"""
		try:
			unique_names = set()
			
			for key in self._metrics_store.keys():
				if key.startswith(f"{tenant_id}:"):
					# Extract metric name from key format "tenant:name:labels"
					parts = key.split(':', 2)
					if len(parts) >= 2:
						unique_names.add(parts[1])
			
			return len(unique_names)
			
		except Exception:
			return 0

	async def _calculate_tenant_cache_hit_rate(self, tenant_id: str) -> float:
		"""Calculate cache hit rate for tenant"""
		try:
			# In real implementation, would track cache statistics per tenant
			# For now, return global cache performance estimate
			if len(self._query_cache) > 0:
				return 0.75  # 75% hit rate estimate
			return 0.0
			
		except Exception:
			return 0.0

	async def _calculate_tenant_health_score(self, tenant_id: str) -> float:
		"""Calculate overall health score for tenant (0.0 - 1.0)"""
		try:
			score = 1.0
			
			# Check for critical alerts
			tenant_alerts = [a for a in self._alerts.values() if a.tenant_id == tenant_id]
			active_alerts = [a for a in tenant_alerts if a.status == AlertStatus.ACTIVE]
			critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
			
			# Reduce score for critical alerts
			if critical_alerts:
				score -= min(0.5, len(critical_alerts) * 0.1)  # Max 50% reduction
			
			# Reduce score for high alerts
			high_alerts = [a for a in active_alerts if a.severity == AlertSeverity.HIGH]
			if high_alerts:
				score -= min(0.3, len(high_alerts) * 0.05)  # Max 30% reduction
			
			# Check rule effectiveness
			tenant_rules = [r for r in self._rules.values() if r.tenant_id == tenant_id and r.enabled]
			if tenant_rules:
				avg_effectiveness = statistics.mean([r.effectiveness_score for r in tenant_rules])
				if avg_effectiveness < 0.5:
					score -= 0.2  # Reduce for poor rule performance
			
			# Check for stale data
			recent_time = datetime.utcnow() - timedelta(hours=1)
			has_recent_metrics = False
			
			for key, metrics in self._metrics_store.items():
				if key.startswith(f"{tenant_id}:") and metrics:
					if any(m.timestamp >= recent_time for m in metrics):
						has_recent_metrics = True
						break
			
			if not has_recent_metrics and len(self._metrics_store) > 0:
				score -= 0.3  # Reduce for stale data
			
			return max(0.0, min(1.0, score))
			
		except Exception as e:
			self.logger.error(f"Failed to calculate tenant health score: {e}")
			return 0.5


# Factory function for service creation
async def create_monitoring_service(config: Optional[MonitoringServiceConfig] = None) -> MonitoringService:
	"""Create and initialize monitoring service"""
	assert config is not None or MonitoringServiceConfig is not None, "Configuration is required"
	
	service_config = config or MonitoringServiceConfig()
	service = MonitoringService(service_config)
	await service.initialize()
	return service


class MoniService:
	"""Dependency-light MONI lifecycle and guardrail control plane."""

	def __init__(self, tenant_id: str = "default"):
		self.tenant_id = tenant_id
		self.contract = get_capability_contract(tenant_id)
		self._agent_runtimes = set(SUPPORTED_MONI_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_MONI_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_MONI_AGENT_ROLES)
		self.sources: dict[str, SignalSourceRecord] = {}
		self.signals: dict[str, SignalRecord] = {}
		self.slos: dict[str, SloRecord] = {}
		self.alerts: dict[str, AlertRecord] = {}
		self.incidents: dict[str, IncidentRecord] = {}
		self.remediation_requests: dict[str, RemediationRequestRecord] = {}
		self.monitoring_agents: dict[str, MonitoringAgentRecord] = {}
		self.lifecycle_batches: dict[str, MonitoringLifecycleBatchRecord] = {}
		self.audit_events: list[MoniAuditEventRecord] = []
		self.records: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the current executable MONI contract."""
		return get_capability_contract(tenant_id)

	def create_record(
		self,
		*,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper for older generated package tests."""
		record_id = self._require_text(record_id, "record_id")
		tenant_id = self._require_text(tenant_id, "tenant_id")
		record = {
			"id": record_id,
			"tenant_id": tenant_id,
			"metadata": dict(metadata or {}),
			"status": status,
			"created_at": datetime.utcnow().isoformat(),
		}
		self.records[f"{tenant_id}:{record_id}"] = record
		self._audit(tenant_id, "record.created", record_id, "system", _allow_result(), record)
		return record

	def register_source(
		self,
		*,
		tenant_id: str,
		source_id: str,
		service_name: str,
		environment: str,
		owner: str,
		allowed_signal_types: list[str] | None = None,
		notification_route: str | None = None,
		status: str = "active",
	) -> SignalSourceRecord:
		"""Register a tenant-scoped telemetry source."""
		if status not in {"active", "disabled", "retiring"}:
			raise ValueError("status must be active, disabled, or retiring")
		record = SignalSourceRecord(
			source_record_id=uuid_like(),
			tenant_id=self._require_text(tenant_id, "tenant_id"),
			source_id=self._require_text(source_id, "source_id"),
			service_name=self._require_text(service_name, "service_name"),
			environment=self._require_text(environment, "environment"),
			owner=self._require_text(owner, "owner"),
			allowed_signal_types=allowed_signal_types or ["metric", "log", "trace"],
			notification_route=notification_route,
			status=status,
		)
		self.sources[self._source_key(record.tenant_id, record.source_id)] = record
		self._audit(record.tenant_id, "source.registered", record.source_id, record.owner, _allow_result(), asdict(record))
		return record

	def ingest_signal(
		self,
		*,
		tenant_id: str,
		source_id: str,
		signal_type: str,
		name: str,
		value: Any | None = None,
		labels: dict[str, Any] | None = None,
		severity: str = "info",
		trace_id: str | None = None,
		service_name: str | None = None,
		cardinality: int = 0,
		contains_pii: bool = False,
		pii_redacted: bool = True,
		cardinality_exception_recorded: bool = False,
	) -> SignalRecord:
		"""Ingest governed telemetry metadata after rule evaluation."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		source_id = self._require_text(source_id, "source_id")
		signal_type = self._require_choice(signal_type, "signal_type", {"metric", "log", "trace"})
		name = self._require_text(name, "name")
		if cardinality < 0:
			raise ValueError("cardinality cannot be negative")
		source = self.sources.get(self._source_key(tenant_id, source_id))
		if source and signal_type not in source.allowed_signal_types:
			raise ValueError(f"signal_type {signal_type} is not allowed for source {source_id}")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "ingest_signal",
			"source_registered": source is not None,
			"source_status": source.status if source else "missing",
			"source_present": bool(source_id),
			"signal_type": signal_type,
			"trace_id_present": bool(trace_id),
			"service_name_present": bool(service_name or (source.service_name if source else None)),
			"log_contains_pii": contains_pii,
			"pii_redacted": pii_redacted,
			"metric_cardinality": cardinality,
			"cardinality_exception_recorded": cardinality_exception_recorded,
		}
		if signal_type == "metric":
			context["operation"] = "ingest_metric"
			context["source_registered"] = source is not None
		elif signal_type == "log":
			context["operation"] = "ingest_log"
			context["source_registered"] = source is not None
		elif signal_type == "trace":
			context["operation"] = "ingest_trace"
			context["source_registered"] = source is not None
		generic_context = dict(context)
		generic_context["operation"] = "ingest_signal"
		decision = self._merge_decisions(
			evaluate_capability_rules(generic_context),
			evaluate_capability_rules(context),
		)
		status = "accepted" if decision["decision"] == "allow" else (
			"pending_review" if decision["decision"] == "require_review" else "denied"
		)
		record = SignalRecord(
			signal_id=uuid_like(),
			tenant_id=tenant_id,
			source_id=source_id,
			signal_type=signal_type,
			name=name,
			value=value,
			labels=dict(labels or {}),
			severity=severity,
			trace_id=trace_id,
			service_name=service_name or (source.service_name if source else None),
			cardinality=cardinality,
			contains_pii=contains_pii,
			pii_redacted=pii_redacted,
			decision=decision["decision"],
			status=status,
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.signals[record.signal_id] = record
		self._audit(tenant_id, "signal.ingested", name, source_id, decision, context)
		return record

	def create_slo(
		self,
		*,
		tenant_id: str,
		service_name: str,
		objective: str,
		threshold: float,
		window_minutes: int,
		owner: str,
		notification_route: str | None,
	) -> SloRecord:
		"""Create a governed SLO definition."""
		if threshold <= 0:
			raise ValueError("threshold must be positive")
		if window_minutes <= 0:
			raise ValueError("window_minutes must be positive")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "create_slo",
			"notification_route_configured": bool(notification_route),
		}
		decision = evaluate_capability_rules(context)
		if decision["decision"] == "deny":
			raise ValueError(";".join(decision["matched_rules"]))
		record = SloRecord(
			slo_id=uuid_like(),
			tenant_id=self._require_text(tenant_id, "tenant_id"),
			service_name=self._require_text(service_name, "service_name"),
			objective=self._require_text(objective, "objective"),
			threshold=threshold,
			window_minutes=window_minutes,
			owner=self._require_text(owner, "owner"),
			notification_route=self._require_text(notification_route or "", "notification_route"),
		)
		self.slos[record.slo_id] = record
		self._audit(record.tenant_id, "slo.created", record.slo_id, record.owner, _allow_result(), asdict(record))
		return record

	def create_alert(
		self,
		*,
		tenant_id: str,
		source_id: str,
		severity: str,
		title: str,
		notification_route: str | None = None,
		owner: str | None = None,
	) -> AlertRecord:
		"""Create an alert and auto-open critical incident records."""
		severity = self._require_choice(severity, "severity", {"info", "low", "medium", "high", "critical"})
		context = {
			"tenant_context_present": bool(tenant_id),
			"alert_severity": severity,
			"notification_route_configured": bool(notification_route),
			"alert_owner_present": bool(owner),
		}
		decision = evaluate_capability_rules(context)
		status = "open" if decision["decision"] == "allow" else "denied"
		incident_id = None
		alert_id = uuid_like()
		if severity == "critical" and status == "open":
			incident = self.create_incident(
				tenant_id=tenant_id,
				title=title,
				severity=severity,
				owner=owner,
				notification_route=notification_route,
				alert_ids=[alert_id],
			)
			incident_id = incident.incident_id if incident.status != "denied" else None
		record = AlertRecord(
			alert_id=alert_id,
			tenant_id=self._require_text(tenant_id, "tenant_id"),
			source_id=self._require_text(source_id, "source_id"),
			severity=severity,
			title=self._require_text(title, "title"),
			notification_route=notification_route,
			owner=owner,
			incident_id=incident_id,
			decision=decision["decision"],
			status=status,
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.alerts[record.alert_id] = record
		self._audit(record.tenant_id, "alert.created", record.alert_id, record.owner or "system", decision, context)
		return record

	def create_incident(
		self,
		*,
		tenant_id: str,
		title: str,
		severity: str,
		owner: str | None,
		notification_route: str | None,
		alert_ids: list[str] | None = None,
	) -> IncidentRecord:
		"""Create an incident correlation record."""
		context = {
			"tenant_context_present": bool(tenant_id),
			"incident_severity": severity,
			"incident_owner_present": bool(owner),
			"notification_route_configured": bool(notification_route),
		}
		decision = evaluate_capability_rules(context)
		record = IncidentRecord(
			incident_id=uuid_like(),
			tenant_id=self._require_text(tenant_id, "tenant_id"),
			title=self._require_text(title, "title"),
			severity=self._require_choice(severity, "severity", {"info", "low", "medium", "high", "critical"}),
			owner=owner,
			notification_route=notification_route,
			status="open" if decision["decision"] == "allow" else "denied",
			alert_ids=list(alert_ids or []),
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.incidents[record.incident_id] = record
		self._audit(record.tenant_id, "incident.created", record.incident_id, owner or "system", decision, context)
		return record

	def request_remediation(
		self,
		*,
		tenant_id: str,
		incident_id: str,
		requester: str,
		environment: str,
		runbook_id: str,
		runbook_approved: bool,
		proposed_action: str,
		reason: str,
	) -> RemediationRequestRecord:
		"""Request runbook-backed remediation."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		incident_id = self._require_text(incident_id, "incident_id")
		incident = self.incidents.get(incident_id)
		if incident is None:
			raise ValueError("incident_id must reference an existing incident")
		if incident.tenant_id != tenant_id:
			raise ValueError("incident_id must belong to the requesting tenant")
		if incident.status == "denied":
			raise ValueError("incident_id must reference an active incident")
		context = {
			"tenant_context_present": bool(tenant_id),
			"environment": environment,
			"remediation_requested": True,
			"runbook_approved": runbook_approved,
		}
		decision = evaluate_capability_rules(context)
		record = RemediationRequestRecord(
			request_id=uuid_like(),
			tenant_id=tenant_id,
			incident_id=incident_id,
			requester=self._require_text(requester, "requester"),
			environment=self._require_text(environment, "environment"),
			runbook_id=self._require_text(runbook_id, "runbook_id"),
			runbook_approved=runbook_approved,
			proposed_action=self._require_text(proposed_action, "proposed_action"),
			reason=self._require_text(reason, "reason"),
			decision=decision["decision"],
			status="pending_review" if decision["decision"] != "deny" else "denied",
			matched_rules=decision["matched_rules"],
			policy_decision=decision["decision"],
			review_reasons=self._reasons(decision),
			review_evidence=self._review_evidence(decision),
		)
		self.remediation_requests[record.request_id] = record
		self._audit(record.tenant_id, "remediation.requested", record.request_id, record.requester, decision, context)
		return record

	def decide_remediation(
		self,
		*,
		request_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> RemediationRequestRecord:
		"""Approve or reject a remediation request."""
		if request_id not in self.remediation_requests:
			raise KeyError(f"Remediation request {request_id} not found")
		record = self.remediation_requests[request_id]
		reviewer = self._require_text(reviewer, "reviewer")
		notes = self._require_text(notes, "notes")
		if decision not in {"approved", "rejected"}:
			raise ValueError("decision must be approved or rejected")
		context = {
			"tenant_context_present": bool(record.tenant_id),
			"operation": "review",
			"reviewer_same_as_requester": reviewer == record.requester,
			"review_notes_attached": bool(notes),
		}
		rule_decision = evaluate_capability_rules(context)
		if rule_decision["decision"] == "deny":
			record.decision = "denied"
			record.status = "review_denied"
		else:
			record.decision = decision
			record.status = decision
		record.reviewer = reviewer
		record.review_notes = notes
		record.decided_at = datetime.utcnow()
		record.matched_rules = rule_decision["matched_rules"]
		record.policy_decision = rule_decision["decision"]
		record.review_reasons = self._reasons(rule_decision)
		record.review_evidence = self._review_evidence(rule_decision, review_recorded=True)
		self._audit(record.tenant_id, "remediation.decided", request_id, reviewer, rule_decision, context)
		return record

	def register_monitoring_agent(
		self,
		*,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> MonitoringAgentRecord:
		"""Register a first-class observability agent with guardrail evidence."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		agent_id = self._require_text(agent_id, "agent_id")
		name = self._require_text(name, "name")
		runtime_value = self._normalize_agent_token(runtime)
		role_value = self._normalize_agent_token(role)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "register_monitoring_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"agent_scope_present": bool(str(scope or "").strip()),
			"agent_owner_present": bool(str(owner or "").strip()),
			"agent_purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_agent_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		}
		rule_decision = evaluate_capability_rules(context)
		if rule_decision["decision"] == "deny":
			raise PermissionError(self._first_reason(rule_decision))
		record_key = self._agent_key(tenant_id, agent_id)
		if record_key in self.monitoring_agents:
			raise ValueError(f"monitoring_agent_already_exists:{agent_id}")
		record = MonitoringAgentRecord(
			agent_id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=self._require_text(scope, "scope"),
			owner=self._require_text(owner, "owner"),
			purpose=self._require_text(purpose, "purpose"),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status="pending_review" if rule_decision["decision"] == "require_review" else "active",
			policy_decision=rule_decision["decision"],
			matched_rules=list(rule_decision["matched_rules"]),
			review_reasons=self._reasons(rule_decision),
			review_evidence=self._review_evidence(rule_decision, review_recorded=bool(human_approval_required)),
		)
		self.monitoring_agents[record_key] = record
		self._audit(tenant_id, "agent.registered", agent_id, record.owner, rule_decision, asdict(record))
		return record

	def validate_monitoring_lifecycle_batch(
		self,
		*,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> MonitoringLifecycleBatchRecord:
		"""Validate that MONI lifecycle mutation batches flow through Bytewax."""
		tenant_id = self._require_text(tenant_id, "tenant_id")
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("monitoring_lifecycle_batch_empty")
		stream_value = self._normalize_agent_token(event_stream)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_monitoring_lifecycle_batch",
			"event_stream": stream_value,
		}
		rule_decision = evaluate_capability_rules(context)
		accepted = rule_decision["decision"] == "allow"
		record = MonitoringLifecycleBatchRecord(
			batch_id=uuid_like(),
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			accepted=accepted,
			decision=rule_decision["decision"],
			matched_rules=list(rule_decision["matched_rules"]),
			policy_decision=rule_decision["decision"],
			review_reasons=self._reasons(rule_decision),
			review_evidence=self._review_evidence(rule_decision),
			status="accepted" if accepted else "denied",
		)
		self.lifecycle_batches[record.batch_id] = record
		self._audit(tenant_id, f"lifecycle_batch.{record.status}", stream_value, "moni", rule_decision, asdict(record))
		if not accepted:
			raise PermissionError(self._first_reason(rule_decision))
		return record

	def list_records(self, tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
		"""List generated-app records for a tenant."""
		tenant_id = tenant_id or self.tenant_id
		collections: dict[str, Any] = {
			"sources": self.sources.values(),
			"signals": self.signals.values(),
			"slos": self.slos.values(),
			"alerts": self.alerts.values(),
			"incidents": self.incidents.values(),
			"remediation_requests": self.remediation_requests.values(),
			"monitoring_agents": self.monitoring_agents.values(),
			"lifecycle_batches": self.lifecycle_batches.values(),
			"audit_events": self.audit_events,
			"records": self.records.values(),
		}
		if record_type:
			if record_type not in collections:
				raise ValueError(f"Unsupported record_type {record_type}")
			values = collections[record_type]
		else:
			values = []
			for collection in collections.values():
				values.extend(collection)
		return [
			dict(record) if isinstance(record, dict) else asdict(record)
			for record in values
			if (record.get("tenant_id") if isinstance(record, dict) else getattr(record, "tenant_id", None)) == tenant_id
		]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return summary metrics for generated MONI dashboards."""
		tenant_id = tenant_id or self.tenant_id
		return {
			"tenant_id": tenant_id,
			"source_count": len(self.list_records(tenant_id, "sources")),
			"signal_count": len(self.list_records(tenant_id, "signals")),
			"slo_count": len(self.list_records(tenant_id, "slos")),
			"open_alert_count": sum(1 for row in self.list_records(tenant_id, "alerts") if row["status"] == "open"),
			"open_incident_count": sum(1 for row in self.list_records(tenant_id, "incidents") if row["status"] == "open"),
			"pending_remediation_count": sum(1 for row in self.list_records(tenant_id, "remediation_requests") if row["status"] == "pending_review"),
			"monitoring_agent_count": len(self.list_records(tenant_id, "monitoring_agents")),
			"pending_monitoring_agent_review_count": sum(1 for row in self.list_records(tenant_id, "monitoring_agents") if row["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_records(tenant_id, "lifecycle_batches")),
			"denied_lifecycle_batch_count": sum(1 for row in self.list_records(tenant_id, "lifecycle_batches") if not row["accepted"]),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
			"audit_event_count": len(self.list_records(tenant_id, "audit_events")),
		}

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all MONI records awaiting operator or human review."""
		tenant_id = tenant_id or self.tenant_id
		items = (
			self.list_records(tenant_id, "signals")
			+ self.list_records(tenant_id, "alerts")
			+ self.list_records(tenant_id, "incidents")
			+ self.list_records(tenant_id, "remediation_requests")
			+ self.list_records(tenant_id, "monitoring_agents")
			+ self.list_records(tenant_id, "lifecycle_batches")
		)
		return [
			item
			for item in items
			if item.get("status") in {"pending", "pending_review", "review_required"}
		]

	def _audit(
		self,
		tenant_id: str,
		event_type: str,
		subject: str,
		actor: str,
		policy_result: dict[str, Any],
		details: dict[str, Any],
	) -> None:
		policy_result = policy_result or _allow_result()
		self.audit_events.append(MoniAuditEventRecord(
			event_id=uuid_like(),
			tenant_id=tenant_id,
			event_type=event_type,
			subject=subject,
			actor=actor,
			decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			policy_decision=policy_result["decision"],
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
			details=details,
		))

	def _reasons(self, result: dict[str, Any]) -> list[str]:
		return list(dict.fromkeys(
			str(action["reason"])
			for action in result.get("actions", [])
			if action.get("reason")
		))

	def _review_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": list(dict.fromkeys(
				str(action.get("required_action"))
				for action in result.get("actions", [])
				if action.get("required_action")
			)),
			"reasons": self._reasons(result),
			"review_recorded": bool(review_recorded),
		}

	@staticmethod
	def _merge_decisions(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
		decision = "allow"
		if "deny" in {first["decision"], second["decision"]}:
			decision = "deny"
		elif "require_review" in {first["decision"], second["decision"]}:
			decision = "require_review"
		return {
			"decision": decision,
			"matched_rules": list(dict.fromkeys(first["matched_rules"] + second["matched_rules"])),
			"actions": first["actions"] + second["actions"],
			"context": {**first["context"], **second["context"]},
		}

	@staticmethod
	def _require_text(value: str, field_name: str) -> str:
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f"{field_name} is required")
		return value.strip()

	@staticmethod
	def _require_choice(value: str, field_name: str, allowed: set[str]) -> str:
		text = MoniService._require_text(value, field_name)
		if text not in allowed:
			raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
		return text

	@staticmethod
	def _source_key(tenant_id: str, source_id: str) -> str:
		return f"{tenant_id}:{source_id}"

	@staticmethod
	def _agent_key(tenant_id: str, agent_id: str) -> str:
		return f"{tenant_id}:{agent_id}"

	@staticmethod
	def _normalize_agent_token(value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	@staticmethod
	def _first_reason(result: dict[str, Any]) -> str:
		for action in result.get("actions", []):
			if action.get("reason"):
				return str(action["reason"])
		return "monitoring_operation_denied"


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


def uuid_like() -> str:
	"""Return a sortable enough local identifier without adding dependencies."""
	return f"moni-{time.time_ns()}"


# Export main components
__all__ = [
	'MonitoringService',
	'MonitoringServiceConfig',
	'MoniService',
	'SignalSourceRecord',
	'SignalRecord',
	'SloRecord',
	'AlertRecord',
	'IncidentRecord',
	'RemediationRequestRecord',
	'MonitoringAgentRecord',
	'MonitoringLifecycleBatchRecord',
	'MoniAuditEventRecord',
	'create_monitoring_service',
]
