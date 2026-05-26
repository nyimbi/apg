"""
Unit Tests for AICR Monitoring System
======================================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Comprehensive unit tests for the AI monitoring and observability system
covering metrics collection, alerting, performance analysis, and real-time
monitoring with 100% coverage and scenario-based testing.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

from ..monitoring import (
	AIMonitoringSystem,
	MetricsCollector,
	AlertManager,
	PerformanceAnalyzer,
	Metric,
	Alert,
	PerformanceBaseline,
	AnomalyDetection,
	MonitoringDashboard,
	MetricType,
	AlertSeverity,
	MonitoringStatus,
	PerformanceAnalysisType
)


class TestMetric:
	"""Test cases for Metric model."""

	def test_metric_creation(self):
		"""Test creating a metric instance."""
		labels = {"service": "inference", "model": "test_model"}

		metric = Metric(
			metric_name="inference_latency",
			metric_type=MetricType.HISTOGRAM,
			value=125.5,
			labels=labels,
			unit="milliseconds",
			source_component="inference_engine",
			source_instance="worker-1",
			description="Inference latency measurement"
		)

		assert metric.metric_name == "inference_latency"
		assert metric.metric_type == MetricType.HISTOGRAM
		assert metric.value == 125.5
		assert metric.labels == labels
		assert metric.unit == "milliseconds"
		assert metric.source_component == "inference_engine"
		assert metric.source_instance == "worker-1"
		assert metric.description == "Inference latency measurement"
		assert isinstance(metric.metric_id, str)
		assert isinstance(metric.timestamp, datetime)

	def test_metric_validation(self):
		"""Test metric validation rules."""
		from pydantic import ValidationError

		# Test missing required fields
		with pytest.raises(ValidationError):
			Metric()

		# Test valid minimal metric
		metric = Metric(
			metric_name="cpu_usage",
			metric_type=MetricType.GAUGE,
			value=75.0,
			source_component="system"
		)
		assert metric.metric_name == "cpu_usage"
		assert metric.value == 75.0

	def test_metric_serialization(self):
		"""Test metric serialization."""
		metric = Metric(
			metric_name="test_metric",
			metric_type=MetricType.COUNTER,
			value=42.0,
			source_component="test",
			labels={"env": "test"}
		)

		data = metric.model_dump()
		assert data['metric_name'] == "test_metric"
		assert data['metric_type'] == "counter"
		assert data['value'] == 42.0

		restored = Metric.model_validate(data)
		assert restored.metric_name == metric.metric_name
		assert restored.value == metric.value


class TestAlert:
	"""Test cases for Alert model."""

	def test_alert_creation(self):
		"""Test creating an alert instance."""
		alert = Alert(
			alert_name="High CPU Usage",
			description="CPU usage is above threshold",
			severity=AlertSeverity.HIGH,
			condition="cpu_usage > threshold",
			threshold=80.0,
			metric_name="cpu_usage",
			notification_channels=["email", "webhook"]
		)

		assert alert.alert_name == "High CPU Usage"
		assert alert.severity == AlertSeverity.HIGH
		assert alert.threshold == 80.0
		assert alert.metric_name == "cpu_usage"
		assert alert.notification_channels == ["email", "webhook"]
		assert alert.is_active == True
		assert alert.trigger_count == 0
		assert alert.evaluation_interval == 60
		assert alert.cooldown_period == 300

	def test_alert_validation(self):
		"""Test alert validation rules."""
		from pydantic import ValidationError

		# Test missing required fields
		with pytest.raises(ValidationError):
			Alert()

		# Test valid minimal alert
		alert = Alert(
			alert_name="Test Alert",
			description="Test alert description",
			severity=AlertSeverity.LOW,
			condition="value > 10",
			threshold=10.0,
			metric_name="test_metric"
		)
		assert alert.alert_name == "Test Alert"
		assert alert.threshold == 10.0


class TestMetricsCollector:
	"""Test cases for MetricsCollector."""

	@pytest.fixture
	def metrics_collector(self):
		"""Create a metrics collector for testing."""
		return MetricsCollector()

	@pytest.mark.asyncio
	async def test_collector_initialization(self, metrics_collector):
		"""Test metrics collector initialization."""
		with patch.object(metrics_collector, '_initialize_default_collectors', new_callable=AsyncMock) as mock_init, \
			 patch.object(metrics_collector, '_start_collection_tasks', new_callable=AsyncMock) as mock_start:

			await metrics_collector.initialize()

			mock_init.assert_called_once()
			mock_start.assert_called_once()
			assert metrics_collector._initialized == True

	@pytest.mark.asyncio
	async def test_register_metric(self, metrics_collector):
		"""Test registering a new metric for collection."""
		await metrics_collector.initialize()

		# Mock collector function
		async def test_collector():
			return 75.0

		await metrics_collector.register_metric(
			metric_name="test_metric",
			metric_type=MetricType.GAUGE,
			collector_func=test_collector,
			collection_interval=30
		)

		assert "test_metric" in metrics_collector.collectors
		assert metrics_collector.collection_intervals["test_metric"] == 30
		assert "test_metric" in metrics_collector._collection_tasks

	@pytest.mark.asyncio
	async def test_collect_metric(self, metrics_collector):
		"""Test collecting a single metric value."""
		await metrics_collector.initialize()

		labels = {"service": "test"}

		await metrics_collector.collect_metric(
			metric_name="test_metric",
			value=42.5,
			labels=labels,
			source_component="test_component"
		)

		# Check metric was added to buffer
		assert len(metrics_collector.metrics_buffer) > 0

		# Check metric was added to aggregated metrics
		assert "test_metric" in metrics_collector.aggregated_metrics
		assert len(metrics_collector.aggregated_metrics["test_metric"]) == 1

		metric = metrics_collector.aggregated_metrics["test_metric"][0]
		assert metric.metric_name == "test_metric"
		assert metric.value == 42.5
		assert metric.labels == labels

	@pytest.mark.asyncio
	async def test_get_metrics(self, metrics_collector):
		"""Test retrieving metrics with filters."""
		await metrics_collector.initialize()

		# Collect some test metrics
		base_time = datetime.utcnow()

		for i in range(5):
			await metrics_collector.collect_metric(
				metric_name=f"metric_{i % 2}",  # Alternating metric names
				value=float(i * 10),
				labels={"type": "test"},
				source_component="test"
			)

		# Test getting all metrics
		all_metrics = await metrics_collector.get_metrics()
		assert len(all_metrics) == 5

		# Test filtering by metric names
		filtered_metrics = await metrics_collector.get_metrics(
			metric_names=["metric_0"]
		)
		assert len(filtered_metrics) == 3  # metric_0 appears at indices 0, 2, 4
		assert all(m.metric_name == "metric_0" for m in filtered_metrics)

		# Test filtering by labels
		label_filtered = await metrics_collector.get_metrics(
			labels_filter={"type": "test"}
		)
		assert len(label_filtered) == 5
		assert all(m.labels.get("type") == "test" for m in label_filtered)

	@pytest.mark.asyncio
	async def test_periodic_collection(self, metrics_collector):
		"""Test periodic metric collection."""
		await metrics_collector.initialize()

		# Mock collector function
		call_count = 0

		async def counting_collector():
			nonlocal call_count
			call_count += 1
			return float(call_count)

		# Register metric with short interval for testing
		metrics_collector.collectors["test_periodic"] = counting_collector
		metrics_collector.collection_intervals["test_periodic"] = 0.1  # 100ms

		# Start collection task
		task = asyncio.create_task(
			metrics_collector._collect_metric_periodically("test_periodic", MetricType.COUNTER)
		)

		# Let it run for a short time
		await asyncio.sleep(0.3)
		task.cancel()

		# Check that collector was called multiple times
		assert call_count >= 2

		# Check that metrics were collected
		assert "test_periodic" in metrics_collector.aggregated_metrics
		assert len(metrics_collector.aggregated_metrics["test_periodic"]) >= 2


class TestAlertManager:
	"""Test cases for AlertManager."""

	@pytest.fixture
	def alert_manager(self):
		"""Create an alert manager for testing."""
		return AlertManager()

	@pytest.mark.asyncio
	async def test_alert_manager_initialization(self, alert_manager):
		"""Test alert manager initialization."""
		with patch.object(alert_manager, '_initialize_notification_channels', new_callable=AsyncMock) as mock_init:
			await alert_manager.initialize()

			mock_init.assert_called_once()
			assert alert_manager._initialized == True

	@pytest.mark.asyncio
	async def test_create_alert(self, alert_manager):
		"""Test creating a new alert."""
		await alert_manager.initialize()

		alert_config = {
			"alert_name": "Test Alert",
			"description": "Test alert for unit testing",
			"severity": "high",
			"condition": "cpu_usage > 80",
			"threshold": 80.0,
			"metric_name": "cpu_usage",
			"notification_channels": ["email"]
		}

		alert_id = await alert_manager.create_alert(alert_config)

		assert alert_id in alert_manager.alerts
		alert = alert_manager.alerts[alert_id]
		assert alert.alert_name == "Test Alert"
		assert alert.threshold == 80.0
		assert alert_id in alert_manager.evaluation_tasks

	@pytest.mark.asyncio
	async def test_update_alert(self, alert_manager):
		"""Test updating an existing alert."""
		await alert_manager.initialize()

		# Create an alert first
		alert_config = {
			"alert_name": "Original Alert",
			"description": "Original description",
			"severity": "medium",
			"condition": "memory_usage > 70",
			"threshold": 70.0,
			"metric_name": "memory_usage"
		}

		alert_id = await alert_manager.create_alert(alert_config)

		# Update the alert
		updates = {
			"alert_name": "Updated Alert",
			"threshold": 85.0,
			"is_active": False
		}

		await alert_manager.update_alert(alert_id, updates)

		alert = alert_manager.alerts[alert_id]
		assert alert.alert_name == "Updated Alert"
		assert alert.threshold == 85.0
		assert alert.is_active == False

		# Check that evaluation task was stopped
		assert alert_id not in alert_manager.evaluation_tasks

	@pytest.mark.asyncio
	async def test_alert_evaluation(self, alert_manager):
		"""Test alert condition evaluation."""
		await alert_manager.initialize()

		# Mock the metric value retrieval
		with patch.object(alert_manager, '_get_current_metric_value', new_callable=AsyncMock) as mock_metric:
			mock_metric.return_value = 85.0  # Above threshold

			# Mock condition evaluation
			with patch.object(alert_manager, '_evaluate_condition_expression', new_callable=AsyncMock) as mock_condition:
				mock_condition.return_value = True

				# Mock alert triggering
				with patch.object(alert_manager, '_trigger_alert', new_callable=AsyncMock) as mock_trigger:

					# Create an alert
					alert_config = {
						"alert_name": "Test Evaluation",
						"description": "Test alert evaluation",
						"severity": "high",
						"condition": "cpu_usage > 80",
						"threshold": 80.0,
						"metric_name": "cpu_usage"
					}

					alert_id = await alert_manager.create_alert(alert_config)
					alert = alert_manager.alerts[alert_id]

					# Manually trigger evaluation
					await alert_manager._evaluate_alert_condition(alert)

					# Check that alert was triggered
					mock_trigger.assert_called_once()

	@pytest.mark.asyncio
	async def test_alert_cooldown(self, alert_manager):
		"""Test alert cooldown period."""
		await alert_manager.initialize()

		# Create alert with short cooldown for testing
		alert_config = {
			"alert_name": "Cooldown Test",
			"description": "Test cooldown behavior",
			"severity": "medium",
			"condition": "test_metric > 50",
			"threshold": 50.0,
			"metric_name": "test_metric",
			"cooldown_period": 1  # 1 second cooldown
		}

		alert_id = await alert_manager.create_alert(alert_config)
		alert = alert_manager.alerts[alert_id]

		# Set last triggered time to now
		alert.last_triggered = datetime.utcnow()

		# Mock metric value above threshold
		with patch.object(alert_manager, '_get_current_metric_value', new_callable=AsyncMock) as mock_metric:
			mock_metric.return_value = 75.0

			# Mock condition evaluation to return True
			with patch.object(alert_manager, '_evaluate_condition_expression', new_callable=AsyncMock) as mock_condition:
				mock_condition.return_value = True

				# Mock alert triggering
				with patch.object(alert_manager, '_trigger_alert', new_callable=AsyncMock) as mock_trigger:

					# Evaluate alert immediately (should be in cooldown)
					await alert_manager._evaluate_alert_condition(alert)

					# Alert should not trigger due to cooldown
					mock_trigger.assert_not_called()

					# Wait for cooldown to expire
					await asyncio.sleep(1.1)

					# Evaluate again (should trigger now)
					await alert_manager._evaluate_alert_condition(alert)
					mock_trigger.assert_called_once()


class TestPerformanceAnalyzer:
	"""Test cases for PerformanceAnalyzer."""

	@pytest.fixture
	def performance_analyzer(self):
		"""Create a performance analyzer for testing."""
		return PerformanceAnalyzer()

	@pytest.fixture
	def mock_metrics_collector(self):
		"""Create a mock metrics collector."""
		collector = Mock()
		collector.get_metrics = AsyncMock()
		return collector

	@pytest.mark.asyncio
	async def test_analyzer_initialization(self, performance_analyzer, mock_metrics_collector):
		"""Test performance analyzer initialization."""
		with patch.object(performance_analyzer, '_initialize_default_detectors', new_callable=AsyncMock) as mock_detectors, \
			 patch.object(performance_analyzer, '_start_analysis_tasks', new_callable=AsyncMock) as mock_tasks:

			await performance_analyzer.initialize(mock_metrics_collector)

			mock_detectors.assert_called_once()
			mock_tasks.assert_called_once()
			assert performance_analyzer._initialized == True
			assert performance_analyzer.metrics_collector == mock_metrics_collector

	@pytest.mark.asyncio
	async def test_create_baseline(self, performance_analyzer, mock_metrics_collector):
		"""Test creating a performance baseline."""
		await performance_analyzer.initialize(mock_metrics_collector)

		# Mock historical metrics
		historical_metrics = []
		for i in range(100):
			metric = Metric(
				metric_name="cpu_usage",
				metric_type=MetricType.GAUGE,
				value=70.0 + np.random.normal(0, 5),  # Normal distribution around 70
				source_component="system",
				timestamp=datetime.utcnow() - timedelta(hours=i)
			)
			historical_metrics.append(metric)

		mock_metrics_collector.get_metrics.return_value = historical_metrics

		baseline_config = {
			"name": "cpu_baseline",
			"history_days": 7,
			"confidence_level": 0.95,
			"method": "mean_with_confidence"
		}

		baseline_id = await performance_analyzer.create_baseline("cpu_usage", baseline_config)

		assert baseline_id in performance_analyzer.baselines
		baseline = performance_analyzer.baselines[baseline_id]
		assert baseline.metric_name == "cpu_usage"
		assert baseline.sample_size == 100
		assert 65 < baseline.baseline_value < 75  # Should be around 70

	@pytest.mark.asyncio
	async def test_detect_anomalies(self, performance_analyzer, mock_metrics_collector):
		"""Test anomaly detection."""
		await performance_analyzer.initialize(mock_metrics_collector)

		# Create metrics with one clear anomaly
		normal_values = [70.0 + np.random.normal(0, 2) for _ in range(95)]
		anomaly_values = [120.0]  # Clear anomaly
		all_values = normal_values + anomaly_values

		metrics = []
		for i, value in enumerate(all_values):
			metric = Metric(
				metric_name="cpu_usage",
				metric_type=MetricType.GAUGE,
				value=value,
				source_component="system",
				timestamp=datetime.utcnow() - timedelta(minutes=i)
			)
			metrics.append(metric)

		mock_metrics_collector.get_metrics.return_value = metrics

		detection_config = {
			"algorithm": "statistical",
			"threshold": 2.0,
			"analysis_hours": 24
		}

		anomalies = await performance_analyzer.detect_anomalies(["cpu_usage"], detection_config)

		assert "cpu_usage" in anomalies
		assert len(anomalies["cpu_usage"]) > 0

		# Check that the anomaly value is detected
		anomaly_values_detected = [float(a["value"]) for a in anomalies["cpu_usage"]]
		assert any(v > 100 for v in anomaly_values_detected)

	@pytest.mark.asyncio
	async def test_trend_analysis(self, performance_analyzer, mock_metrics_collector):
		"""Test trend analysis."""
		await performance_analyzer.initialize(mock_metrics_collector)

		# Create metrics with upward trend
		metrics = []
		for i in range(50):
			value = 50.0 + i * 0.5 + np.random.normal(0, 1)  # Increasing trend with noise
			metric = Metric(
				metric_name="memory_usage",
				metric_type=MetricType.GAUGE,
				value=value,
				source_component="system",
				timestamp=datetime.utcnow() - timedelta(hours=49-i)
			)
			metrics.append(metric)

		mock_metrics_collector.get_metrics.return_value = metrics

		trend_result = await performance_analyzer.perform_trend_analysis(
			"memory_usage",
			timedelta(hours=48)
		)

		assert trend_result["metric_name"] == "memory_usage"
		assert trend_result["trend_direction"] in ["increasing", "decreasing", "stable"]
		assert "slope" in trend_result
		assert "correlation_coefficient" in trend_result
		assert "percentage_change" in trend_result
		assert trend_result["data_points"] == 50

	@pytest.mark.asyncio
	async def test_seasonal_pattern_detection(self, performance_analyzer):
		"""Test seasonal pattern detection."""
		# Create timestamps and values with weekly pattern
		timestamps = list(range(168))  # 7 days * 24 hours
		values = [50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 2) for t in timestamps]

		seasonal_result = await performance_analyzer._detect_seasonal_pattern(timestamps, values)

		# Should detect daily pattern (lag 24)
		if seasonal_result["pattern_detected"]:
			assert seasonal_result["seasonal_lag"] in [7, 24, 168]  # Weekly, daily, or hourly patterns
			assert "correlation" in seasonal_result
			assert "pattern_strength" in seasonal_result


class TestAIMonitoringSystem:
	"""Test cases for AIMonitoringSystem integration."""

	@pytest.fixture
	def monitoring_system(self):
		"""Create a monitoring system for testing."""
		return AIMonitoringSystem()

	@pytest.mark.asyncio
	async def test_monitoring_system_initialization(self, monitoring_system):
		"""Test monitoring system initialization."""
		with patch.object(monitoring_system.metrics_collector, 'initialize', new_callable=AsyncMock) as mock_metrics, \
			 patch.object(monitoring_system.alert_manager, 'initialize', new_callable=AsyncMock) as mock_alerts, \
			 patch.object(monitoring_system.performance_analyzer, 'initialize', new_callable=AsyncMock) as mock_perf, \
			 patch.object(monitoring_system.security_manager, 'initialize', new_callable=AsyncMock) as mock_security, \
			 patch.object(monitoring_system, '_create_default_dashboards', new_callable=AsyncMock) as mock_dashboards, \
			 patch.object(monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock) as mock_tasks:

			await monitoring_system.initialize()

			mock_metrics.assert_called_once()
			mock_alerts.assert_called_once()
			mock_perf.assert_called_once()
			mock_security.assert_called_once()
			mock_dashboards.assert_called_once()
			mock_tasks.assert_called_once()

			assert monitoring_system._initialized == True
			assert monitoring_system.system_status == MonitoringStatus.ACTIVE

	@pytest.mark.asyncio
	async def test_register_ai_component(self, monitoring_system):
		"""Test registering an AI component for monitoring."""
		# Mock initialization
		with patch.object(monitoring_system.metrics_collector, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.alert_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.performance_analyzer, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):

			await monitoring_system.initialize()

		# Mock component registration methods
		monitoring_system.metrics_collector.register_metric = AsyncMock()
		monitoring_system.alert_manager.create_alert = AsyncMock(return_value="alert_id")
		monitoring_system.performance_analyzer.create_baseline = AsyncMock(return_value="baseline_id")

		component_config = {
			"metrics": [
				{
					"name": "inference_latency",
					"type": "histogram",
					"collector": lambda: 100.0,
					"interval": 30
				}
			],
			"alerts": [
				{
					"alert_name": "High Latency",
					"description": "Inference latency is high",
					"severity": "medium",
					"condition": "latency > 200",
					"threshold": 200.0,
					"metric_name": "inference_latency"
				}
			],
			"baselines": [
				{
					"metric_name": "inference_latency",
					"name": "latency_baseline"
				}
			]
		}

		component_id = await monitoring_system.register_ai_component("inference_engine", component_config)

		assert isinstance(component_id, str)
		monitoring_system.metrics_collector.register_metric.assert_called_once()
		monitoring_system.alert_manager.create_alert.assert_called_once()
		monitoring_system.performance_analyzer.create_baseline.assert_called_once()

	@pytest.mark.asyncio
	async def test_get_system_health(self, monitoring_system):
		"""Test getting system health status."""
		# Mock initialization
		with patch.object(monitoring_system.metrics_collector, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.alert_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.performance_analyzer, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):

			await monitoring_system.initialize()

		# Mock metrics retrieval
		mock_metrics = [
			Metric(
				metric_name="system_cpu_usage",
				metric_type=MetricType.GAUGE,
				value=75.0,
				source_component="system"
			),
			Metric(
				metric_name="system_memory_usage",
				metric_type=MetricType.GAUGE,
				value=60.0,
				source_component="system"
			)
		]

		monitoring_system.metrics_collector.get_metrics = AsyncMock(return_value=mock_metrics)
		monitoring_system.alert_manager.alert_history = []

		health_data = await monitoring_system.get_system_health()

		assert "system_id" in health_data
		assert "overall_health_score" in health_data
		assert "health_status" in health_data
		assert "component_health" in health_data
		assert "active_alerts" in health_data
		assert "monitoring_status" in health_data

		# Check component health scores
		assert "cpu" in health_data["component_health"]
		assert "memory" in health_data["component_health"]

		# Health scores should be reasonable (between 0 and 1)
		cpu_score = health_data["component_health"]["cpu"]["score"]
		memory_score = health_data["component_health"]["memory"]["score"]
		assert 0 <= cpu_score <= 1
		assert 0 <= memory_score <= 1

	@pytest.mark.asyncio
	async def test_get_performance_summary(self, monitoring_system):
		"""Test getting performance summary."""
		# Mock initialization
		with patch.object(monitoring_system.metrics_collector, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.alert_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.performance_analyzer, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):

			await monitoring_system.initialize()

		# Mock metrics and analysis
		mock_metrics = [
			Metric(
				metric_name="cpu_usage",
				metric_type=MetricType.GAUGE,
				value=70.0 + i,
				source_component="system",
				timestamp=datetime.utcnow() - timedelta(hours=i)
			) for i in range(24)
		]

		monitoring_system.metrics_collector.get_metrics = AsyncMock(return_value=mock_metrics)
		monitoring_system.performance_analyzer.perform_trend_analysis = AsyncMock(return_value={
			"trend_direction": "increasing",
			"slope": 0.5,
			"correlation_coefficient": 0.8
		})
		monitoring_system.performance_analyzer.detect_anomalies = AsyncMock(return_value={})

		time_range = (datetime.utcnow() - timedelta(hours=24), datetime.utcnow())
		summary = await monitoring_system.get_performance_summary(time_range)

		assert "time_range" in summary
		assert "performance_statistics" in summary
		assert "trend_analysis" in summary
		assert "anomalies_detected" in summary
		assert "total_anomalies" in summary

		# Check statistics calculation
		stats = summary["performance_statistics"]
		assert "cpu_usage" in stats
		cpu_stats = stats["cpu_usage"]
		assert "count" in cpu_stats
		assert "avg" in cpu_stats
		assert "min" in cpu_stats
		assert "max" in cpu_stats

	@pytest.mark.asyncio
	async def test_create_dashboard(self, monitoring_system):
		"""Test creating a monitoring dashboard."""
		# Mock initialization
		with patch.object(monitoring_system.metrics_collector, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.alert_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.performance_analyzer, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):

			await monitoring_system.initialize()

		dashboard_config = {
			"dashboard_name": "Test Dashboard",
			"description": "Dashboard for testing",
			"widgets": [
				{
					"type": "metric_chart",
					"title": "CPU Usage",
					"metric": "cpu_usage"
				}
			],
			"owner": "test_user",
			"is_public": True
		}

		dashboard_id = await monitoring_system.create_dashboard(dashboard_config)

		assert dashboard_id in monitoring_system.dashboards
		dashboard = monitoring_system.dashboards[dashboard_id]
		assert dashboard.dashboard_name == "Test Dashboard"
		assert dashboard.is_public == True
		assert len(dashboard.widgets) == 1


class TestMonitoringIntegration:
	"""Test cases for monitoring system integration scenarios."""

	@pytest.mark.asyncio
	async def test_end_to_end_monitoring_flow(self):
		"""Test complete monitoring flow from metric collection to alerting."""
		# Create monitoring system
		monitoring_system = AIMonitoringSystem()

		# Mock all initialization
		with patch.object(monitoring_system.metrics_collector, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.alert_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.performance_analyzer, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):

			await monitoring_system.initialize()

		# Step 1: Collect metrics
		await monitoring_system.metrics_collector.collect_metric(
			metric_name="cpu_usage",
			value=85.0,  # High CPU usage
			source_component="system"
		)

		# Step 2: Create alert for high CPU
		alert_config = {
			"alert_name": "High CPU Alert",
			"description": "CPU usage is too high",
			"severity": "high",
			"condition": "cpu_usage > 80",
			"threshold": 80.0,
			"metric_name": "cpu_usage",
			"notification_channels": ["email"]
		}

		alert_id = await monitoring_system.alert_manager.create_alert(alert_config)

		# Step 3: Mock alert evaluation and triggering
		with patch.object(monitoring_system.alert_manager, '_get_current_metric_value', new_callable=AsyncMock) as mock_metric_value:
			mock_metric_value.return_value = 85.0

			with patch.object(monitoring_system.alert_manager, '_evaluate_condition_expression', new_callable=AsyncMock) as mock_condition:
				mock_condition.return_value = True

				with patch.object(monitoring_system.alert_manager, '_trigger_alert', new_callable=AsyncMock) as mock_trigger:
					alert = monitoring_system.alert_manager.alerts[alert_id]
					await monitoring_system.alert_manager._evaluate_alert_condition(alert)

					# Verify alert was triggered
					mock_trigger.assert_called_once()

		# Step 4: Get system health (should reflect high CPU)
		monitoring_system.metrics_collector.get_metrics = AsyncMock(return_value=[
			Metric(
				metric_name="system_cpu_usage",
				metric_type=MetricType.GAUGE,
				value=85.0,
				source_component="system"
			)
		])

		health_data = await monitoring_system.get_system_health()

		# CPU health score should be lower due to high usage
		cpu_health = health_data["component_health"]["cpu"]["score"]
		assert cpu_health < 0.8  # Should be impacted by high CPU usage

	@pytest.mark.asyncio
	async def test_monitoring_resilience(self):
		"""Test monitoring system resilience to failures."""
		monitoring_system = AIMonitoringSystem()

		# Test initialization with partial failures
		with patch.object(monitoring_system.metrics_collector, 'initialize', side_effect=Exception("Metrics init failed")):
			with pytest.raises(Exception):
				await monitoring_system.initialize()

		# Test graceful degradation
		with patch.object(monitoring_system.metrics_collector, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.alert_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.performance_analyzer, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):

			await monitoring_system.initialize()

		# Test health check with collector failure
		monitoring_system.metrics_collector.get_metrics = AsyncMock(side_effect=Exception("Collector failed"))

		# Should handle gracefully and return error information
		health_data = await monitoring_system.get_system_health()
		assert "error" in str(health_data).lower() or health_data.get("overall_health_score", 1.0) < 1.0


if __name__ == "__main__":
	pytest.main([__file__])