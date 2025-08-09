"""
APG Integration API Management - Monitoring Tests

Unit and integration tests for monitoring components including metrics collection,
health monitoring, alerting, and system observability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from ..monitoring import (
	MetricsCollector, HealthMonitor, AlertManager, SystemMonitor,
	Metric, MetricType, HealthStatus, AlertSeverity, AlertRule,
	PerformanceMetrics, ResourceMetrics, BusinessMetrics
)

# =============================================================================
# Metrics Collector Tests
# =============================================================================

@pytest.mark.unit
class TestMetricsCollector:
	"""Test metrics collector functionality."""
	
	@pytest.mark.asyncio
	async def test_record_counter_metric(self, metrics_collector):
		"""Test recording counter metrics."""
		# Record counter metric
		await metrics_collector.record_counter(
			name="api_requests_total",
			value=1,
			labels={"method": "GET", "endpoint": "/users", "status": "200"}
		)
		
		# Verify metric was recorded
		metrics = await metrics_collector.get_metrics(
			name="api_requests_total",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert len(metrics) >= 1
		assert metrics[0].name == "api_requests_total"
		assert metrics[0].metric_type == MetricType.COUNTER
		assert metrics[0].labels["method"] == "GET"
	
	@pytest.mark.asyncio
	async def test_record_gauge_metric(self, metrics_collector):
		"""Test recording gauge metrics."""
		# Record gauge metric
		await metrics_collector.record_gauge(
			name="active_connections",
			value=42,
			labels={"service": "api_gateway", "instance": "gateway-1"}
		)
		
		# Verify metric was recorded
		metrics = await metrics_collector.get_metrics(
			name="active_connections",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert len(metrics) >= 1
		assert metrics[0].value == 42
		assert metrics[0].metric_type == MetricType.GAUGE
	
	@pytest.mark.asyncio
	async def test_record_histogram_metric(self, metrics_collector):
		"""Test recording histogram metrics."""
		# Record multiple response time measurements
		response_times = [100, 150, 200, 250, 300]
		
		for response_time in response_times:
			await metrics_collector.record_histogram(
				name="api_response_time_ms",
				value=response_time,
				labels={"endpoint": "/test", "method": "GET"}
			)
		
		# Get histogram metrics
		metrics = await metrics_collector.get_metrics(
			name="api_response_time_ms",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert len(metrics) == 5
		assert all(m.metric_type == MetricType.HISTOGRAM for m in metrics)
		assert sum(m.value for m in metrics) == sum(response_times)
	
	@pytest.mark.asyncio
	async def test_get_aggregated_metrics(self, metrics_collector):
		"""Test getting aggregated metrics."""
		# Record multiple metrics with same name
		for i in range(10):
			await metrics_collector.record_counter(
				name="test_counter",
				value=i + 1,
				labels={"batch": "test"}
			)
		
		# Get aggregated metrics
		total = await metrics_collector.get_aggregated_metric(
			name="test_counter",
			aggregation="sum",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert total == 55  # Sum of 1+2+3+...+10
		
		# Get average
		average = await metrics_collector.get_aggregated_metric(
			name="test_counter",
			aggregation="avg",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert average == 5.5  # Average of 1 through 10
	
	@pytest.mark.asyncio
	async def test_metrics_retention_cleanup(self, metrics_collector):
		"""Test metrics retention and cleanup."""
		# Record old metric
		old_timestamp = datetime.now(timezone.utc) - timedelta(days=10)
		await metrics_collector.record_counter(
			name="old_metric",
			value=1,
			timestamp=old_timestamp,
			labels={}
		)
		
		# Record recent metric
		await metrics_collector.record_counter(
			name="recent_metric",
			value=1,
			labels={}
		)
		
		# Run cleanup (simulate retention policy of 7 days)
		await metrics_collector.cleanup_old_metrics(retention_days=7)
		
		# Old metric should be removed, recent metric should remain
		old_metrics = await metrics_collector.get_metrics(
			name="old_metric",
			start_time=old_timestamp - timedelta(hours=1),
			end_time=old_timestamp + timedelta(hours=1)
		)
		
		recent_metrics = await metrics_collector.get_metrics(
			name="recent_metric",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert len(old_metrics) == 0
		assert len(recent_metrics) >= 1

@pytest.mark.unit
class TestHealthMonitor:
	"""Test health monitor functionality."""
	
	@pytest.mark.asyncio
	async def test_register_health_check(self, health_monitor):
		"""Test registering health check."""
		async def sample_health_check():
			return {
				"status": "healthy",
				"details": {"cpu": 45.2, "memory": 67.8}
			}
		
		# Register health check
		await health_monitor.register_health_check(
			name="sample_service",
			check_function=sample_health_check,
			interval_seconds=60
		)
		
		# Verify health check was registered
		checks = await health_monitor.list_health_checks()
		assert "sample_service" in checks
		assert checks["sample_service"]["interval_seconds"] == 60
	
	@pytest.mark.asyncio
	async def test_perform_health_check(self, health_monitor):
		"""Test performing health check."""
		async def healthy_check():
			return {
				"status": "healthy",
				"response_time_ms": 25,
				"details": {"version": "1.0.0", "uptime": 3600}
			}
		
		# Register and perform health check
		await health_monitor.register_health_check(
			name="healthy_service",
			check_function=healthy_check
		)
		
		result = await health_monitor.perform_health_check("healthy_service")
		
		assert result["status"] == "healthy"
		assert result["response_time_ms"] == 25
		assert "timestamp" in result
	
	@pytest.mark.asyncio
	async def test_health_check_failure(self, health_monitor):
		"""Test health check failure handling."""
		async def failing_check():
			raise Exception("Service unavailable")
		
		# Register failing health check
		await health_monitor.register_health_check(
			name="failing_service",
			check_function=failing_check
		)
		
		result = await health_monitor.perform_health_check("failing_service")
		
		assert result["status"] == "unhealthy"
		assert "error" in result
		assert "Service unavailable" in result["error"]
	
	@pytest.mark.asyncio
	async def test_system_health_aggregation(self, health_monitor):
		"""Test system-wide health aggregation."""
		# Register multiple health checks
		async def healthy_check():
			return {"status": "healthy"}
		
		async def degraded_check():
			return {"status": "degraded", "details": {"high_latency": True}}
		
		async def unhealthy_check():
			raise Exception("Service down")
		
		await health_monitor.register_health_check("service_1", healthy_check)
		await health_monitor.register_health_check("service_2", degraded_check)
		await health_monitor.register_health_check("service_3", unhealthy_check)
		
		# Get system health
		system_health = await health_monitor.get_system_health()
		
		assert "overall_status" in system_health
		assert "services" in system_health
		assert len(system_health["services"]) == 3
		
		# Overall status should be unhealthy due to one failing service
		assert system_health["overall_status"] in ["degraded", "unhealthy"]
	
	@pytest.mark.asyncio
	async def test_health_history_tracking(self, health_monitor):
		"""Test health check history tracking."""
		check_count = 0
		
		async def variable_health_check():
			nonlocal check_count
			check_count += 1
			
			if check_count <= 2:
				return {"status": "healthy"}
			elif check_count <= 4:
				return {"status": "degraded"}
			else:
				raise Exception("Service failed")
		
		# Register health check
		await health_monitor.register_health_check(
			name="variable_service",
			check_function=variable_health_check
		)
		
		# Perform multiple checks
		for _ in range(5):
			await health_monitor.perform_health_check("variable_service")
			await asyncio.sleep(0.1)  # Small delay between checks
		
		# Get health history
		history = await health_monitor.get_health_history(
			service_name="variable_service",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert len(history) == 5
		
		# Check status progression
		statuses = [entry["status"] for entry in history]
		assert statuses[:2] == ["healthy", "healthy"]
		assert statuses[2:4] == ["degraded", "degraded"]
		assert statuses[4] == "unhealthy"

# =============================================================================
# Alert Manager Tests
# =============================================================================

@pytest.mark.unit
class TestAlertManager:
	"""Test alert manager functionality."""
	
	@pytest.mark.asyncio
	async def test_create_alert_rule(self, alert_manager):
		"""Test creating alert rule."""
		rule = AlertRule(
			rule_id="high_error_rate",
			rule_name="High Error Rate",
			description="Alert when error rate exceeds threshold",
			metric_name="api_error_rate",
			condition="greater_than",
			threshold=0.05,  # 5% error rate
			severity=AlertSeverity.WARNING,
			evaluation_interval_seconds=60,
			notification_channels=["email", "slack"]
		)
		
		success = await alert_manager.create_alert_rule(rule)
		assert success is True
		
		# Verify rule was created
		retrieved_rule = await alert_manager.get_alert_rule("high_error_rate")
		assert retrieved_rule is not None
		assert retrieved_rule.rule_name == "High Error Rate"
		assert retrieved_rule.threshold == 0.05
	
	@pytest.mark.asyncio
	async def test_evaluate_alert_rules(self, alert_manager, metrics_collector):
		"""Test evaluating alert rules against metrics."""
		# Create alert rule for high response time
		rule = AlertRule(
			rule_id="high_response_time",
			rule_name="High Response Time",
			description="Alert when response time exceeds 1000ms",
			metric_name="api_response_time_ms",
			condition="greater_than",
			threshold=1000,
			severity=AlertSeverity.CRITICAL,
			evaluation_interval_seconds=30
		)
		
		await alert_manager.create_alert_rule(rule)
		
		# Record metrics that should trigger alert
		await metrics_collector.record_histogram(
			name="api_response_time_ms",
			value=1500,  # Exceeds threshold
			labels={"endpoint": "/slow-endpoint"}
		)
		
		# Evaluate rules
		triggered_alerts = await alert_manager.evaluate_alert_rules()
		
		assert len(triggered_alerts) >= 1
		assert any(alert.rule_id == "high_response_time" for alert in triggered_alerts)
	
	@pytest.mark.asyncio
	async def test_alert_notification(self, alert_manager):
		"""Test alert notification sending."""
		# Mock notification channels
		email_sent = False
		slack_sent = False
		
		async def mock_email_notification(alert):
			nonlocal email_sent
			email_sent = True
		
		async def mock_slack_notification(alert):
			nonlocal slack_sent
			slack_sent = True
		
		# Register notification handlers
		await alert_manager.register_notification_handler("email", mock_email_notification)
		await alert_manager.register_notification_handler("slack", mock_slack_notification)
		
		# Create and trigger alert
		alert_data = {
			"rule_id": "test_rule",
			"rule_name": "Test Alert",
			"severity": AlertSeverity.WARNING,
			"message": "Test alert message",
			"metric_value": 0.08,
			"threshold": 0.05,
			"notification_channels": ["email", "slack"]
		}
		
		await alert_manager.send_alert(alert_data)
		
		# Verify notifications were sent
		assert email_sent is True
		assert slack_sent is True
	
	@pytest.mark.asyncio
	async def test_alert_suppression(self, alert_manager):
		"""Test alert suppression and rate limiting."""
		# Create alert rule with suppression
		rule = AlertRule(
			rule_id="suppressed_alert",
			rule_name="Suppressed Alert",
			description="Alert with suppression",
			metric_name="test_metric",
			condition="greater_than",
			threshold=10,
			severity=AlertSeverity.WARNING,
			suppression_duration_seconds=300  # 5 minutes
		)
		
		await alert_manager.create_alert_rule(rule)
		
		# Trigger alert multiple times
		alert_data = {
			"rule_id": "suppressed_alert",
			"rule_name": "Suppressed Alert",
			"severity": AlertSeverity.WARNING,
			"message": "Test suppression",
			"metric_value": 15,
			"threshold": 10
		}
		
		notifications_sent = 0
		
		async def count_notifications(alert):
			nonlocal notifications_sent
			notifications_sent += 1
		
		await alert_manager.register_notification_handler("test", count_notifications)
		alert_data["notification_channels"] = ["test"]
		
		# Send alert multiple times
		for _ in range(5):
			await alert_manager.send_alert(alert_data)
			await asyncio.sleep(0.1)
		
		# Only one notification should be sent due to suppression
		assert notifications_sent == 1
	
	@pytest.mark.asyncio
	async def test_alert_escalation(self, alert_manager):
		"""Test alert escalation based on severity and duration."""
		# Create escalation rule
		escalation_rule = {
			"rule_id": "escalation_test",
			"initial_severity": AlertSeverity.WARNING,
			"escalation_levels": [
				{"duration_minutes": 5, "severity": AlertSeverity.CRITICAL},
				{"duration_minutes": 10, "severity": AlertSeverity.EMERGENCY}
			],
			"escalation_channels": {
				AlertSeverity.WARNING: ["email"],
				AlertSeverity.CRITICAL: ["email", "slack"],
				AlertSeverity.EMERGENCY: ["email", "slack", "pagerduty"]
			}
		}
		
		await alert_manager.create_escalation_rule(escalation_rule)
		
		# Create initial warning alert
		alert_data = {
			"rule_id": "escalation_test",
			"rule_name": "Escalation Test",
			"severity": AlertSeverity.WARNING,
			"message": "Initial warning",
			"first_triggered": datetime.now(timezone.utc) - timedelta(minutes=6)  # 6 minutes ago
		}
		
		# Evaluate escalation
		escalated_alert = await alert_manager.evaluate_escalation(alert_data)
		
		# Should be escalated to CRITICAL after 5 minutes
		assert escalated_alert.severity == AlertSeverity.CRITICAL
		assert "slack" in escalated_alert.notification_channels

# =============================================================================
# System Monitor Tests
# =============================================================================

@pytest.mark.unit
class TestSystemMonitor:
	"""Test system monitor functionality."""
	
	@pytest_asyncio.fixture
	async def system_monitor(self, metrics_collector, health_monitor, alert_manager):
		"""Create system monitor."""
		monitor = SystemMonitor(metrics_collector, health_monitor, alert_manager)
		await monitor.initialize()
		
		yield monitor
		
		await monitor.cleanup()
	
	@pytest.mark.asyncio
	async def test_collect_performance_metrics(self, system_monitor):
		"""Test collecting performance metrics."""
		# Mock system performance data
		with patch('psutil.cpu_percent', return_value=65.2):
			with patch('psutil.virtual_memory') as mock_memory:
				mock_memory.return_value.percent = 78.5
				mock_memory.return_value.available = 8 * 1024 * 1024 * 1024  # 8GB
				
				with patch('psutil.disk_usage') as mock_disk:
					mock_disk.return_value.percent = 45.0
					
					# Collect performance metrics
					metrics = await system_monitor.collect_performance_metrics()
					
					assert "cpu_usage_percent" in metrics
					assert "memory_usage_percent" in metrics
					assert "disk_usage_percent" in metrics
					assert metrics["cpu_usage_percent"] == 65.2
					assert metrics["memory_usage_percent"] == 78.5
	
	@pytest.mark.asyncio
	async def test_collect_business_metrics(self, system_monitor, metrics_collector):
		"""Test collecting business metrics."""
		# Record some business metrics
		current_time = datetime.now(timezone.utc)
		
		# API calls in the last hour
		for i in range(100):
			await metrics_collector.record_counter(
				name="api_requests_total",
				value=1,
				timestamp=current_time - timedelta(minutes=i % 60),
				labels={"status": "200" if i % 10 != 0 else "500"}
			)
		
		# Collect business metrics
		business_metrics = await system_monitor.collect_business_metrics()
		
		assert "total_api_calls" in business_metrics
		assert "error_rate" in business_metrics
		assert "active_consumers" in business_metrics
		
		# Should have recorded 100 API calls
		assert business_metrics["total_api_calls"] == 100
		
		# Error rate should be 10% (every 10th request fails)
		assert abs(business_metrics["error_rate"] - 0.1) < 0.01
	
	@pytest.mark.asyncio
	async def test_generate_health_report(self, system_monitor):
		"""Test generating comprehensive health report."""
		# Generate health report
		report = await system_monitor.generate_health_report()
		
		assert "timestamp" in report
		assert "overall_status" in report
		assert "performance_metrics" in report
		assert "business_metrics" in report
		assert "service_health" in report
		assert "active_alerts" in report
		
		# Overall status should be valid
		assert report["overall_status"] in ["healthy", "degraded", "unhealthy"]
	
	@pytest.mark.asyncio
	async def test_performance_threshold_monitoring(self, system_monitor):
		"""Test performance threshold monitoring."""
		# Set performance thresholds
		thresholds = {
			"cpu_usage_percent": 80.0,
			"memory_usage_percent": 85.0,
			"disk_usage_percent": 90.0,
			"api_response_time_ms": 1000.0
		}
		
		await system_monitor.set_performance_thresholds(thresholds)
		
		# Mock high resource usage
		with patch('psutil.cpu_percent', return_value=85.0):  # Exceeds threshold
			with patch('psutil.virtual_memory') as mock_memory:
				mock_memory.return_value.percent = 90.0  # Exceeds threshold
				
				# Check thresholds
				violations = await system_monitor.check_performance_thresholds()
				
				assert len(violations) >= 2
				assert any(v["metric"] == "cpu_usage_percent" for v in violations)
				assert any(v["metric"] == "memory_usage_percent" for v in violations)

# =============================================================================
# Monitoring Integration Tests
# =============================================================================

@pytest.mark.integration
class TestMonitoringIntegration:
	"""Test monitoring integration scenarios."""
	
	@pytest.mark.asyncio
	async def test_end_to_end_monitoring_flow(self, metrics_collector, health_monitor, alert_manager):
		"""Test complete monitoring flow from metrics to alerts."""
		# Set up alert rule for high error rate
		rule = AlertRule(
			rule_id="integration_test_alert",
			rule_name="Integration Test Alert",
			description="Test end-to-end monitoring",
			metric_name="api_error_rate",
			condition="greater_than",
			threshold=0.1,  # 10% error rate
			severity=AlertSeverity.WARNING,
			evaluation_interval_seconds=10
		)
		
		await alert_manager.create_alert_rule(rule)
		
		# Record metrics that exceed threshold
		for i in range(20):
			status = "500" if i < 5 else "200"  # 25% error rate
			await metrics_collector.record_counter(
				name="api_requests_total",
				value=1,
				labels={"status": status, "endpoint": "/test"}
			)
		
		# Calculate error rate metric
		total_requests = 20
		error_requests = 5
		error_rate = error_requests / total_requests
		
		await metrics_collector.record_gauge(
			name="api_error_rate",
			value=error_rate,
			labels={"endpoint": "/test"}
		)
		
		# Register health check that reports error rate
		async def error_rate_health_check():
			if error_rate > 0.2:
				return {"status": "unhealthy", "error_rate": error_rate}
			elif error_rate > 0.05:
				return {"status": "degraded", "error_rate": error_rate}
			else:
				return {"status": "healthy", "error_rate": error_rate}
		
		await health_monitor.register_health_check(
			name="error_rate_monitor",
			check_function=error_rate_health_check
		)
		
		# Perform health check
		health_result = await health_monitor.perform_health_check("error_rate_monitor")
		assert health_result["status"] == "unhealthy"
		
		# Evaluate alert rules
		triggered_alerts = await alert_manager.evaluate_alert_rules()
		
		# Should have triggered an alert
		assert len(triggered_alerts) >= 1
		assert any(alert.rule_id == "integration_test_alert" for alert in triggered_alerts)
	
	@pytest.mark.asyncio
	async def test_multi_service_health_monitoring(self, health_monitor):
		"""Test monitoring multiple services with dependencies."""
		# Define service dependencies
		services = {
			"database": {
				"health_check": lambda: {"status": "healthy", "connections": 5},
				"dependencies": []
			},
			"api_service": {
				"health_check": lambda: {"status": "healthy", "version": "1.0.0"},
				"dependencies": ["database"]
			},
			"gateway": {
				"health_check": lambda: {"status": "healthy", "routes": 10},
				"dependencies": ["api_service"]
			}
		}
		
		# Register all health checks
		for service_name, config in services.items():
			await health_monitor.register_health_check(
				name=service_name,
				check_function=config["health_check"]
			)
		
		# Set up dependency tracking
		for service_name, config in services.items():
			if config["dependencies"]:
				await health_monitor.set_service_dependencies(
					service_name, config["dependencies"]
				)
		
		# Get system health with dependency analysis
		system_health = await health_monitor.get_system_health(include_dependencies=True)
		
		assert system_health["overall_status"] == "healthy"
		assert len(system_health["services"]) == 3
		
		# Simulate database failure
		services["database"]["health_check"] = lambda: {"status": "unhealthy", "error": "Connection failed"}
		
		# Update health check
		await health_monitor.register_health_check(
			name="database",
			check_function=services["database"]["health_check"]
		)
		
		# Re-check system health
		system_health = await health_monitor.get_system_health(include_dependencies=True)
		
		# System should be unhealthy due to database failure affecting dependent services
		assert system_health["overall_status"] in ["degraded", "unhealthy"]

# =============================================================================
# Monitoring Performance Tests
# =============================================================================

@pytest.mark.performance
class TestMonitoringPerformance:
	"""Test monitoring performance characteristics."""
	
	@pytest.mark.asyncio
	async def test_metrics_collection_throughput(self, metrics_collector):
		"""Test metrics collection throughput."""
		import time
		
		start_time = time.time()
		metric_count = 1000
		
		# Record many metrics
		for i in range(metric_count):
			await metrics_collector.record_counter(
				name="throughput_test",
				value=1,
				labels={"batch": str(i // 100), "index": str(i)}
			)
		
		end_time = time.time()
		duration = end_time - start_time
		
		throughput = metric_count / duration
		assert throughput > 100  # At least 100 metrics/second
		
		print(f"Recorded {metric_count} metrics in {duration:.2f}s (throughput: {throughput:.1f} metrics/s)")
	
	@pytest.mark.asyncio
	async def test_concurrent_health_checks(self, health_monitor):
		"""Test concurrent health check performance."""
		import time
		
		# Register multiple health checks
		health_checks = {}
		for i in range(20):
			service_name = f"service_{i:02d}"
			
			async def health_check(service_id=i):
				# Simulate variable response times
				await asyncio.sleep(0.01 + (service_id % 5) * 0.01)
				return {"status": "healthy", "service_id": service_id}
			
			health_checks[service_name] = health_check
			await health_monitor.register_health_check(service_name, health_check)
		
		# Perform all health checks concurrently
		start_time = time.time()
		
		tasks = []
		for service_name in health_checks.keys():
			task = health_monitor.perform_health_check(service_name)
			tasks.append(task)
		
		results = await asyncio.gather(*tasks)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# All checks should complete
		assert len(results) == 20
		assert all(result["status"] == "healthy" for result in results)
		
		# Should be faster than sequential execution
		assert duration < 2.0  # Should complete within 2 seconds
		
		throughput = len(results) / duration
		assert throughput > 10  # At least 10 checks/second
		
		print(f"Performed {len(results)} health checks in {duration:.2f}s (throughput: {throughput:.1f} checks/s)")
	
	@pytest.mark.asyncio
	async def test_alert_evaluation_performance(self, alert_manager, metrics_collector):
		"""Test alert rule evaluation performance."""
		import time
		
		# Create multiple alert rules
		rules = []
		for i in range(50):
			rule = AlertRule(
				rule_id=f"perf_rule_{i:02d}",
				rule_name=f"Performance Rule {i}",
				description=f"Performance test rule {i}",
				metric_name=f"test_metric_{i % 10}",  # 10 different metrics
				condition="greater_than",
				threshold=float(i * 10),
				severity=AlertSeverity.WARNING,
				evaluation_interval_seconds=60
			)
			rules.append(rule)
			await alert_manager.create_alert_rule(rule)
		
		# Record metrics for evaluation
		for i in range(100):
			await metrics_collector.record_gauge(
				name=f"test_metric_{i % 10}",
				value=float(i * 15),  # Some will exceed thresholds
				labels={"test": "performance"}
			)
		
		# Evaluate all rules
		start_time = time.time()
		
		triggered_alerts = await alert_manager.evaluate_alert_rules()
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Should evaluate quickly
		assert duration < 5.0  # Less than 5 seconds
		
		# Should have triggered some alerts
		assert len(triggered_alerts) > 0
		
		evaluation_rate = len(rules) / duration
		assert evaluation_rate > 10  # At least 10 evaluations/second
		
		print(f"Evaluated {len(rules)} alert rules in {duration:.2f}s (rate: {evaluation_rate:.1f} rules/s)")
		print(f"Triggered {len(triggered_alerts)} alerts")