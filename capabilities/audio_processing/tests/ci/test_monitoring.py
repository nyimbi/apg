"""
Audio Processing Monitoring Tests

Unit tests for monitoring, logging, alerting, and observability
components.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

from ...monitoring import (
	MonitoringEvent, AlertRule, Alert, StructuredLogger, AlertManager,
	HealthChecker, MonitoringDashboard, monitoring_context, monitored_operation,
	create_structured_logger, create_alert_manager, create_health_checker,
	create_monitoring_dashboard
)


class TestMonitoringEvent:
	"""Test MonitoringEvent data structure"""
	
	def test_event_creation(self):
		"""Test creating monitoring event"""
		event = MonitoringEvent(
			timestamp=datetime.utcnow(),
			event_type="processing_start",
			operation="transcription",
			tenant_id="test_tenant",
			user_id="user_001",
			job_id="job_001",
			status="info"
		)
		
		assert event.event_type == "processing_start"
		assert event.operation == "transcription"
		assert event.tenant_id == "test_tenant"
		assert event.user_id == "user_001"
		assert event.job_id == "job_001"
		assert event.status == "info"
		assert event.duration_ms is None
		assert event.metadata is None
	
	def test_event_with_metadata(self):
		"""Test creating event with metadata"""
		metadata = {"provider": "openai_whisper", "language": "en-US"}
		error_details = {"error_type": "ValidationError", "message": "Invalid input"}
		
		event = MonitoringEvent(
			timestamp=datetime.utcnow(),
			event_type="processing_error",
			operation="transcription",
			tenant_id="test_tenant",
			status="error",
			duration_ms=1500.0,
			metadata=metadata,
			error_details=error_details
		)
		
		assert event.duration_ms == 1500.0
		assert event.metadata == metadata
		assert event.error_details == error_details


class TestAlertRule:
	"""Test AlertRule data structure"""
	
	def test_alert_rule_creation(self):
		"""Test creating alert rule"""
		rule = AlertRule(
			rule_id="high_error_rate",
			name="High Error Rate Alert",
			metric="error_rate",
			operator="gt",
			threshold=0.05,
			duration_seconds=300,
			severity="high",
			notification_channels=["email", "slack"],
			enabled=True
		)
		
		assert rule.rule_id == "high_error_rate"
		assert rule.name == "High Error Rate Alert"
		assert rule.metric == "error_rate"
		assert rule.operator == "gt"
		assert rule.threshold == 0.05
		assert rule.duration_seconds == 300
		assert rule.severity == "high"
		assert "email" in rule.notification_channels
		assert rule.enabled is True
	
	def test_alert_rule_defaults(self):
		"""Test alert rule with default values"""
		rule = AlertRule(
			rule_id="test_rule",
			name="Test Rule",
			metric="cpu_usage",
			operator="gte",
			threshold=80.0
		)
		
		assert rule.duration_seconds == 300
		assert rule.severity == "medium"
		assert rule.notification_channels is None
		assert rule.enabled is True


class TestAlert:
	"""Test Alert data structure"""
	
	def test_alert_creation(self):
		"""Test creating alert"""
		alert = Alert(
			alert_id="alert_001",
			rule_id="high_error_rate",
			title="High Error Rate Detected",
			description="Error rate is 7% (threshold: 5%)",
			severity="high",
			status="firing",
			triggered_at=datetime.utcnow(),
			metadata={"metric_value": 0.07, "threshold": 0.05}
		)
		
		assert alert.alert_id == "alert_001"
		assert alert.rule_id == "high_error_rate"
		assert alert.title == "High Error Rate Detected"
		assert alert.severity == "high"
		assert alert.status == "firing"
		assert alert.resolved_at is None
		assert alert.metadata["metric_value"] == 0.07


class TestStructuredLogger:
	"""Test StructuredLogger functionality"""
	
	@pytest.fixture
	def structured_logger(self):
		"""Create structured logger instance"""
		return StructuredLogger("test_logger")
	
	def test_logger_initialization(self, structured_logger):
		"""Test logger initialization"""
		assert structured_logger.logger.name == "test_logger"
		assert isinstance(structured_logger.events, deque)
		assert len(structured_logger.events) == 0
	
	def test_log_event(self, structured_logger):
		"""Test logging event"""
		event = MonitoringEvent(
			timestamp=datetime.utcnow(),
			event_type="test_event",
			operation="test_operation",
			tenant_id="test_tenant",
			status="info"
		)
		
		structured_logger.log_event(event)
		
		# Event should be stored
		assert len(structured_logger.events) == 1
		assert structured_logger.events[0] == event
	
	def test_log_processing_start(self, structured_logger):
		"""Test logging processing start"""
		metadata = {"provider": "openai_whisper"}
		
		job_id = structured_logger.log_processing_start(
			operation="transcription",
			tenant_id="test_tenant",
			job_id="job_001",
			user_id="user_001",
			metadata=metadata
		)
		
		assert job_id == "job_001"
		assert len(structured_logger.events) == 1
		
		event = structured_logger.events[0]
		assert event.event_type == "processing_start"
		assert event.operation == "transcription"
		assert event.job_id == "job_001"
		assert event.metadata == metadata
	
	def test_log_processing_complete(self, structured_logger):
		"""Test logging processing completion"""
		metadata = {"result_size": 1024}
		
		structured_logger.log_processing_complete(
			operation="synthesis",
			tenant_id="test_tenant",
			job_id="job_002",
			duration_ms=2500.0,
			status="success",
			metadata=metadata
		)
		
		assert len(structured_logger.events) == 1
		
		event = structured_logger.events[0]
		assert event.event_type == "processing_complete"
		assert event.operation == "synthesis"
		assert event.job_id == "job_002"
		assert event.duration_ms == 2500.0
		assert event.status == "success"
		assert event.metadata == metadata
	
	def test_log_error(self, structured_logger):
		"""Test logging error"""
		error = ValueError("Test error message")
		metadata = {"input_size": 512}
		
		structured_logger.log_error(
			operation="analysis",
			tenant_id="test_tenant",
			error=error,
			job_id="job_003",
			metadata=metadata
		)
		
		assert len(structured_logger.events) == 1
		
		event = structured_logger.events[0]
		assert event.event_type == "processing_error"
		assert event.operation == "analysis"
		assert event.job_id == "job_003"
		assert event.status == "error"
		assert event.error_details["error_type"] == "ValueError"
		assert event.error_details["error_message"] == "Test error message"
	
	def test_get_recent_events(self, structured_logger):
		"""Test getting recent events"""
		# Add multiple events
		for i in range(10):
			event = MonitoringEvent(
				timestamp=datetime.utcnow(),
				event_type="test_event" if i % 2 == 0 else "other_event",
				operation="test_operation",
				tenant_id="test_tenant",
				status="info"
			)
			structured_logger.log_event(event)
		
		# Get all recent events
		all_events = structured_logger.get_recent_events(limit=15)
		assert len(all_events) == 10
		
		# Get limited events
		limited_events = structured_logger.get_recent_events(limit=5)
		assert len(limited_events) == 5
		
		# Get filtered events
		filtered_events = structured_logger.get_recent_events(limit=10, event_type="test_event")
		assert len(filtered_events) == 5  # Half of the events
		for event in filtered_events:
			assert event.event_type == "test_event"


class TestAlertManager:
	"""Test AlertManager functionality"""
	
	@pytest.fixture
	def alert_manager(self):
		"""Create alert manager instance"""
		return AlertManager()
	
	def test_alert_manager_initialization(self, alert_manager):
		"""Test alert manager initialization with default rules"""
		assert len(alert_manager.alert_rules) > 0
		assert len(alert_manager.active_alerts) == 0
		assert isinstance(alert_manager.alert_history, deque)
		
		# Check default rules exist
		rule_ids = list(alert_manager.alert_rules.keys())
		expected_rules = ["high_error_rate", "high_latency", "cpu_usage_high", 
						 "memory_usage_critical", "queue_size_large"]
		for expected_rule in expected_rules:
			assert expected_rule in rule_ids
	
	def test_add_remove_rule(self, alert_manager):
		"""Test adding and removing alert rules"""
		initial_count = len(alert_manager.alert_rules)
		
		# Add new rule
		new_rule = AlertRule(
			rule_id="test_rule",
			name="Test Rule",
			metric="test_metric",
			operator="gt",
			threshold=100.0
		)
		
		alert_manager.add_rule(new_rule)
		assert len(alert_manager.alert_rules) == initial_count + 1
		assert "test_rule" in alert_manager.alert_rules
		
		# Remove rule
		result = alert_manager.remove_rule("test_rule")
		assert result is True
		assert len(alert_manager.alert_rules) == initial_count
		assert "test_rule" not in alert_manager.alert_rules
		
		# Try to remove non-existent rule
		result = alert_manager.remove_rule("non_existent")
		assert result is False
	
	def test_register_notification_handler(self, alert_manager):
		"""Test registering notification handlers"""
		handler = MagicMock()
		
		alert_manager.register_notification_handler("test_channel", handler)
		assert "test_channel" in alert_manager.notification_handlers
		assert alert_manager.notification_handlers["test_channel"] == handler
	
	def test_evaluate_condition(self, alert_manager):
		"""Test alert condition evaluation"""
		assert alert_manager._evaluate_condition(10.0, "gt", 5.0) is True
		assert alert_manager._evaluate_condition(10.0, "gt", 15.0) is False
		assert alert_manager._evaluate_condition(10.0, "gte", 10.0) is True
		assert alert_manager._evaluate_condition(10.0, "lt", 15.0) is True
		assert alert_manager._evaluate_condition(10.0, "lte", 10.0) is True
		assert alert_manager._evaluate_condition(10.0, "eq", 10.0) is True
		assert alert_manager._evaluate_condition(10.0, "invalid", 5.0) is False
	
	async def test_check_metrics_and_alert(self, alert_manager):
		"""Test checking metrics and triggering alerts"""
		metrics = {
			"error_rate": 0.08,  # Above 5% threshold
			"cpu_usage": 60.0,   # Below 85% threshold
			"memory_usage": 95.0  # Above 90% threshold (critical)
		}
		
		alerts = await alert_manager.check_metrics_and_alert(metrics, "test_tenant")
		
		# Should trigger alerts for error_rate and memory_usage
		assert len(alerts) >= 2
		
		# Check active alerts
		active_alerts = alert_manager.get_active_alerts()
		assert len(active_alerts) >= 2
		
		# Check alert history
		history = alert_manager.get_alert_history()
		assert len(history) >= 2
	
	async def test_resolve_alert(self, alert_manager):
		"""Test resolving active alert"""
		# Create test alert
		rule = AlertRule(
			rule_id="test_alert_rule",
			name="Test Alert",
			metric="test_metric",
			operator="gt",
			threshold=50.0
		)
		alert_manager.add_rule(rule)
		
		# Trigger alert
		alerts = await alert_manager.check_metrics_and_alert({"test_metric": 75.0})
		assert len(alerts) == 1
		
		alert = alerts[0]
		assert alert.status == "firing"
		
		# Resolve alert
		result = await alert_manager.resolve_alert(alert.alert_id)
		assert result is True
		assert alert.status == "resolved"
		assert alert.resolved_at is not None
	
	def test_get_active_alerts_filtered(self, alert_manager):
		"""Test getting active alerts with severity filter"""
		# Add test alerts with different severities
		high_alert = Alert(
			alert_id="high_001",
			rule_id="test_rule",
			title="High Severity Alert",
			description="Test",
			severity="high",
			status="firing",
			triggered_at=datetime.utcnow()
		)
		
		medium_alert = Alert(
			alert_id="medium_001",
			rule_id="test_rule",
			title="Medium Severity Alert",
			description="Test",
			severity="medium",
			status="firing",
			triggered_at=datetime.utcnow()
		)
		
		alert_manager.active_alerts[high_alert.alert_id] = high_alert
		alert_manager.active_alerts[medium_alert.alert_id] = medium_alert
		
		# Get all active alerts
		all_alerts = alert_manager.get_active_alerts()
		assert len(all_alerts) == 2
		
		# Get only high severity alerts
		high_alerts = alert_manager.get_active_alerts(severity="high")
		assert len(high_alerts) == 1
		assert high_alerts[0].severity == "high"


class TestHealthChecker:
	"""Test HealthChecker functionality"""
	
	@pytest.fixture
	def health_checker(self):
		"""Create health checker instance"""
		return HealthChecker()
	
	def test_health_checker_initialization(self, health_checker):
		"""Test health checker initialization"""
		assert len(health_checker.health_checks) == 0
		assert len(health_checker.health_status) == 0
	
	def test_register_health_check(self, health_checker):
		"""Test registering health check"""
		def test_check():
			return {"status": "healthy", "message": "All good"}
		
		health_checker.register_health_check("test_component", test_check)
		assert "test_component" in health_checker.health_checks
		assert health_checker.health_checks["test_component"] == test_check
	
	async def test_run_health_checks_all_healthy(self, health_checker):
		"""Test running health checks when all components are healthy"""
		def healthy_check():
			return {"status": "healthy", "message": "Component OK"}
		
		def another_healthy_check():
			return {"status": "healthy", "uptime": "5 minutes"}
		
		health_checker.register_health_check("component1", healthy_check)
		health_checker.register_health_check("component2", another_healthy_check)
		
		health_report = await health_checker.run_health_checks()
		
		assert health_report["overall_status"] == "healthy"
		assert len(health_report["components"]) == 2
		assert health_report["components"]["component1"]["status"] == "healthy"
		assert health_report["components"]["component2"]["status"] == "healthy"
	
	async def test_run_health_checks_degraded(self, health_checker):
		"""Test running health checks with degraded component"""
		def healthy_check():
			return {"status": "healthy"}
		
		def degraded_check():
			return {"status": "degraded", "message": "High latency"}
		
		health_checker.register_health_check("component1", healthy_check)
		health_checker.register_health_check("component2", degraded_check)
		
		health_report = await health_checker.run_health_checks()
		
		assert health_report["overall_status"] == "degraded"
		assert health_report["components"]["component1"]["status"] == "healthy"
		assert health_report["components"]["component2"]["status"] == "degraded"
	
	async def test_run_health_checks_with_failure(self, health_checker):
		"""Test running health checks with component failure"""
		def healthy_check():
			return {"status": "healthy"}
		
		def failing_check():
			raise Exception("Component unavailable")
		
		health_checker.register_health_check("component1", healthy_check)
		health_checker.register_health_check("component2", failing_check)
		
		health_report = await health_checker.run_health_checks()
		
		assert health_report["overall_status"] == "unhealthy"
		assert health_report["components"]["component1"]["status"] == "healthy"
		assert health_report["components"]["component2"]["status"] == "unhealthy"
		assert "error" in health_report["components"]["component2"]
	
	def test_get_health_status(self, health_checker):
		"""Test getting current health status"""
		# Initially empty
		status = health_checker.get_health_status()
		assert status == {}
		
		# Set test status
		test_status = {
			"overall_status": "healthy",
			"components": {"test": {"status": "healthy"}},
			"checked_at": datetime.utcnow().isoformat()
		}
		health_checker.health_status = test_status
		
		status = health_checker.get_health_status()
		assert status == test_status


class TestMonitoringDashboard:
	"""Test MonitoringDashboard functionality"""
	
	@pytest.fixture
	def monitoring_dashboard(self):
		"""Create monitoring dashboard instance"""
		logger = StructuredLogger("test_logger")
		alert_manager = AlertManager()
		health_checker = HealthChecker()
		return MonitoringDashboard(logger, alert_manager, health_checker)
	
	async def test_get_dashboard_data(self, monitoring_dashboard):
		"""Test getting dashboard data"""
		# Add some test events
		for i in range(5):
			event = MonitoringEvent(
				timestamp=datetime.utcnow(),
				event_type="processing_complete",
				operation="transcription",
				tenant_id="test_tenant",
				status="completed" if i < 4 else "error",
				duration_ms=1000.0 + i * 100
			)
			monitoring_dashboard.logger.log_event(event)
		
		dashboard_data = await monitoring_dashboard.get_dashboard_data()
		
		assert "timestamp" in dashboard_data
		assert "system_health" in dashboard_data
		assert "active_alerts" in dashboard_data
		assert "recent_events" in dashboard_data
		assert "performance_metrics" in dashboard_data
		assert "resource_utilization" in dashboard_data
		assert "processing_stats" in dashboard_data
	
	async def test_get_performance_metrics(self, monitoring_dashboard):
		"""Test getting performance metrics"""
		# Add test events with varying durations
		durations = [500.0, 1000.0, 1500.0, 2000.0, 800.0]
		for i, duration in enumerate(durations):
			event = MonitoringEvent(
				timestamp=datetime.utcnow(),
				event_type="processing_complete",
				operation="test_operation",
				tenant_id="test_tenant",
				status="completed" if i < 4 else "error",
				duration_ms=duration
			)
			monitoring_dashboard.logger.log_event(event)
		
		metrics = await monitoring_dashboard._get_performance_metrics()
		
		assert "avg_latency" in metrics
		assert "max_latency" in metrics
		assert "throughput" in metrics
		assert "error_rate" in metrics
		
		assert metrics["avg_latency"] > 0
		assert metrics["max_latency"] == 2000.0
		assert metrics["throughput"] == 5
		assert metrics["error_rate"] == 0.2  # 1 error out of 5
	
	def test_get_resource_utilization(self, monitoring_dashboard):
		"""Test getting resource utilization"""
		with patch('psutil.cpu_percent', return_value=65.5), \
			 patch('psutil.virtual_memory') as mock_memory, \
			 patch('psutil.disk_usage') as mock_disk:
			
			# Mock memory info
			mock_memory.return_value = MagicMock(
				percent=75.2,
				available=2 * 1024**3  # 2 GB
			)
			
			# Mock disk info
			mock_disk.return_value = MagicMock(
				used=50 * 1024**3,    # 50 GB
				total=100 * 1024**3,  # 100 GB
				free=50 * 1024**3     # 50 GB
			)
			
			utilization = monitoring_dashboard._get_resource_utilization()
			
			assert utilization["cpu_usage"] == 65.5
			assert utilization["memory_usage"] == 75.2
			assert utilization["memory_available_gb"] == 2.0
			assert utilization["disk_usage"] == 50.0
			assert utilization["disk_free_gb"] == 50.0
	
	async def test_get_processing_stats(self, monitoring_dashboard):
		"""Test getting processing statistics"""
		# Add events for different operations
		operations = ["transcription", "synthesis", "analysis"]
		for op in operations:
			for i in range(3):
				event = MonitoringEvent(
					timestamp=datetime.utcnow(),
					event_type="processing_complete",
					operation=op,
					tenant_id="test_tenant",
					status="completed" if i < 2 else "failed",
					duration_ms=1000.0
				)
				monitoring_dashboard.logger.log_event(event)
		
		stats = await monitoring_dashboard._get_processing_stats()
		
		for op in operations:
			assert op in stats
			assert stats[op]["total"] == 3
			assert stats[op]["successful"] == 2
			assert stats[op]["failed"] == 1


class TestMonitoringContextManager:
	"""Test monitoring context manager"""
	
	async def test_monitoring_context_success(self):
		"""Test monitoring context with successful operation"""
		logger = StructuredLogger("test_logger")
		
		async with monitoring_context("test_operation", "test_tenant", logger) as job_id:
			assert job_id is not None
			await asyncio.sleep(0.1)  # Simulate work
		
		# Should have logged start and completion
		events = logger.get_recent_events()
		assert len(events) == 2
		
		start_event = events[0]
		assert start_event.event_type == "processing_start"
		assert start_event.operation == "test_operation"
		
		complete_event = events[1]
		assert complete_event.event_type == "processing_complete"
		assert complete_event.status == "success"
		assert complete_event.duration_ms > 0
	
	async def test_monitoring_context_failure(self):
		"""Test monitoring context with failed operation"""
		logger = StructuredLogger("test_logger")
		
		with pytest.raises(ValueError):
			async with monitoring_context("test_operation", "test_tenant", logger) as job_id:
				raise ValueError("Test error")
		
		# Should have logged start, error, and completion
		events = logger.get_recent_events()
		assert len(events) == 3
		
		start_event = events[0]
		assert start_event.event_type == "processing_start"
		
		error_event = events[1]
		assert error_event.event_type == "processing_error"
		assert error_event.status == "error"
		
		complete_event = events[2]
		assert complete_event.event_type == "processing_complete"
		assert complete_event.status == "error"


class TestMonitoringDecorator:
	"""Test monitoring decorator"""
	
	def test_monitored_operation_decorator(self):
		"""Test monitored operation decorator"""
		@monitored_operation("test_operation")
		async def test_function(value: int) -> int:
			return value * 2
		
		assert callable(test_function)
		assert hasattr(test_function, '__wrapped__')
	
	async def test_monitored_operation_execution(self):
		"""Test monitored operation decorator execution"""
		logger = StructuredLogger("test_logger")
		
		@monitored_operation("test_operation", logger)
		async def test_function(value: int, tenant_id: str = "default") -> int:
			return value * 2
		
		result = await test_function(5, tenant_id="test_tenant")
		
		assert result == 10
		
		# Should have logged events
		events = logger.get_recent_events()
		assert len(events) == 2  # start and complete
		
		start_event = events[0]
		assert start_event.event_type == "processing_start"
		assert start_event.operation == "test_operation"
		
		complete_event = events[1]
		assert complete_event.event_type == "processing_complete"
		assert complete_event.status == "success"


class TestMonitoringFactories:
	"""Test monitoring factory functions"""
	
	def test_create_structured_logger(self):
		"""Test structured logger factory"""
		logger = create_structured_logger("test_logger")
		assert isinstance(logger, StructuredLogger)
		assert logger.logger.name == "test_logger"
	
	def test_create_alert_manager(self):
		"""Test alert manager factory"""
		alert_manager = create_alert_manager()
		assert isinstance(alert_manager, AlertManager)
		assert len(alert_manager.alert_rules) > 0  # Has default rules
	
	def test_create_health_checker(self):
		"""Test health checker factory"""
		health_checker = create_health_checker()
		assert isinstance(health_checker, HealthChecker)
		assert len(health_checker.health_checks) == 0
	
	def test_create_monitoring_dashboard(self):
		"""Test monitoring dashboard factory"""
		logger = create_structured_logger("test")
		alert_manager = create_alert_manager()
		health_checker = create_health_checker()
		
		dashboard = create_monitoring_dashboard(logger, alert_manager, health_checker)
		assert isinstance(dashboard, MonitoringDashboard)
		assert dashboard.logger == logger
		assert dashboard.alert_manager == alert_manager
		assert dashboard.health_checker == health_checker


class TestMonitoringIntegration:
	"""Test monitoring component integration"""
	
	async def test_full_monitoring_pipeline(self):
		"""Test complete monitoring pipeline"""
		# Create components
		logger = create_structured_logger("integration_test")
		alert_manager = create_alert_manager()
		health_checker = create_health_checker()
		dashboard = create_monitoring_dashboard(logger, alert_manager, health_checker)
		
		# Register health check
		def test_health_check():
			return {"status": "healthy", "message": "All systems operational"}
		
		health_checker.register_health_check("test_service", test_health_check)
		
		# Register notification handler
		notifications = []
		def notification_handler(alert):
			notifications.append(alert)
		
		alert_manager.register_notification_handler("test_channel", notification_handler)
		
		# Simulate processing with monitoring
		async with monitoring_context("integration_test", "test_tenant", logger, metadata={"test": True}):
			await asyncio.sleep(0.1)
		
		# Check metrics and trigger alerts
		high_error_metrics = {"error_rate": 0.10}  # 10% error rate
		alerts = await alert_manager.check_metrics_and_alert(high_error_metrics)
		
		# Get dashboard data
		dashboard_data = await dashboard.get_dashboard_data()
		
		# Verify integration
		assert len(logger.get_recent_events()) == 2  # start and complete
		assert len(alerts) > 0  # Should have triggered error rate alert
		assert dashboard_data["system_health"]["overall_status"] == "healthy"
		assert len(dashboard_data["active_alerts"]) > 0
		assert dashboard_data["performance_metrics"]["throughput"] == 1


class TestMonitoringEdgeCases:
	"""Test monitoring edge cases and error handling"""
	
	def test_structured_logger_event_overflow(self):
		"""Test structured logger event buffer overflow"""
		logger = StructuredLogger("overflow_test")
		
		# Add more events than the buffer can hold
		for i in range(15000):  # More than maxlen=10000
			event = MonitoringEvent(
				timestamp=datetime.utcnow(),
				event_type="test_event",
				operation="test_operation",
				tenant_id="test_tenant"
			)
			logger.log_event(event)
		
		# Should not exceed buffer size
		assert len(logger.events) <= 10000
	
	async def test_alert_manager_notification_failure(self):
		"""Test alert manager behavior when notification fails"""
		alert_manager = AlertManager()
		
		# Register failing notification handler
		def failing_handler(alert):
			raise Exception("Notification service unavailable")
		
		alert_manager.register_notification_handler("failing_channel", failing_handler)
		
		# Add rule with failing notification
		rule = AlertRule(
			rule_id="test_notification_fail",
			name="Test Notification Failure",
			metric="test_metric",
			operator="gt",
			threshold=50.0,
			notification_channels=["failing_channel"]
		)
		alert_manager.add_rule(rule)
		
		# Should handle notification failure gracefully
		alerts = await alert_manager.check_metrics_and_alert({"test_metric": 75.0})
		assert len(alerts) == 1  # Alert should still be created
	
	async def test_health_checker_check_timeout(self):
		"""Test health checker with slow health check"""
		health_checker = HealthChecker()
		
		def slow_check():
			time.sleep(10)  # Simulate slow check
			return {"status": "healthy"}
		
		health_checker.register_health_check("slow_component", slow_check)
		
		# Should handle slow checks (though this is a simple test)
		health_report = await health_checker.run_health_checks()
		
		# In a real implementation, there would be timeout handling
		assert "slow_component" in health_report["components"]