"""
Real-Time Analytics Tests

Comprehensive tests for real-time analytics and dashboard functionality.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from uuid_extensions import uuid7str

from ..models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType
from ..realtime_analytics import (
	RealTimeAnalyticsEngine, 
	RealTimeMetric, 
	Alert, 
	MetricType, 
	AlertSeverity,
	TimeRange
)


class TestRealTimeAnalyticsEngine:
	"""Test real-time analytics engine"""
	
	@pytest.fixture
	async def analytics_engine(self, temp_database):
		"""Create analytics engine for testing"""
		from ..realtime_analytics import create_analytics_engine
		engine = create_analytics_engine(temp_database)
		await engine.start_analytics_engine()
		yield engine
		await engine.stop_analytics_engine()
	
	async def test_analytics_engine_initialization(self, analytics_engine):
		"""Test analytics engine initialization"""
		assert analytics_engine is not None
		assert analytics_engine._running is True
		assert len(analytics_engine._alert_thresholds) > 0
		assert analytics_engine._metrics_buffer is not None
	
	async def test_record_transaction_metric(self, analytics_engine, sample_transaction):
		"""Test recording transaction metrics"""
		initial_count = len(analytics_engine._metrics_buffer)
		
		await analytics_engine.record_transaction_metric(sample_transaction)
		
		# Should add multiple metrics (volume, revenue, payment method)
		assert len(analytics_engine._metrics_buffer) > initial_count
		
		# Check metric types are recorded
		metric_types = [m.metric_type for m in analytics_engine._metrics_buffer]
		assert MetricType.TRANSACTION_VOLUME in metric_types
		assert MetricType.REVENUE_ANALYTICS in metric_types
		assert MetricType.PAYMENT_METHOD_DISTRIBUTION in metric_types
	
	async def test_record_processor_metric(self, analytics_engine):
		"""Test recording processor performance metrics"""
		initial_count = len(analytics_engine._metrics_buffer)
		
		await analytics_engine.record_processor_metric("stripe", True, 450.5)
		
		assert len(analytics_engine._metrics_buffer) > initial_count
		
		# Check processor metrics
		recent_metrics = list(analytics_engine._metrics_buffer)[-2:]  # Last 2 metrics
		success_metric = next(m for m in recent_metrics if m.metric_type == MetricType.SUCCESS_RATE)
		performance_metric = next(m for m in recent_metrics if m.metric_type == MetricType.PROCESSOR_PERFORMANCE)
		
		assert success_metric.value == 1.0  # Success
		assert success_metric.processor_name == "stripe"
		assert performance_metric.value == 450.5
		assert performance_metric.processor_name == "stripe"
	
	async def test_record_fraud_metric(self, analytics_engine):
		"""Test recording fraud detection metrics"""
		initial_count = len(analytics_engine._metrics_buffer)
		
		await analytics_engine.record_fraud_metric("txn_123", 85.5, True)
		
		assert len(analytics_engine._metrics_buffer) > initial_count
		
		fraud_metric = list(analytics_engine._metrics_buffer)[-1]
		assert fraud_metric.metric_type == MetricType.FRAUD_RATE
		assert fraud_metric.value == 1.0  # Is fraud
		assert fraud_metric.metadata["fraud_score"] == 85.5
		assert fraud_metric.metadata["is_fraud"] is True
	
	async def test_get_real_time_dashboard(self, analytics_engine, sample_transaction):
		"""Test getting real-time dashboard data"""
		# Add some test metrics
		await analytics_engine.record_transaction_metric(sample_transaction)
		await analytics_engine.record_processor_metric("mpesa", True, 1200.0)
		await analytics_engine.record_fraud_metric("txn_123", 15.0, False)
		
		dashboard_data = await analytics_engine.get_real_time_dashboard()
		
		assert "timestamp" in dashboard_data
		assert "metrics" in dashboard_data
		assert "charts" in dashboard_data
		assert "alerts" in dashboard_data
		assert "summary" in dashboard_data
		
		# Check metrics structure
		metrics = dashboard_data["metrics"]
		assert "transaction_volume" in metrics
		assert "success_rate" in metrics
		assert "average_response_time" in metrics
		assert "fraud_rate" in metrics
		assert "revenue" in metrics
		
		# Check charts structure
		charts = dashboard_data["charts"]
		assert "time_series" in charts
		assert "payment_methods" in charts
		assert "processor_performance" in charts
		assert "geographic_distribution" in charts
	
	async def test_alert_generation(self, analytics_engine):
		"""Test automatic alert generation"""
		# Generate metrics that should trigger alerts
		
		# Low success rate alert
		for i in range(10):
			success = i < 3  # Only 30% success rate
			await analytics_engine.record_processor_metric("test_processor", success, 500.0)
		
		# Process alerts
		await analytics_engine._check_alert_conditions()
		
		# Should have generated a success rate alert
		success_rate_alerts = [
			alert for alert in analytics_engine._active_alerts.values()
			if alert.metric_type == MetricType.SUCCESS_RATE and not alert.resolved
		]
		assert len(success_rate_alerts) > 0
		
		alert = success_rate_alerts[0]
		assert alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM]
		assert "success rate" in alert.message.lower()
	
	async def test_fraud_rate_alerts(self, analytics_engine):
		"""Test fraud rate alert generation"""
		# Generate high fraud rate
		for i in range(20):
			is_fraud = i < 12  # 60% fraud rate
			await analytics_engine.record_fraud_metric(f"txn_{i}", 75.0 if is_fraud else 25.0, is_fraud)
		
		await analytics_engine._check_alert_conditions()
		
		fraud_alerts = [
			alert for alert in analytics_engine._active_alerts.values()
			if alert.metric_type == MetricType.FRAUD_RATE and not alert.resolved
		]
		assert len(fraud_alerts) > 0
		
		alert = fraud_alerts[0]
		assert alert.severity == AlertSeverity.CRITICAL
		assert "fraud rate" in alert.message.lower()
	
	async def test_resolve_alert(self, analytics_engine):
		"""Test alert resolution"""
		# Generate an alert first
		for i in range(5):
			await analytics_engine.record_processor_metric("test_processor", False, 500.0)
		
		await analytics_engine._check_alert_conditions()
		
		# Get the alert ID
		active_alerts = list(analytics_engine._active_alerts.keys())
		assert len(active_alerts) > 0
		
		alert_id = active_alerts[0]
		
		# Resolve the alert
		resolved = await analytics_engine.resolve_alert(alert_id)
		assert resolved is True
		
		# Check that alert is marked as resolved
		alert = analytics_engine._active_alerts[alert_id]
		assert alert.resolved is True
	
	async def test_custom_report_generation(self, analytics_engine, temp_database, sample_transaction):
		"""Test custom report generation"""
		# Create some test data
		await temp_database.create_payment_transaction(sample_transaction)
		
		report_config = {
			"include_processor_breakdown": True,
			"include_fraud_analysis": True,
			"include_customer_insights": True
		}
		
		start_date = "2025-01-29"
		end_date = "2025-01-30"
		
		report = await analytics_engine.get_custom_report(
			report_config,
			start_date,
			end_date,
			sample_transaction.merchant_id
		)
		
		assert "report_id" in report
		assert "generated_at" in report
		assert "date_range" in report
		assert "base_analytics" in report
		assert "custom_metrics" in report
		
		# Check custom metrics
		custom_metrics = report["custom_metrics"]
		assert "processor_breakdown" in custom_metrics
		assert "fraud_analysis" in custom_metrics
		assert "customer_insights" in custom_metrics
	
	async def test_metric_caching(self, analytics_engine):
		"""Test metric caching functionality"""
		# Get dashboard data twice
		dashboard_data_1 = await analytics_engine.get_real_time_dashboard()
		dashboard_data_2 = await analytics_engine.get_real_time_dashboard()
		
		# Should use cache for second request
		assert "cache_timestamp" in dashboard_data_2
		
		# Cache should work for merchant-specific data too
		merchant_data_1 = await analytics_engine.get_real_time_dashboard("test_merchant")
		merchant_data_2 = await analytics_engine.get_real_time_dashboard("test_merchant")
		
		assert "cache_timestamp" in merchant_data_2
	
	async def test_time_series_generation(self, analytics_engine, sample_transaction):
		"""Test time series data generation"""
		# Create transactions at different times
		now = datetime.now(timezone.utc)
		
		for i in range(5):
			transaction = PaymentTransaction(
				id=uuid7str(),
				amount=1000 * (i + 1),
				currency="KES",
				payment_method_type=PaymentMethodType.MPESA,
				merchant_id="test_merchant",
				customer_id=f"customer_{i}",
				description=f"Test transaction {i}",
				status=PaymentStatus.COMPLETED,
				created_at=now - timedelta(hours=i)
			)
			await analytics_engine.record_transaction_metric(transaction)
		
		dashboard_data = await analytics_engine.get_real_time_dashboard()
		time_series = dashboard_data["charts"]["time_series"]
		
		assert "labels" in time_series
		assert "datasets" in time_series
		
		datasets = time_series["datasets"]
		assert "volume" in datasets
		assert "revenue" in datasets
		assert "success_rate" in datasets
		assert "fraud_rate" in datasets
		
		# Should have data points
		assert len(datasets["volume"]) > 0
		assert len(datasets["revenue"]) > 0
	
	async def test_payment_method_distribution(self, analytics_engine):
		"""Test payment method distribution calculation"""
		# Create transactions with different payment methods
		methods = [PaymentMethodType.MPESA, PaymentMethodType.CREDIT_CARD, PaymentMethodType.PAYPAL]
		
		for i, method in enumerate(methods):
			for j in range(i + 1):  # Different counts for each method
				transaction = PaymentTransaction(
					id=uuid7str(),
					amount=1000,
					currency="KES" if method == PaymentMethodType.MPESA else "USD",
					payment_method_type=method,
					merchant_id="test_merchant",
					customer_id=f"customer_{i}_{j}",
					description=f"Test transaction",
					status=PaymentStatus.COMPLETED
				)
				await analytics_engine.record_transaction_metric(transaction)
		
		dashboard_data = await analytics_engine.get_real_time_dashboard()
		payment_methods = dashboard_data["charts"]["payment_methods"]
		
		assert "labels" in payment_methods
		assert "data" in payment_methods
		assert "percentages" in payment_methods
		
		# Should have data for each payment method
		assert len(payment_methods["labels"]) > 0
		assert len(payment_methods["data"]) == len(payment_methods["labels"])
		assert len(payment_methods["percentages"]) == len(payment_methods["labels"])
		
		# Percentages should sum to 100
		assert abs(sum(payment_methods["percentages"]) - 100.0) < 0.1
	
	async def test_processor_performance_tracking(self, analytics_engine):
		"""Test processor performance tracking"""
		processors = ["mpesa", "stripe", "paypal"]
		
		for processor in processors:
			# Different performance characteristics
			if processor == "mpesa":
				success_rate = 0.98
				avg_response_time = 1200
			elif processor == "stripe":
				success_rate = 0.99
				avg_response_time = 450
			else:  # paypal
				success_rate = 0.97
				avg_response_time = 800
			
			# Generate metrics
			for i in range(10):
				success = i < (success_rate * 10)
				response_time = avg_response_time + (i * 50)  # Some variation
				await analytics_engine.record_processor_metric(processor, success, response_time)
		
		dashboard_data = await analytics_engine.get_real_time_dashboard()
		processor_performance = dashboard_data["charts"]["processor_performance"]
		
		assert len(processor_performance) > 0
		
		for performance in processor_performance:
			assert "processor" in performance
			assert "success_rate" in performance
			assert "average_response_time" in performance
			assert "transaction_count" in performance
			
			# Check reasonable values
			assert 0 <= performance["success_rate"] <= 100
			assert performance["average_response_time"] > 0
			assert performance["transaction_count"] > 0
	
	async def test_concurrent_metric_recording(self, analytics_engine):
		"""Test concurrent metric recording"""
		# Create multiple concurrent metric recording tasks
		tasks = []
		
		for i in range(20):
			transaction = PaymentTransaction(
				id=uuid7str(),
				amount=1000,
				currency="KES",
				payment_method_type=PaymentMethodType.MPESA,
				merchant_id=f"merchant_{i % 3}",  # 3 different merchants
				customer_id=f"customer_{i}",
				description=f"Concurrent transaction {i}",
				status=PaymentStatus.COMPLETED
			)
			tasks.append(analytics_engine.record_transaction_metric(transaction))
		
		# Execute all tasks concurrently
		await asyncio.gather(*tasks)
		
		# Should have recorded all metrics
		volume_metrics = [
			m for m in analytics_engine._metrics_buffer
			if m.metric_type == MetricType.TRANSACTION_VOLUME
		]
		assert len(volume_metrics) >= 20
	
	async def test_subscription_system(self, analytics_engine):
		"""Test real-time subscription system"""
		received_events = []
		
		def test_callback(event_type: str, data: dict):
			received_events.append((event_type, data))
		
		# Subscribe to updates
		await analytics_engine.subscribe_to_updates("test_subscriber", test_callback)
		
		# Generate some metrics that should trigger notifications
		transaction = PaymentTransaction(
			id=uuid7str(),
			amount=5000,
			currency="KES",
			payment_method_type=PaymentMethodType.MPESA,
			merchant_id="test_merchant",
			customer_id="test_customer",
			description="Test subscription transaction",
			status=PaymentStatus.COMPLETED
		)
		
		await analytics_engine.record_transaction_metric(transaction)
		await analytics_engine.record_processor_metric("mpesa", True, 1000.0)
		
		# Should have received notifications
		assert len(received_events) > 0
		
		# Check event types
		event_types = [event_type for event_type, _ in received_events]
		assert "transaction_metric" in event_types
		assert "processor_metric" in event_types
		
		# Unsubscribe
		await analytics_engine.unsubscribe_from_updates("test_subscriber")
	
	async def test_merchant_specific_analytics(self, analytics_engine):
		"""Test merchant-specific analytics filtering"""
		merchant_1 = "merchant_1"
		merchant_2 = "merchant_2"
		
		# Create transactions for different merchants
		for merchant_id in [merchant_1, merchant_2]:
			for i in range(5):
				transaction = PaymentTransaction(
					id=uuid7str(),
					amount=1000 * (i + 1),
					currency="KES",
					payment_method_type=PaymentMethodType.MPESA,
					merchant_id=merchant_id,
					customer_id=f"customer_{merchant_id}_{i}",
					description=f"Test transaction for {merchant_id}",
					status=PaymentStatus.COMPLETED
				)
				await analytics_engine.record_transaction_metric(transaction)
		
		# Get analytics for specific merchant
		merchant_1_dashboard = await analytics_engine.get_real_time_dashboard(merchant_1)
		global_dashboard = await analytics_engine.get_real_time_dashboard()
		
		# Merchant-specific should have different data than global
		assert merchant_1_dashboard != global_dashboard
		
		# Both should have valid structure
		for dashboard in [merchant_1_dashboard, global_dashboard]:
			assert "metrics" in dashboard
			assert "charts" in dashboard
			assert "alerts" in dashboard
			assert "summary" in dashboard


class TestRealTimeMetric:
	"""Test RealTimeMetric class"""
	
	def test_metric_creation(self):
		"""Test creating a real-time metric"""
		timestamp = datetime.now(timezone.utc)
		
		metric = RealTimeMetric(
			metric_type=MetricType.TRANSACTION_VOLUME,
			value=1.0,
			timestamp=timestamp,
			metadata={"transaction_id": "test_123"},
			merchant_id="test_merchant"
		)
		
		assert metric.metric_type == MetricType.TRANSACTION_VOLUME
		assert metric.value == 1.0
		assert metric.timestamp == timestamp
		assert metric.metadata["transaction_id"] == "test_123"
		assert metric.merchant_id == "test_merchant"
	
	def test_metric_to_dict(self):
		"""Test converting metric to dictionary"""
		timestamp = datetime.now(timezone.utc)
		
		metric = RealTimeMetric(
			metric_type=MetricType.SUCCESS_RATE,
			value=0.95,
			timestamp=timestamp,
			metadata={"processor": "stripe"},
			processor_name="stripe"
		)
		
		metric_dict = metric.to_dict()
		
		assert metric_dict["metric_type"] == "success_rate"
		assert metric_dict["value"] == 0.95
		assert metric_dict["timestamp"] == timestamp.isoformat()
		assert metric_dict["metadata"]["processor"] == "stripe"
		assert metric_dict["processor_name"] == "stripe"


class TestAlert:
	"""Test Alert class"""
	
	def test_alert_creation(self):
		"""Test creating an alert"""
		timestamp = datetime.now(timezone.utc)
		
		alert = Alert(
			id="alert_123",
			severity=AlertSeverity.HIGH,
			title="High Fraud Rate",
			message="Fraud rate exceeded threshold",
			metric_type=MetricType.FRAUD_RATE,
			threshold_value=5.0,
			current_value=8.5,
			timestamp=timestamp,
			merchant_id="test_merchant"
		)
		
		assert alert.id == "alert_123"
		assert alert.severity == AlertSeverity.HIGH
		assert alert.title == "High Fraud Rate"
		assert alert.metric_type == MetricType.FRAUD_RATE
		assert alert.threshold_value == 5.0
		assert alert.current_value == 8.5
		assert alert.resolved is False
	
	def test_alert_to_dict(self):
		"""Test converting alert to dictionary"""
		timestamp = datetime.now(timezone.utc)
		
		alert = Alert(
			id="alert_456",
			severity=AlertSeverity.CRITICAL,
			title="System Down",
			message="Payment processor unavailable",
			metric_type=MetricType.SUCCESS_RATE,
			threshold_value=90.0,
			current_value=45.0,
			timestamp=timestamp
		)
		
		alert_dict = alert.to_dict()
		
		assert alert_dict["id"] == "alert_456"
		assert alert_dict["severity"] == "critical"
		assert alert_dict["title"] == "System Down"
		assert alert_dict["metric_type"] == "success_rate"
		assert alert_dict["threshold_value"] == 90.0
		assert alert_dict["current_value"] == 45.0
		assert alert_dict["resolved"] is False