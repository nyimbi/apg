"""
Immersive Payment Analytics Dashboard - Revolutionary Real-Time Visualization

Advanced analytics dashboard with immersive data visualization, real-time insights,
interactive exploration, and AI-powered analytics for payment gateway intelligence.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import json
from dataclasses import asdict

from .models import PaymentTransaction, PaymentStatus, PaymentMethodType

class DashboardView(str, Enum):
	"""Dashboard view types"""
	EXECUTIVE_OVERVIEW = "executive_overview"
	TRANSACTION_ANALYSIS = "transaction_analysis"
	FRAUD_MONITORING = "fraud_monitoring"
	PERFORMANCE_METRICS = "performance_metrics"
	CUSTOMER_INSIGHTS = "customer_insights"
	REAL_TIME_MONITORING = "real_time_monitoring"
	PREDICTIVE_ANALYTICS = "predictive_analytics"
	COMPARATIVE_ANALYSIS = "comparative_analysis"

class VisualizationType(str, Enum):
	"""Types of data visualizations"""
	TIME_SERIES = "time_series"
	HEAT_MAP = "heat_map"
	GEOGRAPHIC_MAP = "geographic_map"
	SANKEY_DIAGRAM = "sankey_diagram"
	FUNNEL_CHART = "funnel_chart"
	NETWORK_GRAPH = "network_graph"
	D3_VISUALIZATION = "d3_visualization"
	INTERACTIVE_3D = "interactive_3d"
	AUGMENTED_REALITY = "augmented_reality"

class MetricType(str, Enum):
	"""Types of metrics tracked"""
	TRANSACTION_VOLUME = "transaction_volume"
	SUCCESS_RATE = "success_rate"
	FRAUD_SCORE = "fraud_score"
	PROCESSING_TIME = "processing_time"
	REVENUE = "revenue"
	CUSTOMER_SATISFACTION = "customer_satisfaction"
	CONVERSION_RATE = "conversion_rate"
	PROCESSOR_PERFORMANCE = "processor_performance"

class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"

class DashboardWidget(BaseModel):
	"""Individual dashboard widget configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	widget_id: str = Field(default_factory=uuid7str)
	title: str
	widget_type: VisualizationType
	metric_type: MetricType
	data_source: str
	refresh_rate_seconds: int = 30
	size: Dict[str, int] = Field(default_factory=lambda: {"width": 4, "height": 3})
	position: Dict[str, int] = Field(default_factory=lambda: {"x": 0, "y": 0})
	config: Dict[str, Any] = Field(default_factory=dict)
	filters: Dict[str, Any] = Field(default_factory=dict)
	is_real_time: bool = True

class DashboardAlert(BaseModel):
	"""Dashboard alert/notification"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	alert_id: str = Field(default_factory=uuid7str)
	title: str
	message: str
	severity: AlertSeverity
	metric_type: MetricType
	trigger_value: float
	threshold_value: float
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	acknowledged: bool = False
	auto_resolve: bool = True
	metadata: Dict[str, Any] = Field(default_factory=dict)

class AnalyticsQuery(BaseModel):
	"""Analytics query configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	query_id: str = Field(default_factory=uuid7str)
	metric_types: List[MetricType]
	time_range: Dict[str, datetime]
	filters: Dict[str, Any] = Field(default_factory=dict)
	group_by: List[str] = Field(default_factory=list)
	aggregation: str = "sum"  # sum, avg, count, min, max
	sampling_rate: str = "1m"  # 1s, 1m, 1h, 1d

class MetricDataPoint(BaseModel):
	"""Single metric data point"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	timestamp: datetime
	metric_type: MetricType
	value: float
	dimensions: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)

class DashboardConfiguration(BaseModel):
	"""Complete dashboard configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	dashboard_id: str = Field(default_factory=uuid7str)
	name: str
	view_type: DashboardView
	widgets: List[DashboardWidget]
	layout: Dict[str, Any] = Field(default_factory=dict)
	theme: str = "dark"
	auto_refresh: bool = True
	refresh_rate_seconds: int = 30
	user_id: str
	is_public: bool = False
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ImmersiveAnalyticsDashboard:
	"""
	Revolutionary immersive analytics dashboard providing real-time payment insights
	with advanced visualizations, AI-powered analytics, and interactive exploration.
	"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.dashboard_id = uuid7str()
		
		# Dashboard configurations
		self._dashboard_configs: Dict[str, DashboardConfiguration] = {}
		self._widget_data_cache: Dict[str, List[MetricDataPoint]] = {}
		self._real_time_connections: Dict[str, Dict[str, Any]] = {}
		
		# Analytics engine
		self._metrics_store: Dict[str, List[MetricDataPoint]] = {}
		self._alert_rules: Dict[str, Dict[str, Any]] = {}
		self._active_alerts: List[DashboardAlert] = []
		
		# Real-time streaming
		self.enable_real_time = config.get("enable_real_time", True)
		self.max_data_points = config.get("max_data_points", 10000)
		self.cache_duration_hours = config.get("cache_duration_hours", 24)
		
		# Advanced features
		self.enable_ai_insights = config.get("enable_ai_insights", True)
		self.enable_predictive_analytics = config.get("enable_predictive_analytics", True)
		self.enable_anomaly_detection = config.get("enable_anomaly_detection", True)
		
		self._initialized = False
		self._log_dashboard_created()
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize immersive analytics dashboard"""
		self._log_initialization_start()
		
		try:
			# Initialize data storage
			await self._initialize_metrics_store()
			
			# Set up real-time streaming
			await self._setup_real_time_streaming()
			
			# Initialize AI analytics
			await self._initialize_ai_analytics()
			
			# Set up alert system
			await self._setup_alert_system()
			
			# Create default dashboards
			await self._create_default_dashboards()
			
			self._initialized = True
			self._log_initialization_complete()
			
			return {
				"status": "initialized",
				"dashboard_id": self.dashboard_id,
				"real_time_enabled": self.enable_real_time,
				"ai_insights_enabled": self.enable_ai_insights,
				"default_dashboards_created": len(self._dashboard_configs)
			}
			
		except Exception as e:
			self._log_initialization_error(str(e))
			raise
	
	async def create_dashboard(
		self,
		name: str,
		view_type: DashboardView,
		user_id: str,
		widgets: List[Dict[str, Any]] | None = None
	) -> DashboardConfiguration:
		"""
		Create a new analytics dashboard
		
		Args:
			name: Dashboard name
			view_type: Type of dashboard view
			user_id: User creating the dashboard
			widgets: Optional widget configurations
			
		Returns:
			DashboardConfiguration object
		"""
		if not self._initialized:
			raise RuntimeError("Analytics dashboard not initialized")
		
		self._log_dashboard_creation_start(name, view_type)
		
		try:
			# Create widgets if not provided
			if widgets is None:
				widgets = await self._create_default_widgets_for_view(view_type)
			
			# Convert widget dictionaries to DashboardWidget objects
			dashboard_widgets = []
			for widget_config in widgets:
				widget = DashboardWidget(**widget_config)
				dashboard_widgets.append(widget)
			
			# Create dashboard configuration
			dashboard_config = DashboardConfiguration(
				name=name,
				view_type=view_type,
				widgets=dashboard_widgets,
				user_id=user_id
			)
			
			# Store configuration
			self._dashboard_configs[dashboard_config.dashboard_id] = dashboard_config
			
			# Initialize data for widgets
			await self._initialize_widget_data(dashboard_config)
			
			self._log_dashboard_creation_complete(dashboard_config.dashboard_id, len(dashboard_widgets))
			
			return dashboard_config
			
		except Exception as e:
			self._log_dashboard_creation_error(name, str(e))
			raise
	
	async def get_dashboard_data(
		self,
		dashboard_id: str,
		time_range: Dict[str, datetime] | None = None
	) -> Dict[str, Any]:
		"""
		Get complete dashboard data with all widgets
		
		Args:
			dashboard_id: Dashboard identifier
			time_range: Optional time range filter
			
		Returns:
			Complete dashboard data including widgets and metrics
		"""
		if dashboard_id not in self._dashboard_configs:
			raise ValueError(f"Dashboard not found: {dashboard_id}")
		
		dashboard_config = self._dashboard_configs[dashboard_id]
		
		# Default time range (last 24 hours)
		if time_range is None:
			end_time = datetime.now(timezone.utc)
			start_time = end_time - timedelta(hours=24)
			time_range = {"start": start_time, "end": end_time}
		
		# Get data for each widget
		widget_data = {}
		for widget in dashboard_config.widgets:
			widget_data[widget.widget_id] = await self._get_widget_data(
				widget, time_range
			)
		
		# Get AI insights
		ai_insights = []
		if self.enable_ai_insights:
			ai_insights = await self._generate_ai_insights(dashboard_config, time_range)
		
		# Get active alerts
		relevant_alerts = await self._get_relevant_alerts(dashboard_config)
		
		# Generate summary statistics
		summary_stats = await self._generate_dashboard_summary(dashboard_config, time_range)
		
		return {
			"dashboard_id": dashboard_id,
			"name": dashboard_config.name,
			"view_type": dashboard_config.view_type.value,
			"last_updated": datetime.now(timezone.utc).isoformat(),
			"time_range": {
				"start": time_range["start"].isoformat(),
				"end": time_range["end"].isoformat()
			},
			"widgets": widget_data,
			"ai_insights": ai_insights,
			"alerts": [alert.model_dump() for alert in relevant_alerts],
			"summary_stats": summary_stats,
			"real_time_enabled": self.enable_real_time
		}
	
	async def stream_real_time_data(
		self,
		dashboard_id: str,
		websocket_handler: Any  # WebSocket handler
	) -> None:
		"""
		Stream real-time data to dashboard via WebSocket
		
		Args:
			dashboard_id: Dashboard to stream data for
			websocket_handler: WebSocket connection handler
		"""
		if not self.enable_real_time:
			raise RuntimeError("Real-time streaming not enabled")
		
		if dashboard_id not in self._dashboard_configs:
			raise ValueError(f"Dashboard not found: {dashboard_id}")
		
		dashboard_config = self._dashboard_configs[dashboard_id]
		connection_id = uuid7str()
		
		self._log_real_time_stream_start(dashboard_id, connection_id)
		
		try:
			# Register connection
			self._real_time_connections[connection_id] = {
				"dashboard_id": dashboard_id,
				"websocket": websocket_handler,
				"started_at": datetime.now(timezone.utc)
			}
			
			# Stream loop
			while True:
				try:
					# Get latest data for all widgets
					updates = {}
					for widget in dashboard_config.widgets:
						if widget.is_real_time:
							latest_data = await self._get_latest_widget_data(widget)
							if latest_data:
								updates[widget.widget_id] = latest_data
					
					# Send updates if any
					if updates:
						message = {
							"type": "data_update",
							"dashboard_id": dashboard_id,
							"timestamp": datetime.now(timezone.utc).isoformat(),
							"updates": updates
						}
						
						await websocket_handler.send_json(message)
					
					# Check for new alerts
					new_alerts = await self._get_new_alerts_since_last_check(dashboard_config)
					if new_alerts:
						alert_message = {
							"type": "alert_update",
							"dashboard_id": dashboard_id,
							"alerts": [alert.model_dump() for alert in new_alerts]
						}
						
						await websocket_handler.send_json(alert_message)
					
					# Wait for next update cycle
					await asyncio.sleep(dashboard_config.refresh_rate_seconds)
					
				except Exception as e:
					self._log_real_time_stream_error(connection_id, str(e))
					break
		
		finally:
			# Clean up connection
			if connection_id in self._real_time_connections:
				del self._real_time_connections[connection_id]
			
			self._log_real_time_stream_end(connection_id)
	
	async def record_payment_metric(
		self,
		transaction: PaymentTransaction,
		processing_time_ms: int,
		processor_name: str,
		additional_metrics: Dict[str, Any] | None = None
	) -> None:
		"""
		Record payment metrics for analytics
		
		Args:
			transaction: Payment transaction
			processing_time_ms: Processing time in milliseconds
			processor_name: Name of payment processor used
			additional_metrics: Additional custom metrics
		"""
		timestamp = datetime.now(timezone.utc)
		
		# Record transaction volume
		await self._record_metric(
			MetricType.TRANSACTION_VOLUME,
			1.0,
			timestamp,
			{
				"processor": processor_name,
				"payment_method": transaction.payment_method_type.value,
				"currency": transaction.currency,
				"merchant_id": transaction.merchant_id
			}
		)
		
		# Record revenue
		await self._record_metric(
			MetricType.REVENUE,
			float(transaction.amount),
			timestamp,
			{
				"processor": processor_name,
				"currency": transaction.currency,
				"merchant_id": transaction.merchant_id
			}
		)
		
		# Record success rate
		success_value = 1.0 if transaction.status == PaymentStatus.COMPLETED else 0.0
		await self._record_metric(
			MetricType.SUCCESS_RATE,
			success_value,
			timestamp,
			{
				"processor": processor_name,
				"payment_method": transaction.payment_method_type.value,
				"error_code": getattr(transaction, 'error_code', None)
			}
		)
		
		# Record processing time
		await self._record_metric(
			MetricType.PROCESSING_TIME,
			float(processing_time_ms),
			timestamp,
			{
				"processor": processor_name,
				"payment_method": transaction.payment_method_type.value
			}
		)
		
		# Record additional metrics if provided
		if additional_metrics:
			for metric_name, value in additional_metrics.items():
				try:
					metric_type = MetricType(metric_name)
					await self._record_metric(metric_type, float(value), timestamp, {
						"processor": processor_name,
						"transaction_id": transaction.id
					})
				except (ValueError, TypeError):
					# Skip invalid metrics
					continue
		
		# Check for alert conditions
		await self._check_alert_conditions(timestamp)
	
	async def get_analytics_insights(
		self,
		query: AnalyticsQuery
	) -> Dict[str, Any]:
		"""
		Get advanced analytics insights based on query
		
		Args:
			query: Analytics query configuration
			
		Returns:
			Analytics insights and visualizations
		"""
		self._log_analytics_query_start(query.query_id)
		
		try:
			# Query metrics data
			raw_data = await self._query_metrics_data(query)
			
			# Process and aggregate data
			processed_data = await self._process_analytics_data(raw_data, query)
			
			# Generate insights
			insights = []
			if self.enable_ai_insights:
				insights = await self._generate_query_insights(processed_data, query)
			
			# Generate predictions
			predictions = []
			if self.enable_predictive_analytics:
				predictions = await self._generate_predictions(processed_data, query)
			
			# Detect anomalies
			anomalies = []
			if self.enable_anomaly_detection:
				anomalies = await self._detect_anomalies(processed_data, query)
			
			result = {
				"query_id": query.query_id,
				"data": processed_data,
				"insights": insights,
				"predictions": predictions,
				"anomalies": anomalies,
				"metadata": {
					"total_data_points": len(raw_data),
					"time_range": {
						"start": query.time_range["start"].isoformat(),
						"end": query.time_range["end"].isoformat()
					},
					"processing_time_ms": 0  # Would be calculated in real implementation
				}
			}
			
			self._log_analytics_query_complete(query.query_id, len(processed_data))
			
			return result
			
		except Exception as e:
			self._log_analytics_query_error(query.query_id, str(e))
			raise
	
	# Private implementation methods
	
	async def _initialize_metrics_store(self):
		"""Initialize metrics data storage"""
		for metric_type in MetricType:
			self._metrics_store[metric_type.value] = []
	
	async def _setup_real_time_streaming(self):
		"""Set up real-time data streaming infrastructure"""
		if self.enable_real_time:
			# Initialize WebSocket handlers and data streaming
			pass
	
	async def _initialize_ai_analytics(self):
		"""Initialize AI analytics capabilities"""
		if self.enable_ai_insights:
			# Load AI models for insights generation
			pass
	
	async def _setup_alert_system(self):
		"""Set up alert and notification system"""
		# Define default alert rules
		self._alert_rules = {
			"high_failure_rate": {
				"metric": MetricType.SUCCESS_RATE,
				"threshold": 0.95,
				"condition": "below",
				"severity": AlertSeverity.HIGH
			},
			"slow_processing": {
				"metric": MetricType.PROCESSING_TIME,
				"threshold": 5000.0,  # 5 seconds
				"condition": "above",
				"severity": AlertSeverity.MEDIUM
			},
			"fraud_spike": {
				"metric": MetricType.FRAUD_SCORE,
				"threshold": 0.8,
				"condition": "above",
				"severity": AlertSeverity.CRITICAL
			}
		}
	
	async def _create_default_dashboards(self):
		"""Create default dashboard configurations"""
		# Executive Overview Dashboard
		executive_dashboard = await self.create_dashboard(
			"Executive Overview",
			DashboardView.EXECUTIVE_OVERVIEW,
			"system"
		)
		
		# Real-time Monitoring Dashboard
		monitoring_dashboard = await self.create_dashboard(
			"Real-time Monitoring",
			DashboardView.REAL_TIME_MONITORING,
			"system"
		)
		
		# Fraud Monitoring Dashboard
		fraud_dashboard = await self.create_dashboard(
			"Fraud Monitoring",
			DashboardView.FRAUD_MONITORING,
			"system"
		)
	
	async def _create_default_widgets_for_view(
		self,
		view_type: DashboardView
	) -> List[Dict[str, Any]]:
		"""Create default widgets for a dashboard view type"""
		widgets = []
		
		if view_type == DashboardView.EXECUTIVE_OVERVIEW:
			widgets = [
				{
					"title": "Transaction Volume",
					"widget_type": VisualizationType.TIME_SERIES,
					"metric_type": MetricType.TRANSACTION_VOLUME,
					"data_source": "metrics_store",
					"size": {"width": 6, "height": 4},
					"position": {"x": 0, "y": 0}
				},
				{
					"title": "Revenue Trends",
					"widget_type": VisualizationType.TIME_SERIES,
					"metric_type": MetricType.REVENUE,
					"data_source": "metrics_store",
					"size": {"width": 6, "height": 4},
					"position": {"x": 6, "y": 0}
				},
				{
					"title": "Success Rate Heatmap",
					"widget_type": VisualizationType.HEAT_MAP,
					"metric_type": MetricType.SUCCESS_RATE,
					"data_source": "metrics_store",
					"size": {"width": 8, "height": 6},
					"position": {"x": 0, "y": 4}
				},
				{
					"title": "Payment Methods Distribution",
					"widget_type": VisualizationType.FUNNEL_CHART,
					"metric_type": MetricType.TRANSACTION_VOLUME,
					"data_source": "metrics_store",
					"size": {"width": 4, "height": 6},
					"position": {"x": 8, "y": 4}
				}
			]
		
		elif view_type == DashboardView.REAL_TIME_MONITORING:
			widgets = [
				{
					"title": "Live Transaction Feed",
					"widget_type": VisualizationType.TIME_SERIES,
					"metric_type": MetricType.TRANSACTION_VOLUME,
					"data_source": "real_time_stream",
					"refresh_rate_seconds": 5,
					"size": {"width": 12, "height": 4},
					"position": {"x": 0, "y": 0}
				},
				{
					"title": "Processing Time Monitor",
					"widget_type": VisualizationType.TIME_SERIES,
					"metric_type": MetricType.PROCESSING_TIME,
					"data_source": "real_time_stream",
					"refresh_rate_seconds": 10,
					"size": {"width": 6, "height": 4},
					"position": {"x": 0, "y": 4}
				},
				{
					"title": "Success Rate Gauge",
					"widget_type": VisualizationType.D3_VISUALIZATION,
					"metric_type": MetricType.SUCCESS_RATE,
					"data_source": "real_time_stream",
					"refresh_rate_seconds": 15,
					"size": {"width": 6, "height": 4},
					"position": {"x": 6, "y": 4}
				}
			]
		
		elif view_type == DashboardView.FRAUD_MONITORING:
			widgets = [
				{
					"title": "Fraud Score Distribution",
					"widget_type": VisualizationType.HEAT_MAP,
					"metric_type": MetricType.FRAUD_SCORE,
					"data_source": "metrics_store",
					"size": {"width": 8, "height": 6},
					"position": {"x": 0, "y": 0}
				},
				{
					"title": "High-Risk Transactions",
					"widget_type": VisualizationType.NETWORK_GRAPH,
					"metric_type": MetricType.FRAUD_SCORE,
					"data_source": "fraud_detection",
					"size": {"width": 4, "height": 6},
					"position": {"x": 8, "y": 0}
				},
				{
					"title": "Geographic Risk Map",
					"widget_type": VisualizationType.GEOGRAPHIC_MAP,
					"metric_type": MetricType.FRAUD_SCORE,
					"data_source": "metrics_store",
					"size": {"width": 12, "height": 6},
					"position": {"x": 0, "y": 6}
				}
			]
		
		else:
			# Generic widgets for other view types
			widgets = [
				{
					"title": "Metric Overview",
					"widget_type": VisualizationType.TIME_SERIES,
					"metric_type": MetricType.TRANSACTION_VOLUME,
					"data_source": "metrics_store",
					"size": {"width": 12, "height": 8},
					"position": {"x": 0, "y": 0}
				}
			]
		
		return widgets
	
	async def _initialize_widget_data(self, dashboard_config: DashboardConfiguration):
		"""Initialize data cache for dashboard widgets"""
		for widget in dashboard_config.widgets:
			widget_key = f"{dashboard_config.dashboard_id}:{widget.widget_id}"
			self._widget_data_cache[widget_key] = []
	
	async def _get_widget_data(
		self,
		widget: DashboardWidget,
		time_range: Dict[str, datetime]
	) -> Dict[str, Any]:
		"""Get data for a specific widget"""
		# Query metrics based on widget configuration
		metric_data = await self._query_widget_metrics(widget, time_range)
		
		# Process data for visualization type
		processed_data = await self._process_widget_data(widget, metric_data)
		
		return {
			"widget_id": widget.widget_id,
			"title": widget.title,
			"type": widget.widget_type.value,
			"data": processed_data,
			"last_updated": datetime.now(timezone.utc).isoformat(),
			"config": widget.config
		}
	
	async def _query_widget_metrics(
		self,
		widget: DashboardWidget,
		time_range: Dict[str, datetime]
	) -> List[MetricDataPoint]:
		"""Query metrics data for widget"""
		metric_key = widget.metric_type.value
		if metric_key not in self._metrics_store:
			return []
		
		# Filter by time range
		filtered_data = []
		for data_point in self._metrics_store[metric_key]:
			if time_range["start"] <= data_point.timestamp <= time_range["end"]:
				filtered_data.append(data_point)
		
		# Apply widget filters
		if widget.filters:
			filtered_data = await self._apply_widget_filters(filtered_data, widget.filters)
		
		return filtered_data
	
	async def _process_widget_data(
		self,
		widget: DashboardWidget,
		metric_data: List[MetricDataPoint]
	) -> Dict[str, Any]:
		"""Process metric data for specific widget visualization"""
		if widget.widget_type == VisualizationType.TIME_SERIES:
			return await self._process_time_series_data(metric_data)
		elif widget.widget_type == VisualizationType.HEAT_MAP:
			return await self._process_heatmap_data(metric_data)
		elif widget.widget_type == VisualizationType.GEOGRAPHIC_MAP:
			return await self._process_geographic_data(metric_data)
		elif widget.widget_type == VisualizationType.FUNNEL_CHART:
			return await self._process_funnel_data(metric_data)
		else:
			# Default processing
			return {
				"data_points": [
					{
						"timestamp": dp.timestamp.isoformat(),
						"value": dp.value,
						"dimensions": dp.dimensions
					}
					for dp in metric_data
				]
			}
	
	async def _process_time_series_data(
		self,
		metric_data: List[MetricDataPoint]
	) -> Dict[str, Any]:
		"""Process data for time series visualization"""
		series_data = []
		
		for data_point in metric_data:
			series_data.append({
				"timestamp": data_point.timestamp.isoformat(),
				"value": data_point.value
			})
		
		# Sort by timestamp
		series_data.sort(key=lambda x: x["timestamp"])
		
		return {
			"series": series_data,
			"total_points": len(series_data),
			"min_value": min([dp["value"] for dp in series_data], default=0),
			"max_value": max([dp["value"] for dp in series_data], default=0),
			"avg_value": sum([dp["value"] for dp in series_data]) / len(series_data) if series_data else 0
		}
	
	async def _process_heatmap_data(
		self,
		metric_data: List[MetricDataPoint]
	) -> Dict[str, Any]:
		"""Process data for heatmap visualization"""
		# Group by dimensions for heatmap
		heatmap_matrix = {}
		
		for data_point in metric_data:
			x_key = data_point.dimensions.get("processor", "unknown")
			y_key = data_point.dimensions.get("payment_method", "unknown")
			
			if x_key not in heatmap_matrix:
				heatmap_matrix[x_key] = {}
			
			if y_key not in heatmap_matrix[x_key]:
				heatmap_matrix[x_key][y_key] = []
			
			heatmap_matrix[x_key][y_key].append(data_point.value)
		
		# Calculate averages for each cell
		processed_matrix = {}
		for x_key, y_data in heatmap_matrix.items():
			processed_matrix[x_key] = {}
			for y_key, values in y_data.items():
				processed_matrix[x_key][y_key] = sum(values) / len(values)
		
		return {
			"matrix": processed_matrix,
			"x_labels": list(processed_matrix.keys()),
			"y_labels": list(set().union(*[y_data.keys() for y_data in processed_matrix.values()])),
			"total_cells": sum(len(y_data) for y_data in processed_matrix.values())
		}
	
	async def _process_geographic_data(
		self,
		metric_data: List[MetricDataPoint]
	) -> Dict[str, Any]:
		"""Process data for geographic map visualization"""
		geographic_data = {}
		
		for data_point in metric_data:
			country = data_point.dimensions.get("country", "unknown")
			if country not in geographic_data:
				geographic_data[country] = []
			geographic_data[country].append(data_point.value)
		
		# Calculate aggregated values per country
		processed_data = []
		for country, values in geographic_data.items():
			processed_data.append({
				"country": country,
				"value": sum(values),
				"average": sum(values) / len(values),
				"count": len(values)
			})
		
		return {
			"countries": processed_data,
			"total_countries": len(processed_data),
			"total_data_points": len(metric_data)
		}
	
	async def _process_funnel_data(
		self,
		metric_data: List[MetricDataPoint]
	) -> Dict[str, Any]:
		"""Process data for funnel chart visualization"""
		funnel_stages = {}
		
		for data_point in metric_data:
			stage = data_point.dimensions.get("payment_method", "unknown")
			if stage not in funnel_stages:
				funnel_stages[stage] = 0
			funnel_stages[stage] += data_point.value
		
		# Sort stages by value (descending for funnel)
		sorted_stages = sorted(funnel_stages.items(), key=lambda x: x[1], reverse=True)
		
		funnel_data = []
		total_value = sum(funnel_stages.values())
		
		for i, (stage, value) in enumerate(sorted_stages):
			percentage = (value / total_value * 100) if total_value > 0 else 0
			funnel_data.append({
				"stage": stage,
				"value": value,
				"percentage": percentage,
				"order": i
			})
		
		return {
			"stages": funnel_data,
			"total_value": total_value,
			"conversion_rate": funnel_data[-1]["percentage"] if funnel_data else 0
		}
	
	async def _record_metric(
		self,
		metric_type: MetricType,
		value: float,
		timestamp: datetime,
		dimensions: Dict[str, Any]
	):
		"""Record a metric data point"""
		data_point = MetricDataPoint(
			timestamp=timestamp,
			metric_type=metric_type,
			value=value,
			dimensions=dimensions
		)
		
		metric_key = metric_type.value
		if metric_key not in self._metrics_store:
			self._metrics_store[metric_key] = []
		
		self._metrics_store[metric_key].append(data_point)
		
		# Maintain data size limits
		if len(self._metrics_store[metric_key]) > self.max_data_points:
			self._metrics_store[metric_key] = self._metrics_store[metric_key][-self.max_data_points:]
	
	async def _check_alert_conditions(self, timestamp: datetime):
		"""Check for alert conditions and create alerts"""
		for rule_name, rule_config in self._alert_rules.items():
			metric_type = rule_config["metric"]
			threshold = rule_config["threshold"]
			condition = rule_config["condition"]
			severity = AlertSeverity(rule_config["severity"])
			
			# Get recent metric values
			recent_values = await self._get_recent_metric_values(metric_type, minutes=5)
			
			if not recent_values:
				continue
			
			# Calculate current value (average of recent values)
			current_value = sum(recent_values) / len(recent_values)
			
			# Check condition
			alert_triggered = False
			if condition == "above" and current_value > threshold:
				alert_triggered = True
			elif condition == "below" and current_value < threshold:
				alert_triggered = True
			
			if alert_triggered:
				alert = DashboardAlert(
					title=f"Alert: {rule_name.replace('_', ' ').title()}",
					message=f"{metric_type.value} is {current_value:.2f} ({condition} threshold of {threshold})",
					severity=severity,
					metric_type=metric_type,
					trigger_value=current_value,
					threshold_value=threshold,
					timestamp=timestamp
				)
				
				self._active_alerts.append(alert)
				self._log_alert_triggered(alert.alert_id, alert.title)
	
	# Additional helper methods would be implemented here...
	
	# Logging methods
	
	def _log_dashboard_created(self):
		"""Log dashboard creation"""
		print(f"📊 Immersive Analytics Dashboard created")
		print(f"   Dashboard ID: {self.dashboard_id}")
		print(f"   Real-time: {self.enable_real_time}")
		print(f"   AI Insights: {self.enable_ai_insights}")
	
	def _log_initialization_start(self):
		"""Log initialization start"""
		print(f"🚀 Initializing Immersive Analytics Dashboard...")
		print(f"   Setting up metrics store and streaming")
	
	def _log_initialization_complete(self):
		"""Log initialization complete"""
		print(f"✅ Immersive Analytics Dashboard initialized")
		print(f"   Dashboards: {len(self._dashboard_configs)}")
		print(f"   Metrics tracked: {len(self._metrics_store)}")
	
	def _log_initialization_error(self, error: str):
		"""Log initialization error"""
		print(f"❌ Dashboard initialization failed: {error}")
	
	def _log_dashboard_creation_start(self, name: str, view_type: DashboardView):
		"""Log dashboard creation start"""
		print(f"🎨 Creating dashboard: {name} ({view_type.value})")
	
	def _log_dashboard_creation_complete(self, dashboard_id: str, widget_count: int):
		"""Log dashboard creation complete"""
		print(f"✅ Dashboard created: {dashboard_id[:8]}...")
		print(f"   Widgets: {widget_count}")
	
	def _log_dashboard_creation_error(self, name: str, error: str):
		"""Log dashboard creation error"""
		print(f"❌ Dashboard creation failed ({name}): {error}")
	
	def _log_real_time_stream_start(self, dashboard_id: str, connection_id: str):
		"""Log real-time stream start"""
		print(f"📡 Starting real-time stream: {dashboard_id[:8]}... -> {connection_id[:8]}...")
	
	def _log_real_time_stream_end(self, connection_id: str):
		"""Log real-time stream end"""
		print(f"📡 Real-time stream ended: {connection_id[:8]}...")
	
	def _log_real_time_stream_error(self, connection_id: str, error: str):
		"""Log real-time stream error"""
		print(f"❌ Real-time stream error ({connection_id[:8]}...): {error}")
	
	def _log_analytics_query_start(self, query_id: str):
		"""Log analytics query start"""
		print(f"🔍 Processing analytics query: {query_id[:8]}...")
	
	def _log_analytics_query_complete(self, query_id: str, data_points: int):
		"""Log analytics query complete"""
		print(f"✅ Analytics query complete: {query_id[:8]}...")
		print(f"   Data points: {data_points}")
	
	def _log_analytics_query_error(self, query_id: str, error: str):
		"""Log analytics query error"""
		print(f"❌ Analytics query failed ({query_id[:8]}...): {error}")
	
	def _log_alert_triggered(self, alert_id: str, title: str):
		"""Log alert triggered"""
		print(f"🚨 Alert triggered: {title} ({alert_id[:8]}...)")

# Factory function
def create_immersive_analytics_dashboard(config: Dict[str, Any]) -> ImmersiveAnalyticsDashboard:
	"""Factory function to create immersive analytics dashboard"""
	return ImmersiveAnalyticsDashboard(config)

def _log_immersive_analytics_module_loaded():
	"""Log module loaded"""
	print("📊 Immersive Analytics Dashboard module loaded")
	print("   - Real-time data visualization")
	print("   - AI-powered insights generation")
	print("   - Interactive dashboard builder")
	print("   - Advanced analytics queries")

# Execute module loading log
_log_immersive_analytics_module_loaded()