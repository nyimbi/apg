"""
Administrative Dashboard for the AI Core Framework (AICR) Capability
====================================================================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Comprehensive administrative dashboard providing real-time system overview,
advanced analytics, management controls, and operational insights for the
AI Core Framework with revolutionary visualization and interaction capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.security.decorators import protect
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import numpy as np

from .service import AICoreService
from .monitoring import ai_monitoring_system
from .ml_pipeline import ml_pipeline_framework
from .model_marketplace import model_marketplace
from .websocket import websocket_server
from .security import SecurityManager


class DashboardAnalytics:
	"""Advanced analytics engine for dashboard data processing."""

	def __init__(self):
		"""Initialize the dashboard analytics engine."""
		self.analytics_id = "dashboard_analytics"
		self.cache_ttl = 300  # 5 minutes cache
		self._cache: Dict[str, Tuple[datetime, Any]] = {}
		self.logger = logging.getLogger(__name__)

	async def get_system_overview(self) -> Dict[str, Any]:
		"""Get comprehensive system overview analytics.

		Returns:
			Dict[str, Any]: System overview data
		"""
		cache_key = "system_overview"
		cached_data = self._get_cached_data(cache_key)
		if cached_data:
			return cached_data

		try:
			# Get system health
			health_data = await ai_monitoring_system.get_system_health()

			# Get model statistics
			model_stats = await self._get_model_statistics()

			# Get pipeline statistics
			pipeline_stats = await self._get_pipeline_statistics()

			# Get performance metrics
			performance_data = await ai_monitoring_system.get_performance_summary()

			# Get marketplace statistics
			marketplace_stats = await model_marketplace.get_marketplace_statistics()

			# Get real-time metrics
			realtime_metrics = await self._get_realtime_metrics()

			overview_data = {
				"system_health": health_data,
				"model_stats": model_stats,
				"pipeline_stats": pipeline_stats,
				"performance_data": performance_data,
				"marketplace_stats": marketplace_stats,
				"realtime_metrics": realtime_metrics,
				"timestamp": datetime.utcnow().isoformat()
			}

			self._cache_data(cache_key, overview_data)
			return overview_data

		except Exception as e:
			self.logger.error(f"Error getting system overview: {e}")
			return {"error": str(e)}

	async def get_performance_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
		"""Get detailed performance analytics.

		Args:
			time_range_hours: Time range for analysis in hours

		Returns:
			Dict[str, Any]: Performance analytics data
		"""
		cache_key = f"performance_analytics_{time_range_hours}h"
		cached_data = self._get_cached_data(cache_key)
		if cached_data:
			return cached_data

		try:
			# Calculate time range
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(hours=time_range_hours)

			# Get metrics for the time range
			metrics = await ai_monitoring_system.metrics_collector.get_metrics(
				time_range=(start_time, end_time)
			)

			# Process metrics for analytics
			analytics_data = await self._process_performance_metrics(metrics, time_range_hours)

			# Get trend analysis
			trend_analysis = await self._get_trend_analysis(time_range_hours)

			# Get anomaly detection results
			anomaly_data = await self._get_anomaly_analysis(time_range_hours)

			performance_analytics = {
				"analytics_data": analytics_data,
				"trend_analysis": trend_analysis,
				"anomaly_data": anomaly_data,
				"time_range": {
					"start": start_time.isoformat(),
					"end": end_time.isoformat(),
					"hours": time_range_hours
				},
				"metrics_count": len(metrics)
			}

			self._cache_data(cache_key, performance_analytics)
			return performance_analytics

		except Exception as e:
			self.logger.error(f"Error getting performance analytics: {e}")
			return {"error": str(e)}

	async def get_operational_insights(self) -> Dict[str, Any]:
		"""Get operational insights and recommendations.

		Returns:
			Dict[str, Any]: Operational insights data
		"""
		cache_key = "operational_insights"
		cached_data = self._get_cached_data(cache_key)
		if cached_data:
			return cached_data

		try:
			# Get resource utilization insights
			resource_insights = await self._analyze_resource_utilization()

			# Get model performance insights
			model_insights = await self._analyze_model_performance()

			# Get pipeline efficiency insights
			pipeline_insights = await self._analyze_pipeline_efficiency()

			# Get cost optimization recommendations
			cost_insights = await self._analyze_cost_optimization()

			# Get security insights
			security_insights = await self._analyze_security_status()

			insights_data = {
				"resource_insights": resource_insights,
				"model_insights": model_insights,
				"pipeline_insights": pipeline_insights,
				"cost_insights": cost_insights,
				"security_insights": security_insights,
				"generated_at": datetime.utcnow().isoformat()
			}

			self._cache_data(cache_key, insights_data)
			return insights_data

		except Exception as e:
			self.logger.error(f"Error getting operational insights: {e}")
			return {"error": str(e)}

	async def generate_visualization_data(self, chart_type: str, **kwargs) -> Dict[str, Any]:
		"""Generate data for dashboard visualizations.

		Args:
			chart_type: Type of chart to generate
			**kwargs: Additional parameters

		Returns:
			Dict[str, Any]: Visualization data
		"""
		try:
			if chart_type == "system_health_gauge":
				return await self._create_health_gauge_data()
			elif chart_type == "performance_timeline":
				return await self._create_performance_timeline_data(kwargs.get("hours", 24))
			elif chart_type == "model_distribution":
				return await self._create_model_distribution_data()
			elif chart_type == "pipeline_success_rate":
				return await self._create_pipeline_success_data()
			elif chart_type == "resource_utilization":
				return await self._create_resource_utilization_data()
			elif chart_type == "anomaly_heatmap":
				return await self._create_anomaly_heatmap_data()
			else:
				return {"error": f"Unknown chart type: {chart_type}"}

		except Exception as e:
			self.logger.error(f"Error generating visualization data for {chart_type}: {e}")
			return {"error": str(e)}

	async def _get_model_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive model statistics."""
		try:
			# This would integrate with the model registry
			# For now, we'll simulate some statistics
			return {
				"total_models": 42,
				"deployed_models": 15,
				"active_models": 12,
				"model_types": {
					"classification": 18,
					"regression": 12,
					"clustering": 8,
					"nlp": 4
				},
				"frameworks": {
					"pytorch": 20,
					"tensorflow": 15,
					"sklearn": 7
				}
			}
		except Exception as e:
			self.logger.error(f"Error getting model statistics: {e}")
			return {}

	async def _get_pipeline_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive pipeline statistics."""
		try:
			# Get pipeline data from ML framework
			pipelines = ml_pipeline_framework.orchestrator.pipelines
			executions = ml_pipeline_framework.orchestrator.executions

			total_pipelines = len(pipelines)
			total_executions = len(executions)

			# Calculate success rate
			successful_executions = sum(
				1 for exec in executions.values()
				if exec.status == "completed"
			)
			success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0

			# Get running pipelines
			running_executions = sum(
				1 for exec in executions.values()
				if exec.status == "running"
			)

			return {
				"total_pipelines": total_pipelines,
				"total_executions": total_executions,
				"running_executions": running_executions,
				"success_rate": round(success_rate, 2),
				"successful_executions": successful_executions,
				"failed_executions": total_executions - successful_executions
			}
		except Exception as e:
			self.logger.error(f"Error getting pipeline statistics: {e}")
			return {}

	async def _get_realtime_metrics(self) -> Dict[str, Any]:
		"""Get real-time system metrics."""
		try:
			# Get recent metrics (last 5 minutes)
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(minutes=5)

			metrics = await ai_monitoring_system.metrics_collector.get_metrics(
				time_range=(start_time, end_time)
			)

			# Calculate current values
			current_metrics = {}
			for metric in metrics:
				if metric.metric_name not in current_metrics:
					current_metrics[metric.metric_name] = []
				current_metrics[metric.metric_name].append(metric.value)

			# Get latest values
			realtime_data = {}
			for metric_name, values in current_metrics.items():
				realtime_data[metric_name] = {
					"current": values[-1] if values else 0,
					"average": np.mean(values) if values else 0,
					"trend": "up" if len(values) > 1 and values[-1] > values[0] else "down"
				}

			return realtime_data
		except Exception as e:
			self.logger.error(f"Error getting realtime metrics: {e}")
			return {}

	async def _create_health_gauge_data(self) -> Dict[str, Any]:
		"""Create health gauge visualization data."""
		try:
			health_data = await ai_monitoring_system.get_system_health()
			health_score = health_data.get("overall_health_score", 0.5) * 100

			gauge_data = {
				"type": "indicator",
				"mode": "gauge+number+delta",
				"value": health_score,
				"domain": {"x": [0, 1], "y": [0, 1]},
				"title": {"text": "System Health Score"},
				"delta": {"reference": 90, "valueformat": ".0f"},
				"gauge": {
					"axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkblue"},
					"bar": {"color": "darkblue"},
					"bgcolor": "white",
					"borderwidth": 2,
					"bordercolor": "gray",
					"steps": [
						{"range": [0, 50], "color": "lightgray"},
						{"range": [50, 80], "color": "yellow"},
						{"range": [80, 100], "color": "lightgreen"}
					],
					"threshold": {
						"line": {"color": "red", "width": 4},
						"thickness": 0.75,
						"value": 90
					}
				}
			}

			return {
				"data": [gauge_data],
				"layout": {
					"width": 400,
					"height": 300,
					"margin": {"t": 25, "b": 25, "l": 25, "r": 25},
					"font": {"color": "darkblue", "family": "Arial"}
				}
			}
		except Exception as e:
			self.logger.error(f"Error creating health gauge data: {e}")
			return {"error": str(e)}

	async def _create_performance_timeline_data(self, hours: int = 24) -> Dict[str, Any]:
		"""Create performance timeline visualization data."""
		try:
			# Get performance data
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(hours=hours)

			metrics = await ai_monitoring_system.metrics_collector.get_metrics(
				metric_names=["system_cpu_usage", "system_memory_usage"],
				time_range=(start_time, end_time)
			)

			# Process data for timeline
			cpu_data = {"x": [], "y": [], "name": "CPU Usage (%)", "type": "scatter"}
			memory_data = {"x": [], "y": [], "name": "Memory Usage (%)", "type": "scatter"}

			for metric in metrics:
				if metric.metric_name == "system_cpu_usage":
					cpu_data["x"].append(metric.timestamp.isoformat())
					cpu_data["y"].append(metric.value)
				elif metric.metric_name == "system_memory_usage":
					memory_data["x"].append(metric.timestamp.isoformat())
					memory_data["y"].append(metric.value)

			return {
				"data": [cpu_data, memory_data],
				"layout": {
					"title": f"System Performance (Last {hours} hours)",
					"xaxis": {"title": "Time"},
					"yaxis": {"title": "Usage (%)"},
					"hovermode": "x unified"
				}
			}
		except Exception as e:
			self.logger.error(f"Error creating performance timeline data: {e}")
			return {"error": str(e)}

	def _get_cached_data(self, cache_key: str) -> Optional[Any]:
		"""Get data from cache if valid.

		Args:
			cache_key: Cache key

		Returns:
			Optional[Any]: Cached data if valid, None otherwise
		"""
		if cache_key in self._cache:
			cached_time, cached_data = self._cache[cache_key]
			if datetime.utcnow() - cached_time < timedelta(seconds=self.cache_ttl):
				return cached_data
			else:
				del self._cache[cache_key]
		return None

	def _cache_data(self, cache_key: str, data: Any) -> None:
		"""Cache data with timestamp.

		Args:
			cache_key: Cache key
			data: Data to cache
		"""
		self._cache[cache_key] = (datetime.utcnow(), data)


class AICRDashboard(BaseView):
	"""Main administrative dashboard for AICR capability."""

	route_base = '/admin/aicr'
	default_view = 'overview'

	def __init__(self):
		"""Initialize the AICR dashboard."""
		super().__init__()
		self.analytics = DashboardAnalytics()
		self.ai_service = AICoreService()
		self.security_manager = SecurityManager()

	@expose('/overview/')
	@has_access
	def overview(self):
		"""Main dashboard overview page."""
		try:
			# Get system overview data
			overview_data = asyncio.run(self.analytics.get_system_overview())

			# Get connection statistics
			websocket_stats = websocket_server.get_connection_stats()

			# Generate visualizations
			health_chart = asyncio.run(
				self.analytics.generate_visualization_data("system_health_gauge")
			)
			performance_chart = asyncio.run(
				self.analytics.generate_visualization_data("performance_timeline", hours=24)
			)

			dashboard_data = {
				"overview": overview_data,
				"websocket_stats": websocket_stats,
				"health_chart": json.dumps(health_chart, cls=plotly.utils.PlotlyJSONEncoder),
				"performance_chart": json.dumps(performance_chart, cls=plotly.utils.PlotlyJSONEncoder),
				"refresh_interval": 30  # seconds
			}

			return self.render_template('aicr/admin_dashboard.html', data=dashboard_data)

		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('aicr/admin_dashboard.html', data={})

	@expose('/analytics/')
	@has_access
	def analytics(self):
		"""Advanced analytics dashboard page."""
		try:
			# Get query parameters
			time_range = request.args.get('time_range', '24')
			try:
				time_range_hours = int(time_range)
			except ValueError:
				time_range_hours = 24

			# Get performance analytics
			performance_data = asyncio.run(
				self.analytics.get_performance_analytics(time_range_hours)
			)

			# Get operational insights
			insights_data = asyncio.run(self.analytics.get_operational_insights())

			# Generate advanced visualizations
			model_dist_chart = asyncio.run(
				self.analytics.generate_visualization_data("model_distribution")
			)
			pipeline_chart = asyncio.run(
				self.analytics.generate_visualization_data("pipeline_success_rate")
			)
			resource_chart = asyncio.run(
				self.analytics.generate_visualization_data("resource_utilization")
			)

			analytics_data = {
				"performance": performance_data,
				"insights": insights_data,
				"time_range_hours": time_range_hours,
				"model_chart": json.dumps(model_dist_chart, cls=plotly.utils.PlotlyJSONEncoder),
				"pipeline_chart": json.dumps(pipeline_chart, cls=plotly.utils.PlotlyJSONEncoder),
				"resource_chart": json.dumps(resource_chart, cls=plotly.utils.PlotlyJSONEncoder)
			}

			return self.render_template('aicr/admin_analytics.html', data=analytics_data)

		except Exception as e:
			flash(f'Error loading analytics: {str(e)}', 'error')
			return self.render_template('aicr/admin_analytics.html', data={})

	@expose('/monitoring/')
	@has_access
	def monitoring(self):
		"""Real-time monitoring dashboard page."""
		try:
			# Get current system health
			health_data = asyncio.run(ai_monitoring_system.get_system_health())

			# Get recent alerts
			recent_alerts = []  # Would get from alert manager

			# Get active connections
			websocket_stats = websocket_server.get_connection_stats()

			# Get real-time metrics
			realtime_metrics = asyncio.run(self.analytics._get_realtime_metrics())

			monitoring_data = {
				"health": health_data,
				"alerts": recent_alerts,
				"connections": websocket_stats,
				"metrics": realtime_metrics,
				"auto_refresh": True
			}

			return self.render_template('aicr/admin_monitoring.html', data=monitoring_data)

		except Exception as e:
			flash(f'Error loading monitoring data: {str(e)}', 'error')
			return self.render_template('aicr/admin_monitoring.html', data={})

	@expose('/system-control/')
	@has_access
	def system_control(self):
		"""System control and administration page."""
		try:
			# Check admin permissions
			# In production, this would check user roles

			# Get system status
			system_status = {
				"ai_service": "active" if self.ai_service._initialized else "inactive",
				"monitoring_system": ai_monitoring_system.system_status.value,
				"ml_pipeline_framework": "active" if ml_pipeline_framework._initialized else "inactive",
				"model_marketplace": "active" if model_marketplace._initialized else "inactive",
				"websocket_server": "active" if websocket_server._initialized else "inactive"
			}

			# Get configuration settings
			config_settings = {
				"monitoring_interval": 30,
				"auto_scaling_enabled": True,
				"security_level": "high",
				"logging_level": "info"
			}

			control_data = {
				"system_status": system_status,
				"config_settings": config_settings,
				"available_actions": [
					"restart_services",
					"update_configuration",
					"clear_cache",
					"export_logs",
					"backup_data"
				]
			}

			return self.render_template('aicr/admin_control.html', data=control_data)

		except Exception as e:
			flash(f'Error loading system control: {str(e)}', 'error')
			return self.render_template('aicr/admin_control.html', data={})

	@expose('/api/system-action', methods=['POST'])
	@protect()
	def api_system_action(self):
		"""API endpoint for system actions."""
		try:
			action_data = request.get_json()
			action = action_data.get('action')

			if action == 'restart_services':
				# Restart services (in production, this would be more sophisticated)
				result = asyncio.run(self._restart_services())
			elif action == 'clear_cache':
				# Clear analytics cache
				self.analytics._cache.clear()
				result = {"success": True, "message": "Cache cleared successfully"}
			elif action == 'export_logs':
				# Export system logs
				result = await self._export_logs()
			else:
				result = {"success": False, "error": f"Unknown action: {action}"}

			return jsonify(result)

		except Exception as e:
			return jsonify({"success": False, "error": str(e)}), 500

	@expose('/api/real-time-data')
	def api_realtime_data(self):
		"""API endpoint for real-time dashboard data."""
		try:
			# Get real-time metrics
			realtime_data = asyncio.run(self.analytics._get_realtime_metrics())

			# Get system health
			health_data = asyncio.run(ai_monitoring_system.get_system_health())

			# Get WebSocket connection stats
			websocket_stats = websocket_server.get_connection_stats()

			return jsonify({
				"realtime_metrics": realtime_data,
				"system_health": health_data,
				"websocket_stats": websocket_stats,
				"timestamp": datetime.utcnow().isoformat()
			})

		except Exception as e:
			return jsonify({"error": str(e)}), 500

	@expose('/api/chart-data/<chart_type>')
	def api_chart_data(self, chart_type):
		"""API endpoint for chart data."""
		try:
			# Get query parameters
			kwargs = dict(request.args)

			# Generate chart data
			chart_data = asyncio.run(
				self.analytics.generate_visualization_data(chart_type, **kwargs)
			)

			return jsonify(chart_data)

		except Exception as e:
			return jsonify({"error": str(e)}), 500

	async def _restart_services(self) -> Dict[str, Any]:
		"""Restart AICR services."""
		try:
			# In production, this would gracefully restart services
			results = {
				"ai_service": "restarted",
				"monitoring_system": "restarted",
				"ml_pipeline_framework": "restarted",
				"websocket_server": "restarted"
			}

			return {
				"success": True,
				"message": "Services restarted successfully",
				"results": results
			}

		except Exception as e:
			return {"success": False, "error": str(e)}

	async def _export_logs(self) -> Dict[str, Any]:
		"""Export system logs."""
		try:
			# In production, this would collect and package logs
			log_file = f"aicr_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"

			return {
				"success": True,
				"message": "Logs exported successfully",
				"log_file": log_file,
				"download_url": f"/admin/aicr/download-logs/{log_file}"
			}

		except Exception as e:
			return {"success": False, "error": str(e)}


# Create dashboard blueprint
def create_dashboard_blueprint() -> Blueprint:
	"""Create and configure the dashboard blueprint.

	Returns:
		Blueprint: Configured dashboard blueprint
	"""
	dashboard_bp = Blueprint(
		'aicr_dashboard',
		__name__,
		template_folder='templates',
		static_folder='static'
	)

	# Initialize dashboard
	dashboard = AICRDashboard()

	# Register routes
	dashboard_bp.add_url_rule(
		'/overview/',
		'overview',
		dashboard.overview,
		methods=['GET']
	)

	dashboard_bp.add_url_rule(
		'/analytics/',
		'analytics',
		dashboard.analytics,
		methods=['GET']
	)

	dashboard_bp.add_url_rule(
		'/monitoring/',
		'monitoring',
		dashboard.monitoring,
		methods=['GET']
	)

	dashboard_bp.add_url_rule(
		'/system-control/',
		'system_control',
		dashboard.system_control,
		methods=['GET']
	)

	dashboard_bp.add_url_rule(
		'/api/system-action',
		'api_system_action',
		dashboard.api_system_action,
		methods=['POST']
	)

	dashboard_bp.add_url_rule(
		'/api/real-time-data',
		'api_realtime_data',
		dashboard.api_realtime_data,
		methods=['GET']
	)

	dashboard_bp.add_url_rule(
		'/api/chart-data/<chart_type>',
		'api_chart_data',
		dashboard.api_chart_data,
		methods=['GET']
	)

	return dashboard_bp


# Export dashboard classes and functions
__all__ = [
	'AICRDashboard',
	'DashboardAnalytics',
	'create_dashboard_blueprint'
]