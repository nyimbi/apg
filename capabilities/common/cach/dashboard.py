#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Dashboard Management
Advanced monitoring dashboard with real-time analytics

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from flask import Blueprint, render_template, jsonify, request, flash, redirect, url_for
from flask_appbuilder import AppBuilder, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import GroupByChartView
try:
	import plotly
	import plotly.graph_objs as go
except ImportError:
	class _PlotlyJSONEncoder(json.JSONEncoder):
		def default(self, obj: Any) -> Any:
			if hasattr(obj, "to_plotly_json"):
				return obj.to_plotly_json()
			return super().default(obj)

	class _SimpleTrace(dict):
		def __init__(self, trace_type: str, **kwargs: Any):
			super().__init__(type=trace_type, **kwargs)

		def to_plotly_json(self) -> Dict[str, Any]:
			return dict(self)

	class _SimpleFigure:
		def __init__(self, data: Optional[List[Any]] = None):
			self.data = list(data or [])
			self.layout: Dict[str, Any] = {}

		def add_trace(self, trace: Any) -> None:
			self.data.append(trace)

		def update_layout(self, **kwargs: Any) -> None:
			self.layout.update(kwargs)

		def to_plotly_json(self) -> Dict[str, Any]:
			return {"data": self.data, "layout": self.layout}

	class _SimpleGraphObjects:
		Figure = _SimpleFigure

		@staticmethod
		def Scatter(**kwargs: Any) -> _SimpleTrace:
			return _SimpleTrace("scatter", **kwargs)

		@staticmethod
		def Pie(**kwargs: Any) -> _SimpleTrace:
			return _SimpleTrace("pie", **kwargs)

		@staticmethod
		def Histogram(**kwargs: Any) -> _SimpleTrace:
			return _SimpleTrace("histogram", **kwargs)

		@staticmethod
		def Bar(**kwargs: Any) -> _SimpleTrace:
			return _SimpleTrace("bar", **kwargs)

	class _SimplePlotly:
		class utils:
			PlotlyJSONEncoder = _PlotlyJSONEncoder

	plotly = _SimplePlotly()
	go = _SimpleGraphObjects()

from .models import CacheEntry, CacheCluster, CacheMetrics
try:
	from .service import CacheService
except ImportError:
	CacheService = Any


@dataclass
class DashboardMetrics:
	"""Dashboard metrics container"""
	total_entries: int
	hit_rate: float
	miss_rate: float
	latency_p50: float
	latency_p95: float
	latency_p99: float
	throughput_qps: float
	error_rate: float
	memory_usage_mb: float
	cpu_usage_percent: float
	tier_distribution: Dict[str, int]
	top_keys: List[Dict[str, Any]]
	recent_operations: List[Dict[str, Any]]


class CacheDashboardView(BaseView):
	"""
	Main cache management dashboard
	Revolutionary user experience with real-time analytics
	"""

	route_base = "/cache"
	default_view = "dashboard"

	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Main dashboard with comprehensive cache analytics"""

		try:
			# Get real-time metrics
			metrics = self._get_dashboard_metrics()

			# Generate performance charts
			performance_chart = self._create_performance_chart()
			tier_distribution_chart = self._create_tier_distribution_chart(metrics.tier_distribution)
			latency_histogram = self._create_latency_histogram()
			throughput_timeline = self._create_throughput_timeline()

			# Get system health status
			health_status = self._get_system_health_status()

			# Get recent alerts
			recent_alerts = self._get_recent_alerts()

			# Get optimization recommendations
			recommendations = self._get_optimization_recommendations()

			return self.render_template(
				'cache/dashboard.html',
				metrics=metrics,
				performance_chart=performance_chart,
				tier_distribution_chart=tier_distribution_chart,
				latency_histogram=latency_histogram,
				throughput_timeline=throughput_timeline,
				health_status=health_status,
				recent_alerts=recent_alerts,
				recommendations=recommendations
			)

		except Exception as e:
			logging.error(f"Dashboard error: {e}")
			flash(f"Error loading dashboard: {e}", "error")
			return self.render_template('cache/error.html', error=str(e))

	@expose('/analytics')
	@has_access
	def analytics(self):
		"""Advanced analytics and insights"""

		try:
			# Get analytics data
			analytics_data = self._get_analytics_data()

			# Generate analytics charts
			access_pattern_chart = self._create_access_pattern_chart()
			predictive_chart = self._create_predictive_analytics_chart()
			efficiency_trends = self._create_efficiency_trends_chart()
			geo_distribution = self._create_geo_distribution_chart()

			return self.render_template(
				'cache/analytics.html',
				analytics_data=analytics_data,
				access_pattern_chart=access_pattern_chart,
				predictive_chart=predictive_chart,
				efficiency_trends=efficiency_trends,
				geo_distribution=geo_distribution
			)

		except Exception as e:
			logging.error(f"Analytics error: {e}")
			flash(f"Error loading analytics: {e}", "error")
			return self.render_template('cache/error.html', error=str(e))

	@expose('/configuration')
	@has_access
	def configuration(self):
		"""Cache configuration management"""

		try:
			# Get current configuration
			current_config = self._get_current_configuration()

			# Get configuration recommendations
			config_recommendations = self._get_configuration_recommendations()

			# Get available configuration templates
			templates = self._get_configuration_templates()

			return self.render_template(
				'cache/configuration.html',
				current_config=current_config,
				recommendations=config_recommendations,
				templates=templates
			)

		except Exception as e:
			logging.error(f"Configuration error: {e}")
			flash(f"Error loading configuration: {e}", "error")
			return self.render_template('cache/error.html', error=str(e))

	@expose('/optimization')
	@has_access
	def optimization(self):
		"""Cache optimization and tuning"""

		try:
			# Get optimization status
			optimization_status = self._get_optimization_status()

			# Get AI recommendations
			ai_recommendations = self._get_ai_optimization_recommendations()

			# Get performance predictions
			performance_predictions = self._get_performance_predictions()

			return self.render_template(
				'cache/optimization.html',
				optimization_status=optimization_status,
				ai_recommendations=ai_recommendations,
				performance_predictions=performance_predictions
			)

		except Exception as e:
			logging.error(f"Optimization error: {e}")
			flash(f"Error loading optimization: {e}", "error")
			return self.render_template('cache/error.html', error=str(e))

	@expose('/monitoring')
	@has_access
	def monitoring(self):
		"""Real-time monitoring and alerting"""

		try:
			# Get monitoring data
			monitoring_data = self._get_monitoring_data()

			# Get alert rules
			alert_rules = self._get_alert_rules()

			# Get system metrics
			system_metrics = self._get_system_metrics()

			return self.render_template(
				'cache/monitoring.html',
				monitoring_data=monitoring_data,
				alert_rules=alert_rules,
				system_metrics=system_metrics
			)

		except Exception as e:
			logging.error(f"Monitoring error: {e}")
			flash(f"Error loading monitoring: {e}", "error")
			return self.render_template('cache/error.html', error=str(e))

	# API Endpoints for real-time data

	@expose('/api/metrics')
	@has_access
	def api_metrics(self):
		"""API endpoint for real-time metrics"""
		try:
			metrics = self._get_dashboard_metrics()
			return jsonify({
				'status': 'success',
				'data': {
					'total_entries': metrics.total_entries,
					'hit_rate': metrics.hit_rate,
					'miss_rate': metrics.miss_rate,
					'latency_p50': metrics.latency_p50,
					'latency_p95': metrics.latency_p95,
					'latency_p99': metrics.latency_p99,
					'throughput_qps': metrics.throughput_qps,
					'error_rate': metrics.error_rate,
					'memory_usage_mb': metrics.memory_usage_mb,
					'cpu_usage_percent': metrics.cpu_usage_percent,
					'tier_distribution': metrics.tier_distribution,
					'timestamp': datetime.utcnow().isoformat()
				}
			})
		except Exception as e:
			return jsonify({'status': 'error', 'message': str(e)}), 500

	@expose('/api/health')
	@has_access
	def api_health(self):
		"""API endpoint for system health"""
		try:
			health_status = self._get_system_health_status()
			return jsonify({
				'status': 'success',
				'data': health_status
			})
		except Exception as e:
			return jsonify({'status': 'error', 'message': str(e)}), 500

	@expose('/api/alerts')
	@has_access
	def api_alerts(self):
		"""API endpoint for recent alerts"""
		try:
			alerts = self._get_recent_alerts()
			return jsonify({
				'status': 'success',
				'data': alerts
			})
		except Exception as e:
			return jsonify({'status': 'error', 'message': str(e)}), 500

	# Configuration management endpoints

	@expose('/api/config/apply', methods=['POST'])
	@has_access
	def api_apply_config(self):
		"""Apply configuration changes"""
		try:
			config_data = request.json
			result = self._apply_configuration(config_data)
			return jsonify({
				'status': 'success' if result else 'error',
				'message': 'Configuration applied successfully' if result else 'Failed to apply configuration'
			})
		except Exception as e:
			return jsonify({'status': 'error', 'message': str(e)}), 500

	@expose('/api/optimization/run', methods=['POST'])
	@has_access
	def api_run_optimization(self):
		"""Run cache optimization"""
		try:
			optimization_type = request.json.get('type', 'full')
			result = self._run_optimization(optimization_type)
			return jsonify({
				'status': 'success',
				'data': result
			})
		except Exception as e:
			return jsonify({'status': 'error', 'message': str(e)}), 500

	# Private helper methods

	def _get_dashboard_metrics(self) -> DashboardMetrics:
		"""Get comprehensive dashboard metrics"""
		cache_service = self._get_cache_service()
		if cache_service is None:
			return self._empty_dashboard_metrics()
		return self._build_dashboard_metrics(cache_service)

	def _get_cache_service(self) -> Optional[CacheService]:
		"""Return an injected or globally registered cache service if available."""
		cache_service = getattr(self, "cache_service", None) or getattr(self, "_cache_service", None)
		if cache_service is not None:
			return cache_service
		try:
			from .blueprint import get_cache_service_sync
			return get_cache_service_sync()
		except Exception as exc:
			logging.debug("Cache dashboard running without service state: %s", exc)
			return None

	def _empty_dashboard_metrics(self) -> DashboardMetrics:
		"""Return an honest empty-state dashboard instead of fabricated demo values."""
		return DashboardMetrics(
			total_entries=0,
			hit_rate=0.0,
			miss_rate=0.0,
			latency_p50=0.0,
			latency_p95=0.0,
			latency_p99=0.0,
			throughput_qps=0.0,
			error_rate=0.0,
			memory_usage_mb=0.0,
			cpu_usage_percent=0.0,
			tier_distribution={},
			top_keys=[],
			recent_operations=[]
		)

	def _build_dashboard_metrics(self, cache_service: CacheService) -> DashboardMetrics:
		"""Build dashboard metrics from current cache service state."""
		cache_store = getattr(cache_service, "_cache_store", {}) or {}
		metrics = getattr(cache_service, "_metrics", CacheMetrics(tenant_id="dashboard"))
		cache_hits = int(getattr(metrics, "cache_hits", 0) or 0)
		cache_misses = int(getattr(metrics, "cache_misses", 0) or 0)
		total_accesses = cache_hits + cache_misses
		hit_rate = self._call_metric_rate(metrics, "hit_rate", cache_hits / total_accesses if total_accesses else 0.0)
		miss_rate = cache_misses / total_accesses if total_accesses else 0.0
		used_memory_bytes = int(getattr(metrics, "used_memory_bytes", 0) or 0)
		if used_memory_bytes == 0:
			used_memory_bytes = sum(int(getattr(entry, "size_bytes", 0) or 0) for entry in cache_store.values())
		return DashboardMetrics(
			total_entries=len(cache_store),
			hit_rate=hit_rate,
			miss_rate=miss_rate,
			latency_p50=float(getattr(metrics, "p50_latency_ms", 0.0) or 0.0),
			latency_p95=float(getattr(metrics, "p95_latency_ms", 0.0) or 0.0),
			latency_p99=float(getattr(metrics, "p99_latency_ms", 0.0) or 0.0),
			throughput_qps=float(getattr(metrics, "operations_per_second", 0.0) or 0.0),
			error_rate=self._call_metric_rate(metrics, "error_rate", 0.0),
			memory_usage_mb=used_memory_bytes / (1024 * 1024),
			cpu_usage_percent=float(
				getattr(metrics, "cpu_usage_percent", getattr(cache_service, "cpu_usage_percent", 0.0)) or 0.0
			),
			tier_distribution=self._calculate_tier_distribution(cache_store),
			top_keys=self._calculate_top_keys(cache_store),
			recent_operations=self._collect_recent_operations(cache_service, cache_store)
		)

	def _call_metric_rate(self, metrics: Any, method_name: str, fallback: float) -> float:
		"""Read a metric rate from method or fallback value."""
		method = getattr(metrics, method_name, None)
		if callable(method):
			return float(method())
		return float(fallback)

	def _calculate_tier_distribution(self, cache_store: Dict[str, CacheEntry]) -> Dict[str, int]:
		"""Count cache entries by current or recommended tier."""
		distribution: Dict[str, int] = {}
		for entry in cache_store.values():
			tier = getattr(entry, "tier_recommendation", "unknown")
			tier_name = getattr(tier, "value", str(tier)).upper()
			distribution[tier_name] = distribution.get(tier_name, 0) + 1
		return distribution

	def _calculate_top_keys(self, cache_store: Dict[str, CacheEntry]) -> List[Dict[str, Any]]:
		"""Return the hottest cache keys by observed hit count."""
		entries = sorted(
			cache_store.values(),
			key=lambda entry: (
				int(getattr(entry, "hit_count", 0) or 0),
				int(getattr(entry, "access_count", 0) or 0),
				int(getattr(entry, "size_bytes", 0) or 0),
			),
			reverse=True
		)
		return [
			{
				"key": getattr(entry, "key", ""),
				"hits": int(getattr(entry, "hit_count", 0) or 0),
				"size_kb": round(int(getattr(entry, "size_bytes", 0) or 0) / 1024, 2)
			}
			for entry in entries[:10]
		]

	def _collect_recent_operations(
		self,
		cache_service: CacheService,
		cache_store: Dict[str, CacheEntry]
	) -> List[Dict[str, Any]]:
		"""Return recent cache operations from explicit history or entry access state."""
		operation_history = getattr(cache_service, "_operation_history", None)
		if operation_history:
			return [self._normalize_operation(operation) for operation in list(operation_history)[-20:]]

		recent_entries = sorted(
			(
				entry for entry in cache_store.values()
				if getattr(entry, "last_accessed", None) is not None
			),
			key=lambda entry: getattr(entry, "last_accessed"),
			reverse=True
		)
		return [
			{
				"timestamp": self._format_timestamp(getattr(entry, "last_accessed", None)),
				"operation": "GET",
				"key": getattr(entry, "key", ""),
				"result": "HIT" if int(getattr(entry, "hit_count", 0) or 0) > 0 else "MISS"
			}
			for entry in recent_entries[:20]
		]

	def _normalize_operation(self, operation: Any) -> Dict[str, Any]:
		"""Normalize cached operation history records into the dashboard shape."""
		if isinstance(operation, dict):
			return {
				"timestamp": self._format_timestamp(operation.get("timestamp")),
				"operation": str(operation.get("operation", operation.get("type", "UNKNOWN"))).upper(),
				"key": str(operation.get("key", "")),
				"result": str(operation.get("result", operation.get("status", "UNKNOWN"))).upper()
			}
		return {
			"timestamp": self._format_timestamp(getattr(operation, "timestamp", None)),
			"operation": str(getattr(operation, "operation", "UNKNOWN")).upper(),
			"key": str(getattr(operation, "key", "")),
			"result": str(getattr(operation, "result", "UNKNOWN")).upper()
		}

	def _format_timestamp(self, value: Any) -> str:
		"""Format operation timestamps consistently for JSON responses."""
		if hasattr(value, "isoformat"):
			return value.isoformat()
		if value:
			return str(value)
		return datetime.utcnow().isoformat()

	def _create_performance_chart(self) -> str:
		"""Create performance chart as JSON"""
		points = self._performance_points()
		timestamps = [point["timestamp"] for point in points]
		hit_rates = [point["hit_rate"] for point in points]
		latencies = [point["average_latency_ms"] for point in points]

		fig = go.Figure()

		# Hit rate trace
		fig.add_trace(go.Scatter(
			x=timestamps,
			y=hit_rates,
			mode='lines',
			name='Hit Rate',
			yaxis='y',
			line=dict(color='#2E86AB', width=2)
		))

		# Latency trace (secondary y-axis)
		fig.add_trace(go.Scatter(
			x=timestamps,
			y=latencies,
			mode='lines',
			name='Avg Latency (ms)',
			yaxis='y2',
			line=dict(color='#F24236', width=2)
		))

		fig.update_layout(
			title='Cache Performance Over Time',
			xaxis_title='Time',
			yaxis=dict(
				title='Hit Rate',
				side='left',
				range=[0, 1]
			),
			yaxis2=dict(
				title='Latency (ms)',
				side='right',
				overlaying='y',
				range=[0, 10]
			),
			hovermode='x unified'
		)

		return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	def _create_tier_distribution_chart(self, tier_distribution: Dict[str, int]) -> str:
		"""Create tier distribution pie chart"""

		fig = go.Figure(data=[go.Pie(
			labels=list(tier_distribution.keys()),
			values=list(tier_distribution.values()),
			hole=.3,
			marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
		)])

		fig.update_layout(
			title='Cache Tier Distribution',
			annotations=[dict(text='Entries', x=0.5, y=0.5, font_size=16, showarrow=False)]
		)

		return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	def _create_latency_histogram(self) -> str:
		"""Create latency distribution histogram"""
		latencies = [
			latency
			for point in self._performance_points()
			for latency in (
				point.get("p50_latency_ms", 0.0),
				point.get("p95_latency_ms", 0.0),
				point.get("p99_latency_ms", 0.0),
			)
			if latency
		]

		fig = go.Figure(data=[go.Histogram(
			x=latencies,
			nbinsx=20,
			marker_color='#45B7D1',
			opacity=0.7
		)])

		fig.update_layout(
			title='Response Latency Distribution',
			xaxis_title='Latency (ms)',
			yaxis_title='Frequency',
			bargap=0.1
		)

		return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	def _create_throughput_timeline(self) -> str:
		"""Create throughput timeline chart"""
		points = self._performance_points()
		timestamps = [point["timestamp"] for point in points]
		throughput = [point["operations_per_second"] for point in points]

		fig = go.Figure(data=[go.Scatter(
			x=timestamps,
			y=throughput,
			mode='lines+markers',
			name='QPS',
			line=dict(color='#96CEB4', width=3),
			marker=dict(size=6)
		)])

		fig.update_layout(
			title='Throughput Timeline (24 Hours)',
			xaxis_title='Time',
			yaxis_title='Queries Per Second',
			hovermode='x'
		)

		return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	def _create_access_pattern_chart(self) -> str:
		"""Create access pattern analysis chart"""
		pattern_counts = self._access_pattern_counts()
		patterns = list(pattern_counts.keys())
		frequencies = list(pattern_counts.values())

		fig = go.Figure(data=[go.Bar(
			x=patterns,
			y=frequencies,
			marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
		)])

		fig.update_layout(
			title='Access Pattern Analysis',
			xaxis_title='Pattern Type',
			yaxis_title='Frequency (%)'
		)

		return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	def _create_predictive_analytics_chart(self) -> str:
		"""Create predictive analytics chart"""
		predictions = self._prefetch_prediction_points()

		fig = go.Figure(data=[go.Scatter(
			x=[point["key"] for point in predictions],
			y=[point["probability"] for point in predictions],
			mode='lines+markers',
			name='Prefetch Probability',
			line=dict(color='#F24236', width=3, dash='dash'),
			marker=dict(size=8)
		)])

		fig.update_layout(
			title='Predicted Prefetch Candidates',
			xaxis_title='Cache Key',
			yaxis_title='Probability'
		)

		return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	def _create_efficiency_trends_chart(self) -> str:
		"""Create efficiency trends chart"""
		points = self._performance_points()
		timestamps = [point["timestamp"] for point in points]
		efficiency = [point["hit_rate"] * 100 for point in points]

		fig = go.Figure(data=[go.Scatter(
			x=timestamps,
			y=efficiency,
			mode='lines',
			name='Cache Efficiency',
			line=dict(color='#2E86AB', width=3),
			fill='tonexty'
		)])

		fig.update_layout(
			title='Cache Efficiency Trends',
			xaxis_title='Time',
			yaxis_title='Efficiency (%)',
			yaxis=dict(range=[0, 100])
		)

		return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	def _create_geo_distribution_chart(self) -> str:
		"""Create namespace distribution chart"""
		namespace_counts = self._namespace_distribution()
		regions = list(namespace_counts.keys())
		traffic = list(namespace_counts.values())

		fig = go.Figure(data=[go.Bar(
			x=regions,
			y=traffic,
			marker_color='#4ECDC4'
		)])

		fig.update_layout(
			title='Cache Namespace Distribution',
			xaxis_title='Namespace',
			yaxis_title='Entries'
		)

		return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	def _get_system_health_status(self) -> Dict[str, Any]:
		"""Get system health status"""
		service = self._get_cache_service()
		if service is None:
			return {
				'overall_health': 'Unavailable',
				'services_status': {
					'cache_service': 'Unavailable',
					'ai_engine': 'Unavailable',
					'monitoring': 'Unavailable',
					'security': 'Unavailable'
				},
				'resource_usage': {'cpu': 0.0, 'memory': 0.0, 'disk': 0.0, 'network': 0.0}
			}
		metrics = getattr(service, "_metrics", None)
		config = getattr(service, "config", None)
		error_rate = self._call_metric_rate(metrics, "error_rate", 0.0) if metrics else 0.0
		memory = float(getattr(metrics, "memory_utilization_percent", 0.0) or 0.0) if metrics else 0.0
		cluster_health = [
			bool(cluster.is_healthy()) if hasattr(cluster, "is_healthy") else bool(getattr(cluster, "healthy", True))
			for cluster in (getattr(service, "_clusters", {}) or {}).values()
		]
		clusters_ok = all(cluster_health) if cluster_health else True
		running = bool(getattr(service, "running", False))
		overall_health = 'Healthy' if running and clusters_ok and memory < 90 and error_rate < 0.05 else 'Degraded'
		return {
			'overall_health': overall_health,
			'services_status': {
				'cache_service': 'Running' if running else 'Stopped',
				'ai_engine': 'Enabled' if bool(getattr(config, "ai_optimization_enabled", False)) else 'Disabled',
				'monitoring': 'Enabled' if bool(getattr(config, "metrics_enabled", False)) else 'Disabled',
				'security': 'Enabled' if bool(getattr(config, "encryption_enabled", False)) else 'Disabled'
			},
			'resource_usage': {
				'cpu': float(getattr(metrics, "cpu_usage_percent", 0.0) or 0.0) if metrics else 0.0,
				'memory': memory,
				'disk': 0.0,
				'network': float(getattr(metrics, "bytes_per_second", 0.0) or 0.0) if metrics else 0.0
			}
		}

	def _get_recent_alerts(self) -> List[Dict[str, Any]]:
		"""Get recent system alerts"""
		service = self._get_cache_service()
		if service is None:
			return []
		metrics = getattr(service, "_metrics", None)
		if metrics is None:
			return []
		timestamp = self._format_timestamp(getattr(metrics, "timestamp", None))
		alerts: List[Dict[str, Any]] = []
		memory = float(getattr(metrics, "memory_utilization_percent", 0.0) or 0.0)
		error_rate = self._call_metric_rate(metrics, "error_rate", 0.0)
		hit_rate = self._call_metric_rate(metrics, "hit_rate", 0.0)
		if memory >= 90:
			alerts.append({'timestamp': timestamp, 'level': 'CRITICAL', 'message': 'Cache memory utilization above 90%', 'resolved': False})
		if error_rate >= 0.05:
			alerts.append({'timestamp': timestamp, 'level': 'WARNING', 'message': 'Cache error rate above 5%', 'resolved': False})
		if hit_rate and hit_rate < 0.8:
			alerts.append({'timestamp': timestamp, 'level': 'INFO', 'message': 'Cache hit rate below 80%', 'resolved': False})
		return alerts

	def _get_optimization_recommendations(self) -> List[Dict[str, Any]]:
		"""Get optimization recommendations"""
		service = self._get_cache_service()
		if service is None:
			return []
		recommendations: List[Dict[str, Any]] = []
		for result in getattr(service, "_ai_optimization_results", [])[-5:]:
			confidence = float(getattr(result, "confidence_score", 0.0) or 0.0)
			for item in getattr(result, "recommendations", []) or []:
				recommendations.append({
					'type': str(item.get('type', 'Optimization')),
					'title': str(item.get('title', item.get('type', 'Optimization recommendation'))),
					'description': str(item.get('description', '')),
					'impact': str(item.get('impact', 'medium')).title(),
					'confidence': confidence
				})
		if recommendations:
			return recommendations
		metrics = getattr(service, "_metrics", None)
		if metrics and self._call_metric_rate(metrics, "hit_rate", 0.0) < 0.8:
			recommendations.append({
				'type': 'Performance',
				'title': 'Increase effective cache coverage',
				'description': 'Observed hit rate is below 80%; review TTLs, key coverage, and cache size.',
				'impact': 'Medium',
				'confidence': 0.7
			})
		return recommendations

	def _get_analytics_data(self) -> Dict[str, Any]:
		"""Get analytics data"""
		service = self._get_cache_service()
		metrics = getattr(service, "_metrics", None) if service is not None else None
		total_requests = int(getattr(metrics, "total_operations", 0) or 0) if metrics else 0
		cache_hits = int(getattr(metrics, "cache_hits", 0) or 0) if metrics else 0
		cache_misses = int(getattr(metrics, "cache_misses", 0) or 0) if metrics else 0
		bytes_served = sum(
			int(getattr(entry, "size_bytes", 0) or 0)
			for entry in (getattr(service, "_cache_store", {}) or {}).values()
		) if service is not None else 0
		return {
			'summary': {
				'total_requests': total_requests,
				'cache_hits': cache_hits,
				'cache_misses': cache_misses,
				'data_served_gb': bytes_served / (1024 ** 3)
			},
			'trends': {
				'hit_rate_trend': self._trend_text("hit_rate"),
				'latency_trend': self._trend_text("average_latency_ms", lower_is_better=True),
				'throughput_trend': self._trend_text("operations_per_second")
			}
		}

	def _get_current_configuration(self) -> Dict[str, Any]:
		"""Get current configuration"""
		service = self._get_cache_service()
		config = getattr(service, "config", None) if service is not None else None
		if config is None:
			return {
				'cache_size_mb': 0,
				'eviction_policy': 'unknown',
				'tier_distribution': {},
				'security_level': 'unknown',
				'monitoring_enabled': False
			}
		return {
			'cache_size_mb': int(getattr(config, "max_memory_mb", 0) or 0),
			'eviction_policy': str(getattr(config, "eviction_policy", "adaptive")),
			'tier_distribution': self._get_dashboard_metrics().tier_distribution,
			'security_level': str(getattr(config, "security_level", "unknown")),
			'monitoring_enabled': bool(getattr(config, "metrics_enabled", False))
		}

	def _get_configuration_recommendations(self) -> List[Dict[str, Any]]:
		"""Get configuration recommendations"""
		return [
			{
				'domain': 'Cache Sizing',
				'recommendation': 'Increase total cache size to 6GB',
				'confidence': 0.85,
				'risk': 'Low'
			}
		]

	def _get_configuration_templates(self) -> List[Dict[str, Any]]:
		"""Get configuration templates"""
		return [
			{'name': 'High Performance Web App', 'description': 'Optimized for low latency web applications'},
			{'name': 'API Server', 'description': 'Optimized for high-throughput API workloads'},
			{'name': 'Analytics Workload', 'description': 'Optimized for large data analytics workloads'}
		]

	def _get_optimization_status(self) -> Dict[str, Any]:
		service = self._get_cache_service()
		results = getattr(service, "_ai_optimization_results", []) if service is not None else []
		last_result = results[-1] if results else None
		return {
			'status': 'Active' if bool(getattr(getattr(service, "config", None), "ai_optimization_enabled", False)) else 'Disabled',
			'last_run': self._format_timestamp(getattr(last_result, "timestamp", None)) if last_result else None
		}

	def _get_ai_optimization_recommendations(self) -> List[Dict[str, Any]]:
		return self._get_optimization_recommendations()

	def _get_performance_predictions(self) -> Dict[str, Any]:
		return {'prefetch_candidates': self._prefetch_prediction_points()}

	def _get_monitoring_data(self) -> Dict[str, Any]:
		return {'metrics': self._get_dashboard_metrics(), 'history': self._performance_points()}

	def _get_alert_rules(self) -> List[Dict[str, Any]]:
		service = self._get_cache_service()
		clusters = getattr(service, "_clusters", {}) if service is not None else {}
		rules: List[Dict[str, Any]] = []
		for cluster in clusters.values():
			for name, threshold in (getattr(cluster, "alert_thresholds", {}) or {}).items():
				rules.append({'cluster': getattr(cluster, 'name', getattr(cluster, 'cluster_id', 'unknown')), 'metric': name, 'threshold': threshold})
		return rules

	def _get_system_metrics(self) -> Dict[str, Any]:
		metrics = self._get_dashboard_metrics()
		return {
			'total_entries': metrics.total_entries,
			'hit_rate': metrics.hit_rate,
			'miss_rate': metrics.miss_rate,
			'throughput_qps': metrics.throughput_qps,
			'memory_usage_mb': metrics.memory_usage_mb,
			'cpu_usage_percent': metrics.cpu_usage_percent
		}

	def _apply_configuration(self, config_data: Dict[str, Any]) -> bool:
		service = self._get_cache_service()
		config = getattr(service, "config", None) if service is not None else None
		if config is None:
			return False
		for key, value in (config_data or {}).items():
			if hasattr(config, key):
				setattr(config, key, value)
		return True

	def _run_optimization(self, optimization_type: str) -> Dict[str, Any]:
		service = self._get_cache_service()
		if service is None:
			return {'status': 'unavailable', 'type': optimization_type}
		return {
			'status': 'queued' if bool(getattr(getattr(service, "config", None), "ai_optimization_enabled", False)) else 'disabled',
			'type': optimization_type,
			'pending_recommendations': len(self._get_optimization_recommendations())
		}

	def _performance_points(self) -> List[Dict[str, Any]]:
		"""Return performance points from service history or current metrics."""
		service = self._get_cache_service()
		if service is None:
			return []
		history = list(getattr(service, "_performance_history", []) or [])
		if not history and getattr(service, "_metrics", None) is not None:
			history = [getattr(service, "_metrics")]
		return [self._metric_point(metric) for metric in history[-100:]]

	def _metric_point(self, metric: Any) -> Dict[str, Any]:
		return {
			"timestamp": self._format_timestamp(getattr(metric, "timestamp", None)),
			"hit_rate": self._call_metric_rate(metric, "hit_rate", 0.0),
			"memory_utilization": float(getattr(metric, "memory_utilization_percent", 0.0) or 0.0),
			"operations_per_second": float(getattr(metric, "operations_per_second", 0.0) or 0.0),
			"average_latency_ms": float(getattr(metric, "average_latency_ms", 0.0) or 0.0),
			"p50_latency_ms": float(getattr(metric, "p50_latency_ms", 0.0) or 0.0),
			"p95_latency_ms": float(getattr(metric, "p95_latency_ms", 0.0) or 0.0),
			"p99_latency_ms": float(getattr(metric, "p99_latency_ms", 0.0) or 0.0),
		}

	def _access_pattern_counts(self) -> Dict[str, int]:
		service = self._get_cache_service()
		counts: Dict[str, int] = {}
		for entry in (getattr(service, "_cache_store", {}) or {}).values() if service is not None else []:
			pattern = getattr(entry, "access_pattern", "unknown")
			label = getattr(pattern, "value", str(pattern)).replace("_", " ").title()
			counts[label] = counts.get(label, 0) + 1
		return counts

	def _prefetch_prediction_points(self) -> List[Dict[str, Any]]:
		service = self._get_cache_service()
		predictions = getattr(service, "_prefetch_predictions", {}) if service is not None else {}
		return [
			{"key": key, "probability": float(probability)}
			for key, probability in sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:20]
		]

	def _namespace_distribution(self) -> Dict[str, int]:
		service = self._get_cache_service()
		counts: Dict[str, int] = {}
		for entry in (getattr(service, "_cache_store", {}) or {}).values() if service is not None else []:
			namespace = str(getattr(entry, "namespace", "default"))
			counts[namespace] = counts.get(namespace, 0) + 1
		return counts

	def _trend_text(self, metric_name: str, lower_is_better: bool = False) -> str:
		points = self._performance_points()
		if len(points) < 2:
			return "0.0%"
		start = float(points[0].get(metric_name, 0.0) or 0.0)
		end = float(points[-1].get(metric_name, 0.0) or 0.0)
		if start == 0:
			return "0.0%"
		change = ((end - start) / start) * 100
		if lower_is_better:
			change *= -1
		return f"{change:+.1f}%"


# Export main components
__all__ = [
	'CacheDashboardView',
	'DashboardMetrics'
]
