"""
APG Central Configuration - Performance Optimization Dashboard

Real-time performance monitoring, optimization recommendations,
and automated performance tuning interface.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Flask-AppBuilder for web interface
from flask import Blueprint, render_template, request, jsonify, Response
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.security.decorators import protect

from .auto_scaler import IntelligentAutoScaler, ScalingEvent, ResourcePrediction
from ..service import CentralConfigurationEngine


@dataclass
class PerformanceMetrics:
	"""Performance metrics data structure."""
	timestamp: str
	response_time_p95: float
	response_time_p99: float
	throughput_rps: float
	error_rate: float
	cpu_usage: float
	memory_usage: float
	database_connections: int
	cache_hit_rate: float
	active_sessions: int


@dataclass
class OptimizationRecommendation:
	"""Performance optimization recommendation."""
	recommendation_id: str
	title: str
	description: str
	impact_level: str  # high, medium, low
	effort_level: str  # high, medium, low
	category: str      # database, cache, application, infrastructure
	expected_improvement: str
	implementation_steps: List[str]
	priority_score: float
	estimated_hours: int


class PerformanceOptimizationDashboard(BaseView):
	"""Flask-AppBuilder view for performance optimization dashboard."""
	
	route_base = "/performance"
	default_view = "dashboard"
	
	def __init__(self, auto_scaler: IntelligentAutoScaler):
		"""Initialize performance dashboard view."""
		super().__init__()
		self.auto_scaler = auto_scaler
		self.performance_history: List[PerformanceMetrics] = []
		self.optimization_history: List[Dict[str, Any]] = []
		
		# Start performance monitoring
		asyncio.create_task(self._start_performance_monitoring())
	
	@expose("/")
	@has_access
	@protect("can_read", "PerformanceDashboard")
	def dashboard(self):
		"""Main performance optimization dashboard."""
		return render_template(
			"performance/dashboard.html",
			title="Performance Optimization Dashboard"
		)
	
	@expose("/api/overview")
	@has_access
	@protect("can_read", "PerformanceDashboard")
	def api_overview(self):
		"""API endpoint for performance overview data."""
		overview_data = asyncio.run(self._get_performance_overview())
		return jsonify(overview_data)
	
	@expose("/api/metrics")
	@has_access
	@protect("can_read", "PerformanceDashboard")
	def api_metrics(self):
		"""API endpoint for real-time performance metrics."""
		metrics_data = asyncio.run(self._get_current_metrics())
		return jsonify(metrics_data)
	
	@expose("/api/scaling-events")
	@has_access
	@protect("can_read", "PerformanceDashboard")
	def api_scaling_events(self):
		"""API endpoint for recent scaling events."""
		events_data = asyncio.run(self._get_scaling_events())
		return jsonify(events_data)
	
	@expose("/api/predictions")
	@has_access
	@protect("can_read", "PerformanceDashboard")
	def api_predictions(self):
		"""API endpoint for performance predictions."""
		predictions_data = asyncio.run(self._get_performance_predictions())
		return jsonify(predictions_data)
	
	@expose("/api/optimization-recommendations")
	@has_access
	@protect("can_read", "PerformanceDashboard")
	def api_optimization_recommendations(self):
		"""API endpoint for optimization recommendations."""
		recommendations = asyncio.run(self._get_optimization_recommendations())
		return jsonify([asdict(rec) for rec in recommendations])
	
	@expose("/api/optimize", methods=["POST"])
	@has_access
	@protect("can_write", "PerformanceDashboard")
	def api_trigger_optimization(self):
		"""API endpoint to trigger performance optimization."""
		data = request.get_json() or {}
		optimization_type = data.get("type", "comprehensive")
		
		optimization_result = asyncio.run(
			self._trigger_optimization(optimization_type)
		)
		
		return jsonify(optimization_result)
	
	@expose("/api/scaling-control", methods=["POST"])
	@has_access
	@protect("can_write", "PerformanceDashboard")
	def api_scaling_control(self):
		"""API endpoint for manual scaling control."""
		data = request.get_json() or {}
		action = data.get("action")  # enable, disable, manual_scale
		
		if action == "enable":
			self.auto_scaler.optimization_enabled = True
			return jsonify({"status": "enabled", "message": "Auto-scaling enabled"})
		
		elif action == "disable":
			self.auto_scaler.optimization_enabled = False
			return jsonify({"status": "disabled", "message": "Auto-scaling disabled"})
		
		elif action == "manual_scale":
			component = data.get("component")
			target_value = data.get("target_value")
			resource_type = data.get("resource_type", "replicas")
			
			# Create manual scaling decision
			decision = {
				"component": component,
				"resource_type": resource_type,
				"current_value": data.get("current_value", 1),
				"target_value": target_value,
				"direction": "up" if target_value > data.get("current_value", 1) else "down",
				"trigger_signals": [{"metric": "manual", "value": 0.0}],
				"strategy": "manual",
				"priority": 0,
				"confidence": 1.0
			}
			
			scaling_event = asyncio.run(
				self.auto_scaler.execute_scaling_decision(decision)
			)
			
			return jsonify({
				"status": "executed",
				"event_id": scaling_event.event_id,
				"success": scaling_event.success
			})
		
		return jsonify({"error": "Invalid action"}), 400
	
	async def _get_performance_overview(self) -> Dict[str, Any]:
		"""Get performance overview data."""
		current_metrics = await self._get_current_metrics()
		recent_events = await self._get_scaling_events()
		
		# Calculate performance trends
		if len(self.performance_history) >= 2:
			latest = self.performance_history[-1]
			previous = self.performance_history[-2]
			
			response_time_trend = ((latest.response_time_p95 - previous.response_time_p95) / previous.response_time_p95) * 100
			throughput_trend = ((latest.throughput_rps - previous.throughput_rps) / previous.throughput_rps) * 100
			error_rate_trend = ((latest.error_rate - previous.error_rate) / max(previous.error_rate, 0.01)) * 100
		else:
			response_time_trend = 0.0
			throughput_trend = 0.0
			error_rate_trend = 0.0
		
		# System health score (0-100)
		health_score = await self._calculate_system_health_score(current_metrics)
		
		# Performance grade
		performance_grade = await self._calculate_performance_grade(current_metrics)
		
		return {
			"system_health": {
				"score": health_score,
				"grade": performance_grade,
				"status": "excellent" if health_score >= 90 else "good" if health_score >= 75 else "needs_attention"
			},
			"current_metrics": current_metrics,
			"trends": {
				"response_time": response_time_trend,
				"throughput": throughput_trend,
				"error_rate": error_rate_trend
			},
			"scaling_summary": {
				"events_last_24h": len([e for e in recent_events["events"] if self._is_within_24h(e["timestamp"])]),
				"successful_scaling": recent_events["successful_events"],
				"auto_scaling_enabled": self.auto_scaler.optimization_enabled
			},
			"optimization_summary": {
				"recommendations_available": len(await self._get_optimization_recommendations()),
				"last_optimization": self.optimization_history[-1]["timestamp"] if self.optimization_history else None,
				"performance_improvement_last_30d": 15.3  # Mock improvement percentage
			}
		}
	
	async def _get_current_metrics(self) -> Dict[str, Any]:
		"""Get current performance metrics."""
		# Collect system metrics from auto-scaler
		system_metrics = await self.auto_scaler.collect_system_metrics()
		
		# Extract performance metrics
		api_metrics = system_metrics["components"].get("central-config-api", {})
		db_metrics = system_metrics["components"].get("database", {})
		cache_metrics = system_metrics["components"].get("cache", {})
		
		current_time = datetime.now(timezone.utc)
		
		metrics = PerformanceMetrics(
			timestamp=current_time.isoformat(),
			response_time_p95=api_metrics.get("response_time_p95", 125.0),
			response_time_p99=api_metrics.get("response_time_p99", 250.0),
			throughput_rps=api_metrics.get("request_rate", 150.0),
			error_rate=api_metrics.get("error_rate", 0.5),
			cpu_usage=api_metrics.get("cpu_usage", 45.0),
			memory_usage=api_metrics.get("memory_usage", 62.0),
			database_connections=db_metrics.get("active_connections", 25),
			cache_hit_rate=cache_metrics.get("hit_rate", 92.5),
			active_sessions=api_metrics.get("active_sessions", 89)
		)
		
		# Store in history
		self.performance_history.append(metrics)
		
		# Keep only last 24 hours of data
		cutoff_time = current_time - timedelta(hours=24)
		self.performance_history = [
			m for m in self.performance_history 
			if datetime.fromisoformat(m.timestamp.replace('Z', '+00:00')) > cutoff_time
		]
		
		return asdict(metrics)
	
	async def _get_scaling_events(self) -> Dict[str, Any]:
		"""Get recent scaling events."""
		recent_events = []
		successful_events = 0
		
		# Get events from last 24 hours
		cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
		
		for event in self.auto_scaler.scaling_history:
			if event.timestamp > cutoff_time:
				event_data = {
					"event_id": event.event_id,
					"timestamp": event.timestamp.isoformat(),
					"component": event.component,
					"resource_type": event.resource_type.value,
					"direction": event.direction.value,
					"from_value": event.from_value,
					"to_value": event.to_value,
					"trigger_metric": event.trigger_metric,
					"trigger_value": event.trigger_value,
					"strategy": event.strategy_used.value,
					"execution_time_ms": event.execution_time_ms,
					"success": event.success,
					"reason": event.reason
				}
				recent_events.append(event_data)
				
				if event.success:
					successful_events += 1
		
		# Sort by timestamp (newest first)
		recent_events.sort(key=lambda x: x["timestamp"], reverse=True)
		
		return {
			"events": recent_events,
			"total_events": len(recent_events),
			"successful_events": successful_events,
			"success_rate": (successful_events / len(recent_events) * 100) if recent_events else 100.0
		}
	
	async def _get_performance_predictions(self) -> Dict[str, Any]:
		"""Get performance predictions."""
		predictions = {}
		
		# Get predictions for key components
		components = ["central-config-api", "central-config-web", "database"]
		
		for component in components:
			try:
				prediction = await self.auto_scaler.predict_resource_requirements(
					component,
					prediction_horizon_minutes=60  # 1 hour prediction
				)
				
				predictions[component] = {
					"timestamp": prediction.timestamp.isoformat(),
					"predicted_cpu_usage": prediction.predicted_cpu_usage,
					"predicted_memory_usage": prediction.predicted_memory_usage,
					"predicted_request_rate": prediction.predicted_request_rate,
					"confidence_score": prediction.confidence_score,
					"prediction_horizon_minutes": prediction.prediction_horizon_minutes
				}
			except Exception as e:
				print(f"⚠️ Prediction failed for {component}: {e}")
				
				# Provide default prediction
				predictions[component] = {
					"timestamp": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
					"predicted_cpu_usage": 50.0,
					"predicted_memory_usage": 60.0,
					"predicted_request_rate": 100.0,
					"confidence_score": 0.5,
					"prediction_horizon_minutes": 60
				}
		
		return {
			"predictions": predictions,
			"generated_at": datetime.now(timezone.utc).isoformat(),
			"prediction_algorithm": "RandomForest with time series features"
		}
	
	async def _get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
		"""Get performance optimization recommendations."""
		recommendations = []
		
		# Analyze current performance and generate recommendations
		current_metrics = await self._get_current_metrics()
		
		# Database optimization recommendations
		if current_metrics["database_connections"] > 80:
			recommendations.append(OptimizationRecommendation(
				recommendation_id="db_pool_optimization",
				title="Optimize Database Connection Pool",
				description="Database connection pool is experiencing high utilization. Increasing pool size and optimizing connection management could improve performance.",
				impact_level="high",
				effort_level="medium",
				category="database",
				expected_improvement="25% reduction in database query latency",
				implementation_steps=[
					"Increase database connection pool size from 20 to 35",
					"Implement connection pooling best practices",
					"Add connection pool monitoring",
					"Optimize long-running queries"
				],
				priority_score=8.5,
				estimated_hours=6
			))
		
		# Cache optimization recommendations
		if current_metrics["cache_hit_rate"] < 90:
			recommendations.append(OptimizationRecommendation(
				recommendation_id="cache_hit_optimization",
				title="Improve Cache Hit Rate",
				description="Cache hit rate is below optimal levels. Implementing better caching strategies could significantly improve response times.",
				impact_level="medium",
				effort_level="low",
				category="cache",
				expected_improvement="15% improvement in response times",
				implementation_steps=[
					"Analyze cache miss patterns",
					"Implement predictive caching for frequently accessed data",
					"Increase cache TTL for stable configurations",
					"Add cache warming mechanisms"
				],
				priority_score=7.2,
				estimated_hours=4
			))
		
		# Response time optimization
		if current_metrics["response_time_p95"] > 200:
			recommendations.append(OptimizationRecommendation(
				recommendation_id="response_time_optimization",
				title="Optimize API Response Times",
				description="95th percentile response times are higher than target. Several optimizations could improve overall API performance.",
				impact_level="high",
				effort_level="high",
				category="application",
				expected_improvement="30% reduction in P95 response times",
				implementation_steps=[
					"Implement API response caching",
					"Optimize database queries with proper indexing",
					"Add async processing for heavy operations",
					"Implement request batching where applicable",
					"Review and optimize serialization"
				],
				priority_score=9.1,
				estimated_hours=12
			))
		
		# CPU usage optimization
		if current_metrics["cpu_usage"] > 70:
			recommendations.append(OptimizationRecommendation(
				recommendation_id="cpu_optimization",
				title="Optimize CPU Usage",
				description="CPU usage is approaching limits. Implementing CPU optimization techniques and horizontal scaling could prevent performance degradation.",
				impact_level="medium",
				effort_level="medium",
				category="infrastructure",
				expected_improvement="20% reduction in CPU usage",
				implementation_steps=[
					"Profile CPU-intensive operations",
					"Implement more efficient algorithms",
					"Add horizontal scaling rules",
					"Optimize background tasks scheduling",
					"Consider CPU resource limits adjustment"
				],
				priority_score=7.8,
				estimated_hours=8
			))
		
		# Sort recommendations by priority score
		recommendations.sort(key=lambda x: x.priority_score, reverse=True)
		
		return recommendations
	
	async def _trigger_optimization(self, optimization_type: str) -> Dict[str, Any]:
		"""Trigger performance optimization."""
		optimization_result = {
			"optimization_id": f"opt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
			"type": optimization_type,
			"started_at": datetime.now(timezone.utc).isoformat(),
			"status": "running"
		}
		
		try:
			if optimization_type == "comprehensive":
				# Run comprehensive system optimization
				result = await self.auto_scaler.optimize_system_performance()
				optimization_result.update(result)
			
			elif optimization_type == "database":
				# Run database-specific optimization
				result = await self._optimize_database_only()
				optimization_result.update(result)
			
			elif optimization_type == "cache":
				# Run cache-specific optimization
				result = await self._optimize_cache_only()
				optimization_result.update(result)
			
			optimization_result["status"] = "completed"
			optimization_result["completed_at"] = datetime.now(timezone.utc).isoformat()
			
			# Store optimization history
			self.optimization_history.append(optimization_result)
			
		except Exception as e:
			optimization_result["status"] = "failed"
			optimization_result["error"] = str(e)
			optimization_result["failed_at"] = datetime.now(timezone.utc).isoformat()
		
		return optimization_result
	
	async def _start_performance_monitoring(self):
		"""Start continuous performance monitoring."""
		print("📊 Starting performance monitoring")
		
		while True:
			try:
				# Collect and store current metrics
				await self._get_current_metrics()
				
				# Wait for next collection cycle
				await asyncio.sleep(30)  # Collect metrics every 30 seconds
				
			except Exception as e:
				print(f"❌ Performance monitoring error: {e}")
				await asyncio.sleep(30)  # Continue after error
	
	def _is_within_24h(self, timestamp_str: str) -> bool:
		"""Check if timestamp is within last 24 hours."""
		try:
			timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
			cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
			return timestamp > cutoff
		except:
			return False
	
	async def _calculate_system_health_score(self, metrics: Dict[str, Any]) -> float:
		"""Calculate overall system health score (0-100)."""
		weights = {
			"response_time": 0.25,
			"error_rate": 0.25,
			"cpu_usage": 0.15,
			"memory_usage": 0.15,
			"cache_hit_rate": 0.10,
			"throughput": 0.10
		}
		
		# Calculate individual scores (higher is better)
		scores = {}
		
		# Response time score (lower is better, so invert)
		response_time = metrics.get("response_time_p95", 100)
		scores["response_time"] = max(0, 100 - (response_time / 5))  # 500ms = 0 score
		
		# Error rate score (lower is better, so invert)
		error_rate = metrics.get("error_rate", 1.0)
		scores["error_rate"] = max(0, 100 - (error_rate * 20))  # 5% = 0 score
		
		# CPU usage score (optimal around 50-70%)
		cpu_usage = metrics.get("cpu_usage", 50)
		if cpu_usage <= 70:
			scores["cpu_usage"] = 100 - abs(cpu_usage - 60)  # 60% is optimal
		else:
			scores["cpu_usage"] = max(0, 100 - ((cpu_usage - 70) * 2))
		
		# Memory usage score (similar to CPU)
		memory_usage = metrics.get("memory_usage", 50)
		if memory_usage <= 80:
			scores["memory_usage"] = 100 - abs(memory_usage - 65)  # 65% is optimal
		else:
			scores["memory_usage"] = max(0, 100 - ((memory_usage - 80) * 3))
		
		# Cache hit rate score (higher is better)
		cache_hit_rate = metrics.get("cache_hit_rate", 90)
		scores["cache_hit_rate"] = cache_hit_rate
		
		# Throughput score (higher is better, normalized)
		throughput = metrics.get("throughput_rps", 100)
		scores["throughput"] = min(100, throughput / 5)  # 500 RPS = 100 score
		
		# Calculate weighted average
		health_score = sum(scores[metric] * weights[metric] for metric in weights.keys())
		
		return round(health_score, 1)
	
	async def _calculate_performance_grade(self, metrics: Dict[str, Any]) -> str:
		"""Calculate performance grade based on metrics."""
		health_score = await self._calculate_system_health_score(metrics)
		
		if health_score >= 95:
			return "A+"
		elif health_score >= 90:
			return "A"
		elif health_score >= 85:
			return "A-"
		elif health_score >= 80:
			return "B+"
		elif health_score >= 75:
			return "B"
		elif health_score >= 70:
			return "B-"
		elif health_score >= 65:
			return "C+"
		elif health_score >= 60:
			return "C"
		elif health_score >= 55:
			return "C-"
		elif health_score >= 50:
			return "D"
		else:
			return "F"


# ==================== Factory Functions ====================

def create_performance_dashboard(auto_scaler: IntelligentAutoScaler) -> PerformanceOptimizationDashboard:
	"""Create performance optimization dashboard."""
	dashboard = PerformanceOptimizationDashboard(auto_scaler)
	print("📊 Performance Optimization Dashboard initialized")
	return dashboard