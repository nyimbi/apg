"""
Dashboard API Endpoints - APG Payment Gateway

Flask-AppBuilder API endpoints for real-time analytics dashboard.

© 2025 Datacraft. All rights reserved.
"""

from flask import request, jsonify
from flask_appbuilder import BaseView, expose
from flask_appbuilder.security.decorators import has_access
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .realtime_analytics import RealTimeAnalyticsEngine, TimeRange, MetricType
from .database import get_database_service
from .auth import authenticate_api_key, require_permission


class DashboardAPI(BaseView):
	"""Dashboard API endpoints for real-time analytics"""
	
	route_base = "/api/v1/dashboard"
	
	def __init__(self):
		super().__init__()
		self._analytics_engine: Optional[RealTimeAnalyticsEngine] = None
		self._loop = None
	
	def _get_analytics_engine(self) -> RealTimeAnalyticsEngine:
		"""Get or create analytics engine instance"""
		if self._analytics_engine is None:
			from .realtime_analytics import create_analytics_engine
			database_service = get_database_service()
			self._analytics_engine = create_analytics_engine(database_service)
		return self._analytics_engine
	
	def _get_event_loop(self):
		"""Get or create event loop for async operations"""
		if self._loop is None:
			try:
				self._loop = asyncio.get_event_loop()
			except RuntimeError:
				self._loop = asyncio.new_event_loop()
				asyncio.set_event_loop(self._loop)
		return self._loop
	
	def _run_async(self, coro):
		"""Run async coroutine in sync context"""
		loop = self._get_event_loop()
		return loop.run_until_complete(coro)
	
	@expose('/overview', methods=['GET'])
	@has_access
	def dashboard_overview(self):
		"""Get dashboard overview with key metrics"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "dashboard.read"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			# Get merchant_id from query params or user context
			merchant_id = request.args.get('merchant_id')
			if not merchant_id and auth_result["user"]["role"] == "merchant":
				merchant_id = auth_result["user"]["merchant_id"]
			
			analytics_engine = self._get_analytics_engine()
			dashboard_data = self._run_async(analytics_engine.get_real_time_dashboard(merchant_id))
			
			return jsonify({
				"success": True,
				"data": dashboard_data
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/metrics/<metric_type>', methods=['GET'])
	@has_access
	def get_metric_data(self, metric_type: str):
		"""Get specific metric data"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Validate metric type
			try:
				metric_type_enum = MetricType(metric_type)
			except ValueError:
				return jsonify({"error": f"Invalid metric type: {metric_type}"}), 400
			
			# Get parameters
			merchant_id = request.args.get('merchant_id')
			time_range = request.args.get('time_range', TimeRange.LAST_24_HOURS.value)
			
			if not merchant_id and auth_result["user"]["role"] == "merchant":
				merchant_id = auth_result["user"]["merchant_id"]
			
			analytics_engine = self._get_analytics_engine()
			
			# Get metric-specific data
			if metric_type_enum == MetricType.TRANSACTION_VOLUME:
				data = self._run_async(self._get_volume_metrics(analytics_engine, merchant_id, time_range))
			elif metric_type_enum == MetricType.SUCCESS_RATE:
				data = self._run_async(self._get_success_rate_metrics(analytics_engine, merchant_id, time_range))
			elif metric_type_enum == MetricType.FRAUD_RATE:
				data = self._run_async(self._get_fraud_metrics(analytics_engine, merchant_id, time_range))
			elif metric_type_enum == MetricType.REVENUE_ANALYTICS:
				data = self._run_async(self._get_revenue_metrics(analytics_engine, merchant_id, time_range))
			else:
				data = {"message": f"Metric type {metric_type} not yet implemented"}
			
			return jsonify({
				"success": True,
				"metric_type": metric_type,
				"time_range": time_range,
				"data": data
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/charts/<chart_type>', methods=['GET'])
	@has_access
	def get_chart_data(self, chart_type: str):
		"""Get chart data for dashboard visualizations"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			merchant_id = request.args.get('merchant_id')
			if not merchant_id and auth_result["user"]["role"] == "merchant":
				merchant_id = auth_result["user"]["merchant_id"]
			
			analytics_engine = self._get_analytics_engine()
			
			if chart_type == "time_series":
				data = self._run_async(self._get_time_series_chart(analytics_engine, merchant_id))
			elif chart_type == "payment_methods":
				data = self._run_async(self._get_payment_method_chart(analytics_engine, merchant_id))
			elif chart_type == "processor_performance":
				data = self._run_async(self._get_processor_performance_chart(analytics_engine, merchant_id))
			elif chart_type == "geographic":
				data = self._run_async(self._get_geographic_chart(analytics_engine, merchant_id))
			else:
				return jsonify({"error": f"Unknown chart type: {chart_type}"}), 400
			
			return jsonify({
				"success": True,
				"chart_type": chart_type,
				"data": data
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/alerts', methods=['GET'])
	@has_access
	def get_active_alerts(self):
		"""Get active alerts for dashboard"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			merchant_id = request.args.get('merchant_id')
			if not merchant_id and auth_result["user"]["role"] == "merchant":
				merchant_id = auth_result["user"]["merchant_id"]
			
			analytics_engine = self._get_analytics_engine()
			alerts = self._run_async(analytics_engine._get_active_alerts(merchant_id))
			
			return jsonify({
				"success": True,
				"alerts": alerts,
				"total_alerts": len(alerts)
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/alerts/<alert_id>/resolve', methods=['POST'])
	@has_access
	def resolve_alert(self, alert_id: str):
		"""Resolve an active alert"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "alerts.manage"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			analytics_engine = self._get_analytics_engine()
			resolved = self._run_async(analytics_engine.resolve_alert(alert_id))
			
			if resolved:
				return jsonify({
					"success": True,
					"message": f"Alert {alert_id} resolved successfully"
				})
			else:
				return jsonify({
					"success": False,
					"error": "Alert not found or already resolved"
				}), 404
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/reports/custom', methods=['POST'])
	@has_access
	def generate_custom_report(self):
		"""Generate custom analytics report"""
		try:
			# Authenticate request
			auth_result = authenticate_api_key(request)
			if not auth_result["success"]:
				return jsonify({"error": "Authentication required"}), 401
			
			# Check permissions
			if not require_permission(auth_result["user"], "reports.generate"):
				return jsonify({"error": "Insufficient permissions"}), 403
			
			data = request.get_json()
			if not data:
				return jsonify({"error": "Request body required"}), 400
			
			# Validate required fields
			required_fields = ["start_date", "end_date", "report_config"]
			for field in required_fields:
				if field not in data:
					return jsonify({"error": f"Missing required field: {field}"}), 400
			
			merchant_id = data.get('merchant_id')
			if not merchant_id and auth_result["user"]["role"] == "merchant":
				merchant_id = auth_result["user"]["merchant_id"]
			
			analytics_engine = self._get_analytics_engine()
			report = self._run_async(analytics_engine.get_custom_report(
				data["report_config"],
				data["start_date"],
				data["end_date"],
				merchant_id
			))
			
			return jsonify({
				"success": True,
				"report": report
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	@expose('/health', methods=['GET'])
	@has_access
	def analytics_health_check(self):
		"""Get analytics engine health status"""
		try:
			analytics_engine = self._get_analytics_engine()
			
			health_data = {
				"analytics_engine": "healthy" if analytics_engine._running else "stopped",
				"metrics_buffer_size": len(analytics_engine._metrics_buffer),
				"active_alerts": len(analytics_engine._active_alerts),
				"subscribers": sum(len(subs) for subs in analytics_engine._subscribers.values()),
				"cache_size": len(analytics_engine._metric_cache),
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
			
			return jsonify({
				"success": True,
				"health": health_data
			})
		
		except Exception as e:
			return jsonify({
				"success": False,
				"error": str(e)
			}), 500
	
	# Helper methods for metric data
	
	async def _get_volume_metrics(self, analytics_engine: RealTimeAnalyticsEngine, merchant_id: Optional[str], time_range: str) -> Dict[str, Any]:
		"""Get transaction volume metrics"""
		dashboard_data = await analytics_engine.get_real_time_dashboard(merchant_id)
		volume_data = dashboard_data["metrics"]["transaction_volume"]
		
		return {
			"current_hour": volume_data["last_hour"],
			"current_24h": volume_data["last_24h"],
			"trend": volume_data["trend"],
			"chart_data": dashboard_data["charts"]["time_series"]["datasets"]["volume"]
		}
	
	async def _get_success_rate_metrics(self, analytics_engine: RealTimeAnalyticsEngine, merchant_id: Optional[str], time_range: str) -> Dict[str, Any]:
		"""Get success rate metrics"""
		dashboard_data = await analytics_engine.get_real_time_dashboard(merchant_id)
		success_data = dashboard_data["metrics"]["success_rate"]
		
		return {
			"current_hour": success_data["last_hour"],
			"current_24h": success_data["last_24h"],
			"trend": success_data["trend"],
			"chart_data": dashboard_data["charts"]["time_series"]["datasets"]["success_rate"]
		}
	
	async def _get_fraud_metrics(self, analytics_engine: RealTimeAnalyticsEngine, merchant_id: Optional[str], time_range: str) -> Dict[str, Any]:
		"""Get fraud rate metrics"""
		dashboard_data = await analytics_engine.get_real_time_dashboard(merchant_id)
		fraud_data = dashboard_data["metrics"]["fraud_rate"]
		
		return {
			"current_hour": fraud_data["last_hour"],
			"current_24h": fraud_data["last_24h"],
			"chart_data": dashboard_data["charts"]["time_series"]["datasets"]["fraud_rate"]
		}
	
	async def _get_revenue_metrics(self, analytics_engine: RealTimeAnalyticsEngine, merchant_id: Optional[str], time_range: str) -> Dict[str, Any]:
		"""Get revenue metrics"""
		dashboard_data = await analytics_engine.get_real_time_dashboard(merchant_id)
		revenue_data = dashboard_data["metrics"]["revenue"]
		
		return {
			"current_hour": revenue_data["last_hour"],
			"current_24h": revenue_data["last_24h"],
			"trend": revenue_data["trend"],
			"chart_data": dashboard_data["charts"]["time_series"]["datasets"]["revenue"]
		}
	
	async def _get_time_series_chart(self, analytics_engine: RealTimeAnalyticsEngine, merchant_id: Optional[str]) -> Dict[str, Any]:
		"""Get time series chart data"""
		dashboard_data = await analytics_engine.get_real_time_dashboard(merchant_id)
		return dashboard_data["charts"]["time_series"]
	
	async def _get_payment_method_chart(self, analytics_engine: RealTimeAnalyticsEngine, merchant_id: Optional[str]) -> Dict[str, Any]:
		"""Get payment method distribution chart data"""
		dashboard_data = await analytics_engine.get_real_time_dashboard(merchant_id)
		return dashboard_data["charts"]["payment_methods"]
	
	async def _get_processor_performance_chart(self, analytics_engine: RealTimeAnalyticsEngine, merchant_id: Optional[str]) -> Dict[str, Any]:
		"""Get processor performance chart data"""
		dashboard_data = await analytics_engine.get_real_time_dashboard(merchant_id)
		return dashboard_data["charts"]["processor_performance"]
	
	async def _get_geographic_chart(self, analytics_engine: RealTimeAnalyticsEngine, merchant_id: Optional[str]) -> Dict[str, Any]:
		"""Get geographic distribution chart data"""
		dashboard_data = await analytics_engine.get_real_time_dashboard(merchant_id)
		return dashboard_data["charts"]["geographic_distribution"]


class WebSocketDashboard(BaseView):
	"""WebSocket endpoints for real-time dashboard updates"""
	
	route_base = "/ws/dashboard"
	
	def __init__(self):
		super().__init__()
		self._analytics_engine: Optional[RealTimeAnalyticsEngine] = None
		self._active_connections = {}
	
	def _get_analytics_engine(self) -> RealTimeAnalyticsEngine:
		"""Get or create analytics engine instance"""
		if self._analytics_engine is None:
			from .realtime_analytics import create_analytics_engine
			database_service = get_database_service()
			self._analytics_engine = create_analytics_engine(database_service)
		return self._analytics_engine
	
	@expose('/connect', methods=['GET'])
	def websocket_connect(self):
		"""WebSocket connection endpoint for real-time updates"""
		# This would be implemented with flask-socketio or similar
		# For now, return connection info
		return jsonify({
			"success": True,
			"websocket_url": "/ws/dashboard/stream",
			"supported_events": [
				"transaction_metric",
				"processor_metric", 
				"fraud_metric",
				"alert",
				"alert_resolved",
				"insights"
			]
		})
	
	async def handle_websocket_connection(self, websocket, path):
		"""Handle WebSocket connection for real-time updates"""
		# This would be the actual WebSocket handler
		connection_id = f"ws_{datetime.now().timestamp()}"
		
		try:
			analytics_engine = self._get_analytics_engine()
			
			# Subscribe to updates
			await analytics_engine.subscribe_to_updates(
				connection_id,
				lambda event_type, data: self._send_to_websocket(websocket, event_type, data)
			)
			
			self._active_connections[connection_id] = websocket
			
			# Keep connection alive
			async for message in websocket:
				# Handle incoming messages if needed
				pass
		
		except Exception as e:
			print(f"WebSocket error: {e}")
		finally:
			# Cleanup
			if connection_id in self._active_connections:
				del self._active_connections[connection_id]
			
			analytics_engine = self._get_analytics_engine()
			await analytics_engine.unsubscribe_from_updates(connection_id)
	
	def _send_to_websocket(self, websocket, event_type: str, data: Dict[str, Any]):
		"""Send data to WebSocket client"""
		message = {
			"event": event_type,
			"data": data,
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		
		# This would use the actual WebSocket library to send
		# websocket.send(json.dumps(message))
		print(f"📡 WebSocket message: {event_type}")


def _log_dashboard_api_module_loaded():
	"""Log dashboard API module loaded"""
	print("🎛️  Dashboard API module loaded")
	print("   - Real-time analytics endpoints")
	print("   - Interactive dashboard data")
	print("   - WebSocket support for live updates")
	print("   - Custom report generation")
	print("   - Alert management API")


# Execute module loading log
_log_dashboard_api_module_loaded()