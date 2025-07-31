"""
APG API Service Mesh - Comprehensive Observability Dashboard

Real-time monitoring, metrics collection, alerting, and visualization
for the revolutionary service mesh with AI-powered insights.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
import httpx

from ..models import SMService, SMEndpoint, SMMetrics, SMHealthCheck, SMAlert
from ..ai_engine import AnomalyDetectionModel, TrafficPredictionModel


class AlertSeverity(str, Enum):
	"""Alert severity levels."""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


@dataclass
class MetricPoint:
	"""Single metric data point."""
	timestamp: datetime
	value: float
	labels: Dict[str, str]
	metadata: Dict[str, Any]


@dataclass
class DashboardWidget:
	"""Dashboard widget configuration."""
	widget_id: str
	title: str
	widget_type: str  # "chart", "metric", "alert", "3d_topology"
	config: Dict[str, Any]
	data_source: str
	refresh_interval_seconds: int


class ObservabilityCollector:
	"""Collects metrics from various sources for observability."""
	
	def __init__(
		self,
		db_session: AsyncSession,
		redis_client: redis.Redis
	):
		self.db_session = db_session
		self.redis_client = redis_client
		self.anomaly_model = AnomalyDetectionModel()
		self.traffic_model = TrafficPredictionModel()
		
		# Metric aggregation buffers
		self._metric_buffer: List[MetricPoint] = []
		self._buffer_size = 1000
	
	async def collect_service_metrics(self, tenant_id: str) -> Dict[str, Any]:
		"""Collect comprehensive service metrics."""
		# Get recent metrics from database
		since = datetime.now(timezone.utc) - timedelta(hours=1)
		
		result = await self.db_session.execute(
			select(
				SMMetrics.service_id,
				func.count(SMMetrics.id).label('request_count'),
				func.avg(SMMetrics.response_time_ms).label('avg_response_time'),
				func.percentile_cont(0.95).within_group(SMMetrics.response_time_ms).label('p95_response_time'),
				func.sum(SMMetrics.error_count).label('total_errors'),
				func.max(SMMetrics.timestamp).label('last_update')
			).where(
				and_(
					SMMetrics.tenant_id == tenant_id,
					SMMetrics.timestamp >= since
				)
			).group_by(SMMetrics.service_id)
		)
		
		service_metrics = {}
		for row in result:
			service_metrics[row.service_id] = {
				'request_count': row.request_count,
				'avg_response_time': float(row.avg_response_time or 0),
				'p95_response_time': float(row.p95_response_time or 0),
				'error_rate': (row.total_errors / row.request_count * 100) if row.request_count > 0 else 0,
				'last_update': row.last_update
			}
		
		# Get real-time metrics from Redis
		for service_id in service_metrics.keys():
			redis_metrics = await self._get_redis_metrics(service_id)
			service_metrics[service_id].update(redis_metrics)
		
		return service_metrics
	
	async def _get_redis_metrics(self, service_id: str) -> Dict[str, Any]:
		"""Get real-time metrics from Redis."""
		pipe = self.redis_client.pipeline()
		
		# Get various metrics
		pipe.get(f"metrics:requests:total:{service_id}")
		pipe.get(f"metrics:errors:total:{service_id}")
		pipe.lrange(f"metrics:response_times:{service_id}", 0, 99)  # Last 100 response times
		
		results = await pipe.execute()
		
		total_requests = int(results[0] or 0)
		total_errors = int(results[1] or 0)
		response_times = [float(rt) for rt in results[2] if rt]
		
		metrics = {
			'real_time_requests': total_requests,
			'real_time_errors': total_errors,
			'current_rps': 0,  # Would calculate from rate of change
			'response_time_trend': response_times[-10:] if response_times else []
		}
		
		return metrics
	
	async def collect_health_metrics(self, tenant_id: str) -> Dict[str, Any]:
		"""Collect service health metrics."""
		# Get recent health checks
		since = datetime.now(timezone.utc) - timedelta(minutes=15)
		
		result = await self.db_session.execute(
			select(SMHealthCheck).where(
				and_(
					SMHealthCheck.checked_at >= since,
					SMHealthCheck.service_id.in_(
						select(SMService.id).where(SMService.tenant_id == tenant_id)
					)
				)
			).order_by(SMHealthCheck.checked_at.desc())
		)
		
		health_checks = result.scalars().all()
		
		# Aggregate health data
		service_health = {}
		for check in health_checks:
			service_id = check.service_id
			if service_id not in service_health:
				service_health[service_id] = {
					'healthy_checks': 0,
					'unhealthy_checks': 0,
					'total_checks': 0,
					'avg_response_time': 0,
					'last_check': None
				}
			
			service_health[service_id]['total_checks'] += 1
			if check.health_status == 'healthy':
				service_health[service_id]['healthy_checks'] += 1
			else:
				service_health[service_id]['unhealthy_checks'] += 1
			
			service_health[service_id]['avg_response_time'] += check.response_time_ms or 0
			service_health[service_id]['last_check'] = check.checked_at
		
		# Calculate health percentages
		for service_id, health in service_health.items():
			if health['total_checks'] > 0:
				health['health_percentage'] = (health['healthy_checks'] / health['total_checks']) * 100
				health['avg_response_time'] /= health['total_checks']
			else:
				health['health_percentage'] = 100
		
		return service_health
	
	async def collect_topology_metrics(self, tenant_id: str) -> Dict[str, Any]:
		"""Collect service mesh topology metrics."""
		# Get services and their connections
		services_result = await self.db_session.execute(
			select(SMService).where(SMService.tenant_id == tenant_id)
		)
		services = services_result.scalars().all()
		
		topology_data = {
			'nodes': [],
			'edges': [],
			'clusters': {},
			'metrics': {}
		}
		
		# Build nodes
		for service in services:
			topology_data['nodes'].append({
				'id': service.id,
				'name': service.name,
				'namespace': service.namespace,
				'version': service.version,
				'status': service.status,
				'endpoints_count': len(service.endpoints) if service.endpoints else 0
			})
		
		# Analyze connections based on metrics
		for service in services:
			service_metrics = await self._get_redis_metrics(service.id)
			topology_data['metrics'][service.id] = service_metrics
		
		return topology_data
	
	async def collect_ai_insights(self, tenant_id: str) -> Dict[str, Any]:
		"""Collect AI-powered insights and predictions."""
		# Get service metrics for AI analysis
		service_metrics = await self.collect_service_metrics(tenant_id)
		
		# Run anomaly detection
		anomalies = await self.anomaly_model.detect_anomalies(service_metrics)
		
		# Generate traffic predictions
		traffic_predictions = await self.traffic_model.predict_traffic(
			service_metrics, 
			horizon_hours=4
		)
		
		# Generate recommendations
		recommendations = await self._generate_ai_recommendations(
			service_metrics, anomalies, traffic_predictions
		)
		
		return {
			'anomalies': anomalies,
			'traffic_predictions': traffic_predictions,
			'recommendations': recommendations,
			'analysis_timestamp': datetime.now(timezone.utc).isoformat()
		}
	
	async def _generate_ai_recommendations(
		self,
		service_metrics: Dict[str, Any],
		anomalies: List[Dict[str, Any]],
		predictions: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		"""Generate AI-powered recommendations."""
		recommendations = []
		
		# Analyze service performance
		for service_id, metrics in service_metrics.items():
			# High error rate recommendation
			if metrics.get('error_rate', 0) > 5:
				recommendations.append({
					'type': 'performance',
					'severity': 'high',
					'service': service_id,
					'title': 'High Error Rate Detected',
					'description': f"Service {service_id} has {metrics['error_rate']:.1f}% error rate",
					'action': 'Enable circuit breaker and investigate root cause',
					'priority': 1
				})
			
			# High latency recommendation
			if metrics.get('p95_response_time', 0) > 1000:
				recommendations.append({
					'type': 'performance',
					'severity': 'medium',
					'service': service_id,
					'title': 'High Latency Detected',
					'description': f"P95 latency is {metrics['p95_response_time']:.1f}ms",
					'action': 'Consider scaling service or optimizing performance',
					'priority': 2
				})
		
		# Anomaly-based recommendations
		for anomaly in anomalies:
			recommendations.append({
				'type': 'anomaly',
				'severity': 'high',
				'service': anomaly.get('service'),
				'title': 'Anomaly Detected',
				'description': anomaly.get('description', 'Unusual behavior detected'),
				'action': 'Investigate anomaly and consider preventive measures',
				'priority': 1
			})
		
		return sorted(recommendations, key=lambda x: x['priority'])


class ObservabilityDashboard:
	"""Comprehensive observability dashboard using Dash."""
	
	def __init__(
		self,
		collector: ObservabilityCollector,
		port: int = 8050
	):
		self.collector = collector
		self.port = port
		
		# Initialize Dash app
		self.app = dash.Dash(
			__name__,
			external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
		)
		
		# Setup layout and callbacks
		self._setup_layout()
		self._setup_callbacks()
		
		# Dashboard state
		self.current_tenant = "default-tenant"
		self.refresh_interval = 30  # seconds
	
	def _setup_layout(self):
		"""Setup the dashboard layout."""
		self.app.layout = dbc.Container([
			# Header
			dbc.Row([
				dbc.Col([
					html.H1("🚀 APG Service Mesh - Observability Dashboard", 
						   className="text-primary mb-0"),
					html.P("Revolutionary AI-Powered Service Mesh Monitoring", 
						  className="text-muted")
				], width=8),
				dbc.Col([
					dbc.Badge("LIVE", color="success", className="me-2", 
							id="status-badge"),
					html.Span(id="last-update", className="text-muted small")
				], width=4, className="text-end")
			], className="mb-4"),
			
			# Control Panel
			dbc.Row([
				dbc.Col([
					dbc.Card([
						dbc.CardBody([
							html.H5("🎛️ Control Panel", className="card-title"),
							dbc.Row([
								dbc.Col([
									dbc.Label("Tenant ID"),
									dbc.Input(
										id="tenant-input",
										value=self.current_tenant,
										placeholder="Enter tenant ID"
									)
								], width=6),
								dbc.Col([
									dbc.Label("Refresh Interval (s)"),
									dbc.Input(
										id="refresh-input",
										type="number",
										value=self.refresh_interval,
										min=5, max=300
									)
								], width=6),
							]),
							html.Hr(),
							dbc.ButtonGroup([
								dbc.Button("🔄 Refresh", id="refresh-btn", color="primary"),
								dbc.Button("🎤 Voice Command", id="voice-btn", color="info"),
								dbc.Button("🎮 3D View", id="3d-btn", color="success"),
							])
						])
					])
				])
			], className="mb-4"),
			
			# Key Metrics Row
			html.Div(id="metrics-cards"),
			
			# Charts Row
			dbc.Row([
				dbc.Col([
					dbc.Card([
						dbc.CardHeader("📊 Request Rate & Latency"),
						dbc.CardBody([
							dcc.Graph(id="requests-latency-chart")
						])
					])
				], width=6),
				dbc.Col([
					dbc.Card([
						dbc.CardHeader("🚨 Error Rate & Success Rate"),
						dbc.CardBody([
							dcc.Graph(id="error-success-chart")
						])
					])
				], width=6),
			], className="mb-4"),
			
			# Service Health Row
			dbc.Row([
				dbc.Col([
					dbc.Card([
						dbc.CardHeader("💚 Service Health Status"),
						dbc.CardBody([
							dcc.Graph(id="health-status-chart")
						])
					])
				], width=8),
				dbc.Col([
					dbc.Card([
						dbc.CardHeader("🧠 AI Insights"),
						dbc.CardBody([
							html.Div(id="ai-insights")
						])
					])
				], width=4),
			], className="mb-4"),
			
			# Topology and Alerts Row
			dbc.Row([
				dbc.Col([
					dbc.Card([
						dbc.CardHeader("🌐 Service Mesh Topology"),
						dbc.CardBody([
							dcc.Graph(id="topology-chart")
						])
					])
				], width=8),
				dbc.Col([
					dbc.Card([
						dbc.CardHeader("⚠️ Alerts & Recommendations"),
						dbc.CardBody([
							html.Div(id="alerts-panel")
						])
					])
				], width=4),
			], className="mb-4"),
			
			# Auto-refresh component
			dcc.Interval(
				id='interval-component',
				interval=self.refresh_interval * 1000,  # milliseconds
				n_intervals=0
			),
			
			# Store components for data sharing
			dcc.Store(id='dashboard-data'),
			dcc.Store(id='ai-data')
			
		], fluid=True)
	
	def _setup_callbacks(self):
		"""Setup dashboard callbacks."""
		
		@self.app.callback(
			[Output('dashboard-data', 'data'),
			 Output('ai-data', 'data'),
			 Output('last-update', 'children')],
			[Input('interval-component', 'n_intervals'),
			 Input('refresh-btn', 'n_clicks')],
			[State('tenant-input', 'value')]
		)
		def update_data(n_intervals, refresh_clicks, tenant_id):
			"""Update dashboard data."""
			if not tenant_id:
				tenant_id = self.current_tenant
			
			# Run async data collection
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				# Collect all metrics
				service_metrics = loop.run_until_complete(
					self.collector.collect_service_metrics(tenant_id)
				)
				health_metrics = loop.run_until_complete(
					self.collector.collect_health_metrics(tenant_id)
				)
				topology_metrics = loop.run_until_complete(
					self.collector.collect_topology_metrics(tenant_id)
				)
				ai_insights = loop.run_until_complete(
					self.collector.collect_ai_insights(tenant_id)
				)
				
				dashboard_data = {
					'service_metrics': service_metrics,
					'health_metrics': health_metrics,
					'topology_metrics': topology_metrics,
					'timestamp': datetime.now().isoformat()
				}
				
				last_update = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
				
				return dashboard_data, ai_insights, last_update
				
			finally:
				loop.close()
		
		@self.app.callback(
			Output('metrics-cards', 'children'),
			[Input('dashboard-data', 'data')]
		)
		def update_metrics_cards(data):
			"""Update key metrics cards."""
			if not data or not data.get('service_metrics'):
				return html.Div("No data available")
			
			service_metrics = data['service_metrics']
			
			# Calculate aggregate metrics
			total_requests = sum(m.get('request_count', 0) for m in service_metrics.values())
			avg_response_time = np.mean([m.get('avg_response_time', 0) for m in service_metrics.values()])
			total_services = len(service_metrics)
			avg_error_rate = np.mean([m.get('error_rate', 0) for m in service_metrics.values()])
			
			cards = dbc.Row([
				dbc.Col([
					dbc.Card([
						dbc.CardBody([
							html.H4(f"{total_requests:,}", className="text-primary"),
							html.P("Total Requests", className="text-muted mb-0")
						])
					], className="text-center")
				], width=3),
				dbc.Col([
					dbc.Card([
						dbc.CardBody([
							html.H4(f"{avg_response_time:.1f}ms", className="text-info"),
							html.P("Avg Response Time", className="text-muted mb-0")
						])
					], className="text-center")
				], width=3),
				dbc.Col([
					dbc.Card([
						dbc.CardBody([
							html.H4(f"{total_services}", className="text-success"),
							html.P("Active Services", className="text-muted mb-0")
						])
					], className="text-center")
				], width=3),
				dbc.Col([
					dbc.Card([
						dbc.CardBody([
							html.H4(f"{avg_error_rate:.1f}%", 
								   className="text-danger" if avg_error_rate > 5 else "text-success"),
							html.P("Error Rate", className="text-muted mb-0")
						])
					], className="text-center")
				], width=3),
			], className="mb-4")
			
			return cards
		
		@self.app.callback(
			Output('requests-latency-chart', 'figure'),
			[Input('dashboard-data', 'data')]
		)
		def update_requests_latency_chart(data):
			"""Update requests and latency chart."""
			if not data or not data.get('service_metrics'):
				return go.Figure()
			
			service_metrics = data['service_metrics']
			
			services = list(service_metrics.keys())
			requests = [service_metrics[s].get('request_count', 0) for s in services]
			latencies = [service_metrics[s].get('avg_response_time', 0) for s in services]
			
			fig = make_subplots(
				rows=1, cols=2,
				subplot_titles=('Request Count', 'Average Response Time'),
				specs=[[{"secondary_y": False}, {"secondary_y": False}]]
			)
			
			# Requests bar chart
			fig.add_trace(
				go.Bar(x=services, y=requests, name='Requests', 
					  marker_color='#2E86AB'),
				row=1, col=1
			)
			
			# Latency bar chart
			fig.add_trace(
				go.Bar(x=services, y=latencies, name='Latency (ms)',
					  marker_color='#A23B72'),
				row=1, col=2
			)
			
			fig.update_layout(
				height=400,
				showlegend=False,
				title_text="Request Rate & Response Time by Service"
			)
			
			return fig
		
		@self.app.callback(
			Output('health-status-chart', 'figure'),
			[Input('dashboard-data', 'data')]
		)
		def update_health_status_chart(data):
			"""Update service health status chart."""
			if not data or not data.get('health_metrics'):
				return go.Figure()
			
			health_metrics = data['health_metrics']
			
			services = list(health_metrics.keys())
			health_percentages = [health_metrics[s].get('health_percentage', 100) for s in services]
			
			# Create color scale based on health
			colors = ['#d4edda' if h >= 95 else '#fff3cd' if h >= 80 else '#f8d7da' 
					 for h in health_percentages]
			
			fig = go.Figure(data=[
				go.Bar(x=services, y=health_percentages, 
					  marker_color=colors,
					  text=[f"{h:.1f}%" for h in health_percentages],
					  textposition='auto')
			])
			
			fig.update_layout(
				title="Service Health Status",
				yaxis_title="Health Percentage",
				yaxis_range=[0, 100],
				height=400
			)
			
			return fig
		
		@self.app.callback(
			Output('ai-insights', 'children'),
			[Input('ai-data', 'data')]
		)
		def update_ai_insights(ai_data):
			"""Update AI insights panel."""
			if not ai_data:
				return html.Div("Loading AI insights...")
			
			recommendations = ai_data.get('recommendations', [])
			anomalies = ai_data.get('anomalies', [])
			
			insights = []
			
			# Add anomalies
			if anomalies:
				insights.append(html.H6("🔍 Anomalies Detected"))
				for anomaly in anomalies[:3]:  # Show top 3
					insights.append(
						dbc.Alert([
							html.Strong(anomaly.get('title', 'Anomaly')),
							html.Br(),
							anomaly.get('description', 'Unknown anomaly')
						], color="warning", className="mb-2")
					)
			
			# Add recommendations
			if recommendations:
				insights.append(html.H6("💡 AI Recommendations"))
				for rec in recommendations[:3]:  # Show top 3
					color = {
						'critical': 'danger',
						'high': 'warning', 
						'medium': 'info',
						'low': 'light'
					}.get(rec.get('severity'), 'info')
					
					insights.append(
						dbc.Alert([
							html.Strong(rec.get('title', 'Recommendation')),
							html.Br(),
							html.Small(rec.get('description', '')),
							html.Br(),
							html.Em(f"Action: {rec.get('action', '')}")
						], color=color, className="mb-2")
					)
			
			if not insights:
				insights.append(
					dbc.Alert("✅ All systems operating normally", color="success")
				)
			
			return html.Div(insights)
		
		@self.app.callback(
			Output('topology-chart', 'figure'),
			[Input('dashboard-data', 'data')]
		)
		def update_topology_chart(data):
			"""Update service mesh topology chart."""
			if not data or not data.get('topology_metrics'):
				return go.Figure()
			
			topology = data['topology_metrics']
			nodes = topology.get('nodes', [])
			
			if not nodes:
				return go.Figure()
			
			# Create network graph
			node_trace = go.Scatter(
				x=[i for i in range(len(nodes))],
				y=[0 for _ in nodes],
				mode='markers+text',
				text=[node['name'] for node in nodes],
				textposition="middle center",
				marker=dict(
					size=20,
					color=[
						'green' if node['status'] == 'active' else 'red' 
						for node in nodes
					]
				),
				name="Services"
			)
			
			fig = go.Figure(data=[node_trace])
			fig.update_layout(
				title="Service Mesh Topology (2D View)",
				showlegend=False,
				height=400,
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
			)
			
			return fig
	
	def run(self, debug: bool = False):
		"""Run the dashboard server."""
		print(f"🚀 Starting APG Service Mesh Observability Dashboard on port {self.port}")
		print(f"📊 Dashboard URL: http://localhost:{self.port}")
		
		self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')


class AlertManager:
	"""Manages alerts and notifications."""
	
	def __init__(self, redis_client: redis.Redis):
		self.redis_client = redis_client
		self.alert_rules: List[Dict[str, Any]] = []
	
	async def create_alert_rule(
		self,
		name: str,
		condition: str,
		severity: AlertSeverity,
		notification_channels: List[str]
	) -> str:
		"""Create a new alert rule."""
		rule_id = f"alert_rule_{int(time.time())}"
		
		rule = {
			'id': rule_id,
			'name': name,
			'condition': condition,
			'severity': severity.value,
			'notification_channels': notification_channels,
			'created_at': datetime.now(timezone.utc).isoformat(),
			'enabled': True
		}
		
		await self.redis_client.setex(
			f"alert_rule:{rule_id}",
			86400,  # 24 hours
			json.dumps(rule)
		)
		
		self.alert_rules.append(rule)
		return rule_id
	
	async def evaluate_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Evaluate alert conditions against current metrics."""
		alerts = []
		
		for rule in self.alert_rules:
			if not rule.get('enabled', True):
				continue
			
			# Simple condition evaluation (would be more sophisticated in production)
			if await self._evaluate_condition(rule['condition'], metrics):
				alert = {
					'rule_id': rule['id'],
					'rule_name': rule['name'],
					'severity': rule['severity'], 
					'message': f"Alert triggered: {rule['name']}",
					'timestamp': datetime.now(timezone.utc).isoformat(),
					'metrics': metrics
				}
				alerts.append(alert)
		
		return alerts
	
	async def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
		"""Evaluate alert condition."""
		# Simple condition evaluation - would use a proper expression parser
		try:
			# Example conditions:
			# "error_rate > 5"
			# "avg_response_time > 1000"
			# "request_count < 10"
			
			if "error_rate >" in condition:
				threshold = float(condition.split(">")[1].strip())
				for service_metrics in metrics.values():
					if service_metrics.get('error_rate', 0) > threshold:
						return True
			
			elif "avg_response_time >" in condition:
				threshold = float(condition.split(">")[1].strip())
				for service_metrics in metrics.values():
					if service_metrics.get('avg_response_time', 0) > threshold:
						return True
			
			return False
			
		except Exception:
			return False


async def setup_observability_dashboard():
	"""Setup and start the observability dashboard."""
	# Create database and Redis connections (mock for example)
	from sqlalchemy.ext.asyncio import create_async_engine
	from sqlalchemy.orm import sessionmaker
	
	engine = create_async_engine("sqlite+aiosqlite:///:memory:")
	session_factory = sessionmaker(engine, class_=AsyncSession)
	db_session = session_factory()
	
	redis_client = await redis.from_url("redis://localhost", decode_responses=True)
	
	# Initialize collector and dashboard
	collector = ObservabilityCollector(db_session, redis_client)
	dashboard = ObservabilityDashboard(collector, port=8050)
	
	# Setup alert manager
	alert_manager = AlertManager(redis_client)
	
	# Create some default alert rules
	await alert_manager.create_alert_rule(
		name="High Error Rate",
		condition="error_rate > 5",
		severity=AlertSeverity.HIGH,
		notification_channels=["email", "slack"]
	)
	
	await alert_manager.create_alert_rule(
		name="High Latency",
		condition="avg_response_time > 1000",
		severity=AlertSeverity.MEDIUM,
		notification_channels=["email"]
	)
	
	print("🎯 Observability dashboard setup complete!")
	print("🚀 Starting dashboard...")
	
	# Run dashboard
	dashboard.run(debug=False)


if __name__ == "__main__":
	asyncio.run(setup_observability_dashboard())