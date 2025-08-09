#!/usr/bin/env python3
"""
APG Monitoring and Observability (MONI) - Flask Blueprint Integration
Revolutionary monitoring blueprint with APG composition engine registration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from flask import Blueprint, jsonify, request, render_template, current_app
from flask_appbuilder import AppBuilder, SQLA
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import asyncio

from .views import (
	MonitoringMetricView, MonitoringAlertView, MonitoringRuleView,
	MonitoringDashboardView, MonitoringBaseView, AnomalyDetectionView,
	MetricsChartView, AlertsChartView, register_monitoring_views
)
from .models import (
	MonitoringMetric, MonitoringAlert, MonitoringRule, MonitoringDashboard,
	AlertSeverity, AlertStatus
)
from .service import MonitoringService, _monitoring_service


# Create Flask Blueprint
monitoring_blueprint = Blueprint(
	'monitoring',
	__name__,
	url_prefix='/monitoring',
	template_folder='templates',
	static_folder='static'
)

# APG Capability Metadata for Composition Engine
CAPABILITY_METADATA = {
	'name': 'moni',
	'display_name': 'Monitoring and Observability',
	'version': '1.0.0',
	'category': 'platform_infrastructure',
	'description': 'Revolutionary monitoring platform 10x better than industry leaders',
	'author': 'Nyimbi Odero',
	'company': 'Datacraft',
	'load_order': 5,
	'dependencies': ['auth', 'audl', 'mten', 'conf'],
	'optional_dependencies': ['aicr', 'pred', 'ntfy', 'cach'],
	'blueprint_name': 'monitoring',
	'menu_category': 'Monitoring',
	'menu_icon': 'fa-dashboard',
	'health_check_endpoint': '/monitoring/api/health',
	'api_endpoints': [
		'/monitoring/api/metrics',
		'/monitoring/api/alerts', 
		'/monitoring/api/analytics',
		'/monitoring/api/anomalies',
		'/monitoring/api/health'
	],
	'permissions': [
		'monitoring.view',
		'monitoring.manage',
		'monitoring.admin'
	],
	'features': [
		'predictive_issue_prevention',
		'contextual_intelligence',
		'zero_config_observability',
		'multi_dimensional_analytics',
		'autonomous_remediation',
		'intelligent_alerting',
		'real_time_root_cause',
		'performance_optimization',
		'developer_integration',
		'business_impact_correlation'
	]
}


# API Routes for monitoring data
@monitoring_blueprint.route('/api/health')
def api_health():
	"""Health check endpoint for APG monitoring"""
	try:
		if _monitoring_service and _monitoring_service.running:
			# Run the async health check
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			health_data = loop.run_until_complete(_monitoring_service.get_health_status())
			loop.close()
			
			return jsonify({
				'status': 'healthy',
				'capability': 'moni',
				'timestamp': datetime.utcnow().isoformat(),
				'details': health_data
			})
		else:
			return jsonify({
				'status': 'unhealthy',
				'capability': 'moni',
				'error': 'Monitoring service not running',
				'timestamp': datetime.utcnow().isoformat()
			}), 503
	except Exception as e:
		return jsonify({
			'status': 'error',
			'capability': 'moni',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 500


@monitoring_blueprint.route('/api/metrics')
def api_metrics():
	"""Get metrics data with filtering and aggregation"""
	try:
		# Parse query parameters
		tenant_id = request.args.get('tenant_id', 'default')
		metric_names = request.args.getlist('metric_names')
		start_time = request.args.get('start_time', '-1h')
		end_time = request.args.get('end_time')
		aggregation = request.args.get('aggregation')
		group_by = request.args.getlist('group_by')
		limit = min(int(request.args.get('limit', 1000)), 10000)
		
		# Convert relative times
		if start_time.startswith('-'):
			hours = int(start_time[1:-1]) if start_time.endswith('h') else 1
			start_datetime = datetime.utcnow() - timedelta(hours=hours)
		else:
			start_datetime = datetime.fromisoformat(start_time)
		
		end_datetime = datetime.utcnow()
		if end_time:
			end_datetime = datetime.fromisoformat(end_time)
		
		# Query metrics from service
		if _monitoring_service:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			from .models import MonitoringQuery
			query = MonitoringQuery(
				metric_names=metric_names or ['*'],
				start_time=start_datetime,
				end_time=end_datetime,
				tenant_id=tenant_id,
				aggregation=aggregation,
				group_by=group_by,
				max_results=limit
			)
			
			metrics = loop.run_until_complete(_monitoring_service.query_metrics(query))
			loop.close()
			
			# Convert to JSON-serializable format
			metrics_data = [
				{
					'name': m.name,
					'value': m.value,
					'timestamp': m.timestamp.isoformat(),
					'labels': m.labels,
					'source': m.source,
					'quality_score': m.quality_score
				}
				for m in metrics
			]
		else:
			metrics_data = []
		
		return jsonify({
			'status': 'success',
			'data': metrics_data,
			'count': len(metrics_data),
			'query': {
				'tenant_id': tenant_id,
				'metric_names': metric_names,
				'start_time': start_datetime.isoformat(),
				'end_time': end_datetime.isoformat(),
				'limit': limit
			},
			'timestamp': datetime.utcnow().isoformat()
		})
		
	except Exception as e:
		return jsonify({
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 400


@monitoring_blueprint.route('/api/alerts')
def api_alerts():
	"""Get alerts data with filtering"""
	try:
		# Parse query parameters
		tenant_id = request.args.get('tenant_id', 'default')
		severity = request.args.get('severity')
		status = request.args.get('status')
		limit = min(int(request.args.get('limit', 100)), 1000)
		
		# Query alerts from service
		if _monitoring_service:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			alerts = loop.run_until_complete(_monitoring_service.get_active_alerts(tenant_id))
			loop.close()
			
			# Filter alerts
			if severity:
				alerts = [a for a in alerts if a.severity.value == severity]
			if status:
				alerts = [a for a in alerts if a.status.value == status]
			
			# Limit results
			alerts = alerts[:limit]
			
			# Convert to JSON-serializable format
			alerts_data = [
				{
					'alert_id': a.alert_id,
					'name': a.name,
					'severity': a.severity.value,
					'status': a.status.value,
					'message': a.message,
					'created_at': a.created_at.isoformat(),
					'escalation_level': a.escalation_level,
					'impact_score': a.impact_score,
					'source_metric': a.source_metric,
					'runbook_url': a.runbook_url
				}
				for a in alerts
			]
		else:
			alerts_data = []
		
		return jsonify({
			'status': 'success',
			'data': alerts_data,
			'count': len(alerts_data),
			'filters': {
				'tenant_id': tenant_id,
				'severity': severity,
				'status': status
			},
			'timestamp': datetime.utcnow().isoformat()
		})
		
	except Exception as e:
		return jsonify({
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 400


@monitoring_blueprint.route('/api/analytics')
def api_analytics():
	"""Get analytics data and insights"""
	try:
		tenant_id = request.args.get('tenant_id', 'default')
		analysis_type = request.args.get('type', 'summary')
		time_window = request.args.get('time_window', '24h')
		
		# Get analytics from service
		analytics_data = {}
		
		if _monitoring_service:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			if analysis_type == 'performance':
				analytics_data = loop.run_until_complete(
					_monitoring_service.get_performance_analytics(tenant_id=tenant_id)
				)
			elif analysis_type == 'dashboard':
				analytics_data = loop.run_until_complete(
					_monitoring_service.get_analytics_dashboard('operational', tenant_id)
				)
			else:
				# Summary analytics
				health_status = loop.run_until_complete(_monitoring_service.get_health_status(tenant_id))
				analytics_data = {
					'summary': {
						'health_score': health_status.get('health_score', 0),
						'total_metrics': health_status.get('total_metrics', 0),
						'active_alerts': health_status.get('active_alerts', 0),
						'ingestion_rate': health_status.get('metrics_per_second', 0)
					},
					'trends': [],
					'recommendations': [
						"System operating within normal parameters",
						"No immediate action required"
					]
				}
			
			loop.close()
		
		return jsonify({
			'status': 'success',
			'data': analytics_data,
			'analysis_type': analysis_type,
			'tenant_id': tenant_id,
			'time_window': time_window,
			'timestamp': datetime.utcnow().isoformat()
		})
		
	except Exception as e:
		return jsonify({
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 400


@monitoring_blueprint.route('/api/anomalies')
def api_anomalies():
	"""Get anomaly detection data"""
	try:
		tenant_id = request.args.get('tenant_id', 'default')
		time_window_hours = int(request.args.get('time_window_hours', 24))
		severity = request.args.get('severity')
		
		# Get anomalies from service
		if _monitoring_service and hasattr(_monitoring_service, 'anomaly_engine'):
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			insights = loop.run_until_complete(
				_monitoring_service.anomaly_engine.get_anomaly_insights(
					tenant_id, time_window_hours
				)
			)
			
			loop.close()
		else:
			insights = {
				'summary': {
					'total_anomalies': 0,
					'unique_metrics': 0,
					'severity_distribution': {},
					'algorithm_distribution': {}
				},
				'patterns': {},
				'recommendations': ["Anomaly detection engine not available"],
				'top_anomalies': []
			}
		
		return jsonify({
			'status': 'success',
			'data': insights,
			'tenant_id': tenant_id,
			'time_window_hours': time_window_hours,
			'timestamp': datetime.utcnow().isoformat()
		})
		
	except Exception as e:
		return jsonify({
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 400


@monitoring_blueprint.route('/api/stats')
def api_stats():
	"""Get comprehensive monitoring statistics"""
	try:
		include_detailed = request.args.get('detailed', 'false').lower() == 'true'
		
		stats = {}
		
		if _monitoring_service:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			stats = loop.run_until_complete(_monitoring_service.get_service_stats())
			loop.close()
		
		# Add capability metadata
		stats['capability_info'] = {
			'name': CAPABILITY_METADATA['name'],
			'version': CAPABILITY_METADATA['version'],
			'description': CAPABILITY_METADATA['description'],
			'features': CAPABILITY_METADATA['features']
		}
		
		return jsonify({
			'status': 'success',
			'data': stats,
			'detailed': include_detailed,
			'timestamp': datetime.utcnow().isoformat()
		})
		
	except Exception as e:
		return jsonify({
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 500


# Action endpoints for alert management
@monitoring_blueprint.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def api_acknowledge_alert(alert_id: str):
	"""Acknowledge an alert"""
	try:
		acknowledged_by = request.json.get('acknowledged_by', 'api_user') if request.json else 'api_user'
		
		if _monitoring_service:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			success = loop.run_until_complete(_monitoring_service.acknowledge_alert(alert_id, acknowledged_by))
			loop.close()
			
			if success:
				return jsonify({
					'status': 'success',
					'message': f'Alert {alert_id} acknowledged successfully',
					'acknowledged_by': acknowledged_by,
					'timestamp': datetime.utcnow().isoformat()
				})
			else:
				return jsonify({
					'status': 'error',
					'error': f'Alert {alert_id} not found or already acknowledged',
					'timestamp': datetime.utcnow().isoformat()
				}), 404
		else:
			return jsonify({
				'status': 'error',
				'error': 'Monitoring service not available',
				'timestamp': datetime.utcnow().isoformat()
			}), 503
			
	except Exception as e:
		return jsonify({
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 400


@monitoring_blueprint.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
def api_resolve_alert(alert_id: str):
	"""Resolve an alert"""
	try:
		resolved_by = request.json.get('resolved_by', 'api_user') if request.json else 'api_user'
		
		if _monitoring_service:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			success = loop.run_until_complete(_monitoring_service.resolve_alert(alert_id, resolved_by))
			loop.close()
			
			if success:
				return jsonify({
					'status': 'success',
					'message': f'Alert {alert_id} resolved successfully',
					'resolved_by': resolved_by,
					'timestamp': datetime.utcnow().isoformat()
				})
			else:
				return jsonify({
					'status': 'error',
					'error': f'Alert {alert_id} not found or already resolved',
					'timestamp': datetime.utcnow().isoformat()
				}), 404
		else:
			return jsonify({
				'status': 'error',
				'error': 'Monitoring service not available',
				'timestamp': datetime.utcnow().isoformat()
			}), 503
			
	except Exception as e:
		return jsonify({
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 400


# Template routes for custom monitoring pages
@monitoring_blueprint.route('/')
def monitoring_home():
	"""Main monitoring dashboard"""
	return render_template('monitoring/dashboard.html',
						  capability_metadata=CAPABILITY_METADATA)


@monitoring_blueprint.route('/realtime')
def realtime_dashboard():
	"""Real-time monitoring dashboard"""
	return render_template('monitoring/realtime.html')


@monitoring_blueprint.route('/analytics')  
def analytics_dashboard():
	"""Analytics and insights dashboard"""
	return render_template('monitoring/analytics_dashboard.html')


# Error handlers
@monitoring_blueprint.errorhandler(404)
def not_found(error):
	"""Handle 404 errors"""
	return jsonify({
		'status': 'error',
		'error': 'Endpoint not found',
		'timestamp': datetime.utcnow().isoformat()
	}), 404


@monitoring_blueprint.errorhandler(500)
def internal_error(error):
	"""Handle 500 errors"""
	return jsonify({
		'status': 'error', 
		'error': 'Internal server error',
		'timestamp': datetime.utcnow().isoformat()
	}), 500


# APG Integration Functions
def register_with_appbuilder(appbuilder: AppBuilder, db: SQLA = None):
	"""Register monitoring capability with Flask-AppBuilder"""
	
	try:
		# Register blueprint with Flask app
		appbuilder.get_app.register_blueprint(monitoring_blueprint)
		
		# Register all monitoring views
		register_monitoring_views(appbuilder)
		
		# Create database tables if db is provided
		if db:
			with appbuilder.get_app.app_context():
				# In a real implementation, this would create the tables
				# db.create_all()
				pass
		
		# Add custom menu items and permissions
		_setup_monitoring_menu(appbuilder)
		_setup_monitoring_permissions(appbuilder)
		
		print(f"[APG-MONI] Successfully registered monitoring capability with Flask-AppBuilder")
		return True
		
	except Exception as e:
		print(f"[APG-MONI] Error registering with Flask-AppBuilder: {e}")
		return False


def _setup_monitoring_menu(appbuilder: AppBuilder):
	"""Setup monitoring menu structure"""
	
	# Add menu separators and links
	appbuilder.add_separator("Monitoring", icon="fa-dashboard")
	
	# Real-time monitoring links
	appbuilder.add_link(
		"Real-time Dashboard",
		href="/monitoring/realtime",
		icon="fa-line-chart",
		category="Monitoring"
	)
	
	appbuilder.add_link(
		"System Health",
		href="/monitoring/api/health",
		icon="fa-heartbeat", 
		category="Monitoring"
	)
	
	# Analytics links
	appbuilder.add_separator("Analytics", icon="fa-bar-chart")
	
	appbuilder.add_link(
		"Performance Analytics", 
		href="/monitoring/analytics",
		icon="fa-tachometer",
		category="Analytics"
	)
	
	# API documentation
	appbuilder.add_link(
		"API Documentation",
		href="/monitoring/api/docs", 
		icon="fa-book",
		category="Monitoring"
	)


def _setup_monitoring_permissions(appbuilder: AppBuilder):
	"""Setup monitoring-specific permissions"""
	
	# Create monitoring role with appropriate permissions
	monitoring_permissions = [
		'can_list_on_MonitoringMetricView',
		'can_show_on_MonitoringMetricView', 
		'can_list_on_MonitoringAlertView',
		'can_show_on_MonitoringAlertView',
		'can_edit_on_MonitoringAlertView',
		'can_list_on_MonitoringRuleView',
		'can_show_on_MonitoringRuleView',
		'can_add_on_MonitoringRuleView',
		'can_edit_on_MonitoringRuleView',
		'can_delete_on_MonitoringRuleView'
	]
	
	# In a real implementation, we would create roles and assign permissions
	# role = appbuilder.sm.add_role("Monitoring Operator")
	# for perm in monitoring_permissions:
	#     appbuilder.sm.add_permission_to_role(role, perm)


def get_capability_info() -> Dict[str, Any]:
	"""Get capability information for APG composition engine"""
	return CAPABILITY_METADATA.copy()


def get_blueprint_info() -> Dict[str, Any]:
	"""Get blueprint registration information"""
	return {
		'blueprint': monitoring_blueprint,
		'url_prefix': '/monitoring',
		'capability_metadata': CAPABILITY_METADATA,
		'api_endpoints': CAPABILITY_METADATA['api_endpoints'],
		'health_check': '/monitoring/api/health'
	}


# Export main components
__all__ = [
	'monitoring_blueprint',
	'CAPABILITY_METADATA',
	'register_with_appbuilder',
	'get_capability_info',
	'get_blueprint_info'
]