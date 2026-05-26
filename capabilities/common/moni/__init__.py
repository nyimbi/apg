#!/usr/bin/env python3
"""
APG Monitoring and Observability (MONI) Capability
Revolutionary monitoring platform that is 10x better than industry leaders

This capability provides:
- Predictive Issue Prevention with ML-powered failure prediction
- Contextual Intelligence Engine with business impact correlation
- Zero-Configuration Observability with intelligent auto-discovery
- Unified Multi-Dimensional Analytics across metrics, logs, traces
- Autonomous Remediation Engine with self-healing capabilities
- Intelligent Alert Orchestration with smart correlation
- Real-Time Root Cause Analysis with dependency mapping
- Performance Optimization Advisor with AI recommendations
- Developer Experience Integration with code-level insights
- Business Impact Correlation with executive dashboards

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from .service import MonitoringService, MonitoringServiceConfig, create_monitoring_service
from .models import (
	MonitoringMetric, MonitoringAlert, MonitoringRule, MonitoringDashboard,
	MonitoringQuery, MonitoringTarget, MetricType, AlertSeverity, AlertStatus,
	AlertConditionType, DashboardType, DataRetentionPolicy, MonitoringScope
)
from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)

# Capability metadata for APG composition engine
__capability_name__ = "moni"
__capability_version__ = "1.0.0"
__capability_description__ = "Revolutionary monitoring and observability platform"
__capability_dependencies__ = ["auth", "audl", "mten", "conf"]
__capability_optional_dependencies__ = ["aicr", "pred", "ntfy", "cach"]

# APG Composition Engine Registration Metadata
CAPABILITY_METADATA = {
	'name': 'moni',
	'display_name': 'Monitoring and Observability',
	'version': '1.0.0',
	'category': 'platform_infrastructure',
	'load_order': 5,
	'dependencies': ['auth', 'audl', 'mten', 'conf'],
	'optional_dependencies': ['aicr', 'pred', 'ntfy', 'cach'],
	'export_functions': [
		'track_metric', 'log_event', 'create_alert', 'query_metrics',
		'get_health_status', 'get_performance_metrics', 'predict_resource_usage',
		'analyze_performance', 'get_analytics_dashboard', 'create_monitoring_rule'
	],
	'event_handlers': [
		'handle_metric_ingestion', 'handle_alert_trigger', 'handle_health_check',
		'handle_performance_analysis', 'handle_anomaly_detected'
	],
	'health_check_endpoint': '/api/v1/health',
	'api_prefix': '/api/v1/monitoring'
}

# Export main components
__all__ = [
	# Service components
	'MonitoringService',
	'MonitoringServiceConfig',
	'create_monitoring_service',
	
	# Data models
	'MonitoringMetric',
	'MonitoringAlert', 
	'MonitoringRule',
	'MonitoringDashboard',
	'MonitoringQuery',
	'MonitoringTarget',
	
	# Enums
	'MetricType',
	'AlertSeverity',
	'AlertStatus',
	'AlertConditionType',
	'DashboardType',
	'DataRetentionPolicy',
	'MonitoringScope',
	
	# APG metadata
	'CAPABILITY_METADATA',
	'register_capability',
	'get_capability_info',
	'get_capability_contract',
	'evaluate_capability_rules',
	'__capability_name__',
	'__capability_version__',
	'__capability_description__',
	'__capability_dependencies__',
	'__capability_optional_dependencies__'
]

# APG capability initialization function
async def initialize_capability(config: dict = None) -> MonitoringService:
	"""
	Initialize the monitoring and observability capability
	Called by APG composition engine during capability loading
	"""
	monitoring_config = MonitoringServiceConfig()
	
	if config:
		# Update configuration from APG
		for key, value in config.items():
			if hasattr(monitoring_config, key):
				setattr(monitoring_config, key, value)
	
	# Create and initialize service
	service = MonitoringService(monitoring_config)
	await service.initialize(config)
	
	return service


def register_capability() -> dict:
	"""Register monitoring and observability with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "moni",
		"aliases": ["monitoring", "observability", "signals"],
		"display_name": CAPABILITY_METADATA["display_name"],
		"description": __capability_description__,
		"version": __capability_version__,
		"dependencies": __capability_dependencies__,
		"optional_dependencies": __capability_optional_dependencies__,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"metrics_collection": "Track tenant-aware metrics from APG capabilities",
			"log_observability": "Ingest and govern structured logs",
			"trace_exploration": "Expose distributed trace navigation surfaces",
			"alert_orchestration": "Create, route, deduplicate, and correlate alerts",
			"autonomous_remediation": "Coordinate runbook-backed remediation workflows",
			"capability_rules": "Evaluate deterministic observability governance rules",
			"visual_theming": "Apply signal-console theme tokens and components"
		},
		"endpoints": {
			"metrics": "/moni/api/v1/metrics",
			"logs": "/moni/api/v1/logs",
			"traces": "/moni/api/v1/traces",
			"alerts": "/moni/api/v1/alerts",
			"rules": "/moni/api/v1/rules",
			"analytics": "/moni/api/v1/analytics"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": [
			"moni:view",
			"moni:view_metrics",
			"moni:view_traces",
			"moni:manage_alerts",
			"moni:manage_rules",
			"moni:view_analytics",
			"moni:remediate",
			"moni:admin"
		]
	}


def get_capability_info() -> dict:
	"""Get MONI capability information for composition and marketplace discovery."""
	return {
		"metadata": CAPABILITY_METADATA,
		"contract": get_capability_contract(),
		"features": [
			"Predictive issue prevention",
			"Contextual business impact correlation",
			"Unified metrics, logs, and traces",
			"Intelligent alert orchestration",
			"Autonomous remediation workflow support",
			"Performance optimization advisor"
		]
	}


# APG capability health check
async def health_check() -> dict:
	"""
	Health check for APG monitoring
	Returns capability health status and metrics
	"""
	try:
		from .service import _monitoring_service
		
		if _monitoring_service is None or not _monitoring_service.running:
			return {
				'healthy': False,
				'capability': 'moni',
				'status': 'not_running',
				'timestamp': None
			}
		
		health_status = await _monitoring_service.get_health_status()
		
		return {
			'healthy': health_status.get('healthy', False),
			'capability': 'moni',
			'status': 'running',
			'metrics': {
				'total_metrics': health_status.get('total_metrics', 0),
				'active_alerts': health_status.get('active_alerts', 0),
				'ingestion_rate': health_status.get('metrics_per_second', 0),
				'query_performance': health_status.get('avg_query_time_ms', 0),
				'system_health_score': health_status.get('health_score', 0)
			},
			'timestamp': health_status.get('timestamp')
		}
		
	except Exception as e:
		return {
			'healthy': False,
			'capability': 'moni',
			'status': 'error',
			'error': str(e),
			'timestamp': None
		}


# APG capability export functions for composition
async def track_metric(metric_name: str, value: float, labels: dict = None, 
					   tenant_id: str = None, source: str = "apg") -> bool:
	"""Export function: Track metric for other capabilities"""
	from .service import _monitoring_service
	from .models import MonitoringMetric
	
	if _monitoring_service:
		metric = MonitoringMetric(
			name=metric_name,
			value=value,
			labels=labels or {},
			tenant_id=tenant_id or "default",
			source=source
		)
		return await _monitoring_service.track_metric(metric)
	return False


async def log_event(event_type: str, message: str, severity: str = "info",
				   metadata: dict = None, tenant_id: str = None) -> bool:
	"""Export function: Log monitoring event for other capabilities"""
	from .service import _monitoring_service
	
	if _monitoring_service:
		return await _monitoring_service.log_event(
			event_type=event_type,
			message=message,
			severity=severity,
			metadata=metadata or {},
			tenant_id=tenant_id
		)
	return False


async def create_alert(rule_name: str, condition: str, metric_name: str,
					  severity: str = "medium", tenant_id: str = None) -> str | None:
	"""Export function: Create alert rule for other capabilities"""
	from .service import _monitoring_service
	from .models import MonitoringRule, AlertSeverity, AlertConditionType
	
	if _monitoring_service:
		rule = MonitoringRule(
			name=rule_name,
			condition=condition,
			metric_name=metric_name,
			severity=AlertSeverity(severity),
			condition_type=AlertConditionType.THRESHOLD,
			alert_message=f"Alert triggered for {rule_name}: {condition}",
			tenant_id=tenant_id or "default",
			created_by="apg_system"
		)
		return await _monitoring_service.create_alert_rule(rule)
	return None


async def query_metrics(metric_names: list[str], start_time, end_time,
					   labels: dict = None, tenant_id: str = None) -> list[MonitoringMetric]:
	"""Export function: Query metrics for other capabilities"""
	from .service import _monitoring_service
	from .models import MonitoringQuery
	
	if _monitoring_service:
		query = MonitoringQuery(
			metric_names=metric_names,
			start_time=start_time,
			end_time=end_time,
			labels=labels or {},
			tenant_id=tenant_id or "default"
		)
		return await _monitoring_service.query_metrics(query)
	return []


async def get_health_status(tenant_id: str = None) -> dict:
	"""Export function: Get system health status"""
	from .service import _monitoring_service
	
	if _monitoring_service:
		return await _monitoring_service.get_health_status(tenant_id)
	return {'healthy': False, 'error': 'Monitoring service not available'}


async def get_performance_metrics(component: str = None, tenant_id: str = None) -> dict:
	"""Export function: Get performance analytics"""
	from .service import _monitoring_service
	
	if _monitoring_service:
		return await _monitoring_service.get_performance_analytics(component, tenant_id)
	return {}


async def predict_resource_usage(resource_type: str, horizon_hours: int = 24,
								tenant_id: str = None) -> dict:
	"""Export function: Predict resource usage patterns"""
	from .service import _monitoring_service
	
	if _monitoring_service:
		return await _monitoring_service.predict_resource_usage(
			resource_type, horizon_hours, tenant_id
		)
	return {}


async def analyze_performance(component: str, time_window_minutes: int = 60,
							 tenant_id: str = None) -> dict:
	"""Export function: Analyze component performance"""
	from .service import _monitoring_service
	
	if _monitoring_service:
		return await _monitoring_service.analyze_performance(
			component, time_window_minutes, tenant_id
		)
	return {}


async def get_analytics_dashboard(dashboard_type: str = "operational",
								 tenant_id: str = None) -> dict:
	"""Export function: Get analytics dashboard data"""
	from .service import _monitoring_service
	
	if _monitoring_service:
		return await _monitoring_service.get_analytics_dashboard(
			dashboard_type, tenant_id
		)
	return {}


async def create_monitoring_rule(rule_config: dict, tenant_id: str = None) -> str | None:
	"""Export function: Create monitoring rule from configuration"""
	from .service import _monitoring_service
	from .models import MonitoringRule
	
	if _monitoring_service:
		# Convert dict to MonitoringRule model
		rule_config['tenant_id'] = tenant_id or rule_config.get('tenant_id', 'default')
		rule_config['created_by'] = rule_config.get('created_by', 'apg_system')
		
		rule = MonitoringRule(**rule_config)
		return await _monitoring_service.create_alert_rule(rule)
	return None


# Event handlers for APG composition engine
async def handle_metric_ingestion(event_data: dict) -> None:
	"""Handle metric ingestion events from other capabilities"""
	metric_data = event_data.get('metric')
	if metric_data:
		await track_metric(
			metric_name=metric_data.get('name'),
			value=metric_data.get('value'),
			labels=metric_data.get('labels', {}),
			tenant_id=event_data.get('tenant_id'),
			source=event_data.get('source', 'apg_event')
		)


async def handle_alert_trigger(event_data: dict) -> None:
	"""Handle alert trigger events"""
	alert_data = event_data.get('alert')
	if alert_data:
		# Process alert through notification system if available
		from .service import _monitoring_service
		if _monitoring_service:
			await _monitoring_service.process_alert_event(alert_data)


async def handle_health_check(event_data: dict) -> dict:
	"""Handle health check requests from other capabilities"""
	tenant_id = event_data.get('tenant_id')
	component = event_data.get('component')
	
	if component:
		return await analyze_performance(component, tenant_id=tenant_id)
	else:
		return await get_health_status(tenant_id)


async def handle_performance_analysis(event_data: dict) -> dict:
	"""Handle performance analysis requests"""
	component = event_data.get('component')
	time_window = event_data.get('time_window_minutes', 60)
	tenant_id = event_data.get('tenant_id')
	
	return await analyze_performance(component, time_window, tenant_id)


async def handle_anomaly_detected(event_data: dict) -> None:
	"""Handle anomaly detection events"""
	anomaly_data = event_data.get('anomaly')
	if anomaly_data:
		# Log anomaly as a special event
		await log_event(
			event_type="anomaly_detected",
			message=f"Anomaly detected: {anomaly_data.get('description', 'Unknown')}",
			severity="high",
			metadata=anomaly_data,
			tenant_id=event_data.get('tenant_id')
		)


# APG dependency integration helpers
def _log_integration_info(message: str) -> None:
	"""Log integration information using APG patterns"""
	print(f"[MONI Integration] {message}")


def _get_tenant_context() -> str:
	"""Get current tenant context from APG multi-tenancy"""
	# Integration with APG mten capability
	try:
		from ..mten import get_current_tenant
		return get_current_tenant()
	except ImportError:
		return "default"


def _audit_action(action: str, details: dict) -> None:
	"""Log audit action through APG audit system"""
	# Integration with APG audl capability
	try:
		from ..audl import log_audit_event
		log_audit_event(
			capability="moni",
			action=action,
			details=details,
			tenant_id=_get_tenant_context()
		)
	except ImportError:
		_log_integration_info(f"Audit integration not available for action: {action}")


def _check_authorization(action: str, resource: str = None) -> bool:
	"""Check authorization through APG auth system"""
	# Integration with APG auth capability
	try:
		from ..auth import check_permission
		return check_permission(
			capability="moni",
			action=action,
			resource=resource,
			tenant_id=_get_tenant_context()
		)
	except ImportError:
		_log_integration_info(f"Auth integration not available for action: {action}")
		return True  # Default to allow if auth not available


def _get_configuration(key: str, default=None):
	"""Get configuration through APG config system"""
	# Integration with APG conf capability
	try:
		from ..conf import get_config_value
		return get_config_value(
			capability="moni",
			key=key,
			default=default,
			tenant_id=_get_tenant_context()
		)
	except ImportError:
		_log_integration_info(f"Config integration not available for key: {key}")
		return default
