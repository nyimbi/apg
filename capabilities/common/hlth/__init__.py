#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Capability Registration
Health checks, diagnostics, incidents, and remediation governance

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from .service import (
	HlthAlertRecord,
	HlthAuditEventRecord,
	HlthBaselineRecord,
	HlthCheckRecord,
	HlthComponentRecord,
	HlthDeploymentGateRecord,
	HlthAgentRecord,
	HlthIncidentRecord,
	HlthLifecycleBatchRecord,
	HlthPredictionRecord,
	HlthRemediationRequestRecord,
	HlthService,
)
from .models import (
	HealthMetric, HealthAlert, HealthBaseline, HealthRule, 
	HealthAction, SystemComponent, HealthReport,
	HealthStatus, HealthSeverity, HealthDimension
)
from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)


# APG Capability Metadata for Composition Engine Registration
CAPABILITY_METADATA = {
	'name': 'hlth',
	'display_name': 'Health Checks and Diagnostics',
	'version': '1.0.0',
	'category': 'platform_infrastructure',
	'description': 'Tenant-scoped health checks, diagnostics, incidents, deployment gates, and remediation governance',
	'author': 'Nyimbi Odero',
	'company': 'Datacraft',
	'copyright': '© 2025 Datacraft',
	
	# APG Integration Configuration
	'load_order': 10,  # Load after monitoring (MONI) and other core capabilities
	'dependencies': [
		'moni',  # Monitoring and Observability - primary data source
		'mqeb',  # Message Queue - health event streaming
		'conf'   # Configuration - health policy management
	],
	'optional_dependencies': [
		'auth',  # Authentication and RBAC - dashboard security
		'audl',  # Audit Logging - complete health event auditing
		'ntfy',  # Notifications - health alert delivery
		'mten',  # Multi-tenancy - tenant-isolated health management
		'aicr',  # AI Orchestration - ML-powered health prediction
		'pred',  # Predictive Analytics - failure forecasting
		'colb',  # Collaboration - team incident response
		'cach'  # Caching - health data optimization
	],
	
	# Blueprint and API Configuration
	'blueprint_name': 'health_management',
	'url_prefix': '/health',
	'menu_category': 'Platform Health',
	'menu_icon': 'fa-heartbeat',
	'menu_priority': 100,
	
	# API Endpoints
	'api_endpoints': [
		'/hlth/api/v1/assessment',     # Real-time health scoring
		'/hlth/api/v1/alerts',         # Alert management
		'/hlth/api/v1/reports',        # Health analytics and reports
		'/hlth/api/v1/remediation',    # Automated remediation actions
		'/hlth/api/v1/config',         # Health policy management
		'/hlth/api/v1/predictions',    # ML-powered predictions
		'/hlth/api/v1/components',     # Component discovery and status
		'/hlth/api/v1/incidents',      # Incident tracking
		'/hlth/api/v1/baselines',      # Health baseline management
		'/hlth/api/v1/dashboard'       # Health dashboard data
	],
	
	# Health Check Configuration
	'health_check_endpoint': '/hlth/api/v1/status',
	'health_check_interval_seconds': 30,
	'health_check_timeout_seconds': 10,
	
	# Permissions
	'permissions': [
		'health.view',              # View health dashboards and reports
		'health.manage',            # Manage health rules and configuration
		'health.remediate',         # Execute remediation actions
		'health.admin',             # Full health management administration
		'health.incidents.view',    # View health incidents
		'health.incidents.manage',  # Manage health incidents
		'health.alerts.acknowledge', # Acknowledge health alerts
		'health.alerts.resolve',    # Resolve health alerts
		'health.reports.generate',  # Generate health reports
		'health.config.modify',     # Modify health configuration
		'health.deployments.review' # Review deployment gate decisions
	],
	
	# Composition features
	'features': [
		'component_registration',
		'governed_health_checks',
		'baseline_lifecycle',
		'prediction_review',
		'critical_alert_and_incident_governance',
		'remediation_review',
		'deployment_gate_decisions',
		'health_agent_composition',
		'lifecycle_batch_validation',
		'generated_application_view_models',
		'health_console_theming',
		'adapter_boundaries'
	],
	
	# Event Types for APG Event Bus
	'event_types': [
		'health.component.discovered',
		'health.component.status_changed',
		'health.alert.triggered',
		'health.alert.resolved',
		'health.remediation.started',
		'health.remediation.completed',
		'health.baseline.updated',
		'health.prediction.generated',
		'health.incident.created',
		'health.agent.registered',
		'health.lifecycle_batch.accepted',
		'health.report.generated'
	],
	
	# Configuration Schema
	'configuration_schema': {
		'health_check_interval_seconds': {
			'type': 'integer',
			'default': 60,
			'min': 10,
			'max': 3600,
			'description': 'Health check interval for components'
		},
		'prediction_window_hours': {
			'type': 'integer', 
			'default': 24,
			'min': 1,
			'max': 168,
			'description': 'Health prediction window in hours'
		},
		'auto_remediation_enabled': {
			'type': 'boolean',
			'default': True,
			'description': 'Enable automated remediation for common issues'
		},
		'alert_correlation_window_minutes': {
			'type': 'integer',
			'default': 5,
			'min': 1,
			'max': 60,
			'description': 'Time window for alert correlation'
		},
		'baseline_learning_period_days': {
			'type': 'integer',
			'default': 7,
			'min': 1,
			'max': 30,
			'description': 'Learning period for health baselines'
		}
	}
}


# Global health service instance (will be initialized by APG composition engine)
_health_service: Optional['SystemHealthService'] = None


async def initialize_capability(config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
	"""
	Initialize the APG System Health Management capability
	Called by APG composition engine during startup
	"""
	global _health_service
	
	try:
		# Import here to avoid circular dependencies
		from .service import SystemHealthService, HealthServiceConfig
		
		# Create service configuration
		service_config = HealthServiceConfig()
		if config:
			# Update configuration with provided values
			for key, value in config.items():
				if hasattr(service_config, key):
					setattr(service_config, key, value)
		
		# Initialize health service
		_health_service = SystemHealthService(service_config)
		await _health_service.initialize()
		
		# Register health event handlers with APG event bus
		await _register_health_event_handlers()
		
		return {
			'status': 'success',
			'message': 'APG System Health Management capability initialized successfully',
			'service': _health_service,
			'capabilities': CAPABILITY_METADATA['features'],
			'timestamp': datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		return {
			'status': 'error',
			'message': f'Failed to initialize health capability: {str(e)}',
			'timestamp': datetime.utcnow().isoformat()
		}


async def health_check() -> Dict[str, Any]:
	"""
	APG health check endpoint for the health capability itself
	Provides health status of the health management system
	"""
	global _health_service
	
	try:
		if not _health_service:
			return {
				'status': 'unhealthy',
				'message': 'Health service not initialized',
				'timestamp': datetime.utcnow().isoformat()
			}
		
		# Get service health status
		service_status = await _health_service.get_service_health()
		
		return {
			'status': 'healthy' if service_status.get('healthy', False) else 'degraded',
			'capability': 'hlth',
			'service_uptime_seconds': service_status.get('uptime_seconds', 0),
			'components_monitored': service_status.get('components_count', 0),
			'active_alerts': service_status.get('active_alerts', 0),
			'prediction_accuracy': service_status.get('prediction_accuracy', 0.0),
			'auto_remediation_success_rate': service_status.get('remediation_success_rate', 0.0),
			'timestamp': datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		return {
			'status': 'unhealthy',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}


async def track_component_health(component_id: str, health_metrics: Dict[str, Any], 
								tenant_id: str = 'default') -> Dict[str, Any]:
	"""
	Track health metrics for a system component
	APG export function for other capabilities to report health
	"""
	global _health_service
	
	try:
		if not _health_service:
			raise RuntimeError("Health service not initialized")
		
		# Create health metrics from provided data
		metrics = []
		for metric_name, value in health_metrics.items():
			if isinstance(value, (int, float)):
				metric = HealthMetric(
					tenant_id=tenant_id,
					component_id=component_id,
					name=metric_name,
					value=float(value),
					dimension=HealthDimension.PERFORMANCE  # Default dimension
				)
				metrics.append(metric)
		
		# Process health metrics
		results = []
		for metric in metrics:
			result = await _health_service.process_health_metric(metric)
			results.append(result)
		
		return {
			'status': 'success',
			'component_id': component_id,
			'metrics_processed': len(results),
			'health_alerts_triggered': sum(1 for r in results if r.get('alert_triggered', False)),
			'timestamp': datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		return {
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}


async def get_component_health_status(component_id: str, tenant_id: str = 'default') -> Dict[str, Any]:
	"""
	Get current health status for a specific component
	APG export function for health status queries
	"""
	global _health_service
	
	try:
		if not _health_service:
			raise RuntimeError("Health service not initialized")
		
		# Get component health status
		health_status = await _health_service.get_component_health_status(component_id, tenant_id)
		
		return {
			'status': 'success',
			'component_id': component_id,
			'tenant_id': tenant_id,
			'health_status': health_status.value if health_status else 'unknown',
			'health_score': await _health_service.calculate_component_health_score(component_id, tenant_id),
			'last_update': datetime.utcnow().isoformat(),
			'timestamp': datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		return {
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}


async def create_health_alert(alert_data: Dict[str, Any], tenant_id: str = 'default') -> Dict[str, Any]:
	"""
	Create a health alert programmatically
	APG export function for other capabilities to create health alerts
	"""
	global _health_service
	
	try:
		if not _health_service:
			raise RuntimeError("Health service not initialized")
		
		# Create health alert from provided data
		alert = HealthAlert(
			tenant_id=tenant_id,
			rule_id=alert_data.get('rule_id', 'manual'),
			component_id=alert_data.get('component_id', ''),
			name=alert_data.get('name', 'Manual Health Alert'),
			message=alert_data.get('message', 'Health alert created programmatically'),
			severity=HealthSeverity(alert_data.get('severity', 'medium')),
			health_status=HealthStatus(alert_data.get('health_status', 'warning')),
			source_metric=alert_data.get('source_metric', 'manual'),
			source_value=alert_data.get('source_value', 0.0),
			threshold_value=alert_data.get('threshold_value', 0.0),
			threshold_operator=alert_data.get('threshold_operator', 'gt')
		)
		
		# Process the alert
		alert_result = await _health_service.process_health_alert(alert)
		
		return {
			'status': 'success',
			'alert_id': alert.alert_id,
			'alert_processed': alert_result.get('processed', False),
			'remediation_triggered': alert_result.get('remediation_triggered', False),
			'timestamp': datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		return {
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}


async def generate_health_report(report_config: Dict[str, Any], tenant_id: str = 'default') -> Dict[str, Any]:
	"""
	Generate a health report programmatically
	APG export function for scheduled or on-demand health reporting
	"""
	global _health_service
	
	try:
		if not _health_service:
			raise RuntimeError("Health service not initialized")
		
		# Generate health report
		report = await _health_service.generate_health_report(
			tenant_id=tenant_id,
			report_type=report_config.get('type', 'executive'),
			component_ids=report_config.get('components', []),
			time_period_hours=report_config.get('time_period_hours', 24)
		)
		
		return {
			'status': 'success',
			'report_id': report.report_id,
			'overall_health_score': report.overall_health_score,
			'health_grade': report.get_health_grade(),
			'total_components': report.total_components,
			'critical_alerts': report.critical_alerts,
			'recommendations': report.recommendations,
			'timestamp': datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		return {
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}


async def predict_component_health(component_id: str, prediction_hours: int = 24, 
								  tenant_id: str = 'default') -> Dict[str, Any]:
	"""
	Predict health status for a component using ML models
	APG export function for predictive health analysis
	"""
	global _health_service
	
	try:
		if not _health_service:
			raise RuntimeError("Health service not initialized")
		
		# Generate health prediction
		prediction = await _health_service.predict_component_health(
			component_id=component_id,
			tenant_id=tenant_id,
			prediction_window_hours=prediction_hours
		)
		
		return {
			'status': 'success',
			'component_id': component_id,
			'tenant_id': tenant_id,
			'prediction_window_hours': prediction_hours,
			'predicted_health_score': prediction.get('health_score', 0.0),
			'predicted_status': prediction.get('status', 'unknown'),
			'confidence': prediction.get('confidence', 'low'),
			'risk_factors': prediction.get('risk_factors', []),
			'recommended_actions': prediction.get('recommended_actions', []),
			'timestamp': datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		return {
			'status': 'error',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}


async def _register_health_event_handlers() -> None:
	"""Register health event handlers with APG event bus"""
	try:
		# This would integrate with APG event bus when available
		# For now, we'll define the event handlers
		
		async def on_component_discovered(event_data: Dict[str, Any]) -> None:
			"""Handle component discovery events"""
			if _health_service:
				await _health_service.handle_component_discovery(event_data)
		
		async def on_metric_ingested(event_data: Dict[str, Any]) -> None:
			"""Handle metric ingestion events from MONI capability"""
			if _health_service:
				await _health_service.handle_metric_ingestion(event_data)
		
		async def on_alert_triggered(event_data: Dict[str, Any]) -> None:
			"""Handle alert trigger events"""
			if _health_service:
				await _health_service.handle_alert_triggered(event_data)
		
		async def on_configuration_changed(event_data: Dict[str, Any]) -> None:
			"""Handle configuration change events from CONF capability"""
			if _health_service:
				await _health_service.handle_configuration_change(event_data)
		
		# Register handlers (would be done through APG event bus)
		event_handlers = {
			'component.discovered': on_component_discovered,
			'metric.ingested': on_metric_ingested,
			'alert.triggered': on_alert_triggered,
			'configuration.changed': on_configuration_changed
		}
		
		# Store handlers for later registration with APG event bus
		if _health_service:
			_health_service._event_handlers = event_handlers
			
	except Exception as e:
		print(f"[HLTH] Error registering event handlers: {e}")


# APG Composition Engine Integration Functions
def get_capability_metadata() -> Dict[str, Any]:
	"""Get capability metadata for APG composition engine registration"""
	metadata = CAPABILITY_METADATA.copy()
	metadata['contract'] = get_capability_contract()
	return metadata


def register_capability() -> Dict[str, Any]:
	"""Register system health management with the APG composition engine."""
	contract = get_capability_contract()
	return {
		'name': 'hlth',
		'aliases': ['system_health', 'health_checks', 'diagnostics'],
		'display_name': CAPABILITY_METADATA['display_name'],
		'description': CAPABILITY_METADATA['description'],
		'version': CAPABILITY_METADATA['version'],
		'dependencies': CAPABILITY_METADATA['dependencies'],
		'optional_dependencies': CAPABILITY_METADATA['optional_dependencies'],
		'api_prefix': contract['ui']['api_prefix'],
		'configuration': contract['configuration'],
		'configuration_schema': contract['configuration_schema'],
		'rule_engine': contract['rule_engine'],
		'capabilities': {
			'component_lifecycle': 'Register and govern tenant system components',
			'health_assessment': 'Record component health checks with tenant context',
			'baseline_lifecycle': 'Maintain baseline evidence for generated workflows',
			'health_prediction': 'Record prediction evidence and review gates',
			'alert_lifecycle': 'Create route-backed health alerts',
			'incident_governance': 'Coordinate health incidents, reports, and remediation',
			'deployment_gates': 'Block or allow deployments based on unresolved critical incidents',
			'autonomous_remediation': 'Coordinate approved health remediation workflows',
			'capability_rules': 'Evaluate deterministic health governance rules',
			'visual_theming': 'Apply health-console theme tokens and components',
			'health_agent_composition': 'Register first-class health agents across Codex, Claude Code, opencode, Pi, and future runtime adapters',
			'lifecycle_batch_validation': 'Validate health lifecycle mutation batches against Bytewax-first stream rules',
			'review_evidence': 'Compose durable pending-review, denial, matched-rule, and reviewer evidence across health lifecycle records'
		},
		'endpoints': {
			'assessment': '/hlth/api/v1/assessment',
			'components': '/hlth/api/v1/components',
			'checks': '/hlth/api/v1/checks',
			'baselines': '/hlth/api/v1/baselines',
			'alerts': '/hlth/api/v1/alerts',
			'incidents': '/hlth/api/v1/incidents',
			'predictions': '/hlth/api/v1/predictions',
			'remediation': '/hlth/api/v1/remediation',
			'deployment_gates': '/hlth/api/v1/deployment-gates',
			'reports': '/hlth/api/v1/reports',
			'audit': '/hlth/api/v1/audit',
			'adapters': '/hlth/api/v1/adapters',
			'agents': '/hlth/api/v1/agents',
			'lifecycle': '/hlth/api/v1/lifecycle'
		},
		'ui_components': {
			route['name']: route['path']
			for route in contract['ui']['routes']
		},
		'ui_manifest': contract['ui'],
		'agents': contract['agents'],
		'streaming': contract['streaming'],
		'review_evidence': contract['review_evidence'],
		'theme': contract['theme'],
		'permissions': CAPABILITY_METADATA['permissions']
	}


def get_capability_info() -> Dict[str, Any]:
	"""Get HLTH capability information for composition and marketplace discovery."""
	return {
		'metadata': CAPABILITY_METADATA,
		'contract': get_capability_contract(),
		'features': [
			'Tenant-aware component registration',
			'Governed health checks and score decisions',
			'Baseline and prediction review workflows',
			'Critical alert and incident ownership',
			'Runbook-backed remediation review',
			'Deployment gate decisions',
			'First-class health-agent composition',
			'Bytewax-first health lifecycle batch validation',
			'Backend-neutral health adapter boundaries'
		]
	}


def get_export_functions() -> Dict[str, Any]:
	"""Get exported functions for other APG capabilities to use"""
	return {
		'initialize_capability': initialize_capability,
		'health_check': health_check,
		'track_component_health': track_component_health,
		'get_component_health_status': get_component_health_status,
		'create_health_alert': create_health_alert,
		'generate_health_report': generate_health_report,
		'predict_component_health': predict_component_health
	}


def get_api_routes() -> List[Dict[str, Any]]:
	"""Get API routes for APG API gateway registration"""
	return [
		{
			'path': '/health/assessment',
			'methods': ['GET', 'POST'],
			'handler': 'health_assessment_handler',
			'auth_required': True,
			'permissions': ['health.view']
		},
		{
			'path': '/health/alerts',
			'methods': ['GET', 'POST', 'PUT', 'DELETE'],
			'handler': 'health_alerts_handler',
			'auth_required': True,
			'permissions': ['health.manage']
		},
		{
			'path': '/health/reports',
			'methods': ['GET', 'POST'],
			'handler': 'health_reports_handler',
			'auth_required': True,
			'permissions': ['health.reports.generate']
		},
		{
			'path': '/health/remediation',
			'methods': ['GET', 'POST', 'PUT'],
			'handler': 'health_remediation_handler',
			'auth_required': True,
			'permissions': ['health.remediate']
		},
		{
			'path': '/health/predictions',
			'methods': ['GET', 'POST'],
			'handler': 'health_predictions_handler',
			'auth_required': True,
			'permissions': ['health.view']
		}
	]


def get_health_service() -> Optional['SystemHealthService']:
	"""Get the global health service instance"""
	return _health_service


# Export all public functions and classes
__all__ = [
	'CAPABILITY_METADATA',
	'register_capability',
	'get_capability_info',
	'initialize_capability',
	'health_check',
	'track_component_health',
	'get_component_health_status', 
	'create_health_alert',
	'generate_health_report',
	'predict_component_health',
	'get_capability_metadata',
	'get_export_functions',
	'get_api_routes',
	'get_health_service',
	'HlthService',
	'HlthComponentRecord',
	'HlthCheckRecord',
	'HlthBaselineRecord',
	'HlthPredictionRecord',
	'HlthAlertRecord',
	'HlthIncidentRecord',
	'HlthRemediationRequestRecord',
	'HlthDeploymentGateRecord',
	'HlthAgentRecord',
	'HlthLifecycleBatchRecord',
	'HlthAuditEventRecord',
	'get_capability_contract',
	'evaluate_capability_rules'
]
