#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - RESTful API Layer
Comprehensive health management API endpoints with APG integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from flask import request, jsonify, Blueprint, g, session
from flask_restful import Resource, Api, reqparse
from marshmallow import Schema, fields, validate, ValidationError

from .models import (
	HealthMetric, HealthAlert, HealthBaseline, HealthRule, 
	HealthAction, SystemComponent, HealthReport,
	HealthStatus, HealthSeverity, HealthDimension
)
from .service import SystemHealthService


# Create Flask Blueprint
health_api_bp = Blueprint('health_api', __name__, url_prefix='/api/v1/health')
api = Api(health_api_bp)


def _clean_text(value: Any) -> Optional[str]:
	"""Return a non-empty stripped string or None."""
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _object_value(source: Any, keys: List[str]) -> Optional[str]:
	"""Read the first present text value from a dict-like or object source."""
	if source is None:
		return None
	for key in keys:
		if isinstance(source, dict):
			value = source.get(key)
		else:
			value = getattr(source, key, None)
		text = _clean_text(value)
		if text:
			return text
	return None


def _first_text(*values: Any) -> str:
	"""Return the first non-empty text value."""
	for value in values:
		text = _clean_text(value)
		if text:
			return text
	return "system"


def resolve_current_user_id() -> str:
	"""Resolve current user from Flask/APG request context."""
	current_user = getattr(request, "current_user", None)
	g_user = getattr(g, "current_user", None) or getattr(g, "user", None) or getattr(g, "auth_user", None)
	return _first_text(
		_object_value(current_user, ["user_id", "id", "username", "email", "sub"]),
		getattr(request, "current_user_id", None),
		_object_value(g_user, ["user_id", "id", "username", "email", "sub"]),
		getattr(g, "user_id", None),
		session.get("user_id"),
		request.headers.get("X-APG-User-ID"),
		request.headers.get("X-User-ID"),
		request.args.get("user_id"),
		os.getenv("APG_USER_ID"),
		os.getenv("APG_DEFAULT_USER_ID"),
		"system",
	)


# Marshmallow Schemas for API Validation
class HealthMetricSchema(Schema):
	"""Schema for health metric API requests"""
	tenant_id = fields.Str(required=True)
	component_id = fields.Str(required=True)
	name = fields.Str(required=True)
	value = fields.Float(required=True)
	dimension = fields.Str(required=True, validate=validate.OneOf(['performance', 'security', 'compliance', 'availability', 'resource_utilization', 'reliability']))
	unit = fields.Str(missing='count')
	business_context = fields.Dict(missing=dict)
	tags = fields.List(fields.Str(), missing=list)
	metadata = fields.Dict(missing=dict)


class ComponentHealthAssessmentSchema(Schema):
	"""Schema for component health assessment requests"""
	tenant_id = fields.Str(required=True)
	component_id = fields.Str(required=True)
	time_window_hours = fields.Int(missing=24, validate=validate.Range(min=1, max=168))
	include_predictions = fields.Bool(missing=True)


class HealthReportGenerationSchema(Schema):
	"""Schema for health report generation requests"""
	tenant_id = fields.Str(required=True)
	report_type = fields.Str(missing='comprehensive', validate=validate.OneOf(['executive', 'operational', 'technical', 'comprehensive']))
	component_ids = fields.List(fields.Str(), missing=list)
	time_period_hours = fields.Int(missing=24, validate=validate.Range(min=1, max=168))
	include_predictions = fields.Bool(missing=True)
	include_recommendations = fields.Bool(missing=True)


# API Resource Classes
class HealthMetricsResource(Resource):
	"""RESTful API resource for health metrics"""
	
	def __init__(self):
		self.health_service = self._get_health_service()
		self.metric_schema = HealthMetricSchema()
	
	def get(self):
		"""Get health metrics with filtering and pagination"""
		try:
			# Parse query parameters
			parser = reqparse.RequestParser()
			parser.add_argument('tenant_id', type=str, required=True)
			parser.add_argument('component_id', type=str, required=False)
			parser.add_argument('dimension', type=str, required=False)
			parser.add_argument('time_window_hours', type=int, default=24)
			parser.add_argument('limit', type=int, default=100)
			parser.add_argument('offset', type=int, default=0)
			args = parser.parse_args()
			
			# Execute async query
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			metrics_result = loop.run_until_complete(
				self.health_service.get_health_metrics(
					tenant_id=args['tenant_id'],
					component_id=args.get('component_id'),
					dimension=args.get('dimension'),
					time_window_hours=args['time_window_hours'],
					limit=args['limit'],
					offset=args['offset']
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': {
					'metrics': [metric.to_dict() for metric in metrics_result['metrics']],
					'total_count': metrics_result['total_count'],
					'limit': args['limit'],
					'offset': args['offset']
				},
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except ValidationError as e:
			return {'status': 'error', 'errors': e.messages}, 400
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500
	
	def post(self):
		"""Submit new health metric"""
		try:
			# Validate request data
			metric_data = self.metric_schema.load(request.get_json())
			
			# Create health metric
			health_metric = HealthMetric(**metric_data)
			
			# Process metric
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			processing_result = loop.run_until_complete(
				self.health_service.process_health_metric(health_metric)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': {
					'metric_id': health_metric.metric_id,
					'processing_result': processing_result
				},
				'message': 'Health metric processed successfully',
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except ValidationError as e:
			return {'status': 'error', 'errors': e.messages}, 400
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class ComponentHealthResource(Resource):
	"""RESTful API resource for component health assessment"""
	
	def __init__(self):
		self.health_service = self._get_health_service()
		self.assessment_schema = ComponentHealthAssessmentSchema()
	
	def get(self, component_id=None):
		"""Get component health status and assessment"""
		try:
			# Parse query parameters
			parser = reqparse.RequestParser()
			parser.add_argument('tenant_id', type=str, required=True)
			parser.add_argument('time_window_hours', type=int, default=24)
			parser.add_argument('include_predictions', type=bool, default=True)
			args = parser.parse_args()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			if component_id:
				# Get specific component health
				health_assessment = loop.run_until_complete(
					self.health_service.assess_component_health(
						component_id=component_id,
						tenant_id=args['tenant_id'],
						time_window_hours=args['time_window_hours']
					)
				)
				
				result_data = {
					'component_id': component_id,
					'health_assessment': health_assessment
				}
				
				if args['include_predictions']:
					prediction = loop.run_until_complete(
						self.health_service.predict_component_health(
							component_id=component_id,
							tenant_id=args['tenant_id']
						)
					)
					result_data['prediction'] = prediction
			
			else:
				# Get all components health summary
				health_summary = loop.run_until_complete(
					self.health_service.get_tenant_health_summary(args['tenant_id'])
				)
				
				result_data = {
					'tenant_health_summary': health_summary
				}
			
			loop.close()
			
			return {
				'status': 'success',
				'data': result_data,
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500
	
	def post(self):
		"""Trigger component health assessment"""
		try:
			# Validate request data
			assessment_data = self.assessment_schema.load(request.get_json())
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			assessment_result = loop.run_until_complete(
				self.health_service.assess_component_health(
					component_id=assessment_data['component_id'],
					tenant_id=assessment_data['tenant_id'],
					time_window_hours=assessment_data['time_window_hours']
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': {
					'component_id': assessment_data['component_id'],
					'assessment_result': assessment_result
				},
				'message': 'Health assessment completed successfully',
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except ValidationError as e:
			return {'status': 'error', 'errors': e.messages}, 400
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class HealthAlertsResource(Resource):
	"""RESTful API resource for health alerts management"""
	
	def __init__(self):
		self.health_service = self._get_health_service()
	
	def get(self):
		"""Get health alerts with filtering"""
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('tenant_id', type=str, required=True)
			parser.add_argument('component_id', type=str, required=False)
			parser.add_argument('severity', type=str, required=False)
			parser.add_argument('status', type=str, required=False)
			parser.add_argument('limit', type=int, default=100)
			parser.add_argument('offset', type=int, default=0)
			args = parser.parse_args()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			alerts_result = loop.run_until_complete(
				self.health_service.get_health_alerts(
					tenant_id=args['tenant_id'],
					component_id=args.get('component_id'),
					severity=args.get('severity'),
					status=args.get('status'),
					limit=args['limit'],
					offset=args['offset']
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': {
					'alerts': [alert.to_dict() for alert in alerts_result['alerts']],
					'total_count': alerts_result['total_count'],
					'active_count': alerts_result['active_count'],
					'critical_count': alerts_result['critical_count']
				},
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500
	
	def patch(self, alert_id):
		"""Update alert status (acknowledge, resolve)"""
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('action', type=str, required=True, choices=['acknowledge', 'resolve', 'close'])
			parser.add_argument('notes', type=str, required=False)
			args = parser.parse_args()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			update_result = loop.run_until_complete(
				self.health_service.update_alert_status(
					alert_id=alert_id,
					action=args['action'],
					notes=args.get('notes', ''),
					updated_by=self._get_current_user()
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': update_result,
				'message': f'Alert {args["action"]}d successfully',
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500

	def _get_current_user(self) -> str:
		"""Get current user from request context"""
		return resolve_current_user_id()

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class HealthReportsResource(Resource):
	"""RESTful API resource for health reports generation"""
	
	def __init__(self):
		self.health_service = self._get_health_service()
		self.report_schema = HealthReportGenerationSchema()
	
	def get(self):
		"""Get existing health reports"""
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('tenant_id', type=str, required=True)
			parser.add_argument('report_type', type=str, required=False)
			parser.add_argument('limit', type=int, default=50)
			args = parser.parse_args()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			reports_result = loop.run_until_complete(
				self.health_service.get_health_reports(
					tenant_id=args['tenant_id'],
					report_type=args.get('report_type'),
					limit=args['limit']
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': {
					'reports': [report.to_dict() for report in reports_result['reports']]
				},
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500
	
	def post(self):
		"""Generate new health report"""
		try:
			# Validate request data
			report_data = self.report_schema.load(request.get_json())
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			health_report = loop.run_until_complete(
				self.health_service.generate_health_report(
					tenant_id=report_data['tenant_id'],
					report_type=report_data['report_type'],
					component_ids=report_data['component_ids'],
					time_period_hours=report_data['time_period_hours']
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': {
					'report_id': health_report.report_id,
					'report': health_report.to_dict()
				},
				'message': 'Health report generated successfully',
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except ValidationError as e:
			return {'status': 'error', 'errors': e.messages}, 400
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class MultiDimensionalAnalysisResource(Resource):
	"""RESTful API resource for multi-dimensional health analysis"""
	
	def __init__(self):
		self.health_service = self._get_health_service()
	
	def get(self):
		"""Get multi-dimensional health analysis"""
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('tenant_id', type=str, required=True)
			parser.add_argument('component_id', type=str, required=False)
			parser.add_argument('time_window_hours', type=int, default=24)
			args = parser.parse_args()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			analysis_result = loop.run_until_complete(
				self.health_service.analyze_multi_dimensional_health(
					tenant_id=args['tenant_id'],
					component_id=args.get('component_id'),
					time_window_hours=args['time_window_hours']
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': analysis_result,
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class HealthPredictionsResource(Resource):
	"""RESTful API resource for health predictions"""
	
	def __init__(self):
		self.health_service = self._get_health_service()
	
	def get(self):
		"""Get health predictions"""
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('tenant_id', type=str, required=True)
			parser.add_argument('component_id', type=str, required=False)
			parser.add_argument('prediction_window_hours', type=int, default=48)
			args = parser.parse_args()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			if args.get('component_id'):
				# Single component prediction
				prediction_result = loop.run_until_complete(
					self.health_service.predict_component_health(
						component_id=args['component_id'],
						tenant_id=args['tenant_id'],
						prediction_window_hours=args['prediction_window_hours']
					)
				)
				
				data = {
					'component_id': args['component_id'],
					'prediction': prediction_result
				}
			else:
				# Tenant-wide predictions
				predictions_result = loop.run_until_complete(
					self.health_service.get_tenant_health_predictions(
						tenant_id=args['tenant_id'],
						prediction_window_hours=args['prediction_window_hours']
					)
				)
				
				data = {
					'tenant_predictions': predictions_result
				}
			
			loop.close()
			
			return {
				'status': 'success',
				'data': data,
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


class RemediationResource(Resource):
	"""RESTful API resource for autonomous remediation"""
	
	def __init__(self):
		self.health_service = self._get_health_service()
	
	def post(self):
		"""Trigger remediation action"""
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('tenant_id', type=str, required=True)
			parser.add_argument('alert_id', type=str, required=True)
			parser.add_argument('remediation_type', type=str, required=False)
			args = parser.parse_args()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			remediation_result = loop.run_until_complete(
				self.health_service.trigger_manual_remediation(
					alert_id=args['alert_id'],
					tenant_id=args['tenant_id'],
					remediation_type=args.get('remediation_type'),
					triggered_by=self._get_current_user()
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': remediation_result,
				'message': 'Remediation triggered successfully',
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500
	
	def get(self):
		"""Get remediation status and history"""
		try:
			parser = reqparse.RequestParser()
			parser.add_argument('tenant_id', type=str, required=True)
			parser.add_argument('component_id', type=str, required=False)
			parser.add_argument('limit', type=int, default=50)
			args = parser.parse_args()
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			remediation_history = loop.run_until_complete(
				self.health_service.get_remediation_history(
					tenant_id=args['tenant_id'],
					component_id=args.get('component_id'),
					limit=args['limit']
				)
			)
			
			loop.close()
			
			return {
				'status': 'success',
				'data': {
					'remediation_history': remediation_history
				},
				'timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			return {'status': 'error', 'message': str(e)}, 500

	def _get_current_user(self) -> str:
		"""Get current user from request context"""
		return resolve_current_user_id()

	def _get_health_service(self) -> SystemHealthService:
		"""Get health service instance"""
		from . import get_health_service
		return get_health_service()


# Register API Resources
api.add_resource(HealthMetricsResource, '/metrics', '/metrics/')
api.add_resource(ComponentHealthResource, '/components', '/components/', '/components/<string:component_id>')
api.add_resource(HealthAlertsResource, '/alerts', '/alerts/', '/alerts/<string:alert_id>')
api.add_resource(HealthReportsResource, '/reports', '/reports/')
api.add_resource(MultiDimensionalAnalysisResource, '/analysis/multi-dimensional', '/analysis/multi-dimensional/')
api.add_resource(HealthPredictionsResource, '/predictions', '/predictions/')
api.add_resource(RemediationResource, '/remediation', '/remediation/')


# Health Status Endpoint
@health_api_bp.route('/status', methods=['GET'])
def health_status():
	"""Health status endpoint for the health management API itself"""
	try:
		from . import get_health_service, health_check
		
		# Run health check
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		health_status_result = loop.run_until_complete(health_check())
		
		loop.close()
		
		return jsonify({
			'status': 'healthy' if health_status_result.get('status') == 'healthy' else 'degraded',
			'capability': 'hlth',
			'api_version': '1.0.0',
			'service_status': health_status_result,
			'timestamp': datetime.utcnow().isoformat()
		})
	
	except Exception as e:
		return jsonify({
			'status': 'unhealthy',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 500


# Export blueprint for Flask app registration
__all__ = ['health_api_bp']
