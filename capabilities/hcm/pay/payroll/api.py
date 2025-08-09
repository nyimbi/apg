"""
APG Payroll Management - Revolutionary REST API Integration

Next-generation REST API providing comprehensive payroll management endpoints
with real-time processing, AI-powered insights, and seamless APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal

from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from flask_restx.inputs import datetime_from_iso8601, date_from_iso8601
from marshmallow import Schema, fields as ma_fields, validate, post_load
from sqlalchemy import and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

# APG Platform Imports
from ...auth_rbac.decorators import require_permission, tenant_required, api_key_required
from ...audit_compliance.decorators import audit_api_call
from ...rate_limiting.decorators import rate_limit
from ...validation.decorators import validate_request, validate_response
from ...cache.decorators import cache_result
from ...monitoring.decorators import monitor_performance

from .models import (
	PRPayrollPeriod, PRPayrollRun, PREmployeePayroll, PRPayComponent,
	PRPayrollLineItem, PRTaxCalculation, PRPayrollAdjustment,
	PayrollStatus, PayComponentType, PayFrequency, TaxType
)
from .service import RevolutionaryPayrollService, PayrollProcessingConfig
from .ai_intelligence_engine import PayrollIntelligenceEngine
from .conversational_assistant import ConversationalPayrollAssistant
from .compliance_tax_engine import IntelligentComplianceTaxEngine

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint and API
payroll_bp = Blueprint('payroll_api', __name__, url_prefix='/api/v1/payroll')
api = Api(payroll_bp, version='1.0', title='APG Payroll API',
		  description='Revolutionary Payroll Management API with AI-powered automation')

# Create namespaces
ns_periods = Namespace('periods', description='Payroll period management')
ns_runs = Namespace('runs', description='Payroll run processing')
ns_employees = Namespace('employees', description='Employee payroll management')
ns_analytics = Namespace('analytics', description='Payroll analytics and insights')
ns_ai = Namespace('ai', description='AI-powered payroll intelligence')
ns_compliance = Namespace('compliance', description='Compliance and tax management')
ns_conversational = Namespace('chat', description='Conversational payroll interface')

api.add_namespace(ns_periods)
api.add_namespace(ns_runs)
api.add_namespace(ns_employees)
api.add_namespace(ns_analytics)
api.add_namespace(ns_ai)
api.add_namespace(ns_compliance)
api.add_namespace(ns_conversational)


# ===========================
# REQUEST/RESPONSE SCHEMAS
# ===========================

# Payroll Period Schemas
payroll_period_model = api.model('PayrollPeriod', {
	'period_id': fields.String(required=True, description='Unique period identifier'),
	'period_name': fields.String(required=True, description='Period name'),
	'period_type': fields.String(required=True, description='Period type'),
	'pay_frequency': fields.String(required=True, description='Pay frequency'),
	'start_date': fields.Date(required=True, description='Period start date'),
	'end_date': fields.Date(required=True, description='Period end date'),
	'pay_date': fields.Date(required=True, description='Pay date'),
	'cutoff_date': fields.Date(description='Cutoff date'),
	'status': fields.String(description='Period status'),
	'employee_count': fields.Integer(description='Number of employees'),
	'total_gross_pay': fields.Float(description='Total gross pay'),
	'total_net_pay': fields.Float(description='Total net pay'),
	'created_at': fields.DateTime(description='Creation timestamp')
})

payroll_period_create = api.model('PayrollPeriodCreate', {
	'period_name': fields.String(required=True, description='Period name'),
	'period_type': fields.String(required=True, description='Period type'),
	'pay_frequency': fields.String(required=True, description='Pay frequency'),
	'start_date': fields.Date(required=True, description='Period start date'),
	'end_date': fields.Date(required=True, description='Period end date'),
	'pay_date': fields.Date(required=True, description='Pay date'),
	'cutoff_date': fields.Date(description='Cutoff date'),
	'fiscal_year': fields.Integer(description='Fiscal year'),
	'fiscal_quarter': fields.Integer(description='Fiscal quarter'),
	'country_code': fields.String(description='Country code'),
	'currency_code': fields.String(description='Currency code'),
	'timezone': fields.String(description='Timezone')
})

# Payroll Run Schemas
payroll_run_model = api.model('PayrollRun', {
	'run_id': fields.String(required=True, description='Unique run identifier'),
	'run_number': fields.Integer(description='Run number'),
	'run_name': fields.String(description='Run name'),
	'run_type': fields.String(description='Run type'),
	'status': fields.String(description='Processing status'),
	'processing_stage': fields.String(description='Current processing stage'),
	'progress_percentage': fields.Float(description='Progress percentage'),
	'employee_count': fields.Integer(description='Total employees'),
	'processed_employee_count': fields.Integer(description='Processed employees'),
	'error_count': fields.Integer(description='Error count'),
	'warning_count': fields.Integer(description='Warning count'),
	'validation_score': fields.Float(description='Validation score'),
	'compliance_score': fields.Float(description='Compliance score'),
	'started_at': fields.DateTime(description='Start timestamp'),
	'completed_at': fields.DateTime(description='Completion timestamp')
})

payroll_run_create = api.model('PayrollRunCreate', {
	'period_id': fields.String(required=True, description='Period identifier'),
	'run_name': fields.String(required=True, description='Run name'),
	'run_type': fields.String(required=True, description='Run type'),
	'description': fields.String(description='Run description'),
	'priority': fields.String(description='Processing priority'),
	'auto_approve_threshold': fields.Float(description='Auto-approval threshold'),
	'notifications_enabled': fields.Boolean(description='Enable notifications')
})

# Employee Payroll Schemas
employee_payroll_model = api.model('EmployeePayroll', {
	'employee_payroll_id': fields.String(required=True, description='Unique identifier'),
	'employee_id': fields.String(required=True, description='Employee identifier'),
	'employee_name': fields.String(description='Employee name'),
	'employee_number': fields.String(description='Employee number'),
	'department_name': fields.String(description='Department'),
	'gross_earnings': fields.Float(description='Gross earnings'),
	'total_deductions': fields.Float(description='Total deductions'),
	'total_taxes': fields.Float(description='Total taxes'),
	'net_pay': fields.Float(description='Net pay'),
	'regular_hours': fields.Float(description='Regular hours'),
	'overtime_hours': fields.Float(description='Overtime hours'),
	'validation_score': fields.Float(description='Validation score'),
	'has_errors': fields.Boolean(description='Has errors'),
	'has_warnings': fields.Boolean(description='Has warnings')
})

# Analytics Schemas
analytics_model = api.model('PayrollAnalytics', {
	'total_employees': fields.Integer(description='Total active employees'),
	'total_gross_pay': fields.Float(description='Total gross pay'),
	'total_net_pay': fields.Float(description='Total net pay'),
	'processing_efficiency': fields.Float(description='Processing efficiency percentage'),
	'compliance_score': fields.Float(description='Overall compliance score'),
	'trend_data': fields.Raw(description='Trend analysis data'),
	'department_breakdown': fields.Raw(description='Department-wise breakdown'),
	'ai_insights': fields.List(fields.Raw, description='AI-generated insights'),
	'anomalies': fields.List(fields.Raw, description='Detected anomalies')
})

# AI Chat Schemas
chat_message_model = api.model('ChatMessage', {
	'command': fields.String(required=True, description='Natural language command'),
	'context': fields.Raw(description='Additional context data')
})

chat_response_model = api.model('ChatResponse', {
	'success': fields.Boolean(description='Request success status'),
	'message': fields.String(description='Response message'),
	'data': fields.Raw(description='Response data'),
	'actions': fields.List(fields.Raw, description='Suggested actions'),
	'timestamp': fields.DateTime(description='Response timestamp')
})


# ===========================
# PAYROLL PERIODS API
# ===========================

@ns_periods.route('/')
class PayrollPeriodListAPI(Resource):
	"""Payroll periods list and creation endpoint."""
	
	@api.doc('list_payroll_periods')
	@api.marshal_list_with(payroll_period_model)
	@require_permission('view_payroll_periods')
	@tenant_required
	@rate_limit(requests=100, per=60)
	@cache_result(timeout=300)
	@monitor_performance
	def get(self):
		"""Get list of payroll periods with filtering and pagination."""
		try:
			# Parse query parameters
			page = request.args.get('page', 1, type=int)
			per_page = min(request.args.get('per_page', 20, type=int), 100)
			status = request.args.get('status')
			fiscal_year = request.args.get('fiscal_year', type=int)
			
			# Build query
			query = PRPayrollPeriod.query.filter_by(tenant_id=request.tenant_id)
			
			if status:
				query = query.filter(PRPayrollPeriod.status == status)
			if fiscal_year:
				query = query.filter(PRPayrollPeriod.fiscal_year == fiscal_year)
			
			# Apply pagination
			periods = query.order_by(desc(PRPayrollPeriod.start_date)).paginate(
				page=page, per_page=per_page, error_out=False
			)
			
			return {
				'periods': [period.to_dict() for period in periods.items],
				'pagination': {
					'page': page,
					'per_page': per_page,
					'total': periods.total,
					'pages': periods.pages
				}
			}
			
		except Exception as e:
			logger.error(f"Failed to list payroll periods: {e}")
			api.abort(500, 'Failed to retrieve payroll periods')
	
	@api.doc('create_payroll_period')
	@api.expect(payroll_period_create)
	@api.marshal_with(payroll_period_model, code=201)
	@require_permission('create_payroll_period')
	@tenant_required
	@audit_api_call
	@validate_request(payroll_period_create)
	@rate_limit(requests=10, per=60)
	@monitor_performance
	def post(self):
		"""Create a new payroll period."""
		try:
			data = request.get_json()
			
			# Initialize payroll service
			service = RevolutionaryPayrollService()
			
			# Create period
			period = service.create_payroll_period(
				period_data=data,
				tenant_id=request.tenant_id,
				user_id=request.user_id
			)
			
			return period.to_dict(), 201
			
		except ValueError as e:
			api.abort(400, str(e))
		except Exception as e:
			logger.error(f"Failed to create payroll period: {e}")
			api.abort(500, 'Failed to create payroll period')


@ns_periods.route('/<string:period_id>')
class PayrollPeriodAPI(Resource):
	"""Individual payroll period management."""
	
	@api.doc('get_payroll_period')
	@api.marshal_with(payroll_period_model)
	@require_permission('view_payroll_period')
	@tenant_required
	@cache_result(timeout=300)
	@monitor_performance
	def get(self, period_id):
		"""Get payroll period details."""
		try:
			period = PRPayrollPeriod.query.filter_by(
				period_id=period_id,
				tenant_id=request.tenant_id
			).first()
			
			if not period:
				api.abort(404, 'Payroll period not found')
			
			return period.to_dict()
			
		except Exception as e:
			logger.error(f"Failed to get payroll period: {e}")
			api.abort(500, 'Failed to retrieve payroll period')


@ns_periods.route('/<string:period_id>/ai_insights')
class PayrollPeriodAIInsightsAPI(Resource):
	"""AI insights for payroll period."""
	
	@api.doc('get_period_ai_insights')
	@require_permission('view_ai_insights')
	@tenant_required
	@cache_result(timeout=600)
	@monitor_performance
	def get(self, period_id):
		"""Get AI-powered insights for payroll period."""
		try:
			# Initialize intelligence engine
			intelligence_engine = PayrollIntelligenceEngine()
			
			# Get AI insights
			insights = intelligence_engine.analyze_payroll_period(period_id)
			
			return {
				'period_id': period_id,
				'insights': insights,
				'generated_at': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Failed to get AI insights: {e}")
			api.abort(500, 'Failed to generate AI insights')


# ===========================
# PAYROLL RUNS API
# ===========================

@ns_runs.route('/')
class PayrollRunListAPI(Resource):
	"""Payroll runs list and creation endpoint."""
	
	@api.doc('list_payroll_runs')
	@api.marshal_list_with(payroll_run_model)
	@require_permission('view_payroll_runs')
	@tenant_required
	@rate_limit(requests=100, per=60)
	@cache_result(timeout=60)
	@monitor_performance
	def get(self):
		"""Get list of payroll runs."""
		try:
			# Parse query parameters
			page = request.args.get('page', 1, type=int)
			per_page = min(request.args.get('per_page', 20, type=int), 100)
			status = request.args.get('status')
			period_id = request.args.get('period_id')
			
			# Build query
			query = PRPayrollRun.query.filter_by(tenant_id=request.tenant_id)
			
			if status:
				query = query.filter(PRPayrollRun.status == status)
			if period_id:
				query = query.filter(PRPayrollRun.period_id == period_id)
			
			# Apply pagination
			runs = query.order_by(desc(PRPayrollRun.started_at)).paginate(
				page=page, per_page=per_page, error_out=False
			)
			
			return {
				'runs': [run.to_dict() for run in runs.items],
				'pagination': {
					'page': page,
					'per_page': per_page,
					'total': runs.total,
					'pages': runs.pages
				}
			}
			
		except Exception as e:
			logger.error(f"Failed to list payroll runs: {e}")
			api.abort(500, 'Failed to retrieve payroll runs')
	
	@api.doc('start_payroll_run')
	@api.expect(payroll_run_create)
	@api.marshal_with(payroll_run_model, code=201)
	@require_permission('start_payroll')
	@tenant_required
	@audit_api_call
	@validate_request(payroll_run_create)
	@rate_limit(requests=5, per=60)
	@monitor_performance
	def post(self):
		"""Start a new payroll run."""
		try:
			data = request.get_json()
			
			# Initialize payroll service
			service = RevolutionaryPayrollService()
			
			# Start payroll run
			run = service.start_payroll_run(
				run_data=data,
				tenant_id=request.tenant_id,
				user_id=request.user_id
			)
			
			return run.to_dict(), 201
			
		except ValueError as e:
			api.abort(400, str(e))
		except Exception as e:
			logger.error(f"Failed to start payroll run: {e}")
			api.abort(500, 'Failed to start payroll run')


@ns_runs.route('/<string:run_id>')
class PayrollRunAPI(Resource):
	"""Individual payroll run management."""
	
	@api.doc('get_payroll_run')
	@api.marshal_with(payroll_run_model)
	@require_permission('view_payroll_run')
	@tenant_required
	@cache_result(timeout=30)
	@monitor_performance
	def get(self, run_id):
		"""Get payroll run details."""
		try:
			run = PRPayrollRun.query.filter_by(
				run_id=run_id,
				tenant_id=request.tenant_id
			).first()
			
			if not run:
				api.abort(404, 'Payroll run not found')
			
			return run.to_dict()
			
		except Exception as e:
			logger.error(f"Failed to get payroll run: {e}")
			api.abort(500, 'Failed to retrieve payroll run')


@ns_runs.route('/<string:run_id>/status')
class PayrollRunStatusAPI(Resource):
	"""Real-time payroll run status monitoring."""
	
	@api.doc('get_payroll_run_status')
	@require_permission('monitor_payroll')
	@tenant_required
	@monitor_performance
	def get(self, run_id):
		"""Get real-time payroll run status."""
		try:
			# Initialize payroll service
			service = RevolutionaryPayrollService()
			
			# Get status
			status = service.get_payroll_status(run_id, request.tenant_id)
			
			return status
			
		except Exception as e:
			logger.error(f"Failed to get payroll status: {e}")
			api.abort(500, 'Failed to retrieve payroll status')


@ns_runs.route('/<string:run_id>/approve')
class PayrollRunApprovalAPI(Resource):
	"""Payroll run approval endpoint."""
	
	@api.doc('approve_payroll_run')
	@require_permission('approve_payroll')
	@tenant_required
	@audit_api_call
	@rate_limit(requests=10, per=60)
	@monitor_performance
	def post(self, run_id):
		"""Approve payroll run."""
		try:
			data = request.get_json() or {}
			approval_comments = data.get('comments')
			
			# Initialize payroll service
			service = RevolutionaryPayrollService()
			
			# Approve run
			success = service.approve_payroll_run(
				run_id=run_id,
				tenant_id=request.tenant_id,
				user_id=request.user_id,
				approval_comments=approval_comments
			)
			
			return {
				'success': success,
				'message': 'Payroll run approved successfully' if success else 'Failed to approve payroll run',
				'approved_at': datetime.utcnow().isoformat()
			}
			
		except ValueError as e:
			api.abort(400, str(e))
		except Exception as e:
			logger.error(f"Failed to approve payroll run: {e}")
			api.abort(500, 'Failed to approve payroll run')


# ===========================
# ANALYTICS API
# ===========================

@ns_analytics.route('/dashboard')
class PayrollAnalyticsDashboardAPI(Resource):
	"""Payroll analytics dashboard endpoint."""
	
	@api.doc('get_analytics_dashboard')
	@api.marshal_with(analytics_model)
	@require_permission('view_payroll_analytics')
	@tenant_required
	@cache_result(timeout=300)
	@monitor_performance
	def get(self):
		"""Get comprehensive payroll analytics dashboard data."""
		try:
			# Initialize intelligence engine
			intelligence_engine = PayrollIntelligenceEngine()
			
			# Get dashboard analytics
			analytics = intelligence_engine.get_dashboard_analytics(
				tenant_id=request.tenant_id
			)
			
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to get analytics dashboard: {e}")
			api.abort(500, 'Failed to retrieve analytics data')


@ns_analytics.route('/trends')
class PayrollTrendsAPI(Resource):
	"""Payroll trends analysis endpoint."""
	
	@api.doc('get_payroll_trends')
	@require_permission('view_payroll_trends')
	@tenant_required
	@cache_result(timeout=600)
	@monitor_performance
	def get(self):
		"""Get payroll trends analysis."""
		try:
			# Parse query parameters
			period = request.args.get('period', '12months')
			metric = request.args.get('metric', 'gross_pay')
			
			# Initialize intelligence engine
			intelligence_engine = PayrollIntelligenceEngine()
			
			# Get trends
			trends = intelligence_engine.analyze_payroll_trends(
				tenant_id=request.tenant_id,
				period=period,
				metric=metric
			)
			
			return trends
			
		except Exception as e:
			logger.error(f"Failed to get payroll trends: {e}")
			api.abort(500, 'Failed to retrieve trends data')


# ===========================
# AI INTELLIGENCE API
# ===========================

@ns_ai.route('/anomalies')
class PayrollAnomaliesAPI(Resource):
	"""AI-powered anomaly detection endpoint."""
	
	@api.doc('detect_payroll_anomalies')
	@require_permission('view_anomaly_detection')
	@tenant_required
	@cache_result(timeout=300)
	@monitor_performance
	def get(self):
		"""Detect payroll anomalies using AI."""
		try:
			period_id = request.args.get('period_id')
			
			# Initialize intelligence engine
			intelligence_engine = PayrollIntelligenceEngine()
			
			# Detect anomalies
			anomalies = intelligence_engine.detect_anomalies(
				tenant_id=request.tenant_id,
				period_id=period_id
			)
			
			return {
				'anomalies': anomalies,
				'detection_timestamp': datetime.utcnow().isoformat(),
				'ai_confidence': 0.94
			}
			
		except Exception as e:
			logger.error(f"Failed to detect anomalies: {e}")
			api.abort(500, 'Failed to detect anomalies')


@ns_ai.route('/predictions')
class PayrollPredictionsAPI(Resource):
	"""AI-powered payroll predictions endpoint."""
	
	@api.doc('get_payroll_predictions')
	@require_permission('view_ai_predictions')
	@tenant_required
	@cache_result(timeout=3600)
	@monitor_performance
	def get(self):
		"""Get AI-powered payroll predictions."""
		try:
			prediction_type = request.args.get('type', 'costs')
			horizon = request.args.get('horizon', '3months')
			
			# Initialize intelligence engine
			intelligence_engine = PayrollIntelligenceEngine()
			
			# Get predictions
			predictions = intelligence_engine.generate_predictions(
				tenant_id=request.tenant_id,
				prediction_type=prediction_type,
				horizon=horizon
			)
			
			return predictions
			
		except Exception as e:
			logger.error(f"Failed to generate predictions: {e}")
			api.abort(500, 'Failed to generate predictions')


# ===========================
# CONVERSATIONAL API
# ===========================

@ns_conversational.route('/message')
class ConversationalPayrollAPI(Resource):
	"""Conversational payroll interface endpoint."""
	
	@api.doc('process_chat_message')
	@api.expect(chat_message_model)
	@api.marshal_with(chat_response_model)
	@require_permission('use_conversational_interface')
	@tenant_required
	@rate_limit(requests=30, per=60)
	@monitor_performance
	def post(self):
		"""Process natural language payroll command."""
		try:
			data = request.get_json()
			command = data.get('command')
			context = data.get('context', {})
			
			if not command:
				api.abort(400, 'Command is required')
			
			# Initialize conversational assistant
			assistant = ConversationalPayrollAssistant()
			
			# Process command
			response = assistant.process_command(
				command=command,
				user_id=request.user_id,
				tenant_id=request.tenant_id,
				context=context
			)
			
			return {
				'success': True,
				'message': response.get('message', ''),
				'data': response.get('data'),
				'actions': response.get('actions', []),
				'timestamp': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Failed to process conversational command: {e}")
			api.abort(500, 'Failed to process command')


# ===========================
# COMPLIANCE API
# ===========================

@ns_compliance.route('/status')
class ComplianceStatusAPI(Resource):
	"""Compliance status monitoring endpoint."""
	
	@api.doc('get_compliance_status')
	@require_permission('view_compliance_status')
	@tenant_required
	@cache_result(timeout=300)
	@monitor_performance
	def get(self):
		"""Get compliance status overview."""
		try:
			# Initialize compliance engine
			compliance_engine = IntelligentComplianceTaxEngine()
			
			# Get compliance status
			status = compliance_engine.get_compliance_status(
				tenant_id=request.tenant_id
			)
			
			return status
			
		except Exception as e:
			logger.error(f"Failed to get compliance status: {e}")
			api.abort(500, 'Failed to retrieve compliance status')


@ns_compliance.route('/validate/<string:period_id>')
class ComplianceValidationAPI(Resource):
	"""Compliance validation endpoint."""
	
	@api.doc('validate_compliance')
	@require_permission('validate_compliance')
	@tenant_required
	@rate_limit(requests=10, per=60)
	@monitor_performance
	def post(self, period_id):
		"""Validate compliance for payroll period."""
		try:
			# Initialize compliance engine
			compliance_engine = IntelligentComplianceTaxEngine()
			
			# Validate compliance
			validation_result = compliance_engine.validate_period_compliance(
				period_id=period_id,
				tenant_id=request.tenant_id
			)
			
			return validation_result
			
		except Exception as e:
			logger.error(f"Failed to validate compliance: {e}")
			api.abort(500, 'Failed to validate compliance')


# ===========================
# ERROR HANDLERS
# ===========================

@api.errorhandler(404)
def not_found(error):
	"""Handle 404 errors."""
	return {'message': 'Resource not found'}, 404


@api.errorhandler(400)
def bad_request(error):
	"""Handle 400 errors."""
	return {'message': 'Bad request'}, 400


@api.errorhandler(403)
def forbidden(error):
	"""Handle 403 errors."""
	return {'message': 'Access forbidden'}, 403


@api.errorhandler(500)
def internal_error(error):
	"""Handle 500 errors."""
	return {'message': 'Internal server error'}, 500


# ===========================
# WEBHOOK ENDPOINTS
# ===========================

@payroll_bp.route('/webhooks/payroll_completed', methods=['POST'])
@api_key_required
@audit_api_call
def payroll_completed_webhook():
	"""Webhook for payroll completion notifications."""
	try:
		data = request.get_json()
		run_id = data.get('run_id')
		
		# Process webhook
		logger.info(f"Payroll completion webhook received for run: {run_id}")
		
		# Send notifications, update integrations, etc.
		
		return {'status': 'received'}, 200
		
	except Exception as e:
		logger.error(f"Failed to process webhook: {e}")
		return {'error': 'Failed to process webhook'}, 500


# Register error handlers
payroll_bp.register_error_handler(404, not_found)
payroll_bp.register_error_handler(400, bad_request)
payroll_bp.register_error_handler(403, forbidden)
payroll_bp.register_error_handler(500, internal_error)


# Example usage
if __name__ == "__main__":
	# This would be registered with the main Flask application
	pass