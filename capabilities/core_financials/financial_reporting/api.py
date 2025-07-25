"""
Financial Reporting API

REST API endpoints for Financial Reporting operations including
statement generation, consolidation, analytical reporting, and distribution.
"""

from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from flask_appbuilder.security.decorators import protect
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from decimal import Decimal
import json

from .models import (
	CFRFReportTemplate, CFRFReportDefinition, CFRFReportLine, CFRFReportPeriod,
	CFRFReportGeneration, CFRFFinancialStatement, CFRFConsolidation, CFRFNotes,
	CFRFDisclosure, CFRFAnalyticalReport, CFRFReportDistribution
)
from .service import FinancialReportingService
from ...auth_rbac.models import db

# Create API blueprint
api_bp = Blueprint('fr_api', __name__, url_prefix='/api/core_financials/fr')
api = Api(api_bp, doc='/doc/', title='Financial Reporting API', version='1.0')

# Define namespaces
templates_ns = Namespace('templates', description='Report template operations')
statements_ns = Namespace('statements', description='Financial statement operations')
consolidations_ns = Namespace('consolidations', description='Consolidation operations')
reports_ns = Namespace('reports', description='Analytical report operations')
generation_ns = Namespace('generation', description='Report generation operations')
distribution_ns = Namespace('distribution', description='Report distribution operations')

api.add_namespace(templates_ns)
api.add_namespace(statements_ns)
api.add_namespace(consolidations_ns)
api.add_namespace(reports_ns)
api.add_namespace(generation_ns)
api.add_namespace(distribution_ns)

# Define common models for API documentation
template_model = api.model('ReportTemplate', {
	'template_id': fields.String(description='Template ID'),
	'template_code': fields.String(required=True, description='Template code'),
	'template_name': fields.String(required=True, description='Template name'),
	'description': fields.String(description='Template description'),
	'statement_type': fields.String(required=True, description='Statement type'),
	'category': fields.String(description='Template category'),
	'format_type': fields.String(description='Format type'),
	'is_active': fields.Boolean(description='Is template active'),
	'version': fields.String(description='Template version'),
	'currency_type': fields.String(description='Currency type'),
	'show_percentages': fields.Boolean(description='Show percentages'),
	'show_variances': fields.Boolean(description='Show variances'),
	'decimal_places': fields.Integer(description='Decimal places')
})

statement_model = api.model('FinancialStatement', {
	'statement_id': fields.String(description='Statement ID'),
	'statement_name': fields.String(description='Statement name'),
	'statement_type': fields.String(description='Statement type'),
	'as_of_date': fields.Date(description='As of date'),
	'currency_code': fields.String(description='Currency code'),
	'reporting_entity': fields.String(description='Reporting entity'),
	'is_final': fields.Boolean(description='Is statement final'),
	'is_published': fields.Boolean(description='Is statement published'),
	'total_assets': fields.Float(description='Total assets'),
	'total_liabilities': fields.Float(description='Total liabilities'),
	'total_equity': fields.Float(description='Total equity'),
	'total_revenue': fields.Float(description='Total revenue'),
	'net_income': fields.Float(description='Net income')
})

generation_request_model = api.model('GenerationRequest', {
	'template_id': fields.String(required=True, description='Template ID'),
	'period_id': fields.String(required=True, description='Period ID'),
	'as_of_date': fields.Date(required=True, description='As of date'),
	'generation_name': fields.String(description='Generation name'),
	'description': fields.String(description='Generation description'),
	'include_adjustments': fields.Boolean(description='Include adjustments'),
	'consolidation_level': fields.String(description='Consolidation level'),
	'currency_code': fields.String(description='Currency code'),
	'output_format': fields.String(description='Output format')
})

consolidation_model = api.model('Consolidation', {
	'consolidation_id': fields.String(description='Consolidation ID'),
	'consolidation_code': fields.String(required=True, description='Consolidation code'),
	'consolidation_name': fields.String(required=True, description='Consolidation name'),
	'parent_entity': fields.String(required=True, description='Parent entity'),
	'subsidiary_entity': fields.String(required=True, description='Subsidiary entity'),
	'consolidation_method': fields.String(required=True, description='Consolidation method'),
	'ownership_percentage': fields.Float(required=True, description='Ownership percentage'),
	'voting_percentage': fields.Float(description='Voting percentage'),
	'effective_from': fields.Date(required=True, description='Effective from date'),
	'effective_to': fields.Date(description='Effective to date'),
	'is_active': fields.Boolean(description='Is consolidation active')
})


def get_current_tenant() -> str:
	"""Get current tenant ID from request context"""
	# This should be implemented based on your authentication system
	return request.headers.get('X-Tenant-ID', 'default_tenant')


def serialize_decimal(obj):
	"""Serialize Decimal objects to float for JSON"""
	if isinstance(obj, Decimal):
		return float(obj)
	raise TypeError


# Report Templates API
@templates_ns.route('/')
class ReportTemplateListAPI(Resource):
	@protect
	@templates_ns.marshal_list_with(template_model)
	def get(self):
		"""Get all report templates"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		templates = db.session.query(CFRFReportTemplate).filter(
			CFRFReportTemplate.tenant_id == tenant_id
		).all()
		
		return templates
	
	@protect
	@templates_ns.expect(template_model)
	@templates_ns.marshal_with(template_model)
	def post(self):
		"""Create a new report template"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		try:
			template = service.create_report_template(request.json)
			return template, 201
		except Exception as e:
			api.abort(400, str(e))


@templates_ns.route('/<string:template_id>')
class ReportTemplateAPI(Resource):
	@protect
	@templates_ns.marshal_with(template_model)
	def get(self, template_id):
		"""Get a specific report template"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		template = service.get_report_template(template_id)
		if not template:
			api.abort(404, 'Template not found')
		
		return template
	
	@protect
	@templates_ns.expect(template_model)
	@templates_ns.marshal_with(template_model)
	def put(self, template_id):
		"""Update a report template"""
		tenant_id = get_current_tenant()
		template = db.session.query(CFRFReportTemplate).filter(
			and_(
				CFRFReportTemplate.template_id == template_id,
				CFRFReportTemplate.tenant_id == tenant_id
			)
		).first()
		
		if not template:
			api.abort(404, 'Template not found')
		
		try:
			for key, value in request.json.items():
				if hasattr(template, key):
					setattr(template, key, value)
			
			db.session.commit()
			return template
		except Exception as e:
			db.session.rollback()
			api.abort(400, str(e))
	
	@protect
	def delete(self, template_id):
		"""Delete a report template"""
		tenant_id = get_current_tenant()
		template = db.session.query(CFRFReportTemplate).filter(
			and_(
				CFRFReportTemplate.template_id == template_id,
				CFRFReportTemplate.tenant_id == tenant_id
			)
		).first()
		
		if not template:
			api.abort(404, 'Template not found')
		
		if template.is_system:
			api.abort(400, 'Cannot delete system template')
		
		try:
			db.session.delete(template)
			db.session.commit()
			return {'message': 'Template deleted successfully'}, 200
		except Exception as e:
			db.session.rollback()
			api.abort(400, str(e))


@templates_ns.route('/by-type/<string:statement_type>')
class ReportTemplateByTypeAPI(Resource):
	@protect
	@templates_ns.marshal_list_with(template_model)
	def get(self, statement_type):
		"""Get templates by statement type"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		templates = service.get_templates_by_type(statement_type)
		return templates


# Financial Statements API
@statements_ns.route('/')
class FinancialStatementListAPI(Resource):
	@protect
	@statements_ns.marshal_list_with(statement_model)
	def get(self):
		"""Get all financial statements"""
		tenant_id = get_current_tenant()
		
		# Get query parameters
		period_id = request.args.get('period_id')
		statement_type = request.args.get('statement_type')
		is_final = request.args.get('is_final')
		
		query = db.session.query(CFRFFinancialStatement).filter(
			CFRFFinancialStatement.tenant_id == tenant_id
		)
		
		if period_id:
			query = query.filter(CFRFFinancialStatement.period_id == period_id)
		
		if statement_type:
			query = query.filter(CFRFFinancialStatement.statement_type == statement_type)
		
		if is_final is not None:
			query = query.filter(CFRFFinancialStatement.is_final == (is_final.lower() == 'true'))
		
		statements = query.order_by(CFRFFinancialStatement.created_at.desc()).all()
		return statements


@statements_ns.route('/<string:statement_id>')
class FinancialStatementAPI(Resource):
	@protect
	@statements_ns.marshal_with(statement_model)
	def get(self, statement_id):
		"""Get a specific financial statement"""
		tenant_id = get_current_tenant()
		
		statement = db.session.query(CFRFFinancialStatement).filter(
			and_(
				CFRFFinancialStatement.statement_id == statement_id,
				CFRFFinancialStatement.tenant_id == tenant_id
			)
		).first()
		
		if not statement:
			api.abort(404, 'Statement not found')
		
		return statement


@statements_ns.route('/<string:statement_id>/data')
class FinancialStatementDataAPI(Resource):
	@protect
	def get(self, statement_id):
		"""Get financial statement data"""
		tenant_id = get_current_tenant()
		
		statement = db.session.query(CFRFFinancialStatement).filter(
			and_(
				CFRFFinancialStatement.statement_id == statement_id,
				CFRFFinancialStatement.tenant_id == tenant_id
			)
		).first()
		
		if not statement:
			api.abort(404, 'Statement not found')
		
		return jsonify(statement.statement_data)


@statements_ns.route('/<string:statement_id>/publish')
class FinancialStatementPublishAPI(Resource):
	@protect
	def post(self, statement_id):
		"""Publish a financial statement"""
		tenant_id = get_current_tenant()
		
		statement = db.session.query(CFRFFinancialStatement).filter(
			and_(
				CFRFFinancialStatement.statement_id == statement_id,
				CFRFFinancialStatement.tenant_id == tenant_id
			)
		).first()
		
		if not statement:
			api.abort(404, 'Statement not found')
		
		if not statement.is_final:
			api.abort(400, 'Statement must be finalized before publishing')
		
		if statement.is_published:
			api.abort(400, 'Statement is already published')
		
		try:
			statement.is_published = True
			db.session.commit()
			return {'message': 'Statement published successfully'}, 200
		except Exception as e:
			db.session.rollback()
			api.abort(400, str(e))


# Report Generation API
@generation_ns.route('/generate')
class ReportGenerationAPI(Resource):
	@protect
	@generation_ns.expect(generation_request_model)
	def post(self):
		"""Generate a financial statement"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		try:
			generation = service.generate_financial_statement(request.json)
			
			return {
				'generation_id': generation.generation_id,
				'status': generation.status,
				'message': 'Generation started successfully'
			}, 202
		except Exception as e:
			api.abort(400, str(e))


@generation_ns.route('/status/<string:generation_id>')
class GenerationStatusAPI(Resource):
	@protect
	def get(self, generation_id):
		"""Get generation status"""
		tenant_id = get_current_tenant()
		
		generation = db.session.query(CFRFReportGeneration).filter(
			and_(
				CFRFReportGeneration.generation_id == generation_id,
				CFRFReportGeneration.tenant_id == tenant_id
			)
		).first()
		
		if not generation:
			api.abort(404, 'Generation not found')
		
		return {
			'generation_id': generation.generation_id,
			'generation_name': generation.generation_name,
			'status': generation.status,
			'progress_percentage': generation.progress_percentage,
			'start_time': generation.start_time.isoformat() if generation.start_time else None,
			'end_time': generation.end_time.isoformat() if generation.end_time else None,
			'duration_seconds': generation.duration_seconds,
			'error_count': generation.error_count,
			'warning_count': generation.warning_count,
			'balance_verified': generation.balance_verified,
			'approval_status': generation.approval_status
		}


@generation_ns.route('/history')
class GenerationHistoryAPI(Resource):
	@protect
	def get(self):
		"""Get generation history"""
		tenant_id = get_current_tenant()
		
		# Get query parameters
		limit = request.args.get('limit', 50, type=int)
		status = request.args.get('status')
		template_id = request.args.get('template_id')
		
		query = db.session.query(CFRFReportGeneration).filter(
			CFRFReportGeneration.tenant_id == tenant_id
		)
		
		if status:
			query = query.filter(CFRFReportGeneration.status == status)
		
		if template_id:
			query = query.filter(CFRFReportGeneration.template_id == template_id)
		
		generations = query.order_by(
			CFRFReportGeneration.created_at.desc()
		).limit(limit).all()
		
		return [{
			'generation_id': g.generation_id,
			'generation_name': g.generation_name,
			'status': g.status,
			'progress_percentage': g.progress_percentage,
			'created_at': g.created_at.isoformat(),
			'duration_seconds': g.duration_seconds,
			'template_id': g.template_id,
			'period_id': g.period_id
		} for g in generations]


# Consolidation API
@consolidations_ns.route('/')
class ConsolidationListAPI(Resource):
	@protect
	@consolidations_ns.marshal_list_with(consolidation_model)
	def get(self):
		"""Get all consolidation rules"""
		tenant_id = get_current_tenant()
		
		consolidations = db.session.query(CFRFConsolidation).filter(
			CFRFConsolidation.tenant_id == tenant_id
		).all()
		
		return consolidations
	
	@protect
	@consolidations_ns.expect(consolidation_model)
	@consolidations_ns.marshal_with(consolidation_model)
	def post(self):
		"""Create a new consolidation rule"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		try:
			consolidation = service.create_consolidation(request.json)
			return consolidation, 201
		except Exception as e:
			api.abort(400, str(e))


@consolidations_ns.route('/<string:consolidation_id>')
class ConsolidationAPI(Resource):
	@protect
	@consolidations_ns.marshal_with(consolidation_model)
	def get(self, consolidation_id):
		"""Get a specific consolidation rule"""
		tenant_id = get_current_tenant()
		
		consolidation = db.session.query(CFRFConsolidation).filter(
			and_(
				CFRFConsolidation.consolidation_id == consolidation_id,
				CFRFConsolidation.tenant_id == tenant_id
			)
		).first()
		
		if not consolidation:
			api.abort(404, 'Consolidation not found')
		
		return consolidation


@consolidations_ns.route('/<string:consolidation_id>/perform')
class ConsolidationPerformAPI(Resource):
	@protect
	def post(self, consolidation_id):
		"""Perform consolidation for a period"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		period_id = request.json.get('period_id')
		if not period_id:
			api.abort(400, 'Period ID is required')
		
		try:
			result = service.perform_consolidation(consolidation_id, period_id)
			return result, 200
		except Exception as e:
			api.abort(400, str(e))


# Analytical Reports API
@reports_ns.route('/')
class AnalyticalReportListAPI(Resource):
	@protect
	def get(self):
		"""Get all analytical reports"""
		tenant_id = get_current_tenant()
		
		reports = db.session.query(CFRFAnalyticalReport).filter(
			CFRFAnalyticalReport.tenant_id == tenant_id
		).all()
		
		return [{
			'report_id': r.report_id,
			'report_code': r.report_code,
			'report_name': r.report_name,
			'report_type': r.report_type,
			'report_category': r.report_category,
			'analysis_type': r.analysis_type,
			'is_scheduled': r.is_scheduled,
			'last_generated': r.last_generated.isoformat() if r.last_generated else None,
			'is_active': r.is_active
		} for r in reports]
	
	@protect
	def post(self):
		"""Create a new analytical report"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		try:
			report = service.create_analytical_report(request.json)
			return {
				'report_id': report.report_id,
				'report_code': report.report_code,
				'report_name': report.report_name,
				'message': 'Analytical report created successfully'
			}, 201
		except Exception as e:
			api.abort(400, str(e))


@reports_ns.route('/<string:report_id>/generate')
class AnalyticalReportGenerateAPI(Resource):
	@protect
	def post(self, report_id):
		"""Generate an analytical report"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		parameters = request.json or {}
		
		try:
			result = service.generate_analytical_report(report_id, parameters)
			return result, 200
		except Exception as e:
			api.abort(400, str(e))


# Distribution API
@distribution_ns.route('/')
class DistributionListAPI(Resource):
	@protect
	def get(self):
		"""Get all distribution lists"""
		tenant_id = get_current_tenant()
		
		distributions = db.session.query(CFRFReportDistribution).filter(
			CFRFReportDistribution.tenant_id == tenant_id
		).all()
		
		return [{
			'distribution_id': d.distribution_id,
			'distribution_name': d.distribution_name,
			'distribution_type': d.distribution_type,
			'delivery_method': d.delivery_method,
			'delivery_format': d.delivery_format,
			'last_distribution': d.last_distribution.isoformat() if d.last_distribution else None,
			'success_count': d.success_count,
			'failure_count': d.failure_count,
			'is_active': d.is_active
		} for d in distributions]
	
	@protect
	def post(self):
		"""Create a new distribution list"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		try:
			distribution = service.create_distribution_list(request.json)
			return {
				'distribution_id': distribution.distribution_id,
				'distribution_name': distribution.distribution_name,
				'message': 'Distribution list created successfully'
			}, 201
		except Exception as e:
			api.abort(400, str(e))


@distribution_ns.route('/<string:distribution_id>/distribute')
class DistributionExecuteAPI(Resource):
	@protect
	def post(self, distribution_id):
		"""Execute report distribution"""
		tenant_id = get_current_tenant()
		service = FinancialReportingService(tenant_id)
		
		report_data = request.json or {}
		
		try:
			result = service.distribute_report(distribution_id, report_data)
			return result, 200
		except Exception as e:
			api.abort(400, str(e))


# Dashboard and Metrics API
@api_bp.route('/metrics')
@protect
def get_dashboard_metrics():
	"""Get financial dashboard metrics"""
	tenant_id = get_current_tenant()
	service = FinancialReportingService(tenant_id)
	
	try:
		# Get current period
		current_period = service.get_current_period()
		
		if current_period:
			summary = service.get_financial_summary(current_period.period_id)
		else:
			summary = {
				'period_id': None,
				'statement_count': 0,
				'total_assets': 0,
				'total_liabilities': 0,
				'total_equity': 0,
				'total_revenue': 0,
				'net_income': 0
			}
		
		# Get generation statistics
		generation_stats = {
			'pending': db.session.query(CFRFReportGeneration).filter(
				and_(
					CFRFReportGeneration.tenant_id == tenant_id,
					CFRFReportGeneration.status == 'pending'
				)
			).count(),
			'running': db.session.query(CFRFReportGeneration).filter(
				and_(
					CFRFReportGeneration.tenant_id == tenant_id,
					CFRFReportGeneration.status == 'running'
				)
			).count(),
			'completed': db.session.query(CFRFReportGeneration).filter(
				and_(
					CFRFReportGeneration.tenant_id == tenant_id,
					CFRFReportGeneration.status == 'completed'
				)
			).count(),
			'failed': db.session.query(CFRFReportGeneration).filter(
				and_(
					CFRFReportGeneration.tenant_id == tenant_id,
					CFRFReportGeneration.status == 'failed'
				)
			).count()
		}
		
		return jsonify({
			'financial_summary': summary,
			'generation_stats': generation_stats,
			'current_period': {
				'period_id': current_period.period_id if current_period else None,
				'period_name': current_period.period_name if current_period else None,
				'start_date': current_period.start_date.isoformat() if current_period else None,
				'end_date': current_period.end_date.isoformat() if current_period else None,
				'is_closed': current_period.is_closed if current_period else None
			}
		})
		
	except Exception as e:
		return jsonify({'error': str(e)}), 500


# Health check endpoint
@api_bp.route('/health')
def health_check():
	"""Health check for Financial Reporting API"""
	return jsonify({
		'status': 'healthy',
		'timestamp': datetime.now().isoformat(),
		'version': '1.0.0'
	})


# Error handlers
@api_bp.errorhandler(400)
def bad_request(error):
	return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@api_bp.errorhandler(404)
def not_found(error):
	return jsonify({'error': 'Not found', 'message': str(error)}), 404


@api_bp.errorhandler(500)
def internal_error(error):
	return jsonify({'error': 'Internal server error', 'message': str(error)}), 500