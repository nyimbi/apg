"""
Regulatory Compliance API

REST API endpoints for pharmaceutical regulatory compliance operations
including submissions, audits, deviations, and compliance monitoring.
"""

from datetime import datetime, date
from typing import Dict, List, Any
from flask import Blueprint, request, jsonify, current_app
from flask_restful import Api, Resource, reqparse
from marshmallow import Schema, fields, validate, ValidationError

from ....auth_rbac.decorators import require_permissions, get_current_tenant
from .models import (
	PHRCRegulatoryFramework, PHRCSubmission, PHRCSubmissionDocument,
	PHRCAudit, PHRCAuditFinding, PHRCDeviation, PHRCCorrectiveAction,
	PHRCComplianceControl, PHRCRegulatoryContact, PHRCInspection,
	PHRCRegulatoryReport
)
from .service import RegulatoryComplianceService


# Marshmallow schemas for request/response validation

class RegulatoryFrameworkSchema(Schema):
	framework_id = fields.Str(dump_only=True)
	framework_code = fields.Str(required=True, validate=validate.Length(max=20))
	framework_name = fields.Str(required=True, validate=validate.Length(max=200))
	region = fields.Str(validate=validate.Length(max=100))
	description = fields.Str()
	website_url = fields.Url()
	is_active = fields.Bool()
	version = fields.Str(validate=validate.Length(max=50))
	effective_date = fields.Date()


class SubmissionSchema(Schema):
	submission_id = fields.Str(dump_only=True)
	submission_number = fields.Str(dump_only=True)
	submission_type = fields.Str(required=True, validate=validate.OneOf([
		'IND', 'NDA', 'BLA', 'ANDA', 'MAA', 'DMF', 'CTD', 'ASMF'
	]))
	submission_title = fields.Str(required=True, validate=validate.Length(max=500))
	description = fields.Str()
	product_name = fields.Str(validate=validate.Length(max=200))
	active_ingredient = fields.Str(validate=validate.Length(max=200))
	therapeutic_area = fields.Str(validate=validate.Length(max=100))
	indication = fields.Str()
	framework_id = fields.Str(required=True)
	status = fields.Str(validate=validate.OneOf([
		'Draft', 'In Review', 'Submitted', 'Under Review', 'Approved', 'Rejected'
	]))
	priority_designation = fields.Str(validate=validate.OneOf([
		'Standard', 'Priority', 'Fast Track', 'Breakthrough', 'Accelerated'
	]))


class AuditSchema(Schema):
	audit_id = fields.Str(dump_only=True)
	audit_number = fields.Str(dump_only=True)
	audit_title = fields.Str(required=True, validate=validate.Length(max=500))
	audit_type = fields.Str(required=True, validate=validate.OneOf([
		'Internal', 'External', 'Self-Assessment', 'Regulatory', 'Vendor'
	]))
	audit_scope = fields.Str()
	framework_id = fields.Str(required=True)
	planned_start_date = fields.Date()
	planned_end_date = fields.Date()
	lead_auditor = fields.Str(validate=validate.Length(max=200))
	auditee_contact = fields.Str(validate=validate.Length(max=200))
	status = fields.Str(validate=validate.OneOf([
		'Planned', 'In Progress', 'Completed', 'Cancelled'
	]))


class DeviationSchema(Schema):
	deviation_id = fields.Str(dump_only=True)
	deviation_number = fields.Str(dump_only=True)
	deviation_title = fields.Str(required=True, validate=validate.Length(max=500))
	description = fields.Str(required=True)
	deviation_type = fields.Str(required=True, validate=validate.OneOf([
		'Process', 'Product', 'System', 'Equipment', 'Personnel', 'Environmental'
	]))
	severity = fields.Str(required=True, validate=validate.OneOf([
		'Critical', 'Major', 'Minor'
	]))
	process_area = fields.Str(validate=validate.Length(max=200))
	product_affected = fields.Str(validate=validate.Length(max=200))
	discovered_by = fields.Str(required=True)
	discovery_method = fields.Str(validate=validate.Length(max=200))


class CorrectiveActionSchema(Schema):
	action_id = fields.Str(dump_only=True)
	action_number = fields.Str(dump_only=True)
	action_title = fields.Str(required=True, validate=validate.Length(max=500))
	description = fields.Str(required=True)
	action_type = fields.Str(required=True, validate=validate.OneOf([
		'Corrective', 'Preventive', 'Both'
	]))
	category = fields.Str(validate=validate.OneOf([
		'Process', 'Training', 'System', 'Documentation', 'Equipment'
	]))
	planned_completion_date = fields.Date(required=True)
	assigned_to = fields.Str(required=True)


# API Blueprint
regulatory_compliance_api = Blueprint('regulatory_compliance_api', __name__)
api = Api(regulatory_compliance_api)


class RegulatoryFrameworkListAPI(Resource):
	"""API for regulatory frameworks"""
	
	@require_permissions('ph.regulatory.read')
	def get(self):
		"""Get list of regulatory frameworks"""
		tenant_id = get_current_tenant()
		
		frameworks = PHRCRegulatoryFramework.query.filter_by(
			tenant_id=tenant_id,
			is_active=True
		).order_by(PHRCRegulatoryFramework.framework_name).all()
		
		schema = RegulatoryFrameworkSchema(many=True)
		return {'frameworks': schema.dump(frameworks)}, 200
	
	@require_permissions('ph.regulatory.write')
	def post(self):
		"""Create new regulatory framework"""
		tenant_id = get_current_tenant()
		
		schema = RegulatoryFrameworkSchema()
		try:
			data = schema.load(request.json)
		except ValidationError as err:
			return {'errors': err.messages}, 400
		
		service = RegulatoryComplianceService(tenant_id)
		
		try:
			framework = service.create_framework(data)
			return {'framework': schema.dump(framework)}, 201
		except Exception as e:
			return {'error': str(e)}, 500


class RegulatoryFrameworkAPI(Resource):
	"""API for individual regulatory framework"""
	
	@require_permissions('ph.regulatory.read')
	def get(self, framework_id):
		"""Get regulatory framework by ID"""
		tenant_id = get_current_tenant()
		
		framework = PHRCRegulatoryFramework.query.filter_by(
			framework_id=framework_id,
			tenant_id=tenant_id
		).first()
		
		if not framework:
			return {'error': 'Framework not found'}, 404
		
		schema = RegulatoryFrameworkSchema()
		return {'framework': schema.dump(framework)}, 200


class SubmissionListAPI(Resource):
	"""API for regulatory submissions"""
	
	@require_permissions('ph.regulatory.read')
	def get(self):
		"""Get list of submissions with filtering"""
		tenant_id = get_current_tenant()
		
		# Parse query parameters
		parser = reqparse.RequestParser()
		parser.add_argument('status', type=str, help='Filter by status')
		parser.add_argument('submission_type', type=str, help='Filter by submission type')
		parser.add_argument('framework_id', type=str, help='Filter by framework')
		parser.add_argument('page', type=int, default=1, help='Page number')
		parser.add_argument('per_page', type=int, default=20, help='Items per page')
		args = parser.parse_args()
		
		# Build query
		query = PHRCSubmission.query.filter_by(tenant_id=tenant_id)
		
		if args['status']:
			query = query.filter(PHRCSubmission.status == args['status'])
		if args['submission_type']:
			query = query.filter(PHRCSubmission.submission_type == args['submission_type'])
		if args['framework_id']:
			query = query.filter(PHRCSubmission.framework_id == args['framework_id'])
		
		# Paginate
		submissions = query.order_by(
			PHRCSubmission.submission_date.desc()
		).paginate(
			page=args['page'],
			per_page=args['per_page'],
			error_out=False
		)
		
		schema = SubmissionSchema(many=True)
		return {
			'submissions': schema.dump(submissions.items),
			'pagination': {
				'page': submissions.page,
				'pages': submissions.pages,
				'per_page': submissions.per_page,
				'total': submissions.total,
				'has_next': submissions.has_next,
				'has_prev': submissions.has_prev
			}
		}, 200
	
	@require_permissions('ph.regulatory.write')
	def post(self):
		"""Create new submission"""
		tenant_id = get_current_tenant()
		
		schema = SubmissionSchema()
		try:
			data = schema.load(request.json)
		except ValidationError as err:
			return {'errors': err.messages}, 400
		
		service = RegulatoryComplianceService(tenant_id)
		
		try:
			submission = service.create_submission(data)
			return {'submission': schema.dump(submission)}, 201
		except Exception as e:
			return {'error': str(e)}, 500


class SubmissionAPI(Resource):
	"""API for individual submission"""
	
	@require_permissions('ph.regulatory.read')
	def get(self, submission_id):
		"""Get submission status and details"""
		tenant_id = get_current_tenant()
		
		service = RegulatoryComplianceService(tenant_id)
		status = service.get_submission_status(submission_id)
		
		if not status:
			return {'error': 'Submission not found'}, 404
		
		return {'submission_status': status}, 200


class SubmissionSubmitAPI(Resource):
	"""API for submitting to regulatory authority"""
	
	@require_permissions('ph.regulatory.submit')
	def post(self, submission_id):
		"""Submit to regulatory authority"""
		tenant_id = get_current_tenant()
		
		parser = reqparse.RequestParser()
		parser.add_argument('submission_date', type=str, help='Submission date (YYYY-MM-DD)')
		args = parser.parse_args()
		
		submission_date = None
		if args['submission_date']:
			try:
				submission_date = datetime.strptime(args['submission_date'], '%Y-%m-%d').date()
			except ValueError:
				return {'error': 'Invalid date format. Use YYYY-MM-DD'}, 400
		
		service = RegulatoryComplianceService(tenant_id)
		
		try:
			success = service.submit_to_authority(submission_id, submission_date)
			if success:
				return {'message': 'Submission filed successfully'}, 200
			else:
				return {'error': 'Submission not found'}, 404
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			return {'error': str(e)}, 500


class AuditListAPI(Resource):
	"""API for regulatory audits"""
	
	@require_permissions('ph.regulatory.audit')
	def get(self):
		"""Get list of audits"""
		tenant_id = get_current_tenant()
		
		parser = reqparse.RequestParser()
		parser.add_argument('status', type=str, help='Filter by status')
		parser.add_argument('audit_type', type=str, help='Filter by audit type')
		parser.add_argument('page', type=int, default=1, help='Page number')
		parser.add_argument('per_page', type=int, default=20, help='Items per page')
		args = parser.parse_args()
		
		query = PHRCAudit.query.filter_by(tenant_id=tenant_id)
		
		if args['status']:
			query = query.filter(PHRCAudit.status == args['status'])
		if args['audit_type']:
			query = query.filter(PHRCAudit.audit_type == args['audit_type'])
		
		audits = query.order_by(
			PHRCAudit.planned_start_date.desc()
		).paginate(
			page=args['page'],
			per_page=args['per_page'],
			error_out=False
		)
		
		schema = AuditSchema(many=True)
		return {
			'audits': schema.dump(audits.items),
			'pagination': {
				'page': audits.page,
				'pages': audits.pages,
				'per_page': audits.per_page,
				'total': audits.total
			}
		}, 200
	
	@require_permissions('ph.regulatory.audit')
	def post(self):
		"""Create new audit"""
		tenant_id = get_current_tenant()
		
		schema = AuditSchema()
		try:
			data = schema.load(request.json)
		except ValidationError as err:
			return {'errors': err.messages}, 400
		
		service = RegulatoryComplianceService(tenant_id)
		
		try:
			audit = service.create_audit(data)
			return {'audit': schema.dump(audit)}, 201
		except Exception as e:
			return {'error': str(e)}, 500


class AuditSummaryAPI(Resource):
	"""API for audit summary"""
	
	@require_permissions('ph.regulatory.audit')
	def get(self, audit_id):
		"""Get audit summary with findings"""
		tenant_id = get_current_tenant()
		
		service = RegulatoryComplianceService(tenant_id)
		summary = service.get_audit_summary(audit_id)
		
		if not summary:
			return {'error': 'Audit not found'}, 404
		
		return {'audit_summary': summary}, 200


class DeviationListAPI(Resource):
	"""API for quality deviations"""
	
	@require_permissions('ph.regulatory.deviation')
	def get(self):
		"""Get list of deviations"""
		tenant_id = get_current_tenant()
		
		parser = reqparse.RequestParser()
		parser.add_argument('status', type=str, help='Filter by status')
		parser.add_argument('severity', type=str, help='Filter by severity')
		parser.add_argument('page', type=int, default=1, help='Page number')
		parser.add_argument('per_page', type=int, default=20, help='Items per page')
		args = parser.parse_args()
		
		query = PHRCDeviation.query.filter_by(tenant_id=tenant_id)
		
		if args['status']:
			query = query.filter(PHRCDeviation.status == args['status'])
		if args['severity']:
			query = query.filter(PHRCDeviation.severity == args['severity'])
		
		deviations = query.order_by(
			PHRCDeviation.discovered_date.desc()
		).paginate(
			page=args['page'],
			per_page=args['per_page'],
			error_out=False
		)
		
		schema = DeviationSchema(many=True)
		return {
			'deviations': schema.dump(deviations.items),
			'pagination': {
				'page': deviations.page,
				'pages': deviations.pages,
				'per_page': deviations.per_page,
				'total': deviations.total
			}
		}, 200
	
	@require_permissions('ph.regulatory.deviation')
	def post(self):
		"""Create new deviation"""
		tenant_id = get_current_tenant()
		
		schema = DeviationSchema()
		try:
			data = schema.load(request.json)
		except ValidationError as err:
			return {'errors': err.messages}, 400
		
		service = RegulatoryComplianceService(tenant_id)
		
		try:
			deviation = service.create_deviation(data)
			return {'deviation': schema.dump(deviation)}, 201
		except Exception as e:
			return {'error': str(e)}, 500


class CorrectiveActionListAPI(Resource):
	"""API for corrective and preventive actions"""
	
	@require_permissions('ph.regulatory.action')
	def get(self):
		"""Get list of CAPAs"""
		tenant_id = get_current_tenant()
		
		parser = reqparse.RequestParser()
		parser.add_argument('status', type=str, help='Filter by status')
		parser.add_argument('action_type', type=str, help='Filter by action type')
		parser.add_argument('assigned_to', type=str, help='Filter by assignee')
		parser.add_argument('page', type=int, default=1, help='Page number')
		parser.add_argument('per_page', type=int, default=20, help='Items per page')
		args = parser.parse_args()
		
		query = PHRCCorrectiveAction.query.filter_by(tenant_id=tenant_id)
		
		if args['status']:
			query = query.filter(PHRCCorrectiveAction.status == args['status'])
		if args['action_type']:
			query = query.filter(PHRCCorrectiveAction.action_type == args['action_type'])
		if args['assigned_to']:
			query = query.filter(PHRCCorrectiveAction.assigned_to == args['assigned_to'])
		
		actions = query.order_by(
			PHRCCorrectiveAction.planned_completion_date.asc()
		).paginate(
			page=args['page'],
			per_page=args['per_page'],
			error_out=False
		)
		
		schema = CorrectiveActionSchema(many=True)
		return {
			'actions': schema.dump(actions.items),
			'pagination': {
				'page': actions.page,
				'pages': actions.pages,
				'per_page': actions.per_page,
				'total': actions.total
			}
		}, 200
	
	@require_permissions('ph.regulatory.action')
	def post(self):
		"""Create new CAPA"""
		tenant_id = get_current_tenant()
		
		schema = CorrectiveActionSchema()
		try:
			data = schema.load(request.json)
		except ValidationError as err:
			return {'errors': err.messages}, 400
		
		service = RegulatoryComplianceService(tenant_id)
		
		try:
			action = service.create_corrective_action(data)
			return {'action': schema.dump(action)}, 201
		except Exception as e:
			return {'error': str(e)}, 500


class ComplianceDashboardAPI(Resource):
	"""API for compliance dashboard"""
	
	@require_permissions('ph.regulatory.read')
	def get(self):
		"""Get compliance dashboard data"""
		tenant_id = get_current_tenant()
		
		service = RegulatoryComplianceService(tenant_id)
		dashboard_data = service.get_compliance_dashboard()
		
		return {'dashboard': dashboard_data}, 200


# Register API endpoints
api.add_resource(RegulatoryFrameworkListAPI, '/frameworks')
api.add_resource(RegulatoryFrameworkAPI, '/frameworks/<string:framework_id>')
api.add_resource(SubmissionListAPI, '/submissions')
api.add_resource(SubmissionAPI, '/submissions/<string:submission_id>')
api.add_resource(SubmissionSubmitAPI, '/submissions/<string:submission_id>/submit')
api.add_resource(AuditListAPI, '/audits')
api.add_resource(AuditSummaryAPI, '/audits/<string:audit_id>/summary')
api.add_resource(DeviationListAPI, '/deviations')
api.add_resource(CorrectiveActionListAPI, '/actions')
api.add_resource(ComplianceDashboardAPI, '/dashboard')


def get_api_info() -> Dict[str, Any]:
	"""Get API information"""
	return {
		'name': 'Regulatory Compliance API',
		'version': '1.0.0',
		'description': 'REST API for pharmaceutical regulatory compliance operations',
		'endpoints': [
			{
				'path': '/frameworks',
				'methods': ['GET', 'POST'],
				'description': 'Manage regulatory frameworks'
			},
			{
				'path': '/frameworks/<framework_id>',
				'methods': ['GET'],
				'description': 'Get specific regulatory framework'
			},
			{
				'path': '/submissions',
				'methods': ['GET', 'POST'],
				'description': 'Manage regulatory submissions'
			},
			{
				'path': '/submissions/<submission_id>',
				'methods': ['GET'],
				'description': 'Get submission status'
			},
			{
				'path': '/submissions/<submission_id>/submit',
				'methods': ['POST'],
				'description': 'Submit to regulatory authority'
			},
			{
				'path': '/audits',
				'methods': ['GET', 'POST'],
				'description': 'Manage regulatory audits'
			},
			{
				'path': '/audits/<audit_id>/summary',
				'methods': ['GET'],
				'description': 'Get audit summary with findings'
			},
			{
				'path': '/deviations',
				'methods': ['GET', 'POST'],
				'description': 'Manage quality deviations'
			},
			{
				'path': '/actions',
				'methods': ['GET', 'POST'],
				'description': 'Manage corrective and preventive actions'
			},
			{
				'path': '/dashboard',
				'methods': ['GET'],
				'description': 'Get compliance dashboard data'
			}
		]
	}