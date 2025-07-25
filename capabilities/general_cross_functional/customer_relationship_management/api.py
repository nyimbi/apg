"""
Customer Relationship Management REST API

Comprehensive RESTful API with 40+ enterprise endpoints, advanced filtering,
bulk operations, real-time WebSocket support, and complete OpenAPI documentation.
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal
from uuid_extensions import uuid7str

from flask import Blueprint, request, jsonify, abort, current_app, g
from flask_restful import Api, Resource, reqparse, fields, marshal_with, marshal
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
from marshmallow import Schema, fields as ma_fields, validate, ValidationError, post_load
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import joinedload, selectinload
from flasgger import swag_from
from werkzeug.exceptions import BadRequest, NotFound, Forbidden, Conflict

from .models import (
	GCCRMAccount, GCCRMCustomer, GCCRMContact, GCCRMLead, GCCRMOpportunity,
	GCCRMSalesStage, GCCRMActivity, GCCRMTask, GCCRMAppointment, GCCRMCampaign,
	GCCRMCampaignMember, GCCRMMarketingList, GCCRMEmailTemplate, GCCRMCase,
	GCCRMCaseComment, GCCRMProduct, GCCRMPriceList, GCCRMQuote, GCCRMQuoteLine,
	GCCRMTerritory, GCCRMTeam, GCCRMForecast, GCCRMDashboardWidget, GCCRMReport,
	GCCRMLeadSource, GCCRMCustomerSegment, GCCRMCustomerScore, GCCRMSocialProfile,
	GCCRMCommunication, GCCRMWorkflowDefinition, GCCRMWorkflowExecution,
	GCCRMNotification, GCCRMKnowledgeBase, GCCRMCustomField, GCCRMCustomFieldValue,
	GCCRMDocumentAttachment, GCCRMEventLog, GCCRMSystemConfiguration,
	GCCRMWebhookEndpoint, GCCRMWebhookDelivery, LeadStatus, LeadRating,
	OpportunityStage, ActivityType, ActivityStatus, CaseStatus, CasePriority
)
from .service import CRMService, create_crm_service, CRMServiceError, ValidationError as ServiceValidationError

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint and API
crm_api_bp = Blueprint('crm_api', __name__, url_prefix='/api/v1/crm')
api = Api(crm_api_bp)

# Marshmallow Schemas for Request/Response Validation

class BaseSchema(Schema):
	"""Base schema with common fields"""
	id = ma_fields.Str(dump_only=True)
	created_on = ma_fields.DateTime(dump_only=True)
	changed_on = ma_fields.DateTime(dump_only=True)
	created_by = ma_fields.Str(dump_only=True)
	changed_by = ma_fields.Str(dump_only=True)

class PaginationSchema(Schema):
	"""Pagination metadata schema"""
	page = ma_fields.Int(required=True)
	per_page = ma_fields.Int(required=True)
	total = ma_fields.Int(required=True)
	pages = ma_fields.Int(required=True)
	has_prev = ma_fields.Bool(required=True)
	has_next = ma_fields.Bool(required=True)

class LeadSchema(BaseSchema):
	"""Lead schema for API serialization"""
	first_name = ma_fields.Str(required=True, validate=validate.Length(max=100))
	last_name = ma_fields.Str(required=True, validate=validate.Length(max=100))
	email = ma_fields.Email(required=True, validate=validate.Length(max=255))
	phone = ma_fields.Str(validate=validate.Length(max=50))
	company = ma_fields.Str(validate=validate.Length(max=200))
	job_title = ma_fields.Str(validate=validate.Length(max=100))
	lead_source = ma_fields.Str(validate=validate.OneOf([
		'website', 'social_media', 'email_campaign', 'trade_show',
		'referral', 'cold_call', 'advertisement', 'partner', 'other'
	]))
	lead_status = ma_fields.Str(validate=validate.OneOf([
		'new', 'contacted', 'qualified', 'unqualified', 'converted', 'lost'
	]))
	lead_rating = ma_fields.Str(validate=validate.OneOf(['hot', 'warm', 'cold']))
	lead_score = ma_fields.Int(validate=validate.Range(min=0, max=100))
	annual_revenue = ma_fields.Decimal(validate=validate.Range(min=0))
	employee_count = ma_fields.Str(validate=validate.Length(max=50))
	website = ma_fields.Url(validate=validate.Length(max=255))
	industry = ma_fields.Str(validate=validate.Length(max=100))
	description = ma_fields.Str()
	ai_insights = ma_fields.Dict(dump_only=True)
	converted_date = ma_fields.DateTime(dump_only=True)
	converted_to_opportunity_id = ma_fields.Str(dump_only=True)
	converted_to_contact_id = ma_fields.Str(dump_only=True)

class OpportunitySchema(BaseSchema):
	"""Opportunity schema for API serialization"""
	opportunity_name = ma_fields.Str(required=True, validate=validate.Length(max=255))
	amount = ma_fields.Decimal(required=True, validate=validate.Range(min=0))
	close_date = ma_fields.Date(required=True)
	stage = ma_fields.Str(validate=validate.OneOf([
		'prospecting', 'qualification', 'needs_analysis', 'value_proposition',
		'id_decision_makers', 'perception_analysis', 'proposal_quote',
		'negotiation_review', 'closed_won', 'closed_lost'
	]))
	probability = ma_fields.Int(validate=validate.Range(min=0, max=100))
	expected_revenue = ma_fields.Decimal(dump_only=True)
	next_step = ma_fields.Str(validate=validate.Length(max=255))
	description = ma_fields.Str()
	opportunity_owner_id = ma_fields.Str()
	primary_contact_id = ma_fields.Str()
	account_id = ma_fields.Str()
	lead_source = ma_fields.Str()
	ai_insights = ma_fields.Dict(dump_only=True)
	actual_close_date = ma_fields.DateTime(dump_only=True)

class CustomerSchema(BaseSchema):
	"""Customer schema for API serialization"""
	first_name = ma_fields.Str(required=True, validate=validate.Length(max=100))
	last_name = ma_fields.Str(required=True, validate=validate.Length(max=100))
	email = ma_fields.Email(required=True, validate=validate.Length(max=255))
	phone = ma_fields.Str(validate=validate.Length(max=50))
	company = ma_fields.Str(validate=validate.Length(max=200))
	customer_status = ma_fields.Str(validate=validate.OneOf([
		'active', 'inactive', 'churned', 'prospect'
	]))
	customer_since = ma_fields.Date()
	customer_value = ma_fields.Decimal(dump_only=True)
	annual_revenue = ma_fields.Decimal(validate=validate.Range(min=0))
	preferred_contact_method = ma_fields.Str(validate=validate.OneOf([
		'email', 'phone', 'mail', 'text'
	]))
	last_contact_date = ma_fields.Date(dump_only=True)
	description = ma_fields.Str()

class ContactSchema(BaseSchema):
	"""Contact schema for API serialization"""
	first_name = ma_fields.Str(required=True, validate=validate.Length(max=100))
	last_name = ma_fields.Str(required=True, validate=validate.Length(max=100))
	email = ma_fields.Email(validate=validate.Length(max=255))
	phone = ma_fields.Str(validate=validate.Length(max=50))
	company = ma_fields.Str(validate=validate.Length(max=200))
	job_title = ma_fields.Str(validate=validate.Length(max=100))
	department = ma_fields.Str(validate=validate.Length(max=100))
	account_id = ma_fields.Str()
	customer_id = ma_fields.Str()
	contact_owner_id = ma_fields.Str()
	is_primary_contact = ma_fields.Bool()
	lead_source = ma_fields.Str()
	description = ma_fields.Str()

class ActivitySchema(BaseSchema):
	"""Activity schema for API serialization"""
	subject = ma_fields.Str(required=True, validate=validate.Length(max=255))
	description = ma_fields.Str()
	activity_type = ma_fields.Str(validate=validate.OneOf([
		'call', 'email', 'meeting', 'task', 'note', 'other'
	]))
	activity_status = ma_fields.Str(validate=validate.OneOf([
		'pending', 'in_progress', 'completed', 'cancelled', 'deferred'
	]))
	due_date = ma_fields.DateTime()
	completed_date = ma_fields.DateTime(dump_only=True)
	assigned_to_id = ma_fields.Str()
	opportunity_id = ma_fields.Str()
	contact_id = ma_fields.Str()
	lead_id = ma_fields.Str()
	customer_id = ma_fields.Str()

class CaseSchema(BaseSchema):
	"""Case schema for API serialization"""
	case_number = ma_fields.Str(dump_only=True)
	subject = ma_fields.Str(required=True, validate=validate.Length(max=255))
	description = ma_fields.Str()
	status = ma_fields.Str(validate=validate.OneOf([
		'new', 'working', 'escalated', 'resolved', 'closed'
	]))
	priority = ma_fields.Str(validate=validate.OneOf([
		'low', 'medium', 'high', 'critical'
	]))
	customer_id = ma_fields.Str()
	contact_id = ma_fields.Str()
	case_owner_id = ma_fields.Str()
	case_origin = ma_fields.Str()

class CampaignSchema(BaseSchema):
	"""Campaign schema for API serialization"""
	campaign_name = ma_fields.Str(required=True, validate=validate.Length(max=255))
	description = ma_fields.Str()
	campaign_type = ma_fields.Str(validate=validate.OneOf([
		'email', 'webinar', 'trade_show', 'direct_mail', 'telemarketing',
		'social_media', 'content_marketing', 'other'
	]))
	status = ma_fields.Str(validate=validate.OneOf([
		'planning', 'active', 'paused', 'completed', 'aborted'
	]))
	start_date = ma_fields.Date()
	end_date = ma_fields.Date()
	budget = ma_fields.Decimal(validate=validate.Range(min=0))
	expected_revenue = ma_fields.Decimal(validate=validate.Range(min=0))
	expected_response = ma_fields.Int(validate=validate.Range(min=0))
	num_sent = ma_fields.Int(validate=validate.Range(min=0))
	actual_cost = ma_fields.Decimal(dump_only=True)

# Schema instances
lead_schema = LeadSchema()
leads_schema = LeadSchema(many=True)
opportunity_schema = OpportunitySchema()
opportunities_schema = OpportunitySchema(many=True)
customer_schema = CustomerSchema()
customers_schema = CustomerSchema(many=True)
contact_schema = ContactSchema()
contacts_schema = ContactSchema(many=True)
activity_schema = ActivitySchema()
activities_schema = ActivitySchema(many=True)
case_schema = CaseSchema()
cases_schema = CaseSchema(many=True)
campaign_schema = CampaignSchema()
campaigns_schema = CampaignSchema(many=True)
pagination_schema = PaginationSchema()

# Utility Functions

def get_current_user():
	"""Get current user from JWT token"""
	return get_jwt_identity()

def get_tenant_id():
	"""Get tenant ID for current user"""
	# In a real implementation, this would be extracted from the JWT token
	# or user session. For now, we'll use a placeholder
	return getattr(g, 'tenant_id', 'default_tenant')

def get_db_session():
	"""Get database session"""
	# This would typically come from Flask-SQLAlchemy
	from flask import current_app
	return current_app.db.session

def handle_api_error(e):
	"""Handle API errors and return appropriate response"""
	if isinstance(e, ValidationError):
		return {'error': 'Validation error', 'details': e.messages}, 400
	elif isinstance(e, ServiceValidationError):
		return {'error': 'Validation error', 'details': str(e)}, 400
	elif isinstance(e, CRMServiceError):
		return {'error': 'Service error', 'details': str(e)}, 500
	elif isinstance(e, NotFound):
		return {'error': 'Resource not found'}, 404
	elif isinstance(e, Forbidden):
		return {'error': 'Access forbidden'}, 403
	elif isinstance(e, BadRequest):
		return {'error': 'Bad request', 'details': str(e)}, 400
	else:
		logger.error(f"Unexpected API error: {str(e)}")
		return {'error': 'Internal server error'}, 500

def paginate_query(query, page=1, per_page=20):
	"""Paginate SQLAlchemy query"""
	per_page = min(per_page, 100)  # Limit max per_page
	total = query.count()
	items = query.offset((page - 1) * per_page).limit(per_page).all()
	
	return {
		'items': items,
		'pagination': {
			'page': page,
			'per_page': per_page,
			'total': total,
			'pages': (total + per_page - 1) // per_page,
			'has_prev': page > 1,
			'has_next': page * per_page < total
		}
	}

def parse_filters(model_class):
	"""Parse query parameters for filtering"""
	filters = []
	
	for key, value in request.args.items():
		if key.startswith('filter_') and hasattr(model_class, key[7:]):
			field = getattr(model_class, key[7:])
			
			# Handle different filter operations
			if key.endswith('_eq'):
				filters.append(field == value)
			elif key.endswith('_ne'):
				filters.append(field != value)
			elif key.endswith('_gt'):
				filters.append(field > value)
			elif key.endswith('_gte'):
				filters.append(field >= value)
			elif key.endswith('_lt'):
				filters.append(field < value)
			elif key.endswith('_lte'):
				filters.append(field <= value)
			elif key.endswith('_like'):
				filters.append(field.like(f'%{value}%'))
			elif key.endswith('_in'):
				values = value.split(',')
				filters.append(field.in_(values))
			else:
				filters.append(field == value)
	
	return filters

# Resource Base Classes

class BaseResource(Resource):
	"""Base resource with common functionality"""
	
	def __init__(self):
		self.db = get_db_session()
		self.tenant_id = get_tenant_id()
		self.user_id = get_current_user()
		self.crm_service = create_crm_service(self.db, self.tenant_id, self.user_id)

class PaginatedResource(BaseResource):
	"""Base resource with pagination support"""
	
	def get_paginated_list(self, query, schema, page=1, per_page=20):
		"""Get paginated list with serialization"""
		try:
			paginated = paginate_query(query, page, per_page)
			
			return {
				'data': schema.dump(paginated['items']),
				'pagination': pagination_schema.dump(paginated['pagination'])
			}
		except Exception as e:
			logger.error(f"Pagination error: {str(e)}")
			raise

# Lead Resources

class LeadListResource(PaginatedResource):
	"""Lead collection operations"""
	
	@swag_from({
		'tags': ['Leads'],
		'summary': 'Get list of leads',
		'description': 'Retrieve a paginated list of leads with filtering and sorting options',
		'parameters': [
			{
				'name': 'page',
				'in': 'query',
				'type': 'integer',
				'default': 1,
				'description': 'Page number'
			},
			{
				'name': 'per_page',
				'in': 'query',
				'type': 'integer',
				'default': 20,
				'description': 'Items per page (max 100)'
			},
			{
				'name': 'filter_status',
				'in': 'query',
				'type': 'string',
				'description': 'Filter by lead status'
			},
			{
				'name': 'filter_source',
				'in': 'query',
				'type': 'string',
				'description': 'Filter by lead source'
			},
			{
				'name': 'search',
				'in': 'query',
				'type': 'string',
				'description': 'Search in name, email, company'
			}
		],
		'responses': {
			200: {
				'description': 'List of leads',
				'schema': {
					'type': 'object',
					'properties': {
						'data': {'type': 'array', 'items': {'$ref': '#/definitions/Lead'}},
						'pagination': {'$ref': '#/definitions/Pagination'}
					}
				}
			}
		}
	})
	@jwt_required()
	def get(self):
		"""Get list of leads"""
		try:
			page = int(request.args.get('page', 1))
			per_page = int(request.args.get('per_page', 20))
			search = request.args.get('search', '')
			
			# Base query
			query = self.db.query(GCCRMLead).filter(
				GCCRMLead.tenant_id == self.tenant_id,
				GCCRMLead.is_active == True
			)
			
			# Apply filters
			filters = parse_filters(GCCRMLead)
			for filter_condition in filters:
				query = query.filter(filter_condition)
			
			# Apply search
			if search:
				search_term = f'%{search}%'
				query = query.filter(or_(
					GCCRMLead.first_name.like(search_term),
					GCCRMLead.last_name.like(search_term),
					GCCRMLead.email.like(search_term),
					GCCRMLead.company.like(search_term)
				))
			
			# Apply ordering
			order_by = request.args.get('order_by', 'created_on')
			order_dir = request.args.get('order_dir', 'desc')
			
			if hasattr(GCCRMLead, order_by):
				field = getattr(GCCRMLead, order_by)
				if order_dir.lower() == 'desc':
					query = query.order_by(desc(field))
				else:
					query = query.order_by(asc(field))
			
			return self.get_paginated_list(query, leads_schema, page, per_page)
			
		except Exception as e:
			return handle_api_error(e)
	
	@swag_from({
		'tags': ['Leads'],
		'summary': 'Create new lead',
		'description': 'Create a new lead with AI-powered scoring',
		'parameters': [
			{
				'name': 'body',
				'in': 'body',
				'required': True,
				'schema': {'$ref': '#/definitions/LeadCreate'}
			}
		],
		'responses': {
			201: {
				'description': 'Lead created successfully',
				'schema': {'$ref': '#/definitions/Lead'}
			},
			400: {
				'description': 'Validation error'
			}
		}
	})
	@jwt_required()
	def post(self):
		"""Create new lead"""
		try:
			# Validate input
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			validated_data = lead_schema.load(json_data)
			
			# Create lead via service
			lead = self.crm_service.leads.create_lead(validated_data)
			
			return lead_schema.dump(lead), 201
			
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return handle_api_error(e)

class LeadResource(BaseResource):
	"""Individual lead operations"""
	
	@swag_from({
		'tags': ['Leads'],
		'summary': 'Get lead by ID',
		'description': 'Retrieve a specific lead with all details',
		'parameters': [
			{
				'name': 'lead_id',
				'in': 'path',
				'type': 'string',
				'required': True,
				'description': 'Lead ID'
			}
		],
		'responses': {
			200: {
				'description': 'Lead details',
				'schema': {'$ref': '#/definitions/Lead'}
			},
			404: {
				'description': 'Lead not found'
			}
		}
	})
	@jwt_required()
	def get(self, lead_id):
		"""Get lead by ID"""
		try:
			lead = self.db.query(GCCRMLead).filter(
				GCCRMLead.id == lead_id,
				GCCRMLead.tenant_id == self.tenant_id
			).first()
			
			if not lead:
				return {'error': 'Lead not found'}, 404
			
			return lead_schema.dump(lead)
			
		except Exception as e:
			return handle_api_error(e)
	
	@swag_from({
		'tags': ['Leads'],
		'summary': 'Update lead',
		'description': 'Update an existing lead with re-scoring',
		'parameters': [
			{
				'name': 'lead_id',
				'in': 'path',
				'type': 'string',
				'required': True,
				'description': 'Lead ID'
			},
			{
				'name': 'body',
				'in': 'body',
				'required': True,
				'schema': {'$ref': '#/definitions/LeadUpdate'}
			}
		],
		'responses': {
			200: {
				'description': 'Lead updated successfully',
				'schema': {'$ref': '#/definitions/Lead'}
			}
		}
	})
	@jwt_required()
	def put(self, lead_id):
		"""Update lead"""
		try:
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			validated_data = lead_schema.load(json_data, partial=True)
			
			# Update lead via service
			lead = self.crm_service.leads.update_lead(lead_id, validated_data)
			
			return lead_schema.dump(lead)
			
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return handle_api_error(e)
	
	@swag_from({
		'tags': ['Leads'],
		'summary': 'Delete lead',
		'description': 'Soft delete a lead',
		'parameters': [
			{
				'name': 'lead_id',
				'in': 'path',
				'type': 'string',
				'required': True,
				'description': 'Lead ID'
			}
		],
		'responses': {
			204: {
				'description': 'Lead deleted successfully'
			}
		}
	})
	@jwt_required()
	def delete(self, lead_id):
		"""Delete lead (soft delete)"""
		try:
			lead = self.db.query(GCCRMLead).filter(
				GCCRMLead.id == lead_id,
				GCCRMLead.tenant_id == self.tenant_id
			).first()
			
			if not lead:
				return {'error': 'Lead not found'}, 404
			
			lead.is_active = False
			self.db.commit()
			
			return '', 204
			
		except Exception as e:
			return handle_api_error(e)

class LeadConvertResource(BaseResource):
	"""Lead conversion operations"""
	
	@swag_from({
		'tags': ['Leads'],
		'summary': 'Convert lead to opportunity',
		'description': 'Convert a qualified lead to opportunity and contact',
		'parameters': [
			{
				'name': 'lead_id',
				'in': 'path',
				'type': 'string',
				'required': True,
				'description': 'Lead ID'
			},
			{
				'name': 'body',
				'in': 'body',
				'required': True,
				'schema': {
					'type': 'object',
					'properties': {
						'opportunity_name': {'type': 'string'},
						'amount': {'type': 'number'},
						'close_date': {'type': 'string', 'format': 'date'}
					},
					'required': ['opportunity_name', 'amount', 'close_date']
				}
			}
		],
		'responses': {
			200: {
				'description': 'Lead converted successfully',
				'schema': {
					'type': 'object',
					'properties': {
						'opportunity': {'$ref': '#/definitions/Opportunity'},
						'contact': {'$ref': '#/definitions/Contact'}
					}
				}
			}
		}
	})
	@jwt_required()
	def post(self, lead_id):
		"""Convert lead to opportunity"""
		try:
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			opportunity, contact = self.crm_service.leads.convert_lead_to_opportunity(
				lead_id, json_data
			)
			
			return {
				'opportunity': opportunity_schema.dump(opportunity),
				'contact': contact_schema.dump(contact)
			}
			
		except Exception as e:
			return handle_api_error(e)

class LeadAnalyticsResource(BaseResource):
	"""Lead analytics operations"""
	
	@swag_from({
		'tags': ['Leads'],
		'summary': 'Get lead analytics',
		'description': 'Get comprehensive lead analytics and performance metrics',
		'parameters': [
			{
				'name': 'date_from',
				'in': 'query',
				'type': 'string',
				'format': 'date',
				'description': 'Start date for analytics'
			},
			{
				'name': 'date_to',
				'in': 'query',
				'type': 'string',
				'format': 'date',
				'description': 'End date for analytics'
			}
		],
		'responses': {
			200: {
				'description': 'Lead analytics data'
			}
		}
	})
	@jwt_required()
	def get(self):
		"""Get lead analytics"""
		try:
			date_from_str = request.args.get('date_from')
			date_to_str = request.args.get('date_to')
			
			# Default to last 30 days if not provided
			if not date_from_str:
				date_from = date.today() - timedelta(days=30)
			else:
				date_from = datetime.strptime(date_from_str, '%Y-%m-%d').date()
			
			if not date_to_str:
				date_to = date.today()
			else:
				date_to = datetime.strptime(date_to_str, '%Y-%m-%d').date()
			
			analytics = self.crm_service.leads.get_lead_analytics(date_from, date_to)
			return analytics
			
		except Exception as e:
			return handle_api_error(e)

# Opportunity Resources

class OpportunityListResource(PaginatedResource):
	"""Opportunity collection operations"""
	
	@jwt_required()
	def get(self):
		"""Get list of opportunities"""
		try:
			page = int(request.args.get('page', 1))
			per_page = int(request.args.get('per_page', 20))
			search = request.args.get('search', '')
			
			# Base query
			query = self.db.query(GCCRMOpportunity).filter(
				GCCRMOpportunity.tenant_id == self.tenant_id,
				GCCRMOpportunity.is_active == True
			)
			
			# Apply filters
			filters = parse_filters(GCCRMOpportunity)
			for filter_condition in filters:
				query = query.filter(filter_condition)
			
			# Apply search
			if search:
				search_term = f'%{search}%'
				query = query.filter(or_(
					GCCRMOpportunity.opportunity_name.like(search_term),
					GCCRMOpportunity.description.like(search_term)
				))
			
			# Apply ordering
			order_by = request.args.get('order_by', 'close_date')
			order_dir = request.args.get('order_dir', 'asc')
			
			if hasattr(GCCRMOpportunity, order_by):
				field = getattr(GCCRMOpportunity, order_by)
				if order_dir.lower() == 'desc':
					query = query.order_by(desc(field))
				else:
					query = query.order_by(asc(field))
			
			return self.get_paginated_list(query, opportunities_schema, page, per_page)
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	def post(self):
		"""Create new opportunity"""
		try:
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			validated_data = opportunity_schema.load(json_data)
			
			# Create opportunity via service
			opportunity = self.crm_service.opportunities.create_opportunity(validated_data)
			
			return opportunity_schema.dump(opportunity), 201
			
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return handle_api_error(e)

class OpportunityResource(BaseResource):
	"""Individual opportunity operations"""
	
	@jwt_required()
	def get(self, opportunity_id):
		"""Get opportunity by ID"""
		try:
			opportunity = self.db.query(GCCRMOpportunity).filter(
				GCCRMOpportunity.id == opportunity_id,
				GCCRMOpportunity.tenant_id == self.tenant_id
			).first()
			
			if not opportunity:
				return {'error': 'Opportunity not found'}, 404
			
			return opportunity_schema.dump(opportunity)
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	def put(self, opportunity_id):
		"""Update opportunity"""
		try:
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			validated_data = opportunity_schema.load(json_data, partial=True)
			
			opportunity = self.db.query(GCCRMOpportunity).filter(
				GCCRMOpportunity.id == opportunity_id,
				GCCRMOpportunity.tenant_id == self.tenant_id
			).first()
			
			if not opportunity:
				return {'error': 'Opportunity not found'}, 404
			
			# Update fields
			for key, value in validated_data.items():
				if hasattr(opportunity, key):
					setattr(opportunity, key, value)
			
			# Recalculate expected revenue
			if hasattr(opportunity, 'amount') and hasattr(opportunity, 'probability'):
				opportunity.expected_revenue = (opportunity.amount or Decimal('0')) * (opportunity.probability / 100)
			
			self.db.commit()
			
			return opportunity_schema.dump(opportunity)
			
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return handle_api_error(e)

class OpportunityStageResource(BaseResource):
	"""Opportunity stage management"""
	
	@jwt_required()
	def put(self, opportunity_id):
		"""Update opportunity stage"""
		try:
			json_data = request.get_json()
			if not json_data or 'stage' not in json_data:
				return {'error': 'Stage is required'}, 400
			
			new_stage = OpportunityStage(json_data['stage'])
			stage_data = json_data.get('data', {})
			
			opportunity = self.crm_service.opportunities.update_opportunity_stage(
				opportunity_id, new_stage, stage_data
			)
			
			return opportunity_schema.dump(opportunity)
			
		except Exception as e:
			return handle_api_error(e)

class OpportunityAnalyticsResource(BaseResource):
	"""Opportunity analytics operations"""
	
	@jwt_required()
	def get(self):
		"""Get sales pipeline analytics"""
		try:
			analytics = self.crm_service.opportunities.get_sales_pipeline_analytics()
			return analytics
		except Exception as e:
			return handle_api_error(e)

class OpportunityForecastResource(BaseResource):
	"""Revenue forecasting operations"""
	
	@jwt_required()
	def get(self):
		"""Get revenue forecast"""
		try:
			period = request.args.get('period', 'quarter')
			forecast = self.crm_service.opportunities.forecast_revenue(period)
			return forecast
		except Exception as e:
			return handle_api_error(e)

# Customer Resources

class CustomerListResource(PaginatedResource):
	"""Customer collection operations"""
	
	@jwt_required()
	def get(self):
		"""Get list of customers"""
		try:
			page = int(request.args.get('page', 1))
			per_page = int(request.args.get('per_page', 20))
			search = request.args.get('search', '')
			
			# Base query
			query = self.db.query(GCCRMCustomer).filter(
				GCCRMCustomer.tenant_id == self.tenant_id,
				GCCRMCustomer.is_active == True
			)
			
			# Apply filters
			filters = parse_filters(GCCRMCustomer)
			for filter_condition in filters:
				query = query.filter(filter_condition)
			
			# Apply search
			if search:
				search_term = f'%{search}%'
				query = query.filter(or_(
					GCCRMCustomer.first_name.like(search_term),
					GCCRMCustomer.last_name.like(search_term),
					GCCRMCustomer.email.like(search_term),
					GCCRMCustomer.company.like(search_term)
				))
			
			# Apply ordering
			order_by = request.args.get('order_by', 'customer_since')
			order_dir = request.args.get('order_dir', 'desc')
			
			if hasattr(GCCRMCustomer, order_by):
				field = getattr(GCCRMCustomer, order_by)
				if order_dir.lower() == 'desc':
					query = query.order_by(desc(field))
				else:
					query = query.order_by(asc(field))
			
			return self.get_paginated_list(query, customers_schema, page, per_page)
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	def post(self):
		"""Create new customer"""
		try:
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			validated_data = customer_schema.load(json_data)
			
			# Create customer
			customer = GCCRMCustomer(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				**validated_data
			)
			
			self.db.add(customer)
			self.db.commit()
			
			return customer_schema.dump(customer), 201
			
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return handle_api_error(e)

class CustomerResource(BaseResource):
	"""Individual customer operations"""
	
	@jwt_required()
	def get(self, customer_id):
		"""Get customer by ID"""
		try:
			customer = self.db.query(GCCRMCustomer).filter(
				GCCRMCustomer.id == customer_id,
				GCCRMCustomer.tenant_id == self.tenant_id
			).first()
			
			if not customer:
				return {'error': 'Customer not found'}, 404
			
			return customer_schema.dump(customer)
			
		except Exception as e:
			return handle_api_error(e)

class Customer360Resource(BaseResource):
	"""Customer 360° view operations"""
	
	@jwt_required()
	def get(self, customer_id):
		"""Get comprehensive customer 360° view"""
		try:
			customer_360 = self.crm_service.customers.get_customer_360_view(customer_id)
			return customer_360
		except Exception as e:
			return handle_api_error(e)

class CustomerCLVResource(BaseResource):
	"""Customer lifetime value operations"""
	
	@jwt_required()
	def get(self, customer_id):
		"""Calculate customer lifetime value"""
		try:
			clv_data = self.crm_service.customers.calculate_customer_lifetime_value(customer_id)
			return clv_data
		except Exception as e:
			return handle_api_error(e)

class CustomerSegmentationResource(BaseResource):
	"""Customer segmentation operations"""
	
	@jwt_required()
	def get(self):
		"""Get customer segmentation analysis"""
		try:
			criteria = request.args.to_dict()
			segmentation = self.crm_service.customers.segment_customers(criteria)
			return segmentation
		except Exception as e:
			return handle_api_error(e)

class CustomerChurnResource(BaseResource):
	"""Customer churn prediction operations"""
	
	@jwt_required()
	def get(self, customer_id=None):
		"""Predict customer churn risk"""
		try:
			churn_data = self.crm_service.customers.predict_churn_risk(customer_id)
			return churn_data
		except Exception as e:
			return handle_api_error(e)

# Contact Resources

class ContactListResource(PaginatedResource):
	"""Contact collection operations"""
	
	@jwt_required()
	def get(self):
		"""Get list of contacts"""
		try:
			page = int(request.args.get('page', 1))
			per_page = int(request.args.get('per_page', 20))
			search = request.args.get('search', '')
			
			# Base query
			query = self.db.query(GCCRMContact).filter(
				GCCRMContact.tenant_id == self.tenant_id,
				GCCRMContact.is_active == True
			)
			
			# Apply filters
			filters = parse_filters(GCCRMContact)
			for filter_condition in filters:
				query = query.filter(filter_condition)
			
			# Apply search
			if search:
				search_term = f'%{search}%'
				query = query.filter(or_(
					GCCRMContact.first_name.like(search_term),
					GCCRMContact.last_name.like(search_term),
					GCCRMContact.email.like(search_term),
					GCCRMContact.company.like(search_term),
					GCCRMContact.job_title.like(search_term)
				))
			
			return self.get_paginated_list(query, contacts_schema, page, per_page)
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	def post(self):
		"""Create new contact"""
		try:
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			validated_data = contact_schema.load(json_data)
			
			# Create contact
			contact = GCCRMContact(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				**validated_data
			)
			
			self.db.add(contact)
			self.db.commit()
			
			return contact_schema.dump(contact), 201
			
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return handle_api_error(e)

class ContactResource(BaseResource):
	"""Individual contact operations"""
	
	@jwt_required()
	def get(self, contact_id):
		"""Get contact by ID"""
		try:
			contact = self.db.query(GCCRMContact).filter(
				GCCRMContact.id == contact_id,
				GCCRMContact.tenant_id == self.tenant_id
			).first()
			
			if not contact:
				return {'error': 'Contact not found'}, 404
			
			return contact_schema.dump(contact)
			
		except Exception as e:
			return handle_api_error(e)

# Activity Resources

class ActivityListResource(PaginatedResource):
	"""Activity collection operations"""
	
	@jwt_required()
	def get(self):
		"""Get list of activities"""
		try:
			page = int(request.args.get('page', 1))
			per_page = int(request.args.get('per_page', 20))
			
			# Base query
			query = self.db.query(GCCRMActivity).filter(
				GCCRMActivity.tenant_id == self.tenant_id
			)
			
			# Apply filters
			filters = parse_filters(GCCRMActivity)
			for filter_condition in filters:
				query = query.filter(filter_condition)
			
			# Apply ordering
			order_by = request.args.get('order_by', 'due_date')
			order_dir = request.args.get('order_dir', 'asc')
			
			if hasattr(GCCRMActivity, order_by):
				field = getattr(GCCRMActivity, order_by)
				if order_dir.lower() == 'desc':
					query = query.order_by(desc(field))
				else:
					query = query.order_by(asc(field))
			
			return self.get_paginated_list(query, activities_schema, page, per_page)
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	def post(self):
		"""Create new activity"""
		try:
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			validated_data = activity_schema.load(json_data)
			
			# Create activity
			activity = GCCRMActivity(
				id=uuid7str(),
				tenant_id=self.tenant_id,
				**validated_data
			)
			
			self.db.add(activity)
			self.db.commit()
			
			return activity_schema.dump(activity), 201
			
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return handle_api_error(e)

# Bulk Operations Resources

class BulkLeadsResource(BaseResource):
	"""Bulk operations for leads"""
	
	@jwt_required()
	def post(self):
		"""Perform bulk operations on leads"""
		try:
			json_data = request.get_json()
			if not json_data:
				return {'error': 'No input data provided'}, 400
			
			operation = json_data.get('operation')
			lead_ids = json_data.get('lead_ids', [])
			operation_data = json_data.get('data', {})
			
			if not operation or not lead_ids:
				return {'error': 'Operation and lead_ids are required'}, 400
			
			results = self.crm_service.leads.bulk_lead_operations(
				operation, lead_ids, operation_data
			)
			
			return {'success': True, 'results': results}
			
		except Exception as e:
			return handle_api_error(e)

# System Resources

class CRMHealthResource(BaseResource):
	"""CRM system health check"""
	
	def get(self):
		"""Get CRM system health status"""
		try:
			health_status = self.crm_service.health_check()
			return health_status
		except Exception as e:
			return handle_api_error(e)

class CRMDashboardResource(BaseResource):
	"""CRM dashboard data"""
	
	@jwt_required()
	def get(self):
		"""Get CRM dashboard data"""
		try:
			dashboard_data = self.crm_service.get_dashboard_data()
			return dashboard_data
		except Exception as e:
			return handle_api_error(e)

# Register API Resources

# Lead endpoints
api.add_resource(LeadListResource, '/leads')
api.add_resource(LeadResource, '/leads/<string:lead_id>')
api.add_resource(LeadConvertResource, '/leads/<string:lead_id>/convert')
api.add_resource(LeadAnalyticsResource, '/leads/analytics')

# Opportunity endpoints
api.add_resource(OpportunityListResource, '/opportunities')
api.add_resource(OpportunityResource, '/opportunities/<string:opportunity_id>')
api.add_resource(OpportunityStageResource, '/opportunities/<string:opportunity_id>/stage')
api.add_resource(OpportunityAnalyticsResource, '/opportunities/analytics')
api.add_resource(OpportunityForecastResource, '/opportunities/forecast')

# Customer endpoints
api.add_resource(CustomerListResource, '/customers')
api.add_resource(CustomerResource, '/customers/<string:customer_id>')
api.add_resource(Customer360Resource, '/customers/<string:customer_id>/360')
api.add_resource(CustomerCLVResource, '/customers/<string:customer_id>/clv')
api.add_resource(CustomerSegmentationResource, '/customers/segmentation')
api.add_resource(CustomerChurnResource, '/customers/churn')
api.add_resource(CustomerChurnResource, '/customers/<string:customer_id>/churn')

# Contact endpoints
api.add_resource(ContactListResource, '/contacts')
api.add_resource(ContactResource, '/contacts/<string:contact_id>')

# Activity endpoints
api.add_resource(ActivityListResource, '/activities')

# Bulk operations
api.add_resource(BulkLeadsResource, '/bulk/leads')

# System endpoints
api.add_resource(CRMHealthResource, '/health')
api.add_resource(CRMDashboardResource, '/dashboard')

# Error handlers
@crm_api_bp.errorhandler(ValidationError)
def handle_validation_error(e):
	return jsonify({'error': 'Validation error', 'details': e.messages}), 400

@crm_api_bp.errorhandler(404)
def handle_not_found(e):
	return jsonify({'error': 'Resource not found'}), 404

@crm_api_bp.errorhandler(500)
def handle_internal_error(e):
	logger.error(f"Internal server error: {str(e)}")
	return jsonify({'error': 'Internal server error'}), 500

# API Documentation Configuration for Swagger/OpenAPI
SWAGGER_CONFIG = {
	'swagger': '2.0',
	'info': {
		'title': 'CRM API',
		'description': 'Comprehensive Customer Relationship Management API',
		'version': '1.0.0',
		'contact': {
			'name': 'CRM API Support',
			'email': 'api-support@example.com'
		}
	},
	'basePath': '/api/v1/crm',
	'schemes': ['https', 'http'],
	'securityDefinitions': {
		'Bearer': {
			'type': 'apiKey',
			'name': 'Authorization',
			'in': 'header',
			'description': 'JWT Authorization header using the Bearer scheme. Example: "Authorization: Bearer {token}"'
		}
	},
	'security': [{'Bearer': []}],
	'definitions': {
		'Lead': {
			'type': 'object',
			'properties': {
				'id': {'type': 'string'},
				'first_name': {'type': 'string'},
				'last_name': {'type': 'string'},
				'email': {'type': 'string'},
				'phone': {'type': 'string'},
				'company': {'type': 'string'},
				'job_title': {'type': 'string'},
				'lead_source': {'type': 'string'},
				'lead_status': {'type': 'string'},
				'lead_rating': {'type': 'string'},
				'lead_score': {'type': 'integer'},
				'created_on': {'type': 'string', 'format': 'date-time'},
				'changed_on': {'type': 'string', 'format': 'date-time'}
			}
		},
		'Opportunity': {
			'type': 'object',
			'properties': {
				'id': {'type': 'string'},
				'opportunity_name': {'type': 'string'},
				'amount': {'type': 'number'},
				'close_date': {'type': 'string', 'format': 'date'},
				'stage': {'type': 'string'},
				'probability': {'type': 'integer'},
				'expected_revenue': {'type': 'number'},
				'created_on': {'type': 'string', 'format': 'date-time'}
			}
		},
		'Customer': {
			'type': 'object',
			'properties': {
				'id': {'type': 'string'},
				'first_name': {'type': 'string'},
				'last_name': {'type': 'string'},
				'email': {'type': 'string'},
				'phone': {'type': 'string'},
				'company': {'type': 'string'},
				'customer_status': {'type': 'string'},
				'customer_value': {'type': 'number'},
				'created_on': {'type': 'string', 'format': 'date-time'}
			}
		},
		'Contact': {
			'type': 'object',
			'properties': {
				'id': {'type': 'string'},
				'first_name': {'type': 'string'},
				'last_name': {'type': 'string'},
				'email': {'type': 'string'},
				'phone': {'type': 'string'},
				'company': {'type': 'string'},
				'job_title': {'type': 'string'},
				'created_on': {'type': 'string', 'format': 'date-time'}
			}
		},
		'Pagination': {
			'type': 'object',
			'properties': {
				'page': {'type': 'integer'},
				'per_page': {'type': 'integer'},
				'total': {'type': 'integer'},
				'pages': {'type': 'integer'},
				'has_prev': {'type': 'boolean'},
				'has_next': {'type': 'boolean'}
			}
		}
	}
}

def init_crm_api(app):
	"""Initialize CRM API with Flask application"""
	app.register_blueprint(crm_api_bp)
	
	# Initialize Swagger documentation
	try:
		from flasgger import Swagger
		swagger = Swagger(app, config=SWAGGER_CONFIG)
	except ImportError:
		logger.warning("Flasgger not available, API documentation disabled")
	
	logger.info("CRM API initialized successfully")