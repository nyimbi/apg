"""
APG Vendor Management - Comprehensive REST API
AI-powered vendor lifecycle management API with advanced analytics and intelligence

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from flask import Blueprint, request, jsonify, current_app
from flask_restful import Api, Resource, abort
from flask_appbuilder.security.decorators import has_access_api
from marshmallow import Schema, fields, ValidationError
from werkzeug.exceptions import BadRequest, NotFound, Unauthorized

from .models import (
	VMVendor, VMPerformance, VMRisk, VMIntelligence,
	VendorStatus, VendorType, RiskSeverity, StrategicImportance
)
from .service import VendorManagementService, VMDatabaseContext
from .intelligence_service import VendorIntelligenceEngine


# ============================================================================
# REQUEST/RESPONSE SCHEMAS FOR API VALIDATION
# ============================================================================

class VendorCreateSchema(Schema):
	"""Schema for vendor creation request"""
	
	vendor_code = fields.Str(required=True, validate=lambda x: len(x) <= 50)
	name = fields.Str(required=True, validate=lambda x: len(x) <= 200)
	legal_name = fields.Str(missing=None, validate=lambda x: len(x) <= 250)
	display_name = fields.Str(missing=None, validate=lambda x: len(x) <= 200)
	
	vendor_type = fields.Str(required=True, validate=lambda x: x in [e.value for e in VendorType])
	category = fields.Str(required=True, validate=lambda x: len(x) <= 100)
	subcategory = fields.Str(missing=None, validate=lambda x: len(x) <= 100)
	industry = fields.Str(missing=None, validate=lambda x: len(x) <= 100)
	
	email = fields.Email(missing=None)
	phone = fields.Str(missing=None, validate=lambda x: len(x) <= 50)
	website = fields.Url(missing=None)
	
	strategic_importance = fields.Str(
		missing='standard',
		validate=lambda x: x in [e.value for e in StrategicImportance]
	)
	preferred_vendor = fields.Bool(missing=False)
	strategic_partner = fields.Bool(missing=False)


class VendorUpdateSchema(Schema):
	"""Schema for vendor update request"""
	
	name = fields.Str(validate=lambda x: len(x) <= 200)
	legal_name = fields.Str(validate=lambda x: len(x) <= 250)
	display_name = fields.Str(validate=lambda x: len(x) <= 200)
	
	status = fields.Str(validate=lambda x: x in [e.value for e in VendorStatus])
	email = fields.Email()
	phone = fields.Str(validate=lambda x: len(x) <= 50)
	website = fields.Url()
	
	strategic_importance = fields.Str(
		validate=lambda x: x in [e.value for e in StrategicImportance]
	)
	preferred_vendor = fields.Bool()
	strategic_partner = fields.Bool()


class PerformanceRecordSchema(Schema):
	"""Schema for performance record creation"""
	
	measurement_period = fields.Str(
		required=True,
		validate=lambda x: x in ['monthly', 'quarterly', 'annual']
	)
	overall_score = fields.Float(required=True, validate=lambda x: 0 <= x <= 100)
	quality_score = fields.Float(required=True, validate=lambda x: 0 <= x <= 100)
	delivery_score = fields.Float(required=True, validate=lambda x: 0 <= x <= 100)
	cost_score = fields.Float(required=True, validate=lambda x: 0 <= x <= 100)
	service_score = fields.Float(required=True, validate=lambda x: 0 <= x <= 100)
	
	on_time_delivery_rate = fields.Float(validate=lambda x: 0 <= x <= 100)
	quality_rejection_rate = fields.Float(validate=lambda x: 0 <= x <= 100)


class RiskRecordSchema(Schema):
	"""Schema for risk record creation"""
	
	risk_type = fields.Str(required=True, validate=lambda x: len(x) <= 50)
	risk_category = fields.Str(required=True, validate=lambda x: len(x) <= 100)
	severity = fields.Str(
		required=True,
		validate=lambda x: x in [e.value for e in RiskSeverity]
	)
	
	title = fields.Str(required=True, validate=lambda x: len(x) <= 200)
	description = fields.Str(required=True)
	root_cause = fields.Str()
	potential_impact = fields.Str()
	
	overall_risk_score = fields.Float(required=True, validate=lambda x: 0 <= x <= 100)
	financial_impact = fields.Float()
	mitigation_strategy = fields.Str()


# ============================================================================
# BASE API RESOURCE WITH COMMON FUNCTIONALITY
# ============================================================================

class BaseVendorResource(Resource):
	"""Base resource class with common vendor management functionality"""
	
	def __init__(self):
		self.db_context = None
		self.vendor_service = None
		self.intelligence_engine = None
	
	def _get_tenant_id(self) -> UUID:
		"""Get current tenant ID from request headers/context"""
		# In production, extract from JWT token or session
		tenant_header = request.headers.get('X-Tenant-ID')
		if tenant_header:
			try:
				return UUID(tenant_header)
			except ValueError:
				abort(400, message="Invalid tenant ID format")
		
		# Default tenant for development
		return UUID('00000000-0000-0000-0000-000000000000')
	
	def _get_current_user_id(self) -> UUID:
		"""Get current user ID from request context"""
		# In production, extract from JWT token or session
		user_header = request.headers.get('X-User-ID')
		if user_header:
			try:
				return UUID(user_header)
			except ValueError:
				abort(400, message="Invalid user ID format")
		
		# Default user for development
		return UUID('00000000-0000-0000-0000-000000000000')
	
	async def _get_vendor_service(self) -> VendorManagementService:
		"""Get vendor service instance"""
		if not self.vendor_service:
			if not self.db_context:
				# In production, get from app config
				connection_string = current_app.config.get(
					'VENDOR_MANAGEMENT_DB_URL',
					'postgresql://localhost/apg'
				)
				self.db_context = VMDatabaseContext(connection_string)
			
			self.vendor_service = VendorManagementService(
				self._get_tenant_id(),
				self.db_context
			)
			self.vendor_service.set_current_user(self._get_current_user_id())
		
		return self.vendor_service
	
	async def _get_intelligence_engine(self) -> VendorIntelligenceEngine:
		"""Get intelligence engine instance"""
		if not self.intelligence_engine:
			if not self.db_context:
				connection_string = current_app.config.get(
					'VENDOR_MANAGEMENT_DB_URL',
					'postgresql://localhost/apg'
				)
				self.db_context = VMDatabaseContext(connection_string)
			
			self.intelligence_engine = VendorIntelligenceEngine(
				self._get_tenant_id(),
				self.db_context
			)
			self.intelligence_engine.set_current_user(self._get_current_user_id())
		
		return self.intelligence_engine
	
	def _run_async(self, coro):
		"""Run async coroutine in sync context"""
		try:
			loop = asyncio.get_event_loop()
		except RuntimeError:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
		
		return loop.run_until_complete(coro)
	
	def _handle_error(self, error: Exception, default_message: str = "An error occurred"):
		"""Standardized error handling"""
		current_app.logger.error(f"API Error: {str(error)}", exc_info=True)
		
		if isinstance(error, ValidationError):
			return {'success': False, 'error': 'Validation error', 'details': error.messages}, 400
		elif isinstance(error, ValueError):
			return {'success': False, 'error': str(error)}, 400
		elif isinstance(error, NotFound):
			return {'success': False, 'error': 'Resource not found'}, 404
		else:
			return {'success': False, 'error': default_message}, 500


# ============================================================================
# VENDOR CRUD API RESOURCES
# ============================================================================

class VendorListResource(BaseVendorResource):
	"""API resource for vendor listing and creation"""
	
	@has_access_api
	def get(self):
		"""List vendors with filtering, pagination, and search"""
		
		try:
			# Extract query parameters
			page = int(request.args.get('page', 1))
			page_size = min(int(request.args.get('page_size', 25)), 100)
			
			filters = {}
			if request.args.get('status'):
				filters['status'] = request.args.get('status')
			if request.args.get('category'):
				filters['category'] = request.args.get('category')
			if request.args.get('vendor_type'):
				filters['vendor_type'] = request.args.get('vendor_type')
			if request.args.get('strategic_importance'):
				filters['strategic_importance'] = request.args.get('strategic_importance')
			if request.args.get('search'):
				filters['search'] = request.args.get('search')
			
			sort_by = request.args.get('sort_by', 'name')
			sort_order = request.args.get('sort_order', 'asc')
			
			# Get vendors from service
			service = self._run_async(self._get_vendor_service())
			vendor_response = self._run_async(
				service.list_vendors(
					page=page,
					page_size=page_size,
					filters=filters,
					sort_by=sort_by,
					sort_order=sort_order
				)
			)
			
			# Format response
			vendors_data = []
			for vendor in vendor_response.vendors:
				vendors_data.append({
					'id': vendor.id,
					'vendor_code': vendor.vendor_code,
					'name': vendor.name,
					'legal_name': vendor.legal_name,
					'display_name': vendor.display_name,
					'vendor_type': vendor.vendor_type.value,
					'category': vendor.category,
					'subcategory': vendor.subcategory,
					'status': vendor.status.value,
					'strategic_importance': vendor.strategic_importance.value,
					'preferred_vendor': vendor.preferred_vendor,
					'strategic_partner': vendor.strategic_partner,
					'performance_score': float(vendor.performance_score),
					'risk_score': float(vendor.risk_score),
					'intelligence_score': float(vendor.intelligence_score),
					'relationship_score': float(vendor.relationship_score),
					'email': vendor.email,
					'phone': vendor.phone,
					'website': vendor.website,
					'created_at': vendor.created_at.isoformat(),
					'updated_at': vendor.updated_at.isoformat()
				})
			
			return {
				'success': True,
				'data': {
					'vendors': vendors_data,
					'pagination': {
						'page': vendor_response.page,
						'page_size': vendor_response.page_size,
						'total_count': vendor_response.total_count,
						'has_next': vendor_response.has_next,
						'has_prev': page > 1,
						'total_pages': (vendor_response.total_count + vendor_response.page_size - 1) // vendor_response.page_size
					}
				}
			}
			
		except Exception as e:
			return self._handle_error(e, "Error retrieving vendors")
	
	@has_access_api
	def post(self):
		"""Create new vendor"""
		
		try:
			# Validate request data
			schema = VendorCreateSchema()
			vendor_data = schema.load(request.get_json())
			
			# Create vendor
			service = self._run_async(self._get_vendor_service())
			vendor = self._run_async(service.create_vendor(vendor_data))
			
			return {
				'success': True,
				'data': {
					'id': vendor.id,
					'vendor_code': vendor.vendor_code,
					'name': vendor.name,
					'status': vendor.status.value,
					'created_at': vendor.created_at.isoformat()
				}
			}, 201
			
		except Exception as e:
			return self._handle_error(e, "Error creating vendor")


class VendorDetailResource(BaseVendorResource):
	"""API resource for individual vendor operations"""
	
	@has_access_api
	def get(self, vendor_id):
		"""Get vendor details with comprehensive information"""
		
		try:
			service = self._run_async(self._get_vendor_service())
			vendor = self._run_async(service.get_vendor_by_id(vendor_id))
			
			if not vendor:
				return {'success': False, 'error': 'Vendor not found'}, 404
			
			# Get additional vendor data
			performance_summary = self._run_async(
				service.get_vendor_performance_summary(vendor_id)
			)
			
			vendor_data = {
				'id': vendor.id,
				'tenant_id': str(vendor.tenant_id),
				'vendor_code': vendor.vendor_code,
				'name': vendor.name,
				'legal_name': vendor.legal_name,
				'display_name': vendor.display_name,
				'vendor_type': vendor.vendor_type.value,
				'category': vendor.category,
				'subcategory': vendor.subcategory,
				'industry': vendor.industry,
				'size_classification': vendor.size_classification.value,
				'status': vendor.status.value,
				'lifecycle_stage': vendor.lifecycle_stage.value,
				'strategic_importance': vendor.strategic_importance.value,
				'preferred_vendor': vendor.preferred_vendor,
				'strategic_partner': vendor.strategic_partner,
				'diversity_category': vendor.diversity_category,
				
				# Contact Information
				'email': vendor.email,
				'phone': vendor.phone,
				'website': vendor.website,
				
				# Address Information
				'address': {
					'address_line1': vendor.address_line1,
					'address_line2': vendor.address_line2,
					'city': vendor.city,
					'state_province': vendor.state_province,
					'postal_code': vendor.postal_code,
					'country': vendor.country
				},
				
				# Financial Information
				'financial': {
					'credit_rating': vendor.credit_rating,
					'payment_terms': vendor.payment_terms,
					'currency': vendor.currency,
					'tax_id': vendor.tax_id,
					'duns_number': vendor.duns_number
				},
				
				# AI Scores
				'scores': {
					'performance_score': float(vendor.performance_score),
					'risk_score': float(vendor.risk_score),
					'intelligence_score': float(vendor.intelligence_score),
					'relationship_score': float(vendor.relationship_score)
				},
				
				# Operational Details
				'capabilities': vendor.capabilities,
				'certifications': vendor.certifications,
				'geographic_coverage': vendor.geographic_coverage,
				'capacity_metrics': vendor.capacity_metrics,
				
				# AI Insights
				'predicted_performance': vendor.predicted_performance,
				'risk_predictions': vendor.risk_predictions,
				'optimization_recommendations': vendor.optimization_recommendations,
				'ai_insights': vendor.ai_insights,
				
				# Performance Summary
				'performance_summary': performance_summary.model_dump() if performance_summary else None,
				
				# Metadata
				'created_at': vendor.created_at.isoformat(),
				'updated_at': vendor.updated_at.isoformat(),
				'version': vendor.version
			}
			
			return {
				'success': True,
				'data': vendor_data
			}
			
		except Exception as e:
			return self._handle_error(e, "Error retrieving vendor details")
	
	@has_access_api
	def put(self, vendor_id):
		"""Update vendor information"""
		
		try:
			# Validate request data
			schema = VendorUpdateSchema()
			update_data = schema.load(request.get_json())
			
			# Update vendor
			service = self._run_async(self._get_vendor_service())
			vendor = self._run_async(service.update_vendor(vendor_id, update_data))
			
			if not vendor:
				return {'success': False, 'error': 'Vendor not found'}, 404
			
			return {
				'success': True,
				'data': {
					'id': vendor.id,
					'name': vendor.name,
					'status': vendor.status.value,
					'updated_at': vendor.updated_at.isoformat(),
					'version': vendor.version
				}
			}
			
		except Exception as e:
			return self._handle_error(e, "Error updating vendor")
	
	@has_access_api
	def delete(self, vendor_id):
		"""Deactivate/soft delete vendor"""
		
		try:
			service = self._run_async(self._get_vendor_service())
			success = self._run_async(service.deactivate_vendor(vendor_id))
			
			if not success:
				return {'success': False, 'error': 'Vendor not found'}, 404
			
			return {
				'success': True,
				'message': 'Vendor deactivated successfully'
			}
			
		except Exception as e:
			return self._handle_error(e, "Error deactivating vendor")


# ============================================================================
# VENDOR PERFORMANCE API RESOURCES
# ============================================================================

class VendorPerformanceResource(BaseVendorResource):
	"""API resource for vendor performance management"""
	
	@has_access_api
	def get(self, vendor_id):
		"""Get vendor performance history"""
		
		try:
			service = self._run_async(self._get_vendor_service())
			performance_summary = self._run_async(
				service.get_vendor_performance_summary(vendor_id)
			)
			
			if not performance_summary:
				return {'success': False, 'error': 'Vendor or performance data not found'}, 404
			
			return {
				'success': True,
				'data': performance_summary.model_dump()
			}
			
		except Exception as e:
			return self._handle_error(e, "Error retrieving performance data")
	
	@has_access_api
	def post(self, vendor_id):
		"""Record new performance data"""
		
		try:
			# Validate request data
			schema = PerformanceRecordSchema()
			performance_data = schema.load(request.get_json())
			performance_data['vendor_id'] = vendor_id
			
			# Record performance
			service = self._run_async(self._get_vendor_service())
			performance = self._run_async(service.record_performance(performance_data))
			
			return {
				'success': True,
				'data': {
					'id': performance.id,
					'vendor_id': performance.vendor_id,
					'overall_score': float(performance.overall_score),
					'measurement_period': performance.measurement_period,
					'created_at': performance.created_at.isoformat()
				}
			}, 201
			
		except Exception as e:
			return self._handle_error(e, "Error recording performance data")


class VendorRiskResource(BaseVendorResource):
	"""API resource for vendor risk management"""
	
	@has_access_api
	def get(self, vendor_id):
		"""Get vendor risk profile"""
		
		try:
			service = self._run_async(self._get_vendor_service())
			risk_profile = self._run_async(service.get_vendor_risk_profile(vendor_id))
			
			if not risk_profile:
				return {'success': False, 'error': 'Vendor risk profile not found'}, 404
			
			return {
				'success': True,
				'data': risk_profile.model_dump()
			}
			
		except Exception as e:
			return self._handle_error(e, "Error retrieving risk profile")
	
	@has_access_api
	def post(self, vendor_id):
		"""Record new risk assessment"""
		
		try:
			# Validate request data
			schema = RiskRecordSchema()
			risk_data = schema.load(request.get_json())
			risk_data['vendor_id'] = vendor_id
			
			# Record risk
			service = self._run_async(self._get_vendor_service())
			risk = self._run_async(service.record_risk(risk_data))
			
			return {
				'success': True,
				'data': {
					'id': risk.id,
					'vendor_id': risk.vendor_id,
					'risk_type': risk.risk_type,
					'severity': risk.severity.value,
					'overall_risk_score': float(risk.overall_risk_score),
					'created_at': risk.created_at.isoformat()
				}
			}, 201
			
		except Exception as e:
			return self._handle_error(e, "Error recording risk assessment")


# ============================================================================
# VENDOR INTELLIGENCE API RESOURCES
# ============================================================================

class VendorIntelligenceResource(BaseVendorResource):
	"""API resource for AI-powered vendor intelligence"""
	
	@has_access_api
	def get(self, vendor_id):
		"""Get latest vendor intelligence insights"""
		
		try:
			service = self._run_async(self._get_vendor_service())
			intelligence = self._run_async(service.get_latest_vendor_intelligence(vendor_id))
			
			if not intelligence:
				return {'success': False, 'error': 'Vendor intelligence not found'}, 404
			
			return {
				'success': True,
				'data': intelligence.model_dump()
			}
			
		except Exception as e:
			return self._handle_error(e, "Error retrieving vendor intelligence")
	
	@has_access_api
	def post(self, vendor_id):
		"""Generate fresh vendor intelligence"""
		
		try:
			service = self._run_async(self._get_vendor_service())
			intelligence = self._run_async(service.generate_vendor_intelligence(vendor_id))
			
			return {
				'success': True,
				'data': {
					'id': intelligence.id,
					'vendor_id': intelligence.vendor_id,
					'confidence_score': float(intelligence.confidence_score),
					'intelligence_date': intelligence.intelligence_date.isoformat(),
					'behavior_patterns_count': len(intelligence.behavior_patterns),
					'predictive_insights_count': len(intelligence.predictive_insights)
				}
			}, 201
			
		except Exception as e:
			return self._handle_error(e, "Error generating vendor intelligence")


class VendorOptimizationResource(BaseVendorResource):
	"""API resource for vendor optimization recommendations"""
	
	@has_access_api
	def post(self, vendor_id):
		"""Generate optimization plan for vendor"""
		
		try:
			objectives = request.get_json().get('objectives', [
				'performance_improvement', 'cost_reduction', 'risk_mitigation'
			])
			
			intelligence_engine = self._run_async(self._get_intelligence_engine())
			optimization_plan = self._run_async(
				intelligence_engine.generate_optimization_plan(vendor_id, objectives)
			)
			
			return {
				'success': True,
				'data': optimization_plan.model_dump()
			}, 201
			
		except Exception as e:
			return self._handle_error(e, "Error generating optimization plan")


# ============================================================================
# VENDOR ANALYTICS API RESOURCES
# ============================================================================

class VendorAnalyticsResource(BaseVendorResource):
	"""API resource for vendor analytics and reporting"""
	
	@has_access_api
	def get(self):
		"""Get comprehensive vendor analytics"""
		
		try:
			service = self._run_async(self._get_vendor_service())
			analytics = self._run_async(service.get_vendor_analytics())
			
			return {
				'success': True,
				'data': analytics
			}
			
		except Exception as e:
			return self._handle_error(e, "Error retrieving vendor analytics")


# ============================================================================
# API BLUEPRINT SETUP
# ============================================================================

def create_vendor_api_blueprint() -> Blueprint:
	"""Create and configure the vendor management API blueprint"""
	
	# Create blueprint
	vendor_api_bp = Blueprint(
		'vendor_management_api',
		__name__,
		url_prefix='/api/v1/vendor-management'
	)
	
	# Create Flask-RESTful API instance
	api = Api(vendor_api_bp, prefix='/api/v1/vendor-management')
	
	# Register API resources
	api.add_resource(VendorListResource, '/vendors')
	api.add_resource(VendorDetailResource, '/vendors/<string:vendor_id>')
	api.add_resource(VendorPerformanceResource, '/vendors/<string:vendor_id>/performance')
	api.add_resource(VendorRiskResource, '/vendors/<string:vendor_id>/risk')
	api.add_resource(VendorIntelligenceResource, '/vendors/<string:vendor_id>/intelligence')
	api.add_resource(VendorOptimizationResource, '/vendors/<string:vendor_id>/optimization')
	api.add_resource(VendorAnalyticsResource, '/analytics')
	
	return vendor_api_bp


# ============================================================================
# API REGISTRATION FUNCTION
# ============================================================================

def register_vendor_api(app):
	"""Register vendor management API with Flask application"""
	
	# Create and register blueprint
	vendor_api_bp = create_vendor_api_blueprint()
	app.register_blueprint(vendor_api_bp)
	
	# Log API registration
	app.logger.info("Vendor Management API registered successfully")
	app.logger.info("Available endpoints:")
	app.logger.info("- GET    /api/v1/vendor-management/vendors")
	app.logger.info("- POST   /api/v1/vendor-management/vendors")
	app.logger.info("- GET    /api/v1/vendor-management/vendors/<id>")
	app.logger.info("- PUT    /api/v1/vendor-management/vendors/<id>")
	app.logger.info("- DELETE /api/v1/vendor-management/vendors/<id>")
	app.logger.info("- GET    /api/v1/vendor-management/vendors/<id>/performance")
	app.logger.info("- POST   /api/v1/vendor-management/vendors/<id>/performance")
	app.logger.info("- GET    /api/v1/vendor-management/vendors/<id>/risk")
	app.logger.info("- POST   /api/v1/vendor-management/vendors/<id>/risk")
	app.logger.info("- GET    /api/v1/vendor-management/vendors/<id>/intelligence")
	app.logger.info("- POST   /api/v1/vendor-management/vendors/<id>/intelligence")
	app.logger.info("- POST   /api/v1/vendor-management/vendors/<id>/optimization")
	app.logger.info("- GET    /api/v1/vendor-management/analytics")