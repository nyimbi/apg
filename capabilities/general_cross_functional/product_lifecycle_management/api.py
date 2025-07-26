"""
Product Lifecycle Management (PLM) Capability - RESTful API

Comprehensive REST API implementation with APG integration patterns,
real-time features, and mobile-responsive design support.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

from flask import Blueprint, request, jsonify, session, current_app, g
from flask_restful import Api, Resource, abort, fields, marshal_with
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from marshmallow import Schema, fields as ma_fields, validate, ValidationError
from functools import wraps
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid_extensions import uuid7str
import json

from .service import PLMProductService, PLMEngineeringChangeService, PLMCollaborationService
from .ai_service import PLMAIService
from .views import (
	PLMProductView, PLMProductStructureView, PLMEngineeringChangeView,
	PLMProductConfigurationView, PLMCollaborationSessionView, PLMComplianceRecordView
)


# Flask Blueprint and API setup
plm_api_bp = Blueprint('plm_api', __name__, url_prefix='/api/v1/plm')
api = Api(plm_api_bp)

# Rate limiting setup (integrated with APG performance infrastructure)
limiter = Limiter(
	key_func=get_remote_address,
	default_limits=["1000 per hour"],
	storage_uri="redis://localhost:6379"  # APG Redis infrastructure
)


# Authentication and authorization decorators

def require_auth(f):
	"""Require APG authentication"""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		if 'user_id' not in session:
			abort(401, message="Authentication required")
		return f(*args, **kwargs)
	return decorated_function


def require_tenant(f):
	"""Require valid tenant context"""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		if 'tenant_id' not in session:
			abort(400, message="Tenant context required")
		g.tenant_id = session['tenant_id']
		g.user_id = session['user_id']
		return f(*args, **kwargs)
	return decorated_function


def require_permission(permission: str):
	"""Require specific APG permission"""
	def decorator(f):
		@wraps(f)
		def decorated_function(*args, **kwargs):
			# Integration with APG auth_rbac capability
			if not _check_user_permission(session.get('user_id'), permission, session.get('tenant_id')):
				abort(403, message=f"Permission '{permission}' required")
			return f(*args, **kwargs)
		return decorated_function
	return decorator


def _check_user_permission(user_id: str, permission: str, tenant_id: str) -> bool:
	"""Check user permission via APG auth_rbac integration"""
	# Simulate APG auth_rbac integration
	# In production, this would call the actual APG auth service
	return True  # For now, allow all authenticated users


# Marshmallow schemas for API validation

class ProductCreateSchema(Schema):
	"""Product creation schema"""
	product_name = ma_fields.Str(required=True, validate=validate.Length(min=3, max=200))
	product_number = ma_fields.Str(required=True, validate=validate.Length(min=3, max=50))
	product_description = ma_fields.Str(allow_none=True, validate=validate.Length(max=2000))
	product_type = ma_fields.Str(required=True, validate=validate.OneOf([
		'manufactured', 'purchased', 'virtual', 'service', 'kit', 
		'raw_material', 'subassembly', 'finished_good'
	]))
	lifecycle_phase = ma_fields.Str(validate=validate.OneOf([
		'concept', 'design', 'prototype', 'development', 'testing',
		'production', 'active', 'mature', 'declining', 'obsolete', 'discontinued'
	]), missing='concept')
	target_cost = ma_fields.Float(validate=validate.Range(min=0), missing=0.0)
	current_cost = ma_fields.Float(validate=validate.Range(min=0), missing=0.0)
	unit_of_measure = ma_fields.Str(missing='each')
	custom_attributes = ma_fields.Dict(missing=dict)
	tags = ma_fields.List(ma_fields.Str(), missing=[])


class ProductUpdateSchema(ProductCreateSchema):
	"""Product update schema (all fields optional)"""
	product_name = ma_fields.Str(validate=validate.Length(min=3, max=200))
	product_number = ma_fields.Str(validate=validate.Length(min=3, max=50))
	product_type = ma_fields.Str(validate=validate.OneOf([
		'manufactured', 'purchased', 'virtual', 'service', 'kit', 
		'raw_material', 'subassembly', 'finished_good'
	]))


class EngineeringChangeCreateSchema(Schema):
	"""Engineering change creation schema"""
	change_title = ma_fields.Str(required=True, validate=validate.Length(min=5, max=200))
	change_description = ma_fields.Str(required=True, validate=validate.Length(min=10, max=2000))
	change_type = ma_fields.Str(required=True, validate=validate.OneOf([
		'design', 'process', 'documentation', 'cost_reduction',
		'quality_improvement', 'safety', 'regulatory', 'urgent'
	]))
	affected_products = ma_fields.List(ma_fields.Str(), required=True, validate=validate.Length(min=1))
	reason_for_change = ma_fields.Str(required=True, validate=validate.Length(min=10, max=1000))
	business_impact = ma_fields.Str(required=True)
	cost_impact = ma_fields.Float(missing=0.0)
	priority = ma_fields.Str(validate=validate.OneOf(['low', 'medium', 'high', 'critical']), missing='medium')


class CollaborationSessionCreateSchema(Schema):
	"""Collaboration session creation schema"""
	session_name = ma_fields.Str(required=True, validate=validate.Length(min=3, max=200))
	description = ma_fields.Str(validate=validate.Length(max=1000))
	session_type = ma_fields.Str(required=True, validate=validate.OneOf([
		'design_review', 'change_review', 'brainstorming', 'problem_solving',
		'training', 'customer_meeting', 'supplier_meeting'
	]))
	scheduled_start = ma_fields.DateTime(required=True)
	scheduled_end = ma_fields.DateTime(required=True)
	max_participants = ma_fields.Int(validate=validate.Range(min=1, max=100), missing=20)
	recording_enabled = ma_fields.Bool(missing=False)
	whiteboard_enabled = ma_fields.Bool(missing=True)
	file_sharing_enabled = ma_fields.Bool(missing=True)
	invited_users = ma_fields.List(ma_fields.Str(), missing=[])


# Error handling

@plm_api_bp.errorhandler(ValidationError)
def handle_validation_error(error):
	"""Handle marshmallow validation errors"""
	return jsonify({
		'success': False,
		'error': 'Validation error',
		'details': error.messages
	}), 400


@plm_api_bp.errorhandler(Exception)
def handle_general_error(error):
	"""Handle general exceptions"""
	current_app.logger.error(f"PLM API error: {error}")
	return jsonify({
		'success': False,
		'error': 'Internal server error',
		'message': str(error) if current_app.debug else 'An error occurred'
	}), 500


# Product Management API Resources

class ProductListAPI(Resource):
	"""Product list and creation API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("100 per minute")]
	
	@require_permission('plm.products.read')
	def get(self):
		"""List products with filtering and pagination"""
		try:
			# Parse query parameters
			page = int(request.args.get('page', 1))
			page_size = min(int(request.args.get('page_size', 20)), 100)
			search_text = request.args.get('search', '')
			product_type = request.args.get('product_type', '')
			lifecycle_phase = request.args.get('lifecycle_phase', '')
			tags = request.args.getlist('tags')
			
			# Cost filters
			cost_min = request.args.get('cost_min', type=float)
			cost_max = request.args.get('cost_max', type=float)
			
			# Date filters
			created_from = request.args.get('created_from')
			created_to = request.args.get('created_to')
			
			# Sorting
			sort_by = request.args.get('sort_by', 'created_at')
			sort_order = request.args.get('sort_order', 'desc')
			
			# Execute search
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			result = loop.run_until_complete(
				service.search_products(
					tenant_id=g.tenant_id,
					user_id=g.user_id,
					search_text=search_text,
					product_type=product_type,
					lifecycle_phase=lifecycle_phase,
					tags=tags,
					cost_min=cost_min,
					cost_max=cost_max,
					created_from=created_from,
					created_to=created_to,
					page=page,
					page_size=page_size,
					sort_by=sort_by,
					sort_order=sort_order
				)
			)
			
			return {
				'success': True,
				'data': result['products'],
				'pagination': {
					'page': page,
					'page_size': page_size,
					'total_pages': result['total_pages'],
					'total_count': result['total_count']
				}
			}
			
		except Exception as e:
			current_app.logger.error(f"Product list error: {e}")
			abort(500, message="Failed to retrieve products")
	
	@require_permission('plm.products.create')
	def post(self):
		"""Create new product"""
		try:
			# Validate input
			schema = ProductCreateSchema()
			data = schema.load(request.json)
			
			# Create product
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			product = loop.run_until_complete(
				service.create_product(
					tenant_id=g.tenant_id,
					product_data=data,
					user_id=g.user_id
				)
			)
			
			if product:
				return {
					'success': True,
					'data': product.model_dump(),
					'message': 'Product created successfully'
				}, 201
			else:
				abort(400, message="Failed to create product")
				
		except ValidationError as e:
			abort(400, message="Validation error", details=e.messages)
		except Exception as e:
			current_app.logger.error(f"Product creation error: {e}")
			abort(500, message="Failed to create product")


class ProductDetailAPI(Resource):
	"""Individual product API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("200 per minute")]
	
	@require_permission('plm.products.read')
	def get(self, product_id: str):
		"""Get product details"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			product = loop.run_until_complete(
				service.get_product(product_id, g.user_id, g.tenant_id)
			)
			
			if product:
				return {
					'success': True,
					'data': product.model_dump()
				}
			else:
				abort(404, message="Product not found")
				
		except Exception as e:
			current_app.logger.error(f"Product detail error: {e}")
			abort(500, message="Failed to retrieve product")
	
	@require_permission('plm.products.update')
	def put(self, product_id: str):
		"""Update product"""
		try:
			# Validate input
			schema = ProductUpdateSchema()
			data = schema.load(request.json)
			
			# Update product
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			product = loop.run_until_complete(
				service.update_product(product_id, data, g.user_id, g.tenant_id)
			)
			
			if product:
				return {
					'success': True,
					'data': product.model_dump(),
					'message': 'Product updated successfully'
				}
			else:
				abort(404, message="Product not found")
				
		except ValidationError as e:
			abort(400, message="Validation error", details=e.messages)
		except Exception as e:
			current_app.logger.error(f"Product update error: {e}")
			abort(500, message="Failed to update product")
	
	@require_permission('plm.products.delete')
	def delete(self, product_id: str):
		"""Soft delete product"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			success = loop.run_until_complete(
				service.delete_product(product_id, g.user_id, g.tenant_id)
			)
			
			if success:
				return {
					'success': True,
					'message': 'Product deleted successfully'
				}
			else:
				abort(404, message="Product not found")
				
		except Exception as e:
			current_app.logger.error(f"Product deletion error: {e}")
			abort(500, message="Failed to delete product")


class ProductStructureAPI(Resource):
	"""Product structure (BOM) API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("100 per minute")]
	
	@require_permission('plm.products.read')
	def get(self, product_id: str):
		"""Get product BOM structure"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			structure = loop.run_until_complete(
				service.get_product_structure(product_id, g.user_id, g.tenant_id)
			)
			
			return {
				'success': True,
				'data': structure
			}
			
		except Exception as e:
			current_app.logger.error(f"Product structure error: {e}")
			abort(500, message="Failed to retrieve product structure")


class ProductDigitalTwinAPI(Resource):
	"""Product digital twin API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("50 per minute")]
	
	@require_permission('plm.products.digital_twin')
	def post(self, product_id: str):
		"""Create digital twin for product"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			twin_id = loop.run_until_complete(
				service.create_digital_twin(product_id, g.user_id, g.tenant_id)
			)
			
			if twin_id:
				return {
					'success': True,
					'digital_twin_id': twin_id,
					'message': 'Digital twin created successfully'
				}, 201
			else:
				abort(400, message="Failed to create digital twin")
				
		except Exception as e:
			current_app.logger.error(f"Digital twin creation error: {e}")
			abort(500, message="Failed to create digital twin")


# Engineering Change Management APIs

class EngineeringChangeListAPI(Resource):
	"""Engineering change list and creation API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("100 per minute")]
	
	@require_permission('plm.changes.read')
	def get(self):
		"""List engineering changes"""
		try:
			# Parse query parameters
			page = int(request.args.get('page', 1))
			page_size = min(int(request.args.get('page_size', 20)), 100)
			status = request.args.get('status', '')
			change_type = request.args.get('change_type', '')
			priority = request.args.get('priority', '')
			
			# Execute search
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMEngineeringChangeService()
			result = loop.run_until_complete(
				service.search_changes(
					tenant_id=g.tenant_id,
					user_id=g.user_id,
					status=status,
					change_type=change_type,
					priority=priority,
					page=page,
					page_size=page_size
				)
			)
			
			return {
				'success': True,
				'data': result['changes'],
				'pagination': {
					'page': page,
					'page_size': page_size,
					'total_pages': result['total_pages'],
					'total_count': result['total_count']
				}
			}
			
		except Exception as e:
			current_app.logger.error(f"Change list error: {e}")
			abort(500, message="Failed to retrieve changes")
	
	@require_permission('plm.changes.create')
	def post(self):
		"""Create new engineering change"""
		try:
			# Validate input
			schema = EngineeringChangeCreateSchema()
			data = schema.load(request.json)
			
			# Create change
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMEngineeringChangeService()
			change = loop.run_until_complete(
				service.create_change(
					tenant_id=g.tenant_id,
					change_data=data,
					user_id=g.user_id
				)
			)
			
			if change:
				return {
					'success': True,
					'data': change.model_dump(),
					'message': 'Engineering change created successfully'
				}, 201
			else:
				abort(400, message="Failed to create engineering change")
				
		except ValidationError as e:
			abort(400, message="Validation error", details=e.messages)
		except Exception as e:
			current_app.logger.error(f"Change creation error: {e}")
			abort(500, message="Failed to create engineering change")


class EngineeringChangeDetailAPI(Resource):
	"""Individual engineering change API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("200 per minute")]
	
	@require_permission('plm.changes.read')
	def get(self, change_id: str):
		"""Get engineering change details"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMEngineeringChangeService()
			change = loop.run_until_complete(
				service.get_change(change_id, g.user_id, g.tenant_id)
			)
			
			if change:
				return {
					'success': True,
					'data': change.model_dump()
				}
			else:
				abort(404, message="Engineering change not found")
				
		except Exception as e:
			current_app.logger.error(f"Change detail error: {e}")
			abort(500, message="Failed to retrieve engineering change")


class EngineeringChangeApprovalAPI(Resource):
	"""Engineering change approval API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("50 per minute")]
	
	@require_permission('plm.changes.approve')
	def put(self, change_id: str):
		"""Approve engineering change"""
		try:
			approval_data = request.json or {}
			comments = approval_data.get('comments', '')
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMEngineeringChangeService()
			success = loop.run_until_complete(
				service.approve_change(change_id, g.user_id, g.tenant_id, comments)
			)
			
			if success:
				return {
					'success': True,
					'message': 'Engineering change approved successfully'
				}
			else:
				abort(400, message="Failed to approve engineering change")
				
		except Exception as e:
			current_app.logger.error(f"Change approval error: {e}")
			abort(500, message="Failed to approve engineering change")


# Collaboration APIs

class CollaborationSessionListAPI(Resource):
	"""Collaboration session list and creation API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("100 per minute")]
	
	@require_permission('plm.collaboration.read')
	def get(self):
		"""List collaboration sessions"""
		try:
			# Parse query parameters
			page = int(request.args.get('page', 1))
			page_size = min(int(request.args.get('page_size', 20)), 100)
			session_type = request.args.get('session_type', '')
			status = request.args.get('status', '')
			
			# Execute search
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMCollaborationService()
			result = loop.run_until_complete(
				service.search_sessions(
					tenant_id=g.tenant_id,
					user_id=g.user_id,
					session_type=session_type,
					status=status,
					page=page,
					page_size=page_size
				)
			)
			
			return {
				'success': True,
				'data': result['sessions'],
				'pagination': {
					'page': page,
					'page_size': page_size,
					'total_pages': result['total_pages'],
					'total_count': result['total_count']
				}
			}
			
		except Exception as e:
			current_app.logger.error(f"Collaboration session list error: {e}")
			abort(500, message="Failed to retrieve collaboration sessions")
	
	@require_permission('plm.collaboration.create')
	def post(self):
		"""Create new collaboration session"""
		try:
			# Validate input
			schema = CollaborationSessionCreateSchema()
			data = schema.load(request.json)
			
			# Create session
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMCollaborationService()
			session = loop.run_until_complete(
				service.create_collaboration_session(
					tenant_id=g.tenant_id,
					session_data=data,
					user_id=g.user_id
				)
			)
			
			if session:
				return {
					'success': True,
					'data': session.model_dump(),
					'message': 'Collaboration session created successfully'
				}, 201
			else:
				abort(400, message="Failed to create collaboration session")
				
		except ValidationError as e:
			abort(400, message="Validation error", details=e.messages)
		except Exception as e:
			current_app.logger.error(f"Collaboration session creation error: {e}")
			abort(500, message="Failed to create collaboration session")


class CollaborationSessionActionAPI(Resource):
	"""Collaboration session action API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("100 per minute")]
	
	@require_permission('plm.collaboration.participate')
	def post(self, session_id: str, action: str):
		"""Perform action on collaboration session"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMCollaborationService()
			
			if action == 'start':
				result = loop.run_until_complete(
					service.start_collaboration_session(session_id, g.user_id, g.tenant_id)
				)
				message = 'Session started successfully'
			elif action == 'join':
				result = loop.run_until_complete(
					service.join_collaboration_session(session_id, g.user_id, g.tenant_id)
				)
				message = 'Joined session successfully'
			elif action == 'leave':
				result = loop.run_until_complete(
					service.leave_collaboration_session(session_id, g.user_id, g.tenant_id)
				)
				message = 'Left session successfully'
			elif action == 'end':
				result = loop.run_until_complete(
					service.end_collaboration_session(session_id, g.user_id, g.tenant_id)
				)
				message = 'Session ended successfully'
			else:
				abort(400, message="Invalid action")
			
			if result:
				return {
					'success': True,
					'data': result if isinstance(result, dict) else None,
					'message': message
				}
			else:
				abort(400, message=f"Failed to {action} session")
				
		except Exception as e:
			current_app.logger.error(f"Collaboration session {action} error: {e}")
			abort(500, message=f"Failed to {action} session")


# Analytics and AI APIs

class AnalyticsAPI(Resource):
	"""PLM analytics API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("50 per minute")]
	
	@require_permission('plm.analytics.read')
	def get(self, metric_type: str):
		"""Get analytics metrics"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			
			if metric_type == 'dashboard':
				metrics = loop.run_until_complete(
					service.get_dashboard_metrics(g.tenant_id, g.user_id)
				)
			elif metric_type == 'performance':
				ai_service = PLMAIService()
				metrics = loop.run_until_complete(
					ai_service.get_lifecycle_performance_insights(g.tenant_id)
				)
			elif metric_type == 'collaboration':
				collaboration_service = PLMCollaborationService()
				metrics = loop.run_until_complete(
					collaboration_service.get_collaboration_analytics(g.tenant_id, g.user_id)
				)
			else:
				abort(400, message="Invalid metric type")
			
			return {
				'success': True,
				'data': metrics
			}
			
		except Exception as e:
			current_app.logger.error(f"Analytics error: {e}")
			abort(500, message="Failed to retrieve analytics")


class AIInsightsAPI(Resource):
	"""AI-powered insights API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("20 per minute")]
	
	@require_permission('plm.ai.insights')
	def get(self, insight_type: str):
		"""Get AI-powered insights"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			ai_service = PLMAIService()
			
			if insight_type == 'innovation':
				insights = loop.run_until_complete(
					ai_service.get_innovation_insights(g.tenant_id)
				)
			elif insight_type == 'cost_optimization':
				insights = loop.run_until_complete(
					ai_service.get_cost_optimization_insights(g.tenant_id)
				)
			elif insight_type == 'supplier_intelligence':
				insights = loop.run_until_complete(
					ai_service.get_supplier_intelligence_insights(g.tenant_id)
				)
			else:
				abort(400, message="Invalid insight type")
			
			return {
				'success': True,
				'data': insights
			}
			
		except Exception as e:
			current_app.logger.error(f"AI insights error: {e}")
			abort(500, message="Failed to retrieve AI insights")


# Mobile-specific APIs

class MobileProductAPI(Resource):
	"""Mobile-optimized product API"""
	
	decorators = [require_auth, require_tenant, limiter.limit("200 per minute")]
	
	@require_permission('plm.products.read')
	def get(self):
		"""Get mobile-optimized product list"""
		try:
			page = int(request.args.get('page', 1))
			page_size = min(int(request.args.get('page_size', 10)), 20)  # Smaller page size for mobile
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = PLMProductService()
			result = loop.run_until_complete(
				service.get_mobile_products(
					tenant_id=g.tenant_id,
					user_id=g.user_id,
					page=page,
					page_size=page_size
				)
			)
			
			return {
				'success': True,
				'data': result['products'],
				'pagination': {
					'page': page,
					'page_size': page_size,
					'total_pages': result['total_pages'],
					'total_count': result['total_count']
				}
			}
			
		except Exception as e:
			current_app.logger.error(f"Mobile product API error: {e}")
			abort(500, message="Failed to retrieve mobile products")


# WebSocket support for real-time features
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect

socketio = SocketIO(cors_allowed_origins="*")

@socketio.on('join_collaboration')
def on_join_collaboration(data):
	"""Join collaboration room"""
	session_id = data['session_id']
	user_id = session.get('user_id')
	
	if user_id:
		join_room(session_id)
		emit('user_joined', {'user_id': user_id}, room=session_id)

@socketio.on('leave_collaboration')
def on_leave_collaboration(data):
	"""Leave collaboration room"""
	session_id = data['session_id']
	user_id = session.get('user_id')
	
	if user_id:
		leave_room(session_id)
		emit('user_left', {'user_id': user_id}, room=session_id)

@socketio.on('collaboration_update')
def on_collaboration_update(data):
	"""Handle collaboration updates"""
	session_id = data['session_id']
	update_data = data['update']
	user_id = session.get('user_id')
	
	if user_id:
		emit('collaboration_update', {
			'user_id': user_id,
			'update': update_data,
			'timestamp': datetime.utcnow().isoformat()
		}, room=session_id, include_self=False)


# Register API resources
api.add_resource(ProductListAPI, '/products')
api.add_resource(ProductDetailAPI, '/products/<string:product_id>')
api.add_resource(ProductStructureAPI, '/products/<string:product_id>/structure')
api.add_resource(ProductDigitalTwinAPI, '/products/<string:product_id>/digital_twin')

api.add_resource(EngineeringChangeListAPI, '/changes')
api.add_resource(EngineeringChangeDetailAPI, '/changes/<string:change_id>')
api.add_resource(EngineeringChangeApprovalAPI, '/changes/<string:change_id>/approve')

api.add_resource(CollaborationSessionListAPI, '/collaborate/sessions')
api.add_resource(CollaborationSessionActionAPI, '/collaborate/sessions/<string:session_id>/<string:action>')

api.add_resource(AnalyticsAPI, '/analytics/<string:metric_type>')
api.add_resource(AIInsightsAPI, '/ai/insights/<string:insight_type>')

api.add_resource(MobileProductAPI, '/mobile/products')


# Module exports
__all__ = [
	'plm_api_bp',
	'api',
	'limiter',
	'socketio'
]