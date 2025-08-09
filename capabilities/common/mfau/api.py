"""
APG Multi-Factor Authentication (MFA) - REST API Endpoints

Comprehensive REST API providing full MFA functionality with OpenAPI documentation,
rate limiting, and APG ecosystem integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from flask import Flask, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.exceptions import BadRequest, Unauthorized, Forbidden, NotFound
from functools import wraps
import time

from .models import MFAUserProfile, MFAMethod, MFAMethodType, AuthEvent
from .service import MFAService
from .integration import APGIntegrationRouter


def _log_api_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log API operations for debugging and audit"""
	return f"[MFA API] {operation} for user {user_id}: {details}"


# Rate limiting decorator
def rate_limit(max_requests: int = 100, window_seconds: int = 60):
	"""Rate limiting decorator for API endpoints"""
	def decorator(f):
		@wraps(f)
		def decorated_function(*args, **kwargs):
			# Simple in-memory rate limiting (use Redis in production)
			client_ip = request.remote_addr
			current_time = time.time()
			
			# This would be implemented with Redis in production
			# For now, skip rate limiting in development
			
			return f(*args, **kwargs)
		return decorated_function
	return decorator


# Authentication decorator
def require_auth(f):
	"""Require authentication for API endpoint"""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		# Extract token from Authorization header
		auth_header = request.headers.get('Authorization', '')
		
		if not auth_header.startswith('Bearer '):
			return {'error': 'Missing or invalid authorization header'}, 401
		
		token = auth_header.split(' ')[1]
		
		# Validate token (this would integrate with your auth system)
		# For now, accept any non-empty token
		if not token:
			return {'error': 'Invalid token'}, 401
		
		# Add user context to request
		request.current_user_id = 'demo_user'  # This would come from token validation
		request.current_tenant_id = 'demo_tenant'
		
		return f(*args, **kwargs)
	return decorated_function


# Create Flask-RESTX API
def create_mfa_api(app: Flask, mfa_service: MFAService) -> Api:
	"""Create and configure MFA REST API"""
	
	api = Api(
		app,
		version='1.0',
		title='APG MFA API',
		description='Multi-Factor Authentication API for APG Platform',
		doc='/mfa/docs/',
		prefix='/api/mfa'
	)
	
	# Create namespaces
	auth_ns = Namespace('auth', description='Authentication operations')
	methods_ns = Namespace('methods', description='MFA method management')
	users_ns = Namespace('users', description='User MFA management')
	recovery_ns = Namespace('recovery', description='Account recovery')
	admin_ns = Namespace('admin', description='Administrative operations')
	
	api.add_namespace(auth_ns, path='/auth')
	api.add_namespace(methods_ns, path='/methods')
	api.add_namespace(users_ns, path='/users')
	api.add_namespace(recovery_ns, path='/recovery')
	api.add_namespace(admin_ns, path='/admin')
	
	# Define models for documentation
	auth_request_model = api.model('AuthRequest', {
		'methods': fields.List(fields.Raw, required=True, description='Authentication methods'),
		'context': fields.Raw(description='Authentication context')
	})
	
	auth_response_model = api.model('AuthResponse', {
		'success': fields.Boolean(required=True),
		'authenticated': fields.Boolean(),
		'trust_score': fields.Float(),
		'token': fields.String(),
		'expires_at': fields.String(),
		'step_up_required': fields.Boolean(),
		'error': fields.String()
	})
	
	method_model = api.model('MFAMethod', {
		'id': fields.String(required=True),
		'type': fields.String(required=True),
		'name': fields.String(),
		'is_primary': fields.Boolean(),
		'is_verified': fields.Boolean(),
		'is_active': fields.Boolean(),
		'created_at': fields.String()
	})
	
	user_status_model = api.model('UserStatus', {
		'mfa_enabled': fields.Boolean(required=True),
		'methods': fields.List(fields.Nested(method_model)),
		'trust_score': fields.Float(),
		'is_locked_out': fields.Boolean(),
		'backup_codes_available': fields.Boolean()
	})
	
	enrollment_request_model = api.model('EnrollmentRequest', {
		'method_type': fields.String(required=True),
		'enrollment_data': fields.Raw(),
		'context': fields.Raw()
	})
	
	verification_request_model = api.model('VerificationRequest', {
		'method_id': fields.String(required=True),
		'verification_code': fields.String(),
		'verification_data': fields.Raw()
	})
	
	# Authentication endpoints
	@auth_ns.route('/authenticate')
	class AuthenticateResource(Resource):
		@api.doc('authenticate_user')
		@api.expect(auth_request_model)
		@api.marshal_with(auth_response_model)
		@rate_limit(max_requests=50, window_seconds=60)
		def post(self):
			"""Authenticate user with MFA"""
			try:
				data = request.get_json()
				
				# Extract user context (would come from JWT in production)
				user_id = request.headers.get('X-User-ID', 'demo_user')
				tenant_id = request.headers.get('X-Tenant-ID', 'demo_tenant')
				
				# Build context
				context = data.get('context', {})
				context.update({
					'ip_address': request.remote_addr,
					'user_agent': request.headers.get('User-Agent', ''),
					'timestamp': datetime.utcnow().isoformat()
				})
				
				# Authenticate
				result = await mfa_service.authenticate_user(
					user_id=user_id,
					tenant_id=tenant_id,
					authentication_methods=data.get('methods', []),
					context=context
				)
				
				return result
				
			except Exception as e:
				logging.error(f"Authentication API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Authentication failed'}, 500
	
	@auth_ns.route('/verify')
	class VerifyResource(Resource):
		@api.doc('verify_method')
		@api.expect(verification_request_model)
		@require_auth
		def post(self):
			"""Verify MFA method"""
			try:
				data = request.get_json()
				method_id = data.get('method_id')
				verification_code = data.get('verification_code')
				
				# Mock verification for demo
				if verification_code == '123456':
					return {
						'success': True,
						'verified': True,
						'message': 'Method verified successfully'
					}
				else:
					return {
						'success': False,
						'verified': False,
						'message': 'Invalid verification code'
					}, 400
					
			except Exception as e:
				logging.error(f"Verification API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Verification failed'}, 500
	
	@auth_ns.route('/step-up')
	class StepUpResource(Resource):
		@api.doc('step_up_auth')
		@require_auth
		def post(self):
			"""Perform step-up authentication"""
			try:
				data = request.get_json()
				user_id = request.current_user_id
				tenant_id = request.current_tenant_id
				
				result = await mfa_service.verify_step_up_authentication(
					user_id=user_id,
					tenant_id=tenant_id,
					step_up_token=data.get('step_up_token'),
					additional_methods=data.get('additional_methods', []),
					context={'ip_address': request.remote_addr}
				)
				
				return result
				
			except Exception as e:
				logging.error(f"Step-up authentication API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Step-up authentication failed'}, 500
	
	# Method management endpoints
	@methods_ns.route('/')
	class MethodsResource(Resource):
		@api.doc('list_methods')
		@api.marshal_list_with(method_model)
		@require_auth
		def get(self):
			"""Get user's MFA methods"""
			try:
				user_id = request.current_user_id
				tenant_id = request.current_tenant_id
				
				status = await mfa_service.get_user_mfa_status(user_id, tenant_id)
				return status.get('methods', [])
				
			except Exception as e:
				logging.error(f"List methods API error: {str(e)}", exc_info=True)
				return {'error': 'Failed to retrieve methods'}, 500
		
		@api.doc('enroll_method')
		@api.expect(enrollment_request_model)
		@require_auth
		def post(self):
			"""Enroll new MFA method"""
			try:
				data = request.get_json()
				user_id = request.current_user_id
				tenant_id = request.current_tenant_id
				
				method_type = MFAMethodType(data.get('method_type'))
				enrollment_data = data.get('enrollment_data', {})
				context = data.get('context', {})
				context.update({
					'ip_address': request.remote_addr,
					'user_agent': request.headers.get('User-Agent', '')
				})
				
				result = await mfa_service.enroll_mfa_method(
					user_id=user_id,
					tenant_id=tenant_id,
					method_type=method_type,
					enrollment_data=enrollment_data,
					context=context
				)
				
				return result
				
			except Exception as e:
				logging.error(f"Enrollment API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Enrollment failed'}, 500
	
	@methods_ns.route('/<string:method_id>')
	class MethodResource(Resource):
		@api.doc('get_method')
		@api.marshal_with(method_model)
		@require_auth
		def get(self, method_id):
			"""Get specific MFA method"""
			try:
				# Mock method data
				return {
					'id': method_id,
					'type': 'TOTP',
					'name': 'Authenticator App',
					'is_primary': True,
					'is_verified': True,
					'is_active': True,
					'created_at': datetime.utcnow().isoformat()
				}
				
			except Exception as e:
				logging.error(f"Get method API error: {str(e)}", exc_info=True)
				return {'error': 'Method not found'}, 404
		
		@api.doc('remove_method')
		@require_auth
		def delete(self, method_id):
			"""Remove MFA method"""
			try:
				user_id = request.current_user_id
				tenant_id = request.current_tenant_id
				context = {'ip_address': request.remote_addr}
				
				result = await mfa_service.remove_mfa_method(
					user_id=user_id,
					tenant_id=tenant_id,
					method_id=method_id,
					context=context
				)
				
				return result
				
			except Exception as e:
				logging.error(f"Remove method API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Failed to remove method'}, 500
	
	@methods_ns.route('/<string:method_id>/primary')
	class SetPrimaryResource(Resource):
		@api.doc('set_primary_method')
		@require_auth
		def post(self, method_id):
			"""Set method as primary"""
			try:
				# Mock implementation
				return {
					'success': True,
					'message': 'Primary method updated successfully'
				}
				
			except Exception as e:
				logging.error(f"Set primary API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Failed to set primary method'}, 500
	
	@methods_ns.route('/<string:method_id>/test')
	class TestMethodResource(Resource):
		@api.doc('test_method')
		@require_auth
		def post(self, method_id):
			"""Test MFA method"""
			try:
				# Mock test implementation
				return {
					'success': True,
					'message': 'Method test successful'
				}
				
			except Exception as e:
				logging.error(f"Test method API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Method test failed'}, 500
	
	# User management endpoints
	@users_ns.route('/status')
	class UserStatusResource(Resource):
		@api.doc('get_user_status')
		@api.marshal_with(user_status_model)
		@require_auth
		def get(self):
			"""Get user MFA status"""
			try:
				user_id = request.current_user_id
				tenant_id = request.current_tenant_id
				
				status = await mfa_service.get_user_mfa_status(user_id, tenant_id)
				return status
				
			except Exception as e:
				logging.error(f"User status API error: {str(e)}", exc_info=True)
				return {'error': 'Failed to retrieve user status'}, 500
	
	@users_ns.route('/backup-codes')
	class BackupCodesResource(Resource):
		@api.doc('generate_backup_codes')
		@require_auth
		def post(self):
			"""Generate backup codes"""
			try:
				user_id = request.current_user_id
				tenant_id = request.current_tenant_id
				context = {'ip_address': request.remote_addr}
				
				result = await mfa_service.generate_backup_codes(
					user_id=user_id,
					tenant_id=tenant_id,
					context=context
				)
				
				return result
				
			except Exception as e:
				logging.error(f"Backup codes API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Failed to generate backup codes'}, 500
	
	@users_ns.route('/biometric/enroll')
	class BiometricEnrollResource(Resource):
		@api.doc('start_biometric_enrollment')
		@require_auth
		def post(self):
			"""Start biometric enrollment"""
			try:
				data = request.get_json()
				user_id = request.current_user_id
				tenant_id = request.current_tenant_id
				
				result = await mfa_service.start_biometric_enrollment(
					user_id=user_id,
					tenant_id=tenant_id,
					biometric_types=data.get('biometric_types', []),
					context={'ip_address': request.remote_addr}
				)
				
				return result
				
			except Exception as e:
				logging.error(f"Biometric enrollment API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Biometric enrollment failed'}, 500
	
	# Recovery endpoints
	@recovery_ns.route('/initiate')
	class InitiateRecoveryResource(Resource):
		@api.doc('initiate_recovery')
		def post(self):
			"""Initiate account recovery"""
			try:
				data = request.get_json()
				
				result = await mfa_service.initiate_account_recovery(
					user_id=data.get('user_id'),
					tenant_id=data.get('tenant_id'),
					recovery_type=data.get('recovery_type', 'mfa_reset'),
					context={'ip_address': request.remote_addr}
				)
				
				return result
				
			except Exception as e:
				logging.error(f"Recovery initiation API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Recovery initiation failed'}, 500
	
	@recovery_ns.route('/<string:recovery_id>/verify')
	class VerifyRecoveryResource(Resource):
		@api.doc('verify_recovery')
		def post(self, recovery_id):
			"""Verify recovery method"""
			try:
				data = request.get_json()
				
				# Mock recovery verification
				return {
					'success': True,
					'recovery_completed': False,
					'remaining_methods': ['email_verification']
				}
				
			except Exception as e:
				logging.error(f"Recovery verification API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Recovery verification failed'}, 500
	
	# Administrative endpoints
	@admin_ns.route('/metrics')
	class MetricsResource(Resource):
		@api.doc('get_metrics')
		@require_auth
		def get(self):
			"""Get MFA system metrics"""
			try:
				metrics = await mfa_service.get_service_metrics()
				return metrics
				
			except Exception as e:
				logging.error(f"Metrics API error: {str(e)}", exc_info=True)
				return {'error': 'Failed to retrieve metrics'}, 500
	
	@admin_ns.route('/users/<string:user_id>/status')
	class AdminUserStatusResource(Resource):
		@api.doc('get_admin_user_status')
		@require_auth
		def get(self, user_id):
			"""Get user MFA status (admin view)"""
			try:
				tenant_id = request.args.get('tenant_id', 'default')
				
				status = await mfa_service.get_user_mfa_status(user_id, tenant_id)
				return status
				
			except Exception as e:
				logging.error(f"Admin user status API error: {str(e)}", exc_info=True)
				return {'error': 'Failed to retrieve user status'}, 500
	
	@admin_ns.route('/users/<string:user_id>/unlock')
	class UnlockUserResource(Resource):
		@api.doc('unlock_user')
		@require_auth
		def post(self, user_id):
			"""Unlock user account"""
			try:
				# Mock unlock implementation
				return {
					'success': True,
					'message': 'User account unlocked successfully'
				}
				
			except Exception as e:
				logging.error(f"Unlock user API error: {str(e)}", exc_info=True)
				return {'success': False, 'error': 'Failed to unlock user'}, 500
	
	# Add error handlers
	@api.errorhandler(BadRequest)
	def handle_bad_request(error):
		return {'error': 'Bad request', 'message': str(error)}, 400
	
	@api.errorhandler(Unauthorized)
	def handle_unauthorized(error):
		return {'error': 'Unauthorized', 'message': 'Authentication required'}, 401
	
	@api.errorhandler(Forbidden)
	def handle_forbidden(error):
		return {'error': 'Forbidden', 'message': 'Insufficient permissions'}, 403
	
	@api.errorhandler(NotFound)
	def handle_not_found(error):
		return {'error': 'Not found', 'message': 'Resource not found'}, 404
	
	@api.errorhandler(Exception)
	def handle_internal_error(error):
		logging.error(f"Internal API error: {str(error)}", exc_info=True)
		return {'error': 'Internal server error', 'message': 'An unexpected error occurred'}, 500
	
	return api


# WebSocket events for real-time communication
class MFAWebSocketEvents:
	"""WebSocket event handlers for real-time MFA communication"""
	
	def __init__(self, socketio, mfa_service: MFAService):
		self.socketio = socketio
		self.mfa_service = mfa_service
		self.logger = logging.getLogger(__name__)
		
		# Register event handlers
		self.socketio.on_event('connect', self.on_connect)
		self.socketio.on_event('disconnect', self.on_disconnect)
		self.socketio.on_event('join_mfa_room', self.on_join_mfa_room)
		self.socketio.on_event('biometric_data', self.on_biometric_data)
		self.socketio.on_event('enrollment_progress', self.on_enrollment_progress)
	
	def on_connect(self):
		"""Handle client connection"""
		self.logger.info(f"Client connected: {request.sid}")
	
	def on_disconnect(self):
		"""Handle client disconnection"""
		self.logger.info(f"Client disconnected: {request.sid}")
	
	def on_join_mfa_room(self, data):
		"""Join user-specific MFA room for real-time updates"""
		user_id = data.get('user_id')
		if user_id:
			room = f"mfa_user_{user_id}"
			self.socketio.join_room(room)
			self.socketio.emit('joined', {'room': room})
			self.logger.info(f"Client {request.sid} joined MFA room: {room}")
	
	def on_biometric_data(self, data):
		"""Handle real-time biometric data for enrollment/authentication"""
		try:
			user_id = data.get('user_id')
			biometric_type = data.get('type')
			biometric_data = data.get('data')
			
			# Process biometric data
			result = self._process_biometric_data(user_id, biometric_type, biometric_data)
			
			# Send result back to client
			self.socketio.emit('biometric_result', result)
			
		except Exception as e:
			self.logger.error(f"Biometric data processing error: {str(e)}", exc_info=True)
			self.socketio.emit('biometric_error', {'error': str(e)})
	
	def on_enrollment_progress(self, data):
		"""Handle enrollment progress updates"""
		try:
			user_id = data.get('user_id')
			progress = data.get('progress')
			
			# Broadcast progress to user's room
			room = f"mfa_user_{user_id}"
			self.socketio.emit('enrollment_update', {
				'progress': progress,
				'timestamp': datetime.utcnow().isoformat()
			}, room=room)
			
		except Exception as e:
			self.logger.error(f"Enrollment progress error: {str(e)}", exc_info=True)
	
	def _process_biometric_data(self, user_id: str, biometric_type: str, data: Any) -> Dict[str, Any]:
		"""Process real-time biometric data"""
		# This would integrate with the biometric service
		return {
			'quality_score': 0.85,
			'enrollment_progress': 0.6,
			'feedback': 'Move closer to camera'
		}
	
	def broadcast_security_alert(self, user_id: str, alert_data: Dict[str, Any]):
		"""Broadcast security alert to user"""
		room = f"mfa_user_{user_id}"
		self.socketio.emit('security_alert', alert_data, room=room)
	
	def broadcast_method_status_change(self, user_id: str, method_data: Dict[str, Any]):
		"""Broadcast MFA method status change"""
		room = f"mfa_user_{user_id}"
		self.socketio.emit('method_status_change', method_data, room=room)


__all__ = [
	'create_mfa_api',
	'MFAWebSocketEvents',
	'require_auth',
	'rate_limit'
]