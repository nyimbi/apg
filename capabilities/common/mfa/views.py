"""
APG Multi-Factor Authentication (MFA) - Flask-AppBuilder Views

Comprehensive Flask-AppBuilder views providing intuitive MFA management
interface with modern UI components and APG integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import logging
import json
from flask import request, jsonify, render_template, redirect, url_for, flash
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from wtforms import Form, StringField, SelectField, BooleanField, TextAreaField, validators
from wtforms.widgets import TextArea

from .models import (
	MFAUserProfile, MFAMethod, AuthEvent, DeviceInfo
)
from .service import MFAService
from .integration import APGIntegrationRouter


class MFAUserProfileView(ModelView):
	"""Flask-AppBuilder view for MFA User Profiles"""
	
	datamodel = SQLAInterface(MFAUserProfile)
	
	# List view configuration
	list_columns = [
		'user_id', 'tenant_id', 'mfa_enabled', 'default_method_type',
		'trust_level', 'last_authentication', 'failed_attempts', 'is_locked_out'
	]
	
	search_columns = ['user_id', 'tenant_id', 'email']
	
	# Show view configuration
	show_columns = [
		'user_id', 'tenant_id', 'mfa_enabled', 'default_method_type',
		'trust_level', 'last_authentication', 'failed_attempts', 'is_locked_out',
		'lockout_until', 'preferences', 'created_at', 'updated_at'
	]
	
	# Edit view configuration
	edit_columns = [
		'mfa_enabled', 'default_method_type', 'trust_level',
		'max_failed_attempts', 'lockout_duration_minutes', 'preferences'
	]
	
	# Add view configuration
	add_columns = [
		'user_id', 'tenant_id', 'mfa_enabled', 'default_method_type',
		'max_failed_attempts', 'lockout_duration_minutes'
	]
	
	# Labels
	label_columns = {
		'user_id': 'User ID',
		'tenant_id': 'Tenant ID',
		'mfa_enabled': 'MFA Enabled',
		'default_method_type': 'Default Method',
		'trust_level': 'Trust Level',
		'last_authentication': 'Last Authentication',
		'failed_attempts': 'Failed Attempts',
		'is_locked_out': 'Locked Out',
		'lockout_until': 'Locked Until',
		'max_failed_attempts': 'Max Failed Attempts',
		'lockout_duration_minutes': 'Lockout Duration (min)',
		'preferences': 'Preferences'
	}
	
	# Formatters
	formatters_columns = {
		'preferences': lambda x: json.dumps(x, indent=2) if x else '',
		'last_authentication': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else 'Never',
		'trust_level': lambda x: x.value if x else 'Unknown'
	}


class MFAMethodView(ModelView):
	"""Flask-AppBuilder view for MFA Methods"""
	
	datamodel = SQLAInterface(MFAMethod)
	
	# List view configuration
	list_columns = [
		'user_id', 'tenant_id', 'method_type', 'method_name',
		'is_primary', 'is_verified', 'last_used', 'created_at'
	]
	
	search_columns = ['user_id', 'tenant_id', 'method_name']
	
	# Show view configuration
	show_columns = [
		'id', 'user_id', 'tenant_id', 'method_type', 'method_name',
		'is_primary', 'is_verified', 'is_active', 'verification_attempts',
		'last_used', 'device_binding', 'backup_data', 'created_at', 'updated_at'
	]
	
	# Edit view configuration
	edit_columns = [
		'method_name', 'is_primary', 'is_active', 'device_binding'
	]
	
	# Labels
	label_columns = {
		'method_type': 'Method Type',
		'method_name': 'Method Name',
		'is_primary': 'Primary Method',
		'is_verified': 'Verified',
		'is_active': 'Active',
		'verification_attempts': 'Verification Attempts',
		'last_used': 'Last Used',
		'device_binding': 'Device Binding',
		'backup_data': 'Backup Data'
	}
	
	# Formatters
	formatters_columns = {
		'method_type': lambda x: x.value if x else '',
		'last_used': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else 'Never',
		'device_binding': lambda x: json.dumps(x, indent=2) if x else '',
		'backup_data': lambda x: json.dumps(x, indent=2) if x else ''
	}


class AuthEventView(ModelView):
	"""Flask-AppBuilder view for Authentication Events"""
	
	datamodel = SQLAInterface(AuthEvent)
	
	# List view configuration
	list_columns = [
		'user_id', 'tenant_id', 'event_type', 'status',
		'method_used', 'risk_score', 'created_at'
	]
	
	search_columns = ['user_id', 'tenant_id', 'event_type', 'status']
	
	# Show view configuration
	show_columns = [
		'id', 'user_id', 'tenant_id', 'event_type', 'status',
		'method_used', 'risk_score', 'location_data', 'device_data',
		'details', 'created_at'
	]
	
	# Labels
	label_columns = {
		'event_type': 'Event Type',
		'method_used': 'Method Used',
		'risk_score': 'Risk Score',
		'location_data': 'Location Data',
		'device_data': 'Device Data'
	}
	
	# Formatters
	formatters_columns = {
		'location_data': lambda x: json.dumps(x, indent=2) if x else '',
		'device_data': lambda x: json.dumps(x, indent=2) if x else '',
		'details': lambda x: json.dumps(x, indent=2) if x else '',
		'created_at': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else ''
	}
	
	# Make read-only
	can_edit = False
	can_add = False
	can_delete = False


class MFADashboardView(BaseView):
	"""Main MFA Dashboard View"""
	
	default_view = 'dashboard'
	
	def __init__(self):
		super().__init__()
		self.logger = logging.getLogger(__name__)
	
	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Main MFA dashboard"""
		try:
			# Get current user context
			user_id = request.current_user.id if hasattr(request, 'current_user') else 'demo_user'
			tenant_id = getattr(request.current_user, 'tenant_id', 'demo_tenant') if hasattr(request, 'current_user') else 'demo_tenant'
			
			# Initialize MFA service (this would come from dependency injection in production)
			# For now, we'll create a mock service
			mfa_service = self._get_mfa_service()
			
			# Get user MFA status
			mfa_status = {
				'mfa_enabled': True,
				'methods': [
					{'id': '1', 'type': 'TOTP', 'name': 'Authenticator App', 'verified': True, 'primary': True},
					{'id': '2', 'type': 'SMS', 'name': 'Phone +1234567890', 'verified': True, 'primary': False}
				],
				'status': 'configured',
				'trust_score': 0.85,
				'recent_events': [
					{'type': 'authentication', 'status': 'success', 'timestamp': '2025-01-29 10:30:00'},
					{'type': 'method_added', 'status': 'success', 'timestamp': '2025-01-28 15:45:00'}
				],
				'backup_codes_available': True,
				'biometric_enrolled': False
			}
			
			# Get system metrics (mock data)
			metrics = {
				'total_users': 1250,
				'active_mfa_users': 1100,
				'success_rate': 98.5,
				'avg_risk_score': 0.25
			}
			
			return self.render_template(
				'mfa/dashboard.html',
				mfa_status=mfa_status,
				metrics=metrics,
				user_id=user_id,
				tenant_id=tenant_id
			)
			
		except Exception as e:
			self.logger.error(f"Dashboard error: {str(e)}", exc_info=True)
			flash('Error loading dashboard', 'error')
			return self.render_template('mfa/error.html', error=str(e))
	
	@expose('/enroll/')
	@has_access
	def enroll(self):
		"""MFA method enrollment page"""
		try:
			# Available enrollment options
			enrollment_options = [
				{
					'type': 'TOTP',
					'name': 'Authenticator App',
					'description': 'Use Google Authenticator, Authy, or similar apps',
					'icon': 'fa-mobile',
					'difficulty': 'Easy'
				},
				{
					'type': 'SMS',
					'name': 'SMS Verification',
					'description': 'Receive codes via text message',
					'icon': 'fa-sms',
					'difficulty': 'Easy'
				},
				{
					'type': 'EMAIL',
					'name': 'Email Verification',
					'description': 'Receive codes via email',
					'icon': 'fa-envelope',
					'difficulty': 'Easy'
				},
				{
					'type': 'FACE_RECOGNITION',
					'name': 'Face Recognition',
					'description': 'Use your face for biometric authentication',
					'icon': 'fa-user',
					'difficulty': 'Medium'
				},
				{
					'type': 'VOICE_RECOGNITION',
					'name': 'Voice Recognition',
					'description': 'Use your voice for biometric authentication',
					'icon': 'fa-microphone',
					'difficulty': 'Medium'
				},
				{
					'type': 'HARDWARE_TOKEN',
					'name': 'Hardware Token',
					'description': 'YubiKey or similar hardware security keys',
					'icon': 'fa-key',
					'difficulty': 'Advanced'
				}
			]
			
			return self.render_template(
				'mfa/enroll.html',
				enrollment_options=enrollment_options
			)
			
		except Exception as e:
			self.logger.error(f"Enrollment page error: {str(e)}", exc_info=True)
			flash('Error loading enrollment page', 'error')
			return redirect(url_for('MFADashboardView.dashboard'))
	
	@expose('/settings/')
	@has_access
	def settings(self):
		"""MFA settings page"""
		try:
			# Get current user settings
			settings = {
				'mfa_enabled': True,
				'default_method': 'TOTP',
				'backup_codes_count': 8,
				'trusted_devices': [
					{'id': '1', 'name': 'iPhone 13', 'last_used': '2025-01-29 10:30:00'},
					{'id': '2', 'name': 'MacBook Pro', 'last_used': '2025-01-28 09:15:00'}
				],
				'notification_preferences': {
					'login_notifications': True,
					'security_alerts': True,
					'method_changes': True
				}
			}
			
			return self.render_template(
				'mfa/settings.html',
				settings=settings
			)
			
		except Exception as e:
			self.logger.error(f"Settings page error: {str(e)}", exc_info=True)
			flash('Error loading settings', 'error')
			return redirect(url_for('MFADashboardView.dashboard'))
	
	@expose('/recovery/')
	@has_access
	def recovery(self):
		"""Account recovery page"""
		try:
			# Recovery options
			recovery_options = [
				{
					'type': 'backup_codes',
					'name': 'Backup Codes',
					'description': 'Use one of your saved backup codes',
					'available': True
				},
				{
					'type': 'email_verification',
					'name': 'Email Verification',
					'description': 'Verify your identity via email',
					'available': True
				},
				{
					'type': 'admin_override',
					'name': 'Administrator Override',
					'description': 'Contact administrator for manual reset',
					'available': True
				}
			]
			
			return self.render_template(
				'mfa/recovery.html',
				recovery_options=recovery_options
			)
			
		except Exception as e:
			self.logger.error(f"Recovery page error: {str(e)}", exc_info=True)
			flash('Error loading recovery page', 'error')
			return redirect(url_for('MFADashboardView.dashboard'))
	
	def _get_mfa_service(self):
		"""Get MFA service instance (mock for now)"""
		# In production, this would be injected via dependency injection
		return None


class MFAAPIView(BaseView):
	"""API endpoints for MFA operations"""
	
	route_base = '/api/mfa'
	
	def __init__(self):
		super().__init__()
		self.logger = logging.getLogger(__name__)
	
	@expose('/status/', methods=['GET'])
	@protect()
	def get_status(self):
		"""Get MFA status for current user"""
		try:
			user_id = request.current_user.id if hasattr(request, 'current_user') else 'demo_user'
			tenant_id = getattr(request.current_user, 'tenant_id', 'demo_tenant') if hasattr(request, 'current_user') else 'demo_tenant'
			
			# Mock status data
			status = {
				'mfa_enabled': True,
				'methods_count': 2,
				'trust_score': 0.85,
				'last_authentication': '2025-01-29T10:30:00Z',
				'is_locked_out': False
			}
			
			return jsonify({'success': True, 'data': status})
			
		except Exception as e:
			self.logger.error(f"Status API error: {str(e)}", exc_info=True)
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/enroll/<method_type>/', methods=['POST'])
	@protect()
	def enroll_method(self, method_type):
		"""Enroll new MFA method"""
		try:
			data = request.get_json()
			user_id = request.current_user.id if hasattr(request, 'current_user') else 'demo_user'
			tenant_id = getattr(request.current_user, 'tenant_id', 'demo_tenant') if hasattr(request, 'current_user') else 'demo_tenant'
			
			# Mock enrollment result
			result = {
				'success': True,
				'method_id': 'new_method_123',
				'verification_required': True,
				'next_step': 'verify_method'
			}
			
			if method_type == 'TOTP':
				result['secret'] = 'JBSWY3DPEHPK3PXP',
				result['qr_code'] = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...'
			
			return jsonify(result)
			
		except Exception as e:
			self.logger.error(f"Enrollment API error: {str(e)}", exc_info=True)
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/verify/', methods=['POST'])
	@protect()
	def verify_method(self):
		"""Verify MFA method"""
		try:
			data = request.get_json()
			method_id = data.get('method_id')
			verification_code = data.get('verification_code')
			
			# Mock verification
			if verification_code == '123456':
				result = {
					'success': True,
					'verified': True,
					'message': 'Method verified successfully'
				}
			else:
				result = {
					'success': False,
					'verified': False,
					'message': 'Invalid verification code'
				}
			
			return jsonify(result)
			
		except Exception as e:
			self.logger.error(f"Verification API error: {str(e)}", exc_info=True)
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/authenticate/', methods=['POST'])
	@protect()
	def authenticate(self):
		"""Authenticate with MFA"""
		try:
			data = request.get_json()
			methods = data.get('methods', [])
			
			# Mock authentication
			result = {
				'success': True,
				'authenticated': True,
				'trust_score': 0.9,
				'token': 'auth_token_123',
				'expires_at': '2025-01-29T18:30:00Z'
			}
			
			return jsonify(result)
			
		except Exception as e:
			self.logger.error(f"Authentication API error: {str(e)}", exc_info=True)
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/remove/<method_id>/', methods=['DELETE'])
	@protect()
	def remove_method(self, method_id):
		"""Remove MFA method"""
		try:
			# Mock removal
			result = {
				'success': True,
				'message': 'Method removed successfully'
			}
			
			return jsonify(result)
			
		except Exception as e:
			self.logger.error(f"Remove method API error: {str(e)}", exc_info=True)
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/backup-codes/', methods=['POST'])
	@protect()
	def generate_backup_codes(self):
		"""Generate backup codes"""
		try:
			# Mock backup codes
			codes = [
				'ABCD-1234', 'EFGH-5678', 'IJKL-9012',
				'MNOP-3456', 'QRST-7890', 'UVWX-1357',
				'YZAB-2468', 'CDEF-9753', 'GHIJ-8642',
				'KLMN-1975'
			]
			
			result = {
				'success': True,
				'backup_codes': codes,
				'message': 'Backup codes generated successfully'
			}
			
			return jsonify(result)
			
		except Exception as e:
			self.logger.error(f"Backup codes API error: {str(e)}", exc_info=True)
			return jsonify({'success': False, 'error': str(e)}), 500


# Create view instances for registration
mfa_user_profile_view = MFAUserProfileView()
mfa_method_view = MFAMethodView()
auth_event_view = AuthEventView()
mfa_dashboard_view = MFADashboardView()
mfa_api_view = MFAAPIView()


def register_mfa_views(appbuilder):
	"""Register all MFA views with Flask-AppBuilder"""
	
	# Register model views
	appbuilder.add_view(
		mfa_user_profile_view,
		"MFA User Profiles",
		icon="fa-users",
		category="MFA Management",
		category_icon="fa-shield"
	)
	
	appbuilder.add_view(
		mfa_method_view,
		"MFA Methods",
		icon="fa-key",
		category="MFA Management"
	)
	
	appbuilder.add_view(
		auth_event_view,
		"Authentication Events",
		icon="fa-history",
		category="MFA Management"
	)
	
	# Register custom views
	appbuilder.add_view_no_menu(mfa_dashboard_view)
	appbuilder.add_view_no_menu(mfa_api_view)
	
	# Add menu links
	appbuilder.add_link(
		"MFA Dashboard",
		href="/mfadashboardview/dashboard/",
		icon="fa-dashboard",
		category="MFA"
	)
	
	appbuilder.add_link(
		"Enroll MFA Method",
		href="/mfadashboardview/enroll/",
		icon="fa-plus",
		category="MFA"
	)
	
	appbuilder.add_link(
		"MFA Settings",
		href="/mfadashboardview/settings/",
		icon="fa-cog",
		category="MFA"
	)
	
	appbuilder.add_link(
		"Account Recovery",
		href="/mfadashboardview/recovery/",
		icon="fa-medkit",
		category="MFA"
	)


__all__ = [
	'MFAUserProfileView',
	'MFAMethodView', 
	'AuthEventView',
	'MFADashboardView',
	'MFAAPIView',
	'register_mfa_views'
]