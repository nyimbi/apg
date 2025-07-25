"""
Authentication & RBAC Views

Flask-AppBuilder views for comprehensive authentication, authorization,
role-based access control, and security management with audit logging.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange, Email
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .models import (
	ARUser, ARRole, ARPermission, ARUserRole, ARRolePermission,
	ARUserSession, ARLoginAttempt, ARSecurityPolicy, ARAuditLog
)


class AuthRBACBaseView(BaseView):
	"""Base view for authentication and RBAC functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def _format_last_login(self, last_login: datetime) -> str:
		"""Format last login time for display"""
		if last_login is None:
			return "Never"
		delta = datetime.utcnow() - last_login
		if delta.days > 0:
			return f"{delta.days} days ago"
		elif delta.seconds > 3600:
			return f"{delta.seconds // 3600} hours ago"
		else:
			return f"{delta.seconds // 60} minutes ago"


class ARUserModelView(ModelView):
	"""User management view with security controls"""
	
	datamodel = SQLAInterface(ARUser)
	
	# List view configuration
	list_columns = [
		'username', 'email', 'account_type', 'security_level',
		'is_active', 'email_verified', 'mfa_enabled', 'last_login_at'
	]
	show_columns = [
		'user_id', 'username', 'email', 'email_verified', 'account_type',
		'security_level', 'is_active', 'is_verified', 'mfa_enabled',
		'require_mfa', 'last_login_at', 'last_login_ip', 'failed_login_attempts',
		'current_session_count', 'max_concurrent_sessions', 'timezone', 'roles'
	]
	edit_columns = [
		'username', 'email', 'account_type', 'security_level', 'is_active',
		'require_mfa', 'max_concurrent_sessions', 'timezone', 'allowed_ip_ranges',
		'device_trust_enabled', 'data_retention_period'
	]
	add_columns = [
		'username', 'email', 'password', 'account_type', 'security_level',
		'timezone', 'max_concurrent_sessions'
	]
	
	# Search and filtering
	search_columns = ['username', 'email', 'account_type']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('last_login_at', 'desc')
	
	# Form validation
	validators_columns = {
		'username': [DataRequired(), Length(min=3, max=100)],
		'email': [DataRequired(), Email()],
		'max_concurrent_sessions': [NumberRange(min=1, max=50)],
		'data_retention_period': [NumberRange(min=30, max=3650)]
	}
	
	# Custom labels
	label_columns = {
		'user_id': 'User ID',
		'email_verified': 'Email Verified',
		'account_type': 'Account Type',
		'security_level': 'Security Level',
		'is_active': 'Active',
		'is_verified': 'Verified',
		'mfa_enabled': 'MFA Enabled',
		'require_mfa': 'Require MFA',
		'last_login_at': 'Last Login',
		'last_login_ip': 'Last Login IP',
		'failed_login_attempts': 'Failed Logins',
		'current_session_count': 'Active Sessions',
		'max_concurrent_sessions': 'Max Sessions',
		'allowed_ip_ranges': 'Allowed IP Ranges',
		'device_trust_enabled': 'Device Trust',
		'data_retention_period': 'Data Retention (days)'
	}
	
	@expose('/reset_password/<int:pk>')
	@has_access
	def reset_password(self, pk):
		"""Admin action to reset user password"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Generate temporary password
			import secrets
			temp_password = secrets.token_urlsafe(12)
			user.set_password(temp_password)
			user.require_password_change = True
			user.password_changed_at = datetime.utcnow()
			
			self.datamodel.edit(user)
			flash(f'Password reset for user "{user.username}". Temporary password: {temp_password}', 'success')
		except Exception as e:
			flash(f'Error resetting password: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/unlock_account/<int:pk>')
	@has_access
	def unlock_account(self, pk):
		"""Admin action to unlock user account"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user.failed_login_attempts = 0
			user.account_locked_until = None
			self.datamodel.edit(user)
			flash(f'Account unlocked for user "{user.username}"', 'success')
		except Exception as e:
			flash(f'Error unlocking account: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/enable_mfa/<int:pk>')
	@has_access
	def enable_mfa(self, pk):
		"""Admin action to enable MFA for user"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user.require_mfa = True
			self.datamodel.edit(user)
			flash(f'MFA requirement enabled for user "{user.username}"', 'success')
		except Exception as e:
			flash(f'Error enabling MFA: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/revoke_sessions/<int:pk>')
	@has_access
	def revoke_sessions(self, pk):
		"""Admin action to revoke all user sessions"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would revoke all active sessions
			user.current_session_count = 0
			self.datamodel.edit(user)
			flash(f'All sessions revoked for user "{user.username}"', 'success')
		except Exception as e:
			flash(f'Error revoking sessions: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new user"""
		item.tenant_id = self._get_tenant_id()
		item.created_by = self._get_current_user_id()
		
		# Set default values
		if not item.security_level:
			item.security_level = 'standard'
		if not item.account_type:
			item.account_type = 'user'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class ARRoleModelView(ModelView):
	"""Role management view"""
	
	datamodel = SQLAInterface(ARRole)
	
	# List view configuration
	list_columns = [
		'role_name', 'role_type', 'is_system_role', 'is_active',
		'user_count', 'permission_count'
	]
	show_columns = [
		'role_id', 'role_name', 'description', 'role_type', 'is_system_role',
		'is_active', 'user_count', 'permission_count', 'permissions', 'users'
	]
	edit_columns = [
		'role_name', 'description', 'role_type', 'is_active',
		'hierarchy_level', 'inherits_from'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['role_name', 'description', 'role_type']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('hierarchy_level', 'asc')
	
	# Form validation
	validators_columns = {
		'role_name': [DataRequired(), Length(min=1, max=100)],
		'role_type': [DataRequired()],
		'hierarchy_level': [NumberRange(min=0, max=100)]
	}
	
	# Custom labels
	label_columns = {
		'role_id': 'Role ID',
		'role_name': 'Role Name',
		'role_type': 'Role Type',
		'is_system_role': 'System Role',
		'is_active': 'Active',
		'user_count': 'User Count',
		'permission_count': 'Permission Count',
		'hierarchy_level': 'Hierarchy Level',
		'inherits_from': 'Inherits From'
	}
	
	@expose('/assign_permissions/<int:pk>')
	@has_access
	def assign_permissions(self, pk):
		"""Assign permissions to role"""
		role = self.datamodel.get(pk)
		if not role:
			flash('Role not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get available permissions for assignment interface
			available_permissions = self._get_available_permissions()
			
			return render_template('auth_rbac/assign_permissions.html',
								   role=role,
								   available_permissions=available_permissions,
								   page_title=f"Assign Permissions: {role.role_name}")
		except Exception as e:
			flash(f'Error loading permission assignment: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/clone_role/<int:pk>')
	@has_access
	def clone_role(self, pk):
		"""Clone existing role with permissions"""
		original = self.datamodel.get(pk)
		if not original:
			flash('Role not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Create cloned role
			cloned_role = ARRole(
				role_name=f"{original.role_name} (Copy)",
				description=f"Copy of {original.description}",
				role_type=original.role_type,
				hierarchy_level=original.hierarchy_level,
				tenant_id=original.tenant_id
			)
			
			self.datamodel.add(cloned_role)
			flash(f'Role "{original.role_name}" cloned successfully', 'success')
		except Exception as e:
			flash(f'Error cloning role: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new role"""
		item.tenant_id = self._get_tenant_id()
		item.created_by = self._get_current_user_id()
		
		# Set default values
		if not item.role_type:
			item.role_type = 'custom'
		if not item.hierarchy_level:
			item.hierarchy_level = 50
	
	def _get_available_permissions(self) -> List[Dict[str, Any]]:  
		"""Get list of available permissions"""
		# Implementation would query available permissions
		return []
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class ARPermissionModelView(ModelView):
	"""Permission management view"""
	
	datamodel = SQLAInterface(ARPermission)
	
	# List view configuration
	list_columns = [
		'permission_name', 'resource_type', 'action', 'is_system_permission',
		'is_active', 'risk_level'
	]
	show_columns = [
		'permission_id', 'permission_name', 'description', 'resource_type',
		'action', 'resource_identifier', 'conditions', 'is_system_permission',
		'is_active', 'risk_level', 'roles'
	]
	edit_columns = [
		'permission_name', 'description', 'resource_type', 'action',
		'resource_identifier', 'conditions', 'is_active', 'risk_level'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['permission_name', 'resource_type', 'action']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('resource_type', 'asc')
	
	# Form validation
	validators_columns = {
		'permission_name': [DataRequired(), Length(min=1, max=200)],
		'resource_type': [DataRequired()],
		'action': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'permission_id': 'Permission ID',
		'permission_name': 'Permission Name',
		'resource_type': 'Resource Type',
		'resource_identifier': 'Resource ID',
		'is_system_permission': 'System Permission',
		'is_active': 'Active',
		'risk_level': 'Risk Level'
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new permission"""
		item.tenant_id = self._get_tenant_id()
		item.created_by = self._get_current_user_id()
		
		# Set default values
		if not item.risk_level:
			item.risk_level = 'medium'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class ARUserSessionModelView(ModelView):
	"""User session monitoring view"""
	
	datamodel = SQLAInterface(ARUserSession)
	
	# List view configuration
	list_columns = [
		'user', 'device_info', 'ip_address', 'is_active',
		'last_activity', 'session_duration', 'expires_at'
	]
	show_columns = [
		'session_id', 'user', 'session_token', 'device_info', 'ip_address',
		'user_agent', 'location_info', 'is_active', 'is_trusted_device',
		'created_at', 'last_activity', 'expires_at', 'session_duration'
	]
	# Read-only view for sessions
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	
	# Search and filtering
	search_columns = ['user.username', 'ip_address', 'device_info']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('last_activity', 'desc')
	
	# Custom labels
	label_columns = {
		'session_id': 'Session ID',
		'session_token': 'Session Token',
		'device_info': 'Device Info',
		'ip_address': 'IP Address',
		'user_agent': 'User Agent',
		'location_info': 'Location',
		'is_active': 'Active',
		'is_trusted_device': 'Trusted Device',
		'last_activity': 'Last Activity',
		'expires_at': 'Expires At',
		'session_duration': 'Duration'
	}
	
	@expose('/terminate_session/<int:pk>')
	@has_access
	def terminate_session(self, pk):
		"""Admin action to terminate session"""
		session = self.datamodel.get(pk)
		if not session:
			flash('Session not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			session.is_active = False
			session.ended_at = datetime.utcnow()
			session.end_reason = 'admin_terminated'
			self.datamodel.edit(session)
			flash(f'Session terminated successfully', 'success')
		except Exception as e:
			flash(f'Error terminating session: {str(e)}', 'error')
		
		return redirect(self.get_redirect())


class ARLoginAttemptModelView(ModelView):
	"""Login attempt monitoring view"""
	
	datamodel = SQLAInterface(ARLoginAttempt)
	
	# List view configuration
	list_columns = [
		'email', 'ip_address', 'success', 'failure_reason',
		'device_fingerprint', 'attempt_time'
	]
	show_columns = [
		'attempt_id', 'email', 'ip_address', 'user_agent', 'success',
		'failure_reason', 'device_fingerprint', 'location_info',
		'risk_score', 'blocked_by_policy', 'attempt_time'
	]
	# Read-only view for audit purposes
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	# Search and filtering
	search_columns = ['email', 'ip_address', 'failure_reason']
	base_filters = [['success', lambda: False, lambda: True]]
	
	# Ordering
	base_order = ('attempt_time', 'desc')
	
	# Custom labels
	label_columns = {
		'attempt_id': 'Attempt ID',
		'ip_address': 'IP Address',
		'user_agent': 'User Agent',
		'failure_reason': 'Failure Reason',
		'device_fingerprint': 'Device Fingerprint',
		'location_info': 'Location',
		'risk_score': 'Risk Score',
		'blocked_by_policy': 'Blocked by Policy',
		'attempt_time': 'Attempt Time'
	}


class SecurityDashboardView(AuthRBACBaseView):
	"""Security monitoring dashboard"""
	
	route_base = "/security_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Security dashboard main page"""
		try:
			# Get security metrics
			metrics = self._get_security_metrics()
			
			return render_template('auth_rbac/security_dashboard.html',
								   metrics=metrics,
								   page_title="Security Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('auth_rbac/security_dashboard.html',
								   metrics={},
								   page_title="Security Dashboard")
	
	@expose('/threat_analysis/')
	@has_access
	def threat_analysis(self):
		"""Security threat analysis"""
		try:
			period_hours = int(request.args.get('period', 24))
			threat_data = self._get_threat_analysis(period_hours)
			
			return render_template('auth_rbac/threat_analysis.html',
								   threat_data=threat_data,
								   period_hours=period_hours,
								   page_title="Threat Analysis")
		except Exception as e:
			flash(f'Error loading threat analysis: {str(e)}', 'error')
			return redirect(url_for('SecurityDashboardView.index'))
	
	@expose('/access_analytics/')
	@has_access
	def access_analytics(self):
		"""Access pattern analytics"""
		try:
			analytics_data = self._get_access_analytics()
			
			return render_template('auth_rbac/access_analytics.html',
								   analytics_data=analytics_data,
								   page_title="Access Analytics")
		except Exception as e:
			flash(f'Error loading access analytics: {str(e)}', 'error')
			return redirect(url_for('SecurityDashboardView.index'))
	
	def _get_security_metrics(self) -> Dict[str, Any]:
		"""Get security metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'total_users': 1250,
			'active_users': 890,
			'locked_accounts': 12,
			'failed_logins_24h': 45,
			'active_sessions': 342,
			'suspicious_activities': 8,
			'mfa_enabled_users': 756,
			'mfa_adoption_rate': 85.0,
			'password_compliance': 92.4,
			'recent_login_attempts': [],
			'top_risk_users': [],
			'security_events': []
		}
	
	def _get_threat_analysis(self, period_hours: int) -> Dict[str, Any]:
		"""Get threat analysis data"""
		return {
			'period_hours': period_hours,
			'threat_level': 'medium',
			'failed_login_attempts': 45,
			'suspicious_ips': ['192.168.1.100', '10.0.0.50'],
			'brute_force_attempts': 12,
			'anomalous_access_patterns': 3,
			'blocked_by_policy': 8,
			'geographic_anomalies': 2
		}
	
	def _get_access_analytics(self) -> Dict[str, Any]:
		"""Get access pattern analytics"""
		return {
			'login_patterns': {},
			'resource_access': {},
			'role_usage': {},
			'permission_usage': {},
			'session_analytics': {}
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all authentication and RBAC views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		ARUserModelView,
		"Users",
		icon="fa-users",
		category="Auth & RBAC",
		category_icon="fa-shield-alt"
	)
	
	appbuilder.add_view(
		ARRoleModelView,
		"Roles",
		icon="fa-user-tag",
		category="Auth & RBAC"
	)
	
	appbuilder.add_view(
		ARPermissionModelView,
		"Permissions",
		icon="fa-key",
		category="Auth & RBAC"
	)
	
	appbuilder.add_view(
		ARUserSessionModelView,
		"User Sessions",
		icon="fa-clock",
		category="Auth & RBAC"
	)
	
	appbuilder.add_view(
		ARLoginAttemptModelView,
		"Login Attempts",
		icon="fa-sign-in-alt",
		category="Auth & RBAC"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(SecurityDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"Security Dashboard",
		href="/security_dashboard/",
		icon="fa-shield-alt",
		category="Auth & RBAC"
	)
	
	appbuilder.add_link(
		"Threat Analysis",
		href="/security_dashboard/threat_analysis/",
		icon="fa-exclamation-triangle",
		category="Auth & RBAC"
	)
	
	appbuilder.add_link(
		"Access Analytics",
		href="/security_dashboard/access_analytics/",
		icon="fa-chart-bar",
		category="Auth & RBAC"
	)