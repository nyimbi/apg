"""
Multi-Tenant Enterprise Views

Flask-AppBuilder views for enterprise-grade multi-tenant deployment management,
tenant administration, SSO configuration, and comprehensive audit capabilities.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, FloatField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange, Email
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .models import (
	MTETenant, MTEUser, MTESSOConfiguration, MTEAuditEvent,
	MTETenantUsage, MTEComplianceReport
)


class MultiTenantEnterpriseBaseView(BaseView):
	"""Base view for multi-tenant enterprise functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_current_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# In a real implementation, this would be determined from user context
		return "default_tenant"
	
	def _format_bytes(self, bytes_value: float) -> str:
		"""Format bytes for display"""
		if bytes_value is None:
			return "N/A"
		
		for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
			if bytes_value < 1024.0:
				return f"{bytes_value:.1f} {unit}"
			bytes_value /= 1024.0
		return f"{bytes_value:.1f} PB"
	
	def _format_percentage(self, value: float) -> str:
		"""Format percentage for display"""
		if value is None:
			return "N/A"
		return f"{value:.1f}%"
	
	def _format_currency(self, value: float) -> str:
		"""Format currency for display"""
		if value is None:
			return "N/A"
		return f"${value:,.2f}"


class MTETenantModelView(ModelView):
	"""Multi-tenant enterprise tenant management view"""
	
	datamodel = SQLAInterface(MTETenant)
	
	# List view configuration
	list_columns = [
		'organization_name', 'domain', 'subscription_tier', 'status',
		'current_users', 'max_users', 'current_storage_gb', 'last_activity'
	]
	show_columns = [
		'tenant_id', 'organization_name', 'domain', 'subdomain', 'logo_url',
		'status', 'subscription_tier', 'billing_status', 'max_users', 'max_digital_twins',
		'max_storage_gb', 'current_users', 'current_digital_twins', 'current_storage_gb',
		'features_enabled', 'compliance_profiles', 'encryption_level',
		'primary_contact_email', 'support_tier', 'last_activity', 'created_at'
	]
	edit_columns = [
		'organization_name', 'domain', 'subdomain', 'logo_url', 'status',
		'subscription_tier', 'billing_status', 'max_users', 'max_digital_twins',
		'max_storage_gb', 'features_enabled', 'custom_branding', 'data_retention_days',
		'encryption_level', 'compliance_profiles', 'primary_contact_email',
		'billing_contact_email', 'support_tier'
	]
	add_columns = [
		'organization_name', 'domain', 'primary_contact_email', 'subscription_tier'
	]
	
	# Search and filtering
	search_columns = ['organization_name', 'domain', 'subscription_tier']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('organization_name', 'asc')
	
	# Form validation
	validators_columns = {
		'organization_name': [DataRequired(), Length(min=3, max=200)],
		'domain': [DataRequired(), Length(min=3, max=100)],
		'primary_contact_email': [DataRequired(), Email()],
		'max_users': [NumberRange(min=1)],
		'max_digital_twins': [NumberRange(min=1)],
		'max_storage_gb': [NumberRange(min=0.1)]
	}
	
	# Custom labels
	label_columns = {
		'tenant_id': 'Tenant ID',
		'organization_name': 'Organization Name',
		'subdomain': 'Custom Subdomain',
		'logo_url': 'Logo URL',
		'subscription_tier': 'Subscription Tier',
		'billing_status': 'Billing Status',
		'max_users': 'Max Users',
		'max_digital_twins': 'Max Digital Twins',
		'max_storage_gb': 'Max Storage (GB)',
		'max_api_calls_per_month': 'Max API Calls/Month',
		'max_concurrent_sessions': 'Max Concurrent Sessions',
		'current_users': 'Current Users',
		'current_digital_twins': 'Current Digital Twins',
		'current_storage_gb': 'Current Storage (GB)',
		'current_api_calls_month': 'Current API Calls (Month)',
		'features_enabled': 'Enabled Features',
		'custom_branding': 'Custom Branding',
		'data_retention_days': 'Data Retention (Days)',
		'backup_enabled': 'Backup Enabled',
		'encryption_level': 'Encryption Level',
		'geographic_region': 'Geographic Region',
		'compliance_profiles': 'Compliance Profiles',
		'security_policies': 'Security Policies',
		'audit_retention_days': 'Audit Retention (Days)',
		'primary_contact_email': 'Primary Contact Email',
		'billing_contact_email': 'Billing Contact Email',
		'support_tier': 'Support Tier',
		'last_activity': 'Last Activity',
		'last_login': 'Last Login',
		'trial_end_date': 'Trial End Date'
	}
	
	@expose('/tenant_dashboard/<int:pk>')
	@has_access
	def tenant_dashboard(self, pk):
		"""View tenant dashboard with metrics and usage"""
		tenant = self.datamodel.get(pk)
		if not tenant:
			flash('Tenant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			dashboard_data = self._get_tenant_dashboard_data(tenant)
			
			return render_template('multi_tenant_enterprise/tenant_dashboard.html',
								   tenant=tenant,
								   dashboard_data=dashboard_data,
								   page_title=f"Tenant Dashboard: {tenant.organization_name}")
		except Exception as e:
			flash(f'Error loading tenant dashboard: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/tenant_users/<int:pk>')
	@has_access
	def tenant_users(self, pk):
		"""View tenant users and access management"""
		tenant = self.datamodel.get(pk)
		if not tenant:
			flash('Tenant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			users_data = self._get_tenant_users_data(tenant)
			
			return render_template('multi_tenant_enterprise/tenant_users.html',
								   tenant=tenant,
								   users_data=users_data,
								   page_title=f"Users: {tenant.organization_name}")
		except Exception as e:
			flash(f'Error loading tenant users: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/usage_analytics/<int:pk>')
	@has_access
	def usage_analytics(self, pk):
		"""View tenant usage analytics and billing information"""
		tenant = self.datamodel.get(pk)
		if not tenant:
			flash('Tenant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			analytics_data = self._get_usage_analytics_data(tenant)
			
			return render_template('multi_tenant_enterprise/usage_analytics.html',
								   tenant=tenant,
								   analytics_data=analytics_data,
								   page_title=f"Usage Analytics: {tenant.organization_name}")
		except Exception as e:
			flash(f'Error loading usage analytics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/suspend_tenant/<int:pk>')
	@has_access
	def suspend_tenant(self, pk):
		"""Suspend tenant account"""
		tenant = self.datamodel.get(pk)
		if not tenant:
			flash('Tenant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if tenant.status == 'active':
				tenant.status = 'suspended'
				self.datamodel.edit(tenant)
				flash(f'Tenant "{tenant.organization_name}" has been suspended', 'success')
			else:
				flash('Tenant cannot be suspended in current state', 'warning')
		except Exception as e:
			flash(f'Error suspending tenant: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/activate_tenant/<int:pk>')
	@has_access
	def activate_tenant(self, pk):
		"""Activate tenant account"""
		tenant = self.datamodel.get(pk)
		if not tenant:
			flash('Tenant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if tenant.status in ['suspended', 'pending']:
				tenant.status = 'active'
				tenant.last_activity = datetime.utcnow()
				self.datamodel.edit(tenant)
				flash(f'Tenant "{tenant.organization_name}" has been activated', 'success')
			else:
				flash('Tenant cannot be activated in current state', 'warning')
		except Exception as e:
			flash(f'Error activating tenant: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new tenant"""
		# Set default values
		if not item.status:
			item.status = 'active'
		if not item.subscription_tier:
			item.subscription_tier = 'basic'
		if not item.encryption_level:
			item.encryption_level = 'standard'
		if not item.support_tier:
			item.support_tier = 'standard'
		
		# Set tier-based defaults
		tier_defaults = self._get_tier_defaults(item.subscription_tier)
		for key, value in tier_defaults.items():
			if not getattr(item, key, None):
				setattr(item, key, value)
	
	def _get_tier_defaults(self, tier: str) -> Dict[str, Any]:
		"""Get default values for subscription tier"""
		tier_configs = {
			'basic': {
				'max_users': 10,
				'max_digital_twins': 50,
				'max_storage_gb': 100.0,
				'features_enabled': ['basic_twins', 'standard_analytics']
			},
			'professional': {
				'max_users': 50,
				'max_digital_twins': 500,
				'max_storage_gb': 1000.0,
				'features_enabled': ['advanced_twins', 'real_time_analytics', 'api_access']
			},
			'enterprise': {
				'max_users': 500,
				'max_digital_twins': 10000,
				'max_storage_gb': 10000.0,
				'features_enabled': ['all_features', 'custom_integrations', 'sso', 'audit_logs']
			}
		}
		return tier_configs.get(tier, tier_configs['basic'])
	
	def _get_tenant_dashboard_data(self, tenant: MTETenant) -> Dict[str, Any]:
		"""Get tenant dashboard data"""
		usage_percentages = tenant.calculate_usage_percentage()
		limits_status = tenant.is_within_limits()
		
		return {
			'overview_metrics': {
				'total_users': tenant.current_users,
				'active_digital_twins': tenant.current_digital_twins,
				'storage_used_gb': tenant.current_storage_gb,
				'api_calls_month': tenant.current_api_calls_month,
				'subscription_tier': tenant.subscription_tier,
				'billing_status': tenant.billing_status
			},
			'usage_percentages': usage_percentages,
			'limits_status': limits_status,
			'recent_activity': {
				'last_login': tenant.last_login,
				'last_activity': tenant.last_activity,
				'account_created': tenant.created_at
			},
			'feature_access': {
				'sso_enabled': 'sso' in tenant.features_enabled,
				'advanced_analytics': 'advanced_analytics' in tenant.features_enabled,
				'api_access': 'api_access' in tenant.features_enabled,
				'custom_branding': bool(tenant.custom_branding)
			},
			'compliance_status': {
				'profiles_enabled': tenant.compliance_profiles,
				'audit_retention_days': tenant.audit_retention_days,
				'encryption_level': tenant.encryption_level
			}
		}
	
	def _get_tenant_users_data(self, tenant: MTETenant) -> Dict[str, Any]:
		"""Get tenant users data"""
		return {
			'user_summary': {
				'total_users': len(tenant.users),
				'active_users': len([u for u in tenant.users if u.is_active]),
				'sso_users': len([u for u in tenant.users if u.sso_provider]),
				'admin_users': len([u for u in tenant.users if u.role in ['tenant_admin', 'super_admin']])
			},
			'role_distribution': {
				'tenant_admin': len([u for u in tenant.users if u.role == 'tenant_admin']),
				'twin_admin': len([u for u in tenant.users if u.role == 'twin_admin']),
				'engineer': len([u for u in tenant.users if u.role == 'engineer']),
				'analyst': len([u for u in tenant.users if u.role == 'analyst']),
				'viewer': len([u for u in tenant.users if u.role == 'viewer'])
			},
			'recent_activity': [
				{
					'user_email': user.email,
					'last_login': user.last_login,
					'login_count': user.total_login_count,
					'status': 'active' if user.is_active else 'inactive'
				}
				for user in sorted(tenant.users, key=lambda x: x.last_login or datetime.min, reverse=True)[:10]
			]
		}
	
	def _get_usage_analytics_data(self, tenant: MTETenant) -> Dict[str, Any]:
		"""Get usage analytics data"""
		return {
			'current_usage': {
				'users': f"{tenant.current_users}/{tenant.max_users}",
				'digital_twins': f"{tenant.current_digital_twins}/{tenant.max_digital_twins}",
				'storage_gb': f"{tenant.current_storage_gb:.1f}/{tenant.max_storage_gb}",
				'api_calls': f"{tenant.current_api_calls_month}/{tenant.max_api_calls_per_month}"
			},
			'usage_trends': {
				'user_growth': [85, 92, 98, 105, 110, 108, 115],  # Mock data
				'storage_growth': [45.2, 52.1, 58.9, 65.3, 72.8, 78.1, 85.6],
				'api_usage': [8500, 9200, 10100, 11500, 12800, 13200, 15000]
			},
			'cost_analysis': {
				'monthly_base_cost': 299.00,
				'overage_charges': 45.50,
				'total_monthly_cost': 344.50,
				'projected_yearly_cost': 4134.00
			},
			'feature_utilization': {
				'sso_logins': 156,
				'api_integrations': 12,
				'advanced_analytics_runs': 45,
				'backup_operations': 30
			}
		}


class MTEUserModelView(ModelView):
	"""Multi-tenant enterprise user management view"""
	
	datamodel = SQLAInterface(MTEUser)
	
	# List view configuration
	list_columns = [
		'email', 'first_name', 'last_name', 'tenant', 'role',
		'is_active', 'sso_provider', 'last_login'
	]
	show_columns = [
		'user_id', 'tenant', 'email', 'first_name', 'last_name', 'display_name',
		'role', 'permissions', 'groups', 'is_active', 'is_verified',
		'sso_provider', 'external_user_id', 'last_login', 'total_login_count',
		'failed_login_attempts', 'timezone', 'language', 'created_at'
	]
	edit_columns = [
		'first_name', 'last_name', 'display_name', 'role', 'permissions',
		'is_active', 'timezone', 'language', 'notification_preferences',
		'force_password_change', 'max_concurrent_sessions'
	]
	add_columns = [
		'tenant', 'email', 'first_name', 'last_name', 'role'
	]
	
	# Search and filtering
	search_columns = ['email', 'first_name', 'last_name', 'role']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('email', 'asc')
	
	# Form validation
	validators_columns = {
		'email': [DataRequired(), Email()],
		'first_name': [DataRequired(), Length(min=1, max=100)],
		'last_name': [DataRequired(), Length(min=1, max=100)],
		'role': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'user_id': 'User ID',
		'tenant_id': 'Tenant ID',
		'display_name': 'Display Name',
		'avatar_url': 'Avatar URL',
		'password_hash': 'Password Hash',
		'external_user_id': 'External User ID',
		'sso_provider': 'SSO Provider',
		'is_active': 'Active',
		'is_verified': 'Verified',
		'email_verified_at': 'Email Verified At',
		'last_login': 'Last Login',
		'last_login_ip': 'Last Login IP',
		'failed_login_attempts': 'Failed Login Attempts',
		'account_locked_until': 'Account Locked Until',
		'force_password_change': 'Force Password Change',
		'current_session_id': 'Current Session ID',
		'max_concurrent_sessions': 'Max Concurrent Sessions',
		'session_timeout_minutes': 'Session Timeout (Minutes)',
		'notification_preferences': 'Notification Preferences',
		'ui_preferences': 'UI Preferences',
		'last_activity': 'Last Activity',
		'total_login_count': 'Total Login Count',
		'invitation_sent_at': 'Invitation Sent',
		'invitation_accepted_at': 'Invitation Accepted'
	}
	
	@expose('/user_activity/<int:pk>')
	@has_access
	def user_activity(self, pk):
		"""View user activity and audit trail"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			activity_data = self._get_user_activity_data(user)
			
			return render_template('multi_tenant_enterprise/user_activity.html',
								   user=user,
								   activity_data=activity_data,
								   page_title=f"User Activity: {user.email}")
		except Exception as e:
			flash(f'Error loading user activity: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/reset_password/<int:pk>')
	@has_access
	def reset_password(self, pk):
		"""Reset user password"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user.force_password_change = True
			user.failed_login_attempts = 0
			if user.account_locked_until:
				user.account_locked_until = None
			
			self.datamodel.edit(user)
			flash(f'Password reset initiated for user {user.email}', 'success')
		except Exception as e:
			flash(f'Error resetting password: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/unlock_account/<int:pk>')
	@has_access
	def unlock_account(self, pk):
		"""Unlock user account"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user.account_locked_until = None
			user.failed_login_attempts = 0
			self.datamodel.edit(user)
			flash(f'Account unlocked for user {user.email}', 'success')
		except Exception as e:
			flash(f'Error unlocking account: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/deactivate_user/<int:pk>')
	@has_access
	def deactivate_user(self, pk):
		"""Deactivate user account"""
		user = self.datamodel.get(pk)
		if not user:
			flash('User not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			user.is_active = False
			self.datamodel.edit(user)
			flash(f'User {user.email} has been deactivated', 'success')
		except Exception as e:
			flash(f'Error deactivating user: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new user"""
		# Set default values
		if not item.role:
			item.role = 'viewer'
		if not item.timezone:
			item.timezone = 'UTC'
		if not item.language:
			item.language = 'en'
		if not item.max_concurrent_sessions:
			item.max_concurrent_sessions = 3
		if not item.session_timeout_minutes:
			item.session_timeout_minutes = 480
		
		# Generate default permissions based on role
		item.permissions = self._get_role_permissions(item.role)
	
	def _get_role_permissions(self, role: str) -> List[str]:
		"""Get permissions for user role"""
		permission_map = {
			'super_admin': [
				'platform.admin', 'tenant.manage', 'user.manage', 'system.config',
				'twin.admin', 'data.full_access', 'audit.access', 'backup.manage'
			],
			'tenant_admin': [
				'tenant.admin', 'user.manage', 'twin.admin', 'data.full_access',
				'config.manage', 'audit.view', 'reports.generate'
			],
			'twin_admin': [
				'twin.admin', 'twin.create', 'twin.delete', 'data.read_write',
				'simulation.run', 'alerts.manage'
			],
			'engineer': [
				'twin.read_write', 'data.read_write', 'simulation.run',
				'reports.view', 'alerts.view'
			],
			'analyst': [
				'twin.read', 'data.read', 'reports.view', 'analytics.access',
				'dashboard.view'
			],
			'viewer': [
				'twin.read', 'data.read', 'dashboard.view'
			],
			'guest': [
				'dashboard.view'
			]
		}
		return permission_map.get(role, [])
	
	def _get_user_activity_data(self, user: MTEUser) -> Dict[str, Any]:
		"""Get user activity data"""
		return {
			'login_summary': {
				'total_logins': user.total_login_count,
				'last_login': user.last_login,
				'last_login_ip': user.last_login_ip,
				'failed_attempts': user.failed_login_attempts,
				'account_locked': user.is_account_locked()
			},
			'session_info': {
				'current_session': user.current_session_id,
				'max_sessions': user.max_concurrent_sessions,
				'session_timeout': user.session_timeout_minutes,
				'last_activity': user.last_activity
			},
			'account_status': {
				'is_active': user.is_active,
				'is_verified': user.is_verified,
				'email_verified': user.email_verified_at,
				'force_password_change': user.force_password_change
			},
			'recent_audit_events': []  # Would be populated with user's audit events
		}


class MTESSOConfigurationModelView(ModelView):
	"""SSO configuration management view"""
	
	datamodel = SQLAInterface(MTESSOConfiguration)
	
	# List view configuration
	list_columns = [
		'provider_name', 'tenant', 'provider', 'provider_domain',
		'is_active', 'is_default', 'total_logins', 'last_used'
	]
	show_columns = [
		'config_id', 'tenant', 'provider', 'provider_name', 'provider_domain',
		'client_id', 'metadata_url', 'attribute_mappings', 'role_mappings',
		'is_active', 'is_default', 'auto_provision_users', 'default_role',
		'total_logins', 'last_used', 'configuration_tested_at', 'created_at'
	]
	edit_columns = [
		'provider_name', 'provider', 'provider_domain', 'client_id',
		'metadata_url', 'attribute_mappings', 'role_mappings', 'is_active',
		'is_default', 'auto_provision_users', 'default_role', 'require_ssl'
	]
	add_columns = [
		'tenant', 'provider', 'provider_name', 'provider_domain', 'client_id'
	]
	
	# Search and filtering
	search_columns = ['provider_name', 'provider', 'provider_domain']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('provider_name', 'asc')
	
	# Form validation
	validators_columns = {
		'provider_name': [DataRequired(), Length(min=3, max=200)],
		'provider': [DataRequired()],
		'provider_domain': [DataRequired()],
		'client_id': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'config_id': 'Config ID',
		'tenant_id': 'Tenant ID',
		'provider_name': 'Provider Name',
		'provider_domain': 'Provider Domain',
		'client_id': 'Client ID',
		'client_secret_encrypted': 'Client Secret (Encrypted)',
		'metadata_url': 'Metadata URL',
		'issuer_url': 'Issuer URL',
		'authorization_url': 'Authorization URL',
		'token_url': 'Token URL',
		'userinfo_url': 'User Info URL',
		'attribute_mappings': 'Attribute Mappings',
		'role_mappings': 'Role Mappings',
		'group_attribute_name': 'Group Attribute Name',
		'is_active': 'Active',
		'is_default': 'Default Provider',
		'auto_provision_users': 'Auto Provision Users',
		'auto_update_users': 'Auto Update Users',
		'default_role': 'Default Role',
		'require_ssl': 'Require SSL',
		'verify_ssl_certificates': 'Verify SSL Certificates',
		'signature_algorithm': 'Signature Algorithm',
		'saml_entity_id': 'SAML Entity ID',
		'saml_acs_url': 'SAML ACS URL',
		'saml_sls_url': 'SAML SLS URL',
		'saml_certificate': 'SAML Certificate',
		'total_logins': 'Total Logins',
		'last_used': 'Last Used',
		'configuration_tested_at': 'Configuration Tested',
		'configuration_test_result': 'Test Result'
	}
	
	@expose('/test_configuration/<int:pk>')
	@has_access
	def test_configuration(self, pk):
		"""Test SSO configuration"""
		config = self.datamodel.get(pk)
		if not config:
			flash('SSO configuration not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			test_result = config.test_configuration()
			if test_result:
				flash(f'SSO configuration test successful for {config.provider_name}', 'success')
			else:
				flash(f'SSO configuration test failed for {config.provider_name}', 'error')
		except Exception as e:
			flash(f'Error testing SSO configuration: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/set_default/<int:pk>')
	@has_access
	def set_default(self, pk):
		"""Set as default SSO provider for tenant"""
		config = self.datamodel.get(pk)
		if not config:
			flash('SSO configuration not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Unset other default configurations for this tenant
			for other_config in config.tenant.sso_configurations:
				if other_config.config_id != config.config_id:
					other_config.is_default = False
					self.datamodel.edit(other_config)
			
			# Set this as default
			config.is_default = True
			self.datamodel.edit(config)
			
			flash(f'{config.provider_name} set as default SSO provider', 'success')
		except Exception as e:
			flash(f'Error setting default SSO provider: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new SSO configuration"""
		# Set default values
		if not item.default_role:
			item.default_role = 'viewer'
		if not item.signature_algorithm:
			item.signature_algorithm = 'RS256'
		if not item.group_attribute_name:
			item.group_attribute_name = 'groups'


class MTEAuditEventModelView(ModelView):
	"""Enterprise audit event management view"""
	
	datamodel = SQLAInterface(MTEAuditEvent)
	
	# List view configuration (read-only)
	list_columns = [
		'timestamp', 'tenant', 'user', 'event_type', 'event_category',
		'severity', 'success', 'risk_score'
	]
	show_columns = [
		'event_id', 'tenant', 'user', 'event_type', 'event_category', 'severity',
		'event_name', 'event_description', 'resource_type', 'resource_name',
		'event_data', 'ip_address', 'user_agent', 'success', 'risk_score',
		'compliance_frameworks', 'timestamp'
	]
	
	# No editing of audit events
	can_create = False
	can_edit = False
	can_delete = False
	
	# Search and filtering
	search_columns = ['event_type', 'event_category', 'resource_type', 'event_name']
	base_filters = []
	
	# Ordering
	base_order = ('timestamp', 'desc')
	
	# Custom labels
	label_columns = {
		'event_id': 'Event ID',
		'tenant_id': 'Tenant ID',
		'user_id': 'User ID',
		'event_type': 'Event Type',
		'event_category': 'Event Category',
		'event_name': 'Event Name',
		'event_description': 'Event Description',
		'resource_type': 'Resource Type',
		'resource_id': 'Resource ID',
		'resource_name': 'Resource Name',
		'event_data': 'Event Data',
		'previous_values': 'Previous Values',
		'new_values': 'New Values',
		'ip_address': 'IP Address',
		'user_agent': 'User Agent',
		'session_id': 'Session ID',
		'request_id': 'Request ID',
		'geographic_location': 'Geographic Location',
		'device_type': 'Device Type',
		'device_os': 'Device OS',
		'error_code': 'Error Code',
		'error_message': 'Error Message',
		'response_time_ms': 'Response Time (ms)',
		'risk_score': 'Risk Score',
		'risk_factors': 'Risk Factors',
		'anomaly_score': 'Anomaly Score',
		'compliance_frameworks': 'Compliance Frameworks',
		'retention_period_days': 'Retention Period (Days)',
		'tamper_proof_hash': 'Tamper Proof Hash',
		'processed_at': 'Processed At',
		'archived_at': 'Archived At'
	}
	
	@expose('/event_details/<int:pk>')
	@has_access
	def event_details(self, pk):
		"""View detailed audit event information"""
		event = self.datamodel.get(pk)
		if not event:
			flash('Audit event not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			event_details = self._get_event_details(event)
			
			return render_template('multi_tenant_enterprise/audit_event_details.html',
								   event=event,
								   event_details=event_details,
								   page_title=f"Audit Event: {event.event_name}")
		except Exception as e:
			flash(f'Error loading audit event details: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def _get_event_details(self, event: MTEAuditEvent) -> Dict[str, Any]:
		"""Get detailed event information"""
		return {
			'event_summary': {
				'event_id': event.event_id,
				'timestamp': event.timestamp,
				'event_type': event.event_type,
				'category': event.event_category,
				'severity': event.severity,
				'success': event.success
			},
			'user_context': {
				'user_email': event.user.email if event.user else 'System',
				'tenant_name': event.tenant.organization_name if event.tenant else 'Unknown',
				'ip_address': event.ip_address,
				'user_agent': event.user_agent,
				'session_id': event.session_id
			},
			'resource_info': {
				'resource_type': event.resource_type,
				'resource_id': event.resource_id,
				'resource_name': event.resource_name
			},
			'risk_assessment': {
				'risk_score': event.risk_score,
				'risk_factors': event.risk_factors,
				'is_high_risk': event.is_high_risk(),
				'should_alert': event.should_alert()
			},
			'compliance_info': {
				'frameworks': event.compliance_frameworks,
				'retention_days': event.retention_period_days,
				'tamper_proof_hash': event.tamper_proof_hash
			}
		}


class MultiTenantEnterpriseDashboardView(MultiTenantEnterpriseBaseView):
	"""Multi-tenant enterprise dashboard"""
	
	route_base = "/multi_tenant_enterprise_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Multi-tenant enterprise dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('multi_tenant_enterprise/dashboard.html',
								   metrics=metrics,
								   page_title="Multi-Tenant Enterprise Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('multi_tenant_enterprise/dashboard.html',
								   metrics={},
								   page_title="Multi-Tenant Enterprise Dashboard")
	
	@expose('/tenant_overview/')
	@has_access
	def tenant_overview(self):
		"""Tenant overview and management"""
		try:
			overview_data = self._get_tenant_overview_data()
			
			return render_template('multi_tenant_enterprise/tenant_overview.html',
								   overview_data=overview_data,
								   page_title="Tenant Overview")
		except Exception as e:
			flash(f'Error loading tenant overview: {str(e)}', 'error')
			return redirect(url_for('MultiTenantEnterpriseDashboardView.index'))
	
	@expose('/security_monitoring/')
	@has_access
	def security_monitoring(self):
		"""Security monitoring and audit dashboard"""
		try:
			security_data = self._get_security_monitoring_data()
			
			return render_template('multi_tenant_enterprise/security_monitoring.html',
								   security_data=security_data,
								   page_title="Security Monitoring")
		except Exception as e:
			flash(f'Error loading security monitoring: {str(e)}', 'error')
			return redirect(url_for('MultiTenantEnterpriseDashboardView.index'))
	
	@expose('/compliance_reports/')
	@has_access
	def compliance_reports(self):
		"""Compliance reports and audit trail"""
		try:
			compliance_data = self._get_compliance_reports_data()
			
			return render_template('multi_tenant_enterprise/compliance_reports.html',
								   compliance_data=compliance_data,
								   page_title="Compliance Reports")
		except Exception as e:
			flash(f'Error loading compliance reports: {str(e)}', 'error')
			return redirect(url_for('MultiTenantEnterpriseDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get multi-tenant enterprise dashboard metrics"""
		# Implementation would calculate real metrics from database
		return {
			'platform_overview': {
				'total_tenants': 47,
				'active_tenants': 42,
				'trial_tenants': 5,
				'total_users': 1250,
				'monthly_revenue': 125000
			},
			'tenant_distribution': {
				'basic_tier': 25,
				'professional_tier': 15,
				'enterprise_tier': 6,
				'premium_tier': 1
			},
			'usage_metrics': {
				'total_digital_twins': 15600,
				'storage_used_tb': 2.8,
				'monthly_api_calls': 4500000,
				'average_session_duration': 45
			},
			'security_metrics': {
				'sso_enabled_tenants': 18,
				'high_risk_events_24h': 3,
				'failed_login_attempts': 45,
				'security_incidents': 1
			},
			'financial_metrics': {
				'monthly_recurring_revenue': 125000,
				'average_revenue_per_tenant': 2659,
				'churn_rate': 2.1,
				'growth_rate': 15.3
			}
		}
	
	def _get_tenant_overview_data(self) -> Dict[str, Any]:
		"""Get tenant overview data"""
		return {
			'tenant_summary': {
				'total_tenants': 47,
				'active_tenants': 42,
				'suspended_tenants': 3,
				'trial_tenants': 5,
				'new_tenants_month': 6
			},
			'subscription_breakdown': {
				'basic': {'count': 25, 'revenue': 7475},
				'professional': {'count': 15, 'revenue': 22425},
				'enterprise': {'count': 6, 'revenue': 71940},
				'premium': {'count': 1, 'revenue': 24999}
			},
			'usage_trends': {
				'user_growth': [1150, 1180, 1205, 1230, 1250],
				'storage_growth': [2.1, 2.3, 2.5, 2.7, 2.8],
				'api_growth': [3800000, 4100000, 4200000, 4350000, 4500000]
			},
			'top_tenants': [
				{
					'name': 'Global Manufacturing Corp',
					'users': 125,
					'digital_twins': 2500,
					'storage_gb': 450,
					'monthly_cost': 24999
				},
				{
					'name': 'Smart City Solutions',
					'users': 89,
					'digital_twins': 1800,
					'storage_gb': 320,
					'monthly_cost': 11990
				}
			]
		}
	
	def _get_security_monitoring_data(self) -> Dict[str, Any]:
		"""Get security monitoring data"""
		return {
			'threat_overview': {
				'high_risk_events_24h': 3,
				'failed_logins_24h': 45,
				'suspicious_ips': 8,
				'security_incidents': 1
			},
			'authentication_metrics': {
				'sso_logins_24h': 1250,
				'local_logins_24h': 180,
				'failed_authentications': 45,
				'account_lockouts': 5
			},
			'access_patterns': {
				'unusual_login_times': 12,
				'new_device_logins': 23,
				'geographic_anomalies': 4,
				'privilege_escalations': 2
			},
			'recent_incidents': [
				{
					'timestamp': '2024-01-15 14:30:00',
					'type': 'Multiple Failed Logins',
					'user': 'user@company.com',
					'risk_score': 8.5,
					'status': 'investigating'
				},
				{
					'timestamp': '2024-01-15 09:15:00',
					'type': 'Unusual Data Export',
					'user': 'analyst@corp.com',
					'risk_score': 7.2,
					'status': 'resolved'
				}
			]
		}
	
	def _get_compliance_reports_data(self) -> Dict[str, Any]:
		"""Get compliance reports data"""
		return {
			'compliance_status': {
				'soc2_compliant_tenants': 24,
				'gdpr_compliant_tenants': 18,
				'hipaa_compliant_tenants': 8,
				'iso27001_compliant_tenants': 12
			},
			'audit_statistics': {
				'total_events_30d': 125000,
				'high_risk_events_30d': 45,
				'compliance_violations_30d': 2,
				'audit_trail_integrity': 99.98
			},
			'recent_reports': [
				{
					'framework': 'SOC2',
					'tenant': 'Enterprise Corp',
					'generated_date': '2024-01-10',
					'status': 'approved',
					'findings': 0
				},
				{
					'framework': 'GDPR',
					'tenant': 'EU Industries',
					'generated_date': '2024-01-08',
					'status': 'under_review',
					'findings': 1
				}
			],
			'compliance_trends': {
				'monthly_reports_generated': [45, 52, 48, 56, 61],
				'compliance_score_avg': [92.5, 93.1, 94.2, 93.8, 94.5],
				'violation_trends': [8, 6, 4, 3, 2]
			}
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all multi-tenant enterprise views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		MTETenantModelView,
		"Tenants",
		icon="fa-building",
		category="Multi-Tenant Enterprise",
		category_icon="fa-users-cog"
	)
	
	appbuilder.add_view(
		MTEUserModelView,
		"Enterprise Users",
		icon="fa-users",
		category="Multi-Tenant Enterprise"
	)
	
	appbuilder.add_view(
		MTESSOConfigurationModelView,
		"SSO Configuration",
		icon="fa-key",
		category="Multi-Tenant Enterprise"
	)
	
	appbuilder.add_view(
		MTEAuditEventModelView,
		"Audit Events",
		icon="fa-clipboard-list",
		category="Multi-Tenant Enterprise"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(MultiTenantEnterpriseDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"Enterprise Dashboard",
		href="/multi_tenant_enterprise_dashboard/",
		icon="fa-dashboard",
		category="Multi-Tenant Enterprise"
	)
	
	appbuilder.add_link(
		"Tenant Overview",
		href="/multi_tenant_enterprise_dashboard/tenant_overview/",
		icon="fa-chart-pie",
		category="Multi-Tenant Enterprise"
	)
	
	appbuilder.add_link(
		"Security Monitoring",
		href="/multi_tenant_enterprise_dashboard/security_monitoring/",
		icon="fa-shield-alt",
		category="Multi-Tenant Enterprise"
	)
	
	appbuilder.add_link(
		"Compliance Reports",
		href="/multi_tenant_enterprise_dashboard/compliance_reports/",
		icon="fa-file-contract",
		category="Multi-Tenant Enterprise"
	)