"""
Profile Management Views

Flask-AppBuilder views for user profile management and registration
with comprehensive CRUD operations, search, and GDPR compliance features.
"""

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, validators
from wtforms.validators import DataRequired, Email, Length, Optional
from datetime import datetime
from typing import Dict, Any, List

from .models import PMUser, PMProfile, PMConsent, PMRegistration, PMPreferences
from .services import ProfileService, RegistrationService, ConsentService, RegistrationRequest, ProfileUpdateRequest
from .events import get_profile_event_emitter
from .exceptions import ProfileManagementError, RegistrationError, ConsentError


class ProfileManagementBaseView(BaseView):
	"""Base view for profile management functionality"""
	
	def __init__(self):
		super().__init__()
		self.profile_service = None
		self.registration_service = None
		self.consent_service = None
		self.event_emitter = get_profile_event_emitter()
	
	def _get_services(self):
		"""Initialize services with current database session"""
		if not self.profile_service:
			from flask_appbuilder import db
			self.profile_service = ProfileService(db.session, self.event_emitter)
			self.registration_service = RegistrationService(db.session, self.event_emitter)
			self.consent_service = ConsentService(db.session, self.event_emitter)


class PMUserModelView(ModelView):
	"""User management view with comprehensive CRUD operations"""
	
	datamodel = SQLAInterface(PMUser)
	
	# List view configuration
	list_columns = [
		'email', 'username', 'tenant_id', 'registration_source',
		'email_verified', 'is_active', 'last_login_at', 'created_on'
	]
	
	# Search configuration
	search_columns = ['email', 'username', 'tenant_id', 'registration_source']
	
	# Form configuration
	add_columns = [
		'email', 'username', 'tenant_id', 'registration_source',
		'data_processing_consent', 'marketing_consent', 'analytics_consent'
	]
	
	edit_columns = [
		'email', 'username', 'is_active', 'email_verified',
		'data_processing_consent', 'marketing_consent', 'analytics_consent',
		'gdpr_deletion_requested'
	]
	
	show_columns = [
		'user_id', 'email', 'username', 'tenant_id', 'registration_source',
		'email_verified', 'is_active', 'last_login_at', 'failed_login_attempts',
		'data_processing_consent', 'marketing_consent', 'analytics_consent',
		'gdpr_deletion_requested', 'gdpr_deletion_scheduled', 'created_on', 'changed_on'
	]
	
	# Order and pagination
	base_order = ('created_on', 'desc')
	page_size = 20
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Labels
	label_columns = {
		'user_id': 'User ID',
		'email': 'Email Address',
		'username': 'Username',
		'tenant_id': 'Tenant',
		'registration_source': 'Registration Source',
		'email_verified': 'Email Verified',
		'is_active': 'Active',
		'last_login_at': 'Last Login',
		'failed_login_attempts': 'Failed Logins',
		'data_processing_consent': 'Data Processing Consent',
		'marketing_consent': 'Marketing Consent',
		'analytics_consent': 'Analytics Consent',
		'gdpr_deletion_requested': 'GDPR Deletion Requested',
		'gdpr_deletion_scheduled': 'GDPR Deletion Scheduled',
		'created_on': 'Created',
		'changed_on': 'Last Modified'
	}
	
	@expose('/verify_email/<user_id>')
	@protect()
	def verify_email_action(self, user_id):
		"""Admin action to manually verify user email"""
		try:
			from flask_appbuilder import db
			user = db.session.query(PMUser).filter(PMUser.user_id == user_id).first()
			
			if user:
				user.email_verified = True
				user.email_verification_token = None
				user.email_verification_expires = None
				db.session.commit()
				
				# Emit verification event
				get_profile_event_emitter().emit('user.email_verified', {
					'user_id': user_id,
					'email': user.email,
					'tenant_id': user.tenant_id,
					'verification_method': 'admin_override'
				})
				
				flash(f'Email verified for user {user.email}', 'success')
			else:
				flash('User not found', 'error')
				
		except Exception as e:
			flash(f'Failed to verify email: {str(e)}', 'error')
		
		return redirect(url_for('PMUserModelView.list'))
	
	@expose('/gdpr_delete/<user_id>')
	@protect()
	def gdpr_delete_action(self, user_id):
		"""Admin action to perform GDPR deletion"""
		try:
			from flask_appbuilder import db
			profile_service = ProfileService(db.session, get_profile_event_emitter())
			
			# Get current user ID for audit
			current_user_id = str(self.appbuilder.sm.user.id) if self.appbuilder.sm.user else 'admin'
			
			success = profile_service.delete_profile(
				user_id=user_id,
				deletion_type='gdpr',
				requestor_id=current_user_id
			)
			
			if success:
				flash('User data anonymized per GDPR requirements', 'success')
			else:
				flash('GDPR deletion failed', 'error')
				
		except Exception as e:
			flash(f'GDPR deletion failed: {str(e)}', 'error')
		
		return redirect(url_for('PMUserModelView.list'))


class PMProfileModelView(ModelView):
	"""Profile management view with privacy controls"""
	
	datamodel = SQLAInterface(PMProfile)
	
	# List view configuration
	list_columns = [
		'user.email', 'first_name', 'last_name', 'display_name',
		'company', 'job_title', 'completion_score', 'verification_level'
	]
	
	# Search configuration
	search_columns = [
		'first_name', 'last_name', 'display_name', 'company',
		'job_title', 'department', 'country'
	]
	
	# Form configuration
	add_columns = [
		'user_id', 'tenant_id', 'first_name', 'last_name', 'display_name',
		'title', 'company', 'job_title', 'department', 'bio',
		'phone_primary', 'website_url', 'linkedin_url', 'country',
		'city', 'timezone', 'locale', 'profile_visibility',
		'contact_visibility', 'search_visibility'
	]
	
	edit_columns = [
		'first_name', 'last_name', 'display_name', 'title',
		'company', 'job_title', 'department', 'bio', 'phone_primary',
		'phone_secondary', 'website_url', 'linkedin_url', 'twitter_handle',
		'github_username', 'country', 'city', 'timezone', 'locale',
		'profile_visibility', 'contact_visibility', 'search_visibility'
	]
	
	show_columns = [
		'user.email', 'first_name', 'last_name', 'display_name',
		'title', 'company', 'job_title', 'department', 'bio',
		'phone_primary', 'phone_secondary', 'website_url',
		'linkedin_url', 'twitter_handle', 'github_username',
		'country', 'city', 'timezone', 'locale', 'completion_score',
		'verification_level', 'profile_visibility', 'contact_visibility',
		'search_visibility', 'created_on', 'changed_on'
	]
	
	# Order and pagination
	base_order = ('changed_on', 'desc')
	page_size = 20
	
	# Labels
	label_columns = {
		'user.email': 'User Email',
		'first_name': 'First Name',
		'last_name': 'Last Name',
		'display_name': 'Display Name',
		'title': 'Title',
		'company': 'Company',
		'job_title': 'Job Title',
		'department': 'Department',
		'bio': 'Biography',
		'phone_primary': 'Primary Phone',
		'phone_secondary': 'Secondary Phone',
		'website_url': 'Website',
		'linkedin_url': 'LinkedIn Profile',
		'twitter_handle': 'Twitter Handle',
		'github_username': 'GitHub Username',
		'country': 'Country',
		'city': 'City',
		'timezone': 'Timezone',
		'locale': 'Locale',
		'completion_score': 'Completion %',
		'verification_level': 'Verification Level',
		'profile_visibility': 'Profile Visibility',
		'contact_visibility': 'Contact Visibility',
		'search_visibility': 'Search Visibility',
		'created_on': 'Created',
		'changed_on': 'Last Updated'
	}
	
	@expose('/calculate_completion/<profile_id>')
	@protect()
	def calculate_completion_action(self, profile_id):
		"""Recalculate profile completion score"""
		try:
			from flask_appbuilder import db
			profile = db.session.query(PMProfile).filter(PMProfile.profile_id == profile_id).first()
			
			if profile:
				profile.calculate_completion_score()
				db.session.commit()
				flash(f'Completion score recalculated: {profile.completion_score}%', 'success')
			else:
				flash('Profile not found', 'error')
				
		except Exception as e:
			flash(f'Failed to calculate completion: {str(e)}', 'error')
		
		return redirect(url_for('PMProfileModelView.list'))


class PMRegistrationModelView(ModelView):
	"""Registration tracking and analytics view"""
	
	datamodel = SQLAInterface(PMRegistration)
	
	# List view configuration
	list_columns = [
		'email', 'tenant_id', 'registration_source', 'registration_step',
		'status', 'email_verification_sent', 'email_verified_at', 'created_on'
	]
	
	# Search configuration
	search_columns = ['email', 'tenant_id', 'registration_source', 'status']
	
	# Show only - registrations should not be directly editable
	show_columns = [
		'registration_id', 'email', 'tenant_id', 'registration_source',
		'registration_step', 'status', 'ip_address', 'user_agent',
		'referrer_url', 'utm_source', 'utm_medium', 'utm_campaign',
		'email_verification_sent', 'email_verified_at', 'completion_time',
		'failure_reason', 'created_on', 'changed_on'
	]
	
	# Read-only view
	base_permissions = ['can_list', 'can_show']
	
	# Order and pagination
	base_order = ('created_on', 'desc')
	page_size = 25
	
	# Labels
	label_columns = {
		'registration_id': 'Registration ID',
		'email': 'Email',
		'tenant_id': 'Tenant',
		'registration_source': 'Source',
		'registration_step': 'Step',
		'status': 'Status',
		'ip_address': 'IP Address',
		'user_agent': 'User Agent',
		'referrer_url': 'Referrer',
		'utm_source': 'UTM Source',
		'utm_medium': 'UTM Medium',
		'utm_campaign': 'UTM Campaign',
		'email_verification_sent': 'Verification Sent',
		'email_verified_at': 'Verified At',
		'completion_time': 'Completion Time (sec)',
		'failure_reason': 'Failure Reason',
		'created_on': 'Started',
		'changed_on': 'Last Updated'
	}


class PMConsentModelView(ModelView):
	"""GDPR consent management view"""
	
	datamodel = SQLAInterface(PMConsent)
	
	# List view configuration
	list_columns = [
		'user.email', 'purpose', 'granted', 'consent_version',
		'consent_method', 'withdrawn_at', 'expires_at', 'created_on'
	]
	
	# Search configuration
	search_columns = ['user.email', 'purpose', 'consent_method']
	
	# Form configuration - limited editing for compliance
	add_columns = [
		'user_id', 'tenant_id', 'purpose', 'granted',
		'consent_version', 'consent_method', 'retention_period'
	]
	
	edit_columns = [
		'granted', 'withdrawal_reason'  # Only allow withdrawal
	]
	
	show_columns = [
		'consent_id', 'user.email', 'tenant_id', 'purpose', 'granted',
		'consent_version', 'consent_method', 'ip_address', 'user_agent',
		'source_page', 'retention_period', 'expires_at', 'withdrawn_at',
		'withdrawal_method', 'withdrawal_reason', 'created_on', 'changed_on'
	]
	
	# Order and pagination
	base_order = ('created_on', 'desc')
	page_size = 25
	
	# Labels
	label_columns = {
		'consent_id': 'Consent ID',
		'user.email': 'User Email',
		'tenant_id': 'Tenant',
		'purpose': 'Purpose',
		'granted': 'Granted',
		'consent_version': 'Policy Version',
		'consent_method': 'Method',
		'ip_address': 'IP Address',
		'user_agent': 'User Agent',
		'source_page': 'Source Page',
		'retention_period': 'Retention (days)',
		'expires_at': 'Expires',
		'withdrawn_at': 'Withdrawn',
		'withdrawal_method': 'Withdrawal Method',
		'withdrawal_reason': 'Withdrawal Reason',
		'created_on': 'Created',
		'changed_on': 'Last Updated'
	}
	
	@expose('/withdraw_consent/<consent_id>')
	@protect()
	def withdraw_consent_action(self, consent_id):
		"""Admin action to withdraw consent"""
		try:
			from flask_appbuilder import db
			consent = db.session.query(PMConsent).filter(PMConsent.consent_id == consent_id).first()
			
			if consent and not consent.withdrawn_at:
				consent.withdraw('admin_action', 'Withdrawn by administrator')
				db.session.commit()
				
				# Emit withdrawal event
				get_profile_event_emitter().emit('consent.withdrawn', {
					'consent_id': consent_id,
					'user_id': consent.user_id,
					'purpose': consent.purpose,
					'withdrawal_method': 'admin_action',
					'withdrawal_reason': 'Withdrawn by administrator'
				})
				
				flash('Consent withdrawn successfully', 'success')
			else:
				flash('Consent not found or already withdrawn', 'error')
				
		except Exception as e:
			flash(f'Failed to withdraw consent: {str(e)}', 'error')
		
		return redirect(url_for('PMConsentModelView.list'))


class UserRegistrationView(ProfileManagementBaseView):
	"""User registration form and workflow"""
	
	route_base = '/profile/register'
	
	@expose('/')
	def index(self):
		"""Registration form"""
		return self.render_template('profile_management/register.html')
	
	@expose('/api', methods=['POST'])
	def register_user(self):
		"""API endpoint for user registration"""
		try:
			self._get_services()
			
			# Get form data
			data = request.get_json() or request.form.to_dict()
			
			# Create registration request
			reg_request = RegistrationRequest(
				email=data.get('email', '').strip().lower(),
				password=data.get('password'),
				username=data.get('username'),
				first_name=data.get('first_name'),
				last_name=data.get('last_name'),
				tenant_id=data.get('tenant_id', 'default'),
				registration_source=data.get('source', 'web_form'),
				consents={
					'data_processing': data.get('data_processing_consent', False),
					'marketing': data.get('marketing_consent', False),
					'analytics': data.get('analytics_consent', False),
					'email_notifications': data.get('email_notifications', True)
				},
				profile_data={
					'first_name': data.get('first_name'),
					'last_name': data.get('last_name'),
					'company': data.get('company'),
					'job_title': data.get('job_title'),
					'timezone': data.get('timezone', 'UTC'),
					'locale': data.get('locale', 'en-US')
				},
				metadata={
					'ip_address': request.remote_addr,
					'user_agent': request.headers.get('User-Agent'),
					'referrer_url': request.referrer,
					'utm_source': data.get('utm_source'),
					'utm_medium': data.get('utm_medium'),
					'utm_campaign': data.get('utm_campaign')
				}
			)
			
			# Start registration process
			registration_id = self.registration_service.start_registration(reg_request)
			
			# Complete registration
			result = self.registration_service.complete_registration(
				registration_id=registration_id,
				send_verification=data.get('send_verification', True)
			)
			
			return jsonify({
				'success': True,
				'user_id': result['user_id'],
				'registration_id': result['registration_id'],
				'verification_required': bool(result.get('verification_token')),
				'profile_completion': result['profile_completion']
			})
			
		except (RegistrationError, ProfileManagementError) as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
		except Exception as e:
			return jsonify({
				'success': False,
				'error': 'Registration failed due to internal error'
			}), 500
	
	@expose('/verify/<token>')
	def verify_email(self, token):
		"""Email verification endpoint"""
		try:
			self._get_services()
			
			success = self.registration_service.verify_email(token)
			
			if success:
				flash('Email verified successfully! You can now log in.', 'success')
			else:
				flash('Invalid or expired verification link.', 'error')
			
		except Exception as e:
			flash('Email verification failed.', 'error')
		
		return redirect(url_for('AuthDBView.login'))


class ProfileSearchView(ProfileManagementBaseView):
	"""Profile search and discovery"""
	
	route_base = '/profile/search'
	
	@expose('/')
	@protect()
	def index(self):
		"""Profile search interface"""
		return self.render_template('profile_management/search.html')
	
	@expose('/api', methods=['GET'])
	@protect()
	def search_profiles(self):
		"""API endpoint for profile search"""
		try:
			self._get_services()
			
			# Get search parameters
			query = request.args.get('q', '')
			tenant_id = request.args.get('tenant_id', 'default')
			limit = int(request.args.get('limit', 20))
			offset = int(request.args.get('offset', 0))
			
			# Build filters
			filters = {}
			if request.args.get('department'):
				filters['department'] = request.args.get('department')
			if request.args.get('company'):
				filters['company'] = request.args.get('company')
			if request.args.get('country'):
				filters['country'] = request.args.get('country')
			if request.args.get('verification_level'):
				filters['verification_level'] = request.args.get('verification_level')
			
			# Get current user ID for privacy filtering
			viewer_user_id = str(self.appbuilder.sm.user.id) if self.appbuilder.sm.user else None
			
			# Perform search
			results = self.profile_service.search_profiles(
				tenant_id=tenant_id,
				query=query,
				filters=filters,
				viewer_user_id=viewer_user_id,
				limit=limit,
				offset=offset
			)
			
			return jsonify(results)
			
		except Exception as e:
			return jsonify({
				'error': 'Search failed',
				'message': str(e)
			}), 500


class ProfileDashboardView(ProfileManagementBaseView):
	"""User profile dashboard and management"""
	
	route_base = '/profile/dashboard'
	
	@expose('/')
	@protect()
	def index(self):
		"""Profile dashboard"""
		return self.render_template('profile_management/dashboard.html')
	
	@expose('/api/profile', methods=['GET'])
	@protect()
	def get_profile(self):
		"""Get current user's profile"""
		try:
			self._get_services()
			
			user_id = str(self.appbuilder.sm.user.id)
			profile = self.profile_service.get_profile(user_id, user_id)
			
			if profile:
				return jsonify(profile)
			else:
				return jsonify({'error': 'Profile not found'}), 404
				
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	@expose('/api/profile', methods=['PUT'])
	@protect()
	def update_profile(self):
		"""Update current user's profile"""
		try:
			self._get_services()
			
			user_id = str(self.appbuilder.sm.user.id)
			updates = request.get_json()
			
			# Create update request
			update_request = ProfileUpdateRequest(
				user_id=user_id,
				updates=updates,
				updated_by=user_id,
				tenant_id='default',  # Get from user context
				validate_permissions=False  # User updating own profile
			)
			
			# Update profile
			updated_profile = self.profile_service.update_profile(update_request)
			
			return jsonify({
				'success': True,
				'profile': updated_profile
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/api/export', methods=['POST'])
	@protect()
	def export_data(self):
		"""Export user's personal data (GDPR)"""
		try:
			self._get_services()
			
			user_id = str(self.appbuilder.sm.user.id)
			export_data = self.profile_service.export_profile_data(user_id, user_id)
			
			return jsonify(export_data)
			
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	@expose('/api/delete', methods=['POST'])
	@protect()
	def request_deletion(self):
		"""Request GDPR deletion of user data"""
		try:
			self._get_services()
			
			user_id = str(self.appbuilder.sm.user.id)
			success = self.profile_service.delete_profile(
				user_id=user_id,
				deletion_type='gdpr',
				requestor_id=user_id
			)
			
			if success:
				return jsonify({'success': True, 'message': 'Data deletion completed'})
			else:
				return jsonify({'success': False, 'error': 'Deletion failed'}), 400
				
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 400


class RegistrationAnalyticsView(ProfileManagementBaseView):
	"""Registration analytics and reporting"""
	
	route_base = '/profile/analytics'
	
	@expose('/')
	@protect()
	def index(self):
		"""Analytics dashboard"""
		return self.render_template('profile_management/analytics.html')
	
	@expose('/api/registration', methods=['GET'])
	@protect()
	def registration_analytics(self):
		"""Get registration analytics"""
		try:
			self._get_services()
			
			tenant_id = request.args.get('tenant_id', 'default')
			days = int(request.args.get('days', 30))
			
			analytics = self.registration_service.get_registration_analytics(tenant_id, days)
			
			return jsonify(analytics)
			
		except Exception as e:
			return jsonify({'error': str(e)}), 500


def register_profile_views(appbuilder):
	"""
	Register all profile management views with Flask-AppBuilder.
	
	This function should be called during application initialization
	to register all profile management views and create the menu structure.
	
	Args:
		appbuilder: Flask-AppBuilder instance
	"""
	
	# Register ModelViews for data management
	appbuilder.add_view(
		PMUserModelView,
		"Users",
		icon="fa-users",
		category="Profile Management",
		category_icon="fa-user-circle"
	)
	
	appbuilder.add_view(
		PMProfileModelView,
		"User Profiles",
		icon="fa-id-card",
		category="Profile Management"
	)
	
	appbuilder.add_view(
		PMRegistrationModelView,
		"Registration Tracking",
		icon="fa-user-plus",
		category="Profile Management"
	)
	
	appbuilder.add_view(
		PMConsentModelView,
		"GDPR Consents",
		icon="fa-shield-alt",
		category="Profile Management"
	)
	
	# Register functional views
	appbuilder.add_view_no_menu(UserRegistrationView, "UserRegistration")
	appbuilder.add_view_no_menu(ProfileSearchView, "ProfileSearch")
	appbuilder.add_view_no_menu(ProfileDashboardView, "ProfileDashboard")
	appbuilder.add_view_no_menu(RegistrationAnalyticsView, "RegistrationAnalytics")
	
	# Add menu links for functional views
	appbuilder.add_link(
		"Registration Form",
		href="/profile/register/",
		icon="fa-user-plus",
		category="Profile Management"
	)
	
	appbuilder.add_link(
		"Search Profiles",
		href="/profile/search/",
		icon="fa-search",
		category="Profile Management"
	)
	
	appbuilder.add_link(
		"My Profile",
		href="/profile/dashboard/",
		icon="fa-user",
		category="Profile Management"
	)
	
	appbuilder.add_link(
		"Registration Analytics",
		href="/profile/analytics/",
		icon="fa-chart-bar",
		category="Profile Management"
	)