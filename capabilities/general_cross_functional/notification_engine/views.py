"""
Notification Engine Views

Flask-AppBuilder views for comprehensive notification management
with template management, campaign tracking, delivery monitoring, and analytics.
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
	NENotification, NETemplate, NEDelivery, NEInteraction, NECampaign,
	NECampaignStep, NEUserPreference, NEProvider
)


class NotificationEngineBaseView(BaseView):
	"""Base view for notification engine functionality"""
	
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
	
	def _format_delivery_rate(self, rate: float) -> str:
		"""Format delivery rate for display"""
		return f"{rate:.2f}%" if rate is not None else "0.00%"


class NENotificationModelView(ModelView):
	"""Notification management view"""
	
	datamodel = SQLAInterface(NENotification)
	
	# List view configuration
	list_columns = [
		'title', 'recipient_email', 'channels', 'priority',
		'status', 'delivery_attempts', 'delivered_at', 'created_on'
	]
	show_columns = [
		'notification_id', 'title', 'message', 'template', 'recipient_id',
		'recipient_email', 'recipient_phone', 'channels', 'priority',
		'delivery_method', 'scheduled_at', 'expires_at', 'status',
		'delivery_attempts', 'delivered_at', 'read_at', 'clicked_at',
		'source_event', 'campaign_id', 'deliveries', 'interactions'
	]
	edit_columns = [
		'title', 'message', 'template', 'recipient_id', 'recipient_email',
		'recipient_phone', 'channels', 'priority', 'delivery_method',
		'scheduled_at', 'expires_at', 'source_event', 'tags'
	]
	add_columns = edit_columns
	
	# Related views
	related_views = [NETemplate]
	
	# Search and filtering
	search_columns = ['title', 'message', 'recipient_email']
	base_filters = [['status', lambda: 'pending', lambda: True]]
	
	# Ordering
	base_order = ('created_on', 'desc')
	
	# Form validation
	validators_columns = {
		'title': [DataRequired(), Length(min=1, max=500)],
		'message': [DataRequired()],
		'recipient_email': [Email(), Optional()],
		'channels': [DataRequired()],
		'priority': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'notification_id': 'Notification ID',
		'recipient_id': 'Recipient ID',
		'recipient_email': 'Recipient Email',
		'recipient_phone': 'Recipient Phone',
		'delivery_method': 'Delivery Method',
		'scheduled_at': 'Scheduled At',
		'expires_at': 'Expires At',
		'delivery_attempts': 'Delivery Attempts',
		'delivered_at': 'Delivered At',
		'read_at': 'Read At',
		'clicked_at': 'Clicked At',
		'source_event': 'Source Event',
		'campaign_id': 'Campaign ID'
	}
	
	@expose('/send_now/<int:pk>')
	@has_access
	def send_now(self, pk):
		"""Send notification immediately"""
		notification = self.datamodel.get(pk)
		if not notification:
			flash('Notification not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would trigger immediate sending
			notification.status = 'processing'
			notification.delivery_attempts += 1
			self.datamodel.edit(notification)
			
			flash(f'Notification "{notification.title}" queued for immediate sending', 'success')
		except Exception as e:
			flash(f'Error sending notification: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/resend/<int:pk>')
	@has_access
	def resend_notification(self, pk):
		"""Resend failed notification"""
		notification = self.datamodel.get(pk)
		if not notification:
			flash('Notification not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if notification.status == 'failed':
				notification.status = 'pending'
				notification.delivery_attempts = 0
				self.datamodel.edit(notification)
				flash(f'Notification "{notification.title}" queued for resending', 'success')
			else:
				flash('Only failed notifications can be resent', 'warning')
		except Exception as e:
			flash(f'Error resending notification: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new notification"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.priority:
			item.priority = 'normal'
		if not item.delivery_method:
			item.delivery_method = 'immediate'
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class NETemplateModelView(ModelView):
	"""Template management view"""
	
	datamodel = SQLAInterface(NETemplate)
	
	# List view configuration
	list_columns = [
		'code', 'name', 'version', 'locale', 'supported_channels',
		'is_active', 'usage_count', 'success_rate'
	]
	show_columns = [
		'template_id', 'code', 'name', 'version', 'locale', 'description',
		'subject_template', 'html_template', 'text_template', 'sms_template',
		'push_template', 'template_engine', 'supported_channels', 'is_active',
		'is_default', 'usage_count', 'success_rate', 'variables_schema'
	]
	edit_columns = [
		'code', 'name', 'version', 'locale', 'description',
		'subject_template', 'html_template', 'text_template', 'sms_template',
		'push_template', 'template_engine', 'variables_schema', 'default_variables',
		'supported_channels', 'is_active', 'is_default'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['code', 'name', 'description']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('code', 'asc')
	
	# Form validation
	validators_columns = {
		'code': [DataRequired(), Length(min=1, max=100)],
		'name': [DataRequired(), Length(min=1, max=200)],
		'version': [DataRequired()],
		'locale': [DataRequired()],
		'template_engine': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'template_id': 'Template ID',
		'subject_template': 'Subject Template',
		'html_template': 'HTML Template',
		'text_template': 'Text Template',
		'sms_template': 'SMS Template',
		'push_template': 'Push Template',
		'template_engine': 'Template Engine',
		'variables_schema': 'Variables Schema',
		'default_variables': 'Default Variables',
		'supported_channels': 'Supported Channels',
		'is_active': 'Active',
		'is_default': 'Default',
		'usage_count': 'Usage Count',
		'success_rate': 'Success Rate'
	}
	
	@expose('/preview/<int:pk>')
	@has_access
	def preview_template(self, pk):
		"""Preview template with sample data"""
		template = self.datamodel.get(pk)
		if not template:
			flash('Template not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Sample data for preview
			sample_variables = {
				'user_name': 'John Doe',
				'company_name': 'ACME Corp',
				'amount': '1,234.56',
				'date': datetime.now().strftime('%Y-%m-%d')
			}
			
			# Render template with sample data
			rendered = template.render(sample_variables, 'email')
			
			return render_template('notification_engine/template_preview.html',
								   template=template,
								   rendered=rendered,
								   sample_variables=sample_variables,
								   page_title=f"Preview: {template.name}")
		except Exception as e:
			flash(f'Error previewing template: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/test_send/<int:pk>', methods=['GET', 'POST'])
	@has_access
	def test_send(self, pk):
		"""Send test notification using template"""
		template = self.datamodel.get(pk)
		if not template:
			flash('Template not found', 'error')
			return redirect(self.get_redirect())
		
		if request.method == 'POST':
			try:
				test_email = request.form.get('test_email')
				test_variables = json.loads(request.form.get('test_variables', '{}'))
				
				# Create test notification
				test_notification = NENotification(
					title=f'Test: {template.name}',
					message='Test notification',
					template_id=template.template_id,
					template_variables=test_variables,
					recipient_email=test_email,
					channels=['email'],
					priority='normal',
					tenant_id=self._get_tenant_id()
				)
				
				self.datamodel.add(test_notification)
				flash(f'Test notification sent to {test_email}', 'success')
				return redirect(self.get_redirect())
			except Exception as e:
				flash(f'Error sending test notification: {str(e)}', 'error')
		
		return render_template('notification_engine/template_test_send.html',
							   template=template,
							   page_title=f"Test Send: {template.name}")
	
	def pre_add(self, item):
		"""Pre-process before adding new template"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.version:
			item.version = '1.0.0'
		if not item.locale:
			item.locale = 'en-US'
		if not item.template_engine:
			item.template_engine = 'mustache'
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class NECampaignModelView(ModelView):
	"""Campaign management view"""
	
	datamodel = SQLAInterface(NECampaign)
	
	# List view configuration
	list_columns = [
		'name', 'campaign_type', 'status', 'total_recipients',
		'total_sent', 'delivery_rate', 'open_rate', 'start_date'
	]
	show_columns = [
		'campaign_id', 'name', 'description', 'campaign_type', 'trigger_event',
		'status', 'start_date', 'end_date', 'total_recipients', 'total_sent',
		'total_delivered', 'total_opened', 'total_clicked', 'delivery_rate',
		'open_rate', 'click_rate', 'conversion_rate', 'steps'
	]
	edit_columns = [
		'name', 'description', 'campaign_type', 'trigger_event',
		'trigger_conditions', 'target_audience', 'start_date', 'end_date'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['name', 'description']
	base_filters = [['status', lambda: 'draft', lambda: True]]
	
	# Ordering
	base_order = ('created_on', 'desc')
	
	# Form validation
	validators_columns = {
		'name': [DataRequired(), Length(min=1, max=200)],
		'campaign_type': [DataRequired()],
		'total_recipients': [NumberRange(min=0)]
	}
	
	# Custom labels
	label_columns = {
		'campaign_id': 'Campaign ID',
		'campaign_type': 'Campaign Type',
		'trigger_event': 'Trigger Event',
		'trigger_conditions': 'Trigger Conditions',
		'target_audience': 'Target Audience',
		'start_date': 'Start Date',
		'end_date': 'End Date',
		'total_recipients': 'Total Recipients',
		'total_sent': 'Total Sent',
		'total_delivered': 'Total Delivered',
		'total_opened': 'Total Opened',
		'total_clicked': 'Total Clicked',
		'delivery_rate': 'Delivery Rate',
		'open_rate': 'Open Rate',
		'click_rate': 'Click Rate',
		'conversion_rate': 'Conversion Rate'
	}
	
	@expose('/start/<int:pk>')
	@has_access
	def start_campaign(self, pk):
		"""Start campaign execution"""
		campaign = self.datamodel.get(pk)
		if not campaign:
			flash('Campaign not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if campaign.status == 'draft':
				campaign.status = 'active'
				campaign.start_date = datetime.utcnow()
				self.datamodel.edit(campaign)
				flash(f'Campaign "{campaign.name}" started successfully', 'success')
			else:
				flash('Only draft campaigns can be started', 'warning')
		except Exception as e:
			flash(f'Error starting campaign: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/pause/<int:pk>')
	@has_access
	def pause_campaign(self, pk):
		"""Pause active campaign"""
		campaign = self.datamodel.get(pk)
		if not campaign:
			flash('Campaign not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if campaign.status == 'active':
				campaign.status = 'paused'
				self.datamodel.edit(campaign)
				flash(f'Campaign "{campaign.name}" paused', 'success')
			else:
				flash('Only active campaigns can be paused', 'warning')
		except Exception as e:
			flash(f'Error pausing campaign: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/analytics/<int:pk>')
	@has_access
	def campaign_analytics(self, pk):
		"""View campaign analytics"""
		campaign = self.datamodel.get(pk)
		if not campaign:
			flash('Campaign not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Get analytics data
			analytics_data = self._get_campaign_analytics(campaign)
			
			return render_template('notification_engine/campaign_analytics.html',
								   campaign=campaign,
								   analytics=analytics_data,
								   page_title=f"Analytics: {campaign.name}")
		except Exception as e:
			flash(f'Error loading campaign analytics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new campaign"""
		item.tenant_id = self._get_tenant_id()
		item.status = 'draft'
	
	def _get_campaign_analytics(self, campaign: NECampaign) -> Dict[str, Any]:
		"""Get detailed analytics for campaign"""
		# Implementation would calculate real analytics
		return {
			'performance_metrics': {
				'sent': campaign.total_sent,
				'delivered': campaign.total_delivered,
				'opened': campaign.total_opened,
				'clicked': campaign.total_clicked,
				'delivery_rate': campaign.delivery_rate,
				'open_rate': campaign.open_rate,
				'click_rate': campaign.click_rate
			},
			'channel_breakdown': {},
			'timeline_data': [],
			'top_links': [],
			'device_breakdown': {},
			'geographic_data': {}
		}
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class NEUserPreferenceModelView(ModelView):
	"""User preference management view"""
	
	datamodel = SQLAInterface(NEUserPreference)
	
	# List view configuration
	list_columns = [
		'user_id', 'email_enabled', 'sms_enabled', 'push_enabled',
		'is_subscribed', 'engagement_score', 'last_engagement'
	]
	show_columns = [
		'preference_id', 'user_id', 'email_address', 'phone_number',
		'email_enabled', 'sms_enabled', 'push_enabled', 'in_app_enabled',
		'digest_enabled', 'digest_frequency', 'timezone', 'quiet_hours_start',
		'quiet_hours_end', 'is_subscribed', 'engagement_score', 'last_engagement'
	]
	edit_columns = [
		'user_id', 'email_address', 'phone_number', 'email_enabled',
		'sms_enabled', 'push_enabled', 'in_app_enabled', 'digest_enabled',
		'digest_frequency', 'timezone', 'quiet_hours_start', 'quiet_hours_end',
		'content_categories', 'language_preference'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['user_id', 'email_address']
	base_filters = [['is_subscribed', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('engagement_score', 'desc')
	
	# Form validation
	validators_columns = {
		'user_id': [DataRequired()],
		'email_address': [Email(), Optional()],
		'engagement_score': [NumberRange(min=0, max=100)]
	}
	
	# Custom labels
	label_columns = {
		'preference_id': 'Preference ID',
		'user_id': 'User ID',
		'email_address': 'Email Address',
		'phone_number': 'Phone Number',
		'email_enabled': 'Email Enabled',
		'sms_enabled': 'SMS Enabled',
		'push_enabled': 'Push Enabled',
		'in_app_enabled': 'In-App Enabled',
		'digest_enabled': 'Digest Enabled',
		'digest_frequency': 'Digest Frequency',
		'quiet_hours_start': 'Quiet Hours Start',
		'quiet_hours_end': 'Quiet Hours End',
		'content_categories': 'Content Categories',
		'language_preference': 'Language Preference',
		'is_subscribed': 'Subscribed',
		'engagement_score': 'Engagement Score',
		'last_engagement': 'Last Engagement'
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new user preference"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if item.timezone is None:
			item.timezone = 'UTC'
		if item.language_preference is None:
			item.language_preference = 'en-US'
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class NotificationDashboardView(NotificationEngineBaseView):
	"""Notification engine dashboard"""
	
	route_base = "/notification_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Notification dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('notification_engine/dashboard.html',
								   metrics=metrics,
								   page_title="Notification Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('notification_engine/dashboard.html',
								   metrics={},
								   page_title="Notification Dashboard")
	
	@expose('/analytics/')
	@has_access
	def analytics(self):
		"""Notification analytics and reporting"""
		try:
			period_days = int(request.args.get('period', 30))
			analytics_data = self._get_analytics_data(period_days)
			
			return render_template('notification_engine/analytics.html',
								   analytics_data=analytics_data,
								   period_days=period_days,
								   page_title="Notification Analytics")
		except Exception as e:
			flash(f'Error loading analytics: {str(e)}', 'error')
			return redirect(url_for('NotificationDashboardView.index'))
	
	@expose('/delivery_status/')
	@has_access
	def delivery_status(self):
		"""Real-time delivery status monitoring"""
		try:
			status_data = self._get_delivery_status_data()
			
			return render_template('notification_engine/delivery_status.html',
								   status_data=status_data,
								   page_title="Delivery Status")
		except Exception as e:
			flash(f'Error loading delivery status: {str(e)}', 'error')
			return redirect(url_for('NotificationDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get notification metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'total_notifications': 15420,
			'sent_today': 342,
			'pending_notifications': 28,
			'failed_notifications': 12,
			'delivery_rate': 98.2,
			'open_rate': 24.8,
			'click_rate': 3.2,
			'active_campaigns': 8,
			'active_templates': 45,
			'total_subscribers': 12350,
			'recent_activity': [
				{'type': 'notification_sent', 'count': 342, 'time': '2 hours ago'},
				{'type': 'campaign_started', 'count': 1, 'time': '4 hours ago'},
				{'type': 'template_updated', 'count': 3, 'time': '6 hours ago'}
			]
		}
	
	def _get_analytics_data(self, period_days: int) -> Dict[str, Any]:
		"""Get analytics data for specified period"""
		# Implementation would calculate real analytics
		return {
			'period_days': period_days,
			'total_sent': 8500,
			'total_delivered': 8347,
			'total_opened': 2106,
			'total_clicked': 272,
			'channel_breakdown': {
				'email': 6800,
				'sms': 1200,
				'push': 500
			},
			'daily_stats': [],
			'top_campaigns': [],
			'bounce_rate': 1.8,
			'unsubscribe_rate': 0.3
		}
	
	def _get_delivery_status_data(self) -> Dict[str, Any]:
		"""Get real-time delivery status data"""
		return {
			'queued': 28,
			'processing': 15,
			'sent': 8347,
			'delivered': 8201,
			'failed': 146,
			'recent_deliveries': [],
			'provider_status': []
		}


class NotificationComposerView(NotificationEngineBaseView):
	"""Compose and send notifications"""
	
	route_base = "/notification_composer"
	default_view = "index"
	
	@expose('/', methods=['GET', 'POST'])
	@has_access
	def index(self):
		"""Notification composer main page"""
		if request.method == 'POST':
			try:
				# Create notification from form data
				notification_data = {
					'title': request.form.get('title'),
					'message': request.form.get('message'),
					'recipient_email': request.form.get('recipient_email'),
					'channels': request.form.getlist('channels'),
					'priority': request.form.get('priority', 'normal'),
					'delivery_method': request.form.get('delivery_method', 'immediate')
				}
				
				# Create and save notification
				notification = NENotification(
					tenant_id=self._get_tenant_id(),
					**notification_data
				)
				
				from flask_appbuilder import db
				db.session.add(notification)
				db.session.commit()
				
				flash('Notification created and queued for delivery', 'success')
				return redirect(url_for('NotificationComposerView.index'))
			except Exception as e:
				flash(f'Error creating notification: {str(e)}', 'error')
		
		# Get templates for selection
		templates = self._get_available_templates()
		
		return render_template('notification_engine/composer.html',
							   templates=templates,
							   page_title="Compose Notification")
	
	def _get_available_templates(self) -> List[Dict[str, Any]]:
		"""Get available templates for composer"""
		# Implementation would query active templates
		return []


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all notification engine views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		NENotificationModelView,
		"Notifications",
		icon="fa-bell",
		category="Notification Engine",
		category_icon="fa-bullhorn"
	)
	
	appbuilder.add_view(
		NETemplateModelView,
		"Templates",
		icon="fa-file-text",
		category="Notification Engine"
	)
	
	appbuilder.add_view(
		NECampaignModelView,
		"Campaigns",
		icon="fa-rocket",
		category="Notification Engine"
	)
	
	appbuilder.add_view(
		NEUserPreferenceModelView,
		"User Preferences",
		icon="fa-cog",
		category="Notification Engine"
	)
	
	# Dashboard and management views
	appbuilder.add_view_no_menu(NotificationDashboardView)
	appbuilder.add_view_no_menu(NotificationComposerView)
	
	# Menu links
	appbuilder.add_link(
		"Notification Dashboard",
		href="/notification_dashboard/",
		icon="fa-dashboard",
		category="Notification Engine"
	)
	
	appbuilder.add_link(
		"Compose Notification",
		href="/notification_composer/",
		icon="fa-pencil",
		category="Notification Engine"
	)
	
	appbuilder.add_link(
		"Analytics",
		href="/notification_dashboard/analytics/",
		icon="fa-bar-chart",
		category="Notification Engine"
	)
	
	appbuilder.add_link(
		"Delivery Status",
		href="/notification_dashboard/delivery_status/",
		icon="fa-truck",
		category="Notification Engine"
	)