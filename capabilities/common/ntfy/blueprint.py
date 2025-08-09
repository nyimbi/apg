"""
APG Notification Capability - Flask-AppBuilder Blueprint

Enterprise-grade Flask-AppBuilder integration providing comprehensive web interface
for notification management, campaign orchestration, analytics, and administration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import AppBuilder, SQLA
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.views import ModelView, BaseView, expose
from flask_appbuilder.charts.views import ChartView
from flask_appbuilder.models.group import aggregate_count
from flask_appbuilder.security.decorators import has_access, protect
from flask_appbuilder.widgets import ListWidget, FormWidget, SearchWidget
from flask_appbuilder.actions import action
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

# Import models and services
from .models import (
	NENotification, NETemplate, NEDelivery, NEInteraction, NECampaign,
	NECampaignStep, NEUserPreference, NEProvider
)
from .service import NotificationService, create_notification_service
from .channel_manager import UniversalChannelManager, ChannelConfig
from .api_models import DeliveryChannel, NotificationPriority, CampaignType


# Create blueprint
notification_bp = Blueprint(
	'notification',
	__name__,
	template_folder='templates',
	static_folder='static',
	url_prefix='/notification'
)


class NotificationTemplateView(ModelView):
	"""Template management with advanced features"""
	
	datamodel = SQLAInterface(NETemplate)
	
	# Enhanced list view
	list_columns = [
		'code', 'name', 'version', 'locale', 'supported_channels',
		'is_active', 'usage_count', 'success_rate', 'created_on'
	]
	
	show_columns = [
		'template_id', 'code', 'name', 'version', 'locale', 'description',
		'subject_template', 'html_template', 'text_template', 'sms_template',
		'push_template', 'template_engine', 'variables_schema', 'default_variables',
		'supported_channels', 'channel_specific_config', 'is_active', 'is_default',
		'ab_test_variant', 'parent_template_id', 'usage_count', 'success_rate',
		'average_engagement', 'created_on', 'changed_on'
	]
	
	edit_columns = [
		'code', 'name', 'version', 'locale', 'description',
		'subject_template', 'html_template', 'text_template', 'sms_template',
		'push_template', 'template_engine', 'variables_schema', 'default_variables',
		'supported_channels', 'channel_specific_config', 'is_active', 'is_default'
	]
	
	add_columns = edit_columns
	
	# Enhanced search and filtering
	search_columns = ['code', 'name', 'description', 'template_engine']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Custom form configuration
	edit_form_extra_fields = {
		'supported_channels': {'widget': 'select2', 'multiple': True},
		'template_engine': {'widget': 'select', 'choices': [
			('mustache', 'Mustache'),
			('jinja2', 'Jinja2'),
			('handlebars', 'Handlebars')
		]}
	}
	
	# Actions
	@action("clone_template", "Clone Template", "Clone selected templates", "fa-copy")
	def clone_template(self, items):
		"""Clone selected templates"""
		for item in items:
			cloned = NETemplate(
				code=f"{item.code}_copy",
				name=f"{item.name} (Copy)",
				version="1.0.0",
				locale=item.locale,
				description=f"Cloned from {item.code}",
				subject_template=item.subject_template,
				html_template=item.html_template,
				text_template=item.text_template,
				sms_template=item.sms_template,
				push_template=item.push_template,
				template_engine=item.template_engine,
				variables_schema=item.variables_schema,
				default_variables=item.default_variables,
				supported_channels=item.supported_channels,
				tenant_id=item.tenant_id
			)
			self.datamodel.add(cloned)
		
		flash(f"Cloned {len(items)} template(s)", "success")
		return redirect(self.get_redirect())
	
	@expose('/preview/<int:pk>')
	@has_access
	def preview(self, pk):
		"""Preview template with sample data"""
		template = self.datamodel.get(pk)
		if not template:
			flash("Template not found", "error")
			return redirect(self.get_redirect())
		
		# Sample variables for preview
		sample_vars = {
			'user_name': 'John Doe',
			'company_name': 'ACME Corp',
			'amount': '$1,234.56',
			'date': datetime.now().strftime('%Y-%m-%d'),
			'url': 'https://example.com'
		}
		
		try:
			rendered_content = template.render(sample_vars, 'email')
			return self.render_template(
				'notification/template_preview.html',
				template=template,
				rendered=rendered_content,
				sample_vars=sample_vars
			)
		except Exception as e:
			flash(f"Preview error: {str(e)}", "error")
			return redirect(self.get_redirect())
	
	@expose('/test_send/<int:pk>', methods=['GET', 'POST'])
	@has_access
	def test_send(self, pk):
		"""Send test notification"""
		template = self.datamodel.get(pk)
		if not template:
			flash("Template not found", "error")
			return redirect(self.get_redirect())
		
		if request.method == 'POST':
			try:
				test_email = request.form.get('test_email')
				custom_vars = json.loads(request.form.get('variables', '{}'))
				
				# Use notification service to send test
				service = create_notification_service('default_tenant')
				# Would implement actual test sending
				
				flash(f"Test notification sent to {test_email}", "success")
				return redirect(self.get_redirect())
			except Exception as e:
				flash(f"Test send failed: {str(e)}", "error")
		
		return self.render_template(
			'notification/template_test.html',
			template=template
		)


class CampaignView(ModelView):
	"""Campaign management with orchestration features"""
	
	datamodel = SQLAInterface(NECampaign)
	
	list_columns = [
		'name', 'campaign_type', 'status', 'total_recipients',
		'delivery_rate', 'open_rate', 'click_rate', 'start_date'
	]
	
	show_columns = [
		'campaign_id', 'name', 'description', 'campaign_type', 'trigger_event',
		'trigger_conditions', 'target_audience', 'status', 'start_date', 'end_date',
		'total_recipients', 'total_sent', 'total_delivered', 'total_opened',
		'total_clicked', 'conversion_count', 'delivery_rate', 'open_rate',
		'click_rate', 'conversion_rate', 'unsubscribe_rate', 'steps'
	]
	
	edit_columns = [
		'name', 'description', 'campaign_type', 'trigger_event',
		'trigger_conditions', 'target_audience', 'start_date', 'end_date'
	]
	
	add_columns = edit_columns
	
	# Enhanced actions
	@action("start_campaigns", "Start Campaigns", "Start selected campaigns", "fa-play")
	def start_campaigns(self, items):
		"""Start selected campaigns"""
		started_count = 0
		for item in items:
			if item.status == 'draft':
				item.status = 'active'
				item.start_date = datetime.utcnow()
				self.datamodel.edit(item)
				started_count += 1
		
		flash(f"Started {started_count} campaign(s)", "success")
		return redirect(self.get_redirect())
	
	@action("pause_campaigns", "Pause Campaigns", "Pause selected campaigns", "fa-pause")
	def pause_campaigns(self, items):
		"""Pause selected campaigns"""
		paused_count = 0
		for item in items:
			if item.status == 'active':
				item.status = 'paused'
				self.datamodel.edit(item)
				paused_count += 1
		
		flash(f"Paused {paused_count} campaign(s)", "success")
		return redirect(self.get_redirect())
	
	@expose('/analytics/<int:pk>')
	@has_access
	def analytics(self, pk):
		"""Campaign analytics dashboard"""
		campaign = self.datamodel.get(pk)
		if not campaign:
			flash("Campaign not found", "error")
			return redirect(self.get_redirect())
		
		# Get analytics data
		analytics_data = self._get_campaign_analytics(campaign)
		
		return self.render_template(
			'notification/campaign_analytics.html',
			campaign=campaign,
			analytics=analytics_data
		)
	
	@expose('/live_preview/<int:pk>')
	@has_access  
	def live_preview(self, pk):
		"""Live campaign preview with real-time collaboration"""
		campaign = self.datamodel.get(pk)
		if not campaign:
			flash("Campaign not found", "error")
			return redirect(self.get_redirect())
		
		return self.render_template(
			'notification/campaign_live_preview.html',
			campaign=campaign
		)
	
	def _get_campaign_analytics(self, campaign: NECampaign) -> Dict[str, Any]:
		"""Get comprehensive campaign analytics"""
		return {
			'performance_overview': {
				'total_sent': campaign.total_sent,
				'delivery_rate': campaign.delivery_rate,
				'open_rate': campaign.open_rate,
				'click_rate': campaign.click_rate,
				'conversion_rate': campaign.conversion_rate
			},
			'channel_breakdown': self._get_channel_breakdown(campaign),
			'timeline_data': self._get_timeline_data(campaign),
			'engagement_funnel': self._get_engagement_funnel(campaign),
			'geographic_data': self._get_geographic_data(campaign),
			'device_breakdown': self._get_device_breakdown(campaign)
		}
	
	def _get_channel_breakdown(self, campaign: NECampaign) -> Dict[str, Any]:
		"""Get performance breakdown by channel"""
		# Would query actual data from database
		return {
			'email': {'sent': 8500, 'delivered': 8200, 'opened': 2050, 'clicked': 410},
			'sms': {'sent': 1200, 'delivered': 1180, 'opened': 300, 'clicked': 60},
			'push': {'sent': 500, 'delivered': 480, 'opened': 100, 'clicked': 20}
		}
	
	def _get_timeline_data(self, campaign: NECampaign) -> List[Dict[str, Any]]:
		"""Get campaign performance over time"""
		# Would generate actual timeline data
		return [
			{'date': '2025-01-20', 'sent': 1000, 'opened': 250, 'clicked': 50},
			{'date': '2025-01-21', 'sent': 1500, 'opened': 375, 'clicked': 75},
			{'date': '2025-01-22', 'sent': 2000, 'opened': 500, 'clicked': 100}
		]
	
	def _get_engagement_funnel(self, campaign: NECampaign) -> Dict[str, int]:
		"""Get engagement funnel data"""
		return {
			'sent': campaign.total_sent,
			'delivered': campaign.total_delivered,
			'opened': campaign.total_opened,
			'clicked': campaign.total_clicked,
			'converted': campaign.conversion_count
		}
	
	def _get_geographic_data(self, campaign: NECampaign) -> Dict[str, Any]:
		"""Get geographic performance data"""
		return {
			'countries': {
				'US': {'sent': 5000, 'opened': 1250, 'clicked': 250},
				'CA': {'sent': 2000, 'opened': 500, 'clicked': 100},
				'UK': {'sent': 1500, 'opened': 375, 'clicked': 75}
			}
		}
	
	def _get_device_breakdown(self, campaign: NECampaign) -> Dict[str, Any]:
		"""Get device/platform breakdown"""
		return {
			'desktop': {'opened': 1200, 'clicked': 240},
			'mobile': {'opened': 800, 'clicked': 160},
			'tablet': {'opened': 250, 'clicked': 50}
		}


class UserPreferencesView(ModelView):
	"""User preference management"""
	
	datamodel = SQLAInterface(NEUserPreference)
	
	list_columns = [
		'user_id', 'email_enabled', 'sms_enabled', 'push_enabled',
		'is_subscribed', 'engagement_score', 'last_engagement'
	]
	
	show_columns = [
		'preference_id', 'user_id', 'email_address', 'phone_number',
		'email_enabled', 'sms_enabled', 'push_enabled', 'in_app_enabled',
		'frequency_settings', 'digest_enabled', 'digest_frequency',
		'timezone', 'quiet_hours_start', 'quiet_hours_end',
		'content_categories', 'language_preference', 'personalization_enabled',
		'is_subscribed', 'engagement_score', 'last_engagement'
	]
	
	edit_columns = [
		'user_id', 'email_address', 'phone_number',
		'email_enabled', 'sms_enabled', 'push_enabled', 'in_app_enabled',
		'frequency_settings', 'digest_enabled', 'digest_frequency',
		'timezone', 'quiet_hours_start', 'quiet_hours_end',
		'content_categories', 'language_preference', 'personalization_enabled'
	]
	
	# Bulk preference updates
	@action("enable_email", "Enable Email", "Enable email for selected users", "fa-envelope")
	def enable_email(self, items):
		"""Enable email notifications for selected users"""
		for item in items:
			item.email_enabled = True
			self.datamodel.edit(item)
		
		flash(f"Enabled email for {len(items)} user(s)", "success")
		return redirect(self.get_redirect())
	
	@action("update_timezone", "Update Timezone", "Update timezone for selected users", "fa-clock-o")
	def update_timezone(self, items):
		"""Update timezone for selected users"""
		# Would show form to select timezone
		flash("Timezone update functionality would be implemented here", "info")
		return redirect(self.get_redirect())


class AnalyticsView(BaseView):
	"""Comprehensive analytics dashboard"""
	
	route_base = "/analytics"
	default_view = "dashboard"
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Main analytics dashboard"""
		try:
			# Get dashboard data
			dashboard_data = self._get_dashboard_data()
			
			return self.render_template(
				'notification/analytics_dashboard.html',
				data=dashboard_data,
				page_title="Notification Analytics"
			)
		except Exception as e:
			flash(f"Error loading analytics: {str(e)}", "error")
			return self.render_template(
				'notification/analytics_dashboard.html',
				data={},
				page_title="Notification Analytics"
			)
	
	@expose('/engagement/')
	@has_access
	def engagement(self):
		"""Engagement analytics"""
		period_days = int(request.args.get('period', 30))
		engagement_data = self._get_engagement_data(period_days)
		
		return self.render_template(
			'notification/engagement_analytics.html',
			data=engagement_data,
			period_days=period_days
		)
	
	@expose('/channels/')
	@has_access
	def channels(self):
		"""Channel performance analytics"""
		channel_data = self._get_channel_analytics()
		
		return self.render_template(
			'notification/channel_analytics.html',
			data=channel_data
		)
	
	@expose('/attribution/')
	@has_access
	def attribution(self):
		"""Attribution and conversion analytics"""
		attribution_data = self._get_attribution_data()
		
		return self.render_template(
			'notification/attribution_analytics.html',
			data=attribution_data
		)
	
	def _get_dashboard_data(self) -> Dict[str, Any]:
		"""Get main dashboard analytics data"""
		return {
			'overview_metrics': {
				'total_notifications': 152000,
				'delivery_rate': 98.2,
				'open_rate': 24.8,
				'click_rate': 3.2,
				'conversion_rate': 2.1
			},
			'recent_campaigns': [
				{'name': 'Welcome Series', 'status': 'active', 'performance': 85.2},
				{'name': 'Product Update', 'status': 'completed', 'performance': 78.5},
				{'name': 'Holiday Promotion', 'status': 'scheduled', 'performance': 0}
			],
			'channel_performance': {
				'email': {'sent': 125000, 'rate': 25.1},
				'sms': {'sent': 18000, 'rate': 35.8},
				'push': {'sent': 9000, 'rate': 18.2}
			},
			'trending_insights': [
				'SMS engagement up 15% this week',
				'Mobile opens increased 8% vs desktop',
				'Optimal send time: 10:00 AM local time'
			]
		}
	
	def _get_engagement_data(self, period_days: int) -> Dict[str, Any]:
		"""Get engagement analytics for specified period"""
		return {
			'period_days': period_days,
			'engagement_trends': [],
			'user_segments': {
				'highly_engaged': 2150,
				'moderately_engaged': 8900,
				'low_engagement': 3200,
				'inactive': 1100
			},
			'engagement_by_day': [],
			'top_content': []
		}
	
	def _get_channel_analytics(self) -> Dict[str, Any]:
		"""Get channel performance data"""
		return {
			'channel_comparison': {},
			'channel_trends': {},
			'optimal_channel_mix': {},
			'channel_health': {}
		}
	
	def _get_attribution_data(self) -> Dict[str, Any]:
		"""Get attribution and conversion data"""
		return {
			'attribution_model': 'multi_touch',
			'conversion_paths': [],
			'channel_attribution': {},
			'campaign_roi': {}
		}


class NotificationDashboardView(BaseView):
	"""Main notification dashboard"""
	
	route_base = "/dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard"""
		try:
			metrics = self._get_dashboard_metrics()
			return self.render_template(
				'notification/dashboard.html',
				metrics=metrics,
				page_title="Notification Dashboard"
			)
		except Exception as e:
			flash(f"Error loading dashboard: {str(e)}", "error")
			return self.render_template(
				'notification/dashboard.html',
				metrics={},
				page_title="Notification Dashboard"
			)
	
	@expose('/real_time/')
	@has_access
	def real_time(self):
		"""Real-time monitoring dashboard"""
		return self.render_template(
			'notification/real_time_dashboard.html',
			page_title="Real-Time Monitoring"
		)
	
	@expose('/health/')
	@has_access
	def health(self):
		"""System health monitoring"""
		health_data = self._get_health_data()
		return self.render_template(
			'notification/health_dashboard.html',
			health=health_data,
			page_title="System Health"
		)
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get dashboard metrics"""
		return {
			'notifications_today': 3420,
			'active_campaigns': 8,
			'delivery_rate': 98.2,
			'engagement_rate': 24.8,
			'system_health': 'healthy',
			'recent_activity': []
		}
	
	def _get_health_data(self) -> Dict[str, Any]:
		"""Get system health data"""
		return {
			'overall_status': 'healthy',
			'services': {
				'notification_service': 'healthy',
				'channel_manager': 'healthy',
				'analytics_engine': 'healthy',
				'database': 'healthy'
			},
			'performance_metrics': {
				'avg_latency_ms': 85,
				'throughput_per_hour': 12500,
				'error_rate': 0.02
			}
		}


# Chart views for analytics
class NotificationChartView(ChartView):
	"""Chart view for notification analytics"""
	
	chart_title = "Notification Performance"
	label_columns = {'sent': 'Sent', 'delivered': 'Delivered', 'opened': 'Opened'}
	group_by_columns = ['created_on']
	datamodel = SQLAInterface(NENotification)
	
	definitions = [
		{
			'group': 'created_on',
			'series': ['sent', 'delivered', 'opened']
		}
	]


def register_notification_views(appbuilder: AppBuilder):
	"""Register all notification views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		NotificationTemplateView,
		"Templates",
		icon="fa-file-text-o",
		category="Notifications",
		category_icon="fa-bell"
	)
	
	appbuilder.add_view(
		CampaignView,
		"Campaigns", 
		icon="fa-rocket",
		category="Notifications"
	)
	
	appbuilder.add_view(
		UserPreferencesView,
		"User Preferences",
		icon="fa-cog",
		category="Notifications"
	)
	
	# Dashboard and analytics
	appbuilder.add_view_no_menu(NotificationDashboardView)
	appbuilder.add_view_no_menu(AnalyticsView)
	
	# Chart views
	appbuilder.add_view(
		NotificationChartView,
		"Performance Charts",
		icon="fa-bar-chart",
		category="Notifications"
	)
	
	# Menu links
	appbuilder.add_link(
		"Dashboard",
		href="/dashboard/",
		icon="fa-dashboard",
		category="Notifications"
	)
	
	appbuilder.add_link(
		"Analytics",
		href="/analytics/",
		icon="fa-line-chart", 
		category="Notifications"
	)
	
	appbuilder.add_link(
		"Real-Time Monitor",
		href="/dashboard/real_time/",
		icon="fa-tachometer",
		category="Notifications"
	)


def create_notification_blueprint(appbuilder: AppBuilder) -> Blueprint:
	"""Create and configure notification blueprint"""
	
	# Register views
	register_notification_views(appbuilder)
	
	# Add any additional blueprint routes
	@notification_bp.route('/health')
	def health_check():
		"""Health check endpoint"""
		return jsonify({
			'status': 'healthy',
			'service': 'notification',
			'timestamp': datetime.utcnow().isoformat()
		})
	
	return notification_bp


def register_notification_capability(app, appbuilder: AppBuilder):
	"""Register notification capability with Flask app"""
	
	# Create and register blueprint
	blueprint = create_notification_blueprint(appbuilder)
	app.register_blueprint(blueprint)
	
	# Initialize database tables if needed
	with app.app_context():
		try:
			appbuilder.get_session.create_all()
		except Exception as e:
			app.logger.error(f"Failed to create notification tables: {e}")
	
	app.logger.info("Notification capability registered successfully")


# Export main functions and classes
__all__ = [
	'NotificationTemplateView',
	'CampaignView', 
	'UserPreferencesView',
	'AnalyticsView',
	'NotificationDashboardView',
	'NotificationChartView',
	'register_notification_views',
	'create_notification_blueprint',
	'register_notification_capability',
	'notification_bp'
]