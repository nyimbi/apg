"""
Customer Relationship Management Views

Modern, responsive Flask-AppBuilder views with real-time updates,
accessibility compliance, and advanced data visualization.
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal

from flask import request, flash, redirect, url_for, jsonify, render_template, abort
from flask_appbuilder import ModelView, BaseView, has_access, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import GroupByChartView, TimeChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.security.decorators import protect
from flask_babel import lazy_gettext as _
from wtforms import Form, StringField, SelectField, DecimalField, DateField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, Email, Optional as WTFOptional, Length, NumberRange

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
from .service import CRMService, create_crm_service

# Configure logging
logger = logging.getLogger(__name__)

# Custom Widgets for Enhanced UI

class ResponsiveListWidget(ListWidget):
	"""Enhanced list widget with responsive design and real-time updates"""
	template = 'crm/widgets/responsive_list.html'

class ResponsiveShowWidget(ShowWidget):
	"""Enhanced show widget with tabbed interface and related data"""
	template = 'crm/widgets/responsive_show.html'

class ResponsiveEditWidget(EditWidget):
	"""Enhanced edit widget with inline validation and auto-save"""
	template = 'crm/widgets/responsive_edit.html'

class DashboardWidget(object):
	"""Custom dashboard widget for CRM metrics"""
	template = 'crm/widgets/dashboard_widget.html'

# Custom Forms with Advanced Validation

class LeadForm(DynamicForm):
	"""Enhanced lead form with AI-powered field suggestions"""
	first_name = StringField(_('First Name'), [DataRequired(), Length(max=100)])
	last_name = StringField(_('Last Name'), [DataRequired(), Length(max=100)])
	email = StringField(_('Email'), [DataRequired(), Email(), Length(max=255)])
	phone = StringField(_('Phone'), [Length(max=50)])
	company = StringField(_('Company'), [Length(max=200)])
	job_title = StringField(_('Job Title'), [Length(max=100)])
	lead_source = SelectField(_('Lead Source'), 
		choices=[
			('website', 'Website'),
			('social_media', 'Social Media'),
			('email_campaign', 'Email Campaign'),
			('trade_show', 'Trade Show'),
			('referral', 'Referral'),
			('cold_call', 'Cold Call'),
			('advertisement', 'Advertisement'),
			('partner', 'Partner'),
			('other', 'Other')
		])
	lead_rating = SelectField(_('Lead Rating'),
		choices=[
			('hot', 'Hot'),
			('warm', 'Warm'),
			('cold', 'Cold')
		])
	annual_revenue = DecimalField(_('Annual Revenue'), [WTFOptional(), NumberRange(min=0)])
	employee_count = StringField(_('Employee Count'), [Length(max=50)])
	website = StringField(_('Website'), [Length(max=255)])
	industry = StringField(_('Industry'), [Length(max=100)])
	description = TextAreaField(_('Description'))

class OpportunityForm(DynamicForm):
	"""Enhanced opportunity form with probability calculation"""
	opportunity_name = StringField(_('Opportunity Name'), [DataRequired(), Length(max=255)])
	amount = DecimalField(_('Amount'), [DataRequired(), NumberRange(min=0)])
	close_date = DateField(_('Expected Close Date'), [DataRequired()])
	stage = SelectField(_('Stage'),
		choices=[
			('prospecting', 'Prospecting'),
			('qualification', 'Qualification'),
			('needs_analysis', 'Needs Analysis'),
			('value_proposition', 'Value Proposition'),
			('id_decision_makers', 'Identify Decision Makers'),
			('perception_analysis', 'Perception Analysis'),
			('proposal_quote', 'Proposal/Quote'),
			('negotiation_review', 'Negotiation/Review'),
			('closed_won', 'Closed Won'),
			('closed_lost', 'Closed Lost')
		])
	probability = DecimalField(_('Probability (%)'), [NumberRange(min=0, max=100)])
	next_step = StringField(_('Next Step'), [Length(max=255)])
	description = TextAreaField(_('Description'))

class CustomerForm(DynamicForm):
	"""Enhanced customer form with 360° view integration"""
	first_name = StringField(_('First Name'), [DataRequired(), Length(max=100)])
	last_name = StringField(_('Last Name'), [DataRequired(), Length(max=100)])
	email = StringField(_('Email'), [DataRequired(), Email(), Length(max=255)])
	phone = StringField(_('Phone'), [Length(max=50)])
	company = StringField(_('Company'), [Length(max=200)])
	customer_status = SelectField(_('Status'),
		choices=[
			('active', 'Active'),
			('inactive', 'Inactive'),
			('churned', 'Churned'),
			('prospect', 'Prospect')
		])
	customer_since = DateField(_('Customer Since'))
	annual_revenue = DecimalField(_('Annual Revenue'), [WTFOptional(), NumberRange(min=0)])
	preferred_contact_method = SelectField(_('Preferred Contact Method'),
		choices=[
			('email', 'Email'),
			('phone', 'Phone'),
			('mail', 'Mail'),
			('text', 'Text')
		])

# Core Entity Views

class LeadModelView(ModelView):
	"""Enhanced Lead management with AI-powered insights"""
	
	datamodel = SQLAInterface(GCCRMLead)
	
	# Display customization
	list_title = _('Lead Management')
	show_title = _('Lead Details')
	add_title = _('Add New Lead')
	edit_title = _('Edit Lead')
	
	# Responsive widgets
	list_widget = ResponsiveListWidget
	show_widget = ResponsiveShowWidget
	edit_widget = ResponsiveEditWidget
	
	# Column configuration
	list_columns = [
		'full_name', 'company', 'email', 'phone', 'lead_source', 
		'lead_status', 'lead_rating', 'lead_score', 'created_on'
	]
	show_columns = [
		'full_name', 'email', 'phone', 'company', 'job_title', 'lead_source',
		'lead_status', 'lead_rating', 'lead_score', 'annual_revenue', 'employee_count',
		'website', 'industry', 'description', 'ai_insights', 'created_on', 'changed_on'
	]
	edit_columns = [
		'first_name', 'last_name', 'email', 'phone', 'company', 'job_title',
		'lead_source', 'lead_rating', 'annual_revenue', 'employee_count',
		'website', 'industry', 'description'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['first_name', 'last_name', 'email', 'company']
	base_filters = [['is_active', '==', True]]
	
	# Ordering
	base_order = ('created_on', 'desc')
	
	# Form configuration
	edit_form = LeadForm
	add_form = LeadForm
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	@expose('/dashboard')
	@has_access
	def lead_dashboard(self):
		"""Lead management dashboard with analytics"""
		try:
			crm_service = create_crm_service(self.datamodel.session, 
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			# Get lead analytics
			date_from = date.today() - timedelta(days=30)
			date_to = date.today()
			analytics = crm_service.leads.get_lead_analytics(date_from, date_to)
			
			return self.render_template(
				'crm/lead_dashboard.html',
				analytics=analytics,
				title='Lead Dashboard'
			)
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return redirect(url_for('LeadModelView.list'))
	
	@expose('/convert/<lead_id>')
	@has_access
	def convert_lead(self, lead_id):
		"""Convert lead to opportunity"""
		try:
			crm_service = create_crm_service(self.datamodel.session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			lead = self.datamodel.get(lead_id)
			if not lead:
				flash('Lead not found', 'error')
				return redirect(url_for('LeadModelView.list'))
			
			if request.method == 'POST':
				opportunity_data = {
					'name': request.form.get('opportunity_name'),
					'amount': Decimal(request.form.get('amount', '0')),
					'close_date': datetime.strptime(request.form.get('close_date'), '%Y-%m-%d').date()
				}
				
				opportunity, contact = crm_service.leads.convert_lead_to_opportunity(lead_id, opportunity_data)
				
				flash(f'Lead successfully converted to opportunity: {opportunity.opportunity_name}', 'success')
				return redirect(url_for('OpportunityModelView.show', pk=opportunity.id))
			
			return self.render_template(
				'crm/convert_lead.html',
				lead=lead,
				title=f'Convert Lead: {lead.first_name} {lead.last_name}'
			)
		
		except Exception as e:
			logger.error(f"Error converting lead {lead_id}: {str(e)}")
			flash(f'Error converting lead: {str(e)}', 'error')
			return redirect(url_for('LeadModelView.list'))
	
	@expose('/ai_insights/<lead_id>')
	@has_access
	def ai_insights(self, lead_id):
		"""Get AI insights for lead"""
		try:
			lead = self.datamodel.get(lead_id)
			if not lead:
				return jsonify({'error': 'Lead not found'}), 404
			
			# Return AI insights as JSON for AJAX requests
			insights = lead.ai_insights or {}
			return jsonify(insights)
		
		except Exception as e:
			logger.error(f"Error getting AI insights for lead {lead_id}: {str(e)}")
			return jsonify({'error': str(e)}), 500

class OpportunityModelView(ModelView):
	"""Enhanced Opportunity management with pipeline visualization"""
	
	datamodel = SQLAInterface(GCCRMOpportunity)
	
	list_title = _('Sales Pipeline')
	show_title = _('Opportunity Details')
	add_title = _('Add New Opportunity')
	edit_title = _('Edit Opportunity')
	
	list_widget = ResponsiveListWidget
	show_widget = ResponsiveShowWidget
	edit_widget = ResponsiveEditWidget
	
	list_columns = [
		'opportunity_name', 'amount', 'expected_revenue', 'stage', 
		'probability', 'close_date', 'opportunity_owner', 'created_on'
	]
	show_columns = [
		'opportunity_name', 'amount', 'expected_revenue', 'stage', 'probability',
		'close_date', 'actual_close_date', 'next_step', 'opportunity_owner',
		'primary_contact', 'account', 'lead_source', 'description',
		'ai_insights', 'created_on', 'changed_on'
	]
	edit_columns = [
		'opportunity_name', 'amount', 'close_date', 'stage', 'probability',
		'next_step', 'description', 'opportunity_owner_id', 'primary_contact_id',
		'account_id'
	]
	add_columns = edit_columns
	
	search_columns = ['opportunity_name', 'description']
	base_filters = [['is_active', '==', True]]
	base_order = ('close_date', 'asc')
	
	edit_form = OpportunityForm
	add_form = OpportunityForm
	
	@expose('/pipeline')
	@has_access
	def pipeline_view(self):
		"""Visual sales pipeline with drag-and-drop functionality"""
		try:
			crm_service = create_crm_service(self.datamodel.session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			analytics = crm_service.opportunities.get_sales_pipeline_analytics()
			
			return self.render_template(
				'crm/pipeline_view.html',
				analytics=analytics,
				title='Sales Pipeline'
			)
		except Exception as e:
			flash(f'Error loading pipeline: {str(e)}', 'error')
			return redirect(url_for('OpportunityModelView.list'))
	
	@expose('/forecast')
	@has_access
	def revenue_forecast(self):
		"""AI-powered revenue forecasting"""
		try:
			crm_service = create_crm_service(self.datamodel.session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			forecast_data = crm_service.opportunities.forecast_revenue()
			
			return self.render_template(
				'crm/revenue_forecast.html',
				forecast=forecast_data,
				title='Revenue Forecast'
			)
		except Exception as e:
			flash(f'Error generating forecast: {str(e)}', 'error')
			return redirect(url_for('OpportunityModelView.list'))
	
	@expose('/update_stage/<opportunity_id>', methods=['POST'])
	@has_access
	def update_stage(self, opportunity_id):
		"""Update opportunity stage via AJAX"""
		try:
			new_stage = request.json.get('stage')
			stage_data = request.json.get('data', {})
			
			crm_service = create_crm_service(self.datamodel.session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			opportunity = crm_service.opportunities.update_opportunity_stage(
				opportunity_id, OpportunityStage(new_stage), stage_data
			)
			
			return jsonify({
				'success': True,
				'opportunity': {
					'id': opportunity.id,
					'stage': opportunity.stage.value,
					'probability': opportunity.probability,
					'expected_revenue': float(opportunity.expected_revenue or 0)
				}
			})
		
		except Exception as e:
			logger.error(f"Error updating opportunity stage: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

class CustomerModelView(ModelView):
	"""Enhanced Customer management with 360° view"""
	
	datamodel = SQLAInterface(GCCRMCustomer)
	
	list_title = _('Customer Management')
	show_title = _('Customer Profile')
	add_title = _('Add New Customer')
	edit_title = _('Edit Customer')
	
	list_widget = ResponsiveListWidget
	show_widget = ResponsiveShowWidget
	edit_widget = ResponsiveEditWidget
	
	list_columns = [
		'full_name', 'company', 'email', 'phone', 'customer_status',
		'customer_value', 'customer_since', 'last_contact_date'
	]
	show_columns = [
		'full_name', 'email', 'phone', 'company', 'customer_status',
		'customer_value', 'customer_since', 'last_contact_date',
		'annual_revenue', 'preferred_contact_method', 'lead_source',
		'description', 'created_on', 'changed_on'
	]
	edit_columns = [
		'first_name', 'last_name', 'email', 'phone', 'company',
		'customer_status', 'customer_since', 'annual_revenue',
		'preferred_contact_method', 'description'
	]
	add_columns = edit_columns
	
	search_columns = ['first_name', 'last_name', 'email', 'company']
	base_filters = [['is_active', '==', True]]
	base_order = ('customer_since', 'desc')
	
	edit_form = CustomerForm
	add_form = CustomerForm
	
	@expose('/360_view/<customer_id>')
	@has_access
	def customer_360_view(self, customer_id):
		"""Comprehensive 360° customer view"""
		try:
			crm_service = create_crm_service(self.datamodel.session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			customer_360 = crm_service.customers.get_customer_360_view(customer_id)
			
			return self.render_template(
				'crm/customer_360.html',
				customer_360=customer_360,
				title=f'Customer: {customer_360["customer_info"]["name"]}'
			)
		except Exception as e:
			flash(f'Error loading customer view: {str(e)}', 'error')
			return redirect(url_for('CustomerModelView.list'))
	
	@expose('/clv/<customer_id>')
	@has_access
	def customer_lifetime_value(self, customer_id):
		"""Calculate and display customer lifetime value"""
		try:
			crm_service = create_crm_service(self.datamodel.session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			clv_data = crm_service.customers.calculate_customer_lifetime_value(customer_id)
			
			return self.render_template(
				'crm/customer_clv.html',
				clv_data=clv_data,
				title='Customer Lifetime Value'
			)
		except Exception as e:
			flash(f'Error calculating CLV: {str(e)}', 'error')
			return redirect(url_for('CustomerModelView.show', pk=customer_id))
	
	@expose('/segmentation')
	@has_access
	def customer_segmentation(self):
		"""AI-powered customer segmentation dashboard"""
		try:
			crm_service = create_crm_service(self.datamodel.session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			segmentation_data = crm_service.customers.segment_customers()
			
			return self.render_template(
				'crm/customer_segmentation.html',
				segmentation=segmentation_data,
				title='Customer Segmentation'
			)
		except Exception as e:
			flash(f'Error loading segmentation: {str(e)}', 'error')
			return redirect(url_for('CustomerModelView.list'))

class ContactModelView(ModelView):
	"""Enhanced Contact management with relationship mapping"""
	
	datamodel = SQLAInterface(GCCRMContact)
	
	list_title = _('Contact Management')
	show_title = _('Contact Details')
	add_title = _('Add New Contact')
	edit_title = _('Edit Contact')
	
	list_widget = ResponsiveListWidget
	show_widget = ResponsiveShowWidget
	edit_widget = ResponsiveEditWidget
	
	list_columns = [
		'full_name', 'email', 'phone', 'company', 'job_title',
		'account', 'contact_owner', 'is_primary_contact', 'created_on'
	]
	show_columns = [
		'full_name', 'email', 'phone', 'company', 'job_title', 'department',
		'account', 'customer', 'contact_owner', 'is_primary_contact',
		'lead_source', 'description', 'created_on', 'changed_on'
	]
	edit_columns = [
		'first_name', 'last_name', 'email', 'phone', 'company', 'job_title',
		'department', 'account_id', 'customer_id', 'contact_owner_id',
		'is_primary_contact', 'description'
	]
	add_columns = edit_columns
	
	search_columns = ['first_name', 'last_name', 'email', 'company', 'job_title']
	base_filters = [['is_active', '==', True]]
	base_order = ('created_on', 'desc')

class ActivityModelView(ModelView):
	"""Enhanced Activity management with smart scheduling"""
	
	datamodel = SQLAInterface(GCCRMActivity)
	
	list_title = _('Activity Management')
	show_title = _('Activity Details')
	add_title = _('Schedule Activity')
	edit_title = _('Edit Activity')
	
	list_widget = ResponsiveListWidget
	show_widget = ResponsiveShowWidget
	edit_widget = ResponsiveEditWidget
	
	list_columns = [
		'subject', 'activity_type', 'activity_status', 'due_date',
		'assigned_to', 'related_opportunity', 'related_contact', 'created_on'
	]
	show_columns = [
		'subject', 'description', 'activity_type', 'activity_status',
		'due_date', 'completed_date', 'assigned_to', 'related_opportunity',
		'related_contact', 'related_lead', 'created_on', 'changed_on'
	]
	edit_columns = [
		'subject', 'description', 'activity_type', 'activity_status',
		'due_date', 'assigned_to_id', 'opportunity_id', 'contact_id', 'lead_id'
	]
	add_columns = edit_columns
	
	search_columns = ['subject', 'description']
	base_order = ('due_date', 'asc')
	
	@expose('/calendar')
	@has_access
	def activity_calendar(self):
		"""Calendar view of activities"""
		try:
			# Get activities for calendar view
			activities = self.datamodel.session.query(GCCRMActivity).filter(
				GCCRMActivity.tenant_id == self.appbuilder.sm.get_user_tenant_id(),
				GCCRMActivity.due_date >= date.today() - timedelta(days=30),
				GCCRMActivity.due_date <= date.today() + timedelta(days=90)
			).all()
			
			calendar_events = []
			for activity in activities:
				calendar_events.append({
					'id': activity.id,
					'title': activity.subject,
					'start': activity.due_date.isoformat() if activity.due_date else None,
					'type': activity.activity_type.value if activity.activity_type else 'task',
					'status': activity.activity_status.value if activity.activity_status else 'pending'
				})
			
			return self.render_template(
				'crm/activity_calendar.html',
				events=calendar_events,
				title='Activity Calendar'
			)
		except Exception as e:
			flash(f'Error loading calendar: {str(e)}', 'error')
			return redirect(url_for('ActivityModelView.list'))

class CaseModelView(ModelView):
	"""Enhanced Case management for customer support"""
	
	datamodel = SQLAInterface(GCCRMCase)
	
	list_title = _('Case Management')
	show_title = _('Case Details')
	add_title = _('Create New Case')
	edit_title = _('Edit Case')
	
	list_widget = ResponsiveListWidget
	show_widget = ResponsiveShowWidget
	edit_widget = ResponsiveEditWidget
	
	list_columns = [
		'case_number', 'subject', 'status', 'priority', 'customer',
		'case_owner', 'created_on'
	]
	show_columns = [
		'case_number', 'subject', 'description', 'status', 'priority',
		'customer', 'contact', 'case_owner', 'case_origin',
		'created_on', 'changed_on'
	]
	edit_columns = [
		'subject', 'description', 'status', 'priority', 'customer_id',
		'contact_id', 'case_owner_id', 'case_origin'
	]
	add_columns = edit_columns
	
	search_columns = ['case_number', 'subject', 'description']
	base_order = ('created_on', 'desc')

class CampaignModelView(ModelView):
	"""Enhanced Campaign management with analytics"""
	
	datamodel = SQLAInterface(GCCRMCampaign)
	
	list_title = _('Campaign Management')
	show_title = _('Campaign Details')
	add_title = _('Create Campaign')
	edit_title = _('Edit Campaign')
	
	list_widget = ResponsiveListWidget
	show_widget = ResponsiveShowWidget
	edit_widget = ResponsiveEditWidget
	
	list_columns = [
		'campaign_name', 'campaign_type', 'status', 'start_date',
		'end_date', 'budget', 'expected_revenue', 'created_on'
	]
	show_columns = [
		'campaign_name', 'description', 'campaign_type', 'status',
		'start_date', 'end_date', 'budget', 'actual_cost',
		'expected_revenue', 'expected_response', 'num_sent',
		'created_on', 'changed_on'
	]
	edit_columns = [
		'campaign_name', 'description', 'campaign_type', 'status',
		'start_date', 'end_date', 'budget', 'expected_revenue',
		'expected_response', 'num_sent'
	]
	add_columns = edit_columns
	
	search_columns = ['campaign_name', 'description']
	base_order = ('start_date', 'desc')

# Dashboard and Analytics Views

class CRMDashboardView(BaseView):
	"""Main CRM dashboard with comprehensive analytics"""
	
	route_base = '/crm_dashboard'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard with key metrics and charts"""
		try:
			crm_service = create_crm_service(self.datamodel.session if hasattr(self, 'datamodel') else self.appbuilder.get_session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			dashboard_data = crm_service.get_dashboard_data()
			
			return self.render_template(
				'crm/dashboard.html',
				dashboard_data=dashboard_data,
				title='CRM Dashboard'
			)
		except Exception as e:
			logger.error(f"Error loading CRM dashboard: {str(e)}")
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('crm/error.html', error=str(e))
	
	@expose('/real_time_metrics')
	@has_access
	def real_time_metrics(self):
		"""Real-time metrics endpoint for AJAX updates"""
		try:
			crm_service = create_crm_service(self.appbuilder.get_session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			metrics = crm_service.get_dashboard_data()
			return jsonify(metrics)
		
		except Exception as e:
			logger.error(f"Error getting real-time metrics: {str(e)}")
			return jsonify({'error': str(e)}), 500

class CRMAnalyticsView(BaseView):
	"""Advanced analytics and reporting"""
	
	route_base = '/crm_analytics'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Analytics homepage"""
		return self.render_template('crm/analytics_home.html', title='CRM Analytics')
	
	@expose('/sales_performance')
	@has_access
	def sales_performance(self):
		"""Sales performance analytics"""
		try:
			# Implementation for sales performance analytics
			return self.render_template(
				'crm/sales_performance.html',
				title='Sales Performance Analytics'
			)
		except Exception as e:
			flash(f'Error loading sales analytics: {str(e)}', 'error')
			return redirect(url_for('CRMAnalyticsView.index'))
	
	@expose('/customer_insights')
	@has_access
	def customer_insights(self):
		"""Customer behavior and insights"""
		try:
			# Implementation for customer insights
			return self.render_template(
				'crm/customer_insights.html',
				title='Customer Insights'
			)
		except Exception as e:
			flash(f'Error loading customer insights: {str(e)}', 'error')
			return redirect(url_for('CRMAnalyticsView.index'))

class CRMReportsView(BaseView):
	"""Comprehensive reporting system"""
	
	route_base = '/crm_reports'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Reports homepage"""
		return self.render_template('crm/reports_home.html', title='CRM Reports')
	
	@expose('/generate_report', methods=['GET', 'POST'])
	@has_access
	def generate_report(self):
		"""Dynamic report generation"""
		if request.method == 'POST':
			try:
				report_type = request.form.get('report_type')
				date_from = datetime.strptime(request.form.get('date_from'), '%Y-%m-%d').date()
				date_to = datetime.strptime(request.form.get('date_to'), '%Y-%m-%d').date()
				
				# Generate report based on type
				report_data = self._generate_report_data(report_type, date_from, date_to)
				
				return self.render_template(
					'crm/generated_report.html',
					report_data=report_data,
					report_type=report_type,
					title=f'{report_type.replace("_", " ").title()} Report'
				)
			except Exception as e:
				flash(f'Error generating report: {str(e)}', 'error')
		
		return self.render_template('crm/generate_report.html', title='Generate Report')
	
	def _generate_report_data(self, report_type: str, date_from: date, date_to: date) -> Dict[str, Any]:
		"""Generate report data based on type and date range"""
		# Implementation would vary based on report type
		return {
			'report_type': report_type,
			'date_from': date_from.isoformat(),
			'date_to': date_to.isoformat(),
			'generated_at': datetime.utcnow().isoformat(),
			'data': {}  # Actual report data would go here
		}

# Chart Views for Data Visualization

class LeadsBySourceChart(GroupByChartView):
	"""Chart showing leads by source"""
	datamodel = SQLAInterface(GCCRMLead)
	chart_title = 'Leads by Source'
	group_by_columns = ['lead_source']

class OpportunityStageChart(GroupByChartView):
	"""Chart showing opportunities by stage"""
	datamodel = SQLAInterface(GCCRMOpportunity)
	chart_title = 'Opportunities by Stage'
	group_by_columns = ['stage']

class RevenueTimeChart(TimeChartView):
	"""Chart showing revenue over time"""
	datamodel = SQLAInterface(GCCRMOpportunity)
	chart_title = 'Revenue Over Time'
	group_by_columns = ['close_date']
	
	def query_filters(self):
		"""Filter to show only closed won opportunities"""
		return [GCCRMOpportunity.stage == OpportunityStage.CLOSED_WON]

# API Views for Mobile and External Integration

class CRMAPIView(BaseView):
	"""RESTful API endpoints for CRM data"""
	
	route_base = '/api/crm'
	default_view = 'health'
	
	@expose('/health')
	def health(self):
		"""API health check"""
		try:
			crm_service = create_crm_service(self.appbuilder.get_session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			health_status = crm_service.health_check()
			return jsonify(health_status)
		
		except Exception as e:
			return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
	
	@expose('/leads', methods=['GET'])
	@protect()
	def get_leads(self):
		"""Get leads with pagination and filtering"""
		try:
			page = int(request.args.get('page', 1))
			per_page = min(int(request.args.get('per_page', 10)), 100)
			
			query = self.appbuilder.get_session.query(GCCRMLead).filter(
				GCCRMLead.tenant_id == self.appbuilder.sm.get_user_tenant_id(),
				GCCRMLead.is_active == True
			)
			
			# Apply filters
			if request.args.get('status'):
				query = query.filter(GCCRMLead.lead_status == request.args.get('status'))
			
			if request.args.get('source'):
				query = query.filter(GCCRMLead.lead_source == request.args.get('source'))
			
			# Pagination
			total = query.count()
			leads = query.offset((page - 1) * per_page).limit(per_page).all()
			
			return jsonify({
				'leads': [
					{
						'id': lead.id,
						'name': f"{lead.first_name} {lead.last_name}",
						'email': lead.email,
						'company': lead.company,
						'status': lead.lead_status.value if lead.lead_status else None,
						'score': lead.lead_score,
						'created_on': lead.created_on.isoformat()
					}
					for lead in leads
				],
				'pagination': {
					'page': page,
					'per_page': per_page,
					'total': total,
					'pages': (total + per_page - 1) // per_page
				}
			})
		
		except Exception as e:
			logger.error(f"Error in leads API: {str(e)}")
			return jsonify({'error': str(e)}), 500

# Bulk Operations Views

class BulkOperationsView(BaseView):
	"""Bulk operations for CRM entities"""
	
	route_base = '/crm_bulk'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Bulk operations homepage"""
		return self.render_template('crm/bulk_operations.html', title='Bulk Operations')
	
	@expose('/bulk_leads', methods=['POST'])
	@has_access
	def bulk_lead_operations(self):
		"""Perform bulk operations on leads"""
		try:
			operation = request.json.get('operation')
			lead_ids = request.json.get('lead_ids', [])
			operation_data = request.json.get('data', {})
			
			crm_service = create_crm_service(self.appbuilder.get_session,
											self.appbuilder.sm.get_user_tenant_id(),
											self.appbuilder.sm.user.id)
			
			results = crm_service.leads.bulk_lead_operations(operation, lead_ids, operation_data)
			
			return jsonify({
				'success': True,
				'results': results
			})
		
		except Exception as e:
			logger.error(f"Error in bulk lead operations: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

# Configuration Views

class CRMConfigView(BaseView):
	"""CRM system configuration and settings"""
	
	route_base = '/crm_config'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Configuration homepage"""
		return self.render_template('crm/config_home.html', title='CRM Configuration')
	
	@expose('/system_settings')
	@has_access
	def system_settings(self):
		"""System-wide CRM settings"""
		try:
			# Get current system configuration
			config = self.appbuilder.get_session.query(GCCRMSystemConfiguration).filter(
				GCCRMSystemConfiguration.tenant_id == self.appbuilder.sm.get_user_tenant_id()
			).first()
			
			return self.render_template(
				'crm/system_settings.html',
				config=config,
				title='System Settings'
			)
		except Exception as e:
			flash(f'Error loading settings: {str(e)}', 'error')
			return redirect(url_for('CRMConfigView.index'))

# Mobile-Optimized Views

class MobileCRMView(BaseView):
	"""Mobile-optimized CRM interface"""
	
	route_base = '/mobile_crm'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Mobile dashboard"""
		return self.render_template('crm/mobile/dashboard.html', title='CRM Mobile')
	
	@expose('/quick_actions')
	@has_access
	def quick_actions(self):
		"""Quick actions for mobile users"""
		return self.render_template('crm/mobile/quick_actions.html', title='Quick Actions')

# WebSocket Views for Real-time Updates

class CRMWebSocketView(BaseView):
	"""WebSocket endpoints for real-time CRM updates"""
	
	route_base = '/crm_ws'
	
	@expose('/connect')
	def websocket_connect(self):
		"""WebSocket connection endpoint"""
		# Implementation would use Flask-SocketIO or similar
		return "WebSocket connection established"


# View Registration Function
def register_crm_views(appbuilder):
	"""Register all CRM views with the AppBuilder"""
	
	# Core entity views
	appbuilder.add_view(LeadModelView, "Leads", icon="fa-users", category="CRM")
	appbuilder.add_view(OpportunityModelView, "Opportunities", icon="fa-bullseye", category="CRM")
	appbuilder.add_view(CustomerModelView, "Customers", icon="fa-user-circle", category="CRM")
	appbuilder.add_view(ContactModelView, "Contacts", icon="fa-address-book", category="CRM")
	appbuilder.add_view(ActivityModelView, "Activities", icon="fa-tasks", category="CRM")
	appbuilder.add_view(CaseModelView, "Cases", icon="fa-support", category="CRM")
	appbuilder.add_view(CampaignModelView, "Campaigns", icon="fa-megaphone", category="CRM")
	
	# Dashboard and analytics
	appbuilder.add_view(CRMDashboardView, "Dashboard", icon="fa-dashboard", category="CRM")
	appbuilder.add_view(CRMAnalyticsView, "Analytics", icon="fa-bar-chart", category="CRM")
	appbuilder.add_view(CRMReportsView, "Reports", icon="fa-file-text", category="CRM")
	
	# Charts
	appbuilder.add_view(LeadsBySourceChart, "Leads by Source", icon="fa-pie-chart", category="CRM Charts")
	appbuilder.add_view(OpportunityStageChart, "Pipeline Stages", icon="fa-bar-chart", category="CRM Charts")
	appbuilder.add_view(RevenueTimeChart, "Revenue Trend", icon="fa-line-chart", category="CRM Charts")
	
	# Utilities
	appbuilder.add_view(BulkOperationsView, "Bulk Operations", icon="fa-cogs", category="CRM Tools")
	appbuilder.add_view(CRMConfigView, "Configuration", icon="fa-gear", category="CRM Tools")
	appbuilder.add_view(MobileCRMView, "Mobile CRM", icon="fa-mobile", category="CRM Tools")
	
	# API endpoints
	appbuilder.add_view_no_menu(CRMAPIView)
	appbuilder.add_view_no_menu(CRMWebSocketView)
	
	# Permissions
	appbuilder.add_permissions_view([
		"can_list", "can_show", "can_add", "can_edit", "can_delete"
	], "LeadModelView")
	appbuilder.add_permissions_view([
		"can_list", "can_show", "can_add", "can_edit", "can_delete"
	], "OpportunityModelView")
	appbuilder.add_permissions_view([
		"can_list", "can_show", "can_add", "can_edit", "can_delete"
	], "CustomerModelView")