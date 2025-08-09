"""
APG Payroll Management - Revolutionary User Interface Views

Next-generation Flask-AppBuilder views providing immersive payroll experience
with real-time processing, AI-powered insights, and conversational interfaces.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, FormWidget
from flask_appbuilder.actions import action
from flask_babel import lazy_gettext, gettext
from wtforms import Form, StringField, SelectField, DateField, DecimalField, TextAreaField
from wtforms.validators import DataRequired, Length, NumberRange
from sqlalchemy import and_, or_, func, desc

# APG Platform Imports
from ...auth_rbac.decorators import require_permission, tenant_required
from ...audit_compliance.mixins import AuditMixin
from ...notification_engine.decorators import notify_on_action
from ...ai_orchestration.decorators import ai_enhanced

from .models import (
	PRPayrollPeriod, PRPayrollRun, PREmployeePayroll, PRPayComponent,
	PRPayrollLineItem, PRTaxCalculation, PRPayrollAdjustment,
	PayrollStatus, PayComponentType, PayFrequency, TaxType
)
from .service import RevolutionaryPayrollService
from .ai_intelligence_engine import PayrollIntelligenceEngine
from .conversational_assistant import ConversationalPayrollAssistant

# Configure logging
logger = logging.getLogger(__name__)


class ImmersivePayrollWidget(ListWidget):
	"""Revolutionary immersive payroll list widget with real-time updates."""
	
	template = 'payroll/immersive_list.html'
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.extra_args = {
			'enable_real_time': True,
			'ai_insights_enabled': True,
			'conversational_mode': True
		}


class PayrollDashboardWidget(ShowWidget):
	"""Advanced payroll dashboard widget with AI analytics."""
	
	template = 'payroll/dashboard.html'
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.extra_args = {
			'show_ai_predictions': True,
			'enable_anomaly_alerts': True,
			'real_time_metrics': True
		}


class ConversationalPayrollForm(Form):
	"""Natural language payroll command form."""
	
	command = TextAreaField(
		lazy_gettext('Natural Language Command'),
		validators=[DataRequired(), Length(min=10, max=1000)],
		render_kw={
			'placeholder': 'Ask me anything about payroll... e.g., "Show me all employees with overtime this period" or "Calculate bonus payments for top performers"',
			'rows': 3,
			'class': 'form-control conversational-input'
		}
	)


class PayrollPeriodView(ModelView, AuditMixin):
	"""Revolutionary payroll period management view."""
	
	datamodel = SQLAInterface(PRPayrollPeriod)
	
	# View configuration
	list_title = lazy_gettext('Payroll Periods - AI-Enhanced Management')
	show_title = lazy_gettext('Payroll Period Details')
	add_title = lazy_gettext('Create New Payroll Period')
	edit_title = lazy_gettext('Edit Payroll Period')
	
	# List view configuration
	list_columns = [
		'period_name', 'period_type', 'pay_frequency', 'start_date', 
		'end_date', 'pay_date', 'status', 'employee_count', 'total_gross_pay'
	]
	
	list_widget = ImmersivePayrollWidget
	
	# Search configuration
	search_columns = [
		'period_name', 'period_type', 'status', 'fiscal_year', 'country_code'
	]
	
	# Show view configuration
	show_columns = [
		'period_name', 'period_type', 'pay_frequency', 'start_date', 'end_date',
		'pay_date', 'cutoff_date', 'fiscal_year', 'fiscal_quarter', 
		'country_code', 'currency_code', 'timezone', 'status', 'employee_count',
		'total_gross_pay', 'total_net_pay', 'ai_predictions', 'created_at'
	]
	
	show_widget = PayrollDashboardWidget
	
	# Form configuration
	add_columns = [
		'period_name', 'period_type', 'pay_frequency', 'start_date', 'end_date',
		'pay_date', 'cutoff_date', 'fiscal_year', 'fiscal_quarter',
		'country_code', 'currency_code', 'timezone'
	]
	
	edit_columns = add_columns
	
	# Validators and formatters
	validators_columns = {
		'period_name': [DataRequired(), Length(max=100)],
		'start_date': [DataRequired()],
		'end_date': [DataRequired()],
		'pay_date': [DataRequired()]
	}
	
	# Order and pagination
	base_order = ('start_date', 'desc')
	page_size = 20
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit']
	
	@expose('/ai_insights/<period_id>')
	@require_permission('view_ai_insights')
	@ai_enhanced
	def ai_insights(self, period_id):
		"""Display AI-powered insights for payroll period."""
		try:
			period = self.datamodel.get(period_id, add_base_filters=True)
			if not period:
				flash(gettext('Period not found'), 'error')
				return redirect(url_for('PayrollPeriodView.list'))
			
			# Get AI insights from intelligence engine
			intelligence_engine = PayrollIntelligenceEngine()
			insights = intelligence_engine.analyze_payroll_period(period_id)
			
			return render_template(
				'payroll/ai_insights.html',
				period=period,
				insights=insights,
				title=f'AI Insights - {period.period_name}'
			)
			
		except Exception as e:
			logger.error(f"Failed to load AI insights: {e}")
			flash(gettext('Failed to load AI insights'), 'error')
			return redirect(url_for('PayrollPeriodView.list'))
	
	@action('start_payroll', lazy_gettext('Start Payroll Processing'), lazy_gettext('Start payroll for selected periods?'), 'fa-play')
	@require_permission('start_payroll')
	@notify_on_action('payroll_started')
	def start_payroll_action(self, items):
		"""Start payroll processing for selected periods."""
		try:
			service = RevolutionaryPayrollService()
			success_count = 0
			
			for period in items:
				if period.status == PayrollStatus.DRAFT:
					# Start payroll run
					run_data = {
						'period_id': period.period_id,
						'run_type': 'regular',
						'run_name': f'Payroll Run - {period.period_name}',
						'priority': 'normal'
					}
					
					service.start_payroll_run(run_data, period.tenant_id, self.current_user.id)
					success_count += 1
			
			if success_count > 0:
				flash(gettext(f'Started payroll processing for {success_count} periods'), 'success')
			else:
				flash(gettext('No eligible periods selected'), 'warning')
				
		except Exception as e:
			logger.error(f"Failed to start payroll: {e}")
			flash(gettext('Failed to start payroll processing'), 'error')
		
		return redirect(url_for('PayrollPeriodView.list'))


class PayrollRunView(ModelView, AuditMixin):
	"""Revolutionary payroll run processing view with real-time monitoring."""
	
	datamodel = SQLAInterface(PRPayrollRun)
	
	# View configuration
	list_title = lazy_gettext('Payroll Runs - Real-Time Processing Monitor')
	show_title = lazy_gettext('Payroll Run Dashboard')
	
	# List view configuration
	list_columns = [
		'run_number', 'run_name', 'period.period_name', 'status', 
		'processing_stage', 'progress_percentage', 'employee_count',
		'validation_score', 'started_at', 'estimated_completion'
	]
	
	list_widget = ImmersivePayrollWidget
	
	# Search configuration
	search_columns = [
		'run_name', 'run_type', 'status', 'processing_stage'
	]
	
	# Show view configuration
	show_columns = [
		'run_number', 'run_name', 'run_type', 'description', 'priority',
		'status', 'processing_stage', 'progress_percentage', 'employee_count',
		'processed_employee_count', 'error_count', 'warning_count',
		'validation_score', 'compliance_score', 'total_gross_pay',
		'total_net_pay', 'started_at', 'completed_at', 'analytics_data'
	]
	
	show_widget = PayrollDashboardWidget
	
	# Order and pagination
	base_order = ('started_at', 'desc')
	page_size = 15
	
	# Permissions
	base_permissions = ['can_list', 'can_show']
	
	@expose('/monitor/<run_id>')
	@require_permission('monitor_payroll')
	def monitor_processing(self, run_id):
		"""Real-time payroll processing monitor."""
		try:
			run = self.datamodel.get(run_id, add_base_filters=True)
			if not run:
				flash(gettext('Payroll run not found'), 'error')
				return redirect(url_for('PayrollRunView.list'))
			
			# Get real-time processing status
			service = RevolutionaryPayrollService()
			status = service.get_payroll_status(run_id, run.tenant_id)
			
			return render_template(
				'payroll/processing_monitor.html',
				run=run,
				status=status,
				title=f'Processing Monitor - {run.run_name}'
			)
			
		except Exception as e:
			logger.error(f"Failed to load processing monitor: {e}")
			flash(gettext('Failed to load processing monitor'), 'error')
			return redirect(url_for('PayrollRunView.list'))
	
	@expose('/api/status/<run_id>')
	@require_permission('monitor_payroll')
	def api_get_status(self, run_id):
		"""API endpoint for real-time status updates."""
		try:
			service = RevolutionaryPayrollService()
			status = service.get_payroll_status(run_id, self.current_user.tenant_id)
			return jsonify(status)
			
		except Exception as e:
			logger.error(f"Failed to get payroll status: {e}")
			return jsonify({'error': str(e)}), 500
	
	@action('approve_payroll', lazy_gettext('Approve Payroll'), lazy_gettext('Approve selected payroll runs?'), 'fa-check')
	@require_permission('approve_payroll')
	@notify_on_action('payroll_approved')
	def approve_payroll_action(self, items):
		"""Approve selected payroll runs."""
		try:
			service = RevolutionaryPayrollService()
			success_count = 0
			
			for run in items:
				if run.status == PayrollStatus.COMPLIANCE_CHECK:
					service.approve_payroll_run(
						run.run_id, 
						run.tenant_id, 
						self.current_user.id
					)
					success_count += 1
			
			if success_count > 0:
				flash(gettext(f'Approved {success_count} payroll runs'), 'success')
			else:
				flash(gettext('No eligible runs selected'), 'warning')
				
		except Exception as e:
			logger.error(f"Failed to approve payroll: {e}")
			flash(gettext('Failed to approve payroll runs'), 'error')
		
		return redirect(url_for('PayrollRunView.list'))


class EmployeePayrollView(ModelView, AuditMixin):
	"""Revolutionary employee payroll management with AI insights."""
	
	datamodel = SQLAInterface(PREmployeePayroll)
	
	# View configuration
	list_title = lazy_gettext('Employee Payroll - AI-Enhanced Analytics')
	show_title = lazy_gettext('Employee Payroll Details')
	
	# List view configuration
	list_columns = [
		'employee_name', 'employee_number', 'department_name', 
		'gross_earnings', 'total_deductions', 'total_taxes', 'net_pay',
		'regular_hours', 'overtime_hours', 'validation_score', 'has_errors'
	]
	
	list_widget = ImmersivePayrollWidget
	
	# Search configuration
	search_columns = [
		'employee_name', 'employee_number', 'department_name', 'position_title'
	]
	
	# Show view configuration
	show_columns = [
		'employee_name', 'employee_number', 'department_name', 'position_title',
		'gross_earnings', 'total_deductions', 'total_taxes', 'net_pay',
		'regular_hours', 'overtime_hours', 'holiday_hours', 'sick_hours',
		'validation_score', 'has_errors', 'has_warnings', 'ai_recommendations',
		'ytd_gross', 'ytd_deductions', 'ytd_taxes', 'ytd_net'
	]
	
	show_widget = PayrollDashboardWidget
	
	# Order and pagination
	base_order = ('employee_name', 'asc')
	page_size = 25
	
	# Permissions
	base_permissions = ['can_list', 'can_show']
	
	@expose('/ai_analysis/<employee_payroll_id>')
	@require_permission('view_ai_analysis')
	@ai_enhanced
	def ai_analysis(self, employee_payroll_id):
		"""Display AI analysis for employee payroll."""
		try:
			payroll = self.datamodel.get(employee_payroll_id, add_base_filters=True)
			if not payroll:
				flash(gettext('Employee payroll not found'), 'error')
				return redirect(url_for('EmployeePayrollView.list'))
			
			# Get AI analysis
			intelligence_engine = PayrollIntelligenceEngine()
			analysis = intelligence_engine.analyze_employee_payroll(employee_payroll_id)
			
			return render_template(
				'payroll/employee_ai_analysis.html',
				payroll=payroll,
				analysis=analysis,
				title=f'AI Analysis - {payroll.employee_name}'
			)
			
		except Exception as e:
			logger.error(f"Failed to load AI analysis: {e}")
			flash(gettext('Failed to load AI analysis'), 'error')
			return redirect(url_for('EmployeePayrollView.list'))


class ConversationalPayrollView(BaseView):
	"""Revolutionary conversational payroll interface."""
	
	route_base = '/payroll/chat'
	default_view = 'index'
	
	@expose('/')
	@require_permission('use_conversational_interface')
	def index(self):
		"""Main conversational interface."""
		form = ConversationalPayrollForm()
		
		return self.render_template(
			'payroll/conversational_interface.html',
			form=form,
			title='Conversational Payroll Assistant'
		)
	
	@expose('/process', methods=['POST'])
	@require_permission('use_conversational_interface')
	def process_command(self):
		"""Process natural language payroll command."""
		try:
			form = ConversationalPayrollForm()
			
			if form.validate_on_submit():
				command = form.command.data
				
				# Process command with conversational assistant
				assistant = ConversationalPayrollAssistant()
				response = assistant.process_command(
					command=command,
					user_id=self.current_user.id,
					tenant_id=self.current_user.tenant_id
				)
				
				return jsonify({
					'success': True,
					'response': response,
					'timestamp': datetime.utcnow().isoformat()
				})
			else:
				return jsonify({
					'success': False,
					'errors': form.errors
				}), 400
				
		except Exception as e:
			logger.error(f"Failed to process conversational command: {e}")
			return jsonify({
				'success': False,
				'error': 'Failed to process command'
			}), 500


class PayrollAnalyticsView(BaseView):
	"""Advanced payroll analytics and reporting dashboard."""
	
	route_base = '/payroll/analytics'
	default_view = 'dashboard'
	
	@expose('/')
	@require_permission('view_payroll_analytics')
	def dashboard(self):
		"""Main analytics dashboard."""
		try:
			# Get analytics data
			intelligence_engine = PayrollIntelligenceEngine()
			analytics = intelligence_engine.get_dashboard_analytics(
				tenant_id=self.current_user.tenant_id
			)
			
			return self.render_template(
				'payroll/analytics_dashboard.html',
				analytics=analytics,
				title='Payroll Analytics Dashboard'
			)
			
		except Exception as e:
			logger.error(f"Failed to load analytics dashboard: {e}")
			flash(gettext('Failed to load analytics'), 'error')
			return redirect(url_for('PayrollPeriodView.list'))
	
	@expose('/efficiency')
	@require_permission('view_efficiency_metrics')
	def efficiency_metrics(self):
		"""Payroll processing efficiency metrics."""
		try:
			# Get efficiency data from the last 12 months
			start_date = date.today() - timedelta(days=365)
			end_date = date.today()
			
			# This would query the efficiency function from schema.sql
			efficiency_data = []  # Database query result
			
			return self.render_template(
				'payroll/efficiency_metrics.html',
				efficiency_data=efficiency_data,
				start_date=start_date,
				end_date=end_date,
				title='Processing Efficiency Metrics'
			)
			
		except Exception as e:
			logger.error(f"Failed to load efficiency metrics: {e}")
			flash(gettext('Failed to load efficiency metrics'), 'error')
			return redirect(url_for('PayrollAnalyticsView.dashboard'))
	
	@expose('/anomalies')
	@require_permission('view_anomaly_detection')
	def anomaly_detection(self):
		"""AI-powered anomaly detection dashboard."""
		try:
			# Get current period anomalies
			intelligence_engine = PayrollIntelligenceEngine()
			anomalies = intelligence_engine.detect_current_period_anomalies(
				tenant_id=self.current_user.tenant_id
			)
			
			return self.render_template(
				'payroll/anomaly_detection.html',
				anomalies=anomalies,
				title='Payroll Anomaly Detection'
			)
			
		except Exception as e:
			logger.error(f"Failed to load anomaly detection: {e}")
			flash(gettext('Failed to load anomaly detection'), 'error')
			return redirect(url_for('PayrollAnalyticsView.dashboard'))


class PayrollComplianceView(BaseView):
	"""Revolutionary compliance monitoring and reporting."""
	
	route_base = '/payroll/compliance'
	default_view = 'dashboard'
	
	@expose('/')
	@require_permission('view_compliance_dashboard')
	def dashboard(self):
		"""Main compliance dashboard."""
		try:
			# Get compliance status
			from .compliance_tax_engine import IntelligentComplianceTaxEngine
			
			compliance_engine = IntelligentComplianceTaxEngine()
			compliance_status = compliance_engine.get_compliance_dashboard(
				tenant_id=self.current_user.tenant_id
			)
			
			return self.render_template(
				'payroll/compliance_dashboard.html',
				compliance_status=compliance_status,
				title='Compliance Monitoring Dashboard'
			)
			
		except Exception as e:
			logger.error(f"Failed to load compliance dashboard: {e}")
			flash(gettext('Failed to load compliance dashboard'), 'error')
			return redirect(url_for('PayrollPeriodView.list'))
	
	@expose('/audit_trail/<period_id>')
	@require_permission('view_audit_trail')
	def audit_trail(self, period_id):
		"""Display audit trail for payroll period."""
		try:
			# Get audit trail data
			audit_data = []  # Query from pr_audit_log table
			
			return self.render_template(
				'payroll/audit_trail.html',
				audit_data=audit_data,
				period_id=period_id,
				title='Payroll Audit Trail'
			)
			
		except Exception as e:
			logger.error(f"Failed to load audit trail: {e}")
			flash(gettext('Failed to load audit trail'), 'error')
			return redirect(url_for('PayrollPeriodView.list'))


# Chart views for advanced analytics
class PayrollTrendsChartView(DirectByChartView):
	"""Payroll trends chart view."""
	
	chart_title = lazy_gettext('Payroll Trends')
	chart_type = 'LineChart'
	direct_columns = {
		'period_name': 'Period',
		'total_gross': 'Gross Pay',
		'total_net': 'Net Pay',
		'employee_count': 'Employee Count'
	}
	base_order = ('start_date', 'asc')


class DepartmentPayrollChartView(DirectByChartView):
	"""Department payroll distribution chart."""
	
	chart_title = lazy_gettext('Payroll by Department')
	chart_type = 'PieChart'
	direct_columns = {
		'department_name': 'Department',
		'total_gross': 'Total Gross Pay'
	}
	group_by_columns = ['department_name']


# Register views with Flask-AppBuilder
def register_payroll_views(appbuilder):
	"""Register all payroll views with Flask-AppBuilder."""
	
	# Main CRUD views
	appbuilder.add_view(
		PayrollPeriodView,
		"Payroll Periods",
		icon="fa-calendar",
		category="Payroll Management",
		category_icon="fa-money"
	)
	
	appbuilder.add_view(
		PayrollRunView,
		"Payroll Runs",
		icon="fa-cogs",
		category="Payroll Management"
	)
	
	appbuilder.add_view(
		EmployeePayrollView,
		"Employee Payroll",
		icon="fa-users",
		category="Payroll Management"
	)
	
	# Advanced features
	appbuilder.add_view(
		ConversationalPayrollView,
		"AI Assistant",
		icon="fa-comments",
		category="Payroll Management"
	)
	
	appbuilder.add_view(
		PayrollAnalyticsView,
		"Analytics Dashboard",
		icon="fa-line-chart",
		category="Payroll Analytics",
		category_icon="fa-bar-chart"
	)
	
	appbuilder.add_view(
		PayrollComplianceView,
		"Compliance Monitor",
		icon="fa-shield",
		category="Payroll Analytics"
	)
	
	# Chart views
	appbuilder.add_view(
		PayrollTrendsChartView,
		"Payroll Trends",
		icon="fa-line-chart",
		category="Payroll Analytics"
	)
	
	appbuilder.add_view(
		DepartmentPayrollChartView,
		"Department Analysis",
		icon="fa-pie-chart",
		category="Payroll Analytics"
	)


# Example usage
if __name__ == "__main__":
	# This would be used in the main Flask-AppBuilder application
	pass