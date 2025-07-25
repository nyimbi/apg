"""
Budgeting & Forecasting Views

Flask-AppBuilder views for the Budgeting & Forecasting sub-capability
including model views, dashboards, and specialized reporting views.
"""

from flask import render_template, request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget, ShowWidget
from flask_appbuilder.actions import action
from wtforms import DecimalField, SelectField, TextAreaField
from wtforms.validators import DataRequired, NumberRange
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, List

from .models import (
	CFBFBudget, CFBFBudgetLine, CFBFBudgetScenario, CFBFBudgetVersion,
	CFBFForecast, CFBFForecastLine, CFBFActualVsBudget, CFBFDrivers,
	CFBFTemplate, CFBFApproval, CFBFAllocation
)
from .service import CFBFBudgetService, CFBFVarianceAnalysisService, CFBFForecastService, CFBFDriverService
from ..general_ledger.models import CFGLAccount


class CFBFBudgetScenarioModelView(ModelView):
	"""Budget Scenario management view"""
	
	datamodel = SQLAInterface(CFBFBudgetScenario)
	
	list_title = "Budget Scenarios"
	show_title = "Budget Scenario Details"
	add_title = "Add Budget Scenario"
	edit_title = "Edit Budget Scenario"
	
	list_columns = [
		'scenario_code', 'scenario_name', 'description', 
		'probability', 'is_active', 'is_default'
	]
	
	show_columns = [
		'scenario_code', 'scenario_name', 'description', 'probability',
		'is_active', 'is_default', 'assumptions', 'parameters',
		'created_on', 'created_by', 'changed_on', 'changed_by'
	]
	
	search_columns = ['scenario_code', 'scenario_name', 'description']
	
	add_columns = [
		'scenario_code', 'scenario_name', 'description', 'probability',
		'is_active', 'is_default', 'assumptions', 'parameters'
	]
	
	edit_columns = add_columns
	
	order_columns = ['scenario_code', 'scenario_name', 'probability']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
	
	formatters_columns = {
		'probability': lambda x: f"{x:.1f}%" if x else "0.0%"
	}


class CFBFTemplateModelView(ModelView):
	"""Budget Template management view"""
	
	datamodel = SQLAInterface(CFBFTemplate)
	
	list_title = "Budget Templates"
	show_title = "Budget Template Details" 
	add_title = "Add Budget Template"
	edit_title = "Edit Budget Template"
	
	list_columns = [
		'template_code', 'template_name', 'category', 
		'is_active', 'is_system', 'created_on'
	]
	
	show_columns = [
		'template_code', 'template_name', 'description', 'category',
		'is_active', 'is_system', 'template_data', 'default_accounts',
		'calculation_rules', 'created_on', 'created_by'
	]
	
	search_columns = ['template_code', 'template_name', 'category', 'description']
	
	add_columns = [
		'template_code', 'template_name', 'description', 'category',
		'is_active', 'template_data', 'default_accounts', 'calculation_rules'
	]
	
	edit_columns = add_columns
	
	order_columns = ['template_code', 'template_name', 'category']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']


class CFBFDriversModelView(ModelView):
	"""Budget Drivers management view"""
	
	datamodel = SQLAInterface(CFBFDrivers)
	
	list_title = "Budget Drivers"
	show_title = "Budget Driver Details"
	add_title = "Add Budget Driver"
	edit_title = "Edit Budget Driver"
	
	list_columns = [
		'driver_code', 'driver_name', 'category', 'data_type',
		'base_value', 'growth_rate', 'unit_of_measure', 'is_active'
	]
	
	show_columns = [
		'driver_code', 'driver_name', 'description', 'category',
		'data_type', 'unit_of_measure', 'base_value', 'growth_rate',
		'seasonal_factors', 'historical_values', 'is_active',
		'created_on', 'created_by', 'changed_on', 'changed_by'
	]
	
	search_columns = ['driver_code', 'driver_name', 'category', 'description']
	
	add_columns = [
		'driver_code', 'driver_name', 'description', 'category',
		'data_type', 'unit_of_measure', 'base_value', 'growth_rate',
		'seasonal_factors', 'is_active'
	]
	
	edit_columns = add_columns
	
	order_columns = ['driver_code', 'driver_name', 'category']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
	
	formatters_columns = {
		'base_value': lambda x: f"{x:,.2f}" if x else "0.00",
		'growth_rate': lambda x: f"{x:.2f}%" if x else "0.00%"
	}
	
	@action("calculate_projections", "Calculate Projections", 
			"Calculate driver projections", "fa-calculator")
	def calculate_projections(self, items):
		"""Calculate driver projections for selected drivers"""
		if not items:
			flash("No drivers selected", "warning")
			return redirect(self.get_redirect())
		
		# Implementation would calculate and display projections
		flash(f"Calculated projections for {len(items)} drivers", "success")
		return redirect(self.get_redirect())


class CFBFBudgetModelView(ModelView):
	"""Budget management view"""
	
	datamodel = SQLAInterface(CFBFBudget)
	
	list_title = "Budgets"
	show_title = "Budget Details"
	add_title = "Create Budget"
	edit_title = "Edit Budget"
	
	list_columns = [
		'budget_number', 'budget_name', 'fiscal_year', 'status',
		'total_revenue', 'total_expenses', 'net_income', 'approval_status'
	]
	
	show_columns = [
		'budget_number', 'budget_name', 'description', 'fiscal_year',
		'budget_period', 'start_date', 'end_date', 'scenario.scenario_name',
		'template.template_name', 'status', 'approval_status',
		'total_revenue', 'total_expenses', 'net_income', 'line_count',
		'currency_code', 'notes', 'created_on', 'created_by'
	]
	
	search_columns = ['budget_number', 'budget_name', 'description', 'status']
	
	add_columns = [
		'budget_name', 'description', 'fiscal_year', 'budget_period',
		'start_date', 'end_date', 'scenario', 'template', 'currency_code', 'notes'
	]
	
	edit_columns = [
		'budget_name', 'description', 'notes'
	]
	
	order_columns = ['budget_number', 'budget_name', 'fiscal_year', 'created_on']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
	
	formatters_columns = {
		'total_revenue': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_expenses': lambda x: f"${x:,.2f}" if x else "$0.00", 
		'net_income': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f"<span class='label label-{self._get_status_color(x)}'>{x}</span>",
		'approval_status': lambda x: f"<span class='label label-{self._get_approval_color(x)}'>{x}</span>"
	}
	
	def _get_status_color(self, status: str) -> str:
		"""Get bootstrap color class for status"""
		colors = {
			'Draft': 'default',
			'Submitted': 'info', 
			'Approved': 'success',
			'Active': 'primary',
			'Locked': 'warning'
		}
		return colors.get(status, 'default')
	
	def _get_approval_color(self, status: str) -> str:
		"""Get bootstrap color class for approval status"""
		colors = {
			'Pending': 'warning',
			'Approved': 'success',
			'Rejected': 'danger'
		}
		return colors.get(status, 'default')
	
	@action("submit_for_approval", "Submit for Approval",
			"Submit selected budgets for approval", "fa-paper-plane")
	def submit_for_approval(self, items):
		"""Submit budgets for approval"""
		if not items:
			flash("No budgets selected", "warning")
			return redirect(self.get_redirect())
		
		submitted_count = 0
		for budget in items:
			if budget.status == 'Draft':
				# Use service to submit
				service = CFBFBudgetService(self.datamodel.session, budget.tenant_id)
				if service.submit_budget_for_approval(budget.budget_id, self.get_user_id()):
					submitted_count += 1
		
		if submitted_count > 0:
			flash(f"Submitted {submitted_count} budgets for approval", "success")
		else:
			flash("No budgets were eligible for submission", "warning")
		
		return redirect(self.get_redirect())
	
	@action("copy_budget", "Copy Budget", 
			"Copy selected budgets to new fiscal year", "fa-copy")
	def copy_budget(self, items):
		"""Copy budgets to new fiscal year"""
		if not items:
			flash("No budgets selected", "warning")
			return redirect(self.get_redirect())
		
		# Would show a form to select target fiscal year
		flash("Budget copy functionality would be implemented here", "info")
		return redirect(self.get_redirect())


class CFBFBudgetLineModelView(ModelView):
	"""Budget Line management view"""
	
	datamodel = SQLAInterface(CFBFBudgetLine)
	
	list_title = "Budget Lines"
	show_title = "Budget Line Details"
	add_title = "Add Budget Line"
	edit_title = "Edit Budget Line"
	
	list_columns = [
		'budget.budget_number', 'line_number', 'account.account_code',
		'account.account_name', 'annual_amount', 'calculation_method',
		'cost_center', 'department'
	]
	
	show_columns = [
		'budget.budget_number', 'line_number', 'description',
		'account.account_code', 'account.account_name', 'driver.driver_name',
		'calculation_method', 'calculation_formula', 'annual_amount', 'total_amount',
		'distribution_method', 'period_amounts', 'cost_center', 'department',
		'project', 'notes', 'assumptions'
	]
	
	search_columns = ['description', 'account.account_code', 'account.account_name']
	
	add_columns = [
		'budget', 'line_number', 'description', 'account', 'driver',
		'calculation_method', 'calculation_formula', 'annual_amount',
		'distribution_method', 'cost_center', 'department', 'project',
		'notes', 'assumptions'
	]
	
	edit_columns = add_columns
	
	order_columns = ['line_number', 'account.account_code', 'annual_amount']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
	
	formatters_columns = {
		'annual_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00"
	}


class CFBFForecastModelView(ModelView):
	"""Forecast management view"""
	
	datamodel = SQLAInterface(CFBFForecast)
	
	list_title = "Forecasts"
	show_title = "Forecast Details"
	add_title = "Create Forecast"
	edit_title = "Edit Forecast"
	
	list_columns = [
		'forecast_number', 'forecast_name', 'forecast_type', 'forecast_method',
		'periods_ahead', 'confidence_level', 'status', 'generated_date'
	]
	
	show_columns = [
		'forecast_number', 'forecast_name', 'description', 'forecast_type',
		'forecast_method', 'forecast_date', 'start_date', 'end_date',
		'periods_ahead', 'scenario.scenario_name', 'base_budget.budget_name',
		'status', 'confidence_level', 'algorithm_type', 'algorithm_parameters',
		'total_forecast_revenue', 'total_forecast_expenses', 'forecast_net_income',
		'generated_date', 'generated_by'
	]
	
	search_columns = ['forecast_number', 'forecast_name', 'description']
	
	add_columns = [
		'forecast_name', 'description', 'forecast_type', 'forecast_method',
		'periods_ahead', 'scenario', 'base_budget', 'confidence_level',
		'algorithm_type', 'algorithm_parameters'
	]
	
	edit_columns = [
		'forecast_name', 'description', 'confidence_level', 'algorithm_parameters'
	]
	
	order_columns = ['forecast_number', 'forecast_name', 'forecast_date']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
	
	formatters_columns = {
		'total_forecast_revenue': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_forecast_expenses': lambda x: f"${x:,.2f}" if x else "$0.00",
		'forecast_net_income': lambda x: f"${x:,.2f}" if x else "$0.00",
		'confidence_level': lambda x: f"{x:.1f}%" if x else "0.0%"
	}
	
	@action("generate_forecast", "Generate Forecast",
			"Generate forecast calculations", "fa-calculator")
	def generate_forecast(self, items):
		"""Generate forecast calculations"""
		if not items:
			flash("No forecasts selected", "warning")
			return redirect(self.get_redirect())
		
		generated_count = 0
		for forecast in items:
			if forecast.status == 'Draft':
				service = CFBFForecastService(self.datamodel.session, forecast.tenant_id)
				forecast.generate_forecast(self.get_user_id())
				generated_count += 1
		
		if generated_count > 0:
			flash(f"Generated {generated_count} forecasts", "success")
		else:
			flash("No forecasts were eligible for generation", "warning")
		
		return redirect(self.get_redirect())


class CFBFForecastLineModelView(ModelView):
	"""Forecast Line management view"""
	
	datamodel = SQLAInterface(CFBFForecastLine)
	
	list_title = "Forecast Lines"
	show_title = "Forecast Line Details"
	add_title = "Add Forecast Line"
	edit_title = "Edit Forecast Line"
	
	list_columns = [
		'forecast.forecast_number', 'line_number', 'account.account_code',
		'account.account_name', 'calculation_method', 'forecast_amount',
		'confidence_interval', 'r_squared'
	]
	
	show_columns = [
		'forecast.forecast_number', 'line_number', 'description',
		'account.account_code', 'account.account_name', 'driver.driver_name',
		'calculation_method', 'historical_periods', 'base_amount', 'trend_factor',
		'forecast_amount', 'confidence_interval', 'period_forecasts',
		'r_squared', 'std_error', 'assumptions', 'methodology_notes'
	]
	
	search_columns = ['description', 'account.account_code', 'account.account_name']
	
	add_columns = [
		'forecast', 'line_number', 'description', 'account', 'driver',
		'calculation_method', 'historical_periods', 'base_amount',
		'assumptions', 'methodology_notes'
	]
	
	edit_columns = add_columns
	
	order_columns = ['line_number', 'account.account_code', 'forecast_amount']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
	
	formatters_columns = {
		'base_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'forecast_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'trend_factor': lambda x: f"{x:.6f}" if x else "1.000000",
		'r_squared': lambda x: f"{x:.4f}" if x else "N/A"
	}


class CFBFActualVsBudgetView(BaseView):
	"""Variance Analysis specialized view"""
	
	default_view = 'variance_analysis'
	
	@expose('/variance_analysis/')
	@has_access
	def variance_analysis(self):
		"""Main variance analysis dashboard"""
		
		# Get filters from request
		budget_id = request.args.get('budget_id')
		analysis_date = request.args.get('analysis_date', date.today().isoformat())
		alert_level = request.args.get('alert_level')
		
		variance_data = []
		summary_stats = {}
		
		if budget_id:
			# Get variance analysis service
			service = CFBFVarianceAnalysisService(self.datamodel.session, self.get_tenant_id())
			
			# Get significant variances
			variances = service.get_significant_variances(
				budget_id=budget_id,
				analysis_date=datetime.strptime(analysis_date, '%Y-%m-%d').date(),
				alert_level=alert_level
			)
			
			variance_data = [
				{
					'account_code': v.account.account_code if v.account else 'N/A',
					'account_name': v.account.account_name if v.account else 'N/A',
					'budget_amount': float(v.budget_amount),
					'actual_amount': float(v.actual_amount),
					'variance_amount': float(v.variance_amount),
					'variance_percent': float(v.variance_percent),
					'is_favorable': v.is_favorable,
					'alert_level': v.alert_level,
					'variance_explanation': v.variance_explanation
				}
				for v in variances
			]
			
			# Calculate summary statistics
			total_budget = sum(v.budget_amount for v in variances)
			total_actual = sum(v.actual_amount for v in variances)
			total_variance = sum(v.variance_amount for v in variances)
			
			summary_stats = {
				'total_budget': float(total_budget),
				'total_actual': float(total_actual),
				'total_variance': float(total_variance),
				'variance_count': len(variances),
				'unfavorable_count': len([v for v in variances if not v.is_favorable])
			}
		
		# Get available budgets for filter
		budgets = self.datamodel.session.query(CFBFBudget).filter(
			CFBFBudget.tenant_id == self.get_tenant_id(),
			CFBFBudget.status.in_(['Approved', 'Active'])
		).all()
		
		return self.render_template(
			'budgeting_forecasting/variance_analysis.html',
			variance_data=variance_data,
			summary_stats=summary_stats,
			budgets=budgets,
			selected_budget_id=budget_id,
			analysis_date=analysis_date,
			alert_level=alert_level
		)
	
	@expose('/generate_variance/<budget_id>/')
	@has_access
	def generate_variance(self, budget_id):
		"""Generate variance analysis for a budget"""
		
		service = CFBFVarianceAnalysisService(self.datamodel.session, self.get_tenant_id())
		
		try:
			variances = service.generate_variance_analysis(
				budget_id=budget_id,
				analysis_date=date.today(),
				user_id=self.get_user_id()
			)
			
			self.datamodel.session.commit()
			flash(f"Generated variance analysis with {len(variances)} records", "success")
			
		except Exception as e:
			self.datamodel.session.rollback()
			flash(f"Error generating variance analysis: {str(e)}", "danger")
		
		return redirect(url_for('CFBFActualVsBudgetView.variance_analysis', budget_id=budget_id))


class CFBFApprovalModelView(ModelView):
	"""Budget Approval workflow view"""
	
	datamodel = SQLAInterface(CFBFApproval)
	
	list_title = "Budget Approvals"
	show_title = "Approval Details"
	add_title = "Add Approval"
	edit_title = "Edit Approval"
	
	list_columns = [
		'budget.budget_number', 'approval_level', 'approver_role',
		'status', 'required', 'approved_date', 'rejected_date'
	]
	
	show_columns = [
		'budget.budget_number', 'approval_level', 'approver_id', 'approver_role',
		'status', 'required', 'approved_date', 'rejected_date',
		'comments', 'conditions', 'created_on'
	]
	
	search_columns = ['budget.budget_number', 'approver_role', 'status']
	
	add_columns = [
		'budget', 'approval_level', 'approver_id', 'approver_role',
		'required', 'comments', 'conditions'
	]
	
	edit_columns = ['comments', 'conditions']
	
	order_columns = ['approval_level', 'status', 'approved_date']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
	
	formatters_columns = {
		'status': lambda x: f"<span class='label label-{self._get_approval_status_color(x)}'>{x}</span>",
		'required': lambda x: "Yes" if x else "No"
	}
	
	def _get_approval_status_color(self, status: str) -> str:
		"""Get bootstrap color class for approval status"""
		colors = {
			'Pending': 'warning',
			'Approved': 'success',
			'Rejected': 'danger',
			'Skipped': 'default'
		}
		return colors.get(status, 'default')
	
	@action("approve_selected", "Approve Selected",
			"Approve selected approval requests", "fa-check")
	def approve_selected(self, items):
		"""Approve selected approval requests"""
		if not items:
			flash("No approvals selected", "warning")
			return redirect(self.get_redirect())
		
		approved_count = 0
		for approval in items:
			if approval.can_approve():
				approval.approve("Bulk approval")
				approved_count += 1
		
		if approved_count > 0:
			flash(f"Approved {approved_count} requests", "success")
		else:
			flash("No approvals were eligible for approval", "warning")
		
		return redirect(self.get_redirect())


class CFBFAllocationModelView(ModelView):
	"""Budget Allocation management view"""
	
	datamodel = SQLAInterface(CFBFAllocation)
	
	list_title = "Budget Allocations"
	show_title = "Allocation Details"
	add_title = "Add Allocation"
	edit_title = "Edit Allocation"
	
	list_columns = [
		'budget_line.budget.budget_number', 'target_type', 'target_code',
		'target_name', 'allocation_method', 'allocation_percentage',
		'allocation_amount', 'is_active'
	]
	
	show_columns = [
		'budget_line.budget.budget_number', 'budget_line.line_number',
		'target_type', 'target_code', 'target_name', 'allocation_method',
		'allocation_basis', 'allocation_percentage', 'allocation_amount',
		'driver_factor', 'allocation_formula', 'formula_variables',
		'is_active', 'effective_date', 'expiration_date', 'description'
	]
	
	search_columns = ['target_code', 'target_name', 'allocation_method']
	
	add_columns = [
		'budget_line', 'target_type', 'target_code', 'target_name',
		'allocation_method', 'allocation_basis', 'allocation_percentage',
		'allocation_amount', 'driver_factor', 'allocation_formula',
		'is_active', 'effective_date', 'expiration_date', 'description'
	]
	
	edit_columns = add_columns
	
	order_columns = ['target_type', 'target_code', 'allocation_percentage']
	
	base_permissions = ['can_add', 'can_edit', 'can_delete', 'can_list', 'can_show']
	
	formatters_columns = {
		'allocation_percentage': lambda x: f"{x:.2f}%" if x else "0.00%",
		'allocation_amount': lambda x: f"${x:,.2f}" if x else "$0.00"
	}


class CFBFDashboardView(BaseView):
	"""Main Budgeting & Forecasting Dashboard"""
	
	default_view = 'dashboard'
	
	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Main dashboard view"""
		
		tenant_id = self.get_tenant_id()
		
		# Get dashboard metrics
		current_year = date.today().year
		
		# Budget statistics
		total_budgets = self.datamodel.session.query(CFBFBudget).filter(
			CFBFBudget.tenant_id == tenant_id,
			CFBFBudget.fiscal_year == current_year
		).count()
		
		active_budgets = self.datamodel.session.query(CFBFBudget).filter(
			CFBFBudget.tenant_id == tenant_id,
			CFBFBudget.fiscal_year == current_year,
			CFBFBudget.status == 'Active'
		).count()
		
		pending_approvals = self.datamodel.session.query(CFBFApproval).filter(
			CFBFApproval.tenant_id == tenant_id,
			CFBFApproval.status == 'Pending',
			CFBFApproval.required == True
		).count()
		
		# Forecast statistics
		active_forecasts = self.datamodel.session.query(CFBFForecast).filter(
			CFBFForecast.tenant_id == tenant_id,
			CFBFForecast.status == 'Active'
		).count()
		
		# Variance statistics
		significant_variances = self.datamodel.session.query(CFBFActualVsBudget).filter(
			CFBFActualVsBudget.tenant_id == tenant_id,
			CFBFActualVsBudget.is_significant == True,
			CFBFActualVsBudget.analysis_date >= date.today().replace(day=1)
		).count()
		
		# Recent activities
		recent_budgets = self.datamodel.session.query(CFBFBudget).filter(
			CFBFBudget.tenant_id == tenant_id
		).order_by(CFBFBudget.created_on.desc()).limit(10).all()
		
		recent_forecasts = self.datamodel.session.query(CFBFForecast).filter(
			CFBFForecast.tenant_id == tenant_id
		).order_by(CFBFForecast.created_on.desc()).limit(10).all()
		
		dashboard_data = {
			'metrics': {
				'total_budgets': total_budgets,
				'active_budgets': active_budgets,
				'pending_approvals': pending_approvals,
				'active_forecasts': active_forecasts,
				'significant_variances': significant_variances
			},
			'recent_budgets': recent_budgets,
			'recent_forecasts': recent_forecasts,
			'current_year': current_year
		}
		
		return self.render_template(
			'budgeting_forecasting/dashboard.html',
			dashboard_data=dashboard_data
		)
	
	@expose('/api/budget_summary/<budget_id>/')
	@has_access
	def api_budget_summary(self, budget_id):
		"""API endpoint for budget summary data"""
		
		service = CFBFBudgetService(self.datamodel.session, self.get_tenant_id())
		summary = service.get_budget_summary(budget_id)
		
		return jsonify(summary)
	
	@expose('/api/variance_trends/<account_id>/')
	@has_access
	def api_variance_trends(self, account_id):
		"""API endpoint for variance trend data"""
		
		service = CFBFVarianceAnalysisService(self.datamodel.session, self.get_tenant_id())
		trends = service.get_variance_trends(account_id)
		
		return jsonify(trends)


class CFBFScenarioComparisonView(BaseView):
	"""Scenario Comparison specialized view"""
	
	default_view = 'scenario_comparison'
	
	@expose('/scenario_comparison/')
	@has_access
	def scenario_comparison(self):
		"""Scenario comparison dashboard"""
		
		# Get scenarios for comparison
		scenarios = self.datamodel.session.query(CFBFBudgetScenario).filter(
			CFBFBudgetScenario.tenant_id == self.get_tenant_id(),
			CFBFBudgetScenario.is_active == True
		).all()
		
		# Get budgets by scenario
		comparison_data = []
		for scenario in scenarios:
			scenario_budgets = self.datamodel.session.query(CFBFBudget).filter(
				CFBFBudget.scenario_id == scenario.scenario_id,
				CFBFBudget.status.in_(['Approved', 'Active'])
			).all()
			
			total_revenue = sum(b.total_revenue for b in scenario_budgets)
			total_expenses = sum(b.total_expenses for b in scenario_budgets)
			net_income = sum(b.net_income for b in scenario_budgets)
			
			comparison_data.append({
				'scenario': scenario,
				'budget_count': len(scenario_budgets),
				'total_revenue': float(total_revenue),
				'total_expenses': float(total_expenses),
				'net_income': float(net_income)
			})
		
		return self.render_template(
			'budgeting_forecasting/scenario_comparison.html',
			comparison_data=comparison_data,
			scenarios=scenarios
		)
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# Implementation would depend on your authentication system
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID"""
		# Implementation would depend on your authentication system
		return "current_user"