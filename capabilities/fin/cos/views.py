"""
Cost Accounting Views

Flask-AppBuilder views for Cost Accounting functionality including
cost centers, activity-based costing, job costing, and variance analysis.
"""

from flask import flash, redirect, request, url_for, jsonify, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface  
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from wtforms import Form, StringField, SelectField, DecimalField, DateField, TextAreaField, BooleanField, IntegerField
from wtforms.validators import DataRequired, NumberRange, Optional
from datetime import date, datetime
from typing import Dict, List, Any
from decimal import Decimal

from .models import (
	CFCACostCenter, CFCACostCategory, CFCACostDriver, CFCACostAllocation,
	CFCACostPool, CFCAActivity, CFCAActivityCost, CFCAProductCost,
	CFCAJobCost, CFCAStandardCost, CFCAVarianceAnalysis
)
from .service import CostAccountingService, CostAllocationRequest
from ...auth_rbac.models import db


class CACostCenterModelView(ModelView):
	"""Cost Center Management View"""
	
	datamodel = SQLAInterface(CFCACostCenter)
	
	list_title = "Cost Centers"
	show_title = "Cost Center Details"
	add_title = "Add Cost Center"
	edit_title = "Edit Cost Center"
	
	list_columns = [
		'center_code', 'center_name', 'center_type', 'responsibility_type',
		'parent_center.center_name', 'manager_name', 'annual_budget', 'is_active'
	]
	
	show_columns = [
		'center_code', 'center_name', 'description', 'center_type', 'responsibility_type',
		'parent_center', 'level', 'path', 'manager_name', 'manager_email',
		'department', 'location', 'annual_budget', 'ytd_actual', 'ytd_budget',
		'effective_date', 'end_date', 'allow_cost_allocation', 'require_job_number',
		'default_currency', 'is_active', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'center_code', 'center_name', 'description', 'center_type', 'responsibility_type',
		'parent_center', 'manager_name', 'manager_email', 'department', 'location',
		'annual_budget', 'effective_date', 'end_date', 'allow_cost_allocation',
		'require_job_number', 'default_currency'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = ['center_code', 'center_name', 'center_type', 'manager_name', 'department']
	
	list_filters = ['center_type', 'responsibility_type', 'is_active', 'department']
	
	order_columns = ['center_code', 'center_name', 'center_type', 'annual_budget']
	
	formatters_columns = {
		'annual_budget': lambda x: f"${x:,.2f}" if x else "$0.00",
		'ytd_actual': lambda x: f"${x:,.2f}" if x else "$0.00",
		'ytd_budget': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	@expose('/hierarchy/')
	@has_access
	def hierarchy(self):
		"""Show cost center hierarchy"""
		service = CostAccountingService(tenant_id='default_tenant')  # TODO: Get from session
		hierarchy_data = service.get_cost_centers_hierarchy()
		
		return self.render_template(
			'cost_accounting/cost_center_hierarchy.html',
			hierarchy=hierarchy_data,
			title="Cost Center Hierarchy"
		)
	
	@expose('/budget_variance/<cost_center_id>/')
	@has_access
	def budget_variance(self, cost_center_id):
		"""Show budget variance analysis"""
		cost_center = CFCACostCenter.query.get(cost_center_id)
		if not cost_center:
			flash('Cost center not found', 'error')
			return redirect(url_for('CACostCenterModelView.list'))
		
		variance_data = cost_center.calculate_budget_variance()
		
		return self.render_template(
			'cost_accounting/budget_variance.html',
			cost_center=cost_center,
			variance=variance_data,
			title=f"Budget Variance - {cost_center.center_name}"
		)


class CACostCategoryModelView(ModelView):
	"""Cost Category Management View"""
	
	datamodel = SQLAInterface(CFCACostCategory)
	
	list_title = "Cost Categories"
	show_title = "Cost Category Details"
	add_title = "Add Cost Category"
	edit_title = "Edit Cost Category"
	
	list_columns = [
		'category_code', 'category_name', 'cost_type', 'cost_behavior',
		'cost_nature', 'parent_category.category_name', 'gl_account_code', 'is_active'
	]
	
	show_columns = [
		'category_code', 'category_name', 'description', 'cost_type', 'cost_behavior',
		'cost_nature', 'parent_category', 'level', 'path', 'is_variable', 'is_traceable',
		'is_controllable', 'gl_account_code', 'gl_account_id', 'effective_date',
		'end_date', 'is_active', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'category_code', 'category_name', 'description', 'cost_type', 'cost_behavior',
		'cost_nature', 'parent_category', 'is_variable', 'is_traceable', 'is_controllable',
		'gl_account_code', 'gl_account_id', 'effective_date', 'end_date'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = ['category_code', 'category_name', 'cost_type', 'cost_nature']
	
	list_filters = ['cost_type', 'cost_behavior', 'cost_nature', 'is_variable', 'is_active']
	
	order_columns = ['category_code', 'category_name', 'cost_type']


class CACostDriverModelView(ModelView):
	"""Cost Driver Management View"""
	
	datamodel = SQLAInterface(CFCACostDriver)
	
	list_title = "Cost Drivers"
	show_title = "Cost Driver Details"
	add_title = "Add Cost Driver"
	edit_title = "Edit Cost Driver"
	
	list_columns = [
		'driver_code', 'driver_name', 'unit_of_measure', 'driver_type',
		'default_rate', 'practical_capacity', 'is_active'
	]
	
	show_columns = [
		'driver_code', 'driver_name', 'description', 'unit_of_measure', 'driver_type',
		'is_volume_based', 'is_activity_based', 'requires_measurement',
		'calculation_method', 'calculation_frequency', 'default_rate',
		'current_capacity', 'practical_capacity', 'effective_date', 'end_date',
		'is_active', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'driver_code', 'driver_name', 'description', 'unit_of_measure', 'driver_type',
		'is_volume_based', 'is_activity_based', 'requires_measurement',
		'calculation_method', 'calculation_frequency', 'default_rate',
		'current_capacity', 'practical_capacity', 'effective_date', 'end_date'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = ['driver_code', 'driver_name', 'driver_type', 'unit_of_measure']
	
	list_filters = ['driver_type', 'is_volume_based', 'is_activity_based', 'is_active']
	
	formatters_columns = {
		'default_rate': lambda x: f"${x:,.4f}" if x else "-",
		'current_capacity': lambda x: f"{x:,.2f}" if x else "-",
		'practical_capacity': lambda x: f"{x:,.2f}" if x else "-"
	}


class CACostAllocationModelView(ModelView):
	"""Cost Allocation Rules Management View"""
	
	datamodel = SQLAInterface(CFCACostAllocation)
	
	list_title = "Cost Allocation Rules"
	show_title = "Cost Allocation Details"
	add_title = "Add Cost Allocation Rule"
	edit_title = "Edit Cost Allocation Rule"
	
	list_columns = [
		'allocation_code', 'allocation_name', 'allocation_method',
		'source_center.center_name', 'target_center.center_name',
		'cost_driver.driver_name', 'allocation_frequency', 'is_active'
	]
	
	show_columns = [
		'allocation_code', 'allocation_name', 'description', 'allocation_method',
		'source_center', 'target_center', 'cost_category', 'cost_driver',
		'allocation_basis', 'allocation_percent', 'allocation_formula',
		'allocation_frequency', 'effective_date', 'end_date', 'is_active',
		'is_automatic', 'requires_approval', 'last_allocation_date'
	]
	
	add_columns = [
		'allocation_code', 'allocation_name', 'description', 'allocation_method',
		'source_center', 'target_center', 'cost_category', 'cost_driver',
		'allocation_basis', 'allocation_percent', 'allocation_formula',
		'allocation_frequency', 'effective_date', 'end_date', 'is_automatic',
		'requires_approval'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = ['allocation_code', 'allocation_name', 'allocation_method']
	
	list_filters = ['allocation_method', 'allocation_frequency', 'is_active', 'is_automatic']
	
	formatters_columns = {
		'allocation_percent': lambda x: f"{x}%" if x else "-"
	}
	
	@expose('/execute/<allocation_id>/')
	@has_access
	def execute_allocation(self, allocation_id):
		"""Execute cost allocation"""
		allocation = CFCACostAllocation.query.get(allocation_id)
		if not allocation:
			flash('Allocation rule not found', 'error')
			return redirect(url_for('CACostAllocationModelView.list'))
		
		# Get cost amount from form or use test amount
		cost_amount = request.form.get('cost_amount', '10000')
		period = request.form.get('period', datetime.now().strftime('%Y-%m'))
		
		try:
			service = CostAccountingService(tenant_id='default_tenant')
			request_obj = CostAllocationRequest(
				source_center_id=allocation.source_center_id,
				allocation_method=allocation.allocation_method,
				cost_amount=Decimal(cost_amount),
				period=period,
				cost_driver_id=allocation.cost_driver_id
			)
			
			result = service.execute_cost_allocation(request_obj)
			
			flash(f'Cost allocation executed successfully. Allocated: ${result["total_allocated"]:,.2f}', 'success')
			
		except Exception as e:
			flash(f'Error executing allocation: {str(e)}', 'error')
		
		return redirect(url_for('CACostAllocationModelView.show', pk=allocation_id))


class CACostPoolModelView(ModelView):
	"""Cost Pool Management View"""
	
	datamodel = SQLAInterface(CFCACostPool)
	
	list_title = "Cost Pools"
	show_title = "Cost Pool Details"
	add_title = "Add Cost Pool"
	edit_title = "Edit Cost Pool"
	
	list_columns = [
		'pool_code', 'pool_name', 'pool_type', 'cost_behavior',
		'cost_center.center_name', 'budgeted_cost', 'actual_cost', 'is_active'
	]
	
	show_columns = [
		'pool_code', 'pool_name', 'description', 'pool_type', 'cost_behavior',
		'cost_center', 'primary_driver', 'budgeted_cost', 'actual_cost',
		'allocated_cost', 'budgeted_activity', 'actual_activity',
		'budgeted_rate', 'actual_rate', 'effective_date', 'end_date', 'is_active'
	]
	
	add_columns = [
		'pool_code', 'pool_name', 'description', 'pool_type', 'cost_behavior',
		'cost_center', 'primary_driver', 'budgeted_cost', 'budgeted_activity',
		'effective_date', 'end_date'
	]
	
	edit_columns = add_columns + ['actual_cost', 'allocated_cost', 'actual_activity', 'is_active']
	
	search_columns = ['pool_code', 'pool_name', 'pool_type']
	
	list_filters = ['pool_type', 'cost_behavior', 'is_active']
	
	formatters_columns = {
		'budgeted_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'actual_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'allocated_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'budgeted_rate': lambda x: f"${x:,.4f}" if x else "$0.00",
		'actual_rate': lambda x: f"${x:,.4f}" if x else "$0.00"
	}
	
	@expose('/rates/<pool_id>/')
	@has_access
	def calculate_rates(self, pool_id):
		"""Calculate and show cost pool rates"""
		cost_pool = CFCACostPool.query.get(pool_id)
		if not cost_pool:
			flash('Cost pool not found', 'error')
			return redirect(url_for('CACostPoolModelView.list'))
		
		rates = cost_pool.calculate_rates()
		utilization = cost_pool.get_utilization_metrics()
		
		return self.render_template(
			'cost_accounting/cost_pool_rates.html',
			cost_pool=cost_pool,
			rates=rates,
			utilization=utilization,
			title=f"Cost Pool Rates - {cost_pool.pool_name}"
		)


class CAActivityModelView(ModelView):
	"""Activity Management View for ABC"""
	
	datamodel = SQLAInterface(CFCAActivity)
	
	list_title = "Activities (ABC)"
	show_title = "Activity Details"
	add_title = "Add Activity"
	edit_title = "Edit Activity"
	
	list_columns = [
		'activity_code', 'activity_name', 'activity_type', 'value_category',
		'cost_pool.pool_name', 'primary_cost_driver.driver_name',
		'estimated_cost_per_unit', 'is_value_added', 'is_active'
	]
	
	show_columns = [
		'activity_code', 'activity_name', 'description', 'activity_type',
		'value_category', 'cost_pool', 'cost_center', 'primary_cost_driver',
		'capacity_measure', 'practical_capacity', 'current_capacity',
		'resource_requirements', 'estimated_cost_per_unit', 'setup_time_minutes',
		'processing_time_minutes', 'quality_rating', 'efficiency_rating',
		'cycle_time_minutes', 'effective_date', 'end_date', 'is_active', 'is_value_added'
	]
	
	add_columns = [
		'activity_code', 'activity_name', 'description', 'activity_type',
		'value_category', 'cost_pool', 'cost_center', 'primary_cost_driver',
		'capacity_measure', 'practical_capacity', 'current_capacity',
		'resource_requirements', 'estimated_cost_per_unit', 'setup_time_minutes',
		'processing_time_minutes', 'quality_rating', 'efficiency_rating',
		'cycle_time_minutes', 'effective_date', 'end_date', 'is_value_added'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = ['activity_code', 'activity_name', 'activity_type', 'value_category']
	
	list_filters = ['activity_type', 'value_category', 'is_value_added', 'is_active']
	
	formatters_columns = {
		'estimated_cost_per_unit': lambda x: f"${x:,.4f}" if x else "-",
		'practical_capacity': lambda x: f"{x:,.2f}" if x else "-",
		'current_capacity': lambda x: f"{x:,.2f}" if x else "-",
		'quality_rating': lambda x: f"{float(x)*100:.1f}%" if x else "-",
		'efficiency_rating': lambda x: f"{float(x)*100:.1f}%" if x else "-"
	}


class CAProductCostModelView(ModelView):
	"""Product Costing View"""
	
	datamodel = SQLAInterface(CFCAProductCost)
	
	list_title = "Product Costs"
	show_title = "Product Cost Details"
	add_title = "Add Product Cost"
	edit_title = "Edit Product Cost"
	
	list_columns = [
		'product_code', 'product_name', 'cost_period', 'costing_method',
		'production_quantity', 'total_cost', 'unit_cost', 'is_completed'
	]
	
	show_columns = [
		'product_code', 'product_name', 'product_category', 'cost_center',
		'cost_category', 'cost_period', 'fiscal_year', 'fiscal_period',
		'costing_method', 'production_quantity', 'completed_quantity',
		'spoiled_quantity', 'direct_material_cost', 'direct_labor_cost',
		'direct_expense_cost', 'allocated_overhead', 'allocated_admin',
		'allocated_selling', 'total_cost', 'unit_cost', 'standard_cost',
		'cost_variance', 'beginning_wip', 'ending_wip', 'is_completed',
		'is_posted', 'completion_date'
	]
	
	add_columns = [
		'product_code', 'product_name', 'product_category', 'cost_center',
		'cost_category', 'cost_period', 'fiscal_year', 'fiscal_period',
		'costing_method', 'production_quantity', 'direct_material_cost',
		'direct_labor_cost', 'direct_expense_cost', 'standard_cost'
	]
	
	edit_columns = add_columns + [
		'completed_quantity', 'spoiled_quantity', 'allocated_overhead',
		'allocated_admin', 'allocated_selling', 'beginning_wip', 'ending_wip',
		'is_completed', 'completion_date'
	]
	
	search_columns = ['product_code', 'product_name', 'product_category', 'cost_period']
	
	list_filters = ['costing_method', 'cost_period', 'is_completed', 'is_posted']
	
	formatters_columns = {
		'production_quantity': lambda x: f"{x:,.4f}" if x else "0",
		'completed_quantity': lambda x: f"{x:,.4f}" if x else "0",
		'total_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'unit_cost': lambda x: f"${x:,.4f}" if x else "$0.00",
		'cost_variance': lambda x: f"${x:,.2f}" if x else "-"
	}
	
	@expose('/cost_breakdown/<product_cost_id>/')
	@has_access
	def cost_breakdown(self, product_cost_id):
		"""Show detailed product cost breakdown"""
		product_cost = CFCAProductCost.query.get(product_cost_id)
		if not product_cost:
			flash('Product cost record not found', 'error')
			return redirect(url_for('CAProductCostModelView.list'))
		
		breakdown = product_cost.get_cost_breakdown()
		
		return self.render_template(
			'cost_accounting/product_cost_breakdown.html',
			product_cost=product_cost,
			breakdown=breakdown,
			title=f"Cost Breakdown - {product_cost.product_name}"
		)


class CAJobCostModelView(ModelView):
	"""Job Costing View"""
	
	datamodel = SQLAInterface(CFCAJobCost)
	
	list_title = "Job Costs"
	show_title = "Job Cost Details"
	add_title = "Add Job"
	edit_title = "Edit Job Cost"
	
	list_columns = [
		'job_number', 'job_name', 'customer_name', 'job_status',
		'budgeted_cost', 'percent_complete', 'start_date', 'is_billable'
	]
	
	show_columns = [
		'job_number', 'job_name', 'job_description', 'cost_center', 'cost_category',
		'customer_code', 'customer_name', 'project_code', 'contract_number',
		'start_date', 'planned_completion_date', 'actual_completion_date',
		'budgeted_cost', 'budgeted_hours', 'contract_value', 'actual_material_cost',
		'actual_labor_cost', 'actual_overhead_cost', 'actual_other_cost',
		'actual_labor_hours', 'actual_machine_hours', 'committed_material_cost',
		'committed_labor_cost', 'committed_other_cost', 'billed_to_date',
		'percent_complete', 'billing_method', 'job_status', 'is_billable', 'is_closed'
	]
	
	add_columns = [
		'job_number', 'job_name', 'job_description', 'cost_center', 'cost_category',
		'customer_code', 'customer_name', 'project_code', 'contract_number',
		'start_date', 'planned_completion_date', 'budgeted_cost', 'budgeted_hours',
		'contract_value', 'billing_method', 'is_billable'
	]
	
	edit_columns = add_columns + [
		'actual_material_cost', 'actual_labor_cost', 'actual_overhead_cost',
		'actual_other_cost', 'actual_labor_hours', 'actual_machine_hours',
		'committed_material_cost', 'committed_labor_cost', 'committed_other_cost',
		'billed_to_date', 'percent_complete', 'actual_completion_date',
		'job_status', 'is_closed'
	]
	
	search_columns = ['job_number', 'job_name', 'customer_name', 'project_code']
	
	list_filters = ['job_status', 'is_billable', 'is_closed', 'billing_method']
	
	formatters_columns = {
		'budgeted_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'contract_value': lambda x: f"${x:,.2f}" if x else "-",
		'actual_material_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'actual_labor_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'billed_to_date': lambda x: f"${x:,.2f}" if x else "$0.00",
		'percent_complete': lambda x: f"{x}%" if x else "0%"
	}
	
	@expose('/profitability/<job_number>/')
	@has_access
	def job_profitability(self, job_number):
		"""Show job profitability analysis"""
		service = CostAccountingService(tenant_id='default_tenant')
		
		try:
			job_summary = service.get_job_cost_summary(job_number)
			
			return self.render_template(
				'cost_accounting/job_profitability.html',
				job_summary=job_summary,
				title=f"Job Profitability - {job_number}"
			)
		
		except ValueError:
			flash(f'Job {job_number} not found', 'error')
			return redirect(url_for('CAJobCostModelView.list'))
	
	@expose('/update_costs/<job_number>/', methods=['GET', 'POST'])
	@has_access
	def update_costs(self, job_number):
		"""Update job costs"""
		if request.method == 'POST':
			try:
				service = CostAccountingService(tenant_id='default_tenant')
				
				# Parse cost updates from form
				cost_updates = {}
				for key, value in request.form.items():
					if key.startswith('cost_') and value:
						category = key.replace('cost_', '')
						cost_updates[category] = {
							'material_cost': Decimal(request.form.get(f'{key}_material', '0')),
							'labor_cost': Decimal(request.form.get(f'{key}_labor', '0')),
							'overhead_cost': Decimal(request.form.get(f'{key}_overhead', '0'))
						}
				
				if 'percent_complete' in request.form:
					cost_updates['percent_complete'] = Decimal(request.form.get('percent_complete', '0'))
				
				updated_jobs = service.update_job_costs(job_number, cost_updates)
				
				flash(f'Job costs updated for {len(updated_jobs)} categories', 'success')
				
			except Exception as e:
				flash(f'Error updating job costs: {str(e)}', 'error')
			
			return redirect(url_for('CAJobCostModelView.job_profitability', job_number=job_number))
		
		# GET request - show form
		job_costs = CFCAJobCost.query.filter_by(job_number=job_number).all()
		if not job_costs:
			flash('Job not found', 'error')
			return redirect(url_for('CAJobCostModelView.list'))
		
		return self.render_template(
			'cost_accounting/update_job_costs.html',
			job_costs=job_costs,
			job_number=job_number,
			title=f"Update Costs - {job_number}"
		)


class CAStandardCostModelView(ModelView):
	"""Standard Cost Management View"""
	
	datamodel = SQLAInterface(CFCAStandardCost)
	
	list_title = "Standard Costs"
	show_title = "Standard Cost Details"
	add_title = "Add Standard Cost"
	edit_title = "Edit Standard Cost"
	
	list_columns = [
		'cost_object_type', 'cost_object_code', 'cost_object_name',
		'cost_category.category_name', 'standard_cost_per_unit',
		'effective_date', 'standard_type', 'is_active', 'is_approved'
	]
	
	show_columns = [
		'cost_object_type', 'cost_object_code', 'cost_object_name',
		'cost_category', 'cost_center', 'standard_cost_per_unit',
		'standard_quantity_per_unit', 'standard_rate_per_quantity',
		'unit_of_measure', 'quantity_unit_of_measure', 'effective_date',
		'end_date', 'fiscal_year', 'version', 'standard_type', 'revision_reason',
		'favorable_variance_threshold', 'unfavorable_variance_threshold',
		'is_active', 'is_approved', 'approved_by', 'approved_date'
	]
	
	add_columns = [
		'cost_object_type', 'cost_object_code', 'cost_object_name',
		'cost_category', 'cost_center', 'standard_cost_per_unit',
		'standard_quantity_per_unit', 'standard_rate_per_quantity',
		'unit_of_measure', 'quantity_unit_of_measure', 'effective_date',
		'end_date', 'fiscal_year', 'version', 'standard_type', 'revision_reason',
		'favorable_variance_threshold', 'unfavorable_variance_threshold'
	]
	
	edit_columns = add_columns + ['is_active', 'is_approved', 'approved_by', 'approved_date']
	
	search_columns = ['cost_object_code', 'cost_object_name', 'cost_object_type']
	
	list_filters = ['cost_object_type', 'standard_type', 'is_active', 'is_approved', 'fiscal_year']
	
	formatters_columns = {
		'standard_cost_per_unit': lambda x: f"${x:,.4f}" if x else "$0.00",
		'standard_rate_per_quantity': lambda x: f"${x:,.4f}" if x else "$0.00",
		'favorable_variance_threshold': lambda x: f"{x}%" if x else "-",
		'unfavorable_variance_threshold': lambda x: f"{x}%" if x else "-"
	}


class CAVarianceAnalysisView(BaseView):
	"""Variance Analysis Dashboard and Reports"""
	
	route_base = '/ca/variance_analysis'
	
	@expose('/')
	@has_access
	def index(self):
		"""Variance analysis dashboard"""
		period = request.args.get('period', datetime.now().strftime('%Y-%m'))
		
		service = CostAccountingService(tenant_id='default_tenant')
		variance_reports = service.get_variance_report(period)
		
		# Summary statistics
		total_variances = len(variance_reports)
		significant_variances = len([v for v in variance_reports if v.is_significant])
		favorable_variances = len([v for v in variance_reports if v.total_variance < 0])
		
		summary_stats = {
			'total_variances': total_variances,
			'significant_variances': significant_variances,
			'favorable_variances': favorable_variances,
			'unfavorable_variances': total_variances - favorable_variances,
			'period': period
		}
		
		return self.render_template(
			'cost_accounting/variance_analysis.html',
			variance_reports=variance_reports,
			summary_stats=summary_stats,
			title="Variance Analysis",
			period=period
		)
	
	@expose('/perform/', methods=['GET', 'POST'])
	@has_access
	def perform_analysis(self):
		"""Perform new variance analysis"""
		if request.method == 'POST':
			try:
				service = CostAccountingService(tenant_id='default_tenant')
				
				analysis_data = {
					'standard_cost_id': request.form.get('standard_cost_id'),
					'actual_cost': Decimal(request.form.get('actual_cost')),
					'actual_quantity': Decimal(request.form.get('actual_quantity')),
					'analysis_period': request.form.get('analysis_period'),
					'cost_center_id': request.form.get('cost_center_id'),
					'variance_type': request.form.get('variance_type', 'Cost'),
					'primary_cause': request.form.get('primary_cause')
				}
				
				variance_analysis = service.perform_variance_analysis(analysis_data)
				
				flash('Variance analysis completed successfully', 'success')
				return redirect(url_for('CAVarianceAnalysisView.show_analysis', 
									  analysis_id=variance_analysis.variance_id))
			
			except Exception as e:
				flash(f'Error performing analysis: {str(e)}', 'error')
		
		# GET request - show form
		standard_costs = CFCAStandardCost.query.filter_by(is_active=True).all()
		cost_centers = CFCACostCenter.query.filter_by(is_active=True).all()
		
		return self.render_template(
			'cost_accounting/perform_variance_analysis.html',
			standard_costs=standard_costs,
			cost_centers=cost_centers,
			title="Perform Variance Analysis"
		)
	
	@expose('/analysis/<analysis_id>/')
	@has_access
	def show_analysis(self, analysis_id):
		"""Show detailed variance analysis"""
		variance_analysis = CFCAVarianceAnalysis.query.get(analysis_id)
		if not variance_analysis:
			flash('Variance analysis not found', 'error')
			return redirect(url_for('CAVarianceAnalysisView.index'))
		
		# Calculate detailed variance components
		variance_components = variance_analysis.calculate_variance_components()
		potential_causes = variance_analysis.get_potential_causes()
		recommended_actions = variance_analysis.recommend_actions()
		
		return self.render_template(
			'cost_accounting/variance_analysis_detail.html',
			variance_analysis=variance_analysis,
			variance_components=variance_components,
			potential_causes=potential_causes,
			recommended_actions=recommended_actions,
			title=f"Variance Analysis - {variance_analysis.cost_object_name}"
		)
	
	@expose('/report/<period>/')
	@has_access
	def period_report(self, period):
		"""Generate variance report for specific period"""
		service = CostAccountingService(tenant_id='default_tenant')
		variance_reports = service.get_variance_report(period)
		
		return self.render_template(
			'cost_accounting/variance_period_report.html',
			variance_reports=variance_reports,
			period=period,
			title=f"Variance Report - {period}"
		)


class CADashboardView(BaseView):
	"""Cost Accounting Dashboard"""
	
	route_base = '/ca/dashboard'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main cost accounting dashboard"""
		period = request.args.get('period', datetime.now().strftime('%Y-%m'))
		
		service = CostAccountingService(tenant_id='default_tenant')
		dashboard_data = service.generate_cost_dashboard_data(period)
		
		return self.render_template(
			'cost_accounting/dashboard.html',
			dashboard_data=dashboard_data,
			title="Cost Accounting Dashboard",
			period=period
		)
	
	@expose('/abc_analysis/')
	@has_access
	def abc_analysis(self):
		"""Activity-Based Costing analysis"""
		period = request.args.get('period', datetime.now().strftime('%Y-%m'))
		
		service = CostAccountingService(tenant_id='default_tenant')
		abc_data = service.get_abc_profitability_analysis(period)
		
		return self.render_template(
			'cost_accounting/abc_analysis.html',
			abc_data=abc_data,
			title="ABC Profitability Analysis",
			period=period
		)
	
	@expose('/job_summary/')
	@has_access
	def job_summary(self):
		"""Job costing summary"""
		status = request.args.get('status', 'Active')
		
		service = CostAccountingService(tenant_id='default_tenant')
		jobs = service.get_jobs_by_status(status)
		
		return self.render_template(
			'cost_accounting/job_summary.html',
			jobs=jobs,
			status=status,
			title=f"Job Summary - {status} Jobs"
		)
	
	@expose('/cost_center_performance/<center_id>/')
	@has_access
	def cost_center_performance(self, center_id):
		"""Cost center performance analysis"""
		period = request.args.get('period', datetime.now().strftime('%Y-%m'))
		
		service = CostAccountingService(tenant_id='default_tenant')
		performance_data = service.get_cost_center_performance(center_id, period)
		
		return self.render_template(
			'cost_accounting/cost_center_performance.html',
			performance_data=performance_data,
			title=f"Performance Analysis - {performance_data['cost_center']['center_name']}",
			period=period
		)