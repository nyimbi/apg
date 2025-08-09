"""
Production Planning Views

Flask-AppBuilder views for production planning functionality including
master production schedules, production orders, demand forecasts, and capacity planning.
"""

from datetime import datetime, date
from decimal import Decimal
from flask import flash, redirect, url_for, request
from flask_appbuilder import ModelView, BaseView, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from wtforms import validators
from wtforms.fields import SelectField, DecimalField, DateField, TextAreaField

from .models import (
	MFPMasterProductionSchedule, MFPProductionOrder, MFPDemandForecast, MFPResourceCapacity,
	ProductionOrderStatus, SchedulingPriority, PlanningHorizon
)
from .service import ProductionPlanningService

class MasterProductionScheduleView(ModelView):
	"""Master Production Schedule management view"""
	
	datamodel = SQLAInterface(MFPMasterProductionSchedule)
	
	list_title = "Master Production Schedules"
	show_title = "Master Production Schedule Details"
	add_title = "Create Master Production Schedule"
	edit_title = "Edit Master Production Schedule"
	
	list_columns = [
		'schedule_name', 'planning_period', 'planning_horizon',
		'product_id', 'facility_id', 'planned_quantity',
		'planned_start_date', 'planned_end_date', 'status', 'priority'
	]
	
	show_columns = [
		'schedule_name', 'planning_period', 'planning_horizon',
		'product_id', 'facility_id', 'production_line_id',
		'planned_quantity', 'planned_start_date', 'planned_end_date',
		'forecast_demand', 'available_capacity', 'capacity_utilization_pct',
		'status', 'priority', 'safety_stock_days', 'lead_time_days',
		'batch_size_min', 'batch_size_max', 'created_by', 'created_at'
	]
	
	add_columns = [
		'schedule_name', 'planning_period', 'planning_horizon',
		'product_id', 'facility_id', 'production_line_id',
		'planned_quantity', 'planned_start_date', 'planned_end_date',
		'forecast_demand', 'available_capacity', 'priority',
		'safety_stock_days', 'lead_time_days', 'batch_size_min', 'batch_size_max'
	]
	
	edit_columns = add_columns
	
	search_columns = [
		'schedule_name', 'planning_period', 'product_id', 'facility_id', 'status'
	]
	
	order_columns = ['planned_start_date', 'planned_end_date', 'priority']
	base_order = ('planned_start_date', 'asc')
	
	formatters_columns = {
		'planned_quantity': lambda x: f"{x:,.2f}" if x else "",
		'forecast_demand': lambda x: f"{x:,.2f}" if x else "",
		'available_capacity': lambda x: f"{x:,.2f}" if x else "",
		'capacity_utilization_pct': lambda x: f"{x:.1f}%" if x else "",
		'planned_start_date': lambda x: x.strftime('%Y-%m-%d') if x else "",
		'planned_end_date': lambda x: x.strftime('%Y-%m-%d') if x else ""
	}
	
	validators_columns = {
		'schedule_name': [validators.DataRequired(), validators.Length(max=200)],
		'planned_quantity': [validators.DataRequired(), validators.NumberRange(min=0.01)],
		'planned_start_date': [validators.DataRequired()],
		'planned_end_date': [validators.DataRequired()]
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new schedule"""
		item.tenant_id = self.get_current_tenant_id()
		item.created_by = self.get_current_user_id()
		
		# Calculate capacity utilization if both values provided
		if item.available_capacity and item.available_capacity > 0:
			item.capacity_utilization_pct = (item.planned_quantity / item.available_capacity) * 100
		
		# Validate date logic
		if item.planned_end_date <= item.planned_start_date:
			flash("Planned end date must be after planned start date", "error")
			return False

class ProductionOrderView(ModelView):
	"""Production Order management view"""
	
	datamodel = SQLAInterface(MFPProductionOrder)
	
	list_title = "Production Orders"
	show_title = "Production Order Details"
	add_title = "Create Production Order"
	edit_title = "Edit Production Order"
	
	list_columns = [
		'order_number', 'product_name', 'facility_id', 'ordered_quantity',
		'scheduled_start_date', 'scheduled_end_date', 'status', 'priority'
	]
	
	show_columns = [
		'order_number', 'order_type', 'master_schedule_id',
		'product_id', 'product_sku', 'product_name',
		'facility_id', 'production_line_id', 'work_center_id',
		'ordered_quantity', 'produced_quantity', 'scrap_quantity', 'rework_quantity',
		'scheduled_start_date', 'scheduled_end_date', 'actual_start_date', 'actual_end_date',
		'status', 'priority', 'estimated_labor_hours', 'actual_labor_hours',
		'estimated_machine_hours', 'actual_machine_hours',
		'bom_id', 'routing_id', 'production_notes', 'special_instructions',
		'quality_requirements', 'created_by', 'created_at'
	]
	
	add_columns = [
		'order_number', 'order_type', 'master_schedule_id',
		'product_id', 'product_sku', 'product_name',
		'facility_id', 'production_line_id', 'work_center_id',
		'ordered_quantity', 'scheduled_start_date', 'scheduled_end_date',
		'priority', 'estimated_labor_hours', 'estimated_machine_hours',
		'bom_id', 'routing_id', 'production_notes', 'special_instructions',
		'quality_requirements'
	]
	
	edit_columns = add_columns + [
		'produced_quantity', 'scrap_quantity', 'rework_quantity', 'status',
		'actual_start_date', 'actual_end_date', 'actual_labor_hours', 'actual_machine_hours'
	]
	
	search_columns = [
		'order_number', 'product_name', 'product_sku', 'facility_id', 'status'
	]
	
	order_columns = ['scheduled_start_date', 'scheduled_end_date', 'priority']
	base_order = ('scheduled_start_date', 'asc')
	
	formatters_columns = {
		'ordered_quantity': lambda x: f"{x:,.2f}" if x else "",
		'produced_quantity': lambda x: f"{x:,.2f}" if x else "",
		'scrap_quantity': lambda x: f"{x:,.2f}" if x else "",
		'rework_quantity': lambda x: f"{x:,.2f}" if x else "",
		'estimated_labor_hours': lambda x: f"{x:,.1f}h" if x else "",
		'actual_labor_hours': lambda x: f"{x:,.1f}h" if x else "",
		'estimated_machine_hours': lambda x: f"{x:,.1f}h" if x else "",
		'actual_machine_hours': lambda x: f"{x:,.1f}h" if x else "",
		'scheduled_start_date': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else "",
		'scheduled_end_date': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else "",
		'actual_start_date': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else "",
		'actual_end_date': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else ""
	}
	
	validators_columns = {
		'order_number': [validators.DataRequired(), validators.Length(max=50)],
		'product_name': [validators.DataRequired(), validators.Length(max=200)],
		'ordered_quantity': [validators.DataRequired(), validators.NumberRange(min=0.01)],
		'scheduled_start_date': [validators.DataRequired()],
		'scheduled_end_date': [validators.DataRequired()]
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new production order"""
		item.tenant_id = self.get_current_tenant_id()
		item.created_by = self.get_current_user_id()
		item.status = ProductionOrderStatus.PLANNED.value
		
		# Validate date logic
		if item.scheduled_end_date <= item.scheduled_start_date:
			flash("Scheduled end date must be after scheduled start date", "error")
			return False

class DemandForecastView(ModelView):
	"""Demand Forecast management view"""
	
	datamodel = SQLAInterface(MFPDemandForecast)
	
	list_title = "Demand Forecasts"
	show_title = "Demand Forecast Details"
	add_title = "Create Demand Forecast"
	edit_title = "Edit Demand Forecast"
	
	list_columns = [
		'forecast_name', 'forecast_period', 'forecast_type',
		'product_id', 'forecast_quantity', 'actual_quantity',
		'forecast_accuracy_pct', 'status'
	]
	
	show_columns = [
		'forecast_name', 'forecast_period', 'forecast_type',
		'product_id', 'facility_id', 'customer_id',
		'forecast_quantity', 'forecast_value', 'actual_quantity', 'actual_value',
		'forecast_error', 'forecast_accuracy_pct',
		'period_start_date', 'period_end_date',
		'forecast_method', 'confidence_level', 'seasonality_factor', 'trend_factor',
		'status', 'is_approved', 'created_by', 'created_at'
	]
	
	add_columns = [
		'forecast_name', 'forecast_period', 'forecast_type',
		'product_id', 'facility_id', 'customer_id',
		'forecast_quantity', 'forecast_value',
		'period_start_date', 'period_end_date',
		'forecast_method', 'confidence_level', 'seasonality_factor', 'trend_factor'
	]
	
	edit_columns = add_columns + [
		'actual_quantity', 'actual_value', 'forecast_error', 'forecast_accuracy_pct',
		'status', 'is_approved'
	]
	
	search_columns = [
		'forecast_name', 'forecast_period', 'forecast_type', 'product_id', 'status'
	]
	
	order_columns = ['period_start_date', 'forecast_accuracy_pct']
	base_order = ('period_start_date', 'desc')
	
	formatters_columns = {
		'forecast_quantity': lambda x: f"{x:,.2f}" if x else "",
		'actual_quantity': lambda x: f"{x:,.2f}" if x else "",
		'forecast_value': lambda x: f"${x:,.2f}" if x else "",
		'actual_value': lambda x: f"${x:,.2f}" if x else "",
		'forecast_error': lambda x: f"{x:,.2f}" if x else "",
		'forecast_accuracy_pct': lambda x: f"{x:.1f}%" if x else "",
		'confidence_level': lambda x: f"{x:.1f}%" if x else "",
		'period_start_date': lambda x: x.strftime('%Y-%m-%d') if x else "",
		'period_end_date': lambda x: x.strftime('%Y-%m-%d') if x else ""
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new forecast"""
		item.tenant_id = self.get_current_tenant_id()
		item.created_by = self.get_current_user_id()
		item.status = "active"

class ResourceCapacityView(ModelView):
	"""Resource Capacity planning view"""
	
	datamodel = SQLAInterface(MFPResourceCapacity)
	
	list_title = "Resource Capacity Planning"
	show_title = "Resource Capacity Details"
	add_title = "Create Resource Capacity"
	edit_title = "Edit Resource Capacity"
	
	list_columns = [
		'resource_name', 'resource_type', 'facility_id', 'planning_period',
		'available_capacity', 'planned_capacity', 'capacity_utilization_pct'
	]
	
	show_columns = [
		'resource_type', 'resource_id', 'resource_name',
		'facility_id', 'work_center_id', 'planning_period', 'capacity_unit',
		'available_capacity', 'planned_capacity', 'actual_capacity',
		'capacity_utilization_pct', 'efficiency_pct', 'availability_pct',
		'period_start_date', 'period_end_date',
		'shifts_per_day', 'hours_per_shift', 'working_days_per_week',
		'max_capacity', 'min_capacity', 'setup_time_hours', 'maintenance_time_hours',
		'status', 'created_by', 'created_at'
	]
	
	add_columns = [
		'resource_type', 'resource_id', 'resource_name',
		'facility_id', 'work_center_id', 'planning_period', 'capacity_unit',
		'available_capacity', 'period_start_date', 'period_end_date',
		'shifts_per_day', 'hours_per_shift', 'working_days_per_week',
		'max_capacity', 'min_capacity', 'setup_time_hours', 'maintenance_time_hours'
	]
	
	edit_columns = add_columns + [
		'planned_capacity', 'actual_capacity', 'capacity_utilization_pct',
		'efficiency_pct', 'availability_pct', 'status'
	]
	
	search_columns = [
		'resource_name', 'resource_type', 'facility_id', 'planning_period'
	]
	
	order_columns = ['capacity_utilization_pct', 'available_capacity']
	base_order = ('capacity_utilization_pct', 'desc')
	
	formatters_columns = {
		'available_capacity': lambda x: f"{x:,.2f}" if x else "",
		'planned_capacity': lambda x: f"{x:,.2f}" if x else "",
		'actual_capacity': lambda x: f"{x:,.2f}" if x else "",
		'capacity_utilization_pct': lambda x: f"{x:.1f}%" if x else "",
		'efficiency_pct': lambda x: f"{x:.1f}%" if x else "",
		'availability_pct': lambda x: f"{x:.1f}%" if x else "",
		'max_capacity': lambda x: f"{x:,.2f}" if x else "",
		'min_capacity': lambda x: f"{x:,.2f}" if x else "",
		'setup_time_hours': lambda x: f"{x:.1f}h" if x else "",
		'maintenance_time_hours': lambda x: f"{x:.1f}h" if x else "",
		'hours_per_shift': lambda x: f"{x:.1f}h" if x else "",
		'period_start_date': lambda x: x.strftime('%Y-%m-%d') if x else "",
		'period_end_date': lambda x: x.strftime('%Y-%m-%d') if x else ""
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new capacity record"""
		item.tenant_id = self.get_current_tenant_id()
		item.created_by = self.get_current_user_id()
		item.status = "active"

class ProductionPlanningDashboardView(BaseView):
	"""Production Planning Dashboard"""
	
	default_view = 'dashboard'
	
	@expose('/dashboard/')
	def dashboard(self):
		"""Production planning dashboard with key metrics"""
		
		# Get current tenant and basic metrics
		tenant_id = self.get_current_tenant_id()
		
		# Mock data for dashboard - in real implementation, get from service
		dashboard_data = {
			'total_active_orders': 156,
			'orders_in_progress': 42,
			'completed_this_week': 28,
			'overdue_orders': 8,
			'avg_capacity_utilization': 78.5,
			'top_facilities': [
				{'name': 'Plant A', 'utilization': 85.2, 'orders': 67},
				{'name': 'Plant B', 'utilization': 72.8, 'orders': 45},
				{'name': 'Plant C', 'utilization': 81.1, 'orders': 44}
			],
			'capacity_alerts': [
				'Machine Center MC-01 exceeds 95% capacity',
				'Labor shortage in Assembly Line 3',
				'Material shortage affecting Order PO-2024-0156'
			]
		}
		
		return self.render_template(
			'production_planning/dashboard.html',
			dashboard_data=dashboard_data,
			title="Production Planning Dashboard"
		)
	
	@expose('/schedule-optimization/')
	def schedule_optimization(self):
		"""Schedule optimization interface"""
		return self.render_template(
			'production_planning/optimization.html',
			title="Production Schedule Optimization"
		)
	
	def get_current_tenant_id(self) -> str:
		"""Get current tenant ID from session/context"""
		return "default-tenant"  # Replace with actual tenant resolution
	
	def get_current_user_id(self) -> str:
		"""Get current user ID from session/context"""
		return "current-user"  # Replace with actual user resolution

class ProductionOrderStatusChartView(DirectByChartView):
	"""Chart view for production order status distribution"""
	
	datamodel = SQLAInterface(MFPProductionOrder)
	chart_title = "Production Orders by Status"
	
	definitions = [
		{
			'group': 'status',
			'series': ['id']
		}
	]

class CapacityUtilizationChartView(DirectByChartView):
	"""Chart view for capacity utilization by facility"""
	
	datamodel = SQLAInterface(MFPResourceCapacity)
	chart_title = "Capacity Utilization by Facility"
	
	definitions = [
		{
			'group': 'facility_id',
			'series': ['capacity_utilization_pct']
		}
	]

__all__ = [
	"MasterProductionScheduleView", "ProductionOrderView", "DemandForecastView",
	"ResourceCapacityView", "ProductionPlanningDashboardView", 
	"ProductionOrderStatusChartView", "CapacityUtilizationChartView"
]