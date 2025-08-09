"""
Enterprise Asset Management UI Views

Flask-AppBuilder views with Pydantic v2 models following APG patterns.
Provides responsive, mobile-friendly interfaces for asset management,
work orders, maintenance, inventory, and analytics with real-time collaboration.

APG UI Integration:
- Flask-AppBuilder compatibility with APG navigation
- Responsive design following APG mobile framework
- Real-time updates via APG collaboration infrastructure
- Role-based access control with APG auth_rbac
- Accessibility compliance (WCAG 2.1 AA) per APG standards
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Annotated
from decimal import Decimal
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
from pydantic.functional_validators import AfterValidator
from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView, GroupByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import SelectField, TextAreaField, DecimalField, IntegerField, DateField
from wtforms.validators import DataRequired, Optional as WTFOptional, NumberRange

from .models import EAAsset, EALocation, EAWorkOrder, EAMaintenanceRecord, EAInventory, EAContract, EAPerformanceRecord
from .service import EAMAssetService, EAMWorkOrderService, EAMInventoryService, EAMAnalyticsService


# =============================================================================
# PYDANTIC V2 MODELS FOR APG INTEGRATION
# =============================================================================

def validate_asset_number(value: str) -> str:
	"""Validate asset number format"""
	if not value or len(value) < 3:
		raise ValueError("Asset number must be at least 3 characters")
	return value.upper().strip()

def validate_positive_decimal(value: Decimal | None) -> Decimal | None:
	"""Validate positive decimal values"""
	if value is not None and value < 0:
		raise ValueError("Value must be positive")
	return value

def validate_percentage(value: Decimal | None) -> Decimal | None:
	"""Validate percentage values (0-100)"""
	if value is not None and (value < 0 or value > 100):
		raise ValueError("Percentage must be between 0 and 100")
	return value


class EAAssetCreateModel(BaseModel):
	"""Pydantic model for asset creation following APG standards"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core Identity
	asset_number: Annotated[str | None, AfterValidator(validate_asset_number)] = Field(
		None, 
		description="Unique asset identifier"
	)
	asset_name: str = Field(..., min_length=1, max_length=200, description="Asset name")
	description: str | None = Field(None, max_length=1000, description="Asset description")
	
	# Classification
	asset_type: str = Field(..., description="Asset type (equipment, vehicle, facility, etc.)")
	asset_category: str = Field(..., description="Asset category (production, support, infrastructure)")
	asset_class: str | None = Field(None, description="Asset class (rotating, static, electrical)")
	criticality_level: str = Field(default="medium", description="Business criticality")
	
	# Technical Details
	manufacturer: str | None = Field(None, max_length=100, description="Manufacturer name")
	model_number: str | None = Field(None, max_length=100, description="Model number")
	serial_number: str | None = Field(None, max_length=100, description="Serial number")
	year_manufactured: int | None = Field(None, ge=1900, le=2030, description="Manufacturing year")
	
	# Location and Assignment
	location_id: str | None = Field(None, description="Location ID")
	parent_asset_id: str | None = Field(None, description="Parent asset ID")
	custodian_employee_id: str | None = Field(None, description="Custodian employee ID")
	department: str | None = Field(None, max_length=50, description="Department")
	cost_center: str | None = Field(None, max_length=20, description="Cost center")
	
	# Financial Information
	purchase_cost: Annotated[Decimal | None, AfterValidator(validate_positive_decimal)] = Field(
		None, 
		description="Purchase cost"
	)
	replacement_cost: Annotated[Decimal | None, AfterValidator(validate_positive_decimal)] = Field(
		None, 
		description="Replacement cost"
	)
	is_capitalized: bool = Field(default=True, description="Is this asset capitalized")
	
	# Operational Details
	installation_date: date | None = Field(None, description="Installation date")
	commissioning_date: date | None = Field(None, description="Commissioning date")
	expected_retirement_date: date | None = Field(None, description="Expected retirement date")
	
	# Maintenance Configuration
	maintenance_strategy: str = Field(default="predictive", description="Maintenance strategy")
	maintenance_frequency_days: int | None = Field(
		None, 
		ge=1, 
		le=3650, 
		description="Maintenance frequency in days"
	)
	
	# Digital Integration
	has_digital_twin: bool = Field(default=False, description="Has digital twin")
	iot_enabled: bool = Field(default=False, description="IoT enabled")
	
	@root_validator
	def validate_dates(cls, values):
		"""Validate date relationships"""
		install_date = values.get('installation_date')
		commission_date = values.get('commissioning_date')
		retire_date = values.get('expected_retirement_date')
		
		if install_date and commission_date and commission_date < install_date:
			raise ValueError("Commissioning date cannot be before installation date")
		
		if commission_date and retire_date and retire_date <= commission_date:
			raise ValueError("Retirement date must be after commissioning date")
		
		return values


class EAAssetUpdateModel(BaseModel):
	"""Pydantic model for asset updates"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	asset_name: str | None = Field(None, min_length=1, max_length=200)
	description: str | None = Field(None, max_length=1000)
	status: str | None = Field(None, description="Asset status")
	operational_status: str | None = Field(None, description="Operational status")
	health_score: Annotated[Decimal | None, AfterValidator(validate_percentage)] = Field(
		None, 
		description="Health score (0-100)"
	)
	condition_status: str | None = Field(None, description="Condition status")
	change_reason: str | None = Field(None, max_length=200, description="Reason for change")


class EAWorkOrderCreateModel(BaseModel):
	"""Pydantic model for work order creation"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core Information
	title: str = Field(..., min_length=1, max_length=200, description="Work order title")
	description: str = Field(..., min_length=1, description="Work order description")
	work_type: str = Field(..., description="Work type (maintenance, repair, inspection, etc.)")
	priority: str = Field(default="medium", description="Priority level")
	
	# Asset and Location
	asset_id: str | None = Field(None, description="Asset ID")
	location_id: str | None = Field(None, description="Location ID")
	
	# Classification
	work_category: str | None = Field(None, description="Work category (mechanical, electrical, etc.)")
	maintenance_type: str | None = Field(None, description="Maintenance type")
	safety_category: str | None = Field(None, description="Safety category")
	
	# Scheduling
	requested_date: datetime = Field(default_factory=datetime.utcnow, description="Requested date")
	scheduled_start: datetime | None = Field(None, description="Scheduled start")
	scheduled_end: datetime | None = Field(None, description="Scheduled end")
	
	# Estimates
	estimated_hours: float | None = Field(None, ge=0, description="Estimated hours")
	estimated_cost: Annotated[Decimal | None, AfterValidator(validate_positive_decimal)] = Field(
		None, 
		description="Estimated cost"
	)
	
	# Assignment
	assigned_to: str | None = Field(None, description="Assigned technician")
	required_crew_size: int = Field(default=1, ge=1, le=20, description="Required crew size")
	
	@root_validator
	def validate_schedule(cls, values):
		"""Validate scheduling dates"""
		start = values.get('scheduled_start')
		end = values.get('scheduled_end')
		
		if start and end and end <= start:
			raise ValueError("Scheduled end must be after scheduled start")
		
		return values


class EAInventoryCreateModel(BaseModel):
	"""Pydantic model for inventory item creation"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core Information
	part_number: str = Field(..., min_length=1, max_length=50, description="Part number")
	description: str = Field(..., min_length=1, max_length=200, description="Description")
	item_type: str = Field(..., description="Item type (spare_part, consumable, tool, material)")
	category: str = Field(..., description="Category")
	
	# Technical Details
	manufacturer: str | None = Field(None, max_length=100, description="Manufacturer")
	manufacturer_part_number: str | None = Field(None, max_length=100, description="Manufacturer part number")
	model_number: str | None = Field(None, max_length=100, description="Model number")
	
	# Inventory Levels
	current_stock: int = Field(default=0, ge=0, description="Current stock level")
	minimum_stock: int = Field(default=0, ge=0, description="Minimum stock level")
	maximum_stock: int = Field(default=0, ge=0, description="Maximum stock level")
	reorder_point: int = Field(default=0, ge=0, description="Reorder point")
	
	# Cost Information
	unit_cost: Annotated[Decimal | None, AfterValidator(validate_positive_decimal)] = Field(
		None, 
		description="Unit cost"
	)
	
	# Storage and Location
	location_id: str | None = Field(None, description="Storage location ID")
	storage_location: str | None = Field(None, max_length=100, description="Specific storage location")
	
	# Vendor Information
	primary_vendor_id: str | None = Field(None, description="Primary vendor ID")
	lead_time_days: int | None = Field(None, ge=0, le=365, description="Lead time in days")
	
	# Configuration
	criticality: str = Field(default="medium", description="Criticality level")
	auto_reorder: bool = Field(default=False, description="Enable auto-reordering")
	
	@root_validator
	def validate_stock_levels(cls, values):
		"""Validate stock level relationships"""
		current = values.get('current_stock', 0)
		minimum = values.get('minimum_stock', 0)
		maximum = values.get('maximum_stock', 0)
		reorder = values.get('reorder_point', 0)
		
		if minimum > maximum and maximum > 0:
			raise ValueError("Minimum stock cannot exceed maximum stock")
		
		if reorder > maximum and maximum > 0:
			raise ValueError("Reorder point cannot exceed maximum stock")
		
		return values


class EALocationCreateModel(BaseModel):
	"""Pydantic model for location creation"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core Information
	location_code: str = Field(..., min_length=1, max_length=50, description="Location code")
	location_name: str = Field(..., min_length=1, max_length=200, description="Location name")
	description: str | None = Field(None, max_length=1000, description="Description")
	location_type: str = Field(..., description="Location type (site, building, floor, room, area)")
	
	# Hierarchy
	parent_location_id: str | None = Field(None, description="Parent location ID")
	
	# Address Information
	address: str | None = Field(None, description="Physical address")
	city: str | None = Field(None, max_length=100, description="City")
	state_province: str | None = Field(None, max_length=50, description="State/Province")
	postal_code: str | None = Field(None, max_length=20, description="Postal code")
	country_code: str | None = Field(None, min_length=2, max_length=2, description="Country code")
	
	# Geographic Coordinates
	gps_latitude: Decimal | None = Field(None, ge=-90, le=90, description="GPS latitude")
	gps_longitude: Decimal | None = Field(None, ge=-180, le=180, description="GPS longitude")
	elevation: float | None = Field(None, description="Elevation in meters")
	
	# Physical Characteristics
	floor_area_sqm: float | None = Field(None, ge=0, description="Floor area in square meters")
	max_capacity: int | None = Field(None, ge=0, description="Maximum capacity")
	
	# Operations
	cost_center: str | None = Field(None, max_length=20, description="Cost center")
	facility_manager_id: str | None = Field(None, description="Facility manager ID")


# =============================================================================
# FLASK-APPBUILDER VIEW CLASSES
# =============================================================================

class EAAssetModelView(ModelView):
	"""Asset Management View with APG integration"""
	
	datamodel = SQLAInterface(EAAsset)
	
	# List view configuration
	list_columns = [
		'asset_number', 'asset_name', 'asset_type', 'status', 
		'criticality_level', 'health_score', 'location.location_name',
		'last_maintenance_date', 'next_maintenance_due'
	]
	
	search_columns = [
		'asset_number', 'asset_name', 'description', 'manufacturer', 
		'model_number', 'serial_number', 'asset_type', 'asset_category'
	]
	
	# Show view configuration
	show_columns = [
		'asset_number', 'asset_name', 'description', 'asset_type', 'asset_category',
		'criticality_level', 'status', 'operational_status', 'health_score',
		'manufacturer', 'model_number', 'serial_number', 'year_manufactured',
		'location.location_name', 'parent_asset.asset_name', 'custodian_employee_id',
		'purchase_cost', 'replacement_cost', 'current_book_value',
		'installation_date', 'commissioning_date', 'warranty_end_date',
		'maintenance_strategy', 'last_maintenance_date', 'next_maintenance_due',
		'has_digital_twin', 'iot_enabled', 'created_on', 'changed_on'
	]
	
	# Edit view configuration
	edit_columns = [
		'asset_name', 'description', 'asset_type', 'asset_category',
		'criticality_level', 'status', 'manufacturer', 'model_number',
		'serial_number', 'location_id', 'parent_asset_id', 'custodian_employee_id',
		'purchase_cost', 'replacement_cost', 'maintenance_strategy',
		'maintenance_frequency_days', 'has_digital_twin', 'iot_enabled'
	]
	
	# Add view configuration
	add_columns = [
		'asset_number', 'asset_name', 'description', 'asset_type', 'asset_category',
		'criticality_level', 'manufacturer', 'model_number', 'serial_number',
		'year_manufactured', 'location_id', 'parent_asset_id', 'custodian_employee_id',
		'purchase_cost', 'replacement_cost', 'installation_date', 'commissioning_date',
		'maintenance_strategy', 'maintenance_frequency_days', 'has_digital_twin', 'iot_enabled'
	]
	
	# Formatters for better display
	formatters_columns = {
		'health_score': lambda x: f"{x:.1f}%" if x else "N/A",
		'purchase_cost': lambda x: f"${x:,.2f}" if x else "N/A",
		'replacement_cost': lambda x: f"${x:,.2f}" if x else "N/A",
		'current_book_value': lambda x: f"${x:,.2f}" if x else "N/A"
	}
	
	# Filters
	base_filters = [['tenant_id', lambda: session.get('tenant_id')]]
	
	# Order
	base_order = ('asset_number', 'asc')
	
	# Page size
	page_size = 25
	
	# Enable CSV export
	export_columns = list_columns + ['description', 'manufacturer', 'model_number', 'serial_number']
	
	@expose('/hierarchy/<asset_id>')
	@has_access
	def hierarchy(self, asset_id):
		"""Show asset hierarchy view"""
		# This would render asset hierarchy with parent/children
		return self.render_template('eam/asset_hierarchy.html', asset_id=asset_id)
	
	@expose('/health_dashboard')
	@has_access
	def health_dashboard(self):
		"""Asset health monitoring dashboard"""
		# This would render real-time health dashboard
		return self.render_template('eam/asset_health_dashboard.html')


class EAWorkOrderModelView(ModelView):
	"""Work Order Management View with APG collaboration"""
	
	datamodel = SQLAInterface(EAWorkOrder)
	
	# List view configuration
	list_columns = [
		'work_order_number', 'title', 'work_type', 'priority', 'status',
		'asset.asset_number', 'assigned_to', 'scheduled_start', 'scheduled_end'
	]
	
	search_columns = [
		'work_order_number', 'title', 'description', 'work_type',
		'asset.asset_number', 'asset.asset_name'
	]
	
	# Show view configuration
	show_columns = [
		'work_order_number', 'title', 'description', 'work_type', 'priority',
		'status', 'asset.asset_number', 'asset.asset_name', 'location.location_name',
		'assigned_to', 'assigned_team', 'estimated_hours', 'actual_hours',
		'estimated_cost', 'actual_cost', 'requested_date', 'scheduled_start',
		'scheduled_end', 'actual_start', 'actual_end', 'completion_percentage',
		'work_performed', 'completion_notes', 'created_on', 'changed_on'
	]
	
	# Edit view configuration
	edit_columns = [
		'title', 'description', 'work_type', 'priority', 'status',
		'asset_id', 'location_id', 'assigned_to', 'estimated_hours',
		'estimated_cost', 'scheduled_start', 'scheduled_end'
	]
	
	# Add view configuration
	add_columns = [
		'title', 'description', 'work_type', 'priority', 'asset_id',
		'location_id', 'estimated_hours', 'estimated_cost', 'scheduled_start',
		'scheduled_end', 'assigned_to'
	]
	
	# Formatters
	formatters_columns = {
		'estimated_cost': lambda x: f"${x:,.2f}" if x else "N/A",
		'actual_cost': lambda x: f"${x:,.2f}" if x else "N/A",
		'completion_percentage': lambda x: f"{x}%" if x else "0%"
	}
	
	# Filters
	base_filters = [['tenant_id', lambda: session.get('tenant_id')]]
	
	# Order
	base_order = ('scheduled_start', 'desc')
	
	# Enable real-time updates
	@expose('/kanban')
	@has_access
	def kanban_board(self):
		"""Kanban board view for work orders"""
		return self.render_template('eam/work_order_kanban.html')
	
	@expose('/calendar')
	@has_access
	def calendar_view(self):
		"""Calendar view for scheduling"""
		return self.render_template('eam/work_order_calendar.html')
	
	@expose('/mobile')
	@has_access
	def mobile_view(self):
		"""Mobile-optimized work order view"""
		return self.render_template('eam/work_order_mobile.html')


class EAInventoryModelView(ModelView):
	"""Inventory Management View with procurement integration"""
	
	datamodel = SQLAInterface(EAInventory)
	
	# List view configuration
	list_columns = [
		'part_number', 'description', 'item_type', 'current_stock',
		'minimum_stock', 'reorder_point', 'unit_cost', 'criticality',
		'primary_vendor_id', 'auto_reorder'
	]
	
	search_columns = [
		'part_number', 'description', 'manufacturer', 'manufacturer_part_number',
		'model_number', 'category'
	]
	
	# Show view configuration
	show_columns = [
		'part_number', 'description', 'item_type', 'category', 'manufacturer',
		'manufacturer_part_number', 'model_number', 'current_stock', 'minimum_stock',
		'maximum_stock', 'reorder_point', 'unit_cost', 'average_cost', 'total_value',
		'location.location_name', 'storage_location', 'primary_vendor_id',
		'lead_time_days', 'criticality', 'auto_reorder', 'annual_usage'
	]
	
	# Edit view configuration
	edit_columns = [
		'description', 'item_type', 'category', 'manufacturer', 'model_number',
		'minimum_stock', 'maximum_stock', 'reorder_point', 'unit_cost',
		'location_id', 'storage_location', 'primary_vendor_id', 'lead_time_days',
		'criticality', 'auto_reorder'
	]
	
	# Formatters
	formatters_columns = {
		'unit_cost': lambda x: f"${x:.2f}" if x else "N/A",
		'average_cost': lambda x: f"${x:.2f}" if x else "N/A",
		'total_value': lambda x: f"${x:,.2f}" if x else "N/A",
		'current_stock': lambda x: f"{x:,}",
		'auto_reorder': lambda x: "✓" if x else "✗"
	}
	
	# Stock level indicators
	def stock_level_formatter(view, context, model, name):
		"""Format stock levels with color indicators"""
		stock = model.current_stock
		reorder = model.reorder_point
		minimum = model.minimum_stock
		
		if stock <= 0:
			return f'<span class="label label-danger">{stock}</span>'
		elif stock <= reorder:
			return f'<span class="label label-warning">{stock}</span>'
		elif stock <= minimum:
			return f'<span class="label label-info">{stock}</span>'
		else:
			return f'<span class="label label-success">{stock}</span>'
	
	formatters_columns['current_stock'] = stock_level_formatter
	
	@expose('/reorder_report')
	@has_access
	def reorder_report(self):
		"""Reorder recommendations report"""
		return self.render_template('eam/inventory_reorder_report.html')
	
	@expose('/stock_movements')
	@has_access
	def stock_movements(self):
		"""Stock movement history"""
		return self.render_template('eam/inventory_movements.html')


class EALocationModelView(ModelView):
	"""Location Management View with hierarchy"""
	
	datamodel = SQLAInterface(EALocation)
	
	# List view configuration
	list_columns = [
		'location_code', 'location_name', 'location_type',
		'parent_location.location_name', 'city', 'is_active'
	]
	
	search_columns = [
		'location_code', 'location_name', 'description', 'city', 'address'
	]
	
	# Show view configuration
	show_columns = [
		'location_code', 'location_name', 'description', 'location_type',
		'parent_location.location_name', 'hierarchy_level', 'address', 'city',
		'state_province', 'postal_code', 'country_code', 'gps_latitude',
		'gps_longitude', 'floor_area_sqm', 'max_capacity', 'current_utilization',
		'cost_center', 'facility_manager_id', 'is_active'
	]
	
	@expose('/hierarchy_tree')
	@has_access
	def hierarchy_tree(self):
		"""Location hierarchy tree view"""
		return self.render_template('eam/location_hierarchy.html')
	
	@expose('/map_view')
	@has_access
	def map_view(self):
		"""Geographic map view of locations"""
		return self.render_template('eam/location_map.html')


class EAMaintenanceModelView(ModelView):
	"""Maintenance Record View"""
	
	datamodel = SQLAInterface(EAMaintenanceRecord)
	
	# List view configuration
	list_columns = [
		'maintenance_number', 'asset.asset_number', 'maintenance_type',
		'started_at', 'completed_at', 'outcome', 'total_cost', 'technician_id'
	]
	
	search_columns = [
		'maintenance_number', 'asset.asset_number', 'description', 'work_performed'
	]
	
	# Show view configuration
	show_columns = [
		'maintenance_number', 'asset.asset_number', 'maintenance_type',
		'maintenance_category', 'description', 'work_performed', 'started_at',
		'completed_at', 'duration_hours', 'downtime_hours', 'labor_cost',
		'parts_cost', 'total_cost', 'outcome', 'quality_rating',
		'health_score_before', 'health_score_after', 'recommendations'
	]
	
	# Enable analytics
	@expose('/analytics')
	@has_access
	def maintenance_analytics(self):
		"""Maintenance analytics dashboard"""
		return self.render_template('eam/maintenance_analytics.html')


class EAPerformanceModelView(ModelView):
	"""Performance Analytics View"""
	
	datamodel = SQLAInterface(EAPerformanceRecord)
	
	# List view configuration
	list_columns = [
		'asset.asset_number', 'measurement_date', 'availability_percentage',
		'oee_overall', 'health_score', 'trend_direction'
	]
	
	# Show view configuration
	show_columns = [
		'asset.asset_number', 'measurement_date', 'measurement_period',
		'availability_percentage', 'oee_availability', 'oee_performance',
		'oee_quality', 'oee_overall', 'health_score', 'energy_consumption',
		'maintenance_cost', 'trend_direction'
	]
	
	# Charts for performance visualization
	@expose('/performance_dashboard')
	@has_access
	def performance_dashboard(self):
		"""Performance analytics dashboard"""
		return self.render_template('eam/performance_dashboard.html')


# =============================================================================
# CUSTOM DASHBOARD VIEWS
# =============================================================================

class EAMDashboardView(BaseView):
	"""Main EAM Dashboard with real-time updates"""
	
	default_view = 'dashboard'
	
	@expose('/')
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Main EAM dashboard with KPIs and alerts"""
		# This would fetch real-time dashboard data
		dashboard_data = {
			'total_assets': 0,
			'critical_alerts': 0,
			'maintenance_due': 0,
			'work_orders_open': 0,
			'average_health_score': 0,
			'fleet_availability': 0
		}
		
		return self.render_template(
			'eam/dashboard.html',
			dashboard_data=dashboard_data
		)
	
	@expose('/mobile_dashboard')
	@has_access
	def mobile_dashboard(self):
		"""Mobile-optimized dashboard"""
		return self.render_template('eam/mobile_dashboard.html')
	
	@expose('/executive_summary')
	@has_access
	def executive_summary(self):
		"""Executive summary dashboard"""
		return self.render_template('eam/executive_summary.html')


class EAMAnalyticsView(BaseView):
	"""Advanced Analytics and Reporting"""
	
	default_view = 'analytics'
	
	@expose('/')
	@expose('/analytics')
	@has_access
	def analytics(self):
		"""Analytics homepage"""
		return self.render_template('eam/analytics_home.html')
	
	@expose('/asset_performance')
	@has_access
	def asset_performance(self):
		"""Asset performance analytics"""
		return self.render_template('eam/analytics_asset_performance.html')
	
	@expose('/maintenance_effectiveness')
	@has_access
	def maintenance_effectiveness(self):
		"""Maintenance effectiveness analysis"""
		return self.render_template('eam/analytics_maintenance.html')
	
	@expose('/cost_analysis')
	@has_access
	def cost_analysis(self):
		"""Cost analysis and optimization"""
		return self.render_template('eam/analytics_cost.html')
	
	@expose('/predictive_insights')
	@has_access
	def predictive_insights(self):
		"""AI-powered predictive insights"""
		return self.render_template('eam/analytics_predictive.html')


# =============================================================================
# CHART VIEWS FOR VISUALIZATION
# =============================================================================

class AssetHealthChart(GroupByChartView):
	"""Asset health distribution chart"""
	datamodel = SQLAInterface(EAAsset)
	chart_title = 'Asset Health Distribution'
	chart_type = 'PieChart'
	group_by_columns = ['condition_status']


class WorkOrderStatusChart(GroupByChartView):
	"""Work order status distribution"""
	datamodel = SQLAInterface(EAWorkOrder)
	chart_title = 'Work Order Status'
	chart_type = 'ColumnChart'
	group_by_columns = ['status']


class MaintenanceCostTrendChart(DirectByChartView):
	"""Maintenance cost trends"""
	datamodel = SQLAInterface(EAMaintenanceRecord)
	chart_title = 'Maintenance Cost Trends'
	chart_type = 'LineChart'
	direct_by_column = 'started_at'
	group_by_columns = ['total_cost']


# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

# Export all view classes for blueprint registration
__all__ = [
	# Model Views
	'EAAssetModelView',
	'EAWorkOrderModelView', 
	'EAInventoryModelView',
	'EALocationModelView',
	'EAMaintenanceModelView',
	'EAPerformanceModelView',
	
	# Dashboard Views
	'EAMDashboardView',
	'EAMAnalyticsView',
	
	# Chart Views
	'AssetHealthChart',
	'WorkOrderStatusChart',
	'MaintenanceCostTrendChart',
	
	# Pydantic Models
	'EAAssetCreateModel',
	'EAAssetUpdateModel',
	'EAWorkOrderCreateModel',
	'EAInventoryCreateModel',
	'EALocationCreateModel'
]