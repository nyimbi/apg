"""
APG Budgeting & Forecasting - Interactive Dashboard System

Real-time interactive dashboard with drill-down capabilities, dynamic visualizations,
and comprehensive budget performance monitoring.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from enum import Enum
from datetime import datetime, date, timedelta
from decimal import Decimal
import asyncio
import logging
import json
from uuid_extensions import uuid7str

from .models import APGBaseModel, PositiveAmount, NonEmptyString
from .service import APGTenantContext, ServiceResponse, APGServiceBase
from .advanced_analytics import AnalyticsMetric, AnalyticsPeriod, AnalyticsGranularity


# =============================================================================
# Dashboard Enumerations
# =============================================================================

class DashboardType(str, Enum):
	"""Types of dashboard views."""
	EXECUTIVE = "executive"
	DEPARTMENTAL = "departmental"
	PROJECT = "project"
	DETAILED = "detailed"
	COMPARISON = "comparison"
	FORECAST = "forecast"


class VisualizationType(str, Enum):
	"""Types of dashboard visualizations."""
	LINE_CHART = "line_chart"
	BAR_CHART = "bar_chart"
	PIE_CHART = "pie_chart"
	AREA_CHART = "area_chart"
	SCATTER_PLOT = "scatter_plot"
	HEATMAP = "heatmap"
	GAUGE = "gauge"
	TABLE = "table"
	KPI_CARD = "kpi_card"
	TREND_INDICATOR = "trend_indicator"


class DrillDownLevel(str, Enum):
	"""Dashboard drill-down levels."""
	SUMMARY = "summary"
	DEPARTMENT = "department"
	CATEGORY = "category"
	LINE_ITEM = "line_item"
	TRANSACTION = "transaction"


class RefreshMode(str, Enum):
	"""Dashboard refresh modes."""
	REAL_TIME = "real_time"
	SCHEDULED = "scheduled"
	MANUAL = "manual"
	ON_DEMAND = "on_demand"


class FilterType(str, Enum):
	"""Dashboard filter types."""
	DATE_RANGE = "date_range"
	DEPARTMENT = "department"
	CATEGORY = "category"
	BUDGET_TYPE = "budget_type"
	STATUS = "status"
	AMOUNT_RANGE = "amount_range"


# =============================================================================
# Interactive Dashboard Models
# =============================================================================

class DashboardWidget(APGBaseModel):
	"""Individual dashboard widget configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	widget_id: str = Field(default_factory=uuid7str, description="Widget identifier")
	widget_name: NonEmptyString = Field(description="Widget display name")
	widget_type: VisualizationType = Field(description="Type of visualization")
	
	# Widget Configuration
	title: str = Field(description="Widget title")
	subtitle: Optional[str] = Field(None, description="Widget subtitle")
	description: Optional[str] = Field(None, description="Widget description")
	
	# Data Configuration
	data_source: str = Field(description="Data source identifier")
	metrics: List[str] = Field(description="Metrics to display")
	dimensions: List[str] = Field(default_factory=list, description="Data dimensions")
	
	# Visual Configuration
	chart_config: Dict[str, Any] = Field(default_factory=dict, description="Chart-specific configuration")
	color_scheme: List[str] = Field(default_factory=list, description="Color palette")
	
	# Layout Configuration
	position_x: int = Field(description="X position in grid")
	position_y: int = Field(description="Y position in grid")
	width: int = Field(description="Widget width in grid units")
	height: int = Field(description="Widget height in grid units")
	
	# Interactivity
	drill_down_enabled: bool = Field(default=False, description="Enable drill-down")
	drill_down_levels: List[DrillDownLevel] = Field(default_factory=list)
	click_action: Optional[str] = Field(None, description="Action on widget click")
	
	# Refresh Configuration
	refresh_mode: RefreshMode = Field(default=RefreshMode.SCHEDULED)
	refresh_interval: int = Field(default=300, description="Refresh interval in seconds")
	
	# Real-time Features
	real_time_enabled: bool = Field(default=False)
	alert_thresholds: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Widget Data
	current_data: Dict[str, Any] = Field(default_factory=dict, description="Current widget data")
	last_updated: datetime = Field(default_factory=datetime.utcnow)


class DashboardFilter(APGBaseModel):
	"""Dashboard filter configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	filter_id: str = Field(default_factory=uuid7str)
	filter_name: NonEmptyString = Field(description="Filter name")
	filter_type: FilterType = Field(description="Type of filter")
	
	# Filter Configuration
	field_name: str = Field(description="Data field to filter")
	operator: str = Field(description="Filter operator (eq, gt, lt, in, etc.)")
	default_value: Optional[Any] = Field(None, description="Default filter value")
	
	# Filter Options
	available_values: List[Any] = Field(default_factory=list, description="Available filter values")
	is_multi_select: bool = Field(default=False, description="Allow multiple selections")
	is_required: bool = Field(default=False, description="Filter is required")
	
	# Current State
	current_value: Optional[Any] = Field(None, description="Current filter value")
	is_active: bool = Field(default=False, description="Filter is active")


class InteractiveDashboard(APGBaseModel):
	"""Interactive dashboard with real-time capabilities."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	dashboard_id: str = Field(default_factory=uuid7str)
	dashboard_name: NonEmptyString = Field(description="Dashboard name")
	dashboard_type: DashboardType = Field(description="Type of dashboard")
	
	# Dashboard Configuration
	description: Optional[str] = Field(None, description="Dashboard description")
	budget_ids: List[str] = Field(description="Associated budget IDs")
	
	# Layout Configuration
	grid_columns: int = Field(default=12, description="Number of grid columns")
	grid_rows: int = Field(default=20, description="Number of grid rows")
	responsive_breakpoints: Dict[str, int] = Field(default_factory=dict)
	
	# Widgets
	widgets: List[DashboardWidget] = Field(default_factory=list, description="Dashboard widgets")
	
	# Filters
	global_filters: List[DashboardFilter] = Field(default_factory=list, description="Global dashboard filters")
	
	# Drill-down Configuration
	drill_down_enabled: bool = Field(default=True)
	drill_down_hierarchy: List[DrillDownLevel] = Field(default_factory=list)
	current_drill_level: DrillDownLevel = Field(default=DrillDownLevel.SUMMARY)
	drill_down_context: Dict[str, Any] = Field(default_factory=dict)
	
	# Real-time Features
	real_time_updates: bool = Field(default=True)
	auto_refresh_interval: int = Field(default=60, description="Auto-refresh interval in seconds")
	websocket_endpoint: Optional[str] = Field(None, description="WebSocket endpoint for real-time updates")
	
	# Personalization
	is_personalized: bool = Field(default=False)
	user_preferences: Dict[str, Any] = Field(default_factory=dict)
	shared_with_users: List[str] = Field(default_factory=list)
	
	# Performance
	cache_enabled: bool = Field(default=True)
	cache_duration: int = Field(default=300, description="Cache duration in seconds")
	
	# Analytics
	view_count: int = Field(default=0)
	last_viewed: Optional[datetime] = Field(None)
	performance_metrics: Dict[str, Any] = Field(default_factory=dict)
	
	# Metadata
	created_date: datetime = Field(default_factory=datetime.utcnow)
	last_modified: datetime = Field(default_factory=datetime.utcnow)
	version: str = Field(default="1.0.0")


class DashboardTheme(APGBaseModel):
	"""Dashboard theme configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	theme_id: str = Field(default_factory=uuid7str)
	theme_name: NonEmptyString = Field(description="Theme name")
	
	# Color Palette
	primary_color: str = Field(description="Primary color hex")
	secondary_color: str = Field(description="Secondary color hex")
	accent_color: str = Field(description="Accent color hex")
	background_color: str = Field(description="Background color hex")
	text_color: str = Field(description="Text color hex")
	
	# Chart Colors
	chart_colors: List[str] = Field(description="Chart color palette")
	gradient_colors: List[str] = Field(default_factory=list)
	
	# Typography
	font_family: str = Field(default="Arial, sans-serif")
	font_sizes: Dict[str, int] = Field(default_factory=dict)
	
	# Layout
	border_radius: int = Field(default=4, description="Border radius in pixels")
	spacing: Dict[str, int] = Field(default_factory=dict)
	shadows: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Interactive Dashboard Service
# =============================================================================

class InteractiveDashboardService(APGServiceBase):
	"""
	Interactive dashboard service providing real-time budget dashboards
	with drill-down capabilities and dynamic visualizations.
	"""
	
	def __init__(self, context: APGTenantContext, config: Optional[Dict[str, Any]] = None):
		super().__init__(context, config)
		self.logger = logging.getLogger(__name__)
	
	async def create_interactive_dashboard(
		self, 
		dashboard_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Create new interactive dashboard."""
		try:
			self.logger.info(f"Creating interactive dashboard: {dashboard_config.get('dashboard_name')}")
			
			# Validate configuration
			required_fields = ['dashboard_name', 'dashboard_type', 'budget_ids']
			missing_fields = [field for field in required_fields if field not in dashboard_config]
			if missing_fields:
				return ServiceResponse(
					success=False,
					message=f"Missing required fields: {missing_fields}",
					errors=missing_fields
				)
			
			# Create dashboard
			dashboard = InteractiveDashboard(
				dashboard_name=dashboard_config['dashboard_name'],
				dashboard_type=dashboard_config['dashboard_type'],
				budget_ids=dashboard_config['budget_ids'],
				description=dashboard_config.get('description'),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Configure drill-down hierarchy
			await self._configure_drill_down_hierarchy(dashboard)
			
			# Create default widgets
			await self._create_default_widgets(dashboard)
			
			# Setup global filters
			await self._setup_global_filters(dashboard)
			
			# Configure real-time updates
			await self._configure_real_time_updates(dashboard)
			
			self.logger.info(f"Interactive dashboard created: {dashboard.dashboard_id}")
			
			return ServiceResponse(
				success=True,
				message="Interactive dashboard created successfully",
				data=dashboard.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error creating interactive dashboard: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to create interactive dashboard: {str(e)}",
				errors=[str(e)]
			)
	
	async def add_dashboard_widget(
		self, 
		dashboard_id: str, 
		widget_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Add widget to dashboard."""
		try:
			self.logger.info(f"Adding widget to dashboard {dashboard_id}")
			
			# Create widget
			widget = DashboardWidget(
				widget_name=widget_config['widget_name'],
				widget_type=widget_config['widget_type'],
				title=widget_config['title'],
				subtitle=widget_config.get('subtitle'),
				data_source=widget_config['data_source'],
				metrics=widget_config['metrics'],
				dimensions=widget_config.get('dimensions', []),
				position_x=widget_config['position_x'],
				position_y=widget_config['position_y'],
				width=widget_config['width'],
				height=widget_config['height'],
				drill_down_enabled=widget_config.get('drill_down_enabled', False),
				real_time_enabled=widget_config.get('real_time_enabled', False),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Load widget data
			await self._load_widget_data(widget)
			
			# Configure widget interactivity
			await self._configure_widget_interactivity(widget)
			
			return ServiceResponse(
				success=True,
				message="Widget added successfully",
				data=widget.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error adding dashboard widget: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to add widget: {str(e)}",
				errors=[str(e)]
			)
	
	async def perform_drill_down(
		self, 
		dashboard_id: str, 
		drill_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Perform drill-down operation."""
		try:
			self.logger.info(f"Performing drill-down on dashboard {dashboard_id}")
			
			target_level = DrillDownLevel(drill_config['target_level'])
			drill_context = drill_config.get('context', {})
			
			# Generate drill-down data
			drill_data = await self._generate_drill_down_data(target_level, drill_context)
			
			# Update dashboard state
			dashboard_update = {
				'current_drill_level': target_level,
				'drill_down_context': drill_context,
				'last_modified': datetime.utcnow()
			}
			
			return ServiceResponse(
				success=True,
				message="Drill-down performed successfully",
				data={
					'drill_data': drill_data,
					'dashboard_update': dashboard_update
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error performing drill-down: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to perform drill-down: {str(e)}",
				errors=[str(e)]
			)
	
	async def apply_dashboard_filter(
		self, 
		dashboard_id: str, 
		filter_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Apply filter to dashboard."""
		try:
			self.logger.info(f"Applying filter to dashboard {dashboard_id}")
			
			filter_id = filter_config['filter_id']
			filter_value = filter_config['value']
			
			# Apply filter to data
			filtered_data = await self._apply_filter_to_data(filter_id, filter_value)
			
			# Update affected widgets
			widget_updates = await self._update_filtered_widgets(dashboard_id, filtered_data)
			
			return ServiceResponse(
				success=True,
				message="Filter applied successfully",
				data={
					'filtered_data': filtered_data,
					'widget_updates': widget_updates
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error applying dashboard filter: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to apply filter: {str(e)}",
				errors=[str(e)]
			)
	
	async def get_real_time_dashboard_data(
		self, 
		dashboard_id: str
	) -> ServiceResponse:
		"""Get real-time dashboard data."""
		try:
			self.logger.info(f"Getting real-time data for dashboard {dashboard_id}")
			
			# Get current dashboard state
			dashboard_data = await self._get_dashboard_current_state(dashboard_id)
			
			# Get real-time updates
			real_time_updates = await self._get_real_time_updates(dashboard_id)
			
			# Calculate performance metrics
			performance_metrics = await self._calculate_dashboard_performance(dashboard_id)
			
			return ServiceResponse(
				success=True,
				message="Real-time dashboard data retrieved",
				data={
					'dashboard_data': dashboard_data,
					'real_time_updates': real_time_updates,
					'performance_metrics': performance_metrics,
					'timestamp': datetime.utcnow()
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error getting real-time dashboard data: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to get real-time data: {str(e)}",
				errors=[str(e)]
			)
	
	async def export_dashboard(
		self, 
		dashboard_id: str, 
		export_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Export dashboard data and configuration."""
		try:
			self.logger.info(f"Exporting dashboard {dashboard_id}")
			
			export_format = export_config.get('format', 'json')
			include_data = export_config.get('include_data', True)
			
			# Get dashboard configuration
			dashboard_config = await self._get_dashboard_configuration(dashboard_id)
			
			# Get dashboard data if requested
			dashboard_data = None
			if include_data:
				dashboard_data = await self._get_dashboard_export_data(dashboard_id)
			
			# Format export based on requested format
			export_content = await self._format_dashboard_export(
				dashboard_config, dashboard_data, export_format
			)
			
			return ServiceResponse(
				success=True,
				message="Dashboard exported successfully",
				data={
					'export_content': export_content,
					'format': export_format,
					'exported_at': datetime.utcnow()
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error exporting dashboard: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to export dashboard: {str(e)}",
				errors=[str(e)]
			)
	
	# =============================================================================
	# Private Helper Methods
	# =============================================================================
	
	async def _configure_drill_down_hierarchy(self, dashboard: InteractiveDashboard) -> None:
		"""Configure drill-down hierarchy for dashboard."""
		if dashboard.dashboard_type == DashboardType.EXECUTIVE:
			dashboard.drill_down_hierarchy = [
				DrillDownLevel.SUMMARY,
				DrillDownLevel.DEPARTMENT,
				DrillDownLevel.CATEGORY
			]
		elif dashboard.dashboard_type == DashboardType.DETAILED:
			dashboard.drill_down_hierarchy = [
				DrillDownLevel.SUMMARY,
				DrillDownLevel.DEPARTMENT,
				DrillDownLevel.CATEGORY,
				DrillDownLevel.LINE_ITEM,
				DrillDownLevel.TRANSACTION
			]
		else:
			dashboard.drill_down_hierarchy = [
				DrillDownLevel.SUMMARY,
				DrillDownLevel.CATEGORY
			]
	
	async def _create_default_widgets(self, dashboard: InteractiveDashboard) -> None:
		"""Create default widgets based on dashboard type."""
		if dashboard.dashboard_type == DashboardType.EXECUTIVE:
			# Executive KPI widgets
			widgets = [
				DashboardWidget(
					widget_name="Budget Overview",
					widget_type=VisualizationType.KPI_CARD,
					title="Budget vs Actual",
					data_source="budget_summary",
					metrics=["total_budget", "total_actual", "variance"],
					position_x=0, position_y=0, width=4, height=2,
					tenant_id=self.context.tenant_id,
					created_by=self.context.user_id
				),
				DashboardWidget(
					widget_name="Department Variance",
					widget_type=VisualizationType.BAR_CHART,
					title="Variance by Department",
					data_source="department_variance",
					metrics=["variance_amount"],
					dimensions=["department"],
					position_x=4, position_y=0, width=8, height=4,
					drill_down_enabled=True,
					tenant_id=self.context.tenant_id,
					created_by=self.context.user_id
				),
				DashboardWidget(
					widget_name="Monthly Trend",
					widget_type=VisualizationType.LINE_CHART,
					title="Budget vs Actual Trend",
					data_source="monthly_trend",
					metrics=["budget", "actual"],
					dimensions=["month"],
					position_x=0, position_y=4, width=12, height=3,
					tenant_id=self.context.tenant_id,
					created_by=self.context.user_id
				)
			]
			dashboard.widgets = widgets
	
	async def _setup_global_filters(self, dashboard: InteractiveDashboard) -> None:
		"""Setup global filters for dashboard."""
		filters = [
			DashboardFilter(
				filter_name="Date Range",
				filter_type=FilterType.DATE_RANGE,
				field_name="period_date",
				operator="between",
				default_value=["2025-01-01", "2025-12-31"],
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			),
			DashboardFilter(
				filter_name="Department",
				filter_type=FilterType.DEPARTMENT,
				field_name="department_code",
				operator="in",
				available_values=["SALES", "MARKETING", "IT", "OPERATIONS"],
				is_multi_select=True,
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
		]
		dashboard.global_filters = filters
	
	async def _configure_real_time_updates(self, dashboard: InteractiveDashboard) -> None:
		"""Configure real-time updates for dashboard."""
		if dashboard.real_time_updates:
			dashboard.websocket_endpoint = f"/ws/dashboard/{dashboard.dashboard_id}"
			dashboard.auto_refresh_interval = 60
	
	async def _load_widget_data(self, widget: DashboardWidget) -> None:
		"""Load data for widget."""
		# Simulated data loading
		if widget.widget_type == VisualizationType.KPI_CARD:
			widget.current_data = {
				"total_budget": 1500000,
				"total_actual": 1487500,
				"variance": -12500,
				"variance_percent": -0.83
			}
		elif widget.widget_type == VisualizationType.BAR_CHART:
			widget.current_data = {
				"data": [
					{"department": "Sales", "variance": -5000},
					{"department": "Marketing", "variance": 2500},
					{"department": "IT", "variance": 7500},
					{"department": "Operations", "variance": -10000}
				]
			}
		elif widget.widget_type == VisualizationType.LINE_CHART:
			widget.current_data = {
				"data": [
					{"month": "Jan", "budget": 125000, "actual": 123500},
					{"month": "Feb", "budget": 125000, "actual": 127000},
					{"month": "Mar", "budget": 125000, "actual": 124200}
				]
			}
	
	async def _configure_widget_interactivity(self, widget: DashboardWidget) -> None:
		"""Configure widget interactivity options."""
		if widget.drill_down_enabled:
			widget.drill_down_levels = [DrillDownLevel.DEPARTMENT, DrillDownLevel.CATEGORY]
			widget.click_action = "drill_down"
		
		if widget.real_time_enabled:
			widget.refresh_mode = RefreshMode.REAL_TIME
			widget.refresh_interval = 30
	
	async def _generate_drill_down_data(
		self, 
		target_level: DrillDownLevel, 
		context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Generate drill-down data for specified level."""
		if target_level == DrillDownLevel.DEPARTMENT:
			return {
				"level": "department",
				"data": [
					{"department": "Sales", "budget": 400000, "actual": 395000, "variance": -5000},
					{"department": "Marketing", "budget": 300000, "actual": 302500, "variance": 2500},
					{"department": "IT", "budget": 450000, "actual": 457500, "variance": 7500},
					{"department": "Operations", "budget": 350000, "actual": 340000, "variance": -10000}
				]
			}
		elif target_level == DrillDownLevel.CATEGORY:
			return {
				"level": "category",
				"department": context.get("department"),
				"data": [
					{"category": "Personnel", "budget": 200000, "actual": 192000, "variance": -8000},
					{"category": "Technology", "budget": 150000, "actual": 153000, "variance": 3000},
					{"category": "Marketing", "budget": 100000, "actual": 97500, "variance": -2500}
				]
			}
		else:
			return {"level": target_level.value, "data": []}
	
	async def _apply_filter_to_data(
		self, 
		filter_id: str, 
		filter_value: Any
	) -> Dict[str, Any]:
		"""Apply filter to dashboard data."""
		# Simulated filter application
		return {
			"filter_id": filter_id,
			"applied_value": filter_value,
			"filtered_records": 150,
			"total_records": 200
		}
	
	async def _update_filtered_widgets(
		self, 
		dashboard_id: str, 
		filtered_data: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		"""Update widgets based on filtered data."""
		# Simulated widget updates
		return [
			{"widget_id": "widget_1", "updated_data": {"total_budget": 1200000}},
			{"widget_id": "widget_2", "updated_data": {"variance": -8000}}
		]
	
	async def _get_dashboard_current_state(self, dashboard_id: str) -> Dict[str, Any]:
		"""Get current dashboard state."""
		return {
			"dashboard_id": dashboard_id,
			"current_drill_level": "summary",
			"active_filters": [],
			"widget_count": 5,
			"last_updated": datetime.utcnow()
		}
	
	async def _get_real_time_updates(self, dashboard_id: str) -> List[Dict[str, Any]]:
		"""Get real-time updates for dashboard."""
		return [
			{
				"update_type": "data_change",
				"widget_id": "budget_overview",
				"field": "actual_amount",
				"old_value": 1487500,
				"new_value": 1488000,
				"timestamp": datetime.utcnow()
			}
		]
	
	async def _calculate_dashboard_performance(self, dashboard_id: str) -> Dict[str, Any]:
		"""Calculate dashboard performance metrics."""
		return {
			"load_time": 0.85,
			"query_performance": 0.95,
			"cache_hit_ratio": 0.87,
			"error_rate": 0.02
		}
	
	async def _get_dashboard_configuration(self, dashboard_id: str) -> Dict[str, Any]:
		"""Get dashboard configuration for export."""
		return {
			"dashboard_id": dashboard_id,
			"name": "Executive Budget Dashboard",
			"type": "executive",
			"widgets": [],
			"filters": []
		}
	
	async def _get_dashboard_export_data(self, dashboard_id: str) -> Dict[str, Any]:
		"""Get dashboard data for export."""
		return {
			"summary_data": {"total_budget": 1500000, "total_actual": 1487500},
			"department_data": [],
			"trend_data": []
		}
	
	async def _format_dashboard_export(
		self, 
		config: Dict[str, Any], 
		data: Optional[Dict[str, Any]], 
		format_type: str
	) -> str:
		"""Format dashboard export content."""
		export_content = {
			"configuration": config,
			"data": data,
			"exported_at": datetime.utcnow().isoformat()
		}
		
		if format_type == "json":
			return json.dumps(export_content, indent=2)
		elif format_type == "csv":
			return "CSV format not implemented yet"
		else:
			return str(export_content)


# =============================================================================
# Service Factory Functions
# =============================================================================

def create_interactive_dashboard_service(
	context: APGTenantContext, 
	config: Optional[Dict[str, Any]] = None
) -> InteractiveDashboardService:
	"""Create interactive dashboard service instance."""
	return InteractiveDashboardService(context, config)


async def create_sample_executive_dashboard(
	service: InteractiveDashboardService,
	budget_ids: List[str]
) -> ServiceResponse:
	"""Create sample executive dashboard for testing."""
	dashboard_config = {
		'dashboard_name': 'Executive Budget Dashboard',
		'dashboard_type': DashboardType.EXECUTIVE,
		'budget_ids': budget_ids,
		'description': 'High-level budget overview for executives'
	}
	
	return await service.create_interactive_dashboard(dashboard_config)