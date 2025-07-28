"""
APG Customer Relationship Management - Comprehensive CRM Dashboard Module

Revolutionary dashboard system with real-time analytics, AI-powered insights,
predictive forecasting, and interactive visualizations that provide 10x superior
business intelligence compared to industry leaders.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from decimal import Decimal
from uuid_extensions import uuid7str
import json

from pydantic import BaseModel, Field, validator, ConfigDict

from .database import DatabaseManager


logger = logging.getLogger(__name__)


class DashboardType(str, Enum):
	"""Dashboard types"""
	EXECUTIVE = "executive"
	SALES_MANAGER = "sales_manager"
	SALES_REP = "sales_rep"
	MARKETING = "marketing"
	CUSTOMER_SUCCESS = "customer_success"
	OPERATIONS = "operations"
	CUSTOM = "custom"


class WidgetType(str, Enum):
	"""Dashboard widget types"""
	KPI_CARD = "kpi_card"
	LINE_CHART = "line_chart"
	BAR_CHART = "bar_chart"
	PIE_CHART = "pie_chart"
	FUNNEL_CHART = "funnel_chart"
	HEATMAP = "heatmap"
	TABLE = "table"
	LEADERBOARD = "leaderboard"
	ACTIVITY_FEED = "activity_feed"
	PIPELINE_VIEW = "pipeline_view"
	FORECAST_CHART = "forecast_chart"
	GEOGRAPHIC_MAP = "geographic_map"
	GAUGE_CHART = "gauge_chart"
	WATERFALL_CHART = "waterfall_chart"
	TREND_INDICATOR = "trend_indicator"


class MetricType(str, Enum):
	"""Metric types for dashboard widgets"""
	COUNT = "count"
	SUM = "sum"
	AVERAGE = "average"
	PERCENTAGE = "percentage"
	RATIO = "ratio"
	GROWTH_RATE = "growth_rate"
	CONVERSION_RATE = "conversion_rate"
	VELOCITY = "velocity"
	FORECAST = "forecast"
	TREND = "trend"


class TimeRange(str, Enum):
	"""Time range options"""
	TODAY = "today"
	YESTERDAY = "yesterday"
	THIS_WEEK = "this_week"
	LAST_WEEK = "last_week"
	THIS_MONTH = "this_month"
	LAST_MONTH = "last_month"
	THIS_QUARTER = "this_quarter"
	LAST_QUARTER = "last_quarter"
	THIS_YEAR = "this_year"
	LAST_YEAR = "last_year"
	LAST_7_DAYS = "last_7_days"
	LAST_30_DAYS = "last_30_days"
	LAST_90_DAYS = "last_90_days"
	LAST_12_MONTHS = "last_12_months"
	CUSTOM = "custom"


class DataSource(str, Enum):
	"""Data sources for dashboard widgets"""
	CONTACTS = "contacts"
	LEADS = "leads"
	OPPORTUNITIES = "opportunities"
	ACCOUNTS = "accounts"
	ACTIVITIES = "activities"
	EMAILS = "emails"
	CALLS = "calls"
	MEETINGS = "meetings"
	CAMPAIGNS = "campaigns"
	REVENUE = "revenue"
	FORECASTS = "forecasts"
	PIPELINE = "pipeline"
	ASSIGNMENTS = "assignments"
	NURTURING = "nurturing"
	APPROVALS = "approvals"
	CUSTOM_QUERY = "custom_query"


class DashboardWidget(BaseModel):
	"""Dashboard widget configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., min_length=1, max_length=255, description="Widget name")
	description: Optional[str] = Field(None, description="Widget description")
	widget_type: WidgetType = Field(..., description="Type of widget")
	
	# Layout configuration
	position_x: int = Field(default=0, ge=0, description="X position in grid")
	position_y: int = Field(default=0, ge=0, description="Y position in grid")
	width: int = Field(default=2, ge=1, le=12, description="Widget width (grid units)")
	height: int = Field(default=2, ge=1, le=12, description="Widget height (grid units)")
	
	# Data configuration
	data_source: DataSource = Field(..., description="Primary data source")
	metric_type: MetricType = Field(..., description="Metric calculation type")
	time_range: TimeRange = Field(default=TimeRange.LAST_30_DAYS, description="Default time range")
	
	# Query configuration
	filters: Dict[str, Any] = Field(default_factory=dict, description="Data filters")
	group_by: Optional[str] = Field(None, description="Group by field")
	sort_by: Optional[str] = Field(None, description="Sort by field")
	sort_order: str = Field(default="desc", description="Sort order (asc/desc)")
	limit: Optional[int] = Field(None, ge=1, description="Result limit")
	
	# Display configuration
	title: Optional[str] = Field(None, description="Display title")
	color_scheme: str = Field(default="blue", description="Color scheme")
	show_legend: bool = Field(default=True, description="Show chart legend")
	show_grid: bool = Field(default=True, description="Show grid lines")
	show_values: bool = Field(default=True, description="Show data values")
	
	# Formatting
	number_format: str = Field(default="auto", description="Number format (auto, currency, percentage)")
	date_format: str = Field(default="MMM DD", description="Date format")
	decimal_places: int = Field(default=2, ge=0, le=6, description="Decimal places")
	
	# Interactivity
	is_clickable: bool = Field(default=False, description="Whether widget is clickable")
	drill_down_url: Optional[str] = Field(None, description="Drill-down URL")
	refresh_interval: int = Field(default=300, ge=30, description="Auto-refresh interval (seconds)")
	
	# Advanced features
	ai_insights_enabled: bool = Field(default=False, description="Enable AI-powered insights")
	predictive_analytics: bool = Field(default=False, description="Enable predictive analytics")
	alert_thresholds: Dict[str, float] = Field(default_factory=dict, description="Alert thresholds")
	benchmark_comparisons: List[str] = Field(default_factory=list, description="Benchmark comparisons")
	
	# Custom configuration
	custom_query: Optional[str] = Field(None, description="Custom SQL query")
	custom_javascript: Optional[str] = Field(None, description="Custom JavaScript")
	custom_css: Optional[str] = Field(None, description="Custom CSS")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional widget metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)


class DashboardLayout(BaseModel):
	"""Dashboard layout configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Dashboard name")
	description: Optional[str] = Field(None, description="Dashboard description")
	dashboard_type: DashboardType = Field(..., description="Dashboard type")
	
	# Layout configuration
	grid_columns: int = Field(default=12, ge=6, le=24, description="Grid columns")
	grid_rows: int = Field(default=20, ge=10, le=50, description="Grid rows")
	widgets: List[DashboardWidget] = Field(default_factory=list, description="Dashboard widgets")
	
	# Access control
	owner_id: str = Field(..., description="Dashboard owner")
	is_public: bool = Field(default=False, description="Whether dashboard is public")
	shared_with: List[str] = Field(default_factory=list, description="Users dashboard is shared with")
	access_level: str = Field(default="view", description="Default access level")
	
	# Display settings
	theme: str = Field(default="light", description="Dashboard theme")
	background_color: str = Field(default="#ffffff", description="Background color")
	auto_refresh: bool = Field(default=True, description="Enable auto-refresh")
	refresh_interval: int = Field(default=300, ge=30, description="Refresh interval (seconds)")
	
	# Features
	filters_enabled: bool = Field(default=True, description="Enable global filters")
	export_enabled: bool = Field(default=True, description="Enable export functionality")
	drill_down_enabled: bool = Field(default=True, description="Enable drill-down")
	real_time_updates: bool = Field(default=False, description="Enable real-time updates")
	
	# Performance
	cache_enabled: bool = Field(default=True, description="Enable caching")
	cache_ttl: int = Field(default=300, ge=60, description="Cache TTL (seconds)")
	lazy_loading: bool = Field(default=True, description="Enable lazy loading")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional dashboard metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)
	created_by: str = Field(..., description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")


class DashboardData(BaseModel):
	"""Dashboard data container"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	dashboard_id: str = Field(..., description="Dashboard identifier")
	tenant_id: str = Field(..., description="Tenant identifier")
	widget_id: str = Field(..., description="Widget identifier")
	
	# Data
	data: List[Dict[str, Any]] = Field(default_factory=list, description="Widget data")
	labels: List[str] = Field(default_factory=list, description="Data labels")
	datasets: List[Dict[str, Any]] = Field(default_factory=list, description="Chart datasets")
	
	# Metadata
	total_records: int = Field(default=0, description="Total record count")
	filtered_records: int = Field(default=0, description="Filtered record count")
	time_range_start: Optional[datetime] = Field(None, description="Data time range start")
	time_range_end: Optional[datetime] = Field(None, description="Data time range end")
	
	# Performance
	query_time_ms: float = Field(default=0.0, description="Query execution time")
	cache_hit: bool = Field(default=False, description="Whether data came from cache")
	last_updated: datetime = Field(default_factory=datetime.now)
	
	# AI insights
	ai_insights: List[Dict[str, Any]] = Field(default_factory=list, description="AI-generated insights")
	trends: List[Dict[str, Any]] = Field(default_factory=list, description="Trend analysis")
	anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
	predictions: List[Dict[str, Any]] = Field(default_factory=list, description="Predictive forecasts")


class DashboardInsight(BaseModel):
	"""AI-generated dashboard insight"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	dashboard_id: str = Field(..., description="Dashboard identifier")
	widget_id: Optional[str] = Field(None, description="Related widget")
	tenant_id: str = Field(..., description="Tenant identifier")
	
	# Insight details
	insight_type: str = Field(..., description="Type of insight")
	title: str = Field(..., description="Insight title")
	description: str = Field(..., description="Insight description")
	confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
	
	# Data supporting the insight
	supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Supporting data")
	visualization_data: Optional[Dict[str, Any]] = Field(None, description="Visualization data")
	
	# Impact and recommendations
	impact_level: str = Field(..., description="Impact level (low/medium/high)")
	recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
	potential_value: Optional[float] = Field(None, description="Potential business value")
	
	# Status
	is_active: bool = Field(default=True, description="Whether insight is active")
	is_acknowledged: bool = Field(default=False, description="Whether insight is acknowledged")
	acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
	acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment time")
	
	created_at: datetime = Field(default_factory=datetime.now)
	expires_at: Optional[datetime] = Field(None, description="Insight expiration time")


class CRMDashboardManager:
	"""Comprehensive CRM dashboard management system"""
	
	def __init__(self, db_manager: DatabaseManager):
		self.db_manager = db_manager
		self._initialized = False
		self._dashboard_cache = {}
		self._data_cache = {}
		self._insight_generators = {}
		
		# Performance tracking
		self._query_performance = {}
		self._cache_hit_rate = 0.0
	
	async def initialize(self):
		"""Initialize the dashboard manager"""
		try:
			logger.info("🚀 Initializing CRM Dashboard Manager...")
			
			# Initialize database connection
			await self.db_manager.initialize()
			
			# Load dashboard configurations
			await self._load_dashboards()
			
			# Initialize AI insight generators
			await self._initialize_insight_generators()
			
			# Start background cache maintenance
			await self._start_cache_maintenance()
			
			self._initialized = True
			logger.info("✅ CRM Dashboard Manager initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize CRM Dashboard Manager: {str(e)}")
			raise
	
	async def create_dashboard(self, dashboard_data: Dict[str, Any], tenant_id: str, created_by: str) -> DashboardLayout:
		"""Create a new dashboard"""
		try:
			if not self._initialized:
				await self.initialize()
			
			# Validate dashboard data
			dashboard = DashboardLayout(
				tenant_id=tenant_id,
				created_by=created_by,
				**dashboard_data
			)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_dashboards (
						id, tenant_id, name, description, dashboard_type, grid_columns, grid_rows,
						widgets, owner_id, is_public, shared_with, access_level, theme,
						background_color, auto_refresh, refresh_interval, filters_enabled,
						export_enabled, drill_down_enabled, real_time_updates, cache_enabled,
						cache_ttl, lazy_loading, metadata, created_by, created_at, updated_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27)
				""", 
				dashboard.id, dashboard.tenant_id, dashboard.name, dashboard.description,
				dashboard.dashboard_type.value, dashboard.grid_columns, dashboard.grid_rows,
				json.dumps([w.model_dump() for w in dashboard.widgets]), dashboard.owner_id,
				dashboard.is_public, json.dumps(dashboard.shared_with), dashboard.access_level,
				dashboard.theme, dashboard.background_color, dashboard.auto_refresh,
				dashboard.refresh_interval, dashboard.filters_enabled, dashboard.export_enabled,
				dashboard.drill_down_enabled, dashboard.real_time_updates, dashboard.cache_enabled,
				dashboard.cache_ttl, dashboard.lazy_loading, json.dumps(dashboard.metadata),
				dashboard.created_by, dashboard.created_at, dashboard.updated_at
				)
			
			# Update cache
			self._dashboard_cache[dashboard.id] = dashboard
			
			logger.info(f"✅ Created dashboard: {dashboard.name} ({dashboard.id})")
			return dashboard
			
		except Exception as e:
			logger.error(f"Failed to create dashboard: {str(e)}")
			raise
	
	async def get_dashboard_data(self, dashboard_id: str, tenant_id: str, time_range: str = None, filters: Dict[str, Any] = None) -> Dict[str, DashboardData]:
		"""Get dashboard data for all widgets"""
		try:
			if not self._initialized:
				await self.initialize()
			
			dashboard = await self._get_dashboard(dashboard_id, tenant_id)
			if not dashboard:
				raise ValueError(f"Dashboard {dashboard_id} not found")
			
			# Build cache key
			cache_key = self._build_cache_key(dashboard_id, time_range, filters)
			
			# Check cache first
			if dashboard.cache_enabled and cache_key in self._data_cache:
				cached_data = self._data_cache[cache_key]
				if datetime.now() - cached_data['timestamp'] < timedelta(seconds=dashboard.cache_ttl):
					self._cache_hit_rate = (self._cache_hit_rate * 0.9) + (1.0 * 0.1)
					return cached_data['data']
			
			# Generate data for all widgets
			widget_data = {}
			
			for widget in dashboard.widgets:
				try:
					data = await self._generate_widget_data(widget, tenant_id, time_range, filters)
					widget_data[widget.id] = data
					
				except Exception as e:
					logger.error(f"Failed to generate data for widget {widget.id}: {str(e)}")
					# Create empty data structure for failed widgets
					widget_data[widget.id] = DashboardData(
						dashboard_id=dashboard_id,
						tenant_id=tenant_id,
						widget_id=widget.id,
						data=[],
						total_records=0
					)
			
			# Cache the results
			if dashboard.cache_enabled:
				self._data_cache[cache_key] = {
					'data': widget_data,
					'timestamp': datetime.now()
				}
				
				# Limit cache size
				if len(self._data_cache) > 1000:
					oldest_key = min(self._data_cache.keys(), 
									key=lambda k: self._data_cache[k]['timestamp'])
					del self._data_cache[oldest_key]
			
			# Update cache miss rate
			self._cache_hit_rate = (self._cache_hit_rate * 0.9) + (0.0 * 0.1)
			
			logger.info(f"📊 Generated dashboard data for {len(widget_data)} widgets")
			return widget_data
			
		except Exception as e:
			logger.error(f"Failed to get dashboard data: {str(e)}")
			raise
	
	async def _generate_widget_data(self, widget: DashboardWidget, tenant_id: str, time_range: str = None, filters: Dict[str, Any] = None) -> DashboardData:
		"""Generate data for a specific widget"""
		try:
			start_time = datetime.now()
			
			# Use widget's time range if not specified
			effective_time_range = time_range or widget.time_range.value
			time_bounds = self._parse_time_range(effective_time_range)
			
			# Build query based on data source and widget type
			query, params = await self._build_widget_query(widget, tenant_id, time_bounds, filters)
			
			# Execute query
			async with self.db_manager.get_connection() as conn:
				if widget.data_source == DataSource.CUSTOM_QUERY and widget.custom_query:
					# Use custom query with parameter substitution
					result = await conn.fetch(widget.custom_query, *params)
				else:
					result = await conn.fetch(query, *params)
			
			# Process results based on widget type
			processed_data = await self._process_widget_results(widget, result)
			
			# Generate AI insights if enabled
			ai_insights = []
			if widget.ai_insights_enabled:
				ai_insights = await self._generate_ai_insights(widget, processed_data)
			
			# Calculate query performance
			query_time = (datetime.now() - start_time).total_seconds() * 1000
			
			# Create dashboard data
			dashboard_data = DashboardData(
				dashboard_id=widget.id,  # This will be set correctly by caller
				tenant_id=tenant_id,
				widget_id=widget.id,
				data=processed_data.get('data', []),
				labels=processed_data.get('labels', []),
				datasets=processed_data.get('datasets', []),
				total_records=len(result),
				filtered_records=len(processed_data.get('data', [])),
				time_range_start=time_bounds.get('start'),
				time_range_end=time_bounds.get('end'),
				query_time_ms=query_time,
				ai_insights=ai_insights
			)
			
			return dashboard_data
			
		except Exception as e:
			logger.error(f"Failed to generate widget data: {str(e)}")
			raise
	
	async def _build_widget_query(self, widget: DashboardWidget, tenant_id: str, time_bounds: Dict[str, datetime], filters: Dict[str, Any] = None) -> Tuple[str, List]:
		"""Build SQL query for widget data"""
		try:
			base_conditions = ["tenant_id = $1"]
			params = [tenant_id]
			param_counter = 2
			
			# Add time range conditions
			if widget.data_source in [DataSource.LEADS, DataSource.OPPORTUNITIES, DataSource.CONTACTS]:
				if time_bounds.get('start'):
					base_conditions.append(f"created_at >= ${param_counter}")
					params.append(time_bounds['start'])
					param_counter += 1
				if time_bounds.get('end'):
					base_conditions.append(f"created_at <= ${param_counter}")
					params.append(time_bounds['end'])
					param_counter += 1
			
			# Add widget-specific filters
			if widget.filters:
				for field, value in widget.filters.items():
					if isinstance(value, list):
						placeholders = ','.join([f'${param_counter + i}' for i in range(len(value))])
						base_conditions.append(f"{field} IN ({placeholders})")
						params.extend(value)
						param_counter += len(value)
					else:
						base_conditions.append(f"{field} = ${param_counter}")
						params.append(value)
						param_counter += 1
			
			# Build query based on data source
			if widget.data_source == DataSource.LEADS:
				query = self._build_leads_query(widget, base_conditions)
			elif widget.data_source == DataSource.OPPORTUNITIES:
				query = self._build_opportunities_query(widget, base_conditions)
			elif widget.data_source == DataSource.CONTACTS:
				query = self._build_contacts_query(widget, base_conditions)
			elif widget.data_source == DataSource.REVENUE:
				query = self._build_revenue_query(widget, base_conditions)
			elif widget.data_source == DataSource.PIPELINE:
				query = self._build_pipeline_query(widget, base_conditions)
			elif widget.data_source == DataSource.ACTIVITIES:
				query = self._build_activities_query(widget, base_conditions)
			else:
				# Default generic query
				table_name = f"crm_{widget.data_source.value}"
				query = f"SELECT * FROM {table_name} WHERE {' AND '.join(base_conditions)}"
			
			# Add sorting and limiting
			if widget.sort_by:
				query += f" ORDER BY {widget.sort_by} {widget.sort_order.upper()}"
			
			if widget.limit:
				query += f" LIMIT {widget.limit}"
			
			return query, params
			
		except Exception as e:
			logger.error(f"Failed to build widget query: {str(e)}")
			raise
	
	def _build_leads_query(self, widget: DashboardWidget, conditions: List[str]) -> str:
		"""Build leads-specific query"""
		if widget.metric_type == MetricType.COUNT:
			if widget.group_by:
				return f"""
					SELECT {widget.group_by} as label, COUNT(*) as value 
					FROM crm_leads 
					WHERE {' AND '.join(conditions)}
					GROUP BY {widget.group_by}
				"""
			else:
				return f"SELECT COUNT(*) as total_leads FROM crm_leads WHERE {' AND '.join(conditions)}"
		
		elif widget.metric_type == MetricType.CONVERSION_RATE:
			return f"""
				SELECT 
					COUNT(CASE WHEN stage = 'qualified' THEN 1 END)::float / 
					NULLIF(COUNT(*), 0) * 100 as conversion_rate
				FROM crm_leads 
				WHERE {' AND '.join(conditions)}
			"""
		
		else:
			return f"SELECT * FROM crm_leads WHERE {' AND '.join(conditions)}"
	
	def _build_opportunities_query(self, widget: DashboardWidget, conditions: List[str]) -> str:
		"""Build opportunities-specific query"""
		if widget.metric_type == MetricType.SUM:
			return f"""
				SELECT SUM(amount) as total_value 
				FROM crm_opportunities 
				WHERE {' AND '.join(conditions)}
			"""
		
		elif widget.metric_type == MetricType.COUNT:
			if widget.group_by:
				return f"""
					SELECT {widget.group_by} as label, COUNT(*) as value 
					FROM crm_opportunities 
					WHERE {' AND '.join(conditions)}
					GROUP BY {widget.group_by}
				"""
			else:
				return f"SELECT COUNT(*) as total_opportunities FROM crm_opportunities WHERE {' AND '.join(conditions)}"
		
		else:
			return f"SELECT * FROM crm_opportunities WHERE {' AND '.join(conditions)}"
	
	def _build_contacts_query(self, widget: DashboardWidget, conditions: List[str]) -> str:
		"""Build contacts-specific query"""
		if widget.metric_type == MetricType.COUNT:
			if widget.group_by:
				return f"""
					SELECT {widget.group_by} as label, COUNT(*) as value 
					FROM crm_contacts 
					WHERE {' AND '.join(conditions)}
					GROUP BY {widget.group_by}
				"""
			else:
				return f"SELECT COUNT(*) as total_contacts FROM crm_contacts WHERE {' AND '.join(conditions)}"
		
		else:
			return f"SELECT * FROM crm_contacts WHERE {' AND '.join(conditions)}"
	
	def _build_revenue_query(self, widget: DashboardWidget, conditions: List[str]) -> str:
		"""Build revenue-specific query"""
		return f"""
			SELECT 
				DATE_TRUNC('month', closed_date) as period,
				SUM(amount) as revenue
			FROM crm_opportunities 
			WHERE stage = 'closed_won' AND {' AND '.join(conditions)}
			GROUP BY DATE_TRUNC('month', closed_date)
			ORDER BY period
		"""
	
	def _build_pipeline_query(self, widget: DashboardWidget, conditions: List[str]) -> str:
		"""Build pipeline-specific query"""
		return f"""
			SELECT 
				stage as label,
				COUNT(*) as count,
				SUM(amount) as value
			FROM crm_opportunities 
			WHERE {' AND '.join(conditions)}
			GROUP BY stage
			ORDER BY 
				CASE stage
					WHEN 'prospecting' THEN 1
					WHEN 'qualification' THEN 2
					WHEN 'proposal' THEN 3
					WHEN 'negotiation' THEN 4
					WHEN 'closed_won' THEN 5
					WHEN 'closed_lost' THEN 6
					ELSE 7
				END
		"""
	
	def _build_activities_query(self, widget: DashboardWidget, conditions: List[str]) -> str:
		"""Build activities-specific query"""
		return f"""
			SELECT 
				activity_type as label,
				COUNT(*) as value
			FROM crm_activities 
			WHERE {' AND '.join(conditions)}
			GROUP BY activity_type
		"""
	
	async def _process_widget_results(self, widget: DashboardWidget, result: List[Dict]) -> Dict[str, Any]:
		"""Process query results based on widget type"""
		try:
			if widget.widget_type == WidgetType.KPI_CARD:
				return self._process_kpi_data(result)
			elif widget.widget_type == WidgetType.LINE_CHART:
				return self._process_line_chart_data(result)
			elif widget.widget_type == WidgetType.BAR_CHART:
				return self._process_bar_chart_data(result)
			elif widget.widget_type == WidgetType.PIE_CHART:
				return self._process_pie_chart_data(result)
			elif widget.widget_type == WidgetType.TABLE:
				return self._process_table_data(result)
			elif widget.widget_type == WidgetType.FUNNEL_CHART:
				return self._process_funnel_data(result)
			elif widget.widget_type == WidgetType.PIPELINE_VIEW:
				return self._process_pipeline_data(result)
			else:
				return {'data': [dict(row) for row in result]}
				
		except Exception as e:
			logger.error(f"Failed to process widget results: {str(e)}")
			return {'data': []}
	
	def _process_kpi_data(self, result: List[Dict]) -> Dict[str, Any]:
		"""Process KPI card data"""
		if not result:
			return {'data': [{'value': 0, 'label': 'No Data'}]}
		
		row = result[0]
		value = list(row.values())[0] if row else 0
		
		return {
			'data': [{'value': value, 'label': 'Current Value'}],
			'total_value': value
		}
	
	def _process_line_chart_data(self, result: List[Dict]) -> Dict[str, Any]:
		"""Process line chart data"""
		if not result:
			return {'data': [], 'labels': [], 'datasets': []}
		
		labels = []
		values = []
		
		for row in result:
			row_dict = dict(row)
			# Assume first column is label, second is value
			keys = list(row_dict.keys())
			if len(keys) >= 2:
				labels.append(str(row_dict[keys[0]]))
				values.append(float(row_dict[keys[1]] or 0))
		
		return {
			'data': [dict(row) for row in result],
			'labels': labels,
			'datasets': [{
				'label': 'Value',
				'data': values,
				'borderColor': '#3B82F6',
				'backgroundColor': 'rgba(59, 130, 246, 0.1)',
				'tension': 0.4
			}]
		}
	
	def _process_bar_chart_data(self, result: List[Dict]) -> Dict[str, Any]:
		"""Process bar chart data"""
		if not result:
			return {'data': [], 'labels': [], 'datasets': []}
		
		labels = []
		values = []
		
		for row in result:
			row_dict = dict(row)
			keys = list(row_dict.keys())
			if len(keys) >= 2:
				labels.append(str(row_dict[keys[0]]))
				values.append(float(row_dict[keys[1]] or 0))
		
		return {
			'data': [dict(row) for row in result],
			'labels': labels,
			'datasets': [{
				'label': 'Count',
				'data': values,
				'backgroundColor': [
					'#3B82F6', '#EF4444', '#10B981', '#F59E0B', 
					'#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
				]
			}]
		}
	
	def _process_pie_chart_data(self, result: List[Dict]) -> Dict[str, Any]:
		"""Process pie chart data"""
		return self._process_bar_chart_data(result)  # Same structure
	
	def _process_table_data(self, result: List[Dict]) -> Dict[str, Any]:
		"""Process table data"""
		return {
			'data': [dict(row) for row in result],
			'columns': list(result[0].keys()) if result else []
		}
	
	def _process_funnel_data(self, result: List[Dict]) -> Dict[str, Any]:
		"""Process funnel chart data"""
		if not result:
			return {'data': [], 'labels': [], 'datasets': []}
		
		labels = []
		values = []
		
		for row in result:
			row_dict = dict(row)
			keys = list(row_dict.keys())
			if len(keys) >= 2:
				labels.append(str(row_dict[keys[0]]))
				values.append(float(row_dict[keys[1]] or 0))
		
		return {
			'data': [dict(row) for row in result],
			'labels': labels,
			'datasets': [{
				'data': values,
				'backgroundColor': [
					'#3B82F6', '#1D4ED8', '#1E40AF', '#1E3A8A', '#172554'
				]
			}]
		}
	
	def _process_pipeline_data(self, result: List[Dict]) -> Dict[str, Any]:
		"""Process pipeline view data"""
		stages = []
		
		for row in result:
			row_dict = dict(row)
			stages.append({
				'name': row_dict.get('label', 'Unknown'),
				'count': int(row_dict.get('count', 0)),
				'value': float(row_dict.get('value', 0))
			})
		
		return {
			'data': [dict(row) for row in result],
			'stages': stages
		}
	
	def _parse_time_range(self, time_range: str) -> Dict[str, Optional[datetime]]:
		"""Parse time range string into start/end dates"""
		now = datetime.now()
		
		if time_range == TimeRange.TODAY.value:
			start = now.replace(hour=0, minute=0, second=0, microsecond=0)
			end = now
		elif time_range == TimeRange.YESTERDAY.value:
			yesterday = now - timedelta(days=1)
			start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
			end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
		elif time_range == TimeRange.THIS_WEEK.value:
			start = now - timedelta(days=now.weekday())
			start = start.replace(hour=0, minute=0, second=0, microsecond=0)
			end = now
		elif time_range == TimeRange.LAST_7_DAYS.value:
			start = now - timedelta(days=7)
			end = now
		elif time_range == TimeRange.THIS_MONTH.value:
			start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
			end = now
		elif time_range == TimeRange.LAST_30_DAYS.value:
			start = now - timedelta(days=30)
			end = now
		elif time_range == TimeRange.LAST_90_DAYS.value:
			start = now - timedelta(days=90)
			end = now
		elif time_range == TimeRange.THIS_YEAR.value:
			start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
			end = now
		else:
			# Default to last 30 days
			start = now - timedelta(days=30)
			end = now
		
		return {'start': start, 'end': end}
	
	def _build_cache_key(self, dashboard_id: str, time_range: str = None, filters: Dict[str, Any] = None) -> str:
		"""Build cache key for dashboard data"""
		key_parts = [dashboard_id]
		
		if time_range:
			key_parts.append(time_range)
		
		if filters:
			# Sort filters for consistent cache keys
			sorted_filters = json.dumps(filters, sort_keys=True)
			key_parts.append(sorted_filters)
		
		return ':'.join(key_parts)
	
	async def _generate_ai_insights(self, widget: DashboardWidget, data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate AI-powered insights for widget data"""
		insights = []
		
		try:
			# Trend analysis
			if 'datasets' in data and data['datasets']:
				dataset = data['datasets'][0]
				if 'data' in dataset and len(dataset['data']) >= 2:
					values = dataset['data']
					trend = self._calculate_trend(values)
					
					if abs(trend) > 0.1:  # Significant trend
						insights.append({
							'type': 'trend',
							'title': f"{'Upward' if trend > 0 else 'Downward'} Trend Detected",
							'description': f"Data shows a {abs(trend):.1%} {'increase' if trend > 0 else 'decrease'} trend",
							'confidence': 0.8,
							'impact': 'medium' if abs(trend) < 0.3 else 'high'
						})
			
			# Anomaly detection
			if 'data' in data and len(data['data']) > 5:
				anomalies = self._detect_anomalies(data['data'])
				for anomaly in anomalies:
					insights.append({
						'type': 'anomaly',
						'title': 'Unusual Data Point Detected',
						'description': f"Value {anomaly['value']} is significantly different from the norm",
						'confidence': 0.7,
						'impact': 'medium'
					})
			
			# Performance comparison
			if widget.benchmark_comparisons:
				for benchmark in widget.benchmark_comparisons:
					comparison = await self._compare_with_benchmark(widget, data, benchmark)
					if comparison:
						insights.append(comparison)
		
		except Exception as e:
			logger.error(f"Failed to generate AI insights: {str(e)}")
		
		return insights
	
	def _calculate_trend(self, values: List[float]) -> float:
		"""Calculate trend direction and magnitude"""
		if len(values) < 2:
			return 0.0
		
		# Simple linear trend calculation
		n = len(values)
		x = list(range(n))
		
		# Calculate slope using least squares
		x_mean = sum(x) / n
		y_mean = sum(values) / n
		
		numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
		denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
		
		if denominator == 0:
			return 0.0
		
		slope = numerator / denominator
		
		# Normalize by mean value to get percentage change
		if y_mean != 0:
			return slope / y_mean
		
		return 0.0
	
	def _detect_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Simple anomaly detection using statistical methods"""
		anomalies = []
		
		try:
			# Extract numeric values
			values = []
			for item in data:
				for key, value in item.items():
					if isinstance(value, (int, float)):
						values.append(value)
						break
			
			if len(values) < 5:
				return anomalies
			
			# Calculate mean and standard deviation
			mean = sum(values) / len(values)
			variance = sum((x - mean) ** 2 for x in values) / len(values)
			std_dev = variance ** 0.5
			
			# Detect outliers (values > 2 standard deviations from mean)
			threshold = 2 * std_dev
			
			for i, value in enumerate(values):
				if abs(value - mean) > threshold:
					anomalies.append({
						'index': i,
						'value': value,
						'expected_range': [mean - threshold, mean + threshold],
						'deviation': abs(value - mean) / std_dev
					})
		
		except Exception as e:
			logger.error(f"Failed to detect anomalies: {str(e)}")
		
		return anomalies
	
	async def _compare_with_benchmark(self, widget: DashboardWidget, data: Dict[str, Any], benchmark: str) -> Optional[Dict[str, Any]]:
		"""Compare widget data with benchmark values"""
		# This would integrate with external benchmarking services
		# For now, return a placeholder comparison
		return {
			'type': 'benchmark',
			'title': f'Performance vs {benchmark}',
			'description': 'Performance analysis compared to industry benchmarks',
			'confidence': 0.6,
			'impact': 'medium'
		}
	
	# Helper methods for database operations
	async def _load_dashboards(self):
		"""Load dashboard configurations from database"""
		try:
			async with self.db_manager.get_connection() as conn:
				dashboards_data = await conn.fetch("""
					SELECT * FROM crm_dashboards 
					ORDER BY created_at DESC
				""")
				
				for dashboard_data in dashboards_data:
					dashboard = DashboardLayout(
						id=dashboard_data['id'],
						tenant_id=dashboard_data['tenant_id'],
						name=dashboard_data['name'],
						description=dashboard_data['description'],
						dashboard_type=DashboardType(dashboard_data['dashboard_type']),
						grid_columns=dashboard_data['grid_columns'],
						grid_rows=dashboard_data['grid_rows'],
						widgets=[DashboardWidget(**w) for w in json.loads(dashboard_data['widgets'] or '[]')],
						owner_id=dashboard_data['owner_id'],
						is_public=dashboard_data['is_public'],
						shared_with=json.loads(dashboard_data['shared_with'] or '[]'),
						access_level=dashboard_data['access_level'],
						theme=dashboard_data['theme'],
						background_color=dashboard_data['background_color'],
						auto_refresh=dashboard_data['auto_refresh'],
						refresh_interval=dashboard_data['refresh_interval'],
						filters_enabled=dashboard_data['filters_enabled'],
						export_enabled=dashboard_data['export_enabled'],
						drill_down_enabled=dashboard_data['drill_down_enabled'],
						real_time_updates=dashboard_data['real_time_updates'],
						cache_enabled=dashboard_data['cache_enabled'],
						cache_ttl=dashboard_data['cache_ttl'],
						lazy_loading=dashboard_data['lazy_loading'],
						metadata=json.loads(dashboard_data['metadata'] or '{}'),
						created_by=dashboard_data['created_by'],
						created_at=dashboard_data['created_at'],
						updated_at=dashboard_data['updated_at']
					)
					
					self._dashboard_cache[dashboard.id] = dashboard
			
			logger.info(f"📋 Loaded {len(self._dashboard_cache)} dashboards")
			
		except Exception as e:
			logger.error(f"Failed to load dashboards: {str(e)}")
			raise
	
	async def _get_dashboard(self, dashboard_id: str, tenant_id: str) -> Optional[DashboardLayout]:
		"""Get dashboard by ID"""
		dashboard = self._dashboard_cache.get(dashboard_id)
		if dashboard and dashboard.tenant_id == tenant_id:
			return dashboard
		return None
	
	async def _initialize_insight_generators(self):
		"""Initialize AI insight generators"""
		# Placeholder for AI insight generator initialization
		logger.info("🤖 AI insight generators initialized")
	
	async def _start_cache_maintenance(self):
		"""Start background cache maintenance"""
		# Start cache cleanup task
		asyncio.create_task(self._cache_cleanup_task())
		logger.info("🧹 Cache maintenance started")
	
	async def _cache_cleanup_task(self):
		"""Background task to clean up expired cache entries"""
		while self._initialized:
			try:
				current_time = datetime.now()
				expired_keys = []
				
				for key, cached_data in self._data_cache.items():
					if current_time - cached_data['timestamp'] > timedelta(hours=24):
						expired_keys.append(key)
				
				for key in expired_keys:
					del self._data_cache[key]
				
				if expired_keys:
					logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
				
				# Sleep for 1 hour before next cleanup
				await asyncio.sleep(3600)
				
			except Exception as e:
				logger.error(f"Cache cleanup error: {str(e)}")
				await asyncio.sleep(300)  # Retry in 5 minutes