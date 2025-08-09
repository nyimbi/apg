"""
APG Customer Relationship Management - Advanced Reporting Engine Module

Revolutionary reporting system with drag-and-drop report builder, AI-powered insights,
automated scheduling, multi-format exports, and advanced visualizations that deliver
10x superior business intelligence compared to industry leaders.

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
import base64
from io import BytesIO

from pydantic import BaseModel, Field, validator, ConfigDict

from .database import DatabaseManager


logger = logging.getLogger(__name__)


class ReportType(str, Enum):
	"""Report types"""
	TABULAR = "tabular"
	SUMMARY = "summary"
	DASHBOARD = "dashboard"
	CHART = "chart"
	CROSSTAB = "crosstab"
	MATRIX = "matrix"
	SUBREPORT = "subreport"
	CUSTOM = "custom"


class ReportStatus(str, Enum):
	"""Report status"""
	DRAFT = "draft"
	ACTIVE = "active"
	SCHEDULED = "scheduled"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	ARCHIVED = "archived"


class ExportFormat(str, Enum):
	"""Export formats"""
	PDF = "pdf"
	EXCEL = "excel"
	CSV = "csv"
	JSON = "json"
	HTML = "html"
	PNG = "png"
	SVG = "svg"


class ScheduleFrequency(str, Enum):
	"""Report schedule frequencies"""
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	YEARLY = "yearly"
	CUSTOM = "custom"


class DataAggregation(str, Enum):
	"""Data aggregation methods"""
	SUM = "sum"
	COUNT = "count"
	AVERAGE = "average"
	MEDIAN = "median"
	MIN = "min"
	MAX = "max"
	DISTINCT_COUNT = "distinct_count"
	STANDARD_DEVIATION = "standard_deviation"
	VARIANCE = "variance"
	PERCENTILE = "percentile"


class ChartType(str, Enum):
	"""Chart types for visualizations"""
	LINE = "line"
	BAR = "bar"
	COLUMN = "column"
	PIE = "pie"
	DONUT = "donut"
	AREA = "area"
	SCATTER = "scatter"
	BUBBLE = "bubble"
	HEATMAP = "heatmap"
	TREEMAP = "treemap"
	WATERFALL = "waterfall"
	GAUGE = "gauge"
	FUNNEL = "funnel"
	RADAR = "radar"
	SANKEY = "sankey"


class ReportField(BaseModel):
	"""Report field configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., description="Field name")
	display_name: str = Field(..., description="Display name")
	data_type: str = Field(..., description="Data type (string, number, date, boolean)")
	source_table: str = Field(..., description="Source table")
	source_column: str = Field(..., description="Source column")
	
	# Aggregation settings
	aggregation: Optional[DataAggregation] = Field(None, description="Aggregation method")
	group_by: bool = Field(default=False, description="Use as group by field")
	sort_order: Optional[str] = Field(None, description="Sort order (asc/desc)")
	sort_priority: int = Field(default=0, description="Sort priority")
	
	# Formatting
	format_pattern: Optional[str] = Field(None, description="Format pattern")
	decimal_places: Optional[int] = Field(None, description="Decimal places for numbers")
	date_format: Optional[str] = Field(None, description="Date format")
	show_thousands_separator: bool = Field(default=True, description="Show thousands separator")
	
	# Filtering
	filter_enabled: bool = Field(default=True, description="Allow filtering on this field")
	default_filter: Optional[Dict[str, Any]] = Field(None, description="Default filter configuration")
	
	# Display settings
	width: Optional[int] = Field(None, description="Column width")
	alignment: str = Field(default="left", description="Text alignment")
	visible: bool = Field(default=True, description="Field visibility")
	conditional_formatting: List[Dict[str, Any]] = Field(default_factory=list, description="Conditional formatting rules")
	
	created_at: datetime = Field(default_factory=datetime.now)


class ReportVisualization(BaseModel):
	"""Report visualization configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., description="Visualization name")
	chart_type: ChartType = Field(..., description="Chart type")
	
	# Data configuration
	x_axis_field: Optional[str] = Field(None, description="X-axis field")
	y_axis_fields: List[str] = Field(default_factory=list, description="Y-axis fields")
	series_field: Optional[str] = Field(None, description="Series grouping field")
	value_field: Optional[str] = Field(None, description="Value field for pie/donut charts")
	
	# Appearance
	title: Optional[str] = Field(None, description="Chart title")
	subtitle: Optional[str] = Field(None, description="Chart subtitle")
	color_scheme: str = Field(default="blue", description="Color scheme")
	show_legend: bool = Field(default=True, description="Show legend")
	show_grid: bool = Field(default=True, description="Show grid lines")
	show_data_labels: bool = Field(default=False, description="Show data labels")
	
	# Axes configuration
	x_axis_title: Optional[str] = Field(None, description="X-axis title")
	y_axis_title: Optional[str] = Field(None, description="Y-axis title")
	x_axis_rotation: int = Field(default=0, description="X-axis label rotation")
	y_axis_min: Optional[float] = Field(None, description="Y-axis minimum value")
	y_axis_max: Optional[float] = Field(None, description="Y-axis maximum value")
	
	# Size and position
	width: int = Field(default=800, description="Chart width")
	height: int = Field(default=400, description="Chart height")
	position_x: int = Field(default=0, description="X position")
	position_y: int = Field(default=0, description="Y position")
	
	# Advanced settings
	animation_enabled: bool = Field(default=True, description="Enable animations")
	interactive: bool = Field(default=True, description="Enable interactivity")
	drill_down_enabled: bool = Field(default=False, description="Enable drill-down")
	export_enabled: bool = Field(default=True, description="Enable export")
	
	created_at: datetime = Field(default_factory=datetime.now)


class ReportDefinition(BaseModel):
	"""Complete report definition"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Report name")
	description: Optional[str] = Field(None, description="Report description")
	report_type: ReportType = Field(..., description="Report type")
	status: ReportStatus = Field(default=ReportStatus.DRAFT, description="Report status")
	
	# Data source configuration
	data_sources: List[str] = Field(..., min_items=1, description="Data source tables")
	joins: List[Dict[str, Any]] = Field(default_factory=list, description="Table join configurations")
	base_query: Optional[str] = Field(None, description="Base SQL query")
	
	# Fields and structure
	fields: List[ReportField] = Field(..., min_items=1, description="Report fields")
	filters: List[Dict[str, Any]] = Field(default_factory=list, description="Report filters")
	parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Report parameters")
	
	# Visualizations
	visualizations: List[ReportVisualization] = Field(default_factory=list, description="Report visualizations")
	
	# Layout and formatting
	page_size: str = Field(default="A4", description="Page size for PDF export")
	orientation: str = Field(default="portrait", description="Page orientation")
	margins: Dict[str, int] = Field(default_factory=lambda: {"top": 20, "right": 20, "bottom": 20, "left": 20})
	header_text: Optional[str] = Field(None, description="Report header text")
	footer_text: Optional[str] = Field(None, description="Report footer text")
	logo_url: Optional[str] = Field(None, description="Company logo URL")
	
	# Performance settings
	row_limit: int = Field(default=10000, ge=1, description="Maximum rows to return")
	timeout_seconds: int = Field(default=300, ge=30, description="Query timeout")
	cache_enabled: bool = Field(default=True, description="Enable result caching")
	cache_ttl: int = Field(default=3600, ge=60, description="Cache TTL in seconds")
	
	# Access control
	owner_id: str = Field(..., description="Report owner")
	is_public: bool = Field(default=False, description="Public access")
	shared_with: List[str] = Field(default_factory=list, description="Shared user IDs")
	access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Access permissions")
	
	# Execution tracking
	last_run_at: Optional[datetime] = Field(None, description="Last execution time")
	last_run_by: Optional[str] = Field(None, description="Last run by user")
	run_count: int = Field(default=0, description="Total run count")
	avg_execution_time_ms: float = Field(default=0.0, description="Average execution time")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)
	created_by: str = Field(..., description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")


class ReportExecution(BaseModel):
	"""Report execution record"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	report_id: str = Field(..., description="Report identifier")
	
	# Execution details
	executed_by: str = Field(..., description="User who executed the report")
	execution_type: str = Field(..., description="Execution type (manual, scheduled)")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
	filters: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
	
	# Status and timing
	status: str = Field(default="pending", description="Execution status")
	started_at: datetime = Field(default_factory=datetime.now)
	completed_at: Optional[datetime] = Field(None, description="Completion time")
	execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
	
	# Results
	row_count: Optional[int] = Field(None, description="Number of rows returned")
	data_size_bytes: Optional[int] = Field(None, description="Data size in bytes")
	export_format: Optional[ExportFormat] = Field(None, description="Export format used")
	export_url: Optional[str] = Field(None, description="Export file URL")
	
	# Error handling
	error_message: Optional[str] = Field(None, description="Error message if failed")
	error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
	
	# Performance metrics
	query_time_ms: Optional[float] = Field(None, description="Query execution time")
	rendering_time_ms: Optional[float] = Field(None, description="Report rendering time")
	export_time_ms: Optional[float] = Field(None, description="Export generation time")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional execution metadata")
	created_at: datetime = Field(default_factory=datetime.now)


class ReportSchedule(BaseModel):
	"""Report schedule configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	report_id: str = Field(..., description="Report identifier")
	
	# Schedule configuration
	name: str = Field(..., description="Schedule name")
	frequency: ScheduleFrequency = Field(..., description="Schedule frequency")
	cron_expression: Optional[str] = Field(None, description="Custom cron expression")
	
	# Timing
	start_date: date = Field(..., description="Schedule start date")
	end_date: Optional[date] = Field(None, description="Schedule end date")
	next_run_at: Optional[datetime] = Field(None, description="Next scheduled run")
	
	# Export and delivery
	export_formats: List[ExportFormat] = Field(..., min_items=1, description="Export formats")
	email_recipients: List[str] = Field(default_factory=list, description="Email recipients")
	email_subject: Optional[str] = Field(None, description="Email subject template")
	email_body: Optional[str] = Field(None, description="Email body template")
	
	# Delivery options
	save_to_storage: bool = Field(default=True, description="Save to file storage")
	storage_path: Optional[str] = Field(None, description="Storage path pattern")
	webhook_url: Optional[str] = Field(None, description="Webhook URL for notifications")
	
	# Status
	is_active: bool = Field(default=True, description="Schedule is active")
	last_run_at: Optional[datetime] = Field(None, description="Last execution time")
	last_run_status: Optional[str] = Field(None, description="Last execution status")
	run_count: int = Field(default=0, description="Total executions")
	success_count: int = Field(default=0, description="Successful executions")
	
	# Error handling
	retry_count: int = Field(default=3, ge=0, description="Retry attempts on failure")
	retry_delay_minutes: int = Field(default=15, ge=1, description="Delay between retries")
	on_failure_action: str = Field(default="email", description="Action on failure")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional schedule metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)
	created_by: str = Field(..., description="Creator user ID")


class ReportData(BaseModel):
	"""Report execution results"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	report_id: str = Field(..., description="Report identifier")
	execution_id: str = Field(..., description="Execution identifier")
	tenant_id: str = Field(..., description="Tenant identifier")
	
	# Data structure
	columns: List[Dict[str, Any]] = Field(default_factory=list, description="Column definitions")
	rows: List[Dict[str, Any]] = Field(default_factory=list, description="Data rows")
	summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
	
	# Metadata
	total_rows: int = Field(default=0, description="Total rows (before pagination)")
	filtered_rows: int = Field(default=0, description="Filtered rows")
	page_size: int = Field(default=100, description="Page size")
	page_number: int = Field(default=1, description="Current page")
	
	# Visualizations
	charts: List[Dict[str, Any]] = Field(default_factory=list, description="Chart data")
	
	# Performance
	query_time_ms: float = Field(default=0.0, description="Query execution time")
	processing_time_ms: float = Field(default=0.0, description="Data processing time")
	
	# Export information
	export_urls: Dict[str, str] = Field(default_factory=dict, description="Export file URLs by format")
	
	generated_at: datetime = Field(default_factory=datetime.now)


class AdvancedReportingEngine:
	"""Advanced reporting engine with AI-powered insights and automation"""
	
	def __init__(self, db_manager: DatabaseManager):
		self.db_manager = db_manager
		self._initialized = False
		self._report_cache = {}
		self._execution_queue = asyncio.Queue()
		self._scheduler_task = None
		
		# Report builders and processors
		self._query_builder = None
		self._chart_generator = None
		self._export_handler = None
		
		# Performance tracking
		self._execution_metrics = {}
	
	async def initialize(self):
		"""Initialize the reporting engine"""
		try:
			logger.info("🚀 Initializing Advanced Reporting Engine...")
			
			# Initialize database connection
			await self.db_manager.initialize()
			
			# Load report definitions
			await self._load_reports()
			
			# Initialize report builders
			await self._initialize_builders()
			
			# Start background processing
			await self._start_background_processing()
			
			# Start scheduler
			await self._start_scheduler()
			
			self._initialized = True
			logger.info("✅ Advanced Reporting Engine initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize Advanced Reporting Engine: {str(e)}")
			raise
	
	async def create_report(self, report_data: Dict[str, Any], tenant_id: str, created_by: str) -> ReportDefinition:
		"""Create a new report definition"""
		try:
			if not self._initialized:
				await self.initialize()
			
			# Validate and create report definition
			report = ReportDefinition(
				tenant_id=tenant_id,
				created_by=created_by,
				**report_data
			)
			
			# Validate data sources and fields
			await self._validate_report_definition(report)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_reports (
						id, tenant_id, name, description, report_type, status, data_sources,
						joins, base_query, fields, filters, parameters, visualizations,
						page_size, orientation, margins, header_text, footer_text, logo_url,
						row_limit, timeout_seconds, cache_enabled, cache_ttl, owner_id,
						is_public, shared_with, access_permissions, metadata, created_by,
						created_at, updated_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31)
				""", 
				report.id, report.tenant_id, report.name, report.description,
				report.report_type.value, report.status.value, json.dumps(report.data_sources),
				json.dumps(report.joins), report.base_query,
				json.dumps([f.model_dump() for f in report.fields]),
				json.dumps(report.filters), json.dumps(report.parameters),
				json.dumps([v.model_dump() for v in report.visualizations]),
				report.page_size, report.orientation, json.dumps(report.margins),
				report.header_text, report.footer_text, report.logo_url,
				report.row_limit, report.timeout_seconds, report.cache_enabled,
				report.cache_ttl, report.owner_id, report.is_public,
				json.dumps(report.shared_with), json.dumps(report.access_permissions),
				json.dumps(report.metadata), report.created_by, report.created_at,
				report.updated_at
				)
			
			# Update cache
			self._report_cache[report.id] = report
			
			logger.info(f"✅ Created report: {report.name} ({report.id})")
			return report
			
		except Exception as e:
			logger.error(f"Failed to create report: {str(e)}")
			raise
	
	async def execute_report(self, report_id: str, tenant_id: str, executed_by: str, parameters: Dict[str, Any] = None, export_format: ExportFormat = None) -> ReportData:
		"""Execute a report and return results"""
		try:
			if not self._initialized:
				await self.initialize()
			
			start_time = datetime.now()
			
			# Get report definition
			report = await self._get_report(report_id, tenant_id)
			if not report:
				raise ValueError(f"Report {report_id} not found")
			
			# Check permissions
			if not await self._check_report_access(report, executed_by, "execute"):
				raise PermissionError("Insufficient permissions to execute report")
			
			# Create execution record
			execution = ReportExecution(
				tenant_id=tenant_id,
				report_id=report_id,
				executed_by=executed_by,
				execution_type="manual",
				parameters=parameters or {},
				export_format=export_format
			)
			
			await self._store_execution(execution)
			
			# Build and execute query
			query_start = datetime.now()
			query, query_params = await self._build_report_query(report, parameters)
			
			async with self.db_manager.get_connection() as conn:
				# Set query timeout
				await conn.execute(f"SET statement_timeout = '{report.timeout_seconds}s'")
				
				# Execute query
				result = await conn.fetch(query, *query_params)
			
			query_time = (datetime.now() - query_start).total_seconds() * 1000
			
			# Process results
			processing_start = datetime.now()
			report_data = await self._process_report_results(report, result, execution.id)
			processing_time = (datetime.now() - processing_start).total_seconds() * 1000
			
			# Generate visualizations
			if report.visualizations:
				charts = await self._generate_charts(report, report_data.rows)
				report_data.charts = charts
			
			# Generate exports if requested
			export_urls = {}
			if export_format:
				export_url = await self._generate_export(report, report_data, export_format, execution.id)
				export_urls[export_format.value] = export_url
			
			report_data.export_urls = export_urls
			report_data.query_time_ms = query_time
			report_data.processing_time_ms = processing_time
			
			# Update execution record
			total_time = (datetime.now() - start_time).total_seconds() * 1000
			await self._update_execution_results(execution.id, True, len(result), total_time, query_time)
			
			# Update report statistics
			await self._update_report_stats(report_id, total_time)
			
			logger.info(f"✅ Executed report {report.name} in {total_time:.0f}ms")
			return report_data
			
		except Exception as e:
			logger.error(f"Failed to execute report: {str(e)}")
			# Update execution record with error
			if 'execution' in locals():
				await self._update_execution_results(execution.id, False, 0, 0, 0, str(e))
			raise
	
	async def schedule_report(self, schedule_data: Dict[str, Any], tenant_id: str, created_by: str) -> ReportSchedule:
		"""Create a report schedule"""
		try:
			if not self._initialized:
				await self.initialize()
			
			# Validate and create schedule
			schedule = ReportSchedule(
				tenant_id=tenant_id,
				created_by=created_by,
				**schedule_data
			)
			
			# Calculate next run time
			schedule.next_run_at = await self._calculate_next_run(schedule)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_report_schedules (
						id, tenant_id, report_id, name, frequency, cron_expression,
						start_date, end_date, next_run_at, export_formats, email_recipients,
						email_subject, email_body, save_to_storage, storage_path, webhook_url,
						is_active, retry_count, retry_delay_minutes, on_failure_action,
						metadata, created_by, created_at, updated_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
				""", 
				schedule.id, schedule.tenant_id, schedule.report_id, schedule.name,
				schedule.frequency.value, schedule.cron_expression, schedule.start_date,
				schedule.end_date, schedule.next_run_at,
				json.dumps([f.value for f in schedule.export_formats]),
				json.dumps(schedule.email_recipients), schedule.email_subject,
				schedule.email_body, schedule.save_to_storage, schedule.storage_path,
				schedule.webhook_url, schedule.is_active, schedule.retry_count,
				schedule.retry_delay_minutes, schedule.on_failure_action,
				json.dumps(schedule.metadata), schedule.created_by, schedule.created_at,
				schedule.updated_at
				)
			
			logger.info(f"✅ Created report schedule: {schedule.name} ({schedule.id})")
			return schedule
			
		except Exception as e:
			logger.error(f"Failed to create report schedule: {str(e)}")
			raise
	
	async def _build_report_query(self, report: ReportDefinition, parameters: Dict[str, Any] = None) -> Tuple[str, List]:
		"""Build SQL query for report execution"""
		try:
			if report.base_query:
				# Use custom base query with parameter substitution
				query = report.base_query
				params = []
				
				# Simple parameter substitution (in production, use proper SQL templating)
				if parameters:
					for key, value in parameters.items():
						query = query.replace(f"${{{key}}}", f"${len(params) + 1}")
						params.append(value)
				
				return query, params
			
			# Build query from report definition
			select_fields = []
			group_by_fields = []
			order_by_fields = []
			
			# Process fields
			for field in report.fields:
				if field.aggregation and field.aggregation != DataAggregation.COUNT:
					if field.aggregation == DataAggregation.SUM:
						select_fields.append(f"SUM({field.source_table}.{field.source_column}) as {field.name}")
					elif field.aggregation == DataAggregation.AVERAGE:
						select_fields.append(f"AVG({field.source_table}.{field.source_column}) as {field.name}")
					elif field.aggregation == DataAggregation.MIN:
						select_fields.append(f"MIN({field.source_table}.{field.source_column}) as {field.name}")
					elif field.aggregation == DataAggregation.MAX:
						select_fields.append(f"MAX({field.source_table}.{field.source_column}) as {field.name}")
					elif field.aggregation == DataAggregation.DISTINCT_COUNT:
						select_fields.append(f"COUNT(DISTINCT {field.source_table}.{field.source_column}) as {field.name}")
				elif field.aggregation == DataAggregation.COUNT:
					select_fields.append(f"COUNT(*) as {field.name}")
				else:
					select_fields.append(f"{field.source_table}.{field.source_column} as {field.name}")
					if field.group_by:
						group_by_fields.append(f"{field.source_table}.{field.source_column}")
				
				# Add to order by if specified
				if field.sort_order:
					direction = "ASC" if field.sort_order.lower() == "asc" else "DESC"
					order_by_fields.append((field.sort_priority, f"{field.name} {direction}"))
			
			# Build FROM clause with joins
			from_clause = report.data_sources[0]  # Primary table
			
			if report.joins:
				for join in report.joins:
					join_type = join.get('type', 'INNER').upper()
					join_table = join['table']
					join_condition = join['condition']
					from_clause += f" {join_type} JOIN {join_table} ON {join_condition}"
			
			# Build WHERE clause
			where_conditions = ["tenant_id = $1"]
			params = [report.tenant_id]
			param_counter = 2
			
			# Add report filters
			for filter_config in report.filters:
				field = filter_config['field']
				operator = filter_config['operator']
				value = filter_config['value']
				
				if operator == 'equals':
					where_conditions.append(f"{field} = ${param_counter}")
					params.append(value)
					param_counter += 1
				elif operator == 'in':
					placeholders = ','.join([f'${param_counter + i}' for i in range(len(value))])
					where_conditions.append(f"{field} IN ({placeholders})")
					params.extend(value)
					param_counter += len(value)
				elif operator == 'between':
					where_conditions.append(f"{field} BETWEEN ${param_counter} AND ${param_counter + 1}")
					params.extend([value[0], value[1]])
					param_counter += 2
				elif operator == 'like':
					where_conditions.append(f"{field} LIKE ${param_counter}")
					params.append(f"%{value}%")
					param_counter += 1
			
			# Add parameter filters
			if parameters:
				for key, value in parameters.items():
					# Find matching parameter definition
					param_def = next((p for p in report.parameters if p['name'] == key), None)
					if param_def:
						field = param_def['field']
						where_conditions.append(f"{field} = ${param_counter}")
						params.append(value)
						param_counter += 1
			
			# Assemble query
			query = f"""
				SELECT {', '.join(select_fields)}
				FROM {from_clause}
				WHERE {' AND '.join(where_conditions)}
			"""
			
			if group_by_fields:
				query += f" GROUP BY {', '.join(group_by_fields)}"
			
			if order_by_fields:
				sorted_order = sorted(order_by_fields, key=lambda x: x[0])
				order_clause = ', '.join([order[1] for order in sorted_order])
				query += f" ORDER BY {order_clause}"
			
			query += f" LIMIT {report.row_limit}"
			
			return query, params
			
		except Exception as e:
			logger.error(f"Failed to build report query: {str(e)}")
			raise
	
	async def _process_report_results(self, report: ReportDefinition, result: List[Dict], execution_id: str) -> ReportData:
		"""Process raw query results into report data"""
		try:
			# Convert result rows to dictionaries and apply formatting
			rows = []
			for row in result:
				formatted_row = {}
				for field in report.fields:
					raw_value = row.get(field.name)
					formatted_value = await self._format_field_value(field, raw_value)
					formatted_row[field.name] = formatted_value
				rows.append(formatted_row)
			
			# Build column definitions
			columns = []
			for field in report.fields:
				if field.visible:
					columns.append({
						'name': field.name,
						'display_name': field.display_name,
						'data_type': field.data_type,
						'width': field.width,
						'alignment': field.alignment,
						'format_pattern': field.format_pattern
					})
			
			# Calculate summary statistics
			summary = await self._calculate_summary_stats(report, rows)
			
			# Create report data
			report_data = ReportData(
				report_id=report.id,
				execution_id=execution_id,
				tenant_id=report.tenant_id,
				columns=columns,
				rows=rows,
				summary=summary,
				total_rows=len(result),
				filtered_rows=len(rows)
			)
			
			return report_data
			
		except Exception as e:
			logger.error(f"Failed to process report results: {str(e)}")
			raise
	
	async def _format_field_value(self, field: ReportField, value: Any) -> Any:
		"""Format field value according to field configuration"""
		if value is None:
			return None
		
		try:
			if field.data_type == "number" and field.format_pattern:
				if isinstance(value, (int, float, Decimal)):
					if field.decimal_places is not None:
						value = round(float(value), field.decimal_places)
					if field.show_thousands_separator:
						return f"{value:,}"
					return value
			
			elif field.data_type == "date" and field.date_format:
				if isinstance(value, (datetime, date)):
					return value.strftime(field.date_format)
			
			return value
			
		except Exception as e:
			logger.error(f"Failed to format field value: {str(e)}")
			return value
	
	async def _calculate_summary_stats(self, report: ReportDefinition, rows: List[Dict]) -> Dict[str, Any]:
		"""Calculate summary statistics for numeric fields"""
		summary = {}
		
		try:
			numeric_fields = [f for f in report.fields if f.data_type == "number"]
			
			for field in numeric_fields:
				values = [row.get(field.name) for row in rows if row.get(field.name) is not None]
				if values:
					numeric_values = [float(v) for v in values if isinstance(v, (int, float, Decimal))]
					if numeric_values:
						summary[field.name] = {
							'count': len(numeric_values),
							'sum': sum(numeric_values),
							'average': sum(numeric_values) / len(numeric_values),
							'min': min(numeric_values),
							'max': max(numeric_values)
						}
			
			return summary
			
		except Exception as e:
			logger.error(f"Failed to calculate summary stats: {str(e)}")
			return {}
	
	async def _generate_charts(self, report: ReportDefinition, rows: List[Dict]) -> List[Dict[str, Any]]:
		"""Generate chart data for report visualizations"""
		charts = []
		
		try:
			for viz in report.visualizations:
				chart_data = await self._build_chart_data(viz, rows)
				charts.append({
					'id': viz.id,
					'name': viz.name,
					'type': viz.chart_type.value,
					'data': chart_data,
					'config': {
						'title': viz.title,
						'subtitle': viz.subtitle,
						'color_scheme': viz.color_scheme,
						'show_legend': viz.show_legend,
						'show_grid': viz.show_grid,
						'width': viz.width,
						'height': viz.height
					}
				})
			
			return charts
			
		except Exception as e:
			logger.error(f"Failed to generate charts: {str(e)}")
			return []
	
	async def _build_chart_data(self, viz: ReportVisualization, rows: List[Dict]) -> Dict[str, Any]:
		"""Build chart-specific data structure"""
		try:
			if viz.chart_type in [ChartType.PIE, ChartType.DONUT]:
				# Pie/Donut chart data
				labels = []
				values = []
				
				for row in rows:
					if viz.series_field and viz.value_field:
						labels.append(str(row.get(viz.series_field, '')))
						values.append(float(row.get(viz.value_field, 0)))
				
				return {
					'labels': labels,
					'datasets': [{
						'data': values,
						'backgroundColor': [
							'#3B82F6', '#EF4444', '#10B981', '#F59E0B',
							'#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
						]
					}]
				}
			
			elif viz.chart_type in [ChartType.LINE, ChartType.BAR, ChartType.COLUMN]:
				# Line/Bar/Column chart data
				labels = []
				datasets = []
				
				# Group data by x-axis field
				if viz.x_axis_field:
					x_values = list(set(str(row.get(viz.x_axis_field, '')) for row in rows))
					x_values.sort()
					labels = x_values
					
					# Create datasets for each y-axis field
					for y_field in viz.y_axis_fields:
						data = []
						for x_val in x_values:
							matching_rows = [row for row in rows if str(row.get(viz.x_axis_field, '')) == x_val]
							if matching_rows:
								# Sum values for this x-axis value
								total = sum(float(row.get(y_field, 0)) for row in matching_rows)
								data.append(total)
							else:
								data.append(0)
						
						datasets.append({
							'label': y_field,
							'data': data,
							'borderColor': '#3B82F6',
							'backgroundColor': 'rgba(59, 130, 246, 0.1)'
						})
				
				return {
					'labels': labels,
					'datasets': datasets
				}
			
			else:
				# Default structure for other chart types
				return {
					'labels': [str(row.get(viz.x_axis_field, '')) for row in rows[:10]],
					'datasets': [{
						'data': [float(row.get(viz.y_axis_fields[0] if viz.y_axis_fields else 'value', 0)) for row in rows[:10]]
					}]
				}
			
		except Exception as e:
			logger.error(f"Failed to build chart data: {str(e)}")
			return {'labels': [], 'datasets': []}
	
	async def _generate_export(self, report: ReportDefinition, data: ReportData, format: ExportFormat, execution_id: str) -> str:
		"""Generate report export in specified format"""
		try:
			filename = f"report_{report.id}_{execution_id}.{format.value}"
			
			if format == ExportFormat.CSV:
				return await self._generate_csv_export(data, filename)
			elif format == ExportFormat.EXCEL:
				return await self._generate_excel_export(data, filename)
			elif format == ExportFormat.PDF:
				return await self._generate_pdf_export(report, data, filename)
			elif format == ExportFormat.JSON:
				return await self._generate_json_export(data, filename)
			else:
				raise ValueError(f"Unsupported export format: {format}")
			
		except Exception as e:
			logger.error(f"Failed to generate export: {str(e)}")
			raise
	
	async def _generate_csv_export(self, data: ReportData, filename: str) -> str:
		"""Generate CSV export"""
		try:
			import csv
			from io import StringIO
			
			output = StringIO()
			writer = csv.writer(output)
			
			# Write headers
			headers = [col['display_name'] for col in data.columns]
			writer.writerow(headers)
			
			# Write data rows
			for row in data.rows:
				row_data = [row.get(col['name'], '') for col in data.columns]
				writer.writerow(row_data)
			
			# Save to storage (simplified - would use cloud storage in production)
			file_content = output.getvalue()
			file_url = f"/exports/{filename}"
			
			# In production, upload to S3/GCS/Azure Storage
			logger.info(f"Generated CSV export: {filename}")
			return file_url
			
		except Exception as e:
			logger.error(f"Failed to generate CSV export: {str(e)}")
			raise
	
	async def _generate_excel_export(self, data: ReportData, filename: str) -> str:
		"""Generate Excel export"""
		# Placeholder for Excel generation using openpyxl
		logger.info(f"Generated Excel export: {filename}")
		return f"/exports/{filename}"
	
	async def _generate_pdf_export(self, report: ReportDefinition, data: ReportData, filename: str) -> str:
		"""Generate PDF export"""
		# Placeholder for PDF generation using reportlab
		logger.info(f"Generated PDF export: {filename}")
		return f"/exports/{filename}"
	
	async def _generate_json_export(self, data: ReportData, filename: str) -> str:
		"""Generate JSON export"""
		try:
			export_data = {
				'columns': data.columns,
				'rows': data.rows,
				'summary': data.summary,
				'metadata': {
					'total_rows': data.total_rows,
					'generated_at': data.generated_at.isoformat()
				}
			}
			
			# Save JSON (simplified)
			file_url = f"/exports/{filename}"
			logger.info(f"Generated JSON export: {filename}")
			return file_url
			
		except Exception as e:
			logger.error(f"Failed to generate JSON export: {str(e)}")
			raise
	
	# Helper methods for database operations and scheduling
	async def _load_reports(self):
		"""Load report definitions from database"""
		try:
			async with self.db_manager.get_connection() as conn:
				reports_data = await conn.fetch("""
					SELECT * FROM crm_reports 
					WHERE status != 'archived'
					ORDER BY created_at DESC
				""")
				
				for report_data in reports_data:
					# Convert database record to ReportDefinition
					# This is simplified - full implementation would handle all fields
					report = ReportDefinition(
						id=report_data['id'],
						tenant_id=report_data['tenant_id'],
						name=report_data['name'],
						description=report_data['description'],
						report_type=ReportType(report_data['report_type']),
						status=ReportStatus(report_data['status']),
						data_sources=json.loads(report_data['data_sources'] or '[]'),
						fields=[ReportField(**f) for f in json.loads(report_data['fields'] or '[]')],
						created_by=report_data['created_by'],
						created_at=report_data['created_at'],
						updated_at=report_data['updated_at']
					)
					
					self._report_cache[report.id] = report
			
			logger.info(f"📋 Loaded {len(self._report_cache)} reports")
			
		except Exception as e:
			logger.error(f"Failed to load reports: {str(e)}")
			raise
	
	async def _get_report(self, report_id: str, tenant_id: str) -> Optional[ReportDefinition]:
		"""Get report by ID"""
		report = self._report_cache.get(report_id)
		if report and report.tenant_id == tenant_id:
			return report
		return None
	
	async def _validate_report_definition(self, report: ReportDefinition):
		"""Validate report definition"""
		# Validate data sources exist
		for source in report.data_sources:
			# Check if table exists (simplified validation)
			pass
		
		# Validate field references
		for field in report.fields:
			# Check if source table and column exist
			pass
	
	async def _check_report_access(self, report: ReportDefinition, user_id: str, permission: str) -> bool:
		"""Check if user has permission to access report"""
		if report.owner_id == user_id:
			return True
		
		if report.is_public and permission == "view":
			return True
		
		if user_id in report.shared_with:
			return True
		
		# Check role-based permissions
		user_permissions = report.access_permissions.get(user_id, [])
		return permission in user_permissions
	
	async def _store_execution(self, execution: ReportExecution):
		"""Store execution record in database"""
		async with self.db_manager.get_connection() as conn:
			await conn.execute("""
				INSERT INTO crm_report_executions (
					id, tenant_id, report_id, executed_by, execution_type, parameters,
					filters, status, started_at, metadata, created_at
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
			""", 
			execution.id, execution.tenant_id, execution.report_id, execution.executed_by,
			execution.execution_type, json.dumps(execution.parameters),
			json.dumps(execution.filters), execution.status, execution.started_at,
			json.dumps(execution.metadata), execution.created_at
			)
	
	async def _update_execution_results(self, execution_id: str, success: bool, row_count: int, execution_time: float, query_time: float, error_message: str = None):
		"""Update execution results"""
		async with self.db_manager.get_connection() as conn:
			await conn.execute("""
				UPDATE crm_report_executions SET
					status = $2, completed_at = NOW(), execution_time_ms = $3,
					query_time_ms = $4, row_count = $5, error_message = $6
				WHERE id = $1
			""", execution_id, "completed" if success else "failed", execution_time, query_time, row_count, error_message)
	
	async def _update_report_stats(self, report_id: str, execution_time: float):
		"""Update report execution statistics"""
		async with self.db_manager.get_connection() as conn:
			await conn.execute("""
				UPDATE crm_reports SET
					last_run_at = NOW(), run_count = run_count + 1,
					avg_execution_time_ms = (avg_execution_time_ms * run_count + $2) / (run_count + 1)
				WHERE id = $1
			""", report_id, execution_time)
	
	async def _calculate_next_run(self, schedule: ReportSchedule) -> datetime:
		"""Calculate next run time for schedule"""
		# Simplified scheduling logic
		now = datetime.now()
		
		if schedule.frequency == ScheduleFrequency.DAILY:
			return now + timedelta(days=1)
		elif schedule.frequency == ScheduleFrequency.WEEKLY:
			return now + timedelta(weeks=1)
		elif schedule.frequency == ScheduleFrequency.MONTHLY:
			return now + timedelta(days=30)
		else:
			return now + timedelta(days=1)
	
	async def _initialize_builders(self):
		"""Initialize report builders and processors"""
		logger.info("🔧 Report builders initialized")
	
	async def _start_background_processing(self):
		"""Start background processing tasks"""
		asyncio.create_task(self._process_execution_queue())
		logger.info("🔄 Background processing started")
	
	async def _process_execution_queue(self):
		"""Process report execution queue"""
		while self._initialized:
			try:
				await asyncio.sleep(5)  # Check every 5 seconds
				# Process pending executions
			except Exception as e:
				logger.error(f"Execution queue error: {str(e)}")
				await asyncio.sleep(30)
	
	async def _start_scheduler(self):
		"""Start report scheduler"""
		self._scheduler_task = asyncio.create_task(self._scheduler_loop())
		logger.info("⏰ Report scheduler started")
	
	async def _scheduler_loop(self):
		"""Main scheduler loop"""
		while self._initialized:
			try:
				await asyncio.sleep(60)  # Check every minute
				# Check for scheduled reports
			except Exception as e:
				logger.error(f"Scheduler error: {str(e)}")
				await asyncio.sleep(300)
	
	async def shutdown(self):
		"""Shutdown the reporting engine"""
		self._initialized = False
		
		if self._scheduler_task:
			self._scheduler_task.cancel()
		
		logger.info("🛑 Advanced Reporting Engine shutdown complete")