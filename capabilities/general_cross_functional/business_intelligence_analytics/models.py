"""
Business Intelligence Analytics - Comprehensive Pydantic Models

Enterprise-grade business intelligence, OLAP, dimensional modeling, and executive
dashboards supporting strategic decision-making across all APG capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.config import ConfigDict
from pydantic.types import EmailStr, HttpUrl, Json
from uuid_extensions import uuid7str


class ConfigDict(ConfigDict):
	extra = 'forbid'
	validate_by_name = True
	validate_by_alias = True


# Base BI Model
class BIAnalyticsBase(BaseModel):
	model_config = ConfigDict()
	
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="Multi-tenant organization identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User ID who created the record")
	updated_by: str = Field(..., description="User ID who last updated the record")
	is_active: bool = Field(default=True, description="Active status flag")
	metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


# Enumeration Types
class BIDimensionType(str, Enum):
	TIME = "time"
	GEOGRAPHY = "geography"
	PRODUCT = "product"
	CUSTOMER = "customer"
	EMPLOYEE = "employee"
	ORGANIZATION = "organization"
	ACCOUNT = "account"
	CHANNEL = "channel"
	CAMPAIGN = "campaign"
	CUSTOM = "custom"


class BIHierarchyType(str, Enum):
	BALANCED = "balanced"
	UNBALANCED = "unbalanced"
	RAGGED = "ragged"
	NETWORK = "network"
	PARENT_CHILD = "parent_child"


class BIMeasureType(str, Enum):
	ADDITIVE = "additive"
	SEMI_ADDITIVE = "semi_additive"
	NON_ADDITIVE = "non_additive"
	CALCULATED = "calculated"
	RATIO = "ratio"
	PERCENTAGE = "percentage"


class BIAggregationType(str, Enum):
	SUM = "sum"
	COUNT = "count"
	AVERAGE = "average"
	MIN = "min"
	MAX = "max"
	MEDIAN = "median"
	DISTINCT_COUNT = "distinct_count"
	VARIANCE = "variance"
	STANDARD_DEVIATION = "standard_deviation"
	CUSTOM = "custom"


class BIChartType(str, Enum):
	LINE_CHART = "line_chart"
	BAR_CHART = "bar_chart"
	COLUMN_CHART = "column_chart"
	PIE_CHART = "pie_chart"
	DONUT_CHART = "donut_chart"
	AREA_CHART = "area_chart"
	SCATTER_PLOT = "scatter_plot"
	BUBBLE_CHART = "bubble_chart"
	HEATMAP = "heatmap"
	TREEMAP = "treemap"
	WATERFALL = "waterfall"
	GAUGE = "gauge"
	FUNNEL = "funnel"
	SANKEY = "sankey"
	GANTT = "gantt"


class BIDashboardType(str, Enum):
	EXECUTIVE = "executive"
	OPERATIONAL = "operational"
	ANALYTICAL = "analytical"
	TACTICAL = "tactical"
	SCORECARD = "scorecard"
	REALTIME = "realtime"
	MOBILE = "mobile"
	CUSTOM = "custom"


class BIReportFormat(str, Enum):
	PDF = "pdf"
	EXCEL = "excel"
	POWERPOINT = "powerpoint"
	WORD = "word"
	CSV = "csv"
	JSON = "json"
	XML = "xml"
	HTML = "html"


class BIProcessingStatus(str, Enum):
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	SCHEDULED = "scheduled"


class BISCDType(str, Enum):
	TYPE_1 = "type_1"  # Overwrite
	TYPE_2 = "type_2"  # Add new record
	TYPE_3 = "type_3"  # Add new attribute
	TYPE_6 = "type_6"  # Hybrid approach


# Data Warehouse Models
class BIDataWarehouse(BIAnalyticsBase):
	"""Enterprise data warehouse configuration"""
	name: str = Field(..., description="Data warehouse name")
	description: Optional[str] = Field(default=None, description="Data warehouse description")
	connection_string: str = Field(..., description="Database connection string")
	schema_name: str = Field(default="dw", description="Database schema name")
	storage_format: str = Field(default="columnar", description="Storage format optimization")
	compression_type: str = Field(default="snappy", description="Data compression type")
	partitioning_strategy: Dict[str, Any] = Field(default_factory=dict, description="Partitioning configuration")
	indexing_strategy: Dict[str, Any] = Field(default_factory=dict, description="Indexing configuration")
	backup_config: Dict[str, Any] = Field(default_factory=dict, description="Backup configuration")
	security_config: Dict[str, Any] = Field(default_factory=dict, description="Security settings")
	performance_config: Dict[str, Any] = Field(default_factory=dict, description="Performance tuning")
	monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring settings")
	data_retention_days: int = Field(default=2555, description="Data retention period (7 years)")
	storage_size_gb: Optional[float] = Field(default=None, description="Current storage size in GB")


class BIDimension(BIAnalyticsBase):
	"""Dimensional model dimension definition"""
	warehouse_id: str = Field(..., description="Data warehouse ID")
	name: str = Field(..., description="Dimension name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Dimension description")
	dimension_type: BIDimensionType = Field(..., description="Type of dimension")
	table_name: str = Field(..., description="Physical table name")
	primary_key: str = Field(..., description="Primary key column")
	natural_key: str = Field(..., description="Business/natural key column")
	scd_type: BISCDType = Field(default=BISCDType.TYPE_2, description="Slowly changing dimension type")
	hierarchy_config: Optional[Dict[str, Any]] = Field(default=None, description="Hierarchy configuration")
	attributes: List[Dict[str, Any]] = Field(default_factory=list, description="Dimension attributes")
	sort_order: int = Field(default=0, description="Display sort order")
	is_conformed: bool = Field(default=False, description="Is this a conformed dimension")
	business_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Business rules")
	data_quality_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Data quality rules")
	refresh_frequency: str = Field(default="daily", description="Data refresh frequency")


class BIHierarchy(BIAnalyticsBase):
	"""Dimension hierarchy definition"""
	dimension_id: str = Field(..., description="Parent dimension ID")
	name: str = Field(..., description="Hierarchy name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Hierarchy description")
	hierarchy_type: BIHierarchyType = Field(..., description="Type of hierarchy")
	levels: List[Dict[str, Any]] = Field(..., description="Hierarchy levels configuration")
	default_member: Optional[str] = Field(default=None, description="Default hierarchy member")
	all_member_name: str = Field(default="All", description="All member display name")
	sort_order: int = Field(default=0, description="Display sort order")
	is_enabled: bool = Field(default=True, description="Hierarchy enabled status")
	navigation_config: Dict[str, Any] = Field(default_factory=dict, description="Navigation settings")
	security_config: Optional[Dict[str, Any]] = Field(default=None, description="Hierarchy security")


class BIFactTable(BIAnalyticsBase):
	"""Fact table definition"""
	warehouse_id: str = Field(..., description="Data warehouse ID")
	name: str = Field(..., description="Fact table name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Fact table description")
	table_name: str = Field(..., description="Physical table name")
	grain_description: str = Field(..., description="Fact table grain description")
	dimension_keys: List[str] = Field(..., description="Foreign keys to dimensions")
	measures: List[str] = Field(..., description="Fact table measures")
	degenerate_dimensions: List[str] = Field(default_factory=list, description="Degenerate dimensions")
	partitioning_key: Optional[str] = Field(default=None, description="Partitioning key column")
	aggregation_strategy: Dict[str, Any] = Field(default_factory=dict, description="Pre-aggregation rules")
	data_lineage: Dict[str, Any] = Field(default_factory=dict, description="Data lineage information")
	refresh_frequency: str = Field(default="hourly", description="Data refresh frequency")
	row_count: Optional[int] = Field(default=None, description="Current row count")
	size_mb: Optional[float] = Field(default=None, description="Table size in MB")


class BIMeasure(BIAnalyticsBase):
	"""Business measure definition"""
	fact_table_id: str = Field(..., description="Parent fact table ID")
	name: str = Field(..., description="Measure name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Measure description")
	measure_type: BIMeasureType = Field(..., description="Type of measure")
	aggregation_type: BIAggregationType = Field(..., description="Default aggregation method")
	source_column: Optional[str] = Field(default=None, description="Source column for base measures")
	calculation_formula: Optional[str] = Field(default=None, description="Formula for calculated measures")
	format_string: str = Field(default="#,##0", description="Display format string")
	data_type: str = Field(default="decimal", description="Data type")
	is_visible: bool = Field(default=True, description="Visible in UI")
	sort_order: int = Field(default=0, description="Display sort order")
	folder_name: Optional[str] = Field(default=None, description="Display folder")
	business_definition: Optional[str] = Field(default=None, description="Business definition")
	calculation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Calculation rules")


# OLAP Cube Models
class BIOLAPCube(BIAnalyticsBase):
	"""OLAP cube definition"""
	warehouse_id: str = Field(..., description="Data warehouse ID")
	name: str = Field(..., description="Cube name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Cube description")
	fact_table_id: str = Field(..., description="Primary fact table ID")
	dimensions: List[str] = Field(..., description="Cube dimension IDs")
	measures: List[str] = Field(..., description="Cube measure IDs")
	calculated_members: List[Dict[str, Any]] = Field(default_factory=list, description="Calculated members")
	named_sets: List[Dict[str, Any]] = Field(default_factory=list, description="Named sets")
	aggregation_design: Dict[str, Any] = Field(default_factory=dict, description="Aggregation strategy")
	partitioning_config: Dict[str, Any] = Field(default_factory=dict, description="Cube partitioning")
	processing_config: Dict[str, Any] = Field(default_factory=dict, description="Processing settings")
	security_config: Dict[str, Any] = Field(default_factory=dict, description="Cube security")
	storage_mode: str = Field(default="molap", description="Storage mode (MOLAP/ROLAP/HOLAP)")
	last_processed: Optional[datetime] = Field(default=None, description="Last processing timestamp")
	processing_status: BIProcessingStatus = Field(default=BIProcessingStatus.PENDING, description="Processing status")
	estimated_size_mb: Optional[float] = Field(default=None, description="Estimated cube size")


class BICubePartition(BIAnalyticsBase):
	"""OLAP cube partition definition"""
	cube_id: str = Field(..., description="Parent cube ID")
	name: str = Field(..., description="Partition name")
	description: Optional[str] = Field(default=None, description="Partition description")
	source_query: str = Field(..., description="Source data query")
	slice_condition: Optional[str] = Field(default=None, description="Partition slice condition")
	storage_mode: str = Field(default="molap", description="Partition storage mode")
	aggregation_prefix: Optional[str] = Field(default=None, description="Aggregation prefix")
	processing_priority: int = Field(default=0, description="Processing priority")
	remote_datasource: Optional[str] = Field(default=None, description="Remote data source")
	estimated_rows: Optional[int] = Field(default=None, description="Estimated row count")
	last_processed: Optional[datetime] = Field(default=None, description="Last processing time")
	processing_duration_seconds: Optional[float] = Field(default=None, description="Processing duration")
	size_mb: Optional[float] = Field(default=None, description="Partition size in MB")


# Dashboard Models
class BIDashboard(BIAnalyticsBase):
	"""Business intelligence dashboard"""
	name: str = Field(..., description="Dashboard name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Dashboard description")
	dashboard_type: BIDashboardType = Field(..., description="Type of dashboard")
	category: str = Field(default="General", description="Dashboard category")
	layout_config: Dict[str, Any] = Field(..., description="Layout configuration")
	theme_config: Dict[str, Any] = Field(default_factory=dict, description="Theme and styling")
	filters: List[Dict[str, Any]] = Field(default_factory=list, description="Global dashboard filters")
	parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Dashboard parameters")
	refresh_interval: int = Field(default=300, description="Auto-refresh interval in seconds")
	mobile_config: Dict[str, Any] = Field(default_factory=dict, description="Mobile optimization")
	access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Access control")
	sharing_config: Dict[str, Any] = Field(default_factory=dict, description="Sharing settings")
	export_config: Dict[str, Any] = Field(default_factory=dict, description="Export configuration")
	subscription_config: Dict[str, Any] = Field(default_factory=dict, description="Subscription settings")
	usage_analytics: Dict[str, Any] = Field(default_factory=dict, description="Usage tracking")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class BIDashboardWidget(BIAnalyticsBase):
	"""Dashboard widget/component"""
	dashboard_id: str = Field(..., description="Parent dashboard ID")
	name: str = Field(..., description="Widget name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Widget description")
	widget_type: str = Field(..., description="Type of widget (chart, table, kpi, etc.)")
	chart_type: Optional[BIChartType] = Field(default=None, description="Chart type if applicable")
	position: Dict[str, Any] = Field(..., description="Widget position and size")
	data_source_config: Dict[str, Any] = Field(..., description="Data source configuration")
	visualization_config: Dict[str, Any] = Field(..., description="Visualization settings")
	interaction_config: Dict[str, Any] = Field(default_factory=dict, description="User interaction settings")
	filter_config: Dict[str, Any] = Field(default_factory=dict, description="Widget-specific filters")
	conditional_formatting: List[Dict[str, Any]] = Field(default_factory=list, description="Conditional formatting rules")
	drill_config: Dict[str, Any] = Field(default_factory=dict, description="Drill-down/through configuration")
	refresh_config: Dict[str, Any] = Field(default_factory=dict, description="Data refresh settings")
	cache_config: Dict[str, Any] = Field(default_factory=dict, description="Caching configuration")
	is_visible: bool = Field(default=True, description="Widget visibility")
	sort_order: int = Field(default=0, description="Display order")


# KPI and Scorecard Models
class BIKPI(BIAnalyticsBase):
	"""Key Performance Indicator definition"""
	name: str = Field(..., description="KPI name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="KPI description")
	category: str = Field(..., description="KPI category")
	business_owner: str = Field(..., description="Business owner/responsible person")
	calculation_formula: str = Field(..., description="KPI calculation formula")
	data_source_config: Dict[str, Any] = Field(..., description="Data source configuration")
	target_value: Optional[float] = Field(default=None, description="Target/goal value")
	threshold_config: Dict[str, Any] = Field(default_factory=dict, description="Performance thresholds")
	trend_direction: str = Field(default="higher_better", description="Desired trend direction")
	frequency: str = Field(default="daily", description="Calculation frequency")
	format_config: Dict[str, Any] = Field(default_factory=dict, description="Display formatting")
	benchmark_config: Dict[str, Any] = Field(default_factory=dict, description="Benchmark comparison")
	historical_context: Dict[str, Any] = Field(default_factory=dict, description="Historical context")
	business_context: Optional[str] = Field(default=None, description="Business context and rationale")
	improvement_actions: List[str] = Field(default_factory=list, description="Suggested improvement actions")
	related_kpis: List[str] = Field(default_factory=list, description="Related KPI IDs")


class BIScorecard(BIAnalyticsBase):
	"""Executive scorecard definition"""
	name: str = Field(..., description="Scorecard name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Scorecard description")
	scorecard_type: str = Field(..., description="Type of scorecard")
	business_objectives: List[Dict[str, Any]] = Field(..., description="Business objectives")
	kpi_groups: List[Dict[str, Any]] = Field(..., description="KPI groupings")
	weight_config: Dict[str, Any] = Field(default_factory=dict, description="KPI weights")
	scoring_methodology: Dict[str, Any] = Field(..., description="Scoring calculation method")
	time_periods: List[Dict[str, Any]] = Field(..., description="Time period configurations")
	benchmark_config: Dict[str, Any] = Field(default_factory=dict, description="Benchmark settings")
	threshold_config: Dict[str, Any] = Field(default_factory=dict, description="Performance thresholds")
	visualization_config: Dict[str, Any] = Field(default_factory=dict, description="Visual representation")
	drill_down_config: Dict[str, Any] = Field(default_factory=dict, description="Drill-down capabilities")
	narrative_config: Dict[str, Any] = Field(default_factory=dict, description="Automated narrative")
	distribution_list: List[str] = Field(default_factory=list, description="Distribution recipients")
	schedule_config: Dict[str, Any] = Field(default_factory=dict, description="Automated scheduling")


# Report Models
class BIReport(BIAnalyticsBase):
	"""Business intelligence report definition"""
	name: str = Field(..., description="Report name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Report description")
	report_category: str = Field(..., description="Report category")
	report_type: str = Field(..., description="Type of report (tabular, crosstab, chart, etc.)")
	data_source_config: Dict[str, Any] = Field(..., description="Data source configuration")
	query_definition: Dict[str, Any] = Field(..., description="Query/MDX definition")
	layout_config: Dict[str, Any] = Field(..., description="Report layout configuration")
	formatting_config: Dict[str, Any] = Field(..., description="Formatting and styling")
	parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Report parameters")
	filters: List[Dict[str, Any]] = Field(default_factory=list, description="Report filters")
	sorting_config: Dict[str, Any] = Field(default_factory=dict, description="Sorting configuration")
	grouping_config: Dict[str, Any] = Field(default_factory=dict, description="Grouping configuration")
	conditional_formatting: List[Dict[str, Any]] = Field(default_factory=list, description="Conditional formatting")
	interactive_features: Dict[str, Any] = Field(default_factory=dict, description="Interactive capabilities")
	export_formats: List[BIReportFormat] = Field(default_factory=list, description="Supported export formats")
	schedule_config: Optional[Dict[str, Any]] = Field(default=None, description="Scheduled execution")
	distribution_config: Dict[str, Any] = Field(default_factory=dict, description="Distribution settings")


class BIReportExecution(BIAnalyticsBase):
	"""Report execution tracking"""
	report_id: str = Field(..., description="Report ID")
	execution_id: str = Field(default_factory=uuid7str, description="Unique execution ID")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
	status: BIProcessingStatus = Field(default=BIProcessingStatus.PENDING, description="Execution status")
	started_at: datetime = Field(default_factory=datetime.utcnow, description="Execution start time")
	completed_at: Optional[datetime] = Field(default=None, description="Execution completion time")
	duration_seconds: Optional[float] = Field(default=None, description="Execution duration")
	output_format: BIReportFormat = Field(default=BIReportFormat.PDF, description="Output format")
	output_location: Optional[str] = Field(default=None, description="Output file location")
	file_size_bytes: Optional[int] = Field(default=None, description="Output file size")
	page_count: Optional[int] = Field(default=None, description="Number of pages")
	row_count: Optional[int] = Field(default=None, description="Number of data rows")
	error_message: Optional[str] = Field(default=None, description="Error details if failed")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
	resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource consumption")
	delivery_status: Dict[str, Any] = Field(default_factory=dict, description="Delivery tracking")


# ETL and Data Integration Models
class BIETLJob(BIAnalyticsBase):
	"""ETL job definition"""
	name: str = Field(..., description="ETL job name")
	description: Optional[str] = Field(default=None, description="Job description")
	job_type: str = Field(..., description="Type of ETL job")
	source_config: Dict[str, Any] = Field(..., description="Source system configuration")
	target_config: Dict[str, Any] = Field(..., description="Target system configuration")
	transformation_rules: List[Dict[str, Any]] = Field(..., description="Data transformation rules")
	data_quality_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Data quality checks")
	schedule_config: Dict[str, Any] = Field(default_factory=dict, description="Job scheduling")
	dependency_config: Dict[str, Any] = Field(default_factory=dict, description="Job dependencies")
	error_handling_config: Dict[str, Any] = Field(default_factory=dict, description="Error handling")
	notification_config: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")
	performance_config: Dict[str, Any] = Field(default_factory=dict, description="Performance tuning")
	security_config: Dict[str, Any] = Field(default_factory=dict, description="Security settings")
	monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring configuration")
	is_enabled: bool = Field(default=True, description="Job enabled status")
	last_run_time: Optional[datetime] = Field(default=None, description="Last execution time")
	next_run_time: Optional[datetime] = Field(default=None, description="Next scheduled run")


class BIETLExecution(BIAnalyticsBase):
	"""ETL job execution tracking"""
	job_id: str = Field(..., description="ETL job ID")
	execution_id: str = Field(default_factory=uuid7str, description="Unique execution ID")
	status: BIProcessingStatus = Field(default=BIProcessingStatus.PENDING, description="Execution status")
	started_at: datetime = Field(default_factory=datetime.utcnow, description="Execution start time")
	completed_at: Optional[datetime] = Field(default=None, description="Execution completion time")
	duration_seconds: Optional[float] = Field(default=None, description="Execution duration")
	rows_extracted: int = Field(default=0, description="Number of rows extracted")
	rows_transformed: int = Field(default=0, description="Number of rows transformed")
	rows_loaded: int = Field(default=0, description="Number of rows loaded")
	rows_rejected: int = Field(default=0, description="Number of rows rejected")
	data_volume_mb: float = Field(default=0.0, description="Data volume processed in MB")
	error_count: int = Field(default=0, description="Number of errors encountered")
	warning_count: int = Field(default=0, description="Number of warnings")
	quality_score: Optional[float] = Field(default=None, description="Data quality score")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
	resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource consumption")
	log_entries: List[Dict[str, Any]] = Field(default_factory=list, description="Execution log entries")
	error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if failed")


# User Analytics and Personalization Models
class BIUserPreferences(BIAnalyticsBase):
	"""User-specific BI preferences"""
	user_id: str = Field(..., description="User ID")
	default_dashboard_id: Optional[str] = Field(default=None, description="Default dashboard")
	theme_preferences: Dict[str, Any] = Field(default_factory=dict, description="UI theme preferences")
	layout_preferences: Dict[str, Any] = Field(default_factory=dict, description="Layout preferences")
	filter_defaults: Dict[str, Any] = Field(default_factory=dict, description="Default filter values")
	notification_preferences: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")
	export_preferences: Dict[str, Any] = Field(default_factory=dict, description="Export preferences")
	favorite_reports: List[str] = Field(default_factory=list, description="Favorite report IDs")
	favorite_dashboards: List[str] = Field(default_factory=list, description="Favorite dashboard IDs")
	recent_items: List[Dict[str, Any]] = Field(default_factory=list, description="Recently accessed items")
	custom_shortcuts: List[Dict[str, Any]] = Field(default_factory=list, description="Custom shortcuts")
	collaboration_preferences: Dict[str, Any] = Field(default_factory=dict, description="Collaboration settings")
	mobile_preferences: Dict[str, Any] = Field(default_factory=dict, description="Mobile app preferences")


class BIUsageAnalytics(BIAnalyticsBase):
	"""BI platform usage analytics"""
	user_id: str = Field(..., description="User ID")
	session_id: str = Field(..., description="Session ID")
	item_type: str = Field(..., description="Type of item accessed (dashboard, report, etc.)")
	item_id: str = Field(..., description="Item ID")
	action_type: str = Field(..., description="Action performed (view, export, share, etc.)")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Action timestamp")
	duration_seconds: Optional[float] = Field(default=None, description="Time spent on action")
	device_info: Dict[str, Any] = Field(default_factory=dict, description="Device information")
	browser_info: Dict[str, Any] = Field(default_factory=dict, description="Browser information")
	location_info: Dict[str, Any] = Field(default_factory=dict, description="Geographic information")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
	interaction_details: Dict[str, Any] = Field(default_factory=dict, description="Interaction details")
	error_info: Optional[Dict[str, Any]] = Field(default=None, description="Error information if applicable")


# Advanced Analytics Models
class BIDataMiningModel(BIAnalyticsBase):
	"""Data mining and advanced analytics model"""
	name: str = Field(..., description="Model name")
	description: Optional[str] = Field(default=None, description="Model description")
	model_type: str = Field(..., description="Type of data mining model")
	algorithm: str = Field(..., description="Algorithm used")
	data_source_config: Dict[str, Any] = Field(..., description="Training data configuration")
	feature_config: Dict[str, Any] = Field(..., description="Feature selection configuration")
	training_config: Dict[str, Any] = Field(..., description="Training parameters")
	validation_config: Dict[str, Any] = Field(..., description="Validation configuration")
	model_artifacts: Dict[str, Any] = Field(default_factory=dict, description="Model artifacts and weights")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Model performance metrics")
	interpretation_config: Dict[str, Any] = Field(default_factory=dict, description="Model interpretation")
	deployment_config: Dict[str, Any] = Field(default_factory=dict, description="Deployment configuration")
	monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Model monitoring")
	version: str = Field(default="1.0", description="Model version")
	is_deployed: bool = Field(default=False, description="Deployment status")
	last_trained: Optional[datetime] = Field(default=None, description="Last training timestamp")
	next_retrain: Optional[datetime] = Field(default=None, description="Next retraining schedule")


class BIForecastModel(BIAnalyticsBase):
	"""Business forecasting model"""
	name: str = Field(..., description="Forecast model name")
	description: Optional[str] = Field(default=None, description="Model description")
	forecast_target: str = Field(..., description="What is being forecasted")
	forecast_horizon: int = Field(..., description="Forecast horizon in time units")
	time_granularity: str = Field(..., description="Time granularity (daily, weekly, monthly)")
	algorithm_type: str = Field(..., description="Forecasting algorithm")
	seasonal_config: Dict[str, Any] = Field(default_factory=dict, description="Seasonality configuration")
	trend_config: Dict[str, Any] = Field(default_factory=dict, description="Trend configuration")
	external_factors: List[str] = Field(default_factory=list, description="External factor variables")
	confidence_levels: List[float] = Field(default_factory=list, description="Confidence interval levels")
	accuracy_metrics: Dict[str, Any] = Field(default_factory=dict, description="Forecast accuracy metrics")
	backtesting_config: Dict[str, Any] = Field(default_factory=dict, description="Backtesting configuration")
	alert_thresholds: Dict[str, Any] = Field(default_factory=dict, description="Forecast alert thresholds")
	business_context: Optional[str] = Field(default=None, description="Business context and assumptions")
	last_forecast_date: Optional[date] = Field(default=None, description="Last forecast generation date")
	forecast_data: Optional[Dict[str, Any]] = Field(default=None, description="Latest forecast results")


# Collaboration and Workflow Models
class BICollaborationSpace(BIAnalyticsBase):
	"""Collaborative BI workspace"""
	name: str = Field(..., description="Workspace name")
	description: Optional[str] = Field(default=None, description="Workspace description")
	workspace_type: str = Field(..., description="Type of workspace")
	owner_id: str = Field(..., description="Workspace owner user ID")
	members: List[Dict[str, Any]] = Field(default_factory=list, description="Workspace members")
	permissions_config: Dict[str, Any] = Field(default_factory=dict, description="Permission settings")
	shared_resources: List[Dict[str, Any]] = Field(default_factory=list, description="Shared BI resources")
	discussion_threads: List[Dict[str, Any]] = Field(default_factory=list, description="Discussion threads")
	version_control: Dict[str, Any] = Field(default_factory=dict, description="Version control settings")
	approval_workflow: Dict[str, Any] = Field(default_factory=dict, description="Approval workflow")
	notification_settings: Dict[str, Any] = Field(default_factory=dict, description="Notification configuration")
	activity_log: List[Dict[str, Any]] = Field(default_factory=list, description="Activity tracking")
	storage_quota_mb: int = Field(default=1000, description="Storage quota in MB")
	is_public: bool = Field(default=False, description="Public workspace flag")


class BIWorkflowTask(BIAnalyticsBase):
	"""BI workflow task definition"""
	workflow_id: str = Field(..., description="Parent workflow ID")
	name: str = Field(..., description="Task name")
	description: Optional[str] = Field(default=None, description="Task description")
	task_type: str = Field(..., description="Type of task")
	assigned_to: Optional[str] = Field(default=None, description="Assigned user ID")
	priority: str = Field(default="medium", description="Task priority")
	due_date: Optional[datetime] = Field(default=None, description="Task due date")
	dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
	task_config: Dict[str, Any] = Field(default_factory=dict, description="Task-specific configuration")
	completion_criteria: Dict[str, Any] = Field(default_factory=dict, description="Completion criteria")
	approval_required: bool = Field(default=False, description="Requires approval")
	status: str = Field(default="pending", description="Task status")
	progress_percentage: float = Field(default=0.0, description="Task progress")
	started_at: Optional[datetime] = Field(default=None, description="Task start time")
	completed_at: Optional[datetime] = Field(default=None, description="Task completion time")
	comments: List[Dict[str, Any]] = Field(default_factory=list, description="Task comments")
	attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Task attachments")


# Validation Methods
@validator('threshold_config', pre=True, always=True)
def validate_threshold_config(cls, v):
	if not isinstance(v, dict):
		raise ValueError("Threshold configuration must be a dictionary")
	return v


@root_validator
def validate_date_consistency(cls, values):
	started_at = values.get('started_at')
	completed_at = values.get('completed_at')
	
	if started_at and completed_at and completed_at < started_at:
		raise ValueError("Completion time cannot be before start time")
	
	return values


@validator('target_value')
def validate_target_value(cls, v):
	if v is not None and not isinstance(v, (int, float)):
		raise ValueError("Target value must be numeric")
	return v


# Industry-Specific Models
class BIFinancialAnalytics(BIAnalyticsBase):
	"""Financial business intelligence models"""
	analysis_type: str = Field(..., description="Type of financial analysis")
	chart_of_accounts_mapping: Dict[str, Any] = Field(..., description="Chart of accounts mapping")
	financial_periods: List[Dict[str, Any]] = Field(..., description="Financial period definitions")
	currency_config: Dict[str, Any] = Field(default_factory=dict, description="Multi-currency configuration")
	consolidation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Consolidation rules")
	variance_analysis_config: Dict[str, Any] = Field(default_factory=dict, description="Variance analysis setup")
	budget_forecast_config: Dict[str, Any] = Field(default_factory=dict, description="Budget and forecast config")
	ratio_analysis_config: Dict[str, Any] = Field(default_factory=dict, description="Financial ratio analysis")
	regulatory_reporting_config: Dict[str, Any] = Field(default_factory=dict, description="Regulatory reporting")
	audit_trail_config: Dict[str, Any] = Field(default_factory=dict, description="Audit trail configuration")


class BISalesAnalytics(BIAnalyticsBase):
	"""Sales business intelligence models"""
	sales_territory_hierarchy: Dict[str, Any] = Field(..., description="Territory hierarchy")
	product_hierarchy: Dict[str, Any] = Field(..., description="Product hierarchy")
	customer_segmentation: Dict[str, Any] = Field(..., description="Customer segmentation rules")
	sales_funnel_config: Dict[str, Any] = Field(..., description="Sales funnel analysis")
	quota_management_config: Dict[str, Any] = Field(default_factory=dict, description="Quota management")
	commission_calculation: Dict[str, Any] = Field(default_factory=dict, description="Commission calculations")
	pipeline_analysis_config: Dict[str, Any] = Field(default_factory=dict, description="Pipeline analysis")
	win_loss_analysis_config: Dict[str, Any] = Field(default_factory=dict, description="Win/loss analysis")
	forecasting_models: List[Dict[str, Any]] = Field(default_factory=list, description="Sales forecasting models")
	performance_benchmarks: Dict[str, Any] = Field(default_factory=dict, description="Performance benchmarks")


class BIOperationalAnalytics(BIAnalyticsBase):
	"""Operational business intelligence models"""
	process_hierarchy: Dict[str, Any] = Field(..., description="Business process hierarchy")
	operational_kpis: List[Dict[str, Any]] = Field(..., description="Operational KPIs")
	efficiency_metrics: Dict[str, Any] = Field(..., description="Efficiency measurement")
	quality_metrics: Dict[str, Any] = Field(..., description="Quality metrics")
	capacity_planning_config: Dict[str, Any] = Field(default_factory=dict, description="Capacity planning")
	resource_utilization_config: Dict[str, Any] = Field(default_factory=dict, description="Resource utilization")
	bottleneck_analysis_config: Dict[str, Any] = Field(default_factory=dict, description="Bottleneck analysis")
	process_optimization_config: Dict[str, Any] = Field(default_factory=dict, description="Process optimization")
	sla_monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="SLA monitoring")
	cost_center_analysis: Dict[str, Any] = Field(default_factory=dict, description="Cost center analysis")