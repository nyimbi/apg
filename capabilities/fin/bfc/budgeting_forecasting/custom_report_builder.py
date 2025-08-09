"""
APG Budgeting & Forecasting - Custom Report Builder

Flexible report generation system with templates, scheduling, and multi-format output.
Supports dynamic report creation, automated scheduling, and comprehensive export options.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from enum import Enum
from datetime import datetime, date, timedelta, time
from decimal import Decimal
import asyncio
import logging
import json
from uuid_extensions import uuid7str
import io
import csv

from .models import APGBaseModel, PositiveAmount, NonEmptyString
from .service import APGTenantContext, ServiceResponse, APGServiceBase


# =============================================================================
# Report Builder Enumerations
# =============================================================================

class ReportType(str, Enum):
	"""Types of reports that can be generated."""
	BUDGET_SUMMARY = "budget_summary"
	VARIANCE_ANALYSIS = "variance_analysis"
	FORECAST_REPORT = "forecast_report"
	DEPARTMENT_BREAKDOWN = "department_breakdown"
	TREND_ANALYSIS = "trend_analysis"
	FINANCIAL_DASHBOARD = "financial_dashboard"
	CUSTOM_QUERY = "custom_query"
	COMPLIANCE_REPORT = "compliance_report"


class ReportFormat(str, Enum):
	"""Supported report output formats."""
	PDF = "pdf"
	EXCEL = "excel"
	CSV = "csv"
	JSON = "json"
	HTML = "html"
	POWERPOINT = "powerpoint"


class ReportFrequency(str, Enum):
	"""Report scheduling frequencies."""
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	YEARLY = "yearly"
	CUSTOM = "custom"


class ReportStatus(str, Enum):
	"""Report generation status."""
	DRAFT = "draft"
	SCHEDULED = "scheduled"
	GENERATING = "generating"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class DataSourceType(str, Enum):
	"""Types of data sources for reports."""
	BUDGET_DATA = "budget_data"
	ACTUAL_DATA = "actual_data"
	FORECAST_DATA = "forecast_data"
	VARIANCE_DATA = "variance_data"
	CUSTOM_QUERY = "custom_query"
	EXTERNAL_API = "external_api"


class AggregationType(str, Enum):
	"""Data aggregation types."""
	SUM = "sum"
	AVERAGE = "average"
	COUNT = "count"
	MIN = "min"
	MAX = "max"
	MEDIAN = "median"
	CUSTOM = "custom"


class FilterOperator(str, Enum):
	"""Filter operators for report data."""
	EQUALS = "equals"
	NOT_EQUALS = "not_equals"
	GREATER_THAN = "greater_than"
	LESS_THAN = "less_than"
	GREATER_EQUAL = "greater_equal"
	LESS_EQUAL = "less_equal"
	CONTAINS = "contains"
	IN = "in"
	NOT_IN = "not_in"
	BETWEEN = "between"


# =============================================================================
# Report Builder Models
# =============================================================================

class ReportField(APGBaseModel):
	"""Individual field in a report."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	field_id: str = Field(default_factory=uuid7str)
	field_name: NonEmptyString = Field(description="Field name")
	display_name: str = Field(description="Display name for the field")
	data_type: str = Field(description="Data type (string, number, date, etc.)")
	
	# Field Configuration
	source_table: str = Field(description="Source table or view")
	source_column: str = Field(description="Source column name")
	aggregation: Optional[AggregationType] = Field(None, description="Aggregation type")
	
	# Formatting
	number_format: Optional[str] = Field(None, description="Number formatting pattern")
	date_format: Optional[str] = Field(None, description="Date formatting pattern")
	decimal_places: Optional[int] = Field(None, description="Number of decimal places")
	
	# Display Options
	column_width: Optional[int] = Field(None, description="Column width in pixels")
	alignment: str = Field(default="left", description="Text alignment")
	is_visible: bool = Field(default=True, description="Field visibility")
	sort_order: Optional[int] = Field(None, description="Sort order")
	
	# Calculations
	is_calculated: bool = Field(default=False, description="Is calculated field")
	calculation_formula: Optional[str] = Field(None, description="Calculation formula")


class ReportFilter(APGBaseModel):
	"""Report filter configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	filter_id: str = Field(default_factory=uuid7str)
	filter_name: NonEmptyString = Field(description="Filter name")
	field_name: str = Field(description="Field to filter on")
	
	# Filter Configuration
	operator: FilterOperator = Field(description="Filter operator")
	value: Any = Field(description="Filter value")
	is_required: bool = Field(default=False, description="Filter is required")
	
	# User Interface
	input_type: str = Field(description="UI input type (text, select, date, etc.)")
	available_values: List[Any] = Field(default_factory=list, description="Available values for select")
	placeholder: Optional[str] = Field(None, description="Placeholder text")
	
	# Current State
	current_value: Optional[Any] = Field(None, description="Current filter value")
	is_active: bool = Field(default=False, description="Filter is active")


class ReportSection(APGBaseModel):
	"""Report section configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	section_id: str = Field(default_factory=uuid7str)
	section_name: NonEmptyString = Field(description="Section name")
	section_type: str = Field(description="Section type (header, data, chart, summary)")
	
	# Section Configuration
	title: str = Field(description="Section title")
	description: Optional[str] = Field(None, description="Section description")
	order_index: int = Field(description="Section order")
	
	# Data Configuration
	data_source: DataSourceType = Field(description="Data source for section")
	fields: List[ReportField] = Field(default_factory=list, description="Fields in section")
	filters: List[ReportFilter] = Field(default_factory=list, description="Section-specific filters")
	
	# Layout Configuration
	layout_type: str = Field(default="table", description="Layout type (table, chart, cards)")
	columns: int = Field(default=1, description="Number of columns")
	
	# Chart Configuration (if applicable)
	chart_type: Optional[str] = Field(None, description="Chart type")
	chart_config: Dict[str, Any] = Field(default_factory=dict, description="Chart configuration")
	
	# Formatting
	styling: Dict[str, Any] = Field(default_factory=dict, description="Section styling")


class ReportTemplate(APGBaseModel):
	"""Reusable report template."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	template_id: str = Field(default_factory=uuid7str)
	template_name: NonEmptyString = Field(description="Template name")
	template_description: Optional[str] = Field(None, description="Template description")
	report_type: ReportType = Field(description="Type of report")
	
	# Template Configuration
	sections: List[ReportSection] = Field(default_factory=list, description="Report sections")
	global_filters: List[ReportFilter] = Field(default_factory=list, description="Global filters")
	
	# Layout Configuration
	page_layout: Dict[str, Any] = Field(default_factory=dict, description="Page layout settings")
	header_template: Optional[str] = Field(None, description="Header template")
	footer_template: Optional[str] = Field(None, description="Footer template")
	
	# Template Metadata
	category: str = Field(description="Template category")
	tags: List[str] = Field(default_factory=list, description="Template tags")
	is_public: bool = Field(default=False, description="Template is publicly available")
	usage_count: int = Field(default=0, description="Number of times used")
	
	# Version Control
	version: str = Field(default="1.0.0", description="Template version")
	parent_template_id: Optional[str] = Field(None, description="Parent template if derived")


class ReportSchedule(APGBaseModel):
	"""Report scheduling configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	schedule_id: str = Field(default_factory=uuid7str)
	schedule_name: NonEmptyString = Field(description="Schedule name")
	report_template_id: str = Field(description="Report template to run")
	
	# Schedule Configuration
	frequency: ReportFrequency = Field(description="Schedule frequency")
	start_date: date = Field(description="Schedule start date")
	end_date: Optional[date] = Field(None, description="Schedule end date")
	
	# Time Configuration
	run_time: time = Field(description="Time to run report")
	timezone: str = Field(default="UTC", description="Timezone for scheduling")
	
	# Advanced Scheduling
	custom_cron: Optional[str] = Field(None, description="Custom cron expression")
	skip_weekends: bool = Field(default=False, description="Skip weekend runs")
	skip_holidays: bool = Field(default=False, description="Skip holiday runs")
	
	# Output Configuration
	output_formats: List[ReportFormat] = Field(description="Output formats")
	delivery_method: str = Field(description="Delivery method (email, storage, api)")
	recipients: List[str] = Field(default_factory=list, description="Report recipients")
	
	# Status
	is_active: bool = Field(default=True, description="Schedule is active")
	last_run: Optional[datetime] = Field(None, description="Last run timestamp")
	next_run: Optional[datetime] = Field(None, description="Next run timestamp")
	run_count: int = Field(default=0, description="Number of times run")


class ReportExecution(APGBaseModel):
	"""Report execution instance."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	execution_id: str = Field(default_factory=uuid7str)
	template_id: str = Field(description="Report template used")
	schedule_id: Optional[str] = Field(None, description="Schedule ID if scheduled")
	
	# Execution Configuration
	report_name: str = Field(description="Generated report name")
	output_format: ReportFormat = Field(description="Output format")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Report parameters")
	
	# Execution Status
	status: ReportStatus = Field(default=ReportStatus.DRAFT)
	start_time: Optional[datetime] = Field(None, description="Execution start time")
	end_time: Optional[datetime] = Field(None, description="Execution end time")
	duration_seconds: Optional[Decimal] = Field(None, description="Execution duration")
	
	# Results
	record_count: Optional[int] = Field(None, description="Number of records in report")
	file_size_bytes: Optional[int] = Field(None, description="Generated file size")
	output_path: Optional[str] = Field(None, description="Output file path")
	
	# Error Handling
	error_message: Optional[str] = Field(None, description="Error message if failed")
	retry_count: int = Field(default=0, description="Number of retries")
	max_retries: int = Field(default=3, description="Maximum retry attempts")
	
	# Metadata
	execution_context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")


# =============================================================================
# Custom Report Builder Service
# =============================================================================

class CustomReportBuilderService(APGServiceBase):
	"""
	Custom report builder service providing flexible report generation,
	template management, scheduling, and multi-format output capabilities.
	"""
	
	def __init__(self, context: APGTenantContext, config: Optional[Dict[str, Any]] = None):
		super().__init__(context, config)
		self.logger = logging.getLogger(__name__)
	
	async def create_report_template(
		self, 
		template_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Create new report template."""
		try:
			self.logger.info(f"Creating report template: {template_config.get('template_name')}")
			
			# Validate configuration
			required_fields = ['template_name', 'report_type', 'sections']
			missing_fields = [field for field in required_fields if field not in template_config]
			if missing_fields:
				return ServiceResponse(
					success=False,
					message=f"Missing required fields: {missing_fields}",
					errors=missing_fields
				)
			
			# Create template
			template = ReportTemplate(
				template_name=template_config['template_name'],
				template_description=template_config.get('template_description'),
				report_type=template_config['report_type'],
				category=template_config.get('category', 'Custom'),
				tags=template_config.get('tags', []),
				is_public=template_config.get('is_public', False),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Create sections
			await self._create_template_sections(template, template_config['sections'])
			
			# Create global filters
			if 'global_filters' in template_config:
				await self._create_template_filters(template, template_config['global_filters'])
			
			# Configure layout
			await self._configure_template_layout(template, template_config.get('layout', {}))
			
			self.logger.info(f"Report template created: {template.template_id}")
			
			return ServiceResponse(
				success=True,
				message="Report template created successfully",
				data=template.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error creating report template: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to create report template: {str(e)}",
				errors=[str(e)]
			)
	
	async def generate_report(
		self, 
		template_id: str, 
		generation_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Generate report from template."""
		try:
			self.logger.info(f"Generating report from template {template_id}")
			
			# Create execution record
			execution = ReportExecution(
				template_id=template_id,
				report_name=generation_config.get('report_name', f'Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
				output_format=generation_config.get('output_format', ReportFormat.PDF),
				parameters=generation_config.get('parameters', {}),
				status=ReportStatus.GENERATING,
				start_time=datetime.utcnow(),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			try:
				# Load template
				template = await self._load_report_template(template_id)
				
				# Apply parameters and filters
				report_data = await self._generate_report_data(template, execution.parameters)
				
				# Generate output in requested format
				output_content = await self._generate_report_output(
					template, report_data, execution.output_format
				)
				
				# Save output file
				output_path = await self._save_report_output(execution, output_content)
				
				# Update execution status
				execution.status = ReportStatus.COMPLETED
				execution.end_time = datetime.utcnow()
				execution.duration_seconds = Decimal((execution.end_time - execution.start_time).total_seconds())
				execution.record_count = len(report_data.get('data', []))
				execution.file_size_bytes = len(output_content) if isinstance(output_content, (str, bytes)) else 0
				execution.output_path = output_path
				
				self.logger.info(f"Report generated successfully: {execution.execution_id}")
				
				return ServiceResponse(
					success=True,
					message="Report generated successfully",
					data={
						'execution': execution.model_dump(),
						'output_path': output_path,
						'download_url': f"/api/reports/download/{execution.execution_id}"
					}
				)
				
			except Exception as e:
				# Update execution with error
				execution.status = ReportStatus.FAILED
				execution.end_time = datetime.utcnow()
				execution.error_message = str(e)
				raise
			
		except Exception as e:
			self.logger.error(f"Error generating report: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to generate report: {str(e)}",
				errors=[str(e)]
			)
	
	async def create_report_schedule(
		self, 
		schedule_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Create report schedule."""
		try:
			self.logger.info(f"Creating report schedule: {schedule_config.get('schedule_name')}")
			
			# Validate configuration
			required_fields = ['schedule_name', 'report_template_id', 'frequency', 'output_formats']
			missing_fields = [field for field in required_fields if field not in schedule_config]
			if missing_fields:
				return ServiceResponse(
					success=False,
					message=f"Missing required fields: {missing_fields}",
					errors=missing_fields
				)
			
			# Create schedule
			schedule = ReportSchedule(
				schedule_name=schedule_config['schedule_name'],
				report_template_id=schedule_config['report_template_id'],
				frequency=schedule_config['frequency'],
				start_date=schedule_config.get('start_date', date.today()),
				end_date=schedule_config.get('end_date'),
				run_time=schedule_config.get('run_time', time(9, 0)),
				timezone=schedule_config.get('timezone', 'UTC'),
				output_formats=schedule_config['output_formats'],
				delivery_method=schedule_config.get('delivery_method', 'email'),
				recipients=schedule_config.get('recipients', []),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Calculate next run time
			await self._calculate_next_run_time(schedule)
			
			self.logger.info(f"Report schedule created: {schedule.schedule_id}")
			
			return ServiceResponse(
				success=True,
				message="Report schedule created successfully",
				data=schedule.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error creating report schedule: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to create report schedule: {str(e)}",
				errors=[str(e)]
			)
	
	async def get_available_templates(
		self, 
		filter_criteria: Optional[Dict[str, Any]] = None
	) -> ServiceResponse:
		"""Get available report templates."""
		try:
			self.logger.info("Getting available report templates")
			
			# Apply filters
			filters = filter_criteria or {}
			
			# Get templates (simulated)
			templates = await self._get_filtered_templates(filters)
			
			return ServiceResponse(
				success=True,
				message="Templates retrieved successfully",
				data={
					'templates': templates,
					'total_count': len(templates),
					'filters_applied': filters
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error getting templates: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to get templates: {str(e)}",
				errors=[str(e)]
			)
	
	async def preview_report(
		self, 
		template_id: str, 
		preview_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Generate report preview."""
		try:
			self.logger.info(f"Generating report preview for template {template_id}")
			
			# Load template
			template = await self._load_report_template(template_id)
			
			# Generate preview data (limited records)
			preview_data = await self._generate_preview_data(
				template, 
				preview_config.get('parameters', {}),
				limit=preview_config.get('limit', 50)
			)
			
			# Generate preview output
			preview_content = await self._generate_preview_output(template, preview_data)
			
			return ServiceResponse(
				success=True,
				message="Report preview generated successfully",
				data={
					'preview_content': preview_content,
					'record_count': len(preview_data.get('data', [])),
					'is_truncated': len(preview_data.get('data', [])) >= preview_config.get('limit', 50)
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error generating report preview: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to generate preview: {str(e)}",
				errors=[str(e)]
			)
	
	# =============================================================================
	# Private Helper Methods
	# =============================================================================
	
	async def _create_template_sections(
		self, 
		template: ReportTemplate, 
		sections_config: List[Dict[str, Any]]
	) -> None:
		"""Create template sections."""
		sections = []
		
		for i, section_config in enumerate(sections_config):
			section = ReportSection(
				section_name=section_config['section_name'],
				section_type=section_config['section_type'],
				title=section_config['title'],
				description=section_config.get('description'),
				order_index=i,
				data_source=section_config['data_source'],
				layout_type=section_config.get('layout_type', 'table'),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Create fields
			if 'fields' in section_config:
				section.fields = await self._create_section_fields(section_config['fields'])
			
			sections.append(section)
		
		template.sections = sections
	
	async def _create_section_fields(self, fields_config: List[Dict[str, Any]]) -> List[ReportField]:
		"""Create section fields."""
		fields = []
		
		for field_config in fields_config:
			field = ReportField(
				field_name=field_config['field_name'],
				display_name=field_config['display_name'],
				data_type=field_config['data_type'],
				source_table=field_config['source_table'],
				source_column=field_config['source_column'],
				aggregation=field_config.get('aggregation'),
				number_format=field_config.get('number_format'),
				alignment=field_config.get('alignment', 'left'),
				is_visible=field_config.get('is_visible', True),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			fields.append(field)
		
		return fields
	
	async def _create_template_filters(
		self, 
		template: ReportTemplate, 
		filters_config: List[Dict[str, Any]]
	) -> None:
		"""Create template filters."""
		filters = []
		
		for filter_config in filters_config:
			filter_obj = ReportFilter(
				filter_name=filter_config['filter_name'],
				field_name=filter_config['field_name'],
				operator=filter_config['operator'],
				value=filter_config['value'],
				is_required=filter_config.get('is_required', False),
				input_type=filter_config.get('input_type', 'text'),
				available_values=filter_config.get('available_values', []),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			filters.append(filter_obj)
		
		template.global_filters = filters
	
	async def _configure_template_layout(
		self, 
		template: ReportTemplate, 
		layout_config: Dict[str, Any]
	) -> None:
		"""Configure template layout."""
		template.page_layout = {
			'page_size': layout_config.get('page_size', 'A4'),
			'orientation': layout_config.get('orientation', 'portrait'),
			'margins': layout_config.get('margins', {'top': 20, 'bottom': 20, 'left': 20, 'right': 20}),
			'header_height': layout_config.get('header_height', 50),
			'footer_height': layout_config.get('footer_height', 30)
		}
		
		template.header_template = layout_config.get('header_template')
		template.footer_template = layout_config.get('footer_template')
	
	async def _load_report_template(self, template_id: str) -> ReportTemplate:
		"""Load report template."""
		# Simulated template loading
		template = ReportTemplate(
			template_id=template_id,
			template_name="Budget Summary Report",
			report_type=ReportType.BUDGET_SUMMARY,
			category="Financial",
			tenant_id=self.context.tenant_id,
			created_by=self.context.user_id
		)
		
		# Add sample sections
		template.sections = [
			ReportSection(
				section_name="Summary",
				section_type="data",
				title="Budget Summary",
				order_index=0,
				data_source=DataSourceType.BUDGET_DATA,
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
		]
		
		return template
	
	async def _generate_report_data(
		self, 
		template: ReportTemplate, 
		parameters: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Generate report data."""
		# Simulated data generation
		return {
			'data': [
				{'department': 'Sales', 'budget': 400000, 'actual': 395000, 'variance': -5000},
				{'department': 'Marketing', 'budget': 300000, 'actual': 302500, 'variance': 2500},
				{'department': 'IT', 'budget': 450000, 'actual': 457500, 'variance': 7500},
				{'department': 'Operations', 'budget': 350000, 'actual': 340000, 'variance': -10000}
			],
			'summary': {
				'total_budget': 1500000,
				'total_actual': 1495000,
				'total_variance': -5000
			},
			'metadata': {
				'generated_at': datetime.utcnow(),
				'parameters': parameters
			}
		}
	
	async def _generate_report_output(
		self, 
		template: ReportTemplate, 
		data: Dict[str, Any], 
		output_format: ReportFormat
	) -> Union[str, bytes]:
		"""Generate report output in specified format."""
		if output_format == ReportFormat.JSON:
			return json.dumps(data, indent=2, default=str)
		
		elif output_format == ReportFormat.CSV:
			output = io.StringIO()
			if data.get('data'):
				fieldnames = data['data'][0].keys()
				writer = csv.DictWriter(output, fieldnames=fieldnames)
				writer.writeheader()
				writer.writerows(data['data'])
			return output.getvalue()
		
		elif output_format == ReportFormat.HTML:
			html_content = "<html><head><title>Budget Report</title></head><body>"
			html_content += f"<h1>{template.template_name}</h1>"
			html_content += "<table border='1'>"
			
			if data.get('data'):
				# Headers
				headers = data['data'][0].keys()
				html_content += "<tr>"
				for header in headers:
					html_content += f"<th>{header}</th>"
				html_content += "</tr>"
				
				# Data rows
				for row in data['data']:
					html_content += "<tr>"
					for value in row.values():
						html_content += f"<td>{value}</td>"
					html_content += "</tr>"
			
			html_content += "</table></body></html>"
			return html_content
		
		else:
			# For PDF, Excel, PowerPoint - would require additional libraries
			return f"Report output for format {output_format} (mock implementation)"
	
	async def _save_report_output(
		self, 
		execution: ReportExecution, 
		content: Union[str, bytes]
	) -> str:
		"""Save report output to storage."""
		# Simulated file saving
		file_extension = {
			ReportFormat.PDF: 'pdf',
			ReportFormat.EXCEL: 'xlsx',
			ReportFormat.CSV: 'csv',
			ReportFormat.JSON: 'json',
			ReportFormat.HTML: 'html',
			ReportFormat.POWERPOINT: 'pptx'
		}.get(execution.output_format, 'txt')
		
		output_path = f"/reports/{self.context.tenant_id}/{execution.execution_id}.{file_extension}"
		
		# In real implementation, would save to file system or cloud storage
		return output_path
	
	async def _calculate_next_run_time(self, schedule: ReportSchedule) -> None:
		"""Calculate next run time for schedule."""
		now = datetime.now()
		
		if schedule.frequency == ReportFrequency.DAILY:
			next_run = now.replace(hour=schedule.run_time.hour, minute=schedule.run_time.minute, second=0, microsecond=0)
			if next_run <= now:
				next_run += timedelta(days=1)
		elif schedule.frequency == ReportFrequency.WEEKLY:
			next_run = now.replace(hour=schedule.run_time.hour, minute=schedule.run_time.minute, second=0, microsecond=0)
			days_ahead = 7 - now.weekday()  # Monday = 0
			next_run += timedelta(days=days_ahead)
		elif schedule.frequency == ReportFrequency.MONTHLY:
			if now.day <= schedule.start_date.day:
				next_run = now.replace(day=schedule.start_date.day, hour=schedule.run_time.hour, minute=schedule.run_time.minute, second=0, microsecond=0)
			else:
				next_month = now.replace(day=1) + timedelta(days=32)
				next_run = next_month.replace(day=schedule.start_date.day, hour=schedule.run_time.hour, minute=schedule.run_time.minute, second=0, microsecond=0)
		else:
			next_run = now + timedelta(days=1)  # Default
		
		schedule.next_run = next_run
	
	async def _get_filtered_templates(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Get filtered report templates."""
		# Simulated template data
		templates = [
			{
				'template_id': 'tmpl_1',
				'template_name': 'Budget Summary Report',
				'report_type': 'budget_summary',
				'category': 'Financial',
				'created_date': datetime.utcnow(),
				'usage_count': 15
			},
			{
				'template_id': 'tmpl_2',
				'template_name': 'Variance Analysis Report',
				'report_type': 'variance_analysis',
				'category': 'Analysis',
				'created_date': datetime.utcnow(),
				'usage_count': 8
			}
		]
		
		# Apply filters (simplified)
		if 'category' in filters:
			templates = [t for t in templates if t['category'] == filters['category']]
		
		return templates
	
	async def _generate_preview_data(
		self, 
		template: ReportTemplate, 
		parameters: Dict[str, Any], 
		limit: int
	) -> Dict[str, Any]:
		"""Generate preview data with limited records."""
		full_data = await self._generate_report_data(template, parameters)
		
		# Limit data for preview
		if full_data.get('data'):
			full_data['data'] = full_data['data'][:limit]
		
		return full_data
	
	async def _generate_preview_output(
		self, 
		template: ReportTemplate, 
		data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Generate preview output."""
		return {
			'html_preview': await self._generate_report_output(template, data, ReportFormat.HTML),
			'data_sample': data['data'][:5] if data.get('data') else [],
			'column_info': [
				{'name': 'department', 'type': 'string'},
				{'name': 'budget', 'type': 'number'},
				{'name': 'actual', 'type': 'number'},
				{'name': 'variance', 'type': 'number'}
			]
		}


# =============================================================================
# Service Factory Functions
# =============================================================================

def create_custom_report_builder_service(
	context: APGTenantContext, 
	config: Optional[Dict[str, Any]] = None
) -> CustomReportBuilderService:
	"""Create custom report builder service instance."""
	return CustomReportBuilderService(context, config)


async def create_sample_budget_template(
	service: CustomReportBuilderService
) -> ServiceResponse:
	"""Create sample budget report template for testing."""
	template_config = {
		'template_name': 'Comprehensive Budget Report',
		'template_description': 'Detailed budget analysis with variance reporting',
		'report_type': ReportType.BUDGET_SUMMARY,
		'category': 'Financial Reports',
		'sections': [
			{
				'section_name': 'Summary',
				'section_type': 'data',
				'title': 'Budget Summary',
				'data_source': DataSourceType.BUDGET_DATA,
				'fields': [
					{
						'field_name': 'department',
						'display_name': 'Department',
						'data_type': 'string',
						'source_table': 'budgets',
						'source_column': 'department_name'
					},
					{
						'field_name': 'budget_amount',
						'display_name': 'Budget Amount',
						'data_type': 'number',
						'source_table': 'budgets',
						'source_column': 'budget_amount',
						'number_format': '$#,##0.00'
					}
				]
			}
		],
		'global_filters': [
			{
				'filter_name': 'Fiscal Year',
				'field_name': 'fiscal_year',
				'operator': FilterOperator.EQUALS,
				'value': '2025',
				'input_type': 'select',
				'available_values': ['2023', '2024', '2025']
			}
		]
	}
	
	return await service.create_report_template(template_config)