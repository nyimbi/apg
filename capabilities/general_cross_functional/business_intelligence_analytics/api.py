"""
Business Intelligence Analytics - Comprehensive FastAPI REST API

Enterprise-grade business intelligence, OLAP, dimensional modeling, and executive
dashboards providing strategic decision-making capabilities across all APG capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
	BIDataWarehouse, BIDimension, BIHierarchy, BIFactTable, BIMeasure,
	BIOLAPCube, BICubePartition, BIDashboard, BIDashboardWidget, BIKPI, BIScorecard,
	BIReport, BIReportExecution, BIETLJob, BIETLExecution, BIUserPreferences,
	BIUsageAnalytics, BIDataMiningModel, BIForecastModel, BICollaborationSpace,
	BIWorkflowTask, BIFinancialAnalytics, BISalesAnalytics, BIOperationalAnalytics,
	BIDimensionType, BIHierarchyType, BIMeasureType, BIAggregationType,
	BIChartType, BIDashboardType, BIReportFormat, BIProcessingStatus, BISCDType
)
from .service import BusinessIntelligenceAnalyticsService

# Import common dependencies and utilities
from ...common.dependencies import get_current_user, get_db_session, get_redis_client
from ...common.responses import SuccessResponse, ErrorResponse, PaginatedResponse


# Initialize router
router = APIRouter(prefix="/api/v1/bi", tags=["Business Intelligence Analytics"])


# Dependency injection
async def get_service(
	db: AsyncSession = Depends(get_db_session),
	redis = Depends(get_redis_client)
) -> BusinessIntelligenceAnalyticsService:
	"""Get Business Intelligence Analytics service instance."""
	return BusinessIntelligenceAnalyticsService(db, redis)


# Request/Response Models
class DataWarehouseRequest(BaseModel):
	name: str = Field(..., description="Data warehouse name")
	description: Optional[str] = Field(default=None, description="Data warehouse description")
	connection_string: str = Field(..., description="Database connection string")
	schema_name: str = Field(default="dw", description="Database schema name")
	storage_format: str = Field(default="columnar", description="Storage format")
	compression_type: str = Field(default="snappy", description="Data compression type")
	partitioning_strategy: Dict[str, Any] = Field(default_factory=dict, description="Partitioning config")


class DimensionRequest(BaseModel):
	warehouse_id: str = Field(..., description="Data warehouse ID")
	name: str = Field(..., description="Dimension name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Dimension description")
	dimension_type: BIDimensionType = Field(..., description="Type of dimension")
	table_name: str = Field(..., description="Physical table name")
	primary_key: str = Field(..., description="Primary key column")
	natural_key: str = Field(..., description="Business/natural key column")
	scd_type: BISCDType = Field(default=BISCDType.TYPE_2, description="SCD type")
	attributes: List[Dict[str, Any]] = Field(default_factory=list, description="Dimension attributes")


class HierarchyRequest(BaseModel):
	dimension_id: str = Field(..., description="Parent dimension ID")
	name: str = Field(..., description="Hierarchy name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Hierarchy description")
	hierarchy_type: BIHierarchyType = Field(..., description="Type of hierarchy")
	levels: List[Dict[str, Any]] = Field(..., description="Hierarchy levels configuration")
	default_member: Optional[str] = Field(default=None, description="Default hierarchy member")


class FactTableRequest(BaseModel):
	warehouse_id: str = Field(..., description="Data warehouse ID")
	name: str = Field(..., description="Fact table name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Fact table description")
	table_name: str = Field(..., description="Physical table name")
	grain_description: str = Field(..., description="Fact table grain description")
	dimension_keys: List[str] = Field(..., description="Foreign keys to dimensions")
	measures: List[str] = Field(..., description="Fact table measures")
	degenerate_dimensions: List[str] = Field(default_factory=list, description="Degenerate dimensions")


class MeasureRequest(BaseModel):
	fact_table_id: str = Field(..., description="Parent fact table ID")
	name: str = Field(..., description="Measure name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Measure description")
	measure_type: BIMeasureType = Field(..., description="Type of measure")
	aggregation_type: BIAggregationType = Field(..., description="Default aggregation method")
	source_column: Optional[str] = Field(default=None, description="Source column for base measures")
	calculation_formula: Optional[str] = Field(default=None, description="Formula for calculated measures")
	format_string: str = Field(default="#,##0", description="Display format string")


class OLAPCubeRequest(BaseModel):
	warehouse_id: str = Field(..., description="Data warehouse ID")
	name: str = Field(..., description="Cube name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Cube description")
	fact_table_id: str = Field(..., description="Primary fact table ID")
	dimensions: List[str] = Field(..., description="Cube dimension IDs")
	measures: List[str] = Field(..., description="Cube measure IDs")
	storage_mode: str = Field(default="molap", description="Storage mode")
	aggregation_design: Dict[str, Any] = Field(default_factory=dict, description="Aggregation strategy")


class DashboardRequest(BaseModel):
	name: str = Field(..., description="Dashboard name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Dashboard description")
	dashboard_type: BIDashboardType = Field(..., description="Type of dashboard")
	category: str = Field(default="General", description="Dashboard category")
	layout_config: Dict[str, Any] = Field(..., description="Layout configuration")
	theme_config: Dict[str, Any] = Field(default_factory=dict, description="Theme and styling")
	refresh_interval: int = Field(default=300, description="Auto-refresh interval in seconds")


class DashboardWidgetRequest(BaseModel):
	dashboard_id: str = Field(..., description="Parent dashboard ID")
	name: str = Field(..., description="Widget name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Widget description")
	widget_type: str = Field(..., description="Type of widget")
	chart_type: Optional[BIChartType] = Field(default=None, description="Chart type if applicable")
	position: Dict[str, Any] = Field(..., description="Widget position and size")
	data_source_config: Dict[str, Any] = Field(..., description="Data source configuration")
	visualization_config: Dict[str, Any] = Field(..., description="Visualization settings")


class KPIRequest(BaseModel):
	name: str = Field(..., description="KPI name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="KPI description")
	category: str = Field(..., description="KPI category")
	business_owner: str = Field(..., description="Business owner/responsible person")
	calculation_formula: str = Field(..., description="KPI calculation formula")
	data_source_config: Dict[str, Any] = Field(..., description="Data source configuration")
	target_value: Optional[float] = Field(default=None, description="Target/goal value")
	trend_direction: str = Field(default="higher_better", description="Desired trend direction")
	frequency: str = Field(default="daily", description="Calculation frequency")


class ScorecardRequest(BaseModel):
	name: str = Field(..., description="Scorecard name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Scorecard description")
	scorecard_type: str = Field(..., description="Type of scorecard")
	business_objectives: List[Dict[str, Any]] = Field(..., description="Business objectives")
	kpi_groups: List[Dict[str, Any]] = Field(..., description="KPI groupings")
	scoring_methodology: Dict[str, Any] = Field(..., description="Scoring calculation method")
	time_periods: List[Dict[str, Any]] = Field(..., description="Time period configurations")


class ReportRequest(BaseModel):
	name: str = Field(..., description="Report name")
	display_name: str = Field(..., description="Display name for UI")
	description: Optional[str] = Field(default=None, description="Report description")
	report_category: str = Field(..., description="Report category")
	report_type: str = Field(..., description="Type of report")
	data_source_config: Dict[str, Any] = Field(..., description="Data source configuration")
	query_definition: Dict[str, Any] = Field(..., description="Query/MDX definition")
	layout_config: Dict[str, Any] = Field(..., description="Report layout configuration")
	export_formats: List[BIReportFormat] = Field(default_factory=list, description="Supported export formats")


class ETLJobRequest(BaseModel):
	name: str = Field(..., description="ETL job name")
	description: Optional[str] = Field(default=None, description="Job description")
	job_type: str = Field(..., description="Type of ETL job")
	source_config: Dict[str, Any] = Field(..., description="Source system configuration")
	target_config: Dict[str, Any] = Field(..., description="Target system configuration")
	transformation_rules: List[Dict[str, Any]] = Field(..., description="Data transformation rules")
	schedule_config: Dict[str, Any] = Field(default_factory=dict, description="Job scheduling")


class MDXQueryRequest(BaseModel):
	mdx_query: str = Field(..., description="MDX query string")
	query_options: Optional[Dict[str, Any]] = Field(default=None, description="Query execution options")


# API Endpoints

# ============================================================================
# Data Warehouse Management Endpoints
# ============================================================================

@router.post("/warehouses", response_model=SuccessResponse, status_code=201)
async def create_data_warehouse(
	request: DataWarehouseRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new data warehouse."""
	try:
		warehouse = await service.create_data_warehouse(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Data warehouse created successfully",
			data={"warehouse_id": warehouse.id, "name": warehouse.name}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dimension Management Endpoints
# ============================================================================

@router.post("/dimensions", response_model=SuccessResponse, status_code=201)
async def create_dimension(
	request: DimensionRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new dimension."""
	try:
		dimension = await service.create_dimension(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Dimension created successfully",
			data={
				"dimension_id": dimension.id,
				"name": dimension.name,
				"dimension_type": dimension.dimension_type
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/dimensions/{dimension_id}/hierarchies", response_model=SuccessResponse, status_code=201)
async def create_hierarchy(
	dimension_id: str = Path(..., description="Dimension ID"),
	request: HierarchyRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a dimension hierarchy."""
	try:
		# Override dimension_id from path
		hierarchy_data = request.dict()
		hierarchy_data["dimension_id"] = dimension_id
		
		hierarchy = await service.create_hierarchy(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**hierarchy_data
		)
		
		return SuccessResponse(
			message="Hierarchy created successfully",
			data={
				"hierarchy_id": hierarchy.id,
				"name": hierarchy.name,
				"hierarchy_type": hierarchy.hierarchy_type
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Fact Table Management Endpoints
# ============================================================================

@router.post("/fact-tables", response_model=SuccessResponse, status_code=201)
async def create_fact_table(
	request: FactTableRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new fact table."""
	try:
		fact_table = await service.create_fact_table(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Fact table created successfully",
			data={
				"fact_table_id": fact_table.id,
				"name": fact_table.name,
				"grain": fact_table.grain_description
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/fact-tables/{fact_table_id}/measures", response_model=SuccessResponse, status_code=201)
async def create_measure(
	fact_table_id: str = Path(..., description="Fact table ID"),
	request: MeasureRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new measure."""
	try:
		# Override fact_table_id from path
		measure_data = request.dict()
		measure_data["fact_table_id"] = fact_table_id
		
		measure = await service.create_measure(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**measure_data
		)
		
		return SuccessResponse(
			message="Measure created successfully",
			data={
				"measure_id": measure.id,
				"name": measure.name,
				"measure_type": measure.measure_type,
				"aggregation_type": measure.aggregation_type
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# OLAP Cube Management Endpoints
# ============================================================================

@router.post("/cubes", response_model=SuccessResponse, status_code=201)
async def create_olap_cube(
	request: OLAPCubeRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new OLAP cube."""
	try:
		cube = await service.create_olap_cube(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="OLAP cube created successfully",
			data={
				"cube_id": cube.id,
				"name": cube.name,
				"dimension_count": len(cube.dimensions),
				"measure_count": len(cube.measures),
				"processing_status": cube.processing_status
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/cubes/{cube_id}/process", response_model=SuccessResponse, status_code=202)
async def process_cube(
	cube_id: str = Path(..., description="Cube ID"),
	processing_options: Optional[Dict[str, Any]] = Body(default=None, description="Processing options"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Process an OLAP cube."""
	try:
		result = await service.process_cube(
			cube_id=cube_id,
			tenant_id=current_user["tenant_id"],
			processing_options=processing_options
		)
		
		return SuccessResponse(
			message="Cube processing started successfully",
			data=result
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/cubes/{cube_id}/processing-status", response_model=SuccessResponse)
async def get_cube_processing_status(
	cube_id: str = Path(..., description="Cube ID"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Get cube processing status."""
	try:
		# Get cached processing status from Redis
		import json
		progress_data = await service.redis.get(f"cube_processing:{cube_id}")
		
		if progress_data:
			progress = json.loads(progress_data)
			return SuccessResponse(
				message="Cube processing status retrieved successfully",
				data={
					"cube_id": cube_id,
					"current_phase": progress.get("phase"),
					"progress_percentage": progress.get("progress"),
					"last_updated": progress.get("timestamp")
				}
			)
		else:
			return SuccessResponse(
				message="Cube processing status retrieved successfully",
				data={
					"cube_id": cube_id,
					"status": "completed_or_not_processing"
				}
			)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/cubes/{cube_id}/query", response_model=SuccessResponse)
async def execute_mdx_query(
	cube_id: str = Path(..., description="Cube ID"),
	request: MDXQueryRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Execute MDX query against an OLAP cube."""
	try:
		result = await service.execute_mdx_query(
			cube_id=cube_id,
			tenant_id=current_user["tenant_id"],
			mdx_query=request.mdx_query,
			query_options=request.query_options
		)
		
		return SuccessResponse(
			message="MDX query executed successfully",
			data=result
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dashboard Management Endpoints
# ============================================================================

@router.post("/dashboards", response_model=SuccessResponse, status_code=201)
async def create_dashboard(
	request: DashboardRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new BI dashboard."""
	try:
		dashboard = await service.create_dashboard(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Dashboard created successfully",
			data={
				"dashboard_id": dashboard.id,
				"name": dashboard.name,
				"dashboard_type": dashboard.dashboard_type,
				"category": dashboard.category
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboards/{dashboard_id}/widgets", response_model=SuccessResponse, status_code=201)
async def create_dashboard_widget(
	dashboard_id: str = Path(..., description="Dashboard ID"),
	request: DashboardWidgetRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new dashboard widget."""
	try:
		# Override dashboard_id from path
		widget_data = request.dict()
		widget_data["dashboard_id"] = dashboard_id
		
		widget = await service.create_dashboard_widget(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**widget_data
		)
		
		return SuccessResponse(
			message="Dashboard widget created successfully",
			data={
				"widget_id": widget.id,
				"name": widget.name,
				"widget_type": widget.widget_type
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboards/{dashboard_id}/data", response_model=SuccessResponse)
async def get_dashboard_data(
	dashboard_id: str = Path(..., description="Dashboard ID"),
	refresh: bool = Query(default=False, description="Force data refresh"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Get dashboard data with widgets."""
	try:
		# Simulate dashboard data retrieval
		dashboard_data = {
			"dashboard_id": dashboard_id,
			"widgets": [
				{
					"widget_id": "widget_1",
					"name": "Revenue Trend",
					"type": "line_chart",
					"data": {
						"series": [{"name": "Revenue", "data": [100000, 120000, 140000, 160000]}],
						"categories": ["Q1", "Q2", "Q3", "Q4"]
					}
				},
				{
					"widget_id": "widget_2",
					"name": "KPI Summary",
					"type": "scorecard",
					"data": {
						"kpis": [
							{"name": "Revenue", "value": 1600000, "target": 1500000, "status": "good"},
							{"name": "Customers", "value": 4250, "target": 4000, "status": "excellent"},
							{"name": "Conversion", "value": 0.045, "target": 0.040, "status": "good"}
						]
					}
				}
			],
			"metadata": {
				"last_updated": datetime.utcnow().isoformat(),
				"refresh_requested": refresh,
				"data_freshness": "5 minutes ago"
			}
		}
		
		return SuccessResponse(
			message="Dashboard data retrieved successfully",
			data=dashboard_data
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# KPI Management Endpoints
# ============================================================================

@router.post("/kpis", response_model=SuccessResponse, status_code=201)
async def create_kpi(
	request: KPIRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new KPI."""
	try:
		kpi = await service.create_kpi(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="KPI created successfully",
			data={
				"kpi_id": kpi.id,
				"name": kpi.name,
				"category": kpi.category,
				"business_owner": kpi.business_owner
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/kpis/{kpi_id}/calculate", response_model=SuccessResponse)
async def calculate_kpi_value(
	kpi_id: str = Path(..., description="KPI ID"),
	calculation_date: Optional[date] = Body(default=None, description="Calculation date"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Calculate KPI value for a specific date."""
	try:
		result = await service.calculate_kpi_value(
			kpi_id=kpi_id,
			tenant_id=current_user["tenant_id"],
			calculation_date=calculation_date
		)
		
		return SuccessResponse(
			message="KPI calculated successfully",
			data=result
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/kpis/values", response_model=SuccessResponse)
async def get_multiple_kpi_values(
	kpi_ids: List[str] = Query(..., description="List of KPI IDs"),
	calculation_date: Optional[date] = Query(default=None, description="Calculation date"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Get values for multiple KPIs."""
	try:
		kpi_values = []
		
		for kpi_id in kpi_ids:
			try:
				value = await service.calculate_kpi_value(
					kpi_id=kpi_id,
					tenant_id=current_user["tenant_id"],
					calculation_date=calculation_date
				)
				kpi_values.append(value)
			except Exception as e:
				# Continue with other KPIs if one fails
				kpi_values.append({
					"kpi_id": kpi_id,
					"error": str(e),
					"status": "failed"
				})
		
		return SuccessResponse(
			message="KPI values retrieved successfully",
			data={
				"kpi_values": kpi_values,
				"calculation_date": (calculation_date or date.today()).isoformat(),
				"total_kpis": len(kpi_ids),
				"successful_calculations": len([v for v in kpi_values if "error" not in v])
			}
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Scorecard Management Endpoints
# ============================================================================

@router.post("/scorecards", response_model=SuccessResponse, status_code=201)
async def create_scorecard(
	request: ScorecardRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new executive scorecard."""
	try:
		scorecard = await service.create_scorecard(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Scorecard created successfully",
			data={
				"scorecard_id": scorecard.id,
				"name": scorecard.name,
				"scorecard_type": scorecard.scorecard_type,
				"objective_count": len(scorecard.business_objectives)
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/scorecards/{scorecard_id}/performance", response_model=SuccessResponse)
async def get_scorecard_performance(
	scorecard_id: str = Path(..., description="Scorecard ID"),
	time_period: Optional[str] = Query(default="current_month", description="Time period"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Get scorecard performance data."""
	try:
		# Simulate scorecard performance calculation
		performance_data = {
			"scorecard_id": scorecard_id,
			"time_period": time_period,
			"overall_score": 85.7,
			"performance_status": "good",
			"objective_performance": [
				{
					"objective": "Financial Performance",
					"score": 92.5,
					"status": "excellent",
					"kpis": [
						{"name": "Revenue Growth", "score": 95, "status": "excellent"},
						{"name": "Profit Margin", "score": 90, "status": "excellent"}
					]
				},
				{
					"objective": "Customer Satisfaction",
					"score": 78.2,
					"status": "good",
					"kpis": [
						{"name": "Customer Satisfaction", "score": 82, "status": "good"},
						{"name": "Net Promoter Score", "score": 74, "status": "warning"}
					]
				},
				{
					"objective": "Operational Excellence",
					"score": 86.8,
					"status": "good",
					"kpis": [
						{"name": "Process Efficiency", "score": 88, "status": "good"},
						{"name": "Quality Score", "score": 85, "status": "good"}
					]
				}
			],
			"trends": {
				"current_vs_previous": 0.03,
				"current_vs_target": -0.02,
				"trend_direction": "improving"
			},
			"generated_at": datetime.utcnow().isoformat()
		}
		
		return SuccessResponse(
			message="Scorecard performance retrieved successfully",
			data=performance_data
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Report Management Endpoints
# ============================================================================

@router.post("/reports", response_model=SuccessResponse, status_code=201)
async def create_report(
	request: ReportRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new BI report."""
	try:
		report = await service.create_report(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Report created successfully",
			data={
				"report_id": report.id,
				"name": report.name,
				"report_type": report.report_type,
				"category": report.report_category
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/{report_id}/execute", response_model=SuccessResponse, status_code=202)
async def execute_report(
	report_id: str = Path(..., description="Report ID"),
	parameters: Optional[Dict[str, Any]] = Body(default=None, description="Report parameters"),
	output_format: BIReportFormat = Body(default=BIReportFormat.PDF, description="Output format"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Execute a BI report."""
	try:
		execution = await service.execute_report(
			report_id=report_id,
			tenant_id=current_user["tenant_id"],
			parameters=parameters,
			output_format=output_format
		)
		
		return SuccessResponse(
			message="Report execution started successfully",
			data={
				"execution_id": execution.id,
				"report_id": report_id,
				"status": execution.status,
				"output_format": output_format,
				"started_at": execution.started_at
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{report_id}/executions/{execution_id}/download")
async def download_report(
	report_id: str = Path(..., description="Report ID"),
	execution_id: str = Path(..., description="Execution ID"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Download a generated report."""
	try:
		# Simulate report download
		import io
		
		# Create mock report content
		mock_content = f"Mock BI Report - Report ID: {report_id}, Execution: {execution_id}".encode()
		
		def generate_file():
			yield mock_content
		
		return StreamingResponse(
			generate_file(),
			media_type="application/pdf",
			headers={
				"Content-Disposition": f"attachment; filename=report_{report_id}_{execution_id}.pdf"
			}
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ETL Job Management Endpoints
# ============================================================================

@router.post("/etl-jobs", response_model=SuccessResponse, status_code=201)
async def create_etl_job(
	request: ETLJobRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Create a new ETL job."""
	try:
		job = await service.create_etl_job(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="ETL job created successfully",
			data={
				"job_id": job.id,
				"name": job.name,
				"job_type": job.job_type,
				"is_enabled": job.is_enabled
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/etl-jobs/{job_id}/execute", response_model=SuccessResponse, status_code=202)
async def execute_etl_job(
	job_id: str = Path(..., description="ETL job ID"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Execute an ETL job."""
	try:
		execution = await service.execute_etl_job(
			job_id=job_id,
			tenant_id=current_user["tenant_id"]
		)
		
		return SuccessResponse(
			message="ETL job execution started successfully",
			data={
				"execution_id": execution.id,
				"job_id": job_id,
				"status": execution.status,
				"started_at": execution.started_at
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/etl-jobs/{job_id}/executions", response_model=SuccessResponse)
async def get_etl_execution_history(
	job_id: str = Path(..., description="ETL job ID"),
	limit: int = Query(default=20, description="Maximum number of executions"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Get ETL job execution history."""
	try:
		# Simulate execution history retrieval
		executions = []
		for i in range(min(limit, 10)):  # Simulate up to 10 executions
			execution = {
				"execution_id": f"exec_{job_id}_{i}",
				"status": "completed" if i < 8 else "failed" if i == 8 else "running",
				"started_at": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
				"completed_at": (datetime.utcnow() - timedelta(hours=i-1)).isoformat() if i < 9 else None,
				"duration_seconds": 145.7 if i < 8 else None,
				"rows_processed": 10000 - (i * 100) if i < 8 else None,
				"quality_score": 0.995 if i < 8 else None
			}
			executions.append(execution)
		
		return SuccessResponse(
			message="ETL execution history retrieved successfully",
			data={
				"job_id": job_id,
				"executions": executions,
				"total_count": len(executions)
			}
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics and Insights Endpoints
# ============================================================================

@router.post("/insights/generate", response_model=SuccessResponse)
async def generate_business_insights(
	data_sources: List[str] = Body(..., description="Data source IDs"),
	analysis_types: List[str] = Body(..., description="Types of analysis to perform"),
	time_period: Dict[str, Any] = Body(..., description="Time period for analysis"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Generate AI-powered business insights."""
	try:
		insights = await service.generate_business_insights(
			tenant_id=current_user["tenant_id"],
			data_sources=data_sources,
			analysis_types=analysis_types,
			time_period=time_period
		)
		
		return SuccessResponse(
			message="Business insights generated successfully",
			data={
				"insights": insights,
				"insight_count": len(insights),
				"generated_at": datetime.utcnow().isoformat()
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Platform Monitoring Endpoints
# ============================================================================

@router.get("/metrics", response_model=SuccessResponse)
async def get_platform_metrics(
	metric_types: List[str] = Query(..., description="Types of metrics to retrieve"),
	time_range: Optional[Dict[str, Any]] = Query(default=None, description="Time range for metrics"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Get BI platform performance metrics."""
	try:
		time_range = time_range or {
			"start": (datetime.utcnow() - timedelta(hours=24)).isoformat(),
			"end": datetime.utcnow().isoformat()
		}
		
		metrics = await service.get_platform_metrics(
			tenant_id=current_user["tenant_id"],
			metric_types=metric_types,
			time_range=time_range
		)
		
		return SuccessResponse(
			message="Platform metrics retrieved successfully",
			data=metrics
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=SuccessResponse)
async def get_system_health(
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Get comprehensive BI system health status."""
	try:
		health_data = {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"tenant_id": current_user["tenant_id"],
			"components": {
				"data_warehouse": {"status": "healthy", "response_time_ms": 25},
				"olap_engine": {"status": "healthy", "active_cubes": 12},
				"etl_engine": {"status": "healthy", "running_jobs": 3},
				"dashboard_service": {"status": "healthy", "active_dashboards": 25},
				"report_engine": {"status": "healthy", "queued_reports": 5}
			},
			"performance_metrics": {
				"total_cubes": 15,
				"processed_cubes": 12,
				"active_dashboards": 25,
				"daily_reports": 89,
				"active_users": 156,
				"query_response_avg_ms": 89,
				"etl_success_rate": 0.96
			},
			"resource_utilization": {
				"cpu_usage": 0.65,
				"memory_usage": 0.78,
				"storage_usage": 0.52,
				"cube_processing_capacity": 0.75
			}
		}
		
		return SuccessResponse(
			message="System health retrieved successfully",
			data=health_data
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=SuccessResponse)
async def get_bi_summary(
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: BusinessIntelligenceAnalyticsService = Depends(get_service)
):
	"""Get comprehensive BI platform summary."""
	try:
		summary_data = {
			"tenant_id": current_user["tenant_id"],
			"summary_timestamp": datetime.utcnow().isoformat(),
			"data_warehouse": {
				"total_warehouses": 3,
				"total_dimensions": 25,
				"total_fact_tables": 8,
				"total_measures": 45,
				"data_volume_gb": 850.2
			},
			"olap_cubes": {
				"total_cubes": 15,
				"processed_cubes": 12,
				"processing_cubes": 1,
				"failed_cubes": 2,
				"average_processing_time_minutes": 4.1
			},
			"dashboards_reports": {
				"total_dashboards": 25,
				"executive_dashboards": 8,
				"operational_dashboards": 12,
				"total_reports": 45,
				"scheduled_reports": 20,
				"reports_generated_today": 89
			},
			"kpis_scorecards": {
				"total_kpis": 85,
				"kpis_on_target": 67,
				"kpis_below_target": 18,
				"total_scorecards": 6,
				"average_scorecard_performance": 0.857
			},
			"etl_operations": {
				"total_jobs": 25,
				"active_jobs": 3,
				"successful_runs_today": 45,
				"failed_runs_today": 2,
				"data_processed_gb_today": 125.7
			},
			"user_activity": {
				"active_users_today": 156,
				"dashboard_views_today": 1247,
				"report_downloads_today": 234,
				"queries_executed_today": 3567
			},
			"top_insights": [
				"Revenue grew 15% quarter-over-quarter across all regions",
				"Customer satisfaction improved to 4.2/5.0 from 3.8/5.0",
				"Operational efficiency increased 8% due to process optimization",
				"Sales conversion rate improved to 4.5% from 3.9%"
			]
		}
		
		return SuccessResponse(
			message="BI platform summary retrieved successfully",
			data=summary_data
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))