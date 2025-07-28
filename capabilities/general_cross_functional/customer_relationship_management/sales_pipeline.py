"""
APG Customer Relationship Management - Sales Pipeline Management Module

Advanced sales pipeline management system with customizable stages, 
probability tracking, forecasting, and comprehensive analytics for 
revolutionary sales performance optimization.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from decimal import Decimal
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, validator

from .models import CRMOpportunity, OpportunityStage, Priority
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class StageType(str, Enum):
	"""Types of pipeline stages"""
	PROSPECTING = "prospecting"
	QUALIFICATION = "qualification"
	NEEDS_ANALYSIS = "needs_analysis"
	PROPOSAL = "proposal"
	NEGOTIATION = "negotiation"
	DECISION = "decision"
	CLOSED_WON = "closed_won"
	CLOSED_LOST = "closed_lost"
	ON_HOLD = "on_hold"
	CUSTOM = "custom"


class StageCategory(str, Enum):
	"""Categories of pipeline stages"""
	EARLY = "early"			# Initial stages of sales process
	MIDDLE = "middle"		# Active selling stages
	LATE = "late"			# Final decision stages
	CLOSED = "closed"		# Deal completed (won/lost)
	INACTIVE = "inactive"	# Paused or on-hold deals


class AutomationTrigger(str, Enum):
	"""Automation triggers for stage transitions"""
	TIME_BASED = "time_based"
	ACTIVITY_BASED = "activity_based"
	SCORE_BASED = "score_based"
	MANUAL_ONLY = "manual_only"


class PipelineStage(BaseModel):
	"""Sales pipeline stage configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	pipeline_id: str
	
	# Basic information
	name: str = Field(..., min_length=1, max_length=100)
	description: Optional[str] = Field(None, max_length=500)
	stage_type: StageType = StageType.CUSTOM
	category: StageCategory = StageCategory.EARLY
	
	# Stage properties
	order: int = Field(..., description="Display order in pipeline")
	probability: float = Field(..., ge=0, le=100, description="Win probability percentage")
	is_active: bool = Field(True, description="Whether stage is active")
	is_closed: bool = Field(False, description="Whether stage represents closed deal")
	is_won: bool = Field(False, description="Whether stage represents won deal")
	
	# Duration and timing
	expected_duration_days: Optional[int] = Field(None, description="Expected days in this stage")
	max_duration_days: Optional[int] = Field(None, description="Maximum days before escalation")
	
	# Automation settings
	automation_trigger: AutomationTrigger = AutomationTrigger.MANUAL_ONLY
	auto_advance_conditions: Dict[str, Any] = Field(default_factory=dict)
	required_activities: List[str] = Field(default_factory=list)
	required_fields: List[str] = Field(default_factory=list)
	
	# Stage-specific settings
	allows_forecasting: bool = Field(True, description="Include in sales forecasting")
	weighted_probability: float = Field(0.0, description="Weighted probability for forecasting")
	conversion_tracking: bool = Field(True, description="Track conversion metrics")
	
	# Notifications and alerts
	enable_alerts: bool = Field(False, description="Enable stage alerts")
	alert_conditions: Dict[str, Any] = Field(default_factory=dict)
	notification_emails: List[str] = Field(default_factory=list)
	
	# Colors and styling
	color: str = Field("#007bff", description="Stage color for visualization")
	icon: str = Field("circle", description="Stage icon")
	
	# Performance tracking
	opportunity_count: int = Field(0, description="Current opportunities in stage")
	total_value: Decimal = Field(Decimal('0'), description="Total value of opportunities")
	average_duration: float = Field(0.0, description="Average days spent in stage")
	conversion_rate: float = Field(0.0, description="Conversion rate to next stage")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class SalesPipeline(BaseModel):
	"""Sales pipeline configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Basic information
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	is_default: bool = Field(False, description="Default pipeline for new opportunities")
	is_active: bool = Field(True, description="Whether pipeline is active")
	
	# Pipeline settings
	currency: str = Field("USD", description="Default currency for opportunities")
	stages: List[PipelineStage] = Field(default_factory=list)
	
	# Automation settings
	enable_stage_automation: bool = Field(False, description="Enable automatic stage progression")
	enable_probability_updates: bool = Field(True, description="Auto-update probabilities")
	enable_forecasting: bool = Field(True, description="Enable sales forecasting")
	
	# Performance tracking
	total_opportunities: int = Field(0, description="Total opportunities in pipeline")
	total_value: Decimal = Field(Decimal('0'), description="Total pipeline value")
	weighted_value: Decimal = Field(Decimal('0'), description="Probability-weighted value")
	average_deal_size: Decimal = Field(Decimal('0'), description="Average opportunity value")
	average_cycle_time: float = Field(0.0, description="Average sales cycle in days")
	win_rate: float = Field(0.0, description="Overall win rate percentage")
	
	# Team assignments
	assigned_teams: List[str] = Field(default_factory=list)
	assigned_users: List[str] = Field(default_factory=list)
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class OpportunityStageHistory(BaseModel):
	"""History of opportunity stage changes"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	opportunity_id: str
	
	# Stage change details
	from_stage_id: Optional[str] = None
	to_stage_id: str
	from_stage_name: Optional[str] = None
	to_stage_name: str
	
	# Change metrics
	previous_probability: Optional[float] = None
	new_probability: float
	previous_value: Optional[Decimal] = None
	new_value: Decimal
	days_in_previous_stage: Optional[int] = None
	
	# Change context
	change_reason: Optional[str] = Field(None, description="Reason for stage change")
	notes: Optional[str] = Field(None, description="Additional notes")
	automated: bool = Field(False, description="Whether change was automated")
	
	# Audit fields
	changed_at: datetime = Field(default_factory=datetime.utcnow)
	changed_by: str
	metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineAnalytics(BaseModel):
	"""Comprehensive pipeline analytics"""
	pipeline_id: str
	pipeline_name: str
	tenant_id: str
	
	# Overall metrics
	total_opportunities: int = 0
	total_value: Decimal = Decimal('0')
	weighted_value: Decimal = Decimal('0')
	average_deal_size: Decimal = Decimal('0')
	
	# Performance metrics
	win_rate: float = 0.0
	loss_rate: float = 0.0
	average_cycle_time: float = 0.0
	median_cycle_time: float = 0.0
	
	# Stage analytics
	stage_metrics: List[Dict[str, Any]] = Field(default_factory=list)
	stage_conversion_rates: Dict[str, float] = Field(default_factory=dict)
	stage_durations: Dict[str, float] = Field(default_factory=dict)
	
	# Velocity metrics
	deals_created_this_month: int = 0
	deals_closed_this_month: int = 0
	deals_won_this_month: int = 0
	deals_lost_this_month: int = 0
	
	# Forecasting data
	forecast_this_month: Decimal = Decimal('0')
	forecast_next_month: Decimal = Decimal('0')
	forecast_this_quarter: Decimal = Decimal('0')
	forecast_confidence: float = 0.0
	
	# Trend analysis
	trend_direction: str = "stable"  # "up", "down", "stable"
	trend_strength: float = 0.0
	trend_period_days: int = 30
	
	# Bottleneck analysis
	bottleneck_stages: List[Dict[str, Any]] = Field(default_factory=list)
	longest_stage_durations: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Team performance
	top_performers: List[Dict[str, Any]] = Field(default_factory=list)
	team_metrics: Dict[str, Any] = Field(default_factory=dict)
	
	# Analysis metadata
	analyzed_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_period_days: int = 30


class SalesPipelineManager:
	"""
	Advanced sales pipeline management system
	
	Provides comprehensive pipeline configuration, stage management,
	opportunity tracking, analytics, and forecasting capabilities.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize sales pipeline manager
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
	
	async def initialize(self):
		"""Initialize the sales pipeline manager"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Sales Pipeline Manager...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		self._initialized = True
		logger.info("✅ Sales Pipeline Manager initialized successfully")
	
	async def create_pipeline(
		self,
		pipeline_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> SalesPipeline:
		"""
		Create a new sales pipeline
		
		Args:
			pipeline_data: Pipeline configuration data
			tenant_id: Tenant identifier
			created_by: User creating the pipeline
			
		Returns:
			Created sales pipeline
		"""
		try:
			logger.info(f"🚀 Creating sales pipeline: {pipeline_data.get('name')}")
			
			# Add required fields
			pipeline_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create pipeline object
			pipeline = SalesPipeline(**pipeline_data)
			
			# Create default stages if none provided
			if not pipeline.stages:
				pipeline.stages = await self._create_default_stages(pipeline, created_by)
			
			# Store pipeline in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_sales_pipelines (
						id, tenant_id, name, description, is_default, is_active,
						currency, enable_stage_automation, enable_probability_updates, enable_forecasting,
						total_opportunities, total_value, weighted_value, average_deal_size,
						average_cycle_time, win_rate, assigned_teams, assigned_users,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
						$15, $16, $17, $18, $19, $20, $21, $22, $23, $24
					)
				""", 
				pipeline.id, pipeline.tenant_id, pipeline.name, pipeline.description,
				pipeline.is_default, pipeline.is_active, pipeline.currency,
				pipeline.enable_stage_automation, pipeline.enable_probability_updates, pipeline.enable_forecasting,
				pipeline.total_opportunities, pipeline.total_value, pipeline.weighted_value, pipeline.average_deal_size,
				pipeline.average_cycle_time, pipeline.win_rate, pipeline.assigned_teams, pipeline.assigned_users,
				pipeline.metadata, pipeline.created_at, pipeline.updated_at,
				pipeline.created_by, pipeline.updated_by, pipeline.version
				)
				
				# Store pipeline stages
				for stage in pipeline.stages:
					await self._store_pipeline_stage(stage)
			
			logger.info(f"✅ Sales pipeline created successfully: {pipeline.id}")
			return pipeline
			
		except Exception as e:
			logger.error(f"Failed to create sales pipeline: {str(e)}", exc_info=True)
			raise
	
	async def create_pipeline_stage(
		self,
		stage_data: Dict[str, Any],
		pipeline_id: str,
		tenant_id: str,
		created_by: str
	) -> PipelineStage:
		"""
		Create a new pipeline stage
		
		Args:
			stage_data: Stage configuration data
			pipeline_id: Pipeline identifier
			tenant_id: Tenant identifier
			created_by: User creating the stage
			
		Returns:
			Created pipeline stage
		"""
		try:
			logger.info(f"📊 Creating pipeline stage: {stage_data.get('name')}")
			
			# Add required fields
			stage_data.update({
				'tenant_id': tenant_id,
				'pipeline_id': pipeline_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Calculate weighted probability
			if 'probability' in stage_data:
				stage_data['weighted_probability'] = stage_data['probability'] * 0.8  # Conservative weighting
			
			# Create stage object
			stage = PipelineStage(**stage_data)
			
			# Store in database
			await self._store_pipeline_stage(stage)
			
			logger.info(f"✅ Pipeline stage created successfully: {stage.id}")
			return stage
			
		except Exception as e:
			logger.error(f"Failed to create pipeline stage: {str(e)}", exc_info=True)
			raise
	
	async def move_opportunity_to_stage(
		self,
		opportunity_id: str,
		to_stage_id: str,
		tenant_id: str,
		changed_by: str,
		change_reason: Optional[str] = None,
		notes: Optional[str] = None
	) -> OpportunityStageHistory:
		"""
		Move an opportunity to a new stage
		
		Args:
			opportunity_id: Opportunity identifier
			to_stage_id: Target stage identifier
			tenant_id: Tenant identifier
			changed_by: User making the change
			change_reason: Reason for the change
			notes: Additional notes
			
		Returns:
			Stage change history record
		"""
		try:
			logger.info(f"🔄 Moving opportunity {opportunity_id} to stage {to_stage_id}")
			
			async with self.db_manager.get_connection() as conn:
				# Get current opportunity data
				opp_row = await conn.fetchrow("""
					SELECT o.*, ps.name as current_stage_name, ps.probability as current_probability
					FROM crm_opportunities o
					LEFT JOIN crm_pipeline_stages ps ON o.stage_id = ps.id
					WHERE o.id = $1 AND o.tenant_id = $2
				""", opportunity_id, tenant_id)
				
				if not opp_row:
					raise ValueError(f"Opportunity not found: {opportunity_id}")
				
				# Get target stage data
				target_stage = await conn.fetchrow("""
					SELECT * FROM crm_pipeline_stages
					WHERE id = $1 AND tenant_id = $2
				""", to_stage_id, tenant_id)
				
				if not target_stage:
					raise ValueError(f"Target stage not found: {to_stage_id}")
				
				# Calculate days in previous stage
				days_in_previous = None
				if opp_row['stage_updated_at']:
					days_in_previous = (datetime.utcnow() - opp_row['stage_updated_at']).days
				
				# Create stage history record
				history = OpportunityStageHistory(
					tenant_id=tenant_id,
					opportunity_id=opportunity_id,
					from_stage_id=opp_row['stage_id'],
					to_stage_id=to_stage_id,
					from_stage_name=opp_row['current_stage_name'],
					to_stage_name=target_stage['name'],
					previous_probability=opp_row['current_probability'],
					new_probability=target_stage['probability'],
					previous_value=opp_row['value'],
					new_value=opp_row['value'],  # Value might be updated separately
					days_in_previous_stage=days_in_previous,
					change_reason=change_reason,
					notes=notes,
					changed_by=changed_by
				)
				
				# Update opportunity stage
				await conn.execute("""
					UPDATE crm_opportunities SET
						stage_id = $3,
						probability = $4,
						stage_updated_at = NOW(),
						updated_at = NOW(),
						updated_by = $5
					WHERE id = $1 AND tenant_id = $2
				""", opportunity_id, tenant_id, to_stage_id, target_stage['probability'], changed_by)
				
				# Store stage history
				await conn.execute("""
					INSERT INTO crm_opportunity_stage_history (
						id, tenant_id, opportunity_id, from_stage_id, to_stage_id,
						from_stage_name, to_stage_name, previous_probability, new_probability,
						previous_value, new_value, days_in_previous_stage, change_reason,
						notes, automated, changed_at, changed_by, metadata
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
					)
				""", 
				history.id, history.tenant_id, history.opportunity_id,
				history.from_stage_id, history.to_stage_id,
				history.from_stage_name, history.to_stage_name,
				history.previous_probability, history.new_probability,
				history.previous_value, history.new_value,
				history.days_in_previous_stage, history.change_reason,
				history.notes, history.automated, history.changed_at,
				history.changed_by, history.metadata
				)
				
				# Update stage statistics
				await self._update_stage_statistics(to_stage_id, tenant_id)
				if opp_row['stage_id']:
					await self._update_stage_statistics(opp_row['stage_id'], tenant_id)
			
			logger.info(f"✅ Opportunity moved successfully to stage: {target_stage['name']}")
			return history
			
		except Exception as e:
			logger.error(f"Failed to move opportunity to stage: {str(e)}", exc_info=True)
			raise
	
	async def get_pipeline_analytics(
		self,
		pipeline_id: str,
		tenant_id: str,
		period_days: int = 30
	) -> PipelineAnalytics:
		"""
		Get comprehensive pipeline analytics
		
		Args:
			pipeline_id: Pipeline identifier
			tenant_id: Tenant identifier
			period_days: Analysis period in days
			
		Returns:
			Pipeline analytics data
		"""
		try:
			logger.info(f"📊 Generating pipeline analytics: {pipeline_id}")
			
			async with self.db_manager.get_connection() as conn:
				# Get pipeline basic info
				pipeline_row = await conn.fetchrow("""
					SELECT name FROM crm_sales_pipelines
					WHERE id = $1 AND tenant_id = $2
				""", pipeline_id, tenant_id)
				
				if not pipeline_row:
					raise ValueError(f"Pipeline not found: {pipeline_id}")
				
				analytics = PipelineAnalytics(
					pipeline_id=pipeline_id,
					pipeline_name=pipeline_row['name'],
					tenant_id=tenant_id,
					analysis_period_days=period_days
				)
				
				# Overall metrics
				overall_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_opportunities,
						COALESCE(SUM(value), 0) as total_value,
						COALESCE(SUM(value * probability / 100), 0) as weighted_value,
						COALESCE(AVG(value), 0) as average_deal_size
					FROM crm_opportunities o
					JOIN crm_pipeline_stages ps ON o.stage_id = ps.id
					WHERE ps.pipeline_id = $1 AND o.tenant_id = $2
					AND o.created_at >= NOW() - INTERVAL '%s days'
				""", pipeline_id, tenant_id, period_days)
				
				if overall_stats:
					analytics.total_opportunities = overall_stats['total_opportunities'] or 0
					analytics.total_value = Decimal(str(overall_stats['total_value'] or 0))
					analytics.weighted_value = Decimal(str(overall_stats['weighted_value'] or 0))
					analytics.average_deal_size = Decimal(str(overall_stats['average_deal_size'] or 0))
				
				# Win/loss rates
				win_loss_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) FILTER (WHERE ps.is_won = true) as won,
						COUNT(*) FILTER (WHERE ps.is_closed = true AND ps.is_won = false) as lost,
						COUNT(*) FILTER (WHERE ps.is_closed = true) as closed
					FROM crm_opportunities o
					JOIN crm_pipeline_stages ps ON o.stage_id = ps.id
					WHERE ps.pipeline_id = $1 AND o.tenant_id = $2
					AND o.created_at >= NOW() - INTERVAL '%s days'
				""", pipeline_id, tenant_id, period_days)
				
				if win_loss_stats and win_loss_stats['closed'] > 0:
					analytics.win_rate = (win_loss_stats['won'] / win_loss_stats['closed']) * 100
					analytics.loss_rate = (win_loss_stats['lost'] / win_loss_stats['closed']) * 100
				
				# Stage metrics
				stage_stats = await conn.fetch("""
					SELECT 
						ps.id, ps.name, ps.probability, ps.color,
						COUNT(o.id) as opportunity_count,
						COALESCE(SUM(o.value), 0) as total_value,
						COALESCE(AVG(EXTRACT(DAYS FROM (NOW() - o.stage_updated_at))), 0) as avg_duration
					FROM crm_pipeline_stages ps
					LEFT JOIN crm_opportunities o ON ps.id = o.stage_id AND o.tenant_id = $2
					WHERE ps.pipeline_id = $1 AND ps.tenant_id = $2
					GROUP BY ps.id, ps.name, ps.probability, ps.color, ps.order
					ORDER BY ps.order
				""", pipeline_id, tenant_id)
				
				analytics.stage_metrics = [
					{
						'stage_id': row['id'],
						'stage_name': row['name'],
						'probability': row['probability'],
						'color': row['color'],
						'opportunity_count': row['opportunity_count'],
						'total_value': float(row['total_value']),
						'average_duration': float(row['avg_duration'])
					}
					for row in stage_stats
				]
				
				# Velocity metrics
				velocity_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) FILTER (WHERE o.created_at >= DATE_TRUNC('month', NOW())) as created_this_month,
						COUNT(*) FILTER (WHERE ps.is_closed = true AND o.updated_at >= DATE_TRUNC('month', NOW())) as closed_this_month,
						COUNT(*) FILTER (WHERE ps.is_won = true AND o.updated_at >= DATE_TRUNC('month', NOW())) as won_this_month,
						COUNT(*) FILTER (WHERE ps.is_closed = true AND ps.is_won = false AND o.updated_at >= DATE_TRUNC('month', NOW())) as lost_this_month
					FROM crm_opportunities o
					JOIN crm_pipeline_stages ps ON o.stage_id = ps.id
					WHERE ps.pipeline_id = $1 AND o.tenant_id = $2
				""", pipeline_id, tenant_id)
				
				if velocity_stats:
					analytics.deals_created_this_month = velocity_stats['created_this_month'] or 0
					analytics.deals_closed_this_month = velocity_stats['closed_this_month'] or 0
					analytics.deals_won_this_month = velocity_stats['won_this_month'] or 0
					analytics.deals_lost_this_month = velocity_stats['lost_this_month'] or 0
				
				# Simple forecasting based on weighted pipeline
				forecast_data = await conn.fetchrow("""
					SELECT 
						COALESCE(SUM(o.value * ps.probability / 100), 0) as forecast_amount
					FROM crm_opportunities o
					JOIN crm_pipeline_stages ps ON o.stage_id = ps.id
					WHERE ps.pipeline_id = $1 AND o.tenant_id = $2
					AND ps.is_closed = false
					AND ps.allows_forecasting = true
				""", pipeline_id, tenant_id)
				
				if forecast_data:
					base_forecast = Decimal(str(forecast_data['forecast_amount'] or 0))
					analytics.forecast_this_month = base_forecast * Decimal('0.3')  # Conservative monthly forecast
					analytics.forecast_next_month = base_forecast * Decimal('0.4')
					analytics.forecast_this_quarter = base_forecast * Decimal('0.8')
					analytics.forecast_confidence = min(80.0, analytics.total_opportunities * 2)  # Simple confidence calculation
			
			logger.info(f"✅ Generated analytics for {analytics.total_opportunities} opportunities")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate pipeline analytics: {str(e)}", exc_info=True)
			raise
	
	async def get_pipeline(
		self,
		pipeline_id: str,
		tenant_id: str
	) -> Optional[SalesPipeline]:
		"""
		Get pipeline by ID
		
		Args:
			pipeline_id: Pipeline identifier
			tenant_id: Tenant identifier
			
		Returns:
			Pipeline if found
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Get pipeline data
				pipeline_row = await conn.fetchrow("""
					SELECT * FROM crm_sales_pipelines
					WHERE id = $1 AND tenant_id = $2
				""", pipeline_id, tenant_id)
				
				if not pipeline_row:
					return None
				
				# Get pipeline stages
				stage_rows = await conn.fetch("""
					SELECT * FROM crm_pipeline_stages
					WHERE pipeline_id = $1 AND tenant_id = $2
					ORDER BY "order"
				""", pipeline_id, tenant_id)
				
				# Convert to pipeline object
				pipeline_dict = dict(pipeline_row)
				pipeline_dict['stages'] = [PipelineStage(**dict(row)) for row in stage_rows]
				
				return SalesPipeline(**pipeline_dict)
				
		except Exception as e:
			logger.error(f"Failed to get pipeline: {str(e)}", exc_info=True)
			raise
	
	async def list_pipelines(
		self,
		tenant_id: str,
		active_only: bool = True
	) -> List[SalesPipeline]:
		"""
		List all pipelines for tenant
		
		Args:
			tenant_id: Tenant identifier
			active_only: Return only active pipelines
			
		Returns:
			List of pipelines
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Build query
				where_clause = "WHERE tenant_id = $1"
				params = [tenant_id]
				
				if active_only:
					where_clause += " AND is_active = true"
				
				# Get pipelines
				pipeline_rows = await conn.fetch(f"""
					SELECT * FROM crm_sales_pipelines
					{where_clause}
					ORDER BY is_default DESC, name
				""", *params)
				
				pipelines = []
				for pipeline_row in pipeline_rows:
					# Get stages for each pipeline
					stage_rows = await conn.fetch("""
						SELECT * FROM crm_pipeline_stages
						WHERE pipeline_id = $1 AND tenant_id = $2
						ORDER BY "order"
					""", pipeline_row['id'], tenant_id)
					
					pipeline_dict = dict(pipeline_row)
					pipeline_dict['stages'] = [PipelineStage(**dict(row)) for row in stage_rows]
					pipelines.append(SalesPipeline(**pipeline_dict))
				
				return pipelines
				
		except Exception as e:
			logger.error(f"Failed to list pipelines: {str(e)}", exc_info=True)
			raise
	
	async def _create_default_stages(
		self,
		pipeline: SalesPipeline,
		created_by: str
	) -> List[PipelineStage]:
		"""Create default pipeline stages"""
		default_stages = [
			{
				'name': 'Prospecting',
				'description': 'Initial contact and research phase',
				'stage_type': StageType.PROSPECTING,
				'category': StageCategory.EARLY,
				'order': 1,
				'probability': 10.0,
				'expected_duration_days': 7,
				'color': '#e74c3c'
			},
			{
				'name': 'Qualification',
				'description': 'Qualifying lead requirements and budget',
				'stage_type': StageType.QUALIFICATION,
				'category': StageCategory.EARLY,
				'order': 2,
				'probability': 25.0,
				'expected_duration_days': 10,
				'color': '#f39c12'
			},
			{
				'name': 'Needs Analysis',
				'description': 'Understanding customer needs and pain points',
				'stage_type': StageType.NEEDS_ANALYSIS,
				'category': StageCategory.MIDDLE,
				'order': 3,
				'probability': 50.0,
				'expected_duration_days': 14,
				'color': '#f1c40f'
			},
			{
				'name': 'Proposal',
				'description': 'Presenting solution and proposal',
				'stage_type': StageType.PROPOSAL,
				'category': StageCategory.MIDDLE,
				'order': 4,
				'probability': 75.0,
				'expected_duration_days': 14,
				'color': '#3498db'
			},
			{
				'name': 'Negotiation',
				'description': 'Negotiating terms and conditions',
				'stage_type': StageType.NEGOTIATION,
				'category': StageCategory.LATE,
				'order': 5,
				'probability': 85.0,
				'expected_duration_days': 7,
				'color': '#9b59b6'
			},
			{
				'name': 'Closed Won',
				'description': 'Deal successfully closed',
				'stage_type': StageType.CLOSED_WON,
				'category': StageCategory.CLOSED,
				'order': 6,
				'probability': 100.0,
				'is_closed': True,
				'is_won': True,
				'color': '#27ae60'
			},
			{
				'name': 'Closed Lost',
				'description': 'Deal lost to competitor or cancelled',
				'stage_type': StageType.CLOSED_LOST,
				'category': StageCategory.CLOSED,
				'order': 7,
				'probability': 0.0,
				'is_closed': True,
				'is_won': False,
				'allows_forecasting': False,
				'color': '#95a5a6'
			}
		]
		
		stages = []
		for stage_data in default_stages:
			stage_data.update({
				'tenant_id': pipeline.tenant_id,
				'pipeline_id': pipeline.id,
				'created_by': created_by,
				'updated_by': created_by,
				'weighted_probability': stage_data['probability'] * 0.8
			})
			stages.append(PipelineStage(**stage_data))
		
		return stages
	
	async def _store_pipeline_stage(self, stage: PipelineStage):
		"""Store pipeline stage in database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_pipeline_stages (
						id, tenant_id, pipeline_id, name, description, stage_type, category,
						"order", probability, is_active, is_closed, is_won,
						expected_duration_days, max_duration_days, automation_trigger,
						auto_advance_conditions, required_activities, required_fields,
						allows_forecasting, weighted_probability, conversion_tracking,
						enable_alerts, alert_conditions, notification_emails,
						color, icon, opportunity_count, total_value, average_duration,
						conversion_rate, metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
						$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
						$29, $30, $31, $32, $33, $34, $35, $36, $37
					)
					ON CONFLICT (id) DO UPDATE SET
						name = EXCLUDED.name,
						description = EXCLUDED.description,
						probability = EXCLUDED.probability,
						is_active = EXCLUDED.is_active,
						updated_at = NOW(),
						updated_by = EXCLUDED.updated_by,
						version = EXCLUDED.version + 1
				""", 
				stage.id, stage.tenant_id, stage.pipeline_id, stage.name, stage.description,
				stage.stage_type.value, stage.category.value, stage.order, stage.probability,
				stage.is_active, stage.is_closed, stage.is_won, stage.expected_duration_days,
				stage.max_duration_days, stage.automation_trigger.value, stage.auto_advance_conditions,
				stage.required_activities, stage.required_fields, stage.allows_forecasting,
				stage.weighted_probability, stage.conversion_tracking, stage.enable_alerts,
				stage.alert_conditions, stage.notification_emails, stage.color, stage.icon,
				stage.opportunity_count, stage.total_value, stage.average_duration,
				stage.conversion_rate, stage.metadata, stage.created_at, stage.updated_at,
				stage.created_by, stage.updated_by, stage.version
				)
		except Exception as e:
			logger.error(f"Failed to store pipeline stage: {str(e)}")
			raise
	
	async def _update_stage_statistics(self, stage_id: str, tenant_id: str):
		"""Update stage performance statistics"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Calculate stage statistics
				stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) as opportunity_count,
						COALESCE(SUM(value), 0) as total_value,
						COALESCE(AVG(EXTRACT(DAYS FROM (NOW() - stage_updated_at))), 0) as avg_duration
					FROM crm_opportunities
					WHERE stage_id = $1 AND tenant_id = $2
				""", stage_id, tenant_id)
				
				if stats:
					await conn.execute("""
						UPDATE crm_pipeline_stages SET
							opportunity_count = $3,
							total_value = $4,
							average_duration = $5,
							updated_at = NOW()
						WHERE id = $1 AND tenant_id = $2
					""", stage_id, tenant_id, stats['opportunity_count'],
					stats['total_value'], stats['avg_duration'])
					
		except Exception as e:
			logger.error(f"Failed to update stage statistics: {str(e)}")
			# Don't raise as this is not critical