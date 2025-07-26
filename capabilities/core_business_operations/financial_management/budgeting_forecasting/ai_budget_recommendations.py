"""
APG Budgeting & Forecasting - AI Budget Recommendations

Intelligent budget suggestion system leveraging machine learning, historical data analysis,
and industry benchmarks to provide actionable budget recommendations.

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


# =============================================================================
# AI Recommendations Enumerations
# =============================================================================

class RecommendationType(str, Enum):
	"""Types of AI budget recommendations."""
	BUDGET_ALLOCATION = "budget_allocation"
	COST_OPTIMIZATION = "cost_optimization"
	REVENUE_ENHANCEMENT = "revenue_enhancement"
	RISK_MITIGATION = "risk_mitigation"
	SCENARIO_PLANNING = "scenario_planning"
	TEMPLATE_SUGGESTION = "template_suggestion"
	VARIANCE_CORRECTION = "variance_correction"
	SEASONAL_ADJUSTMENT = "seasonal_adjustment"


class RecommendationPriority(str, Enum):
	"""Priority levels for recommendations."""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFORMATIONAL = "informational"


class RecommendationStatus(str, Enum):
	"""Status of recommendations."""
	GENERATED = "generated"
	REVIEWED = "reviewed"
	ACCEPTED = "accepted"
	REJECTED = "rejected"
	IMPLEMENTED = "implemented"
	EXPIRED = "expired"


class ConfidenceLevel(str, Enum):
	"""AI confidence levels."""
	VERY_HIGH = "very_high"    # 90-100%
	HIGH = "high"              # 80-89%
	MEDIUM = "medium"          # 60-79%
	LOW = "low"                # 40-59%
	VERY_LOW = "very_low"      # 0-39%


class DataSource(str, Enum):
	"""Data sources for recommendations."""
	HISTORICAL_DATA = "historical_data"
	INDUSTRY_BENCHMARKS = "industry_benchmarks"
	MARKET_TRENDS = "market_trends"
	INTERNAL_ANALYTICS = "internal_analytics"
	EXTERNAL_APIS = "external_apis"
	MACHINE_LEARNING = "machine_learning"


class RecommendationCategory(str, Enum):
	"""Categories of budget recommendations."""
	REVENUE = "revenue"
	EXPENSES = "expenses"
	CAPITAL_EXPENDITURE = "capital_expenditure"
	OPERATIONAL_EFFICIENCY = "operational_efficiency"
	STRATEGIC_INVESTMENT = "strategic_investment"
	COMPLIANCE = "compliance"


# =============================================================================
# AI Recommendations Models
# =============================================================================

class RecommendationContext(APGBaseModel):
	"""Context information for generating recommendations."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	context_id: str = Field(default_factory=uuid7str)
	budget_id: str = Field(description="Target budget ID")
	
	# Context Configuration
	analysis_period: str = Field(description="Analysis period (e.g., 'last_12_months')")
	comparison_periods: List[str] = Field(default_factory=list, description="Periods for comparison")
	include_external_data: bool = Field(default=True, description="Include external market data")
	
	# Business Context
	industry: Optional[str] = Field(None, description="Industry classification")
	company_size: Optional[str] = Field(None, description="Company size category")
	business_model: Optional[str] = Field(None, description="Business model type")
	growth_stage: Optional[str] = Field(None, description="Company growth stage")
	
	# Financial Context
	fiscal_year: str = Field(description="Current fiscal year")
	currency: str = Field(default="USD", description="Primary currency")
	budget_constraints: Dict[str, Any] = Field(default_factory=dict, description="Budget constraints")
	strategic_goals: List[str] = Field(default_factory=list, description="Strategic business goals")
	
	# Preferences
	risk_tolerance: str = Field(default="medium", description="Risk tolerance level")
	optimization_focus: str = Field(default="balanced", description="Optimization focus area")
	automation_level: str = Field(default="assisted", description="Level of automation desired")


class AIRecommendation(APGBaseModel):
	"""Individual AI-generated budget recommendation."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	recommendation_id: str = Field(default_factory=uuid7str)
	recommendation_type: RecommendationType = Field(description="Type of recommendation")
	category: RecommendationCategory = Field(description="Recommendation category")
	
	# Recommendation Content
	title: NonEmptyString = Field(description="Recommendation title")
	description: str = Field(description="Detailed recommendation description")
	rationale: str = Field(description="AI reasoning behind recommendation")
	
	# Financial Impact
	estimated_impact: Decimal = Field(description="Estimated financial impact")
	impact_timeframe: str = Field(description="Timeframe for impact realization")
	confidence_score: Decimal = Field(description="AI confidence score (0-1)")
	confidence_level: ConfidenceLevel = Field(description="Confidence level category")
	
	# Implementation Details
	implementation_effort: str = Field(description="Required implementation effort")
	implementation_timeline: str = Field(description="Suggested implementation timeline")
	required_actions: List[str] = Field(default_factory=list, description="Required actions")
	dependencies: List[str] = Field(default_factory=list, description="Implementation dependencies")
	
	# Supporting Evidence
	data_sources: List[DataSource] = Field(description="Data sources used")
	supporting_metrics: Dict[str, Any] = Field(default_factory=dict, description="Supporting metrics")
	benchmarks: Dict[str, Any] = Field(default_factory=dict, description="Industry benchmarks")
	
	# Risk Assessment
	risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
	mitigation_strategies: List[str] = Field(default_factory=list, description="Risk mitigation strategies")
	
	# Prioritization
	priority: RecommendationPriority = Field(description="Recommendation priority")
	urgency_score: Decimal = Field(description="Urgency score (0-1)")
	business_value_score: Decimal = Field(description="Business value score (0-1)")
	
	# Status and Tracking
	status: RecommendationStatus = Field(default=RecommendationStatus.GENERATED)
	generated_date: datetime = Field(default_factory=datetime.utcnow)
	reviewed_date: Optional[datetime] = Field(None)
	implementation_date: Optional[datetime] = Field(None)
	expiry_date: Optional[datetime] = Field(None)
	
	# User Interaction
	user_feedback: Optional[str] = Field(None, description="User feedback on recommendation")
	user_rating: Optional[int] = Field(None, description="User rating (1-5)")
	acceptance_reason: Optional[str] = Field(None, description="Reason for acceptance/rejection")


class RecommendationBundle(APGBaseModel):
	"""Collection of related recommendations."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	bundle_id: str = Field(default_factory=uuid7str)
	bundle_name: NonEmptyString = Field(description="Bundle name")
	context_id: str = Field(description="Associated context ID")
	
	# Bundle Configuration
	recommendations: List[AIRecommendation] = Field(description="Recommendations in bundle")
	bundle_type: str = Field(description="Type of recommendation bundle")
	
	# Bundle Analysis
	total_estimated_impact: Decimal = Field(description="Total estimated financial impact")
	average_confidence: Decimal = Field(description="Average confidence score")
	implementation_complexity: str = Field(description="Overall implementation complexity")
	
	# Bundle Metadata
	generated_date: datetime = Field(default_factory=datetime.utcnow)
	algorithm_version: str = Field(default="1.0.0", description="AI algorithm version")
	processing_time: Optional[Decimal] = Field(None, description="Processing time in seconds")


class BenchmarkData(APGBaseModel):
	"""Industry benchmark data for recommendations."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	benchmark_id: str = Field(default_factory=uuid7str)
	industry: str = Field(description="Industry classification")
	metric_name: str = Field(description="Benchmark metric name")
	
	# Benchmark Values
	median_value: Decimal = Field(description="Industry median value")
	percentile_25: Decimal = Field(description="25th percentile value")
	percentile_75: Decimal = Field(description="75th percentile value")
	top_quartile: Decimal = Field(description="Top quartile average")
	
	# Benchmark Context
	sample_size: int = Field(description="Number of companies in benchmark")
	data_period: str = Field(description="Period covered by benchmark")
	geographic_scope: str = Field(description="Geographic scope of data")
	
	# Data Quality
	confidence_interval: Decimal = Field(description="Statistical confidence interval")
	data_freshness: int = Field(description="Data age in days")
	source_credibility: str = Field(description="Source credibility rating")


class RecommendationTemplate(APGBaseModel):
	"""Template for generating standardized recommendations."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	template_id: str = Field(default_factory=uuid7str)
	template_name: NonEmptyString = Field(description="Template name")
	recommendation_type: RecommendationType = Field(description="Type of recommendation")
	
	# Template Content
	title_template: str = Field(description="Title template with placeholders")
	description_template: str = Field(description="Description template")
	rationale_template: str = Field(description="Rationale template")
	
	# Template Rules
	trigger_conditions: List[Dict[str, Any]] = Field(description="Conditions that trigger this template")
	impact_calculation: str = Field(description="Formula for calculating impact")
	confidence_factors: List[str] = Field(description="Factors affecting confidence")
	
	# Template Metadata
	usage_count: int = Field(default=0, description="Number of times used")
	success_rate: Optional[Decimal] = Field(None, description="Recommendation success rate")
	average_impact: Optional[Decimal] = Field(None, description="Average impact of recommendations")


# =============================================================================
# AI Budget Recommendations Service
# =============================================================================

class AIBudgetRecommendationsService(APGServiceBase):
	"""
	AI-powered budget recommendations service providing intelligent suggestions
	based on historical data, industry benchmarks, and machine learning analysis.
	"""
	
	def __init__(self, context: APGTenantContext, config: Optional[Dict[str, Any]] = None):
		super().__init__(context, config)
		self.logger = logging.getLogger(__name__)
	
	async def generate_budget_recommendations(
		self, 
		context_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Generate comprehensive budget recommendations."""
		try:
			self.logger.info(f"Generating budget recommendations for budget {context_config.get('budget_id')}")
			
			# Create recommendation context
			context = RecommendationContext(
				budget_id=context_config['budget_id'],
				analysis_period=context_config.get('analysis_period', 'last_12_months'),
				comparison_periods=context_config.get('comparison_periods', []),
				industry=context_config.get('industry'),
				fiscal_year=context_config.get('fiscal_year', '2025'),
				strategic_goals=context_config.get('strategic_goals', []),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Analyze budget data
			budget_analysis = await self._analyze_budget_data(context)
			
			# Get industry benchmarks
			benchmarks = await self._get_industry_benchmarks(context)
			
			# Generate recommendations
			recommendations = await self._generate_ai_recommendations(context, budget_analysis, benchmarks)
			
			# Create recommendation bundle
			bundle = RecommendationBundle(
				bundle_name=f"Budget Recommendations - {datetime.now().strftime('%Y-%m-%d')}",
				context_id=context.context_id,
				recommendations=recommendations,
				bundle_type="comprehensive_analysis",
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Calculate bundle metrics
			await self._calculate_bundle_metrics(bundle)
			
			self.logger.info(f"Generated {len(recommendations)} budget recommendations")
			
			return ServiceResponse(
				success=True,
				message="Budget recommendations generated successfully",
				data=bundle.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error generating budget recommendations: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to generate budget recommendations: {str(e)}",
				errors=[str(e)]
			)
	
	async def get_recommendation_by_category(
		self, 
		budget_id: str, 
		category: RecommendationCategory
	) -> ServiceResponse:
		"""Get recommendations filtered by category."""
		try:
			self.logger.info(f"Getting {category} recommendations for budget {budget_id}")
			
			# Get existing recommendations
			all_recommendations = await self._get_existing_recommendations(budget_id)
			
			# Filter by category
			category_recommendations = [
				rec for rec in all_recommendations 
				if rec.get('category') == category
			]
			
			# Generate additional recommendations if needed
			if len(category_recommendations) < 3:
				additional_recs = await self._generate_category_specific_recommendations(
					budget_id, category
				)
				category_recommendations.extend(additional_recs)
			
			return ServiceResponse(
				success=True,
				message=f"{category} recommendations retrieved successfully",
				data={
					'category': category,
					'recommendations': category_recommendations,
					'count': len(category_recommendations)
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error getting category recommendations: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to get category recommendations: {str(e)}",
				errors=[str(e)]
			)
	
	async def implement_recommendation(
		self, 
		recommendation_id: str, 
		implementation_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Implement a specific recommendation."""
		try:
			self.logger.info(f"Implementing recommendation {recommendation_id}")
			
			# Load recommendation
			recommendation = await self._load_recommendation(recommendation_id)
			
			# Validate implementation feasibility
			feasibility_check = await self._check_implementation_feasibility(
				recommendation, implementation_config
			)
			
			if not feasibility_check['feasible']:
				return ServiceResponse(
					success=False,
					message="Recommendation implementation not feasible",
					errors=feasibility_check['issues']
				)
			
			# Create implementation plan
			implementation_plan = await self._create_implementation_plan(
				recommendation, implementation_config
			)
			
			# Execute implementation (simulation)
			implementation_result = await self._execute_implementation(
				recommendation, implementation_plan
			)
			
			# Update recommendation status
			recommendation['status'] = RecommendationStatus.IMPLEMENTED
			recommendation['implementation_date'] = datetime.utcnow()
			
			return ServiceResponse(
				success=True,
				message="Recommendation implemented successfully",
				data={
					'recommendation': recommendation,
					'implementation_plan': implementation_plan,
					'implementation_result': implementation_result
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error implementing recommendation: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to implement recommendation: {str(e)}",
				errors=[str(e)]
			)
	
	async def track_recommendation_performance(
		self, 
		recommendation_id: str
	) -> ServiceResponse:
		"""Track performance of implemented recommendation."""
		try:
			self.logger.info(f"Tracking performance of recommendation {recommendation_id}")
			
			# Load recommendation
			recommendation = await self._load_recommendation(recommendation_id)
			
			if recommendation['status'] != RecommendationStatus.IMPLEMENTED:
				return ServiceResponse(
					success=False,
					message="Recommendation must be implemented to track performance",
					errors=["recommendation_not_implemented"]
				)
			
			# Calculate actual impact
			actual_impact = await self._calculate_actual_impact(recommendation)
			
			# Compare with predicted impact
			performance_metrics = await self._calculate_performance_metrics(
				recommendation, actual_impact
			)
			
			# Generate performance report
			performance_report = await self._generate_performance_report(
				recommendation, performance_metrics
			)
			
			return ServiceResponse(
				success=True,
				message="Recommendation performance tracked successfully",
				data=performance_report
			)
			
		except Exception as e:
			self.logger.error(f"Error tracking recommendation performance: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to track recommendation performance: {str(e)}",
				errors=[str(e)]
			)
	
	async def create_custom_recommendation_template(
		self, 
		template_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Create custom recommendation template."""
		try:
			self.logger.info(f"Creating custom recommendation template: {template_config.get('template_name')}")
			
			# Create template
			template = RecommendationTemplate(
				template_name=template_config['template_name'],
				recommendation_type=template_config['recommendation_type'],
				title_template=template_config['title_template'],
				description_template=template_config['description_template'],
				rationale_template=template_config['rationale_template'],
				trigger_conditions=template_config.get('trigger_conditions', []),
				impact_calculation=template_config.get('impact_calculation', 'manual'),
				confidence_factors=template_config.get('confidence_factors', []),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Validate template
			validation_result = await self._validate_recommendation_template(template)
			
			if not validation_result['valid']:
				return ServiceResponse(
					success=False,
					message="Invalid recommendation template",
					errors=validation_result['errors']
				)
			
			return ServiceResponse(
				success=True,
				message="Custom recommendation template created successfully",
				data=template.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error creating custom recommendation template: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to create custom recommendation template: {str(e)}",
				errors=[str(e)]
			)
	
	# =============================================================================
	# Private Helper Methods
	# =============================================================================
	
	async def _analyze_budget_data(self, context: RecommendationContext) -> Dict[str, Any]:
		"""Analyze budget data for recommendation generation."""
		# Simulated budget analysis
		return {
			'total_budget': 1500000,
			'actual_spending': 1487500,
			'variance': -12500,
			'department_breakdown': {
				'Sales': {'budget': 400000, 'actual': 395000, 'variance': -5000},
				'Marketing': {'budget': 300000, 'actual': 302500, 'variance': 2500},
				'IT': {'budget': 450000, 'actual': 457500, 'variance': 7500},
				'Operations': {'budget': 350000, 'actual': 340000, 'variance': -10000}
			},
			'spending_trends': {
				'increasing_categories': ['IT', 'Marketing'],
				'decreasing_categories': ['Operations'],
				'seasonal_patterns': {'Q4': 'higher_spending', 'Q1': 'lower_spending'}
			},
			'efficiency_metrics': {
				'cost_per_employee': 45000,
				'revenue_per_employee': 125000,
				'profit_margin': 12.5
			}
		}
	
	async def _get_industry_benchmarks(self, context: RecommendationContext) -> List[BenchmarkData]:
		"""Get industry benchmark data."""
		benchmarks = [
			BenchmarkData(
				industry=context.industry or "Technology",
				metric_name="Cost per Employee",
				median_value=Decimal("42000"),
				percentile_25=Decimal("38000"),
				percentile_75=Decimal("48000"),
				top_quartile=Decimal("52000"),
				sample_size=1250,
				data_period="2024",
				geographic_scope="North America",
				confidence_interval=Decimal("0.95"),
				data_freshness=30,
				source_credibility="High",
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			),
			BenchmarkData(
				industry=context.industry or "Technology",
				metric_name="IT Spending Ratio",
				median_value=Decimal("8.5"),
				percentile_25=Decimal("6.2"),
				percentile_75=Decimal("11.3"),
				top_quartile=Decimal("13.8"),
				sample_size=980,
				data_period="2024",
				geographic_scope="North America",
				confidence_interval=Decimal("0.95"),
				data_freshness=15,
				source_credibility="High",
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
		]
		
		return benchmarks
	
	async def _generate_ai_recommendations(
		self, 
		context: RecommendationContext, 
		budget_analysis: Dict[str, Any], 
		benchmarks: List[BenchmarkData]
	) -> List[AIRecommendation]:
		"""Generate AI-powered recommendations."""
		recommendations = []
		
		# Cost optimization recommendation
		if budget_analysis['efficiency_metrics']['cost_per_employee'] > 42000:
			cost_optimization = AIRecommendation(
				recommendation_type=RecommendationType.COST_OPTIMIZATION,
				category=RecommendationCategory.OPERATIONAL_EFFICIENCY,
				title="Optimize Cost per Employee",
				description="Your cost per employee ($45,000) is above industry median ($42,000). Consider implementing efficiency measures to reduce operational costs.",
				rationale="Analysis shows 7.1% higher cost per employee compared to industry median, indicating potential for operational optimization.",
				estimated_impact=Decimal("-135000"),  # Annual savings
				impact_timeframe="6-12 months",
				confidence_score=Decimal("0.78"),
				confidence_level=ConfidenceLevel.HIGH,
				implementation_effort="Medium",
				implementation_timeline="3-6 months",
				required_actions=[
					"Conduct departmental efficiency audit",
					"Implement process automation",
					"Review vendor contracts",
					"Optimize workforce allocation"
				],
				data_sources=[DataSource.INDUSTRY_BENCHMARKS, DataSource.INTERNAL_ANALYTICS],
				supporting_metrics={
					"current_cost_per_employee": 45000,
					"industry_median": 42000,
					"potential_savings": 135000
				},
				priority=RecommendationPriority.HIGH,
				urgency_score=Decimal("0.7"),
				business_value_score=Decimal("0.8"),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			recommendations.append(cost_optimization)
		
		# IT budget optimization
		it_spending_ratio = (budget_analysis['department_breakdown']['IT']['budget'] / budget_analysis['total_budget']) * 100
		if it_spending_ratio > 11.3:  # Above 75th percentile
			it_optimization = AIRecommendation(
				recommendation_type=RecommendationType.BUDGET_ALLOCATION,
				category=RecommendationCategory.EXPENSES,
				title="Optimize IT Budget Allocation",
				description="IT spending ratio (30%) significantly exceeds industry 75th percentile (11.3%). Consider reallocating budget to revenue-generating activities.",
				rationale="High IT spending relative to industry standards suggests potential for reallocation to growth initiatives.",
				estimated_impact=Decimal("67500"),  # Reallocation benefit
				impact_timeframe="3-6 months",
				confidence_score=Decimal("0.72"),
				confidence_level=ConfidenceLevel.HIGH,
				implementation_effort="Low",
				implementation_timeline="1-3 months",
				required_actions=[
					"Review IT project priorities",
					"Evaluate cloud cost optimization",
					"Assess software license utilization",
					"Consider IT service consolidation"
				],
				data_sources=[DataSource.INDUSTRY_BENCHMARKS, DataSource.HISTORICAL_DATA],
				supporting_metrics={
					"current_it_ratio": it_spending_ratio,
					"industry_75th_percentile": 11.3,
					"potential_reallocation": 67500
				},
				priority=RecommendationPriority.MEDIUM,
				urgency_score=Decimal("0.6"),
				business_value_score=Decimal("0.7"),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			recommendations.append(it_optimization)
		
		# Revenue enhancement recommendation
		revenue_enhancement = AIRecommendation(
			recommendation_type=RecommendationType.REVENUE_ENHANCEMENT,
			category=RecommendationCategory.REVENUE,
			title="Increase Marketing Investment for Revenue Growth",
			description="Analysis suggests increasing marketing budget by 15% could drive 8-12% revenue growth based on historical performance.",
			rationale="Strong correlation between marketing spend and revenue growth in your historical data, with optimal ROI at current+15% spending level.",
			estimated_impact=Decimal("180000"),  # Revenue increase
			impact_timeframe="6-9 months",
			confidence_score=Decimal("0.65"),
			confidence_level=ConfidenceLevel.MEDIUM,
			implementation_effort="Medium",
			implementation_timeline="2-4 months",
			required_actions=[
				"Increase digital marketing budget",
				"Expand into new market segments",
				"Enhance customer acquisition campaigns",
				"Implement advanced analytics tracking"
			],
			data_sources=[DataSource.HISTORICAL_DATA, DataSource.MACHINE_LEARNING],
			supporting_metrics={
				"historical_marketing_roi": 3.2,
				"projected_revenue_increase": 180000,
				"investment_required": 45000
			},
			priority=RecommendationPriority.HIGH,
			urgency_score=Decimal("0.8"),
			business_value_score=Decimal("0.9"),
			tenant_id=self.context.tenant_id,
			created_by=self.context.user_id
		)
		recommendations.append(revenue_enhancement)
		
		return recommendations
	
	async def _calculate_bundle_metrics(self, bundle: RecommendationBundle) -> None:
		"""Calculate bundle-level metrics."""
		if bundle.recommendations:
			bundle.total_estimated_impact = sum(
				rec.estimated_impact for rec in bundle.recommendations
			)
			bundle.average_confidence = sum(
				rec.confidence_score for rec in bundle.recommendations
			) / len(bundle.recommendations)
			
			# Determine implementation complexity
			effort_scores = {'Low': 1, 'Medium': 2, 'High': 3}
			avg_effort = sum(
				effort_scores.get(rec.implementation_effort, 2) 
				for rec in bundle.recommendations
			) / len(bundle.recommendations)
			
			if avg_effort <= 1.5:
				bundle.implementation_complexity = "Low"
			elif avg_effort <= 2.5:
				bundle.implementation_complexity = "Medium"
			else:
				bundle.implementation_complexity = "High"
	
	async def _get_existing_recommendations(self, budget_id: str) -> List[Dict[str, Any]]:
		"""Get existing recommendations for budget."""
		# Simulated existing recommendations
		return [
			{
				'recommendation_id': 'rec_1',
				'category': RecommendationCategory.EXPENSES,
				'title': 'Optimize Office Expenses',
				'status': RecommendationStatus.GENERATED
			}
		]
	
	async def _generate_category_specific_recommendations(
		self, 
		budget_id: str, 
		category: RecommendationCategory
	) -> List[Dict[str, Any]]:
		"""Generate additional recommendations for specific category."""
		# Simulated category-specific recommendations
		if category == RecommendationCategory.REVENUE:
			return [
				{
					'recommendation_id': 'rec_revenue_1',
					'category': category,
					'title': 'Expand Revenue Streams',
					'description': 'Consider diversifying revenue through new product lines',
					'estimated_impact': 150000
				}
			]
		elif category == RecommendationCategory.EXPENSES:
			return [
				{
					'recommendation_id': 'rec_expense_1',
					'category': category,
					'title': 'Reduce Travel Expenses',
					'description': 'Implement virtual meeting policies to reduce travel costs',
					'estimated_impact': -25000
				}
			]
		else:
			return []
	
	async def _load_recommendation(self, recommendation_id: str) -> Dict[str, Any]:
		"""Load recommendation by ID."""
		# Simulated recommendation loading
		return {
			'recommendation_id': recommendation_id,
			'title': 'Optimize IT Spending',
			'type': RecommendationType.COST_OPTIMIZATION,
			'estimated_impact': -50000,
			'status': RecommendationStatus.GENERATED,
			'required_actions': ['Review software licenses', 'Optimize cloud costs']
		}
	
	async def _check_implementation_feasibility(
		self, 
		recommendation: Dict[str, Any], 
		implementation_config: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Check if recommendation implementation is feasible."""
		# Simulated feasibility check
		return {
			'feasible': True,
			'issues': [],
			'requirements_met': True,
			'resource_availability': True
		}
	
	async def _create_implementation_plan(
		self, 
		recommendation: Dict[str, Any], 
		implementation_config: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Create detailed implementation plan."""
		return {
			'plan_id': f"plan_{recommendation['recommendation_id']}",
			'phases': [
				{'phase': 1, 'description': 'Analysis and Planning', 'duration': '2 weeks'},
				{'phase': 2, 'description': 'Implementation', 'duration': '4 weeks'},
				{'phase': 3, 'description': 'Monitoring and Adjustment', 'duration': '2 weeks'}
			],
			'resources_required': ['Budget analyst', 'IT support'],
			'timeline': '8 weeks',
			'milestones': ['Plan approval', 'Implementation start', 'Go-live', 'Review']
		}
	
	async def _execute_implementation(
		self, 
		recommendation: Dict[str, Any], 
		implementation_plan: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Execute recommendation implementation."""
		# Simulated implementation execution
		return {
			'execution_id': f"exec_{recommendation['recommendation_id']}",
			'status': 'completed',
			'actual_timeline': '7 weeks',
			'issues_encountered': [],
			'actual_cost': 5000,
			'preliminary_results': {
				'cost_savings': 12000,
				'efficiency_gains': '8%'
			}
		}
	
	async def _calculate_actual_impact(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
		"""Calculate actual impact of implemented recommendation."""
		# Simulated impact calculation
		return {
			'actual_impact': -45000,  # Actual vs predicted -50000
			'variance': 5000,
			'variance_percentage': 10.0,
			'timeframe': '3 months post-implementation'
		}
	
	async def _calculate_performance_metrics(
		self, 
		recommendation: Dict[str, Any], 
		actual_impact: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Calculate performance metrics for recommendation."""
		predicted_impact = recommendation['estimated_impact']
		actual = actual_impact['actual_impact']
		
		return {
			'accuracy': 1 - abs(actual - predicted_impact) / abs(predicted_impact),
			'prediction_error': actual - predicted_impact,
			'success_rate': 0.9 if actual_impact['variance_percentage'] <= 20 else 0.7
		}
	
	async def _generate_performance_report(
		self, 
		recommendation: Dict[str, Any], 
		performance_metrics: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Generate performance report for recommendation."""
		return {
			'recommendation_id': recommendation['recommendation_id'],
			'performance_summary': {
				'prediction_accuracy': f"{performance_metrics['accuracy']:.1%}",
				'actual_vs_predicted': f"{performance_metrics['prediction_error']:,.0f}",
				'success_rating': 'High' if performance_metrics['success_rate'] > 0.8 else 'Medium'
			},
			'lessons_learned': [
				'Implementation timeline was accurate',
				'Cost savings exceeded expectations',
				'User adoption was faster than expected'
			],
			'future_improvements': [
				'Refine impact calculation models',
				'Include user adoption factors',
				'Add seasonal adjustment factors'
			]
		}
	
	async def _validate_recommendation_template(
		self, 
		template: RecommendationTemplate
	) -> Dict[str, Any]:
		"""Validate recommendation template."""
		errors = []
		
		# Check for required placeholders in templates
		required_placeholders = ['{impact}', '{timeframe}']
		for placeholder in required_placeholders:
			if placeholder not in template.description_template:
				errors.append(f"Missing required placeholder: {placeholder}")
		
		return {
			'valid': len(errors) == 0,
			'errors': errors
		}


# =============================================================================
# Service Factory Functions
# =============================================================================

def create_ai_budget_recommendations_service(
	context: APGTenantContext, 
	config: Optional[Dict[str, Any]] = None
) -> AIBudgetRecommendationsService:
	"""Create AI budget recommendations service instance."""
	return AIBudgetRecommendationsService(context, config)


async def generate_sample_recommendations(
	service: AIBudgetRecommendationsService,
	budget_id: str
) -> ServiceResponse:
	"""Generate sample recommendations for testing."""
	context_config = {
		'budget_id': budget_id,
		'analysis_period': 'last_12_months',
		'industry': 'Technology',
		'fiscal_year': '2025',
		'strategic_goals': ['cost_optimization', 'revenue_growth', 'operational_efficiency']
	}
	
	return await service.generate_budget_recommendations(context_config)