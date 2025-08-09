"""
APG Employee Data Management - Advanced Analytics Dashboard

Revolutionary analytics engine with real-time insights, predictive analytics,
and interactive visualizations for strategic workforce management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
from uuid_extensions import uuid7str

# APG Platform Integration
from ....ai_orchestration.service import AIOrchestrationService
from ....federated_learning.service import FederatedLearningService
from ....real_time_collaboration.service import CollaborationService
from .ai_intelligence_engine import EmployeeAIIntelligenceEngine
from .data_quality_engine import IntelligentDataQualityEngine


class AnalyticsTimeframe(str, Enum):
	"""Analytics timeframe options."""
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	YEARLY = "yearly"
	CUSTOM = "custom"


class MetricType(str, Enum):
	"""Types of HR metrics."""
	HEADCOUNT = "headcount"
	TURNOVER = "turnover"
	RETENTION = "retention"
	PERFORMANCE = "performance"
	ENGAGEMENT = "engagement"
	DIVERSITY = "diversity"
	COMPENSATION = "compensation"
	SKILLS = "skills"
	PRODUCTIVITY = "productivity"
	SATISFACTION = "satisfaction"


@dataclass
class AnalyticsFilter:
	"""Analytics filter configuration."""
	filter_id: str = field(default_factory=uuid7str)
	filter_name: str = ""
	filter_type: str = ""
	field_name: str = ""
	operator: str = "equals"  # equals, not_equals, contains, greater_than, less_than, between
	values: List[Any] = field(default_factory=list)
	enabled: bool = True


@dataclass
class AnalyticsMetric:
	"""Analytics metric definition."""
	metric_id: str = field(default_factory=uuid7str)
	metric_name: str = ""
	metric_type: MetricType = MetricType.HEADCOUNT
	calculation_method: str = "count"
	field_mappings: Dict[str, str] = field(default_factory=dict)
	filters: List[AnalyticsFilter] = field(default_factory=list)
	ai_enhanced: bool = False
	real_time: bool = False


@dataclass
class AnalyticsDashboardConfig:
	"""Dashboard configuration."""
	dashboard_id: str = field(default_factory=uuid7str)
	dashboard_name: str = ""
	description: str = ""
	metrics: List[AnalyticsMetric] = field(default_factory=list)
	filters: List[AnalyticsFilter] = field(default_factory=list)
	refresh_interval: int = 300  # seconds
	real_time_enabled: bool = True
	ai_insights_enabled: bool = True


@dataclass
class AnalyticsResult:
	"""Analytics calculation result."""
	metric_id: str
	metric_name: str
	value: Any
	formatted_value: str
	trend: Optional[str] = None
	trend_percentage: Optional[float] = None
	benchmark: Optional[float] = None
	ai_insights: List[str] = field(default_factory=list)
	calculation_timestamp: datetime = field(default_factory=datetime.utcnow)


class EmployeeAnalyticsDashboard:
	"""Revolutionary analytics dashboard with AI-powered insights and real-time data."""
	
	def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(f"AnalyticsDashboard.{tenant_id}")
		
		# Configuration
		self.config = config or {
			'enable_real_time': True,
			'enable_ai_insights': True,
			'cache_ttl': 300,
			'max_concurrent_queries': 10
		}
		
		# APG Service Integration
		self.ai_orchestration = AIOrchestrationService(tenant_id)
		self.federated_learning = FederatedLearningService(tenant_id)
		self.collaboration = CollaborationService(tenant_id)
		self.ai_intelligence = EmployeeAIIntelligenceEngine(tenant_id)
		self.data_quality = IntelligentDataQualityEngine(tenant_id)
		
		# Dashboard Components
		self.dashboards: Dict[str, AnalyticsDashboardConfig] = {}
		self.metrics_cache: Dict[str, Tuple[datetime, Any]] = {}
		self.real_time_subscriptions: Dict[str, List[str]] = {}
		
		# Analytics Models
		self.predictive_models: Dict[str, Any] = {}
		self.benchmark_data: Dict[str, Dict[str, float]] = {}
		
		# Performance Tracking
		self.analytics_stats = {
			'queries_executed': 0,
			'cache_hits': 0,
			'cache_misses': 0,
			'ai_insights_generated': 0
		}
		
		# Initialize dashboard
		asyncio.create_task(self._initialize_analytics_dashboard())

	async def _log_analytics_operation(self, operation: str, details: Dict[str, Any] = None) -> None:
		"""Log analytics operations for performance tracking."""
		log_details = details or {}
		self.logger.info(f"[ANALYTICS_DASHBOARD] {operation}: {log_details}")

	async def _initialize_analytics_dashboard(self) -> None:
		"""Initialize analytics dashboard components."""
		try:
			# Load default dashboard configurations
			await self._load_default_dashboards()
			
			# Initialize predictive models
			await self._initialize_predictive_models()
			
			# Load benchmark data
			await self._load_benchmark_data()
			
			# Setup real-time subscriptions
			await self._setup_real_time_subscriptions()
			
			self.logger.info("Analytics dashboard initialized successfully")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize analytics dashboard: {str(e)}")
			raise

	# ============================================================================
	# DASHBOARD MANAGEMENT
	# ============================================================================

	async def create_dashboard(self, dashboard_config: AnalyticsDashboardConfig) -> str:
		"""Create new analytics dashboard."""
		try:
			await self._log_analytics_operation("create_dashboard", {
				"dashboard_name": dashboard_config.dashboard_name,
				"metrics_count": len(dashboard_config.metrics)
			})
			
			# Validate dashboard configuration
			await self._validate_dashboard_config(dashboard_config)
			
			# Store dashboard
			self.dashboards[dashboard_config.dashboard_id] = dashboard_config
			
			# Initialize metrics for dashboard
			await self._initialize_dashboard_metrics(dashboard_config)
			
			return dashboard_config.dashboard_id
			
		except Exception as e:
			self.logger.error(f"Failed to create dashboard: {str(e)}")
			raise

	async def get_dashboard_data(self, dashboard_id: str, timeframe: AnalyticsTimeframe = AnalyticsTimeframe.MONTHLY) -> Dict[str, Any]:
		"""Get comprehensive dashboard data with AI insights."""
		try:
			if dashboard_id not in self.dashboards:
				raise ValueError(f"Dashboard not found: {dashboard_id}")
			
			dashboard_config = self.dashboards[dashboard_id]
			
			await self._log_analytics_operation("get_dashboard_data", {
				"dashboard_id": dashboard_id,
				"timeframe": timeframe
			})
			
			# Calculate all metrics
			metrics_results = []
			for metric in dashboard_config.metrics:
				result = await self._calculate_metric(metric, timeframe)
				metrics_results.append(result)
			
			# Generate AI insights for dashboard
			ai_insights = []
			if dashboard_config.ai_insights_enabled:
				ai_insights = await self._generate_dashboard_ai_insights(metrics_results, timeframe)
			
			# Get trend analysis
			trend_analysis = await self._generate_trend_analysis(metrics_results, timeframe)
			
			# Get comparative analysis
			comparative_analysis = await self._generate_comparative_analysis(metrics_results)
			
			return {
				'dashboard_id': dashboard_id,
				'dashboard_name': dashboard_config.dashboard_name,
				'metrics': [result.__dict__ for result in metrics_results],
				'ai_insights': ai_insights,
				'trend_analysis': trend_analysis,
				'comparative_analysis': comparative_analysis,
				'last_updated': datetime.utcnow().isoformat(),
				'data_quality_score': await self.data_quality.get_overall_quality_score()
			}
			
		except Exception as e:
			self.logger.error(f"Failed to get dashboard data: {str(e)}")
			raise

	# ============================================================================
	# METRICS CALCULATION ENGINE
	# ============================================================================

	async def _calculate_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> AnalyticsResult:
		"""Calculate individual metric with AI enhancement."""
		try:
			# Check cache first
			cache_key = f"{metric.metric_id}_{timeframe}_{datetime.utcnow().strftime('%Y%m%d%H')}"
			if cache_key in self.metrics_cache:
				cached_time, cached_result = self.metrics_cache[cache_key]
				if (datetime.utcnow() - cached_time).seconds < self.config['cache_ttl']:
					self.analytics_stats['cache_hits'] += 1
					return cached_result
			
			self.analytics_stats['cache_misses'] += 1
			
			# Calculate base metric value
			base_value = await self._calculate_base_metric_value(metric, timeframe)
			
			# Calculate trend
			trend, trend_percentage = await self._calculate_metric_trend(metric, timeframe)
			
			# Get benchmark
			benchmark = await self._get_metric_benchmark(metric)
			
			# Generate AI insights
			ai_insights = []
			if metric.ai_enhanced:
				ai_insights = await self._generate_metric_ai_insights(metric, base_value, trend_percentage, benchmark)
			
			# Format value
			formatted_value = await self._format_metric_value(base_value, metric.metric_type)
			
			result = AnalyticsResult(
				metric_id=metric.metric_id,
				metric_name=metric.metric_name,
				value=base_value,
				formatted_value=formatted_value,
				trend=trend,
				trend_percentage=trend_percentage,
				benchmark=benchmark,
				ai_insights=ai_insights
			)
			
			# Cache result
			self.metrics_cache[cache_key] = (datetime.utcnow(), result)
			
			self.analytics_stats['queries_executed'] += 1
			
			return result
			
		except Exception as e:
			self.logger.error(f"Failed to calculate metric {metric.metric_name}: {str(e)}")
			raise

	async def _calculate_base_metric_value(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> Any:
		"""Calculate base metric value from data."""
		# This would implement actual data queries based on metric type
		# Simplified implementation for demo
		
		if metric.metric_type == MetricType.HEADCOUNT:
			return await self._calculate_headcount_metric(metric, timeframe)
		elif metric.metric_type == MetricType.TURNOVER:
			return await self._calculate_turnover_metric(metric, timeframe)
		elif metric.metric_type == MetricType.RETENTION:
			return await self._calculate_retention_metric(metric, timeframe)
		elif metric.metric_type == MetricType.PERFORMANCE:
			return await self._calculate_performance_metric(metric, timeframe)
		elif metric.metric_type == MetricType.ENGAGEMENT:
			return await self._calculate_engagement_metric(metric, timeframe)
		elif metric.metric_type == MetricType.DIVERSITY:
			return await self._calculate_diversity_metric(metric, timeframe)
		elif metric.metric_type == MetricType.COMPENSATION:
			return await self._calculate_compensation_metric(metric, timeframe)
		elif metric.metric_type == MetricType.SKILLS:
			return await self._calculate_skills_metric(metric, timeframe)
		else:
			return 0

	async def _calculate_headcount_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> int:
		"""Calculate headcount metrics."""
		# Simulate headcount calculation
		base_headcount = 1247
		
		# Apply filters
		for filter_config in metric.filters:
			if filter_config.enabled:
				if filter_config.filter_type == "department":
					base_headcount = int(base_headcount * 0.3)  # Department subset
				elif filter_config.filter_type == "location":
					base_headcount = int(base_headcount * 0.7)  # Location subset
		
		return base_headcount

	async def _calculate_turnover_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> float:
		"""Calculate turnover rate."""
		# Simulate turnover calculation
		if timeframe == AnalyticsTimeframe.MONTHLY:
			return 0.045  # 4.5% monthly turnover
		elif timeframe == AnalyticsTimeframe.QUARTERLY:
			return 0.12   # 12% quarterly turnover
		elif timeframe == AnalyticsTimeframe.YEARLY:
			return 0.18   # 18% annual turnover
		else:
			return 0.045

	async def _calculate_retention_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> float:
		"""Calculate retention rate."""
		turnover = await self._calculate_turnover_metric(metric, timeframe)
		return 1.0 - turnover

	async def _calculate_performance_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> float:
		"""Calculate average performance rating."""
		# Simulate performance calculation
		return 3.7  # Average rating out of 5

	async def _calculate_engagement_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> float:
		"""Calculate employee engagement score."""
		# Use AI intelligence engine for engagement analysis
		return await self.ai_intelligence.get_average_engagement_score()

	async def _calculate_diversity_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> Dict[str, float]:
		"""Calculate diversity metrics."""
		return {
			'gender_diversity': 0.48,
			'ethnic_diversity': 0.35,
			'age_diversity': 0.67,
			'leadership_diversity': 0.32
		}

	async def _calculate_compensation_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> Dict[str, float]:
		"""Calculate compensation metrics."""
		return {
			'average_salary': 75000.0,
			'median_salary': 68000.0,
			'pay_equity_ratio': 0.97,
			'compensation_growth': 0.05
		}

	async def _calculate_skills_metric(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> Dict[str, Any]:
		"""Calculate skills gap metrics."""
		return await self.ai_intelligence.get_skills_gap_analysis()

	# ============================================================================
	# TREND ANALYSIS AND PREDICTIONS
	# ============================================================================

	async def _calculate_metric_trend(self, metric: AnalyticsMetric, timeframe: AnalyticsTimeframe) -> Tuple[str, float]:
		"""Calculate metric trend and percentage change."""
		try:
			# Get current and previous period values
			current_value = await self._calculate_base_metric_value(metric, timeframe)
			
			# Simulate previous period calculation
			previous_value = current_value * (0.95 + (hash(metric.metric_id) % 10) / 100)
			
			if previous_value == 0:
				return "stable", 0.0
			
			percentage_change = ((current_value - previous_value) / previous_value) * 100
			
			if percentage_change > 2:
				trend = "up"
			elif percentage_change < -2:
				trend = "down"
			else:
				trend = "stable"
			
			return trend, percentage_change
			
		except Exception as e:
			self.logger.error(f"Failed to calculate trend for {metric.metric_name}: {str(e)}")
			return "stable", 0.0

	async def _generate_trend_analysis(self, metrics_results: List[AnalyticsResult], timeframe: AnalyticsTimeframe) -> Dict[str, Any]:
		"""Generate comprehensive trend analysis."""
		try:
			trends = {
				'improving_metrics': [],
				'declining_metrics': [],
				'stable_metrics': [],
				'overall_trend': 'stable',
				'key_insights': []
			}
			
			for result in metrics_results:
				if result.trend == "up":
					trends['improving_metrics'].append({
						'name': result.metric_name,
						'change': result.trend_percentage
					})
				elif result.trend == "down":
					trends['declining_metrics'].append({
						'name': result.metric_name,
						'change': result.trend_percentage
					})
				else:
					trends['stable_metrics'].append({
						'name': result.metric_name,
						'change': result.trend_percentage
					})
			
			# Determine overall trend
			improving_count = len(trends['improving_metrics'])
			declining_count = len(trends['declining_metrics'])
			
			if improving_count > declining_count:
				trends['overall_trend'] = 'improving'
				trends['key_insights'].append(f"{improving_count} metrics showing improvement")
			elif declining_count > improving_count:
				trends['overall_trend'] = 'declining'
				trends['key_insights'].append(f"{declining_count} metrics showing decline")
			
			return trends
			
		except Exception as e:
			self.logger.error(f"Failed to generate trend analysis: {str(e)}")
			return {}

	# ============================================================================
	# AI INSIGHTS GENERATION
	# ============================================================================

	async def _generate_dashboard_ai_insights(self, metrics_results: List[AnalyticsResult], timeframe: AnalyticsTimeframe) -> List[str]:
		"""Generate AI-powered insights for entire dashboard."""
		try:
			insights_prompt = f"""
			Analyze the following HR metrics data and provide strategic insights:
			
			Metrics Data:
			{json.dumps([{
				'name': r.metric_name,
				'value': r.value,
				'trend': r.trend,
				'trend_percentage': r.trend_percentage,
				'benchmark': r.benchmark
			} for r in metrics_results], default=str, indent=2)}
			
			Timeframe: {timeframe}
			
			Provide 3-5 key strategic insights focusing on:
			1. Patterns across metrics
			2. Potential risks or opportunities
			3. Recommended actions
			4. Predictive observations
			
			Return insights as a JSON array of strings.
			"""
			
			ai_insights = await self.ai_orchestration.analyze_text_with_ai(
				prompt=insights_prompt,
				response_format="json",
				model_provider="openai"
			)
			
			if ai_insights and isinstance(ai_insights, list):
				self.analytics_stats['ai_insights_generated'] += len(ai_insights)
				return ai_insights
			
			return []
			
		except Exception as e:
			self.logger.error(f"Failed to generate dashboard AI insights: {str(e)}")
			return []

	async def _generate_metric_ai_insights(self, metric: AnalyticsMetric, value: Any, trend_percentage: float, benchmark: Optional[float]) -> List[str]:
		"""Generate AI insights for individual metric."""
		try:
			insights_prompt = f"""
			Analyze this HR metric and provide insights:
			
			Metric: {metric.metric_name}
			Type: {metric.metric_type}
			Current Value: {value}
			Trend: {trend_percentage:.1f}%
			Benchmark: {benchmark}
			
			Provide 2-3 specific insights about this metric including:
			- Performance assessment
			- Recommended actions
			- Risk/opportunity identification
			
			Return as JSON array of strings.
			"""
			
			ai_insights = await self.ai_orchestration.analyze_text_with_ai(
				prompt=insights_prompt,
				response_format="json",
				model_provider="openai"
			)
			
			if ai_insights and isinstance(ai_insights, list):
				return ai_insights
			
			return []
			
		except Exception as e:
			self.logger.error(f"Failed to generate metric AI insights: {str(e)}")
			return []

	# ============================================================================
	# BENCHMARKING AND COMPARATIVE ANALYSIS
	# ============================================================================

	async def _get_metric_benchmark(self, metric: AnalyticsMetric) -> Optional[float]:
		"""Get industry benchmark for metric."""
		# Industry benchmarks (would be loaded from external data)
		benchmarks = {
			MetricType.TURNOVER: 0.15,  # 15% annual turnover
			MetricType.RETENTION: 0.85,  # 85% retention
			MetricType.PERFORMANCE: 3.5,  # 3.5/5 average performance
			MetricType.ENGAGEMENT: 0.72   # 72% engagement
		}
		
		return benchmarks.get(metric.metric_type)

	async def _generate_comparative_analysis(self, metrics_results: List[AnalyticsResult]) -> Dict[str, Any]:
		"""Generate comparative analysis against benchmarks."""
		try:
			analysis = {
				'above_benchmark': [],
				'below_benchmark': [],
				'at_benchmark': [],
				'no_benchmark': []
			}
			
			for result in metrics_results:
				if result.benchmark is None:
					analysis['no_benchmark'].append(result.metric_name)
					continue
				
				# Convert value to float for comparison
				try:
					value = float(result.value) if not isinstance(result.value, dict) else None
				except (ValueError, TypeError):
					value = None
				
				if value is None:
					analysis['no_benchmark'].append(result.metric_name)
					continue
				
				variance = ((value - result.benchmark) / result.benchmark) * 100
				
				if variance > 5:
					analysis['above_benchmark'].append({
						'name': result.metric_name,
						'variance': variance
					})
				elif variance < -5:
					analysis['below_benchmark'].append({
						'name': result.metric_name,
						'variance': variance
					})
				else:
					analysis['at_benchmark'].append({
						'name': result.metric_name,
						'variance': variance
					})
			
			return analysis
			
		except Exception as e:
			self.logger.error(f"Failed to generate comparative analysis: {str(e)}")
			return {}

	# ============================================================================
	# REAL-TIME DATA AND SUBSCRIPTIONS
	# ============================================================================

	async def setup_real_time_metric(self, metric_id: str, update_interval: int = 60) -> None:
		"""Setup real-time monitoring for specific metric."""
		try:
			# Register real-time subscription
			subscription_id = await self.collaboration.subscribe_to_data_changes(
				data_type="employee_metrics",
				callback=self._handle_metric_update,
				filters={'metric_id': metric_id}
			)
			
			if metric_id not in self.real_time_subscriptions:
				self.real_time_subscriptions[metric_id] = []
			
			self.real_time_subscriptions[metric_id].append(subscription_id)
			
			self.logger.info(f"Real-time monitoring setup for metric: {metric_id}")
			
		except Exception as e:
			self.logger.error(f"Failed to setup real-time metric: {str(e)}")

	async def _handle_metric_update(self, update_data: Dict[str, Any]) -> None:
		"""Handle real-time metric updates."""
		try:
			metric_id = update_data.get('metric_id')
			if metric_id:
				# Clear cache for updated metric
				cache_keys_to_remove = [key for key in self.metrics_cache.keys() if key.startswith(metric_id)]
				for key in cache_keys_to_remove:
					del self.metrics_cache[key]
				
				await self._log_analytics_operation("real_time_update", {
					"metric_id": metric_id,
					"update_type": update_data.get('update_type')
				})
				
		except Exception as e:
			self.logger.error(f"Failed to handle metric update: {str(e)}")

	# ============================================================================
	# UTILITY METHODS
	# ============================================================================

	async def _format_metric_value(self, value: Any, metric_type: MetricType) -> str:
		"""Format metric value for display."""
		try:
			if isinstance(value, dict):
				return json.dumps(value, default=str)
			elif isinstance(value, float):
				if metric_type in [MetricType.TURNOVER, MetricType.RETENTION]:
					return f"{value:.1%}"
				elif metric_type == MetricType.COMPENSATION:
					return f"${value:,.2f}"
				else:
					return f"{value:.2f}"
			elif isinstance(value, int):
				return f"{value:,}"
			else:
				return str(value)
				
		except Exception:
			return str(value)

	async def _load_default_dashboards(self) -> None:
		"""Load default dashboard configurations."""
		# Create default executive dashboard
		executive_dashboard = AnalyticsDashboardConfig(
			dashboard_name="Executive Overview",
			description="High-level workforce metrics for executives",
			metrics=[
				AnalyticsMetric(
					metric_name="Total Headcount",
					metric_type=MetricType.HEADCOUNT,
					calculation_method="count",
					ai_enhanced=True,
					real_time=True
				),
				AnalyticsMetric(
					metric_name="Annual Turnover Rate",
					metric_type=MetricType.TURNOVER,
					calculation_method="percentage",
					ai_enhanced=True
				),
				AnalyticsMetric(
					metric_name="Employee Engagement",
					metric_type=MetricType.ENGAGEMENT,
					calculation_method="average",
					ai_enhanced=True
				),
				AnalyticsMetric(
					metric_name="Diversity Index",
					metric_type=MetricType.DIVERSITY,
					calculation_method="composite",
					ai_enhanced=True
				)
			],
			real_time_enabled=True,
			ai_insights_enabled=True
		)
		
		self.dashboards[executive_dashboard.dashboard_id] = executive_dashboard

	async def _initialize_predictive_models(self) -> None:
		"""Initialize predictive analytics models."""
		try:
			self.predictive_models = await self.ai_orchestration.load_models([
				"turnover_prediction_v3",
				"performance_prediction_v2",
				"engagement_prediction_v2"
			])
		except Exception as e:
			self.logger.error(f"Failed to load predictive models: {str(e)}")

	async def _load_benchmark_data(self) -> None:
		"""Load industry benchmark data."""
		# This would typically load from external benchmark providers
		self.benchmark_data = {
			'industry_averages': {
				'turnover_rate': 0.15,
				'retention_rate': 0.85,
				'engagement_score': 0.72,
				'performance_rating': 3.5
			},
			'best_in_class': {
				'turnover_rate': 0.08,
				'retention_rate': 0.92,
				'engagement_score': 0.85,
				'performance_rating': 4.2
			}
		}

	async def _setup_real_time_subscriptions(self) -> None:
		"""Setup real-time data subscriptions."""
		try:
			# Subscribe to employee data changes
			await self.collaboration.subscribe_to_data_changes(
				data_type="employee_data",
				callback=self._handle_employee_data_change
			)
			
		except Exception as e:
			self.logger.error(f"Failed to setup real-time subscriptions: {str(e)}")

	async def _handle_employee_data_change(self, change_data: Dict[str, Any]) -> None:
		"""Handle employee data changes for real-time updates."""
		# Clear relevant caches when employee data changes
		self.metrics_cache.clear()

	async def _validate_dashboard_config(self, config: AnalyticsDashboardConfig) -> None:
		"""Validate dashboard configuration."""
		if not config.dashboard_name:
			raise ValueError("Dashboard name is required")
		
		if not config.metrics:
			raise ValueError("Dashboard must have at least one metric")
		
		for metric in config.metrics:
			if not metric.metric_name:
				raise ValueError("All metrics must have names")

	async def _initialize_dashboard_metrics(self, config: AnalyticsDashboardConfig) -> None:
		"""Initialize metrics for new dashboard."""
		for metric in config.metrics:
			if metric.real_time:
				await self.setup_real_time_metric(metric.metric_id)

	async def get_analytics_statistics(self) -> Dict[str, Any]:
		"""Get analytics engine performance statistics."""
		return {
			'tenant_id': self.tenant_id,
			'dashboards_count': len(self.dashboards),
			'cached_metrics': len(self.metrics_cache),
			'real_time_subscriptions': len(self.real_time_subscriptions),
			'performance_stats': self.analytics_stats.copy(),
			'uptime': "active",
			'last_update': datetime.utcnow().isoformat()
		}