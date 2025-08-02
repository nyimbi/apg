"""
APG Central Configuration - Advanced Analytics and Reporting Engine

Comprehensive analytics, business intelligence, and automated reporting
with AI-powered insights and predictive analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
from pathlib import Path
import io
import base64

# Data analysis and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Database and ORM
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.sql import extract

from ..models import CCConfiguration, CCAuditLog, CCUser, CCWorkspace
from ..service import CentralConfigurationEngine


class ReportType(Enum):
	"""Types of analytics reports."""
	USAGE_ANALYTICS = "usage_analytics"
	PERFORMANCE_ANALYTICS = "performance_analytics"
	SECURITY_ANALYTICS = "security_analytics"
	BUSINESS_INTELLIGENCE = "business_intelligence"
	CONFIGURATION_INSIGHTS = "configuration_insights"
	USER_BEHAVIOR = "user_behavior"
	TREND_ANALYSIS = "trend_analysis"
	PREDICTIVE_ANALYTICS = "predictive_analytics"


class TimeRange(Enum):
	"""Time ranges for analytics."""
	LAST_24H = "last_24h"
	LAST_7D = "last_7d"
	LAST_30D = "last_30d"
	LAST_90D = "last_90d"
	LAST_12M = "last_12m"
	CUSTOM = "custom"


class VisualizationType(Enum):
	"""Types of data visualizations."""
	LINE_CHART = "line_chart"
	BAR_CHART = "bar_chart"
	PIE_CHART = "pie_chart"
	HEATMAP = "heatmap"
	SCATTER_PLOT = "scatter_plot"
	HISTOGRAM = "histogram"
	BOX_PLOT = "box_plot"
	SUNBURST = "sunburst"
	TREEMAP = "treemap"
	SANKEY = "sankey"


@dataclass
class AnalyticsInsight:
	"""Analytics insight with AI-generated explanation."""
	insight_id: str
	title: str
	description: str
	insight_type: str  # trend, anomaly, pattern, prediction
	confidence_score: float
	impact_level: str  # high, medium, low
	supporting_data: Dict[str, Any]
	recommendations: List[str]
	generated_at: datetime


@dataclass
class ReportSection:
	"""Individual section of an analytics report."""
	section_id: str
	title: str
	description: str
	visualization_type: VisualizationType
	data: Dict[str, Any]
	insights: List[AnalyticsInsight]
	chart_config: Dict[str, Any]


@dataclass
class AnalyticsReport:
	"""Complete analytics report."""
	report_id: str
	title: str
	report_type: ReportType
	time_range: TimeRange
	generated_at: datetime
	generated_by: str
	executive_summary: str
	sections: List[ReportSection]
	key_metrics: Dict[str, Any]
	ai_insights: List[AnalyticsInsight]
	export_formats: List[str]


class AdvancedAnalyticsEngine:
	"""Advanced analytics and reporting engine."""
	
	def __init__(self, config_engine: CentralConfigurationEngine):
		"""Initialize analytics engine."""
		self.config_engine = config_engine
		self.report_cache: Dict[str, AnalyticsReport] = {}
		self.insight_models: Dict[str, Any] = {}
		
		# Set up matplotlib and seaborn styling
		plt.style.use('seaborn-v0_8')
		sns.set_palette("husl")
		
		# Initialize AI insight generation
		asyncio.create_task(self._initialize_insight_models())
	
	# ==================== Report Generation ====================
	
	async def generate_comprehensive_report(
		self,
		report_type: ReportType,
		time_range: TimeRange,
		workspace_id: Optional[str] = None,
		custom_filters: Optional[Dict[str, Any]] = None
	) -> AnalyticsReport:
		"""Generate comprehensive analytics report."""
		report_id = f"report_{uuid.uuid4().hex[:8]}"
		print(f"📊 Generating {report_type.value} report: {report_id}")
		
		# Define time boundaries
		start_time, end_time = await self._get_time_boundaries(time_range)
		
		# Initialize report structure
		report = AnalyticsReport(
			report_id=report_id,
			title=f"{report_type.value.replace('_', ' ').title()} Report",
			report_type=report_type,
			time_range=time_range,
			generated_at=datetime.now(timezone.utc),
			generated_by="analytics_engine",
			executive_summary="",
			sections=[],
			key_metrics={},
			ai_insights=[],
			export_formats=["pdf", "excel", "json"]
		)
		
		# Generate report based on type
		if report_type == ReportType.USAGE_ANALYTICS:
			await self._generate_usage_analytics(report, start_time, end_time, workspace_id)
		
		elif report_type == ReportType.PERFORMANCE_ANALYTICS:
			await self._generate_performance_analytics(report, start_time, end_time, workspace_id)
		
		elif report_type == ReportType.SECURITY_ANALYTICS:
			await self._generate_security_analytics(report, start_time, end_time, workspace_id)
		
		elif report_type == ReportType.BUSINESS_INTELLIGENCE:
			await self._generate_business_intelligence(report, start_time, end_time, workspace_id)
		
		elif report_type == ReportType.CONFIGURATION_INSIGHTS:
			await self._generate_configuration_insights(report, start_time, end_time, workspace_id)
		
		elif report_type == ReportType.USER_BEHAVIOR:
			await self._generate_user_behavior_analytics(report, start_time, end_time, workspace_id)
		
		elif report_type == ReportType.TREND_ANALYSIS:
			await self._generate_trend_analysis(report, start_time, end_time, workspace_id)
		
		elif report_type == ReportType.PREDICTIVE_ANALYTICS:
			await self._generate_predictive_analytics(report, start_time, end_time, workspace_id)
		
		# Generate AI insights
		report.ai_insights = await self._generate_ai_insights(report)
		
		# Generate executive summary
		report.executive_summary = await self._generate_executive_summary(report)
		
		# Cache report
		self.report_cache[report_id] = report
		
		print(f"✅ Report generated: {len(report.sections)} sections, {len(report.ai_insights)} insights")
		return report
	
	# ==================== Usage Analytics ====================
	
	async def _generate_usage_analytics(
		self,
		report: AnalyticsReport,
		start_time: datetime,
		end_time: datetime,
		workspace_id: Optional[str]
	):
		"""Generate usage analytics sections."""
		
		# Configuration Creation Trends
		creation_data = await self._get_configuration_creation_trends(start_time, end_time, workspace_id)
		if creation_data:
			creation_chart = await self._create_line_chart(
				creation_data,
				"Configuration Creation Trends",
				"Date",
				"Configurations Created"
			)
			
			report.sections.append(ReportSection(
				section_id="config_creation_trends",
				title="Configuration Creation Trends",
				description="Analysis of configuration creation patterns over time",
				visualization_type=VisualizationType.LINE_CHART,
				data=creation_data,
				insights=[],
				chart_config=creation_chart
			))
		
		# Most Active Users
		user_activity_data = await self._get_user_activity_analytics(start_time, end_time, workspace_id)
		if user_activity_data:
			user_chart = await self._create_bar_chart(
				user_activity_data,
				"Most Active Users",
				"User",
				"Activities"
			)
			
			report.sections.append(ReportSection(
				section_id="user_activity",
				title="User Activity Analysis",
				description="Top users by configuration management activities",
				visualization_type=VisualizationType.BAR_CHART,
				data=user_activity_data,
				insights=[],
				chart_config=user_chart
			))
		
		# Configuration Categories Distribution
		category_data = await self._get_configuration_categories(workspace_id)
		if category_data:
			category_chart = await self._create_pie_chart(
				category_data,
				"Configuration Categories Distribution"
			)
			
			report.sections.append(ReportSection(
				section_id="config_categories",
				title="Configuration Categories",
				description="Distribution of configurations by category/tag",
				visualization_type=VisualizationType.PIE_CHART,
				data=category_data,
				insights=[],
				chart_config=category_chart
			))
		
		# Update key metrics
		report.key_metrics.update({
			"total_configurations": await self._count_total_configurations(workspace_id),
			"configurations_created_period": sum(creation_data.get("values", [])) if creation_data else 0,
			"active_users_period": len(user_activity_data.get("labels", [])) if user_activity_data else 0,
			"most_popular_category": max(category_data.get("labels", []), key=lambda x: category_data["values"][category_data["labels"].index(x)]) if category_data and category_data.get("labels") else "N/A"
		})
	
	# ==================== Performance Analytics ====================
	
	async def _generate_performance_analytics(
		self,
		report: AnalyticsReport,
		start_time: datetime,
		end_time: datetime,
		workspace_id: Optional[str]
	):
		"""Generate performance analytics sections."""
		
		# Response Time Trends
		response_time_data = await self._get_response_time_trends(start_time, end_time)
		if response_time_data:
			response_chart = await self._create_line_chart(
				response_time_data,
				"API Response Time Trends",
				"Time",
				"Response Time (ms)"
			)
			
			report.sections.append(ReportSection(
				section_id="response_time_trends",
				title="API Response Time Analysis",
				description="Analysis of API response time performance over time",
				visualization_type=VisualizationType.LINE_CHART,
				data=response_time_data,
				insights=[],
				chart_config=response_chart
			))
		
		# Database Performance Metrics
		db_performance_data = await self._get_database_performance_metrics(start_time, end_time)
		if db_performance_data:
			db_chart = await self._create_multi_line_chart(
				db_performance_data,
				"Database Performance Metrics",
				"Time",
				"Value"
			)
			
			report.sections.append(ReportSection(
				section_id="database_performance",
				title="Database Performance",
				description="Database query performance and connection metrics",
				visualization_type=VisualizationType.LINE_CHART,
				data=db_performance_data,
				insights=[],
				chart_config=db_chart
			))
		
		# Cache Hit Rate Analysis
		cache_data = await self._get_cache_performance_metrics(start_time, end_time)
		if cache_data:
			cache_chart = await self._create_line_chart(
				cache_data,
				"Cache Hit Rate Trends",
				"Time",
				"Hit Rate (%)"
			)
			
			report.sections.append(ReportSection(
				section_id="cache_performance",
				title="Cache Performance Analysis",
				description="Cache hit rate and performance optimization opportunities",
				visualization_type=VisualizationType.LINE_CHART,
				data=cache_data,
				insights=[],
				chart_config=cache_chart
			))
		
		# Update key metrics
		avg_response_time = np.mean(response_time_data.get("values", [])) if response_time_data else 0
		report.key_metrics.update({
			"average_response_time_ms": round(avg_response_time, 2),
			"p95_response_time_ms": np.percentile(response_time_data.get("values", []), 95) if response_time_data else 0,
			"cache_hit_rate_avg": np.mean(cache_data.get("values", [])) if cache_data else 0,
			"performance_trend": "improving" if len(response_time_data.get("values", [])) > 1 and response_time_data["values"][-1] < response_time_data["values"][0] else "stable"
		})
	
	# ==================== AI Insight Generation ====================
	
	async def _generate_ai_insights(self, report: AnalyticsReport) -> List[AnalyticsInsight]:
		"""Generate AI-powered insights from report data."""
		insights = []
		
		# Analyze trends in key metrics
		trend_insight = await self._analyze_trends(report)
		if trend_insight:
			insights.append(trend_insight)
		
		# Detect anomalies
		anomaly_insight = await self._detect_anomalies(report)
		if anomaly_insight:
			insights.append(anomaly_insight)
		
		# Pattern recognition
		pattern_insight = await self._recognize_patterns(report)
		if pattern_insight:
			insights.append(pattern_insight)
		
		# Performance optimization opportunities
		optimization_insight = await self._identify_optimization_opportunities(report)
		if optimization_insight:
			insights.append(optimization_insight)
		
		# Usage predictions
		prediction_insight = await self._generate_usage_predictions(report)
		if prediction_insight:
			insights.append(prediction_insight)
		
		return insights
	
	async def _analyze_trends(self, report: AnalyticsReport) -> Optional[AnalyticsInsight]:
		"""Analyze trends in report data."""
		try:
			# Look for configuration creation trends
			for section in report.sections:
				if section.section_id == "config_creation_trends" and section.data:
					values = section.data.get("values", [])
					if len(values) >= 3:
						# Calculate trend direction
						recent_avg = np.mean(values[-3:])
						earlier_avg = np.mean(values[:3]) if len(values) >= 6 else np.mean(values[:-3])
						
						if recent_avg > earlier_avg * 1.2:  # 20% increase
							trend_direction = "significant increase"
							impact = "high"
						elif recent_avg > earlier_avg * 1.1:  # 10% increase
							trend_direction = "moderate increase"
							impact = "medium"
						elif recent_avg < earlier_avg * 0.8:  # 20% decrease
							trend_direction = "significant decrease"
							impact = "high"
						elif recent_avg < earlier_avg * 0.9:  # 10% decrease
							trend_direction = "moderate decrease"
							impact = "medium"
						else:
							trend_direction = "stable"
							impact = "low"
						
						return AnalyticsInsight(
							insight_id=f"trend_{uuid.uuid4().hex[:8]}",
							title=f"Configuration Creation Trend: {trend_direction.title()}",
							description=f"Analysis shows a {trend_direction} in configuration creation activity. Recent average: {recent_avg:.1f} configurations, compared to earlier period: {earlier_avg:.1f}.",
							insight_type="trend",
							confidence_score=0.85,
							impact_level=impact,
							supporting_data={
								"recent_average": recent_avg,
								"earlier_average": earlier_avg,
								"percentage_change": ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0
							},
							recommendations=[
								"Monitor trend continuation over next reporting period",
								"Investigate causes of significant changes" if impact == "high" else "Continue current monitoring approach",
								"Consider capacity planning if upward trend continues"
							],
							generated_at=datetime.now(timezone.utc)
						)
		except Exception as e:
			print(f"❌ Trend analysis failed: {e}")
		
		return None
	
	# ==================== Data Collection Methods ====================
	
	async def _get_configuration_creation_trends(
		self,
		start_time: datetime,
		end_time: datetime,
		workspace_id: Optional[str]
	) -> Dict[str, Any]:
		"""Get configuration creation trends over time."""
		async with self.config_engine.get_db_session() as session:
			query = select(
				func.date(CCConfiguration.created_at).label('date'),
				func.count(CCConfiguration.id).label('count')
			).where(
				and_(
					CCConfiguration.created_at >= start_time,
					CCConfiguration.created_at <= end_time
				)
			)
			
			if workspace_id:
				query = query.where(CCConfiguration.workspace_id == workspace_id)
			
			query = query.group_by(func.date(CCConfiguration.created_at)).order_by('date')
			
			result = await session.execute(query)
			data = result.all()
			
			if not data:
				return {}
			
			return {
				"labels": [str(row.date) for row in data],
				"values": [row.count for row in data],
				"total": sum(row.count for row in data)
			}
	
	async def _get_user_activity_analytics(
		self,
		start_time: datetime,
		end_time: datetime,
		workspace_id: Optional[str]
	) -> Dict[str, Any]:
		"""Get user activity analytics."""
		async with self.config_engine.get_db_session() as session:
			query = select(
				CCAuditLog.user_id,
				func.count(CCAuditLog.id).label('activity_count')
			).where(
				and_(
					CCAuditLog.created_at >= start_time,
					CCAuditLog.created_at <= end_time
				)
			)
			
			if workspace_id:
				query = query.where(CCAuditLog.workspace_id == workspace_id)
			
			query = query.group_by(CCAuditLog.user_id).order_by(desc('activity_count')).limit(10)
			
			result = await session.execute(query)
			data = result.all()
			
			if not data:
				return {}
			
			return {
				"labels": [f"User {row.user_id}" for row in data],
				"values": [row.activity_count for row in data]
			}
	
	async def _get_configuration_categories(self, workspace_id: Optional[str]) -> Dict[str, Any]:
		"""Get configuration categories distribution."""
		async with self.config_engine.get_db_session() as session:
			# This is a simplified approach - in reality, you'd parse tags or categories
			query = select(
				CCConfiguration.security_level,
				func.count(CCConfiguration.id).label('count')
			)
			
			if workspace_id:
				query = query.where(CCConfiguration.workspace_id == workspace_id)
			
			query = query.group_by(CCConfiguration.security_level)
			
			result = await session.execute(query)
			data = result.all()
			
			if not data:
				return {}
			
			return {
				"labels": [row.security_level.value for row in data],
				"values": [row.count for row in data]
			}
	
	# ==================== Visualization Creation ====================
	
	async def _create_line_chart(
		self,
		data: Dict[str, Any],
		title: str,
		x_label: str,
		y_label: str
	) -> Dict[str, Any]:
		"""Create line chart configuration."""
		return {
			"type": "line",
			"title": title,
			"data": {
				"labels": data.get("labels", []),
				"datasets": [{
					"label": y_label,
					"data": data.get("values", []),
					"borderColor": "rgb(75, 192, 192)",
					"backgroundColor": "rgba(75, 192, 192, 0.2)",
					"tension": 0.1
				}]
			},
			"options": {
				"responsive": True,
				"scales": {
					"x": {"title": {"display": True, "text": x_label}},
					"y": {"title": {"display": True, "text": y_label}}
				}
			}
		}
	
	async def _create_bar_chart(
		self,
		data: Dict[str, Any],
		title: str,
		x_label: str,
		y_label: str
	) -> Dict[str, Any]:
		"""Create bar chart configuration."""
		return {
			"type": "bar",
			"title": title,
			"data": {
				"labels": data.get("labels", []),
				"datasets": [{
					"label": y_label,
					"data": data.get("values", []),
					"backgroundColor": [
						"rgba(255, 99, 132, 0.6)",
						"rgba(54, 162, 235, 0.6)",
						"rgba(255, 205, 86, 0.6)",
						"rgba(75, 192, 192, 0.6)",
						"rgba(153, 102, 255, 0.6)"
					]
				}]
			},
			"options": {
				"responsive": True,
				"scales": {
					"x": {"title": {"display": True, "text": x_label}},
					"y": {"title": {"display": True, "text": y_label}}
				}
			}
		}
	
	async def _create_pie_chart(self, data: Dict[str, Any], title: str) -> Dict[str, Any]:
		"""Create pie chart configuration."""
		return {
			"type": "pie",
			"title": title,
			"data": {
				"labels": data.get("labels", []),
				"datasets": [{
					"data": data.get("values", []),
					"backgroundColor": [
						"#FF6384",
						"#36A2EB", 
						"#FFCE56",
						"#4BC0C0",
						"#9966FF",
						"#FF9F40"
					]
				}]
			},
			"options": {
				"responsive": True,
				"plugins": {
					"legend": {"position": "right"}
				}
			}
		}
	
	# ==================== Helper Methods ====================
	
	async def _get_time_boundaries(self, time_range: TimeRange) -> Tuple[datetime, datetime]:
		"""Get start and end time for time range."""
		end_time = datetime.now(timezone.utc)
		
		if time_range == TimeRange.LAST_24H:
			start_time = end_time - timedelta(hours=24)
		elif time_range == TimeRange.LAST_7D:
			start_time = end_time - timedelta(days=7)
		elif time_range == TimeRange.LAST_30D:
			start_time = end_time - timedelta(days=30)
		elif time_range == TimeRange.LAST_90D:
			start_time = end_time - timedelta(days=90)
		elif time_range == TimeRange.LAST_12M:
			start_time = end_time - timedelta(days=365)
		else:  # Default to last 30 days
			start_time = end_time - timedelta(days=30)
		
		return start_time, end_time
	
	async def _initialize_insight_models(self):
		"""Initialize AI models for insight generation."""
		# In a real implementation, this would load trained ML models
		self.insight_models = {
			"trend_analyzer": "mock_model",
			"anomaly_detector": "mock_model",
			"pattern_recognizer": "mock_model"
		}
		print("🤖 AI insight models initialized")
	
	async def _count_total_configurations(self, workspace_id: Optional[str]) -> int:
		"""Count total configurations."""
		async with self.config_engine.get_db_session() as session:
			query = select(func.count(CCConfiguration.id))
			
			if workspace_id:
				query = query.where(CCConfiguration.workspace_id == workspace_id)
			
			result = await session.execute(query)
			return result.scalar() or 0


# ==================== Factory Functions ====================

async def create_analytics_engine(
	config_engine: CentralConfigurationEngine
) -> AdvancedAnalyticsEngine:
	"""Create and initialize analytics engine."""
	engine = AdvancedAnalyticsEngine(config_engine)
	await asyncio.sleep(1)  # Allow initialization
	print("📊 Advanced Analytics Engine initialized")
	return engine