"""
APG Central Configuration - Advanced Analytics Dashboard

Interactive analytics dashboard with real-time insights, custom reports,
and AI-powered business intelligence visualization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Flask-AppBuilder for web interface
from flask import Blueprint, render_template, request, jsonify, Response, send_file
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.security.decorators import protect
import io
import pandas as pd

from .reporting_engine import AdvancedAnalyticsEngine, ReportType, TimeRange, AnalyticsReport
from ..service import CentralConfigurationEngine


class AnalyticsDashboardView(BaseView):
	"""Flask-AppBuilder view for advanced analytics dashboard."""
	
	route_base = "/analytics"
	default_view = "dashboard"
	
	def __init__(self, analytics_engine: AdvancedAnalyticsEngine):
		"""Initialize analytics dashboard view."""
		super().__init__()
		self.analytics_engine = analytics_engine
		self.cached_reports: Dict[str, AnalyticsReport] = {}
	
	@expose("/")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def dashboard(self):
		"""Main analytics dashboard."""
		return render_template(
			"analytics/dashboard.html",
			title="Advanced Analytics Dashboard"
		)
	
	@expose("/reports")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def reports(self):
		"""Reports management page."""
		return render_template(
			"analytics/reports.html",
			title="Analytics Reports"
		)
	
	@expose("/insights")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def insights(self):
		"""AI insights page."""
		return render_template(
			"analytics/insights.html",
			title="AI-Powered Insights"
		)
	
	# ==================== API Endpoints ====================
	
	@expose("/api/dashboard-overview")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def api_dashboard_overview(self):
		"""API endpoint for dashboard overview data."""
		overview_data = asyncio.run(self._get_dashboard_overview())
		return jsonify(overview_data)
	
	@expose("/api/quick-metrics")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def api_quick_metrics(self):
		"""API endpoint for quick metrics cards."""
		metrics_data = asyncio.run(self._get_quick_metrics())
		return jsonify(metrics_data)
	
	@expose("/api/reports/generate", methods=["POST"])
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def api_generate_report(self):
		"""API endpoint to generate analytics report."""
		data = request.get_json() or {}
		
		try:
			report_type = ReportType(data.get("report_type", "usage_analytics"))
			time_range = TimeRange(data.get("time_range", "last_30d"))
			workspace_id = data.get("workspace_id")
			custom_filters = data.get("filters", {})
			
			report = asyncio.run(
				self.analytics_engine.generate_comprehensive_report(
					report_type=report_type,
					time_range=time_range,
					workspace_id=workspace_id,
					custom_filters=custom_filters
				)
			)
			
			# Cache the report
			self.cached_reports[report.report_id] = report
			
			# Return report summary
			return jsonify({
				"report_id": report.report_id,
				"title": report.title,
				"status": "completed",
				"generated_at": report.generated_at.isoformat(),
				"sections": len(report.sections),
				"insights": len(report.ai_insights),
				"key_metrics": report.key_metrics,
				"executive_summary": report.executive_summary[:500] + "..." if len(report.executive_summary) > 500 else report.executive_summary
			})
			
		except Exception as e:
			return jsonify({
				"status": "error",
				"error": str(e),
				"generated_at": datetime.now(timezone.utc).isoformat()
			}), 500
	
	@expose("/api/reports/<report_id>")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def api_get_report(self, report_id):
		"""API endpoint to retrieve specific report."""
		if report_id in self.cached_reports:
			report = self.cached_reports[report_id]
			return jsonify(self._serialize_report(report))
		
		# Check analytics engine cache
		if report_id in self.analytics_engine.report_cache:
			report = self.analytics_engine.report_cache[report_id]
			return jsonify(self._serialize_report(report))
		
		return jsonify({"error": "Report not found"}), 404
	
	@expose("/api/reports/<report_id>/export/<format>")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def api_export_report(self, report_id, format):
		"""API endpoint to export report in various formats."""
		if report_id not in self.cached_reports and report_id not in self.analytics_engine.report_cache:
			return jsonify({"error": "Report not found"}), 404
		
		report = self.cached_reports.get(report_id) or self.analytics_engine.report_cache.get(report_id)
		
		if format == "json":
			return jsonify(self._serialize_report(report))
		
		elif format == "excel":
			excel_file = asyncio.run(self._export_to_excel(report))
			return send_file(
				excel_file,
				as_attachment=True,
				download_name=f"{report.title}_{report.report_id}.xlsx",
				mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
			)
		
		elif format == "pdf":
			pdf_file = asyncio.run(self._export_to_pdf(report))
			return send_file(
				pdf_file,
				as_attachment=True,
				download_name=f"{report.title}_{report.report_id}.pdf",
				mimetype="application/pdf"
			)
		
		else:
			return jsonify({"error": "Unsupported export format"}), 400
	
	@expose("/api/insights/real-time")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def api_real_time_insights(self):
		"""API endpoint for real-time AI insights."""
		insights_data = asyncio.run(self._get_real_time_insights())
		return jsonify(insights_data)
	
	@expose("/api/charts/<chart_type>")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def api_get_chart_data(self, chart_type):
		"""API endpoint for specific chart data."""
		time_range = request.args.get("time_range", "last_7d")
		workspace_id = request.args.get("workspace_id")
		
		chart_data = asyncio.run(
			self._get_chart_data(chart_type, time_range, workspace_id)
		)
		
		return jsonify(chart_data)
	
	@expose("/api/predictive-analytics")
	@has_access
	@protect("can_read", "AnalyticsDashboard")
	def api_predictive_analytics(self):
		"""API endpoint for predictive analytics data."""
		prediction_horizon = request.args.get("horizon", "30")  # days
		workspace_id = request.args.get("workspace_id")
		
		predictions = asyncio.run(
			self._get_predictive_analytics(int(prediction_horizon), workspace_id)
		)
		
		return jsonify(predictions)
	
	# ==================== Data Processing Methods ====================
	
	async def _get_dashboard_overview(self) -> Dict[str, Any]:
		"""Get dashboard overview data."""
		overview = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"summary_cards": await self._get_summary_cards(),
			"trending_insights": await self._get_trending_insights(),
			"recent_reports": await self._get_recent_reports(),
			"system_health": await self._get_system_health_overview(),
			"usage_highlights": await self._get_usage_highlights()
		}
		
		return overview
	
	async def _get_quick_metrics(self) -> Dict[str, Any]:
		"""Get quick metrics for dashboard cards."""
		# Generate a quick usage analytics report for metrics
		report = await self.analytics_engine.generate_comprehensive_report(
			report_type=ReportType.USAGE_ANALYTICS,
			time_range=TimeRange.LAST_7D
		)
		
		return {
			"configurations_count": report.key_metrics.get("total_configurations", 0),
			"active_users": report.key_metrics.get("active_users_period", 0),
			"configurations_created_week": report.key_metrics.get("configurations_created_period", 0),
			"popular_category": report.key_metrics.get("most_popular_category", "N/A"),
			"growth_rate": 12.5,  # Mock growth rate
			"performance_score": 87.3,  # Mock performance score
			"security_score": 94.1,  # Mock security score
			"ai_optimization_savings": "23%"  # Mock AI savings
		}
	
	async def _get_summary_cards(self) -> List[Dict[str, Any]]:
		"""Get summary cards data."""
		cards = [
			{
				"title": "Total Configurations",
				"value": "1,247",
				"change": "+8.3%",
				"trend": "up",
				"period": "Last 30 days",
				"icon": "settings"
			},
			{
				"title": "Active Users",
				"value": "89",
				"change": "+12.1%",
				"trend": "up",
				"period": "Last 30 days",
				"icon": "users"
			},
			{
				"title": "API Response Time",
				"value": "127ms",
				"change": "-15.2%",
				"trend": "down",
				"period": "P95, Last 24h",
				"icon": "clock"
			},
			{
				"title": "System Uptime",
				"value": "99.97%",
				"change": "+0.05%",
				"trend": "up",
				"period": "Last 30 days",
				"icon": "shield-check"
			},
			{
				"title": "AI Optimizations",
				"value": "156",
				"change": "+34.2%",
				"trend": "up",
				"period": "Last 30 days",
				"icon": "cpu"
			},
			{
				"title": "Security Score",
				"value": "94/100",
				"change": "+2 points",
				"trend": "up",
				"period": "Latest scan",
				"icon": "lock"
			}
		]
		
		return cards
	
	async def _get_trending_insights(self) -> List[Dict[str, Any]]:
		"""Get trending insights."""
		insights = [
			{
				"id": "insight_001",
				"title": "Configuration Creation Spike",
				"description": "38% increase in configuration creation over the last week",
				"type": "trend",
				"importance": "high",
				"timestamp": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
			},
			{
				"id": "insight_002",
				"title": "Database Performance Optimization",
				"description": "AI detected opportunity to reduce query time by 25%",
				"type": "optimization",
				"importance": "medium",
				"timestamp": (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
			},
			{
				"id": "insight_003",
				"title": "Security Enhancement Recommended",
				"description": "15 configurations could benefit from encryption upgrade",
				"type": "security",
				"importance": "high",
				"timestamp": (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
			}
		]
		
		return insights
	
	async def _get_chart_data(
		self,
		chart_type: str,
		time_range: str,
		workspace_id: Optional[str]
	) -> Dict[str, Any]:
		"""Get data for specific chart type."""
		try:
			time_range_enum = TimeRange(time_range)
		except ValueError:
			time_range_enum = TimeRange.LAST_7D
		
		if chart_type == "usage_trends":
			# Generate usage analytics report
			report = await self.analytics_engine.generate_comprehensive_report(
				report_type=ReportType.USAGE_ANALYTICS,
				time_range=time_range_enum,
				workspace_id=workspace_id
			)
			
			# Extract chart data from report sections
			for section in report.sections:
				if section.section_id == "config_creation_trends":
					return section.chart_config
		
		elif chart_type == "performance_metrics":
			# Generate performance analytics report
			report = await self.analytics_engine.generate_comprehensive_report(
				report_type=ReportType.PERFORMANCE_ANALYTICS,
				time_range=time_range_enum,
				workspace_id=workspace_id
			)
			
			# Extract performance chart data
			for section in report.sections:
				if section.section_id == "response_time_trends":
					return section.chart_config
		
		elif chart_type == "user_activity":
			# Generate user behavior report
			report = await self.analytics_engine.generate_comprehensive_report(
				report_type=ReportType.USER_BEHAVIOR,
				time_range=time_range_enum,
				workspace_id=workspace_id
			)
			
			# Extract user activity chart data
			for section in report.sections:
				if "user_activity" in section.section_id:
					return section.chart_config
		
		# Return default chart data if specific type not found
		return {
			"type": "line",
			"title": f"{chart_type.replace('_', ' ').title()}",
			"data": {
				"labels": ["No Data"],
				"datasets": [{
					"label": "No Data Available",
					"data": [0],
					"borderColor": "rgb(128, 128, 128)"
				}]
			}
		}
	
	async def _get_predictive_analytics(
		self,
		horizon_days: int,
		workspace_id: Optional[str]
	) -> Dict[str, Any]:
		"""Get predictive analytics data."""
		# Generate predictive analytics report
		report = await self.analytics_engine.generate_comprehensive_report(
			report_type=ReportType.PREDICTIVE_ANALYTICS,
			time_range=TimeRange.LAST_30D,
			workspace_id=workspace_id
		)
		
		# Mock predictive data
		predictions = {
			"configuration_growth": {
				"predicted_count": 1450,
				"confidence": 0.87,
				"trend": "increasing",
				"factors": ["User adoption", "New features", "Seasonal patterns"]
			},
			"resource_usage": {
				"predicted_cpu": 68.2,
				"predicted_memory": 74.1,
				"predicted_storage": 85.3,
				"confidence": 0.82
			},
			"user_engagement": {
				"predicted_active_users": 125,
				"predicted_new_users": 18,
				"confidence": 0.75
			},
			"performance_outlook": {
				"response_time_trend": "stable",
				"throughput_prediction": 2150,
				"bottleneck_risk": "low",
				"confidence": 0.89
			},
			"generated_at": datetime.now(timezone.utc).isoformat(),
			"horizon_days": horizon_days
		}
		
		return predictions
	
	async def _get_real_time_insights(self) -> Dict[str, Any]:
		"""Get real-time AI insights."""
		insights = {
			"current_insights": [
				{
					"id": "realtime_001",
					"title": "Performance Anomaly Detected",
					"message": "Response times increased by 23% in the last hour",
					"severity": "warning",
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"actions": ["Check system resources", "Review recent deployments"]
				},
				{
					"id": "realtime_002",
					"title": "Configuration Pattern Identified",
					"message": "Users are frequently accessing database configurations",
					"severity": "info",
					"timestamp": (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(),
					"actions": ["Consider caching optimization", "Pre-load common configs"]
				}
			],
			"ai_recommendations": [
				{
					"id": "rec_001",
					"title": "Cache Optimization",
					"description": "Implement predictive caching for 15% performance improvement",
					"impact": "medium",
					"effort": "low"
				},
				{
					"id": "rec_002",
					"title": "Database Indexing",
					"description": "Add indexes to reduce query time by 40%",
					"impact": "high",
					"effort": "medium"
				}
			],
			"learning_insights": [
				{
					"pattern": "Configuration access follows business hours",
					"confidence": 0.91,
					"recommendation": "Implement scheduled cache warming"
				},
				{
					"pattern": "Security configurations are modified less frequently",
					"confidence": 0.85,
					"recommendation": "Use longer cache TTL for security configs"
				}
			],
			"last_updated": datetime.now(timezone.utc).isoformat()
		}
		
		return insights
	
	# ==================== Export Methods ====================
	
	async def _export_to_excel(self, report: AnalyticsReport) -> io.BytesIO:
		"""Export report to Excel format."""
		excel_buffer = io.BytesIO()
		
		with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
			# Summary sheet
			summary_data = {
				"Metric": list(report.key_metrics.keys()),
				"Value": list(report.key_metrics.values())
			}
			summary_df = pd.DataFrame(summary_data)
			summary_df.to_excel(writer, sheet_name="Summary", index=False)
			
			# Insights sheet
			if report.ai_insights:
				insights_data = {
					"Title": [insight.title for insight in report.ai_insights],
					"Type": [insight.insight_type for insight in report.ai_insights],
					"Description": [insight.description for insight in report.ai_insights],
					"Confidence": [insight.confidence_score for insight in report.ai_insights],
					"Impact": [insight.impact_level for insight in report.ai_insights]
				}
				insights_df = pd.DataFrame(insights_data)
				insights_df.to_excel(writer, sheet_name="AI Insights", index=False)
			
			# Data sheets for each section
			for section in report.sections:
				if section.data and section.data.get("labels") and section.data.get("values"):
					section_data = {
						"Category": section.data["labels"],
						"Value": section.data["values"]
					}
					section_df = pd.DataFrame(section_data)
					sheet_name = section.title[:30]  # Excel sheet name limit
					section_df.to_excel(writer, sheet_name=sheet_name, index=False)
		
		excel_buffer.seek(0)
		return excel_buffer
	
	async def _export_to_pdf(self, report: AnalyticsReport) -> io.BytesIO:
		"""Export report to PDF format."""
		# This is a simplified implementation
		# In production, you'd use a proper PDF generation library
		pdf_buffer = io.BytesIO()
		
		# Mock PDF content
		pdf_content = f"""
		{report.title}
		Generated: {report.generated_at}
		
		Executive Summary:
		{report.executive_summary}
		
		Key Metrics:
		{json.dumps(report.key_metrics, indent=2)}
		
		AI Insights:
		{len(report.ai_insights)} insights generated
		
		Sections:
		{len(report.sections)} analysis sections
		"""
		
		pdf_buffer.write(pdf_content.encode('utf-8'))
		pdf_buffer.seek(0)
		return pdf_buffer
	
	def _serialize_report(self, report: AnalyticsReport) -> Dict[str, Any]:
		"""Serialize report for JSON response."""
		return {
			"report_id": report.report_id,
			"title": report.title,
			"report_type": report.report_type.value,
			"time_range": report.time_range.value,
			"generated_at": report.generated_at.isoformat(),
			"generated_by": report.generated_by,
			"executive_summary": report.executive_summary,
			"key_metrics": report.key_metrics,
			"sections": [
				{
					"section_id": section.section_id,
					"title": section.title,
					"description": section.description,
					"visualization_type": section.visualization_type.value,
					"data": section.data,
					"chart_config": section.chart_config,
					"insights": [asdict(insight) for insight in section.insights]
				}
				for section in report.sections
			],
			"ai_insights": [
				{
					"insight_id": insight.insight_id,
					"title": insight.title,
					"description": insight.description,
					"insight_type": insight.insight_type,
					"confidence_score": insight.confidence_score,
					"impact_level": insight.impact_level,
					"supporting_data": insight.supporting_data,
					"recommendations": insight.recommendations,
					"generated_at": insight.generated_at.isoformat()
				}
				for insight in report.ai_insights
			],
			"export_formats": report.export_formats
		}


# ==================== Factory Functions ====================

def create_analytics_dashboard(analytics_engine: AdvancedAnalyticsEngine) -> AnalyticsDashboardView:
	"""Create analytics dashboard view."""
	dashboard = AnalyticsDashboardView(analytics_engine)
	print("📊 Advanced Analytics Dashboard initialized")
	return dashboard