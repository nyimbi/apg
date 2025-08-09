#!/usr/bin/env python3
"""APG Cash Management - Intelligent Dashboard Manager

AI-powered dashboard management system with adaptive layouts,
personalized insights, and real-time optimization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardRole(str, Enum):
	"""Dashboard user roles."""
	EXECUTIVE = "executive"
	MANAGER = "manager"
	ANALYST = "analyst"
	OPERATOR = "operator"
	VIEWER = "viewer"

class WidgetType(str, Enum):
	"""Dashboard widget types."""
	KPI_CARD = "kpi_card"
	CHART = "chart"
	TABLE = "table"
	ALERT = "alert"
	TEXT = "text"
	FILTER = "filter"
	ACTION_BUTTON = "action_button"
	EMBEDDED_REPORT = "embedded_report"
	LIVE_FEED = "live_feed"

class LayoutStrategy(str, Enum):
	"""Dashboard layout strategies."""
	FIXED_GRID = "fixed_grid"
	RESPONSIVE_GRID = "responsive_grid"
	MASONRY = "masonry"
	FLOW = "flow"
	ADAPTIVE = "adaptive"
	PERSONALIZED = "personalized"

class PersonalizationLevel(str, Enum):
	"""Personalization levels."""
	NONE = "none"
	BASIC = "basic"
	ADAPTIVE = "adaptive"
	AI_POWERED = "ai_powered"

@dataclass
class UserPreferences:
	"""User dashboard preferences."""
	user_id: str
	role: DashboardRole
	preferred_charts: List[str] = field(default_factory=list)
	layout_preference: LayoutStrategy = LayoutStrategy.RESPONSIVE_GRID
	color_scheme: str = "corporate"
	refresh_interval: int = 60
	auto_alerts: bool = True
	drill_down_enabled: bool = True
	export_preferences: Dict[str, Any] = field(default_factory=dict)
	last_updated: datetime = field(default_factory=datetime.now)

class DashboardWidget(BaseModel):
	"""Dashboard widget configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	widget_id: str = Field(default_factory=uuid7str)
	widget_type: WidgetType
	title: str
	position: Dict[str, int]  # x, y, width, height
	data_source: str
	configuration: Dict[str, Any] = Field(default_factory=dict)
	permissions: List[DashboardRole] = Field(default_factory=list)
	refresh_interval_seconds: int = Field(default=60, ge=10)
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)

class SmartDashboard(BaseModel):
	"""Smart dashboard configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	dashboard_id: str = Field(default_factory=uuid7str)
	title: str
	description: Optional[str] = None
	category: str = "cash_management"
	target_roles: List[DashboardRole] = Field(default_factory=list)
	layout_strategy: LayoutStrategy = LayoutStrategy.ADAPTIVE
	personalization_level: PersonalizationLevel = PersonalizationLevel.AI_POWERED
	widgets: List[DashboardWidget] = Field(default_factory=list)
	filters: List[Dict[str, Any]] = Field(default_factory=list)
	alerts: List[Dict[str, Any]] = Field(default_factory=list)
	auto_refresh: bool = True
	export_enabled: bool = True
	sharing_enabled: bool = True
	created_by: str
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)

class IntelligentDashboardManager:
	"""AI-powered dashboard management system."""
	
	def __init__(
		self,
		tenant_id: str,
		db_pool: asyncpg.Pool,
		redis_url: str = "redis://localhost:6379/0"
	):
		self.tenant_id = tenant_id
		self.db_pool = db_pool
		self.redis_url = redis_url
		
		# Dashboard registry and caching
		self.dashboards: Dict[str, SmartDashboard] = {}
		self.user_preferences: Dict[str, UserPreferences] = {}
		self.active_sessions: Dict[str, Dict[str, Any]] = {}
		
		# AI components
		self.usage_analytics: Dict[str, Any] = {}
		self.personalization_models: Dict[str, Any] = {}
		
		# Real-time updates
		self.update_tasks: Dict[str, asyncio.Task] = {}
		self.alert_tasks: Dict[str, asyncio.Task] = {}
		
		logger.info(f"Initialized IntelligentDashboardManager for tenant {tenant_id}")
	
	async def initialize(self) -> None:
		"""Initialize dashboard manager."""
		try:
			# Load existing dashboards
			await self._load_dashboards()
			
			# Load user preferences
			await self._load_user_preferences()
			
			# Initialize AI models
			await self._initialize_ai_models()
			
			# Start background tasks
			await self._start_background_tasks()
			
			logger.info("Intelligent dashboard manager initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize dashboard manager: {e}")
			raise
	
	async def _load_dashboards(self) -> None:
		"""Load existing dashboards from database."""
		try:
			async with self.db_pool.acquire() as conn:
				query = """
					SELECT dashboard_data 
					FROM cm_dashboards 
					WHERE tenant_id = $1 AND active = true
				"""
				rows = await conn.fetch(query, self.tenant_id)
				
				for row in rows:
					dashboard_data = json.loads(row['dashboard_data'])
					dashboard = SmartDashboard(**dashboard_data)
					self.dashboards[dashboard.dashboard_id] = dashboard
				
				logger.info(f"Loaded {len(self.dashboards)} dashboards")
				
		except Exception as e:
			logger.warning(f"Could not load dashboards: {e}")
	
	async def _load_user_preferences(self) -> None:
		"""Load user preferences from database."""
		try:
			async with self.db_pool.acquire() as conn:
				query = """
					SELECT user_id, preferences_data 
					FROM cm_user_preferences 
					WHERE tenant_id = $1
				"""
				rows = await conn.fetch(query, self.tenant_id)
				
				for row in rows:
					prefs_data = json.loads(row['preferences_data'])
					prefs = UserPreferences(**prefs_data)
					self.user_preferences[row['user_id']] = prefs
				
				logger.info(f"Loaded preferences for {len(self.user_preferences)} users")
				
		except Exception as e:
			logger.warning(f"Could not load user preferences: {e}")
	
	async def _initialize_ai_models(self) -> None:
		"""Initialize AI models for personalization."""
		# Simple usage tracking and recommendation models
		self.personalization_models = {
			"widget_usage": {},
			"layout_preferences": {},
			"interaction_patterns": {},
			"content_relevance": {}
		}
		
		logger.info("AI models initialized")
	
	async def _start_background_tasks(self) -> None:
		"""Start background monitoring and optimization tasks."""
		# Usage analytics task
		self.update_tasks['analytics'] = asyncio.create_task(
			self._analytics_loop()
		)
		
		# Dashboard optimization task
		self.update_tasks['optimization'] = asyncio.create_task(
			self._optimization_loop()
		)
		
		# Alert monitoring task
		self.update_tasks['alerts'] = asyncio.create_task(
			self._alert_monitoring_loop()
		)
		
		logger.info("Background tasks started")
	
	async def create_smart_dashboard(
		self,
		title: str,
		target_roles: List[DashboardRole],
		template: Optional[str] = None,
		user_id: Optional[str] = None
	) -> SmartDashboard:
		"""Create a new smart dashboard with AI optimization."""
		try:
			# Determine optimal layout strategy based on roles
			layout_strategy = self._determine_optimal_layout(target_roles)
			
			# Create base dashboard
			dashboard = SmartDashboard(
				title=title,
				target_roles=target_roles,
				layout_strategy=layout_strategy,
				created_by=user_id or "system"
			)
			
			# Apply template if specified
			if template:
				await self._apply_dashboard_template(dashboard, template)
			else:
				# Generate intelligent default widgets
				await self._generate_default_widgets(dashboard, target_roles)
			
			# Store dashboard
			self.dashboards[dashboard.dashboard_id] = dashboard
			await self._save_dashboard(dashboard)
			
			# Set up real-time updates
			await self._setup_dashboard_monitoring(dashboard.dashboard_id)
			
			logger.info(f"Created smart dashboard: {dashboard.dashboard_id}")
			return dashboard
			
		except Exception as e:
			logger.error(f"Error creating smart dashboard: {e}")
			raise
	
	def _determine_optimal_layout(self, roles: List[DashboardRole]) -> LayoutStrategy:
		"""Determine optimal layout strategy based on user roles."""
		if DashboardRole.EXECUTIVE in roles:
			return LayoutStrategy.ADAPTIVE
		elif DashboardRole.ANALYST in roles:
			return LayoutStrategy.RESPONSIVE_GRID
		elif DashboardRole.OPERATOR in roles:
			return LayoutStrategy.FIXED_GRID
		else:
			return LayoutStrategy.RESPONSIVE_GRID
	
	async def _apply_dashboard_template(
		self,
		dashboard: SmartDashboard,
		template: str
	) -> None:
		"""Apply predefined dashboard template."""
		templates = {
			"executive_summary": {
				"widgets": [
					{
						"widget_type": WidgetType.KPI_CARD,
						"title": "Current Cash Position",
						"position": {"x": 0, "y": 0, "width": 3, "height": 2},
						"data_source": "current_balance_query"
					},
					{
						"widget_type": WidgetType.KPI_CARD,
						"title": "30-Day Forecast",
						"position": {"x": 3, "y": 0, "width": 3, "height": 2},
						"data_source": "forecast_query"
					},
					{
						"widget_type": WidgetType.CHART,
						"title": "Cash Flow Trend",
						"position": {"x": 0, "y": 2, "width": 6, "height": 4},
						"data_source": "cash_flow_trend_query",
						"configuration": {"chart_type": "line"}
					},
					{
						"widget_type": WidgetType.ALERT,
						"title": "Risk Alerts",
						"position": {"x": 6, "y": 0, "width": 3, "height": 3},
						"data_source": "risk_alerts_query"
					}
				]
			},
			"analyst_workbench": {
				"widgets": [
					{
						"widget_type": WidgetType.FILTER,
						"title": "Analysis Filters",
						"position": {"x": 0, "y": 0, "width": 12, "height": 1},
						"data_source": "filter_options_query"
					},
					{
						"widget_type": WidgetType.CHART,
						"title": "Detailed Cash Flow Analysis",
						"position": {"x": 0, "y": 1, "width": 8, "height": 5},
						"data_source": "detailed_analysis_query",
						"configuration": {"chart_type": "waterfall"}
					},
					{
						"widget_type": WidgetType.TABLE,
						"title": "Transaction Details",
						"position": {"x": 8, "y": 1, "width": 4, "height": 5},
						"data_source": "transaction_details_query"
					}
				]
			},
			"operations_monitor": {
				"widgets": [
					{
						"widget_type": WidgetType.LIVE_FEED,
						"title": "Real-Time Transactions",
						"position": {"x": 0, "y": 0, "width": 6, "height": 4},
						"data_source": "real_time_transactions_query"
					},
					{
						"widget_type": WidgetType.KPI_CARD,
						"title": "Today's Volume",
						"position": {"x": 6, "y": 0, "width": 3, "height": 2},
						"data_source": "daily_volume_query"
					},
					{
						"widget_type": WidgetType.ACTION_BUTTON,
						"title": "Process Payments",
						"position": {"x": 9, "y": 0, "width": 3, "height": 2},
						"data_source": "payment_actions"
					}
				]
			}
		}
		
		template_config = templates.get(template, templates["executive_summary"])
		
		for widget_config in template_config["widgets"]:
			widget = DashboardWidget(**widget_config)
			dashboard.widgets.append(widget)
	
	async def _generate_default_widgets(
		self,
		dashboard: SmartDashboard,
		roles: List[DashboardRole]
	) -> None:
		"""Generate intelligent default widgets based on roles."""
		if DashboardRole.EXECUTIVE in roles:
			# Executive widgets - high-level KPIs and trends
			widgets = [
				DashboardWidget(
					widget_type=WidgetType.KPI_CARD,
					title="Cash Position",
					position={"x": 0, "y": 0, "width": 4, "height": 2},
					data_source="executive_cash_position"
				),
				DashboardWidget(
					widget_type=WidgetType.CHART,
					title="Weekly Cash Flow",
					position={"x": 0, "y": 2, "width": 8, "height": 4},
					data_source="weekly_cash_flow",
					configuration={"chart_type": "area"}
				)
			]
		elif DashboardRole.ANALYST in roles:
			# Analyst widgets - detailed analysis tools
			widgets = [
				DashboardWidget(
					widget_type=WidgetType.FILTER,
					title="Analysis Filters",
					position={"x": 0, "y": 0, "width": 12, "height": 1},
					data_source="analysis_filters"
				),
				DashboardWidget(
					widget_type=WidgetType.CHART,
					title="Variance Analysis",
					position={"x": 0, "y": 1, "width": 6, "height": 4},
					data_source="variance_analysis",
					configuration={"chart_type": "scatter"}
				)
			]
		else:
			# Default widgets for other roles
			widgets = [
				DashboardWidget(
					widget_type=WidgetType.KPI_CARD,
					title="Account Summary",
					position={"x": 0, "y": 0, "width": 6, "height": 2},
					data_source="account_summary"
				)
			]
		
		dashboard.widgets.extend(widgets)
	
	async def personalize_dashboard(
		self,
		dashboard_id: str,
		user_id: str,
		preferences: Optional[UserPreferences] = None
	) -> SmartDashboard:
		"""Personalize dashboard based on user preferences and behavior."""
		try:
			dashboard = self.dashboards.get(dashboard_id)
			if not dashboard:
				raise ValueError(f"Dashboard {dashboard_id} not found")
			
			# Get or create user preferences
			user_prefs = preferences or self.user_preferences.get(
				user_id, 
				UserPreferences(user_id=user_id, role=DashboardRole.VIEWER)
			)
			
			# Create personalized copy
			personalized_dashboard = SmartDashboard(**dashboard.dict())
			personalized_dashboard.dashboard_id = uuid7str()
			personalized_dashboard.title = f"{dashboard.title} (Personalized)"
			
			# Apply AI-powered personalization
			await self._apply_ai_personalization(personalized_dashboard, user_prefs)
			
			# Optimize layout for user
			await self._optimize_layout_for_user(personalized_dashboard, user_prefs)
			
			# Apply user-specific configurations
			await self._apply_user_configurations(personalized_dashboard, user_prefs)
			
			# Store personalized dashboard
			self.dashboards[personalized_dashboard.dashboard_id] = personalized_dashboard
			await self._save_dashboard(personalized_dashboard)
			
			logger.info(f"Personalized dashboard for user {user_id}: {personalized_dashboard.dashboard_id}")
			return personalized_dashboard
			
		except Exception as e:
			logger.error(f"Error personalizing dashboard: {e}")
			raise
	
	async def _apply_ai_personalization(
		self,
		dashboard: SmartDashboard,
		user_prefs: UserPreferences
	) -> None:
		"""Apply AI-powered personalization to dashboard."""
		# Analyze user interaction patterns
		usage_patterns = self.usage_analytics.get(user_prefs.user_id, {})
		
		# Prioritize widgets based on usage
		if usage_patterns.get("widget_interactions"):
			popular_widgets = sorted(
				usage_patterns["widget_interactions"].items(),
				key=lambda x: x[1],
				reverse=True
			)
			
			# Reorder widgets to show most used first
			widget_priority = {widget_id: idx for idx, (widget_id, _) in enumerate(popular_widgets)}
			
			dashboard.widgets.sort(
				key=lambda w: widget_priority.get(w.widget_type.value, 999)
			)
		
		# Recommend new widgets based on role and behavior
		recommendations = await self._get_widget_recommendations(user_prefs)
		
		for rec in recommendations[:2]:  # Add top 2 recommendations
			new_widget = DashboardWidget(
				widget_type=WidgetType(rec["widget_type"]),
				title=rec["title"],
				position=rec["position"],
				data_source=rec["data_source"],
				configuration=rec.get("configuration", {})
			)
			dashboard.widgets.append(new_widget)
	
	async def _get_widget_recommendations(
		self,
		user_prefs: UserPreferences
	) -> List[Dict[str, Any]]:
		"""Get AI-powered widget recommendations."""
		recommendations = []
		
		# Role-based recommendations
		role_widgets = {
			DashboardRole.EXECUTIVE: [
				{
					"widget_type": "kpi_card",
					"title": "ROI Metrics",
					"position": {"x": 8, "y": 0, "width": 4, "height": 2},
					"data_source": "roi_metrics_query"
				}
			],
			DashboardRole.ANALYST: [
				{
					"widget_type": "chart",
					"title": "Correlation Analysis",
					"position": {"x": 6, "y": 1, "width": 6, "height": 4},
					"data_source": "correlation_analysis_query",
					"configuration": {"chart_type": "heatmap"}
				}
			],
			DashboardRole.MANAGER: [
				{
					"widget_type": "alert",
					"title": "Team Performance",
					"position": {"x": 0, "y": 6, "width": 6, "height": 2},
					"data_source": "team_performance_query"
				}
			]
		}
		
		recommendations.extend(role_widgets.get(user_prefs.role, []))
		
		return recommendations
	
	async def _optimize_layout_for_user(
		self,
		dashboard: SmartDashboard,
		user_prefs: UserPreferences
	) -> None:
		"""Optimize dashboard layout for specific user."""
		if user_prefs.layout_preference == LayoutStrategy.PERSONALIZED:
			# Apply personalized layout optimization
			await self._apply_personalized_layout(dashboard, user_prefs)
		else:
			dashboard.layout_strategy = user_prefs.layout_preference
	
	async def _apply_personalized_layout(
		self,
		dashboard: SmartDashboard,
		user_prefs: UserPreferences
	) -> None:
		"""Apply AI-optimized personalized layout."""
		# Simple layout optimization based on widget importance
		important_widgets = []
		standard_widgets = []
		
		for widget in dashboard.widgets:
			if widget.widget_type in [WidgetType.KPI_CARD, WidgetType.ALERT]:
				important_widgets.append(widget)
			else:
				standard_widgets.append(widget)
		
		# Reorganize layout - important widgets at top
		y_offset = 0
		for i, widget in enumerate(important_widgets):
			widget.position = {
				"x": (i % 3) * 4,
				"y": y_offset,
				"width": 4,
				"height": 2
			}
			if (i + 1) % 3 == 0:
				y_offset += 2
		
		if important_widgets and len(important_widgets) % 3 != 0:
			y_offset += 2
		
		# Place standard widgets below
		for i, widget in enumerate(standard_widgets):
			widget.position = {
				"x": (i % 2) * 6,
				"y": y_offset + (i // 2) * 4,
				"width": 6,
				"height": 4
			}
	
	async def _apply_user_configurations(
		self,
		dashboard: SmartDashboard,
		user_prefs: UserPreferences
	) -> None:
		"""Apply user-specific configurations."""
		# Apply refresh interval preference
		for widget in dashboard.widgets:
			widget.refresh_interval_seconds = max(
				user_prefs.refresh_interval,
				widget.refresh_interval_seconds
			)
		
		# Apply auto-refresh setting
		dashboard.auto_refresh = user_prefs.auto_alerts
		
		# Filter widgets based on role permissions
		dashboard.widgets = [
			widget for widget in dashboard.widgets
			if not widget.permissions or user_prefs.role in widget.permissions
		]
	
	async def track_user_interaction(
		self,
		user_id: str,
		dashboard_id: str,
		interaction_type: str,
		widget_id: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> None:
		"""Track user interactions for AI learning."""
		try:
			interaction = {
				"timestamp": datetime.now().isoformat(),
				"user_id": user_id,
				"dashboard_id": dashboard_id,
				"interaction_type": interaction_type,
				"widget_id": widget_id,
				"metadata": metadata or {}
			}
			
			# Update usage analytics
			if user_id not in self.usage_analytics:
				self.usage_analytics[user_id] = {
					"total_interactions": 0,
					"widget_interactions": {},
					"session_duration": [],
					"preferred_times": [],
					"interaction_patterns": []
				}
			
			analytics = self.usage_analytics[user_id]
			analytics["total_interactions"] += 1
			analytics["interaction_patterns"].append(interaction)
			
			if widget_id:
				analytics["widget_interactions"][widget_id] = \
					analytics["widget_interactions"].get(widget_id, 0) + 1
			
			# Update AI models
			await self._update_personalization_models(user_id, interaction)
			
		except Exception as e:
			logger.error(f"Error tracking user interaction: {e}")
	
	async def _update_personalization_models(
		self,
		user_id: str,
		interaction: Dict[str, Any]
	) -> None:
		"""Update personalization models with new interaction data."""
		# Simple learning algorithm for demonstration
		models = self.personalization_models
		
		# Update widget usage patterns
		widget_id = interaction.get("widget_id")
		if widget_id:
			if user_id not in models["widget_usage"]:
				models["widget_usage"][user_id] = {}
			models["widget_usage"][user_id][widget_id] = \
				models["widget_usage"][user_id].get(widget_id, 0) + 1
		
		# Update interaction patterns
		interaction_type = interaction["interaction_type"]
		if user_id not in models["interaction_patterns"]:
			models["interaction_patterns"][user_id] = {}
		models["interaction_patterns"][user_id][interaction_type] = \
			models["interaction_patterns"][user_id].get(interaction_type, 0) + 1
	
	async def generate_smart_alerts(
		self,
		dashboard_id: str,
		user_id: Optional[str] = None
	) -> List[Dict[str, Any]]:
		"""Generate intelligent alerts for dashboard."""
		try:
			alerts = []
			
			# Get dashboard data
			dashboard = self.dashboards.get(dashboard_id)
			if not dashboard:
				return alerts
			
			# Analyze current cash flow data
			async with self.db_pool.acquire() as conn:
				# Check for low balance alerts
				balance_query = """
					SELECT account_id, current_balance, minimum_balance
					FROM cm_accounts 
					WHERE tenant_id = $1 AND current_balance < minimum_balance * 1.1
				"""
				low_balance_accounts = await conn.fetch(balance_query, self.tenant_id)
				
				for account in low_balance_accounts:
					alerts.append({
						"type": "warning",
						"title": "Low Balance Alert",
						"message": f"Account {account['account_id']} balance is approaching minimum",
						"severity": "medium",
						"account_id": account['account_id'],
						"current_balance": float(account['current_balance']),
						"threshold": float(account['minimum_balance']),
						"timestamp": datetime.now().isoformat()
					})
				
				# Check for unusual transaction patterns
				pattern_query = """
					SELECT account_id, COUNT(*) as transaction_count,
						   AVG(amount) as avg_amount, SUM(amount) as total_amount
					FROM cm_cash_flows 
					WHERE tenant_id = $1 AND transaction_date >= CURRENT_DATE - INTERVAL '7 days'
					GROUP BY account_id
					HAVING COUNT(*) > (
						SELECT AVG(daily_count) * 2 
						FROM (
							SELECT COUNT(*) as daily_count 
							FROM cm_cash_flows 
							WHERE tenant_id = $1 AND transaction_date >= CURRENT_DATE - INTERVAL '30 days'
							GROUP BY account_id, DATE(transaction_date)
						) daily_stats
					)
				"""
				unusual_patterns = await conn.fetch(pattern_query, self.tenant_id)
				
				for pattern in unusual_patterns:
					alerts.append({
						"type": "info",
						"title": "Unusual Activity",
						"message": f"High transaction volume detected for account {pattern['account_id']}",
						"severity": "low",
						"account_id": pattern['account_id'],
						"transaction_count": pattern['transaction_count'],
						"total_amount": float(pattern['total_amount']),
						"timestamp": datetime.now().isoformat()
					})
			
			# Personalize alerts if user specified
			if user_id and user_id in self.user_preferences:
				user_prefs = self.user_preferences[user_id]
				if not user_prefs.auto_alerts:
					alerts = [alert for alert in alerts if alert["severity"] != "low"]
			
			return alerts
			
		except Exception as e:
			logger.error(f"Error generating smart alerts: {e}")
			return []
	
	async def export_dashboard(
		self,
		dashboard_id: str,
		format: str = "pdf",
		include_data: bool = True
	) -> bytes:
		"""Export dashboard in various formats."""
		try:
			dashboard = self.dashboards.get(dashboard_id)
			if not dashboard:
				raise ValueError(f"Dashboard {dashboard_id} not found")
			
			if format.lower() == "json":
				export_data = dashboard.dict()
				if include_data:
					# Add current widget data
					export_data["widget_data"] = {}
					for widget in dashboard.widgets:
						widget_data = await self._get_widget_data(widget)
						export_data["widget_data"][widget.widget_id] = widget_data
				
				return json.dumps(export_data, indent=2, default=str).encode()
			
			elif format.lower() == "pdf":
				# For PDF export, we'd typically use a library like reportlab
				# This is a simplified placeholder
				pdf_content = f"Dashboard Export: {dashboard.title}\n"
				pdf_content += f"Generated at: {datetime.now()}\n"
				pdf_content += f"Widgets: {len(dashboard.widgets)}\n"
				return pdf_content.encode()
			
			else:
				raise ValueError(f"Unsupported export format: {format}")
				
		except Exception as e:
			logger.error(f"Error exporting dashboard {dashboard_id}: {e}")
			raise
	
	async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
		"""Get current data for a widget."""
		try:
			# This would typically execute the widget's data source query
			# and return the current data
			return {
				"widget_id": widget.widget_id,
				"title": widget.title,
				"type": widget.widget_type.value,
				"last_updated": datetime.now().isoformat(),
				"data": []  # Placeholder for actual data
			}
		except Exception as e:
			logger.error(f"Error getting widget data for {widget.widget_id}: {e}")
			return {}
	
	async def _setup_dashboard_monitoring(self, dashboard_id: str) -> None:
		"""Set up monitoring for dashboard."""
		if dashboard_id not in self.update_tasks:
			self.update_tasks[dashboard_id] = asyncio.create_task(
				self._dashboard_monitoring_loop(dashboard_id)
			)
	
	async def _dashboard_monitoring_loop(self, dashboard_id: str) -> None:
		"""Monitor dashboard for updates and optimizations."""
		try:
			dashboard = self.dashboards.get(dashboard_id)
			if not dashboard:
				return
			
			while dashboard_id in self.dashboards:
				# Check for performance optimizations
				await self._optimize_dashboard_performance(dashboard_id)
				
				# Update real-time widgets
				await self._update_real_time_widgets(dashboard_id)
				
				# Check for alert conditions
				alerts = await self.generate_smart_alerts(dashboard_id)
				if alerts:
					await self._process_dashboard_alerts(dashboard_id, alerts)
				
				# Wait for next check
				await asyncio.sleep(60)  # Check every minute
				
		except Exception as e:
			logger.error(f"Dashboard monitoring error for {dashboard_id}: {e}")
		finally:
			if dashboard_id in self.update_tasks:
				del self.update_tasks[dashboard_id]
	
	async def _optimize_dashboard_performance(self, dashboard_id: str) -> None:
		"""Optimize dashboard performance based on usage patterns."""
		try:
			dashboard = self.dashboards.get(dashboard_id)
			if not dashboard:
				return
			
			# Analyze widget performance
			for widget in dashboard.widgets:
				# Adjust refresh intervals based on usage
				usage_count = self.usage_analytics.get("widget_interactions", {}).get(
					widget.widget_id, 0
				)
				
				if usage_count > 100:  # High usage
					widget.refresh_interval_seconds = max(30, widget.refresh_interval_seconds)
				elif usage_count < 10:  # Low usage
					widget.refresh_interval_seconds = min(300, widget.refresh_interval_seconds * 2)
			
			# Update dashboard
			await self._save_dashboard(dashboard)
			
		except Exception as e:
			logger.error(f"Error optimizing dashboard performance: {e}")
	
	async def _update_real_time_widgets(self, dashboard_id: str) -> None:
		"""Update real-time widgets in dashboard."""
		# This would typically push updates to connected clients
		pass
	
	async def _process_dashboard_alerts(
		self,
		dashboard_id: str,
		alerts: List[Dict[str, Any]]
	) -> None:
		"""Process alerts for dashboard."""
		# This would typically send notifications or update alert widgets
		logger.info(f"Processing {len(alerts)} alerts for dashboard {dashboard_id}")
	
	async def _analytics_loop(self) -> None:
		"""Background analytics processing loop."""
		while True:
			try:
				# Process usage analytics
				await self._process_usage_analytics()
				
				# Update personalization models
				await self._train_personalization_models()
				
				# Clean up old data
				await self._cleanup_old_analytics_data()
				
				await asyncio.sleep(3600)  # Run every hour
				
			except Exception as e:
				logger.error(f"Analytics loop error: {e}")
				await asyncio.sleep(1800)  # Retry in 30 minutes
	
	async def _optimization_loop(self) -> None:
		"""Background optimization loop."""
		while True:
			try:
				# Optimize all dashboards
				for dashboard_id in list(self.dashboards.keys()):
					await self._optimize_dashboard_performance(dashboard_id)
				
				await asyncio.sleep(7200)  # Run every 2 hours
				
			except Exception as e:
				logger.error(f"Optimization loop error: {e}")
				await asyncio.sleep(3600)  # Retry in 1 hour
	
	async def _alert_monitoring_loop(self) -> None:
		"""Background alert monitoring loop."""
		while True:
			try:
				# Check alerts for all dashboards
				for dashboard_id in list(self.dashboards.keys()):
					alerts = await self.generate_smart_alerts(dashboard_id)
					if alerts:
						await self._process_dashboard_alerts(dashboard_id, alerts)
				
				await asyncio.sleep(300)  # Check every 5 minutes
				
			except Exception as e:
				logger.error(f"Alert monitoring loop error: {e}")
				await asyncio.sleep(600)  # Retry in 10 minutes
	
	async def _process_usage_analytics(self) -> None:
		"""Process accumulated usage analytics."""
		# Aggregate interaction patterns
		for user_id, analytics in self.usage_analytics.items():
			if len(analytics["interaction_patterns"]) > 1000:
				# Keep only recent interactions
				analytics["interaction_patterns"] = analytics["interaction_patterns"][-500:]
	
	async def _train_personalization_models(self) -> None:
		"""Train/update personalization models."""
		# Simple model updates based on usage patterns
		# In a real implementation, this would use proper ML algorithms
		pass
	
	async def _cleanup_old_analytics_data(self) -> None:
		"""Clean up old analytics data."""
		cutoff_date = datetime.now() - timedelta(days=30)
		
		for user_id, analytics in list(self.usage_analytics.items()):
			# Remove old interaction patterns
			analytics["interaction_patterns"] = [
				pattern for pattern in analytics["interaction_patterns"]
				if datetime.fromisoformat(pattern["timestamp"]) > cutoff_date
			]
			
			# Remove user if no recent activity
			if not analytics["interaction_patterns"]:
				del self.usage_analytics[user_id]
	
	async def _save_dashboard(self, dashboard: SmartDashboard) -> None:
		"""Save dashboard to database."""
		try:
			async with self.db_pool.acquire() as conn:
				query = """
					INSERT INTO cm_dashboards (tenant_id, dashboard_id, dashboard_data, created_at, updated_at)
					VALUES ($1, $2, $3, $4, $5)
					ON CONFLICT (tenant_id, dashboard_id) 
					DO UPDATE SET dashboard_data = $3, updated_at = $5
				"""
				await conn.execute(
					query,
					self.tenant_id,
					dashboard.dashboard_id,
					json.dumps(dashboard.dict(), default=str),
					dashboard.created_at,
					datetime.now()
				)
		except Exception as e:
			logger.error(f"Error saving dashboard: {e}")
	
	async def cleanup(self) -> None:
		"""Cleanup resources."""
		# Stop all background tasks
		for task in self.update_tasks.values():
			task.cancel()
		
		for task in self.alert_tasks.values():
			task.cancel()
		
		# Save final state
		for dashboard in self.dashboards.values():
			await self._save_dashboard(dashboard)
		
		logger.info("Intelligent dashboard manager cleanup completed")

# Global dashboard manager instance
_dashboard_manager: Optional[IntelligentDashboardManager] = None

async def get_dashboard_manager(
	tenant_id: str,
	db_pool: asyncpg.Pool
) -> IntelligentDashboardManager:
	"""Get or create dashboard manager instance."""
	global _dashboard_manager
	
	if _dashboard_manager is None or _dashboard_manager.tenant_id != tenant_id:
		_dashboard_manager = IntelligentDashboardManager(tenant_id, db_pool)
		await _dashboard_manager.initialize()
	
	return _dashboard_manager

if __name__ == "__main__":
	async def main():
		# Example usage would require a real database connection
		print("Intelligent Dashboard Manager initialized")
		print("This module provides AI-powered dashboard management")
	
	asyncio.run(main())