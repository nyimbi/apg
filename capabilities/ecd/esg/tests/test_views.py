#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Sustainability & ESG Management - Views Tests

Comprehensive test suite for Flask-AppBuilder views, UI components,
and dashboard functionality.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from flask import Flask
from flask_appbuilder import AppBuilder
from sqlalchemy.orm import Session

from ..views import (
	ESGTenantView, ESGMetricView, ESGTargetView, ESGStakeholderView,
	ESGSupplierView, ESGInitiativeView, ESGReportView,
	ESGExecutiveDashboardView, ESGMetricsView, ESGTargetsView,
	ESGStakeholdersView, ESGSuppliersView, ESGInitiativesView,
	ESGReportsView, ESGStakeholderPortalView,
	ESGDashboardWidget, ESGMetricWidget, ESGStakeholderPortalWidget
)
from ..models import (
	ESGTenant, ESGMetric, ESGTarget, ESGStakeholder, ESGSupplier,
	ESGInitiative, ESGReport, ESGMetricType, ESGTargetStatus
)
from ..service import ESGManagementService, ESGServiceConfig
from . import ESGTestConfig, TEST_TENANT_ID, TEST_USER_ID


class TestPydanticViews:
	"""Test Pydantic view models for API serialization"""
	
	def test_esg_tenant_view_validation(self):
		"""Test ESG tenant view model validation"""
		valid_data = {
			"id": "test_tenant",
			"name": "Test Corporation",
			"slug": "test-corp",
			"description": "Test corporation",
			"industry": "technology",
			"headquarters_country": "USA",
			"employee_count": 1000,
			"annual_revenue": Decimal("500000000.00"),
			"esg_frameworks": ["GRI", "SASB"],
			"ai_enabled": True,
			"is_active": True,
			"subscription_tier": "enterprise",
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		
		tenant_view = ESGTenantView(**valid_data)
		
		assert tenant_view.id == "test_tenant"
		assert tenant_view.name == "Test Corporation"
		assert tenant_view.slug == "test-corp"
		assert tenant_view.employee_count == 1000
		assert len(tenant_view.esg_frameworks) == 2
		assert tenant_view.ai_enabled is True
	
	def test_esg_tenant_view_slug_validation(self):
		"""Test tenant view slug pattern validation"""
		# Valid slug
		valid_data = {
			"id": "test",
			"name": "Test",
			"slug": "valid-slug-123",
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		tenant_view = ESGTenantView(**valid_data)
		assert tenant_view.slug == "valid-slug-123"
		
		# Invalid slug should raise validation error
		invalid_data = valid_data.copy()
		invalid_data["slug"] = "Invalid Slug!"  # Contains spaces and special chars
		
		with pytest.raises(ValueError):
			ESGTenantView(**invalid_data)
	
	def test_esg_metric_view_validation(self):
		"""Test ESG metric view model validation"""
		valid_data = {
			"id": "test_metric",
			"tenant_id": TEST_TENANT_ID,
			"name": "Test Metric",
			"code": "TEST_METRIC",
			"metric_type": ESGMetricType.ENVIRONMENTAL,
			"category": "energy",
			"unit": "kwh",
			"current_value": Decimal("1250.75"),
			"target_value": Decimal("1000.00"),
			"baseline_value": Decimal("1500.00"),
			"is_kpi": True,
			"is_public": False,
			"is_automated": True,
			"data_quality_score": Decimal("94.5"),
			"ai_predictions": {"trend": "improving"},
			"trend_analysis": {"direction": "decreasing"},
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		
		metric_view = ESGMetricView(**valid_data)
		
		assert metric_view.name == "Test Metric"
		assert metric_view.code == "TEST_METRIC"
		assert metric_view.metric_type == ESGMetricType.ENVIRONMENTAL
		assert metric_view.current_value == Decimal("1250.75")
		assert metric_view.data_quality_score == Decimal("94.5")
		assert metric_view.is_kpi is True
	
	def test_esg_metric_view_code_validation(self):
		"""Test metric view code pattern validation"""
		valid_data = {
			"id": "test",
			"tenant_id": TEST_TENANT_ID,
			"name": "Test",
			"code": "VALID_CODE_123",
			"metric_type": ESGMetricType.SOCIAL,
			"category": "test",
			"unit": "count",
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		
		metric_view = ESGMetricView(**valid_data)
		assert metric_view.code == "VALID_CODE_123"
		
		# Invalid code should raise validation error
		invalid_data = valid_data.copy()
		invalid_data["code"] = "invalid-code"  # Contains hyphens, not allowed
		
		with pytest.raises(ValueError):
			ESGMetricView(**invalid_data)
	
	def test_esg_target_view_validation(self):
		"""Test ESG target view model validation"""
		start_date = datetime(2024, 1, 1)
		target_date = datetime(2025, 12, 31)
		
		valid_data = {
			"id": "test_target",
			"tenant_id": TEST_TENANT_ID,
			"metric_id": "test_metric",
			"name": "Test Target",
			"target_value": Decimal("100.0"),
			"baseline_value": Decimal("50.0"),
			"current_progress": Decimal("75.0"),
			"start_date": start_date,
			"target_date": target_date,
			"status": ESGTargetStatus.ON_TRACK,
			"achievement_probability": Decimal("85.5"),
			"owner_id": "test_owner",
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		
		target_view = ESGTargetView(**valid_data)
		
		assert target_view.name == "Test Target"
		assert target_view.target_value == Decimal("100.0")
		assert target_view.current_progress == Decimal("75.0")
		assert target_view.status == ESGTargetStatus.ON_TRACK
		assert target_view.start_date == start_date
		assert target_view.target_date == target_date
	
	def test_esg_target_view_date_validation(self):
		"""Test target view date validation"""
		# Target date before start date should raise validation error
		invalid_data = {
			"id": "test",
			"tenant_id": TEST_TENANT_ID,
			"metric_id": "test_metric",
			"name": "Invalid Target",
			"target_value": Decimal("100.0"),
			"start_date": datetime(2025, 1, 1),
			"target_date": datetime(2024, 12, 31),  # Before start date
			"owner_id": "test_owner",
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		
		with pytest.raises(ValueError):
			ESGTargetView(**invalid_data)
	
	def test_esg_stakeholder_view_validation(self):
		"""Test ESG stakeholder view model validation"""
		valid_data = {
			"id": "test_stakeholder",
			"tenant_id": TEST_TENANT_ID,
			"name": "Test Stakeholder",
			"organization": "Test Organization",
			"stakeholder_type": "investor",
			"email": "test@example.com",
			"country": "USA",
			"engagement_score": Decimal("87.5"),
			"sentiment_score": Decimal("75.2"),
			"influence_score": Decimal("92.0"),
			"portal_access": True,
			"is_active": True,
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		
		stakeholder_view = ESGStakeholderView(**valid_data)
		
		assert stakeholder_view.name == "Test Stakeholder"
		assert stakeholder_view.engagement_score == Decimal("87.5")
		assert stakeholder_view.portal_access is True


class TestCustomWidgets:
	"""Test custom UI widgets"""
	
	def test_esg_dashboard_widget_initialization(self):
		"""Test ESG dashboard widget initialization"""
		widget = ESGDashboardWidget(
			include_charts=True,
			include_kpis=True,
			real_time_updates=True
		)
		
		assert widget.template == 'esg/widgets/dashboard_widget.html'
		assert widget.include_charts is True
		assert widget.include_kpis is True
		assert widget.real_time_updates is True
		
		# Test with different configuration
		widget2 = ESGDashboardWidget(
			include_charts=False,
			real_time_updates=False
		)
		
		assert widget2.include_charts is False
		assert widget2.real_time_updates is False
		assert widget2.include_kpis is True  # Default value
	
	def test_esg_metric_widget_initialization(self):
		"""Test ESG metric widget initialization"""
		widget = ESGMetricWidget(
			show_trend=True,
			show_ai_insights=True
		)
		
		assert widget.template == 'esg/widgets/metric_widget.html'
		assert widget.show_trend is True
		assert widget.show_ai_insights is True
		
		# Test with AI insights disabled
		widget2 = ESGMetricWidget(show_ai_insights=False)
		
		assert widget2.show_ai_insights is False
		assert widget2.show_trend is True  # Default value
	
	def test_esg_stakeholder_portal_widget_initialization(self):
		"""Test ESG stakeholder portal widget initialization"""
		widget = ESGStakeholderPortalWidget(
			portal_mode="internal",
			engagement_tracking=True
		)
		
		assert widget.template == 'esg/widgets/stakeholder_portal_widget.html'
		assert widget.portal_mode == "internal"
		assert widget.engagement_tracking is True
		
		# Test external portal mode
		widget2 = ESGStakeholderPortalWidget(portal_mode="external")
		
		assert widget2.portal_mode == "external"
		assert widget2.engagement_tracking is True  # Default value


class TestESGExecutiveDashboardView:
	"""Test executive dashboard view"""
	
	@pytest.fixture
	def mock_dashboard_view(self, flask_app: Flask, app_builder: AppBuilder):
		"""Create mock dashboard view with dependencies"""
		view = ESGExecutiveDashboardView()
		view.appbuilder = app_builder
		
		# Mock user methods
		view.get_user_id = Mock(return_value=TEST_USER_ID)
		view.get_user_tenant_id = Mock(return_value=TEST_TENANT_ID)
		
		# Mock template rendering
		view.render_template = Mock(return_value="rendered_template")
		
		return view
	
	@pytest.mark.asyncio
	async def test_dashboard_data_retrieval(self, mock_dashboard_view):
		"""Test dashboard data retrieval"""
		# Mock ESG service
		mock_service = Mock(spec=ESGManagementService)
		mock_service.get_metrics = AsyncMock(return_value=[])
		
		with patch('sustainability_esg_management.views.ESGManagementService', return_value=mock_service):
			dashboard_data = await mock_dashboard_view._get_executive_dashboard_data(
				mock_service, TEST_USER_ID
			)
		
		# Verify dashboard data structure
		assert "key_metrics" in dashboard_data
		assert "active_targets" in dashboard_data
		assert "recent_reports" in dashboard_data
		assert "stakeholder_summary" in dashboard_data
		assert "ai_insights" in dashboard_data
		assert "last_updated" in dashboard_data
		
		# Verify service was called
		mock_service.get_metrics.assert_called_with(
			user_id=TEST_USER_ID,
			filters={"is_kpi": True, "limit": 10}
		)
	
	def test_serialize_metric(self, mock_dashboard_view, sample_esg_metrics: List[ESGMetric]):
		"""Test metric serialization for dashboard"""
		metric = sample_esg_metrics[0]
		
		serialized = mock_dashboard_view._serialize_metric(metric)
		
		# Verify serialized structure
		assert "id" in serialized
		assert "name" in serialized
		assert "current_value" in serialized
		assert "target_value" in serialized
		assert "unit" in serialized
		assert "trend" in serialized
		assert "data_quality" in serialized
		assert "last_updated" in serialized
		
		# Verify data types
		assert isinstance(serialized["current_value"], float)
		assert isinstance(serialized["last_updated"], str)
	
	def test_user_context_methods(self, mock_dashboard_view):
		"""Test user context retrieval methods"""
		user_id = mock_dashboard_view.get_user_id()
		tenant_id = mock_dashboard_view.get_user_tenant_id()
		
		assert user_id == TEST_USER_ID
		assert tenant_id == TEST_TENANT_ID


class TestESGMetricsView:
	"""Test ESG metrics management view"""
	
	@pytest.fixture
	def mock_metrics_view(self, flask_app: Flask, app_builder: AppBuilder):
		"""Create mock metrics view with dependencies"""
		view = ESGMetricsView()
		view.appbuilder = app_builder
		
		# Mock database session
		view.appbuilder.get_session = Mock()
		
		# Mock user methods
		view.get_user_id = Mock(return_value=TEST_USER_ID)
		view.get_user_tenant_id = Mock(return_value=TEST_TENANT_ID)
		
		return view
	
	def test_view_configuration(self, mock_metrics_view):
		"""Test metrics view configuration"""
		# Verify list configuration
		assert "name" in mock_metrics_view.list_columns
		assert "code" in mock_metrics_view.list_columns
		assert "metric_type" in mock_metrics_view.list_columns
		assert "current_value" in mock_metrics_view.list_columns
		
		# Verify search configuration
		assert "name" in mock_metrics_view.search_columns
		assert "code" in mock_metrics_view.search_columns
		assert "description" in mock_metrics_view.search_columns
		
		# Verify form configuration
		assert "name" in mock_metrics_view.add_columns
		assert "metric_type" in mock_metrics_view.add_columns
		assert "unit" in mock_metrics_view.add_columns
		
		# Verify custom widgets
		assert mock_metrics_view.list_widget == ESGDashboardWidget
		assert mock_metrics_view.show_widget == ESGMetricWidget
	
	@pytest.mark.asyncio
	async def test_ai_insights_endpoint(self, mock_metrics_view):
		"""Test AI insights endpoint"""
		mock_service = Mock(spec=ESGManagementService)
		mock_service._initialize_metric_ai_predictions = AsyncMock(return_value={
			"predictions": {"6_month": 95.5},
			"confidence": 0.89
		})
		
		with patch('sustainability_esg_management.views.ESGManagementService', return_value=mock_service):
			with patch('flask.jsonify') as mock_jsonify:
				await mock_metrics_view.ai_insights("test_metric_id")
		
		# Verify service was called
		mock_service._initialize_metric_ai_predictions.assert_called_with(
			"test_metric_id", TEST_USER_ID
		)
		
		# Verify response structure
		mock_jsonify.assert_called_once()
		call_args = mock_jsonify.call_args[0][0]
		assert call_args["status"] == "success"
		assert call_args["metric_id"] == "test_metric_id"
		assert "insights" in call_args
	
	@pytest.mark.asyncio
	async def test_record_measurement_endpoint(self, mock_metrics_view):
		"""Test measurement recording endpoint"""
		mock_service = Mock(spec=ESGManagementService)
		mock_measurement = Mock()
		mock_measurement.id = "test_measurement_id"
		mock_service.record_measurement = AsyncMock(return_value=mock_measurement)
		
		measurement_data = {
			"metric_id": "test_metric",
			"value": 1250.75,
			"measurement_date": datetime.utcnow().isoformat()
		}
		
		with patch('sustainability_esg_management.views.ESGManagementService', return_value=mock_service):
			with patch('flask.request') as mock_request:
				mock_request.get_json.return_value = measurement_data
				
				with patch('flask.jsonify') as mock_jsonify:
					await mock_metrics_view.record_measurement()
		
		# Verify service was called
		mock_service.record_measurement.assert_called_with(
			user_id=TEST_USER_ID,
			measurement_data=measurement_data
		)
		
		# Verify success response
		mock_jsonify.assert_called_once()
		call_args = mock_jsonify.call_args[0][0]
		assert call_args["status"] == "success"
		assert call_args["measurement_id"] == "test_measurement_id"


class TestESGTargetsView:
	"""Test ESG targets management view"""
	
	@pytest.fixture
	def mock_targets_view(self, flask_app: Flask, app_builder: AppBuilder):
		"""Create mock targets view with dependencies"""
		view = ESGTargetsView()
		view.appbuilder = app_builder
		
		# Mock database session
		view.appbuilder.get_session = Mock()
		
		# Mock user methods
		view.get_user_id = Mock(return_value=TEST_USER_ID)
		view.get_user_tenant_id = Mock(return_value=TEST_TENANT_ID)
		
		return view
	
	def test_view_configuration(self, mock_targets_view):
		"""Test targets view configuration"""
		# Verify list configuration
		assert "name" in mock_targets_view.list_columns
		assert "target_value" in mock_targets_view.list_columns
		assert "current_progress" in mock_targets_view.list_columns
		assert "status" in mock_targets_view.list_columns
		assert "achievement_probability" in mock_targets_view.list_columns
		
		# Verify form configuration
		assert "name" in mock_targets_view.add_columns
		assert "metric" in mock_targets_view.add_columns
		assert "target_value" in mock_targets_view.add_columns
		assert "target_date" in mock_targets_view.add_columns
		
		# Verify ordering
		assert mock_targets_view.base_order == ('target_date', 'asc')
	
	@pytest.mark.asyncio
	async def test_predict_achievement_endpoint(self, mock_targets_view):
		"""Test target achievement prediction endpoint"""
		mock_service = Mock(spec=ESGManagementService)
		mock_service._predict_target_achievement = AsyncMock(return_value={
			"probability": 85.5,
			"predicted_completion_date": "2025-10-15",
			"confidence": "high"
		})
		
		with patch('sustainability_esg_management.views.ESGManagementService', return_value=mock_service):
			with patch('flask.jsonify') as mock_jsonify:
				await mock_targets_view.predict_achievement("test_target_id")
		
		# Verify service was called
		mock_service._predict_target_achievement.assert_called_with(
			"test_target_id", TEST_USER_ID
		)
		
		# Verify response
		mock_jsonify.assert_called_once()
		call_args = mock_jsonify.call_args[0][0]
		assert call_args["status"] == "success"
		assert call_args["target_id"] == "test_target_id"
		assert "prediction" in call_args
	
	@pytest.mark.asyncio
	async def test_create_milestones_endpoint(self, mock_targets_view):
		"""Test milestone creation endpoint"""
		mock_service = Mock(spec=ESGManagementService)
		mock_milestones = [Mock(), Mock(), Mock()]
		mock_service._create_default_milestones = AsyncMock(return_value=mock_milestones)
		
		with patch('sustainability_esg_management.views.ESGManagementService', return_value=mock_service):
			with patch('flask.jsonify') as mock_jsonify:
				await mock_targets_view.create_milestones("test_target_id")
		
		# Verify service was called
		mock_service._create_default_milestones.assert_called_with(
			"test_target_id", TEST_USER_ID
		)
		
		# Verify response
		mock_jsonify.assert_called_once()
		call_args = mock_jsonify.call_args[0][0]
		assert call_args["status"] == "success"
		assert call_args["milestones_created"] == 3


class TestESGStakeholdersView:
	"""Test ESG stakeholders management view"""
	
	@pytest.fixture
	def mock_stakeholders_view(self, flask_app: Flask, app_builder: AppBuilder):
		"""Create mock stakeholders view with dependencies"""
		view = ESGStakeholdersView()
		view.appbuilder = app_builder
		
		# Mock database session
		view.appbuilder.get_session = Mock()
		
		# Mock user methods
		view.get_user_id = Mock(return_value=TEST_USER_ID)
		view.get_user_tenant_id = Mock(return_value=TEST_TENANT_ID)
		
		return view
	
	def test_view_configuration(self, mock_stakeholders_view):
		"""Test stakeholders view configuration"""
		# Verify list configuration
		assert "name" in mock_stakeholders_view.list_columns
		assert "organization" in mock_stakeholders_view.list_columns
		assert "stakeholder_type" in mock_stakeholders_view.list_columns
		assert "engagement_score" in mock_stakeholders_view.list_columns
		assert "portal_access" in mock_stakeholders_view.list_columns
		
		# Verify custom widget
		assert mock_stakeholders_view.list_widget == ESGStakeholderPortalWidget
	
	@pytest.mark.asyncio
	async def test_engagement_analytics_endpoint(self, mock_stakeholders_view):
		"""Test stakeholder engagement analytics endpoint"""
		mock_service = Mock(spec=ESGManagementService)
		mock_service._initialize_stakeholder_analytics = AsyncMock(return_value={
			"engagement_insights": {"satisfaction_level": "high"},
			"influence_score": 85.5
		})
		
		with patch('sustainability_esg_management.views.ESGManagementService', return_value=mock_service):
			with patch('flask.jsonify') as mock_jsonify:
				await mock_stakeholders_view.engagement_analytics("test_stakeholder_id")
		
		# Verify service was called
		mock_service._initialize_stakeholder_analytics.assert_called_with(
			"test_stakeholder_id", TEST_USER_ID
		)
		
		# Verify response
		mock_jsonify.assert_called_once()
		call_args = mock_jsonify.call_args[0][0]
		assert call_args["status"] == "success"
		assert call_args["stakeholder_id"] == "test_stakeholder_id"
		assert "analytics" in call_args


class TestESGReportsView:
	"""Test ESG reports management view"""
	
	@pytest.fixture
	def mock_reports_view(self, flask_app: Flask, app_builder: AppBuilder):
		"""Create mock reports view with dependencies"""
		view = ESGReportsView()
		view.appbuilder = app_builder
		
		# Mock user methods
		view.get_user_tenant_id = Mock(return_value=TEST_TENANT_ID)
		
		return view
	
	def test_view_configuration(self, mock_reports_view):
		"""Test reports view configuration"""
		# Verify list configuration
		assert "name" in mock_reports_view.list_columns
		assert "report_type" in mock_reports_view.list_columns
		assert "framework" in mock_reports_view.list_columns
		assert "status" in mock_reports_view.list_columns
		assert "auto_generated" in mock_reports_view.list_columns
		
		# Verify ordering (most recent first)
		assert mock_reports_view.base_order == ('reporting_year', 'desc')
	
	def test_generate_report_endpoint(self, mock_reports_view):
		"""Test report generation endpoint"""
		with patch('flask.request') as mock_request:
			mock_request.get_json.return_value = {
				"report_type": "sustainability",
				"framework": "GRI",
				"period": "2024"
			}
			
			with patch('flask.jsonify') as mock_jsonify:
				response = mock_reports_view.generate_report()
		
		# Verify response structure
		mock_jsonify.assert_called_once()
		call_args = mock_jsonify.call_args[0][0]
		assert call_args["status"] == "success"
		assert "message" in call_args
		assert "estimated_completion" in call_args


class TestESGStakeholderPortalView:
	"""Test public stakeholder portal view"""
	
	@pytest.fixture
	def mock_portal_view(self, flask_app: Flask, app_builder: AppBuilder):
		"""Create mock portal view with dependencies"""
		view = ESGStakeholderPortalView()
		view.appbuilder = app_builder
		
		# Mock template rendering
		view.render_template = Mock(return_value="rendered_template")
		
		# Mock data retrieval methods
		view._get_public_esg_data = Mock(return_value={
			"key_metrics": [],
			"sustainability_goals": [],
			"recent_initiatives": [],
			"transparency_score": 85.0
		})
		
		view._get_public_reports = Mock(return_value=[])
		
		return view
	
	def test_public_dashboard_endpoint(self, mock_portal_view):
		"""Test public dashboard endpoint"""
		with patch('flask.request') as mock_request:
			mock_request.args.get.return_value = "test_tenant"
			
			result = mock_portal_view.public_dashboard()
		
		# Verify data retrieval was called
		mock_portal_view._get_public_esg_data.assert_called_with("test_tenant")
		
		# Verify template rendering
		mock_portal_view.render_template.assert_called_with(
			'esg/public_portal.html',
			public_data=mock_portal_view._get_public_esg_data.return_value,
			tenant_id="test_tenant"
		)
	
	def test_public_reports_endpoint(self, mock_portal_view):
		"""Test public reports endpoint"""
		with patch('flask.request') as mock_request:
			mock_request.args.get.return_value = "test_tenant"
			
			result = mock_portal_view.public_reports()
		
		# Verify data retrieval was called
		mock_portal_view._get_public_reports.assert_called_with("test_tenant")
		
		# Verify template rendering
		mock_portal_view.render_template.assert_called_with(
			'esg/public_reports.html',
			reports=mock_portal_view._get_public_reports.return_value,
			tenant_id="test_tenant"
		)
	
	def test_get_public_esg_data(self, mock_portal_view):
		"""Test public ESG data retrieval"""
		# Reset mock to test actual implementation
		mock_portal_view._get_public_esg_data = ESGStakeholderPortalView._get_public_esg_data.__get__(mock_portal_view)
		
		public_data = mock_portal_view._get_public_esg_data("test_tenant")
		
		# Verify data structure
		assert "key_metrics" in public_data
		assert "sustainability_goals" in public_data
		assert "recent_initiatives" in public_data
		assert "transparency_score" in public_data
		
		# Verify data types
		assert isinstance(public_data["key_metrics"], list)
		assert isinstance(public_data["transparency_score"], float)
	
	def test_get_public_reports(self, mock_portal_view):
		"""Test public reports retrieval"""
		# Reset mock to test actual implementation
		mock_portal_view._get_public_reports = ESGStakeholderPortalView._get_public_reports.__get__(mock_portal_view)
		
		public_reports = mock_portal_view._get_public_reports("test_tenant")
		
		# Verify returns list (empty in mock implementation)
		assert isinstance(public_reports, list)


class TestViewSecurity:
	"""Test view security and access control"""
	
	def test_view_permissions(self):
		"""Test view permission configuration"""
		metrics_view = ESGMetricsView()
		
		# Verify base permissions are configured
		expected_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
		assert metrics_view.base_permissions == expected_permissions
	
	def test_tenant_filtering(self):
		"""Test tenant-based data filtering"""
		metrics_view = ESGMetricsView()
		
		# Mock get_user_tenant_id method
		metrics_view.get_user_tenant_id = Mock(return_value=TEST_TENANT_ID)
		
		# Verify base filter is configured for tenant isolation
		assert len(metrics_view.base_filters) > 0
		
		# Verify tenant filter structure
		tenant_filter = metrics_view.base_filters[0]
		assert tenant_filter[0] == 'tenant_id'  # Field name
		assert callable(tenant_filter[1])  # Filter function
		assert tenant_filter[2] == 'equal'  # Filter operator


@pytest.mark.integration
class TestViewIntegration:
	"""Test view integration with Flask-AppBuilder"""
	
	def test_model_view_datamodel(self):
		"""Test model view datamodel configuration"""
		from flask_appbuilder.models.sqla.interface import SQLAInterface
		
		# Test metrics view
		metrics_view = ESGMetricsView()
		assert isinstance(metrics_view.datamodel, type(SQLAInterface))
		
		# Test targets view
		targets_view = ESGTargetsView()
		assert isinstance(targets_view.datamodel, type(SQLAInterface))
		
		# Test stakeholders view
		stakeholders_view = ESGStakeholdersView()
		assert isinstance(stakeholders_view.datamodel, type(SQLAInterface))
	
	def test_view_route_configuration(self):
		"""Test view route configuration"""
		# Test executive dashboard routes
		dashboard_view = ESGExecutiveDashboardView()
		assert dashboard_view.route_base == "/esg/executive"
		assert dashboard_view.default_view == "dashboard"
		
		# Test stakeholder portal routes
		portal_view = ESGStakeholderPortalView()
		assert portal_view.route_base == "/esg/portal"
		assert portal_view.default_view == "public_dashboard"
	
	def test_view_titles_configuration(self):
		"""Test view titles and labels"""
		metrics_view = ESGMetricsView()
		
		assert metrics_view.list_title == "ESG Metrics"
		assert metrics_view.show_title == "ESG Metric Details"
		assert metrics_view.add_title == "Add ESG Metric"
		assert metrics_view.edit_title == "Edit ESG Metric"
		
		targets_view = ESGTargetsView()
		
		assert targets_view.list_title == "ESG Targets"
		assert targets_view.show_title == "ESG Target Details"
		assert targets_view.add_title == "Create ESG Target"
		assert targets_view.edit_title == "Edit ESG Target"


@pytest.mark.performance
class TestViewPerformance:
	"""Test view performance"""
	
	def test_dashboard_data_performance(self, performance_timer):
		"""Test dashboard data retrieval performance"""
		dashboard_view = ESGExecutiveDashboardView()
		
		# Mock dependencies for performance test
		mock_service = Mock(spec=ESGManagementService)
		mock_service.get_metrics = AsyncMock(return_value=[])
		
		# Mock helper methods
		dashboard_view._get_active_targets = AsyncMock(return_value=[])
		dashboard_view._get_recent_reports = AsyncMock(return_value=[])
		dashboard_view._get_stakeholder_summary = AsyncMock(return_value={})
		dashboard_view._get_ai_insights = AsyncMock(return_value={})
		
		async def test_performance():
			performance_timer.start()
			
			await dashboard_view._get_executive_dashboard_data(mock_service, TEST_USER_ID)
			
			performance_timer.stop()
		
		# Run performance test
		asyncio.run(test_performance())
		
		# Dashboard data should load quickly
		performance_timer.assert_max_time(1.0, "Dashboard data retrieval too slow")
	
	def test_serialization_performance(self, performance_timer, sample_esg_metrics: List[ESGMetric]):
		"""Test metric serialization performance"""
		dashboard_view = ESGExecutiveDashboardView()
		
		performance_timer.start()
		
		# Serialize multiple metrics
		serialized_metrics = []
		for metric in sample_esg_metrics * 10:  # Test with more metrics
			serialized = dashboard_view._serialize_metric(metric)
			serialized_metrics.append(serialized)
		
		performance_timer.stop()
		
		# Serialization should be fast
		performance_timer.assert_max_time(0.1, "Metric serialization too slow")
		
		# Verify results
		assert len(serialized_metrics) == len(sample_esg_metrics) * 10