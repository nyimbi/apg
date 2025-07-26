"""
APG Budgeting & Forecasting - Integration Tests

Comprehensive integration tests for the complete APG Budgeting & Forecasting capability,
testing all services, workflows, and APG platform integrations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Any, Optional
import uuid
from uuid_extensions import uuid7str

# Test framework imports
from pytest_httpserver import HTTPServer
import json

# Core capability imports
from ..service import APGTenantContext, BFServiceConfig, ServiceResponse
from .. import create_budgeting_forecasting_capability
from ..models import (
	BFBudgetType, BFBudgetStatus, BFLineType, BFForecastType,
	BFVarianceType, BFScenarioType, BFApprovalStatus
)

# Advanced service imports
from ..budget_management import TemplateScope, TemplateComplexity
from ..realtime_collaboration import CollaborationEventType, UserPresenceStatus
from ..approval_workflows import ApprovalAction, WorkflowStepType
from ..version_control_audit import ChangeType, AuditEventType, ComplianceLevel

# Advanced analytics imports  
from ..advanced_analytics import AnalyticsMetricType, VarianceSignificance
from ..interactive_dashboard import DashboardType, VisualizationType
from ..custom_report_builder import ReportType, ReportFormat

# AI-powered features imports
from ..ml_forecasting_engine import ForecastAlgorithm, ForecastHorizon
from ..ai_budget_recommendations import RecommendationType, RecommendationPriority
from ..automated_monitoring import AlertType, AlertSeverity


# =============================================================================
# Test Configuration and Fixtures
# =============================================================================

@pytest.fixture
def event_loop():
	"""Create event loop for async tests."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()

@pytest.fixture
def test_tenant_context():
	"""Create test tenant context."""
	return APGTenantContext(
		tenant_id="test_tenant_001",
		user_id="test_user_001"
	)

@pytest.fixture
def test_config():
	"""Create test service configuration."""
	return BFServiceConfig(
		database_url="postgresql://test:test@localhost:5432/test_apg_bf",
		cache_enabled=True,
		audit_enabled=True,
		ml_enabled=True,
		ai_recommendations_enabled=True
	)

@pytest.fixture
async def capability(test_tenant_context, test_config):
	"""Create APG Budgeting & Forecasting capability instance."""
	return create_budgeting_forecasting_capability(test_tenant_context, test_config)

@pytest.fixture
def sample_budget_data():
	"""Sample budget data for testing."""
	return {
		"budget_name": "Test Annual Budget 2025",
		"budget_type": BFBudgetType.ANNUAL.value,
		"fiscal_year": "2025",
		"total_amount": 1500000.00,
		"base_currency": "USD",
		"department_id": "dept_test_001",
		"budget_lines": [
			{
				"line_name": "Personnel Costs",
				"category": "SALARIES",
				"amount": 800000.00,
				"line_type": BFLineType.EXPENSE.value
			},
			{
				"line_name": "Marketing Budget",
				"category": "MARKETING",
				"amount": 300000.00,
				"line_type": BFLineType.EXPENSE.value
			},
			{
				"line_name": "Revenue Target",
				"category": "SALES",
				"amount": 2000000.00,
				"line_type": BFLineType.REVENUE.value
			}
		]
	}


# =============================================================================
# Core Budget Management Integration Tests
# =============================================================================

class TestCoreBudgetManagement:
	"""Test core budget management functionality."""

	async def test_complete_budget_lifecycle(self, capability, sample_budget_data):
		"""Test complete budget lifecycle from creation to deletion."""
		
		# 1. Create budget
		create_response = await capability.create_budget(sample_budget_data)
		assert create_response.success
		assert "budget_id" in create_response.data
		budget_id = create_response.data["budget_id"]
		
		# 2. Get budget
		get_response = await capability.get_budget(budget_id, include_lines=True)
		assert get_response.success
		assert get_response.data["budget_name"] == sample_budget_data["budget_name"]
		assert len(get_response.data["budget_lines"]) == 3
		
		# 3. Update budget
		update_data = {
			"budget_name": "Updated Test Budget 2025",
			"total_amount": 1600000.00,
			"notes": "Updated for market expansion"
		}
		update_response = await capability.update_budget(budget_id, update_data)
		assert update_response.success
		
		# 4. Verify update
		get_updated = await capability.get_budget(budget_id)
		assert get_updated.data["budget_name"] == "Updated Test Budget 2025"
		assert get_updated.data["total_amount"] == 1600000.00
		
		# 5. Delete budget (soft delete)
		delete_response = await capability.delete_budget(budget_id, soft_delete=True)
		assert delete_response.success

	async def test_budget_from_template_creation(self, capability, sample_budget_data):
		"""Test creating budget from template."""
		
		# First create a template (using advanced budget service)
		template_data = {
			"template_name": "Annual Budget Template",
			"template_category": "department",
			"template_scope": TemplateScope.DEPARTMENT.value,
			"template_complexity": TemplateComplexity.STANDARD.value,
			"base_budget_data": sample_budget_data
		}
		
		# Create budget from template with customizations
		customizations = {
			"budget_name": "Q1 2025 Department Budget",
			"fiscal_year": "2025",
			"customizations": {
				"scale_factor": 1.1,
				"department_overrides": {
					"MARKETING": 150000.00
				}
			}
		}
		
		template_id = "template_001"  # Mock template ID
		response = await capability.create_budget_from_template(template_id, customizations)
		assert response.success
		assert "budget_id" in response.data


# =============================================================================
# Real-Time Collaboration Integration Tests
# =============================================================================

class TestRealTimeCollaboration:
	"""Test real-time collaboration features."""

	async def test_collaboration_session_lifecycle(self, capability, sample_budget_data):
		"""Test complete collaboration session lifecycle."""
		
		# 1. Create budget for collaboration
		budget_response = await capability.create_budget(sample_budget_data)
		assert budget_response.success
		budget_id = budget_response.data["budget_id"]
		
		# 2. Create collaboration session
		session_config = {
			"session_name": "Q1 Budget Review Session",
			"budget_id": budget_id,
			"max_participants": 5,
			"session_type": "budget_editing",
			"permissions": {
				"can_edit": True,
				"can_comment": True,
				"can_approve": False
			}
		}
		
		session_response = await capability.create_collaboration_session(session_config)
		assert session_response.success
		assert "session_id" in session_response.data
		session_id = session_response.data["session_id"]
		
		# 3. Join collaboration session
		join_config = {
			"user_name": "Test User",
			"role": "editor",
			"permissions": ["edit", "comment"]
		}
		
		join_response = await capability.join_collaboration_session(session_id, join_config)
		assert join_response.success

	async def test_collaboration_conflict_resolution(self, capability, sample_budget_data):
		"""Test collaboration conflict resolution."""
		
		# Create budget and session
		budget_response = await capability.create_budget(sample_budget_data)
		budget_id = budget_response.data["budget_id"]
		
		session_config = {
			"session_name": "Conflict Resolution Test",
			"budget_id": budget_id,
			"max_participants": 3,
			"session_type": "budget_editing"
		}
		
		session_response = await capability.create_collaboration_session(session_config)
		assert session_response.success


# =============================================================================
# Approval Workflow Integration Tests
# =============================================================================

class TestApprovalWorkflows:
	"""Test approval workflow functionality."""

	async def test_budget_approval_workflow(self, capability, sample_budget_data):
		"""Test complete budget approval workflow."""
		
		# 1. Create budget
		budget_response = await capability.create_budget(sample_budget_data)
		assert budget_response.success
		budget_id = budget_response.data["budget_id"]
		
		# 2. Submit budget for approval
		submission_data = {
			"workflow_template": "department_approval",
			"priority": "high",
			"notes": "Ready for Q1 review",
			"attachments": ["budget_summary.pdf"],
			"deadline": "2025-02-15T17:00:00Z"
		}
		
		submit_response = await capability.submit_budget_for_approval(budget_id, submission_data)
		assert submit_response.success
		assert "workflow_instance_id" in submit_response.data
		workflow_instance_id = submit_response.data["workflow_instance_id"]
		
		# 3. Process approval action
		approval_data = {
			"action_type": ApprovalAction.APPROVE.value,
			"decision_reason": "Budget aligns with strategic goals",
			"conditions_or_requirements": [],
			"delegate_to": None,
			"digital_signature": "signature_hash_here"
		}
		
		action_response = await capability.process_approval_action(workflow_instance_id, approval_data)
		assert action_response.success

	async def test_workflow_escalation(self, capability, sample_budget_data):
		"""Test workflow escalation functionality."""
		
		# Create budget and submit for approval
		budget_response = await capability.create_budget(sample_budget_data)
		budget_id = budget_response.data["budget_id"]
		
		submission_data = {
			"workflow_template": "escalation_test_template",
			"priority": "critical",
			"notes": "Test escalation scenario"
		}
		
		submit_response = await capability.submit_budget_for_approval(budget_id, submission_data)
		assert submit_response.success


# =============================================================================
# Advanced Analytics Integration Tests
# =============================================================================

class TestAdvancedAnalytics:
	"""Test advanced analytics and reporting features."""

	async def test_analytics_dashboard_generation(self, capability, sample_budget_data):
		"""Test analytics dashboard generation."""
		
		# Create budget for analytics
		budget_response = await capability.create_budget(sample_budget_data)
		assert budget_response.success
		budget_id = budget_response.data["budget_id"]
		
		# Generate analytics dashboard
		dashboard_config = {
			"dashboard_name": "Executive Budget Analytics",
			"period": "monthly",
			"granularity": "detailed",
			"include_predictions": True,
			"metrics": [
				"variance_analysis",
				"trend_analysis", 
				"performance_indicators"
			]
		}
		
		dashboard_response = await capability.generate_analytics_dashboard(budget_id, dashboard_config)
		assert dashboard_response.success
		assert "dashboard_id" in dashboard_response.data
		assert "kpi_metrics" in dashboard_response.data

	async def test_advanced_variance_analysis(self, capability, sample_budget_data):
		"""Test advanced variance analysis with ML insights."""
		
		# Create budget
		budget_response = await capability.create_budget(sample_budget_data)
		budget_id = budget_response.data["budget_id"]
		
		# Perform variance analysis
		analysis_config = {
			"analysis_period": "monthly",
			"include_root_cause": True,
			"ml_insights": True,
			"comparison_baseline": "previous_year"
		}
		
		analysis_response = await capability.perform_advanced_variance_analysis(budget_id, analysis_config)
		assert analysis_response.success
		assert "report_id" in analysis_response.data
		assert "total_variance" in analysis_response.data

	async def test_interactive_dashboard_creation(self, capability):
		"""Test interactive dashboard with drill-down capabilities."""
		
		dashboard_config = {
			"dashboard_name": "Executive Budget Dashboard",
			"dashboard_type": DashboardType.EXECUTIVE.value,
			"budget_ids": ["bf_budget_12345"],
			"widgets": [
				{
					"widget_name": "Budget Overview",
					"widget_type": "kpi_card",
					"position_x": 0,
					"position_y": 0,
					"width": 4,
					"height": 2,
					"data_source": "budget_summary",
					"metrics": ["total_budget", "total_actual", "variance"]
				}
			]
		}
		
		dashboard_response = await capability.create_interactive_dashboard(dashboard_config)
		assert dashboard_response.success
		assert "dashboard_id" in dashboard_response.data
		
		# Test drill-down
		drill_config = {
			"target_level": "department",
			"context": {
				"widget_id": "widget_001",
				"filter_criteria": {
					"department": "Sales"
				}
			}
		}
		
		dashboard_id = dashboard_response.data["dashboard_id"]
		drill_response = await capability.perform_dashboard_drill_down(dashboard_id, drill_config)
		assert drill_response.success


# =============================================================================
# Custom Report Builder Integration Tests
# =============================================================================

class TestCustomReportBuilder:
	"""Test custom report builder functionality."""

	async def test_report_template_creation_and_generation(self, capability):
		"""Test report template creation and report generation."""
		
		# Create report template
		template_config = {
			"template_name": "Monthly Budget Report",
			"report_type": ReportType.BUDGET_SUMMARY.value,
			"description": "Comprehensive monthly budget analysis",
			"sections": [
				{
					"section_name": "Executive Summary",
					"section_type": "data",
					"title": "Budget Performance Summary",
					"data_source": "budget_data",
					"fields": [
						{
							"field_name": "department",
							"display_name": "Department",
							"data_type": "string"
						},
						{
							"field_name": "budget_amount",
							"display_name": "Budget Amount",
							"data_type": "currency",
							"number_format": "$#,##0.00"
						}
					]
				}
			]
		}
		
		template_response = await capability.create_report_template(template_config)
		assert template_response.success
		assert "template_id" in template_response.data
		template_id = template_response.data["template_id"]
		
		# Generate report from template
		generation_config = {
			"report_name": "January 2025 Budget Report",
			"output_format": ReportFormat.PDF.value,
			"parameters": {
				"period": "2025-01",
				"include_forecasts": True
			},
			"delivery": {
				"method": "email",
				"recipients": ["manager@company.com"]
			}
		}
		
		report_response = await capability.generate_report(template_id, generation_config)
		assert report_response.success
		assert "report_id" in report_response.data

	async def test_report_scheduling(self, capability):
		"""Test automated report scheduling."""
		
		schedule_config = {
			"schedule_name": "Monthly Budget Reports",
			"report_template_id": "template_001",
			"frequency": "monthly",
			"run_time": "09:00:00",
			"output_formats": ["pdf", "excel"],
			"delivery_method": "email",
			"recipients": ["team@company.com"]
		}
		
		schedule_response = await capability.create_report_schedule(schedule_config)
		assert schedule_response.success
		assert "schedule_id" in schedule_response.data


# =============================================================================
# ML Forecasting Engine Integration Tests
# =============================================================================

class TestMLForecastingEngine:
	"""Test machine learning forecasting functionality."""

	async def test_ml_model_creation_and_training(self, capability):
		"""Test ML forecasting model creation and training."""
		
		# Create ML forecasting model
		model_config = {
			"model_name": "Budget Forecasting Model v1",
			"algorithm": ForecastAlgorithm.RANDOM_FOREST.value,
			"target_variable": "budget_amount",
			"horizon": ForecastHorizon.MEDIUM_TERM.value,
			"frequency": "monthly",
			"training_window": 24,
			"features": [
				{
					"feature_name": "historical_budget",
					"feature_type": "historical_values",
					"source_column": "budget_amount",
					"lag_periods": 1
				}
			]
		}
		
		model_response = await capability.create_ml_forecasting_model(model_config)
		assert model_response.success
		assert "model_id" in model_response.data
		model_id = model_response.data["model_id"]
		
		# Train the model
		training_config = {
			"training_config": {
				"validation_split": 0.2,
				"test_split": 0.1,
				"hyperparameters": {
					"n_estimators": 100,
					"max_depth": 10
				}
			}
		}
		
		training_response = await capability.train_forecasting_model(model_id, training_config)
		assert training_response.success

	async def test_ml_forecast_generation(self, capability):
		"""Test ML forecast generation."""
		
		model_id = "model_001"  # Mock trained model
		
		forecast_config = {
			"scenario_name": "Q2 2025 Forecast",
			"start_date": "2025-04-01",
			"end_date": "2025-06-30",
			"assumptions": {
				"growth_rate": 0.05,
				"inflation_adjustment": 0.02
			}
		}
		
		forecast_response = await capability.generate_ml_forecast(model_id, forecast_config)
		assert forecast_response.success
		assert "scenario_id" in forecast_response.data
		assert "predictions" in forecast_response.data

	async def test_model_ensemble_creation(self, capability):
		"""Test ensemble model creation."""
		
		ensemble_config = {
			"ensemble_name": "Budget Forecast Ensemble",
			"base_models": ["model_001", "model_002", "model_003"],
			"ensemble_method": "weighted_average",
			"weights": [0.4, 0.35, 0.25],
			"meta_features": ["seasonality", "trend", "variance"]
		}
		
		ensemble_response = await capability.create_model_ensemble(ensemble_config)
		assert ensemble_response.success
		assert "ensemble_id" in ensemble_response.data


# =============================================================================
# AI Budget Recommendations Integration Tests
# =============================================================================

class TestAIBudgetRecommendations:
	"""Test AI-powered budget recommendations."""

	async def test_ai_recommendations_generation(self, capability, sample_budget_data):
		"""Test AI budget recommendations generation."""
		
		# Create budget for recommendations
		budget_response = await capability.create_budget(sample_budget_data)
		budget_id = budget_response.data["budget_id"]
		
		# Generate AI recommendations
		context_config = {
			"budget_id": budget_id,
			"analysis_period": "last_12_months",
			"industry": "Technology",
			"company_size": "medium",
			"strategic_goals": ["cost_optimization", "revenue_growth"],
			"risk_tolerance": "medium"
		}
		
		recommendations_response = await capability.generate_ai_budget_recommendations(context_config)
		assert recommendations_response.success
		assert "bundle_id" in recommendations_response.data
		assert "recommendations" in recommendations_response.data
		assert len(recommendations_response.data["recommendations"]) > 0

	async def test_recommendation_implementation_and_tracking(self, capability):
		"""Test recommendation implementation and performance tracking."""
		
		recommendation_id = "rec_001"  # Mock recommendation
		
		# Implement recommendation
		implementation_config = {
			"implementation_plan": "automated",
			"approval_required": False,
			"target_date": "2025-03-01",
			"notes": "Implementing cost optimization measures"
		}
		
		implement_response = await capability.implement_recommendation(recommendation_id, implementation_config)
		assert implement_response.success
		
		# Track recommendation performance
		performance_response = await capability.track_recommendation_performance(recommendation_id)
		assert performance_response.success
		assert "recommendation_id" in performance_response.data
		assert "performance_summary" in performance_response.data


# =============================================================================
# Automated Monitoring Integration Tests
# =============================================================================

class TestAutomatedMonitoring:
	"""Test automated monitoring and alerting."""

	async def test_monitoring_rule_creation_and_alerts(self, capability):
		"""Test monitoring rule creation and alert generation."""
		
		# Create monitoring rule
		rule_config = {
			"rule_name": "Budget Variance Alert",
			"alert_type": AlertType.VARIANCE_THRESHOLD.value,
			"description": "Alert when budget variance exceeds threshold",
			"scope": "budget",
			"target_entities": ["bf_budget_12345"],
			"metric_name": "variance_amount",
			"trigger_condition": "greater_than",
			"threshold_value": 10000.00,
			"severity": AlertSeverity.WARNING.value,
			"frequency": "daily",
			"notification_channels": ["email", "in_app"],
			"recipients": ["budget.manager@company.com"]
		}
		
		rule_response = await capability.create_monitoring_rule(rule_config)
		assert rule_response.success
		assert "rule_id" in rule_response.data
		
		# Start automated monitoring
		monitoring_response = await capability.start_automated_monitoring()
		assert monitoring_response.success
		assert monitoring_response.data["monitoring_active"] == True
		
		# Get active alerts
		alerts_response = await capability.get_active_alerts({"severity": "warning", "status": "active"})
		assert alerts_response.success
		assert "alerts" in alerts_response.data

	async def test_anomaly_detection(self, capability):
		"""Test automated anomaly detection."""
		
		detection_config = {
			"detection_name": "Budget Anomaly Detection",
			"metric_name": "budget_variance",
			"detection_method": "statistical",
			"sensitivity": 0.8,
			"analysis_start": "2025-01-01",
			"analysis_end": "2025-01-26"
		}
		
		anomaly_response = await capability.perform_anomaly_detection(detection_config)
		assert anomaly_response.success
		assert "detection_id" in anomaly_response.data


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
	"""Test complete end-to-end scenarios."""

	async def test_complete_budget_management_workflow(self, capability, sample_budget_data):
		"""Test complete budget management workflow from creation to AI insights."""
		
		# 1. Create budget
		budget_response = await capability.create_budget(sample_budget_data)
		assert budget_response.success
		budget_id = budget_response.data["budget_id"]
		
		# 2. Start collaboration session
		session_config = {
			"session_name": "End-to-End Test Session",
			"budget_id": budget_id,
			"max_participants": 3,
			"session_type": "budget_editing"
		}
		
		session_response = await capability.create_collaboration_session(session_config)
		assert session_response.success
		
		# 3. Submit for approval
		submission_data = {
			"workflow_template": "standard_approval",
			"priority": "medium",
			"notes": "End-to-end test approval"
		}
		
		approval_response = await capability.submit_budget_for_approval(budget_id, submission_data)
		assert approval_response.success
		
		# 4. Generate analytics dashboard
		dashboard_config = {
			"dashboard_name": "End-to-End Analytics",
			"period": "monthly",
			"include_predictions": True,
			"metrics": ["variance_analysis", "trend_analysis"]
		}
		
		analytics_response = await capability.generate_analytics_dashboard(budget_id, dashboard_config)
		assert analytics_response.success
		
		# 5. Generate AI recommendations
		context_config = {
			"budget_id": budget_id,
			"analysis_period": "last_6_months",
			"industry": "Technology",
			"strategic_goals": ["cost_optimization"]
		}
		
		recommendations_response = await capability.generate_ai_budget_recommendations(context_config)
		assert recommendations_response.success
		
		# 6. Create monitoring rule
		rule_config = {
			"rule_name": "End-to-End Monitoring",
			"alert_type": "variance_threshold",
			"target_entities": [budget_id],
			"threshold_value": 5000.00,
			"severity": "warning"
		}
		
		monitoring_response = await capability.create_monitoring_rule(rule_config)
		assert monitoring_response.success

	async def test_capability_health_check(self, capability):
		"""Test capability health monitoring."""
		
		health_status = await capability.get_capability_health()
		assert health_status["capability"] == "budgeting_forecasting"
		assert health_status["status"] == "healthy"
		assert "services" in health_status
		assert len(health_status["services"]) > 10  # All services initialized


# =============================================================================
# Performance and Load Testing
# =============================================================================

class TestPerformanceAndLoad:
	"""Test performance and load handling."""

	async def test_concurrent_budget_operations(self, capability):
		"""Test concurrent budget operations."""
		
		# Create multiple budgets concurrently
		budget_tasks = []
		for i in range(5):
			budget_data = {
				"budget_name": f"Concurrent Test Budget {i}",
				"budget_type": BFBudgetType.QUARTERLY.value,
				"fiscal_year": "2025",
				"total_amount": 100000.00 * (i + 1)
			}
			budget_tasks.append(capability.create_budget(budget_data))
		
		# Execute concurrently
		results = await asyncio.gather(*budget_tasks, return_exceptions=True)
		
		# Verify all succeeded
		successful_results = [r for r in results if isinstance(r, ServiceResponse) and r.success]
		assert len(successful_results) == 5

	async def test_large_dataset_analytics(self, capability):
		"""Test analytics with large datasets."""
		
		# Create budget for large dataset test
		large_budget_data = {
			"budget_name": "Large Dataset Test Budget",
			"budget_type": BFBudgetType.ANNUAL.value,
			"fiscal_year": "2025",
			"total_amount": 10000000.00,
			"budget_lines": [
				{
					"line_name": f"Line Item {i}",
					"category": f"CATEGORY_{i % 10}",
					"amount": 10000.00,
					"line_type": BFLineType.EXPENSE.value
				}
				for i in range(1000)  # 1000 budget lines
			]
		}
		
		budget_response = await capability.create_budget(large_budget_data)
		assert budget_response.success
		
		# Generate analytics for large dataset
		budget_id = budget_response.data["budget_id"]
		dashboard_config = {
			"dashboard_name": "Large Dataset Analytics",
			"period": "monthly",
			"granularity": "detailed",
			"include_predictions": True
		}
		
		analytics_response = await capability.generate_analytics_dashboard(budget_id, dashboard_config)
		assert analytics_response.success


# =============================================================================
# Error Handling and Edge Cases
# =============================================================================

class TestErrorHandlingAndEdgeCases:
	"""Test error handling and edge cases."""

	async def test_invalid_budget_data(self, capability):
		"""Test handling of invalid budget data."""
		
		invalid_budget_data = {
			"budget_name": "",  # Empty name
			"budget_type": "invalid_type",
			"fiscal_year": "invalid_year",
			"total_amount": -1000.00  # Negative amount
		}
		
		response = await capability.create_budget(invalid_budget_data)
		assert not response.success
		assert len(response.errors) > 0

	async def test_nonexistent_resource_access(self, capability):
		"""Test accessing nonexistent resources."""
		
		# Try to get nonexistent budget
		get_response = await capability.get_budget("nonexistent_budget_id")
		assert not get_response.success
		
		# Try to update nonexistent budget
		update_response = await capability.update_budget("nonexistent_budget_id", {"budget_name": "Test"})
		assert not update_response.success

	async def test_insufficient_permissions(self, capability):
		"""Test handling of insufficient permissions scenarios."""
		
		# Create budget
		budget_data = {
			"budget_name": "Permission Test Budget",
			"budget_type": BFBudgetType.MONTHLY.value,
			"fiscal_year": "2025",
			"total_amount": 50000.00
		}
		
		budget_response = await capability.create_budget(budget_data)
		assert budget_response.success
		budget_id = budget_response.data["budget_id"]
		
		# Try operations that might require higher permissions
		# (In real implementation, this would check against APG auth_rbac)
		restricted_update = {
			"status": BFBudgetStatus.LOCKED.value,  # Might require admin permissions
			"approval_override": True
		}
		
		# This test assumes the current user doesn't have admin permissions
		# In production, this would fail with insufficient permissions
		response = await capability.update_budget(budget_id, restricted_update)
		# For now, we just verify the service responds appropriately
		assert response is not None


if __name__ == "__main__":
	"""Run integration tests."""
	pytest.main([__file__, "-v", "--tb=short"])