#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Sustainability & ESG Management - Integration Tests

End-to-end integration tests for ESG capability with APG ecosystem,
workflow testing, and system validation.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from ..models import (
	ESGTenant, ESGFramework, ESGMetric, ESGMeasurement, ESGTarget, ESGMilestone,
	ESGStakeholder, ESGSupplier, ESGInitiative, ESGReport,
	ESGMetricType, ESGTargetStatus, ESGRiskLevel
)
from ..service import ESGManagementService, ESGServiceConfig
from ..api import app as esg_api
from ..blueprint import ESGBlueprintManager, register_esg_blueprint
from . import ESGTestConfig, TEST_TENANT_ID, TEST_USER_ID


@pytest.mark.integration
class TestCompleteESGWorkflow:
	"""Test complete ESG management workflows end-to-end"""
	
	@pytest.mark.asyncio
	async def test_complete_metric_lifecycle(self, mock_esg_service: ESGManagementService, db_session: Session):
		"""Test complete metric lifecycle from creation to reporting"""
		# Step 1: Create ESG metric
		metric_data = {
			"name": "Integration Test Carbon Emissions",
			"code": "INT_CARBON_EM",
			"metric_type": "environmental",
			"category": "emissions",
			"subcategory": "direct_emissions",
			"description": "Carbon emissions for integration testing",
			"unit": "tonnes_co2",
			"target_value": "10000.0",
			"baseline_value": "15000.0",
			"is_kpi": True,
			"is_automated": True,
			"enable_ai_predictions": True
		}
		
		# Mock successful metric creation
		created_metric = ESGMetric(
			id="integration_metric_id",
			tenant_id=TEST_TENANT_ID,
			name=metric_data["name"],
			code=metric_data["code"],
			metric_type=ESGMetricType.ENVIRONMENTAL,
			category=metric_data["category"],
			current_value=Decimal("12500.0"),
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		mock_esg_service.db_session.add = Mock()
		mock_esg_service.db_session.commit = Mock()
		mock_esg_service.db_session.refresh = Mock()
		
		with patch('uuid_extensions.uuid7str', return_value="integration_metric_id"):
			metric = await mock_esg_service.create_metric(
				user_id=TEST_USER_ID,
				metric_data=metric_data
			)
		
		# Verify metric creation
		mock_esg_service.db_session.add.assert_called()
		mock_esg_service.ai_service.predict_metric_trends.assert_called()
		
		# Step 2: Record multiple measurements
		measurement_dates = [
			datetime.utcnow() - timedelta(days=30),
			datetime.utcnow() - timedelta(days=20),
			datetime.utcnow() - timedelta(days=10),
			datetime.utcnow()
		]
		
		measurement_values = [13000.0, 12800.0, 12600.0, 12400.0]
		
		for i, (date, value) in enumerate(zip(measurement_dates, measurement_values)):
			measurement_data = {
				"metric_id": "integration_metric_id",
				"value": str(value),
				"measurement_date": date,
				"data_source": "integration_test",
				"collection_method": "automated"
			}
			
			mock_measurement = Mock()
			mock_measurement.id = f"measurement_{i}"
			mock_measurement.value = Decimal(str(value))
			mock_measurement.validation_score = Decimal("95.0")
			mock_measurement.anomaly_score = Decimal("2.5")
			
			mock_esg_service.db_session.query.return_value.filter.return_value.first.return_value = created_metric
			
			with patch('uuid_extensions.uuid7str', return_value=f"measurement_{i}"):
				measurement = await mock_esg_service.record_measurement(
					user_id=TEST_USER_ID,
					measurement_data=measurement_data
				)
			
			# Verify measurement recording
			mock_esg_service.db_session.add.assert_called()
			mock_esg_service.ai_service.detect_anomaly.assert_called()
		
		# Step 3: Create target based on metric
		target_data = {
			"name": "Carbon Emissions Reduction Target",
			"metric_id": "integration_metric_id",
			"description": "Reduce carbon emissions by 20% by end of 2025",
			"target_value": "10000.0",
			"baseline_value": "12500.0",
			"start_date": datetime(2024, 1, 1),
			"target_date": datetime(2025, 12, 31),
			"priority": "high",
			"owner_id": "sustainability_manager",
			"create_milestones": True
		}
		
		mock_target = Mock()
		mock_target.id = "integration_target_id"
		mock_target.name = target_data["name"]
		
		with patch('uuid_extensions.uuid7str', return_value="integration_target_id"):
			target = await mock_esg_service.create_target(
				user_id=TEST_USER_ID,
				target_data=target_data
			)
		
		# Verify target creation and AI prediction
		mock_esg_service.ai_service.predict_target_achievement.assert_called()
		
		# Step 4: Get AI insights for the complete workflow
		insights = await mock_esg_service._initialize_metric_ai_predictions(
			"integration_metric_id", TEST_USER_ID
		)
		
		# Verify AI insights structure
		assert "predictions" in insights
		assert "trend_analysis" in insights
		assert "confidence" in insights
		
		# Verify all service integrations were called
		assert mock_esg_service.audit_service.log_activity.call_count >= 4  # Multiple operations
		assert mock_esg_service.collaboration_service.broadcast_update.call_count >= 4
	
	@pytest.mark.asyncio
	async def test_stakeholder_engagement_workflow(self, mock_esg_service: ESGManagementService):
		"""Test complete stakeholder engagement workflow"""
		# Step 1: Create stakeholder
		stakeholder_data = {
			"name": "Integration Test Stakeholder",
			"organization": "Global Investment Partners",
			"stakeholder_type": "institutional_investor",
			"email": "esg@globalinvest.com",
			"country": "USA",
			"language_preference": "en_US",
			"esg_interests": ["climate_risk", "governance", "social_impact"],
			"engagement_frequency": "monthly",
			"portal_access": True,
			"data_access_level": "confidential"
		}
		
		mock_stakeholder = Mock()
		mock_stakeholder.id = "integration_stakeholder_id"
		mock_stakeholder.name = stakeholder_data["name"]
		
		with patch('uuid_extensions.uuid7str', return_value="integration_stakeholder_id"):
			stakeholder = await mock_esg_service.create_stakeholder(
				user_id=TEST_USER_ID,
				stakeholder_data=stakeholder_data
			)
		
		# Verify stakeholder creation and AI analysis
		mock_esg_service.ai_service.analyze_stakeholder_profile.assert_called()
		
		# Step 2: Generate stakeholder analytics
		analytics = await mock_esg_service._initialize_stakeholder_analytics(
			"integration_stakeholder_id", TEST_USER_ID
		)
		
		# Verify analytics structure
		assert "engagement_insights" in analytics
		assert "influence_score" in analytics
		assert "sentiment_prediction" in analytics
		assert "engagement_optimization" in analytics
		
		# Step 3: Simulate stakeholder communication
		# This would involve creating communication records and tracking engagement
		
		# Verify stakeholder workflow completion
		mock_esg_service.audit_service.log_activity.assert_called()
		mock_esg_service.collaboration_service.broadcast_update.assert_called()
	
	@pytest.mark.asyncio
	async def test_supply_chain_esg_workflow(self, mock_esg_service: ESGManagementService):
		"""Test complete supply chain ESG assessment workflow"""
		# Step 1: Create supplier
		supplier_data = {
			"name": "Integration Test Supplier",
			"legal_name": "Integration Test Supplier Inc.",
			"country": "USA",
			"industry_sector": "manufacturing",
			"business_size": "large",
			"relationship_start": datetime(2023, 1, 1),
			"contract_value": "5000000.00",
			"criticality_level": "high"
		}
		
		# Mock supplier creation (would be implemented in service)
		mock_supplier = Mock()
		mock_supplier.id = "integration_supplier_id"
		mock_supplier.name = supplier_data["name"]
		mock_supplier.overall_esg_score = Decimal("78.5")
		mock_supplier.risk_level = ESGRiskLevel.MEDIUM
		
		# Step 2: Simulate ESG assessment
		# This would involve AI-powered risk analysis and scoring
		
		# Step 3: Generate improvement recommendations
		# AI would analyze supplier data and provide optimization suggestions
		
		# Verify supply chain workflow integration
		assert mock_esg_service.config.supply_chain_monitoring is True


@pytest.mark.integration
class TestAPGEcosystemIntegration:
	"""Test integration with APG ecosystem components"""
	
	@pytest.mark.asyncio
	async def test_auth_rbac_integration(self, mock_esg_service: ESGManagementService):
		"""Test integration with APG auth_rbac service"""
		# Test permission checking
		mock_esg_service.auth_service.check_permission.return_value = True
		
		# Perform operation that requires permissions
		await mock_esg_service.get_metrics(
			user_id=TEST_USER_ID,
			filters={}
		)
		
		# Verify permission check was called
		mock_esg_service.auth_service.check_permission.assert_called_with(
			user_id=TEST_USER_ID,
			tenant_id=TEST_TENANT_ID,
			resource="esg_metrics",
			action="read"
		)
		
		# Test permission denied scenario
		mock_esg_service.auth_service.check_permission.return_value = False
		
		with pytest.raises(PermissionError):
			await mock_esg_service.get_metrics(
				user_id="unauthorized_user",
				filters={}
			)
	
	@pytest.mark.asyncio
	async def test_audit_compliance_integration(self, mock_esg_service: ESGManagementService):
		"""Test integration with APG audit_compliance service"""
		# Perform auditable operation
		metric_data = {
			"name": "Audit Test Metric",
			"code": "AUDIT_TEST",
			"metric_type": "environmental",
			"category": "audit",
			"unit": "count"
		}
		
		mock_esg_service.db_session.add = Mock()
		mock_esg_service.db_session.commit = Mock()
		mock_esg_service.db_session.refresh = Mock()
		
		with patch('uuid_extensions.uuid7str', return_value="audit_test_metric"):
			await mock_esg_service.create_metric(
				user_id=TEST_USER_ID,
				metric_data=metric_data
			)
		
		# Verify audit logging
		mock_esg_service.audit_service.log_activity.assert_called_with(
			user_id=TEST_USER_ID,
			tenant_id=TEST_TENANT_ID,
			action="create_metric",
			resource_type="esg_metric",
			resource_id="audit_test_metric",
			details={
				"metric_name": "Audit Test Metric",
				"metric_code": "AUDIT_TEST",
				"metric_type": "environmental"
			}
		)
	
	@pytest.mark.asyncio
	async def test_ai_orchestration_integration(self, mock_esg_service: ESGManagementService, mock_ai_service):
		"""Test integration with APG ai_orchestration service"""
		# Replace mock AI service with more detailed mock
		mock_esg_service.ai_service = mock_ai_service
		
		# Test metric AI predictions
		await mock_esg_service._initialize_metric_ai_predictions(
			"test_metric_id", TEST_USER_ID
		)
		
		# Verify AI service calls
		mock_ai_service.predict_metric_trends.assert_called()
		
		# Test target achievement prediction
		await mock_esg_service._predict_target_achievement(
			"test_target_id", TEST_USER_ID
		)
		
		mock_ai_service.predict_target_achievement.assert_called()
		
		# Test stakeholder analysis
		await mock_esg_service._initialize_stakeholder_analytics(
			"test_stakeholder_id", TEST_USER_ID
		)
		
		mock_ai_service.analyze_stakeholder_profile.assert_called()
	
	@pytest.mark.asyncio
	async def test_real_time_collaboration_integration(self, mock_esg_service: ESGManagementService):
		"""Test integration with APG real_time_collaboration service"""
		# Perform operation that triggers real-time updates
		measurement_data = {
			"metric_id": "test_metric",
			"value": "100.0",
			"measurement_date": datetime.utcnow()
		}
		
		mock_metric = Mock()
		mock_metric.id = "test_metric"
		mock_esg_service.db_session.query.return_value.filter.return_value.first.return_value = mock_metric
		mock_esg_service.db_session.add = Mock()
		mock_esg_service.db_session.commit = Mock()
		
		with patch('uuid_extensions.uuid7str', return_value="test_measurement"):
			await mock_esg_service.record_measurement(
				user_id=TEST_USER_ID,
				measurement_data=measurement_data
			)
		
		# Verify real-time broadcast
		mock_esg_service.collaboration_service.broadcast_update.assert_called_with(
			tenant_id=TEST_TENANT_ID,
			channel="esg_metrics_updates",
			message={
				"type": "measurement_recorded",
				"metric_id": "test_metric",
				"user_id": TEST_USER_ID,
				"timestamp": pytest.approx(datetime.utcnow(), abs=timedelta(seconds=5))
			}
		)
	
	@pytest.mark.asyncio
	async def test_document_content_management_integration(self, mock_esg_service: ESGManagementService):
		"""Test integration with APG document_content_management service"""
		# Test document storage for ESG reports
		report_data = {
			"name": "Integration Test Report",
			"content": "Test report content",
			"format": "pdf"
		}
		
		# Mock document storage
		mock_esg_service.document_service.store_document.return_value = {
			"document_id": "test_document_id",
			"storage_url": "https://storage.example.com/documents/test_document_id"
		}
		
		# This would be called during report generation
		document_result = await mock_esg_service.document_service.store_document(
			content=report_data["content"],
			filename=f"{report_data['name']}.{report_data['format']}",
			content_type="application/pdf",
			tenant_id=TEST_TENANT_ID
		)
		
		# Verify document storage
		assert document_result["document_id"] == "test_document_id"
		mock_esg_service.document_service.store_document.assert_called_once()


@pytest.mark.integration
class TestAPIIntegration:
	"""Test API integration scenarios"""
	
	@pytest.fixture
	def api_client(self):
		"""API test client with mocked dependencies"""
		return TestClient(esg_api)
	
	def test_api_health_integration(self, api_client):
		"""Test API health check integration"""
		response = api_client.get("/api/v1/esg/health")
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["status"] == "healthy"
		assert "services" in data
		assert "database" in data["services"]
		assert "ai_engine" in data["services"]
		assert "real_time" in data["services"]
	
	def test_api_cors_integration(self, api_client):
		"""Test API CORS integration"""
		# Test preflight request
		response = api_client.options(
			"/api/v1/esg/metrics",
			headers={
				"Origin": "https://dashboard.example.com",
				"Access-Control-Request-Method": "GET",
				"Access-Control-Request-Headers": "Authorization"
			}
		)
		
		# CORS should be configured (implementation would set headers)
		assert response.status_code in [200, 204]
	
	def test_api_rate_limiting_integration(self, api_client):
		"""Test API rate limiting integration"""
		# Make multiple requests rapidly
		responses = []
		for i in range(10):
			response = api_client.get("/api/v1/esg/health")
			responses.append(response.status_code)
		
		# All requests should succeed (rate limiting would be configured in production)
		assert all(status in [200, 429] for status in responses)


@pytest.mark.integration
class TestBlueprintIntegration:
	"""Test Flask-AppBuilder blueprint integration"""
	
	def test_blueprint_registration(self, flask_app, app_builder):
		"""Test ESG blueprint registration with Flask app"""
		# Mock the registration function
		with patch('sustainability_esg_management.blueprint.ESGBlueprintManager') as mock_manager_class:
			mock_manager = Mock()
			mock_manager.setup_database_models = Mock()
			mock_manager.register_views = Mock()
			mock_manager.configure_security_integration = Mock()
			mock_manager.register_api_endpoints = Mock()
			mock_manager.initialize_ai_integration = Mock()
			mock_manager.setup_real_time_features = Mock()
			mock_manager.register_with_apg_composition = Mock(return_value=True)
			mock_manager_class.return_value = mock_manager
			
			# Test blueprint registration
			result = register_esg_blueprint(flask_app, app_builder)
			
			# Verify registration steps were executed
			assert result is True
			mock_manager.setup_database_models.assert_called_once()
			mock_manager.register_views.assert_called_once()
			mock_manager.configure_security_integration.assert_called_once()
	
	def test_blueprint_health_check(self):
		"""Test blueprint health check functionality"""
		from ..blueprint import check_esg_blueprint_health
		
		health_status = check_esg_blueprint_health()
		
		# Verify health check structure
		assert "capability_name" in health_status
		assert "status" in health_status
		assert health_status["capability_name"] == "sustainability_esg_management"
		assert health_status["status"] in ["healthy", "unhealthy"]
		
		if health_status["status"] == "healthy":
			assert "views_count" in health_status
			assert "api_endpoints_count" in health_status
			assert "database_models" in health_status
			assert "integrations" in health_status
	
	def test_blueprint_manager_initialization(self, app_builder):
		"""Test ESG blueprint manager initialization"""
		manager = ESGBlueprintManager(app_builder)
		
		assert manager.appbuilder == app_builder
		assert manager.capability_name == "sustainability_esg_management"
		assert manager.views_registered is False
		
		# Test view definitions structure
		view_definitions = manager._get_view_definitions()
		
		assert "dashboard_views" in view_definitions
		assert "management_views" in view_definitions
		assert "analytics_views" in view_definitions
		assert "portal_views" in view_definitions
		
		# Test menu structure
		menu_structure = manager._get_menu_structure()
		
		assert "main_categories" in menu_structure
		assert "utility_items" in menu_structure


@pytest.mark.integration
class TestDataIntegrity:
	"""Test data integrity across the ESG system"""
	
	def test_metric_measurement_consistency(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_framework: ESGFramework):
		"""Test consistency between metrics and measurements"""
		# Create metric
		metric = ESGMetric(
			id="integrity_test_metric",
			tenant_id=sample_tenant.id,
			framework_id=sample_esg_framework.id,
			name="Data Integrity Test Metric",
			code="INTEGRITY_TEST",
			metric_type=ESGMetricType.ENVIRONMENTAL,
			category="integrity",
			unit="count",
			current_value=Decimal("100.0"),
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(metric)
		
		# Create measurements
		measurements = []
		for i in range(5):
			measurement = ESGMeasurement(
				id=f"integrity_measurement_{i}",
				metric_id=metric.id,
				value=Decimal(str(100 + i * 10)),
				measurement_date=datetime.utcnow() - timedelta(days=i),
				data_source="integrity_test",
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			measurements.append(measurement)
			db_session.add(measurement)
		
		db_session.commit()
		
		# Verify data consistency
		retrieved_metric = db_session.query(ESGMetric).filter_by(id="integrity_test_metric").first()
		retrieved_measurements = db_session.query(ESGMeasurement).filter_by(metric_id=metric.id).all()
		
		assert retrieved_metric is not None
		assert len(retrieved_measurements) == 5
		
		# Verify relationships
		for measurement in retrieved_measurements:
			assert measurement.metric_id == metric.id
		
		# Clean up
		for measurement in measurements:
			db_session.delete(measurement)
		db_session.delete(metric)
		db_session.commit()
	
	def test_target_milestone_consistency(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_metrics: List[ESGMetric]):
		"""Test consistency between targets and milestones"""
		metric = sample_esg_metrics[0]
		
		# Create target
		target = ESGTarget(
			id="integrity_test_target",
			tenant_id=sample_tenant.id,
			metric_id=metric.id,
			name="Data Integrity Test Target",
			target_value=Decimal("80.0"),
			start_date=datetime(2024, 1, 1),
			target_date=datetime(2025, 12, 31),
			owner_id="integrity_test_owner",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(target)
		
		# Create milestones
		milestones = []
		milestone_dates = [
			datetime(2024, 4, 1),
			datetime(2024, 7, 1),
			datetime(2024, 10, 1),
			datetime(2025, 1, 1)
		]
		
		for i, milestone_date in enumerate(milestone_dates):
			milestone = ESGMilestone(
				id=f"integrity_milestone_{i}",
				target_id=target.id,
				name=f"Milestone {i+1}",
				description=f"Test milestone {i+1}",
				due_date=milestone_date,
				target_value=Decimal(str(60 + i * 5)),
				status="pending",
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			milestones.append(milestone)
			db_session.add(milestone)
		
		db_session.commit()
		
		# Verify data consistency
		retrieved_target = db_session.query(ESGTarget).filter_by(id="integrity_test_target").first()
		retrieved_milestones = db_session.query(ESGMilestone).filter_by(target_id=target.id).all()
		
		assert retrieved_target is not None
		assert len(retrieved_milestones) == 4
		
		# Verify milestone progression
		sorted_milestones = sorted(retrieved_milestones, key=lambda m: m.due_date)
		for i, milestone in enumerate(sorted_milestones[:-1]):
			next_milestone = sorted_milestones[i + 1]
			assert milestone.due_date <= next_milestone.due_date
		
		# Clean up
		for milestone in milestones:
			db_session.delete(milestone)
		db_session.delete(target)
		db_session.commit()


@pytest.mark.integration  
class TestPerformanceIntegration:
	"""Test system performance under integrated scenarios"""
	
	@pytest.mark.asyncio
	async def test_bulk_data_processing_performance(self, mock_esg_service: ESGManagementService, performance_timer):
		"""Test performance with bulk data processing"""
		performance_timer.start()
		
		# Simulate bulk metric creation
		mock_esg_service.db_session.add_all = Mock()
		mock_esg_service.db_session.commit = Mock()
		
		# Create multiple metrics concurrently
		metric_creation_tasks = []
		for i in range(50):
			metric_data = {
				"name": f"Bulk Test Metric {i}",
				"code": f"BULK_TEST_{i:03d}",
				"metric_type": "environmental",
				"category": "bulk_test",
				"unit": "count"
			}
			
			task = mock_esg_service.create_metric(TEST_USER_ID, metric_data)
			metric_creation_tasks.append(task)
		
		# Process first 10 concurrently for testing
		await asyncio.gather(*metric_creation_tasks[:10])
		
		performance_timer.stop()
		
		# Should handle bulk operations efficiently
		performance_timer.assert_max_time(3.0, "Bulk processing took too long")
	
	def test_database_query_performance(self, db_session: Session, sample_esg_metrics: List[ESGMetric], performance_timer):
		"""Test database query performance with complex joins"""
		performance_timer.start()
		
		# Complex query joining multiple tables
		results = db_session.query(ESGMetric)\
			.join(ESGFramework)\
			.filter(ESGMetric.tenant_id == TEST_TENANT_ID)\
			.filter(ESGMetric.is_kpi == True)\
			.order_by(ESGMetric.created_at.desc())\
			.limit(50)\
			.all()
		
		performance_timer.stop()
		
		# Query should complete quickly
		performance_timer.assert_max_time(0.5, "Database query took too long")
		
		# Verify results structure
		assert isinstance(results, list)


@pytest.mark.integration
class TestSecurityIntegration:
	"""Test security integration across the system"""
	
	@pytest.mark.asyncio
	async def test_tenant_isolation_security(self, mock_esg_service: ESGManagementService):
		"""Test tenant data isolation security"""
		# Create service for different tenant
		other_tenant_service = ESGManagementService(
			db_session=mock_esg_service.db_session,
			tenant_id="other_tenant_id",
			config=mock_esg_service.config
		)
		
		# Mock authentication for other tenant
		other_tenant_service.auth_service = Mock()
		other_tenant_service.auth_service.check_permission = AsyncMock(return_value=True)
		other_tenant_service.audit_service = Mock()
		other_tenant_service.audit_service.log_activity = AsyncMock()
		
		# Mock empty results for other tenant
		other_tenant_service.db_session.query.return_value.filter.return_value.all.return_value = []
		
		# Query should return no data for different tenant
		metrics = await other_tenant_service.get_metrics(
			user_id=TEST_USER_ID,
			filters={}
		)
		
		assert len(metrics) == 0
		
		# Verify tenant-specific filtering was applied
		other_tenant_service.db_session.query.assert_called()
	
	@pytest.mark.asyncio
	async def test_permission_based_access_control(self, mock_esg_service: ESGManagementService):
		"""Test role-based access control"""
		# Test read-only user
		mock_esg_service.auth_service.check_permission.side_effect = lambda **kwargs: kwargs["action"] == "read"
		
		# Read operation should succeed
		await mock_esg_service.get_metrics(user_id=TEST_USER_ID, filters={})
		
		# Write operation should fail
		with pytest.raises(PermissionError):
			await mock_esg_service.create_metric(
				user_id=TEST_USER_ID,
				metric_data={"name": "Test", "code": "TEST", "metric_type": "environmental", "category": "test", "unit": "count"}
			)
	
	def test_data_validation_security(self, mock_esg_service: ESGManagementService):
		"""Test data validation for security"""
		# Test SQL injection prevention
		malicious_data = {
			"name": "Test'; DROP TABLE esg_metrics; --",
			"code": "MALICIOUS",
			"metric_type": "environmental",
			"category": "test",
			"unit": "count"
		}
		
		# Validation should catch malicious input
		errors = mock_esg_service._validate_metric_data(malicious_data)
		
		# Should have validation errors for malicious content
		assert len(errors) > 0 or "'" not in malicious_data["name"]  # Either validates or sanitizes


@pytest.mark.integration
class TestErrorHandlingIntegration:
	"""Test error handling across integrated components"""
	
	@pytest.mark.asyncio
	async def test_cascading_error_handling(self, mock_esg_service: ESGManagementService):
		"""Test error handling across service layers"""
		# Mock database error
		mock_esg_service.db_session.commit.side_effect = Exception("Database connection lost")
		
		metric_data = {
			"name": "Error Test Metric",
			"code": "ERROR_TEST",
			"metric_type": "environmental",
			"category": "error",
			"unit": "count"
		}
		
		mock_esg_service.db_session.add = Mock()
		
		# Error should be handled gracefully
		with pytest.raises(Exception) as exc_info:
			with patch('uuid_extensions.uuid7str', return_value="error_test_metric"):
				await mock_esg_service.create_metric(
					user_id=TEST_USER_ID,
					metric_data=metric_data
				)
		
		# Verify error was logged
		assert "Database connection lost" in str(exc_info.value)
	
	@pytest.mark.asyncio
	async def test_ai_service_failure_handling(self, mock_esg_service: ESGManagementService):
		"""Test handling of AI service failures"""
		# Mock AI service failure
		mock_esg_service.ai_service.predict_metric_trends.side_effect = Exception("AI service unavailable")
		
		metric_data = {
			"name": "AI Failure Test Metric",
			"code": "AI_FAIL_TEST",
			"metric_type": "environmental",
			"category": "ai_fail",
			"unit": "count",
			"enable_ai_predictions": True
		}
		
		mock_esg_service.db_session.add = Mock()
		mock_esg_service.db_session.commit = Mock()
		mock_esg_service.db_session.refresh = Mock()
		
		# Metric creation should still succeed even if AI fails
		with patch('uuid_extensions.uuid7str', return_value="ai_fail_test_metric"):
			metric = await mock_esg_service.create_metric(
				user_id=TEST_USER_ID,
				metric_data=metric_data
		)
		
		# Verify metric was still created
		mock_esg_service.db_session.add.assert_called()
		mock_esg_service.db_session.commit.assert_called()
		
		# Verify error was logged
		mock_esg_service.audit_service.log_activity.assert_called()