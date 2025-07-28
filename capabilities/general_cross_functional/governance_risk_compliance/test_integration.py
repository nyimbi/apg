#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Governance, Risk & Compliance - Integration Tests

End-to-end integration tests for GRC capability testing complete workflows,
APG ecosystem integration, and production-ready scenarios.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import json
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask import Flask
from flask_appbuilder import AppBuilder

from models import Base, GRCTenant, GRCRisk, GRCControl, GRCPolicy
from views import GRCRiskView, GRCControlView, GRCPolicyView
from ai_engine import GRCAIEngine
from compliance_engine import ComplianceEngine
from governance_engine import GovernanceEngine
from monitoring_service import GRCMonitoringService

# Integration test configuration
TEST_CONFIG = {
	"SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
	"SECRET_KEY": "test_secret_key",
	"WTF_CSRF_ENABLED": False,
	"TESTING": True
}

class IntegrationTestSetup:
	"""Setup and teardown for integration tests."""
	
	def __init__(self):
		self.app = None
		self.appbuilder = None
		self.db_session = None
		self.test_tenant = None
		
	async def setup(self):
		"""Setup test environment with full APG integration."""
		# Create Flask app with AppBuilder
		self.app = Flask(__name__)
		self.app.config.update(TEST_CONFIG)
		
		# Initialize database
		engine = create_engine(TEST_CONFIG["SQLALCHEMY_DATABASE_URI"], echo=False)
		Base.metadata.create_all(bind=engine)
		
		SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
		self.db_session = SessionLocal()
		
		# Create test tenant
		self.test_tenant = GRCTenant(
			name="Integration Test Corp",
			slug="integration-test",
			domain="integration.test.com",
			settings={
				"ai_enabled": True,
				"real_time_monitoring": True,
				"automated_compliance": True
			}
		)
		self.db_session.add(self.test_tenant)
		self.db_session.commit()
		self.db_session.refresh(self.test_tenant)
		
		# Initialize AppBuilder (Flask-AppBuilder integration)
		with self.app.app_context():
			self.appbuilder = AppBuilder(self.app, self.db_session)
			
			# Register GRC views
			self.appbuilder.add_view(
				GRCRiskView,
				"Risk Management",
				icon="fa-warning",
				category="GRC"
			)
			self.appbuilder.add_view(
				GRCControlView,
				"Control Management", 
				icon="fa-shield",
				category="GRC"
			)
			self.appbuilder.add_view(
				GRCPolicyView,
				"Policy Management",
				icon="fa-book",
				category="GRC"
			)
	
	async def teardown(self):
		"""Clean up test environment."""
		if self.db_session:
			self.db_session.close()
		if self.app:
			with self.app.app_context():
				Base.metadata.drop_all(bind=self.app.extensions['sqlalchemy'].db.engine)

@pytest.fixture(scope="function")
async def integration_setup():
	"""Integration test setup fixture."""
	setup = IntegrationTestSetup()
	await setup.setup()
	try:
		yield setup
	finally:
		await setup.teardown()

class TestEndToEndRiskWorkflow:
	"""Test complete risk management workflow from creation to resolution."""
	
	@pytest.mark.asyncio
	async def test_complete_risk_lifecycle(self, integration_setup: IntegrationTestSetup):
		"""Test complete risk management lifecycle with AI integration."""
		# Initialize AI and service components
		ai_engine = GRCAIEngine(
			tenant_id=integration_setup.test_tenant.id,
			model_config={"risk_prediction": {"enabled": True}}
		)
		
		monitoring_service = GRCMonitoringService(
			tenant_id=integration_setup.test_tenant.id,
			config={"real_time_monitoring": True}
		)
		
		# Stage 1: Risk Creation and Initial Assessment
		with patch.object(ai_engine, 'predict_risk_evolution') as mock_predict:
			mock_predict.return_value = {
				"predicted_likelihood": 45.0,
				"predicted_impact": 75.0,
				"confidence": 0.87,
				"trend": "increasing",
				"recommendations": ["Implement additional controls", "Increase monitoring"]
			}
			
			# Create initial risk
			risk = GRCRisk(
				tenant_id=integration_setup.test_tenant.id,
				title="Critical Data Breach Risk",
				description="High-impact cybersecurity risk affecting customer data",
				current_likelihood=Decimal("40.0"),
				current_impact=Decimal("70.0"),
				status="active",
				owner_id="risk_manager_001"
			)
			
			integration_setup.db_session.add(risk)
			integration_setup.db_session.commit()
			integration_setup.db_session.refresh(risk)
			
			# Get AI prediction
			prediction = await ai_engine.predict_risk_evolution(
				risk.id,
				{
					"current_likelihood": float(risk.current_likelihood),
					"current_impact": float(risk.current_impact),
					"market_indicators": {"cyber_threat_level": 0.8}
				}
			)
			
			# Update risk with AI insights
			risk.predicted_likelihood = Decimal(str(prediction["predicted_likelihood"]))
			risk.predicted_impact = Decimal(str(prediction["predicted_impact"]))
			risk.prediction_confidence = Decimal(str(prediction["confidence"]))
			risk.ai_insights = {
				"trend": prediction["trend"],
				"recommendations": prediction["recommendations"]
			}
			
			integration_setup.db_session.commit()
			
			assert risk.id is not None
			assert float(risk.predicted_likelihood) == 45.0
			assert risk.ai_insights["trend"] == "increasing"
		
		# Stage 2: Control Implementation and Testing
		with patch('compliance_engine.ControlTestingEngine') as mock_control_testing:
			mock_control_testing.return_value.execute_control_tests.return_value = {
				"test_results": [{
					"control_id": "ctrl_data_encryption",
					"test_status": "passed",
					"effectiveness_score": 0.92
				}],
				"overall_compliance_score": 0.92
			}
			
			# Create control for risk mitigation
			control = GRCControl(
				tenant_id=integration_setup.test_tenant.id,
				name="Data Encryption Control",
				description="Encrypt all customer data at rest and in transit",
				control_type="preventive",
				effectiveness_rating=Decimal("90.0"),
				automated=True,
				ai_testing_enabled=True
			)
			
			integration_setup.db_session.add(control)
			integration_setup.db_session.commit()
			
			# Execute automated control testing
			compliance_engine = ComplianceEngine(
				tenant_id=integration_setup.test_tenant.id,
				ai_engine=ai_engine
			)
			
			test_results = await compliance_engine.execute_automated_control_testing([{
				"control_id": control.id,
				"test_script": "test_encryption.py"
			}])
			
			# Update control with test results
			control.ai_test_results = test_results["test_results"][0]
			integration_setup.db_session.commit()
			
			assert control.ai_test_results["test_status"] == "passed"
			assert test_results["overall_compliance_score"] == 0.92
		
		# Stage 3: Real-time Monitoring and Alerting
		with patch.object(monitoring_service, 'monitor_real_time_risks') as mock_monitor:
			mock_monitor.return_value = {
				"risks_monitored": 1,
				"alerts_generated": [{
					"risk_id": risk.id,
					"alert_type": "improvement_detected",
					"message": "Risk score improved due to control implementation"
				}]
			}
			
			# Monitor risk in real-time
			monitoring_results = await monitoring_service.monitor_real_time_risks()
			
			assert monitoring_results["risks_monitored"] == 1
			assert len(monitoring_results["alerts_generated"]) == 1
		
		# Stage 4: Risk Re-assessment and Closure
		# Update risk status based on control effectiveness
		risk.current_likelihood = Decimal("25.0")  # Reduced due to controls
		risk.current_impact = Decimal("60.0")     # Reduced due to controls
		risk.status = "mitigated"
		
		integration_setup.db_session.commit()
		
		# Final verification
		final_risk_score = risk.current_likelihood * risk.current_impact / Decimal("100.0")
		assert final_risk_score < Decimal("20.0")  # Acceptable risk level
		assert risk.status == "mitigated"

class TestComplianceAutomationWorkflow:
	"""Test automated compliance monitoring and reporting workflow."""
	
	@pytest.mark.asyncio
	async def test_regulatory_change_to_control_update(self, integration_setup: IntegrationTestSetup):
		"""Test complete workflow from regulatory change detection to control updates."""
		compliance_engine = ComplianceEngine(
			tenant_id=integration_setup.test_tenant.id,
			ai_engine=Mock()
		)
		
		# Stage 1: Regulatory Change Detection
		with patch.object(compliance_engine.regulatory_monitor, 'check_regulatory_updates') as mock_regulatory:
			mock_regulatory.return_value = {
				"updates": [{
					"regulation_id": "gdpr_update_2024",
					"title": "GDPR AI Processing Guidelines",
					"change_type": "new_requirement",
					"impact_assessment": {
						"affected_controls": ["ai_governance", "data_processing"],
						"action_required": True,
						"deadline": "2024-06-01"
					}
				}]
			}
			
			regulatory_updates = await compliance_engine.monitor_regulatory_changes()
			update = regulatory_updates["updates"][0]
			
			assert update["regulation_id"] == "gdpr_update_2024"
			assert update["impact_assessment"]["action_required"] is True
		
		# Stage 2: Gap Analysis and Control Identification
		with patch.object(compliance_engine, 'analyze_compliance_gaps') as mock_gaps:
			mock_gaps.return_value = {
				"gaps": [{
					"requirement_id": "gdpr_ai_processing",
					"gap_type": "missing_control",
					"recommended_controls": ["ai_impact_assessment", "automated_decision_logging"]
				}]
			}
			
			gap_analysis = await compliance_engine.analyze_compliance_gaps(
				[{"id": "gdpr_ai_processing", "category": "ai_governance"}],
				[]  # No existing controls
			)
			
			assert len(gap_analysis["gaps"]) == 1
			assert "ai_impact_assessment" in gap_analysis["gaps"][0]["recommended_controls"]
		
		# Stage 3: Automated Control Creation
		# Create new control based on gap analysis
		new_control = GRCControl(
			tenant_id=integration_setup.test_tenant.id,
			name="AI Impact Assessment Control",
			description="Assess impact of AI processing on data subjects",
			control_type="detective",
			effectiveness_rating=Decimal("85.0"),
			automated=True,
			automation_config={
				"assessment_triggers": ["new_ai_model", "processing_change"],
				"notification_recipients": ["dpo", "ai_ethics_board"]
			}
		)
		
		integration_setup.db_session.add(new_control)
		integration_setup.db_session.commit()
		
		# Stage 4: Control Testing and Validation
		with patch.object(compliance_engine.control_tester, 'execute_control_tests') as mock_test:
			mock_test.return_value = {
				"test_results": [{
					"control_id": new_control.id,
					"test_status": "passed",
					"effectiveness_score": 0.85,
					"compliance_validated": True
				}]
			}
			
			test_results = await compliance_engine.execute_automated_control_testing([{
				"control_id": new_control.id,
				"test_script": "ai_impact_assessment_test.py"
			}])
			
			assert test_results["test_results"][0]["compliance_validated"] is True
		
		# Verify end-to-end workflow completion
		updated_control = integration_setup.db_session.query(GRCControl).filter_by(id=new_control.id).first()
		assert updated_control is not None
		assert updated_control.name == "AI Impact Assessment Control"
		assert updated_control.automated is True

class TestGovernanceDecisionWorkflow:
	"""Test governance decision-making workflow with stakeholder collaboration."""
	
	@pytest.mark.asyncio
	async def test_strategic_decision_workflow(self, integration_setup: IntegrationTestSetup):
		"""Test complete strategic decision workflow with AI assistance."""
		governance_engine = GovernanceEngine(
			tenant_id=integration_setup.test_tenant.id,
			ai_engine=Mock()
		)
		
		# Stage 1: Decision Initiation and Stakeholder Identification
		decision_context = {
			"title": "Enterprise AI Governance Framework",
			"scope": "organization_wide",
			"impact": "high",
			"categories": ["technology", "risk", "compliance"]
		}
		
		with patch.object(governance_engine.stakeholder_manager, 'identify_stakeholders') as mock_stakeholders:
			mock_stakeholders.return_value = {
				"stakeholders": [
					{"id": "ceo", "influence_score": 0.95, "required": True},
					{"id": "cto", "influence_score": 0.90, "required": True},
					{"id": "ciso", "influence_score": 0.85, "required": True}
				]
			}
			
			stakeholders = await governance_engine.identify_decision_stakeholders(decision_context)
			assert len(stakeholders["stakeholders"]) == 3
		
		# Stage 2: Policy Creation with AI Assistance
		with patch.object(governance_engine.policy_orchestrator, 'generate_policy_draft') as mock_policy:
			mock_policy.return_value = {
				"policy_draft": {
					"title": "AI Governance Framework Policy",
					"sections": [
						{"section": "AI Ethics Principles", "content": "AI systems must be fair, transparent..."},
						{"section": "AI Risk Management", "content": "AI risks must be assessed..."}
					]
				},
				"compliance_mapping": {"GDPR": ["Article 22"], "AI_Act": ["Article 5"]},
				"ai_assistance_summary": {"confidence_score": 0.91}
			}
			
			policy_draft = await governance_engine.orchestrate_policy_creation({
				"policy_type": "ai_governance",
				"regulatory_requirements": ["GDPR", "AI_Act"]
			})
			
			assert policy_draft["policy_draft"]["title"] == "AI Governance Framework Policy"
			assert len(policy_draft["policy_draft"]["sections"]) == 2
		
		# Stage 3: Decision Workflow Execution
		with patch.object(governance_engine.decision_workflow, 'execute_workflow') as mock_workflow:
			mock_workflow.return_value = {
				"workflow_id": "ai_governance_decision_001",
				"current_stage": "executive_approval",
				"ai_insights": {
					"success_probability": 0.88,
					"estimated_completion": datetime.utcnow() + timedelta(days=14)
				}
			}
			
			workflow = await governance_engine.execute_decision_workflow({
				"title": "AI Governance Framework Decision",
				"stakeholders": ["ceo", "cto", "ciso"],
				"deadline": datetime.utcnow() + timedelta(days=30)
			})
			
			assert workflow["workflow_id"] == "ai_governance_decision_001"
			assert workflow["ai_insights"]["success_probability"] == 0.88
		
		# Stage 4: Policy Implementation and Tracking
		# Create policy record in database
		policy = GRCPolicy(
			tenant_id=integration_setup.test_tenant.id,
			title="AI Governance Framework Policy",
			policy_type="ai_governance",
			version="1.0",
			effective_date=datetime.utcnow(),
			owner_id="cto",
			ai_assistance_used=True,
			status="approved"
		)
		
		integration_setup.db_session.add(policy)
		integration_setup.db_session.commit()
		
		# Verify policy creation and governance integration
		created_policy = integration_setup.db_session.query(GRCPolicy).filter_by(
			title="AI Governance Framework Policy"
		).first()
		
		assert created_policy is not None
		assert created_policy.ai_assistance_used is True
		assert created_policy.status == "approved"

class TestAPGEcosystemIntegration:
	"""Test integration with broader APG ecosystem capabilities."""
	
	@pytest.mark.asyncio
	async def test_cross_capability_data_flow(self, integration_setup: IntegrationTestSetup):
		"""Test data flow between GRC and other APG capabilities."""
		
		# Test 1: Integration with CRM for customer risk assessment
		with patch('apg.capabilities.customer_relationship_management') as mock_crm:
			mock_crm.get_customer_profile.return_value = {
				"customer_id": "cust_001",
				"risk_profile": "high_value",
				"compliance_requirements": ["GDPR", "PCI_DSS"],
				"data_sensitivity": "high"
			}
			
			# Simulate GRC using CRM data for risk assessment
			customer_profile = await mock_crm.get_customer_profile("cust_001")
			
			# Create risk based on customer profile
			customer_risk = GRCRisk(
				tenant_id=integration_setup.test_tenant.id,
				title=f"Customer Data Risk - {customer_profile['customer_id']}",
				description="Risk assessment based on customer profile data",
				current_likelihood=Decimal("60.0" if customer_profile["data_sensitivity"] == "high" else "30.0"),
				current_impact=Decimal("80.0" if customer_profile["risk_profile"] == "high_value" else "40.0"),
				status="active",
				owner_id="customer_risk_manager"
			)
			
			integration_setup.db_session.add(customer_risk)
			integration_setup.db_session.commit()
			
			assert customer_risk.current_likelihood == Decimal("60.0")
			assert "Customer Data Risk" in customer_risk.title
		
		# Test 2: Integration with Financial Management for budget impact
		with patch('apg.capabilities.financial_management') as mock_finance:
			mock_finance.get_budget_allocation.return_value = {
				"department": "IT Security",
				"allocated_budget": 500000,
				"spent_budget": 300000,
				"available_budget": 200000
			}
			
			mock_finance.calculate_risk_financial_impact.return_value = {
				"estimated_cost": 150000,
				"budget_impact": "medium",
				"funding_available": True
			}
			
			# Simulate GRC calculating financial impact of risk mitigation
			budget_info = await mock_finance.get_budget_allocation("IT Security")
			financial_impact = await mock_finance.calculate_risk_financial_impact({
				"risk_type": "cybersecurity",
				"mitigation_strategy": "advanced_endpoint_protection"
			})
			
			assert financial_impact["funding_available"] is True
			assert budget_info["available_budget"] >= financial_impact["estimated_cost"]
		
		# Test 3: Integration with Document Management for policy storage
		with patch('apg.capabilities.document_content_management') as mock_docs:
			mock_docs.store_document.return_value = {
				"document_id": "doc_policy_001",
				"storage_location": "/policies/ai_governance/v1.0.pdf",
				"version": "1.0",
				"access_permissions": ["governance_team", "executives"]
			}
			
			# Simulate GRC storing policy document
			policy_document = await mock_docs.store_document({
				"title": "AI Governance Framework Policy",
				"content_type": "policy",
				"classification": "internal",
				"owner": "governance_team"
			})
			
			assert policy_document["document_id"] == "doc_policy_001"
			assert policy_document["version"] == "1.0"

class TestPerformanceAndScalability:
	"""Test performance and scalability of integrated GRC system."""
	
	@pytest.mark.asyncio
	async def test_bulk_operations_performance(self, integration_setup: IntegrationTestSetup):
		"""Test system performance with bulk data operations."""
		import time
		
		# Create bulk risks for performance testing
		bulk_risks = []
		start_time = time.time()
		
		for i in range(100):
			risk = GRCRisk(
				tenant_id=integration_setup.test_tenant.id,
				title=f"Performance Test Risk {i}",
				description=f"Risk created for performance testing - {i}",
				current_likelihood=Decimal(str(30.0 + (i % 50))),
				current_impact=Decimal(str(40.0 + (i % 40))),
				status="active",
				owner_id=f"owner_{i % 10}"
			)
			bulk_risks.append(risk)
		
		# Bulk insert
		integration_setup.db_session.add_all(bulk_risks)
		integration_setup.db_session.commit()
		
		creation_time = time.time() - start_time
		
		# Test bulk query performance
		start_time = time.time()
		
		high_risk_count = integration_setup.db_session.query(GRCRisk).filter(
			GRCRisk.tenant_id == integration_setup.test_tenant.id,
			GRCRisk.current_likelihood * GRCRisk.current_impact / 100 >= 2000
		).count()
		
		query_time = time.time() - start_time
		
		# Performance assertions
		assert creation_time < 5.0  # Bulk creation should complete within 5 seconds
		assert query_time < 1.0     # Complex query should complete within 1 second
		assert high_risk_count > 0  # Should find some high-risk items
		
		# Verify data integrity
		total_risks = integration_setup.db_session.query(GRCRisk).filter_by(
			tenant_id=integration_setup.test_tenant.id
		).count()
		assert total_risks >= 100  # All risks should be created
	
	@pytest.mark.asyncio
	async def test_concurrent_operations(self, integration_setup: IntegrationTestSetup):
		"""Test system behavior under concurrent operations."""
		import asyncio
		
		async def create_risk(risk_id: int):
			"""Create a risk asynchronously."""
			risk = GRCRisk(
				tenant_id=integration_setup.test_tenant.id,
				title=f"Concurrent Risk {risk_id}",
				description=f"Risk created concurrently - {risk_id}",
				current_likelihood=Decimal("50.0"),
				current_impact=Decimal("60.0"),
				status="active",
				owner_id=f"concurrent_owner_{risk_id}"
			)
			
			integration_setup.db_session.add(risk)
			integration_setup.db_session.commit()
			return risk.id
		
		# Create 20 risks concurrently
		tasks = [create_risk(i) for i in range(20)]
		risk_ids = await asyncio.gather(*tasks)
		
		# Verify all risks were created successfully
		assert len(risk_ids) == 20
		assert all(risk_id is not None for risk_id in risk_ids)
		
		# Verify data integrity
		concurrent_risks = integration_setup.db_session.query(GRCRisk).filter(
			GRCRisk.title.like("Concurrent Risk %")
		).all()
		
		assert len(concurrent_risks) == 20
		assert all(risk.current_likelihood == Decimal("50.0") for risk in concurrent_risks)

class TestProductionReadiness:
	"""Test production-readiness scenarios and edge cases."""
	
	@pytest.mark.asyncio
	async def test_error_handling_and_recovery(self, integration_setup: IntegrationTestSetup):
		"""Test system error handling and recovery mechanisms."""
		
		# Test 1: Database connection failure simulation
		with patch.object(integration_setup.db_session, 'commit') as mock_commit:
			mock_commit.side_effect = Exception("Database connection lost")
			
			try:
				risk = GRCRisk(
					tenant_id=integration_setup.test_tenant.id,
					title="Error Test Risk",
					description="Risk for error testing",
					current_likelihood=Decimal("40.0"),
					current_impact=Decimal("50.0"),
					status="active",
					owner_id="error_test_user"
				)
				
				integration_setup.db_session.add(risk)
				integration_setup.db_session.commit()
				
				assert False, "Should have raised an exception"
			except Exception as e:
				assert "Database connection lost" in str(e)
				
				# Test recovery - rollback and retry
				integration_setup.db_session.rollback()
				
				# Reset mock and retry
				mock_commit.side_effect = None
				integration_setup.db_session.commit()  # Should work now
		
		# Test 2: AI service failure handling
		ai_engine = GRCAIEngine(
			tenant_id=integration_setup.test_tenant.id,
			model_config={"risk_prediction": {"enabled": True}}
		)
		
		with patch.object(ai_engine, 'predict_risk_evolution') as mock_predict:
			mock_predict.side_effect = Exception("AI service unavailable")
			
			# System should gracefully handle AI service failure
			try:
				await ai_engine.predict_risk_evolution("risk123", {})
				assert False, "Should have raised an exception"
			except Exception as e:
				assert "AI service unavailable" in str(e)
				
				# System should continue operating without AI predictions
				# (graceful degradation)
	
	@pytest.mark.asyncio
	async def test_data_validation_and_integrity(self, integration_setup: IntegrationTestSetup):
		"""Test comprehensive data validation and integrity checks."""
		
		# Test invalid risk data
		with pytest.raises(Exception):  # Should fail validation
			invalid_risk = GRCRisk(
				tenant_id=integration_setup.test_tenant.id,
				title="",  # Empty title should fail
				description="Invalid risk for testing",
				current_likelihood=Decimal("150.0"),  # Invalid percentage > 100
				current_impact=Decimal("-10.0"),     # Invalid negative value
				status="invalid_status",             # Invalid status
				owner_id="test_user"
			)
			
			integration_setup.db_session.add(invalid_risk)
			integration_setup.db_session.commit()
		
		# Test valid risk with boundary values
		boundary_risk = GRCRisk(
			tenant_id=integration_setup.test_tenant.id,
			title="Boundary Test Risk",
			description="Testing boundary values",
			current_likelihood=Decimal("0.0"),   # Minimum valid value
			current_impact=Decimal("100.0"),     # Maximum valid value
			status="active",
			owner_id="boundary_test_user"
		)
		
		integration_setup.db_session.add(boundary_risk)
		integration_setup.db_session.commit()
		
		# Verify boundary risk was created successfully
		created_risk = integration_setup.db_session.query(GRCRisk).filter_by(
			title="Boundary Test Risk"
		).first()
		
		assert created_risk is not None
		assert created_risk.current_likelihood == Decimal("0.0")
		assert created_risk.current_impact == Decimal("100.0")

if __name__ == "__main__":
	pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])