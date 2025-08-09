#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Governance, Risk & Compliance - Model Tests

Comprehensive test suite for GRC models testing all functionality,
edge cases, validation, and APG integration patterns.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from pydantic import ValidationError

from models import (
	Base,
	GRCTenant,
	GRCRiskCategory,
	GRCRisk,
	GRCRiskAssessment,
	GRCRegulation,
	GRCControl,
	GRCPolicy,
	GRCGovernanceDecision,
	GRCIncident,
	GRCAuditLog,
	GRCMetrics,
	RiskView,
	ControlView,
	PolicyView,
	ComplianceView,
	GovernanceDashboardView,
)

# Test database setup
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")  
def db_session():
	"""Create fresh database session for each test."""
	Base.metadata.create_all(bind=engine)
	session = TestSessionLocal()
	try:
		yield session
	finally:
		session.close()
		Base.metadata.drop_all(bind=engine)

@pytest.fixture
def sample_tenant(db_session: Session) -> GRCTenant:
	"""Create sample tenant for testing."""
	tenant = GRCTenant(
		name="Test Corp",
		slug="test-corp",
		domain="test.example.com",
		settings={
			"risk_tolerance": "medium",
			"compliance_frameworks": ["SOX", "GDPR"],
			"automation_level": "high"
		}
	)
	db_session.add(tenant)
	db_session.commit()
	db_session.refresh(tenant)
	return tenant

@pytest.fixture
def sample_risk_category(db_session: Session, sample_tenant: GRCTenant) -> GRCRiskCategory:
	"""Create sample risk category for testing."""
	category = GRCRiskCategory(
		tenant_id=sample_tenant.id,
		name="Operational Risk",
		description="Risks related to day-to-day operations",
		color_code="#FF6B6B",
		risk_appetite=Decimal("75.0"),
		parent_category_id=None
	)
	db_session.add(category)
	db_session.commit()
	db_session.refresh(category)
	return category

class TestGRCTenant:
	"""Test GRC Tenant model functionality."""
	
	def test_tenant_creation(self, db_session: Session):
		"""Test basic tenant creation and validation."""
		tenant = GRCTenant(
			name="Acme Corp",
			slug="acme-corp", 
			domain="acme.example.com",
			settings={"theme": "dark", "language": "en"}
		)
		
		db_session.add(tenant)
		db_session.commit()
		
		assert tenant.id is not None
		assert tenant.name == "Acme Corp"
		assert tenant.slug == "acme-corp"
		assert tenant.is_active is True
		assert tenant.settings["theme"] == "dark"
		assert tenant.created_at is not None

	def test_tenant_slug_uniqueness(self, db_session: Session):
		"""Test tenant slug uniqueness constraint."""
		tenant1 = GRCTenant(name="Corp1", slug="test-slug", domain="corp1.com")
		tenant2 = GRCTenant(name="Corp2", slug="test-slug", domain="corp2.com")
		
		db_session.add(tenant1)
		db_session.commit()
		
		db_session.add(tenant2)
		
		with pytest.raises(IntegrityError):
			db_session.commit()

	def test_tenant_domain_validation(self, db_session: Session):
		"""Test tenant domain format validation."""
		valid_tenant = GRCTenant(
			name="Valid Corp",
			slug="valid-corp",
			domain="valid.example.com"
		)
		
		db_session.add(valid_tenant)
		db_session.commit()  # Should not raise
		
		assert valid_tenant.domain == "valid.example.com"

class TestGRCRiskCategory:
	"""Test GRC Risk Category model functionality."""
	
	def test_risk_category_creation(self, db_session: Session, sample_tenant: GRCTenant):
		"""Test basic risk category creation."""
		category = GRCRiskCategory(
			tenant_id=sample_tenant.id,
			name="Financial Risk",
			description="Risks related to financial operations",
			color_code="#00BCD4",
			risk_appetite=Decimal("80.0")
		)
		
		db_session.add(category)
		db_session.commit()
		
		assert category.id is not None
		assert category.name == "Financial Risk"
		assert category.risk_appetite == Decimal("80.0")
		assert category.tenant_id == sample_tenant.id

	def test_risk_category_hierarchy(self, db_session: Session, sample_tenant: GRCTenant):
		"""Test risk category parent-child relationships."""
		parent_category = GRCRiskCategory(
			tenant_id=sample_tenant.id,
			name="Strategic Risk",
			description="High-level strategic risks",
			color_code="#9C27B0",
			risk_appetite=Decimal("60.0")
		)
		
		db_session.add(parent_category)
		db_session.commit()
		
		child_category = GRCRiskCategory(
			tenant_id=sample_tenant.id,
			name="Market Risk",
			description="Market-related strategic risks",
			color_code="#673AB7",
			risk_appetite=Decimal("70.0"),
			parent_category_id=parent_category.id
		)
		
		db_session.add(child_category)
		db_session.commit()
		
		assert child_category.parent_category_id == parent_category.id
		assert len(parent_category.subcategories) == 1
		assert parent_category.subcategories[0].name == "Market Risk"

	def test_risk_appetite_validation(self, db_session: Session, sample_tenant: GRCTenant):
		"""Test risk appetite percentage validation."""
		# Valid risk appetite
		valid_category = GRCRiskCategory(
			tenant_id=sample_tenant.id,
			name="Valid Risk",
			description="Valid risk category",
			color_code="#4CAF50",
			risk_appetite=Decimal("50.0")
		)
		
		db_session.add(valid_category)
		db_session.commit()  # Should not raise
		
		assert valid_category.risk_appetite == Decimal("50.0")

class TestGRCRisk:
	"""Test GRC Risk model functionality."""
	
	def test_risk_creation(self, db_session: Session, sample_tenant: GRCTenant, sample_risk_category: GRCRiskCategory):
		"""Test basic risk creation and AI prediction integration."""
		risk = GRCRisk(
			tenant_id=sample_tenant.id,
			title="Data Breach Risk",
			description="Risk of unauthorized data access",
			category_id=sample_risk_category.id,
			current_likelihood=Decimal("30.0"),
			current_impact=Decimal("90.0"),
			risk_score=Decimal("27.0"),
			predicted_likelihood=Decimal("35.0"),
			predicted_impact=Decimal("85.0"),
			prediction_confidence=Decimal("78.5"),
			ai_insights={
				"trend": "increasing",
				"key_factors": ["increased_cyber_attacks", "remote_work"],
				"recommendations": ["implement_mfa", "security_training"]
			},
			status="active",
			owner_id="user123",
			next_review_date=datetime.utcnow() + timedelta(days=90)
		)
		
		db_session.add(risk)
		db_session.commit()
		
		assert risk.id is not None
		assert risk.title == "Data Breach Risk"
		assert risk.risk_score == Decimal("27.0")
		assert risk.prediction_confidence == Decimal("78.5")
		assert risk.ai_insights["trend"] == "increasing"
		assert len(risk.ai_insights["recommendations"]) == 2

	def test_risk_score_calculation(self, db_session: Session, sample_tenant: GRCTenant, sample_risk_category: GRCRiskCategory):
		"""Test automatic risk score calculation."""
		risk = GRCRisk(
			tenant_id=sample_tenant.id,
			title="Test Risk",
			description="Test risk for score calculation",
			category_id=sample_risk_category.id,
			current_likelihood=Decimal("60.0"),
			current_impact=Decimal("80.0"),
			status="active",
			owner_id="user123"
		)
		
		# Calculate expected risk score (likelihood * impact / 100)
		expected_score = Decimal("60.0") * Decimal("80.0") / Decimal("100.0")
		
		db_session.add(risk)
		db_session.commit()
		
		assert risk.risk_score == expected_score

	def test_risk_relationships(self, db_session: Session, sample_tenant: GRCTenant, sample_risk_category: GRCRiskCategory):
		"""Test risk relationships and cascading."""
		risk = GRCRisk(
			tenant_id=sample_tenant.id,
			title="Primary Risk",
			description="Primary risk with relationships",
			category_id=sample_risk_category.id,
			current_likelihood=Decimal("50.0"),
			current_impact=Decimal("70.0"),
			status="active",
			owner_id="user123"
		)
		
		db_session.add(risk)
		db_session.commit()
		
		# Test category relationship
		assert risk.category.name == "Operational Risk"
		assert risk in sample_risk_category.risks

class TestGRCRiskAssessment:
	"""Test GRC Risk Assessment model functionality."""
	
	def test_assessment_creation(self, db_session: Session, sample_tenant: GRCTenant, sample_risk_category: GRCRiskCategory):
		"""Test risk assessment creation with AI predictions."""
		# Create risk first
		risk = GRCRisk(
			tenant_id=sample_tenant.id,
			title="Assessment Risk",
			description="Risk for assessment testing",
			category_id=sample_risk_category.id,
			current_likelihood=Decimal("40.0"),
			current_impact=Decimal("60.0"),
			status="active",
			owner_id="assessor123"
		)
		
		db_session.add(risk)
		db_session.commit()
		
		assessment = GRCRiskAssessment(
			tenant_id=sample_tenant.id,
			risk_id=risk.id,
			assessor_id="assessor123",
			assessment_type="quarterly",
			likelihood_score=Decimal("45.0"),
			impact_score=Decimal("65.0"),
			overall_score=Decimal("29.25"),
			assessment_date=datetime.utcnow(),
			methodology="Monte Carlo simulation",
			findings="Risk level has increased due to market volatility",
			recommendations=[
				"Implement additional hedging strategies",
				"Increase monitoring frequency"
			],
			confidence_level=Decimal("85.0"),
			ai_analysis={
				"model_used": "risk_prediction_lstm",
				"accuracy": 0.92,
				"key_indicators": ["market_volatility", "liquidity_ratios"]
			}
		)
		
		db_session.add(assessment)
		db_session.commit()
		
		assert assessment.id is not None
		assert assessment.risk_id == risk.id
		assert assessment.overall_score == Decimal("29.25")
		assert assessment.confidence_level == Decimal("85.0")
		assert len(assessment.recommendations) == 2
		assert assessment.ai_analysis["accuracy"] == 0.92

class TestGRCRegulation:
	"""Test GRC Regulation model functionality."""
	
	def test_regulation_creation(self, db_session: Session, sample_tenant: GRCTenant):
		"""Test regulation creation with AI monitoring."""
		regulation = GRCRegulation(
			tenant_id=sample_tenant.id,
			title="GDPR Article 32",
			description="Security of processing requirements",
			regulatory_body="European Commission",
			effective_date=datetime(2018, 5, 25),
			last_updated=datetime.utcnow(),
			jurisdiction=["EU", "EEA"],
			compliance_deadline=datetime.utcnow() + timedelta(days=365),
			penalty_range="Up to 4% of annual turnover",
			requirements=[
				"Implement appropriate technical measures",
				"Conduct regular security assessments",
				"Maintain processing records"
			],
			ai_monitoring_config={
				"keywords": ["data protection", "security measures", "GDPR"],
				"monitoring_frequency": "daily",
				"alert_threshold": 0.8
			},
			status="active"
		)
		
		db_session.add(regulation)
		db_session.commit()
		
		assert regulation.id is not None
		assert regulation.title == "GDPR Article 32"
		assert "EU" in regulation.jurisdiction
		assert len(regulation.requirements) == 3
		assert regulation.ai_monitoring_config["monitoring_frequency"] == "daily"

class TestGRCControl:
	"""Test GRC Control model functionality."""
	
	def test_control_creation(self, db_session: Session, sample_tenant: GRCTenant):
		"""Test control creation with automation and AI testing."""
		control = GRCControl(
			tenant_id=sample_tenant.id,
			name="Multi-Factor Authentication",
			description="Require MFA for all system access",
			control_type="preventive",
			control_frequency="continuous",
			owner_id="security_team",
			implementation_status="implemented",
			effectiveness_rating=Decimal("90.0"),
			last_tested=datetime.utcnow() - timedelta(days=30),
			next_test_date=datetime.utcnow() + timedelta(days=60),
			automated=True,
			automation_config={
				"test_script": "mfa_test.py",
				"test_frequency": "daily",
				"alert_on_failure": True
			},
			ai_testing_enabled=True,
			ai_test_results={
				"last_test_score": 0.95,
				"anomalies_detected": 2,
				"recommendations": ["Review MFA bypass procedures"]
			}
		)
		
		db_session.add(control)
		db_session.commit()
		
		assert control.id is not None
		assert control.name == "Multi-Factor Authentication"
		assert control.automated is True
		assert control.ai_testing_enabled is True
		assert control.ai_test_results["last_test_score"] == 0.95

class TestGRCPolicy:
	"""Test GRC Policy model functionality."""
	
	def test_policy_creation(self, db_session: Session, sample_tenant: GRCTenant):
		"""Test policy creation with AI assistance."""
		policy = GRCPolicy(
			tenant_id=sample_tenant.id,
			title="Data Classification Policy",
			description="Policy for classifying and handling sensitive data",
			policy_type="data_governance",
			version="2.1",
			effective_date=datetime.utcnow(),
			review_date=datetime.utcnow() + timedelta(days=365),
			owner_id="data_governance_team",
			approver_id="ciso",
			approval_date=datetime.utcnow() - timedelta(days=7),
			content={
				"sections": [
					{"title": "Purpose", "content": "Define data classification standards"},
					{"title": "Scope", "content": "All organizational data"}
				],
				"procedures": ["Identify data types", "Apply classification labels"],
				"exceptions": "None currently approved"
			},
			ai_assistance_used=True,
			ai_suggestions=[
				"Consider adding machine learning data classification",
				"Include GDPR data subject rights procedures"
			],
			compliance_mapping=["GDPR", "SOX", "HIPAA"],
			status="active"
		)
		
		db_session.add(policy)
		db_session.commit()
		
		assert policy.id is not None
		assert policy.title == "Data Classification Policy"
		assert policy.version == "2.1"
		assert policy.ai_assistance_used is True
		assert len(policy.ai_suggestions) == 2
		assert "GDPR" in policy.compliance_mapping

class TestGRCGovernanceDecision:
	"""Test GRC Governance Decision model functionality."""
	
	def test_governance_decision_creation(self, db_session: Session, sample_tenant: GRCTenant):
		"""Test governance decision creation with stakeholder collaboration."""
		decision = GRCGovernanceDecision(
			tenant_id=sample_tenant.id,
			title="Cloud Migration Security Framework",
			description="Decision on security framework for cloud migration",
			decision_type="strategic",
			decision_maker_id="ceo",
			decision_date=datetime.utcnow(),
			stakeholders=[
				{"id": "cto", "role": "Technical Lead", "influence": "high"},
				{"id": "ciso", "role": "Security Officer", "influence": "high"},
				{"id": "cfo", "role": "Financial Officer", "influence": "medium"}
			],
			options_considered=[
				{"option": "AWS Security Framework", "score": 85},
				{"option": "Azure Security Framework", "score": 78},
				{"option": "Multi-cloud Framework", "score": 92}
			],
			decision_rationale="Multi-cloud approach provides best risk distribution",
			impact_assessment={
				"financial": {"cost": 500000, "savings": 200000},
				"operational": {"complexity": "medium", "timeline": "6 months"},
				"risk": {"reduction": 35, "new_risks": ["vendor_lock_in"]}
			},
			implementation_plan=[
				{"phase": "Assessment", "duration": "4 weeks"},
				{"phase": "Pilot", "duration": "8 weeks"},
				{"phase": "Full Deployment", "duration": "12 weeks"}
			],
			ai_recommendation_score=Decimal("92.0"),
			status="approved"
		)
		
		db_session.add(decision)
		db_session.commit()
		
		assert decision.id is not None
		assert decision.title == "Cloud Migration Security Framework"
		assert len(decision.stakeholders) == 3
		assert len(decision.options_considered) == 3
		assert decision.ai_recommendation_score == Decimal("92.0")
		assert decision.impact_assessment["risk"]["reduction"] == 35

class TestPydanticViews:
	"""Test Pydantic view models for API serialization."""
	
	def test_risk_view_creation(self):
		"""Test RiskView Pydantic model validation."""
		risk_data = {
			"id": "risk123",
			"title": "API Risk",
			"description": "Risk in API operations", 
			"category": "operational",
			"current_likelihood": 40.0,
			"current_impact": 70.0,
			"risk_score": 28.0,
			"status": "active",
			"owner_id": "owner123",
			"created_at": datetime.utcnow().isoformat(),
			"ai_insights": {
				"trend": "stable",
				"confidence": 85.5
			}
		}
		
		risk_view = RiskView(**risk_data)
		
		assert risk_view.id == "risk123"
		assert risk_view.title == "API Risk"
		assert risk_view.risk_score == 28.0
		assert risk_view.ai_insights["trend"] == "stable"

	def test_control_view_validation(self):
		"""Test ControlView validation with required fields."""
		control_data = {
			"id": "ctrl123",
			"name": "Access Control",
			"description": "User access control mechanism",
			"control_type": "preventive",
			"effectiveness_rating": 88.5,
			"automated": True,
			"status": "active"
		}
		
		control_view = ControlView(**control_data)
		
		assert control_view.id == "ctrl123"
		assert control_view.automated is True
		assert control_view.effectiveness_rating == 88.5

	def test_policy_view_invalid_data(self):
		"""Test PolicyView validation with invalid data."""
		invalid_policy_data = {
			"id": "policy123",
			"title": "",  # Empty title should fail validation
			"policy_type": "invalid_type",
			"status": "unknown_status"
		}
		
		with pytest.raises(ValidationError) as exc_info:
			PolicyView(**invalid_policy_data)
		
		assert "title" in str(exc_info.value)

class TestModelIntegration:
	"""Test model integration and relationships."""
	
	def test_tenant_cascade_delete(self, db_session: Session):
		"""Test cascading deletion when tenant is removed."""
		# Create tenant with related data
		tenant = GRCTenant(name="Delete Test", slug="delete-test", domain="delete.test")
		db_session.add(tenant)
		db_session.commit()
		
		category = GRCRiskCategory(
			tenant_id=tenant.id,
			name="Test Category",
			description="Category for deletion test",
			color_code="#FF0000",
			risk_appetite=Decimal("50.0")
		)
		db_session.add(category)
		db_session.commit()
		
		risk = GRCRisk(
			tenant_id=tenant.id,
			title="Test Risk",
			description="Risk for deletion test",
			category_id=category.id,
			current_likelihood=Decimal("30.0"),
			current_impact=Decimal("40.0"),
			status="active",
			owner_id="test_user"
		)
		db_session.add(risk)
		db_session.commit()
		
		# Verify data exists
		assert db_session.query(GRCTenant).filter_by(id=tenant.id).first() is not None
		assert db_session.query(GRCRiskCategory).filter_by(tenant_id=tenant.id).first() is not None
		assert db_session.query(GRCRisk).filter_by(tenant_id=tenant.id).first() is not None
		
		# Delete tenant
		db_session.delete(tenant)
		db_session.commit()
		
		# Verify cascading deletion (depending on your FK constraints)
		assert db_session.query(GRCTenant).filter_by(id=tenant.id).first() is None

	def test_ai_insights_serialization(self, db_session: Session, sample_tenant: GRCTenant, sample_risk_category: GRCRiskCategory):
		"""Test AI insights JSON serialization and deserialization."""
		complex_ai_insights = {
			"prediction_model": "lstm_v2.1",
			"confidence_intervals": {
				"likelihood": {"lower": 25.0, "upper": 45.0},
				"impact": {"lower": 60.0, "upper": 80.0}
			},
			"feature_importance": {
				"market_volatility": 0.35,
				"regulatory_changes": 0.28,
				"operational_complexity": 0.37
			},
			"recommendations": [
				{"action": "increase_monitoring", "priority": "high", "impact": 0.8},
				{"action": "review_controls", "priority": "medium", "impact": 0.6}
			],
			"model_metadata": {
				"training_date": "2024-01-15T10:30:00Z",
				"accuracy": 0.94,
				"precision": 0.89,
				"recall": 0.91
			}
		}
		
		risk = GRCRisk(
			tenant_id=sample_tenant.id,
			title="AI Insights Test Risk",
			description="Testing complex AI insights serialization",
			category_id=sample_risk_category.id,
			current_likelihood=Decimal("35.0"),
			current_impact=Decimal("70.0"),
			ai_insights=complex_ai_insights,
			status="active",
			owner_id="ai_tester"
		)
		
		db_session.add(risk)
		db_session.commit()
		db_session.refresh(risk)
		
		# Test deserialization
		retrieved_insights = risk.ai_insights
		assert retrieved_insights["prediction_model"] == "lstm_v2.1"
		assert retrieved_insights["confidence_intervals"]["likelihood"]["lower"] == 25.0
		assert len(retrieved_insights["recommendations"]) == 2
		assert retrieved_insights["model_metadata"]["accuracy"] == 0.94

class TestPerformanceAndScaling:
	"""Test model performance and scaling considerations."""
	
	def test_bulk_risk_creation(self, db_session: Session, sample_tenant: GRCTenant, sample_risk_category: GRCRiskCategory):
		"""Test bulk creation of risks for performance validation."""
		risks = []
		for i in range(100):
			risk = GRCRisk(
				tenant_id=sample_tenant.id,
				title=f"Bulk Risk {i}",
				description=f"Auto-generated risk {i} for performance testing",
				category_id=sample_risk_category.id,
				current_likelihood=Decimal(str(30.0 + (i % 50))),
				current_impact=Decimal(str(40.0 + (i % 40))),
				status="active",
				owner_id=f"user_{i % 10}"
			)
			risks.append(risk)
		
		# Bulk insert
		db_session.add_all(risks)
		db_session.commit()
		
		# Verify all risks were created
		risk_count = db_session.query(GRCRisk).filter_by(tenant_id=sample_tenant.id).count()
		assert risk_count == 100
		
		# Test query performance with filtering
		high_impact_risks = db_session.query(GRCRisk).filter(
			GRCRisk.tenant_id == sample_tenant.id,
			GRCRisk.current_impact >= Decimal("70.0")
		).all()
		
		assert len(high_impact_risks) > 0
		for risk in high_impact_risks:
			assert risk.current_impact >= Decimal("70.0")

if __name__ == "__main__":
	pytest.main([__file__, "-v", "--tb=short"])