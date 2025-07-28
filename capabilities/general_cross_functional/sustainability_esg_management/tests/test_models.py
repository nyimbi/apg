#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Sustainability & ESG Management - Models Tests

Comprehensive test suite for ESG models with validation,
relationships, and APG integration testing.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
	ESGTenant, ESGFramework, ESGMetric, ESGMeasurement, ESGTarget, ESGMilestone,
	ESGStakeholder, ESGCommunication, ESGSupplier, ESGSupplierAssessment,
	ESGInitiative, ESGReport, ESGRisk,
	ESGFrameworkType, ESGMetricType, ESGMetricUnit, ESGTargetStatus,
	ESGReportStatus, ESGInitiativeStatus, ESGRiskLevel
)
from . import ESGTestConfig, TEST_TENANT_ID, TEST_USER_ID


class TestESGTenant:
	"""Test ESG tenant model and functionality"""
	
	def test_tenant_creation(self, db_session: Session):
		"""Test basic tenant creation"""
		tenant = ESGTenant(
			id="test_tenant_creation",
			name="Test Corporation",
			slug="test-corp",
			description="Test corporation for ESG testing",
			industry="technology",
			headquarters_country="USA",
			employee_count=1000,
			annual_revenue=Decimal("500000000.00"),
			esg_frameworks=["GRI", "SASB"],
			ai_enabled=True,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(tenant)
		db_session.commit()
		
		assert tenant.id == "test_tenant_creation"
		assert tenant.name == "Test Corporation"
		assert tenant.slug == "test-corp"
		assert tenant.ai_enabled is True
		assert len(tenant.esg_frameworks) == 2
		assert "GRI" in tenant.esg_frameworks
		assert tenant.created_at is not None
		assert tenant.updated_at is not None
	
	def test_tenant_slug_validation(self, db_session: Session):
		"""Test tenant slug validation"""
		# Valid slug
		tenant1 = ESGTenant(
			id="test_slug_valid",
			name="Valid Slug Test",
			slug="valid-slug-123",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(tenant1)
		db_session.commit()
		assert tenant1.slug == "valid-slug-123"
	
	def test_tenant_ai_configuration(self, db_session: Session):
		"""Test tenant AI configuration handling"""
		ai_config = {
			"sustainability_prediction": {"enabled": True, "model": "lstm_v2"},
			"stakeholder_intelligence": {"enabled": True, "model": "bert_sentiment"},
			"supply_chain_analysis": {"enabled": False, "model": "graph_neural"}
		}
		
		tenant = ESGTenant(
			id="test_ai_config",
			name="AI Config Test",
			slug="ai-config-test",
			ai_enabled=True,
			ai_configuration=ai_config,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(tenant)
		db_session.commit()
		
		assert tenant.ai_configuration["sustainability_prediction"]["enabled"] is True
		assert tenant.ai_configuration["supply_chain_analysis"]["enabled"] is False
	
	def test_tenant_settings_json(self, db_session: Session):
		"""Test tenant settings JSON field"""
		settings = {
			"timezone": "UTC",
			"language": "en_US",
			"notifications": {"email": True, "sms": False, "in_app": True},
			"reporting": {"frequency": "monthly", "format": "pdf"}
		}
		
		tenant = ESGTenant(
			id="test_settings",
			name="Settings Test",
			slug="settings-test",
			settings=settings,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(tenant)
		db_session.commit()
		
		assert tenant.settings["timezone"] == "UTC"
		assert tenant.settings["notifications"]["email"] is True
		assert tenant.settings["reporting"]["frequency"] == "monthly"
	
	def test_tenant_uniqueness_constraints(self, db_session: Session):
		"""Test tenant uniqueness constraints"""
		tenant1 = ESGTenant(
			id="unique_test_1", 
			name="Unique Test 1",
			slug="unique-test",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(tenant1)
		db_session.commit()
		
		# Try to create tenant with same slug
		tenant2 = ESGTenant(
			id="unique_test_2",
			name="Unique Test 2", 
			slug="unique-test",  # Same slug should fail
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(tenant2)
		
		with pytest.raises(IntegrityError):
			db_session.commit()


class TestESGFramework:
	"""Test ESG framework model"""
	
	def test_framework_creation(self, db_session: Session, sample_tenant: ESGTenant):
		"""Test ESG framework creation"""
		framework = ESGFramework(
			id="test_framework_tcfd",
			tenant_id=sample_tenant.id,
			name="Task Force on Climate-related Financial Disclosures",
			code="TCFD",
			framework_type=ESGFrameworkType.TCFD,
			version="2023",
			description="Climate-related financial risk disclosure framework",
			official_url="https://www.fsb-tcfd.org/",
			effective_date=datetime(2023, 1, 1),
			categories=[
				{"code": "TCFD_GOV", "name": "Governance", "description": "Climate governance"},
				{"code": "TCFD_STRAT", "name": "Strategy", "description": "Climate strategy"}
			],
			is_mandatory=False,
			is_active=True,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(framework)
		db_session.commit()
		
		assert framework.name == "Task Force on Climate-related Financial Disclosures"
		assert framework.code == "TCFD"
		assert framework.framework_type == ESGFrameworkType.TCFD
		assert len(framework.categories) == 2
		assert framework.is_active is True
	
	def test_framework_categories_json(self, db_session: Session, sample_tenant: ESGTenant):
		"""Test framework categories JSON handling"""
		categories = [
			{"code": "ENV", "name": "Environmental", "description": "Environmental indicators"},
			{"code": "SOC", "name": "Social", "description": "Social indicators"},
			{"code": "GOV", "name": "Governance", "description": "Governance indicators"}
		]
		
		framework = ESGFramework(
			id="test_categories",
			tenant_id=sample_tenant.id,
			name="Test Categories Framework",
			code="TCF",
			framework_type=ESGFrameworkType.CUSTOM,
			categories=categories,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(framework)
		db_session.commit()
		
		assert len(framework.categories) == 3
		assert framework.categories[0]["code"] == "ENV"
		assert framework.categories[1]["name"] == "Social"


class TestESGMetric:
	"""Test ESG metric model and validation"""
	
	def test_metric_creation(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_framework: ESGFramework):
		"""Test basic ESG metric creation"""
		metric = ESGMetric(
			id="test_metric_water_usage",
			tenant_id=sample_tenant.id,
			framework_id=sample_esg_framework.id,
			name="Water Usage Intensity",
			code="WATER_USAGE_INT",
			metric_type=ESGMetricType.ENVIRONMENTAL,
			category="water_stewardship",
			subcategory="water_consumption",
			description="Water consumption per unit of production",
			calculation_method="Total water consumed / Production units",
			unit=ESGMetricUnit.LITERS_PER_UNIT,
			current_value=Decimal("125.50"),
			target_value=Decimal("100.00"),
			baseline_value=Decimal("150.00"),
			measurement_period="monthly",
			is_kpi=True,
			is_automated=True,
			data_quality_score=Decimal("92.5"),
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(metric)
		db_session.commit()
		
		assert metric.name == "Water Usage Intensity"
		assert metric.code == "WATER_USAGE_INT"
		assert metric.metric_type == ESGMetricType.ENVIRONMENTAL
		assert metric.unit == ESGMetricUnit.LITERS_PER_UNIT
		assert metric.current_value == Decimal("125.50")
		assert metric.is_kpi is True
		assert metric.data_quality_score == Decimal("92.5")
	
	def test_metric_ai_predictions(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_framework: ESGFramework):
		"""Test metric AI predictions JSON field"""
		ai_predictions = {
			"predicted_6_month": 118.75,
			"predicted_12_month": 105.25,
			"confidence": 0.87,
			"trend": "improving",
			"factors": ["efficiency_improvements", "technology_upgrade"]
		}
		
		metric = ESGMetric(
			id="test_ai_predictions",
			tenant_id=sample_tenant.id,
			framework_id=sample_esg_framework.id,
			name="AI Predictions Test Metric",
			code="AI_PRED_TEST",
			metric_type=ESGMetricType.ENVIRONMENTAL,
			category="test",
			unit=ESGMetricUnit.PERCENTAGE,
			ai_predictions=ai_predictions,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(metric)
		db_session.commit()
		
		assert metric.ai_predictions["predicted_6_month"] == 118.75
		assert metric.ai_predictions["confidence"] == 0.87
		assert "efficiency_improvements" in metric.ai_predictions["factors"]
	
	def test_metric_trend_analysis(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_framework: ESGFramework):
		"""Test metric trend analysis JSON field"""
		trend_analysis = {
			"direction": "improving",
			"strength": 0.72,
			"seasonality": "moderate",
			"volatility": "low",
			"change_points": ["2023-06-01", "2023-09-15"],
			"correlation_factors": ["energy_efficiency", "process_optimization"]
		}
		
		metric = ESGMetric(
			id="test_trend_analysis",
			tenant_id=sample_tenant.id,
			framework_id=sample_esg_framework.id,
			name="Trend Analysis Test",
			code="TREND_TEST",
			metric_type=ESGMetricType.SOCIAL,
			category="test",
			unit=ESGMetricUnit.PERCENTAGE,
			trend_analysis=trend_analysis,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(metric)
		db_session.commit()
		
		assert metric.trend_analysis["direction"] == "improving"
		assert metric.trend_analysis["strength"] == 0.72
		assert len(metric.trend_analysis["change_points"]) == 2
	
	def test_metric_validation_rules(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_framework: ESGFramework):
		"""Test metric validation rules JSON field"""
		validation_rules = [
			{"type": "range_check", "min_value": 0, "max_value": 1000},
			{"type": "trend_check", "max_deviation_percent": 20},
			{"type": "quality_check", "min_data_quality": 80}
		]
		
		metric = ESGMetric(
			id="test_validation_rules",
			tenant_id=sample_tenant.id,
			framework_id=sample_esg_framework.id,
			name="Validation Rules Test",
			code="VALID_TEST",
			metric_type=ESGMetricType.GOVERNANCE,
			category="test",
			unit=ESGMetricUnit.COUNT,
			validation_rules=validation_rules,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(metric)
		db_session.commit()
		
		assert len(metric.validation_rules) == 3
		assert metric.validation_rules[0]["type"] == "range_check"
		assert metric.validation_rules[1]["max_deviation_percent"] == 20


class TestESGTarget:
	"""Test ESG target model and relationships"""
	
	def test_target_creation(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_metrics: List[ESGMetric]):
		"""Test ESG target creation"""
		target = ESGTarget(
			id="test_target_renewable_energy",
			tenant_id=sample_tenant.id,
			metric_id=sample_esg_metrics[0].id,
			name="Renewable Energy Target 2030",
			description="Achieve 100% renewable energy by 2030",
			target_value=Decimal("100.0"),
			baseline_value=Decimal("25.0"),
			current_progress=Decimal("45.5"),
			start_date=datetime(2024, 1, 1),
			target_date=datetime(2030, 12, 31),
			review_frequency="quarterly",
			status=ESGTargetStatus.ON_TRACK,
			priority="high",
			achievement_probability=Decimal("82.5"),
			predicted_completion_date=datetime(2029, 8, 15),
			owner_id="sustainability_director",
			stakeholders=["ceo", "cfo", "sustainability_team"],
			is_public=True,
			milestone_tracking=True,
			automated_reporting=True,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(target)
		db_session.commit()
		
		assert target.name == "Renewable Energy Target 2030"
		assert target.target_value == Decimal("100.0")
		assert target.current_progress == Decimal("45.5")
		assert target.status == ESGTargetStatus.ON_TRACK
		assert target.achievement_probability == Decimal("82.5")
		assert len(target.stakeholders) == 3
		assert target.is_public is True
	
	def test_target_risk_factors(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_metrics: List[ESGMetric]):
		"""Test target risk factors JSON field"""
		risk_factors = [
			"regulatory_changes",
			"technology_availability", 
			"budget_constraints",
			"market_conditions"
		]
		
		target = ESGTarget(
			id="test_risk_factors",
			tenant_id=sample_tenant.id,
			metric_id=sample_esg_metrics[0].id,
			name="Risk Factors Test Target",
			target_value=Decimal("50.0"),
			start_date=datetime(2024, 1, 1),
			target_date=datetime(2025, 12, 31),
			risk_factors=risk_factors,
			owner_id="test_owner",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(target)
		db_session.commit()
		
		assert len(target.risk_factors) == 4
		assert "regulatory_changes" in target.risk_factors
		assert "budget_constraints" in target.risk_factors
	
	def test_target_optimization_recommendations(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_metrics: List[ESGMetric]):
		"""Test target optimization recommendations"""
		recommendations = [
			{"action": "increase_investment", "impact": 0.25, "cost": 1500000, "timeline": "6_months"},
			{"action": "technology_upgrade", "impact": 0.18, "cost": 800000, "timeline": "12_months"},
			{"action": "process_optimization", "impact": 0.12, "cost": 200000, "timeline": "3_months"}
		]
		
		target = ESGTarget(
			id="test_optimization",
			tenant_id=sample_tenant.id,
			metric_id=sample_esg_metrics[0].id,
			name="Optimization Test Target",
			target_value=Decimal("75.0"),
			start_date=datetime(2024, 1, 1),
			target_date=datetime(2026, 12, 31),
			optimization_recommendations=recommendations,
			owner_id="test_owner",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(target)
		db_session.commit()
		
		assert len(target.optimization_recommendations) == 3
		assert target.optimization_recommendations[0]["action"] == "increase_investment"
		assert target.optimization_recommendations[0]["impact"] == 0.25
		assert target.optimization_recommendations[2]["timeline"] == "3_months"


class TestESGStakeholder:
	"""Test ESG stakeholder model"""
	
	def test_stakeholder_creation(self, db_session: Session, sample_tenant: ESGTenant):
		"""Test stakeholder creation with full details"""
		communication_prefs = {
			"preferred_channels": ["email", "portal", "webinar"],
			"frequency": "monthly",
			"report_formats": ["pdf", "interactive", "summary"],
			"language": "en_US",
			"timezone": "UTC"
		}
		
		engagement_insights = {
			"primary_concerns": ["climate_change", "social_impact", "governance"],
			"satisfaction_level": "high",
			"engagement_trend": "increasing",
			"response_rate": 0.87,
			"preferred_meeting_times": ["tuesday_morning", "thursday_afternoon"]
		}
		
		stakeholder = ESGStakeholder(
			id="test_stakeholder_pension_fund",
			tenant_id=sample_tenant.id,
			name="Sustainable Pension Fund",
			organization="Global Pension Partners",
			stakeholder_type="institutional_investor",
			email="esg@globalpension.com",
			phone="+1-555-9876",
			country="USA",
			region="North America",
			language_preference="en_US",
			communication_preferences=communication_prefs,
			esg_interests=["climate_risk", "diversity_inclusion", "supply_chain"],
			engagement_frequency="monthly",
			engagement_score=Decimal("91.5"),
			last_engagement=datetime.utcnow() - timedelta(days=12),
			next_engagement=datetime.utcnow() + timedelta(days=18),
			total_interactions=42,
			sentiment_score=Decimal("78.2"),
			engagement_insights=engagement_insights,
			influence_score=Decimal("95.0"),
			portal_access=True,
			data_access_level="confidential",
			is_active=True,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(stakeholder)
		db_session.commit()
		
		assert stakeholder.name == "Sustainable Pension Fund"
		assert stakeholder.stakeholder_type == "institutional_investor"
		assert stakeholder.engagement_score == Decimal("91.5")
		assert stakeholder.influence_score == Decimal("95.0")
		assert len(stakeholder.esg_interests) == 3
		assert stakeholder.communication_preferences["frequency"] == "monthly"
		assert stakeholder.engagement_insights["satisfaction_level"] == "high"
	
	def test_stakeholder_engagement_tracking(self, db_session: Session, sample_tenant: ESGTenant):
		"""Test stakeholder engagement tracking fields"""
		stakeholder = ESGStakeholder(
			id="test_engagement_tracking",
			tenant_id=sample_tenant.id,
			name="Engagement Test Stakeholder",
			stakeholder_type="community",
			engagement_frequency="quarterly",
			engagement_score=Decimal("65.0"),
			sentiment_score=Decimal("55.8"),
			total_interactions=15,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(stakeholder)
		db_session.commit()
		
		# Test engagement score is within valid range
		assert 0 <= stakeholder.engagement_score <= 100
		assert -100 <= stakeholder.sentiment_score <= 100
		assert stakeholder.total_interactions >= 0


class TestESGSupplier:
	"""Test ESG supplier model and assessments"""
	
	def test_supplier_creation(self, db_session: Session, sample_tenant: ESGTenant):
		"""Test supplier creation with ESG scoring"""
		ai_risk_analysis = {
			"risk_probability": 0.25,
			"impact_severity": 0.40,
			"risk_trend": "stable",
			"key_risk_indicators": ["carbon_intensity", "labor_practices", "compliance_score"],
			"mitigation_strategies": ["audit_frequency_increase", "training_program", "contract_revision"]
		}
		
		improvement_recommendations = [
			{"area": "energy_efficiency", "priority": "high", "estimated_impact": 0.30, "timeline": "12_months"},
			{"area": "waste_reduction", "priority": "medium", "estimated_impact": 0.18, "timeline": "6_months"},
			{"area": "worker_safety", "priority": "high", "estimated_impact": 0.25, "timeline": "3_months"}
		]
		
		performance_trends = {
			"esg_score_trend": "improving",
			"risk_level_trend": "decreasing",
			"engagement_trend": "increasing",
			"compliance_trend": "stable"
		}
		
		supplier = ESGSupplier(
			id="test_supplier_solar_panels",
			tenant_id=sample_tenant.id,
			name="Solar Panel Manufacturing Co.",
			legal_name="Solar Panel Manufacturing Corporation",
			registration_number="SPM-2020-1234",
			primary_contact="Maria Rodriguez",
			email="esg@solarpanelco.com",
			phone="+1-555-7890",
			country="MEX",
			address="Industrial Zone 5, Mexico City, Mexico",
			industry_sector="renewable_energy_manufacturing",
			business_size="large",
			relationship_start=datetime(2021, 3, 1),
			contract_value=Decimal("25000000.00"),
			criticality_level="high",
			overall_esg_score=Decimal("84.5"),
			environmental_score=Decimal("88.0"),
			social_score=Decimal("82.5"),
			governance_score=Decimal("83.0"),
			risk_level=ESGRiskLevel.LOW,
			risk_factors=["regulatory_compliance", "supply_chain_disruption"],
			last_assessment=datetime.utcnow() - timedelta(days=90),
			next_assessment=datetime.utcnow() + timedelta(days=275),
			ai_risk_analysis=ai_risk_analysis,
			improvement_recommendations=improvement_recommendations,
			performance_trends=performance_trends,
			improvement_program_participant=True,
			certifications=[
				{"name": "ISO 14001", "issued_date": "2022-01-15", "expiry_date": "2025-01-15"},
				{"name": "OHSAS 18001", "issued_date": "2021-08-20", "expiry_date": "2024-08-20"}
			],
			compliance_status={
				"environmental": "compliant",
				"social": "compliant",
				"governance": "compliant",
				"last_audit": "2023-11-20"
			},
			is_active=True,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(supplier)
		db_session.commit()
		
		assert supplier.name == "Solar Panel Manufacturing Co."
		assert supplier.overall_esg_score == Decimal("84.5")
		assert supplier.risk_level == ESGRiskLevel.LOW
		assert len(supplier.improvement_recommendations) == 3
		assert supplier.ai_risk_analysis["risk_probability"] == 0.25
		assert supplier.performance_trends["esg_score_trend"] == "improving"
		assert len(supplier.certifications) == 2


class TestESGMeasurement:
	"""Test ESG measurement model"""
	
	def test_measurement_creation(self, db_session: Session, sample_esg_metrics: List[ESGMetric]):
		"""Test ESG measurement recording"""
		measurement = ESGMeasurement(
			id="test_measurement_001",
			metric_id=sample_esg_metrics[0].id,
			value=Decimal("1250.75"),
			measurement_date=datetime.utcnow(),
			period_start=datetime.utcnow() - timedelta(days=30),
			period_end=datetime.utcnow(),
			data_source="automated_sensors",
			collection_method="iot_sensors",
			metadata={
				"sensor_id": "ENV_001",
				"calibration_date": "2024-01-15",
				"measurement_precision": 0.01,
				"environmental_conditions": {"temperature": 22.5, "humidity": 45.2}
			},
			validation_score=Decimal("94.5"),
			anomaly_score=Decimal("5.2"),
			quality_flags=["validated", "normal_range"],
			notes="Automated measurement from IoT sensor network",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		db_session.add(measurement)
		db_session.commit()
		
		assert measurement.value == Decimal("1250.75")
		assert measurement.data_source == "automated_sensors"
		assert measurement.validation_score == Decimal("94.5")
		assert measurement.metadata["sensor_id"] == "ENV_001"
		assert "validated" in measurement.quality_flags


class TestModelRelationships:
	"""Test relationships between ESG models"""
	
	def test_tenant_framework_relationship(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_framework: ESGFramework):
		"""Test tenant to framework relationship"""
		# Framework should be linked to tenant
		assert sample_esg_framework.tenant_id == sample_tenant.id
		
		# Tenant should have access to frameworks
		frameworks = db_session.query(ESGFramework).filter_by(tenant_id=sample_tenant.id).all()
		assert len(frameworks) >= 1
		assert sample_esg_framework in frameworks
	
	def test_metric_target_relationship(self, db_session: Session, sample_esg_metrics: List[ESGMetric], sample_esg_targets: List[ESGTarget]):
		"""Test metric to target relationship"""
		# Target should be linked to metric
		target = sample_esg_targets[0]
		metric = sample_esg_metrics[0]
		
		assert target.metric_id == metric.id
		
		# Check relationship navigation
		targets_for_metric = db_session.query(ESGTarget).filter_by(metric_id=metric.id).all()
		assert target in targets_for_metric
	
	def test_stakeholder_tenant_relationship(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_stakeholders: List[ESGStakeholder]):
		"""Test stakeholder to tenant relationship"""
		stakeholder = sample_esg_stakeholders[0]
		
		assert stakeholder.tenant_id == sample_tenant.id
		
		# Tenant should have stakeholders
		tenant_stakeholders = db_session.query(ESGStakeholder).filter_by(tenant_id=sample_tenant.id).all()
		assert stakeholder in tenant_stakeholders


@pytest.mark.performance
class TestModelPerformance:
	"""Test model performance and optimization"""
	
	def test_bulk_metric_creation_performance(self, db_session: Session, sample_tenant: ESGTenant, sample_esg_framework: ESGFramework, performance_timer):
		"""Test bulk creation of metrics for performance"""
		performance_timer.start()
		
		metrics = []
		for i in range(100):
			metric = ESGMetric(
				id=f"perf_test_metric_{i:03d}",
				tenant_id=sample_tenant.id,
				framework_id=sample_esg_framework.id,
				name=f"Performance Test Metric {i}",
				code=f"PERF_TEST_{i:03d}",
				metric_type=ESGMetricType.ENVIRONMENTAL,
				category="performance_test",
				unit=ESGMetricUnit.PERCENTAGE,
				current_value=Decimal(str(i * 1.5)),
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			metrics.append(metric)
		
		db_session.add_all(metrics)
		db_session.commit()
		
		performance_timer.stop()
		
		# Should create 100 metrics in under 2 seconds
		performance_timer.assert_max_time(2.0, "Bulk metric creation took too long")
		
		# Verify all metrics were created
		created_metrics = db_session.query(ESGMetric).filter(
			ESGMetric.category == "performance_test"
		).count()
		assert created_metrics == 100
	
	def test_complex_query_performance(self, db_session: Session, sample_tenant: ESGTenant, performance_timer):
		"""Test complex query performance"""
		performance_timer.start()
		
		# Complex query joining multiple tables
		results = db_session.query(ESGMetric)\
			.join(ESGFramework)\
			.join(ESGTarget)\
			.filter(ESGMetric.tenant_id == sample_tenant.id)\
			.filter(ESGMetric.is_kpi == True)\
			.filter(ESGTarget.status == ESGTargetStatus.ON_TRACK)\
			.all()
		
		performance_timer.stop()
		
		# Complex query should complete in under 500ms
		performance_timer.assert_max_time(0.5, "Complex query took too long")


@pytest.mark.integration
class TestModelIntegration:
	"""Test model integration with APG ecosystem"""
	
	def test_tenant_with_full_esg_setup(self, db_session: Session):
		"""Test complete tenant setup with all ESG components"""
		# Create tenant
		tenant = ESGTenant(
			id="integration_test_tenant",
			name="Integration Test Corp",
			slug="integration-test",
			industry="manufacturing",
			ai_enabled=True,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(tenant)
		
		# Create framework
		framework = ESGFramework(
			id="integration_framework",
			tenant_id=tenant.id,
			name="Integration Test Framework",
			code="ITF",
			framework_type=ESGFrameworkType.CUSTOM,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(framework)
		
		# Create metric
		metric = ESGMetric(
			id="integration_metric",
			tenant_id=tenant.id,
			framework_id=framework.id,
			name="Integration Test Metric",
			code="INT_TEST",
			metric_type=ESGMetricType.ENVIRONMENTAL,
			category="integration",
			unit=ESGMetricUnit.PERCENTAGE,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(metric)
		
		# Create target
		target = ESGTarget(
			id="integration_target",
			tenant_id=tenant.id,
			metric_id=metric.id,
			name="Integration Test Target",
			target_value=Decimal("80.0"),
			start_date=datetime(2024, 1, 1),
			target_date=datetime(2025, 12, 31),
			owner_id="integration_owner",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(target)
		
		# Create stakeholder
		stakeholder = ESGStakeholder(
			id="integration_stakeholder",
			tenant_id=tenant.id,
			name="Integration Test Stakeholder",
			stakeholder_type="investor",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		db_session.add(stakeholder)
		
		db_session.commit()
		
		# Verify all components are properly linked
		assert tenant.id == "integration_test_tenant"
		assert framework.tenant_id == tenant.id
		assert metric.tenant_id == tenant.id
		assert metric.framework_id == framework.id
		assert target.tenant_id == tenant.id
		assert target.metric_id == metric.id
		assert stakeholder.tenant_id == tenant.id
		
		# Test cleanup
		db_session.delete(stakeholder)
		db_session.delete(target)
		db_session.delete(metric)
		db_session.delete(framework)
		db_session.delete(tenant)
		db_session.commit()