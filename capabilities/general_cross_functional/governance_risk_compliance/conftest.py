#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Governance, Risk & Compliance - PyTest Configuration

Global test configuration, fixtures, and utilities for GRC capability testing.
Provides shared test infrastructure, mock services, and testing utilities.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
import os
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from flask import Flask
from flask_appbuilder import AppBuilder

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
	GRCMetrics
)

# Test configuration constants
TEST_DATABASE_URL = "sqlite:///:memory:"
TEST_TENANT_SLUG = "test-tenant"

# Flask application configuration for testing
FLASK_TEST_CONFIG = {
	"SQLALCHEMY_DATABASE_URI": TEST_DATABASE_URL,
	"SECRET_KEY": "test_secret_key_grc_2024",
	"WTF_CSRF_ENABLED": False,
	"TESTING": True,
	"DEBUG": True,
	"SERVER_NAME": "localhost:5000"
}

# Mock configuration for external services
MOCK_AI_CONFIG = {
	"models": {
		"risk_prediction": {"accuracy": 0.95, "enabled": True},
		"compliance_monitoring": {"threshold": 0.8, "enabled": True},
		"fraud_detection": {"ensemble_size": 5, "enabled": True}
	},
	"endpoints": {
		"prediction_api": "http://mock-ai-service:8080/predict",
		"training_api": "http://mock-ai-service:8080/train"
	}
}

@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()

@pytest.fixture(scope="session")
def test_database_engine():
	"""Create database engine for testing session."""
	engine = create_engine(
		TEST_DATABASE_URL,
		echo=False,  # Set to True for SQL debugging
		pool_pre_ping=True,
		pool_recycle=300
	)
	
	# Create all tables
	Base.metadata.create_all(bind=engine)
	
	yield engine
	
	# Cleanup
	Base.metadata.drop_all(bind=engine)
	engine.dispose()

@pytest.fixture(scope="function")
def db_session(test_database_engine) -> Generator[Session, None, None]:
	"""Create fresh database session for each test function."""
	SessionLocal = sessionmaker(
		autocommit=False,
		autoflush=False, 
		bind=test_database_engine
	)
	
	session = SessionLocal()
	
	try:
		yield session
	finally:
		session.rollback()
		session.close()

@pytest.fixture(scope="function")
def flask_app() -> Generator[Flask, None, None]:
	"""Create Flask application instance for testing."""
	app = Flask(__name__)
	app.config.update(FLASK_TEST_CONFIG)
	
	with app.app_context():
		yield app

@pytest.fixture(scope="function")
def flask_client(flask_app: Flask):
	"""Create Flask test client."""
	return flask_app.test_client()

@pytest.fixture(scope="function")
def app_builder(flask_app: Flask, db_session: Session):
	"""Create Flask-AppBuilder instance for testing."""
	with flask_app.app_context():
		appbuilder = AppBuilder(flask_app, db_session)
		yield appbuilder

# Tenant and basic data fixtures
@pytest.fixture(scope="function")
def test_tenant(db_session: Session) -> GRCTenant:
	"""Create test tenant with default configuration."""
	tenant = GRCTenant(
		name="Test Corporation",
		slug=TEST_TENANT_SLUG,
		domain="test.example.com",
		settings={
			"timezone": "UTC",
			"language": "en",
			"ai_features": {
				"risk_prediction": True,
				"compliance_monitoring": True,
				"governance_assistance": True
			},
			"notification_preferences": {
				"email": True,
				"sms": False,
				"in_app": True
			},
			"security": {
				"mfa_required": True,
				"session_timeout": 3600,
				"password_policy": "strong"
			}
		},
		is_active=True
	)
	
	db_session.add(tenant)
	db_session.commit()
	db_session.refresh(tenant)
	
	return tenant

@pytest.fixture(scope="function")
def test_risk_category(db_session: Session, test_tenant: GRCTenant) -> GRCRiskCategory:
	"""Create test risk category."""
	category = GRCRiskCategory(
		tenant_id=test_tenant.id,
		name="Operational Risk",
		description="Risks related to day-to-day business operations",
		color_code="#FF6B6B",
		risk_appetite=Decimal("75.0"),
		parent_category_id=None
	)
	
	db_session.add(category)
	db_session.commit()
	db_session.refresh(category)
	
	return category

@pytest.fixture(scope="function")
def test_risk(db_session: Session, test_tenant: GRCTenant, test_risk_category: GRCRiskCategory) -> GRCRisk:
	"""Create test risk with AI insights."""
	risk = GRCRisk(
		tenant_id=test_tenant.id,
		title="Test Cybersecurity Risk",
		description="Sample cybersecurity risk for testing purposes",
		category_id=test_risk_category.id,
		current_likelihood=Decimal("40.0"),
		current_impact=Decimal("70.0"),
		risk_score=Decimal("28.0"),
		predicted_likelihood=Decimal("45.0"),
		predicted_impact=Decimal("65.0"),
		prediction_confidence=Decimal("87.5"),
		ai_insights={
			"trend": "increasing",
			"key_factors": ["increased_remote_work", "new_attack_vectors"],
			"recommendations": [
				"Implement zero-trust architecture",
				"Enhance employee security training",
				"Deploy advanced endpoint detection"
			],
			"model_metadata": {
				"model_version": "v2.1",
				"training_date": "2024-01-15",
				"accuracy": 0.92
			}
		},
		status="active",
		owner_id="risk_manager_001",
		next_review_date=datetime.utcnow() + timedelta(days=90),
		created_by="system_admin",
		last_updated_by="risk_manager_001"
	)
	
	db_session.add(risk)
	db_session.commit()
	db_session.refresh(risk)
	
	return risk

@pytest.fixture(scope="function")
def test_control(db_session: Session, test_tenant: GRCTenant) -> GRCControl:
	"""Create test control with automation configuration."""
	control = GRCControl(
		tenant_id=test_tenant.id,
		name="Multi-Factor Authentication Control",
		description="Enforce MFA for all user accounts accessing sensitive systems",
		control_type="preventive",
		control_frequency="continuous",
		owner_id="security_team_lead",
		implementation_status="implemented",
		effectiveness_rating=Decimal("92.5"),
		last_tested=datetime.utcnow() - timedelta(days=15),
		next_test_date=datetime.utcnow() + timedelta(days=75),
		automated=True,
		automation_config={
			"test_script": "mfa_compliance_check.py",
			"test_frequency": "daily",
			"alert_on_failure": True,
			"auto_remediation": False,
			"test_parameters": {
				"coverage_threshold": 0.98,
				"exception_limit": 5
			}
		},
		ai_testing_enabled=True,
		ai_test_results={
			"last_test_score": 0.95,
			"anomalies_detected": 1,
			"performance_trend": "stable",
			"recommendations": [
				"Review bypass procedures for emergency accounts",
				"Consider implementing adaptive authentication"
			],
			"test_history": [
				{"date": "2024-01-10", "score": 0.94},
				{"date": "2024-01-11", "score": 0.95},
				{"date": "2024-01-12", "score": 0.93}
			]
		}
	)
	
	db_session.add(control)
	db_session.commit()
	db_session.refresh(control)
	
	return control

@pytest.fixture(scope="function")
def test_policy(db_session: Session, test_tenant: GRCTenant) -> GRCPolicy:
	"""Create test policy with AI assistance."""
	policy = GRCPolicy(
		tenant_id=test_tenant.id,
		title="Information Security Policy",
		description="Comprehensive information security policy covering data protection, access controls, and incident response",
		policy_type="security",
		version="2.0",
		effective_date=datetime.utcnow(),
		review_date=datetime.utcnow() + timedelta(days=365),
		owner_id="ciso",
		approver_id="ceo",
		approval_date=datetime.utcnow() - timedelta(days=14),
		content={
			"sections": [
				{
					"title": "Purpose and Scope",
					"content": "This policy establishes information security requirements for all organizational assets and data.",
					"subsections": ["Applicability", "Definitions", "Responsibilities"]
				},
				{
					"title": "Access Control Requirements",
					"content": "Access to information systems must be controlled through multi-factor authentication and role-based permissions.",
					"subsections": ["User Access Management", "Privileged Access", "Remote Access"]
				},
				{
					"title": "Data Protection Standards",
					"content": "All sensitive data must be classified, encrypted, and handled according to established procedures.",
					"subsections": ["Data Classification", "Encryption Requirements", "Data Retention"]
				}
			],
			"procedures": [
				"User account provisioning and deprovisioning",
				"Incident response and reporting",
				"Security awareness training"
			],
			"compliance_requirements": [
				"Annual security risk assessment",
				"Quarterly access reviews",
				"Monthly security metrics reporting"
			]
		},
		ai_assistance_used=True,
		ai_suggestions=[
			"Consider adding zero-trust principles to access control section",
			"Include cloud security requirements in infrastructure section",
			"Add privacy-by-design principles for GDPR compliance"
		],
		compliance_mapping=["ISO27001", "NIST_CSF", "GDPR", "SOX"],
		status="active"
	)
	
	db_session.add(policy)
	db_session.commit()
	db_session.refresh(policy)
	
	return policy

@pytest.fixture(scope="function")
def test_regulation(db_session: Session, test_tenant: GRCTenant) -> GRCRegulation:
	"""Create test regulation with AI monitoring."""
	regulation = GRCRegulation(
		tenant_id=test_tenant.id,
		title="GDPR Article 32 - Security of Processing",
		description="Requirements for implementing appropriate technical and organizational measures for data security",
		regulatory_body="European Commission",
		effective_date=datetime(2018, 5, 25),
		last_updated=datetime.utcnow() - timedelta(days=30),
		jurisdiction=["EU", "EEA", "UK"],
		compliance_deadline=datetime.utcnow() + timedelta(days=90),
		penalty_range="Up to €20 million or 4% of annual turnover",
		requirements=[
			"Implement appropriate technical measures for data security",
			"Establish organizational measures for ongoing security",
			"Conduct regular testing and evaluation of security measures",
			"Document security incidents and breaches"
		],
		ai_monitoring_config={
			"keywords": [
				"data protection",
				"security measures", 
				"GDPR",
				"technical safeguards",
				"organizational measures"
			],
			"monitoring_frequency": "daily",
			"alert_threshold": 0.8,
			"sources": [
				"official_gazette",
				"regulatory_websites",
				"legal_databases"
			]
		},
		status="active"
	)
	
	db_session.add(regulation)
	db_session.commit()
	db_session.refresh(regulation)
	
	return regulation

# Mock service fixtures
@pytest.fixture(scope="function")
def mock_ai_engine():
	"""Create mock AI engine for testing."""
	mock = Mock()
	
	# Configure mock methods with realistic responses
	mock.predict_risk_evolution = AsyncMock(return_value={
		"predicted_likelihood": 42.5,
		"predicted_impact": 68.3,
		"confidence": 0.89,
		"trend": "increasing",
		"recommendations": ["Enhance monitoring", "Review controls"]
	})
	
	mock.detect_compliance_anomalies = AsyncMock(return_value={
		"anomalies": [],
		"total_processed": 100,
		"anomaly_count": 0,
		"model_confidence": 0.91
	})
	
	mock.process_regulatory_document = AsyncMock(return_value={
		"requirements": [],
		"compliance_obligations": [],
		"related_frameworks": []
	})
	
	mock.detect_fraudulent_transactions = AsyncMock(return_value={
		"predictions": [],
		"low_risk_count": 95,
		"high_risk_count": 5,
		"ensemble_confidence": 0.88
	})
	
	return mock

@pytest.fixture(scope="function")
def mock_websocket_server():
	"""Create mock WebSocket server for real-time testing."""
	mock = Mock()
	
	mock.start_server = AsyncMock()
	mock.stop_server = AsyncMock()
	mock.broadcast_alert = AsyncMock()
	mock.send_to_client = AsyncMock()
	mock.get_connected_clients = Mock(return_value=[])
	
	return mock

@pytest.fixture(scope="function")
def mock_prometheus_metrics():
	"""Create mock Prometheus metrics for monitoring testing."""
	mock = Mock()
	
	# Mock metric types
	mock.Counter = Mock()
	mock.Histogram = Mock()
	mock.Gauge = Mock()
	mock.Summary = Mock()
	
	# Mock metric instances
	mock.risk_assessments_total = Mock()
	mock.control_tests_duration = Mock()
	mock.compliance_score = Mock()
	mock.ai_prediction_accuracy = Mock()
	
	return mock

# Utility fixtures and helpers
@pytest.fixture(scope="function")
def temp_directory():
	"""Create temporary directory for file-based testing."""
	temp_dir = tempfile.mkdtemp(prefix="grc_test_")
	yield temp_dir
	shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def sample_risk_data() -> Dict[str, Any]:
	"""Provide sample risk data for testing."""
	return {
		"title": "Sample Test Risk",
		"description": "A sample risk for testing purposes",
		"current_likelihood": 35.0,
		"current_impact": 60.0,
		"status": "active",
		"owner_id": "test_user",
		"category": "operational",
		"ai_insights": {
			"trend": "stable",
			"confidence": 0.85,
			"recommendations": ["Monitor closely", "Review controls"]
		}
	}

@pytest.fixture(scope="function")
def sample_control_data() -> Dict[str, Any]:
	"""Provide sample control data for testing."""
	return {
		"name": "Sample Test Control",
		"description": "A sample control for testing purposes",
		"control_type": "preventive",
		"effectiveness_rating": 85.0,
		"automated": True,
		"owner_id": "control_owner",
		"automation_config": {
			"test_frequency": "weekly",
			"alert_on_failure": True
		}
	}

# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_timer():
	"""Utility for measuring test performance."""
	import time
	
	class PerformanceTimer:
		def __init__(self):
			self.start_time = None
			self.end_time = None
			
		def start(self):
			self.start_time = time.time()
			
		def stop(self):
			self.end_time = time.time()
			
		def elapsed(self) -> float:
			if self.start_time and self.end_time:
				return self.end_time - self.start_time
			return 0.0
	
	return PerformanceTimer()

# Async context managers for testing
@pytest.fixture(scope="function")
async def async_db_session(test_database_engine):
	"""Async database session for async testing."""
	from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
	from sqlalchemy.orm import sessionmaker
	
	# Note: For testing purposes, we'll use the sync engine
	# In production, you'd use an async engine
	SessionLocal = sessionmaker(
		bind=test_database_engine,
		class_=Session,
		autocommit=False,
		autoflush=False
	)
	
	session = SessionLocal()
	try:
		yield session
	finally:
		await asyncio.create_task(asyncio.coroutine(session.close)())

# Test data generators
def generate_test_risks(count: int, tenant_id: str, category_id: str) -> List[GRCRisk]:
	"""Generate multiple test risks for bulk testing."""
	risks = []
	
	for i in range(count):
		risk = GRCRisk(
			tenant_id=tenant_id,
			title=f"Bulk Test Risk {i+1}",
			description=f"Risk {i+1} generated for bulk testing",
			category_id=category_id,
			current_likelihood=Decimal(str(20.0 + (i * 5) % 60)),
			current_impact=Decimal(str(30.0 + (i * 7) % 50)),
			status="active",
			owner_id=f"owner_{i % 5}"
		)
		risks.append(risk)
	
	return risks

def generate_test_controls(count: int, tenant_id: str) -> List[GRCControl]:
	"""Generate multiple test controls for bulk testing."""
	controls = []
	control_types = ["preventive", "detective", "corrective"]
	
	for i in range(count):
		control = GRCControl(
			tenant_id=tenant_id,
			name=f"Bulk Test Control {i+1}",
			description=f"Control {i+1} generated for bulk testing",
			control_type=control_types[i % 3],
			effectiveness_rating=Decimal(str(70.0 + (i * 3) % 25)),
			automated=i % 2 == 0,
			owner_id=f"control_owner_{i % 3}"
		)
		controls.append(control)
	
	return controls

# Custom pytest markers
def pytest_configure(config):
	"""Configure custom pytest markers."""
	config.addinivalue_line(
		"markers", "integration: mark test as integration test"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as performance test"
	)
	config.addinivalue_line(
		"markers", "ai_dependent: mark test as requiring AI services"
	)
	config.addinivalue_line(
		"markers", "database_heavy: mark test as database intensive"
	)
	config.addinivalue_line(
		"markers", "async_test: mark test as async test"
	)

# Test environment validation
@pytest.fixture(autouse=True, scope="session")
def validate_test_environment():
	"""Validate test environment setup."""
	# Check Python version
	import sys
	assert sys.version_info >= (3, 12), "Tests require Python 3.12 or higher"
	
	# Check required environment variables
	required_env_vars = ["PYTHONPATH"]
	for var in required_env_vars:
		if var not in os.environ:
			pytest.skip(f"Required environment variable {var} not set")
	
	# Validate test database
	try:
		engine = create_engine(TEST_DATABASE_URL)
		with engine.connect() as conn:
			conn.execute(text("SELECT 1"))
		engine.dispose()
	except Exception as e:
		pytest.skip(f"Test database not accessible: {e}")

# Cleanup fixtures
@pytest.fixture(autouse=True, scope="function")
def cleanup_test_data(db_session: Session):
	"""Automatically cleanup test data after each test."""
	yield
	
	# Cleanup order matters due to foreign key constraints
	try:
		db_session.query(GRCAuditLog).delete()
		db_session.query(GRCIncident).delete()
		db_session.query(GRCRiskAssessment).delete()
		db_session.query(GRCRisk).delete()
		db_session.query(GRCRiskCategory).delete()
		db_session.query(GRCControl).delete()
		db_session.query(GRCPolicy).delete()
		db_session.query(GRCRegulation).delete()
		db_session.query(GRCGovernanceDecision).delete()
		db_session.query(GRCMetrics).delete()
		db_session.query(GRCTenant).delete()
		db_session.commit()
	except Exception:
		db_session.rollback()
		raise