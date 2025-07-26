"""
Product Lifecycle Management (PLM) Capability - Test Configuration

Pytest configuration and fixtures for PLM capability testing.
Follows APG async testing patterns with real objects and pytest-httpserver.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
import asyncio
import tempfile
import sqlite3
from typing import Dict, Any, AsyncGenerator, Generator
from datetime import datetime, timedelta
from uuid_extensions import uuid7str
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask import Flask
from flask_appbuilder import AppBuilder
from flask_sqlalchemy import SQLAlchemy
from pytest_httpserver import HTTPServer
import threading
import json

from ..models import (
	PLProduct, PLProductStructure, PLEngineeringChange,
	PLProductConfiguration, PLCollaborationSession, PLComplianceRecord,
	PLManufacturingIntegration, PLDigitalTwinBinding, Base
)
from ..service import PLMProductService, PLMEngineeringChangeService, PLMCollaborationService
from ..ai_service import PLMAIService
from .. import PLMCapability, PLMIntegrationService


# Test Configuration

@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope="session")
def test_database_url() -> str:
	"""Create temporary in-memory SQLite database for testing"""
	return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_flask_app(test_database_url: str) -> Flask:
	"""Create Flask application for testing"""
	app = Flask(__name__)
	app.config.update({
		'TESTING': True,
		'SQLALCHEMY_DATABASE_URI': test_database_url,
		'SQLALCHEMY_TRACK_MODIFICATIONS': False,
		'SECRET_KEY': 'test-secret-key-for-sessions',
		'WTF_CSRF_ENABLED': False,
		'APG_TENANT_ID': 'test_tenant',
		'APG_USER_ID': 'test_user'
	})
	return app


@pytest.fixture(scope="session")
def test_db(test_flask_app: Flask) -> SQLAlchemy:
	"""Create SQLAlchemy database for testing"""
	from flask_sqlalchemy import SQLAlchemy
	
	db = SQLAlchemy()
	with test_flask_app.app_context():
		db.init_app(test_flask_app)
		# Create all tables
		db.create_all()
		yield db
		db.drop_all()


@pytest.fixture(scope="session")  
def test_appbuilder(test_flask_app: Flask, test_db: SQLAlchemy) -> AppBuilder:
	"""Create Flask-AppBuilder for testing"""
	appbuilder = AppBuilder(test_flask_app, test_db.session)
	return appbuilder


@pytest.fixture
async def test_tenant_id() -> str:
	"""Test tenant ID for multi-tenant testing"""
	return f"test_tenant_{uuid7str()[:8]}"


@pytest.fixture
async def test_user_id() -> str:
	"""Test user ID for authentication testing"""
	return f"test_user_{uuid7str()[:8]}"


# Service Fixtures

@pytest.fixture
async def plm_product_service() -> PLMProductService:
	"""PLM Product Service fixture"""
	return PLMProductService()


@pytest.fixture
async def plm_change_service() -> PLMEngineeringChangeService:
	"""PLM Engineering Change Service fixture"""
	return PLMEngineeringChangeService()


@pytest.fixture
async def plm_collaboration_service() -> PLMCollaborationService:
	"""PLM Collaboration Service fixture"""
	return PLMCollaborationService()


@pytest.fixture
async def plm_ai_service() -> PLMAIService:
	"""PLM AI Service fixture"""
	return PLMAIService()


@pytest.fixture
async def plm_integration_service() -> PLMIntegrationService:
	"""PLM Integration Service fixture"""
	return PLMIntegrationService()


@pytest.fixture
async def plm_capability() -> PLMCapability:
	"""PLM Capability fixture"""
	capability = PLMCapability()
	await capability.initialize()
	return capability


# Model Data Fixtures

@pytest.fixture
async def sample_product_data(test_tenant_id: str, test_user_id: str) -> Dict[str, Any]:
	"""Sample product data for testing"""
	return {
		'tenant_id': test_tenant_id,
		'created_by': test_user_id,
		'product_name': 'Test Product Alpha',
		'product_number': 'TEST-001',
		'product_description': 'Test product for unit testing',
		'product_type': 'manufactured',
		'lifecycle_phase': 'design',
		'target_cost': 1000.00,
		'current_cost': 950.00,
		'unit_of_measure': 'each',
		'custom_attributes': {'test_attribute': 'test_value'},
		'tags': ['test', 'unit-test', 'alpha']
	}


@pytest.fixture
async def sample_change_data(test_tenant_id: str, test_user_id: str) -> Dict[str, Any]:
	"""Sample engineering change data for testing"""
	return {
		'tenant_id': test_tenant_id,
		'created_by': test_user_id,
		'change_title': 'Test Engineering Change',
		'change_description': 'Test change description for unit testing',
		'change_type': 'design',
		'change_category': 'product_improvement',
		'affected_products': ['test-product-1', 'test-product-2'],
		'reason_for_change': 'Testing engineering change workflow',
		'business_impact': 'Improved test coverage and reliability',
		'cost_impact': 500.00,
		'priority': 'medium'
	}


@pytest.fixture  
async def sample_collaboration_data(test_tenant_id: str, test_user_id: str) -> Dict[str, Any]:
	"""Sample collaboration session data for testing"""
	start_time = datetime.utcnow() + timedelta(hours=1)
	end_time = start_time + timedelta(hours=2)
	
	return {
		'tenant_id': test_tenant_id,
		'host_user_id': test_user_id,
		'session_name': 'Test Collaboration Session',
		'description': 'Test collaboration session for unit testing',
		'session_type': 'design_review',
		'scheduled_start': start_time,
		'scheduled_end': end_time,
		'max_participants': 10,
		'recording_enabled': False,
		'whiteboard_enabled': True,
		'file_sharing_enabled': True
	}


# Database Fixtures

@pytest.fixture
async def test_product(
	test_db: SQLAlchemy, 
	sample_product_data: Dict[str, Any],
	test_flask_app: Flask
) -> PLProduct:
	"""Create test product in database"""
	with test_flask_app.app_context():
		product = PLProduct(**sample_product_data)
		test_db.session.add(product)
		test_db.session.commit()
		test_db.session.refresh(product)
		return product


@pytest.fixture
async def test_change(
	test_db: SQLAlchemy,
	sample_change_data: Dict[str, Any],
	test_flask_app: Flask
) -> PLEngineeringChange:
	"""Create test engineering change in database"""
	with test_flask_app.app_context():
		change = PLEngineeringChange(**sample_change_data)
		test_db.session.add(change)
		test_db.session.commit()
		test_db.session.refresh(change)
		return change


@pytest.fixture
async def test_collaboration_session(
	test_db: SQLAlchemy,
	sample_collaboration_data: Dict[str, Any],
	test_flask_app: Flask
) -> PLCollaborationSession:
	"""Create test collaboration session in database"""
	with test_flask_app.app_context():
		session = PLCollaborationSession(**sample_collaboration_data)
		test_db.session.add(session)
		test_db.session.commit()
		test_db.session.refresh(session)
		return session


@pytest.fixture
async def multiple_test_products(
	test_db: SQLAlchemy,
	test_tenant_id: str,
	test_user_id: str,
	test_flask_app: Flask
) -> list[PLProduct]:
	"""Create multiple test products for pagination and search testing"""
	with test_flask_app.app_context():
		products = []
		
		product_data_list = [
			{
				'tenant_id': test_tenant_id,
				'created_by': test_user_id,
				'product_name': 'Alpha Product',
				'product_number': 'ALPHA-001',
				'product_type': 'manufactured',
				'lifecycle_phase': 'design',
				'target_cost': 1000.00,
				'tags': ['alpha', 'test']
			},
			{
				'tenant_id': test_tenant_id,
				'created_by': test_user_id,
				'product_name': 'Beta Product',
				'product_number': 'BETA-001',
				'product_type': 'purchased',
				'lifecycle_phase': 'production',
				'target_cost': 2000.00,
				'tags': ['beta', 'test']
			},
			{
				'tenant_id': test_tenant_id,
				'created_by': test_user_id,
				'product_name': 'Gamma Product',
				'product_number': 'GAMMA-001',
				'product_type': 'virtual',
				'lifecycle_phase': 'active',
				'target_cost': 3000.00,
				'tags': ['gamma', 'test']
			}
		]
		
		for data in product_data_list:
			product = PLProduct(**data)
			test_db.session.add(product)
			products.append(product)
		
		test_db.session.commit()
		
		for product in products:
			test_db.session.refresh(product)
		
		return products


# HTTP Server Fixtures (for external API testing)

@pytest.fixture
def http_server() -> Generator[HTTPServer, None, None]:
	"""HTTP server for testing external API integrations"""
	server = HTTPServer(host="127.0.0.1", port=0)
	server.start()
	yield server
	server.stop()


@pytest.fixture
def mock_manufacturing_api(http_server: HTTPServer) -> str:
	"""Mock manufacturing API for integration testing"""
	# Mock manufacturing BOM sync endpoint
	http_server.expect_request(
		"/api/v1/manufacturing/bom/sync",
		method="POST"
	).respond_with_json(
		{"success": True, "bom_id": "mock_bom_123"},
		status=201
	)
	
	# Mock manufacturing status endpoint
	http_server.expect_request(
		"/api/v1/manufacturing/status",
		method="GET"
	).respond_with_json(
		{"status": "active", "capacity": 85},
		status=200
	)
	
	return f"http://{http_server.host}:{http_server.port}"


@pytest.fixture
def mock_digital_twin_api(http_server: HTTPServer) -> str:
	"""Mock digital twin API for integration testing"""
	# Mock digital twin creation endpoint
	http_server.expect_request(
		"/api/v1/digital-twins",
		method="POST"
	).respond_with_json(
		{"success": True, "twin_id": f"twin_{uuid7str()[:8]}"},
		status=201
	)
	
	# Mock digital twin data endpoint
	http_server.expect_request(
		"/api/v1/digital-twins/*/data",
		method="GET"
	).respond_with_json(
		{
			"3d_model": "mock_model_data",
			"properties": {"height": 10, "width": 5, "depth": 3},
			"status": "active"
		},
		status=200
	)
	
	return f"http://{http_server.host}:{http_server.port}"


@pytest.fixture
def mock_ai_orchestration_api(http_server: HTTPServer) -> str:
	"""Mock AI orchestration API for AI service testing"""
	# Mock design optimization endpoint
	http_server.expect_request(
		"/api/v1/ai/design/optimize",
		method="POST"
	).respond_with_json(
		{
			"optimization_id": f"opt_{uuid7str()[:8]}",
			"status": "completed",
			"optimized_parameters": {
				"weight_reduction": 15.5,
				"cost_reduction": 8.2,
				"strength_improvement": 12.1
			},
			"recommendations": [
				"Reduce material thickness by 2mm",
				"Optimize internal geometry",
				"Consider alternative materials"
			]
		},
		status=200
	)
	
	# Mock innovation insights endpoint
	http_server.expect_request(
		"/api/v1/ai/innovation/insights",
		method="GET"
	).respond_with_json(
		{
			"insights": [
				"Market trending towards sustainable materials",
				"Additive manufacturing adoption increasing",
				"IoT integration becoming standard"
			],
			"confidence_scores": [0.89, 0.76, 0.82]
		},
		status=200
	)
	
	return f"http://{http_server.host}:{http_server.port}"


# Performance Testing Fixtures

@pytest.fixture
def performance_test_data(test_tenant_id: str, test_user_id: str) -> Dict[str, Any]:
	"""Generate data for performance testing"""
	return {
		'tenant_id': test_tenant_id,
		'user_id': test_user_id,
		'product_count': 100,
		'change_count': 50,
		'collaboration_count': 25
	}


# Utility Functions

def assert_async_result(coro, expected_result=None, should_raise=False):
	"""Helper function to test async functions"""
	loop = asyncio.get_event_loop()
	
	if should_raise:
		with pytest.raises(Exception):
			loop.run_until_complete(coro)
	else:
		result = loop.run_until_complete(coro)
		if expected_result is not None:
			assert result == expected_result
		return result


def create_mock_session_data(test_tenant_id: str, test_user_id: str) -> Dict[str, Any]:
	"""Create mock session data for Flask testing"""
	return {
		'tenant_id': test_tenant_id,
		'user_id': test_user_id,
		'authenticated': True,
		'permissions': [
			'plm.products.read',
			'plm.products.create',
			'plm.products.update',
			'plm.changes.read',
			'plm.changes.create',
			'plm.collaboration.read',
			'plm.collaboration.create'
		]
	}


# Module exports for test utilities
__all__ = [
	'test_database_url',
	'test_flask_app',
	'test_db',
	'test_appbuilder',
	'test_tenant_id',
	'test_user_id',
	'plm_product_service',
	'plm_change_service',
	'plm_collaboration_service',
	'plm_ai_service',
	'plm_integration_service',
	'plm_capability',
	'sample_product_data',
	'sample_change_data',
	'sample_collaboration_data',
	'test_product',
	'test_change',
	'test_collaboration_session',
	'multiple_test_products',
	'http_server',
	'mock_manufacturing_api',
	'mock_digital_twin_api',
	'mock_ai_orchestration_api',
	'performance_test_data',
	'assert_async_result',
	'create_mock_session_data'
]