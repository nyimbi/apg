"""
APG Biometric Authentication - API Tests

Comprehensive test suite for revolutionary biometric authentication API endpoints.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from flask import Flask
from flask.testing import FlaskClient
from flask_appbuilder import AppBuilder
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..models import Base, BiUser, BiVerification, BiCollaboration, BiBehavioralSession
from ..api import biometric_bp
from ..service import BiometricAuthenticationService

# Test Configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_TENANT_ID = "test_tenant_api"

@pytest.fixture
def app():
	"""Create Flask test application"""
	app = Flask(__name__)
	app.config['TESTING'] = True
	app.config['SECRET_KEY'] = 'test_secret_key'
	app.config['WTF_CSRF_ENABLED'] = False
	
	# Register blueprint
	app.register_blueprint(biometric_bp)
	
	return app

@pytest.fixture
def client(app):
	"""Create Flask test client"""
	return app.test_client()

@pytest.fixture
async def db_session():
	"""Create test database session"""
	engine = create_async_engine(TEST_DATABASE_URL, echo=False)
	
	# Create tables
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	
	# Create session
	async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
	
	async with async_session() as session:
		yield session
	
	# Cleanup
	await engine.dispose()

@pytest.fixture
def auth_headers():
	"""Create authentication headers for API requests"""
	return {
		'Authorization': 'Bearer test_token_123',
		'Content-Type': 'application/json',
		'X-Tenant-ID': TEST_TENANT_ID
	}

@pytest.fixture
def sample_user_data():
	"""Sample user data for API testing"""
	return {
		'external_user_id': 'api_test_user_001',
		'tenant_id': TEST_TENANT_ID,
		'first_name': 'Jane',
		'last_name': 'Smith',
		'email': 'jane.smith@example.com',
		'phone': '+1987654321'
	}

@pytest.fixture
def sample_verification_request():
	"""Sample biometric verification request"""
	return {
		'user_id': 'test_user_id_123',
		'verification_type': 'facial_recognition',
		'biometric_data': {
			'template_data': 'base64_encoded_template',
			'quality_score': 0.92,
			'capture_metadata': {
				'device_type': 'mobile_camera',
				'resolution': '1920x1080'
			},
			'behavioral_data': {
				'keystroke_patterns': [{'key': 'a', 'dwell_time': 150}],
				'mouse_patterns': [{'x': 100, 'y': 200}]
			}
		},
		'modality': 'face',
		'business_context': {
			'transaction_type': 'account_access',
			'risk_level': 'medium'
		},
		'liveness_required': True,
		'collaboration_enabled': False
	}

@pytest.fixture
def sample_nl_query_request():
	"""Sample natural language query request"""
	return {
		'query': 'Show me all failed verifications from the last 24 hours',
		'user_context': {
			'user_id': 'test_analyst_001',
			'role': 'fraud_analyst',
			'department': 'security'
		},
		'security_clearance': 'elevated',
		'conversation_history': []
	}

@pytest.fixture
def sample_collaboration_request():
	"""Sample collaboration session request"""
	return {
		'session_name': 'High-Risk Identity Verification Review',
		'verification_id': 'test_verification_123',
		'participants': ['expert_001', 'expert_002', 'specialist_003'],
		'expertise_requirements': ['fraud_detection', 'document_analysis'],
		'priority': 'high'
	}

@pytest.fixture
def sample_behavioral_session_request():
	"""Sample behavioral session request"""
	return {
		'user_id': 'test_user_behavioral_001',
		'device_fingerprint': 'test_device_fingerprint_api_456',
		'platform': 'Android',
		'user_agent': 'Mozilla/5.0 (Linux; Android 11; SM-G975F)'
	}

# Authentication Endpoint Tests

class TestBiometricVerificationAPI:
	"""Test suite for biometric verification API endpoints"""
	
	@patch('capabilities.common.biometric.api.get_biometric_service')
	def test_biometric_verification_success(self, mock_get_service, client, auth_headers, sample_verification_request):
		"""Test successful biometric verification via API"""
		# Arrange
		mock_service = AsyncMock()
		mock_get_service.return_value = mock_service
		
		# Mock service responses
		mock_verification = Mock()
		mock_verification.id = 'test_verification_id_123'
		mock_verification.status.value = 'verified'
		mock_verification.confidence_score = 0.95
		mock_verification.risk_score = 0.15
		mock_verification.processing_time_ms = 250
		mock_verification.contextual_risk_assessment = {'risk_level': 'low'}
		mock_verification.fraud_prediction = {'fraud_probability': 0.02}
		mock_verification.fusion_analysis = {'fusion_confidence': 0.95}
		mock_verification.regulatory_requirements = {'compliance': 'GDPR'}
		
		mock_fusion_result = Mock()
		mock_fusion_result.fusion_confidence = 0.95
		
		mock_service.start_verification.return_value = mock_verification
		mock_service.process_biometric_verification.return_value = mock_fusion_result
		mock_service.complete_verification.return_value = mock_verification
		
		# Act
		response = client.post(
			'/api/v1/biometric/auth/verify',
			data=json.dumps(sample_verification_request),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['verification_id'] == 'test_verification_id_123'
		assert data['status'] == 'verified'
		assert data['confidence_score'] == 0.95
		assert data['risk_score'] == 0.15
		assert data['processing_time_ms'] == 250
		assert 'contextual_intelligence' in data
		assert 'predictive_analytics' in data
		assert 'fusion_analysis' in data
		assert 'compliance_status' in data
	
	def test_biometric_verification_missing_data(self, client, auth_headers):
		"""Test biometric verification with missing required data"""
		# Arrange
		invalid_request = {
			'verification_type': 'facial_recognition'
			# Missing user_id and biometric_data
		}
		
		# Act
		response = client.post(
			'/api/v1/biometric/auth/verify',
			data=json.dumps(invalid_request),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 400
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'error' in data
	
	def test_biometric_verification_unauthorized(self, client, sample_verification_request):
		"""Test biometric verification without authorization"""
		# Act
		response = client.post(
			'/api/v1/biometric/auth/verify',
			data=json.dumps(sample_verification_request),
			headers={'Content-Type': 'application/json'}  # No auth header
		)
		
		# Assert
		assert response.status_code == 401  # or 403 depending on implementation
	
	@patch('capabilities.common.biometric.api.get_biometric_service')
	def test_verification_status_success(self, mock_get_service, client, auth_headers):
		"""Test successful verification status retrieval"""
		# Arrange
		verification_id = 'test_verification_status_123'
		mock_service = AsyncMock()
		mock_get_service.return_value = mock_service
		
		mock_verification = Mock()
		mock_verification.id = verification_id
		mock_verification.status.value = 'completed'
		mock_verification.confidence_score = 0.88
		mock_verification.risk_score = 0.25
		mock_verification.processing_time_ms = 300
		mock_verification.started_at.isoformat.return_value = '2025-01-29T10:00:00'
		mock_verification.completed_at.isoformat.return_value = '2025-01-29T10:00:01'
		mock_verification.contextual_risk_assessment = {'context': 'analyzed'}
		mock_verification.fraud_prediction = {'prediction': 'low_risk'}
		mock_verification.behavioral_forecast = {'behavior': 'normal'}
		mock_verification.regulatory_requirements = {'compliance': 'complete'}
		mock_verification.collaboration_session_id = None
		mock_verification.audit_logs = [Mock()]
		
		mock_service._get_verification_with_relations.return_value = mock_verification
		
		# Act
		response = client.get(
			f'/api/v1/biometric/auth/verify/{verification_id}/status',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['verification']['id'] == verification_id
		assert data['verification']['status'] == 'completed'
		assert data['verification']['confidence_score'] == 0.88
		assert data['verification']['collaboration_active'] is False
		assert data['verification']['audit_trail_complete'] is True
	
	@patch('capabilities.common.biometric.api.get_biometric_service')
	def test_verification_status_not_found(self, mock_get_service, client, auth_headers):
		"""Test verification status for nonexistent verification"""
		# Arrange
		verification_id = 'nonexistent_verification_id'
		mock_service = AsyncMock()
		mock_get_service.return_value = mock_service
		mock_service._get_verification_with_relations.return_value = None
		
		# Act
		response = client.get(
			f'/api/v1/biometric/auth/verify/{verification_id}/status',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 404
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'not found' in data['error'].lower()

# Natural Language Interface Tests

class TestNaturalLanguageAPI:
	"""Test suite for natural language interface API endpoints"""
	
	@patch('capabilities.common.biometric.api.get_biometric_service')
	def test_natural_language_query_success(self, mock_get_service, client, auth_headers, sample_nl_query_request):
		"""Test successful natural language query processing"""
		# Arrange
		mock_service = AsyncMock()
		mock_get_service.return_value = mock_service
		
		mock_response = {
			'natural_language_response': 'Found 5 failed verifications in the last 24 hours. The main failure reasons were liveness detection failures (60%) and low quality biometric data (40%).',
			'structured_data': {
				'failed_verifications': 5,
				'failure_reasons': {'liveness_failure': 3, 'low_quality': 2},
				'time_period': '24_hours'
			},
			'confidence': 0.92,
			'follow_up_suggestions': [
				'Would you like to see detailed failure analysis?',
				'Should I generate a report for the security team?'
			],
			'conversation_context': {
				'intent': {'category': 'verification_status', 'confidence': 0.92},
				'user_context': sample_nl_query_request['user_context'],
				'timestamp': datetime.utcnow().isoformat()
			}
		}
		
		mock_service.process_natural_language_query.return_value = mock_response
		
		# Act
		response = client.post(
			'/api/v1/biometric/nl/query',
			data=json.dumps(sample_nl_query_request),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'Found 5 failed verifications' in data['response']
		assert data['structured_data']['failed_verifications'] == 5
		assert data['confidence'] == 0.92
		assert len(data['follow_up_suggestions']) == 2
		assert 'conversation_context' in data
	
	def test_natural_language_query_missing_query(self, client, auth_headers):
		"""Test natural language query with missing query text"""
		# Arrange
		invalid_request = {
			'user_context': {'user_id': 'test_user'},
			'security_clearance': 'standard'
			# Missing 'query' field
		}
		
		# Act
		response = client.post(
			'/api/v1/biometric/nl/query',
			data=json.dumps(invalid_request),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 400
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'error' in data
	
	@patch('capabilities.common.biometric.api.session')
	def test_conversation_history_success(self, mock_session, client, auth_headers):
		"""Test conversation history retrieval"""
		# Arrange
		mock_conversation_history = [
			{
				'timestamp': '2025-01-29T10:00:00',
				'query': 'Show me verification statistics',
				'response': 'Here are your verification statistics...',
				'confidence': 0.95
			},
			{
				'timestamp': '2025-01-29T10:05:00',
				'query': 'What about fraud trends?',
				'response': 'Fraud trends show decreasing pattern...',
				'confidence': 0.88
			}
		]
		
		mock_session.get.return_value = mock_conversation_history
		
		# Act
		response = client.get(
			'/api/v1/biometric/nl/conversation/history',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['total_interactions'] == 2
		assert len(data['conversation_history']) == 2
		assert data['last_interaction'] == '2025-01-29T10:05:00'
	
	@patch('capabilities.common.biometric.api.session')
	def test_conversation_history_empty(self, mock_session, client, auth_headers):
		"""Test conversation history when no history exists"""
		# Arrange
		mock_session.get.return_value = []
		
		# Act
		response = client.get(
			'/api/v1/biometric/nl/conversation/history',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['total_interactions'] == 0
		assert data['conversation_history'] == []
		assert data['last_interaction'] is None

# Collaboration API Tests

class TestCollaborationAPI:
	"""Test suite for collaborative verification API endpoints"""
	
	@patch('capabilities.common.biometric.api.get_db_session')
	@patch('capabilities.common.biometric.api.get_current_tenant_id')
	def test_start_collaborative_session_success(self, mock_get_tenant, mock_get_db, client, auth_headers, sample_collaboration_request):
		"""Test successful collaborative session creation"""
		# Arrange
		mock_get_tenant.return_value = TEST_TENANT_ID
		mock_db_session = AsyncMock()
		mock_get_db.return_value = mock_db_session
		
		# Mock database operations
		mock_collaboration = Mock()
		mock_collaboration.id = 'test_collaboration_123'
		mock_collaboration.session_name = sample_collaboration_request['session_name']
		mock_collaboration.participants = [{'user_id': p, 'role': 'reviewer'} for p in sample_collaboration_request['participants']]
		
		# Act
		response = client.post(
			'/api/v1/biometric/collaboration/start',
			data=json.dumps(sample_collaboration_request),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['collaboration_id'] == 'test_collaboration_123'
		assert data['session_name'] == sample_collaboration_request['session_name']
		assert len(data['participants']) == 3
		assert data['status'] == 'active'
		assert '/biometric/collaboration/real_time_workspace/' in data['workspace_url']
	
	def test_start_collaborative_session_missing_data(self, client, auth_headers):
		"""Test collaborative session creation with missing data"""
		# Arrange
		invalid_request = {
			'session_name': 'Test Session'
			# Missing verification_id and participants
		}
		
		# Act
		response = client.post(
			'/api/v1/biometric/collaboration/start',
			data=json.dumps(invalid_request),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 400
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'error' in data
	
	@patch('capabilities.common.biometric.api.get_db_session')
	@patch('capabilities.common.biometric.api.get_current_tenant_id')
	def test_get_collaborative_workspace_success(self, mock_get_tenant, mock_get_db, client, auth_headers):
		"""Test successful collaborative workspace data retrieval"""
		# Arrange
		collaboration_id = 'test_collaboration_workspace_123'
		mock_get_tenant.return_value = TEST_TENANT_ID
		mock_db_session = AsyncMock()
		mock_get_db.return_value = mock_db_session
		
		# Mock collaboration object
		mock_collaboration = Mock()
		mock_collaboration.synchronization_state = {'status': 'active'}
		mock_collaboration.participants = [{'user_id': 'expert1', 'role': 'reviewer'}]
		mock_collaboration.live_annotations = [{'id': 'annotation1', 'text': 'Test annotation'}]
		mock_collaboration.discussion_threads = [{'id': 'thread1', 'topic': 'Risk assessment'}]
		mock_collaboration.expert_consultations = []
		mock_collaboration.consensus_tracking = {'achieved': False, 'progress': 0.6}
		mock_collaboration.spatial_interactions = []
		mock_collaboration.ar_collaborations = {}
		mock_collaboration.started_at = datetime.utcnow() - timedelta(minutes=10)
		
		# Mock database query
		mock_result = Mock()
		mock_result.scalar_one_or_none.return_value = mock_collaboration
		mock_db_session.execute.return_value = mock_result
		
		# Act
		response = client.get(
			f'/api/v1/biometric/collaboration/{collaboration_id}/workspace',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['collaboration_id'] == collaboration_id
		assert 'workspace_data' in data
		assert data['workspace_data']['session_state']['status'] == 'active'
		assert len(data['workspace_data']['active_participants']) == 1
		assert data['workspace_data']['session_metrics']['participant_count'] == 1
		assert data['workspace_data']['session_metrics']['duration_minutes'] > 0
	
	@patch('capabilities.common.biometric.api.get_db_session')
	@patch('capabilities.common.biometric.api.get_current_tenant_id')
	def test_get_collaborative_workspace_not_found(self, mock_get_tenant, mock_get_db, client, auth_headers):
		"""Test collaborative workspace retrieval for nonexistent session"""
		# Arrange
		collaboration_id = 'nonexistent_collaboration'
		mock_get_tenant.return_value = TEST_TENANT_ID
		mock_db_session = AsyncMock()
		mock_get_db.return_value = mock_db_session
		
		# Mock database query returning None
		mock_result = Mock()
		mock_result.scalar_one_or_none.return_value = None
		mock_db_session.execute.return_value = mock_result
		
		# Act
		response = client.get(
			f'/api/v1/biometric/collaboration/{collaboration_id}/workspace',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 404
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'not found' in data['error'].lower()

# Behavioral Biometrics API Tests

class TestBehavioralAPI:
	"""Test suite for behavioral biometrics API endpoints"""
	
	@patch('capabilities.common.biometric.api.get_biometric_service')
	def test_start_behavioral_session_success(self, mock_get_service, client, auth_headers, sample_behavioral_session_request):
		"""Test successful behavioral session creation"""
		# Arrange
		mock_service = AsyncMock()
		mock_get_service.return_value = mock_service
		
		mock_behavioral_session = Mock()
		mock_behavioral_session.id = 'test_behavioral_session_123'
		mock_behavioral_session.session_token = 'secure_session_token_456'
		mock_behavioral_session.ambient_authentication = {
			'ambient_signatures': {},
			'friction_score': 0.02
		}
		mock_behavioral_session.contextual_strength = {
			'baseline': 0.5,
			'current': 0.8,
			'trend': 'increasing'
		}
		
		mock_service.start_behavioral_session.return_value = mock_behavioral_session
		
		# Act
		response = client.post(
			'/api/v1/biometric/behavioral/session/start',
			data=json.dumps(sample_behavioral_session_request),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['session_id'] == 'test_behavioral_session_123'
		assert data['session_token'] == 'secure_session_token_456'
		assert 'ambient_authentication' in data
		assert 'contextual_strength' in data
		assert data['monitoring_active'] is True
		assert data['ambient_authentication']['friction_score'] == 0.02
	
	def test_start_behavioral_session_missing_data(self, client, auth_headers):
		"""Test behavioral session creation with missing data"""
		# Arrange
		invalid_request = {
			'user_id': 'test_user'
			# Missing device_fingerprint, platform, user_agent
		}
		
		# Act
		response = client.post(
			'/api/v1/biometric/behavioral/session/start',
			data=json.dumps(invalid_request),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 400
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'error' in data
	
	@patch('capabilities.common.biometric.api.get_db_session')
	@patch('capabilities.common.biometric.api.get_current_tenant_id')
	def test_get_behavioral_patterns_success(self, mock_get_tenant, mock_get_db, client, auth_headers):
		"""Test successful behavioral pattern analysis retrieval"""
		# Arrange
		session_id = 'test_behavioral_patterns_123'
		mock_get_tenant.return_value = TEST_TENANT_ID
		mock_db_session = AsyncMock()
		mock_get_db.return_value = mock_db_session
		
		# Mock behavioral session
		mock_behavioral_session = Mock()
		mock_behavioral_session.keystroke_patterns = [{'pattern': 'typing_data'}]
		mock_behavioral_session.mouse_movements = [{'movement': 'mouse_data'}]
		mock_behavioral_session.touch_interactions = [{'touch': 'interaction_data'}]
		mock_behavioral_session.environmental_context = {'environment': 'office'}
		mock_behavioral_session.confidence_timeline = [{'timestamp': '2025-01-29T10:00:00', 'confidence': 0.9}]
		mock_behavioral_session.anomaly_count = 0
		mock_behavioral_session.risk_incidents = 0
		mock_behavioral_session.average_confidence = 0.92
		mock_behavioral_session.invisible_challenges = []
		mock_behavioral_session.seamless_handoffs = []
		mock_behavioral_session.ambient_authentication = {'friction_score': 0.015}
		mock_behavioral_session.started_at = datetime.utcnow() - timedelta(minutes=30)
		mock_behavioral_session.status = 'active'
		
		# Mock database query
		mock_result = Mock()
		mock_result.scalar_one_or_none.return_value = mock_behavioral_session
		mock_db_session.execute.return_value = mock_result
		
		# Act
		response = client.get(
			f'/api/v1/biometric/behavioral/session/{session_id}/patterns',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['session_id'] == session_id
		assert 'pattern_analysis' in data
		assert data['pattern_analysis']['anomaly_detection']['anomaly_count'] == 0
		assert data['pattern_analysis']['anomaly_detection']['average_confidence'] == 0.92
		assert data['pattern_analysis']['zero_friction_metrics']['friction_score'] == 0.015
		assert data['session_duration_minutes'] == 30
		assert data['monitoring_active'] is True
	
	@patch('capabilities.common.biometric.api.get_db_session')
	@patch('capabilities.common.biometric.api.get_current_tenant_id')
	def test_get_behavioral_patterns_not_found(self, mock_get_tenant, mock_get_db, client, auth_headers):
		"""Test behavioral pattern retrieval for nonexistent session"""
		# Arrange
		session_id = 'nonexistent_behavioral_session'
		mock_get_tenant.return_value = TEST_TENANT_ID
		mock_db_session = AsyncMock()
		mock_get_db.return_value = mock_db_session
		
		# Mock database query returning None
		mock_result = Mock()
		mock_result.scalar_one_or_none.return_value = None
		mock_db_session.execute.return_value = mock_result
		
		# Act
		response = client.get(
			f'/api/v1/biometric/behavioral/session/{session_id}/patterns',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 404
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'not found' in data['error'].lower()

# Analytics API Tests

class TestAnalyticsAPI:
	"""Test suite for analytics dashboard API endpoints"""
	
	@patch('capabilities.common.biometric.api.get_db_session')
	@patch('capabilities.common.biometric.api.get_current_tenant_id')
	def test_get_analytics_dashboard_success(self, mock_get_tenant, mock_get_db, client, auth_headers):
		"""Test successful analytics dashboard data retrieval"""
		# Arrange
		mock_get_tenant.return_value = TEST_TENANT_ID
		mock_db_session = AsyncMock()
		mock_get_db.return_value = mock_db_session
		
		# Mock database query results
		mock_db_session.scalar.side_effect = [
			100,  # total_verifications
			25,   # active_behavioral_sessions
			5     # active_collaborations
		]
		
		# Act
		response = client.get(
			'/api/v1/biometric/analytics/dashboard',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'dashboard_data' in data
		assert data['revolutionary_features_active'] == 10
		
		# Verify dashboard sections
		dashboard = data['dashboard_data']
		assert 'verification_metrics' in dashboard
		assert 'behavioral_analytics' in dashboard
		assert 'collaborative_insights' in dashboard
		assert 'security_intelligence' in dashboard
		assert 'compliance_status' in dashboard
		assert 'revolutionary_features' in dashboard
		
		# Verify specific metrics
		assert dashboard['verification_metrics']['total_verifications'] == 100
		assert dashboard['behavioral_analytics']['active_sessions'] == 25
		assert dashboard['collaborative_insights']['active_collaborations'] == 5
		assert dashboard['compliance_status']['global_compliance_score'] == 0.97
		
		# Verify revolutionary features are all active
		features = dashboard['revolutionary_features']
		assert features['contextual_intelligence'] is True
		assert features['natural_language_queries'] is True
		assert features['predictive_analytics'] is True
		assert features['zero_friction_auth'] is True
	
	@patch('capabilities.common.biometric.api.get_db_session')
	@patch('capabilities.common.biometric.api.get_current_tenant_id')
	def test_get_predictive_analytics_success(self, mock_get_tenant, mock_get_db, client, auth_headers):
		"""Test successful predictive analytics data retrieval"""
		# Arrange
		mock_get_tenant.return_value = TEST_TENANT_ID
		mock_db_session = AsyncMock()
		mock_get_db.return_value = mock_db_session
		
		# Mock recent verifications query
		mock_verifications = [Mock() for _ in range(50)]  # 50 mock verifications
		mock_result = Mock()
		mock_result.scalars.return_value.all.return_value = mock_verifications
		mock_db_session.execute.return_value = mock_result
		
		# Act
		response = client.get(
			'/api/v1/biometric/analytics/predictive',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'predictive_insights' in data
		assert data['data_points_analyzed'] == 50
		assert data['forecast_confidence'] == 0.92
		
		# Verify predictive insights structure
		insights = data['predictive_insights']
		assert 'fraud_trend_forecast' in insights
		assert 'risk_evolution' in insights
		assert 'behavioral_predictions' in insights
		assert 'threat_intelligence' in insights
		
		# Verify fraud forecast
		fraud_forecast = insights['fraud_trend_forecast']
		assert 'current_rate' in fraud_forecast
		assert 'predicted_rate_7d' in fraud_forecast
		assert 'predicted_rate_30d' in fraud_forecast
		assert 'confidence_interval' in fraud_forecast
		assert 'trend_direction' in fraud_forecast

# Security API Tests

class TestSecurityAPI:
	"""Test suite for adaptive security API endpoints"""
	
	def test_get_adaptive_security_status_success(self, client, auth_headers):
		"""Test successful adaptive security status retrieval"""
		# Act
		response = client.get(
			'/api/v1/biometric/security/adaptive/status',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'security_status' in data
		assert data['revolutionary_security_active'] is True
		
		# Verify security status structure
		security_status = data['security_status']
		assert 'current_threat_level' in security_status
		assert 'active_adaptations' in security_status
		assert 'evolution_metrics' in security_status
		assert 'threat_intelligence' in security_status
		assert 'deepfake_detection' in security_status
		assert 'behavioral_security' in security_status
		
		# Verify specific security metrics
		assert security_status['current_threat_level'] == 'low'
		assert security_status['deepfake_detection']['quantum_detection_active'] is True
		assert security_status['deepfake_detection']['detection_accuracy'] == 0.999
		assert security_status['behavioral_security']['continuous_learning_active'] is True

# Compliance API Tests

class TestComplianceAPI:
	"""Test suite for universal compliance API endpoints"""
	
	def test_get_universal_compliance_status_success(self, client, auth_headers):
		"""Test successful universal compliance status retrieval"""
		# Act
		response = client.get(
			'/api/v1/biometric/security/universal/status',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'compliance_status' in data
		assert data['universal_orchestration_active'] is True
		
		# Verify compliance status structure
		compliance_status = data['compliance_status']
		assert 'global_compliance_score' in compliance_status
		assert 'regulatory_frameworks' in compliance_status
		assert 'jurisdiction_coverage' in compliance_status
		assert 'automated_compliance' in compliance_status
		assert 'regulatory_intelligence' in compliance_status
		
		# Verify specific compliance metrics
		assert compliance_status['global_compliance_score'] == 0.97
		assert compliance_status['jurisdiction_coverage']['total_jurisdictions'] == 200
		assert compliance_status['automated_compliance']['automation_rate'] == 1.0
		assert compliance_status['regulatory_intelligence']['monitoring_active'] is True
		
		# Verify regulatory frameworks
		frameworks = compliance_status['regulatory_frameworks']
		assert 'GDPR' in frameworks
		assert 'CCPA' in frameworks
		assert 'BIPA' in frameworks
		assert 'KYC_AML' in frameworks
		
		for framework_name, framework_data in frameworks.items():
			assert 'status' in framework_data
			assert 'score' in framework_data
			assert 'last_audit' in framework_data

# User Management API Tests

class TestUserAPI:
	"""Test suite for user management API endpoints"""
	
	@patch('capabilities.common.biometric.api.get_biometric_service')
	def test_create_user_success(self, mock_get_service, client, auth_headers, sample_user_data):
		"""Test successful user creation via API"""
		# Arrange
		mock_service = AsyncMock()
		mock_get_service.return_value = mock_service
		
		mock_user = Mock()
		mock_user.id = 'test_user_api_123'
		mock_user.external_user_id = sample_user_data['external_user_id']
		mock_user.global_identity_id = 'global_id_456'
		
		mock_service.create_user.return_value = mock_user
		
		# Act
		response = client.post(
			'/api/v1/biometric/users/create',
			data=json.dumps(sample_user_data),
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['user_id'] == 'test_user_api_123'
		assert data['external_user_id'] == sample_user_data['external_user_id']
		assert data['global_identity_id'] == 'global_id_456'
		assert data['contextual_intelligence_initialized'] is True
		assert data['behavioral_profiling_active'] is True
		assert data['universal_compliance_configured'] is True
		assert data['zero_friction_auth_ready'] is True
	
	@patch('capabilities.common.biometric.api.get_biometric_service')
	def test_get_user_profile_success(self, mock_get_service, client, auth_headers):
		"""Test successful user profile retrieval"""
		# Arrange
		user_id = 'test_user_profile_123'
		mock_service = AsyncMock()
		mock_get_service.return_value = mock_service
		
		mock_user = Mock()
		mock_user.id = user_id
		mock_user.external_user_id = 'profile_test_user'
		mock_user.global_identity_id = 'global_profile_456'
		mock_user.behavioral_profile = {'baseline_established': True}
		mock_user.risk_profile = {'risk_factors': ['factor1', 'factor2']}
		mock_user.security_adaptations = {'adaptations': ['adaptation1']}
		mock_user.biometrics = [Mock(), Mock()]  # 2 biometric modalities
		mock_user.verifications = [Mock(), Mock(), Mock()]  # 3 verifications
		mock_user.jurisdiction_compliance = {'GDPR': 'compliant'}
		mock_user.last_activity = datetime.utcnow()
		mock_user.contextual_patterns = {'pattern': 'data'}
		mock_user.threat_intelligence = {'threat': 'data'}
		mock_user.invisible_auth_profile = {'auth': 'data'}
		
		mock_service._get_user_by_id.return_value = mock_user
		
		# Act
		response = client.get(
			f'/api/v1/biometric/users/{user_id}',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['privacy_protection_active'] is True
		
		# Verify user profile structure
		profile = data['user_profile']
		assert profile['id'] == user_id
		assert profile['external_user_id'] == 'profile_test_user'
		assert profile['global_identity_id'] == 'global_profile_456'
		assert profile['biometric_modalities'] == 2
		assert profile['verification_history'] == 3
		
		# Verify contextual intelligence data
		contextual = profile['contextual_intelligence']
		assert contextual['behavioral_baseline_established'] is True
		assert contextual['risk_profile_maturity'] == 2
		assert contextual['adaptation_count'] == 1
		
		# Verify revolutionary features status
		features = profile['revolutionary_features_active']
		assert features['contextual_intelligence'] is True
		assert features['behavioral_profiling'] is True
		assert features['adaptive_security'] is True
		assert features['zero_friction_auth'] is True
		assert features['universal_identity'] is True
	
	@patch('capabilities.common.biometric.api.get_biometric_service')
	def test_get_user_profile_not_found(self, mock_get_service, client, auth_headers):
		"""Test user profile retrieval for nonexistent user"""
		# Arrange
		user_id = 'nonexistent_user_id'
		mock_service = AsyncMock()
		mock_get_service.return_value = mock_service
		mock_service._get_user_by_id.return_value = None
		
		# Act
		response = client.get(
			f'/api/v1/biometric/users/{user_id}',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 404
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'not found' in data['error'].lower()

# Health Check Tests

class TestHealthCheckAPI:
	"""Test suite for API health check endpoint"""
	
	def test_health_check_success(self, client):
		"""Test successful health check"""
		# Act
		response = client.get('/api/v1/biometric/health')
		
		# Assert
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'health_status' in data
		
		# Verify health status structure
		health = data['health_status']
		assert health['api_status'] == 'healthy'
		assert health['database_connection'] == 'active'
		assert 'revolutionary_features' in health
		assert 'performance_metrics' in health
		assert 'market_superiority' in health
		assert health['version'] == '1.0.0'
		
		# Verify revolutionary features are all active
		features = health['revolutionary_features']
		assert all(status == 'active' for status in features.values())
		assert len(features) == 10  # All 10 revolutionary features
		
		# Verify performance metrics
		performance = health['performance_metrics']
		assert performance['average_response_time_ms'] == 150
		assert performance['throughput_rps'] == 1000
		assert performance['accuracy_rate'] == 0.998
		assert performance['uptime_percentage'] == 99.99
		
		# Verify market superiority claims
		superiority = health['market_superiority']
		assert '2x better' in superiority['accuracy_advantage']
		assert '3x faster' in superiority['speed_advantage']
		assert '70% cost reduction' in superiority['cost_advantage']
		assert superiority['unique_features'] == 10

# Error Handling Tests

class TestErrorHandling:
	"""Test suite for API error handling"""
	
	def test_invalid_json_request(self, client, auth_headers):
		"""Test handling of invalid JSON in request"""
		# Act
		response = client.post(
			'/api/v1/biometric/auth/verify',
			data='invalid json data',
			headers=auth_headers
		)
		
		# Assert
		assert response.status_code == 400
		data = json.loads(response.data)
		assert data['success'] is False
		assert 'error' in data
	
	def test_missing_content_type(self, client, auth_headers):
		"""Test handling of missing content type header"""
		# Arrange
		headers_without_content_type = {k: v for k, v in auth_headers.items() if k != 'Content-Type'}
		
		# Act
		response = client.post(
			'/api/v1/biometric/auth/verify',
			data='{"test": "data"}',
			headers=headers_without_content_type
		)
		
		# Assert
		# Should still work or return appropriate error based on Flask configuration
		assert response.status_code in [200, 400, 415]  # Various acceptable responses
	
	def test_internal_server_error_handling(self, client, auth_headers):
		"""Test handling of internal server errors"""
		# This test would need to be implemented with proper mocking
		# to simulate internal server errors
		pass

# Performance Tests

class TestAPIPerformance:
	"""Test suite for API performance benchmarks"""
	
	@pytest.mark.performance
	def test_verification_endpoint_response_time(self, client, auth_headers, sample_verification_request):
		"""Test verification endpoint response time meets requirements"""
		# This test would measure actual response times
		# and verify they meet the <300ms requirement
		pass
	
	@pytest.mark.performance
	def test_natural_language_query_response_time(self, client, auth_headers, sample_nl_query_request):
		"""Test natural language query response time"""
		# This test would measure NL query response times
		# and verify they meet the <200ms requirement
		pass
	
	@pytest.mark.performance
	def test_concurrent_request_handling(self, client, auth_headers):
		"""Test handling of concurrent API requests"""
		# This test would simulate concurrent requests
		# and verify system can handle 1000+ concurrent requests
		pass

# Integration Tests

class TestAPIIntegration:
	"""Test suite for API integration scenarios"""
	
	@pytest.mark.integration
	def test_complete_verification_workflow(self, client, auth_headers, sample_user_data, sample_verification_request):
		"""Test complete end-to-end verification workflow through API"""
		# This test would go through the complete workflow:
		# 1. Create user
		# 2. Start verification
		# 3. Process biometric
		# 4. Complete verification
		# 5. Check status
		# 6. Query via natural language
		pass
	
	@pytest.mark.integration
	def test_collaborative_verification_workflow(self, client, auth_headers):
		"""Test collaborative verification workflow"""
		# This test would simulate:
		# 1. Start collaborative session
		# 2. Multiple participants join
		# 3. Collaborative analysis
		# 4. Consensus building
		# 5. Final decision
		pass
	
	@pytest.mark.integration
	def test_behavioral_monitoring_workflow(self, client, auth_headers, sample_behavioral_session_request):
		"""Test behavioral monitoring workflow"""
		# This test would simulate:
		# 1. Start behavioral session
		# 2. Submit behavioral data
		# 3. Monitor patterns
		# 4. Detect anomalies
		# 5. Adapt authentication
		pass

# Run the tests
if __name__ == '__main__':
	pytest.main([__file__, '-v', '--tb=short'])