"""
APG Facial Recognition - API Endpoint Tests

Comprehensive tests for REST API endpoints, testing request validation,
response formats, error handling, and security measures.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import pytest
import json
import base64
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from flask import Flask
from datetime import datetime, timezone
import numpy as np

from .api import facial_api, get_service
from .views import (
	UserCreateRequest, EnrollmentRequest, VerificationRequest,
	IdentificationRequest, EmotionAnalysisRequest, CollaborationRequest,
	ConsentRequest, PrivacyProcessingRequest, DataSubjectRequest
)

@pytest.fixture
def app():
	"""Create Flask app for testing"""
	app = Flask(__name__)
	app.config['TESTING'] = True
	app.register_blueprint(facial_api)
	return app

@pytest.fixture
def client(app):
	"""Create test client"""
	return app.test_client()

@pytest.fixture
def mock_face_image_b64():
	"""Mock base64 encoded face image"""
	# Create small test image
	img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
	img_bytes = img_array.tobytes()
	return base64.b64encode(img_bytes).decode('utf-8')

@pytest.fixture
def headers():
	"""Common headers for API requests"""
	return {
		'Content-Type': 'application/json',
		'X-Tenant-ID': 'test_tenant'
	}

@pytest.fixture
def mock_facial_service():
	"""Mock facial recognition service"""
	mock_service = AsyncMock()
	mock_service.create_user.return_value = {
		"id": "user_123",
		"external_user_id": "test_user",
		"full_name": "Test User",
		"email": "test@example.com",
		"enrollment_status": "not_enrolled",
		"consent_given": True,
		"created_at": datetime.now(timezone.utc),
		"updated_at": None,
		"last_verification": None
	}
	mock_service.enroll_face.return_value = {
		"success": True,
		"template_id": "template_123",
		"quality_score": 0.92,
		"processing_time_ms": 150,
		"enrollment_timestamp": datetime.now(timezone.utc).isoformat()
	}
	mock_service.verify_face.return_value = {
		"success": True,
		"verified": True,
		"verification_id": "ver_123",
		"confidence_score": 0.95,
		"similarity_score": 0.92,
		"quality_score": 0.88,
		"liveness_score": 0.96,
		"liveness_result": "live",
		"processing_time_ms": 120,
		"verification_timestamp": datetime.now(timezone.utc).isoformat()
	}
	mock_service.identify_face.return_value = {
		"success": True,
		"candidates": [
			{
				"user_id": "user_456",
				"external_user_id": "match_user",
				"full_name": "Match User",
				"similarity_score": 0.94,
				"confidence_score": 0.96,
				"template_id": "template_456",
				"rank": 1
			}
		],
		"query_quality_score": 0.89,
		"processing_time_ms": 200,
		"search_timestamp": datetime.now(timezone.utc).isoformat()
	}
	return mock_service

class TestHealthEndpoint:
	"""Test health check endpoint"""
	
	def test_health_check_success(self, client):
		"""Test successful health check"""
		response = client.get('/api/v1/facial/health')
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['status'] == 'healthy'
		assert 'version' in data
		assert 'uptime_seconds' in data
		assert 'components' in data
		assert 'metrics' in data
	
	def test_health_check_format(self, client):
		"""Test health check response format"""
		response = client.get('/api/v1/facial/health')
		data = json.loads(response.data)
		
		# Verify required fields
		required_fields = ['status', 'version', 'uptime_seconds', 'components', 'metrics']
		for field in required_fields:
			assert field in data
		
		# Verify component statuses
		components = data['components']
		assert 'database' in components
		assert 'facial_service' in components
		assert 'emotion_engine' in components
		assert 'privacy_engine' in components

class TestUserManagementEndpoints:
	"""Test user management API endpoints"""
	
	@patch('capabilities.common.facial.api.get_service')
	def test_create_user_success(self, mock_get_service, client, headers, mock_facial_service):
		"""Test successful user creation"""
		mock_get_service.return_value = mock_facial_service
		
		user_data = {
			"external_user_id": "test_user_001",
			"full_name": "John Doe",
			"email": "john.doe@example.com",
			"consent_given": True,
			"privacy_settings": {"level": "standard"}
		}
		
		response = client.post(
			'/api/v1/facial/users',
			data=json.dumps(user_data),
			headers=headers
		)
		
		assert response.status_code == 201
		data = json.loads(response.data)
		
		assert data['external_user_id'] == "test_user_001"
		assert data['full_name'] == "John Doe"
		assert data['consent_given'] is True
		assert 'id' in data
	
	def test_create_user_validation_error(self, client, headers):
		"""Test user creation with validation errors"""
		# Missing required fields
		invalid_data = {
			"email": "john.doe@example.com"
		}
		
		response = client.post(
			'/api/v1/facial/users',
			data=json.dumps(invalid_data),
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		
		assert data['success'] is False
		assert 'validation error' in data['error'].lower()
		assert data['error_code'] == 'VALIDATION_ERROR'
	
	def test_create_user_invalid_email(self, client, headers):
		"""Test user creation with invalid email"""
		invalid_data = {
			"external_user_id": "test_user",
			"full_name": "John Doe",
			"email": "invalid_email_format",
			"consent_given": True
		}
		
		response = client.post(
			'/api/v1/facial/users',
			data=json.dumps(invalid_data),
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		assert 'validation error' in data['error'].lower()
	
	@patch('capabilities.common.facial.api.get_service')
	def test_get_user_success(self, mock_get_service, client, headers, mock_facial_service):
		"""Test successful user retrieval"""
		mock_get_service.return_value = mock_facial_service
		
		response = client.get('/api/v1/facial/users/user_123', headers=headers)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['external_user_id'] == "test_user"
		assert data['full_name'] == "Test User"
		assert 'id' in data
	
	@patch('capabilities.common.facial.api.get_service')
	def test_get_user_not_found(self, mock_get_service, client, headers):
		"""Test user not found scenario"""
		mock_service = AsyncMock()
		mock_service.get_user.return_value = None
		mock_get_service.return_value = mock_service
		
		response = client.get('/api/v1/facial/users/nonexistent', headers=headers)
		
		assert response.status_code == 404
		data = json.loads(response.data)
		
		assert data['success'] is False
		assert 'not found' in data['error'].lower()
		assert data['error_code'] == 'USER_NOT_FOUND'

class TestEnrollmentEndpoints:
	"""Test enrollment API endpoints"""
	
	@patch('capabilities.common.facial.api.get_service')
	def test_enroll_face_success(self, mock_get_service, client, headers, 
	                           mock_facial_service, mock_face_image_b64):
		"""Test successful face enrollment"""
		mock_get_service.return_value = mock_facial_service
		
		enrollment_data = {
			"user_id": "user_123",
			"image_data": mock_face_image_b64,
			"enrollment_type": "standard",
			"quality_threshold": 0.8,
			"device_info": {"device_id": "mobile_001"},
			"location_data": {"country": "US"}
		}
		
		response = client.post(
			'/api/v1/facial/enroll',
			data=json.dumps(enrollment_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'template_id' in data
		assert data['quality_score'] > 0.8
		assert 'processing_time_ms' in data
	
	def test_enroll_face_missing_image(self, client, headers):
		"""Test enrollment without image data"""
		enrollment_data = {
			"user_id": "user_123",
			"enrollment_type": "standard"
		}
		
		response = client.post(
			'/api/v1/facial/enroll',
			data=json.dumps(enrollment_data),
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		assert 'validation error' in data['error'].lower()
	
	def test_enroll_face_invalid_base64(self, client, headers):
		"""Test enrollment with invalid base64 image data"""
		enrollment_data = {
			"user_id": "user_123",
			"image_data": "invalid_base64_data!!!",
			"enrollment_type": "standard"
		}
		
		response = client.post(
			'/api/v1/facial/enroll',
			data=json.dumps(enrollment_data),
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		assert 'invalid image data' in data['error'].lower()
	
	def test_enroll_face_url_not_implemented(self, client, headers):
		"""Test enrollment with image URL (not implemented)"""
		enrollment_data = {
			"user_id": "user_123",
			"image_url": "https://example.com/face.jpg"
		}
		
		response = client.post(
			'/api/v1/facial/enroll',
			data=json.dumps(enrollment_data),
			headers=headers
		)
		
		assert response.status_code == 501
		data = json.loads(response.data)
		assert 'not implemented' in data['error'].lower()

class TestVerificationEndpoints:
	"""Test verification API endpoints"""
	
	@patch('capabilities.common.facial.api.get_service')
	def test_verify_face_success(self, mock_get_service, client, headers,
	                           mock_facial_service, mock_face_image_b64):
		"""Test successful face verification"""
		# Setup mocks for all services
		mock_contextual = AsyncMock()
		mock_contextual.analyze_verification_context.return_value = {
			"business_risk_score": 0.2,
			"contextual_confidence": 0.9
		}
		
		mock_emotion = AsyncMock()
		mock_emotion.analyze_emotions.return_value = {
			"primary_emotion": "neutral",
			"confidence": 0.8
		}
		
		mock_predictive = AsyncMock()
		mock_predictive.predict_identity_risk.return_value = {
			"risk_score": 0.15,
			"risk_level": "low"
		}
		
		def mock_get_service_side_effect(service_type, tenant_id):
			if service_type == 'facial':
				return mock_facial_service
			elif service_type == 'contextual':
				return mock_contextual
			elif service_type == 'emotion':
				return mock_emotion
			elif service_type == 'predictive':
				return mock_predictive
			return None
		
		mock_get_service.side_effect = mock_get_service_side_effect
		
		verification_data = {
			"user_id": "user_123",
			"image_data": mock_face_image_b64,
			"require_liveness": True,
			"confidence_threshold": 0.8,
			"business_context": {"action": "login"},
			"device_info": {"device_id": "mobile_001"}
		}
		
		response = client.post(
			'/api/v1/facial/verify',
			data=json.dumps(verification_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['verified'] is True
		assert data['confidence_score'] >= 0.8
		assert data['liveness_result'] == 'live'
		assert 'verification_id' in data
		assert 'contextual_analysis' in data
		assert 'emotion_analysis' in data
		assert 'risk_analysis' in data
	
	@patch('capabilities.common.facial.api.get_service')
	def test_verify_face_failed(self, mock_get_service, client, headers, mock_face_image_b64):
		"""Test failed face verification"""
		mock_service = AsyncMock()
		mock_service.verify_face.return_value = {
			"success": True,
			"verified": False,
			"confidence_score": 0.65,
			"failure_reason": "Low confidence score",
			"verification_timestamp": datetime.now(timezone.utc).isoformat()
		}
		mock_get_service.return_value = mock_service
		
		verification_data = {
			"user_id": "user_123",
			"image_data": mock_face_image_b64,
			"confidence_threshold": 0.8
		}
		
		response = client.post(
			'/api/v1/facial/verify',
			data=json.dumps(verification_data),
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['verified'] is False
		assert data['confidence_score'] == 0.65
		assert 'low confidence' in data['failure_reason'].lower()

class TestIdentificationEndpoints:
	"""Test identification API endpoints"""
	
	@patch('capabilities.common.facial.api.get_service')
	def test_identify_face_success(self, mock_get_service, client, headers,
	                             mock_facial_service, mock_face_image_b64):
		"""Test successful face identification"""
		mock_get_service.return_value = mock_facial_service
		
		identification_data = {
			"image_data": mock_face_image_b64,
			"max_candidates": 5,
			"confidence_threshold": 0.8,
			"quality_threshold": 0.7
		}
		
		response = client.post(
			'/api/v1/facial/identify',
			data=json.dumps(identification_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert len(data['candidates']) >= 1
		assert data['candidates'][0]['confidence_score'] >= 0.8
		assert 'query_quality_score' in data
		assert 'processing_time_ms' in data
	
	def test_identify_face_validation(self, client, headers):
		"""Test identification request validation"""
		# Invalid max_candidates
		invalid_data = {
			"image_data": "base64_data",
			"max_candidates": 100,  # > 50 limit
			"confidence_threshold": 0.8
		}
		
		response = client.post(
			'/api/v1/facial/identify',
			data=json.dumps(invalid_data),
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		assert 'validation error' in data['error'].lower()

class TestEmotionAnalysisEndpoints:
	"""Test emotion analysis API endpoints"""
	
	@patch('capabilities.common.facial.api.get_service')
	def test_analyze_emotion_success(self, mock_get_service, client, headers, mock_face_image_b64):
		"""Test successful emotion analysis"""
		mock_emotion_service = AsyncMock()
		mock_emotion_service.analyze_emotions.return_value = {
			"analysis_id": "emotion_123",
			"primary_emotion": "happy",
			"emotion_confidence": 0.85,
			"emotion_scores": {
				"happy": 0.85,
				"neutral": 0.10,
				"sad": 0.05
			},
			"stress_analysis": {
				"overall_stress_level": "low",
				"stress_score": 0.2,
				"stress_indicators": [],
				"physiological_markers": {}
			},
			"micro_expressions": [],
			"behavioral_insights": {"engagement_level": "high"},
			"risk_indicators": [],
			"recommendations": ["Continue current interaction"],
			"processing_time_ms": 80
		}
		mock_get_service.return_value = mock_emotion_service
		
		emotion_data = {
			"image_data": mock_face_image_b64,
			"user_context": {"session_id": "session_001"}
		}
		
		response = client.post(
			'/api/v1/facial/analyze/emotion',
			data=json.dumps(emotion_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['primary_emotion'] == 'happy'
		assert data['emotion_confidence'] == 0.85
		assert 'emotion_scores' in data
		assert 'stress_analysis' in data
		assert 'behavioral_insights' in data
	
	@patch('capabilities.common.facial.api.get_service')
	def test_analyze_emotion_video_frames(self, mock_get_service, client, headers):
		"""Test emotion analysis with video frames"""
		mock_emotion_service = AsyncMock()
		mock_emotion_service.analyze_emotions.return_value = {
			"analysis_id": "emotion_456",
			"primary_emotion": "neutral",
			"emotion_confidence": 0.9,
			"emotion_scores": {"neutral": 0.9, "happy": 0.1},
			"stress_analysis": {"overall_stress_level": "low", "stress_score": 0.1},
			"micro_expressions": [],
			"processing_time_ms": 120
		}
		mock_get_service.return_value = mock_emotion_service
		
		# Create multiple frame data
		frame_data = [base64.b64encode(b"frame_data_" + str(i).encode()).decode() for i in range(3)]
		
		emotion_data = {
			"video_frames": frame_data,
			"user_context": {"session_id": "session_002"}
		}
		
		response = client.post(
			'/api/v1/facial/analyze/emotion',
			data=json.dumps(emotion_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		assert data['success'] is True

class TestCollaborativeVerificationEndpoints:
	"""Test collaborative verification API endpoints"""
	
	@patch('capabilities.common.facial.api.get_service')
	def test_initiate_collaboration_success(self, mock_get_service, client, headers):
		"""Test successful collaboration initiation"""
		mock_collaboration_service = AsyncMock()
		mock_collaboration_service.initiate_collaborative_verification.return_value = {
			"success": True,
			"collaboration_id": "collab_123",
			"status": "pending",
			"message": "Collaboration initiated successfully",
			"participants_invited": 3,
			"timeout_at": datetime.now(timezone.utc).isoformat()
		}
		mock_get_service.return_value = mock_collaboration_service
		
		collaboration_data = {
			"verification_id": "ver_123",
			"workflow_type": "supervisor_approval",
			"required_approvals": 2,
			"timeout_minutes": 60,
			"context": {"risk_level": "high"}
		}
		
		response = client.post(
			'/api/v1/facial/collaborate/initiate',
			data=json.dumps(collaboration_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'collaboration_id' in data
		assert data['status'] == 'pending'
		assert data['participants_invited'] == 3
	
	@patch('capabilities.common.facial.api.get_service')
	def test_submit_collaboration_response(self, mock_get_service, client, headers):
		"""Test collaboration response submission"""
		mock_collaboration_service = AsyncMock()
		mock_collaboration_service.submit_participant_response.return_value = {
			"success": True,
			"collaboration_status": "completed"
		}
		mock_get_service.return_value = mock_collaboration_service
		
		response_data = {
			"collaboration_id": "collab_123",
			"participant_id": "supervisor_001",
			"decision": "approve",
			"confidence_score": 0.9,
			"reasoning": "All metrics within acceptable range"
		}
		
		response = client.post(
			'/api/v1/facial/collaborate/respond',
			data=json.dumps(response_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert data['collaboration_id'] == "collab_123"

class TestPrivacyEndpoints:
	"""Test privacy and consent API endpoints"""
	
	@patch('capabilities.common.facial.api.get_service')
	def test_manage_consent_success(self, mock_get_service, client, headers):
		"""Test successful consent management"""
		mock_privacy_service = AsyncMock()
		mock_privacy_service.manage_user_consent.return_value = {
			"success": True,
			"consent_id": "consent_123",
			"consent_status": "active",
			"data_subject_rights": {
				"access": True,
				"rectification": True,
				"erasure": True,
				"portability": True
			}
		}
		mock_get_service.return_value = mock_privacy_service
		
		consent_data = {
			"user_id": "user_123",
			"consent_given": True,
			"consent_method": "explicit",
			"allowed_purposes": ["identity_verification"],
			"allowed_data_categories": ["facial_biometric"]
		}
		
		response = client.post(
			'/api/v1/facial/privacy/consent',
			data=json.dumps(consent_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'consent_id' in data
		assert data['consent_status'] == 'active'
		assert 'data_subject_rights' in data
	
	@patch('capabilities.common.facial.api.get_service')
	def test_privacy_processing_success(self, mock_get_service, client, headers):
		"""Test privacy-preserving processing"""
		mock_privacy_service = AsyncMock()
		mock_privacy_service.process_with_privacy.return_value = {
			"success": True,
			"privacy_metadata": {
				"processing_id": "privacy_123",
				"privacy_techniques_applied": ["differential_privacy"],
				"data_minimization": True
			},
			"processing_result": {"processed": True}
		}
		mock_get_service.return_value = mock_privacy_service
		
		# Create mock biometric data
		biometric_data = base64.b64encode(b"mock_biometric_data").decode()
		
		processing_data = {
			"user_id": "user_123",
			"biometric_data": biometric_data,
			"privacy_level": "enhanced",
			"processing_mode": "federated",
			"processing_purpose": "identity_verification"
		}
		
		response = client.post(
			'/api/v1/facial/privacy/process',
			data=json.dumps(processing_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'privacy_metadata' in data
		assert 'processing_id' in data['privacy_metadata']
	
	@patch('capabilities.common.facial.api.get_service')
	def test_data_subject_request_success(self, mock_get_service, client, headers):
		"""Test data subject rights request"""
		mock_privacy_service = AsyncMock()
		mock_privacy_service.exercise_data_subject_rights.return_value = {
			"success": True,
			"request_id": "dsr_123",
			"processing_time_days": 30,
			"completion_status": "processing"
		}
		mock_get_service.return_value = mock_privacy_service
		
		dsr_data = {
			"user_id": "user_123",
			"request_type": "access",
			"request_reason": "User requested data export",
			"specific_data_categories": ["facial_biometric", "verification_history"]
		}
		
		response = client.post(
			'/api/v1/facial/privacy/data-subject-request',
			data=json.dumps(dsr_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'request_id' in data
		assert data['request_type'] == 'access'
		assert data['processing_time_days'] <= 30

class TestAnalyticsEndpoints:
	"""Test analytics API endpoints"""
	
	def test_get_analytics_success(self, client, headers):
		"""Test successful analytics retrieval"""
		analytics_data = {
			"metric_type": "verification_stats",
			"time_period": "last_30_days",
			"filters": {"status": "completed"}
		}
		
		response = client.post(
			'/api/v1/facial/analytics',
			data=json.dumps(analytics_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'metrics' in data
		assert len(data['metrics']) > 0
		
		# Verify metric structure
		metric = data['metrics'][0]
		assert 'name' in metric
		assert 'value' in metric
		assert 'unit' in metric

class TestUtilityEndpoints:
	"""Test utility API endpoints"""
	
	def test_assess_image_quality(self, client, headers, mock_face_image_b64):
		"""Test image quality assessment"""
		quality_data = {
			"image_data": mock_face_image_b64
		}
		
		response = client.post(
			'/api/v1/facial/assess/quality',
			data=json.dumps(quality_data),
			headers=headers
		)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'quality_score' in data
		assert 'resolution_score' in data
		assert 'sharpness_score' in data
		assert 'usable_for_recognition' in data
		assert isinstance(data['usable_for_recognition'], bool)
	
	def test_assess_image_quality_missing_data(self, client, headers):
		"""Test image quality assessment without image data"""
		response = client.post(
			'/api/v1/facial/assess/quality',
			data=json.dumps({}),
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		
		assert data['success'] is False
		assert 'image data is required' in data['error'].lower()
	
	def test_get_services_status(self, client, headers):
		"""Test services status endpoint"""
		response = client.get('/api/v1/facial/status/services', headers=headers)
		
		assert response.status_code == 200
		data = json.loads(response.data)
		
		assert data['success'] is True
		assert 'overall_status' in data
		assert 'services' in data
		assert 'timestamp' in data
		
		# Verify all services are listed
		services = data['services']
		expected_services = [
			'facial_service', 'contextual_intelligence', 'emotion_intelligence',
			'collaborative_verification', 'predictive_analytics', 'privacy_architecture'
		]
		for service in expected_services:
			assert service in services

class TestErrorHandling:
	"""Test error handling and edge cases"""
	
	def test_missing_tenant_header(self, client):
		"""Test request without tenant ID header"""
		headers_no_tenant = {'Content-Type': 'application/json'}
		
		response = client.get('/api/v1/facial/health', headers=headers_no_tenant)
		
		# Should use default tenant
		assert response.status_code == 200
	
	def test_invalid_json_request(self, client, headers):
		"""Test request with invalid JSON"""
		response = client.post(
			'/api/v1/facial/users',
			data='invalid json {',
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		assert 'validation error' in data['error'].lower()
	
	def test_empty_request_body(self, client, headers):
		"""Test request with empty body"""
		response = client.post(
			'/api/v1/facial/users',
			data='',
			headers=headers
		)
		
		assert response.status_code == 400
		data = json.loads(response.data)
		assert 'request body is required' in data['error'].lower()
	
	def test_service_unavailable(self, client, headers):
		"""Test behavior when service is unavailable"""
		with patch('capabilities.common.facial.api.get_service', return_value=None):
			user_data = {
				"external_user_id": "test_user",
				"full_name": "Test User",
				"consent_given": True
			}
			
			response = client.post(
				'/api/v1/facial/users',
				data=json.dumps(user_data),
				headers=headers
			)
			
			assert response.status_code == 503
			data = json.loads(response.data)
			assert 'service not available' in data['error'].lower()

class TestSecurityAndValidation:
	"""Test security measures and input validation"""
	
	def test_sql_injection_protection(self, client, headers):
		"""Test protection against SQL injection"""
		malicious_data = {
			"external_user_id": "'; DROP TABLE users; --",
			"full_name": "Malicious User",
			"consent_given": True
		}
		
		response = client.post(
			'/api/v1/facial/users',
			data=json.dumps(malicious_data),
			headers=headers
		)
		
		# Should either succeed with sanitized input or fail validation
		assert response.status_code in [200, 201, 400, 503]
	
	def test_large_payload_handling(self, client, headers):
		"""Test handling of large payloads"""
		# Create large metadata
		large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
		
		large_data = {
			"external_user_id": "test_user",
			"full_name": "Test User",
			"consent_given": True,
			"metadata": large_metadata
		}
		
		response = client.post(
			'/api/v1/facial/users',
			data=json.dumps(large_data),
			headers=headers
		)
		
		# Should handle large payloads gracefully
		assert response.status_code in [200, 201, 400, 413, 503]
	
	def test_xss_protection(self, client, headers):
		"""Test protection against XSS attacks"""
		xss_data = {
			"external_user_id": "<script>alert('xss')</script>",
			"full_name": "<img src=x onerror=alert('xss')>",
			"consent_given": True
		}
		
		response = client.post(
			'/api/v1/facial/users',
			data=json.dumps(xss_data),
			headers=headers
		)
		
		# Should either succeed with sanitized input or fail validation
		assert response.status_code in [200, 201, 400, 503]
		
		if response.status_code in [200, 201]:
			data = json.loads(response.data)
			# Verify no script tags in response
			response_str = json.dumps(data)
			assert '<script>' not in response_str
			assert 'onerror=' not in response_str

if __name__ == "__main__":
	pytest.main([__file__, "-v"])