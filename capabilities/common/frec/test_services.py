"""
APG Facial Recognition - Service Integration Tests

Comprehensive integration tests for facial recognition services,
testing the full workflow from enrollment to verification.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import json
import base64

from .service import FacialRecognitionService
from .database import FacialDatabaseService
from .encryption import TemplateEncryptionService
from .face_engine import FaceProcessingEngine
from .liveness_engine import LivenessDetectionEngine
from .contextual_intelligence import ContextualIntelligenceEngine
from .emotion_intelligence import EmotionIntelligenceEngine
from .collaborative_verification import CollaborativeVerificationEngine
from .predictive_analytics import PredictiveAnalyticsEngine
from .privacy_architecture import PrivacyArchitectureEngine

@pytest.fixture
def mock_face_image():
	"""Mock face image as numpy array"""
	# Create a simple 224x224x3 RGB image
	return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

@pytest.fixture
def mock_video_frames():
	"""Mock video frames for liveness detection"""
	# Create 5 frames of 224x224x3 RGB images
	return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]

@pytest.fixture
async def facial_service():
	"""Create facial recognition service with mocked dependencies"""
	service = FacialRecognitionService(
		database_url="sqlite:///:memory:",
		encryption_key="test_key_32_characters_long_123",
		tenant_id="test_tenant"
	)
	await service.initialize()
	return service

@pytest.fixture
def mock_database_service():
	"""Mock database service"""
	mock_db = AsyncMock(spec=FacialDatabaseService)
	mock_db.create_user.return_value = {
		"id": "user_123",
		"external_user_id": "test_user",
		"full_name": "Test User",
		"enrollment_status": "not_enrolled",
		"consent_given": True,
		"created_at": datetime.now(timezone.utc),
		"updated_at": None,
		"last_verification": None
	}
	mock_db.get_user.return_value = mock_db.create_user.return_value
	mock_db.store_template.return_value = "template_123"
	mock_db.get_user_templates.return_value = [
		{
			"id": "template_123",
			"quality_score": Decimal("0.95"),
			"encrypted_template": b"mock_encrypted_template"
		}
	]
	return mock_db

@pytest.fixture
def mock_encryption_service():
	"""Mock encryption service"""
	mock_encryption = Mock(spec=TemplateEncryptionService)
	mock_encryption.encrypt_template.return_value = b"encrypted_template_data"
	mock_encryption.decrypt_template.return_value = np.random.rand(512).astype(np.float32)
	return mock_encryption

@pytest.fixture
def mock_face_engine():
	"""Mock face processing engine"""
	mock_engine = Mock(spec=FaceProcessingEngine)
	mock_engine.detect_faces.return_value = [
		{
			"bbox": [50, 50, 150, 150],
			"confidence": 0.99,
			"landmarks": np.random.rand(68, 2)
		}
	]
	mock_engine.extract_features.return_value = {
		"features": np.random.rand(512).astype(np.float32),
		"quality_score": 0.92,
		"pose_angles": {"yaw": 5.2, "pitch": -2.1, "roll": 1.8}
	}
	mock_engine.compare_features.return_value = {
		"similarity_score": 0.88,
		"confidence_score": 0.95
	}
	return mock_engine

@pytest.fixture
def mock_liveness_engine():
	"""Mock liveness detection engine"""
	mock_liveness = Mock(spec=LivenessDetectionEngine)
	mock_liveness.detect_liveness.return_value = {
		"is_live": True,
		"liveness_score": 0.96,
		"confidence": 0.98,
		"method": "active_challenge",
		"challenge_passed": True,
		"spoof_detection": {"probability": 0.02, "type": None}
	}
	return mock_liveness

class TestFacialRecognitionService:
	"""Test main facial recognition service"""
	
	@pytest.mark.asyncio
	async def test_create_user(self, facial_service, mock_database_service):
		"""Test user creation"""
		with patch.object(facial_service, 'database_service', mock_database_service):
			user_data = {
				"external_user_id": "test_user_001",
				"full_name": "John Doe",
				"email": "john.doe@example.com",
				"consent_given": True
			}
			
			result = await facial_service.create_user(user_data)
			
			assert result is not None
			assert result["external_user_id"] == "test_user_001"
			assert result["full_name"] == "John Doe"
			mock_database_service.create_user.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_enroll_face_success(self, facial_service, mock_face_image, 
	                                  mock_database_service, mock_encryption_service, 
	                                  mock_face_engine, mock_liveness_engine):
		"""Test successful face enrollment"""
		# Setup mocks
		facial_service.database_service = mock_database_service
		facial_service.encryption_service = mock_encryption_service
		facial_service.face_engine = mock_face_engine
		facial_service.liveness_engine = mock_liveness_engine
		
		enrollment_metadata = {
			"device_info": {"device_id": "mobile_001"},
			"location_data": {"country": "US"},
			"enrollment_type": "standard"
		}
		
		result = await facial_service.enroll_face(
			"user_123", 
			mock_face_image, 
			enrollment_metadata
		)
		
		assert result["success"] is True
		assert "template_id" in result
		assert result["quality_score"] > 0.8
		assert "processing_time_ms" in result
		
		# Verify method calls
		mock_face_engine.detect_faces.assert_called_once()
		mock_face_engine.extract_features.assert_called_once()
		mock_encryption_service.encrypt_template.assert_called_once()
		mock_database_service.store_template.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_enroll_face_no_face_detected(self, facial_service, mock_face_image,
	                                           mock_database_service, mock_face_engine):
		"""Test enrollment when no face is detected"""
		# Setup mocks
		facial_service.database_service = mock_database_service
		facial_service.face_engine = mock_face_engine
		
		# Mock no face detection
		mock_face_engine.detect_faces.return_value = []
		
		result = await facial_service.enroll_face("user_123", mock_face_image, {})
		
		assert result["success"] is False
		assert "no face detected" in result["error"].lower()
	
	@pytest.mark.asyncio
	async def test_verify_face_success(self, facial_service, mock_face_image,
	                                  mock_database_service, mock_encryption_service,
	                                  mock_face_engine, mock_liveness_engine):
		"""Test successful face verification"""
		# Setup mocks
		facial_service.database_service = mock_database_service
		facial_service.encryption_service = mock_encryption_service
		facial_service.face_engine = mock_face_engine
		facial_service.liveness_engine = mock_liveness_engine
		
		verification_config = {
			"require_liveness": True,
			"confidence_threshold": 0.8,
			"business_context": {"action": "login"}
		}
		
		result = await facial_service.verify_face(
			"user_123",
			mock_face_image,
			verification_config
		)
		
		assert result["success"] is True
		assert result["verified"] is True
		assert result["confidence_score"] >= 0.8
		assert result["liveness_result"] == "live"
		assert "verification_id" in result
		
		# Verify method calls
		mock_face_engine.detect_faces.assert_called_once()
		mock_face_engine.extract_features.assert_called_once()
		mock_face_engine.compare_features.assert_called_once()
		mock_liveness_engine.detect_liveness.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_verify_face_low_confidence(self, facial_service, mock_face_image,
	                                         mock_database_service, mock_face_engine):
		"""Test verification with low confidence score"""
		# Setup mocks
		facial_service.database_service = mock_database_service
		facial_service.face_engine = mock_face_engine
		
		# Mock low confidence comparison
		mock_face_engine.compare_features.return_value = {
			"similarity_score": 0.65,
			"confidence_score": 0.70
		}
		
		verification_config = {"confidence_threshold": 0.8}
		
		result = await facial_service.verify_face(
			"user_123",
			mock_face_image,
			verification_config
		)
		
		assert result["success"] is True
		assert result["verified"] is False
		assert result["confidence_score"] == 0.70
		assert "low confidence" in result["failure_reason"].lower()
	
	@pytest.mark.asyncio
	async def test_identify_face_success(self, facial_service, mock_face_image,
	                                    mock_database_service, mock_face_engine):
		"""Test successful face identification (1:N matching)"""
		# Setup mocks
		facial_service.database_service = mock_database_service
		facial_service.face_engine = mock_face_engine
		
		# Mock multiple user templates
		mock_database_service.get_all_templates.return_value = [
			{
				"id": "template_1",
				"user_id": "user_1",
				"external_user_id": "ext_user_1",
				"full_name": "User One",
				"encrypted_template": b"template_1_data"
			},
			{
				"id": "template_2", 
				"user_id": "user_2",
				"external_user_id": "ext_user_2",
				"full_name": "User Two",
				"encrypted_template": b"template_2_data"
			}
		]
		
		# Mock comparison results
		def mock_compare_side_effect(features1, features2):
			if np.array_equal(features2, np.ones(512)):  # Mock template_1
				return {"similarity_score": 0.92, "confidence_score": 0.95}
			else:  # Mock template_2
				return {"similarity_score": 0.65, "confidence_score": 0.70}
		
		mock_face_engine.compare_features.side_effect = mock_compare_side_effect
		
		identification_config = {
			"max_candidates": 5,
			"confidence_threshold": 0.8
		}
		
		result = await facial_service.identify_face(mock_face_image, identification_config)
		
		assert result["success"] is True
		assert len(result["candidates"]) >= 1
		assert result["candidates"][0]["confidence_score"] >= 0.8
		assert result["candidates"][0]["external_user_id"] == "ext_user_1"

class TestContextualIntelligenceEngine:
	"""Test contextual intelligence engine"""
	
	@pytest.mark.asyncio
	async def test_analyze_verification_context(self):
		"""Test contextual analysis of verification"""
		engine = ContextualIntelligenceEngine("test_tenant")
		
		verification_context = {
			"user_context": {
				"user_id": "user_123",
				"historical_patterns": {
					"typical_login_times": ["09:00", "13:00", "17:00"],
					"common_locations": ["office", "home"],
					"usual_devices": ["mobile_001", "laptop_002"]
				}
			},
			"primary_result": {
				"verified": True,
				"confidence_score": 0.88
			},
			"business_context": {
				"action": "high_value_transaction",
				"transaction_amount": 10000,
				"risk_category": "high"
			},
			"device_context": {
				"device_id": "unknown_device_003",
				"device_type": "mobile",
				"os": "Android",
				"location": {"country": "Unknown"}
			},
			"temporal_context": {
				"current_time": "03:30",
				"timezone": "UTC",
				"day_of_week": "Sunday"
			}
		}
		
		result = await engine.analyze_verification_context(verification_context)
		
		assert "business_risk_score" in result
		assert "contextual_confidence" in result
		assert "risk_factors" in result
		assert "recommendations" in result
		
		# Should flag unusual time and unknown device
		risk_factors = result["risk_factors"]
		assert any("unusual time" in factor.lower() for factor in risk_factors)
		assert any("unknown device" in factor.lower() for factor in risk_factors)
	
	@pytest.mark.asyncio
	async def test_learn_user_patterns(self):
		"""Test learning user behavioral patterns"""
		engine = ContextualIntelligenceEngine("test_tenant")
		
		# Simulate historical verification data
		verification_history = [
			{
				"timestamp": "2025-01-15T09:15:00Z",
				"device_id": "mobile_001",
				"location": {"city": "San Francisco", "country": "US"},
				"success": True
			},
			{
				"timestamp": "2025-01-15T13:30:00Z", 
				"device_id": "laptop_002",
				"location": {"city": "San Francisco", "country": "US"},
				"success": True
			},
			{
				"timestamp": "2025-01-16T09:10:00Z",
				"device_id": "mobile_001", 
				"location": {"city": "San Francisco", "country": "US"},
				"success": True
			}
		]
		
		patterns = await engine.learn_user_patterns("user_123", verification_history)
		
		assert "temporal_patterns" in patterns
		assert "device_patterns" in patterns
		assert "location_patterns" in patterns
		assert "success_patterns" in patterns
		
		# Should identify common login times
		temporal = patterns["temporal_patterns"]
		assert "peak_hours" in temporal
		assert 9 in temporal["peak_hours"] or "09" in str(temporal["peak_hours"])

class TestEmotionIntelligenceEngine:
	"""Test emotion intelligence engine"""
	
	@pytest.mark.asyncio
	async def test_analyze_emotions(self, mock_video_frames):
		"""Test emotion analysis from video frames"""
		engine = EmotionIntelligenceEngine("test_tenant")
		
		user_context = {
			"user_id": "user_123",
			"session_id": "session_001",
			"verification_type": "high_security"
		}
		
		result = await engine.analyze_emotions(mock_video_frames, user_context)
		
		assert "analysis_id" in result
		assert "primary_emotion" in result
		assert "emotion_confidence" in result
		assert "emotion_scores" in result
		assert "stress_analysis" in result
		assert "micro_expressions" in result
		assert "behavioral_insights" in result
		
		# Verify emotion scores sum to approximately 1.0
		emotion_scores = result["emotion_scores"]
		total_score = sum(emotion_scores.values())
		assert 0.9 <= total_score <= 1.1
	
	@pytest.mark.asyncio
	async def test_detect_stress_indicators(self, mock_video_frames):
		"""Test stress detection from facial analysis"""
		engine = EmotionIntelligenceEngine("test_tenant")
		
		# Mock stress indicators in frames
		with patch.object(engine, '_analyze_physiological_markers') as mock_physio:
			mock_physio.return_value = {
				"heart_rate_variability": 0.85,  # High stress
				"micro_tremors": 0.6,
				"eye_movement_patterns": 0.7,
				"facial_muscle_tension": 0.8
			}
			
			result = await engine.analyze_emotions(mock_video_frames, {})
			
			stress_analysis = result["stress_analysis"]
			assert stress_analysis["overall_stress_level"] in ["low", "medium", "high"]
			assert "stress_indicators" in stress_analysis
			assert "physiological_markers" in stress_analysis

class TestCollaborativeVerificationEngine:
	"""Test collaborative verification engine"""
	
	@pytest.mark.asyncio
	async def test_initiate_collaborative_verification(self):
		"""Test initiating collaborative verification workflow"""
		engine = CollaborativeVerificationEngine("test_tenant")
		
		verification_request = {
			"verification_id": "ver_123",
			"verification_type": "supervisor_approval",
			"context": {
				"risk_level": "high",
				"transaction_amount": 50000,
				"user_anomalies": ["unusual_location", "new_device"]
			},
			"metadata": {
				"business_unit": "finance",
				"compliance_required": True
			}
		}
		
		result = await engine.initiate_collaborative_verification(verification_request)
		
		assert result["success"] is True
		assert "collaboration_id" in result
		assert "participants_invited" in result
		assert "timeout_at" in result
		assert result["status"] == "pending"
		
		# Should invite appropriate participants for high-risk verification
		assert result["participants_invited"] > 0
	
	@pytest.mark.asyncio
	async def test_submit_participant_response(self):
		"""Test participant response submission"""
		engine = CollaborativeVerificationEngine("test_tenant")
		
		# First initiate collaboration
		verification_request = {
			"verification_id": "ver_123",
			"verification_type": "consensus_review",
			"context": {"risk_level": "medium"}
		}
		
		init_result = await engine.initiate_collaborative_verification(verification_request)
		collaboration_id = init_result["collaboration_id"]
		
		# Submit participant response
		participant_response = {
			"collaboration_id": collaboration_id,
			"participant_id": "supervisor_001",
			"decision": "approve",
			"confidence_score": 0.9,
			"reasoning": "All verification metrics within acceptable ranges",
			"additional_evidence": {"reviewed_history": True}
		}
		
		result = await engine.submit_participant_response(participant_response)
		
		assert result["success"] is True
		assert "collaboration_status" in result
		
		# Should update collaboration status
		collaboration_status = await engine.get_collaboration_status(collaboration_id)
		assert collaboration_status["responses_received"] >= 1

class TestPredictiveAnalyticsEngine:
	"""Test predictive analytics engine"""
	
	@pytest.mark.asyncio
	async def test_predict_identity_risk(self):
		"""Test identity risk prediction"""
		engine = PredictiveAnalyticsEngine("test_tenant")
		
		risk_context = {
			"primary_result": {
				"verified": True,
				"confidence_score": 0.82,
				"similarity_score": 0.79
			},
			"user_context": {
				"user_id": "user_123",
				"enrollment_age_days": 30,
				"verification_history_count": 15,
				"recent_failures": 1
			},
			"business_context": {
				"action": "account_access",
				"risk_category": "medium",
				"value_at_risk": 5000
			},
			"device_context": {
				"device_fingerprint": "unknown",
				"device_reputation": 0.6,
				"location_risk": 0.3
			}
		}
		
		result = await engine.predict_identity_risk(risk_context)
		
		assert "risk_score" in result
		assert "risk_level" in result
		assert "confidence_interval" in result
		assert "contributing_factors" in result
		assert "recommendations" in result
		
		# Risk score should be between 0 and 1
		assert 0 <= result["risk_score"] <= 1
		assert result["risk_level"] in ["low", "medium", "high", "critical"]
	
	@pytest.mark.asyncio
	async def test_predict_verification_success(self):
		"""Test verification success prediction"""
		engine = PredictiveAnalyticsEngine("test_tenant")
		
		prediction_context = {
			"user_profile": {
				"enrollment_quality": 0.95,
				"historical_success_rate": 0.96,
				"template_age_days": 60
			},
			"environmental_factors": {
				"lighting_quality": 0.8,
				"image_quality": 0.85,
				"pose_variation": 0.1
			},
			"system_context": {
				"current_load": 0.3,
				"algorithm_version": "1.0.0",
				"performance_baseline": 0.94
			}
		}
		
		result = await engine.predict_verification_success(prediction_context)
		
		assert "predicted_success_probability" in result
		assert "confidence_score" in result
		assert "key_factors" in result
		assert "optimization_suggestions" in result
		
		# Probability should be between 0 and 1
		assert 0 <= result["predicted_success_probability"] <= 1

class TestPrivacyArchitectureEngine:
	"""Test privacy architecture engine"""
	
	@pytest.mark.asyncio
	async def test_manage_user_consent(self):
		"""Test user consent management"""
		engine = PrivacyArchitectureEngine("test_tenant")
		
		consent_data = {
			"user_id": "user_123",
			"consent_given": True,
			"consent_method": "explicit",
			"allowed_purposes": ["identity_verification", "fraud_prevention"],
			"allowed_data_categories": ["facial_biometric", "device_fingerprint"],
			"granular_control": {
				"analytics": True,
				"research": False,
				"marketing": False
			},
			"legal_basis": "consent"
		}
		
		result = await engine.manage_user_consent("user_123", consent_data)
		
		assert result["success"] is True
		assert "consent_id" in result
		assert "consent_status" in result
		assert "data_subject_rights" in result
		
		# Should provide data subject rights information
		rights = result["data_subject_rights"]
		assert "access" in rights
		assert "rectification" in rights
		assert "erasure" in rights
		assert "portability" in rights
	
	@pytest.mark.asyncio
	async def test_process_with_privacy(self):
		"""Test privacy-preserving biometric processing"""
		engine = PrivacyArchitectureEngine("test_tenant")
		
		# Mock biometric data
		biometric_data = np.random.rand(512).astype(np.float32).tobytes()
		
		processing_config = {
			"user_id": "user_123",
			"privacy_level": "enhanced",
			"processing_mode": "federated",
			"processing_purpose": "identity_verification",
			"data_categories": ["facial_biometric"],
			"retention_policy": "short_term",
			"legal_basis": "consent"
		}
		
		result = await engine.process_with_privacy(biometric_data, processing_config)
		
		assert result["success"] is True
		assert "privacy_metadata" in result
		assert "processing_id" in result["privacy_metadata"]
		
		# Should apply privacy-preserving techniques
		privacy_metadata = result["privacy_metadata"]
		assert "privacy_techniques_applied" in privacy_metadata
		assert "data_minimization" in privacy_metadata
	
	@pytest.mark.asyncio
	async def test_exercise_data_subject_rights(self):
		"""Test data subject rights exercise"""
		engine = PrivacyArchitectureEngine("test_tenant")
		
		# Test data access request
		access_request = {
			"user_id": "user_123",
			"request_type": "access",
			"specific_data_categories": ["facial_biometric", "verification_history"],
			"verification_method": "identity_verification"
		}
		
		result = await engine.exercise_data_subject_rights(
			"user_123", 
			"access", 
			access_request
		)
		
		assert result["success"] is True
		assert "request_id" in result
		assert "processing_time_days" in result
		assert result["processing_time_days"] <= 30  # GDPR compliance
		
		# For access requests, should provide data export
		if "data_export" in result:
			assert isinstance(result["data_export"], dict)

class TestIntegrationWorkflows:
	"""Test complete end-to-end workflows"""
	
	@pytest.mark.asyncio
	async def test_complete_enrollment_workflow(self, facial_service, mock_face_image):
		"""Test complete enrollment workflow"""
		# Mock all dependencies
		with patch.multiple(
			facial_service,
			database_service=AsyncMock(),
			encryption_service=Mock(),
			face_engine=Mock(),
			liveness_engine=Mock()
		):
			# Setup mocks
			facial_service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			facial_service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.95
			}
			facial_service.encryption_service.encrypt_template.return_value = b"encrypted"
			facial_service.database_service.store_template.return_value = "template_123"
			
			# Test workflow
			user_data = {"external_user_id": "test_user", "full_name": "Test User"}
			user = await facial_service.create_user(user_data)
			
			enrollment_result = await facial_service.enroll_face(
				user["id"],
				mock_face_image,
				{"enrollment_type": "standard"}
			)
			
			assert enrollment_result["success"] is True
			assert "template_id" in enrollment_result
			
			# Verify workflow steps
			facial_service.face_engine.detect_faces.assert_called_once()
			facial_service.face_engine.extract_features.assert_called_once()
			facial_service.encryption_service.encrypt_template.assert_called_once()
			facial_service.database_service.store_template.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_complete_verification_workflow(self, facial_service, mock_face_image):
		"""Test complete verification workflow with all engines"""
		# Mock all services
		with patch.multiple(
			facial_service,
			database_service=AsyncMock(),
			encryption_service=Mock(),
			face_engine=Mock(),
			liveness_engine=Mock()
		):
			# Setup mocks for successful verification
			facial_service.face_engine.detect_faces.return_value = [
				{"bbox": [50, 50, 150, 150], "confidence": 0.99}
			]
			facial_service.face_engine.extract_features.return_value = {
				"features": np.random.rand(512).astype(np.float32),
				"quality_score": 0.90
			}
			facial_service.face_engine.compare_features.return_value = {
				"similarity_score": 0.92,
				"confidence_score": 0.95
			}
			facial_service.liveness_engine.detect_liveness.return_value = {
				"is_live": True,
				"liveness_score": 0.96
			}
			facial_service.database_service.get_user_templates.return_value = [
				{"id": "template_123", "encrypted_template": b"encrypted_data"}
			]
			facial_service.encryption_service.decrypt_template.return_value = np.random.rand(512)
			
			# Test verification workflow
			verification_config = {
				"require_liveness": True,
				"confidence_threshold": 0.8,
				"business_context": {"action": "login"}
			}
			
			result = await facial_service.verify_face(
				"user_123",
				mock_face_image,
				verification_config
			)
			
			assert result["success"] is True
			assert result["verified"] is True
			assert result["confidence_score"] >= 0.8
			assert result["liveness_result"] == "live"
			
			# Verify all workflow steps
			facial_service.face_engine.detect_faces.assert_called_once()
			facial_service.face_engine.extract_features.assert_called_once()
			facial_service.face_engine.compare_features.assert_called_once()
			facial_service.liveness_engine.detect_liveness.assert_called_once()

if __name__ == "__main__":
	pytest.main([__file__, "-v"])