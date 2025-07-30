"""
APG Facial Recognition - Database Models Unit Tests

Comprehensive unit tests for SQLAlchemy models and Pydantic validation,
ensuring data integrity and model relationships work correctly.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import json

from .models import (
	Base, FaUser, FaTemplate, FaVerification, FaEmotion, FaCollaboration,
	FaAuditLog, FaSettings, FaVerificationType, FaEmotionType,
	FaProcessingStatus, FaLivenessResult
)
from .views import (
	UserCreateRequest, UserUpdateRequest, UserResponse,
	EnrollmentRequest, EnrollmentResponse,
	VerificationRequest, VerificationResponse,
	EmotionAnalysisRequest, EmotionAnalysisResponse,
	ConsentRequest, PrivacyProcessingRequest
)

# Test Database Setup
@pytest.fixture(scope="function")
def test_engine():
	"""Create test database engine"""
	engine = create_engine(
		"sqlite:///:memory:",
		poolclass=StaticPool,
		connect_args={"check_same_thread": False}
	)
	Base.metadata.create_all(engine)
	return engine

@pytest.fixture(scope="function")
def test_session(test_engine):
	"""Create test database session"""
	TestingSessionLocal = sessionmaker(bind=test_engine)
	session = TestingSessionLocal()
	try:
		yield session
	finally:
		session.close()

@pytest.fixture
def sample_user_data():
	"""Sample user data for testing"""
	return {
		"tenant_id": "test_tenant",
		"external_user_id": "user_001",
		"full_name": "John Doe",
		"email": "john.doe@example.com",
		"phone_number": "+1234567890",
		"consent_given": True,
		"privacy_settings": {"level": "standard", "analytics": True},
		"metadata": {"department": "engineering", "role": "developer"}
	}

@pytest.fixture
def sample_template_data():
	"""Sample template data for testing"""
	return {
		"tenant_id": "test_tenant",
		"template_version": 1,
		"quality_score": Decimal("0.95"),
		"algorithm_version": "1.0.0",
		"encrypted_template": b"encrypted_biometric_data_here",
		"template_metadata": {"extraction_method": "deep_learning"},
		"is_active": True
	}

class TestFaUserModel:
	"""Test FaUser SQLAlchemy model"""
	
	def test_create_user(self, test_session, sample_user_data):
		"""Test creating a new user"""
		user = FaUser(**sample_user_data)
		test_session.add(user)
		test_session.commit()
		
		# Verify user was created
		assert user.id is not None
		assert user.external_user_id == "user_001"
		assert user.full_name == "John Doe"
		assert user.consent_given is True
		assert user.created_at is not None
		assert user.enrollment_status == "not_enrolled"
	
	def test_user_constraints(self, test_session, sample_user_data):
		"""Test user model constraints"""
		# Test unique constraint on external_user_id + tenant_id
		user1 = FaUser(**sample_user_data)
		test_session.add(user1)
		test_session.commit()
		
		# Try to create duplicate user
		user2 = FaUser(**sample_user_data)
		test_session.add(user2)
		
		with pytest.raises(Exception):  # Should raise integrity error
			test_session.commit()
	
	def test_user_relationships(self, test_session, sample_user_data, sample_template_data):
		"""Test user relationships with templates"""
		user = FaUser(**sample_user_data)
		test_session.add(user)
		test_session.flush()  # Get user ID
		
		# Create template for user
		template_data = {**sample_template_data, "user_id": user.id}
		template = FaTemplate(**template_data)
		test_session.add(template)
		test_session.commit()
		
		# Test relationship
		assert len(user.templates) == 1
		assert user.templates[0].template_version == 1
		assert template.user == user

class TestFaTemplateModel:
	"""Test FaTemplate SQLAlchemy model"""
	
	def test_create_template(self, test_session, sample_user_data, sample_template_data):
		"""Test creating a facial template"""
		# Create user first
		user = FaUser(**sample_user_data)
		test_session.add(user)
		test_session.flush()
		
		# Create template
		template_data = {**sample_template_data, "user_id": user.id}
		template = FaTemplate(**template_data)
		test_session.add(template)
		test_session.commit()
		
		# Verify template was created
		assert template.id is not None
		assert template.quality_score == Decimal("0.95")
		assert template.is_active is True
		assert template.created_at is not None
	
	def test_template_versioning(self, test_session, sample_user_data, sample_template_data):
		"""Test template versioning for same user"""
		# Create user
		user = FaUser(**sample_user_data)
		test_session.add(user)
		test_session.flush()
		
		# Create multiple template versions
		for version in [1, 2, 3]:
			template_data = {
				**sample_template_data,
				"user_id": user.id,
				"template_version": version
			}
			template = FaTemplate(**template_data)
			test_session.add(template)
		
		test_session.commit()
		
		# Verify all versions exist
		templates = test_session.query(FaTemplate).filter_by(user_id=user.id).all()
		assert len(templates) == 3
		assert {t.template_version for t in templates} == {1, 2, 3}

class TestFaVerificationModel:
	"""Test FaVerification SQLAlchemy model"""
	
	def test_create_verification(self, test_session, sample_user_data, sample_template_data):
		"""Test creating a verification record"""
		# Setup user and template
		user = FaUser(**sample_user_data)
		test_session.add(user)
		test_session.flush()
		
		template_data = {**sample_template_data, "user_id": user.id}
		template = FaTemplate(**template_data)
		test_session.add(template)
		test_session.flush()
		
		# Create verification
		verification = FaVerification(
			tenant_id="test_tenant",
			user_id=user.id,
			template_id=template.id,
			verification_type=FaVerificationType.AUTHENTICATION,
			status=FaProcessingStatus.COMPLETED,
			confidence_score=Decimal("0.92"),
			similarity_score=Decimal("0.88"),
			input_quality_score=Decimal("0.85"),
			liveness_result=FaLivenessResult.LIVE,
			liveness_score=Decimal("0.95"),
			processing_time_ms=150,
			business_context={"transaction_type": "login"},
			device_info={"device_id": "mobile_001"},
			location_data={"country": "US", "city": "San Francisco"}
		)
		test_session.add(verification)
		test_session.commit()
		
		# Verify verification was created
		assert verification.id is not None
		assert verification.confidence_score == Decimal("0.92")
		assert verification.verification_type == FaVerificationType.AUTHENTICATION
		assert verification.status == FaProcessingStatus.COMPLETED
	
	def test_verification_relationships(self, test_session, sample_user_data, sample_template_data):
		"""Test verification relationships"""
		# Setup user and template
		user = FaUser(**sample_user_data)
		test_session.add(user)
		test_session.flush()
		
		template = FaTemplate(**{**sample_template_data, "user_id": user.id})
		test_session.add(template)
		test_session.flush()
		
		# Create verification
		verification = FaVerification(
			tenant_id="test_tenant",
			user_id=user.id,
			template_id=template.id,
			verification_type=FaVerificationType.AUTHENTICATION,
			status=FaProcessingStatus.COMPLETED,
			confidence_score=Decimal("0.92")
		)
		test_session.add(verification)
		test_session.commit()
		
		# Test relationships
		assert verification.user == user
		assert verification.template == template
		assert len(user.verifications) == 1
		assert len(template.verifications) == 1

class TestFaEmotionModel:
	"""Test FaEmotion SQLAlchemy model"""
	
	def test_create_emotion_analysis(self, test_session, sample_user_data, sample_template_data):
		"""Test creating emotion analysis record"""
		# Setup verification
		user = FaUser(**sample_user_data)
		test_session.add(user)
		test_session.flush()
		
		template = FaTemplate(**{**sample_template_data, "user_id": user.id})
		test_session.add(template)
		test_session.flush()
		
		verification = FaVerification(
			tenant_id="test_tenant",
			user_id=user.id,
			template_id=template.id,
			verification_type=FaVerificationType.AUTHENTICATION,
			status=FaProcessingStatus.COMPLETED,
			confidence_score=Decimal("0.92")
		)
		test_session.add(verification)
		test_session.flush()
		
		# Create emotion analysis
		emotion = FaEmotion(
			tenant_id="test_tenant",
			verification_id=verification.id,
			primary_emotion=FaEmotionType.HAPPY,
			confidence_score=Decimal("0.85"),
			emotion_scores={
				"happy": 0.85,
				"neutral": 0.10,
				"sad": 0.05
			},
			stress_level="low",
			stress_indicators=["none"],
			micro_expressions=[
				{"type": "smile", "confidence": 0.9, "duration": 0.5}
			],
			risk_indicators=[],
			processing_time_ms=75
		)
		test_session.add(emotion)
		test_session.commit()
		
		# Verify emotion analysis
		assert emotion.id is not None
		assert emotion.primary_emotion == FaEmotionType.HAPPY
		assert emotion.confidence_score == Decimal("0.85")
		assert emotion.verification == verification

class TestPydanticValidationModels:
	"""Test Pydantic validation models"""
	
	def test_user_create_request_validation(self):
		"""Test UserCreateRequest validation"""
		# Valid request
		valid_data = {
			"external_user_id": "user_001",
			"full_name": "John Doe",
			"email": "john.doe@example.com",
			"consent_given": True
		}
		request = UserCreateRequest(**valid_data)
		assert request.external_user_id == "user_001"
		assert request.full_name == "John Doe"
		
		# Invalid email
		with pytest.raises(Exception):
			UserCreateRequest(
				external_user_id="user_001",
				full_name="John Doe",
				email="invalid_email",
				consent_given=True
			)
		
		# Missing required fields
		with pytest.raises(Exception):
			UserCreateRequest(email="john@example.com")
	
	def test_enrollment_request_validation(self):
		"""Test EnrollmentRequest validation"""
		# Valid with image_data
		valid_data = {
			"user_id": "user_001",
			"image_data": "base64_encoded_image_data",
			"quality_threshold": 0.8
		}
		request = EnrollmentRequest(**valid_data)
		assert request.user_id == "user_001"
		assert request.quality_threshold == 0.8
		
		# Valid with image_url
		valid_data_url = {
			"user_id": "user_001",
			"image_url": "https://example.com/face.jpg"
		}
		request_url = EnrollmentRequest(**valid_data_url)
		assert request_url.image_url == "https://example.com/face.jpg"
		
		# Invalid - no image source
		with pytest.raises(Exception):
			EnrollmentRequest(user_id="user_001")
		
		# Invalid quality threshold
		with pytest.raises(Exception):
			EnrollmentRequest(
				user_id="user_001",
				image_data="data",
				quality_threshold=1.5  # > 1.0
			)
	
	def test_verification_request_validation(self):
		"""Test VerificationRequest validation"""
		# Valid request
		valid_data = {
			"user_id": "user_001",
			"image_data": "base64_image",
			"confidence_threshold": 0.85,
			"require_liveness": True,
			"business_context": {"action": "login"}
		}
		request = VerificationRequest(**valid_data)
		assert request.user_id == "user_001"
		assert request.confidence_threshold == 0.85
		assert request.require_liveness is True
		
		# Invalid confidence threshold
		with pytest.raises(Exception):
			VerificationRequest(
				user_id="user_001",
				image_data="data",
				confidence_threshold=1.2  # > 1.0
			)
	
	def test_emotion_analysis_request_validation(self):
		"""Test EmotionAnalysisRequest validation"""
		# Valid with video frames
		valid_data = {
			"video_frames": ["frame1_base64", "frame2_base64"],
			"user_context": {"session_id": "session_001"}
		}
		request = EmotionAnalysisRequest(**valid_data)
		assert len(request.video_frames) == 2
		
		# Valid with single image
		valid_single = {
			"image_data": "single_frame_base64"
		}
		request_single = EmotionAnalysisRequest(**valid_single)
		assert request_single.image_data == "single_frame_base64"
	
	def test_privacy_processing_request_validation(self):
		"""Test PrivacyProcessingRequest validation"""
		# Valid request
		valid_data = {
			"user_id": "user_001",
			"biometric_data": "encrypted_biometric_data",
			"privacy_level": "enhanced",
			"processing_mode": "federated",
			"processing_purpose": "identity_verification"
		}
		request = PrivacyProcessingRequest(**valid_data)
		assert request.privacy_level.value == "enhanced"
		assert request.processing_mode.value == "federated"
		
		# Invalid privacy level
		with pytest.raises(Exception):
			PrivacyProcessingRequest(
				user_id="user_001",
				biometric_data="data",
				privacy_level="invalid_level"
			)

class TestModelSerialization:
	"""Test model serialization and deserialization"""
	
	def test_user_response_serialization(self):
		"""Test UserResponse model serialization"""
		user_data = {
			"tenant_id": "test_tenant",
			"id": "user_123",
			"external_user_id": "ext_001",
			"full_name": "John Doe",
			"email": "john@example.com",
			"enrollment_status": "enrolled",
			"consent_given": True,
			"privacy_settings": {"level": "high"},
			"metadata": {"role": "admin"}
		}
		
		user_response = UserResponse(**user_data)
		
		# Test serialization
		serialized = user_response.dict()
		assert serialized["tenant_id"] == "test_tenant"
		assert serialized["full_name"] == "John Doe"
		assert serialized["privacy_settings"]["level"] == "high"
		
		# Test JSON serialization
		json_str = user_response.json()
		assert "John Doe" in json_str
		assert "test_tenant" in json_str
	
	def test_verification_response_serialization(self):
		"""Test VerificationResponse serialization"""
		verification_data = {
			"success": True,
			"verified": True,
			"verification_id": "ver_123",
			"confidence_score": 0.95,
			"similarity_score": 0.92,
			"quality_score": 0.88,
			"liveness_score": 0.96,
			"liveness_result": "live",
			"processing_time_ms": 150,
			"risk_analysis": {"risk_level": "low"},
			"emotion_analysis": {"primary_emotion": "neutral"},
			"contextual_analysis": {"business_risk": "low"}
		}
		
		response = VerificationResponse(**verification_data)
		
		# Test serialization
		serialized = response.dict()
		assert serialized["success"] is True
		assert serialized["confidence_score"] == 0.95
		assert serialized["risk_analysis"]["risk_level"] == "low"
		
		# Test exclusion of None values
		response_partial = VerificationResponse(
			success=True,
			verified=False,
			failure_reason="Low quality image"
		)
		serialized_partial = response_partial.dict(exclude_none=True)
		assert "confidence_score" not in serialized_partial
		assert "failure_reason" in serialized_partial

# Performance and stress tests
class TestModelPerformance:
	"""Test model performance and limits"""
	
	def test_bulk_user_creation(self, test_session):
		"""Test creating multiple users efficiently"""
		import time
		
		start_time = time.time()
		users = []
		
		for i in range(100):
			user = FaUser(
				tenant_id="test_tenant",
				external_user_id=f"user_{i:03d}",
				full_name=f"User {i}",
				email=f"user{i}@example.com",
				consent_given=True
			)
			users.append(user)
		
		test_session.bulk_save_objects(users)
		test_session.commit()
		
		end_time = time.time()
		
		# Should complete quickly (under 1 second for 100 users)
		assert (end_time - start_time) < 1.0
		
		# Verify all users were created
		user_count = test_session.query(FaUser).count()
		assert user_count == 100
	
	def test_large_json_metadata(self, test_session, sample_user_data):
		"""Test handling large JSON metadata fields"""
		# Create large metadata
		large_metadata = {
			"preferences": {f"setting_{i}": f"value_{i}" for i in range(1000)},
			"history": [{"action": f"action_{i}", "timestamp": f"2025-01-{i%30+1:02d}"} for i in range(500)],
			"analytics": {f"metric_{i}": i * 0.123 for i in range(200)}
		}
		
		user_data = {**sample_user_data, "metadata": large_metadata}
		user = FaUser(**user_data)
		test_session.add(user)
		test_session.commit()
		
		# Verify large metadata was stored and retrieved correctly
		retrieved_user = test_session.query(FaUser).filter_by(id=user.id).first()
		assert len(retrieved_user.metadata["preferences"]) == 1000
		assert len(retrieved_user.metadata["history"]) == 500
		assert len(retrieved_user.metadata["analytics"]) == 200

if __name__ == "__main__":
	pytest.main([__file__, "-v"])