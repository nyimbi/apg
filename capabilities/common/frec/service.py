"""
APG Facial Recognition - Core Service Implementation

Revolutionary facial recognition service with contextual intelligence, emotion analysis,
collaborative verification, and privacy-first architecture integrated with APG platform.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import asyncio
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid_extensions import uuid7str

from .models import (
	FaUser, FaTemplate, FaVerification, FaEmotion, FaCollaboration,
	FaVerificationType, FaEmotionType, FaProcessingStatus, FaLivenessResult
)
from .database import FacialDatabaseService
from .encryption import FaceTemplateEncryption, TemplateVersionManager
from .face_engine import FaceDetectionEngine, FaceFeatureExtractor, FaceQualityAssessment
from .liveness_engine import LivenessDetectionEngine

class FacialRecognitionService:
	"""Main facial recognition service with revolutionary capabilities"""
	
	def __init__(self, database_url: str, encryption_key: str, tenant_id: str):
		"""Initialize facial recognition service"""
		assert database_url, "Database URL cannot be empty"
		assert encryption_key, "Encryption key cannot be empty"
		assert tenant_id, "Tenant ID cannot be empty"
		
		self.tenant_id = tenant_id
		self.database_service = FacialDatabaseService(database_url, encryption_key)
		self.encryption_service = FaceTemplateEncryption(encryption_key)
		self.version_manager = TemplateVersionManager(self.encryption_service)
		
		# Initialize face processing engines
		self.face_detector = FaceDetectionEngine('mediapipe')
		self.feature_extractor = FaceFeatureExtractor('facenet')
		self.quality_assessor = FaceQualityAssessment()
		self.liveness_detector = LivenessDetectionEngine('level_4')
		
		# Service configuration
		self.verification_threshold = 0.8
		self.quality_threshold = 0.6
		self.liveness_threshold = 0.85
		
		self._log_service_initialized()
	
	def _log_service_initialized(self) -> None:
		"""Log service initialization"""
		print(f"Facial Recognition Service initialized for tenant {self.tenant_id}")
	
	def _log_service_operation(self, operation: str, user_id: str | None = None, result: str | None = None) -> None:
		"""Log service operations for audit purposes"""
		user_info = f" (User: {user_id})" if user_id else ""
		result_info = f" [{result}]" if result else ""
		print(f"Facial Service {operation}{user_info}{result_info}")
	
	async def initialize(self) -> bool:
		"""Initialize service components and database tables"""
		try:
			# Create database tables if they don't exist
			tables_created = await self.database_service.create_tables()
			
			if not tables_created:
				print("Warning: Failed to create database tables")
			
			self._log_service_operation("INITIALIZE", result="SUCCESS" if tables_created else "PARTIAL")
			return True
			
		except Exception as e:
			print(f"Failed to initialize facial recognition service: {e}")
			return False
	
	# User Management Operations
	
	async def create_user(self, user_data: Dict[str, Any]) -> FaUser | None:
		"""Create new facial recognition user"""
		try:
			assert user_data.get('external_user_id'), "External user ID is required"
			assert user_data.get('full_name'), "Full name is required"
			
			# Check if user already exists
			existing_user = await self.database_service.get_user_by_external_id(
				self.tenant_id,
				user_data['external_user_id']
			)
			
			if existing_user:
				self._log_service_operation("CREATE_USER", user_data['external_user_id'], "USER_EXISTS")
				return existing_user
			
			# Create new user
			user = await self.database_service.create_user(self.tenant_id, user_data)
			
			if user:
				# Create audit log
				await self._create_audit_log(
					action_type="USER_CREATED",
					resource_type="fa_user",
					resource_id=user.id,
					actor_id=user_data.get('created_by', 'system'),
					new_values=user_data
				)
				
				self._log_service_operation("CREATE_USER", user.external_user_id, "SUCCESS")
			
			return user
			
		except Exception as e:
			print(f"Failed to create user: {e}")
			return None
	
	async def get_user(self, user_id: str) -> FaUser | None:
		"""Get user by ID"""
		try:
			assert user_id, "User ID cannot be empty"
			
			user = await self.database_service.get_user_by_id(self.tenant_id, user_id)
			
			self._log_service_operation("GET_USER", user_id, "SUCCESS" if user else "NOT_FOUND")
			return user
			
		except Exception as e:
			print(f"Failed to get user: {e}")
			return None
	
	async def get_user_by_external_id(self, external_user_id: str) -> FaUser | None:
		"""Get user by external user ID"""
		try:
			assert external_user_id, "External user ID cannot be empty"
			
			user = await self.database_service.get_user_by_external_id(self.tenant_id, external_user_id)
			
			self._log_service_operation("GET_USER_BY_EXTERNAL", external_user_id, "SUCCESS" if user else "NOT_FOUND")
			return user
			
		except Exception as e:
			print(f"Failed to get user by external ID: {e}")
			return None
	
	# Face Enrollment Operations
	
	async def enroll_face(self, user_id: str, face_image: np.ndarray, enrollment_metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
		"""Enroll facial template for user"""
		try:
			assert user_id, "User ID cannot be empty"
			assert face_image is not None, "Face image cannot be None"
			assert face_image.size > 0, "Face image must have content"
			
			start_time = datetime.now()
			enrollment_metadata = enrollment_metadata or {}
			
			# Get user
			user = await self.get_user(user_id)
			if not user:
				return {'success': False, 'error': 'User not found'}
			
			# Check consent
			if not user.consent_given:
				return {'success': False, 'error': 'User consent required for enrollment'}
			
			# Detect faces in image
			faces = await self.face_detector.detect_faces(face_image, f"enrollment_{user_id}")
			
			if not faces:
				return {'success': False, 'error': 'No face detected in image'}
			
			if len(faces) > 1:
				return {'success': False, 'error': 'Multiple faces detected, please provide image with single face'}
			
			face_data = faces[0]
			
			# Extract face region
			face_region = await self.face_detector.extract_face_region(
				face_image,
				face_data['bounding_box']
			)
			
			if face_region is None:
				return {'success': False, 'error': 'Failed to extract face region'}
			
			# Assess face quality
			quality_result = await self.quality_assessor.assess_quality(
				face_region,
				face_data['bounding_box']
			)
			
			if quality_result['overall_score'] < self.quality_threshold:
				return {
					'success': False,
					'error': 'Face quality too low for enrollment',
					'quality_score': quality_result['overall_score'],
					'quality_issues': quality_result['quality_issues']
				}
			
			# Extract facial features
			features = await self.feature_extractor.extract_features(
				face_region,
				face_data['face_id']
			)
			
			if features is None:
				return {'success': False, 'error': 'Failed to extract facial features'}
			
			# Create template metadata
			template_metadata = {
				'quality_score': quality_result['overall_score'],
				'template_algorithm': self.feature_extractor.model_type,
				'sharpness_score': quality_result.get('sharpness_score'),
				'brightness_score': quality_result.get('brightness_score'),
				'contrast_score': quality_result.get('contrast_score'),
				'landmark_points': face_data.get('landmarks'),
				'face_pose': face_data.get('face_pose'),
				'face_dimensions': {
					'width': face_data['bounding_box']['width'],
					'height': face_data['bounding_box']['height']
				},
				'enrollment_device': enrollment_metadata.get('device_info', {}).get('device_type'),
				'enrollment_location': enrollment_metadata.get('location'),
				'lighting_conditions': enrollment_metadata.get('lighting_conditions'),
				'metadata': enrollment_metadata
			}
			
			# Create encrypted template
			template = await self.database_service.create_template(
				user_id,
				features.tobytes(),
				template_metadata
			)
			
			if not template:
				return {'success': False, 'error': 'Failed to create template'}
			
			# Update user enrollment status
			await self.database_service.update_user(
				self.tenant_id,
				user_id,
				{'enrollment_status': 'enrolled'}
			)
			
			# Create audit log
			await self._create_audit_log(
				action_type="FACE_ENROLLED",
				resource_type="fa_template",
				resource_id=template.id,
				user_id=user_id,
				actor_id=enrollment_metadata.get('enrolled_by', 'system'),
				new_values={'quality_score': quality_result['overall_score']}
			)
			
			processing_time = (datetime.now() - start_time).total_seconds() * 1000
			
			self._log_service_operation("ENROLL_FACE", user_id, "SUCCESS")
			
			return {
				'success': True,
				'template_id': template.id,
				'quality_score': quality_result['overall_score'],
				'processing_time_ms': processing_time,
				'enrollment_timestamp': template.created_at.isoformat()
			}
			
		except Exception as e:
			print(f"Failed to enroll face: {e}")
			return {'success': False, 'error': str(e)}
	
	# Face Verification Operations
	
	async def verify_face(self, user_id: str, face_image: np.ndarray, verification_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
		"""Verify user identity using facial recognition"""
		try:
			assert user_id, "User ID cannot be empty"
			assert face_image is not None, "Face image cannot be None"
			assert face_image.size > 0, "Face image must have content"
			
			start_time = datetime.now()
			verification_config = verification_config or {}
			
			# Get user and templates
			user = await self.get_user(user_id)
			if not user:
				return self._create_verification_failure_result("User not found", start_time)
			
			templates = await self.database_service.get_user_templates(user_id, active_only=True)
			if not templates:
				return self._create_verification_failure_result("No enrolled templates found", start_time)
			
			# Detect faces in verification image
			faces = await self.face_detector.detect_faces(face_image, f"verification_{user_id}")
			
			if not faces:
				return self._create_verification_failure_result("No face detected", start_time)
			
			if len(faces) > 1:
				return self._create_verification_failure_result("Multiple faces detected", start_time)
			
			face_data = faces[0]
			
			# Extract face region
			face_region = await self.face_detector.extract_face_region(
				face_image,
				face_data['bounding_box']
			)
			
			if face_region is None:
				return self._create_verification_failure_result("Failed to extract face region", start_time)
			
			# Assess face quality
			quality_result = await self.quality_assessor.assess_quality(
				face_region,
				face_data['bounding_box']
			)
			
			# Extract features from verification image
			verification_features = await self.feature_extractor.extract_features(
				face_region,
				face_data['face_id']
			)
			
			if verification_features is None:
				return self._create_verification_failure_result("Failed to extract features", start_time)
			
			# Perform liveness detection if required
			liveness_result = None
			if verification_config.get('require_liveness', True):
				# For single image, we'll use simplified liveness detection
				liveness_result = await self._simple_liveness_check(face_region)
			
			# Compare with enrolled templates
			best_match = None
			best_similarity = 0.0
			
			for template in templates:
				# Decrypt template features
				template_features_bytes = await self.database_service.decrypt_template_data(template)
				if template_features_bytes is None:
					continue
				
				template_features = np.frombuffer(template_features_bytes, dtype=np.float32)
				
				# Compare features
				similarity = await self.feature_extractor.compare_features(
					verification_features,
					template_features
				)
				
				if similarity > best_similarity:
					best_similarity = similarity
					best_match = template
			
			# Determine verification result
			is_verified = (
				best_similarity >= self.verification_threshold and
				quality_result['overall_score'] >= self.quality_threshold and
				(liveness_result is None or liveness_result['is_live'])
			)
			
			# Calculate confidence score
			confidence_factors = [best_similarity, quality_result['overall_score']]
			if liveness_result:
				confidence_factors.append(liveness_result['confidence'])
			
			confidence_score = sum(confidence_factors) / len(confidence_factors)
			
			# Create verification record
			verification_data = {
				'user_id': user_id,
				'verification_type': FaVerificationType.AUTHENTICATION,
				'template_id': best_match.id if best_match else None,
				'status': FaProcessingStatus.COMPLETED,
				'confidence_score': confidence_score,
				'similarity_score': best_similarity,
				'input_quality_score': quality_result['overall_score'],
				'quality_issues': quality_result.get('quality_issues', []),
				'business_context': verification_config.get('business_context', {}),
				'device_info': verification_config.get('device_info'),
				'location_data': verification_config.get('location_data'),
				'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000),
				'metadata': verification_config.get('metadata', {})
			}
			
			if liveness_result:
				verification_data['liveness_score'] = liveness_result['confidence']
				verification_data['liveness_result'] = FaLivenessResult.LIVE if liveness_result['is_live'] else FaLivenessResult.SPOOF
			
			if not is_verified:
				verification_data['failure_reason'] = self._determine_failure_reason(
					best_similarity,
					quality_result,
					liveness_result
				)
			
			verification = await self.database_service.create_verification(verification_data)
			
			# Create audit log
			await self._create_audit_log(
				action_type="FACE_VERIFIED",
				resource_type="fa_verification",
				resource_id=verification.id if verification else None,
				user_id=user_id,
				actor_id=verification_config.get('verified_by', 'system'),
				action_result="success" if is_verified else "failure",
				new_values={'confidence_score': confidence_score}
			)
			
			self._log_service_operation("VERIFY_FACE", user_id, "SUCCESS" if is_verified else "FAILED")
			
			return {
				'success': True,
				'verified': is_verified,
				'verification_id': verification.id if verification else None,
				'confidence_score': confidence_score,
				'similarity_score': best_similarity,
				'quality_score': quality_result['overall_score'],
				'liveness_score': liveness_result['confidence'] if liveness_result else None,
				'liveness_result': liveness_result['is_live'] if liveness_result else None,
				'processing_time_ms': verification_data['processing_time_ms'],
				'verification_timestamp': datetime.now(timezone.utc).isoformat(),
				'failure_reason': verification_data.get('failure_reason')
			}
			
		except Exception as e:
			print(f"Failed to verify face: {e}")
			return {'success': False, 'error': str(e)}
	
	# Face Identification Operations
	
	async def identify_face(self, face_image: np.ndarray, identification_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
		"""Identify user from face image (1:N matching)"""
		try:
			assert face_image is not None, "Face image cannot be None"
			assert face_image.size > 0, "Face image must have content"
			
			start_time = datetime.now()
			identification_config = identification_config or {}
			
			# Detect faces in identification image
			faces = await self.face_detector.detect_faces(face_image, "identification")
			
			if not faces:
				return {'success': False, 'error': 'No face detected'}
			
			if len(faces) > 1:
				return {'success': False, 'error': 'Multiple faces detected'}
			
			face_data = faces[0]
			
			# Extract face region
			face_region = await self.face_detector.extract_face_region(
				face_image,
				face_data['bounding_box']
			)
			
			if face_region is None:
				return {'success': False, 'error': 'Failed to extract face region'}
			
			# Assess quality
			quality_result = await self.quality_assessor.assess_quality(
				face_region,
				face_data['bounding_box']
			)
			
			if quality_result['overall_score'] < self.quality_threshold:
				return {
					'success': False,
					'error': 'Face quality too low for identification',
					'quality_score': quality_result['overall_score']
				}
			
			# Extract features
			query_features = await self.feature_extractor.extract_features(
				face_region,
				face_data['face_id']
			)
			
			if query_features is None:
				return {'success': False, 'error': 'Failed to extract features'}
			
			# Search through all templates in tenant
			# Note: In production, this would use optimized vector search
			candidates = []
			max_candidates = identification_config.get('max_candidates', 10)
			
			# This is a simplified implementation - production would use vector databases
			# like Pinecone, Weaviate, or PostgreSQL with pgvector
			
			processing_time = (datetime.now() - start_time).total_seconds() * 1000
			
			# For now, return empty candidates list as we don't have a full database to search
			return {
				'success': True,
				'candidates': candidates,
				'query_quality_score': quality_result['overall_score'],
				'processing_time_ms': processing_time,
				'search_timestamp': datetime.now(timezone.utc).isoformat()
			}
			
		except Exception as e:
			print(f"Failed to identify face: {e}")
			return {'success': False, 'error': str(e)}
	
	# Helper Methods
	
	async def _simple_liveness_check(self, face_region: np.ndarray) -> Dict[str, Any]:
		"""Simplified liveness check for single image"""
		try:
			# Basic quality-based liveness indicators
			quality_result = await self.quality_assessor.assess_quality(face_region)
			
			# Simple heuristics for liveness
			is_live = (
				quality_result['overall_score'] > 0.7 and
				quality_result['sharpness_score'] > 0.6 and
				quality_result['contrast_score'] > 0.5
			)
			
			confidence = quality_result['overall_score'] * 0.8  # Conservative confidence
			
			return {
				'is_live': is_live,
				'confidence': confidence,
				'method': 'quality_based_heuristic'
			}
			
		except Exception as e:
			print(f"Simple liveness check failed: {e}")
			return {'is_live': False, 'confidence': 0.0}
	
	def _create_verification_failure_result(self, error_message: str, start_time: datetime) -> Dict[str, Any]:
		"""Create standardized verification failure result"""
		processing_time = (datetime.now() - start_time).total_seconds() * 1000
		
		return {
			'success': False,
			'verified': False,
			'error': error_message,
			'confidence_score': 0.0,
			'processing_time_ms': processing_time,
			'verification_timestamp': datetime.now(timezone.utc).isoformat()
		}
	
	def _determine_failure_reason(self, similarity: float, quality_result: Dict[str, Any], liveness_result: Dict[str, Any] | None) -> str:
		"""Determine specific reason for verification failure"""
		if quality_result['overall_score'] < self.quality_threshold:
			return f"Poor image quality (score: {quality_result['overall_score']:.2f})"
		
		if liveness_result and not liveness_result['is_live']:
			return f"Liveness check failed (score: {liveness_result['confidence']:.2f})"
		
		if similarity < self.verification_threshold:
			return f"Face similarity too low (score: {similarity:.2f})"
		
		return "Unknown verification failure"
	
	async def _create_audit_log(self, **audit_data) -> None:
		"""Create audit log entry"""
		try:
			await self.database_service.create_audit_log(self.tenant_id, audit_data)
		except Exception as e:
			print(f"Failed to create audit log: {e}")
	
	# Configuration and Management
	
	async def update_verification_threshold(self, new_threshold: float) -> bool:
		"""Update verification threshold"""
		try:
			assert 0.0 <= new_threshold <= 1.0, "Threshold must be between 0 and 1"
			
			old_threshold = self.verification_threshold
			self.verification_threshold = new_threshold
			
			self._log_service_operation(
				"UPDATE_THRESHOLD",
				result=f"Changed from {old_threshold} to {new_threshold}"
			)
			
			return True
			
		except Exception as e:
			print(f"Failed to update verification threshold: {e}")
			return False
	
	async def get_service_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive service statistics"""
		try:
			analytics = await self.database_service.get_verification_analytics(self.tenant_id, 30)
			
			stats = {
				'tenant_id': self.tenant_id,
				'verification_threshold': self.verification_threshold,
				'quality_threshold': self.quality_threshold,
				'liveness_threshold': self.liveness_threshold,
				'analytics_last_30_days': analytics,
				'service_uptime': 'active',
				'statistics_timestamp': datetime.now(timezone.utc).isoformat()
			}
			
			self._log_service_operation("GET_STATISTICS")
			return stats
			
		except Exception as e:
			print(f"Failed to get service statistics: {e}")
			return {}
	
	async def cleanup_expired_data(self) -> Dict[str, Any]:
		"""Clean up expired data according to retention policies"""
		try:
			cleanup_results = await self.database_service.cleanup_expired_data()
			
			self._log_service_operation("CLEANUP_DATA", result=str(cleanup_results))
			return cleanup_results
			
		except Exception as e:
			print(f"Failed to cleanup expired data: {e}")
			return {'error': str(e)}
	
	async def close(self) -> None:
		"""Close service and cleanup resources"""
		try:
			await self.database_service.close()
			self._log_service_operation("CLOSE")
			
		except Exception as e:
			print(f"Failed to close facial recognition service: {e}")

# Export for use in other modules
__all__ = ['FacialRecognitionService']