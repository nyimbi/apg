"""
APG Multi-Factor Authentication (MFA) - Biometric Enrollment Wizard

User-friendly biometric enrollment wizard with intelligent guidance,
quality assessment, and seamless onboarding experience.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import logging
import base64
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime, timedelta
from uuid_extensions import uuid7str
from enum import Enum

from .models import (
	MFAUserProfile, MFAMethod, MFAMethodType, BiometricTemplate,
	TrustLevel, DeviceBinding
)
from .biometric_service import BiometricService, BiometricQualityAssessment
from .integration import (
	APGIntegrationRouter, MFANotificationRequest,
	create_mfa_notification
)


def _log_enrollment_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log enrollment operations for debugging and audit"""
	return f"[Enrollment Wizard] {operation} for user {user_id}: {details}"


class EnrollmentStep(str, Enum):
	"""Enrollment wizard steps"""
	INTRODUCTION = "introduction"
	DEVICE_VERIFICATION = "device_verification"  
	BIOMETRIC_SELECTION = "biometric_selection"
	FACE_ENROLLMENT = "face_enrollment"
	VOICE_ENROLLMENT = "voice_enrollment"
	BEHAVIORAL_BASELINE = "behavioral_baseline"
	BACKUP_CODES = "backup_codes"
	VERIFICATION_TEST = "verification_test"
	COMPLETION = "completion"


class EnrollmentSession:
	"""Enrollment session state management"""
	
	def __init__(self, user_id: str, tenant_id: str):
		self.session_id = uuid7str()
		self.user_id = user_id
		self.tenant_id = tenant_id
		self.current_step = EnrollmentStep.INTRODUCTION
		self.started_at = datetime.utcnow()
		self.completed_steps: List[EnrollmentStep] = []
		self.enrollment_data: Dict[str, Any] = {}
		self.biometric_methods: Dict[str, Dict[str, Any]] = {}
		self.quality_assessments: Dict[str, Dict[str, Any]] = {}
		self.attempt_counts: Dict[str, int] = {}
		self.device_info: Dict[str, Any] = {}
		self.user_preferences: Dict[str, Any] = {}
	
	def mark_step_complete(self, step: EnrollmentStep) -> None:
		"""Mark a step as completed"""
		if step not in self.completed_steps:
			self.completed_steps.append(step)
	
	def is_step_completed(self, step: EnrollmentStep) -> bool:
		"""Check if a step is completed"""
		return step in self.completed_steps
	
	def get_progress_percentage(self) -> float:
		"""Get enrollment progress percentage"""
		total_steps = len(EnrollmentStep)
		completed_count = len(self.completed_steps)
		return (completed_count / total_steps) * 100


class BiometricEnrollmentWizard:
	"""
	Comprehensive biometric enrollment wizard with intelligent guidance,
	quality feedback, and user-friendly experience.
	"""
	
	def __init__(self, 
				biometric_service: BiometricService,
				apg_integration_router: APGIntegrationRouter,
				database_client: Any):
		"""Initialize enrollment wizard"""
		self.biometric_service = biometric_service
		self.apg_router = apg_integration_router
		self.db = database_client
		self.logger = logging.getLogger(__name__)
		
		# Active enrollment sessions
		self.active_sessions: Dict[str, EnrollmentSession] = {}
		
		# Enrollment configuration
		self.max_enrollment_attempts = 3
		self.enrollment_timeout_minutes = 30
		self.min_quality_threshold = 0.7
		self.recommended_methods = [
			MFAMethodType.BIOMETRIC_FACE,
			MFAMethodType.TOKEN_TOTP,
			MFAMethodType.BACKUP_CODES
		]
	
	async def start_enrollment(self, 
							  user_id: str, 
							  tenant_id: str,
							  device_info: Dict[str, Any],
							  user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""
		Start biometric enrollment process.
		
		Args:
			user_id: User starting enrollment
			tenant_id: Tenant context
			device_info: Device information
			user_preferences: Optional user preferences
		
		Returns:
			Enrollment session initialization result
		"""
		try:
			self.logger.info(_log_enrollment_operation("start_enrollment", user_id))
			
			# Create enrollment session
			session = EnrollmentSession(user_id, tenant_id)
			session.device_info = device_info
			session.user_preferences = user_preferences or {}
			
			# Store session
			self.active_sessions[session.session_id] = session
			
			# Check device capabilities
			device_capabilities = await self._assess_device_capabilities(device_info)
			
			# Generate personalized enrollment plan
			enrollment_plan = await self._generate_enrollment_plan(
				device_capabilities, user_preferences
			)
			
			# Send welcome notification
			await self._send_enrollment_notification(
				user_id, tenant_id, "enrollment_started",
				"Biometric enrollment started", 
				"Your secure biometric enrollment process has begun."
			)
			
			return {
				"success": True,
				"session_id": session.session_id,
				"current_step": session.current_step,
				"enrollment_plan": enrollment_plan,
				"device_capabilities": device_capabilities,
				"estimated_time_minutes": self._estimate_enrollment_time(enrollment_plan),
				"next_action": await self._get_step_instructions(session, EnrollmentStep.INTRODUCTION)
			}
			
		except Exception as e:
			self.logger.error(f"Enrollment start error for user {user_id}: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "enrollment_start_failed",
				"message": str(e)
			}
	
	async def process_enrollment_step(self,
									 session_id: str,
									 step: EnrollmentStep,
									 step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Process a specific enrollment step.
		
		Args:
			session_id: Enrollment session ID
			step: Current step being processed
			step_data: Data for the current step
		
		Returns:
			Step processing result and next instructions
		"""
		try:
			# Get session
			session = self.active_sessions.get(session_id)
			if not session:
				return {
					"success": False,
					"error": "session_not_found",
					"message": "Enrollment session not found or expired"
				}
			
			self.logger.info(_log_enrollment_operation(
				f"process_step_{step}", session.user_id, f"session={session_id}"
			))
			
			# Check session timeout
			if self._is_session_expired(session):
				await self._cleanup_session(session_id)
				return {
					"success": False,
					"error": "session_expired",
					"message": "Enrollment session has expired"
				}
			
			# Process step based on type
			if step == EnrollmentStep.INTRODUCTION:
				result = await self._process_introduction_step(session, step_data)
			elif step == EnrollmentStep.DEVICE_VERIFICATION:
				result = await self._process_device_verification_step(session, step_data)
			elif step == EnrollmentStep.BIOMETRIC_SELECTION:
				result = await self._process_biometric_selection_step(session, step_data)
			elif step == EnrollmentStep.FACE_ENROLLMENT:
				result = await self._process_face_enrollment_step(session, step_data)
			elif step == EnrollmentStep.VOICE_ENROLLMENT:
				result = await self._process_voice_enrollment_step(session, step_data)
			elif step == EnrollmentStep.BEHAVIORAL_BASELINE:
				result = await self._process_behavioral_baseline_step(session, step_data)
			elif step == EnrollmentStep.BACKUP_CODES:
				result = await self._process_backup_codes_step(session, step_data)
			elif step == EnrollmentStep.VERIFICATION_TEST:
				result = await self._process_verification_test_step(session, step_data)
			elif step == EnrollmentStep.COMPLETION:
				result = await self._process_completion_step(session, step_data)
			else:
				return {
					"success": False,
					"error": "invalid_step",
					"message": f"Unknown enrollment step: {step}"
				}
			
			# Update session state if step was successful
			if result.get("success"):
				session.mark_step_complete(step)
				session.enrollment_data[step] = step_data
				
				# Determine next step
				next_step = self._get_next_step(session)
				if next_step:
					session.current_step = next_step
					result["next_step"] = next_step
					result["next_action"] = await self._get_step_instructions(session, next_step)
				else:
					# Enrollment complete
					result["enrollment_complete"] = True
					await self._finalize_enrollment(session)
			
			# Add progress information
			result["progress"] = {
				"percentage": session.get_progress_percentage(),
				"completed_steps": len(session.completed_steps),
				"total_steps": len(EnrollmentStep),
				"current_step": session.current_step
			}
			
			return result
			
		except Exception as e:
			self.logger.error(f"Enrollment step processing error: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "step_processing_failed",
				"message": str(e)
			}
	
	async def get_enrollment_guidance(self,
									 session_id: str,
									 guidance_type: str) -> Dict[str, Any]:
		"""
		Get contextual guidance for enrollment process.
		
		Args:
			session_id: Enrollment session ID
			guidance_type: Type of guidance requested
		
		Returns:
			Contextual guidance and tips
		"""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return {
					"success": False,
					"error": "session_not_found"
				}
			
			if guidance_type == "quality_tips":
				guidance = await self._get_quality_improvement_tips(session)
			elif guidance_type == "troubleshooting":
				guidance = await self._get_troubleshooting_guidance(session)
			elif guidance_type == "accessibility":
				guidance = await self._get_accessibility_guidance(session)
			elif guidance_type == "privacy":
				guidance = await self._get_privacy_information(session)
			else:
				return {
					"success": False,
					"error": "invalid_guidance_type"
				}
			
			return {
				"success": True,
				"guidance_type": guidance_type,
				"guidance": guidance
			}
			
		except Exception as e:
			self.logger.error(f"Enrollment guidance error: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "guidance_failed",
				"message": str(e)
			}
	
	# Step processing methods
	
	async def _process_introduction_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process introduction and consent step"""
		try:
			# Validate required consents
			required_consents = ["biometric_processing", "data_storage", "terms_of_service"]
			given_consents = step_data.get("consents", [])
			
			missing_consents = [consent for consent in required_consents if consent not in given_consents]
			
			if missing_consents:
				return {
					"success": False,
					"error": "missing_consents",
					"message": "Required consents not provided",
					"missing_consents": missing_consents
				}
			
			# Store consent information
			session.enrollment_data["consents"] = given_consents
			session.enrollment_data["consent_timestamp"] = datetime.utcnow().isoformat()
			
			return {
				"success": True,
				"message": "Introduction completed successfully",
				"consents_recorded": given_consents
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "introduction_failed",
				"message": str(e)
			}
	
	async def _process_device_verification_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process device verification and security check"""
		try:
			# Verify device security
			from .anti_spoofing import SecurityHardeningService
			security_service = SecurityHardeningService()
			
			device_security = await security_service.validate_device_security(session.device_info)
			
			if not device_security["is_secure"]:
				return {
					"success": False,
					"error": "device_not_secure",
					"message": "Device does not meet security requirements",
					"security_issues": device_security["security_issues"],
					"recommendations": device_security["recommendations"]
				}
			
			# Create device binding
			device_binding = DeviceBinding(
				device_id=session.device_info.get("device_id", ""),
				device_type=session.device_info.get("device_type", ""),
				device_name=session.device_info.get("device_name", ""),
				device_fingerprint=session.device_info.get("fingerprint", ""),
				trust_level=TrustLevel.MEDIUM,
				tenant_id=session.tenant_id,
				created_by=session.user_id,
				updated_by=session.user_id
			)
			
			# Store device binding
			await self._store_device_binding(device_binding)
			session.enrollment_data["device_binding_id"] = device_binding.id
			
			return {
				"success": True,
				"message": "Device verification completed",
				"device_binding_id": device_binding.id,
				"security_score": device_security["security_score"]
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "device_verification_failed",
				"message": str(e)
			}
	
	async def _process_biometric_selection_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process biometric method selection"""
		try:
			selected_methods = step_data.get("selected_methods", [])
			
			if not selected_methods:
				return {
					"success": False,
					"error": "no_methods_selected",
					"message": "At least one biometric method must be selected"
				}
			
			# Validate method availability on device
			device_capabilities = await self._assess_device_capabilities(session.device_info)
			
			unavailable_methods = []
			for method in selected_methods:
				if method == "face" and not device_capabilities.get("camera_available"):
					unavailable_methods.append("face")
				elif method == "voice" and not device_capabilities.get("microphone_available"):
					unavailable_methods.append("voice")
			
			if unavailable_methods:
				return {
					"success": False,
					"error": "methods_unavailable",
					"message": "Selected methods not available on device",
					"unavailable_methods": unavailable_methods
				}
			
			session.enrollment_data["selected_biometric_methods"] = selected_methods
			
			return {
				"success": True,
				"message": "Biometric methods selected",
				"selected_methods": selected_methods,
				"enrollment_order": self._optimize_enrollment_order(selected_methods)
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "biometric_selection_failed",
				"message": str(e)
			}
	
	async def _process_face_enrollment_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process face biometric enrollment"""
		try:
			# Check if face enrollment was selected
			selected_methods = session.enrollment_data.get("selected_biometric_methods", [])
			if "face" not in selected_methods:
				return {
					"success": True,
					"message": "Face enrollment skipped (not selected)",
					"skipped": True
				}
			
			# Get face data
			face_data = step_data.get("face_data")
			if not face_data:
				return {
					"success": False,
					"error": "no_face_data",
					"message": "Face image data is required"
				}
			
			# Track attempt count
			attempt_key = "face_enrollment"
			session.attempt_counts[attempt_key] = session.attempt_counts.get(attempt_key, 0) + 1
			
			if session.attempt_counts[attempt_key] > self.max_enrollment_attempts:
				return {
					"success": False,
					"error": "max_attempts_exceeded",
					"message": "Maximum enrollment attempts exceeded"
				}
			
			# Process face enrollment
			enrollment_result = await self.biometric_service.enroll_biometric(
				session.user_id,
				session.tenant_id,
				"face",
				face_data,
				step_data.get("metadata", {})
			)
			
			if enrollment_result["success"]:
				# Store successful enrollment
				session.biometric_methods["face"] = enrollment_result
				
				# Send confirmation notification
				await self._send_enrollment_notification(
					session.user_id, session.tenant_id, "face_enrolled",
					"Face biometric enrolled", 
					"Your face biometric has been successfully enrolled."
				)
				
				return {
					"success": True,
					"message": "Face enrollment completed successfully",
					"template_id": enrollment_result["template_id"],
					"quality_score": enrollment_result["quality_score"],
					"attempt_number": session.attempt_counts[attempt_key]
				}
			else:
				# Provide specific feedback for improvement
				feedback = await self._generate_enrollment_feedback(
					"face", enrollment_result, session.attempt_counts[attempt_key]
				)
				
				return {
					"success": False,
					"error": enrollment_result.get("error", "face_enrollment_failed"),
					"message": enrollment_result.get("message", "Face enrollment failed"),
					"feedback": feedback,
					"retry_allowed": session.attempt_counts[attempt_key] < self.max_enrollment_attempts,
					"attempts_remaining": self.max_enrollment_attempts - session.attempt_counts[attempt_key]
				}
			
		except Exception as e:
			return {
				"success": False,
				"error": "face_enrollment_error",
				"message": str(e)
			}
	
	async def _process_voice_enrollment_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process voice biometric enrollment"""
		try:
			# Check if voice enrollment was selected
			selected_methods = session.enrollment_data.get("selected_biometric_methods", [])
			if "voice" not in selected_methods:
				return {
					"success": True,
					"message": "Voice enrollment skipped (not selected)",
					"skipped": True
				}
			
			# Get voice data
			voice_data = step_data.get("voice_data")
			if not voice_data:
				return {
					"success": False,
					"error": "no_voice_data",
					"message": "Voice audio data is required"
				}
			
			# Track attempt count
			attempt_key = "voice_enrollment"
			session.attempt_counts[attempt_key] = session.attempt_counts.get(attempt_key, 0) + 1
			
			if session.attempt_counts[attempt_key] > self.max_enrollment_attempts:
				return {
					"success": False,
					"error": "max_attempts_exceeded",
					"message": "Maximum enrollment attempts exceeded"
				}
			
			# Process voice enrollment
			enrollment_result = await self.biometric_service.enroll_biometric(
				session.user_id,
				session.tenant_id,
				"voice",
				voice_data,
				step_data.get("metadata", {})
			)
			
			if enrollment_result["success"]:
				# Store successful enrollment
				session.biometric_methods["voice"] = enrollment_result
				
				# Send confirmation notification
				await self._send_enrollment_notification(
					session.user_id, session.tenant_id, "voice_enrolled",
					"Voice biometric enrolled", 
					"Your voice biometric has been successfully enrolled."
				)
				
				return {
					"success": True,
					"message": "Voice enrollment completed successfully",
					"template_id": enrollment_result["template_id"],
					"quality_score": enrollment_result["quality_score"],
					"attempt_number": session.attempt_counts[attempt_key]
				}
			else:
				# Provide specific feedback for improvement
				feedback = await self._generate_enrollment_feedback(
					"voice", enrollment_result, session.attempt_counts[attempt_key]
				)
				
				return {
					"success": False,
					"error": enrollment_result.get("error", "voice_enrollment_failed"),
					"message": enrollment_result.get("message", "Voice enrollment failed"),
					"feedback": feedback,
					"retry_allowed": session.attempt_counts[attempt_key] < self.max_enrollment_attempts,
					"attempts_remaining": self.max_enrollment_attempts - session.attempt_counts[attempt_key]
				}
			
		except Exception as e:
			return {
				"success": False,
				"error": "voice_enrollment_error",
				"message": str(e)
			}
	
	async def _process_behavioral_baseline_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process behavioral baseline establishment"""
		try:
			# Check if behavioral analysis was requested
			if not session.user_preferences.get("enable_behavioral_analysis", True):
				return {
					"success": True,
					"message": "Behavioral baseline skipped (disabled by user)",
					"skipped": True
				}
			
			# Get behavioral data
			behavioral_data = step_data.get("behavioral_data", {})
			
			# Validate minimum data requirements
			required_patterns = ["keystroke_dynamics", "mouse_movements"]
			missing_patterns = [p for p in required_patterns if p not in behavioral_data]
			
			if missing_patterns:
				return {
					"success": False,
					"error": "insufficient_behavioral_data",
					"message": "Insufficient behavioral data for baseline",
					"missing_patterns": missing_patterns,
					"collection_guidance": await self._get_behavioral_collection_guidance()
				}
			
			# Process behavioral enrollment
			enrollment_result = await self.biometric_service.enroll_biometric(
				session.user_id,
				session.tenant_id,
				"behavioral",
				behavioral_data,
				step_data.get("metadata", {})
			)
			
			if enrollment_result["success"]:
				session.biometric_methods["behavioral"] = enrollment_result
				
				return {
					"success": True,
					"message": "Behavioral baseline established successfully",
					"template_id": enrollment_result["template_id"],
					"quality_score": enrollment_result["quality_score"],
					"patterns_analyzed": len(behavioral_data)
				}
			else:
				return {
					"success": False,
					"error": enrollment_result.get("error", "behavioral_baseline_failed"),
					"message": enrollment_result.get("message", "Behavioral baseline establishment failed"),
					"guidance": await self._get_behavioral_improvement_guidance(enrollment_result)
				}
			
		except Exception as e:
			return {
				"success": False,
				"error": "behavioral_baseline_error",
				"message": str(e)
			}
	
	async def _process_backup_codes_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process backup codes generation"""
		try:
			# Generate backup codes using token service
			from .token_service import TokenService
			token_service = TokenService(self.db)
			
			backup_codes = await token_service.generate_backup_codes(
				session.user_id, session.tenant_id, count=10
			)
			
			# Store backup codes info in session
			session.enrollment_data["backup_codes_generated"] = True
			session.enrollment_data["backup_codes_count"] = len(backup_codes)
			
			# Send backup codes notification
			await self._send_enrollment_notification(
				session.user_id, session.tenant_id, "backup_codes_generated",
				"Backup codes generated", 
				f"Your {len(backup_codes)} backup recovery codes have been generated."
			)
			
			return {
				"success": True,
				"message": "Backup codes generated successfully",
				"backup_codes": backup_codes,
				"backup_codes_count": len(backup_codes),
				"security_warning": "Store these codes securely - they will not be shown again"
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "backup_codes_failed",
				"message": str(e)
			}
	
	async def _process_verification_test_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process verification test of enrolled biometrics"""
		try:
			test_results = {}
			overall_success = True
			
			# Test each enrolled biometric method
			for method_type, method_info in session.biometric_methods.items():
				if method_type in step_data.get("test_data", {}):
					test_data = step_data["test_data"][method_type]
					
					# Perform verification test
					verification_result = await self.biometric_service.verify_biometric(
						session.user_id,
						session.tenant_id,
						method_type,
						test_data.get("biometric_data"),
						method_info["template_id"],
						test_data.get("metadata", {})
					)
					
					test_results[method_type] = {
						"success": verification_result.get("success", False),
						"match_score": verification_result.get("match_score", 0.0),
						"confidence": verification_result.get("confidence_score", 0.0),
						"liveness_detected": verification_result.get("liveness_detected", False)
					}
					
					if not verification_result.get("success"):
						overall_success = False
			
			if overall_success and test_results:
				return {
					"success": True,
					"message": "Verification test passed successfully",
					"test_results": test_results,
					"avg_confidence": sum(r["confidence"] for r in test_results.values()) / len(test_results)
				}
			else:
				failed_methods = [method for method, result in test_results.items() if not result["success"]]
				
				return {
					"success": False,
					"error": "verification_test_failed",
					"message": "Verification test failed for some methods",
					"test_results": test_results,
					"failed_methods": failed_methods,
					"retry_guidance": await self._get_verification_retry_guidance(failed_methods)
				}
			
		except Exception as e:
			return {
				"success": False,
				"error": "verification_test_error",
				"message": str(e)
			}
	
	async def _process_completion_step(self, session: EnrollmentSession, step_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process enrollment completion"""
		try:
			# Finalize enrollment
			finalization_result = await self._finalize_enrollment(session)
			
			if finalization_result["success"]:
				# Send completion notification
				await self._send_enrollment_notification(
					session.user_id, session.tenant_id, "enrollment_completed",
					"Biometric enrollment completed", 
					"Your biometric enrollment has been completed successfully."
				)
				
				return {
					"success": True,
					"message": "Enrollment completed successfully",
					"enrolled_methods": list(session.biometric_methods.keys()),
					"user_profile_id": finalization_result["user_profile_id"],
					"summary": await self._generate_enrollment_summary(session)
				}
			else:
				return {
					"success": False,
					"error": "enrollment_finalization_failed",
					"message": finalization_result.get("message", "Failed to finalize enrollment")
				}
			
		except Exception as e:
			return {
				"success": False,
				"error": "completion_error",
				"message": str(e)
			}
	
	# Helper methods
	
	async def _assess_device_capabilities(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
		"""Assess device capabilities for biometric enrollment"""
		capabilities = {
			"camera_available": device_info.get("has_camera", False),
			"microphone_available": device_info.get("has_microphone", False),
			"touch_available": device_info.get("has_touch", False),
			"sensors_available": device_info.get("has_sensors", False),
			"secure_hardware": device_info.get("has_secure_element", False),
			"biometric_hardware": device_info.get("has_biometric_hardware", False)
		}
		
		# Assess quality levels
		capabilities["camera_quality"] = self._assess_camera_quality(device_info)
		capabilities["microphone_quality"] = self._assess_microphone_quality(device_info)
		
		return capabilities
	
	def _assess_camera_quality(self, device_info: Dict[str, Any]) -> str:
		"""Assess camera quality level"""
		resolution = device_info.get("camera_resolution", "")
		
		if "1080p" in resolution or "2K" in resolution or "4K" in resolution:
			return "high"
		elif "720p" in resolution:
			return "medium"
		else:
			return "low"
	
	def _assess_microphone_quality(self, device_info: Dict[str, Any]) -> str:
		"""Assess microphone quality level"""
		# Simplified assessment based on device type
		device_type = device_info.get("device_type", "").lower()
		
		if "professional" in device_type or "studio" in device_type:
			return "high"
		elif "smartphone" in device_type or "tablet" in device_type:
			return "medium"
		else:
			return "low"
	
	async def _generate_enrollment_plan(self, 
									   device_capabilities: Dict[str, Any], 
									   user_preferences: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		"""Generate personalized enrollment plan"""
		plan = {
			"recommended_methods": [],
			"optional_methods": [],
			"estimated_time_minutes": 0,
			"accessibility_options": [],
			"quality_requirements": {}
		}
		
		# Determine recommended methods based on capabilities
		if device_capabilities.get("camera_available"):
			plan["recommended_methods"].append({
				"type": "face",
				"priority": 1,
				"reason": "Face recognition provides excellent security and convenience"
			})
		
		if device_capabilities.get("microphone_available"):
			plan["recommended_methods"].append({
				"type": "voice",
				"priority": 2,
				"reason": "Voice recognition adds additional security layer"
			})
		
		# Always recommend backup codes
		plan["recommended_methods"].append({
			"type": "backup_codes",
			"priority": 3,
			"reason": "Backup codes ensure account recovery capability"
		})
		
		# Add optional methods
		if device_capabilities.get("touch_available"):
			plan["optional_methods"].append({
				"type": "behavioral",
				"reason": "Behavioral patterns provide continuous authentication"
			})
		
		# Estimate time
		plan["estimated_time_minutes"] = len(plan["recommended_methods"]) * 3 + 5
		
		# Add accessibility options
		if user_preferences and user_preferences.get("accessibility_needed"):
			plan["accessibility_options"] = [
				"voice_guidance",
				"larger_text",
				"high_contrast",
				"alternative_input_methods"
			]
		
		return plan
	
	def _estimate_enrollment_time(self, enrollment_plan: Dict[str, Any]) -> int:
		"""Estimate total enrollment time"""
		base_time = 5  # Introduction and setup
		method_time = len(enrollment_plan.get("recommended_methods", [])) * 3
		testing_time = 3
		completion_time = 2
		
		return base_time + method_time + testing_time + completion_time
	
	async def _get_step_instructions(self, session: EnrollmentSession, step: EnrollmentStep) -> Dict[str, Any]:
		"""Get instructions for specific enrollment step"""
		instructions = {
			EnrollmentStep.INTRODUCTION: {
				"title": "Welcome to Biometric Enrollment",
				"description": "Set up secure biometric authentication for your account",
				"actions": ["Read privacy policy", "Provide consent", "Continue"],
				"estimated_time": "2 minutes"
			},
			EnrollmentStep.DEVICE_VERIFICATION: {
				"title": "Device Security Verification",
				"description": "Verify your device meets security requirements",
				"actions": ["Allow device scanning", "Review security status"],
				"estimated_time": "1 minute"
			},
			EnrollmentStep.BIOMETRIC_SELECTION: {
				"title": "Choose Authentication Methods",
				"description": "Select which biometric methods to set up",
				"actions": ["Select face recognition", "Select voice recognition", "Continue"],
				"estimated_time": "1 minute"
			},
			EnrollmentStep.FACE_ENROLLMENT: {
				"title": "Face Recognition Setup",
				"description": "Capture your face for secure authentication",
				"actions": ["Position face in frame", "Follow movement prompts", "Complete capture"],
				"estimated_time": "3 minutes"
			},
			EnrollmentStep.VOICE_ENROLLMENT: {
				"title": "Voice Recognition Setup", 
				"description": "Record your voice for secure authentication",
				"actions": ["Speak passphrase clearly", "Repeat if needed", "Complete recording"],
				"estimated_time": "3 minutes"
			},
			EnrollmentStep.BEHAVIORAL_BASELINE: {
				"title": "Behavioral Pattern Learning",
				"description": "Establish your unique interaction patterns",
				"actions": ["Use device normally", "Type sample text", "Move mouse naturally"],
				"estimated_time": "5 minutes"
			},
			EnrollmentStep.BACKUP_CODES: {
				"title": "Generate Backup Codes",
				"description": "Create recovery codes for account access",
				"actions": ["Generate codes", "Save codes securely", "Confirm storage"],
				"estimated_time": "2 minutes"
			},
			EnrollmentStep.VERIFICATION_TEST: {
				"title": "Test Your Setup",
				"description": "Verify your biometric authentication works",
				"actions": ["Test face recognition", "Test voice recognition", "Confirm success"],
				"estimated_time": "3 minutes"
			},
			EnrollmentStep.COMPLETION: {
				"title": "Enrollment Complete",
				"description": "Your biometric authentication is ready",
				"actions": ["Review summary", "Complete setup"],
				"estimated_time": "1 minute"
			}
		}
		
		return instructions.get(step, {})
	
	def _get_next_step(self, session: EnrollmentSession) -> Optional[EnrollmentStep]:
		"""Determine next enrollment step"""
		steps_order = [
			EnrollmentStep.INTRODUCTION,
			EnrollmentStep.DEVICE_VERIFICATION,
			EnrollmentStep.BIOMETRIC_SELECTION,
			EnrollmentStep.FACE_ENROLLMENT,
			EnrollmentStep.VOICE_ENROLLMENT,
			EnrollmentStep.BEHAVIORAL_BASELINE,
			EnrollmentStep.BACKUP_CODES,
			EnrollmentStep.VERIFICATION_TEST,
			EnrollmentStep.COMPLETION
		]
		
		for step in steps_order:
			if not session.is_step_completed(step):
				return step
		
		return None  # All steps completed
	
	def _optimize_enrollment_order(self, selected_methods: List[str]) -> List[str]:
		"""Optimize enrollment order for best user experience"""
		# Prioritize easier methods first
		priority_order = ["face", "voice", "behavioral"]
		
		optimized_order = []
		for method in priority_order:
			if method in selected_methods:
				optimized_order.append(method)
		
		# Add any remaining methods
		for method in selected_methods:
			if method not in optimized_order:
				optimized_order.append(method)
		
		return optimized_order
	
	async def _generate_enrollment_feedback(self, 
										   method_type: str, 
										   enrollment_result: Dict[str, Any], 
										   attempt_number: int) -> Dict[str, Any]:
		"""Generate specific feedback for enrollment improvement"""
		feedback = {
			"method": method_type,
			"attempt": attempt_number,
			"issues": [],
			"improvements": [],
			"tips": []
		}
		
		error = enrollment_result.get("error", "")
		
		if method_type == "face":
			if "poor_quality" in error:
				feedback["issues"].append("Image quality too low")
				feedback["improvements"].append("Ensure good lighting")
				feedback["improvements"].append("Clean camera lens")
				feedback["tips"].append("Face camera directly")
			elif "liveness_failed" in error:
				feedback["issues"].append("Liveness detection failed")
				feedback["improvements"].append("Blink naturally during capture")
				feedback["improvements"].append("Move head slightly")
				feedback["tips"].append("Ensure you're not using a photo")
		
		elif method_type == "voice":
			if "poor_quality" in error:
				feedback["issues"].append("Audio quality too low")
				feedback["improvements"].append("Speak clearly and loudly")
				feedback["improvements"].append("Reduce background noise")
				feedback["tips"].append("Hold device closer to mouth")
			elif "liveness_failed" in error:
				feedback["issues"].append("Voice liveness detection failed")
				feedback["improvements"].append("Speak naturally")
				feedback["improvements"].append("Vary your tone slightly")
				feedback["tips"].append("Don't use recorded audio")
		
		return feedback
	
	async def _get_quality_improvement_tips(self, session: EnrollmentSession) -> Dict[str, Any]:
		"""Get quality improvement tips"""
		return {
			"face": [
				"Ensure good, even lighting on your face",
				"Look directly at the camera",
				"Remove glasses if they cause glare",
				"Keep a neutral expression",
				"Stay still during capture"
			],
			"voice": [
				"Speak clearly and at normal volume",
				"Reduce background noise",
				"Hold device 6-12 inches from your mouth",
				"Speak naturally, don't whisper or shout",
				"Complete the full passphrase"
			],
			"general": [
				"Use a stable internet connection",
				"Ensure device has sufficient battery",
				"Close other apps during enrollment",
				"Find a quiet, well-lit environment"
			]
		}
	
	async def _get_troubleshooting_guidance(self, session: EnrollmentSession) -> Dict[str, Any]:
		"""Get troubleshooting guidance"""
		return {
			"common_issues": [
				{
					"issue": "Camera not working",
					"solutions": ["Check camera permissions", "Restart app", "Clean camera lens"]
				},
				{
					"issue": "Microphone not working", 
					"solutions": ["Check microphone permissions", "Reduce background noise", "Restart app"]
				},
				{
					"issue": "Poor image quality",
					"solutions": ["Improve lighting", "Clean camera", "Move closer to camera"]
				}
			],
			"contact_support": {
				"available": True,
				"methods": ["email", "chat", "phone"],
				"hours": "24/7"
			}
		}
	
	async def _get_accessibility_guidance(self, session: EnrollmentSession) -> Dict[str, Any]:
		"""Get accessibility guidance"""
		return {
			"visual_impairment": [
				"Voice guidance available",
				"High contrast mode supported",
				"Screen reader compatible",
				"Large text options available"
			],
			"hearing_impairment": [
				"Visual feedback provided",
				"Text instructions available",
				"Vibration feedback supported"
			],
			"motor_impairment": [
				"Voice control options",
				"Extended time limits",
				"Alternative input methods",
				"Assistance mode available"
			]
		}
	
	async def _get_privacy_information(self, session: EnrollmentSession) -> Dict[str, Any]:
		"""Get privacy information"""
		return {
			"data_collection": [
				"Biometric templates are encrypted",
				"Raw biometric data is not stored",
				"Data is processed locally when possible",
				"Templates cannot be reverse-engineered"
			],
			"data_usage": [
				"Used only for authentication",
				"Not shared with third parties",
				"Not used for advertising",
				"User controls data retention"
			],
			"data_protection": [
				"End-to-end encryption",
				"Secure storage",
				"Regular security audits",
				"Compliance with privacy laws"
			]
		}
	
	async def _get_behavioral_collection_guidance(self) -> Dict[str, Any]:
		"""Get guidance for behavioral data collection"""
		return {
			"typing_patterns": [
				"Type naturally as you normally would",
				"Include typical typing errors and corrections",
				"Use your regular typing speed",
				"Type at least 100 characters"
			],
			"mouse_movements": [
				"Move mouse naturally around screen",
				"Include typical clicking patterns",
				"Use scrolling and selection gestures",
				"Interact for at least 2 minutes"
			],
			"interaction_patterns": [
				"Use applications as you normally would",
				"Include pauses and hesitations",
				"Show typical usage patterns",
				"Complete realistic tasks"
			]
		}
	
	async def _get_behavioral_improvement_guidance(self, enrollment_result: Dict[str, Any]) -> Dict[str, Any]:
		"""Get guidance for improving behavioral data"""
		return {
			"data_quality": [
				"Increase interaction time",
				"Use more varied input patterns",
				"Include different types of actions",
				"Ensure natural interaction rhythm"
			],
			"consistency": [
				"Maintain consistent typing style",
				"Use typical mouse movement patterns",
				"Show regular interaction habits",
				"Avoid artificial behaviors"
			]
		}
	
	async def _get_verification_retry_guidance(self, failed_methods: List[str]) -> Dict[str, Any]:
		"""Get guidance for retrying verification"""
		guidance = {}
		
		for method in failed_methods:
			if method == "face":
				guidance[method] = [
					"Ensure same lighting conditions as enrollment",
					"Position face in same way as during enrollment",
					"Look directly at camera",
					"Blink naturally during verification"
				]
			elif method == "voice":
				guidance[method] = [
					"Speak the same passphrase as enrollment",
					"Use similar volume and tone",
					"Reduce background noise",
					"Speak clearly and naturally"
				]
		
		return guidance
	
	def _is_session_expired(self, session: EnrollmentSession) -> bool:
		"""Check if enrollment session has expired"""
		elapsed_time = datetime.utcnow() - session.started_at
		return elapsed_time.total_seconds() > (self.enrollment_timeout_minutes * 60)
	
	async def _cleanup_session(self, session_id: str) -> None:
		"""Clean up expired enrollment session"""
		if session_id in self.active_sessions:
			del self.active_sessions[session_id]
	
	async def _finalize_enrollment(self, session: EnrollmentSession) -> Dict[str, Any]:
		"""Finalize enrollment and create user profile"""
		try:
			# Create or update MFA user profile
			user_profile = await self._create_or_update_user_profile(session)
			
			# Create MFA method records
			await self._create_mfa_method_records(session, user_profile)
			
			# Clean up session
			await self._cleanup_session(session.session_id)
			
			return {
				"success": True,
				"user_profile_id": user_profile.id,
				"enrolled_methods": list(session.biometric_methods.keys())
			}
			
		except Exception as e:
			self.logger.error(f"Enrollment finalization error: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "finalization_failed",
				"message": str(e)
			}
	
	async def _generate_enrollment_summary(self, session: EnrollmentSession) -> Dict[str, Any]:
		"""Generate enrollment completion summary"""
		enrolled_methods = list(session.biometric_methods.keys())
		
		summary = {
			"enrollment_id": session.session_id,
			"completion_time": datetime.utcnow().isoformat(),
			"duration_minutes": (datetime.utcnow() - session.started_at).total_seconds() / 60,
			"enrolled_methods": enrolled_methods,
			"method_count": len(enrolled_methods),
			"backup_codes_generated": session.enrollment_data.get("backup_codes_generated", False),
			"security_level": self._calculate_security_level(enrolled_methods),
			"next_steps": [
				"Test authentication in a real scenario",
				"Save backup codes securely",
				"Review security settings"
			]
		}
		
		return summary
	
	def _calculate_security_level(self, enrolled_methods: List[str]) -> str:
		"""Calculate overall security level based on enrolled methods"""
		method_scores = {
			"face": 3,
			"voice": 2,
			"behavioral": 2,
			"backup_codes": 1
		}
		
		total_score = sum(method_scores.get(method, 0) for method in enrolled_methods)
		
		if total_score >= 6:
			return "high"
		elif total_score >= 4:
			return "medium"
		else:
			return "low"
	
	async def _send_enrollment_notification(self, 
										   user_id: str, 
										   tenant_id: str, 
										   notification_type: str,
										   title: str, 
										   message: str) -> None:
		"""Send enrollment notification"""
		try:
			notification_request = create_mfa_notification(
				user_id, notification_type, title, message, tenant_id
			)
			
			await self.apg_router.route_integration_event(notification_request)
			
		except Exception as e:
			self.logger.error(f"Failed to send enrollment notification: {str(e)}")
	
	# Database operations (placeholders)
	
	async def _store_device_binding(self, device_binding: DeviceBinding) -> None:
		"""Store device binding in database"""
		# Implementation depends on database client
		pass
	
	async def _create_or_update_user_profile(self, session: EnrollmentSession) -> MFAUserProfile:
		"""Create or update MFA user profile"""
		# Implementation depends on database client
		return MFAUserProfile(
			user_id=session.user_id,
			tenant_id=session.tenant_id,
			created_by=session.user_id,
			updated_by=session.user_id
		)
	
	async def _create_mfa_method_records(self, session: EnrollmentSession, user_profile: MFAUserProfile) -> None:
		"""Create MFA method records"""
		# Implementation depends on database client
		pass


__all__ = [
	"BiometricEnrollmentWizard",
	"EnrollmentSession",
	"EnrollmentStep"
]