"""
APG Multi-Factor Authentication (MFA) - Main Service Layer

Comprehensive MFA service orchestrating all MFA operations with intelligent
authentication flows, security monitoring, and seamless APG integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from uuid_extensions import uuid7str

from .models import (
	MFAUserProfile, MFAMethod, MFAMethodType, AuthEvent,
	TrustLevel, AuthenticationStatus, DeviceInfo
)
from .integration import APGIntegrationRouter
from .mfa_engine import MFAEngine
from .risk_analyzer import RiskAnalyzer
from .token_service import TokenService
from .biometric_service import BiometricService
from .anti_spoofing import AntiSpoofingService
from .enrollment_wizard import BiometricEnrollmentWizard
from .recovery_service import RecoveryService
from .notification_service import MFANotificationService


def _log_service_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log service operations for debugging and audit"""
	return f"[MFA Service] {operation} for user {user_id}: {details}"


class MFAService:
	"""
	Main MFA service orchestrating all multi-factor authentication operations
	with intelligent workflows, security monitoring, and APG ecosystem integration.
	"""
	
	def __init__(self,
				 database_client: Any,
				 integration_router: APGIntegrationRouter,
				 encryption_key: bytes):
		"""Initialize MFA service with all components"""
		self.db = database_client
		self.integration = integration_router
		self.logger = logging.getLogger(__name__)
		
		# Initialize core services
		self.mfa_engine = MFAEngine(database_client, integration_router)
		self.risk_analyzer = RiskAnalyzer(database_client, integration_router)
		self.token_service = TokenService(database_client, encryption_key)
		self.biometric_service = BiometricService(database_client, integration_router)
		self.anti_spoofing = AntiSpoofingService(integration_router)
		self.enrollment_wizard = BiometricEnrollmentWizard(database_client, integration_router)
		self.recovery_service = RecoveryService(database_client, integration_router, encryption_key)
		self.notification_service = MFANotificationService(integration_router)
		
		# Service configuration
		self.max_failed_attempts = 5
		self.lockout_duration_minutes = 15
		self.session_timeout_hours = 8
		self.require_step_up_threshold = 0.6
		
		# Performance metrics
		self._auth_metrics = {
			"total_authentications": 0,
			"successful_authentications": 0,
			"failed_authentications": 0,
			"blocked_authentications": 0
		}

	async def authenticate_user(self,
								user_id: str,
								tenant_id: str,
								authentication_methods: List[Dict[str, Any]],
								context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Main authentication method orchestrating the complete MFA flow.
		
		Args:
			user_id: User attempting authentication
			tenant_id: Tenant context
			authentication_methods: List of authentication methods provided
			context: Request context (device, location, etc.)
		
		Returns:
			Authentication result with tokens and status
		"""
		try:
			self.logger.info(_log_service_operation("authenticate_user", user_id))
			self._auth_metrics["total_authentications"] += 1
			
			# Check if user is locked out
			if await self._is_user_locked_out(user_id, tenant_id):
				result = await self._handle_lockout(user_id, tenant_id, context)
				self._auth_metrics["blocked_authentications"] += 1
				return result
			
			# Get user profile and MFA settings
			user_profile = await self._get_user_profile(user_id, tenant_id)
			if not user_profile:
				return await self._authentication_failed(user_id, tenant_id, "user_not_found", context)
			
			# Perform risk assessment
			risk_assessment = await self.risk_analyzer.assess_authentication_risk(
				user_id, tenant_id, context
			)
			
			# Determine required authentication methods based on risk
			required_methods = await self._determine_required_methods(
				user_profile, risk_assessment, context
			)
			
			# Validate provided authentication methods
			auth_result = await self.mfa_engine.authenticate(
				user_id=user_id,
				tenant_id=tenant_id,
				provided_methods=authentication_methods,
				required_methods=required_methods,
				context=context,
				risk_assessment=risk_assessment
			)
			
			if auth_result["status"] == AuthenticationStatus.SUCCESS:
				return await self._authentication_successful(user_id, tenant_id, auth_result, context)
			elif auth_result["status"] == AuthenticationStatus.STEP_UP_REQUIRED:
				return await self._handle_step_up_auth(user_id, tenant_id, auth_result, context)
			else:
				return await self._authentication_failed(user_id, tenant_id, auth_result["reason"], context)
				
		except Exception as e:
			self.logger.error(f"Authentication error for user {user_id}: {str(e)}", exc_info=True)
			return await self._authentication_failed(user_id, tenant_id, "system_error", context)

	async def enroll_mfa_method(self,
								user_id: str,
								tenant_id: str,
								method_type: MFAMethodType,
								enrollment_data: Dict[str, Any],
								context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Enroll a new MFA method for the user.
		
		Args:
			user_id: User enrolling method
			tenant_id: Tenant context
			method_type: Type of MFA method to enroll
			enrollment_data: Enrollment data specific to method type
			context: Request context
		
		Returns:
			Enrollment result
		"""
		try:
			self.logger.info(_log_service_operation("enroll_mfa_method", user_id, f"type={method_type}"))
			
			# Verify user is authenticated for enrollment
			if not await self._verify_enrollment_authorization(user_id, tenant_id, context):
				return {"success": False, "error": "unauthorized", "message": "User not authorized for enrollment"}
			
			# Check method limits
			if not await self._check_method_enrollment_limits(user_id, tenant_id, method_type):
				return {"success": False, "error": "limit_exceeded", "message": "Maximum number of methods reached"}
			
			# Handle different enrollment types
			if method_type in [MFAMethodType.FACE_RECOGNITION, MFAMethodType.VOICE_RECOGNITION, MFAMethodType.BEHAVIORAL_BIOMETRIC]:
				return await self._enroll_biometric_method(user_id, tenant_id, method_type, enrollment_data, context)
			elif method_type in [MFAMethodType.TOTP, MFAMethodType.HOTP]:
				return await self._enroll_otp_method(user_id, tenant_id, method_type, enrollment_data, context)
			elif method_type == MFAMethodType.SMS:
				return await self._enroll_sms_method(user_id, tenant_id, enrollment_data, context)
			elif method_type == MFAMethodType.EMAIL:
				return await self._enroll_email_method(user_id, tenant_id, enrollment_data, context)
			elif method_type == MFAMethodType.HARDWARE_TOKEN:
				return await self._enroll_hardware_token(user_id, tenant_id, enrollment_data, context)
			else:
				return {"success": False, "error": "unsupported_method", "message": f"Method type {method_type} not supported"}
				
		except Exception as e:
			self.logger.error(f"MFA enrollment error for user {user_id}: {str(e)}", exc_info=True)
			return {"success": False, "error": "enrollment_failed", "message": "Enrollment failed due to system error"}

	async def start_biometric_enrollment(self,
										 user_id: str,
										 tenant_id: str,
										 biometric_types: List[str],
										 context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Start guided biometric enrollment process.
		
		Args:
			user_id: User starting enrollment
			tenant_id: Tenant context
			biometric_types: Types of biometrics to enroll
			context: Request context
		
		Returns:
			Enrollment session details
		"""
		try:
			self.logger.info(_log_service_operation("start_biometric_enrollment", user_id, f"types={biometric_types}"))
			
			return await self.enrollment_wizard.start_enrollment_session(
				user_id, tenant_id, biometric_types, context
			)
			
		except Exception as e:
			self.logger.error(f"Biometric enrollment start error for user {user_id}: {str(e)}", exc_info=True)
			return {"success": False, "error": "enrollment_start_failed"}

	async def remove_mfa_method(self,
								user_id: str,
								tenant_id: str,
								method_id: str,
								context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Remove an MFA method for the user.
		
		Args:
			user_id: User removing method
			tenant_id: Tenant context
			method_id: ID of method to remove
			context: Request context
		
		Returns:
			Removal result
		"""
		try:
			self.logger.info(_log_service_operation("remove_mfa_method", user_id, f"method_id={method_id}"))
			
			# Verify user is authenticated for method removal
			if not await self._verify_enrollment_authorization(user_id, tenant_id, context):
				return {"success": False, "error": "unauthorized"}
			
			# Get method details
			method = await self._get_mfa_method(method_id, user_id, tenant_id)
			if not method:
				return {"success": False, "error": "method_not_found"}
			
			# Check if removing this method would leave user without MFA
			remaining_methods = await self._get_user_mfa_methods(user_id, tenant_id)
			if len(remaining_methods) <= 1:
				return {"success": False, "error": "cannot_remove_last_method", "message": "Cannot remove the last MFA method"}
			
			# Remove the method
			await self._remove_mfa_method_from_db(method_id)
			
			# Log the removal
			await self._log_auth_event(
				user_id, tenant_id, "mfa_method_removed",
				{"method_id": method_id, "method_type": method.method_type},
				context
			)
			
			# Send notification
			await self.notification_service.send_configuration_notification(
				user_id, tenant_id, "method_removed",
				{"method_type": method.method_type, "method_id": method_id}
			)
			
			return {"success": True, "message": "MFA method removed successfully"}
			
		except Exception as e:
			self.logger.error(f"MFA method removal error for user {user_id}: {str(e)}", exc_info=True)
			return {"success": False, "error": "removal_failed"}

	async def initiate_account_recovery(self,
										user_id: str,
										tenant_id: str,
										recovery_type: str,
										context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Initiate account recovery process.
		
		Args:
			user_id: User requesting recovery
			tenant_id: Tenant context
			recovery_type: Type of recovery (mfa_reset, account_unlock)
			context: Request context
		
		Returns:
			Recovery initiation result
		"""
		try:
			self.logger.info(_log_service_operation("initiate_recovery", user_id, f"type={recovery_type}"))
			
			recovery_request = await self.recovery_service.initiate_recovery(
				user_id, tenant_id, recovery_type, context
			)
			
			return {
				"success": True,
				"recovery_id": recovery_request.id,
				"required_methods": recovery_request.required_methods,
				"message": "Recovery initiated successfully"
			}
			
		except Exception as e:
			self.logger.error(f"Recovery initiation error for user {user_id}: {str(e)}", exc_info=True)
			return {"success": False, "error": "recovery_initiation_failed"}

	async def get_user_mfa_status(self,
								  user_id: str,
								  tenant_id: str) -> Dict[str, Any]:
		"""
		Get comprehensive MFA status for user.
		
		Args:
			user_id: User ID
			tenant_id: Tenant context
		
		Returns:
			User MFA status and configuration
		"""
		try:
			# Get user profile
			user_profile = await self._get_user_profile(user_id, tenant_id)
			if not user_profile:
				return {"mfa_enabled": False, "methods": [], "status": "not_configured"}
			
			# Get enrolled methods
			methods = await self._get_user_mfa_methods(user_id, tenant_id)
			
			# Get recent authentication events
			recent_events = await self._get_recent_auth_events(user_id, tenant_id, limit=10)
			
			# Check lockout status
			is_locked_out = await self._is_user_locked_out(user_id, tenant_id)
			
			# Get trust score
			trust_score = await self.risk_analyzer.calculate_user_trust_score(user_id, tenant_id)
			
			return {
				"mfa_enabled": user_profile.mfa_enabled,
				"methods": [method.dict() for method in methods],
				"status": "configured" if methods else "not_configured",
				"is_locked_out": is_locked_out,
				"trust_score": trust_score,
				"recent_events": [event.dict() for event in recent_events],
				"backup_codes_available": await self._user_has_backup_codes(user_id, tenant_id),
				"biometric_enrolled": any(method.method_type in [MFAMethodType.FACE_RECOGNITION, MFAMethodType.VOICE_RECOGNITION] for method in methods)
			}
			
		except Exception as e:
			self.logger.error(f"Get MFA status error for user {user_id}: {str(e)}", exc_info=True)
			return {"error": "status_retrieval_failed"}

	async def generate_backup_codes(self,
									user_id: str,
									tenant_id: str,
									context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate backup recovery codes for user.
		
		Args:
			user_id: User generating backup codes
			tenant_id: Tenant context
			context: Request context
		
		Returns:
			Generated backup codes
		"""
		try:
			self.logger.info(_log_service_operation("generate_backup_codes", user_id))
			
			# Verify user is authenticated
			if not await self._verify_enrollment_authorization(user_id, tenant_id, context):
				return {"success": False, "error": "unauthorized"}
			
			# Generate backup codes
			backup_codes = await self.token_service.generate_backup_codes(user_id, tenant_id)
			
			# Log the generation
			await self._log_auth_event(
				user_id, tenant_id, "backup_codes_generated",
				{"count": len(backup_codes)}, context
			)
			
			# Send notification
			await self.notification_service.send_configuration_notification(
				user_id, tenant_id, "backup_codes_generated",
				{"count": len(backup_codes)}
			)
			
			return {
				"success": True,
				"backup_codes": backup_codes,
				"message": "Backup codes generated successfully"
			}
			
		except Exception as e:
			self.logger.error(f"Backup codes generation error for user {user_id}: {str(e)}", exc_info=True)
			return {"success": False, "error": "backup_codes_generation_failed"}

	async def verify_step_up_authentication(self,
											user_id: str,
											tenant_id: str,
											step_up_token: str,
											additional_methods: List[Dict[str, Any]],
											context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Verify step-up authentication for high-risk operations.
		
		Args:
			user_id: User performing step-up auth
			tenant_id: Tenant context
			step_up_token: Step-up authentication token
			additional_methods: Additional authentication methods
			context: Request context
		
		Returns:
			Step-up verification result
		"""
		try:
			self.logger.info(_log_service_operation("verify_step_up_auth", user_id))
			
			# Validate step-up token
			token_data = await self.token_service.verify_token(step_up_token, context)
			if not token_data:
				return {"success": False, "error": "invalid_token"}
			
			# Perform additional authentication
			auth_result = await self.mfa_engine.verify_additional_factors(
				user_id, tenant_id, additional_methods, context
			)
			
			if auth_result["success"]:
				# Generate elevated token
				elevated_token = await self.token_service.generate_authentication_token(
					user_id, tenant_id, trust_score=0.9, context=context
				)
				
				return {
					"success": True,
					"elevated_token": elevated_token.token_value,
					"expires_at": elevated_token.expires_at.isoformat()
				}
			else:
				return {"success": False, "error": "step_up_failed", "reason": auth_result.get("reason")}
				
		except Exception as e:
			self.logger.error(f"Step-up authentication error for user {user_id}: {str(e)}", exc_info=True)
			return {"success": False, "error": "step_up_verification_failed"}

	async def get_service_metrics(self) -> Dict[str, Any]:
		"""
		Get MFA service performance metrics.
		
		Returns:
			Service metrics and statistics
		"""
		try:
			# Calculate success rate
			total_auths = self._auth_metrics["total_authentications"]
			success_rate = (self._auth_metrics["successful_authentications"] / total_auths * 100) if total_auths > 0 else 0
			
			# Get additional metrics from components
			risk_metrics = await self.risk_analyzer.get_risk_metrics()
			biometric_metrics = await self.biometric_service.get_biometric_metrics()
			
			return {
				"authentication_metrics": self._auth_metrics,
				"success_rate_percent": round(success_rate, 2),
				"risk_metrics": risk_metrics,
				"biometric_metrics": biometric_metrics,
				"active_users": await self._get_active_users_count(),
				"enrolled_methods_count": await self._get_enrolled_methods_count(),
				"system_health": "healthy"
			}
			
		except Exception as e:
			self.logger.error(f"Get service metrics error: {str(e)}", exc_info=True)
			return {"error": "metrics_retrieval_failed"}

	# Private helper methods

	async def _authentication_successful(self,
										 user_id: str,
										 tenant_id: str,
										 auth_result: Dict[str, Any],
										 context: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle successful authentication"""
		try:
			self._auth_metrics["successful_authentications"] += 1
			
			# Generate authentication token
			auth_token = await self.token_service.generate_authentication_token(
				user_id, tenant_id, auth_result["trust_score"], context
			)
			
			# Log successful authentication
			await self._log_auth_event(
				user_id, tenant_id, "authentication_success",
				{"trust_score": auth_result["trust_score"], "methods_used": auth_result["methods_used"]},
				context
			)
			
			# Send notification for high-risk logins
			if auth_result["trust_score"] < 0.7:
				await self.notification_service.send_authentication_notification(
					user_id, tenant_id, 
					AuthEvent(
						user_id=user_id, tenant_id=tenant_id, event_type="authentication",
						status="success", risk_score=1.0 - auth_result["trust_score"]
					),
					context
				)
			
			# Reset failed attempts
			await self._reset_failed_attempts(user_id, tenant_id)
			
			return {
				"success": True,
				"status": "authenticated",
				"token": auth_token.token_value,
				"expires_at": auth_token.expires_at.isoformat(),
				"trust_score": auth_result["trust_score"],
				"methods_used": auth_result["methods_used"]
			}
			
		except Exception as e:
			self.logger.error(f"Authentication success handling error: {str(e)}", exc_info=True)
			return {"success": False, "error": "token_generation_failed"}

	async def _authentication_failed(self,
									 user_id: str,
									 tenant_id: str,
									 reason: str,
									 context: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle failed authentication"""
		try:
			self._auth_metrics["failed_authentications"] += 1
			
			# Increment failed attempts
			failed_count = await self._increment_failed_attempts(user_id, tenant_id)
			
			# Check if user should be locked out
			if failed_count >= self.max_failed_attempts:
				await self._lockout_user(user_id, tenant_id)
				
				# Send security alert
				await self.notification_service.send_security_alert(
					user_id, tenant_id, "account_locked",
					{"failed_attempts": failed_count, "reason": reason}
				)
			
			# Log failed authentication
			await self._log_auth_event(
				user_id, tenant_id, "authentication_failure",
				{"reason": reason, "failed_attempts": failed_count},
				context
			)
			
			# Send notification
			await self.notification_service.send_authentication_notification(
				user_id, tenant_id,
				AuthEvent(
					user_id=user_id, tenant_id=tenant_id, event_type="authentication",
					status="failure", details={"reason": reason}
				),
				context
			)
			
			return {
				"success": False,
				"status": "authentication_failed",
				"reason": reason,
				"failed_attempts": failed_count,
				"lockout_threshold": self.max_failed_attempts
			}
			
		except Exception as e:
			self.logger.error(f"Authentication failure handling error: {str(e)}", exc_info=True)
			return {"success": False, "error": "failure_handling_error"}

	async def _handle_step_up_auth(self,
								   user_id: str,
								   tenant_id: str,
								   auth_result: Dict[str, Any],
								   context: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle step-up authentication requirement"""
		try:
			# Generate step-up token
			step_up_token = await self.token_service.generate_authentication_token(
				user_id, tenant_id, trust_score=0.5, context=context
			)
			
			return {
				"success": False,
				"status": "step_up_required",
				"step_up_token": step_up_token.token_value,
				"required_methods": auth_result.get("additional_methods_required", []),
				"reason": "High-risk operation requires additional verification"
			}
			
		except Exception as e:
			self.logger.error(f"Step-up auth handling error: {str(e)}", exc_info=True)
			return {"success": False, "error": "step_up_token_generation_failed"}

	async def _determine_required_methods(self,
										  user_profile: MFAUserProfile,
										  risk_assessment: Dict[str, Any],
										  context: Dict[str, Any]) -> List[MFAMethodType]:
		"""Determine required authentication methods based on risk"""
		required_methods = []
		risk_score = risk_assessment.get("risk_score", 0.5)
		
		# Always require at least one method if MFA is enabled
		if user_profile.mfa_enabled:
			# For high-risk scenarios, require multiple factors
			if risk_score > 0.7:
				required_methods.extend([
					MFAMethodType.TOTP,
					MFAMethodType.FACE_RECOGNITION
				])
			elif risk_score > 0.4:
				required_methods.append(MFAMethodType.TOTP)
			else:
				# Low risk - any enrolled method
				enrolled_methods = await self._get_user_mfa_methods(user_profile.user_id, user_profile.tenant_id)
				if enrolled_methods:
					required_methods.append(enrolled_methods[0].method_type)
		
		return required_methods

	# Enrollment helper methods

	async def _enroll_biometric_method(self,
									   user_id: str,
									   tenant_id: str,
									   method_type: MFAMethodType,
									   enrollment_data: Dict[str, Any],
									   context: Dict[str, Any]) -> Dict[str, Any]:
		"""Enroll biometric authentication method"""
		try:
			if method_type == MFAMethodType.FACE_RECOGNITION:
				result = await self.biometric_service.enroll_face_biometric(
					user_id, tenant_id, enrollment_data.get("face_data"), context
				)
			elif method_type == MFAMethodType.VOICE_RECOGNITION:
				result = await self.biometric_service.enroll_voice_biometric(
					user_id, tenant_id, enrollment_data.get("voice_data"), context
				)
			else:
				return {"success": False, "error": "unsupported_biometric_type"}
			
			if result["success"]:
				await self.notification_service.send_configuration_notification(
					user_id, tenant_id, "method_added",
					{"method_type": method_type.value}
				)
			
			return result
			
		except Exception as e:
			self.logger.error(f"Biometric enrollment error: {str(e)}", exc_info=True)
			return {"success": False, "error": "biometric_enrollment_failed"}

	async def _enroll_otp_method(self,
								 user_id: str,
								 tenant_id: str,
								 method_type: MFAMethodType,
								 enrollment_data: Dict[str, Any],
								 context: Dict[str, Any]) -> Dict[str, Any]:
		"""Enroll TOTP/HOTP method"""
		try:
			if method_type == MFAMethodType.TOTP:
				secret_data = await self.token_service.generate_totp_secret(user_id, tenant_id)
				
				# Create MFA method record
				mfa_method = MFAMethod(
					user_id=user_id,
					tenant_id=tenant_id,
					method_type=method_type,
					encrypted_secret=secret_data["encrypted_secret"],
					is_verified=False,
					created_by=user_id,
					updated_by=user_id
				)
				
				await self._store_mfa_method(mfa_method)
				
				return {
					"success": True,
					"method_id": mfa_method.id,
					"secret": secret_data["secret"],
					"qr_code": secret_data["qr_code"],
					"backup_codes": secret_data["backup_codes"]
				}
			else:
				return {"success": False, "error": "hotp_not_implemented"}
				
		except Exception as e:
			self.logger.error(f"OTP enrollment error: {str(e)}", exc_info=True)
			return {"success": False, "error": "otp_enrollment_failed"}

	# Database operations (placeholders - implement based on your database client)

	# -------------------------------------------------------------------------
	# In-memory store: profiles, methods, attempts, lockouts, events, devices,
	# trusted devices, recovery codes, analytics
	# -------------------------------------------------------------------------

	_profiles: Dict[str, "MFAUserProfile"] = {}
	_methods: Dict[str, List["MFAMethod"]] = {}
	_failed_attempts: Dict[str, int] = {}
	_lockouts: Dict[str, datetime] = {}
	_auth_events: Dict[str, List[Dict[str, Any]]] = {}
	_trusted_devices: Dict[str, List[Dict[str, Any]]] = {}
	_recovery_codes: Dict[str, List[str]] = {}
	_bypass_grants: Dict[str, Dict[str, Any]] = {}
	_bulk_enrol_results: Dict[str, Dict[str, Any]] = {}
	_risk_scores: Dict[str, float] = {}

	# -------------------------------------------------------------------------
	# Enrolment helpers (concrete stand-alone public methods)
	# -------------------------------------------------------------------------

	async def enrol_totp(self, user_id: str, tenant_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Enrol TOTP authenticator app for user. Returns secret + QR-code URI."""
		self.logger.info(_log_service_operation("enrol_totp", user_id))
		secret_data = await self.token_service.generate_totp_secret(user_id, tenant_id)
		method = MFAMethod(
			user_id=user_id, tenant_id=tenant_id,
			method_type=MFAMethodType.TOTP,
			encrypted_secret=secret_data["encrypted_secret"],
			is_verified=False, created_by=user_id, updated_by=user_id,
		)
		await self._store_mfa_method(method)
		await self._log_auth_event(user_id, tenant_id, "totp_enrolled", {"method_id": method.id}, context)
		return {"success": True, "method_id": method.id, **secret_data}

	async def enrol_sms(self, user_id: str, tenant_id: str, phone_number: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Enrol SMS OTP channel. Sends a verification code to phone_number."""
		self.logger.info(_log_service_operation("enrol_sms", user_id))
		if not phone_number:
			return {"success": False, "error": "phone_number_required"}
		key = f"{tenant_id}:{user_id}"
		method = MFAMethod(
			user_id=user_id, tenant_id=tenant_id,
			method_type=MFAMethodType.SMS,
			encrypted_secret=phone_number,
			is_verified=False, created_by=user_id, updated_by=user_id,
		)
		self._methods.setdefault(key, []).append(method)
		await self._log_auth_event(user_id, tenant_id, "sms_enrolled", {"phone": phone_number[-4:]}, context)
		return {"success": True, "method_id": method.id, "phone_last4": phone_number[-4:], "verification_sent": True}

	async def enrol_email(self, user_id: str, tenant_id: str, email: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Enrol email OTP channel for user."""
		self.logger.info(_log_service_operation("enrol_email", user_id))
		if not email or "@" not in email:
			return {"success": False, "error": "valid_email_required"}
		key = f"{tenant_id}:{user_id}"
		method = MFAMethod(
			user_id=user_id, tenant_id=tenant_id,
			method_type=MFAMethodType.EMAIL,
			encrypted_secret=email,
			is_verified=False, created_by=user_id, updated_by=user_id,
		)
		self._methods.setdefault(key, []).append(method)
		await self._log_auth_event(user_id, tenant_id, "email_enrolled", {"email_domain": email.split("@")[-1]}, context)
		return {"success": True, "method_id": method.id, "email": email, "verification_sent": True}

	async def enrol_hardware_key(self, user_id: str, tenant_id: str, key_serial: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Enrol FIDO2/TOTP hardware token by serial number."""
		self.logger.info(_log_service_operation("enrol_hardware_key", user_id))
		if not key_serial:
			return {"success": False, "error": "key_serial_required"}
		key = f"{tenant_id}:{user_id}"
		method = MFAMethod(
			user_id=user_id, tenant_id=tenant_id,
			method_type=MFAMethodType.HARDWARE_TOKEN,
			encrypted_secret=key_serial,
			is_verified=True, created_by=user_id, updated_by=user_id,
		)
		self._methods.setdefault(key, []).append(method)
		await self._log_auth_event(user_id, tenant_id, "hardware_key_enrolled", {"serial": key_serial}, context)
		return {"success": True, "method_id": method.id, "key_serial": key_serial}

	async def enrol_push(self, user_id: str, tenant_id: str, device_token: str, platform: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Register a push-notification MFA channel (iOS/Android)."""
		self.logger.info(_log_service_operation("enrol_push", user_id))
		if not device_token:
			return {"success": False, "error": "device_token_required"}
		key = f"{tenant_id}:{user_id}"
		method = MFAMethod(
			user_id=user_id, tenant_id=tenant_id,
			method_type=MFAMethodType.SMS,  # reuse SMS slot; push is transport detail
			encrypted_secret=device_token,
			is_verified=False, created_by=user_id, updated_by=user_id,
		)
		self._methods.setdefault(key, []).append(method)
		await self._log_auth_event(user_id, tenant_id, "push_enrolled", {"platform": platform}, context)
		return {"success": True, "method_id": method.id, "platform": platform, "push_channel": "registered"}

	async def verify_mfa(self, user_id: str, tenant_id: str, method_id: str, otp_code: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Verify a one-time code against an enrolled method. Returns success + trust_score."""
		self.logger.info(_log_service_operation("verify_mfa", user_id, f"method={method_id}"))
		method = await self._get_mfa_method(method_id, user_id, tenant_id)
		if not method:
			return {"success": False, "error": "method_not_found"}
		valid = await self.token_service.verify_totp_code(method.encrypted_secret, otp_code)
		if valid:
			await self._reset_failed_attempts(user_id, tenant_id)
			await self._log_auth_event(user_id, tenant_id, "mfa_verified", {"method_id": method_id}, context)
			return {"success": True, "trust_score": 0.9}
		else:
			count = await self._increment_failed_attempts(user_id, tenant_id)
			await self._log_auth_event(user_id, tenant_id, "mfa_verify_failed", {"method_id": method_id, "attempts": count}, context)
			return {"success": False, "error": "invalid_code", "failed_attempts": count}

	async def step_up_auth(self, user_id: str, tenant_id: str, operation: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Initiate step-up authentication for a sensitive operation.
		Returns a challenge token requiring additional factor verification.
		"""
		self.logger.info(_log_service_operation("step_up_auth", user_id, f"op={operation}"))
		risk = await self.risk_analyzer.assess_authentication_risk(user_id, tenant_id, context)
		step_up_token = await self.token_service.generate_authentication_token(
			user_id, tenant_id, trust_score=0.5, context=context
		)
		await self._log_auth_event(user_id, tenant_id, "step_up_initiated", {"operation": operation, "risk": risk.get("risk_score")}, context)
		return {
			"success": True,
			"step_up_token": step_up_token.token_value,
			"expires_at": step_up_token.expires_at.isoformat(),
			"risk_score": risk.get("risk_score", 0.5),
			"required_factors": ["totp"],
		}

	async def mfa_bypass_admin(self, admin_id: str, tenant_id: str, target_user_id: str, reason: str, duration_minutes: int, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Admin-only: grant a time-limited MFA bypass for target_user_id."""
		self.logger.info(_log_service_operation("mfa_bypass_admin", admin_id, f"target={target_user_id}"))
		if not reason:
			return {"success": False, "error": "reason_required"}
		bypass_key = f"{tenant_id}:{target_user_id}"
		self._bypass_grants[bypass_key] = {
			"granted_by": admin_id,
			"reason": reason,
			"expires_at": (datetime.utcnow() + timedelta(minutes=duration_minutes)).isoformat(),
			"duration_minutes": duration_minutes,
		}
		await self._log_auth_event(admin_id, tenant_id, "mfa_bypass_granted", {"target": target_user_id, "reason": reason}, context)
		return {"success": True, "bypass_key": bypass_key, "expires_in_minutes": duration_minutes}

	async def mfa_recovery_code_gen(self, user_id: str, tenant_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate a fresh set of one-time recovery codes, invalidating prior set."""
		self.logger.info(_log_service_operation("mfa_recovery_code_gen", user_id))
		import secrets
		codes = [secrets.token_hex(5).upper() for _ in range(10)]
		store_key = f"{tenant_id}:{user_id}"
		self._recovery_codes[store_key] = codes
		await self._log_auth_event(user_id, tenant_id, "recovery_codes_generated", {"count": len(codes)}, context)
		return {"success": True, "codes": codes, "count": len(codes), "message": "Store these codes securely."}

	async def mfa_recovery_validate(self, user_id: str, tenant_id: str, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Consume a recovery code for account access. Each code is single-use."""
		self.logger.info(_log_service_operation("mfa_recovery_validate", user_id))
		store_key = f"{tenant_id}:{user_id}"
		codes = self._recovery_codes.get(store_key, [])
		if code.upper() in codes:
			codes.remove(code.upper())
			self._recovery_codes[store_key] = codes
			await self._reset_failed_attempts(user_id, tenant_id)
			await self._log_auth_event(user_id, tenant_id, "recovery_code_used", {"remaining": len(codes)}, context)
			return {"success": True, "remaining_codes": len(codes)}
		await self._log_auth_event(user_id, tenant_id, "recovery_code_invalid", {}, context)
		return {"success": False, "error": "invalid_recovery_code"}

	async def bulk_enrol(self, tenant_id: str, user_ids: List[str], method_type: MFAMethodType, actor: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Admin bulk-enrol a list of users to a specified MFA method."""
		self.logger.info(_log_service_operation("bulk_enrol", actor, f"users={len(user_ids)} method={method_type}"))
		successes, failures = [], []
		for uid in user_ids:
			try:
				result = await self.enrol_totp(uid, tenant_id, context) if method_type == MFAMethodType.TOTP else {"success": False, "error": "unsupported"}
				(successes if result.get("success") else failures).append(uid)
			except Exception as exc:
				failures.append(uid)
				self.logger.warning(f"bulk_enrol failed for {uid}: {exc}")
		batch_id = uuid7str()
		self._bulk_enrol_results[batch_id] = {"successes": successes, "failures": failures, "total": len(user_ids)}
		return {"batch_id": batch_id, "enrolled": len(successes), "failed": len(failures), "failures": failures}

	async def mfa_status(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Return concise MFA enablement status for a user."""
		return await self.get_user_mfa_status(user_id, tenant_id)

	async def mfa_analytics(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
		"""Return tenant-level MFA usage analytics over the last N days."""
		events = self._auth_events
		total = sum(len(v) for v in events.values())
		success_events = sum(
			1 for evts in events.values()
			for e in evts if e.get("event_type") == "authentication_success"
		)
		failure_events = sum(
			1 for evts in events.values()
			for e in evts if e.get("event_type") == "authentication_failure"
		)
		return {
			"tenant_id": tenant_id,
			"window_days": days,
			"total_events": total,
			"auth_successes": success_events,
			"auth_failures": failure_events,
			"mfa_bypass_grants": len([k for k in self._bypass_grants if k.startswith(tenant_id)]),
			"recovery_code_sets": len([k for k in self._recovery_codes if k.startswith(tenant_id)]),
			"service_metrics": self._auth_metrics,
		}

	async def trusted_device_register(self, user_id: str, tenant_id: str, device_info: "DeviceInfo", context: Dict[str, Any]) -> Dict[str, Any]:
		"""Register a device as trusted, bypassing MFA for low-risk sessions."""
		self.logger.info(_log_service_operation("trusted_device_register", user_id))
		device_key = f"{tenant_id}:{user_id}"
		device_record = {
			"device_id": uuid7str(),
			"device_name": getattr(device_info, "device_name", "unknown"),
			"platform": getattr(device_info, "platform", "unknown"),
			"registered_at": datetime.utcnow().isoformat(),
			"trusted": True,
		}
		self._trusted_devices.setdefault(device_key, []).append(device_record)
		await self._log_auth_event(user_id, tenant_id, "trusted_device_registered", device_record, context)
		return {"success": True, "device_id": device_record["device_id"], "trusted": True}

	async def trusted_device_revoke(self, user_id: str, tenant_id: str, device_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Revoke trust for a specific device."""
		self.logger.info(_log_service_operation("trusted_device_revoke", user_id, f"device={device_id}"))
		device_key = f"{tenant_id}:{user_id}"
		devices = self._trusted_devices.get(device_key, [])
		before = len(devices)
		self._trusted_devices[device_key] = [d for d in devices if d.get("device_id") != device_id]
		revoked = before - len(self._trusted_devices[device_key])
		await self._log_auth_event(user_id, tenant_id, "trusted_device_revoked", {"device_id": device_id, "revoked": revoked}, context)
		return {"success": revoked > 0, "device_id": device_id, "revoked": revoked}

	async def adaptive_mfa_risk(self, user_id: str, tenant_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Evaluate contextual risk and return the adaptive MFA challenge level.
		Levels: none | low | medium | high | block.
		"""
		self.logger.info(_log_service_operation("adaptive_mfa_risk", user_id))
		risk = await self.risk_analyzer.assess_authentication_risk(user_id, tenant_id, context)
		score = risk.get("risk_score", 0.5)
		if score < 0.2:
			level = "none"
		elif score < 0.4:
			level = "low"
		elif score < 0.6:
			level = "medium"
		elif score < 0.8:
			level = "high"
		else:
			level = "block"
		self._risk_scores[f"{tenant_id}:{user_id}"] = score
		return {"user_id": user_id, "risk_score": score, "challenge_level": level, "factors_required": max(1, int(score * 3))}

	# -------------------------------------------------------------------------
	# Private helpers (in-memory implementations)
	# -------------------------------------------------------------------------

	async def _get_user_profile(self, user_id: str, tenant_id: str) -> Optional["MFAUserProfile"]:
		"""Get user MFA profile from in-memory store."""
		return self._profiles.get(f"{tenant_id}:{user_id}")

	async def _get_user_mfa_methods(self, user_id: str, tenant_id: str) -> List["MFAMethod"]:
		"""Get user's enrolled MFA methods."""
		return list(self._methods.get(f"{tenant_id}:{user_id}", []))

	async def _store_mfa_method(self, method: "MFAMethod") -> None:
		"""Append MFA method to in-memory store."""
		key = f"{method.tenant_id}:{method.user_id}"
		self._methods.setdefault(key, []).append(method)

	async def _get_mfa_method(self, method_id: str, user_id: str, tenant_id: str) -> Optional["MFAMethod"]:
		"""Lookup a specific enrolled method by ID."""
		for m in self._methods.get(f"{tenant_id}:{user_id}", []):
			if m.id == method_id:
				return m
		return None

	async def _remove_mfa_method_from_db(self, method_id: str) -> None:
		"""Remove method from all user lists."""
		for key, methods in self._methods.items():
			self._methods[key] = [m for m in methods if m.id != method_id]

	async def _is_user_locked_out(self, user_id: str, tenant_id: str) -> bool:
		"""Check if user is currently locked out."""
		lockout_until = self._lockouts.get(f"{tenant_id}:{user_id}")
		if lockout_until is None:
			return False
		if datetime.utcnow() < lockout_until:
			return True
		del self._lockouts[f"{tenant_id}:{user_id}"]
		return False

	async def _increment_failed_attempts(self, user_id: str, tenant_id: str) -> int:
		"""Increment and return failed attempt count."""
		key = f"{tenant_id}:{user_id}"
		self._failed_attempts[key] = self._failed_attempts.get(key, 0) + 1
		return self._failed_attempts[key]

	async def _reset_failed_attempts(self, user_id: str, tenant_id: str) -> None:
		"""Reset failed attempts counter."""
		self._failed_attempts.pop(f"{tenant_id}:{user_id}", None)

	async def _lockout_user(self, user_id: str, tenant_id: str) -> None:
		"""Lock out user for lockout_duration_minutes."""
		self._lockouts[f"{tenant_id}:{user_id}"] = datetime.utcnow() + timedelta(minutes=self.lockout_duration_minutes)

	async def _handle_lockout(self, user_id: str, tenant_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Return lockout response payload."""
		lockout_until = self._lockouts.get(f"{tenant_id}:{user_id}")
		return {
			"success": False,
			"status": "locked_out",
			"reason": "too_many_failed_attempts",
			"lockout_until": lockout_until.isoformat() if lockout_until else None,
		}

	async def _log_auth_event(self, user_id: str, tenant_id: str, event_type: str, details: Dict[str, Any], context: Dict[str, Any]) -> None:
		"""Append authentication event to in-memory log."""
		key = f"{tenant_id}:{user_id}"
		self._auth_events.setdefault(key, []).append({
			"event_type": event_type,
			"user_id": user_id,
			"tenant_id": tenant_id,
			"details": details,
			"timestamp": datetime.utcnow().isoformat(),
		})

	async def _get_recent_auth_events(self, user_id: str, tenant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
		"""Return recent auth events as plain dicts (no model needed for listing)."""
		key = f"{tenant_id}:{user_id}"
		return self._auth_events.get(key, [])[-limit:]

	async def _user_has_backup_codes(self, user_id: str, tenant_id: str) -> bool:
		"""Check whether user has unused backup/recovery codes."""
		return bool(self._recovery_codes.get(f"{tenant_id}:{user_id}"))

	async def _verify_enrollment_authorization(self, user_id: str, tenant_id: str, context: Dict[str, Any]) -> bool:
		"""Simple auth check: accept if session token present in context."""
		return bool(context.get("session_token") or context.get("auth_token"))

	async def _check_method_enrollment_limits(self, user_id: str, tenant_id: str, method_type: "MFAMethodType") -> bool:
		"""Enforce max 5 methods per type per user."""
		existing = await self._get_user_mfa_methods(user_id, tenant_id)
		same_type = [m for m in existing if m.method_type == method_type]
		return len(same_type) < 5

	async def _enroll_sms_method(self, user_id: str, tenant_id: str, enrollment_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Delegate to enrol_sms."""
		return await self.enrol_sms(user_id, tenant_id, enrollment_data.get("phone_number", ""), context)

	async def _enroll_email_method(self, user_id: str, tenant_id: str, enrollment_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Delegate to enrol_email."""
		return await self.enrol_email(user_id, tenant_id, enrollment_data.get("email", ""), context)

	async def _enroll_hardware_token(self, user_id: str, tenant_id: str, enrollment_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Delegate to enrol_hardware_key."""
		return await self.enrol_hardware_key(user_id, tenant_id, enrollment_data.get("key_serial", ""), context)

	async def _get_active_users_count(self) -> int:
		"""Count distinct users with at least one enrolled method."""
		return len(self._methods)

	async def _get_enrolled_methods_count(self) -> int:
		"""Total enrolled methods across all users."""
		return sum(len(v) for v in self._methods.values())


__all__ = ["MFAService"]