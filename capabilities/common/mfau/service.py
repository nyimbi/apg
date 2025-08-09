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

	async def _get_user_profile(self, user_id: str, tenant_id: str) -> Optional[MFAUserProfile]:
		"""Get user MFA profile"""
		pass

	async def _get_user_mfa_methods(self, user_id: str, tenant_id: str) -> List[MFAMethod]:
		"""Get user's enrolled MFA methods"""
		pass

	async def _store_mfa_method(self, method: MFAMethod) -> None:
		"""Store MFA method"""
		pass

	async def _is_user_locked_out(self, user_id: str, tenant_id: str) -> bool:
		"""Check if user is locked out"""
		pass

	async def _increment_failed_attempts(self, user_id: str, tenant_id: str) -> int:
		"""Increment and return failed attempt count"""
		pass

	async def _reset_failed_attempts(self, user_id: str, tenant_id: str) -> None:
		"""Reset failed attempts counter"""
		pass

	async def _lockout_user(self, user_id: str, tenant_id: str) -> None:
		"""Lock out user temporarily"""
		pass

	async def _log_auth_event(self, user_id: str, tenant_id: str, event_type: str, details: Dict[str, Any], context: Dict[str, Any]) -> None:
		"""Log authentication event"""
		pass


__all__ = ["MFAService"]