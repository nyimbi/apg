"""
APG Multi-Factor Authentication (MFA) - Notification Integration Service

Lightweight notification integration service that leverages the common/notification
capability for all delivery, focusing on MFA-specific notification logic.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from .models import MFAUserProfile, AuthEvent, TrustLevel
from .integration import APGIntegrationRouter


def _log_notification_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log notification operations for debugging and audit"""
	return f"[MFA Notification Service] {operation} for user {user_id}: {details}"


class MFANotificationType(str, Enum):
	"""MFA-specific notification types"""
	# Authentication events
	AUTH_SUCCESS = "mfa_auth_success"
	AUTH_FAILURE = "mfa_auth_failure"
	AUTH_BLOCKED = "mfa_auth_blocked"
	
	# Security events
	SECURITY_ALERT = "mfa_security_alert"
	SUSPICIOUS_ACTIVITY = "mfa_suspicious_activity"
	DEVICE_REGISTERED = "mfa_device_registered"
	DEVICE_REMOVED = "mfa_device_removed"
	
	# Recovery events
	RECOVERY_INITIATED = "mfa_recovery_initiated"
	RECOVERY_COMPLETED = "mfa_recovery_completed"
	RECOVERY_DENIED = "mfa_recovery_denied"
	
	# Configuration changes
	MFA_ENABLED = "mfa_enabled"
	MFA_DISABLED = "mfa_disabled"
	MFA_METHOD_ADDED = "mfa_method_added"
	MFA_METHOD_REMOVED = "mfa_method_removed"
	
	# Verification codes
	VERIFICATION_CODE = "mfa_verification_code"
	BACKUP_CODES_GENERATED = "mfa_backup_codes_generated"


class MFANotificationService:
	"""
	MFA-specific notification service that integrates with the common/notification
	capability for delivery while handling MFA-specific logic and templates.
	"""
	
	def __init__(self, integration_router: APGIntegrationRouter):
		"""Initialize MFA notification service"""
		self.integration = integration_router
		self.logger = logging.getLogger(__name__)
		
		# Default settings for MFA notifications
		self.default_channels = ["email", "in_app"]
		self.verification_code_expiry_minutes = 10
		self.security_alert_priority = "high"

	async def send_authentication_notification(self,
											   user_id: str,
											   tenant_id: str,
											   auth_event: AuthEvent,
											   context: Dict[str, Any]) -> bool:
		"""
		Send authentication event notification via common/notification capability.
		
		Args:
			user_id: User ID
			tenant_id: Tenant context
			auth_event: Authentication event details
			context: Additional context
		
		Returns:
			True if notification queued successfully
		"""
		try:
			# Determine notification type and priority based on auth event
			if auth_event.status == "success":
				notification_type = MFANotificationType.AUTH_SUCCESS
				priority = "low"
			elif auth_event.status == "failure":
				notification_type = MFANotificationType.AUTH_FAILURE
				priority = "normal"
			elif auth_event.status == "blocked":
				notification_type = MFANotificationType.AUTH_BLOCKED
				priority = "high"
			else:
				notification_type = MFANotificationType.SECURITY_ALERT
				priority = "high"
			
			# Prepare notification payload for common/notification
			notification_payload = {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"notification_type": notification_type.value,
				"priority": priority,
				"channels": self.default_channels,
				"template_data": {
					"auth_event": auth_event.dict(),
					"context": context,
					"timestamp": datetime.utcnow().isoformat(),
					"location": context.get("location", {}),
					"device": context.get("device", {}),
					"risk_score": context.get("risk_score", 0.0)
				}
			}
			
			# Send via common/notification capability
			result = await self.integration.call_capability(
				"notification",
				"send_notification",
				notification_payload
			)
			
			success = result.get("success", False)
			if success:
				self.logger.info(_log_notification_operation(
					"send_auth_notification", user_id, f"type={notification_type.value}"
				))
			
			return success
			
		except Exception as e:
			self.logger.error(f"Authentication notification error for user {user_id}: {str(e)}", exc_info=True)
			return False

	async def send_security_alert(self,
								  user_id: str,
								  tenant_id: str,
								  alert_type: str,
								  details: Dict[str, Any],
								  priority: str = "high") -> bool:
		"""
		Send security alert notification via common/notification capability.
		
		Args:
			user_id: User ID
			tenant_id: Tenant context
			alert_type: Type of security alert
			details: Alert details
			priority: Notification priority
		
		Returns:
			True if notification queued successfully
		"""
		try:
			self.logger.info(_log_notification_operation("send_security_alert", user_id, f"type={alert_type}"))
			
			notification_payload = {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"notification_type": MFANotificationType.SECURITY_ALERT.value,
				"priority": priority,
				"channels": self.default_channels,
				"template_data": {
					"alert_type": alert_type,
					"details": details,
					"timestamp": datetime.utcnow().isoformat(),
					"tenant_id": tenant_id
				}
			}
			
			result = await self.integration.call_capability(
				"notification",
				"send_notification",
				notification_payload
			)
			
			return result.get("success", False)
			
		except Exception as e:
			self.logger.error(f"Security alert notification error for user {user_id}: {str(e)}", exc_info=True)
			return False

	async def send_verification_code(self,
									 user_id: str,
									 tenant_id: str,
									 channel: str,
									 target: str,
									 verification_code: str,
									 context: Dict[str, Any]) -> bool:
		"""
		Send verification code via specified channel using common/notification.
		
		Args:
			user_id: User ID
			tenant_id: Tenant context
			channel: Delivery channel (email/sms)
			target: Target address (email/phone)
			verification_code: Verification code
			context: Additional context
		
		Returns:
			True if code sent successfully
		"""
		try:
			self.logger.info(_log_notification_operation(
				"send_verification_code", user_id, f"channel={channel}, target={target[:3]}***"
			))
			
			notification_payload = {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"notification_type": MFANotificationType.VERIFICATION_CODE.value,
				"priority": "urgent",
				"channels": [channel],
				"target_override": target,
				"template_data": {
					"verification_code": verification_code,
					"expires_in_minutes": self.verification_code_expiry_minutes,
					"timestamp": datetime.utcnow().isoformat(),
					"context": context
				}
			}
			
			result = await self.integration.call_capability(
				"notification",
				"send_notification",
				notification_payload
			)
			
			return result.get("success", False)
			
		except Exception as e:
			self.logger.error(f"Verification code notification error for user {user_id}: {str(e)}", exc_info=True)
			return False

	async def send_recovery_notification(self,
										 user_id: str,
										 tenant_id: str,
										 recovery_event: str,
										 recovery_data: Dict[str, Any]) -> bool:
		"""
		Send account recovery notification via common/notification capability.
		
		Args:
			user_id: User ID
			tenant_id: Tenant context
			recovery_event: Recovery event type
			recovery_data: Recovery details
		
		Returns:
			True if notification queued successfully
		"""
		try:
			# Map recovery events to notification types
			event_mapping = {
				"initiated": MFANotificationType.RECOVERY_INITIATED,
				"completed": MFANotificationType.RECOVERY_COMPLETED,
				"denied": MFANotificationType.RECOVERY_DENIED
			}
			
			notification_type = event_mapping.get(recovery_event, MFANotificationType.SECURITY_ALERT)
			priority = "high" if recovery_event in ["initiated", "denied"] else "normal"
			
			notification_payload = {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"notification_type": notification_type.value,
				"priority": priority,
				"channels": self.default_channels,
				"template_data": {
					"recovery_event": recovery_event,
					"recovery_data": recovery_data,
					"timestamp": datetime.utcnow().isoformat()
				}
			}
			
			result = await self.integration.call_capability(
				"notification",
				"send_notification",
				notification_payload
			)
			
			return result.get("success", False)
			
		except Exception as e:
			self.logger.error(f"Recovery notification error for user {user_id}: {str(e)}", exc_info=True)
			return False

	async def send_configuration_notification(self,
											  user_id: str,
											  tenant_id: str,
											  config_change: str,
											  details: Dict[str, Any]) -> bool:
		"""
		Send MFA configuration change notification via common/notification capability.
		
		Args:
			user_id: User ID
			tenant_id: Tenant context
			config_change: Configuration change type
			details: Change details
		
		Returns:
			True if notification queued successfully
		"""
		try:
			# Map configuration changes to notification types
			change_mapping = {
				"mfa_enabled": MFANotificationType.MFA_ENABLED,
				"mfa_disabled": MFANotificationType.MFA_DISABLED,
				"method_added": MFANotificationType.MFA_METHOD_ADDED,
				"method_removed": MFANotificationType.MFA_METHOD_REMOVED,
				"device_registered": MFANotificationType.DEVICE_REGISTERED,
				"device_removed": MFANotificationType.DEVICE_REMOVED,
				"backup_codes_generated": MFANotificationType.BACKUP_CODES_GENERATED
			}
			
			notification_type = change_mapping.get(config_change, MFANotificationType.SECURITY_ALERT)
			
			notification_payload = {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"notification_type": notification_type.value,
				"priority": "normal",
				"channels": self.default_channels,
				"template_data": {
					"config_change": config_change,
					"details": details,
					"timestamp": datetime.utcnow().isoformat()
				}
			}
			
			result = await self.integration.call_capability(
				"notification",
				"send_notification",
				notification_payload
			)
			
			return result.get("success", False)
			
		except Exception as e:
			self.logger.error(f"Configuration notification error for user {user_id}: {str(e)}", exc_info=True)
			return False

	async def register_mfa_notification_templates(self) -> bool:
		"""
		Register MFA notification templates with common/notification capability.
		
		Returns:
			True if templates registered successfully
		"""
		try:
			templates = [
				{
					"notification_type": MFANotificationType.AUTH_SUCCESS.value,
					"channel": "email",
					"subject_template": "Successful MFA Login - {tenant_name}",
					"body_template": "Hello {user_name},\n\nSuccessful multi-factor authentication at {timestamp} from {location}.\n\nIf this wasn't you, please contact support immediately."
				},
				{
					"notification_type": MFANotificationType.AUTH_FAILURE.value,
					"channel": "email",
					"subject_template": "Failed MFA Attempt - {tenant_name}",
					"body_template": "Hello {user_name},\n\nFailed multi-factor authentication attempt at {timestamp} from {location}.\n\nIf this wasn't you, your account may be at risk."
				},
				{
					"notification_type": MFANotificationType.VERIFICATION_CODE.value,
					"channel": "sms",
					"subject_template": "MFA Verification Code",
					"body_template": "Your MFA verification code is: {verification_code}\n\nThis code expires in {expires_in_minutes} minutes."
				},
				{
					"notification_type": MFANotificationType.VERIFICATION_CODE.value,
					"channel": "email",
					"subject_template": "MFA Verification Code - {tenant_name}",
					"body_template": "Hello {user_name},\n\nYour verification code is: {verification_code}\n\nThis code expires in {expires_in_minutes} minutes.\n\nIf you didn't request this code, please ignore this message."
				},
				{
					"notification_type": MFANotificationType.SECURITY_ALERT.value,
					"channel": "email",
					"subject_template": "MFA Security Alert - {tenant_name}",
					"body_template": "Hello {user_name},\n\nMFA Security alert: {alert_type}\n\nDetails: {details}\n\nPlease review your account immediately."
				},
				{
					"notification_type": MFANotificationType.RECOVERY_INITIATED.value,
					"channel": "email",
					"subject_template": "MFA Account Recovery Initiated - {tenant_name}",
					"body_template": "Hello {user_name},\n\nAccount recovery has been initiated for your MFA settings.\n\nIf this wasn't you, please contact support immediately."
				}
			]
			
			for template in templates:
				await self.integration.call_capability(
					"notification",
					"register_template",
					template
				)
			
			self.logger.info("MFA notification templates registered successfully")
			return True
			
		except Exception as e:
			self.logger.error(f"Template registration error: {str(e)}", exc_info=True)
			return False


__all__ = [
	"MFANotificationService",
	"MFANotificationType"
]