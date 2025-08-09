"""
APG Multi-Factor Authentication (MFA) - Integration Models

Integration models and interfaces for seamless APG capability composition
with auth_rbac, audit_compliance, ai_orchestration, and other capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from typing import Optional, Any, Dict, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from .models import APGBase, MFAMethodType, AuthenticationStatus, RiskLevel


class APGIntegrationEvent(BaseModel):
	"""Base class for APG capability integration events"""
	model_config = ConfigDict(extra='forbid')
	
	event_type: str = Field(description="Type of integration event")
	source_capability: str = Field(default="mfa", description="Source capability name")
	target_capability: str = Field(description="Target capability name")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
	correlation_id: str = Field(description="Correlation ID for tracking")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")


# Auth RBAC Integration Models
class RBACAuthenticationRequest(APGBase):
	"""Authentication request to APG auth_rbac capability"""
	user_id: str = Field(description="User requesting authentication")
	resource_id: Optional[str] = Field(default=None, description="Resource being accessed")
	action: str = Field(description="Action being performed")
	context: Dict[str, Any] = Field(default_factory=dict, description="Authentication context")
	mfa_required: bool = Field(default=False, description="Whether MFA is required")
	trust_level_required: float = Field(default=0.5, description="Minimum trust level required")


class RBACAuthenticationResponse(BaseModel):
	"""Authentication response from APG auth_rbac capability"""
	model_config = ConfigDict(extra='forbid')
	
	request_id: str = Field(description="Original request identifier")
	user_id: str = Field(description="Authenticated user ID")
	granted: bool = Field(description="Whether access was granted")
	permissions: List[str] = Field(default_factory=list, description="Granted permissions")
	roles: List[str] = Field(default_factory=list, description="User roles")
	session_token: Optional[str] = Field(default=None, description="Session token if granted")
	mfa_challenge: Optional[Dict[str, Any]] = Field(default=None, description="MFA challenge if required")


class RBACIntegrationEvent(APGIntegrationEvent):
	"""Integration event for auth_rbac capability"""
	target_capability: str = Field(default="auth_rbac", description="Target capability")
	authentication_data: Dict[str, Any] = Field(description="Authentication data payload")


# Audit Compliance Integration Models
class AuditEvent(APGBase):
	"""Audit event for APG audit_compliance capability"""
	event_category: str = Field(description="Category of audit event")
	event_action: str = Field(description="Specific action performed") 
	user_id: str = Field(description="User who performed the action")
	resource_type: str = Field(default="mfa", description="Type of resource affected")
	resource_id: str = Field(description="ID of resource affected")
	
	# Event details
	event_data: Dict[str, Any] = Field(default_factory=dict, description="Detailed event data")
	risk_level: RiskLevel = Field(description="Risk level of the event")
	compliance_flags: List[str] = Field(default_factory=list, description="Compliance requirements met")
	
	# Context information
	ip_address: Optional[str] = Field(default=None, description="Source IP address")
	user_agent: Optional[str] = Field(default=None, description="User agent string")
	session_id: Optional[str] = Field(default=None, description="Session identifier")
	
	# Audit trail
	event_hash: str = Field(description="Cryptographic hash of event")
	previous_event_hash: Optional[str] = Field(default=None, description="Hash of previous event")
	blockchain_transaction_id: Optional[str] = Field(default=None, description="Blockchain transaction ID")


class AuditComplianceIntegrationEvent(APGIntegrationEvent):
	"""Integration event for audit_compliance capability"""
	target_capability: str = Field(default="audit_compliance", description="Target capability")
	audit_data: AuditEvent = Field(description="Audit event data")


# AI Orchestration Integration Models
class AIRiskAssessmentRequest(APGBase):
	"""Risk assessment request to APG ai_orchestration capability"""
	user_id: str = Field(description="User being assessed")
	session_context: Dict[str, Any] = Field(description="Current session context")
	behavioral_data: Dict[str, Any] = Field(default_factory=dict, description="User behavioral data")
	device_context: Dict[str, Any] = Field(default_factory=dict, description="Device context data")
	location_context: Dict[str, Any] = Field(default_factory=dict, description="Location context data")
	historical_patterns: Dict[str, Any] = Field(default_factory=dict, description="Historical user patterns")
	threat_intelligence: Dict[str, Any] = Field(default_factory=dict, description="Current threat intel")


class AIRiskAssessmentResponse(BaseModel):
	"""Risk assessment response from APG ai_orchestration capability"""
	model_config = ConfigDict(extra='forbid')
	
	request_id: str = Field(description="Original request identifier")
	user_id: str = Field(description="User who was assessed")
	risk_score: float = Field(description="Calculated risk score (0.0-1.0)")
	confidence_score: float = Field(description="Confidence in assessment (0.0-1.0)")
	risk_factors: List[Dict[str, Any]] = Field(description="Individual risk factors")
	recommended_actions: List[str] = Field(description="Recommended security actions")
	model_version: str = Field(description="AI model version used")
	processing_time_ms: int = Field(description="Processing time in milliseconds")


class AIOrchestrationIntegrationEvent(APGIntegrationEvent):
	"""Integration event for ai_orchestration capability"""
	target_capability: str = Field(default="ai_orchestration", description="Target capability")
	ai_request_data: Dict[str, Any] = Field(description="AI processing request data")


# Computer Vision Integration Models
class BiometricVerificationRequest(APGBase):
	"""Biometric verification request to APG computer_vision capability"""
	user_id: str = Field(description="User requesting verification")
	biometric_type: str = Field(description="Type of biometric (face, voice, etc.)")
	biometric_data: str = Field(description="Encoded biometric data for verification")
	stored_template_id: str = Field(description="ID of stored biometric template")
	verification_context: Dict[str, Any] = Field(default_factory=dict, description="Verification context")
	liveness_detection_required: bool = Field(default=True, description="Whether liveness detection is required")
	anti_spoofing_level: str = Field(default="high", description="Anti-spoofing security level")


class BiometricVerificationResponse(BaseModel):
	"""Biometric verification response from APG computer_vision capability"""
	model_config = ConfigDict(extra='forbid')
	
	request_id: str = Field(description="Original request identifier")
	user_id: str = Field(description="User who was verified")
	verified: bool = Field(description="Whether verification was successful")
	confidence_score: float = Field(description="Verification confidence (0.0-1.0)")
	match_score: float = Field(description="Biometric match score (0.0-1.0)")
	liveness_detected: bool = Field(description="Whether liveness was detected")
	spoofing_detected: bool = Field(description="Whether spoofing was detected")
	quality_score: float = Field(description="Biometric quality score (0.0-1.0)")
	processing_time_ms: int = Field(description="Processing time in milliseconds")
	error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if failed")


class ComputerVisionIntegrationEvent(APGIntegrationEvent):
	"""Integration event for computer_vision capability"""
	target_capability: str = Field(default="computer_vision", description="Target capability")
	biometric_data: Dict[str, Any] = Field(description="Biometric processing data")


# Real-time Collaboration Integration Models
class CollaborativeAuthRequest(APGBase):
	"""Collaborative authentication request for team scenarios"""
	requesting_user_id: str = Field(description="User requesting collaborative auth")
	target_user_id: str = Field(description="User whose access is being delegated")
	delegation_scope: List[str] = Field(description="Scope of delegated access")
	expiry_time: datetime = Field(description="When delegation expires")
	justification: str = Field(description="Business justification for delegation")
	approval_required: bool = Field(default=True, description="Whether approval is required")


class CollaborativeAuthResponse(BaseModel):
	"""Collaborative authentication response"""
	model_config = ConfigDict(extra='forbid')
	
	request_id: str = Field(description="Original request identifier")
	delegation_token: Optional[str] = Field(default=None, description="Delegation token if approved")
	status: str = Field(description="Status of delegation request")
	approval_required_from: List[str] = Field(default_factory=list, description="Users who must approve")
	expires_at: Optional[datetime] = Field(default=None, description="Token expiration time")


class RealTimeCollaborationIntegrationEvent(APGIntegrationEvent):
	"""Integration event for real_time_collaboration capability"""
	target_capability: str = Field(default="real_time_collaboration", description="Target capability")
	collaboration_data: Dict[str, Any] = Field(description="Collaboration event data")


# Notification Engine Integration Models
class MFANotificationRequest(APGBase):
	"""Notification request to APG notification_engine capability"""
	user_id: str = Field(description="User to notify")
	notification_type: str = Field(description="Type of notification")
	notification_title: str = Field(description="Notification title")
	notification_message: str = Field(description="Notification message content")
	urgency_level: str = Field(default="normal", description="Urgency level (low, normal, high, critical)")
	channels: List[str] = Field(default_factory=list, description="Notification channels to use")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional notification metadata")
	
	# Security context
	security_event: bool = Field(default=False, description="Whether this is a security event")
	requires_acknowledgment: bool = Field(default=False, description="Whether acknowledgment is required")
	expire_after_minutes: Optional[int] = Field(default=None, description="Notification expiry time")


class MFANotificationResponse(BaseModel):
	"""Notification response from APG notification_engine capability"""
	model_config = ConfigDict(extra='forbid')
	
	request_id: str = Field(description="Original request identifier")
	notification_id: str = Field(description="Created notification identifier")
	sent_channels: List[str] = Field(description="Channels where notification was sent")
	failed_channels: List[str] = Field(default_factory=list, description="Channels where sending failed")
	delivery_status: str = Field(description="Overall delivery status")
	estimated_delivery_time: Optional[datetime] = Field(default=None, description="Estimated delivery time")


class NotificationEngineIntegrationEvent(APGIntegrationEvent):
	"""Integration event for notification_engine capability"""
	target_capability: str = Field(default="notification_engine", description="Target capability")
	notification_data: MFANotificationRequest = Field(description="Notification request data")


# Integration Event Router
class APGIntegrationRouter:
	"""Router for managing APG capability integration events"""
	
	def __init__(self):
		self.event_handlers: Dict[str, callable] = {}
		self.integration_clients: Dict[str, Any] = {}
	
	def register_capability_client(self, capability_name: str, client: Any) -> None:
		"""Register a client for APG capability integration"""
		self.integration_clients[capability_name] = client
	
	def register_event_handler(self, event_type: str, handler: callable) -> None:
		"""Register an event handler for specific integration events"""
		self.event_handlers[event_type] = handler
	
	async def route_integration_event(self, event: APGIntegrationEvent) -> Any:
		"""Route integration event to appropriate capability"""
		capability_client = self.integration_clients.get(event.target_capability)
		if not capability_client:
			raise ValueError(f"No client registered for capability: {event.target_capability}")
		
		handler = self.event_handlers.get(event.event_type)
		if handler:
			return await handler(event, capability_client)
		else:
			raise ValueError(f"No handler registered for event type: {event.event_type}")


# Integration helper functions
def create_rbac_auth_request(user_id: str, resource_id: str, action: str, 
							context: Dict[str, Any]) -> RBACAuthenticationRequest:
	"""Create RBAC authentication request"""
	return RBACAuthenticationRequest(
		user_id=user_id,
		resource_id=resource_id,
		action=action,
		context=context,
		tenant_id=context.get('tenant_id', ''),
		created_by=user_id,
		updated_by=user_id
	)


def create_audit_event(user_id: str, action: str, resource_id: str, 
					  event_data: Dict[str, Any], tenant_id: str) -> AuditEvent:
	"""Create audit event for compliance logging"""
	return AuditEvent(
		event_category="authentication",
		event_action=action,
		user_id=user_id,
		resource_id=resource_id,
		event_data=event_data,
		risk_level=RiskLevel.MEDIUM,
		event_hash="",  # Will be calculated by audit service
		tenant_id=tenant_id,
		created_by=user_id,
		updated_by=user_id
	)


def create_ai_risk_request(user_id: str, session_context: Dict[str, Any], 
						  tenant_id: str) -> AIRiskAssessmentRequest:
	"""Create AI risk assessment request"""
	return AIRiskAssessmentRequest(
		user_id=user_id,
		session_context=session_context,
		tenant_id=tenant_id,
		created_by=user_id,
		updated_by=user_id
	)


def create_biometric_verification_request(user_id: str, biometric_type: str,
										 biometric_data: str, template_id: str,
										 tenant_id: str) -> BiometricVerificationRequest:
	"""Create biometric verification request"""
	return BiometricVerificationRequest(
		user_id=user_id,
		biometric_type=biometric_type,
		biometric_data=biometric_data,
		stored_template_id=template_id,
		tenant_id=tenant_id,
		created_by=user_id,
		updated_by=user_id
	)


def create_mfa_notification(user_id: str, notification_type: str, title: str,
						   message: str, tenant_id: str) -> MFANotificationRequest:
	"""Create MFA notification request"""
	return MFANotificationRequest(
		user_id=user_id,
		notification_type=notification_type,
		notification_title=title,
		notification_message=message,
		tenant_id=tenant_id,
		created_by=user_id,
		updated_by=user_id
	)


# Export all integration models
__all__ = [
	"APGIntegrationEvent",
	"RBACAuthenticationRequest",
	"RBACAuthenticationResponse", 
	"RBACIntegrationEvent",
	"AuditEvent",
	"AuditComplianceIntegrationEvent",
	"AIRiskAssessmentRequest",
	"AIRiskAssessmentResponse",
	"AIOrchestrationIntegrationEvent",
	"BiometricVerificationRequest",
	"BiometricVerificationResponse",
	"ComputerVisionIntegrationEvent",
	"CollaborativeAuthRequest",
	"CollaborativeAuthResponse",
	"RealTimeCollaborationIntegrationEvent",
	"MFANotificationRequest",
	"MFANotificationResponse",
	"NotificationEngineIntegrationEvent",
	"APGIntegrationRouter",
	"create_rbac_auth_request",
	"create_audit_event",
	"create_ai_risk_request",
	"create_biometric_verification_request",
	"create_mfa_notification"
]