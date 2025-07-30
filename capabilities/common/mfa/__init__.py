"""
APG Multi-Factor Authentication (MFA) Capability

Enterprise-grade multi-factor authentication capability providing intelligent
adaptive authentication, biometric support, and seamless APG integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

# Core models and types
from .models import (
	MFAUserProfile, MFAMethod, MFAMethodType, AuthEvent,
	TrustLevel, AuthenticationStatus, DeviceInfo,
	RiskAssessment, RiskFactor, BiometricTemplate,
	RecoveryMethod, AuthToken, DeviceBinding
)

# Core services
from .service import MFAService
from .mfa_engine import MFAEngine
from .risk_analyzer import RiskAnalyzer
from .token_service import TokenService, HardwareTokenValidator, OfflineTokenVerifier
from .biometric_service import BiometricService
from .anti_spoofing import AntiSpoofingService
from .enrollment_wizard import BiometricEnrollmentWizard
from .recovery_service import RecoveryService, RecoveryRequest, SecurityQuestion, TrustedContact
from .notification_service import MFANotificationService, MFANotificationType
from .business_logic import BusinessLogicOrchestrator, BusinessRule, WorkflowContext

# Integration and composition
from .integration import APGIntegrationRouter
from .composition import MFACompositionEngine, APGEvent, CapabilityWorkflow

# Flask integration
from .views import (
	MFAUserProfileView, MFAMethodView, AuthEventView,
	MFADashboardView, MFAAPIView, register_mfa_views
)
from .blueprint import MFABlueprint, create_mfa_blueprint, register_mfa_capability
from .api import create_mfa_api, MFAWebSocketEvents

__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@gmail.com"

# APG Capability Metadata for composition engine registration
APG_CAPABILITY_INFO = {
	"id": "mfa",
	"name": "Multi-Factor Authentication",
	"version": __version__,
	"description": "Enterprise-grade multi-factor authentication with biometric support",
	"category": "security",
	"author": __author__,
	"email": __email__,
	"company": "Datacraft",
	"website": "www.datacraft.co.ke",
	
	# APG integration metadata
	"apg_version": "1.0.0",
	"dependencies": [
		"auth_rbac",
		"notification", 
		"audit_compliance",
		"ai_orchestration",
		"computer_vision"
	],
	"provides": [
		"multi_factor_authentication",
		"biometric_authentication", 
		"risk_assessment",
		"account_recovery",
		"security_analytics",
		"intelligent_authentication",
		"adaptive_security"
	],
	"endpoints": {
		"api": "/api/mfa",
		"health": "/mfa/health",
		"dashboard": "/mfadashboardview/dashboard/",
		"enrollment": "/mfadashboardview/enroll/",
		"settings": "/mfadashboardview/settings/",
		"recovery": "/mfadashboardview/recovery/"
	},
	
	# Revolutionary differentiators that make this 10x better than industry leaders
	"differentiators": [
		"Intelligent Adaptive Authentication",
		"Contextual Biometric Fusion",
		"Zero-Friction Pre-Authentication", 
		"AI-Powered Recovery Assistant",
		"Seamless Offline Operations",
		"Privacy-First Architecture",
		"Enterprise-Grade Orchestration",
		"Real-Time Collaboration Support",
		"Immersive Security Analytics",
		"Revolutionary UX Innovation"
	],
	
	# Technical capabilities
	"features": {
		"authentication_methods": [
			"TOTP", "HOTP", "SMS", "EMAIL", "PUSH",
			"FACE_RECOGNITION", "VOICE_RECOGNITION", 
			"BEHAVIORAL_BIOMETRIC", "HARDWARE_TOKEN"
		],
		"biometric_support": True,
		"anti_spoofing": True,
		"liveness_detection": True,
		"risk_assessment": True,
		"trust_scoring": True,
		"account_recovery": True,
		"backup_codes": True,
		"offline_support": True,
		"real_time_notifications": True,
		"enterprise_policies": True,
		"workflow_orchestration": True,
		"audit_compliance": True,
		"delegation_tokens": True,
		"step_up_authentication": True,
		"contextual_intelligence": True
	},
	
	# Events published by this capability
	"events": {
		"publishes": [
			"mfa.authentication.success",
			"mfa.authentication.failure",
			"mfa.authentication.blocked",
			"mfa.method.enrolled",
			"mfa.method.removed",
			"mfa.method.verified",
			"mfa.security.alert",
			"mfa.recovery.initiated",
			"mfa.recovery.completed",
			"mfa.risk.assessment",
			"mfa.biometric.enrolled",
			"mfa.workflow.executed"
		],
		"subscribes": [
			"auth.user.login",
			"auth.user.logout",
			"auth.session.expired",
			"security.threat.detected",
			"ai.analysis.complete",
			"notification.delivered",
			"audit.compliance.check"
		]
	},
	
	# Performance and scalability
	"performance": {
		"concurrent_users": "unlimited",
		"authentication_latency": "<100ms",
		"biometric_processing": "<2s", 
		"risk_assessment": "<50ms",
		"offline_capability": True,
		"horizontal_scaling": True,
		"multi_tenant": True
	}
}

__all__ = [
	# Core models
	"MFAUserProfile", "MFAMethod", "MFAMethodType", "AuthEvent",
	"TrustLevel", "AuthenticationStatus", "DeviceInfo",
	"RiskAssessment", "RiskFactor", "BiometricTemplate",
	"RecoveryMethod", "AuthToken", "DeviceBinding",
	
	# Core services
	"MFAService", "MFAEngine", "RiskAnalyzer", 
	"TokenService", "HardwareTokenValidator", "OfflineTokenVerifier",
	"BiometricService", "AntiSpoofingService", "BiometricEnrollmentWizard",
	"RecoveryService", "RecoveryRequest", "SecurityQuestion", "TrustedContact",
	"MFANotificationService", "MFANotificationType",
	"BusinessLogicOrchestrator", "BusinessRule", "WorkflowContext",
	
	# Integration and composition
	"APGIntegrationRouter", "MFACompositionEngine", "APGEvent", "CapabilityWorkflow",
	
	# Flask integration
	"MFAUserProfileView", "MFAMethodView", "AuthEventView",
	"MFADashboardView", "MFAAPIView", "register_mfa_views",
	"MFABlueprint", "create_mfa_blueprint", "register_mfa_capability",
	"create_mfa_api", "MFAWebSocketEvents",
	
	# Capability metadata
	"APG_CAPABILITY_INFO"
]