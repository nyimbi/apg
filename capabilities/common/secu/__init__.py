"""
APG Security Framework Capability

Comprehensive enterprise security controls framework providing zero-trust architecture,
AI-powered threat detection, and automated compliance for the APG platform.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import secrets
import threading
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from contextlib import asynccontextmanager
from .capability_contract import evaluate_capability_rules, get_capability_contract

# Security Framework Enums
class SecurityLevel(str, Enum):
	"""Security classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	SECRET = "secret"
	TOP_SECRET = "top_secret"

class RiskLevel(str, Enum):
	"""Risk assessment levels"""
	MINIMAL = "minimal"
	LOW = "low"
	MODERATE = "moderate"
	HIGH = "high"
	CRITICAL = "critical"

class ThreatType(str, Enum):
	"""Types of security threats"""
	MALWARE = "malware"
	PHISHING = "phishing"
	INSIDER_THREAT = "insider_threat"
	DATA_EXFILTRATION = "data_exfiltration"
	PRIVILEGE_ESCALATION = "privilege_escalation"
	BRUTE_FORCE = "brute_force"
	SQL_INJECTION = "sql_injection"
	XSS = "xss"
	CSRF = "csrf"
	DDoS = "ddos"
	APT = "apt"  # Advanced Persistent Threat
	ZERO_DAY = "zero_day"
	SOCIAL_ENGINEERING = "social_engineering"
	UNKNOWN = "unknown"

class ComplianceFramework(str, Enum):
	"""Supported compliance frameworks"""
	SOX = "sox"
	GDPR = "gdpr"
	HIPAA = "hipaa"
	ISO_27001 = "iso_27001"
	PCI_DSS = "pci_dss"
	NIST = "nist"
	SOC2 = "soc2"
	CCPA = "ccpa"

class SecurityAction(str, Enum):
	"""Security response actions"""
	ALLOW = "allow"
	DENY = "deny"
	CHALLENGE = "challenge"
	MONITOR = "monitor"
	QUARANTINE = "quarantine"
	BLOCK = "block"
	ISOLATE = "isolate"
	ALERT = "alert"

class DeviceTrustLevel(str, Enum):
	"""Device trust classification"""
	TRUSTED = "trusted"
	KNOWN = "known"
	UNKNOWN = "unknown"
	COMPROMISED = "compromised"
	BLACKLISTED = "blacklisted"

# Core Security Models
class DeviceContext(BaseModel):
	"""Device context information for security assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	device_id: str = Field(..., description="Unique device identifier")
	device_type: str = Field(..., description="Device type (desktop, mobile, tablet)")
	os_type: str = Field(..., description="Operating system type")
	os_version: str = Field(..., description="Operating system version")
	browser_type: Optional[str] = Field(default=None, description="Browser type")
	browser_version: Optional[str] = Field(default=None, description="Browser version")
	trust_level: DeviceTrustLevel = Field(default=DeviceTrustLevel.UNKNOWN, description="Device trust level")
	last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last seen timestamp")
	fingerprint: str = Field(..., description="Device fingerprint hash")

class NetworkContext(BaseModel):
	"""Network context information for security assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	ip_address: str = Field(..., description="Client IP address")
	country: Optional[str] = Field(default=None, description="Country of origin")
	region: Optional[str] = Field(default=None, description="Region/state")
	city: Optional[str] = Field(default=None, description="City")
	isp: Optional[str] = Field(default=None, description="Internet service provider")
	is_vpn: bool = Field(default=False, description="VPN connection detected")
	is_proxy: bool = Field(default=False, description="Proxy connection detected")
	is_tor: bool = Field(default=False, description="Tor network detected")
	is_known_malicious: bool = Field(default=False, description="Known malicious IP")
	reputation_score: float = Field(default=0.0, ge=0.0, le=100.0, description="IP reputation score")

class RiskScore(BaseModel):
	"""Risk assessment score with contributing factors"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall risk score")
	level: RiskLevel = Field(..., description="Risk level classification")
	behavioral_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Behavioral risk component")
	device_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Device risk component")
	network_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Network risk component")
	temporal_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Temporal risk component")
	confidence: float = Field(default=0.0, ge=0.0, le=100.0, description="Confidence in assessment")
	factors: List[str] = Field(default_factory=list, description="Contributing risk factors")
	calculated_at: datetime = Field(default_factory=datetime.utcnow, description="Calculation timestamp")

class ThreatIndicator(BaseModel):
	"""Security threat indicator"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Threat indicator ID")
	threat_type: ThreatType = Field(..., description="Type of threat detected")
	severity: RiskLevel = Field(..., description="Threat severity level")
	confidence: float = Field(..., ge=0.0, le=100.0, description="Detection confidence")
	source: str = Field(..., description="Detection source system")
	description: str = Field(..., description="Threat description")
	indicators: Dict[str, Any] = Field(default_factory=dict, description="Technical indicators")
	mitigation: Optional[str] = Field(default=None, description="Mitigation strategy")
	detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
	expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")

class SecurityContext(BaseModel):
	"""Complete security context for risk assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Security context ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	user_id: str = Field(..., description="User identifier")
	session_id: str = Field(..., description="Session identifier")
	request_id: Optional[str] = Field(default=None, description="Request identifier")
	capability_id: Optional[str] = Field(default=None, description="APG capability identifier")
	action: str = Field(..., description="Action being performed")
	resource: Optional[str] = Field(default=None, description="Resource being accessed")
	device_context: DeviceContext = Field(..., description="Device information")
	network_context: NetworkContext = Field(..., description="Network information")
	risk_score: Optional[RiskScore] = Field(default=None, description="Current risk assessment")
	threat_indicators: List[ThreatIndicator] = Field(default_factory=list, description="Active threats")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Context creation time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

class ComplianceStatus(BaseModel):
	"""Compliance status tracking"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	framework: ComplianceFramework = Field(..., description="Compliance framework")
	status: str = Field(..., description="Compliance status")
	score: float = Field(..., ge=0.0, le=100.0, description="Compliance score")
	requirements_met: int = Field(..., ge=0, description="Requirements satisfied")
	requirements_total: int = Field(..., ge=0, description="Total requirements")
	violations: List[str] = Field(default_factory=list, description="Current violations")
	last_assessment: datetime = Field(default_factory=datetime.utcnow, description="Last assessment time")
	next_assessment: Optional[datetime] = Field(default=None, description="Next assessment due")

class SecurityPolicy(BaseModel):
	"""Security policy definition"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Policy identifier")
	name: str = Field(..., description="Policy name")
	description: str = Field(..., description="Policy description")
	tenant_id: Optional[str] = Field(default=None, description="Tenant-specific policy")
	capability_id: Optional[str] = Field(default=None, description="Capability-specific policy")
	conditions: Dict[str, Any] = Field(default_factory=dict, description="Policy conditions")
	actions: List[SecurityAction] = Field(default_factory=list, description="Actions to take")
	priority: int = Field(default=100, description="Policy priority")
	enabled: bool = Field(default=True, description="Policy enabled status")
	created_by: str = Field(..., description="Policy creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
	expires_at: Optional[datetime] = Field(default=None, description="Policy expiration")

# APG Integration Classes
class APGSecurityContext:
	"""Thread-local security context manager for APG integration"""
	
	def __init__(self):
		self._local = threading.local()
	
	def set_context(self, context: SecurityContext) -> None:
		"""Set security context for current thread"""
		self._local.context = context
	
	def get_context(self) -> Optional[SecurityContext]:
		"""Get security context for current thread"""
		return getattr(self._local, 'context', None)
	
	def clear_context(self) -> None:
		"""Clear security context for current thread"""
		if hasattr(self._local, 'context'):
			delattr(self._local, 'context')

# Global security context manager
_security_context = APGSecurityContext()

def set_security_context(context: SecurityContext) -> None:
	"""Set security context for current thread"""
	_security_context.set_context(context)

def get_security_context() -> Optional[SecurityContext]:
	"""Get security context for current thread"""
	return _security_context.get_context()

def clear_security_context() -> None:
	"""Clear security context for current thread"""
	_security_context.clear_context()

@asynccontextmanager
async def security_context(context: SecurityContext):
	"""Async context manager for security context"""
	old_context = get_security_context()
	try:
		set_security_context(context)
		yield context
	finally:
		if old_context:
			set_security_context(old_context)
		else:
			clear_security_context()

# APG Capability Dependencies
class APGDependencies:
	"""Manager for APG capability dependencies"""
	
	def __init__(self):
		self._auth_service = None
		self._config_service = None
		self._audit_service = None
		self._tenant_service = None
	
	async def initialize(self):
		"""Initialize APG capability dependencies"""
		try:
			# Import and initialize authentication service
			from ..auth import get_authentication_manager
			self._auth_service = await get_authentication_manager()
		except ImportError as e:
			self._log_dependency_error("auth", e)
		
		try:
			# Import and initialize configuration service
			from ..conf import get_config_manager
			self._config_service = await get_config_manager()
		except ImportError as e:
			self._log_dependency_error("conf", e)
		
		try:
			# Import and initialize audit service
			from ..audl import get_audit_logger
			self._audit_service = await get_audit_logger()
		except ImportError as e:
			self._log_dependency_error("audl", e)
		
		try:
			# Import and initialize tenant service
			from ..mten import get_tenant_service
			self._tenant_service = await get_tenant_service()
		except ImportError as e:
			self._log_dependency_error("mten", e)
	
	def _log_dependency_error(self, capability: str, error: Exception):
		"""Log dependency initialization error"""
		print(f"Warning: APG {capability} capability not available: {error}")
	
	@property
	def auth_service(self):
		"""Get authentication service"""
		return self._auth_service
	
	@property
	def config_service(self):
		"""Get configuration service"""
		return self._config_service
	
	@property
	def audit_service(self):
		"""Get audit service"""
		return self._audit_service
	
	@property 
	def tenant_service(self):
		"""Get tenant service"""
		return self._tenant_service

# Global dependencies manager
_apg_dependencies = APGDependencies()

async def initialize_apg_dependencies():
	"""Initialize APG capability dependencies"""
	await _apg_dependencies.initialize()

def get_apg_dependencies() -> APGDependencies:
	"""Get APG dependencies manager"""
	return _apg_dependencies

# Security Framework Core Classes (to be implemented in service.py)
class SecurityFramework:
	"""APG Security Framework main class"""
	
	def __init__(self):
		self.dependencies = get_apg_dependencies()
		self.initialized = False
	
	async def initialize(self):
		"""Initialize security framework"""
		if not self.initialized:
			await initialize_apg_dependencies()
			self.initialized = True
	
	def _log_security_event(self, message: str, **kwargs):
		"""Log security event with APG audit integration"""
		print(f"[SECURITY] {message}")
		# Backlog: integrate with APG audit logging.

# Global security framework instance
_security_framework = None

async def get_security_framework() -> SecurityFramework:
	"""Get global security framework instance"""
	global _security_framework
	if _security_framework is None:
		_security_framework = SecurityFramework()
		await _security_framework.initialize()
	return _security_framework

async def init_security_framework() -> SecurityFramework:
	"""Initialize and return security framework"""
	return await get_security_framework()

# APG Composition Engine Registration
APG_SECURITY_METADATA = {
	"capability_name": "secu",
	"capability_title": "Security Framework",
	"version": "1.0.0",
	"description": "Enterprise security controls framework with zero-trust architecture",
	"category": "security_foundation",
	"dependencies": ["auth", "conf", "audl"],
	"provides": [
		"risk_assessment",
		"threat_detection", 
		"security_policies",
		"compliance_automation",
		"zero_trust_architecture",
		"policy_exception_governance",
		"incident_response_governance",
		"device_quarantine_governance",
		"security_audit_timeline",
		"security_agents"
	],
	"load_priority": 50,  # Load after dependencies
	"multi_tenant": True,
	"enterprise_features": True
}

def get_capability_info(tenant_id: str = "default", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
	"""Return executable SECU capability metadata and contract details."""
	contract = get_capability_contract(tenant_id, overrides)
	return {
		"metadata": APG_SECURITY_METADATA,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"ui_manifest": contract["ui"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"]
	}

def register_capability() -> Dict[str, Any]:
	"""Register SECU with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "secu",
		"display_name": "Security Framework",
		"description": APG_SECURITY_METADATA["description"],
		"version": APG_SECURITY_METADATA["version"],
		"dependencies": APG_SECURITY_METADATA["dependencies"],
		"provides": APG_SECURITY_METADATA["provides"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"permissions": [
			"secu:view",
			"secu:view_risk",
			"secu:view_threats",
			"secu:view_compliance",
			"secu:manage_policies",
			"secu:approve_exception",
			"secu:respond",
			"secu:admin"
		]
	}

# Export main interfaces
__all__ = [
	# Enums
	"SecurityLevel", "RiskLevel", "ThreatType", "ComplianceFramework", 
	"SecurityAction", "DeviceTrustLevel",
	
	# Models
	"DeviceContext", "NetworkContext", "RiskScore", "ThreatIndicator",
	"SecurityContext", "ComplianceStatus", "SecurityPolicy",
	
	# Context Management
	"set_security_context", "get_security_context", "clear_security_context", 
	"security_context",
	
	# Main Classes
	"SecurityFramework", "APGDependencies",
	
	# Initialization Functions
	"get_security_framework", "init_security_framework", 
	"initialize_apg_dependencies", "get_apg_dependencies",
	
	# Metadata
	"APG_SECURITY_METADATA",
	"get_capability_contract", "evaluate_capability_rules",
	"get_capability_info", "register_capability"
]
