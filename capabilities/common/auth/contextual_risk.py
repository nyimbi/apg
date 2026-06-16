"""
Contextual Risk Engine Enhancement

Advanced risk assessment engine providing multi-dimensional risk scoring
and adaptive authentication requirements based on context and behavior.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import json
import math
import ipaddress
import re
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass
import numpy as np

from.behavioral_auth import BehavioralRiskLevel, AuthScore

class RiskFactor(str, Enum):
	"""Types of risk factors considered in assessment"""
	LOCATION = "location"
	DEVICE = "device"
	TIME = "time"
	BEHAVIOR = "behavior"
	NETWORK = "network"
	VELOCITY = "velocity"
	REPUTATION = "reputation"
	CONTEXT = "context"

class AuthRequirement(str, Enum):
	"""Authentication requirements that can be imposed"""
	PASSWORD_ONLY = "password_only"
	MFA_REQUIRED = "mfa_required"
	BIOMETRIC_REQUIRED = "biometric_required"
	SECURITY_QUESTIONS = "security_questions"
	ADMIN_APPROVAL = "admin_approval"
	ACCESS_DENIED = "access_denied"

class RiskLevel(str, Enum):
	"""Overall risk levels"""
	VERY_LOW = "very_low"
	LOW = "low"
	MODERATE = "moderate"
	HIGH = "high"
	VERY_HIGH = "very_high"
	CRITICAL = "critical"

class LocationRisk(BaseModel):
	"""Location-based risk assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	country: Optional[str] = Field(default=None, description="Country code")
	region: Optional[str] = Field(default=None, description="State/region")
	city: Optional[str] = Field(default=None, description="City")
	ip_address: str = Field(..., description="Source IP address")
	
	# Risk indicators
	is_known_location: bool = Field(default=False, description="Previously seen location")
	is_vpn: bool = Field(default=False, description="VPN/proxy detected")
	is_tor: bool = Field(default=False, description="Tor network detected")
	is_high_risk_country: bool = Field(default=False, description="High-risk country")
	distance_from_usual: float = Field(default=0.0, description="Distance from usual location (km)")
	
	# Reputation scores
	ip_reputation_score: float = Field(default=0.5, description="IP reputation (0.0-1.0)", ge=0.0, le=1.0)
	geo_reputation_score: float = Field(default=0.5, description="Geographic reputation (0.0-1.0)", ge=0.0, le=1.0)

class DeviceRisk(BaseModel):
	"""Device-based risk assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	device_id: Optional[str] = Field(default=None, description="Device fingerprint/ID")
	user_agent: str = Field(..., description="User agent string")
	os_family: Optional[str] = Field(default=None, description="Operating system")
	browser_family: Optional[str] = Field(default=None, description="Browser family")
	device_type: Optional[str] = Field(default=None, description="Device type (mobile/desktop/tablet)")
	
	# Risk indicators
	is_known_device: bool = Field(default=False, description="Previously seen device")
	is_jailbroken: bool = Field(default=False, description="Device is jailbroken/rooted")
	has_malware_indicators: bool = Field(default=False, description="Malware indicators detected")
	browser_integrity_score: float = Field(default=1.0, description="Browser integrity (0.0-1.0)", ge=0.0, le=1.0)
	device_reputation_score: float = Field(default=0.5, description="Device reputation (0.0-1.0)", ge=0.0, le=1.0)

class TimeRisk(BaseModel):
	"""Time-based risk assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Access timestamp")
	timezone: str = Field(default="UTC", description="User timezone")
	
	# Risk indicators
	is_unusual_time: bool = Field(default=False, description="Outside usual access hours")
	is_weekend: bool = Field(default=False, description="Weekend access")
	is_holiday: bool = Field(default=False, description="Holiday access")
	time_deviation_score: float = Field(default=0.0, description="Deviation from usual pattern", ge=0.0, le=1.0)
	velocity_risk_score: float = Field(default=0.0, description="Impossible travel velocity", ge=0.0, le=1.0)

class NetworkRisk(BaseModel):
	"""Network-based risk assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	ip_address: str = Field(..., description="Source IP address")
	asn: Optional[int] = Field(default=None, description="Autonomous System Number")
	isp: Optional[str] = Field(default=None, description="Internet Service Provider")
	
	# Risk indicators
	is_residential: bool = Field(default=True, description="Residential IP address")
	is_datacenter: bool = Field(default=False, description="Datacenter IP")
	is_corporate: bool = Field(default=False, description="Corporate network")
	is_public_wifi: bool = Field(default=False, description="Public WiFi network")
	threat_intel_score: float = Field(default=0.0, description="Threat intelligence score", ge=0.0, le=1.0)
	
	# Network behavior
	connection_count: int = Field(default=1, description="Number of concurrent connections")
	bandwidth_anomaly: bool = Field(default=False, description="Unusual bandwidth usage")

class AuthContext(BaseModel):
	"""Authentication context for risk assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str = Field(..., description="User identifier")
	session_id: Optional[str] = Field(default=None, description="Session identifier")
	tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")
	
	# Risk components
	location_risk: LocationRisk = Field(..., description="Location-based risk")
	device_risk: DeviceRisk = Field(..., description="Device-based risk")
	time_risk: TimeRisk = Field(..., description="Time-based risk")
	network_risk: NetworkRisk = Field(..., description="Network-based risk")
	
	# Additional context
	resource_requested: Optional[str] = Field(default=None, description="Requested resource")
	action_requested: Optional[str] = Field(default=None, description="Requested action")
	previous_auth_method: Optional[str] = Field(default=None, description="Previous authentication method")
	
	# Behavioral data
	behavioral_score: Optional[AuthScore] = Field(default=None, description="Behavioral authentication score")

class RiskAssessment(BaseModel):
	"""Comprehensive risk assessment result"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Assessment identifier")
	user_id: str = Field(..., description="User identifier")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")
	
	# Overall risk
	overall_risk_score: float = Field(..., description="Overall risk score (0.0-1.0)", ge=0.0, le=1.0)
	risk_level: RiskLevel = Field(..., description="Risk level classification")
	
	# Individual factor scores
	factor_scores: Dict[RiskFactor, float] = Field(..., description="Risk scores by factor")
	factor_weights: Dict[RiskFactor, float] = Field(default_factory=dict, description="Factor weights used")
	
	# Authentication requirements
	required_auth_methods: List[AuthRequirement] = Field(
		default_factory=list, description="Required authentication methods"
	)
	confidence_threshold: float = Field(default=0.8, description="Required confidence threshold")
	
	# Risk details
	risk_reasons: List[str] = Field(default_factory=list, description="Reasons for risk assessment")
	risk_mitigations: List[str] = Field(default_factory=list, description="Suggested risk mitigations")
	
	# Context
	assessment_context: AuthContext = Field(..., description="Assessment context")

class ContextualRiskEngine:
	"""Enhanced contextual risk assessment engine"""
	
	def __init__(self):
		# Risk factor weights (can be adjusted per tenant)
		self.default_weights = {
			RiskFactor.LOCATION: 0.25,
			RiskFactor.DEVICE: 0.20,
			RiskFactor.TIME: 0.15,
			RiskFactor.BEHAVIOR: 0.25,
			RiskFactor.NETWORK: 0.10,
			RiskFactor.VELOCITY: 0.05
		}
		
		# Risk thresholds for different levels
		self.risk_thresholds = {
			RiskLevel.VERY_LOW: 0.0,
			RiskLevel.LOW: 0.2,
			RiskLevel.MODERATE: 0.4,
			RiskLevel.HIGH: 0.6,
			RiskLevel.VERY_HIGH: 0.8,
			RiskLevel.CRITICAL: 0.9
		}
		
		# User location history and patterns
		self._user_locations: Dict[str, List[LocationRisk]] = {}
		self._user_devices: Dict[str, List[DeviceRisk]] = {}
		self._user_time_patterns: Dict[str, List[datetime]] = {}
		
		# Threat intelligence cache
		self._ip_reputation_cache: Dict[str, float] = {}
		self._threat_intel_cache: Dict[str, Dict[str, Any]] = {}
		
		# Tenant-specific configurations
		self._tenant_weights: Dict[str, Dict[RiskFactor, float]] = {}
		self._tenant_policies: Dict[str, Dict[str, Any]] = {}
		self._global_risk_policy: Dict[str, Any] = self._default_risk_policy()
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[ContextualRisk INFO] {message} {kwargs if kwargs else ''}")
	
	def _log_warning(self, message: str, **kwargs):
		"""Log warning message"""
		print(f"[ContextualRisk WARNING] {message} {kwargs if kwargs else ''}")
	
	def _log_error(self, message: str, **kwargs):
		"""Log error message"""
		print(f"[ContextualRisk ERROR] {message} {kwargs if kwargs else ''}")

	def configure_risk_intelligence(self, policy: Dict[str, Any], tenant_id: Optional[str] = None) -> None:
		"""Configure deterministic risk intelligence inputs.

		The engine remains dependency-light: provider feeds can be normalized into
		this policy shape by adapters, while generated applications can pass local
		CIDR lists, exact IP scores, device reputation lists, and calendar data.
		"""
		assert isinstance(policy, dict), "Risk intelligence policy must be a dictionary"
		if tenant_id:
			current = dict(self._tenant_policies.get(tenant_id, {}))
			current.update(policy)
			self._tenant_policies[tenant_id] = current
		else:
			self._global_risk_policy.update(policy)
		self._ip_reputation_cache.clear()
		self._threat_intel_cache.clear()
	
	async def assess_location_risk(self, user_id: str, location_data: Dict[str, Any]) -> LocationRisk:
		"""Assess location-based risk factors"""
		assert user_id, "User ID is required"
		
		# Extract location information
		ip_address = location_data.get('ip_address', '')
		country = location_data.get('country')
		region = location_data.get('region')
		city = location_data.get('city')
		
		# Check if location is known for this user
		user_locations = self._user_locations.get(user_id, [])
		is_known_location = any(
			loc.country == country and loc.region == region and loc.city == city
			for loc in user_locations[-10:]  # Check last 10 locations
		)
		
		# Calculate distance from usual locations
		distance_from_usual = 0.0
		if user_locations and not is_known_location:
			distance_from_usual = await self._calculate_location_distance(
				location_data, user_locations
			)
		
		# Check IP reputation and threat intelligence
		ip_reputation = await self._get_ip_reputation(ip_address)
		
		# Detect VPN/proxy/Tor
		is_vpn = await self._detect_vpn(ip_address)
		is_tor = await self._detect_tor(ip_address)
		
		high_risk_countries = {
			str(item).upper()
			for item in self._combined_risk_policy().get("high_risk_countries", [])
		}
		is_high_risk_country = str(country).upper() in high_risk_countries if country else False
		
		location_risk = LocationRisk(
			country=country,
			region=region,
			city=city,
			ip_address=ip_address,
			is_known_location=is_known_location,
			is_vpn=is_vpn,
			is_tor=is_tor,
			is_high_risk_country=is_high_risk_country,
			distance_from_usual=distance_from_usual,
			ip_reputation_score=ip_reputation,
			geo_reputation_score=max(0.0, 1.0 - (distance_from_usual / 10000))  # Normalize by 10k km
		)
		
		return location_risk
	
	async def assess_device_risk(self, user_id: str, device_data: Dict[str, Any]) -> DeviceRisk:
		"""Assess device-based risk factors"""
		assert user_id, "User ID is required"
		
		# Extract device information
		device_id = device_data.get('device_id')
		user_agent = device_data.get('user_agent', '')
		os_family = device_data.get('os_family')
		browser_family = device_data.get('browser_family')
		device_type = device_data.get('device_type')
		
		# Check if device is known for this user
		user_devices = self._user_devices.get(user_id, [])
		is_known_device = any(
			dev.device_id == device_id or dev.user_agent == user_agent
			for dev in user_devices[-20:]  # Check last 20 devices
		)
		
		# Detect security risks
		is_jailbroken = await self._detect_jailbreak(device_data)
		has_malware_indicators = await self._detect_malware(device_data)
		browser_integrity = await self._assess_browser_integrity(user_agent)
		
		# Calculate device reputation
		device_reputation = await self._get_device_reputation(device_id, user_agent)
		
		device_risk = DeviceRisk(
			device_id=device_id,
			user_agent=user_agent,
			os_family=os_family,
			browser_family=browser_family,
			device_type=device_type,
			is_known_device=is_known_device,
			is_jailbroken=is_jailbroken,
			has_malware_indicators=has_malware_indicators,
			browser_integrity_score=browser_integrity,
			device_reputation_score=device_reputation
		)
		
		return device_risk
	
	async def assess_time_risk(self, user_id: str, time_data: Dict[str, Any]) -> TimeRisk:
		"""Assess time-based risk factors"""
		assert user_id, "User ID is required"
		
		timestamp = time_data.get('timestamp', datetime.utcnow())
		if isinstance(timestamp, str):
			timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
		
		timezone = time_data.get('timezone', 'UTC')
		
		# Get user's historical access patterns
		user_times = self._user_time_patterns.get(user_id, [])
		
		# Analyze time patterns
		is_unusual_time = await self._is_unusual_access_time(user_id, timestamp)
		is_weekend = timestamp.weekday() >= 5  # Saturday/Sunday
		is_holiday = await self._is_holiday(timestamp, timezone)
		
		# Calculate time deviation from usual pattern
		time_deviation = await self._calculate_time_deviation(user_times, timestamp)
		
		# Check for impossible travel velocity
		velocity_risk = await self._assess_velocity_risk(user_id, timestamp, time_data.get('location'))
		
		time_risk = TimeRisk(
			timestamp=timestamp,
			timezone=timezone,
			is_unusual_time=is_unusual_time,
			is_weekend=is_weekend,
			is_holiday=is_holiday,
			time_deviation_score=time_deviation,
			velocity_risk_score=velocity_risk
		)
		
		return time_risk
	
	async def assess_network_risk(self, user_id: str, network_data: Dict[str, Any]) -> NetworkRisk:
		"""Assess network-based risk factors"""
		assert user_id, "User ID is required"
		
		ip_address = network_data.get('ip_address', '')
		asn = network_data.get('asn')
		isp = network_data.get('isp')
		
		# Classify network type
		is_residential = await self._is_residential_ip(ip_address)
		is_datacenter = await self._is_datacenter_ip(ip_address)
		is_corporate = await self._is_corporate_ip(ip_address)
		is_public_wifi = await self._is_public_wifi(ip_address)
		
		# Get threat intelligence score
		threat_intel = await self._get_threat_intelligence(ip_address)
		
		# Analyze network behavior
		connection_count = network_data.get('connection_count', 1)
		bandwidth_anomaly = network_data.get('bandwidth_anomaly', False)
		
		network_risk = NetworkRisk(
			ip_address=ip_address,
			asn=asn,
			isp=isp,
			is_residential=is_residential,
			is_datacenter=is_datacenter,
			is_corporate=is_corporate,
			is_public_wifi=is_public_wifi,
			threat_intel_score=threat_intel,
			connection_count=connection_count,
			bandwidth_anomaly=bandwidth_anomaly
		)
		
		return network_risk
	
	async def calculate_auth_requirements(self, context: AuthContext) -> RiskAssessment:
		"""Calculate comprehensive risk assessment and authentication requirements"""
		assert context, "Authentication context is required"
		
		self._log_info("Calculating risk assessment", user_id=context.user_id)
		
		# Get risk factor weights (tenant-specific if available)
		weights = self._tenant_weights.get(context.tenant_id, self.default_weights)
		
		# Calculate individual risk scores
		factor_scores = {}
		
		# Location risk
		location_score = await self._score_location_risk(context.location_risk)
		factor_scores[RiskFactor.LOCATION] = location_score
		
		# Device risk
		device_score = await self._score_device_risk(context.device_risk)
		factor_scores[RiskFactor.DEVICE] = device_score
		
		# Time risk
		time_score = await self._score_time_risk(context.time_risk)
		factor_scores[RiskFactor.TIME] = time_score
		
		# Network risk
		network_score = await self._score_network_risk(context.network_risk)
		factor_scores[RiskFactor.NETWORK] = network_score
		
		# Behavioral risk (if available)
		behavioral_score = 0.0
		if context.behavioral_score:
			# Invert behavioral confidence to get risk score
			behavioral_score = 1.0 - context.behavioral_score.confidence
		factor_scores[RiskFactor.BEHAVIOR] = behavioral_score
		
		# Velocity risk (from time assessment)
		velocity_score = context.time_risk.velocity_risk_score
		factor_scores[RiskFactor.VELOCITY] = velocity_score
		
		# Calculate weighted overall risk score
		overall_risk_score = sum(
			factor_scores[factor] * weights.get(factor, 0.0)
			for factor in factor_scores.keys()
		)
		
		# Clamp to [0, 1]
		overall_risk_score = max(0.0, min(1.0, overall_risk_score))
		overall_risk_score = self._apply_factor_risk_floor(overall_risk_score, factor_scores, context)
		
		# Determine risk level
		risk_level = self._determine_risk_level(overall_risk_score)
		
		# Determine authentication requirements
		auth_requirements = await self._determine_auth_requirements(
			overall_risk_score, risk_level, factor_scores, context
		)
		
		# Generate risk reasons and mitigations
		risk_reasons = self._generate_risk_reasons(factor_scores, context)
		risk_mitigations = self._generate_risk_mitigations(risk_level, auth_requirements)
		
		assessment = RiskAssessment(
			user_id=context.user_id,
			overall_risk_score=overall_risk_score,
			risk_level=risk_level,
			factor_scores=factor_scores,
			factor_weights=weights,
			required_auth_methods=auth_requirements,
			confidence_threshold=self._calculate_confidence_threshold(risk_level),
			risk_reasons=risk_reasons,
			risk_mitigations=risk_mitigations,
			assessment_context=context
		)
		
		# Update user patterns
		await self._update_user_patterns(context)
		
		self._log_info("Risk assessment complete", 
					   user_id=context.user_id,
					   risk_score=overall_risk_score,
					   risk_level=risk_level.value,
					   auth_methods=len(auth_requirements))
		
		return assessment
	
	async def _score_location_risk(self, location_risk: LocationRisk) -> float:
		"""Calculate location risk score"""
		risk_factors = []
		
		# Unknown location penalty
		if not location_risk.is_known_location:
			risk_factors.append(0.3)
		
		# VPN/Proxy/Tor penalties
		if location_risk.is_vpn:
			risk_factors.append(0.4)
		if location_risk.is_tor:
			risk_factors.append(0.8)
		
		# High-risk country penalty
		if location_risk.is_high_risk_country:
			risk_factors.append(0.6)
		
		# Distance penalty (normalized by 5000km)
		distance_penalty = min(0.5, location_risk.distance_from_usual / 5000)
		risk_factors.append(distance_penalty)
		
		# IP reputation penalty
		ip_reputation_penalty = 1.0 - location_risk.ip_reputation_score
		risk_factors.append(ip_reputation_penalty * 0.5)
		
		# Geographic reputation penalty
		geo_reputation_penalty = 1.0 - location_risk.geo_reputation_score
		risk_factors.append(geo_reputation_penalty * 0.3)
		
		# Calculate weighted average
		return min(1.0, np.mean(risk_factors)) if risk_factors else 0.0
	
	async def _score_device_risk(self, device_risk: DeviceRisk) -> float:
		"""Calculate device risk score"""
		risk_factors = []
		
		# Unknown device penalty
		if not device_risk.is_known_device:
			risk_factors.append(0.4)
		
		# Security risk penalties
		if device_risk.is_jailbroken:
			risk_factors.append(0.7)
		if device_risk.has_malware_indicators:
			risk_factors.append(0.9)
		
		# Browser integrity penalty
		browser_penalty = 1.0 - device_risk.browser_integrity_score
		risk_factors.append(browser_penalty * 0.6)
		
		# Device reputation penalty
		reputation_penalty = 1.0 - device_risk.device_reputation_score
		risk_factors.append(reputation_penalty * 0.4)
		
		return min(1.0, np.mean(risk_factors)) if risk_factors else 0.0
	
	async def _score_time_risk(self, time_risk: TimeRisk) -> float:
		"""Calculate time-based risk score"""
		risk_factors = []
		
		# Unusual time penalty
		if time_risk.is_unusual_time:
			risk_factors.append(0.3)
		
		# Weekend access (minor penalty for business apps)
		if time_risk.is_weekend:
			risk_factors.append(0.1)
		
		# Holiday access penalty
		if time_risk.is_holiday:
			risk_factors.append(0.2)
		
		# Time deviation penalty
		risk_factors.append(time_risk.time_deviation_score * 0.4)
		
		# Velocity risk penalty
		risk_factors.append(time_risk.velocity_risk_score)
		
		return min(1.0, np.mean(risk_factors)) if risk_factors else 0.0
	
	async def _score_network_risk(self, network_risk: NetworkRisk) -> float:
		"""Calculate network risk score"""
		risk_factors = []
		
		# Network type penalties
		if network_risk.is_datacenter:
			risk_factors.append(0.5)
		if network_risk.is_public_wifi:
			risk_factors.append(0.4)
		
		# Threat intelligence penalty
		risk_factors.append(network_risk.threat_intel_score)
		
		# Multiple connections penalty
		if network_risk.connection_count > 5:
			connection_penalty = min(0.3, (network_risk.connection_count - 5) * 0.05)
			risk_factors.append(connection_penalty)
		
		# Bandwidth anomaly penalty
		if network_risk.bandwidth_anomaly:
			risk_factors.append(0.3)
		
		return min(1.0, np.mean(risk_factors)) if risk_factors else 0.0
	
	def _determine_risk_level(self, risk_score: float) -> RiskLevel:
		"""Determine risk level from risk score"""
		for risk_level, threshold in reversed(list(self.risk_thresholds.items())):
			if risk_score >= threshold:
				return risk_level
		return RiskLevel.VERY_LOW

	def _apply_factor_risk_floor(
		self,
		weighted_score: float,
		factor_scores: Dict[RiskFactor, float],
		context: AuthContext,
	) -> float:
		"""Prevent severe single-factor evidence from being averaged away."""
		floor = weighted_score
		if context.location_risk.is_tor:
			floor = max(floor, 0.75)
		if context.location_risk.is_vpn and context.network_risk.threat_intel_score >= 0.5:
			floor = max(floor, 0.60)
		if context.time_risk.velocity_risk_score >= 0.8:
			floor = max(floor, 0.60)
		elif context.time_risk.velocity_risk_score >= 0.5:
			floor = max(floor, 0.45)
		if context.device_risk.has_malware_indicators:
			floor = max(floor, 0.70)
		if context.device_risk.is_jailbroken and factor_scores.get(RiskFactor.DEVICE, 0.0) >= 0.5:
			floor = max(floor, 0.55)
		if context.network_risk.threat_intel_score >= 0.85:
			floor = max(floor, 0.70)
		return self._clamp_score(floor)
	
	async def _determine_auth_requirements(self, risk_score: float, risk_level: RiskLevel,
										   factor_scores: Dict[RiskFactor, float],
										   context: AuthContext) -> List[AuthRequirement]:
		"""Determine authentication requirements based on risk assessment"""
		requirements = []
		
		# Base requirements by risk level
		if risk_level == RiskLevel.VERY_LOW:
			requirements.append(AuthRequirement.PASSWORD_ONLY)
		elif risk_level == RiskLevel.LOW:
			requirements.append(AuthRequirement.PASSWORD_ONLY)
		elif risk_level == RiskLevel.MODERATE:
			requirements.append(AuthRequirement.MFA_REQUIRED)
		elif risk_level == RiskLevel.HIGH:
			requirements.extend([AuthRequirement.MFA_REQUIRED, AuthRequirement.BIOMETRIC_REQUIRED])
		elif risk_level == RiskLevel.VERY_HIGH:
			requirements.extend([
				AuthRequirement.MFA_REQUIRED,
				AuthRequirement.BIOMETRIC_REQUIRED,
				AuthRequirement.SECURITY_QUESTIONS
			])
		else:  # CRITICAL
			requirements.extend([
				AuthRequirement.MFA_REQUIRED,
				AuthRequirement.BIOMETRIC_REQUIRED,
				AuthRequirement.SECURITY_QUESTIONS,
				AuthRequirement.ADMIN_APPROVAL
			])
		
		# Specific factor-based requirements
		if factor_scores.get(RiskFactor.BEHAVIOR, 0) > 0.7:
			if AuthRequirement.BIOMETRIC_REQUIRED not in requirements:
				requirements.append(AuthRequirement.BIOMETRIC_REQUIRED)
		
		if factor_scores.get(RiskFactor.LOCATION, 0) > 0.8:
			if AuthRequirement.SECURITY_QUESTIONS not in requirements:
				requirements.append(AuthRequirement.SECURITY_QUESTIONS)
		
		if factor_scores.get(RiskFactor.DEVICE, 0) > 0.8:
			if AuthRequirement.ADMIN_APPROVAL not in requirements:
				requirements.append(AuthRequirement.ADMIN_APPROVAL)
		
		# Resource-specific requirements
		if context.resource_requested:
			if "admin" in context.resource_requested.lower():
				if AuthRequirement.MFA_REQUIRED not in requirements:
					requirements.append(AuthRequirement.MFA_REQUIRED)
		
		return requirements
	
	def _calculate_confidence_threshold(self, risk_level: RiskLevel) -> float:
		"""Calculate required confidence threshold based on risk level"""
		thresholds = {
			RiskLevel.VERY_LOW: 0.5,
			RiskLevel.LOW: 0.6,
			RiskLevel.MODERATE: 0.7,
			RiskLevel.HIGH: 0.8,
			RiskLevel.VERY_HIGH: 0.9,
			RiskLevel.CRITICAL: 0.95
		}
		return thresholds.get(risk_level, 0.8)
	
	def _generate_risk_reasons(self, factor_scores: Dict[RiskFactor, float], 
							   context: AuthContext) -> List[str]:
		"""Generate human-readable risk reasons"""
		reasons = []
		
		for factor, score in factor_scores.items():
			if score > 0.5:
				if factor == RiskFactor.LOCATION:
					if not context.location_risk.is_known_location:
						reasons.append("Access from unknown location")
					if context.location_risk.is_vpn:
						reasons.append("VPN/proxy usage detected")
					if context.location_risk.distance_from_usual > 1000:
						reasons.append(f"Access from {int(context.location_risk.distance_from_usual)}km away")
				
				elif factor == RiskFactor.DEVICE:
					if not context.device_risk.is_known_device:
						reasons.append("Unknown device")
					if context.device_risk.is_jailbroken:
						reasons.append("Compromised device detected")
				
				elif factor == RiskFactor.BEHAVIOR:
					reasons.append("Unusual behavioral patterns")
				
				elif factor == RiskFactor.TIME:
					if context.time_risk.is_unusual_time:
						reasons.append("Access at unusual time")
					if context.time_risk.velocity_risk_score > 0.5:
						reasons.append("Impossible travel velocity")
				
				elif factor == RiskFactor.NETWORK:
					if context.network_risk.threat_intel_score > 0.5:
						reasons.append("Network threat indicators")

		if context.time_risk.velocity_risk_score > 0.5 and "Impossible travel velocity" not in reasons:
			reasons.append("Impossible travel velocity")
		if context.location_risk.is_tor and "Tor network usage detected" not in reasons:
			reasons.append("Tor network usage detected")
		if context.network_risk.threat_intel_score > 0.5 and "Network threat indicators" not in reasons:
			reasons.append("Network threat indicators")
		if context.device_risk.has_malware_indicators and "Device malware indicators detected" not in reasons:
			reasons.append("Device malware indicators detected")
		
		return reasons
	
	def _generate_risk_mitigations(self, risk_level: RiskLevel, 
								   auth_requirements: List[AuthRequirement]) -> List[str]:
		"""Generate risk mitigation recommendations"""
		mitigations = []
		
		if AuthRequirement.MFA_REQUIRED in auth_requirements:
			mitigations.append("Multi-factor authentication required")
		
		if AuthRequirement.BIOMETRIC_REQUIRED in auth_requirements:
			mitigations.append("Biometric verification required")
		
		if AuthRequirement.SECURITY_QUESTIONS in auth_requirements:
			mitigations.append("Additional identity verification needed")
		
		if AuthRequirement.ADMIN_APPROVAL in auth_requirements:
			mitigations.append("Administrator approval required")
		
		if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
			mitigations.extend([
				"Enhanced session monitoring",
				"Limited session duration",
				"Restricted resource access"
			])
		
		return mitigations
	
	async def _update_user_patterns(self, context: AuthContext):
		"""Update user behavioral patterns for future risk assessment"""
		user_id = context.user_id
		
		# Update location history
		if user_id not in self._user_locations:
			self._user_locations[user_id] = []
		self._user_locations[user_id].append(context.location_risk)
		self._user_locations[user_id] = self._user_locations[user_id][-50:]  # Keep last 50
		
		# Update device history
		if user_id not in self._user_devices:
			self._user_devices[user_id] = []
		self._user_devices[user_id].append(context.device_risk)
		self._user_devices[user_id] = self._user_devices[user_id][-20:]  # Keep last 20
		
		# Update time patterns
		if user_id not in self._user_time_patterns:
			self._user_time_patterns[user_id] = []
		self._user_time_patterns[user_id].append(context.time_risk.timestamp)
		self._user_time_patterns[user_id] = self._user_time_patterns[user_id][-100:]  # Keep last 100
	
	# Utility methods for risk assessment
	async def _calculate_location_distance(self, location_data: Dict[str, Any], 
										   user_locations: List[LocationRisk]) -> float:
		"""Calculate distance from usual locations using local geo evidence."""
		current = self._location_point(location_data)
		if not current:
			current_country = location_data.get('country')
			if not current_country:
				return 0.0
			return 0.0 if current_country in {loc.country for loc in user_locations[-5:]} else 2500.0

		distances = []
		for location in user_locations[-5:]:
			previous = self._location_point({
				"country": location.country,
				"region": location.region,
				"city": location.city,
			})
			if previous:
				distances.append(self._haversine_km(current, previous))
		return min(distances) if distances else 0.0
	
	async def _get_ip_reputation(self, ip_address: str) -> float:
		"""Get deterministic IP reputation score from local intelligence."""
		if ip_address in self._ip_reputation_cache:
			return self._ip_reputation_cache[ip_address]
		
		try:
			ip = ipaddress.ip_address(ip_address)
			policy = self._combined_risk_policy()
			exact_scores = policy.get("ip_reputation_scores", {})
			if ip_address in exact_scores:
				reputation = self._clamp_score(float(exact_scores[ip_address]))
			elif self._ip_in_policy_networks(ip, "tor_exit_cidrs"):
				reputation = 0.05
			elif self._ip_in_policy_networks(ip, "vpn_cidrs"):
				reputation = 0.45
			elif self._ip_in_policy_networks(ip, "blocked_cidrs"):
				reputation = 0.10
			elif ip.is_loopback:
				reputation = 1.0
			elif ip.is_private:
				reputation = 0.92
			elif ip.is_multicast or ip.is_unspecified:
				reputation = 0.05
			elif ip.is_reserved:
				reputation = 0.70
			elif self._ip_in_policy_networks(ip, "datacenter_cidrs"):
				reputation = 0.58
			else:
				reputation = 0.68 + self._stable_fraction(ip_address) * 0.24
		except ValueError:
			reputation = 0.20
		
		self._ip_reputation_cache[ip_address] = reputation
		return reputation
	
	async def _detect_vpn(self, ip_address: str) -> bool:
		"""Detect VPN or proxy usage from configured CIDR intelligence."""
		return self._ip_matches_policy(ip_address, "vpn_cidrs")
	
	async def _detect_tor(self, ip_address: str) -> bool:
		"""Detect Tor exit-node usage from configured CIDR intelligence."""
		return self._ip_matches_policy(ip_address, "tor_exit_cidrs")
	
	async def _detect_jailbreak(self, device_data: Dict[str, Any]) -> bool:
		"""Detect jailbroken/rooted devices from attestation indicators."""
		indicator_keys = {
			"is_jailbroken",
			"is_rooted",
			"root_detected",
			"bootloader_unlocked",
			"attestation_failed",
		}
		indicators = {str(item).lower() for item in device_data.get("security_indicators", [])}
		return any(bool(device_data.get(key)) for key in indicator_keys) or bool(
			{"jailbreak", "root", "su_binary", "magisk", "cydia"} & indicators
		)
	
	async def _detect_malware(self, device_data: Dict[str, Any]) -> bool:
		"""Detect malware indicators from local device posture evidence."""
		indicator_keys = {
			"has_malware",
			"malware_detected",
			"edr_alert",
			"tamper_detected",
			"debugger_attached",
		}
		indicators = {str(item).lower() for item in device_data.get("security_indicators", [])}
		return any(bool(device_data.get(key)) for key in indicator_keys) or bool(
			{"malware", "trojan", "keylogger", "screen_overlay", "hooking"} & indicators
		)
	
	async def _assess_browser_integrity(self, user_agent: str) -> float:
		"""Assess browser integrity from user-agent consistency signals."""
		if not user_agent:
			return 0.35
		ua = user_agent.lower()
		if any(token in ua for token in ("headless", "phantomjs", "selenium", "webdriver")):
			return 0.25
		if any(token in ua for token in ("curl", "wget", "python-requests", "scrapy", "bot")):
			return 0.30
		if not re.search(r"(mozilla|chrome|safari|firefox|edg|opr|mobile)", ua):
			return 0.55
		if len(user_agent) < 32:
			return 0.65
		return 0.92
	
	async def _get_device_reputation(self, device_id: Optional[str], user_agent: str) -> float:
		"""Get deterministic device reputation from local posture intelligence."""
		policy = self._combined_risk_policy()
		if device_id and device_id in set(policy.get("trusted_device_ids", [])):
			return 0.96
		if device_id and device_id in set(policy.get("blocked_device_ids", [])):
			return 0.05
		browser_integrity = await self._assess_browser_integrity(user_agent)
		if not device_id:
			return min(0.62, browser_integrity)
		return self._clamp_score(0.55 + (self._stable_fraction(device_id) * 0.35) + (browser_integrity * 0.10))
	
	async def _is_unusual_access_time(self, user_id: str, timestamp: datetime) -> bool:
		"""Check if access time is unusual for user"""
		user_times = self._user_time_patterns.get(user_id, [])
		if len(user_times) < 10:
			return False  # Not enough data
		
		# Check if current hour is in user's usual hours
		current_hour = timestamp.hour
		usual_hours = [t.hour for t in user_times[-50:]]
		hour_counts = {}
		for hour in usual_hours:
			hour_counts[hour] = hour_counts.get(hour, 0) + 1
		
		# If current hour represents less than 5% of usual activity, it's unusual
		current_hour_count = hour_counts.get(current_hour, 0)
		total_accesses = len(usual_hours)
		return (current_hour_count / total_accesses) < 0.05
	
	async def _is_holiday(self, timestamp: datetime, timezone: str) -> bool:
		"""Check if timestamp matches configured or default holiday dates."""
		policy = self._combined_risk_policy()
		date_key = timestamp.date().isoformat()
		month_day = timestamp.strftime("%m-%d")
		holiday_dates = set(policy.get("holiday_dates", []))
		holiday_month_days = set(policy.get("holiday_month_days", []))
		return date_key in holiday_dates or month_day in holiday_month_days
	
	async def _calculate_time_deviation(self, user_times: List[datetime], 
										current_time: datetime) -> float:
		"""Calculate time deviation from usual pattern"""
		if len(user_times) < 5:
			return 0.0
		
		# Calculate typical access hours
		usual_hours = [t.hour for t in user_times[-30:]]
		if not usual_hours:
			return 0.0
		
		mean_hour = np.mean(usual_hours)
		std_hour = np.std(usual_hours)
		
		if std_hour == 0:
			return 0.0
		
		# Calculate z-score for current hour
		current_hour = current_time.hour
		z_score = abs(current_hour - mean_hour) / std_hour
		
		# Convert z-score to 0-1 range
		return min(1.0, z_score / 4.0)  # 4 standard deviations = max deviation
	
	async def _assess_velocity_risk(self, user_id: str, timestamp: datetime, 
									location_data: Optional[Dict[str, Any]]) -> float:
		"""Assess impossible travel velocity risk"""
		if not location_data:
			return 0.0
		
		user_locations = self._user_locations.get(user_id, [])
		if not user_locations:
			return 0.0
		
		# Get last location and time
		last_location = user_locations[-1]
		user_times = self._user_time_patterns.get(user_id, [])
		if not user_times:
			return 0.0
		
		last_time = user_times[-1]
		time_diff_hours = (timestamp - last_time).total_seconds() / 3600
		
		if time_diff_hours <= 0:
			return 0.0
		
		distance_km = await self._calculate_location_distance(location_data, [last_location])
		
		if distance_km == 0:
			return 0.0
		
		# Calculate required speed (km/h)
		required_speed = distance_km / time_diff_hours
		
		# Commercial flight speed ~900 km/h is maximum reasonable speed
		max_reasonable_speed = 1000  # km/h
		
		if required_speed > max_reasonable_speed:
			# Impossible travel - high risk
			return min(1.0, required_speed / max_reasonable_speed - 1.0)
		
		return 0.0
	
	async def _is_residential_ip(self, ip_address: str) -> bool:
		"""Check whether an IP should be treated as residential."""
		try:
			ip = ipaddress.ip_address(ip_address)
		except ValueError:
			return False
		return not (
			ip.is_private
			or self._ip_in_policy_networks(ip, "corporate_cidrs")
			or self._ip_in_policy_networks(ip, "datacenter_cidrs")
			or self._ip_in_policy_networks(ip, "public_wifi_cidrs")
			or self._ip_in_policy_networks(ip, "vpn_cidrs")
			or self._ip_in_policy_networks(ip, "tor_exit_cidrs")
		)
	
	async def _is_datacenter_ip(self, ip_address: str) -> bool:
		"""Check whether an IP belongs to configured datacenter ranges."""
		return self._ip_matches_policy(ip_address, "datacenter_cidrs")
	
	async def _is_corporate_ip(self, ip_address: str) -> bool:
		"""Check whether an IP is corporate/private network traffic."""
		try:
			ip = ipaddress.ip_address(ip_address)
		except ValueError:
			return False
		return ip.is_private or self._ip_in_policy_networks(ip, "corporate_cidrs")
	
	async def _is_public_wifi(self, ip_address: str) -> bool:
		"""Check whether an IP belongs to configured public-WiFi ranges."""
		return self._ip_matches_policy(ip_address, "public_wifi_cidrs")
	
	async def _get_threat_intelligence(self, ip_address: str) -> float:
		"""Get deterministic threat intelligence score for IP."""
		if ip_address in self._threat_intel_cache:
			return self._threat_intel_cache[ip_address].get('risk_score', 0.0)
		
		policy = self._combined_risk_policy()
		exact_scores = policy.get("threat_ip_scores", {})
		if ip_address in exact_scores:
			risk_score = self._clamp_score(float(exact_scores[ip_address]))
		else:
			try:
				ip = ipaddress.ip_address(ip_address)
				if self._ip_in_policy_networks(ip, "blocked_cidrs"):
					risk_score = 0.95
				elif self._ip_in_policy_networks(ip, "tor_exit_cidrs"):
					risk_score = 0.90
				elif self._ip_in_policy_networks(ip, "vpn_cidrs"):
					risk_score = 0.45
				elif self._ip_in_policy_networks(ip, "datacenter_cidrs"):
					risk_score = 0.30
				elif ip.is_private or ip.is_loopback:
					risk_score = 0.03
				elif ip.is_multicast or ip.is_unspecified:
					risk_score = 0.80
				else:
					risk_score = 0.04 + self._stable_fraction(ip_address) * 0.14
			except ValueError:
				risk_score = 0.75
		self._threat_intel_cache[ip_address] = {'risk_score': risk_score}
		return risk_score
	
	def set_tenant_weights(self, tenant_id: str, weights: Dict[RiskFactor, float]):
		"""Set tenant-specific risk factor weights"""
		assert tenant_id, "Tenant ID is required"
		self._tenant_weights[tenant_id] = weights
		self._log_info("Updated tenant risk weights", tenant_id=tenant_id)
	
	def get_tenant_weights(self, tenant_id: str) -> Dict[RiskFactor, float]:
		"""Get tenant-specific risk factor weights"""
		return self._tenant_weights.get(tenant_id, self.default_weights)
	
	def update_risk_threshold(self, risk_level: RiskLevel, threshold: float):
		"""Update risk level threshold"""
		assert 0.0 <= threshold <= 1.0, "Threshold must be between 0.0 and 1.0"
		self.risk_thresholds[risk_level] = threshold
		self._log_info("Updated risk threshold", risk_level=risk_level.value, threshold=threshold)
	
	async def get_user_risk_profile(self, user_id: str) -> Dict[str, Any]:
		"""Get user's risk profile and patterns"""
		locations = self._user_locations.get(user_id, [])
		devices = self._user_devices.get(user_id, [])
		times = self._user_time_patterns.get(user_id, [])
		
		profile = {
			"user_id": user_id,
			"location_count": len(locations),
			"device_count": len(devices),
			"access_count": len(times),
			"usual_locations": len(set(f"{loc.country}-{loc.city}" for loc in locations)),
			"usual_devices": len(set(dev.device_id for dev in devices if dev.device_id)),
			"access_pattern": {
				"usual_hours": list(set(t.hour for t in times)) if times else [],
				"usual_days": list(set(t.weekday() for t in times)) if times else []
			}
		}
		
		return profile
	
	def clear_user_patterns(self, user_id: str):
		"""Clear user patterns (GDPR compliance)"""
		if user_id in self._user_locations:
			del self._user_locations[user_id]
		if user_id in self._user_devices:
			del self._user_devices[user_id]
		if user_id in self._user_time_patterns:
			del self._user_time_patterns[user_id]
		
		self._log_info("User risk patterns cleared", user_id=user_id)

	def _default_risk_policy(self) -> Dict[str, Any]:
		"""Return default local risk intelligence for dependency-light execution."""
		return {
			"blocked_cidrs": [],
			"vpn_cidrs": [],
			"tor_exit_cidrs": [
				"185.220.100.0/22",
				"185.220.101.0/24",
			],
			"datacenter_cidrs": [
				"192.0.2.0/24",
				"198.51.100.0/24",
				"203.0.113.0/24",
			],
			"corporate_cidrs": [],
			"public_wifi_cidrs": [],
			"ip_reputation_scores": {},
			"threat_ip_scores": {},
			"trusted_device_ids": [],
			"blocked_device_ids": [],
			"holiday_dates": [],
			"holiday_month_days": ["01-01", "12-25"],
			"country_centroids": {
				"BR": [-14.2350, -51.9253],
				"CN": [35.8617, 104.1954],
				"DE": [51.1657, 10.4515],
				"FR": [46.2276, 2.2137],
				"GB": [55.3781, -3.4360],
				"IN": [20.5937, 78.9629],
				"KE": [-0.0236, 37.9062],
				"NG": [9.0820, 8.6753],
				"US": [39.8283, -98.5795],
				"ZA": [-30.5595, 22.9375],
			},
		}

	def _combined_risk_policy(self) -> Dict[str, Any]:
		"""Combine global and tenant policies for helper methods without tenant context."""
		combined = dict(self._global_risk_policy)
		for policy in self._tenant_policies.values():
			for key, value in policy.items():
				if isinstance(value, dict):
					merged = dict(combined.get(key, {}))
					merged.update(value)
					combined[key] = merged
				elif isinstance(value, list):
					combined[key] = list(combined.get(key, [])) + value
				else:
					combined[key] = value
		return combined

	def _ip_matches_policy(self, ip_address: str, policy_key: str) -> bool:
		"""Return True when an IP is inside configured networks."""
		try:
			ip = ipaddress.ip_address(ip_address)
		except ValueError:
			return False
		return self._ip_in_policy_networks(ip, policy_key)

	def _ip_in_policy_networks(self, ip: ipaddress._BaseAddress, policy_key: str) -> bool:
		"""Return True when an IP matches the named CIDR policy."""
		for cidr in self._combined_risk_policy().get(policy_key, []):
			try:
				if ip in ipaddress.ip_network(cidr, strict=False):
					return True
			except ValueError:
				continue
		return False

	def _stable_fraction(self, value: str) -> float:
		"""Return a stable 0..1 fraction for deterministic scoring."""
		digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
		return int(digest[:8], 16) / 0xFFFFFFFF

	def _clamp_score(self, value: float) -> float:
		"""Clamp score values to the Pydantic model range."""
		return max(0.0, min(1.0, value))

	def _location_point(self, location_data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
		"""Resolve a location dictionary to latitude/longitude."""
		lat = location_data.get("latitude", location_data.get("lat"))
		lon = location_data.get("longitude", location_data.get("lon"))
		if lat is not None and lon is not None:
			try:
				return float(lat), float(lon)
			except (TypeError, ValueError):
				return None
		country = location_data.get("country")
		if not country:
			return None
		centroids = self._combined_risk_policy().get("country_centroids", {})
		point = centroids.get(str(country).upper())
		if not point or len(point) != 2:
			return None
		return float(point[0]), float(point[1])

	def _haversine_km(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
		"""Calculate great-circle distance in kilometers."""
		lat1, lon1 = map(math.radians, point_a)
		lat2, lon2 = map(math.radians, point_b)
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = (
			math.sin(dlat / 2) ** 2
			+ math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
		)
		return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
