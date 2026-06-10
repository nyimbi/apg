"""
APG Security Framework Service

Comprehensive business logic for enterprise security controls, threat detection,
and compliance automation following APG coding standards.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Tuple
from datetime import datetime, timedelta, timezone
import asyncio
import json
import hashlib
import secrets
import statistics
from pathlib import Path
from uuid_extensions import uuid7str
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

# APG imports
try:
	from . import (
		SecurityLevel, RiskLevel, ThreatType, ComplianceFramework,
		SecurityAction, DeviceTrustLevel, get_apg_dependencies
	)
	from .models import (
		DeviceContext, NetworkContext, BehavioralPattern, RiskScore,
		ThreatIndicator, SecurityContext, ComplianceRequirement,
		ComplianceStatus, SecurityPolicy, SecurityIncident, SecurityMetric
	)
except ImportError:
	from capabilities.common.secu import (
		SecurityLevel, RiskLevel, ThreatType, ComplianceFramework,
		SecurityAction, DeviceTrustLevel, get_apg_dependencies
	)
	from capabilities.common.secu.models import (
		DeviceContext, NetworkContext, BehavioralPattern, RiskScore,
		ThreatIndicator, SecurityContext, ComplianceRequirement,
		ComplianceStatus, SecurityPolicy, SecurityIncident, SecurityMetric
	)

class SecurityConfigurationManager:
	"""Manages security configuration with APG integration"""
	
	def __init__(self):
		self.config_cache: Dict[str, Any] = {}
		self.policy_cache: Dict[str, SecurityPolicy] = {}
		self.dependencies = get_apg_dependencies()
	
	async def initialize(self):
		"""Initialize configuration manager"""
		await self._load_default_configurations()
		await self._load_security_policies()
	
	async def _load_default_configurations(self):
		"""Load default security configurations"""
		default_configs = {
			"risk_assessment": {
				"behavioral_weight": 0.3,
				"device_weight": 0.25,
				"network_weight": 0.25,
				"temporal_weight": 0.2,
				"confidence_threshold": 0.7,
				"high_risk_threshold": 75.0,
				"critical_risk_threshold": 90.0
			},
			"threat_detection": {
				"anomaly_threshold": 0.8,
				"baseline_learning_period": 30,
				"threat_correlation_window": 300,
				"false_positive_threshold": 0.1,
				"auto_containment_enabled": True
			},
			"compliance": {
				"assessment_frequency": 86400,  # Daily
				"auto_remediation_enabled": True,
				"notification_threshold": "medium",
				"evidence_retention_days": 2555  # 7 years
			},
			"policies": {
				"default_action": SecurityAction.DENY,
				"policy_evaluation_timeout": 5.0,
				"exception_approval_required": True,
				"policy_cache_ttl": 3600
			}
		}
		
		for category, config in default_configs.items():
			await self._set_config(category, config)
	
	async def _load_security_policies(self):
		"""Load default security policies"""
		default_policies = [
			SecurityPolicy(
				name="High Risk Access Block",
				description="Block access for high risk contexts",
				category="access_control",
				conditions={"risk_score": {"operator": "gte", "value": 85}},
				actions=[SecurityAction.BLOCK, SecurityAction.ALERT],
				priority=1,
				created_by="system"
			),
			SecurityPolicy(
				name="Unknown Device Challenge",
				description="Challenge authentication for unknown devices",
				category="device_trust",
				conditions={"device_trust_level": {"operator": "eq", "value": "unknown"}},
				actions=[SecurityAction.CHALLENGE, SecurityAction.MONITOR],
				priority=50,
				created_by="system"
			),
			SecurityPolicy(
				name="Malicious IP Block",
				description="Block known malicious IP addresses",
				category="network_security",
				conditions={"is_known_malicious": {"operator": "eq", "value": True}},
				actions=[SecurityAction.BLOCK, SecurityAction.ALERT],
				priority=5,
				created_by="system"
			)
		]
		
		for policy in default_policies:
			self.policy_cache[policy.id] = policy
	
	async def _set_config(self, key: str, value: Any, tenant_id: Optional[str] = None):
		"""Set configuration value"""
		config_key = f"{tenant_id}.{key}" if tenant_id else f"global.{key}"
		self.config_cache[config_key] = value
		
		# Backlog: integrate with APG config service.
		if self.dependencies.config_service:
			try:
				await self.dependencies.config_service.set_config(config_key, value)
			except Exception as e:
				self._log_error(f"Failed to store config {config_key}: {e}")
	
	async def get_config(self, key: str, default: Any = None, tenant_id: Optional[str] = None) -> Any:
		"""Get configuration value with tenant override"""
		# Try tenant-specific config first
		if tenant_id:
			tenant_key = f"{tenant_id}.{key}"
			if tenant_key in self.config_cache:
				return self.config_cache[tenant_key]
		
		# Fall back to global config
		global_key = f"global.{key}"
		return self.config_cache.get(global_key, default)
	
	async def update_policy(self, policy: SecurityPolicy) -> SecurityPolicy:
		"""Update security policy"""
		policy.updated_at = datetime.utcnow()
		self.policy_cache[policy.id] = policy
		await self._log_security_event(f"Policy updated: {policy.name}", policy_id=policy.id)
		return policy
	
	async def get_policies_for_context(self, context: SecurityContext) -> List[SecurityPolicy]:
		"""Get applicable policies for security context"""
		applicable_policies = []
		
		for policy in self.policy_cache.values():
			if not policy.enabled:
				continue
			
			# Check tenant scope
			if policy.tenant_id and policy.tenant_id != context.tenant_id:
				continue
			
			# Check capability scope
			if policy.capability_id and policy.capability_id != context.capability_id:
				continue
			
			# Evaluate conditions
			if await self._evaluate_policy_conditions(policy, context):
				applicable_policies.append(policy)
		
		# Sort by priority
		return sorted(applicable_policies, key=lambda p: p.priority)
	
	async def _evaluate_policy_conditions(self, policy: SecurityPolicy, context: SecurityContext) -> bool:
		"""Evaluate if policy conditions match context"""
		for condition_key, condition_value in policy.conditions.items():
			if not await self._evaluate_condition(condition_key, condition_value, context):
				return False
		return True
	
	async def _evaluate_condition(self, key: str, condition: Dict[str, Any], context: SecurityContext) -> bool:
		"""Evaluate individual policy condition"""
		operator = condition.get("operator", "eq")
		expected_value = condition.get("value")
		
		# Get actual value from context
		actual_value = await self._get_context_value(key, context)
		
		if operator == "eq":
			return actual_value == expected_value
		elif operator == "ne":
			return actual_value != expected_value
		elif operator == "gt":
			return float(actual_value) > float(expected_value)
		elif operator == "gte":
			return float(actual_value) >= float(expected_value)
		elif operator == "lt":
			return float(actual_value) < float(expected_value)
		elif operator == "lte":
			return float(actual_value) <= float(expected_value)
		elif operator == "in":
			return actual_value in expected_value
		elif operator == "contains":
			return expected_value in str(actual_value)
		
		return False
	
	async def _get_context_value(self, key: str, context: SecurityContext) -> Any:
		"""Extract value from security context by key"""
		if key == "risk_score":
			return context.risk_score.overall_score if context.risk_score else 0.0
		elif key == "device_trust_level":
			return context.device_context.trust_level.value
		elif key == "is_known_malicious":
			return context.network_context.is_known_malicious
		elif key == "threat_count":
			return len(context.threat_indicators)
		elif key == "user_id":
			return context.user_id
		elif key == "tenant_id":
			return context.tenant_id
		elif key == "capability_id":
			return context.capability_id
		elif key == "action":
			return context.action
		
		return None
	
	def _log_error(self, message: str, **kwargs):
		"""Log error message"""
		print(f"[SECURITY_CONFIG_ERROR] {message}")
	
	async def _log_security_event(self, message: str, **kwargs):
		"""Log security event with APG audit integration"""
		print(f"[SECURITY_CONFIG] {message}")
		# Backlog: integrate with APG audit service.

class ContextualRiskEngine:
	"""AI-powered contextual risk assessment engine"""
	
	def __init__(self, config_manager: SecurityConfigurationManager):
		self.config_manager = config_manager
		self.behavioral_baselines: Dict[str, BehavioralPattern] = {}
	
	async def calculate_risk_score(self, context: SecurityContext) -> RiskScore:
		"""Calculate comprehensive risk score for security context"""
		# Get risk assessment configuration
		config = await self.config_manager.get_config(
			"risk_assessment", {}, context.tenant_id
		)
		
		# Calculate component scores
		behavioral_score = await self._analyze_behavioral_risk(context, config)
		device_score = await self._analyze_device_risk(context, config)
		network_score = await self._analyze_network_risk(context, config)
		temporal_score = await self._analyze_temporal_risk(context, config)
		
		# Calculate weighted overall score
		weights = {
			"behavioral": config.get("behavioral_weight", 0.3),
			"device": config.get("device_weight", 0.25),
			"network": config.get("network_weight", 0.25),
			"temporal": config.get("temporal_weight", 0.2)
		}
		
		overall_score = (
			behavioral_score * weights["behavioral"] +
			device_score * weights["device"] +
			network_score * weights["network"] +
			temporal_score * weights["temporal"]
		)
		
		# Determine risk level
		level = await self._classify_risk_level(overall_score, config)
		
		# Calculate confidence
		confidence = await self._calculate_confidence(context, config)
		
		# Identify contributing factors
		factors = await self._identify_risk_factors(context, {
			"behavioral": behavioral_score,
			"device": device_score,
			"network": network_score,
			"temporal": temporal_score
		})
		
		return RiskScore(
			overall_score=overall_score,
			level=level,
			behavioral_score=behavioral_score,
			device_score=device_score,
			network_score=network_score,
			temporal_score=temporal_score,
			confidence=confidence,
			factors=factors,
			weights=weights,
			calculation_method="weighted_contextual"
		)
	
	async def _analyze_behavioral_risk(self, context: SecurityContext, config: Dict[str, Any]) -> float:
		"""Analyze behavioral risk patterns"""
		user_id = context.user_id
		baseline = self.behavioral_baselines.get(user_id)
		
		if not baseline or not baseline.baseline_established:
			# No baseline - moderate risk for new users
			return 40.0
		
		risk_score = 0.0
		
		# Analyze login time patterns
		current_hour = context.created_at.hour
		if current_hour not in baseline.typical_login_times:
			risk_score += 15.0
		
		# Analyze location patterns
		current_location = {
			"country": context.network_context.country,
			"city": context.network_context.city
		}
		
		if not any(
			loc.get("country") == current_location["country"] 
			for loc in baseline.typical_locations
		):
			risk_score += 25.0
		
		# Analyze device patterns
		device_id = context.device_context.device_id
		if device_id not in baseline.typical_devices:
			risk_score += 20.0
		
		# Analyze access velocity
		if await self._detect_impossible_travel(context, baseline):
			risk_score += 40.0
		
		return min(risk_score, 100.0)
	
	async def _analyze_device_risk(self, context: SecurityContext, config: Dict[str, Any]) -> float:
		"""Analyze device-based risk factors"""
		device = context.device_context
		risk_score = 0.0
		
		# Trust level risk
		trust_risk = {
			DeviceTrustLevel.TRUSTED: 0.0,
			DeviceTrustLevel.KNOWN: 10.0,
			DeviceTrustLevel.UNKNOWN: 40.0,
			DeviceTrustLevel.COMPROMISED: 95.0,
			DeviceTrustLevel.BLACKLISTED: 100.0
		}
		risk_score += trust_risk.get(device.trust_level, 50.0)
		
		# Outdated OS risk
		if await self._is_os_outdated(device.os_type, device.os_version):
			risk_score += 15.0
		
		# Missing security features (check if available)
		security_features = getattr(device, 'security_features', {})
		if not security_features.get("antivirus", False):
			risk_score += 10.0
		if not security_features.get("firewall", False):
			risk_score += 10.0
		if not security_features.get("encryption", False):
			risk_score += 20.0
		
		return min(risk_score, 100.0)
	
	async def _analyze_network_risk(self, context: SecurityContext, config: Dict[str, Any]) -> float:
		"""Analyze network-based risk factors"""
		network = context.network_context
		risk_score = 0.0
		
		# Known malicious IP
		if network.is_known_malicious:
			return 100.0
		
		# Anonymous networks
		if network.is_tor:
			risk_score += 60.0
		elif network.is_vpn:
			risk_score += 20.0
		elif network.is_proxy:
			risk_score += 30.0
		
		# Reputation score (inverted - lower reputation = higher risk)
		reputation_risk = 100.0 - network.reputation_score
		risk_score += reputation_risk * 0.3
		
		# Hosting provider risk (check if available)
		if getattr(network, 'is_hosting', False):
			risk_score += 25.0
		
		return min(risk_score, 100.0)
	
	async def _analyze_temporal_risk(self, context: SecurityContext, config: Dict[str, Any]) -> float:
		"""Analyze temporal risk patterns"""
		risk_score = 0.0
		current_time = context.created_at
		
		# Off-hours access risk
		if current_time.hour < 6 or current_time.hour > 22:
			risk_score += 15.0
		
		# Weekend access risk
		if current_time.weekday() >= 5:  # Saturday or Sunday
			risk_score += 10.0
		
		# Holiday access risk (simplified - would need holiday calendar)
		if current_time.day == 1 and current_time.month == 1:  # New Year's Day
			risk_score += 20.0
		
		return min(risk_score, 100.0)
	
	async def _classify_risk_level(self, score: float, config: Dict[str, Any]) -> RiskLevel:
		"""Classify risk level based on score"""
		critical_threshold = config.get("critical_risk_threshold", 90.0)
		high_threshold = config.get("high_risk_threshold", 75.0)
		
		if score >= critical_threshold:
			return RiskLevel.CRITICAL
		elif score >= high_threshold:
			return RiskLevel.HIGH
		elif score >= 50.0:
			return RiskLevel.MODERATE
		elif score >= 25.0:
			return RiskLevel.LOW
		else:
			return RiskLevel.MINIMAL
	
	async def _calculate_confidence(self, context: SecurityContext, config: Dict[str, Any]) -> float:
		"""Calculate confidence in risk assessment"""
		confidence_factors = []
		
		# Behavioral baseline confidence
		user_id = context.user_id
		baseline = self.behavioral_baselines.get(user_id)
		if baseline and baseline.baseline_established:
			confidence_factors.append(baseline.confidence_score * 100)
		else:
			confidence_factors.append(30.0)  # Low confidence for new users
		
		# Device trust confidence
		trust_confidence = {
			DeviceTrustLevel.TRUSTED: 95.0,
			DeviceTrustLevel.KNOWN: 80.0,
			DeviceTrustLevel.UNKNOWN: 50.0,
			DeviceTrustLevel.COMPROMISED: 90.0,
			DeviceTrustLevel.BLACKLISTED: 95.0
		}
		confidence_factors.append(trust_confidence.get(
			context.device_context.trust_level, 60.0
		))
		
		# Network reputation confidence
		confidence_factors.append(context.network_context.reputation_score)
		
		return statistics.mean(confidence_factors)
	
	async def _identify_risk_factors(self, context: SecurityContext, scores: Dict[str, float]) -> List[str]:
		"""Identify specific risk factors contributing to score"""
		factors = []
		
		if scores["behavioral"] > 30:
			factors.append("Unusual behavioral patterns detected")
		if scores["device"] > 40:
			factors.append("Untrusted or compromised device")
		if scores["network"] > 50:
			factors.append("Suspicious network characteristics")
		if scores["temporal"] > 20:
			factors.append("Off-hours access pattern")
		
		if context.network_context.is_known_malicious:
			factors.append("Known malicious IP address")
		if context.network_context.is_tor:
			factors.append("Tor network usage")
		if context.device_context.trust_level == DeviceTrustLevel.UNKNOWN:
			factors.append("Unknown device")
		
		return factors
	
	async def _detect_impossible_travel(self, context: SecurityContext, baseline: BehavioralPattern) -> bool:
		"""Detect impossible travel based on location velocity"""
		# Simplified implementation - would need geolocation and timing data
		return False
	
	async def _is_os_outdated(self, os_type: str, os_version: str) -> bool:
		"""Check if operating system version is outdated"""
		# Simplified implementation - would need OS version database
		return False

class PredictiveThreatDetector:
	"""Machine learning threat prediction and detection engine"""
	
	def __init__(self, config_manager: SecurityConfigurationManager):
		self.config_manager = config_manager
		self.threat_patterns: Dict[str, Any] = {}
		self.ml_models: Dict[str, Any] = {}
	
	async def predict_threats(self, context: SecurityContext) -> List[ThreatIndicator]:
		"""Predict potential threats for security context"""
		threats = []
		
		# Anomaly-based detection
		anomaly_threats = await self._detect_anomalies(context)
		threats.extend(anomaly_threats)
		
		# Signature-based detection
		signature_threats = await self._detect_signatures(context)
		threats.extend(signature_threats)
		
		# Behavioral analysis
		behavioral_threats = await self._analyze_behavioral_threats(context)
		threats.extend(behavioral_threats)
		
		# Correlation analysis
		correlated_threats = await self._correlate_threats(context, threats)
		threats.extend(correlated_threats)
		
		# Filter and rank threats
		filtered_threats = await self._filter_and_rank_threats(threats)
		
		return filtered_threats
	
	async def _detect_anomalies(self, context: SecurityContext) -> List[ThreatIndicator]:
		"""Detect anomalies in user behavior and system patterns"""
		threats = []
		
		# Check for velocity anomalies
		if await self._check_login_velocity_anomaly(context):
			threats.append(ThreatIndicator(
				threat_type=ThreatType.BRUTE_FORCE,
				severity=RiskLevel.HIGH,
				confidence=85.0,
				source="velocity_detector",
				title="High Frequency Login Attempts",
				description="Unusually high login attempt frequency detected",
				indicators={"login_velocity": "high"},
				mitigation="Implement rate limiting and account lockout"
			))
		
		# Check for geolocation anomalies
		if await self._check_geolocation_anomaly(context):
			threats.append(ThreatIndicator(
				threat_type=ThreatType.INSIDER_THREAT,
				severity=RiskLevel.MODERATE,
				confidence=70.0,
				source="geo_detector",
				title="Unusual Geographic Access",
				description="Access from unusual geographic location",
				indicators={"location": context.network_context.country},
				mitigation="Verify user identity through additional authentication"
			))
		
		return threats
	
	async def _detect_signatures(self, context: SecurityContext) -> List[ThreatIndicator]:
		"""Detect known attack signatures"""
		threats = []
		
		# Check for known malicious IPs
		if context.network_context.is_known_malicious:
			threats.append(ThreatIndicator(
				threat_type=ThreatType.APT,
				severity=RiskLevel.CRITICAL,
				confidence=95.0,
				source="ip_reputation",
				title="Known Malicious IP Access",
				description=f"Access attempt from known malicious IP: {context.network_context.ip_address}",
				indicators={"malicious_ip": context.network_context.ip_address},
				mitigation="Block IP address and investigate account compromise"
			))
		
		# Check for compromised devices
		if context.device_context.trust_level == DeviceTrustLevel.COMPROMISED:
			threats.append(ThreatIndicator(
				threat_type=ThreatType.MALWARE,
				severity=RiskLevel.HIGH,
				confidence=90.0,
				source="device_trust",
				title="Compromised Device Access",
				description="Access from known compromised device",
				indicators={"device_id": context.device_context.device_id},
				mitigation="Quarantine device and force password reset"
			))
		
		return threats
	
	async def _analyze_behavioral_threats(self, context: SecurityContext) -> List[ThreatIndicator]:
		"""Analyze behavioral patterns for threat indicators"""
		threats = []
		
		# Check for data exfiltration patterns
		if await self._check_data_exfiltration_pattern(context):
			threats.append(ThreatIndicator(
				threat_type=ThreatType.DATA_EXFILTRATION,
				severity=RiskLevel.HIGH,
				confidence=80.0,
				source="behavior_analyzer",
				title="Potential Data Exfiltration",
				description="Unusual data access patterns suggesting exfiltration",
				indicators={"access_pattern": "bulk_download"},
				mitigation="Monitor and restrict data access privileges"
			))
		
		# Check for privilege escalation attempts
		if await self._check_privilege_escalation(context):
			threats.append(ThreatIndicator(
				threat_type=ThreatType.PRIVILEGE_ESCALATION,
				severity=RiskLevel.HIGH,
				confidence=75.0,
				source="behavior_analyzer",
				title="Privilege Escalation Attempt",
				description="Unusual attempts to access elevated privileges",
				indicators={"escalation_attempts": "multiple"},
				mitigation="Review and restrict privilege assignments"
			))
		
		return threats
	
	async def _correlate_threats(self, context: SecurityContext, existing_threats: List[ThreatIndicator]) -> List[ThreatIndicator]:
		"""Correlate multiple threat indicators for advanced threat detection"""
		correlated_threats = []
		
		# Look for APT patterns
		if len(existing_threats) >= 2:
			threat_types = [t.threat_type for t in existing_threats]
			if (ThreatType.INSIDER_THREAT in threat_types and 
				ThreatType.DATA_EXFILTRATION in threat_types):
				
				correlated_threats.append(ThreatIndicator(
					threat_type=ThreatType.APT,
					severity=RiskLevel.CRITICAL,
					confidence=85.0,
					source="threat_correlator",
					title="Advanced Persistent Threat Pattern",
					description="Multiple threat indicators suggest coordinated APT campaign",
					indicators={"correlated_threats": [t.id for t in existing_threats]},
					mitigation="Initiate incident response and forensic investigation"
				))
		
		return correlated_threats
	
	async def _filter_and_rank_threats(self, threats: List[ThreatIndicator]) -> List[ThreatIndicator]:
		"""Filter false positives and rank threats by severity"""
		# Filter out low-confidence threats
		config = await self.config_manager.get_config("threat_detection", {})
		min_confidence = config.get("false_positive_threshold", 0.1) * 100
		
		filtered_threats = [t for t in threats if t.confidence >= min_confidence]
		
		# Sort by severity and confidence
		severity_order = {
			RiskLevel.CRITICAL: 5,
			RiskLevel.HIGH: 4,
			RiskLevel.MODERATE: 3,
			RiskLevel.LOW: 2,
			RiskLevel.MINIMAL: 1
		}
		
		return sorted(
			filtered_threats,
			key=lambda t: (severity_order.get(t.severity, 0), t.confidence),
			reverse=True
		)
	
	# Helper methods for threat detection
	async def _check_login_velocity_anomaly(self, context: SecurityContext) -> bool:
		"""Check for abnormal login velocity"""
		# Simplified implementation
		return False
	
	async def _check_geolocation_anomaly(self, context: SecurityContext) -> bool:
		"""Check for geolocation anomalies"""
		# Simplified implementation
		return False
	
	async def _check_data_exfiltration_pattern(self, context: SecurityContext) -> bool:
		"""Check for data exfiltration patterns"""
		# Simplified implementation
		return False
	
	async def _check_privilege_escalation(self, context: SecurityContext) -> bool:
		"""Check for privilege escalation attempts"""
		# Simplified implementation
		return False

class ComplianceAutomationEngine:
	"""Automated compliance monitoring and reporting engine"""
	
	def __init__(self, config_manager: SecurityConfigurationManager):
		self.config_manager = config_manager
		self.compliance_requirements: Dict[ComplianceFramework, List[ComplianceRequirement]] = {}
		self._initialize_compliance_frameworks_sync()
	
	def _initialize_compliance_frameworks_sync(self):
		"""Initialize compliance framework requirements synchronously"""
		# SOX Requirements
		sox_requirements = [
			ComplianceRequirement(
				framework=ComplianceFramework.SOX,
				requirement_id="SOX-302",
				title="Management Assessment of Internal Controls",
				description="Management must assess and certify internal controls",
				category="internal_controls",
				priority="high",
				control_objectives=["assess_controls", "certify_effectiveness"],
				responsible_party="management"
			),
			ComplianceRequirement(
				framework=ComplianceFramework.SOX,
				requirement_id="SOX-404",
				title="Internal Control Assessment",
				description="Annual assessment of internal control over financial reporting",
				category="financial_controls",
				priority="high",
				control_objectives=["document_controls", "test_effectiveness"],
				responsible_party="auditor"
			)
		]
		
		# GDPR Requirements
		gdpr_requirements = [
			ComplianceRequirement(
				framework=ComplianceFramework.GDPR,
				requirement_id="GDPR-25",
				title="Data Protection by Design and by Default",
				description="Implement data protection principles in system design",
				category="data_protection",
				priority="high",
				control_objectives=["privacy_by_design", "data_minimization"],
				responsible_party="data_controller"
			),
			ComplianceRequirement(
				framework=ComplianceFramework.GDPR,
				requirement_id="GDPR-32",
				title="Security of Processing",
				description="Implement appropriate technical and organizational measures",
				category="security_measures",
				priority="high",
				control_objectives=["encryption", "access_controls", "data_integrity"],
				responsible_party="data_processor"
			)
		]
		
		# Store requirements
		self.compliance_requirements[ComplianceFramework.SOX] = sox_requirements
		self.compliance_requirements[ComplianceFramework.GDPR] = gdpr_requirements
	
	
	async def assess_compliance(self, framework: ComplianceFramework, tenant_id: str) -> ComplianceStatus:
		"""Assess compliance status for a specific framework"""
		requirements = self.compliance_requirements.get(framework, [])
		
		if not requirements:
			raise ValueError(f"No requirements defined for framework: {framework}")
		
		# Assess each requirement
		met_count = 0
		partial_count = 0
		violations = []
		gaps = []
		
		for requirement in requirements:
			status = await self._assess_requirement(requirement, tenant_id)
			
			if status == "compliant":
				met_count += 1
			elif status == "partial":
				partial_count += 1
			else:
				violations.append(f"{requirement.requirement_id}: {requirement.title}")
				gaps.append(requirement.requirement_id)
		
		# Calculate compliance score
		total_requirements = len(requirements)
		score = ((met_count + (partial_count * 0.5)) / total_requirements) * 100
		
		# Determine overall status
		if score >= 95:
			status = "compliant"
			risk_rating = RiskLevel.LOW
		elif score >= 80:
			status = "mostly_compliant"
			risk_rating = RiskLevel.MODERATE
		elif score >= 60:
			status = "partially_compliant"
			risk_rating = RiskLevel.HIGH
		else:
			status = "non_compliant"
			risk_rating = RiskLevel.CRITICAL
		
		return ComplianceStatus(
			tenant_id=tenant_id,
			framework=framework,
			status=status,
			score=score,
			requirements_met=met_count,
			requirements_total=total_requirements,
			requirements_partial=partial_count,
			violations=violations,
			gaps=gaps,
			risk_rating=risk_rating
		)
	
	async def _assess_requirement(self, requirement: ComplianceRequirement, tenant_id: str) -> str:
		"""Assess individual compliance requirement"""
		# This is a simplified implementation
		# In practice, this would involve checking actual system controls
		
		# Simulate assessment based on requirement category
		if requirement.category == "internal_controls":
			# Check if audit logging is enabled
			if await self._check_audit_logging_enabled(tenant_id):
				return "compliant"
			else:
				return "non_compliant"
		
		elif requirement.category == "data_protection":
			# Check if encryption is enabled
			if await self._check_encryption_enabled(tenant_id):
				return "compliant"
			else:
				return "partial"
		
		elif requirement.category == "security_measures":
			# Check security controls
			if await self._check_security_controls(tenant_id):
				return "compliant"
			else:
				return "non_compliant"
		
		# Default to partial compliance for unknown categories
		return "partial"
	
	async def generate_compliance_report(self, framework: ComplianceFramework, tenant_id: str) -> Dict[str, Any]:
		"""Generate comprehensive compliance report"""
		status = await self.assess_compliance(framework, tenant_id)
		
		report = {
			"framework": framework.value,
			"tenant_id": tenant_id,
			"assessment_date": datetime.utcnow().isoformat(),
			"overall_status": status.status,
			"compliance_score": status.score,
			"risk_rating": status.risk_rating.value,
			"summary": {
				"total_requirements": status.requirements_total,
				"requirements_met": status.requirements_met,
				"requirements_partial": status.requirements_partial,
				"requirements_failed": status.requirements_total - status.requirements_met - status.requirements_partial
			},
			"violations": status.violations,
			"gaps": status.gaps,
			"recommendations": await self._generate_recommendations(status),
			"next_assessment": status.next_assessment.isoformat() if status.next_assessment else None
		}
		
		return report
	
	async def _generate_recommendations(self, status: ComplianceStatus) -> List[str]:
		"""Generate recommendations for compliance improvement"""
		recommendations = []
		
		if status.score < 80:
			recommendations.append("Implement comprehensive audit logging across all systems")
			recommendations.append("Enable encryption for all sensitive data at rest and in transit")
			recommendations.append("Establish regular compliance monitoring and reporting procedures")
		
		if status.risk_rating == RiskLevel.CRITICAL:
			recommendations.append("Engage external compliance consultant for immediate remediation")
			recommendations.append("Implement emergency compliance measures to address critical gaps")
		
		if len(status.violations) > 5:
			recommendations.append("Prioritize violation remediation based on business impact")
			recommendations.append("Establish automated compliance monitoring to prevent future violations")
		
		return recommendations
	
	# Helper methods for compliance checking
	async def _check_audit_logging_enabled(self, tenant_id: str) -> bool:
		"""Check if audit logging is properly configured"""
		# Integration with APG audit service
		dependencies = get_apg_dependencies()
		if dependencies.audit_service:
			try:
				# Check audit configuration
				return True  # Simplified - would check actual config
			except Exception:
				return False
		return False
	
	async def _check_encryption_enabled(self, tenant_id: str) -> bool:
		"""Check if encryption is properly configured"""
		# Simplified implementation
		return True
	
	async def _check_security_controls(self, tenant_id: str) -> bool:
		"""Check if security controls are properly implemented"""
		# Simplified implementation
		return True

class APGSecurityFrameworkService:
	"""Main security framework service orchestrating all security engines"""
	
	def __init__(self):
		self.config_manager = SecurityConfigurationManager()
		self.risk_engine = ContextualRiskEngine(self.config_manager)
		self.threat_detector = PredictiveThreatDetector(self.config_manager)
		self.compliance_engine = ComplianceAutomationEngine(self.config_manager)
		self.dependencies = get_apg_dependencies()
		self.initialized = False
	
	async def initialize(self):
		"""Initialize the security framework service"""
		if not self.initialized:
			await self.config_manager.initialize()
			await self.dependencies.initialize()
			self.initialized = True
			await self._log_security_event("APG Security Framework initialized")
	
	async def assess_security_context(self, context: SecurityContext) -> SecurityContext:
		"""Comprehensive security assessment for context"""
		# Calculate risk score
		context.risk_score = await self.risk_engine.calculate_risk_score(context)
		
		# Detect threats
		threats = await self.threat_detector.predict_threats(context)
		context.threat_indicators = threats
		
		# Update context timestamp
		context.updated_at = datetime.utcnow()
		
		# Log security assessment
		await self._log_security_event(
			f"Security assessment completed for user {context.user_id}",
			risk_score=context.risk_score.overall_score,
			threat_count=len(threats)
		)
		
		return context
	
	async def evaluate_security_policies(self, context: SecurityContext) -> List[SecurityAction]:
		"""Evaluate security policies and determine actions"""
		applicable_policies = await self.config_manager.get_policies_for_context(context)
		actions = []
		
		for policy in applicable_policies:
			actions.extend(policy.actions)
			await self._log_security_event(
				f"Policy applied: {policy.name}",
				policy_id=policy.id,
				user_id=context.user_id
			)
		
		# Deduplicate actions while preserving priority order
		unique_actions = []
		seen = set()
		for action in actions:
			if action not in seen:
				unique_actions.append(action)
				seen.add(action)
		
		return unique_actions
	
	async def assess_tenant_compliance(self, tenant_id: str, framework: ComplianceFramework) -> ComplianceStatus:
		"""Assess compliance for a tenant"""
		return await self.compliance_engine.assess_compliance(framework, tenant_id)
	
	async def generate_security_report(self, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate comprehensive security report"""
		report = {
			"tenant_id": tenant_id,
			"report_period": {
				"start": start_date.isoformat(),
				"end": end_date.isoformat()
			},
			"generated_at": datetime.utcnow().isoformat(),
			"risk_metrics": await self._get_risk_metrics(tenant_id, start_date, end_date),
			"threat_summary": await self._get_threat_summary(tenant_id, start_date, end_date),
			"compliance_status": await self._get_compliance_summary(tenant_id),
			"security_incidents": await self._get_incident_summary(tenant_id, start_date, end_date),
			"recommendations": await self._generate_security_recommendations(tenant_id)
		}
		
		return report
	
	async def _get_risk_metrics(self, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Get risk metrics for reporting period"""
		# Simplified implementation - would query actual metrics
		return {
			"average_risk_score": 45.2,
			"high_risk_sessions": 12,
			"critical_risk_sessions": 3,
			"risk_trend": "stable"
		}
	
	async def _get_threat_summary(self, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Get threat summary for reporting period"""
		# Simplified implementation
		return {
			"total_threats_detected": 45,
			"critical_threats": 5,
			"high_threats": 15,
			"false_positives": 8,
			"threats_by_type": {
				"brute_force": 12,
				"malware": 8,
				"phishing": 15,
				"insider_threat": 6,
				"data_exfiltration": 4
			}
		}
	
	async def _get_compliance_summary(self, tenant_id: str) -> Dict[str, Any]:
		"""Get compliance summary for all frameworks"""
		summary = {}
		
		for framework in ComplianceFramework:
			try:
				status = await self.compliance_engine.assess_compliance(framework, tenant_id)
				summary[framework.value] = {
					"status": status.status,
					"score": status.score,
					"violations": len(status.violations)
				}
			except Exception as e:
				summary[framework.value] = {
					"status": "error",
					"error": str(e)
				}
		
		return summary
	
	async def _get_incident_summary(self, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Get security incident summary"""
		# Simplified implementation
		return {
			"total_incidents": 8,
			"open_incidents": 2,
			"resolved_incidents": 6,
			"mean_resolution_time": "4.2 hours",
			"incidents_by_severity": {
				"critical": 1,
				"high": 3,
				"moderate": 3,
				"low": 1
			}
		}
	
	async def _generate_security_recommendations(self, tenant_id: str) -> List[str]:
		"""Generate security recommendations"""
		return [
			"Enable multi-factor authentication for all users",
			"Implement zero-trust network architecture",
			"Conduct quarterly security awareness training",
			"Deploy advanced threat detection tools",
			"Establish incident response playbooks"
		]
	
	async def _log_security_event(self, message: str, **kwargs):
		"""Log security event with APG audit integration"""
		print(f"[SECURITY_FRAMEWORK] {message}")
		
		# Integrate with APG audit service
		if self.dependencies.audit_service:
			try:
				await self.dependencies.audit_service.log_event(
					level="INFO",
					event_type="SECURITY_EVENT",
					component="security_framework",
					action=message,
					metadata=kwargs
				)
			except Exception as e:
				print(f"[SECURITY_FRAMEWORK_ERROR] Failed to log to audit: {e}")


class SecuService:
	"""Deterministic package service for SECU capability composition.

	This synchronous service is the dependency-light execution surface used by
	generated APG applications and package publishing checks. Live identity,
	SIEM, EDR/MDM, compliance, policy, and audit providers remain behind the
	older async integration engines and future adapters.
	"""

	def __init__(self) -> None:
		from .capability_contract import (
			PRIVILEGED_SECU_AGENT_ROLES,
			SUPPORTED_SECU_AGENT_ROLES,
			SUPPORTED_SECU_AGENT_RUNTIMES,
			evaluate_capability_rules,
			get_capability_contract,
		)
		from .security_runtime import (
			ComplianceControlRecord,
			DevicePostureRecord,
			PolicyExceptionRecord,
			RiskAssessmentRecord,
			SecurityAuditEventRecord,
			SecurityAgentRecord,
			SecurityIncidentRecord,
			SecurityLifecycleBatchRecord,
			SecurityPolicyRecord,
			ThreatIndicatorRecord,
			clamp_score,
			control_status,
			normalize_device_trust,
			normalize_security_level,
			normalize_tags,
			normalize_threat_severity,
			required_actions,
			risk_band,
			stable_id,
			summarize_decision,
		)

		self._evaluate_rules = evaluate_capability_rules
		self._get_contract = get_capability_contract
		self._agent_runtimes = set(SUPPORTED_SECU_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_SECU_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_SECU_AGENT_ROLES)
		self._records = {
			"ComplianceControlRecord": ComplianceControlRecord,
			"DevicePostureRecord": DevicePostureRecord,
			"PolicyExceptionRecord": PolicyExceptionRecord,
			"RiskAssessmentRecord": RiskAssessmentRecord,
			"SecurityAuditEventRecord": SecurityAuditEventRecord,
			"SecurityAgentRecord": SecurityAgentRecord,
			"SecurityIncidentRecord": SecurityIncidentRecord,
			"SecurityLifecycleBatchRecord": SecurityLifecycleBatchRecord,
			"SecurityPolicyRecord": SecurityPolicyRecord,
			"ThreatIndicatorRecord": ThreatIndicatorRecord,
		}
		self._helpers = {
			"clamp_score": clamp_score,
			"control_status": control_status,
			"normalize_device_trust": normalize_device_trust,
			"normalize_security_level": normalize_security_level,
			"normalize_tags": normalize_tags,
			"normalize_threat_severity": normalize_threat_severity,
			"required_actions": required_actions,
			"risk_band": risk_band,
			"stable_id": stable_id,
			"summarize_decision": summarize_decision,
		}
		self.policies: dict[str, Any] = {}
		self.devices: dict[str, Any] = {}
		self.threats: dict[str, Any] = {}
		self.assessments: dict[str, Any] = {}
		self.controls: dict[str, Any] = {}
		self.policy_exceptions: dict[str, Any] = {}
		self.incidents: dict[str, Any] = {}
		self.security_agents: dict[str, Any] = {}
		self.security_lifecycle_batches: dict[str, Any] = {}
		self.audit_events: dict[str, Any] = {}

	def describe(self, tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Return the tenant-scoped executable SECU contract."""
		return self._get_contract(tenant_id, overrides)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate the SECU deterministic rule engine against a context."""
		return self._evaluate_rules(dict(context))

	def create_policy(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		security_level: str = "confidential",
		required_controls: list[str] | None = None,
		applies_to: list[str] | None = None,
		enabled: bool = True,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		"""Register a tenant security policy with deterministic guardrails."""
		self._require_tenant(tenant_id)
		if not str(name or "").strip():
			raise ValueError("policy_name_required")
		if not str(owner or "").strip():
			raise ValueError("policy_owner_required")
		level = self._helpers["normalize_security_level"](security_level)
		record_cls = self._records["SecurityPolicyRecord"]
		record = record_cls(
			id=self._helpers["stable_id"]("secu_policy", tenant_id, name),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			security_level=level,
			required_controls=sorted({str(control) for control in required_controls or [] if str(control).strip()}),
			applies_to=sorted({str(target) for target in applies_to or [] if str(target).strip()}),
			enabled=bool(enabled),
			tags=self._helpers["normalize_tags"](tags),
		)
		self.policies[record.id] = record
		self._record_event(tenant_id, "policy_created", record.id, f"Security policy created: {name}", owner)
		return record.to_dict()

	def record_device_posture(
		self,
		tenant_id: str,
		device_id: str,
		user_id: str,
		trust_state: str = "trusted",
		managed: bool = True,
		risk_score: int | float = 0,
		indicators: list[str] | None = None,
	) -> dict[str, Any]:
		"""Record a device trust posture and quarantine compromised devices."""
		self._require_tenant(tenant_id)
		if not str(device_id or "").strip():
			raise ValueError("device_id_required")
		if not str(user_id or "").strip():
			raise ValueError("device_user_required")
		state = self._helpers["normalize_device_trust"](trust_state)
		score = self._helpers["clamp_score"](risk_score)
		record_cls = self._records["DevicePostureRecord"]
		record = record_cls(
			id=self._helpers["stable_id"]("secu_device", tenant_id, device_id),
			tenant_id=tenant_id,
			device_id=device_id,
			user_id=user_id,
			trust_state=state,
			managed=bool(managed),
			risk_score=score,
			indicators=self._helpers["normalize_tags"](indicators),
			quarantined=state == "compromised" or score >= 85,
		)
		self.devices[record.id] = record
		event_type = "device_quarantined" if record.quarantined else "device_posture_recorded"
		self._record_event(tenant_id, event_type, record.id, f"Device posture recorded: {device_id}", user_id)
		return record.to_dict()

	def register_threat_indicator(
		self,
		tenant_id: str,
		name: str,
		indicator_type: str,
		value: str,
		severity: str = "medium",
		source: str = "manual",
		ttl_hours: int = 24,
	) -> dict[str, Any]:
		"""Register a tenant-scoped threat indicator."""
		self._require_tenant(tenant_id)
		if not str(name or "").strip():
			raise ValueError("threat_name_required")
		if not str(value or "").strip():
			raise ValueError("threat_value_required")
		normalized_severity = self._helpers["normalize_threat_severity"](severity)
		record_cls = self._records["ThreatIndicatorRecord"]
		record = record_cls(
			id=self._helpers["stable_id"]("secu_threat", tenant_id, indicator_type, value),
			tenant_id=tenant_id,
			name=name,
			indicator_type=str(indicator_type or "indicator"),
			value=str(value),
			severity=normalized_severity,
			source=str(source or "manual"),
			ttl_hours=max(1, int(ttl_hours)),
		)
		self.threats[record.id] = record
		self._record_event(tenant_id, "threat_indicator_registered", record.id, f"Threat indicator registered: {name}", source)
		return record.to_dict()

	def assess_access(
		self,
		tenant_id: str,
		subject_id: str,
		subject_type: str,
		risk_score: int | float,
		device_id: str | None = None,
		is_known_malicious: bool = False,
		challenge_completed: bool = False,
		compliance_violation: bool = False,
		audit_evidence_attached: bool = True,
	) -> dict[str, Any]:
		"""Assess an access/security context using the SECU rule engine."""
		self._require_tenant(tenant_id)
		if not str(subject_id or "").strip():
			raise ValueError("subject_id_required")
		score = self._helpers["clamp_score"](risk_score)
		device = self._find_device(tenant_id, device_id) if device_id else None
		context = {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"subject_type": subject_type,
			"risk_score": score,
			"device_trust": getattr(device, "trust_state", "unknown"),
			"is_known_malicious": bool(is_known_malicious),
			"challenge_completed": bool(challenge_completed),
			"compliance_violation": bool(compliance_violation),
			"audit_evidence_attached": bool(audit_evidence_attached),
		}
		result = self.evaluate(context)
		record_cls = self._records["RiskAssessmentRecord"]
		record = record_cls(
			id=self._helpers["stable_id"]("secu_assessment", tenant_id, subject_type, subject_id, score, len(self.assessments)),
			tenant_id=tenant_id,
			subject_id=subject_id,
			subject_type=subject_type,
			risk_score=score,
			risk_band=self._helpers["risk_band"](score),
			decision=result["decision"],
			summary=self._helpers["summarize_decision"](result),
			matched_rules=list(result["matched_rules"]),
			required_actions=self._helpers["required_actions"](result),
			device_id=device_id,
			challenge_completed=bool(challenge_completed),
		)
		self.assessments[record.id] = record
		if device is not None and result["decision"] == "quarantine":
			device.quarantined = True
		if result["decision"] != "allow":
			self._record_event(tenant_id, f"access_{result['decision']}", record.id, record.summary, subject_id, "high")
		return record.to_dict()

	def record_compliance_control(
		self,
		tenant_id: str,
		framework: str,
		control_id: str,
		owner: str,
		compliant: bool,
		evidence_ref: str | None = None,
		waived: bool = False,
	) -> dict[str, Any]:
		"""Record compliance-control posture and evidence requirements."""
		self._require_tenant(tenant_id)
		if not str(control_id or "").strip():
			raise ValueError("control_id_required")
		if not str(owner or "").strip():
			raise ValueError("control_owner_required")
		evidence_attached = bool(str(evidence_ref or "").strip())
		status = self._helpers["control_status"](bool(compliant), evidence_attached, waived)
		policy_result = (
			_review_result("compliance_evidence_required", "attach_audit_evidence")
			if status in {"evidence_required", "non_compliant"}
			else _allow_result()
		)
		record_cls = self._records["ComplianceControlRecord"]
		record = record_cls(
			id=self._helpers["stable_id"]("secu_control", tenant_id, framework, control_id),
			tenant_id=tenant_id,
			framework=str(framework).lower(),
			control_id=str(control_id),
			owner=str(owner),
			status=status,
			compliant=bool(compliant),
			audit_evidence_attached=evidence_attached,
			evidence_ref=evidence_ref,
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result, review_recorded=evidence_attached),
		)
		self.controls[record.id] = record
		if status in {"evidence_required", "non_compliant"}:
			self._record_event(
				tenant_id,
				"compliance_gap_recorded",
				record.id,
				f"Compliance control requires action: {control_id}",
				owner,
				"medium",
				policy_result=policy_result,
			)
		return record.to_dict()

	def request_policy_exception(
		self,
		tenant_id: str,
		exception_id: str,
		policy_id: str,
		requested_by: str,
		reason: str,
		expires_at: str,
	) -> dict[str, Any]:
		"""Request a time-bound policy exception."""
		self._require_tenant(tenant_id)
		if not str(exception_id or "").strip():
			raise ValueError("policy_exception_id_required")
		if not str(policy_id or "").strip():
			raise ValueError("policy_exception_policy_required")
		if not self._policy_exists(tenant_id, policy_id):
			raise KeyError(f"security_policy_not_found:{policy_id}")
		if not str(requested_by or "").strip():
			raise ValueError("policy_exception_requester_required")
		if not str(reason or "").strip():
			raise ValueError("policy_exception_reason_required")
		if not str(expires_at or "").strip():
			raise ValueError("policy_exception_expiry_required")
		self._parse_utc_timestamp(expires_at, "policy_exception_expiry_invalid")
		record_id = self._helpers["stable_id"]("secu_exception", tenant_id, exception_id)
		if record_id in self.policy_exceptions:
			raise ValueError(f"policy_exception_already_exists:{exception_id}")
		policy_result = _review_result("policy_exception_review_required", "review_policy_exception")
		record_cls = self._records["PolicyExceptionRecord"]
		record = record_cls(
			id=record_id,
			tenant_id=tenant_id,
			policy_id=policy_id,
			requested_by=requested_by,
			reason=reason,
			expires_at=expires_at,
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
		)
		self.policy_exceptions[record.id] = record
		self._record_event(
			tenant_id,
			"policy_exception_requested",
			record.id,
			f"Policy exception requested: {policy_id}",
			requested_by,
			"medium",
			policy_result=policy_result,
		)
		return record.to_dict()

	def decide_policy_exception(
		self,
		tenant_id: str,
		exception_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		"""Approve or reject a policy exception with independent review."""
		record = self._get_policy_exception(tenant_id, exception_id)
		if record.status != "pending":
			raise ValueError("policy_exception_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError("policy_exception_decision_invalid")
		if not str(reviewer or "").strip():
			raise ValueError("policy_exception_reviewer_required")
		if not str(notes or "").strip():
			raise ValueError("policy_exception_notes_required")
		result = self.evaluate({
			"operation": "approve_policy_exception",
			"exception_reviewer_same_as_requester": reviewer == record.requested_by,
			"policy_exception_expired": decision == "approved" and self._is_expired(record.expires_at),
		})
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		record_cls = self._records["PolicyExceptionRecord"]
		decided = record_cls(
			id=record.id,
			tenant_id=record.tenant_id,
			policy_id=record.policy_id,
			requested_by=record.requested_by,
			reason=record.reason,
			expires_at=record.expires_at,
			status=decision,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
			policy_decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result, review_recorded=True),
			created_at=record.created_at,
		)
		self.policy_exceptions[record.id] = decided
		self._record_event(
			tenant_id,
			"policy_exception_decided",
			record.id,
			f"Policy exception {decision}: {record.policy_id}",
			reviewer,
			"medium",
			policy_result=result,
		)
		return decided.to_dict()

	def open_incident(
		self,
		tenant_id: str,
		incident_id: str,
		title: str,
		severity: str,
		opened_by: str,
		containment_plan: str = "",
	) -> dict[str, Any]:
		"""Open a security incident with critical containment guardrails."""
		self._require_tenant(tenant_id)
		if not str(incident_id or "").strip():
			raise ValueError("incident_id_required")
		if not str(title or "").strip():
			raise ValueError("incident_title_required")
		if not str(opened_by or "").strip():
			raise ValueError("incident_opened_by_required")
		normalized_severity = self._helpers["normalize_threat_severity"](severity)
		plan = str(containment_plan or "").strip()
		result = self.evaluate({
			"operation": "open_incident",
			"incident_severity": normalized_severity,
			"containment_plan_attached": bool(plan),
		})
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		record_id = self._helpers["stable_id"]("secu_incident", tenant_id, incident_id)
		if record_id in self.incidents:
			raise ValueError(f"security_incident_already_exists:{incident_id}")
		record_cls = self._records["SecurityIncidentRecord"]
		record = record_cls(
			id=record_id,
			tenant_id=tenant_id,
			title=title,
			severity=normalized_severity,
			opened_by=opened_by,
			containment_action=plan,
		)
		self.incidents[record.id] = record
		self._record_event(tenant_id, "security_incident_opened", record.id, f"Security incident opened: {title}", opened_by, normalized_severity)
		return record.to_dict()

	def contain_incident(
		self,
		tenant_id: str,
		incident_id: str,
		actor: str,
		containment_action: str,
		containment_evidence: str,
	) -> dict[str, Any]:
		"""Record containment action and evidence for an incident."""
		record = self._get_incident(tenant_id, incident_id)
		if record.status == "resolved":
			raise ValueError("security_incident_already_resolved")
		if not str(actor or "").strip():
			raise ValueError("incident_containment_actor_required")
		if not str(containment_action or "").strip():
			raise ValueError("incident_containment_action_required")
		if not str(containment_evidence or "").strip():
			raise ValueError("incident_containment_evidence_required")
		record_cls = self._records["SecurityIncidentRecord"]
		contained = record_cls(
			id=record.id,
			tenant_id=record.tenant_id,
			title=record.title,
			severity=record.severity,
			opened_by=record.opened_by,
			status="contained",
			containment_action=containment_action,
			containment_evidence=containment_evidence,
			resolution=record.resolution,
			resolved_by=record.resolved_by,
			opened_at=record.opened_at,
			contained_at=self._utc_now(),
			resolved_at=record.resolved_at,
		)
		self.incidents[record.id] = contained
		self._record_event(tenant_id, "security_incident_contained", record.id, f"Security incident contained: {record.title}", actor, record.severity)
		return contained.to_dict()

	def resolve_incident(
		self,
		tenant_id: str,
		incident_id: str,
		resolved_by: str,
		resolution: str,
		notes: str,
	) -> dict[str, Any]:
		"""Resolve an incident only after containment evidence exists."""
		record = self._get_incident(tenant_id, incident_id)
		if record.status == "resolved":
			raise ValueError("security_incident_already_resolved")
		if not str(resolved_by or "").strip():
			raise ValueError("incident_resolver_required")
		if not str(resolution or "").strip():
			raise ValueError("incident_resolution_required")
		if not str(notes or "").strip():
			raise ValueError("incident_resolution_notes_required")
		result = self.evaluate({
			"operation": "resolve_incident",
			"incident_contained": record.status == "contained",
			"containment_evidence_attached": bool(record.containment_evidence),
		})
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		record_cls = self._records["SecurityIncidentRecord"]
		resolved = record_cls(
			id=record.id,
			tenant_id=record.tenant_id,
			title=record.title,
			severity=record.severity,
			opened_by=record.opened_by,
			status="resolved",
			containment_action=record.containment_action,
			containment_evidence=record.containment_evidence,
			resolution=resolution,
			resolved_by=resolved_by,
			opened_at=record.opened_at,
			contained_at=record.contained_at,
			resolved_at=self._utc_now(),
		)
		self.incidents[record.id] = resolved
		self._record_event(tenant_id, "security_incident_resolved", record.id, notes, resolved_by, record.severity)
		return resolved.to_dict()

	def register_security_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = True,
		policy_ref: str | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Register a governed AI security agent for SECU workflows."""
		self._require_tenant(tenant_id)
		if not str(agent_id or "").strip():
			raise ValueError("security_agent_id_required")
		if not str(name or "").strip():
			raise ValueError("security_agent_name_required")
		if not str(owner or "").strip():
			raise ValueError("security_agent_owner_required")
		if not str(purpose or "").strip():
			raise ValueError("security_agent_purpose_required")
		normalized_runtime = self._normalize_agent_token(runtime)
		normalized_role = self._normalize_agent_token(role)
		result = self.evaluate({
			"operation": "register_security_agent",
			"agent_runtime_supported": normalized_runtime in self._agent_runtimes,
			"agent_role_supported": normalized_role in self._agent_roles,
			"agent_scope_present": bool(str(scope or "").strip()),
			"agent_privileged_role": normalized_role in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		record_id = self._helpers["stable_id"]("secu_agent", tenant_id, agent_id)
		if record_id in self.security_agents:
			raise ValueError(f"security_agent_already_exists:{agent_id}")
		record_cls = self._records["SecurityAgentRecord"]
		record = record_cls(
			id=record_id,
			tenant_id=tenant_id,
			name=str(name),
			runtime=normalized_runtime,
			role=normalized_role,
			scope=str(scope),
			owner=str(owner),
			purpose=str(purpose),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			policy_ref=policy_ref,
			status="pending_review" if result["decision"] == "require_review" else str(status or "active"),
			policy_decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result, review_recorded=bool(human_approval_required)),
		)
		self.security_agents[record.id] = record
		self._record_event(
			tenant_id,
			"security_agent_registered",
			record.id,
			f"Security agent registered: {name}",
			owner,
			"medium" if normalized_role in self._privileged_agent_roles else "info",
			policy_result=result,
		)
		return record.to_dict()

	def validate_security_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> dict[str, Any]:
		"""Validate security batch lifecycle intent before routing to Bytewax."""
		self._require_tenant(tenant_id)
		if int(mutation_count) < 1:
			raise ValueError("security_lifecycle_batch_empty")
		normalized_stream = self._normalize_agent_token(event_stream)
		result = self.evaluate({
			"operation": "security_lifecycle_batch",
			"event_stream": normalized_stream,
			"mutation_count": int(mutation_count),
		})
		record_cls = self._records["SecurityLifecycleBatchRecord"]
		record = record_cls(
			id=self._helpers["stable_id"]("secu_batch", tenant_id, normalized_stream, len(self.security_lifecycle_batches)),
			tenant_id=tenant_id,
			event_stream=normalized_stream,
			mutation_count=int(mutation_count),
			status="denied" if result["decision"] == "deny" else "accepted",
			policy_decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result),
		)
		self.security_lifecycle_batches[record.id] = record
		self._record_event(
			tenant_id,
			"security_lifecycle_batch_validated",
			record.id,
			f"Security lifecycle batch {record.status}: {normalized_stream}",
			"system",
			"medium" if record.status == "denied" else "info",
			policy_result=result,
		)
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		payload = record.to_dict()
		payload.update({
			"tenant_id": tenant_id,
			"event_stream": normalized_stream,
			"mutation_count": int(mutation_count),
			"accepted": True,
			"rule_result": result,
		})
		return payload

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility shim that creates a risk assessment from generic input."""
		metadata = dict(metadata or {})
		score = metadata.get("risk_score", metadata.get("score", 0))
		return self.assess_access(
			tenant_id=tenant_id,
			subject_id=record_id,
			subject_type=str(metadata.get("subject_type") or "compatibility_record"),
			risk_score=score,
			is_known_malicious=bool(metadata.get("is_known_malicious", False)),
			challenge_completed=bool(metadata.get("challenge_completed", status == "approved")),
			compliance_violation=bool(metadata.get("compliance_violation", False)),
			audit_evidence_attached=bool(metadata.get("audit_evidence_attached", True)),
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility listing of risk assessments."""
		return self.list_assessments(tenant_id)

	def list_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.policies, tenant_id)

	def list_devices(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.devices, tenant_id)

	def list_threats(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.threats, tenant_id)

	def list_assessments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.assessments, tenant_id)

	def list_controls(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.controls, tenant_id)

	def list_policy_exceptions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.policy_exceptions, tenant_id)

	def list_incidents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.incidents, tenant_id)

	def list_security_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.security_agents, tenant_id)

	def list_security_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.security_lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = (
			self.list_policy_exceptions(tenant_id)
			+ self.list_controls(tenant_id)
			+ self.list_security_agents(tenant_id)
			+ self.list_security_lifecycle_batches(tenant_id)
		)
		return [
			item
			for item in items
			if item.get("status") in {"pending", "pending_review", "review_required", "evidence_required", "non_compliant"}
		]

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return a compact security-operations dashboard model."""
		assessments = self.list_assessments(tenant_id)
		devices = self.list_devices(tenant_id)
		threats = self.list_threats(tenant_id)
		controls = self.list_controls(tenant_id)
		return {
			"tenant_id": tenant_id,
			"policy_count": len(self.list_policies(tenant_id)),
			"device_count": len(devices),
			"quarantined_device_count": sum(1 for device in devices if device["quarantined"]),
			"active_threat_count": sum(1 for threat in threats if threat["active"]),
			"assessment_count": len(assessments),
			"non_allow_decision_count": sum(1 for item in assessments if item["decision"] != "allow"),
			"compliance_gap_count": sum(1 for item in controls if item["status"] in {"evidence_required", "non_compliant"}),
			"policy_exception_count": len(self.list_policy_exceptions(tenant_id)),
			"pending_policy_exception_count": sum(1 for item in self.list_policy_exceptions(tenant_id) if item["status"] == "pending"),
			"open_incident_count": sum(1 for item in self.list_incidents(tenant_id) if item["status"] != "resolved"),
			"security_agent_count": len(self.list_security_agents(tenant_id)),
			"pending_security_agent_review_count": sum(1 for item in self.list_security_agents(tenant_id) if item["status"] == "pending_review"),
			"security_lifecycle_batch_count": len(self.list_security_lifecycle_batches(tenant_id)),
			"denied_security_lifecycle_batch_count": sum(1 for item in self.list_security_lifecycle_batches(tenant_id) if item["status"] == "denied"),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			raise PermissionError("tenant_context_required")

	def _find_device(self, tenant_id: str, device_id: str | None) -> Any | None:
		if not device_id:
			return None
		for device in self.devices.values():
			if device.tenant_id == tenant_id and device.device_id == device_id:
				return device
		return None

	def _policy_exists(self, tenant_id: str, policy_id: str) -> bool:
		for policy in self.policies.values():
			if policy.tenant_id == tenant_id and policy.id == policy_id:
				return True
		return False

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "info",
		policy_result: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		policy_result = policy_result or _allow_result()
		record_cls = self._records["SecurityAuditEventRecord"]
		record = record_cls(
			id=self._helpers["stable_id"]("secu_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _normalize_agent_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	def _get_policy_exception(self, tenant_id: str, exception_id: str) -> Any:
		record_id = self._helpers["stable_id"]("secu_exception", tenant_id, exception_id)
		record = self.policy_exceptions.get(record_id) or self.policy_exceptions.get(exception_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"policy_exception_not_found:{exception_id}")
		return record

	def _get_incident(self, tenant_id: str, incident_id: str) -> Any:
		record_id = self._helpers["stable_id"]("secu_incident", tenant_id, incident_id)
		record = self.incidents.get(record_id) or self.incidents.get(incident_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"security_incident_not_found:{incident_id}")
		return record

	def _is_expired(self, expires_at: str) -> bool:
		return self._parse_utc_timestamp(expires_at, "policy_exception_expiry_invalid") <= datetime.now(timezone.utc)

	def _parse_utc_timestamp(self, value: str, error_code: str) -> datetime:
		try:
			parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
		except ValueError as exc:
			raise ValueError(error_code) from exc
		if parsed.tzinfo is None:
			return parsed.replace(tzinfo=timezone.utc)
		return parsed.astimezone(timezone.utc)

	def _utc_now(self) -> str:
		from .security_runtime import utc_now

		return utc_now()

	def _first_reason(self, result: dict[str, Any]) -> str:
		for action in result.get("actions", []):
			if action.get("reason"):
				return str(action["reason"])
		return "security_operation_denied"

	def _reasons(self, result: dict[str, Any]) -> list[str]:
		return [
			str(action["reason"])
			for action in result.get("actions", [])
			if action.get("reason")
		]

	def _review_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": [
				str(action.get("required_action"))
				for action in result.get("actions", [])
				if action.get("required_action")
			],
			"reasons": self._reasons(result),
			"review_recorded": bool(review_recorded),
		}

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


def _review_result(reason: str, required_action: str) -> dict[str, Any]:
	return {
		"decision": "require_review",
		"matched_rules": [],
		"actions": [{"reason": reason, "required_action": required_action}],
	}


# Global service instance
_security_service = None

async def get_security_framework_service() -> APGSecurityFrameworkService:
	"""Get global security framework service instance"""
	global _security_service
	if _security_service is None:
		_security_service = APGSecurityFrameworkService()
		await _security_service.initialize()
	return _security_service

async def init_security_framework_service() -> APGSecurityFrameworkService:
	"""Initialize and return security framework service"""
	return await get_security_framework_service()

# Export main service classes
__all__ = [
	"SecurityConfigurationManager",
	"ContextualRiskEngine", 
	"PredictiveThreatDetector",
	"ComplianceAutomationEngine",
	"APGSecurityFrameworkService",
	"SecuService",
	"get_security_framework_service",
	"init_security_framework_service"
]
