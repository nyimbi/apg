"""
APG Multi-Factor Authentication (MFA) - Risk Assessment Service

Intelligent risk assessment service using APG ai_orchestration capability
for behavioral pattern analysis, device trust scoring, and threat intelligence.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from uuid_extensions import uuid7str

from .models import (
	MFAUserProfile, RiskAssessment, RiskFactor, RiskLevel, AuthEvent,
	TrustLevel, DeviceBinding, MFAMethod
)
from .integration import (
	APGIntegrationRouter, AIRiskAssessmentRequest, AIRiskAssessmentResponse,
	create_ai_risk_request, create_audit_event
)


def _log_risk_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log risk assessment operations for debugging and audit"""
	return f"[Risk Analyzer] {operation} for user {user_id}: {details}"


class RiskAnalyzer:
	"""
	Intelligent risk assessment engine that analyzes user behavior, device context,
	location patterns, and threat intelligence to determine authentication risk.
	"""
	
	def __init__(self, 
				apg_integration_router: APGIntegrationRouter,
				database_client: Any,
				threat_intelligence_client: Optional[Any] = None):
		"""Initialize risk analyzer"""
		self.apg_router = apg_integration_router
		self.db = database_client
		self.threat_intel = threat_intelligence_client
		self.logger = logging.getLogger(__name__)
		
		# Risk factor weights (configurable)
		self.risk_weights = {
			"device_trust": 0.25,
			"location_anomaly": 0.20,
			"behavioral_deviation": 0.20,
			"temporal_anomaly": 0.15,
			"threat_intelligence": 0.10,
			"authentication_history": 0.10
		}
		
		# Risk thresholds
		self.risk_thresholds = {
			RiskLevel.MINIMAL: 0.0,
			RiskLevel.LOW: 0.2,
			RiskLevel.MEDIUM: 0.4,
			RiskLevel.HIGH: 0.7,
			RiskLevel.CRITICAL: 0.9
		}
	
	async def assess_authentication_risk(self,
										user_id: str,
										tenant_id: str,
										session_id: str,
										context: Dict[str, Any],
										user_profile: MFAUserProfile) -> RiskAssessment:
		"""
		Perform comprehensive risk assessment for authentication attempt.
		
		Args:
			user_id: User requesting authentication
			tenant_id: Tenant context
			session_id: Authentication session ID
			context: Request context (device, location, network, etc.)
			user_profile: User's MFA profile with history and baselines
		
		Returns:
			Comprehensive risk assessment with scoring and recommendations
		"""
		start_time = datetime.utcnow()
		
		try:
			self.logger.info(_log_risk_operation("assess_risk", user_id, f"session {session_id}"))
			
			# Collect all risk factors
			risk_factors = []
			
			# 1. Device Trust Assessment
			device_risk = await self._assess_device_risk(context, user_profile)
			risk_factors.append(device_risk)
			
			# 2. Location Anomaly Detection
			location_risk = await self._assess_location_risk(context, user_profile)
			risk_factors.append(location_risk)
			
			# 3. Behavioral Deviation Analysis
			behavioral_risk = await self._assess_behavioral_risk(context, user_profile)
			risk_factors.append(behavioral_risk)
			
			# 4. Temporal Pattern Analysis
			temporal_risk = await self._assess_temporal_risk(context, user_profile)
			risk_factors.append(temporal_risk)
			
			# 5. Threat Intelligence Check
			threat_intel_risk = await self._assess_threat_intelligence_risk(context)
			risk_factors.append(threat_intel_risk)
			
			# 6. Authentication History Analysis
			history_risk = await self._assess_authentication_history_risk(user_profile, context)
			risk_factors.append(history_risk)
			
			# 7. Advanced AI-powered risk assessment via APG AI orchestration
			ai_risk_enhancement = await self._perform_ai_risk_assessment(
				user_id, tenant_id, context, user_profile, risk_factors
			)
			
			# 8. Calculate overall risk score
			overall_risk_score = self._calculate_overall_risk_score(risk_factors, ai_risk_enhancement)
			
			# 9. Determine risk level and confidence
			risk_level = self._determine_risk_level(overall_risk_score)
			confidence_level = self._calculate_confidence_level(risk_factors, ai_risk_enhancement)
			
			# 10. Generate recommendations
			recommendations = self._generate_risk_recommendations(
				risk_factors, overall_risk_score, risk_level, user_profile
			)
			
			# Create risk assessment record
			processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			risk_assessment = RiskAssessment(
				user_id=user_id,
				tenant_id=tenant_id,
				session_id=session_id,
				overall_risk_score=overall_risk_score,
				risk_level=risk_level,
				confidence_level=confidence_level,
				risk_factors=risk_factors,
				device_context=context.get("device", {}),
				location_context=context.get("location", {}),
				behavioral_context=context.get("behavioral", {}),
				temporal_context=context.get("temporal", {}),
				recommended_auth_methods=recommendations["auth_methods"],
				recommended_actions=recommendations["actions"],
				model_version=ai_risk_enhancement.get("model_version", "risk_analyzer_v1"),
				processing_time_ms=processing_time_ms,
				created_by=user_id,
				updated_by=user_id
			)
			
			# Store risk assessment
			await self._store_risk_assessment(risk_assessment)
			
			# Update user risk profile
			await self._update_user_risk_profile(user_profile, risk_assessment)
			
			self.logger.info(_log_risk_operation(
				"assess_risk_complete", user_id,
				f"risk_score={overall_risk_score:.3f}, level={risk_level}, time={processing_time_ms}ms"
			))
			
			return risk_assessment
			
		except Exception as e:
			self.logger.error(f"Risk assessment error for user {user_id}: {str(e)}", exc_info=True)
			
			# Return conservative high-risk assessment on error
			return RiskAssessment(
				user_id=user_id,
				tenant_id=tenant_id,
				session_id=session_id,
				overall_risk_score=0.8,  # High risk on error
				risk_level=RiskLevel.HIGH,
				confidence_level=0.3,  # Low confidence due to error
				risk_factors=[{
					"factor_type": "system_error",
					"factor_name": "Risk Assessment System Error",
					"risk_score": 0.8,
					"confidence": 0.3,
					"evidence": {"error": str(e)},
					"mitigation_suggestions": ["Require additional authentication", "Manual review"]
				}],
				model_version="error_fallback",
				processing_time_ms=0,
				created_by=user_id,
				updated_by=user_id
			)
	
	async def _assess_device_risk(self, context: Dict[str, Any], user_profile: MFAUserProfile) -> RiskFactor:
		"""Assess risk based on device context and trust history"""
		device_context = context.get("device", {})
		device_id = device_context.get("device_id", "unknown")
		
		# Check if device is in user's trusted devices
		is_trusted_device = device_id in user_profile.trusted_devices
		
		# Get device trust score from profile
		device_trust_score = user_profile.device_trust_scores.get(device_id, 0.0)
		
		# Analyze device characteristics
		device_fingerprint = device_context.get("fingerprint", "")
		is_mobile = device_context.get("is_mobile", False)
		is_rooted_jailbroken = device_context.get("is_rooted_jailbroken", False)
		os_version = device_context.get("os_version", "")
		app_version = device_context.get("app_version", "")
		
		# Calculate device risk factors
		risk_factors = []
		
		if not is_trusted_device:
			risk_factors.append(("untrusted_device", 0.4, "Device not in trusted list"))
		
		if device_trust_score < 0.3:
			risk_factors.append(("low_device_trust", 0.3, f"Low trust score: {device_trust_score}"))
		
		if is_rooted_jailbroken:
			risk_factors.append(("rooted_jailbroken", 0.6, "Device is rooted/jailbroken"))
		
		if not device_fingerprint:
			risk_factors.append(("no_fingerprint", 0.2, "No device fingerprint available"))
		
		# Calculate overall device risk
		if risk_factors:
			device_risk = min(sum(factor[1] for factor in risk_factors) / len(risk_factors), 1.0)
		else:
			device_risk = 0.1  # Minimal risk for trusted devices
		
		# Device risk confidence based on available data
		confidence = 0.9 if device_fingerprint and device_id != "unknown" else 0.5
		
		return RiskFactor(
			factor_type="device_trust",
			factor_name="Device Trust Assessment",
			risk_score=device_risk,
			confidence=confidence,
			evidence={
				"device_id": device_id,
				"is_trusted": is_trusted_device,
				"trust_score": device_trust_score,
				"is_rooted_jailbroken": is_rooted_jailbroken,
				"risk_factors": [f[2] for f in risk_factors]
			},
			mitigation_suggestions=[
				"Device verification required",
				"Additional authentication factor",
				"Device registration process"
			] if device_risk > 0.5 else []
		)
	
	async def _assess_location_risk(self, context: Dict[str, Any], user_profile: MFAUserProfile) -> RiskFactor:
		"""Assess risk based on location context and user's location history"""
		location_context = context.get("location", {})
		
		current_location = {
			"ip_address": location_context.get("ip_address", ""),
			"country": location_context.get("country", ""),
			"city": location_context.get("city", ""),
			"coordinates": location_context.get("coordinates", [])
		}
		
		# Check against user's trusted locations
		trusted_locations = user_profile.trusted_locations
		is_trusted_location = self._is_location_trusted(current_location, trusted_locations)
		
		# Calculate distance from usual locations
		min_distance = self._calculate_min_distance_from_trusted_locations(
			current_location, trusted_locations
		)
		
		# Analyze location characteristics
		is_tor_proxy = location_context.get("is_tor", False)
		is_vpn = location_context.get("is_vpn", False)
		is_datacenter = location_context.get("is_datacenter", False)
		reputation_score = location_context.get("reputation_score", 0.5)
		
		# Calculate location risk factors
		risk_factors = []
		
		if not is_trusted_location:
			if min_distance > 1000:  # More than 1000km from trusted locations
				risk_factors.append(("distant_location", 0.5, f"Distance: {min_distance}km"))
			else:
				risk_factors.append(("new_location", 0.2, "New but nearby location"))
		
		if is_tor_proxy:
			risk_factors.append(("tor_usage", 0.8, "Tor network detected"))
		
		if is_vpn:
			risk_factors.append(("vpn_usage", 0.3, "VPN detected"))
		
		if is_datacenter:
			risk_factors.append(("datacenter_ip", 0.4, "Datacenter IP address"))
		
		if reputation_score < 0.3:
			risk_factors.append(("bad_reputation", 0.6, f"Low IP reputation: {reputation_score}"))
		
		# Calculate overall location risk
		if risk_factors:
			location_risk = min(sum(factor[1] for factor in risk_factors) / len(risk_factors), 1.0)
		else:
			location_risk = 0.1  # Minimal risk for trusted locations
		
		# Location confidence based on data quality
		confidence = 0.8 if current_location["ip_address"] and current_location["country"] else 0.4
		
		return RiskFactor(
			factor_type="location_anomaly",
			factor_name="Location Anomaly Analysis",
			risk_score=location_risk,
			confidence=confidence,
			evidence={
				"current_location": current_location,
				"is_trusted": is_trusted_location,
				"min_distance_km": min_distance,
				"is_tor": is_tor_proxy,
				"is_vpn": is_vpn,
				"reputation_score": reputation_score,
				"risk_factors": [f[2] for f in risk_factors]
			},
			mitigation_suggestions=[
				"Location verification required",
				"Email/SMS confirmation",
				"Geographic restrictions"
			] if location_risk > 0.5 else []
		)
	
	async def _assess_behavioral_risk(self, context: Dict[str, Any], user_profile: MFAUserProfile) -> RiskFactor:
		"""Assess risk based on behavioral patterns and deviations from baseline"""
		behavioral_context = context.get("behavioral", {})
		baseline = user_profile.behavioral_baseline
		
		if not baseline:
			# No baseline established yet
			return RiskFactor(
				factor_type="behavioral_deviation",
				factor_name="Behavioral Pattern Analysis",
				risk_score=0.3,  # Medium risk for no baseline
				confidence=0.2,
				evidence={"status": "no_baseline_established"},
				mitigation_suggestions=["Establish behavioral baseline"]
			)
		
		# Analyze behavioral patterns
		patterns_to_analyze = [
			"typing_speed", "typing_rhythm", "mouse_movements", 
			"click_patterns", "navigation_patterns", "session_duration"
		]
		
		deviations = []
		for pattern in patterns_to_analyze:
			current_value = behavioral_context.get(pattern)
			baseline_value = baseline.get(pattern)
			
			if current_value is not None and baseline_value is not None:
				deviation = self._calculate_behavioral_deviation(
					current_value, baseline_value, pattern
				)
				deviations.append((pattern, deviation))
		
		if not deviations:
			return RiskFactor(
				factor_type="behavioral_deviation",
				factor_name="Behavioral Pattern Analysis",
				risk_score=0.4,
				confidence=0.3,
				evidence={"status": "insufficient_behavioral_data"},
				mitigation_suggestions=["Collect more behavioral data"]
			)
		
		# Calculate overall behavioral risk
		avg_deviation = sum(dev[1] for dev in deviations) / len(deviations)
		behavioral_risk = min(avg_deviation, 1.0)
		
		# High deviations indicate potential compromise or different user
		if behavioral_risk > 0.7:
			risk_level = "high"
			suggestions = ["Strong authentication required", "Account security review"]
		elif behavioral_risk > 0.4:
			risk_level = "medium"
			suggestions = ["Additional verification recommended"]
		else:
			risk_level = "low"
			suggestions = []
		
		confidence = 0.8 if len(deviations) >= 3 else 0.5
		
		return RiskFactor(
			factor_type="behavioral_deviation",
			factor_name="Behavioral Pattern Analysis",
			risk_score=behavioral_risk,
			confidence=confidence,
			evidence={
				"average_deviation": avg_deviation,
				"pattern_deviations": dict(deviations),
				"risk_level": risk_level,
				"patterns_analyzed": len(deviations)
			},
			mitigation_suggestions=suggestions
		)
	
	async def _assess_temporal_risk(self, context: Dict[str, Any], user_profile: MFAUserProfile) -> RiskFactor:
		"""Assess risk based on timing patterns and unusual access times"""
		temporal_context = context.get("temporal", {})
		current_time = datetime.utcnow()
		
		# Get user's typical access patterns from history
		auth_history = await self._get_recent_auth_history(user_profile.user_id, user_profile.tenant_id)
		
		if not auth_history:
			return RiskFactor(
				factor_type="temporal_anomaly",
				factor_name="Temporal Pattern Analysis",
				risk_score=0.2,
				confidence=0.3,
				evidence={"status": "no_auth_history"},
				mitigation_suggestions=[]
			)
		
		# Analyze temporal patterns
		hour_of_day = current_time.hour
		day_of_week = current_time.weekday()
		
		# Calculate user's typical access times
		typical_hours = self._calculate_typical_access_hours(auth_history)
		typical_days = self._calculate_typical_access_days(auth_history)
		
		# Check for anomalies
		anomalies = []
		
		if hour_of_day not in typical_hours:
			hour_risk = 0.3 if 22 <= hour_of_day <= 6 else 0.2  # Higher risk at night
			anomalies.append(("unusual_hour", hour_risk, f"Access at {hour_of_day}:00"))
		
		if day_of_week not in typical_days:
			day_risk = 0.2
			anomalies.append(("unusual_day", day_risk, f"Access on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}"))
		
		# Check for rapid successive attempts
		last_auth = user_profile.last_successful_auth
		if last_auth:
			time_since_last = (current_time - last_auth).total_seconds()
			if time_since_last < 60:  # Less than 1 minute
				anomalies.append(("rapid_retry", 0.4, f"Last auth {time_since_last:.0f}s ago"))
		
		# Check for multiple locations in short time (impossible travel)
		location_context = context.get("location", {})
		if location_context and auth_history:
			impossible_travel = self._detect_impossible_travel(location_context, auth_history[0] if auth_history else None)
			if impossible_travel:
				anomalies.append(("impossible_travel", 0.8, "Impossible travel detected"))
		
		# Calculate overall temporal risk
		if anomalies:
			temporal_risk = min(sum(anomaly[1] for anomaly in anomalies) / len(anomalies), 1.0)
		else:
			temporal_risk = 0.1
		
		confidence = 0.8 if len(auth_history) >= 10 else 0.5
		
		return RiskFactor(
			factor_type="temporal_anomaly",
			factor_name="Temporal Pattern Analysis",
			risk_score=temporal_risk,
			confidence=confidence,
			evidence={
				"current_hour": hour_of_day,
				"current_day": day_of_week,
				"typical_hours": typical_hours,
				"typical_days": typical_days,
				"anomalies": [a[2] for a in anomalies],
				"auth_history_count": len(auth_history)
			},
			mitigation_suggestions=[
				"Time-based access restrictions",
				"Unusual timing verification"
			] if temporal_risk > 0.5 else []
		)
	
	async def _assess_threat_intelligence_risk(self, context: Dict[str, Any]) -> RiskFactor:
		"""Assess risk based on current threat intelligence data"""
		if not self.threat_intel:
			return RiskFactor(
				factor_type="threat_intelligence",
				factor_name="Threat Intelligence Check",
				risk_score=0.2,  # Conservative default
				confidence=0.1,
				evidence={"status": "threat_intel_not_available"},
				mitigation_suggestions=[]
			)
		
		ip_address = context.get("location", {}).get("ip_address", "")
		user_agent = context.get("device", {}).get("user_agent", "")
		
		threats_detected = []
		
		try:
			# Check IP against threat intelligence feeds
			if ip_address:
				ip_threat_data = await self.threat_intel.check_ip_reputation(ip_address)
				if ip_threat_data.get("is_malicious"):
					threats_detected.append(("malicious_ip", 0.9, f"IP in threat feed: {ip_threat_data.get('category')}"))
				elif ip_threat_data.get("reputation_score", 1.0) < 0.3:
					threats_detected.append(("low_ip_reputation", 0.4, f"Low IP reputation: {ip_threat_data.get('reputation_score')}"))
			
			# Check for known attack patterns
			attack_patterns = await self.threat_intel.check_attack_patterns(context)
			for pattern in attack_patterns:
				threats_detected.append(("attack_pattern", pattern["risk_score"], pattern["description"]))
			
			# Check for compromised credentials indicators
			if user_agent:
				credential_check = await self.threat_intel.check_credential_exposure(
					context.get("user_id", ""), ip_address
				)
				if credential_check.get("potentially_compromised"):
					threats_detected.append(("credential_exposure", 0.7, "Potential credential compromise detected"))
		
		except Exception as e:
			self.logger.warning(f"Threat intelligence check failed: {str(e)}")
			return RiskFactor(
				factor_type="threat_intelligence",
				factor_name="Threat Intelligence Check",
				risk_score=0.3,  # Conservative on error
				confidence=0.2,
				evidence={"status": "threat_intel_error", "error": str(e)},
				mitigation_suggestions=["Manual security review"]
			)
		
		# Calculate threat intelligence risk
		if threats_detected:
			threat_risk = min(max(threat[1] for threat in threats_detected), 1.0)
			suggestions = ["Security team review", "Enhanced monitoring", "Account verification"]
		else:
			threat_risk = 0.1
			suggestions = []
		
		return RiskFactor(
			factor_type="threat_intelligence",
			factor_name="Threat Intelligence Check",
			risk_score=threat_risk,
			confidence=0.9,
			evidence={
				"threats_detected": [threat[2] for threat in threats_detected],
				"ip_checked": bool(ip_address),
				"patterns_checked": True
			},
			mitigation_suggestions=suggestions
		)
	
	async def _assess_authentication_history_risk(self, user_profile: MFAUserProfile, context: Dict[str, Any]) -> RiskFactor:
		"""Assess risk based on user's authentication history and patterns"""
		total_auths = user_profile.total_authentications
		successful_auths = user_profile.successful_authentications
		failed_auths = user_profile.failed_authentications
		
		if total_auths == 0:
			return RiskFactor(
				factor_type="authentication_history",
				factor_name="Authentication History Analysis",
				risk_score=0.4,  # New users have medium risk
				confidence=0.2,
				evidence={"status": "new_user", "total_authentications": 0},
				mitigation_suggestions=["New user verification process"]
			)
		
		# Calculate success rate
		success_rate = successful_auths / total_auths if total_auths > 0 else 0.0
		
		# Analyze recent failure patterns
		recent_failures = failed_auths
		if user_profile.last_successful_auth:
			hours_since_success = (datetime.utcnow() - user_profile.last_successful_auth).total_seconds() / 3600
		else:
			hours_since_success = float('inf')
		
		# Calculate risk factors
		risk_factors = []
		
		if success_rate < 0.5:
			risk_factors.append(("low_success_rate", 0.6, f"Success rate: {success_rate:.2f}"))
		
		if recent_failures > 5:
			risk_factors.append(("recent_failures", 0.4, f"Recent failures: {recent_failures}"))
		
		if hours_since_success > 168:  # More than a week
			risk_factors.append(("long_inactivity", 0.3, f"Inactive for {hours_since_success:.0f} hours"))
		
		# Check for lockout status
		if user_profile.lockout_until and user_profile.lockout_until > datetime.utcnow():
			risk_factors.append(("account_locked", 0.8, "Account is currently locked"))
		
		# Calculate overall history risk
		if risk_factors:
			history_risk = min(sum(factor[1] for factor in risk_factors) / len(risk_factors), 1.0)
		else:
			history_risk = 0.1
		
		confidence = min(total_auths / 50, 1.0)  # Higher confidence with more history
		
		return RiskFactor(
			factor_type="authentication_history",
			factor_name="Authentication History Analysis",
			risk_score=history_risk,
			confidence=confidence,
			evidence={
				"total_authentications": total_auths,
				"success_rate": success_rate,
				"recent_failures": recent_failures,
				"hours_since_success": hours_since_success,
				"is_locked": bool(user_profile.lockout_until),
				"risk_factors": [f[2] for f in risk_factors]
			},
			mitigation_suggestions=[
				"Account security review",
				"Password reset recommendation"
			] if history_risk > 0.5 else []
		)
	
	async def _perform_ai_risk_assessment(self,
										 user_id: str,
										 tenant_id: str,
										 context: Dict[str, Any],
										 user_profile: MFAUserProfile,
										 risk_factors: List[RiskFactor]) -> Dict[str, Any]:
		"""Perform advanced AI-powered risk assessment via APG AI orchestration"""
		try:
			# Prepare AI request data
			ai_request = create_ai_risk_request(user_id, {
				"session_context": context,
				"user_profile_data": {
					"base_risk_score": user_profile.base_risk_score,
					"trust_score": user_profile.trust_score,
					"total_authentications": user_profile.total_authentications,
					"success_rate": user_profile.successful_authentications / max(user_profile.total_authentications, 1),
					"behavioral_baseline": user_profile.behavioral_baseline,
					"device_trust_scores": user_profile.device_trust_scores
				},
				"risk_factors": [rf.model_dump() for rf in risk_factors],
				"analysis_type": "authentication_risk_assessment"
			}, tenant_id)
			
			# Send to AI orchestration capability
			ai_response = await self.apg_router.route_integration_event(ai_request)
			
			if isinstance(ai_response, AIRiskAssessmentResponse):
				return {
					"ai_risk_score": ai_response.risk_score,
					"ai_confidence": ai_response.confidence_score,
					"ai_factors": ai_response.risk_factors,
					"ai_recommendations": ai_response.recommended_actions,
					"model_version": ai_response.model_version,
					"processing_time_ms": ai_response.processing_time_ms
				}
			else:
				self.logger.warning("AI risk assessment returned unexpected response type")
				return self._get_fallback_ai_assessment()
		
		except Exception as e:
			self.logger.error(f"AI risk assessment failed: {str(e)}")
			return self._get_fallback_ai_assessment()
	
	def _get_fallback_ai_assessment(self) -> Dict[str, Any]:
		"""Fallback AI assessment when AI orchestration is unavailable"""
		return {
			"ai_risk_score": 0.5,
			"ai_confidence": 0.3,
			"ai_factors": [],
			"ai_recommendations": ["Use fallback risk assessment"],
			"model_version": "fallback",
			"processing_time_ms": 0
		}
	
	def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor], ai_enhancement: Dict[str, Any]) -> float:
		"""Calculate overall risk score using weighted factors and AI enhancement"""
		# Calculate weighted risk from traditional factors
		traditional_risk = 0.0
		total_weight = 0.0
		
		for factor in risk_factors:
			weight = self.risk_weights.get(factor.factor_type, 0.1)
			weighted_risk = factor.risk_score * factor.confidence * weight
			traditional_risk += weighted_risk
			total_weight += weight
		
		if total_weight > 0:
			traditional_risk = traditional_risk / total_weight
		
		# Blend with AI assessment
		ai_risk = ai_enhancement.get("ai_risk_score", 0.5)
		ai_confidence = ai_enhancement.get("ai_confidence", 0.3)
		
		# Weight blend: 70% traditional, 30% AI (adjust based on AI confidence)
		ai_weight = 0.3 * ai_confidence
		traditional_weight = 1.0 - ai_weight
		
		overall_risk = (traditional_risk * traditional_weight) + (ai_risk * ai_weight)
		
		return min(overall_risk, 1.0)
	
	def _determine_risk_level(self, risk_score: float) -> RiskLevel:
		"""Determine categorical risk level from score"""
		for level, threshold in reversed(list(self.risk_thresholds.items())):
			if risk_score >= threshold:
				return level
		return RiskLevel.MINIMAL
	
	def _calculate_confidence_level(self, risk_factors: List[RiskFactor], ai_enhancement: Dict[str, Any]) -> float:
		"""Calculate overall confidence in risk assessment"""
		if not risk_factors:
			return 0.1
		
		# Average confidence from risk factors
		factor_confidence = sum(factor.confidence for factor in risk_factors) / len(risk_factors)
		
		# AI confidence
		ai_confidence = ai_enhancement.get("ai_confidence", 0.3)
		
		# Weighted average
		overall_confidence = (factor_confidence * 0.7) + (ai_confidence * 0.3)
		
		return min(overall_confidence, 1.0)
	
	def _generate_risk_recommendations(self,
									  risk_factors: List[RiskFactor],
									  risk_score: float,
									  risk_level: RiskLevel,
									  user_profile: MFAUserProfile) -> Dict[str, List[str]]:
		"""Generate authentication method and action recommendations based on risk"""
		auth_methods = []
		actions = []
		
		# Base authentication requirements
		if risk_level == RiskLevel.CRITICAL:
			auth_methods.extend(["biometric_multi_modal", "hardware_token", "backup_codes"])
			actions.extend(["Account security review", "Immediate notification", "Enhanced monitoring"])
		elif risk_level == RiskLevel.HIGH:
			auth_methods.extend(["biometric_face", "token_totp", "sms"])
			actions.extend(["Security notification", "Location verification"])
		elif risk_level == RiskLevel.MEDIUM:
			auth_methods.extend(["token_totp", "biometric_face"])
			actions.extend(["Monitor session"])
		else:
			auth_methods.extend(["token_totp"])
		
		# Add specific recommendations from risk factors
		for factor in risk_factors:
			if factor.mitigation_suggestions:
				actions.extend(factor.mitigation_suggestions)
		
		# Remove duplicates and return
		return {
			"auth_methods": list(set(auth_methods)),
			"actions": list(set(actions))
		}
	
	# Helper methods for risk calculations
	
	def _is_location_trusted(self, current_location: Dict[str, Any], trusted_locations: List[Dict[str, Any]]) -> bool:
		"""Check if current location matches any trusted location"""
		if not trusted_locations:
			return False
		
		current_ip = current_location.get("ip_address", "")
		current_country = current_location.get("country", "")
		
		for trusted in trusted_locations:
			if current_ip == trusted.get("ip_address", ""):
				return True
			if current_country == trusted.get("country", "") and current_country:
				return True
		
		return False
	
	def _calculate_min_distance_from_trusted_locations(self, current_location: Dict[str, Any], trusted_locations: List[Dict[str, Any]]) -> float:
		"""Calculate minimum distance from current location to any trusted location"""
		if not trusted_locations or not current_location.get("coordinates"):
			return 0.0  # Can't calculate distance
		
		current_coords = current_location["coordinates"]
		if len(current_coords) != 2:
			return 0.0
		
		min_distance = float('inf')
		for trusted in trusted_locations:
			trusted_coords = trusted.get("coordinates", [])
			if len(trusted_coords) == 2:
				distance = self._calculate_haversine_distance(current_coords, trusted_coords)
				min_distance = min(min_distance, distance)
		
		return min_distance if min_distance != float('inf') else 0.0
	
	def _calculate_haversine_distance(self, coords1: List[float], coords2: List[float]) -> float:
		"""Calculate distance between two coordinate points using Haversine formula"""
		import math
		
		lat1, lon1 = math.radians(coords1[0]), math.radians(coords1[1])
		lat2, lon2 = math.radians(coords2[0]), math.radians(coords2[1])
		
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		
		a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
		c = 2 * math.asin(math.sqrt(a))
		
		# Earth's radius in kilometers
		r = 6371
		
		return c * r
	
	def _calculate_behavioral_deviation(self, current_value: Any, baseline_value: Any, pattern_type: str) -> float:
		"""Calculate behavioral deviation for specific pattern type"""
		if pattern_type in ["typing_speed", "session_duration"]:
			# Numeric comparison
			try:
				current = float(current_value)
				baseline = float(baseline_value)
				if baseline == 0:
					return 0.5
				deviation = abs(current - baseline) / baseline
				return min(deviation, 1.0)
			except (ValueError, TypeError):
				return 0.5
		
		elif pattern_type in ["typing_rhythm", "mouse_movements", "click_patterns"]:
			# Pattern similarity comparison (simplified)
			if isinstance(current_value, list) and isinstance(baseline_value, list):
				if len(current_value) != len(baseline_value):
					return 0.6
				
				differences = sum(abs(c - b) for c, b in zip(current_value, baseline_value))
				max_possible_diff = len(current_value) * max(max(current_value + baseline_value), 1)
				return min(differences / max_possible_diff, 1.0)
		
		# Default case
		return 0.5 if current_value != baseline_value else 0.0
	
	def _calculate_typical_access_hours(self, auth_history: List[Dict[str, Any]]) -> List[int]:
		"""Calculate user's typical access hours from history"""
		if not auth_history:
			return list(range(24))  # Allow all hours if no history
		
		hour_counts = {}
		for event in auth_history:
			if event.get("timestamp"):
				try:
					timestamp = datetime.fromisoformat(event["timestamp"])
					hour = timestamp.hour
					hour_counts[hour] = hour_counts.get(hour, 0) + 1
				except (ValueError, TypeError):
					continue
		
		if not hour_counts:
			return list(range(24))
		
		# Consider hours with >10% of total accesses as typical
		total_accesses = sum(hour_counts.values())
		threshold = total_accesses * 0.1
		
		typical_hours = [hour for hour, count in hour_counts.items() if count >= threshold]
		
		# Ensure at least some hours are considered typical
		if not typical_hours:
			typical_hours = [hour for hour, count in sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
		
		return typical_hours
	
	def _calculate_typical_access_days(self, auth_history: List[Dict[str, Any]]) -> List[int]:
		"""Calculate user's typical access days from history"""
		if not auth_history:
			return list(range(7))  # Allow all days if no history
		
		day_counts = {}
		for event in auth_history:
			if event.get("timestamp"):
				try:
					timestamp = datetime.fromisoformat(event["timestamp"])
					day = timestamp.weekday()
					day_counts[day] = day_counts.get(day, 0) + 1
				except (ValueError, TypeError):
					continue
		
		if not day_counts:
			return list(range(7))
		
		# Consider days with >5% of total accesses as typical
		total_accesses = sum(day_counts.values())
		threshold = total_accesses * 0.05
		
		typical_days = [day for day, count in day_counts.items() if count >= threshold]
		
		# Ensure at least some days are considered typical
		if not typical_days:
			typical_days = [day for day, count in sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
		
		return typical_days
	
	def _detect_impossible_travel(self, current_location: Dict[str, Any], last_auth_event: Optional[Dict[str, Any]]) -> bool:
		"""Detect impossible travel between locations"""
		if not last_auth_event or not current_location.get("coordinates"):
			return False
		
		last_location = last_auth_event.get("location", {})
		if not last_location.get("coordinates"):
			return False
		
		try:
			last_timestamp = datetime.fromisoformat(last_auth_event["timestamp"])
			time_diff_hours = (datetime.utcnow() - last_timestamp).total_seconds() / 3600
			
			if time_diff_hours >= 24:  # More than 24 hours - travel is possible
				return False
			
			distance_km = self._calculate_haversine_distance(
				current_location["coordinates"],
				last_location["coordinates"]
			)
			
			# Maximum possible travel speed (commercial aviation)
			max_speed_kmh = 900
			max_possible_distance = max_speed_kmh * time_diff_hours
			
			return distance_km > max_possible_distance
		
		except (ValueError, TypeError, KeyError):
			return False
	
	# Database operations (placeholders - implement based on your database client)
	
	async def _store_risk_assessment(self, risk_assessment: RiskAssessment) -> None:
		"""Store risk assessment in database"""
		# Implementation depends on database client
		pass
	
	async def _update_user_risk_profile(self, user_profile: MFAUserProfile, risk_assessment: RiskAssessment) -> None:
		"""Update user's risk profile based on assessment"""
		# Update base risk score with exponential moving average
		alpha = 0.1  # Learning rate
		user_profile.base_risk_score = (
			(1 - alpha) * user_profile.base_risk_score +
			alpha * risk_assessment.overall_risk_score
		)
		
		# Update ML insights
		user_profile.ml_insights["last_risk_assessment"] = {
			"timestamp": risk_assessment.created_at.isoformat(),
			"risk_score": risk_assessment.overall_risk_score,
			"risk_level": risk_assessment.risk_level,
			"confidence": risk_assessment.confidence_level
		}
		
		# Implementation depends on database client
		pass
	
	async def _get_recent_auth_history(self, user_id: str, tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
		"""Get recent authentication history for user"""
		# Implementation depends on database client
		return []


__all__ = ["RiskAnalyzer"]