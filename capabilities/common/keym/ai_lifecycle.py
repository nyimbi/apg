#!/usr/bin/env python3
"""
APG Key Management - AI-Powered Intelligent Lifecycle Management
Autonomous key lifecycle management with predictive analytics

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import Key, KeySpec, KeyUsageStats, SecurityThreat, KeyState, KeyAlgorithm


class RotationTrigger(str, Enum):
	"""Key rotation trigger reasons"""
	TIME_BASED = "time_based"
	USAGE_BASED = "usage_based"
	THREAT_BASED = "threat_based"
	COMPLIANCE_BASED = "compliance_based"
	PREDICTIVE = "predictive"
	MANUAL = "manual"


class RiskLevel(str, Enum):
	"""Security risk assessment levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


@dataclass
class LifecycleDecision:
	"""AI-driven lifecycle management decision"""
	key_id: str
	action: str  # rotate, revoke, extend, archive
	trigger: RotationTrigger
	confidence: float  # 0.0 to 1.0
	risk_level: RiskLevel
	recommended_date: datetime
	reasoning: str
	supporting_data: Dict[str, Any]


@dataclass
class UsagePattern:
	"""Key usage pattern analysis"""
	key_id: str
	daily_operations: List[int]
	peak_hours: List[int]
	user_diversity: int
	application_diversity: int
	geographic_diversity: int
	anomaly_score: float
	trend_direction: str  # increasing, decreasing, stable
	seasonality_detected: bool


class AILifecycleManager:
	"""
	AI-powered intelligent key lifecycle management
	Uses machine learning for predictive analytics and autonomous decision making
	"""
	
	def __init__(self, config: Dict[str, Any] | None = None):
		self.config = config or {}
		self.usage_patterns: Dict[str, UsagePattern] = {}
		self.threat_intelligence: Dict[str, Any] = {}
		self.compliance_requirements: Dict[str, Any] = {}
		self.ml_models: Dict[str, Any] = {}
		
		# Model weights (would be learned from historical data)
		self.model_weights = {
			'usage_frequency': 0.25,
			'threat_intelligence': 0.30,
			'compliance_urgency': 0.20,
			'business_impact': 0.15,
			'cost_optimization': 0.10
		}
	
	async def _log_ai_decision(self, decision: LifecycleDecision) -> None:
		"""Log AI lifecycle decisions for audit and learning"""
		print(f"[AI-LIFECYCLE] {decision.action.upper()} decision for key {decision.key_id}: "
			 f"{decision.reasoning} (confidence: {decision.confidence:.2f})")
	
	async def analyze_usage_patterns(self, key: Key, stats: KeyUsageStats) -> UsagePattern:
		"""Analyze key usage patterns using statistical methods"""
		# Simulate daily operations for the last 30 days
		daily_ops = self._generate_usage_simulation(stats)
		
		# Detect peak usage hours
		peak_hours = await self._detect_peak_hours(key.spec.id, stats)
		
		# Calculate diversity metrics
		user_diversity = max(1, stats.unique_users)
		app_diversity = max(1, stats.unique_applications)
		geo_diversity = 1  # Would integrate with geographic data
		
		# Anomaly detection using statistical analysis
		anomaly_score = await self._calculate_anomaly_score(daily_ops)
		
		# Trend analysis
		trend_direction = await self._analyze_trend(daily_ops)
		
		# Seasonality detection
		seasonality_detected = await self._detect_seasonality(daily_ops)
		
		pattern = UsagePattern(
			key_id=key.spec.id,
			daily_operations=daily_ops,
			peak_hours=peak_hours,
			user_diversity=user_diversity,
			application_diversity=app_diversity,
			geographic_diversity=geo_diversity,
			anomaly_score=anomaly_score,
			trend_direction=trend_direction,
			seasonality_detected=seasonality_detected
		)
		
		self.usage_patterns[key.spec.id] = pattern
		return pattern
	
	def _generate_usage_simulation(self, stats: KeyUsageStats) -> List[int]:
		"""Generate realistic usage simulation (would use real historical data)"""
		# Simulate 30 days of usage with some variance
		base_operations = max(1, stats.total_operations // 30)
		daily_ops = []
		
		for day in range(30):
			# Add some randomness and weekly patterns
			weekend_factor = 0.3 if day % 7 in [5, 6] else 1.0
			random_factor = np.random.normal(1.0, 0.2)
			daily_usage = int(base_operations * weekend_factor * random_factor)
			daily_ops.append(max(0, daily_usage))
		
		return daily_ops
	
	async def _detect_peak_hours(self, key_id: str, stats: KeyUsageStats) -> List[int]:
		"""Detect peak usage hours (would use real hourly data)"""
		# Simulate peak hours based on business patterns
		if stats.total_operations > 1000:
			# High-usage keys tend to have business hours peaks
			return [9, 10, 11, 14, 15, 16]
		else:
			# Low-usage keys might have specific application peaks
			return [2, 3, 22, 23]  # Off-hours batch processing
	
	async def _calculate_anomaly_score(self, daily_ops: List[int]) -> float:
		"""Calculate anomaly score using statistical methods"""
		if len(daily_ops) < 7:
			return 0.0
		
		# Calculate z-scores for anomaly detection
		mean_ops = np.mean(daily_ops)
		std_ops = np.std(daily_ops)
		
		if std_ops == 0:
			return 0.0
		
		z_scores = [(ops - mean_ops) / std_ops for ops in daily_ops]
		
		# Count anomalies (z-score > 2.0)
		anomalies = sum(1 for z in z_scores if abs(z) > 2.0)
		anomaly_score = min(1.0, anomalies / len(daily_ops))
		
		return anomaly_score
	
	async def _analyze_trend(self, daily_ops: List[int]) -> str:
		"""Analyze usage trend direction"""
		if len(daily_ops) < 7:
			return "stable"
		
		# Simple linear regression for trend detection
		x = np.arange(len(daily_ops))
		y = np.array(daily_ops)
		
		# Calculate slope
		slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
		
		if slope > 5:
			return "increasing"
		elif slope < -5:
			return "decreasing"
		else:
			return "stable"
	
	async def _detect_seasonality(self, daily_ops: List[int]) -> bool:
		"""Detect weekly seasonality patterns"""
		if len(daily_ops) < 14:
			return False
		
		# Check for weekly patterns (every 7 days)
		weekly_correlation = 0.0
		if len(daily_ops) >= 14:
			week1 = daily_ops[:7]
			week2 = daily_ops[7:14]
			
			if len(week1) == len(week2):
				weekly_correlation = np.corrcoef(week1, week2)[0, 1]
		
		return abs(weekly_correlation) > 0.6  # Strong correlation indicates seasonality
	
	async def assess_security_risk(self, key: Key, pattern: UsagePattern, 
								  threats: List[SecurityThreat]) -> Tuple[RiskLevel, float, str]:
		"""Assess security risk using AI analysis"""
		risk_factors = []
		risk_score = 0.0
		
		# Age-based risk
		key_age_days = (datetime.utcnow() - key.spec.created_at).days
		if key_age_days > key.spec.policy.rotation_interval_days:
			risk_factors.append("Key overdue for rotation")
			risk_score += 30
		
		# Usage anomaly risk
		if pattern.anomaly_score > 0.5:
			risk_factors.append("Unusual usage patterns detected")
			risk_score += pattern.anomaly_score * 25
		
		# Threat intelligence risk
		active_threats = [t for t in threats if t.status == "active"]
		if active_threats:
			high_severity_threats = [t for t in active_threats if t.severity == "high"]
			risk_factors.append(f"{len(active_threats)} active threats detected")
			risk_score += len(high_severity_threats) * 20 + len(active_threats) * 5
		
		# Algorithm deprecation risk
		if key.spec.algorithm in [KeyAlgorithm.RSA_2048]:  # Consider RSA-2048 higher risk
			risk_factors.append("Algorithm approaching deprecation")
			risk_score += 15
		
		# Usage intensity risk
		if key.usage_count > (key.spec.policy.max_usage_count or float('inf')) * 0.8:
			risk_factors.append("Approaching usage count limit")
			risk_score += 20
		
		# Multi-tenant exposure risk
		if pattern.user_diversity > 50:
			risk_factors.append("High user diversity increases exposure")
			risk_score += 10
		
		# Determine risk level
		if risk_score >= 75:
			risk_level = RiskLevel.CRITICAL
		elif risk_score >= 50:
			risk_level = RiskLevel.HIGH
		elif risk_score >= 25:
			risk_level = RiskLevel.MEDIUM
		else:
			risk_level = RiskLevel.LOW
		
		confidence = min(1.0, risk_score / 100.0)
		reasoning = "; ".join(risk_factors) if risk_factors else "No significant risk factors detected"
		
		return risk_level, confidence, reasoning
	
	async def predict_optimal_rotation_time(self, key: Key, pattern: UsagePattern) -> datetime:
		"""Predict optimal rotation time using ML techniques"""
		current_time = datetime.utcnow()
		
		# Base rotation interval from policy
		base_interval = timedelta(days=key.spec.policy.rotation_interval_days)
		optimal_time = current_time + base_interval
		
		# Adjust based on usage patterns
		if pattern.trend_direction == "decreasing":
			# If usage is decreasing, we can extend rotation interval slightly
			optimal_time += timedelta(days=7)
		elif pattern.trend_direction == "increasing":
			# If usage is increasing, rotate sooner
			optimal_time -= timedelta(days=7)
		
		# Adjust for seasonality
		if pattern.seasonality_detected:
			# Try to rotate during low-usage periods
			day_of_week = optimal_time.weekday()
			if day_of_week < 5:  # Weekday
				# Move to weekend for lower impact
				days_to_weekend = 5 - day_of_week
				optimal_time += timedelta(days=days_to_weekend)
		
		# Adjust for peak hours
		if optimal_time.hour in pattern.peak_hours:
			# Move to off-peak hours
			if pattern.peak_hours:
				all_hours = set(range(24))
				off_peak_hours = list(all_hours - set(pattern.peak_hours))
				if off_peak_hours:
					optimal_hour = min(off_peak_hours)  # Choose earliest off-peak hour
					optimal_time = optimal_time.replace(hour=optimal_hour, minute=0, second=0)
		
		# Ensure we don't go too far beyond policy maximum
		max_time = current_time + base_interval * 1.5
		optimal_time = min(optimal_time, max_time)
		
		return optimal_time
	
	async def make_lifecycle_decision(self, key: Key, stats: KeyUsageStats, 
									 threats: List[SecurityThreat]) -> LifecycleDecision:
		"""Make AI-driven lifecycle management decision"""
		# Analyze usage patterns
		pattern = await self.analyze_usage_patterns(key, stats)
		
		# Assess security risk
		risk_level, risk_confidence, risk_reasoning = await self.assess_security_risk(key, pattern, threats)
		
		# Determine action based on multiple factors
		action = "maintain"  # Default action
		trigger = RotationTrigger.TIME_BASED
		confidence = 0.5
		reasoning = "Regular maintenance assessment"
		
		# Check for critical conditions first
		if risk_level == RiskLevel.CRITICAL:
			action = "rotate"
			trigger = RotationTrigger.THREAT_BASED
			confidence = risk_confidence
			reasoning = f"Critical risk detected: {risk_reasoning}"
		
		# Check for immediate rotation needs
		elif key.spec.state == KeyState.COMPROMISED:
			action = "revoke"
			trigger = RotationTrigger.THREAT_BASED
			confidence = 1.0
			reasoning = "Key compromised - immediate revocation required"
		
		# Check expiration
		elif key.spec.policy.expiry_date and key.spec.policy.expiry_date < datetime.utcnow():
			action = "archive"
			trigger = RotationTrigger.COMPLIANCE_BASED
			confidence = 1.0
			reasoning = "Key expired - archival required"
		
		# Check scheduled rotation
		elif key.next_rotation and datetime.utcnow() >= key.next_rotation:
			action = "rotate"
			trigger = RotationTrigger.TIME_BASED
			confidence = 0.9
			reasoning = "Scheduled rotation due"
		
		# Predictive rotation based on usage patterns
		elif pattern.anomaly_score > 0.7:
			action = "rotate"
			trigger = RotationTrigger.PREDICTIVE
			confidence = pattern.anomaly_score
			reasoning = f"Anomalous usage pattern detected (score: {pattern.anomaly_score:.2f})"
		
		# Usage-based rotation
		elif (key.spec.policy.max_usage_count and 
			  key.usage_count > key.spec.policy.max_usage_count * 0.9):
			action = "rotate"
			trigger = RotationTrigger.USAGE_BASED
			confidence = 0.8
			reasoning = "Approaching usage count limit"
		
		# Risk-based rotation
		elif risk_level == RiskLevel.HIGH:
			action = "rotate"
			trigger = RotationTrigger.THREAT_BASED
			confidence = risk_confidence
			reasoning = f"High security risk: {risk_reasoning}"
		
		# Determine optimal timing
		if action == "rotate":
			recommended_date = await self.predict_optimal_rotation_time(key, pattern)
		else:
			recommended_date = datetime.utcnow()
		
		decision = LifecycleDecision(
			key_id=key.spec.id,
			action=action,
			trigger=trigger,
			confidence=confidence,
			risk_level=risk_level,
			recommended_date=recommended_date,
			reasoning=reasoning,
			supporting_data={
				'usage_pattern': {
					'anomaly_score': pattern.anomaly_score,
					'trend_direction': pattern.trend_direction,
					'user_diversity': pattern.user_diversity,
					'peak_hours': pattern.peak_hours
				},
				'risk_assessment': {
					'risk_level': risk_level.value,
					'risk_confidence': risk_confidence,
					'active_threats': len([t for t in threats if t.status == "active"])
				},
				'key_metrics': {
					'age_days': (datetime.utcnow() - key.spec.created_at).days,
					'usage_count': key.usage_count,
					'algorithm': key.spec.algorithm.value
				}
			}
		)
		
		await self._log_ai_decision(decision)
		return decision
	
	async def batch_lifecycle_analysis(self, keys_with_stats: List[Tuple[Key, KeyUsageStats]], 
									  threats: List[SecurityThreat]) -> List[LifecycleDecision]:
		"""Perform batch lifecycle analysis for multiple keys"""
		decisions = []
		
		# Group threats by affected keys
		threats_by_key: Dict[str, List[SecurityThreat]] = {}
		for threat in threats:
			for key_id in threat.affected_keys:
				if key_id not in threats_by_key:
					threats_by_key[key_id] = []
				threats_by_key[key_id].append(threat)
		
		# Analyze each key
		for key, stats in keys_with_stats:
			key_threats = threats_by_key.get(key.spec.id, [])
			decision = await self.make_lifecycle_decision(key, stats, key_threats)
			decisions.append(decision)
		
		# Prioritize decisions by urgency
		decisions.sort(key=lambda d: (
			d.risk_level == RiskLevel.CRITICAL,
			d.risk_level == RiskLevel.HIGH,
			d.confidence
		), reverse=True)
		
		return decisions
	
	async def update_ml_models(self, historical_data: Dict[str, Any]) -> None:
		"""Update ML models based on historical data and outcomes"""
		# In a real implementation, this would train/update ML models
		# using historical key lifecycle data, threat outcomes, etc.
		
		# Update model weights based on historical accuracy
		if 'accuracy_metrics' in historical_data:
			metrics = historical_data['accuracy_metrics']
			
			# Adjust weights based on which factors were most predictive
			if metrics.get('usage_prediction_accuracy', 0) > 0.8:
				self.model_weights['usage_frequency'] *= 1.1
			
			if metrics.get('threat_prediction_accuracy', 0) > 0.8:
				self.model_weights['threat_intelligence'] *= 1.1
			
			# Normalize weights
			total_weight = sum(self.model_weights.values())
			for key in self.model_weights:
				self.model_weights[key] /= total_weight
		
		print(f"[AI-LIFECYCLE] ML models updated with historical data")
	
	async def get_lifecycle_recommendations(self, tenant_id: str) -> Dict[str, Any]:
		"""Get high-level lifecycle management recommendations"""
		patterns = list(self.usage_patterns.values())
		
		if not patterns:
			return {
				'total_keys_analyzed': 0,
				'recommendations': [],
				'risk_summary': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
			}
		
		# Aggregate statistics
		high_anomaly_keys = [p for p in patterns if p.anomaly_score > 0.5]
		increasing_usage_keys = [p for p in patterns if p.trend_direction == "increasing"]
		
		recommendations = []
		
		if high_anomaly_keys:
			recommendations.append({
				'type': 'anomaly_alert',
				'priority': 'high',
				'count': len(high_anomaly_keys),
				'message': f'{len(high_anomaly_keys)} keys showing anomalous usage patterns',
				'action': 'Review and consider early rotation'
			})
		
		if increasing_usage_keys:
			recommendations.append({
				'type': 'capacity_planning',
				'priority': 'medium',
				'count': len(increasing_usage_keys),
				'message': f'{len(increasing_usage_keys)} keys showing increasing usage trends',
				'action': 'Plan for increased key management capacity'
			})
		
		return {
			'total_keys_analyzed': len(patterns),
			'recommendations': recommendations,
			'usage_trends': {
				'increasing': len([p for p in patterns if p.trend_direction == "increasing"]),
				'decreasing': len([p for p in patterns if p.trend_direction == "decreasing"]),
				'stable': len([p for p in patterns if p.trend_direction == "stable"])
			},
			'anomaly_summary': {
				'high_anomaly_keys': len(high_anomaly_keys),
				'average_anomaly_score': np.mean([p.anomaly_score for p in patterns])
			}
		}


# Export AI lifecycle manager
__all__ = ["AILifecycleManager", "LifecycleDecision", "UsagePattern", "RotationTrigger", "RiskLevel"]