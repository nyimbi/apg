#!/usr/bin/env python3
"""
APG Key Management - Security Intelligence & Anomaly Detection
Behavioral analytics and advanced threat detection for key security

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from .models import Key, KeyOperation, SecurityThreat, AuditEvent, KeyUsageStats


class AnomalyType(str, Enum):
	"""Types of security anomalies"""
	UNUSUAL_VOLUME = "unusual_volume"
	UNUSUAL_TIMING = "unusual_timing" 
	SUSPICIOUS_USER = "suspicious_user"
	SUSPICIOUS_IP = "suspicious_ip"
	UNUSUAL_GEOGRAPHIC = "unusual_geographic"
	PRIVILEGE_ESCALATION = "privilege_escalation"
	CONCURRENT_ACCESS = "concurrent_access"
	RAPID_SUCCESSIVE = "rapid_successive"
	UNUSUAL_APPLICATION = "unusual_application"
	PATTERN_DEVIATION = "pattern_deviation"


class ThreatSeverity(str, Enum):
	"""Threat severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


@dataclass
class UserBehaviorProfile:
	"""User behavior profile for anomaly detection"""
	user_id: str
	total_operations: int = 0
	unique_keys_accessed: Set[str] = field(default_factory=set)
	typical_hours: List[int] = field(default_factory=list)
	typical_days: List[int] = field(default_factory=list)
	common_ips: Set[str] = field(default_factory=set)
	operation_frequencies: Dict[str, int] = field(default_factory=dict)
	average_session_duration: float = 0.0
	last_seen: datetime | None = None
	risk_score: float = 0.0
	
	def update_profile(self, operation: KeyOperation) -> None:
		"""Update profile with new operation data"""
		self.total_operations += 1
		self.unique_keys_accessed.add(operation.key_id)
		
		if operation.requested_at:
			hour = operation.requested_at.hour
			day = operation.requested_at.weekday()
			
			if hour not in self.typical_hours:
				self.typical_hours.append(hour)
			if day not in self.typical_days:
				self.typical_days.append(day)
		
		if operation.request_ip:
			self.common_ips.add(operation.request_ip)
		
		op_type = operation.operation_type
		self.operation_frequencies[op_type] = self.operation_frequencies.get(op_type, 0) + 1
		
		self.last_seen = datetime.utcnow()


@dataclass
class SecurityEvent:
	"""Security event for threat correlation"""
	event_id: str
	timestamp: datetime
	event_type: str
	severity: ThreatSeverity
	source_ip: str | None
	user_id: str | None
	key_id: str | None
	details: Dict[str, Any]
	confidence: float


@dataclass
class AnomalyAlert:
	"""Anomaly detection alert"""
	alert_id: str
	anomaly_type: AnomalyType
	severity: ThreatSeverity
	confidence: float
	affected_keys: List[str]
	affected_users: List[str]
	detection_time: datetime
	description: str
	recommended_actions: List[str]
	supporting_evidence: Dict[str, Any]


class SecurityIntelligenceEngine:
	"""
	Advanced security intelligence and anomaly detection engine
	Provides behavioral analytics, threat correlation, and predictive security
	"""
	
	def __init__(self, config: Dict[str, Any] | None = None):
		self.config = config or {}
		
		# Behavioral profiles
		self.user_profiles: Dict[str, UserBehaviorProfile] = {}
		self.key_access_patterns: Dict[str, Dict[str, Any]] = {}
		
		# Security events and alerts
		self.security_events: deque = deque(maxlen=10000)
		self.active_alerts: Dict[str, AnomalyAlert] = {}
		self.threat_intelligence_feeds: Dict[str, Any] = {}
		
		# ML model parameters (would be trained on real data)
		self.anomaly_thresholds = {
			'volume_multiplier': 5.0,
			'time_deviation_hours': 8,
			'new_ip_threshold': 0.1,
			'rapid_operations_threshold': 10,
			'geographic_distance_km': 1000
		}
		
		# Time-series data for trend analysis
		self.operation_timeseries: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
	
	async def _log_security_event(self, event: SecurityEvent) -> None:
		"""Log security event for analysis and correlation"""
		self.security_events.append(event)
		print(f"[SECURITY-INTEL] {event.event_type} event: {event.severity.value} severity "
			 f"(confidence: {event.confidence:.2f})")
	
	async def analyze_operation(self, operation: KeyOperation, key: Key) -> List[AnomalyAlert]:
		"""Analyze operation for security anomalies"""
		alerts = []
		
		# Update user behavior profile
		if operation.user_id:
			await self._update_user_profile(operation)
		
		# Perform various anomaly detections
		volume_alerts = await self._detect_volume_anomalies(operation, key)
		timing_alerts = await self._detect_timing_anomalies(operation)
		access_alerts = await self._detect_access_anomalies(operation)
		behavioral_alerts = await self._detect_behavioral_anomalies(operation)
		
		alerts.extend(volume_alerts)
		alerts.extend(timing_alerts)
		alerts.extend(access_alerts)
		alerts.extend(behavioral_alerts)
		
		# Store alerts
		for alert in alerts:
			self.active_alerts[alert.alert_id] = alert
		
		return alerts
	
	async def _update_user_profile(self, operation: KeyOperation) -> None:
		"""Update user behavioral profile"""
		user_id = operation.user_id
		if not user_id:
			return
		
		if user_id not in self.user_profiles:
			self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
		
		profile = self.user_profiles[user_id]
		profile.update_profile(operation)
		
		# Calculate risk score based on recent behavior
		profile.risk_score = await self._calculate_user_risk_score(profile)
	
	async def _calculate_user_risk_score(self, profile: UserBehaviorProfile) -> float:
		"""Calculate user risk score based on behavioral patterns"""
		risk_score = 0.0
		
		# High operation count in short time
		if profile.total_operations > 1000:
			risk_score += 0.1
		
		# Access to many different keys
		if len(profile.unique_keys_accessed) > 50:
			risk_score += 0.15
		
		# Unusual time patterns
		if len(profile.typical_hours) > 16:  # Active more than 16 hours per day
			risk_score += 0.1
		
		# Many different IP addresses
		if len(profile.common_ips) > 10:
			risk_score += 0.2
		
		# Recent activity
		if profile.last_seen and (datetime.utcnow() - profile.last_seen).days > 30:
			risk_score += 0.1  # Inactive users returning
		
		return min(1.0, risk_score)
	
	async def _detect_volume_anomalies(self, operation: KeyOperation, key: Key) -> List[AnomalyAlert]:
		"""Detect unusual volume of operations"""
		alerts = []
		
		# Get recent operations for this key
		key_id = operation.key_id
		current_time = datetime.utcnow()
		
		# Count operations in last hour
		recent_ops = [ts for ts, count in self.operation_timeseries[key_id] 
					 if current_time - ts < timedelta(hours=1)]
		
		# Update time series
		self.operation_timeseries[key_id].append((current_time, 1))
		
		# Keep only last 24 hours of data
		cutoff_time = current_time - timedelta(hours=24)
		self.operation_timeseries[key_id] = [
			(ts, count) for ts, count in self.operation_timeseries[key_id] 
			if ts > cutoff_time
		]
		
		# Check for volume anomaly
		hourly_ops = len(recent_ops)
		if hourly_ops > self.anomaly_thresholds['volume_multiplier'] * 5:  # Threshold: 25 ops/hour
			alert = AnomalyAlert(
				alert_id=f"vol_{key_id}_{int(current_time.timestamp())}",
				anomaly_type=AnomalyType.UNUSUAL_VOLUME,
				severity=ThreatSeverity.HIGH if hourly_ops > 50 else ThreatSeverity.MEDIUM,
				confidence=min(1.0, hourly_ops / 100.0),
				affected_keys=[key_id],
				affected_users=[operation.user_id] if operation.user_id else [],
				detection_time=current_time,
				description=f"Unusual volume: {hourly_ops} operations in last hour (normal: ~5)",
				recommended_actions=[
					"Review user activity logs",
					"Verify application behavior",
					"Consider temporary key suspension"
				],
				supporting_evidence={
					'hourly_operations': hourly_ops,
					'threshold': 25,
					'key_algorithm': key.spec.algorithm.value
				}
			)
			alerts.append(alert)
		
		return alerts
	
	async def _detect_timing_anomalies(self, operation: KeyOperation) -> List[AnomalyAlert]:
		"""Detect unusual timing patterns"""
		alerts = []
		
		if not operation.user_id or not operation.requested_at:
			return alerts
		
		profile = self.user_profiles.get(operation.user_id)
		if not profile:
			return alerts
		
		current_hour = operation.requested_at.hour
		current_day = operation.requested_at.weekday()
		
		# Check if operation is outside typical hours
		if (profile.typical_hours and 
			current_hour not in profile.typical_hours and 
			len(profile.typical_hours) >= 5):  # Need sufficient historical data
			
			# Calculate how far outside normal hours
			hour_distances = [min(abs(current_hour - h), 24 - abs(current_hour - h)) 
							 for h in profile.typical_hours]
			min_distance = min(hour_distances)
			
			if min_distance >= self.anomaly_thresholds['time_deviation_hours']:
				alert = AnomalyAlert(
					alert_id=f"time_{operation.user_id}_{int(operation.requested_at.timestamp())}",
					anomaly_type=AnomalyType.UNUSUAL_TIMING,
					severity=ThreatSeverity.MEDIUM,
					confidence=min(1.0, min_distance / 12.0),
					affected_keys=[operation.key_id],
					affected_users=[operation.user_id],
					detection_time=datetime.utcnow(),
					description=f"Access at unusual hour: {current_hour}:00 (typical: {profile.typical_hours})",
					recommended_actions=[
						"Verify user identity",
						"Check for account compromise",
						"Review authentication logs"
					],
					supporting_evidence={
						'access_hour': current_hour,
						'typical_hours': profile.typical_hours,
						'hour_deviation': min_distance
					}
				)
				alerts.append(alert)
		
		return alerts
	
	async def _detect_access_anomalies(self, operation: KeyOperation) -> List[AnomalyAlert]:
		"""Detect suspicious access patterns"""
		alerts = []
		
		if not operation.user_id:
			return alerts
		
		profile = self.user_profiles.get(operation.user_id)
		if not profile:
			return alerts
		
		# Check for new IP address
		if (operation.request_ip and 
			operation.request_ip not in profile.common_ips and
			len(profile.common_ips) >= 3):  # User has established IP pattern
			
			alert = AnomalyAlert(
				alert_id=f"ip_{operation.user_id}_{int(datetime.utcnow().timestamp())}",
				anomaly_type=AnomalyType.SUSPICIOUS_IP,
				severity=ThreatSeverity.MEDIUM,
				confidence=0.7,
				affected_keys=[operation.key_id],
				affected_users=[operation.user_id],
				detection_time=datetime.utcnow(),
				description=f"Access from new IP address: {operation.request_ip}",
				recommended_actions=[
					"Verify user location",
					"Check for VPN/proxy usage",
					"Require additional authentication"
				],
				supporting_evidence={
					'new_ip': operation.request_ip,
					'known_ips': list(profile.common_ips),
					'total_operations': profile.total_operations
				}
			)
			alerts.append(alert)
		
		# Check for rapid successive operations
		current_time = datetime.utcnow()
		if operation.requested_at:
			recent_ops = [event for event in self.security_events 
						 if (event.user_id == operation.user_id and 
							current_time - event.timestamp < timedelta(minutes=5))]
			
			if len(recent_ops) > self.anomaly_thresholds['rapid_operations_threshold']:
				alert = AnomalyAlert(
					alert_id=f"rapid_{operation.user_id}_{int(current_time.timestamp())}",
					anomaly_type=AnomalyType.RAPID_SUCCESSIVE,
					severity=ThreatSeverity.HIGH,
					confidence=min(1.0, len(recent_ops) / 20.0),
					affected_keys=[operation.key_id],
					affected_users=[operation.user_id],
					detection_time=current_time,
					description=f"Rapid successive operations: {len(recent_ops)} in 5 minutes",
					recommended_actions=[
						"Check for automated attack",
						"Rate limit user operations", 
						"Investigate application behavior"
					],
					supporting_evidence={
						'operations_count': len(recent_ops),
						'time_window_minutes': 5,
						'threshold': self.anomaly_thresholds['rapid_operations_threshold']
					}
				)
				alerts.append(alert)
		
		return alerts
	
	async def _detect_behavioral_anomalies(self, operation: KeyOperation) -> List[AnomalyAlert]:
		"""Detect behavioral pattern deviations"""
		alerts = []
		
		if not operation.user_id:
			return alerts
		
		profile = self.user_profiles.get(operation.user_id)
		if not profile or profile.total_operations < 50:  # Need baseline
			return alerts
		
		# Check for unusual operation type
		op_type = operation.operation_type
		total_ops = sum(profile.operation_frequencies.values())
		
		if total_ops > 0:
			current_frequency = profile.operation_frequencies.get(op_type, 0) / total_ops
			
			# If this operation type is very rare for this user
			if current_frequency < 0.01 and profile.operation_frequencies.get(op_type, 0) < 3:
				alert = AnomalyAlert(
					alert_id=f"behavior_{operation.user_id}_{int(datetime.utcnow().timestamp())}",
					anomaly_type=AnomalyType.PATTERN_DEVIATION,
					severity=ThreatSeverity.LOW,
					confidence=0.6,
					affected_keys=[operation.key_id],
					affected_users=[operation.user_id],
					detection_time=datetime.utcnow(),
					description=f"Unusual operation type '{op_type}' for user (frequency: {current_frequency:.1%})",
					recommended_actions=[
						"Verify user intent",
						"Check application requirements",
						"Review operation context"
					],
					supporting_evidence={
						'operation_type': op_type,
						'user_frequency': current_frequency,
						'operation_history': dict(profile.operation_frequencies)
					}
				)
				alerts.append(alert)
		
		return alerts
	
	async def correlate_threats(self, time_window_minutes: int = 60) -> List[SecurityThreat]:
		"""Correlate security events to identify complex threats"""
		threats = []
		current_time = datetime.utcnow()
		cutoff_time = current_time - timedelta(minutes=time_window_minutes)
		
		# Get recent events
		recent_events = [event for event in self.security_events 
						if event.timestamp > cutoff_time]
		
		# Group events by various correlation criteria
		threats.extend(await self._correlate_by_user(recent_events))
		threats.extend(await self._correlate_by_ip(recent_events))
		threats.extend(await self._correlate_by_pattern(recent_events))
		
		return threats
	
	async def _correlate_by_user(self, events: List[SecurityEvent]) -> List[SecurityThreat]:
		"""Correlate events by user to detect account compromise"""
		threats = []
		user_events: Dict[str, List[SecurityEvent]] = defaultdict(list)
		
		for event in events:
			if event.user_id:
				user_events[event.user_id].append(event)
		
		for user_id, user_event_list in user_events.items():
			if len(user_event_list) >= 3:  # Multiple suspicious events
				high_severity_events = [e for e in user_event_list if e.severity == ThreatSeverity.HIGH]
				
				if high_severity_events or len(user_event_list) >= 5:
					threat = SecurityThreat(
						tenant_id="correlation_engine",  # Would use actual tenant
						threat_type="account_compromise",
						severity="high",
						confidence=min(1.0, len(user_event_list) / 10.0),
						affected_keys=list(set([e.key_id for e in user_event_list if e.key_id])),
						source_ip=user_event_list[0].source_ip,
						user_id=user_id,
						detection_method="event_correlation",
						indicators={
							'correlated_events': len(user_event_list),
							'high_severity_events': len(high_severity_events),
							'time_span_minutes': (max(e.timestamp for e in user_event_list) - 
												  min(e.timestamp for e in user_event_list)).total_seconds() / 60
						}
					)
					threats.append(threat)
		
		return threats
	
	async def _correlate_by_ip(self, events: List[SecurityEvent]) -> List[SecurityThreat]:
		"""Correlate events by IP to detect distributed attacks"""
		threats = []
		ip_events: Dict[str, List[SecurityEvent]] = defaultdict(list)
		
		for event in events:
			if event.source_ip:
				ip_events[event.source_ip].append(event)
		
		for source_ip, ip_event_list in ip_events.items():
			if len(ip_event_list) >= 5:  # Many events from single IP
				unique_users = set([e.user_id for e in ip_event_list if e.user_id])
				
				if len(unique_users) >= 3:  # Multiple users from same IP
					threat = SecurityThreat(
						tenant_id="correlation_engine",
						threat_type="distributed_attack",
						severity="high",
						confidence=min(1.0, len(ip_event_list) / 20.0),
						affected_keys=list(set([e.key_id for e in ip_event_list if e.key_id])),
						source_ip=source_ip,
						detection_method="event_correlation",
						indicators={
							'events_from_ip': len(ip_event_list),
							'unique_users_affected': len(unique_users),
							'affected_users': list(unique_users)
						}
					)
					threats.append(threat)
		
		return threats
	
	async def _correlate_by_pattern(self, events: List[SecurityEvent]) -> List[SecurityThreat]:
		"""Correlate events by attack patterns"""
		threats = []
		
		# Look for escalation patterns
		escalation_events = [e for e in events if 'privilege' in e.event_type.lower()]
		if len(escalation_events) >= 2:
			threat = SecurityThreat(
				tenant_id="correlation_engine",
				threat_type="privilege_escalation",
				severity="critical",
				confidence=0.8,
				affected_keys=list(set([e.key_id for e in escalation_events if e.key_id])),
				detection_method="pattern_correlation",
				indicators={
					'escalation_events': len(escalation_events),
					'event_types': [e.event_type for e in escalation_events]
				}
			)
			threats.append(threat)
		
		return threats
	
	async def get_security_dashboard(self) -> Dict[str, Any]:
		"""Generate security intelligence dashboard data"""
		current_time = datetime.utcnow()
		
		# Active alerts summary
		alert_summary = defaultdict(int)
		for alert in self.active_alerts.values():
			alert_summary[alert.severity.value] += 1
		
		# User risk distribution
		high_risk_users = [p for p in self.user_profiles.values() if p.risk_score > 0.7]
		medium_risk_users = [p for p in self.user_profiles.values() if 0.3 < p.risk_score <= 0.7]
		
		# Recent threat trends
		recent_events = [e for e in self.security_events 
						if current_time - e.timestamp < timedelta(hours=24)]
		
		threat_trends = defaultdict(int)
		for event in recent_events:
			threat_trends[event.event_type] += 1
		
		return {
			'dashboard_generated_at': current_time,
			'alert_summary': dict(alert_summary),
			'total_active_alerts': len(self.active_alerts),
			'user_risk_distribution': {
				'high_risk': len(high_risk_users),
				'medium_risk': len(medium_risk_users),
				'low_risk': len(self.user_profiles) - len(high_risk_users) - len(medium_risk_users)
			},
			'threat_trends_24h': dict(threat_trends),
			'total_events_24h': len(recent_events),
			'top_risk_users': [
				{'user_id': p.user_id, 'risk_score': p.risk_score, 'operations': p.total_operations}
				for p in sorted(self.user_profiles.values(), key=lambda x: x.risk_score, reverse=True)[:5]
			],
			'anomaly_detection_stats': {
				'profiles_monitored': len(self.user_profiles),
				'keys_monitored': len(self.key_access_patterns),
				'detection_rules_active': len(self.anomaly_thresholds)
			}
		}
	
	async def update_threat_intelligence(self, intel_feeds: Dict[str, Any]) -> None:
		"""Update threat intelligence from external feeds"""
		self.threat_intelligence_feeds.update(intel_feeds)
		
		# Update detection thresholds based on current threat landscape
		if 'global_threat_level' in intel_feeds:
			threat_level = intel_feeds['global_threat_level']
			
			if threat_level == 'high':
				# Tighten thresholds during high threat periods
				self.anomaly_thresholds['volume_multiplier'] *= 0.7
				self.anomaly_thresholds['rapid_operations_threshold'] = int(
					self.anomaly_thresholds['rapid_operations_threshold'] * 0.8
				)
			elif threat_level == 'low':
				# Relax thresholds during low threat periods
				self.anomaly_thresholds['volume_multiplier'] *= 1.2
		
		print(f"[SECURITY-INTEL] Threat intelligence updated from {len(intel_feeds)} feeds")


# Export security intelligence components
__all__ = [
	"SecurityIntelligenceEngine", "AnomalyAlert", "SecurityEvent", 
	"UserBehaviorProfile", "AnomalyType", "ThreatSeverity"
]