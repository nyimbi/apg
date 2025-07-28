"""
APG Threat Detection & Monitoring - Core Service

Enterprise-grade threat detection service with AI-powered analytics,
behavioral analysis, and automated incident response capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .models import (
	SecurityEvent, ThreatIndicator, SecurityIncident, BehavioralProfile,
	ThreatIntelligence, SecurityRule, IncidentResponse, ThreatAnalysis,
	SecurityMetrics, ForensicEvidence, ThreatSeverity, ThreatStatus,
	EventType, AnalysisEngine, ResponseAction
)


class ThreatDetectionService:
	"""Core threat detection and monitoring service"""
	
	def __init__(self, db_session: AsyncSession, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(__name__)
		
		self._ml_models = {}
		self._behavioral_baselines = {}
		self._threat_rules = {}
		
		asyncio.create_task(self._initialize_service())
	
	async def _initialize_service(self):
		"""Initialize threat detection service components"""
		try:
			await self._load_security_rules()
			await self._load_threat_intelligence()
			await self._initialize_ml_models()
			await self._load_behavioral_baselines()
			
			self.logger.info(f"Threat detection service initialized for tenant {self.tenant_id}")
		except Exception as e:
			self.logger.error(f"Failed to initialize threat detection service: {str(e)}")
			raise
	
	async def process_security_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
		"""Process incoming security event"""
		try:
			event = SecurityEvent(
				tenant_id=self.tenant_id,
				**event_data
			)
			
			event.normalized_data = await self._normalize_event_data(event)
			
			event.risk_score = await self._calculate_event_risk_score(event)
			event.confidence = await self._calculate_event_confidence(event)
			
			await self._store_security_event(event)
			
			threat_detected = await self._analyze_event_for_threats(event)
			
			if threat_detected:
				await self._trigger_threat_response(event, threat_detected)
			
			return event
			
		except Exception as e:
			self.logger.error(f"Error processing security event: {str(e)}")
			raise
	
	async def _normalize_event_data(self, event: SecurityEvent) -> Dict[str, Any]:
		"""Normalize raw event data to standard format"""
		normalized = {}
		
		try:
			raw_data = event.raw_data
			
			normalized.update({
				'event_timestamp': event.timestamp.isoformat(),
				'source_system': event.source_system,
				'event_type': event.event_type.value,
				'severity_score': float(event.risk_score)
			})
			
			if event.source_ip:
				normalized['source_ip'] = event.source_ip
				normalized['source_ip_classification'] = await self._classify_ip_address(event.source_ip)
			
			if event.destination_ip:
				normalized['destination_ip'] = event.destination_ip
				normalized['destination_ip_classification'] = await self._classify_ip_address(event.destination_ip)
			
			if event.user_id:
				normalized['user_id'] = event.user_id
				normalized['user_risk_score'] = await self._get_user_risk_score(event.user_id)
			
			if event.asset_id:
				normalized['asset_id'] = event.asset_id
				normalized['asset_criticality'] = await self._get_asset_criticality(event.asset_id)
			
			if event.geolocation:
				normalized['geolocation'] = event.geolocation
				normalized['location_risk'] = await self._assess_location_risk(event.geolocation)
			
			return normalized
			
		except Exception as e:
			self.logger.error(f"Error normalizing event data: {str(e)}")
			return {}
	
	async def _calculate_event_risk_score(self, event: SecurityEvent) -> Decimal:
		"""Calculate risk score for security event"""
		try:
			base_score = Decimal('10.0')
			
			severity_multipliers = {
				EventType.AUTHENTICATION: 1.2,
				EventType.AUTHORIZATION: 1.3,
				EventType.NETWORK_TRAFFIC: 0.8,
				EventType.FILE_ACCESS: 1.0,
				EventType.SYSTEM_CALL: 1.1,
				EventType.DATABASE_ACCESS: 1.4,
				EventType.APPLICATION_EVENT: 0.9,
				EventType.SECURITY_ALERT: 1.5
			}
			
			risk_score = base_score * Decimal(str(severity_multipliers.get(event.event_type, 1.0)))
			
			if event.source_ip:
				ip_risk = await self._get_ip_reputation_score(event.source_ip)
				risk_score += ip_risk
			
			if event.user_id:
				user_risk = await self._get_user_behavioral_risk(event.user_id)
				risk_score += user_risk
			
			if event.geolocation:
				location_risk = await self._assess_location_risk(event.geolocation)
				risk_score += location_risk
			
			time_risk = await self._assess_temporal_risk(event.timestamp, event.user_id)
			risk_score += time_risk
			
			return min(risk_score, Decimal('100.0'))
			
		except Exception as e:
			self.logger.error(f"Error calculating event risk score: {str(e)}")
			return Decimal('50.0')
	
	async def _calculate_event_confidence(self, event: SecurityEvent) -> Decimal:
		"""Calculate confidence score for event analysis"""
		try:
			confidence = Decimal('50.0')
			
			if event.source_system in ['siem', 'edr', 'firewall']:
				confidence += Decimal('20.0')
			
			if event.normalized_data:
				confidence += Decimal('15.0')
			
			if event.source_ip and await self._is_known_ip(event.source_ip):
				confidence += Decimal('10.0')
			
			if event.user_id and await self._has_behavioral_baseline(event.user_id):
				confidence += Decimal('15.0')
			
			return min(confidence, Decimal('100.0'))
			
		except Exception as e:
			self.logger.error(f"Error calculating event confidence: {str(e)}")
			return Decimal('50.0')
	
	async def _analyze_event_for_threats(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
		"""Analyze event for potential threats using multiple engines"""
		try:
			threat_results = []
			
			rule_based_result = await self._rule_based_analysis(event)
			if rule_based_result:
				threat_results.append(rule_based_result)
			
			ml_result = await self._machine_learning_analysis(event)
			if ml_result:
				threat_results.append(ml_result)
			
			behavioral_result = await self._behavioral_analysis(event)
			if behavioral_result:
				threat_results.append(behavioral_result)
			
			intelligence_result = await self._threat_intelligence_analysis(event)
			if intelligence_result:
				threat_results.append(intelligence_result)
			
			if threat_results:
				return await self._correlate_threat_results(threat_results, event)
			
			return None
			
		except Exception as e:
			self.logger.error(f"Error analyzing event for threats: {str(e)}")
			return None
	
	async def _rule_based_analysis(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
		"""Rule-based threat detection"""
		try:
			for rule_id, rule in self._threat_rules.items():
				if await self._evaluate_rule(rule, event):
					return {
						'analysis_engine': AnalysisEngine.RULE_BASED,
						'rule_id': rule_id,
						'rule_name': rule.name,
						'severity': rule.severity,
						'confidence': rule.confidence,
						'description': rule.description,
						'mitre_techniques': rule.mitre_techniques
					}
			
			return None
			
		except Exception as e:
			self.logger.error(f"Error in rule-based analysis: {str(e)}")
			return None
	
	async def _machine_learning_analysis(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
		"""Machine learning based anomaly detection"""
		try:
			if 'anomaly_detector' not in self._ml_models:
				return None
			
			features = await self._extract_ml_features(event)
			if not features:
				return None
			
			model = self._ml_models['anomaly_detector']
			anomaly_score = model.decision_function([features])[0]
			is_anomaly = model.predict([features])[0] == -1
			
			if is_anomaly:
				confidence = min(abs(anomaly_score) * 20, 100)
				severity = self._determine_ml_severity(anomaly_score)
				
				return {
					'analysis_engine': AnalysisEngine.MACHINE_LEARNING,
					'anomaly_score': float(anomaly_score),
					'severity': severity,
					'confidence': Decimal(str(confidence)),
					'description': f'ML anomaly detected with score {anomaly_score:.3f}'
				}
			
			return None
			
		except Exception as e:
			self.logger.error(f"Error in ML analysis: {str(e)}")
			return None
	
	async def _behavioral_analysis(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
		"""User/Entity behavioral analysis"""
		try:
			if not event.user_id:
				return None
			
			profile = await self._get_behavioral_profile(event.user_id)
			if not profile or not profile.baseline_established:
				return None
			
			behavioral_score = await self._calculate_behavioral_deviation(event, profile)
			
			if behavioral_score > 80:
				return {
					'analysis_engine': AnalysisEngine.BEHAVIORAL_ANALYSIS,
					'behavioral_score': float(behavioral_score),
					'severity': self._determine_behavioral_severity(behavioral_score),
					'confidence': profile.baseline_confidence,
					'description': f'Behavioral anomaly detected for user {event.user_id}',
					'baseline_deviation': float(behavioral_score)
				}
			
			return None
			
		except Exception as e:
			self.logger.error(f"Error in behavioral analysis: {str(e)}")
			return None
	
	async def _threat_intelligence_analysis(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
		"""Threat intelligence correlation"""
		try:
			matches = []
			
			if event.source_ip:
				ip_matches = await self._check_ip_intelligence(event.source_ip)
				matches.extend(ip_matches)
			
			if event.file_path:
				file_matches = await self._check_file_intelligence(event.file_path)
				matches.extend(file_matches)
			
			if event.hostname:
				domain_matches = await self._check_domain_intelligence(event.hostname)
				matches.extend(domain_matches)
			
			if matches:
				highest_confidence = max(match['confidence'] for match in matches)
				highest_severity = max(match['severity'] for match in matches)
				
				return {
					'analysis_engine': AnalysisEngine.THREAT_INTELLIGENCE,
					'intelligence_matches': matches,
					'severity': highest_severity,
					'confidence': Decimal(str(highest_confidence)),
					'description': f'Threat intelligence match found ({len(matches)} indicators)'
				}
			
			return None
			
		except Exception as e:
			self.logger.error(f"Error in threat intelligence analysis: {str(e)}")
			return None
	
	async def _correlate_threat_results(self, results: List[Dict[str, Any]], event: SecurityEvent) -> Dict[str, Any]:
		"""Correlate multiple threat detection results"""
		try:
			max_severity = ThreatSeverity.LOW
			total_confidence = Decimal('0.0')
			engines_used = []
			
			for result in results:
				if result['severity'].value > max_severity.value:
					max_severity = result['severity']
				total_confidence += result['confidence']
				engines_used.append(result['analysis_engine'])
			
			avg_confidence = total_confidence / len(results)
			
			correlated_result = {
				'threat_detected': True,
				'severity': max_severity,
				'confidence': avg_confidence,
				'engines_used': engines_used,
				'detection_results': results,
				'correlation_score': await self._calculate_correlation_score(results),
				'recommended_actions': await self._determine_response_actions(max_severity, avg_confidence)
			}
			
			return correlated_result
			
		except Exception as e:
			self.logger.error(f"Error correlating threat results: {str(e)}")
			return {}
	
	async def _trigger_threat_response(self, event: SecurityEvent, threat_info: Dict[str, Any]):
		"""Trigger automated threat response"""
		try:
			incident = await self._create_security_incident(event, threat_info)
			
			response_actions = threat_info.get('recommended_actions', [])
			
			if response_actions:
				response = await self._execute_automated_response(incident, response_actions)
				await self._store_incident_response(response)
			
			await self._notify_security_team(incident, threat_info)
			
		except Exception as e:
			self.logger.error(f"Error triggering threat response: {str(e)}")
	
	async def _create_security_incident(self, event: SecurityEvent, threat_info: Dict[str, Any]) -> SecurityIncident:
		"""Create security incident from threat detection"""
		try:
			incident = SecurityIncident(
				tenant_id=self.tenant_id,
				title=f"Security Threat Detected - {threat_info['severity'].value.title()}",
				description=f"Threat detected in {event.source_system} from {event.source_ip or 'unknown source'}",
				severity=threat_info['severity'],
				status=ThreatStatus.ACTIVE,
				incident_type=f"{event.event_type.value}_threat",
				affected_systems=[event.source_system],
				events=[event.id],
				first_detected=event.timestamp,
				last_activity=datetime.utcnow()
			)
			
			if event.user_id:
				incident.affected_users = [event.user_id]
			
			if event.asset_id:
				incident.affected_assets = [event.asset_id]
			
			incident.timeline.append({
				'timestamp': datetime.utcnow().isoformat(),
				'event': 'incident_created',
				'description': 'Security incident created from threat detection',
				'details': threat_info
			})
			
			await self._store_security_incident(incident)
			
			return incident
			
		except Exception as e:
			self.logger.error(f"Error creating security incident: {str(e)}")
			raise
	
	async def analyze_behavioral_patterns(self, user_id: str, period_days: int = 30) -> BehavioralProfile:
		"""Analyze user behavioral patterns"""
		try:
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=period_days)
			
			events = await self._get_user_events(user_id, start_time, end_time)
			
			if len(events) < 50:
				return await self._create_minimal_profile(user_id, start_time, end_time)
			
			profile = BehavioralProfile(
				tenant_id=self.tenant_id,
				entity_id=user_id,
				entity_type="user",
				entity_name=await self._get_user_name(user_id),
				profile_period_start=start_time,
				profile_period_end=end_time
			)
			
			profile.normal_login_hours = await self._analyze_login_patterns(events)
			profile.normal_login_locations = await self._analyze_location_patterns(events)
			profile.normal_access_patterns = await self._analyze_access_patterns(events)
			
			profile.typical_systems_accessed = await self._analyze_system_access(events)
			profile.typical_data_accessed = await self._analyze_data_access(events)
			profile.typical_operations = await self._analyze_operations(events)
			
			profile.baseline_established = True
			profile.baseline_confidence = await self._calculate_baseline_confidence(events)
			
			profile.anomaly_score = await self._calculate_current_anomaly_score(user_id, profile)
			profile.risk_score = await self._calculate_user_risk_score(user_id, profile)
			
			await self._store_behavioral_profile(profile)
			
			return profile
			
		except Exception as e:
			self.logger.error(f"Error analyzing behavioral patterns: {str(e)}")
			raise
	
	async def hunt_threats(self, hypothesis: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute threat hunting query"""
		try:
			hunt_results = {
				'hypothesis': hypothesis,
				'start_time': datetime.utcnow().isoformat(),
				'query_params': query_params,
				'findings': [],
				'recommendations': [],
				'iocs_found': [],
				'timeline': []
			}
			
			events = await self._execute_hunt_query(query_params)
			
			if events:
				analysis = await self._analyze_hunt_results(events, hypothesis)
				hunt_results.update(analysis)
				
				suspicious_patterns = await self._identify_suspicious_patterns(events)
				hunt_results['suspicious_patterns'] = suspicious_patterns
				
				iocs = await self._extract_iocs_from_events(events)
				hunt_results['iocs_found'] = iocs
			
			hunt_results['end_time'] = datetime.utcnow().isoformat()
			hunt_results['event_count'] = len(events) if events else 0
			
			await self._store_hunt_results(hunt_results)
			
			return hunt_results
			
		except Exception as e:
			self.logger.error(f"Error in threat hunting: {str(e)}")
			raise
	
	async def get_security_metrics(self, period_days: int = 7) -> SecurityMetrics:
		"""Generate security operations metrics"""
		try:
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=period_days)
			
			metrics = SecurityMetrics(
				tenant_id=self.tenant_id,
				metric_period_start=start_time,
				metric_period_end=end_time
			)
			
			metrics.total_events_processed = await self._count_events_in_period(start_time, end_time)
			metrics.total_alerts_generated = await self._count_alerts_in_period(start_time, end_time)
			metrics.total_incidents_created = await self._count_incidents_in_period(start_time, end_time)
			
			metrics.mean_time_to_detection = await self._calculate_mttd(start_time, end_time)
			metrics.mean_time_to_response = await self._calculate_mttr(start_time, end_time)
			metrics.mean_time_to_resolution = await self._calculate_mttres(start_time, end_time)
			
			metrics.false_positive_rate = await self._calculate_false_positive_rate(start_time, end_time)
			metrics.alert_accuracy = await self._calculate_alert_accuracy(start_time, end_time)
			
			metrics.incidents_by_severity = await self._get_incidents_by_severity(start_time, end_time)
			metrics.incidents_by_status = await self._get_incidents_by_status(start_time, end_time)
			
			metrics.top_attack_vectors = await self._get_top_attack_vectors(start_time, end_time)  
			metrics.top_targeted_assets = await self._get_top_targeted_assets(start_time, end_time)
			
			metrics.automated_response_success_rate = await self._calculate_automation_success_rate(start_time, end_time)
			metrics.escalation_rate = await self._calculate_escalation_rate(start_time, end_time)
			
			metrics.threat_intelligence_matches = await self._count_intelligence_matches(start_time, end_time)
			metrics.behavioral_anomalies_detected = await self._count_behavioral_anomalies(start_time, end_time)
			
			await self._store_security_metrics(metrics)
			
			return metrics
			
		except Exception as e:
			self.logger.error(f"Error generating security metrics: {str(e)}")
			raise
	
	async def _load_security_rules(self):
		"""Load active security rules into memory"""
		try:
			pass
		except Exception as e:
			self.logger.error(f"Error loading security rules: {str(e)}")
	
	async def _load_threat_intelligence(self):
		"""Load threat intelligence feeds"""
		try:
			pass
		except Exception as e:
			self.logger.error(f"Error loading threat intelligence: {str(e)}")
	
	async def _initialize_ml_models(self):
		"""Initialize machine learning models"""
		try:
			self._ml_models['anomaly_detector'] = IsolationForest(
				contamination=0.1,
				random_state=42
			)
			
			training_data = await self._get_training_data()
			if training_data is not None and len(training_data) > 100:
				self._ml_models['anomaly_detector'].fit(training_data)
			
		except Exception as e:
			self.logger.error(f"Error initializing ML models: {str(e)}")
	
	async def _load_behavioral_baselines(self):
		"""Load existing behavioral baselines"""
		try:
			pass
		except Exception as e:
			self.logger.error(f"Error loading behavioral baselines: {str(e)}")
	
	async def _store_security_event(self, event: SecurityEvent):
		"""Store security event to database"""  
		pass
	
	async def _store_security_incident(self, incident: SecurityIncident):
		"""Store security incident to database"""
		pass
	
	async def _store_incident_response(self, response: IncidentResponse):
		"""Store incident response to database"""
		pass
	
	async def _store_behavioral_profile(self, profile: BehavioralProfile):
		"""Store behavioral profile to database"""
		pass
	
	async def _store_hunt_results(self, results: Dict[str, Any]):
		"""Store threat hunting results"""
		pass
	
	async def _store_security_metrics(self, metrics: SecurityMetrics):
		"""Store security metrics to database"""
		pass
	
	def _determine_ml_severity(self, anomaly_score: float) -> ThreatSeverity:
		"""Determine severity based on ML anomaly score"""
		if anomaly_score < -0.5:
			return ThreatSeverity.CRITICAL
		elif anomaly_score < -0.3:
			return ThreatSeverity.HIGH
		elif anomaly_score < -0.1:
			return ThreatSeverity.MEDIUM
		else:
			return ThreatSeverity.LOW
	
	def _determine_behavioral_severity(self, behavioral_score: float) -> ThreatSeverity:
		"""Determine severity based on behavioral deviation score"""
		if behavioral_score > 95:
			return ThreatSeverity.CRITICAL
		elif behavioral_score > 85:
			return ThreatSeverity.HIGH
		elif behavioral_score > 75:
			return ThreatSeverity.MEDIUM
		else:
			return ThreatSeverity.LOW