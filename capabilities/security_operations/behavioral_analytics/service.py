"""
APG Behavioral Analytics - Core Service

Enterprise behavioral analytics service with advanced statistical modeling,
real-time anomaly detection, and predictive behavioral analysis.

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
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
	BehavioralProfile, BehavioralBaseline, BehavioralAnomaly, PeerGroup,
	RiskAssessment, BehavioralMetrics, BehavioralAlert, EntityType,
	BehaviorType, AnomalyType, RiskLevel, BaselineStatus
)


class BehavioralAnalyticsService:
	"""Core behavioral analytics and anomaly detection service"""
	
	def __init__(self, db_session: AsyncSession, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(__name__)
		
		self._profiles_cache = {}
		self._baselines_cache = {}
		self._peer_groups = {}
		self._ml_models = {}
		
		asyncio.create_task(self._initialize_service())
	
	async def _initialize_service(self):
		"""Initialize behavioral analytics service"""
		try:
			await self._load_behavioral_profiles()
			await self._load_baselines()
			await self._initialize_ml_models()
			await self._load_peer_groups()
			
			self.logger.info(f"Behavioral analytics service initialized for tenant {self.tenant_id}")
		except Exception as e:
			self.logger.error(f"Failed to initialize behavioral analytics service: {str(e)}")
			raise
	
	async def create_behavioral_profile(self, entity_id: str, entity_type: EntityType, 
										profile_data: Dict[str, Any]) -> BehavioralProfile:
		"""Create comprehensive behavioral profile"""
		try:
			profile = BehavioralProfile(
				tenant_id=self.tenant_id,
				entity_id=entity_id,
				entity_type=entity_type,
				**profile_data
			)
			
			# Analyze historical data to establish baseline
			historical_data = await self._collect_historical_data(entity_id, entity_type)
			
			if len(historical_data) >= 50:  # Minimum data points for baseline
				profile = await self._establish_behavioral_baseline(profile, historical_data)
				profile.baseline_status = BaselineStatus.ESTABLISHED
				profile.baseline_confidence = await self._calculate_baseline_confidence(historical_data)
			else:
				profile.baseline_status = BaselineStatus.ESTABLISHING
				profile.baseline_confidence = Decimal('0.0')
			
			# Assign to peer group
			peer_group = await self._assign_peer_group(profile)
			if peer_group:
				profile.peer_group_id = peer_group.id
				profile.peer_comparison_score = await self._calculate_peer_comparison(profile, peer_group)
			
			await self._store_behavioral_profile(profile)
			
			return profile
			
		except Exception as e:
			self.logger.error(f"Error creating behavioral profile: {str(e)}")
			raise
	
	async def _establish_behavioral_baseline(self, profile: BehavioralProfile, 
											historical_data: List[Dict[str, Any]]) -> BehavioralProfile:
		"""Establish statistical baseline from historical data"""
		try:
			behavior_types = [
				BehaviorType.ACCESS_PATTERN,
				BehaviorType.LOGIN_BEHAVIOR,
				BehaviorType.DATA_ACCESS,
				BehaviorType.COMMUNICATION,
				BehaviorType.SYSTEM_USAGE,
				BehaviorType.NETWORK_ACTIVITY
			]
			
			for behavior_type in behavior_types:
				baseline_data = await self._analyze_behavior_type(historical_data, behavior_type)
				
				if baseline_data:
					# Store baseline for this behavior type
					baseline = await self._create_baseline(profile.id, behavior_type, baseline_data)
					
					# Update profile with baseline summary
					await self._update_profile_baseline_summary(profile, behavior_type, baseline)
			
			return profile
			
		except Exception as e:
			self.logger.error(f"Error establishing behavioral baseline: {str(e)}")
			return profile
	
	async def _analyze_behavior_type(self, historical_data: List[Dict[str, Any]], 
									behavior_type: BehaviorType) -> Optional[Dict[str, Any]]:
		"""Analyze specific behavior type from historical data"""
		try:
			filtered_data = []
			
			for record in historical_data:
				if record.get('behavior_type') == behavior_type.value:
					filtered_data.append(record)
			
			if len(filtered_data) < 10:  # Minimum data points
				return None
			
			# Extract numeric values for statistical analysis
			values = []
			for record in filtered_data:
				if 'value' in record and isinstance(record['value'], (int, float)):
					values.append(float(record['value']))
			
			if not values:
				return None
			
			# Calculate statistical measures
			values_array = np.array(values)
			
			baseline_data = {
				'mean_value': float(np.mean(values_array)),
				'median_value': float(np.median(values_array)),
				'standard_deviation': float(np.std(values_array)),
				'percentile_25': float(np.percentile(values_array, 25)),
				'percentile_75': float(np.percentile(values_array, 75)),
				'percentile_95': float(np.percentile(values_array, 95)),
				'percentile_99': float(np.percentile(values_array, 99)),
				'min_value': float(np.min(values_array)),
				'max_value': float(np.max(values_array)),
				'sample_size': len(values)
			}
			
			# Calculate temporal patterns
			temporal_patterns = await self._analyze_temporal_patterns(filtered_data)
			baseline_data.update(temporal_patterns)
			
			return baseline_data
			
		except Exception as e:
			self.logger.error(f"Error analyzing behavior type {behavior_type}: {str(e)}")
			return None
	
	async def _analyze_temporal_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Analyze temporal patterns in behavioral data"""
		try:
			hourly_patterns = {}
			daily_patterns = {}
			weekly_patterns = {}
			
			for record in data:
				timestamp = record.get('timestamp')
				if timestamp:
					if isinstance(timestamp, str):
						timestamp = datetime.fromisoformat(timestamp)
					
					hour = timestamp.hour
					day = timestamp.day
					weekday = timestamp.weekday()
					
					# Count occurrences by time periods
					hourly_patterns[str(hour)] = hourly_patterns.get(str(hour), 0) + 1
					daily_patterns[str(day)] = daily_patterns.get(str(day), 0) + 1
					weekly_patterns[str(weekday)] = weekly_patterns.get(str(weekday), 0) + 1
			
			return {
				'hourly_patterns': hourly_patterns,
				'daily_patterns': daily_patterns,
				'weekly_patterns': weekly_patterns
			}
			
		except Exception as e:
			self.logger.error(f"Error analyzing temporal patterns: {str(e)}")
			return {}
	
	async def detect_behavioral_anomalies(self, entity_id: str, current_data: Dict[str, Any]) -> List[BehavioralAnomaly]:
		"""Detect behavioral anomalies using statistical and ML methods"""
		try:
			anomalies = []
			
			# Get behavioral profile and baselines
			profile = await self._get_behavioral_profile(entity_id)
			if not profile or profile.baseline_status != BaselineStatus.ESTABLISHED:
				return anomalies
			
			baselines = await self._get_baselines_for_profile(profile.id)
			
			# Analyze each behavior type
			for baseline in baselines:
				anomaly = await self._detect_anomaly_for_baseline(profile, baseline, current_data)
				if anomaly:
					anomalies.append(anomaly)
			
			# Peer group comparison
			if profile.peer_group_id:
				peer_anomalies = await self._detect_peer_group_anomalies(profile, current_data)
				anomalies.extend(peer_anomalies)
			
			# ML-based anomaly detection
			ml_anomalies = await self._detect_ml_anomalies(profile, current_data)
			anomalies.extend(ml_anomalies)
			
			# Store detected anomalies
			for anomaly in anomalies:
				await self._store_behavioral_anomaly(anomaly)
			
			return anomalies
			
		except Exception as e:
			self.logger.error(f"Error detecting behavioral anomalies: {str(e)}")
			return []
	
	async def _detect_anomaly_for_baseline(self, profile: BehavioralProfile, 
										  baseline: BehavioralBaseline, 
										  current_data: Dict[str, Any]) -> Optional[BehavioralAnomaly]:
		"""Detect anomaly against statistical baseline"""
		try:
			metric_name = baseline.metric_name
			observed_value = current_data.get(metric_name)
			
			if observed_value is None:
				return None
			
			observed_value = float(observed_value)
			
			# Calculate z-score
			mean = float(baseline.mean_value)
			std_dev = float(baseline.standard_deviation)
			
			if std_dev == 0:
				return None  # No variance in baseline
			
			z_score = (observed_value - mean) / std_dev
			
			# Determine if this is an anomaly (using 3-sigma rule)
			is_anomaly = abs(z_score) > 3.0
			
			if not is_anomaly:
				return None
			
			# Calculate anomaly score
			anomaly_score = min(abs(z_score) * 20, 100)  # Scale to 0-100
			
			# Determine anomaly type
			anomaly_type = AnomalyType.STATISTICAL
			if observed_value > float(baseline.percentile_99):
				anomaly_type = AnomalyType.VOLUMETRIC
			elif self._is_temporal_anomaly(current_data, baseline):
				anomaly_type = AnomalyType.TEMPORAL
			
			# Create anomaly record
			anomaly = BehavioralAnomaly(
				tenant_id=self.tenant_id,
				profile_id=profile.id,
				baseline_id=baseline.id,
				entity_id=profile.entity_id,
				entity_type=profile.entity_type,
				anomaly_type=anomaly_type,
				behavior_type=baseline.behavior_type,
				metric_name=metric_name,
				observed_value=Decimal(str(observed_value)),
				expected_value=baseline.mean_value,
				deviation_score=Decimal(str(abs(z_score))),
				anomaly_score=Decimal(str(anomaly_score)),
				z_score=Decimal(str(z_score)),
				event_timestamp=current_data.get('timestamp', datetime.utcnow()),
				context_data=current_data
			)
			
			# Assess risk level
			anomaly.risk_level = await self._assess_anomaly_risk(anomaly, profile)
			anomaly.risk_score = await self._calculate_anomaly_risk_score(anomaly)
			
			return anomaly
			
		except Exception as e:
			self.logger.error(f"Error detecting anomaly for baseline: {str(e)}")
			return None
	
	async def _detect_peer_group_anomalies(self, profile: BehavioralProfile, 
										   current_data: Dict[str, Any]) -> List[BehavioralAnomaly]:
		"""Detect anomalies based on peer group comparison"""
		try:
			anomalies = []
			
			if not profile.peer_group_id:
				return anomalies
			
			peer_group = await self._get_peer_group(profile.peer_group_id)
			if not peer_group:
				return anomalies
			
			# Compare against peer group baselines
			for metric_name, baseline_data in peer_group.group_baselines.items():
				observed_value = current_data.get(metric_name)
				if observed_value is None:
					continue
				
				observed_value = float(observed_value)
				peer_mean = float(baseline_data.get('mean', 0))
				peer_std = float(baseline_data.get('std', 0))
				
				if peer_std == 0:
					continue
				
				# Calculate deviation from peer group
				peer_z_score = (observed_value - peer_mean) / peer_std
				
				# Check if significantly different from peers
				if abs(peer_z_score) > float(peer_group.outlier_threshold):
					anomaly = BehavioralAnomaly(
						tenant_id=self.tenant_id,
						profile_id=profile.id,
						entity_id=profile.entity_id,
						entity_type=profile.entity_type,
						anomaly_type=AnomalyType.PEER_DEVIATION,
						behavior_type=BehaviorType.ACCESS_PATTERN,  # Default
						metric_name=metric_name,
						observed_value=Decimal(str(observed_value)),
						expected_value=Decimal(str(peer_mean)),
						deviation_score=Decimal(str(abs(peer_z_score))),
						anomaly_score=Decimal(str(min(abs(peer_z_score) * 25, 100))),
						z_score=Decimal(str(peer_z_score)),
						event_timestamp=current_data.get('timestamp', datetime.utcnow()),
						context_data=current_data
					)
					
					anomaly.risk_level = await self._assess_anomaly_risk(anomaly, profile)
					anomaly.risk_score = await self._calculate_anomaly_risk_score(anomaly)
					
					anomalies.append(anomaly)
			
			return anomalies
			
		except Exception as e:
			self.logger.error(f"Error detecting peer group anomalies: {str(e)}")
			return []
	
	async def _detect_ml_anomalies(self, profile: BehavioralProfile, 
								   current_data: Dict[str, Any]) -> List[BehavioralAnomaly]:
		"""Detect anomalies using machine learning models"""
		try:
			anomalies = []
			
			if 'isolation_forest' not in self._ml_models:
				return anomalies
			
			# Extract features for ML model
			features = await self._extract_ml_features(current_data, profile)
			if not features:
				return anomalies
			
			# Get anomaly score from Isolation Forest
			model = self._ml_models['isolation_forest']
			anomaly_score = model.decision_function([features])[0]
			is_anomaly = model.predict([features])[0] == -1
			
			if is_anomaly:
				ml_anomaly = BehavioralAnomaly(
					tenant_id=self.tenant_id,
					profile_id=profile.id,
					entity_id=profile.entity_id,
					entity_type=profile.entity_type,
					anomaly_type=AnomalyType.PATTERN,
					behavior_type=BehaviorType.SYSTEM_USAGE,
					metric_name="ml_composite_score",
					observed_value=Decimal(str(anomaly_score)),
					expected_value=Decimal('0.0'),
					deviation_score=Decimal(str(abs(anomaly_score))),
					anomaly_score=Decimal(str(min(abs(anomaly_score) * 100, 100))),
					event_timestamp=current_data.get('timestamp', datetime.utcnow()),
					context_data=current_data
				)
				
				ml_anomaly.risk_level = await self._assess_anomaly_risk(ml_anomaly, profile)
				ml_anomaly.risk_score = await self._calculate_anomaly_risk_score(ml_anomaly)
				
				anomalies.append(ml_anomaly)
			
			return anomalies
			
		except Exception as e:
			self.logger.error(f"Error detecting ML anomalies: {str(e)}")
			return []
	
	async def assess_behavioral_risk(self, entity_id: str, assessment_period: int = 30) -> RiskAssessment:
		"""Assess comprehensive behavioral risk for entity"""
		try:
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=assessment_period)
			
			profile = await self._get_behavioral_profile(entity_id)
			if not profile:
				raise ValueError(f"No behavioral profile found for entity {entity_id}")
			
			# Get anomalies in assessment period
			anomalies = await self._get_anomalies_in_period(entity_id, start_time, end_time)
			
			assessment = RiskAssessment(
				tenant_id=self.tenant_id,
				entity_id=entity_id,
				entity_type=profile.entity_type,
				assessment_period_start=start_time,
				assessment_period_end=end_time
			)
			
			# Calculate risk components
			assessment.behavioral_risk = await self._calculate_behavioral_risk(anomalies)
			assessment.temporal_risk = await self._calculate_temporal_risk(anomalies)
			assessment.access_risk = await self._calculate_access_risk(anomalies)
			assessment.peer_deviation_risk = await self._calculate_peer_deviation_risk(anomalies)
			
			# Calculate overall risk score
			risk_weights = {
				'behavioral': 0.3,
				'temporal': 0.2,
				'access': 0.3,
				'peer_deviation': 0.2
			}
			
			overall_risk = (
				float(assessment.behavioral_risk) * risk_weights['behavioral'] +
				float(assessment.temporal_risk) * risk_weights['temporal'] +
				float(assessment.access_risk) * risk_weights['access'] +
				float(assessment.peer_deviation_risk) * risk_weights['peer_deviation']
			)
			
			assessment.overall_risk_score = Decimal(str(overall_risk))
			assessment.risk_level = await self._determine_risk_level(assessment.overall_risk_score)
			
			# Analyze anomaly summary
			assessment.total_anomalies = len(anomalies)
			assessment.critical_anomalies = len([a for a in anomalies if a.risk_level == RiskLevel.CRITICAL])
			assessment.high_risk_anomalies = len([a for a in anomalies if a.risk_level == RiskLevel.HIGH])
			
			# Generate recommendations
			assessment.recommended_actions = await self._generate_risk_recommendations(assessment, anomalies)
			assessment.monitoring_recommendations = await self._generate_monitoring_recommendations(assessment)
			
			# Analyze trend
			assessment.risk_trend = await self._analyze_risk_trend(entity_id)
			assessment.trend_confidence = await self._calculate_trend_confidence(entity_id)
			
			await self._store_risk_assessment(assessment)
			
			return assessment
			
		except Exception as e:
			self.logger.error(f"Error assessing behavioral risk: {str(e)}")
			raise
	
	async def create_peer_group(self, group_data: Dict[str, Any]) -> PeerGroup:
		"""Create peer group for comparative analysis"""
		try:
			peer_group = PeerGroup(
				tenant_id=self.tenant_id,
				**group_data
			)
			
			# Find matching entities based on criteria
			matching_entities = await self._find_matching_entities(peer_group.grouping_criteria)
			peer_group.member_entities = matching_entities
			peer_group.member_count = len(matching_entities)
			
			# Calculate group baselines
			peer_group.group_baselines = await self._calculate_group_baselines(matching_entities)
			
			await self._store_peer_group(peer_group)
			
			return peer_group
			
		except Exception as e:
			self.logger.error(f"Error creating peer group: {str(e)}")
			raise
	
	async def generate_behavioral_metrics(self, period_days: int = 30) -> BehavioralMetrics:
		"""Generate behavioral analytics metrics"""
		try:
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=period_days)
			
			metrics = BehavioralMetrics(
				tenant_id=self.tenant_id,
				metric_period_start=start_time,
				metric_period_end=end_time
			)
			
			# Profile metrics
			metrics.total_profiles = await self._count_total_profiles()
			metrics.active_profiles = await self._count_active_profiles()
			metrics.profiles_with_baselines = await self._count_profiles_with_baselines()
			
			if metrics.total_profiles > 0:
				metrics.baseline_establishment_rate = Decimal(str(
					(metrics.profiles_with_baselines / metrics.total_profiles) * 100
				))
			
			# Anomaly metrics
			anomalies = await self._get_anomalies_in_period_all(start_time, end_time)
			metrics.total_anomalies_detected = len(anomalies)
			metrics.critical_anomalies = len([a for a in anomalies if a.risk_level == RiskLevel.CRITICAL])
			metrics.high_risk_anomalies = len([a for a in anomalies if a.risk_level == RiskLevel.HIGH])
			
			# Calculate false positive rate
			false_positives = len([a for a in anomalies if a.is_false_positive])
			if metrics.total_anomalies_detected > 0:
				metrics.false_positive_rate = Decimal(str(
					(false_positives / metrics.total_anomalies_detected) * 100
				))
			
			# Detection performance
			investigated_anomalies = [a for a in anomalies if a.is_investigated]
			if investigated_anomalies:
				detection_times = []
				for anomaly in investigated_anomalies:
					if anomaly.detection_timestamp and anomaly.event_timestamp:
						detection_time = anomaly.detection_timestamp - anomaly.event_timestamp
						detection_times.append(detection_time)
				
				if detection_times:
					metrics.mean_time_to_detection = sum(detection_times, timedelta()) / len(detection_times)
			
			# Risk assessment metrics
			high_risk_entities = await self._count_high_risk_entities()
			metrics.high_risk_entities = high_risk_entities
			
			# Peer group metrics
			metrics.total_peer_groups = await self._count_peer_groups()
			if metrics.total_peer_groups > 0:
				total_members = await self._count_total_peer_group_members()
				metrics.average_group_size = Decimal(str(total_members / metrics.total_peer_groups))
			
			await self._store_behavioral_metrics(metrics)
			
			return metrics
			
		except Exception as e:
			self.logger.error(f"Error generating behavioral metrics: {str(e)}")
			raise
	
	# Helper methods for implementation
	async def _load_behavioral_profiles(self):
		"""Load behavioral profiles into cache"""
		pass
	
	async def _load_baselines(self):
		"""Load behavioral baselines into cache"""
		pass
	
	async def _initialize_ml_models(self):
		"""Initialize ML models for anomaly detection"""
		try:
			# Initialize Isolation Forest for anomaly detection
			self._ml_models['isolation_forest'] = IsolationForest(
				contamination=0.1,
				random_state=42
			)
			
			# Train on sample data if available
			training_data = await self._get_ml_training_data()
			if training_data is not None and len(training_data) > 100:
				self._ml_models['isolation_forest'].fit(training_data)
			
		except Exception as e:
			self.logger.error(f"Error initializing ML models: {str(e)}")
	
	async def _load_peer_groups(self):
		"""Load peer groups into cache"""
		pass
	
	# Placeholder implementations for database operations
	async def _store_behavioral_profile(self, profile: BehavioralProfile):
		"""Store behavioral profile to database"""
		pass
	
	async def _store_behavioral_anomaly(self, anomaly: BehavioralAnomaly):
		"""Store behavioral anomaly to database"""
		pass
	
	async def _store_peer_group(self, peer_group: PeerGroup):
		"""Store peer group to database"""
		pass
	
	async def _store_risk_assessment(self, assessment: RiskAssessment):
		"""Store risk assessment to database"""
		pass
	
	async def _store_behavioral_metrics(self, metrics: BehavioralMetrics):
		"""Store behavioral metrics to database"""
		pass