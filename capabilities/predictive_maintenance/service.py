#!/usr/bin/env python3
"""
AI-Powered Predictive Maintenance System
=======================================

Advanced machine learning system for predictive maintenance of digital twins.
Includes anomaly detection, failure prediction, maintenance optimization,
and automated alert generation.
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from pathlib import Path

# ML and analytics imports
try:
	from sklearn.ensemble import IsolationForest, RandomForestClassifier
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	from sklearn.metrics import classification_report, confusion_matrix
	from sklearn.model_selection import train_test_split
	from sklearn.cluster import DBSCAN
	import joblib
except ImportError:
	print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
	from scipy import stats
	from scipy.signal import find_peaks, savgol_filter
except ImportError:
	print("Warning: scipy not available. Install with: pip install scipy")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictive_maintenance")

class MaintenanceType(Enum):
	"""Types of maintenance operations"""
	PREVENTIVE = "preventive"
	CORRECTIVE = "corrective"
	PREDICTIVE = "predictive"
	CONDITION_BASED = "condition_based"
	EMERGENCY = "emergency"

class FailureMode(Enum):
	"""Common failure modes for industrial equipment"""
	WEAR = "wear"
	FATIGUE = "fatigue"
	CORROSION = "corrosion"
	OVERHEATING = "overheating"
	VIBRATION = "vibration"
	ELECTRICAL = "electrical"
	MECHANICAL = "mechanical"
	HYDRAULIC = "hydraulic"
	PNEUMATIC = "pneumatic"
	SOFTWARE = "software"

class MaintenancePriority(Enum):
	"""Maintenance priority levels"""
	CRITICAL = "critical"		# Immediate action required
	HIGH = "high"				# Action required within 24 hours
	MEDIUM = "medium"			# Action required within 1 week
	LOW = "low"					# Action required within 1 month
	ROUTINE = "routine"			# Scheduled maintenance

@dataclass
class MaintenanceAlert:
	"""Maintenance alert with prediction details"""
	alert_id: str
	twin_id: str
	asset_name: str
	failure_mode: FailureMode
	predicted_failure_time: datetime
	confidence_score: float
	priority: MaintenancePriority
	estimated_cost: float
	recommended_actions: List[str]
	affected_components: List[str]
	risk_score: float
	created_at: datetime
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'alert_id': self.alert_id,
			'twin_id': self.twin_id,
			'asset_name': self.asset_name,
			'failure_mode': self.failure_mode.value,
			'predicted_failure_time': self.predicted_failure_time.isoformat(),
			'confidence_score': self.confidence_score,
			'priority': self.priority.value,
			'estimated_cost': self.estimated_cost,
			'recommended_actions': self.recommended_actions,
			'affected_components': self.affected_components,
			'risk_score': self.risk_score,
			'created_at': self.created_at.isoformat()
		}

@dataclass
class MaintenanceRecommendation:
	"""Maintenance recommendation with optimization"""
	recommendation_id: str
	twin_id: str
	maintenance_type: MaintenanceType
	priority: MaintenancePriority
	recommended_date: datetime
	estimated_duration: float  # hours
	estimated_cost: float
	required_parts: List[Dict[str, Any]]
	required_skills: List[str]
	safety_requirements: List[str]
	business_impact: Dict[str, Any]
	cost_benefit_analysis: Dict[str, Any]
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'recommendation_id': self.recommendation_id,
			'twin_id': self.twin_id,
			'maintenance_type': self.maintenance_type.value,
			'priority': self.priority.value,
			'recommended_date': self.recommended_date.isoformat(),
			'estimated_duration': self.estimated_duration,
			'estimated_cost': self.estimated_cost,
			'required_parts': self.required_parts,
			'required_skills': self.required_skills,
			'safety_requirements': self.safety_requirements,
			'business_impact': self.business_impact,
			'cost_benefit_analysis': self.cost_benefit_analysis
		}

class AnomalyDetector:
	"""Advanced anomaly detection for equipment monitoring"""
	
	def __init__(self, contamination: float = 0.1):
		self.isolation_forest = IsolationForest(
			contamination=contamination,
			random_state=42,
			n_estimators=200
		)
		self.scaler = StandardScaler()
		self.is_trained = False
		self.feature_names = []
		
	def train(self, data: pd.DataFrame, features: List[str]):
		"""Train anomaly detection model"""
		self.feature_names = features
		X = data[features].values
		
		# Handle missing values
		X = np.nan_to_num(X)
		
		# Scale features
		X_scaled = self.scaler.fit_transform(X)
		
		# Train isolation forest
		self.isolation_forest.fit(X_scaled)
		self.is_trained = True
		
		logger.info(f"Anomaly detector trained on {len(data)} samples with {len(features)} features")
		
	def detect_anomalies(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
		"""Detect anomalies in new data"""
		if not self.is_trained:
			raise ValueError("Model must be trained before detecting anomalies")
			
		X = data[self.feature_names].values
		X = np.nan_to_num(X)
		X_scaled = self.scaler.transform(X)
		
		# Predict anomalies (-1 = anomaly, 1 = normal)
		anomaly_labels = self.isolation_forest.predict(X_scaled)
		anomaly_scores = self.isolation_forest.decision_function(X_scaled)
		
		return anomaly_labels, anomaly_scores

class FailurePredictionModel:
	"""Machine learning model for failure prediction"""
	
	def __init__(self):
		self.model = RandomForestClassifier(
			n_estimators=200,
			max_depth=15,
			random_state=42,
			class_weight='balanced'
		)
		self.scaler = StandardScaler()
		self.is_trained = False
		self.feature_names = []
		self.failure_modes = []
		
	def prepare_features(self, telemetry_data: pd.DataFrame) -> pd.DataFrame:
		"""Engineer features for failure prediction"""
		features = telemetry_data.copy()
		
		# Time-based features
		if 'timestamp' in features.columns:
			features['timestamp'] = pd.to_datetime(features['timestamp'])
			features['hour'] = features['timestamp'].dt.hour
			features['day_of_week'] = features['timestamp'].dt.dayofweek
			features['month'] = features['timestamp'].dt.month
		
		# Statistical features (rolling windows)
		numeric_cols = features.select_dtypes(include=[np.number]).columns
		for col in numeric_cols:
			if col not in ['hour', 'day_of_week', 'month']:
				# Rolling statistics
				features[f'{col}_rolling_mean_24h'] = features[col].rolling(24).mean()
				features[f'{col}_rolling_std_24h'] = features[col].rolling(24).std()
				features[f'{col}_rolling_max_24h'] = features[col].rolling(24).max()
				features[f'{col}_rolling_min_24h'] = features[col].rolling(24).min()
				
				# Rate of change
				features[f'{col}_diff'] = features[col].diff()
				features[f'{col}_pct_change'] = features[col].pct_change()
				
				# Trend indicators
				features[f'{col}_trend'] = features[col].rolling(12).apply(
					lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 12 else 0
				)
		
		# Fill NaN values
		features = features.fillna(method='forward').fillna(0)
		
		return features
		
	def train(self, telemetry_data: pd.DataFrame, failure_labels: pd.Series, 
			  feature_columns: List[str] = None):
		"""Train failure prediction model"""
		
		# Prepare features
		features_df = self.prepare_features(telemetry_data)
		
		if feature_columns is None:
			# Use all numeric columns except timestamp-based
			feature_columns = [col for col in features_df.columns 
							  if features_df[col].dtype in [np.float64, np.int64]
							  and col not in ['hour', 'day_of_week', 'month']]
		
		self.feature_names = feature_columns
		X = features_df[feature_columns].values
		y = failure_labels.values
		
		# Handle missing values
		X = np.nan_to_num(X)
		
		# Scale features
		X_scaled = self.scaler.fit_transform(X)
		
		# Split data
		X_train, X_test, y_train, y_test = train_test_split(
			X_scaled, y, test_size=0.2, random_state=42, stratify=y
		)
		
		# Train model
		self.model.fit(X_train, y_train)
		self.is_trained = True
		
		# Evaluate model
		y_pred = self.model.predict(X_test)
		report = classification_report(y_test, y_pred, output_dict=True)
		
		logger.info(f"Failure prediction model trained with accuracy: {report['accuracy']:.3f}")
		
		return report
		
	def predict_failure(self, telemetry_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
		"""Predict failure probability and time to failure"""
		if not self.is_trained:
			raise ValueError("Model must be trained before making predictions")
			
		features_df = self.prepare_features(telemetry_data)
		X = features_df[self.feature_names].values
		X = np.nan_to_num(X)
		X_scaled = self.scaler.transform(X)
		
		# Predict failure probability
		failure_probability = self.model.predict_proba(X_scaled)
		failure_predictions = self.model.predict(X_scaled)
		
		return failure_predictions, failure_probability

class MaintenanceOptimizer:
	"""Optimization engine for maintenance scheduling"""
	
	def __init__(self):
		self.cost_models = {}
		self.constraint_models = {}
		
	def calculate_total_cost(self, maintenance_plan: Dict[str, Any]) -> float:
		"""Calculate total cost of maintenance plan"""
		
		# Direct maintenance costs
		direct_cost = maintenance_plan.get('parts_cost', 0) + \
					 maintenance_plan.get('labor_cost', 0) + \
					 maintenance_plan.get('overhead_cost', 0)
		
		# Downtime costs
		downtime_hours = maintenance_plan.get('downtime_hours', 0)
		production_rate = maintenance_plan.get('production_rate', 1000)  # units/hour
		revenue_per_unit = maintenance_plan.get('revenue_per_unit', 10)
		downtime_cost = downtime_hours * production_rate * revenue_per_unit
		
		# Opportunity costs
		opportunity_cost = maintenance_plan.get('opportunity_cost', 0)
		
		return direct_cost + downtime_cost + opportunity_cost
		
	def optimize_schedule(self, maintenance_requests: List[Dict[str, Any]], 
						 constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Optimize maintenance schedule considering constraints"""
		
		# Simple greedy optimization (can be enhanced with more sophisticated algorithms)
		optimized_schedule = []
		available_resources = constraints.get('available_resources', {})
		time_windows = constraints.get('time_windows', [])
		
		# Sort requests by priority and cost-benefit ratio
		sorted_requests = sorted(
			maintenance_requests,
			key=lambda x: (
				x.get('priority_score', 0),
				-x.get('cost_benefit_ratio', 0)
			),
			reverse=True
		)
		
		for request in sorted_requests:
			# Check resource availability
			required_resources = request.get('required_resources', {})
			can_schedule = True
			
			for resource, amount in required_resources.items():
				if available_resources.get(resource, 0) < amount:
					can_schedule = False
					break
			
			if can_schedule:
				optimized_schedule.append(request)
				# Update available resources
				for resource, amount in required_resources.items():
					available_resources[resource] -= amount
		
		return optimized_schedule

class PredictiveMaintenanceEngine:
	"""Main predictive maintenance engine"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.anomaly_detector = AnomalyDetector()
		self.failure_predictor = FailurePredictionModel()
		self.maintenance_optimizer = MaintenanceOptimizer()
		
		# Thresholds and parameters
		self.anomaly_threshold = self.config.get('anomaly_threshold', -0.5)
		self.failure_probability_threshold = self.config.get('failure_probability_threshold', 0.7)
		self.prediction_horizon = self.config.get('prediction_horizon', 30)  # days
		
		# Active alerts and recommendations
		self.active_alerts: Dict[str, MaintenanceAlert] = {}
		self.active_recommendations: Dict[str, MaintenanceRecommendation] = {}
		
		logger.info("Predictive Maintenance Engine initialized")
		
	async def analyze_twin_health(self, twin_id: str, telemetry_data: pd.DataFrame) -> Dict[str, Any]:
		"""Comprehensive health analysis of a digital twin"""
		
		try:
			# 1. Anomaly Detection
			if self.anomaly_detector.is_trained:
				anomaly_labels, anomaly_scores = self.anomaly_detector.detect_anomalies(telemetry_data)
				anomaly_count = np.sum(anomaly_labels == -1)
				avg_anomaly_score = np.mean(anomaly_scores)
			else:
				anomaly_count = 0
				avg_anomaly_score = 0
				
			# 2. Failure Prediction
			if self.failure_predictor.is_trained:
				failure_predictions, failure_probabilities = self.failure_predictor.predict_failure(telemetry_data)
				max_failure_prob = np.max(failure_probabilities[:, 1]) if failure_probabilities.shape[1] > 1 else 0
			else:
				max_failure_prob = 0
				
			# 3. Health Score Calculation
			health_score = self._calculate_health_score(
				anomaly_score=avg_anomaly_score,
				failure_probability=max_failure_prob,
				telemetry_data=telemetry_data
			)
			
			# 4. Generate Alerts if necessary
			alerts = []
			if max_failure_prob > self.failure_probability_threshold:
				alert = await self._generate_maintenance_alert(
					twin_id=twin_id,
					failure_probability=max_failure_prob,
					telemetry_data=telemetry_data
				)
				alerts.append(alert)
				
			# 5. Generate Recommendations
			recommendations = await self._generate_maintenance_recommendations(
				twin_id=twin_id,
				health_score=health_score,
				failure_probability=max_failure_prob,
				telemetry_data=telemetry_data
			)
			
			return {
				'twin_id': twin_id,
				'health_score': health_score,
				'anomaly_count': int(anomaly_count),
				'avg_anomaly_score': float(avg_anomaly_score),
				'max_failure_probability': float(max_failure_prob),
				'alerts': [alert.to_dict() if isinstance(alert, MaintenanceAlert) else alert for alert in alerts],
				'recommendations': [rec.to_dict() if isinstance(rec, MaintenanceRecommendation) else rec for rec in recommendations],
				'analysis_timestamp': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Error analyzing twin health for {twin_id}: {e}")
			return {
				'twin_id': twin_id,
				'error': str(e),
				'analysis_timestamp': datetime.utcnow().isoformat()
			}
			
	def _calculate_health_score(self, anomaly_score: float, failure_probability: float, 
							   telemetry_data: pd.DataFrame) -> float:
		"""Calculate overall health score (0-100)"""
		
		# Base score
		base_score = 100.0
		
		# Penalty for anomalies
		anomaly_penalty = max(0, (-anomaly_score - 0.5) * 30)
		
		# Penalty for failure probability
		failure_penalty = failure_probability * 50
		
		# Penalty for trend deterioration
		trend_penalty = 0
		if not telemetry_data.empty:
			numeric_cols = telemetry_data.select_dtypes(include=[np.number]).columns
			for col in numeric_cols[:3]:  # Check first 3 numeric columns
				if len(telemetry_data[col]) > 5:
					slope, _, _, _, _ = stats.linregress(range(len(telemetry_data[col])), telemetry_data[col])
					if slope < 0:  # Deteriorating trend
						trend_penalty += abs(slope) * 5
		
		# Calculate final score
		health_score = max(0, base_score - anomaly_penalty - failure_penalty - trend_penalty)
		return min(100, health_score)
		
	async def _generate_maintenance_alert(self, twin_id: str, failure_probability: float,
										 telemetry_data: pd.DataFrame) -> MaintenanceAlert:
		"""Generate maintenance alert for high failure probability"""
		
		import uuid
		
		# Determine failure mode based on sensor patterns
		failure_mode = self._identify_failure_mode(telemetry_data)
		
		# Calculate time to failure
		predicted_failure_time = datetime.utcnow() + timedelta(
			days=max(1, int((1 - failure_probability) * self.prediction_horizon))
		)
		
		# Determine priority
		if failure_probability > 0.9:
			priority = MaintenancePriority.CRITICAL
		elif failure_probability > 0.8:
			priority = MaintenancePriority.HIGH
		elif failure_probability > 0.7:
			priority = MaintenancePriority.MEDIUM
		else:
			priority = MaintenancePriority.LOW
			
		# Generate recommendations
		recommended_actions = self._get_recommended_actions(failure_mode, priority)
		
		alert = MaintenanceAlert(
			alert_id=f"alert_{uuid.uuid4().hex[:8]}",
			twin_id=twin_id,
			asset_name=f"Asset_{twin_id}",
			failure_mode=failure_mode,
			predicted_failure_time=predicted_failure_time,
			confidence_score=failure_probability,
			priority=priority,
			estimated_cost=self._estimate_maintenance_cost(failure_mode, priority),
			recommended_actions=recommended_actions,
			affected_components=self._get_affected_components(failure_mode),
			risk_score=failure_probability * 100,
			created_at=datetime.utcnow()
		)
		
		self.active_alerts[alert.alert_id] = alert
		return alert
		
	async def _generate_maintenance_recommendations(self, twin_id: str, health_score: float,
												  failure_probability: float, 
												  telemetry_data: pd.DataFrame) -> List[MaintenanceRecommendation]:
		"""Generate maintenance recommendations"""
		
		recommendations = []
		
		# Preventive maintenance based on health score
		if health_score < 80:
			rec = self._create_preventive_maintenance_recommendation(twin_id, health_score)
			recommendations.append(rec)
			
		# Predictive maintenance based on failure probability
		if failure_probability > 0.5:
			rec = self._create_predictive_maintenance_recommendation(twin_id, failure_probability)
			recommendations.append(rec)
			
		return recommendations
		
	def _identify_failure_mode(self, telemetry_data: pd.DataFrame) -> FailureMode:
		"""Identify most likely failure mode based on telemetry patterns"""
		
		# Simple heuristic-based identification (can be enhanced with ML)
		if 'temperature' in telemetry_data.columns:
			temp_values = telemetry_data['temperature'].values
			if np.mean(temp_values) > 80:  # High temperature
				return FailureMode.OVERHEATING
				
		if 'vibration' in telemetry_data.columns:
			vib_values = telemetry_data['vibration'].values
			if np.std(vib_values) > 2:  # High vibration variability
				return FailureMode.VIBRATION
				
		# Default to mechanical wear
		return FailureMode.WEAR
		
	def _get_recommended_actions(self, failure_mode: FailureMode, priority: MaintenancePriority) -> List[str]:
		"""Get recommended maintenance actions"""
		
		action_map = {
			FailureMode.OVERHEATING: [
				"Check cooling system",
				"Inspect heat exchangers",
				"Verify temperature sensors",
				"Clean air filters"
			],
			FailureMode.VIBRATION: [
				"Check bearing condition",
				"Inspect alignment",
				"Verify mounting bolts",
				"Balance rotating components"
			],
			FailureMode.WEAR: [
				"Inspect wearing parts",
				"Check lubrication system",
				"Replace filters",
				"Verify operating parameters"
			]
		}
		
		return action_map.get(failure_mode, ["Perform general inspection"])
		
	def _get_affected_components(self, failure_mode: FailureMode) -> List[str]:
		"""Get components affected by failure mode"""
		
		component_map = {
			FailureMode.OVERHEATING: ["Cooling system", "Heat exchanger", "Temperature sensors"],
			FailureMode.VIBRATION: ["Bearings", "Coupling", "Motor mount", "Drive shaft"],
			FailureMode.WEAR: ["Seals", "Gaskets", "Filters", "Moving parts"]
		}
		
		return component_map.get(failure_mode, ["General components"])
		
	def _estimate_maintenance_cost(self, failure_mode: FailureMode, priority: MaintenancePriority) -> float:
		"""Estimate maintenance cost"""
		
		base_costs = {
			FailureMode.OVERHEATING: 5000,
			FailureMode.VIBRATION: 3000,
			FailureMode.WEAR: 2000
		}
		
		priority_multipliers = {
			MaintenancePriority.CRITICAL: 3.0,
			MaintenancePriority.HIGH: 2.0,
			MaintenancePriority.MEDIUM: 1.5,
			MaintenancePriority.LOW: 1.0
		}
		
		base_cost = base_costs.get(failure_mode, 2000)
		multiplier = priority_multipliers.get(priority, 1.0)
		
		return base_cost * multiplier
		
	def _create_preventive_maintenance_recommendation(self, twin_id: str, health_score: float) -> MaintenanceRecommendation:
		"""Create preventive maintenance recommendation"""
		
		import uuid
		
		return MaintenanceRecommendation(
			recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
			twin_id=twin_id,
			maintenance_type=MaintenanceType.PREVENTIVE,
			priority=MaintenancePriority.MEDIUM,
			recommended_date=datetime.utcnow() + timedelta(days=7),
			estimated_duration=4.0,
			estimated_cost=1500.0,
			required_parts=[
				{"part_name": "Filter", "quantity": 2, "cost": 150},
				{"part_name": "Lubricant", "quantity": 1, "cost": 75}
			],
			required_skills=["Mechanical", "Basic electrical"],
			safety_requirements=["PPE", "Lockout/tagout"],
			business_impact={
				"downtime_hours": 2,
				"production_loss": 5000,
				"quality_impact": "minimal"
			},
			cost_benefit_analysis={
				"maintenance_cost": 1500,
				"avoided_failure_cost": 15000,
				"net_benefit": 13500,
				"roi": 9.0
			}
		)
		
	def _create_predictive_maintenance_recommendation(self, twin_id: str, failure_probability: float) -> MaintenanceRecommendation:
		"""Create predictive maintenance recommendation"""
		
		import uuid
		
		priority = MaintenancePriority.HIGH if failure_probability > 0.8 else MaintenancePriority.MEDIUM
		
		return MaintenanceRecommendation(
			recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
			twin_id=twin_id,
			maintenance_type=MaintenanceType.PREDICTIVE,
			priority=priority,
			recommended_date=datetime.utcnow() + timedelta(days=3),
			estimated_duration=6.0,
			estimated_cost=3500.0,
			required_parts=[
				{"part_name": "Bearing", "quantity": 1, "cost": 500},
				{"part_name": "Seal kit", "quantity": 1, "cost": 200}
			],
			required_skills=["Advanced mechanical", "Vibration analysis"],
			safety_requirements=["PPE", "Confined space", "Lockout/tagout"],
			business_impact={
				"downtime_hours": 4,
				"production_loss": 10000,
				"quality_impact": "moderate"
			},
			cost_benefit_analysis={
				"maintenance_cost": 3500,
				"avoided_failure_cost": 25000,
				"net_benefit": 21500,
				"roi": 6.14
			}
		)

# Test and example usage
async def test_predictive_maintenance():
	"""Test the predictive maintenance system"""
	
	# Create sample telemetry data
	np.random.seed(42)
	dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
	
	telemetry_data = pd.DataFrame({
		'timestamp': dates,
		'temperature': 75 + np.random.normal(0, 5, 1000) + np.linspace(0, 10, 1000),  # Increasing trend
		'vibration': 1.5 + np.random.normal(0, 0.3, 1000),
		'pressure': 10 + np.random.normal(0, 1, 1000),
		'flow_rate': 100 + np.random.normal(0, 5, 1000)
	})
	
	# Create failure labels (simulate some failures near the end)
	failure_labels = pd.Series([0] * 950 + [1] * 50)
	
	# Initialize predictive maintenance engine
	pm_engine = PredictiveMaintenanceEngine()
	
	# Train models
	print("Training anomaly detection model...")
	pm_engine.anomaly_detector.train(
		telemetry_data, 
		['temperature', 'vibration', 'pressure', 'flow_rate']
	)
	
	print("Training failure prediction model...")
	pm_engine.failure_predictor.train(
		telemetry_data,
		failure_labels
	)
	
	# Analyze recent data
	recent_data = telemetry_data.tail(100)
	analysis_result = await pm_engine.analyze_twin_health('twin_001', recent_data)
	
	print(f"\nHealth Analysis Results:")
	print(f"Twin ID: {analysis_result['twin_id']}")
	print(f"Health Score: {analysis_result['health_score']:.1f}")
	print(f"Anomaly Count: {analysis_result['anomaly_count']}")
	print(f"Max Failure Probability: {analysis_result['max_failure_probability']:.3f}")
	print(f"Number of Alerts: {len(analysis_result['alerts'])}")
	print(f"Number of Recommendations: {len(analysis_result['recommendations'])}")
	
	if analysis_result['alerts']:
		print(f"\nFirst Alert Details:")
		alert = analysis_result['alerts'][0]
		print(f"  Priority: {alert['priority']}")
		print(f"  Failure Mode: {alert['failure_mode']}")
		print(f"  Confidence: {alert['confidence_score']:.3f}")
		print(f"  Estimated Cost: ${alert['estimated_cost']:,.2f}")

if __name__ == "__main__":
	asyncio.run(test_predictive_maintenance())