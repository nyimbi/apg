"""
APG Facial Recognition - Predictive Identity Analytics Engine

Revolutionary predictive intelligence for identity verification with pattern analysis,
risk forecasting, and proactive security measures using advanced machine learning.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import asyncio
import numpy as np
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid_extensions import uuid7str
from enum import Enum

try:
	from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingClassifier
	from sklearn.preprocessing import StandardScaler, LabelEncoder
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score, precision_score, recall_score
	from sklearn.cluster import DBSCAN, KMeans
	import pandas as pd
except ImportError as e:
	print(f"Optional ML dependencies not available: {e}")

class PredictionType(Enum):
	IDENTITY_RISK = "identity_risk"
	FRAUD_PROBABILITY = "fraud_probability"
	TEMPLATE_AGING = "template_aging"
	BEHAVIORAL_ANOMALY = "behavioral_anomaly"
	ACCESS_PATTERN = "access_pattern"
	SECURITY_THREAT = "security_threat"

class RiskLevel(Enum):
	VERY_LOW = "very_low"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"

class PredictiveAnalyticsEngine:
	"""Advanced predictive analytics for identity verification"""
	
	def __init__(self, tenant_id: str):
		"""Initialize predictive analytics engine"""
		assert tenant_id, "Tenant ID cannot be empty"
		
		self.tenant_id = tenant_id
		self.prediction_enabled = True
		self.learning_enabled = True
		self.real_time_analysis = True
		
		# Prediction models
		self.models = {}
		self.model_metadata = {}
		self.training_data = {}
		self.feature_scalers = {}
		
		# Historical data for analysis
		self.identity_patterns = {}
		self.risk_history = {}
		self.fraud_patterns = {}
		self.behavioral_baselines = {}
		
		# Prediction thresholds
		self.risk_thresholds = {
			RiskLevel.VERY_LOW: 0.1,
			RiskLevel.LOW: 0.3,
			RiskLevel.MEDIUM: 0.5,
			RiskLevel.HIGH: 0.7,
			RiskLevel.CRITICAL: 0.9
		}
		
		self._initialize_models()
		self._log_engine_initialized()
	
	def _initialize_models(self) -> None:
		"""Initialize predictive models"""
		try:
			# Initialize model configurations
			self.model_configs = {
				PredictionType.IDENTITY_RISK: {
					'model_type': 'random_forest_classifier',
					'features': ['verification_confidence', 'historical_success_rate', 'device_trust', 'location_risk', 'time_risk'],
					'target': 'identity_risk_score',
					'update_frequency_hours': 24
				},
				PredictionType.FRAUD_PROBABILITY: {
					'model_type': 'gradient_boosting_classifier',
					'features': ['verification_attempts', 'location_changes', 'device_changes', 'time_patterns', 'behavioral_anomalies'],
					'target': 'fraud_probability',
					'update_frequency_hours': 12
				},
				PredictionType.TEMPLATE_AGING: {
					'model_type': 'random_forest_regressor',
					'features': ['template_age_days', 'usage_frequency', 'quality_degradation', 'environmental_factors'],
					'target': 'aging_score',
					'update_frequency_hours': 168  # Weekly
				},
				PredictionType.BEHAVIORAL_ANOMALY: {
					'model_type': 'isolation_forest',
					'features': ['access_time', 'location_deviation', 'device_pattern', 'verification_confidence', 'emotional_state'],
					'target': 'anomaly_score',
					'update_frequency_hours': 6
				},
				PredictionType.ACCESS_PATTERN: {
					'model_type': 'time_series_predictor',
					'features': ['hour_of_day', 'day_of_week', 'location', 'device_type', 'historical_pattern'],
					'target': 'access_probability',
					'update_frequency_hours': 48
				},
				PredictionType.SECURITY_THREAT: {
					'model_type': 'ensemble_classifier',
					'features': ['risk_indicators', 'threat_intelligence', 'behavioral_changes', 'contextual_anomalies'],
					'target': 'threat_level',
					'update_frequency_hours': 1  # Real-time updates
				}
			}
			
			# Initialize models
			for prediction_type, config in self.model_configs.items():
				self._initialize_prediction_model(prediction_type, config)
			
		except Exception as e:
			print(f"Failed to initialize models: {e}")
	
	def _initialize_prediction_model(self, prediction_type: PredictionType, config: Dict[str, Any]) -> None:
		"""Initialize individual prediction model"""
		try:
			model_type = config['model_type']
			
			if model_type == 'random_forest_classifier' and 'RandomForestRegressor' in globals():
				model = GradientBoostingClassifier(
					n_estimators=100,
					learning_rate=0.1,
					max_depth=6,
					random_state=42
				)
			elif model_type == 'gradient_boosting_classifier' and 'GradientBoostingClassifier' in globals():
				model = GradientBoostingClassifier(
					n_estimators=150,
					learning_rate=0.05,
					max_depth=8,
					random_state=42
				)
			elif model_type == 'random_forest_regressor' and 'RandomForestRegressor' in globals():
				model = RandomForestRegressor(
					n_estimators=100,
					max_depth=10,
					random_state=42
				)
			elif model_type == 'isolation_forest' and 'IsolationForest' in globals():
				model = IsolationForest(
					contamination=0.1,
					random_state=42,
					n_estimators=100
				)
			else:
				# Fallback to simple model
				model = None
			
			self.models[prediction_type] = model
			self.feature_scalers[prediction_type] = StandardScaler() if 'StandardScaler' in globals() else None
			
			self.model_metadata[prediction_type] = {
				'config': config,
				'last_trained': None,
				'training_samples': 0,
				'accuracy': 0.0,
				'feature_importance': {},
				'version': '1.0.0'
			}
			
			# Initialize training data storage
			self.training_data[prediction_type] = {
				'features': [],
				'targets': [],
				'timestamps': []
			}
			
		except Exception as e:
			print(f"Failed to initialize model for {prediction_type}: {e}")
	
	def _log_engine_initialized(self) -> None:
		"""Log engine initialization"""
		print(f"Predictive Analytics Engine initialized for tenant {self.tenant_id}")
	
	def _log_prediction_operation(self, operation: str, prediction_type: str | None = None, result: str | None = None) -> None:
		"""Log prediction operations"""
		type_info = f" (Type: {prediction_type})" if prediction_type else ""
		result_info = f" [{result}]" if result else ""
		print(f"Predictive Analytics {operation}{type_info}{result_info}")
	
	async def predict_identity_risk(self, verification_context: Dict[str, Any]) -> Dict[str, Any]:
		"""Predict identity verification risk"""
		try:
			assert verification_context, "Verification context cannot be empty"
			
			prediction_id = uuid7str()
			start_time = datetime.now(timezone.utc)
			
			# Extract features for prediction
			features = await self._extract_risk_features(verification_context)
			
			# Make prediction
			risk_prediction = await self._make_prediction(
				PredictionType.IDENTITY_RISK,
				features,
				verification_context
			)
			
			# Analyze risk factors
			risk_factors = await self._analyze_risk_factors(features, verification_context)
			
			# Generate risk recommendations
			recommendations = self._generate_risk_recommendations(risk_prediction, risk_factors)
			
			# Calculate confidence intervals
			confidence_interval = self._calculate_confidence_interval(
				PredictionType.IDENTITY_RISK,
				features,
				risk_prediction['prediction_score']
			)
			
			result = {
				'prediction_id': prediction_id,
				'prediction_type': PredictionType.IDENTITY_RISK.value,
				'timestamp': start_time.isoformat(),
				'user_id': verification_context.get('user_id'),
				'risk_score': risk_prediction['prediction_score'],
				'risk_level': self._classify_risk_level(risk_prediction['prediction_score']).value,
				'confidence': risk_prediction['confidence'],
				'confidence_interval': confidence_interval,
				'risk_factors': risk_factors,
				'feature_contributions': risk_prediction.get('feature_contributions', {}),
				'recommendations': recommendations,
				'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
				'model_version': self.model_metadata[PredictionType.IDENTITY_RISK]['version']
			}
			
			# Learn from this prediction
			if self.learning_enabled:
				await self._store_prediction_for_learning(PredictionType.IDENTITY_RISK, features, result)
			
			self._log_prediction_operation(
				"PREDICT_RISK",
				PredictionType.IDENTITY_RISK.value,
				f"Risk: {result['risk_level']}, Score: {result['risk_score']:.3f}"
			)
			
			return result
			
		except Exception as e:
			print(f"Failed to predict identity risk: {e}")
			return {'error': str(e), 'prediction_type': PredictionType.IDENTITY_RISK.value}
	
	async def _extract_risk_features(self, verification_context: Dict[str, Any]) -> Dict[str, float]:
		"""Extract features for risk prediction"""
		try:
			features = {}
			
			# Verification confidence features
			primary_result = verification_context.get('primary_result', {})
			features['verification_confidence'] = primary_result.get('confidence_score', 0.5)
			features['similarity_score'] = primary_result.get('similarity_score', 0.5)
			features['quality_score'] = primary_result.get('quality_score', 0.5)
			features['liveness_score'] = primary_result.get('liveness_score', 0.5)
			
			# User context features
			user_context = verification_context.get('user_context', {})
			user_id = user_context.get('user_id')
			
			if user_id and user_id in self.identity_patterns:
				user_patterns = self.identity_patterns[user_id]
				features['historical_success_rate'] = user_patterns.get('success_rate', 0.8)
				features['avg_verification_time'] = user_patterns.get('avg_verification_time', 5.0)
				features['pattern_consistency'] = user_patterns.get('pattern_consistency', 0.7)
			else:
				features['historical_success_rate'] = 0.5  # Unknown user
				features['avg_verification_time'] = 10.0
				features['pattern_consistency'] = 0.5
			
			# Device and location features
			device_context = verification_context.get('device_context', {})
			features['device_trust'] = self._calculate_device_trust(device_context)
			features['device_consistency'] = self._calculate_device_consistency(user_id, device_context)
			
			location_context = verification_context.get('location_context', {})
			features['location_risk'] = self._calculate_location_risk(location_context)
			features['location_consistency'] = self._calculate_location_consistency(user_id, location_context)
			
			# Temporal features
			temporal_context = verification_context.get('temporal_context', {})
			features['time_risk'] = self._calculate_time_risk(temporal_context)
			features['access_frequency'] = self._calculate_access_frequency(user_id, temporal_context)
			
			# Business context features
			business_context = verification_context.get('business_context', {})
			features['operation_sensitivity'] = self._map_sensitivity_to_score(
				business_context.get('operation_sensitivity', 'low')
			)
			features['transaction_amount_risk'] = self._calculate_transaction_risk(
				business_context.get('transaction_amount', 0)
			)
			
			# Contextual intelligence features
			contextual_analysis = verification_context.get('contextual_analysis', {})
			features['contextual_risk'] = contextual_analysis.get('risk_score', 0.5)
			features['behavioral_anomaly'] = len(contextual_analysis.get('anomaly_indicators', [])) / 10.0
			
			# Emotion and stress features
			emotion_analysis = verification_context.get('emotion_analysis', {})
			features['stress_level'] = emotion_analysis.get('stress_analysis', {}).get('stress_score', 0.3)
			features['emotional_risk'] = len(emotion_analysis.get('risk_indicators', [])) / 5.0
			
			return features
			
		except Exception as e:
			print(f"Failed to extract risk features: {e}")
			return {}
	
	def _calculate_device_trust(self, device_context: Dict[str, Any]) -> float:
		"""Calculate device trust score"""
		try:
			trust_score = 0.5  # Default
			
			if device_context.get('is_trusted_device'):
				trust_score += 0.3
			if device_context.get('device_registered'):
				trust_score += 0.2
			if not device_context.get('is_jailbroken', False):
				trust_score += 0.2
			if not device_context.get('has_malware_indicators', False):
				trust_score += 0.2
			
			return min(1.0, trust_score)
			
		except Exception:
			return 0.5
	
	def _calculate_device_consistency(self, user_id: str, device_context: Dict[str, Any]) -> float:
		"""Calculate device usage consistency"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.5
			
			user_patterns = self.identity_patterns[user_id]
			typical_devices = user_patterns.get('typical_devices', [])
			current_device = device_context.get('device_id')
			
			if current_device in typical_devices:
				return 0.9
			elif device_context.get('device_type') in [d.get('type') for d in typical_devices]:
				return 0.6
			else:
				return 0.2
				
		except Exception:
			return 0.5
	
	def _calculate_location_risk(self, location_context: Dict[str, Any]) -> float:
		"""Calculate location-based risk"""
		try:
			risk_score = 0.0
			
			if location_context.get('is_high_risk_country'):
				risk_score += 0.3
			if location_context.get('is_vpn'):
				risk_score += 0.2
			if location_context.get('is_tor'):
				risk_score += 0.4
			if location_context.get('suspicious_ip'):
				risk_score += 0.3
			
			return min(1.0, risk_score)
			
		except Exception:
			return 0.0
	
	def _calculate_location_consistency(self, user_id: str, location_context: Dict[str, Any]) -> float:
		"""Calculate location usage consistency"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.5
			
			user_patterns = self.identity_patterns[user_id]
			typical_locations = user_patterns.get('typical_locations', [])
			current_location = location_context.get('coordinates', {})
			
			# Simple location matching (would use proper geospatial calculations)
			for location in typical_locations:
				if self._locations_are_similar(current_location, location):
					return 0.9
			
			# Check country/city level
			current_country = location_context.get('country')
			typical_countries = [loc.get('country') for loc in typical_locations]
			
			if current_country in typical_countries:
				return 0.6
			else:
				return 0.2
				
		except Exception:
			return 0.5
	
	def _locations_are_similar(self, loc1: Dict[str, Any], loc2: Dict[str, Any]) -> bool:
		"""Check if two locations are similar"""
		try:
			lat1, lon1 = loc1.get('latitude', 0), loc1.get('longitude', 0)
			lat2, lon2 = loc2.get('latitude', 0), loc2.get('longitude', 0)
			
			# Simple distance check (would use proper geospatial calculations)
			distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
			return distance < 0.01  # Approximately 1km
			
		except Exception:
			return False
	
	def _calculate_time_risk(self, temporal_context: Dict[str, Any]) -> float:
		"""Calculate time-based risk"""
		try:
			current_hour = datetime.now(timezone.utc).hour
			
			# High-risk hours (late night/early morning)
			if 0 <= current_hour <= 5 or current_hour >= 23:
				return 0.3
			# Medium-risk hours
			elif 6 <= current_hour <= 8 or 22 <= current_hour <= 23:
				return 0.1
			# Normal hours
			else:
				return 0.0
				
		except Exception:
			return 0.0
	
	def _calculate_access_frequency(self, user_id: str, temporal_context: Dict[str, Any]) -> float:
		"""Calculate access frequency pattern"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.5
			
			user_patterns = self.identity_patterns[user_id]
			typical_frequency = user_patterns.get('daily_access_frequency', 3)
			
			# Normalize to 0-1 scale
			normalized_frequency = min(1.0, typical_frequency / 10.0)
			return normalized_frequency
			
		except Exception:
			return 0.5
	
	def _map_sensitivity_to_score(self, sensitivity: str) -> float:
		"""Map operation sensitivity to numerical score"""
		sensitivity_mapping = {
			'low': 0.1,
			'medium': 0.5,
			'high': 0.8,
			'critical': 1.0
		}
		return sensitivity_mapping.get(sensitivity.lower(), 0.3)
	
	def _calculate_transaction_risk(self, amount: float) -> float:
		"""Calculate transaction amount risk"""
		try:
			if amount <= 1000:
				return 0.0
			elif amount <= 10000:
				return 0.2
			elif amount <= 50000:
				return 0.5
			elif amount <= 100000:
				return 0.7
			else:
				return 0.9
				
		except Exception:
			return 0.0
	
	async def _make_prediction(self, prediction_type: PredictionType, features: Dict[str, float], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Make prediction using trained model"""
		try:
			if prediction_type not in self.models or self.models[prediction_type] is None:
				# Fallback to heuristic prediction
				return await self._heuristic_prediction(prediction_type, features, context)
			
			model = self.models[prediction_type]
			scaler = self.feature_scalers[prediction_type]
			config = self.model_configs[prediction_type]
			
			# Prepare feature vector
			feature_vector = []
			for feature_name in config['features']:
				feature_vector.append(features.get(feature_name, 0.0))
			
			feature_array = np.array(feature_vector).reshape(1, -1)
			
			# Scale features if scaler is available
			if scaler is not None:
				try:
					feature_array = scaler.transform(feature_array)
				except Exception:
					# Scaler not fitted yet, use raw features
					pass
			
			# Make prediction
			if hasattr(model, 'predict_proba'):
				prediction_proba = model.predict_proba(feature_array)[0]
				prediction_score = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
			elif hasattr(model, 'decision_function'):
				decision_score = model.decision_function(feature_array)[0]
				# Normalize to 0-1 range
				prediction_score = max(0.0, min(1.0, (decision_score + 1) / 2))
			else:
				prediction_score = model.predict(feature_array)[0]
				prediction_score = max(0.0, min(1.0, prediction_score))
			
			# Calculate confidence
			confidence = self._calculate_prediction_confidence(model, feature_array, prediction_score)
			
			# Get feature contributions if available
			feature_contributions = self._calculate_feature_contributions(
				model, feature_array, config['features']
			)
			
			return {
				'prediction_score': float(prediction_score),
				'confidence': confidence,
				'feature_contributions': feature_contributions,
				'model_used': True
			}
			
		except Exception as e:
			print(f"Failed to make prediction with model: {e}")
			return await self._heuristic_prediction(prediction_type, features, context)
	
	async def _heuristic_prediction(self, prediction_type: PredictionType, features: Dict[str, float], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Fallback heuristic prediction when ML model is not available"""
		try:
			if prediction_type == PredictionType.IDENTITY_RISK:
				# Simple weighted average of risk factors
				risk_components = [
					features.get('verification_confidence', 0.5) * -0.3,  # Higher confidence = lower risk
					features.get('device_trust', 0.5) * -0.2,
					features.get('location_risk', 0.0) * 0.3,
					features.get('time_risk', 0.0) * 0.1,
					features.get('behavioral_anomaly', 0.0) * 0.2,
					features.get('stress_level', 0.3) * 0.1
				]
				
				risk_score = 0.5 + sum(risk_components)
				risk_score = max(0.0, min(1.0, risk_score))
				
				return {
					'prediction_score': risk_score,
					'confidence': 0.7,
					'feature_contributions': {f: v for f, v in features.items()},
					'model_used': False
				}
			
			# Default prediction for other types
			return {
				'prediction_score': 0.5,
				'confidence': 0.5,
				'feature_contributions': features,
				'model_used': False
			}
			
		except Exception as e:
			print(f"Failed to make heuristic prediction: {e}")
			return {
				'prediction_score': 0.5,
				'confidence': 0.3,
				'feature_contributions': {},
				'model_used': False
			}
	
	def _calculate_prediction_confidence(self, model, feature_array: np.ndarray, prediction_score: float) -> float:
		"""Calculate confidence in prediction"""
		try:
			# Different confidence calculations based on model type
			if hasattr(model, 'predict_proba'):
				proba = model.predict_proba(feature_array)[0]
				# Confidence is max probability
				confidence = max(proba)
			elif hasattr(model, 'decision_function'):
				# For SVM-like models, use distance from decision boundary
				decision_score = abs(model.decision_function(feature_array)[0])
				confidence = min(1.0, decision_score / 2.0)
			else:
				# For other models, use prediction certainty
				confidence = abs(prediction_score - 0.5) * 2  # Distance from uncertainty
			
			return max(0.1, min(1.0, confidence))
			
		except Exception:
			return 0.5
	
	def _calculate_feature_contributions(self, model, feature_array: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
		"""Calculate feature contributions to prediction"""
		try:
			contributions = {}
			
			if hasattr(model, 'feature_importances_'):
				# For tree-based models
				importances = model.feature_importances_
				for i, feature_name in enumerate(feature_names):
					if i < len(importances):
						contributions[feature_name] = float(importances[i])
			else:
				# Equal contributions for other models
				equal_contribution = 1.0 / len(feature_names)
				for feature_name in feature_names:
					contributions[feature_name] = equal_contribution
			
			return contributions
			
		except Exception:
			return {}
	
	def _classify_risk_level(self, risk_score: float) -> RiskLevel:
		"""Classify numerical risk score into risk level"""
		if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
			return RiskLevel.CRITICAL
		elif risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
			return RiskLevel.HIGH
		elif risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
			return RiskLevel.MEDIUM
		elif risk_score >= self.risk_thresholds[RiskLevel.LOW]:
			return RiskLevel.LOW
		else:
			return RiskLevel.VERY_LOW
	
	async def _analyze_risk_factors(self, features: Dict[str, float], context: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Analyze and identify specific risk factors"""
		try:
			risk_factors = []
			
			# Check verification confidence
			if features.get('verification_confidence', 1.0) < 0.7:
				risk_factors.append({
					'factor': 'low_verification_confidence',
					'severity': 'medium',
					'value': features.get('verification_confidence'),
					'description': 'Primary verification confidence is below threshold'
				})
			
			# Check device trust
			if features.get('device_trust', 1.0) < 0.5:
				risk_factors.append({
					'factor': 'untrusted_device',
					'severity': 'high',
					'value': features.get('device_trust'),
					'description': 'Device is not trusted or has security concerns'
				})
			
			# Check location risk
			if features.get('location_risk', 0.0) > 0.3:
				risk_factors.append({
					'factor': 'high_risk_location',
					'severity': 'medium',
					'value': features.get('location_risk'),
					'description': 'Access from high-risk location detected'
				})
			
			# Check behavioral anomalies
			if features.get('behavioral_anomaly', 0.0) > 0.3:
				risk_factors.append({
					'factor': 'behavioral_anomaly',
					'severity': 'high',
					'value': features.get('behavioral_anomaly'),
					'description': 'Unusual behavioral patterns detected'
				})
			
			# Check stress indicators
			if features.get('stress_level', 0.0) > 0.7:
				risk_factors.append({
					'factor': 'high_stress_detected',
					'severity': 'medium',
					'value': features.get('stress_level'),
					'description': 'High stress levels may indicate coercion'
				})
			
			# Check time-based risk
			if features.get('time_risk', 0.0) > 0.2:
				risk_factors.append({
					'factor': 'unusual_access_time',
					'severity': 'low',
					'value': features.get('time_risk'),
					'description': 'Access during unusual hours'
				})
			
			return risk_factors
			
		except Exception as e:
			print(f"Failed to analyze risk factors: {e}")
			return []
	
	def _generate_risk_recommendations(self, prediction: Dict[str, Any], risk_factors: List[Dict[str, Any]]) -> List[str]:
		"""Generate recommendations based on risk analysis"""
		try:
			recommendations = []
			risk_score = prediction['prediction_score']
			
			# High-level recommendations based on risk score
			if risk_score >= 0.8:
				recommendations.append("CRITICAL: Deny access and escalate to security team")
				recommendations.append("Require additional verification methods")
				recommendations.append("Initiate fraud investigation protocol")
			elif risk_score >= 0.6:
				recommendations.append("HIGH RISK: Require supervisor approval")
				recommendations.append("Implement additional verification steps")
				recommendations.append("Monitor session closely")
			elif risk_score >= 0.4:
				recommendations.append("MEDIUM RISK: Increase verification threshold")
				recommendations.append("Log detailed session information")
			elif risk_score >= 0.2:
				recommendations.append("LOW RISK: Standard monitoring")
			else:
				recommendations.append("VERY LOW RISK: Normal processing")
			
			# Specific recommendations based on risk factors
			for factor in risk_factors:
				if factor['factor'] == 'untrusted_device':
					recommendations.append("Register device before proceeding")
				elif factor['factor'] == 'high_risk_location':
					recommendations.append("Verify location through secondary means")
				elif factor['factor'] == 'behavioral_anomaly':
					recommendations.append("Request additional identity verification")
				elif factor['factor'] == 'high_stress_detected':
					recommendations.append("Check for potential coercion")
			
			return recommendations
			
		except Exception as e:
			print(f"Failed to generate risk recommendations: {e}")
			return ["Review verification manually"]
	
	def _calculate_confidence_interval(self, prediction_type: PredictionType, features: Dict[str, float], prediction_score: float) -> Dict[str, float]:
		"""Calculate confidence interval for prediction"""
		try:
			# Simplified confidence interval calculation
			# In practice, this would use proper statistical methods
			
			base_uncertainty = 0.1
			feature_uncertainty = len([v for v in features.values() if v == 0.0]) * 0.02
			model_uncertainty = 0.05 if self.models.get(prediction_type) else 0.15
			
			total_uncertainty = base_uncertainty + feature_uncertainty + model_uncertainty
			
			lower_bound = max(0.0, prediction_score - total_uncertainty)
			upper_bound = min(1.0, prediction_score + total_uncertainty)
			
			return {
				'lower_bound': lower_bound,
				'upper_bound': upper_bound,
				'uncertainty': total_uncertainty
			}
			
		except Exception:
			return {
				'lower_bound': max(0.0, prediction_score - 0.2),
				'upper_bound': min(1.0, prediction_score + 0.2),
				'uncertainty': 0.2
			}
	
	async def _store_prediction_for_learning(self, prediction_type: PredictionType, features: Dict[str, float], result: Dict[str, Any]) -> None:
		"""Store prediction for future model training"""
		try:
			if not self.learning_enabled:
				return
			
			training_data = self.training_data[prediction_type]
			config = self.model_configs[prediction_type]
			
			# Prepare feature vector
			feature_vector = []
			for feature_name in config['features']:
				feature_vector.append(features.get(feature_name, 0.0))
			
			# Store training data
			training_data['features'].append(feature_vector)
			training_data['targets'].append(result['risk_score'])
			training_data['timestamps'].append(datetime.now(timezone.utc).isoformat())
			
			# Limit training data size
			max_samples = 10000
			if len(training_data['features']) > max_samples:
				training_data['features'] = training_data['features'][-max_samples:]
				training_data['targets'] = training_data['targets'][-max_samples:]
				training_data['timestamps'] = training_data['timestamps'][-max_samples:]
			
			# Check if model should be retrained
			await self._check_model_retraining(prediction_type)
			
		except Exception as e:
			print(f"Failed to store prediction for learning: {e}")
	
	async def _check_model_retraining(self, prediction_type: PredictionType) -> None:
		"""Check if model should be retrained based on new data"""
		try:
			config = self.model_configs[prediction_type]
			metadata = self.model_metadata[prediction_type]
			
			# Check if enough time has passed since last training
			last_trained = metadata['last_trained']
			update_frequency_hours = config['update_frequency_hours']
			
			if last_trained:
				last_trained_time = datetime.fromisoformat(last_trained.replace('Z', '+00:00'))
				time_since_training = datetime.now(timezone.utc) - last_trained_time
				
				if time_since_training.total_seconds() < update_frequency_hours * 3600:
					return  # Not enough time passed
			
			# Check if enough new data is available
			training_data = self.training_data[prediction_type]
			if len(training_data['features']) < 100:
				return  # Not enough data
			
			# Retrain model
			await self._retrain_model(prediction_type)
			
		except Exception as e:
			print(f"Failed to check model retraining: {e}")
	
	async def _retrain_model(self, prediction_type: PredictionType) -> bool:
		"""Retrain prediction model with new data"""
		try:
			if prediction_type not in self.models:
				return False
			
			model = self.models[prediction_type]
			scaler = self.feature_scalers[prediction_type]
			training_data = self.training_data[prediction_type]
			
			if not training_data['features'] or not model:
				return False
			
			# Prepare training data
			X = np.array(training_data['features'])
			y = np.array(training_data['targets'])
			
			if len(X) < 10:  # Need minimum samples
				return False
			
			# Scale features
			if scaler is not None:
				X = scaler.fit_transform(X)
			
			# Split data for validation
			if len(X) > 20:
				X_train, X_test, y_train, y_test = train_test_split(
					X, y, test_size=0.2, random_state=42
				)
			else:
				X_train, X_test, y_train, y_test = X, X, y, y
			
			# Train model
			model.fit(X_train, y_train)
			
			# Evaluate model
			if len(X_test) > 0:
				if hasattr(model, 'predict_proba'):
					y_pred_proba = model.predict_proba(X_test)
					y_pred = (y_pred_proba[:, 1] > 0.5).astype(int) if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
				else:
					y_pred = model.predict(X_test)
				
				# Calculate accuracy (simplified)
				accuracy = np.mean(np.abs(y_pred - y_test) < 0.2)  # Within 20% tolerance
			else:
				accuracy = 0.0
			
			# Update metadata
			metadata = self.model_metadata[prediction_type]
			metadata['last_trained'] = datetime.now(timezone.utc).isoformat()
			metadata['training_samples'] = len(X)
			metadata['accuracy'] = accuracy
			
			self._log_prediction_operation(
				"RETRAIN_MODEL",
				prediction_type.value,
				f"Samples: {len(X)}, Accuracy: {accuracy:.3f}"
			)
			
			return True
			
		except Exception as e:
			print(f"Failed to retrain model: {e}")
			return False
	
	async def predict_fraud_probability(self, verification_context: Dict[str, Any]) -> Dict[str, Any]:
		"""Predict probability of fraudulent verification attempt"""
		try:
			# Extract fraud-specific features
			features = await self._extract_fraud_features(verification_context)
			
			# Make fraud prediction
			fraud_prediction = await self._make_prediction(
				PredictionType.FRAUD_PROBABILITY,
				features,
				verification_context
			)
			
			# Analyze fraud indicators
			fraud_indicators = self._analyze_fraud_indicators(features, verification_context)
			
			result = {
				'prediction_id': uuid7str(),
				'prediction_type': PredictionType.FRAUD_PROBABILITY.value,
				'timestamp': datetime.now(timezone.utc).isoformat(),
				'fraud_probability': fraud_prediction['prediction_score'],
				'fraud_risk_level': self._classify_risk_level(fraud_prediction['prediction_score']).value,
				'confidence': fraud_prediction['confidence'],
				'fraud_indicators': fraud_indicators,
				'feature_contributions': fraud_prediction.get('feature_contributions', {}),
				'recommended_actions': self._generate_fraud_recommendations(fraud_prediction, fraud_indicators)
			}
			
			self._log_prediction_operation(
				"PREDICT_FRAUD",
				PredictionType.FRAUD_PROBABILITY.value,
				f"Probability: {result['fraud_probability']:.3f}"
			)
			
			return result
			
		except Exception as e:
			print(f"Failed to predict fraud probability: {e}")
			return {'error': str(e), 'prediction_type': PredictionType.FRAUD_PROBABILITY.value}
	
	async def _extract_fraud_features(self, verification_context: Dict[str, Any]) -> Dict[str, float]:
		"""Extract fraud-specific features"""
		try:
			features = {}
			
			# Get base risk features
			base_features = await self._extract_risk_features(verification_context)
			features.update(base_features)
			
			# Add fraud-specific features
			user_context = verification_context.get('user_context', {})
			user_id = user_context.get('user_id')
			
			# Verification attempt patterns
			features['recent_failed_attempts'] = self._get_recent_failed_attempts(user_id)
			features['verification_velocity'] = self._calculate_verification_velocity(user_id)
			features['multiple_device_usage'] = self._check_multiple_device_usage(user_id)
			
			# Location and device anomalies
			features['location_jumps'] = self._calculate_location_jumps(user_id)
			features['device_switching_frequency'] = self._calculate_device_switching(user_id)
			features['suspicious_timing'] = self._calculate_suspicious_timing(verification_context)
			
			# Business context anomalies
			business_context = verification_context.get('business_context', {})
			features['unusual_transaction_pattern'] = self._analyze_transaction_pattern(
				user_id, business_context
			)
			features['off_hours_access'] = self._check_off_hours_access(verification_context)
			
			return features
			
		except Exception as e:
			print(f"Failed to extract fraud features: {e}")
			return {}
	
	def _get_recent_failed_attempts(self, user_id: str) -> float:
		"""Get number of recent failed verification attempts"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.0
			
			user_patterns = self.identity_patterns[user_id]
			recent_failures = user_patterns.get('recent_failed_attempts', 0)
			
			# Normalize to 0-1 scale (more than 5 failures is very suspicious)
			return min(1.0, recent_failures / 5.0)
			
		except Exception:
			return 0.0
	
	def _calculate_verification_velocity(self, user_id: str) -> float:
		"""Calculate verification attempt velocity"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.0
			
			user_patterns = self.identity_patterns[user_id]
			attempts_last_hour = user_patterns.get('attempts_last_hour', 0)
			
			# Normal users rarely verify more than 3 times per hour
			return min(1.0, attempts_last_hour / 3.0)
			
		except Exception:
			return 0.0
	
	def _check_multiple_device_usage(self, user_id: str) -> float:
		"""Check for suspicious multiple device usage"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.0
			
			user_patterns = self.identity_patterns[user_id]
			devices_last_24h = user_patterns.get('devices_last_24h', 1)
			
			# Using more than 3 devices in 24h is suspicious
			return min(1.0, max(0.0, (devices_last_24h - 1) / 3.0))
			
		except Exception:
			return 0.0
	
	def _calculate_location_jumps(self, user_id: str) -> float:
		"""Calculate suspicious location changes"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.0
			
			user_patterns = self.identity_patterns[user_id]
			location_jumps = user_patterns.get('location_jumps_last_24h', 0)
			
			# More than 2 location jumps in 24h is suspicious
			return min(1.0, location_jumps / 2.0)
			
		except Exception:
			return 0.0
	
	def _calculate_device_switching(self, user_id: str) -> float:
		"""Calculate device switching frequency"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.0
			
			user_patterns = self.identity_patterns[user_id]
			device_switches = user_patterns.get('device_switches_last_week', 0)
			
			# Frequent device switching can indicate fraud
			return min(1.0, device_switches / 5.0)
			
		except Exception:
			return 0.0
	
	def _calculate_suspicious_timing(self, verification_context: Dict[str, Any]) -> float:
		"""Calculate suspicious timing patterns"""
		try:
			# Check for rapid successive attempts
			temporal_context = verification_context.get('temporal_context', {})
			time_since_last_attempt = temporal_context.get('time_since_last_attempt_minutes', 60)
			
			# Attempts within 1 minute are suspicious
			if time_since_last_attempt < 1:
				return 1.0
			elif time_since_last_attempt < 5:
				return 0.5
			else:
				return 0.0
				
		except Exception:
			return 0.0
	
	def _analyze_transaction_pattern(self, user_id: str, business_context: Dict[str, Any]) -> float:
		"""Analyze unusual transaction patterns"""
		try:
			if not user_id or user_id not in self.identity_patterns:
				return 0.0
			
			user_patterns = self.identity_patterns[user_id]
			typical_transaction_amount = user_patterns.get('typical_transaction_amount', 1000)
			current_amount = business_context.get('transaction_amount', 0)
			
			if current_amount == 0:
				return 0.0
			
			# Calculate deviation from typical amount
			ratio = current_amount / typical_transaction_amount
			
			if ratio > 10 or ratio < 0.1:  # 10x higher or 10x lower
				return 1.0
			elif ratio > 5 or ratio < 0.2:  # 5x higher or 5x lower
				return 0.5
			else:
				return 0.0
				
		except Exception:
			return 0.0
	
	def _check_off_hours_access(self, verification_context: Dict[str, Any]) -> float:
		"""Check for off-hours access patterns"""
		try:
			current_hour = datetime.now(timezone.utc).hour
			
			# Define off-hours (10 PM to 6 AM)
			if 22 <= current_hour or current_hour <= 6:
				return 0.5
			else:
				return 0.0
				
		except Exception:
			return 0.0
	
	def _analyze_fraud_indicators(self, features: Dict[str, float], context: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Analyze specific fraud indicators"""
		try:
			indicators = []
			
			# High-risk indicators
			if features.get('recent_failed_attempts', 0) > 0.5:
				indicators.append({
					'indicator': 'multiple_failed_attempts',
					'severity': 'high',
					'value': features.get('recent_failed_attempts'),
					'description': 'Multiple recent failed verification attempts'
				})
			
			if features.get('verification_velocity', 0) > 0.6:
				indicators.append({
					'indicator': 'high_verification_velocity',
					'severity': 'high',
					'value': features.get('verification_velocity'),
					'description': 'Unusually high verification attempt frequency'
				})
			
			if features.get('location_jumps', 0) > 0.5:
				indicators.append({
					'indicator': 'impossible_travel',
					'severity': 'critical',
					'value': features.get('location_jumps'),
					'description': 'Geographically impossible travel detected'
				})
			
			if features.get('multiple_device_usage', 0) > 0.5:
				indicators.append({
					'indicator': 'device_proliferation',
					'severity': 'medium',
					'value': features.get('multiple_device_usage'),
					'description': 'Multiple devices used in short timeframe'
				})
			
			return indicators
			
		except Exception as e:
			print(f"Failed to analyze fraud indicators: {e}")
			return []
	
	def _generate_fraud_recommendations(self, prediction: Dict[str, Any], indicators: List[Dict[str, Any]]) -> List[str]:
		"""Generate fraud prevention recommendations"""
		try:
			recommendations = []
			fraud_probability = prediction['prediction_score']
			
			if fraud_probability >= 0.8:
				recommendations.append("CRITICAL: Block transaction immediately")
				recommendations.append("Initiate fraud investigation")
				recommendations.append("Lock account pending investigation")
			elif fraud_probability >= 0.6:
				recommendations.append("HIGH RISK: Require additional verification")
				recommendations.append("Contact user through alternative channel")
				recommendations.append("Review transaction details manually")
			elif fraud_probability >= 0.4:
				recommendations.append("MEDIUM RISK: Enhanced monitoring")
				recommendations.append("Request additional authentication factors")
			
			# Specific recommendations based on indicators
			for indicator in indicators:
				if indicator['indicator'] == 'impossible_travel':
					recommendations.append("Verify location through secondary means")
				elif indicator['indicator'] == 'device_proliferation':
					recommendations.append("Restrict to registered devices only")
				elif indicator['indicator'] == 'high_verification_velocity':
					recommendations.append("Implement rate limiting")
			
			return recommendations
			
		except Exception as e:
			print(f"Failed to generate fraud recommendations: {e}")
			return ["Manual review required"]
	
	def get_engine_statistics(self) -> Dict[str, Any]:
		"""Get predictive analytics engine statistics"""
		try:
			model_stats = {}
			for prediction_type, metadata in self.model_metadata.items():
				model_stats[prediction_type.value] = {
					'trained': metadata['last_trained'] is not None,
					'training_samples': metadata['training_samples'],
					'accuracy': metadata['accuracy'],
					'version': metadata['version']
				}
			
			return {
				'tenant_id': self.tenant_id,
				'prediction_enabled': self.prediction_enabled,
				'learning_enabled': self.learning_enabled,
				'real_time_analysis': self.real_time_analysis,
				'supported_predictions': [pt.value for pt in PredictionType],
				'risk_levels': [rl.value for rl in RiskLevel],
				'models': model_stats,
				'users_with_patterns': len(self.identity_patterns),
				'total_training_samples': sum(
					len(data['features']) for data in self.training_data.values()
				)
			}
			
		except Exception as e:
			print(f"Failed to get engine statistics: {e}")
			return {'error': str(e)}

# Export for use in other modules
__all__ = ['PredictiveAnalyticsEngine', 'PredictionType', 'RiskLevel']