#!/usr/bin/env python3
"""
APG Key Management - Advanced AI/ML Features
Cutting-edge AI/ML features for intelligent key management

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import tensorflow as tf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import sqlite3
from uuid_extensions import uuid7str

from .service import KeyManagementService
from .models import KeyAlgorithm, KeyUsage


class MLModelType(str, Enum):
	"""Machine learning model types"""
	ANOMALY_DETECTION = "anomaly_detection"
	KEY_RECOMMENDATION = "key_recommendation"
	USAGE_PREDICTION = "usage_prediction"
	SECURITY_SCORING = "security_scoring"
	LIFECYCLE_OPTIMIZATION = "lifecycle_optimization"
	THREAT_DETECTION = "threat_detection"


class PredictionConfidence(str, Enum):
	"""Prediction confidence levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERY_HIGH = "very_high"


@dataclass
class MLPrediction:
	"""Machine learning prediction result"""
	model_type: MLModelType
	prediction: Any
	confidence: PredictionConfidence
	probability: float
	features_used: List[str]
	timestamp: datetime = field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
	"""ML model training configuration"""
	model_type: MLModelType
	training_data_size: int
	validation_split: float = 0.2
	epochs: int = 100
	batch_size: int = 32
	learning_rate: float = 0.001
	early_stopping_patience: int = 10
	feature_selection: bool = True
	hyperparameter_tuning: bool = True


class FeatureExtractor:
	"""Advanced feature extraction for key management data"""
	
	def __init__(self, service: KeyManagementService):
		self.service = service
		self.feature_cache: Dict[str, Any] = {}
	
	async def extract_key_features(self, key_id: str) -> Dict[str, float]:
		"""Extract comprehensive features from key data"""
		if key_id in self.feature_cache:
			return self.feature_cache[key_id]
		
		# Get key information
		key_info = await self.service.retrieve_key(key_id, "system")
		
		# Basic key features
		features = {
			'algorithm_complexity': self._algorithm_complexity_score(key_info.spec.algorithm),
			'key_age_days': (datetime.utcnow() - key_info.spec.created_at).days,
			'usage_count': await self._get_key_usage_count(key_id),
			'usage_frequency': await self._calculate_usage_frequency(key_id),
			'last_used_days_ago': await self._get_days_since_last_use(key_id)
		}
		
		# Security features
		security_features = await self._extract_security_features(key_id)
		features.update(security_features)
		
		# Behavioral features
		behavioral_features = await self._extract_behavioral_features(key_id)
		features.update(behavioral_features)
		
		# Contextual features
		contextual_features = await self._extract_contextual_features(key_id)
		features.update(contextual_features)
		
		# Cache features
		self.feature_cache[key_id] = features
		
		return features
	
	def _algorithm_complexity_score(self, algorithm: KeyAlgorithm) -> float:
		"""Calculate complexity score for cryptographic algorithm"""
		complexity_scores = {
			KeyAlgorithm.AES_128: 0.6,
			KeyAlgorithm.AES_256: 0.8,
			KeyAlgorithm.RSA_2048: 0.7,
			KeyAlgorithm.RSA_4096: 0.9,
			KeyAlgorithm.ECDSA_P256: 0.75,
			KeyAlgorithm.ECDSA_P384: 0.85,
			KeyAlgorithm.KYBER_512: 0.95,
			KeyAlgorithm.KYBER_768: 0.98,
			KeyAlgorithm.KYBER_1024: 1.0
		}
		return complexity_scores.get(algorithm, 0.5)
	
	async def _extract_security_features(self, key_id: str) -> Dict[str, float]:
		"""Extract security-related features"""
		features = {}
		
		try:
			# Get security events for this key
			if hasattr(self.service, '_db_pool') and self.service._db_pool:
				async with self.service._db_pool.acquire() as conn:
					# Security incidents count
					incidents = await conn.fetchval("""
						SELECT COUNT(*) FROM km_security_events 
						WHERE key_id = $1 AND event_type = 'security_incident'
					""", key_id)
					
					features['security_incidents_count'] = float(incidents or 0)
					
					# Failed access attempts
					failed_attempts = await conn.fetchval("""
						SELECT COUNT(*) FROM km_audit_log 
						WHERE key_id = $1 AND action = 'access' AND status = 'failed'
						AND timestamp > $2
					""", key_id, datetime.utcnow() - timedelta(days=30))
					
					features['failed_access_attempts'] = float(failed_attempts or 0)
					
					# Unique users accessing key
					unique_users = await conn.fetchval("""
						SELECT COUNT(DISTINCT user_id) FROM km_audit_log 
						WHERE key_id = $1 AND action = 'access' AND status = 'success'
					""", key_id)
					
					features['unique_users_count'] = float(unique_users or 0)
		
		except Exception as e:
			logging.error(f"Error extracting security features for key {key_id}: {e}")
			features.update({
				'security_incidents_count': 0.0,
				'failed_access_attempts': 0.0,
				'unique_users_count': 0.0
			})
		
		return features
	
	async def _extract_behavioral_features(self, key_id: str) -> Dict[str, float]:
		"""Extract behavioral usage patterns"""
		features = {}
		
		try:
			if hasattr(self.service, '_db_pool') and self.service._db_pool:
				async with self.service._db_pool.acquire() as conn:
					# Usage patterns over time
					usage_by_hour = await conn.fetch("""
						SELECT EXTRACT(HOUR FROM timestamp) as hour, COUNT(*) as count
						FROM km_audit_log 
						WHERE key_id = $1 AND action IN ('encrypt', 'decrypt')
						AND timestamp > $2
						GROUP BY EXTRACT(HOUR FROM timestamp)
					""", key_id, datetime.utcnow() - timedelta(days=30))
					
					# Calculate usage pattern regularity
					if usage_by_hour:
						hourly_counts = [row['count'] for row in usage_by_hour]
						features['usage_pattern_variance'] = float(np.var(hourly_counts))
						features['usage_pattern_mean'] = float(np.mean(hourly_counts))
						features['peak_usage_hour'] = float(max(usage_by_hour, key=lambda x: x['count'])['hour'])
					else:
						features.update({
							'usage_pattern_variance': 0.0,
							'usage_pattern_mean': 0.0,
							'peak_usage_hour': 0.0
						})
					
					# Geographic diversity of access
					geo_locations = await conn.fetchval("""
						SELECT COUNT(DISTINCT source_ip) FROM km_audit_log 
						WHERE key_id = $1 AND timestamp > $2
					""", key_id, datetime.utcnow() - timedelta(days=30))
					
					features['geographic_diversity'] = float(geo_locations or 0)
		
		except Exception as e:
			logging.error(f"Error extracting behavioral features for key {key_id}: {e}")
			features.update({
				'usage_pattern_variance': 0.0,
				'usage_pattern_mean': 0.0,
				'peak_usage_hour': 0.0,
				'geographic_diversity': 0.0
			})
		
		return features
	
	async def _extract_contextual_features(self, key_id: str) -> Dict[str, float]:
		"""Extract contextual environment features"""
		features = {}
		
		# Tenant-level features
		try:
			key_info = await self.service.retrieve_key(key_id, "system")
			tenant_id = key_info.spec.tenant_id
			
			if hasattr(self.service, '_db_pool') and self.service._db_pool:
				async with self.service._db_pool.acquire() as conn:
					# Total keys in tenant
					tenant_keys = await conn.fetchval("""
						SELECT COUNT(*) FROM km_keys WHERE tenant_id = $1
					""", tenant_id)
					
					features['tenant_key_count'] = float(tenant_keys or 0)
					
					# Tenant activity level
					tenant_activity = await conn.fetchval("""
						SELECT COUNT(*) FROM km_audit_log 
						WHERE tenant_id = $1 AND timestamp > $2
					""", tenant_id, datetime.utcnow() - timedelta(days=7))
					
					features['tenant_activity_level'] = float(tenant_activity or 0)
		
		except Exception as e:
			logging.error(f"Error extracting contextual features for key {key_id}: {e}")
			features.update({
				'tenant_key_count': 0.0,
				'tenant_activity_level': 0.0
			})
		
		return features


class AnomalyDetectionModel:
	"""Advanced anomaly detection for key usage patterns"""
	
	def __init__(self, feature_extractor: FeatureExtractor):
		self.feature_extractor = feature_extractor
		self.model = IsolationForest(contamination=0.1, random_state=42)
		self.scaler = StandardScaler()
		self.is_trained = False
		self.feature_names: List[str] = []
	
	async def train(self, key_ids: List[str]) -> Dict[str, Any]:
		"""Train anomaly detection model"""
		logging.info(f"Training anomaly detection model with {len(key_ids)} keys")
		
		# Extract features for all keys
		features_list = []
		valid_key_ids = []
		
		for key_id in key_ids:
			try:
				features = await self.feature_extractor.extract_key_features(key_id)
				features_list.append(list(features.values()))
				valid_key_ids.append(key_id)
				
				if not self.feature_names:
					self.feature_names = list(features.keys())
			
			except Exception as e:
				logging.warning(f"Error extracting features for key {key_id}: {e}")
				continue
		
		if len(features_list) < 10:
			raise ValueError("Insufficient training data. Need at least 10 valid keys.")
		
		# Convert to numpy array and scale
		X = np.array(features_list)
		X_scaled = self.scaler.fit_transform(X)
		
		# Train isolation forest
		self.model.fit(X_scaled)
		self.is_trained = True
		
		# Calculate training metrics
		anomaly_scores = self.model.decision_function(X_scaled)
		outliers = self.model.predict(X_scaled)
		
		training_metrics = {
			'total_samples': len(features_list),
			'outliers_detected': np.sum(outliers == -1),
			'outlier_percentage': (np.sum(outliers == -1) / len(features_list)) * 100,
			'mean_anomaly_score': np.mean(anomaly_scores),
			'std_anomaly_score': np.std(anomaly_scores),
			'feature_count': len(self.feature_names)
		}
		
		logging.info(f"Anomaly detection model trained: {training_metrics}")
		return training_metrics
	
	async def predict_anomaly(self, key_id: str) -> MLPrediction:
		"""Predict if key usage pattern is anomalous"""
		if not self.is_trained:
			raise RuntimeError("Model not trained. Call train() first.")
		
		# Extract features
		features = await self.feature_extractor.extract_key_features(key_id)
		feature_values = [features.get(name, 0.0) for name in self.feature_names]
		
		# Scale features
		X = np.array([feature_values])
		X_scaled = self.scaler.transform(X)
		
		# Make prediction
		anomaly_prediction = self.model.predict(X_scaled)[0]
		anomaly_score = self.model.decision_function(X_scaled)[0]
		
		# Convert to probability and confidence
		probability = abs(anomaly_score) / 2.0  # Normalize to 0-1 range
		
		if probability > 0.8:
			confidence = PredictionConfidence.VERY_HIGH
		elif probability > 0.6:
			confidence = PredictionConfidence.HIGH
		elif probability > 0.4:
			confidence = PredictionConfidence.MEDIUM
		else:
			confidence = PredictionConfidence.LOW
		
		return MLPrediction(
			model_type=MLModelType.ANOMALY_DETECTION,
			prediction=anomaly_prediction == -1,  # True if anomaly
			confidence=confidence,
			probability=probability,
			features_used=self.feature_names,
			metadata={
				'anomaly_score': anomaly_score,
				'feature_values': dict(zip(self.feature_names, feature_values))
			}
		)


class KeyRecommendationModel:
	"""AI-powered key algorithm and configuration recommendations"""
	
	def __init__(self, feature_extractor: FeatureExtractor):
		self.feature_extractor = feature_extractor
		self.model = RandomForestClassifier(n_estimators=100, random_state=42)
		self.scaler = StandardScaler()
		self.label_encoder = LabelEncoder()
		self.is_trained = False
		self.feature_names: List[str] = []
	
	async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Train key recommendation model"""
		logging.info(f"Training key recommendation model with {len(training_data)} samples")
		
		if len(training_data) < 50:
			raise ValueError("Insufficient training data. Need at least 50 samples.")
		
		# Prepare training data
		features_list = []
		labels = []
		
		for sample in training_data:
			try:
				# Extract features from usage context
				context_features = self._extract_context_features(sample)
				features_list.append(list(context_features.values()))
				
				# Extract label (recommended algorithm)
				labels.append(sample['recommended_algorithm'])
				
				if not self.feature_names:
					self.feature_names = list(context_features.keys())
			
			except Exception as e:
				logging.warning(f"Error processing training sample: {e}")
				continue
		
		# Convert to numpy arrays
		X = np.array(features_list)
		y = self.label_encoder.fit_transform(labels)
		
		# Split data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		
		# Scale features
		X_train_scaled = self.scaler.fit_transform(X_train)
		X_test_scaled = self.scaler.transform(X_test)
		
		# Train model
		self.model.fit(X_train_scaled, y_train)
		self.is_trained = True
		
		# Evaluate model
		y_pred = self.model.predict(X_test_scaled)
		accuracy = accuracy_score(y_test, y_pred)
		precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
		
		training_metrics = {
			'total_samples': len(features_list),
			'training_samples': len(X_train),
			'test_samples': len(X_test),
			'accuracy': accuracy,
			'precision': precision,
			'recall': recall,
			'f1_score': f1,
			'feature_count': len(self.feature_names),
			'unique_algorithms': len(self.label_encoder.classes_)
		}
		
		logging.info(f"Key recommendation model trained: {training_metrics}")
		return training_metrics
	
	def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, float]:
		"""Extract features from usage context"""
		features = {
			'data_size_log': np.log10(max(context.get('data_size', 1), 1)),
			'operations_per_day': float(context.get('operations_per_day', 0)),
			'security_level': float(context.get('security_level', 3)),  # 1-5 scale
			'compliance_requirements': float(context.get('compliance_requirements', 0)),  # 0-1
			'performance_priority': float(context.get('performance_priority', 0.5)),  # 0-1
			'storage_cost_sensitivity': float(context.get('storage_cost_sensitivity', 0.5)),
			'network_latency_ms': float(context.get('network_latency_ms', 10)),
			'user_count': float(context.get('user_count', 1)),
			'geographic_distribution': float(context.get('geographic_distribution', 1))
		}
		
		# Categorical features (one-hot encoded)
		use_case = context.get('use_case', 'general')
		use_case_features = {
			'use_case_general': 1.0 if use_case == 'general' else 0.0,
			'use_case_document': 1.0 if use_case == 'document' else 0.0,
			'use_case_database': 1.0 if use_case == 'database' else 0.0,
			'use_case_communication': 1.0 if use_case == 'communication' else 0.0,
			'use_case_iot': 1.0 if use_case == 'iot' else 0.0
		}
		
		features.update(use_case_features)
		return features
	
	async def recommend_algorithm(self, context: Dict[str, Any]) -> MLPrediction:
		"""Recommend optimal key algorithm for given context"""
		if not self.is_trained:
			raise RuntimeError("Model not trained. Call train() first.")
		
		# Extract context features
		context_features = self._extract_context_features(context)
		feature_values = [context_features.get(name, 0.0) for name in self.feature_names]
		
		# Scale features
		X = np.array([feature_values])
		X_scaled = self.scaler.transform(X)
		
		# Make prediction
		prediction = self.model.predict(X_scaled)[0]
		probabilities = self.model.predict_proba(X_scaled)[0]
		
		# Get recommended algorithm
		recommended_algorithm = self.label_encoder.inverse_transform([prediction])[0]
		confidence_score = max(probabilities)
		
		# Determine confidence level
		if confidence_score > 0.8:
			confidence = PredictionConfidence.VERY_HIGH
		elif confidence_score > 0.6:
			confidence = PredictionConfidence.HIGH
		elif confidence_score > 0.4:
			confidence = PredictionConfidence.MEDIUM
		else:
			confidence = PredictionConfidence.LOW
		
		# Get feature importance
		feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
		
		return MLPrediction(
			model_type=MLModelType.KEY_RECOMMENDATION,
			prediction=recommended_algorithm,
			confidence=confidence,
			probability=confidence_score,
			features_used=self.feature_names,
			metadata={
				'all_probabilities': dict(zip(self.label_encoder.classes_, probabilities)),
				'feature_importance': feature_importance,
				'context': context
			}
		)


class UsagePredictionModel:
	"""Predict key usage patterns using time series analysis"""
	
	def __init__(self, feature_extractor: FeatureExtractor):
		self.feature_extractor = feature_extractor
		self.model = None
		self.is_trained = False
		self.sequence_length = 24  # Hours of history to use for prediction
	
	async def train(self, usage_data: Dict[str, List[Tuple[datetime, int]]]) -> Dict[str, Any]:
		"""Train usage prediction model"""
		logging.info(f"Training usage prediction model with data for {len(usage_data)} keys")
		
		# Build LSTM model for time series prediction
		model = tf.keras.Sequential([
			tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
			tf.keras.layers.LSTM(50, return_sequences=False),
			tf.keras.layers.Dense(25),
			tf.keras.layers.Dense(1)
		])
		
		model.compile(optimizer='adam', loss='mean_squared_error')
		
		# Prepare training data
		X_train, y_train = self._prepare_time_series_data(usage_data)
		
		if len(X_train) < 100:
			raise ValueError("Insufficient time series data. Need at least 100 sequences.")
		
		# Train model
		history = model.fit(
			X_train, y_train,
			batch_size=32,
			epochs=50,
			validation_split=0.2,
			verbose=0
		)
		
		self.model = model
		self.is_trained = True
		
		training_metrics = {
			'total_sequences': len(X_train),
			'final_loss': history.history['loss'][-1],
			'final_val_loss': history.history['val_loss'][-1],
			'epochs_trained': len(history.history['loss'])
		}
		
		logging.info(f"Usage prediction model trained: {training_metrics}")
		return training_metrics
	
	def _prepare_time_series_data(self, usage_data: Dict[str, List[Tuple[datetime, int]]]) -> Tuple[np.ndarray, np.ndarray]:
		"""Prepare time series data for training"""
		sequences = []
		targets = []
		
		for key_id, time_series in usage_data.items():
			if len(time_series) < self.sequence_length + 1:
				continue
			
			# Sort by timestamp
			time_series.sort(key=lambda x: x[0])
			
			# Extract usage counts
			usage_counts = [count for _, count in time_series]
			
			# Create sequences
			for i in range(len(usage_counts) - self.sequence_length):
				seq = usage_counts[i:i + self.sequence_length]
				target = usage_counts[i + self.sequence_length]
				
				sequences.append(seq)
				targets.append(target)
		
		X = np.array(sequences).reshape(-1, self.sequence_length, 1)
		y = np.array(targets)
		
		return X, y
	
	async def predict_usage(self, key_id: str, hours_ahead: int = 24) -> MLPrediction:
		"""Predict key usage for specified hours ahead"""
		if not self.is_trained:
			raise RuntimeError("Model not trained. Call train() first.")
		
		# Get recent usage history
		recent_usage = await self._get_recent_usage_history(key_id, self.sequence_length)
		
		if len(recent_usage) < self.sequence_length:
			# Not enough history, return low confidence prediction
			return MLPrediction(
				model_type=MLModelType.USAGE_PREDICTION,
				prediction=0,
				confidence=PredictionConfidence.LOW,
				probability=0.1,
				features_used=['historical_usage'],
				metadata={'reason': 'insufficient_history'}
			)
		
		# Prepare input sequence
		X = np.array(recent_usage).reshape(1, self.sequence_length, 1)
		
		# Make prediction
		predicted_usage = self.model.predict(X, verbose=0)[0][0]
		
		# Calculate confidence based on recent usage variance
		usage_variance = np.var(recent_usage)
		if usage_variance < 1:
			confidence = PredictionConfidence.VERY_HIGH
		elif usage_variance < 10:
			confidence = PredictionConfidence.HIGH
		elif usage_variance < 50:
			confidence = PredictionConfidence.MEDIUM
		else:
			confidence = PredictionConfidence.LOW
		
		return MLPrediction(
			model_type=MLModelType.USAGE_PREDICTION,
			prediction=max(0, int(predicted_usage)),
			confidence=confidence,
			probability=min(1.0, 1.0 / (1.0 + usage_variance / 10)),
			features_used=['historical_usage'],
			metadata={
				'hours_ahead': hours_ahead,
				'recent_usage': recent_usage,
				'usage_variance': usage_variance
			}
		)
	
	async def _get_recent_usage_history(self, key_id: str, hours: int) -> List[int]:
		"""Get recent hourly usage history for a key"""
		try:
			if hasattr(self.feature_extractor.service, '_db_pool'):
				async with self.feature_extractor.service._db_pool.acquire() as conn:
					usage_data = await conn.fetch("""
						SELECT 
							DATE_TRUNC('hour', timestamp) as hour,
							COUNT(*) as usage_count
						FROM km_audit_log 
						WHERE key_id = $1 
						AND action IN ('encrypt', 'decrypt')
						AND timestamp > $2
						GROUP BY DATE_TRUNC('hour', timestamp)
						ORDER BY hour DESC
						LIMIT $3
					""", key_id, datetime.utcnow() - timedelta(hours=hours), hours)
					
					return [row['usage_count'] for row in usage_data]
		
		except Exception as e:
			logging.error(f"Error getting usage history for key {key_id}: {e}")
		
		return []


class SecurityScoringModel:
	"""AI-based security risk scoring for keys"""
	
	def __init__(self, feature_extractor: FeatureExtractor):
		self.feature_extractor = feature_extractor
		self.model = tf.keras.Sequential([
			tf.keras.layers.Dense(64, activation='relu'),
			tf.keras.layers.Dropout(0.3),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dropout(0.3),
			tf.keras.layers.Dense(16, activation='relu'),
			tf.keras.layers.Dense(1, activation='sigmoid')  # Security score 0-1
		])
		
		self.model.compile(
			optimizer='adam',
			loss='binary_crossentropy',
			metrics=['accuracy']
		)
		
		self.scaler = StandardScaler()
		self.is_trained = False
		self.feature_names: List[str] = []
	
	async def train(self, labeled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Train security scoring model"""
		logging.info(f"Training security scoring model with {len(labeled_data)} samples")
		
		# Extract features and labels
		features_list = []
		scores = []
		
		for sample in labeled_data:
			key_id = sample['key_id']
			security_score = sample['security_score']  # 0-1 scale
			
			features = await self.feature_extractor.extract_key_features(key_id)
			features_list.append(list(features.values()))
			scores.append(security_score)
			
			if not self.feature_names:
				self.feature_names = list(features.keys())
		
		# Prepare data
		X = np.array(features_list)
		y = np.array(scores)
		
		# Scale features
		X_scaled = self.scaler.fit_transform(X)
		
		# Split data
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
		
		# Train model
		history = self.model.fit(
			X_train, y_train,
			validation_data=(X_test, y_test),
			epochs=100,
			batch_size=32,
			verbose=0
		)
		
		self.is_trained = True
		
		# Evaluate model
		test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
		
		training_metrics = {
			'total_samples': len(features_list),
			'test_accuracy': test_accuracy,
			'test_loss': test_loss,
			'final_val_loss': history.history['val_loss'][-1]
		}
		
		logging.info(f"Security scoring model trained: {training_metrics}")
		return training_metrics
	
	async def calculate_security_score(self, key_id: str) -> MLPrediction:
		"""Calculate security risk score for a key"""
		if not self.is_trained:
			raise RuntimeError("Model not trained. Call train() first.")
		
		# Extract features
		features = await self.feature_extractor.extract_key_features(key_id)
		feature_values = [features.get(name, 0.0) for name in self.feature_names]
		
		# Scale features
		X = np.array([feature_values])
		X_scaled = self.scaler.transform(X)
		
		# Make prediction
		security_score = self.model.predict(X_scaled, verbose=0)[0][0]
		
		# Determine confidence and risk level
		if security_score < 0.3:
			risk_level = "HIGH"
			confidence = PredictionConfidence.HIGH
		elif security_score < 0.6:
			risk_level = "MEDIUM"
			confidence = PredictionConfidence.MEDIUM
		else:
			risk_level = "LOW"
			confidence = PredictionConfidence.HIGH
		
		return MLPrediction(
			model_type=MLModelType.SECURITY_SCORING,
			prediction=risk_level,
			confidence=confidence,
			probability=float(security_score),
			features_used=self.feature_names,
			metadata={
				'security_score': float(security_score),
				'feature_values': dict(zip(self.feature_names, feature_values))
			}
		)


class AIMLManager:
	"""Central manager for all AI/ML capabilities"""
	
	def __init__(self, service: KeyManagementService):
		self.service = service
		self.feature_extractor = FeatureExtractor(service)
		
		# Initialize models
		self.anomaly_model = AnomalyDetectionModel(self.feature_extractor)
		self.recommendation_model = KeyRecommendationModel(self.feature_extractor)
		self.usage_model = UsagePredictionModel(self.feature_extractor)
		self.security_model = SecurityScoringModel(self.feature_extractor)
		
		# Model registry
		self.models: Dict[MLModelType, Any] = {
			MLModelType.ANOMALY_DETECTION: self.anomaly_model,
			MLModelType.KEY_RECOMMENDATION: self.recommendation_model,
			MLModelType.USAGE_PREDICTION: self.usage_model,
			MLModelType.SECURITY_SCORING: self.security_model
		}
		
		# Prediction history
		self.prediction_history: List[MLPrediction] = []
		
		# Training scheduler
		self._training_tasks: Dict[str, asyncio.Task] = {}
	
	async def train_all_models(self) -> Dict[str, Any]:
		"""Train all AI/ML models"""
		logging.info("Starting comprehensive AI/ML model training")
		
		training_results = {}
		
		# Get training data
		key_ids = await self._get_available_key_ids()
		
		if len(key_ids) < 50:
			logging.warning(f"Limited training data: only {len(key_ids)} keys available")
		
		# Train anomaly detection model
		try:
			anomaly_results = await self.anomaly_model.train(key_ids)
			training_results['anomaly_detection'] = anomaly_results
		except Exception as e:
			logging.error(f"Failed to train anomaly detection model: {e}")
			training_results['anomaly_detection'] = {'error': str(e)}
		
		# Train recommendation model
		try:
			recommendation_data = await self._generate_recommendation_training_data()
			recommendation_results = await self.recommendation_model.train(recommendation_data)
			training_results['key_recommendation'] = recommendation_results
		except Exception as e:
			logging.error(f"Failed to train recommendation model: {e}")
			training_results['key_recommendation'] = {'error': str(e)}
		
		# Train usage prediction model
		try:
			usage_data = await self._get_usage_time_series_data()
			usage_results = await self.usage_model.train(usage_data)
			training_results['usage_prediction'] = usage_results
		except Exception as e:
			logging.error(f"Failed to train usage prediction model: {e}")
			training_results['usage_prediction'] = {'error': str(e)}
		
		# Train security scoring model
		try:
			security_data = await self._generate_security_training_data()
			security_results = await self.security_model.train(security_data)
			training_results['security_scoring'] = security_results
		except Exception as e:
			logging.error(f"Failed to train security scoring model: {e}")
			training_results['security_scoring'] = {'error': str(e)}
		
		logging.info("AI/ML model training completed")
		return training_results
	
	async def get_comprehensive_insights(self, key_id: str) -> Dict[str, MLPrediction]:
		"""Get comprehensive AI insights for a key"""
		insights = {}
		
		# Anomaly detection
		try:
			anomaly_prediction = await self.anomaly_model.predict_anomaly(key_id)
			insights['anomaly_detection'] = anomaly_prediction
		except Exception as e:
			logging.error(f"Anomaly detection failed for key {key_id}: {e}")
		
		# Security scoring
		try:
			security_prediction = await self.security_model.calculate_security_score(key_id)
			insights['security_scoring'] = security_prediction
		except Exception as e:
			logging.error(f"Security scoring failed for key {key_id}: {e}")
		
		# Usage prediction
		try:
			usage_prediction = await self.usage_model.predict_usage(key_id)
			insights['usage_prediction'] = usage_prediction
		except Exception as e:
			logging.error(f"Usage prediction failed for key {key_id}: {e}")
		
		# Store predictions in history
		for insight in insights.values():
			self.prediction_history.append(insight)
		
		return insights
	
	async def recommend_key_configuration(self, context: Dict[str, Any]) -> MLPrediction:
		"""Get AI-powered key configuration recommendation"""
		try:
			recommendation = await self.recommendation_model.recommend_algorithm(context)
			self.prediction_history.append(recommendation)
			return recommendation
		except Exception as e:
			logging.error(f"Key recommendation failed: {e}")
			raise
	
	async def start_automated_training(self, interval_hours: int = 24):
		"""Start automated model retraining"""
		async def training_loop():
			while True:
				try:
					await asyncio.sleep(interval_hours * 3600)  # Convert hours to seconds
					logging.info("Starting scheduled model retraining")
					await self.train_all_models()
				except asyncio.CancelledError:
					break
				except Exception as e:
					logging.error(f"Automated training failed: {e}")
		
		task = asyncio.create_task(training_loop())
		self._training_tasks['automated_training'] = task
		
		logging.info(f"Started automated training with {interval_hours}h interval")
	
	async def stop_automated_training(self):
		"""Stop automated model retraining"""
		for task_name, task in self._training_tasks.items():
			task.cancel()
			try:
				await task
			except asyncio.CancelledError:
				pass
		
		self._training_tasks.clear()
		logging.info("Stopped automated training")
	
	async def _get_available_key_ids(self) -> List[str]:
		"""Get list of available key IDs for training"""
		try:
			if hasattr(self.service, '_db_pool') and self.service._db_pool:
				async with self.service._db_pool.acquire() as conn:
					result = await conn.fetch("SELECT id FROM km_keys WHERE status = 'active'")
					return [row['id'] for row in result]
		except Exception as e:
			logging.error(f"Error getting key IDs: {e}")
		
		return []
	
	async def _generate_recommendation_training_data(self) -> List[Dict[str, Any]]:
		"""Generate training data for key recommendation model"""
		# This would normally come from historical usage data and expert labels
		# For now, generate synthetic training data
		training_data = []
		
		use_cases = ['general', 'document', 'database', 'communication', 'iot']
		algorithms = ['AES_256', 'RSA_2048', 'ECDSA_P256']
		
		for _ in range(100):
			use_case = np.random.choice(use_cases)
			
			# Generate realistic context based on use case
			if use_case == 'iot':
				context = {
					'data_size': np.random.randint(10, 1000),
					'operations_per_day': np.random.randint(100, 10000),
					'security_level': np.random.randint(2, 4),
					'performance_priority': np.random.uniform(0.7, 1.0),
					'use_case': use_case
				}
				recommended_algorithm = 'ECDSA_P256'
			elif use_case == 'database':
				context = {
					'data_size': np.random.randint(1000, 100000),
					'operations_per_day': np.random.randint(1000, 50000),
					'security_level': np.random.randint(3, 5),
					'performance_priority': np.random.uniform(0.5, 0.8),
					'use_case': use_case
				}
				recommended_algorithm = 'AES_256'
			else:
				context = {
					'data_size': np.random.randint(100, 10000),
					'operations_per_day': np.random.randint(10, 5000),
					'security_level': np.random.randint(2, 5),
					'performance_priority': np.random.uniform(0.3, 0.9),
					'use_case': use_case
				}
				recommended_algorithm = np.random.choice(algorithms)
			
			training_data.append({
				**context,
				'recommended_algorithm': recommended_algorithm
			})
		
		return training_data
	
	async def _get_usage_time_series_data(self) -> Dict[str, List[Tuple[datetime, int]]]:
		"""Get time series usage data for training"""
		usage_data = {}
		
		try:
			if hasattr(self.service, '_db_pool') and self.service._db_pool:
				async with self.service._db_pool.acquire() as conn:
					# Get usage data for last 30 days, grouped by hour
					result = await conn.fetch("""
						SELECT 
							key_id,
							DATE_TRUNC('hour', timestamp) as hour,
							COUNT(*) as usage_count
						FROM km_audit_log 
						WHERE action IN ('encrypt', 'decrypt')
						AND timestamp > $1
						GROUP BY key_id, DATE_TRUNC('hour', timestamp)
						ORDER BY key_id, hour
					""", datetime.utcnow() - timedelta(days=30))
					
					for row in result:
						key_id = row['key_id']
						if key_id not in usage_data:
							usage_data[key_id] = []
						
						usage_data[key_id].append((row['hour'], row['usage_count']))
		
		except Exception as e:
			logging.error(f"Error getting usage time series data: {e}")
		
		return usage_data
	
	async def _generate_security_training_data(self) -> List[Dict[str, Any]]:
		"""Generate training data for security scoring model"""
		# This would normally come from security expert assessments
		# For now, generate synthetic training data
		key_ids = await self._get_available_key_ids()
		training_data = []
		
		for key_id in key_ids[:100]:  # Limit to 100 keys for demo
			# Generate synthetic security score based on key characteristics
			features = await self.feature_extractor.extract_key_features(key_id)
			
			# Simple heuristic for security score
			security_score = 0.8  # Base score
			
			if features.get('security_incidents_count', 0) > 0:
				security_score -= 0.3
			
			if features.get('failed_access_attempts', 0) > 5:
				security_score -= 0.2
			
			if features.get('key_age_days', 0) > 365:
				security_score -= 0.1
			
			if features.get('unique_users_count', 0) > 10:
				security_score -= 0.1
			
			security_score = max(0.1, min(0.9, security_score))
			
			training_data.append({
				'key_id': key_id,
				'security_score': security_score
			})
		
		return training_data


# Factory function
async def create_aiml_system(service: KeyManagementService) -> AIMLManager:
	"""Create and initialize AI/ML system"""
	aiml_manager = AIMLManager(service)
	
	# Start automated training
	await aiml_manager.start_automated_training(interval_hours=24)
	
	logging.info("AI/ML system initialized")
	return aiml_manager


# Export main components
__all__ = [
	'AIMLManager', 'FeatureExtractor', 'AnomalyDetectionModel',
	'KeyRecommendationModel', 'UsagePredictionModel', 'SecurityScoringModel',
	'MLPrediction', 'TrainingConfig',
	'MLModelType', 'PredictionConfidence',
	'create_aiml_system'
]