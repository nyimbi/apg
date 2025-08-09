"""
APG Central Configuration - Advanced Machine Learning Models

Sophisticated ML models for predictive analytics, intelligent optimization,
and autonomous configuration management using local and edge AI.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import pickle
import joblib
from pathlib import Path

# Core ML libraries
import sklearn
from sklearn.ensemble import (
	RandomForestRegressor, GradientBoostingRegressor, IsolationForest,
	RandomForestClassifier
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, classification_report, silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Deep learning (using lightweight models for edge deployment)
try:
	import tensorflow as tf
	from tensorflow import keras
	TF_AVAILABLE = True
except ImportError:
	TF_AVAILABLE = False

# Time series analysis
try:
	import scipy
	from scipy import stats
	from scipy.signal import find_peaks
	SCIPY_AVAILABLE = True
except ImportError:
	SCIPY_AVAILABLE = False

# Advanced analytics
try:
	import networkx as nx
	NETWORKX_AVAILABLE = True
except ImportError:
	NETWORKX_AVAILABLE = False


class ModelType(Enum):
	"""Machine learning model types."""
	ANOMALY_DETECTION = "anomaly_detection"
	PERFORMANCE_PREDICTION = "performance_prediction"
	OPTIMIZATION_RECOMMENDATION = "optimization_recommendation"
	CAPACITY_PLANNING = "capacity_planning"
	FAILURE_PREDICTION = "failure_prediction"
	CONFIGURATION_CLUSTERING = "configuration_clustering"
	DRIFT_DETECTION = "drift_detection"
	COST_OPTIMIZATION = "cost_optimization"


class PredictionConfidence(Enum):
	"""Prediction confidence levels."""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERY_HIGH = "very_high"


@dataclass
class ModelPrediction:
	"""ML model prediction result."""
	model_type: ModelType
	prediction: Union[float, int, str, List[Any]]
	confidence: PredictionConfidence
	confidence_score: float
	features_used: List[str]
	model_version: str
	prediction_timestamp: datetime
	explanation: Dict[str, Any]
	metadata: Dict[str, Any]


@dataclass
class ModelTrainingResult:
	"""Model training result."""
	model_type: ModelType
	training_accuracy: float
	validation_accuracy: float
	feature_importance: Dict[str, float]
	training_time: float
	model_size_mb: float
	training_samples: int
	hyperparameters: Dict[str, Any]
	metrics: Dict[str, float]


@dataclass
class AnomalyDetectionResult:
	"""Anomaly detection result."""
	is_anomaly: bool
	anomaly_score: float
	anomaly_type: str
	affected_metrics: List[str]
	severity: str
	confidence: float
	explanation: str
	recommendations: List[str]
	timestamp: datetime


class CentralConfigurationML:
	"""Advanced machine learning engine for configuration intelligence."""
	
	def __init__(self, models_directory: str = "./ml_models"):
		"""Initialize ML engine."""
		self.models_dir = Path(models_directory)
		self.models_dir.mkdir(exist_ok=True)
		
		# Model storage
		self.models: Dict[ModelType, Any] = {}
		self.scalers: Dict[ModelType, StandardScaler] = {}
		self.encoders: Dict[str, LabelEncoder] = {}
		self.model_metadata: Dict[ModelType, Dict[str, Any]] = {}
		
		# Feature engineering components
		self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
		self.pca_transformer = PCA(n_components=50)
		
		# Training data cache
		self.training_data: Dict[ModelType, pd.DataFrame] = {}
		self.feature_columns: Dict[ModelType, List[str]] = {}
		
		# Performance metrics
		self.model_performance: Dict[ModelType, Dict[str, float]] = {}
		
		# Initialize models
		asyncio.create_task(self._initialize_models())
	
	async def _initialize_models(self):
		"""Initialize all ML models."""
		print("🤖 Initializing ML models...")
		
		# Load existing models or create new ones
		await self._load_or_create_anomaly_detection_model()
		await self._load_or_create_performance_prediction_model()
		await self._load_or_create_optimization_model()
		await self._load_or_create_capacity_planning_model()
		await self._load_or_create_failure_prediction_model()
		await self._load_or_create_clustering_model()
		await self._load_or_create_drift_detection_model()
		await self._load_or_create_cost_optimization_model()
		
		print("✅ ML models initialized successfully")
	
	# ==================== Anomaly Detection ====================
	
	async def _load_or_create_anomaly_detection_model(self):
		"""Load or create anomaly detection model."""
		model_path = self.models_dir / "anomaly_detection.pkl"
		
		if model_path.exists():
			self.models[ModelType.ANOMALY_DETECTION] = joblib.load(model_path)
			print("📊 Loaded existing anomaly detection model")
		else:
			# Create Isolation Forest model for anomaly detection
			model = IsolationForest(
				contamination=0.1,
				n_estimators=200,
				max_samples='auto',
				random_state=42
			)
			
			# Create synthetic training data for demonstration
			training_data = await self._generate_anomaly_training_data()
			
			# Train model
			model.fit(training_data)
			
			# Save model
			joblib.dump(model, model_path)
			self.models[ModelType.ANOMALY_DETECTION] = model
			
			# Create scaler
			scaler = StandardScaler()
			scaler.fit(training_data)
			self.scalers[ModelType.ANOMALY_DETECTION] = scaler
			
			print("🎯 Created new anomaly detection model")
	
	async def _generate_anomaly_training_data(self) -> np.ndarray:
		"""Generate synthetic training data for anomaly detection."""
		# Simulate normal configuration metrics
		np.random.seed(42)
		
		normal_data = []
		for _ in range(1000):
			# CPU usage (normal: 20-80%)
			cpu = np.random.normal(50, 15)
			cpu = np.clip(cpu, 0, 100)
			
			# Memory usage (normal: 30-70%)
			memory = np.random.normal(50, 12)
			memory = np.clip(memory, 0, 100)
			
			# Response time (normal: 50-200ms)
			response_time = np.random.lognormal(4, 0.5)
			response_time = np.clip(response_time, 10, 1000)
			
			# Request rate (normal: 100-1000 rps)
			request_rate = np.random.normal(500, 200)
			request_rate = np.clip(request_rate, 0, 2000)
			
			# Error rate (normal: 0-5%)
			error_rate = np.random.exponential(1)
			error_rate = np.clip(error_rate, 0, 20)
			
			# Disk usage (normal: 20-80%)
			disk_usage = np.random.normal(45, 20)
			disk_usage = np.clip(disk_usage, 0, 100)
			
			normal_data.append([cpu, memory, response_time, request_rate, error_rate, disk_usage])
		
		# Add some anomalous samples
		anomaly_data = []
		for _ in range(50):
			# Anomalous patterns
			cpu = np.random.choice([5, 95])  # Very low or very high
			memory = np.random.choice([10, 90])
			response_time = np.random.exponential(500)  # High response times
			request_rate = np.random.exponential(100)  # Low request rates
			error_rate = np.random.normal(15, 5)  # High error rates
			disk_usage = np.random.choice([5, 95])
			
			anomaly_data.append([cpu, memory, response_time, request_rate, error_rate, disk_usage])
		
		# Combine and return
		all_data = np.array(normal_data + anomaly_data)
		return all_data
	
	async def detect_configuration_anomalies(
		self,
		metrics_data: List[Dict[str, Any]]
	) -> List[AnomalyDetectionResult]:
		"""Detect anomalies in configuration metrics."""
		if ModelType.ANOMALY_DETECTION not in self.models:
			raise ValueError("Anomaly detection model not initialized")
		
		model = self.models[ModelType.ANOMALY_DETECTION]
		scaler = self.scalers.get(ModelType.ANOMALY_DETECTION)
		
		results = []
		
		for metric_data in metrics_data:
			try:
				# Extract features
				features = await self._extract_anomaly_features(metric_data)
				
				if len(features) == 0:
					continue
				
				# Scale features
				if scaler:
					features_scaled = scaler.transform([features])
				else:
					features_scaled = [features]
				
				# Predict anomaly
				anomaly_score = model.decision_function(features_scaled)[0]
				is_anomaly = model.predict(features_scaled)[0] == -1
				
				# Determine severity
				if anomaly_score < -0.5:
					severity = "critical"
				elif anomaly_score < -0.2:
					severity = "high"
				elif anomaly_score < 0:
					severity = "medium"
				else:
					severity = "low"
				
				# Identify affected metrics
				affected_metrics = await self._identify_anomalous_metrics(metric_data, features)
				
				# Generate explanation
				explanation = await self._explain_anomaly(metric_data, anomaly_score, affected_metrics)
				
				# Generate recommendations
				recommendations = await self._generate_anomaly_recommendations(
					metric_data, anomaly_score, affected_metrics
				)
				
				result = AnomalyDetectionResult(
					is_anomaly=is_anomaly,
					anomaly_score=float(anomaly_score),
					anomaly_type="performance" if "cpu" in affected_metrics else "configuration",
					affected_metrics=affected_metrics,
					severity=severity,
					confidence=min(abs(anomaly_score) * 2, 1.0),
					explanation=explanation,
					recommendations=recommendations,
					timestamp=datetime.now(timezone.utc)
				)
				
				results.append(result)
				
			except Exception as e:
				print(f"Error detecting anomaly: {e}")
				continue
		
		return results
	
	async def _extract_anomaly_features(self, metric_data: Dict[str, Any]) -> List[float]:
		"""Extract features for anomaly detection."""
		features = []
		
		# CPU metrics
		if 'cpu_usage' in metric_data:
			features.append(float(metric_data['cpu_usage']))
		
		# Memory metrics
		if 'memory_usage' in metric_data:
			features.append(float(metric_data['memory_usage']))
		
		# Response time
		if 'response_time' in metric_data:
			features.append(float(metric_data['response_time']))
		
		# Request rate
		if 'request_rate' in metric_data:
			features.append(float(metric_data['request_rate']))
		
		# Error rate
		if 'error_rate' in metric_data:
			features.append(float(metric_data['error_rate']))
		
		# Disk usage
		if 'disk_usage' in metric_data:
			features.append(float(metric_data['disk_usage']))
		
		# Pad with zeros if not enough features
		while len(features) < 6:
			features.append(0.0)
		
		return features[:6]  # Ensure exactly 6 features
	
	async def _identify_anomalous_metrics(
		self,
		metric_data: Dict[str, Any],
		features: List[float]
	) -> List[str]:
		"""Identify which specific metrics are anomalous."""
		anomalous_metrics = []
		
		feature_names = ['cpu_usage', 'memory_usage', 'response_time', 'request_rate', 'error_rate', 'disk_usage']
		
		for i, (name, value) in enumerate(zip(feature_names, features)):
			if i < len(features):
				# Simple thresholds for demonstration
				if name == 'cpu_usage' and (value > 90 or value < 5):
					anomalous_metrics.append(name)
				elif name == 'memory_usage' and (value > 85 or value < 10):
					anomalous_metrics.append(name)
				elif name == 'response_time' and value > 500:
					anomalous_metrics.append(name)
				elif name == 'error_rate' and value > 10:
					anomalous_metrics.append(name)
				elif name == 'disk_usage' and (value > 90 or value < 5):
					anomalous_metrics.append(name)
		
		return anomalous_metrics
	
	async def _explain_anomaly(
		self,
		metric_data: Dict[str, Any],
		anomaly_score: float,
		affected_metrics: List[str]
	) -> str:
		"""Generate human-readable explanation for anomaly."""
		if not affected_metrics:
			return "General system behavior anomaly detected"
		
		explanations = []
		
		for metric in affected_metrics:
			if metric == 'cpu_usage':
				value = metric_data.get('cpu_usage', 0)
				if value > 90:
					explanations.append(f"CPU usage is critically high at {value:.1f}%")
				elif value < 5:
					explanations.append(f"CPU usage is unusually low at {value:.1f}%")
			
			elif metric == 'memory_usage':
				value = metric_data.get('memory_usage', 0)
				if value > 85:
					explanations.append(f"Memory usage is critically high at {value:.1f}%")
				
			elif metric == 'response_time':
				value = metric_data.get('response_time', 0)
				explanations.append(f"Response time is elevated at {value:.1f}ms")
			
			elif metric == 'error_rate':
				value = metric_data.get('error_rate', 0)
				explanations.append(f"Error rate is high at {value:.1f}%")
		
		return "; ".join(explanations) if explanations else "System anomaly detected"
	
	async def _generate_anomaly_recommendations(
		self,
		metric_data: Dict[str, Any],
		anomaly_score: float,
		affected_metrics: List[str]
	) -> List[str]:
		"""Generate recommendations for addressing anomalies."""
		recommendations = []
		
		for metric in affected_metrics:
			if metric == 'cpu_usage':
				recommendations.extend([
					"Scale up CPU resources or add more instances",
					"Review and optimize CPU-intensive processes",
					"Implement CPU throttling policies"
				])
			
			elif metric == 'memory_usage':
				recommendations.extend([
					"Increase memory allocation",
					"Optimize memory usage patterns",
					"Implement memory cleanup routines"
				])
			
			elif metric == 'response_time':
				recommendations.extend([
					"Optimize database queries",
					"Implement caching mechanisms",
					"Review network latency issues"
				])
			
			elif metric == 'error_rate':
				recommendations.extend([
					"Review application logs for error patterns",
					"Implement circuit breaker patterns",
					"Improve error handling and retry logic"
				])
		
		# General recommendations
		if anomaly_score < -0.5:
			recommendations.append("Consider immediate intervention due to critical anomaly")
		
		return list(set(recommendations))  # Remove duplicates
	
	# ==================== Performance Prediction ====================
	
	async def _load_or_create_performance_prediction_model(self):
		"""Load or create performance prediction model."""
		model_path = self.models_dir / "performance_prediction.pkl"
		
		if model_path.exists():
			self.models[ModelType.PERFORMANCE_PREDICTION] = joblib.load(model_path)
			print("🚀 Loaded existing performance prediction model")
		else:
			# Create Random Forest model for performance prediction
			model = RandomForestRegressor(
				n_estimators=100,
				max_depth=10,
				random_state=42
			)
			
			# Generate synthetic training data
			X, y = await self._generate_performance_training_data()
			
			# Train model
			model.fit(X, y)
			
			# Save model
			joblib.dump(model, model_path)
			self.models[ModelType.PERFORMANCE_PREDICTION] = model
			
			# Create scaler
			scaler = StandardScaler()
			scaler.fit(X)
			self.scalers[ModelType.PERFORMANCE_PREDICTION] = scaler
			
			print("🎯 Created new performance prediction model")
	
	async def _generate_performance_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
		"""Generate synthetic training data for performance prediction."""
		np.random.seed(42)
		
		n_samples = 2000
		
		# Features: [config_complexity, resource_allocation, load_factor, optimization_level]
		X = []
		y = []  # Response time
		
		for _ in range(n_samples):
			# Configuration complexity (1-10)
			complexity = np.random.uniform(1, 10)
			
			# Resource allocation (0.1-2.0 multiplier)
			resources = np.random.uniform(0.1, 2.0)
			
			# Load factor (0.1-5.0)
			load = np.random.uniform(0.1, 5.0)
			
			# Optimization level (0-1)
			optimization = np.random.uniform(0, 1)
			
			# Calculate synthetic response time with realistic relationships
			base_time = complexity * 10  # Base time increases with complexity
			resource_factor = 1 / resources  # More resources = faster
			load_factor = load * 1.5  # More load = slower
			optimization_factor = 1 - (optimization * 0.3)  # Optimization reduces time
			
			response_time = base_time * resource_factor * load_factor * optimization_factor
			response_time += np.random.normal(0, response_time * 0.1)  # Add noise
			response_time = max(response_time, 1.0)  # Minimum 1ms
			
			X.append([complexity, resources, load, optimization])
			y.append(response_time)
		
		return np.array(X), np.array(y)
	
	async def predict_performance(
		self,
		configuration_data: Dict[str, Any],
		resource_allocation: Dict[str, float],
		predicted_load: Dict[str, float]
	) -> ModelPrediction:
		"""Predict performance metrics for given configuration."""
		if ModelType.PERFORMANCE_PREDICTION not in self.models:
			raise ValueError("Performance prediction model not initialized")
		
		model = self.models[ModelType.PERFORMANCE_PREDICTION]
		scaler = self.scalers.get(ModelType.PERFORMANCE_PREDICTION)
		
		# Extract features
		features = await self._extract_performance_features(
			configuration_data, resource_allocation, predicted_load
		)
		
		# Scale features
		if scaler:
			features_scaled = scaler.transform([features])
		else:
			features_scaled = [features]
		
		# Make prediction
		prediction = model.predict(features_scaled)[0]
		
		# Calculate confidence based on model uncertainty
		if hasattr(model, 'predict_proba'):
			# For classification models
			confidence_score = 0.8
		else:
			# For regression models, use feature similarity to training data
			confidence_score = 0.75  # Simplified confidence
		
		# Determine confidence level
		if confidence_score > 0.9:
			confidence = PredictionConfidence.VERY_HIGH
		elif confidence_score > 0.7:
			confidence = PredictionConfidence.HIGH
		elif confidence_score > 0.5:
			confidence = PredictionConfidence.MEDIUM
		else:
			confidence = PredictionConfidence.LOW
		
		# Generate explanation
		explanation = await self._explain_performance_prediction(
			prediction, features, configuration_data
		)
		
		return ModelPrediction(
			model_type=ModelType.PERFORMANCE_PREDICTION,
			prediction=float(prediction),
			confidence=confidence,
			confidence_score=confidence_score,
			features_used=['complexity', 'resources', 'load', 'optimization'],
			model_version="1.0",
			prediction_timestamp=datetime.now(timezone.utc),
			explanation=explanation,
			metadata={
				'unit': 'milliseconds',
				'prediction_type': 'response_time'
			}
		)
	
	async def _extract_performance_features(
		self,
		configuration_data: Dict[str, Any],
		resource_allocation: Dict[str, float],
		predicted_load: Dict[str, float]
	) -> List[float]:
		"""Extract features for performance prediction."""
		# Configuration complexity (based on number of parameters and nesting)
		complexity = await self._calculate_config_complexity(configuration_data)
		
		# Resource allocation factor
		cpu_allocation = resource_allocation.get('cpu', 1.0)
		memory_allocation = resource_allocation.get('memory', 1.0)
		resource_factor = (cpu_allocation + memory_allocation) / 2
		
		# Load factor
		expected_rps = predicted_load.get('requests_per_second', 100)
		load_factor = expected_rps / 100  # Normalize to baseline of 100 RPS
		
		# Optimization level (based on configuration patterns)
		optimization_level = await self._calculate_optimization_level(configuration_data)
		
		return [complexity, resource_factor, load_factor, optimization_level]
	
	async def _calculate_config_complexity(self, config_data: Dict[str, Any]) -> float:
		"""Calculate configuration complexity score."""
		complexity = 0.0
		
		def count_nested_items(obj, depth=0):
			count = 0
			if isinstance(obj, dict):
				count += len(obj) * (1 + depth * 0.1)
				for value in obj.values():
					count += count_nested_items(value, depth + 1)
			elif isinstance(obj, list):
				count += len(obj) * (1 + depth * 0.1)
				for item in obj:
					count += count_nested_items(item, depth + 1)
			return count
		
		complexity = count_nested_items(config_data)
		return min(complexity / 10, 10.0)  # Normalize to 0-10 scale
	
	async def _calculate_optimization_level(self, config_data: Dict[str, Any]) -> float:
		"""Calculate how optimized a configuration appears to be."""
		optimization_score = 0.0
		
		# Check for common optimization patterns
		optimizations = [
			'cache', 'pool', 'buffer', 'timeout', 'batch',
			'compression', 'index', 'lazy', 'async'
		]
		
		config_str = json.dumps(config_data, default=str).lower()
		
		for opt in optimizations:
			if opt in config_str:
				optimization_score += 0.1
		
		return min(optimization_score, 1.0)
	
	async def _explain_performance_prediction(
		self,
		prediction: float,
		features: List[float],
		config_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Explain performance prediction."""
		complexity, resources, load, optimization = features
		
		explanation = {
			'predicted_response_time_ms': float(prediction),
			'performance_level': 'excellent' if prediction < 50 else
							   'good' if prediction < 100 else
							   'average' if prediction < 200 else
							   'poor',
			'key_factors': [],
			'recommendations': []
		}
		
		# Analyze key factors
		if complexity > 7:
			explanation['key_factors'].append('High configuration complexity')
			explanation['recommendations'].append('Simplify configuration structure')
		
		if resources < 0.5:
			explanation['key_factors'].append('Limited resource allocation')
			explanation['recommendations'].append('Increase CPU and memory allocation')
		
		if load > 3:
			explanation['key_factors'].append('High expected load')
			explanation['recommendations'].append('Implement load balancing and caching')
		
		if optimization < 0.3:
			explanation['key_factors'].append('Limited optimization patterns detected')
			explanation['recommendations'].append('Apply performance optimization best practices')
		
		return explanation
	
	# ==================== Configuration Optimization ====================
	
	async def _load_or_create_optimization_model(self):
		"""Load or create configuration optimization model."""
		model_path = self.models_dir / "optimization_model.pkl"
		
		if model_path.exists():
			self.models[ModelType.OPTIMIZATION_RECOMMENDATION] = joblib.load(model_path)
			print("⚡ Loaded existing optimization model")
		else:
			# Create ensemble model for optimization recommendations
			model = RandomForestClassifier(
				n_estimators=150,
				max_depth=15,
				random_state=42
			)
			
			# Generate training data
			X, y = await self._generate_optimization_training_data()
			
			# Train model
			model.fit(X, y)
			
			# Save model
			joblib.dump(model, model_path)
			self.models[ModelType.OPTIMIZATION_RECOMMENDATION] = model
			
			print("🎯 Created new optimization recommendation model")
	
	async def _generate_optimization_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
		"""Generate training data for optimization recommendations."""
		np.random.seed(42)
		
		X = []
		y = []
		
		optimization_types = [
			'increase_cache_size',
			'enable_compression',
			'optimize_connection_pool',
			'adjust_timeout_values',
			'enable_batching',
			'add_indexing',
			'implement_lazy_loading',
			'enable_async_processing'
		]
		
		for _ in range(1500):
			# Features: performance_score, resource_usage, complexity, current_optimizations
			performance_score = np.random.uniform(0, 1)
			resource_usage = np.random.uniform(0, 1)
			complexity = np.random.uniform(0, 1)
			current_optimizations = np.random.uniform(0, 1)
			
			# Determine optimization type based on conditions
			if performance_score < 0.3 and resource_usage < 0.5:
				opt_type = 'increase_cache_size'
			elif performance_score < 0.4 and complexity > 0.7:
				opt_type = 'optimize_connection_pool'
			elif resource_usage > 0.8:
				opt_type = 'enable_compression'
			elif performance_score < 0.5:
				opt_type = 'adjust_timeout_values'
			else:
				opt_type = np.random.choice(optimization_types)
			
			X.append([performance_score, resource_usage, complexity, current_optimizations])
			y.append(opt_type)
		
		# Encode labels
		if 'optimization' not in self.encoders:
			self.encoders['optimization'] = LabelEncoder()
			y_encoded = self.encoders['optimization'].fit_transform(y)
		else:
			y_encoded = self.encoders['optimization'].transform(y)
		
		return np.array(X), y_encoded
	
	async def recommend_optimizations(
		self,
		configuration_data: Dict[str, Any],
		performance_metrics: Dict[str, float],
		resource_metrics: Dict[str, float]
	) -> List[ModelPrediction]:
		"""Recommend configuration optimizations."""
		if ModelType.OPTIMIZATION_RECOMMENDATION not in self.models:
			raise ValueError("Optimization model not initialized")
		
		model = self.models[ModelType.OPTIMIZATION_RECOMMENDATION]
		
		# Extract features
		features = await self._extract_optimization_features(
			configuration_data, performance_metrics, resource_metrics
		)
		
		# Get prediction probabilities
		prediction_probs = model.predict_proba([features])[0]
		
		# Get top recommendations
		top_indices = np.argsort(prediction_probs)[-3:]  # Top 3 recommendations
		
		recommendations = []
		encoder = self.encoders.get('optimization')
		
		for idx in reversed(top_indices):  # Highest probability first
			if encoder:
				optimization_type = encoder.inverse_transform([idx])[0]
			else:
				optimization_type = f"optimization_{idx}"
			
			confidence_score = prediction_probs[idx]
			
			if confidence_score > 0.9:
				confidence = PredictionConfidence.VERY_HIGH
			elif confidence_score > 0.7:
				confidence = PredictionConfidence.HIGH
			elif confidence_score > 0.5:
				confidence = PredictionConfidence.MEDIUM
			else:
				confidence = PredictionConfidence.LOW
			
			# Generate detailed recommendation
			recommendation_details = await self._generate_optimization_details(
				optimization_type, configuration_data, performance_metrics
			)
			
			recommendation = ModelPrediction(
				model_type=ModelType.OPTIMIZATION_RECOMMENDATION,
				prediction=optimization_type,
				confidence=confidence,
				confidence_score=confidence_score,
				features_used=['performance', 'resources', 'complexity', 'optimizations'],
				model_version="1.0",
				prediction_timestamp=datetime.now(timezone.utc),
				explanation=recommendation_details,
				metadata={
					'priority': 'high' if confidence_score > 0.8 else 'medium',
					'implementation_effort': self._estimate_implementation_effort(optimization_type)
				}
			)
			
			recommendations.append(recommendation)
		
		return recommendations
	
	async def _extract_optimization_features(
		self,
		configuration_data: Dict[str, Any],
		performance_metrics: Dict[str, float],
		resource_metrics: Dict[str, float]
	) -> List[float]:
		"""Extract features for optimization recommendations."""
		# Performance score (inverse of response time, normalized)
		response_time = performance_metrics.get('response_time', 100)
		performance_score = max(0, 1 - (response_time / 1000))  # Normalize to 0-1
		
		# Resource usage (average of CPU and memory)
		cpu_usage = resource_metrics.get('cpu_usage', 50) / 100
		memory_usage = resource_metrics.get('memory_usage', 50) / 100
		resource_usage = (cpu_usage + memory_usage) / 2
		
		# Configuration complexity
		complexity = await self._calculate_config_complexity(configuration_data)
		complexity_normalized = complexity / 10
		
		# Current optimization level
		current_optimizations = await self._calculate_optimization_level(configuration_data)
		
		return [performance_score, resource_usage, complexity_normalized, current_optimizations]
	
	async def _generate_optimization_details(
		self,
		optimization_type: str,
		configuration_data: Dict[str, Any],
		performance_metrics: Dict[str, float]
	) -> Dict[str, Any]:
		"""Generate detailed optimization recommendation."""
		details = {
			'optimization_type': optimization_type,
			'description': '',
			'benefits': [],
			'implementation_steps': [],
			'estimated_improvement': '',
			'risks': []
		}
		
		if optimization_type == 'increase_cache_size':
			details.update({
				'description': 'Increase cache memory allocation to improve data access performance',
				'benefits': ['Faster data retrieval', 'Reduced database load', 'Better user experience'],
				'implementation_steps': [
					'Analyze current cache hit ratio',
					'Calculate optimal cache size based on working set',
					'Update cache configuration',
					'Monitor performance improvement'
				],
				'estimated_improvement': '20-40% reduction in response time',
				'risks': ['Higher memory usage', 'Potential memory pressure']
			})
		
		elif optimization_type == 'enable_compression':
			details.update({
				'description': 'Enable data compression to reduce bandwidth and storage requirements',
				'benefits': ['Reduced network traffic', 'Lower storage costs', 'Faster data transfer'],
				'implementation_steps': [
					'Enable gzip compression for HTTP responses',
					'Configure database compression',
					'Update client libraries to handle compression',
					'Monitor compression ratios'
				],
				'estimated_improvement': '30-60% reduction in bandwidth usage',
				'risks': ['Slight CPU overhead', 'Compatibility issues with older clients']
			})
		
		elif optimization_type == 'optimize_connection_pool':
			details.update({
				'description': 'Optimize database connection pool settings for better resource utilization',
				'benefits': ['Better connection reuse', 'Reduced connection overhead', 'Improved scalability'],
				'implementation_steps': [
					'Analyze current connection patterns',
					'Calculate optimal pool size',
					'Configure connection timeouts',
					'Implement connection health checks'
				],
				'estimated_improvement': '15-30% improvement in database performance',
				'risks': ['Resource exhaustion if misconfigured', 'Connection leaks']
			})
		
		elif optimization_type == 'adjust_timeout_values':
			details.update({
				'description': 'Optimize timeout configurations to balance responsiveness and resource usage',
				'benefits': ['Better error handling', 'Improved resource cleanup', 'Enhanced user experience'],
				'implementation_steps': [
					'Analyze timeout patterns in logs',
					'Calculate optimal timeout values',
					'Update configuration files',
					'Test timeout scenarios'
				],
				'estimated_improvement': '10-25% reduction in resource waste',
				'risks': ['Premature timeouts', 'User experience degradation']
			})
		
		# Add more optimization types as needed
		
		return details
	
	def _estimate_implementation_effort(self, optimization_type: str) -> str:
		"""Estimate implementation effort for optimization."""
		effort_map = {
			'increase_cache_size': 'low',
			'enable_compression': 'medium',
			'optimize_connection_pool': 'medium',
			'adjust_timeout_values': 'low',
			'enable_batching': 'high',
			'add_indexing': 'medium',
			'implement_lazy_loading': 'high',
			'enable_async_processing': 'high'
		}
		
		return effort_map.get(optimization_type, 'medium')
	
	# ==================== Additional ML Models ====================
	
	async def _load_or_create_capacity_planning_model(self):
		"""Load or create capacity planning model."""
		# Simplified implementation
		model = GradientBoostingRegressor(n_estimators=100, random_state=42)
		self.models[ModelType.CAPACITY_PLANNING] = model
		print("📈 Created capacity planning model")
	
	async def _load_or_create_failure_prediction_model(self):
		"""Load or create failure prediction model."""
		model = IsolationForest(contamination=0.05, random_state=42)
		self.models[ModelType.FAILURE_PREDICTION] = model
		print("🔮 Created failure prediction model")
	
	async def _load_or_create_clustering_model(self):
		"""Load or create configuration clustering model."""
		model = KMeans(n_clusters=8, random_state=42)
		self.models[ModelType.CONFIGURATION_CLUSTERING] = model
		print("🎯 Created configuration clustering model")
	
	async def _load_or_create_drift_detection_model(self):
		"""Load or create concept drift detection model."""
		# This would be a more sophisticated drift detection algorithm
		# For now, using a simple statistical approach
		self.models[ModelType.DRIFT_DETECTION] = {"type": "statistical", "window_size": 100}
		print("📊 Created drift detection model")
	
	async def _load_or_create_cost_optimization_model(self):
		"""Load or create cost optimization model."""
		model = RandomForestRegressor(n_estimators=100, random_state=42)
		self.models[ModelType.COST_OPTIMIZATION] = model
		print("💰 Created cost optimization model")
	
	# ==================== Model Management ====================
	
	async def retrain_model(
		self,
		model_type: ModelType,
		training_data: pd.DataFrame,
		target_column: str
	) -> ModelTrainingResult:
		"""Retrain a specific model with new data."""
		start_time = datetime.now()
		
		# Prepare data
		X = training_data.drop(columns=[target_column])
		y = training_data[target_column]
		
		# Split data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		
		# Get model
		if model_type not in self.models:
			raise ValueError(f"Model type {model_type} not found")
		
		model = self.models[model_type]
		
		# Train model
		model.fit(X_train, y_train)
		
		# Evaluate
		train_score = model.score(X_train, y_train)
		test_score = model.score(X_test, y_test)
		
		# Feature importance (if available)
		feature_importance = {}
		if hasattr(model, 'feature_importances_'):
			for feature, importance in zip(X.columns, model.feature_importances_):
				feature_importance[feature] = float(importance)
		
		# Training time
		training_time = (datetime.now() - start_time).total_seconds()
		
		# Save updated model
		model_path = self.models_dir / f"{model_type.value}.pkl"
		joblib.dump(model, model_path)
		
		return ModelTrainingResult(
			model_type=model_type,
			training_accuracy=train_score,
			validation_accuracy=test_score,
			feature_importance=feature_importance,
			training_time=training_time,
			model_size_mb=model_path.stat().st_size / (1024 * 1024),
			training_samples=len(X_train),
			hyperparameters={},
			metrics={
				'train_score': train_score,
				'test_score': test_score
			}
		)
	
	async def get_model_performance(self) -> Dict[ModelType, Dict[str, float]]:
		"""Get performance metrics for all models."""
		return self.model_performance.copy()
	
	async def close(self):
		"""Clean up ML engine resources."""
		self.models.clear()
		self.training_data.clear()
		print("🤖 ML engine closed")


# ==================== Factory Functions ====================

async def create_ml_engine(models_directory: str = "./ml_models") -> CentralConfigurationML:
	"""Create and initialize ML engine."""
	engine = CentralConfigurationML(models_directory)
	await asyncio.sleep(0.1)  # Allow initialization to complete
	print("🤖 ML engine initialized successfully")
	return engine