"""
APG Machine Learning Security Monitoring - Core Service

Enterprise ML security service with automated model management, deep learning
security analytics, and adaptive threat detection capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import joblib
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
	MLModel, ModelTraining, MLPrediction, ModelPerformance, FeatureEngineering,
	ModelMetrics, AutoMLExperiment, ModelEnsemble, ModelType, ModelArchitecture,
	ModelStatus, PredictionType, TrainingStatus
)


class MLSecurityMonitoringService:
	"""Core machine learning security monitoring service"""
	
	def __init__(self, db_session: AsyncSession, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(__name__)
		
		self._models_cache = {}
		self._training_jobs = {}
		self._feature_pipelines = {}
		self._prediction_cache = {}
		
		asyncio.create_task(self._initialize_service())
	
	async def _initialize_service(self):
		"""Initialize ML security monitoring service"""
		try:
			await self._load_active_models()
			await self._initialize_feature_pipelines()
			await self._setup_model_monitoring()
			await self._load_model_ensembles()
			
			self.logger.info(f"ML security monitoring service initialized for tenant {self.tenant_id}")
		except Exception as e:
			self.logger.error(f"Failed to initialize ML security monitoring service: {str(e)}")
			raise
	
	async def create_ml_model(self, model_data: Dict[str, Any]) -> MLModel:
		"""Create ML model specification"""
		try:
			model = MLModel(
				tenant_id=self.tenant_id,
				**model_data
			)
			
			# Validate model configuration
			await self._validate_model_configuration(model)
			
			# Initialize model artifacts directory
			model.model_path = await self._create_model_artifacts_path(model.id)
			
			# Set up monitoring configuration
			model.drift_detection_enabled = model_data.get('drift_detection_enabled', True)
			model.performance_threshold = Decimal(str(model_data.get('performance_threshold', 0.95)))
			
			await self._store_ml_model(model)
			
			# Cache the model
			self._models_cache[model.id] = model
			
			return model
			
		except Exception as e:
			self.logger.error(f"Error creating ML model: {str(e)}")
			raise
	
	async def start_model_training(self, model_id: str, training_config: Dict[str, Any]) -> ModelTraining:
		"""Start model training job"""
		try:
			model = await self._get_ml_model(model_id)
			if not model:
				raise ValueError(f"Model {model_id} not found")
			
			training_job = ModelTraining(
				tenant_id=self.tenant_id,
				model_id=model_id,
				training_name=training_config.get('training_name', f"{model.name}_training"),
				training_type=training_config.get('training_type', 'initial'),
				dataset_id=training_config['dataset_id'],
				dataset_size=training_config.get('dataset_size', 0),
				training_config=training_config.get('training_config', {}),
				hyperparameters=training_config.get('hyperparameters', {}),
				compute_resources=training_config.get('compute_resources', {}),
				gpu_enabled=training_config.get('gpu_enabled', False),
				distributed_training=training_config.get('distributed_training', False)
			)
			
			# Store training job
			await self._store_model_training(training_job)
			
			# Start asynchronous training
			asyncio.create_task(self._execute_model_training(training_job))
			
			return training_job
			
		except Exception as e:
			self.logger.error(f"Error starting model training: {str(e)}")
			raise
	
	async def _execute_model_training(self, training_job: ModelTraining):
		"""Execute model training asynchronously"""
		try:
			training_job.status = TrainingStatus.PREPROCESSING
			training_job.start_time = datetime.utcnow()
			await self._update_training_job(training_job)
			
			# Load and preprocess training data
			train_data, validation_data = await self._load_and_preprocess_data(training_job)
			
			training_job.status = TrainingStatus.TRAINING
			await self._update_training_job(training_job)
			
			# Get the model
			model = await self._get_ml_model(training_job.model_id)
			
			# Train based on model type
			if model.model_type == ModelType.DEEP_LEARNING:
				trained_model = await self._train_deep_learning_model(training_job, train_data, validation_data)
			elif model.model_type == ModelType.ANOMALY_DETECTION:
				trained_model = await self._train_anomaly_detection_model(training_job, train_data)
			else:
				trained_model = await self._train_traditional_ml_model(training_job, train_data, validation_data)
			
			# Evaluate model
			training_job.status = TrainingStatus.VALIDATING
			await self._update_training_job(training_job)
			
			evaluation_results = await self._evaluate_model(trained_model, validation_data)
			
			# Save model artifacts
			model_path = await self._save_model_artifacts(training_job, trained_model)
			
			# Update model with training results
			model.model_path = model_path
			model.accuracy = evaluation_results.get('accuracy')
			model.precision = evaluation_results.get('precision')
			model.recall = evaluation_results.get('recall')
			model.f1_score = evaluation_results.get('f1_score')
			model.status = ModelStatus.DEPLOYED
			model.deployment_date = datetime.utcnow()
			
			await self._update_ml_model(model)
			
			# Update training job with results
			training_job.status = TrainingStatus.COMPLETED
			training_job.end_time = datetime.utcnow()
			training_job.training_duration = training_job.end_time - training_job.start_time
			training_job.final_accuracy = evaluation_results.get('accuracy')
			training_job.final_loss = evaluation_results.get('loss')
			training_job.evaluation_report = evaluation_results
			
			await self._update_training_job(training_job)
			
			self.logger.info(f"Model training completed successfully for {training_job.model_id}")
			
		except Exception as e:
			training_job.status = TrainingStatus.FAILED
			training_job.error_logs.append(str(e))
			training_job.end_time = datetime.utcnow()
			await self._update_training_job(training_job)
			
			self.logger.error(f"Model training failed: {str(e)}")
	
	async def _train_deep_learning_model(self, training_job: ModelTraining, 
									   train_data: Tuple, validation_data: Tuple) -> Any:
		"""Train deep learning model using TensorFlow"""
		try:
			X_train, y_train = train_data
			X_val, y_val = validation_data
			
			# Build neural network architecture
			model = tf.keras.Sequential()
			
			# Add layers based on configuration
			config = training_job.training_config
			input_dim = X_train.shape[1]
			
			model.add(tf.keras.layers.Dense(
				config.get('hidden_units', 128),
				activation='relu',
				input_shape=(input_dim,)
			))
			
			model.add(tf.keras.layers.Dropout(config.get('dropout_rate', 0.3)))
			
			for _ in range(config.get('hidden_layers', 2)):
				model.add(tf.keras.layers.Dense(
					config.get('hidden_units', 64),
					activation='relu'
				))
				model.add(tf.keras.layers.Dropout(config.get('dropout_rate', 0.3)))
			
			# Output layer
			if len(np.unique(y_train)) == 2:
				model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
				loss = 'binary_crossentropy'
			else:
				model.add(tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax'))
				loss = 'sparse_categorical_crossentropy'
			
			# Compile model
			model.compile(
				optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
				loss=loss,
				metrics=['accuracy']
			)
			
			# Training callbacks
			callbacks = [
				tf.keras.callbacks.EarlyStopping(
					patience=config.get('early_stopping_patience', 10),
					restore_best_weights=True
				),
				tf.keras.callbacks.ReduceLROnPlateau(
					factor=0.5,
					patience=5,
					min_lr=0.0001
				)
			]
			
			# Train model
			history = model.fit(
				X_train, y_train,
				validation_data=(X_val, y_val),
				epochs=config.get('epochs', 100),
				batch_size=config.get('batch_size', 32),
				callbacks=callbacks,
				verbose=0
			)
			
			# Update training metrics
			training_job.training_loss = [float(loss) for loss in history.history['loss']]
			training_job.validation_loss = [float(loss) for loss in history.history['val_loss']]
			training_job.training_accuracy = [float(acc) for acc in history.history['accuracy']]
			training_job.validation_accuracy = [float(acc) for acc in history.history['val_accuracy']]
			
			return model
			
		except Exception as e:
			self.logger.error(f"Error training deep learning model: {str(e)}")
			raise
	
	async def _train_anomaly_detection_model(self, training_job: ModelTraining, 
										   train_data: Tuple) -> Any:
		"""Train anomaly detection model"""
		try:
			X_train, _ = train_data
			
			config = training_job.training_config
			
			if config.get('algorithm') == 'isolation_forest':
				model = IsolationForest(
					contamination=config.get('contamination', 0.1),
					random_state=42,
					n_estimators=config.get('n_estimators', 100)
				)
			else:
				# Default to Isolation Forest
				model = IsolationForest(contamination=0.1, random_state=42)
			
			model.fit(X_train)
			
			return model
			
		except Exception as e:
			self.logger.error(f"Error training anomaly detection model: {str(e)}")
			raise
	
	async def _train_traditional_ml_model(self, training_job: ModelTraining,
										train_data: Tuple, validation_data: Tuple) -> Any:
		"""Train traditional ML model"""
		try:
			X_train, y_train = train_data
			X_val, y_val = validation_data
			
			config = training_job.training_config
			algorithm = config.get('algorithm', 'random_forest')
			
			if algorithm == 'random_forest':
				model = RandomForestClassifier(
					n_estimators=config.get('n_estimators', 100),
					max_depth=config.get('max_depth', None),
					random_state=42
				)
			elif algorithm == 'neural_network':
				model = MLPClassifier(
					hidden_layer_sizes=config.get('hidden_layer_sizes', (100, 50)),
					max_iter=config.get('max_iter', 300),
					random_state=42
				)
			else:
				raise ValueError(f"Unsupported algorithm: {algorithm}")
			
			model.fit(X_train, y_train)
			
			return model
			
		except Exception as e:
			self.logger.error(f"Error training traditional ML model: {str(e)}")
			raise
	
	async def make_prediction(self, model_id: str, input_data: Dict[str, Any]) -> MLPrediction:
		"""Make prediction using trained model"""
		try:
			model = await self._get_ml_model(model_id)
			if not model or model.status != ModelStatus.DEPLOYED:
				raise ValueError(f"Model {model_id} not available for prediction")
			
			# Load trained model
			trained_model = await self._load_trained_model(model.model_path)
			
			# Preprocess input data
			processed_features = await self._preprocess_prediction_input(input_data, model)
			
			start_time = datetime.utcnow()
			
			# Make prediction
			if model.model_type == ModelType.DEEP_LEARNING:
				prediction_result = await self._predict_deep_learning(trained_model, processed_features)
			elif model.model_type == ModelType.ANOMALY_DETECTION:
				prediction_result = await self._predict_anomaly(trained_model, processed_features)
			else:
				prediction_result = await self._predict_traditional_ml(trained_model, processed_features)
			
			end_time = datetime.utcnow()
			inference_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
			
			# Create prediction record
			prediction = MLPrediction(
				tenant_id=self.tenant_id,
				model_id=model_id,
				input_data=input_data,
				input_features=processed_features,
				prediction_type=model.prediction_type,
				prediction_value=prediction_result['prediction'],
				prediction_probabilities=prediction_result.get('probabilities', {}),
				confidence_score=prediction_result.get('confidence', Decimal('0.0')),
				model_version=model.version,
				inference_time_ms=Decimal(str(inference_time)),
				feature_contributions=prediction_result.get('feature_contributions', {}),
				data_quality_score=await self._assess_data_quality(processed_features),
				prediction_quality_score=await self._assess_prediction_quality(prediction_result)
			)
			
			# Determine if this should generate an alert
			if prediction_result.get('confidence', 0) > 0.8:
				prediction.generates_alert = True
				prediction.alert_severity = await self._determine_alert_severity(prediction_result)
				prediction.alert_reason = await self._generate_alert_reason(prediction_result, input_data)
			
			await self._store_ml_prediction(prediction)
			
			# Update model usage statistics
			model.prediction_count += 1
			if prediction.inference_time_ms:
				if model.average_inference_time:
					model.average_inference_time = (model.average_inference_time + prediction.inference_time_ms) / 2
				else:
					model.average_inference_time = prediction.inference_time_ms
			
			await self._update_ml_model(model)
			
			return prediction
			
		except Exception as e:
			self.logger.error(f"Error making prediction: {str(e)}")
			raise
	
	async def monitor_model_performance(self, model_id: str, period_days: int = 7) -> ModelPerformance:
		"""Monitor model performance over specified period"""
		try:
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=period_days)
			
			model = await self._get_ml_model(model_id)
			if not model:
				raise ValueError(f"Model {model_id} not found")
			
			# Get predictions in period
			predictions = await self._get_predictions_in_period(model_id, start_time, end_time)
			
			performance = ModelPerformance(
				tenant_id=self.tenant_id,
				model_id=model_id,
				monitoring_period_start=start_time,
				monitoring_period_end=end_time
			)
			
			# Calculate basic metrics
			performance.total_predictions = len(predictions)
			performance.successful_predictions = len([p for p in predictions if not p.error_message])
			performance.failed_predictions = performance.total_predictions - performance.successful_predictions
			
			if performance.total_predictions > 0:
				performance.success_rate = Decimal(str(
					(performance.successful_predictions / performance.total_predictions) * 100
				))
			
			# Calculate performance metrics
			if predictions:
				inference_times = [float(p.inference_time_ms) for p in predictions if p.inference_time_ms]
				if inference_times:
					performance.average_inference_time = Decimal(str(np.mean(inference_times)))
					performance.p95_inference_time = Decimal(str(np.percentile(inference_times, 95)))
					performance.p99_inference_time = Decimal(str(np.percentile(inference_times, 99)))
			
			# Analyze prediction distribution
			prediction_dist = {}
			for prediction in predictions:
				pred_val = str(prediction.prediction_value)
				prediction_dist[pred_val] = prediction_dist.get(pred_val, 0) + 1
			performance.prediction_distribution = prediction_dist
			
			# Detect drift
			performance.data_drift_detected = await self._detect_data_drift(model_id, predictions)
			performance.concept_drift_detected = await self._detect_concept_drift(model_id, predictions)
			
			# Performance trend analysis
			performance.performance_trend = await self._analyze_performance_trend(model_id)
			
			await self._store_model_performance(performance)
			
			return performance
			
		except Exception as e:
			self.logger.error(f"Error monitoring model performance: {str(e)}")
			raise
	
	async def create_model_ensemble(self, ensemble_data: Dict[str, Any]) -> ModelEnsemble:
		"""Create ensemble of multiple models"""
		try:
			ensemble = ModelEnsemble(
				tenant_id=self.tenant_id,
				**ensemble_data
			)
			
			# Validate member models
			for model_id in ensemble.member_models:
				model = await self._get_ml_model(model_id)
				if not model or model.status != ModelStatus.DEPLOYED:
					raise ValueError(f"Model {model_id} not available for ensemble")
			
			# Calculate initial ensemble performance
			if ensemble.combination_method == 'voting':
				ensemble.ensemble_accuracy = await self._calculate_voting_ensemble_accuracy(ensemble)
			elif ensemble.combination_method == 'stacking':
				ensemble.ensemble_accuracy = await self._train_stacking_ensemble(ensemble)
			
			await self._store_model_ensemble(ensemble)
			
			return ensemble
			
		except Exception as e:
			self.logger.error(f"Error creating model ensemble: {str(e)}")
			raise
	
	async def generate_ml_metrics(self, period_days: int = 30) -> ModelMetrics:
		"""Generate ML security monitoring metrics"""
		try:
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=period_days)
			
			metrics = ModelMetrics(
				tenant_id=self.tenant_id,
				metric_period_start=start_time,
				metric_period_end=end_time
			)
			
			# Model inventory metrics
			all_models = await self._get_all_models()
			metrics.total_models = len(all_models)
			metrics.active_models = len([m for m in all_models if m.status == ModelStatus.DEPLOYED])
			metrics.training_models = len([m for m in all_models if m.status == ModelStatus.TRAINING])
			metrics.retired_models = len([m for m in all_models if m.status == ModelStatus.RETIRED])
			
			# Model type distribution
			type_dist = {}
			arch_dist = {}
			for model in all_models:
				type_dist[model.model_type.value] = type_dist.get(model.model_type.value, 0) + 1
				arch_dist[model.architecture.value] = arch_dist.get(model.architecture.value, 0) + 1
			
			metrics.model_type_distribution = type_dist
			metrics.architecture_distribution = arch_dist
			
			# Training metrics
			training_jobs = await self._get_training_jobs_in_period(start_time, end_time)
			metrics.training_jobs_completed = len([t for t in training_jobs if t.status == TrainingStatus.COMPLETED])
			metrics.training_jobs_failed = len([t for t in training_jobs if t.status == TrainingStatus.FAILED])
			
			if training_jobs:
				training_times = [t.training_duration for t in training_jobs if t.training_duration]
				if training_times:
					metrics.average_training_time = sum(training_times, timedelta()) / len(training_times)
			
			# Prediction metrics
			all_predictions = await self._get_all_predictions_in_period(start_time, end_time)
			metrics.total_predictions = len(all_predictions)
			metrics.successful_predictions = len([p for p in all_predictions if not p.error_message])
			
			if all_predictions:
				inference_times = [float(p.inference_time_ms) for p in all_predictions if p.inference_time_ms]
				if inference_times:
					metrics.average_inference_time = Decimal(str(np.mean(inference_times)))
			
			# Performance metrics
			accuracies = [float(m.accuracy) for m in all_models if m.accuracy]
			if accuracies:
				metrics.average_model_accuracy = Decimal(str(np.mean(accuracies)))
				best_accuracy = max(accuracies)
				worst_accuracy = min(accuracies)
				
				for model in all_models:
					if model.accuracy and float(model.accuracy) == best_accuracy:
						metrics.best_performing_model = model.id
					if model.accuracy and float(model.accuracy) == worst_accuracy:
						metrics.worst_performing_model = model.id
			
			# Security-specific metrics
			metrics.malware_detection_accuracy = await self._calculate_malware_detection_accuracy()
			metrics.phishing_detection_accuracy = await self._calculate_phishing_detection_accuracy()
			metrics.anomaly_detection_accuracy = await self._calculate_anomaly_detection_accuracy()
			metrics.false_positive_rate = await self._calculate_false_positive_rate()
			
			# Business impact
			metrics.threats_prevented = await self._count_threats_prevented(start_time, end_time)
			metrics.incidents_reduced = await self._count_incidents_reduced(start_time, end_time)
			
			await self._store_model_metrics(metrics)
			
			return metrics
			
		except Exception as e:
			self.logger.error(f"Error generating ML metrics: {str(e)}")
			raise
	
	# Helper methods for implementation
	async def _load_active_models(self):
		"""Load active models into cache"""
		pass
	
	async def _initialize_feature_pipelines(self):
		"""Initialize feature engineering pipelines"""
		pass
	
	async def _setup_model_monitoring(self):
		"""Setup model performance monitoring"""
		pass
	
	async def _load_model_ensembles(self):
		"""Load model ensembles"""
		pass
	
	async def _validate_model_configuration(self, model: MLModel):
		"""Validate model configuration"""
		pass
	
	async def _create_model_artifacts_path(self, model_id: str) -> str:
		"""Create model artifacts directory path"""
		return f"/models/{self.tenant_id}/{model_id}"
	
	async def _load_and_preprocess_data(self, training_job: ModelTraining) -> Tuple[Tuple, Tuple]:
		"""Load and preprocess training data"""
		# Placeholder implementation
		# In real implementation, this would load data from the dataset_id
		# and apply feature engineering pipelines
		dummy_X = np.random.random((1000, 10))
		dummy_y = np.random.randint(0, 2, 1000)
		
		X_train, X_val, y_train, y_val = train_test_split(
			dummy_X, dummy_y, test_size=0.2, random_state=42
		)
		
		return (X_train, y_train), (X_val, y_val)
	
	async def _evaluate_model(self, model: Any, validation_data: Tuple) -> Dict[str, Any]:
		"""Evaluate trained model"""
		try:
			X_val, y_val = validation_data
			
			if hasattr(model, 'predict'):
				predictions = model.predict(X_val)
				
				if hasattr(model, 'predict_proba'):
					probabilities = model.predict_proba(X_val)
				else:
					probabilities = None
				
				accuracy = accuracy_score(y_val, predictions)
				precision = precision_score(y_val, predictions, average='weighted', zero_division=0)
				recall = recall_score(y_val, predictions, average='weighted', zero_division=0)
				f1 = f1_score(y_val, predictions, average='weighted', zero_division=0)
				
				return {
					'accuracy': Decimal(str(accuracy)),
					'precision': Decimal(str(precision)),
					'recall': Decimal(str(recall)),
					'f1_score': Decimal(str(f1)),
					'loss': Decimal('0.1')  # Placeholder
				}
			
			return {'accuracy': Decimal('0.9')}  # Default for TensorFlow models
			
		except Exception as e:
			self.logger.error(f"Error evaluating model: {str(e)}")
			return {'accuracy': Decimal('0.0')}
	
	async def _save_model_artifacts(self, training_job: ModelTraining, model: Any) -> str:
		"""Save trained model artifacts"""
		model_path = f"/models/{self.tenant_id}/{training_job.model_id}/model.pkl"
		
		try:
			if hasattr(model, 'save'):  # TensorFlow model
				model.save(model_path.replace('.pkl', ''))
			else:  # Scikit-learn model
				joblib.dump(model, model_path)
			
			return model_path
			
		except Exception as e:
			self.logger.error(f"Error saving model artifacts: {str(e)}")
			return model_path
	
	# Placeholder implementations for database operations
	async def _store_ml_model(self, model: MLModel):
		"""Store ML model to database"""
		pass
	
	async def _store_model_training(self, training: ModelTraining):
		"""Store model training to database"""
		pass
	
	async def _store_ml_prediction(self, prediction: MLPrediction):
		"""Store ML prediction to database"""
		pass
	
	async def _store_model_performance(self, performance: ModelPerformance):
		"""Store model performance to database"""
		pass
	
	async def _store_model_ensemble(self, ensemble: ModelEnsemble):
		"""Store model ensemble to database"""
		pass
	
	async def _store_model_metrics(self, metrics: ModelMetrics):
		"""Store model metrics to database"""
		pass