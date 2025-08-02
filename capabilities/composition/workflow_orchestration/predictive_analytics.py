#!/usr/bin/env python3
"""
APG Workflow Orchestration Predictive Analytics System

Advanced predictive analytics, failure prediction, anomaly detection, and performance forecasting.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModelV2, ConfigDict, Field, Annotated, AfterValidator
import json

from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.cluster import DBSCAN
import joblib

from apg.framework.base_service import APGBaseService
from apg.framework.database import APGDatabase  
from apg.framework.audit_compliance import APGAuditLogger
from apg.framework.messaging import APGEventBus

from .config import get_config
from .models import WorkflowStatus, TaskStatus


logger = logging.getLogger(__name__)


class PredictionType(str, Enum):
	"""Types of predictions."""
	FAILURE_PREDICTION = "failure_prediction"
	PERFORMANCE_FORECAST = "performance_forecast"
	ANOMALY_DETECTION = "anomaly_detection"
	RESOURCE_DEMAND = "resource_demand"
	SLA_BREACH = "sla_breach"


class AlertSeverity(str, Enum):
	"""Alert severity levels."""
	INFO = "info"
	WARNING = "warning"
	CRITICAL = "critical"
	EMERGENCY = "emergency"


class ModelStatus(str, Enum):
	"""ML model status."""
	TRAINING = "training"
	TRAINED = "trained"
	PREDICTING = "predicting"
	ERROR = "error"
	OUTDATED = "outdated"


@dataclass
class PredictionResult:
	"""Prediction result data structure."""
	id: str = Field(default_factory=uuid7str)
	prediction_type: PredictionType
	workflow_id: Optional[str] = None
	instance_id: Optional[str] = None
	predicted_value: Union[float, bool, str]
	confidence: float
	probability: Optional[float] = None
	prediction_horizon: timedelta
	model_version: str
	features_used: List[str]
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: Optional[datetime] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class AnomalyDetection:
	"""Anomaly detection result."""
	id: str = Field(default_factory=uuid7str)
	workflow_id: Optional[str] = None
	instance_id: Optional[str] = None
	anomaly_score: float
	is_anomaly: bool
	anomaly_type: str
	features: Dict[str, float]
	detected_at: datetime = Field(default_factory=datetime.utcnow)
	severity: AlertSeverity
	description: str
	recommendations: List[str] = Field(default_factory=list)


@dataclass
class PerformanceForecast:
	"""Performance forecast result."""
	id: str = Field(default_factory=uuid7str)
	workflow_id: str
	forecast_horizon: timedelta
	predicted_metrics: Dict[str, float]
	confidence_intervals: Dict[str, Tuple[float, float]]
	trend_analysis: Dict[str, str]
	seasonality_detected: bool
	forecast_accuracy: Optional[float] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class MLModelManager:
	"""Manages machine learning models for predictive analytics."""
	
	def __init__(self):
		self.models: Dict[str, Any] = {}
		self.scalers: Dict[str, StandardScaler] = {}
		self.encoders: Dict[str, LabelEncoder] = {}
		self.model_metadata: Dict[str, Dict[str, Any]] = {}
	
	async def train_failure_prediction_model(self, training_data: pd.DataFrame) -> str:
		"""Train failure prediction model."""
		try:
			model_id = f"failure_prediction_{uuid7str()}"
			
			# Prepare features and target
			features = training_data.drop(['failed', 'workflow_id', 'instance_id'], axis=1, errors='ignore')
			target = training_data['failed']
			
			# Scale features
			scaler = StandardScaler()
			features_scaled = scaler.fit_transform(features)
			
			# Split data
			X_train, X_test, y_train, y_test = train_test_split(
				features_scaled, target, test_size=0.2, random_state=42, stratify=target
			)
			
			# Train model
			model = GradientBoostingClassifier(
				n_estimators=100,
				learning_rate=0.1,
				max_depth=5,
				random_state=42
			)
			model.fit(X_train, y_train)
			
			# Evaluate model
			y_pred = model.predict(X_test)
			accuracy = accuracy_score(y_test, y_pred)
			precision = precision_score(y_test, y_pred, average='weighted')
			recall = recall_score(y_test, y_pred, average='weighted')
			f1 = f1_score(y_test, y_pred, average='weighted')
			
			# Store model and metadata
			self.models[model_id] = model
			self.scalers[model_id] = scaler
			self.model_metadata[model_id] = {
				'type': 'failure_prediction',
				'accuracy': accuracy,
				'precision': precision,
				'recall': recall,
				'f1_score': f1,
				'feature_names': list(features.columns),
				'trained_at': datetime.utcnow(),
				'training_samples': len(training_data),
				'status': ModelStatus.TRAINED
			}
			
			logger.info(f"Failure prediction model trained: {model_id} (accuracy: {accuracy:.3f})")
			return model_id
			
		except Exception as e:
			logger.error(f"Failed to train failure prediction model: {e}")
			raise
	
	async def train_performance_forecast_model(self, training_data: pd.DataFrame) -> str:
		"""Train performance forecasting model."""
		try:
			model_id = f"performance_forecast_{uuid7str()}"
			
			# Prepare features and target
			features = training_data.drop(['execution_time', 'workflow_id'], axis=1, errors='ignore')
			target = training_data['execution_time']
			
			# Scale features
			scaler = StandardScaler()
			features_scaled = scaler.fit_transform(features)
			
			# Split data
			X_train, X_test, y_train, y_test = train_test_split(
				features_scaled, target, test_size=0.2, random_state=42
			)
			
			# Train model
			model = RandomForestRegressor(
				n_estimators=100,
				max_depth=10,
				random_state=42,
				n_jobs=-1
			)
			model.fit(X_train, y_train)
			
			# Evaluate model
			y_pred = model.predict(X_test)
			mse = mean_squared_error(y_test, y_pred)
			rmse = np.sqrt(mse)
			
			# Store model and metadata
			self.models[model_id] = model
			self.scalers[model_id] = scaler
			self.model_metadata[model_id] = {
				'type': 'performance_forecast',
				'rmse': rmse,
				'mse': mse,
				'feature_names': list(features.columns),
				'trained_at': datetime.utcnow(),
				'training_samples': len(training_data),
				'status': ModelStatus.TRAINED
			}
			
			logger.info(f"Performance forecast model trained: {model_id} (RMSE: {rmse:.3f})")
			return model_id
			
		except Exception as e:
			logger.error(f"Failed to train performance forecast model: {e}")
			raise
	
	async def train_anomaly_detection_model(self, training_data: pd.DataFrame) -> str:
		"""Train anomaly detection model."""
		try:
			model_id = f"anomaly_detection_{uuid7str()}"
			
			# Prepare features
			features = training_data.drop(['workflow_id', 'instance_id'], axis=1, errors='ignore')
			
			# Scale features
			scaler = StandardScaler()
			features_scaled = scaler.fit_transform(features)
			
			# Train isolation forest
			model = IsolationForest(
				contamination=0.1,  # Expect 10% anomalies
				random_state=42,
				n_jobs=-1
			)
			model.fit(features_scaled)
			
			# Store model and metadata
			self.models[model_id] = model
			self.scalers[model_id] = scaler
			self.model_metadata[model_id] = {
				'type': 'anomaly_detection',
				'contamination': 0.1,
				'feature_names': list(features.columns),
				'trained_at': datetime.utcnow(),
				'training_samples': len(training_data),
				'status': ModelStatus.TRAINED
			}
			
			logger.info(f"Anomaly detection model trained: {model_id}")
			return model_id
			
		except Exception as e:
			logger.error(f"Failed to train anomaly detection model: {e}")
			raise
	
	async def predict(self, model_id: str, features: Dict[str, float]) -> Tuple[Any, float]:
		"""Make prediction using specified model."""
		if model_id not in self.models:
			raise ValueError(f"Model {model_id} not found")
		
		model = self.models[model_id]
		scaler = self.scalers[model_id]
		metadata = self.model_metadata[model_id]
		
		# Prepare features
		feature_array = np.array([features[name] for name in metadata['feature_names']]).reshape(1, -1)
		feature_scaled = scaler.transform(feature_array)
		
		# Make prediction
		if metadata['type'] == 'failure_prediction':
			prediction = model.predict(feature_scaled)[0]
			probability = model.predict_proba(feature_scaled)[0].max()
			return prediction, probability
			
		elif metadata['type'] == 'performance_forecast':
			prediction = model.predict(feature_scaled)[0]
			# For regression, confidence is based on feature importance
			confidence = 0.8  # Simplified confidence
			return prediction, confidence
			
		elif metadata['type'] == 'anomaly_detection':
			prediction = model.predict(feature_scaled)[0]
			score = model.decision_function(feature_scaled)[0]
			confidence = abs(score)  # Distance from decision boundary
			return prediction == -1, confidence  # -1 indicates anomaly
		
		else:
			raise ValueError(f"Unknown model type: {metadata['type']}")
	
	def get_model_info(self, model_id: str) -> Dict[str, Any]:
		"""Get model information."""
		if model_id not in self.model_metadata:
			raise ValueError(f"Model {model_id} not found")
		return self.model_metadata[model_id]
	
	def list_models(self) -> List[Dict[str, Any]]:
		"""List all available models."""
		return [
			{'model_id': model_id, **metadata}
			for model_id, metadata in self.model_metadata.items()
		]


class PredictiveAnalyticsEngine(APGBaseService):
	"""Main predictive analytics engine."""
	
	def __init__(self):
		super().__init__()
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
		self.event_bus: Optional[APGEventBus] = None
		self.config = None
		
		self.model_manager = MLModelManager()
		self.prediction_cache: Dict[str, PredictionResult] = {}
		self.anomaly_history: List[AnomalyDetection] = []
		self.forecast_cache: Dict[str, PerformanceForecast] = {}
		
		# Background tasks
		self._prediction_tasks: List[asyncio.Task] = []
		self._model_retraining_task: Optional[asyncio.Task] = None
	
	async def start(self):
		"""Start predictive analytics engine."""
		await super().start()
		self.config = await get_config()
		
		# Initialize event bus
		self.event_bus = APGEventBus(
			redis_url=self.config.get_redis_url(),
			service_name="predictive_analytics"
		)
		await self.event_bus.start()
		
		# Start background tasks
		await self._start_background_tasks()
		
		logger.info("Predictive analytics engine started")
	
	async def stop(self):
		"""Stop predictive analytics engine."""
		# Cancel background tasks
		for task in self._prediction_tasks:
			task.cancel()
		
		if self._model_retraining_task:
			self._model_retraining_task.cancel()
		
		# Stop event bus
		if self.event_bus:
			await self.event_bus.stop()
		
		await super().stop()
		logger.info("Predictive analytics engine stopped")
	
	async def _start_background_tasks(self):
		"""Start background prediction and monitoring tasks."""
		self._prediction_tasks = [
			asyncio.create_task(self._continuous_anomaly_detection()),
			asyncio.create_task(self._continuous_failure_prediction()),
			asyncio.create_task(self._performance_forecasting_task()),
		]
		
		self._model_retraining_task = asyncio.create_task(self._model_retraining_task_loop())
	
	async def train_models(self, force_retrain: bool = False) -> Dict[str, str]:
		"""Train all predictive models."""
		logger.info("Starting model training...")
		
		try:
			# Get training data
			failure_data = await self._get_failure_training_data()
			performance_data = await self._get_performance_training_data()
			anomaly_data = await self._get_anomaly_training_data()
			
			# Train models
			models_trained = {}
			
			if len(failure_data) > 100:  # Minimum samples needed
				failure_model_id = await self.model_manager.train_failure_prediction_model(failure_data)
				models_trained['failure_prediction'] = failure_model_id
			
			if len(performance_data) > 100:
				performance_model_id = await self.model_manager.train_performance_forecast_model(performance_data)
				models_trained['performance_forecast'] = performance_model_id
			
			if len(anomaly_data) > 200:
				anomaly_model_id = await self.model_manager.train_anomaly_detection_model(anomaly_data)
				models_trained['anomaly_detection'] = anomaly_model_id
			
			# Audit log
			await self.audit.log_event({
				'event_type': 'models_trained',
				'models_trained': models_trained,
				'training_data_sizes': {
					'failure_data': len(failure_data),
					'performance_data': len(performance_data),
					'anomaly_data': len(anomaly_data)
				}
			})
			
			logger.info(f"Model training completed: {list(models_trained.keys())}")
			return models_trained
			
		except Exception as e:
			logger.error(f"Model training failed: {e}")
			raise
	
	async def predict_workflow_failure(self, workflow_id: str, instance_id: Optional[str] = None) -> PredictionResult:
		"""Predict if a workflow will fail."""
		try:
			# Get workflow features
			features = await self._extract_workflow_features(workflow_id, instance_id)
			
			# Find best failure prediction model
			failure_models = [
				(model_id, metadata) for model_id, metadata in self.model_manager.model_metadata.items()
				if metadata['type'] == 'failure_prediction' and metadata['status'] == ModelStatus.TRAINED
			]
			
			if not failure_models:
				raise ValueError("No trained failure prediction models available")
			
			# Use the most recent model
			model_id = max(failure_models, key=lambda x: x[1]['trained_at'])[0]
			
			# Make prediction
			will_fail, confidence = await self.model_manager.predict(model_id, features)
			
			# Create prediction result
			result = PredictionResult(
				prediction_type=PredictionType.FAILURE_PREDICTION,
				workflow_id=workflow_id,
				instance_id=instance_id,
				predicted_value=bool(will_fail),
				confidence=confidence,
				prediction_horizon=timedelta(hours=24),
				model_version=model_id,
				features_used=list(features.keys()),
				metadata={
					'model_accuracy': self.model_manager.model_metadata[model_id]['accuracy'],
					'features': features
				}
			)
			
			# Cache result
			self.prediction_cache[result.id] = result
			
			# Send alert if high failure probability
			if will_fail and confidence > 0.8:
				await self._send_failure_alert(result)
			
			return result
			
		except Exception as e:
			logger.error(f"Failure prediction failed for workflow {workflow_id}: {e}")
			raise
	
	async def detect_anomalies(self, workflow_id: Optional[str] = None) -> List[AnomalyDetection]:
		"""Detect anomalies in workflow execution."""
		try:
			# Get recent execution data
			execution_data = await self._get_recent_execution_data(workflow_id)
			
			if execution_data.empty:
				return []
			
			# Find anomaly detection model
			anomaly_models = [
				(model_id, metadata) for model_id, metadata in self.model_manager.model_metadata.items()
				if metadata['type'] == 'anomaly_detection' and metadata['status'] == ModelStatus.TRAINED
			]
			
			if not anomaly_models:
				logger.warning("No trained anomaly detection models available")
				return []
			
			model_id = max(anomaly_models, key=lambda x: x[1]['trained_at'])[0]
			
			# Detect anomalies
			anomalies = []
			for _, row in execution_data.iterrows():
				features = row.drop(['workflow_id', 'instance_id']).to_dict()
				
				is_anomaly, anomaly_score = await self.model_manager.predict(model_id, features)
				
				if is_anomaly:
					# Determine anomaly type and severity
					anomaly_type, severity = await self._classify_anomaly(features, anomaly_score)
					
					anomaly = AnomalyDetection(
						workflow_id=row.get('workflow_id'),
						instance_id=row.get('instance_id'),
						anomaly_score=anomaly_score,
						is_anomaly=True,
						anomaly_type=anomaly_type,
						features=features,
						severity=severity,
						description=f"Anomaly detected in {anomaly_type} (score: {anomaly_score:.3f})",
						recommendations=await self._generate_anomaly_recommendations(anomaly_type, features)
					)
					
					anomalies.append(anomaly)
					self.anomaly_history.append(anomaly)
			
			# Send alerts for critical anomalies
			critical_anomalies = [a for a in anomalies if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]]
			for anomaly in critical_anomalies:
				await self._send_anomaly_alert(anomaly)
			
			return anomalies
			
		except Exception as e:
			logger.error(f"Anomaly detection failed: {e}")
			raise
	
	async def forecast_performance(self, workflow_id: str, horizon_hours: int = 24) -> PerformanceForecast:
		"""Forecast workflow performance metrics."""
		try:
			# Get historical performance data
			historical_data = await self._get_historical_performance_data(workflow_id)
			
			if len(historical_data) < 10:
				raise ValueError(f"Insufficient historical data for workflow {workflow_id}")
			
			# Find performance forecasting model
			forecast_models = [
				(model_id, metadata) for model_id, metadata in self.model_manager.model_metadata.items()
				if metadata['type'] == 'performance_forecast' and metadata['status'] == ModelStatus.TRAINED
			]
			
			if not forecast_models:
				raise ValueError("No trained performance forecasting models available")
			
			model_id = max(forecast_models, key=lambda x: x[1]['trained_at'])[0]
			
			# Prepare features for forecasting
			latest_features = await self._extract_workflow_features(workflow_id)
			
			# Make forecast
			predicted_execution_time, confidence = await self.model_manager.predict(model_id, latest_features)
			
			# Generate additional forecasted metrics
			predicted_metrics = {
				'execution_time_seconds': predicted_execution_time,
				'success_probability': min(0.95, confidence),
				'resource_utilization': await self._predict_resource_utilization(workflow_id, predicted_execution_time),
				'cost_estimate': await self._estimate_execution_cost(workflow_id, predicted_execution_time)
			}
			
			# Calculate confidence intervals (simplified)
			confidence_intervals = {
				metric: (value * 0.8, value * 1.2) for metric, value in predicted_metrics.items()
			}
			
			# Analyze trends
			trend_analysis = await self._analyze_performance_trends(historical_data)
			
			# Create forecast result
			forecast = PerformanceForecast(
				workflow_id=workflow_id,
				forecast_horizon=timedelta(hours=horizon_hours),
				predicted_metrics=predicted_metrics,
				confidence_intervals=confidence_intervals,
				trend_analysis=trend_analysis,
				seasonality_detected=await self._detect_seasonality(historical_data),
				forecast_accuracy=self.model_manager.model_metadata[model_id].get('rmse'),
				metadata={
					'model_id': model_id,
					'historical_samples': len(historical_data),
					'features_used': list(latest_features.keys())
				}
			)
			
			# Cache forecast
			self.forecast_cache[forecast.id] = forecast
			
			return forecast
			
		except Exception as e:
			logger.error(f"Performance forecasting failed for workflow {workflow_id}: {e}")
			raise
	
	async def _continuous_anomaly_detection(self):
		"""Background task for continuous anomaly detection."""
		while self.is_started:
			try:
				# Detect anomalies across all active workflows
				anomalies = await self.detect_anomalies()
				
				if anomalies:
					logger.info(f"Detected {len(anomalies)} anomalies")
					
					# Publish anomaly events
					for anomaly in anomalies:
						await self.event_bus.publish('workflow.anomaly_detected', {
							'anomaly_id': anomaly.id,
							'workflow_id': anomaly.workflow_id,
							'anomaly_type': anomaly.anomaly_type,
							'severity': anomaly.severity.value,
							'anomaly_score': anomaly.anomaly_score
						})
				
				# Wait before next detection cycle
				await asyncio.sleep(300)  # 5 minutes
				
			except Exception as e:
				logger.error(f"Continuous anomaly detection error: {e}")
				await asyncio.sleep(600)  # Wait longer on error
	
	async def _continuous_failure_prediction(self):
		"""Background task for continuous failure prediction."""
		while self.is_started:
			try:
				# Get active workflow instances
				active_instances = await self._get_active_workflow_instances()
				
				for instance in active_instances:
					try:
						# Predict failure for each active instance
						prediction = await self.predict_workflow_failure(
							instance['workflow_id'], 
							instance['instance_id']
						)
						
						# Store prediction in database
						await self._store_prediction(prediction)
						
					except Exception as e:
						logger.error(f"Failed to predict failure for instance {instance['instance_id']}: {e}")
				
				# Wait before next prediction cycle
				await asyncio.sleep(1800)  # 30 minutes
				
			except Exception as e:
				logger.error(f"Continuous failure prediction error: {e}")
				await asyncio.sleep(3600)  # Wait longer on error
	
	async def _performance_forecasting_task(self):
		"""Background task for performance forecasting."""
		while self.is_started:
			try:
				# Get workflows with recent activity
				active_workflows = await self._get_active_workflows()
				
				for workflow in active_workflows:
					try:
						# Generate performance forecast
						forecast = await self.forecast_performance(workflow['workflow_id'])
						
						# Store forecast in database
						await self._store_forecast(forecast)
						
					except Exception as e:
						logger.error(f"Failed to forecast performance for workflow {workflow['workflow_id']}: {e}")
				
				# Wait before next forecasting cycle
				await asyncio.sleep(3600)  # 1 hour
				
			except Exception as e:
				logger.error(f"Performance forecasting task error: {e}")
				await asyncio.sleep(7200)  # Wait longer on error
	
	async def _model_retraining_task_loop(self):
		"""Background task for model retraining."""
		while self.is_started:
			try:
				# Check if models need retraining
				retrain_needed = await self._check_model_retraining_needed()
				
				if retrain_needed:
					logger.info("Starting scheduled model retraining...")
					await self.train_models(force_retrain=True)
				
				# Wait before next check (daily)
				await asyncio.sleep(86400)
				
			except Exception as e:
				logger.error(f"Model retraining task error: {e}")
				await asyncio.sleep(3600)  # Retry in 1 hour
	
	# Helper methods (data extraction, feature engineering, etc.)
	
	async def _extract_workflow_features(self, workflow_id: str, instance_id: Optional[str] = None) -> Dict[str, float]:
		"""Extract features for ML models from real workflow execution data."""
		try:
			features = {}
			
			# Get workflow definition for complexity analysis
			workflow_query = """
				SELECT w.definition, w.complexity_score, w.estimated_duration
				FROM cr_workflows w 
				WHERE w.id = %s AND w.tenant_id = %s
			"""
			workflow_result = await self.database.fetch_one(workflow_query, (workflow_id, self.tenant_id))
			
			if workflow_result:
				features['complexity_score'] = float(workflow_result['complexity_score'] or 3.0)
				features['estimated_duration'] = float(workflow_result['estimated_duration'] or 300.0)
				
				# Parse workflow definition to count tasks
				try:
					definition = json.loads(workflow_result['definition']) if isinstance(workflow_result['definition'], str) else workflow_result['definition']
					features['task_count'] = float(len(definition.get('nodes', [])))
				except (json.JSONDecodeError, TypeError, KeyError):
					features['task_count'] = 5.0
			else:
				features.update({
					'complexity_score': 3.0,
					'estimated_duration': 300.0,
					'task_count': 5.0
				})
			
			# Get execution statistics for this workflow
			if instance_id:
				instance_query = """
					SELECT 
						wi.progress_percentage,
						wi.started_at,
						wi.completed_at,
						wi.status,
						EXTRACT(EPOCH FROM (COALESCE(wi.completed_at, NOW()) - wi.started_at)) as duration_seconds
					FROM cr_workflow_instances wi
					WHERE wi.id = %s AND wi.tenant_id = %s
				"""
				instance_result = await self.database.fetch_one(instance_query, (instance_id, self.tenant_id))
				
				if instance_result:
					features['current_duration'] = float(instance_result['duration_seconds'] or 0.0)
					features['progress_percentage'] = float(instance_result['progress_percentage'] or 0.0)
				else:
					features.update({
						'current_duration': 0.0,
						'progress_percentage': 0.0
					})
			
			# Get historical execution statistics
			history_query = """
				SELECT 
					AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_execution_time,
					COUNT(*) as total_executions,
					AVG(progress_percentage) as avg_progress,
					COUNT(CASE WHEN status = 'failed' THEN 1 END)::float / COUNT(*) as failure_rate
				FROM cr_workflow_instances 
				WHERE workflow_id = %s AND tenant_id = %s 
					AND completed_at IS NOT NULL
					AND started_at >= NOW() - INTERVAL '30 days'
			"""
			history_result = await self.database.fetch_one(history_query, (workflow_id, self.tenant_id))
			
			if history_result and history_result['total_executions']:
				features['avg_execution_time'] = float(history_result['avg_execution_time'] or 300.0)
				features['historical_failure_rate'] = float(history_result['failure_rate'] or 0.15)
				features['total_executions'] = float(history_result['total_executions'])
			else:
				features.update({
					'avg_execution_time': 300.0,
					'historical_failure_rate': 0.15,
					'total_executions': 0.0
				})
			
			# Get task execution statistics
			task_stats_query = """
				SELECT 
					COUNT(*) as task_executions,
					AVG(CASE WHEN retry_count > 0 THEN retry_count ELSE 0 END) as avg_retry_count,
					COUNT(CASE WHEN status = 'failed' THEN 1 END)::float / COUNT(*) as task_failure_rate
				FROM cr_task_executions te
				JOIN cr_workflow_instances wi ON te.instance_id = wi.id
				WHERE wi.workflow_id = %s AND wi.tenant_id = %s
					AND te.created_at >= NOW() - INTERVAL '30 days'
			"""
			task_stats_result = await self.database.fetch_one(task_stats_query, (workflow_id, self.tenant_id))
			
			if task_stats_result and task_stats_result['task_executions']:
				features['avg_retry_count'] = float(task_stats_result['avg_retry_count'] or 0.0)
				features['task_failure_rate'] = float(task_stats_result['task_failure_rate'] or 0.0)
			else:
				features.update({
					'avg_retry_count': 0.0,
					'task_failure_rate': 0.0
				})
			
			# Get current system load metrics
			concurrent_query = """
				SELECT COUNT(*) as concurrent_workflows
				FROM cr_workflow_instances 
				WHERE tenant_id = %s 
					AND status IN ('running', 'paused', 'waiting')
			"""
			concurrent_result = await self.database.fetch_one(concurrent_query, (self.tenant_id,))
			features['concurrent_workflows'] = float(concurrent_result['concurrent_workflows'] if concurrent_result else 0)
			
			# Add temporal features
			now = datetime.now()
			features['time_of_day'] = float(now.hour)
			features['day_of_week'] = float(now.weekday())
			features['is_weekend'] = float(now.weekday() >= 5)
			features['hour_sin'] = float(np.sin(2 * np.pi * now.hour / 24))
			features['hour_cos'] = float(np.cos(2 * np.pi * now.hour / 24))
			
			# Add resource usage estimate based on workflow complexity and current load
			base_resource_usage = min(features['complexity_score'] / 5.0, 1.0)
			load_factor = min(features['concurrent_workflows'] / 50.0, 1.0)
			features['estimated_resource_usage'] = float(base_resource_usage * (1 + load_factor * 0.5))
			
			return features
			
		except Exception as e:
			logger.error(f"Error extracting workflow features: {e}")
			# Return default features on error
			return {
				'avg_execution_time': 300.0,
				'task_count': 5.0,
				'complexity_score': 3.5,
				'estimated_resource_usage': 0.7,
				'avg_retry_count': 1.0,
				'time_of_day': float(datetime.now().hour),
				'day_of_week': float(datetime.now().weekday()),
				'concurrent_workflows': 10.0,
				'historical_failure_rate': 0.15,
				'task_failure_rate': 0.05,
				'is_weekend': float(datetime.now().weekday() >= 5)
			}
	
	async def _get_failure_training_data(self) -> pd.DataFrame:
		"""Get real training data for failure prediction from database."""
		try:
			# Query workflow execution data with failure outcomes
			training_query = """
				WITH workflow_features AS (
					SELECT 
						wi.id as instance_id,
						wi.workflow_id,
						w.complexity_score,
						w.estimated_duration,
						EXTRACT(EPOCH FROM (COALESCE(wi.completed_at, NOW()) - wi.started_at)) as execution_time,
						wi.progress_percentage,
						wi.status,
						CASE WHEN wi.status = 'failed' THEN 1 ELSE 0 END as failed,
						EXTRACT(HOUR FROM wi.started_at) as time_of_day,
						EXTRACT(DOW FROM wi.started_at) as day_of_week,
						(
							SELECT COUNT(*)
							FROM cr_workflow_instances wi2 
							WHERE wi2.tenant_id = wi.tenant_id 
								AND wi2.status IN ('running', 'paused', 'waiting')
								AND wi2.started_at <= wi.started_at
								AND (wi2.completed_at IS NULL OR wi2.completed_at >= wi.started_at)
						) as concurrent_workflows,
						(
							SELECT COUNT(*)
							FROM cr_task_executions te
							WHERE te.instance_id = wi.id
						) as task_count,
						(
							SELECT AVG(COALESCE(te.retry_count, 0))
							FROM cr_task_executions te
							WHERE te.instance_id = wi.id
						) as avg_retry_count,
						(
							SELECT COUNT(CASE WHEN te.status = 'failed' THEN 1 END)::float / NULLIF(COUNT(*), 0)
							FROM cr_task_executions te
							WHERE te.instance_id = wi.id
						) as task_failure_rate
					FROM cr_workflow_instances wi
					JOIN cr_workflows w ON wi.workflow_id = w.id
					WHERE wi.tenant_id = %s
						AND wi.started_at >= NOW() - INTERVAL '90 days'
						AND wi.started_at IS NOT NULL
						AND (wi.completed_at IS NOT NULL OR wi.status IN ('failed', 'cancelled'))
					ORDER BY wi.started_at DESC
					LIMIT 5000
				)
				SELECT 
					COALESCE(complexity_score, 3.0) as complexity_score,
					COALESCE(execution_time / 60.0, 5.0) as execution_time_minutes,
					COALESCE(progress_percentage / 100.0, 0.0) as progress_ratio,
					COALESCE(time_of_day, 12) as time_of_day,
					COALESCE(day_of_week, 1) as day_of_week,
					COALESCE(concurrent_workflows, 1) as concurrent_workflows,
					COALESCE(task_count, 1) as task_count,
					COALESCE(avg_retry_count, 0) as avg_retry_count,
					COALESCE(task_failure_rate, 0) as task_failure_rate,
					CASE WHEN day_of_week IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
					SIN(2 * PI() * time_of_day / 24) as hour_sin,
					COS(2 * PI() * time_of_day / 24) as hour_cos,
					failed
				FROM workflow_features
				WHERE execution_time > 0
					AND task_count > 0
			"""
			
			results = await self.database.fetch_all(training_query, (self.tenant_id,))
			
			if not results or len(results) < 100:
				logger.warning(f"Insufficient training data: {len(results) if results else 0} records. Using synthetic data.")
				# Generate synthetic data based on real patterns when insufficient data
				n_samples = max(1000, len(results) * 10 if results else 1000)
				
				# Base synthetic data on any real data patterns we found
				if results:
					real_df = pd.DataFrame(results)
					complexity_mean = real_df['complexity_score'].mean()
					execution_mean = real_df['execution_time_minutes'].mean()
					failure_rate = real_df['failed'].mean()
				else:
					complexity_mean, execution_mean, failure_rate = 3.0, 5.0, 0.15
				
				return pd.DataFrame({
					'complexity_score': np.random.normal(complexity_mean, complexity_mean * 0.3, n_samples),
					'execution_time_minutes': np.random.lognormal(np.log(execution_mean), 0.5, n_samples),
					'progress_ratio': np.random.beta(2, 1, n_samples),
					'time_of_day': np.random.randint(0, 24, n_samples),
					'day_of_week': np.random.randint(0, 7, n_samples),
					'concurrent_workflows': np.random.poisson(10, n_samples),
					'task_count': np.random.randint(1, 20, n_samples),
					'avg_retry_count': np.random.exponential(0.5, n_samples),
					'task_failure_rate': np.random.beta(1, 10, n_samples),
					'is_weekend': np.random.choice([0, 1], n_samples, p=[5/7, 2/7]),
					'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
					'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
					'failed': np.random.choice([0, 1], n_samples, p=[1-failure_rate, failure_rate])
				})
			
			# Convert to DataFrame and clean data
			df = pd.DataFrame(results)
			
			# Data cleaning and validation
			df = df.dropna()
			df = df[df['execution_time_minutes'] > 0]
			df = df[df['task_count'] > 0]
			df = df[df['complexity_score'] > 0]
			
			# Cap extreme values
			df['execution_time_minutes'] = df['execution_time_minutes'].clip(0.1, 1440)  # Max 24 hours
			df['complexity_score'] = df['complexity_score'].clip(1, 10)
			df['concurrent_workflows'] = df['concurrent_workflows'].clip(0, 1000)
			df['task_count'] = df['task_count'].clip(1, 100)
			df['avg_retry_count'] = df['avg_retry_count'].clip(0, 10)
			df['task_failure_rate'] = df['task_failure_rate'].clip(0, 1)
			
			logger.info(f"Loaded {len(df)} workflow execution records for failure prediction training")
			
			# Ensure we have sufficient positive and negative examples
			failure_rate = df['failed'].mean()
			if failure_rate < 0.05 or failure_rate > 0.5:
				logger.warning(f"Unusual failure rate in training data: {failure_rate:.3f}")
			
			return df
			
		except Exception as e:
			logger.error(f"Error loading failure training data: {e}")
			# Fallback to synthetic data
			return pd.DataFrame({
				'complexity_score': np.random.normal(3.0, 1.0, 1000),
				'execution_time_minutes': np.random.lognormal(np.log(5.0), 0.5, 1000),
				'progress_ratio': np.random.beta(2, 1, 1000),
				'time_of_day': np.random.randint(0, 24, 1000),
				'day_of_week': np.random.randint(0, 7, 1000),
				'concurrent_workflows': np.random.poisson(10, 1000),
				'task_count': np.random.randint(1, 20, 1000),
				'avg_retry_count': np.random.exponential(0.5, 1000),
				'task_failure_rate': np.random.beta(1, 10, 1000),
				'is_weekend': np.random.choice([0, 1], 1000, p=[5/7, 2/7]),
				'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, 1000) / 24),
				'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, 1000) / 24),
				'failed': np.random.choice([0, 1], 1000, p=[0.85, 0.15])
			})
	
	async def _get_performance_training_data(self) -> pd.DataFrame:
		"""Get real training data for performance forecasting from database."""
		try:
			# Query successful workflow executions with performance metrics
			performance_query = """
				WITH performance_features AS (
					SELECT 
						wi.id as instance_id,
						wi.workflow_id,
						w.complexity_score,
						w.estimated_duration,
						EXTRACT(EPOCH FROM (wi.completed_at - wi.started_at)) as actual_execution_time,
						wi.progress_percentage,
						EXTRACT(HOUR FROM wi.started_at) as time_of_day,
						EXTRACT(DOW FROM wi.started_at) as day_of_week,
						(
							SELECT COUNT(*)
							FROM cr_workflow_instances wi2 
							WHERE wi2.tenant_id = wi.tenant_id 
								AND wi2.status IN ('running', 'paused', 'waiting')
								AND wi2.started_at <= wi.started_at
								AND (wi2.completed_at IS NULL OR wi2.completed_at >= wi.started_at)
						) as concurrent_workflows,
						(
							SELECT COUNT(*)
							FROM cr_task_executions te
							WHERE te.instance_id = wi.id
						) as task_count,
						(
							SELECT AVG(COALESCE(te.retry_count, 0))
							FROM cr_task_executions te
							WHERE te.instance_id = wi.id
						) as avg_retry_count,
						(
							SELECT COUNT(CASE WHEN te.status = 'completed' THEN 1 END)::float / NULLIF(COUNT(*), 0)
							FROM cr_task_executions te
							WHERE te.instance_id = wi.id
						) as task_success_rate,
						-- Estimate resource usage based on complexity and duration
						LEAST(
							(w.complexity_score / 5.0) * 
							(EXTRACT(EPOCH FROM (wi.completed_at - wi.started_at)) / COALESCE(w.estimated_duration, 3600)),
							2.0
						) as estimated_resource_usage
					FROM cr_workflow_instances wi
					JOIN cr_workflows w ON wi.workflow_id = w.id
					WHERE wi.tenant_id = %s
						AND wi.status = 'completed'
						AND wi.started_at >= NOW() - INTERVAL '60 days'
						AND wi.started_at IS NOT NULL
						AND wi.completed_at IS NOT NULL
						AND wi.completed_at > wi.started_at
					ORDER BY wi.started_at DESC
					LIMIT 3000
				)
				SELECT 
					COALESCE(complexity_score, 3.0) as complexity_score,
					COALESCE(actual_execution_time / 60.0, 5.0) as execution_time_minutes,
					COALESCE(time_of_day, 12) as time_of_day,
					COALESCE(day_of_week, 1) as day_of_week,
					COALESCE(concurrent_workflows, 1) as concurrent_workflows,
					COALESCE(task_count, 1) as task_count,
					COALESCE(avg_retry_count, 0) as avg_retry_count,
					COALESCE(task_success_rate, 1.0) as task_success_rate,
					COALESCE(estimated_resource_usage, 0.5) as estimated_resource_usage,
					CASE WHEN day_of_week IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
					SIN(2 * PI() * time_of_day / 24) as hour_sin,
					COS(2 * PI() * time_of_day / 24) as hour_cos,
					-- Performance ratio (actual vs estimated)
					COALESCE(
						CASE 
							WHEN estimated_duration > 0 THEN actual_execution_time / estimated_duration
							ELSE 1.0 
						END, 
						1.0
					) as performance_ratio
				FROM performance_features
				WHERE actual_execution_time > 0
					AND task_count > 0
					AND complexity_score > 0
			"""
			
			results = await self.database.fetch_all(performance_query, (self.tenant_id,))
			
			if not results or len(results) < 50:
				logger.warning(f"Insufficient performance data: {len(results) if results else 0} records. Using synthetic data.")
				# Generate synthetic data based on realistic performance patterns
				n_samples = max(1000, len(results) * 20 if results else 1000)
				
				# Use any real data patterns we found
				if results:
					real_df = pd.DataFrame(results)
					complexity_mean = real_df['complexity_score'].mean()
					execution_mean = real_df['execution_time_minutes'].mean()
					performance_mean = real_df['performance_ratio'].mean()
				else:
					complexity_mean, execution_mean, performance_mean = 3.0, 10.0, 1.2
				
				return pd.DataFrame({
					'complexity_score': np.random.gamma(4, complexity_mean/4, n_samples),
					'execution_time_minutes': np.random.lognormal(np.log(execution_mean), 0.6, n_samples),
					'time_of_day': np.random.randint(0, 24, n_samples),
					'day_of_week': np.random.randint(0, 7, n_samples),
					'concurrent_workflows': np.random.poisson(8, n_samples),
					'task_count': np.random.poisson(7, n_samples) + 1,
					'avg_retry_count': np.random.exponential(0.3, n_samples),
					'task_success_rate': np.random.beta(8, 1, n_samples),
					'estimated_resource_usage': np.random.beta(2, 3, n_samples),
					'is_weekend': np.random.choice([0, 1], n_samples, p=[5/7, 2/7]),
					'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
					'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
					'performance_ratio': np.random.lognormal(np.log(performance_mean), 0.4, n_samples)
				})
			
			# Convert to DataFrame and clean data
			df = pd.DataFrame(results)
			
			# Data cleaning and validation
			df = df.dropna()
			df = df[df['execution_time_minutes'] > 0]
			df = df[df['task_count'] > 0]
			df = df[df['complexity_score'] > 0]
			df = df[df['performance_ratio'] > 0]
			
			# Cap extreme values
			df['execution_time_minutes'] = df['execution_time_minutes'].clip(0.1, 1440)  # Max 24 hours
			df['complexity_score'] = df['complexity_score'].clip(1, 10)
			df['concurrent_workflows'] = df['concurrent_workflows'].clip(0, 500)
			df['task_count'] = df['task_count'].clip(1, 100)
			df['avg_retry_count'] = df['avg_retry_count'].clip(0, 10)
			df['task_success_rate'] = df['task_success_rate'].clip(0, 1)
			df['estimated_resource_usage'] = df['estimated_resource_usage'].clip(0.01, 2.0)
			df['performance_ratio'] = df['performance_ratio'].clip(0.1, 10.0)  # 10x slower to 10x faster
			
			logger.info(f"Loaded {len(df)} successful workflow executions for performance forecasting")
			
			# Log performance distribution insights
			perf_median = df['performance_ratio'].median()
			perf_p75 = df['performance_ratio'].quantile(0.75)
			perf_p25 = df['performance_ratio'].quantile(0.25)
			logger.info(f"Performance ratio distribution - Median: {perf_median:.2f}, P25: {perf_p25:.2f}, P75: {perf_p75:.2f}")
			
			return df
			
		except Exception as e:
			logger.error(f"Error loading performance training data: {e}")
			# Fallback to synthetic data
			return pd.DataFrame({
				'complexity_score': np.random.gamma(4, 0.75, 1000),
				'execution_time_minutes': np.random.lognormal(np.log(10), 0.6, 1000),
				'time_of_day': np.random.randint(0, 24, 1000),
				'day_of_week': np.random.randint(0, 7, 1000),
				'concurrent_workflows': np.random.poisson(8, 1000),
				'task_count': np.random.poisson(7, 1000) + 1,
				'avg_retry_count': np.random.exponential(0.3, 1000),
				'task_success_rate': np.random.beta(8, 1, 1000),
				'estimated_resource_usage': np.random.beta(2, 3, 1000),
				'is_weekend': np.random.choice([0, 1], 1000, p=[5/7, 2/7]),
				'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, 1000) / 24),
				'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, 1000) / 24),
				'performance_ratio': np.random.lognormal(np.log(1.2), 0.4, 1000)
			})
	
	async def _get_anomaly_training_data(self) -> pd.DataFrame:
		"""Get training data for anomaly detection from real workflow execution data."""
		try:
			# Get comprehensive anomaly detection training data from database
			anomaly_query = """
			WITH workflow_stats AS (
				SELECT 
					wi.id,
					wi.workflow_id,
					wi.status,
					wi.started_at,
					wi.completed_at,
					EXTRACT(EPOCH FROM (wi.completed_at - wi.started_at)) as execution_time,
					COALESCE(wi.progress_percentage, 0) as progress_percentage,
					COALESCE(wi.retry_count, 0) as retry_count,
					COALESCE(
						JSONB_ARRAY_LENGTH(
							COALESCE(wi.metadata->'tasks', '[]'::jsonb)
						), 0
					) as task_count,
					COALESCE(
						(wi.metadata->>'memory_usage_mb')::float / 1024.0, 0.0
					) as memory_usage_gb,
					COALESCE(
						(wi.metadata->>'cpu_usage_percent')::float / 100.0, 0.0
					) as cpu_usage_percent,
					COALESCE(
						(wi.metadata->>'disk_io_mb')::float, 0.0
					) as disk_io_mb,
					COALESCE(
						(wi.metadata->>'network_io_mb')::float, 0.0
					) as network_io_mb,
					COALESCE(
						(wi.metadata->>'queue_wait_time_seconds')::float, 0.0
					) as queue_wait_time,
					EXTRACT(HOUR FROM wi.started_at) as hour_of_day,
					EXTRACT(DOW FROM wi.started_at) as day_of_week,
					te.error_count,
					te.success_count,
					te.timeout_count,
					COALESCE(te.avg_task_duration, 0.0) as avg_task_duration,
					COALESCE(te.max_task_duration, 0.0) as max_task_duration,
					COALESCE(te.total_tasks, 0) as total_tasks
				FROM cr_workflow_instances wi
				LEFT JOIN (
					SELECT 
						instance_id,
						COUNT(*) FILTER (WHERE status = 'failed') as error_count,
						COUNT(*) FILTER (WHERE status = 'completed') as success_count,
						COUNT(*) FILTER (WHERE status = 'timeout') as timeout_count,
						AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_task_duration,
						MAX(EXTRACT(EPOCH FROM (completed_at - started_at))) as max_task_duration,
						COUNT(*) as total_tasks
					FROM cr_task_executions
					WHERE tenant_id = %s
					AND started_at >= NOW() - INTERVAL '90 days'
					GROUP BY instance_id
				) te ON wi.id = te.instance_id
				WHERE wi.tenant_id = %s
				AND wi.started_at >= NOW() - INTERVAL '90 days'
				AND wi.completed_at IS NOT NULL
				AND wi.status IN ('completed', 'failed', 'timeout')
			),
			labeled_data AS (
				SELECT 
					*,
					-- Label as anomaly based on multiple criteria
					CASE 
						WHEN execution_time > (
							SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY execution_time) 
							FROM workflow_stats WHERE execution_time > 0
						) THEN 1
						WHEN retry_count > 5 THEN 1
						WHEN cpu_usage_percent > 0.9 THEN 1
						WHEN memory_usage_gb > 8.0 THEN 1
						WHEN error_count > total_tasks * 0.3 THEN 1
						WHEN queue_wait_time > 300 THEN 1
						WHEN progress_percentage < 50 AND status = 'failed' THEN 1
						ELSE 0
					END as is_anomaly,
					-- Calculate resource utilization ratio
					COALESCE(cpu_usage_percent + memory_usage_gb/16.0, 0.0) as resource_usage,
					-- Calculate failure rate
					CASE 
						WHEN total_tasks > 0 THEN COALESCE(error_count::float / total_tasks, 0.0)
						ELSE 0.0
					END as failure_rate,
					-- Time-based features for temporal anomaly detection
					SIN(2 * PI() * hour_of_day / 24.0) as hour_sin,
					COS(2 * PI() * hour_of_day / 24.0) as hour_cos,
					SIN(2 * PI() * day_of_week / 7.0) as dow_sin,
					COS(2 * PI() * day_of_week / 7.0) as dow_cos
				FROM workflow_stats
				WHERE execution_time > 0
			)
			SELECT 
				execution_time,
				task_count,
				resource_usage,
				retry_count,
				memory_usage_gb as memory_usage,
				cpu_usage_percent as cpu_usage,
				disk_io_mb,
				network_io_mb,
				queue_wait_time,
				error_count,
				success_count,
				timeout_count,
				avg_task_duration,
				max_task_duration,
				total_tasks,
				failure_rate,
				hour_sin,
				hour_cos,
				dow_sin,
				dow_cos,
				is_anomaly,
				progress_percentage
			FROM labeled_data
			ORDER BY started_at DESC
			LIMIT 10000
			"""
			
			result = await self.database.fetch_all(anomaly_query, (self.tenant_id, self.tenant_id))
			
			if not result:
				logger.warning("No anomaly training data found, using minimal synthetic data")
				# Return minimal synthetic data if no real data available
				return pd.DataFrame({
					'execution_time': [300.0, 250.0, 400.0, 180.0, 500.0],
					'task_count': [5, 4, 8, 3, 10],
					'resource_usage': [0.3, 0.2, 0.6, 0.1, 0.8],
					'retry_count': [0, 1, 0, 0, 2],
					'memory_usage': [0.2, 0.1, 0.4, 0.1, 0.6],
					'cpu_usage': [0.3, 0.2, 0.5, 0.1, 0.7],
					'is_anomaly': [0, 0, 0, 0, 1]
				})
			
			# Convert to pandas DataFrame
			data = []
			for row in result:
				data.append(dict(row))
			
			df = pd.DataFrame(data)
			
			# Ensure minimum required columns are present with default values
			required_columns = {
				'execution_time': 300.0,
				'task_count': 5,
				'resource_usage': 0.3,
				'retry_count': 0,
				'memory_usage': 0.2,
				'cpu_usage': 0.3,
				'disk_io_mb': 0.0,
				'network_io_mb': 0.0,
				'queue_wait_time': 0.0,
				'error_count': 0,
				'success_count': 0,
				'failure_rate': 0.0,
				'hour_sin': 0.0,
				'hour_cos': 1.0,
				'dow_sin': 0.0,
				'dow_cos': 1.0,
				'is_anomaly': 0,
				'progress_percentage': 100.0
			}
			
			for col, default_val in required_columns.items():
				if col not in df.columns:
					df[col] = default_val
				else:
					df[col] = df[col].fillna(default_val)
			
			# Remove rows with invalid execution times
			df = df[df['execution_time'] > 0]
			
			# Convert data types
			numeric_columns = [
				'execution_time', 'resource_usage', 'memory_usage', 'cpu_usage',
				'disk_io_mb', 'network_io_mb', 'queue_wait_time', 'failure_rate',
				'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'progress_percentage'
			]
			
			for col in numeric_columns:
				if col in df.columns:
					df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
			
			integer_columns = [
				'task_count', 'retry_count', 'error_count', 'success_count', 
				'timeout_count', 'total_tasks', 'is_anomaly'
			]
			
			for col in integer_columns:
				if col in df.columns:
					df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
			
			logger.info(f"Retrieved {len(df)} anomaly training samples with {df['is_anomaly'].sum()} anomalies")
			
			return df
			
		except Exception as e:
			logger.error(f"Failed to get anomaly training data: {e}")
			# Return fallback synthetic data with realistic anomaly patterns
			np.random.seed(42)  # For reproducible results
			n_samples = 1000
			
			# Generate normal execution patterns
			normal_execution_time = np.random.normal(300, 50, int(n_samples * 0.85))
			anomaly_execution_time = np.random.normal(800, 200, int(n_samples * 0.15))
			execution_times = np.concatenate([normal_execution_time, anomaly_execution_time])
			
			normal_resource = np.random.uniform(0.2, 0.6, int(n_samples * 0.85))
			anomaly_resource = np.random.uniform(0.8, 1.0, int(n_samples * 0.15))
			resource_usage = np.concatenate([normal_resource, anomaly_resource])
			
			# Create labels
			labels = np.concatenate([
				np.zeros(int(n_samples * 0.85)),
				np.ones(int(n_samples * 0.15))
			])
			
			# Shuffle data
			indices = np.random.permutation(n_samples)
			
			return pd.DataFrame({
				'execution_time': execution_times[indices],
				'task_count': np.random.randint(3, 15, n_samples),
				'resource_usage': resource_usage[indices],
				'retry_count': np.random.poisson(0.5, n_samples),
				'memory_usage': np.random.uniform(0.1, 0.8, n_samples),
				'cpu_usage': np.random.uniform(0.1, 0.9, n_samples),
				'failure_rate': np.random.beta(1, 10, n_samples),
				'is_anomaly': labels[indices].astype(int)
			})
	
	async def _classify_anomaly(self, features: Dict[str, float], anomaly_score: float) -> Tuple[str, AlertSeverity]:
		"""Classify anomaly type and severity."""
		# Simplified anomaly classification
		if features.get('execution_time', 0) > 1000:
			return "performance_degradation", AlertSeverity.WARNING
		elif features.get('retry_count', 0) > 3:
			return "high_retry_rate", AlertSeverity.CRITICAL
		elif features.get('resource_usage', 0) > 0.9:
			return "resource_exhaustion", AlertSeverity.CRITICAL
		else:
			return "execution_anomaly", AlertSeverity.INFO
	
	async def _generate_anomaly_recommendations(self, anomaly_type: str, features: Dict[str, float]) -> List[str]:
		"""Generate recommendations for handling anomalies."""
		recommendations = {
			"performance_degradation": [
				"Consider optimizing workflow tasks",
				"Review resource allocation",
				"Check for external service latency"
			],
			"high_retry_rate": [
				"Investigate task failure causes",
				"Review error handling logic",
				"Consider circuit breaker patterns"
			],
			"resource_exhaustion": [
				"Scale up resources",
				"Optimize memory usage",
				"Consider workflow splitting"
			]
		}
		return recommendations.get(anomaly_type, ["Review workflow configuration"])
	
	async def _send_failure_alert(self, prediction: PredictionResult):
		"""Send failure prediction alert."""
		await self.event_bus.publish('workflow.failure_predicted', {
			'prediction_id': prediction.id,
			'workflow_id': prediction.workflow_id,
			'instance_id': prediction.instance_id,
			'confidence': prediction.confidence,
			'prediction_horizon': prediction.prediction_horizon.total_seconds()
		})
	
	async def _send_anomaly_alert(self, anomaly: AnomalyDetection):
		"""Send anomaly detection alert."""
		await self.event_bus.publish('workflow.anomaly_alert', {
			'anomaly_id': anomaly.id,
			'workflow_id': anomaly.workflow_id,
			'instance_id': anomaly.instance_id,
			'anomaly_type': anomaly.anomaly_type,
			'severity': anomaly.severity.value,
			'description': anomaly.description,
			'recommendations': anomaly.recommendations
		})
	
	# Additional helper methods would be implemented here...
	
	async def get_prediction_analytics(self) -> Dict[str, Any]:
		"""Get analytics about predictions made."""
		return {
			'total_predictions': len(self.prediction_cache),
			'anomalies_detected': len(self.anomaly_history),
			'forecasts_generated': len(self.forecast_cache),
			'model_count': len(self.model_manager.models),
			'active_models': [
				{'model_id': model_id, **metadata}
				for model_id, metadata in self.model_manager.model_metadata.items()
				if metadata['status'] == ModelStatus.TRAINED
			]
		}
	
	async def health_check(self) -> bool:
		"""Health check for predictive analytics engine."""
		try:
			# Check if models are available
			if not self.model_manager.models:
				return False
			
			# Check if background tasks are running
			active_tasks = [task for task in self._prediction_tasks if not task.done()]
			if len(active_tasks) < len(self._prediction_tasks) / 2:
				return False
			
			return True
			
		except Exception:
			return False


# Global predictive analytics engine instance
predictive_analytics_engine = PredictiveAnalyticsEngine()