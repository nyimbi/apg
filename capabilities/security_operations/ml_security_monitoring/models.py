"""
APG Machine Learning Security Monitoring - Pydantic Models

Enterprise ML security models with advanced deep learning architectures,
automated model management, and adaptive security intelligence.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from uuid_extensions import uuid7str


class ModelType(str, Enum):
	CLASSIFICATION = "classification"
	REGRESSION = "regression"
	CLUSTERING = "clustering"
	ANOMALY_DETECTION = "anomaly_detection"
	TIME_SERIES = "time_series"
	DEEP_LEARNING = "deep_learning"
	ENSEMBLE = "ensemble"


class ModelArchitecture(str, Enum):
	LINEAR = "linear"
	TREE_BASED = "tree_based"
	NEURAL_NETWORK = "neural_network"
	DEEP_NEURAL_NETWORK = "deep_neural_network"
	CONVOLUTIONAL = "convolutional"
	RECURRENT = "recurrent"
	TRANSFORMER = "transformer"
	AUTOENCODER = "autoencoder"


class ModelStatus(str, Enum):
	TRAINING = "training"
	VALIDATING = "validating"
	DEPLOYED = "deployed"
	RETIRED = "retired"
	FAILED = "failed"
	UPDATING = "updating"


class PredictionType(str, Enum):
	THREAT_CLASSIFICATION = "threat_classification"
	ANOMALY_SCORE = "anomaly_score"
	RISK_ASSESSMENT = "risk_assessment"
	BEHAVIORAL_ANALYSIS = "behavioral_analysis"
	MALWARE_DETECTION = "malware_detection"
	PHISHING_DETECTION = "phishing_detection"


class TrainingStatus(str, Enum):
	QUEUED = "queued"
	PREPROCESSING = "preprocessing"
	TRAINING = "training"
	VALIDATING = "validating"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class MLModel(BaseModel):
	"""Machine learning model specification and metadata"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	name: str = Field(description="Model name")
	description: str = Field(description="Model description")
	version: str = Field(description="Model version")
	
	model_type: ModelType
	architecture: ModelArchitecture
	prediction_type: PredictionType
	
	# Model configuration
	hyperparameters: Dict[str, Any] = Field(default_factory=dict)
	feature_config: Dict[str, Any] = Field(default_factory=dict)
	preprocessing_config: Dict[str, Any] = Field(default_factory=dict)
	
	# Training configuration
	training_config: Dict[str, Any] = Field(default_factory=dict)
	validation_config: Dict[str, Any] = Field(default_factory=dict)
	
	# Model artifacts
	model_path: Optional[str] = None
	model_size_mb: Optional[Decimal] = None
	feature_importance: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Performance metrics
	accuracy: Optional[Decimal] = None
	precision: Optional[Decimal] = None
	recall: Optional[Decimal] = None
	f1_score: Optional[Decimal] = None
	auc_roc: Optional[Decimal] = None
	
	# Deployment information
	status: ModelStatus = ModelStatus.TRAINING
	deployment_date: Optional[datetime] = None
	endpoint_url: Optional[str] = None
	
	# Usage statistics
	prediction_count: int = 0
	error_count: int = 0
	average_inference_time: Optional[Decimal] = None
	
	# Model lineage
	parent_model_id: Optional[str] = None
	training_job_id: Optional[str] = None
	dataset_versions: List[str] = Field(default_factory=list)
	
	# Monitoring
	drift_detection_enabled: bool = True
	performance_threshold: Decimal = Field(default=Decimal('0.95'))
	
	created_by: str = Field(description="Model creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ModelTraining(BaseModel):
	"""Model training job specification and tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	model_id: str = Field(description="Associated model ID")
	
	training_name: str = Field(description="Training job name")
	training_type: str = Field(description="Type of training (initial, retrain, finetune)")
	
	# Training data
	dataset_id: str = Field(description="Training dataset ID")
	dataset_size: int = Field(description="Number of training samples")
	validation_split: Decimal = Field(default=Decimal('0.2'), ge=0, le=1)
	
	# Training configuration
	training_config: Dict[str, Any] = Field(default_factory=dict)
	hyperparameters: Dict[str, Any] = Field(default_factory=dict)
	
	# Resource allocation
	compute_resources: Dict[str, Any] = Field(default_factory=dict)
	gpu_enabled: bool = False
	distributed_training: bool = False
	
	# Training progress
	status: TrainingStatus = TrainingStatus.QUEUED
	progress_percentage: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	current_epoch: Optional[int] = None
	total_epochs: Optional[int] = None
	
	# Training metrics
	training_loss: List[Decimal] = Field(default_factory=list)
	validation_loss: List[Decimal] = Field(default_factory=list)
	training_accuracy: List[Decimal] = Field(default_factory=list)
	validation_accuracy: List[Decimal] = Field(default_factory=list)
	
	# Performance results
	final_accuracy: Optional[Decimal] = None
	final_loss: Optional[Decimal] = None
	best_accuracy: Optional[Decimal] = None
	best_epoch: Optional[int] = None
	
	# Training duration
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	training_duration: Optional[timedelta] = None
	
	# Resource usage
	max_memory_usage: Optional[Decimal] = None
	avg_cpu_usage: Optional[Decimal] = None
	avg_gpu_usage: Optional[Decimal] = None
	
	# Training logs
	training_logs: List[Dict[str, Any]] = Field(default_factory=list)
	error_logs: List[str] = Field(default_factory=list)
	
	# Output artifacts
	model_artifacts: Dict[str, str] = Field(default_factory=dict)
	evaluation_report: Optional[Dict[str, Any]] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class MLPrediction(BaseModel):
	"""ML model prediction result"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	model_id: str = Field(description="Model used for prediction")
	
	# Prediction input
	input_data: Dict[str, Any] = Field(default_factory=dict)
	input_features: Dict[str, Any] = Field(default_factory=dict)
	
	# Prediction output
	prediction_type: PredictionType
	prediction_value: Union[str, int, float, List[Any]] = Field(description="Main prediction")
	prediction_probabilities: Dict[str, Decimal] = Field(default_factory=dict)
	confidence_score: Decimal = Field(ge=0, le=100)
	
	# Model information
	model_version: str = Field(description="Model version used")
	inference_time_ms: Decimal = Field(description="Inference time in milliseconds")
	
	# Feature analysis
	feature_contributions: Dict[str, Decimal] = Field(default_factory=dict)
	feature_importance: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Prediction context
	prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
	batch_id: Optional[str] = None
	request_id: Optional[str] = None
	
	# Quality metrics
	data_quality_score: Optional[Decimal] = None
	prediction_quality_score: Optional[Decimal] = None
	
	# Feedback and validation
	actual_outcome: Optional[Union[str, int, float]] = None
	feedback_provided: bool = False
	feedback_timestamp: Optional[datetime] = None
	
	# Alert generation
	generates_alert: bool = False
	alert_severity: Optional[str] = None
	alert_reason: Optional[str] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ModelPerformance(BaseModel):
	"""Model performance monitoring and metrics"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	model_id: str = Field(description="Monitored model ID")
	
	# Monitoring period
	monitoring_period_start: datetime
	monitoring_period_end: datetime
	
	# Prediction metrics
	total_predictions: int = 0
	successful_predictions: int = 0
	failed_predictions: int = 0
	success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Performance metrics
	average_inference_time: Decimal = Field(description="Average inference time in ms")
	p95_inference_time: Decimal = Field(description="95th percentile inference time")
	p99_inference_time: Decimal = Field(description="99th percentile inference time")
	
	# Accuracy metrics (when ground truth available)
	accuracy: Optional[Decimal] = None
	precision: Optional[Decimal] = None
	recall: Optional[Decimal] = None
	f1_score: Optional[Decimal] = None
	
	# Distribution metrics
	prediction_distribution: Dict[str, int] = Field(default_factory=dict)
	confidence_distribution: Dict[str, int] = Field(default_factory=dict)
	
	# Drift detection
	data_drift_detected: bool = False
	data_drift_score: Optional[Decimal] = None
	concept_drift_detected: bool = False
	concept_drift_score: Optional[Decimal] = None
	
	# Error analysis
	error_types: Dict[str, int] = Field(default_factory=dict)
	error_patterns: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Resource utilization
	avg_cpu_usage: Optional[Decimal] = None
	avg_memory_usage: Optional[Decimal] = None
	peak_memory_usage: Optional[Decimal] = None
	
	# Alert thresholds
	performance_alerts: List[Dict[str, Any]] = Field(default_factory=list)
	threshold_violations: List[str] = Field(default_factory=list)
	
	# Comparison with baseline
	baseline_comparison: Optional[Dict[str, Decimal]] = None
	performance_trend: str = Field(default="stable")  # improving, degrading, stable
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class FeatureEngineering(BaseModel):
	"""Feature engineering pipeline and transformations"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	pipeline_name: str = Field(description="Feature engineering pipeline name")
	version: str = Field(description="Pipeline version")
	
	# Input data schema
	input_schema: Dict[str, Any] = Field(default_factory=dict)
	
	# Feature transformations
	transformations: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Feature selection
	feature_selection_method: Optional[str] = None
	selected_features: List[str] = Field(default_factory=list)
	feature_importance_scores: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Scaling and normalization
	scaling_method: Optional[str] = None
	scaling_parameters: Dict[str, Any] = Field(default_factory=dict)
	
	# Categorical encoding
	categorical_encoders: Dict[str, Any] = Field(default_factory=dict)
	
	# Time-based features
	temporal_features: List[str] = Field(default_factory=list)
	time_windows: List[str] = Field(default_factory=list)
	
	# Aggregated features
	aggregation_functions: List[str] = Field(default_factory=list)
	aggregation_windows: List[str] = Field(default_factory=list)
	
	# Output schema
	output_schema: Dict[str, Any] = Field(default_factory=dict)
	feature_count: int = Field(description="Number of output features")
	
	# Pipeline artifacts
	pipeline_path: Optional[str] = None
	transformer_artifacts: Dict[str, str] = Field(default_factory=dict)
	
	# Validation metrics
	data_quality_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	feature_stability_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	created_by: str = Field(description="Pipeline creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ModelMetrics(BaseModel):
	"""Comprehensive ML model metrics and KPIs"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	metric_period_start: datetime
	metric_period_end: datetime
	
	# Model inventory
	total_models: int = 0
	active_models: int = 0
	training_models: int = 0
	retired_models: int = 0
	
	# Model types distribution
	model_type_distribution: Dict[str, int] = Field(default_factory=dict)
	architecture_distribution: Dict[str, int] = Field(default_factory=dict)
	
	# Training metrics
	training_jobs_completed: int = 0
	training_jobs_failed: int = 0
	average_training_time: Optional[timedelta] = None
	
	# Prediction metrics
	total_predictions: int = 0
	successful_predictions: int = 0
	average_inference_time: Decimal = Field(default=Decimal('0.0'))
	
	# Performance metrics
	average_model_accuracy: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	best_performing_model: Optional[str] = None
	worst_performing_model: Optional[str] = None
	
	# Resource utilization
	total_compute_hours: Decimal = Field(default=Decimal('0.0'))
	average_memory_usage: Decimal = Field(default=Decimal('0.0'))
	gpu_utilization: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Quality metrics
	data_quality_issues: int = 0
	model_drift_alerts: int = 0
	performance_degradation_alerts: int = 0
	
	# Security metrics
	malware_detection_accuracy: Optional[Decimal] = None
	phishing_detection_accuracy: Optional[Decimal] = None
	anomaly_detection_accuracy: Optional[Decimal] = None
	false_positive_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Business impact
	threats_prevented: int = 0
	incidents_reduced: int = 0
	cost_savings: Optional[Decimal] = None
	
	# Operational metrics
	model_deployment_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	average_model_lifecycle: Optional[timedelta] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class AutoMLExperiment(BaseModel):
	"""Automated machine learning experiment"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	experiment_name: str = Field(description="Experiment name")
	objective: str = Field(description="Optimization objective")
	
	# Problem definition
	problem_type: ModelType
	target_variable: str = Field(description="Target variable name")
	evaluation_metric: str = Field(description="Primary evaluation metric")
	
	# Dataset configuration
	dataset_id: str = Field(description="Training dataset")
	feature_columns: List[str] = Field(default_factory=list)
	categorical_columns: List[str] = Field(default_factory=list)
	
	# AutoML configuration
	max_trials: int = Field(default=100, description="Maximum number of trials")
	max_training_time: timedelta = Field(default=timedelta(hours=24))
	
	# Algorithm selection
	algorithms_to_try: List[str] = Field(default_factory=list)
	hyperparameter_search_space: Dict[str, Any] = Field(default_factory=dict)
	
	# Experiment results
	best_model_id: Optional[str] = None
	best_score: Optional[Decimal] = None
	trials_completed: int = 0
	
	# Trial results
	trial_results: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Experiment status
	status: TrainingStatus = TrainingStatus.QUEUED
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	
	# Resource usage
	compute_resources_used: Dict[str, Any] = Field(default_factory=dict)
	
	created_by: str = Field(description="Experiment creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ModelEnsemble(BaseModel):
	"""Ensemble of multiple ML models"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	ensemble_name: str = Field(description="Ensemble name")
	ensemble_type: str = Field(description="Type of ensemble (voting, stacking, bagging)")
	
	# Member models
	member_models: List[str] = Field(default_factory=list, description="Model IDs in ensemble")
	model_weights: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Ensemble configuration
	combination_method: str = Field(description="How predictions are combined")
	meta_model_id: Optional[str] = None  # For stacking ensembles
	
	# Performance
	ensemble_accuracy: Optional[Decimal] = None
	member_accuracies: Dict[str, Decimal] = Field(default_factory=dict)
	diversity_score: Optional[Decimal] = None
	
	# Deployment
	status: ModelStatus = ModelStatus.TRAINING
	deployment_endpoint: Optional[str] = None
	
	created_by: str = Field(description="Ensemble creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None