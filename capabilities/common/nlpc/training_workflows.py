"""
APG NLP Model Training and Fine-tuning Workflows

Enterprise model training platform with automated workflows, hyperparameter optimization,
distributed training support, and integration with annotation data.

Features:
- Automated training pipeline with data preparation
- Hyperparameter optimization with grid/random/Bayesian search
- Distributed training coordination
- Model versioning and experiment tracking
- Integration with annotation workbench
- Production deployment automation
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Callable, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import tempfile
import shutil
from pathlib import Path
from uuid_extensions import uuid7str

from models import (
	NLPModel, ModelProvider, NLPTaskType, LanguageCode,
	ProcessingRequest, ProcessingResult
)

# Configure logging
logger = logging.getLogger(__name__)

class TrainingStatus(str, Enum):
	"""Training job status"""
	QUEUED = "queued"
	PREPARING = "preparing"
	TRAINING = "training"
	VALIDATING = "validating"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	DEPLOYED = "deployed"

# Alias for compatibility
ExperimentStatus = TrainingStatus

class OptimizationStrategy(str, Enum):
	"""Hyperparameter optimization strategies"""
	GRID_SEARCH = "grid_search"
	RANDOM_SEARCH = "random_search"
	BAYESIAN_OPTIMIZATION = "bayesian_optimization"
	GENETIC_ALGORITHM = "genetic_algorithm"
	EARLY_STOPPING = "early_stopping"

class ModelArchitecture(str, Enum):
	"""Supported model architectures"""
	TRANSFORMER = "transformer"
	BERT = "bert"
	ROBERTA = "roberta"
	DISTILBERT = "distilbert"
	LSTM = "lstm"
	CNN = "cnn"
	HYBRID = "hybrid"
	CUSTOM = "custom"

@dataclass
class TrainingDataset:
	"""Training dataset configuration"""
	dataset_id: str = field(default_factory=uuid7str)
	name: str = ""
	description: str = ""
	task_type: Optional[NLPTaskType] = None
	language: Optional[LanguageCode] = None
	
	# Data splits
	train_size: int = 0
	validation_size: int = 0
	test_size: int = 0
	
	# Data sources
	annotation_project_ids: List[str] = field(default_factory=list)
	external_datasets: List[str] = field(default_factory=list)
	synthetic_data_ratio: float = 0.0
	
	# Data characteristics
	label_distribution: Dict[str, int] = field(default_factory=dict)
	average_text_length: float = 0.0
	vocabulary_size: int = 0
	
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class HyperparameterSpace:
	"""Hyperparameter search space definition"""
	learning_rate: Dict[str, Any] = field(default_factory=lambda: {
		"type": "continuous",
		"min": 1e-6,
		"max": 1e-2,
		"scale": "log"
	})
	batch_size: Dict[str, Any] = field(default_factory=lambda: {
		"type": "categorical",
		"choices": [8, 16, 32, 64, 128]
	})
	num_epochs: Dict[str, Any] = field(default_factory=lambda: {
		"type": "integer",
		"min": 5,
		"max": 100
	})
	dropout_rate: Dict[str, Any] = field(default_factory=lambda: {
		"type": "continuous",
		"min": 0.0,
		"max": 0.5
	})
	weight_decay: Dict[str, Any] = field(default_factory=lambda: {
		"type": "continuous",
		"min": 0.0,
		"max": 0.1
	})

@dataclass
class TrainingExperiment:
	"""Training experiment tracking"""
	experiment_id: str = field(default_factory=uuid7str)
	name: str = ""
	description: str = ""
	
	# Configuration
	model_architecture: ModelArchitecture = ModelArchitecture.TRANSFORMER
	base_model: Optional[str] = None
	task_type: Optional[NLPTaskType] = None
	dataset_id: str = ""
	
	# Hyperparameters
	hyperparameters: Dict[str, Any] = field(default_factory=dict)
	optimization_strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH
	optimization_trials: int = 10
	
	# Training progress
	status: TrainingStatus = TrainingStatus.QUEUED
	current_epoch: int = 0
	total_epochs: int = 0
	
	# Metrics
	training_metrics: Dict[str, List[float]] = field(default_factory=dict)
	validation_metrics: Dict[str, List[float]] = field(default_factory=dict)
	test_metrics: Dict[str, float] = field(default_factory=dict)
	best_validation_score: float = 0.0
	
	# Resources
	gpu_hours_used: float = 0.0
	memory_peak_gb: float = 0.0
	compute_cost: float = 0.0
	
	# Timestamps
	created_at: datetime = field(default_factory=datetime.utcnow)
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	
	@property
	def duration_minutes(self) -> float:
		"""Calculate training duration in minutes"""
		if self.started_at and self.completed_at:
			return (self.completed_at - self.started_at).total_seconds() / 60
		elif self.started_at:
			return (datetime.utcnow() - self.started_at).total_seconds() / 60
		return 0.0
	
	@property
	def training_efficiency(self) -> float:
		"""Calculate training efficiency score"""
		if self.duration_minutes == 0:
			return 0.0
		
		# Higher score for better validation performance in less time
		time_factor = max(0, 1.0 - (self.duration_minutes / 1440))  # Normalize by 24 hours
		performance_factor = self.best_validation_score
		
		return (time_factor * 0.3) + (performance_factor * 0.7)

@dataclass
class ModelVersion:
	"""Model version tracking"""
	version_id: str = field(default_factory=uuid7str)
	model_name: str = ""
	version_number: str = "1.0.0"
	experiment_id: str = ""
	
	# Model artifacts
	model_path: str = ""
	tokenizer_path: str = ""
	config_path: str = ""
	metrics_path: str = ""
	
	# Performance
	validation_accuracy: float = 0.0
	test_accuracy: float = 0.0
	inference_latency_ms: float = 0.0
	model_size_mb: float = 0.0
	
	# Deployment
	deployment_status: str = "development"  # development, staging, production
	deployment_url: Optional[str] = None
	api_version: Optional[str] = None
	
	created_at: datetime = field(default_factory=datetime.utcnow)
	deployed_at: Optional[datetime] = None

class TrainingWorkflowManager:
	"""Enterprise training workflow orchestrator"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for training workflow manager"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Training state
		self.active_experiments: Dict[str, TrainingExperiment] = {}
		self.training_datasets: Dict[str, TrainingDataset] = {}
		self.model_versions: Dict[str, ModelVersion] = {}
		self.training_queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
		
		# Resource management
		self.gpu_resources: Dict[str, Dict[str, Any]] = {}
		self.training_workers: Dict[str, asyncio.Task] = {}
		self.resource_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
		
		# Optimization state
		self.hyperparameter_searches: Dict[str, Dict[str, Any]] = {}
		self.optimization_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
		
		self._setup_training_config()
		self._log_manager_initialized()
	
	def _setup_training_config(self) -> None:
		"""Setup training configuration"""
		self.max_concurrent_experiments = self.config.get("max_concurrent_experiments", 5)
		self.default_gpu_memory_gb = self.config.get("default_gpu_memory_gb", 16)
		self.model_storage_path = Path(self.config.get("model_storage_path", "/tmp/nlp_models"))
		self.enable_distributed_training = self.config.get("enable_distributed_training", False)
		self.auto_deployment = self.config.get("auto_deployment", False)
		self.cost_budget_per_experiment = self.config.get("cost_budget_per_experiment", 100.0)
		
		# Create storage directory
		self.model_storage_path.mkdir(parents=True, exist_ok=True)
	
	def _log_manager_initialized(self) -> None:
		"""Log manager initialization"""
		logger.info(f"Training workflow manager initialized for tenant: {self.tenant_id}")
		logger.info(f"Model storage path: {self.model_storage_path}")
	
	async def create_training_dataset(self, dataset_config: Dict[str, Any]) -> TrainingDataset:
		"""Create training dataset from annotation projects and external sources"""
		dataset = TrainingDataset(
			name=dataset_config["name"],
			description=dataset_config.get("description", ""),
			task_type=NLPTaskType(dataset_config["task_type"]) if "task_type" in dataset_config else None,
			language=LanguageCode(dataset_config["language"]) if "language" in dataset_config else None,
			annotation_project_ids=dataset_config.get("annotation_project_ids", []),
			external_datasets=dataset_config.get("external_datasets", []),
			synthetic_data_ratio=dataset_config.get("synthetic_data_ratio", 0.0)
		)
		
		# Prepare dataset
		await self._prepare_dataset(dataset, dataset_config)
		
		# Store dataset
		self.training_datasets[dataset.dataset_id] = dataset
		
		self._log_dataset_created(dataset.dataset_id, dataset.name)
		
		return dataset
	
	def _log_dataset_created(self, dataset_id: str, dataset_name: str) -> None:
		"""Log dataset creation"""
		logger.info(f"Training dataset created: {dataset_id} ({dataset_name})")
	
	async def _prepare_dataset(self, dataset: TrainingDataset, config: Dict[str, Any]) -> None:
		"""Prepare dataset from various sources"""
		# Mock dataset preparation
		# In real implementation, this would:
		# 1. Load data from annotation projects
		# 2. Merge with external datasets
		# 3. Generate synthetic data if needed
		# 4. Split into train/validation/test sets
		# 5. Compute statistics
		
		# Simulated data preparation
		total_size = config.get("expected_size", 10000)
		
		# Split ratios
		train_ratio = config.get("train_ratio", 0.8)
		val_ratio = config.get("val_ratio", 0.1)
		test_ratio = config.get("test_ratio", 0.1)
		
		dataset.train_size = int(total_size * train_ratio)
		dataset.validation_size = int(total_size * val_ratio)
		dataset.test_size = int(total_size * test_ratio)
		
		# Mock statistics
		dataset.label_distribution = config.get("label_distribution", {
			"positive": 4000,
			"negative": 3500,
			"neutral": 2500
		})
		dataset.average_text_length = config.get("average_text_length", 150.0)
		dataset.vocabulary_size = config.get("vocabulary_size", 50000)
		
		dataset.updated_at = datetime.utcnow()
	
	async def create_experiment(self, experiment_config: Dict[str, Any]) -> TrainingExperiment:
		"""Create new training experiment"""
		experiment = TrainingExperiment(
			name=experiment_config["name"],
			description=experiment_config.get("description", ""),
			model_architecture=ModelArchitecture(experiment_config.get("architecture", "transformer")),
			base_model=experiment_config.get("base_model"),
			task_type=NLPTaskType(experiment_config["task_type"]),
			dataset_id=experiment_config["dataset_id"],
			hyperparameters=experiment_config.get("hyperparameters", {}),
			optimization_strategy=OptimizationStrategy(experiment_config.get("optimization_strategy", "random_search")),
			optimization_trials=experiment_config.get("optimization_trials", 10),
			total_epochs=experiment_config.get("epochs", 50)
		)
		
		# Validate dataset exists
		if experiment.dataset_id not in self.training_datasets:
			raise ValueError(f"Dataset not found: {experiment.dataset_id}")
		
		# Store experiment
		self.active_experiments[experiment.experiment_id] = experiment
		
		# Add to training queue
		priority = experiment_config.get("priority", "normal")
		await self.training_queues[priority].put(experiment.experiment_id)
		
		self._log_experiment_created(experiment.experiment_id, experiment.name)
		
		return experiment
	
	def _log_experiment_created(self, experiment_id: str, experiment_name: str) -> None:
		"""Log experiment creation"""
		logger.info(f"Training experiment created: {experiment_id} ({experiment_name})")
	
	async def start_training_workers(self) -> None:
		"""Start training worker tasks"""
		for priority in ["high", "normal", "low"]:
			worker_id = f"worker_{priority}_{self.tenant_id}"
			if worker_id not in self.training_workers:
				self.training_workers[worker_id] = asyncio.create_task(
					self._training_worker(priority)
				)
		
		logger.info(f"Training workers started: {len(self.training_workers)}")
	
	async def _training_worker(self, priority: str) -> None:
		"""Training worker for specific priority queue"""
		logger.info(f"Training worker started for priority: {priority}")
		
		while True:
			try:
				# Get next experiment from queue
				experiment_id = await self.training_queues[priority].get()
				
				if experiment_id in self.active_experiments:
					experiment = self.active_experiments[experiment_id]
					
					# Check resource availability
					if len([e for e in self.active_experiments.values() 
							if e.status == TrainingStatus.TRAINING]) >= self.max_concurrent_experiments:
						# Re-queue if at capacity
						await self.training_queues[priority].put(experiment_id)
						await asyncio.sleep(60)  # Wait before checking again
						continue
					
					# Start training
					await self._execute_training(experiment)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Training worker error ({priority}): {e}")
				await asyncio.sleep(10)  # Brief pause on error
	
	async def _execute_training(self, experiment: TrainingExperiment) -> None:
		"""Execute training experiment"""
		experiment.status = TrainingStatus.PREPARING
		experiment.started_at = datetime.utcnow()
		
		try:
			# Prepare training environment
			await self._prepare_training_environment(experiment)
			
			# Hyperparameter optimization
			if experiment.optimization_strategy != OptimizationStrategy.EARLY_STOPPING:
				await self._optimize_hyperparameters(experiment)
			
			# Main training loop
			experiment.status = TrainingStatus.TRAINING
			await self._run_training_loop(experiment)
			
			# Validation
			experiment.status = TrainingStatus.VALIDATING
			await self._validate_model(experiment)
			
			# Create model version
			model_version = await self._create_model_version(experiment)
			
			# Auto-deployment if enabled
			if self.auto_deployment and model_version.validation_accuracy > 0.85:
				await self._deploy_model(model_version)
			
			experiment.status = TrainingStatus.COMPLETED
			experiment.completed_at = datetime.utcnow()
			
			self._log_training_completed(experiment.experiment_id, experiment.duration_minutes)
			
		except Exception as e:
			experiment.status = TrainingStatus.FAILED
			experiment.completed_at = datetime.utcnow()
			self._log_training_failed(experiment.experiment_id, str(e))
	
	def _log_training_completed(self, experiment_id: str, duration_minutes: float) -> None:
		"""Log training completion"""
		logger.info(f"Training completed: {experiment_id} (duration: {duration_minutes:.1f} minutes)")
	
	def _log_training_failed(self, experiment_id: str, error: str) -> None:
		"""Log training failure"""
		logger.error(f"Training failed: {experiment_id} - {error}")
	
	async def _prepare_training_environment(self, experiment: TrainingExperiment) -> None:
		"""Prepare training environment and resources"""
		# Mock environment preparation
		await asyncio.sleep(0.1)  # Simulate setup time
		
		# Initialize training metrics
		experiment.training_metrics = {
			"loss": [],
			"accuracy": [],
			"f1_score": [],
			"learning_rate": []
		}
		experiment.validation_metrics = {
			"val_loss": [],
			"val_accuracy": [],
			"val_f1_score": []
		}
	
	async def _optimize_hyperparameters(self, experiment: TrainingExperiment) -> None:
		"""Perform hyperparameter optimization"""
		if experiment.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
			await self._grid_search_optimization(experiment)
		elif experiment.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
			await self._random_search_optimization(experiment)
		elif experiment.optimization_strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
			await self._bayesian_optimization(experiment)
		else:
			# Use default hyperparameters
			experiment.hyperparameters = self._get_default_hyperparameters(experiment)
	
	async def _random_search_optimization(self, experiment: TrainingExperiment) -> None:
		"""Random search hyperparameter optimization"""
		import random
		
		search_space = HyperparameterSpace()
		best_score = 0.0
		best_params = {}
		
		for trial in range(experiment.optimization_trials):
			# Sample random hyperparameters
			trial_params = {}
			
			# Learning rate (log scale)
			trial_params["learning_rate"] = 10 ** random.uniform(-6, -2)
			
			# Batch size (categorical)
			trial_params["batch_size"] = random.choice([8, 16, 32, 64, 128])
			
			# Epochs (integer)
			trial_params["num_epochs"] = random.randint(5, 100)
			
			# Dropout rate
			trial_params["dropout_rate"] = random.uniform(0.0, 0.5)
			
			# Weight decay
			trial_params["weight_decay"] = random.uniform(0.0, 0.1)
			
			# Mock evaluation
			score = await self._evaluate_hyperparameters(experiment, trial_params)
			
			self.optimization_history[experiment.experiment_id].append({
				"trial": trial + 1,
				"parameters": trial_params,
				"score": score,
				"timestamp": datetime.utcnow()
			})
			
			if score > best_score:
				best_score = score
				best_params = trial_params
		
		experiment.hyperparameters = best_params
		experiment.best_validation_score = best_score
		
		logger.info(f"Hyperparameter optimization completed for {experiment.experiment_id}: best score {best_score:.4f}")
	
	async def _grid_search_optimization(self, experiment: TrainingExperiment) -> None:
		"""Grid search hyperparameter optimization"""
		# Define grid
		param_grid = {
			"learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
			"batch_size": [16, 32, 64],
			"dropout_rate": [0.1, 0.2, 0.3]
		}
		
		best_score = 0.0
		best_params = {}
		trial = 0
		
		# Grid search
		for lr in param_grid["learning_rate"]:
			for bs in param_grid["batch_size"]:
				for dr in param_grid["dropout_rate"]:
					if trial >= experiment.optimization_trials:
						break
					
					trial_params = {
						"learning_rate": lr,
						"batch_size": bs,
						"dropout_rate": dr,
						"num_epochs": 20,  # Fixed for grid search
						"weight_decay": 0.01
					}
					
					score = await self._evaluate_hyperparameters(experiment, trial_params)
					
					self.optimization_history[experiment.experiment_id].append({
						"trial": trial + 1,
						"parameters": trial_params,
						"score": score,
						"timestamp": datetime.utcnow()
					})
					
					if score > best_score:
						best_score = score
						best_params = trial_params
					
					trial += 1
		
		experiment.hyperparameters = best_params
		experiment.best_validation_score = best_score
	
	async def _bayesian_optimization(self, experiment: TrainingExperiment) -> None:
		"""Bayesian hyperparameter optimization (simplified)"""
		# Simplified Bayesian optimization using random sampling
		# In real implementation, would use libraries like Optuna or Hyperopt
		
		await self._random_search_optimization(experiment)
		logger.info(f"Bayesian optimization completed (using random search fallback)")
	
	async def _evaluate_hyperparameters(self, experiment: TrainingExperiment, 
										params: Dict[str, Any]) -> float:
		"""Evaluate hyperparameter configuration"""
		# Mock evaluation - simulate training with these parameters
		await asyncio.sleep(0.05)  # Simulate evaluation time
		
		# Mock score calculation based on reasonable heuristics
		base_score = 0.7
		
		# Learning rate effect
		lr = params["learning_rate"]
		if 1e-5 <= lr <= 5e-4:
			base_score += 0.1
		elif lr > 1e-3:
			base_score -= 0.15  # Too high
		
		# Batch size effect
		bs = params["batch_size"]
		if 16 <= bs <= 64:
			base_score += 0.05
		
		# Dropout effect
		dr = params.get("dropout_rate", 0.1)
		if 0.1 <= dr <= 0.3:
			base_score += 0.05
		elif dr > 0.4:
			base_score -= 0.1  # Too much regularization
		
		# Add some randomness
		import random
		base_score += random.uniform(-0.1, 0.1)
		
		return max(0.0, min(1.0, base_score))
	
	def _get_default_hyperparameters(self, experiment: TrainingExperiment) -> Dict[str, Any]:
		"""Get default hyperparameters for model architecture"""
		defaults = {
			"learning_rate": 5e-5,
			"batch_size": 32,
			"num_epochs": 20,
			"dropout_rate": 0.1,
			"weight_decay": 0.01,
			"warmup_steps": 500,
			"max_grad_norm": 1.0
		}
		
		# Architecture-specific adjustments
		if experiment.model_architecture == ModelArchitecture.BERT:
			defaults["learning_rate"] = 2e-5
			defaults["warmup_steps"] = 1000
		elif experiment.model_architecture == ModelArchitecture.LSTM:
			defaults["learning_rate"] = 1e-3
			defaults["batch_size"] = 64
		
		return defaults
	
	async def _run_training_loop(self, experiment: TrainingExperiment) -> None:
		"""Execute main training loop"""
		total_epochs = experiment.hyperparameters.get("num_epochs", experiment.total_epochs)
		
		for epoch in range(total_epochs):
			experiment.current_epoch = epoch + 1
			
			# Simulate training step
			await self._train_epoch(experiment, epoch)
			
			# Simulate validation step
			val_score = await self._validate_epoch(experiment, epoch)
			
			# Update best validation score
			if val_score > experiment.best_validation_score:
				experiment.best_validation_score = val_score
			
			# Early stopping check
			if await self._should_early_stop(experiment, epoch):
				logger.info(f"Early stopping triggered for {experiment.experiment_id} at epoch {epoch + 1}")
				break
			
			# Progress logging
			if (epoch + 1) % 5 == 0:
				logger.info(f"Training progress {experiment.experiment_id}: epoch {epoch + 1}/{total_epochs}, val_score: {val_score:.4f}")
	
	async def _train_epoch(self, experiment: TrainingExperiment, epoch: int) -> None:
		"""Simulate training epoch"""
		await asyncio.sleep(0.01)  # Simulate training time
		
		# Mock training metrics
		import random
		
		# Loss should generally decrease over time
		loss = 2.0 * (0.9 ** epoch) + random.uniform(-0.1, 0.1)
		accuracy = min(0.95, 0.5 + (epoch * 0.02) + random.uniform(-0.02, 0.02))
		f1_score = min(0.93, 0.45 + (epoch * 0.025) + random.uniform(-0.02, 0.02))
		
		experiment.training_metrics["loss"].append(loss)
		experiment.training_metrics["accuracy"].append(accuracy)
		experiment.training_metrics["f1_score"].append(f1_score)
		experiment.training_metrics["learning_rate"].append(experiment.hyperparameters.get("learning_rate", 5e-5))
	
	async def _validate_epoch(self, experiment: TrainingExperiment, epoch: int) -> float:
		"""Simulate validation epoch"""
		await asyncio.sleep(0.005)  # Simulate validation time
		
		import random
		
		# Validation metrics (slightly worse than training)
		val_loss = experiment.training_metrics["loss"][-1] + 0.1 + random.uniform(-0.05, 0.05)
		val_accuracy = max(0.0, experiment.training_metrics["accuracy"][-1] - 0.02 + random.uniform(-0.02, 0.02))
		val_f1_score = max(0.0, experiment.training_metrics["f1_score"][-1] - 0.02 + random.uniform(-0.02, 0.02))
		
		experiment.validation_metrics["val_loss"].append(val_loss)
		experiment.validation_metrics["val_accuracy"].append(val_accuracy)
		experiment.validation_metrics["val_f1_score"].append(val_f1_score)
		
		return val_f1_score  # Use F1 score as main validation metric
	
	async def _should_early_stop(self, experiment: TrainingExperiment, epoch: int) -> bool:
		"""Check if training should stop early"""
		if len(experiment.validation_metrics["val_f1_score"]) < 5:
			return False  # Need at least 5 epochs
		
		# Check if validation score has not improved in last 5 epochs
		recent_scores = experiment.validation_metrics["val_f1_score"][-5:]
		max_recent = max(recent_scores)
		current_score = recent_scores[-1]
		
		# Stop if no improvement and current score is declining
		if max_recent > current_score + 0.01:
			declining_count = sum(
				1 for i in range(1, len(recent_scores))
				if recent_scores[i] <= recent_scores[i-1]
			)
			return declining_count >= 3
		
		return False
	
	async def _validate_model(self, experiment: TrainingExperiment) -> None:
		"""Final model validation on test set"""
		await asyncio.sleep(0.02)  # Simulate test evaluation
		
		import random
		
		# Test metrics (should be similar to best validation)
		test_accuracy = experiment.best_validation_score + random.uniform(-0.03, 0.01)
		test_f1_score = experiment.best_validation_score + random.uniform(-0.02, 0.01)
		test_precision = test_f1_score + random.uniform(-0.02, 0.02)
		test_recall = test_f1_score + random.uniform(-0.02, 0.02)
		
		experiment.test_metrics = {
			"test_accuracy": max(0.0, test_accuracy),
			"test_f1_score": max(0.0, test_f1_score),
			"test_precision": max(0.0, test_precision),
			"test_recall": max(0.0, test_recall)
		}
	
	async def _create_model_version(self, experiment: TrainingExperiment) -> ModelVersion:
		"""Create model version from completed experiment"""
		# Create version directory
		version_dir = self.model_storage_path / experiment.experiment_id
		version_dir.mkdir(parents=True, exist_ok=True)
		
		# Mock model artifact paths
		model_version = ModelVersion(
			model_name=f"{experiment.name}_model",
			version_number=self._generate_version_number(experiment.name),
			experiment_id=experiment.experiment_id,
			model_path=str(version_dir / "model.pkl"),
			tokenizer_path=str(version_dir / "tokenizer.pkl"),
			config_path=str(version_dir / "config.json"),
			metrics_path=str(version_dir / "metrics.json"),
			validation_accuracy=experiment.best_validation_score,
			test_accuracy=experiment.test_metrics.get("test_accuracy", 0.0),
			inference_latency_ms=50.0 + (experiment.best_validation_score * 10),  # Mock latency
			model_size_mb=100.0 + (hash(experiment.experiment_id) % 400)  # Mock size
		)
		
		# Save model artifacts (mock)
		await self._save_model_artifacts(model_version, experiment)
		
		# Store version
		self.model_versions[model_version.version_id] = model_version
		
		self._log_model_version_created(model_version.version_id, model_version.model_name)
		
		return model_version
	
	def _generate_version_number(self, model_name: str) -> str:
		"""Generate semantic version number for model"""
		existing_versions = [
			v for v in self.model_versions.values()
			if v.model_name.startswith(model_name)
		]
		
		major = 1
		minor = len(existing_versions)
		patch = 0
		
		return f"{major}.{minor}.{patch}"
	
	async def _save_model_artifacts(self, model_version: ModelVersion, experiment: TrainingExperiment) -> None:
		"""Save model artifacts to storage"""
		# Save model config
		config_data = {
			"experiment_id": experiment.experiment_id,
			"model_architecture": experiment.model_architecture.value,
			"hyperparameters": experiment.hyperparameters,
			"task_type": experiment.task_type.value if experiment.task_type else None,
			"training_metrics": experiment.training_metrics,
			"validation_metrics": experiment.validation_metrics,
			"test_metrics": experiment.test_metrics
		}
		
		with open(model_version.config_path, 'w') as f:
			json.dump(config_data, f, indent=2, default=str)
		
		# Save metrics summary
		metrics_data = {
			"validation_accuracy": model_version.validation_accuracy,
			"test_accuracy": model_version.test_accuracy,
			"inference_latency_ms": model_version.inference_latency_ms,
			"model_size_mb": model_version.model_size_mb,
			"training_duration_minutes": experiment.duration_minutes,
			"training_efficiency": experiment.training_efficiency
		}
		
		with open(model_version.metrics_path, 'w') as f:
			json.dump(metrics_data, f, indent=2)
		
		# Mock model and tokenizer files (would be actual model artifacts in real implementation)
		with open(model_version.model_path, 'wb') as f:
			pickle.dump({"mock_model": True}, f)
		
		with open(model_version.tokenizer_path, 'wb') as f:
			pickle.dump({"mock_tokenizer": True}, f)
	
	def _log_model_version_created(self, version_id: str, model_name: str) -> None:
		"""Log model version creation"""
		logger.info(f"Model version created: {version_id} ({model_name})")
	
	async def _deploy_model(self, model_version: ModelVersion) -> None:
		"""Deploy model to production environment"""
		# Mock deployment
		model_version.deployment_status = "production"
		model_version.deployment_url = f"https://api.nlp.example.com/models/{model_version.version_id}"
		model_version.api_version = "v1"
		model_version.deployed_at = datetime.utcnow()
		
		logger.info(f"Model deployed: {model_version.version_id} at {model_version.deployment_url}")
	
	def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
		"""Get detailed experiment status"""
		if experiment_id not in self.active_experiments:
			return {}
		
		experiment = self.active_experiments[experiment_id]
		dataset = self.training_datasets.get(experiment.dataset_id, None)
		
		# Calculate progress
		progress_percent = 0.0
		if experiment.total_epochs > 0:
			progress_percent = (experiment.current_epoch / experiment.total_epochs) * 100
		
		return {
			"experiment_id": experiment_id,
			"name": experiment.name,
			"status": experiment.status.value,
			"progress_percent": round(progress_percent, 1),
			"current_epoch": experiment.current_epoch,
			"total_epochs": experiment.total_epochs,
			"best_validation_score": round(experiment.best_validation_score, 4),
			"training_efficiency": round(experiment.training_efficiency, 3),
			"duration_minutes": round(experiment.duration_minutes, 1),
			"gpu_hours_used": experiment.gpu_hours_used,
			"compute_cost": experiment.compute_cost,
			"dataset_info": {
				"dataset_id": dataset.dataset_id if dataset else None,
				"dataset_name": dataset.name if dataset else None,
				"train_size": dataset.train_size if dataset else 0
			},
			"optimization": {
				"strategy": experiment.optimization_strategy.value,
				"trials_completed": len(self.optimization_history.get(experiment_id, [])),
				"best_hyperparameters": experiment.hyperparameters
			},
			"timestamps": {
				"created_at": experiment.created_at.isoformat(),
				"started_at": experiment.started_at.isoformat() if experiment.started_at else None,
				"completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None
			}
		}
	
	def get_training_dashboard(self) -> Dict[str, Any]:
		"""Get comprehensive training dashboard data"""
		# Active experiments
		active_experiments = [
			{
				"experiment_id": exp.experiment_id,
				"name": exp.name,
				"status": exp.status.value,
				"progress": (exp.current_epoch / max(exp.total_epochs, 1)) * 100,
				"best_score": exp.best_validation_score
			}
			for exp in self.active_experiments.values()
		]
		
		# Recent completions
		completed_experiments = [
			exp for exp in self.active_experiments.values()
			if exp.status == TrainingStatus.COMPLETED and exp.completed_at
		]
		completed_experiments.sort(key=lambda x: x.completed_at, reverse=True)
		recent_completions = completed_experiments[:10]
		
		# Model versions
		model_versions = list(self.model_versions.values())
		model_versions.sort(key=lambda x: x.created_at, reverse=True)
		
		# Resource usage
		total_gpu_hours = sum(exp.gpu_hours_used for exp in self.active_experiments.values())
		total_compute_cost = sum(exp.compute_cost for exp in self.active_experiments.values())
		
		return {
			"tenant_id": self.tenant_id,
			"timestamp": datetime.utcnow().isoformat(),
			"summary": {
				"total_experiments": len(self.active_experiments),
				"active_experiments": len([e for e in self.active_experiments.values() if e.status == TrainingStatus.TRAINING]),
				"completed_experiments": len([e for e in self.active_experiments.values() if e.status == TrainingStatus.COMPLETED]),
				"failed_experiments": len([e for e in self.active_experiments.values() if e.status == TrainingStatus.FAILED]),
				"total_datasets": len(self.training_datasets),
				"total_model_versions": len(self.model_versions),
				"deployed_models": len([v for v in self.model_versions.values() if v.deployment_status == "production"])
			},
			"active_experiments": active_experiments[:20],  # Limit to most recent
			"recent_completions": [
				{
					"experiment_id": exp.experiment_id,
					"name": exp.name,
					"best_score": exp.best_validation_score,
					"duration_minutes": exp.duration_minutes,
					"efficiency": exp.training_efficiency,
					"completed_at": exp.completed_at.isoformat()
				}
				for exp in recent_completions
			],
			"model_versions": [
				{
					"version_id": v.version_id,
					"model_name": v.model_name,
					"version_number": v.version_number,
					"validation_accuracy": v.validation_accuracy,
					"deployment_status": v.deployment_status,
					"created_at": v.created_at.isoformat()
				}
				for v in model_versions[:20]
			],
			"resource_usage": {
				"total_gpu_hours": round(total_gpu_hours, 2),
				"total_compute_cost": round(total_compute_cost, 2),
				"active_workers": len(self.training_workers)
			}
		}
	
	async def stop_training_workers(self) -> None:
		"""Stop all training workers"""
		for worker_id, worker_task in self.training_workers.items():
			worker_task.cancel()
			try:
				await worker_task
			except asyncio.CancelledError:
				pass
		
		self.training_workers.clear()
		logger.info("All training workers stopped")
	
	async def cleanup(self) -> None:
		"""Cleanup training workflow manager resources"""
		# Stop workers
		await self.stop_training_workers()
		
		# Clear state
		self.active_experiments.clear()
		self.training_datasets.clear()
		self.model_versions.clear()
		self.training_queues.clear()
		self.gpu_resources.clear()
		self.hyperparameter_searches.clear()
		self.optimization_history.clear()
		
		logger.info(f"Training workflow manager cleanup completed for tenant: {self.tenant_id}")

# Export main classes
__all__ = [
	"TrainingWorkflowManager", "TrainingDataset", "TrainingExperiment", 
	"ModelVersion", "HyperparameterSpace", "TrainingStatus", "ExperimentStatus",
	"OptimizationStrategy", "ModelArchitecture", "HyperparameterOptimizer", "ModelRepository"
]