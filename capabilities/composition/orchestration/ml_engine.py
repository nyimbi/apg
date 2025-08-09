"""
APG Workflow Machine Learning Engine

Advanced ML integration for workflow orchestration including deep learning,
reinforcement learning, federated learning, and AutoML capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import pickle
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from uuid_extensions import uuid7str

# ML Libraries
try:
	import torch
	import torch.nn as nn
	import torch.optim as optim
	from torch.utils.data import DataLoader, Dataset
	import torch.distributed as dist
	from torch.nn.parallel import DistributedDataParallel as DDP
	TORCH_AVAILABLE = True
except ImportError:
	TORCH_AVAILABLE = False

try:
	import tensorflow as tf
	from tensorflow import keras
	TF_AVAILABLE = True
except ImportError:
	TF_AVAILABLE = False

try:
	import sklearn
	from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
	from sklearn.model_selection import train_test_split, GridSearchCV
	from sklearn.preprocessing import StandardScaler, LabelEncoder
	from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
	SKLEARN_AVAILABLE = True
except ImportError:
	SKLEARN_AVAILABLE = False

try:
	import xgboost as xgb
	XGB_AVAILABLE = True
except ImportError:
	XGB_AVAILABLE = False

try:
	from autogluon.tabular import TabularPredictor
	AUTOGLUON_AVAILABLE = True
except ImportError:
	AUTOGLUON_AVAILABLE = False

from .models import WorkflowTemplate, WorkflowExecution, WorkflowNode

logger = logging.getLogger(__name__)

@dataclass
class MLModelMetrics:
	"""ML model performance metrics."""
	accuracy: float = 0.0
	precision: float = 0.0
	recall: float = 0.0
	f1_score: float = 0.0
	mse: float = 0.0
	mae: float = 0.0
	r2_score: float = 0.0
	loss: float = 0.0
	training_time: float = 0.0
	inference_time: float = 0.0
	model_size_mb: float = 0.0
	timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class FederatedLearningRound:
	"""Federated learning round information."""
	round_id: str = field(default_factory=uuid7str)
	client_count: int = 0
	aggregation_method: str = "federated_averaging"
	global_model_version: int = 0
	client_updates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	aggregated_weights: Optional[Dict[str, Any]] = None
	round_metrics: MLModelMetrics = field(default_factory=MLModelMetrics)
	started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	completed_at: Optional[datetime] = None

class MLModelType(str):
	"""ML model types supported."""
	DEEP_LEARNING = "deep_learning"
	REINFORCEMENT_LEARNING = "reinforcement_learning"
	FEDERATED_LEARNING = "federated_learning"
	AUTOML = "automl"
	ENSEMBLE = "ensemble"
	TRADITIONAL_ML = "traditional_ml"

class MLTask(BaseModel):
	"""ML task configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	task_id: str = Field(default_factory=uuid7str, description="Task ID")
	task_type: str = Field(..., description="Task type (classification, regression, clustering, etc.)")
	model_type: MLModelType = Field(..., description="ML model type")
	
	# Data configuration
	dataset_path: str = Field(..., description="Path to training dataset")
	target_column: str = Field(..., description="Target column name")
	feature_columns: List[str] = Field(default_factory=list, description="Feature column names")
	
	# Model configuration
	model_config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
	hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
	
	# Training configuration
	batch_size: int = Field(default=32, description="Training batch size")
	epochs: int = Field(default=100, description="Training epochs")
	learning_rate: float = Field(default=0.001, description="Learning rate")
	validation_split: float = Field(default=0.2, description="Validation split ratio")
	
	# Metadata
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	status: str = Field(default="pending", description="Task status")
	metrics: Optional[MLModelMetrics] = Field(default=None, description="Training metrics")

class AutoMLConfig(BaseModel):
	"""AutoML configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	time_limit: int = Field(default=3600, description="Time limit in seconds")
	quality_preset: str = Field(default="medium_quality", description="Quality preset")
	eval_metric: str = Field(default="auto", description="Evaluation metric")
	include_models: List[str] = Field(default_factory=list, description="Models to include")
	exclude_models: List[str] = Field(default_factory=list, description="Models to exclude")
	presets: List[str] = Field(default_factory=list, description="AutoML presets")
	
	# Advanced options
	ensemble_size: int = Field(default=10, description="Ensemble size")
	num_bag_folds: int = Field(default=8, description="Number of bagging folds")
	num_stack_levels: int = Field(default=1, description="Number of stacking levels")
	auto_stack: bool = Field(default=True, description="Enable auto stacking")

class ReinforcementLearningConfig(BaseModel):
	"""Reinforcement learning configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	algorithm: str = Field(default="dqn", description="RL algorithm (dqn, ppo, a3c, etc.)")
	environment_type: str = Field(default="custom", description="Environment type")
	
	# Network architecture
	network_layers: List[int] = Field(default_factory=lambda: [256, 256], description="Network layers")
	activation_function: str = Field(default="relu", description="Activation function")
	
	# Training parameters
	episodes: int = Field(default=1000, description="Training episodes")
	max_steps_per_episode: int = Field(default=1000, description="Max steps per episode")
	exploration_rate: float = Field(default=1.0, description="Initial exploration rate")
	exploration_decay: float = Field(default=0.995, description="Exploration decay rate")
	min_exploration_rate: float = Field(default=0.01, description="Minimum exploration rate")
	discount_factor: float = Field(default=0.99, description="Discount factor (gamma)")
	
	# Memory and replay
	memory_size: int = Field(default=10000, description="Replay memory size")
	batch_size: int = Field(default=32, description="Batch size for training")
	target_update_frequency: int = Field(default=100, description="Target network update frequency")

class FederatedLearningConfig(BaseModel):
	"""Federated learning configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	aggregation_method: str = Field(default="federated_averaging", description="Aggregation method")
	min_clients_per_round: int = Field(default=2, description="Minimum clients per round")
	max_clients_per_round: int = Field(default=10, description="Maximum clients per round")
	rounds: int = Field(default=100, description="Total federated learning rounds")
	
	# Client configuration
	client_epochs: int = Field(default=5, description="Local epochs per client")
	client_batch_size: int = Field(default=32, description="Client batch size")
	client_learning_rate: float = Field(default=0.01, description="Client learning rate")
	
	# Privacy and security
	differential_privacy: bool = Field(default=False, description="Enable differential privacy")
	noise_multiplier: float = Field(default=1.0, description="Noise multiplier for DP")
	max_grad_norm: float = Field(default=1.0, description="Maximum gradient norm")
	secure_aggregation: bool = Field(default=False, description="Enable secure aggregation")

class MLEngine:
	"""
	Advanced Machine Learning Engine for workflow orchestration.
	
	Features:
	- Deep Learning with PyTorch/TensorFlow
	- Reinforcement Learning for workflow optimization
	- Federated Learning for distributed training
	- AutoML for automated model selection
	- Model lifecycle management
	- Distributed training support
	"""
	
	def __init__(self, config: Dict[str, Any], db_session: Session):
		self.config = config
		self.db_session = db_session
		self.models: Dict[str, Any] = {}
		self.training_tasks: Dict[str, MLTask] = {}
		self.federated_rounds: Dict[str, FederatedLearningRound] = {}
		
		# Performance optimization
		self.executor = ThreadPoolExecutor(max_workers=4)
		self.model_cache: Dict[str, Any] = {}
		self.cache_lock = threading.Lock()
		
		# Distributed training support
		self.distributed_enabled = False
		self.rank = 0
		self.world_size = 1
		
		logger.info("ML Engine initialized")
	
	async def initialize(self) -> None:
		"""Initialize the ML engine."""
		try:
			# Check library availability
			self._check_ml_libraries()
			
			# Initialize distributed training if configured
			if self.config.get('distributed_training', {}).get('enabled', False):
				await self._initialize_distributed_training()
			
			# Load pre-trained models
			await self._load_pretrained_models()
			
			logger.info("ML Engine initialization completed")
		except Exception as e:
			logger.error(f"Failed to initialize ML engine: {e}")
			raise
	
	def _check_ml_libraries(self) -> None:
		"""Check availability of ML libraries."""
		available_libs = []
		if TORCH_AVAILABLE:
			available_libs.append("PyTorch")
		if TF_AVAILABLE:
			available_libs.append("TensorFlow")
		if SKLEARN_AVAILABLE:
			available_libs.append("scikit-learn")
		if XGB_AVAILABLE:
			available_libs.append("XGBoost")
		if AUTOGLUON_AVAILABLE:
			available_libs.append("AutoGluon")
		
		logger.info(f"Available ML libraries: {', '.join(available_libs)}")
		
		if not available_libs:
			logger.warning("No ML libraries available. Some features may be limited.")
	
	async def _initialize_distributed_training(self) -> None:
		"""Initialize distributed training."""
		try:
			if TORCH_AVAILABLE:
				dist_config = self.config.get('distributed_training', {})
				backend = dist_config.get('backend', 'nccl')
				init_method = dist_config.get('init_method', 'env://')
				
				dist.init_process_group(backend=backend, init_method=init_method)
				self.rank = dist.get_rank()
				self.world_size = dist.get_world_size()
				self.distributed_enabled = True
				
				logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")
		except Exception as e:
			logger.warning(f"Failed to initialize distributed training: {e}")
	
	async def _load_pretrained_models(self) -> None:
		"""Load pre-trained models."""
		try:
			models_dir = Path(self.config.get('models_directory', './models'))
			if models_dir.exists():
				for model_file in models_dir.glob('*.pkl'):
					try:
						with open(model_file, 'rb') as f:
							model = pickle.load(f)
						model_name = model_file.stem
						self.models[model_name] = model
						logger.info(f"Loaded pre-trained model: {model_name}")
					except Exception as e:
						logger.warning(f"Failed to load model {model_file}: {e}")
		except Exception as e:
			logger.error(f"Failed to load pre-trained models: {e}")
	
	async def train_deep_learning_model(self, task: MLTask) -> MLModelMetrics:
		"""Train a deep learning model."""
		if not TORCH_AVAILABLE and not TF_AVAILABLE:
			raise ValueError("No deep learning framework available")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Load and prepare data
			X_train, X_val, y_train, y_val = await self._prepare_training_data(task)
			
			# Choose framework
			framework = task.model_config.get('framework', 'pytorch')
			
			if framework == 'pytorch' and TORCH_AVAILABLE:
				model, metrics = await self._train_pytorch_model(task, X_train, X_val, y_train, y_val)
			elif framework == 'tensorflow' and TF_AVAILABLE:
				model, metrics = await self._train_tensorflow_model(task, X_train, X_val, y_train, y_val)
			else:
				raise ValueError(f"Framework {framework} not available")
			
			# Save model
			model_path = await self._save_model(task.task_id, model, framework)
			
			# Update metrics
			training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
			metrics.training_time = training_time
			metrics.model_size_mb = self._get_model_size(model_path)
			
			# Cache model
			with self.cache_lock:
				self.model_cache[task.task_id] = model
			
			task.status = "completed"
			task.metrics = metrics
			
			logger.info(f"Deep learning model training completed: {task.task_id}")
			return metrics
			
		except Exception as e:
			task.status = "failed"
			logger.error(f"Deep learning training failed: {e}")
			raise
	
	async def _train_pytorch_model(self, task: MLTask, X_train, X_val, y_train, y_val) -> Tuple[Any, MLModelMetrics]:
		"""Train PyTorch model."""
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		# Create model
		input_size = X_train.shape[1]
		hidden_sizes = task.model_config.get('hidden_sizes', [256, 128, 64])
		output_size = len(np.unique(y_train)) if task.task_type == 'classification' else 1
		dropout = task.model_config.get('dropout', 0.2)
		
		class MLPModel(nn.Module):
			def __init__(self, input_size, hidden_sizes, output_size, dropout):
				super().__init__()
				layers = []
				prev_size = input_size
				
				for hidden_size in hidden_sizes:
					layers.extend([
						nn.Linear(prev_size, hidden_size),
						nn.ReLU(),
						nn.Dropout(dropout)
					])
					prev_size = hidden_size
				
				layers.append(nn.Linear(prev_size, output_size))
				if task.task_type == 'classification' and output_size > 1:
					layers.append(nn.Softmax(dim=1))
				
				self.network = nn.Sequential(*layers)
			
			def forward(self, x):
				return self.network(x)
		
		model = MLPModel(input_size, hidden_sizes, output_size, dropout).to(device)
		
		# Distributed training setup
		if self.distributed_enabled and torch.cuda.device_count() > 1:
			model = DDP(model, device_ids=[self.rank])
		
		# Loss function and optimizer
		if task.task_type == 'classification':
			criterion = nn.CrossEntropyLoss()
		else:
			criterion = nn.MSELoss()
		
		optimizer = optim.Adam(model.parameters(), lr=task.learning_rate)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
		
		# Convert to tensors
		X_train_tensor = torch.FloatTensor(X_train).to(device)
		y_train_tensor = torch.LongTensor(y_train) if task.task_type == 'classification' else torch.FloatTensor(y_train)
		y_train_tensor = y_train_tensor.to(device)
		
		X_val_tensor = torch.FloatTensor(X_val).to(device)
		y_val_tensor = torch.LongTensor(y_val) if task.task_type == 'classification' else torch.FloatTensor(y_val)
		y_val_tensor = y_val_tensor.to(device)
		
		# Training loop
		train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
		train_loader = DataLoader(train_dataset, batch_size=task.batch_size, shuffle=True)
		
		best_val_loss = float('inf')
		early_stopping_patience = 20
		early_stopping_counter = 0
		
		for epoch in range(task.epochs):
			model.train()
			epoch_loss = 0.0
			
			for batch_X, batch_y in train_loader:
				optimizer.zero_grad()
				outputs = model(batch_X)
				
				if task.task_type == 'classification':
					loss = criterion(outputs, batch_y)
				else:
					loss = criterion(outputs.squeeze(), batch_y)
				
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
			
			# Validation
			model.eval()
			with torch.no_grad():
				val_outputs = model(X_val_tensor)
				if task.task_type == 'classification':
					val_loss = criterion(val_outputs, y_val_tensor)
				else:
					val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
			
			scheduler.step(val_loss)
			
			# Early stopping
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				early_stopping_counter = 0
			else:
				early_stopping_counter += 1
				if early_stopping_counter >= early_stopping_patience:
					logger.info(f"Early stopping at epoch {epoch}")
					break
		
		# Calculate metrics
		model.eval()
		with torch.no_grad():
			val_outputs = model(X_val_tensor)
			if task.task_type == 'classification':
				val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
				accuracy = accuracy_score(y_val, val_predictions)
				f1 = f1_score(y_val, val_predictions, average='weighted')
				metrics = MLModelMetrics(accuracy=accuracy, f1_score=f1, loss=best_val_loss.item())
			else:
				val_predictions = val_outputs.squeeze().cpu().numpy()
				mse = mean_squared_error(y_val, val_predictions)
				metrics = MLModelMetrics(mse=mse, loss=best_val_loss.item())
		
		return model, metrics
	
	async def _train_tensorflow_model(self, task: MLTask, X_train, X_val, y_train, y_val) -> Tuple[Any, MLModelMetrics]:
		"""Train TensorFlow model."""
		# Create model
		input_shape = (X_train.shape[1],)
		hidden_sizes = task.model_config.get('hidden_sizes', [256, 128, 64])
		dropout = task.model_config.get('dropout', 0.2)
		
		model = keras.Sequential()
		model.add(keras.layers.Input(shape=input_shape))
		
		for hidden_size in hidden_sizes:
			model.add(keras.layers.Dense(hidden_size, activation='relu'))
			model.add(keras.layers.Dropout(dropout))
		
		if task.task_type == 'classification':
			output_size = len(np.unique(y_train))
			model.add(keras.layers.Dense(output_size, activation='softmax'))
			model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		else:
			model.add(keras.layers.Dense(1))
			model.compile(optimizer='adam', loss='mse', metrics=['mae'])
		
		# Callbacks
		early_stopping = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
		reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=10)
		
		# Training
		history = model.fit(
			X_train, y_train,
			validation_data=(X_val, y_val),
			epochs=task.epochs,
			batch_size=task.batch_size,
			callbacks=[early_stopping, reduce_lr],
			verbose=0
		)
		
		# Calculate metrics
		val_loss = min(history.history['val_loss'])
		if task.task_type == 'classification':
			val_accuracy = max(history.history['val_accuracy'])
			metrics = MLModelMetrics(accuracy=val_accuracy, loss=val_loss)
		else:
			val_mae = min(history.history['val_mae'])
			metrics = MLModelMetrics(mae=val_mae, mse=val_loss)
		
		return model, metrics
	
	async def train_reinforcement_learning_model(self, task: MLTask, rl_config: ReinforcementLearningConfig) -> MLModelMetrics:
		"""Train reinforcement learning model."""
		if not TORCH_AVAILABLE:
			raise ValueError("PyTorch is required for reinforcement learning")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Create RL environment (simplified workflow optimization environment)
			env = await self._create_rl_environment(task)
			
			# Initialize RL agent
			agent = await self._create_rl_agent(env, rl_config)
			
			# Training loop
			total_rewards = []
			for episode in range(rl_config.episodes):
				state = env.reset()
				episode_reward = 0
				
				for step in range(rl_config.max_steps_per_episode):
					action = agent.choose_action(state, rl_config.exploration_rate)
					next_state, reward, done, _ = env.step(action)
					
					agent.store_experience(state, action, reward, next_state, done)
					agent.train()
					
					state = next_state
					episode_reward += reward
					
					if done:
						break
				
				total_rewards.append(episode_reward)
				
				# Decay exploration rate
				rl_config.exploration_rate = max(
					rl_config.min_exploration_rate,
					rl_config.exploration_rate * rl_config.exploration_decay
				)
				
				# Update target network
				if episode % rl_config.target_update_frequency == 0:
					agent.update_target_network()
			
			# Save model
			model_path = await self._save_model(task.task_id, agent.q_network, 'pytorch')
			
			# Calculate metrics
			avg_reward = np.mean(total_rewards[-100:])  # Last 100 episodes
			training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
			
			metrics = MLModelMetrics(
				accuracy=avg_reward,  # Use reward as performance metric
				training_time=training_time,
				model_size_mb=self._get_model_size(model_path)
			)
			
			# Cache model
			with self.cache_lock:
				self.model_cache[task.task_id] = agent
			
			task.status = "completed"
			task.metrics = metrics
			
			logger.info(f"RL model training completed: {task.task_id}")
			return metrics
			
		except Exception as e:
			task.status = "failed"
			logger.error(f"RL training failed: {e}")
			raise
	
	async def train_federated_learning_model(self, task: MLTask, fl_config: FederatedLearningConfig, client_data: Dict[str, Any]) -> MLModelMetrics:
		"""Train federated learning model."""
		if not TORCH_AVAILABLE:
			raise ValueError("PyTorch is required for federated learning")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Initialize global model
			global_model = await self._create_global_model(task)
			
			best_metrics = MLModelMetrics()
			
			# Federated learning rounds
			for round_num in range(fl_config.rounds):
				fl_round = FederatedLearningRound(
					round_id=f"{task.task_id}_round_{round_num}",
					global_model_version=round_num,
					aggregation_method=fl_config.aggregation_method
				)
				
				# Select clients for this round
				available_clients = list(client_data.keys())
				selected_clients = np.random.choice(
					available_clients,
					size=min(fl_config.max_clients_per_round, len(available_clients)),
					replace=False
				)
				
				fl_round.client_count = len(selected_clients)
				
				# Client training
				client_updates = {}
				for client_id in selected_clients:
					update = await self._train_federated_client(
						client_id, global_model, client_data[client_id], fl_config
					)
					client_updates[client_id] = update
					fl_round.client_updates[client_id] = {
						'weights_hash': hashlib.md5(str(update).encode()).hexdigest()[:16],
						'samples_count': len(client_data[client_id])
					}
				
				# Aggregate updates
				fl_round.aggregated_weights = await self._aggregate_federated_updates(
					client_updates, fl_config.aggregation_method
				)
				
				# Update global model
				global_model.load_state_dict(fl_round.aggregated_weights)
				
				# Evaluate global model
				round_metrics = await self._evaluate_federated_model(global_model, task)
				fl_round.round_metrics = round_metrics
				fl_round.completed_at = datetime.now(timezone.utc)
				
				self.federated_rounds[fl_round.round_id] = fl_round
				
				# Track best metrics
				if round_metrics.accuracy > best_metrics.accuracy:
					best_metrics = round_metrics
				
				logger.info(f"FL Round {round_num} completed: accuracy={round_metrics.accuracy:.4f}")
			
			# Save final model
			model_path = await self._save_model(task.task_id, global_model, 'pytorch')
			
			# Update metrics
			training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
			best_metrics.training_time = training_time
			best_metrics.model_size_mb = self._get_model_size(model_path)
			
			# Cache model
			with self.cache_lock:
				self.model_cache[task.task_id] = global_model
			
			task.status = "completed"
			task.metrics = best_metrics
			
			logger.info(f"Federated learning completed: {task.task_id}")
			return best_metrics
			
		except Exception as e:
			task.status = "failed"
			logger.error(f"Federated learning failed: {e}")
			raise
	
	async def train_automl_model(self, task: MLTask, automl_config: AutoMLConfig) -> MLModelMetrics:
		"""Train AutoML model."""
		if not AUTOGLUON_AVAILABLE:
			raise ValueError("AutoGluon is required for AutoML")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Load and prepare data
			X_train, X_val, y_train, y_val = await self._prepare_training_data(task)
			
			# Combine features and target for AutoGluon
			import pandas as pd
			train_df = pd.DataFrame(X_train, columns=task.feature_columns)
			train_df[task.target_column] = y_train
			
			val_df = pd.DataFrame(X_val, columns=task.feature_columns)
			val_df[task.target_column] = y_val
			
			# Configure AutoML predictor
			predictor_kwargs = {
				'label': task.target_column,
				'eval_metric': automl_config.eval_metric,
				'path': f'./automl_models/{task.task_id}',
				'verbosity': 1
			}
			
			fit_kwargs = {
				'time_limit': automl_config.time_limit,
				'presets': automl_config.presets or ['medium_quality'],
				'num_bag_folds': automl_config.num_bag_folds,
				'num_stack_levels': automl_config.num_stack_levels,
				'auto_stack': automl_config.auto_stack
			}
			
			if automl_config.include_models:
				fit_kwargs['included_model_types'] = automl_config.include_models
			if automl_config.exclude_models:
				fit_kwargs['excluded_model_types'] = automl_config.exclude_models
			
			# Train AutoML model
			predictor = TabularPredictor(**predictor_kwargs)
			predictor.fit(train_df, **fit_kwargs)
			
			# Evaluate model
			val_predictions = predictor.predict(val_df.drop(columns=[task.target_column]))
			
			if task.task_type == 'classification':
				accuracy = accuracy_score(y_val, val_predictions)
				f1 = f1_score(y_val, val_predictions, average='weighted')
				metrics = MLModelMetrics(accuracy=accuracy, f1_score=f1)
			else:
				mse = mean_squared_error(y_val, val_predictions)
				mae = np.mean(np.abs(y_val - val_predictions))
				metrics = MLModelMetrics(mse=mse, mae=mae)
			
			# Update metrics
			training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
			metrics.training_time = training_time
			
			# Get model info
			leaderboard = predictor.leaderboard(val_df)
			best_model = leaderboard.iloc[0]['model']
			logger.info(f"Best AutoML model: {best_model}")
			
			# Cache model
			with self.cache_lock:
				self.model_cache[task.task_id] = predictor
			
			task.status = "completed"
			task.metrics = metrics
			
			logger.info(f"AutoML training completed: {task.task_id}")
			return metrics
			
		except Exception as e:
			task.status = "failed"
			logger.error(f"AutoML training failed: {e}")
			raise
	
	async def predict(self, model_id: str, input_data: np.ndarray) -> np.ndarray:
		"""Make predictions using trained model."""
		try:
			start_time = datetime.now(timezone.utc)
			
			# Get model from cache or load
			model = await self._get_model(model_id)
			
			# Make prediction based on model type
			if hasattr(model, 'predict'):  # AutoGluon or sklearn
				predictions = model.predict(input_data)
			elif TORCH_AVAILABLE and isinstance(model, torch.nn.Module):  # PyTorch
				model.eval()
				with torch.no_grad():
					input_tensor = torch.FloatTensor(input_data)
					predictions = model(input_tensor).numpy()
			elif TF_AVAILABLE and hasattr(model, 'predict'):  # TensorFlow
				predictions = model.predict(input_data)
			else:
				raise ValueError(f"Unsupported model type for prediction: {type(model)}")
			
			inference_time = (datetime.now(timezone.utc) - start_time).total_seconds()
			logger.debug(f"Inference completed in {inference_time:.4f}s")
			
			return predictions
			
		except Exception as e:
			logger.error(f"Prediction failed for model {model_id}: {e}")
			raise
	
	async def optimize_workflow_with_ml(self, workflow_id: str) -> Dict[str, Any]:
		"""Optimize workflow using ML insights."""
		try:
			# Get workflow execution history
			executions = self.db_session.query(WorkflowExecution).filter(
				WorkflowExecution.workflow_id == workflow_id
			).all()
			
			if not executions:
				return {"error": "No execution history available"}
			
			# Extract features from executions
			features = []
			targets = []
			
			for execution in executions:
				if execution.metrics:
					feature_vector = [
						execution.metrics.get('duration', 0),
						execution.metrics.get('cpu_usage', 0),
						execution.metrics.get('memory_usage', 0),
						len(execution.execution_steps or []),
						1 if execution.status == 'completed' else 0
					]
					features.append(feature_vector)
					targets.append(execution.metrics.get('success_score', 0.5))
			
			if len(features) < 10:
				return {"error": "Insufficient execution history for optimization"}
			
			# Train optimization model
			X = np.array(features)
			y = np.array(targets)
			
			if SKLEARN_AVAILABLE:
				model = GradientBoostingRegressor(n_estimators=100, random_state=42)
				model.fit(X, y)
				
				# Get feature importance
				feature_names = ['duration', 'cpu_usage', 'memory_usage', 'step_count', 'success_rate']
				importance = dict(zip(feature_names, model.feature_importances_))
				
				# Generate optimization recommendations
				recommendations = []
				
				if importance['cpu_usage'] > 0.3:
					recommendations.append({
						'type': 'resource_optimization',
						'message': 'Consider optimizing CPU-intensive operations',
						'priority': 'high'
					})
				
				if importance['memory_usage'] > 0.3:
					recommendations.append({
						'type': 'memory_optimization',
						'message': 'Consider optimizing memory usage',
						'priority': 'high'
					})
				
				if importance['step_count'] > 0.2:
					recommendations.append({
						'type': 'workflow_simplification',
						'message': 'Consider reducing workflow complexity',
						'priority': 'medium'
					})
				
				return {
					'model_accuracy': model.score(X, y),
					'feature_importance': importance,
					'recommendations': recommendations,
					'optimization_score': np.mean(y)
				}
			
			return {"error": "ML libraries not available for optimization"}
			
		except Exception as e:
			logger.error(f"Workflow optimization failed: {e}")
			return {"error": str(e)}
	
	async def _prepare_training_data(self, task: MLTask) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""Prepare training data for ML task."""
		# This is a simplified implementation
		# In practice, you would load actual data from the specified path
		
		# Generate synthetic data for demonstration
		n_samples = 1000
		n_features = len(task.feature_columns) if task.feature_columns else 10
		
		X = np.random.randn(n_samples, n_features)
		
		if task.task_type == 'classification':
			y = np.random.randint(0, 3, n_samples)  # 3 classes
		else:
			y = np.random.randn(n_samples)
		
		# Split data
		if SKLEARN_AVAILABLE:
			X_train, X_val, y_train, y_val = train_test_split(
				X, y, test_size=task.validation_split, random_state=42
			)
		else:
			split_idx = int(len(X) * (1 - task.validation_split))
			X_train, X_val = X[:split_idx], X[split_idx:]
			y_train, y_val = y[:split_idx], y[split_idx:]
		
		return X_train, X_val, y_train, y_val
	
	async def _save_model(self, task_id: str, model: Any, framework: str) -> str:
		"""Save trained model to disk."""
		models_dir = Path('./models')
		models_dir.mkdir(exist_ok=True)
		
		model_path = models_dir / f"{task_id}_{framework}.pkl"
		
		try:
			if framework == 'pytorch' and TORCH_AVAILABLE:
				torch.save(model.state_dict(), model_path)
			elif framework == 'tensorflow' and TF_AVAILABLE:
				model.save(str(model_path).replace('.pkl', '.h5'))
				model_path = model_path.with_suffix('.h5')
			else:
				with open(model_path, 'wb') as f:
					pickle.dump(model, f)
			
			logger.info(f"Model saved: {model_path}")
			return str(model_path)
			
		except Exception as e:
			logger.error(f"Failed to save model: {e}")
			raise
	
	def _get_model_size(self, model_path: str) -> float:
		"""Get model file size in MB."""
		try:
			path = Path(model_path)
			if path.exists():
				return path.stat().st_size / (1024 * 1024)
		except Exception:
			pass
		return 0.0
	
	async def _get_model(self, model_id: str) -> Any:
		"""Get model from cache or load from disk."""
		with self.cache_lock:
			if model_id in self.model_cache:
				return self.model_cache[model_id]
		
		# Try to load from disk
		models_dir = Path('./models')
		for model_file in models_dir.glob(f"{model_id}_*.pkl"):
			try:
				with open(model_file, 'rb') as f:
					model = pickle.load(f)
				
				with self.cache_lock:
					self.model_cache[model_id] = model
				
				return model
			except Exception as e:
				logger.warning(f"Failed to load model {model_file}: {e}")
		
		raise ValueError(f"Model {model_id} not found")
	
	async def _create_rl_environment(self, task: MLTask):
		"""Create RL environment for workflow optimization."""
		# Simplified workflow environment
		class WorkflowEnvironment:
			def __init__(self, state_size=10, action_size=5):
				self.state_size = state_size
				self.action_size = action_size
				self.state = None
				self.reset()
			
			def reset(self):
				self.state = np.random.randn(self.state_size)
				return self.state
			
			def step(self, action):
				# Simulate environment step
				reward = np.random.randn()  # Random reward
				self.state = np.random.randn(self.state_size)
				done = np.random.random() < 0.1  # 10% chance of episode end
				return self.state, reward, done, {}
		
		return WorkflowEnvironment()
	
	async def _create_rl_agent(self, env, rl_config: ReinforcementLearningConfig):
		"""Create RL agent."""
		if not TORCH_AVAILABLE:
			raise ValueError("PyTorch required for RL agent")
		
		class DQNAgent:
			def __init__(self, state_size, action_size, config):
				self.state_size = state_size
				self.action_size = action_size
				self.memory = []
				self.memory_size = config.memory_size
				
				# Q-networks
				self.q_network = self._build_network()
				self.target_network = self._build_network()
				self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
				
				self.update_target_network()
			
			def _build_network(self):
				layers = rl_config.network_layers
				model = nn.Sequential(
					nn.Linear(self.state_size, layers[0]),
					nn.ReLU(),
					nn.Linear(layers[0], layers[1]),
					nn.ReLU(),
					nn.Linear(layers[1], self.action_size)
				)
				return model
			
			def choose_action(self, state, epsilon):
				if np.random.random() < epsilon:
					return np.random.randint(self.action_size)
				
				with torch.no_grad():
					state_tensor = torch.FloatTensor(state).unsqueeze(0)
					q_values = self.q_network(state_tensor)
					return q_values.argmax().item()
			
			def store_experience(self, state, action, reward, next_state, done):
				self.memory.append((state, action, reward, next_state, done))
				if len(self.memory) > self.memory_size:
					self.memory.pop(0)
			
			def train(self):
				if len(self.memory) < rl_config.batch_size:
					return
				
				batch = np.random.choice(len(self.memory), rl_config.batch_size, replace=False)
				states = torch.FloatTensor([self.memory[i][0] for i in batch])
				actions = torch.LongTensor([self.memory[i][1] for i in batch])
				rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
				next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
				dones = torch.BoolTensor([self.memory[i][4] for i in batch])
				
				current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
				next_q_values = self.target_network(next_states).max(1)[0].detach()
				target_q_values = rewards + (rl_config.discount_factor * next_q_values * ~dones)
				
				loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
				
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			
			def update_target_network(self):
				self.target_network.load_state_dict(self.q_network.state_dict())
		
		return DQNAgent(env.state_size, env.action_size, rl_config)
	
	async def _create_global_model(self, task: MLTask):
		"""Create global model for federated learning."""
		if not TORCH_AVAILABLE:
			raise ValueError("PyTorch required for federated learning")
		
		# Simple neural network for federated learning
		input_size = len(task.feature_columns) if task.feature_columns else 10
		hidden_sizes = task.model_config.get('hidden_sizes', [64, 32])
		output_size = 1
		
		class FederatedModel(nn.Module):
			def __init__(self):
				super().__init__()
				layers = []
				prev_size = input_size
				
				for hidden_size in hidden_sizes:
					layers.extend([
						nn.Linear(prev_size, hidden_size),
						nn.ReLU()
					])
					prev_size = hidden_size
				
				layers.append(nn.Linear(prev_size, output_size))
				self.network = nn.Sequential(*layers)
			
			def forward(self, x):
				return self.network(x)
		
		return FederatedModel()
	
	async def _train_federated_client(self, client_id: str, global_model, client_data: Any, fl_config: FederatedLearningConfig):
		"""Train model on client data."""
		# Clone global model for client training
		client_model = type(global_model)()
		client_model.load_state_dict(global_model.state_dict())
		
		# Simulate client training
		optimizer = optim.SGD(client_model.parameters(), lr=fl_config.client_learning_rate)
		criterion = nn.MSELoss()
		
		# Generate synthetic client data
		X_client = np.random.randn(100, 10)  # 100 samples, 10 features
		y_client = np.random.randn(100, 1)   # Target values
		
		X_tensor = torch.FloatTensor(X_client)
		y_tensor = torch.FloatTensor(y_client)
		
		# Client training loop
		for epoch in range(fl_config.client_epochs):
			for i in range(0, len(X_tensor), fl_config.client_batch_size):
				batch_X = X_tensor[i:i+fl_config.client_batch_size]
				batch_y = y_tensor[i:i+fl_config.client_batch_size]
				
				optimizer.zero_grad()
				outputs = client_model(batch_X)
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()
		
		return client_model.state_dict()
	
	async def _aggregate_federated_updates(self, client_updates: Dict[str, Any], aggregation_method: str):
		"""Aggregate client updates using specified method."""
		if aggregation_method == "federated_averaging":
			# Simple federated averaging
			aggregated_weights = {}
			
			# Get all parameter names from first client
			first_client = next(iter(client_updates.values()))
			
			for param_name in first_client.keys():
				# Average parameter across all clients
				param_sum = sum(client_update[param_name] for client_update in client_updates.values())
				aggregated_weights[param_name] = param_sum / len(client_updates)
			
			return aggregated_weights
		
		raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
	
	async def _evaluate_federated_model(self, model, task: MLTask) -> MLModelMetrics:
		"""Evaluate federated model."""
		# Generate test data
		X_test = np.random.randn(100, 10)
		y_test = np.random.randn(100, 1)
		
		model.eval()
		with torch.no_grad():
			X_tensor = torch.FloatTensor(X_test)
			predictions = model(X_tensor).numpy()
		
		# Calculate metrics
		mse = mean_squared_error(y_test, predictions)
		mae = np.mean(np.abs(y_test - predictions))
		
		# Simulate accuracy for classification-like metric
		accuracy = max(0, 1 - mse)  # Simple transformation
		
		return MLModelMetrics(accuracy=accuracy, mse=mse, mae=mae)
	
	def get_training_status(self, task_id: str) -> Dict[str, Any]:
		"""Get training status for a task."""
		if task_id in self.training_tasks:
			task = self.training_tasks[task_id]
			return {
				'task_id': task_id,
				'status': task.status,
				'task_type': task.task_type,
				'model_type': task.model_type,
				'created_at': task.created_at.isoformat(),
				'metrics': task.metrics.model_dump() if task.metrics else None
			}
		
		return {'error': 'Task not found'}
	
	def list_available_models(self) -> List[Dict[str, Any]]:
		"""List all available trained models."""
		models = []
		
		# Models in cache
		with self.cache_lock:
			for model_id in self.model_cache.keys():
				models.append({
					'model_id': model_id,
					'location': 'cache',
					'type': 'unknown'
				})
		
		# Models on disk
		models_dir = Path('./models')
		if models_dir.exists():
			for model_file in models_dir.glob('*.pkl'):
				model_id = model_file.stem.split('_')[0]
				if not any(m['model_id'] == model_id for m in models):
					models.append({
						'model_id': model_id,
						'location': 'disk',
						'type': 'pickle',
						'size_mb': self._get_model_size(str(model_file))
					})
		
		return models
	
	async def cleanup_models(self, max_cache_size: int = 10) -> Dict[str, int]:
		"""Cleanup model cache and old model files."""
		cleaned_cache = 0
		cleaned_files = 0
		
		# Cleanup cache if too large
		with self.cache_lock:
			if len(self.model_cache) > max_cache_size:
				# Keep only most recently used models (simplified)
				models_to_remove = list(self.model_cache.keys())[max_cache_size:]
				for model_id in models_to_remove:
					del self.model_cache[model_id]
					cleaned_cache += 1
		
		# Cleanup old model files (older than 30 days)
		models_dir = Path('./models')
		if models_dir.exists():
			cutoff_time = datetime.now() - timedelta(days=30)
			
			for model_file in models_dir.glob('*.pkl'):
				try:
					if datetime.fromtimestamp(model_file.stat().st_mtime) < cutoff_time:
						model_file.unlink()
						cleaned_files += 1
				except Exception as e:
					logger.warning(f"Failed to cleanup model file {model_file}: {e}")
		
		return {'cleaned_cache': cleaned_cache, 'cleaned_files': cleaned_files}
	
	async def shutdown(self) -> None:
		"""Shutdown the ML engine."""
		try:
			# Cancel any running training tasks
			for task in self.training_tasks.values():
				if task.status == "running":
					task.status = "cancelled"
			
			# Cleanup distributed training
			if self.distributed_enabled and TORCH_AVAILABLE:
				dist.destroy_process_group()
			
			# Shutdown thread pool
			self.executor.shutdown(wait=True)
			
			# Clear caches
			with self.cache_lock:
				self.model_cache.clear()
			
			logger.info("ML Engine shutdown completed")
		except Exception as e:
			logger.error(f"Error during ML engine shutdown: {e}")

# Global ML engine instance
_ml_engine_instance: Optional[MLEngine] = None

def get_ml_engine(config: Optional[Dict[str, Any]] = None, db_session: Optional[Session] = None) -> MLEngine:
	"""Get the global ML engine instance."""
	global _ml_engine_instance
	if _ml_engine_instance is None:
		if config is None:
			config = {}
		_ml_engine_instance = MLEngine(config, db_session)
	return _ml_engine_instance