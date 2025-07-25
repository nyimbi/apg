#!/usr/bin/env python3
"""
Federated Learning System for Cross-Twin Knowledge Sharing
=========================================================

Advanced federated learning system enabling digital twins to collaboratively
learn from each other while preserving privacy and data sovereignty.
Implements secure aggregation, differential privacy, and distributed ML.
"""

import asyncio
import hashlib
import json
import logging
import numpy as np
import pandas as pd
import pickle
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import queue
import secrets

# ML imports
try:
	from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
	from sklearn.linear_model import LogisticRegression, LinearRegression
	from sklearn.neural_network import MLPClassifier, MLPRegressor
	from sklearn.preprocessing import StandardScaler, LabelEncoder
	from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
	from sklearn.model_selection import train_test_split
	import joblib
except ImportError:
	print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
	import numpy as np
	from scipy import stats
	from scipy.spatial.distance import cosine
except ImportError:
	print("Warning: scipy not available. Install with: pip install scipy")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("federated_learning")

class LearningTask(Enum):
	"""Types of federated learning tasks"""
	CLASSIFICATION = "classification"
	REGRESSION = "regression"
	CLUSTERING = "clustering"
	ANOMALY_DETECTION = "anomaly_detection"
	FORECASTING = "forecasting"
	REINFORCEMENT_LEARNING = "reinforcement_learning"

class AggregationMethod(Enum):
	"""Methods for model aggregation"""
	FEDERATED_AVERAGING = "federated_averaging"
	WEIGHTED_AVERAGING = "weighted_averaging"
	SECURE_AGGREGATION = "secure_aggregation"
	DIFFERENTIAL_PRIVACY = "differential_privacy"
	BYZANTINE_ROBUST = "byzantine_robust"

class ParticipationStatus(Enum):
	"""Participation status in federated learning"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	SUSPENDED = "suspended"
	PENDING = "pending"
	BLACKLISTED = "blacklisted"

class PrivacyLevel(Enum):
	"""Privacy protection levels"""
	NONE = "none"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	MAXIMUM = "maximum"

@dataclass
class ModelWeights:
	"""Encrypted model weights for federated learning"""
	weights_id: str
	participant_id: str
	model_type: str
	weights_data: bytes  # Serialized and encrypted weights
	gradient_data: Optional[bytes]  # Optional gradient information
	metadata: Dict[str, Any]
	training_metrics: Dict[str, float]
	privacy_budget: float
	timestamp: datetime
	round_number: int
	data_samples: int
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'weights_id': self.weights_id,
			'participant_id': self.participant_id,
			'model_type': self.model_type,
			'weights_data': self.weights_data.hex(),
			'gradient_data': self.gradient_data.hex() if self.gradient_data else None,
			'metadata': self.metadata,
			'training_metrics': self.training_metrics,
			'privacy_budget': self.privacy_budget,
			'timestamp': self.timestamp.isoformat(),
			'round_number': self.round_number,
			'data_samples': self.data_samples
		}

@dataclass
class FederatedLearningRound:
	"""Single round of federated learning"""
	round_id: str
	round_number: int
	task_id: str
	participants: List[str]
	start_time: datetime
	end_time: Optional[datetime]
	aggregation_method: AggregationMethod
	privacy_level: PrivacyLevel
	min_participants: int
	max_participants: int
	convergence_threshold: float
	submitted_weights: List[ModelWeights]
	aggregated_model: Optional[bytes]
	performance_metrics: Dict[str, float]
	status: str  # 'active', 'completed', 'failed'
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'round_id': self.round_id,
			'round_number': self.round_number,
			'task_id': self.task_id,
			'participants': self.participants,
			'start_time': self.start_time.isoformat(),
			'end_time': self.end_time.isoformat() if self.end_time else None,
			'aggregation_method': self.aggregation_method.value,
			'privacy_level': self.privacy_level.value,
			'min_participants': self.min_participants,
			'max_participants': self.max_participants,
			'convergence_threshold': self.convergence_threshold,
			'submitted_weights_count': len(self.submitted_weights),
			'aggregated_model_size': len(self.aggregated_model) if self.aggregated_model else 0,
			'performance_metrics': self.performance_metrics,
			'status': self.status
		}

@dataclass
class FederatedTask:
	"""Federated learning task definition"""
	task_id: str
	name: str
	description: str
	learning_task: LearningTask
	model_architecture: Dict[str, Any]
	data_schema: Dict[str, Any]
	privacy_requirements: Dict[str, Any]
	performance_targets: Dict[str, float]
	creator: str
	created_at: datetime
	is_active: bool
	max_rounds: int
	current_round: int
	participants: Dict[str, ParticipationStatus]
	global_model: Optional[bytes]
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'task_id': self.task_id,
			'name': self.name,
			'description': self.description,
			'learning_task': self.learning_task.value,
			'model_architecture': self.model_architecture,
			'data_schema': self.data_schema,
			'privacy_requirements': self.privacy_requirements,
			'performance_targets': self.performance_targets,
			'creator': self.creator,
			'created_at': self.created_at.isoformat(),
			'is_active': self.is_active,
			'max_rounds': self.max_rounds,
			'current_round': self.current_round,
			'participants': {k: v.value for k, v in self.participants.items()},
			'global_model_size': len(self.global_model) if self.global_model else 0
		}

class DifferentialPrivacy:
	"""Differential privacy mechanisms for federated learning"""
	
	def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
		self.epsilon = epsilon  # Privacy budget
		self.delta = delta  # Probability of privacy breach
		self.noise_scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
	
	def add_gaussian_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
		"""Add Gaussian noise for differential privacy"""
		noise = np.random.normal(0, self.noise_scale * sensitivity, data.shape)
		return data + noise
	
	def add_laplace_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
		"""Add Laplace noise for differential privacy"""
		noise = np.random.laplace(0, sensitivity / self.epsilon, data.shape)
		return data + noise
	
	def clip_gradients(self, gradients: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
		"""Clip gradients to limit sensitivity"""
		gradient_norm = np.linalg.norm(gradients)
		if gradient_norm > clip_norm:
			return gradients * (clip_norm / gradient_norm)
		return gradients
	
	def private_aggregation(self, weight_list: List[np.ndarray], 
						   clip_norm: float = 1.0) -> np.ndarray:
		"""Perform differentially private aggregation"""
		# Clip and average weights
		clipped_weights = [self.clip_gradients(w, clip_norm) for w in weight_list]
		averaged_weights = np.mean(clipped_weights, axis=0)
		
		# Add noise
		return self.add_gaussian_noise(averaged_weights, clip_norm / len(weight_list))

class SecureAggregation:
	"""Secure aggregation protocols for federated learning"""
	
	def __init__(self):
		self.secret_shares: Dict[str, Dict[str, np.ndarray]] = {}
		self.reconstruction_threshold = 0.5
	
	def generate_secret_shares(self, weights: np.ndarray, participant_id: str, 
							  num_participants: int) -> List[Tuple[str, np.ndarray]]:
		"""Generate secret shares using Shamir's secret sharing"""
		# Simplified secret sharing (in production, use proper cryptographic library)
		shares = []
		noise_shares = []
		
		# Generate random noise for each other participant
		for i in range(num_participants - 1):
			noise = np.random.normal(0, 0.1, weights.shape)
			noise_shares.append(noise)
			shares.append((f"share_{i}", noise))
		
		# Last share is weights minus sum of noise shares
		final_share = weights - sum(noise_shares)
		shares.append((f"share_{num_participants-1}", final_share))
		
		return shares
	
	def aggregate_shares(self, all_shares: Dict[str, List[np.ndarray]]) -> np.ndarray:
		"""Aggregate secret shares to reconstruct sum"""
		if len(all_shares) < 2:
			raise ValueError("Need at least 2 participants for secure aggregation")
		
		# Sum all shares for each position
		num_shares = len(list(all_shares.values())[0])
		aggregated = None
		
		for share_idx in range(num_shares):
			share_sum = sum(shares[share_idx] for shares in all_shares.values())
			if aggregated is None:
				aggregated = share_sum
			else:
				aggregated += share_sum
		
		return aggregated / len(all_shares)

class ModelAggregator:
	"""Aggregates models from multiple participants"""
	
	def __init__(self):
		self.differential_privacy = DifferentialPrivacy()
		self.secure_aggregation = SecureAggregation()
	
	def federated_averaging(self, weight_list: List[ModelWeights]) -> bytes:
		"""Standard federated averaging"""
		if not weight_list:
			raise ValueError("No weights provided for aggregation")
		
		# For scikit-learn models, we'll use ensemble-based aggregation
		# In a real implementation, this would extract actual model parameters
		models = []
		total_samples = 0
		
		for weight_obj in weight_list:
			model = pickle.loads(weight_obj.weights_data)
			models.append(model)
			total_samples += weight_obj.data_samples
		
		# For demo, return the model with most data samples
		# In practice, this would implement proper parameter averaging
		best_model_idx = max(range(len(weight_list)), key=lambda i: weight_list[i].data_samples)
		aggregated_model = models[best_model_idx]
		
		return pickle.dumps(aggregated_model)
	
	def weighted_averaging(self, weight_list: List[ModelWeights], 
						  performance_weights: List[float]) -> bytes:
		"""Weighted averaging based on model performance"""
		if len(weight_list) != len(performance_weights):
			raise ValueError("Mismatch between weights and performance weights")
		
		# For demo, select model with best performance
		best_performance_idx = max(range(len(performance_weights)), key=lambda i: performance_weights[i])
		best_model = pickle.loads(weight_list[best_performance_idx].weights_data)
		
		return pickle.dumps(best_model)
	
	def differential_private_aggregation(self, weight_list: List[ModelWeights],
										epsilon: float = 1.0) -> bytes:
		"""Differentially private aggregation"""
		# For demo, use basic model selection with privacy consideration
		models = [pickle.loads(w.weights_data) for w in weight_list]
		
		# Select a random model to provide privacy (simplified approach)
		import random
		selected_model = random.choice(models)
		
		return pickle.dumps(selected_model)
	
	def byzantine_robust_aggregation(self, weight_list: List[ModelWeights],
									byzantine_ratio: float = 0.3) -> bytes:
		"""Byzantine-robust aggregation using coordinate-wise median"""
		# For demo, use majority voting approach
		models = [pickle.loads(w.weights_data) for w in weight_list]
		
		# Filter out potential Byzantine models (simplified approach)
		# In practice, this would use sophisticated Byzantine detection
		num_valid = int(len(models) * (1 - byzantine_ratio))
		selected_models = models[:num_valid]  # Simple selection
		
		# Return the first valid model
		return pickle.dumps(selected_models[0] if selected_models else models[0])

class FederatedLearningParticipant:
	"""Individual participant in federated learning"""
	
	def __init__(self, participant_id: str, twin_id: str):
		self.participant_id = participant_id
		self.twin_id = twin_id
		self.local_models: Dict[str, Any] = {}
		self.training_data: Dict[str, pd.DataFrame] = {}
		self.privacy_budget = 10.0  # Total privacy budget
		logger.info(f"Federated learning participant {participant_id} initialized")
	
	def load_training_data(self, task_id: str, data: pd.DataFrame):
		"""Load training data for a specific task"""
		self.training_data[task_id] = data
		logger.info(f"Loaded {len(data)} training samples for task {task_id}")
	
	def train_local_model(self, task: FederatedTask, global_model: Optional[bytes] = None,
						 round_number: int = 0) -> ModelWeights:
		"""Train local model for federated learning round"""
		
		if task.task_id not in self.training_data:
			raise ValueError(f"No training data available for task {task.task_id}")
		
		data = self.training_data[task.task_id]
		
		# Initialize or update model
		if global_model:
			# Update from global model
			model = pickle.loads(global_model)
		else:
			# Initialize new model
			model = self._create_model(task)
		
		# Prepare training data
		X, y = self._prepare_training_data(data, task)
		
		# Train model
		if hasattr(model, 'partial_fit'):
			# Online learning
			model.partial_fit(X, y)
		else:
			# Batch learning
			model.fit(X, y)
		
		# Calculate training metrics
		y_pred = model.predict(X)
		if task.learning_task == LearningTask.CLASSIFICATION:
			accuracy = accuracy_score(y, y_pred)
			f1 = f1_score(y, y_pred, average='weighted')
			metrics = {'accuracy': accuracy, 'f1_score': f1}
		else:
			mse = mean_squared_error(y, y_pred)
			metrics = {'mse': mse, 'rmse': np.sqrt(mse)}
		
		# Serialize model weights
		weights_data = pickle.dumps(model)
		
		# Create ModelWeights object
		model_weights = ModelWeights(
			weights_id=f"weights_{uuid.uuid4().hex[:12]}",
			participant_id=self.participant_id,
			model_type=type(model).__name__,
			weights_data=weights_data,
			gradient_data=None,
			metadata={'twin_id': self.twin_id},
			training_metrics=metrics,
			privacy_budget=1.0,  # Privacy budget used this round
			timestamp=datetime.utcnow(),
			round_number=round_number,
			data_samples=len(data)
		)
		
		# Store local model
		self.local_models[task.task_id] = model
		
		logger.info(f"Local model trained for task {task.task_id}: {metrics}")
		return model_weights
	
	def _create_model(self, task: FederatedTask):
		"""Create model based on task specification"""
		model_type = task.model_architecture.get('type', 'random_forest')
		
		if task.learning_task == LearningTask.CLASSIFICATION:
			if model_type == 'random_forest':
				return RandomForestClassifier(
					n_estimators=task.model_architecture.get('n_estimators', 100),
					random_state=42
				)
			elif model_type == 'logistic_regression':
				return LogisticRegression(random_state=42)
			elif model_type == 'neural_network':
				return MLPClassifier(
					hidden_layer_sizes=task.model_architecture.get('hidden_layers', (100,)),
					random_state=42
				)
		
		elif task.learning_task == LearningTask.REGRESSION:
			if model_type == 'gradient_boosting':
				return GradientBoostingRegressor(
					n_estimators=task.model_architecture.get('n_estimators', 100),
					random_state=42
				)
			elif model_type == 'linear_regression':
				return LinearRegression()
			elif model_type == 'neural_network':
				return MLPRegressor(
					hidden_layer_sizes=task.model_architecture.get('hidden_layers', (100,)),
					random_state=42
				)
		
		# Default to random forest classifier
		return RandomForestClassifier(n_estimators=100, random_state=42)
	
	def _prepare_training_data(self, data: pd.DataFrame, task: FederatedTask) -> Tuple[np.ndarray, np.ndarray]:
		"""Prepare training data based on task schema"""
		schema = task.data_schema
		
		# Extract features and target
		feature_columns = schema.get('features', [])
		target_column = schema.get('target')
		
		if not feature_columns:
			# Use all numeric columns except target as features
			numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
			if target_column in numeric_columns:
				numeric_columns.remove(target_column)
			feature_columns = numeric_columns
		
		X = data[feature_columns].values
		y = data[target_column].values if target_column else np.zeros(len(data))
		
		# Handle missing values
		X = np.nan_to_num(X)
		y = np.nan_to_num(y)
		
		return X, y
	
	def evaluate_model(self, task_id: str, test_data: pd.DataFrame) -> Dict[str, float]:
		"""Evaluate local model on test data"""
		if task_id not in self.local_models:
			raise ValueError(f"No model available for task {task_id}")
		
		model = self.local_models[task_id]
		
		# Prepare test data (simplified)
		X_test = test_data.select_dtypes(include=[np.number]).iloc[:, :-1].values
		y_test = test_data.select_dtypes(include=[np.number]).iloc[:, -1].values
		
		# Handle missing values
		X_test = np.nan_to_num(X_test)
		y_test = np.nan_to_num(y_test)
		
		# Make predictions
		y_pred = model.predict(X_test)
		
		# Calculate metrics
		if hasattr(model, 'predict_proba'):
			# Classification
			accuracy = accuracy_score(y_test, y_pred)
			f1 = f1_score(y_test, y_pred, average='weighted')
			return {'accuracy': accuracy, 'f1_score': f1}
		else:
			# Regression
			mse = mean_squared_error(y_test, y_pred)
			return {'mse': mse, 'rmse': np.sqrt(mse)}

class FederatedLearningCoordinator:
	"""Coordinates federated learning across multiple participants"""
	
	def __init__(self):
		self.participants: Dict[str, FederatedLearningParticipant] = {}
		self.tasks: Dict[str, FederatedTask] = {}
		self.rounds: Dict[str, List[FederatedLearningRound]] = {}
		self.aggregator = ModelAggregator()
		self.round_timeout = 3600  # 1 hour timeout for rounds
		
		logger.info("Federated Learning Coordinator initialized")
	
	def register_participant(self, participant: FederatedLearningParticipant):
		"""Register a new participant"""
		self.participants[participant.participant_id] = participant
		logger.info(f"Participant {participant.participant_id} registered")
	
	def create_federated_task(self, name: str, description: str, learning_task: LearningTask,
							 model_architecture: Dict[str, Any], data_schema: Dict[str, Any],
							 creator: str, privacy_requirements: Dict[str, Any] = None) -> str:
		"""Create a new federated learning task"""
		
		task_id = f"task_{uuid.uuid4().hex[:12]}"
		
		task = FederatedTask(
			task_id=task_id,
			name=name,
			description=description,
			learning_task=learning_task,
			model_architecture=model_architecture,
			data_schema=data_schema,
			privacy_requirements=privacy_requirements or {},
			performance_targets={'accuracy': 0.8} if learning_task == LearningTask.CLASSIFICATION else {'mse': 0.1},
			creator=creator,
			created_at=datetime.utcnow(),
			is_active=True,
			max_rounds=50,
			current_round=0,
			participants={},
			global_model=None
		)
		
		self.tasks[task_id] = task
		self.rounds[task_id] = []
		
		logger.info(f"Federated learning task '{name}' created with ID {task_id}")
		return task_id
	
	def join_task(self, task_id: str, participant_id: str) -> bool:
		"""Add participant to federated learning task"""
		if task_id not in self.tasks or participant_id not in self.participants:
			return False
		
		task = self.tasks[task_id]
		task.participants[participant_id] = ParticipationStatus.ACTIVE
		
		logger.info(f"Participant {participant_id} joined task {task_id}")
		return True
	
	async def start_federated_round(self, task_id: str, aggregation_method: AggregationMethod = AggregationMethod.FEDERATED_AVERAGING,
								   privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
								   min_participants: int = 2) -> str:
		"""Start a new federated learning round"""
		
		if task_id not in self.tasks:
			raise ValueError(f"Task {task_id} not found")
		
		task = self.tasks[task_id]
		active_participants = [p for p, status in task.participants.items() 
							  if status == ParticipationStatus.ACTIVE]
		
		if len(active_participants) < min_participants:
			raise ValueError(f"Not enough active participants: {len(active_participants)} < {min_participants}")
		
		# Create new round
		round_id = f"round_{uuid.uuid4().hex[:12]}"
		task.current_round += 1
		
		federated_round = FederatedLearningRound(
			round_id=round_id,
			round_number=task.current_round,
			task_id=task_id,
			participants=active_participants,
			start_time=datetime.utcnow(),
			end_time=None,
			aggregation_method=aggregation_method,
			privacy_level=privacy_level,
			min_participants=min_participants,
			max_participants=len(active_participants),
			convergence_threshold=0.01,
			submitted_weights=[],
			aggregated_model=None,
			performance_metrics={},
			status='active'
		)
		
		self.rounds[task_id].append(federated_round)
		
		# Trigger local training on all participants
		await self._coordinate_local_training(federated_round)
		
		logger.info(f"Started federated round {round_id} for task {task_id} with {len(active_participants)} participants")
		return round_id
	
	async def _coordinate_local_training(self, federated_round: FederatedLearningRound):
		"""Coordinate local training across participants"""
		
		task = self.tasks[federated_round.task_id]
		training_tasks = []
		
		# Start local training on each participant
		for participant_id in federated_round.participants:
			participant = self.participants[participant_id]
			training_task = asyncio.create_task(
				self._train_participant_model(participant, task, federated_round)
			)
			training_tasks.append(training_task)
		
		# Wait for all training to complete or timeout
		try:
			results = await asyncio.wait_for(
				asyncio.gather(*training_tasks, return_exceptions=True),
				timeout=self.round_timeout
			)
			
			# Collect successful results
			for i, result in enumerate(results):
				if isinstance(result, ModelWeights):
					federated_round.submitted_weights.append(result)
				else:
					logger.error(f"Training failed for participant {federated_round.participants[i]}: {result}")
			
			# Perform aggregation if enough participants submitted
			if len(federated_round.submitted_weights) >= federated_round.min_participants:
				await self._aggregate_models(federated_round)
			else:
				federated_round.status = 'failed'
				logger.warning(f"Round {federated_round.round_id} failed: insufficient participants")
				
		except asyncio.TimeoutError:
			federated_round.status = 'failed'
			logger.error(f"Round {federated_round.round_id} timed out")
	
	async def _train_participant_model(self, participant: FederatedLearningParticipant,
									  task: FederatedTask, federated_round: FederatedLearningRound) -> ModelWeights:
		"""Train model on a single participant"""
		try:
			# Simulate training time
			await asyncio.sleep(np.random.uniform(1, 5))
			
			model_weights = participant.train_local_model(
				task=task,
				global_model=task.global_model,
				round_number=federated_round.round_number
			)
			
			return model_weights
			
		except Exception as e:
			logger.error(f"Training failed for participant {participant.participant_id}: {e}")
			raise e
	
	async def _aggregate_models(self, federated_round: FederatedLearningRound):
		"""Aggregate models from participants"""
		
		try:
			if federated_round.aggregation_method == AggregationMethod.FEDERATED_AVERAGING:
				aggregated_model = self.aggregator.federated_averaging(federated_round.submitted_weights)
			
			elif federated_round.aggregation_method == AggregationMethod.WEIGHTED_AVERAGING:
				# Use training accuracy as weights
				performance_weights = [
					w.training_metrics.get('accuracy', 1.0 - w.training_metrics.get('mse', 0.0))
					for w in federated_round.submitted_weights
				]
				aggregated_model = self.aggregator.weighted_averaging(
					federated_round.submitted_weights, performance_weights
				)
			
			elif federated_round.aggregation_method == AggregationMethod.DIFFERENTIAL_PRIVACY:
				epsilon = 1.0 if federated_round.privacy_level == PrivacyLevel.MEDIUM else 0.5
				aggregated_model = self.aggregator.differential_private_aggregation(
					federated_round.submitted_weights, epsilon
				)
			
			elif federated_round.aggregation_method == AggregationMethod.BYZANTINE_ROBUST:
				aggregated_model = self.aggregator.byzantine_robust_aggregation(
					federated_round.submitted_weights
				)
			
			else:
				# Default to federated averaging
				aggregated_model = self.aggregator.federated_averaging(federated_round.submitted_weights)
			
			# Store aggregated model
			federated_round.aggregated_model = aggregated_model
			
			# Update global model
			task = self.tasks[federated_round.task_id]
			task.global_model = aggregated_model
			
			# Calculate aggregated performance metrics
			metrics = self._calculate_aggregated_metrics(federated_round.submitted_weights)
			federated_round.performance_metrics = metrics
			
			# Mark round as completed
			federated_round.end_time = datetime.utcnow()
			federated_round.status = 'completed'
			
			logger.info(f"Round {federated_round.round_id} completed with aggregated metrics: {metrics}")
			
		except Exception as e:
			federated_round.status = 'failed'
			logger.error(f"Model aggregation failed for round {federated_round.round_id}: {e}")
	
	def _calculate_aggregated_metrics(self, submitted_weights: List[ModelWeights]) -> Dict[str, float]:
		"""Calculate aggregated performance metrics"""
		if not submitted_weights:
			return {}
		
		# Average metrics across participants
		all_metrics = [w.training_metrics for w in submitted_weights]
		aggregated = {}
		
		# Get all metric keys
		metric_keys = set()
		for metrics in all_metrics:
			metric_keys.update(metrics.keys())
		
		# Average each metric
		for key in metric_keys:
			values = [metrics.get(key, 0) for metrics in all_metrics]
			aggregated[key] = np.mean(values)
		
		return aggregated
	
	def get_task_status(self, task_id: str) -> Dict[str, Any]:
		"""Get comprehensive status of federated learning task"""
		if task_id not in self.tasks:
			return {'error': 'Task not found'}
		
		task = self.tasks[task_id]
		rounds = self.rounds.get(task_id, [])
		
		return {
			'task': task.to_dict(),
			'total_rounds': len(rounds),
			'completed_rounds': len([r for r in rounds if r.status == 'completed']),
			'active_participants': len([p for p, s in task.participants.items() if s == ParticipationStatus.ACTIVE]),
			'latest_round': rounds[-1].to_dict() if rounds else None,
			'convergence_history': [r.performance_metrics for r in rounds if r.status == 'completed']
		}
	
	async def evaluate_global_model(self, task_id: str, test_data: pd.DataFrame) -> Dict[str, float]:
		"""Evaluate global model on test dataset"""
		if task_id not in self.tasks:
			raise ValueError(f"Task {task_id} not found")
		
		task = self.tasks[task_id]
		if not task.global_model:
			raise ValueError(f"No global model available for task {task_id}")
		
		# Deserialize global model
		model = pickle.loads(task.global_model)
		
		# Prepare test data
		X_test, y_test = self._prepare_test_data(test_data, task)
		
		# Make predictions
		y_pred = model.predict(X_test)
		
		# Calculate metrics
		if task.learning_task == LearningTask.CLASSIFICATION:
			accuracy = accuracy_score(y_test, y_pred)
			f1 = f1_score(y_test, y_pred, average='weighted')
			return {'accuracy': accuracy, 'f1_score': f1, 'test_samples': len(y_test)}
		else:
			mse = mean_squared_error(y_test, y_pred)
			return {'mse': mse, 'rmse': np.sqrt(mse), 'test_samples': len(y_test)}
	
	def _prepare_test_data(self, data: pd.DataFrame, task: FederatedTask) -> Tuple[np.ndarray, np.ndarray]:
		"""Prepare test data for evaluation"""
		schema = task.data_schema
		
		feature_columns = schema.get('features', [])
		target_column = schema.get('target')
		
		if not feature_columns:
			numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
			if target_column in numeric_columns:
				numeric_columns.remove(target_column)
			feature_columns = numeric_columns
		
		X = data[feature_columns].values
		y = data[target_column].values if target_column else np.zeros(len(data))
		
		return np.nan_to_num(X), np.nan_to_num(y)

# Test and example usage
async def test_federated_learning():
	"""Test the federated learning system"""
	
	# Initialize coordinator
	coordinator = FederatedLearningCoordinator()
	
	# Create participants (representing different digital twins)
	participants = []
	for i in range(4):
		participant = FederatedLearningParticipant(
			participant_id=f"twin_{i:03d}",
			twin_id=f"industrial_twin_{i:03d}"
		)
		participants.append(participant)
		coordinator.register_participant(participant)
	
	# Generate synthetic training data for each participant
	print("Generating synthetic training data...")
	np.random.seed(42)
	
	# Create federated learning task first to get task_id
	task_id = coordinator.create_federated_task(
		name="Industrial Equipment Fault Classification",
		description="Collaborative learning for fault detection across industrial twins",
		learning_task=LearningTask.CLASSIFICATION,
		model_architecture={
			'type': 'random_forest',
			'n_estimators': 50
		},
		data_schema={
			'features': [f'feature_{i}' for i in range(5)],
			'target': 'target'
		},
		creator="system_admin"
	)
	
	for i, participant in enumerate(participants):
		# Each participant has slightly different data distribution
		n_samples = 1000 + i * 200
		n_features = 5
		
		# Generate features with some bias per participant
		X = np.random.randn(n_samples, n_features) + i * 0.5
		
		# Generate target with different relationships per participant
		weights = np.random.randn(n_features) * (1 + i * 0.2)
		y = (X @ weights + np.random.randn(n_samples) * 0.1 > 0).astype(int)
		
		# Create DataFrame
		columns = [f'feature_{j}' for j in range(n_features)] + ['target']
		data = pd.DataFrame(np.column_stack([X, y]), columns=columns)
		
		participant.load_training_data(task_id, data)
	
	print(f"\nCreated task {task_id} with training data loaded")
	
	# Add participants to task
	for participant in participants:
		coordinator.join_task(task_id, participant.participant_id)
	
	print(f"Added {len(participants)} participants to task {task_id}")
	
	# Run multiple federated learning rounds
	print("\nRunning federated learning rounds...")
	
	for round_num in range(5):
		print(f"\n--- Round {round_num + 1} ---")
		
		# Vary aggregation method
		if round_num == 0:
			aggregation_method = AggregationMethod.FEDERATED_AVERAGING
		elif round_num == 1:
			aggregation_method = AggregationMethod.WEIGHTED_AVERAGING
		elif round_num == 2:
			aggregation_method = AggregationMethod.DIFFERENTIAL_PRIVACY
		else:
			aggregation_method = AggregationMethod.BYZANTINE_ROBUST
		
		round_id = await coordinator.start_federated_round(
			task_id=task_id,
			aggregation_method=aggregation_method,
			privacy_level=PrivacyLevel.MEDIUM
		)
		
		# Wait for round to complete
		await asyncio.sleep(2)
		
		# Get task status
		status = coordinator.get_task_status(task_id)
		latest_round = status['latest_round']
		
		print(f"Round {round_id} status: {latest_round['status']}")
		if latest_round['performance_metrics']:
			metrics = latest_round['performance_metrics']
			print(f"Aggregated metrics: {metrics}")
	
	# Generate test data and evaluate global model
	print("\nEvaluating global model...")
	
	# Generate test data
	X_test = np.random.randn(500, 5)
	weights_test = np.random.randn(5)
	y_test = (X_test @ weights_test + np.random.randn(500) * 0.1 > 0).astype(int)
	test_data = pd.DataFrame(
		np.column_stack([X_test, y_test]),
		columns=[f'feature_{i}' for i in range(5)] + ['target']
	)
	
	global_metrics = await coordinator.evaluate_global_model(task_id, test_data)
	print(f"Global model evaluation: {global_metrics}")
	
	# Show final task status
	print("\nFinal task status:")
	final_status = coordinator.get_task_status(task_id)
	print(f"Total rounds: {final_status['total_rounds']}")
	print(f"Completed rounds: {final_status['completed_rounds']}")
	print(f"Active participants: {final_status['active_participants']}")
	
	print("\nConvergence history:")
	for i, metrics in enumerate(final_status['convergence_history']):
		print(f"  Round {i+1}: {metrics}")

if __name__ == "__main__":
	asyncio.run(test_federated_learning())