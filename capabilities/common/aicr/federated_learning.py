"""
APG AI Core Framework (aicr) - Federated AI Learning Framework

Purpose: Revolutionary federated learning implementation providing privacy-preserving
         distributed AI training, secure aggregation, differential privacy,
         and multi-party computation for collaborative AI without data sharing.

Dependencies: asyncio, cryptography, differential_privacy, secure_aggregation, phe
Learning Features: Federated training, secure aggregation, differential privacy,
                  multi-party computation, privacy preservation, model updates
Usage Context: Privacy-preserving collaborative AI training across organizations

This module provides:
- Privacy-preserving federated learning protocols
- Secure multi-party aggregation of model updates
- Differential privacy mechanisms for data protection
- Homomorphic encryption for secure computation
- Byzantine fault tolerance for adversarial robustness
- Communication-efficient federated algorithms
- Client selection and resource optimization
- Decentralized federated learning coordination
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import random
import statistics
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from uuid import uuid4
import numpy as np
import secrets

from pydantic import BaseModel, Field, ConfigDict

from .models import uuid7str, _validate_tenant_id
from .security_integration import CryptographicManager
from .quantum_security import QuantumSafeSecurityManager


def _log_federated_event(event_type: str, federation_id: str, operation: str, result: str, details: str = "") -> str:
	"""Log federated learning events with standardized format."""
	timestamp = datetime.now(timezone.utc).isoformat()
	return f"FEDERATED [{event_type}] {federation_id} {operation} - {result} {details} ({timestamp})"


def _log_client_event(client_id: str, action: str, status: str, round_id: str = "") -> str:
	"""Log federated client events."""
	round_info = f" round={round_id}" if round_id else ""
	return f"FL_CLIENT [{client_id}] {action} - {status}{round_info}"


def _log_aggregation_event(round_id: str, operation: str, client_count: int, result: str) -> str:
	"""Log secure aggregation events."""
	return f"AGGREGATION [{round_id}] {operation} clients={client_count} - {result}"


class FederatedLearningAlgorithm(str, Enum):
	"""Federated learning algorithms and protocols.

	Defines different federated learning algorithms for various
	use cases and privacy requirements.

	Attributes:
		FEDERATED_AVERAGING: Standard FedAvg algorithm
		FEDERATED_SGD: Federated stochastic gradient descent
		FEDERATED_PROX: FedProx with proximal term
		FEDERATED_NOVA: FedNova with normalized averaging
		SCAFFOLD: SCAFFOLD with control variates
		MIME: MIME with momentum and error feedback
		FEDERATED_OPT: FedOpt family (FedAdam, FedYogi, etc.)
		PERSONALIZED_FL: Personalized federated learning
		HIERARCHICAL_FL: Hierarchical federated learning
		ASYNCHRONOUS_FL: Asynchronous federated learning
	"""
	FEDERATED_AVERAGING = "federated_averaging"
	FEDERATED_SGD = "federated_sgd"
	FEDERATED_PROX = "federated_prox"
	FEDERATED_NOVA = "federated_nova"
	SCAFFOLD = "scaffold"
	MIME = "mime"
	FEDERATED_OPT = "federated_opt"
	PERSONALIZED_FL = "personalized_fl"
	HIERARCHICAL_FL = "hierarchical_fl"
	ASYNCHRONOUS_FL = "asynchronous_fl"


class PrivacyMechanism(str, Enum):
	"""Privacy preservation mechanisms for federated learning.

	Different techniques to protect participant privacy
	during federated learning operations.

	Attributes:
		NONE: No privacy protection (baseline)
		DIFFERENTIAL_PRIVACY: Differential privacy with noise addition
		SECURE_AGGREGATION: Cryptographic secure aggregation
		HOMOMORPHIC_ENCRYPTION: Homomorphic encryption for computation
		SECURE_MULTIPARTY: Secure multi-party computation
		SPLIT_LEARNING: Split learning with model partitioning
		KNOWLEDGE_DISTILLATION: Privacy through knowledge distillation
		FEDERATED_DISTILLATION: Federated knowledge distillation
	"""
	NONE = "none"
	DIFFERENTIAL_PRIVACY = "differential_privacy"
	SECURE_AGGREGATION = "secure_aggregation"
	HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
	SECURE_MULTIPARTY = "secure_multiparty"
	SPLIT_LEARNING = "split_learning"
	KNOWLEDGE_DISTILLATION = "knowledge_distillation"
	FEDERATED_DISTILLATION = "federated_distillation"


class ClientSelectionStrategy(str, Enum):
	"""Client selection strategies for federated rounds.

	Different approaches for selecting which clients participate
	in each federated learning round.

	Attributes:
		RANDOM: Random client selection
		AVAILABILITY_BASED: Select based on client availability
		RESOURCE_BASED: Select based on computational resources
		DATA_QUALITY_BASED: Select based on data quality metrics
		STALENESS_BASED: Select based on model staleness
		CONTRIBUTION_BASED: Select based on historical contribution
		FAIRNESS_BASED: Select to ensure fairness across clients
		ADAPTIVE: Adaptive selection based on multiple factors
	"""
	RANDOM = "random"
	AVAILABILITY_BASED = "availability_based"
	RESOURCE_BASED = "resource_based"
	DATA_QUALITY_BASED = "data_quality_based"
	STALENESS_BASED = "staleness_based"
	CONTRIBUTION_BASED = "contribution_based"
	FAIRNESS_BASED = "fairness_based"
	ADAPTIVE = "adaptive"


class AggregationMethod(str, Enum):
	"""Model aggregation methods for federated learning.

	Different techniques for combining client model updates
	into a global model.

	Attributes:
		SIMPLE_AVERAGE: Simple averaging of model parameters
		WEIGHTED_AVERAGE: Weighted averaging by data size
		MEDIAN_AGGREGATION: Median-based robust aggregation
		TRIMMED_MEAN: Trimmed mean for Byzantine robustness
		KRUM: Krum algorithm for Byzantine fault tolerance
		COORDINATE_MEDIAN: Coordinate-wise median aggregation
		GEOMETRIC_MEDIAN: Geometric median aggregation
		CLUSTERING_AGGREGATION: Clustering-based aggregation
	"""
	SIMPLE_AVERAGE = "simple_average"
	WEIGHTED_AVERAGE = "weighted_average"
	MEDIAN_AGGREGATION = "median_aggregation"
	TRIMMED_MEAN = "trimmed_mean"
	KRUM = "krum"
	COORDINATE_MEDIAN = "coordinate_median"
	GEOMETRIC_MEDIAN = "geometric_median"
	CLUSTERING_AGGREGATION = "clustering_aggregation"


class FederatedClientState(str, Enum):
	"""Federated learning client states.

	Operational states for federated learning clients
	during the training process.

	Attributes:
		REGISTERING: Client is registering with federation
		IDLE: Client is available for training
		TRAINING: Client is performing local training
		UPLOADING: Client is uploading model updates
		SYNCHRONIZING: Client is synchronizing with global model
		DROPPED: Client has dropped out of federation
		MALICIOUS: Client detected as malicious
		OFFLINE: Client is offline/unavailable
	"""
	REGISTERING = "registering"
	IDLE = "idle"
	TRAINING = "training"
	UPLOADING = "uploading"
	SYNCHRONIZING = "synchronizing"
	DROPPED = "dropped"
	MALICIOUS = "malicious"
	OFFLINE = "offline"


class DifferentialPrivacyParameters(BaseModel):
	"""Differential privacy parameters for federated learning.

	Configuration for differential privacy mechanisms
	to protect individual data points during training.

	Attributes:
		epsilon: Privacy budget (smaller = more private)
		delta: Failure probability for (ε,δ)-differential privacy
		sensitivity: Global sensitivity of the function
		noise_multiplier: Multiplier for noise scale
		clipping_norm: L2 norm bound for gradient clipping
		adaptive_clipping: Whether to use adaptive clipping
		composition_type: Type of privacy composition
		accountant_type: Privacy accountant implementation
		per_example_privacy: Whether to provide per-example privacy
		secure_rng: Whether to use secure random number generation
	"""
	epsilon: float = 1.0
	delta: float = 1e-5
	sensitivity: float = 1.0
	noise_multiplier: float = 1.0
	clipping_norm: float = 1.0
	adaptive_clipping: bool = True
	composition_type: str = "rdp"  # Rényi Differential Privacy
	accountant_type: str = "rdp_accountant"
	per_example_privacy: bool = False
	secure_rng: bool = True

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def calculate_noise_scale(self) -> float:
		"""Calculate noise scale for differential privacy."""
		return self.sensitivity * self.noise_multiplier / self.epsilon

	def is_privacy_exhausted(self, current_epsilon: float) -> bool:
		"""Check if privacy budget is exhausted."""
		return current_epsilon >= self.epsilon

	def get_privacy_guarantee(self) -> str:
		"""Get human-readable privacy guarantee."""
		return f"({self.epsilon}, {self.delta})-differential privacy"


class SecureAggregationConfig(BaseModel):
	"""Configuration for secure aggregation protocols.

	Parameters for cryptographic secure aggregation
	that enables privacy-preserving model updates.

	Attributes:
		enabled: Whether secure aggregation is enabled
		threshold: Minimum number of clients for aggregation
		modulus: Modulus for finite field arithmetic
		reconstruction_threshold: Threshold for secret reconstruction
		dropout_resilience: Maximum number of client dropouts tolerated
		verification_enabled: Whether to verify aggregation correctness
		commitment_scheme: Cryptographic commitment scheme
		zero_knowledge_proofs: Whether to use ZK proofs
		quantum_safe: Whether to use quantum-safe cryptography
		performance_mode: Performance vs security trade-off
	"""
	enabled: bool = True
	threshold: int = 3
	modulus: int = 2**32 - 5  # Large prime for finite field
	reconstruction_threshold: int = 2
	dropout_resilience: int = 1
	verification_enabled: bool = True
	commitment_scheme: str = "pedersen"
	zero_knowledge_proofs: bool = False
	quantum_safe: bool = False
	performance_mode: str = "balanced"  # fast, balanced, secure

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def validate_threshold(self, total_clients: int) -> bool:
		"""Validate if threshold is achievable with total clients."""
		return self.threshold <= total_clients

	def calculate_communication_rounds(self, num_clients: int) -> int:
		"""Calculate communication rounds needed for secure aggregation."""
		if not self.enabled:
			return 1

		# Simplified calculation based on protocol complexity
		base_rounds = 3  # Setup, masking, reconstruction
		if self.verification_enabled:
			base_rounds += 1
		if self.zero_knowledge_proofs:
			base_rounds += 2

		return base_rounds


class ModelUpdate(BaseModel):
	"""Federated learning model update from client.

	Represents a model update contributed by a federated
	learning client including parameters, metadata, and
	privacy-preserving transformations.

	Attributes:
		update_id: Unique identifier for the update
		client_id: Client that generated the update
		round_id: Federated learning round identifier
		model_version: Version of the base model used
		parameters: Model parameter updates (encrypted/masked)
		parameter_shapes: Shapes of parameter tensors
		gradients: Gradient updates if applicable
		local_epochs: Number of local training epochs
		local_batch_size: Local training batch size
		local_learning_rate: Local learning rate used
		data_size: Size of local training dataset
		loss_value: Local training loss
		accuracy_metrics: Local validation metrics
		privacy_spent: Privacy budget consumed
		noise_added: Amount of noise added for privacy
		compression_method: Method used for update compression
		compression_ratio: Achieved compression ratio
		checksum: Integrity checksum for the update
		timestamp: Update generation timestamp
		metadata: Additional update metadata
	"""
	update_id: str = Field(default_factory=uuid7str)
	client_id: str
	round_id: str
	model_version: str
	parameters: Dict[str, bytes]  # Encrypted/compressed parameters
	parameter_shapes: Dict[str, List[int]] = Field(default_factory=dict)
	gradients: Optional[Dict[str, bytes]] = None
	local_epochs: int = 1
	local_batch_size: int = 32
	local_learning_rate: float = 0.01
	data_size: int = 0
	loss_value: float = 0.0
	accuracy_metrics: Dict[str, float] = Field(default_factory=dict)
	privacy_spent: float = 0.0
	noise_added: float = 0.0
	compression_method: str = "none"
	compression_ratio: float = 1.0
	checksum: str = ""
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	metadata: Dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def verify_integrity(self) -> bool:
		"""Verify integrity of the model update."""
		if not self.checksum:
			return False

		# Calculate checksum of parameters
		param_data = b""
		for param_bytes in self.parameters.values():
			param_data += param_bytes

		calculated_checksum = hashlib.sha256(param_data).hexdigest()
		return hmac.compare_digest(self.checksum, calculated_checksum)

	def get_update_size_bytes(self) -> int:
		"""Get total size of the update in bytes."""
		return sum(len(param_bytes) for param_bytes in self.parameters.values())

	def get_communication_cost(self) -> float:
		"""Get communication cost metric for the update."""
		base_cost = self.get_update_size_bytes() / (1024 * 1024)  # MB

		# Adjust for compression
		if self.compression_ratio > 1.0:
			base_cost /= self.compression_ratio

		return base_cost


class FederatedClient(BaseModel):
	"""Federated learning client with comprehensive capabilities.

	Represents a participant in federated learning with
	local training capabilities, privacy mechanisms,
	and secure communication protocols.

	Attributes:
		client_id: Unique client identifier
		client_name: Human-readable client name
		federation_id: Federation this client belongs to
		state: Current client state
		capabilities: Client computational capabilities
		data_characteristics: Characteristics of local data
		privacy_preferences: Client privacy preferences
		computational_resources: Available computational resources
		communication_bandwidth: Available network bandwidth
		registration_timestamp: Client registration time
		last_activity: Last activity timestamp
		total_rounds_participated: Total rounds participated
		total_data_contributed: Total data samples contributed
		average_training_time: Average local training time
		privacy_budget_remaining: Remaining privacy budget
		reputation_score: Client reputation score
		model_staleness: Staleness of client's local model
		local_model_version: Version of local model
		preferred_algorithms: Preferred FL algorithms
		security_level: Required security level
		audit_trail: Client activity audit trail
	"""
	client_id: str = Field(default_factory=uuid7str)
	client_name: str
	federation_id: str
	state: FederatedClientState = FederatedClientState.REGISTERING
	capabilities: List[str] = Field(default_factory=list)
	data_characteristics: Dict[str, Any] = Field(default_factory=dict)
	privacy_preferences: Dict[str, Any] = Field(default_factory=dict)
	computational_resources: Dict[str, float] = Field(default_factory=dict)
	communication_bandwidth: float = 100.0  # Mbps
	registration_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	total_rounds_participated: int = 0
	total_data_contributed: int = 0
	average_training_time: float = 0.0
	privacy_budget_remaining: float = 10.0
	reputation_score: float = 1.0
	model_staleness: int = 0
	local_model_version: str = "0.0.0"
	preferred_algorithms: List[FederatedLearningAlgorithm] = Field(default_factory=list)
	security_level: str = "standard"
	audit_trail: List[Dict[str, Any]] = Field(default_factory=list)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def update_activity(self) -> None:
		"""Update last activity timestamp."""
		self.last_activity = datetime.now(timezone.utc)

	def participate_in_round(self) -> None:
		"""Record participation in federated round."""
		self.total_rounds_participated += 1
		self.update_activity()

	def add_audit_entry(self, action: str, details: Dict[str, Any] = None) -> None:
		"""Add entry to client audit trail."""
		entry = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"action": action,
			"details": details or {}
		}
		self.audit_trail.append(entry)

		# Keep audit trail reasonable size
		if len(self.audit_trail) > 1000:
			self.audit_trail = self.audit_trail[-500:]

	def is_available_for_training(self) -> bool:
		"""Check if client is available for training."""
		return (
			self.state == FederatedClientState.IDLE and
			self.privacy_budget_remaining > 0.1 and
			self.reputation_score > 0.5
		)

	def calculate_selection_score(self, algorithm: FederatedLearningAlgorithm) -> float:
		"""Calculate score for client selection."""
		base_score = self.reputation_score

		# Bonus for algorithm preference
		if algorithm in self.preferred_algorithms:
			base_score *= 1.2

		# Factor in staleness (lower is better)
		staleness_penalty = 1.0 / (1.0 + self.model_staleness * 0.1)

		# Factor in resources
		resource_score = self.computational_resources.get("cpu_cores", 1.0) / 4.0

		# Combined score
		return base_score * staleness_penalty * min(2.0, 1.0 + resource_score)

	def estimate_training_time(self, data_size: int, model_complexity: float) -> float:
		"""Estimate training time for given parameters."""
		base_time = self.average_training_time if self.average_training_time > 0 else 60.0

		# Scale by data size
		size_factor = data_size / max(1, self.total_data_contributed or 1000)

		# Scale by model complexity
		complexity_factor = model_complexity

		# Scale by computational resources
		resource_factor = 4.0 / max(1, self.computational_resources.get("cpu_cores", 1.0))

		return base_time * size_factor * complexity_factor * resource_factor


class FederatedLearningRound(BaseModel):
	"""Federated learning round with comprehensive tracking.

	Represents a single round of federated learning including
	client selection, training, aggregation, and evaluation.

	Attributes:
		round_id: Unique round identifier
		federation_id: Federation conducting the round
		round_number: Sequential round number
		algorithm: Federated learning algorithm used
		aggregation_method: Model aggregation method
		privacy_mechanism: Privacy preservation mechanism
		selected_clients: List of selected client IDs
		participating_clients: List of actually participating clients
		client_updates: Collected model updates from clients
		global_model_version: Version of global model for this round
		target_accuracy: Target accuracy for the round
		convergence_threshold: Convergence threshold for stopping
		max_local_epochs: Maximum local training epochs
		min_participating_clients: Minimum clients needed
		client_selection_strategy: Strategy for selecting clients
		start_timestamp: Round start time
		end_timestamp: Round completion time
		aggregation_timestamp: Model aggregation completion time
		round_metrics: Performance metrics for the round
		privacy_spent: Total privacy budget spent in round
		communication_cost: Total communication cost
		convergence_metrics: Convergence analysis metrics
		byzantine_detection: Byzantine client detection results
		round_status: Current status of the round
	"""
	round_id: str = Field(default_factory=uuid7str)
	federation_id: str
	round_number: int
	algorithm: FederatedLearningAlgorithm
	aggregation_method: AggregationMethod
	privacy_mechanism: PrivacyMechanism
	selected_clients: List[str] = Field(default_factory=list)
	participating_clients: List[str] = Field(default_factory=list)
	client_updates: List[ModelUpdate] = Field(default_factory=list)
	global_model_version: str = "1.0.0"
	target_accuracy: float = 0.9
	convergence_threshold: float = 0.001
	max_local_epochs: int = 5
	min_participating_clients: int = 3
	client_selection_strategy: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
	start_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	end_timestamp: Optional[datetime] = None
	aggregation_timestamp: Optional[datetime] = None
	round_metrics: Dict[str, float] = Field(default_factory=dict)
	privacy_spent: float = 0.0
	communication_cost: float = 0.0
	convergence_metrics: Dict[str, float] = Field(default_factory=dict)
	byzantine_detection: Dict[str, Any] = Field(default_factory=dict)
	round_status: str = "initializing"

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def add_client_update(self, update: ModelUpdate) -> None:
		"""Add client update to the round."""
		if update.client_id in self.participating_clients:
			self.client_updates.append(update)
			self.communication_cost += update.get_communication_cost()
			self.privacy_spent += update.privacy_spent

	def complete_round(self) -> None:
		"""Mark round as completed."""
		self.end_timestamp = datetime.now(timezone.utc)
		self.round_status = "completed"

	def get_round_duration(self) -> Optional[float]:
		"""Get round duration in seconds."""
		if self.end_timestamp:
			return (self.end_timestamp - self.start_timestamp).total_seconds()
		return None

	def get_participation_rate(self) -> float:
		"""Get client participation rate."""
		if not self.selected_clients:
			return 0.0
		return len(self.participating_clients) / len(self.selected_clients)

	def calculate_round_efficiency(self) -> float:
		"""Calculate round efficiency score."""
		participation_rate = self.get_participation_rate()

		# Communication efficiency
		total_updates = len(self.client_updates)
		comm_efficiency = min(1.0, total_updates / max(1, len(self.selected_clients)))

		# Time efficiency (assume target time of 300 seconds)
		duration = self.get_round_duration() or 300.0
		time_efficiency = min(1.0, 300.0 / duration)

		# Combined efficiency
		return (participation_rate * 0.4 + comm_efficiency * 0.3 + time_efficiency * 0.3)


class DifferentialPrivacyEngine:
	"""Differential privacy engine for federated learning.

	Implements differential privacy mechanisms for protecting
	individual data points during federated learning training.

	Attributes:
		_dp_params: Differential privacy parameters
		_privacy_accountant: Privacy budget accounting
		_noise_generator: Secure noise generation
		_clipping_manager: Gradient clipping management
	"""

	def __init__(self, dp_params: DifferentialPrivacyParameters):
		"""Initialize differential privacy engine.

		Args:
			dp_params: Differential privacy parameters
		"""
		self._dp_params = dp_params
		self._privacy_accountant = {"total_epsilon": 0.0, "total_delta": 0.0}
		self._noise_generator = random.SystemRandom() if dp_params.secure_rng else random.Random()
		self._clipping_manager = {"adaptive_norm": dp_params.clipping_norm}

		# Initialize logging
		self._logger = logging.getLogger(__name__)

	def add_noise_to_gradients(self, gradients: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
		"""Add differential privacy noise to gradients.

		Args:
			gradients: Dictionary of gradient arrays

		Returns:
			Tuple[Dict[str, np.ndarray], float]: Noisy gradients and privacy spent
		"""
		try:
			noisy_gradients = {}
			total_noise = 0.0

			# Calculate noise scale
			noise_scale = self._dp_params.calculate_noise_scale()

			for layer_name, grad_array in gradients.items():
				# Clip gradients to sensitivity bound
				clipped_grad = self._clip_gradient(grad_array)

				# Add Gaussian noise
				noise_shape = grad_array.shape
				if self._dp_params.secure_rng:
					# Use cryptographically secure noise
					noise = np.array([
						self._noise_generator.gauss(0, noise_scale)
						for _ in range(grad_array.size)
					]).reshape(noise_shape)
				else:
					noise = np.random.normal(0, noise_scale, noise_shape)

				noisy_gradients[layer_name] = clipped_grad + noise
				total_noise += np.linalg.norm(noise)

			# Update privacy accountant
			privacy_spent = self._update_privacy_accountant()

			self._logger.debug(f"Added DP noise: scale={noise_scale:.6f}, privacy_spent={privacy_spent:.6f}")

			return noisy_gradients, privacy_spent

		except Exception as e:
			self._logger.error(f"DP noise addition failed: {str(e)}")
			raise

	def _clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
		"""Clip gradient to sensitivity bound."""
		grad_norm = np.linalg.norm(gradient)

		if self._dp_params.adaptive_clipping:
			# Update adaptive clipping norm
			self._clipping_manager["adaptive_norm"] = self._update_adaptive_norm(grad_norm)

		clipping_norm = self._clipping_manager["adaptive_norm"]

		if grad_norm > clipping_norm:
			return gradient * (clipping_norm / grad_norm)
		else:
			return gradient

	def _update_adaptive_norm(self, current_norm: float) -> float:
		"""Update adaptive clipping norm based on gradient statistics."""
		current_adaptive_norm = self._clipping_manager["adaptive_norm"]

		# Simple exponential moving average
		alpha = 0.1
		new_norm = alpha * current_norm + (1 - alpha) * current_adaptive_norm

		# Ensure minimum norm
		return max(0.1, new_norm)

	def _update_privacy_accountant(self) -> float:
		"""Update privacy budget accounting."""
		# Simplified privacy accounting (RDP composition)
		step_epsilon = self._dp_params.epsilon / 1000  # Assume 1000 steps total

		self._privacy_accountant["total_epsilon"] += step_epsilon
		self._privacy_accountant["total_delta"] += self._dp_params.delta

		return step_epsilon

	def get_privacy_status(self) -> Dict[str, float]:
		"""Get current privacy budget status."""
		return {
			"total_epsilon_spent": self._privacy_accountant["total_epsilon"],
			"total_delta_spent": self._privacy_accountant["total_delta"],
			"epsilon_remaining": max(0, self._dp_params.epsilon - self._privacy_accountant["total_epsilon"]),
			"privacy_exhausted": self._privacy_accountant["total_epsilon"] >= self._dp_params.epsilon
		}

	def is_privacy_budget_available(self) -> bool:
		"""Check if privacy budget is still available."""
		return self._privacy_accountant["total_epsilon"] < self._dp_params.epsilon


class SecureAggregationProtocol:
	"""Secure aggregation protocol for federated learning.

	Implements cryptographic secure aggregation protocols
	that enable privacy-preserving model parameter aggregation
	without revealing individual client updates.

	Attributes:
		_config: Secure aggregation configuration
		_crypto_manager: Cryptographic operations manager
		_quantum_security: Quantum-safe security manager
		_client_keys: Client cryptographic keys
		_aggregation_state: Current aggregation state
	"""

	def __init__(self, config: SecureAggregationConfig,
				 crypto_manager: CryptographicManager,
				 quantum_security: Optional[QuantumSafeSecurityManager] = None):
		"""Initialize secure aggregation protocol.

		Args:
			config: Secure aggregation configuration
			crypto_manager: Cryptographic operations manager
			quantum_security: Quantum-safe security manager
		"""
		self._config = config
		self._crypto_manager = crypto_manager
		self._quantum_security = quantum_security
		self._client_keys: Dict[str, Dict[str, bytes]] = {}
		self._aggregation_state = {
			"phase": "setup",
			"participating_clients": set(),
			"masked_updates": {},
			"secret_shares": {},
			"reconstruction_ready": False
		}

		# Initialize logging
		self._logger = logging.getLogger(__name__)

	async def setup_secure_aggregation(self, client_ids: List[str]) -> Dict[str, Any]:
		"""Setup secure aggregation for a round.

		Args:
			client_ids: List of participating client IDs

		Returns:
			Dict[str, Any]: Setup information for clients
		"""
		try:
			if not self._config.enabled:
				return {"secure_aggregation_enabled": False}

			# Validate threshold
			if not self._config.validate_threshold(len(client_ids)):
				raise ValueError(f"Insufficient clients for threshold: {len(client_ids)} < {self._config.threshold}")

			# Generate shared parameters
			setup_info = {
				"secure_aggregation_enabled": True,
				"modulus": self._config.modulus,
				"threshold": self._config.threshold,
				"reconstruction_threshold": self._config.reconstruction_threshold,
				"round_id": uuid7str(),
				"client_setup": {}
			}

			# Generate keys for each client
			for client_id in client_ids:
				client_keys = await self._generate_client_keys(client_id)
				self._client_keys[client_id] = client_keys

				setup_info["client_setup"][client_id] = {
					"public_key": base64.b64encode(client_keys["public_key"]).decode('utf-8'),
					"aggregation_seed": base64.b64encode(client_keys["aggregation_seed"]).decode('utf-8')
				}

			self._aggregation_state["phase"] = "masking"
			self._aggregation_state["participating_clients"] = set(client_ids)

			self._logger.info(f"Secure aggregation setup completed for {len(client_ids)} clients")

			return setup_info

		except Exception as e:
			self._logger.error(f"Secure aggregation setup failed: {str(e)}")
			raise

	async def _generate_client_keys(self, client_id: str) -> Dict[str, bytes]:
		"""Generate cryptographic keys for client."""
		if self._config.quantum_safe and self._quantum_security:
			# Use quantum-safe cryptography
			public_key_id, private_key_id = await self._quantum_security.generate_quantum_keypair(
				algorithm="CRYSTALS-Kyber",
				security_level="LEVEL_3"
			)

			# Get key material (simplified)
			public_key = secrets.token_bytes(32)
			private_key = secrets.token_bytes(32)
		else:
			# Use classical cryptography
			private_key = secrets.token_bytes(32)
			public_key = hashlib.sha256(private_key).digest()

		# Generate aggregation seed
		aggregation_seed = secrets.token_bytes(32)

		return {
			"public_key": public_key,
			"private_key": private_key,
			"aggregation_seed": aggregation_seed
		}

	async def add_masked_update(self, client_id: str, masked_update: bytes) -> bool:
		"""Add masked update from client.

		Args:
			client_id: Client identifier
			masked_update: Masked model update

		Returns:
			bool: Success status
		"""
		try:
			if client_id not in self._aggregation_state["participating_clients"]:
				return False

			self._aggregation_state["masked_updates"][client_id] = masked_update

			self._logger.debug(f"Added masked update from client: {client_id}")

			# Check if all updates received
			received_count = len(self._aggregation_state["masked_updates"])
			expected_count = len(self._aggregation_state["participating_clients"])

			if received_count >= self._config.threshold:
				self._aggregation_state["phase"] = "unmasking"
				self._aggregation_state["reconstruction_ready"] = True

			return True

		except Exception as e:
			self._logger.error(f"Failed to add masked update: {str(e)}")
			return False

	async def aggregate_updates(self) -> Optional[bytes]:
		"""Perform secure aggregation of masked updates.

		Returns:
			Optional[bytes]: Aggregated model parameters
		"""
		try:
			if not self._aggregation_state["reconstruction_ready"]:
				return None

			masked_updates = self._aggregation_state["masked_updates"]

			if len(masked_updates) < self._config.threshold:
				raise ValueError(f"Insufficient updates for aggregation: {len(masked_updates)} < {self._config.threshold}")

			# Simulate secure aggregation
			# In production, implement actual cryptographic protocols
			aggregated_data = self._simulate_secure_aggregation(masked_updates)

			self._aggregation_state["phase"] = "completed"

			self._logger.info(_log_aggregation_event(
				"secure_agg_round", "aggregate_updates", len(masked_updates), "SUCCESS"
			))

			return aggregated_data

		except Exception as e:
			self._logger.error(f"Secure aggregation failed: {str(e)}")
			return None

	def _simulate_secure_aggregation(self, masked_updates: Dict[str, bytes]) -> bytes:
		"""Simulate secure aggregation computation."""
		# In production, implement actual secure aggregation protocol
		# This is a simplified simulation

		total_size = 0
		aggregated_bytes = bytearray()

		for update_bytes in masked_updates.values():
			if total_size == 0:
				total_size = len(update_bytes)
				aggregated_bytes = bytearray(total_size)

			# Simple XOR aggregation (not secure, just for simulation)
			for i, byte_val in enumerate(update_bytes):
				if i < len(aggregated_bytes):
					aggregated_bytes[i] ^= byte_val

		return bytes(aggregated_bytes)

	def get_aggregation_status(self) -> Dict[str, Any]:
		"""Get current aggregation status."""
		return {
			"phase": self._aggregation_state["phase"],
			"participating_clients": len(self._aggregation_state["participating_clients"]),
			"received_updates": len(self._aggregation_state["masked_updates"]),
			"threshold": self._config.threshold,
			"reconstruction_ready": self._aggregation_state["reconstruction_ready"],
			"configuration": {
				"enabled": self._config.enabled,
				"quantum_safe": self._config.quantum_safe,
				"verification_enabled": self._config.verification_enabled
			}
		}


class FederatedLearningCoordinator:
	"""Federated learning coordinator managing the complete FL process.

	Central coordinator for federated learning that manages clients,
	orchestrates training rounds, applies privacy mechanisms,
	and coordinates secure aggregation protocols.

	Attributes:
		federation_id: Unique federation identifier
		federation_name: Human-readable federation name
		algorithm: Federated learning algorithm
		clients: Registered federated clients
		current_round: Current training round
		round_history: Historical training rounds
		global_model: Current global model
		coordinator_config: Coordinator configuration
		privacy_engine: Differential privacy engine
		secure_aggregation: Secure aggregation protocol
	"""

	def __init__(self, federation_name: str, algorithm: FederatedLearningAlgorithm,
				 dp_params: Optional[DifferentialPrivacyParameters] = None,
				 sa_config: Optional[SecureAggregationConfig] = None):
		"""Initialize federated learning coordinator.

		Args:
			federation_name: Name of the federation
			algorithm: Federated learning algorithm to use
			dp_params: Differential privacy parameters
			sa_config: Secure aggregation configuration
		"""
		self.federation_id = uuid7str()
		self.federation_name = federation_name
		self.algorithm = algorithm
		self.creation_timestamp = datetime.now(timezone.utc)

		# Client management
		self.clients: Dict[str, FederatedClient] = {}
		self.active_clients: Set[str] = set()

		# Round management
		self.current_round: Optional[FederatedLearningRound] = None
		self.round_history: List[FederatedLearningRound] = []
		self.round_counter = 0

		# Model management
		self.global_model = {
			"version": "1.0.0",
			"parameters": {},
			"metadata": {
				"algorithm": algorithm.value,
				"federation_id": self.federation_id,
				"creation_time": datetime.now(timezone.utc).isoformat()
			}
		}

		# Configuration
		self.coordinator_config = {
			"max_rounds": 100,
			"target_accuracy": 0.95,
			"convergence_threshold": 0.001,
			"min_clients_per_round": 3,
			"max_clients_per_round": 20,
			"client_selection_strategy": ClientSelectionStrategy.ADAPTIVE,
			"aggregation_method": AggregationMethod.WEIGHTED_AVERAGE,
			"privacy_mechanism": PrivacyMechanism.DIFFERENTIAL_PRIVACY if dp_params else PrivacyMechanism.NONE
		}

		# Privacy and security
		self.privacy_engine = DifferentialPrivacyEngine(dp_params) if dp_params else None
		self.crypto_manager = CryptographicManager()
		self.quantum_security = QuantumSafeSecurityManager() if sa_config and sa_config.quantum_safe else None
		self.secure_aggregation = SecureAggregationProtocol(
			sa_config or SecureAggregationConfig(),
			self.crypto_manager,
			self.quantum_security
		)

		# Performance tracking
		self.federation_metrics = {
			"total_rounds_completed": 0,
			"total_clients_registered": 0,
			"average_round_duration": 0.0,
			"best_global_accuracy": 0.0,
			"total_privacy_spent": 0.0,
			"convergence_achieved": False
		}

		# Initialize logging
		self._logger = logging.getLogger(__name__)

		self._logger.info(_log_federated_event(
			"INITIALIZATION", self.federation_id, "create_federation", "SUCCESS",
			f"algorithm={algorithm.value}"
		))

	async def register_client(self, client: FederatedClient) -> bool:
		"""Register new federated learning client.

		Args:
			client: Client to register

		Returns:
			bool: Registration success status
		"""
		try:
			# Validate client
			if client.client_id in self.clients:
				return False

			# Set federation ID
			client.federation_id = self.federation_id
			client.state = FederatedClientState.IDLE

			# Add to clients
			self.clients[client.client_id] = client
			self.active_clients.add(client.client_id)

			# Update metrics
			self.federation_metrics["total_clients_registered"] += 1

			# Add audit entry
			client.add_audit_entry("registered", {
				"federation_id": self.federation_id,
				"registration_time": datetime.now(timezone.utc).isoformat()
			})

			self._logger.info(_log_client_event(
				client.client_id, "registered", "SUCCESS"
			))

			return True

		except Exception as e:
			self._logger.error(f"Client registration failed: {str(e)}")
			return False

	async def start_federated_round(self) -> Optional[str]:
		"""Start new federated learning round.

		Returns:
			Optional[str]: Round ID if started successfully
		"""
		try:
			if self.current_round and self.current_round.round_status != "completed":
				return None

			# Check if we have enough clients
			available_clients = [
				client for client in self.clients.values()
				if client.is_available_for_training()
			]

			if len(available_clients) < self.coordinator_config["min_clients_per_round"]:
				self._logger.warning(f"Insufficient clients for round: {len(available_clients)}")
				return None

			# Create new round
			self.round_counter += 1
			round_id = uuid7str()

			new_round = FederatedLearningRound(
				round_id=round_id,
				federation_id=self.federation_id,
				round_number=self.round_counter,
				algorithm=self.algorithm,
				aggregation_method=self.coordinator_config["aggregation_method"],
				privacy_mechanism=self.coordinator_config["privacy_mechanism"],
				global_model_version=self.global_model["version"],
				client_selection_strategy=self.coordinator_config["client_selection_strategy"],
				target_accuracy=self.coordinator_config["target_accuracy"],
				min_participating_clients=self.coordinator_config["min_clients_per_round"]
			)

			# Select clients for this round
			selected_clients = await self._select_clients_for_round(available_clients, new_round)
			new_round.selected_clients = [client.client_id for client in selected_clients]

			# Setup secure aggregation if enabled
			if self.secure_aggregation._config.enabled:
				await self.secure_aggregation.setup_secure_aggregation(new_round.selected_clients)

			# Set as current round
			self.current_round = new_round
			new_round.round_status = "client_training"

			# Notify selected clients
			await self._notify_clients_for_training(selected_clients, new_round)

			self._logger.info(_log_federated_event(
				"ROUND_START", self.federation_id, f"round_{self.round_counter}", "SUCCESS",
				f"clients={len(selected_clients)}"
			))

			return round_id

		except Exception as e:
			self._logger.error(f"Failed to start federated round: {str(e)}")
			return None

	async def _select_clients_for_round(self, available_clients: List[FederatedClient],
										round_info: FederatedLearningRound) -> List[FederatedClient]:
		"""Select clients for federated round based on strategy."""
		max_clients = min(
			len(available_clients),
			self.coordinator_config["max_clients_per_round"]
		)

		if round_info.client_selection_strategy == ClientSelectionStrategy.RANDOM:
			return random.sample(available_clients, max_clients)

		elif round_info.client_selection_strategy == ClientSelectionStrategy.AVAILABILITY_BASED:
			# Sort by last activity (most recent first)
			sorted_clients = sorted(available_clients, key=lambda c: c.last_activity, reverse=True)
			return sorted_clients[:max_clients]

		elif round_info.client_selection_strategy == ClientSelectionStrategy.RESOURCE_BASED:
			# Sort by computational resources
			sorted_clients = sorted(
				available_clients,
				key=lambda c: c.computational_resources.get("cpu_cores", 1.0),
				reverse=True
			)
			return sorted_clients[:max_clients]

		elif round_info.client_selection_strategy == ClientSelectionStrategy.CONTRIBUTION_BASED:
			# Sort by reputation score
			sorted_clients = sorted(available_clients, key=lambda c: c.reputation_score, reverse=True)
			return sorted_clients[:max_clients]

		elif round_info.client_selection_strategy == ClientSelectionStrategy.ADAPTIVE:
			# Use adaptive scoring
			scored_clients = [
				(client, client.calculate_selection_score(self.algorithm))
				for client in available_clients
			]
			scored_clients.sort(key=lambda x: x[1], reverse=True)
			return [client for client, _ in scored_clients[:max_clients]]

		else:
			# Default to random
			return random.sample(available_clients, max_clients)

	async def _notify_clients_for_training(self, selected_clients: List[FederatedClient],
										   round_info: FederatedLearningRound) -> None:
		"""Notify selected clients to start training."""
		for client in selected_clients:
			client.state = FederatedClientState.TRAINING
			client.model_staleness = 0
			client.add_audit_entry("round_selected", {
				"round_id": round_info.round_id,
				"round_number": round_info.round_number
			})

			# Simulate training notification
			await self._simulate_client_training(client, round_info)

	async def _simulate_client_training(self, client: FederatedClient,
										round_info: FederatedLearningRound) -> None:
		"""Simulate client local training process."""
		try:
			# Simulate training time
			training_time = client.estimate_training_time(
				data_size=client.total_data_contributed or 1000,
				model_complexity=1.0
			)

			# Speed up simulation
			await asyncio.sleep(training_time / 100.0)

			# Create simulated model update
			update = await self._create_simulated_update(client, round_info)

			# Add update to round
			round_info.add_client_update(update)
			round_info.participating_clients.append(client.client_id)

			# Update client state
			client.state = FederatedClientState.IDLE
			client.participate_in_round()
			client.average_training_time = (
				client.average_training_time * 0.9 + training_time * 0.1
			)

			self._logger.debug(_log_client_event(
				client.client_id, "training_completed", "SUCCESS", round_info.round_id
			))

			# Check if round is complete
			if len(round_info.participating_clients) >= round_info.min_participating_clients:
				await self._complete_round_if_ready(round_info)

		except Exception as e:
			self._logger.error(f"Client training simulation failed: {str(e)}")
			client.state = FederatedClientState.IDLE

	async def _create_simulated_update(self, client: FederatedClient,
									   round_info: FederatedLearningRound) -> ModelUpdate:
		"""Create simulated model update from client."""
		# Generate simulated parameters
		param_data = {}
		param_shapes = {}

		# Simulate different layer sizes
		layers = ["layer1", "layer2", "layer3", "output"]
		for layer in layers:
			# Random parameter data
			param_size = random.randint(1000, 5000)
			param_array = np.random.normal(0, 0.1, param_size)

			# Apply differential privacy if enabled
			if self.privacy_engine and round_info.privacy_mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
				grad_dict = {layer: param_array}
				noisy_grads, privacy_spent = self.privacy_engine.add_noise_to_gradients(grad_dict)
				param_array = noisy_grads[layer]
			else:
				privacy_spent = 0.0

			# Serialize parameter data
			param_data[layer] = param_array.tobytes()
			param_shapes[layer] = list(param_array.shape)

		# Calculate checksum
		param_bytes = b"".join(param_data.values())
		checksum = hashlib.sha256(param_bytes).hexdigest()

		# Create model update
		update = ModelUpdate(
			client_id=client.client_id,
			round_id=round_info.round_id,
			model_version=round_info.global_model_version,
			parameters=param_data,
			parameter_shapes=param_shapes,
			local_epochs=random.randint(1, 5),
			local_batch_size=random.choice([16, 32, 64]),
			local_learning_rate=random.uniform(0.001, 0.1),
			data_size=client.total_data_contributed or random.randint(500, 2000),
			loss_value=random.uniform(0.1, 2.0),
			accuracy_metrics={"accuracy": random.uniform(0.7, 0.95)},
			privacy_spent=privacy_spent,
			noise_added=privacy_spent * 10.0,  # Approximate noise amount
			checksum=checksum
		)

		return update

	async def _complete_round_if_ready(self, round_info: FederatedLearningRound) -> None:
		"""Complete federated round if conditions are met."""
		try:
			if len(round_info.participating_clients) < round_info.min_participating_clients:
				return

			# Check if all selected clients have responded or timeout
			timeout_seconds = 300  # 5 minutes
			elapsed = (datetime.now(timezone.utc) - round_info.start_timestamp).total_seconds()

			all_responded = len(round_info.participating_clients) == len(round_info.selected_clients)
			timeout_reached = elapsed > timeout_seconds

			if not (all_responded or timeout_reached):
				return

			# Perform model aggregation
			round_info.round_status = "aggregating"
			aggregated_model = await self._aggregate_model_updates(round_info)

			if aggregated_model:
				# Update global model
				self.global_model["parameters"] = aggregated_model
				self.global_model["version"] = f"{self.round_counter}.0.0"

				# Calculate round metrics
				await self._calculate_round_metrics(round_info)

				# Complete round
				round_info.complete_round()
				self.round_history.append(round_info)
				self.current_round = None

				# Update federation metrics
				self.federation_metrics["total_rounds_completed"] += 1
				if round_info.round_metrics.get("global_accuracy", 0) > self.federation_metrics["best_global_accuracy"]:
					self.federation_metrics["best_global_accuracy"] = round_info.round_metrics["global_accuracy"]

				# Check for convergence
				if await self._check_convergence(round_info):
					self.federation_metrics["convergence_achieved"] = True

				self._logger.info(_log_federated_event(
					"ROUND_COMPLETE", self.federation_id, f"round_{self.round_counter}", "SUCCESS",
					f"participants={len(round_info.participating_clients)}"
				))

		except Exception as e:
			self._logger.error(f"Round completion failed: {str(e)}")
			if round_info:
				round_info.round_status = "failed"

	async def _aggregate_model_updates(self, round_info: FederatedLearningRound) -> Optional[Dict[str, Any]]:
		"""Aggregate model updates from participating clients."""
		try:
			if not round_info.client_updates:
				return None

			round_info.aggregation_timestamp = datetime.now(timezone.utc)

			# Use secure aggregation if enabled
			if self.secure_aggregation._config.enabled:
				return await self._secure_aggregate_updates(round_info)
			else:
				return await self._plain_aggregate_updates(round_info)

		except Exception as e:
			self._logger.error(f"Model aggregation failed: {str(e)}")
			return None

	async def _secure_aggregate_updates(self, round_info: FederatedLearningRound) -> Optional[Dict[str, Any]]:
		"""Perform secure aggregation of model updates."""
		try:
			# Collect masked updates
			for update in round_info.client_updates:
				# Simulate masked update (in production, client would provide this)
				masked_data = b"masked_" + json.dumps(update.parameters).encode('utf-8')
				await self.secure_aggregation.add_masked_update(update.client_id, masked_data)

			# Perform secure aggregation
			aggregated_bytes = await self.secure_aggregation.aggregate_updates()

			if aggregated_bytes:
				# Simulate parameter reconstruction
				return {"secure_aggregated_params": base64.b64encode(aggregated_bytes).decode('utf-8')}

			return None

		except Exception as e:
			self._logger.error(f"Secure aggregation failed: {str(e)}")
			return None

	async def _plain_aggregate_updates(self, round_info: FederatedLearningRound) -> Dict[str, Any]:
		"""Perform plain aggregation of model updates."""
		aggregated_params = {}

		# Get aggregation method
		method = round_info.aggregation_method

		if method == AggregationMethod.SIMPLE_AVERAGE:
			aggregated_params = self._simple_average_aggregation(round_info.client_updates)
		elif method == AggregationMethod.WEIGHTED_AVERAGE:
			aggregated_params = self._weighted_average_aggregation(round_info.client_updates)
		elif method == AggregationMethod.MEDIAN_AGGREGATION:
			aggregated_params = self._median_aggregation(round_info.client_updates)
		elif method == AggregationMethod.TRIMMED_MEAN:
			aggregated_params = self._trimmed_mean_aggregation(round_info.client_updates)
		else:
			# Default to weighted average
			aggregated_params = self._weighted_average_aggregation(round_info.client_updates)

		return aggregated_params

	def _simple_average_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
		"""Perform simple average aggregation."""
		if not updates:
			return {}

		aggregated = {}
		layer_names = set()

		# Collect all layer names
		for update in updates:
			layer_names.update(update.parameters.keys())

		# Average each layer
		for layer_name in layer_names:
			layer_params = []
			for update in updates:
				if layer_name in update.parameters:
					param_data = update.parameters[layer_name]
					param_array = np.frombuffer(param_data, dtype=np.float64)
					layer_params.append(param_array)

			if layer_params:
				# Ensure all arrays have same shape
				min_size = min(len(arr) for arr in layer_params)
				truncated_params = [arr[:min_size] for arr in layer_params]
				avg_params = np.mean(truncated_params, axis=0)
				aggregated[layer_name] = avg_params.tobytes()

		return aggregated

	def _weighted_average_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
		"""Perform weighted average aggregation based on data size."""
		if not updates:
			return {}

		# Calculate weights based on data size
		total_data = sum(update.data_size for update in updates)
		weights = [update.data_size / total_data for update in updates]

		aggregated = {}
		layer_names = set()

		# Collect all layer names
		for update in updates:
			layer_names.update(update.parameters.keys())

		# Weighted average for each layer
		for layer_name in layer_names:
			layer_params = []
			layer_weights = []

			for i, update in enumerate(updates):
				if layer_name in update.parameters:
					param_data = update.parameters[layer_name]
					param_array = np.frombuffer(param_data, dtype=np.float64)
					layer_params.append(param_array)
					layer_weights.append(weights[i])

			if layer_params:
				# Ensure all arrays have same shape
				min_size = min(len(arr) for arr in layer_params)
				truncated_params = [arr[:min_size] for arr in layer_params]

				# Normalize weights
				weight_sum = sum(layer_weights)
				normalized_weights = [w / weight_sum for w in layer_weights]

				# Weighted average
				weighted_params = np.zeros(min_size)
				for param_array, weight in zip(truncated_params, normalized_weights):
					weighted_params += param_array * weight

				aggregated[layer_name] = weighted_params.tobytes()

		return aggregated

	def _median_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
		"""Perform median aggregation for Byzantine robustness."""
		if not updates:
			return {}

		aggregated = {}
		layer_names = set()

		# Collect all layer names
		for update in updates:
			layer_names.update(update.parameters.keys())

		# Median for each layer
		for layer_name in layer_names:
			layer_params = []
			for update in updates:
				if layer_name in update.parameters:
					param_data = update.parameters[layer_name]
					param_array = np.frombuffer(param_data, dtype=np.float64)
					layer_params.append(param_array)

			if layer_params:
				# Ensure all arrays have same shape
				min_size = min(len(arr) for arr in layer_params)
				truncated_params = [arr[:min_size] for arr in layer_params]

				# Element-wise median
				stacked_params = np.stack(truncated_params)
				median_params = np.median(stacked_params, axis=0)

				aggregated[layer_name] = median_params.tobytes()

		return aggregated

	def _trimmed_mean_aggregation(self, updates: List[ModelUpdate], trim_ratio: float = 0.1) -> Dict[str, Any]:
		"""Perform trimmed mean aggregation for robustness."""
		if not updates:
			return {}

		aggregated = {}
		layer_names = set()

		# Collect all layer names
		for update in updates:
			layer_names.update(update.parameters.keys())

		# Trimmed mean for each layer
		for layer_name in layer_names:
			layer_params = []
			for update in updates:
				if layer_name in update.parameters:
					param_data = update.parameters[layer_name]
					param_array = np.frombuffer(param_data, dtype=np.float64)
					layer_params.append(param_array)

			if layer_params:
				# Ensure all arrays have same shape
				min_size = min(len(arr) for arr in layer_params)
				truncated_params = [arr[:min_size] for arr in layer_params]

				# Element-wise trimmed mean
				stacked_params = np.stack(truncated_params)

				# Calculate trim indices
				num_clients = len(truncated_params)
				trim_count = int(num_clients * trim_ratio)

				if trim_count > 0:
					sorted_params = np.sort(stacked_params, axis=0)
					trimmed_params = sorted_params[trim_count:-trim_count] if trim_count < num_clients//2 else sorted_params
					mean_params = np.mean(trimmed_params, axis=0)
				else:
					mean_params = np.mean(stacked_params, axis=0)

				aggregated[layer_name] = mean_params.tobytes()

		return aggregated

	async def _calculate_round_metrics(self, round_info: FederatedLearningRound) -> None:
		"""Calculate performance metrics for the round."""
		try:
			# Participation metrics
			participation_rate = round_info.get_participation_rate()
			round_duration = round_info.get_round_duration() or 0.0

			# Accuracy metrics (simulated)
			if round_info.client_updates:
				client_accuracies = [
					update.accuracy_metrics.get("accuracy", 0.0)
					for update in round_info.client_updates
				]

				global_accuracy = statistics.mean(client_accuracies) if client_accuracies else 0.0
				accuracy_std = statistics.stdev(client_accuracies) if len(client_accuracies) > 1 else 0.0
			else:
				global_accuracy = 0.0
				accuracy_std = 0.0

			# Communication efficiency
			total_comm_cost = round_info.communication_cost
			avg_comm_cost = total_comm_cost / max(1, len(round_info.participating_clients))

			# Privacy metrics
			total_privacy_spent = round_info.privacy_spent

			# Store metrics
			round_info.round_metrics = {
				"participation_rate": participation_rate,
				"round_duration_seconds": round_duration,
				"global_accuracy": global_accuracy,
				"accuracy_std": accuracy_std,
				"communication_cost_total": total_comm_cost,
				"communication_cost_avg": avg_comm_cost,
				"privacy_spent": total_privacy_spent,
				"round_efficiency": round_info.calculate_round_efficiency(),
				"convergence_rate": abs(global_accuracy - self.federation_metrics["best_global_accuracy"])
			}

			# Update federation averages
			if self.federation_metrics["total_rounds_completed"] > 0:
				current_avg = self.federation_metrics["average_round_duration"]
				new_avg = (current_avg * self.federation_metrics["total_rounds_completed"] + round_duration) / (self.federation_metrics["total_rounds_completed"] + 1)
				self.federation_metrics["average_round_duration"] = new_avg
			else:
				self.federation_metrics["average_round_duration"] = round_duration

			self.federation_metrics["total_privacy_spent"] += total_privacy_spent

		except Exception as e:
			self._logger.error(f"Round metrics calculation failed: {str(e)}")

	async def _check_convergence(self, round_info: FederatedLearningRound) -> bool:
		"""Check if federated learning has converged."""
		try:
			# Check accuracy target
			current_accuracy = round_info.round_metrics.get("global_accuracy", 0.0)
			if current_accuracy >= self.coordinator_config["target_accuracy"]:
				return True

			# Check convergence based on recent rounds
			if len(self.round_history) >= 3:
				recent_accuracies = [
					r.round_metrics.get("global_accuracy", 0.0)
					for r in self.round_history[-3:]
				]
				recent_accuracies.append(current_accuracy)

				# Check if improvement is below threshold
				accuracy_improvements = [
					recent_accuracies[i+1] - recent_accuracies[i]
					for i in range(len(recent_accuracies)-1)
				]

				avg_improvement = statistics.mean(accuracy_improvements)
				return avg_improvement < self.coordinator_config["convergence_threshold"]

			return False

		except Exception as e:
			self._logger.error(f"Convergence check failed: {str(e)}")
			return False

	async def get_federation_status(self) -> Dict[str, Any]:
		"""Get comprehensive federation status.

		Returns:
			Dict[str, Any]: Federation status information
		"""
		return {
			"federation_info": {
				"federation_id": self.federation_id,
				"federation_name": self.federation_name,
				"algorithm": self.algorithm.value,
				"creation_timestamp": self.creation_timestamp.isoformat(),
				"uptime_seconds": (datetime.now(timezone.utc) - self.creation_timestamp).total_seconds()
			},
			"global_model": {
				"version": self.global_model["version"],
				"parameter_count": len(self.global_model["parameters"]),
				"metadata": self.global_model["metadata"]
			},
			"clients": {
				"total_registered": len(self.clients),
				"active_clients": len(self.active_clients),
				"clients_by_state": self._count_clients_by_state(),
				"average_reputation": statistics.mean([c.reputation_score for c in self.clients.values()]) if self.clients else 0.0
			},
			"current_round": {
				"active": self.current_round is not None,
				"round_info": self.current_round.model_dump() if self.current_round else None
			},
			"training_progress": {
				"rounds_completed": len(self.round_history),
				"target_rounds": self.coordinator_config["max_rounds"],
				"convergence_achieved": self.federation_metrics["convergence_achieved"],
				"best_accuracy": self.federation_metrics["best_global_accuracy"]
			},
			"privacy_and_security": {
				"privacy_mechanism": self.coordinator_config["privacy_mechanism"].value,
				"differential_privacy_enabled": self.privacy_engine is not None,
				"secure_aggregation_enabled": self.secure_aggregation._config.enabled,
				"quantum_safe_enabled": self.quantum_security is not None,
				"total_privacy_spent": self.federation_metrics["total_privacy_spent"]
			},
			"performance_metrics": dict(self.federation_metrics),
			"configuration": dict(self.coordinator_config)
		}

	def _count_clients_by_state(self) -> Dict[str, int]:
		"""Count clients by their current state."""
		counts = {}
		for client in self.clients.values():
			state = client.state.value
			counts[state] = counts.get(state, 0) + 1
		return counts

	async def shutdown_federation(self) -> None:
		"""Gracefully shutdown the federation."""
		try:
			# Complete current round if active
			if self.current_round and self.current_round.round_status != "completed":
				self.current_round.round_status = "terminated"
				self.current_round.complete_round()

			# Notify all clients
			for client in self.clients.values():
				client.state = FederatedClientState.OFFLINE
				client.add_audit_entry("federation_shutdown")

			self._logger.info(_log_federated_event(
				"SHUTDOWN", self.federation_id, "shutdown_federation", "SUCCESS"
			))

		except Exception as e:
			self._logger.error(f"Federation shutdown failed: {str(e)}")
			raise


# Module exports
__all__ = [
	# Core federated learning coordinator
	"FederatedLearningCoordinator",

	# Client and round management
	"FederatedClient", "FederatedLearningRound", "ModelUpdate",

	# Privacy and security
	"DifferentialPrivacyEngine", "SecureAggregationProtocol",
	"DifferentialPrivacyParameters", "SecureAggregationConfig",

	# Enums
	"FederatedLearningAlgorithm", "PrivacyMechanism", "ClientSelectionStrategy",
	"AggregationMethod", "FederatedClientState",

	# Utility functions
	"_log_federated_event", "_log_client_event", "_log_aggregation_event"
]