"""
Federated Learning Models

Database models for federated learning system, cross-twin knowledge sharing,
secure aggregation, privacy-preserving ML, and collaborative model training.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class FLFederation(Model, AuditMixin, BaseMixin):
	"""
	Federated learning federation configuration and management.
	
	Represents a federation of digital twins collaborating on
	machine learning tasks with privacy-preserving protocols.
	"""
	__tablename__ = 'fl_federation'
	
	# Identity
	federation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Federation Information
	federation_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	federation_type = Column(String(50), nullable=False, index=True)  # horizontal, vertical, transfer
	domain = Column(String(100), nullable=False, index=True)  # healthcare, manufacturing, finance, etc.
	
	# Governance and Policies
	governance_model = Column(String(50), default='democratic', index=True)  # democratic, hierarchical, consortium
	privacy_level = Column(String(20), default='high', index=True)  # low, medium, high, maximum
	data_sharing_policy = Column(JSON, default=dict)  # Data sharing rules and restrictions
	model_sharing_policy = Column(JSON, default=dict)  # Model sharing policies
	
	# Technical Configuration
	aggregation_algorithm = Column(String(50), default='fedavg', index=True)  # fedavg, fedprox, scaffold
	communication_protocol = Column(String(50), default='grpc')  # grpc, rest, mqtt
	encryption_method = Column(String(50), default='aes256')  # Encryption for model updates
	differential_privacy_enabled = Column(Boolean, default=True)
	differential_privacy_epsilon = Column(Float, default=1.0)  # Privacy budget
	
	# Membership and Status
	status = Column(String(20), default='active', index=True)  # active, inactive, suspended, archived
	is_open = Column(Boolean, default=False)  # Open for new participants
	requires_approval = Column(Boolean, default=True)  # Require approval for joining
	max_participants = Column(Integer, default=50)
	current_participant_count = Column(Integer, default=0)
	
	# Quality and Trust
	min_data_quality_score = Column(Float, default=0.7)  # Minimum data quality for participation
	trust_threshold = Column(Float, default=0.8)  # Minimum trust score for participation
	reputation_system_enabled = Column(Boolean, default=True)
	
	# Coordination
	coordinator_id = Column(String(36), nullable=False, index=True)  # Federation coordinator
	backup_coordinators = Column(JSON, default=list)  # List of backup coordinator IDs
	round_duration_minutes = Column(Integer, default=60)  # Duration of each federated round
	min_participants_per_round = Column(Integer, default=3)  # Minimum participants needed
	
	# Performance and Metrics
	total_rounds_completed = Column(Integer, default=0)
	best_global_accuracy = Column(Float, nullable=True)
	average_round_duration = Column(Float, nullable=True)  # Average round duration in minutes
	total_models_trained = Column(Integer, default=0)
	
	# Financial and Incentives
	incentive_mechanism = Column(String(50), nullable=True)  # token, reputation, none
	reward_distribution = Column(JSON, default=dict)  # How rewards are distributed
	cost_sharing_model = Column(JSON, default=dict)  # Cost sharing arrangements
	
	# Relationships
	participants = relationship("FLParticipant", back_populates="federation")
	learning_tasks = relationship("FLLearningTask", back_populates="federation")
	training_rounds = relationship("FLTrainingRound", back_populates="federation")
	
	def __repr__(self):
		return f"<FLFederation {self.federation_name}>"
	
	def can_start_new_round(self) -> bool:
		"""Check if federation can start a new training round"""
		active_participants = len([p for p in self.participants if p.status == 'active'])
		return (
			self.status == 'active' and
			active_participants >= self.min_participants_per_round and
			self.current_participant_count >= self.min_participants_per_round
		)
	
	def calculate_participation_rate(self) -> float:
		"""Calculate average participation rate across rounds"""
		if self.total_rounds_completed == 0:
			return 0.0
		
		# Implementation would calculate actual participation rate
		# For now, return a placeholder
		return 0.85
	
	def update_global_metrics(self, accuracy: float, round_duration: float):
		"""Update federation-wide metrics"""
		if self.best_global_accuracy is None or accuracy > self.best_global_accuracy:
			self.best_global_accuracy = accuracy
		
		if self.average_round_duration is None:
			self.average_round_duration = round_duration
		else:
			# Running average
			total_duration = self.average_round_duration * self.total_rounds_completed
			self.average_round_duration = (total_duration + round_duration) / (self.total_rounds_completed + 1)
		
		self.total_rounds_completed += 1


class FLParticipant(Model, AuditMixin, BaseMixin):
	"""
	Federated learning participant (digital twin) in a federation.
	
	Represents an individual digital twin participating in
	federated learning with capability and trust tracking.
	"""
	__tablename__ = 'fl_participant'
	
	# Identity
	participant_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	federation_id = Column(String(36), ForeignKey('fl_federation.federation_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Participant Information
	participant_name = Column(String(200), nullable=False, index=True)
	digital_twin_id = Column(String(36), nullable=False, index=True)
	organization = Column(String(200), nullable=True)
	contact_email = Column(String(200), nullable=True)
	
	# Participation Status
	status = Column(String(20), default='pending', index=True)  # pending, active, inactive, suspended, banned
	joined_at = Column(DateTime, nullable=False, index=True)
	last_activity = Column(DateTime, nullable=True)
	approval_status = Column(String(20), default='pending')  # pending, approved, rejected
	approved_by = Column(String(36), nullable=True)
	approval_date = Column(DateTime, nullable=True)
	
	# Capabilities and Resources
	compute_capacity = Column(JSON, default=dict)  # Available compute resources
	data_characteristics = Column(JSON, default=dict)  # Data schema, size, quality
	supported_algorithms = Column(JSON, default=list)  # Supported ML algorithms
	technical_capabilities = Column(JSON, default=dict)  # Technical capabilities
	
	# Privacy and Security
	privacy_requirements = Column(JSON, default=dict)  # Privacy requirements and constraints
	security_level = Column(String(20), default='standard')  # standard, high, maximum
	data_sovereignty_restrictions = Column(JSON, default=dict)  # Data sovereignty rules
	compliance_certifications = Column(JSON, default=list)  # Compliance certifications
	
	# Performance and Trust
	trust_score = Column(Float, default=0.5, index=True)  # 0-1 trust score
	reputation_score = Column(Float, default=0.5)  # 0-1 reputation score
	data_quality_score = Column(Float, nullable=True)  # 0-1 data quality score
	model_contribution_score = Column(Float, default=0.0)  # Contribution to global model
	
	# Participation Metrics
	rounds_participated = Column(Integer, default=0)
	rounds_invited = Column(Integer, default=0)
	successful_contributions = Column(Integer, default=0)
	failed_contributions = Column(Integer, default=0)
	average_training_time = Column(Float, nullable=True)  # Average local training time
	
	# Network and Communication
	endpoint_url = Column(String(500), nullable=True)  # Communication endpoint
	public_key = Column(Text, nullable=True)  # Public key for encryption
	communication_preferences = Column(JSON, default=dict)  # Communication settings
	bandwidth_constraints = Column(JSON, default=dict)  # Network bandwidth limits
	
	# Incentives and Rewards
	total_rewards_earned = Column(Float, default=0.0)
	contribution_weight = Column(Float, default=1.0)  # Weight in aggregation
	incentive_preferences = Column(JSON, default=dict)  # Reward preferences
	
	# Relationships
	federation = relationship("FLFederation", back_populates="participants")
	local_models = relationship("FLLocalModel", back_populates="participant")
	contributions = relationship("FLModelUpdate", back_populates="participant")
	
	def __repr__(self):
		return f"<FLParticipant {self.participant_name}>"
	
	def calculate_participation_rate(self) -> float:
		"""Calculate participation rate as percentage of invited rounds"""
		if self.rounds_invited == 0:
			return 0.0
		return (self.rounds_participated / self.rounds_invited) * 100
	
	def calculate_success_rate(self) -> float:
		"""Calculate success rate of contributions"""
		total_contributions = self.successful_contributions + self.failed_contributions
		if total_contributions == 0:
			return 0.0
		return (self.successful_contributions / total_contributions) * 100
	
	def update_trust_score(self, performance_factor: float, reliability_factor: float):
		"""Update trust score based on performance and reliability"""
		# Weighted average of current trust score and new factors
		weight = 0.1  # Learning rate
		new_score = (performance_factor + reliability_factor) / 2
		self.trust_score = (1 - weight) * self.trust_score + weight * new_score
		self.trust_score = max(0.0, min(1.0, self.trust_score))  # Clamp to [0, 1]
	
	def is_eligible_for_round(self, task_requirements: Dict[str, Any]) -> bool:
		"""Check if participant is eligible for a training round"""
		if self.status != 'active':
			return False
		
		if self.trust_score < task_requirements.get('min_trust_score', 0.5):
			return False
		
		if self.data_quality_score and self.data_quality_score < task_requirements.get('min_data_quality', 0.7):
			return False
		
		return True


class FLLearningTask(Model, AuditMixin, BaseMixin):
	"""
	Federated learning task definition and configuration.
	
	Defines a specific machine learning task to be solved
	collaboratively across federation participants.
	"""
	__tablename__ = 'fl_learning_task'
	
	# Identity
	task_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	federation_id = Column(String(36), ForeignKey('fl_federation.federation_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Task Information
	task_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=False)
	task_type = Column(String(50), nullable=False, index=True)  # classification, regression, clustering
	domain = Column(String(100), nullable=False)  # Application domain
	use_case = Column(String(200), nullable=True)  # Specific use case
	
	# Task Configuration
	algorithm_type = Column(String(50), nullable=False)  # neural_network, tree_ensemble, linear
	model_architecture = Column(JSON, default=dict)  # Model architecture specification
	hyperparameters = Column(JSON, default=dict)  # Training hyperparameters
	optimization_objective = Column(String(50), nullable=False)  # accuracy, f1_score, mse, etc.
	
	# Data Requirements
	data_schema = Column(JSON, default=dict)  # Required data schema
	min_data_samples = Column(Integer, default=1000)  # Minimum data samples per participant
	data_quality_requirements = Column(JSON, default=dict)  # Data quality constraints
	feature_requirements = Column(JSON, default=dict)  # Required features
	
	# Training Configuration
	max_training_rounds = Column(Integer, default=100)
	convergence_threshold = Column(Float, default=0.001)  # Convergence criteria
	target_accuracy = Column(Float, nullable=True)  # Target accuracy to achieve
	early_stopping_patience = Column(Integer, default=10)  # Rounds without improvement
	
	# Privacy and Security
	differential_privacy_budget = Column(Float, nullable=True)  # Privacy budget for this task
	secure_aggregation_required = Column(Boolean, default=True)
	homomorphic_encryption = Column(Boolean, default=False)
	trusted_execution_environment = Column(Boolean, default=False)
	
	# Resource Requirements
	min_participants = Column(Integer, default=3)
	max_participants = Column(Integer, nullable=True)
	estimated_round_duration = Column(Integer, default=30)  # Minutes per round
	compute_requirements = Column(JSON, default=dict)  # Compute resource needs
	
	# Status and Progress
	status = Column(String(20), default='draft', index=True)  # draft, active, completed, failed, cancelled
	current_round = Column(Integer, default=0)
	best_global_accuracy = Column(Float, nullable=True)
	last_improvement_round = Column(Integer, nullable=True)
	
	# Timing
	created_by = Column(String(36), nullable=False, index=True)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	deadline = Column(DateTime, nullable=True)
	
	# Results and Metrics
	final_model_accuracy = Column(Float, nullable=True)
	final_model_metrics = Column(JSON, default=dict)  # Final model performance metrics
	convergence_metrics = Column(JSON, default=dict)  # Convergence history
	participant_contributions = Column(JSON, default=dict)  # Contribution breakdown
	
	# Relationships
	federation = relationship("FLFederation", back_populates="learning_tasks")
	training_rounds = relationship("FLTrainingRound", back_populates="task")
	global_models = relationship("FLGlobalModel", back_populates="task")
	
	def __repr__(self):
		return f"<FLLearningTask {self.task_name}>"
	
	def is_converged(self, current_accuracy: float) -> bool:
		"""Check if task has converged based on accuracy improvement"""
		if self.best_global_accuracy is None:
			return False
		
		improvement = current_accuracy - self.best_global_accuracy
		return improvement < self.convergence_threshold
	
	def should_early_stop(self) -> bool:
		"""Check if early stopping criteria are met"""
		if self.last_improvement_round is None:
			return False
		
		rounds_without_improvement = self.current_round - self.last_improvement_round
		return rounds_without_improvement >= self.early_stopping_patience
	
	def update_best_accuracy(self, accuracy: float):
		"""Update best global accuracy and last improvement round"""
		if self.best_global_accuracy is None or accuracy > self.best_global_accuracy:
			self.best_global_accuracy = accuracy
			self.last_improvement_round = self.current_round
	
	def calculate_progress(self) -> float:
		"""Calculate task progress as percentage"""
		if self.max_training_rounds == 0:
			return 0.0
		return min(100.0, (self.current_round / self.max_training_rounds) * 100)


class FLTrainingRound(Model, AuditMixin, BaseMixin):
	"""
	Individual federated learning training round execution.
	
	Tracks the execution of a single federated learning round
	including participant selection, model updates, and aggregation.
	"""
	__tablename__ = 'fl_training_round'
	
	# Identity
	round_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	federation_id = Column(String(36), ForeignKey('fl_federation.federation_id'), nullable=False, index=True)
	task_id = Column(String(36), ForeignKey('fl_learning_task.task_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Round Information
	round_number = Column(Integer, nullable=False, index=True)
	round_type = Column(String(50), default='standard')  # standard, evaluation, calibration
	
	# Participant Selection
	selected_participants = Column(JSON, default=list)  # List of selected participant IDs
	participant_selection_strategy = Column(String(50), default='random')  # random, stratified, performance
	target_participant_count = Column(Integer, default=10)
	actual_participant_count = Column(Integer, default=0)
	
	# Round Execution
	status = Column(String(20), default='preparing', index=True)  # preparing, training, aggregating, completed, failed
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True)
	timeout_at = Column(DateTime, nullable=True)  # Round timeout
	duration_minutes = Column(Float, nullable=True)
	
	# Model State
	global_model_before = Column(LargeBinary, nullable=True)  # Serialized global model before round
	global_model_after = Column(LargeBinary, nullable=True)  # Serialized global model after round
	model_updates_received = Column(Integer, default=0)
	model_updates_expected = Column(Integer, default=0)
	
	# Aggregation
	aggregation_algorithm = Column(String(50), default='fedavg')
	aggregation_weights = Column(JSON, default=dict)  # Participant weights in aggregation
	aggregation_started_at = Column(DateTime, nullable=True)
	aggregation_completed_at = Column(DateTime, nullable=True)
	aggregation_duration_seconds = Column(Float, nullable=True)
	
	# Performance Metrics
	global_accuracy_before = Column(Float, nullable=True)
	global_accuracy_after = Column(Float, nullable=True)
	accuracy_improvement = Column(Float, nullable=True)
	convergence_metrics = Column(JSON, default=dict)  # Detailed convergence metrics
	
	# Privacy and Security
	differential_privacy_applied = Column(Boolean, default=False)
	noise_scale = Column(Float, nullable=True)  # DP noise scale
	privacy_budget_consumed = Column(Float, nullable=True)
	secure_aggregation_used = Column(Boolean, default=False)
	
	# Quality and Validation
	model_validation_metrics = Column(JSON, default=dict)  # Validation results
	data_quality_checks = Column(JSON, default=dict)  # Data quality validation
	byzantine_detection_results = Column(JSON, default=dict)  # Byzantine participant detection
	
	# Communication and Network
	total_communication_bytes = Column(Float, default=0.0)  # Total data transferred
	average_communication_delay = Column(Float, nullable=True)  # Average network delay
	failed_communications = Column(Integer, default=0)
	network_issues = Column(JSON, default=list)  # Network-related issues
	
	# Relationships
	federation = relationship("FLFederation", back_populates="training_rounds")
	task = relationship("FLLearningTask", back_populates="training_rounds")
	model_updates = relationship("FLModelUpdate", back_populates="training_round")
	
	def __repr__(self):
		return f"<FLTrainingRound {self.round_number} for {self.task.task_name}>"
	
	def calculate_participation_rate(self) -> float:
		"""Calculate actual vs expected participation rate"""
		if self.target_participant_count == 0:
			return 0.0
		return (self.actual_participant_count / self.target_participant_count) * 100
	
	def calculate_success_rate(self) -> float:
		"""Calculate model update success rate"""
		if self.model_updates_expected == 0:
			return 0.0
		return (self.model_updates_received / self.model_updates_expected) * 100
	
	def is_timed_out(self) -> bool:
		"""Check if round has timed out"""
		if self.timeout_at is None:
			return False
		return datetime.utcnow() > self.timeout_at
	
	def calculate_improvement(self):
		"""Calculate accuracy improvement from this round"""
		if self.global_accuracy_before and self.global_accuracy_after:
			self.accuracy_improvement = self.global_accuracy_after - self.global_accuracy_before
		return self.accuracy_improvement


class FLLocalModel(Model, AuditMixin, BaseMixin):
	"""
	Local model maintained by a federated learning participant.
	
	Tracks the local model state, training data, and performance
	for each participant in the federation.
	"""
	__tablename__ = 'fl_local_model'
	
	# Identity
	local_model_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	participant_id = Column(String(36), ForeignKey('fl_participant.participant_id'), nullable=False, index=True)
	task_id = Column(String(36), ForeignKey('fl_learning_task.task_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Model Information
	model_name = Column(String(200), nullable=False)
	model_version = Column(String(20), default='1.0.0')
	algorithm_type = Column(String(50), nullable=False)
	
	# Model State
	model_parameters = Column(LargeBinary, nullable=True)  # Serialized model parameters
	model_weights = Column(LargeBinary, nullable=True)  # Serialized model weights
	model_architecture = Column(JSON, default=dict)  # Model architecture
	hyperparameters = Column(JSON, default=dict)  # Training hyperparameters
	
	# Training Data
	training_data_size = Column(Integer, default=0)  # Number of training samples
	validation_data_size = Column(Integer, default=0)  # Number of validation samples
	data_schema = Column(JSON, default=dict)  # Local data schema
	data_distribution = Column(JSON, default=dict)  # Data distribution statistics
	data_quality_metrics = Column(JSON, default=dict)  # Data quality assessments
	
	# Performance Metrics
	local_accuracy = Column(Float, nullable=True)  # Local model accuracy
	local_loss = Column(Float, nullable=True)  # Local training loss
	validation_accuracy = Column(Float, nullable=True)  # Validation accuracy
	validation_loss = Column(Float, nullable=True)  # Validation loss
	performance_metrics = Column(JSON, default=dict)  # Detailed performance metrics
	
	# Training History
	total_training_rounds = Column(Integer, default=0)
	last_training_round = Column(Integer, nullable=True)
	training_epochs_per_round = Column(Integer, default=1)
	total_training_time_minutes = Column(Float, default=0.0)
	average_training_time_per_round = Column(Float, nullable=True)
	
	# Model Updates
	last_update_timestamp = Column(DateTime, nullable=True)
	update_frequency = Column(String(20), default='per_round')  # per_round, scheduled, event_driven
	pending_updates = Column(Boolean, default=False)
	model_drift_detected = Column(Boolean, default=False)
	
	# Privacy and Security
	differential_privacy_enabled = Column(Boolean, default=False)
	privacy_budget_used = Column(Float, default=0.0)
	gradient_clipping_norm = Column(Float, nullable=True)
	model_poisoning_detection = Column(JSON, default=dict)  # Poisoning detection results
	
	# Resource Usage
	memory_usage_mb = Column(Float, nullable=True)
	cpu_time_seconds = Column(Float, nullable=True)
	gpu_time_seconds = Column(Float, nullable=True)
	storage_size_mb = Column(Float, nullable=True)
	
	# Status and Health
	status = Column(String(20), default='active', index=True)  # active, inactive, error, updating
	health_score = Column(Float, default=1.0)  # 0-1 health score
	last_health_check = Column(DateTime, nullable=True)
	error_log = Column(JSON, default=list)  # Error history
	
	# Relationships
	participant = relationship("FLParticipant", back_populates="local_models")
	model_updates = relationship("FLModelUpdate", back_populates="local_model")
	
	def __repr__(self):
		return f"<FLLocalModel {self.model_name} by {self.participant.participant_name}>"
	
	def calculate_training_efficiency(self) -> float:
		"""Calculate training efficiency based on accuracy vs time"""
		if not self.local_accuracy or not self.total_training_time_minutes:
			return 0.0
		
		# Simple efficiency metric: accuracy per minute of training
		return self.local_accuracy / self.total_training_time_minutes
	
	def update_performance_metrics(self, accuracy: float, loss: float, round_time: float):
		"""Update performance metrics after a training round"""
		self.local_accuracy = accuracy
		self.local_loss = loss
		self.total_training_rounds += 1
		self.total_training_time_minutes += round_time
		
		if self.total_training_rounds > 0:
			self.average_training_time_per_round = self.total_training_time_minutes / self.total_training_rounds
		
		self.last_update_timestamp = datetime.utcnow()
	
	def detect_model_drift(self, threshold: float = 0.1) -> bool:
		"""Detect if model performance has drifted significantly"""
		# Implementation would compare current performance with baseline
		# For now, return False as placeholder
		return False


class FLModelUpdate(Model, AuditMixin, BaseMixin):
	"""
	Model update contribution from a participant to the global model.
	
	Tracks individual model updates sent by participants during
	federated learning rounds with validation and security checks.
	"""
	__tablename__ = 'fl_model_update'
	
	# Identity
	update_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	training_round_id = Column(String(36), ForeignKey('fl_training_round.round_id'), nullable=False, index=True)
	participant_id = Column(String(36), ForeignKey('fl_participant.participant_id'), nullable=False, index=True)
	local_model_id = Column(String(36), ForeignKey('fl_local_model.local_model_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Update Information
	update_type = Column(String(50), default='gradients')  # gradients, parameters, weights
	model_delta = Column(LargeBinary, nullable=True)  # Serialized model update
	update_size_bytes = Column(Integer, default=0)
	compression_algorithm = Column(String(50), nullable=True)  # gzip, lz4, none
	compression_ratio = Column(Float, nullable=True)
	
	# Training Metadata
	local_epochs = Column(Integer, default=1)  # Number of local training epochs
	batch_size = Column(Integer, nullable=True)
	learning_rate = Column(Float, nullable=True)
	training_samples_used = Column(Integer, default=0)
	local_training_loss = Column(Float, nullable=True)
	local_training_accuracy = Column(Float, nullable=True)
	
	# Quality and Validation
	validation_metrics = Column(JSON, default=dict)  # Local validation results
	gradient_norm = Column(Float, nullable=True)  # L2 norm of gradients
	parameter_norm = Column(Float, nullable=True)  # L2 norm of parameters
	update_quality_score = Column(Float, nullable=True)  # 0-1 quality score
	passes_quality_checks = Column(Boolean, default=True)
	
	# Privacy and Security
	differential_privacy_applied = Column(Boolean, default=False)
	noise_added = Column(Float, nullable=True)  # Amount of DP noise added
	gradient_clipped = Column(Boolean, default=False)
	clipping_norm = Column(Float, nullable=True)
	encrypted_payload = Column(Boolean, default=False)
	signature_verified = Column(Boolean, default=False)
	
	# Timing and Communication
	created_at_participant = Column(DateTime, nullable=False)  # When update was created locally
	sent_at = Column(DateTime, nullable=False, index=True)  # When update was sent
	received_at = Column(DateTime, nullable=False)  # When update was received
	processed_at = Column(DateTime, nullable=True)  # When update was processed
	communication_delay_ms = Column(Float, nullable=True)  # Network delay
	
	# Status and Processing
	status = Column(String(20), default='received', index=True)  # received, validated, processed, rejected
	validation_status = Column(String(20), default='pending')  # pending, passed, failed
	rejection_reason = Column(Text, nullable=True)  # Reason for rejection
	processed_by = Column(String(36), nullable=True)  # Who processed the update
	
	# Aggregation
	aggregation_weight = Column(Float, default=1.0)  # Weight in aggregation
	contribution_score = Column(Float, nullable=True)  # Contribution to global model
	included_in_global_model = Column(Boolean, default=False)
	aggregation_timestamp = Column(DateTime, nullable=True)
	
	# Byzantine Detection
	byzantine_score = Column(Float, default=0.0)  # 0-1 Byzantine likelihood
	anomaly_detected = Column(Boolean, default=False)
	similarity_scores = Column(JSON, default=dict)  # Similarity to other updates
	outlier_detection_results = Column(JSON, default=dict)  # Outlier analysis
	
	# Relationships
	training_round = relationship("FLTrainingRound", back_populates="model_updates")
	participant = relationship("FLParticipant", back_populates="contributions")
	local_model = relationship("FLLocalModel", back_populates="model_updates")
	
	def __repr__(self):
		return f"<FLModelUpdate from {self.participant.participant_name}>"
	
	def calculate_communication_delay(self) -> float:
		"""Calculate communication delay in milliseconds"""
		if self.sent_at and self.received_at:
			delay = self.received_at - self.sent_at
			self.communication_delay_ms = delay.total_seconds() * 1000
			return self.communication_delay_ms
		return 0.0
	
	def validate_update(self) -> bool:
		"""Perform validation checks on the model update"""
		checks_passed = 0
		total_checks = 0
		
		# Check gradient/parameter norms
		total_checks += 1
		if self.gradient_norm and self.gradient_norm < 100.0:  # Reasonable gradient norm
			checks_passed += 1
		
		# Check update size
		total_checks += 1
		if self.update_size_bytes > 0 and self.update_size_bytes < 100 * 1024 * 1024:  # < 100MB
			checks_passed += 1
		
		# Check training metrics
		total_checks += 1
		if self.local_training_accuracy and 0.0 <= self.local_training_accuracy <= 1.0:
			checks_passed += 1
		
		self.passes_quality_checks = (checks_passed / total_checks) >= 0.8
		self.validation_status = 'passed' if self.passes_quality_checks else 'failed'
		
		return self.passes_quality_checks
	
	def calculate_byzantine_score(self, other_updates: List['FLModelUpdate']) -> float:
		"""Calculate Byzantine score based on similarity to other updates"""
		# Implementation would compare this update with others to detect anomalies
		# For now, return a low score indicating non-Byzantine behavior
		self.byzantine_score = 0.1
		return self.byzantine_score


class FLGlobalModel(Model, AuditMixin, BaseMixin):
	"""
	Global federated learning model aggregated from participant contributions.
	
	Represents the global model state at different points in the
	federated learning process with performance tracking.
	"""
	__tablename__ = 'fl_global_model'
	
	# Identity
	global_model_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	task_id = Column(String(36), ForeignKey('fl_learning_task.task_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Model Information
	model_name = Column(String(200), nullable=False)
	model_version = Column(String(20), nullable=False)
	round_number = Column(Integer, nullable=False, index=True)
	algorithm_type = Column(String(50), nullable=False)
	
	# Model State
	model_parameters = Column(LargeBinary, nullable=True)  # Serialized global model
	model_architecture = Column(JSON, default=dict)  # Model architecture
	hyperparameters = Column(JSON, default=dict)  # Global hyperparameters
	aggregation_metadata = Column(JSON, default=dict)  # Aggregation details
	
	# Performance Metrics
	global_accuracy = Column(Float, nullable=True)
	global_loss = Column(Float, nullable=True)
	validation_accuracy = Column(Float, nullable=True)
	validation_loss = Column(Float, nullable=True)
	test_accuracy = Column(Float, nullable=True)
	test_loss = Column(Float, nullable=True)
	detailed_metrics = Column(JSON, default=dict)  # Detailed performance metrics
	
	# Aggregation Information
	contributing_participants = Column(JSON, default=list)  # List of contributor IDs
	participant_weights = Column(JSON, default=dict)  # Aggregation weights
	total_data_samples = Column(Integer, default=0)  # Total samples across participants
	aggregation_algorithm = Column(String(50), default='fedavg')
	aggregation_timestamp = Column(DateTime, nullable=False)
	
	# Quality and Validation
	model_quality_score = Column(Float, nullable=True)  # 0-1 quality score
	convergence_metrics = Column(JSON, default=dict)  # Convergence analysis
	stability_metrics = Column(JSON, default=dict)  # Model stability analysis
	robustness_score = Column(Float, nullable=True)  # Robustness assessment
	
	# Privacy and Security
	differential_privacy_applied = Column(Boolean, default=False)
	total_privacy_budget_used = Column(Float, nullable=True)
	model_poisoning_detected = Column(Boolean, default=False)
	security_validation_passed = Column(Boolean, default=True)
	
	# Comparison with Previous Version
	improvement_over_previous = Column(Float, nullable=True)  # Accuracy improvement
	parameter_change_magnitude = Column(Float, nullable=True)  # L2 norm of parameter changes
	convergence_rate = Column(Float, nullable=True)  # Rate of convergence
	
	# Distribution and Deployment
	is_final_model = Column(Boolean, default=False)
	deployment_ready = Column(Boolean, default=False)
	deployment_metadata = Column(JSON, default=dict)  # Deployment information
	download_count = Column(Integer, default=0)  # How many times downloaded
	
	# Relationships
	task = relationship("FLLearningTask", back_populates="global_models")
	
	def __repr__(self):
		return f"<FLGlobalModel {self.model_name} v{self.model_version} Round {self.round_number}>"
	
	def calculate_improvement(self, previous_model: Optional['FLGlobalModel']) -> float:
		"""Calculate improvement over previous model version"""
		if not previous_model or not self.global_accuracy or not previous_model.global_accuracy:
			return 0.0
		
		self.improvement_over_previous = self.global_accuracy - previous_model.global_accuracy
		return self.improvement_over_previous
	
	def assess_convergence(self, recent_models: List['FLGlobalModel']) -> Dict[str, float]:
		"""Assess convergence based on recent model performance"""
		if len(recent_models) < 2:
			return {'converged': False, 'rate': 0.0, 'stability': 0.0}
		
		# Calculate accuracy variance in recent models
		accuracies = [m.global_accuracy for m in recent_models if m.global_accuracy]
		if len(accuracies) < 2:
			return {'converged': False, 'rate': 0.0, 'stability': 0.0}
		
		import statistics
		variance = statistics.variance(accuracies)
		stability = 1.0 - min(variance, 1.0)  # Higher stability = lower variance
		
		# Simple convergence check based on variance
		converged = variance < 0.001
		
		convergence_data = {
			'converged': converged,
			'rate': self.convergence_rate or 0.0,
			'stability': stability,
			'variance': variance
		}
		
		self.convergence_metrics = convergence_data
		return convergence_data
	
	def validate_model_integrity(self) -> bool:
		"""Validate model integrity and security"""
		# Implementation would perform security checks
		# For now, assume model passes validation
		self.security_validation_passed = True
		return True