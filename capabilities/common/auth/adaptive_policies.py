"""
Adaptive Policy Learning Engine

Machine learning-powered policy optimization system that continuously learns
from access decisions and security outcomes to automatically refine authorization policies.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import json
import math
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

class PolicyOutcome(str, Enum):
	"""Outcomes of policy decisions"""
	SUCCESS = "success"                    # Policy worked as expected
	FALSE_POSITIVE = "false_positive"      # Incorrectly denied legitimate access
	FALSE_NEGATIVE = "false_negative"      # Incorrectly allowed malicious access
	SECURITY_INCIDENT = "security_incident" # Led to security breach
	USER_COMPLAINT = "user_complaint"      # User complained about decision
	ADMIN_OVERRIDE = "admin_override"      # Admin had to override decision

class LearningMode(str, Enum):
	"""Policy learning modes"""
	PASSIVE = "passive"      # Learn but don't auto-apply changes
	SUPERVISED = "supervised" # Learn with human oversight
	ACTIVE = "active"        # Automatically apply learned changes
	A_B_TEST = "a_b_test"   # Test new policies against existing ones

class PolicyType(str, Enum):
	"""Types of policies that can be learned"""
	ACCESS_CONTROL = "access_control"      # Basic access decisions
	RISK_BASED = "risk_based"             # Risk-based authentication
	TIME_BASED = "time_based"             # Time-based access controls
	LOCATION_BASED = "location_based"     # Location-based policies
	RESOURCE_SPECIFIC = "resource_specific" # Resource-specific rules
	BEHAVIORAL = "behavioral"             # Behavioral pattern policies

class PolicyDecisionRecord(BaseModel):
	"""Record of a policy decision and its outcome"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Decision record identifier")
	policy_id: str = Field(..., description="Policy that made the decision")
	user_id: str = Field(..., description="User involved in decision")
	
	# Decision context
	resource: str = Field(..., description="Resource being accessed")
	action: str = Field(..., description="Action being performed")
	context: Dict[str, Any] = Field(..., description="Decision context")
	
	# Decision details
	decision: str = Field(..., description="Policy decision (allow/deny)")
	confidence: float = Field(..., description="Decision confidence", ge=0.0, le=1.0)
	reasoning: List[str] = Field(default_factory=list, description="Decision reasoning")
	
	# Timing
	decided_at: datetime = Field(default_factory=datetime.utcnow, description="Decision timestamp")
	outcome_recorded_at: Optional[datetime] = Field(default=None, description="When outcome was recorded")
	
	# Outcome tracking
	outcome: Optional[PolicyOutcome] = Field(default=None, description="Actual outcome")
	outcome_details: Dict[str, Any] = Field(default_factory=dict, description="Outcome details")
	feedback_source: Optional[str] = Field(default=None, description="Source of outcome feedback")
	
	# Learning metrics
	was_correct: Optional[bool] = Field(default=None, description="Was the decision correct")
	impact_score: float = Field(default=0.0, description="Impact of decision", ge=0.0, le=1.0)

class PolicyMetrics(BaseModel):
	"""Policy performance metrics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	policy_id: str = Field(..., description="Policy identifier")
	
	# Basic metrics
	total_decisions: int = Field(default=0, description="Total decisions made")
	correct_decisions: int = Field(default=0, description="Correct decisions")
	false_positives: int = Field(default=0, description="False positive count")
	false_negatives: int = Field(default=0, description="False negative count")
	
	# Performance metrics
	accuracy: float = Field(default=0.0, description="Decision accuracy", ge=0.0, le=1.0)
	precision: float = Field(default=0.0, description="Precision (1 - FP rate)", ge=0.0, le=1.0)
	recall: float = Field(default=0.0, description="Recall (1 - FN rate)", ge=0.0, le=1.0)
	f1_score: float = Field(default=0.0, description="F1 score", ge=0.0, le=1.0)
	
	# User experience metrics
	user_satisfaction: float = Field(default=0.5, description="User satisfaction", ge=0.0, le=1.0)
	admin_overrides: int = Field(default=0, description="Admin override count")
	user_complaints: int = Field(default=0, description="User complaint count")
	
	# Time-based metrics
	avg_decision_time: float = Field(default=0.0, description="Average decision time (ms)")
	trend_direction: str = Field(default="stable", description="Performance trend")
	
	# Calculation timestamp
	calculated_at: datetime = Field(default_factory=datetime.utcnow, description="Metrics calculation time")

class AdaptivePolicyRule(BaseModel):
	"""Learned policy rule"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Rule identifier")
	policy_id: str = Field(..., description="Parent policy identifier")
	
	# Rule definition
	conditions: List[Dict[str, Any]] = Field(..., description="Rule conditions")
	action: str = Field(..., description="Action to take (allow/deny/challenge)")
	priority: int = Field(default=100, description="Rule priority (lower = higher priority)")
	
	# Learning metadata
	learned_from_samples: int = Field(..., description="Number of samples used to learn rule")
	confidence: float = Field(..., description="Confidence in rule", ge=0.0, le=1.0)
	support: float = Field(..., description="Rule support (coverage)", ge=0.0, le=1.0)
	
	# Performance tracking
	applications: int = Field(default=0, description="Times rule has been applied")
	successes: int = Field(default=0, description="Successful applications")
	effectiveness: float = Field(default=0.0, description="Rule effectiveness", ge=0.0, le=1.0)
	
	# Lifecycle
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Rule creation time")
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
	is_active: bool = Field(default=True, description="Rule is active")

class PolicyLearningSession(BaseModel):
	"""Learning session for policy optimization"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Learning session identifier")
	policy_id: str = Field(..., description="Policy being optimized")
	
	# Session configuration
	learning_mode: LearningMode = Field(..., description="Learning mode")
	optimization_target: str = Field(..., description="What to optimize (accuracy, user_satisfaction, etc.)")
	
	# Data used
	training_samples: int = Field(..., description="Number of training samples")
	validation_samples: int = Field(..., description="Number of validation samples")
	sample_date_range: Tuple[datetime, datetime] = Field(..., description="Date range of samples")
	
	# Results
	rules_generated: int = Field(default=0, description="New rules generated")
	rules_modified: int = Field(default=0, description="Existing rules modified")
	rules_removed: int = Field(default=0, description="Rules removed")
	
	# Performance improvement
	accuracy_before: float = Field(..., description="Accuracy before learning")
	accuracy_after: float = Field(..., description="Accuracy after learning")
	improvement: float = Field(..., description="Performance improvement")
	
	# Session metadata
	started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
	completed_at: Optional[datetime] = Field(default=None, description="Session completion time")
	status: str = Field(default="running", description="Session status")

@dataclass
class FeatureImportance:
	"""Importance of features in policy decisions"""
	feature_name: str
	importance_score: float
	sample_count: int
	correlation_with_outcome: float

class DecisionTreeNode:
	"""Node in learned decision tree"""
	
	def __init__(self, feature: Optional[str] = None, threshold: Optional[Any] = None,
				 decision: Optional[str] = None, confidence: float = 0.0):
		self.feature = feature
		self.threshold = threshold
		self.decision = decision
		self.confidence = confidence
		self.left_child: Optional['DecisionTreeNode'] = None
		self.right_child: Optional['DecisionTreeNode'] = None
		self.samples = 0
		self.impurity = 0.0
	
	def is_leaf(self) -> bool:
		"""Check if this is a leaf node"""
		return self.decision is not None
	
	def predict(self, context: Dict[str, Any]) -> Tuple[str, float]:
		"""Make prediction using this node"""
		if self.is_leaf():
			return self.decision, self.confidence
		
		# Navigate tree based on feature value
		feature_value = context.get(self.feature)
		if feature_value is None:
			# Missing feature - use majority class
			return self.decision or "deny", 0.5
		
		# Compare with threshold
		if isinstance(feature_value, (int, float)):
			go_left = feature_value <= self.threshold
		else:
			go_left = str(feature_value) <= str(self.threshold)
		
		if go_left and self.left_child:
			return self.left_child.predict(context)
		elif not go_left and self.right_child:
			return self.right_child.predict(context)
		else:
			# No child node - use current decision
			return self.decision or "deny", self.confidence

class AdaptivePolicyEngine:
	"""Main adaptive policy learning engine"""
	
	def __init__(self):
		# Decision tracking
		self._decision_records: Dict[str, PolicyDecisionRecord] = {}
		self._policy_metrics: Dict[str, PolicyMetrics] = {}
		self._learned_rules: Dict[str, List[AdaptivePolicyRule]] = {}
		
		# Learning configuration
		self._learning_modes: Dict[str, LearningMode] = {}
		self._feature_importance: Dict[str, List[FeatureImportance]] = {}
		self._decision_trees: Dict[str, DecisionTreeNode] = {}
		
		# Performance tracking
		self._learning_sessions: Dict[str, PolicyLearningSession] = {}
		self._recent_decisions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
		
		# A/B testing
		self._ab_tests: Dict[str, Dict[str, Any]] = {}
		
		# Configuration
		self.min_samples_for_learning = 100
		self.confidence_threshold = 0.7
		self.max_rules_per_policy = 50
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[AdaptivePolicy INFO] {message} {kwargs if kwargs else ''}")
	
	def _log_warning(self, message: str, **kwargs):
		"""Log warning message"""
		print(f"[AdaptivePolicy WARNING] {message} {kwargs if kwargs else ''}")
	
	def _log_error(self, message: str, **kwargs):
		"""Log error message"""
		print(f"[AdaptivePolicy ERROR] {message} {kwargs if kwargs else ''}")
	
	async def record_decision(self, policy_id: str, user_id: str, resource: str,
							  action: str, context: Dict[str, Any], decision: str,
							  confidence: float, reasoning: List[str]) -> str:
		"""Record a policy decision for learning"""
		decision_record = PolicyDecisionRecord(
			policy_id=policy_id,
			user_id=user_id,
			resource=resource,
			action=action,
			context=context,
			decision=decision,
			confidence=confidence,
			reasoning=reasoning
		)
		
		self._decision_records[decision_record.id] = decision_record
		self._recent_decisions[policy_id].append(decision_record.id)
		
		self._log_info("Decision recorded for learning",
					   policy_id=policy_id,
					   decision_id=decision_record.id,
					   decision=decision,
					   confidence=confidence)
		
		return decision_record.id
	
	async def record_outcome(self, decision_id: str, outcome: PolicyOutcome,
							 outcome_details: Dict[str, Any] = None,
							 feedback_source: str = "system") -> bool:
		"""Record the actual outcome of a policy decision"""
		decision_record = self._decision_records.get(decision_id)
		if not decision_record:
			self._log_warning("Decision record not found", decision_id=decision_id)
			return False
		
		# Update record with outcome
		decision_record.outcome = outcome
		decision_record.outcome_details = outcome_details or {}
		decision_record.outcome_recorded_at = datetime.utcnow()
		decision_record.feedback_source = feedback_source
		
		# Determine if decision was correct
		decision_record.was_correct = self._evaluate_decision_correctness(
			decision_record.decision, outcome
		)
		
		# Calculate impact score
		decision_record.impact_score = self._calculate_impact_score(outcome, outcome_details)
		
		self._log_info("Outcome recorded",
					   decision_id=decision_id,
					   outcome=outcome.value,
					   was_correct=decision_record.was_correct,
					   impact_score=decision_record.impact_score)
		
		# Trigger incremental learning if enough data
		await self._check_incremental_learning(decision_record.policy_id)
		
		return True
	
	def _evaluate_decision_correctness(self, decision: str, outcome: PolicyOutcome) -> bool:
		"""Evaluate if a policy decision was correct based on outcome"""
		if outcome == PolicyOutcome.SUCCESS:
			return True
		elif outcome == PolicyOutcome.FALSE_POSITIVE:
			return decision != "allow"  # Should have denied
		elif outcome == PolicyOutcome.FALSE_NEGATIVE:
			return decision != "deny"   # Should have allowed
		elif outcome in [PolicyOutcome.SECURITY_INCIDENT, PolicyOutcome.USER_COMPLAINT]:
			return False  # Decision led to negative outcome
		elif outcome == PolicyOutcome.ADMIN_OVERRIDE:
			return False  # Admin disagreed with decision
		else:
			return True   # Default to correct for unknown outcomes
	
	def _calculate_impact_score(self, outcome: PolicyOutcome, 
								outcome_details: Dict[str, Any]) -> float:
		"""Calculate impact score of decision outcome"""
		base_scores = {
			PolicyOutcome.SUCCESS: 0.0,
			PolicyOutcome.FALSE_POSITIVE: 0.3,
			PolicyOutcome.FALSE_NEGATIVE: 0.7,
			PolicyOutcome.SECURITY_INCIDENT: 0.9,
			PolicyOutcome.USER_COMPLAINT: 0.4,
			PolicyOutcome.ADMIN_OVERRIDE: 0.5
		}
		
		base_score = base_scores.get(outcome, 0.5)
		
		# Adjust based on details
		severity = outcome_details.get('severity', 'medium')
		if severity == 'critical':
			base_score = min(1.0, base_score * 1.5)
		elif severity == 'low':
			base_score = max(0.0, base_score * 0.7)
		
		# Consider affected users
		affected_users = outcome_details.get('affected_users', 1)
		if affected_users > 1:
			base_score = min(1.0, base_score * (1.0 + math.log10(affected_users) * 0.2))
		
		return base_score
	
	async def _check_incremental_learning(self, policy_id: str):
		"""Check if incremental learning should be triggered"""
		recent_decisions = list(self._recent_decisions[policy_id])
		
		# Check if we have enough recent decisions with outcomes
		decisions_with_outcomes = [
			self._decision_records[decision_id] for decision_id in recent_decisions[-100:]
			if self._decision_records[decision_id].outcome is not None
		]
		
		if len(decisions_with_outcomes) >= 20:  # Minimum for incremental learning
			# Check if recent performance is degrading
			recent_accuracy = sum(
				1 for d in decisions_with_outcomes[-20:]
				if d.was_correct
			) / min(20, len(decisions_with_outcomes))
			
			current_metrics = self._policy_metrics.get(policy_id)
			if current_metrics and recent_accuracy < current_metrics.accuracy * 0.9:
				self._log_info("Performance degradation detected, triggering incremental learning",
							   policy_id=policy_id,
							   recent_accuracy=recent_accuracy,
							   baseline_accuracy=current_metrics.accuracy)
				
				await self.learn_from_outcomes(policy_id, learning_mode=LearningMode.ACTIVE)
	
	async def calculate_policy_metrics(self, policy_id: str) -> PolicyMetrics:
		"""Calculate comprehensive metrics for a policy"""
		# Get all decisions for this policy
		policy_decisions = [
			record for record in self._decision_records.values()
			if record.policy_id == policy_id and record.outcome is not None
		]
		
		if not policy_decisions:
			return PolicyMetrics(policy_id=policy_id)
		
		# Basic counts
		total_decisions = len(policy_decisions)
		correct_decisions = sum(1 for d in policy_decisions if d.was_correct)
		false_positives = sum(1 for d in policy_decisions 
							 if d.outcome == PolicyOutcome.FALSE_POSITIVE)
		false_negatives = sum(1 for d in policy_decisions 
							 if d.outcome == PolicyOutcome.FALSE_NEGATIVE)
		
		# Calculate metrics
		accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0.0
		
		# Precision: True Positives / (True Positives + False Positives)
		allow_decisions = sum(1 for d in policy_decisions if d.decision == "allow")
		precision = ((allow_decisions - false_negatives) / allow_decisions 
					if allow_decisions > 0 else 0.0)
		
		# Recall: True Positives / (True Positives + False Negatives)
		actual_positives = allow_decisions + false_negatives
		recall = ((allow_decisions - false_negatives) / actual_positives 
				 if actual_positives > 0 else 0.0)
		
		# F1 Score
		f1_score = (2 * precision * recall / (precision + recall)
				   if precision + recall > 0 else 0.0)
		
		# User experience metrics
		user_complaints = sum(1 for d in policy_decisions 
							 if d.outcome == PolicyOutcome.USER_COMPLAINT)
		admin_overrides = sum(1 for d in policy_decisions 
							 if d.outcome == PolicyOutcome.ADMIN_OVERRIDE)
		
		# User satisfaction (inverse of complaints and false positives)
		negative_experiences = user_complaints + false_positives + admin_overrides
		user_satisfaction = max(0.0, 1.0 - (negative_experiences / total_decisions))
		
		# Performance trend
		if len(policy_decisions) >= 20:
			recent_accuracy = sum(1 for d in policy_decisions[-10:] if d.was_correct) / 10
			older_accuracy = sum(1 for d in policy_decisions[-20:-10] if d.was_correct) / 10
			
			if recent_accuracy > older_accuracy * 1.05:
				trend_direction = "improving"
			elif recent_accuracy < older_accuracy * 0.95:
				trend_direction = "degrading"
			else:
				trend_direction = "stable"
		else:
			trend_direction = "insufficient_data"
		
		metrics = PolicyMetrics(
			policy_id=policy_id,
			total_decisions=total_decisions,
			correct_decisions=correct_decisions,
			false_positives=false_positives,
			false_negatives=false_negatives,
			accuracy=accuracy,
			precision=precision,
			recall=recall,
			f1_score=f1_score,
			user_satisfaction=user_satisfaction,
			admin_overrides=admin_overrides,
			user_complaints=user_complaints,
			trend_direction=trend_direction
		)
		
		self._policy_metrics[policy_id] = metrics
		
		self._log_info("Policy metrics calculated",
					   policy_id=policy_id,
					   accuracy=accuracy,
					   total_decisions=total_decisions,
					   trend=trend_direction)
		
		return metrics
	
	async def learn_from_outcomes(self, policy_id: str, 
								  learning_mode: LearningMode = LearningMode.SUPERVISED,
								  optimization_target: str = "accuracy") -> PolicyLearningSession:
		"""Learn and adapt policy from recorded outcomes"""
		self._log_info("Starting policy learning session",
					   policy_id=policy_id,
					   learning_mode=learning_mode.value,
					   optimization_target=optimization_target)
		
		# Get training data
		policy_decisions = [
			record for record in self._decision_records.values()
			if record.policy_id == policy_id and record.outcome is not None
		]
		
		if len(policy_decisions) < self.min_samples_for_learning:
			raise ValueError(f"Insufficient data for learning: {len(policy_decisions)} samples "
							f"(minimum: {self.min_samples_for_learning})")
		
		# Calculate baseline metrics
		current_metrics = await self.calculate_policy_metrics(policy_id)
		baseline_accuracy = current_metrics.accuracy
		
		# Create learning session
		sample_dates = [d.decided_at for d in policy_decisions]
		learning_session = PolicyLearningSession(
			policy_id=policy_id,
			learning_mode=learning_mode,
			optimization_target=optimization_target,
			training_samples=len(policy_decisions),
			validation_samples=0,  # Will be set during cross-validation
			sample_date_range=(min(sample_dates), max(sample_dates)),
			accuracy_before=baseline_accuracy
		)
		
		# Split data for training and validation
		np.random.shuffle(policy_decisions)
		split_point = int(len(policy_decisions) * 0.8)
		training_data = policy_decisions[:split_point]
		validation_data = policy_decisions[split_point:]
		learning_session.validation_samples = len(validation_data)
		
		# Extract features and learn patterns
		feature_importance = await self._analyze_feature_importance(training_data)
		decision_tree = await self._learn_decision_tree(training_data, feature_importance)
		new_rules = await self._extract_rules_from_tree(decision_tree, policy_id)
		
		# Validate learned rules
		validation_accuracy = await self._validate_rules(new_rules, validation_data)
		
		# Apply rules based on learning mode
		rules_applied = 0
		if learning_mode in [LearningMode.ACTIVE, LearningMode.SUPERVISED]:
			if validation_accuracy > baseline_accuracy or learning_mode == LearningMode.SUPERVISED:
				await self._apply_learned_rules(policy_id, new_rules)
				rules_applied = len(new_rules)
		
		# Update session results
		learning_session.rules_generated = len(new_rules)
		learning_session.rules_modified = 0  # TODO: Track modified rules
		learning_session.accuracy_after = validation_accuracy
		learning_session.improvement = validation_accuracy - baseline_accuracy
		learning_session.completed_at = datetime.utcnow()
		learning_session.status = "completed"
		
		# Store session
		self._learning_sessions[learning_session.id] = learning_session
		
		# Store feature importance
		self._feature_importance[policy_id] = feature_importance
		self._decision_trees[policy_id] = decision_tree
		
		self._log_info("Policy learning session completed",
					   session_id=learning_session.id,
					   rules_generated=len(new_rules),
					   rules_applied=rules_applied,
					   accuracy_improvement=learning_session.improvement,
					   validation_accuracy=validation_accuracy)
		
		return learning_session
	
	async def _analyze_feature_importance(self, training_data: List[PolicyDecisionRecord]) -> List[FeatureImportance]:
		"""Analyze importance of different features in policy decisions"""
		feature_stats = defaultdict(lambda: {
			'correct_decisions': 0,
			'total_decisions': 0,
			'values': [],
			'outcomes': []
		})
		
		# Collect feature statistics
		for record in training_data:
			for feature_name, feature_value in record.context.items():
				if isinstance(feature_value, (str, int, float, bool)):
					stats = feature_stats[feature_name]
					stats['total_decisions'] += 1
					stats['values'].append(feature_value)
					stats['outcomes'].append(1 if record.was_correct else 0)
					
					if record.was_correct:
						stats['correct_decisions'] += 1
		
		# Calculate importance scores
		feature_importance = []
		for feature_name, stats in feature_stats.items():
			if stats['total_decisions'] >= 10:  # Minimum sample size
				# Calculate accuracy for this feature
				accuracy = stats['correct_decisions'] / stats['total_decisions']
				
				# Calculate correlation with outcomes
				if len(set(stats['values'])) > 1:  # Feature has variance
					try:
						# For numerical features, calculate correlation
						if all(isinstance(v, (int, float)) for v in stats['values'][:10]):
							correlation = abs(np.corrcoef(stats['values'], stats['outcomes'])[0, 1])
						else:
							# For categorical features, use chi-square-like measure
							correlation = self._calculate_categorical_correlation(
								stats['values'], stats['outcomes']
							)
					except Exception:
						correlation = 0.0
				else:
					correlation = 0.0
				
				# Importance score combines accuracy and correlation
				importance_score = (accuracy + correlation) / 2
				
				feature_importance.append(FeatureImportance(
					feature_name=feature_name,
					importance_score=importance_score,
					sample_count=stats['total_decisions'],
					correlation_with_outcome=correlation
				))
		
		# Sort by importance
		feature_importance.sort(key=lambda x: x.importance_score, reverse=True)
		
		return feature_importance[:20]  # Top 20 most important features
	
	def _calculate_categorical_correlation(self, values: List, outcomes: List[int]) -> float:
		"""Calculate correlation for categorical features"""
		value_outcome_map = defaultdict(list)
		for value, outcome in zip(values, outcomes):
			value_outcome_map[value].append(outcome)
		
		if len(value_outcome_map) <= 1:
			return 0.0
		
		# Calculate variance in success rates across categories
		success_rates = []
		for value, outcomes in value_outcome_map.items():
			if len(outcomes) >= 3:  # Minimum sample per category
				success_rate = sum(outcomes) / len(outcomes)
				success_rates.append(success_rate)
		
		if len(success_rates) <= 1:
			return 0.0
		
		# Return normalized variance (higher variance = higher correlation)
		variance = statistics.variance(success_rates)
		return min(1.0, variance * 4)  # Normalize to [0,1]
	
	async def _learn_decision_tree(self, training_data: List[PolicyDecisionRecord],
								   feature_importance: List[FeatureImportance]) -> DecisionTreeNode:
		"""Learn decision tree from training data"""
		# Simplified decision tree learning using most important features
		if not training_data or not feature_importance:
			return DecisionTreeNode(decision="deny", confidence=0.5)
		
		# Use top features for tree construction
		important_features = [fi.feature_name for fi in feature_importance[:5]]
		
		# Build tree recursively
		root = await self._build_tree_node(training_data, important_features, depth=0, max_depth=5)
		
		return root
	
	async def _build_tree_node(self, data: List[PolicyDecisionRecord],
							   available_features: List[str],
							   depth: int, max_depth: int) -> DecisionTreeNode:
		"""Build a single decision tree node"""
		if not data or depth >= max_depth:
			# Create leaf node with majority decision
			allow_count = sum(1 for d in data if d.decision == "allow")
			deny_count = len(data) - allow_count
			
			if allow_count > deny_count:
				decision = "allow"
				confidence = allow_count / len(data)
			else:
				decision = "deny"
				confidence = deny_count / len(data)
			
			node = DecisionTreeNode(decision=decision, confidence=confidence)
			node.samples = len(data)
			return node
		
		# Calculate current impurity (Gini index)
		allow_count = sum(1 for d in data if d.decision == "allow")
		total = len(data)
		if total == 0:
			current_impurity = 0.0
		else:
			p_allow = allow_count / total
			p_deny = 1 - p_allow
			current_impurity = 1 - (p_allow**2 + p_deny**2)
		
		# Find best split
		best_feature = None
		best_threshold = None
		best_gain = 0
		best_left_data = []
		best_right_data = []
		
		for feature in available_features:
			# Get all unique values for this feature
			feature_values = []
			for record in data:
				value = record.context.get(feature)
				if value is not None:
					feature_values.append(value)
			
			if len(set(feature_values)) <= 1:
				continue  # No variance in this feature
			
			# For numerical features, try different thresholds
			if all(isinstance(v, (int, float)) for v in feature_values):
				unique_values = sorted(set(feature_values))
				thresholds = [(unique_values[i] + unique_values[i+1]) / 2 
							 for i in range(len(unique_values) - 1)]
			else:
				# For categorical features, try each unique value as threshold
				thresholds = list(set(feature_values))
			
			for threshold in thresholds:
				# Split data
				left_data = []
				right_data = []
				
				for record in data:
					feature_value = record.context.get(feature)
					if feature_value is None:
						continue
					
					if isinstance(feature_value, (int, float)):
						if feature_value <= threshold:
							left_data.append(record)
						else:
							right_data.append(record)
					else:
						if str(feature_value) <= str(threshold):
							left_data.append(record)
						else:
							right_data.append(record)
				
				if len(left_data) == 0 or len(right_data) == 0:
					continue  # No split achieved
				
				# Calculate weighted impurity after split
				left_allow = sum(1 for d in left_data if d.decision == "allow")
				left_impurity = 1 - ((left_allow/len(left_data))**2 + 
									((len(left_data)-left_allow)/len(left_data))**2)
				
				right_allow = sum(1 for d in right_data if d.decision == "allow")
				right_impurity = 1 - ((right_allow/len(right_data))**2 + 
									 ((len(right_data)-right_allow)/len(right_data))**2)
				
				weighted_impurity = (len(left_data)/total * left_impurity + 
									len(right_data)/total * right_impurity)
				
				# Calculate information gain
				gain = current_impurity - weighted_impurity
				
				if gain > best_gain:
					best_gain = gain
					best_feature = feature
					best_threshold = threshold
					best_left_data = left_data
					best_right_data = right_data
		
		# If no good split found, create leaf
		if best_gain <= 0.01:  # Minimum gain threshold
			allow_count = sum(1 for d in data if d.decision == "allow")
			if allow_count > len(data) - allow_count:
				decision = "allow"
				confidence = allow_count / len(data)
			else:
				decision = "deny"
				confidence = (len(data) - allow_count) / len(data)
			
			node = DecisionTreeNode(decision=decision, confidence=confidence)
			node.samples = len(data)
			return node
		
		# Create internal node
		node = DecisionTreeNode(feature=best_feature, threshold=best_threshold)
		node.samples = len(data)
		node.impurity = current_impurity
		
		# Recursively build children
		remaining_features = [f for f in available_features if f != best_feature]
		node.left_child = await self._build_tree_node(best_left_data, remaining_features, depth+1, max_depth)
		node.right_child = await self._build_tree_node(best_right_data, remaining_features, depth+1, max_depth)
		
		return node
	
	async def _extract_rules_from_tree(self, tree: DecisionTreeNode, policy_id: str) -> List[AdaptivePolicyRule]:
		"""Extract rules from decision tree"""
		rules = []
		
		def extract_path_rules(node: DecisionTreeNode, path_conditions: List[Dict[str, Any]]):
			if node.is_leaf():
				# Create rule from path conditions
				if node.confidence >= self.confidence_threshold:
					rule = AdaptivePolicyRule(
						policy_id=policy_id,
						conditions=path_conditions.copy(),
						action=node.decision,
						learned_from_samples=node.samples,
						confidence=node.confidence,
						support=node.samples / tree.samples if tree.samples > 0 else 0.0,
						priority=100 - int(node.confidence * 50)  # Higher confidence = higher priority
					)
					rules.append(rule)
				return
			
			# Traverse left child (feature <= threshold)
			if node.left_child:
				left_conditions = path_conditions + [{
					"feature": node.feature,
					"operator": "<=",
					"value": node.threshold
				}]
				extract_path_rules(node.left_child, left_conditions)
			
			# Traverse right child (feature > threshold)
			if node.right_child:
				right_conditions = path_conditions + [{
					"feature": node.feature,
					"operator": ">",
					"value": node.threshold
				}]
				extract_path_rules(node.right_child, right_conditions)
		
		extract_path_rules(tree, [])
		
		# Sort rules by confidence and support
		rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
		
		# Limit number of rules
		return rules[:self.max_rules_per_policy]
	
	async def _validate_rules(self, rules: List[AdaptivePolicyRule],
							  validation_data: List[PolicyDecisionRecord]) -> float:
		"""Validate learned rules against validation data"""
		if not rules or not validation_data:
			return 0.0
		
		correct_predictions = 0
		total_predictions = 0
		
		for record in validation_data:
			# Apply rules to predict decision
			predicted_decision = await self._apply_rules_to_context(rules, record.context)
			actual_decision = record.decision
			
			if predicted_decision == actual_decision:
				correct_predictions += 1
			total_predictions += 1
		
		accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
		return accuracy
	
	async def _apply_rules_to_context(self, rules: List[AdaptivePolicyRule],
									  context: Dict[str, Any]) -> str:
		"""Apply rules to context and return predicted decision"""
		# Apply rules in priority order
		for rule in sorted(rules, key=lambda r: r.priority):
			if self._rule_matches_context(rule, context):
				return rule.action
		
		# Default decision if no rules match
		return "deny"
	
	def _rule_matches_context(self, rule: AdaptivePolicyRule, context: Dict[str, Any]) -> bool:
		"""Check if rule conditions match the given context"""
		for condition in rule.conditions:
			feature = condition.get("feature")
			operator = condition.get("operator")
			expected_value = condition.get("value")
			
			actual_value = context.get(feature)
			
			if actual_value is None:
				return False  # Missing feature
			
			# Apply condition
			if operator == "<=":
				if not (actual_value <= expected_value):
					return False
			elif operator == ">":
				if not (actual_value > expected_value):
					return False
			elif operator == "==":
				if actual_value != expected_value:
					return False
			elif operator == "!=":
				if actual_value == expected_value:
					return False
			elif operator == "in":
				if actual_value not in expected_value:
					return False
			elif operator == "not_in":
				if actual_value in expected_value:
					return False
		
		return True  # All conditions match
	
	async def _apply_learned_rules(self, policy_id: str, rules: List[AdaptivePolicyRule]):
		"""Apply learned rules to policy"""
		if policy_id not in self._learned_rules:
			self._learned_rules[policy_id] = []
		
		# Add new rules
		self._learned_rules[policy_id].extend(rules)
		
		# Sort all rules by priority
		self._learned_rules[policy_id].sort(key=lambda r: r.priority)
		
		# Limit total rules
		if len(self._learned_rules[policy_id]) > self.max_rules_per_policy:
			self._learned_rules[policy_id] = self._learned_rules[policy_id][:self.max_rules_per_policy]
		
		self._log_info("Learned rules applied to policy",
					   policy_id=policy_id,
					   new_rules=len(rules),
					   total_rules=len(self._learned_rules[policy_id]))
	
	async def predict_decision(self, policy_id: str, context: Dict[str, Any]) -> Tuple[str, float, List[str]]:
		"""Use learned rules to predict access decision"""
		learned_rules = self._learned_rules.get(policy_id, [])
		if not learned_rules:
			return "deny", 0.5, ["No learned rules available"]
		
		# Apply rules
		for rule in learned_rules:
			if self._rule_matches_context(rule, context):
				reasoning = [
					f"Rule {rule.id[:8]} matched",
					f"Conditions: {len(rule.conditions)} matched",
					f"Confidence: {rule.confidence:.2f}",
					f"Historical effectiveness: {rule.effectiveness:.2f}"
				]
				return rule.action, rule.confidence, reasoning
		
		# Use decision tree if available
		decision_tree = self._decision_trees.get(policy_id)
		if decision_tree:
			decision, confidence = decision_tree.predict(context)
			reasoning = ["Decision tree prediction", f"Tree confidence: {confidence:.2f}"]
			return decision, confidence, reasoning
		
		# Default fallback
		return "deny", 0.3, ["No matching rules, default deny"]
	
	async def start_ab_test(self, policy_id: str, test_rules: List[AdaptivePolicyRule],
							test_percentage: float = 0.1) -> str:
		"""Start A/B test of new rules against existing policy"""
		test_id = uuid7str()
		
		self._ab_tests[test_id] = {
			"policy_id": policy_id,
			"test_rules": test_rules,
			"control_rules": self._learned_rules.get(policy_id, []).copy(),
			"test_percentage": test_percentage,
			"started_at": datetime.utcnow(),
			"test_decisions": [],
			"control_decisions": []
		}
		
		self._log_info("A/B test started",
					   test_id=test_id,
					   policy_id=policy_id,
					   test_percentage=test_percentage,
					   test_rules=len(test_rules))
		
		return test_id
	
	async def evaluate_ab_test(self, test_id: str) -> Dict[str, Any]:
		"""Evaluate A/B test results"""
		ab_test = self._ab_tests.get(test_id)
		if not ab_test:
			raise ValueError("A/B test not found")
		
		test_decisions = ab_test["test_decisions"]
		control_decisions = ab_test["control_decisions"]
		
		if len(test_decisions) < 20 or len(control_decisions) < 20:
			return {
				"status": "insufficient_data",
				"test_samples": len(test_decisions),
				"control_samples": len(control_decisions)
			}
		
		# Calculate metrics for both groups
		test_accuracy = sum(1 for d in test_decisions if d["was_correct"]) / len(test_decisions)
		control_accuracy = sum(1 for d in control_decisions if d["was_correct"]) / len(control_decisions)
		
		test_satisfaction = 1.0 - sum(1 for d in test_decisions if d["outcome"] in ["user_complaint", "false_positive"]) / len(test_decisions)
		control_satisfaction = 1.0 - sum(1 for d in control_decisions if d["outcome"] in ["user_complaint", "false_positive"]) / len(control_decisions)
		
		# Statistical significance (simplified)
		accuracy_improvement = test_accuracy - control_accuracy
		satisfaction_improvement = test_satisfaction - control_satisfaction
		
		# Confidence based on sample size and difference
		confidence = min(0.95, max(0.5, abs(accuracy_improvement) * math.sqrt(len(test_decisions))))
		
		results = {
			"status": "complete",
			"test_accuracy": test_accuracy,
			"control_accuracy": control_accuracy,
			"accuracy_improvement": accuracy_improvement,
			"test_satisfaction": test_satisfaction,
			"control_satisfaction": control_satisfaction,
			"satisfaction_improvement": satisfaction_improvement,
			"confidence": confidence,
			"recommendation": "deploy" if accuracy_improvement > 0.02 and confidence > 0.8 else "reject"
		}
		
		self._log_info("A/B test evaluation complete",
					   test_id=test_id,
					   accuracy_improvement=accuracy_improvement,
					   recommendation=results["recommendation"])
		
		return results
	
	def get_policy_performance_summary(self, policy_id: str) -> Dict[str, Any]:
		"""Get comprehensive performance summary for policy"""
		metrics = self._policy_metrics.get(policy_id, PolicyMetrics(policy_id=policy_id))
		feature_importance = self._feature_importance.get(policy_id, [])
		learned_rules = self._learned_rules.get(policy_id, [])
		learning_sessions = [
			session for session in self._learning_sessions.values()
			if session.policy_id == policy_id
		]
		
		return {
			"policy_id": policy_id,
			"current_metrics": metrics.model_dump(),
			"top_features": [
				{
					"name": fi.feature_name,
					"importance": fi.importance_score,
					"samples": fi.sample_count
				}
				for fi in feature_importance[:5]
			],
			"learned_rules_count": len(learned_rules),
			"learning_sessions_count": len(learning_sessions),
			"last_learning_session": learning_sessions[-1].model_dump() if learning_sessions else None,
			"recommendations": self._generate_policy_recommendations(policy_id, metrics)
		}
	
	def _generate_policy_recommendations(self, policy_id: str, metrics: PolicyMetrics) -> List[str]:
		"""Generate recommendations for policy improvement"""
		recommendations = []
		
		if metrics.accuracy < 0.8:
			recommendations.append("Consider collecting more training data for better accuracy")
		
		if metrics.false_positives > metrics.total_decisions * 0.1:
			recommendations.append("High false positive rate - review restrictive rules")
		
		if metrics.false_negatives > metrics.total_decisions * 0.05:
			recommendations.append("High false negative rate - review permissive rules")
		
		if metrics.user_satisfaction < 0.7:
			recommendations.append("Low user satisfaction - consider more flexible policies")
		
		if metrics.trend_direction == "degrading":
			recommendations.append("Performance is degrading - immediate retraining recommended")
		
		if not self._learned_rules.get(policy_id):
			recommendations.append("No learned rules - start with supervised learning")
		
		return recommendations
	
	def set_learning_mode(self, policy_id: str, mode: LearningMode):
		"""Set learning mode for a policy"""
		self._learning_modes[policy_id] = mode
		self._log_info("Learning mode set", policy_id=policy_id, mode=mode.value)
	
	def clear_policy_data(self, policy_id: str):
		"""Clear all data for a policy (cleanup/reset)"""
		# Clear decision records
		decision_ids_to_remove = [
			record_id for record_id, record in self._decision_records.items()
			if record.policy_id == policy_id
		]
		for record_id in decision_ids_to_remove:
			del self._decision_records[record_id]
		
		# Clear other data structures
		if policy_id in self._policy_metrics:
			del self._policy_metrics[policy_id]
		if policy_id in self._learned_rules:
			del self._learned_rules[policy_id]
		if policy_id in self._feature_importance:
			del self._feature_importance[policy_id]
		if policy_id in self._decision_trees:
			del self._decision_trees[policy_id]
		if policy_id in self._recent_decisions:
			del self._recent_decisions[policy_id]
		
		self._log_info("Policy data cleared", policy_id=policy_id, records_removed=len(decision_ids_to_remove))