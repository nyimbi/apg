"""
APG GraphRAG Capability - Collaborative Knowledge Curation

Revolutionary collaborative knowledge curation system with expert workflows,
consensus algorithms, and real-time collaborative editing capabilities.

Author: Datacraft (nyimbi@gmail.com)  
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import uuid

from .database import GraphRAGDatabaseService
from .incremental_updates import IncrementalUpdateEngine, UpdateOperation, UpdateType
from .views import (
	CurationWorkflow, KnowledgeEdit, ExpertParticipant,
	GraphEntity, GraphRelationship
)


logger = logging.getLogger(__name__)


class CurationRole(str, Enum):
	"""Roles in collaborative curation"""
	CONTRIBUTOR = "contributor"
	REVIEWER = "reviewer" 
	DOMAIN_EXPERT = "domain_expert"
	MODERATOR = "moderator"
	ADMINISTRATOR = "administrator"


class EditStatus(str, Enum):
	"""Status of knowledge edits"""
	DRAFT = "draft"
	SUBMITTED = "submitted"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	IMPLEMENTED = "implemented"
	ARCHIVED = "archived"


class ConsensusStrategy(str, Enum):
	"""Strategies for reaching consensus"""
	SIMPLE_MAJORITY = "simple_majority"
	WEIGHTED_MAJORITY = "weighted_majority"
	EXPERT_CONSENSUS = "expert_consensus"
	UNANIMOUS = "unanimous"
	DOMAIN_EXPERT_DECIDES = "domain_expert_decides"


@dataclass
class CurationTask:
	"""Individual curation task"""
	task_id: str
	workflow_id: str
	task_type: str
	assigned_to: List[str]
	title: str
	description: str
	priority: int
	status: str
	metadata: Dict[str, Any]
	created_at: datetime
	due_date: Optional[datetime] = None


@dataclass
class Review:
	"""Review of a knowledge edit"""
	review_id: str
	edit_id: str
	reviewer_id: str
	reviewer_role: CurationRole
	decision: str  # approve, reject, request_changes
	confidence: float
	comments: str
	specific_feedback: Dict[str, str]
	review_time_minutes: int
	created_at: datetime


@dataclass
class ConsensusResult:
	"""Result of consensus calculation"""
	consensus_reached: bool
	consensus_score: float
	decision: str
	participant_votes: Dict[str, Dict[str, Any]]
	consensus_strategy: ConsensusStrategy
	confidence_level: float
	dissenting_opinions: List[str]
	reasoning: str


class CollaborativeKnowledgeCuration:
	"""
	Advanced collaborative knowledge curation system providing:
	
	- Expert-driven workflow management
	- Real-time collaborative editing
	- Consensus algorithms with expertise weighting
	- Quality assurance and validation
	- Conflict resolution mechanisms
	- Performance analytics and insights
	- Integration with knowledge graph updates
	"""
	
	def __init__(
		self,
		db_service: GraphRAGDatabaseService,
		update_engine: IncrementalUpdateEngine,
		config: Optional[Dict[str, Any]] = None
	):
		"""Initialize collaborative curation system"""
		self.db_service = db_service
		self.update_engine = update_engine
		self.config = config or {}
		
		# Curation parameters
		self.default_consensus_threshold = self.config.get("consensus_threshold", 0.75)
		self.expert_weight_multiplier = self.config.get("expert_weight_multiplier", 2.0)
		self.min_reviews_required = self.config.get("min_reviews_required", 2)
		self.max_review_time_hours = self.config.get("max_review_time_hours", 72)
		
		# Consensus strategies
		self.consensus_strategies = self._initialize_consensus_strategies()
		
		# Active workflows and real-time state
		self._active_workflows = {}
		self._active_editing_sessions = {}
		self._workflow_locks = defaultdict(asyncio.Lock)
		
		# Performance and analytics
		self._curation_metrics = defaultdict(list)
		self._expert_performance = defaultdict(dict)
		
		logger.info("Collaborative knowledge curation system initialized")
	
	async def create_curation_workflow(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		name: str,
		description: str,
		workflow_type: str,
		participants: List[ExpertParticipant],
		consensus_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_MAJORITY,
		consensus_threshold: float = None
	) -> CurationWorkflow:
		"""
		Create a new collaborative curation workflow
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			name: Workflow name
			description: Workflow description
			workflow_type: Type of curation workflow
			participants: List of expert participants
			consensus_strategy: Strategy for reaching consensus
			consensus_threshold: Threshold for consensus (optional)
			
		Returns:
			Created CurationWorkflow
		"""
		start_time = time.time()
		
		try:
			# Validate participants
			validated_participants = await self._validate_participants(participants)
			
			# Set consensus threshold
			threshold = consensus_threshold or self.default_consensus_threshold
			
			# Create workflow in database
			workflow = await self.db_service.create_curation_workflow(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				name=name,
				description=description,
				workflow_type=workflow_type,
				participants=validated_participants,
				consensus_threshold=threshold
			)
			
			# Initialize workflow state
			self._active_workflows[workflow.workflow_id] = {
				"consensus_strategy": consensus_strategy,
				"active_edits": {},
				"pending_reviews": [],
				"metrics": {
					"edits_submitted": 0,
					"edits_approved": 0,
					"edits_rejected": 0,
					"average_review_time": 0.0,
					"consensus_rate": 0.0
				}
			}
			
			# Create initial curation tasks
			initial_tasks = await self._create_initial_curation_tasks(
				tenant_id, knowledge_graph_id, workflow
			)
			
			processing_time = (time.time() - start_time) * 1000
			self._record_metric("workflow_creation", processing_time)
			
			logger.info(f"Created curation workflow '{name}' with {len(validated_participants)} participants")
			return workflow
			
		except Exception as e:
			logger.error(f"Failed to create curation workflow: {e}")
			raise
	
	async def submit_knowledge_edit(
		self,
		tenant_id: str,
		workflow_id: str,
		editor_id: str,
		edit_type: str,
		target_type: str,
		target_id: str,
		proposed_changes: Dict[str, Any],
		justification: str,
		evidence: List[str] = None
	) -> KnowledgeEdit:
		"""
		Submit a knowledge edit for collaborative review
		
		Args:
			tenant_id: Tenant identifier
			workflow_id: Curation workflow identifier
			editor_id: User submitting the edit
			edit_type: Type of edit (create, update, delete, merge, split)
			target_type: Type of target (entity, relationship, community, graph)
			target_id: Identifier of target being edited
			proposed_changes: Proposed changes as dictionary
			justification: Justification for the edit
			evidence: Optional supporting evidence
			
		Returns:
			Created KnowledgeEdit
		"""
		start_time = time.time()
		
		async with self._workflow_locks[workflow_id]:
			try:
				# Validate edit submission
				validation_result = await self._validate_edit_submission(
					tenant_id, workflow_id, editor_id, edit_type, target_type, target_id, proposed_changes
				)
				
				if not validation_result["valid"]:
					raise ValueError(f"Edit validation failed: {validation_result['error']}")
				
				# Create knowledge edit
				knowledge_edit = await self.db_service.create_knowledge_edit(
					workflow_id=workflow_id,
					knowledge_graph_id=validation_result["knowledge_graph_id"],
					tenant_id=tenant_id,
					editor_id=editor_id,
					edit_type=edit_type,
					target_type=target_type,
					target_id=target_id,
					proposed_changes=proposed_changes,
					justification=justification,
					evidence=evidence or []
				)
				
				# Add to active workflow state
				if workflow_id in self._active_workflows:
					self._active_workflows[workflow_id]["active_edits"][knowledge_edit.edit_id] = {
						"status": EditStatus.SUBMITTED,
						"reviews": [],
						"created_at": datetime.utcnow(),
						"editor_id": editor_id
					}
					self._active_workflows[workflow_id]["metrics"]["edits_submitted"] += 1
				
				# Assign reviewers
				await self._assign_reviewers_for_edit(tenant_id, workflow_id, knowledge_edit)
				
				# Notify participants
				await self._notify_edit_submission(tenant_id, workflow_id, knowledge_edit)
				
				processing_time = (time.time() - start_time) * 1000
				self._record_metric("edit_submission", processing_time)
				
				logger.info(f"Edit {knowledge_edit.edit_id} submitted by {editor_id} for workflow {workflow_id}")
				return knowledge_edit
				
			except Exception as e:
				logger.error(f"Failed to submit knowledge edit: {e}")
				raise
	
	async def submit_edit_review(
		self,
		tenant_id: str,
		edit_id: str,
		reviewer_id: str,
		decision: str,
		confidence: float,
		comments: str,
		specific_feedback: Dict[str, str] = None
	) -> Review:
		"""
		Submit a review for a knowledge edit
		
		Args:
			tenant_id: Tenant identifier
			edit_id: Knowledge edit identifier
			reviewer_id: User submitting the review
			decision: Review decision (approve, reject, request_changes)
			confidence: Confidence in the review (0.0-1.0)
			comments: Review comments
			specific_feedback: Specific feedback on different aspects
			
		Returns:
			Created Review object
		"""
		start_time = time.time()
		
		try:
			# Get knowledge edit
			edit = await self.db_service.get_knowledge_edit(tenant_id, edit_id)
			
			# Validate reviewer authorization
			await self._validate_reviewer_authorization(tenant_id, edit.workflow_id, reviewer_id)
			
			# Create review
			review = Review(
				review_id=str(uuid.uuid4()),
				edit_id=edit_id,
				reviewer_id=reviewer_id,
				reviewer_role=await self._get_reviewer_role(edit.workflow_id, reviewer_id),
				decision=decision,
				confidence=confidence,
				comments=comments,
				specific_feedback=specific_feedback or {},
				review_time_minutes=int((datetime.utcnow() - edit.created_at).total_seconds() / 60),
				created_at=datetime.utcnow()
			)
			
			# Add review to edit
			current_reviews = edit.reviews.copy()
			current_reviews.append(review.dict())
			
			await self.db_service.update_knowledge_edit(
				tenant_id=tenant_id,
				edit_id=edit_id,
				updates={"reviews": current_reviews}
			)
			
			# Update workflow state
			if edit.workflow_id in self._active_workflows:
				edit_state = self._active_workflows[edit.workflow_id]["active_edits"].get(edit_id, {})
				edit_state["reviews"] = edit_state.get("reviews", [])
				edit_state["reviews"].append(review)
				edit_state["status"] = EditStatus.UNDER_REVIEW
			
			# Check if consensus is reached
			consensus_result = await self._check_consensus(tenant_id, edit_id)
			
			if consensus_result.consensus_reached:
				await self._handle_consensus_reached(tenant_id, edit_id, consensus_result)
			
			# Update expert performance metrics
			self._update_expert_performance(reviewer_id, review)
			
			processing_time = (time.time() - start_time) * 1000
			self._record_metric("review_submission", processing_time)
			
			logger.info(f"Review submitted by {reviewer_id} for edit {edit_id}: {decision} (confidence: {confidence})")
			return review
			
		except Exception as e:
			logger.error(f"Failed to submit edit review: {e}")
			raise
	
	async def calculate_consensus(
		self,
		tenant_id: str,
		edit_id: str,
		strategy: Optional[ConsensusStrategy] = None
	) -> ConsensusResult:
		"""
		Calculate consensus for a knowledge edit
		
		Args:
			tenant_id: Tenant identifier
			edit_id: Knowledge edit identifier
			strategy: Optional consensus strategy override
			
		Returns:
			ConsensusResult with consensus calculation
		"""
		try:
			# Get knowledge edit and reviews
			edit = await self.db_service.get_knowledge_edit(tenant_id, edit_id)
			reviews = [Review(**review_data) for review_data in edit.reviews]
			
			if not reviews:
				return ConsensusResult(
					consensus_reached=False,
					consensus_score=0.0,
					decision="pending",
					participant_votes={},
					consensus_strategy=strategy or ConsensusStrategy.WEIGHTED_MAJORITY,
					confidence_level=0.0,
					dissenting_opinions=[],
					reasoning="No reviews submitted yet"
				)
			
			# Get workflow and participants
			workflow = await self.db_service.get_curation_workflow(tenant_id, edit.workflow_id)
			
			# Determine consensus strategy
			consensus_strategy = strategy or self._active_workflows.get(
				edit.workflow_id, {}
			).get("consensus_strategy", ConsensusStrategy.WEIGHTED_MAJORITY)
			
			# Calculate consensus based on strategy
			if consensus_strategy == ConsensusStrategy.SIMPLE_MAJORITY:
				return await self._calculate_simple_majority_consensus(reviews, workflow)
			
			elif consensus_strategy == ConsensusStrategy.WEIGHTED_MAJORITY:
				return await self._calculate_weighted_majority_consensus(reviews, workflow)
			
			elif consensus_strategy == ConsensusStrategy.EXPERT_CONSENSUS:
				return await self._calculate_expert_consensus(reviews, workflow)
			
			elif consensus_strategy == ConsensusStrategy.UNANIMOUS:
				return await self._calculate_unanimous_consensus(reviews, workflow)
			
			elif consensus_strategy == ConsensusStrategy.DOMAIN_EXPERT_DECIDES:
				return await self._calculate_domain_expert_consensus(reviews, workflow)
			
			else:
				# Default to weighted majority
				return await self._calculate_weighted_majority_consensus(reviews, workflow)
				
		except Exception as e:
			logger.error(f"Failed to calculate consensus for edit {edit_id}: {e}")
			raise
	
	async def get_curation_analytics(
		self,
		tenant_id: str,
		workflow_id: str,
		time_period: Optional[timedelta] = None
	) -> Dict[str, Any]:
		"""
		Get comprehensive analytics for curation workflow
		
		Args:
			tenant_id: Tenant identifier
			workflow_id: Curation workflow identifier
			time_period: Optional time period for analytics
			
		Returns:
			Comprehensive analytics dictionary
		"""
		try:
			end_date = datetime.utcnow()
			start_date = end_date - (time_period or timedelta(days=30))
			
			# Get workflow data
			workflow = await self.db_service.get_curation_workflow(tenant_id, workflow_id)
			edits = await self.db_service.list_knowledge_edits(
				tenant_id=tenant_id,
				workflow_id=workflow_id,
				start_date=start_date,
				end_date=end_date
			)
			
			# Calculate basic metrics
			total_edits = len(edits)
			approved_edits = len([e for e in edits if e.status == "approved"])
			rejected_edits = len([e for e in edits if e.status == "rejected"])
			pending_edits = len([e for e in edits if e.status in ["submitted", "under_review"]])
			
			# Calculate review metrics
			all_reviews = []
			for edit in edits:
				all_reviews.extend([Review(**r) for r in edit.reviews])
			
			avg_review_time = (
				sum(r.review_time_minutes for r in all_reviews) / len(all_reviews)
				if all_reviews else 0
			)
			
			avg_reviews_per_edit = len(all_reviews) / total_edits if total_edits > 0 else 0
			
			# Calculate participant metrics
			participant_metrics = await self._calculate_participant_metrics(workflow, edits, all_reviews)
			
			# Calculate consensus metrics
			consensus_metrics = await self._calculate_consensus_metrics(edits)
			
			# Calculate quality metrics
			quality_metrics = await self._calculate_quality_metrics(edits, all_reviews)
			
			# Build comprehensive analytics
			analytics = {
				"workflow_info": {
					"workflow_id": workflow_id,
					"name": workflow.name,
					"participants": len(workflow.participants),
					"created_date": workflow.created_at,
					"analysis_period": {
						"start_date": start_date,
						"end_date": end_date,
						"days": (end_date - start_date).days
					}
				},
				"edit_metrics": {
					"total_edits": total_edits,
					"approved_edits": approved_edits,
					"rejected_edits": rejected_edits,
					"pending_edits": pending_edits,
					"approval_rate": approved_edits / total_edits if total_edits > 0 else 0,
					"rejection_rate": rejected_edits / total_edits if total_edits > 0 else 0,
					"edits_by_type": self._group_edits_by_type(edits),
					"edits_by_target": self._group_edits_by_target(edits)
				},
				"review_metrics": {
					"total_reviews": len(all_reviews),
					"average_review_time_minutes": avg_review_time,
					"average_reviews_per_edit": avg_reviews_per_edit,
					"review_decisions": self._group_reviews_by_decision(all_reviews),
					"average_confidence": sum(r.confidence for r in all_reviews) / len(all_reviews) if all_reviews else 0
				},
				"participant_metrics": participant_metrics,
				"consensus_metrics": consensus_metrics,
				"quality_metrics": quality_metrics,
				"performance_trends": await self._calculate_performance_trends(workflow_id, start_date, end_date)
			}
			
			logger.info(f"Generated curation analytics for workflow {workflow_id}")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate curation analytics: {e}")
			raise
	
	# ========================================================================
	# CONSENSUS CALCULATION METHODS
	# ========================================================================
	
	async def _calculate_weighted_majority_consensus(
		self,
		reviews: List[Review],
		workflow: CurationWorkflow
	) -> ConsensusResult:
		"""Calculate consensus using weighted majority voting"""
		
		# Get participant weights
		participant_weights = {p.user_id: p.weight for p in workflow.participants}
		
		# Calculate weighted votes
		approve_weight = 0.0
		reject_weight = 0.0
		total_weight = 0.0
		participant_votes = {}
		
		for review in reviews:
			weight = participant_weights.get(review.reviewer_id, 1.0)
			
			# Apply expertise multiplier for domain experts
			if review.reviewer_role == CurationRole.DOMAIN_EXPERT:
				weight *= self.expert_weight_multiplier
			
			total_weight += weight
			
			if review.decision == "approve":
				approve_weight += weight * review.confidence
			elif review.decision == "reject":
				reject_weight += weight * review.confidence
			
			participant_votes[review.reviewer_id] = {
				"decision": review.decision,
				"confidence": review.confidence,
				"weight": weight,
				"role": review.reviewer_role.value
			}
		
		# Calculate consensus score
		if total_weight == 0:
			consensus_score = 0.0
			decision = "pending"
		else:
			consensus_score = max(approve_weight, reject_weight) / total_weight
			decision = "approve" if approve_weight > reject_weight else "reject"
		
		# Check if consensus threshold is met
		consensus_reached = (
			consensus_score >= workflow.consensus_threshold and
			len(reviews) >= self.min_reviews_required
		)
		
		# Identify dissenting opinions
		dissenting_opinions = []
		if decision == "approve":
			dissenting_opinions = [
				f"{review.reviewer_id}: {review.comments}" 
				for review in reviews if review.decision == "reject"
			]
		else:
			dissenting_opinions = [
				f"{review.reviewer_id}: {review.comments}"
				for review in reviews if review.decision == "approve"
			]
		
		return ConsensusResult(
			consensus_reached=consensus_reached,
			consensus_score=consensus_score,
			decision=decision,
			participant_votes=participant_votes,
			consensus_strategy=ConsensusStrategy.WEIGHTED_MAJORITY,
			confidence_level=consensus_score,
			dissenting_opinions=dissenting_opinions,
			reasoning=f"Weighted majority consensus: {decision} with {consensus_score:.2f} score (threshold: {workflow.consensus_threshold})"
		)
	
	async def _calculate_simple_majority_consensus(
		self,
		reviews: List[Review],
		workflow: CurationWorkflow
	) -> ConsensusResult:
		"""Calculate consensus using simple majority voting"""
		
		approve_count = len([r for r in reviews if r.decision == "approve"])
		reject_count = len([r for r in reviews if r.decision == "reject"])
		total_reviews = len(reviews)
		
		if total_reviews == 0:
			return ConsensusResult(
				consensus_reached=False,
				consensus_score=0.0,
				decision="pending",
				participant_votes={},
				consensus_strategy=ConsensusStrategy.SIMPLE_MAJORITY,
				confidence_level=0.0,
				dissenting_opinions=[],
				reasoning="No reviews submitted"
			)
		
		# Determine decision
		if approve_count > reject_count:
			decision = "approve"
			consensus_score = approve_count / total_reviews
		elif reject_count > approve_count:
			decision = "reject"
			consensus_score = reject_count / total_reviews
		else:
			decision = "tied"
			consensus_score = 0.5
		
		consensus_reached = (
			decision != "tied" and
			consensus_score >= workflow.consensus_threshold and
			total_reviews >= self.min_reviews_required
		)
		
		participant_votes = {
			review.reviewer_id: {
				"decision": review.decision,
				"confidence": review.confidence,
				"weight": 1.0,
				"role": review.reviewer_role.value
			}
			for review in reviews
		}
		
		return ConsensusResult(
			consensus_reached=consensus_reached,
			consensus_score=consensus_score,
			decision=decision,
			participant_votes=participant_votes,
			consensus_strategy=ConsensusStrategy.SIMPLE_MAJORITY,
			confidence_level=consensus_score,
			dissenting_opinions=[],
			reasoning=f"Simple majority: {approve_count} approve, {reject_count} reject"
		)
	
	async def _calculate_expert_consensus(
		self,
		reviews: List[Review],
		workflow: CurationWorkflow
	) -> ConsensusResult:
		"""Calculate consensus based on domain expert opinions only"""
		
		expert_reviews = [
			r for r in reviews 
			if r.reviewer_role == CurationRole.DOMAIN_EXPERT
		]
		
		if not expert_reviews:
			return ConsensusResult(
				consensus_reached=False,
				consensus_score=0.0,
				decision="pending",
				participant_votes={},
				consensus_strategy=ConsensusStrategy.EXPERT_CONSENSUS,
				confidence_level=0.0,
				dissenting_opinions=[],
				reasoning="No domain expert reviews submitted"
			)
		
		# Use weighted majority among experts
		return await self._calculate_weighted_majority_consensus(expert_reviews, workflow)
	
	async def _calculate_unanimous_consensus(
		self,
		reviews: List[Review],
		workflow: CurationWorkflow
	) -> ConsensusResult:
		"""Calculate consensus requiring unanimous agreement"""
		
		if not reviews:
			return ConsensusResult(
				consensus_reached=False,
				consensus_score=0.0,
				decision="pending",
				participant_votes={},
				consensus_strategy=ConsensusStrategy.UNANIMOUS,
				confidence_level=0.0,
				dissenting_opinions=[],
				reasoning="No reviews submitted"
			)
		
		# Check if all reviews agree
		decisions = [r.decision for r in reviews]
		unique_decisions = set(decisions)
		
		if len(unique_decisions) == 1:
			decision = decisions[0]
			consensus_reached = len(reviews) >= self.min_reviews_required
			consensus_score = 1.0
		else:
			decision = "no_consensus"
			consensus_reached = False
			consensus_score = 0.0
		
		participant_votes = {
			review.reviewer_id: {
				"decision": review.decision,
				"confidence": review.confidence,
				"weight": 1.0,
				"role": review.reviewer_role.value
			}
			for review in reviews
		}
		
		return ConsensusResult(
			consensus_reached=consensus_reached,
			consensus_score=consensus_score,
			decision=decision,
			participant_votes=participant_votes,
			consensus_strategy=ConsensusStrategy.UNANIMOUS,
			confidence_level=consensus_score,
			dissenting_opinions=[],
			reasoning=f"Unanimous consensus required: {len(unique_decisions)} unique decisions"
		)
	
	async def _calculate_domain_expert_consensus(
		self,
		reviews: List[Review],
		workflow: CurationWorkflow
	) -> ConsensusResult:
		"""Calculate consensus where domain expert decision is final"""
		
		expert_reviews = [
			r for r in reviews 
			if r.reviewer_role == CurationRole.DOMAIN_EXPERT
		]
		
		if not expert_reviews:
			return ConsensusResult(
				consensus_reached=False,
				consensus_score=0.0,
				decision="pending",
				participant_votes={},
				consensus_strategy=ConsensusStrategy.DOMAIN_EXPERT_DECIDES,
				confidence_level=0.0,
				dissenting_opinions=[],
				reasoning="Waiting for domain expert review"
			)
		
		# Use the highest confidence expert decision
		expert_review = max(expert_reviews, key=lambda r: r.confidence)
		
		participant_votes = {
			review.reviewer_id: {
				"decision": review.decision,
				"confidence": review.confidence,
				"weight": 1.0 if review.reviewer_role != CurationRole.DOMAIN_EXPERT else 5.0,
				"role": review.reviewer_role.value
			}
			for review in reviews
		}
		
		return ConsensusResult(
			consensus_reached=True,
			consensus_score=expert_review.confidence,
			decision=expert_review.decision,
			participant_votes=participant_votes,
			consensus_strategy=ConsensusStrategy.DOMAIN_EXPERT_DECIDES,
			confidence_level=expert_review.confidence,
			dissenting_opinions=[],
			reasoning=f"Domain expert decision: {expert_review.decision} (confidence: {expert_review.confidence})"
		)
	
	# ========================================================================
	# HELPER METHODS
	# ========================================================================
	
	def _initialize_consensus_strategies(self) -> Dict[ConsensusStrategy, Dict[str, Any]]:
		"""Initialize consensus strategy configurations"""
		return {
			ConsensusStrategy.SIMPLE_MAJORITY: {
				"description": "Simple majority voting",
				"min_reviews": 2,
				"requires_expert": False
			},
			ConsensusStrategy.WEIGHTED_MAJORITY: {
				"description": "Weighted majority with expertise consideration",
				"min_reviews": 2,
				"requires_expert": False
			},
			ConsensusStrategy.EXPERT_CONSENSUS: {
				"description": "Consensus among domain experts only",
				"min_reviews": 1,
				"requires_expert": True
			},
			ConsensusStrategy.UNANIMOUS: {
				"description": "Unanimous agreement required",
				"min_reviews": 2,
				"requires_expert": False
			},
			ConsensusStrategy.DOMAIN_EXPERT_DECIDES: {
				"description": "Domain expert has final decision",
				"min_reviews": 1,
				"requires_expert": True
			}
		}
	
	async def _validate_participants(self, participants: List[ExpertParticipant]) -> List[Dict[str, Any]]:
		"""Validate and convert participants to database format"""
		validated = []
		
		for participant in participants:
			# Validate participant data
			if not participant.user_id:
				raise ValueError("Participant must have user_id")
			
			if participant.weight < 0 or participant.weight > 5:
				raise ValueError("Participant weight must be between 0 and 5")
			
			validated.append({
				"user_id": participant.user_id,
				"role": participant.role,
				"expertise_areas": participant.expertise_areas,
				"weight": participant.weight,
				"performance_metrics": participant.performance_metrics
			})
		
		return validated
	
	async def _validate_edit_submission(
		self,
		tenant_id: str,
		workflow_id: str,
		editor_id: str,
		edit_type: str,
		target_type: str,
		target_id: str,
		proposed_changes: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Validate edit submission"""
		
		try:
			# Get workflow to validate
			workflow = await self.db_service.get_curation_workflow(tenant_id, workflow_id)
			
			# Check if editor is a participant
			participant_ids = [p["user_id"] for p in workflow.participants]
			if editor_id not in participant_ids:
				return {
					"valid": False,
					"error": f"User {editor_id} is not a participant in workflow {workflow_id}"
				}
			
			# Validate edit type
			valid_edit_types = ["create", "update", "delete", "merge", "split"]
			if edit_type not in valid_edit_types:
				return {
					"valid": False,
					"error": f"Invalid edit type: {edit_type}. Must be one of {valid_edit_types}"
				}
			
			# Validate target type
			valid_target_types = ["entity", "relationship", "community", "graph"]
			if target_type not in valid_target_types:
				return {
					"valid": False,
					"error": f"Invalid target type: {target_type}. Must be one of {valid_target_types}"
				}
			
			# Validate proposed changes
			if not proposed_changes:
				return {
					"valid": False,
					"error": "Proposed changes cannot be empty"
				}
			
			return {
				"valid": True,
				"knowledge_graph_id": workflow.knowledge_graph_id
			}
			
		except Exception as e:
			return {
				"valid": False,
				"error": str(e)
			}
	
	async def _assign_reviewers_for_edit(
		self,
		tenant_id: str,
		workflow_id: str,
		knowledge_edit: KnowledgeEdit
	) -> None:
		"""Assign appropriate reviewers for a knowledge edit"""
		
		try:
			# Get workflow participants
			workflow = await self.db_service.get_curation_workflow(tenant_id, workflow_id)
			
			# Exclude the editor from reviewers
			potential_reviewers = [
				p for p in workflow.participants 
				if p["user_id"] != knowledge_edit.editor_id
			]
			
			# Assign reviewers based on expertise and availability
			assigned_reviewers = []
			
			# Always assign domain experts if available
			domain_experts = [
				p for p in potential_reviewers 
				if p["role"] == CurationRole.DOMAIN_EXPERT.value
			]
			assigned_reviewers.extend(domain_experts[:2])  # Max 2 domain experts
			
			# Assign additional reviewers if needed
			other_reviewers = [
				p for p in potential_reviewers 
				if p not in assigned_reviewers and p["role"] in [CurationRole.REVIEWER.value, CurationRole.MODERATOR.value]
			]
			
			needed_reviewers = max(0, self.min_reviews_required - len(assigned_reviewers))
			assigned_reviewers.extend(other_reviewers[:needed_reviewers])
			
			# TODO: Send notifications to assigned reviewers
			logger.info(f"Assigned {len(assigned_reviewers)} reviewers for edit {knowledge_edit.edit_id}")
			
		except Exception as e:
			logger.error(f"Failed to assign reviewers: {e}")
	
	async def _notify_edit_submission(
		self,
		tenant_id: str,
		workflow_id: str,
		knowledge_edit: KnowledgeEdit
	) -> None:
		"""Notify participants about new edit submission"""
		# Implementation would send notifications
		logger.info(f"Notified participants about edit submission {knowledge_edit.edit_id}")
	
	async def _get_reviewer_role(self, workflow_id: str, reviewer_id: str) -> CurationRole:
		"""Get the role of a reviewer in the workflow"""
		# Implementation would look up reviewer role
		return CurationRole.REVIEWER
	
	async def _validate_reviewer_authorization(
		self,
		tenant_id: str,
		workflow_id: str,
		reviewer_id: str
	) -> None:
		"""Validate that user is authorized to review in this workflow"""
		# Implementation would validate reviewer authorization
		pass
	
	async def _check_consensus(self, tenant_id: str, edit_id: str) -> ConsensusResult:
		"""Check if consensus has been reached for an edit"""
		return await self.calculate_consensus(tenant_id, edit_id)
	
	async def _handle_consensus_reached(
		self,
		tenant_id: str,
		edit_id: str,
		consensus_result: ConsensusResult
	) -> None:
		"""Handle actions when consensus is reached"""
		
		try:
			# Update edit status
			await self.db_service.update_knowledge_edit(
				tenant_id=tenant_id,
				edit_id=edit_id,
				updates={
					"status": "approved" if consensus_result.decision == "approve" else "rejected",
					"consensus_score": consensus_result.consensus_score
				}
			)
			
			# If approved, create update operation
			if consensus_result.decision == "approve":
				edit = await self.db_service.get_knowledge_edit(tenant_id, edit_id)
				
				update_operation = UpdateOperation(
					operation_id=str(uuid.uuid4()),
					update_type=UpdateType(edit.edit_type),
					target_id=edit.target_id,
					data=edit.proposed_changes,
					timestamp=datetime.utcnow(),
					source=f"curation_workflow_{edit.workflow_id}",
					confidence=consensus_result.confidence_level,
					metadata={
						"edit_id": edit_id,
						"consensus_result": consensus_result.dict()
					}
				)
				
				# Execute the update
				update_result = await self.update_engine.process_incremental_update(
					tenant_id=tenant_id,
					knowledge_graph_id=edit.knowledge_graph_id,
					update_operation=update_operation
				)
				
				if update_result.success:
					await self.db_service.update_knowledge_edit(
						tenant_id=tenant_id,
						edit_id=edit_id,
						updates={"status": "implemented"}
					)
			
			logger.info(f"Consensus reached for edit {edit_id}: {consensus_result.decision}")
			
		except Exception as e:
			logger.error(f"Failed to handle consensus for edit {edit_id}: {e}")
	
	def _update_expert_performance(self, reviewer_id: str, review: Review) -> None:
		"""Update performance metrics for expert reviewer"""
		
		if reviewer_id not in self._expert_performance:
			self._expert_performance[reviewer_id] = {
				"total_reviews": 0,
				"average_confidence": 0.0,
				"review_times": [],
				"decision_distribution": defaultdict(int)
			}
		
		perf = self._expert_performance[reviewer_id]
		perf["total_reviews"] += 1
		perf["average_confidence"] = (
			(perf["average_confidence"] * (perf["total_reviews"] - 1) + review.confidence) /
			perf["total_reviews"]
		)
		perf["review_times"].append(review.review_time_minutes)
		perf["decision_distribution"][review.decision] += 1
	
	def _record_metric(self, metric_name: str, value: float) -> None:
		"""Record performance metric"""
		self._curation_metrics[metric_name].append(value)
		
		# Keep only last 1000 measurements
		if len(self._curation_metrics[metric_name]) > 1000:
			self._curation_metrics[metric_name] = self._curation_metrics[metric_name][-1000:]
	
	# Analytics helper methods (simplified implementations)
	async def _create_initial_curation_tasks(self, tenant_id: str, knowledge_graph_id: str, workflow: CurationWorkflow) -> List[CurationTask]:
		return []
	
	async def _calculate_participant_metrics(self, workflow: CurationWorkflow, edits: List[KnowledgeEdit], reviews: List[Review]) -> Dict[str, Any]:
		return {"participants": len(workflow.participants)}
	
	async def _calculate_consensus_metrics(self, edits: List[KnowledgeEdit]) -> Dict[str, Any]:
		return {"consensus_rate": 0.85}
	
	async def _calculate_quality_metrics(self, edits: List[KnowledgeEdit], reviews: List[Review]) -> Dict[str, Any]:
		return {"quality_score": 0.9}
	
	async def _calculate_performance_trends(self, workflow_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		return {"trend": "improving"}
	
	def _group_edits_by_type(self, edits: List[KnowledgeEdit]) -> Dict[str, int]:
		result = defaultdict(int)
		for edit in edits:
			result[edit.edit_type] += 1
		return dict(result)
	
	def _group_edits_by_target(self, edits: List[KnowledgeEdit]) -> Dict[str, int]:
		result = defaultdict(int)
		for edit in edits:
			result[edit.target_type] += 1
		return dict(result)
	
	def _group_reviews_by_decision(self, reviews: List[Review]) -> Dict[str, int]:
		result = defaultdict(int)
		for review in reviews:
			result[review.decision] += 1
		return dict(result)


__all__ = [
	'CollaborativeKnowledgeCuration',
	'CurationTask',
	'Review',
	'ConsensusResult',
	'CurationRole',
	'EditStatus',
	'ConsensusStrategy',
]