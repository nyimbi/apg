"""
© 2025 Datacraft
Emotional Intelligence Integration System for Workflow Orchestration

This module provides emotional intelligence capabilities for workflow orchestration,
including user sentiment analysis, stress-aware scheduling, and empathetic messaging.
Leverages existing APG capabilities for NLP, notifications, and user management.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from annotated_types import Annotated
from uuid_extensions import uuid7str

from apg.core.base_service import APGBaseService
from apg.core.database import DatabaseManager
from apg.core.auth import AuthManager
from apg.core.notifications import NotificationService, NotificationConfig
from apg.core.nlp import NLPService, SentimentAnalysis, TextProcessor
from apg.core.user_management import UserService, UserProfile
from apg.core.metrics import MetricsCollector
from apg.common.logging import get_logger
from apg.common.exceptions import APGException

logger = get_logger(__name__)

class EmotionalState(str, Enum):
	"""User emotional states based on behavioral patterns"""
	CALM = "calm"
	EXCITED = "excited"
	STRESSED = "stressed"
	FRUSTRATED = "frustrated"
	FOCUSED = "focused"
	OVERWHELMED = "overwhelmed"
	SATISFIED = "satisfied"
	ANXIOUS = "anxious"
	CONFIDENT = "confident"
	UNCERTAIN = "uncertain"

class StressLevel(str, Enum):
	"""Stress levels for workload management"""
	LOW = "low"
	MODERATE = "moderate"
	HIGH = "high"
	CRITICAL = "critical"

class InteractionMode(str, Enum):
	"""Interaction modes for empathetic messaging"""
	SUPPORTIVE = "supportive"
	ENCOURAGING = "encouraging"
	INSTRUCTIONAL = "instructional"
	REASSURING = "reassuring"
	CELEBRATORY = "celebratory"
	GENTLE = "gentle"
	PROFESSIONAL = "professional"

class UserSentiment(BaseModel):
	"""User sentiment analysis result"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str
	sentiment_score: float = Field(ge=-1.0, le=1.0, description="Sentiment from -1 (negative) to 1 (positive)")
	emotional_state: EmotionalState
	confidence: float = Field(ge=0.0, le=1.0)
	stress_level: StressLevel
	engagement_level: float = Field(ge=0.0, le=1.0)
	satisfaction_score: float = Field(ge=0.0, le=1.0)
	frustration_indicators: List[str] = Field(default_factory=list)
	timestamp: datetime = Field(default_factory=datetime.utcnow)

class WorkloadImpact(BaseModel):
	"""Impact assessment for workload changes"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	current_workload: int
	recommended_workload: int
	stress_reduction: float
	productivity_impact: float
	wellbeing_score: float
	recommendations: List[str]

class EmpatheticMessage(BaseModel):
	"""Empathetic message with emotional context"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	user_id: str
	content: str
	interaction_mode: InteractionMode
	emotional_context: EmotionalState
	personalization_factors: Dict[str, Any] = Field(default_factory=dict)
	delivery_time: Optional[datetime] = None
	channel: str = "system"
	priority: str = "normal"

class StressIndicator(BaseModel):
	"""Stress indicator from user behavior"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	indicator_type: str
	severity: float = Field(ge=0.0, le=1.0)
	description: str
	detected_at: datetime = Field(default_factory=datetime.utcnow)
	source: str

class EmotionalIntelligenceConfig(BaseModel):
	"""Configuration for emotional intelligence system"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	sentiment_analysis_interval: int = 300  # seconds
	stress_monitoring_enabled: bool = True
	empathetic_messaging_enabled: bool = True
	workload_adjustment_enabled: bool = True
	sentiment_history_days: int = 30
	stress_threshold_moderate: float = 0.6
	stress_threshold_high: float = 0.8
	message_personalization_level: float = 0.7
	adaptive_scheduling_factor: float = 0.3

class SentimentAnalyzer:
	"""Advanced sentiment analysis using APG NLP capabilities"""
	
	def __init__(self, nlp_service: NLPService, config: EmotionalIntelligenceConfig):
		self.nlp_service = nlp_service
		self.config = config
		self.sentiment_model = nlp_service.get_sentiment_analyzer()
		self.text_processor = nlp_service.get_text_processor()
	
	async def analyze_user_sentiment(self, user_id: str, text_inputs: List[str]) -> UserSentiment:
		"""Analyze user sentiment from various text inputs"""
		try:
			# Process text inputs
			processed_texts = []
			for text in text_inputs:
				processed = await self.text_processor.process(text)
				processed_texts.append(processed)
			
			# Combine texts for analysis
			combined_text = " ".join(processed_texts)
			
			# Perform sentiment analysis
			sentiment_result = await self.sentiment_model.analyze(combined_text)
			
			# Extract emotional indicators
			emotional_state = self._determine_emotional_state(sentiment_result)
			stress_level = self._assess_stress_level(sentiment_result, text_inputs)
			frustration_indicators = self._identify_frustration_indicators(text_inputs)
			
			# Calculate engagement and satisfaction
			engagement_level = self._calculate_engagement(sentiment_result, text_inputs)
			satisfaction_score = self._calculate_satisfaction(sentiment_result)
			
			return UserSentiment(
				user_id=user_id,
				sentiment_score=sentiment_result.polarity,
				emotional_state=emotional_state,
				confidence=sentiment_result.confidence,
				stress_level=stress_level,
				engagement_level=engagement_level,
				satisfaction_score=satisfaction_score,
				frustration_indicators=frustration_indicators
			)
		
		except Exception as e:
			logger.error(f"Sentiment analysis failed for user {user_id}: {str(e)}")
			# Return neutral sentiment as fallback
			return UserSentiment(
				user_id=user_id,
				sentiment_score=0.0,
				emotional_state=EmotionalState.CALM,
				confidence=0.5,
				stress_level=StressLevel.LOW,
				engagement_level=0.5,
				satisfaction_score=0.5
			)
	
	def _determine_emotional_state(self, sentiment_result: Any) -> EmotionalState:
		"""Determine emotional state from sentiment analysis"""
		polarity = sentiment_result.polarity
		subjectivity = getattr(sentiment_result, 'subjectivity', 0.5)
		
		if polarity > 0.6:
			return EmotionalState.EXCITED if subjectivity > 0.7 else EmotionalState.SATISFIED
		elif polarity > 0.2:
			return EmotionalState.CONFIDENT if subjectivity < 0.5 else EmotionalState.FOCUSED
		elif polarity > -0.2:
			return EmotionalState.CALM
		elif polarity > -0.6:
			return EmotionalState.UNCERTAIN if subjectivity > 0.6 else EmotionalState.ANXIOUS
		else:
			return EmotionalState.FRUSTRATED if subjectivity > 0.7 else EmotionalState.STRESSED
	
	def _assess_stress_level(self, sentiment_result: Any, text_inputs: List[str]) -> StressLevel:
		"""Assess stress level from text analysis"""
		stress_indicators = 0
		stress_keywords = [
			"urgent", "deadline", "pressure", "overwhelmed", "stressed",
			"can't", "won't work", "failing", "broken", "emergency"
		]
		
		combined_text = " ".join(text_inputs).lower()
		
		for keyword in stress_keywords:
			if keyword in combined_text:
				stress_indicators += 1
		
		# Calculate stress score
		stress_score = min(1.0, stress_indicators / len(stress_keywords) + abs(min(0, sentiment_result.polarity)))
		
		if stress_score >= self.config.stress_threshold_high:
			return StressLevel.CRITICAL
		elif stress_score >= self.config.stress_threshold_moderate:
			return StressLevel.HIGH
		elif stress_score >= 0.3:
			return StressLevel.MODERATE
		else:
			return StressLevel.LOW
	
	def _identify_frustration_indicators(self, text_inputs: List[str]) -> List[str]:
		"""Identify specific frustration indicators"""
		indicators = []
		combined_text = " ".join(text_inputs).lower()
		
		frustration_patterns = {
			"repeated_failures": ["again", "still", "keep failing", "won't work"],
			"time_pressure": ["urgent", "asap", "immediately", "rush"],
			"confusion": ["don't understand", "confused", "unclear", "help"],
			"technical_issues": ["error", "bug", "broken", "not working"],
			"workload_concerns": ["too much", "overwhelmed", "can't handle"]
		}
		
		for indicator_type, patterns in frustration_patterns.items():
			for pattern in patterns:
				if pattern in combined_text:
					indicators.append(indicator_type)
					break
		
		return indicators
	
	def _calculate_engagement(self, sentiment_result: Any, text_inputs: List[str]) -> float:
		"""Calculate user engagement level"""
		text_length = sum(len(text) for text in text_inputs)
		engagement_keywords = ["excited", "interesting", "great", "love", "awesome"]
		
		combined_text = " ".join(text_inputs).lower()
		keyword_matches = sum(1 for keyword in engagement_keywords if keyword in combined_text)
		
		# Base engagement on text length, keyword matches, and sentiment
		base_engagement = min(1.0, text_length / 500)  # Normalize by typical message length
		keyword_boost = min(0.3, keyword_matches * 0.1)
		sentiment_factor = max(0, sentiment_result.polarity) * 0.3
		
		return min(1.0, base_engagement + keyword_boost + sentiment_factor)
	
	def _calculate_satisfaction(self, sentiment_result: Any) -> float:
		"""Calculate user satisfaction score"""
		# Map sentiment polarity to satisfaction (0-1 scale)
		return max(0.0, (sentiment_result.polarity + 1) / 2)

class StressAwareScheduler:
	"""Stress-aware workflow scheduling system"""
	
	def __init__(self, config: EmotionalIntelligenceConfig):
		self.config = config
		self.stress_adjustments: Dict[str, float] = {}
	
	async def adjust_workload_for_stress(self, user_id: str, current_sentiment: UserSentiment, 
										 current_workload: List[Dict[str, Any]]) -> WorkloadImpact:
		"""Adjust workload based on user stress levels"""
		try:
			current_count = len(current_workload)
			stress_factor = self._get_stress_factor(current_sentiment.stress_level)
			
			# Calculate recommended workload
			recommended_count = max(1, int(current_count * stress_factor))
			
			# Assess impact
			stress_reduction = self._calculate_stress_reduction(current_sentiment.stress_level, stress_factor)
			productivity_impact = self._estimate_productivity_impact(stress_factor)
			wellbeing_score = self._calculate_wellbeing_score(current_sentiment, stress_factor)
			
			# Generate recommendations
			recommendations = self._generate_workload_recommendations(
				current_sentiment, current_count, recommended_count
			)
			
			return WorkloadImpact(
				current_workload=current_count,
				recommended_workload=recommended_count,
				stress_reduction=stress_reduction,
				productivity_impact=productivity_impact,
				wellbeing_score=wellbeing_score,
				recommendations=recommendations
			)
		
		except Exception as e:
			logger.error(f"Workload adjustment failed for user {user_id}: {str(e)}")
			return WorkloadImpact(
				current_workload=len(current_workload),
				recommended_workload=len(current_workload),
				stress_reduction=0.0,
				productivity_impact=0.0,
				wellbeing_score=0.5,
				recommendations=["Continue with current workload"]
			)
	
	def _get_stress_factor(self, stress_level: StressLevel) -> float:
		"""Get workload adjustment factor based on stress level"""
		stress_factors = {
			StressLevel.LOW: 1.0,
			StressLevel.MODERATE: 0.8,
			StressLevel.HIGH: 0.6,
			StressLevel.CRITICAL: 0.4
		}
		return stress_factors.get(stress_level, 1.0)
	
	def _calculate_stress_reduction(self, stress_level: StressLevel, stress_factor: float) -> float:
		"""Calculate expected stress reduction from workload adjustment"""
		base_reduction = {
			StressLevel.LOW: 0.0,
			StressLevel.MODERATE: 0.3,
			StressLevel.HIGH: 0.5,
			StressLevel.CRITICAL: 0.7
		}
		return base_reduction.get(stress_level, 0.0) * (1 - stress_factor)
	
	def _estimate_productivity_impact(self, stress_factor: float) -> float:
		"""Estimate productivity impact of workload adjustment"""
		# Positive impact for stress reduction, negative for reduced tasks
		return (stress_factor - 1) * 0.5  # Will be negative for reduced workload
	
	def _calculate_wellbeing_score(self, sentiment: UserSentiment, stress_factor: float) -> float:
		"""Calculate expected wellbeing score after adjustment"""
		current_wellbeing = sentiment.satisfaction_score
		stress_improvement = (1 - stress_factor) * 0.3
		return min(1.0, current_wellbeing + stress_improvement)
	
	def _generate_workload_recommendations(self, sentiment: UserSentiment, 
										   current: int, recommended: int) -> List[str]:
		"""Generate specific workload recommendations"""
		recommendations = []
		
		if recommended < current:
			diff = current - recommended
			recommendations.append(f"Reduce active tasks by {diff} to manage stress levels")
			
			if sentiment.stress_level == StressLevel.CRITICAL:
				recommendations.append("Consider delegating urgent tasks to team members")
				recommendations.append("Schedule a break or wellness check-in")
			
			if sentiment.frustration_indicators:
				recommendations.append("Focus on resolving blocking issues before adding new tasks")
		
		elif recommended == current:
			recommendations.append("Current workload appears manageable")
			
			if sentiment.engagement_level < 0.5:
				recommendations.append("Consider adding more engaging or varied tasks")
		
		# Add specific recommendations based on emotional state
		if sentiment.emotional_state == EmotionalState.OVERWHELMED:
			recommendations.append("Break large tasks into smaller, manageable chunks")
		elif sentiment.emotional_state == EmotionalState.FRUSTRATED:
			recommendations.append("Prioritize resolving technical blockers")
		elif sentiment.emotional_state == EmotionalState.ANXIOUS:
			recommendations.append("Provide clear task expectations and deadlines")
		
		return recommendations

class EmpatheticMessageGenerator:
	"""Generate empathetic messages based on emotional context"""
	
	def __init__(self, nlp_service: NLPService, user_service: UserService):
		self.nlp_service = nlp_service
		self.user_service = user_service
		self.message_templates = self._load_message_templates()
	
	async def generate_empathetic_message(self, user_id: str, sentiment: UserSentiment, 
										  context: Dict[str, Any]) -> EmpatheticMessage:
		"""Generate personalized empathetic message"""
		try:
			# Get user profile for personalization
			user_profile = await self.user_service.get_user_profile(user_id)
			
			# Determine interaction mode
			interaction_mode = self._determine_interaction_mode(sentiment, context)
			
			# Generate message content
			content = await self._generate_message_content(
				sentiment, context, user_profile, interaction_mode
			)
			
			# Personalize message
			personalized_content = self._personalize_message(content, user_profile, sentiment)
			
			# Determine delivery timing
			delivery_time = self._calculate_optimal_delivery_time(sentiment, context)
			
			return EmpatheticMessage(
				user_id=user_id,
				content=personalized_content,
				interaction_mode=interaction_mode,
				emotional_context=sentiment.emotional_state,
				personalization_factors={
					"stress_level": sentiment.stress_level,
					"engagement_level": sentiment.engagement_level,
					"satisfaction_score": sentiment.satisfaction_score,
					"user_preferences": user_profile.preferences if user_profile else {}
				},
				delivery_time=delivery_time,
				channel=self._select_delivery_channel(sentiment, user_profile),
				priority=self._determine_message_priority(sentiment)
			)
		
		except Exception as e:
			logger.error(f"Message generation failed for user {user_id}: {str(e)}")
			return EmpatheticMessage(
				user_id=user_id,
				content="Thank you for your continued work on the workflow system.",
				interaction_mode=InteractionMode.PROFESSIONAL,
				emotional_context=sentiment.emotional_state
			)
	
	def _determine_interaction_mode(self, sentiment: UserSentiment, context: Dict[str, Any]) -> InteractionMode:
		"""Determine appropriate interaction mode"""
		if sentiment.stress_level in [StressLevel.HIGH, StressLevel.CRITICAL]:
			return InteractionMode.SUPPORTIVE
		elif sentiment.emotional_state == EmotionalState.FRUSTRATED:
			return InteractionMode.REASSURING
		elif sentiment.emotional_state in [EmotionalState.EXCITED, EmotionalState.SATISFIED]:
			return InteractionMode.CELEBRATORY
		elif sentiment.emotional_state == EmotionalState.UNCERTAIN:
			return InteractionMode.ENCOURAGING
		elif sentiment.engagement_level < 0.4:
			return InteractionMode.ENCOURAGING
		else:
			return InteractionMode.PROFESSIONAL
	
	async def _generate_message_content(self, sentiment: UserSentiment, context: Dict[str, Any],
										user_profile: Any, interaction_mode: InteractionMode) -> str:
		"""Generate message content using NLP templates"""
		template_key = f"{sentiment.emotional_state}_{interaction_mode}"
		template = self.message_templates.get(template_key, self.message_templates["default"])
		
		# Context-aware content generation
		if context.get("workflow_completed"):
			if sentiment.satisfaction_score > 0.7:
				return "Congratulations on completing your workflow! Your attention to detail and persistence really paid off."
			else:
				return "Great job completing your workflow. I noticed it was challenging - your perseverance is admirable."
		
		elif context.get("workflow_failed"):
			if sentiment.stress_level == StressLevel.CRITICAL:
				return "I understand this failure is frustrating, especially with everything else you're managing. Let's break this down into smaller steps."
			else:
				return "Workflow failures can be discouraging, but they're also learning opportunities. Let's figure out what happened together."
		
		elif context.get("high_workload"):
			return "I notice you have quite a lot on your plate right now. Would it help to prioritize or defer some of the less urgent tasks?"
		
		return template
	
	def _personalize_message(self, content: str, user_profile: Any, sentiment: UserSentiment) -> str:
		"""Add personalization to message content"""
		if not user_profile:
			return content
		
		# Add user's preferred name/title
		if hasattr(user_profile, 'preferred_name') and user_profile.preferred_name:
			if not any(name in content for name in [user_profile.preferred_name, "you"]):
				content = f"{user_profile.preferred_name}, {content.lower()}"
		
		# Adjust tone based on user preferences
		if hasattr(user_profile, 'communication_style'):
			if user_profile.communication_style == "formal":
				content = content.replace("Let's", "We can").replace("you're", "you are")
			elif user_profile.communication_style == "casual":
				content = content.replace("We can", "Let's").replace("you are", "you're")
		
		return content
	
	def _calculate_optimal_delivery_time(self, sentiment: UserSentiment, context: Dict[str, Any]) -> Optional[datetime]:
		"""Calculate optimal message delivery time"""
		now = datetime.utcnow()
		
		# Immediate delivery for critical stress
		if sentiment.stress_level == StressLevel.CRITICAL:
			return now
		
		# Immediate delivery for significant events
		if context.get("workflow_completed") or context.get("workflow_failed"):
			return now
		
		# Delay supportive messages to avoid interruption
		if sentiment.emotional_state == EmotionalState.FOCUSED:
			return now + timedelta(minutes=30)
		
		# Default to immediate delivery
		return now
	
	def _select_delivery_channel(self, sentiment: UserSentiment, user_profile: Any) -> str:
		"""Select appropriate delivery channel"""
		if sentiment.stress_level == StressLevel.CRITICAL:
			return "priority_notification"
		elif hasattr(user_profile, 'preferred_channel'):
			return user_profile.preferred_channel
		else:
			return "system"
	
	def _determine_message_priority(self, sentiment: UserSentiment) -> str:
		"""Determine message priority"""
		if sentiment.stress_level == StressLevel.CRITICAL:
			return "high"
		elif sentiment.stress_level == StressLevel.HIGH:
			return "medium"
		else:
			return "normal"
	
	def _load_message_templates(self) -> Dict[str, str]:
		"""Load empathetic message templates"""
		return {
			"stressed_supportive": "I can see you're dealing with a lot right now. Let's see how we can make this more manageable for you.",
			"frustrated_reassuring": "I understand your frustration. These technical challenges can be really tough, but we'll work through this together.",
			"overwhelmed_supportive": "It looks like you have a lot going on. Would it help to focus on just the most important tasks for now?",
			"excited_celebratory": "I love your enthusiasm! Your energy really makes a difference in how smoothly things run.",
			"satisfied_celebratory": "You should feel proud of what you've accomplished. Your work is making a real impact.",
			"anxious_encouraging": "It's natural to feel uncertain about new challenges. You've handled difficult situations before, and I'm confident you can do this too.",
			"uncertain_encouraging": "When things feel unclear, it often helps to start with what you do know. Let's build from there.",
			"focused_professional": "I can see you're in a good flow. I'll keep things streamlined so you can maintain your focus.",
			"calm_professional": "Everything seems to be running smoothly. Let me know if you need any adjustments.",
			"confident_professional": "Your confidence in tackling these workflows is evident. Keep up the excellent work.",
			"default": "Thank you for your continued work. I'm here to help make your workflow experience as smooth as possible."
		}

class EmotionalIntelligenceService(APGBaseService):
	"""Main emotional intelligence service"""
	
	def __init__(self, config: EmotionalIntelligenceConfig, 
				 nlp_service: NLPService, 
				 notification_service: NotificationService,
				 user_service: UserService,
				 db_manager: DatabaseManager):
		super().__init__()
		self.config = config
		self.nlp_service = nlp_service
		self.notification_service = notification_service
		self.user_service = user_service
		self.db_manager = db_manager
		
		# Components
		self.sentiment_analyzer = SentimentAnalyzer(nlp_service, config)
		self.stress_scheduler = StressAwareScheduler(config)
		self.message_generator = EmpatheticMessageGenerator(nlp_service, user_service)
		
		# State tracking
		self.user_sentiments: Dict[str, UserSentiment] = {}
		self.stress_indicators: Dict[str, List[StressIndicator]] = {}
		self.message_history: Dict[str, List[EmpatheticMessage]] = {}
		
		# Background tasks
		self.monitoring_task: Optional[asyncio.Task] = None
		self.running = False
	
	async def start(self) -> None:
		"""Start emotional intelligence monitoring"""
		if self.running:
			return
		
		self.running = True
		self.monitoring_task = asyncio.create_task(self._monitoring_loop())
		logger.info("Emotional intelligence service started")
	
	async def stop(self) -> None:
		"""Stop emotional intelligence monitoring"""
		self.running = False
		if self.monitoring_task:
			self.monitoring_task.cancel()
			try:
				await self.monitoring_task
			except asyncio.CancelledError:
				pass
		logger.info("Emotional intelligence service stopped")
	
	async def analyze_user_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> UserSentiment:
		"""Analyze user sentiment from interaction data"""
		try:
			# Extract text inputs from interaction
			text_inputs = []
			
			if "message" in interaction_data:
				text_inputs.append(interaction_data["message"])
			
			if "comments" in interaction_data:
				text_inputs.extend(interaction_data["comments"])
			
			if "feedback" in interaction_data:
				text_inputs.append(interaction_data["feedback"])
			
			# Fallback for empty text inputs
			if not text_inputs:
				text_inputs = ["User interaction without text content"]
			
			# Analyze sentiment
			sentiment = await self.sentiment_analyzer.analyze_user_sentiment(user_id, text_inputs)
			
			# Store sentiment
			self.user_sentiments[user_id] = sentiment
			
			# Check for stress indicators
			await self._check_stress_indicators(user_id, sentiment, interaction_data)
			
			# Generate empathetic response if needed
			if self.config.empathetic_messaging_enabled:
				await self._generate_empathetic_response(user_id, sentiment, interaction_data)
			
			return sentiment
		
		except Exception as e:
			logger.error(f"User interaction analysis failed: {str(e)}")
			raise APGException(f"Failed to analyze user interaction: {str(e)}")
	
	async def adjust_workflow_for_user_state(self, user_id: str, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Adjust workflow scheduling based on user emotional state"""
		try:
			# Get current user sentiment
			sentiment = self.user_sentiments.get(user_id)
			if not sentiment:
				# Analyze current state if not available
				sentiment = await self.analyze_user_interaction(user_id, {"message": "Current workflow state"})
			
			# Skip adjustment if not enabled
			if not self.config.workload_adjustment_enabled:
				return workflow_data
			
			# Get current workload
			current_workload = workflow_data.get("tasks", [])
			
			# Calculate workload adjustment
			workload_impact = await self.stress_scheduler.adjust_workload_for_stress(
				user_id, sentiment, current_workload
			)
			
			# Apply adjustments if beneficial
			if workload_impact.wellbeing_score > 0.6 and workload_impact.recommended_workload != workload_impact.current_workload:
				adjusted_tasks = current_workload[:workload_impact.recommended_workload]
				deferred_tasks = current_workload[workload_impact.recommended_workload:]
				
				# Update workflow data
				workflow_data["tasks"] = adjusted_tasks
				workflow_data["deferred_tasks"] = deferred_tasks
				workflow_data["workload_adjustment"] = {
					"applied": True,
					"reason": f"Stress level: {sentiment.stress_level}",
					"impact": workload_impact.dict(),
					"recommendations": workload_impact.recommendations
				}
				
				# Notify user about adjustment
				await self._notify_workload_adjustment(user_id, workload_impact)
			
			return workflow_data
		
		except Exception as e:
			logger.error(f"Workflow adjustment failed for user {user_id}: {str(e)}")
			return workflow_data
	
	async def get_user_emotional_state(self, user_id: str) -> Optional[UserSentiment]:
		"""Get current emotional state for a user"""
		return self.user_sentiments.get(user_id)
	
	async def get_stress_indicators(self, user_id: str) -> List[StressIndicator]:
		"""Get stress indicators for a user"""
		return self.stress_indicators.get(user_id, [])
	
	async def _monitoring_loop(self) -> None:
		"""Background monitoring loop"""
		while self.running:
			try:
				await asyncio.sleep(self.config.sentiment_analysis_interval)
				
				# Monitor all active users
				active_users = await self._get_active_users()
				
				for user_id in active_users:
					await self._monitor_user_wellbeing(user_id)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Monitoring loop error: {str(e)}")
				await asyncio.sleep(60)  # Wait before retrying
	
	async def _check_stress_indicators(self, user_id: str, sentiment: UserSentiment, interaction_data: Dict[str, Any]) -> None:
		"""Check and record stress indicators"""
		indicators = []
		
		# High stress level
		if sentiment.stress_level in [StressLevel.HIGH, StressLevel.CRITICAL]:
			indicators.append(StressIndicator(
				indicator_type="high_stress_level",
				severity=0.8 if sentiment.stress_level == StressLevel.HIGH else 1.0,
				description=f"User stress level detected as {sentiment.stress_level}",
				source="sentiment_analysis"
			))
		
		# Low satisfaction
		if sentiment.satisfaction_score < 0.3:
			indicators.append(StressIndicator(
				indicator_type="low_satisfaction",
				severity=1.0 - sentiment.satisfaction_score,
				description=f"Low satisfaction score: {sentiment.satisfaction_score}",
				source="sentiment_analysis"
			))
		
		# Frustration indicators
		if sentiment.frustration_indicators:
			indicators.append(StressIndicator(
				indicator_type="frustration_patterns",
				severity=min(1.0, len(sentiment.frustration_indicators) * 0.2),
				description=f"Frustration patterns: {', '.join(sentiment.frustration_indicators)}",
				source="text_analysis"
			))
		
		# Store indicators
		if user_id not in self.stress_indicators:
			self.stress_indicators[user_id] = []
		
		self.stress_indicators[user_id].extend(indicators)
		
		# Keep only recent indicators
		cutoff_time = datetime.utcnow() - timedelta(days=7)
		self.stress_indicators[user_id] = [
			indicator for indicator in self.stress_indicators[user_id]
			if indicator.detected_at > cutoff_time
		]
	
	async def _generate_empathetic_response(self, user_id: str, sentiment: UserSentiment, context: Dict[str, Any]) -> None:
		"""Generate and deliver empathetic response"""
		try:
			# Generate message
			message = await self.message_generator.generate_empathetic_message(user_id, sentiment, context)
			
			# Store message
			if user_id not in self.message_history:
				self.message_history[user_id] = []
			self.message_history[user_id].append(message)
			
			# Deliver message
			notification_config = NotificationConfig(
				recipient_id=user_id,
				title="Workflow Assistant",
				message=message.content,
				channel=message.channel,
				priority=message.priority,
				delivery_time=message.delivery_time
			)
			
			await self.notification_service.send_notification(notification_config)
			
		except Exception as e:
			logger.error(f"Empathetic response generation failed: {str(e)}")
	
	async def _notify_workload_adjustment(self, user_id: str, workload_impact: WorkloadImpact) -> None:
		"""Notify user about workload adjustment"""
		try:
			message = f"I've adjusted your workload to help manage stress levels. "
			message += f"Reduced from {workload_impact.current_workload} to {workload_impact.recommended_workload} active tasks. "
			
			if workload_impact.recommendations:
				message += f"\n\nRecommendations:\n• " + "\n• ".join(workload_impact.recommendations)
			
			notification_config = NotificationConfig(
				recipient_id=user_id,
				title="Workload Adjustment",
				message=message,
				channel="system",
				priority="medium"
			)
			
			await self.notification_service.send_notification(notification_config)
			
		except Exception as e:
			logger.error(f"Workload adjustment notification failed: {str(e)}")
	
	async def _get_active_users(self) -> List[str]:
		"""Get list of currently active users"""
		try:
			# Get active users from user service
			active_users = await self.user_service.get_active_users()
			return [user.id for user in active_users]
		except Exception as e:
			logger.error(f"Failed to get active users: {str(e)}")
			return []
	
	async def _monitor_user_wellbeing(self, user_id: str) -> None:
		"""Monitor individual user wellbeing"""
		try:
			# Check recent activity and sentiment
			current_sentiment = self.user_sentiments.get(user_id)
			if not current_sentiment:
				return
			
			# Check if intervention is needed
			if current_sentiment.stress_level == StressLevel.CRITICAL:
				await self._trigger_wellbeing_intervention(user_id, current_sentiment)
			
		except Exception as e:
			logger.error(f"User wellbeing monitoring failed for {user_id}: {str(e)}")
	
	async def _trigger_wellbeing_intervention(self, user_id: str, sentiment: UserSentiment) -> None:
		"""Trigger wellbeing intervention for critical stress"""
		try:
			message = "I've noticed you may be experiencing high stress levels. "
			message += "Your wellbeing is important. Consider taking a break or reaching out for support. "
			message += "I've also reduced your current workload to help manage the pressure."
			
			notification_config = NotificationConfig(
				recipient_id=user_id,
				title="Wellbeing Check",
				message=message,
				channel="priority_notification",
				priority="high"
			)
			
			await self.notification_service.send_notification(notification_config)
			
		except Exception as e:
			logger.error(f"Wellbeing intervention failed: {str(e)}")

# Factory function for creating emotional intelligence service
async def create_emotional_intelligence_service(
	config: Optional[EmotionalIntelligenceConfig] = None,
	nlp_service: Optional[NLPService] = None,
	notification_service: Optional[NotificationService] = None,
	user_service: Optional[UserService] = None,
	db_manager: Optional[DatabaseManager] = None
) -> EmotionalIntelligenceService:
	"""Create and configure emotional intelligence service"""
	
	if config is None:
		config = EmotionalIntelligenceConfig()
	
	# Initialize required services if not provided
	if nlp_service is None:
		from apg.core.nlp import create_nlp_service
		nlp_service = await create_nlp_service()
	
	if notification_service is None:
		from apg.core.notifications import create_notification_service
		notification_service = await create_notification_service()
	
	if user_service is None:
		from apg.core.user_management import create_user_service
		user_service = await create_user_service()
	
	if db_manager is None:
		from apg.core.database import create_database_manager
		db_manager = await create_database_manager()
	
	return EmotionalIntelligenceService(
		config=config,
		nlp_service=nlp_service,
		notification_service=notification_service,
		user_service=user_service,
		db_manager=db_manager
	)