#!/usr/bin/env python3
"""
Base Agent Architecture
======================

Base class and interfaces for all APG autonomous agents.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid_extensions import uuid7str

class AgentRole(Enum):
	"""Agent role definitions"""
	ORCHESTRATOR = "orchestrator"
	ARCHITECT = "architect"
	DEVELOPER = "developer"
	TESTER = "tester"
	DEVOPS = "devops"
	SECURITY = "security"
	PERFORMANCE = "performance"
	DOCUMENTATION = "documentation"

class AgentStatus(Enum):
	"""Agent status definitions"""
	IDLE = "idle"
	THINKING = "thinking"
	WORKING = "working"
	WAITING = "waiting"
	COMPLETED = "completed"
	ERROR = "error"
	OFFLINE = "offline"

class MessageType(Enum):
	"""Message types for agent communication"""
	TASK_REQUEST = "task_request"
	TASK_RESPONSE = "task_response"
	COLLABORATION_REQUEST = "collaboration_request"
	COLLABORATION_RESPONSE = "collaboration_response"
	STATUS_UPDATE = "status_update"
	QUESTION = "question"
	ANSWER = "answer"
	NOTIFICATION = "notification"
	ERROR = "error"

@dataclass
class AgentMessage:
	"""Message structure for agent communication"""
	id: str = field(default_factory=uuid7str)
	sender_id: str = ""
	recipient_id: str = ""
	message_type: MessageType = MessageType.NOTIFICATION
	content: Dict[str, Any] = field(default_factory=dict)
	timestamp: datetime = field(default_factory=datetime.utcnow)
	requires_response: bool = False
	parent_message_id: Optional[str] = None
	priority: int = 5  # 1-10, 1 is highest priority

@dataclass
class AgentTask:
	"""Task structure for agent work"""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	description: str = ""
	requirements: Dict[str, Any] = field(default_factory=dict)
	deliverables: List[str] = field(default_factory=list)
	dependencies: List[str] = field(default_factory=list)
	assigned_agent: Optional[str] = None
	status: str = "pending"
	priority: int = 5
	created_at: datetime = field(default_factory=datetime.utcnow)
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	results: Dict[str, Any] = field(default_factory=dict)
	context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentCapability:
	"""Agent capability definition"""
	name: str
	description: str
	skill_level: int  # 1-10, 10 is expert
	domains: List[str] = field(default_factory=list)
	tools: List[str] = field(default_factory=list)
	experience_points: int = 0

@dataclass
class AgentMemory:
	"""Agent memory structure"""
	id: str = field(default_factory=uuid7str)
	agent_id: str = ""
	memory_type: str = "working"  # working, episodic, semantic, procedural
	content: Dict[str, Any] = field(default_factory=dict)
	importance: int = 5  # 1-10
	created_at: datetime = field(default_factory=datetime.utcnow)
	last_accessed: datetime = field(default_factory=datetime.utcnow)
	access_count: int = 0
	tags: List[str] = field(default_factory=list)

class BaseAgent(ABC):
	"""
	Base class for all APG autonomous agents.
	
	Provides core functionality for agent communication, task management,
	memory, learning, and coordination.
	"""
	
	def __init__(
		self,
		agent_id: str,
		role: AgentRole,
		name: str,
		description: str = "",
		capabilities: List[AgentCapability] = None,
		config: Dict[str, Any] = None
	):
		self.agent_id = agent_id
		self.role = role
		self.name = name
		self.description = description
		self.capabilities = capabilities or []
		self.config = config or {}
		
		# Agent state
		self.status = AgentStatus.IDLE
		self.current_task: Optional[AgentTask] = None
		self.task_queue: List[AgentTask] = []
		
		# Communication
		self.message_queue: List[AgentMessage] = []
		self.pending_responses: Dict[str, AgentMessage] = {}
		
		# Memory systems
		self.working_memory: List[AgentMemory] = []
		self.episodic_memory: List[AgentMemory] = []
		self.semantic_memory: List[AgentMemory] = []
		self.procedural_memory: List[AgentMemory] = []
		
		# Learning and improvement
		self.experience_points = 0
		self.performance_metrics: Dict[str, float] = {}
		self.feedback_history: List[Dict[str, Any]] = []
		self.learning_engine = None  # Will be initialized in _initialize()
		
		# Coordination
		self.known_agents: Dict[str, Dict[str, Any]] = {}
		self.collaboration_history: List[Dict[str, Any]] = []
		
		# Logging
		self.logger = logging.getLogger(f"agent.{self.role.value}.{self.agent_id}")
		
		# Initialize agent
		self._initialize()
	
	def _initialize(self):
		"""Initialize agent-specific components"""
		self.logger.info(f"Initializing {self.role.value} agent: {self.name}")
		self._setup_capabilities()
		self._load_memory()
		self._setup_tools()
		self._setup_learning_engine()
	
	@abstractmethod
	def _setup_capabilities(self):
		"""Setup agent-specific capabilities"""
		pass
	
	@abstractmethod  
	def _setup_tools(self):
		"""Setup agent-specific tools"""
		pass
	
	def _load_memory(self):
		"""Load agent memory from persistent storage"""
		# Memory loading logic
		pass
	
	def _save_memory(self):
		"""Save agent memory to persistent storage"""
		# Memory saving logic  
		pass
	
	def _setup_learning_engine(self):
		"""Setup the learning engine for this agent"""
		try:
			from .learning_engine import AgentLearningEngine
			self.learning_engine = AgentLearningEngine(
				agent_id=self.agent_id,
				config=self.config.get('learning', {})
			)
			self.logger.info("Learning engine initialized")
		except ImportError:
			self.logger.warning("Learning engine not available")
			self.learning_engine = None
	
	# Task Management
	
	async def receive_task(self, task: AgentTask) -> bool:
		"""
		Receive a new task for execution.
		
		Args:
			task: The task to execute
			
		Returns:
			bool: True if task was accepted, False otherwise
		"""
		if not self._can_accept_task(task):
			return False
		
		self.task_queue.append(task)
		self.logger.info(f"Accepted task: {task.name}")
		
		# Start processing if idle
		if self.status == AgentStatus.IDLE:
			await self._process_next_task()
		
		return True
	
	def _can_accept_task(self, task: AgentTask) -> bool:
		"""Check if agent can accept the given task"""
		# Check if agent has required capabilities
		required_capabilities = task.requirements.get('capabilities', [])
		agent_capability_names = [cap.name for cap in self.capabilities]
		
		for req_cap in required_capabilities:
			if req_cap not in agent_capability_names:
				return False
		
		# Check task queue capacity
		max_queue_size = self.config.get('max_queue_size', 10)
		if len(self.task_queue) >= max_queue_size:
			return False
		
		return True
	
	async def _process_next_task(self):
		"""Process the next task in the queue"""
		if not self.task_queue or self.status != AgentStatus.IDLE:
			return
		
		# Sort by priority
		self.task_queue.sort(key=lambda t: t.priority)
		task = self.task_queue.pop(0)
		
		self.current_task = task
		self.status = AgentStatus.WORKING
		task.status = "in_progress"
		task.started_at = datetime.utcnow()
		
		self.logger.info(f"Starting task: {task.name}")
		
		try:
			# Execute the task
			results = await self.execute_task(task)
			
			# Mark task as completed
			task.status = "completed"
			task.completed_at = datetime.utcnow()
			task.results = results
			
			# Store episodic memory
			await self._store_episodic_memory(task, results)
			
			# Update performance metrics
			self._update_performance_metrics(task, True)
			
			# Process task completion for learning
			if self.learning_engine:
				await self.learning_engine.process_task_completion(task, results)
			
			self.logger.info(f"Completed task: {task.name}")
			
		except Exception as e:
			task.status = "error"
			task.results = {"error": str(e)}
			self._update_performance_metrics(task, False)
			
			# Process task failure for learning
			if self.learning_engine:
				await self.learning_engine.process_task_completion(task, {"error": str(e)})
			
			self.logger.error(f"Task failed: {task.name} - {e}")
		
		finally:
			self.current_task = None
			self.status = AgentStatus.IDLE
			
			# Process next task if available
			if self.task_queue:
				await self._process_next_task()
	
	@abstractmethod
	async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
		"""
		Execute a specific task.
		
		Args:
			task: The task to execute
			
		Returns:
			Dict containing task results
		"""
		pass
	
	# Communication
	
	async def send_message(self, message: AgentMessage) -> bool:
		"""
		Send a message to another agent.
		
		Args:
			message: The message to send
			
		Returns:
			bool: True if message was sent successfully
		"""
		message.sender_id = self.agent_id
		
		# Find the target agent and deliver message
		# This would integrate with the agent orchestrator
		self.logger.info(f"Sending message to {message.recipient_id}: {message.message_type.value}")
		
		if message.requires_response:
			self.pending_responses[message.id] = message
		
		# In a real implementation, this would go through the message bus
		return True
	
	async def receive_message(self, message: AgentMessage):
		"""
		Receive a message from another agent.
		
		Args:
			message: The received message
		"""
		self.message_queue.append(message)
		self.logger.info(f"Received message from {message.sender_id}: {message.message_type.value}")
		
		# Process message based on type
		await self._process_message(message)
	
	async def _process_message(self, message: AgentMessage):
		"""Process a received message"""
		if message.message_type == MessageType.TASK_REQUEST:
			await self._handle_task_request(message)
		elif message.message_type == MessageType.COLLABORATION_REQUEST:
			await self._handle_collaboration_request(message)
		elif message.message_type == MessageType.QUESTION:
			await self._handle_question(message)
		elif message.message_type == MessageType.STATUS_UPDATE:
			await self._handle_status_update(message)
		# Add more message type handlers
	
	async def _handle_task_request(self, message: AgentMessage):
		"""Handle a task request message"""
		task_data = message.content.get('task', {})
		task = AgentTask(**task_data)
		
		accepted = await self.receive_task(task)
		
		# Send response
		response = AgentMessage(
			recipient_id=message.sender_id,
			message_type=MessageType.TASK_RESPONSE,
			content={
				'task_id': task.id,
				'accepted': accepted,
				'reason': 'Task accepted' if accepted else 'Cannot accept task'
			},
			parent_message_id=message.id
		)
		await self.send_message(response)
	
	async def _handle_collaboration_request(self, message: AgentMessage):
		"""Handle a collaboration request"""
		# Collaboration logic
		pass
	
	async def _handle_question(self, message: AgentMessage):
		"""Handle a question from another agent"""
		question = message.content.get('question', '')
		answer = await self._answer_question(question, message.content)
		
		response = AgentMessage(
			recipient_id=message.sender_id,
			message_type=MessageType.ANSWER,
			content={'answer': answer},
			parent_message_id=message.id
		)
		await self.send_message(response)
	
	async def _answer_question(self, question: str, context: Dict[str, Any]) -> str:
		"""Answer a question based on agent knowledge"""
		# Question answering logic based on agent's expertise
		return "I need to think about that..."
	
	async def _handle_status_update(self, message: AgentMessage):
		"""Handle a status update from another agent"""
		agent_id = message.sender_id
		status_info = message.content
		
		# Update known agent information
		if agent_id not in self.known_agents:
			self.known_agents[agent_id] = {}
		self.known_agents[agent_id].update(status_info)
	
	# Memory Management
	
	async def _store_memory(self, memory: AgentMemory):
		"""Store a memory in the appropriate memory system"""
		if memory.memory_type == "working":
			self.working_memory.append(memory)
			# Limit working memory size
			max_working = self.config.get('max_working_memory', 50)
			if len(self.working_memory) > max_working:
				self.working_memory.pop(0)
		elif memory.memory_type == "episodic":
			self.episodic_memory.append(memory)
		elif memory.memory_type == "semantic":
			self.semantic_memory.append(memory)
		elif memory.memory_type == "procedural":
			self.procedural_memory.append(memory)
	
	async def _store_episodic_memory(self, task: AgentTask, results: Dict[str, Any]):
		"""Store episodic memory of task execution"""
		memory = AgentMemory(
			agent_id=self.agent_id,
			memory_type="episodic",
			content={
				'task': task.__dict__,
				'results': results,
				'context': task.context,
				'performance': self._calculate_task_performance(task)
			},
			importance=self._calculate_memory_importance(task),
			tags=[task.name, self.role.value, 'task_execution']
		)
		await self._store_memory(memory)
	
	def _calculate_task_performance(self, task: AgentTask) -> float:
		"""Calculate performance score for a task"""
		if task.completed_at and task.started_at:
			duration = (task.completed_at - task.started_at).total_seconds()
			# Simple performance metric based on completion time
			return max(0.0, min(10.0, 10.0 - (duration / 3600)))  # Normalize to 1-hour baseline
		return 0.0
	
	def _calculate_memory_importance(self, task: AgentTask) -> int:
		"""Calculate importance score for memory storage"""
		importance = task.priority
		
		# Adjust based on task complexity
		if 'complexity' in task.requirements:
			importance += task.requirements['complexity']
		
		# Adjust based on collaboration
		if 'collaboration' in task.context:
			importance += 2
		
		return min(10, max(1, importance))
	
	async def recall_memory(self, query: str, memory_type: str = None, limit: int = 10) -> List[AgentMemory]:
		"""
		Recall memories based on a query.
		
		Args:
			query: Search query
			memory_type: Type of memory to search (optional)
			limit: Maximum number of memories to return
			
		Returns:
			List of relevant memories
		"""
		memories = []
		
		if memory_type is None or memory_type == "working":
			memories.extend(self.working_memory)
		if memory_type is None or memory_type == "episodic":
			memories.extend(self.episodic_memory)
		if memory_type is None or memory_type == "semantic":
			memories.extend(self.semantic_memory)
		if memory_type is None or memory_type == "procedural":
			memories.extend(self.procedural_memory)
		
		# Simple relevance scoring based on tags and content
		relevant_memories = []
		query_words = query.lower().split()
		
		for memory in memories:
			score = 0
			
			# Check tags
			for tag in memory.tags:
				if any(word in tag.lower() for word in query_words):
					score += 2
			
			# Check content (simplified)
			content_str = str(memory.content).lower()
			for word in query_words:
				if word in content_str:
					score += 1
			
			if score > 0:
				relevant_memories.append((memory, score))
		
		# Sort by relevance and importance
		relevant_memories.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
		
		# Update access information
		result_memories = []
		for memory, score in relevant_memories[:limit]:
			memory.last_accessed = datetime.utcnow()
			memory.access_count += 1
			result_memories.append(memory)
		
		return result_memories
	
	# Learning and Improvement
	
	def _update_performance_metrics(self, task: AgentTask, success: bool):
		"""Update agent performance metrics"""
		task_type = task.requirements.get('type', 'general')
		
		if task_type not in self.performance_metrics:
			self.performance_metrics[task_type] = 0.5  # Start at neutral
		
		# Simple learning rate
		learning_rate = 0.1
		target = 1.0 if success else 0.0
		
		self.performance_metrics[task_type] = (
			(1 - learning_rate) * self.performance_metrics[task_type] +
			learning_rate * target
		)
		
		# Update experience points
		if success:
			self.experience_points += task.priority
		
		# Update capability experience
		for capability in self.capabilities:
			if capability.name in task.requirements.get('capabilities', []):
				capability.experience_points += 1 if success else 0
	
	async def learn_from_feedback(self, feedback: Dict[str, Any]):
		"""Learn from external feedback"""
		self.feedback_history.append({
			'feedback': feedback,
			'timestamp': datetime.utcnow(),
			'context': {
				'current_task': self.current_task.id if self.current_task else None,
				'status': self.status.value
			}
		})
		
		# Store as semantic memory
		memory = AgentMemory(
			agent_id=self.agent_id,
			memory_type="semantic",
			content=feedback,
			importance=feedback.get('importance', 5),
			tags=['feedback', 'learning']
		)
		await self._store_memory(memory)
		
		# Process feedback for learning
		if self.learning_engine:
			await self.learning_engine.process_feedback(feedback)
	
	# Learning Methods
	
	async def run_learning_session(self) -> Dict[str, Any]:
		"""Run a learning session to improve agent performance"""
		if not self.learning_engine:
			return {'status': 'not_available', 'message': 'Learning engine not initialized'}
		
		return await self.learning_engine.run_learning_session()
	
	async def create_learning_goal(self, goal_type: str, target_metric: str, target_value: float):
		"""Create a new learning goal"""
		if not self.learning_engine:
			return
		
		from .learning_engine import LearningGoal
		goal = LearningGoal(
			goal_type=goal_type,
			target_metric=target_metric,
			target_value=target_value,
			priority=7
		)
		await self.learning_engine.create_learning_goal(goal)
	
	def get_learning_status(self) -> Dict[str, Any]:
		"""Get current learning status"""
		if not self.learning_engine:
			return {'status': 'not_available'}
		
		return self.learning_engine.get_learning_status()
	
	async def configure_learning(self, config: Dict[str, Any]):
		"""Configure learning parameters"""
		if self.learning_engine:
			await self.learning_engine.set_learning_configuration(config)
	
	# Utility Methods
	
	def get_status_info(self) -> Dict[str, Any]:
		"""Get current agent status information"""
		return {
			'agent_id': self.agent_id,
			'role': self.role.value,
			'name': self.name,
			'status': self.status.value,
			'current_task': self.current_task.id if self.current_task else None,
			'queue_size': len(self.task_queue),
			'capabilities': [cap.name for cap in self.capabilities],
			'experience_points': self.experience_points,
			'performance_metrics': self.performance_metrics
		}
	
	async def shutdown(self):
		"""Gracefully shutdown the agent"""
		self.logger.info(f"Shutting down agent: {self.name}")
		
		# Complete current task if possible
		if self.current_task and self.status == AgentStatus.WORKING:
			self.logger.info("Attempting to complete current task before shutdown")
			# Give some time to finish
			await asyncio.sleep(5)
		
		# Save memory
		self._save_memory()
		
		# Update status
		self.status = AgentStatus.OFFLINE
		
		self.logger.info(f"Agent {self.name} shutdown complete")