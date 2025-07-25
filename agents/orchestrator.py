#!/usr/bin/env python3
"""
Agent Orchestrator
=================

Central orchestrator for managing and coordinating multiple APG agents.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid_extensions import uuid7str

from .base_agent import (
	BaseAgent, AgentRole, AgentStatus, AgentMessage, AgentTask, 
	MessageType, AgentCapability
)

class ProjectPhase:
	"""Project development phases"""
	ANALYSIS = "analysis"
	ARCHITECTURE = "architecture"
	DEVELOPMENT = "development"
	TESTING = "testing"
	DEPLOYMENT = "deployment"
	MONITORING = "monitoring"

class AgentOrchestrator:
	"""
	Central orchestrator for managing APG autonomous agents.
	
	Coordinates agent activities, manages communication, assigns tasks,
	and ensures project completion through collaborative multi-agent workflow.
	"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.orchestrator_id = uuid7str()
		
		# Agent management
		self.agents: Dict[str, BaseAgent] = {}
		self.agent_roles: Dict[AgentRole, List[str]] = {}
		
		# Task and project management
		self.active_projects: Dict[str, Dict[str, Any]] = {}
		self.task_queue: List[AgentTask] = []
		self.completed_tasks: List[AgentTask] = []
		
		# Communication
		self.message_bus: List[AgentMessage] = []
		self.message_handlers: Dict[MessageType, callable] = {}
		
		# Coordination
		self.collaboration_sessions: Dict[str, Dict[str, Any]] = {}
		self.resource_locks: Dict[str, str] = {}  # resource_id -> agent_id
		
		# Performance tracking
		self.performance_metrics: Dict[str, Any] = {}
		self.quality_metrics: Dict[str, Any] = {}
		
		# Logging
		self.logger = logging.getLogger("orchestrator")
		
		# Initialize orchestrator
		self._initialize()
	
	def _initialize(self):
		"""Initialize orchestrator components"""
		self.logger.info("Initializing Agent Orchestrator")
		
		# Setup message handlers
		self._setup_message_handlers()
		
		# Initialize performance tracking
		self.performance_metrics = {
			'projects_completed': 0,
			'tasks_completed': 0,
			'avg_project_duration': 0.0,
			'agent_utilization': {},
			'collaboration_efficiency': 0.0
		}
		
		# Start background tasks
		asyncio.create_task(self._process_messages())
		asyncio.create_task(self._monitor_agents())
		asyncio.create_task(self._optimize_workload())
		asyncio.create_task(self._manage_learning_sessions())
	
	def _setup_message_handlers(self):
		"""Setup message type handlers"""
		self.message_handlers = {
			MessageType.TASK_REQUEST: self._handle_task_request,
			MessageType.TASK_RESPONSE: self._handle_task_response,
			MessageType.COLLABORATION_REQUEST: self._handle_collaboration_request,
			MessageType.STATUS_UPDATE: self._handle_status_update,
			MessageType.QUESTION: self._handle_question,
			MessageType.ERROR: self._handle_error
		}
	
	# Agent Management
	
	async def register_agent(self, agent: BaseAgent) -> bool:
		"""
		Register a new agent with the orchestrator.
		
		Args:
			agent: The agent to register
			
		Returns:
			bool: True if registration successful
		"""
		if agent.agent_id in self.agents:
			self.logger.warning(f"Agent {agent.agent_id} already registered")
			return False
		
		self.agents[agent.agent_id] = agent
		
		# Add to role mapping
		if agent.role not in self.agent_roles:
			self.agent_roles[agent.role] = []
		self.agent_roles[agent.role].append(agent.agent_id)
		
		# Initialize performance tracking
		self.performance_metrics['agent_utilization'][agent.agent_id] = 0.0
		
		self.logger.info(f"Registered {agent.role.value} agent: {agent.name}")
		return True
	
	async def unregister_agent(self, agent_id: str) -> bool:
		"""
		Unregister an agent from the orchestrator.
		
		Args:
			agent_id: ID of the agent to unregister
			
		Returns:
			bool: True if unregistration successful
		"""
		if agent_id not in self.agents:
			return False
		
		agent = self.agents[agent_id]
		
		# Gracefully shutdown agent
		await agent.shutdown()
		
		# Remove from role mapping
		if agent.role in self.agent_roles:
			self.agent_roles[agent.role].remove(agent_id)
		
		# Remove from agents
		del self.agents[agent_id]
		
		# Clean up performance tracking
		if agent_id in self.performance_metrics['agent_utilization']:
			del self.performance_metrics['agent_utilization'][agent_id]
		
		self.logger.info(f"Unregistered agent: {agent.name}")
		return True
	
	def get_agents_by_role(self, role: AgentRole) -> List[BaseAgent]:
		"""Get all agents with a specific role"""
		if role not in self.agent_roles:
			return []
		
		return [self.agents[agent_id] for agent_id in self.agent_roles[role]]
	
	def get_available_agents(self, role: AgentRole = None) -> List[BaseAgent]:
		"""Get all available (idle) agents, optionally filtered by role"""
		available = []
		
		for agent in self.agents.values():
			if role is None or agent.role == role:
				if agent.status == AgentStatus.IDLE:
					available.append(agent)
		
		return available
	
	# Project Management
	
	async def start_project(
		self, 
		project_spec: Dict[str, Any],
		project_name: str = None
	) -> str:
		"""
		Start a new project with multi-agent collaboration.
		
		Args:
			project_spec: Project specification (APG AST or requirements)
			project_name: Optional project name
			
		Returns:
			str: Project ID
		"""
		project_id = uuid7str()
		project_name = project_name or f"Project_{project_id[:8]}"
		
		project = {
			'id': project_id,
			'name': project_name,
			'spec': project_spec,
			'status': 'started',
			'phase': ProjectPhase.ANALYSIS,
			'created_at': datetime.utcnow(),
			'agents_assigned': [],
			'tasks': [],
			'deliverables': {},
			'timeline': {},
			'quality_gates': []
		}
		
		self.active_projects[project_id] = project
		
		self.logger.info(f"Started project: {project_name} ({project_id})")
		
		# Begin project workflow
		await self._initiate_project_workflow(project_id)
		
		return project_id
	
	async def _initiate_project_workflow(self, project_id: str):
		"""Initiate the multi-agent project workflow"""
		project = self.active_projects[project_id]
		
		# Phase 1: Analysis - Architect analyzes requirements
		analysis_task = AgentTask(
			name=f"Analyze Requirements - {project['name']}",
			description="Analyze project requirements and create technical specification",
			requirements={
				'type': 'analysis',
				'capabilities': ['requirement_analysis', 'architecture_design'],
				'project_spec': project['spec']
			},
			deliverables=[
				'technical_specification',
				'architecture_overview',
				'technology_stack',
				'development_phases'
			],
			priority=9,
			context={
				'project_id': project_id,
				'phase': ProjectPhase.ANALYSIS
			}
		)
		
		# Assign to architect
		architect_agents = self.get_available_agents(AgentRole.ARCHITECT)
		if architect_agents:
			architect = architect_agents[0]  # Use first available architect
			await self._assign_task_to_agent(analysis_task, architect.agent_id)
			project['agents_assigned'].append(architect.agent_id)
		else:
			self.logger.error(f"No architect available for project {project_id}")
			project['status'] = 'error'
	
	async def _assign_task_to_agent(self, task: AgentTask, agent_id: str) -> bool:
		"""Assign a task to a specific agent"""
		if agent_id not in self.agents:
			return False
		
		agent = self.agents[agent_id]
		task.assigned_agent = agent_id
		
		success = await agent.receive_task(task)
		if success:
			self.task_queue.append(task)
			self.logger.info(f"Assigned task '{task.name}' to {agent.name}")
		
		return success
	
	async def _continue_project_workflow(self, project_id: str, completed_task: AgentTask):
		"""Continue project workflow after task completion"""
		project = self.active_projects[project_id]
		
		# Update project with task results
		project['deliverables'][completed_task.id] = completed_task.results
		
		# Determine next phase based on current phase and completed task
		current_phase = project['phase']
		
		if current_phase == ProjectPhase.ANALYSIS:
			await self._start_architecture_phase(project_id, completed_task)
		elif current_phase == ProjectPhase.ARCHITECTURE:
			await self._start_development_phase(project_id, completed_task)
		elif current_phase == ProjectPhase.DEVELOPMENT:
			await self._start_testing_phase(project_id, completed_task)
		elif current_phase == ProjectPhase.TESTING:
			await self._start_deployment_phase(project_id, completed_task)
		elif current_phase == ProjectPhase.DEPLOYMENT:
			await self._complete_project(project_id, completed_task)
	
	async def _start_architecture_phase(self, project_id: str, analysis_task: AgentTask):
		"""Start the architecture design phase"""
		project = self.active_projects[project_id]
		project['phase'] = ProjectPhase.ARCHITECTURE
		
		# Create architecture tasks based on analysis results
		analysis_results = analysis_task.results
		
		architecture_task = AgentTask(
			name=f"Design Architecture - {project['name']}",
			description="Design detailed system architecture and component specifications",
			requirements={
				'type': 'architecture',
				'capabilities': ['architecture_design', 'system_design'],
				'analysis_results': analysis_results
			},
			deliverables=[
				'system_architecture',
				'component_specifications',
				'api_specifications',
				'database_schema',
				'deployment_architecture'
			],
			priority=8,
			context={
				'project_id': project_id,
				'phase': ProjectPhase.ARCHITECTURE,
				'depends_on': [analysis_task.id]
			}
		)
		
		# Assign to architect (could be same or different)
		architect_agents = self.get_available_agents(AgentRole.ARCHITECT)
		if architect_agents:
			await self._assign_task_to_agent(architecture_task, architect_agents[0].agent_id)
	
	async def _start_development_phase(self, project_id: str, architecture_task: AgentTask):
		"""Start the development phase with parallel development tasks"""
		project = self.active_projects[project_id]
		project['phase'] = ProjectPhase.DEVELOPMENT
		
		architecture_results = architecture_task.results
		components = architecture_results.get('components', [])
		
		# Create development tasks for each component
		development_tasks = []
		
		for component in components:
			dev_task = AgentTask(
				name=f"Develop {component['name']} - {project['name']}",
				description=f"Implement {component['name']} component",
				requirements={
					'type': 'development',
					'capabilities': ['code_generation', 'software_development'],
					'component_spec': component,
					'architecture': architecture_results
				},
				deliverables=[
					'source_code',
					'unit_tests',
					'integration_tests',
					'documentation'
				],
				priority=7,
				context={
					'project_id': project_id,
					'phase': ProjectPhase.DEVELOPMENT,
					'component': component['name'],
					'depends_on': [architecture_task.id]
				}
			)
			development_tasks.append(dev_task)
		
		# Assign tasks to available developers
		developer_agents = self.get_available_agents(AgentRole.DEVELOPER)
		
		for i, task in enumerate(development_tasks):
			if i < len(developer_agents):
				await self._assign_task_to_agent(task, developer_agents[i].agent_id)
			else:
				# Queue for when developers become available
				self.task_queue.append(task)
	
	async def _start_testing_phase(self, project_id: str, completed_task: AgentTask):
		"""Start the testing phase"""
		project = self.active_projects[project_id]
		
		# Check if all development tasks are complete
		dev_tasks = [t for t in project['tasks'] if t.context.get('phase') == ProjectPhase.DEVELOPMENT]
		completed_dev_tasks = [t for t in dev_tasks if t.status == 'completed']
		
		if len(completed_dev_tasks) < len(dev_tasks):
			return  # Wait for all development tasks to complete
		
		project['phase'] = ProjectPhase.TESTING
		
		# Create comprehensive testing task
		testing_task = AgentTask(
			name=f"Test Application - {project['name']}",
			description="Perform comprehensive testing of the application",
			requirements={
				'type': 'testing',
				'capabilities': ['automated_testing', 'quality_assurance'],
				'development_results': [t.results for t in completed_dev_tasks]
			},
			deliverables=[
				'test_results',
				'quality_report',
				'performance_metrics',
				'security_assessment'
			],
			priority=8,
			context={
				'project_id': project_id,
				'phase': ProjectPhase.TESTING,
				'depends_on': [t.id for t in completed_dev_tasks]
			}
		)
		
		# Assign to tester
		tester_agents = self.get_available_agents(AgentRole.TESTER)
		if tester_agents:
			await self._assign_task_to_agent(testing_task, tester_agents[0].agent_id)
	
	async def _start_deployment_phase(self, project_id: str, testing_task: AgentTask):
		"""Start the deployment phase"""
		project = self.active_projects[project_id]
		project['phase'] = ProjectPhase.DEPLOYMENT
		
		testing_results = testing_task.results
		
		deployment_task = AgentTask(
			name=f"Deploy Application - {project['name']}",
			description="Deploy application to target environment",
			requirements={
				'type': 'deployment',
				'capabilities': ['deployment', 'devops', 'infrastructure'],
				'testing_results': testing_results,
				'architecture': project['deliverables']
			},
			deliverables=[
				'deployment_package',
				'infrastructure_config',
				'monitoring_setup',
				'deployment_report'
			],
			priority=9,
			context={
				'project_id': project_id,
				'phase': ProjectPhase.DEPLOYMENT,
				'depends_on': [testing_task.id]
			}
		)
		
		# Assign to DevOps agent
		devops_agents = self.get_available_agents(AgentRole.DEVOPS)
		if devops_agents:
			await self._assign_task_to_agent(deployment_task, devops_agents[0].agent_id)
	
	async def _complete_project(self, project_id: str, deployment_task: AgentTask):
		"""Complete the project"""
		project = self.active_projects[project_id]
		project['status'] = 'completed'
		project['completed_at'] = datetime.utcnow()
		project['final_deliverables'] = deployment_task.results
		
		# Calculate project metrics
		duration = (project['completed_at'] - project['created_at']).total_seconds()
		project['duration'] = duration
		
		# Update performance metrics
		self.performance_metrics['projects_completed'] += 1
		
		# Calculate average project duration
		total_projects = self.performance_metrics['projects_completed']
		current_avg = self.performance_metrics['avg_project_duration']
		self.performance_metrics['avg_project_duration'] = (
			(current_avg * (total_projects - 1) + duration) / total_projects
		)
		
		self.logger.info(f"Project completed: {project['name']} (Duration: {duration/3600:.2f} hours)")
		
		# Move to completed projects (optional archival)
		# del self.active_projects[project_id]
	
	# Communication Management
	
	async def send_message(self, message: AgentMessage) -> bool:
		"""Send a message through the orchestrator message bus"""
		message.timestamp = datetime.utcnow()
		self.message_bus.append(message)
		
		# Deliver to recipient
		if message.recipient_id in self.agents:
			await self.agents[message.recipient_id].receive_message(message)
			return True
		
		return False
	
	async def _process_messages(self):
		"""Background task to process messages"""
		while True:
			try:
				# Process messages that need orchestrator handling
				for message in self.message_bus[:]:
					if message.recipient_id == self.orchestrator_id:
						await self._handle_orchestrator_message(message)
						self.message_bus.remove(message)
				
				await asyncio.sleep(0.1)  # Short polling interval
			except Exception as e:
				self.logger.error(f"Error processing messages: {e}")
	
	async def _handle_orchestrator_message(self, message: AgentMessage):
		"""Handle messages directed to the orchestrator"""
		handler = self.message_handlers.get(message.message_type)
		if handler:
			await handler(message)
		else:
			self.logger.warning(f"No handler for message type: {message.message_type}")
	
	async def _handle_task_request(self, message: AgentMessage):
		"""Handle task request from an agent"""
		# Agent requesting help with a task
		requesting_agent_id = message.sender_id
		task_requirements = message.content.get('requirements', {})
		
		# Find suitable agent for collaboration
		suitable_agents = self._find_suitable_agents(task_requirements)
		
		if suitable_agents:
			# Create collaboration session
			collaboration_id = uuid7str()
			self.collaboration_sessions[collaboration_id] = {
				'id': collaboration_id,
				'requester': requesting_agent_id,
				'collaborators': suitable_agents,
				'task_requirements': task_requirements,
				'status': 'active',
				'created_at': datetime.utcnow()
			}
			
			# Notify collaborators
			for agent_id in suitable_agents:
				collab_message = AgentMessage(
					recipient_id=agent_id,
					message_type=MessageType.COLLABORATION_REQUEST,
					content={
						'collaboration_id': collaboration_id,
						'requester': requesting_agent_id,
						'requirements': task_requirements
					}
				)
				await self.send_message(collab_message)
	
	def _find_suitable_agents(self, requirements: Dict[str, Any]) -> List[str]:
		"""Find agents suitable for collaboration based on requirements"""
		required_capabilities = requirements.get('capabilities', [])
		required_role = requirements.get('role')
		
		suitable_agents = []
		
		for agent in self.agents.values():
			if agent.status != AgentStatus.IDLE:
				continue
			
			if required_role and agent.role.value != required_role:
				continue
			
			# Check if agent has required capabilities
			agent_capabilities = [cap.name for cap in agent.capabilities]
			if all(cap in agent_capabilities for cap in required_capabilities):
				suitable_agents.append(agent.agent_id)
		
		return suitable_agents
	
	async def _handle_task_response(self, message: AgentMessage):
		"""Handle task completion response"""
		task_id = message.content.get('task_id')
		status = message.content.get('status')
		results = message.content.get('results', {})
		
		# Find the completed task
		completed_task = None
		for task in self.task_queue:
			if task.id == task_id:
				completed_task = task
				break
		
		if completed_task:
			completed_task.status = status
			completed_task.results = results
			completed_task.completed_at = datetime.utcnow()
			
			# Move to completed tasks
			self.task_queue.remove(completed_task)
			self.completed_tasks.append(completed_task)
			
			# Continue project workflow if this is a project task
			project_id = completed_task.context.get('project_id')
			if project_id and project_id in self.active_projects:
				await self._continue_project_workflow(project_id, completed_task)
			
			# Update performance metrics
			self.performance_metrics['tasks_completed'] += 1
			
			# Update agent utilization
			agent_id = completed_task.assigned_agent
			if agent_id in self.performance_metrics['agent_utilization']:
				current_util = self.performance_metrics['agent_utilization'][agent_id]
				self.performance_metrics['agent_utilization'][agent_id] = min(1.0, current_util + 0.1)
	
	async def _handle_collaboration_request(self, message: AgentMessage):
		"""Handle collaboration request between agents"""
		# Facilitate collaboration between agents
		pass
	
	async def _handle_status_update(self, message: AgentMessage):
		"""Handle agent status updates"""
		agent_id = message.sender_id
		status_info = message.content
		
		# Update agent tracking
		if agent_id in self.agents:
			# Log status change
			self.logger.debug(f"Agent {agent_id} status update: {status_info}")
	
	async def _handle_question(self, message: AgentMessage):
		"""Handle questions directed to orchestrator"""
		question = message.content.get('question', '')
		context = message.content.get('context', {})
		
		# Answer based on orchestrator knowledge
		answer = await self._answer_orchestrator_question(question, context)
		
		response = AgentMessage(
			recipient_id=message.sender_id,
			message_type=MessageType.ANSWER,
			content={'answer': answer},
			parent_message_id=message.id
		)
		await self.send_message(response)
	
	async def _answer_orchestrator_question(self, question: str, context: Dict[str, Any]) -> str:
		"""Answer questions about projects, agents, or system state"""
		question_lower = question.lower()
		
		if 'project' in question_lower and 'status' in question_lower:
			# Project status question
			project_id = context.get('project_id')
			if project_id and project_id in self.active_projects:
				project = self.active_projects[project_id]
				return f"Project {project['name']} is in {project['phase']} phase with status {project['status']}"
		
		elif 'agents' in question_lower and 'available' in question_lower:
			# Available agents question
			available = self.get_available_agents()
			return f"Currently {len(available)} agents are available: {[a.name for a in available]}"
		
		elif 'performance' in question_lower:
			# Performance metrics question
			return f"System performance: {self.performance_metrics['projects_completed']} projects completed, avg duration {self.performance_metrics['avg_project_duration']/3600:.2f} hours"
		
		return "I need more specific information to answer that question."
	
	async def _handle_error(self, message: AgentMessage):
		"""Handle error messages from agents"""
		error_info = message.content
		agent_id = message.sender_id
		
		self.logger.error(f"Agent {agent_id} reported error: {error_info}")
		
		# Take corrective action if needed
		# Could involve reassigning tasks, restarting agents, etc.
	
	# Monitoring and Optimization
	
	async def _monitor_agents(self):
		"""Background task to monitor agent health and performance"""
		while True:
			try:
				for agent in self.agents.values():
					# Check agent health
					if agent.status == AgentStatus.ERROR:
						self.logger.warning(f"Agent {agent.name} is in error state")
						# Could implement recovery logic
					
					# Update utilization metrics
					if agent.status == AgentStatus.WORKING:
						agent_id = agent.agent_id
						if agent_id in self.performance_metrics['agent_utilization']:
							self.performance_metrics['agent_utilization'][agent_id] = min(
								1.0, 
								self.performance_metrics['agent_utilization'][agent_id] + 0.01
							)
				
				await asyncio.sleep(30)  # Monitor every 30 seconds
			except Exception as e:
				self.logger.error(f"Error monitoring agents: {e}")
	
	async def _optimize_workload(self):
		"""Background task to optimize workload distribution"""
		while True:
			try:
				# Load balancing logic
				await self._balance_workload()
				
				# Task priority optimization
				await self._optimize_task_priorities()
				
				await asyncio.sleep(60)  # Optimize every minute
			except Exception as e:
				self.logger.error(f"Error optimizing workload: {e}")
	
	async def _balance_workload(self):
		"""Balance workload across available agents"""
		# Find overloaded and underutilized agents
		overloaded = []
		underutilized = []
		
		for agent_id, utilization in self.performance_metrics['agent_utilization'].items():
			if utilization > 0.8:
				overloaded.append(agent_id)
			elif utilization < 0.3:
				underutilized.append(agent_id)
		
		# Rebalance if needed
		if overloaded and underutilized:
			self.logger.info(f"Rebalancing workload: {len(overloaded)} overloaded, {len(underutilized)} underutilized")
			# Implement task redistribution logic
	
	async def _optimize_task_priorities(self):
		"""Optimize task priorities based on project deadlines and dependencies"""
		# Sort tasks by project priority and dependencies
		self.task_queue.sort(key=lambda t: (
			t.priority,
			-len(t.dependencies),  # Fewer dependencies = higher priority
			t.created_at  # Earlier tasks first
		))
	
	async def _manage_learning_sessions(self):
		"""Background task to manage agent learning sessions"""
		while True:
			try:
				# Run learning sessions for agents periodically
				for agent in self.agents.values():
					if hasattr(agent, 'learning_engine') and agent.learning_engine:
						learning_status = agent.get_learning_status()
						
						# Check if it's time for a learning session
						if self._should_run_learning_session(agent, learning_status):
							self.logger.info(f"Starting learning session for {agent.name}")
							
							try:
								session_results = await agent.run_learning_session()
								self._process_learning_results(agent, session_results)
							except Exception as e:
								self.logger.error(f"Learning session failed for {agent.name}: {e}")
				
				# Create system-wide learning goals
				await self._create_system_learning_goals()
				
				await asyncio.sleep(3600)  # Run every hour
			except Exception as e:
				self.logger.error(f"Error managing learning sessions: {e}")
	
	def _should_run_learning_session(self, agent: BaseAgent, learning_status: Dict[str, Any]) -> bool:
		"""Determine if agent should run a learning session"""
		if not learning_status.get('learning_enabled', False):
			return False
		
		# Check if enough time has passed since last session
		last_session = datetime.fromisoformat(learning_status.get('last_session', '2000-01-01T00:00:00'))
		next_session = datetime.fromisoformat(learning_status.get('next_session', '2000-01-01T00:00:00'))
		
		if datetime.utcnow() < next_session:
			return False
		
		# Check if agent has sufficient events to learn from
		recent_performance = learning_status.get('recent_performance', {})
		total_tasks = recent_performance.get('total_tasks', 0)
		
		return total_tasks >= 3  # Minimum tasks to trigger learning
	
	def _process_learning_results(self, agent: BaseAgent, session_results: Dict[str, Any]):
		"""Process results from an agent learning session"""
		if session_results.get('status') == 'completed':
			# Update agent utilization based on learning improvements
			agent_id = agent.agent_id
			improvements = session_results.get('improvements_identified', [])
			
			if improvements and agent_id in self.performance_metrics['agent_utilization']:
				# Small boost for successful learning
				current_util = self.performance_metrics['agent_utilization'][agent_id]
				self.performance_metrics['agent_utilization'][agent_id] = min(1.0, current_util + 0.02)
			
			# Log significant improvements
			if len(improvements) > 0:
				self.logger.info(f"Agent {agent.name} identified {len(improvements)} improvements")
	
	async def _create_system_learning_goals(self):
		"""Create system-wide learning goals for agents"""
		# Analyze system performance and create collaborative learning goals
		if len(self.active_projects) > 0:
			# Goal: Improve project completion time
			avg_duration = self.performance_metrics.get('avg_project_duration', 0)
			if avg_duration > 0:
				target_reduction = avg_duration * 0.9  # 10% improvement goal
				
				for agent in self.agents.values():
					if hasattr(agent, 'create_learning_goal'):
						await agent.create_learning_goal(
							'performance_optimization',
							'project_duration',
							target_reduction
						)
			
			# Goal: Improve collaboration efficiency
			collaboration_efficiency = self.performance_metrics.get('collaboration_efficiency', 0.5)
			if collaboration_efficiency < 0.8:
				target_efficiency = min(1.0, collaboration_efficiency + 0.1)
				
				for agent in self.agents.values():
					if hasattr(agent, 'create_learning_goal'):
						await agent.create_learning_goal(
							'collaboration_improvement',
							'collaboration_efficiency',
							target_efficiency
						)
	
	# Utility Methods
	
	def get_system_status(self) -> Dict[str, Any]:
		"""Get comprehensive system status"""
		return {
			'orchestrator_id': self.orchestrator_id,
			'agents': {
				'total': len(self.agents),
				'by_role': {role.value: len(agents) for role, agents in self.agent_roles.items()},
				'by_status': self._get_agents_by_status()
			},
			'projects': {
				'active': len(self.active_projects),
				'phases': self._get_projects_by_phase()
			},
			'tasks': {
				'queued': len(self.task_queue),
				'completed': len(self.completed_tasks)
			},
			'performance': self.performance_metrics,
			'collaboration_sessions': len(self.collaboration_sessions)
		}
	
	def _get_agents_by_status(self) -> Dict[str, int]:
		"""Get agent count by status"""
		status_counts = {}
		for agent in self.agents.values():
			status = agent.status.value
			status_counts[status] = status_counts.get(status, 0) + 1
		return status_counts
	
	def _get_projects_by_phase(self) -> Dict[str, int]:
		"""Get project count by phase"""
		phase_counts = {}
		for project in self.active_projects.values():
			phase = project['phase']
			phase_counts[phase] = phase_counts.get(phase, 0) + 1
		return phase_counts
	
	async def shutdown(self):
		"""Gracefully shutdown the orchestrator and all agents"""
		self.logger.info("Shutting down Agent Orchestrator")
		
		# Shutdown all agents
		for agent in self.agents.values():
			await agent.shutdown()
		
		# Clear state
		self.agents.clear()
		self.agent_roles.clear()
		self.active_projects.clear()
		
		self.logger.info("Agent Orchestrator shutdown complete")