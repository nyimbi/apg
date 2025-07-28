"""
APG Security Orchestration - Core Service

Enterprise security orchestration service with intelligent workflow automation,
multi-system integration, and adaptive response coordination.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import aiohttp
import paramiko
from celery import Celery
from jinja2 import Template
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
	SecurityPlaybook, WorkflowExecution, AutomationAction, ToolIntegration,
	ResponseCoordination, OrchestrationMetrics, ApprovalWorkflow, WorkflowTemplate,
	PlaybookType, WorkflowStatus, ActionType, ExecutionMode, TriggerType
)


class SecurityOrchestrationService:
	"""Core security orchestration and automated response service"""
	
	def __init__(self, db_session: AsyncSession, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(__name__)
		
		self._playbooks_cache = {}
		self._active_executions = {}
		self._tool_integrations = {}
		self._workflow_engine = None
		self._celery_app = None
		
		asyncio.create_task(self._initialize_service())
	
	async def _initialize_service(self):
		"""Initialize security orchestration service"""
		try:
			await self._load_security_playbooks()
			await self._initialize_workflow_engine()
			await self._load_tool_integrations()
			await self._setup_celery_tasks()
			await self._start_execution_monitor()
			
			self.logger.info(f"Security orchestration service initialized for tenant {self.tenant_id}")
		except Exception as e:
			self.logger.error(f"Failed to initialize security orchestration service: {str(e)}")
			raise
	
	async def create_security_playbook(self, playbook_data: Dict[str, Any]) -> SecurityPlaybook:
		"""Create comprehensive security playbook"""
		try:
			playbook = SecurityPlaybook(
				tenant_id=self.tenant_id,
				**playbook_data
			)
			
			# Validate playbook workflow
			validation_result = await self._validate_playbook_workflow(playbook)
			if not validation_result['valid']:
				raise ValueError(f"Invalid playbook workflow: {validation_result['errors']}")
			
			playbook.is_validated = True
			playbook.validation_results = validation_result
			
			# Set up retry policy if not provided
			if not playbook.retry_policy:
				playbook.retry_policy = {
					'max_retries': 3,
					'retry_delay': 60,
					'backoff_factor': 2
				}
			
			# Set up error handling if not provided
			if not playbook.error_handling:
				playbook.error_handling = {
					'continue_on_error': False,
					'notification_on_error': True,
					'rollback_on_failure': True
				}
			
			await self._store_security_playbook(playbook)
			
			# Cache the playbook
			self._playbooks_cache[playbook.id] = playbook
			
			# Test playbook if requested
			if playbook_data.get('test_after_creation', False):
				test_result = await self._test_playbook(playbook.id)
				playbook.last_tested = datetime.utcnow()
				await self._update_security_playbook(playbook)
			
			return playbook
			
		except Exception as e:
			self.logger.error(f"Error creating security playbook: {str(e)}")
			raise
	
	async def execute_security_playbook(self, playbook_id: str, 
									  execution_params: Dict[str, Any],
									  trigger_data: Dict[str, Any] = None) -> WorkflowExecution:
		"""Execute security playbook with comprehensive orchestration"""
		try:
			playbook = await self._get_security_playbook(playbook_id)
			if not playbook:
				raise ValueError(f"Playbook {playbook_id} not found")
			
			if not playbook.is_active:
				raise ValueError(f"Playbook {playbook_id} is not active")
			
			# Create execution instance
			execution = WorkflowExecution(
				tenant_id=self.tenant_id,
				playbook_id=playbook_id,
				playbook_version=playbook.version,
				execution_name=execution_params.get('execution_name', f"{playbook.name}_execution"),
				trigger_type=execution_params.get('trigger_type', TriggerType.MANUAL),
				trigger_data=trigger_data or {},
				triggered_by=execution_params.get('triggered_by', 'system'),
				input_parameters=execution_params.get('input_parameters', {}),
				execution_context=execution_params.get('execution_context', {})
			)
			
			# Check if approval is required
			if playbook.requires_approval:
				execution.approval_required = True
				approval_workflow = await self._create_approval_workflow(execution, playbook)
				execution.status = WorkflowStatus.PENDING
				await self._store_workflow_execution(execution)
				
				# Wait for approval or timeout
				approval_result = await self._wait_for_approval(approval_workflow)
				if approval_result['decision'] != 'approved':
					execution.status = WorkflowStatus.CANCELLED
					execution.end_time = datetime.utcnow()
					await self._update_workflow_execution(execution)
					return execution
			
			# Start execution
			execution.status = WorkflowStatus.RUNNING
			execution.start_time = datetime.utcnow()
			await self._store_workflow_execution(execution)
			
			# Add to active executions
			self._active_executions[execution.id] = execution
			
			# Execute workflow asynchronously
			asyncio.create_task(self._execute_workflow_async(execution, playbook))
			
			return execution
			
		except Exception as e:
			self.logger.error(f"Error executing security playbook: {str(e)}")
			raise
	
	async def _execute_workflow_async(self, execution: WorkflowExecution, playbook: SecurityPlaybook):
		"""Execute workflow asynchronously with error handling"""
		try:
			# Parse workflow definition
			workflow_steps = await self._parse_workflow_definition(playbook.workflow_definition)
			
			# Execute based on execution mode
			if playbook.execution_mode == ExecutionMode.SEQUENTIAL:
				await self._execute_sequential_workflow(execution, workflow_steps)
			elif playbook.execution_mode == ExecutionMode.PARALLEL:
				await self._execute_parallel_workflow(execution, workflow_steps)
			elif playbook.execution_mode == ExecutionMode.CONDITIONAL:
				await self._execute_conditional_workflow(execution, workflow_steps)
			else:
				await self._execute_sequential_workflow(execution, workflow_steps)
			
			# Mark execution as completed
			execution.status = WorkflowStatus.COMPLETED
			execution.end_time = datetime.utcnow()
			execution.execution_duration = execution.end_time - execution.start_time
			execution.progress_percentage = Decimal('100.0')
			
			await self._update_workflow_execution(execution)
			
			# Update playbook statistics
			playbook.execution_count += 1
			playbook.success_count += 1
			
			if playbook.average_execution_time:
				playbook.average_execution_time = (
					playbook.average_execution_time + execution.execution_duration
				) / 2
			else:
				playbook.average_execution_time = execution.execution_duration
			
			await self._update_security_playbook(playbook)
			
			# Remove from active executions
			self._active_executions.pop(execution.id, None)
			
			self.logger.info(f"Workflow execution {execution.id} completed successfully")
			
		except Exception as e:
			# Handle execution failure
			execution.status = WorkflowStatus.FAILED
			execution.end_time = datetime.utcnow()
			execution.error_details = {'error': str(e), 'type': type(e).__name__}
			execution.error_message = str(e)
			
			await self._update_workflow_execution(execution)
			
			# Execute rollback if configured
			if playbook.error_handling.get('rollback_on_failure', False):
				await self._execute_rollback_actions(execution, playbook)
			
			# Update playbook failure statistics
			playbook.execution_count += 1
			playbook.failure_count += 1
			await self._update_security_playbook(playbook)
			
			# Remove from active executions
			self._active_executions.pop(execution.id, None)
			
			self.logger.error(f"Workflow execution {execution.id} failed: {str(e)}")
	
	async def _execute_sequential_workflow(self, execution: WorkflowExecution, workflow_steps: List[Dict]):
		"""Execute workflow steps sequentially"""
		try:
			total_steps = len(workflow_steps)
			
			for i, step in enumerate(workflow_steps):
				# Update current step
				execution.current_step = step['name']
				execution.progress_percentage = Decimal(str((i / total_steps) * 100))
				await self._update_workflow_execution(execution)
				
				# Execute step
				step_result = await self._execute_workflow_step(execution, step)
				
				if step_result['success']:
					execution.completed_steps.append(step['name'])
					execution.execution_results.append(step_result)
				else:
					execution.failed_steps.append(step['name'])
					execution.execution_results.append(step_result)
					
					# Handle step failure
					if not step.get('continue_on_failure', False):
						raise Exception(f"Step {step['name']} failed: {step_result.get('error', 'Unknown error')}")
				
				await self._update_workflow_execution(execution)
			
		except Exception as e:
			self.logger.error(f"Error executing sequential workflow: {str(e)}")
			raise
	
	async def _execute_parallel_workflow(self, execution: WorkflowExecution, workflow_steps: List[Dict]):
		"""Execute workflow steps in parallel"""
		try:
			# Create tasks for all steps
			tasks = []
			for step in workflow_steps:
				task = asyncio.create_task(self._execute_workflow_step(execution, step))
				tasks.append((step['name'], task))
			
			# Wait for all tasks to complete
			completed_steps = []
			failed_steps = []
			
			for step_name, task in tasks:
				try:
					result = await task
					if result['success']:
						completed_steps.append(step_name)
						execution.execution_results.append(result)
					else:
						failed_steps.append(step_name)
						execution.execution_results.append(result)
				except Exception as e:
					failed_steps.append(step_name)
					execution.execution_results.append({
						'step_name': step_name,
						'success': False,
						'error': str(e)
					})
			
			execution.completed_steps.extend(completed_steps)
			execution.failed_steps.extend(failed_steps)
			execution.progress_percentage = Decimal('100.0')
			
			await self._update_workflow_execution(execution)
			
			# Check if any critical steps failed
			if failed_steps:
				critical_failures = [s for s in workflow_steps 
								   if s['name'] in failed_steps and s.get('critical', False)]
				if critical_failures:
					raise Exception(f"Critical steps failed: {[s['name'] for s in critical_failures]}")
			
		except Exception as e:
			self.logger.error(f"Error executing parallel workflow: {str(e)}")
			raise
	
	async def _execute_workflow_step(self, execution: WorkflowExecution, step: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute individual workflow step"""
		try:
			step_name = step['name']
			action_type = ActionType(step['type'])
			
			# Create automation action record
			action = AutomationAction(
				tenant_id=self.tenant_id,
				execution_id=execution.id,
				action_name=step_name,
				action_type=action_type,
				action_config=step.get('config', {}),
				input_data=step.get('input', {}),
				step_number=step.get('order', 1)
			)
			
			action.status = WorkflowStatus.RUNNING
			action.start_time = datetime.utcnow()
			await self._store_automation_action(action)
			
			# Execute based on action type
			if action_type == ActionType.HTTP_REQUEST:
				result = await self._execute_http_action(action)
			elif action_type == ActionType.EMAIL_NOTIFICATION:
				result = await self._execute_email_action(action)
			elif action_type == ActionType.SECURITY_TOOL:
				result = await self._execute_security_tool_action(action)
			elif action_type == ActionType.SYSTEM_COMMAND:
				result = await self._execute_system_command_action(action)
			elif action_type == ActionType.CUSTOM_SCRIPT:
				result = await self._execute_custom_script_action(action)
			else:
				result = {'success': False, 'error': f'Unsupported action type: {action_type}'}
			
			# Update action with results
			action.end_time = datetime.utcnow()
			action.execution_time = action.end_time - action.start_time
			action.success = result['success']
			action.output_data = result.get('output', {})
			
			if result['success']:
				action.status = WorkflowStatus.COMPLETED
			else:
				action.status = WorkflowStatus.FAILED
				action.error_message = result.get('error', 'Unknown error')
			
			await self._update_automation_action(action)
			
			return {
				'step_name': step_name,
				'action_id': action.id,
				'success': result['success'],
				'output': result.get('output', {}),
				'error': result.get('error'),
				'execution_time': action.execution_time.total_seconds() if action.execution_time else 0
			}
			
		except Exception as e:
			self.logger.error(f"Error executing workflow step {step.get('name', 'unknown')}: {str(e)}")
			return {
				'step_name': step.get('name', 'unknown'),
				'success': False,
				'error': str(e)
			}
	
	async def _execute_http_action(self, action: AutomationAction) -> Dict[str, Any]:
		"""Execute HTTP request action"""
		try:
			config = action.action_config
			url = config['url']
			method = config.get('method', 'GET').upper()
			headers = config.get('headers', {})
			data = config.get('data', {})
			timeout = config.get('timeout', 30)
			
			async with aiohttp.ClientSession() as session:
				if method == 'GET':
					async with session.get(url, headers=headers, timeout=timeout) as response:
						result_data = await response.text()
						return {
							'success': response.status < 400,
							'output': {
								'status_code': response.status,
								'response_text': result_data,
								'headers': dict(response.headers)
							}
						}
				elif method == 'POST':
					async with session.post(url, headers=headers, json=data, timeout=timeout) as response:
						result_data = await response.text()
						return {
							'success': response.status < 400,
							'output': {
								'status_code': response.status,
								'response_text': result_data,
								'headers': dict(response.headers)
							}
						}
				else:
					return {'success': False, 'error': f'Unsupported HTTP method: {method}'}
			
		except Exception as e:
			return {'success': False, 'error': str(e)}
	
	async def _execute_email_action(self, action: AutomationAction) -> Dict[str, Any]:
		"""Execute email notification action"""
		try:
			config = action.action_config
			# Placeholder implementation for email sending
			# In real implementation, this would integrate with email service
			
			return {
				'success': True,
				'output': {
					'email_sent': True,
					'recipients': config.get('recipients', []),
					'subject': config.get('subject', 'Security Alert'),
					'timestamp': datetime.utcnow().isoformat()
				}
			}
			
		except Exception as e:
			return {'success': False, 'error': str(e)}
	
	async def _execute_security_tool_action(self, action: AutomationAction) -> Dict[str, Any]:
		"""Execute security tool integration action"""
		try:
			config = action.action_config
			tool_name = config['tool_name']
			
			# Get tool integration
			tool_integration = await self._get_tool_integration(tool_name)
			if not tool_integration:
				return {'success': False, 'error': f'Tool integration {tool_name} not found'}
			
			# Execute tool-specific action
			if tool_name.lower() in ['splunk', 'elasticsearch']:
				return await self._execute_siem_action(tool_integration, config)
			elif tool_name.lower() in ['nessus', 'qualys']:
				return await self._execute_vulnerability_scanner_action(tool_integration, config)
			elif tool_name.lower() in ['crowdstrike', 'sentinelone']:
				return await self._execute_edr_action(tool_integration, config)
			else:
				return await self._execute_generic_tool_action(tool_integration, config)
			
		except Exception as e:
			return {'success': False, 'error': str(e)}
	
	async def create_tool_integration(self, integration_data: Dict[str, Any]) -> ToolIntegration:
		"""Create security tool integration"""
		try:
			integration = ToolIntegration(
				tenant_id=self.tenant_id,
				**integration_data
			)
			
			# Test integration connectivity
			test_result = await self._test_tool_connectivity(integration)
			integration.is_healthy = test_result['healthy']
			integration.test_results = test_result
			integration.last_tested = datetime.utcnow()
			
			await self._store_tool_integration(integration)
			
			# Cache the integration
			self._tool_integrations[integration.tool_name] = integration
			
			return integration
			
		except Exception as e:
			self.logger.error(f"Error creating tool integration: {str(e)}")
			raise
	
	async def coordinate_multi_team_response(self, coordination_data: Dict[str, Any]) -> ResponseCoordination:
		"""Coordinate multi-team security response"""
		try:
			coordination = ResponseCoordination(
				tenant_id=self.tenant_id,
				**coordination_data
			)
			
			# Initialize coordination workflow
			coordination.start_time = datetime.utcnow()
			coordination.overall_status = WorkflowStatus.RUNNING
			
			# Set up communication channels
			await self._setup_coordination_channels(coordination)
			
			# Assign initial tasks to teams
			await self._assign_team_tasks(coordination)
			
			# Start coordination monitoring
			asyncio.create_task(self._monitor_coordination_progress(coordination))
			
			await self._store_response_coordination(coordination)
			
			return coordination
			
		except Exception as e:
			self.logger.error(f"Error coordinating multi-team response: {str(e)}")
			raise
	
	async def generate_orchestration_metrics(self, period_days: int = 30) -> OrchestrationMetrics:
		"""Generate security orchestration metrics"""
		try:
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=period_days)
			
			metrics = OrchestrationMetrics(
				tenant_id=self.tenant_id,
				metric_period_start=start_time,
				metric_period_end=end_time
			)
			
			# Playbook metrics
			all_playbooks = await self._get_all_playbooks()
			metrics.total_playbooks = len(all_playbooks)
			metrics.active_playbooks = len([p for p in all_playbooks if p.is_active])
			
			# Execution metrics
			executions = await self._get_executions_in_period(start_time, end_time)
			metrics.executed_playbooks = len(executions)
			metrics.successful_executions = len([e for e in executions if e.status == WorkflowStatus.COMPLETED])
			metrics.failed_executions = len([e for e in executions if e.status == WorkflowStatus.FAILED])
			
			if metrics.executed_playbooks > 0:
				metrics.success_rate = Decimal(str(
					(metrics.successful_executions / metrics.executed_playbooks) * 100
				))
			
			# Performance metrics
			if executions:
				execution_times = [e.execution_duration for e in executions if e.execution_duration]
				if execution_times:
					metrics.average_execution_time = sum(execution_times, timedelta()) / len(execution_times)
					metrics.median_execution_time = sorted(execution_times)[len(execution_times) // 2]
			
			# Tool integration metrics
			integrations = await self._get_all_tool_integrations()
			metrics.total_integrations = len(integrations)
			metrics.healthy_integrations = len([i for i in integrations if i.is_healthy])
			
			if metrics.total_integrations > 0:
				metrics.integration_health_rate = Decimal(str(
					(metrics.healthy_integrations / metrics.total_integrations) * 100
				))
			
			# Action metrics
			actions = await self._get_actions_in_period(start_time, end_time)
			metrics.total_actions_executed = len(actions)
			metrics.successful_actions = len([a for a in actions if a.success])
			metrics.failed_actions = len([a for a in actions if not a.success])
			
			if metrics.total_actions_executed > 0:
				metrics.action_success_rate = Decimal(str(
					(metrics.successful_actions / metrics.total_actions_executed) * 100
				))
			
			# Coordination metrics
			coordinations = await self._get_coordinations_in_period(start_time, end_time)
			metrics.coordinated_responses = len(coordinations)
			
			if coordinations:
				team_counts = [len(c.teams_involved) for c in coordinations]
				metrics.average_teams_per_response = Decimal(str(sum(team_counts) / len(team_counts)))
			
			await self._store_orchestration_metrics(metrics)
			
			return metrics
			
		except Exception as e:
			self.logger.error(f"Error generating orchestration metrics: {str(e)}")
			raise
	
	# Helper methods for implementation
	async def _load_security_playbooks(self):
		"""Load security playbooks into cache"""
		pass
	
	async def _initialize_workflow_engine(self):
		"""Initialize workflow execution engine"""
		pass
	
	async def _load_tool_integrations(self):
		"""Load tool integrations into cache"""
		pass
	
	async def _setup_celery_tasks(self):
		"""Setup Celery for async task execution"""
		pass
	
	async def _start_execution_monitor(self):
		"""Start execution monitoring background task"""
		pass
	
	async def _validate_playbook_workflow(self, playbook: SecurityPlaybook) -> Dict[str, Any]:
		"""Validate playbook workflow definition"""
		return {'valid': True, 'errors': []}
	
	async def _parse_workflow_definition(self, workflow_def: Dict[str, Any]) -> List[Dict]:
		"""Parse workflow definition into executable steps"""
		return workflow_def.get('steps', [])
	
	# Placeholder implementations for database operations
	async def _store_security_playbook(self, playbook: SecurityPlaybook):
		"""Store security playbook to database"""
		pass
	
	async def _store_workflow_execution(self, execution: WorkflowExecution):
		"""Store workflow execution to database"""
		pass
	
	async def _store_automation_action(self, action: AutomationAction):
		"""Store automation action to database"""
		pass
	
	async def _store_tool_integration(self, integration: ToolIntegration):
		"""Store tool integration to database"""
		pass
	
	async def _store_response_coordination(self, coordination: ResponseCoordination):
		"""Store response coordination to database"""
		pass
	
	async def _store_orchestration_metrics(self, metrics: OrchestrationMetrics):
		"""Store orchestration metrics to database"""
		pass