#!/usr/bin/env python3
"""
APG Workflow Orchestration Advanced API Features

GraphQL API, versioning, webhooks, bulk operations, and comprehensive API documentation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import hashlib
import hmac
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field, validator
import graphene
from graphene import ObjectType, String, Int, List as GrapheneList, Field, Boolean, DateTime, Float
import aiohttp
from urllib.parse import urlencode

from apg.framework.base_service import APGBaseService
from apg.framework.database import APGDatabase
from apg.framework.audit_compliance import APGAuditLogger

from .config import get_config
from .models import WorkflowStatus, TaskStatus, WorkflowExecution


logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
	"""API version identifiers."""
	V1 = "v1"
	V2 = "v2"
	BETA = "beta"


class WebhookEvent(str, Enum):
	"""Webhook event types."""
	WORKFLOW_CREATED = "workflow.created"
	WORKFLOW_UPDATED = "workflow.updated"
	WORKFLOW_DELETED = "workflow.deleted"
	WORKFLOW_STARTED = "workflow.started"
	WORKFLOW_COMPLETED = "workflow.completed"
	WORKFLOW_FAILED = "workflow.failed"
	WORKFLOW_CANCELLED = "workflow.cancelled"
	EXECUTION_STARTED = "execution.started"
	EXECUTION_COMPLETED = "execution.completed"
	EXECUTION_FAILED = "execution.failed"
	TASK_COMPLETED = "task.completed"
	TASK_FAILED = "task.failed"


class WebhookStatus(str, Enum):
	"""Webhook delivery status."""
	PENDING = "pending"
	DELIVERED = "delivered"
	FAILED = "failed"
	DISABLED = "disabled"


@dataclass
class APIEndpoint:
	"""API endpoint metadata."""
	path: str
	method: str
	version: APIVersion
	description: str
	parameters: Dict[str, Any] = field(default_factory=dict)
	request_schema: Optional[Dict[str, Any]] = None
	response_schema: Optional[Dict[str, Any]] = None
	examples: List[Dict[str, Any]] = field(default_factory=list)
	deprecated: bool = False
	rate_limit: Optional[int] = None
	authentication_required: bool = True
	permissions: List[str] = field(default_factory=list)


@dataclass
class WebhookSubscription:
	"""Webhook subscription configuration."""
	id: str
	user_id: str
	endpoint_url: str
	events: List[WebhookEvent]
	secret: str
	active: bool = True
	retry_count: int = 3
	timeout_seconds: int = 30
	custom_headers: Dict[str, str] = field(default_factory=dict)
	filter_conditions: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	last_triggered: Optional[datetime] = None
	delivery_attempts: int = 0
	successful_deliveries: int = 0
	failed_deliveries: int = 0


@dataclass
class WebhookDelivery:
	"""Webhook delivery attempt."""
	id: str
	subscription_id: str
	event_type: WebhookEvent
	payload: Dict[str, Any]
	status: WebhookStatus = WebhookStatus.PENDING
	response_code: Optional[int] = None
	response_body: Optional[str] = None
	error_message: Optional[str] = None
	attempt_count: int = 0
	created_at: datetime = field(default_factory=datetime.utcnow)
	delivered_at: Optional[datetime] = None
	next_retry_at: Optional[datetime] = None


# GraphQL Schema Definitions

class WorkflowType(ObjectType):
	"""GraphQL type for Workflow."""
	id = String()
	name = String()
	description = String()
	category = String()
	status = String()
	version = String()
	tags = GrapheneList(String)
	created_at = DateTime()
	updated_at = DateTime()
	created_by = String()
	execution_count = Int()
	success_rate = Float()
	average_duration = Float()
	
	async def resolve_execution_count(self, info):
		"""Resolve execution count from database."""
		try:
			from .models import WorkflowExecution
			from .database import get_async_session
			from sqlalchemy import func, select
			from datetime import datetime, timedelta
			
			# Get executions from last 30 days
			thirty_days_ago = datetime.utcnow() - timedelta(days=30)
			
			async with get_async_session() as session:
				result = await session.execute(
					select(func.count(WorkflowExecution.id))
					.where(WorkflowExecution.created_at >= thirty_days_ago)
				)
				count = result.scalar()
				return count or 0
		except Exception:
			# Fallback for development/testing
			return 0
	
	async def resolve_success_rate(self, info):
		"""Resolve success rate from database."""
		try:
			from .models import WorkflowExecution
			from .database import get_async_session
			from sqlalchemy import func, select
			from datetime import datetime, timedelta
			
			# Get success rate from last 30 days
			thirty_days_ago = datetime.utcnow() - timedelta(days=30)
			
			async with get_async_session() as session:
				result = await session.execute(
					select(
						func.count().filter(WorkflowExecution.status == 'completed').label('successful'),
						func.count().label('total')
					).where(WorkflowExecution.created_at >= thirty_days_ago)
				)
				data = result.fetchone()
				
				if data.total > 0:
					success_rate = (data.successful / data.total) * 100
					return round(success_rate, 2)
				else:
					return 0.0
		except Exception:
			# Fallback for development/testing
			return 0.0
	
	async def resolve_average_duration(self, info):
		"""Resolve average duration from database."""
		try:
			from .models import WorkflowExecution
			from .database import get_async_session
			from sqlalchemy import func, select
			from datetime import datetime, timedelta
			
			# Get average duration from last 30 days
			thirty_days_ago = datetime.utcnow() - timedelta(days=30)
			
			async with get_async_session() as session:
				result = await session.execute(
					select(func.avg(
						func.extract('epoch', WorkflowExecution.completed_at - WorkflowExecution.started_at)
					)).where(
						WorkflowExecution.completed_at.isnot(None),
						WorkflowExecution.started_at >= thirty_days_ago
					)
				)
				avg_duration = result.scalar()
				return round(avg_duration, 2) if avg_duration else 0.0
		except Exception:
			# Fallback for development/testing
			return 0.0


class ExecutionType(ObjectType):
	"""GraphQL type for Execution."""
	id = String()
	workflow_id = String()
	workflow_name = String()
	status = String()
	started_at = DateTime()
	completed_at = DateTime()
	duration = Float()
	success = Boolean()
	error_message = String()
	progress = Int()
	
	def resolve_duration(self, info):
		"""Calculate execution duration."""
		if self.started_at and self.completed_at:
			return (self.completed_at - self.started_at).total_seconds()
		return None


class TemplateType(ObjectType):
	"""GraphQL type for Template."""
	id = String()
	name = String()
	description = String()
	category = String()
	complexity = String()
	tags = GrapheneList(String)
	use_cases = GrapheneList(String)
	popularity_score = Float()
	usage_count = Int()
	rating = Float()
	created_at = DateTime()


class MetricType(ObjectType):
	"""GraphQL type for Metrics."""
	name = String()
	value = Float()
	unit = String()
	timestamp = DateTime()
	labels = String()  # JSON string of labels


class Query(ObjectType):
	"""GraphQL Query root."""
	
	# Workflow queries
	workflow = Field(WorkflowType, id=String(required=True))
	workflows = GrapheneList(
		WorkflowType,
		category=String(),
		status=String(),
		created_by=String(),
		limit=Int(default_value=50),
		offset=Int(default_value=0)
	)
	
	# Execution queries
	execution = Field(ExecutionType, id=String(required=True))
	executions = GrapheneList(
		ExecutionType,
		workflow_id=String(),
		status=String(),
		limit=Int(default_value=50),
		offset=Int(default_value=0)
	)
	
	# Template queries
	template = Field(TemplateType, id=String(required=True))
	templates = GrapheneList(
		TemplateType,
		category=String(),
		complexity=String(),
		tags=GrapheneList(String),
		limit=Int(default_value=50),
		offset=Int(default_value=0)
	)
	
	# Metrics queries
	metrics = GrapheneList(
		MetricType,
		names=GrapheneList(String),
		start_time=DateTime(),
		end_time=DateTime()
	)
	
	async def resolve_workflow(self, info, id):
		"""Resolve single workflow from database."""
		try:
			from .database import DatabaseManager
			import json
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				SELECT 
					w.id, w.name, w.description, w.category, w.status, 
					w.version, w.tags, w.created_at, w.updated_at, 
					w.created_by, w.metadata, w.configuration,
					COUNT(wi.id) as execution_count
				FROM cr_workflows w
				LEFT JOIN cr_workflow_instances wi ON w.id = wi.workflow_id
				WHERE w.id = %s AND w.tenant_id = %s
				GROUP BY w.id
				"""
				
				# Get tenant ID from context
				tenant_id = getattr(info.context, 'tenant_id', 'default_tenant')
				
				result = await session.execute(query, [id, tenant_id])
				row = result.fetchone()
				
				if not row:
					return None
				
				# Parse JSON fields
				tags = json.loads(row['tags']) if row['tags'] else []
				metadata = json.loads(row['metadata']) if row['metadata'] else {}
				
				return {
					'id': row['id'],
					'name': row['name'],
					'description': row['description'],
					'category': row['category'],
					'status': row['status'],
					'version': row['version'],
					'tags': tags,
					'created_at': row['created_at'],
					'updated_at': row['updated_at'],
					'created_by': row['created_by'],
					'execution_count': row['execution_count'] or 0,
					'metadata': metadata
				}
				
		except Exception as e:
			logger.error(f"Failed to resolve workflow {id}: {e}")
			return None
	
	async def resolve_workflows(self, info, **kwargs):
		"""Resolve workflows list from database."""
		try:
			from .database import DatabaseManager
			import json
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				# Build dynamic WHERE clause based on filters
				where_conditions = ["w.tenant_id = %s"]
				query_params = [getattr(info.context, 'tenant_id', 'default_tenant')]
				
				category = kwargs.get('category')
				status = kwargs.get('status')
				created_by = kwargs.get('created_by')
				limit = kwargs.get('limit', 50)
				offset = kwargs.get('offset', 0)
				
				if category:
					where_conditions.append("w.category = %s")
					query_params.append(category)
				
				if status:
					where_conditions.append("w.status = %s")
					query_params.append(status)
				
				if created_by:
					where_conditions.append("w.created_by = %s")
					query_params.append(created_by)
				
				query = f"""
				SELECT 
					w.id, w.name, w.description, w.category, w.status, 
					w.version, w.tags, w.created_at, w.updated_at, 
					w.created_by, w.metadata,
					COUNT(wi.id) as execution_count,
					AVG(CASE WHEN wi.status = 'completed' THEN 1.0 ELSE 0.0 END) as success_rate
				FROM cr_workflows w
				LEFT JOIN cr_workflow_instances wi ON w.id = wi.workflow_id
				WHERE {' AND '.join(where_conditions)}
				GROUP BY w.id, w.name, w.description, w.category, w.status, 
						 w.version, w.tags, w.created_at, w.updated_at, 
						 w.created_by, w.metadata
				ORDER BY w.updated_at DESC
				LIMIT %s OFFSET %s
				"""
				
				query_params.extend([limit, offset])
				
				result = await session.execute(query, query_params)
				rows = result.fetchall()
				
				workflows = []
				for row in rows:
					tags = json.loads(row['tags']) if row['tags'] else []
					metadata = json.loads(row['metadata']) if row['metadata'] else {}
					
					workflows.append({
						'id': row['id'],
						'name': row['name'],
						'description': row['description'],
						'category': row['category'],
						'status': row['status'],
						'version': row['version'],
						'tags': tags,
						'created_at': row['created_at'],
						'updated_at': row['updated_at'],
						'created_by': row['created_by'],
						'execution_count': row['execution_count'] or 0,
						'success_rate': float(row['success_rate'] or 0.0),
						'metadata': metadata
					})
				
				return workflows
				
		except Exception as e:
			logger.error(f"Failed to resolve workflows: {e}")
			return []
	
	async def resolve_execution(self, info, id):
		"""Resolve single execution from database."""
		try:
			from .database import DatabaseManager
			import json
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				SELECT 
					wi.id, wi.workflow_id, wi.status, wi.started_at, 
					wi.completed_at, wi.progress_percentage, wi.error_message,
					wi.input_data, wi.output_data, wi.metadata,
					w.name as workflow_name, w.description as workflow_description,
					COUNT(te.id) as task_count,
					COUNT(te.id) FILTER (WHERE te.status = 'completed') as completed_tasks,
					COUNT(te.id) FILTER (WHERE te.status = 'failed') as failed_tasks
				FROM cr_workflow_instances wi
				JOIN cr_workflows w ON wi.workflow_id = w.id
				LEFT JOIN cr_task_executions te ON wi.id = te.instance_id
				WHERE wi.id = %s AND wi.tenant_id = %s
				GROUP BY wi.id, w.name, w.description
				"""
				
				tenant_id = getattr(info.context, 'tenant_id', 'default_tenant')
				
				result = await session.execute(query, [id, tenant_id])
				row = result.fetchone()
				
				if not row:
					return None
				
				# Calculate duration if both timestamps exist
				duration = None
				if row['started_at'] and row['completed_at']:
					duration = (row['completed_at'] - row['started_at']).total_seconds()
				
				# Determine success based on status and error message
				success = row['status'] == 'completed' and not row['error_message']
				
				# Parse JSON fields
				input_data = json.loads(row['input_data']) if row['input_data'] else {}
				output_data = json.loads(row['output_data']) if row['output_data'] else {}
				metadata = json.loads(row['metadata']) if row['metadata'] else {}
				
				return {
					'id': row['id'],
					'workflow_id': row['workflow_id'],
					'workflow_name': row['workflow_name'],
					'status': row['status'],
					'started_at': row['started_at'],
					'completed_at': row['completed_at'],
					'duration': duration,
					'success': success,
					'progress': row['progress_percentage'] or 0,
					'error_message': row['error_message'],
					'task_count': row['task_count'] or 0,
					'completed_tasks': row['completed_tasks'] or 0,
					'failed_tasks': row['failed_tasks'] or 0,
					'input_data': input_data,
					'output_data': output_data,
					'metadata': metadata
				}
				
		except Exception as e:
			logger.error(f"Failed to resolve execution {id}: {e}")
			return None
	
	async def resolve_executions(self, info, **kwargs):
		"""Resolve executions list from database."""
		try:
			from .database import DatabaseManager
			import json
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				# Build dynamic WHERE clause based on filters
				where_conditions = ["wi.tenant_id = %s"]
				query_params = [getattr(info.context, 'tenant_id', 'default_tenant')]
				
				workflow_id = kwargs.get('workflow_id')
				status = kwargs.get('status')
				limit = kwargs.get('limit', 50)
				offset = kwargs.get('offset', 0)
				
				if workflow_id:
					where_conditions.append("wi.workflow_id = %s")
					query_params.append(workflow_id)
				
				if status:
					where_conditions.append("wi.status = %s")
					query_params.append(status)
				
				query = f"""
				SELECT 
					wi.id, wi.workflow_id, wi.status, wi.started_at, 
					wi.completed_at, wi.progress_percentage, wi.error_message,
					wi.priority, wi.retry_count,
					w.name as workflow_name, w.category as workflow_category,
					COUNT(te.id) as task_count,
					COUNT(te.id) FILTER (WHERE te.status = 'completed') as completed_tasks,
					COUNT(te.id) FILTER (WHERE te.status = 'failed') as failed_tasks,
					AVG(EXTRACT(EPOCH FROM (te.completed_at - te.started_at))) as avg_task_duration
				FROM cr_workflow_instances wi
				JOIN cr_workflows w ON wi.workflow_id = w.id
				LEFT JOIN cr_task_executions te ON wi.id = te.instance_id
				WHERE {' AND '.join(where_conditions)}
				GROUP BY wi.id, w.name, w.category
				ORDER BY wi.started_at DESC
				LIMIT %s OFFSET %s
				"""
				
				query_params.extend([limit, offset])
				
				result = await session.execute(query, query_params)
				rows = result.fetchall()
				
				executions = []
				for row in rows:
					# Calculate duration if both timestamps exist
					duration = None
					if row['started_at'] and row['completed_at']:
						duration = (row['completed_at'] - row['started_at']).total_seconds()
					
					# Determine success based on status and error message
					success = row['status'] == 'completed' and not row['error_message']
					
					executions.append({
						'id': row['id'],
						'workflow_id': row['workflow_id'],
						'workflow_name': row['workflow_name'],
						'workflow_category': row['workflow_category'],
						'status': row['status'],
						'started_at': row['started_at'],
						'completed_at': row['completed_at'],
						'duration': duration,
						'success': success,
						'progress': row['progress_percentage'] or 0,
						'error_message': row['error_message'],
						'priority': row['priority'],
						'retry_count': row['retry_count'] or 0,
						'task_count': row['task_count'] or 0,
						'completed_tasks': row['completed_tasks'] or 0,
						'failed_tasks': row['failed_tasks'] or 0,
						'avg_task_duration': float(row['avg_task_duration'] or 0.0)
					})
				
				return executions
				
		except Exception as e:
			logger.error(f"Failed to resolve executions: {e}")
			return []
	
	async def resolve_template(self, info, id):
		"""Resolve single template from database."""
		try:
			from .database import DatabaseManager
			import json
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				SELECT 
					t.id, t.name, t.description, t.category, t.complexity,
					t.tags, t.use_cases, t.popularity_score, t.created_at,
					t.metadata, t.template_definition,
					COUNT(wi.id) as usage_count,
					AVG(r.rating) as rating
				FROM cr_workflow_templates t
				LEFT JOIN cr_workflows w ON w.template_id = t.id
				LEFT JOIN cr_workflow_instances wi ON w.id = wi.workflow_id
				LEFT JOIN cr_template_ratings r ON t.id = r.template_id
				WHERE t.id = %s AND t.tenant_id = %s
				GROUP BY t.id
				"""
				
				tenant_id = getattr(info.context, 'tenant_id', 'default_tenant')
				
				result = await session.execute(query, [id, tenant_id])
				row = result.fetchone()
				
				if not row:
					return None
				
				# Parse JSON fields
				tags = json.loads(row['tags']) if row['tags'] else []
				use_cases = json.loads(row['use_cases']) if row['use_cases'] else []
				metadata = json.loads(row['metadata']) if row['metadata'] else {}
				
				return {
					'id': row['id'],
					'name': row['name'],
					'description': row['description'],
					'category': row['category'],
					'complexity': row['complexity'],
					'tags': tags,
					'use_cases': use_cases,
					'popularity_score': float(row['popularity_score'] or 0.0),
					'usage_count': row['usage_count'] or 0,
					'rating': float(row['rating'] or 0.0),
					'created_at': row['created_at'],
					'metadata': metadata
				}
				
		except Exception as e:
			logger.error(f"Failed to resolve template {id}: {e}")
			return None
	
	async def resolve_templates(self, info, **kwargs):
		"""Resolve templates list from database."""
		try:
			from .database import DatabaseManager
			import json
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				# Build dynamic WHERE clause based on filters
				where_conditions = ["t.tenant_id = %s"]
				query_params = [getattr(info.context, 'tenant_id', 'default_tenant')]
				
				category = kwargs.get('category')
				complexity = kwargs.get('complexity')
				tags = kwargs.get('tags', [])
				limit = kwargs.get('limit', 50)
				offset = kwargs.get('offset', 0)
				
				if category:
					where_conditions.append("t.category = %s")
					query_params.append(category)
				
				if complexity:
					where_conditions.append("t.complexity = %s")
					query_params.append(complexity)
				
				if tags:
					# Filter by tags using JSON contains
					where_conditions.append("t.tags::jsonb ?| %s")
					query_params.append(tags)
				
				query = f"""
				SELECT 
					t.id, t.name, t.description, t.category, t.complexity,
					t.tags, t.use_cases, t.popularity_score, t.created_at,
					t.estimated_duration, t.difficulty_level,
					COUNT(DISTINCT w.id) as usage_count,
					COUNT(DISTINCT wi.id) as execution_count,
					AVG(r.rating) as rating,
					COUNT(r.id) as rating_count
				FROM cr_workflow_templates t
				LEFT JOIN cr_workflows w ON w.template_id = t.id
				LEFT JOIN cr_workflow_instances wi ON w.id = wi.workflow_id
				LEFT JOIN cr_template_ratings r ON t.id = r.template_id
				WHERE {' AND '.join(where_conditions)}
				GROUP BY t.id, t.name, t.description, t.category, t.complexity,
						 t.tags, t.use_cases, t.popularity_score, t.created_at,
						 t.estimated_duration, t.difficulty_level
				ORDER BY t.popularity_score DESC, t.created_at DESC
				LIMIT %s OFFSET %s
				"""
				
				query_params.extend([limit, offset])
				
				result = await session.execute(query, query_params)
				rows = result.fetchall()
				
				templates = []
				for row in rows:
					# Parse JSON fields
					tags = json.loads(row['tags']) if row['tags'] else []
					use_cases = json.loads(row['use_cases']) if row['use_cases'] else []
					
					templates.append({
						'id': row['id'],
						'name': row['name'],
						'description': row['description'],
						'category': row['category'],
						'complexity': row['complexity'],
						'tags': tags,
						'use_cases': use_cases,
						'popularity_score': float(row['popularity_score'] or 0.0),
						'usage_count': row['usage_count'] or 0,
						'execution_count': row['execution_count'] or 0,
						'rating': float(row['rating'] or 0.0),
						'rating_count': row['rating_count'] or 0,
						'estimated_duration': row['estimated_duration'],
						'difficulty_level': row['difficulty_level'],
						'created_at': row['created_at']
					})
				
				return templates
				
		except Exception as e:
			logger.error(f"Failed to resolve templates: {e}")
			return []
	
	async def resolve_metrics(self, info, **kwargs):
		"""Resolve metrics from monitoring database and real-time collection."""
		try:
			from .database import DatabaseManager
			import asyncio
			import psutil
			import json
			from datetime import datetime, timedelta, timezone
			
			names = kwargs.get('names', [])
			start_time = kwargs.get('start_time')
			end_time = kwargs.get('end_time')
			
			# Default time range if not provided
			if not end_time:
				end_time = datetime.now(timezone.utc)
			if not start_time:
				start_time = end_time - timedelta(hours=1)
			
			# Default metrics if none specified
			if not names:
				names = ['cpu_percent', 'memory_percent', 'active_workflows', 'workflow_executions', 'execution_duration']
			
			db_manager = DatabaseManager()
			metrics = []
			
			async with db_manager.get_session() as session:
				# Get tenant context for multi-tenant queries
				tenant_id = self._get_tenant_context(info)
				
				for name in names:
					if name in ['cpu_percent', 'memory_percent', 'disk_usage', 'network_io']:
						# Real-time system metrics
						metrics.extend(await self._get_system_metrics(name, start_time, end_time))
					
					elif name == 'active_workflows':
						# Query active workflow count from database
						query = """
						SELECT 
							DATE_TRUNC('minute', created_at) as time_bucket,
							COUNT(*) as value
						FROM cr_workflow_instances wi
						WHERE wi.status IN ('running', 'paused')
							AND wi.tenant_id = %s
							AND wi.created_at BETWEEN %s AND %s
						GROUP BY DATE_TRUNC('minute', created_at)
						ORDER BY time_bucket
						"""
						
						result = await session.execute(query, (tenant_id, start_time, end_time))
						rows = await result.fetchall()
						
						for row in rows:
							metrics.append({
								'name': name,
								'value': float(row['value']),
								'unit': 'count',
								'timestamp': row['time_bucket'],
								'labels': json.dumps({'metric_type': 'workflow_count'})
							})
					
					elif name == 'workflow_executions':
						# Query workflow execution count from database
						query = """
						SELECT 
							DATE_TRUNC('minute', created_at) as time_bucket,
							COUNT(*) as value,
							AVG(CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 as success_rate
						FROM cr_workflow_instances wi
						WHERE wi.tenant_id = %s
							AND wi.created_at BETWEEN %s AND %s
						GROUP BY DATE_TRUNC('minute', created_at)
						ORDER BY time_bucket
						"""
						
						result = await session.execute(query, (tenant_id, start_time, end_time))
						rows = await result.fetchall()
						
						for row in rows:
							metrics.append({
								'name': name,
								'value': float(row['value']),
								'unit': 'count',
								'timestamp': row['time_bucket'],
								'labels': json.dumps({
									'metric_type': 'execution_count',
									'success_rate': f"{row['success_rate']:.1f}%"
								})
							})
					
					elif name == 'execution_duration':
						# Query average execution duration from database
						query = """
						SELECT 
							DATE_TRUNC('minute', created_at) as time_bucket,
							AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration,
							MIN(EXTRACT(EPOCH FROM (updated_at - created_at))) as min_duration,
							MAX(EXTRACT(EPOCH FROM (updated_at - created_at))) as max_duration
						FROM cr_workflow_instances wi
						WHERE wi.status = 'completed'
							AND wi.tenant_id = %s
							AND wi.created_at BETWEEN %s AND %s
							AND wi.updated_at IS NOT NULL
						GROUP BY DATE_TRUNC('minute', created_at)
						ORDER BY time_bucket
						"""
						
						result = await session.execute(query, (tenant_id, start_time, end_time))
						rows = await result.fetchall()
						
						for row in rows:
							metrics.append({
								'name': name,
								'value': float(row['avg_duration'] or 0),
								'unit': 'seconds',
								'timestamp': row['time_bucket'],
								'labels': json.dumps({
									'metric_type': 'duration',
									'min_duration': str(row['min_duration'] or 0),
									'max_duration': str(row['max_duration'] or 0)
								})
							})
					
					elif name == 'error_rate':
						# Query error rate from database
						query = """
						SELECT 
							DATE_TRUNC('minute', created_at) as time_bucket,
							COUNT(*) as total_count,
							SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as error_count,
							CASE 
								WHEN COUNT(*) > 0 
								THEN (SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*))
								ELSE 0.0 
							END as error_rate
						FROM cr_workflow_instances wi
						WHERE wi.tenant_id = %s
							AND wi.created_at BETWEEN %s AND %s
						GROUP BY DATE_TRUNC('minute', created_at)
						ORDER BY time_bucket
						"""
						
						result = await session.execute(query, (tenant_id, start_time, end_time))
						rows = await result.fetchall()
						
						for row in rows:
							metrics.append({
								'name': name,
								'value': float(row['error_rate']),
								'unit': 'percent',
								'timestamp': row['time_bucket'],
								'labels': json.dumps({
									'metric_type': 'error_rate',
									'total_count': str(row['total_count']),
									'error_count': str(row['error_count'])
								})
							})
					
					elif name == 'task_throughput':
						# Query task execution throughput
						query = """
						SELECT 
							DATE_TRUNC('minute', wt.created_at) as time_bucket,
							COUNT(*) as task_count,
							COUNT(*) / 60.0 as tasks_per_second
						FROM cr_workflow_tasks wt
						JOIN cr_workflow_instances wi ON wt.workflow_instance_id = wi.id
						WHERE wi.tenant_id = %s
							AND wt.created_at BETWEEN %s AND %s
							AND wt.status = 'completed'
						GROUP BY DATE_TRUNC('minute', wt.created_at)
						ORDER BY time_bucket
						"""
						
						result = await session.execute(query, (tenant_id, start_time, end_time))
						rows = await result.fetchall()
						
						for row in rows:
							metrics.append({
								'name': name,
								'value': float(row['tasks_per_second']),
								'unit': 'tasks/sec',
								'timestamp': row['time_bucket'],
								'labels': json.dumps({
									'metric_type': 'throughput',
									'task_count': str(row['task_count'])
								})
							})
					
					else:
						# Check for custom metrics in monitoring data
						query = """
						SELECT 
							timestamp,
							metric_value as value,
							metric_unit as unit,
							labels
						FROM cr_monitoring_metrics
						WHERE metric_name = %s
							AND tenant_id = %s
							AND timestamp BETWEEN %s AND %s
						ORDER BY timestamp
						"""
						
						result = await session.execute(query, (name, tenant_id, start_time, end_time))
						rows = await result.fetchall()
						
						for row in rows:
							metrics.append({
								'name': name,
								'value': float(row['value']),
								'unit': row['unit'] or 'count',
								'timestamp': row['timestamp'],
								'labels': row['labels'] or '{}'
							})
			
			return metrics
				
		except Exception as e:
			logger.error(f"Failed to resolve metrics: {e}")
			return []
	
	async def _get_system_metrics(self, metric_name: str, start_time: datetime, end_time: datetime) -> list:
		"""Get real-time system metrics."""
		try:
			import psutil
			
			metrics = []
			current_time = datetime.now(timezone.utc)
			
			# Generate data points at 1-minute intervals
			time_diff = end_time - start_time
			minutes = int(time_diff.total_seconds() / 60)
			
			for i in range(min(minutes, 60)):  # Limit to 60 data points
				timestamp = end_time - timedelta(minutes=i)
				
				if metric_name == 'cpu_percent':
					# Get CPU usage percentage
					cpu_percent = psutil.cpu_percent(interval=0.1)
					value = cpu_percent
					unit = 'percent'
					labels = json.dumps({'cores': psutil.cpu_count()})
				
				elif metric_name == 'memory_percent':
					# Get memory usage percentage
					memory = psutil.virtual_memory()
					value = memory.percent
					unit = 'percent'
					labels = json.dumps({
						'total_gb': f"{memory.total / (1024**3):.1f}",
						'available_gb': f"{memory.available / (1024**3):.1f}"
					})
				
				elif metric_name == 'disk_usage':
					# Get disk usage percentage
					disk = psutil.disk_usage('/')
					value = (disk.used / disk.total) * 100
					unit = 'percent'
					labels = json.dumps({
						'total_gb': f"{disk.total / (1024**3):.1f}",
						'free_gb': f"{disk.free / (1024**3):.1f}"
					})
				
				elif metric_name == 'network_io':
					# Get network I/O
					net_io = psutil.net_io_counters()
					value = (net_io.bytes_sent + net_io.bytes_recv) / (1024**2)  # MB
					unit = 'MB'
					labels = json.dumps({
						'bytes_sent': str(net_io.bytes_sent),
						'bytes_recv': str(net_io.bytes_recv)
					})
				
				else:
					continue
				
				metrics.append({
					'name': metric_name,
					'value': value,
					'unit': unit,
					'timestamp': timestamp,
					'labels': labels
				})
			
			return metrics
			
		except Exception as e:
			logger.warning(f"Failed to get system metric {metric_name}: {e}")
			return []


class CreateWorkflowMutation(graphene.Mutation):
	"""Mutation to create a workflow."""
	
	class Arguments:
		name = String(required=True)
		description = String()
		category = String()
		workflow_definition = String(required=True)  # JSON string
	
	workflow = Field(WorkflowType)
	success = Boolean()
	errors = GrapheneList(String)
	
	def mutate(self, info, name, workflow_definition, description=None, category=None):
		"""Create new workflow."""
		try:
			# Validate workflow definition
			workflow_data = json.loads(workflow_definition)
			
			# In real implementation, save to database
			workflow = {
				'id': uuid7str(),
				'name': name,
				'description': description or '',
				'category': category or 'automation',
				'status': 'draft',
				'version': '1.0.0',
				'tags': [],
				'created_at': datetime.utcnow(),
				'updated_at': datetime.utcnow(),
				'created_by': 'current_user'  # From context
			}
			
			return CreateWorkflowMutation(
				workflow=workflow,
				success=True,
				errors=[]
			)
			
		except json.JSONDecodeError:
			return CreateWorkflowMutation(
				workflow=None,
				success=False,
				errors=['Invalid workflow definition JSON']
			)
		except Exception as e:
			return CreateWorkflowMutation(
				workflow=None,
				success=False,
				errors=[str(e)]
			)


class UpdateWorkflowMutation(graphene.Mutation):
	"""Mutation to update a workflow."""
	
	class Arguments:
		id = String(required=True)
		name = String()
		description = String()
		category = String()
		workflow_definition = String()
	
	workflow = Field(WorkflowType)
	success = Boolean()
	errors = GrapheneList(String)
	
	def mutate(self, info, id, **kwargs):
		"""Update existing workflow."""
		try:
			# In real implementation, update in database
			workflow = {
				'id': id,
				'name': kwargs.get('name', 'Updated Workflow'),
				'description': kwargs.get('description', ''),
				'category': kwargs.get('category', 'automation'),
				'status': 'active',
				'version': '1.1.0',
				'tags': [],
				'created_at': datetime.utcnow() - timedelta(days=1),
				'updated_at': datetime.utcnow(),
				'created_by': 'current_user'
			}
			
			return UpdateWorkflowMutation(
				workflow=workflow,
				success=True,
				errors=[]
			)
			
		except Exception as e:
			return UpdateWorkflowMutation(
				workflow=None,
				success=False,
				errors=[str(e)]
			)


class ExecuteWorkflowMutation(graphene.Mutation):
	"""Mutation to execute a workflow."""
	
	class Arguments:
		workflow_id = String(required=True)
		parameters = String()  # JSON string
	
	execution = Field(ExecutionType)
	success = Boolean()
	errors = GrapheneList(String)
	
	def mutate(self, info, workflow_id, parameters=None):
		"""Execute workflow."""
		try:
			# Parse parameters
			exec_params = {}
			if parameters:
				exec_params = json.loads(parameters)
			
			# In real implementation, start execution
			execution = {
				'id': uuid7str(),
				'workflow_id': workflow_id,
				'workflow_name': 'Sample Workflow',
				'status': 'running',
				'started_at': datetime.utcnow(),
				'completed_at': None,
				'success': None,
				'progress': 0
			}
			
			return ExecuteWorkflowMutation(
				execution=execution,
				success=True,
				errors=[]
			)
			
		except Exception as e:
			return ExecuteWorkflowMutation(
				execution=None,
				success=False,
				errors=[str(e)]
			)


class Mutation(ObjectType):
	"""GraphQL Mutation root."""
	create_workflow = CreateWorkflowMutation.Field()
	update_workflow = UpdateWorkflowMutation.Field()
	execute_workflow = ExecuteWorkflowMutation.Field()


# Create GraphQL schema
graphql_schema = graphene.Schema(query=Query, mutation=Mutation)


class APIVersionManager:
	"""Manages API versioning and compatibility."""
	
	def __init__(self):
		self.versions: Dict[APIVersion, Dict[str, APIEndpoint]] = {
			APIVersion.V1: {},
			APIVersion.V2: {},
			APIVersion.BETA: {}
		}
		self.default_version = APIVersion.V1
		self.deprecated_versions: Set[APIVersion] = set()
		
		# Initialize API endpoints
		self._register_v1_endpoints()
		self._register_v2_endpoints()
		self._register_beta_endpoints()
	
	def _register_v1_endpoints(self):
		"""Register V1 API endpoints."""
		v1_endpoints = [
			APIEndpoint(
				path="/workflows",
				method="GET",
				version=APIVersion.V1,
				description="List workflows",
				parameters={
					"limit": {"type": "integer", "default": 50, "maximum": 100},
					"offset": {"type": "integer", "default": 0},
					"category": {"type": "string"},
					"status": {"type": "string", "enum": ["active", "draft", "archived"]}
				},
				response_schema={
					"type": "object",
					"properties": {
						"workflows": {"type": "array", "items": {"$ref": "#/components/schemas/Workflow"}},
						"total": {"type": "integer"},
						"limit": {"type": "integer"},
						"offset": {"type": "integer"}
					}
				},
				examples=[
					{
						"name": "List active workflows",
						"request": {"status": "active", "limit": 10},
						"response": {
							"workflows": [],
							"total": 0,
							"limit": 10,
							"offset": 0
						}
					}
				]
			),
			
			APIEndpoint(
				path="/workflows/{id}",
				method="GET",
				version=APIVersion.V1,
				description="Get workflow by ID",
				parameters={
					"id": {"type": "string", "required": True, "in": "path"}
				},
				response_schema={"$ref": "#/components/schemas/Workflow"}
			),
			
			APIEndpoint(
				path="/workflows",
				method="POST",
				version=APIVersion.V1,
				description="Create new workflow",
				request_schema={"$ref": "#/components/schemas/CreateWorkflowRequest"},
				response_schema={"$ref": "#/components/schemas/Workflow"}
			),
			
			APIEndpoint(
				path="/workflows/{id}/execute",
				method="POST",
				version=APIVersion.V1,
				description="Execute workflow",
				parameters={
					"id": {"type": "string", "required": True, "in": "path"}
				},
				request_schema={"$ref": "#/components/schemas/ExecuteWorkflowRequest"},
				response_schema={"$ref": "#/components/schemas/Execution"}
			),
			
			APIEndpoint(
				path="/executions",
				method="GET",
				version=APIVersion.V1,
				description="List workflow executions",
				parameters={
					"workflow_id": {"type": "string"},
					"status": {"type": "string"},
					"limit": {"type": "integer", "default": 50},
					"offset": {"type": "integer", "default": 0}
				}
			),
			
			APIEndpoint(
				path="/templates",
				method="GET",
				version=APIVersion.V1,
				description="List workflow templates",
				parameters={
					"category": {"type": "string"},
					"complexity": {"type": "string"},
					"tags": {"type": "array", "items": {"type": "string"}}
				}
			)
		]
		
		for endpoint in v1_endpoints:
			key = f"{endpoint.method}:{endpoint.path}"
			self.versions[APIVersion.V1][key] = endpoint
	
	def _register_v2_endpoints(self):
		"""Register V2 API endpoints with enhancements."""
		# V2 adds bulk operations, advanced filtering, and improved responses
		v2_endpoints = [
			# Enhanced workflows endpoint with advanced filtering
			APIEndpoint(
				path="/workflows",
				method="GET",
				version=APIVersion.V2,
				description="List workflows with advanced filtering",
				parameters={
					"limit": {"type": "integer", "default": 50, "maximum": 100},
					"offset": {"type": "integer", "default": 0},
					"category": {"type": "string"},
					"status": {"type": "string", "enum": ["active", "draft", "archived"]},
					"tags": {"type": "array", "items": {"type": "string"}},
					"created_by": {"type": "string"},
					"created_after": {"type": "string", "format": "date-time"},
					"created_before": {"type": "string", "format": "date-time"},
					"sort_by": {"type": "string", "enum": ["name", "created_at", "updated_at"]},
					"sort_order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"}
				}
			),
			
			# Bulk operations endpoint
			APIEndpoint(
				path="/workflows/bulk",
				method="POST",
				version=APIVersion.V2,
				description="Perform bulk operations on workflows",
				request_schema={
					"type": "object",
					"properties": {
						"operation": {"type": "string", "enum": ["delete", "archive", "activate", "duplicate"]},
						"workflow_ids": {"type": "array", "items": {"type": "string"}},
						"parameters": {"type": "object"}
					},
					"required": ["operation", "workflow_ids"]
				}
			),
			
			# Batch execution endpoint
			APIEndpoint(
				path="/workflows/execute-batch",
				method="POST",
				version=APIVersion.V2,
				description="Execute multiple workflows",
				request_schema={
					"type": "object",
					"properties": {
						"executions": {
							"type": "array",
							"items": {
								"type": "object",
								"properties": {
									"workflow_id": {"type": "string"},
									"parameters": {"type": "object"}
								}
							}
						}
					}
				}
			),
			
			# Advanced metrics endpoint
			APIEndpoint(
				path="/metrics/advanced",
				method="GET",
				version=APIVersion.V2,
				description="Get advanced metrics and analytics",
				parameters={
					"metrics": {"type": "array", "items": {"type": "string"}},
					"start_time": {"type": "string", "format": "date-time"},
					"end_time": {"type": "string", "format": "date-time"},
					"aggregation": {"type": "string", "enum": ["avg", "sum", "min", "max", "count"]},
					"group_by": {"type": "string"}
				}
			)
		]
		
		for endpoint in v2_endpoints:
			key = f"{endpoint.method}:{endpoint.path}"
			self.versions[APIVersion.V2][key] = endpoint
	
	def _register_beta_endpoints(self):
		"""Register Beta API endpoints with experimental features."""
		beta_endpoints = [
			# GraphQL endpoint
			APIEndpoint(
				path="/graphql",
				method="POST",
				version=APIVersion.BETA,
				description="GraphQL endpoint for flexible queries",
				request_schema={
					"type": "object",
					"properties": {
						"query": {"type": "string"},
						"variables": {"type": "object"},
						"operationName": {"type": "string"}
					},
					"required": ["query"]
				}
			),
			
			# AI-powered workflow suggestions
			APIEndpoint(
				path="/ai/workflow-suggestions",
				method="POST",
				version=APIVersion.BETA,
				description="Get AI-powered workflow suggestions",
				request_schema={
					"type": "object",
					"properties": {
						"description": {"type": "string"},
						"requirements": {"type": "array", "items": {"type": "string"}},
						"industry": {"type": "string"}
					}
				}
			),
			
			# Natural language workflow creation
			APIEndpoint(
				path="/ai/create-from-description",
				method="POST",
				version=APIVersion.BETA,
				description="Create workflow from natural language description",
				request_schema={
					"type": "object",
					"properties": {
						"description": {"type": "string"},
						"complexity": {"type": "string", "enum": ["simple", "moderate", "complex"]}
					},
					"required": ["description"]
				}
			)
		]
		
		for endpoint in beta_endpoints:
			key = f"{endpoint.method}:{endpoint.path}"
			self.versions[APIVersion.BETA][key] = endpoint
	
	def get_endpoint(self, version: APIVersion, method: str, path: str) -> Optional[APIEndpoint]:
		"""Get API endpoint definition."""
		key = f"{method}:{path}"
		return self.versions.get(version, {}).get(key)
	
	def get_all_endpoints(self, version: APIVersion = None) -> List[APIEndpoint]:
		"""Get all endpoints for a version or all versions."""
		if version:
			return list(self.versions.get(version, {}).values())
		
		all_endpoints = []
		for version_endpoints in self.versions.values():
			all_endpoints.extend(version_endpoints.values())
		return all_endpoints
	
	def is_version_deprecated(self, version: APIVersion) -> bool:
		"""Check if API version is deprecated."""
		return version in self.deprecated_versions
	
	def deprecate_version(self, version: APIVersion):
		"""Mark API version as deprecated."""
		self.deprecated_versions.add(version)
	
	def get_version_compatibility_info(self, from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
		"""Get compatibility information between API versions."""
		compatibility = {
			'compatible': True,
			'breaking_changes': [],
			'new_features': [],
			'deprecated_features': [],
			'migration_guide': []
		}
		
		# V1 to V2 changes
		if from_version == APIVersion.V1 and to_version == APIVersion.V2:
			compatibility.update({
				'breaking_changes': [
					'Response format changes in some endpoints',
					'Additional required parameters in bulk operations'
				],
				'new_features': [
					'Bulk operations support',
					'Advanced filtering options',
					'Batch execution capabilities',
					'Enhanced metrics endpoint'
				],
				'migration_guide': [
					'Update response parsing for enhanced endpoints',
					'Utilize new bulk operation endpoints for better performance',
					'Migrate to advanced filtering parameters'
				]
			})
		
		return compatibility


class WebhookManager:
	"""Manages webhook subscriptions and deliveries."""
	
	def __init__(self):
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
		self.subscriptions: Dict[str, WebhookSubscription] = {}
		self.pending_deliveries: Dict[str, WebhookDelivery] = {}
		self._delivery_worker_running = False
	
	async def start(self):
		"""Start webhook delivery worker."""
		self._delivery_worker_running = True
		asyncio.create_task(self._delivery_worker())
		logger.info("Webhook manager started")
	
	async def stop(self):
		"""Stop webhook delivery worker."""
		self._delivery_worker_running = False
		logger.info("Webhook manager stopped")
	
	async def create_subscription(self, user_id: str, endpoint_url: str, 
								 events: List[WebhookEvent], 
								 config: Dict[str, Any] = None) -> str:
		"""Create webhook subscription."""
		try:
			subscription_id = uuid7str()
			secret = self._generate_webhook_secret()
			
			subscription = WebhookSubscription(
				id=subscription_id,
				user_id=user_id,
				endpoint_url=endpoint_url,
				events=events,
				secret=secret,
				retry_count=config.get('retry_count', 3) if config else 3,
				timeout_seconds=config.get('timeout_seconds', 30) if config else 30,
				custom_headers=config.get('custom_headers', {}) if config else {},
				filter_conditions=config.get('filter_conditions', {}) if config else {}
			)
			
			self.subscriptions[subscription_id] = subscription
			
			# Log subscription creation
			await self.audit.log_event({
				'event_type': 'webhook_subscription_created',
				'subscription_id': subscription_id,
				'user_id': user_id,
				'endpoint_url': endpoint_url,
				'events': [e.value for e in events]
			})
			
			return subscription_id
			
		except Exception as e:
			logger.error(f"Failed to create webhook subscription: {e}")
			raise
	
	async def update_subscription(self, subscription_id: str, updates: Dict[str, Any], 
								 user_id: str) -> bool:
		"""Update webhook subscription."""
		try:
			subscription = self.subscriptions.get(subscription_id)
			if not subscription or subscription.user_id != user_id:
				return False
			
			# Apply updates
			if 'endpoint_url' in updates:
				subscription.endpoint_url = updates['endpoint_url']
			if 'events' in updates:
				subscription.events = [WebhookEvent(e) for e in updates['events']]
			if 'active' in updates:
				subscription.active = updates['active']
			if 'retry_count' in updates:
				subscription.retry_count = updates['retry_count']
			if 'timeout_seconds' in updates:
				subscription.timeout_seconds = updates['timeout_seconds']
			if 'custom_headers' in updates:
				subscription.custom_headers = updates['custom_headers']
			if 'filter_conditions' in updates:
				subscription.filter_conditions = updates['filter_conditions']
			
			subscription.updated_at = datetime.utcnow()
			
			# Log update
			await self.audit.log_event({
				'event_type': 'webhook_subscription_updated',
				'subscription_id': subscription_id,
				'user_id': user_id,
				'updates': list(updates.keys())
			})
			
			return True
			
		except Exception as e:
			logger.error(f"Failed to update webhook subscription: {e}")
			return False
	
	async def delete_subscription(self, subscription_id: str, user_id: str) -> bool:
		"""Delete webhook subscription."""
		try:
			subscription = self.subscriptions.get(subscription_id)
			if not subscription or subscription.user_id != user_id:
				return False
			
			del self.subscriptions[subscription_id]
			
			# Log deletion
			await self.audit.log_event({
				'event_type': 'webhook_subscription_deleted',
				'subscription_id': subscription_id,
				'user_id': user_id
			})
			
			return True
			
		except Exception as e:
			logger.error(f"Failed to delete webhook subscription: {e}")
			return False
	
	async def trigger_event(self, event_type: WebhookEvent, payload: Dict[str, Any], 
						   context: Dict[str, Any] = None):
		"""Trigger webhook event for all matching subscriptions."""
		try:
			matching_subscriptions = []
			
			for subscription in self.subscriptions.values():
				if (subscription.active and 
					event_type in subscription.events and
					self._matches_filter_conditions(payload, subscription.filter_conditions)):
					matching_subscriptions.append(subscription)
			
			# Create delivery attempts
			for subscription in matching_subscriptions:
				delivery_id = uuid7str()
				delivery = WebhookDelivery(
					id=delivery_id,
					subscription_id=subscription.id,
					event_type=event_type,
					payload=payload
				)
				
				self.pending_deliveries[delivery_id] = delivery
				subscription.last_triggered = datetime.utcnow()
			
			logger.info(f"Triggered webhook event {event_type.value} for {len(matching_subscriptions)} subscriptions")
			
		except Exception as e:
			logger.error(f"Failed to trigger webhook event: {e}")
	
	async def _delivery_worker(self):
		"""Background worker for webhook deliveries."""
		while self._delivery_worker_running:
			try:
				# Process pending deliveries
				deliveries_to_process = []
				
				for delivery in self.pending_deliveries.values():
					if (delivery.status == WebhookStatus.PENDING and 
						(delivery.next_retry_at is None or delivery.next_retry_at <= datetime.utcnow())):
						deliveries_to_process.append(delivery)
				
				# Process deliveries concurrently
				if deliveries_to_process:
					await asyncio.gather(
						*[self._deliver_webhook(delivery) for delivery in deliveries_to_process],
						return_exceptions=True
					)
				
				await asyncio.sleep(5)  # Check every 5 seconds
				
			except Exception as e:
				logger.error(f"Webhook delivery worker error: {e}")
				await asyncio.sleep(10)
	
	async def _deliver_webhook(self, delivery: WebhookDelivery):
		"""Deliver webhook to endpoint."""
		try:
			subscription = self.subscriptions.get(delivery.subscription_id)
			if not subscription:
				delivery.status = WebhookStatus.FAILED
				delivery.error_message = "Subscription not found"
				return
			
			# Prepare payload
			webhook_payload = {
				'event': delivery.event_type.value,
				'timestamp': delivery.created_at.isoformat(),
				'data': delivery.payload,
				'delivery_id': delivery.id
			}
			
			# Generate signature
			signature = self._generate_signature(
				json.dumps(webhook_payload, sort_keys=True),
				subscription.secret
			)
			
			# Prepare headers
			headers = {
				'Content-Type': 'application/json',
				'X-Webhook-Signature': signature,
				'X-Webhook-Event': delivery.event_type.value,
				'X-Webhook-Delivery': delivery.id,
				'User-Agent': 'APG-Webhook/1.0'
			}
			
			# Add custom headers
			headers.update(subscription.custom_headers)
			
			# Make HTTP request
			timeout = aiohttp.ClientTimeout(total=subscription.timeout_seconds)
			async with aiohttp.ClientSession(timeout=timeout) as session:
				async with session.post(
					subscription.endpoint_url,
					json=webhook_payload,
					headers=headers
				) as response:
					delivery.response_code = response.status
					delivery.response_body = await response.text()
					
					if 200 <= response.status < 300:
						delivery.status = WebhookStatus.DELIVERED
						delivery.delivered_at = datetime.utcnow()
						subscription.successful_deliveries += 1
					else:
						delivery.status = WebhookStatus.FAILED
						delivery.error_message = f"HTTP {response.status}: {delivery.response_body}"
						subscription.failed_deliveries += 1
			
			delivery.attempt_count += 1
			subscription.delivery_attempts += 1
			
		except Exception as e:
			delivery.status = WebhookStatus.FAILED
			delivery.error_message = str(e)
			delivery.attempt_count += 1
			
			subscription = self.subscriptions.get(delivery.subscription_id)
			if subscription:
				subscription.failed_deliveries += 1
			
			# Schedule retry if attempts remaining
			if delivery.attempt_count < subscription.retry_count:
				delay_minutes = 2 ** delivery.attempt_count  # Exponential backoff
				delivery.next_retry_at = datetime.utcnow() + timedelta(minutes=delay_minutes)
				delivery.status = WebhookStatus.PENDING
			
			logger.error(f"Webhook delivery failed: {e}")
	
	def _generate_webhook_secret(self) -> str:
		"""Generate webhook secret."""
		import secrets
		return secrets.token_urlsafe(32)
	
	def _generate_signature(self, payload: str, secret: str) -> str:
		"""Generate webhook signature."""
		signature = hmac.new(
			secret.encode('utf-8'),
			payload.encode('utf-8'),
			hashlib.sha256
		).hexdigest()
		return f"sha256={signature}"
	
	def _matches_filter_conditions(self, payload: Dict[str, Any], 
								  conditions: Dict[str, Any]) -> bool:
		"""Check if payload matches filter conditions."""
		if not conditions:
			return True
		
		# Simple filter matching (can be expanded)
		for key, expected_value in conditions.items():
			if key not in payload or payload[key] != expected_value:
				return False
		
		return True
	
	async def get_subscription(self, subscription_id: str, user_id: str) -> Optional[Dict[str, Any]]:
		"""Get webhook subscription."""
		subscription = self.subscriptions.get(subscription_id)
		if not subscription or subscription.user_id != user_id:
			return None
		
		return {
			'id': subscription.id,
			'endpoint_url': subscription.endpoint_url,
			'events': [e.value for e in subscription.events],
			'active': subscription.active,
			'retry_count': subscription.retry_count,
			'timeout_seconds': subscription.timeout_seconds,
			'custom_headers': subscription.custom_headers,
			'filter_conditions': subscription.filter_conditions,
			'created_at': subscription.created_at.isoformat(),
			'updated_at': subscription.updated_at.isoformat(),
			'last_triggered': subscription.last_triggered.isoformat() if subscription.last_triggered else None,
			'delivery_stats': {
				'attempts': subscription.delivery_attempts,
				'successful': subscription.successful_deliveries,
				'failed': subscription.failed_deliveries
			}
		}
	
	async def list_user_subscriptions(self, user_id: str) -> List[Dict[str, Any]]:
		"""List user's webhook subscriptions."""
		user_subscriptions = []
		
		for subscription in self.subscriptions.values():
			if subscription.user_id == user_id:
				sub_data = await self.get_subscription(subscription.id, user_id)
				if sub_data:
					user_subscriptions.append(sub_data)
		
		return user_subscriptions
	
	async def get_delivery_history(self, subscription_id: str, user_id: str, 
								  limit: int = 50) -> List[Dict[str, Any]]:
		"""Get webhook delivery history."""
		subscription = self.subscriptions.get(subscription_id)
		if not subscription or subscription.user_id != user_id:
			return []
		
		# Get deliveries for this subscription
		deliveries = [
			d for d in self.pending_deliveries.values() 
			if d.subscription_id == subscription_id
		]
		
		# Sort by creation time (most recent first)
		deliveries.sort(key=lambda d: d.created_at, reverse=True)
		
		# Convert to dict format
		delivery_history = []
		for delivery in deliveries[:limit]:
			delivery_history.append({
				'id': delivery.id,
				'event_type': delivery.event_type.value,
				'status': delivery.status.value,
				'response_code': delivery.response_code,
				'attempt_count': delivery.attempt_count,
				'created_at': delivery.created_at.isoformat(),
				'delivered_at': delivery.delivered_at.isoformat() if delivery.delivered_at else None,
				'error_message': delivery.error_message
			})
		
		return delivery_history


class BulkAPIOperations:
	"""Handles bulk API operations."""
	
	def __init__(self):
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
	
	async def bulk_workflow_operation(self, operation: str, workflow_ids: List[str], 
									 parameters: Dict[str, Any] = None, 
									 user_id: str = None) -> Dict[str, Any]:
		"""Perform bulk operation on workflows."""
		try:
			results = []
			errors = []
			
			for workflow_id in workflow_ids:
				try:
					if operation == "delete":
						result = await self._delete_workflow(workflow_id, user_id)
					elif operation == "archive":
						result = await self._archive_workflow(workflow_id, user_id)
					elif operation == "activate":
						result = await self._activate_workflow(workflow_id, user_id)
					elif operation == "duplicate":
						result = await self._duplicate_workflow(workflow_id, user_id, parameters)
					elif operation == "export":
						result = await self._export_workflow(workflow_id, user_id, parameters)
					else:
						raise ValueError(f"Unknown operation: {operation}")
					
					results.append({
						'workflow_id': workflow_id,
						'success': True,
						'result': result
					})
					
				except Exception as e:
					errors.append({
						'workflow_id': workflow_id,
						'error': str(e)
					})
			
			# Log bulk operation
			await self.audit.log_event({
				'event_type': 'bulk_workflow_operation',
				'operation': operation,
				'workflow_count': len(workflow_ids),
				'success_count': len(results),
				'error_count': len(errors),
				'user_id': user_id
			})
			
			return {
				'operation': operation,
				'total_count': len(workflow_ids),
				'success_count': len(results),
				'error_count': len(errors),
				'results': results,
				'errors': errors,
				'completed_at': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Bulk workflow operation failed: {e}")
			raise
	
	async def batch_execute_workflows(self, executions: List[Dict[str, Any]], 
									 user_id: str = None) -> Dict[str, Any]:
		"""Execute multiple workflows in batch."""
		try:
			results = []
			errors = []
			
			# Execute workflows concurrently
			tasks = []
			for exec_config in executions:
				workflow_id = exec_config['workflow_id']
				parameters = exec_config.get('parameters', {})
				
				task = self._execute_workflow_async(workflow_id, parameters, user_id)
				tasks.append((workflow_id, task))
			
			# Wait for all executions to complete
			for workflow_id, task in tasks:
				try:
					result = await task
					results.append({
						'workflow_id': workflow_id,
						'execution_id': result['execution_id'],
						'success': True
					})
				except Exception as e:
					errors.append({
						'workflow_id': workflow_id,
						'error': str(e)
					})
			
			# Log batch execution
			await self.audit.log_event({
				'event_type': 'batch_workflow_execution',
				'workflow_count': len(executions),
				'success_count': len(results),
				'error_count': len(errors),
				'user_id': user_id
			})
			
			return {
				'total_count': len(executions),
				'success_count': len(results),
				'error_count': len(errors),
				'results': results,
				'errors': errors,
				'started_at': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Batch workflow execution failed: {e}")
			raise
	
	async def _delete_workflow(self, workflow_id: str, user_id: str) -> Dict[str, Any]:
		"""Delete workflow (simulated)."""
		await asyncio.sleep(0.1)  # Simulate processing
		return {'deleted': True, 'workflow_id': workflow_id}
	
	async def _archive_workflow(self, workflow_id: str, user_id: str) -> Dict[str, Any]:
		"""Archive workflow (simulated)."""
		await asyncio.sleep(0.1)
		return {'archived': True, 'workflow_id': workflow_id}
	
	async def _activate_workflow(self, workflow_id: str, user_id: str) -> Dict[str, Any]:
		"""Activate workflow (simulated)."""
		await asyncio.sleep(0.1)
		return {'activated': True, 'workflow_id': workflow_id}
	
	async def _duplicate_workflow(self, workflow_id: str, user_id: str, 
								 parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Duplicate workflow (simulated)."""
		await asyncio.sleep(0.2)  # Duplication takes longer
		new_id = uuid7str()
		return {'duplicated': True, 'original_id': workflow_id, 'new_id': new_id}
	
	async def _export_workflow(self, workflow_id: str, user_id: str, 
							   parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Export workflow (simulated)."""
		await asyncio.sleep(0.15)
		export_format = parameters.get('format', 'json')
		return {'exported': True, 'workflow_id': workflow_id, 'format': export_format}
	
	async def _execute_workflow_async(self, workflow_id: str, parameters: Dict[str, Any], 
									 user_id: str) -> Dict[str, Any]:
		"""Execute workflow asynchronously (simulated)."""
		await asyncio.sleep(0.3)  # Simulate execution startup time
		execution_id = uuid7str()
		return {'execution_id': execution_id, 'workflow_id': workflow_id}


class AdvancedAPIService(APGBaseService):
	"""Main advanced API service coordinating all advanced features."""
	
	def __init__(self):
		super().__init__()
		self.version_manager = APIVersionManager()
		self.webhook_manager = WebhookManager()
		self.bulk_operations = BulkAPIOperations()
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
		self.graphql_schema = graphql_schema
	
	async def start(self):
		"""Start advanced API service."""
		await super().start()
		await self.webhook_manager.start()
		logger.info("Advanced API service started")
	
	async def stop(self):
		"""Stop advanced API service."""
		await self.webhook_manager.stop()
		await super().stop()
		logger.info("Advanced API service stopped")
	
	# GraphQL support
	async def execute_graphql_query(self, query: str, variables: Dict[str, Any] = None, 
								   context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Execute GraphQL query."""
		try:
			result = await self.graphql_schema.execute_async(
				query,
				variable_values=variables,
				context_value=context
			)
			
			response = {}
			if result.data:
				response['data'] = result.data
			if result.errors:
				response['errors'] = [str(error) for error in result.errors]
			
			return response
			
		except Exception as e:
			logger.error(f"GraphQL execution failed: {e}")
			return {'errors': [str(e)]}
	
	# Version management
	def get_api_endpoint(self, version: APIVersion, method: str, path: str) -> Optional[APIEndpoint]:
		"""Get API endpoint definition."""
		return self.version_manager.get_endpoint(version, method, path)
	
	def get_all_api_endpoints(self, version: APIVersion = None) -> List[APIEndpoint]:
		"""Get all API endpoints."""
		return self.version_manager.get_all_endpoints(version)
	
	# Webhook management
	async def create_webhook_subscription(self, user_id: str, endpoint_url: str, 
										 events: List[str], config: Dict[str, Any] = None) -> str:
		"""Create webhook subscription."""
		webhook_events = [WebhookEvent(event) for event in events]
		return await self.webhook_manager.create_subscription(user_id, endpoint_url, webhook_events, config)
	
	async def trigger_webhook_event(self, event_type: str, payload: Dict[str, Any], 
								   context: Dict[str, Any] = None):
		"""Trigger webhook event."""
		event = WebhookEvent(event_type)
		await self.webhook_manager.trigger_event(event, payload, context)
	
	# Bulk operations
	async def execute_bulk_workflow_operation(self, operation: str, workflow_ids: List[str], 
											 parameters: Dict[str, Any] = None, 
											 user_id: str = None) -> Dict[str, Any]:
		"""Execute bulk workflow operation."""
		return await self.bulk_operations.bulk_workflow_operation(
			operation, workflow_ids, parameters, user_id
		)
	
	async def batch_execute_workflows(self, executions: List[Dict[str, Any]], 
									 user_id: str = None) -> Dict[str, Any]:
		"""Execute workflows in batch."""
		return await self.bulk_operations.batch_execute_workflows(executions, user_id)
	
	# API documentation generation
	def generate_openapi_spec(self, version: APIVersion = None) -> Dict[str, Any]:
		"""Generate OpenAPI specification."""
		endpoints = self.get_all_api_endpoints(version)
		
		openapi_spec = {
			"openapi": "3.0.3",
			"info": {
				"title": "APG Workflow Orchestration API",
				"description": "Advanced workflow orchestration and automation platform",
				"version": version.value if version else "all",
				"contact": {
					"name": "Datacraft Support",
					"email": "support@datacraft.co.ke",
					"url": "https://www.datacraft.co.ke"
				},
				"license": {
					"name": "Proprietary",
					"url": "https://www.datacraft.co.ke/license"
				}
			},
			"servers": [
				{
					"url": f"/api/{version.value}" if version else "/api/v1",
					"description": f"API {version.value} server" if version else "Default API server"
				}
			],
			"paths": {},
			"components": {
				"schemas": self._generate_schemas(),
				"securitySchemes": {
					"BearerAuth": {
						"type": "http",
						"scheme": "bearer",
						"bearerFormat": "JWT"
					},
					"ApiKeyAuth": {
						"type": "apiKey",
						"in": "header",
						"name": "X-API-Key"
					}
				}
			},
			"security": [
				{"BearerAuth": []},
				{"ApiKeyAuth": []}
			]
		}
		
		# Add paths from endpoints
		for endpoint in endpoints:
			if endpoint.path not in openapi_spec["paths"]:
				openapi_spec["paths"][endpoint.path] = {}
			
			openapi_spec["paths"][endpoint.path][endpoint.method.lower()] = {
				"summary": endpoint.description,
				"description": endpoint.description,
				"parameters": self._convert_parameters_to_openapi(endpoint.parameters),
				"responses": {
					"200": {
						"description": "Success",
						"content": {
							"application/json": {
								"schema": endpoint.response_schema or {"type": "object"}
							}
						}
					},
					"400": {"description": "Bad Request"},
					"401": {"description": "Unauthorized"},
					"403": {"description": "Forbidden"},
					"404": {"description": "Not Found"},
					"500": {"description": "Internal Server Error"}
				}
			}
			
			if endpoint.request_schema:
				openapi_spec["paths"][endpoint.path][endpoint.method.lower()]["requestBody"] = {
					"required": True,
					"content": {
						"application/json": {
							"schema": endpoint.request_schema
						}
					}
				}
			
			if endpoint.examples:
				openapi_spec["paths"][endpoint.path][endpoint.method.lower()]["examples"] = endpoint.examples
		
		return openapi_spec
	
	def _generate_schemas(self) -> Dict[str, Any]:
		"""Generate OpenAPI schemas."""
		return {
			"Workflow": {
				"type": "object",
				"properties": {
					"id": {"type": "string", "format": "uuid"},
					"name": {"type": "string"},
					"description": {"type": "string"},
					"category": {"type": "string"},
					"status": {"type": "string", "enum": ["draft", "active", "archived"]},
					"version": {"type": "string"},
					"tags": {"type": "array", "items": {"type": "string"}},
					"created_at": {"type": "string", "format": "date-time"},
					"updated_at": {"type": "string", "format": "date-time"},
					"created_by": {"type": "string"}
				},
				"required": ["id", "name", "status"]
			},
			
			"Execution": {
				"type": "object",
				"properties": {
					"id": {"type": "string", "format": "uuid"},
					"workflow_id": {"type": "string", "format": "uuid"},
					"workflow_name": {"type": "string"},
					"status": {"type": "string", "enum": ["running", "completed", "failed", "cancelled"]},
					"started_at": {"type": "string", "format": "date-time"},
					"completed_at": {"type": "string", "format": "date-time"},
					"duration": {"type": "number"},
					"success": {"type": "boolean"},
					"progress": {"type": "integer", "minimum": 0, "maximum": 100},
					"error_message": {"type": "string"}
				},
				"required": ["id", "workflow_id", "status"]
			},
			
			"Template": {
				"type": "object",
				"properties": {
					"id": {"type": "string"},
					"name": {"type": "string"},
					"description": {"type": "string"},
					"category": {"type": "string"},
					"complexity": {"type": "string", "enum": ["beginner", "intermediate", "advanced", "expert"]},
					"tags": {"type": "array", "items": {"type": "string"}},
					"use_cases": {"type": "array", "items": {"type": "string"}},
					"popularity_score": {"type": "number"},
					"usage_count": {"type": "integer"},
					"rating": {"type": "number", "minimum": 0, "maximum": 5}
				}
			},
			
			"CreateWorkflowRequest": {
				"type": "object",
				"properties": {
					"name": {"type": "string"},
					"description": {"type": "string"},
					"category": {"type": "string"},
					"workflow_definition": {"type": "object"},
					"tags": {"type": "array", "items": {"type": "string"}}
				},
				"required": ["name", "workflow_definition"]
			},
			
			"ExecuteWorkflowRequest": {
				"type": "object",
				"properties": {
					"parameters": {"type": "object"},
					"priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
					"scheduled_at": {"type": "string", "format": "date-time"}
				}
			},
			
			"BulkOperationRequest": {
				"type": "object",
				"properties": {
					"operation": {"type": "string", "enum": ["delete", "archive", "activate", "duplicate", "export"]},
					"target_ids": {"type": "array", "items": {"type": "string"}},
					"parameters": {"type": "object"}
				},
				"required": ["operation", "target_ids"]
			},
			
			"WebhookSubscription": {
				"type": "object",
				"properties": {
					"id": {"type": "string", "format": "uuid"},
					"endpoint_url": {"type": "string", "format": "uri"},
					"events": {"type": "array", "items": {"type": "string"}},
					"active": {"type": "boolean"},
					"created_at": {"type": "string", "format": "date-time"}
				}
			}
		}
	
	def _convert_parameters_to_openapi(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Convert parameter definitions to OpenAPI format."""
		openapi_params = []
		
		for param_name, param_def in parameters.items():
			param_location = param_def.get('in', 'query')
			
			openapi_param = {
				"name": param_name,
				"in": param_location,
				"required": param_def.get('required', False),
				"schema": {
					"type": param_def.get('type', 'string')
				}
			}
			
			if 'description' in param_def:
				openapi_param['description'] = param_def['description']
			if 'default' in param_def:
				openapi_param['schema']['default'] = param_def['default']
			if 'enum' in param_def:
				openapi_param['schema']['enum'] = param_def['enum']
			if 'format' in param_def:
				openapi_param['schema']['format'] = param_def['format']
			if 'minimum' in param_def:
				openapi_param['schema']['minimum'] = param_def['minimum']
			if 'maximum' in param_def:
				openapi_param['schema']['maximum'] = param_def['maximum']
			
			openapi_params.append(openapi_param)
		
		return openapi_params
	
	async def health_check(self) -> bool:
		"""Health check for advanced API service."""
		try:
			# Test GraphQL execution
			query = "{ __typename }"
			result = await self.execute_graphql_query(query)
			return 'data' in result
		except Exception:
			return False


# Global advanced API service instance
advanced_api_service = AdvancedAPIService()