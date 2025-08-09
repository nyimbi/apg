#!/usr/bin/env python3
"""
APG Workflow Orchestration User Experience Features

Advanced UX features including search/filtering, bulk operations, sharing, 
accessibility, help systems, and user interface enhancements.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field, validator
import difflib

from apg.framework.base_service import APGBaseService
from apg.framework.database import APGDatabase
from apg.framework.audit_compliance import APGAuditLogger

from .config import get_config
from .models import WorkflowStatus, TaskStatus


logger = logging.getLogger(__name__)


class SearchScope(str, Enum):
	"""Search scope options."""
	ALL = "all"
	WORKFLOWS = "workflows"
	TEMPLATES = "templates"
	EXECUTIONS = "executions"
	COMPONENTS = "components"
	DOCUMENTATION = "documentation"


class SortOrder(str, Enum):
	"""Sort order options."""
	ASC = "asc"
	DESC = "desc"


class ShareLevel(str, Enum):
	"""Sharing permission levels."""
	VIEW = "view"
	EDIT = "edit"
	ADMIN = "admin"


class NotificationType(str, Enum):
	"""User notification types."""
	INFO = "info"
	SUCCESS = "success"
	WARNING = "warning"
	ERROR = "error"


@dataclass
class SearchFilter:
	"""Search filter configuration."""
	field: str
	operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains, starts_with, ends_with
	value: Any
	label: Optional[str] = None


@dataclass
class SearchQuery:
	"""Comprehensive search query."""
	query: str = ""
	scope: SearchScope = SearchScope.ALL
	filters: List[SearchFilter] = field(default_factory=list)
	sort_by: str = "updated_at"
	sort_order: SortOrder = SortOrder.DESC
	limit: int = 50
	offset: int = 0
	include_archived: bool = False
	tenant_scope: bool = True


@dataclass
class SearchResult:
	"""Search result item."""
	id: str
	title: str
	description: str
	type: str
	score: float
	highlights: Dict[str, List[str]] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)
	url: Optional[str] = None
	thumbnail: Optional[str] = None
	tags: List[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BulkOperation:
	"""Bulk operation configuration."""
	id: str
	operation: str
	target_ids: List[str]
	parameters: Dict[str, Any] = field(default_factory=dict)
	status: str = "pending"  # pending, running, completed, failed
	progress: int = 0
	total: int = 0
	results: List[Dict[str, Any]] = field(default_factory=list)
	errors: List[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)
	completed_at: Optional[datetime] = None


@dataclass
class ShareConfiguration:
	"""Resource sharing configuration."""
	id: str
	resource_type: str
	resource_id: str
	shared_by: str
	shared_with: List[str] = field(default_factory=list)
	share_level: ShareLevel = ShareLevel.VIEW
	expires_at: Optional[datetime] = None
	public_link: Optional[str] = None
	password_protected: bool = False
	download_allowed: bool = True
	created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserPreference:
	"""User preference setting."""
	key: str
	value: Any
	user_id: str
	category: str = "general"
	description: Optional[str] = None
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HelpContent:
	"""Help system content."""
	id: str
	title: str
	content: str
	category: str
	tags: List[str] = field(default_factory=list)
	difficulty: str = "beginner"  # beginner, intermediate, advanced
	format: str = "markdown"  # markdown, html, video, interactive
	attachments: List[str] = field(default_factory=list)
	related_topics: List[str] = field(default_factory=list)
	views: int = 0
	helpful_votes: int = 0
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


class AdvancedSearchEngine:
	"""Advanced search and filtering engine."""
	
	def __init__(self):
		self.database = APGDatabase()
		self.indexed_fields = {
			'workflows': ['name', 'description', 'tags', 'category'],
			'templates': ['name', 'description', 'use_cases', 'tags'],
			'executions': ['workflow_name', 'status', 'error_message'],
			'components': ['name', 'description', 'type', 'category']
		}
	
	async def search(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
		"""Execute advanced search query."""
		try:
			results = []
			total_count = 0
			
			if query.scope == SearchScope.ALL:
				# Search across all scopes
				for scope in [SearchScope.WORKFLOWS, SearchScope.TEMPLATES, 
							 SearchScope.EXECUTIONS, SearchScope.COMPONENTS]:
					scope_query = SearchQuery(
						query=query.query,
						scope=scope,
						filters=query.filters,
						sort_by=query.sort_by,
						sort_order=query.sort_order,
						limit=query.limit // 4,  # Distribute limit across scopes
						offset=query.offset,
						include_archived=query.include_archived,
						tenant_scope=query.tenant_scope
					)
					scope_results, scope_count = await self._search_scope(scope_query)
					results.extend(scope_results)
					total_count += scope_count
			else:
				results, total_count = await self._search_scope(query)
			
			# Sort results by relevance score
			results.sort(key=lambda r: r.score, reverse=True)
			
			# Apply global limit
			if query.limit > 0:
				results = results[:query.limit]
			
			return results, total_count
			
		except Exception as e:
			logger.error(f"Search failed: {e}")
			return [], 0
	
	async def _search_scope(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
		"""Search within specific scope."""
		if query.scope == SearchScope.WORKFLOWS:
			return await self._search_workflows(query)
		elif query.scope == SearchScope.TEMPLATES:
			return await self._search_templates(query)
		elif query.scope == SearchScope.EXECUTIONS:
			return await self._search_executions(query)
		elif query.scope == SearchScope.COMPONENTS:
			return await self._search_components(query)
		elif query.scope == SearchScope.DOCUMENTATION:
			return await self._search_documentation(query)
		else:
			return [], 0
	
	async def _search_workflows(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
		"""Search workflows."""
		# Build SQL query
		base_query = """
		SELECT id, name, description, tags, category, status, created_at, updated_at
		FROM wo_workflows
		WHERE 1=1
		"""
		
		params = {}
		conditions = []
		
		# Add text search
		if query.query:
			conditions.append("""
			(name ILIKE %(query)s OR description ILIKE %(query)s OR 
			 tags::text ILIKE %(query)s OR category ILIKE %(query)s)
			""")
			params['query'] = f"%{query.query}%"
		
		# Add filters
		for filter_item in query.filters:
			condition, filter_params = self._build_filter_condition(filter_item, "wo_workflows")
			if condition:
				conditions.append(condition)
				params.update(filter_params)
		
		# Add archived filter
		if not query.include_archived:
			conditions.append("status != 'archived'")
		
		# Add tenant scope
		if query.tenant_scope:
			conditions.append("tenant_id = %(tenant_id)s")
			params['tenant_id'] = 'default'  # Get from context
		
		# Combine conditions
		if conditions:
			base_query += " AND " + " AND ".join(conditions)
		
		# Add sorting
		sort_field = self._validate_sort_field(query.sort_by, 'workflows')
		base_query += f" ORDER BY {sort_field} {query.sort_order.value}"
		
		# Add pagination
		base_query += " LIMIT %(limit)s OFFSET %(offset)s"
		params['limit'] = query.limit
		params['offset'] = query.offset
		
		try:
			# Execute query (simulated)
			workflows = await self._execute_search_query(base_query, params)
			
			# Convert to search results
			results = []
			for workflow in workflows:
				score = self._calculate_relevance_score(workflow, query.query, 'workflow')
				highlights = self._generate_highlights(workflow, query.query)
				
				result = SearchResult(
					id=workflow['id'],
					title=workflow['name'],
					description=workflow['description'] or '',
					type='workflow',
					score=score,
					highlights=highlights,
					metadata={
						'category': workflow.get('category'),
						'status': workflow.get('status'),
						'tags': workflow.get('tags', [])
					},
					url=f"/workflow_orchestration/workflows/{workflow['id']}",
					tags=workflow.get('tags', []),
					created_at=workflow['created_at'],
					updated_at=workflow['updated_at']
				)
				results.append(result)
			
			# Get total count
			count_query = base_query.replace(
				"SELECT id, name, description, tags, category, status, created_at, updated_at",
				"SELECT COUNT(*)"
			).split(" LIMIT ")[0]  # Remove LIMIT clause
			
			total_count = len(workflows)  # Simplified for demo
			
			return results, total_count
			
		except Exception as e:
			logger.error(f"Workflow search failed: {e}")
			return [], 0
	
	async def _search_templates(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
		"""Search workflow templates."""
		# Simulated template search
		templates = [
			{
				'id': 'etl_pipeline',
				'name': 'ETL Pipeline',
				'description': 'Extract, Transform, Load data pipeline',
				'category': 'data_processing',
				'tags': ['ETL', 'data', 'pipeline'],
				'use_cases': ['Data migration', 'Data synchronization'],
				'created_at': datetime.utcnow(),
				'updated_at': datetime.utcnow()
			}
		]
		
		results = []
		for template in templates:
			if query.query and query.query.lower() not in template['name'].lower():
				continue
			
			score = self._calculate_relevance_score(template, query.query, 'template')
			highlights = self._generate_highlights(template, query.query)
			
			result = SearchResult(
				id=template['id'],
				title=template['name'],
				description=template['description'],
				type='template',
				score=score,
				highlights=highlights,
				metadata={
					'category': template['category'],
					'use_cases': template['use_cases']
				},
				url=f"/workflow_orchestration/templates/{template['id']}",
				tags=template['tags'],
				created_at=template['created_at'],
				updated_at=template['updated_at']
			)
			results.append(result)
		
		return results, len(results)
	
	async def _search_executions(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
		"""Search workflow executions."""
		# Simulated execution search
		executions = []
		return [], 0
	
	async def _search_components(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
		"""Search workflow components."""
		# Simulated component search
		components = []
		return [], 0
	
	async def _search_documentation(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
		"""Search documentation."""
		# Simulated documentation search
		docs = []
		return [], 0
	
	def _build_filter_condition(self, filter_item: SearchFilter, table: str) -> Tuple[str, Dict[str, Any]]:
		"""Build SQL condition from filter."""
		field = filter_item.field
		operator = filter_item.operator
		value = filter_item.value
		
		param_key = f"filter_{field}_{id(filter_item)}"
		
		if operator == "eq":
			return f"{field} = %({param_key})s", {param_key: value}
		elif operator == "ne":
			return f"{field} != %({param_key})s", {param_key: value}
		elif operator == "gt":
			return f"{field} > %({param_key})s", {param_key: value}
		elif operator == "lt":
			return f"{field} < %({param_key})s", {param_key: value}
		elif operator == "gte":
			return f"{field} >= %({param_key})s", {param_key: value}
		elif operator == "lte":
			return f"{field} <= %({param_key})s", {param_key: value}
		elif operator == "in":
			return f"{field} = ANY(%({param_key})s)", {param_key: value}
		elif operator == "not_in":
			return f"{field} != ALL(%({param_key})s)", {param_key: value}
		elif operator == "contains":
			return f"{field} ILIKE %({param_key})s", {param_key: f"%{value}%"}
		elif operator == "starts_with":
			return f"{field} ILIKE %({param_key})s", {param_key: f"{value}%"}
		elif operator == "ends_with":
			return f"{field} ILIKE %({param_key})s", {param_key: f"%{value}"}
		else:
			return "", {}
	
	def _validate_sort_field(self, field: str, scope: str) -> str:
		"""Validate and return safe sort field."""
		valid_fields = {
			'workflows': ['name', 'created_at', 'updated_at', 'status'],
			'templates': ['name', 'created_at', 'updated_at', 'popularity_score'],
			'executions': ['created_at', 'status', 'duration'],
			'components': ['name', 'type', 'created_at']
		}
		
		if field in valid_fields.get(scope, []):
			return field
		return 'updated_at'  # Default
	
	def _calculate_relevance_score(self, item: Dict[str, Any], query: str, item_type: str) -> float:
		"""Calculate relevance score for search result."""
		if not query:
			return 1.0
		
		score = 0.0
		query_lower = query.lower()
		
		# Title/name match (highest weight)
		name = item.get('name', '').lower()
		if query_lower in name:
			if name == query_lower:
				score += 10.0  # Exact match
			elif name.startswith(query_lower):
				score += 8.0   # Starts with
			else:
				score += 5.0   # Contains
		
		# Description match
		description = item.get('description', '').lower()
		if query_lower in description:
			score += 3.0
		
		# Tags match
		tags = item.get('tags', [])
		for tag in tags:
			if query_lower in tag.lower():
				score += 2.0
		
		# Additional scoring based on item type
		if item_type == 'template':
			use_cases = item.get('use_cases', [])
			for use_case in use_cases:
				if query_lower in use_case.lower():
					score += 1.5
		
		# Normalize score
		return min(score, 10.0)
	
	def _generate_highlights(self, item: Dict[str, Any], query: str) -> Dict[str, List[str]]:
		"""Generate search result highlights."""
		if not query:
			return {}
		
		highlights = {}
		query_lower = query.lower()
		
		# Highlight in name
		name = item.get('name', '')
		if query_lower in name.lower():
			highlighted = self._highlight_text(name, query)
			highlights['name'] = [highlighted]
		
		# Highlight in description
		description = item.get('description', '')
		if query_lower in description.lower():
			highlighted = self._highlight_text(description, query)
			highlights['description'] = [highlighted]
		
		return highlights
	
	def _highlight_text(self, text: str, query: str, max_length: int = 200) -> str:
		"""Highlight query terms in text."""
		if not query or not text:
			return text
		
		# Find query position
		query_lower = query.lower()
		text_lower = text.lower()
		
		start_pos = text_lower.find(query_lower)
		if start_pos == -1:
			return text[:max_length] + "..." if len(text) > max_length else text
		
		# Calculate excerpt bounds
		excerpt_start = max(0, start_pos - 50)
		excerpt_end = min(len(text), start_pos + len(query) + 50)
		
		excerpt = text[excerpt_start:excerpt_end]
		
		# Add ellipsis if truncated
		if excerpt_start > 0:
			excerpt = "..." + excerpt
		if excerpt_end < len(text):
			excerpt = excerpt + "..."
		
		# Highlight the query term
		highlighted = re.sub(
			re.escape(query), 
			f"<mark>{query}</mark>", 
			excerpt, 
			flags=re.IGNORECASE
		)
		
		return highlighted
	
	async def _execute_search_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Execute search query against database."""
		# Simulated database query execution
		# In real implementation, this would use the actual database
		return [
			{
				'id': uuid7str(),
				'name': 'Sample Workflow',
				'description': 'A sample workflow for demonstration',
				'tags': ['sample', 'demo'],
				'category': 'automation',
				'status': 'active',
				'created_at': datetime.utcnow(),
				'updated_at': datetime.utcnow()
			}
		]


class BulkOperationsManager:
	"""Manages bulk operations on workflows and resources."""
	
	def __init__(self):
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
		self.active_operations: Dict[str, BulkOperation] = {}
	
	async def start_bulk_operation(self, operation: str, target_ids: List[str], 
								  parameters: Dict[str, Any] = None, 
								  user_id: str = None) -> str:
		"""Start a new bulk operation."""
		try:
			operation_id = uuid7str()
			bulk_op = BulkOperation(
				id=operation_id,
				operation=operation,
				target_ids=target_ids,
				parameters=parameters or {},
				total=len(target_ids)
			)
			
			self.active_operations[operation_id] = bulk_op
			
			# Start operation in background
			asyncio.create_task(self._execute_bulk_operation(bulk_op, user_id))
			
			# Log operation start
			await self.audit.log_event({
				'event_type': 'bulk_operation_started',
				'operation_id': operation_id,
				'operation': operation,
				'target_count': len(target_ids),
				'user_id': user_id
			})
			
			return operation_id
			
		except Exception as e:
			logger.error(f"Failed to start bulk operation: {e}")
			raise
	
	async def _execute_bulk_operation(self, bulk_op: BulkOperation, user_id: str = None):
		"""Execute bulk operation."""
		try:
			bulk_op.status = "running"
			
			if bulk_op.operation == "delete":
				await self._bulk_delete(bulk_op)
			elif bulk_op.operation == "archive":
				await self._bulk_archive(bulk_op)
			elif bulk_op.operation == "activate":
				await self._bulk_activate(bulk_op)
			elif bulk_op.operation == "duplicate":
				await self._bulk_duplicate(bulk_op)
			elif bulk_op.operation == "export":
				await self._bulk_export(bulk_op)
			elif bulk_op.operation == "tag":
				await self._bulk_tag(bulk_op)
			elif bulk_op.operation == "move_category":
				await self._bulk_move_category(bulk_op)
			else:
				raise ValueError(f"Unknown bulk operation: {bulk_op.operation}")
			
			bulk_op.status = "completed"
			bulk_op.completed_at = datetime.utcnow()
			
			# Log completion
			await self.audit.log_event({
				'event_type': 'bulk_operation_completed',
				'operation_id': bulk_op.id,
				'operation': bulk_op.operation,
				'success_count': len(bulk_op.results),
				'error_count': len(bulk_op.errors),
				'user_id': user_id
			})
			
		except Exception as e:
			bulk_op.status = "failed"
			bulk_op.errors.append(str(e))
			bulk_op.completed_at = datetime.utcnow()
			logger.error(f"Bulk operation {bulk_op.id} failed: {e}")
	
	async def _bulk_delete(self, bulk_op: BulkOperation):
		"""Execute bulk delete operation."""
		for i, target_id in enumerate(bulk_op.target_ids):
			try:
				# Simulate deletion
				await asyncio.sleep(0.1)  # Simulate processing time
				
				bulk_op.results.append({
					'id': target_id,
					'action': 'deleted',
					'success': True
				})
				
			except Exception as e:
				bulk_op.errors.append(f"Failed to delete {target_id}: {str(e)}")
				bulk_op.results.append({
					'id': target_id,
					'action': 'delete',
					'success': False,
					'error': str(e)
				})
			
			bulk_op.progress = i + 1
	
	async def _bulk_archive(self, bulk_op: BulkOperation):
		"""Execute bulk archive operation."""
		for i, target_id in enumerate(bulk_op.target_ids):
			try:
				# Simulate archiving
				await asyncio.sleep(0.1)
				
				bulk_op.results.append({
					'id': target_id,
					'action': 'archived',
					'success': True
				})
				
			except Exception as e:
				bulk_op.errors.append(f"Failed to archive {target_id}: {str(e)}")
				bulk_op.results.append({
					'id': target_id,
					'action': 'archive',
					'success': False,
					'error': str(e)
				})
			
			bulk_op.progress = i + 1
	
	async def _bulk_activate(self, bulk_op: BulkOperation):
		"""Execute bulk activate operation."""
		for i, target_id in enumerate(bulk_op.target_ids):
			try:
				# Simulate activation
				await asyncio.sleep(0.1)
				
				bulk_op.results.append({
					'id': target_id,
					'action': 'activated',
					'success': True
				})
				
			except Exception as e:
				bulk_op.errors.append(f"Failed to activate {target_id}: {str(e)}")
				bulk_op.results.append({
					'id': target_id,
					'action': 'activate',
					'success': False,
					'error': str(e)
				})
			
			bulk_op.progress = i + 1
	
	async def _bulk_duplicate(self, bulk_op: BulkOperation):
		"""Execute bulk duplicate operation."""
		for i, target_id in enumerate(bulk_op.target_ids):
			try:
				# Simulate duplication
				await asyncio.sleep(0.2)  # Duplication takes longer
				
				new_id = uuid7str()
				bulk_op.results.append({
					'id': target_id,
					'action': 'duplicated',
					'success': True,
					'new_id': new_id
				})
				
			except Exception as e:
				bulk_op.errors.append(f"Failed to duplicate {target_id}: {str(e)}")
				bulk_op.results.append({
					'id': target_id,
					'action': 'duplicate',
					'success': False,
					'error': str(e)
				})
			
			bulk_op.progress = i + 1
	
	async def _bulk_export(self, bulk_op: BulkOperation):
		"""Execute bulk export operation."""
		export_format = bulk_op.parameters.get('format', 'json')
		
		try:
			# Simulate export process
			await asyncio.sleep(1.0)  # Export takes time
			
			export_file = f"bulk_export_{bulk_op.id}.{export_format}"
			
			bulk_op.results.append({
				'action': 'exported',
				'success': True,
				'export_file': export_file,
				'exported_count': len(bulk_op.target_ids)
			})
			
		except Exception as e:
			bulk_op.errors.append(f"Export failed: {str(e)}")
		
		bulk_op.progress = bulk_op.total
	
	async def _bulk_tag(self, bulk_op: BulkOperation):
		"""Execute bulk tagging operation."""
		tags_to_add = bulk_op.parameters.get('add_tags', [])
		tags_to_remove = bulk_op.parameters.get('remove_tags', [])
		
		for i, target_id in enumerate(bulk_op.target_ids):
			try:
				# Simulate tagging
				await asyncio.sleep(0.05)
				
				actions = []
				if tags_to_add:
					actions.append(f"added tags: {', '.join(tags_to_add)}")
				if tags_to_remove:
					actions.append(f"removed tags: {', '.join(tags_to_remove)}")
				
				bulk_op.results.append({
					'id': target_id,
					'action': 'tagged',
					'success': True,
					'changes': actions
				})
				
			except Exception as e:
				bulk_op.errors.append(f"Failed to tag {target_id}: {str(e)}")
				bulk_op.results.append({
					'id': target_id,
					'action': 'tag',
					'success': False,
					'error': str(e)
				})
			
			bulk_op.progress = i + 1
	
	async def _bulk_move_category(self, bulk_op: BulkOperation):
		"""Execute bulk category move operation."""
		new_category = bulk_op.parameters.get('category')
		if not new_category:
			raise ValueError("Category parameter is required")
		
		for i, target_id in enumerate(bulk_op.target_ids):
			try:
				# Simulate category change
				await asyncio.sleep(0.05)
				
				bulk_op.results.append({
					'id': target_id,
					'action': 'moved_category',
					'success': True,
					'new_category': new_category
				})
				
			except Exception as e:
				bulk_op.errors.append(f"Failed to move {target_id}: {str(e)}")
				bulk_op.results.append({
					'id': target_id,
					'action': 'move_category',
					'success': False,
					'error': str(e)
				})
			
			bulk_op.progress = i + 1
	
	async def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
		"""Get bulk operation status."""
		bulk_op = self.active_operations.get(operation_id)
		if not bulk_op:
			return None
		
		return {
			'id': bulk_op.id,
			'operation': bulk_op.operation,
			'status': bulk_op.status,
			'progress': bulk_op.progress,
			'total': bulk_op.total,
			'progress_percent': (bulk_op.progress / bulk_op.total * 100) if bulk_op.total > 0 else 0,
			'success_count': len([r for r in bulk_op.results if r.get('success', False)]),
			'error_count': len(bulk_op.errors),
			'errors': bulk_op.errors,
			'created_at': bulk_op.created_at.isoformat(),
			'completed_at': bulk_op.completed_at.isoformat() if bulk_op.completed_at else None
		}
	
	async def cancel_operation(self, operation_id: str) -> bool:
		"""Cancel a running bulk operation."""
		bulk_op = self.active_operations.get(operation_id)
		if bulk_op and bulk_op.status == "running":
			bulk_op.status = "cancelled"
			bulk_op.completed_at = datetime.utcnow()
			return True
		return False
	
	def cleanup_completed_operations(self, older_than_hours: int = 24):
		"""Clean up completed operations older than specified hours."""
		cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
		
		to_remove = []
		for op_id, bulk_op in self.active_operations.items():
			if (bulk_op.status in ["completed", "failed", "cancelled"] and 
				bulk_op.completed_at and bulk_op.completed_at < cutoff_time):
				to_remove.append(op_id)
		
		for op_id in to_remove:
			del self.active_operations[op_id]
		
		logger.info(f"Cleaned up {len(to_remove)} completed bulk operations")


class SharingManager:
	"""Manages resource sharing and collaboration features."""
	
	def __init__(self):
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
		self.shares: Dict[str, ShareConfiguration] = {}
	
	async def create_share(self, resource_type: str, resource_id: str, 
						  shared_by: str, share_config: Dict[str, Any]) -> str:
		"""Create a new resource share."""
		try:
			share_id = uuid7str()
			
			share = ShareConfiguration(
				id=share_id,
				resource_type=resource_type,
				resource_id=resource_id,
				shared_by=shared_by,
				shared_with=share_config.get('shared_with', []),
				share_level=ShareLevel(share_config.get('share_level', ShareLevel.VIEW)),
				expires_at=share_config.get('expires_at'),
				password_protected=share_config.get('password_protected', False),
				download_allowed=share_config.get('download_allowed', True)
			)
			
			# Generate public link if requested
			if share_config.get('create_public_link', False):
				share.public_link = f"/shared/{share_id}"
			
			self.shares[share_id] = share
			
			# Log share creation
			await self.audit.log_event({
				'event_type': 'share_created',
				'share_id': share_id,
				'resource_type': resource_type,
				'resource_id': resource_id,
				'shared_by': shared_by,
				'share_level': share.share_level.value,
				'public_link': bool(share.public_link)
			})
			
			return share_id
			
		except Exception as e:
			logger.error(f"Failed to create share: {e}")
			raise
	
	async def get_share(self, share_id: str) -> Optional[Dict[str, Any]]:
		"""Get share configuration."""
		share = self.shares.get(share_id)
		if not share:
			return None
		
		# Check expiration
		if share.expires_at and datetime.utcnow() > share.expires_at:
			return None
		
		return {
			'id': share.id,
			'resource_type': share.resource_type,
			'resource_id': share.resource_id,
			'shared_by': share.shared_by,
			'shared_with': share.shared_with,
			'share_level': share.share_level.value,
			'expires_at': share.expires_at.isoformat() if share.expires_at else None,
			'public_link': share.public_link,
			'password_protected': share.password_protected,
			'download_allowed': share.download_allowed,
			'created_at': share.created_at.isoformat()
		}
	
	async def update_share(self, share_id: str, updates: Dict[str, Any], 
						  user_id: str) -> bool:
		"""Update share configuration."""
		try:
			share = self.shares.get(share_id)
			if not share:
				return False
			
			# Check permissions
			if share.shared_by != user_id:
				return False
			
			# Apply updates
			if 'shared_with' in updates:
				share.shared_with = updates['shared_with']
			if 'share_level' in updates:
				share.share_level = ShareLevel(updates['share_level'])
			if 'expires_at' in updates:
				share.expires_at = updates['expires_at']
			if 'password_protected' in updates:
				share.password_protected = updates['password_protected']
			if 'download_allowed' in updates:
				share.download_allowed = updates['download_allowed']
			
			# Log update
			await self.audit.log_event({
				'event_type': 'share_updated',
				'share_id': share_id,
				'updated_by': user_id,
				'updates': list(updates.keys())
			})
			
			return True
			
		except Exception as e:
			logger.error(f"Failed to update share: {e}")
			return False
	
	async def revoke_share(self, share_id: str, user_id: str) -> bool:
		"""Revoke a resource share."""
		try:
			share = self.shares.get(share_id)
			if not share:
				return False
			
			# Check permissions
			if share.shared_by != user_id:
				return False
			
			# Remove share
			del self.shares[share_id]
			
			# Log revocation
			await self.audit.log_event({
				'event_type': 'share_revoked',
				'share_id': share_id,
				'revoked_by': user_id
			})
			
			return True
			
		except Exception as e:
			logger.error(f"Failed to revoke share: {e}")
			return False
	
	async def check_access(self, resource_type: str, resource_id: str, 
						  user_id: str) -> Optional[ShareLevel]:
		"""Check user access level to a shared resource."""
		for share in self.shares.values():
			if (share.resource_type == resource_type and 
				share.resource_id == resource_id):
				
				# Check expiration
				if share.expires_at and datetime.utcnow() > share.expires_at:
					continue
				
				# Check if user has access
				if (user_id == share.shared_by or 
					user_id in share.shared_with or 
					share.public_link):
					return share.share_level
		
		return None
	
	async def list_user_shares(self, user_id: str, resource_type: str = None) -> List[Dict[str, Any]]:
		"""List shares created by or shared with a user."""
		user_shares = []
		
		for share in self.shares.values():
			# Check if user is involved in this share
			if (user_id == share.shared_by or user_id in share.shared_with):
				
				# Filter by resource type if specified
				if resource_type and share.resource_type != resource_type:
					continue
				
				# Check expiration
				if share.expires_at and datetime.utcnow() > share.expires_at:
					continue
				
				share_info = {
					'id': share.id,
					'resource_type': share.resource_type,
					'resource_id': share.resource_id,
					'share_level': share.share_level.value,
					'is_owner': user_id == share.shared_by,
					'expires_at': share.expires_at.isoformat() if share.expires_at else None,
					'created_at': share.created_at.isoformat()
				}
				
				user_shares.append(share_info)
		
		return user_shares


class UserPreferencesManager:
	"""Manages user preferences and settings."""
	
	def __init__(self):
		self.database = APGDatabase()
		self.preferences: Dict[str, Dict[str, UserPreference]] = {}
	
	async def get_user_preferences(self, user_id: str, category: str = None) -> Dict[str, Any]:
		"""Get user preferences."""
		user_prefs = self.preferences.get(user_id, {})
		
		if category:
			return {
				key: pref.value for key, pref in user_prefs.items() 
				if pref.category == category
			}
		else:
			return {key: pref.value for key, pref in user_prefs.items()}
	
	async def set_user_preference(self, user_id: str, key: str, value: Any, 
								 category: str = "general", description: str = None):
		"""Set a user preference."""
		if user_id not in self.preferences:
			self.preferences[user_id] = {}
		
		preference = UserPreference(
			key=key,
			value=value,
			user_id=user_id,
			category=category,
			description=description
		)
		
		self.preferences[user_id][key] = preference
	
	async def get_default_preferences(self) -> Dict[str, Any]:
		"""Get default preferences for new users."""
		return {
			# UI Preferences
			'theme': 'light',
			'sidebar_collapsed': False,
			'dashboard_layout': 'grid',
			'items_per_page': 25,
			'auto_refresh': True,
			'refresh_interval': 30,
			
			# Workflow Preferences
			'default_workflow_category': 'automation',
			'auto_save_interval': 300,  # 5 minutes
			'show_grid': True,
			'snap_to_grid': True,
			'grid_size': 20,
			
			# Notification Preferences
			'email_notifications': True,
			'browser_notifications': True,
			'workflow_completion_notifications': True,
			'workflow_failure_notifications': True,
			'share_notifications': True,
			
			# Advanced Preferences
			'advanced_mode': False,
			'show_performance_metrics': False,
			'debug_mode': False,
			'experimental_features': False
		}


class HelpSystem:
	"""Comprehensive help and documentation system."""
	
	def __init__(self):
		self.help_content: Dict[str, HelpContent] = {}
		self._create_default_content()
	
	def _create_default_content(self):
		"""Create default help content."""
		help_topics = [
			HelpContent(
				id="getting_started",
				title="Getting Started with Workflow Orchestration",
				content="""
# Getting Started

Welcome to APG Workflow Orchestration! This guide will help you create your first workflow.

## Creating Your First Workflow

1. Navigate to the **Workflows** section
2. Click **Create New Workflow**
3. Choose a template or start from scratch
4. Use the drag-and-drop canvas to design your workflow
5. Configure component properties
6. Save and test your workflow

## Key Concepts

- **Components**: Building blocks of workflows (tasks, decisions, loops, etc.)
- **Connections**: Links between components that define execution flow
- **Templates**: Pre-built workflows for common use cases
- **Executions**: Runtime instances of workflows

## Next Steps

- Explore the [Component Library](#component_library)
- Learn about [Advanced Features](#advanced_features)
- Check out [Templates](#templates) for inspiration
				""",
				category="basics",
				tags=["getting started", "tutorial", "basics"],
				difficulty="beginner"
			),
			
			HelpContent(
				id="component_library",
				title="Component Library Reference",
				content="""
# Component Library

The component library provides pre-built components for common workflow tasks.

## Basic Components

### Start Component
- Marks the beginning of a workflow
- Can be triggered manually, on schedule, or by events

### End Component  
- Marks the end of a workflow
- Can perform cleanup actions

### Task Component
- Generic processing component
- Supports data processing, validation, and transformation

## Flow Control Components

### Decision Component
- Conditional logic for workflow branching
- Supports complex expressions and conditions

### Loop Component
- Iterative execution with for, while, and foreach loops
- Configurable iteration limits and conditions

## Integration Components

### HTTP Request Component
- Make API calls to external services
- Supports all HTTP methods and authentication

### Database Query Component
- Execute database queries
- Supports multiple database types

### Email Component
- Send email notifications
- Supports HTML and plain text formats

## Advanced Components

### Script Component
- Execute custom Python or JavaScript code
- Sandboxed execution environment

### ML Prediction Component
- Machine learning predictions
- Supports both local and remote models
				""",
				category="reference",
				tags=["components", "reference", "library"],
				difficulty="intermediate"
			),
			
			HelpContent(
				id="templates",
				title="Workflow Templates Guide",
				content="""
# Workflow Templates

Templates provide pre-built workflows for common business processes.

## Available Templates

### Data Processing
- **ETL Pipeline**: Extract, transform, and load data
- **Data Quality Assessment**: Validate and report data quality
- **File Processing Automation**: Automated file processing

### Business Processes
- **Approval Workflow**: Multi-step approval processes
- **Order Processing**: E-commerce order fulfillment
- **Customer Onboarding**: Automated customer setup

### Integration
- **API Synchronization**: Sync data between systems
- **Database Migration**: Move data between databases
- **System Integration**: Connect multiple systems

### Analytics
- **Sales Analytics**: Generate sales reports and insights
- **Performance Monitoring**: Monitor system performance
- **Audit Reporting**: Generate compliance reports

## Using Templates

1. Browse available templates by category
2. Preview template components and flow
3. Customize parameters for your needs
4. Deploy and test the workflow
5. Modify as needed for your specific requirements

## Creating Custom Templates

1. Design and test your workflow
2. Add parameter placeholders
3. Document use cases and configuration
4. Share with your team or community
				""",
				category="templates",
				tags=["templates", "examples", "business processes"],
				difficulty="beginner"
			),
			
			HelpContent(
				id="advanced_features",
				title="Advanced Features Guide",
				content="""
# Advanced Features

Explore powerful features for complex workflow scenarios.

## Real-time Collaboration

- Multiple users can edit workflows simultaneously
- Live cursors and change indicators
- Conflict resolution and version control
- Comments and annotations

## Monitoring and Analytics

- Real-time workflow execution monitoring
- Performance metrics and dashboards
- Custom alerts and notifications
- Historical analysis and reporting

## Security Features

- Role-based access control
- Workflow and resource sharing
- Audit logging and compliance
- Data encryption and privacy

## Integration Capabilities

- Native APG capability connectors
- External system integrations
- Custom connector development
- Webhook and event-driven workflows

## Automation Features

- Intelligent workflow routing
- Predictive failure detection
- Self-healing mechanisms
- Resource optimization

## Performance Optimization

- Workflow execution optimization
- Bottleneck detection and resolution
- Resource allocation strategies
- Scalability enhancements
				""",
				category="advanced",
				tags=["advanced", "features", "collaboration", "monitoring"],
				difficulty="advanced"
			)
		]
		
		for content in help_topics:
			self.help_content[content.id] = content
	
	async def search_help(self, query: str, category: str = None) -> List[Dict[str, Any]]:
		"""Search help content."""
		results = []
		query_lower = query.lower()
		
		for content in self.help_content.values():
			# Filter by category if specified
			if category and content.category != category:
				continue
			
			# Calculate relevance score
			score = 0.0
			
			# Title match
			if query_lower in content.title.lower():
				score += 10.0
			
			# Content match
			if query_lower in content.content.lower():
				score += 5.0
			
			# Tags match
			for tag in content.tags:
				if query_lower in tag.lower():
					score += 3.0
			
			if score > 0:
				results.append({
					'id': content.id,
					'title': content.title,
					'category': content.category,
					'difficulty': content.difficulty,
					'tags': content.tags,
					'score': score,
					'excerpt': content.content[:200] + "..." if len(content.content) > 200 else content.content
				})
		
		# Sort by relevance
		results.sort(key=lambda x: x['score'], reverse=True)
		return results
	
	async def get_help_content(self, content_id: str) -> Optional[Dict[str, Any]]:
		"""Get specific help content."""
		content = self.help_content.get(content_id)
		if not content:
			return None
		
		# Increment view count
		content.views += 1
		
		return {
			'id': content.id,
			'title': content.title,
			'content': content.content,
			'category': content.category,
			'tags': content.tags,
			'difficulty': content.difficulty,
			'format': content.format,
			'related_topics': content.related_topics,
			'views': content.views,
			'helpful_votes': content.helpful_votes,
			'created_at': content.created_at.isoformat(),
			'updated_at': content.updated_at.isoformat()
		}
	
	async def vote_helpful(self, content_id: str, helpful: bool) -> bool:
		"""Vote on help content helpfulness."""
		content = self.help_content.get(content_id)
		if not content:
			return False
		
		if helpful:
			content.helpful_votes += 1
		
		return True
	
	async def get_help_categories(self) -> List[Dict[str, Any]]:
		"""Get help content categories."""
		categories = {}
		
		for content in self.help_content.values():
			if content.category not in categories:
				categories[content.category] = {
					'name': content.category.replace('_', ' ').title(),
					'count': 0,
					'topics': []
				}
			
			categories[content.category]['count'] += 1
			categories[content.category]['topics'].append({
				'id': content.id,
				'title': content.title,
				'difficulty': content.difficulty
			})
		
		return list(categories.values())


class UserExperienceService(APGBaseService):
	"""Main user experience service coordinating all UX features."""
	
	def __init__(self):
		super().__init__()
		self.search_engine = AdvancedSearchEngine()
		self.bulk_operations = BulkOperationsManager()
		self.sharing_manager = SharingManager()
		self.preferences_manager = UserPreferencesManager()
		self.help_system = HelpSystem()
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
	
	async def start(self):
		"""Start user experience service."""
		await super().start()
		
		# Schedule cleanup tasks
		asyncio.create_task(self._periodic_cleanup())
		
		logger.info("User experience service started")
	
	async def _periodic_cleanup(self):
		"""Periodic cleanup of expired resources."""
		while True:
			try:
				# Clean up completed bulk operations
				self.bulk_operations.cleanup_completed_operations()
				
				# Clean up expired shares
				# (Implementation would go in sharing_manager)
				
				await asyncio.sleep(3600)  # Run every hour
				
			except Exception as e:
				logger.error(f"Cleanup task error: {e}")
				await asyncio.sleep(3600)
	
	# Search functionality
	async def search(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
		"""Execute advanced search."""
		return await self.search_engine.search(query)
	
	# Bulk operations
	async def start_bulk_operation(self, operation: str, target_ids: List[str], 
								  parameters: Dict[str, Any] = None, 
								  user_id: str = None) -> str:
		"""Start bulk operation."""
		return await self.bulk_operations.start_bulk_operation(
			operation, target_ids, parameters, user_id
		)
	
	async def get_bulk_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
		"""Get bulk operation status."""
		return await self.bulk_operations.get_operation_status(operation_id)
	
	# Sharing functionality
	async def create_share(self, resource_type: str, resource_id: str, 
						  shared_by: str, share_config: Dict[str, Any]) -> str:
		"""Create resource share."""
		return await self.sharing_manager.create_share(
			resource_type, resource_id, shared_by, share_config
		)
	
	async def get_share(self, share_id: str) -> Optional[Dict[str, Any]]:
		"""Get share configuration."""
		return await self.sharing_manager.get_share(share_id)
	
	# User preferences
	async def get_user_preferences(self, user_id: str, category: str = None) -> Dict[str, Any]:
		"""Get user preferences."""
		return await self.preferences_manager.get_user_preferences(user_id, category)
	
	async def set_user_preference(self, user_id: str, key: str, value: Any, 
								 category: str = "general"):
		"""Set user preference."""
		await self.preferences_manager.set_user_preference(user_id, key, value, category)
	
	# Help system
	async def search_help(self, query: str, category: str = None) -> List[Dict[str, Any]]:
		"""Search help content."""
		return await self.help_system.search_help(query, category)
	
	async def get_help_content(self, content_id: str) -> Optional[Dict[str, Any]]:
		"""Get help content."""
		return await self.help_system.get_help_content(content_id)
	
	async def health_check(self) -> bool:
		"""Health check for user experience service."""
		try:
			# Test search functionality
			query = SearchQuery(query="test", limit=1)
			results, count = await self.search(query)
			return True
		except Exception:
			return False


# Global user experience service instance
user_experience_service = UserExperienceService()