"""
APG Budgeting & Forecasting - Advanced Budget Management

Enterprise-grade budget management with template support, versioning, 
approval workflows, and real-time collaboration features.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
from uuid import UUID
import json
import logging
from abc import ABC, abstractmethod

import asyncpg
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict

from .models import (
	BFBudget, BFBudgetLine, APGBaseModel,
	BFBudgetType, BFBudgetStatus, BFLineType, BFApprovalStatus,
	PositiveAmount, CurrencyCode, FiscalYear, NonEmptyString
)
from .service import APGTenantContext, BFServiceConfig, ServiceResponse, APGServiceBase


# =============================================================================
# Budget Template Management
# =============================================================================

class BudgetTemplate(BaseModel):
	"""Budget template model for reusable budget structures."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	template_id: str = Field(...)
	template_name: NonEmptyString = Field(..., max_length=255)
	template_description: Optional[str] = Field(None)
	template_category: str = Field(..., max_length=100)  # annual, quarterly, project, department
	
	# Template configuration
	is_public: bool = Field(default=False)
	is_system: bool = Field(default=False)
	owner_tenant_id: str = Field(...)
	shared_with_tenants: List[str] = Field(default_factory=list)
	usage_count: int = Field(default=0, ge=0)
	
	# Template structure
	template_data: Dict[str, Any] = Field(...)
	line_items_template: List[Dict[str, Any]] = Field(default_factory=list)
	default_settings: Dict[str, Any] = Field(default_factory=dict)
	
	# Customization options
	customizable_fields: List[str] = Field(default_factory=list)
	required_fields: List[str] = Field(default_factory=list)
	field_validations: Dict[str, Any] = Field(default_factory=dict)
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(...)
	tags: List[str] = Field(default_factory=list)


class BudgetVersion(BaseModel):
	"""Budget version model for version control and history tracking."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	version_id: str = Field(...)
	budget_id: str = Field(...)
	version_number: int = Field(..., ge=1)
	version_name: Optional[str] = Field(None, max_length=100)
	
	# Version metadata
	is_current: bool = Field(default=True)
	is_baseline: bool = Field(default=False)
	is_archived: bool = Field(default=False)
	
	# Change tracking
	change_summary: Optional[str] = Field(None)
	changes_made: List[Dict[str, Any]] = Field(default_factory=list)
	affected_line_items: List[str] = Field(default_factory=list)
	
	# Version data snapshot
	budget_snapshot: Dict[str, Any] = Field(...)
	line_items_snapshot: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Approval and workflow
	requires_approval: bool = Field(default=True)
	approval_status: BFApprovalStatus = Field(default=BFApprovalStatus.PENDING)
	approved_by: Optional[str] = Field(None)
	approval_date: Optional[datetime] = Field(None)
	approval_notes: Optional[str] = Field(None)
	
	# Timestamps
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(...)


class BudgetCollaboration(BaseModel):
	"""Budget collaboration model for real-time collaborative editing."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	collaboration_id: str = Field(...)
	budget_id: str = Field(...)
	session_id: str = Field(...)
	
	# Participant information
	user_id: str = Field(...)
	user_name: str = Field(...)
	user_role: str = Field(..., max_length=50)
	
	# Activity tracking
	activity_type: str = Field(...)  # edit, comment, review, approve
	target_section: Optional[str] = Field(None)  # line_item, total, metadata
	target_line_id: Optional[str] = Field(None)
	
	# Change details
	field_changed: Optional[str] = Field(None)
	old_value: Optional[str] = Field(None)
	new_value: Optional[str] = Field(None)
	change_reason: Optional[str] = Field(None)
	
	# Comments and discussion
	comment_text: Optional[str] = Field(None)
	reply_to_comment_id: Optional[str] = Field(None)
	mentions: List[str] = Field(default_factory=list)  # @user mentions
	
	# Conflict resolution
	has_conflict: bool = Field(default=False)
	conflict_with_user: Optional[str] = Field(None)
	conflict_resolution: Optional[str] = Field(None)
	
	# Timestamps
	activity_timestamp: datetime = Field(default_factory=datetime.utcnow)
	last_seen_timestamp: Optional[datetime] = Field(None)


# =============================================================================
# Advanced Budget Management Service
# =============================================================================

class AdvancedBudgetService(APGServiceBase):
	"""
	Advanced budget management service with template support,
	versioning, approval workflows, and real-time collaboration.
	"""
	
	async def create_budget_from_template(self, template_id: str, budget_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a new budget from a template with customization options."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.create_from_template'):
				raise PermissionError("Insufficient permissions to create budget from template")
			
			# Get template
			template = await self._get_budget_template(template_id)
			if not template:
				raise ValueError(f"Template {template_id} not found")
			
			# Check template access permissions
			if not await self._can_access_template(template, self.context.tenant_id):
				raise PermissionError("No access to this template")
			
			# Merge template data with provided budget data
			merged_data = await self._merge_template_data(template, budget_data)
			
			# Inject tenant and user context
			merged_data.update({
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id,
				'template_id': template_id
			})
			
			# Create budget model
			budget = BFBudget(**merged_data)
			
			# Generate unique budget code if not provided
			if not budget.budget_code:
				budget.budget_code = await self._generate_budget_code(budget.fiscal_year)
			
			# Start database transaction
			async with self._connection.transaction():
				# Save budget
				budget_id = await self._insert_budget(budget)
				
				# Create initial version
				await self._create_budget_version(budget_id, 1, "Initial version from template", budget.dict())
				
				# Create line items from template
				if 'line_items_template' in template and template['line_items_template']:
					await self._create_line_items_from_template(budget_id, template['line_items_template'])
				
				# Update template usage count
				await self._increment_template_usage(template_id)
				
				# Create document folder if APG document_management is available
				if budget.document_folder_id:
					await self._create_document_folder(budget)
				
				# Audit the creation
				await self._audit_action('create_from_template', 'budget', budget_id, 
										new_data={'template_id': template_id, 'budget_code': budget.budget_code})
			
			return ServiceResponse(
				success=True,
				message=f"Budget '{budget.budget_name}' created from template '{template['template_name']}'",
				data={
					'budget_id': budget_id, 
					'budget_code': budget.budget_code,
					'template_id': template_id
				},
				metadata={'tenant_id': self.context.tenant_id}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_budget_from_template')
	
	async def create_budget_version(self, budget_id: str, version_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a new version of an existing budget."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.version', budget_id):
				raise PermissionError("Insufficient permissions to create budget version")
			
			# Get current budget
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			# Get current version number
			current_version = await self._get_latest_version_number(budget_id)
			new_version_number = current_version + 1
			
			# Create version model
			version = BudgetVersion(
				version_id=f"{budget_id}_v{new_version_number}",
				budget_id=budget_id,
				version_number=new_version_number,
				version_name=version_data.get('version_name'),
				change_summary=version_data.get('change_summary'),
				budget_snapshot=dict(budget),
				created_by=self.context.user_id,
				**version_data
			)
			
			# Start database transaction
			async with self._connection.transaction():
				# Mark previous versions as not current
				await self._connection.execute("""
					UPDATE budget_versions 
					SET is_current = FALSE 
					WHERE budget_id = $1
				""", budget_id)
				
				# Insert new version
				await self._insert_budget_version(version)
				
				# Update budget with version information
				await self._connection.execute("""
					UPDATE budgets 
					SET version = $1, updated_at = NOW(), updated_by = $2
					WHERE id = $3
				""", new_version_number, self.context.user_id, budget_id)
				
				# Audit the version creation
				await self._audit_action('create_version', 'budget', budget_id, 
										new_data={'version_number': new_version_number})
			
			return ServiceResponse(
				success=True,
				message=f"Budget version {new_version_number} created successfully",
				data={'version_id': version.version_id, 'version_number': new_version_number}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_budget_version')
	
	async def submit_budget_for_approval(self, budget_id: str, approval_data: Dict[str, Any]) -> ServiceResponse:
		"""Submit budget for comprehensive approval workflow."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.submit_approval', budget_id):
				raise PermissionError("Insufficient permissions to submit budget for approval")
			
			# Get budget and validate
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			if budget['status'] not in ['draft', 'under_review']:
				raise ValueError(f"Budget in status '{budget['status']}' cannot be submitted for approval")
			
			# Validate budget completeness
			validation_result = await self._validate_budget_completeness(budget_id)
			if not validation_result['is_valid']:
				return ServiceResponse(
					success=False,
					message="Budget validation failed",
					errors=validation_result['errors'],
					data={'validation_details': validation_result}
				)
			
			# Start approval workflow
			workflow_config = await self._get_approval_workflow_config(budget_id)
			workflow_id = await self._create_approval_workflow(budget_id, workflow_config, approval_data)
			
			# Start database transaction
			async with self._connection.transaction():
				# Update budget status
				await self._connection.execute("""
					UPDATE budgets 
					SET status = 'submitted',
						workflow_state = 'pending_approval',
						workflow_instance_id = $1,
						updated_at = NOW(),
						updated_by = $2
					WHERE id = $3
				""", workflow_id, self.context.user_id, budget_id)
				
				# Create approval tracking record
				await self._create_approval_tracking(budget_id, workflow_id, approval_data)
				
				# Send notifications to approvers
				await self._send_approval_notifications(budget_id, workflow_id)
				
				# Audit the submission
				await self._audit_action('submit_approval', 'budget', budget_id, 
										new_data={'workflow_id': workflow_id, **approval_data})
			
			return ServiceResponse(
				success=True,
				message="Budget submitted for approval successfully",
				data={
					'workflow_id': workflow_id,
					'approval_levels': len(workflow_config.get('approval_levels', [])),
					'estimated_approval_time': workflow_config.get('estimated_completion_days', 5)
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'submit_budget_for_approval')
	
	async def start_collaborative_session(self, budget_id: str, session_config: Dict[str, Any]) -> ServiceResponse:
		"""Start a real-time collaborative editing session for budget planning."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.collaborate', budget_id):
				raise PermissionError("Insufficient permissions to start collaborative session")
			
			# Get budget and validate
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			if budget['status'] in ['approved', 'locked', 'closed']:
				raise ValueError(f"Cannot collaborate on budget in status '{budget['status']}'")
			
			# Create collaboration session
			session_id = f"collab_{budget_id}_{int(datetime.utcnow().timestamp())}"
			
			collaboration = BudgetCollaboration(
				collaboration_id=f"{session_id}_{self.context.user_id}",
				budget_id=budget_id,
				session_id=session_id,
				user_id=self.context.user_id,
				user_name=session_config.get('user_name', 'Unknown User'),
				user_role=session_config.get('user_role', 'contributor'),
				activity_type='session_start',
				created_by=self.context.user_id
			)
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert collaboration record
				await self._insert_collaboration_record(collaboration)
				
				# Update budget with collaboration status
				await self._connection.execute("""
					UPDATE budgets 
					SET collaboration_enabled = TRUE,
						last_activity_date = CURRENT_DATE,
						active_contributors = array_append(
							COALESCE(active_contributors, '{}'), $1
						),
						updated_at = NOW()
					WHERE id = $2
				""", self.context.user_id, budget_id)
				
				# Initialize session state in cache/Redis (if available)
				await self._initialize_session_state(session_id, budget_id)
				
				# Audit the session start
				await self._audit_action('start_collaboration', 'budget', budget_id, 
										new_data={'session_id': session_id})
			
			return ServiceResponse(
				success=True,
				message="Collaborative session started successfully",
				data={
					'session_id': session_id,
					'collaboration_id': collaboration.collaboration_id,
					'session_config': session_config,
					'real_time_endpoint': f"/api/budgets/{budget_id}/collaborate/{session_id}"
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'start_collaborative_session')
	
	async def bulk_update_budget_lines(self, budget_id: str, updates: List[Dict[str, Any]]) -> ServiceResponse:
		"""Perform bulk updates on multiple budget lines with validation and conflict resolution."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.bulk_edit', budget_id):
				raise PermissionError("Insufficient permissions to perform bulk updates")
			
			# Get budget and validate
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			if budget['status'] not in ['draft', 'under_review']:
				raise ValueError("Budget is not editable in current status")
			
			# Validate all updates before applying any
			validation_results = []
			for i, update in enumerate(updates):
				validation = await self._validate_line_update(budget_id, update)
				validation_results.append({'index': i, 'valid': validation['is_valid'], 'errors': validation.get('errors', [])})
			
			# Check for validation failures
			failed_validations = [v for v in validation_results if not v['valid']]
			if failed_validations:
				return ServiceResponse(
					success=False,
					message=f"Validation failed for {len(failed_validations)} updates",
					errors=[f"Update {v['index']}: {', '.join(v['errors'])}" for v in failed_validations],
					data={'validation_results': validation_results}
				)
			
			# Start database transaction
			async with self._connection.transaction():
				updated_lines = []
				total_amount_change = Decimal('0.00')
				
				for update in updates:
					line_id = update.get('line_id')
					if update.get('operation') == 'update':
						# Update existing line
						result = await self._update_budget_line(budget_id, line_id, update)
						if result:
							updated_lines.append(result)
							total_amount_change += result.get('amount_change', Decimal('0.00'))
					
					elif update.get('operation') == 'insert':
						# Insert new line
						result = await self._insert_budget_line_from_update(budget_id, update)
						if result:
							updated_lines.append(result)
							total_amount_change += result.get('budgeted_amount', Decimal('0.00'))
					
					elif update.get('operation') == 'delete':
						# Soft delete line
						result = await self._soft_delete_budget_line(budget_id, line_id)
						if result:
							updated_lines.append(result)
							total_amount_change -= result.get('budgeted_amount', Decimal('0.00'))
				
				# Recalculate budget totals
				await self._recalculate_budget_totals(budget_id)
				
				# Create version snapshot if significant changes
				if abs(total_amount_change) > Decimal('1000.00'):  # Configurable threshold
					await self._create_budget_version(
						budget_id, 
						await self._get_latest_version_number(budget_id) + 1,
						f"Bulk update: {len(updated_lines)} lines modified",
						await self._get_budget_snapshot(budget_id)
					)
				
				# Audit the bulk operation
				await self._audit_action('bulk_update', 'budget', budget_id, 
										new_data={
											'lines_updated': len(updated_lines),
											'total_amount_change': str(total_amount_change)
										})
			
			return ServiceResponse(
				success=True,
				message=f"Successfully updated {len(updated_lines)} budget lines",
				data={
					'updated_lines': updated_lines,
					'total_amount_change': total_amount_change,
					'budget_totals_recalculated': True
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'bulk_update_budget_lines')
	
	# =============================================================================
	# Helper Methods
	# =============================================================================
	
	async def _get_budget_template(self, template_id: str) -> Optional[Dict[str, Any]]:
		"""Get budget template by ID."""
		return await self._connection.fetchrow("""
			SELECT * FROM bf_shared.budget_templates 
			WHERE template_id = $1 AND is_deleted = FALSE
		""", template_id)
	
	async def _can_access_template(self, template: Dict[str, Any], tenant_id: str) -> bool:
		"""Check if tenant can access the template."""
		# Public templates are accessible to all
		if template.get('is_public'):
			return True
		
		# Own templates are always accessible
		if template.get('owner_tenant_id') == tenant_id:
			return True
		
		# Check if shared with this tenant
		shared_with = template.get('shared_with_tenants', [])
		return tenant_id in shared_with
	
	async def _merge_template_data(self, template: Dict[str, Any], budget_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Merge template data with provided budget data."""
		template_data = template.get('template_data', {})
		default_settings = template.get('default_settings', {})
		
		# Start with template defaults
		merged = {**default_settings, **template_data}
		
		# Override with provided budget data
		merged.update(budget_data)
		
		# Ensure required fields are present
		required_fields = template.get('required_fields', [])
		for field in required_fields:
			if field not in merged or not merged[field]:
				raise ValueError(f"Required field '{field}' missing from budget data")
		
		return merged
	
	async def _create_line_items_from_template(self, budget_id: str, line_items_template: List[Dict[str, Any]]) -> None:
		"""Create budget line items from template."""
		for i, template_line in enumerate(line_items_template):
			line_data = {
				**template_line,
				'budget_id': budget_id,
				'line_number': i + 1,
				'tenant_id': self.context.tenant_id  # Will be inherited from budget
			}
			
			# Create and validate line
			budget_line = BFBudgetLine(**line_data)
			await self._insert_budget_line(budget_line)
	
	async def _increment_template_usage(self, template_id: str) -> None:
		"""Increment template usage count."""
		await self._connection.execute("""
			UPDATE bf_shared.budget_templates 
			SET usage_count = usage_count + 1,
				updated_at = NOW()
			WHERE template_id = $1
		""", template_id)
	
	async def _get_latest_version_number(self, budget_id: str) -> int:
		"""Get the latest version number for a budget."""
		result = await self._connection.fetchval("""
			SELECT COALESCE(MAX(version_number), 0) 
			FROM budget_versions 
			WHERE budget_id = $1
		""", budget_id)
		return result or 0
	
	async def _create_budget_version(self, budget_id: str, version_number: int, 
									change_summary: str, budget_snapshot: Dict[str, Any]) -> None:
		"""Create a budget version record."""
		version = BudgetVersion(
			version_id=f"{budget_id}_v{version_number}",
			budget_id=budget_id,
			version_number=version_number,
			change_summary=change_summary,
			budget_snapshot=budget_snapshot,
			created_by=self.context.user_id
		)
		await self._insert_budget_version(version)
	
	async def _insert_budget_version(self, version: BudgetVersion) -> None:
		"""Insert budget version into database."""
		version_dict = version.dict()
		columns = list(version_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(version_dict.values())
		
		query = f"""
			INSERT INTO budget_versions ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
		"""
		
		await self._connection.execute(query, *values)
	
	async def _validate_budget_completeness(self, budget_id: str) -> Dict[str, Any]:
		"""Validate that budget is complete and ready for approval."""
		errors = []
		
		# Check if budget has line items
		line_count = await self._connection.fetchval("""
			SELECT COUNT(*) FROM budget_lines 
			WHERE budget_id = $1 AND is_deleted = FALSE
		""", budget_id)
		
		if line_count == 0:
			errors.append("Budget must have at least one line item")
		
		# Check if all required fields are filled
		budget = await self._get_budget(budget_id)
		required_budget_fields = ['budget_name', 'budget_type', 'fiscal_year', 'base_currency']
		
		for field in required_budget_fields:
			if not budget.get(field):
				errors.append(f"Required field '{field}' is missing")
		
		# Check for line items with zero amounts
		zero_amount_lines = await self._connection.fetchval("""
			SELECT COUNT(*) FROM budget_lines 
			WHERE budget_id = $1 AND budgeted_amount = 0 AND is_deleted = FALSE
		""", budget_id)
		
		if zero_amount_lines > 0:
			errors.append(f"{zero_amount_lines} line items have zero budgeted amounts")
		
		return {
			'is_valid': len(errors) == 0,
			'errors': errors,
			'line_count': line_count,
			'zero_amount_lines': zero_amount_lines
		}
	
	async def _get_approval_workflow_config(self, budget_id: str) -> Dict[str, Any]:
		"""Get approval workflow configuration for budget."""
		# This would typically come from tenant configuration
		# For now, return a default workflow
		return {
			'approval_levels': [
				{'level': 1, 'role': 'manager', 'required': True},
				{'level': 2, 'role': 'director', 'required': True},
				{'level': 3, 'role': 'cfo', 'required': False}  # Only for large budgets
			],
			'estimated_completion_days': 5,
			'parallel_approval': False,
			'escalation_days': 3
		}
	
	async def _create_approval_workflow(self, budget_id: str, workflow_config: Dict[str, Any], 
									   approval_data: Dict[str, Any]) -> str:
		"""Create approval workflow instance."""
		# This would integrate with APG workflow_engine
		workflow_id = f"bf_approval_{budget_id}_{int(datetime.utcnow().timestamp())}"
		
		# Mock workflow creation
		self.logger.info(f"Creating approval workflow: {workflow_id}")
		
		return workflow_id
	
	async def _create_approval_tracking(self, budget_id: str, workflow_id: str, approval_data: Dict[str, Any]) -> None:
		"""Create approval tracking record."""
		await self._connection.execute("""
			INSERT INTO budget_approvals (
				budget_id, workflow_id, submitted_by, submitted_at,
				approval_notes, priority_level
			) VALUES ($1, $2, $3, NOW(), $4, $5)
		""", budget_id, workflow_id, self.context.user_id,
			approval_data.get('notes', ''), approval_data.get('priority', 'normal'))
	
	async def _send_approval_notifications(self, budget_id: str, workflow_id: str) -> None:
		"""Send notifications to approvers."""
		# This would integrate with APG notification_engine
		self.logger.info(f"Sending approval notifications for budget {budget_id}, workflow {workflow_id}")
	
	async def _insert_collaboration_record(self, collaboration: BudgetCollaboration) -> None:
		"""Insert collaboration record into database."""
		collab_dict = collaboration.dict()
		columns = list(collab_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(collab_dict.values())
		
		query = f"""
			INSERT INTO budget_collaboration ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
		"""
		
		await self._connection.execute(query, *values)
	
	async def _initialize_session_state(self, session_id: str, budget_id: str) -> None:
		"""Initialize collaboration session state."""
		# This would typically initialize Redis/cache state for real-time collaboration
		self.logger.info(f"Initializing session state for {session_id}")
	
	async def _validate_line_update(self, budget_id: str, update: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate a single line update."""
		errors = []
		
		# Validate required fields
		if update.get('operation') in ['update', 'insert']:
			required_fields = ['line_description', 'budgeted_amount', 'account_code']
			for field in required_fields:
				if field not in update or not update[field]:
					errors.append(f"Required field '{field}' missing")
		
		# Validate amount is positive
		if 'budgeted_amount' in update:
			try:
				amount = Decimal(str(update['budgeted_amount']))
				if amount < 0:
					errors.append("Budgeted amount cannot be negative")
			except (ValueError, TypeError):
				errors.append("Invalid budgeted amount format")
		
		return {
			'is_valid': len(errors) == 0,
			'errors': errors
		}
	
	async def _update_budget_line(self, budget_id: str, line_id: str, update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		"""Update a single budget line."""
		# Build dynamic update query
		update_fields = []
		values = []
		param_count = 1
		
		for field, value in update.items():
			if field not in ['line_id', 'operation']:
				update_fields.append(f"{field} = ${param_count}")
				values.append(value)
				param_count += 1
		
		if not update_fields:
			return None
		
		values.extend([self.context.user_id, line_id])
		
		query = f"""
			UPDATE budget_lines 
			SET {', '.join(update_fields)}, updated_at = NOW(), updated_by = ${param_count}
			WHERE id = ${param_count + 1}
			RETURNING id, budgeted_amount
		"""
		
		result = await self._connection.fetchrow(query, *values)
		return dict(result) if result else None
	
	async def _insert_budget_line_from_update(self, budget_id: str, update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		"""Insert new budget line from update data."""
		line_data = {
			**update,
			'budget_id': budget_id,
			'tenant_id': self.context.tenant_id
		}
		
		# Remove operation field
		line_data.pop('operation', None)
		
		# Create and validate line
		budget_line = BFBudgetLine(**line_data)
		line_id = await self._insert_budget_line(budget_line)
		
		return {'id': line_id, 'budgeted_amount': budget_line.budgeted_amount}
	
	async def _soft_delete_budget_line(self, budget_id: str, line_id: str) -> Optional[Dict[str, Any]]:
		"""Soft delete a budget line."""
		result = await self._connection.fetchrow("""
			UPDATE budget_lines 
			SET is_deleted = TRUE,
				deleted_at = NOW(),
				deleted_by = $1,
				updated_at = NOW()
			WHERE id = $2 AND budget_id = $3
			RETURNING id, budgeted_amount
		""", self.context.user_id, line_id, budget_id)
		
		return dict(result) if result else None
	
	async def _get_budget_snapshot(self, budget_id: str) -> Dict[str, Any]:
		"""Get complete budget snapshot for versioning."""
		budget = await self._get_budget(budget_id)
		lines = await self._connection.fetch("""
			SELECT * FROM budget_lines 
			WHERE budget_id = $1 AND is_deleted = FALSE
		""", budget_id)
		
		return {
			'budget': dict(budget),
			'lines': [dict(line) for line in lines]
		}


# =============================================================================
# Service Factory and Export
# =============================================================================

def create_advanced_budget_service(context: APGTenantContext, config: BFServiceConfig) -> AdvancedBudgetService:
	"""Factory function to create advanced budget service."""
	return AdvancedBudgetService(context, config)


# Export advanced budget management classes
__all__ = [
	'BudgetTemplate',
	'BudgetVersion', 
	'BudgetCollaboration',
	'AdvancedBudgetService',
	'create_advanced_budget_service'
]


def _log_advanced_budget_summary() -> str:
	"""Log summary of advanced budget management capabilities."""
	return f"Advanced Budget Management loaded: {len(__all__)} components with template, versioning, and collaboration support"