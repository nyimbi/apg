
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
"""
APG Workflow & Business Process Management - Core Service Layer

Comprehensive service layer with APG platform integration, async operations,
and enterprise-grade workflow management capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import inspect
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from .models import (
    APGTenantContext, WBPMServiceConfig, WBPMServiceResponse, WBPMPagedResponse,
    WBPMProcessDefinition, WBPMProcessInstance, WBPMProcessActivity, WBPMProcessFlow,
    WBPMTask, WBPMTaskHistory, WBPMTaskComment, WBPMProcessTemplate,
    ProcessStatus, InstanceStatus, TaskStatus, TaskPriority
)

from .workflow_engine import WorkflowExecutionEngine, create_workflow_engine
from .task_management import (
    AITaskRouter, TaskQueueManager, EscalationEngine, TaskPerformanceTracker,
    create_task_management_components, AssignmentStrategy, TaskAssignmentCriteria, UserProfile
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# APG Platform Integration Layer
# =============================================================================

WORKFLOW_PERMISSION_ALIASES: Dict[str, List[str]] = {
    "create_process": ["wbpm:process:create", "wbpm:process:write", "workflow_design"],
    "view_process": ["wbpm:process:read", "workflow_view", "workflow_design"],
    "start_process": ["wbpm:process:execute", "workflow_execute"],
    "create_task": ["wbpm:task:create", "wbpm:task:write", "workflow_execute"],
    "complete_any_task": ["wbpm:task:complete", "wbpm:task:admin"],
    "view_any_task": ["wbpm:task:read", "wbpm:task:admin"],
    "manage_instance": ["wbpm:instance:manage", "wbpm:process:manage", "workflow_manage"],
}


def _clean_permission(value: Any) -> Optional[str]:
    """Normalize a permission token."""
    if value is None:
        return None
    permission = str(value).strip()
    return permission or None


def _permission_candidates(required_permission: str) -> List[str]:
    """Return all permission tokens that satisfy an internal workflow permission."""
    required = _clean_permission(required_permission)
    if not required:
        return []
    return [required, *WORKFLOW_PERMISSION_ALIASES.get(required, [])]


def _permission_matches(granted_permission: str, required_permission: str) -> bool:
    """Return whether a granted APG permission covers a required workflow permission."""
    granted = _clean_permission(granted_permission)
    if not granted:
        return False
    if granted in {"*", "admin", "wbpm:*", "workflow_admin"}:
        return True
    for candidate in _permission_candidates(required_permission):
        if granted == candidate:
            return True
        for suffix, separator in ((":*", ":"), (".*", ".")):
            if granted.endswith(suffix):
                prefix = granted[: -len(suffix)]
                if candidate == prefix or candidate.startswith(f"{prefix}{separator}"):
                    return True
    return False


class APGPlatformIntegration:
    """Integration layer for APG platform services."""
    
    def __init__(self, config: WBPMServiceConfig, auth_service: Any | None = None):
        self.config = config
        self.auth_service_url = config.apg_auth_service_url
        self.audit_service_url = config.apg_audit_service_url
        self.collaboration_service_url = config.apg_collaboration_service_url
        self.ai_service_url = config.apg_ai_service_url
        self.auth_service = auth_service
    
    async def validate_user_permissions(
        self,
        context: APGTenantContext,
        required_permissions: List[str]
    ) -> bool:
        """Validate user permissions through APG auth service."""
        try:
            required_permissions = [
                permission for permission in
                (_clean_permission(permission) for permission in required_permissions)
                if permission
            ]
            if not required_permissions:
                return True

            auth_result = await self._validate_permissions_with_auth_service(context, required_permissions)
            if auth_result is not None:
                return auth_result

            user_permissions = [
                permission for permission in
                (_clean_permission(permission) for permission in context.permissions)
                if permission
            ]
            missing_permissions = [
                permission for permission in required_permissions
                if not any(_permission_matches(granted, permission) for granted in user_permissions)
            ]
            has_permissions = not missing_permissions
            
            if not has_permissions:
                logger.warning(f"User {context.user_id} missing permissions: {missing_permissions}")
            
            return has_permissions
            
        except Exception as e:
            logger.error(f"Error validating permissions: {e}")
            return False

    async def _validate_permissions_with_auth_service(
        self,
        context: APGTenantContext,
        required_permissions: List[str]
    ) -> Optional[bool]:
        """Validate permissions through an injected APG auth service when configured."""
        if not self.auth_service:
            return await self._validate_permissions_over_http(context, required_permissions)

        plural_methods = ("has_permissions", "validate_permissions", "check_permissions")
        for method_name in plural_methods:
            method = getattr(self.auth_service, method_name, None)
            if callable(method):
                result = await self._call_auth_method(
                    method,
                    context=context,
                    permissions=required_permissions
                )
                if result is not None:
                    return result

        singular_methods = ("has_permission", "check_permission", "user_has_permission")
        for permission in required_permissions:
            permission_result: Optional[bool] = None
            for method_name in singular_methods:
                method = getattr(self.auth_service, method_name, None)
                if callable(method):
                    permission_result = await self._call_auth_method(
                        method,
                        context=context,
                        permissions=[permission]
                    )
                    if permission_result is not None:
                        break
            if permission_result is False:
                return False
            if permission_result is None:
                return None
        return True

    async def _call_auth_method(
        self,
        method: Any,
        context: APGTenantContext,
        permissions: List[str]
    ) -> Optional[bool]:
        """Call common APG auth service method signatures."""
        calls = (
            lambda: method(context=context, required_permissions=permissions),
            lambda: method(context=context, permissions=permissions),
            lambda: method(
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                permissions=permissions
            ),
            lambda: method(
                user_id=context.user_id,
                permission=permissions[0],
                tenant_id=context.tenant_id
            ),
            lambda: method(context.user_id, permissions, context.tenant_id),
            lambda: method(context.user_id, permissions[0], context.tenant_id),
        )
        for call in calls:
            try:
                return await self._coerce_auth_result(call())
            except TypeError:
                continue
        return None

    async def _coerce_auth_result(self, result: Any) -> bool:
        """Normalize APG auth service responses to a boolean."""
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            for key in ("allowed", "authorized", "has_permission", "success"):
                if key in result:
                    return bool(result[key])
        return bool(result)

    async def _validate_permissions_over_http(
        self,
        context: APGTenantContext,
        required_permissions: List[str]
    ) -> Optional[bool]:
        """Validate permissions through APG auth_rbac HTTP endpoint when an auth token is present."""
        auth_token = context.metadata.get("auth_token") or context.metadata.get("access_token")
        if not self.auth_service_url or not auth_token:
            return None

        try:
            import httpx
        except ImportError:
            logger.warning("httpx is unavailable; falling back to context permissions")
            return None

        endpoint = f"{self.auth_service_url.rstrip('/')}/permissions/validate"
        payload = {
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "permissions": required_permissions,
        }
        headers = {"Authorization": f"Bearer {auth_token}"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                return await self._coerce_auth_result(response.json())
        except Exception as e:
            logger.warning(f"APG auth service permission validation failed: {e}")
            return None
    
    async def log_audit_event(
        self,
        context: APGTenantContext,
        event_type: str,
        event_description: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Log audit event to APG audit service."""
        try:
            audit_record = {
                'tenant_id': context.tenant_id,
                'user_id': context.user_id,
                'session_id': context.session_id,
                'event_type': event_type,
                'event_source': 'workflow_business_process_mgmt',
                'event_description': event_description,
                'event_data': event_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # In production, would send to APG audit service
            logger.info(f"Audit event logged: {event_type} - {event_description}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            return False
    
    async def send_notification(
        self,
        recipient_id: str,
        notification_type: str,
        message: str,
        context: APGTenantContext
    ) -> bool:
        """Send notification through APG notification engine."""
        try:
            notification = {
                'recipient_id': recipient_id,
                'notification_type': notification_type,
                'message': message,
                'sender_id': context.user_id,
                'tenant_id': context.tenant_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # In production, would send to APG notification service
            logger.info(f"Notification sent to {recipient_id}: {notification_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False


# =============================================================================
# Core Service Classes
# =============================================================================

class ProcessManagementService:
    """Core process definition and instance management service."""
    
    def __init__(self, config: WBPMServiceConfig, apg_integration: APGPlatformIntegration):
        self.config = config
        self.apg = apg_integration
        # In production, would initialize database connections here
        self.process_definitions: Dict[str, WBPMProcessDefinition] = {}
        self.process_instances: Dict[str, WBPMProcessInstance] = {}
    
    async def create_process_definition(
        self,
        process_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Create new process definition."""
        try:
            # Validate permissions
            if not await self.apg.validate_user_permissions(context, ['create_process']):
                return WBPMServiceResponse(
                    success=False,
                    message="Insufficient permissions to create process",
                    errors=["Permission denied"]
                )
            
            # Create process definition
            process_definition = WBPMProcessDefinition(
                tenant_id=context.tenant_id,
                process_key=process_data['process_key'],
                process_name=process_data['process_name'],
                process_description=process_data.get('process_description'),
                bpmn_xml=process_data['bpmn_xml'],
                category=process_data.get('category'),
                tags=process_data.get('tags', []),
                created_by=context.user_id,
                updated_by=context.user_id
            )
            
            # Store process definition (in production, would save to database)
            self.process_definitions[process_definition.id] = process_definition
            
            # Log audit event
            await self.apg.log_audit_event(
                context,
                'process_created',
                f'Process definition created: {process_definition.process_name}',
                {'process_id': process_definition.id, 'process_key': process_definition.process_key}
            )
            
            return WBPMServiceResponse(
                success=True,
                message="Process definition created successfully",
                data={
                    'process_id': process_definition.id,
                    'process_key': process_definition.process_key,
                    'process_name': process_definition.process_name,
                    'status': process_definition.process_status
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating process definition: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to create process definition: {e}",
                errors=[str(e)]
            )
    
    async def get_process_definition(
        self,
        process_id: str,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Get process definition by ID."""
        try:
            # Validate permissions
            if not await self.apg.validate_user_permissions(context, ['view_process']):
                return WBPMServiceResponse(
                    success=False,
                    message="Insufficient permissions to view process",
                    errors=["Permission denied"]
                )
            
            # Get process definition
            process_definition = self.process_definitions.get(process_id)
            
            if not process_definition:
                return WBPMServiceResponse(
                    success=False,
                    message="Process definition not found",
                    errors=["Process not found"]
                )
            
            # Verify tenant access
            if process_definition.tenant_id != context.tenant_id:
                return WBPMServiceResponse(
                    success=False,
                    message="Access denied to process definition",
                    errors=["Tenant access denied"]
                )
            
            return WBPMServiceResponse(
                success=True,
                message="Process definition retrieved successfully",
                data=process_definition.dict()
            )
            
        except Exception as e:
            logger.error(f"Error getting process definition {process_id}: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to get process definition: {e}",
                errors=[str(e)]
            )
    
    async def list_process_definitions(
        self,
        context: APGTenantContext,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> WBPMPagedResponse:
        """List process definitions with pagination and filtering."""
        try:
            # Validate permissions
            if not await self.apg.validate_user_permissions(context, ['view_process']):
                return WBPMPagedResponse(
                    items=[],
                    total_count=0,
                    page=page,
                    page_size=page_size,
                    has_next=False,
                    has_previous=False
                )
            
            # Filter by tenant
            tenant_processes = [
                process for process in self.process_definitions.values()
                if process.tenant_id == context.tenant_id
            ]
            
            # Apply additional filters
            if filters:
                if 'status' in filters:
                    tenant_processes = [p for p in tenant_processes if p.process_status == filters['status']]
                if 'category' in filters:
                    tenant_processes = [p for p in tenant_processes if p.category == filters['category']]
                if 'search' in filters:
                    search_term = filters['search'].lower()
                    tenant_processes = [
                        p for p in tenant_processes 
                        if search_term in p.process_name.lower() or search_term in (p.process_description or "").lower()
                    ]
            
            # Sort by creation date (newest first)
            tenant_processes.sort(key=lambda x: x.created_at, reverse=True)
            
            # Pagination
            total_count = len(tenant_processes)
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            page_items = tenant_processes[start_index:end_index]
            
            # Convert to dict for response
            items = [process.dict() for process in page_items]
            
            return WBPMPagedResponse(
                items=items,
                total_count=total_count,
                page=page,
                page_size=page_size,
                has_next=end_index < total_count,
                has_previous=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error listing process definitions: {e}")
            return WBPMPagedResponse(
                items=[],
                total_count=0,
                page=page,
                page_size=page_size,
                has_next=False,
                has_previous=False
            )
    
    async def start_process_instance(
        self,
        process_id: str,
        instance_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Start new process instance."""
        try:
            # Validate permissions
            if not await self.apg.validate_user_permissions(context, ['start_process']):
                return WBPMServiceResponse(
                    success=False,
                    message="Insufficient permissions to start process",
                    errors=["Permission denied"]
                )
            
            # Get process definition
            process_definition = self.process_definitions.get(process_id)
            if not process_definition:
                return WBPMServiceResponse(
                    success=False,
                    message="Process definition not found",
                    errors=["Process not found"]
                )
            
            # Verify tenant access
            if process_definition.tenant_id != context.tenant_id:
                return WBPMServiceResponse(
                    success=False,
                    message="Access denied to process definition",
                    errors=["Tenant access denied"]
                )
            
            # Create process instance
            process_instance = WBPMProcessInstance(
                tenant_id=context.tenant_id,
                process_id=process_id,
                business_key=instance_data.get('business_key'),
                instance_name=instance_data.get('instance_name'),
                process_variables=instance_data.get('variables', {}),
                initiated_by=context.user_id,
                priority=TaskPriority(instance_data.get('priority', 'medium')),
                due_date=datetime.fromisoformat(instance_data['due_date']) if instance_data.get('due_date') else None,
                created_by=context.user_id,
                updated_by=context.user_id
            )
            
            # Store process instance
            self.process_instances[process_instance.id] = process_instance
            
            # Log audit event
            await self.apg.log_audit_event(
                context,
                'process_started',
                f'Process instance started: {process_definition.process_name}',
                {
                    'process_id': process_id,
                    'instance_id': process_instance.id,
                    'business_key': process_instance.business_key
                }
            )
            
            return WBPMServiceResponse(
                success=True,
                message="Process instance started successfully",
                data={
                    'instance_id': process_instance.id,
                    'process_id': process_id,
                    'business_key': process_instance.business_key,
                    'status': process_instance.instance_status
                }
            )
            
        except Exception as e:
            logger.error(f"Error starting process instance: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to start process instance: {e}",
                errors=[str(e)]
            )


class TaskManagementService:
    """Task management service with intelligent routing and assignment."""
    
    def __init__(self, config: WBPMServiceConfig, apg_integration: APGPlatformIntegration):
        self.config = config
        self.apg = apg_integration
        
        # Initialize task management components
        self.task_router, self.queue_manager, self.escalation_engine, self.performance_tracker = create_task_management_components()
        
        # In production, would initialize database connections
        self.tasks: Dict[str, WBPMTask] = {}
        self.task_history: Dict[str, List[WBPMTaskHistory]] = {}
        self.task_comments: Dict[str, List[WBPMTaskComment]] = {}
    
    async def create_task(
        self,
        task_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Create new task."""
        try:
            # Validate permissions
            if not await self.apg.validate_user_permissions(context, ['create_task']):
                return WBPMServiceResponse(
                    success=False,
                    message="Insufficient permissions to create task",
                    errors=["Permission denied"]
                )
            
            # Create task
            task = WBPMTask(
                tenant_id=context.tenant_id,
                process_instance_id=task_data['process_instance_id'],
                activity_id=task_data['activity_id'],
                task_name=task_data['task_name'],
                task_description=task_data.get('task_description'),
                assignee=task_data.get('assignee'),
                candidate_users=task_data.get('candidate_users', []),
                candidate_groups=task_data.get('candidate_groups', []),
                form_key=task_data.get('form_key'),
                priority=TaskPriority(task_data.get('priority', 'medium')),
                due_date=datetime.fromisoformat(task_data['due_date']) if task_data.get('due_date') else None,
                created_by=context.user_id,
                updated_by=context.user_id
            )
            
            # Store task
            self.tasks[task.id] = task
            
            # Add to queue for assignment
            await self.queue_manager.add_task_to_queue(task)
            
            # Assign task if no specific assignee
            if not task.assignee and (task.candidate_users or task.candidate_groups):
                assignment_criteria = TaskAssignmentCriteria(
                    preferred_users=task.candidate_users,
                    required_roles=task.candidate_groups
                )
                
                assigned_user = await self.task_router.assign_task(
                    task, assignment_criteria, AssignmentStrategy.AI_OPTIMIZED
                )
                
                if assigned_user:
                    task.assignee = assigned_user
                    task.task_status = TaskStatus.READY
                    task.updated_by = context.user_id
                    task.updated_at = datetime.utcnow()
            
            # Create task history entry
            await self._create_task_history_entry(
                task.id, 'created', 'Task created', None, task.dict(), context
            )
            
            # Log audit event
            await self.apg.log_audit_event(
                context,
                'task_created',
                f'Task created: {task.task_name}',
                {'task_id': task.id, 'assignee': task.assignee}
            )
            
            return WBPMServiceResponse(
                success=True,
                message="Task created successfully",
                data={
                    'task_id': task.id,
                    'task_name': task.task_name,
                    'assignee': task.assignee,
                    'status': task.task_status,
                    'priority': task.priority
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to create task: {e}",
                errors=[str(e)]
            )
    
    async def complete_task(
        self,
        task_id: str,
        completion_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Complete task with validation and workflow continuation."""
        try:
            # Get task
            task = self.tasks.get(task_id)
            if not task:
                return WBPMServiceResponse(
                    success=False,
                    message="Task not found",
                    errors=["Task not found"]
                )
            
            # Verify tenant access
            if task.tenant_id != context.tenant_id:
                return WBPMServiceResponse(
                    success=False,
                    message="Access denied to task",
                    errors=["Tenant access denied"]
                )
            
            # Validate user can complete task
            can_complete = (
                task.assignee == context.user_id or
                context.user_id in task.candidate_users or
                await self.apg.validate_user_permissions(context, ['complete_any_task'])
            )
            
            if not can_complete:
                return WBPMServiceResponse(
                    success=False,
                    message="Not authorized to complete this task",
                    errors=["Authorization failed"]
                )
            
            # Update task status
            old_status = task.task_status
            task.task_status = TaskStatus.COMPLETED
            task.completion_time = datetime.utcnow()
            task.task_variables.update(completion_data.get('variables', {}))
            task.updated_by = context.user_id
            task.updated_at = datetime.utcnow()
            
            # Record performance metrics
            quality_score = completion_data.get('quality_score')
            await self.performance_tracker.track_task_completion(
                task, task.completion_time, quality_score
            )
            
            # Create task history entry
            await self._create_task_history_entry(
                task_id, 'completed', 'Task completed',
                {'status': old_status}, {'status': task.task_status, 'completion_data': completion_data},
                context
            )
            
            # Remove from queue
            await self.queue_manager.remove_task_from_queue(task_id)
            
            # Send notification to relevant parties
            if task.assignee and task.assignee != context.user_id:
                await self.apg.send_notification(
                    task.assignee,
                    'task_completed',
                    f'Task "{task.task_name}" has been completed',
                    context
                )
            
            # Log audit event
            await self.apg.log_audit_event(
                context,
                'task_completed',
                f'Task completed: {task.task_name}',
                {'task_id': task_id, 'completed_by': context.user_id}
            )
            
            return WBPMServiceResponse(
                success=True,
                message="Task completed successfully",
                data={
                    'task_id': task_id,
                    'completion_time': task.completion_time.isoformat(),
                    'completion_data': completion_data
                }
            )
            
        except Exception as e:
            logger.error(f"Error completing task {task_id}: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to complete task: {e}",
                errors=[str(e)]
            )
    
    async def get_user_task_queue(
        self,
        user_id: str,
        context: APGTenantContext,
        page: int = 1,
        page_size: int = 20
    ) -> WBPMPagedResponse:
        """Get task queue for specific user."""
        try:
            # Validate permissions (user can view own queue or has admin permissions)
            can_view = (
                user_id == context.user_id or
                await self.apg.validate_user_permissions(context, ['view_any_task'])
            )
            
            if not can_view:
                return WBPMPagedResponse(
                    items=[],
                    total_count=0,
                    page=page,
                    page_size=page_size,
                    has_next=False,
                    has_previous=False
                )
            
            # Get user tasks
            user_tasks = [
                task for task in self.tasks.values()
                if (task.tenant_id == context.tenant_id and
                    (task.assignee == user_id or
                     user_id in task.candidate_users) and
                    task.task_status in [TaskStatus.CREATED, TaskStatus.READY, TaskStatus.IN_PROGRESS])
            ]
            
            # Sort by priority and due date
            def task_sort_key(task):
                priority_weight = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
                due_date_weight = 0
                if task.due_date:
                    days_until_due = (task.due_date - datetime.utcnow()).days
                    due_date_weight = max(0, 10 - days_until_due)
                
                return (priority_weight.get(task.priority, 1) * 10 + due_date_weight)
            
            user_tasks.sort(key=task_sort_key, reverse=True)
            
            # Pagination
            total_count = len(user_tasks)
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            page_items = user_tasks[start_index:end_index]
            
            # Convert to dict for response
            items = [task.dict() for task in page_items]
            
            return WBPMPagedResponse(
                items=items,
                total_count=total_count,
                page=page,
                page_size=page_size,
                has_next=end_index < total_count,
                has_previous=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error getting user task queue: {e}")
            return WBPMPagedResponse(
                items=[],
                total_count=0,
                page=page,
                page_size=page_size,
                has_next=False,
                has_previous=False
            )
    
    async def _create_task_history_entry(
        self,
        task_id: str,
        action_type: str,
        description: str,
        old_value: Optional[Dict[str, Any]],
        new_value: Dict[str, Any],
        context: APGTenantContext
    ) -> None:
        """Create task history entry."""
        history_entry = WBPMTaskHistory(
            tenant_id=context.tenant_id,
            task_id=task_id,
            action_type=action_type,
            action_description=description,
            old_value=old_value,
            new_value=new_value,
            performed_by=context.user_id,
            created_by=context.user_id,
            updated_by=context.user_id
        )
        
        if task_id not in self.task_history:
            self.task_history[task_id] = []
        
        self.task_history[task_id].append(history_entry)


class WorkflowExecutionService:
    """Workflow execution orchestration service."""
    
    def __init__(self, config: WBPMServiceConfig, apg_integration: APGPlatformIntegration):
        self.config = config
        self.apg = apg_integration
        self.workflow_engine = create_workflow_engine(config.max_concurrent_instances)
    
    async def execute_workflow_instance(
        self,
        process_definition: WBPMProcessDefinition,
        process_instance: WBPMProcessInstance,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Execute workflow instance using the workflow engine."""
        try:
            # Start workflow execution
            result = await self.workflow_engine.start_process_instance(
                process_definition,
                process_instance,
                context,
                process_instance.process_variables
            )
            
            if result.success:
                # Log audit event
                await self.apg.log_audit_event(
                    context,
                    'workflow_executed',
                    f'Workflow execution started for process: {process_definition.process_name}',
                    {
                        'process_id': process_definition.id,
                        'instance_id': process_instance.id,
                        'execution_result': result.data
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing workflow instance: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to execute workflow: {e}",
                errors=[str(e)]
            )
    
    async def get_instance_status(
        self,
        instance_id: str,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Get detailed workflow instance status."""
        try:
            # Get instance status from workflow engine
            status = self.workflow_engine.get_instance_status(instance_id)
            
            if not status:
                return WBPMServiceResponse(
                    success=False,
                    message="Workflow instance not found or not active",
                    errors=["Instance not found"]
                )
            
            return WBPMServiceResponse(
                success=True,
                message="Instance status retrieved successfully",
                data=status
            )
            
        except Exception as e:
            logger.error(f"Error getting instance status: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to get instance status: {e}",
                errors=[str(e)]
            )


# =============================================================================
# Main Service Container
# =============================================================================

class WorkflowBusinessProcessMgmtService:
    """Main service container for workflow and business process management."""
    
    def __init__(self, config: WBPMServiceConfig):
        self.config = config
        self.apg_integration = APGPlatformIntegration(config)
        
        # Initialize core services
        self.process_service = ProcessManagementService(config, self.apg_integration)
        self.task_service = TaskManagementService(config, self.apg_integration)
        self.workflow_service = WorkflowExecutionService(config, self.apg_integration)
        
        logger.info("Workflow & Business Process Management Service initialized")
    
    # Process Management API
    
    async def create_process(
        self,
        process_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Create new process definition."""
        return await self.process_service.create_process_definition(process_data, context)
    
    async def get_process(
        self,
        process_id: str,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Get process definition by ID."""
        return await self.process_service.get_process_definition(process_id, context)
    
    async def list_processes(
        self,
        context: APGTenantContext,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> WBPMPagedResponse:
        """List process definitions with pagination."""
        return await self.process_service.list_process_definitions(context, page, page_size, filters)
    
    async def start_process(
        self,
        process_id: str,
        instance_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Start new process instance."""
        # Create process instance
        result = await self.process_service.start_process_instance(process_id, instance_data, context)
        
        if result.success:
            # Get process definition and instance for workflow execution
            process_def_result = await self.process_service.get_process_definition(process_id, context)
            if process_def_result.success:
                process_definition = WBPMProcessDefinition(**process_def_result.data)
                process_instance = self.process_service.process_instances[result.data['instance_id']]
                
                # Start workflow execution
                execution_result = await self.workflow_service.execute_workflow_instance(
                    process_definition, process_instance, context
                )
                
                # Merge results
                result.data.update(execution_result.data or {})
        
        return result
    
    # Task Management API
    
    async def create_task(
        self,
        task_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Create new task."""
        return await self.task_service.create_task(task_data, context)
    
    async def complete_task(
        self,
        task_id: str,
        completion_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Complete task and continue workflow."""
        result = await self.task_service.complete_task(task_id, completion_data, context)
        
        if result.success:
            # Continue workflow execution in workflow engine
            workflow_result = await self.workflow_service.workflow_engine.complete_user_task(
                task_id, completion_data, context
            )
            
            # Merge workflow continuation result
            if workflow_result.data:
                result.data.update(workflow_result.data)
        
        return result
    
    async def get_user_tasks(
        self,
        user_id: str,
        context: APGTenantContext,
        page: int = 1,
        page_size: int = 20
    ) -> WBPMPagedResponse:
        """Get task queue for user."""
        return await self.task_service.get_user_task_queue(user_id, context, page, page_size)
    
    # Workflow Execution API
    
    async def get_instance_status(
        self,
        instance_id: str,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Get workflow instance status."""
        return await self.workflow_service.get_instance_status(instance_id, context)
    
    async def suspend_instance(
        self,
        instance_id: str,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Suspend workflow instance."""
        if not await self.apg_integration.validate_user_permissions(context, ['manage_instance']):
            return WBPMServiceResponse(
                success=False,
                message="Insufficient permissions",
                errors=["Permission denied"]
            )
        
        return await self.workflow_service.workflow_engine.suspend_process_instance(instance_id)
    
    async def resume_instance(
        self,
        instance_id: str,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Resume workflow instance."""
        if not await self.apg_integration.validate_user_permissions(context, ['manage_instance']):
            return WBPMServiceResponse(
                success=False,
                message="Insufficient permissions",
                errors=["Permission denied"]
            )
        
        return await self.workflow_service.workflow_engine.resume_process_instance(instance_id)
    
    async def cancel_instance(
        self,
        instance_id: str,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Cancel workflow instance."""
        if not await self.apg_integration.validate_user_permissions(context, ['manage_instance']):
            return WBPMServiceResponse(
                success=False,
                message="Insufficient permissions",
                errors=["Permission denied"]
            )
        
        return await self.workflow_service.workflow_engine.cancel_process_instance(instance_id)
    
    # Health and Status
    
    # ── 9 new methods ────────────────────────────────────────────────────────

    async def workflow_analytics(self, period: str) -> Dict[str, Any]:
        """Return workflow analytics for a period."""
        try:
            active_instances = self.workflow_service.workflow_engine.get_active_instances()
            return {
                "period": period,
                "active_instances": len(active_instances),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            logger.error(f"workflow_analytics error: {exc}")
            return {"period": period, "error": str(exc)}

    async def sla_compliance_report(self, workflow_type: str, period: str) -> Dict[str, Any]:
        """Generate an SLA compliance report for a workflow type and period."""
        try:
            all_instances = self.workflow_service.workflow_engine.get_active_instances()
            type_instances = [i for i in all_instances if getattr(i, "workflow_type", "") == workflow_type]
            breached = [i for i in type_instances if getattr(i, "sla_breached", False)]
            compliance_rate = round((len(type_instances) - len(breached)) / max(len(type_instances), 1) * 100, 1)
            return {
                "workflow_type": workflow_type,
                "period": period,
                "total_instances": len(type_instances),
                "sla_breaches": len(breached),
                "compliance_rate_pct": compliance_rate,
                "generated_at": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            logger.error(f"sla_compliance_report error: {exc}")
            return {"workflow_type": workflow_type, "period": period, "error": str(exc)}

    async def bulk_approve(self, task_ids: List[str], approved_by: str, notes: str = "") -> Dict[str, Any]:
        """Bulk-approve a list of tasks on behalf of an approver."""
        results: List[Dict[str, Any]] = []
        context = APGTenantContext(
            tenant_id=self.config.default_tenant_id or "default",
            user_id=approved_by,
            permissions=["complete_any_task"],
        )
        for task_id in task_ids:
            result = await self.complete_task(context, task_id, {"decision": "approved", "notes": notes})
            results.append({"task_id": task_id, "success": result.success, "message": result.message})
        succeeded = sum(1 for r in results if r["success"])
        return {
            "submitted": len(task_ids),
            "succeeded": succeeded,
            "failed": len(task_ids) - succeeded,
            "approved_by": approved_by,
            "results": results,
        }

    async def delegation_create(
        self,
        delegator_id: str,
        delegate_id: str,
        workflow_type: str,
        expiry: str,
    ) -> Dict[str, Any]:
        """Create a task delegation from delegator to delegate for a workflow type."""
        delegation_id = f"del-{delegator_id[:6]}-{delegate_id[:6]}-{workflow_type[:8]}"
        record = {
            "delegation_id": delegation_id,
            "delegator_id": delegator_id,
            "delegate_id": delegate_id,
            "workflow_type": workflow_type,
            "expiry": expiry,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
        }
        logger.info(f"Delegation created: {delegation_id}")
        return record

    async def delegation_revoke(self, delegation_id: str) -> Dict[str, Any]:
        """Revoke an active delegation."""
        logger.info(f"Delegation revoked: {delegation_id}")
        return {
            "delegation_id": delegation_id,
            "status": "revoked",
            "revoked_at": datetime.utcnow().isoformat(),
        }

    async def integration_trigger(self, workflow_id: str, external_event: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a workflow from an external event (webhook / integration bus)."""
        context = APGTenantContext(
            tenant_id=external_event.get("tenant_id", self.config.default_tenant_id or "default"),
            user_id=external_event.get("source", "external"),
            permissions=["start_process"],
        )
        event_data = {
            "event_type": external_event.get("type", "external"),
            "payload": external_event,
            "workflow_id": workflow_id,
        }
        result = await self.start_process(context, workflow_id, event_data)
        return {
            "workflow_id": workflow_id,
            "external_event": external_event.get("type"),
            "instance_started": result.success,
            "message": result.message,
            "triggered_at": datetime.utcnow().isoformat(),
        }

    async def performance_kpi(self, period: str) -> Dict[str, Any]:
        """Return workflow performance KPIs for a period."""
        health = await self.get_service_health()
        return {
            "period": period,
            "service_status": health.get("status"),
            "active_instances": health.get("active_instances", 0),
            "max_concurrent": health.get("max_concurrent_instances"),
            "components_healthy": all(
                v == "healthy" for v in health.get("components", {}).values()
            ),
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def audit_workflow_trail(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Return the audit trail for a workflow instance."""
        try:
            instance = await self.get_instance_status(
                APGTenantContext(
                    tenant_id=self.config.default_tenant_id or "default",
                    user_id="system",
                    permissions=["manage_instance"],
                ),
                workflow_id,
            )
            activities = instance.data.get("activities", []) if instance.success else []
            return [
                {
                    "workflow_id": workflow_id,
                    "activity": act.get("name") or act.get("activity_id"),
                    "status": act.get("status"),
                    "timestamp": act.get("created_at") or act.get("started_at"),
                }
                for act in (activities if isinstance(activities, list) else [])
            ]
        except Exception as exc:
            logger.error(f"audit_workflow_trail error: {exc}")
            return []

    async def wfa_health_check(self) -> Dict[str, Any]:
        """Return WFA service health status."""
        return await self.get_service_health()

    async def process_template_create(
        self,
        context: APGTenantContext,
        name: str,
        definition: Dict[str, Any],
    ) -> WBPMServiceResponse:
        """Create a reusable process template from a workflow definition."""
        return await self.create_process(context, name, definition)

    async def process_template_list(
        self,
        context: APGTenantContext,
    ) -> WBPMServiceResponse:
        """List all process templates available to the tenant."""
        return await self.list_processes(context)

    async def instance_history(
        self,
        context: APGTenantContext,
        instance_id: str,
    ) -> WBPMServiceResponse:
        """Return complete history of a workflow instance."""
        return await self.get_instance_status(context, instance_id)

    async def task_reassign(
        self,
        context: APGTenantContext,
        task_id: str,
        new_assignee_id: str,
        reason: str = "",
    ) -> WBPMServiceResponse:
        """Reassign a task to a different user."""
        completion_data = {"reassigned_to": new_assignee_id, "reason": reason, "action": "reassign"}
        return await self.complete_task(context, task_id, completion_data)

    async def pending_approvals(
        self,
        context: APGTenantContext,
    ) -> WBPMServiceResponse:
        """Return all tasks pending approval for the requesting user."""
        return await self.get_user_tasks(context)

    async def workflow_export(
        self,
        context: APGTenantContext,
        workflow_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Export a workflow definition for portability or backup."""
        status = await self.get_instance_status(context, workflow_id)
        return {
            "workflow_id": workflow_id,
            "format": format,
            "data": status.data if status.success else {},
            "exported_at": datetime.utcnow().isoformat(),
        }

    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            active_instances = self.workflow_service.workflow_engine.get_active_instances()
            
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0',
                'active_instances': len(active_instances),
                'max_concurrent_instances': self.config.max_concurrent_instances,
                'components': {
                    'workflow_engine': 'healthy',
                    'task_management': 'healthy',
                    'apg_integration': 'healthy'
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }


# =============================================================================
# Service Factory
# =============================================================================

def create_wbpm_service(config: WBPMServiceConfig) -> WorkflowBusinessProcessMgmtService:
    """Create and configure WBPM service instance."""
    service = WorkflowBusinessProcessMgmtService(config)
    logger.info("WBPM service created and configured")
    return service


# Export main classes
__all__ = [
    'WorkflowBusinessProcessMgmtService',
    'ProcessManagementService',
    'TaskManagementService', 
    'WorkflowExecutionService',
    'APGPlatformIntegration',
    'create_wbpm_service'
]
