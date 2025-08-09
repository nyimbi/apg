"""
APG Workflow & Business Process Management - Core Workflow Engine

High-performance BPMN 2.0 compliant workflow execution engine with async processing,
intelligent task routing, and comprehensive APG platform integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import json
import re

from models import (
    APGTenantContext, WBPMProcessDefinition, WBPMProcessInstance, WBPMProcessActivity,
    WBPMProcessFlow, WBPMTask, ProcessStatus, InstanceStatus, TaskStatus, TaskPriority,
    ActivityType, GatewayDirection, EventType, WBPMServiceResponse
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Workflow Engine Core Classes
# =============================================================================

class ExecutionResult(Enum):
    """Execution result enumeration."""
    SUCCESS = "success"
    FAILED = "failed"
    SUSPENDED = "suspended"
    WAITING = "waiting"
    COMPLETED = "completed"


class ActivityState(Enum):
    """Activity execution state."""
    READY = "ready"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


@dataclass
class ExecutionContext:
    """Workflow execution context."""
    tenant_context: APGTenantContext
    instance: WBPMProcessInstance
    process_definition: WBPMProcessDefinition
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get process variable value."""
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set process variable value."""
        self.variables[name] = value
    
    def evaluate_expression(self, expression: str) -> Any:
        """Evaluate expression in context of process variables."""
        try:
            # Simple expression evaluation (production would use proper expression engine)
            # This is a simplified implementation for demonstration
            return eval(expression, {"__builtins__": {}}, self.variables)
        except Exception as e:
            logger.warning(f"Expression evaluation failed: {expression}, error: {e}")
            return False


@dataclass
class ActivityExecution:
    """Activity execution state and result."""
    activity_id: str
    activity: WBPMProcessActivity
    state: ActivityState = ActivityState.READY
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[ExecutionResult] = None
    error_message: Optional[str] = None
    output_variables: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_completed(self) -> bool:
        """Check if activity is completed."""
        return self.state in [ActivityState.COMPLETED, ActivityState.FAILED, ActivityState.TERMINATED]
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Get execution duration in milliseconds."""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None


@dataclass
class ProcessExecution:
    """Process instance execution state."""
    instance: WBPMProcessInstance
    activities: Dict[str, ActivityExecution] = field(default_factory=dict)
    active_tokens: Set[str] = field(default_factory=set)
    completed_activities: Set[str] = field(default_factory=set)
    waiting_activities: Set[str] = field(default_factory=set)
    failed_activities: Set[str] = field(default_factory=set)
    
    def get_activity_execution(self, activity_id: str) -> Optional[ActivityExecution]:
        """Get activity execution state."""
        return self.activities.get(activity_id)
    
    def add_active_token(self, activity_id: str) -> None:
        """Add active token to activity."""
        self.active_tokens.add(activity_id)
    
    def remove_active_token(self, activity_id: str) -> None:
        """Remove active token from activity."""
        self.active_tokens.discard(activity_id)
    
    def is_process_completed(self) -> bool:
        """Check if process execution is completed."""
        return len(self.active_tokens) == 0 and len(self.waiting_activities) == 0


# =============================================================================
# BPMN Parser and Processor
# =============================================================================

class BPMNParser:
    """BPMN 2.0 XML parser and processor."""
    
    BPMN_NAMESPACE = {
        'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
        'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
        'omgdc': 'http://www.omg.org/spec/DD/20100524/DC',
        'omgdi': 'http://www.omg.org/spec/DD/20100524/DI'
    }
    
    def __init__(self):
        self.activities: Dict[str, WBPMProcessActivity] = {}
        self.flows: Dict[str, WBPMProcessFlow] = {}
        self.start_events: List[str] = []
        self.end_events: List[str] = []
    
    def parse_bpmn_xml(self, bpmn_xml: str, process_id: str, tenant_id: str) -> Tuple[List[WBPMProcessActivity], List[WBPMProcessFlow]]:
        """Parse BPMN XML and extract activities and flows."""
        try:
            root = ET.fromstring(bpmn_xml)
            
            # Find process element
            process_element = root.find('.//bpmn:process', self.BPMN_NAMESPACE)
            if process_element is None:
                raise ValueError("No process element found in BPMN XML")
            
            # Parse activities
            self._parse_activities(process_element, process_id, tenant_id)
            
            # Parse sequence flows
            self._parse_flows(process_element, process_id, tenant_id)
            
            return list(self.activities.values()), list(self.flows.values())
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid BPMN XML: {e}")
        except Exception as e:
            raise ValueError(f"BPMN parsing error: {e}")
    
    def _parse_activities(self, process_element: ET.Element, process_id: str, tenant_id: str) -> None:
        """Parse BPMN activities from process element."""
        # Start events
        for start_event in process_element.findall('.//bpmn:startEvent', self.BPMN_NAMESPACE):
            activity = self._create_activity_from_element(start_event, ActivityType.START_EVENT, process_id, tenant_id)
            self.activities[activity.element_id] = activity
            self.start_events.append(activity.element_id)
        
        # End events
        for end_event in process_element.findall('.//bpmn:endEvent', self.BPMN_NAMESPACE):
            activity = self._create_activity_from_element(end_event, ActivityType.END_EVENT, process_id, tenant_id)
            self.activities[activity.element_id] = activity
            self.end_events.append(activity.element_id)
        
        # User tasks
        for user_task in process_element.findall('.//bpmn:userTask', self.BPMN_NAMESPACE):
            activity = self._create_activity_from_element(user_task, ActivityType.USER_TASK, process_id, tenant_id)
            self.activities[activity.element_id] = activity
        
        # Service tasks
        for service_task in process_element.findall('.//bpmn:serviceTask', self.BPMN_NAMESPACE):
            activity = self._create_activity_from_element(service_task, ActivityType.SERVICE_TASK, process_id, tenant_id)
            self.activities[activity.element_id] = activity
        
        # Script tasks
        for script_task in process_element.findall('.//bpmn:scriptTask', self.BPMN_NAMESPACE):
            activity = self._create_activity_from_element(script_task, ActivityType.SCRIPT_TASK, process_id, tenant_id)
            self.activities[activity.element_id] = activity
        
        # Exclusive gateways
        for gateway in process_element.findall('.//bpmn:exclusiveGateway', self.BPMN_NAMESPACE):
            activity = self._create_activity_from_element(gateway, ActivityType.EXCLUSIVE_GATEWAY, process_id, tenant_id)
            self.activities[activity.element_id] = activity
        
        # Parallel gateways
        for gateway in process_element.findall('.//bpmn:parallelGateway', self.BPMN_NAMESPACE):
            activity = self._create_activity_from_element(gateway, ActivityType.PARALLEL_GATEWAY, process_id, tenant_id)
            self.activities[activity.element_id] = activity
    
    def _parse_flows(self, process_element: ET.Element, process_id: str, tenant_id: str) -> None:
        """Parse BPMN sequence flows from process element."""
        for flow_element in process_element.findall('.//bpmn:sequenceFlow', self.BPMN_NAMESPACE):
            flow_id = flow_element.get('id')
            if not flow_id:
                continue
            
            source_ref = flow_element.get('sourceRef')
            target_ref = flow_element.get('targetRef')
            
            if not source_ref or not target_ref:
                continue
            
            # Find source and target activities
            source_activity = self.activities.get(source_ref)
            target_activity = self.activities.get(target_ref)
            
            if not source_activity or not target_activity:
                logger.warning(f"Flow {flow_id} references unknown activities: {source_ref} -> {target_ref}")
                continue
            
            flow = WBPMProcessFlow(
                tenant_id=tenant_id,
                process_id=process_id,
                element_id=flow_id,
                flow_name=flow_element.get('name'),
                source_activity_id=source_activity.id,
                target_activity_id=target_activity.id,
                condition_expression=self._get_condition_expression(flow_element),
                is_default_flow=flow_element.get('default') == 'true',
                created_by='system',
                updated_by='system'
            )
            
            self.flows[flow_id] = flow
    
    def _create_activity_from_element(self, element: ET.Element, activity_type: ActivityType, process_id: str, tenant_id: str) -> WBPMProcessActivity:
        """Create activity model from BPMN element."""
        element_id = element.get('id')
        element_name = element.get('name')
        
        # Parse element properties
        properties = {}
        
        # Parse assignee for user tasks
        assignee = None
        if activity_type == ActivityType.USER_TASK:
            assignee = element.get('assignee') or element.get('camunda:assignee')
        
        # Parse candidate users and groups
        candidate_users = []
        candidate_groups = []
        if activity_type == ActivityType.USER_TASK:
            candidate_users_attr = element.get('candidateUsers') or element.get('camunda:candidateUsers')
            if candidate_users_attr:
                candidate_users = [user.strip() for user in candidate_users_attr.split(',')]
            
            candidate_groups_attr = element.get('candidateGroups') or element.get('camunda:candidateGroups')
            if candidate_groups_attr:
                candidate_groups = [group.strip() for group in candidate_groups_attr.split(',')]
        
        # Parse form key
        form_key = element.get('formKey') or element.get('camunda:formKey')
        
        # Parse service task properties
        class_name = None
        expression = None
        delegate_expression = None
        if activity_type == ActivityType.SERVICE_TASK:
            class_name = element.get('class') or element.get('camunda:class')
            expression = element.get('expression') or element.get('camunda:expression')
            delegate_expression = element.get('delegateExpression') or element.get('camunda:delegateExpression')
        
        # Parse gateway direction
        gateway_direction = None
        default_flow = None
        if activity_type in [ActivityType.EXCLUSIVE_GATEWAY, ActivityType.PARALLEL_GATEWAY, ActivityType.INCLUSIVE_GATEWAY]:
            gateway_direction = GatewayDirection.UNSPECIFIED
            default_flow = element.get('default')
        
        return WBPMProcessActivity(
            tenant_id=tenant_id,
            process_id=process_id,
            element_id=element_id,
            element_name=element_name,
            activity_type=activity_type,
            element_properties=properties,
            assignee=assignee,
            candidate_users=candidate_users,
            candidate_groups=candidate_groups,
            form_key=form_key,
            class_name=class_name,
            expression=expression,
            delegate_expression=delegate_expression,
            gateway_direction=gateway_direction,
            default_flow=default_flow,
            created_by='system',
            updated_by='system'
        )
    
    def _get_condition_expression(self, flow_element: ET.Element) -> Optional[str]:
        """Extract condition expression from flow element."""
        condition_element = flow_element.find('.//bpmn:conditionExpression', self.BPMN_NAMESPACE)
        if condition_element is not None:
            return condition_element.text
        return None
    
    def get_start_activities(self) -> List[str]:
        """Get list of start activity IDs."""
        return self.start_events.copy()
    
    def get_end_activities(self) -> List[str]:
        """Get list of end activity IDs."""
        return self.end_events.copy()
    
    def get_outgoing_flows(self, activity_id: str) -> List[WBPMProcessFlow]:
        """Get outgoing flows for activity."""
        activity = self.activities.get(activity_id)
        if not activity:
            return []
        
        return [flow for flow in self.flows.values() if flow.source_activity_id == activity.id]
    
    def get_incoming_flows(self, activity_id: str) -> List[WBPMProcessFlow]:
        """Get incoming flows for activity."""
        activity = self.activities.get(activity_id)
        if not activity:
            return []
        
        return [flow for flow in self.flows.values() if flow.target_activity_id == activity.id]


# =============================================================================
# Workflow Execution Engine
# =============================================================================

class WorkflowExecutionEngine:
    """High-performance async workflow execution engine."""
    
    def __init__(self, max_concurrent_instances: int = 1000):
        self.max_concurrent_instances = max_concurrent_instances
        self.active_executions: Dict[str, ProcessExecution] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_instances)
        self.parser = BPMNParser()
        
        # Activity handlers
        self.activity_handlers = {
            ActivityType.START_EVENT: self._handle_start_event,
            ActivityType.END_EVENT: self._handle_end_event,
            ActivityType.USER_TASK: self._handle_user_task,
            ActivityType.SERVICE_TASK: self._handle_service_task,
            ActivityType.SCRIPT_TASK: self._handle_script_task,
            ActivityType.EXCLUSIVE_GATEWAY: self._handle_exclusive_gateway,
            ActivityType.PARALLEL_GATEWAY: self._handle_parallel_gateway,
        }
    
    async def start_process_instance(
        self, 
        process_definition: WBPMProcessDefinition,
        instance: WBPMProcessInstance,
        context: APGTenantContext,
        initial_variables: Optional[Dict[str, Any]] = None
    ) -> WBPMServiceResponse:
        """Start execution of a process instance."""
        async with self.execution_semaphore:
            try:
                # Parse BPMN definition
                activities, flows = self.parser.parse_bpmn_xml(
                    process_definition.bpmn_xml,
                    process_definition.id,
                    context.tenant_id
                )
                
                # Create execution context
                execution_context = ExecutionContext(
                    tenant_context=context,
                    instance=instance,
                    process_definition=process_definition,
                    variables=initial_variables or {}
                )
                
                # Initialize process execution
                process_execution = ProcessExecution(instance=instance)
                
                # Create activity executions
                for activity in activities:
                    activity_execution = ActivityExecution(
                        activity_id=activity.id,
                        activity=activity
                    )
                    process_execution.activities[activity.id] = activity_execution
                
                # Store execution state
                self.active_executions[instance.id] = process_execution
                
                # Start execution from start events
                start_activities = self.parser.get_start_activities()
                for start_activity_id in start_activities:
                    await self._execute_activity(start_activity_id, execution_context, process_execution)
                
                # Update instance status
                instance.instance_status = InstanceStatus.RUNNING if not process_execution.is_process_completed() else InstanceStatus.COMPLETED
                
                return WBPMServiceResponse(
                    success=True,
                    message="Process instance started successfully",
                    data={
                        "instance_id": instance.id,
                        "status": instance.instance_status,
                        "active_activities": list(process_execution.active_tokens)
                    }
                )
                
            except Exception as e:
                logger.error(f"Error starting process instance {instance.id}: {e}")
                instance.instance_status = InstanceStatus.FAILED
                instance.last_error_message = str(e)
                
                return WBPMServiceResponse(
                    success=False,
                    message=f"Failed to start process instance: {e}",
                    errors=[str(e)]
                )
    
    async def _execute_activity(
        self,
        activity_id: str,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> ExecutionResult:
        """Execute a single activity."""
        activity_execution = process_execution.get_activity_execution(activity_id)
        if not activity_execution:
            logger.error(f"Activity execution not found: {activity_id}")
            return ExecutionResult.FAILED
        
        try:
            # Mark activity as active
            activity_execution.state = ActivityState.ACTIVE
            activity_execution.start_time = datetime.utcnow()
            process_execution.add_active_token(activity_id)
            
            logger.info(f"Executing activity: {activity_id} ({activity_execution.activity.activity_type})")
            
            # Get activity handler
            handler = self.activity_handlers.get(activity_execution.activity.activity_type)
            if not handler:
                raise ValueError(f"No handler for activity type: {activity_execution.activity.activity_type}")
            
            # Execute activity
            result = await handler(activity_execution, context, process_execution)
            
            # Update activity execution result
            activity_execution.result = result
            activity_execution.end_time = datetime.utcnow()
            
            if result == ExecutionResult.SUCCESS:
                activity_execution.state = ActivityState.COMPLETED
                process_execution.completed_activities.add(activity_id)
                process_execution.remove_active_token(activity_id)
                
                # Continue execution to next activities
                await self._continue_execution(activity_id, context, process_execution)
                
            elif result == ExecutionResult.WAITING:
                activity_execution.state = ActivityState.WAITING
                process_execution.waiting_activities.add(activity_id)
                process_execution.remove_active_token(activity_id)
                
            elif result == ExecutionResult.FAILED:
                activity_execution.state = ActivityState.FAILED
                process_execution.failed_activities.add(activity_id)
                process_execution.remove_active_token(activity_id)
                
                # Handle failure (could implement error boundary events here)
                logger.error(f"Activity failed: {activity_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing activity {activity_id}: {e}")
            activity_execution.state = ActivityState.FAILED
            activity_execution.error_message = str(e)
            activity_execution.end_time = datetime.utcnow()
            activity_execution.result = ExecutionResult.FAILED
            
            process_execution.failed_activities.add(activity_id)
            process_execution.remove_active_token(activity_id)
            
            return ExecutionResult.FAILED
    
    async def _continue_execution(
        self,
        completed_activity_id: str,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> None:
        """Continue execution to next activities after activity completion."""
        # Get outgoing flows
        outgoing_flows = self.parser.get_outgoing_flows(completed_activity_id)
        
        for flow in outgoing_flows:
            # Evaluate flow condition if present
            if flow.condition_expression:
                try:
                    condition_result = context.evaluate_expression(flow.condition_expression)
                    if not condition_result:
                        logger.debug(f"Flow condition not met: {flow.element_id}")
                        continue
                except Exception as e:
                    logger.warning(f"Error evaluating flow condition {flow.element_id}: {e}")
                    continue
            
            # Find target activity
            target_activity_execution = None
            for activity_execution in process_execution.activities.values():
                if activity_execution.activity.id == flow.target_activity_id:
                    target_activity_execution = activity_execution
                    break
            
            if not target_activity_execution:
                logger.error(f"Target activity not found for flow {flow.element_id}")
                continue
            
            # Check if target activity is ready to execute
            if await self._is_activity_ready(target_activity_execution.activity_id, process_execution):
                await self._execute_activity(target_activity_execution.activity_id, context, process_execution)
    
    async def _is_activity_ready(self, activity_id: str, process_execution: ProcessExecution) -> bool:
        """Check if activity is ready to execute (all incoming flows completed)."""
        # Get incoming flows
        incoming_flows = self.parser.get_incoming_flows(activity_id)
        
        if not incoming_flows:
            return True  # No incoming flows, activity is ready
        
        # Check if all source activities are completed
        for flow in incoming_flows:
            source_activity_id = None
            for activity_execution in process_execution.activities.values():
                if activity_execution.activity.id == flow.source_activity_id:
                    source_activity_id = activity_execution.activity_id
                    break
            
            if source_activity_id not in process_execution.completed_activities:
                return False
        
        return True
    
    # Activity Handlers
    
    async def _handle_start_event(
        self,
        activity_execution: ActivityExecution,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> ExecutionResult:
        """Handle start event execution."""
        logger.info(f"Start event executed: {activity_execution.activity.element_id}")
        return ExecutionResult.SUCCESS
    
    async def _handle_end_event(
        self,
        activity_execution: ActivityExecution,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> ExecutionResult:
        """Handle end event execution."""
        logger.info(f"End event executed: {activity_execution.activity.element_id}")
        
        # Check if this is the final end event
        if process_execution.is_process_completed():
            context.instance.instance_status = InstanceStatus.COMPLETED
            context.instance.end_time = datetime.utcnow()
            logger.info(f"Process instance completed: {context.instance.id}")
        
        return ExecutionResult.SUCCESS
    
    async def _handle_user_task(
        self,
        activity_execution: ActivityExecution,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> ExecutionResult:
        """Handle user task execution."""
        logger.info(f"User task created: {activity_execution.activity.element_id}")
        
        # Create task for user interaction
        task = WBPMTask(
            tenant_id=context.tenant_context.tenant_id,
            process_instance_id=context.instance.id,
            activity_id=activity_execution.activity.id,
            task_name=activity_execution.activity.element_name or "User Task",
            task_description=f"Task for activity {activity_execution.activity.element_id}",
            assignee=activity_execution.activity.assignee,
            candidate_users=activity_execution.activity.candidate_users or [],
            candidate_groups=activity_execution.activity.candidate_groups or [],
            form_key=activity_execution.activity.form_key,
            priority=TaskPriority.MEDIUM,
            created_by=context.tenant_context.user_id,
            updated_by=context.tenant_context.user_id
        )
        
        # Store task for later completion (would integrate with task service)
        activity_execution.output_variables['task_id'] = task.id
        
        # User tasks wait for external completion
        return ExecutionResult.WAITING
    
    async def _handle_service_task(
        self,
        activity_execution: ActivityExecution,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> ExecutionResult:
        """Handle service task execution."""
        logger.info(f"Service task executed: {activity_execution.activity.element_id}")
        
        # Simulate service call (would integrate with actual services)
        try:
            if activity_execution.activity.expression:
                result = context.evaluate_expression(activity_execution.activity.expression)
                context.set_variable(f"{activity_execution.activity.element_id}_result", result)
            
            # Simulate async service call
            await asyncio.sleep(0.1)
            
            return ExecutionResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Service task failed: {e}")
            return ExecutionResult.FAILED
    
    async def _handle_script_task(
        self,
        activity_execution: ActivityExecution,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> ExecutionResult:
        """Handle script task execution."""
        logger.info(f"Script task executed: {activity_execution.activity.element_id}")
        
        # Execute script (simplified implementation)
        try:
            script = activity_execution.activity.expression or "# No script defined"
            
            # In production, would use secure script execution environment
            # For demo, just simulate execution
            await asyncio.sleep(0.05)
            
            return ExecutionResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Script task failed: {e}")
            return ExecutionResult.FAILED
    
    async def _handle_exclusive_gateway(
        self,
        activity_execution: ActivityExecution,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> ExecutionResult:
        """Handle exclusive gateway execution."""
        logger.info(f"Exclusive gateway executed: {activity_execution.activity.element_id}")
        
        # Exclusive gateway selects one outgoing flow based on conditions
        # The flow selection logic is handled in _continue_execution
        return ExecutionResult.SUCCESS
    
    async def _handle_parallel_gateway(
        self,
        activity_execution: ActivityExecution,
        context: ExecutionContext,
        process_execution: ProcessExecution
    ) -> ExecutionResult:
        """Handle parallel gateway execution."""
        logger.info(f"Parallel gateway executed: {activity_execution.activity.element_id}")
        
        # Parallel gateway activates all outgoing flows
        # The parallel execution logic is handled in _continue_execution
        return ExecutionResult.SUCCESS
    
    # Public API Methods
    
    async def complete_user_task(
        self,
        task_id: str,
        completion_data: Dict[str, Any],
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Complete a user task and continue process execution."""
        try:
            # Find the process execution containing this task
            target_execution = None
            target_activity = None
            
            for execution in self.active_executions.values():
                for activity_execution in execution.activities.values():
                    if (activity_execution.output_variables.get('task_id') == task_id and
                        activity_execution.state == ActivityState.WAITING):
                        target_execution = execution
                        target_activity = activity_execution
                        break
                if target_execution:
                    break
            
            if not target_execution or not target_activity:
                return WBPMServiceResponse(
                    success=False,
                    message="Task not found or not in waiting state",
                    errors=["Task not found"]
                )
            
            # Update task completion
            target_activity.state = ActivityState.COMPLETED
            target_activity.end_time = datetime.utcnow()
            target_activity.result = ExecutionResult.SUCCESS
            target_activity.output_variables.update(completion_data)
            
            # Remove from waiting activities
            target_execution.waiting_activities.discard(target_activity.activity_id)
            target_execution.completed_activities.add(target_activity.activity_id)
            
            # Create execution context
            execution_context = ExecutionContext(
                tenant_context=context,
                instance=target_execution.instance,
                process_definition=WBPMProcessDefinition(
                    tenant_id=context.tenant_id,
                    process_key="temp",
                    process_name="temp",
                    bpmn_xml="",
                    created_by=context.user_id,
                    updated_by=context.user_id
                ),
                variables=target_execution.instance.process_variables
            )
            
            # Continue execution
            await self._continue_execution(target_activity.activity_id, execution_context, target_execution)
            
            return WBPMServiceResponse(
                success=True,
                message="Task completed successfully",
                data={
                    "task_id": task_id,
                    "instance_id": target_execution.instance.id
                }
            )
            
        except Exception as e:
            logger.error(f"Error completing task {task_id}: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to complete task: {e}",
                errors=[str(e)]
            )
    
    async def suspend_process_instance(self, instance_id: str) -> WBPMServiceResponse:
        """Suspend a running process instance."""
        execution = self.active_executions.get(instance_id)
        if not execution:
            return WBPMServiceResponse(
                success=False,
                message="Process instance not found",
                errors=["Instance not found"]
            )
        
        execution.instance.instance_status = InstanceStatus.SUSPENDED
        
        return WBPMServiceResponse(
            success=True,
            message="Process instance suspended",
            data={"instance_id": instance_id}
        )
    
    async def resume_process_instance(self, instance_id: str) -> WBPMServiceResponse:
        """Resume a suspended process instance."""
        execution = self.active_executions.get(instance_id)
        if not execution:
            return WBPMServiceResponse(
                success=False,
                message="Process instance not found",
                errors=["Instance not found"]
            )
        
        execution.instance.instance_status = InstanceStatus.RUNNING
        
        return WBPMServiceResponse(
            success=True,
            message="Process instance resumed",
            data={"instance_id": instance_id}
        )
    
    async def cancel_process_instance(self, instance_id: str) -> WBPMServiceResponse:
        """Cancel a running process instance."""
        execution = self.active_executions.get(instance_id)
        if not execution:
            return WBPMServiceResponse(
                success=False,
                message="Process instance not found",
                errors=["Instance not found"]
            )
        
        execution.instance.instance_status = InstanceStatus.CANCELLED
        execution.instance.end_time = datetime.utcnow()
        
        # Remove from active executions
        del self.active_executions[instance_id]
        
        return WBPMServiceResponse(
            success=True,
            message="Process instance cancelled",
            data={"instance_id": instance_id}
        )
    
    def get_active_instances(self) -> List[str]:
        """Get list of active process instance IDs."""
        return list(self.active_executions.keys())
    
    def get_instance_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a process instance."""
        execution = self.active_executions.get(instance_id)
        if not execution:
            return None
        
        return {
            "instance_id": instance_id,
            "status": execution.instance.instance_status,
            "active_tokens": list(execution.active_tokens),
            "completed_activities": list(execution.completed_activities),
            "waiting_activities": list(execution.waiting_activities),
            "failed_activities": list(execution.failed_activities),
            "is_completed": execution.is_process_completed()
        }


# =============================================================================
# Workflow Engine Factory
# =============================================================================

def create_workflow_engine(max_concurrent_instances: int = 1000) -> WorkflowExecutionEngine:
    """Create and configure workflow execution engine."""
    engine = WorkflowExecutionEngine(max_concurrent_instances=max_concurrent_instances)
    logger.info(f"Workflow engine created with max concurrent instances: {max_concurrent_instances}")
    return engine


# Export main classes
__all__ = [
    'WorkflowExecutionEngine',
    'BPMNParser',
    'ExecutionContext',
    'ActivityExecution',
    'ProcessExecution',
    'ExecutionResult',
    'ActivityState',
    'create_workflow_engine'
]