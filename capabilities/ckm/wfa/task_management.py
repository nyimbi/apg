"""
APG Workflow & Business Process Management - Task Management System

Intelligent task management with AI-powered routing, assignment optimization,
and comprehensive task lifecycle management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import hashlib

from models import (
    APGTenantContext, WBPMTask, WBPMTaskHistory, WBPMTaskComment,
    TaskStatus, TaskPriority, WBPMServiceResponse, WBPMPagedResponse
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Task Management Core Classes
# =============================================================================

class AssignmentStrategy(str, Enum):
    """Task assignment strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    SKILL_BASED = "skill_based"
    PERFORMANCE_BASED = "performance_based"
    AI_OPTIMIZED = "ai_optimized"
    MANUAL = "manual"


class EscalationTrigger(str, Enum):
    """Escalation trigger enumeration."""
    TIMEOUT = "timeout"
    MISSED_SLA = "missed_sla"
    ERROR_COUNT = "error_count"
    MANUAL = "manual"
    PRIORITY_CHANGE = "priority_change"


@dataclass
class TaskAssignmentCriteria:
    """Criteria for task assignment."""
    required_skills: List[str] = field(default_factory=list)
    required_roles: List[str] = field(default_factory=list)
    preferred_users: List[str] = field(default_factory=list)
    excluded_users: List[str] = field(default_factory=list)
    workload_limit: Optional[int] = None
    sla_requirements: Optional[Dict[str, Any]] = None
    geographic_constraints: Optional[Dict[str, Any]] = None


@dataclass
class UserProfile:
    """User profile for task assignment."""
    user_id: str
    skills: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    current_workload: int = 0
    availability_schedule: Dict[str, Any] = field(default_factory=dict)
    geographic_location: Optional[str] = None
    language_preferences: List[str] = field(default_factory=list)
    
    def calculate_fitness_score(self, criteria: TaskAssignmentCriteria) -> float:
        """Calculate fitness score for task assignment."""
        score = 0.0
        
        # Skill matching
        if criteria.required_skills:
            matching_skills = set(self.skills) & set(criteria.required_skills)
            skill_score = len(matching_skills) / len(criteria.required_skills)
            score += skill_score * 0.4
        
        # Role matching
        if criteria.required_roles:
            matching_roles = set(self.roles) & set(criteria.required_roles)
            role_score = len(matching_roles) / len(criteria.required_roles)
            score += role_score * 0.3
        
        # Performance score
        score += self.performance_score * 0.2
        
        # Workload consideration (inverse relationship)
        if criteria.workload_limit and self.current_workload > 0:
            workload_score = max(0, 1 - (self.current_workload / criteria.workload_limit))
            score += workload_score * 0.1
        
        return min(score, 1.0)


@dataclass
class TaskQueueMetrics:
    """Task queue performance metrics."""
    total_tasks: int = 0
    pending_tasks: int = 0
    in_progress_tasks: int = 0
    completed_tasks: int = 0
    overdue_tasks: int = 0
    average_completion_time: float = 0.0
    average_waiting_time: float = 0.0
    sla_compliance_rate: float = 0.0


@dataclass
class EscalationRule:
    """Task escalation rule definition."""
    rule_id: str
    trigger: EscalationTrigger
    condition: str  # Expression to evaluate
    escalation_target: str  # User or group to escalate to
    notification_message: str
    delay_minutes: int = 0
    max_escalations: int = 3
    is_active: bool = True


# =============================================================================
# Intelligent Task Router
# =============================================================================

class AITaskRouter:
    """AI-powered task routing and assignment system."""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.assignment_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, float] = {}
    
    async def assign_task(
        self,
        task: WBPMTask,
        criteria: TaskAssignmentCriteria,
        strategy: AssignmentStrategy = AssignmentStrategy.AI_OPTIMIZED
    ) -> Optional[str]:
        """Assign task to optimal user based on strategy and criteria."""
        try:
            # Get eligible users
            eligible_users = await self._get_eligible_users(criteria)
            
            if not eligible_users:
                logger.warning(f"No eligible users found for task {task.id}")
                return None
            
            # Apply assignment strategy
            selected_user = await self._apply_assignment_strategy(
                task, eligible_users, strategy, criteria
            )
            
            if selected_user:
                # Update user workload
                if selected_user in self.user_profiles:
                    self.user_profiles[selected_user].current_workload += 1
                
                # Record assignment
                await self._record_assignment(task.id, selected_user, strategy)
                
                logger.info(f"Task {task.id} assigned to user {selected_user} using {strategy}")
            
            return selected_user
            
        except Exception as e:
            logger.error(f"Error assigning task {task.id}: {e}")
            return None
    
    async def _get_eligible_users(self, criteria: TaskAssignmentCriteria) -> List[str]:
        """Get list of users eligible for task assignment."""
        eligible_users = []
        
        for user_id, profile in self.user_profiles.items():
            # Check excluded users
            if user_id in criteria.excluded_users:
                continue
            
            # Check workload limit
            if criteria.workload_limit and profile.current_workload >= criteria.workload_limit:
                continue
            
            # Check required skills
            if criteria.required_skills:
                if not set(criteria.required_skills).issubset(set(profile.skills)):
                    continue
            
            # Check required roles
            if criteria.required_roles:
                if not set(criteria.required_roles).intersection(set(profile.roles)):
                    continue
            
            eligible_users.append(user_id)
        
        return eligible_users
    
    async def _apply_assignment_strategy(
        self,
        task: WBPMTask,
        eligible_users: List[str],
        strategy: AssignmentStrategy,
        criteria: TaskAssignmentCriteria
    ) -> Optional[str]:
        """Apply specific assignment strategy to select user."""
        if strategy == AssignmentStrategy.ROUND_ROBIN:
            return await self._round_robin_assignment(eligible_users)
        
        elif strategy == AssignmentStrategy.LEAST_LOADED:
            return await self._least_loaded_assignment(eligible_users)
        
        elif strategy == AssignmentStrategy.SKILL_BASED:
            return await self._skill_based_assignment(eligible_users, criteria)
        
        elif strategy == AssignmentStrategy.PERFORMANCE_BASED:
            return await self._performance_based_assignment(eligible_users)
        
        elif strategy == AssignmentStrategy.AI_OPTIMIZED:
            return await self._ai_optimized_assignment(task, eligible_users, criteria)
        
        elif strategy == AssignmentStrategy.MANUAL:
            # Manual assignment requires external input
            return None
        
        return None
    
    async def _round_robin_assignment(self, eligible_users: List[str]) -> str:
        """Round-robin assignment strategy."""
        if not eligible_users:
            return None
        
        # Simple round-robin based on assignment count
        assignment_counts = {}
        for record in self.assignment_history:
            user = record.get('user_id')
            if user in eligible_users:
                assignment_counts[user] = assignment_counts.get(user, 0) + 1
        
        # Find user with minimum assignments
        min_assignments = min(assignment_counts.get(user, 0) for user in eligible_users)
        candidates = [user for user in eligible_users if assignment_counts.get(user, 0) == min_assignments]
        
        return random.choice(candidates)
    
    async def _least_loaded_assignment(self, eligible_users: List[str]) -> str:
        """Least loaded assignment strategy."""
        if not eligible_users:
            return None
        
        # Find user with minimum current workload
        min_workload = min(
            self.user_profiles.get(user, UserProfile(user)).current_workload
            for user in eligible_users
        )
        
        candidates = [
            user for user in eligible_users
            if self.user_profiles.get(user, UserProfile(user)).current_workload == min_workload
        ]
        
        return random.choice(candidates)
    
    async def _skill_based_assignment(
        self,
        eligible_users: List[str],
        criteria: TaskAssignmentCriteria
    ) -> str:
        """Skill-based assignment strategy."""
        if not eligible_users:
            return None
        
        # Calculate skill match scores
        user_scores = []
        for user_id in eligible_users:
            profile = self.user_profiles.get(user_id, UserProfile(user_id))
            fitness_score = profile.calculate_fitness_score(criteria)
            user_scores.append((user_id, fitness_score))
        
        # Sort by fitness score (descending)
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return user with highest fitness score
        return user_scores[0][0] if user_scores else None
    
    async def _performance_based_assignment(self, eligible_users: List[str]) -> str:
        """Performance-based assignment strategy."""
        if not eligible_users:
            return None
        
        # Sort users by performance score (descending)
        user_scores = [
            (user_id, self.user_profiles.get(user_id, UserProfile(user_id)).performance_score)
            for user_id in eligible_users
        ]
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        return user_scores[0][0] if user_scores else None
    
    async def _ai_optimized_assignment(
        self,
        task: WBPMTask,
        eligible_users: List[str],
        criteria: TaskAssignmentCriteria
    ) -> str:
        """AI-optimized assignment using machine learning."""
        if not eligible_users:
            return None
        
        # Combine multiple factors for optimal assignment
        user_scores = []
        
        for user_id in eligible_users:
            profile = self.user_profiles.get(user_id, UserProfile(user_id))
            
            # Calculate composite score
            fitness_score = profile.calculate_fitness_score(criteria)
            performance_weight = 0.3
            workload_weight = 0.2
            historical_weight = 0.1
            
            # Workload factor (inverse)
            workload_factor = 1.0 - min(profile.current_workload / 10.0, 1.0)
            
            # Historical success factor
            historical_factor = self._get_historical_success_rate(user_id, task.task_name)
            
            composite_score = (
                fitness_score * (1.0 - performance_weight - workload_weight - historical_weight) +
                profile.performance_score * performance_weight +
                workload_factor * workload_weight +
                historical_factor * historical_weight
            )
            
            user_scores.append((user_id, composite_score))
        
        # Sort by composite score and add some randomization for top candidates
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top 20% with weighted random selection
        top_count = max(1, len(user_scores) // 5)
        top_candidates = user_scores[:top_count]
        
        # Weighted random selection
        weights = [score for _, score in top_candidates]
        total_weight = sum(weights)
        
        if total_weight > 0:
            rand_value = random.uniform(0, total_weight)
            cumulative_weight = 0
            
            for user_id, weight in top_candidates:
                cumulative_weight += weight
                if rand_value <= cumulative_weight:
                    return user_id
        
        return top_candidates[0][0] if top_candidates else None
    
    def _get_historical_success_rate(self, user_id: str, task_type: str) -> float:
        """Get historical success rate for user and task type."""
        # Simplified implementation - would use actual historical data
        cache_key = f"{user_id}:{task_type}"
        return self.performance_cache.get(cache_key, 0.8)
    
    async def _record_assignment(self, task_id: str, user_id: str, strategy: AssignmentStrategy) -> None:
        """Record task assignment for analytics."""
        assignment_record = {
            'task_id': task_id,
            'user_id': user_id,
            'strategy': strategy,
            'timestamp': datetime.utcnow(),
            'assignment_id': hashlib.md5(f"{task_id}:{user_id}".encode()).hexdigest()[:16]
        }
        
        self.assignment_history.append(assignment_record)
        
        # Keep only recent history (last 10000 assignments)
        if len(self.assignment_history) > 10000:
            self.assignment_history = self.assignment_history[-10000:]
    
    def update_user_profile(self, user_id: str, profile: UserProfile) -> None:
        """Update user profile information."""
        self.user_profiles[user_id] = profile
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile information."""
        return self.user_profiles.get(user_id)


# =============================================================================
# Task Queue Manager
# =============================================================================

class TaskQueueManager:
    """Advanced task queue management with optimization."""
    
    def __init__(self):
        self.task_queues: Dict[str, List[WBPMTask]] = {}
        self.priority_weights = {
            TaskPriority.CRITICAL: 1000,
            TaskPriority.HIGH: 100,
            TaskPriority.MEDIUM: 10,
            TaskPriority.LOW: 1
        }
    
    async def add_task_to_queue(self, task: WBPMTask, queue_name: str = "default") -> None:
        """Add task to specified queue with priority ordering."""
        if queue_name not in self.task_queues:
            self.task_queues[queue_name] = []
        
        queue = self.task_queues[queue_name]
        
        # Insert task in priority order
        inserted = False
        for i, existing_task in enumerate(queue):
            if self._calculate_task_score(task) > self._calculate_task_score(existing_task):
                queue.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            queue.append(task)
        
        logger.info(f"Task {task.id} added to queue {queue_name} (position: {queue.index(task)})")
    
    async def get_next_task(self, queue_name: str = "default", user_id: Optional[str] = None) -> Optional[WBPMTask]:
        """Get next task from queue for specified user."""
        queue = self.task_queues.get(queue_name, [])
        
        for i, task in enumerate(queue):
            # Check if task is eligible for user
            if self._is_task_eligible_for_user(task, user_id):
                return queue.pop(i)
        
        return None
    
    async def remove_task_from_queue(self, task_id: str, queue_name: str = "default") -> bool:
        """Remove task from queue."""
        queue = self.task_queues.get(queue_name, [])
        
        for i, task in enumerate(queue):
            if task.id == task_id:
                queue.pop(i)
                logger.info(f"Task {task_id} removed from queue {queue_name}")
                return True
        
        return False
    
    async def get_queue_metrics(self, queue_name: str = "default") -> TaskQueueMetrics:
        """Get queue performance metrics."""
        queue = self.task_queues.get(queue_name, [])
        
        metrics = TaskQueueMetrics()
        metrics.total_tasks = len(queue)
        
        now = datetime.utcnow()
        
        for task in queue:
            if task.task_status == TaskStatus.CREATED or task.task_status == TaskStatus.READY:
                metrics.pending_tasks += 1
            elif task.task_status == TaskStatus.IN_PROGRESS:
                metrics.in_progress_tasks += 1
            elif task.task_status == TaskStatus.COMPLETED:
                metrics.completed_tasks += 1
            
            # Check if task is overdue
            if task.due_date and task.due_date < now:
                metrics.overdue_tasks += 1
        
        return metrics
    
    def _calculate_task_score(self, task: WBPMTask) -> float:
        """Calculate task priority score for queue ordering."""
        base_score = self.priority_weights.get(task.priority, 1)
        
        # Add urgency factor based on due date
        urgency_factor = 1.0
        if task.due_date:
            time_to_due = (task.due_date - datetime.utcnow()).total_seconds()
            if time_to_due > 0:
                # More urgent as due date approaches
                urgency_factor = max(1.0, 86400 / max(time_to_due, 3600))  # 24 hours / time remaining
            else:
                # Overdue tasks get highest urgency
                urgency_factor = 10.0
        
        return base_score * urgency_factor
    
    def _is_task_eligible_for_user(self, task: WBPMTask, user_id: Optional[str]) -> bool:
        """Check if task is eligible for user assignment."""
        if not user_id:
            return True
        
        # Check if task is specifically assigned to user
        if task.assignee and task.assignee != user_id:
            return False
        
        # Check if user is in candidate users
        if task.candidate_users and user_id not in task.candidate_users:
            return False
        
        # Additional eligibility checks could be added here
        return True
    
    def get_queue_status(self, queue_name: str = "default") -> Dict[str, Any]:
        """Get current queue status."""
        queue = self.task_queues.get(queue_name, [])
        
        status_counts = {}
        for task in queue:
            status = task.task_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "queue_name": queue_name,
            "total_tasks": len(queue),
            "status_breakdown": status_counts,
            "next_task_id": queue[0].id if queue else None
        }


# =============================================================================
# Escalation Engine
# =============================================================================

class EscalationEngine:
    """Task escalation management with configurable rules."""
    
    def __init__(self):
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.escalation_history: List[Dict[str, Any]] = []
    
    async def check_escalation_triggers(self, task: WBPMTask) -> List[EscalationRule]:
        """Check if task meets any escalation triggers."""
        triggered_rules = []
        
        for rule in self.escalation_rules.values():
            if not rule.is_active:
                continue
            
            if await self._evaluate_escalation_condition(task, rule):
                triggered_rules.append(rule)
        
        return triggered_rules
    
    async def escalate_task(
        self,
        task: WBPMTask,
        rule: EscalationRule,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Escalate task according to rule."""
        try:
            # Record escalation
            escalation_record = {
                'escalation_id': f"esc_{task.id}_{rule.rule_id}",
                'task_id': task.id,
                'rule_id': rule.rule_id,
                'trigger': rule.trigger,
                'escalation_target': rule.escalation_target,
                'timestamp': datetime.utcnow(),
                'tenant_id': context.tenant_id
            }
            
            self.escalation_history.append(escalation_record)
            
            # Update task assignment
            if rule.escalation_target.startswith('user:'):
                new_assignee = rule.escalation_target[5:]  # Remove 'user:' prefix
                task.assignee = new_assignee
                task.updated_by = context.user_id
                task.updated_at = datetime.utcnow()
            
            # Increase task priority if not already at maximum
            if task.priority != TaskPriority.CRITICAL:
                if task.priority == TaskPriority.LOW:
                    task.priority = TaskPriority.MEDIUM
                elif task.priority == TaskPriority.MEDIUM:
                    task.priority = TaskPriority.HIGH
                elif task.priority == TaskPriority.HIGH:
                    task.priority = TaskPriority.CRITICAL
            
            logger.info(f"Task {task.id} escalated using rule {rule.rule_id}")
            
            return WBPMServiceResponse(
                success=True,
                message=f"Task escalated successfully using rule {rule.rule_id}",
                data={
                    "escalation_id": escalation_record['escalation_id'],
                    "new_assignee": task.assignee,
                    "new_priority": task.priority
                }
            )
            
        except Exception as e:
            logger.error(f"Error escalating task {task.id}: {e}")
            return WBPMServiceResponse(
                success=False,
                message=f"Failed to escalate task: {e}",
                errors=[str(e)]
            )
    
    async def _evaluate_escalation_condition(self, task: WBPMTask, rule: EscalationRule) -> bool:
        """Evaluate if escalation condition is met."""
        try:
            if rule.trigger == EscalationTrigger.TIMEOUT:
                # Check if task has been in progress too long
                if task.claim_time:
                    elapsed_time = datetime.utcnow() - task.claim_time
                    timeout_minutes = int(rule.condition) if rule.condition.isdigit() else 60
                    return elapsed_time > timedelta(minutes=timeout_minutes)
            
            elif rule.trigger == EscalationTrigger.MISSED_SLA:
                # Check if task has missed SLA
                if task.due_date:
                    return datetime.utcnow() > task.due_date
            
            elif rule.trigger == EscalationTrigger.PRIORITY_CHANGE:
                # Check if task priority has changed to critical
                return task.priority == TaskPriority.CRITICAL
            
            # Additional trigger evaluations would be implemented here
            
            return False
            
        except Exception as e:
            logger.warning(f"Error evaluating escalation condition for rule {rule.rule_id}: {e}")
            return False
    
    def add_escalation_rule(self, rule: EscalationRule) -> None:
        """Add new escalation rule."""
        self.escalation_rules[rule.rule_id] = rule
        logger.info(f"Escalation rule added: {rule.rule_id}")
    
    def remove_escalation_rule(self, rule_id: str) -> bool:
        """Remove escalation rule."""
        if rule_id in self.escalation_rules:
            del self.escalation_rules[rule_id]
            logger.info(f"Escalation rule removed: {rule_id}")
            return True
        return False
    
    def get_escalation_history(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get escalation history for task or all tasks."""
        if task_id:
            return [record for record in self.escalation_history if record['task_id'] == task_id]
        return self.escalation_history.copy()


# =============================================================================
# Task Performance Tracker
# =============================================================================

class TaskPerformanceTracker:
    """Track and analyze task performance metrics."""
    
    def __init__(self):
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}
    
    async def track_task_completion(
        self,
        task: WBPMTask,
        completion_time: datetime,
        quality_score: Optional[float] = None
    ) -> None:
        """Track task completion metrics."""
        if task.claim_time:
            processing_time = (completion_time - task.claim_time).total_seconds() / 60  # minutes
        else:
            processing_time = 0
        
        if task.create_time:
            total_time = (completion_time - task.create_time).total_seconds() / 60  # minutes
        else:
            total_time = processing_time
        
        performance_record = {
            'task_id': task.id,
            'user_id': task.assignee,
            'task_type': task.task_name,
            'priority': task.priority,
            'processing_time_minutes': processing_time,
            'total_time_minutes': total_time,
            'quality_score': quality_score,
            'completion_time': completion_time,
            'was_overdue': task.due_date and completion_time > task.due_date if task.due_date else False
        }
        
        user_id = task.assignee or 'unassigned'
        if user_id not in self.performance_data:
            self.performance_data[user_id] = []
        
        self.performance_data[user_id].append(performance_record)
        
        # Keep only recent data (last 1000 records per user)
        if len(self.performance_data[user_id]) > 1000:
            self.performance_data[user_id] = self.performance_data[user_id][-1000:]
    
    async def calculate_user_performance_score(self, user_id: str) -> float:
        """Calculate overall performance score for user."""
        user_data = self.performance_data.get(user_id, [])
        if not user_data:
            return 0.5  # Default neutral score
        
        # Recent performance is weighted more heavily
        recent_data = user_data[-100:]  # Last 100 tasks
        
        if not recent_data:
            return 0.5
        
        # Calculate various performance factors
        total_tasks = len(recent_data)
        completed_on_time = sum(1 for record in recent_data if not record['was_overdue'])
        avg_quality = sum(record['quality_score'] or 0.7 for record in recent_data) / total_tasks
        
        # SLA compliance score
        sla_score = completed_on_time / total_tasks
        
        # Quality score (normalized)
        quality_score = min(avg_quality, 1.0)
        
        # Combined performance score
        performance_score = (sla_score * 0.6) + (quality_score * 0.4)
        
        return round(performance_score, 3)
    
    async def get_task_type_metrics(self, task_type: str) -> Dict[str, Any]:
        """Get performance metrics for specific task type."""
        all_records = []
        for user_data in self.performance_data.values():
            task_records = [record for record in user_data if record['task_type'] == task_type]
            all_records.extend(task_records)
        
        if not all_records:
            return {}
        
        total_tasks = len(all_records)
        avg_processing_time = sum(record['processing_time_minutes'] for record in all_records) / total_tasks
        avg_total_time = sum(record['total_time_minutes'] for record in all_records) / total_tasks
        on_time_count = sum(1 for record in all_records if not record['was_overdue'])
        
        return {
            'task_type': task_type,
            'total_tasks': total_tasks,
            'average_processing_time_minutes': round(avg_processing_time, 2),
            'average_total_time_minutes': round(avg_total_time, 2),
            'sla_compliance_rate': round(on_time_count / total_tasks, 3),
            'completion_rate': 1.0  # All tracked tasks are completed
        }


# =============================================================================
# Task Management Service Factory
# =============================================================================

def create_task_management_components() -> Tuple[AITaskRouter, TaskQueueManager, EscalationEngine, TaskPerformanceTracker]:
    """Create and configure task management components."""
    router = AITaskRouter()
    queue_manager = TaskQueueManager()
    escalation_engine = EscalationEngine()
    performance_tracker = TaskPerformanceTracker()
    
    # Add default escalation rules
    timeout_rule = EscalationRule(
        rule_id="default_timeout",
        trigger=EscalationTrigger.TIMEOUT,
        condition="60",  # 60 minutes
        escalation_target="user:supervisor",
        notification_message="Task has exceeded timeout threshold",
        delay_minutes=0
    )
    escalation_engine.add_escalation_rule(timeout_rule)
    
    sla_rule = EscalationRule(
        rule_id="default_sla",
        trigger=EscalationTrigger.MISSED_SLA,
        condition="true",
        escalation_target="user:manager",
        notification_message="Task has missed SLA deadline",
        delay_minutes=5
    )
    escalation_engine.add_escalation_rule(sla_rule)
    
    logger.info("Task management components created and configured")
    
    return router, queue_manager, escalation_engine, performance_tracker


# Export main classes
__all__ = [
    'AITaskRouter',
    'TaskQueueManager', 
    'EscalationEngine',
    'TaskPerformanceTracker',
    'AssignmentStrategy',
    'TaskAssignmentCriteria',
    'UserProfile',
    'EscalationRule',
    'EscalationTrigger',
    'TaskQueueMetrics',
    'create_task_management_components'
]