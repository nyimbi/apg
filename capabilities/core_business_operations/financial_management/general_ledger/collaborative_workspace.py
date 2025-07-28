"""
APG Financial Management General Ledger - Real-Time Collaborative Workspace

Revolutionary collaborative workspace that transforms GL work from isolated individual
tasks into seamless team collaboration with real-time awareness and smart workflows.

Features:
- Real-time presence and activity awareness
- Smart conflict resolution and merging
- Contextual collaboration tools
- Workflow orchestration and handoffs
- Team performance analytics
- Intelligent work distribution

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class UserActivity(Enum):
    """Types of user activities in the workspace"""
    VIEWING = "viewing"
    EDITING = "editing"
    CREATING = "creating"
    REVIEWING = "reviewing"
    APPROVING = "approving"
    COMMENTING = "commenting"
    IDLE = "idle"
    AWAY = "away"


class WorkflowStatus(Enum):
    """Workflow step statuses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class UserPresence:
    """Real-time user presence information"""
    user_id: str
    user_name: str
    avatar_url: str
    activity: UserActivity
    current_location: str  # which record/page they're on
    last_seen: datetime
    cursor_position: Optional[Dict[str, Any]] = None
    active_selections: List[str] = None
    typing_indicator: bool = False


@dataclass
class CollaborationContext:
    """Context for collaborative work"""
    workspace_id: str
    entity_type: str  # journal_entry, account, period_close, etc.
    entity_id: str
    active_users: List[UserPresence]
    pending_changes: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    workflow_state: Dict[str, Any]


@dataclass
class SmartWorkflowStep:
    """Individual step in a smart workflow"""
    step_id: str
    name: str
    description: str
    assigned_to: Optional[str]
    assigned_role: Optional[str]
    status: WorkflowStatus
    dependencies: List[str]
    estimated_duration: timedelta
    actual_duration: Optional[timedelta]
    auto_assign_criteria: Dict[str, Any]
    validation_rules: List[Dict[str, Any]]
    required_skills: List[str]


class CollaborativeWorkspace:
    """Real-time collaborative workspace for GL operations"""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.active_sessions: Dict[str, UserPresence] = {}
        self.workspace_contexts: Dict[str, CollaborationContext] = {}
        self.workflow_engine = SmartWorkflowEngine()
        self.conflict_resolver = ConflictResolver()
        self.presence_manager = PresenceManager()
        
        logger.info(f"Collaborative workspace initialized for tenant {tenant_id}")
    
    async def join_workspace(self, user_id: str, user_name: str, 
                           entity_type: str, entity_id: str) -> CollaborationContext:
        """
        🎯 GAME CHANGER #2A: Real-Time Presence Awareness
        
        Users can see who else is working on the same data in real-time:
        - Live cursors and selections
        - Activity indicators (editing, reviewing, etc.)
        - Typing indicators for comments/descriptions
        - Smart notifications when conflicts might occur
        """
        try:
            workspace_id = f"{entity_type}:{entity_id}"
            
            # Create user presence
            presence = UserPresence(
                user_id=user_id,
                user_name=user_name,
                avatar_url=f"/api/users/{user_id}/avatar",
                activity=UserActivity.VIEWING,
                current_location=f"/{entity_type}/{entity_id}",
                last_seen=datetime.now(timezone.utc),
                active_selections=[]
            )
            
            # Add to active sessions
            self.active_sessions[user_id] = presence
            
            # Get or create workspace context
            if workspace_id not in self.workspace_contexts:
                self.workspace_contexts[workspace_id] = CollaborationContext(
                    workspace_id=workspace_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    active_users=[],
                    pending_changes=[],
                    conflicts=[],
                    workflow_state={}
                )
            
            context = self.workspace_contexts[workspace_id]
            
            # Add user to workspace
            context.active_users = [u for u in context.active_users if u.user_id != user_id]
            context.active_users.append(presence)
            
            # Notify other users
            await self._notify_user_joined(workspace_id, presence)
            
            # Initialize smart workflow if needed
            if not context.workflow_state:
                context.workflow_state = await self.workflow_engine.initialize_workflow(
                    entity_type, entity_id, user_id
                )
            
            logger.info(f"User {user_name} joined workspace {workspace_id}")
            return context
            
        except Exception as e:
            logger.error(f"Error joining workspace: {e}")
            raise
    
    async def update_user_activity(self, user_id: str, activity: UserActivity,
                                 location: str, cursor_position: Dict[str, Any] = None):
        """
        🎯 GAME CHANGER #2B: Live Activity Tracking
        
        Track and broadcast user activities in real-time:
        - What field they're editing
        - Where their cursor is
        - What they're selecting
        - If they're typing
        """
        try:
            if user_id in self.active_sessions:
                presence = self.active_sessions[user_id]
                presence.activity = activity
                presence.current_location = location
                presence.cursor_position = cursor_position
                presence.last_seen = datetime.now(timezone.utc)
                
                # Broadcast to other users in same workspace
                await self._broadcast_presence_update(user_id, presence)
                
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
    
    async def propose_change(self, user_id: str, workspace_id: str, 
                           change_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 GAME CHANGER #2C: Smart Conflict Prevention
        
        Instead of overwriting changes, intelligently merge them:
        - Detect potential conflicts before they happen
        - Suggest merge strategies
        - Auto-resolve non-conflicting changes
        - Provide conflict resolution UI
        """
        try:
            context = self.workspace_contexts.get(workspace_id)
            if not context:
                raise ValueError(f"Workspace {workspace_id} not found")
            
            # Check for conflicts with pending changes
            conflicts = await self.conflict_resolver.detect_conflicts(
                change_data, context.pending_changes
            )
            
            if conflicts:
                # Handle conflicts intelligently
                resolution = await self.conflict_resolver.suggest_resolution(conflicts)
                
                return {
                    "status": "conflict_detected",
                    "conflicts": conflicts,
                    "suggested_resolution": resolution,
                    "can_auto_resolve": resolution.get("auto_resolvable", False)
                }
            
            # No conflicts - add to pending changes
            change_id = str(uuid.uuid4())
            change_record = {
                "change_id": change_id,
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": change_data,
                "status": "pending"
            }
            
            context.pending_changes.append(change_record)
            
            # Notify other users of pending change
            await self._notify_pending_change(workspace_id, change_record)
            
            return {
                "status": "accepted",
                "change_id": change_id,
                "requires_approval": await self._requires_approval(change_data)
            }
            
        except Exception as e:
            logger.error(f"Error proposing change: {e}")
            raise
    
    async def start_smart_workflow(self, entity_type: str, entity_id: str,
                                 workflow_type: str, initiator_id: str) -> Dict[str, Any]:
        """
        🎯 GAME CHANGER #2D: Intelligent Workflow Orchestration
        
        Automatically orchestrates complex GL processes:
        - Period close workflows with dependencies
        - Multi-step approval processes
        - Automated work distribution based on skills/availability
        - Smart escalation and notifications
        """
        try:
            workflow = await self.workflow_engine.create_workflow(
                workflow_type, entity_type, entity_id, initiator_id
            )
            
            # Auto-assign initial steps based on criteria
            await self.workflow_engine.auto_assign_steps(workflow)
            
            # Send notifications to assigned users
            await self._notify_workflow_assignments(workflow)
            
            return {
                "workflow_id": workflow["workflow_id"],
                "steps": workflow["steps"],
                "estimated_completion": workflow["estimated_completion"],
                "assigned_users": workflow["assigned_users"]
            }
            
        except Exception as e:
            logger.error(f"Error starting smart workflow: {e}")
            raise


class SmartWorkflowEngine:
    """Intelligent workflow orchestration engine"""
    
    def __init__(self):
        self.workflow_templates = self._load_workflow_templates()
        self.user_skills = {}  # Would be loaded from user management
        self.user_availability = {}  # Would be loaded from calendar/workload
    
    async def create_workflow(self, workflow_type: str, entity_type: str,
                            entity_id: str, initiator_id: str) -> Dict[str, Any]:
        """Create a new smart workflow instance"""
        
        template = self.workflow_templates.get(workflow_type)
        if not template:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        workflow_id = str(uuid.uuid4())
        
        # Create workflow instance from template
        workflow = {
            "workflow_id": workflow_id,
            "type": workflow_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "initiator_id": initiator_id,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "steps": self._instantiate_steps(template["steps"], entity_id),
            "estimated_completion": self._calculate_estimated_completion(template["steps"]),
            "assigned_users": []
        }
        
        return workflow
    
    async def auto_assign_steps(self, workflow: Dict[str, Any]):
        """Automatically assign workflow steps to best available users"""
        
        for step in workflow["steps"]:
            if step["status"] == WorkflowStatus.PENDING.value and not step.get("assigned_to"):
                # Find best user for this step
                best_user = await self._find_best_user_for_step(step)
                
                if best_user:
                    step["assigned_to"] = best_user
                    step["assigned_at"] = datetime.now(timezone.utc).isoformat()
                    step["status"] = WorkflowStatus.IN_PROGRESS.value
                    
                    workflow["assigned_users"].append(best_user)
    
    async def _find_best_user_for_step(self, step: Dict[str, Any]) -> Optional[str]:
        """Find the best available user for a workflow step"""
        
        required_skills = step.get("required_skills", [])
        
        # Score users based on:
        # - Skill match
        # - Current workload
        # - Historical performance
        # - Availability
        
        candidate_scores = {}
        
        for user_id, skills in self.user_skills.items():
            score = 0
            
            # Skill matching (40% of score)
            skill_match = len(set(required_skills) & set(skills)) / len(required_skills) if required_skills else 1
            score += skill_match * 40
            
            # Availability (30% of score)
            availability = self.user_availability.get(user_id, 0.5)
            score += availability * 30
            
            # Performance history (30% of score)
            performance = await self._get_user_performance(user_id, step["step_type"])
            score += performance * 30
            
            candidate_scores[user_id] = score
        
        # Return user with highest score
        if candidate_scores:
            return max(candidate_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _load_workflow_templates(self) -> Dict[str, Any]:
        """Load predefined workflow templates"""
        
        return {
            "period_close": {
                "name": "Period Close Process",
                "description": "Complete month-end closing workflow",
                "steps": [
                    {
                        "step_type": "validate_trial_balance",
                        "name": "Validate Trial Balance",
                        "required_skills": ["accounting", "analytics"],
                        "estimated_duration": "PT30M",  # 30 minutes
                        "dependencies": []
                    },
                    {
                        "step_type": "review_journal_entries",
                        "name": "Review Journal Entries",
                        "required_skills": ["accounting", "review"],
                        "estimated_duration": "PT2H",  # 2 hours
                        "dependencies": ["validate_trial_balance"]
                    },
                    {
                        "step_type": "generate_reports",
                        "name": "Generate Financial Reports",
                        "required_skills": ["reporting", "analysis"],
                        "estimated_duration": "PT1H",
                        "dependencies": ["review_journal_entries"]
                    },
                    {
                        "step_type": "management_review",
                        "name": "Management Review",
                        "required_skills": ["management", "approval"],
                        "estimated_duration": "PT45M",
                        "dependencies": ["generate_reports"]
                    },
                    {
                        "step_type": "close_period",
                        "name": "Close Period",
                        "required_skills": ["accounting", "system_admin"],
                        "estimated_duration": "PT15M",
                        "dependencies": ["management_review"]
                    }
                ]
            },
            "journal_approval": {
                "name": "Journal Entry Approval",
                "description": "Multi-level journal entry approval process",
                "steps": [
                    {
                        "step_type": "initial_review",
                        "name": "Initial Review",
                        "required_skills": ["accounting"],
                        "estimated_duration": "PT15M",
                        "dependencies": []
                    },
                    {
                        "step_type": "supervisor_approval",
                        "name": "Supervisor Approval",
                        "required_skills": ["supervisor", "approval"],
                        "estimated_duration": "PT10M",
                        "dependencies": ["initial_review"]
                    },
                    {
                        "step_type": "manager_approval",
                        "name": "Manager Approval",
                        "required_skills": ["manager", "approval"],
                        "estimated_duration": "PT15M",
                        "dependencies": ["supervisor_approval"],
                        "condition": "amount > 10000"
                    }
                ]
            }
        }


class ConflictResolver:
    """Intelligent conflict detection and resolution"""
    
    async def detect_conflicts(self, new_change: Dict[str, Any],
                             pending_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between new change and pending changes"""
        
        conflicts = []
        
        for pending_change in pending_changes:
            conflict = await self._check_field_conflicts(new_change, pending_change)
            if conflict:
                conflicts.append(conflict)
        
        return conflicts
    
    async def suggest_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest intelligent conflict resolution strategies"""
        
        if not conflicts:
            return {"strategy": "no_conflict"}
        
        # Analyze conflict types and suggest best resolution
        conflict_types = [c.get("type") for c in conflicts]
        
        if "field_overwrite" in conflict_types:
            return {
                "strategy": "merge_with_user_choice",
                "description": "Some fields have conflicting changes. User input required.",
                "auto_resolvable": False,
                "resolution_ui": "field_by_field_comparison"
            }
        
        if "amount_discrepancy" in conflict_types:
            return {
                "strategy": "calculate_difference",
                "description": "Amount conflicts detected. Suggest creating adjustment entry.",
                "auto_resolvable": True,
                "suggested_action": "create_adjustment_entry"
            }
        
        return {
            "strategy": "manual_review",
            "description": "Complex conflicts require manual review",
            "auto_resolvable": False
        }
    
    async def _check_field_conflicts(self, change1: Dict[str, Any],
                                   change2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for conflicts between two changes"""
        
        overlapping_fields = set(change1.get("fields", {}).keys()) & set(change2.get("fields", {}).keys())
        
        if overlapping_fields:
            conflicting_fields = []
            
            for field in overlapping_fields:
                if change1["fields"][field] != change2["fields"][field]:
                    conflicting_fields.append({
                        "field": field,
                        "value1": change1["fields"][field],
                        "value2": change2["fields"][field],
                        "user1": change1.get("user_id"),
                        "user2": change2.get("user_id")
                    })
            
            if conflicting_fields:
                return {
                    "type": "field_overwrite",
                    "conflicting_fields": conflicting_fields,
                    "severity": "medium"
                }
        
        return None


class PresenceManager:
    """Manages real-time user presence and activity"""
    
    def __init__(self):
        self.presence_cache = {}
        self.activity_history = {}
    
    async def update_presence(self, user_id: str, presence: UserPresence):
        """Update user presence information"""
        
        self.presence_cache[user_id] = presence
        
        # Store activity history for analytics
        if user_id not in self.activity_history:
            self.activity_history[user_id] = []
        
        self.activity_history[user_id].append({
            "timestamp": presence.last_seen.isoformat(),
            "activity": presence.activity.value,
            "location": presence.current_location
        })
        
        # Keep only last 100 activities
        if len(self.activity_history[user_id]) > 100:
            self.activity_history[user_id] = self.activity_history[user_id][-100:]
    
    async def get_workspace_presence(self, workspace_id: str) -> List[UserPresence]:
        """Get all users present in a workspace"""
        
        workspace_users = []
        
        for user_id, presence in self.presence_cache.items():
            if workspace_id in presence.current_location:
                workspace_users.append(presence)
        
        return workspace_users
    
    async def cleanup_inactive_users(self):
        """Remove users who have been inactive for too long"""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        
        inactive_users = [
            user_id for user_id, presence in self.presence_cache.items()
            if presence.last_seen < cutoff_time
        ]
        
        for user_id in inactive_users:
            del self.presence_cache[user_id]


# Export collaborative workspace classes
__all__ = [
    'CollaborativeWorkspace',
    'UserPresence',
    'CollaborationContext',
    'SmartWorkflowStep',
    'SmartWorkflowEngine',
    'ConflictResolver',
    'PresenceManager',
    'UserActivity',
    'WorkflowStatus'
]