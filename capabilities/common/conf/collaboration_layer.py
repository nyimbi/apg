"""
APG Configuration Management Real-Time Collaboration Layer

Implements real-time collaborative configuration management enabling multiple
users to work on configurations simultaneously with conflict resolution,
change tracking, and collaborative workflows.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from uuid_extensions import uuid7str
import asyncio
import logging
from dataclasses import dataclass, field
import json
from concurrent.futures import ThreadPoolExecutor

try:
    from .models import (
        CMResource, ConfigurationDSL, ValidationResult, 
        ResourceType, CloudProvider, ResourceState
    )
    from ..secu.models import SecurityContext
except ImportError:
    from models import (
        CMResource, ConfigurationDSL, ValidationResult,
        ResourceType, CloudProvider, ResourceState
    )
    # Mock SecurityContext for testing
    class SecurityContext:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

logger = logging.getLogger(__name__)


class CollaborationEventType(StrEnum):
    """Collaboration event types"""
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    CONFIGURATION_LOCK = "configuration_lock"
    CONFIGURATION_UNLOCK = "configuration_unlock"
    REAL_TIME_EDIT = "real_time_edit"
    CONFIGURATION_SAVE = "configuration_save"
    COMMENT_ADD = "comment_add"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_GRANTED = "approval_granted"
    CONFLICT_DETECTED = "conflict_detected"
    MERGE_COMPLETED = "merge_completed"


class CollaborationPermission(StrEnum):
    """Collaboration permissions"""
    VIEW_ONLY = "view_only"
    COMMENT = "comment"
    EDIT = "edit"
    APPROVE = "approve"
    ADMIN = "admin"


class ConflictResolutionStrategy(StrEnum):
    """Conflict resolution strategies"""
    MANUAL = "manual"              # Manual resolution required
    LAST_WRITE_WINS = "last_write_wins"  # Last change wins
    MERGE_AUTOMATIC = "merge_automatic"  # Automatic merge if possible
    REVIEWER_DECIDES = "reviewer_decides" # Designated reviewer decides


@dataclass
class CollaborationUser:
    """User participating in collaboration"""
    id: str = field(default_factory=uuid7str)
    user_id: str = ""
    display_name: str = ""
    email: str = ""
    permissions: List[CollaborationPermission] = field(default_factory=list)
    active: bool = True
    cursor_position: Dict[str, Any] = field(default_factory=dict)
    selection: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    joined_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConfigurationLock:
    """Configuration lock for exclusive editing"""
    id: str = field(default_factory=uuid7str)
    resource_id: str = ""
    user_id: str = ""
    lock_type: str = "exclusive"  # exclusive, shared
    locked_sections: List[str] = field(default_factory=list)  # Empty = full lock
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=30))
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationComment:
    """Comment on configuration"""
    id: str = field(default_factory=uuid7str)
    resource_id: str = ""
    user_id: str = ""
    content: str = ""
    section_path: str = ""  # JSON path to configuration section
    line_number: Optional[int] = None
    resolved: bool = False
    thread_id: Optional[str] = None
    parent_comment_id: Optional[str] = None
    mentions: List[str] = field(default_factory=list)
    attachments: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConfigurationChange:
    """Individual configuration change"""
    id: str = field(default_factory=uuid7str)
    resource_id: str = ""
    user_id: str = ""
    change_type: str = ""  # add, modify, delete
    path: str = ""  # JSON path
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    applied: bool = False
    conflict_with: List[str] = field(default_factory=list)  # Conflicting change IDs


@dataclass
class ConfigurationConflict:
    """Configuration merge conflict"""
    id: str = field(default_factory=uuid7str)
    resource_id: str = ""
    conflicting_changes: List[str] = field(default_factory=list)  # Change IDs
    conflict_type: str = ""  # path_conflict, value_conflict, dependency_conflict
    path: str = ""
    base_value: Any = None
    user_values: Dict[str, Any] = field(default_factory=dict)  # user_id -> value
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MANUAL
    resolved: bool = False
    resolution_value: Any = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CollaborationSession:
    """Collaboration session for a configuration"""
    id: str = field(default_factory=uuid7str)
    resource_id: str = ""
    name: str = ""
    description: str = ""
    owner_id: str = ""
    participants: Dict[str, CollaborationUser] = field(default_factory=dict)
    active_locks: Dict[str, ConfigurationLock] = field(default_factory=dict)
    comments: Dict[str, CollaborationComment] = field(default_factory=dict)
    changes: Dict[str, ConfigurationChange] = field(default_factory=dict)
    conflicts: Dict[str, ConfigurationConflict] = field(default_factory=dict)
    base_configuration: Optional[ConfigurationDSL] = None
    current_configuration: Optional[ConfigurationDSL] = None
    auto_save_enabled: bool = True
    auto_save_interval: int = 30  # seconds
    conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MANUAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class CollaborationEventHandler:
    """Handles collaboration events and notifications"""
    
    def __init__(self):
        self.event_handlers: Dict[CollaborationEventType, List[Callable]] = {}
        self.subscribers: Dict[str, Set[str]] = {}  # resource_id -> user_ids
        self.user_connections: Dict[str, Dict[str, Any]] = {}  # user_id -> connection info
    
    def subscribe(self, resource_id: str, user_id: str, connection_info: Dict[str, Any]):
        """Subscribe user to collaboration events for a resource"""
        if resource_id not in self.subscribers:
            self.subscribers[resource_id] = set()
        
        self.subscribers[resource_id].add(user_id)
        self.user_connections[user_id] = connection_info
        
        logger.info(f"User {user_id} subscribed to collaboration events for resource {resource_id}")
    
    def unsubscribe(self, resource_id: str, user_id: str):
        """Unsubscribe user from collaboration events"""
        if resource_id in self.subscribers:
            self.subscribers[resource_id].discard(user_id)
        
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        
        logger.info(f"User {user_id} unsubscribed from collaboration events for resource {resource_id}")
    
    async def emit_event(
        self,
        event_type: CollaborationEventType,
        resource_id: str,
        data: Dict[str, Any],
        exclude_user: Optional[str] = None
    ):
        """Emit collaboration event to subscribed users"""
        if resource_id not in self.subscribers:
            return
        
        recipients = self.subscribers[resource_id].copy()
        if exclude_user:
            recipients.discard(exclude_user)
        
        event_data = {
            "type": event_type.value,
            "resource_id": resource_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # In a real implementation, this would send to WebSocket connections, message queues, etc.
        for user_id in recipients:
            connection = self.user_connections.get(user_id)
            if connection:
                await self._send_to_connection(user_id, connection, event_data)
        
        logger.info(f"Emitted {event_type.value} event to {len(recipients)} users for resource {resource_id}")
    
    async def _send_to_connection(self, user_id: str, connection: Dict[str, Any], event_data: Dict[str, Any]):
        """Send event data to user connection (mock implementation)"""
        # This would integrate with actual WebSocket/SSE/message queue infrastructure
        logger.debug(f"Sending event to user {user_id}: {event_data['type']}")


class RealTimeCollaborationManager:
    """Manages real-time collaborative configuration editing"""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.event_handler = CollaborationEventHandler()
        self.auto_save_tasks: Dict[str, asyncio.Task] = {}
        self.conflict_resolver = ConfigurationConflictResolver()
        self._initialized = False
    
    async def initialize(self):
        """Initialize collaboration manager"""
        if not self._initialized:
            # Start background tasks for cleanup and auto-save
            asyncio.create_task(self._cleanup_expired_sessions())
            asyncio.create_task(self._cleanup_expired_locks())
            
            self._initialized = True
            logger.info("Real-Time Collaboration Manager initialized")
    
    async def create_collaboration_session(
        self,
        resource_id: str,
        owner_id: str,
        name: str = "",
        base_configuration: Optional[ConfigurationDSL] = None
    ) -> str:
        """Create new collaboration session"""
        assert self._initialized, "Collaboration manager not initialized"
        
        session = CollaborationSession(
            resource_id=resource_id,
            name=name or f"Collaboration session for {resource_id[:8]}",
            owner_id=owner_id,
            base_configuration=base_configuration,
            current_configuration=base_configuration
        )
        
        self.sessions[session.id] = session
        
        # Start auto-save if enabled
        if session.auto_save_enabled:
            await self._start_auto_save(session.id)
        
        logger.info(f"Created collaboration session {session.id} for resource {resource_id}")
        return session.id
    
    async def join_collaboration_session(
        self,
        session_id: str,
        user_id: str,
        display_name: str,
        email: str = "",
        permissions: List[CollaborationPermission] = None
    ) -> bool:
        """Join collaboration session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if permissions is None:
            permissions = [CollaborationPermission.EDIT]
        
        user = CollaborationUser(
            user_id=user_id,
            display_name=display_name,
            email=email,
            permissions=permissions
        )
        
        session.participants[user_id] = user
        session.updated_at = datetime.utcnow()
        
        # Subscribe to events
        self.event_handler.subscribe(session.resource_id, user_id, {"session_id": session_id})
        
        # Notify other participants
        await self.event_handler.emit_event(
            CollaborationEventType.USER_JOIN,
            session.resource_id,
            {
                "user_id": user_id,
                "display_name": display_name,
                "permissions": [p.value for p in permissions]
            },
            exclude_user=user_id
        )
        
        logger.info(f"User {user_id} joined collaboration session {session_id}")
        return True
    
    async def leave_collaboration_session(self, session_id: str, user_id: str):
        """Leave collaboration session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Release any locks held by the user
        await self._release_user_locks(session_id, user_id)
        
        # Remove from participants
        if user_id in session.participants:
            user = session.participants[user_id]
            del session.participants[user_id]
            
            # Unsubscribe from events
            self.event_handler.unsubscribe(session.resource_id, user_id)
            
            # Notify other participants
            await self.event_handler.emit_event(
                CollaborationEventType.USER_LEAVE,
                session.resource_id,
                {
                    "user_id": user_id,
                    "display_name": user.display_name
                },
                exclude_user=user_id
            )
        
        session.updated_at = datetime.utcnow()
        logger.info(f"User {user_id} left collaboration session {session_id}")
    
    async def acquire_configuration_lock(
        self,
        session_id: str,
        user_id: str,
        lock_sections: List[str] = None,
        lock_duration: int = 30
    ) -> Optional[str]:
        """Acquire lock on configuration or sections"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if user has edit permissions
        if user_id not in session.participants:
            return None
        
        user = session.participants[user_id]
        if CollaborationPermission.EDIT not in user.permissions:
            return None
        
        # Check for conflicting locks
        if await self._has_conflicting_locks(session, lock_sections or []):
            return None
        
        # Create lock
        lock = ConfigurationLock(
            resource_id=session.resource_id,
            user_id=user_id,
            locked_sections=lock_sections or [],
            expires_at=datetime.utcnow() + timedelta(minutes=lock_duration)
        )
        
        session.active_locks[lock.id] = lock
        session.updated_at = datetime.utcnow()
        
        # Notify other participants
        await self.event_handler.emit_event(
            CollaborationEventType.CONFIGURATION_LOCK,
            session.resource_id,
            {
                "lock_id": lock.id,
                "user_id": user_id,
                "sections": lock.locked_sections,
                "expires_at": lock.expires_at.isoformat()
            },
            exclude_user=user_id
        )
        
        logger.info(f"User {user_id} acquired lock {lock.id} on session {session_id}")
        return lock.id
    
    async def release_configuration_lock(self, session_id: str, lock_id: str, user_id: str) -> bool:
        """Release configuration lock"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if lock_id not in session.active_locks:
            return False
        
        lock = session.active_locks[lock_id]
        
        # Only lock owner or session owner can release
        if lock.user_id != user_id and session.owner_id != user_id:
            return False
        
        del session.active_locks[lock_id]
        session.updated_at = datetime.utcnow()
        
        # Notify other participants
        await self.event_handler.emit_event(
            CollaborationEventType.CONFIGURATION_UNLOCK,
            session.resource_id,
            {
                "lock_id": lock_id,
                "user_id": lock.user_id,
                "sections": lock.locked_sections
            }
        )
        
        logger.info(f"Released lock {lock_id} on session {session_id}")
        return True
    
    async def apply_configuration_change(
        self,
        session_id: str,
        user_id: str,
        change_type: str,
        path: str,
        old_value: Any,
        new_value: Any
    ) -> Optional[str]:
        """Apply real-time configuration change"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Verify user permissions and locks
        if not await self._can_user_edit_path(session, user_id, path):
            return None
        
        # Create change
        change = ConfigurationChange(
            resource_id=session.resource_id,
            user_id=user_id,
            change_type=change_type,
            path=path,
            old_value=old_value,
            new_value=new_value
        )
        
        # Check for conflicts with pending changes
        conflicts = await self._detect_change_conflicts(session, change)
        
        if conflicts:
            # Create conflict record
            conflict = ConfigurationConflict(
                resource_id=session.resource_id,
                conflicting_changes=[change.id] + conflicts,
                conflict_type="value_conflict",
                path=path,
                base_value=old_value,
                user_values={user_id: new_value}
            )
            
            session.conflicts[conflict.id] = conflict
            
            await self.event_handler.emit_event(
                CollaborationEventType.CONFLICT_DETECTED,
                session.resource_id,
                {
                    "conflict_id": conflict.id,
                    "path": path,
                    "conflicting_users": [change.user_id for change_id in conflicts for change in [session.changes[change_id]] if change]
                }
            )
            
            logger.warning(f"Conflict detected for change {change.id} at path {path}")
        
        session.changes[change.id] = change
        
        # Apply change to current configuration if no conflicts
        if not conflicts:
            await self._apply_change_to_configuration(session, change)
            change.applied = True
        
        session.updated_at = datetime.utcnow()
        
        # Notify other participants
        await self.event_handler.emit_event(
            CollaborationEventType.REAL_TIME_EDIT,
            session.resource_id,
            {
                "change_id": change.id,
                "user_id": user_id,
                "type": change_type,
                "path": path,
                "value": new_value,
                "has_conflicts": len(conflicts) > 0
            },
            exclude_user=user_id
        )
        
        logger.info(f"Applied change {change.id} by user {user_id} on session {session_id}")
        return change.id
    
    async def add_comment(
        self,
        session_id: str,
        user_id: str,
        content: str,
        section_path: str = "",
        line_number: Optional[int] = None,
        mentions: List[str] = None
    ) -> Optional[str]:
        """Add comment to configuration"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if user_id not in session.participants:
            return None
        
        user = session.participants[user_id]
        if CollaborationPermission.COMMENT not in user.permissions and CollaborationPermission.EDIT not in user.permissions:
            return None
        
        comment = CollaborationComment(
            resource_id=session.resource_id,
            user_id=user_id,
            content=content,
            section_path=section_path,
            line_number=line_number,
            mentions=mentions or []
        )
        
        session.comments[comment.id] = comment
        session.updated_at = datetime.utcnow()
        
        # Notify participants and mentioned users
        await self.event_handler.emit_event(
            CollaborationEventType.COMMENT_ADD,
            session.resource_id,
            {
                "comment_id": comment.id,
                "user_id": user_id,
                "content": content,
                "section_path": section_path,
                "mentions": mentions or []
            }
        )
        
        logger.info(f"Added comment {comment.id} by user {user_id} on session {session_id}")
        return comment.id
    
    async def resolve_conflict(
        self,
        session_id: str,
        conflict_id: str,
        resolution_value: Any,
        resolved_by: str
    ) -> bool:
        """Resolve configuration conflict"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if conflict_id not in session.conflicts:
            return False
        
        conflict = session.conflicts[conflict_id]
        
        # Verify resolver has appropriate permissions
        if resolved_by not in session.participants:
            return False
        
        resolver = session.participants[resolved_by]
        if CollaborationPermission.APPROVE not in resolver.permissions and session.owner_id != resolved_by:
            return False
        
        # Apply resolution
        conflict.resolved = True
        conflict.resolution_value = resolution_value
        conflict.resolved_by = resolved_by
        conflict.resolved_at = datetime.utcnow()
        
        # Apply resolved changes
        for change_id in conflict.conflicting_changes:
            if change_id in session.changes:
                change = session.changes[change_id]
                change.new_value = resolution_value
                if not change.applied:
                    await self._apply_change_to_configuration(session, change)
                    change.applied = True
        
        session.updated_at = datetime.utcnow()
        
        # Notify participants
        await self.event_handler.emit_event(
            CollaborationEventType.MERGE_COMPLETED,
            session.resource_id,
            {
                "conflict_id": conflict_id,
                "path": conflict.path,
                "resolution_value": resolution_value,
                "resolved_by": resolved_by
            }
        )
        
        logger.info(f"Resolved conflict {conflict_id} by user {resolved_by} on session {session_id}")
        return True
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current collaboration session state"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "resource_id": session.resource_id,
            "name": session.name,
            "owner_id": session.owner_id,
            "participants": [
                {
                    "user_id": user.user_id,
                    "display_name": user.display_name,
                    "permissions": [p.value for p in user.permissions],
                    "active": user.active,
                    "last_activity": user.last_activity.isoformat()
                }
                for user in session.participants.values()
            ],
            "active_locks": [
                {
                    "lock_id": lock.id,
                    "user_id": lock.user_id,
                    "sections": lock.locked_sections,
                    "expires_at": lock.expires_at.isoformat()
                }
                for lock in session.active_locks.values()
            ],
            "pending_conflicts": [
                {
                    "conflict_id": conflict.id,
                    "path": conflict.path,
                    "type": conflict.conflict_type,
                    "users": list(conflict.user_values.keys())
                }
                for conflict in session.conflicts.values()
                if not conflict.resolved
            ],
            "comment_count": len(session.comments),
            "change_count": len(session.changes),
            "current_configuration": session.current_configuration.model_dump() if session.current_configuration else None,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }
    
    # Helper methods
    async def _has_conflicting_locks(self, session: CollaborationSession, sections: List[str]) -> bool:
        """Check if there are conflicting locks"""
        for lock in session.active_locks.values():
            if lock.expires_at < datetime.utcnow():
                continue
            
            # Check for overlapping sections
            if not sections and not lock.locked_sections:  # Both full locks
                return True
            elif not sections or not lock.locked_sections:  # One full lock
                return True
            elif any(section in lock.locked_sections for section in sections):  # Section overlap
                return True
        
        return False
    
    async def _release_user_locks(self, session_id: str, user_id: str):
        """Release all locks held by a user"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        user_locks = [lock_id for lock_id, lock in session.active_locks.items() if lock.user_id == user_id]
        
        for lock_id in user_locks:
            await self.release_configuration_lock(session_id, lock_id, user_id)
    
    async def _can_user_edit_path(self, session: CollaborationSession, user_id: str, path: str) -> bool:
        """Check if user can edit specific path"""
        if user_id not in session.participants:
            return False
        
        user = session.participants[user_id]
        if CollaborationPermission.EDIT not in user.permissions:
            return False
        
        # Check locks
        for lock in session.active_locks.values():
            if lock.expires_at < datetime.utcnow():
                continue
                
            if lock.user_id != user_id:
                # Check if path is locked by another user
                if not lock.locked_sections or any(path.startswith(section) for section in lock.locked_sections):
                    return False
        
        return True
    
    async def _detect_change_conflicts(self, session: CollaborationSession, change: ConfigurationChange) -> List[str]:
        """Detect conflicts with other pending changes"""
        conflicts = []
        
        for other_change_id, other_change in session.changes.items():
            if (other_change.path == change.path and 
                other_change.user_id != change.user_id and
                other_change.timestamp > change.timestamp - timedelta(seconds=30)):  # Recent changes
                conflicts.append(other_change_id)
        
        return conflicts
    
    async def _apply_change_to_configuration(self, session: CollaborationSession, change: ConfigurationChange):
        """Apply change to current configuration"""
        if not session.current_configuration:
            return
        
        # This would implement the actual configuration modification logic
        # For now, we'll just update the timestamp
        session.updated_at = datetime.utcnow()
        
        logger.debug(f"Applied change {change.id} to configuration")
    
    async def _start_auto_save(self, session_id: str):
        """Start auto-save task for session"""
        if session_id in self.auto_save_tasks:
            return
        
        async def auto_save_task():
            while session_id in self.sessions:
                await asyncio.sleep(self.sessions[session_id].auto_save_interval)
                if session_id in self.sessions:
                    await self._auto_save_session(session_id)
        
        self.auto_save_tasks[session_id] = asyncio.create_task(auto_save_task())
    
    async def _auto_save_session(self, session_id: str):
        """Auto-save session changes"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Save current configuration state
        # This would integrate with the main configuration service
        logger.debug(f"Auto-saved session {session_id}")
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if (session.expires_at and session.expires_at < current_time) or \
                   (len(session.participants) == 0 and (current_time - session.updated_at).total_seconds() > 3600):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self._cleanup_session(session_id)
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired collaboration sessions")
    
    async def _cleanup_expired_locks(self):
        """Background task to cleanup expired locks"""
        while True:
            await asyncio.sleep(60)  # Run every minute
            
            current_time = datetime.utcnow()
            
            for session in self.sessions.values():
                expired_locks = [
                    lock_id for lock_id, lock in session.active_locks.items()
                    if lock.expires_at < current_time
                ]
                
                for lock_id in expired_locks:
                    lock = session.active_locks[lock_id]
                    await self.release_configuration_lock(session.id, lock_id, lock.user_id)
    
    async def _cleanup_session(self, session_id: str):
        """Cleanup collaboration session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Notify remaining participants
        for user_id in list(session.participants.keys()):
            await self.leave_collaboration_session(session_id, user_id)
        
        # Cancel auto-save task
        if session_id in self.auto_save_tasks:
            self.auto_save_tasks[session_id].cancel()
            del self.auto_save_tasks[session_id]
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info(f"Cleaned up collaboration session {session_id}")


class ConfigurationConflictResolver:
    """Handles automatic conflict resolution"""
    
    async def resolve_conflict(
        self,
        conflict: ConfigurationConflict,
        strategy: ConflictResolutionStrategy
    ) -> Tuple[bool, Any, str]:
        """
        Resolve conflict using specified strategy
        
        Returns:
            (success, resolution_value, reason)
        """
        if strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return await self._resolve_last_write_wins(conflict)
        elif strategy == ConflictResolutionStrategy.MERGE_AUTOMATIC:
            return await self._resolve_automatic_merge(conflict)
        else:
            return False, None, "Manual resolution required"
    
    async def _resolve_last_write_wins(self, conflict: ConfigurationConflict) -> Tuple[bool, Any, str]:
        """Resolve using last write wins strategy"""
        if not conflict.user_values:
            return False, None, "No user values to resolve"
        
        # Find the most recent value (this would require change timestamps)
        # For now, just pick the first value
        latest_value = list(conflict.user_values.values())[0]
        return True, latest_value, "Last write wins"
    
    async def _resolve_automatic_merge(self, conflict: ConfigurationConflict) -> Tuple[bool, Any, str]:
        """Attempt automatic merge if possible"""
        # This would implement smart merging logic based on configuration structure
        # For now, just use last write wins
        return await self._resolve_last_write_wins(conflict)


# Global collaboration manager instance
_collaboration_manager = None

async def get_collaboration_manager() -> RealTimeCollaborationManager:
    """Get global collaboration manager instance"""
    global _collaboration_manager
    if _collaboration_manager is None:
        _collaboration_manager = RealTimeCollaborationManager()
        await _collaboration_manager.initialize()
    return _collaboration_manager

# Export main classes
__all__ = [
    "CollaborationEventType",
    "CollaborationPermission", 
    "ConflictResolutionStrategy",
    "CollaborationUser",
    "ConfigurationLock",
    "CollaborationComment",
    "ConfigurationChange",
    "ConfigurationConflict",
    "CollaborationSession",
    "CollaborationEventHandler",
    "RealTimeCollaborationManager",
    "ConfigurationConflictResolver",
    "get_collaboration_manager"
]