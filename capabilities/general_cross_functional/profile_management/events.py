"""
Profile Management Events

Event system for profile management capability to enable composition
with other APG capabilities through event-driven integration.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProfileEventType(str, Enum):
    """Profile management event types"""
    
    # User lifecycle events
    USER_REGISTRATION_STARTED = "user.registration_started"
    USER_REGISTERED = "user.registered"
    USER_EMAIL_VERIFIED = "user.email_verified"
    USER_EMAIL_VERIFICATION_SENT = "user.email_verification_sent"
    USER_DELETED = "user.deleted"
    USER_GDPR_DELETION_COMPLETED = "user.gdpr_deletion_completed"
    USER_DATA_EXPORTED = "user.data_exported"
    
    # Profile events
    PROFILE_CREATED = "profile.created"
    PROFILE_UPDATED = "profile.updated"
    PROFILE_DELETED = "profile.deleted"
    PROFILE_COMPLETION_CHANGED = "profile.completion_changed"
    PROFILE_VERIFIED = "profile.verified"
    
    # Consent events
    CONSENT_GRANTED = "consent.granted"
    CONSENT_WITHDRAWN = "consent.withdrawn"
    CONSENT_EXPIRED = "consent.expired"
    
    # Preference events
    PREFERENCES_UPDATED = "preferences.updated"
    NOTIFICATION_PREFERENCES_CHANGED = "notification_preferences.changed"
    PRIVACY_SETTINGS_CHANGED = "privacy_settings.changed"


@dataclass
class ProfileEvent:
    """Profile management event data structure"""
    
    event_id: str
    event_type: ProfileEventType
    timestamp: datetime
    user_id: str
    tenant_id: str
    data: Dict[str, Any]
    source: str = "profile_management"
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'data': self.data,
            'source': self.source,
            'version': self.version
        }


class ProfileEventEmitter:
    """
    Event emitter for profile management events.
    
    Supports event-driven integration with other APG capabilities
    through a publish-subscribe pattern.
    """
    
    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def emit(self, event_type: str, data: Dict[str, Any], 
            user_id: str = None, tenant_id: str = None) -> str:
        """
        Emit a profile management event.
        
        Args:
            event_type: Type of event being emitted
            data: Event data payload
            user_id: User associated with event (extracted from data if not provided)
            tenant_id: Tenant associated with event (extracted from data if not provided)
            
        Returns:
            event_id: Unique identifier for the emitted event
        """
        try:
            # Extract user_id and tenant_id from data if not provided
            if not user_id:
                user_id = data.get('user_id', 'unknown')
            if not tenant_id:
                tenant_id = data.get('tenant_id', 'unknown')
            
            # Create event
            event = ProfileEvent(
                event_id=str(uuid.uuid4()),
                event_type=ProfileEventType(event_type),
                timestamp=datetime.utcnow(),
                user_id=user_id,
                tenant_id=tenant_id,
                data=data
            )
            
            # Call registered handlers for this event type
            handlers = self.event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler failed for {event_type}: {str(e)}")
            
            # Call global handlers
            for handler in self.global_handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Global event handler failed for {event_type}: {str(e)}")
            
            self.logger.debug(f"Emitted event {event_type} with ID {event.event_id}")
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Failed to emit event {event_type}: {str(e)}")
            return ""
    
    def on(self, event_type: str, handler: Callable[[ProfileEvent], None]) -> None:
        """
        Register an event handler for specific event type.
        
        Args:
            event_type: Event type to listen for
            handler: Function to call when event is emitted
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for event type {event_type}")
    
    def on_all(self, handler: Callable[[ProfileEvent], None]) -> None:
        """
        Register a global event handler for all events.
        
        Args:
            handler: Function to call for any emitted event
        """
        self.global_handlers.append(handler)
        self.logger.debug("Registered global event handler")
    
    def off(self, event_type: str, handler: Callable[[ProfileEvent], None]) -> bool:
        """
        Unregister an event handler.
        
        Args:
            event_type: Event type to stop listening for
            handler: Handler function to remove
            
        Returns:
            True if handler was found and removed
        """
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                self.logger.debug(f"Unregistered handler for event type {event_type}")
                return True
            except ValueError:
                pass
        
        return False
    
    def get_handlers(self, event_type: str = None) -> Dict[str, List[Callable]]:
        """
        Get registered event handlers.
        
        Args:
            event_type: Specific event type, or None for all handlers
            
        Returns:
            Dictionary of event handlers
        """
        if event_type:
            return {event_type: self.event_handlers.get(event_type, [])}
        
        return self.event_handlers.copy()
    
    def clear_handlers(self, event_type: str = None) -> None:
        """
        Clear event handlers.
        
        Args:
            event_type: Specific event type to clear, or None for all handlers
        """
        if event_type:
            if event_type in self.event_handlers:
                del self.event_handlers[event_type]
                self.logger.debug(f"Cleared handlers for event type {event_type}")
        else:
            self.event_handlers.clear()
            self.global_handlers.clear()
            self.logger.debug("Cleared all event handlers")


# Pre-defined event data structures for common events
class ProfileEvents:
    """
    Standardized event data structures for profile management events.
    Used for type safety and documentation of event payloads.
    """
    
    @staticmethod
    def user_registered(user_id: str, email: str, tenant_id: str, 
                       registration_source: str, profile_completion: int) -> Dict[str, Any]:
        """User registration completed event data"""
        return {
            'user_id': user_id,
            'email': email,
            'tenant_id': tenant_id,
            'registration_source': registration_source,
            'profile_completion': profile_completion,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def profile_updated(user_id: str, updated_by: str, changes: Dict[str, Any],
                       completion_score: int, tenant_id: str) -> Dict[str, Any]:
        """Profile updated event data"""
        return {
            'user_id': user_id,
            'updated_by': updated_by,
            'changes': changes,
            'completion_score': completion_score,
            'tenant_id': tenant_id,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def consent_granted(consent_id: str, user_id: str, purpose: str, 
                       version: str, tenant_id: str) -> Dict[str, Any]:
        """Consent granted event data"""
        return {
            'consent_id': consent_id,
            'user_id': user_id,
            'purpose': purpose,
            'version': version,
            'tenant_id': tenant_id,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def consent_withdrawn(consent_id: str, user_id: str, purpose: str,
                         withdrawal_method: str, withdrawal_reason: str = None) -> Dict[str, Any]:
        """Consent withdrawn event data"""
        return {
            'consent_id': consent_id,
            'user_id': user_id,
            'purpose': purpose,
            'withdrawal_method': withdrawal_method,
            'withdrawal_reason': withdrawal_reason,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def user_deleted(user_id: str, deletion_type: str, deleted_by: str) -> Dict[str, Any]:
        """User deleted event data"""
        return {
            'user_id': user_id,
            'deletion_type': deletion_type,
            'deleted_by': deleted_by,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def gdpr_deletion_completed(user_id: str, deletion_date: str, 
                               requested_by: str) -> Dict[str, Any]:
        """GDPR deletion completed event data"""
        return {
            'user_id': user_id,
            'deletion_date': deletion_date,
            'requested_by': requested_by,
            'anonymization_completed': True,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def email_verification_sent(user_id: str, email: str, 
                               verification_token: str) -> Dict[str, Any]:
        """Email verification sent event data"""
        return {
            'user_id': user_id,
            'email': email,
            'verification_token': verification_token,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def data_exported(user_id: str, exported_by: str, 
                     export_timestamp: str) -> Dict[str, Any]:
        """User data exported event data"""
        return {
            'user_id': user_id,
            'exported_by': exported_by,
            'export_timestamp': export_timestamp,
            'export_type': 'gdpr_request',
            'timestamp': datetime.utcnow().isoformat()
        }


# Capability integration event handlers
def register_notification_integration(event_emitter: ProfileEventEmitter):
    """
    Register event handlers for notification engine integration.
    This demonstrates how other capabilities can subscribe to profile events.
    """
    
    def send_welcome_email(event: ProfileEvent):
        """Send welcome email when user registers"""
        if event.event_type == ProfileEventType.USER_REGISTERED:
            # Integration point with notification engine
            logger.info(f"Would send welcome email to user {event.user_id}")
    
    def send_verification_email(event: ProfileEvent):
        """Send email verification when token is generated"""
        if event.event_type == ProfileEventType.USER_EMAIL_VERIFICATION_SENT:
            # Integration point with notification engine
            logger.info(f"Would send verification email to user {event.user_id}")
    
    def notify_profile_changes(event: ProfileEvent):
        """Notify relevant parties of profile changes"""
        if event.event_type == ProfileEventType.PROFILE_UPDATED:
            changes = event.data.get('changes', {})
            if 'contact_visibility' in changes or 'profile_visibility' in changes:
                # Integration point with notification engine
                logger.info(f"Would notify of privacy setting changes for user {event.user_id}")
    
    # Register handlers
    event_emitter.on(ProfileEventType.USER_REGISTERED.value, send_welcome_email)
    event_emitter.on(ProfileEventType.USER_EMAIL_VERIFICATION_SENT.value, send_verification_email)
    event_emitter.on(ProfileEventType.PROFILE_UPDATED.value, notify_profile_changes)


def register_audit_integration(event_emitter: ProfileEventEmitter):
    """
    Register event handlers for audit logging integration.
    This demonstrates how audit capability can subscribe to profile events.
    """
    
    def audit_profile_changes(event: ProfileEvent):
        """Audit all profile-related events"""
        audit_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'user_id': event.user_id,
            'tenant_id': event.tenant_id,
            'timestamp': event.timestamp.isoformat(),
            'data': event.data
        }
        
        # Integration point with audit logging capability
        logger.info(f"Would audit event {event.event_type.value} for user {event.user_id}")
    
    # Register global handler for all events
    event_emitter.on_all(audit_profile_changes)


def register_analytics_integration(event_emitter: ProfileEventEmitter):
    """
    Register event handlers for analytics integration.
    This demonstrates how analytics capability can subscribe to profile events.
    """
    
    def track_registration_funnel(event: ProfileEvent):
        """Track registration funnel metrics"""
        if event.event_type in [
            ProfileEventType.USER_REGISTRATION_STARTED,
            ProfileEventType.USER_REGISTERED,
            ProfileEventType.USER_EMAIL_VERIFIED
        ]:
            # Integration point with analytics capability
            logger.info(f"Would track registration metric: {event.event_type.value}")
    
    def track_profile_engagement(event: ProfileEvent):
        """Track profile engagement metrics"""
        if event.event_type == ProfileEventType.PROFILE_UPDATED:
            completion_score = event.data.get('completion_score', 0)
            # Integration point with analytics capability
            logger.info(f"Would track profile completion: {completion_score}% for user {event.user_id}")
    
    # Register specific handlers
    event_emitter.on(ProfileEventType.USER_REGISTRATION_STARTED.value, track_registration_funnel)
    event_emitter.on(ProfileEventType.USER_REGISTERED.value, track_registration_funnel)
    event_emitter.on(ProfileEventType.USER_EMAIL_VERIFIED.value, track_registration_funnel)
    event_emitter.on(ProfileEventType.PROFILE_UPDATED.value, track_profile_engagement)


# Global event emitter instance for capability composition
_global_event_emitter = ProfileEventEmitter()

def get_profile_event_emitter() -> ProfileEventEmitter:
    """Get global profile event emitter instance - capability composition keyword"""
    return _global_event_emitter

def emit_profile_event(event_type: str, data: Dict[str, Any]) -> str:
    """Emit profile event using global emitter - capability composition keyword"""
    return _global_event_emitter.emit(event_type, data)