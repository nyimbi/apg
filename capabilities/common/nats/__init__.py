"""APG NATS JetStream capability — durable event bus.

Provides NATSEventAdapter (AuditAdapter implementation) and NATSConnector
(BaseConnector implementation) that publish DomainEvents to NATS JetStream.

Activated automatically when NATS_URL env var is set. Falls back to
NullAuditAdapter when NATS is not configured.

Subject convention: apg.events.{capability_id}.{event_type}
Stream name: APG_EVENTS  Subjects: apg.events.>

Usage::

    # Auto-wired via get_audit_adapter() factory
    adapter = get_nats_audit_adapter()
    await adapter.log_event("workflow_started", "user1", "tenant1", "wf-123", {...})
"""
from .nats_adapter import NATSEventAdapter, NATSConnector, get_nats_audit_adapter
from .stream_setup import setup_apg_stream, APG_STREAM_NAME, APG_SUBJECT_PREFIX
from .subject_registry import subject_for, parse_subject

__all__ = [
	"NATSEventAdapter",
	"NATSConnector",
	"get_nats_audit_adapter",
	"setup_apg_stream",
	"APG_STREAM_NAME",
	"APG_SUBJECT_PREFIX",
	"subject_for",
	"parse_subject",
]
