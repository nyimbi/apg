"""
APG Event Streaming Bus - Package Initialization

Enterprise-grade event streaming platform providing real-time event-driven
communication, stream processing, and message orchestration for the APG ecosystem.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@gmail.com"
__description__ = "APG Event Streaming Bus - Real-time event-driven communication platform"

# Core service imports
from .service import (
    EventStreamingService,
    EventPublishingService,
    EventConsumptionService,
    StreamProcessingService,
    EventSourcingService,
    SchemaRegistryService
)

# Model imports
from .models import (
    ESEvent,
    ESStream,
    ESSubscription,
    ESConsumerGroup,
    ESSchema,
    ESMetrics,
    ESAuditLog,
    EventStatus,
    StreamStatus,
    SubscriptionStatus,
    EventType,
    DeliveryMode
)

try:
    from .api import api_app, router
except Exception:
    api_app = None
    router = None

try:
    from .views import (
        EventStreamView,
        SubscriptionView,
        ConsumerGroupView,
        SchemaView,
        MetricsView,
        StreamingDashboardView
    )
except Exception:
    EventStreamView = None
    SubscriptionView = None
    ConsumerGroupView = None
    SchemaView = None
    MetricsView = None
    StreamingDashboardView = None

try:
    from .apg_integration import (
        APGEventStreamingIntegration,
        APGCapabilityInfo,
        EventRoutingRule,
        CrossCapabilityWorkflow,
        EventCompositionPattern
    )
except Exception:
    APGEventStreamingIntegration = None
    APGCapabilityInfo = None
    EventRoutingRule = None
    CrossCapabilityWorkflow = None
    EventCompositionPattern = None

__all__ = [
    # Core services
    "EventStreamingService",
    "EventPublishingService", 
    "EventConsumptionService",
    "StreamProcessingService",
    "EventSourcingService",
    "SchemaRegistryService",
    
    # Data models
    "ESEvent",
    "ESStream",
    "ESSubscription", 
    "ESConsumerGroup",
    "ESSchema",
    "ESMetrics",
    "ESAuditLog",
    "EventStatus",
    "StreamStatus",
    "SubscriptionStatus",
    "EventType",
    "DeliveryMode",
    
    # API components
    "api_app",
    "router",
    
    # UI views
    "EventStreamView",
    "SubscriptionView",
    "ConsumerGroupView", 
    "SchemaView",
    "MetricsView",
    "StreamingDashboardView",
    
    # APG integration
    "APGEventStreamingIntegration",
    "APGCapabilityInfo",
    "EventRoutingRule",
    "CrossCapabilityWorkflow",
    "EventCompositionPattern"
]
