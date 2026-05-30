"""APG CKM Notification System capability package."""

from __future__ import annotations

from .capability_contract import (
	SUPPORTED_CHANNELS,
	SUPPORTED_NOTIFICATION_AGENT_ROLES,
	SUPPORTED_NOTIFICATION_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .lifecycle import (
	NotificationAgent,
	NotificationDelivery,
	NotificationLifecycleService,
	NotificationPreference,
	NotificationProvider,
	NotificationTemplate,
)


__version__ = "1.0.0"
__author__ = "Datacraft"

APG_CAPABILITY_INFO = {
	"id": "ckm_not",
	"name": "Notification System",
	"version": __version__,
	"description": "Tenant-scoped notification templates, campaigns, delivery governance, preferences, provider registry, and AI-agent review guardrails.",
	"category": "ckm",
	"provides": get_capability_contract()["provides"],
	"requires": get_capability_contract()["requires"],
	"supported_channels": SUPPORTED_CHANNELS,
	"supported_agent_runtimes": SUPPORTED_NOTIFICATION_AGENT_RUNTIMES,
	"streaming": streaming_manifest(),
}

__all__ = [
	"APG_CAPABILITY_INFO",
	"NotificationAgent",
	"NotificationDelivery",
	"NotificationLifecycleService",
	"NotificationPreference",
	"NotificationProvider",
	"NotificationTemplate",
	"SUPPORTED_CHANNELS",
	"SUPPORTED_NOTIFICATION_AGENT_ROLES",
	"SUPPORTED_NOTIFICATION_AGENT_RUNTIMES",
	"evaluate_capability_rules",
	"get_capability_contract",
	"streaming_manifest",
]
