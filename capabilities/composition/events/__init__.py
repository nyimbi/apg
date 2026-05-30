"""APG event streaming capability package."""

from __future__ import annotations

from .capability_contract import (
	EVENT_BUS_STREAM,
	SUPPORTED_EVENT_AGENT_ROLES,
	SUPPORTED_EVENT_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import (
	BytewaxDataflowRuntime,
	CompositionEventsService,
	EventConsumptionService,
	EventPublishingService,
	EventSourcingService,
	EventStreamingService,
	SchemaRegistryService,
	StreamProcessingService,
)


__version__ = "2.1.0"
__capability_id__ = "composition_events"
__apg_dependencies__ = ["auth", "audl", "ntfy", "registry", "composition_access"]
__apg_optional_dependencies__ = ["i18n", "mchn"]


def register_capability() -> dict[str, object]:
	"""Return package metadata used by APG capability discovery."""
	contract = get_capability_contract()
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"version": __version__,
		"provides": contract["provides"],
		"requires": contract["requires"],
		"ui": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


__all__ = [
	"BytewaxDataflowRuntime",
	"CompositionEventsService",
	"EVENT_BUS_STREAM",
	"EventConsumptionService",
	"EventPublishingService",
	"EventSourcingService",
	"EventStreamingService",
	"SUPPORTED_EVENT_AGENT_ROLES",
	"SUPPORTED_EVENT_AGENT_RUNTIMES",
	"SchemaRegistryService",
	"StreamProcessingService",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_capability_contract",
	"register_capability",
	"streaming_manifest",
]
