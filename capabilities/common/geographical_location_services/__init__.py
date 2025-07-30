"""
APG Geographical Location Services Capability

Comprehensive geofencing, location intelligence, and spatial analytics
capability for location-aware enterprise applications.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field

# Geographical Services Metadata
__version__ = "1.0.0"
__capability_id__ = "geographical_location_services"
__description__ = "Comprehensive geofencing and location intelligence platform"

class GeofenceType(str, Enum):
	"""Types of geofences supported."""
	POLYGON = "polygon"
	CIRCLE = "circle"
	ADMINISTRATIVE = "administrative"
	CUSTOM = "custom"

class LocationEventType(str, Enum):
	"""Types of location events."""
	ENTER = "enter"
	EXIT = "exit"
	DWELL = "dwell"
	MOVE = "move"

@dataclass
class Coordinate:
	"""Geographical coordinate."""
	latitude: float
	longitude: float
	altitude: Optional[float] = None

class GCGeofence(BaseModel):
	"""Geofence definition model."""
	fence_id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: Optional[str] = None
	fence_type: GeofenceType
	coordinates: List[Coordinate]
	radius_meters: Optional[float] = None  # For circular fences
	rules: List[Dict[str, Any]] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	is_active: bool = True
	created_at: str
	updated_at: str

class GCLocationEvent(BaseModel):
	"""Location event model."""
	event_id: str = Field(default_factory=uuid7str)
	tenant_id: str
	entity_id: str  # User, asset, device ID
	entity_type: str  # user, asset, device, vehicle
	fence_id: str
	event_type: LocationEventType
	coordinates: Coordinate
	timestamp: str
	metadata: Dict[str, Any] = Field(default_factory=dict)

# APG Composition Engine Registration
CAPABILITY_METADATA = {
	"capability_id": "general_cross_functional.geographical_location_services",
	"version": __version__,
	"category": "cross_functional",
	"provides_services": [
		"geofencing_engine",
		"location_intelligence",
		"spatial_analytics",
		"location_compliance",
		"territory_management"
	],
	"dependencies": [
		"auth_rbac",
		"audit_compliance",
		"notification_engine",
		"general_cross_functional.workflow_business_process_mgmt"
	],
	"integrates_with": [
		"general_cross_functional.customer_relationship_management",
		"general_cross_functional.enterprise_asset_management",
		"core_business_operations.human_capital_management",
		"emerging_technologies.edge_computing_iot"
	],
	"data_models": ["GCGeofence", "GCLocationEvent", "GCLocationRule", "GCTerritory"]
}

def get_capability_info() -> Dict[str, Any]:
	"""Get geographical location services capability information."""
	return CAPABILITY_METADATA

__all__ = [
	"GeofenceType",
	"LocationEventType", 
	"Coordinate",
	"GCGeofence",
	"GCLocationEvent",
	"CAPABILITY_METADATA",
	"get_capability_info"
]