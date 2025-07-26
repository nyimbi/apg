"""
APG Geographical Location Services - Data Models

Comprehensive data models for geofencing, location tracking, and spatial analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json

from pydantic import BaseModel, Field, validator, ConfigDict
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str

from general_cross_functional.workflow_business_process_mgmt.models import APGTenantContext, APGBaseModel

Base = declarative_base()

# =============================================================================
# Enums and Configuration
# =============================================================================

class GCGeofenceType(str, Enum):
	"""Types of geofences supported."""
	POLYGON = "polygon"
	CIRCLE = "circle"
	ADMINISTRATIVE = "administrative"
	ROUTE = "route"
	BUILDING = "building"
	CAMPUS = "campus"

class GCLocationEventType(str, Enum):
	"""Types of location events."""
	ENTER = "enter"
	EXIT = "exit"
	DWELL = "dwell"
	MOVE = "move"
	SPEED_VIOLATION = "speed_violation"
	ROUTE_DEVIATION = "route_deviation"

class GCEntityType(str, Enum):
	"""Types of entities that can be tracked."""
	USER = "user"
	ASSET = "asset"
	VEHICLE = "vehicle"
	DEVICE = "device"
	SHIPMENT = "shipment"
	EMPLOYEE = "employee"

class GCRuleType(str, Enum):
	"""Types of geofence rules."""
	NOTIFICATION = "notification"
	ACCESS_CONTROL = "access_control"
	WORKFLOW_TRIGGER = "workflow_trigger"
	COMPLIANCE_CHECK = "compliance_check"
	ANALYTICS_TRACK = "analytics_track"

class GCComplianceType(str, Enum):
	"""Types of compliance requirements."""
	GDPR = "gdpr"
	DATA_RESIDENCY = "data_residency"
	TAX_JURISDICTION = "tax_jurisdiction"
	LABOR_LAW = "labor_law"
	ENVIRONMENTAL = "environmental"
	SAFETY = "safety"

# =============================================================================
# Core Data Models
# =============================================================================

class GCCoordinate(BaseModel):
	"""Geographical coordinate with optional metadata."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
	longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
	altitude: Optional[float] = Field(None, description="Altitude in meters")
	accuracy_meters: Optional[float] = Field(None, description="Location accuracy in meters")
	timestamp: Optional[datetime] = Field(None, description="Timestamp of coordinate capture")
	
	@validator('latitude')
	def validate_latitude(cls, v):
		if not -90 <= v <= 90:
			raise ValueError('Latitude must be between -90 and 90 degrees')
		return v
	
	@validator('longitude') 
	def validate_longitude(cls, v):
		if not -180 <= v <= 180:
			raise ValueError('Longitude must be between -180 and 180 degrees')
		return v

class GCBoundary(BaseModel):
	"""Geographical boundary definition."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	boundary_type: GCGeofenceType
	coordinates: List[GCCoordinate]
	center_point: Optional[GCCoordinate] = None
	radius_meters: Optional[float] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	@validator('coordinates')
	def validate_coordinates(cls, v, values):
		boundary_type = values.get('boundary_type')
		if boundary_type == GCGeofenceType.POLYGON and len(v) < 3:
			raise ValueError('Polygon must have at least 3 coordinates')
		elif boundary_type == GCGeofenceType.CIRCLE and len(v) != 1:
			raise ValueError('Circle must have exactly 1 center coordinate')
		return v

# =============================================================================
# SQLAlchemy Database Models
# =============================================================================

class GCGeofence(Base, APGBaseModel):
	"""Geofence definition database model."""
	__tablename__ = 'gc_geofences'
	__table_args__ = (
		Index('idx_gc_geofence_tenant', 'tenant_id'),
		Index('idx_gc_geofence_type', 'fence_type'),
		Index('idx_gc_geofence_active', 'is_active'),
		Index('idx_gc_geofence_spatial', 'spatial_index'),
	)
	
	# Primary identification
	fence_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	name = Column(String(255), nullable=False)
	description = Column(Text)
	
	# Geofence definition
	fence_type = Column(String(50), nullable=False)
	boundary_data = Column(JSON, nullable=False)  # Serialized GCBoundary
	spatial_index = Column(String(255))  # For spatial indexing
	
	# Configuration
	rules = Column(JSON, default=list)
	compliance_requirements = Column(JSON, default=list)
	metadata = Column(JSON, default=dict)
	
	# Status and lifecycle
	is_active = Column(Boolean, default=True, nullable=False)
	priority = Column(Integer, default=1)
	created_by = Column(String(36), nullable=False)
	
	# APG Integration
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	# Relationships
	location_events = relationship("GCLocationEvent", back_populates="geofence")
	rules_config = relationship("GCGeofenceRule", back_populates="geofence")

class GCLocationEvent(Base, APGBaseModel):
	"""Location event database model."""
	__tablename__ = 'gc_location_events'
	__table_args__ = (
		Index('idx_gc_event_tenant', 'tenant_id'),
		Index('idx_gc_event_entity', 'entity_id'),
		Index('idx_gc_event_fence', 'fence_id'),
		Index('idx_gc_event_timestamp', 'event_timestamp'),
		Index('idx_gc_event_type', 'event_type'),
	)
	
	# Primary identification
	event_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Entity information
	entity_id = Column(String(36), nullable=False, index=True)
	entity_type = Column(String(50), nullable=False)
	
	# Location event details
	fence_id = Column(String(36), ForeignKey('gc_geofences.fence_id'), nullable=False)
	event_type = Column(String(50), nullable=False)
	event_timestamp = Column(DateTime, nullable=False, index=True)
	
	# Geographical data
	latitude = Column(Float, nullable=False)
	longitude = Column(Float, nullable=False) 
	altitude = Column(Float)
	accuracy_meters = Column(Float)
	
	# Event context
	previous_fence_id = Column(String(36))
	duration_seconds = Column(Integer)  # For dwell events
	speed_kmh = Column(Float)
	direction_degrees = Column(Float)
	
	# Metadata and processing
	metadata = Column(JSON, default=dict)
	processed = Column(Boolean, default=False)
	processing_results = Column(JSON, default=dict)
	
	# APG Integration
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	
	# Relationships
	geofence = relationship("GCGeofence", back_populates="location_events")

class GCGeofenceRule(Base, APGBaseModel):
	"""Geofence rules and actions configuration."""
	__tablename__ = 'gc_geofence_rules'
	__table_args__ = (
		Index('idx_gc_rule_fence', 'fence_id'),
		Index('idx_gc_rule_type', 'rule_type'),
		Index('idx_gc_rule_active', 'is_active'),
	)
	
	# Primary identification
	rule_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	fence_id = Column(String(36), ForeignKey('gc_geofences.fence_id'), nullable=False)
	
	# Rule definition
	rule_type = Column(String(50), nullable=False)
	rule_name = Column(String(255), nullable=False)
	description = Column(Text)
	
	# Trigger conditions
	trigger_events = Column(JSON, nullable=False)  # List of event types
	entity_filters = Column(JSON, default=dict)  # Entity type/ID filters
	time_constraints = Column(JSON, default=dict)  # Time-based rules
	
	# Actions
	action_config = Column(JSON, nullable=False)
	notification_config = Column(JSON, default=dict)
	workflow_triggers = Column(JSON, default=list)
	
	# Status and priority
	is_active = Column(Boolean, default=True, nullable=False)
	priority = Column(Integer, default=1)
	
	# APG Integration
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36), nullable=False)
	
	# Relationships
	geofence = relationship("GCGeofence", back_populates="rules_config")

class GCEntityLocation(Base, APGBaseModel):
	"""Current location tracking for entities."""
	__tablename__ = 'gc_entity_locations'
	__table_args__ = (
		Index('idx_gc_entity_location_tenant', 'tenant_id'),
		Index('idx_gc_entity_location_entity', 'entity_id'),
		Index('idx_gc_entity_location_timestamp', 'last_updated'),
	)
	
	# Primary identification
	location_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	entity_id = Column(String(36), nullable=False, index=True)
	entity_type = Column(String(50), nullable=False)
	
	# Current location
	latitude = Column(Float, nullable=False)
	longitude = Column(Float, nullable=False)
	altitude = Column(Float)
	accuracy_meters = Column(Float)
	
	# Current status
	current_fences = Column(JSON, default=list)  # List of fence IDs entity is in
	speed_kmh = Column(Float)
	direction_degrees = Column(Float)
	last_updated = Column(DateTime, nullable=False, index=True)
	
	# Tracking metadata
	tracking_enabled = Column(Boolean, default=True)
	tracking_frequency_seconds = Column(Integer, default=60)
	metadata = Column(JSON, default=dict)

class GCTerritory(Base, APGBaseModel):
	"""Territory management for sales, service, and operations."""
	__tablename__ = 'gc_territories'
	__table_args__ = (
		Index('idx_gc_territory_tenant', 'tenant_id'),
		Index('idx_gc_territory_type', 'territory_type'),
	)
	
	# Primary identification
	territory_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	name = Column(String(255), nullable=False)
	description = Column(Text)
	
	# Territory definition
	territory_type = Column(String(50), nullable=False)  # sales, service, delivery, etc.
	boundary_data = Column(JSON, nullable=False)  # Serialized boundary
	parent_territory_id = Column(String(36), ForeignKey('gc_territories.territory_id'))
	
	# Assignment and management
	assigned_users = Column(JSON, default=list)  # List of user IDs
	assigned_assets = Column(JSON, default=list)  # List of asset IDs
	territory_rules = Column(JSON, default=dict)
	
	# Performance and analytics
	performance_metrics = Column(JSON, default=dict)
	metadata = Column(JSON, default=dict)
	
	# Status
	is_active = Column(Boolean, default=True, nullable=False)
	
	# APG Integration
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36), nullable=False)
	
	# Self-referential relationship for hierarchy
	children = relationship("GCTerritory", backref="parent", remote_side=[territory_id])

# =============================================================================
# Pydantic Models for API and Service Layer
# =============================================================================

class GCGeofenceCreate(BaseModel):
	"""Geofence creation model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = None
	fence_type: GCGeofenceType
	boundary: GCBoundary
	rules: List[Dict[str, Any]] = Field(default_factory=list)
	compliance_requirements: List[GCComplianceType] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	is_active: bool = True
	priority: int = Field(1, ge=1, le=10)

class GCGeofenceResponse(BaseModel):
	"""Geofence response model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	fence_id: str
	tenant_id: str
	name: str
	description: Optional[str]
	fence_type: GCGeofenceType
	boundary: GCBoundary
	rules: List[Dict[str, Any]]
	compliance_requirements: List[GCComplianceType]
	metadata: Dict[str, Any]
	is_active: bool
	priority: int
	created_at: datetime
	updated_at: datetime
	created_by: str

class GCLocationEventCreate(BaseModel):
	"""Location event creation model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	entity_id: str
	entity_type: GCEntityType
	coordinate: GCCoordinate
	metadata: Dict[str, Any] = Field(default_factory=dict)

class GCLocationEventResponse(BaseModel):
	"""Location event response model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	event_id: str
	tenant_id: str
	entity_id: str
	entity_type: GCEntityType
	fence_id: Optional[str]
	event_type: GCLocationEventType
	coordinate: GCCoordinate
	event_timestamp: datetime
	duration_seconds: Optional[int]
	metadata: Dict[str, Any]
	processed: bool
	processing_results: Dict[str, Any]

class GCTerritoryCreate(BaseModel):
	"""Territory creation model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = None
	territory_type: str
	boundary: GCBoundary
	parent_territory_id: Optional[str] = None
	assigned_users: List[str] = Field(default_factory=list)
	assigned_assets: List[str] = Field(default_factory=list)
	territory_rules: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# Query and Analysis Models
# =============================================================================

class GCLocationQuery(BaseModel):
	"""Location query parameters."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	entity_ids: Optional[List[str]] = None
	entity_types: Optional[List[GCEntityType]] = None
	fence_ids: Optional[List[str]] = None
	event_types: Optional[List[GCLocationEventType]] = None
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	bounding_box: Optional[Tuple[GCCoordinate, GCCoordinate]] = None  # SW, NE corners
	limit: int = Field(100, ge=1, le=1000)
	offset: int = Field(0, ge=0)

class GCAnalyticsRequest(BaseModel):
	"""Analytics request parameters."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	analysis_type: str  # dwell_time, route_efficiency, territory_coverage, etc.
	entities: Optional[List[str]] = None
	territories: Optional[List[str]] = None
	time_range: Tuple[datetime, datetime]
	grouping: Optional[str] = None  # day, week, month, territory, entity_type
	metrics: List[str] = Field(default_factory=list)
	filters: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# Service Response Models
# =============================================================================

class GCServiceResponse(BaseModel):
	"""Standard service response model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	success: bool
	message: str
	data: Optional[Dict[str, Any]] = None
	errors: Optional[List[str]] = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)

class GCGeofenceValidationResult(BaseModel):
	"""Geofence validation result."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	is_valid: bool
	errors: List[str] = Field(default_factory=list)
	warnings: List[str] = Field(default_factory=list)
	suggestions: List[str] = Field(default_factory=list)

# =============================================================================
# Model Exports
# =============================================================================

__all__ = [
	# Enums
	"GCGeofenceType",
	"GCLocationEventType", 
	"GCEntityType",
	"GCRuleType",
	"GCComplianceType",
	
	# Data Models
	"GCCoordinate",
	"GCBoundary",
	
	# Database Models
	"GCGeofence",
	"GCLocationEvent",
	"GCGeofenceRule",
	"GCEntityLocation",
	"GCTerritory",
	
	# API Models
	"GCGeofenceCreate",
	"GCGeofenceResponse",
	"GCLocationEventCreate",
	"GCLocationEventResponse",
	"GCTerritoryCreate",
	
	# Query Models
	"GCLocationQuery",
	"GCAnalyticsRequest",
	
	# Response Models
	"GCServiceResponse",
	"GCGeofenceValidationResult"
]