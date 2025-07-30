"""
Geographical Location Services - Comprehensive Pydantic v2 Models

Enterprise-grade geospatial data models for the APG platform providing:
- Advanced geocoding and address validation
- Geofencing and real-time location tracking  
- Territory management and spatial analytics
- Route optimization and logistics intelligence
- Compliance and regulatory geographic management

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, Set
from enum import Enum
from decimal import Decimal
from uuid import UUID
import json
import numpy as np

from pydantic import BaseModel, Field, ConfigDict, field_validator, computed_field
from pydantic.functional_validators import AfterValidator
from uuid_extensions import uuid7str

# =============================================================================
# Core Configuration and Base Models
# =============================================================================

class GLSBase(BaseModel):
	"""Base model for all Geographical Location Services models."""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		use_enum_values=True,
		str_strip_whitespace=True
	)
	
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="Multi-tenant organization identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User ID who created the record")
	updated_by: str = Field(..., description="User ID who last updated the record")
	is_active: bool = Field(default=True, description="Active status flag")
	metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

# =============================================================================
# Enums and Type Definitions
# =============================================================================

class GLSCoordinateSystem(str, Enum):
	"""Supported geographic coordinate systems."""
	WGS84 = "WGS84"  # World Geodetic System 1984
	UTM = "UTM"      # Universal Transverse Mercator
	STATE_PLANE = "STATE_PLANE"  # US State Plane
	MERCATOR = "MERCATOR"  # Web Mercator
	LAMBERT = "LAMBERT"    # Lambert Conformal Conic

class GLSLocationAccuracy(str, Enum):
	"""Location accuracy levels."""
	HIGH = "high"        # < 5 meters
	MEDIUM = "medium"    # 5-50 meters  
	LOW = "low"          # 50-500 meters
	APPROXIMATE = "approximate"  # > 500 meters

class GLSGeofenceType(str, Enum):
	"""Types of geofences supported."""
	CIRCLE = "circle"
	POLYGON = "polygon"
	RECTANGLE = "rectangle"
	ADMINISTRATIVE = "administrative"
	ROUTE_CORRIDOR = "route_corridor"
	BUILDING = "building"
	CAMPUS = "campus"
	ZONE = "zone"

class GLSEventType(str, Enum):
	"""Location event types."""
	ENTER = "enter"
	EXIT = "exit"
	DWELL = "dwell"
	MOVE = "move"
	STOP = "stop"
	SPEED_VIOLATION = "speed_violation"
	ROUTE_DEVIATION = "route_deviation"
	PANIC_BUTTON = "panic_button"
	LOW_BATTERY = "low_battery"
	OFFLINE = "offline"

class GLSEntityType(str, Enum):
	"""Types of trackable entities."""
	PERSON = "person"
	EMPLOYEE = "employee"
	VISITOR = "visitor"
	VEHICLE = "vehicle"
	ASSET = "asset"
	EQUIPMENT = "equipment"
	SHIPMENT = "shipment"
	DEVICE = "device"
	SMARTPHONE = "smartphone"
	TABLET = "tablet"

class GLSTerritoryType(str, Enum):
	"""Territory management types."""
	SALES = "sales"
	SERVICE = "service"
	DELIVERY = "delivery"
	MAINTENANCE = "maintenance"
	SECURITY = "security"
	ADMINISTRATIVE = "administrative"
	CUSTOM = "custom"

class GLSRouteOptimization(str, Enum):
	"""Route optimization objectives."""
	SHORTEST_DISTANCE = "shortest_distance"
	FASTEST_TIME = "fastest_time"
	LEAST_TRAFFIC = "least_traffic"
	FUEL_EFFICIENT = "fuel_efficient"
	BALANCED = "balanced"

class GLSComplianceType(str, Enum):
	"""Geographic compliance types."""
	GDPR = "gdpr"
	CCPA = "ccpa"
	DATA_RESIDENCY = "data_residency"
	TAX_JURISDICTION = "tax_jurisdiction"
	LABOR_LAWS = "labor_laws"
	ENVIRONMENTAL = "environmental"
	SAFETY_REGULATIONS = "safety_regulations"
	EXPORT_CONTROL = "export_control"

class GLSWeatherCondition(str, Enum):
	"""Weather conditions affecting location services."""
	CLEAR = "clear"
	RAIN = "rain"
	SNOW = "snow"
	FOG = "fog"
	STORM = "storm"
	EXTREME_HEAT = "extreme_heat"
	EXTREME_COLD = "extreme_cold"

class GLSH3Resolution(int, Enum):
	"""H3 grid resolution levels for hierarchical indexing."""
	CONTINENT = 0      # ~4,250 km edge length
	COUNTRY = 1        # ~607 km edge length  
	REGION = 2         # ~218 km edge length
	METROPOLITAN = 3   # ~78 km edge length
	CITY = 4           # ~28 km edge length
	DISTRICT = 5       # ~10 km edge length
	NEIGHBORHOOD = 6   # ~3.7 km edge length
	BLOCK = 7          # ~1.3 km edge length
	BUILDING = 8       # ~0.5 km edge length
	PRECISE = 9        # ~0.17 km edge length
	ULTRA_PRECISE = 10 # ~65 m edge length

class GLSFuzzyMatchType(str, Enum):
	"""Fuzzy matching algorithms for location resolution."""
	LEVENSHTEIN = "levenshtein"
	JARO_WINKLER = "jaro_winkler"
	SOUNDEX = "soundex"
	METAPHONE = "metaphone"
	FUZZY_PARTIAL = "fuzzy_partial"
	TOKEN_SET = "token_set"
	TOKEN_SORT = "token_sort"

class GLSAdminLevel(str, Enum):
	"""Administrative division levels."""
	COUNTRY = "country"         # ISO 3166-1 countries
	ADMIN1 = "admin1"          # States, provinces, regions
	ADMIN2 = "admin2"          # Counties, districts, departments
	ADMIN3 = "admin3"          # Municipalities, townships
	ADMIN4 = "admin4"          # Villages, neighborhoods
	LOCALITY = "locality"       # Cities, towns
	SUBLOCALITY = "sublocality" # Districts within cities

class GLSTrajectoryPattern(str, Enum):
	"""Movement trajectory patterns."""
	LINEAR = "linear"
	CIRCULAR = "circular"
	PERIODIC = "periodic"
	RANDOM_WALK = "random_walk"
	COMMUTING = "commuting"
	EXPLORATORY = "exploratory"
	CONSTRAINED = "constrained"
	ANOMALOUS = "anomalous"

class GLSClusteringAlgorithm(str, Enum):
	"""Spatial clustering algorithms."""
	DBSCAN = "dbscan"
	KMEANS = "kmeans"
	GRID_BASED = "grid_based"
	HIERARCHICAL = "hierarchical"
	OPTICS = "optics"
	MEAN_SHIFT = "mean_shift"
	GAUSSIAN_MIXTURE = "gaussian_mixture"

class GLSIndexType(str, Enum):
	"""Spatiotemporal indexing methods."""
	H3_GRID = "h3_grid"
	QUADTREE = "quadtree"
	GEOHASH = "geohash"
	R_TREE = "r_tree"
	KD_TREE = "kd_tree"
	GRID_INDEX = "grid_index"
	TEMPORAL_B_TREE = "temporal_b_tree"

class GLSMapRenderer(str, Enum):
	"""Map rendering engines."""
	FOLIUM = "folium"
	MATPLOTLIB = "matplotlib"
	PLOTLY = "plotly"
	LEAFLET = "leaflet"
	MAPBOX = "mapbox"

class GLSDataSource(str, Enum):
	"""Geographic data sources."""
	GEONAMES = "geonames"
	OPENSTREETMAP = "openstreetmap"
	NATURAL_EARTH = "natural_earth"
	GADM = "gadm"
	WHO_BOUNDARIES = "who_boundaries"
	CUSTOM = "custom"

class GLSExportFormat(str, Enum):
	"""Data export formats."""
	JSON = "json"
	CSV = "csv"
	GEOJSON = "geojson"
	KML = "kml"
	GPX = "gpx"
	SHAPEFILE = "shapefile"
	PARQUET = "parquet"
	HDF5 = "hdf5"

# =============================================================================
# Core Geographic Data Models
# =============================================================================

def validate_latitude(v: float) -> float:
	"""Validate latitude is within valid range."""
	if not -90 <= v <= 90:
		raise ValueError(f"Latitude must be between -90 and 90 degrees, got {v}")
	return v

def validate_longitude(v: float) -> float:
	"""Validate longitude is within valid range."""
	if not -180 <= v <= 180:
		raise ValueError(f"Longitude must be between -180 and 180 degrees, got {v}")
	return v

class GLSCoordinate(BaseModel):
	"""Precise geographic coordinate with validation and metadata."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	latitude: float = Field(..., description="Latitude in decimal degrees")
	longitude: float = Field(..., description="Longitude in decimal degrees")
	altitude: Optional[float] = Field(None, description="Altitude in meters above sea level")
	accuracy_meters: Optional[float] = Field(None, ge=0, description="Horizontal accuracy in meters")
	timestamp: Optional[datetime] = Field(None, description="Timestamp when coordinate was recorded")
	coordinate_system: GLSCoordinateSystem = Field(default=GLSCoordinateSystem.WGS84, description="Coordinate reference system")
	accuracy_level: Optional[GLSLocationAccuracy] = Field(None, description="Accuracy classification")
	source: Optional[str] = Field(None, description="Source of the coordinate data")
	
	# H3 encoding support
	h3_indices: Dict[int, str] = Field(default_factory=dict, description="H3 indices at different resolutions")
	geohash: Optional[str] = Field(None, description="Geohash encoding")
	
	# Spatiotemporal indexing
	spatial_index: Optional[str] = Field(None, description="Spatial index identifier")
	temporal_index: Optional[str] = Field(None, description="Temporal index identifier")
	
	@field_validator('latitude')
	@classmethod
	def validate_latitude_range(cls, v: float) -> float:
		return validate_latitude(v)
	
	@field_validator('longitude')  
	@classmethod
	def validate_longitude_range(cls, v: float) -> float:
		return validate_longitude(v)
	
	@computed_field
	@property
	def primary_h3_index(self) -> Optional[str]:
		"""Get primary H3 index at city level (resolution 4)."""
		return self.h3_indices.get(GLSH3Resolution.CITY.value)

class GLSAddress(BaseModel):
	"""Comprehensive address model with validation and geocoding support."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Address components
	street_number: Optional[str] = Field(None, description="Street number")
	street_name: Optional[str] = Field(None, description="Street name")
	street_type: Optional[str] = Field(None, description="Street type (Ave, St, Rd, etc.)")
	unit_number: Optional[str] = Field(None, description="Unit, apartment, or suite number")
	building_name: Optional[str] = Field(None, description="Building or complex name")
	
	# Locality information
	neighborhood: Optional[str] = Field(None, description="Neighborhood or district")
	city: Optional[str] = Field(None, description="City or municipality")
	county: Optional[str] = Field(None, description="County or equivalent")
	state_province: Optional[str] = Field(None, description="State, province, or region")
	postal_code: Optional[str] = Field(None, description="Postal or ZIP code")
	country: str = Field(..., description="Country name or ISO code")
	
	# Full formatted address
	formatted_address: Optional[str] = Field(None, description="Complete formatted address")
	
	# Geocoding information
	coordinate: Optional[GLSCoordinate] = Field(None, description="Geocoded coordinate")
	geocoding_accuracy: Optional[GLSLocationAccuracy] = Field(None, description="Geocoding accuracy level")
	geocoding_source: Optional[str] = Field(None, description="Geocoding service used")
	geocoding_timestamp: Optional[datetime] = Field(None, description="When address was geocoded")
	
	# Validation status
	is_validated: bool = Field(default=False, description="Whether address has been validated")
	validation_score: Optional[float] = Field(None, ge=0, le=1, description="Address validation confidence score")
	validation_warnings: List[str] = Field(default_factory=list, description="Address validation warnings")
	
	# Fuzzy matching support
	fuzzy_matches: List[Dict[str, Any]] = Field(default_factory=list, description="Fuzzy matching candidates")
	match_confidence: Optional[float] = Field(None, ge=0, le=1, description="Fuzzy match confidence score")
	match_algorithm: Optional[GLSFuzzyMatchType] = Field(None, description="Fuzzy matching algorithm used")
	
	# Administrative resolution
	admin_hierarchy: Dict[GLSAdminLevel, str] = Field(default_factory=dict, description="Administrative hierarchy")
	geonames_id: Optional[int] = Field(None, description="GeoNames database ID")
	admin_codes: Dict[str, str] = Field(default_factory=dict, description="Administrative codes")

class GLSBoundary(BaseModel):
	"""Geographic boundary definition for geofences and territories."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	boundary_type: GLSGeofenceType = Field(..., description="Type of boundary")
	coordinates: List[GLSCoordinate] = Field(..., min_length=1, description="Boundary coordinates")
	
	# Circle-specific properties
	center_point: Optional[GLSCoordinate] = Field(None, description="Center point for circular boundaries")
	radius_meters: Optional[float] = Field(None, gt=0, description="Radius in meters for circular boundaries")
	
	# Additional properties
	elevation_min: Optional[float] = Field(None, description="Minimum elevation in meters")
	elevation_max: Optional[float] = Field(None, description="Maximum elevation in meters")
	coordinate_system: GLSCoordinateSystem = Field(default=GLSCoordinateSystem.WGS84, description="Coordinate system used")
	
	@field_validator('coordinates')
	@classmethod
	def validate_boundary_coordinates(cls, v: List[GLSCoordinate], info) -> List[GLSCoordinate]:
		"""Validate coordinates based on boundary type."""
		boundary_type = info.data.get('boundary_type')
		
		if boundary_type == GLSGeofenceType.CIRCLE:
			if len(v) != 1:
				raise ValueError("Circle boundary must have exactly 1 center coordinate")
		elif boundary_type == GLSGeofenceType.POLYGON:
			if len(v) < 3:
				raise ValueError("Polygon boundary must have at least 3 coordinates")
		elif boundary_type == GLSGeofenceType.RECTANGLE:
			if len(v) != 2:
				raise ValueError("Rectangle boundary must have exactly 2 coordinates (SW and NE corners)")
		
		return v

# =============================================================================
# Geofencing Models
# =============================================================================

class GLSGeofence(GLSBase):
	"""Comprehensive geofence model with rules and compliance."""
	
	name: str = Field(..., min_length=1, max_length=255, description="Geofence name")
	description: Optional[str] = Field(None, description="Geofence description")
	boundary: GLSBoundary = Field(..., description="Geographic boundary definition")
	
	# Configuration
	fence_type: GLSGeofenceType = Field(..., description="Type of geofence")
	priority: int = Field(default=1, ge=1, le=10, description="Processing priority (1=lowest, 10=highest)")
	
	# Rules and actions
	trigger_events: List[GLSEventType] = Field(default_factory=list, description="Events that trigger this geofence")
	entity_filters: Dict[str, Any] = Field(default_factory=dict, description="Filters for which entities trigger this geofence")
	time_restrictions: Optional[Dict[str, Any]] = Field(None, description="Time-based restrictions")
	
	# Actions and notifications
	notification_config: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")
	workflow_triggers: List[str] = Field(default_factory=list, description="Workflows to trigger")
	compliance_requirements: List[GLSComplianceType] = Field(default_factory=list, description="Compliance requirements")
	
	# Status and performance
	entry_count: int = Field(default=0, ge=0, description="Number of entities currently inside")
	total_entries: int = Field(default=0, ge=0, description="Total historical entries")
	last_triggered: Optional[datetime] = Field(None, description="Last time this geofence was triggered")

class GLSGeofenceRule(GLSBase):
	"""Business rules and actions for geofences."""
	
	geofence_id: str = Field(..., description="Associated geofence ID")
	rule_name: str = Field(..., min_length=1, max_length=255, description="Rule name")
	rule_description: Optional[str] = Field(None, description="Rule description")
	
	# Trigger conditions
	trigger_events: List[GLSEventType] = Field(..., min_length=1, description="Events that trigger this rule")
	entity_conditions: Dict[str, Any] = Field(default_factory=dict, description="Entity-based conditions")
	time_conditions: Dict[str, Any] = Field(default_factory=dict, description="Time-based conditions")
	location_conditions: Dict[str, Any] = Field(default_factory=dict, description="Location-based conditions")
	
	# Actions
	immediate_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions to take immediately")
	delayed_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions with delays")
	escalation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Escalation workflow")
	
	# Configuration
	is_enabled: bool = Field(default=True, description="Whether rule is enabled")
	execution_order: int = Field(default=1, ge=1, description="Order of execution when multiple rules apply")

# =============================================================================
# Location Tracking Models
# =============================================================================

class GLSLocationEvent(GLSBase):
	"""Location event model for tracking entity movements."""
	
	entity_id: str = Field(..., description="ID of the entity that generated this event")
	entity_type: GLSEntityType = Field(..., description="Type of entity")
	
	# Event details
	event_type: GLSEventType = Field(..., description="Type of location event")
	event_timestamp: datetime = Field(..., description="When the event occurred")
	coordinate: GLSCoordinate = Field(..., description="Location where event occurred")
	
	# Geofence information
	geofence_id: Optional[str] = Field(None, description="Associated geofence ID")
	previous_geofence_id: Optional[str] = Field(None, description="Previous geofence ID (for exit events)")
	
	# Movement data
	speed_kmh: Optional[float] = Field(None, ge=0, description="Speed in kilometers per hour")
	heading_degrees: Optional[float] = Field(None, ge=0, lt=360, description="Direction of movement in degrees")
	distance_traveled: Optional[float] = Field(None, ge=0, description="Distance traveled since last event in meters")
	
	# Dwell information
	dwell_duration_seconds: Optional[int] = Field(None, ge=0, description="Time spent in location for dwell events")
	
	# Processing status
	is_processed: bool = Field(default=False, description="Whether event has been processed")
	processing_results: Dict[str, Any] = Field(default_factory=dict, description="Results of event processing")
	rule_violations: List[str] = Field(default_factory=list, description="Rules violated by this event")

class GLSEntityLocation(GLSBase):
	"""Current location and status of trackable entities."""
	
	entity_id: str = Field(..., description="Entity identifier")
	entity_type: GLSEntityType = Field(..., description="Type of entity")
	entity_name: Optional[str] = Field(None, description="Human-readable entity name")
	
	# Current location
	current_coordinate: GLSCoordinate = Field(..., description="Current location")
	last_movement: Optional[datetime] = Field(None, description="Last time entity moved")
	
	# Current status
	current_geofences: List[str] = Field(default_factory=list, description="Geofences entity is currently in")
	speed_kmh: Optional[float] = Field(None, ge=0, description="Current speed")
	heading_degrees: Optional[float] = Field(None, ge=0, lt=360, description="Current heading")
	
	# Tracking configuration
	tracking_enabled: bool = Field(default=True, description="Whether tracking is enabled")
	tracking_interval_seconds: int = Field(default=60, ge=1, description="Tracking update interval")
	high_accuracy_mode: bool = Field(default=False, description="Whether to use high accuracy mode")
	
	# Status information
	battery_level: Optional[float] = Field(None, ge=0, le=100, description="Battery level percentage")
	signal_strength: Optional[float] = Field(None, ge=0, le=100, description="Signal strength percentage")
	is_online: bool = Field(default=True, description="Whether entity is currently online")
	last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last time entity was seen")

# =============================================================================
# Territory Management Models  
# =============================================================================

class GLSTerritory(GLSBase):
	"""Territory management for sales, service, and operations."""
	
	name: str = Field(..., min_length=1, max_length=255, description="Territory name")
	description: Optional[str] = Field(None, description="Territory description")
	territory_type: GLSTerritoryType = Field(..., description="Type of territory")
	
	# Geographic definition
	boundary: GLSBoundary = Field(..., description="Territory boundary")
	parent_territory_id: Optional[str] = Field(None, description="Parent territory for hierarchical organization")
	
	# Assignments
	assigned_users: List[str] = Field(default_factory=list, description="Users assigned to this territory")
	assigned_assets: List[str] = Field(default_factory=list, description="Assets assigned to this territory")
	responsible_manager: Optional[str] = Field(None, description="Manager responsible for this territory")
	
	# Rules and configuration
	access_rules: Dict[str, Any] = Field(default_factory=dict, description="Access and permission rules")
	performance_targets: Dict[str, Any] = Field(default_factory=dict, description="Performance targets and KPIs")
	operational_hours: Dict[str, Any] = Field(default_factory=dict, description="Operational hours and schedules")
	
	# Analytics and performance
	customer_count: int = Field(default=0, ge=0, description="Number of customers in territory")
	revenue_target: Optional[Decimal] = Field(None, ge=0, description="Revenue target for territory")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Current performance metrics")

# =============================================================================
# Route and Logistics Models
# =============================================================================

class GLSWaypoint(BaseModel):
	"""Waypoint for route planning and navigation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	coordinate: GLSCoordinate = Field(..., description="Waypoint location")
	address: Optional[GLSAddress] = Field(None, description="Address information")
	name: Optional[str] = Field(None, description="Waypoint name")
	
	# Timing constraints
	arrival_time_window_start: Optional[datetime] = Field(None, description="Earliest acceptable arrival time")
	arrival_time_window_end: Optional[datetime] = Field(None, description="Latest acceptable arrival time")
	service_duration_minutes: int = Field(default=0, ge=0, description="Time required at this location")
	
	# Requirements
	required_skills: List[str] = Field(default_factory=list, description="Skills required for this waypoint")
	priority: int = Field(default=1, ge=1, le=10, description="Waypoint priority")
	special_instructions: Optional[str] = Field(None, description="Special instructions for this location")

class GLSRoute(GLSBase):
	"""Optimized route with waypoints and navigation information."""
	
	route_name: str = Field(..., min_length=1, max_length=255, description="Route name")
	route_description: Optional[str] = Field(None, description="Route description")
	
	# Route definition
	waypoints: List[GLSWaypoint] = Field(..., min_length=2, description="Route waypoints in order")
	optimization_objective: GLSRouteOptimization = Field(..., description="Optimization objective")
	
	# Route properties
	total_distance_km: Optional[float] = Field(None, ge=0, description="Total route distance")
	estimated_duration_minutes: Optional[int] = Field(None, ge=0, description="Estimated travel time")
	estimated_fuel_cost: Optional[Decimal] = Field(None, ge=0, description="Estimated fuel cost")
	
	# Assignments
	assigned_vehicle: Optional[str] = Field(None, description="Assigned vehicle ID")
	assigned_driver: Optional[str] = Field(None, description="Assigned driver ID")
	
	# Status
	route_status: str = Field(default="planned", description="Current route status")
	planned_start_time: Optional[datetime] = Field(None, description="Planned start time")
	actual_start_time: Optional[datetime] = Field(None, description="Actual start time")
	completion_time: Optional[datetime] = Field(None, description="Route completion time")
	
	# Navigation data
	turn_by_turn_directions: List[Dict[str, Any]] = Field(default_factory=list, description="Turn-by-turn navigation")
	traffic_conditions: Dict[str, Any] = Field(default_factory=dict, description="Current traffic conditions")

# =============================================================================
# Analytics and Reporting Models
# =============================================================================

class GLSLocationAnalytics(BaseModel):
	"""Location analytics and insights model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	analysis_period_start: datetime = Field(..., description="Start of analysis period")
	analysis_period_end: datetime = Field(..., description="End of analysis period")
	
	# Entity analytics
	entity_metrics: Dict[str, Any] = Field(default_factory=dict, description="Per-entity metrics")
	movement_patterns: Dict[str, Any] = Field(default_factory=dict, description="Movement pattern analysis")
	dwell_time_analysis: Dict[str, Any] = Field(default_factory=dict, description="Dwell time statistics")
	
	# Geofence analytics
	geofence_performance: Dict[str, Any] = Field(default_factory=dict, description="Geofence utilization metrics")
	violation_summary: Dict[str, Any] = Field(default_factory=dict, description="Rule violation summary")
	
	# Territory analytics
	territory_coverage: Dict[str, Any] = Field(default_factory=dict, description="Territory coverage analysis")
	territory_performance: Dict[str, Any] = Field(default_factory=dict, description="Territory performance metrics")
	
	# Route analytics
	route_efficiency: Dict[str, Any] = Field(default_factory=dict, description="Route efficiency metrics")
	traffic_impact: Dict[str, Any] = Field(default_factory=dict, description="Traffic impact analysis")

class GLSComplianceReport(GLSBase):
	"""Geographic compliance monitoring and reporting."""
	
	report_name: str = Field(..., min_length=1, max_length=255, description="Report name")
	compliance_type: GLSComplianceType = Field(..., description="Type of compliance being monitored")
	
	# Report period
	report_period_start: datetime = Field(..., description="Start of reporting period")
	report_period_end: datetime = Field(..., description="End of reporting period")
	
	# Compliance status
	overall_compliance_score: float = Field(..., ge=0, le=100, description="Overall compliance percentage")
	violations: List[Dict[str, Any]] = Field(default_factory=list, description="Compliance violations found")
	recommendations: List[str] = Field(default_factory=list, description="Compliance improvement recommendations")
	
	# Geographic breakdown
	jurisdiction_compliance: Dict[str, Any] = Field(default_factory=dict, description="Compliance by jurisdiction")
	territory_compliance: Dict[str, Any] = Field(default_factory=dict, description="Compliance by territory")
	
	# Remediation tracking
	remediation_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Remediation actions taken")
	next_review_date: Optional[date] = Field(None, description="Next compliance review date")

# =============================================================================
# Weather and Environmental Models
# =============================================================================

class GLSWeatherData(BaseModel):
	"""Weather information affecting location services."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	coordinate: GLSCoordinate = Field(..., description="Location of weather observation")
	observation_time: datetime = Field(..., description="Time of weather observation")
	
	# Weather conditions
	condition: GLSWeatherCondition = Field(..., description="Primary weather condition")
	temperature_celsius: Optional[float] = Field(None, description="Temperature in Celsius")
	humidity_percentage: Optional[float] = Field(None, ge=0, le=100, description="Relative humidity")
	wind_speed_kmh: Optional[float] = Field(None, ge=0, description="Wind speed in km/h")
	wind_direction_degrees: Optional[float] = Field(None, ge=0, lt=360, description="Wind direction")
	precipitation_mm: Optional[float] = Field(None, ge=0, description="Precipitation in millimeters")
	visibility_km: Optional[float] = Field(None, ge=0, description="Visibility in kilometers")
	
	# Forecasting
	forecast_confidence: Optional[float] = Field(None, ge=0, le=100, description="Forecast confidence percentage")
	weather_alerts: List[str] = Field(default_factory=list, description="Active weather alerts")
	
	# Impact assessment
	location_service_impact: Optional[str] = Field(None, description="Impact on location services")
	recommended_adjustments: List[str] = Field(default_factory=list, description="Recommended service adjustments")

# =============================================================================
# API Request/Response Models
# =============================================================================

class GLSLocationQuery(BaseModel):
	"""Query parameters for location data retrieval."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Entity filters
	entity_ids: Optional[List[str]] = Field(None, description="Specific entity IDs to query")
	entity_types: Optional[List[GLSEntityType]] = Field(None, description="Entity types to include")
	
	# Geographic filters
	bounding_box: Optional[Tuple[GLSCoordinate, GLSCoordinate]] = Field(None, description="SW and NE corners of bounding box")
	geofence_ids: Optional[List[str]] = Field(None, description="Geofence IDs to filter by")
	territory_ids: Optional[List[str]] = Field(None, description="Territory IDs to filter by")
	
	# Time filters
	start_time: Optional[datetime] = Field(None, description="Start time for query")
	end_time: Optional[datetime] = Field(None, description="End time for query")
	
	# Event filters
	event_types: Optional[List[GLSEventType]] = Field(None, description="Event types to include")
	
	# Pagination and limits
	limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of results")
	offset: int = Field(default=0, ge=0, description="Result offset for pagination")
	
	# Sorting
	sort_by: Optional[str] = Field(None, description="Field to sort by")
	sort_order: Literal["asc", "desc"] = Field(default="desc", description="Sort order")

class GLSBatchGeocodeRequest(BaseModel):
	"""Batch geocoding request model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	addresses: List[GLSAddress] = Field(..., min_length=1, max_length=1000, description="Addresses to geocode")
	geocoding_options: Dict[str, Any] = Field(default_factory=dict, description="Geocoding service options")
	return_multiple_matches: bool = Field(default=False, description="Whether to return multiple matches per address")
	confidence_threshold: float = Field(default=0.8, ge=0, le=1, description="Minimum confidence threshold")

class GLSServiceResponse(BaseModel):
	"""Standard service response model."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	success: bool = Field(..., description="Whether operation was successful")
	message: str = Field(..., description="Response message")
	data: Optional[Dict[str, Any]] = Field(None, description="Response data")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	warnings: List[str] = Field(default_factory=list, description="Warning messages")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
	request_id: str = Field(default_factory=uuid7str, description="Unique request identifier")

# =============================================================================
# Advanced Spatiotemporal Models
# =============================================================================

class GLSAdministrativeRegion(BaseModel):
	"""Administrative region with hierarchical information."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	region_id: str = Field(..., description="Unique region identifier")
	geonames_id: Optional[int] = Field(None, description="GeoNames database ID")
	name: str = Field(..., description="Region name")
	name_variants: List[str] = Field(default_factory=list, description="Alternative names and translations")
	admin_level: GLSAdminLevel = Field(..., description="Administrative level")
	parent_region_id: Optional[str] = Field(None, description="Parent region ID")
	iso_codes: Dict[str, str] = Field(default_factory=dict, description="ISO country/region codes")
	
	# Geographic properties
	boundary: GLSBoundary = Field(..., description="Administrative boundary")
	centroid: GLSCoordinate = Field(..., description="Geographic centroid")
	area_km2: Optional[float] = Field(None, ge=0, description="Area in square kilometers")
	population: Optional[int] = Field(None, ge=0, description="Population count")
	
	# Data source information
	data_source: GLSDataSource = Field(..., description="Data source")
	data_quality: float = Field(..., ge=0, le=1, description="Data quality score")
	last_updated: datetime = Field(..., description="Last update timestamp")

class GLSTrajectory(BaseModel):
	"""Entity movement trajectory with pattern analysis."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	trajectory_id: str = Field(default_factory=uuid7str, description="Unique trajectory identifier")
	entity_id: str = Field(..., description="Entity identifier")
	entity_type: GLSEntityType = Field(..., description="Type of entity")
	
	# Trajectory data
	coordinates: List[GLSCoordinate] = Field(..., min_length=2, description="Trajectory coordinates")
	start_time: datetime = Field(..., description="Trajectory start time")
	end_time: datetime = Field(..., description="Trajectory end time")
	
	# Trajectory metrics
	total_distance_km: float = Field(..., ge=0, description="Total distance traveled")
	duration_seconds: int = Field(..., ge=0, description="Total duration in seconds")
	average_speed_kmh: float = Field(..., ge=0, description="Average speed")
	max_speed_kmh: float = Field(..., ge=0, description="Maximum speed")
	
	# Pattern analysis
	detected_patterns: List[GLSTrajectoryPattern] = Field(default_factory=list, description="Detected movement patterns")
	pattern_confidence: Dict[GLSTrajectoryPattern, float] = Field(default_factory=dict, description="Pattern confidence scores")
	anomaly_score: Optional[float] = Field(None, ge=0, le=1, description="Trajectory anomaly score")
	
	# Spatial analysis
	convex_hull: Optional[GLSBoundary] = Field(None, description="Trajectory convex hull")
	visited_h3_cells: Set[str] = Field(default_factory=set, description="H3 cells visited")
	dwell_points: List[Dict[str, Any]] = Field(default_factory=list, description="Identified dwell points")
	
	# Temporal patterns
	temporal_patterns: Dict[str, Any] = Field(default_factory=dict, description="Temporal movement patterns")
	periodic_components: List[Dict[str, Any]] = Field(default_factory=list, description="Periodic pattern components")

class GLSSpatialCluster(BaseModel):
	"""Spatial cluster analysis results."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	cluster_id: str = Field(default_factory=uuid7str, description="Unique cluster identifier")
	algorithm: GLSClusteringAlgorithm = Field(..., description="Clustering algorithm used")
	
	# Cluster properties
	center: GLSCoordinate = Field(..., description="Cluster center point")
	radius_km: float = Field(..., ge=0, description="Cluster radius in kilometers")
	member_count: int = Field(..., ge=1, description="Number of cluster members")
	density: float = Field(..., ge=0, description="Cluster density")
	
	# Cluster members
	member_entities: List[str] = Field(..., description="Entity IDs in cluster")
	member_coordinates: List[GLSCoordinate] = Field(..., description="Member coordinates")
	
	# Temporal properties
	formation_time: datetime = Field(..., description="When cluster formed")
	duration_seconds: Optional[int] = Field(None, ge=0, description="Cluster duration")
	stability_score: float = Field(..., ge=0, le=1, description="Cluster stability score")
	
	# Statistical properties
	confidence_interval: float = Field(..., ge=0, le=1, description="Statistical confidence")
	silhouette_score: Optional[float] = Field(None, ge=-1, le=1, description="Cluster quality score")
	inertia: Optional[float] = Field(None, ge=0, description="Within-cluster sum of squares")

class GLSHotspot(BaseModel):
	"""Spatiotemporal hotspot detection results."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	hotspot_id: str = Field(default_factory=uuid7str, description="Unique hotspot identifier")
	hotspot_type: str = Field(..., description="Type of hotspot (activity, incident, etc.)")
	
	# Spatial properties
	location: GLSCoordinate = Field(..., description="Hotspot center location")
	boundary: GLSBoundary = Field(..., description="Hotspot boundary")
	h3_cells: Set[str] = Field(..., description="H3 cells comprising hotspot")
	
	# Temporal properties
	time_window_start: datetime = Field(..., description="Hotspot time window start")
	time_window_end: datetime = Field(..., description="Hotspot time window end")
	peak_time: Optional[datetime] = Field(None, description="Peak activity time")
	
	# Statistical properties
	intensity: float = Field(..., ge=0, description="Hotspot intensity score")
	significance_level: float = Field(..., ge=0, le=1, description="Statistical significance")
	z_score: Optional[float] = Field(None, description="Z-score for statistical testing")
	p_value: Optional[float] = Field(None, ge=0, le=1, description="P-value for significance test")
	
	# Event aggregation
	event_count: int = Field(..., ge=0, description="Number of events in hotspot")
	event_density: float = Field(..., ge=0, description="Events per unit area/time")
	contributing_entities: List[str] = Field(default_factory=list, description="Entities contributing to hotspot")

class GLSPredictionModel(BaseModel):
	"""Predictive modeling results for entities and events."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	model_id: str = Field(default_factory=uuid7str, description="Unique model identifier")
	model_type: str = Field(..., description="Type of prediction model")
	entity_id: Optional[str] = Field(None, description="Target entity ID if applicable")
	
	# Model metadata
	training_period_start: datetime = Field(..., description="Training data start time")
	training_period_end: datetime = Field(..., description="Training data end time")
	model_accuracy: float = Field(..., ge=0, le=1, description="Model accuracy score")
	confidence_interval: float = Field(..., ge=0, le=1, description="Prediction confidence")
	
	# Predictions
	predicted_locations: List[Tuple[GLSCoordinate, datetime, float]] = Field(
		default_factory=list, description="Predicted locations with timestamps and confidence"
	)
	predicted_events: List[Dict[str, Any]] = Field(default_factory=list, description="Predicted events")
	risk_assessment: Dict[str, float] = Field(default_factory=dict, description="Risk scores by category")
	
	# Model features
	feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
	model_parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
	
	# Validation metrics
	validation_metrics: Dict[str, float] = Field(default_factory=dict, description="Model validation metrics")
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last model update")

class GLSAnomalyDetection(BaseModel):
	"""Anomaly detection results for spatiotemporal data."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	anomaly_id: str = Field(default_factory=uuid7str, description="Unique anomaly identifier")
	entity_id: str = Field(..., description="Entity with anomalous behavior")
	detection_time: datetime = Field(..., description="When anomaly was detected")
	
	# Anomaly properties
	anomaly_type: str = Field(..., description="Type of anomaly detected")
	anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly severity score")
	confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
	
	# Spatial anomaly details
	anomalous_location: Optional[GLSCoordinate] = Field(None, description="Anomalous location")
	expected_location: Optional[GLSCoordinate] = Field(None, description="Expected location")
	spatial_deviation_km: Optional[float] = Field(None, ge=0, description="Spatial deviation in kilometers")
	
	# Temporal anomaly details
	anomalous_time: Optional[datetime] = Field(None, description="Anomalous time")
	expected_time_window: Optional[Tuple[datetime, datetime]] = Field(None, description="Expected time window")
	temporal_deviation_seconds: Optional[int] = Field(None, ge=0, description="Temporal deviation in seconds")
	
	# Context and explanation
	context_factors: List[str] = Field(default_factory=list, description="Contextual factors")
	explanation: Optional[str] = Field(None, description="Anomaly explanation")
	recommended_actions: List[str] = Field(default_factory=list, description="Recommended response actions")

class GLSMapConfiguration(BaseModel):
	"""Map rendering and visualization configuration."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	renderer: GLSMapRenderer = Field(..., description="Map rendering engine")
	tile_provider: str = Field(default="openstreetmap", description="Map tile provider")
	
	# Map properties
	center: GLSCoordinate = Field(..., description="Map center coordinate")
	zoom_level: int = Field(default=10, ge=1, le=20, description="Initial zoom level")
	width: int = Field(default=800, ge=100, description="Map width in pixels")
	height: int = Field(default=600, ge=100, description="Map height in pixels")
	
	# Styling
	color_scheme: str = Field(default="default", description="Color scheme name")
	style_options: Dict[str, Any] = Field(default_factory=dict, description="Renderer-specific style options")
	
	# Layers
	base_layers: List[str] = Field(default_factory=list, description="Base map layers")
	overlay_layers: List[str] = Field(default_factory=list, description="Overlay layers")
	
	# Export settings
	export_format: GLSExportFormat = Field(default=GLSExportFormat.JSON, description="Default export format")
	dpi: int = Field(default=300, ge=72, description="Export DPI for raster formats")
	
	# Interactive features
	enable_clustering: bool = Field(default=True, description="Enable marker clustering")
	enable_heatmap: bool = Field(default=False, description="Enable heat map overlay")
	enable_animation: bool = Field(default=False, description="Enable temporal animation")
	animation_speed: float = Field(default=1.0, gt=0, description="Animation speed multiplier")

class GLSRealTimeStream(BaseModel):
	"""Real-time data streaming configuration and status."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	stream_id: str = Field(default_factory=uuid7str, description="Unique stream identifier")
	entity_id: str = Field(..., description="Entity being streamed")
	
	# Streaming configuration
	update_interval_seconds: int = Field(default=30, ge=1, description="Update interval in seconds")
	data_retention_hours: int = Field(default=24, ge=1, description="Data retention period")
	max_buffer_size: int = Field(default=1000, ge=1, description="Maximum buffer size")
	
	# Stream status
	is_active: bool = Field(default=True, description="Whether stream is active")
	last_update: datetime = Field(default_factory=datetime.utcnow, description="Last data update")
	update_count: int = Field(default=0, ge=0, description="Total updates received")
	
	# Quality metrics
	data_quality_score: float = Field(default=1.0, ge=0, le=1, description="Stream data quality")
	latency_ms: Optional[int] = Field(None, ge=0, description="Average latency in milliseconds")
	packet_loss_rate: float = Field(default=0.0, ge=0, le=1, description="Packet loss rate")
	
	# Connected clients
	connected_clients: int = Field(default=0, ge=0, description="Number of connected clients")
	client_subscriptions: List[str] = Field(default_factory=list, description="Client subscription IDs")

class GLSSpatialIndex(BaseModel):
	"""Spatial indexing configuration and metadata."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	index_id: str = Field(default_factory=uuid7str, description="Unique index identifier")
	index_type: GLSIndexType = Field(..., description="Type of spatial index")
	
	# Index configuration
	h3_resolution: Optional[GLSH3Resolution] = Field(None, description="H3 resolution level")
	grid_size_meters: Optional[float] = Field(None, gt=0, description="Grid size for grid-based indices")
	max_depth: Optional[int] = Field(None, ge=1, description="Maximum tree depth")
	
	# Index statistics
	total_entries: int = Field(default=0, ge=0, description="Total indexed entries")
	index_size_bytes: int = Field(default=0, ge=0, description="Index size in bytes")
	build_time_seconds: Optional[float] = Field(None, ge=0, description="Index build time")
	
	# Performance metrics
	average_query_time_ms: Optional[float] = Field(None, ge=0, description="Average query time")
	cache_hit_rate: float = Field(default=0.0, ge=0, le=1, description="Cache hit rate")
	
	# Maintenance
	last_rebuild: Optional[datetime] = Field(None, description="Last index rebuild time")
	needs_rebuild: bool = Field(default=False, description="Whether index needs rebuilding")

# =============================================================================
# Enhanced Request/Response Models
# =============================================================================

class GLSFuzzySearchRequest(BaseModel):
	"""Fuzzy location search request with advanced matching."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	query: str = Field(..., min_length=1, description="Search query string")
	match_types: List[GLSFuzzyMatchType] = Field(..., min_length=1, description="Fuzzy matching algorithms to use")
	confidence_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum confidence threshold")
	max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
	
	# Geographic constraints
	bounding_box: Optional[Tuple[GLSCoordinate, GLSCoordinate]] = Field(None, description="Geographic constraint")
	admin_levels: List[GLSAdminLevel] = Field(default_factory=list, description="Administrative levels to search")
	data_sources: List[GLSDataSource] = Field(default_factory=list, description="Data sources to search")
	
	# Additional filters
	population_min: Optional[int] = Field(None, ge=0, description="Minimum population filter")
	area_min_km2: Optional[float] = Field(None, ge=0, description="Minimum area filter")

class GLSTrajectoryAnalysisRequest(BaseModel):
	"""Trajectory analysis request with pattern detection."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	entity_id: str = Field(..., description="Entity to analyze")
	time_window_start: datetime = Field(..., description="Analysis time window start")
	time_window_end: datetime = Field(..., description="Analysis time window end")
	
	# Analysis options
	detect_patterns: bool = Field(default=True, description="Enable pattern detection")
	detect_anomalies: bool = Field(default=True, description="Enable anomaly detection")
	cluster_dwell_points: bool = Field(default=True, description="Cluster dwell points")
	
	# Parameters
	min_dwell_time_seconds: int = Field(default=300, ge=0, description="Minimum dwell time")
	speed_threshold_kmh: float = Field(default=5.0, ge=0, description="Speed threshold for stops")
	pattern_confidence_threshold: float = Field(default=0.8, ge=0, le=1, description="Pattern confidence threshold")

class GLSHotspotAnalysisRequest(BaseModel):
	"""Hotspot analysis request with spatiotemporal clustering."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	analysis_area: GLSBoundary = Field(..., description="Area to analyze")
	time_window_start: datetime = Field(..., description="Analysis time window start")
	time_window_end: datetime = Field(..., description="Analysis time window end")
	
	# Hotspot parameters
	h3_resolution: GLSH3Resolution = Field(default=GLSH3Resolution.NEIGHBORHOOD, description="H3 resolution for analysis")
	clustering_algorithm: GLSClusteringAlgorithm = Field(default=GLSClusteringAlgorithm.DBSCAN, description="Clustering algorithm")
	significance_level: float = Field(default=0.05, ge=0, le=1, description="Statistical significance level")
	
	# Filters
	entity_types: List[GLSEntityType] = Field(default_factory=list, description="Entity types to include")
	event_types: List[GLSEventType] = Field(default_factory=list, description="Event types to include")
	min_events_per_hotspot: int = Field(default=10, ge=1, description="Minimum events per hotspot")

class GLSPredictiveAnalysisRequest(BaseModel):
	"""Predictive analysis request for forecasting."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	entity_id: str = Field(..., description="Entity to predict")
	prediction_horizon_hours: int = Field(..., ge=1, le=168, description="Prediction horizon in hours")
	
	# Model configuration
	model_type: str = Field(default="lstm", description="Prediction model type")
	training_days: int = Field(default=30, ge=7, description="Training data period in days")
	confidence_level: float = Field(default=0.95, ge=0, le=1, description="Confidence level for predictions")
	
	# Features to include
	include_weather: bool = Field(default=True, description="Include weather data")
	include_traffic: bool = Field(default=True, description="Include traffic data")
	include_events: bool = Field(default=True, description="Include event data")
	include_patterns: bool = Field(default=True, description="Include historical patterns")

# =============================================================================
# Model Registry and Exports
# =============================================================================

__all__ = [
	# Base models
	"GLSBase",
	
	# Enums
	"GLSCoordinateSystem",
	"GLSLocationAccuracy", 
	"GLSGeofenceType",
	"GLSEventType",
	"GLSEntityType",
	"GLSTerritoryType",
	"GLSRouteOptimization",
	"GLSComplianceType",
	"GLSWeatherCondition",
	"GLSH3Resolution",
	"GLSFuzzyMatchType",
	"GLSAdminLevel",
	"GLSTrajectoryPattern",
	"GLSClusteringAlgorithm",
	"GLSIndexType",
	"GLSMapRenderer",
	"GLSDataSource",
	"GLSExportFormat",
	
	# Core geographic models
	"GLSCoordinate",
	"GLSAddress", 
	"GLSBoundary",
	
	# Geofencing models
	"GLSGeofence",
	"GLSGeofenceRule",
	
	# Location tracking models
	"GLSLocationEvent",
	"GLSEntityLocation",
	
	# Territory models
	"GLSTerritory",
	
	# Route and logistics models
	"GLSWaypoint",
	"GLSRoute",
	
	# Analytics models
	"GLSLocationAnalytics",
	"GLSComplianceReport",
	
	# Weather and environmental models
	"GLSWeatherData",
	
	# Advanced spatiotemporal models
	"GLSAdministrativeRegion",
	"GLSTrajectory",
	"GLSSpatialCluster",
	"GLSHotspot",
	"GLSPredictionModel",
	"GLSAnomalyDetection",
	"GLSMapConfiguration",
	"GLSRealTimeStream",
	"GLSSpatialIndex",
	
	# Enhanced request/response models
	"GLSFuzzySearchRequest",
	"GLSTrajectoryAnalysisRequest",
	"GLSHotspotAnalysisRequest",
	"GLSPredictiveAnalysisRequest",
	
	# API models
	"GLSLocationQuery",
	"GLSBatchGeocodeRequest",
	"GLSServiceResponse"
]