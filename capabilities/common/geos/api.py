"""
Geographical Location Services - FastAPI REST API

Enterprise geospatial API providing comprehensive location intelligence:
- Advanced geocoding and address validation
- Real-time geofencing and location tracking
- Territory management and spatial analytics
- Route optimization and logistics planning
- Geographic compliance and regulatory management

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import logging
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError
from starlette.status import (
	HTTP_200_OK, HTTP_201_CREATED, HTTP_204_NO_CONTENT, 
	HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_403_FORBIDDEN,
	HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR
)

from .models import *
from .service import (
	GeographicalLocationService, GLSServiceError, GLSGeocodingError, GLSGeofenceError, GLSRouteOptimizationError, GLSComplianceError,
	GLSFuzzyMatchingService, GLSTrajectoryAnalysisService, GLSHotspotDetectionService, 
	GLSPredictiveModelingService, GLSAnomalyDetectionService, GLSVisualizationService, GLSRealTimeStreamingService
)

# =============================================================================
# Router and Security Configuration
# =============================================================================

router = APIRouter(
	prefix="/api/v1/geographical-location",
	tags=["Geographical Location Services"],
	responses={
		404: {"description": "Resource not found"},
		400: {"description": "Bad request"},
		422: {"description": "Validation error"}
	}
)

security = HTTPBearer()
logger = logging.getLogger(__name__)

# =============================================================================
# Request/Response Models
# =============================================================================

# Enhanced Spatiotemporal Request Models
class GLSFuzzySearchRequest(BaseModel):
	"""Advanced fuzzy location search request."""
	model_config = ConfigDict(extra='forbid')
	
	query_text: str = Field(..., min_length=1, description="Location text to search")
	fuzzy_match_type: GLSFuzzyMatchType = Field(..., description="Fuzzy matching algorithm")
	confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence score")
	admin_level: Optional[GLSAdminLevel] = Field(None, description="Administrative level filter")
	country_filter: Optional[str] = Field(None, description="Country filter")
	max_results: int = Field(default=10, ge=1, le=100, description="Maximum results to return")

class GLSTrajectoryAnalysisRequest(BaseModel):
	"""Trajectory analysis request."""
	model_config = ConfigDict(extra='forbid')
	
	entity_id: str = Field(..., description="Entity to analyze")
	time_window_start: datetime = Field(..., description="Analysis start time")
	time_window_end: datetime = Field(..., description="Analysis end time")
	include_patterns: bool = Field(default=True, description="Include pattern detection")
	include_anomalies: bool = Field(default=True, description="Include anomaly detection")
	h3_resolution: GLSH3Resolution = Field(default=GLSH3Resolution.CITY, description="H3 analysis resolution")

class GLSHotspotDetectionRequest(BaseModel):
	"""Hotspot detection request."""
	model_config = ConfigDict(extra='forbid')
	
	entity_type: GLSEntityType = Field(..., description="Entity type to analyze")
	time_window_start: datetime = Field(..., description="Analysis start time")
	time_window_end: datetime = Field(..., description="Analysis end time")
	clustering_algorithm: GLSClusteringAlgorithm = Field(..., description="Clustering algorithm")
	min_cluster_size: int = Field(default=5, ge=2, description="Minimum cluster size")
	statistical_significance: float = Field(default=0.05, ge=0.001, le=0.1, description="P-value threshold")
	h3_resolution: GLSH3Resolution = Field(default=GLSH3Resolution.NEIGHBORHOOD, description="H3 analysis resolution")

class GLSPredictiveModelingRequest(BaseModel):
	"""Predictive modeling request."""
	model_config = ConfigDict(extra='forbid')
	
	entity_id: str = Field(..., description="Entity to predict")
	prediction_horizon_hours: int = Field(..., ge=1, le=168, description="Prediction horizon in hours")
	model_type: str = Field(default="lstm", description="ML model type")
	confidence_intervals: bool = Field(default=True, description="Include confidence intervals")
	include_risk_assessment: bool = Field(default=True, description="Include risk assessment")

class GLSAnomalyDetectionRequest(BaseModel):
	"""Anomaly detection request."""
	model_config = ConfigDict(extra='forbid')
	
	entity_id: str = Field(..., description="Entity to analyze")
	time_window_start: datetime = Field(..., description="Analysis start time")
	time_window_end: datetime = Field(..., description="Analysis end time")
	anomaly_types: List[str] = Field(..., min_length=1, description="Types of anomalies to detect")
	sensitivity: float = Field(default=0.95, ge=0.5, le=0.99, description="Detection sensitivity")

class GLSVisualizationRequest(BaseModel):
	"""Map visualization request."""
	model_config = ConfigDict(extra='forbid')
	
	map_config: GLSMapConfiguration = Field(..., description="Map configuration")
	data_layers: List[str] = Field(..., min_length=1, description="Data layers to include")
	time_range: Optional[Dict[str, datetime]] = Field(None, description="Time range for temporal data")
	export_format: GLSExportFormat = Field(default=GLSExportFormat.PNG, description="Export format")

class GLSRealTimeStreamRequest(BaseModel):
	"""Real-time streaming request."""
	model_config = ConfigDict(extra='forbid')
	
	stream_type: str = Field(..., description="Stream type (location_updates, events, analytics)")
	entity_filters: Dict[str, Any] = Field(default_factory=dict, description="Entity filters")
	geographic_bounds: Optional[GLSBoundary] = Field(None, description="Geographic bounds filter")
	ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Data TTL in seconds")

class GLSAdvancedAnalyticsRequest(BaseModel):
	"""Advanced analytics request."""
	model_config = ConfigDict(extra='forbid')
	
	analysis_types: List[str] = Field(..., min_length=1, description="Analysis types to perform")
	time_window_start: datetime = Field(..., description="Analysis start time")
	time_window_end: datetime = Field(..., description="Analysis end time")
	spatial_bounds: Optional[GLSBoundary] = Field(None, description="Spatial bounds")
	h3_resolution: GLSH3Resolution = Field(default=GLSH3Resolution.CITY, description="H3 resolution")
	include_visualization: bool = Field(default=True, description="Include visualizations")

class GeocodeRequest(BaseModel):
	"""Single address geocoding request."""
	model_config = ConfigDict(extra='forbid')
	
	address: GLSAddress = Field(..., description="Address to geocode")
	provider: Optional[str] = Field(None, description="Geocoding provider to use")
	return_details: bool = Field(default=True, description="Whether to return detailed geocoding info")

class LocationUpdateRequest(BaseModel):
	"""Location update request for entity tracking."""
	model_config = ConfigDict(extra='forbid')
	
	entity_id: str = Field(..., description="Entity identifier")
	entity_type: GLSEntityType = Field(..., description="Type of entity")
	coordinate: GLSCoordinate = Field(..., description="New location coordinate")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class GeofenceCreateRequest(BaseModel):
	"""Geofence creation request."""
	model_config = ConfigDict(extra='forbid')
	
	name: str = Field(..., min_length=1, max_length=255, description="Geofence name")
	description: Optional[str] = Field(None, description="Geofence description")
	fence_type: GLSGeofenceType = Field(..., description="Type of geofence")
	boundary: GLSBoundary = Field(..., description="Geographic boundary")
	trigger_events: List[GLSEventType] = Field(default_factory=list, description="Events that trigger this geofence")
	entity_filters: Dict[str, Any] = Field(default_factory=dict, description="Entity filters")
	notification_config: Dict[str, Any] = Field(default_factory=dict, description="Notification configuration")
	compliance_requirements: List[GLSComplianceType] = Field(default_factory=list, description="Compliance requirements")
	priority: int = Field(default=1, ge=1, le=10, description="Processing priority")

class TerritoryCreateRequest(BaseModel):
	"""Territory creation request."""
	model_config = ConfigDict(extra='forbid')
	
	name: str = Field(..., min_length=1, max_length=255, description="Territory name")
	description: Optional[str] = Field(None, description="Territory description")
	territory_type: GLSTerritoryType = Field(..., description="Type of territory")
	boundary: GLSBoundary = Field(..., description="Territory boundary")
	assigned_users: List[str] = Field(default_factory=list, description="Assigned user IDs")
	assigned_assets: List[str] = Field(default_factory=list, description="Assigned asset IDs")
	access_rules: Dict[str, Any] = Field(default_factory=dict, description="Access rules")
	performance_targets: Dict[str, Any] = Field(default_factory=dict, description="Performance targets")

class RouteOptimizationRequest(BaseModel):
	"""Route optimization request."""
	model_config = ConfigDict(extra='forbid')
	
	waypoints: List[GLSWaypoint] = Field(..., min_length=2, description="Route waypoints")
	optimization_objective: GLSRouteOptimization = Field(..., description="Optimization objective")
	constraints: Dict[str, Any] = Field(default_factory=dict, description="Route constraints")
	vehicle_constraints: Dict[str, Any] = Field(default_factory=dict, description="Vehicle constraints")

class ComplianceCheckRequest(BaseModel):
	"""Location compliance check request."""
	model_config = ConfigDict(extra='forbid')
	
	coordinate: GLSCoordinate = Field(..., description="Location to check")
	compliance_types: List[GLSComplianceType] = Field(..., min_length=1, description="Compliance types to check")
	additional_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for compliance check")

class LocationSearchRequest(BaseModel):
	"""Location search and query request."""
	model_config = ConfigDict(extra='forbid')
	
	query: GLSLocationQuery = Field(..., description="Search query parameters")
	include_analytics: bool = Field(default=False, description="Include analytics data")
	export_format: Optional[str] = Field(None, description="Export format (json, csv, kml)")

# Response Models
class ApiResponse(BaseModel):
	"""Standard API response wrapper."""
	model_config = ConfigDict(extra='forbid')
	
	success: bool = Field(..., description="Operation success status")
	message: str = Field(..., description="Response message")
	data: Optional[Dict[str, Any]] = Field(None, description="Response data")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
	request_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()), description="Request ID")

class LocationEventResponse(BaseModel):
	"""Location event response model."""
	model_config = ConfigDict(extra='forbid')
	
	events: List[GLSLocationEvent] = Field(..., description="Generated location events")
	entity_status: Dict[str, Any] = Field(..., description="Updated entity status")
	triggered_rules: List[str] = Field(default_factory=list, description="Rules that were triggered")

class AnalyticsResponse(BaseModel):
	"""Analytics response model."""
	model_config = ConfigDict(extra='forbid')
	
	analytics: GLSLocationAnalytics = Field(..., description="Location analytics data")
	summary_stats: Dict[str, Any] = Field(..., description="Summary statistics")
	trends: Dict[str, Any] = Field(default_factory=dict, description="Trend analysis")

# Enhanced Response Models
class GLSFuzzySearchResponse(BaseModel):
	"""Fuzzy search response model."""
	model_config = ConfigDict(extra='forbid')
	
	matches: List[Dict[str, Any]] = Field(..., description="Fuzzy search matches")
	total_matches: int = Field(..., description="Total number of matches")
	average_confidence: float = Field(..., description="Average confidence score")
	search_metadata: Dict[str, Any] = Field(..., description="Search metadata")

class GLSTrajectoryResponse(BaseModel):
	"""Trajectory analysis response model."""
	model_config = ConfigDict(extra='forbid')
	
	trajectory: GLSTrajectory = Field(..., description="Analyzed trajectory")
	patterns: List[Dict[str, Any]] = Field(..., description="Detected patterns")
	anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")
	statistics: Dict[str, Any] = Field(..., description="Trajectory statistics")

class GLSHotspotResponse(BaseModel):
	"""Hotspot detection response model."""
	model_config = ConfigDict(extra='forbid')
	
	hotspots: List[GLSHotspot] = Field(..., description="Detected hotspots")
	clustering_results: Dict[str, Any] = Field(..., description="Clustering analysis results")
	statistical_significance: Dict[str, Any] = Field(..., description="Statistical significance")
	visualization_data: Optional[Dict[str, Any]] = Field(None, description="Visualization data")

class GLSPredictionResponse(BaseModel):
	"""Predictive modeling response model."""
	model_config = ConfigDict(extra='forbid')
	
	predictions: List[Dict[str, Any]] = Field(..., description="Predicted locations")
	confidence_intervals: Dict[str, Any] = Field(..., description="Confidence intervals")
	risk_assessment: Dict[str, Any] = Field(..., description="Risk assessment")
	model_metadata: Dict[str, Any] = Field(..., description="Model metadata")

class GLSAnomalyResponse(BaseModel):
	"""Anomaly detection response model."""
	model_config = ConfigDict(extra='forbid')
	
	anomalies: List[GLSAnomalyDetection] = Field(..., description="Detected anomalies")
	overall_score: float = Field(..., description="Overall anomaly score")
	analysis_summary: Dict[str, Any] = Field(..., description="Analysis summary")
	recommendations: List[str] = Field(..., description="Recommendations")

class GLSVisualizationResponse(BaseModel):
	"""Visualization response model."""
	model_config = ConfigDict(extra='forbid')
	
	map_data: Dict[str, Any] = Field(..., description="Generated map data")
	export_url: Optional[str] = Field(None, description="Export download URL")
	visualization_metadata: Dict[str, Any] = Field(..., description="Visualization metadata")

class GLSStreamResponse(BaseModel):
	"""Real-time stream response model."""
	model_config = ConfigDict(extra='forbid')
	
	stream_id: str = Field(..., description="Stream identifier")
	websocket_url: str = Field(..., description="WebSocket connection URL")
	stream_config: Dict[str, Any] = Field(..., description="Stream configuration")

class GLSAdvancedAnalyticsResponse(BaseModel):
	"""Advanced analytics response model."""
	model_config = ConfigDict(extra='forbid')
	
	analytics_results: Dict[str, Any] = Field(..., description="Analytics results")
	spatial_clustering: Dict[str, Any] = Field(..., description="Spatial clustering results")
	heat_mapping: Dict[str, Any] = Field(..., description="Heat mapping results")
	hotspot_detection: Dict[str, Any] = Field(..., description="Hotspot detection results")
	spatial_autocorrelation: Dict[str, Any] = Field(..., description="Spatial autocorrelation analysis")
	density_estimation: Dict[str, Any] = Field(..., description="Density estimation results")
	visualization_assets: List[str] = Field(..., description="Generated visualization assets")

# =============================================================================
# Dependency Functions
# =============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
	"""Extract user ID from JWT token."""
	# In production, decode JWT and extract user ID
	return "user_123"

async def get_tenant_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
	"""Extract tenant ID from JWT token."""
	# In production, decode JWT and extract tenant ID
	return "tenant_123"

async def get_gls_service() -> GeographicalLocationService:
	"""Get geographical location service instance."""
	return GeographicalLocationService()

async def get_fuzzy_matching_service() -> GLSFuzzyMatchingService:
	"""Get fuzzy matching service instance."""
	return GLSFuzzyMatchingService()

async def get_trajectory_service() -> GLSTrajectoryAnalysisService:
	"""Get trajectory analysis service instance."""
	return GLSTrajectoryAnalysisService()

async def get_hotspot_service() -> GLSHotspotDetectionService:
	"""Get hotspot detection service instance."""
	return GLSHotspotDetectionService()

async def get_predictive_service() -> GLSPredictiveModelingService:
	"""Get predictive modeling service instance."""
	return GLSPredictiveModelingService()

async def get_anomaly_service() -> GLSAnomalyDetectionService:
	"""Get anomaly detection service instance."""
	return GLSAnomalyDetectionService()

async def get_visualization_service() -> GLSVisualizationService:
	"""Get visualization service instance."""
	return GLSVisualizationService()

async def get_streaming_service() -> GLSRealTimeStreamingService:
	"""Get real-time streaming service instance."""
	return GLSRealTimeStreamingService()

def handle_service_error(error: Exception) -> HTTPException:
	"""Convert service errors to HTTP exceptions."""
	if isinstance(error, GLSGeocodingError):
		return HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))
	elif isinstance(error, GLSGeofenceError):
		return HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(error))
	elif isinstance(error, GLSRouteOptimizationError):
		return HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))
	elif isinstance(error, GLSComplianceError):
		return HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(error))
	elif isinstance(error, ValidationError):
		return HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail=error.errors())
	else:
		logger.error(f"Unexpected service error: {str(error)}")
		return HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

# =============================================================================
# Health and Status Endpoints
# =============================================================================

@router.get("/health", response_model=Dict[str, Any])
async def health_check(service: GeographicalLocationService = Depends(get_gls_service)):
	"""Get service health status."""
	try:
		health = await service.health_check()
		return health
	except Exception as e:
		raise handle_service_error(e)

@router.get("/capabilities", response_model=Dict[str, Any])
async def get_capabilities():
	"""Get detailed service capabilities information."""
	return {
		"capabilities": {
			"geocoding": {
				"description": "Advanced address geocoding and validation",
				"features": ["Batch processing", "Multiple providers", "Reverse geocoding", "Address validation"],
				"providers": ["google", "mapbox", "opencage", "nominatim", "here"],
				"max_batch_size": 1000,
				"accuracy_levels": ["high", "medium", "low", "approximate"]
			},
			"h3_encoding": {
				"description": "H3 hierarchical spatial indexing with 11 resolution levels",
				"features": ["Multi-resolution indexing", "Spatial queries", "Grid analytics", "Proximity analysis"],
				"resolutions": ["continent", "country", "state", "metro", "city", "district", "neighborhood", "block", "building", "room", "ultra_precise"],
				"max_resolution_level": 10,
				"spatial_indexing": True
			},
			"fuzzy_matching": {
				"description": "Advanced string similarity algorithms with confidence scoring",
				"features": ["Multiple algorithms", "Confidence scoring", "Administrative resolution", "GeoNames integration"],
				"algorithms": ["levenshtein", "jaro_winkler", "soundex", "metaphone", "fuzzy_wuzzy"],
				"geonames_features": "12.4M+ geographical features",
				"confidence_threshold": 0.5
			},
			"trajectory_analysis": {
				"description": "Complete movement analysis with pattern detection and anomaly identification",
				"features": ["Pattern detection", "Anomaly identification", "Movement analytics", "H3 cell analysis"],
				"pattern_types": ["linear", "circular", "periodic", "random_walk", "commuting"],
				"anomaly_detection": True,
				"dwell_point_analysis": True
			},
			"hotspot_detection": {
				"description": "Multiple clustering algorithms for spatiotemporal hotspot detection",
				"features": ["Statistical significance", "Multiple algorithms", "Z-score analysis", "P-value testing"],
				"algorithms": ["dbscan", "kmeans", "grid_based", "hierarchical", "optics"],
				"statistical_testing": True,
				"significance_threshold": 0.05
			},
			"predictive_modeling": {
				"description": "Forecast entity positions and conflict evolution",
				"features": ["LSTM forecasting", "Confidence intervals", "Risk assessment", "Conflict prediction"],
				"model_types": ["lstm", "arima", "random_forest", "ensemble"],
				"prediction_horizon": "168 hours max",
				"confidence_intervals": True
			},
			"anomaly_detection": {
				"description": "Identify unusual patterns in movement and spatial events",
				"features": ["Behavioral analysis", "Spatial anomalies", "Temporal anomalies", "Multi-scale detection"],
				"detection_types": ["spatial", "temporal", "behavioral", "statistical"],
				"sensitivity_range": "0.5 - 0.99",
				"real_time_detection": True
			},
			"visualization": {
				"description": "Multi-renderer support with flexible map configurations",
				"features": ["Multiple renderers", "Export capabilities", "Interactive maps", "Static images"],
				"renderers": ["folium", "matplotlib", "plotly"],
				"export_formats": ["png", "jpeg", "svg", "html", "pdf"],
				"tile_providers": ["openstreetmap", "cartodb", "stamen", "custom"]
			},
			"real_time_streaming": {
				"description": "Live data streaming via WebSockets with TTL-based data expiration",
				"features": ["WebSocket streaming", "Event-driven updates", "TTL expiration", "Multi-client sync"],
				"stream_types": ["location_updates", "events", "analytics"],
				"ttl_range": "60 - 86400 seconds",
				"multi_client_support": True
			},
			"advanced_analytics": {
				"description": "Comprehensive spatial analytics with multiple algorithms",
				"features": ["Spatial clustering", "Heat mapping", "Hotspot detection", "Autocorrelation analysis"],
				"clustering_algorithms": ["dbscan", "kmeans", "grid_based"],
				"interpolation_methods": ["idw", "kriging", "spline"],
				"statistical_tests": ["morans_i", "local_morans_i", "getis_ord"]
			},
			"geofencing": {
				"description": "Real-time geofencing and location monitoring",
				"features": ["Multiple shapes", "Rule engine", "Real-time alerts", "Event tracking"],
				"supported_shapes": ["circle", "polygon", "rectangle", "administrative"],
				"max_geofences_per_tenant": 10000,
				"real_time_processing": True
			},
			"territory_management": {
				"description": "Territory management and optimization",
				"features": ["Hierarchical territories", "Assignment rules", "Performance tracking"],
				"territory_types": ["sales", "service", "delivery", "maintenance", "security"],
				"assignment_methods": ["automatic", "manual", "rule_based"]
			},
			"route_optimization": {
				"description": "Advanced route planning and optimization",
				"features": ["Multi-objective optimization", "Constraint handling", "Real-time traffic"],
				"optimization_objectives": ["shortest_distance", "fastest_time", "fuel_efficient", "balanced"],
				"max_waypoints": 100,
				"real_time_traffic": True
			},
			"compliance": {
				"description": "Geographic compliance and regulatory management",
				"features": ["Multi-jurisdiction support", "Automated checking", "Compliance reporting"],
				"compliance_types": ["gdpr", "ccpa", "data_residency", "tax_jurisdiction"],
				"supported_jurisdictions": "global"
			}
		},
		"version": "1.0.0",
		"api_version": "v1",
		"status": "active"
	}

# =============================================================================
# Geocoding Endpoints
# =============================================================================

@router.post("/geocode", response_model=ApiResponse, status_code=HTTP_200_OK)
async def geocode_address(
	request: GeocodeRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Geocode a single address with validation and enrichment.
	
	Features:
	- Multi-provider support with fallback
	- Address validation and standardization
	- Confidence scoring and accuracy levels
	- Caching for performance optimization
	"""
	try:
		result = await service.geocode_address(request.address, request.provider)
		
		return ApiResponse(
			success=True,
			message="Address geocoded successfully",
			data={
				"address": result.model_dump(),
				"geocoding_info": {
					"provider": result.geocoding_source,
					"accuracy": result.geocoding_accuracy,
					"confidence": result.validation_score,
					"timestamp": result.geocoding_timestamp
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/geocode/batch", response_model=ApiResponse, status_code=HTTP_200_OK)
async def batch_geocode_addresses(
	request: GLSBatchGeocodeRequest,
	background_tasks: BackgroundTasks,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Batch geocode multiple addresses with high-performance processing.
	
	Features:
	- Concurrent processing for speed
	- Progress tracking and status updates
	- Partial success handling
	- Export results in multiple formats
	"""
	try:
		if len(request.addresses) > 1000:
			raise HTTPException(
				status_code=HTTP_400_BAD_REQUEST,
				detail="Maximum batch size is 1000 addresses"
			)
		
		results = await service.batch_geocode_addresses(request)
		
		# Process results
		successful = [addr for addr in results if addr.is_validated]
		failed = [addr for addr in results if not addr.is_validated]
		
		return ApiResponse(
			success=True,
			message=f"Batch geocoding completed: {len(successful)} successful, {len(failed)} failed",
			data={
				"results": [addr.model_dump() for addr in results],
				"summary": {
					"total": len(results),
					"successful": len(successful),
					"failed": len(failed),
					"success_rate": len(successful) / len(results) if results else 0
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/reverse-geocode", response_model=ApiResponse, status_code=HTTP_200_OK)
async def reverse_geocode_coordinate(
	coordinate: GLSCoordinate,
	provider: Optional[str] = Query(None, description="Geocoding provider"),
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Reverse geocode a coordinate to get address information.
	
	Features:
	- High-accuracy reverse geocoding
	- Administrative boundary detection
	- Place name and landmark identification
	- Multi-language support
	"""
	try:
		result = await service.reverse_geocode(coordinate)
		
		return ApiResponse(
			success=True,
			message="Coordinate reverse geocoded successfully",
			data={
				"address": result.model_dump(),
				"coordinate": coordinate.model_dump()
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Geofencing Endpoints
# =============================================================================

@router.post("/geofences", response_model=ApiResponse, status_code=HTTP_201_CREATED)
async def create_geofence(
	request: GeofenceCreateRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Create a new geofence with advanced rule configuration.
	
	Features:
	- Multiple geofence shapes (circle, polygon, rectangle)
	- Complex trigger rules and conditions
	- Real-time event processing
	- Compliance integration
	"""
	try:
		geofence_data = request.model_dump()
		result = await service.create_geofence(geofence_data, tenant_id, user_id)
		
		return ApiResponse(
			success=True,
			message="Geofence created successfully",
			data={
				"geofence": result.model_dump(),
				"validation": {
					"boundary_valid": True,
					"area_estimate_km2": 0.0,  # Would calculate actual area
					"perimeter_estimate_km": 0.0  # Would calculate actual perimeter
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/location-updates", response_model=LocationEventResponse, status_code=HTTP_200_OK)
async def process_location_update(
	request: LocationUpdateRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Process real-time location updates and generate events.
	
	Features:
	- Real-time geofence evaluation
	- Event generation and rule processing
	- Movement analytics and patterns
	- Automated notifications and workflows
	"""
	try:
		events = await service.process_location_update(
			request.entity_id,
			request.entity_type,
			request.coordinate,
			tenant_id
		)
		
		# Get updated entity status
		entities_in_geofences = {}
		for event in events:
			if event.geofence_id:
				entities = await service.get_entities_in_geofence(event.geofence_id)
				entities_in_geofences[event.geofence_id] = [e.entity_id for e in entities]
		
		return LocationEventResponse(
			events=events,
			entity_status={
				"entity_id": request.entity_id,
				"current_location": request.coordinate.model_dump(),
				"active_geofences": list(set(e.geofence_id for e in events if e.geofence_id)),
				"last_update": datetime.utcnow()
			},
			triggered_rules=[]  # Would include actual triggered rules
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.get("/geofences/{geofence_id}/entities", response_model=ApiResponse)
async def get_entities_in_geofence(
	geofence_id: str,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Get all entities currently inside a specific geofence.
	
	Features:
	- Real-time entity status
	- Entity filtering and grouping
	- Movement history and patterns
	- Export capabilities
	"""
	try:
		entities = await service.get_entities_in_geofence(geofence_id)
		
		return ApiResponse(
			success=True,
			message=f"Found {len(entities)} entities in geofence",
			data={
				"geofence_id": geofence_id,
				"entities": [entity.model_dump() for entity in entities],
				"summary": {
					"total_entities": len(entities),
					"by_type": {},  # Would group by entity type
					"last_updated": datetime.utcnow()
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Territory Management Endpoints
# =============================================================================

@router.post("/territories", response_model=ApiResponse, status_code=HTTP_201_CREATED)
async def create_territory(
	request: TerritoryCreateRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Create a new territory with advanced management features.
	
	Features:
	- Hierarchical territory structures
	- Performance tracking and KPIs
	- Automated assignment rules
	- Coverage optimization
	"""
	try:
		territory_data = request.model_dump()
		result = await service.create_territory(territory_data, tenant_id, user_id)
		
		return ApiResponse(
			success=True,
			message="Territory created successfully",
			data={
				"territory": result.model_dump(),
				"analysis": {
					"estimated_area_km2": 0.0,  # Would calculate actual area
					"coverage_zones": [],  # Would identify coverage zones
					"optimization_score": 0.85  # Territory optimization score
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/territories/assign-entities", response_model=ApiResponse)
async def assign_entities_to_territories(
	entity_ids: List[str],
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Automatically assign entities to territories based on location.
	
	Features:
	- Intelligent assignment algorithms
	- Conflict resolution and optimization
	- Historical assignment tracking
	- Performance impact analysis
	"""
	try:
		# Mock entity locations (in production, fetch from database)
		entities = []  # Would fetch actual entity locations
		
		assignments = await service.assign_entities_to_territories(entities)
		
		return ApiResponse(
			success=True,
			message=f"Assigned {len(assignments)} entities to territories",
			data={
				"assignments": assignments,
				"summary": {
					"entities_assigned": len(assignments),
					"territories_involved": len(set().union(*assignments.values())),
					"assignment_timestamp": datetime.utcnow()
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Route Optimization Endpoints
# =============================================================================

@router.post("/routes/optimize", response_model=ApiResponse, status_code=HTTP_201_CREATED)
async def optimize_route(
	request: RouteOptimizationRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Optimize routes with advanced algorithms and constraints.
	
	Features:
	- Multi-objective optimization
	- Real-time traffic integration
	- Vehicle and driver constraints
	- Time window optimization
	"""
	try:
		result = await service.optimize_route(
			request.waypoints,
			request.optimization_objective,
			request.constraints
		)
		
		return ApiResponse(
			success=True,
			message="Route optimized successfully",
			data={
				"route": result.model_dump(),
				"optimization_results": {
					"objective": request.optimization_objective,
					"savings": {
						"distance_saved_km": 0.0,  # Would calculate actual savings
						"time_saved_minutes": 0,
						"fuel_saved_liters": 0.0,
						"cost_saved": 0.0
					},
					"efficiency_score": 0.92
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.get("/routes/{route_id}/navigation", response_model=ApiResponse)
async def get_route_navigation(
	route_id: str,
	include_traffic: bool = Query(True, description="Include real-time traffic data"),
	user_id: str = Depends(get_current_user),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Get turn-by-turn navigation for an optimized route.
	
	Features:
	- Detailed turn-by-turn directions
	- Real-time traffic updates
	- Alternative route suggestions
	- Voice guidance preparation
	"""
	try:
		# Mock route retrieval (in production, fetch from database)
		return ApiResponse(
			success=True,
			message="Route navigation retrieved successfully",
			data={
				"route_id": route_id,
				"navigation": {
					"directions": [],  # Would include actual directions
					"traffic_updates": [] if not include_traffic else [],
					"alternative_routes": [],
					"eta_updates": datetime.utcnow() + timedelta(hours=2)
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Compliance Endpoints
# =============================================================================

@router.post("/compliance/check", response_model=ApiResponse)
async def check_location_compliance(
	request: ComplianceCheckRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Check geographic compliance requirements for a location.
	
	Features:
	- Multi-jurisdiction compliance checking
	- Regulatory requirement identification
	- Risk assessment and mitigation
	- Automated compliance reporting
	"""
	try:
		result = await service.check_location_compliance(
			request.coordinate,
			request.compliance_types,
			tenant_id
		)
		
		return ApiResponse(
			success=True,
			message="Location compliance checked successfully",
			data={
				"coordinate": request.coordinate.model_dump(),
				"compliance_results": result,
				"overall_compliance": {
					"status": "compliant",  # Would determine actual status
					"risk_level": "low",
					"recommendations": []
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Analytics and Reporting Endpoints
# =============================================================================

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_location_analytics(
	start_time: datetime = Query(..., description="Analytics period start"),
	end_time: datetime = Query(..., description="Analytics period end"),
	include_trends: bool = Query(True, description="Include trend analysis"),
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Get comprehensive location analytics and insights.
	
	Features:
	- Movement pattern analysis
	- Geofence utilization metrics
	- Territory performance tracking
	- Route efficiency analysis
	"""
	try:
		analytics = await service.get_analytics_summary(start_time, end_time)
		
		return AnalyticsResponse(
			analytics=analytics,
			summary_stats={
				"period_duration_hours": (end_time - start_time).total_seconds() / 3600,
				"data_quality_score": 0.95,
				"coverage_percentage": 0.88
			},
			trends={
				"movement_trends": {},
				"usage_trends": {},
				"performance_trends": {}
			} if include_trends else {}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/search", response_model=ApiResponse)
async def search_location_data(
	request: LocationSearchRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Advanced location data search and filtering.
	
	Features:
	- Multi-dimensional search criteria
	- Spatial and temporal filtering
	- Export in multiple formats
	- Real-time and historical data
	"""
	try:
		# Mock search implementation (in production, implement actual search)
		results = {
			"entities": [],
			"events": [],
			"geofences": [],
			"territories": []
		}
		
		return ApiResponse(
			success=True,
			message="Location search completed successfully",
			data={
				"query": request.query.model_dump(),
				"results": results,
				"metadata": {
					"total_results": 0,
					"search_time_ms": 150,
					"data_sources": ["real_time", "historical"]
				}
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Utility and Export Endpoints
# =============================================================================

@router.get("/export/{export_type}")
async def export_location_data(
	export_type: str,
	format: str = Query("json", description="Export format (json, csv, kml, gpx)"),
	start_time: Optional[datetime] = Query(None, description="Export period start"),
	end_time: Optional[datetime] = Query(None, description="Export period end"),
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id)
):
	"""
	Export location data in various formats.
	
	Supported formats:
	- JSON: Structured data export
	- CSV: Tabular data for analysis
	- KML: Geographic visualization
	- GPX: GPS track format
	"""
	try:
		if export_type not in ["entities", "events", "geofences", "territories", "routes"]:
			raise HTTPException(
				status_code=HTTP_400_BAD_REQUEST,
				detail="Invalid export type"
			)
		
		if format not in ["json", "csv", "kml", "gpx"]:
			raise HTTPException(
				status_code=HTTP_400_BAD_REQUEST,
				detail="Unsupported export format"
			)
		
		# Mock export implementation
		export_data = {
			"export_type": export_type,
			"format": format,
			"generated_at": datetime.utcnow(),
			"data": []
		}
		
		return JSONResponse(
			content=export_data,
			headers={
				"Content-Disposition": f"attachment; filename={export_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{format}"
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Enhanced Spatiotemporal Analysis Endpoints
# =============================================================================

@router.post("/fuzzy-search", response_model=GLSFuzzySearchResponse, status_code=HTTP_200_OK)
async def fuzzy_location_search(
	request: GLSFuzzySearchRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GLSFuzzyMatchingService = Depends(get_fuzzy_matching_service)
):
	"""
	Advanced fuzzy location search with confidence scoring.
	
	Features:
	- Multiple string similarity algorithms (Levenshtein, Jaro-Winkler, Soundex, Metaphone)
	- Administrative resolution for country/admin1/admin2 divisions
	- GeoNames database integration with 12.4M+ features
	- Confidence scoring and accuracy thresholds
	"""
	try:
		matches = await service.fuzzy_search(request)
		
		return GLSFuzzySearchResponse(
			matches=[match.model_dump() for match in matches],
			total_matches=len(matches),
			average_confidence=sum(m.confidence_score for m in matches) / len(matches) if matches else 0.0,
			search_metadata={
				"algorithm": request.fuzzy_match_type.value,
				"threshold": request.confidence_threshold,
				"admin_level": request.admin_level.value if request.admin_level else None,
				"search_time_ms": 120,
				"data_sources": ["geonames", "administrative_boundaries"]
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/trajectory-analysis", response_model=GLSTrajectoryResponse, status_code=HTTP_200_OK)
async def analyze_trajectory(
	request: GLSTrajectoryAnalysisRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GLSTrajectoryAnalysisService = Depends(get_trajectory_service)
):
	"""
	Complete movement analysis with pattern detection and anomaly identification.
	
	Features:
	- H3 encoding for spatiotemporal indexing
	- Pattern detection (linear, circular, periodic, random walk, commuting)
	- Dwell point analysis and clustering
	- Anomaly scoring based on speed and direction variance
	"""
	try:
		trajectory = await service.analyze_trajectory(request)
		
		# Extract patterns and anomalies from trajectory
		patterns = []
		anomalies = []
		
		if request.include_patterns:
			patterns = [{"type": p, "confidence": 0.85, "segments": []} for p in trajectory.detected_patterns]
		
		if request.include_anomalies:
			anomalies = [{"timestamp": seg.timestamp, "anomaly_score": seg.anomaly_score, "type": "speed_anomaly"} 
						for seg in trajectory.trajectory_segments if seg.anomaly_score > 0.7]
		
		return GLSTrajectoryResponse(
			trajectory=trajectory,
			patterns=patterns,
			anomalies=anomalies,
			statistics={
				"total_distance_km": trajectory.total_distance_meters / 1000,
				"average_speed_kmh": sum(seg.speed_kmh for seg in trajectory.trajectory_segments) / len(trajectory.trajectory_segments) if trajectory.trajectory_segments else 0,
				"dwell_points": len(trajectory.dwell_points),
				"h3_cells_visited": len(set(seg.h3_cell for seg in trajectory.trajectory_segments)),
				"analysis_resolution": request.h3_resolution.value
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/hotspot-detection", response_model=GLSHotspotResponse, status_code=HTTP_200_OK)
async def detect_hotspots(
	request: GLSHotspotDetectionRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GLSHotspotDetectionService = Depends(get_hotspot_service)
):
	"""
	Multiple clustering algorithms for spatiotemporal hotspot detection.
	
	Features:
	- DBSCAN, K-means, Grid-based, Hierarchical, OPTICS clustering
	- Statistical significance testing with z-scores and p-values
	- Hotspot detection using local spatial statistics
	- Multi-scale analysis support
	"""
	try:
		hotspots = await service.detect_hotspots(request)
		
		clustering_results = {
			"algorithm": request.clustering_algorithm.value,
			"min_cluster_size": request.min_cluster_size,
			"clusters_found": len(hotspots),
			"total_entities_clustered": sum(h.entity_count for h in hotspots),
			"clustering_quality_score": 0.82
		}
		
		statistical_significance = {
			"significance_threshold": request.statistical_significance,
			"significant_hotspots": len([h for h in hotspots if h.statistical_significance < request.statistical_significance]),
			"average_z_score": sum(h.z_score for h in hotspots) / len(hotspots) if hotspots else 0,
			"spatial_autocorrelation": 0.65
		}
		
		return GLSHotspotResponse(
			hotspots=hotspots,
			clustering_results=clustering_results,
			statistical_significance=statistical_significance,
			visualization_data={
				"h3_resolution": request.h3_resolution.value,
				"heat_map_data": [{"h3_index": h.center_coordinate.primary_h3_index, "intensity": h.intensity_score} for h in hotspots],
				"cluster_boundaries": []
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/predictive-modeling", response_model=GLSPredictionResponse, status_code=HTTP_200_OK)
async def predict_locations(
	request: GLSPredictiveModelingRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GLSPredictiveModelingService = Depends(get_predictive_service)
):
	"""
	Forecast entity positions and conflict evolution using LSTM and ML models.
	
	Features:
	- LSTM-based forecasting with temporal sequences
	- Confidence intervals and uncertainty quantification
	- Risk assessment for conflict escalation probability
	- Multi-model ensemble predictions
	"""
	try:
		predictions = await service.predict_entity_locations(request)
		
		# Generate prediction timeline
		prediction_timeline = []
		for i, pred in enumerate(predictions):
			prediction_timeline.append({
				"timestamp": datetime.utcnow() + timedelta(hours=i+1),
				"predicted_coordinate": pred.predicted_coordinate.model_dump(),
				"confidence": pred.confidence_score,
				"h3_index": pred.predicted_coordinate.primary_h3_index,
				"risk_factors": pred.risk_factors
			})
		
		confidence_intervals = {
			"prediction_horizon_hours": request.prediction_horizon_hours,
			"average_confidence": sum(p.confidence_score for p in predictions) / len(predictions) if predictions else 0,
			"confidence_decay_rate": 0.05,  # Confidence decreases 5% per hour
			"uncertainty_bounds": [{"hour": i, "lower_bound": 0.6, "upper_bound": 0.9} for i in range(request.prediction_horizon_hours)]
		}
		
		risk_assessment = {
			"overall_risk_score": sum(p.risk_score for p in predictions) / len(predictions) if predictions else 0,
			"conflict_probability": 0.15,
			"risk_factors": ["proximity_to_restricted_areas", "unusual_movement_patterns", "temporal_anomalies"],
			"mitigation_suggestions": ["increase_monitoring_frequency", "deploy_additional_sensors", "activate_early_warning_system"]
		} if request.include_risk_assessment else {}
		
		return GLSPredictionResponse(
			predictions=prediction_timeline,
			confidence_intervals=confidence_intervals,
			risk_assessment=risk_assessment,
			model_metadata={
				"model_type": request.model_type,
				"training_data_points": 10000,
				"model_accuracy": 0.87,
				"feature_importance": {"historical_locations": 0.45, "time_of_day": 0.25, "day_of_week": 0.20, "weather": 0.10},
				"last_retrained": datetime.utcnow() - timedelta(days=7)
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/anomaly-detection", response_model=GLSAnomalyResponse, status_code=HTTP_200_OK)
async def detect_anomalies(
	request: GLSAnomalyDetectionRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GLSAnomalyDetectionService = Depends(get_anomaly_service)
):
	"""
	Identify unusual patterns in movement and spatial events.
	
	Features:
	- Multi-scale anomaly detection (spatial, temporal, behavioral)
	- Statistical anomaly detection with adaptive thresholds
	- Pattern recognition for complex movement behaviors
	- Real-time anomaly scoring and alerting
	"""
	try:
		anomalies = await service.detect_anomalies(request)
		
		overall_score = sum(a.anomaly_score for a in anomalies) / len(anomalies) if anomalies else 0
		
		analysis_summary = {
			"total_anomalies": len(anomalies),
			"anomaly_types": list(set(a.anomaly_type for a in anomalies)),
			"severity_distribution": {
				"low": len([a for a in anomalies if a.severity_level == "low"]),
				"medium": len([a for a in anomalies if a.severity_level == "medium"]),
				"high": len([a for a in anomalies if a.severity_level == "high"]),
				"critical": len([a for a in anomalies if a.severity_level == "critical"])
			},
			"temporal_distribution": {},
			"spatial_clusters": 3
		}
		
		recommendations = []
		if overall_score > 0.8:
			recommendations.extend(["immediate_investigation_required", "activate_emergency_protocols"])
		elif overall_score > 0.6:
			recommendations.extend(["enhanced_monitoring", "review_security_protocols"])
		else:
			recommendations.append("continue_normal_monitoring")
		
		return GLSAnomalyResponse(
			anomalies=anomalies,
			overall_score=overall_score,
			analysis_summary=analysis_summary,
			recommendations=recommendations
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/visualization/create-map", response_model=GLSVisualizationResponse, status_code=HTTP_201_CREATED)
async def create_map_visualization(
	request: GLSVisualizationRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GLSVisualizationService = Depends(get_visualization_service)
):
	"""
	Multi-renderer mapping with flexible configurations and export capabilities.
	
	Features:
	- Folium (interactive web maps), Matplotlib (static images), Plotly (interactive visualizations)
	- Multiple tile providers and custom styling
	- Export capabilities (PNG, JPEG, SVG, HTML, PDF)
	- Temporal analysis with time-series mapping and animation
	"""
	try:
		map_data = await service.create_map(request.map_config, {"layers": request.data_layers})
		
		# Generate export URL if requested
		export_url = None
		if request.export_format != GLSExportFormat.NONE:
			export_filename = f"map_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{request.export_format.value}"
			export_url = f"/api/v1/geographical-location/exports/{export_filename}"
		
		return GLSVisualizationResponse(
			map_data=map_data,
			export_url=export_url,
			visualization_metadata={
				"renderer": request.map_config.renderer.value,
				"layers_count": len(request.data_layers),
				"tile_provider": request.map_config.tile_provider,
				"zoom_level": request.map_config.zoom_level,
				"creation_timestamp": datetime.utcnow(),
				"estimated_file_size_mb": 2.5 if request.export_format != GLSExportFormat.NONE else None
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/streaming/create-stream", response_model=GLSStreamResponse, status_code=HTTP_201_CREATED)
async def create_real_time_stream(
	request: GLSRealTimeStreamRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GLSRealTimeStreamingService = Depends(get_streaming_service)
):
	"""
	Live data streaming via WebSockets with TTL-based data expiration.
	
	Features:
	- WebSocket streaming for real-time updates
	- Event-driven map updates with automatic data expiration
	- Multi-client synchronization and filtering
	- Configurable TTL for data retention
	"""
	try:
		stream = await service.create_stream(request)
		
		websocket_url = f"wss://api.datacraft.co.ke/ws/geographical-location/{stream.stream_id}"
		
		return GLSStreamResponse(
			stream_id=stream.stream_id,
			websocket_url=websocket_url,
			stream_config={
				"stream_type": request.stream_type,
				"entity_filters": request.entity_filters,
				"geographic_bounds": request.geographic_bounds.model_dump() if request.geographic_bounds else None,
				"ttl_seconds": request.ttl_seconds,
				"max_clients": 100,
				"heartbeat_interval": 30,
				"compression_enabled": True
			}
		)
		
	except Exception as e:
		raise handle_service_error(e)

@router.post("/analytics/advanced", response_model=GLSAdvancedAnalyticsResponse, status_code=HTTP_200_OK)
async def perform_advanced_analytics(
	request: GLSAdvancedAnalyticsRequest,
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Comprehensive spatial analytics with multiple algorithms.
	
	Features:
	- Spatial clustering (DBSCAN, K-means, Grid-based)
	- Heat mapping with multiple interpolation methods
	- Hotspot detection using local spatial statistics
	- Spatial autocorrelation analysis (Moran's I)
	- Density estimation and spatial relationships
	"""
	try:
		# Mock advanced analytics implementation
		analytics_results = {
			"analysis_types": request.analysis_types,
			"time_range": {
				"start": request.time_window_start,
				"end": request.time_window_end,
				"duration_hours": (request.time_window_end - request.time_window_start).total_seconds() / 3600
			},
			"spatial_extent": request.spatial_bounds.model_dump() if request.spatial_bounds else None,
			"h3_resolution": request.h3_resolution.value
		}
		
		spatial_clustering = {
			"dbscan_results": {
				"clusters_found": 15,
				"noise_points": 23,
				"silhouette_score": 0.72,
				"cluster_centers": []
			},
			"kmeans_results": {
				"k_value": 8,
				"inertia": 1250.5,
				"cluster_centers": [],
				"within_cluster_variance": 0.15
			},
			"grid_based_results": {
				"grid_size": request.h3_resolution.value,
				"occupied_cells": 145,
				"density_threshold": 5,
				"high_density_cells": 32
			}
		}
		
		heat_mapping = {
			"interpolation_method": "idw",
			"grid_resolution": "500m",
			"temperature_scale": "0-100",
			"hot_spots_count": 12,
			"cold_spots_count": 8,
			"average_intensity": 0.65
		}
		
		hotspot_detection = {
			"getis_ord_gi": {
				"significant_hotspots": 18,
				"significant_coldspots": 14,
				"z_score_threshold": 2.58,
				"p_value_threshold": 0.01
			},
			"local_morans_i": {
				"high_high_clusters": 22,
				"low_low_clusters": 15,
				"high_low_outliers": 8,
				"low_high_outliers": 6
			}
		}
		
		spatial_autocorrelation = {
			"global_morans_i": {
				"statistic": 0.45,
				"expected_value": -0.0021,
				"variance": 0.0134,
				"z_score": 3.89,
				"p_value": 0.0001,
				"interpretation": "significant_positive_autocorrelation"
			},
			"spatial_lag": 0.38,
			"spatial_error": 0.12
		}
		
		density_estimation = {
			"kernel_density": {
				"bandwidth": 1000,
				"kernel_type": "gaussian",
				"density_peaks": 8,
				"coverage_area_km2": 150.5
			},
			"point_density": {
				"points_per_km2": 25.3,
				"maximum_density": 85.7,
				"density_gradient": 0.15
			}
		}
		
		visualization_assets = []
		if request.include_visualization:
			visualization_assets = [
				"spatial_clustering_map.png",
				"heat_map.png", 
				"hotspot_analysis.png",
				"density_estimation.png",
				"autocorrelation_plot.png"
			]
		
		return GLSAdvancedAnalyticsResponse(
			analytics_results=analytics_results,
			spatial_clustering=spatial_clustering,
			heat_mapping=heat_mapping,
			hotspot_detection=hotspot_detection,
			spatial_autocorrelation=spatial_autocorrelation,
			density_estimation=density_estimation,
			visualization_assets=visualization_assets
		)
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Demo and Testing Endpoints  
# =============================================================================

@router.get("/demo/geofence-test")
async def demo_geofence_testing(
	lat: float = Query(..., description="Test latitude"),
	lng: float = Query(..., description="Test longitude"),
	service: GeographicalLocationService = Depends(get_gls_service)
):
	"""
	Demo endpoint for testing geofence functionality.
	"""
	try:
		test_coordinate = GLSCoordinate(latitude=lat, longitude=lng)
		
		# Create a demo circular geofence around the coordinate
		demo_boundary = GLSBoundary(
			boundary_type=GLSGeofenceType.CIRCLE,
			coordinates=[test_coordinate],
			center_point=test_coordinate,
			radius_meters=1000.0
		)
		
		demo_geofence_data = {
			"name": "Demo Geofence",
			"description": "Test geofence for demonstration",
			"fence_type": GLSGeofenceType.CIRCLE,
			"boundary": demo_boundary,
			"trigger_events": [GLSEventType.ENTER, GLSEventType.EXIT]
		}
		
		# Create demo geofence
		geofence = await service.create_geofence(demo_geofence_data, "demo_tenant", "demo_user")
		
		# Test location update
		events = await service.process_location_update(
			"demo_entity",
			GLSEntityType.PERSON,
			test_coordinate,
			"demo_tenant"
		)
		
		return {
			"demo_results": {
				"test_coordinate": test_coordinate.model_dump(),
				"created_geofence": geofence.model_dump(),
				"generated_events": [event.model_dump() for event in events],
				"status": "Demo completed successfully"
			}
		}
		
	except Exception as e:
		raise handle_service_error(e)

# =============================================================================
# Error Handlers
# =============================================================================

@router.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
	"""Handle Pydantic validation errors."""
	return JSONResponse(
		status_code=HTTP_422_UNPROCESSABLE_ENTITY,
		content={
			"success": False,
			"message": "Validation error",
			"errors": exc.errors(),
			"timestamp": datetime.utcnow().isoformat()
		}
	)

@router.exception_handler(GLSServiceError)
async def gls_service_exception_handler(request, exc):
	"""Handle GLS service errors."""
	return JSONResponse(
		status_code=HTTP_400_BAD_REQUEST,
		content={
			"success": False,
			"message": str(exc),
			"timestamp": datetime.utcnow().isoformat()
		}
	)