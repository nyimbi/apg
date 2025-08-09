# Geographical Location Services - API Reference

**Version:** 2.0.0  
**Base URL:** `https://api.datacraft.co.ke/api/v1/geographical-location`  
**Authentication:** Bearer Token Required  

## 📋 Table of Contents

1. [Authentication](#authentication)
2. [Error Handling](#error-handling)
3. [Rate Limiting](#rate-limiting)
4. [Health & Status Endpoints](#health--status-endpoints)
5. [Enhanced Spatiotemporal Endpoints](#enhanced-spatiotemporal-endpoints)
6. [Traditional Location Endpoints](#traditional-location-endpoints)
7. [Real-Time Streaming](#real-time-streaming)
8. [Export & Utility Endpoints](#export--utility-endpoints)
9. [WebSocket API](#websocket-api)
10. [SDK Examples](#sdk-examples)

---

## 🔐 Authentication

All API endpoints require JWT Bearer token authentication:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

### Token Requirements
- **Scopes Required**: `gls:read`, `gls:write`, `gls:stream` (depending on endpoint)
- **Token Expiry**: 24 hours
- **Refresh**: Automatic refresh available

---

## ⚠️ Error Handling

### Standard Error Response Format

```json
{
    "success": false,
    "message": "Error description",
    "errors": ["Detailed error messages"],
    "timestamp": "2025-01-28T10:30:00Z",
    "request_id": "req_123456789"
}
```

### HTTP Status Codes

| Code | Description | When Used |
|------|-------------|-----------|
| 200 | OK | Successful GET/POST operations |
| 201 | Created | Resource created successfully |
| 204 | No Content | Successful DELETE operations |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side errors |

---

## ⏱️ Rate Limiting

### Rate Limits by Endpoint Type

| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Health/Status | 1000 req/min | Per IP |
| Geocoding | 500 req/min | Per API key |
| Fuzzy Search | 200 req/min | Per API key |
| Trajectory Analysis | 100 req/min | Per API key |
| Predictive Modeling | 50 req/min | Per API key |
| Real-time Streaming | 10 connections | Per API key |

### Rate Limit Headers

```http
X-RateLimit-Limit: 500
X-RateLimit-Remaining: 487
X-RateLimit-Reset: 1643723400
Retry-After: 60
```

---

## 🏥 Health & Status Endpoints

### GET /health

Check service health status.

#### Response
```json
{
    "status": "healthy",
    "version": "2.0.0",
    "services": {
        "database": "connected",
        "cache": "connected",
        "h3_indexer": "operational",
        "ml_models": "loaded"
    },
    "performance": {
        "avg_response_time_ms": 45,
        "active_connections": 1250,
        "cache_hit_rate": 0.87
    },
    "timestamp": "2025-01-28T10:30:00Z"
}
```

### GET /capabilities

Get detailed service capabilities.

#### Response
```json
{
    "capabilities": {
        "h3_encoding": {
            "description": "H3 hierarchical spatial indexing with 11 resolution levels",
            "features": ["Multi-resolution indexing", "Spatial queries", "Grid analytics"],
            "resolutions": ["continent", "country", "state", "metro", "city", "district", "neighborhood", "block", "building", "room", "ultra_precise"],
            "max_resolution_level": 10,
            "spatial_indexing": true
        },
        "fuzzy_matching": {
            "description": "Advanced string similarity algorithms with confidence scoring",
            "algorithms": ["levenshtein", "jaro_winkler", "soundex", "metaphone", "fuzzy_wuzzy"],
            "geonames_features": "12.4M+ geographical features",
            "confidence_threshold": 0.5
        }
    },
    "version": "2.0.0",
    "api_version": "v1",
    "status": "active"
}
```

---

## 🚀 Enhanced Spatiotemporal Endpoints

### POST /fuzzy-search

Advanced fuzzy location search with confidence scoring.

#### Request Body
```json
{
    "query_text": "New York Cty",
    "fuzzy_match_type": "jaro_winkler",
    "confidence_threshold": 0.7,
    "admin_level": "city",
    "country_filter": "US",
    "max_results": 10
}
```

#### Parameters
- `query_text` (string, required): Location text to search
- `fuzzy_match_type` (enum, required): Algorithm - `levenshtein`, `jaro_winkler`, `soundex`, `metaphone`, `fuzzy_wuzzy`
- `confidence_threshold` (float, 0-1): Minimum confidence score
- `admin_level` (enum, optional): Administrative level filter - `country`, `admin1`, `admin2`, `locality`
- `country_filter` (string, optional): ISO country code filter
- `max_results` (int, 1-100): Maximum results to return

#### Response
```json
{
    "matches": [
        {
            "name": "New York City",
            "coordinate": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "h3_indices": {
                    "4": "84186b9fdbfffff",
                    "5": "85186b9fdafffff"
                },
                "primary_h3_index": "84186b9fdbfffff"
            },
            "confidence_score": 0.95,
            "admin_hierarchy": {
                "country": "United States",
                "admin1": "New York",
                "admin2": "New York County",
                "locality": "New York City"
            },
            "data_source": "geonames",
            "feature_code": "PPL"
        }
    ],
    "total_matches": 1,
    "average_confidence": 0.95,
    "search_metadata": {
        "algorithm": "jaro_winkler",
        "threshold": 0.7,
        "admin_level": "city",
        "search_time_ms": 120,
        "data_sources": ["geonames", "administrative_boundaries"]
    }
}
```

### POST /trajectory-analysis

Complete movement analysis with pattern detection and anomaly identification.

#### Request Body
```json
{
    "entity_id": "vehicle_123",
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-01T23:59:59Z",
    "include_patterns": true,
    "include_anomalies": true,
    "h3_resolution": "city"
}
```

#### Parameters
- `entity_id` (string, required): Entity identifier to analyze
- `time_window_start` (datetime, required): Analysis start time
- `time_window_end` (datetime, required): Analysis end time
- `include_patterns` (boolean): Include pattern detection results
- `include_anomalies` (boolean): Include anomaly detection results
- `h3_resolution` (enum): H3 analysis resolution level

#### Response
```json
{
    "trajectory": {
        "entity_id": "vehicle_123",
        "start_time": "2025-01-01T00:00:00Z",
        "end_time": "2025-01-01T23:59:59Z",
        "trajectory_segments": [
            {
                "timestamp": "2025-01-01T08:00:00Z",
                "coordinate": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "h3_indices": {"4": "84186b9fdbfffff"},
                    "primary_h3_index": "84186b9fdbfffff"
                },
                "speed_kmh": 35.5,
                "bearing_degrees": 45.2,
                "h3_cell": "84186b9fdbfffff",
                "dwell_time_minutes": 5,
                "anomaly_score": 0.15
            }
        ],
        "detected_patterns": ["commuting", "periodic"],
        "total_distance_meters": 45280.5,
        "dwell_points": [
            {
                "coordinate": {
                    "latitude": 40.7589,
                    "longitude": -73.9851
                },
                "arrival_time": "2025-01-01T09:00:00Z",
                "departure_time": "2025-01-01T17:30:00Z",
                "duration_minutes": 510,
                "visit_frequency": 0.8,
                "poi_type": "office"
            }
        ]
    },
    "patterns": [
        {
            "type": "commuting",
            "confidence": 0.85,
            "segments": [],
            "description": "Regular home-to-office pattern detected"
        }
    ],
    "anomalies": [
        {
            "timestamp": "2025-01-01T14:30:00Z",
            "anomaly_score": 0.75,
            "type": "speed_anomaly",
            "description": "Unusual speed detected: 120 km/h in urban area"
        }
    ],
    "statistics": {
        "total_distance_km": 45.28,
        "average_speed_kmh": 28.5,
        "dwell_points": 3,
        "h3_cells_visited": 145,
        "analysis_resolution": "city"
    }
}
```

### POST /hotspot-detection

Multiple clustering algorithms for spatiotemporal hotspot detection.

#### Request Body
```json
{
    "entity_type": "person",
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-07T23:59:59Z",
    "clustering_algorithm": "dbscan",
    "min_cluster_size": 5,
    "statistical_significance": 0.05,
    "h3_resolution": "neighborhood"
}
```

#### Parameters
- `entity_type` (enum, required): Entity type to analyze - `person`, `vehicle`, `asset`, `device`
- `time_window_start` (datetime, required): Analysis start time
- `time_window_end` (datetime, required): Analysis end time
- `clustering_algorithm` (enum, required): Algorithm - `dbscan`, `kmeans`, `grid_based`, `hierarchical`, `optics`
- `min_cluster_size` (int, ≥2): Minimum entities per cluster
- `statistical_significance` (float, 0.001-0.1): P-value threshold
- `h3_resolution` (enum): H3 analysis resolution

#### Response
```json
{
    "hotspots": [
        {
            "hotspot_id": "hs_001",
            "center_coordinate": {
                "latitude": 40.7589,
                "longitude": -73.9851,
                "primary_h3_index": "86186b9fdbfffff"
            },
            "entity_count": 25,
            "intensity_score": 0.85,
            "z_score": 3.45,
            "statistical_significance": 0.0006,
            "time_window_start": "2025-01-01T00:00:00Z",
            "time_window_end": "2025-01-07T23:59:59Z",
            "h3_resolution": "neighborhood",
            "cluster_radius_meters": 150.5
        }
    ],
    "clustering_results": {
        "algorithm": "dbscan",
        "min_cluster_size": 5,
        "clusters_found": 8,
        "total_entities_clustered": 156,
        "clustering_quality_score": 0.82
    },
    "statistical_significance": {
        "significance_threshold": 0.05,
        "significant_hotspots": 6,
        "average_z_score": 2.87,
        "spatial_autocorrelation": 0.65
    },
    "visualization_data": {
        "h3_resolution": "neighborhood",
        "heat_map_data": [
            {
                "h3_index": "86186b9fdbfffff",
                "intensity": 0.85
            }
        ],
        "cluster_boundaries": []
    }
}
```

### POST /predictive-modeling

Forecast entity positions and conflict evolution using LSTM and ML models.

#### Request Body
```json
{
    "entity_id": "asset_456",
    "prediction_horizon_hours": 24,
    "model_type": "lstm",
    "confidence_intervals": true,
    "include_risk_assessment": true
}
```

#### Parameters
- `entity_id` (string, required): Entity to predict for
- `prediction_horizon_hours` (int, 1-168): Prediction time horizon
- `model_type` (string): Model type - `lstm`, `arima`, `random_forest`, `ensemble`
- `confidence_intervals` (boolean): Include confidence bounds
- `include_risk_assessment` (boolean): Include risk analysis

#### Response
```json
{
    "predictions": [
        {
            "timestamp": "2025-01-29T11:30:00Z",
            "predicted_coordinate": {
                "latitude": 40.7500,
                "longitude": -73.9900,
                "primary_h3_index": "84186b9fdbfffff"
            },
            "confidence": 0.87,
            "h3_index": "84186b9fdbfffff",
            "risk_factors": ["proximity_to_restricted_area"]
        }
    ],
    "confidence_intervals": {
        "prediction_horizon_hours": 24,
        "average_confidence": 0.82,
        "confidence_decay_rate": 0.05,
        "uncertainty_bounds": [
            {
                "hour": 1,
                "lower_bound": 0.85,
                "upper_bound": 0.95
            }
        ]
    },
    "risk_assessment": {
        "overall_risk_score": 0.25,
        "conflict_probability": 0.15,
        "risk_factors": ["proximity_to_restricted_areas", "unusual_movement_patterns"],
        "mitigation_suggestions": ["increase_monitoring_frequency", "deploy_additional_sensors"]
    },
    "model_metadata": {
        "model_type": "lstm",
        "training_data_points": 10000,
        "model_accuracy": 0.87,
        "feature_importance": {
            "historical_locations": 0.45,
            "time_of_day": 0.25,
            "day_of_week": 0.20,
            "weather": 0.10
        },
        "last_retrained": "2025-01-21T10:30:00Z"
    }
}
```

### POST /anomaly-detection

Identify unusual patterns in movement and spatial events.

#### Request Body
```json
{
    "entity_id": "user_789",
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-07T23:59:59Z",
    "anomaly_types": ["spatial", "temporal", "behavioral"],
    "sensitivity": 0.95
}
```

#### Parameters
- `entity_id` (string, required): Entity to analyze
- `time_window_start` (datetime, required): Analysis start time
- `time_window_end` (datetime, required): Analysis end time
- `anomaly_types` (array, required): Types to detect - `spatial`, `temporal`, `behavioral`, `statistical`
- `sensitivity` (float, 0.5-0.99): Detection sensitivity

#### Response
```json
{
    "anomalies": [
        {
            "anomaly_id": "anom_001",
            "entity_id": "user_789",
            "anomaly_type": "spatial",
            "timestamp": "2025-01-03T15:30:00Z",
            "coordinate": {
                "latitude": 40.6892,
                "longitude": -74.0445
            },
            "anomaly_score": 0.87,
            "severity_level": "high",
            "description": "Location significantly outside normal movement pattern",
            "context": {
                "normal_area_radius_km": 5.2,
                "deviation_distance_km": 15.8,
                "historical_visits_to_area": 0
            }
        }
    ],
    "overall_score": 0.72,
    "analysis_summary": {
        "total_anomalies": 4,
        "anomaly_types": ["spatial", "temporal"],
        "severity_distribution": {
            "low": 1,
            "medium": 2,
            "high": 1,
            "critical": 0
        },
        "temporal_distribution": {},
        "spatial_clusters": 3
    },
    "recommendations": ["enhanced_monitoring", "review_security_protocols"]
}
```

### POST /visualization/create-map

Multi-renderer mapping with flexible configurations and export capabilities.

#### Request Body
```json
{
    "map_config": {
        "renderer": "folium",
        "center_coordinate": {
            "latitude": 40.7128,
            "longitude": -74.0060
        },
        "zoom_level": 12,
        "tile_provider": "openstreetmap",
        "width": 800,
        "height": 600,
        "style_config": {
            "theme": "light",
            "color_scheme": "default"
        }
    },
    "data_layers": ["trajectories", "hotspots", "geofences"],
    "time_range": {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-07T23:59:59Z"
    },
    "export_format": "png"
}
```

#### Response
```json
{
    "map_data": {
        "map_html": "<div id='map'>...</div>",
        "javascript": "var map = L.map('map')...",
        "css": ".leaflet-container { ... }",
        "layers": [
            {
                "layer_id": "trajectories",
                "type": "polyline",
                "data_points": 1250,
                "style": {"color": "#ff0000", "weight": 3}
            }
        ]
    },
    "export_url": "/api/v1/geographical-location/exports/map_20250128_103000.png",
    "visualization_metadata": {
        "renderer": "folium",
        "layers_count": 3,
        "tile_provider": "openstreetmap",
        "zoom_level": 12,
        "creation_timestamp": "2025-01-28T10:30:00Z",
        "estimated_file_size_mb": 2.5
    }
}
```

### POST /streaming/create-stream

Live data streaming via WebSockets with TTL-based data expiration.

#### Request Body
```json
{
    "stream_type": "location_updates",
    "entity_filters": {
        "entity_type": "vehicle",
        "territory_id": "sales_north"
    },
    "geographic_bounds": {
        "boundary_type": "rectangle",
        "coordinates": [
            {"latitude": 40.7000, "longitude": -74.0200},
            {"latitude": 40.7500, "longitude": -73.9800}
        ]
    },
    "ttl_seconds": 3600
}
```

#### Response
```json
{
    "stream_id": "stream_abc123",
    "websocket_url": "wss://api.datacraft.co.ke/ws/geographical-location/stream_abc123",
    "stream_config": {
        "stream_type": "location_updates",
        "entity_filters": {
            "entity_type": "vehicle",
            "territory_id": "sales_north"
        },
        "geographic_bounds": {
            "boundary_type": "rectangle",
            "coordinates": [
                {"latitude": 40.7000, "longitude": -74.0200},
                {"latitude": 40.7500, "longitude": -73.9800}
            ]
        },
        "ttl_seconds": 3600,
        "max_clients": 100,
        "heartbeat_interval": 30,
        "compression_enabled": true
    }
}
```

### POST /analytics/advanced

Comprehensive spatial analytics with multiple algorithms.

#### Request Body
```json
{
    "analysis_types": ["spatial_clustering", "heat_mapping", "hotspot_detection", "spatial_autocorrelation"],
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-07T23:59:59Z",
    "spatial_bounds": {
        "boundary_type": "circle",
        "center_point": {
            "latitude": 40.7128,
            "longitude": -74.0060
        },
        "radius_meters": 5000
    },
    "h3_resolution": "city",
    "include_visualization": true
}
```

#### Response
```json
{
    "analytics_results": {
        "analysis_types": ["spatial_clustering", "heat_mapping", "hotspot_detection", "spatial_autocorrelation"],
        "time_range": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-07T23:59:59Z",
            "duration_hours": 168
        },
        "h3_resolution": "city"
    },
    "spatial_clustering": {
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
        }
    },
    "heat_mapping": {
        "interpolation_method": "idw",
        "grid_resolution": "500m",
        "temperature_scale": "0-100",
        "hot_spots_count": 12,
        "cold_spots_count": 8,
        "average_intensity": 0.65
    },
    "hotspot_detection": {
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
    },
    "spatial_autocorrelation": {
        "global_morans_i": {
            "statistic": 0.45,
            "expected_value": -0.0021,
            "variance": 0.0134,
            "z_score": 3.89,
            "p_value": 0.0001,
            "interpretation": "significant_positive_autocorrelation"
        }
    },
    "density_estimation": {
        "kernel_density": {
            "bandwidth": 1000,
            "kernel_type": "gaussian",
            "density_peaks": 8,
            "coverage_area_km2": 150.5
        }
    },
    "visualization_assets": [
        "spatial_clustering_map.png",
        "heat_map.png",
        "hotspot_analysis.png",
        "density_estimation.png",
        "autocorrelation_plot.png"
    ]
}
```

---

## 🗺️ Traditional Location Endpoints

### POST /geocode

Single address geocoding with validation and enrichment.

#### Request Body
```json
{
    "address": {
        "street": "350 5th Ave",
        "city": "New York",
        "state": "NY",
        "postal_code": "10118",
        "country": "US"
    },
    "provider": "google",
    "return_details": true
}
```

#### Response
```json
{
    "success": true,
    "message": "Address geocoded successfully",
    "data": {
        "address": {
            "street": "350 5th Ave",
            "city": "New York",
            "state": "NY",
            "postal_code": "10118",
            "country": "US",
            "coordinate": {
                "latitude": 40.7484,
                "longitude": -73.9857,
                "h3_indices": {
                    "4": "84186b9fdbfffff",
                    "5": "85186b9fdbfffff"
                },
                "primary_h3_index": "84186b9fdbfffff"
            },
            "is_validated": true,
            "validation_score": 0.98,
            "geocoding_source": "google",
            "geocoding_accuracy": "high",
            "geocoding_timestamp": "2025-01-28T10:30:00Z"
        },
        "geocoding_info": {
            "provider": "google",
            "accuracy": "high",
            "confidence": 0.98,
            "timestamp": "2025-01-28T10:30:00Z"
        }
    }
}
```

### POST /geocode/batch

Batch geocode multiple addresses with high-performance processing.

#### Request Body
```json
{
    "addresses": [
        {
            "id": "addr_001",
            "street": "350 5th Ave",
            "city": "New York",
            "state": "NY",
            "country": "US"
        },
        {
            "id": "addr_002",
            "street": "1600 Pennsylvania Ave",
            "city": "Washington",
            "state": "DC",
            "country": "US"
        }
    ],
    "provider": "google",
    "parallel_processing": true,
    "export_format": "json"
}
```

#### Response
```json
{
    "success": true,
    "message": "Batch geocoding completed: 2 successful, 0 failed",
    "data": {
        "results": [
            {
                "id": "addr_001",
                "address": {
                    "street": "350 5th Ave",
                    "city": "New York",
                    "state": "NY",
                    "country": "US",
                    "coordinate": {
                        "latitude": 40.7484,
                        "longitude": -73.9857,
                        "primary_h3_index": "84186b9fdbfffff"
                    },
                    "is_validated": true,
                    "validation_score": 0.98
                }
            }
        ],
        "summary": {
            "total": 2,
            "successful": 2,
            "failed": 0,
            "success_rate": 1.0
        }
    }
}
```

---

## 🌐 WebSocket API

### Connection

Connect to the WebSocket endpoint returned from the `/streaming/create-stream` endpoint:

```javascript
const ws = new WebSocket('wss://api.datacraft.co.ke/ws/geographical-location/stream_abc123');
```

### Authentication

Send authentication message immediately after connection:

```javascript
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your_jwt_token'
    }));
};
```

### Message Types

#### Location Update
```json
{
    "type": "location_update",
    "timestamp": "2025-01-28T10:30:00Z",
    "entity_id": "vehicle_123",
    "entity_type": "vehicle",
    "coordinate": {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "primary_h3_index": "84186b9fdbfffff"
    },
    "metadata": {
        "speed_kmh": 45.5,
        "bearing_degrees": 135.2,
        "accuracy_meters": 5.0
    }
}
```

#### Geofence Event
```json
{
    "type": "geofence_event",
    "timestamp": "2025-01-28T10:30:00Z",
    "entity_id": "vehicle_123",
    "geofence_id": "gf_sales_territory",
    "event_type": "enter",
    "coordinate": {
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    "metadata": {
        "dwell_time_seconds": 0,
        "previous_location": {
            "latitude": 40.7100,
            "longitude": -74.0080
        }
    }
}
```

#### Anomaly Alert
```json
{
    "type": "anomaly_alert",
    "timestamp": "2025-01-28T10:30:00Z",
    "entity_id": "user_789",
    "anomaly_type": "spatial",
    "severity": "high",
    "anomaly_score": 0.87,
    "coordinate": {
        "latitude": 40.6892,
        "longitude": -74.0445
    },
    "description": "Location significantly outside normal movement pattern"
}
```

#### Heartbeat
```json
{
    "type": "heartbeat",
    "timestamp": "2025-01-28T10:30:00Z",
    "active_streams": 1,
    "message_queue_size": 0
}
```

---

## 📦 SDK Examples

### Python SDK

```python
import asyncio
import aiohttp
from datacraft_gls import GLSClient

# Initialize client
client = GLSClient(
    base_url="https://api.datacraft.co.ke/api/v1/geographical-location",
    api_key="your_api_key"
)

# Fuzzy search
async def fuzzy_search_example():
    result = await client.fuzzy_search(
        query_text="New York",
        fuzzy_match_type="jaro_winkler",
        confidence_threshold=0.8
    )
    print(f"Found {len(result.matches)} matches")

# Trajectory analysis
async def trajectory_analysis_example():
    trajectory = await client.analyze_trajectory(
        entity_id="vehicle_123",
        time_window_start="2025-01-01T00:00:00Z",
        time_window_end="2025-01-01T23:59:59Z",
        include_patterns=True
    )
    print(f"Detected patterns: {trajectory.detected_patterns}")

# Real-time streaming
async def streaming_example():
    stream = await client.create_stream(
        stream_type="location_updates",
        entity_filters={"entity_type": "vehicle"}
    )
    
    async with client.websocket(stream.websocket_url) as ws:
        async for message in ws:
            print(f"Received: {message}")

# Run examples
asyncio.run(fuzzy_search_example())
```

### JavaScript SDK

```javascript
import { GLSClient } from '@datacraft/gls-sdk';

// Initialize client
const client = new GLSClient({
    baseUrl: 'https://api.datacraft.co.ke/api/v1/geographical-location',
    apiKey: 'your_api_key'
});

// Fuzzy search
async function fuzzySearchExample() {
    const result = await client.fuzzySearch({
        queryText: 'New York',
        fuzzyMatchType: 'jaro_winkler',
        confidenceThreshold: 0.8
    });
    console.log(`Found ${result.matches.length} matches`);
}

// Predictive modeling
async function predictiveModelingExample() {
    const predictions = await client.predictLocations({
        entityId: 'asset_456',
        predictionHorizonHours: 24,
        modelType: 'lstm',
        includeRiskAssessment: true
    });
    console.log(`Risk score: ${predictions.riskAssessment.overallRiskScore}`);
}

// Real-time streaming
async function streamingExample() {
    const stream = await client.createStream({
        streamType: 'location_updates',
        entityFilters: { entityType: 'vehicle' }
    });
    
    const ws = new WebSocket(stream.websocketUrl);
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Location update:', data);
    };
}

// Run examples
fuzzySearchExample();
predictiveModelingExample();
streamingExample();
```

### cURL Examples

#### Fuzzy Search
```bash
curl -X POST "https://api.datacraft.co.ke/api/v1/geographical-location/fuzzy-search" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "New York Cty",
    "fuzzy_match_type": "jaro_winkler",
    "confidence_threshold": 0.7,
    "max_results": 10
  }'
```

#### Trajectory Analysis
```bash
curl -X POST "https://api.datacraft.co.ke/api/v1/geographical-location/trajectory-analysis" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "vehicle_123",
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-01T23:59:59Z",
    "include_patterns": true,
    "h3_resolution": "city"
  }'
```

#### Create Visualization
```bash
curl -X POST "https://api.datacraft.co.ke/api/v1/geographical-location/visualization/create-map" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "map_config": {
      "renderer": "folium",
      "center_coordinate": {
        "latitude": 40.7128,
        "longitude": -74.0060
      },
      "zoom_level": 12
    },
    "data_layers": ["trajectories", "hotspots"],
    "export_format": "png"
  }'
```

---

## 📞 Support & Resources

### Rate Limit Information
- Check response headers for current limits
- Contact support for rate limit increases
- Use batch endpoints for high-volume operations

### Error Troubleshooting
- Check request format against schemas
- Verify authentication token validity
- Review error messages in response body
- Check service status at `/health` endpoint

### Support Channels
- **Technical Support**: support@datacraft.co.ke
- **API Documentation**: docs.datacraft.co.ke/gls
- **Status Page**: status.datacraft.co.ke
- **GitHub Issues**: github.com/datacraft/apg-gls/issues

---

*© 2025 Datacraft. All rights reserved. This API reference is part of the APG Platform documentation.*