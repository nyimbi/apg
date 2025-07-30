# Geographical Location Services - User Guide

**Version:** 2.0.0  
**Author:** Nyimbi Odero <nyimbi@gmail.com>  
**Company:** Datacraft  
**Website:** www.datacraft.co.ke  

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [H3 Spatial Indexing Guide](#h3-spatial-indexing-guide)
4. [Fuzzy Location Matching](#fuzzy-location-matching)
5. [Trajectory Analysis](#trajectory-analysis)
6. [Hotspot Detection](#hotspot-detection)
7. [Predictive Modeling](#predictive-modeling)
8. [Anomaly Detection](#anomaly-detection)
9. [Map Visualization](#map-visualization)
10. [Real-Time Streaming](#real-time-streaming)
11. [Advanced Analytics](#advanced-analytics)
12. [Best Practices](#best-practices)
13. [Common Use Cases](#common-use-cases)
14. [Troubleshooting](#troubleshooting)

---

## 🚀 Getting Started

### Prerequisites

Before using the Geographical Location Services, ensure you have:

- **API Access**: Valid JWT token with appropriate scopes
- **Basic Understanding**: Familiarity with REST APIs and JSON
- **Coordinates**: Understanding of latitude/longitude coordinates
- **Time Zones**: Data should be in UTC format

### First Steps

1. **Test API Access**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
   https://api.datacraft.co.ke/api/v1/geographical-location/health
   ```

2. **Check Capabilities**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
   https://api.datacraft.co.ke/api/v1/geographical-location/capabilities
   ```

3. **Simple Geocoding Test**
   ```bash
   curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
   -H "Content-Type: application/json" \
   -d '{"address": {"street": "350 5th Ave", "city": "New York", "state": "NY", "country": "US"}}' \
   https://api.datacraft.co.ke/api/v1/geographical-location/geocode
   ```

---

## 🧠 Core Concepts

### Coordinate System

All coordinates use the **WGS84** datum with decimal degrees:

```json
{
    "latitude": 40.7128,    // Range: -90 to +90
    "longitude": -74.0060,  // Range: -180 to +180
    "altitude": 10.5        // Optional, meters above sea level
}
```

### Entity Types

The system supports various entity types for tracking and analysis:

- **`person`**: Individual people, employees, customers
- **`vehicle`**: Cars, trucks, delivery vehicles, fleet assets
- **`asset`**: Equipment, containers, valuable items
- **`device`**: IoT sensors, mobile devices, tracking beacons

### Time Windows

Most analytical functions require time windows in ISO 8601 format:

```json
{
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-07T23:59:59Z"
}
```

**Important**: Always use UTC timezone (Z suffix) for consistency.

### Administrative Hierarchies

Geographic data is organized in administrative levels:

- **Country**: Sovereign nations (ISO 3166-1)
- **Admin1**: States, provinces, regions
- **Admin2**: Counties, districts, prefectures  
- **Locality**: Cities, towns, villages

---

## 🔷 H3 Spatial Indexing Guide

### What is H3?

H3 is a hierarchical spatial indexing system that divides the Earth's surface into hexagonal cells at multiple resolution levels. Each location gets encoded into H3 indices at all resolution levels.

### Resolution Levels

| Level | Name | Edge Length | Use Case |
|-------|------|-------------|----------|
| 0 | Continent | ~4,250 km | Global analysis |
| 1 | Country | ~607 km | National planning |
| 2 | State | ~86.7 km | Regional analysis |
| 3 | Metro | ~12.4 km | Metropolitan areas |
| 4 | City | ~1.77 km | Urban planning |
| 5 | District | ~253 m | Neighborhood analysis |
| 6 | Neighborhood | ~36.2 m | Block-level analysis |
| 7 | Block | ~5.17 m | Building precision |
| 8 | Building | ~0.74 m | Room-level tracking |
| 9 | Room | ~0.11 m | High precision |
| 10 | Ultra-precise | ~0.015 m | Centimeter accuracy |

### Understanding H3 Indices

Every coordinate automatically gets H3 indices at all levels:

```json
{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "h3_indices": {
        "0": "8047fffffffffff",
        "4": "84186b9fdbfffff",  // City level (primary)
        "6": "86186b9fdbfffff",  // Neighborhood level
        "8": "88186b9fdbffffff"  // Building level
    },
    "primary_h3_index": "84186b9fdbfffff"  // City level (resolution 4)
}
```

### Choosing the Right Resolution

**For Analysis Type:**
- **Movement Patterns**: Use resolution 4-6 (city to neighborhood)
- **Hotspot Detection**: Use resolution 5-7 (district to block)
- **Territory Management**: Use resolution 3-5 (metro to district)
- **Asset Tracking**: Use resolution 6-8 (neighborhood to building)
- **Indoor Positioning**: Use resolution 8-10 (building to ultra-precise)

**For Time Periods:**
- **Real-time tracking**: Higher resolution (6-8)
- **Daily patterns**: Medium resolution (4-6)
- **Weekly/monthly trends**: Lower resolution (2-4)

### Working with H3 in Queries

```json
{
    "entity_id": "vehicle_123",
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-07T23:59:59Z",
    "h3_resolution": "city"  // Will use resolution 4
}
```

---

## 🔍 Fuzzy Location Matching

### Overview

Fuzzy location matching helps find geographic locations even when the input text contains errors, typos, or variations in naming.

### Matching Algorithms

#### Levenshtein Distance
- **Best for**: Character-level errors, typos
- **Example**: "New Yrok" → "New York" (score: 0.89)

#### Jaro-Winkler Similarity  
- **Best for**: Prefix matching, similar spellings
- **Example**: "NYC" → "New York City" (score: 0.75)

#### Soundex
- **Best for**: Phonetic similarities
- **Example**: "Filadelfia" → "Philadelphia" (score: 0.85)

#### Metaphone
- **Best for**: Advanced phonetic matching
- **Example**: "Nyu York" → "New York" (score: 0.92)

#### FuzzyWuzzy
- **Best for**: Complex string variations
- **Example**: "St. Louis Missouri" → "Saint Louis, MO" (score: 0.88)

### Practical Usage

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

### Confidence Thresholds

- **0.9-1.0**: Excellent matches, very high confidence
- **0.8-0.9**: Good matches, high confidence  
- **0.7-0.8**: Acceptable matches, medium confidence
- **0.6-0.7**: Possible matches, low confidence
- **<0.6**: Poor matches, not recommended

### Tips for Better Results

1. **Use appropriate algorithms**:
   - Typos: Levenshtein
   - Abbreviations: Jaro-Winkler
   - Pronunciation: Soundex/Metaphone

2. **Set reasonable thresholds**:
   - Critical applications: ≥0.8
   - General use: ≥0.7
   - Exploratory: ≥0.6

3. **Filter by administrative level**:
   ```json
   {"admin_level": "city"}  // Only search cities
   ```

4. **Use country filters for performance**:
   ```json
   {"country_filter": "US"}  // Only search US locations
   ```

---

## 📈 Trajectory Analysis

### Overview

Trajectory analysis examines the movement patterns of entities over time, detecting patterns, anomalies, and behavioral insights.

### Key Components

#### Movement Patterns

**Linear Movement**
- Consistent direction with minimal deviation
- Example: Highway driving, direct flights
- Characteristics: Low bearing variance, consistent speed

**Circular Movement** 
- Returning to starting point or near-circular paths
- Example: Patrol routes, delivery rounds
- Characteristics: High directional variance, enclosed path

**Periodic Movement**
- Regular time-based patterns
- Example: Daily commuting, weekly routines
- Characteristics: Temporal regularity, location repetition

**Random Walk**
- No discernible pattern, high unpredictability
- Example: Tourist exploration, emergency response
- Characteristics: High spatial and temporal variance

**Commuting Pattern**
- Regular movement between fixed locations
- Example: Home-work-home patterns
- Characteristics: Two primary locations, time regularity

#### Dwell Points

Locations where entities spend significant time:

```json
{
    "coordinate": {"latitude": 40.7589, "longitude": -73.9851},
    "arrival_time": "2025-01-01T09:00:00Z",
    "departure_time": "2025-01-01T17:30:00Z",
    "duration_minutes": 510,
    "visit_frequency": 0.8,
    "poi_type": "office"
}
```

### Practical Usage

```json
{
    "entity_id": "employee_456",
    "time_window_start": "2025-01-01T00:00:00Z", 
    "time_window_end": "2025-01-07T23:59:59Z",
    "include_patterns": true,
    "include_anomalies": true,
    "h3_resolution": "district"
}
```

### Interpreting Results

**Pattern Confidence Scores:**
- **0.9-1.0**: Very strong pattern, highly predictable
- **0.7-0.9**: Clear pattern, good predictability
- **0.5-0.7**: Moderate pattern, some predictability
- **<0.5**: Weak pattern, low predictability

**Anomaly Scores:**
- **0.8-1.0**: Highly anomalous, needs investigation
- **0.6-0.8**: Moderately anomalous, worth noting
- **0.4-0.6**: Slightly unusual, normal variation
- **<0.4**: Normal behavior

### Use Cases

1. **Employee Monitoring**: Detect unusual work patterns
2. **Fleet Management**: Optimize vehicle routes
3. **Security**: Identify suspicious movement patterns
4. **Urban Planning**: Understand traffic flows
5. **Healthcare**: Monitor patient mobility patterns

---

## 🔥 Hotspot Detection

### Overview

Hotspot detection identifies areas with unusually high concentrations of entities or activities using statistical clustering algorithms.

### Clustering Algorithms

#### DBSCAN (Density-Based Spatial Clustering)
- **Best for**: Irregular shaped clusters, noise filtering
- **Parameters**: min_cluster_size, distance threshold
- **Output**: Dense clusters with noise points identified

#### K-Means
- **Best for**: Spherical clusters, known number of clusters
- **Parameters**: k_value (number of clusters)
- **Output**: Circular clusters with centroids

#### Grid-Based Clustering
- **Best for**: Regular patterns, performance optimization
- **Parameters**: grid_size (H3 resolution)
- **Output**: Grid cells with density counts

#### Hierarchical Clustering
- **Best for**: Multi-level cluster analysis
- **Parameters**: linkage method, distance threshold
- **Output**: Dendrograms with nested clusters

#### OPTICS (Ordering Points To Identify Clustering Structure)
- **Best for**: Variable density clusters
- **Parameters**: min_samples, max_epsilon
- **Output**: Reachability plots with cluster ordering

### Statistical Significance

Hotspots are validated using statistical tests:

**Z-Score Analysis:**
- Measures how many standard deviations above the mean
- **Z > 2.58**: Significant at 99% confidence (p < 0.01)
- **Z > 1.96**: Significant at 95% confidence (p < 0.05)
- **Z > 1.65**: Significant at 90% confidence (p < 0.10)

**P-Value Testing:**
- Probability that the observed pattern occurred by chance
- **p < 0.01**: Highly significant
- **p < 0.05**: Significant  
- **p < 0.10**: Marginally significant

### Practical Usage

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

### Interpreting Results

**Intensity Scores:**
- **0.8-1.0**: Very intense hotspot, high activity concentration
- **0.6-0.8**: Moderate hotspot, notable activity increase
- **0.4-0.6**: Mild hotspot, slight activity increase
- **<0.4**: Weak hotspot, minimal concentration

**Entity Counts:**
- Number of entities contributing to the hotspot
- Higher counts = more reliable hotspot identification
- Consider minimum thresholds based on your use case

### Use Cases

1. **Crime Analysis**: Identify crime hotspots for patrol allocation
2. **Disease Surveillance**: Detect disease outbreak clusters
3. **Traffic Management**: Find traffic congestion points
4. **Retail Analytics**: Identify high foot-traffic areas
5. **Emergency Response**: Locate incident concentration areas

---

## 🔮 Predictive Modeling

### Overview

Predictive modeling uses machine learning to forecast future entity locations based on historical movement patterns.

### Model Types

#### LSTM (Long Short-Term Memory)
- **Best for**: Sequential data with long-term dependencies
- **Strengths**: Handles complex temporal patterns
- **Use cases**: Regular commuting, seasonal patterns

#### ARIMA (Auto-Regressive Integrated Moving Average)
- **Best for**: Time series with trends and seasonality
- **Strengths**: Statistical robustness, interpretability
- **Use cases**: Predictable periodic movement

#### Random Forest
- **Best for**: Non-linear patterns with multiple features
- **Strengths**: Feature importance, robustness to outliers
- **Use cases**: Complex multi-factor movement decisions

#### Ensemble Models
- **Best for**: Maximum accuracy by combining models
- **Strengths**: Reduces individual model weaknesses
- **Use cases**: Critical applications requiring high accuracy

### Prediction Horizons

**Short-term (1-6 hours):**
- High accuracy (typically 85-95%)
- Good for operational decisions
- Real-time route optimization

**Medium-term (6-24 hours):**
- Good accuracy (typically 70-85%)
- Useful for planning and scheduling
- Resource allocation

**Long-term (1-7 days):**
- Moderate accuracy (typically 60-75%)
- Strategic planning applications
- Trend analysis

### Practical Usage

```json
{
    "entity_id": "delivery_truck_789",
    "prediction_horizon_hours": 8,
    "model_type": "lstm",
    "confidence_intervals": true,
    "include_risk_assessment": true
}
```

### Understanding Predictions

**Confidence Intervals:**
- **Upper/Lower Bounds**: Range of likely positions
- **Confidence Decay**: Accuracy decreases over time
- **Uncertainty Quantification**: Statistical confidence in predictions

**Risk Assessment:**
- **Conflict Probability**: Likelihood of issues occurring
- **Risk Factors**: Contributing risk elements
- **Mitigation Strategies**: Recommended actions

### Feature Importance

Models consider various factors:

1. **Historical Locations (45%)**: Past movement patterns
2. **Time of Day (25%)**: Temporal patterns
3. **Day of Week (20%)**: Weekly routines  
4. **Weather (10%)**: Environmental conditions

### Use Cases

1. **Fleet Management**: Predict vehicle locations for dispatch
2. **Supply Chain**: Forecast shipment arrivals
3. **Security**: Anticipate personnel movements
4. **Emergency Response**: Predict resource needs
5. **Urban Planning**: Model traffic flow evolution

---

## ⚠️ Anomaly Detection

### Overview

Anomaly detection identifies unusual patterns in movement and spatial events that deviate from normal behavior.

### Anomaly Types

#### Spatial Anomalies
- **Definition**: Locations significantly outside normal areas
- **Detection**: Distance from usual locations
- **Example**: Employee visiting competitor's office

#### Temporal Anomalies  
- **Definition**: Activities at unusual times
- **Detection**: Deviation from time patterns
- **Example**: Accessing building at 3 AM

#### Behavioral Anomalies
- **Definition**: Movement patterns unlike normal behavior
- **Detection**: Speed, direction, route deviations
- **Example**: Taking unusual route to known destination

#### Statistical Anomalies
- **Definition**: Data points outside statistical norms
- **Detection**: Z-score, percentile analysis
- **Example**: Speed exceeding 99th percentile

### Severity Levels

- **Critical (0.9-1.0)**: Immediate attention required
- **High (0.7-0.9)**: Investigation recommended
- **Medium (0.5-0.7)**: Monitor closely
- **Low (0.3-0.5)**: Note but not concerning

### Sensitivity Settings

**High Sensitivity (0.95-0.99):**
- Detects subtle anomalies
- May have false positives
- Use for security-critical applications

**Medium Sensitivity (0.85-0.95):**
- Balanced detection
- Good for general monitoring
- Recommended for most use cases

**Low Sensitivity (0.70-0.85):**
- Only major anomalies
- Fewer false positives
- Use for noisy environments

### Practical Usage

```json
{
    "entity_id": "employee_123",
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-07T23:59:59Z",
    "anomaly_types": ["spatial", "temporal", "behavioral"],
    "sensitivity": 0.90
}
```

### Response Actions

**Immediate Actions (Critical/High):**
1. Generate alerts
2. Notify security personnel  
3. Initiate investigation protocols
4. Consider automatic response systems

**Monitoring Actions (Medium/Low):**
1. Log for analysis
2. Increase monitoring frequency
3. Review patterns over time
4. Update baseline models

### Use Cases

1. **Fraud Detection**: Unusual transaction locations
2. **Security Monitoring**: Suspicious movement patterns
3. **Fleet Management**: Vehicle route deviations
4. **Healthcare**: Patient mobility anomalies
5. **Asset Protection**: Unexpected equipment movement

---

## 🗺️ Map Visualization

### Overview

Create interactive and static maps with multiple data layers using different rendering engines.

### Rendering Engines

#### Folium (Interactive Web Maps)
- **Best for**: Web applications, interactive exploration
- **Output**: HTML with JavaScript (Leaflet.js)
- **Features**: Zoom, pan, layer controls, popups
- **Export**: HTML, PNG (via screenshot)

#### Matplotlib (Static Images)
- **Best for**: Reports, presentations, publications
- **Output**: PNG, JPEG, SVG, PDF
- **Features**: High-quality graphics, scientific plotting
- **Customization**: Full control over styling

#### Plotly (Interactive Visualizations)
- **Best for**: Dashboard applications, 3D visualization
- **Output**: HTML, PNG, SVG, PDF
- **Features**: 3D plots, animations, statistical charts
- **Interactivity**: Zoom, rotate, hover effects

### Map Configuration

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
    }
}
```

### Tile Providers

- **OpenStreetMap**: Free, detailed street maps
- **CartoDB**: Clean, minimal design options
- **Stamen**: Artistic, stylized map designs
- **Custom**: Your own map tiles

### Data Layers

**Available Layers:**
- `trajectories`: Movement paths and routes
- `hotspots`: Detected activity concentrations  
- `geofences`: Boundary areas and zones
- `entities`: Current entity positions
- `predictions`: Forecasted locations
- `anomalies`: Detected unusual patterns

### Practical Usage

```json
{
    "map_config": {
        "renderer": "folium",
        "center_coordinate": {"latitude": 40.7128, "longitude": -74.0060},
        "zoom_level": 10
    },
    "data_layers": ["trajectories", "hotspots", "geofences"],
    "time_range": {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-07T23:59:59Z"
    },
    "export_format": "png"
}
```

### Styling Options

**Color Schemes:**
- `default`: Standard colors
- `heat`: Red-orange-yellow for intensity
- `cool`: Blue-green-cyan for calm data
- `custom`: User-defined color palette

**Themes:**
- `light`: Light background, dark features
- `dark`: Dark background, light features  
- `satellite`: Satellite imagery background
- `terrain`: Topographic/terrain view

### Use Cases

1. **Dashboard Integration**: Real-time monitoring displays
2. **Report Generation**: Static maps for documents
3. **Analysis Presentation**: Interactive exploration tools
4. **Mobile Applications**: Embedded map components
5. **Public Displays**: Large screen visualizations

---

## 📡 Real-Time Streaming

### Overview

Stream live location data and events through WebSocket connections with configurable data retention and filtering.

### Stream Types

#### Location Updates
- Real-time entity position changes
- Includes coordinates, speed, bearing
- Configurable update frequency

#### Geofence Events
- Entry, exit, and dwell notifications
- Rule-based event generation
- Custom metadata inclusion

#### Analytics Updates
- Live hotspot detection results
- Streaming anomaly alerts
- Updated predictions and forecasts

### Stream Configuration

```json
{
    "stream_type": "location_updates",
    "entity_filters": {
        "entity_type": "vehicle",
        "territory_id": "sales_north"
    },
    "geographic_bounds": {
        "boundary_type": "circle",
        "center_point": {"latitude": 40.7128, "longitude": -74.0060},
        "radius_meters": 5000
    },
    "ttl_seconds": 3600
}
```

### Data Retention (TTL)

**TTL Settings:**
- **60-300 seconds**: Real-time applications, high frequency updates
- **300-1800 seconds**: Dashboard applications, moderate updates  
- **1800-3600 seconds**: Monitoring applications, low frequency
- **3600-86400 seconds**: Historical analysis, data archival

### WebSocket Messages

**Location Update:**
```json
{
    "type": "location_update",
    "timestamp": "2025-01-28T10:30:00Z",
    "entity_id": "vehicle_123",
    "coordinate": {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "primary_h3_index": "84186b9fdbfffff"
    },
    "metadata": {
        "speed_kmh": 45.5,
        "bearing_degrees": 135.2
    }
}
```

**Geofence Event:**
```json
{
    "type": "geofence_event", 
    "timestamp": "2025-01-28T10:30:00Z",
    "entity_id": "vehicle_123",
    "geofence_id": "sales_territory_north",
    "event_type": "enter",
    "coordinate": {"latitude": 40.7128, "longitude": -74.0060}
}
```

### Connection Management

**Authentication:**
```javascript
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your_jwt_token'  
    }));
};
```

**Heartbeat:**
- Automatic heartbeat every 30 seconds
- Connection health monitoring
- Automatic reconnection on failure

### Performance Considerations

**Connection Limits:**
- 100 concurrent connections per API key
- 10,000 messages per minute per connection
- Compression enabled for efficiency

**Filtering:**
- Server-side filtering reduces bandwidth
- Geographic bounds limit data volume
- Entity filters focus on relevant updates

### Use Cases

1. **Fleet Tracking**: Real-time vehicle monitoring
2. **Asset Security**: Immediate theft/movement alerts
3. **Emergency Response**: Live incident coordination
4. **Logistics**: Shipment tracking and updates
5. **Smart Cities**: Traffic and crowd monitoring

---

## 📊 Advanced Analytics

### Overview

Comprehensive spatial analytics using multiple algorithms to understand patterns, relationships, and trends in location data.

### Analysis Types

#### Spatial Clustering
Groups nearby locations to identify concentrated areas:

**DBSCAN Results:**
- Clusters with irregular shapes
- Noise point identification
- Density-based grouping

**K-Means Results:**
- Spherical cluster shapes
- Fixed number of clusters
- Centroid identification

**Grid-Based Results:**
- Regular grid cell analysis
- H3 hexagonal aggregation
- Efficient for large datasets

#### Heat Mapping
Creates smooth density surfaces from point data:

**Interpolation Methods:**
- **IDW (Inverse Distance Weighting)**: Nearby points have more influence
- **Kriging**: Statistical interpolation with error estimation
- **Spline**: Smooth surface fitting through points

**Applications:**
- Population density visualization
- Activity intensity mapping
- Risk assessment surfaces

#### Hotspot Detection
Statistical identification of significant clusters:

**Getis-Ord Gi* Statistics:**
- Local spatial statistics
- Z-score calculation for significance
- Hot and cold spot identification

**Local Moran's I:**
- Spatial autocorrelation analysis
- High-High and Low-Low cluster detection
- Spatial outlier identification

#### Spatial Autocorrelation
Measures spatial relationships and dependencies:

**Global Moran's I:**
- Overall spatial pattern assessment
- Values: -1 (dispersed) to +1 (clustered)
- Statistical significance testing

**Spatial Lag/Error Analysis:**
- Spatial dependency modeling
- Neighbor influence quantification
- Regression analysis enhancement

### Practical Usage

```json
{
    "analysis_types": [
        "spatial_clustering",
        "heat_mapping", 
        "hotspot_detection",
        "spatial_autocorrelation"
    ],
    "time_window_start": "2025-01-01T00:00:00Z",
    "time_window_end": "2025-01-07T23:59:59Z",
    "spatial_bounds": {
        "boundary_type": "rectangle",
        "coordinates": [
            {"latitude": 40.7000, "longitude": -74.0200},
            {"latitude": 40.7500, "longitude": -73.9800}
        ]
    },
    "h3_resolution": "district",
    "include_visualization": true
}
```

### Interpreting Results

**Spatial Autocorrelation Values:**
- **Moran's I = 0**: Random spatial pattern
- **Moran's I > 0**: Clustered pattern (similar values near each other)  
- **Moran's I < 0**: Dispersed pattern (dissimilar values near each other)

**Statistical Significance:**
- **p < 0.001**: Highly significant pattern
- **p < 0.01**: Very significant pattern
- **p < 0.05**: Significant pattern
- **p ≥ 0.05**: Not statistically significant

**Clustering Quality Metrics:**
- **Silhouette Score**: -1 to 1, higher is better
- **Within-Cluster Variance**: Lower is better for compact clusters
- **Between-Cluster Distance**: Higher is better for separated clusters

### Use Cases

1. **Urban Planning**: Analyze development patterns
2. **Epidemiology**: Study disease spread patterns
3. **Retail**: Optimize store locations
4. **Transportation**: Understand traffic flows
5. **Environmental**: Monitor pollution patterns

---

## 💡 Best Practices

### Performance Optimization

#### Choosing Appropriate Resolutions

**For Real-Time Applications:**
- Use H3 resolution 4-6 (city to neighborhood)
- Limit time windows to recent data (last 24-48 hours)
- Consider caching frequently accessed patterns

**For Historical Analysis:**
- Use H3 resolution 2-4 (state to city) for large datasets
- Process data in chunks for large time windows
- Use appropriate clustering algorithms for data size

#### Batch Processing

**Large Dataset Handling:**
- Use batch geocoding for >100 addresses
- Process trajectory analysis in weekly chunks
- Aggregate hotspot detection over monthly periods

**API Rate Limits:**
- Respect rate limits (see API reference)
- Implement exponential backoff for retries
- Use WebSocket streaming for real-time needs

### Data Quality

#### Coordinate Validation

Always validate coordinates before processing:
```python
def validate_coordinate(lat, lng):
    return -90 <= lat <= 90 and -180 <= lng <= 180
```

#### Time Zone Consistency

- Always use UTC timestamps
- Convert local times to UTC before API calls
- Account for daylight saving time changes

#### Data Completeness

- Handle missing coordinates gracefully
- Interpolate gaps in trajectory data carefully
- Document data quality issues in metadata

### Security Considerations

#### Data Privacy

**Location Data Protection:**
- Implement data retention policies
- Use appropriate H3 resolution for privacy (lower resolution = more private)
- Consider data anonymization for non-essential uses

**Access Control:**
- Use principle of least privilege
- Implement role-based access controls
- Log all data access for audit trails

#### API Security

**Token Management:**
- Rotate API tokens regularly
- Use environment variables for tokens
- Implement token refresh mechanisms

### Error Handling

#### Graceful Degradation

```python
try:
    trajectory = await analyze_trajectory(request)
except APIException as e:
    if e.status_code == 429:  # Rate limited
        await asyncio.sleep(60)  # Wait and retry
        trajectory = await analyze_trajectory(request)
    else:
        # Use cached data or simplified analysis
        trajectory = get_cached_trajectory(request.entity_id)
```

#### Monitoring and Alerting

- Monitor API response times
- Set up alerts for high error rates
- Track data quality metrics over time

---

## 🎯 Common Use Cases

### 1. Fleet Management System

**Objective**: Track delivery vehicles and optimize routes

**Implementation:**
```python
# Real-time vehicle tracking
stream = await gls_client.create_stream(
    stream_type="location_updates",
    entity_filters={"entity_type": "vehicle", "fleet_id": "delivery"}
)

# Analyze delivery patterns
trajectory = await gls_client.analyze_trajectory(
    entity_id="truck_001",
    time_window_start=datetime.now() - timedelta(days=7),
    time_window_end=datetime.now(),
    h3_resolution="district"
)

# Detect traffic hotspots
hotspots = await gls_client.detect_hotspots(
    entity_type="vehicle",
    clustering_algorithm="dbscan",
    time_window_start=datetime.now() - timedelta(hours=2),
    h3_resolution="neighborhood"
)

# Optimize routes
optimized_route = await gls_client.optimize_route(
    waypoints=delivery_locations,
    optimization_objective="fastest_time",
    constraints={"avoid_hotspots": [h.center_coordinate for h in hotspots]}
)
```

**Benefits:**
- Reduce fuel costs by 15-20%
- Improve delivery time predictability
- Automatic rerouting around traffic
- Driver behavior monitoring

### 2. Employee Safety Monitoring

**Objective**: Ensure field worker safety and compliance

**Implementation:**
```python
# Monitor employee locations
for employee in field_workers:
    # Check for anomalies
    anomalies = await gls_client.detect_anomalies(
        entity_id=employee.id,
        time_window_start=datetime.now() - timedelta(hours=8),
        time_window_end=datetime.now(),
        anomaly_types=["spatial", "temporal"],
        sensitivity=0.90
    )
    
    # Emergency alert for high anomalies
    critical_anomalies = [a for a in anomalies if a.severity_level == "critical"]
    if critical_anomalies:
        await emergency_service.alert_supervisors(employee, critical_anomalies)
    
    # Check geofence compliance
    events = await gls_client.process_location_update(
        entity_id=employee.id,
        current_location=employee.current_location
    )
    
    exit_events = [e for e in events if e.event_type == "exit" and e.geofence.is_restricted]
    if exit_events:
        await compliance_service.log_violation(employee, exit_events)
```

**Benefits:**
- Improved worker safety response times
- Automated compliance monitoring
- Reduced insurance costs
- Better emergency coordination

### 3. Retail Site Selection

**Objective**: Identify optimal locations for new stores

**Implementation:**
```python
# Analyze foot traffic patterns
hotspots = await gls_client.detect_hotspots(
    entity_type="person",
    time_window_start=datetime.now() - timedelta(days=30),
    time_window_end=datetime.now(),
    clustering_algorithm="dbscan",
    h3_resolution="neighborhood"
)

# Competitor analysis using fuzzy search
competitors = []
for area in candidate_areas:
    nearby_stores = await gls_client.fuzzy_search(
        query_text=f"coffee shop near {area.name}",
        fuzzy_match_type="jaro_winkler",
        confidence_threshold=0.8,
        max_results=20
    )
    competitors.extend(nearby_stores.matches)

# Predictive modeling for customer flow
for candidate_location in candidate_sites:
    predicted_traffic = await gls_client.predict_locations(
        entity_id="foot_traffic_model",
        prediction_horizon_hours=168,  # 1 week
        model_type="ensemble"
    )
    
    # Score the location
    score = calculate_site_score(
        foot_traffic=len([h for h in hotspots if is_nearby(h, candidate_location)]),
        competition_density=len([c for c in competitors if is_nearby(c, candidate_location)]),
        predicted_growth=predicted_traffic.confidence_intervals.average_confidence
    )
    
    candidate_location.viability_score = score
```

**Benefits:**
- Data-driven site selection
- Reduced investment risk
- Competitive advantage analysis
- Revenue optimization

### 4. Smart City Traffic Management

**Objective**: Optimize traffic flow and reduce congestion

**Implementation:**
```python
# Real-time traffic monitoring
traffic_stream = await gls_client.create_stream(
    stream_type="location_updates",
    entity_filters={"entity_type": "vehicle"},
    geographic_bounds=city_boundaries
)

# Detect congestion hotspots
congestion_hotspots = await gls_client.detect_hotspots(
    entity_type="vehicle",
    clustering_algorithm="grid_based",
    time_window_start=datetime.now() - timedelta(hours=1),
    h3_resolution="block",
    min_cluster_size=20
)

# Predict traffic evolution
for hotspot in congestion_hotspots:
    traffic_prediction = await gls_client.predict_locations(
        entity_id=f"traffic_hotspot_{hotspot.hotspot_id}",
        prediction_horizon_hours=2,
        model_type="lstm"
    )
    
    if traffic_prediction.risk_assessment.overall_risk_score > 0.7:
        # Adjust traffic signals
        await traffic_control.optimize_signal_timing(
            intersection=hotspot.center_coordinate,
            predicted_flow=traffic_prediction.predictions
        )
        
        # Suggest alternate routes
        await navigation_service.broadcast_alternate_routes(
            affected_area=hotspot.center_coordinate,
            alternate_routes=await find_alternate_routes(hotspot)
        )
```

**Benefits:**
- Reduced average commute times
- Lower emissions from idling
- Improved emergency vehicle access
- Better public transportation efficiency

### 5. Healthcare Asset Tracking

**Objective**: Track medical equipment and ensure availability

**Implementation:**
```python
# Track critical medical equipment
critical_assets = ["ventilator", "defibrillator", "wheelchair", "iv_pump"]

for asset_type in critical_assets:
    # Get current asset locations
    assets = await asset_service.get_assets_by_type(asset_type)
    
    # Analyze movement patterns
    for asset in assets:
        trajectory = await gls_client.analyze_trajectory(
            entity_id=asset.id,
            time_window_start=datetime.now() - timedelta(days=7),
            time_window_end=datetime.now(),
            h3_resolution="building"
        )
        
        # Detect anomalies (potential theft or misplacement)
        anomalies = await gls_client.detect_anomalies(
            entity_id=asset.id,
            time_window_start=datetime.now() - timedelta(hours=24),
            anomaly_types=["spatial", "temporal"],
            sensitivity=0.95
        )
        
        if anomalies:
            await security_service.investigate_asset(asset, anomalies)
    
    # Optimize asset distribution
    demand_hotspots = await gls_client.detect_hotspots(
        entity_type="patient",
        time_window_start=datetime.now() - timedelta(days=30),
        h3_resolution="room"
    )
    
    # Recommend asset repositioning
    recommendations = optimize_asset_placement(assets, demand_hotspots)
    await facility_management.implement_recommendations(recommendations)
```

**Benefits:**
- Reduced equipment search times
- Improved patient care efficiency
- Lower equipment loss rates
- Better resource allocation

---

## 🔧 Troubleshooting

### Common Issues

#### 1. API Authentication Errors

**Problem**: 401 Unauthorized responses

**Solutions:**
```python
# Check token validity
response = requests.get(
    "https://api.datacraft.co.ke/api/v1/geographical-location/health",
    headers={"Authorization": f"Bearer {token}"}
)
if response.status_code == 401:
    token = refresh_token()
```

**Checklist:**
- ✅ Token not expired
- ✅ Correct Bearer format
- ✅ Required scopes included
- ✅ API key valid for environment

#### 2. H3 Index Issues

**Problem**: Inconsistent H3 indices or resolution errors

**Solutions:**
```python
# Validate coordinates before H3 generation
def safe_h3_generation(lat, lng):
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError(f"Invalid coordinates: {lat}, {lng}")
    
    return generate_h3_indices(lat, lng)

# Use appropriate resolution for use case
resolution_map = {
    "global_analysis": 2,
    "city_planning": 4,
    "neighborhood_study": 6,
    "building_tracking": 8
}
```

**Checklist:**
- ✅ Coordinates within valid ranges
- ✅ Appropriate resolution for analysis
- ✅ Consistent resolution across queries
- ✅ Handle edge cases (poles, date line)

#### 3. Fuzzy Search Not Finding Results

**Problem**: No matches for location queries

**Solutions:**
```python
# Try different algorithms and thresholds
algorithms = ["levenshtein", "jaro_winkler", "soundex", "metaphone"]
thresholds = [0.9, 0.8, 0.7, 0.6]

for algorithm in algorithms:
    for threshold in thresholds:
        matches = await fuzzy_search(
            query_text=query,
            fuzzy_match_type=algorithm,
            confidence_threshold=threshold
        )
        if matches:
            break
    if matches:
        break
```

**Checklist:**
- ✅ Try multiple algorithms
- ✅ Lower confidence threshold
- ✅ Remove country/admin filters
- ✅ Check for typos in query text

#### 4. Trajectory Analysis Empty Results

**Problem**: No patterns or segments detected

**Solutions:**
```python
# Check data availability and quality
trajectory_points = await get_entity_locations(
    entity_id=entity_id,
    time_window_start=start_time,
    time_window_end=end_time
)

if len(trajectory_points) < 10:
    print("Insufficient data points for analysis")
    return None

# Adjust time window
if not trajectory_points:
    # Expand time window
    expanded_start = start_time - timedelta(days=7)
    trajectory_points = await get_entity_locations(
        entity_id, expanded_start, end_time
    )
```

**Checklist:**
- ✅ Sufficient data points (>10)
- ✅ Appropriate time window
- ✅ Entity actually moved during period
- ✅ Correct entity ID

#### 5. WebSocket Connection Issues

**Problem**: Connections dropping or not receiving messages

**Solutions:**
```python
# Implement connection resilience
class ResilientWebSocket:
    def __init__(self, url, max_retries=5):
        self.url = url
        self.max_retries = max_retries
        self.retry_count = 0
    
    async def connect_with_retry(self):
        while self.retry_count < self.max_retries:
            try:
                self.ws = await websockets.connect(self.url)
                self.retry_count = 0  # Reset on successful connection
                return self.ws
            except Exception as e:
                self.retry_count += 1
                wait_time = 2 ** self.retry_count
                await asyncio.sleep(wait_time)
        
        raise Exception("Max retries exceeded")
```

**Checklist:**
- ✅ Stable internet connection
- ✅ Proper authentication message sent
- ✅ Heartbeat responses sent
- ✅ Implement reconnection logic

### Performance Issues

#### 1. Slow API Responses

**Optimization Strategies:**

```python
# Use appropriate time windows
# Instead of:
time_window = timedelta(days=365)  # Too large

# Use:
time_window = timedelta(days=30)   # More reasonable

# Batch requests when possible
addresses_batch = chunk_list(all_addresses, 100)
for batch in addresses_batch:
    results = await batch_geocode(batch)
```

#### 2. High Memory Usage

**Memory Optimization:**

```python
# Process data in chunks
def process_large_dataset(entities, chunk_size=1000):
    for i in range(0, len(entities), chunk_size):
        chunk = entities[i:i + chunk_size]
        yield process_chunk(chunk)

# Use generators for large results
def get_trajectories_generator(entities):
    for entity in entities:
        yield analyze_trajectory(entity)
```

### Data Quality Issues

#### 1. Inconsistent Coordinates

**Validation and Cleaning:**

```python
def clean_coordinates(coordinates):
    cleaned = []
    for coord in coordinates:
        # Remove obvious errors
        if not (-90 <= coord.latitude <= 90):
            continue
        if not (-180 <= coord.longitude <= 180):
            continue
        
        # Remove duplicate consecutive points
        if cleaned and distance(cleaned[-1], coord) < 1:  # <1 meter
            continue
            
        cleaned.append(coord)
    
    return cleaned
```

#### 2. Time Zone Issues

**Standardization:**

```python
from datetime import timezone

def normalize_timestamp(timestamp_str, source_timezone=None):
    """Convert all timestamps to UTC"""
    dt = datetime.fromisoformat(timestamp_str)
    
    if dt.tzinfo is None:
        if source_timezone:
            dt = dt.replace(tzinfo=source_timezone)
        else:
            dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.astimezone(timezone.utc)
```

### Getting Help

#### 1. Check Service Status

```python
health_response = await gls_client.health_check()
if health_response["status"] != "healthy":
    print("Service issues detected:", health_response)
```

#### 2. Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("gls_client")
```

#### 3. Contact Support

- **Email**: support@datacraft.co.ke
- **Documentation**: docs.datacraft.co.ke/gls
- **Status Page**: status.datacraft.co.ke
- **GitHub Issues**: github.com/datacraft/apg-gls/issues

Include in support requests:
- Request/Response examples
- Error messages and stack traces
- Entity IDs and time windows
- Expected vs actual behavior

---

*© 2025 Datacraft. All rights reserved. This user guide is part of the APG Platform documentation.*